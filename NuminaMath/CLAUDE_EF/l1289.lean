import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_a_l1289_128991

-- Define the function f
noncomputable def f (x : ℝ) : ℝ :=
  if x ≥ 0 then x^3 else -x^3

-- State the theorem
theorem range_of_a (a : ℝ) :
  (f (3*a - 1) ≥ 8 * f a) ↔ (a ≤ 1/5 ∨ a ≥ 1) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_a_l1289_128991


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_value_proof_l1289_128999

theorem tan_value_proof (φ : ℝ) 
  (h1 : Real.cos (3 * Real.pi / 2 - φ) = 3 / 5)
  (h2 : |φ| < Real.pi / 2) : 
  Real.tan φ = -3 / 4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_value_proof_l1289_128999


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_distance_curve_to_line_l1289_128951

-- Define the line l in Cartesian coordinates
def line_l (x y : ℝ) : Prop := x - y = 1

-- Define the curve C
noncomputable def curve_C (α : ℝ) : ℝ × ℝ := (1 + Real.sqrt 3 * Real.cos α, Real.sin α)

-- State the theorem
theorem max_distance_curve_to_line :
  ∃ (d : ℝ), d = Real.sqrt 3 ∧
  (∀ (α : ℝ), ∃ (p : ℝ × ℝ), p = curve_C α ∧
    (∀ (x y : ℝ), line_l x y → abs (x - p.1 + y - p.2) / Real.sqrt 2 ≤ d) ∧
  (∃ (α : ℝ), ∃ (x y : ℝ), line_l x y ∧
    abs (x - (curve_C α).1 + y - (curve_C α).2) / Real.sqrt 2 = d)) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_distance_curve_to_line_l1289_128951


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_point_properties_l1289_128986

/-- Parabola with vertex (-2, 3) and focus (-2, 4) -/
def Parabola : Set (ℝ × ℝ) :=
  {p | (p.1 + 2)^2 = 4 * (p.2 - 3)}

/-- The focus of the parabola -/
def F : ℝ × ℝ := (-2, 4)

/-- The point P on the parabola -/
def P : ℝ × ℝ := (48, 628)

/-- Distance between two points in ℝ² -/
noncomputable def distance (p q : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2)

theorem parabola_point_properties :
  P ∈ Parabola ∧ 
  P.1 > 0 ∧ P.2 > 0 ∧
  distance P F = 51 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_point_properties_l1289_128986


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_safe_paths_count_l1289_128911

/-- Represents a point on the grid --/
structure Point where
  x : Nat
  y : Nat

/-- The starting point (Jack's house) --/
def start : Point := ⟨0, 0⟩

/-- The ending point (Jill's house) --/
def finish : Point := ⟨4, 3⟩

/-- The dangerous intersection to avoid --/
def danger : Point := ⟨2, 2⟩

/-- Calculates the number of paths between two points --/
def numPaths (p q : Point) : Nat :=
  Nat.choose ((q.x - p.x) + (q.y - p.y)) (q.x - p.x)

/-- Calculates the number of paths passing through the dangerous point --/
def numDangerPaths : Nat :=
  numPaths start danger * numPaths danger finish

/-- The total number of paths from start to end --/
def totalPaths : Nat := numPaths start finish

/-- The number of safe paths avoiding the dangerous intersection --/
def safePaths : Nat := totalPaths - numDangerPaths

theorem safe_paths_count : safePaths = 17 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_safe_paths_count_l1289_128911


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_b_formula_l1289_128980

def b : ℕ → ℚ
  | 0 => 2  -- Added case for 0
  | 1 => 2
  | 2 => 4/9
  | n+3 => (2 * b (n+1) * b (n+2)) / (3 * b (n+1) - 2 * b (n+2))

theorem b_formula (n : ℕ) (h : n ≥ 1) : b n = 8 / (4 * n - 2) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_b_formula_l1289_128980


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_odd_sum_square_inductive_l1289_128963

/-- Represents a sequence of natural numbers -/
def NaturalSequence : Type := ℕ → ℕ

/-- Defines the sequence 1, 1+3, 1+3+5, ... -/
def oddSum : NaturalSequence :=
  λ n => (2 * n - 1) * n

/-- Defines the sequence 1², 2², 3², ... -/
def squareSequence : NaturalSequence :=
  λ n => n * n

/-- Definition of inductive reasoning -/
def is_inductive_reasoning (hypothesis : NaturalSequence → Prop) : Prop :=
  ∃ (k : ℕ), ∀ (seq : NaturalSequence), (∀ (n : ℕ), n ≤ k → hypothesis seq) → 
  hypothesis seq

/-- The statement to be proved -/
theorem odd_sum_square_inductive : 
  is_inductive_reasoning (λ seq => seq = oddSum) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_odd_sum_square_inductive_l1289_128963


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_bus_driver_max_regular_hours_verify_compensation_l1289_128998

/-- Represents the bus driver's compensation structure and work details --/
structure BusDriverCompensation where
  regularRate : ℚ
  overtimeRateMultiplier : ℚ
  totalCompensation : ℚ
  totalHoursWorked : ℚ

/-- Calculates the maximum regular hours given the compensation structure --/
noncomputable def maxRegularHours (comp : BusDriverCompensation) : ℚ :=
  let overtimeRate := comp.regularRate * (1 + comp.overtimeRateMultiplier)
  (comp.totalCompensation - overtimeRate * comp.totalHoursWorked) / (comp.regularRate - overtimeRate)

/-- Theorem stating that the maximum regular hours for the given scenario is 40 --/
theorem bus_driver_max_regular_hours :
  let comp := BusDriverCompensation.mk 16 (3/4) 1200 60
  maxRegularHours comp = 40 := by
  sorry

/-- Verifies that the calculated maximum regular hours satisfies the total compensation --/
theorem verify_compensation (comp : BusDriverCompensation) 
  (h : maxRegularHours comp = 40) :
  let regularHours := maxRegularHours comp
  let overtimeHours := comp.totalHoursWorked - regularHours
  let overtimeRate := comp.regularRate * (1 + comp.overtimeRateMultiplier)
  comp.regularRate * regularHours + overtimeRate * overtimeHours = comp.totalCompensation := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_bus_driver_max_regular_hours_verify_compensation_l1289_128998


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_b_and_d_know_grades_l1289_128942

-- Define the grade type
inductive Grade
| Excellent
| Good
deriving BEq, Repr

-- Define the student type
inductive Student
| A
| B
| C
| D
deriving BEq, Repr

-- Function to represent what grades a student sees
def sees : Student → List (Student × Grade) 
| Student.A => [(Student.B, Grade.Excellent), (Student.C, Grade.Good)]  -- Example assignment
| Student.B => [(Student.C, Grade.Good)]  -- Example assignment
| Student.D => [(Student.A, Grade.Excellent)]  -- Example assignment
| Student.C => []

-- Define the condition that there are 2 excellent and 2 good grades
axiom two_excellent_two_good : 
  (List.count Grade.Excellent (List.map Prod.snd (sees Student.A) ++ 
   List.map Prod.snd (sees Student.B) ++ 
   List.map Prod.snd (sees Student.D))) = 2

-- A doesn't know their own grade
axiom a_doesnt_know : ¬(∃ g : Grade, ∀ g' : Grade, g ≠ g' → 
  List.count g' (List.map Prod.snd (sees Student.A)) = 2)

-- Theorem: B and D can know their own grades
theorem b_and_d_know_grades : 
  (∃ g : Grade, g = Grade.Excellent ∨ g = Grade.Good) ∧
  (∃ g : Grade, g = Grade.Excellent ∨ g = Grade.Good) := by
  sorry

#check b_and_d_know_grades

end NUMINAMATH_CALUDE_ERRORFEEDBACK_b_and_d_know_grades_l1289_128942


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parallelogram_diagonal_theorem_sum_mnp_is_81_l1289_128933

/-- Represents a parallelogram ABCD with projections P, Q, R, S -/
structure Parallelogram where
  -- Area of the parallelogram
  area : ℝ
  -- Length of PQ (projection of A and C onto BD)
  pq_length : ℝ
  -- Length of RS (projection of B and D onto AC)
  rs_length : ℝ

/-- The square of the length of the longer diagonal of the parallelogram -/
noncomputable def diagonal_squared (p : Parallelogram) : ℝ := 32 + 8 * Real.sqrt 41

/-- Theorem stating the relationship between the parallelogram's properties and its diagonal length -/
theorem parallelogram_diagonal_theorem (p : Parallelogram) 
  (h_area : p.area = 15)
  (h_pq : p.pq_length = 6)
  (h_rs : p.rs_length = 8) : 
  diagonal_squared p = 32 + 8 * Real.sqrt 41 := by
  sorry

/-- The sum of m, n, and p in the expression m + n√p -/
def sum_mnp : ℕ := 32 + 8 + 41

theorem sum_mnp_is_81 : sum_mnp = 81 := by
  rfl

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parallelogram_diagonal_theorem_sum_mnp_is_81_l1289_128933


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_optimal_diagonals_l1289_128943

/-- Regular polygon with 400 sides -/
structure RegularPolygon400 where
  vertices : Fin 400 → ℝ × ℝ
  center : ℝ × ℝ
  is_regular : ∀ (i j : Fin 400), dist (vertices i) (vertices j) = dist (vertices 0) (vertices 1)
  is_centered : ∀ (i : Fin 400), dist (vertices i) center = dist (vertices 0) center

/-- Distance function between two points -/
def myDist (p q : ℝ × ℝ) : ℝ := sorry

/-- Deviation function for a given diagonal -/
def deviation (p : RegularPolygon400) (k : Fin 400) : ℝ :=
  |myDist (p.vertices 0) (p.vertices k) - myDist (p.vertices 200) (p.vertices k)| - myDist (p.vertices 0) p.center

/-- Theorem stating that the specified diagonals minimize the deviation -/
theorem optimal_diagonals (p : RegularPolygon400) :
  ∀ (k : Fin 400), k ≠ 54 ∧ k ≠ 146 ∧ k ≠ 254 ∧ k ≠ 346 →
    (min (deviation p 54) (min (deviation p 146) (min (deviation p 254) (deviation p 346))))
    ≤ deviation p k :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_optimal_diagonals_l1289_128943


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_number_of_functions_A_to_B_l1289_128946

def A : Finset ℕ := {1, 2, 3}
def B : Finset ℕ := {4, 5}

theorem number_of_functions_A_to_B : Fintype.card (A → B) = 6 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_number_of_functions_A_to_B_l1289_128946


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trajectory_of_Q_l1289_128992

-- Define the circle
def my_circle (x y : ℝ) : Prop := x^2 + y^2 = 1

-- Define the fixed point M
def M : ℝ × ℝ := (-1, 2)

-- Define the relation between P, M, and Q
def relation (P Q : ℝ × ℝ) : Prop :=
  P.1 - M.1 = 2 * (M.1 - Q.1) ∧ P.2 - M.2 = 2 * (M.2 - Q.2)

theorem trajectory_of_Q :
  ∀ (P Q : ℝ × ℝ),
    my_circle P.1 P.2 →
    relation P Q →
    (Q.1 + 3/2)^2 + (Q.2 - 3)^2 = 1/4 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_trajectory_of_Q_l1289_128992


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_property_l1289_128966

theorem function_property (f : ℕ → ℕ) 
  (h1 : ∀ n, f (f n) + f n = 2 * n + 3)
  (h2 : f 0 = 1)
  (h3 : f 2015 = 2016) :
  ∃ n, f n = 2015 ∧ n = 2015 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_property_l1289_128966


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_russian_doll_purchase_l1289_128965

/-- Given Daniel's savings in USD and the conditions of the Russian doll problem,
    calculate the number of dolls he can buy. -/
theorem russian_doll_purchase
  (savings_usd : ℝ)
  (original_price : ℝ)
  (discount_rate : ℝ)
  (exchange_rate : ℝ)
  (h1 : original_price = 280)
  (h2 : discount_rate = 0.2)
  (h3 : exchange_rate = 70)
  : ℕ :=
  let discounted_price := original_price * (1 - discount_rate)
  let dolls_buyable := (savings_usd * exchange_rate) / discounted_price
  (Int.floor dolls_buyable).toNat

#check russian_doll_purchase

end NUMINAMATH_CALUDE_ERRORFEEDBACK_russian_doll_purchase_l1289_128965


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_properties_l1289_128915

-- Define the triangle ABC
structure Triangle where
  A : ℝ × ℝ
  altitude1 : ℝ → ℝ → Prop
  altitude2 : ℝ → ℝ → Prop

-- Define the given triangle
noncomputable def givenTriangle : Triangle where
  A := (1, 2)
  altitude1 := λ x y => x + y = 0
  altitude2 := λ x y => 2*x - 3*y + 1 = 0

-- Define the orthocenter
noncomputable def orthocenter (t : Triangle) : ℝ × ℝ := (-1/5, 1/5)

-- Define the equation of the altitude from B to AC
def altitudeBtoAC (t : Triangle) : ℝ → ℝ → Prop :=
  λ x y => 9*x - 11*y + 13 = 0

-- Theorem statement
theorem triangle_properties :
  (orthocenter givenTriangle = (-1/5, 1/5)) ∧
  (altitudeBtoAC givenTriangle = λ x y => 9*x - 11*y + 13 = 0) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_properties_l1289_128915


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_decreasing_on_neg_reals_l1289_128938

-- Define the logarithm with base 10
noncomputable def log10 (x : ℝ) : ℝ := Real.log x / Real.log 10

-- Define the function f(x) = lg(x^2)
noncomputable def f (x : ℝ) : ℝ := log10 (x^2)

-- Theorem statement
theorem f_decreasing_on_neg_reals :
  StrictAntiOn f (Set.Iio 0) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_decreasing_on_neg_reals_l1289_128938


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_2016th_number_l1289_128940

/-- Definition of the sequence -/
def mySequence : ℕ → ℕ
| n => 
  let group := (n - 1) / 6 + 1
  let position := (n - 1) % 6
  match position with
  | 0 => group
  | 1 => group + 1
  | 2 => group + 2
  | 3 => group + 2
  | 4 => group + 1
  | _ => group

theorem sequence_2016th_number : mySequence 2016 = 336 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_2016th_number_l1289_128940


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_inequalities_l1289_128961

-- Define a structure for a triangle
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ
  h_a : ℝ
  h_b : ℝ
  m_a : ℝ
  m_b : ℝ
  -- Add necessary conditions
  pos_sides : a > 0 ∧ b > 0 ∧ c > 0
  angle_sum : A + B + C = π
  pos_angles : A > 0 ∧ B > 0 ∧ C > 0

-- Theorem statement
theorem triangle_inequalities (t : Triangle) :
  (t.a ≥ t.b) ↔ (Real.sin t.A ≥ Real.sin t.B) ∧ (t.m_a ≤ t.m_b) ∧ (t.h_a ≤ t.h_b) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_inequalities_l1289_128961


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_role_assignment_count_l1289_128968

def number_of_assignments (men women male_roles female_roles : ℕ) : ℕ :=
  (Nat.factorial men / Nat.factorial (men - male_roles)) * 
  (Nat.factorial women / Nat.factorial (women - female_roles)) * 
  (men + women - male_roles - female_roles)

theorem role_assignment_count : 
  number_of_assignments 4 5 3 2 = 1920 := by
  rw [number_of_assignments]
  norm_num
  rfl

#eval number_of_assignments 4 5 3 2

end NUMINAMATH_CALUDE_ERRORFEEDBACK_role_assignment_count_l1289_128968


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_equal_areas_l1289_128981

theorem intersection_equal_areas : ∃ (c : ℝ),
  (∃ (a : ℝ), a > 0 ∧ c = 2*a - 3*a^3) ∧
  (∫ (x : ℝ) in Set.Icc 0 a, (2*x - 3*x^3 - c) = 0) ∧
  c = 4/9 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_equal_areas_l1289_128981


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_power_equality_l1289_128904

theorem power_equality (x : ℝ) (h : (27 : ℝ)^3 = (9 : ℝ)^x) : (3 : ℝ)^(-x) = 1 / Real.sqrt 19683 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_power_equality_l1289_128904


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_properties_l1289_128921

noncomputable def sequence_a (n : ℕ) : ℝ := 2 * n + 1

noncomputable def sum_S (n : ℕ) : ℝ := n^2 + 2*n

noncomputable def sequence_b (n : ℕ) : ℝ := (sequence_a n - 5) / 2^n

noncomputable def sum_b (n : ℕ) : ℝ := (1/9) * (4 - (3*n + 1) / 4^(n-1))

theorem sequence_properties :
  ∀ n : ℕ, n ≥ 1 →
  (∀ k : ℕ, k ≥ 1 → sequence_a k > 0) ∧
  (sum_S n)^2 - (n^2 + 2*n - 1)*(sum_S n) - (n^2 + 2*n) = 0 →
  (∀ k : ℕ, k ≥ 1 → sequence_a k = 2*k + 1) ∧
  (sum_b n = (Finset.sum (Finset.range n) (fun i => sequence_b (2*(i+1))))) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_properties_l1289_128921


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_count_satisfying_condition_l1289_128907

def b (n : ℕ) : ℕ → ℕ
  | 0 => 0  -- base case, not used in the problem
  | m + 1 => if (b n m) % 3 = 0 then (b n m) / 3 else 2 * (b n m) + 2

def satisfies_condition (n : ℕ) : Bool :=
  n > 0 && n ≤ 1500 && n < b n 1 && n < b n 2 && n < b n 3

theorem count_satisfying_condition :
  (Finset.filter (fun n => satisfies_condition n) (Finset.range 1501)).card = 1000 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_count_satisfying_condition_l1289_128907


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_binary_95_l1289_128993

/-- Represents a natural number in binary (base 2) form -/
def toBinary (n : ℕ) : List Bool :=
  if n = 0 then [false]
  else
    let rec go (m : ℕ) : List Bool :=
      if m = 0 then []
      else (m % 2 = 1) :: go (m / 2)
    go n

/-- Converts a list of booleans representing a binary number to a natural number -/
def fromBinary (l : List Bool) : ℕ :=
  l.foldl (fun acc b => 2 * acc + if b then 1 else 0) 0

theorem binary_95 :
  toBinary 95 = [true, true, true, true, true, true, false, true] :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_binary_95_l1289_128993


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_monotone_increasing_l1289_128906

-- Define the function f(x) = log₁/₃(x² - 4)
noncomputable def f (x : ℝ) : ℝ := Real.log (x^2 - 4) / Real.log (1/3)

-- State the theorem
theorem f_monotone_increasing :
  StrictMonoOn f (Set.Iio (-2)) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_monotone_increasing_l1289_128906


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_count_divisors_not_div_by_3_eq_6_l1289_128902

/-- The number of positive divisors of 180 that are not divisible by 3 -/
def count_divisors_not_div_by_3 : ℕ :=
  (Finset.filter (fun d => d ∣ 180 ∧ ¬(3 ∣ d)) (Nat.divisors 180)).card

/-- Theorem stating that the number of positive divisors of 180 
    that are not divisible by 3 is equal to 6 -/
theorem count_divisors_not_div_by_3_eq_6 : 
  count_divisors_not_div_by_3 = 6 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_count_divisors_not_div_by_3_eq_6_l1289_128902


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_goods_train_speed_calculation_l1289_128931

/-- The speed of the goods train in kmph -/
noncomputable def goods_train_speed : ℝ := 20

/-- The speed of the man's train in kmph -/
noncomputable def mans_train_speed : ℝ := 120

/-- The time taken for the goods train to pass the man in seconds -/
noncomputable def passing_time : ℝ := 9

/-- The length of the goods train in meters -/
noncomputable def goods_train_length : ℝ := 350

/-- Conversion factor from kmph to m/s -/
noncomputable def kmph_to_ms : ℝ := 5 / 18

theorem goods_train_speed_calculation :
  (mans_train_speed + goods_train_speed) * kmph_to_ms * passing_time = goods_train_length :=
by
  -- Convert all values to ℝ explicitly
  have h1 : (120 : ℝ) + (20 : ℝ) = (140 : ℝ) := by norm_num
  have h2 : (5 : ℝ) / (18 : ℝ) = (5/18 : ℝ) := by norm_num
  have h3 : (140 : ℝ) * (5/18 : ℝ) * (9 : ℝ) = (350 : ℝ) := by norm_num
  
  -- Rewrite using the definitions
  rw [mans_train_speed, goods_train_speed, kmph_to_ms, passing_time, goods_train_length]
  
  -- Apply the equalities we proved
  rw [h1, h2]
  exact h3

end NUMINAMATH_CALUDE_ERRORFEEDBACK_goods_train_speed_calculation_l1289_128931


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_angle_solution_l1289_128976

-- Define the triangle ABC
structure Triangle where
  A : ℝ
  B : ℝ
  C : ℝ

-- Define the conditions
def isArithmeticMean (x y z : ℝ) : Prop := z = (x + y) / 2

theorem triangle_angle_solution (ABC : Triangle) 
  (h1 : isArithmeticMean ABC.A ABC.B ABC.C)
  (h2 : isArithmeticMean ABC.A ABC.B ((Real.sqrt 3) + 1))
  (h3 : ABC.C = 2 * Real.sqrt 2) :
  ABC.A = 75 ∨ ABC.A = 45 := by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_angle_solution_l1289_128976


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_dot_product_l1289_128924

/-- Given an ellipse C: x²/4 + y²/3 = 1, with A(-2, 0) as the left vertex, 
    B(0, √3) as the upper vertex, and F(1, 0) as the right focus, 
    the dot product of vectors AB and AF is 6. -/
theorem ellipse_dot_product : 
  let C : ℝ → ℝ → Prop := fun x y ↦ x^2/4 + y^2/3 = 1
  let A : ℝ × ℝ := (-2, 0)
  let B : ℝ × ℝ := (0, Real.sqrt 3)
  let F : ℝ × ℝ := (1, 0)
  let AB : ℝ × ℝ := (B.1 - A.1, B.2 - A.2)
  let AF : ℝ × ℝ := (F.1 - A.1, F.2 - A.2)
  AB.1 * AF.1 + AB.2 * AF.2 = 6 := by
  sorry

#check ellipse_dot_product

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_dot_product_l1289_128924


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ratio_increases_l1289_128952

/-- Represents a circle with parallel chords -/
structure CircleWithChords where
  r : ℝ  -- radius of the circle
  h : ℝ  -- distance from center to chord
  d : ℝ  -- distance between chords

/-- Area of the trapezoid CDFE -/
noncomputable def K (c : CircleWithChords) : ℝ :=
  8 * Real.sqrt (2 * c.r * c.h - c.h^2)

/-- Area of the rectangle ELMF when r = 10 -/
noncomputable def R (c : CircleWithChords) : ℝ :=
  8 * Real.sqrt (100 - (10 - c.h)^2)

/-- Ratio of K to R -/
noncomputable def ratio (c : CircleWithChords) : ℝ :=
  K c / R c

theorem ratio_increases (c : CircleWithChords) (r' : ℝ) :
  c.r < r' → c.d = 4 → ratio { r := r', h := c.h, d := c.d } > ratio c := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_ratio_increases_l1289_128952


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_angle_sine_l1289_128978

-- Define the function f as noncomputable
noncomputable def f (x : ℝ) : ℝ := 5/4 * (Real.cos x)^2 - (Real.sqrt 3)/2 * Real.sin x * Real.cos x - 1/4 * (Real.sin x)^2

-- State the theorem
theorem triangle_angle_sine (A B C : ℝ) : 
  0 < A ∧ 0 < B ∧ 0 < C ∧  -- Acute triangle condition
  A + B + C = Real.pi ∧    -- Sum of angles in a triangle
  Real.cos B = 3/5 ∧       -- Given condition
  f C = -1/4 →             -- Given condition
  Real.sin A = (4 + 3 * Real.sqrt 3) / 10 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_angle_sine_l1289_128978


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_extremum_implies_zero_derivative_zero_derivative_not_sufficient_for_extremum_derivative_zero_necessary_not_sufficient_l1289_128918

open Real Function

-- Define a real-valued function f
variable (f : ℝ → ℝ)
-- Define a real number x₀
variable (x₀ : ℝ)

-- Assume f is differentiable at x₀
variable (hf : DifferentiableAt ℝ f x₀)

-- Define what it means for x₀ to be an extremum point of f
def IsExtremumPoint (f : ℝ → ℝ) (x₀ : ℝ) : Prop :=
  ∃ δ > 0, ∀ x, |x - x₀| < δ → f x ≤ f x₀ ∨ f x ≥ f x₀

-- First part: If x₀ is an extremum point, then f'(x₀) = 0
theorem extremum_implies_zero_derivative 
  (h : IsExtremumPoint f x₀) : 
  deriv f x₀ = 0 :=
sorry

-- Second part: There exists a function where f'(x₀) = 0 but x₀ is not an extremum point
theorem zero_derivative_not_sufficient_for_extremum :
  ∃ f : ℝ → ℝ, DifferentiableAt ℝ f 0 ∧ deriv f 0 = 0 ∧ ¬IsExtremumPoint f 0 :=
sorry

-- Theorem stating that f'(x₀) = 0 is a necessary but not sufficient condition for x₀ to be an extremum point
theorem derivative_zero_necessary_not_sufficient :
  (∀ f : ℝ → ℝ, ∀ x₀ : ℝ, DifferentiableAt ℝ f x₀ → IsExtremumPoint f x₀ → deriv f x₀ = 0) ∧
  (∃ f : ℝ → ℝ, ∃ x₀ : ℝ, DifferentiableAt ℝ f x₀ ∧ deriv f x₀ = 0 ∧ ¬IsExtremumPoint f x₀) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_extremum_implies_zero_derivative_zero_derivative_not_sufficient_for_extremum_derivative_zero_necessary_not_sufficient_l1289_128918


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_overlapping_squares_area_l1289_128916

noncomputable def overlapping_area_of_rotated_squares (side_length : ℝ) (α : ℝ) : ℝ :=
  4 * (2 - Real.sqrt 3)

theorem overlapping_squares_area :
  ∀ (side_length : ℝ) (α : ℝ),
    side_length = 2 →
    α = Real.pi / 3 →
    Real.cos α = 1 / 2 →
    overlapping_area_of_rotated_squares side_length α = 4 * (2 - Real.sqrt 3) :=
by
  intros side_length α h1 h2 h3
  unfold overlapping_area_of_rotated_squares
  rfl


end NUMINAMATH_CALUDE_ERRORFEEDBACK_overlapping_squares_area_l1289_128916


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_equation_hyperbola_equation_l1289_128934

-- Define the ellipse parameters
noncomputable def minor_axis : ℝ := 6
noncomputable def eccentricity : ℝ := 2 * Real.sqrt 2 / 3

-- Define the given ellipse equation
def given_ellipse (x y : ℝ) : Prop := 4 * x^2 + 9 * y^2 = 36

-- Define the point that the hyperbola passes through
noncomputable def point : ℝ × ℝ := (4, Real.sqrt 3)

-- Theorem for the ellipse equation
theorem ellipse_equation :
  ∃ (a b : ℝ), (a > 0 ∧ b > 0) ∧
  ((∀ x y, x^2 / a^2 + y^2 / b^2 = 1 ↔ x^2 / 81 + y^2 / 9 = 1) ∨
   (∀ x y, x^2 / b^2 + y^2 / a^2 = 1 ↔ y^2 / 81 + x^2 / 9 = 1)) ∧
  b = minor_axis / 2 ∧
  eccentricity = Real.sqrt (a^2 - b^2) / a :=
by sorry

-- Theorem for the hyperbola equation
theorem hyperbola_equation :
  ∃ (a b : ℝ), (a > 0 ∧ b > 0) ∧
  (∀ x y, x^2 / a^2 - y^2 / b^2 = 1 ↔ x^2 / 4 - y^2 = 1) ∧
  (∀ x y, given_ellipse x y ↔ x^2 / 9 + y^2 / 4 = 1) ∧
  a^2 + b^2 = 5 ∧
  point.1^2 / a^2 - point.2^2 / b^2 = 1 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_equation_hyperbola_equation_l1289_128934


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rhombus_other_diagonal_l1289_128997

/-- The area of a rhombus given its diagonals -/
noncomputable def rhombusArea (d1 d2 : ℝ) : ℝ := (d1 * d2) / 2

theorem rhombus_other_diagonal 
  (d1 : ℝ) (area : ℝ) (h1 : d1 = 50) (h2 : area = 625) :
  ∃ d2 : ℝ, d2 = 25 ∧ rhombusArea d1 d2 = area := by
  sorry

#check rhombus_other_diagonal

end NUMINAMATH_CALUDE_ERRORFEEDBACK_rhombus_other_diagonal_l1289_128997


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_theorem_l1289_128929

theorem max_value_theorem (a b c : ℝ) 
  (h1 : a > 0) 
  (h2 : b > a) 
  (h3 : ∀ x : ℝ, a * x^2 + b * x + c ≥ 0) : 
  (∀ a b c : ℝ, (b - a) / (a + 2*b + 4*c) ≤ 1/8) ∧ 
  (∃ a b c : ℝ, (b - a) / (a + 2*b + 4*c) = 1/8) := by
  sorry

#check max_value_theorem

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_theorem_l1289_128929


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_stack_height_for_8cm_pipes_l1289_128912

/-- The height of a stack of three identical cylindrical pipes -/
noncomputable def stack_height (pipe_diameter : ℝ) : ℝ :=
  pipe_diameter + pipe_diameter / 2 * Real.sqrt 3

/-- Theorem: The height of a stack of three identical cylindrical pipes
    with diameter 8 cm, arranged in a triangular formation,
    is equal to 8 + 4√3 cm. -/
theorem stack_height_for_8cm_pipes :
  stack_height 8 = 8 + 4 * Real.sqrt 3 := by
  sorry

-- Remove the #eval statement as it's not computable
-- #eval stack_height 8

end NUMINAMATH_CALUDE_ERRORFEEDBACK_stack_height_for_8cm_pipes_l1289_128912


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sphere_contains_points_l1289_128948

theorem sphere_contains_points (cube_side : ℝ) (num_points : ℕ) (sphere_radius : ℝ) :
  cube_side = 15 →
  num_points = 11000 →
  sphere_radius = 1 →
  ∃ (center : ℝ × ℝ × ℝ),
    (∃ (points : Finset (ℝ × ℝ × ℝ)),
      points.card ≥ 6 ∧
      (∀ p ∈ points,
        p.1 ≥ 0 ∧ p.1 ≤ cube_side ∧
        p.2.1 ≥ 0 ∧ p.2.1 ≤ cube_side ∧
        p.2.2 ≥ 0 ∧ p.2.2 ≤ cube_side) ∧
      (∀ p ∈ points,
        (p.1 - center.1)^2 + (p.2.1 - center.2.1)^2 + (p.2.2 - center.2.2)^2 ≤ sphere_radius^2)) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_sphere_contains_points_l1289_128948


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_calculation_proof_system_of_equations_proof_l1289_128996

-- Problem 1
theorem calculation_proof : 
  Real.sqrt 12 + (Real.pi - 203) ^ (0 : ℤ) + (1 / 2) ^ (-1 : ℤ) - 6 * Real.tan (30 * Real.pi / 180) = 3 := by
  sorry

-- Problem 2
theorem system_of_equations_proof :
  ∃ (x y : ℝ), x + 2 * y = 4 ∧ x + 3 * y = 5 ∧ x = 2 ∧ y = 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_calculation_proof_system_of_equations_proof_l1289_128996


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_proof_l1289_128944

-- Define the triangle ABC
structure Triangle where
  A : ℝ
  B : ℝ
  C : ℝ
  a : ℝ
  b : ℝ
  c : ℝ

-- State the theorem
theorem triangle_proof (t : Triangle) 
  (h1 : Real.sin (2 * t.B) = Real.sqrt 2 * Real.sin t.B)
  (h2 : (1/2) * t.a * t.c * Real.sin t.B = 6)
  (h3 : t.a = 4) :
  t.B = π/4 ∧ t.b = Real.sqrt 10 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_proof_l1289_128944


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_A_wins_bound_l1289_128964

/-- Represents the probability of winning a single point -/
def probability_win_point (p : ℝ) : Prop := 0 ≤ p ∧ p ≤ 1/2

/-- Represents the rules of the tennis game -/
def tennis_game_rules : Prop :=
  ∀ (score_A score_B : ℕ),
    (score_A = 4 ∧ score_B ≤ 2) ∨ 
    (score_B = 4 ∧ score_A ≤ 2) ∨
    (score_A ≥ 3 ∧ score_B ≥ 3 ∧ (score_A : ℤ) - (score_B : ℤ) = 2 ∨ (score_B : ℤ) - (score_A : ℤ) = 2)

/-- The probability of player A winning the game -/
noncomputable def probability_A_wins (p : ℝ) : ℝ := sorry

/-- Theorem stating that the probability of player A winning is not greater than 2p^2 -/
theorem probability_A_wins_bound (p : ℝ) 
  (h_p : probability_win_point p) 
  (h_rules : tennis_game_rules) : 
  probability_A_wins p ≤ 2 * p^2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_A_wins_bound_l1289_128964


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_difference_l1289_128917

-- Define the function g
noncomputable def g (n : ℝ) : ℝ :=
  if n < 0 then n^2 + 2 else 3*n - 30

-- State the theorem
theorem solution_difference : 
  let a₁ := -2 * Real.sqrt 2
  let a₂ := 40 / 3
  (∃ (a₁ a₂ : ℝ), g (-3) + g 3 + g a₁ = 0 ∧ g (-3) + g 3 + g a₂ = 0 ∧ 
   a₁ ≠ a₂ ∧ (a₂ - a₁ = (40 + 6 * Real.sqrt 2) / 3 ∨ a₁ - a₂ = (40 + 6 * Real.sqrt 2) / 3)) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_difference_l1289_128917


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rhombus_perimeter_l1289_128914

-- Define the rhombus
def Rhombus (d1 d2 : ℝ) := d1 > 0 ∧ d2 > 0

-- Define the perimeter of the rhombus
noncomputable def perimeterRhombus (d1 d2 : ℝ) : ℝ := 
  4 * Real.sqrt ((d1/2)^2 + (d2/2)^2)

-- Theorem statement
theorem rhombus_perimeter (d1 d2 : ℝ) (h : Rhombus d1 d2) :
  d1 = 24 ∧ d2 = 16 → perimeterRhombus d1 d2 = 16 * Real.sqrt 13 :=
by
  intro h_diagonals
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_rhombus_perimeter_l1289_128914


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_roots_of_composite_polynomial_l1289_128995

/-- Given two polynomials P and Q with real coefficients, where
    P(x) = x^2 + x/2 + b and Q(x) = x^2 + cx + d,
    and P(x)Q(x) = Q(P(x)) for all real x,
    prove that the real roots of P(Q(x)) = 0 are x = (-c ± √(c^2 + 2)) / 2. -/
theorem roots_of_composite_polynomial (b c d : ℝ) :
  let P := fun (x : ℝ) ↦ x^2 + x/2 + b
  let Q := fun (x : ℝ) ↦ x^2 + c*x + d
  (∀ x, P x * Q x = Q (P x)) →
  (∃ x, P (Q x) = 0) ↔ (∃ x, x = (-c + Real.sqrt (c^2 + 2))/2 ∨ x = (-c - Real.sqrt (c^2 + 2))/2) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_roots_of_composite_polynomial_l1289_128995


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_g_l1289_128908

-- Define the function f
noncomputable def f : ℝ → ℝ := sorry

-- Define the domain of f
def domain_f : Set ℝ := Set.Icc (-8) 4

-- Define the function g in terms of f
noncomputable def g (x : ℝ) : ℝ := f (-3 * x)

-- Theorem: The domain of g is [-4/3, 8/3]
theorem domain_of_g :
  {x : ℝ | g x = g x} = Set.Icc (-4/3) (8/3) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_g_l1289_128908


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_power_equation_solution_l1289_128958

theorem power_equation_solution (n : ℝ) : (17 : ℝ)^(4*n) = (1/17 : ℝ)^(n-34) → n = 34/5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_power_equation_solution_l1289_128958


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_meat_needed_for_community_event_l1289_128977

/-- The amount of meat (in pounds) needed for a given number of hamburgers in large batches -/
noncomputable def meat_needed (normal_meat : ℝ) (normal_burgers : ℝ) (large_batch_increase : ℝ) (num_burgers : ℝ) : ℝ :=
  (normal_meat / normal_burgers) * (1 + large_batch_increase) * num_burgers

/-- Proof that 13.2 pounds of meat are needed for 30 hamburgers in a large batch -/
theorem meat_needed_for_community_event : 
  meat_needed 4 10 0.1 30 = 13.2 := by
  -- Unfold the definition of meat_needed
  unfold meat_needed
  -- Perform the calculation
  simp [mul_assoc, mul_comm, mul_div_assoc]
  -- The rest of the proof is omitted
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_meat_needed_for_community_event_l1289_128977


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_powderman_distance_at_explosion_l1289_128945

/-- The time it takes for the explosion to occur -/
def explosion_time : ℝ := 45

/-- The speed of the powderman in yards per second -/
def powderman_speed : ℝ := 5

/-- The speed of sound in feet per second -/
def sound_speed : ℝ := 1080

/-- The distance traveled by the powderman in feet after t seconds -/
def powderman_distance (t : ℝ) : ℝ := powderman_speed * t * 3

/-- The distance traveled by sound in feet after t seconds -/
def sound_distance (t : ℝ) : ℝ := sound_speed * (t - explosion_time)

/-- The theorem stating the distance traveled by the powderman when he hears the explosion -/
theorem powderman_distance_at_explosion : 
  ∃ t : ℝ, t > explosion_time ∧ 
  powderman_distance t = sound_distance t ∧ 
  ⌊powderman_distance t / 3⌋ = 228 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_powderman_distance_at_explosion_l1289_128945


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_wang_water_usage_unique_l1289_128974

/-- Represents the cost function for hot spring water usage -/
noncomputable def hot_spring_cost (x : ℝ) : ℝ :=
  if x ≤ 5 then 8 * x
  else if x ≤ 8 then 12 * x - 20
  else if x ≤ 10 then 16 * x - 52
  else 0

/-- The cost of tap water per ton -/
def tap_water_cost : ℝ := 2

/-- The total water usage of Mr. Wang -/
def total_water_usage : ℝ := 16

/-- The total cost of Mr. Wang's water usage -/
def total_cost : ℝ := 72

/-- Theorem stating that Mr. Wang's water usage is uniquely determined -/
theorem wang_water_usage_unique : 
  ∃! (tap_usage hot_usage : ℝ), 
    tap_usage + hot_usage = total_water_usage ∧
    tap_water_cost * tap_usage + hot_spring_cost hot_usage = total_cost ∧
    tap_usage = 10 ∧ hot_usage = 6 := by
  sorry

#check wang_water_usage_unique

end NUMINAMATH_CALUDE_ERRORFEEDBACK_wang_water_usage_unique_l1289_128974


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l1289_128975

noncomputable def f (x : ℝ) : ℝ := |Real.cos x| + Real.cos (abs x)

theorem f_properties :
  (∀ y, y ∈ Set.range f → 0 ≤ y ∧ y ≤ 2) ∧
  (∀ x, f (-x) = f x) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l1289_128975


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_root_implies_a_greater_than_negative_one_l1289_128900

-- Define the function f as noncomputable
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := Real.log x / Real.log (1/3) + 1/x + a

-- State the theorem
theorem root_implies_a_greater_than_negative_one (a : ℝ) :
  (∃ x : ℝ, x > 1 ∧ f a x = 0) →
  (∀ x y : ℝ, 0 < x ∧ x < y → f a y < f a x) →
  a > -1 :=
by
  -- Proof sketch
  intro h1 h2
  -- Assume the existence of a root and the decreasing property
  have h3 : f a 1 > 0 := by sorry
  -- Show that f(1) > 0
  have h4 : 1 + a > 0 := by sorry
  -- Conclude that a > -1
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_root_implies_a_greater_than_negative_one_l1289_128900


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_profit_maximization_l1289_128939

/-- Profit function given promotional expense t -/
noncomputable def profit (t : ℝ) : ℝ := 27 - 18 / (2 * t + 1) - t

/-- The maximum profit achievable -/
def max_profit : ℝ := 21.5

/-- The optimal promotional expense -/
def optimal_expense : ℝ := 2.5

theorem profit_maximization (t : ℝ) (h : t ≥ 0) :
  profit t ≤ max_profit ∧ 
  profit optimal_expense = max_profit := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_profit_maximization_l1289_128939


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_current_at_12_ohms_l1289_128941

/-- A battery with voltage 48V connected to a resistance produces a current. -/
noncomputable def battery_current (R : ℝ) : ℝ := 48 / R

/-- The theorem states that when the resistance is 12Ω, the current is 4A. -/
theorem current_at_12_ohms :
  battery_current 12 = 4 := by
  -- Unfold the definition of battery_current
  unfold battery_current
  -- Simplify the division
  norm_num


end NUMINAMATH_CALUDE_ERRORFEEDBACK_current_at_12_ohms_l1289_128941


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_of_sight_not_blocked_l1289_128926

noncomputable def circleSet : Set (ℝ × ℝ) :=
  {(x, y) | x^2 + y^2 = 1}

def pointA : ℝ × ℝ := (0, -2)
def pointB (a : ℝ) : ℝ × ℝ := (a, 2)

def lineOfSight (a : ℝ) : Set (ℝ × ℝ) :=
  {(x, y) | ∃ t : ℝ, 0 ≤ t ∧ t ≤ 1 ∧ x = t * a ∧ y = -2 + 4 * t}

def notBlocked (a : ℝ) : Prop :=
  ∀ p : ℝ × ℝ, p ∈ lineOfSight a → p ∉ circleSet

theorem line_of_sight_not_blocked (a : ℝ) :
  notBlocked a ↔ (a < -4 * Real.sqrt 3 / 3 ∨ a > 4 * Real.sqrt 3 / 3) := by
  sorry

#check line_of_sight_not_blocked

end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_of_sight_not_blocked_l1289_128926


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_root_in_interval_l1289_128949

-- Define the function f(x) = ln x + 2x - 5
noncomputable def f (x : ℝ) : ℝ := Real.log x + 2 * x - 5

-- State the theorem
theorem root_in_interval :
  (f 2 < 0) → (f 3 > 0) → ∃ x, x ∈ Set.Ioo 2 3 ∧ f x = 0 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_root_in_interval_l1289_128949


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cartesian_to_polar_conversion_l1289_128932

/-- Given a point P(1, √3) in Cartesian coordinates, prove that its polar coordinates are (2, π/3) -/
theorem cartesian_to_polar_conversion (x y : ℝ) (h1 : x = 1) (h2 : y = Real.sqrt 3) :
  let r : ℝ := Real.sqrt (x^2 + y^2)
  let θ : ℝ := Real.arctan (y / x)
  r = 2 ∧ θ = π / 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cartesian_to_polar_conversion_l1289_128932


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_S_is_54_l1289_128990

/-- A rectangle with sides of length 12 and 18 -/
structure Rectangle where
  length : ℝ
  width : ℝ
  is_rectangle : length = 18 ∧ width = 12

/-- The region S within the rectangle -/
noncomputable def S (r : Rectangle) : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.1 ≥ r.width / 4 ∧ p.1 ≤ 3 * r.width / 4 ∧
               p.2 ≥ r.length / 4 ∧ p.2 ≤ 3 * r.length / 4}

/-- The area of region S in the rectangle -/
noncomputable def area_S (r : Rectangle) : ℝ :=
  (r.width / 2) * (r.length / 2)

theorem area_S_is_54 (r : Rectangle) : area_S r = 54 := by
  have h1 : r.length = 18 := (r.is_rectangle).left
  have h2 : r.width = 12 := (r.is_rectangle).right
  unfold area_S
  rw [h1, h2]
  norm_num
  
#check area_S_is_54

end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_S_is_54_l1289_128990


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_point_and_sum_l1289_128909

-- Define the functions h and j
def h : ℝ → ℝ := sorry
def j : ℝ → ℝ := sorry

-- State the given conditions
axiom h3_eq_j3 : h 3 = j 3
axiom h5_eq_j5 : h 5 = j 5
axiom h7_eq_j7 : h 7 = j 7
axiom h9_eq_j9 : h 9 = j 9

axiom h3_eq_3 : h 3 = 3
axiom h5_eq_5 : h 5 = 5
axiom h7_eq_7 : h 7 = 7
axiom h9_eq_9 : h 9 = 9

-- Theorem statement
theorem intersection_point_and_sum :
  ∃ x y : ℝ, h (3 * x) = 3 * j x ∧ x = 3 ∧ y = 9 ∧ x + y = 12 :=
by
  -- Proof goes here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_point_and_sum_l1289_128909


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_infinitely_many_perfect_squares_in_a_l1289_128971

/-- The floor function -/
noncomputable def floor (x : ℝ) : ℤ :=
  Int.floor x

/-- The sequence a_n -/
noncomputable def a (n : ℕ) : ℤ :=
  floor (n * Real.sqrt 2)

/-- Statement: There are infinitely many perfect squares in the sequence a_n -/
theorem infinitely_many_perfect_squares_in_a :
  ∀ N : ℕ, ∃ n : ℕ, n > N ∧ ∃ m : ℕ, a n = m^2 := by
  sorry

#check infinitely_many_perfect_squares_in_a

end NUMINAMATH_CALUDE_ERRORFEEDBACK_infinitely_many_perfect_squares_in_a_l1289_128971


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_man_speed_calculation_l1289_128950

/-- Calculates the speed of a man given the parameters of a train passing him. -/
noncomputable def manSpeed (trainLength : ℝ) (trainSpeed : ℝ) (passingTime : ℝ) : ℝ :=
  let trainSpeedMps := trainSpeed * 1000 / 3600
  let relativeSpeed := trainLength / passingTime
  let manSpeedMps := trainSpeedMps - relativeSpeed
  manSpeedMps * 3600 / 1000

/-- Theorem stating that given the specified conditions, the man's speed is approximately 7.9952 kmph. -/
theorem man_speed_calculation (trainLength trainSpeed passingTime : ℝ)
    (hLength : trainLength = 300)
    (hSpeed : trainSpeed = 68)
    (hTime : passingTime = 17.998560115190784) :
    abs (manSpeed trainLength trainSpeed passingTime - 7.9952) < 0.0001 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_man_speed_calculation_l1289_128950


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_between_curves_l1289_128973

/-- The minimum distance between points on y = (1/2)e^x and y = ln(2x) -/
theorem min_distance_between_curves : ∃ (d : ℝ), 
  d = Real.sqrt 2 * (1 - Real.log 2) ∧
  ∀ (x₁ x₂ : ℝ), x₁ > 0 → x₂ > 0 →
    d ≤ Real.sqrt ((x₁ - x₂)^2 + ((1/2) * Real.exp x₁ - Real.log (2 * x₂))^2) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_between_curves_l1289_128973


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_evaluation_and_special_case_l1289_128988

noncomputable def f (x : ℝ) := Real.sqrt 2 * Real.cos (x - Real.pi/12)

theorem f_evaluation_and_special_case :
  (f (-Real.pi/6) = 1) ∧
  (∀ θ : ℝ, θ ∈ Set.Ioo (3*Real.pi/2) (2*Real.pi) → Real.cos θ = 3/5 → f (2*θ + Real.pi/3) = 17/25) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_evaluation_and_special_case_l1289_128988


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_increasing_iff_b_in_range_l1289_128989

noncomputable def f (b : ℝ) (x : ℝ) : ℝ :=
  if x > 1 then (2*b - 1)/x + b + 3
  else -x^2 + (2-b)*x

theorem f_increasing_iff_b_in_range (b : ℝ) :
  (∀ x y : ℝ, x < y → f b x < f b y) ↔ b ∈ Set.Icc (-1/4 : ℝ) 0 := by
  sorry

#check f_increasing_iff_b_in_range

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_increasing_iff_b_in_range_l1289_128989


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_num_solutions_is_15_l1289_128972

/-- The system of equations --/
def system (x y z w : ℝ) : Prop :=
  x = z + w + z*w*x ∧
  y = w + x + w*x*y ∧
  z = x + y + x*y*z ∧
  w = y + z + y*z*w

/-- The set of solutions to the system --/
def solution_set : Set (ℝ × ℝ × ℝ × ℝ) :=
  {p : ℝ × ℝ × ℝ × ℝ | 
    let (x, y, z, w) := p
    system x y z w ∧ 
    x ≠ 0 ∧ y ≠ 0 ∧ z ≠ 0 ∧ w ≠ 0 ∧
    x*y*z*w = 1
  }

/-- The number of solutions to the system --/
noncomputable def num_solutions : ℕ := solution_set.ncard

/-- The main theorem --/
theorem num_solutions_is_15 : num_solutions = 15 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_num_solutions_is_15_l1289_128972


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_quadrilateral_prism_volume_l1289_128905

/-- A quadrilateral prism with specific properties -/
structure QuadrilateralPrism where
  -- The base is a rhombus with a 60° angle
  base_is_rhombus : Bool
  base_angle : ℝ
  base_angle_is_60 : base_angle = Real.pi / 3
  -- Each side face forms a 60° angle with the base
  side_face_angle : ℝ
  side_face_angle_is_60 : side_face_angle = Real.pi / 3
  -- Point M inside the prism
  M : Point
  -- M is equidistant from base and each side face
  M_equidistant : ℝ
  M_equidistant_is_1 : M_equidistant = 1

/-- The volume of the quadrilateral prism -/
noncomputable def prism_volume (p : QuadrilateralPrism) : ℝ :=
  8 * Real.sqrt 3

/-- Theorem stating that the volume of the quadrilateral prism is 8√3 -/
theorem quadrilateral_prism_volume (p : QuadrilateralPrism) :
  prism_volume p = 8 * Real.sqrt 3 := by
  -- Unfold the definition of prism_volume
  unfold prism_volume
  -- The equality holds by definition
  rfl


end NUMINAMATH_CALUDE_ERRORFEEDBACK_quadrilateral_prism_volume_l1289_128905


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_prime_factor_of_2323_l1289_128928

theorem largest_prime_factor_of_2323 :
  (Nat.factors 2323).maximum? = some 101 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_prime_factor_of_2323_l1289_128928


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_percent_not_won_approaches_40_l1289_128947

/-- Represents the statistics of a sports team --/
structure TeamStats where
  winLossRatio : Rat
  ties : Nat

/-- Calculates the percentage of games not won --/
noncomputable def percentNotWon (stats : TeamStats) : ℝ :=
  let x := stats.winLossRatio.num
  let y := stats.winLossRatio.den
  let t := stats.ties
  (y + t) / (x + y + t) * 100

/-- Theorem stating that the percentage of games not won approaches 40% --/
theorem percent_not_won_approaches_40 (stats : TeamStats) 
    (h1 : stats.winLossRatio = 3/2) 
    (h2 : stats.ties = 5) : 
  ∀ ε > 0, ∃ N : ℕ, ∀ n ≥ N, |percentNotWon stats - 40| < ε :=
by
  sorry

#check percent_not_won_approaches_40

end NUMINAMATH_CALUDE_ERRORFEEDBACK_percent_not_won_approaches_40_l1289_128947


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_of_3_equals_9_l1289_128959

noncomputable def power_function (α : ℝ) (x : ℝ) : ℝ := x ^ α

theorem f_of_3_equals_9 
  (a : ℝ) (h_a : a > 0)
  (log_a : ℝ → ℝ) 
  (h_log : ∀ (x : ℝ), x > 0 → log_a x = Real.log x / Real.log a)
  (y : ℝ → ℝ) 
  (h_y : ∀ (x : ℝ), x > 3/2 → y x = log_a (2*x - 3) + 4)
  (x_0 y_0 : ℝ) 
  (h_point : y_0 = log_a (2*x_0 - 3) + 4)
  (α : ℝ) (h_α : α > 0)
  (f : ℝ → ℝ) 
  (h_f : ∀ (x : ℝ), x > 0 → f x = power_function α x)
  (h_A : y_0 = f x_0) :
  f 3 = 9 := by
  sorry

#check f_of_3_equals_9

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_of_3_equals_9_l1289_128959


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_sum_l1289_128919

-- Define the parabola
def parabola (x y : ℝ) : Prop := y^2 = 4*x

-- Define the line
def line (x y : ℝ) : Prop := x + 2*y - 12 = 0

-- Define the distance functions
noncomputable def d₁ (x y : ℝ) : ℝ := abs (x - 1)
noncomputable def d₂ (x y : ℝ) : ℝ := abs (x + 2*y - 12) / Real.sqrt 5

-- Theorem statement
theorem min_distance_sum :
  ∃ (min : ℝ), min = (11 / 5) * Real.sqrt 5 ∧
  ∀ (x y : ℝ), parabola x y → d₁ x y + d₂ x y ≥ min :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_sum_l1289_128919


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_extreme_points_inequality_l1289_128910

-- Define the function f
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := (x + 2) / Real.exp x + a * x - 2

-- State the theorem
theorem extreme_points_inequality (a : ℝ) (x₁ x₂ : ℝ) :
  (∃ x₁ x₂, x₁ < x₂ ∧ 
   (∀ x, deriv (f a) x = 0 → x = x₁ ∨ x = x₂) ∧
   deriv (f a) x₁ = 0 ∧ deriv (f a) x₂ = 0) →
  Real.exp x₂ - Real.exp x₁ > 2 / a - 2 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_extreme_points_inequality_l1289_128910


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_divisibility_and_inequality_l1289_128957

theorem divisibility_and_inequality (m n : ℤ) : 
  m > 1 → 
  n > 1 → 
  (∀ k : ℤ, k * k ≠ n) → 
  (∃ k : ℤ, k * m = n^2 + n + 1) → 
  |m - n| > Real.sqrt (3 * (n : ℝ)) - 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_divisibility_and_inequality_l1289_128957


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_solution_l1289_128983

noncomputable def fractional_part (x : ℝ) : ℝ := x - ⌊x⌋

def is_arithmetic_sequence (a b c : ℝ) : Prop := b - a = c - b

theorem arithmetic_sequence_solution (x : ℝ) (h1 : x ≠ 0) :
  is_arithmetic_sequence (fractional_part x) ⌊x⌋ x ∧ ⌊x⌋ = 3 * fractional_part x →
  x = 5/3 := by
  sorry

#check arithmetic_sequence_solution

end NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_solution_l1289_128983


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_x_zero_value_l1289_128962

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := x * Real.log x + 2 * x

-- State the theorem
theorem x_zero_value (x₀ : ℝ) (h : x₀ > 0) :
  (deriv f x₀ = 5) → x₀ = Real.exp 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_x_zero_value_l1289_128962


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_octal_subtraction_theorem_l1289_128967

/-- Represents a number in base 8 -/
structure OctalNumber where
  value : ℕ
  is_valid : value < 8^64 := by sorry

/-- Converts an OctalNumber to its decimal (ℕ) representation -/
def octal_to_decimal (n : OctalNumber) : ℕ := sorry

/-- Converts a decimal (ℕ) to its OctalNumber representation -/
def decimal_to_octal (n : ℕ) : OctalNumber := sorry

/-- Subtracts two OctalNumbers and returns the result as an OctalNumber -/
def octal_subtract (a b : OctalNumber) : OctalNumber :=
  decimal_to_octal (octal_to_decimal a - octal_to_decimal b)

/-- Create OctalNumber from a natural number -/
def mk_octal (n : ℕ) : OctalNumber := sorry

theorem octal_subtraction_theorem :
  octal_subtract (mk_octal 46) (mk_octal 17) = mk_octal 27 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_octal_subtraction_theorem_l1289_128967


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unicorn_step_distance_l1289_128987

/-- The distance each unicorn moves forward with each step -/
noncomputable def step_distance (num_unicorns : ℕ) (flowers_per_step : ℕ) (total_distance : ℝ) (total_flowers : ℕ) : ℝ :=
  (total_distance * 1000) / ((total_flowers : ℝ) / (num_unicorns * flowers_per_step : ℝ))

theorem unicorn_step_distance :
  step_distance 6 4 9 72000 = 3 := by
  -- Unfold the definition of step_distance
  unfold step_distance
  -- Simplify the expression
  simp
  -- The proof is completed with 'sorry' as requested
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_unicorn_step_distance_l1289_128987


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_constant_term_binomial_expansion_l1289_128901

theorem constant_term_binomial_expansion :
  let binomial := fun (x : ℝ) => (3*x - 1/x)^6
  ∃ (c : ℝ), ∀ (x : ℝ), x ≠ 0 → 
    ∃ (p : ℝ → ℝ), binomial x = c + x * p x ∧ c = -540 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_constant_term_binomial_expansion_l1289_128901


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_first_valve_fills_in_two_hours_l1289_128922

/-- Represents the time taken to fill the pool with both valves open (in minutes) -/
noncomputable def both_valves_time : ℝ := 48

/-- Represents the difference in water flow between the second and first valve (in cubic meters per minute) -/
noncomputable def valve_difference : ℝ := 50

/-- Represents the capacity of the pool (in cubic meters) -/
noncomputable def pool_capacity : ℝ := 12000

/-- Represents the rate at which the first valve fills the pool (in cubic meters per minute) -/
noncomputable def first_valve_rate : ℝ := pool_capacity / (2 * both_valves_time) - valve_difference / 2

/-- Represents the time taken for the first valve alone to fill the pool (in hours) -/
noncomputable def first_valve_time : ℝ := pool_capacity / (60 * first_valve_rate)

theorem first_valve_fills_in_two_hours : first_valve_time = 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_first_valve_fills_in_two_hours_l1289_128922


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_distance_bound_l1289_128960

noncomputable def hyperbola (x y : ℝ) : Prop := x^2 - y^2 = 1

noncomputable def right_branch (x y : ℝ) : Prop := hyperbola x y ∧ x > 0

noncomputable def line (x y : ℝ) : Prop := y = x + 2

noncomputable def distance_to_line (x y : ℝ) : ℝ := |x - y + 2| / Real.sqrt 2

theorem max_distance_bound (m : ℝ) : 
  (∀ x y : ℝ, right_branch x y → distance_to_line x y > m) → 
  m ≤ Real.sqrt 2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_distance_bound_l1289_128960


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_of_inscribing_square_l1289_128984

/-- The equation of the circle -/
def circle_equation (p : ℝ × ℝ) : Prop :=
  let (x, y) := p
  x^2 = -y^2 + 10*x - 6*y + 15

/-- The circle is inscribed in a square with sides parallel to the axes -/
def inscribed_in_square (c : (ℝ × ℝ) → Prop) : Prop :=
  ∃ (center : ℝ × ℝ) (r : ℝ),
    (∀ (p : ℝ × ℝ), c p ↔ (p.1 - center.1)^2 + (p.2 - center.2)^2 = r^2) ∧
    (∃ (s : ℝ), s = 2*r)

/-- The area of the square that inscribes the given circle is 76 -/
theorem area_of_inscribing_square :
  inscribed_in_square circle_equation →
  (∃ (s : ℝ), s^2 = 76 ∧ 
    (∀ (x y : ℝ), circle_equation (x, y) → 
      x ≥ -s/2 ∧ x ≤ s/2 ∧ y ≥ -s/2 ∧ y ≤ s/2)) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_of_inscribing_square_l1289_128984


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_complement_equality_l1289_128923

-- Define the sets
def U : Set ℝ := Set.univ
def A : Set ℝ := {x : ℝ | 2 < x ∧ x ≤ 4}
def B : Set ℝ := {x : ℝ | ∃ n : ℤ, x = n ∧ n^2 - 7*n + 10 < 0}

-- Define the theorem
theorem intersection_complement_equality :
  A ∩ (Set.compl B) = Set.Ioo 2 3 ∪ Set.Ioo 3 4 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_complement_equality_l1289_128923


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_is_102_l1289_128982

/-- Triangle DEF with altitude DL --/
structure Triangle where
  D : ℝ × ℝ
  E : ℝ × ℝ
  F : ℝ × ℝ
  L : ℝ × ℝ

/-- Given properties of the triangle --/
def triangle_properties (t : Triangle) : Prop :=
  let (dx, dy) := t.D
  let (ex, ey) := t.E
  let (fx, fy) := t.F
  let (lx, ly) := t.L
  ∃ k : ℝ, 0 < k ∧ k < 1 ∧ lx = ex + k * (fx - ex) ∧ ly = ey + k * (fy - ey) ∧  -- L is between E and F
  (dx - lx) * (fx - ex) + (dy - ly) * (fy - ey) = 0 ∧  -- DL perpendicular to EF
  (dx - ex)^2 + (dy - ey)^2 = 15^2 ∧  -- DE = 15
  (lx - ex)^2 + (ly - ey)^2 = 9^2 ∧  -- EL = 9
  (fx - ex)^2 + (fy - ey)^2 = 17^2  -- EF = 17

/-- The area of the triangle --/
noncomputable def triangle_area (t : Triangle) : ℝ :=
  let (dx, dy) := t.D
  let (ex, ey) := t.E
  let (fx, fy) := t.F
  abs ((ex - dx) * (fy - dy) - (fx - dx) * (ey - dy)) / 2

/-- Theorem: The area of the triangle is 102 --/
theorem triangle_area_is_102 (t : Triangle) (h : triangle_properties t) :
  triangle_area t = 102 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_is_102_l1289_128982


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_regular_ticket_price_l1289_128969

/-- Represents the price of a regular admission ticket in dollars -/
def regular_price : ℕ := sorry

/-- Represents the price of a student admission ticket in dollars -/
def student_price : ℕ := 4

/-- Represents the total number of people admitted -/
def total_people : ℕ := 3240

/-- Represents the total revenue from ticket sales in dollars -/
def total_revenue : ℕ := 22680

/-- Represents the ratio of regular tickets to student tickets -/
def ticket_ratio : ℕ := 3

theorem regular_ticket_price : 
  regular_price = 8 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_regular_ticket_price_l1289_128969


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_monotonic_increasing_interval_increasing_interval_is_correct_l1289_128935

-- Define the function f(x)
noncomputable def f (x : ℝ) : ℝ := Real.log (x^2 - 3*x) / Real.log 10

-- Define the domain of f(x)
def domain (x : ℝ) : Prop := x^2 - 3*x > 0

-- State the theorem
theorem monotonic_increasing_interval :
  ∀ x : ℝ, domain x → (∀ y : ℝ, y > x → f y > f x) ↔ x > 3 :=
by
  sorry

-- Define the interval (3, +∞)
def increasing_interval : Set ℝ := Set.Ioi 3

-- State that the increasing interval is (3, +∞)
theorem increasing_interval_is_correct :
  ∀ x : ℝ, x ∈ increasing_interval ↔ (domain x ∧ ∀ y : ℝ, y > x → f y > f x) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_monotonic_increasing_interval_increasing_interval_is_correct_l1289_128935


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_prime_factor_253_l1289_128927

def numbers : List Nat := [143, 187, 221, 253, 289]

/-- The largest prime factor of a natural number -/
def largestPrimeFactor (n : Nat) : Nat :=
  (Nat.factors n).maximum?.getD 1

theorem largest_prime_factor_253 :
  ∀ n ∈ numbers, largestPrimeFactor 253 ≥ largestPrimeFactor n := by
  intro n hn
  sorry

#eval largestPrimeFactor 253  -- Expected output: 23

end NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_prime_factor_253_l1289_128927


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_table_pattern_2017_l1289_128985

/-- Represents the position of a number in the table -/
structure Position where
  row : Nat
  col : Nat
deriving Repr

/-- The table pattern function that maps a natural number to its position -/
def tablePattern (n : Nat) : Position :=
  let cycle := n % 9
  let completeCycles := n / 9
  match cycle with
  | 0 => ⟨completeCycles * 3, 3⟩
  | 1 => ⟨completeCycles * 3 + 1, 1⟩
  | 2 => ⟨completeCycles * 3 + 1, 2⟩
  | 3 => ⟨completeCycles * 3 + 1, 3⟩
  | 4 => ⟨completeCycles * 3 + 2, 1⟩
  | 5 => ⟨completeCycles * 3 + 2, 2⟩
  | 6 => ⟨completeCycles * 3 + 2, 3⟩
  | 7 => ⟨completeCycles * 3 + 3, 1⟩
  | _ => ⟨completeCycles * 3 + 3, 2⟩

theorem table_pattern_2017 :
  let pos := tablePattern 2017
  pos.row - pos.col = 672 := by
  sorry

#eval tablePattern 8
#eval tablePattern 2017

end NUMINAMATH_CALUDE_ERRORFEEDBACK_table_pattern_2017_l1289_128985


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_four_digit_factorial_sum_exists_l1289_128994

def factorial (n : ℕ) : ℕ :=
  match n with
  | 0 => 1
  | n + 1 => (n + 1) * factorial n

def digit_factorial_sum (n : ℕ) : ℕ :=
  let digits := n.digits 10
  (digits.map factorial).sum

def max_digit (n : ℕ) : ℕ :=
  (n.digits 10).foldl max 0

theorem four_digit_factorial_sum_exists : ∃ (n : ℕ), 
  1000 ≤ n ∧ n < 10000 ∧ 
  n = digit_factorial_sum n ∧
  max_digit n ≤ 6 ∧
  n = 1080 := by
  sorry

#eval digit_factorial_sum 1080
#eval max_digit 1080

end NUMINAMATH_CALUDE_ERRORFEEDBACK_four_digit_factorial_sum_exists_l1289_128994


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tin_in_mixed_alloy_l1289_128979

/-- Represents an alloy with its total weight and component ratios -/
structure Alloy where
  total_weight : ℝ
  lead_ratio : ℝ
  tin_ratio : ℝ
  copper_ratio : ℝ

/-- Calculates the amount of tin in an alloy -/
noncomputable def tin_amount (a : Alloy) : ℝ :=
  (a.tin_ratio / (a.lead_ratio + a.tin_ratio + a.copper_ratio)) * a.total_weight

/-- The theorem to be proved -/
theorem tin_in_mixed_alloy (alloy_A alloy_B alloy_C : Alloy)
  (hA : alloy_A = { total_weight := 120, lead_ratio := 2, tin_ratio := 3, copper_ratio := 0 })
  (hB : alloy_B = { total_weight := 180, lead_ratio := 0, tin_ratio := 3, copper_ratio := 5 })
  (hC : alloy_C = { total_weight := 100, lead_ratio := 3, tin_ratio := 2, copper_ratio := 6 }) :
  tin_amount alloy_A + tin_amount alloy_B + tin_amount alloy_C = 157.68 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_tin_in_mixed_alloy_l1289_128979


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_two_mile_revolutions_l1289_128956

/-- The number of revolutions required for a wheel to travel a given distance -/
noncomputable def revolutions (diameter : ℝ) (distance : ℝ) : ℝ :=
  distance / (Real.pi * diameter)

/-- Theorem: The number of revolutions for a 10-foot diameter wheel to travel two miles -/
theorem two_mile_revolutions :
  let wheel_diameter : ℝ := 10
  let two_miles_in_feet : ℝ := 2 * 5280
  revolutions wheel_diameter two_miles_in_feet = 1056 / Real.pi := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_two_mile_revolutions_l1289_128956


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_S_is_origin_l1289_128936

def S : Set ℂ := {z : ℂ | ∃ (r₁ r₂ : ℝ), (1 + 2*Complex.I)*z = r₁ ∧ (2 - 3*Complex.I)*z = r₂}

theorem S_is_origin : S = {0} := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_S_is_origin_l1289_128936


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_constant_term_binomial_expansion_l1289_128920

theorem constant_term_binomial_expansion :
  let general_term (r : ℕ) := (-1 : ℝ)^r * (Nat.choose 6 r : ℝ) * x^(3 - r)
  ∀ x : ℝ, x > 0 →
    ∃ term : ℝ, term = general_term 3 ∧ term = -20 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_constant_term_binomial_expansion_l1289_128920


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_odd_function_implies_a_equals_one_l1289_128937

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a - 2 / (2^x + 1)

theorem odd_function_implies_a_equals_one (a : ℝ) :
  (∀ x, f a x = -f a (-x)) → a = 1 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_odd_function_implies_a_equals_one_l1289_128937


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_second_car_speed_is_40_l1289_128913

/-- Calculates the speed of the second car given the conditions of the problem -/
noncomputable def calculate_second_car_speed (time : ℝ) (first_car_speed : ℝ) (total_distance : ℝ) : ℝ :=
  (total_distance - first_car_speed * time) / time

/-- Theorem stating the speed of the second car is 40 mph under the given conditions -/
theorem second_car_speed_is_40 (time : ℝ) (first_car_speed : ℝ) (total_distance : ℝ)
  (h1 : time = 5)
  (h2 : first_car_speed = 50)
  (h3 : total_distance = 450) :
  calculate_second_car_speed time first_car_speed total_distance = 40 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_second_car_speed_is_40_l1289_128913


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_of_f_l1289_128955

-- Define the function f(x)
noncomputable def f (x : ℝ) : ℝ := min (min (3 * x + 1) (x + 3)) (-x + 9)

-- State the theorem
theorem max_value_of_f : 
  ∃ (M : ℝ), (∀ x, f x ≤ M) ∧ (∃ x₀, f x₀ = M) ∧ M = 6 := by
  -- Proof goes here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_of_f_l1289_128955


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_fuel_at_40_optimal_speed_min_fuel_l1289_128953

-- Define the fuel consumption function
noncomputable def fuel_consumption (x : ℝ) : ℝ := (1 / 128000) * x^3 - (3 / 80) * x + 8

-- Define the total fuel consumption for a 100 km trip
noncomputable def total_fuel (x : ℝ) : ℝ := (fuel_consumption x) * (100 / x)

-- Maximum speed of the car
def max_speed : ℝ := 120

-- Theorem for fuel consumption at 40 km/h
theorem fuel_at_40 : total_fuel 40 = 17.5 := by sorry

-- Theorem for optimal speed
theorem optimal_speed : ∀ x, 0 < x → x ≤ max_speed → total_fuel x ≥ total_fuel 80 := by sorry

-- Theorem for minimum fuel consumption
theorem min_fuel : total_fuel 80 = 11.25 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_fuel_at_40_optimal_speed_min_fuel_l1289_128953


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_lambda_range_l1289_128970

theorem lambda_range (l : ℝ) : 
  (∀ x ∈ Set.Icc (1/2 : ℝ) 2, 2 * x^2 - l * x + 1 ≥ 0) → 
  l ∈ Set.Iic (2 * Real.sqrt 2) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_lambda_range_l1289_128970


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cylinder_volume_difference_l1289_128954

/-- The volume of a cylinder given its circumference and height -/
noncomputable def cylinderVolume (circumference height : ℝ) : ℝ :=
  (circumference ^ 2 * height) / (4 * Real.pi)

/-- The problem statement -/
theorem cylinder_volume_difference : 
  let volumeA := cylinderVolume 10 8
  let volumeB := cylinderVolume 9 7
  Real.pi * |volumeB - volumeA| = 233 / 4 := by
  sorry

#eval (233 : ℚ) / 4  -- To show that 233/4 = 58.25

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cylinder_volume_difference_l1289_128954


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_set_theorem_l1289_128930

def satisfies_conditions (n : ℕ) : Prop :=
  (n < 10^100) ∧
  (n ∣ 2^n) ∧
  ((n-1) ∣ (2^n - 1)) ∧
  ((n-2) ∣ (2^n - 2))

theorem solution_set_theorem : 
  {n : ℕ | n > 0 ∧ satisfies_conditions n} = {4, 16, 65536, 2^256} :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_set_theorem_l1289_128930


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_point_transformation_l1289_128903

def reflect_xz (p : ℝ × ℝ × ℝ) : ℝ × ℝ × ℝ :=
  (p.1, -p.2.1, p.2.2)

def rotate_x_180 (p : ℝ × ℝ × ℝ) : ℝ × ℝ × ℝ :=
  (p.1, -p.2.1, -p.2.2)

def reflect_yz (p : ℝ × ℝ × ℝ) : ℝ × ℝ × ℝ :=
  (-p.1, p.2.1, p.2.2)

def translate (p : ℝ × ℝ × ℝ) (v : ℝ × ℝ × ℝ) : ℝ × ℝ × ℝ :=
  (p.1 + v.1, p.2.1 + v.2.1, p.2.2 + v.2.2)

theorem point_transformation :
  let initial_point : ℝ × ℝ × ℝ := (2, 2, 2)
  let translation_vector : ℝ × ℝ × ℝ := (1, -1, 0)
  translate (reflect_yz (rotate_x_180 (reflect_xz initial_point))) translation_vector = (-1, 1, -2) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_point_transformation_l1289_128903


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_magnitude_product_l1289_128925

theorem complex_magnitude_product : 
  Complex.abs ((3 * Real.sqrt 5 - 3 * Complex.I) * (Real.sqrt 7 + 7 * Complex.I)) = 12 * Real.sqrt 21 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_magnitude_product_l1289_128925

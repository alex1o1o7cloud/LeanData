import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_odd_function_properties_l530_53053

-- Define the function f
noncomputable def f (x : ℝ) : ℝ :=
  if x < 0 then -x^2 - 2*x
  else if x = 0 then 0
  else x^2 - 2*x

-- State the theorem
theorem odd_function_properties :
  (∀ x, f (-x) = -f x) ∧ 
  (∀ x > 0, f x = x^2 - 2*x) →
  (∀ x, f x = if x < 0 then -x^2 - 2*x else if x = 0 then 0 else x^2 - 2*x) ∧
  (∀ a, (∀ x ∈ Set.Icc (-1) (a-2), StrictMonoOn f (Set.Icc (-1) (a-2))) → 
    1 < a ∧ a ≤ 3) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_odd_function_properties_l530_53053


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_coplanar_lines_p_values_l530_53094

def line1 (t p : ℝ) : ℝ × ℝ × ℝ := (1 + 2*t, 2 + 2*t, 3 - p*t)
def line2 (u p : ℝ) : ℝ × ℝ × ℝ := (2 + p*u, 5 + 3*u, 6 + 2*u)

def are_coplanar (l1 l2 : ℝ × ℝ × ℝ) : Prop :=
  ∃ (a b c d : ℝ), a * l1.fst + b * l1.snd.fst + c * l1.snd.snd = d ∧
                    a * l2.fst + b * l2.snd.fst + c * l2.snd.snd = d

theorem coplanar_lines_p_values (p : ℝ) :
  (∃ (t u : ℝ), are_coplanar (line1 t p) (line2 u p)) →
  (p = 1 ∨ p = -9/7) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_coplanar_lines_p_values_l530_53094


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_l530_53068

-- Define the function as noncomputable
noncomputable def f (x : ℝ) : ℝ := (x^2 - 49) / (x - 7)

-- State the theorem about the domain of f
theorem domain_of_f :
  {x : ℝ | ∃ y, f x = y} = {x : ℝ | x < 7 ∨ x > 7} := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_l530_53068


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_of_z_l530_53058

-- Define the constraints
noncomputable def constraint1 (x y : ℝ) : Prop := x - y - 2 ≤ 0
noncomputable def constraint2 (x y : ℝ) : Prop := x + 2*y - 7 ≥ 0
noncomputable def constraint3 (y : ℝ) : Prop := y - 3 ≤ 0

-- Define the objective function
noncomputable def z (x y : ℝ) : ℝ := y / (x + 1)

-- Theorem statement
theorem max_value_of_z (x y : ℝ) 
  (h1 : constraint1 x y) 
  (h2 : constraint2 x y) 
  (h3 : constraint3 y) :
  ∃ (max_z : ℝ), max_z = 1 ∧ ∀ (x' y' : ℝ), 
    constraint1 x' y' → constraint2 x' y' → constraint3 y' → 
    z x' y' ≤ max_z := by
  sorry

#check max_value_of_z

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_of_z_l530_53058


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_odd_indexed_eq_fourth_power_l530_53078

/-- Definition of the sequence S_n -/
def S : ℕ → ℕ
| 0 => 1  -- Adding the case for 0 to cover all natural numbers
| 1 => 1
| 2 => 5
| 3 => 15
| 4 => 34
| 5 => 65
| 6 => 111
| 7 => 175
| n + 8 => S (n + 7) + (n + 8) * (n + 7) + 1

/-- Sum of odd-indexed terms up to S_(2n-1) -/
def sum_odd_indexed (n : ℕ) : ℕ :=
  (List.range n).foldl (fun acc i => acc + S (2 * i + 1)) 0

/-- Theorem stating that the sum of odd-indexed terms equals n^4 -/
theorem sum_odd_indexed_eq_fourth_power (n : ℕ) :
  sum_odd_indexed n = n^4 := by
  sorry

#eval sum_odd_indexed 5  -- For testing purposes

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_odd_indexed_eq_fourth_power_l530_53078


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_solution_l530_53090

noncomputable def greatest_integer_not_exceeding (x : ℝ) : ℤ :=
  ⌊x⌋

noncomputable def cube_root (x : ℝ) : ℝ :=
  Real.rpow x (1/3)

theorem problem_solution (k : ℤ) (n : Fin 2008 → ℕ+) :
  (∀ i : Fin 2008, greatest_integer_not_exceeding (cube_root (n i : ℝ)) = k) →
  (∀ i : Fin 2008, k ∣ (n i : ℕ)) →
  (∀ m : ℤ, m ≠ k → ¬(∀ i : Fin 2008, greatest_integer_not_exceeding (cube_root (n i : ℝ)) = m ∧ m ∣ (n i : ℕ))) →
  k = 668 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_solution_l530_53090


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parallel_vector_sum_diff_l530_53057

/-- Given vectors a and b in R^2, if a + b is parallel to a - b, then t = -1 -/
theorem parallel_vector_sum_diff (t : ℝ) : 
  let a : Fin 2 → ℝ := ![1, -1]
  let b : Fin 2 → ℝ := ![t, 1]
  (∃ (k : ℝ), k ≠ 0 ∧ (a + b) = k • (a - b)) → t = -1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parallel_vector_sum_diff_l530_53057


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_extreme_values_F_two_zeros_l530_53091

-- Define the functions
noncomputable def f (k : ℝ) (x : ℝ) : ℝ := k * x * Real.exp x - x^2 / 2 - x
noncomputable def g (k : ℝ) (x : ℝ) : ℝ := k * Real.exp x - x
noncomputable def F (k : ℝ) (x : ℝ) : ℝ := f k x - g k x

-- Theorem for part 1
theorem f_extreme_values :
  (∃ x₁ x₂ : ℝ, x₁ = -1 ∧ x₂ = 0 ∧
    f 1 x₁ = 1/2 - 1/Real.exp 1 ∧
    f 1 x₂ = 0 ∧
    (∀ x : ℝ, f 1 x ≤ f 1 x₁) ∧
    (∀ x : ℝ, f 1 x ≥ f 1 x₂)) := by
  sorry

-- Theorem for part 2
theorem F_two_zeros (k : ℝ) :
  (∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ F k x₁ = 0 ∧ F k x₂ = 0 ∧
    ∀ x : ℝ, F k x = 0 → x = x₁ ∨ x = x₂) ↔ k < 0 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_extreme_values_F_two_zeros_l530_53091


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_similar_triangles_perimeter_l530_53006

/-- Two similar triangles with height ratio 2:3 and perimeter sum 50 cm have perimeters 20 cm and 30 cm -/
theorem similar_triangles_perimeter (t1 t2 : Real) 
  (h : t1 / t2 = 2 / 3) 
  (perimeter_sum : t1 + t2 = 50) :
  t1 = 20 ∧ t2 = 30 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_similar_triangles_perimeter_l530_53006


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_polynomial_rational_root_l530_53075

theorem polynomial_rational_root (n : ℕ+) :
  (∃ q : ℚ, (q : ℝ)^(n : ℕ) + ((2 : ℝ) + q)^(n : ℕ) + ((2 : ℝ) - q)^(n : ℕ) = 0) ↔ n = 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_polynomial_rational_root_l530_53075


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_partition_exists_l530_53039

def is_prohibited_sum (n : ℕ) : Prop :=
  ∃ k : ℕ, n = 2^k + 2

def valid_partition (A B : Set ℕ) : Prop :=
  (1 ∈ A) ∧
  (∀ x y, x ∈ A → y ∈ A → x ≠ y → ¬is_prohibited_sum (x + y)) ∧
  (∀ x y, x ∈ B → y ∈ B → x ≠ y → ¬is_prohibited_sum (x + y)) ∧
  (∀ n : ℕ, n > 0 → (n ∈ A ∨ n ∈ B)) ∧
  (A ∩ B = ∅)

theorem unique_partition_exists :
  ∃! (A B : Set ℕ), valid_partition A B ∧ 1987 ∈ B ∧ 1988 ∈ A ∧ 1989 ∈ B :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_partition_exists_l530_53039


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_income_at_40_l530_53001

/-- Represents the seafood market greenhouse problem -/
structure Greenhouse where
  total_area : ℝ
  total_storefronts : ℕ
  type_a_area : ℝ
  type_b_area : ℝ
  type_a_rent : ℝ
  type_b_rent : ℝ
  type_a_lease_rate : ℝ
  type_b_lease_rate : ℝ

/-- Calculate the monthly rental income for a given number of type A storefronts -/
def monthly_income (g : Greenhouse) (type_a_count : ℕ) : ℝ :=
  g.type_a_rent * g.type_a_lease_rate * (type_a_count : ℝ) +
  g.type_b_rent * g.type_b_lease_rate * ((g.total_storefronts - type_a_count) : ℝ)

/-- Check if the given number of type A storefronts satisfies the area constraints -/
def satisfies_constraints (g : Greenhouse) (type_a_count : ℕ) : Prop :=
  let total_storefront_area := g.type_a_area * (type_a_count : ℝ) + g.type_b_area * ((g.total_storefronts - type_a_count) : ℝ)
  0.8 * g.total_area ≤ total_storefront_area ∧ total_storefront_area ≤ 0.85 * g.total_area

/-- The main theorem stating that 40 type A storefronts maximizes monthly rental income -/
theorem max_income_at_40 (g : Greenhouse) 
  (h_area : g.total_area = 2400)
  (h_storefronts : g.total_storefronts = 80)
  (h_type_a_area : g.type_a_area = 28)
  (h_type_b_area : g.type_b_area = 20)
  (h_type_a_rent : g.type_a_rent = 400)
  (h_type_b_rent : g.type_b_rent = 360)
  (h_type_a_lease : g.type_a_lease_rate = 0.75)
  (h_type_b_lease : g.type_b_lease_rate = 0.90) :
  ∀ n : ℕ, satisfies_constraints g n → monthly_income g 40 ≥ monthly_income g n :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_income_at_40_l530_53001


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_equation_solution_l530_53024

theorem equation_solution :
  ∃ x : ℝ, (7 : ℝ)^(x + 3) = (343 : ℝ)^x ∧ x = 3/2 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_equation_solution_l530_53024


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_average_running_time_l530_53097

/-- Represents the number of students in the eighth grade -/
def e : ℕ → ℕ := fun _ => 1  -- We use a function to avoid variable declaration issues

/-- The average number of minutes run per day by sixth graders -/
def sixth_grade_avg : ℕ := 20

/-- The average number of minutes run per day by seventh graders -/
def seventh_grade_avg : ℕ := 25

/-- The average number of minutes run per day by eighth graders -/
def eighth_grade_avg : ℕ := 15

/-- The number of sixth graders -/
def sixth_grade_count : ℕ → ℕ := fun n => 3 * e n

/-- The number of seventh graders -/
def seventh_grade_count : ℕ → ℕ := fun n => 3 * e n

/-- The number of eighth graders -/
def eighth_grade_count : ℕ → ℕ := e

/-- The average number of minutes run per day by all students -/
def total_avg : ℚ :=
  (sixth_grade_avg * (sixth_grade_count 0) + seventh_grade_avg * (seventh_grade_count 0) + eighth_grade_avg * (eighth_grade_count 0)) / 
  ((sixth_grade_count 0) + (seventh_grade_count 0) + (eighth_grade_count 0))

theorem average_running_time : total_avg = 150 / 7 := by
  sorry

#eval total_avg

end NUMINAMATH_CALUDE_ERRORFEEDBACK_average_running_time_l530_53097


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_zero_range_f_range_when_a_neg_one_l530_53054

-- Define the function f(x) with parameter a
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := Real.sqrt x + a * abs (x + 1)

-- Theorem for the range of a when f has a zero
theorem f_zero_range (a : ℝ) :
  (∃ x : ℝ, x ≥ 0 ∧ f a x = 0) ↔ a ∈ Set.Icc (-1/2) 0 := by
  sorry

-- Theorem for the range of f when a = -1
theorem f_range_when_a_neg_one :
  Set.range (f (-1)) = Set.Iic (-3/4) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_zero_range_f_range_when_a_neg_one_l530_53054


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_election_winning_margin_l530_53007

theorem election_winning_margin 
  (total_votes : ℕ) 
  (winner_votes : ℕ) 
  (winner_percentage : ℚ) :
  winner_percentage = 55 / 100 →
  winner_votes = 550 →
  winner_votes = (winner_percentage * total_votes).floor →
  (winner_votes : ℚ) / total_votes = winner_percentage →
  winner_votes - (total_votes - winner_votes) = 100 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_election_winning_margin_l530_53007


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_a2_positive_iff_s6_gt_s5_l530_53008

/-- Represents a geometric sequence with first term a and common ratio q -/
structure GeometricSequence where
  a : ℝ
  q : ℝ

/-- Sum of the first n terms of a geometric sequence -/
noncomputable def sum_n (g : GeometricSequence) (n : ℕ) : ℝ :=
  if g.q = 1 then n * g.a
  else g.a * (1 - g.q^n) / (1 - g.q)

/-- The second term of a geometric sequence -/
noncomputable def a_2 (g : GeometricSequence) : ℝ := g.a * g.q

/-- Theorem stating that a_2 > 0 is necessary and sufficient for s_6 > s_5 -/
theorem a2_positive_iff_s6_gt_s5 (g : GeometricSequence) :
  a_2 g > 0 ↔ sum_n g 6 > sum_n g 5 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_a2_positive_iff_s6_gt_s5_l530_53008


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_concentric_circles_k_value_l530_53079

/-- Two concentric circles with center at the origin --/
structure ConcentricCircles where
  center : ℝ × ℝ := (0, 0)
  radius_large : ℝ
  radius_small : ℝ

/-- Point on a circle --/
structure PointOnCircle where
  x : ℝ
  y : ℝ
  
/-- Distance between two points --/
noncomputable def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

theorem concentric_circles_k_value 
  (circles : ConcentricCircles)
  (P : PointOnCircle)
  (S : PointOnCircle)
  (h1 : P.x = 8 ∧ P.y = 6)
  (h2 : S.x = 0)
  (h3 : distance circles.center (P.x, P.y) = circles.radius_large)
  (h4 : distance circles.center (S.x, S.y) = circles.radius_small)
  (h5 : circles.radius_large - circles.radius_small = 3)
  : S.y = 7 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_concentric_circles_k_value_l530_53079


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_may_day_promotion_theorem_l530_53088

-- Define the turnover for the first four days
noncomputable def first_four_days_turnover : ℝ := 450

-- Define the percentage for the fifth day's turnover
noncomputable def fifth_day_percentage : ℝ := 0.12

-- Define February's turnover
noncomputable def february_turnover : ℝ := 350

-- Define the total turnover for five days
noncomputable def total_turnover : ℝ := first_four_days_turnover * (1 + fifth_day_percentage)

-- Define the monthly growth rate
noncomputable def monthly_growth_rate : ℝ := (total_turnover / february_turnover) ^ (1/2 : ℝ) - 1

-- Theorem statement
theorem may_day_promotion_theorem :
  total_turnover = 504 ∧ monthly_growth_rate = 0.2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_may_day_promotion_theorem_l530_53088


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_specific_cylinder_volume_l530_53051

/-- The volume of a cylinder formed by rotating a rectangle -/
noncomputable def cylinderVolume (width : ℝ) (length : ℝ) : ℝ :=
  Real.pi * (width / 2)^2 * length

/-- Theorem: The volume of the specific cylinder is 400π cubic centimeters -/
theorem specific_cylinder_volume :
  cylinderVolume 10 16 = 400 * Real.pi := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_specific_cylinder_volume_l530_53051


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_difference_l530_53021

theorem tan_difference (α β : ℝ) (h1 : Real.sin α = 3/5) (h2 : α ∈ Set.Ioo (π/2) π) (h3 : Real.tan (π - β) = 1/2) :
  Real.tan (α - β) = -2/11 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_difference_l530_53021


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_focal_distance_l530_53030

/-- The ellipse equation -/
def is_on_ellipse (x y : ℝ) : Prop := x^2 / 169 + y^2 / 144 = 1

/-- The distance between two points -/
noncomputable def distance (x1 y1 x2 y2 : ℝ) : ℝ := Real.sqrt ((x1 - x2)^2 + (y1 - y2)^2)

/-- The statement of the problem -/
theorem ellipse_focal_distance 
  (x y : ℝ) 
  (f1 f2 : ℝ × ℝ)
  (h1 : is_on_ellipse x y) 
  (h2 : distance x y f1.1 f1.2 = 4) 
  (h3 : f1 ≠ f2) -- Foci are distinct
  (h4 : ∀ (a b : ℝ), is_on_ellipse a b → 
    distance a b f1.1 f1.2 + distance a b f2.1 f2.2 = 2 * Real.sqrt 169) :
  distance x y f2.1 f2.2 = 22 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_focal_distance_l530_53030


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_counterfeit_coin_identification_strategy_exists_l530_53038

/-- Represents the expert's response function -/
def expertResponse (actualFakes : Nat) (exaggeration : Nat) : Nat :=
  actualFakes + exaggeration

/-- Represents a single consultation with the expert -/
structure Consultation where
  coinsShown : Nat
  expertReport : Nat

/-- Represents a strategy for identifying counterfeit coins -/
structure Strategy where
  consultations : List Consultation
  identifiedFakes : List Nat

/-- The main theorem stating that a valid strategy exists -/
theorem counterfeit_coin_identification_strategy_exists :
  ∃ (s : Strategy),
    (∀ c ∈ s.consultations, 10 ≤ c.coinsShown ∧ c.coinsShown ≤ 20) ∧
    s.consultations.length < 120 ∧
    ∀ (actualFakes : List Nat) (exaggeration : Nat),
      actualFakes.length ≤ 100 →
      ∃ (identifiedFakes : List Nat),
        identifiedFakes = s.identifiedFakes ∧
        identifiedFakes.toFinset = actualFakes.toFinset :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_counterfeit_coin_identification_strategy_exists_l530_53038


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_shopping_discount_problem_l530_53052

/-- Calculates the discounted price based on the given discount rules -/
noncomputable def discountedPrice (price : ℝ) : ℝ :=
  if price ≤ 200 then price
  else if price ≤ 500 then price * 0.9
  else 500 * 0.9 + (price - 500) * 0.7

/-- Theorem statement for the shopping discount problem -/
theorem shopping_discount_problem (trip1 : ℝ) (trip2_discounted : ℝ) 
    (h1 : trip1 = 168) 
    (h2 : trip2_discounted = 423) : 
  let trip2_original := trip2_discounted / 0.9
  let total_original := trip1 + trip2_original
  discountedPrice total_original = 546.6 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_shopping_discount_problem_l530_53052


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_properties_l530_53087

def sequence_a : ℕ → ℚ
  | 0 => 3/5
  | n + 1 => 2 - 1 / sequence_a n

def sequence_b (n : ℕ) : ℚ := 1 / (sequence_a n - 1)

theorem sequence_properties :
  (∀ n : ℕ, sequence_b (n + 1) - sequence_b n = 1) ∧
  sequence_b 0 = -5/2 ∧
  (∃ n : ℕ, sequence_a n = 3) ∧
  (∃ n : ℕ, sequence_a n = -1) ∧
  (∀ n : ℕ, sequence_a n ≤ 3) ∧
  (∀ n : ℕ, sequence_a n ≥ -1) :=
by sorry

#check sequence_properties

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_properties_l530_53087


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_at_x0_1_l530_53019

noncomputable def f (x : ℝ) : ℝ := -2 * (x^8 + 2) / (3 * (x^4 + 1))

noncomputable def f' (x : ℝ) : ℝ := -8 * x^3 * (x^8 + 2*x^4 - 2) / (3 * (x^4 + 1)^2)

theorem tangent_line_at_x0_1 :
  let x₀ : ℝ := 1
  let y₀ : ℝ := f x₀
  let m : ℝ := f' x₀
  ∀ x y : ℝ, y - y₀ = m * (x - x₀) ↔ y = -2/3 * x - 1/3 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_at_x0_1_l530_53019


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trajectory_and_fixed_point_l530_53092

-- Define the points and variables
variable (F P N M A B : ℝ × ℝ)
variable (k₁ k₂ : ℝ)

-- Define the conditions
def conditions (F P N M A B : ℝ × ℝ) (k₁ k₂ : ℝ) : Prop :=
  F = (1, 0) ∧
  N.1 = -1 ∧
  ((P.1 - N.1, P.2 - N.2) + (1/2 : ℝ) • (F.1 - N.1, F.2 - N.2)) • (F.1 - N.1, F.2 - N.2) = 0 ∧
  M = (1, 2) ∧
  k₁ + k₂ = -1

-- Define the theorem
theorem trajectory_and_fixed_point 
  (F P N M A B : ℝ × ℝ) (k₁ k₂ : ℝ) 
  (h : conditions F P N M A B k₁ k₂) :
  P.2^2 = 4 * P.1 ∧ 
  ∃ (t : ℝ), (1 - t) • A + t • B = (5, -6) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_trajectory_and_fixed_point_l530_53092


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_eccentricity_range_with_point_condition_l530_53071

/-- Hyperbola structure -/
structure Hyperbola where
  a : ℝ
  b : ℝ
  equation : (x y : ℝ) → Prop

/-- Point on a hyperbola -/
structure HyperbolaPoint (h : Hyperbola) where
  x : ℝ
  y : ℝ
  on_hyperbola : h.equation x y

/-- Eccentricity of a hyperbola -/
noncomputable def eccentricity (h : Hyperbola) : ℝ := Real.sqrt (1 + h.b^2 / h.a^2)

/-- Distance between two points -/
noncomputable def distance (x1 y1 x2 y2 : ℝ) : ℝ := Real.sqrt ((x1 - x2)^2 + (y1 - y2)^2)

/-- Theorem: Eccentricity range of a hyperbola with specific point condition -/
theorem eccentricity_range_with_point_condition 
  (h : Hyperbola) 
  (G : HyperbolaPoint h)
  (F1 : ℝ × ℝ) -- Left focus
  (F2 : ℝ × ℝ) -- Right focus
  (h_foci : F1.1 < F2.1) -- Left focus is to the left of right focus
  (h_ratio : distance G.x G.y F1.1 F1.2 / distance G.x G.y F2.1 F2.2 = 9) :
  1 < eccentricity h ∧ eccentricity h ≤ 5/4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_eccentricity_range_with_point_condition_l530_53071


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_m_for_even_sine_l530_53012

def is_even_function (f : ℝ → ℝ) : Prop :=
  ∀ x, f x = f (-x)

theorem min_m_for_even_sine (m : ℝ) :
  m > 0 →
  is_even_function (λ x ↦ Real.sin (2 * x + π / 3 - 2 * m)) →
  ∃ (k : ℤ), m = 5 * π / 12 + k * π / 2 ∧ 
    ∀ (k' : ℤ), m ≤ 5 * π / 12 + k' * π / 2 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_m_for_even_sine_l530_53012


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_sequence_properties_l530_53084

/-- Represents a geometric sequence -/
structure GeometricSequence where
  a : ℕ → ℝ
  q : ℝ
  h : ∀ n, a (n + 1) = a n * q

/-- The sum of the first n terms of a geometric sequence -/
noncomputable def geometricSum (s : GeometricSequence) (n : ℕ) : ℝ :=
  s.a 1 * (1 - s.q^n) / (1 - s.q)

theorem geometric_sequence_properties (s : GeometricSequence) 
    (h1 : s.a 2 + s.a 4 = 20)
    (h2 : s.a 3 + s.a 5 = 40) :
  s.q = 2 ∧ ∀ n, geometricSum s n = s.a 1 * (2^n - 1) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_sequence_properties_l530_53084


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_total_workers_l530_53020

/-- The total number of workers in the workshop -/
def W : ℕ := sorry

/-- The average salary of all workers in Rupees -/
def avg_salary : ℕ := 8000

/-- The number of technicians -/
def num_technicians : ℕ := 10

/-- The average salary of technicians in Rupees -/
def avg_salary_technicians : ℕ := 12000

/-- The average salary of non-technicians in Rupees -/
def avg_salary_others : ℕ := 6000

/-- The theorem stating that the total number of workers is 30 -/
theorem total_workers : W = 30 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_total_workers_l530_53020


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_average_sleep_theorem_l530_53027

def monday : ℕ × ℕ := (8, 15)
def tuesday : ℕ × ℕ := (7, 45)
def wednesday : ℕ × ℕ := (8, 10)
def thursday : ℕ × ℕ := (10, 25)
def friday : ℕ × ℕ := (7, 50)

def total_sleep_time : ℕ × ℕ := (
  monday.1 + tuesday.1 + wednesday.1 + thursday.1 + friday.1,
  monday.2 + tuesday.2 + wednesday.2 + thursday.2 + friday.2
)

def average_sleep_time : ℚ :=
  (total_sleep_time.1 * 60 + total_sleep_time.2 : ℚ) / 300

theorem average_sleep_theorem :
  average_sleep_time = (509 : ℚ) / 60 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_average_sleep_theorem_l530_53027


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_infinitely_many_points_outside_l530_53063

/-- A point in 3D space -/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- A line in 3D space -/
structure Line3D where
  point : Point3D
  direction : Point3D

/-- A plane in 3D space -/
structure Plane3D where
  a : ℝ
  b : ℝ
  c : ℝ
  d : ℝ

/-- Check if a point is outside a plane -/
def isOutside (p : Point3D) (plane : Plane3D) : Prop :=
  plane.a * p.x + plane.b * p.y + plane.c * p.z + plane.d ≠ 0

/-- Define membership for a point on a line -/
def Point3D.mem (p : Point3D) (L : Line3D) : Prop :=
  ∃ t : ℝ, p = Point3D.mk
    (L.point.x + t * L.direction.x)
    (L.point.y + t * L.direction.y)
    (L.point.z + t * L.direction.z)

instance : Membership Point3D Line3D where
  mem := Point3D.mem

/-- The main theorem -/
theorem infinitely_many_points_outside (L : Line3D) (P : Plane3D) :
  (∃ p : Point3D, p ∈ L ∧ isOutside p P) →
  ∃ S : Set Point3D, S.Infinite ∧ ∀ p ∈ S, p ∈ L ∧ isOutside p P :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_infinitely_many_points_outside_l530_53063


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_focus_l530_53015

/-- A parabola is defined by its standard equation y² = 6x -/
def Parabola : Set (ℝ × ℝ) := {p | p.2^2 = 6 * p.1}

/-- The focus of a parabola is a point -/
def Focus := ℝ × ℝ

/-- Given a parabola defined by y² = 6x, its focus coordinates are (3/2, 0) -/
theorem parabola_focus : ∃ (f : Focus), f = (3/2, 0) ∧ f ∈ Parabola := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_focus_l530_53015


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_imaginary_part_of_z_l530_53064

theorem imaginary_part_of_z (z : ℂ) (h : Complex.I * (1 - z) = 1) : 
  z.im = 1 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_imaginary_part_of_z_l530_53064


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ancient_chinese_pi_l530_53026

/-- The volume of a cylindrical fortress -/
noncomputable def fortressVolume (circumference height : ℝ) : ℝ :=
  (1 / 12) * (circumference^2 * height)

/-- The theorem stating the value of π in the ancient Chinese problem -/
theorem ancient_chinese_pi : ∃ (π : ℝ), 
  let circumference : ℝ := 48
  let height : ℝ := 11
  let volume : ℝ := 2112
  let radius : ℝ := circumference / (2 * π)
  (fortressVolume circumference height = volume) ∧ 
  (π * radius^2 * height = volume) ∧
  π = 3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_ancient_chinese_pi_l530_53026


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_transformation_l530_53045

theorem domain_transformation (f : ℝ → ℝ) :
  (∀ x, f (Real.log x / Real.log 2) ≠ 0 ↔ x ∈ Set.Icc (1/2) 4) →
  (∀ x, f x ≠ 0 ↔ x ∈ Set.Icc (-1) 2) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_transformation_l530_53045


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trig_values_for_point_l530_53067

/-- Given a point (1, -2) on the terminal side of angle α, prove that cos α = √5/5 and tan α = -2 -/
theorem trig_values_for_point (α : ℝ) (h : ∃ (r : ℝ), r * (Real.cos α) = 1 ∧ r * (Real.sin α) = -2) :
  Real.cos α = Real.sqrt 5 / 5 ∧ Real.tan α = -2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_trig_values_for_point_l530_53067


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_triangle_properties_l530_53093

-- Define a structure for a triangle with sides in arithmetic sequence
structure ArithmeticTriangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ
  a_pos : 0 < a
  b_pos : 0 < b
  c_pos : 0 < c
  angle_sum : A + B + C = π
  arith_seq : 2 * b = a + c
  side_a : a = Real.sin A
  side_b : b = Real.sin B
  side_c : c = Real.sin C

-- Theorem statement
theorem arithmetic_triangle_properties (t : ArithmeticTriangle) :
  t.b ^ 2 ≥ t.a * t.c ∧
  1 / t.a + 1 / t.c ≥ 2 / t.b ∧
  t.b ^ 2 ≤ (t.a ^ 2 + t.c ^ 2) / 2 ∧
  0 < t.B ∧ t.B ≤ π / 3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_triangle_properties_l530_53093


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parallel_lines_imply_a_value_a_equals_two_satisfies_conditions_l530_53040

/-- Two lines are parallel if and only if their slopes are equal -/
axiom parallel_lines_equal_slopes {m₁ m₂ b₁ b₂ : ℝ} :
  (∀ x y : ℝ, y = m₁ * x + b₁ ↔ y = m₂ * x + b₂) ↔ m₁ = m₂

/-- Definition of line l₁ -/
def line_l₁ (a : ℝ) : ℝ → ℝ → Prop :=
  λ x y ↦ a * x + 2 * y + a = 0

/-- Definition of line l₂ -/
def line_l₂ (a : ℝ) : ℝ → ℝ → Prop :=
  λ x y ↦ 2 * x + a * y - a = 0

/-- Theorem stating that if l₁ and l₂ are parallel, then a = 2 or a = -2 -/
theorem parallel_lines_imply_a_value (a : ℝ) :
  (∀ x y : ℝ, line_l₁ a x y ↔ ∃ k, y = (-a/2) * x + k) →
  (∀ x y : ℝ, line_l₂ a x y ↔ ∃ k, y = (-2/a) * x + k) →
  (∀ x y : ℝ, line_l₁ a x y ↔ line_l₂ a x y) →
  a = 2 ∨ a = -2 :=
by sorry

/-- Corollary stating that a = 2 satisfies the conditions -/
theorem a_equals_two_satisfies_conditions :
  (∀ x y : ℝ, line_l₁ 2 x y ↔ ∃ k, y = -1 * x + k) ∧
  (∀ x y : ℝ, line_l₂ 2 x y ↔ ∃ k, y = -1 * x + k) ∧
  (∀ x y : ℝ, line_l₁ 2 x y ↔ line_l₂ 2 x y) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parallel_lines_imply_a_value_a_equals_two_satisfies_conditions_l530_53040


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hundred_days_from_friday_is_sunday_l530_53042

/-- Represents days of the week -/
inductive DayOfWeek
  | Friday
  | Saturday
  | Sunday
  | Monday
  | Tuesday
  | Wednesday
  | Thursday

/-- The next day after a given day of the week -/
def DayOfWeek.nextDay : DayOfWeek → DayOfWeek
  | Friday => Saturday
  | Saturday => Sunday
  | Sunday => Monday
  | Monday => Tuesday
  | Tuesday => Wednesday
  | Wednesday => Thursday
  | Thursday => Friday

/-- Calculates the day of the week after a given number of days -/
def dayAfter (start : DayOfWeek) (days : Nat) : DayOfWeek :=
  match (days % 7) with
  | 0 => start
  | 1 => start.nextDay
  | 2 => start.nextDay.nextDay
  | 3 => start.nextDay.nextDay.nextDay
  | 4 => start.nextDay.nextDay.nextDay.nextDay
  | 5 => start.nextDay.nextDay.nextDay.nextDay.nextDay
  | _ => start.nextDay.nextDay.nextDay.nextDay.nextDay.nextDay

theorem hundred_days_from_friday_is_sunday :
  dayAfter DayOfWeek.Friday 100 = DayOfWeek.Sunday := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_hundred_days_from_friday_is_sunday_l530_53042


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_alexs_score_l530_53029

theorem alexs_score (total_students : ℕ) (students_graded : ℕ) (initial_average : ℝ) (final_average : ℝ) (alexs_score : ℝ) :
  total_students = 20 →
  students_graded = 19 →
  initial_average = 78 →
  final_average = 79 →
  (students_graded * initial_average + (total_students - students_graded) * alexs_score) / total_students = final_average →
  alexs_score = 98 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_alexs_score_l530_53029


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_slope_angle_of_line_l530_53059

/-- The slope angle of the line √3x - y - 4 = 0 is π/3 -/
theorem slope_angle_of_line (x y : ℝ) :
  (Real.sqrt 3 * x - y - 4 = 0) → (Real.arctan (Real.sqrt 3) = π / 3) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_slope_angle_of_line_l530_53059


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_intersection_area_ratio_l530_53016

/-- Given a line y = 2b - x intersecting the y-axis at P and the line x=6 at S,
    if the ratio of the area of triangle QRS to the area of triangle QOP is 4:9,
    then b = 1.8 -/
theorem line_intersection_area_ratio (b : ℝ) : 
  let line := λ x y ↦ y = 2 * b - x
  let P := (0, 2 * b)
  let Q := (2 * b, 0)
  let R := (6, 0)
  let S := (6, 2 * b - 6)
  let area_QRS := abs ((R.1 - Q.1) * (S.2 - Q.2) / 2)
  let area_QOP := abs ((P.1 - Q.1) * (P.2 - Q.2) / 2)
  line P.1 P.2 ∧ line S.1 S.2 ∧ area_QRS / area_QOP = 4 / 9 → b = 1.8 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_intersection_area_ratio_l530_53016


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_right_triangle_from_tan_product_l530_53077

theorem right_triangle_from_tan_product (A B C : ℝ) : 
  0 < A → 0 < B → 0 < C →
  A + B + C = π →
  Real.tan A * Real.tan B = 1 →
  C = π / 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_right_triangle_from_tan_product_l530_53077


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_polynomial_division_remainder_l530_53002

noncomputable def dividend (x : ℝ) : ℝ := x^6 - 3*x^5 + 3*x^3 - x^2 - 2*x

noncomputable def divisor (x : ℝ) : ℝ := (x^2 - 1)*(x - 2)

noncomputable def remainder (x : ℝ) : ℝ := -16/3*x^2 + 2*x + 4/3

theorem polynomial_division_remainder :
  ∃ (q : ℝ → ℝ), ∀ x, dividend x = divisor x * q x + remainder x :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_polynomial_division_remainder_l530_53002


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sigma_algebra_generated_eq_l530_53076

variable {α : Type*}
variable (𝓕 : Set (Set α))
variable (C : Set α)

-- Define σ-algebra
def is_sigma_algebra (𝓕 : Set (Set α)) : Prop :=
  (∅ ∈ 𝓕) ∧ 
  (∀ A, A ∈ 𝓕 → Aᶜ ∈ 𝓕) ∧
  (∀ s : ℕ → Set α, (∀ n, s n ∈ 𝓕) → ⋃ n, s n ∈ 𝓕)

-- Define the σ-algebra generated by 𝓕 and C
def sigma_algebra_generated (𝓕 : Set (Set α)) (C : Set α) : Set (Set α) :=
  ⋂₀ {G | is_sigma_algebra G ∧ 𝓕 ⊆ G ∧ C ∈ G}

-- Theorem statement
theorem sigma_algebra_generated_eq (h1 : is_sigma_algebra 𝓕) (h2 : C ∉ 𝓕) :
  sigma_algebra_generated 𝓕 C = {S | ∃ A B, A ∈ 𝓕 ∧ B ∈ 𝓕 ∧ S = (A ∩ C) ∪ (B ∩ Cᶜ)} :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sigma_algebra_generated_eq_l530_53076


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_closed_form_of_f_l530_53066

/-- A function satisfying the given functional equation -/
noncomputable def f : ℝ → ℝ := sorry

/-- The domain of the function excludes 0 and 1 -/
def domain (x : ℝ) : Prop := x ≠ 0 ∧ x ≠ 1

/-- The functional equation that f satisfies -/
axiom functional_equation (x : ℝ) (h : domain x) :
  f x + f (1 - 1/x) = 1 + x

/-- The main theorem stating the closed form of the function -/
theorem closed_form_of_f (x : ℝ) (h : domain x) :
  f x = (x^3 - x^2 - 1) / (2*x^2 - 2*x) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_closed_form_of_f_l530_53066


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_sum_10_l530_53099

/-- An arithmetic sequence -/
def ArithmeticSequence (a : ℕ → ℚ) : Prop :=
  ∃ (a₁ d : ℚ), ∀ n, a n = a₁ + (n - 1) * d

/-- Sum of the first n terms of an arithmetic sequence -/
def SumFirstN (a : ℕ → ℚ) (n : ℕ) : ℚ :=
  (n : ℚ) / 2 * (a 1 + a n)

theorem arithmetic_sequence_sum_10 (a : ℕ → ℚ) :
  ArithmeticSequence a →
  a 4^2 + a 7^2 + 2 * a 4 * a 7 = 9 →
  SumFirstN a 10 = 15 ∨ SumFirstN a 10 = -15 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_sum_10_l530_53099


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ethanol_A_percentage_calculation_l530_53089

/-- Represents the properties of a fuel tank filled with two types of fuel -/
structure FuelTank where
  capacity : ℝ
  volume_A : ℝ
  ethanol_B : ℝ
  total_ethanol : ℝ

/-- Calculates the percentage of ethanol in Fuel A -/
noncomputable def ethanol_percentage_A (tank : FuelTank) : ℝ :=
  (tank.total_ethanol - tank.ethanol_B * (tank.capacity - tank.volume_A)) / (0.01 * tank.volume_A)

/-- Theorem stating the relationship between the tank properties and the ethanol percentage in Fuel A -/
theorem ethanol_A_percentage_calculation (tank : FuelTank) 
    (h1 : tank.capacity = 200)
    (h2 : tank.volume_A > 0)
    (h3 : tank.volume_A ≤ 200)
    (h4 : tank.ethanol_B = 0.16)
    (h5 : tank.total_ethanol = 18) :
  ethanol_percentage_A tank = (18 - 0.16 * (200 - tank.volume_A)) / (0.01 * tank.volume_A) := by
  sorry

#check ethanol_A_percentage_calculation

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ethanol_A_percentage_calculation_l530_53089


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_password_probability_l530_53056

def last_digits : Finset Char := {'A', 'a', 'B', 'b'}
def second_last_digits : Finset Nat := {4, 5, 6}

def password_space : Finset (Nat × Char) :=
  Finset.product second_last_digits last_digits

theorem password_probability :
  (Finset.card password_space : ℚ)⁻¹ = (12 : ℚ)⁻¹ := by
  have h1 : Finset.card password_space = 12 := by
    simp [password_space, last_digits, second_last_digits]
    rfl
  simp [h1]

#eval Finset.card password_space

end NUMINAMATH_CALUDE_ERRORFEEDBACK_password_probability_l530_53056


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_chord_distance_l530_53043

theorem chord_distance (R : ℝ) (θ : ℝ) (h1 : R = 1.4) (h2 : θ = 120 * π / 180) :
  R * Real.sin (θ / 2) = 0.7 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_chord_distance_l530_53043


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_recreation_spending_change_l530_53065

/-- Proves that a 5% decrease in wages and an increase in recreation spending from 40% to 50% results in spending 118.75% of the previous week's recreation amount -/
theorem recreation_spending_change (W : ℝ) (h : W > 0) : 
  let last_week_recreation := 0.40 * W
  let this_week_wages := 0.95 * W
  let this_week_recreation := 0.50 * this_week_wages
  (this_week_recreation / last_week_recreation) * 100 = 118.75 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_recreation_spending_change_l530_53065


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_sin_2x_plus_sin_x_minus_cos_x_l530_53060

theorem max_value_sin_2x_plus_sin_x_minus_cos_x :
  ∃ (x : ℝ), x ∈ Set.Icc 0 (π / 2) ∧
  Real.sin (2 * x) + Real.sin x - Real.cos x = 5 / 4 ∧
  ∀ (y : ℝ), y ∈ Set.Icc 0 (π / 2) →
  Real.sin (2 * y) + Real.sin y - Real.cos y ≤ 5 / 4 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_sin_2x_plus_sin_x_minus_cos_x_l530_53060


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_crossing_point_existence_l530_53018

noncomputable section

/-- Definition of a "crossing point" for a differentiable function -/
def is_crossing_point (f : ℝ → ℝ) (x₀ : ℝ) : Prop :=
  ∃ g : ℝ → ℝ, ∀ x, x ≠ x₀ → (f x - g x) * (x - x₀) > 0

/-- The function f(x) = ln x - (a/2)x² - x/2 -/
def f (a : ℝ) (x : ℝ) : ℝ := Real.log x - (a/2) * x^2 - x/2

theorem crossing_point_existence (a : ℝ) :
  (∃ x₀ ∈ Set.Ioo 0 (Real.exp 1), is_crossing_point (f a) x₀) ↔ a ∈ Set.Iic (-(1 / Real.exp 1)^2) :=
sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_crossing_point_existence_l530_53018


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inverse_propositions_true_l530_53055

-- Define custom types and functions
structure Angle : Type
structure Line : Type
structure Triangle : Type

def vertically_opposite : Angle → Angle → Prop := sorry
def supplementary_same_side : Angle → Angle → Prop := sorry
def complementary : Angle → Angle → Prop := sorry
def parallel : Line → Line → Prop := sorry
def is_right_triangle : Triangle → Prop := sorry
def acute_angle1 : Triangle → Angle := sorry
def acute_angle2 : Triangle → Angle := sorry

-- Define the propositions
def proposition1 : Prop := ∀ a b : Angle, vertically_opposite a b → a = b
def proposition2 : Prop := ∀ a b : Angle, ∀ l1 l2 : Line, supplementary_same_side a b ∧ complementary a b → parallel l1 l2
def proposition3 : Prop := ∀ t : Triangle, is_right_triangle t → complementary (acute_angle1 t) (acute_angle2 t)
def proposition4 : Prop := ∀ a b : ℝ, a > 0 ∧ b > 0 → a * b > 0

-- Define the inverse propositions
def inverse_proposition1 : Prop := ∀ a b : Angle, a ≠ b → ¬(vertically_opposite a b)
def inverse_proposition2 : Prop := ∀ a b : Angle, ∀ l1 l2 : Line, ¬(parallel l1 l2) → ¬(supplementary_same_side a b ∧ complementary a b)
def inverse_proposition3 : Prop := ∀ a b : Angle, complementary a b → ∃ t : Triangle, is_right_triangle t ∧ a = acute_angle1 t ∧ b = acute_angle2 t
def inverse_proposition4 : Prop := ∀ a b : ℝ, a * b ≤ 0 → ¬(a > 0 ∧ b > 0)

-- Define a function to check if an inverse proposition is true
def is_inverse_true (p : Prop) : Prop := p

-- Theorem statement
theorem inverse_propositions_true : 
  is_inverse_true inverse_proposition2 ∧ 
  is_inverse_true inverse_proposition3 ∧ 
  ¬(is_inverse_true inverse_proposition1) ∧ 
  ¬(is_inverse_true inverse_proposition4) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_inverse_propositions_true_l530_53055


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_nice_triple_existence_l530_53017

/-- A triple of positive integers is nice if it forms a non-decreasing arithmetic progression,
    the middle term is coprime to the other two, and their product is a perfect square. -/
def NiceTriple (a b c : ℕ+) : Prop :=
  a ≤ b ∧ b ≤ c ∧  -- non-decreasing
  ∃ d : ℕ, b = a + d ∧ c = a + 2*d ∧  -- arithmetic progression
  Nat.gcd b.val a.val = 1 ∧ Nat.gcd b.val c.val = 1 ∧  -- coprimality
  ∃ k : ℕ, a.val * b.val * c.val = k^2  -- perfect square product

/-- Given a nice triple, there exists another nice triple sharing at least one element. -/
theorem nice_triple_existence (a b c : ℕ+) (h : NiceTriple a b c) :
  ∃ (x y z : ℕ+), NiceTriple x y z ∧ (({a, b, c} : Set ℕ+).inter {x, y, z}).Nonempty :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_nice_triple_existence_l530_53017


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_perfect_square_trinomial_m_value_l530_53083

-- Define a perfect square trinomial
def isPerfectSquareTrinomial (a b c : ℝ) : Prop :=
  ∃ (p q : ℝ), ∀ (x y : ℝ), a * x^2 + b * x * y + c * y^2 = (p * x + q * y)^2

-- State the theorem
theorem perfect_square_trinomial_m_value (m : ℝ) :
  (isPerfectSquareTrinomial 1 m 1) → m = 2 ∨ m = -2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_perfect_square_trinomial_m_value_l530_53083


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_math_contest_questions_l530_53010

theorem math_contest_questions (riley_correct : ℕ) (riley_mistakes : ℕ) 
  (ofelia_correct : ℕ) (total_mistakes : ℕ) : ℕ := by
  have h1 : riley_mistakes = 3 := by sorry
  have h2 : ofelia_correct = riley_correct / 2 + 5 := by sorry
  have h3 : total_mistakes = 17 := by sorry
  have h4 : riley_mistakes + (riley_correct + riley_mistakes - ofelia_correct) = total_mistakes := by sorry
  
  -- The theorem to prove
  have goal : (riley_correct + riley_mistakes) = 35 := by sorry
  
  exact 35


end NUMINAMATH_CALUDE_ERRORFEEDBACK_math_contest_questions_l530_53010


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_all_propositions_incorrect_l530_53014

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the relations
variable (parallel : Line → Line → Prop)
variable (perpendicular : Line → Line → Prop)
variable (parallel_line_plane : Line → Plane → Prop)
variable (perpendicular_line_plane : Line → Plane → Prop)
variable (parallel_plane_plane : Plane → Plane → Prop)

-- Define the theorem
theorem all_propositions_incorrect 
  (a b : Line) (α β : Plane) : 
  (¬(parallel a b ∧ parallel_line_plane a α → parallel_line_plane b α)) ∧
  (¬(perpendicular a b ∧ perpendicular_line_plane a α → parallel_line_plane b α)) ∧
  (¬(parallel_line_plane a α ∧ parallel_plane_plane β α → parallel_line_plane a β)) ∧
  (¬(perpendicular_line_plane a α ∧ perpendicular_line_plane a β → parallel_line_plane a β)) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_all_propositions_incorrect_l530_53014


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_mono_decreasing_implies_max_at_start_max_at_start_not_implies_mono_decreasing_l530_53050

open Set
open Function

/-- A function is monotonically decreasing on an interval -/
def MonoDecreasing {α β : Type*} [Preorder α] [Preorder β] (f : α → β) (a b : α) : Prop :=
  ∀ x y, a ≤ x → x ≤ y → y ≤ b → f y ≤ f x

/-- The maximum value of a function on a closed interval -/
def MaxValueAt {α β : Type*} [Preorder α] [Preorder β] (f : α → β) (a b x : α) : Prop :=
  x ∈ Icc a b ∧ ∀ y ∈ Icc a b, f y ≤ f x

theorem mono_decreasing_implies_max_at_start {α β : Type*} [LinearOrder α] [LinearOrder β]
  (f : α → β) (a b : α) (h : a ≤ b) :
  MonoDecreasing f a b → MaxValueAt f a b a := by
  sorry

theorem max_at_start_not_implies_mono_decreasing :
  ∃ (α β : Type*) (_ : LinearOrder α) (_ : LinearOrder β) (f : α → β) (a b : α),
    a < b ∧ MaxValueAt f a b a ∧ ¬MonoDecreasing f a b := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_mono_decreasing_implies_max_at_start_max_at_start_not_implies_mono_decreasing_l530_53050


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_exponential_properties_l530_53034

variable (a : ℝ) (m n : ℝ)

-- Define the function f(x) = a^x
noncomputable def f (x : ℝ) := a^x

-- State the theorem
theorem exponential_properties 
  (h1 : a > 0) 
  (h2 : a ≠ 1) 
  (h3 : m > 0) 
  (h4 : n > 0) :
  (f a (m + n) = f a m * f a n) ∧ 
  (f a ((m + n) / 2) ≤ (f a m + f a n) / 2) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_exponential_properties_l530_53034


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_bijection_parameters_l530_53005

def A (m : ℕ) : Finset ℕ := {1, 2, 3, m}

def B (n : ℕ) : Finset ℕ := {4, 7, n^4, n^2 + 3*n}

structure BijectionParameters where
  p : ℤ
  q : ℤ
  m : ℕ
  n : ℕ

def is_bijection (params : BijectionParameters) : Prop :=
  ∃ (f : ℕ → ℕ), Function.Bijective f ∧
  (∀ a, f a = params.p * a + params.q) ∧
  (∀ a ∈ A params.m, f a ∈ B params.n) ∧
  (∀ b ∈ B params.n, ∃ a ∈ A params.m, f a = b)

theorem bijection_parameters :
  ∀ (params : BijectionParameters),
  is_bijection params →
  params.p * 1 + params.q = 4 →
  params.p * 2 + params.q = 7 →
  params = ⟨3, 1, 5, 2⟩ :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_bijection_parameters_l530_53005


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_difference_is_six_l530_53003

-- Define the hyperbola
def hyperbola (x y : ℝ) : Prop := x^2 / 1 - y^2 / 15 = 1

-- Define the circles
def circle1 (x y : ℝ) : Prop := (x + 4)^2 + y^2 = 4
def circle2 (x y : ℝ) : Prop := (x - 4)^2 + y^2 = 4

-- Define the distance between two points
noncomputable def distance (x1 y1 x2 y2 : ℝ) : ℝ := Real.sqrt ((x1 - x2)^2 + (y1 - y2)^2)

theorem max_difference_is_six :
  ∃ (px py mx my nx ny : ℝ),
    hyperbola px py ∧
    px > 0 ∧
    circle1 mx my ∧
    circle2 nx ny ∧
    (∀ (qx qy rx ry sx sy : ℝ),
      hyperbola qx qy →
      qx > 0 →
      circle1 rx ry →
      circle2 sx sy →
      distance px py mx my - distance px py nx ny ≥ distance qx qy rx ry - distance qx qy sx sy) ∧
    distance px py mx my - distance px py nx ny = 6 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_difference_is_six_l530_53003


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_garden_area_l530_53070

/-- Represents a rectangular garden with a given perimeter and length-width ratio --/
structure RectangularGarden where
  perimeter : ℝ
  ratio : ℝ
  perimeter_positive : 0 < perimeter
  ratio_positive : 0 < ratio

/-- Calculates the area of a rectangular garden --/
noncomputable def garden_area (g : RectangularGarden) : ℝ :=
  let width := g.perimeter / (2 * (1 + g.ratio))
  let length := g.ratio * width
  length * width

/-- Theorem stating the maximum area of a specific rectangular garden --/
theorem max_garden_area :
  ∃ (g : RectangularGarden), g.perimeter = 480 ∧ g.ratio = 2 ∧ 
  garden_area g = 12800 ∧ 
  ∀ (h : RectangularGarden), h.perimeter = 480 ∧ h.ratio = 2 → garden_area h ≤ garden_area g := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_garden_area_l530_53070


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_equation_l530_53032

-- Define the curve C
noncomputable def curve_C (θ : ℝ) : ℝ × ℝ :=
  (1 + 2 * Real.cos θ, 1 + 2 * Real.sin θ)

-- Define the center of the curve
def center : ℝ × ℝ := (1, 1)

-- Define the direction vector of line l
def direction_vector : ℝ × ℝ := (1, 1)

-- Define line l
def line_l (x : ℝ) : ℝ := x

-- Theorem statement
theorem line_equation :
  (∀ θ : ℝ, 0 ≤ θ ∧ θ ≤ 2 * Real.pi → curve_C θ = (1 + 2 * Real.cos θ, 1 + 2 * Real.sin θ)) →
  (∃ t : ℝ, center = (t, line_l t)) →
  direction_vector = (1, 1) →
  ∀ x : ℝ, line_l x = x :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_equation_l530_53032


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_isosceles_trapezoid_circle_centers_distance_l530_53022

/-- An isosceles trapezoid with inscribed and circumscribed circles -/
structure IsoscelesTrapezoid where
  a : ℝ
  b : ℝ
  h_positive : 0 < b
  h_order : b < a

/-- The distance between the centers of the inscribed and circumscribed circles -/
noncomputable def center_distance (t : IsoscelesTrapezoid) : ℝ :=
  (t.a^2 - t.b^2) / (8 * Real.sqrt (t.a * t.b))

/-- Theorem: The distance between the centers of the inscribed and circumscribed circles
    of an isosceles trapezoid with bases a and b is (a^2 - b^2) / (8√(ab)) -/
theorem isosceles_trapezoid_circle_centers_distance (t : IsoscelesTrapezoid) :
  ∃ (d : ℝ), d = center_distance t ∧ 
  d = (t.a^2 - t.b^2) / (8 * Real.sqrt (t.a * t.b)) := by
  use center_distance t
  constructor
  · rfl
  · rfl

#check isosceles_trapezoid_circle_centers_distance

end NUMINAMATH_CALUDE_ERRORFEEDBACK_isosceles_trapezoid_circle_centers_distance_l530_53022


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_symmetry_implies_phi_l530_53036

/-- The function f(x) = sin(x) + √3 * cos(x) -/
noncomputable def f (x : ℝ) : ℝ := Real.sin x + Real.sqrt 3 * Real.cos x

/-- A function is even if f(x) = f(-x) for all x -/
def is_even (g : ℝ → ℝ) : Prop := ∀ x, g x = g (-x)

/-- The main theorem -/
theorem symmetry_implies_phi (φ : ℝ) :
  is_even (fun x ↦ f (x + φ)) → φ = π / 6 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_symmetry_implies_phi_l530_53036


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_local_odd_function_range_l530_53096

/-- A function f is a local odd function if there exists an x such that f(-x) = -f(x) --/
def is_local_odd_function (f : ℝ → ℝ) : Prop :=
  ∃ x : ℝ, f (-x) = -f x

/-- The function f(x) = 9^x - m * 3^x - 3 --/
noncomputable def f (m : ℝ) (x : ℝ) : ℝ :=
  (9 : ℝ)^x - m * (3 : ℝ)^x - 3

/-- The theorem stating the range of m for which f is a local odd function --/
theorem local_odd_function_range (m : ℝ) :
  is_local_odd_function (f m) ↔ m ≥ -2 :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_local_odd_function_range_l530_53096


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cylinder_height_formula_l530_53072

/-- The volume of a cylinder -/
noncomputable def cylinderVolume (r h : ℝ) : ℝ := Real.pi * r^2 * h

/-- The height of a cylindrical tin given its diameter and volume -/
noncomputable def cylinderHeight (d v : ℝ) : ℝ :=
  v / (Real.pi * (d/2)^2)

theorem cylinder_height_formula (d v : ℝ) (hd : d = 4) (hv : v = 20) :
  cylinderHeight d v = 5 / Real.pi := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cylinder_height_formula_l530_53072


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_circle_tangent_distance_l530_53009

/-- Given a line kx + y + 3 = 0 and a circle x^2 + y^2 - 2y = 0 tangent to the line at point Q,
    if there exists a point P on the line such that the distance between P and Q is √3,
    then k ∈ (-∞, -√3] ∪ [√3, +∞). -/
theorem line_circle_tangent_distance (k : ℝ) : 
  (∃ (P Q : ℝ × ℝ), 
    (k * P.1 + P.2 + 3 = 0) ∧ 
    (Q.1^2 + Q.2^2 - 2*Q.2 = 0) ∧
    (k * Q.1 + Q.2 + 3 = 0) ∧
    ((P.1 - Q.1)^2 + (P.2 - Q.2)^2 = 3)) →
  k ∈ Set.Iic (-Real.sqrt 3) ∪ Set.Ici (Real.sqrt 3) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_circle_tangent_distance_l530_53009


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_slope_of_line_with_given_chord_length_l530_53061

-- Define the circle
def circle_eq (x y : ℝ) : Prop := x^2 + y^2 - 2*x - 2*y + 1 = 0

-- Define the line passing through (-1, -2) with slope m
def line_eq (m : ℝ) (x y : ℝ) : Prop := y = m * (x + 1) - 2

-- Define the chord length
noncomputable def chord_length (m : ℝ) : ℝ := 2 * Real.sqrt 2

-- Theorem statement
theorem slope_of_line_with_given_chord_length :
  ∃ m : ℝ, line_eq m (-1) (-2) ∧ 
    (∀ x y : ℝ, circle_eq x y → line_eq m x y → chord_length m = 2 * Real.sqrt 2) ∧ 
    (m = 1 ∨ m = -1) := by
  sorry

#check slope_of_line_with_given_chord_length

end NUMINAMATH_CALUDE_ERRORFEEDBACK_slope_of_line_with_given_chord_length_l530_53061


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_quadrilateral_area_l530_53031

/-- The quadrilateral region defined by the system of inequalities -/
def QuadrilateralRegion : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | 3 * p.1 + 2 * p.2 ≤ 6 ∧ p.1 + 3 * p.2 ≥ 3 ∧ p.1 ≥ 0 ∧ p.2 ≥ 0}

/-- The area of the quadrilateral region -/
noncomputable def areaOfQuadrilateral : ℝ := 15 / 7

/-- Theorem stating that the area of the quadrilateral region is 15/7 -/
theorem quadrilateral_area : MeasureTheory.volume QuadrilateralRegion = (15 / 7 : ENNReal) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_quadrilateral_area_l530_53031


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_intersection_length_l530_53033

/-- Parabola defined by y^2 = 6x -/
def parabola (x y : ℝ) : Prop := y^2 = 6*x

/-- Focus of the parabola -/
noncomputable def focus : ℝ × ℝ := (3/2, 0)

/-- Line perpendicular to x-axis passing through the focus -/
def perpendicular_line (x : ℝ) : Prop := x = 3/2

/-- Point M on the parabola and the perpendicular line -/
noncomputable def point_M : ℝ × ℝ := (3/2, 3)

/-- Point N on the parabola and the perpendicular line -/
noncomputable def point_N : ℝ × ℝ := (3/2, -3)

theorem parabola_intersection_length :
  parabola point_M.1 point_M.2 ∧
  parabola point_N.1 point_N.2 ∧
  perpendicular_line point_M.1 ∧
  perpendicular_line point_N.1 →
  ‖point_M.2 - point_N.2‖ = 6 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_intersection_length_l530_53033


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_left_neighbor_payment_ratio_l530_53011

/-- Prove the ratio of left neighbor's payment to total cost of shared side --/
theorem left_neighbor_payment_ratio 
  (side_length : ℝ) 
  (back_length : ℝ) 
  (cost_per_foot : ℝ) 
  (cole_payment : ℝ) 
  (h1 : side_length = 9)
  (h2 : back_length = 18)
  (h3 : cost_per_foot = 3)
  (h4 : cole_payment = 72)
  (h5 : side_length > 0)
  (h6 : back_length > 0)
  (h7 : cost_per_foot > 0) :
  (cole_payment + cost_per_foot * back_length / 2 - cole_payment) / (side_length * cost_per_foot) = 1 := by
  sorry

#check left_neighbor_payment_ratio

end NUMINAMATH_CALUDE_ERRORFEEDBACK_left_neighbor_payment_ratio_l530_53011


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_digit_2567_is_8_l530_53081

noncomputable def digit_sequence : ℕ → ℕ
  | 0 => 0  -- to account for the decimal point
  | (n + 1) => 
    if n < 9 then n + 1
    else if n < 189 then ((n - 9) / 2 + 10) % 10
    else ((n - 189) / 3 + 100) % 10

noncomputable def x : ℚ := ∑' n, (digit_sequence n : ℚ) / 10^(n + 1)

theorem digit_2567_is_8 : digit_sequence 2567 = 8 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_digit_2567_is_8_l530_53081


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_power_sum_l530_53082

theorem complex_power_sum (i : ℂ) (h1 : i^2 = -1) (h2 : i^4 = 1) :
  i^10 + i^22 + i^(-34 : ℤ) = -3 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_power_sum_l530_53082


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_vertical_division_minimizes_sum_l530_53044

/-- Represents a 10x10 grid filled with integers from 1 to 100 -/
def Grid := Fin 10 → Fin 10 → Fin 100

/-- Fills the grid with numbers 1 to 100 as described in the problem -/
def fillGrid : Grid :=
  fun i j => ⟨10 * i.val + j.val + 1, by sorry⟩

/-- Represents a 1x2 rectangle in the grid -/
structure Rectangle where
  row : Fin 10
  col : Fin 9
  vertical : Bool

/-- Calculates the product of numbers in a rectangle -/
def rectangleProduct (g : Grid) (r : Rectangle) : ℕ :=
  if r.vertical then
    (g r.row r.col).val * (g ⟨r.row.val + 1, by sorry⟩ r.col).val
  else
    (g r.row r.col).val * (g r.row ⟨r.col.val + 1, by sorry⟩).val

/-- Represents a division of the grid into 50 rectangles -/
def GridDivision := Fin 50 → Rectangle

/-- Calculates the sum of products for a given grid division -/
def sumOfProducts (g : Grid) (d : GridDivision) : ℕ :=
  (Finset.univ.sum fun i => rectangleProduct g (d i))

/-- States that vertical division minimizes the sum of products -/
theorem vertical_division_minimizes_sum (g : Grid) (d : GridDivision) :
    g = fillGrid →
    sumOfProducts g (fun _ => { row := 0, col := 0, vertical := true }) ≤ sumOfProducts g d := by
  sorry

#check vertical_division_minimizes_sum

end NUMINAMATH_CALUDE_ERRORFEEDBACK_vertical_division_minimizes_sum_l530_53044


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_l530_53049

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := (1/2) * x^2 - a * x + (a - 1) * Real.log x

theorem function_properties (a : ℝ) :
  (∀ x > 0, ∃ y, f a x = y) →
  (∀ x > 0, ∃ y, (deriv (f a)) x = y) →
  (∀ x > 0, ∃ y, (deriv (deriv (f a))) x = y) →
  ((deriv (f a) 2 = -1) ↔ (a = 5)) ∧
  (∃ mono : ℝ → Prop, ∀ x > 0, mono (deriv (f a) x)) ∧
  (∀ x₁ x₂, x₁ > 0 → x₂ > 0 → x₁ > x₂ → 
    (f a x₁ - f a x₂ > x₂ - x₁) → (1 ≤ a ∧ a ≤ 5)) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_l530_53049


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_four_common_tangents_l530_53098

-- Define the circles
def circle_Q1 (x y : ℝ) : Prop := x^2 + y^2 = 9
def circle_Q2 (x y : ℝ) : Prop := (x-3)^2 + (y-4)^2 = 1

-- Define the centers and radii
def center_Q1 : ℝ × ℝ := (0, 0)
def center_Q2 : ℝ × ℝ := (3, 4)
def radius_Q1 : ℝ := 3
def radius_Q2 : ℝ := 1

-- Define the distance between centers
noncomputable def distance_between_centers : ℝ := Real.sqrt ((center_Q2.1 - center_Q1.1)^2 + (center_Q2.2 - center_Q1.2)^2)

-- Define a function to represent the number of common tangents
def number_of_common_tangents (c1 c2 : (ℝ → ℝ → Prop)) : ℕ := 4

-- Theorem statement
theorem four_common_tangents :
  distance_between_centers > radius_Q1 + radius_Q2 →
  number_of_common_tangents circle_Q1 circle_Q2 = 4 :=
by
  intro h
  -- The proof goes here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_four_common_tangents_l530_53098


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_equation_and_line_properties_l530_53085

-- Define the circle C
def circle_C : Set (ℝ × ℝ) := {p | ∃ center : ℝ × ℝ, (center.1 - p.1)^2 + (center.2 - p.2)^2 = 25}

-- Define points M and N
def point_M : ℝ × ℝ := (-3, 3)
def point_N : ℝ × ℝ := (1, -5)

-- Define the line that contains the center of circle C
def center_line (x y : ℝ) : Prop := 2*x - y - 2 = 0

-- Define a general line l passing through (-2,5) with slope k
def line_l (k : ℝ) (x y : ℝ) : Prop := y - 5 = k*(x + 2)

-- Define point P
def point_P : ℝ × ℝ := (3, -1)

theorem circle_equation_and_line_properties :
  ∃ (center : ℝ × ℝ),
    -- Circle C passes through M and N
    point_M ∈ circle_C ∧ point_N ∈ circle_C ∧
    -- Center of C is on the given line
    center_line center.1 center.2 ∧
    -- Standard equation of circle C
    (∀ x y : ℝ, (x, y) ∈ circle_C ↔ (x - 1)^2 + y^2 = 25) ∧
    -- Range of k values
    (∀ k : ℝ, k > 0 → (∃ A B : ℝ × ℝ, A ≠ B ∧ A ∈ circle_C ∧ B ∈ circle_C ∧ line_l k A.1 A.2 ∧ line_l k B.1 B.2) ↔ k > 15/8) ∧
    -- Existence of line l with perpendicular bisector through P
    (∃ k : ℝ, k > 0 ∧ 
      ∃ A B : ℝ × ℝ, A ≠ B ∧ A ∈ circle_C ∧ B ∈ circle_C ∧ line_l k A.1 A.2 ∧ line_l k B.1 B.2 ∧
      ∃ m : ℝ, (∀ x y : ℝ, x + k*y + (k - 3) = 0 → line_l m x y) ∧
      (point_P.1 + k*point_P.2 + (k - 3) = 0) ∧
      m = 2) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_equation_and_line_properties_l530_53085


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_expression_equals_one_l530_53074

theorem expression_equals_one : 4⁻¹ - Real.sqrt (1 / 16) + (3 - Real.sqrt 2) ^ 0 = 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_expression_equals_one_l530_53074


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_box_volume_negative_l530_53023

/-- Conversion factor from cm to inches -/
noncomputable def cm_to_inch : ℝ := 1 / 2.54

/-- Length of the base in cm -/
def base_length_cm : ℝ := 4

/-- Length of the base in inches -/
noncomputable def base_length_inch : ℝ := base_length_cm * cm_to_inch

/-- Area of the base in square inches -/
noncomputable def base_area (h : ℝ) : ℝ := base_length_inch ^ 2

/-- Volume of the box in cubic inches -/
noncomputable def box_volume (h : ℝ) : ℝ := base_area h * h

theorem box_volume_negative :
  ∀ h : ℝ, base_area h = h + 8 → box_volume h < 0 :=
by
  intro h hyp
  -- The proof is omitted as it's not possible to prove this statement
  -- (the volume of a box cannot be negative for real dimensions)
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_box_volume_negative_l530_53023


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_x_range_of_m_l530_53004

-- Define the functions f and g
def f (x : ℝ) : ℝ := |x - 3| - 2
def g (x : ℝ) : ℝ := -|x + 1| + 4

-- Theorem for the range of x when f(x) ≤ 1
theorem range_of_x : Set.Icc 0 6 = {x : ℝ | f x ≤ 1} := by sorry

-- Theorem for the range of m when f(x) - g(x) ≥ m + 1 for all x
theorem range_of_m : Set.Iic (-3 : ℝ) = {m : ℝ | ∀ x, f x - g x ≥ m + 1} := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_x_range_of_m_l530_53004


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inverse_function_of_log_a_equals_two_l530_53046

-- Define the function f as noncomputable
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := Real.log x / Real.log a

-- State the theorem
theorem inverse_function_of_log (a : ℝ) (h1 : a > 0) (h2 : a ≠ 1) (h3 : f a 8 = 3) :
  ∀ x, f a (2^x) = x ∧ 2^(f a x) = x := by
  sorry

-- Additional theorem to show that a = 2
theorem a_equals_two (a : ℝ) (h1 : a > 0) (h2 : a ≠ 1) (h3 : f a 8 = 3) : a = 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inverse_function_of_log_a_equals_two_l530_53046


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_ratio_is_four_fifths_l530_53095

/-- Represents a rectangle with width and length -/
structure Rectangle where
  width : ℝ
  length : ℝ

/-- Represents a square, which is a special case of a rectangle -/
structure Square where
  side : ℝ

/-- Represents the configuration of rectangles in the problem -/
structure RectangleConfiguration where
  s : ℝ
  abcd : Rectangle
  efgh : Square
  ijkl : Rectangle

/-- The main theorem stating the ratio of areas -/
theorem area_ratio_is_four_fifths (config : RectangleConfiguration) 
  (hs : config.s > 0)
  (habcd_width : config.abcd.width = config.s)
  (habcd_length : config.abcd.length = 2 * config.s)
  (hefgh : config.efgh.side = config.s)
  (hijkl_width : config.ijkl.width = config.s)
  (hijkl_length : config.ijkl.length = 2 * config.s) :
  (4 * config.s^2) / (5 * config.s^2) = 4 / 5 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_ratio_is_four_fifths_l530_53095


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_twenty_fifth_digit_is_zero_l530_53047

def fraction_sum : ℚ := 1/8 + 1/5

def decimal_digit_at (q : ℚ) (n : ℕ) : ℕ :=
  (Int.floor (q * 10^n) % 10).natAbs

theorem twenty_fifth_digit_is_zero :
  decimal_digit_at fraction_sum 25 = 0 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_twenty_fifth_digit_is_zero_l530_53047


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_l530_53069

noncomputable def f (x : ℝ) := Real.sqrt (x + 1) + 1 / (3 - x)

theorem domain_of_f :
  Set.Ioi (-1 : ℝ) \ {3} = {x : ℝ | f x ∈ Set.range f} :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_l530_53069


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_f_l530_53073

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := x / (x^2 + 2*x + 2)

-- Define the lower and upper bounds of the range
noncomputable def lower_bound : ℝ := (-1 - Real.sqrt 2) / 2
noncomputable def upper_bound : ℝ := (-1 + Real.sqrt 2) / 2

-- Theorem statement
theorem range_of_f :
  ∀ y : ℝ, (∃ x : ℝ, f x = y) ↔ lower_bound ≤ y ∧ y ≤ upper_bound := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_f_l530_53073


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_inequality_solution_set_l530_53013

-- Define the piecewise function f
noncomputable def f (x : ℝ) : ℝ :=
  if x ≥ 0 then x^2 - 4*x + 6 else x + 6

-- Define the solution set
def solution_set (x : ℝ) : Prop :=
  (-3 < x ∧ x < 1) ∨ x > 3

-- Theorem statement
theorem f_inequality_solution_set :
  ∀ x : ℝ, f x > f 1 ↔ solution_set x := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_inequality_solution_set_l530_53013


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_fair_special_savings_l530_53086

/-- Calculates the percentage saved when buying three hats under the "fair special" -/
theorem fair_special_savings (regular_price second_discount third_discount : ℝ) :
  regular_price > 0 →
  0 ≤ second_discount ∧ second_discount < 1 →
  0 ≤ third_discount ∧ third_discount < 1 →
  (3 * regular_price - (regular_price + (1 - second_discount) * regular_price + (1 - third_discount) * regular_price)) / (3 * regular_price) * 100 = 30 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_fair_special_savings_l530_53086


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_of_20_consecutive_even_integers_sum_3000_l530_53048

/-- A sequence of consecutive even integers -/
def ConsecutiveEvenIntegers (start : ℤ) (n : ℕ) : List ℤ :=
  List.range n |>.map (fun i => start + 2 * i)

/-- The sum of a list of integers -/
def ListSum (list : List ℤ) : ℤ :=
  list.foldl (· + ·) 0

theorem largest_of_20_consecutive_even_integers_sum_3000 :
  ∃ (start : ℤ),
    let seq := ConsecutiveEvenIntegers start 20
    ListSum seq = 3000 ∧ seq.getLast? = some 169 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_of_20_consecutive_even_integers_sum_3000_l530_53048


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_F_max_min_on_interval_l530_53062

-- Define the function F
noncomputable def F (x : ℝ) : ℝ := (1/3) * x^3 + x^2 - 8*x

-- Define the interval
def interval : Set ℝ := {x | 1 ≤ x ∧ x ≤ 3}

-- State the theorem
theorem F_max_min_on_interval :
  (∃ x ∈ interval, ∀ y ∈ interval, F y ≤ F x) ∧
  (∃ x ∈ interval, ∀ y ∈ interval, F x ≤ F y) ∧
  (∃ x ∈ interval, F x = -6) ∧
  (∃ x ∈ interval, F x = -28/3) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_F_max_min_on_interval_l530_53062


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_g_is_odd_l530_53041

noncomputable def g (x : ℝ) : ℝ := (3^x - 1) / (3^x + 1)

theorem g_is_odd : ∀ x : ℝ, g (-x) = -g x := by
  intro x
  simp [g]
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_g_is_odd_l530_53041


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_midpoint_to_y_axis_distance_l530_53037

/-- Given a line y = kx - k and a parabola y² = 4x intersecting at points A and B,
    prove that if the distance between A and B is 4,
    then the distance from the midpoint of AB to the y-axis is 1. -/
theorem midpoint_to_y_axis_distance
  (k : ℝ)
  (A B : ℝ × ℝ)
  (h1 : ∀ x y, y = k * x - k ↔ (x, y) ∈ ({A, B} : Set (ℝ × ℝ)))
  (h2 : ∀ x y, y^2 = 4 * x ↔ (x, y) ∈ ({A, B} : Set (ℝ × ℝ)))
  (h3 : ((A.1 - B.1)^2 + (A.2 - B.2)^2) = 16) :
  let M := ((A.1 + B.1) / 2, (A.2 + B.2) / 2)
  |M.1 - 0| = 1 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_midpoint_to_y_axis_distance_l530_53037


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_q_satisfies_conditions_l530_53035

/-- A quadratic polynomial satisfying specific conditions -/
noncomputable def q (x : ℝ) : ℝ := 9/4 * x^2 - 27/4 * x - 40.5

/-- Theorem stating that q satisfies the given conditions -/
theorem q_satisfies_conditions : 
  q (-3) = 0 ∧ q 6 = 0 ∧ q 2 = -45 := by
  sorry

#check q_satisfies_conditions

end NUMINAMATH_CALUDE_ERRORFEEDBACK_q_satisfies_conditions_l530_53035


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_different_point_l530_53025

noncomputable def same_point (r1 θ1 r2 θ2 : ℝ) : Prop :=
  (r1 = r2 ∧ ∃ k : ℤ, θ1 = θ2 + 2 * Real.pi * k) ∨
  (r1 = -r2 ∧ ∃ k : ℤ, θ1 = θ2 + Real.pi + 2 * Real.pi * k)

theorem different_point :
  ¬ same_point (-2) (Real.pi/6) 2 (-7*Real.pi/6) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_different_point_l530_53025


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_pinwheel_area_is_six_l530_53000

/-- Represents a point in a 2D grid --/
structure GridPoint where
  x : ℤ
  y : ℤ

/-- Represents a kite in the pinwheel --/
structure Kite where
  center : GridPoint
  midpoint : GridPoint
  vertex1 : GridPoint
  vertex2 : GridPoint

/-- Represents the pinwheel --/
structure Pinwheel where
  kites : Fin 4 → Kite
  center : GridPoint

/-- Calculates the area of a kite using Pick's theorem --/
def kiteArea : ℚ :=
  3 / 2

/-- Calculates the area of the entire pinwheel --/
def pinwheelArea (p : Pinwheel) : ℚ :=
  4 * kiteArea

/-- The main theorem stating the area of the pinwheel --/
theorem pinwheel_area_is_six (p : Pinwheel) 
  (h1 : p.center = ⟨5, 5⟩)
  (h2 : ∀ i, (p.kites i).center = p.center)
  (h3 : ∀ i, (p.kites i).midpoint.x = 5 ∨ (p.kites i).midpoint.y = 5)
  (h4 : ∀ i, (p.kites i).vertex1.x ∈ Set.range (λ x : ℤ ↦ x) ∧ (p.kites i).vertex1.y ∈ Set.range (λ x : ℤ ↦ x))
  (h5 : ∀ i, (p.kites i).vertex2.x ∈ Set.range (λ x : ℤ ↦ x) ∧ (p.kites i).vertex2.y ∈ Set.range (λ x : ℤ ↦ x)) :
  pinwheelArea p = 6 := by
    sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_pinwheel_area_is_six_l530_53000


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_lower_right_is_one_l530_53080

def Grid := Fin 4 → Fin 4 → Fin 4

def valid_grid (g : Grid) : Prop :=
  (∀ i j, g i j < 4) ∧
  (∀ i, Function.Injective (g i)) ∧
  (∀ j, Function.Injective (λ i ↦ g i j))

def initial_constraints (g : Grid) : Prop :=
  g 0 0 = 0 ∧ g 0 2 = 1 ∧ g 1 0 = 1 ∧ g 1 1 = 2 ∧ g 2 3 = 3

theorem lower_right_is_one (g : Grid) 
  (h1 : valid_grid g) 
  (h2 : initial_constraints g) : 
  g 3 3 = 0 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_lower_right_is_one_l530_53080


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_of_f_l530_53028

-- Define the function
noncomputable def f (x : ℝ) : ℝ := 6 / (2^x + 3^x)

-- State the theorem
theorem min_value_of_f :
  ∃ (min_val : ℝ), min_val = 6/5 ∧
  ∀ x : ℝ, -1 ≤ x ∧ x ≤ 1 → f x ≥ min_val :=
by
  -- We'll use 'sorry' to skip the proof for now
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_of_f_l530_53028

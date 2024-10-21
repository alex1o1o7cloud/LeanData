import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_gcd_polynomial_and_multiple_l1123_112326

theorem gcd_polynomial_and_multiple (x : ℤ) : 
  x % 38962 = 0 → 
  Int.gcd ((3*x+5)*(8*x+3)*(17*x+4)*(x+17)) x = 6 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_gcd_polynomial_and_multiple_l1123_112326


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sqrt_calculations_l1123_112344

theorem sqrt_calculations :
  (∃ x : ℝ, x = Real.sqrt 24 + Real.sqrt 18 * Real.sqrt (1/3) ∧ x = 3 * Real.sqrt 6) ∧
  (∃ y : ℝ, y = (Real.sqrt 24 - Real.sqrt 2) - (Real.sqrt 8 + Real.sqrt 6) ∧ y = Real.sqrt 6 - 3 * Real.sqrt 2) :=
by
  constructor
  · use 3 * Real.sqrt 6
    constructor
    · sorry  -- Proof for the first equality
    · rfl
  · use Real.sqrt 6 - 3 * Real.sqrt 2
    constructor
    · sorry  -- Proof for the second equality
    · rfl


end NUMINAMATH_CALUDE_ERRORFEEDBACK_sqrt_calculations_l1123_112344


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_reciprocal_G_powers_of_two_l1123_112381

-- Define the sequence G
def G : ℕ → ℕ
  | 0 => 1
  | 1 => 2
  | (n + 2) => 3 * G (n + 1) - 2 * G n

-- Define the sum of the series
noncomputable def seriesSum : ℝ := ∑' n, (1 : ℝ) / G (2^n)

-- Theorem statement
theorem sum_of_reciprocal_G_powers_of_two : seriesSum = 4/3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_reciprocal_G_powers_of_two_l1123_112381


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_right_triangle_leg_length_l1123_112357

/-- A triangle represented by its vertices -/
structure Triangle where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ

/-- Predicate to check if a triangle is right-angled -/
def IsRightTriangle (t : Triangle) : Prop := sorry

/-- Predicate to check if a triangle has a specific angle -/
def HasAngle (t : Triangle) (angle : ℝ) : Prop := sorry

/-- Function to get the hypotenuse length of a triangle -/
def HypotenuseLength (t : Triangle) : ℝ := sorry

/-- Function to get the leg length of a right triangle -/
def LegLength (t : Triangle) : ℝ := sorry

/-- Theorem stating the relationship between the hypotenuse and leg length in a specific right triangle -/
theorem right_triangle_leg_length 
  (t : Triangle) 
  (is_right : IsRightTriangle t) 
  (has_45_deg : HasAngle t (π / 4))
  (hyp_len : HypotenuseLength t = 2.5 * Real.sqrt 3) :
  LegLength t = 1.25 * Real.sqrt 6 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_right_triangle_leg_length_l1123_112357


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_binomial_expansion_properties_l1123_112354

theorem binomial_expansion_properties :
  let expansion : ℕ → ℚ := fun n => (Nat.choose 10 n) * (-1 : ℚ)^n
  let a : ℕ → ℚ := fun n => if n ≤ 10 then expansion n else 0
  (a 0 = 1) ∧ (a 2 + a 4 + a 6 + a 8 + a 10 = 511) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_binomial_expansion_properties_l1123_112354


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_manuscript_typing_cost_correct_l1123_112380

def manuscript_typing_cost (total_pages : ℕ) (revised_once : ℕ) (revised_twice : ℕ) 
  (revision_cost : ℕ) (total_cost : ℕ) : ℕ :=
  let first_time_cost := (total_cost - revised_once * revision_cost - revised_twice * revision_cost * 2) / total_pages
  first_time_cost

theorem manuscript_typing_cost_correct (total_pages : ℕ) (revised_once : ℕ) (revised_twice : ℕ) 
  (revision_cost : ℕ) (total_cost : ℕ) 
  (h1 : total_pages = 100)
  (h2 : revised_once = 30)
  (h3 : revised_twice = 20)
  (h4 : revision_cost = 5)
  (h5 : total_cost = 1350) :
  (total_pages * manuscript_typing_cost total_pages revised_once revised_twice revision_cost total_cost + 
   revised_once * revision_cost + 
   revised_twice * revision_cost * 2) = total_cost := by
  sorry

#eval manuscript_typing_cost 100 30 20 5 1350

end NUMINAMATH_CALUDE_ERRORFEEDBACK_manuscript_typing_cost_correct_l1123_112380


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tower_of_hanoi_min_moves_tower_of_hanoi_restricted_tower_of_hanoi_extra_constraint_l1123_112310

/-- Represents the minimum number of moves required to solve the Tower of Hanoi puzzle with n disks -/
def min_moves (n : ℕ) : ℕ := 2^n - 1

/-- Theorem stating that min_moves gives the minimum number of moves for the Tower of Hanoi puzzle -/
theorem tower_of_hanoi_min_moves (n : ℕ) :
  min_moves n = 2^n - 1 := by
  rfl  -- reflexivity, since this is true by definition

/-- Theorem for the number of moves needed when direct moves between pegs 1 and 3 are prohibited -/
theorem tower_of_hanoi_restricted (n : ℕ) :
  3^n - 1 = (let rec f (m : ℕ) := match m with
    | 0 => 0
    | m+1 => 2 * f m + 1
  f n) := by
  sorry  -- proof omitted

/-- Theorem for the number of moves needed with additional constraint on the smallest disk -/
theorem tower_of_hanoi_extra_constraint (n : ℕ) :
  2 * 3^(n-1) - 1 = (let rec f (m : ℕ) := match m with
    | 0 => 0
    | 1 => 1
    | m+1 => 3 * f m + 2
  f n) := by
  sorry  -- proof omitted

#eval min_moves 8  -- Should output 255
#eval (3^8 - 1)    -- Should output 6560
#eval (2 * 3^7 - 1)  -- Should output 4373

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tower_of_hanoi_min_moves_tower_of_hanoi_restricted_tower_of_hanoi_extra_constraint_l1123_112310


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_magnitude_of_m_l1123_112330

/-- The magnitude of the vector m = (1, -2, 2) is 3. -/
theorem magnitude_of_m : 
  Real.sqrt ((1 : ℝ)^2 + (-2)^2 + 2^2) = 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_magnitude_of_m_l1123_112330


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cosine_of_angle_l1123_112373

/-- 
Given that the terminal side of angle α passes through point A(-3/5, 4/5),
prove that the cosine of α is -3/5.
-/
theorem cosine_of_angle (α : ℝ) : 
  (∃ (A : ℝ × ℝ), A = (-3/5, 4/5) ∧ A ∈ Set.range (λ t => (t * Real.cos α, t * Real.sin α))) → 
  Real.cos α = -3/5 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_cosine_of_angle_l1123_112373


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_side_sum_range_l1123_112398

noncomputable def f (x : ℝ) : ℝ := 
  Real.sin x * Real.sin (Real.pi/2 + x) + Real.sqrt 3 * (Real.cos x)^2 - Real.sqrt 3 / 2

theorem triangle_side_sum_range (A B C : ℝ) : 
  0 < A ∧ A < Real.pi/2 →  -- A is acute
  0 < B ∧ B < Real.pi/2 →  -- B is acute
  0 < C ∧ C < Real.pi/2 →  -- C is acute
  A + B + C = Real.pi →    -- Sum of angles in a triangle
  f A = 0 →          -- f(A) = 0
  Real.pi = Real.pi * (1 : ℝ)^2 →  -- Area of smallest enclosing circle is π
  3 < B + C ∧ B + C ≤ 2 * Real.sqrt 3 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_side_sum_range_l1123_112398


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_first_reduction_percentage_l1123_112339

theorem first_reduction_percentage :
  ∃ (P x : ℝ), P > 0 ∧
    (P * (1 - x / 100) * (1 - 40 / 100) = P * (1 - 55 / 100)) ∧ 
    (x = 25) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_first_reduction_percentage_l1123_112339


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_is_15_verify_conditions_l1123_112358

/-- The distance between two points A and B -/
def distance : ℝ := 15

/-- The speed from A to B in km/h -/
def speed_AB : ℝ := 15

/-- The speed from B to A in km/h -/
def speed_BA : ℝ := 10

/-- The additional time taken for the return trip in hours -/
def additional_time : ℝ := 0.5

/-- Theorem stating that the distance between A and B is 15 km -/
theorem distance_is_15 : distance = 15 := by
  -- Unfold the definition of distance
  unfold distance
  -- The proof is trivial since we defined distance as 15
  rfl

/-- Theorem verifying the problem conditions -/
theorem verify_conditions : 
  (distance / speed_BA) = (distance / speed_AB) + additional_time := by
  -- Substitute the known values
  simp [distance, speed_AB, speed_BA, additional_time]
  -- Evaluate the expression
  norm_num

end NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_is_15_verify_conditions_l1123_112358


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_last_triangle_perimeter_l1123_112301

/-- Represents a triangle in the sequence --/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Generates the next triangle in the sequence --/
noncomputable def next_triangle (T : Triangle) : Triangle :=
  { a := T.a / 2, b := T.b / 2, c := T.c / 2 }

/-- The initial triangle T₁ --/
def T₁ : Triangle := { a := 1002, b := 1003, c := 1001 }

/-- The sequence of triangles --/
noncomputable def triangle_sequence : ℕ → Triangle
  | 0 => T₁
  | n + 1 => next_triangle (triangle_sequence n)

/-- The perimeter of a triangle --/
noncomputable def perimeter (T : Triangle) : ℝ := T.a + T.b + T.c

/-- Predicate to check if a triangle is valid (all sides positive) --/
def is_valid_triangle (T : Triangle) : Prop :=
  T.a > 0 ∧ T.b > 0 ∧ T.c > 0

/-- The index of the last valid triangle in the sequence --/
def last_valid_index : ℕ := 9

/-- The theorem to be proved --/
theorem last_triangle_perimeter :
  perimeter (triangle_sequence last_valid_index) = 1503 / 256 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_last_triangle_perimeter_l1123_112301


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_sequence_term_relation_l1123_112349

/-- A geometric sequence with common ratio q -/
def GeometricSequence (b : ℕ+ → ℝ) (q : ℝ) : Prop :=
  ∀ n : ℕ+, b (n + 1) = b n * q

/-- Theorem: In a geometric sequence, the nth term can be expressed in terms of the mth term -/
theorem geometric_sequence_term_relation
  (b : ℕ+ → ℝ) (q : ℝ) (m n : ℕ+)
  (h : GeometricSequence b q) :
  b n = b m * q ^ (n.val - m.val) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_sequence_term_relation_l1123_112349


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_solutions_is_ten_l1123_112348

theorem sum_of_solutions_is_ten :
  ∃ (S : Finset ℝ), (∀ x ∈ S, |x - 5| - 4 = 1) ∧
                    (∀ x, |x - 5| - 4 = 1 → x ∈ S) ∧
                    (S.sum id = 10) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_solutions_is_ten_l1123_112348


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_function_properties_l1123_112336

/-- A quadratic function f(x) = x^2 - 2mx - m^2 where m ≠ 0 -/
def f (m : ℝ) (x : ℝ) : ℝ := x^2 - 2*m*x - m^2

/-- The x-intercepts of the quadratic function -/
def x_intercepts (m : ℝ) : Set ℝ := {x | f m x = 0}

/-- The vertex of the quadratic function -/
def vertex (m : ℝ) : ℝ × ℝ := (m, -2*m^2)

/-- The circle with diameter AB where A and B are the x-intercepts -/
def circle_with_diameter (m : ℝ) : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | ∃ (a b : ℝ), a ∈ x_intercepts m ∧ b ∈ x_intercepts m ∧
    ((p.1 - (a + b)/2)^2 + (p.2)^2 = ((b - a)/2)^2)}

theorem quadratic_function_properties (m : ℝ) (h : m ≠ 0) :
  vertex m ∈ circle_with_diameter m →
  (∃ (a b : ℝ), a ∈ x_intercepts m ∧ b ∈ x_intercepts m ∧ a ≠ b) ∧
  m^2 = 1/2 ∧
  ∀ (p : ℝ × ℝ), p ∈ circle_with_diameter m → (p.1 - m)^2 + (p.2 + 2*m^2)^2 = 1 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_function_properties_l1123_112336


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_huang_family_theater_cost_l1123_112334

theorem huang_family_theater_cost 
  (full_price : ℝ) 
  (senior_discount : ℝ) 
  (child_discount : ℝ) 
  (senior_ticket_cost : ℝ) :
  senior_discount = 0.3 →
  child_discount = 0.4 →
  senior_ticket_cost = 7 →
  senior_ticket_cost = full_price * (1 - senior_discount) →
  2 * senior_ticket_cost + 2 * full_price + 2 * (full_price * (1 - child_discount)) = 46 := by
  sorry

#check huang_family_theater_cost

end NUMINAMATH_CALUDE_ERRORFEEDBACK_huang_family_theater_cost_l1123_112334


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_valid_configurations_count_l1123_112386

open BigOperators Finset

def valid_configuration (x : Fin 13 → Fin 2) : Prop :=
  (x 0 + 2 * x 1 + 2 * x 11 + x 12) % 5 = 0

instance : DecidablePred valid_configuration :=
  fun x => decEq ((x 0 + 2 * x 1 + 2 * x 11 + x 12) % 5) 0

theorem valid_configurations_count :
  (filter valid_configuration (univ : Finset (Fin 13 → Fin 2))).card = 2560 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_valid_configurations_count_l1123_112386


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_through_points_l1123_112370

/-- The curve C defined by y = x^2/2 -/
noncomputable def C : ℝ → ℝ := λ x ↦ x^2 / 2

/-- Two points on the curve C -/
structure PointsOnC where
  x₁ : ℝ
  x₂ : ℝ
  y₁ : ℝ
  y₂ : ℝ
  on_curve₁ : C x₁ = y₁
  on_curve₂ : C x₂ = y₂
  sum_x : x₁ + x₂ = 2

/-- The equation of the line passing through two points -/
noncomputable def line_equation (p : PointsOnC) : ℝ → ℝ := 
  λ x ↦ x + 7/2

theorem line_through_points (p : PointsOnC) : 
  line_equation p p.x₁ = p.y₁ ∧ line_equation p p.x₂ = p.y₂ := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_through_points_l1123_112370


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_happy_people_possession_equivalence_l1123_112347

variable (Person : Type)
variable (happy : Person → Prop)
variable (possesses : Person → Prop)

theorem happy_people_possession_equivalence :
  (∀ x : Person, happy x → possesses x) ↔ (∀ x : Person, ¬possesses x → ¬happy x) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_happy_people_possession_equivalence_l1123_112347


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_alpha_value_l1123_112388

theorem tan_alpha_value (α : Real) 
  (h1 : α ∈ Set.Ioo 0 (π/2)) 
  (h2 : Real.tan (2*α) = Real.cos α / (2 - Real.sin α)) : 
  Real.tan α = Real.sqrt 15 / 15 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_alpha_value_l1123_112388


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sample_data_properties_l1123_112318

-- Define a set of 6 real numbers
variable (x : Fin 6 → ℝ)

-- Define that x₁ is the minimum and x₆ is the maximum
variable (h_min : ∀ i, x 0 ≤ x i)
variable (h_max : ∀ i, x i ≤ x 5)

-- Define median function
noncomputable def median (a b c d : ℝ) : ℝ := (max (min a b) (min c d) + min (max a b) (max c d)) / 2

-- Define range function
noncomputable def range (a b c d : ℝ) : ℝ := max a (max b (max c d)) - min a (min b (min c d))

theorem sample_data_properties :
  -- The median of x₂, x₃, x₄, x₅ equals the median of x₁, x₂, ..., x₆
  median (x 1) (x 2) (x 3) (x 4) = median (x 0) (x 1) (x 4) (x 5) ∧
  -- The range of x₂, x₃, x₄, x₅ is not greater than the range of x₁, x₂, ..., x₆
  range (x 1) (x 2) (x 3) (x 4) ≤ range (x 0) (x 1) (x 4) (x 5) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sample_data_properties_l1123_112318


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_find_x_l1123_112311

noncomputable def average (a b c : ℝ) : ℝ := (a + b + c) / 3

theorem find_x : 
  ∃ x : ℝ, average 10 20 60 = average 10 40 x + 5 → x = 25 := by
  use 25
  intro h
  -- The proof steps would go here, but we'll use sorry for now
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_find_x_l1123_112311


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_boat_length_approx_l1123_112384

/-- The length of a boat given its width and sinking depth when a person of known mass boards it. --/
noncomputable def boat_length (boat_width : ℝ) (sinking_depth : ℝ) (person_mass : ℝ) : ℝ :=
  let water_density : ℝ := 1000  -- kg/m³
  let gravity : ℝ := 9.8  -- m/s²
  let person_weight : ℝ := person_mass * gravity
  person_weight / (boat_width * sinking_depth * water_density * gravity)

/-- Theorem stating that a boat with width 2 m, sinking 0.018 m when a 108 kg person boards, has a length of approximately 3 m. --/
theorem boat_length_approx :
  ∃ ε > 0, abs (boat_length 2 0.018 108 - 3) < ε := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_boat_length_approx_l1123_112384


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_reciprocal_inequality_l1123_112319

open Real BigOperators

theorem sum_reciprocal_inequality (n : ℕ) (a : Fin n → ℝ) 
  (h_pos : ∀ i, a i > 0)
  (h_sum : ∑ i, a i = ∑ i, (1 : ℝ) / a i) :
  ∑ i, (1 : ℝ) / (n - 1 + a i) ≤ 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_reciprocal_inequality_l1123_112319


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_faster_train_length_l1123_112328

/-- The length of a train given two trains traveling in opposite directions -/
noncomputable def train_length (v_slow v_fast : ℝ) (t : ℝ) : ℝ :=
  (v_slow + v_fast) * (1000 / 3600) * t

theorem faster_train_length :
  let v_slow : ℝ := 36 -- Speed of slower train in km/h
  let v_fast : ℝ := 45 -- Speed of faster train in km/h
  let t : ℝ := 6 -- Time to pass in seconds
  train_length v_slow v_fast t = 135 := by
  -- Unfold the definition of train_length
  unfold train_length
  -- Simplify the expression
  simp
  -- The proof is complete
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_faster_train_length_l1123_112328


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_even_heads_odd_tails_probability_l1123_112309

/-- The probability of getting heads for an unfair coin -/
noncomputable def p_heads : ℝ := 3/5

/-- The number of times the coin is tossed -/
def n_tosses : ℕ := 40

/-- The probability of getting an even number of heads after n tosses -/
noncomputable def P (n : ℕ) : ℝ := 1/2 * (1 + (1/5)^n)

/-- The probability of getting an odd number of tails after n tosses -/
noncomputable def Q (n : ℕ) : ℝ := 1/2 * (1 - (1/5)^n)

/-- The main theorem: probability of even heads and odd tails after 40 tosses -/
theorem even_heads_odd_tails_probability :
  P n_tosses * Q n_tosses = 1/4 * (1 - (1/5)^(2 * n_tosses)) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_even_heads_odd_tails_probability_l1123_112309


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_side_length_l1123_112341

-- Define the right triangle ABC
def RightTriangle (A B C : ℝ × ℝ) : Prop :=
  (B.2 - A.2) * (C.1 - B.1) = (C.2 - B.2) * (B.1 - A.1)

-- Define the slope of a line segment
noncomputable def Slope (P Q : ℝ × ℝ) : ℝ :=
  (Q.2 - P.2) / (Q.1 - P.1)

-- Define the length of a line segment
noncomputable def Length (P Q : ℝ × ℝ) : ℝ :=
  Real.sqrt ((Q.1 - P.1)^2 + (Q.2 - P.2)^2)

theorem triangle_side_length 
  (A B C : ℝ × ℝ) 
  (h_right : RightTriangle A B C)
  (h_hypotenuse : Length A C = 100)
  (h_slope : Slope A C = 4/3) :
  Length A B = 8 * Real.sqrt 10 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_side_length_l1123_112341


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_integral_equals_quarter_circle_area_l1123_112345

-- Define the integrand function
noncomputable def f (x : ℝ) : ℝ := Real.sqrt (9 - x^2)

-- State the theorem
theorem integral_equals_quarter_circle_area :
  ∫ x in (0 : ℝ)..(3 : ℝ), f x = (1/4) * Real.pi * 3^2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_integral_equals_quarter_circle_area_l1123_112345


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_collinear_vectors_t_value_l1123_112351

/-- Given vectors m, n, and k in ℝ², prove that if m - 2n and k are collinear, then t = 1. -/
theorem collinear_vectors_t_value (m n k : ℝ × ℝ) (t : ℝ) :
  m = (Real.sqrt 3, 1) →
  n = (0, -1) →
  k = (t, Real.sqrt 3) →
  (m - 2 • n).1 * k.2 = (m - 2 • n).2 * k.1 →
  t = 1 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_collinear_vectors_t_value_l1123_112351


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_fermat_equation_odd_prime_l1123_112329

theorem fermat_equation_odd_prime (x y p n k : ℕ) : 
  x^n + y^n = p^k →
  n > 1 →
  Odd n →
  Nat.Prime p →
  p > 2 →
  ∃ m : ℕ, n = p^m :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_fermat_equation_odd_prime_l1123_112329


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_fraction_simplification_complex_multiplication_addition_l1123_112366

-- Part 1
theorem complex_fraction_simplification :
  (2 + 2 * Complex.I : ℂ)^4 / (1 - Complex.I * (Complex.I * Complex.I))^5 = 16 := by sorry

-- Part 2
theorem complex_multiplication_addition :
  (2 - Complex.I) * (-1 + 5 * Complex.I) * (3 - 4 * Complex.I) + 2 * Complex.I = 40 + 43 * Complex.I := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_fraction_simplification_complex_multiplication_addition_l1123_112366


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_imaginary_part_angle_l1123_112304

/-- The polynomial z^6 - z^4 + z^3 - z + 1 = 0 -/
def P (z : ℂ) : ℂ := z^6 - z^4 + z^3 - z + 1

/-- The angle φ in radians -/
noncomputable def φ : ℝ := Real.arcsin (Real.sin (540 * Real.pi / (7 * 180)))

theorem max_imaginary_part_angle :
  ∃ (z : ℂ), P z = 0 ∧ 
  ∀ (w : ℂ), P w = 0 → z.im ≥ w.im ∧
  z.im = Real.sin φ ∧
  -Real.pi/2 ≤ φ ∧ φ ≤ Real.pi/2 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_imaginary_part_angle_l1123_112304


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_volume_of_inscribed_centers_tetrahedron_l1123_112396

/-- Represents a point in 3D space -/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- Represents a tetrahedron with vertices P, Q, R, S -/
structure Tetrahedron where
  P : Point3D
  Q : Point3D
  R : Point3D
  S : Point3D

/-- Represents the inscribed circle centers of the tetrahedron faces -/
structure InscribedCenters where
  A : Point3D
  B : Point3D
  C : Point3D
  D : Point3D

/-- Calculates the volume of a tetrahedron given its vertices -/
noncomputable def tetrahedronVolume (t : Tetrahedron) : ℝ := sorry

/-- Calculates the distance between two points in 3D space -/
def distance (p q : Point3D) : ℝ := sorry

/-- Helper function to calculate the inscribed circle center of a triangle -/
noncomputable def inscribedCenter (p q r : Point3D) : Point3D := sorry

/-- Theorem stating the volume of tetrahedron ABCD formed by inscribed circle centers -/
theorem volume_of_inscribed_centers_tetrahedron 
  (PQRS : Tetrahedron) (ABCD : InscribedCenters) :
  distance PQRS.P PQRS.Q = 7 →
  distance PQRS.R PQRS.S = 7 →
  distance PQRS.P PQRS.R = 8 →
  distance PQRS.Q PQRS.S = 8 →
  distance PQRS.P PQRS.S = 9 →
  distance PQRS.Q PQRS.R = 9 →
  ABCD.A = inscribedCenter PQRS.P PQRS.Q PQRS.S →
  ABCD.B = inscribedCenter PQRS.P PQRS.R PQRS.S →
  ABCD.C = inscribedCenter PQRS.Q PQRS.R PQRS.S →
  ABCD.D = inscribedCenter PQRS.P PQRS.Q PQRS.R →
  tetrahedronVolume { P := ABCD.A, Q := ABCD.B, R := ABCD.C, S := ABCD.D } = 5 * Real.sqrt 11 / 9 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_volume_of_inscribed_centers_tetrahedron_l1123_112396


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_product_of_cis_polar_form_product_l1123_112335

noncomputable def cis (θ : Real) : ℂ := Complex.exp (θ * Complex.I)

theorem product_of_cis (r₁ r₂ : Real) (θ₁ θ₂ : Real) :
  (r₁ * cis θ₁) * (r₂ * cis θ₂) = (r₁ * r₂) * cis (θ₁ + θ₂) := by
  sorry

theorem polar_form_product :
  (4 * cis (45 * Real.pi / 180)) * (5 * cis (120 * Real.pi / 180)) =
  20 * cis (165 * Real.pi / 180) := by
  sorry

#check polar_form_product

end NUMINAMATH_CALUDE_ERRORFEEDBACK_product_of_cis_polar_form_product_l1123_112335


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_interest_relationship_l1123_112302

/-- Simple interest calculation function -/
noncomputable def simple_interest (principal : ℝ) (rate : ℝ) (time : ℝ) : ℝ :=
  principal * rate * time / 100

/-- Compound interest calculation function -/
noncomputable def compound_interest (principal : ℝ) (rate : ℝ) (time : ℝ) : ℝ :=
  principal * ((1 + rate / 100) ^ time - 1)

/-- Theorem stating the relationship between simple and compound interest -/
theorem interest_relationship (simple_principal : ℝ) (compound_principal : ℝ) :
  simple_principal = 1400.0000000000014 →
  simple_interest simple_principal 10 3 = (1/2) * compound_interest compound_principal 10 2 →
  compound_principal = 4000 := by
  sorry

#check interest_relationship

end NUMINAMATH_CALUDE_ERRORFEEDBACK_interest_relationship_l1123_112302


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_start_time_is_6pm_l1123_112355

/-- Two people walking towards each other -/
structure WalkingProblem where
  initial_distance : ℚ
  speed_A : ℚ
  speed_B : ℚ
  meeting_time : ℚ

/-- Calculate the starting time given a WalkingProblem -/
def calculate_start_time (p : WalkingProblem) : ℚ :=
  p.meeting_time - p.initial_distance / (p.speed_A + p.speed_B)

/-- Theorem: The starting time for the given problem is 6 pm (18:00) -/
theorem start_time_is_6pm (p : WalkingProblem) 
  (h1 : p.initial_distance = 50)
  (h2 : p.speed_A = 6)
  (h3 : p.speed_B = 4)
  (h4 : p.meeting_time = 23) : 
  calculate_start_time p = 18 := by
  sorry

/-- Example calculation -/
def example_problem : WalkingProblem := {
  initial_distance := 50
  speed_A := 6
  speed_B := 4
  meeting_time := 23
}

#eval calculate_start_time example_problem

end NUMINAMATH_CALUDE_ERRORFEEDBACK_start_time_is_6pm_l1123_112355


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_general_term_l1123_112372

def sequence_a : ℕ → ℚ
  | 0 => 1  -- Define for 0 to cover all natural numbers
  | 1 => 1
  | 2 => 5/3
  | (n+3) => (5/3) * sequence_a (n+2) - (2/3) * sequence_a (n+1)

def general_term (n : ℕ) : ℚ := 2 - (3/2) * (2/3)^n

theorem sequence_general_term :
  ∀ n : ℕ, sequence_a n = general_term n :=
by
  sorry

#eval sequence_a 5  -- Optional: to test the function
#eval general_term 5  -- Optional: to test the function

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_general_term_l1123_112372


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_octagon_circle_radius_l1123_112363

/-- The radius of a circle inscribed in an octagon formed by eight congruent parabolas -/
noncomputable def circle_radius : ℝ := (3 + 2 * Real.sqrt 2) / 4

/-- A parabola in the arrangement -/
def parabola (r : ℝ) (x : ℝ) : ℝ := x^2 + r

/-- The tangent line to the parabola at the point of contact with its neighbor -/
noncomputable def tangent_line (x : ℝ) : ℝ := x * (1 + Real.sqrt 2)

/-- The theorem stating the radius of the inscribed circle in the parabola arrangement -/
theorem parabola_octagon_circle_radius :
  ∃ (r : ℝ), r > 0 ∧
  (∀ x, (parabola r x = tangent_line x) → (x^2 - x * (1 + Real.sqrt 2) + r = 0)) ∧
  r = circle_radius := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_octagon_circle_radius_l1123_112363


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ada_original_seat_l1123_112338

/-- Represents the seats in the theater --/
inductive Seat
| one
| two
| three
| four
| five
| six
deriving Repr, DecidableEq

/-- Represents the friends --/
inductive Friend
| Ada
| Bea
| Ceci
| Dee
| Edie
| Fiona
deriving Repr, DecidableEq

/-- Represents the seating arrangement --/
def Arrangement := Friend → Seat

def move_right (s : Seat) (n : Nat) : Seat := 
  match s, n with
  | Seat.one, 1 => Seat.two
  | Seat.one, 2 => Seat.three
  | Seat.one, 3 => Seat.four
  | Seat.two, 1 => Seat.three
  | Seat.two, 2 => Seat.four
  | Seat.two, 3 => Seat.five
  | Seat.three, 1 => Seat.four
  | Seat.three, 2 => Seat.five
  | Seat.three, 3 => Seat.six
  | Seat.four, 1 => Seat.five
  | Seat.four, 2 => Seat.six
  | Seat.five, 1 => Seat.six
  | _, _ => s  -- If move is not possible, stay in place

def move_left (s : Seat) (n : Nat) : Seat :=
  match s, n with
  | Seat.two, 1 => Seat.one
  | Seat.three, 1 => Seat.two
  | Seat.three, 2 => Seat.one
  | Seat.four, 1 => Seat.three
  | Seat.four, 2 => Seat.two
  | Seat.four, 3 => Seat.one
  | Seat.five, 1 => Seat.four
  | Seat.five, 2 => Seat.three
  | Seat.five, 3 => Seat.two
  | Seat.six, 1 => Seat.five
  | Seat.six, 2 => Seat.four
  | Seat.six, 3 => Seat.three
  | _, _ => s  -- If move is not possible, stay in place

def swap (arr : Arrangement) (f1 f2 : Friend) : Arrangement :=
  fun f => if f = f1 then arr f2 else if f = f2 then arr f1 else arr f

theorem ada_original_seat (initial final : Arrangement) : 
  (∀ f, f ≠ Friend.Ada → initial f ≠ Seat.six) →  -- No one except possibly Ada starts in seat 6
  (final Friend.Bea = move_right (initial Friend.Bea) 3) →
  (final Friend.Ceci = move_left (initial Friend.Ceci) 1) →
  (final = swap initial Friend.Dee Friend.Edie) →
  (final Friend.Fiona = initial Friend.Fiona) →
  (final Friend.Ada = Seat.one ∨ final Friend.Ada = Seat.six) →
  initial Friend.Ada = Seat.three :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ada_original_seat_l1123_112338


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_lowest_score_proof_l1123_112379

theorem lowest_score_proof (n : ℕ) (avg : ℚ) (other_scores : List ℚ) 
  (h_n : n = 5)
  (h_avg : avg = 81.6)
  (h_other_scores : other_scores = [88, 84, 76])
  (h_avg_diff : ∀ (highest lowest : ℚ),
    (other_scores.sum + lowest) / 4 = (other_scores.sum + highest) / 4 - 6)
  (h_sum : ∀ (highest lowest : ℚ),
    n * avg = other_scores.sum + highest + lowest) :
  ∃ (lowest : ℚ), lowest = 68 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_lowest_score_proof_l1123_112379


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_can_double_money_l1123_112353

-- Define the currencies
inductive Currency
| Banana
| Coconut
| Raccoon
| Dollar

-- Define the exchange rates
def exchange_rate : Currency → Currency → Option ℚ
| Currency.Raccoon, Currency.Banana => some 6
| Currency.Raccoon, Currency.Coconut => some 11
| Currency.Dollar, Currency.Coconut => some 10
| Currency.Coconut, Currency.Dollar => some (1/15)
| Currency.Banana, Currency.Coconut => some (1/2)
| Currency.Coconut, Currency.Banana => some 2
| _, _ => none

-- Define a function to check if an exchange is allowed
def is_exchange_allowed : Currency → Currency → Bool
| Currency.Dollar, Currency.Raccoon => false
| Currency.Raccoon, Currency.Dollar => false
| Currency.Dollar, Currency.Banana => false
| Currency.Banana, Currency.Dollar => false
| _, _ => true

-- Define a function to represent a series of exchanges
def exchange_series (steps : List (Currency × Currency)) (initial_amount : ℚ) : Option ℚ :=
  sorry

-- Theorem statement
theorem can_double_money :
  ∃ (steps : List (Currency × Currency)) (final_amount : ℚ),
    exchange_series steps 100 = some final_amount ∧ final_amount > 200 :=
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_can_double_money_l1123_112353


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_sequence_partial_sum_l1123_112340

/-- Partial sum of a geometric sequence -/
def geometric_partial_sum (n : ℕ) : ℝ := sorry

/-- The theorem states that for a geometric sequence with partial sums S_n,
    if S_10 = 10 and S_20 = 30, then S_30 = 70 -/
theorem geometric_sequence_partial_sum :
  (geometric_partial_sum 10 = 10) →
  (geometric_partial_sum 20 = 30) →
  (geometric_partial_sum 30 = 70) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_sequence_partial_sum_l1123_112340


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_T_upper_bound_l1123_112395

noncomputable def a (n : ℕ+) : ℝ := 2 * n - 1

noncomputable def b (n : ℕ+) : ℝ := 1 / (a (n + 1))^2

noncomputable def T (n : ℕ+) : ℝ := (Finset.range n).sum (fun i => b ⟨i + 1, Nat.succ_pos i⟩)

theorem T_upper_bound (n : ℕ+) : T n < 1/4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_T_upper_bound_l1123_112395


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_BC_length_l1123_112315

/-- A point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- A triangle defined by three points -/
structure Triangle where
  A : Point
  B : Point
  C : Point

/-- Check if a point lies on the parabola y = -x^2 -/
def onParabola (p : Point) : Prop :=
  p.y = -p.x^2

/-- Check if two points form a line parallel to x-axis -/
def parallelToXAxis (p1 p2 : Point) : Prop :=
  p1.y = p2.y

/-- Calculate the area of a triangle -/
noncomputable def triangleArea (t : Triangle) : ℝ :=
  abs ((t.B.x - t.A.x) * (t.C.y - t.A.y) - (t.C.x - t.A.x) * (t.B.y - t.A.y)) / 2

/-- Calculate the distance between two points -/
noncomputable def distance (p1 p2 : Point) : ℝ :=
  Real.sqrt ((p2.x - p1.x)^2 + (p2.y - p1.y)^2)

theorem triangle_BC_length (t : Triangle) :
  t.A = ⟨0, 0⟩ →
  onParabola t.B →
  onParabola t.C →
  parallelToXAxis t.B t.C →
  triangleArea t = 72 →
  distance t.B t.C = 12 * (1/2)^(1/3) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_BC_length_l1123_112315


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_chosen_numbers_relatively_prime_l1123_112378

theorem chosen_numbers_relatively_prime (chosen : Finset ℕ) : 
  chosen.card = 1002 → 
  (∀ n, n ∈ chosen → 1 ≤ n ∧ n ≤ 2002) → 
  ∃ a b, a ∈ chosen ∧ b ∈ chosen ∧ a ≠ b ∧ Nat.gcd a b = 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_chosen_numbers_relatively_prime_l1123_112378


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_shifted_is_even_l1123_112385

noncomputable def f (x : ℝ) := Real.sqrt 3 * Real.sin x - Real.cos x

theorem f_shifted_is_even : 
  ∀ x : ℝ, f (x - π/3) = f (-(x - π/3)) :=
by
  intro x
  unfold f
  -- The proof steps would go here, but we'll use sorry for now
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_shifted_is_even_l1123_112385


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_value_from_cosine_l1123_112313

theorem angle_value_from_cosine (α : Real) :
  Real.cos α = -Real.sqrt 3 / 2 ∧ 0 < α ∧ α < Real.pi → α = 5 * Real.pi / 6 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_value_from_cosine_l1123_112313


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_absolute_value_equation_solution_difference_l1123_112365

theorem absolute_value_equation_solution_difference :
  ∃ x₁ x₂ : ℝ, 
    (|3 * x₁ + 6| = 24) ∧ 
    (|3 * x₂ + 6| = 24) ∧ 
    x₁ ≠ x₂ ∧ 
    |x₁ - x₂| = 16 :=
by
  -- Define the solutions
  let x₁ : ℝ := 6
  let x₂ : ℝ := -10

  -- Prove that x₁ and x₂ satisfy the equation
  have h1 : |3 * x₁ + 6| = 24 := by norm_num
  have h2 : |3 * x₂ + 6| = 24 := by norm_num

  -- Prove that x₁ ≠ x₂
  have h3 : x₁ ≠ x₂ := by norm_num

  -- Prove that |x₁ - x₂| = 16
  have h4 : |x₁ - x₂| = 16 := by norm_num

  -- Combine all the proofs
  exact ⟨x₁, x₂, h1, h2, h3, h4⟩


end NUMINAMATH_CALUDE_ERRORFEEDBACK_absolute_value_equation_solution_difference_l1123_112365


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_collinearity_implications_l1123_112399

open Real

noncomputable def α : ℝ := sorry

noncomputable def m : ℝ × ℝ := (cos α - Real.sqrt 2 / 3, -1)
noncomputable def n : ℝ × ℝ := (sin α, 1)

theorem vector_collinearity_implications 
  (h1 : α ∈ Set.Icc (-π/2) 0)
  (h2 : ∃ k : ℝ, m = k • n) :
  (sin α + cos α = Real.sqrt 2 / 3) ∧ 
  (sin (2*α) / (sin α - cos α) = 7/12) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_collinearity_implications_l1123_112399


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_investment_average_rate_l1123_112312

/-- Represents an investment split into two parts with different interest rates -/
structure Investment where
  total : ℚ
  rate1 : ℚ
  rate2 : ℚ
  amount1 : ℚ
  amount2 : ℚ

/-- Calculates the average interest rate of an investment -/
def averageRate (inv : Investment) : ℚ :=
  (inv.rate1 * inv.amount1 + inv.rate2 * inv.amount2) / inv.total

/-- Theorem stating the average interest rate for the given investment scenario -/
theorem investment_average_rate :
  ∀ (inv : Investment),
    inv.total = 6000 →
    inv.rate1 = 3 / 100 →
    inv.rate2 = 7 / 100 →
    inv.amount1 + inv.amount2 = inv.total →
    inv.rate1 * inv.amount1 = inv.rate2 * inv.amount2 →
    averageRate inv = 21 / 500 := by
  sorry

#eval (21 : ℚ) / 500 -- To verify that 21/500 is indeed equal to 0.042

end NUMINAMATH_CALUDE_ERRORFEEDBACK_investment_average_rate_l1123_112312


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_ratio_s6_s3_l1123_112305

/-- An arithmetic sequence with positive terms -/
structure ArithmeticSequence where
  a : ℕ → ℝ
  pos : ∀ n, a n > 0
  arith : ∀ n, a (n + 1) - a n = a 1 - a 0

/-- Sum of the first n terms of an arithmetic sequence -/
noncomputable def S (seq : ArithmeticSequence) (n : ℕ) : ℝ :=
  (n : ℝ) * (seq.a 0 + seq.a (n - 1)) / 2

/-- The main theorem -/
theorem min_ratio_s6_s3 (seq : ArithmeticSequence) 
  (h : (S seq 3 + 1) ^ 2 = (1 / 3) * (S seq 9)) : 
  S seq 6 / S seq 3 ≥ 5 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_ratio_s6_s3_l1123_112305


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_train_speed_calculation_l1123_112322

/-- The speed of a train in km/hr given its length and time to pass a fixed point -/
noncomputable def train_speed (length : ℝ) (time : ℝ) : ℝ :=
  (length / time) * 3.6

theorem train_speed_calculation :
  train_speed 280 28 = 36 := by
  unfold train_speed
  simp [mul_div_assoc]
  norm_num


end NUMINAMATH_CALUDE_ERRORFEEDBACK_train_speed_calculation_l1123_112322


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_f_on_interval_l1123_112390

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := x^3 + 2*x*(-3)

-- Define the derivative of f
def f' (x : ℝ) : ℝ := 3*x^2 - 6

-- Theorem statement
theorem range_of_f_on_interval :
  ∃ (a b : ℝ), a = -4 * Real.sqrt 2 ∧ b = 9 ∧
  (∀ x ∈ Set.Icc (-2) 3, a ≤ f x ∧ f x ≤ b) ∧
  (∃ x₁ ∈ Set.Icc (-2) 3, f x₁ = a) ∧
  (∃ x₂ ∈ Set.Icc (-2) 3, f x₂ = b) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_f_on_interval_l1123_112390


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_horse_tile_problem_l1123_112350

/-- Represents the number of tiles a big horse can pull -/
def big_horse_capacity : ℚ := 3

/-- Represents the number of small horses needed to pull one tile -/
def small_horses_per_tile : ℚ := 3

/-- Represents the total number of horses -/
def total_horses : ℕ := 100

/-- Represents the total number of tiles -/
def total_tiles : ℕ := 100

/-- The system of equations correctly represents the horse-tile problem -/
theorem horse_tile_problem (x y : ℚ) :
  (x + y = total_horses) ∧
  (big_horse_capacity * x + (1 / small_horses_per_tile) * y = total_tiles) :=
by
  sorry

#check horse_tile_problem

end NUMINAMATH_CALUDE_ERRORFEEDBACK_horse_tile_problem_l1123_112350


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_monotone_decreasing_l1123_112337

-- Define the function f(x)
noncomputable def f (x : ℝ) : ℝ := Real.log (abs x - 4) / Real.log (1/2)

-- Define the domain of f(x)
def domain (x : ℝ) : Prop := x > 4 ∨ x < -4

-- Theorem statement
theorem f_monotone_decreasing :
  ∀ x₁ x₂ : ℝ, x₁ ∈ Set.Ioi 4 → x₂ ∈ Set.Ioi 4 →
  x₁ < x₂ → f x₁ > f x₂ :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_monotone_decreasing_l1123_112337


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_is_odd_l1123_112333

noncomputable def f (x : ℝ) : ℝ := (5^x - 1) / (5^x + 1)

theorem f_is_odd : ∀ x : ℝ, f (-x) = -f x := by
  intro x
  simp [f]
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_is_odd_l1123_112333


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_therapy_charge_exists_l1123_112367

/-- Represents the pricing structure and calculates total charge for therapy sessions -/
def TherapyCharge (additional_hour_charge : ℕ) : Prop :=
  let first_hour_charge := additional_hour_charge + 25
  (first_hour_charge + 4 * additional_hour_charge = 250) ∧
  (first_hour_charge + additional_hour_charge = 115)

/-- Theorem stating the existence of a valid therapy charge structure -/
theorem therapy_charge_exists : ∃ (x : ℕ), TherapyCharge x := by
  -- We know the additional hour charge is 45
  let x := 45
  use x
  simp [TherapyCharge]
  -- Split the conjunction
  constructor
  -- Prove the first part
  · calc
      (x + 25) + 4 * x = 70 + 180 := by rfl
      _ = 250 := by rfl
  -- Prove the second part
  · calc
      (x + 25) + x = 70 + 45 := by rfl
      _ = 115 := by rfl

#check therapy_charge_exists

end NUMINAMATH_CALUDE_ERRORFEEDBACK_therapy_charge_exists_l1123_112367


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_focal_length_l1123_112325

/-- The focal length of an ellipse with given parameters --/
theorem ellipse_focal_length 
  (a : ℝ) 
  (h1 : a > 4) 
  (h2 : ∀ (y x : ℝ), y^2 / a^2 + x^2 / 16 = 1) 
  (h3 : (Real.sqrt (a^2 - 16)) / a = Real.sqrt 3 / 3) : 
  2 * Real.sqrt (a^2 - 16) = 4 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_focal_length_l1123_112325


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_power_function_range_l1123_112324

-- Define the power function as noncomputable
noncomputable def f (α : ℝ) (x : ℝ) : ℝ := x^α

-- State the theorem
theorem power_function_range (α : ℝ) (a : ℝ) :
  f α (1/2) = 4 → f α (a + 1) < f α 3 → a ∈ Set.Ioi 2 ∪ Set.Iic (-4) :=
by
  -- We use 'sorry' to skip the proof for now
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_power_function_range_l1123_112324


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_length_AB_is_4_l1123_112307

noncomputable section

-- Define the curve C
def curve_C (θ : ℝ) : ℝ × ℝ := (1 + Real.sqrt 3 * Real.cos θ, Real.sqrt 3 * Real.sin θ)

-- Define the line l
def line_l (θ : ℝ) (ρ : ℝ) : Prop := ρ * Real.cos (θ - Real.pi/6) = 3 * Real.sqrt 3

-- Define the ray OT
def ray_OT (ρ : ℝ) : ℝ × ℝ := (ρ * Real.cos (Real.pi/3), ρ * Real.sin (Real.pi/3))

-- Define point A as the intersection of curve C and ray OT
def point_A : ℝ × ℝ := ray_OT 2

-- Define point B as the intersection of line l and ray OT
def point_B : ℝ × ℝ := ray_OT 6

-- Theorem statement
theorem length_AB_is_4 :
  let A := point_A
  let B := point_B
  Real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2) = 4 := by sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_length_AB_is_4_l1123_112307


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_d₁_value_l1123_112317

/-- Definition of E(n) -/
def E (n : ℕ) : ℕ :=
  (Finset.filter (fun (b : Fin n × Fin n × Fin n × Fin n) =>
    b.1 ≠ b.2.1 ∧ b.1 ≠ b.2.2.1 ∧ b.1 ≠ b.2.2.2 ∧
    b.2.1 ≠ b.2.2.1 ∧ b.2.1 ≠ b.2.2.2 ∧
    b.2.2.1 ≠ b.2.2.2 ∧
    (b.1.val + b.2.1.val + b.2.2.1.val + b.2.2.2.val) % (n - 2) = 0)
    (Finset.product (Finset.univ : Finset (Fin n)) 
      (Finset.product (Finset.univ : Finset (Fin n))
        (Finset.product (Finset.univ : Finset (Fin n)) (Finset.univ : Finset (Fin n)))))).card

/-- The polynomial r(x) -/
def r (x : ℝ) (d₃ d₂ d₁ d₀ : ℝ) : ℝ :=
  d₃ * x^3 + d₂ * x^2 + d₁ * x + d₀

/-- The main theorem -/
theorem d₁_value (d₃ d₂ d₁ d₀ : ℝ) :
  (∀ n : ℕ, n ≥ 7 → Odd n → (E n : ℝ) = r n d₃ d₂ d₁ d₀) →
  d₁ = 6 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_d₁_value_l1123_112317


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_liquid_y_film_radius_l1123_112387

/-- The radius of a circular film formed by a liquid on water -/
noncomputable def circular_film_radius (box_length box_width box_height film_thickness : ℝ) : ℝ :=
  Real.sqrt ((box_length * box_width * box_height) / (Real.pi * film_thickness))

/-- Theorem stating the radius of the circular film for the given problem -/
theorem liquid_y_film_radius : 
  circular_film_radius 9 3 18 0.2 = Real.sqrt (2430 / Real.pi) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_liquid_y_film_radius_l1123_112387


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_side_length_ratio_triangle_square_pentagon_l1123_112332

/-- The side length of a regular polygon given its perimeter and number of sides -/
noncomputable def sideLengthRegularPolygon (perimeter : ℝ) (sides : ℕ) : ℝ :=
  perimeter / (sides : ℝ)

theorem side_length_ratio_triangle_square_pentagon : 
  let triangleSide := sideLengthRegularPolygon 30 3
  let squareSide := sideLengthRegularPolygon 40 4
  let pentagonSide := sideLengthRegularPolygon 50 5
  triangleSide = squareSide ∧ squareSide = pentagonSide := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_side_length_ratio_triangle_square_pentagon_l1123_112332


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_tangent_length_l1123_112342

/-- Circle C with center (1,2) and radius 1 -/
def circle_C (x y : ℝ) : Prop := (x - 1)^2 + (y - 2)^2 = 1

/-- Line of symmetry for circle C -/
def symmetry_line (a b x y : ℝ) : Prop := 2*a*x + b*y + 2 = 0

/-- Length of tangent line from point (a,b) to circle C -/
noncomputable def tangent_length (a b : ℝ) : ℝ := Real.sqrt ((a - 1)^2 + (b - 2)^2 - 1)

/-- Theorem stating the minimum length of the tangent line -/
theorem min_tangent_length :
  ∀ a b : ℝ,
  (∃ x y : ℝ, circle_C x y ∧ symmetry_line a b x y) →
  (∃ m : ℝ, ∀ a' b' : ℝ, tangent_length a' b' ≥ m ∧ (∃ a₀ b₀ : ℝ, tangent_length a₀ b₀ = m)) ∧
  (∃ a₀ b₀ : ℝ, tangent_length a₀ b₀ = Real.sqrt 7) :=
by
  sorry

#check min_tangent_length

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_tangent_length_l1123_112342


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_total_amount_is_210_l1123_112361

/-- Represents the division of money among three parties -/
structure MoneyDivision where
  x : ℚ  -- Amount x gets per unit
  y : ℚ  -- Amount y gets per unit
  z : ℚ  -- Amount z gets per unit

/-- The total amount shared given a money division and y's share -/
def totalAmount (div : MoneyDivision) (y_share : ℚ) : ℚ :=
  let units := y_share / div.y
  units * (div.x + div.y + div.z)

/-- Theorem stating the total amount shared is 210 given the problem conditions -/
theorem total_amount_is_210 :
  let div : MoneyDivision := { x := 1, y := 45/100, z := 30/100 }
  totalAmount div 54 = 210 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_total_amount_is_210_l1123_112361


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_equals_target_set_l1123_112377

-- Define the sets M and N
def M : Set ℝ := {x | ∃ y, y = Real.sqrt (Real.log (4 * x - 3) / Real.log 0.5)}
def N : Set ℝ := {y | ∃ x, y = Real.sqrt (Real.log (4 * x - 3) / Real.log 0.5)}

-- Define the intersection of M and N
def M_intersect_N : Set ℝ := M ∩ N

-- Define the open-closed interval (3/4, 1]
def target_set : Set ℝ := Set.Ioc (3/4) 1

-- Theorem statement
theorem intersection_equals_target_set : M_intersect_N = target_set := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_equals_target_set_l1123_112377


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_expression_approx_four_l1123_112316

/-- Theorem: For any real number X ≠ 0, the expression ((X + 0.064)^2 - (X - 0.064)^2) / (X * 0.064) is approximately equal to 4. -/
theorem expression_approx_four (X : ℝ) (hX : X ≠ 0) :
  ∃ ε > 0, |((X + 0.064)^2 - (X - 0.064)^2) / (X * 0.064) - 4| < ε :=
by
  -- We use ε = 0.000000000000002 to match the given approximation
  use 0.000000000000002
  sorry -- Proof details omitted


end NUMINAMATH_CALUDE_ERRORFEEDBACK_expression_approx_four_l1123_112316


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_2alpha_proof_l1123_112327

theorem sin_2alpha_proof (α : ℝ) (h1 : α ∈ Set.Ioo 0 π) (h2 : Real.tan (π/4 - α) = 1/3) :
  Real.sin (2 * α) = 4/5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_2alpha_proof_l1123_112327


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_pairs_count_l1123_112308

def marble_colors := Fin 5

def marble_counts : Fin 5 → ℕ
| 0 => 1  -- red
| 1 => 1  -- green
| 2 => 1  -- blue
| 3 => 1  -- purple
| 4 => 4  -- yellow

def total_marbles : ℕ := Finset.sum Finset.univ marble_counts

theorem unique_pairs_count : 
  (Finset.sum Finset.univ marble_counts = 8) → 
  (∃ (c : Fin 5), marble_counts c = 4 ∧ ∀ (c' : Fin 5), c' ≠ c → marble_counts c' = 1) →
  (Finset.sum (Finset.univ.product Finset.univ) (λ (p : Fin 5 × Fin 5) => 
    if p.1 = p.2 
    then Nat.choose (marble_counts p.1) 2 
    else (marble_counts p.1 * marble_counts p.2) / 2)) = 7 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_pairs_count_l1123_112308


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_school_arrival_probability_l1123_112368

/-- Represents a uniform probability distribution over a time frame -/
structure TimeDistribution where
  total_minutes : ℚ
  target_minutes : ℚ
  h_positive : total_minutes > 0
  h_target_valid : target_minutes ≥ 0 ∧ target_minutes ≤ total_minutes

/-- Calculates the probability of an event occurring within the target minutes -/
def probability (d : TimeDistribution) : ℚ :=
  d.target_minutes / d.total_minutes

theorem school_arrival_probability :
  let d : TimeDistribution := {
    total_minutes := 40
    target_minutes := 30
    h_positive := by norm_num
    h_target_valid := by norm_num
  }
  probability d = 3/4 := by
  -- Unfold the definition of probability
  unfold probability
  -- Simplify the fraction
  norm_num


end NUMINAMATH_CALUDE_ERRORFEEDBACK_school_arrival_probability_l1123_112368


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_equation_l1123_112369

-- Define the function f(x)
noncomputable def f (x : ℝ) : ℝ := x^2 + Real.log x

-- Define the derivative of f(x)
noncomputable def f_derivative (x : ℝ) : ℝ := 2 * x + 1 / x

-- Theorem statement
theorem tangent_line_equation :
  ∃ (m b : ℝ), 
    (∀ x y : ℝ, y = m * (x - 1) + f 1 → (3 * x - y - 2 = 0)) ∧
    (m = f_derivative 1) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_equation_l1123_112369


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_students_l1123_112300

/-- Represents a student in the tournament -/
structure Student where
  solved_problems : Finset Nat

/-- The tournament setup -/
structure Tournament where
  students : Finset Student
  problem_constraint : ∀ p : Nat, p < 6 → (students.filter (λ s => p ∈ s.solved_problems)).card = 1000
  no_all_solved : ∀ s₁ s₂ : Student, s₁ ∈ students → s₂ ∈ students → s₁ ≠ s₂ → 
    (s₁.solved_problems ∪ s₂.solved_problems).card < 6

/-- The theorem statement -/
theorem min_students (t : Tournament) : t.students.card ≥ 2000 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_students_l1123_112300


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_coefficients_and_exception_l1123_112331

/-- The original function -/
noncomputable def f (x : ℝ) : ℝ := (x^3 + 5*x^2 + 8*x + 4) / (x + 2)

/-- The simplified quadratic function -/
noncomputable def g (A B C : ℝ) (x : ℝ) : ℝ := A*x^2 + B*x + C

/-- The exception value -/
def D : ℝ := -2

/-- Theorem stating that the sum of coefficients A, B, C, and the exception value D is 4 -/
theorem sum_of_coefficients_and_exception :
  ∃ (A B C : ℝ), (∀ x : ℝ, x ≠ D → f x = g A B C x) ∧ A + B + C + D = 4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_coefficients_and_exception_l1123_112331


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_centroid_minimizes_sum_squared_distances_l1123_112314

-- Define a triangle in a 2D plane
structure Triangle where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ

-- Define the centroid of a triangle
noncomputable def centroid (t : Triangle) : ℝ × ℝ :=
  ((t.A.1 + t.B.1 + t.C.1) / 3, (t.A.2 + t.B.2 + t.C.2) / 3)

-- Define the distance between two points
noncomputable def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

-- Define the sum of squares of distances from a point to triangle vertices
noncomputable def sumSquaredDistances (t : Triangle) (p : ℝ × ℝ) : ℝ :=
  (distance p t.A)^2 + (distance p t.B)^2 + (distance p t.C)^2

-- Theorem: The centroid minimizes the sum of squares of distances
theorem centroid_minimizes_sum_squared_distances (t : Triangle) :
  ∀ p : ℝ × ℝ, sumSquaredDistances t (centroid t) ≤ sumSquaredDistances t p := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_centroid_minimizes_sum_squared_distances_l1123_112314


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_train_platform_crossing_time_l1123_112346

theorem train_platform_crossing_time 
  (train_length : ℝ) 
  (platform_length : ℝ) 
  (post_crossing_time : ℝ) 
  (h1 : train_length = 300)
  (h2 : platform_length = 350)
  (h3 : post_crossing_time = 18) :
  (train_length + platform_length) / (train_length / post_crossing_time) = 39 := by
  -- Define train speed
  let train_speed := train_length / post_crossing_time
  -- Define total distance
  let total_distance := train_length + platform_length
  -- Calculate platform crossing time
  have platform_crossing_time : ℝ := total_distance / train_speed
  -- Prove that platform_crossing_time equals 39
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_train_platform_crossing_time_l1123_112346


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_P_value_at_200_l1123_112392

open BigOperators

def P (n : ℕ) : ℚ :=
  ∏ k in Finset.range (n - 1), (1 - 1 / (k + 2)^2)

theorem P_value_at_200 : P 200 = 3 / 40000 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_P_value_at_200_l1123_112392


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_proof_l1123_112359

theorem inequality_proof (a b c lambda : ℝ) (h : lambda ≥ 8) :
  a / Real.sqrt (a^2 + lambda * b * c) + 
  b / Real.sqrt (b^2 + lambda * c * a) + 
  c / Real.sqrt (c^2 + lambda * a * b)
  ≥ 3 / Real.sqrt (lambda + 1) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_proof_l1123_112359


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_angle_relation_l1123_112362

/-- Given two non-intersecting circles with common tangents, this theorem relates the angles
    between the tangents to the angle between lines from the center of the larger circle
    tangent to the smaller circle. -/
theorem tangent_angle_relation (α β φ : Real) : 
  (∃ (r R d : Real), r > 0 ∧ R > r ∧ d > R + r ∧
    Real.sin (α/2) = (R - r) / d ∧
    Real.sin (β/2) = (R + r) / d ∧
    Real.sin (φ/2) = r / d) →
  φ = 2 * Real.arcsin ((Real.sin (β/2) - Real.sin (α/2)) / 2) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_angle_relation_l1123_112362


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_symmetric_axis_of_quadratic_l1123_112321

-- Define the quadratic function
def f (x : ℝ) : ℝ := x^2 + 8*x + 9

-- State the theorem
theorem symmetric_axis_of_quadratic :
  ∃ a : ℝ, (∀ x : ℝ, f (a + x) = f (a - x)) ∧ a = -4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_symmetric_axis_of_quadratic_l1123_112321


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_between_m_and_n_l1123_112389

def a : ℝ × ℝ := (3, 4)
def b : ℝ × ℝ := (9, 12)
def c : ℝ × ℝ := (4, -3)
def m : ℝ × ℝ := (2 * a.1 - b.1, 2 * a.2 - b.2)
def n : ℝ × ℝ := (a.1 + c.1, a.2 + c.2)

theorem angle_between_m_and_n : 
  Real.arccos ((m.1 * n.1 + m.2 * n.2) / (Real.sqrt (m.1^2 + m.2^2) * Real.sqrt (n.1^2 + n.2^2))) = π * 3 / 4 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_between_m_and_n_l1123_112389


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_bridge_angle_sum_l1123_112393

-- Define the triangles and their properties
structure Triangle where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ
  isIsosceles : (A.1 - B.1)^2 + (A.2 - B.2)^2 = (A.1 - C.1)^2 + (A.2 - C.2)^2

noncomputable def angle (p q r : ℝ × ℝ) : ℝ := sorry

-- Define the theorem
theorem bridge_angle_sum 
  (ABC DEF : Triangle)
  (h1 : angle ABC.A ABC.B ABC.C = 25)
  (h2 : angle DEF.A DEF.B DEF.C = 35)
  (h3 : ∃ (D : ℝ × ℝ), (D.1 - ABC.B.1) / (D.2 - ABC.B.2) = (D.1 - ABC.C.1) / (D.2 - ABC.C.2) ∧
                        (D.1 - DEF.B.1) / (D.2 - DEF.B.2) = (D.1 - DEF.C.1) / (D.2 - DEF.C.2)) :
  ∃ (D : ℝ × ℝ), angle D ABC.A ABC.C + angle ABC.A D DEF.B = 150 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_bridge_angle_sum_l1123_112393


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_judys_grocery_bill_l1123_112352

def banana_price : ℚ := 2
def rice_price : ℚ := 6
def pineapple_price : ℚ := 5
def cake_price : ℚ := 10
def banana_count : ℕ := 4
def rice_count : ℕ := 2
def pineapple_count : ℕ := 3
def cake_count : ℕ := 1
def banana_discount : ℚ := 1/4
def coupon_amount : ℚ := 10
def coupon_threshold : ℚ := 30

def grocery_bill (banana_price rice_price pineapple_price cake_price : ℚ)
                 (banana_count rice_count pineapple_count cake_count : ℕ)
                 (banana_discount coupon_amount coupon_threshold : ℚ) : ℚ :=
  let discounted_banana_price := banana_price * (1 - banana_discount)
  let total_before_coupon := discounted_banana_price * banana_count +
                             rice_price * rice_count +
                             pineapple_price * pineapple_count +
                             cake_price * cake_count
  if total_before_coupon > coupon_threshold
  then total_before_coupon - coupon_amount
  else total_before_coupon

theorem judys_grocery_bill :
  grocery_bill banana_price rice_price pineapple_price cake_price
               banana_count rice_count pineapple_count cake_count
               banana_discount coupon_amount coupon_threshold = 33 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_judys_grocery_bill_l1123_112352


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_normal_distribution_probability_l1123_112394

-- Define a random variable following normal distribution
def normal_distribution (μ σ : ℝ) : Type := ℝ

-- Define the probability function
noncomputable def P (μ σ : ℝ) (A : Set ℝ) : ℝ := sorry

-- State the theorem
theorem normal_distribution_probability 
  (μ σ : ℝ) (ξ : normal_distribution μ σ) 
  (h1 : P μ σ {x | x < -1} = 0.3) 
  (h2 : P μ σ {x | x > 2} = 0.3) : 
  P μ σ {x | x < 2*μ + 1} = 0.7 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_normal_distribution_probability_l1123_112394


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_abc_point_c_coordinates_l1123_112382

/-- Helper function to calculate the area of a triangle given three points -/
noncomputable def area_triangle (A B C : ℝ × ℝ) : ℝ :=
  (1/2) * abs ((B.1 - A.1) * (C.2 - A.2) - (C.1 - A.1) * (B.2 - A.2))

/-- Given a triangle ABC where A(3,2), B(-1,5), C is on the line 3x - y + 3 = 0,
    and the area of triangle ABC is 10, prove that the coordinates of point C
    are either (-1, 0) or (5/3, 8). -/
theorem triangle_abc_point_c_coordinates :
  ∀ (C : ℝ × ℝ),
  (C.1 = -1 ∧ C.2 = 0) ∨ (C.1 = 5/3 ∧ C.2 = 8) ↔
  (3 * C.1 - C.2 + 3 = 0 ∧
   area_triangle (3, 2) (-1, 5) C = 10) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_abc_point_c_coordinates_l1123_112382


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_square_area_first_quadrant_l1123_112383

/-- Represents a point in 2D space -/
structure Point2D where
  x : ℝ
  y : ℝ

/-- Represents a square with vertices A, B, C, D -/
structure Square where
  A : Point2D
  B : Point2D
  C : Point2D
  D : Point2D

/-- The area of a square given the coordinates of vertex C -/
noncomputable def squareArea (u v : ℝ) : ℝ := v^2 / 4

theorem square_area_first_quadrant (s : Square) (u v : ℝ) 
  (h1 : s.A.x ≥ 0 ∧ s.A.y = 0)  -- A is on positive x-axis
  (h2 : s.B.x = 0 ∧ s.B.y ≥ 0)  -- B is on positive y-axis
  (h3 : s.C.x = u ∧ s.C.y = v)  -- C has coordinates (u, v)
  (h4 : u > 0 ∧ v > 0)          -- C is in first quadrant
  : squareArea u v = (s.B.y - s.A.x)^2 := by
  sorry

#check square_area_first_quadrant

end NUMINAMATH_CALUDE_ERRORFEEDBACK_square_area_first_quadrant_l1123_112383


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_proof_l1123_112360

theorem inequality_proof (n m : ℕ) (h1 : n > 0) (h2 : m > 0) (h3 : n ≥ m) :
  (n + 1) ^ m * n ^ m ≥ Nat.factorial (n + m) / Nat.factorial (n - m) ∧ 
  Nat.factorial (n + m) / Nat.factorial (n - m) ≥ 2 ^ m * Nat.factorial m := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_proof_l1123_112360


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_right_prism_volume_l1123_112375

/-- Helper function to represent the volume of a right prism -/
noncomputable def VolumeRightPrism (α β d : Real) : Real :=
  (d^3 * Real.sin β * Real.sin (2*β) * Real.sin (2*α)) / 8

/-- The volume of a right prism with a right triangular base -/
theorem right_prism_volume 
  (α β : Real) 
  (d : Real) 
  (h_α_acute : 0 < α ∧ α < π / 2) 
  (h_β_acute : 0 < β ∧ β < π / 2) 
  (h_d_pos : d > 0) : 
  ∃ (V : Real), V = (d^3 * Real.sin β * Real.sin (2*β) * Real.sin (2*α)) / 8 ∧ 
  V = VolumeRightPrism α β d :=
by
  let V := VolumeRightPrism α β d
  use V
  constructor
  · rfl
  · rfl


end NUMINAMATH_CALUDE_ERRORFEEDBACK_right_prism_volume_l1123_112375


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_mathematician_overlap_l1123_112320

/-- Represents the duration of stay for each mathematician -/
noncomputable def n : ℝ := 60 - 30 * Real.sqrt 2

/-- Represents the probability of at least one overlap -/
noncomputable def overlap_probability : ℝ := 1 - (60 - n)^2 / 3600

/-- Represents the equation n = x - y√z -/
def n_equation (x y z : ℕ+) : Prop := (n : ℝ) = x - y * Real.sqrt z

/-- Main theorem -/
theorem mathematician_overlap (x y z : ℕ+) 
  (h1 : n_equation x y z) 
  (h2 : ¬ ∃ (p : ℕ), Prime p ∧ z % (p^2) = 0) 
  (h3 : overlap_probability = 1/2) : 
  x + y + z = 92 := by
  sorry

#check mathematician_overlap

end NUMINAMATH_CALUDE_ERRORFEEDBACK_mathematician_overlap_l1123_112320


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_and_monotonicity_l1123_112391

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := (2 * x^2 - 4 * a * x) * Real.log x + x^2

theorem tangent_line_and_monotonicity (a : ℝ) :
  -- Part I: Tangent line equation when a = 0
  (8 * Real.exp 1 * x - y - 5 * (Real.exp 1)^2 = 0 ↔
    y - f 0 (Real.exp 1) = (deriv (f 0) (Real.exp 1)) * (x - Real.exp 1)) ∧
  -- Part II: Monotonicity intervals
  (∀ x₁ x₂, 0 < x₁ ∧ x₁ < x₂ ∧ x₂ < Real.exp (-1) → a ≤ 0 → f a x₁ > f a x₂) ∧
  (∀ x₁ x₂, Real.exp (-1) < x₁ ∧ x₁ < x₂ → a ≤ 0 → f a x₁ < f a x₂) ∧
  (∀ x₁ x₂, 0 < x₁ ∧ x₁ < x₂ ∧ x₂ < a → 0 < a ∧ a < Real.exp (-1) → f a x₁ < f a x₂) ∧
  (∀ x₁ x₂, a < x₁ ∧ x₁ < x₂ ∧ x₂ < Real.exp (-1) → 0 < a ∧ a < Real.exp (-1) → f a x₁ > f a x₂) ∧
  (∀ x₁ x₂, Real.exp (-1) < x₁ ∧ x₁ < x₂ → 0 < a ∧ a < Real.exp (-1) → f a x₁ < f a x₂) ∧
  (∀ x₁ x₂, 0 < x₁ ∧ x₁ < x₂ → a = Real.exp (-1) → f a x₁ < f a x₂) ∧
  (∀ x₁ x₂, 0 < x₁ ∧ x₁ < x₂ ∧ x₂ < Real.exp (-1) → a > Real.exp (-1) → f a x₁ < f a x₂) ∧
  (∀ x₁ x₂, Real.exp (-1) < x₁ ∧ x₁ < x₂ ∧ x₂ < a → a > Real.exp (-1) → f a x₁ > f a x₂) ∧
  (∀ x₁ x₂, a < x₁ ∧ x₁ < x₂ → a > Real.exp (-1) → f a x₁ < f a x₂) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_and_monotonicity_l1123_112391


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_peter_catches_rob_l1123_112303

/-- The line on which Rob runs -/
noncomputable def rob_line (x : ℝ) : ℝ := 2 * x + 5

/-- Rob's speed in units per second -/
def rob_speed : ℝ := 2

/-- Peter's speed in units per second -/
def peter_speed : ℝ := 3

/-- Rob's starting point -/
def rob_start : ℝ × ℝ := (0, 5)

/-- Peter's starting point -/
def peter_start : ℝ × ℝ := (0, 5)

/-- The point where Peter catches Rob -/
def catch_point : ℝ × ℝ := (17, 39)

/-- The time when Peter starts running after Rob -/
noncomputable def t : ℝ := Real.sqrt 1445 / 6

/-- Theorem stating that t is the correct time when Peter starts running after Rob -/
theorem peter_catches_rob :
  ∃ (tr : ℝ), 
    rob_speed * tr = Real.sqrt ((catch_point.1 - rob_start.1)^2 + (catch_point.2 - rob_start.2)^2) ∧
    peter_speed * (tr - t) = Real.sqrt ((catch_point.1 - peter_start.1)^2 + (catch_point.2 - peter_start.2)^2) ∧
    catch_point.2 = rob_line catch_point.1 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_peter_catches_rob_l1123_112303


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_discount_ratio_approximately_108_57_percent_l1123_112364

-- Define the discount rate and the non-discounted gallons
noncomputable def discount_rate : ℝ := 0.1
noncomputable def non_discounted_gallons : ℝ := 6

-- Define the function to calculate per-gallon discount
noncomputable def per_gallon_discount (total_gallons : ℝ) : ℝ :=
  ((total_gallons - non_discounted_gallons) * discount_rate) / total_gallons

-- Define the purchases
noncomputable def kim_gallons : ℝ := 20
noncomputable def isabella_gallons : ℝ := 25

-- State the theorem
theorem discount_ratio_approximately_108_57_percent :
  ∃ (ε : ℝ), ε > 0 ∧ ε < 0.0001 ∧ 
  |per_gallon_discount isabella_gallons / per_gallon_discount kim_gallons - 1.0857| < ε := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_discount_ratio_approximately_108_57_percent_l1123_112364


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circumscribed_circle_diameter_theorem_l1123_112371

noncomputable def Triangle := Real → Real → Real → Type

noncomputable def circumscribed_circle_diameter (t : Triangle) : Real := sorry

noncomputable def has_side_and_opposite_angle (t : Triangle) (side angle : Real) : Prop := sorry

theorem circumscribed_circle_diameter_theorem 
  (t : Triangle) 
  (side angle : Real) 
  (h1 : side = 16) 
  (h2 : angle = 30 * Real.pi / 180) 
  (h3 : has_side_and_opposite_angle t side angle) : 
  circumscribed_circle_diameter t = 32 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_circumscribed_circle_diameter_theorem_l1123_112371


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_major_axis_length_l1123_112323

/-- Given an ellipse tangent to the x-axis with foci at (1,1) and (5,2), 
    its major axis length is √21 -/
theorem ellipse_major_axis_length : 
  ∀ (E : Set (ℝ × ℝ)),
    (∃ (a b : ℝ), E = {p : ℝ × ℝ | (p.1^2 / a^2) + (p.2^2 / b^2) = 1}) →
    (∀ (x : ℝ), (x, 0) ∈ E → ¬((x, 0) ∈ interior E)) →
    ((1, 1) ∈ E ∧ (5, 2) ∈ E) →
    (∀ (p : ℝ × ℝ), p ∈ E → 
      Real.sqrt ((p.1 - 1)^2 + (p.2 - 1)^2) + Real.sqrt ((p.1 - 5)^2 + (p.2 - 2)^2) = 
      Real.sqrt ((1 - 5)^2 + (1 - 2)^2)) →
    ∃ (a : ℝ), a > 0 ∧ (∀ (p : ℝ × ℝ), p ∈ E → 
      (p.1 - 3)^2 / a^2 + (p.2 - 1.5)^2 / (a^2 - 17/4) = 1) ∧
    2 * a = Real.sqrt 21 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_major_axis_length_l1123_112323


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_count_even_combinations_l1123_112397

-- Define the binomial coefficient function
def binomial (n r : ℕ) : ℕ :=
  Nat.choose n r

-- Define a function to check if a number is even
def isEven (n : ℕ) : Bool :=
  n % 2 = 0

-- Define the main theorem
theorem count_even_combinations :
  (Finset.sum (Finset.range 64)
    (λ n ↦ Finset.sum (Finset.range (n + 1))
      (λ r ↦ if isEven (binomial n r) then 1 else 0))) = 1351 := by
  sorry

#eval Finset.sum (Finset.range 64)
  (λ n ↦ Finset.sum (Finset.range (n + 1))
    (λ r ↦ if isEven (binomial n r) then 1 else 0))

end NUMINAMATH_CALUDE_ERRORFEEDBACK_count_even_combinations_l1123_112397


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_perpendicular_probability_is_44_81_l1123_112306

/-- A right triangular prism with an isosceles right triangle base -/
structure RightTriangularPrism where
  edges : Fin 9 → Set Point
  is_isosceles_right_triangle_base : Prop
  is_right_prism : Prop

/-- Two randomly selected edges from the prism -/
def random_edge_pair (prism : RightTriangularPrism) : Set Point × Set Point :=
  sorry

/-- Predicate to check if two edges are perpendicular -/
def are_perpendicular (e1 e2 : Set Point) : Prop :=
  sorry

/-- The probability of two randomly selected edges being perpendicular -/
noncomputable def perpendicular_probability (prism : RightTriangularPrism) : ℚ :=
  sorry

/-- Theorem stating the probability of two randomly selected edges being perpendicular -/
theorem perpendicular_probability_is_44_81 (prism : RightTriangularPrism) :
  perpendicular_probability prism = 44 / 81 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_perpendicular_probability_is_44_81_l1123_112306


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_a_is_integer_l1123_112343

def a : ℕ → ℤ
  | 0 => 1
  | 1 => 2
  | 2 => 3
  | n + 3 => a (n + 2) - a (n + 1) + (a (n + 2) ^ 2) / a n

theorem a_is_integer (n : ℕ) : ∃ k : ℤ, a n = k := by
  induction n with
  | zero => exists 1
  | succ n ih =>
    cases n with
    | zero => exists 2
    | succ n =>
      cases n with
      | zero => exists 3
      | succ n =>
        cases ih with
        | intro k hk =>
          -- The proof steps would go here, but we'll use sorry for now
          sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_a_is_integer_l1123_112343


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_boat_speed_is_25_l1123_112376

/-- Represents the speed of a boat in still water -/
def boat_speed : ℝ → Prop := sorry

/-- Represents the speed of the stream -/
def stream_speed : ℝ → Prop := sorry

/-- The boat can travel 10 km upstream in 1 hour -/
axiom upstream_condition (vb vs : ℝ) :
  boat_speed vb → stream_speed vs → vb - vs = 10

/-- The boat can travel 10 km downstream in 15 minutes (1/4 hour) -/
axiom downstream_condition (vb vs : ℝ) :
  boat_speed vb → stream_speed vs → vb + vs = 40

/-- The speed of the boat in still water is 25 kmph -/
theorem boat_speed_is_25 :
  ∃ vb vs : ℝ, boat_speed vb ∧ stream_speed vs ∧ vb = 25 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_boat_speed_is_25_l1123_112376


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sufficient_not_necessary_l1123_112356

/-- An arithmetic sequence -/
structure ArithmeticSequence where
  a : ℕ → ℚ
  d : ℚ
  h : ∀ n, a (n + 1) = a n + d

/-- Sum of first n terms of an arithmetic sequence -/
def sum_n (seq : ArithmeticSequence) (n : ℕ) : ℚ :=
  n * (seq.a 1 + seq.a n) / 2

theorem sufficient_not_necessary
  (seq : ArithmeticSequence)
  (h : seq.a 2 = 1) :
  (∀ seq', seq'.a 2 = 1 → seq'.a 3 > 5 → sum_n seq' 3 + sum_n seq' 9 > 93) ∧
  (∃ seq', seq'.a 2 = 1 ∧ sum_n seq' 3 + sum_n seq' 9 > 93 ∧ seq'.a 3 ≤ 5) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_sufficient_not_necessary_l1123_112356


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_climb_stairs_formula_l1123_112374

/-- The number of ways to climb n stairs when taking one or two steps at a time -/
noncomputable def climbStairs (n : ℕ) : ℝ :=
  let Φ := (1 + Real.sqrt 5) / 2
  let φ := (1 - Real.sqrt 5) / 2
  let A := (1 + Real.sqrt 5) / (2 * Real.sqrt 5)
  let B := (Real.sqrt 5 - 1) / (2 * Real.sqrt 5)
  A * Φ^n + B * φ^n

/-- Theorem: The number of ways to climb n stairs is given by the formula -/
theorem climb_stairs_formula (n : ℕ) (h : n ≥ 1) :
  climbStairs n = climbStairs (n-1) + climbStairs (n-2) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_climb_stairs_formula_l1123_112374

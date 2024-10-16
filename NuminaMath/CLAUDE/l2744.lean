import Mathlib

namespace NUMINAMATH_CALUDE_intersection_equals_interval_l2744_274476

-- Define the sets A and B
def A : Set ℝ := {x | ∃ y, y = Real.log (x - 1) / Real.log 10}
def B : Set ℝ := {x | -x^2 + 2 ≤ 2}

-- Define the open interval (1, 2]
def openClosedInterval : Set ℝ := {x | 1 < x ∧ x ≤ 2}

-- State the theorem
theorem intersection_equals_interval : A ∩ B = openClosedInterval := by sorry

end NUMINAMATH_CALUDE_intersection_equals_interval_l2744_274476


namespace NUMINAMATH_CALUDE_periodic_function_proof_l2744_274478

open Real

theorem periodic_function_proof (a : ℚ) (b d c : ℝ) 
  (f : ℝ → ℝ) 
  (h_range : ∀ x, f x ∈ Set.Icc (-1) 1)
  (h_eq : ∀ x, f (x + a + b) - f (x + b) = c * (x + 2 * a + ⌊x⌋ - 2 * ⌊x + a⌋ - ⌊b⌋) + d) :
  ∃ p : ℝ, p > 0 ∧ ∀ x, f (x + p) = f x :=
sorry

end NUMINAMATH_CALUDE_periodic_function_proof_l2744_274478


namespace NUMINAMATH_CALUDE_basic_structures_are_sequential_conditional_loop_modular_not_basic_structure_l2744_274474

/-- The set of basic algorithm structures -/
inductive AlgorithmStructure
  | Sequential
  | Conditional
  | Loop
  | Modular

/-- The set of basic algorithm structures contains exactly Sequential, Conditional, and Loop -/
def basic_structures : Set AlgorithmStructure :=
  {AlgorithmStructure.Sequential, AlgorithmStructure.Conditional, AlgorithmStructure.Loop}

/-- The theorem stating that the basic structures are exactly Sequential, Conditional, and Loop -/
theorem basic_structures_are_sequential_conditional_loop :
  basic_structures = {AlgorithmStructure.Sequential, AlgorithmStructure.Conditional, AlgorithmStructure.Loop} :=
by sorry

/-- The theorem stating that Modular is not a basic structure -/
theorem modular_not_basic_structure :
  AlgorithmStructure.Modular ∉ basic_structures :=
by sorry

end NUMINAMATH_CALUDE_basic_structures_are_sequential_conditional_loop_modular_not_basic_structure_l2744_274474


namespace NUMINAMATH_CALUDE_first_equation_value_l2744_274411

theorem first_equation_value (x y a : ℝ) 
  (eq1 : 2 * x + y = a) 
  (eq2 : x + 2 * y = 10) 
  (eq3 : (x + y) / 3 = 4) : 
  a = 12 := by
sorry

end NUMINAMATH_CALUDE_first_equation_value_l2744_274411


namespace NUMINAMATH_CALUDE_modulus_equality_necessary_not_sufficient_l2744_274414

variable {E : Type*} [NormedAddCommGroup E] [InnerProductSpace ℝ E]

theorem modulus_equality_necessary_not_sufficient :
  (∀ (u v : E), u ≠ 0 ∧ v ≠ 0 → (‖u‖ = ‖v‖ → u = v) ↔ false) ∧
  (∀ (u v : E), u ≠ 0 ∧ v ≠ 0 → (u = v → ‖u‖ = ‖v‖)) :=
by sorry

end NUMINAMATH_CALUDE_modulus_equality_necessary_not_sufficient_l2744_274414


namespace NUMINAMATH_CALUDE_mowing_area_calculation_l2744_274404

/-- Given that 3 mowers can mow 3 hectares in 3 days, 
    this theorem proves that 5 mowers can mow 25/3 hectares in 5 days. -/
theorem mowing_area_calculation 
  (mowers_initial : ℕ) 
  (days_initial : ℕ) 
  (area_initial : ℚ) 
  (mowers_final : ℕ) 
  (days_final : ℕ) 
  (h1 : mowers_initial = 3) 
  (h2 : days_initial = 3) 
  (h3 : area_initial = 3) 
  (h4 : mowers_final = 5) 
  (h5 : days_final = 5) :
  (area_initial * mowers_final * days_final) / (mowers_initial * days_initial) = 25 / 3 := by
  sorry

#check mowing_area_calculation

end NUMINAMATH_CALUDE_mowing_area_calculation_l2744_274404


namespace NUMINAMATH_CALUDE_newspapers_collected_l2744_274459

/-- The number of newspapers collected by Chris and Lily -/
theorem newspapers_collected (chris_newspapers lily_newspapers : ℕ) 
  (h1 : chris_newspapers = 42)
  (h2 : lily_newspapers = 23) :
  chris_newspapers + lily_newspapers = 65 := by
  sorry

end NUMINAMATH_CALUDE_newspapers_collected_l2744_274459


namespace NUMINAMATH_CALUDE_find_m_value_l2744_274493

theorem find_m_value (m : ℤ) : 
  (∃ (a b : ℝ), ∀ (x : ℝ), (2/3 * (m + 4) * x^(|m| - 3) + 6 > 0) ↔ (a * x + b > 0)) →
  (m + 4 ≠ 0) →
  m = 4 := by
sorry

end NUMINAMATH_CALUDE_find_m_value_l2744_274493


namespace NUMINAMATH_CALUDE_worker_distance_at_explosion_l2744_274429

/-- The time in seconds when the bomb explodes -/
def bomb_time : ℝ := 45

/-- The speed of the worker in yards per second -/
def worker_speed : ℝ := 6

/-- The speed of sound in feet per second -/
def sound_speed : ℝ := 1100

/-- Conversion factor from yards to feet -/
def yards_to_feet : ℝ := 3

/-- The distance run by the worker after t seconds, in feet -/
def worker_distance (t : ℝ) : ℝ := worker_speed * yards_to_feet * t

/-- The distance traveled by sound after the bomb explodes, in feet -/
def sound_distance (t : ℝ) : ℝ := sound_speed * (t - bomb_time)

/-- The time when the worker hears the explosion -/
noncomputable def explosion_time : ℝ := 
  (sound_speed * bomb_time) / (sound_speed - worker_speed * yards_to_feet)

/-- The theorem stating that the worker runs approximately 275 yards when he hears the explosion -/
theorem worker_distance_at_explosion : 
  ∃ ε > 0, abs (worker_distance explosion_time / yards_to_feet - 275) < ε :=
sorry

end NUMINAMATH_CALUDE_worker_distance_at_explosion_l2744_274429


namespace NUMINAMATH_CALUDE_problem_proof_l2744_274456

theorem problem_proof : (-2)^0 - 3 * Real.tan (30 * π / 180) - |Real.sqrt 3 - 2| = -1 := by
  sorry

end NUMINAMATH_CALUDE_problem_proof_l2744_274456


namespace NUMINAMATH_CALUDE_rotation_exists_l2744_274446

-- Define a 3D point
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

-- Define a triangle in 3D space
structure Triangle3D where
  A : Point3D
  O : Point3D
  B : Point3D

-- Define congruence for triangles
def congruent (t1 t2 : Triangle3D) : Prop :=
  (t1.A.x - t1.O.x)^2 + (t1.A.y - t1.O.y)^2 + (t1.A.z - t1.O.z)^2 =
    (t2.A.x - t2.O.x)^2 + (t2.A.y - t2.O.y)^2 + (t2.A.z - t2.O.z)^2 ∧
  (t1.B.x - t1.O.x)^2 + (t1.B.y - t1.O.y)^2 + (t1.B.z - t1.O.z)^2 =
    (t2.B.x - t2.O.x)^2 + (t2.B.y - t2.O.y)^2 + (t2.B.z - t2.O.z)^2

-- Define when two triangles are not in the same plane
def not_coplanar (t1 t2 : Triangle3D) : Prop :=
  ¬ ∃ (a b c d : ℝ),
    a * (t1.A.x - t1.O.x) + b * (t1.A.y - t1.O.y) + c * (t1.A.z - t1.O.z) + d = 0 ∧
    a * (t1.B.x - t1.O.x) + b * (t1.B.y - t1.O.y) + c * (t1.B.z - t1.O.z) + d = 0 ∧
    a * (t2.A.x - t2.O.x) + b * (t2.A.y - t2.O.y) + c * (t2.A.z - t2.O.z) + d = 0 ∧
    a * (t2.B.x - t2.O.x) + b * (t2.B.y - t2.O.y) + c * (t2.B.z - t2.O.z) + d = 0

-- Define rotation in 3D space
structure Rotation3D where
  axis : Point3D
  angle : ℝ

-- Theorem statement
theorem rotation_exists (t1 t2 : Triangle3D)
  (h1 : congruent t1 t2)
  (h2 : t1.O = t2.O)
  (h3 : not_coplanar t1 t2) :
  ∃ (r : Rotation3D), r.axis.x * (t1.A.x - t1.O.x) + r.axis.y * (t1.A.y - t1.O.y) + r.axis.z * (t1.A.z - t1.O.z) = 0 ∧
                      r.axis.x * (t1.B.x - t1.O.x) + r.axis.y * (t1.B.y - t1.O.y) + r.axis.z * (t1.B.z - t1.O.z) = 0 :=
by sorry

end NUMINAMATH_CALUDE_rotation_exists_l2744_274446


namespace NUMINAMATH_CALUDE_pradeep_marks_l2744_274422

theorem pradeep_marks (total_marks : ℕ) (pass_percentage : ℚ) (fail_margin : ℕ) (obtained_marks : ℕ) : 
  total_marks = 550 →
  pass_percentage = 40 / 100 →
  obtained_marks = (pass_percentage * total_marks).floor - fail_margin →
  obtained_marks = 200 := by
sorry

end NUMINAMATH_CALUDE_pradeep_marks_l2744_274422


namespace NUMINAMATH_CALUDE_total_pencils_l2744_274433

theorem total_pencils (jessica_pencils sandy_pencils jason_pencils : ℕ) 
  (h1 : jessica_pencils = 8)
  (h2 : sandy_pencils = 8)
  (h3 : jason_pencils = 8) :
  jessica_pencils + sandy_pencils + jason_pencils = 24 :=
by sorry

end NUMINAMATH_CALUDE_total_pencils_l2744_274433


namespace NUMINAMATH_CALUDE_gcd_of_B_is_five_l2744_274495

/-- The set of all numbers that can be represented as the sum of five consecutive positive integers -/
def B : Set ℕ := {n : ℕ | ∃ y : ℕ, y > 0 ∧ n = (y - 2) + (y - 1) + y + (y + 1) + (y + 2)}

/-- The greatest common divisor of all numbers in B is 5 -/
theorem gcd_of_B_is_five : ∃ d : ℕ, d > 0 ∧ (∀ n ∈ B, d ∣ n) ∧ (∀ m : ℕ, (∀ n ∈ B, m ∣ n) → m ∣ d) ∧ d = 5 := by
  sorry

end NUMINAMATH_CALUDE_gcd_of_B_is_five_l2744_274495


namespace NUMINAMATH_CALUDE_sum_congruence_mod_9_l2744_274494

theorem sum_congruence_mod_9 : 
  (2 + 33 + 444 + 5555 + 66666 + 777777 + 8888888 + 99999999) % 9 = 6 := by
  sorry

end NUMINAMATH_CALUDE_sum_congruence_mod_9_l2744_274494


namespace NUMINAMATH_CALUDE_bud_is_eight_years_old_l2744_274420

def buds_age (uncle_age : ℕ) : ℕ :=
  uncle_age / 3

theorem bud_is_eight_years_old (uncle_age : ℕ) (h : uncle_age = 24) :
  buds_age uncle_age = 8 := by
  sorry

end NUMINAMATH_CALUDE_bud_is_eight_years_old_l2744_274420


namespace NUMINAMATH_CALUDE_apple_products_cost_l2744_274483

/-- Calculate the final cost of Apple products after discounts, taxes, and cashback -/
theorem apple_products_cost (iphone_price iwatch_price ipad_price : ℝ)
  (iphone_discount iwatch_discount ipad_discount : ℝ)
  (iphone_tax iwatch_tax ipad_tax : ℝ)
  (cashback : ℝ)
  (h1 : iphone_price = 800)
  (h2 : iwatch_price = 300)
  (h3 : ipad_price = 500)
  (h4 : iphone_discount = 0.15)
  (h5 : iwatch_discount = 0.10)
  (h6 : ipad_discount = 0.05)
  (h7 : iphone_tax = 0.07)
  (h8 : iwatch_tax = 0.05)
  (h9 : ipad_tax = 0.06)
  (h10 : cashback = 0.02) :
  ∃ (total_cost : ℝ), 
    abs (total_cost - 1484.31) < 0.01 ∧
    total_cost = 
      (1 - cashback) * 
      ((iphone_price * (1 - iphone_discount) * (1 + iphone_tax)) +
       (iwatch_price * (1 - iwatch_discount) * (1 + iwatch_tax)) +
       (ipad_price * (1 - ipad_discount) * (1 + ipad_tax))) :=
by sorry


end NUMINAMATH_CALUDE_apple_products_cost_l2744_274483


namespace NUMINAMATH_CALUDE_function_upper_bound_condition_l2744_274423

theorem function_upper_bound_condition (a : ℝ) (h_a : a > 0) :
  (∀ x : ℝ, x ∈ Set.Icc 0 1 → a * x - x^2 ≤ 1) ↔ a ≤ 2 :=
by sorry

end NUMINAMATH_CALUDE_function_upper_bound_condition_l2744_274423


namespace NUMINAMATH_CALUDE_max_value_quadratic_l2744_274443

theorem max_value_quadratic (x y : ℝ) (hx : x > 0) (hy : y > 0) 
  (h : x^2 - 3*x*y + 4*y^2 = 12) : 
  x^2 + 3*x*y + 4*y^2 ≤ 84 ∧ ∃ (x₀ y₀ : ℝ), x₀ > 0 ∧ y₀ > 0 ∧ 
  x₀^2 - 3*x₀*y₀ + 4*y₀^2 = 12 ∧ x₀^2 + 3*x₀*y₀ + 4*y₀^2 = 84 := by
  sorry

end NUMINAMATH_CALUDE_max_value_quadratic_l2744_274443


namespace NUMINAMATH_CALUDE_division_problem_l2744_274486

theorem division_problem (dividend : ℕ) (divisor : ℕ) (remainder : ℕ) (quotient : ℕ) :
  dividend = 217 →
  divisor = 4 →
  remainder = 1 →
  dividend = divisor * quotient + remainder →
  quotient = 54 := by
sorry

end NUMINAMATH_CALUDE_division_problem_l2744_274486


namespace NUMINAMATH_CALUDE_volume_increase_l2744_274427

/-- Represents a rectangular solid -/
structure RectangularSolid where
  baseArea : ℝ
  height : ℝ

/-- Calculates the volume of a rectangular solid -/
def volume (solid : RectangularSolid) : ℝ :=
  solid.baseArea * solid.height

/-- Theorem: Increase in volume of a rectangular solid -/
theorem volume_increase (solid : RectangularSolid) 
  (h1 : solid.baseArea = 12)
  (h2 : 5 > 0) :
  volume { baseArea := solid.baseArea, height := solid.height + 5 } - volume solid = 60 := by
  sorry

end NUMINAMATH_CALUDE_volume_increase_l2744_274427


namespace NUMINAMATH_CALUDE_least_common_multiple_of_band_sets_l2744_274435

theorem least_common_multiple_of_band_sets : Nat.lcm (Nat.lcm 2 9) 14 = 126 := by
  sorry

end NUMINAMATH_CALUDE_least_common_multiple_of_band_sets_l2744_274435


namespace NUMINAMATH_CALUDE_f_of_f_3_l2744_274406

-- Define the piecewise function f
noncomputable def f (x : ℝ) : ℝ :=
  if x ≤ 1 then x^2 + 1 else 2/x

-- Theorem statement
theorem f_of_f_3 : f (f 3) = 13/9 := by
  sorry

end NUMINAMATH_CALUDE_f_of_f_3_l2744_274406


namespace NUMINAMATH_CALUDE_smallest_number_in_S_l2744_274482

def S : Set ℝ := {3.2, 2.3, 3, 2.23, 3.22}

theorem smallest_number_in_S : 
  ∃ (x : ℝ), x ∈ S ∧ ∀ y ∈ S, x ≤ y ∧ x = 2.23 := by
  sorry

end NUMINAMATH_CALUDE_smallest_number_in_S_l2744_274482


namespace NUMINAMATH_CALUDE_sufficient_condition_product_greater_than_one_l2744_274496

theorem sufficient_condition_product_greater_than_one :
  ∀ (a b : ℝ), a > 1 → b > 1 → a * b > 1 := by
  sorry

end NUMINAMATH_CALUDE_sufficient_condition_product_greater_than_one_l2744_274496


namespace NUMINAMATH_CALUDE_square_sum_product_l2744_274473

theorem square_sum_product (x : ℝ) :
  (Real.sqrt (9 + x) + Real.sqrt (16 - x) = 8) →
  ((9 + x) * (16 - x) = 380.25) := by
  sorry

end NUMINAMATH_CALUDE_square_sum_product_l2744_274473


namespace NUMINAMATH_CALUDE_wildflower_color_difference_l2744_274462

/-- Given the following conditions about wildflowers:
  - Total wildflowers picked: 44
  - Yellow and white flowers: 13
  - Red and yellow flowers: 17
  - Red and white flowers: 14
Prove that there are 4 more flowers containing red than containing white. -/
theorem wildflower_color_difference 
  (total : ℕ) 
  (yellow_white : ℕ) 
  (red_yellow : ℕ) 
  (red_white : ℕ) 
  (h_total : total = 44)
  (h_yellow_white : yellow_white = 13)
  (h_red_yellow : red_yellow = 17)
  (h_red_white : red_white = 14) :
  (red_yellow + red_white) - (yellow_white + red_white) = 4 :=
by sorry

end NUMINAMATH_CALUDE_wildflower_color_difference_l2744_274462


namespace NUMINAMATH_CALUDE_intersection_theorem_l2744_274428

def M : Set ℝ := {x | 0 ≤ x ∧ x ≤ 2}

def N : Set ℝ := {x | ∃ y, y = Real.log (1 - x^2)}

theorem intersection_theorem : M ∩ N = {x | 0 ≤ x ∧ x < 1} := by
  sorry

end NUMINAMATH_CALUDE_intersection_theorem_l2744_274428


namespace NUMINAMATH_CALUDE_proposition_false_negation_true_l2744_274499

-- Define the properties of a quadrilateral
structure Quadrilateral :=
  (has_one_pair_parallel_sides : Bool)
  (has_one_pair_equal_sides : Bool)
  (is_parallelogram : Bool)

-- Define the proposition
def proposition (q : Quadrilateral) : Prop :=
  q.has_one_pair_parallel_sides ∧ q.has_one_pair_equal_sides → q.is_parallelogram

-- Define the negation of the proposition
def negation_proposition (q : Quadrilateral) : Prop :=
  q.has_one_pair_parallel_sides ∧ q.has_one_pair_equal_sides ∧ ¬q.is_parallelogram

-- Theorem stating that the proposition is false and its negation is true
theorem proposition_false_negation_true :
  (∃ q : Quadrilateral, ¬(proposition q)) ∧
  (∀ q : Quadrilateral, negation_proposition q → True) :=
sorry

end NUMINAMATH_CALUDE_proposition_false_negation_true_l2744_274499


namespace NUMINAMATH_CALUDE_range_of_a_l2744_274425

/-- The function f(x) defined in the problem -/
def f (a : ℝ) (x : ℝ) : ℝ := |x - 1| + |2*x - a|

/-- The theorem stating the range of a -/
theorem range_of_a (a : ℝ) :
  (∀ x : ℝ, f a x ≥ (1/4)*a^2 + 1) → a ∈ Set.Icc (-2) 0 :=
by sorry

end NUMINAMATH_CALUDE_range_of_a_l2744_274425


namespace NUMINAMATH_CALUDE_sine_inequality_l2744_274475

theorem sine_inequality (n : ℕ+) (θ : ℝ) : |Real.sin (n * θ)| ≤ n * |Real.sin θ| := by
  sorry

end NUMINAMATH_CALUDE_sine_inequality_l2744_274475


namespace NUMINAMATH_CALUDE_rotation_center_l2744_274405

theorem rotation_center (f : ℂ → ℂ) (c : ℂ) : 
  (f = fun z ↦ ((1 - Complex.I * Real.sqrt 2) * z + (-4 * Real.sqrt 2 + 6 * Complex.I)) / 2) →
  (c = (2 * Real.sqrt 2) / 3 - (2 * Complex.I) / 3) →
  f c = c := by
sorry

end NUMINAMATH_CALUDE_rotation_center_l2744_274405


namespace NUMINAMATH_CALUDE_school_field_trip_cost_l2744_274458

/-- Calculates the total cost for a school field trip to a farm -/
theorem school_field_trip_cost (num_students : ℕ) (num_adults : ℕ) 
  (student_fee : ℕ) (adult_fee : ℕ) : 
  num_students = 35 → num_adults = 4 → student_fee = 5 → adult_fee = 6 →
  num_students * student_fee + num_adults * adult_fee = 199 :=
by sorry

end NUMINAMATH_CALUDE_school_field_trip_cost_l2744_274458


namespace NUMINAMATH_CALUDE_nonnegative_root_condition_l2744_274436

/-- A polynomial of degree 4 with coefficient q -/
def polynomial (q : ℝ) (x : ℝ) : ℝ := x^4 + q*x^3 + x^2 + q*x + 4

/-- The condition for the existence of a non-negative real root -/
def has_nonnegative_root (q : ℝ) : Prop :=
  ∃ x : ℝ, x ≥ 0 ∧ polynomial q x = 0

/-- The theorem stating the condition on q for the existence of a non-negative root -/
theorem nonnegative_root_condition (q : ℝ) : 
  has_nonnegative_root q ↔ q ≤ -2 * Real.sqrt 2 :=
sorry

end NUMINAMATH_CALUDE_nonnegative_root_condition_l2744_274436


namespace NUMINAMATH_CALUDE_age_difference_robert_elizabeth_l2744_274467

theorem age_difference_robert_elizabeth : 
  ∀ (robert_age patrick_age elizabeth_age : ℕ),
  robert_age = 28 →
  patrick_age = robert_age / 2 →
  elizabeth_age = patrick_age - 4 →
  robert_age - elizabeth_age = 18 :=
by
  sorry

end NUMINAMATH_CALUDE_age_difference_robert_elizabeth_l2744_274467


namespace NUMINAMATH_CALUDE_marble_sculpture_weight_l2744_274464

theorem marble_sculpture_weight (original_weight : ℝ) : 
  original_weight > 0 →
  (0.75 * (0.80 * (0.70 * original_weight))) = 105 →
  original_weight = 250 :=
by
  sorry

end NUMINAMATH_CALUDE_marble_sculpture_weight_l2744_274464


namespace NUMINAMATH_CALUDE_equation_solutions_l2744_274408

theorem equation_solutions :
  (∀ x : ℝ, x^2 - 6*x = -1 ↔ x = 3 - 2*Real.sqrt 2 ∨ x = 3 + 2*Real.sqrt 2) ∧
  (∀ x : ℝ, x*(2*x - 1) = 2*(2*x - 1) ↔ x = 1/2 ∨ x = 2) := by
  sorry

end NUMINAMATH_CALUDE_equation_solutions_l2744_274408


namespace NUMINAMATH_CALUDE_car_wash_earnings_l2744_274407

theorem car_wash_earnings (friday_earnings : ℕ) (x : ℚ) : 
  friday_earnings = 147 →
  friday_earnings + (friday_earnings * x + 7) + (friday_earnings + 78) = 673 →
  x = 2 := by
sorry

end NUMINAMATH_CALUDE_car_wash_earnings_l2744_274407


namespace NUMINAMATH_CALUDE_inequality_and_equality_condition_l2744_274409

theorem inequality_and_equality_condition (x y : ℝ) 
  (hx : x > -1) (hy : y > -1) (hsum : x + y = 1) :
  (x / (y + 1) + y / (x + 1) ≥ 2 / 3) ∧
  (x / (y + 1) + y / (x + 1) = 2 / 3 ↔ x = 1 / 2 ∧ y = 1 / 2) := by
  sorry

end NUMINAMATH_CALUDE_inequality_and_equality_condition_l2744_274409


namespace NUMINAMATH_CALUDE_circleplus_two_three_l2744_274447

-- Define the operation ⊕
def circleplus (a b : ℝ) : ℝ := 4 * a + 5 * b

-- State the theorem
theorem circleplus_two_three : circleplus 2 3 = 23 := by
  sorry

end NUMINAMATH_CALUDE_circleplus_two_three_l2744_274447


namespace NUMINAMATH_CALUDE_batsman_average_after_12th_innings_l2744_274498

/-- Represents a batsman's cricket statistics -/
structure BatsmanStats where
  innings : ℕ
  totalRuns : ℕ
  average : ℚ

/-- Calculates the new average after an innings -/
def newAverage (stats : BatsmanStats) (runsScored : ℕ) : ℚ :=
  (stats.totalRuns + runsScored) / (stats.innings + 1)

/-- Theorem: A batsman's average after 12 innings is 64, given the conditions -/
theorem batsman_average_after_12th_innings
  (stats : BatsmanStats)
  (h1 : stats.innings = 11)
  (h2 : newAverage stats 75 = stats.average + 1) :
  newAverage stats 75 = 64 := by
  sorry

end NUMINAMATH_CALUDE_batsman_average_after_12th_innings_l2744_274498


namespace NUMINAMATH_CALUDE_tan_210_degrees_l2744_274497

theorem tan_210_degrees : Real.tan (210 * Real.pi / 180) = -Real.sqrt 3 / 3 := by
  sorry

end NUMINAMATH_CALUDE_tan_210_degrees_l2744_274497


namespace NUMINAMATH_CALUDE_range_of_f_l2744_274455

def f (x : ℝ) := x^4 - 4*x^2 + 4

theorem range_of_f :
  Set.range f = Set.Ici (0 : ℝ) := by sorry

end NUMINAMATH_CALUDE_range_of_f_l2744_274455


namespace NUMINAMATH_CALUDE_bertrand_odd_conjecture_counterexample_l2744_274432

-- Define what we mean by a "large" number
def isLarge (n : ℕ) : Prop := n ≥ 100

-- Define an odd number
def isOdd (n : ℕ) : Prop := ∃ k, n = 2*k + 1

-- Define a prime number
def isPrime (p : ℕ) : Prop := p > 1 ∧ ∀ d, d ∣ p → d = 1 ∨ d = p

-- Bertrand's Odd Conjecture
def bertrandOddConjecture : Prop := 
  ∀ n, isLarge n → isOdd n → 
    ∃ p q r, isPrime p ∧ isPrime q ∧ isPrime r ∧ 
             isOdd p ∧ isOdd q ∧ isOdd r ∧
             n = p + q + r

-- Theorem: There exists a counterexample to Bertrand's Odd Conjecture
theorem bertrand_odd_conjecture_counterexample :
  ∃ n, isLarge n ∧ isOdd n ∧ 
    ¬(∃ p q r, isPrime p ∧ isPrime q ∧ isPrime r ∧ 
               isOdd p ∧ isOdd q ∧ isOdd r ∧
               n = p + q + r) :=
by sorry

end NUMINAMATH_CALUDE_bertrand_odd_conjecture_counterexample_l2744_274432


namespace NUMINAMATH_CALUDE_large_hexagon_area_l2744_274401

/-- Represents a regular hexagon -/
structure RegularHexagon where
  area : ℝ

/-- The large regular hexagon containing smaller hexagons -/
def large_hexagon : RegularHexagon := sorry

/-- One of the smaller regular hexagons -/
def small_hexagon : RegularHexagon := sorry

/-- The number of small hexagons in the large hexagon -/
def num_small_hexagons : ℕ := 7

/-- The number of small hexagons in the shaded area -/
def num_shaded_hexagons : ℕ := 6

/-- The area of the shaded part (6 small hexagons) -/
def shaded_area : ℝ := 180

theorem large_hexagon_area : large_hexagon.area = 270 := by sorry

end NUMINAMATH_CALUDE_large_hexagon_area_l2744_274401


namespace NUMINAMATH_CALUDE_mixture_ratio_proof_l2744_274461

theorem mixture_ratio_proof (p q : ℝ) : 
  p + q = 35 →
  p / (q + 13) = 5 / 7 →
  p / q = 4 / 3 :=
by sorry

end NUMINAMATH_CALUDE_mixture_ratio_proof_l2744_274461


namespace NUMINAMATH_CALUDE_isosceles_right_triangle_exists_l2744_274489

/-- A coloring of an infinite grid using three colors -/
def GridColoring := ℤ × ℤ → Fin 3

/-- An isosceles right triangle on the grid -/
structure IsoscelesRightTriangle where
  a : ℤ × ℤ
  b : ℤ × ℤ
  c : ℤ × ℤ
  is_right : (b.1 - a.1) * (c.1 - a.1) + (b.2 - a.2) * (c.2 - a.2) = 0
  is_isosceles : (b.1 - a.1)^2 + (b.2 - a.2)^2 = (c.1 - a.1)^2 + (c.2 - a.2)^2

/-- The main theorem: In any three-coloring of an infinite grid, 
    there exists an isosceles right triangle with vertices of the same color -/
theorem isosceles_right_triangle_exists (coloring : GridColoring) : 
  ∃ t : IsoscelesRightTriangle, 
    coloring t.a = coloring t.b ∧ coloring t.b = coloring t.c :=
sorry

end NUMINAMATH_CALUDE_isosceles_right_triangle_exists_l2744_274489


namespace NUMINAMATH_CALUDE_beads_left_in_container_l2744_274460

/-- The number of beads left in a container after some are removed -/
theorem beads_left_in_container (green brown red removed : ℕ) : 
  green = 1 → brown = 2 → red = 3 → removed = 2 →
  green + brown + red - removed = 4 := by
  sorry

end NUMINAMATH_CALUDE_beads_left_in_container_l2744_274460


namespace NUMINAMATH_CALUDE_not_both_perfect_squares_l2744_274445

/-- For any natural numbers x and y, at least one of x^2 + y + 1 or y^2 + 4x + 3 is not a perfect square. -/
theorem not_both_perfect_squares (x y : ℕ) : 
  ¬(∃ a b : ℕ, (x^2 + y + 1 = a^2) ∧ (y^2 + 4*x + 3 = b^2)) := by
  sorry

end NUMINAMATH_CALUDE_not_both_perfect_squares_l2744_274445


namespace NUMINAMATH_CALUDE_sum_of_roots_l2744_274485

theorem sum_of_roots (x y : ℝ) 
  (hx : x^3 - 3*x^2 + 5*x - 17 = 0)
  (hy : y^3 - 3*y^2 + 5*y + 11 = 0) : 
  x + y = 2 := by sorry

end NUMINAMATH_CALUDE_sum_of_roots_l2744_274485


namespace NUMINAMATH_CALUDE_abc_inequality_l2744_274421

theorem abc_inequality (a b c t : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  a * b * c * (a^t + b^t + c^t) ≥ 
  a^(t+2) * (-a + b + c) + b^(t+2) * (a - b + c) + c^(t+2) * (a + b - c) ∧
  (a * b * c * (a^t + b^t + c^t) = 
   a^(t+2) * (-a + b + c) + b^(t+2) * (a - b + c) + c^(t+2) * (a + b - c) ↔ 
   a = b ∧ b = c) :=
by sorry

end NUMINAMATH_CALUDE_abc_inequality_l2744_274421


namespace NUMINAMATH_CALUDE_rods_in_one_mile_l2744_274481

/-- Conversion factor from miles to chains -/
def mile_to_chain : ℚ := 10

/-- Conversion factor from chains to rods -/
def chain_to_rod : ℚ := 22

/-- The number of rods in one mile -/
def rods_in_mile : ℚ := mile_to_chain * chain_to_rod

theorem rods_in_one_mile :
  rods_in_mile = 220 :=
by sorry

end NUMINAMATH_CALUDE_rods_in_one_mile_l2744_274481


namespace NUMINAMATH_CALUDE_triangle_angle_inequalities_l2744_274417

theorem triangle_angle_inequalities (α β γ : Real) 
  (h : α + β + γ = π) : 
  (Real.sin (α/2) * Real.sin (β/2) * Real.sin (γ/2) ≤ 1/8) ∧
  (Real.cos α * Real.cos β * Real.cos γ ≤ 1/8) := by
  sorry

end NUMINAMATH_CALUDE_triangle_angle_inequalities_l2744_274417


namespace NUMINAMATH_CALUDE_cube_dimension_reduction_l2744_274426

theorem cube_dimension_reduction (initial_face_area : ℝ) (reduction : ℝ) : 
  initial_face_area = 36 ∧ reduction = 1 → 
  (3 : ℝ) * (Real.sqrt initial_face_area - reduction) = 15 := by
  sorry

end NUMINAMATH_CALUDE_cube_dimension_reduction_l2744_274426


namespace NUMINAMATH_CALUDE_special_circle_distances_l2744_274466

/-- A circle with specific properties and a point on its circumference -/
structure SpecialCircle where
  r : ℕ
  u : ℕ
  v : ℕ
  p : ℕ
  q : ℕ
  m : ℕ
  n : ℕ
  h_r_odd : Odd r
  h_circle_eq : u^2 + v^2 = r^2
  h_u_prime_power : u = p^m
  h_v_prime_power : v = q^n
  h_p_prime : Nat.Prime p
  h_q_prime : Nat.Prime q
  h_u_gt_v : u > v

/-- The theorem to be proved -/
theorem special_circle_distances (c : SpecialCircle) :
  let A : ℝ × ℝ := (c.r, 0)
  let B : ℝ × ℝ := (-c.r, 0)
  let C : ℝ × ℝ := (0, -c.r)
  let D : ℝ × ℝ := (0, c.r)
  let P : ℝ × ℝ := (c.u, c.v)
  let M : ℝ × ℝ := (c.u, 0)
  let N : ℝ × ℝ := (0, c.v)
  |A.1 - M.1| = 1 ∧
  |B.1 - M.1| = 9 ∧
  |C.2 - N.2| = 8 ∧
  |D.2 - N.2| = 2 :=
by sorry

end NUMINAMATH_CALUDE_special_circle_distances_l2744_274466


namespace NUMINAMATH_CALUDE_mothers_full_time_proportion_l2744_274453

/-- The proportion of mothers holding full-time jobs -/
def proportion_mothers_full_time : ℝ := sorry

/-- The proportion of fathers holding full-time jobs -/
def proportion_fathers_full_time : ℝ := 0.75

/-- The proportion of parents who are women -/
def proportion_women : ℝ := 0.4

/-- The proportion of parents who do not hold full-time jobs -/
def proportion_not_full_time : ℝ := 0.19

theorem mothers_full_time_proportion :
  proportion_mothers_full_time = 0.9 :=
by sorry

end NUMINAMATH_CALUDE_mothers_full_time_proportion_l2744_274453


namespace NUMINAMATH_CALUDE_linear_equation_power_l2744_274440

/-- If $2x^{n-3}-\frac{1}{3}y^{2m+1}=0$ is a linear equation in $x$ and $y$, then $n^m = 1$. -/
theorem linear_equation_power (n m : ℕ) :
  (∀ x y : ℝ, ∃ a b c : ℝ, 2 * x^(n-3) - (1/3) * y^(2*m+1) = a * x + b * y + c) →
  n^m = 1 := by
  sorry

end NUMINAMATH_CALUDE_linear_equation_power_l2744_274440


namespace NUMINAMATH_CALUDE_perseverance_permutations_count_l2744_274454

/-- The number of letters in "PERSEVERANCE" -/
def word_length : ℕ := 11

/-- The number of occurrences of 'E' in "PERSEVERANCE" -/
def e_count : ℕ := 3

/-- The number of occurrences of 'R' in "PERSEVERANCE" -/
def r_count : ℕ := 2

/-- The number of unique permutations of the letters in "PERSEVERANCE" -/
def perseverance_permutations : ℕ := word_length.factorial / (e_count.factorial * r_count.factorial * r_count.factorial)

theorem perseverance_permutations_count :
  perseverance_permutations = 1663200 :=
by sorry

end NUMINAMATH_CALUDE_perseverance_permutations_count_l2744_274454


namespace NUMINAMATH_CALUDE_purely_imaginary_complex_number_l2744_274444

theorem purely_imaginary_complex_number (x : ℝ) : 
  (Complex.ofReal (x^2 - 1) + Complex.I * Complex.ofReal (x + 1)).im ≠ 0 ∧
  (Complex.ofReal (x^2 - 1) + Complex.I * Complex.ofReal (x + 1)).re = 0 →
  x = 1 := by
sorry

end NUMINAMATH_CALUDE_purely_imaginary_complex_number_l2744_274444


namespace NUMINAMATH_CALUDE_symmetry_properties_l2744_274410

/-- A point in a 2D plane. -/
structure Point2D where
  x : ℝ
  y : ℝ

/-- Symmetry with respect to the x-axis. -/
def symmetryOX (p : Point2D) : Point2D :=
  ⟨p.x, -p.y⟩

/-- Symmetry with respect to the y-axis. -/
def symmetryOY (p : Point2D) : Point2D :=
  ⟨-p.x, p.y⟩

theorem symmetry_properties (p : Point2D) : 
  (symmetryOX p = ⟨p.x, -p.y⟩) ∧ (symmetryOY p = ⟨-p.x, p.y⟩) := by
  sorry

end NUMINAMATH_CALUDE_symmetry_properties_l2744_274410


namespace NUMINAMATH_CALUDE_orchestra_members_count_l2744_274416

theorem orchestra_members_count : ∃! n : ℕ, 
  150 < n ∧ n < 250 ∧ 
  n % 8 = 1 ∧ 
  n % 6 = 2 ∧ 
  n % 9 = 3 := by
  sorry

end NUMINAMATH_CALUDE_orchestra_members_count_l2744_274416


namespace NUMINAMATH_CALUDE_deck_width_proof_l2744_274471

/-- Proves that for a rectangular pool of 20 feet by 22 feet, surrounded by a deck of uniform width,
    if the total area of the pool and deck is 728 square feet, then the width of the deck is 3 feet. -/
theorem deck_width_proof (w : ℝ) : 
  (20 + 2*w) * (22 + 2*w) = 728 → w = 3 :=
by sorry

end NUMINAMATH_CALUDE_deck_width_proof_l2744_274471


namespace NUMINAMATH_CALUDE_parallelogram_area_parallelogram_area_is_15_l2744_274490

/-- Given a parallelogram with vertices (4,4), (7,4), (5,9), and (8,9) in a rectangular coordinate system,
    prove that its area is 15 square units. -/
theorem parallelogram_area : ℝ → Prop :=
  fun area =>
    let x₁ : ℝ := 4
    let y₁ : ℝ := 4
    let x₂ : ℝ := 7
    let y₂ : ℝ := 4
    let x₃ : ℝ := 5
    let y₃ : ℝ := 9
    let x₄ : ℝ := 8
    let y₄ : ℝ := 9
    let base : ℝ := x₂ - x₁
    let height : ℝ := y₃ - y₁
    area = base * height

/-- Proof of the parallelogram area theorem -/
theorem parallelogram_area_is_15 : parallelogram_area 15 := by
  sorry

end NUMINAMATH_CALUDE_parallelogram_area_parallelogram_area_is_15_l2744_274490


namespace NUMINAMATH_CALUDE_perfect_square_divisibility_l2744_274439

theorem perfect_square_divisibility (x y : ℕ+) (h : (2 * x * y) ∣ (x^2 + y^2 - x)) : 
  ∃ (n : ℕ+), x = n^2 := by
sorry

end NUMINAMATH_CALUDE_perfect_square_divisibility_l2744_274439


namespace NUMINAMATH_CALUDE_max_reflections_is_18_l2744_274437

/-- The angle between the lines in degrees -/
def angle : ℝ := 5

/-- The maximum angle for perpendicular reflection in degrees -/
def max_angle : ℝ := 90

/-- The maximum number of reflections -/
def max_reflections : ℕ := 18

/-- Theorem stating that the maximum number of reflections is 18 -/
theorem max_reflections_is_18 :
  ∀ n : ℕ, n * angle ≤ max_angle → n ≤ max_reflections :=
by sorry

end NUMINAMATH_CALUDE_max_reflections_is_18_l2744_274437


namespace NUMINAMATH_CALUDE_ball_selection_problem_l2744_274484

/-- The number of red balls in the box -/
def red_balls : ℕ := 12

/-- The number of blue balls in the box -/
def blue_balls : ℕ := 7

/-- The total number of balls in the box -/
def total_balls : ℕ := red_balls + blue_balls

/-- The number of ways to select 3 red balls and 2 blue balls -/
def ways_to_select : ℕ := Nat.choose red_balls 3 * Nat.choose blue_balls 2

/-- The probability of drawing 2 blue balls first, then 1 red ball -/
def prob_draw : ℚ :=
  (Nat.choose blue_balls 2 * Nat.choose red_balls 1) / Nat.choose total_balls 3

/-- The final result -/
theorem ball_selection_problem :
  ways_to_select * prob_draw = 388680 / 323 := by sorry

end NUMINAMATH_CALUDE_ball_selection_problem_l2744_274484


namespace NUMINAMATH_CALUDE_geometric_series_sum_l2744_274477

/-- Sum of a finite geometric series -/
def geometricSum (a : ℚ) (r : ℚ) (n : ℕ) : ℚ :=
  a * (1 - r^n) / (1 - r)

/-- The sum of the first 6 terms of the geometric series with first term 1/4 and common ratio 1/4 -/
theorem geometric_series_sum :
  geometricSum (1/4 : ℚ) (1/4 : ℚ) 6 = 4095/12288 := by
  sorry

end NUMINAMATH_CALUDE_geometric_series_sum_l2744_274477


namespace NUMINAMATH_CALUDE_inequality_proof_l2744_274487

theorem inequality_proof (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) 
  (sum_one : a + b + c = 1) : 
  10 * (a^3 + b^3 + c^3) - 9 * (a^5 + b^5 + c^5) ≥ 1 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l2744_274487


namespace NUMINAMATH_CALUDE_cut_prism_faces_cut_prism_faces_proof_l2744_274431

/-- A triangular prism with 9 edges -/
structure TriangularPrism :=
  (edges : ℕ)
  (edges_eq : edges = 9)

/-- The result of cutting a triangular prism parallel to its base from the midpoints of its side edges -/
structure CutPrism extends TriangularPrism :=
  (additional_faces : ℕ)
  (additional_faces_eq : additional_faces = 3)

/-- The theorem stating that a cut triangular prism has 8 faces in total -/
theorem cut_prism_faces (cp : CutPrism) : ℕ :=
  8

#check cut_prism_faces

/-- Proof of the theorem -/
theorem cut_prism_faces_proof (cp : CutPrism) : cut_prism_faces cp = 8 := by
  sorry

end NUMINAMATH_CALUDE_cut_prism_faces_cut_prism_faces_proof_l2744_274431


namespace NUMINAMATH_CALUDE_computer_price_increase_l2744_274488

theorem computer_price_increase (y : ℝ) (h1 : 1.30 * y = 351) (h2 : 2 * y = 540) :
  2 * y = 540 := by
  sorry

end NUMINAMATH_CALUDE_computer_price_increase_l2744_274488


namespace NUMINAMATH_CALUDE_weight_of_b_l2744_274415

-- Define the weights as real numbers
variable (a b c : ℝ)

-- Define the conditions
def average_abc : Prop := (a + b + c) / 3 = 30
def average_ab : Prop := (a + b) / 2 = 25
def average_bc : Prop := (b + c) / 2 = 28

-- Theorem statement
theorem weight_of_b (h1 : average_abc a b c) (h2 : average_ab a b) (h3 : average_bc b c) : b = 16 := by
  sorry

end NUMINAMATH_CALUDE_weight_of_b_l2744_274415


namespace NUMINAMATH_CALUDE_cab_driver_income_l2744_274430

theorem cab_driver_income (income_day1 income_day2 income_day3 income_day4 : ℕ)
  (average_income : ℕ) (total_days : ℕ) :
  income_day1 = 200 →
  income_day2 = 150 →
  income_day3 = 750 →
  income_day4 = 400 →
  average_income = 400 →
  total_days = 5 →
  (income_day1 + income_day2 + income_day3 + income_day4 + 
    (average_income * total_days - (income_day1 + income_day2 + income_day3 + income_day4))) / total_days = average_income →
  average_income * total_days - (income_day1 + income_day2 + income_day3 + income_day4) = 500 :=
by sorry

end NUMINAMATH_CALUDE_cab_driver_income_l2744_274430


namespace NUMINAMATH_CALUDE_four_is_integer_l2744_274491

-- Define the set of natural numbers
def NaturalNumber : Type := ℕ

-- Define the set of integers
def Integer : Type := ℤ

-- Define the property that all natural numbers are integers
axiom natural_are_integers : ∀ (n : NaturalNumber), Integer

-- Define 4 as a natural number
axiom four_is_natural : NaturalNumber

-- Theorem to prove
theorem four_is_integer : Integer :=
  sorry

end NUMINAMATH_CALUDE_four_is_integer_l2744_274491


namespace NUMINAMATH_CALUDE_polynomial_divisibility_and_divisor_l2744_274442

theorem polynomial_divisibility_and_divisor (m : ℝ) : 
  (∀ x : ℝ, ∃ k : ℝ, 3 * x^2 + 5 * x + m = (x + 2) * k) → 
  m = -2 ∧ ∃ n : ℤ, 2 = m * n := by
  sorry

end NUMINAMATH_CALUDE_polynomial_divisibility_and_divisor_l2744_274442


namespace NUMINAMATH_CALUDE_ratio_odd_even_divisors_l2744_274451

def M : ℕ := 39 * 48 * 77 * 150

def sum_odd_divisors (n : ℕ) : ℕ := sorry
def sum_even_divisors (n : ℕ) : ℕ := sorry

theorem ratio_odd_even_divisors :
  (sum_odd_divisors M) * 62 = sum_even_divisors M :=
sorry

end NUMINAMATH_CALUDE_ratio_odd_even_divisors_l2744_274451


namespace NUMINAMATH_CALUDE_student_scores_average_l2744_274438

theorem student_scores_average (math physics chem : ℕ) : 
  math + physics = 30 → 
  chem = physics + 20 → 
  (math + chem) / 2 = 25 := by
sorry

end NUMINAMATH_CALUDE_student_scores_average_l2744_274438


namespace NUMINAMATH_CALUDE_line_passes_through_intercepts_l2744_274413

/-- A line that intersects the x-axis at (3, 0) and the y-axis at (0, -5) -/
def line_equation (x y : ℝ) : Prop :=
  x / 3 - y / 5 = 1

/-- The x-intercept of the line -/
def x_intercept : ℝ := 3

/-- The y-intercept of the line -/
def y_intercept : ℝ := -5

theorem line_passes_through_intercepts :
  line_equation x_intercept 0 ∧ line_equation 0 y_intercept := by
  sorry

end NUMINAMATH_CALUDE_line_passes_through_intercepts_l2744_274413


namespace NUMINAMATH_CALUDE_stewart_farm_ratio_l2744_274403

def stewart_farm (horse_food_per_day : ℕ) (total_horse_food : ℕ) (num_sheep : ℕ) : Prop :=
  ∃ (num_horses : ℕ),
    horse_food_per_day * num_horses = total_horse_food ∧
    (num_sheep : ℚ) / num_horses = 5 / 7

theorem stewart_farm_ratio :
  stewart_farm 230 12880 40 := by
  sorry

end NUMINAMATH_CALUDE_stewart_farm_ratio_l2744_274403


namespace NUMINAMATH_CALUDE_f_expression_l2744_274448

/-- A linear function f satisfying specific conditions -/
def f (x : ℝ) : ℝ := sorry

/-- f is a linear function -/
axiom f_linear : ∃ (a b : ℝ), ∀ x, f x = a * x + b

/-- f(-2) = -1 -/
axiom f_neg_two : f (-2) = -1

/-- f(0) + f(2) = 10 -/
axiom f_sum : f 0 + f 2 = 10

/-- Theorem: f(x) = 2x + 3 -/
theorem f_expression : ∀ x, f x = 2 * x + 3 := by sorry

end NUMINAMATH_CALUDE_f_expression_l2744_274448


namespace NUMINAMATH_CALUDE_probability_a1_or_b1_not_both_is_half_l2744_274434

-- Define the number of students with excellent grades in each subject
def math_students : ℕ := 3
def physics_students : ℕ := 2
def chemistry_students : ℕ := 2

-- Define the total number of possible team combinations
def total_combinations : ℕ := math_students * physics_students * chemistry_students

-- Define the number of combinations where either A₁ or B₁ is selected, but not both
def target_combinations : ℕ := 1 * 1 * chemistry_students + (math_students - 1) * 1 * chemistry_students

-- Theorem statement
theorem probability_a1_or_b1_not_both_is_half : 
  (target_combinations : ℚ) / total_combinations = 1 / 2 := by sorry

end NUMINAMATH_CALUDE_probability_a1_or_b1_not_both_is_half_l2744_274434


namespace NUMINAMATH_CALUDE_hyperbola_foci_l2744_274479

/-- The hyperbola equation -/
def hyperbola_equation (x y : ℝ) : Prop :=
  x^2 / 4 - y^2 / 5 = 1

/-- The coordinates of a focus of the hyperbola -/
def focus_coordinate : ℝ × ℝ := (3, 0)

/-- Theorem: The coordinates of the foci of the hyperbola x^2/4 - y^2/5 = 1 are (±3, 0) -/
theorem hyperbola_foci :
  (∀ x y, hyperbola_equation x y → 
    (x = focus_coordinate.1 ∧ y = focus_coordinate.2) ∨ 
    (x = -focus_coordinate.1 ∧ y = focus_coordinate.2)) :=
by sorry

end NUMINAMATH_CALUDE_hyperbola_foci_l2744_274479


namespace NUMINAMATH_CALUDE_quadratic_inequality_l2744_274465

theorem quadratic_inequality (x : ℝ) :
  9 * x^2 - 6 * x + 1 > 0 ↔ x < 1/3 ∨ x > 1/3 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_inequality_l2744_274465


namespace NUMINAMATH_CALUDE_basketball_team_grouping_probability_l2744_274480

theorem basketball_team_grouping_probability :
  let total_teams : ℕ := 7
  let group_size_1 : ℕ := 3
  let group_size_2 : ℕ := 4
  let specific_teams : ℕ := 2
  
  let total_arrangements : ℕ := (Nat.choose total_teams group_size_1) * (Nat.choose group_size_1 group_size_1) +
                                (Nat.choose total_teams group_size_2) * (Nat.choose group_size_2 group_size_2)
  
  let favorable_arrangements : ℕ := (Nat.choose specific_teams specific_teams) *
                                    ((Nat.choose (total_teams - specific_teams) (group_size_1 - specific_teams)) +
                                     (Nat.choose (total_teams - specific_teams) (group_size_2 - specific_teams))) *
                                    (Nat.factorial specific_teams)
  
  (favorable_arrangements : ℚ) / total_arrangements = 3 / 7 :=
sorry

end NUMINAMATH_CALUDE_basketball_team_grouping_probability_l2744_274480


namespace NUMINAMATH_CALUDE_unique_n_value_l2744_274463

def is_three_digit (n : ℕ) : Prop := 100 ≤ n ∧ n ≤ 999

theorem unique_n_value (m n : ℕ) 
  (h1 : m > 0)
  (h2 : is_three_digit n)
  (h3 : Nat.lcm m n = 690)
  (h4 : ¬(3 ∣ n))
  (h5 : ¬(2 ∣ m)) :
  n = 230 := by
sorry

end NUMINAMATH_CALUDE_unique_n_value_l2744_274463


namespace NUMINAMATH_CALUDE_trigonometric_product_equals_one_l2744_274449

theorem trigonometric_product_equals_one :
  let x : Real := 40 * π / 180
  let y : Real := 50 * π / 180
  (1 - 1 / Real.cos x) * (1 + 1 / Real.sin y) * (1 - 1 / Real.sin x) * (1 + 1 / Real.cos y) = 1 := by
  sorry

end NUMINAMATH_CALUDE_trigonometric_product_equals_one_l2744_274449


namespace NUMINAMATH_CALUDE_smaller_angle_at_8_is_120_l2744_274412

/-- The number of hour marks on a clock face -/
def clock_hours : ℕ := 12

/-- The number of degrees in a full circle -/
def full_circle_degrees : ℕ := 360

/-- The hour being considered (8 o'clock) -/
def current_hour : ℕ := 8

/-- The angle between adjacent hour marks on a clock face -/
def angle_between_hours : ℚ := full_circle_degrees / clock_hours

/-- The position of the hour hand at the current hour -/
def hour_hand_position : ℚ := current_hour * angle_between_hours

/-- The smaller angle between clock hands at the given hour -/
def smaller_angle_at_hour (h : ℕ) : ℚ :=
  min (h * angle_between_hours) (full_circle_degrees - h * angle_between_hours)

theorem smaller_angle_at_8_is_120 :
  smaller_angle_at_hour current_hour = 120 :=
sorry

end NUMINAMATH_CALUDE_smaller_angle_at_8_is_120_l2744_274412


namespace NUMINAMATH_CALUDE_system_of_equations_solution_l2744_274450

theorem system_of_equations_solution (x y : ℚ) : 
  2 * x - 3 * y = 24 ∧ x + 2 * y = 15 → y = 6/7 := by
  sorry

end NUMINAMATH_CALUDE_system_of_equations_solution_l2744_274450


namespace NUMINAMATH_CALUDE_intersection_line_canonical_equations_l2744_274457

/-- Given two planes in 3D space, this theorem proves that the canonical equations
    of the line formed by their intersection have a specific form. -/
theorem intersection_line_canonical_equations
  (plane1 : ℝ → ℝ → ℝ → Prop)
  (plane2 : ℝ → ℝ → ℝ → Prop)
  (h1 : ∀ x y z, plane1 x y z ↔ 4*x + y - 3*z + 2 = 0)
  (h2 : ∀ x y z, plane2 x y z ↔ 2*x - y + z - 8 = 0) :
  ∃ (line : ℝ → ℝ → ℝ → Prop),
    (∀ x y z, line x y z ↔ (x - 1) / (-2) = (y + 6) / (-10) ∧ (y + 6) / (-10) = z / (-6)) ∧
    (∀ x y z, line x y z ↔ plane1 x y z ∧ plane2 x y z) :=
by sorry

end NUMINAMATH_CALUDE_intersection_line_canonical_equations_l2744_274457


namespace NUMINAMATH_CALUDE_books_per_shelf_l2744_274470

/-- Given four shelves with books and a round-trip distance, 
    prove the number of books on each shelf. -/
theorem books_per_shelf 
  (num_shelves : ℕ) 
  (round_trip_distance : ℕ) 
  (h1 : num_shelves = 4)
  (h2 : round_trip_distance = 3200)
  (h3 : ∃ (books_per_shelf : ℕ), 
    num_shelves * books_per_shelf = round_trip_distance / 2) :
  ∃ (books_per_shelf : ℕ), books_per_shelf = 400 :=
by
  sorry

end NUMINAMATH_CALUDE_books_per_shelf_l2744_274470


namespace NUMINAMATH_CALUDE_inequality_proof_l2744_274469

theorem inequality_proof (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a ≠ b) :
  4 * (a^2 + b^2) > (a + b)^2 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l2744_274469


namespace NUMINAMATH_CALUDE_hyperbola_vertices_distance_l2744_274400

/-- The distance between the vertices of a hyperbola with equation x^2/64 - y^2/49 = 1 is 16 -/
theorem hyperbola_vertices_distance : 
  ∀ (x y : ℝ), x^2/64 - y^2/49 = 1 → ∃ (v1 v2 : ℝ × ℝ), 
    (v1.1^2/64 - v1.2^2/49 = 1) ∧ 
    (v2.1^2/64 - v2.2^2/49 = 1) ∧ 
    (v1.2 = 0) ∧ (v2.2 = 0) ∧
    (v2.1 = -v1.1) ∧
    (v2.1 - v1.1 = 16) :=
by sorry

end NUMINAMATH_CALUDE_hyperbola_vertices_distance_l2744_274400


namespace NUMINAMATH_CALUDE_total_sleep_time_is_36_75_l2744_274492

-- Define sleep time for each day
def monday_sleep : Float := 7
def tuesday_sleep : Float := 6.5
def wednesday_sleep : Float := 2
def thursday_sleep : Float := 9
def friday_sleep : Float := 4.75
def saturday_sleep : Float := 3.75
def sunday_sleep : Float := 4

-- Define total sleep time for the week
def total_sleep_time : Float :=
  monday_sleep + tuesday_sleep + wednesday_sleep + thursday_sleep +
  friday_sleep + saturday_sleep + sunday_sleep

-- Theorem statement
theorem total_sleep_time_is_36_75 :
  total_sleep_time = 36.75 := by
  sorry

end NUMINAMATH_CALUDE_total_sleep_time_is_36_75_l2744_274492


namespace NUMINAMATH_CALUDE_seashell_count_l2744_274424

/-- Given a collection of seashells with specific counts for different colors,
    calculate the number of shells that are not red, green, or blue. -/
theorem seashell_count (total : ℕ) (red green blue : ℕ) 
    (h_total : total = 501)
    (h_red : red = 123)
    (h_green : green = 97)
    (h_blue : blue = 89) :
    total - (red + green + blue) = 192 := by
  sorry

end NUMINAMATH_CALUDE_seashell_count_l2744_274424


namespace NUMINAMATH_CALUDE_sophomores_in_sample_l2744_274468

-- Define the total number of students in each grade
def freshmen : ℕ := 400
def sophomores : ℕ := 600
def juniors : ℕ := 500

-- Define the total sample size
def sample_size : ℕ := 100

-- Theorem to prove
theorem sophomores_in_sample :
  (sophomores * sample_size) / (freshmen + sophomores + juniors) = 40 := by
  sorry

end NUMINAMATH_CALUDE_sophomores_in_sample_l2744_274468


namespace NUMINAMATH_CALUDE_x_range_for_equation_l2744_274419

theorem x_range_for_equation (x y : ℝ) (h : x / y = x - y) : x ≥ 4 ∨ x ≤ 0 := by
  sorry

end NUMINAMATH_CALUDE_x_range_for_equation_l2744_274419


namespace NUMINAMATH_CALUDE_shaded_grid_percentage_l2744_274418

theorem shaded_grid_percentage (total_squares : ℕ) (shaded_squares : ℕ) 
  (h1 : total_squares = 36) (h2 : shaded_squares = 16) : 
  (shaded_squares : ℚ) / total_squares * 100 = 44.4444444444444444 := by
  sorry

end NUMINAMATH_CALUDE_shaded_grid_percentage_l2744_274418


namespace NUMINAMATH_CALUDE_smallest_divisible_by_1_to_10_l2744_274472

theorem smallest_divisible_by_1_to_10 : ∃ (n : ℕ), n > 0 ∧ (∀ k : ℕ, 1 ≤ k ∧ k ≤ 10 → k ∣ n) ∧ (∀ m : ℕ, m > 0 ∧ (∀ k : ℕ, 1 ≤ k ∧ k ≤ 10 → k ∣ m) → n ≤ m) ∧ n = 2520 := by
  sorry

end NUMINAMATH_CALUDE_smallest_divisible_by_1_to_10_l2744_274472


namespace NUMINAMATH_CALUDE_sqrt_equation_solution_l2744_274402

theorem sqrt_equation_solution (x : ℝ) :
  Real.sqrt (9 - 2 * x) = 8 → x = -55/2 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_equation_solution_l2744_274402


namespace NUMINAMATH_CALUDE_trevor_coin_count_l2744_274452

theorem trevor_coin_count : 
  let total_coins : ℕ := 77
  let quarters : ℕ := 29
  let dimes : ℕ := total_coins - quarters
  total_coins - quarters = dimes :=
by sorry

end NUMINAMATH_CALUDE_trevor_coin_count_l2744_274452


namespace NUMINAMATH_CALUDE_odd_periodic_function_property_l2744_274441

def is_odd (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = -f x

def has_period (f : ℝ → ℝ) (p : ℝ) : Prop := ∀ x, f (x + p) = f x

theorem odd_periodic_function_property (f : ℝ → ℝ) 
  (h_odd : is_odd f) 
  (h_period : has_period f 4) 
  (h_f1 : f 1 = 1) : 
  f (-1) + f 8 = -1 := by
  sorry

end NUMINAMATH_CALUDE_odd_periodic_function_property_l2744_274441

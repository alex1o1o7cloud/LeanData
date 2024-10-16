import Mathlib

namespace NUMINAMATH_CALUDE_union_equals_A_l3_368

-- Define sets A and B
def A : Set ℝ := {x | x^2 - 5*x - 24 < 0}
def B (a : ℝ) : Set ℝ := {x | (x - 2*a)*(x - a) < 0}

-- State the theorem
theorem union_equals_A (a : ℝ) : A ∪ B a = A ↔ a ∈ Set.Icc (-3/2) 4 := by
  sorry

end NUMINAMATH_CALUDE_union_equals_A_l3_368


namespace NUMINAMATH_CALUDE_contrapositive_false_l3_321

theorem contrapositive_false : ¬(∀ x y : ℝ, (x ≤ 0 ∨ y ≤ 0) → x + y ≤ 0) := by
  sorry

end NUMINAMATH_CALUDE_contrapositive_false_l3_321


namespace NUMINAMATH_CALUDE_kids_joined_soccer_l3_348

theorem kids_joined_soccer (initial_kids final_kids : ℕ) (h1 : initial_kids = 14) (h2 : final_kids = 36) :
  final_kids - initial_kids = 22 := by
  sorry

end NUMINAMATH_CALUDE_kids_joined_soccer_l3_348


namespace NUMINAMATH_CALUDE_smallest_integer_y_l3_304

theorem smallest_integer_y : ∀ y : ℤ, (5 : ℚ) / 8 < (y - 3 : ℚ) / 19 → y ≥ 15 :=
by sorry

end NUMINAMATH_CALUDE_smallest_integer_y_l3_304


namespace NUMINAMATH_CALUDE_cos_alpha_minus_pi_sixth_l3_370

theorem cos_alpha_minus_pi_sixth (α : ℝ) (h : Real.sin (α + π/3) = 4/5) : 
  Real.cos (α - π/6) = 4/5 := by
sorry

end NUMINAMATH_CALUDE_cos_alpha_minus_pi_sixth_l3_370


namespace NUMINAMATH_CALUDE_paytons_score_l3_355

theorem paytons_score (total_students : ℕ) (students_without_payton : ℕ) 
  (avg_without_payton : ℚ) (new_avg : ℚ) :
  total_students = 15 →
  students_without_payton = 14 →
  avg_without_payton = 80 →
  new_avg = 81 →
  (students_without_payton * avg_without_payton + 
    (total_students - students_without_payton) * 
    ((total_students * new_avg - students_without_payton * avg_without_payton) / 
    (total_students - students_without_payton))) / total_students = new_avg →
  (total_students * new_avg - students_without_payton * avg_without_payton) / 
  (total_students - students_without_payton) = 95 := by
sorry

end NUMINAMATH_CALUDE_paytons_score_l3_355


namespace NUMINAMATH_CALUDE_sin_shift_l3_351

theorem sin_shift (x : ℝ) : Real.sin (2 * x + π / 3) = Real.sin (2 * (x + π / 6)) := by
  sorry

end NUMINAMATH_CALUDE_sin_shift_l3_351


namespace NUMINAMATH_CALUDE_x_one_value_l3_302

theorem x_one_value (x₁ x₂ x₃ x₄ : ℝ) 
  (h_order : 0 ≤ x₄ ∧ x₄ ≤ x₃ ∧ x₃ ≤ x₂ ∧ x₂ ≤ x₁ ∧ x₁ ≤ 1)
  (h_equation : (1 - x₁)^2 + (x₁ - x₂)^2 + (x₂ - x₃)^2 + (x₃ - x₄)^2 + x₄^2 = 1/3) :
  x₁ = 4/5 := by
sorry

end NUMINAMATH_CALUDE_x_one_value_l3_302


namespace NUMINAMATH_CALUDE_quadratic_inequality_range_l3_386

theorem quadratic_inequality_range (a : ℝ) :
  (∀ x : ℝ, x^2 + 2*a*x + a > 0) → 0 < a ∧ a < 1 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_inequality_range_l3_386


namespace NUMINAMATH_CALUDE_product_of_real_parts_l3_312

theorem product_of_real_parts : ∃ (z₁ z₂ : ℂ),
  (z₁^2 - 4*z₁ = 3*Complex.I) ∧
  (z₂^2 - 4*z₂ = 3*Complex.I) ∧
  (z₁ ≠ z₂) ∧
  (Complex.re z₁ * Complex.re z₂ = -1/2) := by
  sorry

end NUMINAMATH_CALUDE_product_of_real_parts_l3_312


namespace NUMINAMATH_CALUDE_geometric_sequence_sum_l3_315

theorem geometric_sequence_sum (a₁ a₄ r : ℚ) (h₁ : a₁ = 4096) (h₂ : a₄ = 16) (h₃ : r = 1/4) :
  a₁ * r + a₁ * r^2 = 320 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_sum_l3_315


namespace NUMINAMATH_CALUDE_correct_multiplication_l3_331

theorem correct_multiplication (x : ℕ) (h : 63 + x = 70) : 36 * x = 252 := by
  sorry

end NUMINAMATH_CALUDE_correct_multiplication_l3_331


namespace NUMINAMATH_CALUDE_race_head_start_l3_344

/-- Proves that if A's speed is 32/27 times B's speed, then A needs to give B
    a head start of 5/32 of the race length for the race to end in a dead heat. -/
theorem race_head_start (L : ℝ) (Va Vb : ℝ) (h : Va = (32/27) * Vb) :
  (L / Va = (L - (5/32) * L) / Vb) := by
  sorry

end NUMINAMATH_CALUDE_race_head_start_l3_344


namespace NUMINAMATH_CALUDE_always_less_than_log_sum_implies_less_than_one_l3_307

theorem always_less_than_log_sum_implies_less_than_one (a : ℝ) : 
  (∀ x : ℝ, a < Real.log (|x - 3| + |x + 7|)) → a < 1 := by
  sorry

end NUMINAMATH_CALUDE_always_less_than_log_sum_implies_less_than_one_l3_307


namespace NUMINAMATH_CALUDE_extremum_at_one_implies_a_eq_neg_two_l3_347

/-- Given a cubic function f(x) = x^3 + ax^2 + x + b with an extremum at x = 1, 
    prove that a = -2. -/
theorem extremum_at_one_implies_a_eq_neg_two (a b : ℝ) :
  let f : ℝ → ℝ := λ x => x^3 + a*x^2 + x + b
  (∃ ε > 0, ∀ x ∈ Set.Ioo (1 - ε) (1 + ε), f x ≤ f 1 ∨ ∀ x ∈ Set.Ioo (1 - ε) (1 + ε), f x ≥ f 1) →
  a = -2 := by
sorry

end NUMINAMATH_CALUDE_extremum_at_one_implies_a_eq_neg_two_l3_347


namespace NUMINAMATH_CALUDE_triangle_area_from_intersecting_lines_triangle_area_from_intersecting_lines_proof_l3_319

/-- Given two lines intersecting at P(1,6), one with slope 1 and the other with slope 2,
    the area of the triangle formed by P and the x-intercepts of these lines is 9 square units. -/
theorem triangle_area_from_intersecting_lines : ℝ → Prop :=
  fun area =>
    let P : ℝ × ℝ := (1, 6)
    let slope1 : ℝ := 1
    let slope2 : ℝ := 2
    let Q : ℝ × ℝ := (P.1 - P.2 / slope1, 0)  -- x-intercept of line with slope 1
    let R : ℝ × ℝ := (P.1 - P.2 / slope2, 0)  -- x-intercept of line with slope 2
    let base : ℝ := R.1 - Q.1
    let height : ℝ := P.2
    area = (1/2) * base * height ∧ area = 9

/-- Proof of the theorem -/
theorem triangle_area_from_intersecting_lines_proof : 
  ∃ (area : ℝ), triangle_area_from_intersecting_lines area :=
by
  sorry

#check triangle_area_from_intersecting_lines
#check triangle_area_from_intersecting_lines_proof

end NUMINAMATH_CALUDE_triangle_area_from_intersecting_lines_triangle_area_from_intersecting_lines_proof_l3_319


namespace NUMINAMATH_CALUDE_price_change_percentage_l3_327

theorem price_change_percentage (P : ℝ) (x : ℝ) : 
  P * (1 + x/100) * (1 - x/100) = 0.64 * P → x = 60 := by
  sorry

end NUMINAMATH_CALUDE_price_change_percentage_l3_327


namespace NUMINAMATH_CALUDE_train_length_l3_372

/-- The length of a train given its speed and time to cross a pole -/
theorem train_length (speed : ℝ) (time : ℝ) : 
  speed = 60 → time = 3 → ∃ (length : ℝ), abs (length - 50.01) < 0.01 := by
  sorry


end NUMINAMATH_CALUDE_train_length_l3_372


namespace NUMINAMATH_CALUDE_triangle_area_l3_330

-- Define the curve
def curve (x : ℝ) : ℝ := x^3

-- Define the tangent line at (1, 1)
def tangent_line (x : ℝ) : ℝ := 3*x - 2

-- Define the x-axis (y = 0)
def x_axis (x : ℝ) : ℝ := 0

-- Define the vertical line x = 2
def vertical_line : ℝ := 2

-- Theorem statement
theorem triangle_area : 
  let x_intercept : ℝ := 2/3
  let height : ℝ := tangent_line vertical_line
  (1/2) * (vertical_line - x_intercept) * height = 8/3 := by sorry

end NUMINAMATH_CALUDE_triangle_area_l3_330


namespace NUMINAMATH_CALUDE_largest_gcd_of_ten_numbers_summing_to_1001_l3_335

theorem largest_gcd_of_ten_numbers_summing_to_1001 :
  ∃ (a₁ a₂ a₃ a₄ a₅ a₆ a₇ a₈ a₉ a₁₀ : ℕ),
    a₁ + a₂ + a₃ + a₄ + a₅ + a₆ + a₇ + a₈ + a₉ + a₁₀ = 1001 ∧
    (∀ d : ℕ, d > 0 → d ∣ a₁ ∧ d ∣ a₂ ∧ d ∣ a₃ ∧ d ∣ a₄ ∧ d ∣ a₅ ∧
                      d ∣ a₆ ∧ d ∣ a₇ ∧ d ∣ a₈ ∧ d ∣ a₉ ∧ d ∣ a₁₀ → d ≤ 91) ∧
    91 ∣ a₁ ∧ 91 ∣ a₂ ∧ 91 ∣ a₃ ∧ 91 ∣ a₄ ∧ 91 ∣ a₅ ∧
    91 ∣ a₆ ∧ 91 ∣ a₇ ∧ 91 ∣ a₈ ∧ 91 ∣ a₉ ∧ 91 ∣ a₁₀ := by
  sorry

end NUMINAMATH_CALUDE_largest_gcd_of_ten_numbers_summing_to_1001_l3_335


namespace NUMINAMATH_CALUDE_geometric_arithmetic_geometric_sequence_l3_369

theorem geometric_arithmetic_geometric_sequence :
  ∀ a q : ℝ,
  (∀ x y z : ℝ, x = a ∧ y = a * q ∧ z = a * q^2 →
    (2 * (a * q + 8) = a + a * q^2) ∧
    ((a * q + 8)^2 = a * (a * q^2 + 64))) →
  ((a = 4 ∧ q = 3) ∨ (a = 4/9 ∧ q = -5)) :=
by sorry

end NUMINAMATH_CALUDE_geometric_arithmetic_geometric_sequence_l3_369


namespace NUMINAMATH_CALUDE_fourth_degree_polynomial_abs_value_l3_341

/-- A fourth-degree polynomial with real coefficients -/
def fourth_degree_polynomial (f : ℝ → ℝ) : Prop :=
  ∃ a b c d e : ℝ, ∀ x, f x = a * x^4 + b * x^3 + c * x^2 + d * x + e

/-- The absolute value of f at specific points is 16 -/
def abs_value_16 (f : ℝ → ℝ) : Prop :=
  |f 1| = 16 ∧ |f 3| = 16 ∧ |f 4| = 16 ∧ |f 5| = 16 ∧ |f 7| = 16

theorem fourth_degree_polynomial_abs_value (f : ℝ → ℝ) :
  fourth_degree_polynomial f → abs_value_16 f → |f 0| = 436 := by
  sorry

end NUMINAMATH_CALUDE_fourth_degree_polynomial_abs_value_l3_341


namespace NUMINAMATH_CALUDE_a_investment_is_6300_l3_361

/-- Represents the investment and profit scenario of a partnership business -/
structure BusinessPartnership where
  /-- A's investment amount -/
  a_investment : ℝ
  /-- B's investment amount -/
  b_investment : ℝ
  /-- C's investment amount -/
  c_investment : ℝ
  /-- Total profit -/
  total_profit : ℝ
  /-- A's share of the profit -/
  a_profit : ℝ

/-- Theorem stating that given the conditions, A's investment is 6300 -/
theorem a_investment_is_6300 (bp : BusinessPartnership)
  (h1 : bp.b_investment = 4200)
  (h2 : bp.c_investment = 10500)
  (h3 : bp.total_profit = 12700)
  (h4 : bp.a_profit = 3810)
  (h5 : bp.a_profit / bp.total_profit = bp.a_investment / (bp.a_investment + bp.b_investment + bp.c_investment)) :
  bp.a_investment = 6300 := by
  sorry

end NUMINAMATH_CALUDE_a_investment_is_6300_l3_361


namespace NUMINAMATH_CALUDE_fourth_mile_relation_l3_390

/-- Represents the relationship between distance and time for a mile -/
structure MileData where
  n : ℕ      -- The mile number
  time : ℝ    -- Time taken to cover the mile (in hours)
  distance : ℝ -- Distance covered (in miles)

/-- The constant k in the inverse relationship -/
def k : ℝ := 2

/-- The inverse relationship between distance and time for a given mile -/
def inverse_relation (md : MileData) : Prop :=
  md.distance = k / md.time

/-- Theorem stating the relationship for the 2nd and 4th miles -/
theorem fourth_mile_relation 
  (mile2 : MileData) 
  (mile4 : MileData) 
  (h1 : mile2.n = 2) 
  (h2 : mile2.time = 2) 
  (h3 : mile2.distance = 1) 
  (h4 : mile4.n = 4) 
  (h5 : inverse_relation mile2) 
  (h6 : inverse_relation mile4) : 
  mile4.time = 4 ∧ mile4.distance = 0.5 := by
  sorry


end NUMINAMATH_CALUDE_fourth_mile_relation_l3_390


namespace NUMINAMATH_CALUDE_power_division_l3_317

theorem power_division (a : ℝ) (h : a ≠ 0) : a^3 / a^2 = a := by
  sorry

end NUMINAMATH_CALUDE_power_division_l3_317


namespace NUMINAMATH_CALUDE_f_composition_pi_12_l3_342

noncomputable def f (x : ℝ) : ℝ :=
  if x ≤ 0 then 4 * x^2 - 1 else Real.sin x^2 - Real.cos x^2

theorem f_composition_pi_12 : f (f (π / 12)) = 2 := by
  sorry

end NUMINAMATH_CALUDE_f_composition_pi_12_l3_342


namespace NUMINAMATH_CALUDE_zero_is_monomial_l3_354

/-- A monomial is a polynomial with a single term. -/
def IsMonomial (p : Polynomial ℝ) : Prop :=
  ∃ c a, p = c * Polynomial.X ^ a

/-- Zero is a monomial. -/
theorem zero_is_monomial : IsMonomial (0 : Polynomial ℝ) := by
  sorry

end NUMINAMATH_CALUDE_zero_is_monomial_l3_354


namespace NUMINAMATH_CALUDE_line_through_point_with_equal_intercepts_l3_314

/-- A line passing through (1, 3) with equal absolute intercepts has one of three specific equations -/
theorem line_through_point_with_equal_intercepts :
  ∀ (f : ℝ → ℝ),
  (f 1 = 3) →  -- Line passes through (1, 3)
  (∃ a : ℝ, a ≠ 0 ∧ f 0 = a ∧ f a = 0) →  -- Equal absolute intercepts
  (∀ x, f x = 3 * x) ∨  -- y = 3x
  (∀ x, x + f x = 4) ∨  -- x + y - 4 = 0
  (∀ x, x - f x = -2)  -- x - y + 2 = 0
  := by sorry

end NUMINAMATH_CALUDE_line_through_point_with_equal_intercepts_l3_314


namespace NUMINAMATH_CALUDE_hash_difference_l3_328

-- Define the # operation
def hash (x y : ℤ) : ℤ := x * y - 3 * x + y

-- State the theorem
theorem hash_difference : hash 8 5 - hash 5 8 = -12 := by sorry

end NUMINAMATH_CALUDE_hash_difference_l3_328


namespace NUMINAMATH_CALUDE_hungry_bear_purchase_cost_l3_392

/-- Represents the cost of items at Hungry Bear Diner -/
structure DinerCost where
  sandwich_price : ℕ
  soda_price : ℕ
  cookie_price : ℕ

/-- Calculates the total cost of a purchase at Hungry Bear Diner -/
def total_cost (prices : DinerCost) (num_sandwiches num_sodas num_cookies : ℕ) : ℕ :=
  prices.sandwich_price * num_sandwiches +
  prices.soda_price * num_sodas +
  prices.cookie_price * num_cookies

/-- Theorem stating that the total cost of 3 sandwiches, 5 sodas, and 4 cookies is $35 -/
theorem hungry_bear_purchase_cost :
  ∃ (prices : DinerCost),
    prices.sandwich_price = 4 ∧
    prices.soda_price = 3 ∧
    prices.cookie_price = 2 ∧
    total_cost prices 3 5 4 = 35 := by
  sorry

end NUMINAMATH_CALUDE_hungry_bear_purchase_cost_l3_392


namespace NUMINAMATH_CALUDE_hyperbola_eccentricity_l3_364

theorem hyperbola_eccentricity (a b c : ℝ) (h1 : a > 0) (h2 : b > 0) :
  (∀ x y : ℝ, x^2 / a^2 - y^2 / b^2 = 1) →
  (b * c / (a^2 + b^2).sqrt = b) →
  (2 * a ≤ b) →
  let e := c / a
  e > Real.sqrt 5 := by sorry

end NUMINAMATH_CALUDE_hyperbola_eccentricity_l3_364


namespace NUMINAMATH_CALUDE_dividend_calculation_l3_379

theorem dividend_calculation (divisor quotient remainder dividend : ℤ) :
  divisor = 800 →
  quotient = 594 →
  remainder = -968 →
  dividend = divisor * quotient + remainder →
  dividend = 474232 := by
sorry

end NUMINAMATH_CALUDE_dividend_calculation_l3_379


namespace NUMINAMATH_CALUDE_valid_circular_arrangement_exists_l3_349

/-- A type representing a circular arrangement of 9 numbers -/
def CircularArrangement := Fin 9 → Fin 9

/-- Check if two numbers in the arrangement are adjacent -/
def are_adjacent (arr : CircularArrangement) (i j : Fin 9) : Prop :=
  (j = i + 1) ∨ (i = 8 ∧ j = 0)

/-- Check if a number is valid in the arrangement (1 to 9) -/
def is_valid_number (n : Fin 9) : Prop := n.val + 1 ∈ Finset.range 10

/-- Check if the sum of two numbers is not divisible by 3, 5, or 7 -/
def sum_not_divisible (a b : Fin 9) : Prop :=
  ¬(((a.val + 1) + (b.val + 1)) % 3 = 0) ∧
  ¬(((a.val + 1) + (b.val + 1)) % 5 = 0) ∧
  ¬(((a.val + 1) + (b.val + 1)) % 7 = 0)

/-- The main theorem stating that a valid circular arrangement exists -/
theorem valid_circular_arrangement_exists : ∃ (arr : CircularArrangement),
  (∀ i : Fin 9, is_valid_number (arr i)) ∧
  (∀ i j : Fin 9, are_adjacent arr i j → sum_not_divisible (arr i) (arr j)) ∧
  Function.Injective arr :=
sorry

end NUMINAMATH_CALUDE_valid_circular_arrangement_exists_l3_349


namespace NUMINAMATH_CALUDE_pyramid_triangular_faces_area_l3_339

/-- The area of triangular faces of a right square-based pyramid -/
theorem pyramid_triangular_faces_area
  (base_edge : ℝ)
  (lateral_edge : ℝ)
  (h_base : base_edge = 8)
  (h_lateral : lateral_edge = 7) :
  4 * (1/2 * base_edge * Real.sqrt (lateral_edge^2 - (base_edge/2)^2)) = 16 * Real.sqrt 33 :=
by sorry

end NUMINAMATH_CALUDE_pyramid_triangular_faces_area_l3_339


namespace NUMINAMATH_CALUDE_negation_of_implication_l3_374

theorem negation_of_implication (a b c : ℝ) :
  ¬(a > b → a + c > b + c) ↔ (a ≤ b → a + c ≤ b + c) := by sorry

end NUMINAMATH_CALUDE_negation_of_implication_l3_374


namespace NUMINAMATH_CALUDE_recipe_people_l3_380

/-- The number of people the original recipe is intended for -/
def P : ℕ := sorry

/-- The number of eggs required for the original recipe -/
def original_eggs : ℕ := 2

/-- The number of people Tyler wants to make the cake for -/
def tyler_people : ℕ := 8

/-- The number of eggs Tyler needs for his cake -/
def tyler_eggs : ℕ := 4

theorem recipe_people : P = 4 := by
  sorry

end NUMINAMATH_CALUDE_recipe_people_l3_380


namespace NUMINAMATH_CALUDE_right_triangle_theorem_l3_310

/-- Right triangle DEF with given side lengths and midpoint N on hypotenuse -/
structure RightTriangle where
  /-- Length of side DE -/
  de : ℝ
  /-- Length of side DF -/
  df : ℝ
  /-- Right angle at E -/
  right_angle : de ^ 2 + df ^ 2 = (de + df) ^ 2 / 4
  /-- N is midpoint of EF -/
  n_midpoint : True

/-- Properties of the right triangle -/
def triangle_properties (t : RightTriangle) : Prop :=
  let dn := (t.de ^ 2 + t.df ^ 2).sqrt / 2
  let area := t.de * t.df / 2
  let centroid_distance := 2 * dn / 3
  dn = 5.0 ∧ area = 24.0 ∧ centroid_distance = 3.3

/-- Theorem stating the properties of the specific right triangle -/
theorem right_triangle_theorem :
  ∃ t : RightTriangle, t.de = 6 ∧ t.df = 8 ∧ triangle_properties t :=
sorry

end NUMINAMATH_CALUDE_right_triangle_theorem_l3_310


namespace NUMINAMATH_CALUDE_additional_men_equal_initial_l3_318

theorem additional_men_equal_initial (initial_men : ℕ) (initial_days food_supply : ℝ) : 
  initial_men * initial_days = food_supply ∧
  initial_days = 50 ∧
  initial_men = 20 →
  ∃ (additional_men : ℕ),
    (initial_men + additional_men) * 25 = food_supply ∧
    additional_men = initial_men :=
by sorry

end NUMINAMATH_CALUDE_additional_men_equal_initial_l3_318


namespace NUMINAMATH_CALUDE_largest_prime_divisor_l3_325

theorem largest_prime_divisor :
  ∃ (p : ℕ), Nat.Prime p ∧ 
    p ∣ (2^(p+1) + 3^(p+1) + 5^(p+1) + 7^(p+1)) ∧
    ∀ (q : ℕ), Nat.Prime q → q ∣ (2^(q+1) + 3^(q+1) + 5^(q+1) + 7^(q+1)) → q ≤ p :=
by
  use 29
  sorry

end NUMINAMATH_CALUDE_largest_prime_divisor_l3_325


namespace NUMINAMATH_CALUDE_product_of_solutions_abs_y_eq_3_abs_y_minus_2_l3_384

theorem product_of_solutions_abs_y_eq_3_abs_y_minus_2 :
  ∃ (y₁ y₂ : ℝ), (|y₁| = 3 * (|y₁| - 2)) ∧ (|y₂| = 3 * (|y₂| - 2)) ∧ y₁ ≠ y₂ ∧ y₁ * y₂ = -9 :=
by sorry

end NUMINAMATH_CALUDE_product_of_solutions_abs_y_eq_3_abs_y_minus_2_l3_384


namespace NUMINAMATH_CALUDE_pascal_ratio_in_row_98_l3_360

/-- Pascal's Triangle entry -/
def pascal (n : ℕ) (k : ℕ) : ℕ := Nat.choose n k

/-- Check if three consecutive entries in a row are in ratio 4:5:6 -/
def hasRatio456 (n : ℕ) : Prop :=
  ∃ r : ℕ, 
    (pascal n r : ℚ) / (pascal n (r + 1)) = 4 / 5 ∧
    (pascal n (r + 1) : ℚ) / (pascal n (r + 2)) = 5 / 6

theorem pascal_ratio_in_row_98 : hasRatio456 98 := by
  sorry

end NUMINAMATH_CALUDE_pascal_ratio_in_row_98_l3_360


namespace NUMINAMATH_CALUDE_angle_y_value_l3_343

-- Define the triangles and angles
def triangle_ABC (A B C : ℝ) : Prop := A + B + C = 180

def right_triangle (A B : ℝ) : Prop := A + B = 90

-- State the theorem
theorem angle_y_value :
  ∀ A B C D E y : ℝ,
  triangle_ABC 50 70 C →
  right_triangle D y →
  D = C →
  y = 30 :=
by sorry

end NUMINAMATH_CALUDE_angle_y_value_l3_343


namespace NUMINAMATH_CALUDE_total_slices_eq_207_l3_340

/-- The total number of watermelon and fruit slices at a family picnic --/
def total_slices : ℕ :=
  let danny_watermelon := 3 * 10
  let sister_watermelon := 1 * 15
  let cousin_watermelon := 2 * 8
  let cousin_apples := 5 * 4
  let aunt_watermelon := 4 * 12
  let aunt_oranges := 7 * 6
  let grandfather_watermelon := 1 * 6
  let grandfather_pineapples := 3 * 10
  danny_watermelon + sister_watermelon + cousin_watermelon + cousin_apples +
  aunt_watermelon + aunt_oranges + grandfather_watermelon + grandfather_pineapples

theorem total_slices_eq_207 : total_slices = 207 := by
  sorry

end NUMINAMATH_CALUDE_total_slices_eq_207_l3_340


namespace NUMINAMATH_CALUDE_quadratic_transformation_l3_366

theorem quadratic_transformation (x : ℝ) : x^2 - 8*x - 9 = (x - 4)^2 - 25 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_transformation_l3_366


namespace NUMINAMATH_CALUDE_problem_solution_l3_363

def U : Set ℝ := Set.univ

def A : Set ℝ := {x | x^2 + 3*x + 2 = 0}

def B (m : ℝ) : Set ℝ := {x | x^2 + (m+1)*x + m = 0}

theorem problem_solution (m : ℝ) : (U \ A) ∩ B m = ∅ → m = 1 ∨ m = 2 := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l3_363


namespace NUMINAMATH_CALUDE_angle_symmetry_l3_365

/-- Two angles are symmetric about the y-axis if their sum is congruent to 180° modulo 360° -/
def symmetric_about_y_axis (α β : Real) : Prop :=
  ∃ k : ℤ, α + β = k * 360 + 180

theorem angle_symmetry (α β : Real) :
  symmetric_about_y_axis α β →
  ∃ k : ℤ, α + β = k * 360 + 180 := by
  sorry

end NUMINAMATH_CALUDE_angle_symmetry_l3_365


namespace NUMINAMATH_CALUDE_additional_rows_l3_376

theorem additional_rows (initial_rows : ℕ) (initial_trees_per_row : ℕ) (new_trees_per_row : ℕ) :
  initial_rows = 24 →
  initial_trees_per_row = 42 →
  new_trees_per_row = 28 →
  (initial_rows * initial_trees_per_row) / new_trees_per_row - initial_rows = 12 :=
by sorry

end NUMINAMATH_CALUDE_additional_rows_l3_376


namespace NUMINAMATH_CALUDE_expression_equality_l3_382

theorem expression_equality : 200 * (200 - 5) - (200 * 200 - 5) = -995 := by
  sorry

end NUMINAMATH_CALUDE_expression_equality_l3_382


namespace NUMINAMATH_CALUDE_right_triangle_sinC_l3_358

theorem right_triangle_sinC (A B C : Real) (h1 : A + B + C = Real.pi) 
  (h2 : B = Real.pi / 2) (h3 : Real.tan A = 3 / 4) : Real.sin C = 4 / 5 := by
  sorry

end NUMINAMATH_CALUDE_right_triangle_sinC_l3_358


namespace NUMINAMATH_CALUDE_vector_dot_product_l3_367

/-- Given two vectors a and b in ℝ², prove that their dot product is -12
    when their sum is (1, -3) and their difference is (3, 7). -/
theorem vector_dot_product (a b : ℝ × ℝ) 
    (h1 : a.1 + b.1 = 1 ∧ a.2 + b.2 = -3)
    (h2 : a.1 - b.1 = 3 ∧ a.2 - b.2 = 7) :
    a.1 * b.1 + a.2 * b.2 = -12 := by
  sorry

end NUMINAMATH_CALUDE_vector_dot_product_l3_367


namespace NUMINAMATH_CALUDE_geometric_progression_proof_l3_329

theorem geometric_progression_proof (b : ℕ → ℚ) 
  (h1 : b 4 - b 2 = -45/32) 
  (h2 : b 6 - b 4 = -45/512) 
  (h_geom : ∀ n : ℕ, b (n + 1) = b 1 * (b 2 / b 1) ^ n) :
  ((b 1 = -6 ∧ b 2 / b 1 = -1/4) ∨ (b 1 = 6 ∧ b 2 / b 1 = 1/4)) :=
sorry

end NUMINAMATH_CALUDE_geometric_progression_proof_l3_329


namespace NUMINAMATH_CALUDE_four_lattice_points_l3_350

-- Define the equation
def equation (x y : ℤ) : Prop := x^2 - y^2 = 53

-- Define a lattice point as a pair of integers
def LatticePoint : Type := ℤ × ℤ

-- Define a function to check if a lattice point satisfies the equation
def satisfies_equation (p : LatticePoint) : Prop :=
  equation p.1 p.2

-- Theorem: There are exactly 4 lattice points satisfying the equation
theorem four_lattice_points : 
  ∃! (s : Finset LatticePoint), (∀ p ∈ s, satisfies_equation p) ∧ s.card = 4 :=
sorry

end NUMINAMATH_CALUDE_four_lattice_points_l3_350


namespace NUMINAMATH_CALUDE_min_sum_of_product_144_l3_309

theorem min_sum_of_product_144 :
  ∀ c d : ℤ, c * d = 144 → (∀ x y : ℤ, x * y = 144 → c + d ≤ x + y) ∧ (∃ a b : ℤ, a * b = 144 ∧ a + b = -145) :=
by sorry

end NUMINAMATH_CALUDE_min_sum_of_product_144_l3_309


namespace NUMINAMATH_CALUDE_arrangement_count_is_70_l3_395

/-- The number of ways to arrange 6 indistinguishable objects of type A
    and 4 indistinguishable objects of type B in a row of 10 positions,
    where type A objects must occupy the first and last positions. -/
def arrangement_count : ℕ := sorry

/-- Theorem stating that the number of arrangements is 70 -/
theorem arrangement_count_is_70 : arrangement_count = 70 := by sorry

end NUMINAMATH_CALUDE_arrangement_count_is_70_l3_395


namespace NUMINAMATH_CALUDE_monica_wednesday_study_time_l3_357

/-- Represents the study schedule of Monica over five days -/
structure StudySchedule where
  wednesday : ℝ
  thursday : ℝ
  friday : ℝ
  weekend : ℝ
  total : ℝ

/-- The study schedule satisfies the given conditions -/
def validSchedule (s : StudySchedule) : Prop :=
  s.thursday = 3 * s.wednesday ∧
  s.friday = 1.5 * s.wednesday ∧
  s.weekend = 5.5 * s.wednesday ∧
  s.total = 22 ∧
  s.total = s.wednesday + s.thursday + s.friday + s.weekend

/-- Theorem stating that Monica studied 2 hours on Wednesday -/
theorem monica_wednesday_study_time (s : StudySchedule) 
  (h : validSchedule s) : s.wednesday = 2 := by
  sorry

end NUMINAMATH_CALUDE_monica_wednesday_study_time_l3_357


namespace NUMINAMATH_CALUDE_max_a_is_maximum_l3_332

/-- The maximum value of a such that the line y = mx + 1 does not pass through
    any lattice points for 0 < x ≤ 200 and 1/2 < m < a -/
def max_a : ℚ := 101 / 201

/-- Predicate to check if a point (x, y) is a lattice point -/
def is_lattice_point (x y : ℚ) : Prop := ∃ (n m : ℤ), x = n ∧ y = m

/-- Predicate to check if the line y = mx + 1 passes through a lattice point -/
def line_passes_lattice_point (m : ℚ) (x : ℚ) : Prop :=
  ∃ (y : ℚ), is_lattice_point x y ∧ y = m * x + 1

theorem max_a_is_maximum :
  ∀ (a : ℚ), (∀ (m : ℚ), 1/2 < m → m < a →
    ∀ (x : ℚ), 0 < x → x ≤ 200 → ¬ line_passes_lattice_point m x) →
  a ≤ max_a :=
sorry

end NUMINAMATH_CALUDE_max_a_is_maximum_l3_332


namespace NUMINAMATH_CALUDE_probability_of_meeting_l3_338

def knockout_tournament (n : ℕ) := n > 1

def num_matches (n : ℕ) (h : knockout_tournament n) : ℕ := n - 1

def num_pairs (n : ℕ) : ℕ := n.choose 2

theorem probability_of_meeting (n : ℕ) (h : knockout_tournament n) :
  (num_matches n h : ℚ) / (num_pairs n : ℚ) = 31 / 496 :=
sorry

end NUMINAMATH_CALUDE_probability_of_meeting_l3_338


namespace NUMINAMATH_CALUDE_difference_c_minus_a_l3_333

theorem difference_c_minus_a (a b c : ℝ) : 
  (a + b) / 2 = 30 → c - a = 60 → c - a = 60 := by
  sorry

end NUMINAMATH_CALUDE_difference_c_minus_a_l3_333


namespace NUMINAMATH_CALUDE_solve_frog_pond_l3_377

def frog_pond_problem (initial_frogs : ℕ) : Prop :=
  let tadpoles := 3 * initial_frogs
  let surviving_tadpoles := (2 * tadpoles) / 3
  let total_frogs := initial_frogs + surviving_tadpoles
  (total_frogs = 8) ∧ (total_frogs - 7 = 1)

theorem solve_frog_pond : ∃ (n : ℕ), frog_pond_problem n ∧ n = 2 := by
  sorry

end NUMINAMATH_CALUDE_solve_frog_pond_l3_377


namespace NUMINAMATH_CALUDE_decimal_to_base5_l3_378

theorem decimal_to_base5 : 
  (3 * 5^2 + 2 * 5^1 + 3 * 5^0 : ℕ) = 88 := by
  sorry

end NUMINAMATH_CALUDE_decimal_to_base5_l3_378


namespace NUMINAMATH_CALUDE_min_portraits_theorem_l3_336

def min_year := 1600
def max_year := 2008
def max_age := 80

def ScientistData := {birth : ℕ // min_year ≤ birth ∧ birth ≤ max_year}

def death_year (s : ScientistData) : ℕ := s.val + (Nat.min max_age (max_year - s.val))

def product_ratio (scientists : List ScientistData) : ℚ :=
  (scientists.map death_year).prod / (scientists.map (λ s => s.val)).prod

theorem min_portraits_theorem :
  ∃ (scientists : List ScientistData),
    scientists.length = 5 ∧
    product_ratio scientists = 5/4 ∧
    ∀ (smaller_list : List ScientistData),
      smaller_list.length < 5 →
      product_ratio smaller_list < 5/4 :=
sorry

end NUMINAMATH_CALUDE_min_portraits_theorem_l3_336


namespace NUMINAMATH_CALUDE_number_equation_l3_320

theorem number_equation (y : ℝ) : y = (1 / y) * (-y) - 5 → y = -6 := by
  sorry

end NUMINAMATH_CALUDE_number_equation_l3_320


namespace NUMINAMATH_CALUDE_tammy_catches_l3_337

theorem tammy_catches (joe derek tammy : ℕ) : 
  joe = 23 →
  derek = 2 * joe - 4 →
  tammy = derek / 3 + 16 →
  tammy = 30 := by
sorry

end NUMINAMATH_CALUDE_tammy_catches_l3_337


namespace NUMINAMATH_CALUDE_intersection_of_A_and_B_l3_362

-- Define the sets A and B
def A : Set ℝ := {x | x^2 - x - 6 < 0}
def B : Set ℝ := {x | x^2 + 2*x - 8 > 0}

-- State the theorem
theorem intersection_of_A_and_B :
  A ∩ B = {x : ℝ | 2 < x ∧ x < 3} := by sorry

end NUMINAMATH_CALUDE_intersection_of_A_and_B_l3_362


namespace NUMINAMATH_CALUDE_problem_solution_l3_352

theorem problem_solution : ∃ x : ℤ, x - (28 - (37 - (15 - 16))) = 55 ∧ x = 65 := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l3_352


namespace NUMINAMATH_CALUDE_dinner_set_cost_calculation_l3_393

/-- The cost calculation for John's dinner set purchase --/
theorem dinner_set_cost_calculation :
  let fork_cost : ℚ := 25
  let knife_cost : ℚ := 30
  let spoon_cost : ℚ := 20
  let silverware_cost : ℚ := fork_cost + knife_cost + spoon_cost
  let plate_cost : ℚ := silverware_cost * (1/2)
  let total_cost : ℚ := silverware_cost + plate_cost
  let discount_rate : ℚ := 1/10
  let final_cost : ℚ := total_cost * (1 - discount_rate)
  final_cost = 101.25 := by
  sorry

end NUMINAMATH_CALUDE_dinner_set_cost_calculation_l3_393


namespace NUMINAMATH_CALUDE_expand_binomials_l3_313

theorem expand_binomials (x : ℝ) : (7 * x + 9) * (3 * x + 4) = 21 * x^2 + 55 * x + 36 := by
  sorry

end NUMINAMATH_CALUDE_expand_binomials_l3_313


namespace NUMINAMATH_CALUDE_two_digit_sum_product_l3_316

/-- A function that returns the tens digit of a two-digit number -/
def tens_digit (n : ℕ) : ℕ := n / 10

/-- A function that returns the ones digit of a two-digit number -/
def ones_digit (n : ℕ) : ℕ := n % 10

/-- Theorem: If c is a 2-digit positive integer where the sum of its digits is 10
    and the product of its digits is 25, then c = 55 -/
theorem two_digit_sum_product (c : ℕ) : 
  10 ≤ c ∧ c ≤ 99 ∧ 
  tens_digit c + ones_digit c = 10 ∧
  tens_digit c * ones_digit c = 25 →
  c = 55 := by
  sorry


end NUMINAMATH_CALUDE_two_digit_sum_product_l3_316


namespace NUMINAMATH_CALUDE_sqrt_720_equals_12_sqrt_5_l3_397

theorem sqrt_720_equals_12_sqrt_5 : Real.sqrt 720 = 12 * Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_720_equals_12_sqrt_5_l3_397


namespace NUMINAMATH_CALUDE_high_school_elite_season_games_l3_359

/-- The number of teams in the "High School Elite" basketball league -/
def num_teams : ℕ := 8

/-- The number of times each team plays every other team -/
def games_per_pairing : ℕ := 3

/-- The number of games each team plays against non-conference opponents -/
def non_conference_games : ℕ := 5

/-- The total number of games in a season for the "High School Elite" league -/
def total_games : ℕ := (num_teams.choose 2 * games_per_pairing) + (num_teams * non_conference_games)

theorem high_school_elite_season_games :
  total_games = 124 := by sorry

end NUMINAMATH_CALUDE_high_school_elite_season_games_l3_359


namespace NUMINAMATH_CALUDE_bake_sale_donation_l3_303

/-- Calculates the total donation to the homeless shelter from a bake sale fundraiser --/
theorem bake_sale_donation (total_earnings : ℕ) (ingredient_cost : ℕ) (personal_donation : ℕ) : 
  total_earnings = 400 →
  ingredient_cost = 100 →
  personal_donation = 10 →
  ((total_earnings - ingredient_cost) / 2 + personal_donation : ℕ) = 160 := by
  sorry

end NUMINAMATH_CALUDE_bake_sale_donation_l3_303


namespace NUMINAMATH_CALUDE_fraction_irreducible_l3_308

theorem fraction_irreducible (a : ℤ) : 
  Nat.gcd (Int.natAbs (a^3 + 2*a)) (Int.natAbs (a^4 + 3*a^2 + 1)) = 1 := by
  sorry

end NUMINAMATH_CALUDE_fraction_irreducible_l3_308


namespace NUMINAMATH_CALUDE_arccos_equation_solution_l3_353

theorem arccos_equation_solution :
  ∃! x : ℝ, 
    x = 1 / (2 * Real.sqrt (19 - 8 * Real.sqrt 2)) ∧
    Real.arccos (4 * x) - Real.arccos (2 * x) = π / 4 ∧
    0 ≤ 4 * x ∧ 4 * x ≤ 1 ∧
    0 ≤ 2 * x ∧ 2 * x ≤ 1 :=
by sorry

end NUMINAMATH_CALUDE_arccos_equation_solution_l3_353


namespace NUMINAMATH_CALUDE_product_469111_9999_l3_385

theorem product_469111_9999 : 469111 * 9999 = 4690418889 := by
  sorry

end NUMINAMATH_CALUDE_product_469111_9999_l3_385


namespace NUMINAMATH_CALUDE_consecutive_integers_square_difference_l3_388

theorem consecutive_integers_square_difference (n : ℕ) : 
  n > 0 ∧ n * (n + 1) = 2720 → (n + 1)^2 - n^2 = 103 := by
  sorry

end NUMINAMATH_CALUDE_consecutive_integers_square_difference_l3_388


namespace NUMINAMATH_CALUDE_banana_arrangements_l3_389

theorem banana_arrangements :
  let total_letters : ℕ := 6
  let freq_b : ℕ := 1
  let freq_n : ℕ := 2
  let freq_a : ℕ := 3
  (total_letters = freq_b + freq_n + freq_a) →
  (Nat.factorial total_letters) / (Nat.factorial freq_b * Nat.factorial freq_n * Nat.factorial freq_a) = 60 :=
by sorry

end NUMINAMATH_CALUDE_banana_arrangements_l3_389


namespace NUMINAMATH_CALUDE_sin_negative_sixty_degrees_l3_391

theorem sin_negative_sixty_degrees : Real.sin (-(60 * π / 180)) = -Real.sqrt 3 / 2 := by
  sorry

end NUMINAMATH_CALUDE_sin_negative_sixty_degrees_l3_391


namespace NUMINAMATH_CALUDE_problem_statement_l3_323

theorem problem_statement (x y z w : ℝ) 
  (h : (x - y) * (z - w) / ((y - z) * (w - x)) = 3 / 7) :
  (x - z) * (y - w) / ((x - y) * (z - w)) = -1 := by
sorry

end NUMINAMATH_CALUDE_problem_statement_l3_323


namespace NUMINAMATH_CALUDE_equation_solutions_l3_322

theorem equation_solutions (x : ℝ) :
  x ∈ Set.Ioo 0 π ∧ (Real.sin x + Real.cos x) * Real.tan x = 2 * Real.cos x ↔
  x = (1/2) * (Real.arctan 3 + Real.arcsin (Real.sqrt 10 / 10)) ∨
  x = (1/2) * (π - Real.arcsin (Real.sqrt 10 / 10) + Real.arctan 3) :=
by sorry

end NUMINAMATH_CALUDE_equation_solutions_l3_322


namespace NUMINAMATH_CALUDE_complex_equation_solution_l3_375

theorem complex_equation_solution (z : ℂ) (h : z * (1 - Complex.I) = 2) : z = 1 + Complex.I := by
  sorry

end NUMINAMATH_CALUDE_complex_equation_solution_l3_375


namespace NUMINAMATH_CALUDE_power_of_ten_zeros_l3_345

theorem power_of_ten_zeros (n : ℕ) : ∃ k : ℕ, (5000^50) * 100^2 = k * 10^154 ∧ 10^154 ≤ k ∧ k < 10^155 := by
  sorry

end NUMINAMATH_CALUDE_power_of_ten_zeros_l3_345


namespace NUMINAMATH_CALUDE_odd_implies_exists_zero_sum_but_not_conversely_l3_356

/-- A function f: ℝ → ℝ is odd if f(-x) = -f(x) for all x ∈ ℝ -/
def IsOdd (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = -f x

/-- The property that there exists an x such that f(x) + f(-x) = 0 -/
def ExistsZeroSum (f : ℝ → ℝ) : Prop :=
  ∃ x, f x + f (-x) = 0

/-- Theorem stating that if a function is odd, then there exists an x such that f(x) + f(-x) = 0,
    but the converse is not necessarily true -/
theorem odd_implies_exists_zero_sum_but_not_conversely :
  (∃ f : ℝ → ℝ, IsOdd f → ExistsZeroSum f) ∧
  (∃ g : ℝ → ℝ, ExistsZeroSum g ∧ ¬IsOdd g) :=
sorry

end NUMINAMATH_CALUDE_odd_implies_exists_zero_sum_but_not_conversely_l3_356


namespace NUMINAMATH_CALUDE_reynas_lamps_l3_371

/-- The number of light bulbs in each lamp -/
def bulbs_per_lamp : ℕ := 7

/-- The fraction of lamps with burnt-out bulbs -/
def fraction_with_burnt_bulbs : ℚ := 1 / 4

/-- The number of burnt-out bulbs in lamps with burnt-out bulbs -/
def burnt_bulbs_per_lamp : ℕ := 2

/-- The total number of working light bulbs -/
def total_working_bulbs : ℕ := 130

/-- The number of lamps Reyna has -/
def num_lamps : ℕ := 20

theorem reynas_lamps :
  (bulbs_per_lamp * num_lamps : ℚ) * (1 - fraction_with_burnt_bulbs) +
  (bulbs_per_lamp - burnt_bulbs_per_lamp : ℚ) * num_lamps * fraction_with_burnt_bulbs =
  total_working_bulbs := by sorry

end NUMINAMATH_CALUDE_reynas_lamps_l3_371


namespace NUMINAMATH_CALUDE_fifth_power_sum_equality_l3_398

theorem fifth_power_sum_equality : ∃ n : ℕ+, 120^5 + 97^5 + 79^5 + 44^5 = n^5 ∧ n = 144 := by
  sorry

end NUMINAMATH_CALUDE_fifth_power_sum_equality_l3_398


namespace NUMINAMATH_CALUDE_floor_square_minus_floor_product_l3_346

theorem floor_square_minus_floor_product (x : ℝ) : x = 13.2 →
  ⌊x^2⌋ - ⌊x⌋ * ⌊x⌋ = 5 := by sorry

end NUMINAMATH_CALUDE_floor_square_minus_floor_product_l3_346


namespace NUMINAMATH_CALUDE_consecutive_integers_sqrt_19_l3_301

theorem consecutive_integers_sqrt_19 (a b : ℤ) : 
  (b = a + 1) → (a < Real.sqrt 19) → (Real.sqrt 19 < b) → (a + b = 9) := by
  sorry

end NUMINAMATH_CALUDE_consecutive_integers_sqrt_19_l3_301


namespace NUMINAMATH_CALUDE_min_distance_line_circle_min_distance_specific_line_circle_l3_381

/-- Given a line and a circle in a 2D plane, this theorem states that 
    the minimum distance between any point on the line and any point on the circle 
    is equal to the difference between the distance from the circle's center 
    to the line and the radius of the circle. -/
theorem min_distance_line_circle (a b c d e f : ℝ) :
  let line := {p : ℝ × ℝ | a * p.1 + b * p.2 + c = 0}
  let circle := {p : ℝ × ℝ | (p.1 - d)^2 + (p.2 - e)^2 = f^2}
  let center := (d, e)
  let radius := f
  let dist_center_to_line := |a * d + b * e + c| / Real.sqrt (a^2 + b^2)
  ∃ (p : ℝ × ℝ) (q : ℝ × ℝ), p ∈ line ∧ q ∈ circle ∧
    ∀ (p' : ℝ × ℝ) (q' : ℝ × ℝ), p' ∈ line → q' ∈ circle →
      dist_center_to_line - radius ≤ Real.sqrt ((p'.1 - q'.1)^2 + (p'.2 - q'.2)^2) :=
by sorry

/-- The minimum distance between a point on the line 2x + y - 6 = 0
    and a point on the circle (x-1)² + (y+2)² = 5 is √5/5. -/
theorem min_distance_specific_line_circle :
  let line := {p : ℝ × ℝ | 2 * p.1 + p.2 - 6 = 0}
  let circle := {p : ℝ × ℝ | (p.1 - 1)^2 + (p.2 + 2)^2 = 5}
  ∃ (p : ℝ × ℝ) (q : ℝ × ℝ), p ∈ line ∧ q ∈ circle ∧
    ∀ (p' : ℝ × ℝ) (q' : ℝ × ℝ), p' ∈ line → q' ∈ circle →
      Real.sqrt 5 / 5 ≤ Real.sqrt ((p'.1 - q'.1)^2 + (p'.2 - q'.2)^2) :=
by sorry

end NUMINAMATH_CALUDE_min_distance_line_circle_min_distance_specific_line_circle_l3_381


namespace NUMINAMATH_CALUDE_carly_running_ratio_l3_326

/-- Carly's running schedule over four weeks -/
def running_schedule (r : ℚ) : Fin 4 → ℚ
  | 0 => 2                    -- First week
  | 1 => 2 * r + 3            -- Second week
  | 2 => 9/7 * (2 * r + 3)    -- Third week
  | 3 => 4                    -- Fourth week

theorem carly_running_ratio :
  ∃ r : ℚ,
    running_schedule r 2 = 9 ∧
    running_schedule r 3 = running_schedule r 2 - 5 ∧
    running_schedule r 1 / running_schedule r 0 = 7/2 := by
  sorry

end NUMINAMATH_CALUDE_carly_running_ratio_l3_326


namespace NUMINAMATH_CALUDE_sin_145_cos_35_l3_387

theorem sin_145_cos_35 :
  Real.sin (145 * π / 180) * Real.cos (35 * π / 180) = (1/2) * Real.sin (70 * π / 180) :=
by sorry

end NUMINAMATH_CALUDE_sin_145_cos_35_l3_387


namespace NUMINAMATH_CALUDE_zeros_product_greater_than_e_squared_l3_306

/-- Given a function f(x) = ax² - bx + ln x where a and b are real numbers,
    and g(x) = f(x) - ax² = -bx + ln x has two distinct zeros x₁ and x₂,
    prove that x₁x₂ > e² -/
theorem zeros_product_greater_than_e_squared (a b x₁ x₂ : ℝ) 
  (h₁ : x₁ ≠ x₂) 
  (h₂ : -b * x₁ + Real.log x₁ = 0) 
  (h₃ : -b * x₂ + Real.log x₂ = 0) : 
  x₁ * x₂ > Real.exp 2 := by
  sorry

end NUMINAMATH_CALUDE_zeros_product_greater_than_e_squared_l3_306


namespace NUMINAMATH_CALUDE_correct_categorization_l3_311

-- Define the given numbers
def numbers : List ℚ := [-2/9, -9, -301, -314/100, 2004, 0, 22/7]

-- Define the sets
def fractions : Set ℚ := {x | x ∈ numbers ∧ x ≠ 0 ∧ ∃ (a b : ℤ), b ≠ 0 ∧ x = a / b}
def negative_fractions : Set ℚ := {x | x ∈ fractions ∧ x < 0}
def integers : Set ℚ := {x | x ∈ numbers ∧ ∃ (n : ℤ), x = n}
def positive_integers : Set ℚ := {x | x ∈ integers ∧ x > 0}
def positive_rationals : Set ℚ := {x | x ∈ numbers ∧ x > 0 ∧ ∃ (a b : ℤ), b > 0 ∧ x = a / b}

-- State the theorem
theorem correct_categorization :
  fractions = {-2/9, 22/7} ∧
  negative_fractions = {-2/9} ∧
  integers = {-9, -301, 2004, 0} ∧
  positive_integers = {2004} ∧
  positive_rationals = {2004, 22/7} :=
sorry

end NUMINAMATH_CALUDE_correct_categorization_l3_311


namespace NUMINAMATH_CALUDE_number_of_observations_l3_324

theorem number_of_observations (initial_mean : ℝ) (wrong_value : ℝ) (correct_value : ℝ) (new_mean : ℝ) : 
  initial_mean = 36 → 
  wrong_value = 23 → 
  correct_value = 46 → 
  new_mean = 36.5 → 
  ∃ n : ℕ, n * initial_mean + (correct_value - wrong_value) = n * new_mean ∧ n = 46 :=
by sorry

end NUMINAMATH_CALUDE_number_of_observations_l3_324


namespace NUMINAMATH_CALUDE_existence_of_divalent_radical_with_bounded_growth_l3_373

/-- A set of positive integers is a divalent radical if any sufficiently large positive integer
    can be expressed as the sum of two elements in the set. -/
def IsDivalentRadical (A : Set ℕ+) : Prop :=
  ∃ N : ℕ+, ∀ n : ℕ+, n ≥ N → ∃ a b : ℕ+, a ∈ A ∧ b ∈ A ∧ (a : ℕ) + b = n

/-- A(x) is the set of all elements in A that do not exceed x -/
def ASubset (A : Set ℕ+) (x : ℝ) : Set ℕ+ :=
  {a ∈ A | (a : ℝ) ≤ x}

theorem existence_of_divalent_radical_with_bounded_growth :
  ∃ (A : Set ℕ+) (C : ℝ), A.Nonempty ∧ IsDivalentRadical A ∧
    ∀ x : ℝ, x ≥ 1 → (ASubset A x).ncard ≤ C * Real.sqrt x :=
sorry

end NUMINAMATH_CALUDE_existence_of_divalent_radical_with_bounded_growth_l3_373


namespace NUMINAMATH_CALUDE_unique_magnitude_quadratic_l3_396

theorem unique_magnitude_quadratic :
  ∃! m : ℝ, ∃ w : ℂ, w^2 - 6*w + 40 = 0 ∧ Complex.abs w = m := by sorry

end NUMINAMATH_CALUDE_unique_magnitude_quadratic_l3_396


namespace NUMINAMATH_CALUDE_sin_30_degrees_l3_305

theorem sin_30_degrees : Real.sin (30 * π / 180) = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_sin_30_degrees_l3_305


namespace NUMINAMATH_CALUDE_students_not_reading_l3_334

theorem students_not_reading (total_students : ℕ) (girls : ℕ) (boys : ℕ) 
  (girls_reading_fraction : ℚ) (boys_reading_fraction : ℚ) :
  total_students = girls + boys →
  girls = 12 →
  boys = 10 →
  girls_reading_fraction = 5/6 →
  boys_reading_fraction = 4/5 →
  total_students - (↑girls * girls_reading_fraction).floor - (↑boys * boys_reading_fraction).floor = 4 := by
  sorry

end NUMINAMATH_CALUDE_students_not_reading_l3_334


namespace NUMINAMATH_CALUDE_tangent_line_intersection_l3_383

theorem tangent_line_intersection (f : ℝ → ℝ) (x₀ y₀ : ℝ) :
  f x₀ = y₀ →
  (∀ x, f x = x^3 + 11) →
  x₀ = 1 →
  y₀ = 12 →
  ∃ m : ℝ, ∀ x y, y - y₀ = m * (x - x₀) →
    y = 0 →
    x = -3 :=
by sorry

end NUMINAMATH_CALUDE_tangent_line_intersection_l3_383


namespace NUMINAMATH_CALUDE_problem_solution_l3_399

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := Real.log x + (1 - x) / (a * x)

theorem problem_solution (a : ℝ) (h : a > 0) :
  (∀ x ≥ 1, Monotone (f a) → a ≥ 1) ∧
  (∀ x ∈ Set.Icc 1 2,
    (0 < a ∧ a ≤ 1/2 → f a x ≥ Real.log 2 - 1/(2*a)) ∧
    (1/2 < a ∧ a < 1 → f a x ≥ Real.log (1/a) + 1 - 1/a) ∧
    (a ≥ 1 → f a x ≥ 0)) ∧
  (∀ n : ℕ, n > 1 → Real.log n > (Finset.range (n-1)).sum (λ i => 1 / (i + 2))) :=
sorry

end NUMINAMATH_CALUDE_problem_solution_l3_399


namespace NUMINAMATH_CALUDE_scale_division_l3_300

/-- Represents the length of an object in feet and inches -/
structure Length where
  feet : ℕ
  inches : ℕ
  h : inches < 12

/-- Converts a Length to total inches -/
def Length.toInches (l : Length) : ℕ := l.feet * 12 + l.inches

/-- Converts total inches to a Length -/
def inchesToLength (totalInches : ℕ) : Length :=
  { feet := totalInches / 12,
    inches := totalInches % 12,
    h := by
      apply Nat.mod_lt
      exact Nat.zero_lt_succ 11 }

theorem scale_division (scale : Length) (h : scale.feet = 6 ∧ scale.inches = 8) :
  let totalInches := scale.toInches
  let halfInches := totalInches / 2
  let halfLength := inchesToLength halfInches
  halfLength.feet = 3 ∧ halfLength.inches = 4 := by
  sorry


end NUMINAMATH_CALUDE_scale_division_l3_300


namespace NUMINAMATH_CALUDE_movie_length_after_cut_l3_394

/-- Calculates the final length of a movie after cutting a scene -/
theorem movie_length_after_cut (original_length cut_length : ℕ) : 
  original_length = 60 → cut_length = 3 → original_length - cut_length = 57 := by
  sorry

end NUMINAMATH_CALUDE_movie_length_after_cut_l3_394

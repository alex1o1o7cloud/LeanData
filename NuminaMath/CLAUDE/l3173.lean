import Mathlib

namespace NUMINAMATH_CALUDE_smallest_number_with_digit_sum_2017_properties_l3173_317337

/-- The smallest natural number with digit sum 2017 -/
def smallest_number_with_digit_sum_2017 : ℕ :=
  1 * 10^224 + (10^224 - 1)

/-- The digit sum of a natural number -/
def digit_sum (n : ℕ) : ℕ :=
  sorry

/-- The number of digits in a natural number -/
def num_digits (n : ℕ) : ℕ :=
  sorry

theorem smallest_number_with_digit_sum_2017_properties :
  digit_sum smallest_number_with_digit_sum_2017 = 2017 ∧
  num_digits smallest_number_with_digit_sum_2017 = 225 ∧
  smallest_number_with_digit_sum_2017 / 10^224 = 1 ∧
  ∀ m : ℕ, m < smallest_number_with_digit_sum_2017 → digit_sum m ≠ 2017 :=
by sorry

end NUMINAMATH_CALUDE_smallest_number_with_digit_sum_2017_properties_l3173_317337


namespace NUMINAMATH_CALUDE_unfair_die_expected_value_is_nine_eighths_l3173_317397

def unfair_die_expected_value (p1 p2 p3 p4 p5 : ℚ) : ℚ :=
  let p6 := 1 - (p1 + p2 + p3 + p4 + p5)
  1 * p1 + 2 * p2 + 3 * p3 + 4 * p4 + 5 * p5 + 6 * p6

theorem unfair_die_expected_value_is_nine_eighths :
  unfair_die_expected_value (1/6) (1/8) (1/12) (1/12) (1/12) = 9/8 := by
  sorry

#eval unfair_die_expected_value (1/6) (1/8) (1/12) (1/12) (1/12)

end NUMINAMATH_CALUDE_unfair_die_expected_value_is_nine_eighths_l3173_317397


namespace NUMINAMATH_CALUDE_constant_term_expansion_l3173_317363

theorem constant_term_expansion (x : ℝ) : 
  let expansion := (x + 2 / Real.sqrt x) ^ 6
  ∃ (coefficient : ℝ), coefficient = 240 ∧ 
    (∃ (other_terms : ℝ → ℝ), expansion = coefficient + other_terms x ∧ 
      (∀ y : ℝ, other_terms y ≠ 0 → y ≠ 0)) :=
by sorry

end NUMINAMATH_CALUDE_constant_term_expansion_l3173_317363


namespace NUMINAMATH_CALUDE_opposite_of_a_is_smallest_positive_integer_l3173_317351

theorem opposite_of_a_is_smallest_positive_integer (a : ℤ) : 
  (∃ (x : ℤ), x > 0 ∧ ∀ (y : ℤ), y > 0 → x ≤ y) ∧ (-a = x) → 3*a - 2 = -5 := by
  sorry

end NUMINAMATH_CALUDE_opposite_of_a_is_smallest_positive_integer_l3173_317351


namespace NUMINAMATH_CALUDE_solution_set_abs_inequality_l3173_317375

theorem solution_set_abs_inequality :
  {x : ℝ | |1 - 2*x| < 3} = Set.Ioo (-1) 2 := by
  sorry

end NUMINAMATH_CALUDE_solution_set_abs_inequality_l3173_317375


namespace NUMINAMATH_CALUDE_power_function_conditions_l3173_317309

def α_set : Set ℚ := {-1, 1, 2, 3/5, 7/2}

def is_odd_function (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = -f x

def has_domain_R (f : ℝ → ℝ) : Prop :=
  ∀ x, ∃ y, f x = y

def satisfies_conditions (α : ℚ) : Prop :=
  let f := fun x => x ^ (α : ℝ)
  has_domain_R f ∧ is_odd_function f

theorem power_function_conditions :
  ∀ α ∈ α_set, satisfies_conditions α ↔ α ∈ ({1, 3/5} : Set ℚ) :=
sorry

end NUMINAMATH_CALUDE_power_function_conditions_l3173_317309


namespace NUMINAMATH_CALUDE_equation_solution_range_l3173_317318

theorem equation_solution_range (x a : ℝ) : 
  (2 * x + a) / (x - 1) = 1 → x > 0 → x ≠ 1 → a < -1 ∧ a ≠ -2 :=
by sorry

end NUMINAMATH_CALUDE_equation_solution_range_l3173_317318


namespace NUMINAMATH_CALUDE_melissa_games_played_l3173_317323

/-- Given that Melissa scored 12 points in each game and a total of 36 points,
    prove that she played 3 games. -/
theorem melissa_games_played (points_per_game : ℕ) (total_points : ℕ) 
  (h1 : points_per_game = 12) 
  (h2 : total_points = 36) : 
  total_points / points_per_game = 3 := by
  sorry

end NUMINAMATH_CALUDE_melissa_games_played_l3173_317323


namespace NUMINAMATH_CALUDE_curve_is_two_intersecting_lines_l3173_317353

/-- The equation of the curve -/
def curve_equation (x y : ℝ) : Prop :=
  2 * x^2 - y^2 - 4 * x - 4 * y - 2 = 0

/-- The first line equation derived from the curve equation -/
def line1 (x y : ℝ) : Prop :=
  y = Real.sqrt 2 * x - Real.sqrt 2 - 2

/-- The second line equation derived from the curve equation -/
def line2 (x y : ℝ) : Prop :=
  y = -Real.sqrt 2 * x + Real.sqrt 2 - 2

/-- Theorem stating that the curve equation represents two intersecting lines -/
theorem curve_is_two_intersecting_lines :
  ∃ (x₁ y₁ x₂ y₂ : ℝ), 
    (∀ x y, curve_equation x y ↔ (line1 x y ∨ line2 x y)) ∧ 
    (line1 x₁ y₁ ∧ line2 x₁ y₁) ∧
    (x₁ ≠ x₂ ∨ y₁ ≠ y₂) :=
  sorry

end NUMINAMATH_CALUDE_curve_is_two_intersecting_lines_l3173_317353


namespace NUMINAMATH_CALUDE_odd_power_decomposition_l3173_317322

theorem odd_power_decomposition (m : ℤ) : 
  ∃ (a b k : ℤ), Odd a ∧ Odd b ∧ k ≥ 0 ∧ 2 * m = a^19 + b^99 + k * 2^1999 := by
  sorry

end NUMINAMATH_CALUDE_odd_power_decomposition_l3173_317322


namespace NUMINAMATH_CALUDE_inverse_32_mod_97_l3173_317386

theorem inverse_32_mod_97 (h : (2⁻¹ : ZMod 97) = 49) : (32⁻¹ : ZMod 97) = 49 := by
  sorry

end NUMINAMATH_CALUDE_inverse_32_mod_97_l3173_317386


namespace NUMINAMATH_CALUDE_discount_percentage_l3173_317376

theorem discount_percentage (M : ℝ) (C : ℝ) (S : ℝ) 
  (h1 : C = 0.64 * M) 
  (h2 : S = C * 1.28125) : 
  (M - S) / M * 100 = 18.08 := by
  sorry

end NUMINAMATH_CALUDE_discount_percentage_l3173_317376


namespace NUMINAMATH_CALUDE_apps_deleted_l3173_317350

/-- Given that Dave initially had 150 apps on his phone and 65 apps remained after deletion,
    prove that the number of apps deleted is 85. -/
theorem apps_deleted (initial_apps : ℕ) (remaining_apps : ℕ) (h1 : initial_apps = 150) (h2 : remaining_apps = 65) :
  initial_apps - remaining_apps = 85 := by
  sorry

end NUMINAMATH_CALUDE_apps_deleted_l3173_317350


namespace NUMINAMATH_CALUDE_joes_dad_marshmallows_joes_dad_marshmallows_proof_l3173_317303

theorem joes_dad_marshmallows : ℕ → Prop :=
  fun d : ℕ =>
    let joe_marshmallows : ℕ := 4 * d
    let dad_roasted : ℕ := d / 3
    let joe_roasted : ℕ := joe_marshmallows / 2
    dad_roasted + joe_roasted = 49 → d = 21

-- The proof goes here
theorem joes_dad_marshmallows_proof : joes_dad_marshmallows 21 := by
  sorry

end NUMINAMATH_CALUDE_joes_dad_marshmallows_joes_dad_marshmallows_proof_l3173_317303


namespace NUMINAMATH_CALUDE_function_symmetry_l3173_317317

theorem function_symmetry (a : ℝ) : 
  let f : ℝ → ℝ := λ x => (x^2 + x + 1) / (x^2 + 1)
  f a = 2/3 → f (-a) = 4/3 := by
sorry

end NUMINAMATH_CALUDE_function_symmetry_l3173_317317


namespace NUMINAMATH_CALUDE_sum_of_digits_l3173_317314

def is_valid_number (n : ℕ) : Prop :=
  ∃ (a b c : ℕ),
    n = 100 * a + 10 * b + c ∧
    b = a + c ∧
    100 * c + 10 * b + a = n + 99 ∧
    n = 253

theorem sum_of_digits (n : ℕ) (h : is_valid_number n) : 
  ∃ (a b c : ℕ), n = 100 * a + 10 * b + c ∧ a + b + c = 10 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_digits_l3173_317314


namespace NUMINAMATH_CALUDE_choose_captains_l3173_317393

theorem choose_captains (n : ℕ) (k : ℕ) (h1 : n = 15) (h2 : k = 4) :
  Nat.choose n k = 1365 := by
  sorry

end NUMINAMATH_CALUDE_choose_captains_l3173_317393


namespace NUMINAMATH_CALUDE_feifei_leilei_age_sum_feifei_leilei_age_sum_proof_l3173_317387

theorem feifei_leilei_age_sum : ℕ → ℕ → Prop :=
  fun feifei_age leilei_age =>
    (feifei_age = leilei_age / 2 + 12) →
    (feifei_age + 1 = 2 * (leilei_age + 1) - 34) →
    (feifei_age + leilei_age = 57)

theorem feifei_leilei_age_sum_proof : ∃ (f l : ℕ), feifei_leilei_age_sum f l :=
  sorry

end NUMINAMATH_CALUDE_feifei_leilei_age_sum_feifei_leilei_age_sum_proof_l3173_317387


namespace NUMINAMATH_CALUDE_sin_two_x_value_l3173_317315

theorem sin_two_x_value (x : ℝ) 
  (h : Real.sin (Real.pi + x) + Real.sin ((3 * Real.pi) / 2 + x) = 1/2) : 
  Real.sin (2 * x) = -(3/4) := by
  sorry

end NUMINAMATH_CALUDE_sin_two_x_value_l3173_317315


namespace NUMINAMATH_CALUDE_rachel_distance_to_nicholas_l3173_317329

/-- The distance between two points given speed and time -/
def distance (speed : ℝ) (time : ℝ) : ℝ := speed * time

/-- Theorem: Rachel's distance to Nicholas's house -/
theorem rachel_distance_to_nicholas : distance 2 5 = 10 := by
  sorry

end NUMINAMATH_CALUDE_rachel_distance_to_nicholas_l3173_317329


namespace NUMINAMATH_CALUDE_johns_payment_is_1500_l3173_317374

/-- Calculates the personal payment for hearing aids given insurance details --/
def calculate_personal_payment (cost_per_aid : ℕ) (num_aids : ℕ) (deductible : ℕ) 
  (coverage_percent : ℚ) (coverage_limit : ℕ) : ℕ :=
  let total_cost := cost_per_aid * num_aids
  let after_deductible := total_cost - deductible
  let insurance_payment := min (coverage_limit) (↑(Nat.floor (coverage_percent * ↑after_deductible)))
  total_cost - insurance_payment

/-- Theorem stating that John's personal payment for hearing aids is $1500 --/
theorem johns_payment_is_1500 : 
  calculate_personal_payment 2500 2 500 (4/5) 3500 = 1500 := by
  sorry

end NUMINAMATH_CALUDE_johns_payment_is_1500_l3173_317374


namespace NUMINAMATH_CALUDE_p_necessary_not_sufficient_for_q_l3173_317361

theorem p_necessary_not_sufficient_for_q :
  (∃ x : ℝ, x < 1 ∧ ¬(x^2 + x - 2 < 0)) ∧
  (∀ x : ℝ, x^2 + x - 2 < 0 → x < 1) :=
by sorry

end NUMINAMATH_CALUDE_p_necessary_not_sufficient_for_q_l3173_317361


namespace NUMINAMATH_CALUDE_definite_integral_3x_minus_sinx_l3173_317313

theorem definite_integral_3x_minus_sinx : 
  ∫ x in (0)..(π/2), (3*x - Real.sin x) = 3*π^2/8 - 1 := by sorry

end NUMINAMATH_CALUDE_definite_integral_3x_minus_sinx_l3173_317313


namespace NUMINAMATH_CALUDE_prime_p_cube_condition_l3173_317321

theorem prime_p_cube_condition (p : ℕ) : 
  Prime p → (∃ n : ℕ, 13 * p + 1 = n^3) → p = 2 ∨ p = 211 := by
sorry

end NUMINAMATH_CALUDE_prime_p_cube_condition_l3173_317321


namespace NUMINAMATH_CALUDE_adam_cat_food_packages_l3173_317307

theorem adam_cat_food_packages : 
  ∀ (c : ℕ), -- c represents the number of packages of cat food
  (10 * c = 7 * 5 + 55) → c = 9 :=
by
  sorry

end NUMINAMATH_CALUDE_adam_cat_food_packages_l3173_317307


namespace NUMINAMATH_CALUDE_smallest_integer_satisfying_inequalities_l3173_317310

theorem smallest_integer_satisfying_inequalities :
  ∀ x : ℤ, (x + 8 > 10 ∧ -3*x < -9) → x ≥ 4 :=
by sorry

end NUMINAMATH_CALUDE_smallest_integer_satisfying_inequalities_l3173_317310


namespace NUMINAMATH_CALUDE_line_x_intercept_l3173_317385

theorem line_x_intercept (t : ℝ) (h : t ∈ Set.Icc 0 (2 * Real.pi)) :
  let x := 2 * Real.cos t + 3
  let y := -1 + 5 * Real.sin t
  y = 0 → Real.sin t = 1/5 ∧ x = 2 * Real.cos (Real.arcsin (1/5)) + 3 := by
  sorry

end NUMINAMATH_CALUDE_line_x_intercept_l3173_317385


namespace NUMINAMATH_CALUDE_three_possible_values_for_sum_l3173_317398

theorem three_possible_values_for_sum (x y : ℤ) 
  (h : x^2 + y^2 + 1 ≤ 2*x + 2*y) : 
  ∃ (S : Finset ℤ), (Finset.card S = 3) ∧ ((x + y) ∈ S) :=
sorry

end NUMINAMATH_CALUDE_three_possible_values_for_sum_l3173_317398


namespace NUMINAMATH_CALUDE_compute_expression_l3173_317391

theorem compute_expression : 6^2 - 4*5 + 4^2 = 32 := by
  sorry

end NUMINAMATH_CALUDE_compute_expression_l3173_317391


namespace NUMINAMATH_CALUDE_det_value_for_quadratic_root_l3173_317340

-- Define the determinant function for a 2x2 matrix
def det (a b c d : ℝ) : ℝ := a * d - b * c

-- State the theorem
theorem det_value_for_quadratic_root (x : ℝ) (h : x^2 - 3*x + 1 = 0) :
  det (x + 1) (3*x) (x - 2) (x - 1) = 1 := by
  sorry

end NUMINAMATH_CALUDE_det_value_for_quadratic_root_l3173_317340


namespace NUMINAMATH_CALUDE_rayden_lily_duck_ratio_l3173_317333

/-- Proves the ratio of Rayden's ducks to Lily's ducks is 3:1 -/
theorem rayden_lily_duck_ratio :
  let lily_ducks : ℕ := 20
  let lily_geese : ℕ := 10
  let rayden_geese : ℕ := 4 * lily_geese
  let total_difference : ℕ := 70
  let rayden_total : ℕ := lily_ducks + lily_geese + total_difference
  let rayden_ducks : ℕ := rayden_total - rayden_geese
  (rayden_ducks : ℚ) / lily_ducks = 3 / 1 := by sorry

end NUMINAMATH_CALUDE_rayden_lily_duck_ratio_l3173_317333


namespace NUMINAMATH_CALUDE_cos_equality_implies_angle_l3173_317305

theorem cos_equality_implies_angle (n : ℤ) : 
  0 ≤ n ∧ n ≤ 180 → Real.cos (n * π / 180) = Real.cos (315 * π / 180) → n = 45 := by
  sorry

end NUMINAMATH_CALUDE_cos_equality_implies_angle_l3173_317305


namespace NUMINAMATH_CALUDE_intersection_theorem_l3173_317388

/-- The number of intersection points between two curves -/
def intersection_count (a : ℝ) : ℕ := sorry

/-- First curve equation: x^2 + y^2 = 4a^2 -/
def curve1 (a x y : ℝ) : Prop := x^2 + y^2 = 4 * a^2

/-- Second curve equation: y = x^2 - 4a + 1 -/
def curve2 (a x y : ℝ) : Prop := y = x^2 - 4 * a + 1

theorem intersection_theorem (a : ℝ) :
  intersection_count a = 3 ↔ a > 1/8 := by sorry

end NUMINAMATH_CALUDE_intersection_theorem_l3173_317388


namespace NUMINAMATH_CALUDE_joohee_ate_17_chocolates_l3173_317396

-- Define the total number of chocolates
def total_chocolates : ℕ := 25

-- Define the relationship between Joo-hee's and Jun-seong's chocolates
def joohee_chocolates (junseong_chocolates : ℕ) : ℕ :=
  2 * junseong_chocolates + 1

-- Theorem statement
theorem joohee_ate_17_chocolates :
  ∃ (junseong_chocolates : ℕ),
    junseong_chocolates + joohee_chocolates junseong_chocolates = total_chocolates ∧
    joohee_chocolates junseong_chocolates = 17 :=
  sorry

end NUMINAMATH_CALUDE_joohee_ate_17_chocolates_l3173_317396


namespace NUMINAMATH_CALUDE_ratio_is_five_l3173_317301

/-- The equation holds for all real x except -3, 0, and 6 -/
def equation_holds (P Q : ℤ) : Prop :=
  ∀ x : ℝ, x ≠ -3 ∧ x ≠ 0 ∧ x ≠ 6 →
    (P : ℝ) / (x + 3) + (Q : ℝ) / (x^2 - 6*x) = (x^2 - 4*x + 15) / (x^3 + x^2 - 18*x)

theorem ratio_is_five (P Q : ℤ) (h : equation_holds P Q) : (Q : ℚ) / P = 5 := by
  sorry

end NUMINAMATH_CALUDE_ratio_is_five_l3173_317301


namespace NUMINAMATH_CALUDE_house_height_l3173_317342

theorem house_height (house_shadow : ℝ) (pole_height : ℝ) (pole_shadow : ℝ)
  (h1 : house_shadow = 84)
  (h2 : pole_height = 14)
  (h3 : pole_shadow = 28) :
  (house_shadow / pole_shadow) * pole_height = 42 :=
by sorry

end NUMINAMATH_CALUDE_house_height_l3173_317342


namespace NUMINAMATH_CALUDE_coffee_cheesecake_set_price_l3173_317338

/-- Calculates the discounted price of a coffee and cheesecake set --/
def discounted_set_price (coffee_price cheesecake_price : ℚ) (discount_rate : ℚ) : ℚ :=
  let total_price := coffee_price + cheesecake_price
  let discount_amount := discount_rate * total_price
  total_price - discount_amount

/-- Proves that the final price of a coffee and cheesecake set with a 25% discount is $12 --/
theorem coffee_cheesecake_set_price :
  discounted_set_price 6 10 (25 / 100) = 12 :=
by sorry

end NUMINAMATH_CALUDE_coffee_cheesecake_set_price_l3173_317338


namespace NUMINAMATH_CALUDE_bruce_initial_eggs_l3173_317381

theorem bruce_initial_eggs (bruce_final : ℕ) (eggs_lost : ℕ) : 
  bruce_final = 5 → eggs_lost = 70 → bruce_final + eggs_lost = 75 := by
  sorry

end NUMINAMATH_CALUDE_bruce_initial_eggs_l3173_317381


namespace NUMINAMATH_CALUDE_x_value_theorem_l3173_317320

theorem x_value_theorem (x : ℝ) : x * (x * (x + 1) + 2) + 3 = x^3 + x^2 + x - 6 → x = -9 := by
  sorry

end NUMINAMATH_CALUDE_x_value_theorem_l3173_317320


namespace NUMINAMATH_CALUDE_binary_1011001_equals_base5_324_l3173_317356

def binary_to_decimal (b : List Bool) : ℕ :=
  b.enum.foldl (λ acc (i, bit) => acc + if bit then 2^i else 0) 0

def decimal_to_base5 (n : ℕ) : List ℕ :=
  if n = 0 then [0] else
    let rec aux (m : ℕ) (acc : List ℕ) : List ℕ :=
      if m = 0 then acc else aux (m / 5) ((m % 5) :: acc)
    aux n []

theorem binary_1011001_equals_base5_324 : 
  decimal_to_base5 (binary_to_decimal [true, false, false, true, true, false, true]) = [3, 2, 4] := by
  sorry

end NUMINAMATH_CALUDE_binary_1011001_equals_base5_324_l3173_317356


namespace NUMINAMATH_CALUDE_f_decreasing_interval_l3173_317332

def f (x : ℝ) : ℝ := x^2 - 2*x + 4

theorem f_decreasing_interval :
  ∀ x y : ℝ, x < y → y ≤ 1 → f y ≤ f x := by
sorry

end NUMINAMATH_CALUDE_f_decreasing_interval_l3173_317332


namespace NUMINAMATH_CALUDE_extra_page_number_l3173_317324

theorem extra_page_number (n : ℕ) (k : ℕ) : 
  n = 62 → 
  (n * (n + 1)) / 2 + k = 1986 → 
  k = 33 := by
sorry

end NUMINAMATH_CALUDE_extra_page_number_l3173_317324


namespace NUMINAMATH_CALUDE_x_gt_one_sufficient_not_necessary_for_x_gt_zero_l3173_317331

theorem x_gt_one_sufficient_not_necessary_for_x_gt_zero :
  (∀ x : ℝ, x > 1 → x > 0) ∧ 
  (∃ x : ℝ, x > 0 ∧ ¬(x > 1)) :=
by sorry

end NUMINAMATH_CALUDE_x_gt_one_sufficient_not_necessary_for_x_gt_zero_l3173_317331


namespace NUMINAMATH_CALUDE_min_monochromatic_triangles_K15_l3173_317312

/-- A coloring of the edges of a complete graph using two colors. -/
def TwoColoring (n : ℕ) := Fin n → Fin n → Fin 2

/-- The number of monochromatic triangles in a two-colored complete graph. -/
def monochromaticTriangles (n : ℕ) (c : TwoColoring n) : ℕ := sorry

/-- Theorem: The minimum number of monochromatic triangles in K₁₅ is 88. -/
theorem min_monochromatic_triangles_K15 :
  (∃ c : TwoColoring 15, monochromaticTriangles 15 c = 88) ∧
  (∀ c : TwoColoring 15, monochromaticTriangles 15 c ≥ 88) := by
  sorry

end NUMINAMATH_CALUDE_min_monochromatic_triangles_K15_l3173_317312


namespace NUMINAMATH_CALUDE_nice_sequence_classification_l3173_317349

/-- A sequence of integers -/
def IntegerSequence := ℕ → ℤ

/-- A function from positive integers to positive integers -/
def PositiveIntFunction := ℕ+ → ℕ+

/-- A nice sequence satisfies the given condition for some function f -/
def IsNice (a : IntegerSequence) : Prop :=
  ∃ f : PositiveIntFunction, ∀ i j n : ℕ+,
    (a i.val - a j.val) % n.val = 0 ↔ (i.val - j.val) % f n = 0

/-- A sequence is periodic with period k -/
def IsPeriodic (a : IntegerSequence) (k : ℕ+) : Prop :=
  ∀ i : ℕ, a (i + k) = a i

/-- A sequence is an arithmetic sequence -/
def IsArithmetic (a : IntegerSequence) : Prop :=
  ∃ d : ℤ, ∀ i : ℕ, a (i + 1) = a i + d

/-- The main theorem: nice sequences are either constant, periodic with period 2, or arithmetic -/
theorem nice_sequence_classification (a : IntegerSequence) :
  IsNice a → (IsPeriodic a 1 ∨ IsPeriodic a 2 ∨ IsArithmetic a) :=
sorry

end NUMINAMATH_CALUDE_nice_sequence_classification_l3173_317349


namespace NUMINAMATH_CALUDE_constant_sum_implies_parallelogram_l3173_317371

-- Define a convex quadrilateral
structure ConvexQuadrilateral where
  vertices : Fin 4 → ℝ × ℝ
  is_convex : sorry -- Additional condition to ensure convexity

-- Define a function to calculate the distance from a point to a line
def distanceToLine (p : ℝ × ℝ) (l : (ℝ × ℝ) × (ℝ × ℝ)) : ℝ := sorry

-- Define a function to check if a point is inside the quadrilateral
def isInsideQuadrilateral (q : ConvexQuadrilateral) (p : ℝ × ℝ) : Prop := sorry

-- Define a function to calculate the sum of distances from a point to all sides
def sumOfDistances (q : ConvexQuadrilateral) (p : ℝ × ℝ) : ℝ :=
  distanceToLine p (q.vertices 0, q.vertices 1) +
  distanceToLine p (q.vertices 1, q.vertices 2) +
  distanceToLine p (q.vertices 2, q.vertices 3) +
  distanceToLine p (q.vertices 3, q.vertices 0)

-- Define what it means for a quadrilateral to be a parallelogram
def isParallelogram (q : ConvexQuadrilateral) : Prop := sorry

-- The main theorem
theorem constant_sum_implies_parallelogram (q : ConvexQuadrilateral) :
  (∃ k : ℝ, ∀ p : ℝ × ℝ, isInsideQuadrilateral q p → sumOfDistances q p = k) →
  isParallelogram q := by sorry

end NUMINAMATH_CALUDE_constant_sum_implies_parallelogram_l3173_317371


namespace NUMINAMATH_CALUDE_largest_three_digit_multiple_of_13_l3173_317334

theorem largest_three_digit_multiple_of_13 : 
  ∀ n : ℕ, n ≤ 999 → n ≥ 100 → n % 13 = 0 → n ≤ 987 :=
by
  sorry

end NUMINAMATH_CALUDE_largest_three_digit_multiple_of_13_l3173_317334


namespace NUMINAMATH_CALUDE_second_quarter_profit_l3173_317370

theorem second_quarter_profit 
  (annual_profit : ℕ)
  (first_quarter_profit : ℕ)
  (third_quarter_profit : ℕ)
  (fourth_quarter_profit : ℕ)
  (h1 : annual_profit = 8000)
  (h2 : first_quarter_profit = 1500)
  (h3 : third_quarter_profit = 3000)
  (h4 : fourth_quarter_profit = 2000) :
  annual_profit - (first_quarter_profit + third_quarter_profit + fourth_quarter_profit) = 1500 :=
by
  sorry

end NUMINAMATH_CALUDE_second_quarter_profit_l3173_317370


namespace NUMINAMATH_CALUDE_new_person_weight_l3173_317395

theorem new_person_weight (n : ℕ) (avg_increase : ℝ) (old_weight : ℝ) :
  n = 7 ∧ avg_increase = 3.5 ∧ old_weight = 75 →
  (n : ℝ) * avg_increase + old_weight = 99.5 :=
by sorry

end NUMINAMATH_CALUDE_new_person_weight_l3173_317395


namespace NUMINAMATH_CALUDE_regular_polygons_covering_plane_l3173_317358

/-- A function that returns true if a regular n-gon can completely and tightly cover a plane without gaps -/
def can_cover_plane (n : ℕ) : Prop :=
  n ≥ 3 ∧ ∃ k : ℕ, k ≥ 3 ∧ k * (1 - 2 / n) = 2

/-- The theorem stating which regular polygons can completely and tightly cover a plane without gaps -/
theorem regular_polygons_covering_plane :
  ∀ n : ℕ, can_cover_plane n ↔ n = 3 ∨ n = 4 ∨ n = 6 :=
sorry

end NUMINAMATH_CALUDE_regular_polygons_covering_plane_l3173_317358


namespace NUMINAMATH_CALUDE_continuous_multiplicative_function_is_exponential_l3173_317304

open Real

/-- A function f: ℝ → ℝ satisfying the given conditions -/
def ContinuousMultiplicativeFunction (f : ℝ → ℝ) : Prop :=
  Continuous f ∧ f 0 = 1 ∧ ∀ x y, f (x + y) ≥ f x * f y

/-- The main theorem statement -/
theorem continuous_multiplicative_function_is_exponential
  (f : ℝ → ℝ) (hf : ContinuousMultiplicativeFunction f) :
  ∃ a : ℝ, a > 0 ∧ ∀ x, f x = a^x :=
sorry

end NUMINAMATH_CALUDE_continuous_multiplicative_function_is_exponential_l3173_317304


namespace NUMINAMATH_CALUDE_vector_decomposition_l3173_317311

theorem vector_decomposition (x p q r : ℝ × ℝ × ℝ) 
  (hx : x = (-9, -8, -3))
  (hp : p = (1, 4, 1))
  (hq : q = (-3, 2, 0))
  (hr : r = (1, -1, 2)) :
  ∃ (α β γ : ℝ), x = α • p + β • q + γ • r ∧ α = -3 ∧ β = 2 ∧ γ = 0 :=
by sorry

end NUMINAMATH_CALUDE_vector_decomposition_l3173_317311


namespace NUMINAMATH_CALUDE_south_american_stamps_cost_l3173_317360

structure Country where
  name : String
  continent : String
  price : Rat
  stamps_50s : Nat
  stamps_60s : Nat

def brazil : Country := {
  name := "Brazil"
  continent := "South America"
  price := 6/100
  stamps_50s := 4
  stamps_60s := 7
}

def peru : Country := {
  name := "Peru"
  continent := "South America"
  price := 4/100
  stamps_50s := 6
  stamps_60s := 4
}

def france : Country := {
  name := "France"
  continent := "Europe"
  price := 6/100
  stamps_50s := 8
  stamps_60s := 4
}

def spain : Country := {
  name := "Spain"
  continent := "Europe"
  price := 5/100
  stamps_50s := 3
  stamps_60s := 9
}

def south_american_countries : List Country := [brazil, peru]

def total_cost (countries : List Country) : Rat :=
  countries.foldl (fun acc country => 
    acc + (country.price * (country.stamps_50s + country.stamps_60s : Rat))) 0

theorem south_american_stamps_cost :
  total_cost south_american_countries = 106/100 := by
  sorry

#eval total_cost south_american_countries

end NUMINAMATH_CALUDE_south_american_stamps_cost_l3173_317360


namespace NUMINAMATH_CALUDE_lune_area_l3173_317377

/-- The area of a lune formed by two overlapping semicircles -/
theorem lune_area (r₁ r₂ : ℝ) (h₁ : r₁ = 3) (h₂ : r₂ = 4) : 
  (π * r₂^2 / 2) - (π * r₁^2 / 2) = 3.5 * π := by sorry

end NUMINAMATH_CALUDE_lune_area_l3173_317377


namespace NUMINAMATH_CALUDE_difference_solution_equation_problems_l3173_317316

/-- Definition of a difference solution equation -/
def is_difference_solution_equation (a b : ℝ) : Prop :=
  ∃ x : ℝ, a * x = b ∧ x = b - a

theorem difference_solution_equation_problems :
  -- Part 1
  is_difference_solution_equation 2 4 ∧
  -- Part 2
  (∀ a b : ℝ, is_difference_solution_equation 4 (a * b + a) →
    3 * (a * b + a) = 16) ∧
  -- Part 3
  (∀ m n : ℝ, is_difference_solution_equation 4 (m * n + m) ∧
    is_difference_solution_equation (-2) (m * n + n) →
    3 * (m * n + m) - 9 * (m * n + n)^2 = 0) :=
by
  sorry

end NUMINAMATH_CALUDE_difference_solution_equation_problems_l3173_317316


namespace NUMINAMATH_CALUDE_lucky_lucy_theorem_l3173_317308

/-- The expression with parentheses -/
def expr_with_parentheses (a b c d e : ℤ) : ℤ := a + (b - (c + (d - e)))

/-- The expression without parentheses -/
def expr_without_parentheses (a b c d e : ℤ) : ℤ := a + b - c + d - e

/-- The theorem stating that the expressions are equal when e = 8 -/
theorem lucky_lucy_theorem (a b c d : ℤ) (ha : a = 2) (hb : b = 4) (hc : c = 6) (hd : d = 8) :
  ∃ e : ℤ, expr_with_parentheses a b c d e = expr_without_parentheses a b c d e ∧ e = 8 := by
  sorry

end NUMINAMATH_CALUDE_lucky_lucy_theorem_l3173_317308


namespace NUMINAMATH_CALUDE_max_log_sum_l3173_317382

theorem max_log_sum (x y : ℝ) (h_pos_x : x > 0) (h_pos_y : y > 0) (h_eq : 4 * x + y = 40) :
  (Real.log x + Real.log y) ≤ 2 * Real.log 10 :=
sorry

end NUMINAMATH_CALUDE_max_log_sum_l3173_317382


namespace NUMINAMATH_CALUDE_cookie_cutter_sides_l3173_317300

/-- The number of sides on a triangle -/
def triangle_sides : ℕ := 3

/-- The number of sides on a square -/
def square_sides : ℕ := 4

/-- The number of sides on a hexagon -/
def hexagon_sides : ℕ := 6

/-- The number of triangle-shaped cookie cutters -/
def num_triangles : ℕ := 6

/-- The number of square-shaped cookie cutters -/
def num_squares : ℕ := 4

/-- The number of hexagon-shaped cookie cutters -/
def num_hexagons : ℕ := 2

/-- The total number of sides on all cookie cutters -/
def total_sides : ℕ := num_triangles * triangle_sides + num_squares * square_sides + num_hexagons * hexagon_sides

theorem cookie_cutter_sides : total_sides = 46 := by
  sorry

end NUMINAMATH_CALUDE_cookie_cutter_sides_l3173_317300


namespace NUMINAMATH_CALUDE_complex_trajectory_l3173_317339

/-- The trajectory of a complex number with given modulus -/
theorem complex_trajectory (x y : ℝ) (h : Complex.abs (x - 2 + y * Complex.I) = 2 * Real.sqrt 2) :
  (x - 2)^2 + y^2 = 8 := by
  sorry

end NUMINAMATH_CALUDE_complex_trajectory_l3173_317339


namespace NUMINAMATH_CALUDE_cheesecake_factory_savings_l3173_317347

/-- Calculates the combined savings of three employees over a period of time. -/
def combinedSavings (hourlyWage : ℚ) (hoursPerDay : ℚ) (daysPerWeek : ℚ) (weeks : ℚ)
  (savingsRate1 savingsRate2 savingsRate3 : ℚ) : ℚ :=
  let monthlyEarnings := hourlyWage * hoursPerDay * daysPerWeek * weeks
  let savings1 := monthlyEarnings * savingsRate1
  let savings2 := monthlyEarnings * savingsRate2
  let savings3 := monthlyEarnings * savingsRate3
  savings1 + savings2 + savings3

/-- The combined savings of three employees at a Cheesecake factory after four weeks. -/
theorem cheesecake_factory_savings :
  combinedSavings 10 10 5 4 (2/5) (3/5) (1/2) = 3000 := by
  sorry

end NUMINAMATH_CALUDE_cheesecake_factory_savings_l3173_317347


namespace NUMINAMATH_CALUDE_max_attached_squares_l3173_317373

/-- Represents a square in 2D space -/
structure Square :=
  (side_length : ℝ)
  (center : ℝ × ℝ)

/-- Checks if two squares are touching but not overlapping -/
def are_touching (s1 s2 : Square) : Prop :=
  sorry

/-- Checks if a square is touching the perimeter of another square -/
def is_touching_perimeter (s1 s2 : Square) : Prop :=
  sorry

/-- The configuration of squares attached to a given square -/
structure SquareConfiguration :=
  (given_square : Square)
  (attached_squares : List Square)

/-- Checks if a configuration is valid according to the problem conditions -/
def is_valid_configuration (config : SquareConfiguration) : Prop :=
  ∀ s ∈ config.attached_squares,
    is_touching_perimeter s config.given_square ∧
    ∀ t ∈ config.attached_squares, s ≠ t → ¬(are_touching s t)

/-- The main theorem: maximum number of attached squares is 8 -/
theorem max_attached_squares (config : SquareConfiguration) :
  is_valid_configuration config →
  config.attached_squares.length ≤ 8 :=
sorry

end NUMINAMATH_CALUDE_max_attached_squares_l3173_317373


namespace NUMINAMATH_CALUDE_sum_of_roots_l3173_317378

theorem sum_of_roots (α β : ℝ) 
  (h1 : α^3 - 3*α^2 + 5*α - 17 = 0)
  (h2 : β^3 - 3*β^2 + 5*β + 11 = 0) : 
  α + β = 2 := by sorry

end NUMINAMATH_CALUDE_sum_of_roots_l3173_317378


namespace NUMINAMATH_CALUDE_basement_water_pump_time_l3173_317390

/-- Calculates the time required to pump water out of a flooded basement. -/
theorem basement_water_pump_time
  (basement_length : ℝ)
  (basement_width : ℝ)
  (water_depth_inches : ℝ)
  (num_pumps : ℕ)
  (pump_rate : ℝ)
  (cubic_foot_to_gallon : ℝ)
  (h1 : basement_length = 30)
  (h2 : basement_width = 40)
  (h3 : water_depth_inches = 24)
  (h4 : num_pumps = 4)
  (h5 : pump_rate = 10)
  (h6 : cubic_foot_to_gallon = 7.5) :
  (basement_length * basement_width * (water_depth_inches / 12) * cubic_foot_to_gallon) /
  (num_pumps * pump_rate) = 450 := by
  sorry

#check basement_water_pump_time

end NUMINAMATH_CALUDE_basement_water_pump_time_l3173_317390


namespace NUMINAMATH_CALUDE_no_increase_employees_l3173_317336

theorem no_increase_employees (total : ℕ) (salary_percent : ℚ) (travel_percent : ℚ) :
  total = 480 →
  salary_percent = 10 / 100 →
  travel_percent = 20 / 100 →
  total - (total * salary_percent).floor - (total * travel_percent).floor = 336 :=
by sorry

end NUMINAMATH_CALUDE_no_increase_employees_l3173_317336


namespace NUMINAMATH_CALUDE_iAmALiar_false_for_knights_and_knaves_iAmALiar_identifies_spy_l3173_317328

-- Define the types of characters
inductive Character : Type
| Knight : Character
| Knave : Character
| Spy : Character

-- Define the property of telling the truth
def tellsTruth (c : Character) : Prop :=
  match c with
  | Character.Knight => true
  | Character.Knave => false
  | Character.Spy => false

-- Define the statement "I am a liar"
def iAmALiar (c : Character) : Prop :=
  ¬(tellsTruth c)

-- Theorem: The statement "I am a liar" is false for both knights and knaves
theorem iAmALiar_false_for_knights_and_knaves :
  ∀ c : Character, c ≠ Character.Spy → ¬(iAmALiar c = tellsTruth c) :=
by sorry

-- Theorem: The statement "I am a liar" immediately identifies the speaker as a spy
theorem iAmALiar_identifies_spy :
  ∀ c : Character, iAmALiar c = tellsTruth c → c = Character.Spy :=
by sorry

end NUMINAMATH_CALUDE_iAmALiar_false_for_knights_and_knaves_iAmALiar_identifies_spy_l3173_317328


namespace NUMINAMATH_CALUDE_arithmetic_sequence_terms_l3173_317326

/-- An arithmetic sequence with first term 11, common difference 4, and last term 107 has 25 terms -/
theorem arithmetic_sequence_terms (a₁ : ℕ) (d : ℕ) (aₙ : ℕ) (n : ℕ) :
  a₁ = 11 → d = 4 → aₙ = 107 → aₙ = a₁ + (n - 1) * d → n = 25 := by
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_terms_l3173_317326


namespace NUMINAMATH_CALUDE_orange_cost_calculation_l3173_317379

theorem orange_cost_calculation (family_size : ℕ) (planned_spending : ℚ) (savings_percentage : ℚ) (oranges_received : ℕ) : 
  family_size = 4 → 
  planned_spending = 15 → 
  savings_percentage = 40 / 100 → 
  oranges_received = family_size →
  (planned_spending * savings_percentage) / oranges_received = 3/2 := by
sorry

end NUMINAMATH_CALUDE_orange_cost_calculation_l3173_317379


namespace NUMINAMATH_CALUDE_c_months_is_six_l3173_317352

/-- Represents the rental scenario for a pasture -/
structure PastureRental where
  total_rent : ℕ
  a_horses : ℕ
  a_months : ℕ
  b_horses : ℕ
  b_months : ℕ
  c_horses : ℕ
  b_payment : ℕ

/-- Calculates the number of months c put in the horses -/
def calculate_c_months (rental : PastureRental) : ℕ :=
  sorry

/-- Theorem stating that c put in the horses for 6 months -/
theorem c_months_is_six (rental : PastureRental)
  (h1 : rental.total_rent = 870)
  (h2 : rental.a_horses = 12)
  (h3 : rental.a_months = 8)
  (h4 : rental.b_horses = 16)
  (h5 : rental.b_months = 9)
  (h6 : rental.c_horses = 18)
  (h7 : rental.b_payment = 360) :
  calculate_c_months rental = 6 :=
sorry

end NUMINAMATH_CALUDE_c_months_is_six_l3173_317352


namespace NUMINAMATH_CALUDE_parabola_intersection_l3173_317367

-- Define the two parabolas
def f (x : ℝ) : ℝ := 3 * x^2 - 8 * x + 2
def g (x : ℝ) : ℝ := 6 * x^2 + 4 * x + 2

-- Define the intersection points
def intersection_points : Set (ℝ × ℝ) := {(-4, 82), (0, 2)}

-- Theorem statement
theorem parabola_intersection :
  ∀ x y : ℝ, f x = g x ∧ y = f x ↔ (x, y) ∈ intersection_points :=
by sorry

end NUMINAMATH_CALUDE_parabola_intersection_l3173_317367


namespace NUMINAMATH_CALUDE_sin_30_degrees_l3173_317346

theorem sin_30_degrees : Real.sin (π / 6) = 1 / 2 := by sorry

end NUMINAMATH_CALUDE_sin_30_degrees_l3173_317346


namespace NUMINAMATH_CALUDE_total_cost_of_toys_l3173_317345

-- Define the costs of the toys
def marbles_cost : ℚ := 9.05
def football_cost : ℚ := 4.95
def baseball_cost : ℚ := 6.52

-- Theorem stating the total cost
theorem total_cost_of_toys :
  marbles_cost + football_cost + baseball_cost = 20.52 := by
  sorry

end NUMINAMATH_CALUDE_total_cost_of_toys_l3173_317345


namespace NUMINAMATH_CALUDE_watercolor_painting_distribution_l3173_317366

theorem watercolor_painting_distribution (total_paintings : ℕ) (paintings_per_room : ℕ) (num_rooms : ℕ) : 
  total_paintings = 32 → paintings_per_room = 8 → num_rooms * paintings_per_room = total_paintings → num_rooms = 4 := by
  sorry

end NUMINAMATH_CALUDE_watercolor_painting_distribution_l3173_317366


namespace NUMINAMATH_CALUDE_otimes_composition_l3173_317354

-- Define the new operation
def otimes (x y : ℝ) : ℝ := x^2 + y^2

-- State the theorem
theorem otimes_composition (x : ℝ) : otimes x (otimes x x) = x^2 + 4*x^4 := by
  sorry

end NUMINAMATH_CALUDE_otimes_composition_l3173_317354


namespace NUMINAMATH_CALUDE_total_marbles_is_240_l3173_317344

/-- The number of marbles in a dozen -/
def dozen : ℕ := 12

/-- The number of red marbles Jessica has -/
def jessica_marbles : ℕ := 3 * dozen

/-- The number of red marbles Sandy has -/
def sandy_marbles : ℕ := 4 * jessica_marbles

/-- The number of red marbles Alex has -/
def alex_marbles : ℕ := jessica_marbles + 2 * dozen

/-- The total number of red marbles Jessica, Sandy, and Alex have -/
def total_marbles : ℕ := jessica_marbles + sandy_marbles + alex_marbles

theorem total_marbles_is_240 : total_marbles = 240 := by
  sorry

end NUMINAMATH_CALUDE_total_marbles_is_240_l3173_317344


namespace NUMINAMATH_CALUDE_robin_gum_count_l3173_317359

/-- The number of gum packages Robin has -/
def num_packages : ℕ := 9

/-- The number of gum pieces in each package -/
def pieces_per_package : ℕ := 15

/-- The total number of gum pieces Robin has -/
def total_pieces : ℕ := num_packages * pieces_per_package

theorem robin_gum_count : total_pieces = 135 := by
  sorry

end NUMINAMATH_CALUDE_robin_gum_count_l3173_317359


namespace NUMINAMATH_CALUDE_lasagna_mince_amount_l3173_317365

/-- Proves that the amount of ground mince used for each lasagna is 2 pounds -/
theorem lasagna_mince_amount 
  (total_dishes : ℕ) 
  (cottage_pie_mince : ℕ) 
  (total_mince : ℕ) 
  (cottage_pies : ℕ) 
  (h1 : total_dishes = 100)
  (h2 : cottage_pie_mince = 3)
  (h3 : total_mince = 500)
  (h4 : cottage_pies = 100) :
  (total_mince - cottage_pies * cottage_pie_mince) / (total_dishes - cottage_pies) = 2 := by
  sorry

#check lasagna_mince_amount

end NUMINAMATH_CALUDE_lasagna_mince_amount_l3173_317365


namespace NUMINAMATH_CALUDE_absolute_value_expression_l3173_317335

theorem absolute_value_expression (x : ℤ) (h : x = -2023) :
  |abs (abs x - x) - abs x| - x = 4046 := by
  sorry

end NUMINAMATH_CALUDE_absolute_value_expression_l3173_317335


namespace NUMINAMATH_CALUDE_quadratic_equation_roots_l3173_317325

theorem quadratic_equation_roots (k : ℝ) : 
  (∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ k * x₁^2 - 2 * x₁ + 3 = 0 ∧ k * x₂^2 - 2 * x₂ + 3 = 0) ↔ 
  (k ≤ 1/3 ∧ k ≠ 0) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_equation_roots_l3173_317325


namespace NUMINAMATH_CALUDE_polynomial_determination_l3173_317330

theorem polynomial_determination (p : ℝ → ℝ) : 
  (∃ a b c : ℝ, ∀ x, p x = a * x^2 + b * x + c) →  -- p is quadratic
  p 3 = 0 →                                        -- p(3) = 0
  p (-1) = 0 →                                     -- p(-1) = 0
  p 2 = 10 →                                       -- p(2) = 10
  ∀ x, p x = -10/3 * x^2 + 20/3 * x + 10 :=        -- conclusion
by sorry

end NUMINAMATH_CALUDE_polynomial_determination_l3173_317330


namespace NUMINAMATH_CALUDE_probability_six_distinct_numbers_l3173_317372

theorem probability_six_distinct_numbers (n : ℕ) (h : n = 6) :
  (Nat.factorial n : ℚ) / (n ^ n : ℚ) = 5 / 324 := by
  sorry

end NUMINAMATH_CALUDE_probability_six_distinct_numbers_l3173_317372


namespace NUMINAMATH_CALUDE_hyperbola_equation_l3173_317368

/-- Given a hyperbola and a parabola with specific properties, prove the equation of the hyperbola -/
theorem hyperbola_equation (m n : ℝ) (h : m < 0 ∧ n > 0) :
  (∀ x y : ℝ, x^2 / m + y^2 / n = 1) →  -- Hyperbola equation
  (2 * Real.sqrt 3 / 3 : ℝ) = 2 / Real.sqrt n →  -- Eccentricity condition
  (∃ c : ℝ, c = 2 ∧ ∀ x y : ℝ, x^2 = 8*y → y = c/2) →  -- Shared focus with parabola
  (∀ x y : ℝ, y^2 / 3 - x^2 = 1) :=
by sorry

end NUMINAMATH_CALUDE_hyperbola_equation_l3173_317368


namespace NUMINAMATH_CALUDE_proportional_sum_equation_l3173_317399

theorem proportional_sum_equation (x y z a : ℝ) : 
  (∃ (k : ℝ), x = 2*k ∧ y = 3*k ∧ z = 5*k) →  -- x, y, z are proportional to 2, 3, 5
  x + y + z = 100 →                           -- sum is 100
  y = a*x - 10 →                              -- equation for y
  a = 2 :=                                    -- conclusion: a = 2
by
  sorry

end NUMINAMATH_CALUDE_proportional_sum_equation_l3173_317399


namespace NUMINAMATH_CALUDE_prob_second_day_A_l3173_317319

-- Define the probabilities
def prob_A_given_A : ℝ := 0.7
def prob_A_given_B : ℝ := 0.5
def prob_first_day_A : ℝ := 0.5
def prob_first_day_B : ℝ := 0.5

-- State the theorem
theorem prob_second_day_A :
  prob_first_day_A * prob_A_given_A + prob_first_day_B * prob_A_given_B = 0.6 := by
  sorry

end NUMINAMATH_CALUDE_prob_second_day_A_l3173_317319


namespace NUMINAMATH_CALUDE_triangle_area_proof_l3173_317392

theorem triangle_area_proof (A B C : Real) (a b c : Real) (f : Real → Real) :
  -- Triangle conditions
  0 < A ∧ 0 < B ∧ 0 < C ∧ A + B + C = π →
  -- Side lengths are positive
  0 < a ∧ 0 < b ∧ 0 < c →
  -- Given condition
  b^2 + c^2 - a^2 = b*c →
  -- a = 2
  a = 2 →
  -- Definition of function f
  (∀ x, f x = Real.sqrt 3 * Real.sin (x/2) * Real.cos (x/2) + Real.cos (x/2)^2) →
  -- f reaches maximum at B
  (∀ x, f x ≤ f B) →
  -- Conclusion: area of triangle is √3
  (1/2) * a^2 * Real.sin A = Real.sqrt 3 :=
by sorry

end NUMINAMATH_CALUDE_triangle_area_proof_l3173_317392


namespace NUMINAMATH_CALUDE_P_less_than_Q_l3173_317364

theorem P_less_than_Q (a : ℝ) (ha : a ≥ 0) : 
  Real.sqrt a + Real.sqrt (a + 7) < Real.sqrt (a + 3) + Real.sqrt (a + 4) :=
by sorry

end NUMINAMATH_CALUDE_P_less_than_Q_l3173_317364


namespace NUMINAMATH_CALUDE_product_divisible_by_3_probability_l3173_317383

/-- A standard die has 6 sides -/
def standard_die : ℕ := 6

/-- The number of rolls -/
def num_rolls : ℕ := 8

/-- The probability of rolling a number divisible by 3 on a standard die -/
def prob_divisible_by_3 : ℚ := 1 / 3

/-- The probability of rolling a number not divisible by 3 on a standard die -/
def prob_not_divisible_by_3 : ℚ := 2 / 3

/-- The probability that the product of all rolls is divisible by 3 -/
def prob_product_divisible_by_3 : ℚ := 6305 / 6561

theorem product_divisible_by_3_probability :
  prob_product_divisible_by_3 = 1 - (prob_not_divisible_by_3 ^ num_rolls) :=
sorry

end NUMINAMATH_CALUDE_product_divisible_by_3_probability_l3173_317383


namespace NUMINAMATH_CALUDE_complex_magnitude_problem_l3173_317348

theorem complex_magnitude_problem (x y : ℝ) (h : (2 + Complex.I) * y = x + y * Complex.I) (hy : y ≠ 0) :
  Complex.abs ((x / y) + Complex.I) = Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_complex_magnitude_problem_l3173_317348


namespace NUMINAMATH_CALUDE_sum_of_squares_l3173_317384

theorem sum_of_squares (a b c : ℝ) : 
  (a * b + b * c + a * c = 50) → 
  (a + b + c = 16) → 
  (a^2 + b^2 + c^2 = 156) := by
sorry

end NUMINAMATH_CALUDE_sum_of_squares_l3173_317384


namespace NUMINAMATH_CALUDE_intersection_A_B_l3173_317380

def U : Set ℝ := Set.univ

def A : Set ℝ := {x | x * (x - 2) < 0}

def B : Set ℝ := {x | x - 1 > 0}

theorem intersection_A_B : A ∩ B = {x | 1 < x ∧ x < 2} := by sorry

end NUMINAMATH_CALUDE_intersection_A_B_l3173_317380


namespace NUMINAMATH_CALUDE_divisible_by_six_percentage_l3173_317389

theorem divisible_by_six_percentage (n : ℕ) (h : n = 150) : 
  (Finset.filter (fun x => x % 6 = 0) (Finset.range (n + 1))).card / n = 1 / 6 := by
  sorry

end NUMINAMATH_CALUDE_divisible_by_six_percentage_l3173_317389


namespace NUMINAMATH_CALUDE_invalid_altitudes_l3173_317306

/-- A triple of positive real numbers represents valid altitudes of a triangle if and only if
    the sum of the reciprocals of any two is greater than the reciprocal of the third. -/
def ValidAltitudes (h₁ h₂ h₃ : ℝ) : Prop :=
  h₁ > 0 ∧ h₂ > 0 ∧ h₃ > 0 ∧
  1/h₁ + 1/h₂ > 1/h₃ ∧
  1/h₂ + 1/h₃ > 1/h₁ ∧
  1/h₃ + 1/h₁ > 1/h₂

/-- The triple (5, 12, 13) cannot be the lengths of the three altitudes of a triangle. -/
theorem invalid_altitudes : ¬ ValidAltitudes 5 12 13 := by
  sorry

end NUMINAMATH_CALUDE_invalid_altitudes_l3173_317306


namespace NUMINAMATH_CALUDE_range_of_x_l3173_317394

theorem range_of_x (x : ℝ) (h1 : 1 / x ≤ 4) (h2 : 1 / x ≥ -2) : x ≥ 1 / 4 ∨ x ≤ -1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_range_of_x_l3173_317394


namespace NUMINAMATH_CALUDE_base7_to_base4_conversion_l3173_317343

/-- Converts a number from base 7 to base 10 -/
def base7ToBase10 (n : ℕ) : ℕ := sorry

/-- Converts a number from base 10 to base 4 -/
def base10ToBase4 (n : ℕ) : ℕ := sorry

/-- The given number in base 7 -/
def given_number : ℕ := 563

theorem base7_to_base4_conversion :
  base10ToBase4 (base7ToBase10 given_number) = 10202 := by sorry

end NUMINAMATH_CALUDE_base7_to_base4_conversion_l3173_317343


namespace NUMINAMATH_CALUDE_hexagon_covers_ground_l3173_317302

def interior_angle (n : ℕ) : ℚ :=
  (n - 2) * 180 / n

def can_cover_ground (n : ℕ) : Prop :=
  ∃ k : ℕ, k * interior_angle n = 360

theorem hexagon_covers_ground :
  can_cover_ground 6 ∧
  ¬can_cover_ground 5 ∧
  ¬can_cover_ground 8 ∧
  ¬can_cover_ground 12 :=
sorry

end NUMINAMATH_CALUDE_hexagon_covers_ground_l3173_317302


namespace NUMINAMATH_CALUDE_simplest_quadratic_radical_l3173_317355

-- Define a function to check if a number is a perfect square
def is_perfect_square (n : ℚ) : Prop :=
  ∃ m : ℚ, m * m = n

-- Define a function to check if a quadratic radical is in its simplest form
def is_simplest_quadratic_radical (n : ℚ) : Prop :=
  n > 0 ∧ ¬is_perfect_square n ∧ (∀ m : ℕ, m > 1 → ¬is_perfect_square (n / (m * m : ℚ)))

-- Theorem statement
theorem simplest_quadratic_radical :
  is_simplest_quadratic_radical 7 ∧
  ¬is_simplest_quadratic_radical 9 ∧
  ¬is_simplest_quadratic_radical 20 ∧
  ¬is_simplest_quadratic_radical (1/3) :=
by sorry

end NUMINAMATH_CALUDE_simplest_quadratic_radical_l3173_317355


namespace NUMINAMATH_CALUDE_square_root_of_nine_l3173_317362

theorem square_root_of_nine : 
  ∃ (x : ℝ), x^2 = 9 ↔ x = 3 ∨ x = -3 :=
by sorry

end NUMINAMATH_CALUDE_square_root_of_nine_l3173_317362


namespace NUMINAMATH_CALUDE_triangle_altitude_l3173_317357

theorem triangle_altitude (area : ℝ) (base : ℝ) (altitude : ℝ) : 
  area = 500 → base = 50 → area = (1/2) * base * altitude → altitude = 20 := by
  sorry

end NUMINAMATH_CALUDE_triangle_altitude_l3173_317357


namespace NUMINAMATH_CALUDE_parallelogram_area_calculation_l3173_317327

/-- The area of the parallelogram formed by vectors u and z -/
def parallelogramArea (u z : Fin 2 → ℝ) : ℝ :=
  |u 0 * z 1 - u 1 * z 0|

/-- The problem statement -/
theorem parallelogram_area_calculation :
  let u : Fin 2 → ℝ := ![3, 4]
  let z : Fin 2 → ℝ := ![8, -1]
  parallelogramArea u z = 35 := by
sorry

end NUMINAMATH_CALUDE_parallelogram_area_calculation_l3173_317327


namespace NUMINAMATH_CALUDE_first_video_length_l3173_317369

/-- Given information about Kimiko's YouTube watching --/
structure YoutubeWatching where
  total_time : ℕ
  second_video_length : ℕ
  last_video_length : ℕ

/-- The theorem stating the length of the first video --/
theorem first_video_length (info : YoutubeWatching)
  (h1 : info.total_time = 510)
  (h2 : info.second_video_length = 270)
  (h3 : info.last_video_length = 60) :
  510 - info.second_video_length - 2 * info.last_video_length = 120 := by
  sorry

#check first_video_length

end NUMINAMATH_CALUDE_first_video_length_l3173_317369


namespace NUMINAMATH_CALUDE_solve_equation_l3173_317341

theorem solve_equation (x : ℚ) :
  (2 / (x + 2) + 4 / (x + 2) + (2 * x) / (x + 2) = 5) → x = -4/3 := by
  sorry

end NUMINAMATH_CALUDE_solve_equation_l3173_317341

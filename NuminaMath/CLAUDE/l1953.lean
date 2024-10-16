import Mathlib

namespace NUMINAMATH_CALUDE_polygon_sides_l1953_195347

theorem polygon_sides (sum_interior_angles : ℝ) : sum_interior_angles = 1080 → ∃ n : ℕ, n = 8 ∧ sum_interior_angles = (n - 2) * 180 :=
by
  sorry

end NUMINAMATH_CALUDE_polygon_sides_l1953_195347


namespace NUMINAMATH_CALUDE_remaining_average_l1953_195353

theorem remaining_average (total : ℕ) (subset : ℕ) (total_avg : ℚ) (subset_avg : ℚ) :
  total = 6 →
  subset = 4 →
  total_avg = 8 →
  subset_avg = 5 →
  (total_avg * total - subset_avg * subset) / (total - subset) = 14 := by
  sorry

end NUMINAMATH_CALUDE_remaining_average_l1953_195353


namespace NUMINAMATH_CALUDE_andrews_stickers_l1953_195310

theorem andrews_stickers (total : ℕ) (daniels : ℕ) (freds_extra : ℕ) 
  (h1 : total = 750)
  (h2 : daniels = 250)
  (h3 : freds_extra = 120) :
  total - (daniels + (daniels + freds_extra)) = 130 := by
  sorry

end NUMINAMATH_CALUDE_andrews_stickers_l1953_195310


namespace NUMINAMATH_CALUDE_lcm_factor_problem_l1953_195333

theorem lcm_factor_problem (A B : ℕ+) (H : ℕ+) (X Y : ℕ+) :
  H = 23 →
  Y = 14 →
  max A B = 322 →
  (A.lcm B : ℕ+) = H * X * Y →
  X = 23 :=
by sorry

end NUMINAMATH_CALUDE_lcm_factor_problem_l1953_195333


namespace NUMINAMATH_CALUDE_cosine_identity_from_system_l1953_195389

theorem cosine_identity_from_system (A B C a b c : ℝ) 
  (eq1 : a = b * Real.cos C + c * Real.cos B)
  (eq2 : b = c * Real.cos A + a * Real.cos C)
  (eq3 : c = a * Real.cos B + b * Real.cos A)
  (h : a ≠ 0 ∨ b ≠ 0 ∨ c ≠ 0) : 
  (Real.cos A)^2 + (Real.cos B)^2 + (Real.cos C)^2 + 2 * Real.cos A * Real.cos B * Real.cos C = 1 := by
sorry

end NUMINAMATH_CALUDE_cosine_identity_from_system_l1953_195389


namespace NUMINAMATH_CALUDE_soccer_ball_holes_percentage_l1953_195311

theorem soccer_ball_holes_percentage 
  (total_balls : ℕ) 
  (successfully_inflated : ℕ) 
  (overinflation_rate : ℚ) :
  total_balls = 100 →
  successfully_inflated = 48 →
  overinflation_rate = 1/5 →
  ∃ (x : ℚ), 
    0 ≤ x ∧ 
    x ≤ 1 ∧ 
    (1 - x) * (1 - overinflation_rate) * total_balls = successfully_inflated ∧
    x = 2/5 := by
  sorry

end NUMINAMATH_CALUDE_soccer_ball_holes_percentage_l1953_195311


namespace NUMINAMATH_CALUDE_residue_calculation_l1953_195316

theorem residue_calculation (m : ℕ) (h : m = 17) : 
  (220 * 18 - 28 * 5 + 4) % m = 12 := by
  sorry

end NUMINAMATH_CALUDE_residue_calculation_l1953_195316


namespace NUMINAMATH_CALUDE_f_max_value_l1953_195376

noncomputable def f (x : ℝ) : ℝ := Real.log 2 * Real.log 5 - Real.log (2 * x) * Real.log (5 * x)

theorem f_max_value :
  ∃ (max : ℝ), (∀ (x : ℝ), x > 0 → f x ≤ max) ∧ (∃ (x : ℝ), x > 0 ∧ f x = max) ∧ max = 1/4 := by
  sorry

end NUMINAMATH_CALUDE_f_max_value_l1953_195376


namespace NUMINAMATH_CALUDE_nori_crayons_left_l1953_195305

def crayons_problem (boxes : ℕ) (crayons_per_box : ℕ) (given_to_mae : ℕ) (extra_to_lea : ℕ) : ℕ :=
  let total := boxes * crayons_per_box
  let after_mae := total - given_to_mae
  let given_to_lea := given_to_mae + extra_to_lea
  after_mae - given_to_lea

theorem nori_crayons_left :
  crayons_problem 4 8 5 7 = 15 := by
  sorry

end NUMINAMATH_CALUDE_nori_crayons_left_l1953_195305


namespace NUMINAMATH_CALUDE_continuous_function_property_P_l1953_195322

open Function Set Real

theorem continuous_function_property_P 
  (f : ℝ → ℝ) 
  (hf_cont : Continuous f) 
  (hf_dom : ∀ x, x ∈ (Set.Ioc 0 1) → f x ≠ 0) 
  (hf_eq : f 0 = f 1) :
  ∀ k : ℕ, k ≥ 2 → ∃ x₀ ∈ Set.Icc 0 (1 - 1/k), f x₀ = f (x₀ + 1/k) :=
sorry

end NUMINAMATH_CALUDE_continuous_function_property_P_l1953_195322


namespace NUMINAMATH_CALUDE_intersection_equality_implies_a_range_l1953_195369

def A : Set ℝ := {x | (1/2 : ℝ) ≤ x ∧ x ≤ 3}
def B (a : ℝ) : Set ℝ := {x | x^2 + a < 0}

theorem intersection_equality_implies_a_range (a : ℝ) :
  (Aᶜ ∩ B a = B a) → a ≥ -1/4 := by
  sorry

end NUMINAMATH_CALUDE_intersection_equality_implies_a_range_l1953_195369


namespace NUMINAMATH_CALUDE_expression_simplification_l1953_195326

theorem expression_simplification (x y : ℝ) :
  7 * y - 3 * x + 8 + 2 * y^2 - x + 12 = 2 * y^2 + 7 * y - 4 * x + 20 := by
  sorry

end NUMINAMATH_CALUDE_expression_simplification_l1953_195326


namespace NUMINAMATH_CALUDE_abby_damon_weight_l1953_195346

/-- The weights of four people satisfying certain conditions -/
structure Weights where
  a : ℝ  -- Abby's weight
  b : ℝ  -- Bart's weight
  c : ℝ  -- Cindy's weight
  d : ℝ  -- Damon's weight
  ab_sum : a + b = 300
  bc_sum : b + c = 280
  cd_sum : c + d = 290
  ac_bd_diff : a + c = b + d + 10

/-- Theorem stating that given the conditions, Abby and Damon's combined weight is 310 pounds -/
theorem abby_damon_weight (w : Weights) : w.a + w.d = 310 := by
  sorry

end NUMINAMATH_CALUDE_abby_damon_weight_l1953_195346


namespace NUMINAMATH_CALUDE_units_digit_not_zero_l1953_195395

theorem units_digit_not_zero (a b : Nat) (ha : a ∈ Finset.range 100) (hb : b ∈ Finset.range 100) :
  (5^a + 6^b) % 10 ≠ 0 := by
sorry

end NUMINAMATH_CALUDE_units_digit_not_zero_l1953_195395


namespace NUMINAMATH_CALUDE_total_pencils_l1953_195358

/-- The number of colors in a rainbow -/
def rainbow_colors : ℕ := 7

/-- The number of people who bought color boxes -/
def number_of_people : ℕ := 3

/-- The number of pencils in each color box -/
def pencils_per_box : ℕ := rainbow_colors

theorem total_pencils :
  rainbow_colors * number_of_people = 21 :=
sorry

end NUMINAMATH_CALUDE_total_pencils_l1953_195358


namespace NUMINAMATH_CALUDE_specific_meal_cost_l1953_195364

/-- Calculates the total amount spent on a meal including tip -/
def totalSpent (lunchCost drinkCost tipPercentage : ℚ) : ℚ :=
  let subtotal := lunchCost + drinkCost
  let tipAmount := (tipPercentage / 100) * subtotal
  subtotal + tipAmount

/-- Theorem: Given the specific costs and tip percentage, the total spent is $68.13 -/
theorem specific_meal_cost :
  totalSpent 50.20 4.30 25 = 68.13 := by sorry

end NUMINAMATH_CALUDE_specific_meal_cost_l1953_195364


namespace NUMINAMATH_CALUDE_no_max_value_cubic_l1953_195394

/-- The function f(x) = 3x^2 + 6x^3 + 27x + 100 has no maximum value over the real numbers -/
theorem no_max_value_cubic (x : ℝ) : 
  ¬∃ (M : ℝ), ∀ (x : ℝ), 3*x^2 + 6*x^3 + 27*x + 100 ≤ M :=
by sorry

end NUMINAMATH_CALUDE_no_max_value_cubic_l1953_195394


namespace NUMINAMATH_CALUDE_kid_ticket_cost_prove_kid_ticket_cost_l1953_195361

theorem kid_ticket_cost (adult_price : ℝ) (total_tickets : ℕ) (total_profit : ℝ) (kid_tickets : ℕ) : ℝ :=
  let adult_tickets := total_tickets - kid_tickets
  let adult_profit := adult_tickets * adult_price
  let kid_profit := total_profit - adult_profit
  let kid_price := kid_profit / kid_tickets
  kid_price

theorem prove_kid_ticket_cost :
  kid_ticket_cost 6 175 750 75 = 2 := by
  sorry

end NUMINAMATH_CALUDE_kid_ticket_cost_prove_kid_ticket_cost_l1953_195361


namespace NUMINAMATH_CALUDE_tax_difference_proof_l1953_195381

-- Define the item price and tax rates
def item_price : ℝ := 15
def tax_rate_1 : ℝ := 0.08
def tax_rate_2 : ℝ := 0.072
def discount_rate : ℝ := 0.005

-- Define the effective tax rate after discount
def effective_tax_rate : ℝ := tax_rate_2 - discount_rate

-- Theorem statement
theorem tax_difference_proof :
  (item_price * (1 + tax_rate_1)) - (item_price * (1 + effective_tax_rate)) = 0.195 := by
  sorry

end NUMINAMATH_CALUDE_tax_difference_proof_l1953_195381


namespace NUMINAMATH_CALUDE_cuboid_inequality_l1953_195321

theorem cuboid_inequality (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) 
  (h_unit_diagonal : a^2 + b^2 + c^2 = 1) : 
  4*a + 4*b + 4*c + 4*a*b + 4*a*c + 4*b*c + 4*a*b*c < 12 := by
  sorry

end NUMINAMATH_CALUDE_cuboid_inequality_l1953_195321


namespace NUMINAMATH_CALUDE_line_circle_intersection_l1953_195357

-- Define the line l
def line_l (k : ℝ) (x y : ℝ) : Prop :=
  (2*k+1)*x + (k-1)*y - (4*k-1) = 0

-- Define the circle C
def circle_C (x y : ℝ) : Prop :=
  x^2 + y^2 - 4*x - 2*y + 1 = 0

-- Define the point P
def point_P : ℝ × ℝ := (4, 4)

-- Define the minimum |AB| line
def min_AB_line (x y : ℝ) : Prop :=
  x - y + 1 = 0

-- Define the tangent lines
def tangent_line_1 (x : ℝ) : Prop := x = 4
def tangent_line_2 (x y : ℝ) : Prop := y = (5/12)*x + (28/12)

theorem line_circle_intersection :
  ∀ k : ℝ,
  (∀ x y : ℝ, line_l k x y ∧ circle_C x y → 
    (∀ x' y' : ℝ, min_AB_line x' y' → 
      (x - x')^2 + (y - y')^2 ≤ (x - x')^2 + (y - y')^2)) ∧
  (∃ x y : ℝ, min_AB_line x y ∧ circle_C x y ∧
    ∃ x' y' : ℝ, min_AB_line x' y' ∧ circle_C x' y' ∧
    (x - x')^2 + (y - y')^2 = 8) ∧
  (∀ x y : ℝ, (tangent_line_1 x ∨ tangent_line_2 x y) →
    (x - 4)^2 + (y - 4)^2 = ((x - 2)^2 + (y - 1)^2 - 4)^2 / ((x - 2)^2 + (y - 1)^2)) :=
by sorry

end NUMINAMATH_CALUDE_line_circle_intersection_l1953_195357


namespace NUMINAMATH_CALUDE_gcd_lcm_product_l1953_195337

theorem gcd_lcm_product (a b : ℕ) (ha : a = 225) (hb : b = 252) :
  (Nat.gcd a b) * (Nat.lcm a b) = 56700 := by
  sorry

end NUMINAMATH_CALUDE_gcd_lcm_product_l1953_195337


namespace NUMINAMATH_CALUDE_ember_nate_ages_l1953_195344

/-- Given that Ember is initially half as old as Nate, and Nate is initially 14 years old,
    prove that when Ember's age becomes 14, Nate's age will be 21. -/
theorem ember_nate_ages (ember_initial : ℕ) (nate_initial : ℕ) (ember_final : ℕ) (nate_final : ℕ) :
  ember_initial = nate_initial / 2 →
  nate_initial = 14 →
  ember_final = 14 →
  nate_final = nate_initial + (ember_final - ember_initial) →
  nate_final = 21 := by
sorry

end NUMINAMATH_CALUDE_ember_nate_ages_l1953_195344


namespace NUMINAMATH_CALUDE_value_of_d_l1953_195354

theorem value_of_d (c d : ℚ) (h1 : c / d = 4) (h2 : c = 15 - 4 * d) : d = 15 / 8 := by
  sorry

end NUMINAMATH_CALUDE_value_of_d_l1953_195354


namespace NUMINAMATH_CALUDE_initial_oranges_count_l1953_195393

/-- The number of apples in the box -/
def num_apples : ℕ := 14

/-- The number of oranges to be removed -/
def oranges_removed : ℕ := 6

/-- The percentage of apples after removing oranges -/
def apple_percentage : ℚ := 70 / 100

theorem initial_oranges_count : 
  ∃ (initial_oranges : ℕ), 
    (num_apples : ℚ) / ((num_apples : ℚ) + (initial_oranges - oranges_removed : ℚ)) = apple_percentage ∧ 
    initial_oranges = 12 := by
  sorry

end NUMINAMATH_CALUDE_initial_oranges_count_l1953_195393


namespace NUMINAMATH_CALUDE_number_division_problem_l1953_195323

theorem number_division_problem :
  ∃ x : ℝ, (x / 9 + x + 9 = 69) ∧ x = 54 := by
  sorry

end NUMINAMATH_CALUDE_number_division_problem_l1953_195323


namespace NUMINAMATH_CALUDE_condition_analysis_l1953_195343

theorem condition_analysis :
  (∃ a b : ℝ, (1 / a > 1 / b ∧ a ≥ b) ∨ (1 / a ≤ 1 / b ∧ a < b)) ∧
  (∀ A B : Set α, A = ∅ → A ∩ B = ∅) ∧
  (∃ A B : Set α, A ∩ B = ∅ ∧ A ≠ ∅) ∧
  (∀ a b : ℝ, a^2 + b^2 ≠ 0 ↔ |a| + |b| ≠ 0) ∧
  (∃ a b : ℝ, ∃ n : ℕ, n ≥ 2 ∧ (a^n > b^n ∧ ¬(a > b ∧ b > 0))) :=
by sorry

end NUMINAMATH_CALUDE_condition_analysis_l1953_195343


namespace NUMINAMATH_CALUDE_binary_ternary_equality_l1953_195386

theorem binary_ternary_equality (a b : ℕ) (h1 : a > 0) (h2 : b > 0) 
  (h3 : b ≤ 1) (h4 : a ≤ 2) (h5 : 9 + 2*b = 9*a + 2) : 2*a + b = 3 := by
  sorry

end NUMINAMATH_CALUDE_binary_ternary_equality_l1953_195386


namespace NUMINAMATH_CALUDE_range_of_positive_integers_in_list_K_l1953_195390

def consecutive_integers (start : ℤ) (n : ℕ) : List ℤ :=
  List.range n |>.map (λ i => start + i)

def positive_integers (l : List ℤ) : List ℤ :=
  l.filter (λ x => x > 0)

def range (l : List ℤ) : ℤ :=
  l.maximum.getD 0 - l.minimum.getD 0

theorem range_of_positive_integers_in_list_K :
  let K := consecutive_integers (-5) 12
  range (positive_integers K) = 5 := by
  sorry

end NUMINAMATH_CALUDE_range_of_positive_integers_in_list_K_l1953_195390


namespace NUMINAMATH_CALUDE_julie_work_hours_l1953_195329

/-- Given Julie's work conditions, prove she needs to work 18 hours per week during school year --/
theorem julie_work_hours : 
  ∀ (summer_weeks : ℕ) (summer_hours_per_week : ℕ) (summer_earnings : ℕ) 
    (school_weeks : ℕ) (school_earnings : ℕ),
  summer_weeks = 10 →
  summer_hours_per_week = 60 →
  summer_earnings = 7500 →
  school_weeks = 40 →
  school_earnings = 9000 →
  (school_earnings * summer_weeks * summer_hours_per_week) / 
    (summer_earnings * school_weeks) = 18 := by
sorry

end NUMINAMATH_CALUDE_julie_work_hours_l1953_195329


namespace NUMINAMATH_CALUDE_stratified_sampling_theorem_l1953_195308

/-- Represents a high school with stratified sampling -/
structure HighSchool where
  first_year : ℕ
  second_year : ℕ
  third_year : ℕ
  total_sample : ℕ
  first_year_sample : ℕ
  third_year_sample : ℕ

/-- The total number of students in the high school -/
def total_students (hs : HighSchool) : ℕ :=
  hs.first_year + hs.second_year + hs.third_year

/-- The sampling ratio for the second year -/
def sampling_ratio (hs : HighSchool) : ℚ :=
  (hs.total_sample - hs.first_year_sample - hs.third_year_sample : ℚ) / hs.second_year

theorem stratified_sampling_theorem (hs : HighSchool) 
  (h1 : hs.second_year = 900)
  (h2 : hs.total_sample = 370)
  (h3 : hs.first_year_sample = 120)
  (h4 : hs.third_year_sample = 100)
  (h5 : sampling_ratio hs = 1 / 6) :
  total_students hs = 2220 := by
  sorry

#check stratified_sampling_theorem

end NUMINAMATH_CALUDE_stratified_sampling_theorem_l1953_195308


namespace NUMINAMATH_CALUDE_discount_rates_sum_l1953_195336

/-- The discount rate for Fox jeans -/
def fox_discount_rate : ℝ := sorry

/-- The discount rate for Pony jeans -/
def pony_discount_rate : ℝ := 0.1

/-- The regular price of Fox jeans -/
def fox_regular_price : ℝ := 15

/-- The regular price of Pony jeans -/
def pony_regular_price : ℝ := 18

/-- The number of Fox jeans purchased -/
def fox_quantity : ℕ := 3

/-- The number of Pony jeans purchased -/
def pony_quantity : ℕ := 2

/-- The total savings from the purchase -/
def total_savings : ℝ := 9

theorem discount_rates_sum :
  fox_discount_rate + pony_discount_rate = 0.22 :=
by
  have h1 : fox_quantity * fox_regular_price * fox_discount_rate +
            pony_quantity * pony_regular_price * pony_discount_rate = total_savings :=
    by sorry
  sorry

end NUMINAMATH_CALUDE_discount_rates_sum_l1953_195336


namespace NUMINAMATH_CALUDE_average_shirts_sold_per_day_l1953_195378

theorem average_shirts_sold_per_day 
  (morning_day1 : ℕ) 
  (afternoon_day1 : ℕ) 
  (day2 : ℕ) 
  (h1 : morning_day1 = 250) 
  (h2 : afternoon_day1 = 20) 
  (h3 : day2 = 320) : 
  (morning_day1 + afternoon_day1 + day2) / 2 = 295 := by
sorry

end NUMINAMATH_CALUDE_average_shirts_sold_per_day_l1953_195378


namespace NUMINAMATH_CALUDE_system_equations_range_l1953_195372

theorem system_equations_range (a b x y : ℝ) : 
  3 * x - y = 2 * a - 5 →
  x + 2 * y = 3 * a + 3 →
  x > 0 →
  y > 0 →
  a - b = 4 →
  b < 2 →
  a > 1 ∧ -2 < a + b ∧ a + b < 8 := by
sorry

end NUMINAMATH_CALUDE_system_equations_range_l1953_195372


namespace NUMINAMATH_CALUDE_extreme_point_condition_monotonicity_for_maximum_two_solutions_condition_l1953_195392

noncomputable section

-- Define the function f
def f (c b x : ℝ) : ℝ := c * Real.log x + 0.5 * x^2 + b * x

-- Define the derivative of f
def f' (c b x : ℝ) : ℝ := c / x + x + b

theorem extreme_point_condition (c b : ℝ) (hc : c ≠ 0) : 
  f' c b 1 = 0 ↔ b + c + 1 = 0 :=
sorry

theorem monotonicity_for_maximum (c b : ℝ) (hc : c ≠ 0) (hmax : c > 1) :
  (∀ x, 0 < x → x < 1 → (f' c b x > 0)) ∧ 
  (∀ x, 1 < x → x < c → (f' c b x < 0)) ∧
  (∀ x, x > c → (f' c b x > 0)) :=
sorry

theorem two_solutions_condition (c : ℝ) :
  (∃ x₁ x₂, x₁ ≠ x₂ ∧ f c (-1-c) x₁ = 0 ∧ f c (-1-c) x₂ = 0 ∧ 
   (∀ x, f c (-1-c) x = 0 → x = x₁ ∨ x = x₂)) ↔ 
  (-0.5 < c ∧ c < 0) :=
sorry

end NUMINAMATH_CALUDE_extreme_point_condition_monotonicity_for_maximum_two_solutions_condition_l1953_195392


namespace NUMINAMATH_CALUDE_smallest_integer_with_remainder_one_l1953_195341

theorem smallest_integer_with_remainder_one : ∃ k : ℕ,
  k > 1 ∧
  k % 13 = 1 ∧
  k % 7 = 1 ∧
  k % 3 = 1 ∧
  k % 5 = 1 ∧
  (∀ m : ℕ, m > 1 ∧ m % 13 = 1 ∧ m % 7 = 1 ∧ m % 3 = 1 ∧ m % 5 = 1 → k ≤ m) ∧
  k = 1366 :=
by sorry

end NUMINAMATH_CALUDE_smallest_integer_with_remainder_one_l1953_195341


namespace NUMINAMATH_CALUDE_cube_volume_surface_area_l1953_195307

theorem cube_volume_surface_area (x : ℝ) : 
  (∃ s : ℝ, s > 0 ∧ s^3 = 7*x ∧ 6*s^2 = x) → x = 42 := by
  sorry

end NUMINAMATH_CALUDE_cube_volume_surface_area_l1953_195307


namespace NUMINAMATH_CALUDE_largest_y_value_l1953_195350

/-- The largest possible value of y for regular polygons Q1 (x-gon) and Q2 (y-gon) -/
theorem largest_y_value (x y : ℕ) : 
  x ≥ y → 
  y ≥ 3 → 
  (x - 2) * y * 29 = (y - 2) * x * 28 → 
  y ≤ 57 :=
by sorry

end NUMINAMATH_CALUDE_largest_y_value_l1953_195350


namespace NUMINAMATH_CALUDE_arithmetic_sqrt_of_nine_l1953_195383

theorem arithmetic_sqrt_of_nine (x : ℝ) :
  (x ≥ 0 ∧ x^2 = 9) → x = 3 := by sorry

end NUMINAMATH_CALUDE_arithmetic_sqrt_of_nine_l1953_195383


namespace NUMINAMATH_CALUDE_negative_integer_solution_of_inequality_l1953_195370

theorem negative_integer_solution_of_inequality :
  ∀ x : ℤ, x < 0 →
    (((2 * x - 1 : ℚ) / 3) - ((5 * x + 1 : ℚ) / 2) ≤ 1) ↔ x = -1 := by
  sorry

end NUMINAMATH_CALUDE_negative_integer_solution_of_inequality_l1953_195370


namespace NUMINAMATH_CALUDE_lg_ratio_theorem_l1953_195367

-- Define the logarithm function (base 10)
noncomputable def lg (x : ℝ) : ℝ := Real.log x / Real.log 10

-- State the theorem
theorem lg_ratio_theorem (a b : ℝ) (h1 : lg 2 = a) (h2 : lg 3 = b) :
  (lg 12) / (lg 15) = (2 * a + b) / (1 - a + b) := by
  sorry

end NUMINAMATH_CALUDE_lg_ratio_theorem_l1953_195367


namespace NUMINAMATH_CALUDE_factor_w4_minus_81_l1953_195318

theorem factor_w4_minus_81 (w : ℝ) : w^4 - 81 = (w - 3) * (w + 3) * (w^2 + 9) := by
  sorry

end NUMINAMATH_CALUDE_factor_w4_minus_81_l1953_195318


namespace NUMINAMATH_CALUDE_max_sum_of_squares_l1953_195388

theorem max_sum_of_squares (a b c d : ℝ) : 
  a + b = 18 →
  a * b + c + d = 85 →
  a * d + b * c = 170 →
  c * d = 105 →
  a^2 + b^2 + c^2 + d^2 ≤ 308 :=
by sorry

end NUMINAMATH_CALUDE_max_sum_of_squares_l1953_195388


namespace NUMINAMATH_CALUDE_saras_high_school_basketball_games_l1953_195317

theorem saras_high_school_basketball_games 
  (defeated_games won_games total_games : ℕ) : 
  defeated_games = 4 → 
  won_games = 8 → 
  total_games = defeated_games + won_games → 
  total_games = 12 :=
by sorry

end NUMINAMATH_CALUDE_saras_high_school_basketball_games_l1953_195317


namespace NUMINAMATH_CALUDE_no_real_solutions_quadratic_l1953_195385

theorem no_real_solutions_quadratic (k : ℝ) :
  (∀ x : ℝ, x^2 - 4*x + k ≠ 0) ↔ k > 4 := by
sorry

end NUMINAMATH_CALUDE_no_real_solutions_quadratic_l1953_195385


namespace NUMINAMATH_CALUDE_monica_books_next_year_l1953_195313

/-- The number of books Monica read last year -/
def books_last_year : ℕ := 16

/-- The number of books Monica read this year -/
def books_this_year : ℕ := 2 * books_last_year

/-- The number of books Monica will read next year -/
def books_next_year : ℕ := 2 * books_this_year + 5

/-- Theorem stating the number of books Monica will read next year -/
theorem monica_books_next_year : books_next_year = 69 := by
  sorry

end NUMINAMATH_CALUDE_monica_books_next_year_l1953_195313


namespace NUMINAMATH_CALUDE_sin_equality_implies_equal_frequencies_l1953_195335

theorem sin_equality_implies_equal_frequencies
  (α β γ τ : ℝ)
  (h_pos : 0 < α ∧ 0 < β ∧ 0 < γ ∧ 0 < τ)
  (h_eq : ∀ x : ℝ, Real.sin (α * x) + Real.sin (β * x) = Real.sin (γ * x) + Real.sin (τ * x)) :
  α = γ ∨ α = τ := by
  sorry

end NUMINAMATH_CALUDE_sin_equality_implies_equal_frequencies_l1953_195335


namespace NUMINAMATH_CALUDE_inscribed_sphere_radius_eq_one_l1953_195319

/-- A right triangular pyramid with base edge length 6 and lateral edge length √21 -/
structure RightTriangularPyramid where
  base_edge_length : ℝ
  lateral_edge_length : ℝ
  base_edge_length_eq : base_edge_length = 6
  lateral_edge_length_eq : lateral_edge_length = Real.sqrt 21

/-- The radius of the inscribed sphere of a right triangular pyramid -/
def inscribed_sphere_radius (p : RightTriangularPyramid) : ℝ :=
  1 -- Definition, not proof

/-- Theorem: The radius of the inscribed sphere of a right triangular pyramid
    with base edge length 6 and lateral edge length √21 is equal to 1 -/
theorem inscribed_sphere_radius_eq_one (p : RightTriangularPyramid) :
  inscribed_sphere_radius p = 1 := by
  sorry

#check inscribed_sphere_radius_eq_one

end NUMINAMATH_CALUDE_inscribed_sphere_radius_eq_one_l1953_195319


namespace NUMINAMATH_CALUDE_age_problem_l1953_195371

theorem age_problem (a₁ a₂ a₃ a₄ a₅ : ℕ) : 
  a₁ < a₂ ∧ a₂ < a₃ ∧ a₃ < a₄ ∧ a₄ < a₅ →
  (a₁ + a₂ + a₃) / 3 = 18 →
  a₅ - a₄ = 5 →
  (a₃ + a₄ + a₅) / 3 = 26 →
  a₂ - a₁ = 7 →
  (a₁ + a₅) / 2 = 22 →
  a₁ = 13 ∧ a₂ = 20 ∧ a₃ = 21 ∧ a₄ = 26 ∧ a₅ = 31 :=
by sorry

#check age_problem

end NUMINAMATH_CALUDE_age_problem_l1953_195371


namespace NUMINAMATH_CALUDE_chessboard_symmetry_l1953_195303

-- Define a chessboard
structure Chessboard :=
  (ranks : Fin 8)
  (files : Fin 8)

-- Define a chess square
structure Square :=
  (file : Char)
  (rank : Nat)

-- Define symmetry on the chessboard
def symmetric (s1 s2 : Square) (b : Chessboard) : Prop :=
  s1.file = s2.file ∧ s1.rank + s2.rank = 9

-- Define the line of symmetry
def lineOfSymmetry (b : Chessboard) : Prop :=
  ∀ (s1 s2 : Square), symmetric s1 s2 b → (s1.rank = 4 ∧ s2.rank = 5) ∨ (s1.rank = 5 ∧ s2.rank = 4)

-- Theorem statement
theorem chessboard_symmetry (b : Chessboard) :
  lineOfSymmetry b ∧
  symmetric (Square.mk 'e' 2) (Square.mk 'e' 7) b ∧
  symmetric (Square.mk 'h' 5) (Square.mk 'h' 4) b :=
sorry

end NUMINAMATH_CALUDE_chessboard_symmetry_l1953_195303


namespace NUMINAMATH_CALUDE_no_distinct_unit_fraction_sum_l1953_195365

theorem no_distinct_unit_fraction_sum (p : ℕ) (h_prime : Nat.Prime p) (h_ge_5 : p ≥ 5) :
  ¬∃ (a b : ℕ), a ≠ b ∧ a > 0 ∧ b > 0 ∧ (p - 1 : ℚ) / p = 1 / a + 1 / b :=
by sorry

end NUMINAMATH_CALUDE_no_distinct_unit_fraction_sum_l1953_195365


namespace NUMINAMATH_CALUDE_purple_ring_weight_l1953_195351

/-- The weight of the purple ring in an experiment, given the weights of other rings and the total weight -/
theorem purple_ring_weight :
  let orange_weight : ℚ := 0.08333333333333333
  let white_weight : ℚ := 0.4166666666666667
  let total_weight : ℚ := 0.8333333333
  let purple_weight : ℚ := total_weight - orange_weight - white_weight
  purple_weight = 0.3333333333 := by
  sorry

end NUMINAMATH_CALUDE_purple_ring_weight_l1953_195351


namespace NUMINAMATH_CALUDE_min_value_reciprocal_sum_l1953_195360

theorem min_value_reciprocal_sum (x y z : ℝ) 
  (hx : x > 0) (hy : y > 0) (hz : z > 0) (hxyz : x * y * z = 1) :
  1 / (x + y) + 1 / (y + z) + 1 / (z + x) ≥ 3 / 2 := by
  sorry

end NUMINAMATH_CALUDE_min_value_reciprocal_sum_l1953_195360


namespace NUMINAMATH_CALUDE_board_cut_multiple_l1953_195304

/-- Given a board of 120 cm cut into two pieces, where the shorter piece is 35 cm
    and the longer piece is 15 cm longer than m times the shorter piece, m must equal 2. -/
theorem board_cut_multiple (m : ℝ) : 
  (35 : ℝ) + (m * 35 + 15) = 120 → m = 2 := by
  sorry

end NUMINAMATH_CALUDE_board_cut_multiple_l1953_195304


namespace NUMINAMATH_CALUDE_tan_equality_l1953_195391

theorem tan_equality : 
  3.439 * Real.tan (110 * π / 180) + Real.tan (50 * π / 180) + Real.tan (20 * π / 180) = 
  Real.tan (110 * π / 180) * Real.tan (50 * π / 180) * Real.tan (20 * π / 180) := by
  sorry

end NUMINAMATH_CALUDE_tan_equality_l1953_195391


namespace NUMINAMATH_CALUDE_greatest_power_of_two_l1953_195384

theorem greatest_power_of_two (n : ℕ) : 
  ∃ k : ℕ, 2^k ∣ (10^1503 - 4^752) ∧ ∀ m : ℕ, 2^m ∣ (10^1503 - 4^752) → m ≤ k := by
  sorry

end NUMINAMATH_CALUDE_greatest_power_of_two_l1953_195384


namespace NUMINAMATH_CALUDE_product_trailing_zeros_l1953_195334

/-- The number of trailing zeros in a natural number -/
def trailingZeros (n : ℕ) : ℕ := sorry

/-- The product of 50 and 360 -/
def product : ℕ := 50 * 360

theorem product_trailing_zeros : trailingZeros product = 3 := by sorry

end NUMINAMATH_CALUDE_product_trailing_zeros_l1953_195334


namespace NUMINAMATH_CALUDE_intersection_points_theorem_l1953_195374

def original_equation (x : ℝ) : Prop := x^2 - 3*x + 2 = 0

def pair_A (x y : ℝ) : Prop := (y = x^2 - x) ∧ (y = 2*x - 2)
def pair_B (x y : ℝ) : Prop := (y = x^2 - 3*x + 2) ∧ (y = 0)
def pair_C (x y : ℝ) : Prop := (y = x - 1) ∧ (y = x + 1)
def pair_D (x y : ℝ) : Prop := (y = x^2 - 3*x + 3) ∧ (y = 1)

theorem intersection_points_theorem :
  (∃ x y : ℝ, pair_A x y ∧ original_equation x) ∧
  (∃ x y : ℝ, pair_B x y ∧ original_equation x) ∧
  (∃ x y : ℝ, pair_D x y ∧ original_equation x) ∧
  ¬(∃ x y : ℝ, pair_C x y ∧ original_equation x) :=
sorry

end NUMINAMATH_CALUDE_intersection_points_theorem_l1953_195374


namespace NUMINAMATH_CALUDE_perpendicular_vectors_imply_m_value_l1953_195366

/-- Given vectors in R² -/
def a : Fin 2 → ℝ := ![3, 4]
def b (m : ℝ) : Fin 2 → ℝ := ![-1, 2*m]
def c (m : ℝ) : Fin 2 → ℝ := ![m, -4]

/-- The sum of vectors a and b -/
def a_plus_b (m : ℝ) : Fin 2 → ℝ := ![a 0 + b m 0, a 1 + b m 1]

/-- Dot product of two 2D vectors -/
def dot_product (v w : Fin 2 → ℝ) : ℝ := (v 0 * w 0) + (v 1 * w 1)

/-- Theorem stating that if c is perpendicular to (a + b), then m = -8/3 -/
theorem perpendicular_vectors_imply_m_value :
  ∀ m : ℝ, dot_product (c m) (a_plus_b m) = 0 → m = -8/3 := by
sorry

end NUMINAMATH_CALUDE_perpendicular_vectors_imply_m_value_l1953_195366


namespace NUMINAMATH_CALUDE_dogs_not_liking_any_food_l1953_195368

theorem dogs_not_liking_any_food (total : ℕ) (watermelon salmon chicken : Finset ℕ) :
  total = 100 →
  watermelon.card = 20 →
  salmon.card = 70 →
  (watermelon ∩ salmon).card = 10 →
  chicken.card = 15 →
  (watermelon ∩ chicken).card = 5 →
  (salmon ∩ chicken).card = 8 →
  (watermelon ∩ salmon ∩ chicken).card = 3 →
  (total : ℤ) - (watermelon ∪ salmon ∪ chicken).card = 15 := by
  sorry

end NUMINAMATH_CALUDE_dogs_not_liking_any_food_l1953_195368


namespace NUMINAMATH_CALUDE_exam_pass_count_l1953_195332

theorem exam_pass_count (total : ℕ) (avg_all : ℚ) (avg_pass : ℚ) (avg_fail : ℚ) :
  total = 120 →
  avg_all = 35 →
  avg_pass = 39 →
  avg_fail = 15 →
  ∃ pass_count : ℕ,
    pass_count = 100 ∧
    pass_count ≤ total ∧
    (pass_count : ℚ) * avg_pass + (total - pass_count : ℚ) * avg_fail = (total : ℚ) * avg_all :=
by sorry

end NUMINAMATH_CALUDE_exam_pass_count_l1953_195332


namespace NUMINAMATH_CALUDE_ball_motion_time_formula_l1953_195331

/-- Represents the motion of a ball thrown upward -/
structure BallMotion where
  h : ℝ     -- Initial height
  V₀ : ℝ    -- Initial velocity
  g : ℝ     -- Gravitational acceleration
  t : ℝ     -- Time
  V : ℝ     -- Final velocity
  S : ℝ     -- Displacement

/-- The theorem stating the relationship between time, displacement, velocities, and height -/
theorem ball_motion_time_formula (b : BallMotion) 
  (hS : b.S = b.h + (1/2) * b.g * b.t^2 + b.V₀ * b.t)
  (hV : b.V = b.g * b.t + b.V₀) :
  b.t = (2 * (b.S - b.h)) / (b.V + b.V₀) :=
by sorry

end NUMINAMATH_CALUDE_ball_motion_time_formula_l1953_195331


namespace NUMINAMATH_CALUDE_add_2405_minutes_to_midnight_l1953_195345

-- Define a custom time type
structure Time where
  hours : Nat
  minutes : Nat

-- Define a function to add minutes to a time
def addMinutes (t : Time) (m : Nat) : Time :=
  let totalMinutes := t.hours * 60 + t.minutes + m
  { hours := (totalMinutes / 60) % 24, minutes := totalMinutes % 60 }

-- Theorem statement
theorem add_2405_minutes_to_midnight :
  addMinutes { hours := 0, minutes := 0 } 2405 = { hours := 16, minutes := 5 } := by
  sorry

end NUMINAMATH_CALUDE_add_2405_minutes_to_midnight_l1953_195345


namespace NUMINAMATH_CALUDE_building_height_is_100_l1953_195387

/-- The height of a building with an elevator --/
def building_height (acceleration : ℝ) (constant_velocity : ℝ) (constant_time : ℝ) (acc_time : ℝ) : ℝ :=
  -- Distance during acceleration and deceleration
  2 * (0.5 * acceleration * acc_time^2) +
  -- Distance during constant velocity
  constant_velocity * constant_time

/-- Theorem stating the height of the building --/
theorem building_height_is_100 :
  building_height 2.5 5 18 2 = 100 := by
  sorry

#eval building_height 2.5 5 18 2

end NUMINAMATH_CALUDE_building_height_is_100_l1953_195387


namespace NUMINAMATH_CALUDE_smallest_power_comparison_l1953_195349

theorem smallest_power_comparison : 127^8 < 63^10 ∧ 63^10 < 33^12 := by
  sorry

end NUMINAMATH_CALUDE_smallest_power_comparison_l1953_195349


namespace NUMINAMATH_CALUDE_inverse_variation_proof_l1953_195379

/-- Given that y^4 varies inversely with z^2 and y = 3 when z = 1, prove that y = √3 when z = 3 -/
theorem inverse_variation_proof (y z : ℝ) (h1 : ∃ k : ℝ, ∀ y z, y^4 * z^2 = k) 
  (h2 : ∃ y₀ z₀, y₀ = 3 ∧ z₀ = 1 ∧ y₀^4 * z₀^2 = (3 : ℝ)^4 * 1^2) :
  ∃ y₁, y₁^4 * 3^2 = 3^4 * 1^2 ∧ y₁ = Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_inverse_variation_proof_l1953_195379


namespace NUMINAMATH_CALUDE_flour_to_add_l1953_195339

/-- Given a cake recipe and partially added ingredients, calculate the remaining amount to be added -/
theorem flour_to_add (total_required : ℕ) (already_added : ℕ) (h : total_required ≥ already_added) :
  total_required - already_added = 8 - 4 :=
by
  sorry

#check flour_to_add

end NUMINAMATH_CALUDE_flour_to_add_l1953_195339


namespace NUMINAMATH_CALUDE_diamond_equation_solution_l1953_195397

def diamond (X Y : ℚ) : ℚ := 4 * X + 3 * Y + 7

theorem diamond_equation_solution :
  ∃! X : ℚ, diamond X 5 = 75 ∧ X = 53 / 4 :=
by
  sorry

end NUMINAMATH_CALUDE_diamond_equation_solution_l1953_195397


namespace NUMINAMATH_CALUDE_solution_count_correct_l1953_195342

/-- The number of integers n satisfying the equation 1 + ⌊(100n)/103⌋ = ⌈(97n)/100⌉ -/
def solution_count : ℕ := 10300

/-- Function g(n) defined as ⌈(97n)/100⌉ - ⌊(100n)/103⌋ -/
def g (n : ℤ) : ℤ := ⌈(97 * n : ℚ) / 100⌉ - ⌊(100 * n : ℚ) / 103⌋

/-- The main theorem stating that the number of solutions is equal to solution_count -/
theorem solution_count_correct :
  (∑' n : ℤ, if 1 + ⌊(100 * n : ℚ) / 103⌋ = ⌈(97 * n : ℚ) / 100⌉ then 1 else 0) = solution_count :=
sorry

/-- Lemma showing the periodic behavior of g(n) -/
lemma g_periodic (n : ℤ) : g (n + 10300) = g n + 3 :=
sorry

/-- Lemma stating that for each residue class modulo 10300, there exists a unique solution -/
lemma unique_solution_per_residue_class (r : ℤ) :
  ∃! n : ℤ, g n = 1 ∧ n ≡ r [ZMOD 10300] :=
sorry

end NUMINAMATH_CALUDE_solution_count_correct_l1953_195342


namespace NUMINAMATH_CALUDE_tan_double_angle_special_case_l1953_195324

theorem tan_double_angle_special_case (α : Real) 
  (h : (2 * Real.sin α + Real.cos α) / (Real.sin α - Real.cos α) = 3) : 
  Real.tan (2 * α) = -8/15 := by
  sorry

end NUMINAMATH_CALUDE_tan_double_angle_special_case_l1953_195324


namespace NUMINAMATH_CALUDE_ladder_distance_l1953_195382

theorem ladder_distance (ladder_length height : ℝ) 
  (h1 : ladder_length = 25)
  (h2 : height = 20) :
  ∃ (distance : ℝ), distance^2 + height^2 = ladder_length^2 ∧ distance = 15 :=
sorry

end NUMINAMATH_CALUDE_ladder_distance_l1953_195382


namespace NUMINAMATH_CALUDE_base_12_division_remainder_l1953_195328

def base_12_to_decimal (n : ℕ) : ℕ :=
  1 * 12^3 + 5 * 12^2 + 3 * 12 + 4

theorem base_12_division_remainder :
  (base_12_to_decimal 1534) % 9 = 2 := by
sorry

end NUMINAMATH_CALUDE_base_12_division_remainder_l1953_195328


namespace NUMINAMATH_CALUDE_largest_x_and_multiples_l1953_195399

theorem largest_x_and_multiples :
  let x := Int.floor ((23 - 7) / -3)
  x = -6 ∧
  (x^2 * 1 = 36 ∧ x^2 * 2 = 72 ∧ x^2 * 3 = 108) :=
by sorry

end NUMINAMATH_CALUDE_largest_x_and_multiples_l1953_195399


namespace NUMINAMATH_CALUDE_min_squared_distance_to_origin_l1953_195375

/-- The minimum value of x^2 + y^2 for points on the line x + y - 4 = 0 is 8 -/
theorem min_squared_distance_to_origin (x y : ℝ) : 
  x + y - 4 = 0 → (∀ a b : ℝ, a + b - 4 = 0 → x^2 + y^2 ≤ a^2 + b^2) → x^2 + y^2 = 8 := by
  sorry

end NUMINAMATH_CALUDE_min_squared_distance_to_origin_l1953_195375


namespace NUMINAMATH_CALUDE_min_sum_of_log_arithmetic_sequence_l1953_195396

theorem min_sum_of_log_arithmetic_sequence (x y : ℝ) 
  (hx : x > 1) (hy : y > 1) 
  (h_seq : (Real.log x + Real.log y) / 2 = 2) : 
  (∀ a b : ℝ, a > 1 → b > 1 → (Real.log a + Real.log b) / 2 = 2 → x + y ≤ a + b) ∧ 
  ∃ a b : ℝ, a > 1 ∧ b > 1 ∧ (Real.log a + Real.log b) / 2 = 2 ∧ a + b = 200 :=
by sorry

end NUMINAMATH_CALUDE_min_sum_of_log_arithmetic_sequence_l1953_195396


namespace NUMINAMATH_CALUDE_a_5_equals_one_l1953_195330

def geometric_sequence (a : ℕ → ℝ) (r : ℝ) :=
  ∀ n, a (n + 1) = r * a n

theorem a_5_equals_one
  (a : ℕ → ℝ)
  (h_geo : geometric_sequence a 2)
  (h_pos : ∀ n, a n > 0)
  (h_prod : a 3 * a 11 = 16) :
  a 5 = 1 := by
sorry

end NUMINAMATH_CALUDE_a_5_equals_one_l1953_195330


namespace NUMINAMATH_CALUDE_cookies_per_box_type3_is_16_l1953_195327

/-- The number of cookies in each box of the third type -/
def cookies_per_box_type3 (
  cookies_per_box_type1 : ℕ)
  (cookies_per_box_type2 : ℕ)
  (boxes_sold_type1 : ℕ)
  (boxes_sold_type2 : ℕ)
  (boxes_sold_type3 : ℕ)
  (total_cookies_sold : ℕ) : ℕ :=
  (total_cookies_sold - (cookies_per_box_type1 * boxes_sold_type1 + cookies_per_box_type2 * boxes_sold_type2)) / boxes_sold_type3

theorem cookies_per_box_type3_is_16 :
  cookies_per_box_type3 12 20 50 80 70 3320 = 16 := by
  sorry

end NUMINAMATH_CALUDE_cookies_per_box_type3_is_16_l1953_195327


namespace NUMINAMATH_CALUDE_relationship_between_variables_l1953_195306

theorem relationship_between_variables (a b c d : ℝ) 
  (h : (3 * a + 2 * b) / (2 * b + 4 * c) = (4 * c + 3 * d) / (3 * d + 3 * a)) :
  3 * a = 4 * c ∨ 3 * a + 3 * d + 2 * b + 4 * c = 0 :=
by sorry

end NUMINAMATH_CALUDE_relationship_between_variables_l1953_195306


namespace NUMINAMATH_CALUDE_recliner_sales_increase_l1953_195362

theorem recliner_sales_increase 
  (price_decrease : ℝ) 
  (gross_increase : ℝ) 
  (h1 : price_decrease = 0.20) 
  (h2 : gross_increase = 0.20000000000000014) : 
  (1 + gross_increase) / (1 - price_decrease) - 1 = 0.5 := by sorry

end NUMINAMATH_CALUDE_recliner_sales_increase_l1953_195362


namespace NUMINAMATH_CALUDE_parallel_line_intersection_not_always_parallel_l1953_195300

-- Define the types for planes and lines
variable (Plane Line : Type)

-- Define the parallel and intersection operations
variable (parallel : Line → Plane → Prop)
variable (parallel_lines : Line → Line → Prop)
variable (intersect : Plane → Plane → Line → Prop)

-- Define the given conditions
variable (α β : Plane)
variable (m n : Line)
variable (h_distinct_planes : α ≠ β)
variable (h_distinct_lines : m ≠ n)

-- State the theorem
theorem parallel_line_intersection_not_always_parallel :
  ¬(∀ (α β : Plane) (m n : Line),
    α ≠ β → m ≠ n →
    parallel m α → intersect α β n → parallel_lines m n) :=
sorry

end NUMINAMATH_CALUDE_parallel_line_intersection_not_always_parallel_l1953_195300


namespace NUMINAMATH_CALUDE_simplify_fractions_l1953_195356

theorem simplify_fractions :
  (∀ (a b c : ℝ), a ≠ 0 → b ≠ 0 → c ≠ 0 → (3 * a^2 * b) / (6 * a * b^2 * c) = a / (2 * b * c)) ∧
  (∀ (x y : ℝ), x ≠ y → (2 * (x - y)^3) / (y - x) = -2 * (x - y)^2) :=
by sorry

end NUMINAMATH_CALUDE_simplify_fractions_l1953_195356


namespace NUMINAMATH_CALUDE_quadratic_equation_roots_l1953_195315

theorem quadratic_equation_roots (m : ℝ) : 
  (∃ x : ℝ, x^2 - 4*x + m = 0 ∧ x = -2) → 
  m = -12 ∧ ∃ y : ℝ, y^2 - 4*y + m = 0 ∧ y = 6 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_equation_roots_l1953_195315


namespace NUMINAMATH_CALUDE_cricket_team_average_age_l1953_195352

theorem cricket_team_average_age :
  ∀ (team_size : ℕ) (captain_age : ℕ) (wicket_keeper_age_diff : ℕ) (A : ℝ),
    team_size = 11 →
    captain_age = 27 →
    wicket_keeper_age_diff = 3 →
    (team_size : ℝ) * A = 
      (captain_age : ℝ) + (captain_age + wicket_keeper_age_diff : ℝ) + 
      ((team_size - 2 : ℝ) * (A - 1)) →
    A = 24 := by
  sorry

end NUMINAMATH_CALUDE_cricket_team_average_age_l1953_195352


namespace NUMINAMATH_CALUDE_triangle_height_l1953_195340

theorem triangle_height (base : ℝ) (area : ℝ) (height : ℝ) : 
  base = 3 → area = 6 → area = (base * height) / 2 → height = 4 := by
  sorry

end NUMINAMATH_CALUDE_triangle_height_l1953_195340


namespace NUMINAMATH_CALUDE_modular_inverse_17_mod_23_l1953_195373

theorem modular_inverse_17_mod_23 :
  (∃ x : ℤ, (11 * x) % 23 = 1) →
  (∃ y : ℤ, (17 * y) % 23 = 1 ∧ 0 ≤ y ∧ y ≤ 22) ∧
  (∀ z : ℤ, (17 * z) % 23 = 1 → z % 23 = 19) :=
by sorry

end NUMINAMATH_CALUDE_modular_inverse_17_mod_23_l1953_195373


namespace NUMINAMATH_CALUDE_smallest_solution_quadratic_l1953_195314

theorem smallest_solution_quadratic :
  ∃ (x : ℝ), x = 2/3 ∧ 6*x^2 - 19*x + 10 = 0 ∧ ∀ (y : ℝ), 6*y^2 - 19*y + 10 = 0 → x ≤ y :=
by sorry

end NUMINAMATH_CALUDE_smallest_solution_quadratic_l1953_195314


namespace NUMINAMATH_CALUDE_a_range_when_f_decreasing_l1953_195380

/-- A piecewise function f(x) defined based on the parameter a -/
noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  if x < 1 then (a - 3) * x + 3 * a else Real.log x / Real.log a

/-- Theorem stating that if f is decreasing on ℝ, then a is in the open interval (3/4, 1) -/
theorem a_range_when_f_decreasing (a : ℝ) :
  (∀ x y : ℝ, x < y → f a x > f a y) → a ∈ Set.Ioo (3/4) 1 := by
  sorry

end NUMINAMATH_CALUDE_a_range_when_f_decreasing_l1953_195380


namespace NUMINAMATH_CALUDE_expression_simplification_l1953_195312

theorem expression_simplification :
  (∀ x y z : ℝ, x = 4 * Real.sqrt 5 ∧ y = Real.sqrt 20 ∧ z = Real.sqrt (1/2) →
    x - y + z = 2 * Real.sqrt 5 + Real.sqrt 2 / 2) ∧
  (∀ a b c : ℝ, a = Real.sqrt 6 ∧ b = 2 * Real.sqrt 3 ∧ c = Real.sqrt 24 →
    a * b - c / Real.sqrt 3 = 4 * Real.sqrt 2) := by
  sorry

end NUMINAMATH_CALUDE_expression_simplification_l1953_195312


namespace NUMINAMATH_CALUDE_sum_of_solutions_quadratic_sum_of_solutions_specific_l1953_195398

theorem sum_of_solutions_quadratic (a b c : ℝ) (h : a ≠ 0) :
  let equation := fun x => a * x^2 + b * x + c
  let sum_of_roots := -b / a
  equation 0 = 0 → sum_of_roots = (-b) / a :=
by
  sorry

-- Specific instance for the given problem
theorem sum_of_solutions_specific :
  let equation := fun x => -48 * x^2 + 96 * x + 180
  let sum_of_roots := 2
  (∀ x, equation x = 0 → x = sum_of_roots ∨ x = 0) :=
by
  sorry

end NUMINAMATH_CALUDE_sum_of_solutions_quadratic_sum_of_solutions_specific_l1953_195398


namespace NUMINAMATH_CALUDE_sequence_existence_l1953_195320

theorem sequence_existence (n : ℕ) (hn : n ≥ 3) :
  (∃ a : ℕ → ℝ, 
    (a (n + 1) = a 1) ∧ 
    (a (n + 2) = a 2) ∧ 
    (∀ i ∈ Finset.range n, a i * a (i + 1) + 1 = a (i + 2)))
  ↔ 
  (∃ k : ℕ, n = 3 * k) :=
by sorry

end NUMINAMATH_CALUDE_sequence_existence_l1953_195320


namespace NUMINAMATH_CALUDE_contrapositive_equivalence_l1953_195348

theorem contrapositive_equivalence :
  (∀ x : ℝ, (x^2 ≥ 4 → x ≤ -2 ∨ x ≥ 2)) ↔
  (∀ x : ℝ, (-2 < x ∧ x < 2 → x^2 < 4)) :=
by sorry

end NUMINAMATH_CALUDE_contrapositive_equivalence_l1953_195348


namespace NUMINAMATH_CALUDE_conic_touches_square_l1953_195325

/-- The conic equation derived from the differential equation -/
def conic (h : ℝ) (x y : ℝ) : Prop :=
  y^2 + 2*h*x*y + x^2 = 9*(1 - h^2)

/-- The square with sides touching the conic -/
def square (x y : ℝ) : Prop :=
  (x = 3 ∨ x = -3 ∨ y = 3 ∨ y = -3) ∧ (abs x ≤ 3 ∧ abs y ≤ 3)

/-- The theorem stating that the conic touches the sides of the square -/
theorem conic_touches_square (h : ℝ) (h_bounds : 0 ≤ h ∧ h ≤ 1) :
  ∃ (x y : ℝ), conic h x y ∧ square x y :=
sorry

end NUMINAMATH_CALUDE_conic_touches_square_l1953_195325


namespace NUMINAMATH_CALUDE_function_property_implies_injective_l1953_195355

/-- A function f: ℕ → ℕ is injective if for all x y: ℕ, f x = f y implies x = y -/
def Injective (f : ℕ → ℕ) : Prop :=
  ∀ x y : ℕ, f x = f y → x = y

/-- A natural number n is a perfect square if there exists an m such that n = m^2 -/
def IsPerfectSquare (n : ℕ) : Prop :=
  ∃ m : ℕ, n = m^2

/-- The main theorem: if f satisfies the given condition, then f is injective -/
theorem function_property_implies_injective (f : ℕ → ℕ) 
    (h : ∀ x y : ℕ, x ≠ 0 ∧ y ≠ 0 → (IsPerfectSquare (f x + y) ↔ IsPerfectSquare (x + f y))) : 
    Injective f := by
  sorry

#check function_property_implies_injective

end NUMINAMATH_CALUDE_function_property_implies_injective_l1953_195355


namespace NUMINAMATH_CALUDE_digit_150_is_3_l1953_195302

/-- The decimal expansion of 5/37 has a repeating block of length 3 -/
def repeating_block_length : ℕ := 3

/-- The repeating block in the decimal expansion of 5/37 is [1, 3, 5] -/
def repeating_block : List ℕ := [1, 3, 5]

/-- The 150th digit after the decimal point in the decimal expansion of 5/37 -/
def digit_150 : ℕ := repeating_block[(150 - 1) % repeating_block_length]

theorem digit_150_is_3 : digit_150 = 3 := by
  sorry

end NUMINAMATH_CALUDE_digit_150_is_3_l1953_195302


namespace NUMINAMATH_CALUDE_square_difference_equals_150_l1953_195363

theorem square_difference_equals_150 : (15 + 5)^2 - (5^2 + 15^2) = 150 := by
  sorry

end NUMINAMATH_CALUDE_square_difference_equals_150_l1953_195363


namespace NUMINAMATH_CALUDE_savings_calculation_l1953_195338

/-- Calculates the savings of a person given their income and the ratio of income to expenditure -/
def calculate_savings (income : ℕ) (income_ratio : ℕ) (expenditure_ratio : ℕ) : ℕ :=
  income - (income * expenditure_ratio) / income_ratio

/-- Proves that given a person's income of 14000 and income to expenditure ratio of 7:6, their savings are 2000 -/
theorem savings_calculation :
  calculate_savings 14000 7 6 = 2000 := by
  sorry

end NUMINAMATH_CALUDE_savings_calculation_l1953_195338


namespace NUMINAMATH_CALUDE_f_composition_fixed_points_l1953_195377

def f (x : ℝ) := x^3 - 3*x^2

theorem f_composition_fixed_points :
  ∃ (x : ℝ), f (f x) = f x ∧ (x = 0 ∨ x = 3) :=
sorry

end NUMINAMATH_CALUDE_f_composition_fixed_points_l1953_195377


namespace NUMINAMATH_CALUDE_equal_roots_quadratic_l1953_195301

/-- 
Given a quadratic equation 3x^2 + 6x + m = 0, if it has two equal real roots,
then m = 3.
-/
theorem equal_roots_quadratic (m : ℝ) : 
  (∃ x : ℝ, 3 * x^2 + 6 * x + m = 0 ∧ 
   ∀ y : ℝ, 3 * y^2 + 6 * y + m = 0 → y = x) → 
  m = 3 := by sorry

end NUMINAMATH_CALUDE_equal_roots_quadratic_l1953_195301


namespace NUMINAMATH_CALUDE_equation_solution_l1953_195309

theorem equation_solution (x y : ℝ) : (x + y)^2 = (x + 1) * (y - 1) → x = -1 ∧ y = 1 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l1953_195309


namespace NUMINAMATH_CALUDE_largest_angle_in_special_triangle_l1953_195359

theorem largest_angle_in_special_triangle (A B C : ℝ) (h_triangle : A + B + C = π)
  (h_ratio : (Real.sin B + Real.sin C) / (Real.sin C + Real.sin A) = 4/5 ∧
             (Real.sin C + Real.sin A) / (Real.sin A + Real.sin B) = 5/6) :
  max A (max B C) = 2*π/3 :=
by sorry

end NUMINAMATH_CALUDE_largest_angle_in_special_triangle_l1953_195359

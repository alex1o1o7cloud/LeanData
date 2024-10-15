import Mathlib

namespace NUMINAMATH_CALUDE_det_B_equals_two_l513_51363

open Matrix

theorem det_B_equals_two (x y : ℝ) :
  let B : Matrix (Fin 2) (Fin 2) ℝ := !![x, 2; -3, y]
  (B + 2 * B⁻¹ = 0) → det B = 2 := by
  sorry

end NUMINAMATH_CALUDE_det_B_equals_two_l513_51363


namespace NUMINAMATH_CALUDE_fraction_to_decimal_l513_51311

theorem fraction_to_decimal : (7 : ℚ) / 16 = 0.4375 := by sorry

end NUMINAMATH_CALUDE_fraction_to_decimal_l513_51311


namespace NUMINAMATH_CALUDE_line_point_z_coordinate_l513_51384

/-- Given a line passing through two points in 3D space, 
    find the z-coordinate of a point on the line with a specific x-coordinate. -/
theorem line_point_z_coordinate 
  (p1 : ℝ × ℝ × ℝ) 
  (p2 : ℝ × ℝ × ℝ) 
  (x : ℝ) 
  (h1 : p1 = (1, 3, 2)) 
  (h2 : p2 = (4, 2, -1)) 
  (h3 : x = 7) : 
  ∃ (y z : ℝ), (∃ (t : ℝ), 
    (1 + 3*t, 3 - t, 2 - 3*t) = (x, y, z)) ∧ z = -4 :=
sorry

end NUMINAMATH_CALUDE_line_point_z_coordinate_l513_51384


namespace NUMINAMATH_CALUDE_mary_extra_flour_l513_51320

/-- Given a recipe that calls for a certain amount of flour and the actual amount used,
    calculate the extra amount of flour used. -/
def extra_flour (recipe_amount : ℝ) (actual_amount : ℝ) : ℝ :=
  actual_amount - recipe_amount

/-- Theorem stating that Mary used 2 extra cups of flour -/
theorem mary_extra_flour :
  let recipe_amount : ℝ := 7.0
  let actual_amount : ℝ := 9.0
  extra_flour recipe_amount actual_amount = 2 := by
  sorry

end NUMINAMATH_CALUDE_mary_extra_flour_l513_51320


namespace NUMINAMATH_CALUDE_women_at_gathering_l513_51322

/-- The number of women at a social gathering --/
def number_of_women (number_of_men : ℕ) (dances_per_man : ℕ) (dances_per_woman : ℕ) : ℕ :=
  (number_of_men * dances_per_man) / dances_per_woman

/-- Theorem: At a social gathering with the given conditions, 20 women attended --/
theorem women_at_gathering :
  let number_of_men : ℕ := 15
  let dances_per_man : ℕ := 4
  let dances_per_woman : ℕ := 3
  number_of_women number_of_men dances_per_man dances_per_woman = 20 := by
sorry

#eval number_of_women 15 4 3

end NUMINAMATH_CALUDE_women_at_gathering_l513_51322


namespace NUMINAMATH_CALUDE_point_six_units_from_negative_three_l513_51300

theorem point_six_units_from_negative_three (x : ℝ) : 
  (|x - (-3)| = 6) ↔ (x = 3 ∨ x = -9) := by sorry

end NUMINAMATH_CALUDE_point_six_units_from_negative_three_l513_51300


namespace NUMINAMATH_CALUDE_product_list_price_l513_51371

/-- Given a product with the following properties:
  - Sold at 90% of its list price
  - Earns a profit of 20%
  - Has a cost price of 21 yuan
  Prove that its list price is 28 yuan. -/
theorem product_list_price (list_price : ℝ) : 
  (0.9 * list_price - 21 = 21 * 0.2) → list_price = 28 := by
  sorry

end NUMINAMATH_CALUDE_product_list_price_l513_51371


namespace NUMINAMATH_CALUDE_f_not_monotonic_iff_k_in_range_l513_51395

-- Define the function f
def f (x : ℝ) : ℝ := x^3 - 12*x

-- Define the property of being not monotonic on an interval
def not_monotonic (f : ℝ → ℝ) (a b : ℝ) : Prop :=
  ∃ x y z, a < x ∧ x < y ∧ y < z ∧ z < b ∧
  ((f x < f y ∧ f y > f z) ∨ (f x > f y ∧ f y < f z))

-- Theorem statement
theorem f_not_monotonic_iff_k_in_range (k : ℝ) :
  not_monotonic f (k - 1) (k + 1) ↔ (-3 < k ∧ k < -1) ∨ (1 < k ∧ k < 3) :=
sorry

end NUMINAMATH_CALUDE_f_not_monotonic_iff_k_in_range_l513_51395


namespace NUMINAMATH_CALUDE_sin_sixty_degrees_l513_51301

theorem sin_sixty_degrees : Real.sin (π / 3) = Real.sqrt 3 / 2 := by
  sorry

end NUMINAMATH_CALUDE_sin_sixty_degrees_l513_51301


namespace NUMINAMATH_CALUDE_triangle_expression_simplification_l513_51389

theorem triangle_expression_simplification (a b c : ℝ) 
  (h1 : a + b > c) (h2 : b + c > a) (h3 : a + c > b) : 
  |a + b - c| - |b - a - c| = 2*b - 2*c := by
  sorry

end NUMINAMATH_CALUDE_triangle_expression_simplification_l513_51389


namespace NUMINAMATH_CALUDE_linear_polynomial_impossibility_l513_51358

theorem linear_polynomial_impossibility (a b : ℝ) : 
  ¬(∃ (f : ℝ → ℝ), 
    (∀ x, f x = a * x + b) ∧ 
    (|f 0 - 1| < 1) ∧ 
    (|f 1 - 3| < 1) ∧ 
    (|f 2 - 9| < 1)) := by
  sorry

end NUMINAMATH_CALUDE_linear_polynomial_impossibility_l513_51358


namespace NUMINAMATH_CALUDE_circle_center_is_3_neg4_l513_51305

/-- The equation of a circle in the xy-plane -/
def circle_equation (x y : ℝ) : Prop :=
  x^2 - 6*x + y^2 + 8*y - 12 = 0

/-- The center of a circle -/
def circle_center (h k : ℝ) : Prop :=
  ∀ x y : ℝ, circle_equation x y ↔ (x - h)^2 + (y - k)^2 = (x^2 - 6*x + y^2 + 8*y - 12 + 37) / 2

theorem circle_center_is_3_neg4 : circle_center 3 (-4) := by
  sorry

end NUMINAMATH_CALUDE_circle_center_is_3_neg4_l513_51305


namespace NUMINAMATH_CALUDE_school_tournament_games_l513_51347

/-- The number of games in a round-robin tournament for n teams -/
def roundRobinGames (n : ℕ) : ℕ := n.choose 2

/-- The total number of games in a multi-grade round-robin tournament -/
def totalGames (grade1 grade2 grade3 : ℕ) : ℕ :=
  roundRobinGames grade1 + roundRobinGames grade2 + roundRobinGames grade3

theorem school_tournament_games :
  totalGames 5 8 3 = 41 := by sorry

end NUMINAMATH_CALUDE_school_tournament_games_l513_51347


namespace NUMINAMATH_CALUDE_bluray_price_l513_51380

/-- The price of a Blu-ray movie given the following conditions:
  * 8 DVDs cost $12 each
  * There are 4 Blu-ray movies
  * The average price of all 12 movies is $14
-/
theorem bluray_price :
  ∀ (x : ℝ),
  (8 * 12 + 4 * x) / 12 = 14 →
  x = 18 :=
by sorry

end NUMINAMATH_CALUDE_bluray_price_l513_51380


namespace NUMINAMATH_CALUDE_baking_scoop_size_l513_51378

theorem baking_scoop_size (total_ingredients : ℚ) (num_scoops : ℕ) (scoop_size : ℚ) :
  total_ingredients = 3.75 ∧ num_scoops = 15 ∧ total_ingredients = num_scoops * scoop_size →
  scoop_size = 1/4 := by
  sorry

end NUMINAMATH_CALUDE_baking_scoop_size_l513_51378


namespace NUMINAMATH_CALUDE_percentage_kindergarten_combined_l513_51348

/-- Percentage of Kindergarten students in combined schools -/
theorem percentage_kindergarten_combined (pinegrove_total : ℕ) (maplewood_total : ℕ)
  (pinegrove_k_percent : ℚ) (maplewood_k_percent : ℚ)
  (h1 : pinegrove_total = 150)
  (h2 : maplewood_total = 250)
  (h3 : pinegrove_k_percent = 18/100)
  (h4 : maplewood_k_percent = 14/100) :
  (pinegrove_k_percent * pinegrove_total + maplewood_k_percent * maplewood_total) /
  (pinegrove_total + maplewood_total) = 155/1000 := by
  sorry

#check percentage_kindergarten_combined

end NUMINAMATH_CALUDE_percentage_kindergarten_combined_l513_51348


namespace NUMINAMATH_CALUDE_number_with_remainder_36_mod_45_l513_51328

theorem number_with_remainder_36_mod_45 (k : ℤ) :
  k % 45 = 36 → ∃ (n : ℕ), k = 45 * n + 36 :=
by sorry

end NUMINAMATH_CALUDE_number_with_remainder_36_mod_45_l513_51328


namespace NUMINAMATH_CALUDE_circle_tangent_to_line_l513_51324

-- Define the circle's equation
def circle_equation (x y : ℝ) : Prop := (x + 1)^2 + (y - 2)^2 = 5

-- Define the line's equation
def line_equation (x y : ℝ) : Prop := 2*x - y - 1 = 0

-- State the theorem
theorem circle_tangent_to_line :
  -- The circle has center (-1, 2)
  ∃ (x₀ y₀ : ℝ), x₀ = -1 ∧ y₀ = 2 ∧
  -- The circle is tangent to the line
  ∃ (x y : ℝ), circle_equation x y ∧ line_equation x y ∧
  -- Any point satisfying both equations is unique (tangency condition)
  ∀ (x' y' : ℝ), circle_equation x' y' ∧ line_equation x' y' → x' = x ∧ y' = y :=
sorry

end NUMINAMATH_CALUDE_circle_tangent_to_line_l513_51324


namespace NUMINAMATH_CALUDE_total_gross_profit_after_discounts_l513_51385

/-- Calculate the total gross profit for three items after discounts --/
theorem total_gross_profit_after_discounts
  (price_A price_B price_C : ℝ)
  (gross_profit_percentage : ℝ)
  (discount_A discount_B discount_C : ℝ)
  (h1 : price_A = 91)
  (h2 : price_B = 110)
  (h3 : price_C = 240)
  (h4 : gross_profit_percentage = 1.60)
  (h5 : discount_A = 0.10)
  (h6 : discount_B = 0.05)
  (h7 : discount_C = 0.12) :
  let cost_A := price_A / (1 + gross_profit_percentage)
  let cost_B := price_B / (1 + gross_profit_percentage)
  let cost_C := price_C / (1 + gross_profit_percentage)
  let discounted_price_A := price_A * (1 - discount_A)
  let discounted_price_B := price_B * (1 - discount_B)
  let discounted_price_C := price_C * (1 - discount_C)
  let gross_profit_A := discounted_price_A - cost_A
  let gross_profit_B := discounted_price_B - cost_B
  let gross_profit_C := discounted_price_C - cost_C
  let total_gross_profit := gross_profit_A + gross_profit_B + gross_profit_C
  ∃ ε > 0, |total_gross_profit - 227.98| < ε :=
by
  sorry

end NUMINAMATH_CALUDE_total_gross_profit_after_discounts_l513_51385


namespace NUMINAMATH_CALUDE_multiple_of_six_l513_51390

theorem multiple_of_six (n : ℤ) : 
  (∃ k : ℤ, n = 6 * k) → (∃ m : ℤ, n = 2 * m) ∧ (∃ p : ℤ, n = 3 * p) := by
  sorry

end NUMINAMATH_CALUDE_multiple_of_six_l513_51390


namespace NUMINAMATH_CALUDE_stock_price_after_two_years_l513_51336

/-- The final stock price after a 150% increase followed by a 30% decrease, given an initial price of $120 -/
theorem stock_price_after_two_years (initial_price : ℝ) (first_year_increase : ℝ) (second_year_decrease : ℝ) :
  initial_price = 120 →
  first_year_increase = 150 / 100 →
  second_year_decrease = 30 / 100 →
  initial_price * (1 + first_year_increase) * (1 - second_year_decrease) = 210 := by
  sorry

end NUMINAMATH_CALUDE_stock_price_after_two_years_l513_51336


namespace NUMINAMATH_CALUDE_pony_jeans_discount_rate_l513_51399

theorem pony_jeans_discount_rate
  (fox_price : ℝ)
  (pony_price : ℝ)
  (total_savings : ℝ)
  (fox_quantity : ℕ)
  (pony_quantity : ℕ)
  (total_discount_rate : ℝ)
  (h1 : fox_price = 15)
  (h2 : pony_price = 18)
  (h3 : total_savings = 8.64)
  (h4 : fox_quantity = 3)
  (h5 : pony_quantity = 2)
  (h6 : total_discount_rate = 22) :
  ∃ (fox_discount : ℝ) (pony_discount : ℝ),
    fox_discount + pony_discount = total_discount_rate ∧
    fox_quantity * fox_price * (fox_discount / 100) + pony_quantity * pony_price * (pony_discount / 100) = total_savings ∧
    pony_discount = 14 :=
by sorry

end NUMINAMATH_CALUDE_pony_jeans_discount_rate_l513_51399


namespace NUMINAMATH_CALUDE_zeroPointThreeBarSix_eq_elevenThirties_l513_51346

/-- Represents a repeating decimal with a non-repeating part and a repeating part -/
structure RepeatingDecimal where
  nonRepeating : ℚ
  repeating : ℚ
  repeatingLessThanOne : repeating < 1

/-- The value of a repeating decimal as a rational number -/
def RepeatingDecimal.toRational (d : RepeatingDecimal) : ℚ :=
  d.nonRepeating + d.repeating / (1 - (1/10)^(d.repeating.den))

/-- 0.3̄6 as a RepeatingDecimal -/
def zeroPointThreeBarSix : RepeatingDecimal :=
  { nonRepeating := 3/10
    repeating := 6/10
    repeatingLessThanOne := by sorry }

theorem zeroPointThreeBarSix_eq_elevenThirties : 
  zeroPointThreeBarSix.toRational = 11/30 := by sorry

end NUMINAMATH_CALUDE_zeroPointThreeBarSix_eq_elevenThirties_l513_51346


namespace NUMINAMATH_CALUDE_tan_double_angle_gt_double_tan_l513_51397

theorem tan_double_angle_gt_double_tan (α : Real) (h1 : 0 < α) (h2 : α < π/4) :
  Real.tan (2 * α) > 2 * Real.tan α := by
  sorry

end NUMINAMATH_CALUDE_tan_double_angle_gt_double_tan_l513_51397


namespace NUMINAMATH_CALUDE_extreme_point_condition_monotonicity_for_maximum_two_solutions_condition_l513_51369

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

end NUMINAMATH_CALUDE_extreme_point_condition_monotonicity_for_maximum_two_solutions_condition_l513_51369


namespace NUMINAMATH_CALUDE_forgotten_angle_measure_l513_51307

theorem forgotten_angle_measure (n : ℕ) (sum_without_one : ℝ) : 
  n ≥ 3 → 
  sum_without_one = 2070 → 
  (n - 2) * 180 - sum_without_one = 90 := by
  sorry

end NUMINAMATH_CALUDE_forgotten_angle_measure_l513_51307


namespace NUMINAMATH_CALUDE_polygon_three_sides_l513_51362

/-- A polygon with n sides where the sum of interior angles is less than the sum of exterior angles. -/
structure Polygon (n : ℕ) where
  interior_sum : ℝ
  exterior_sum : ℝ
  interior_less : interior_sum < exterior_sum
  exterior_360 : exterior_sum = 360

/-- Theorem: If a polygon's interior angle sum is less than its exterior angle sum (which is 360°), then it has 3 sides. -/
theorem polygon_three_sides {n : ℕ} (p : Polygon n) : n = 3 := by
  sorry

end NUMINAMATH_CALUDE_polygon_three_sides_l513_51362


namespace NUMINAMATH_CALUDE_problem_proof_l513_51388

theorem problem_proof : -1^2023 + (-8) / (-4) - |(-5)| = -4 := by
  sorry

end NUMINAMATH_CALUDE_problem_proof_l513_51388


namespace NUMINAMATH_CALUDE_marble_theorem_l513_51366

def marble_problem (wolfgang_marbles : ℕ) : Prop :=
  let ludo_marbles : ℕ := wolfgang_marbles + (wolfgang_marbles / 4)
  let total_wolfgang_ludo : ℕ := wolfgang_marbles + ludo_marbles
  let michael_marbles : ℕ := (2 * total_wolfgang_ludo) / 3
  let total_marbles : ℕ := wolfgang_marbles + ludo_marbles + michael_marbles
  wolfgang_marbles = 16 →
  total_marbles / 3 = 20

theorem marble_theorem : marble_problem 16 := by
  sorry

end NUMINAMATH_CALUDE_marble_theorem_l513_51366


namespace NUMINAMATH_CALUDE_banana_arrangement_count_l513_51387

/-- The number of distinct arrangements of the letters in "BANANA" -/
def banana_arrangements : ℕ := 60

/-- The total number of letters in "BANANA" -/
def total_letters : ℕ := 6

/-- The number of occurrences of the letter 'A' in "BANANA" -/
def a_count : ℕ := 3

/-- The number of occurrences of the letter 'N' in "BANANA" -/
def n_count : ℕ := 2

/-- The number of occurrences of the letter 'B' in "BANANA" -/
def b_count : ℕ := 1

theorem banana_arrangement_count :
  banana_arrangements = (Nat.factorial total_letters) / ((Nat.factorial a_count) * (Nat.factorial n_count)) :=
by sorry

end NUMINAMATH_CALUDE_banana_arrangement_count_l513_51387


namespace NUMINAMATH_CALUDE_probability_a_b_same_area_l513_51338

def total_employees : ℕ := 4
def employees_per_area : ℕ := 2
def num_areas : ℕ := 2

def probability_same_area (total : ℕ) (per_area : ℕ) (areas : ℕ) : ℚ :=
  if total = total_employees ∧ per_area = employees_per_area ∧ areas = num_areas then
    1 / 3
  else
    0

theorem probability_a_b_same_area :
  probability_same_area total_employees employees_per_area num_areas = 1 / 3 := by
  sorry

end NUMINAMATH_CALUDE_probability_a_b_same_area_l513_51338


namespace NUMINAMATH_CALUDE_peach_difference_l513_51367

theorem peach_difference (jill_peaches steven_peaches jake_peaches : ℕ) : 
  jill_peaches = 87 →
  steven_peaches = jill_peaches + 18 →
  jake_peaches = jill_peaches + 13 →
  steven_peaches - jake_peaches = 5 := by
sorry

end NUMINAMATH_CALUDE_peach_difference_l513_51367


namespace NUMINAMATH_CALUDE_gcd_lcm_product_l513_51332

theorem gcd_lcm_product (a b : ℕ) (ha : a = 225) (hb : b = 252) :
  (Nat.gcd a b) * (Nat.lcm a b) = 56700 := by
  sorry

end NUMINAMATH_CALUDE_gcd_lcm_product_l513_51332


namespace NUMINAMATH_CALUDE_initial_oranges_count_l513_51370

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

end NUMINAMATH_CALUDE_initial_oranges_count_l513_51370


namespace NUMINAMATH_CALUDE_johns_weekly_sleep_l513_51386

/-- Calculates the total sleep for a week given specific sleep patterns -/
def totalSleepInWeek (daysInWeek : ℕ) (lowSleepDays : ℕ) (lowSleepHours : ℝ) 
                     (recommendedSleep : ℝ) (percentNormalSleep : ℝ) : ℝ :=
  let normalSleepDays := daysInWeek - lowSleepDays
  let normalSleepHours := recommendedSleep * percentNormalSleep
  lowSleepDays * lowSleepHours + normalSleepDays * normalSleepHours

/-- Proves that John's total sleep for the week is 30 hours -/
theorem johns_weekly_sleep : 
  totalSleepInWeek 7 2 3 8 0.6 = 30 := by
  sorry


end NUMINAMATH_CALUDE_johns_weekly_sleep_l513_51386


namespace NUMINAMATH_CALUDE_rubber_duck_race_l513_51310

theorem rubber_duck_race (regular_price large_price large_count total : ℕ) :
  regular_price = 3 →
  large_price = 5 →
  large_count = 185 →
  total = 1588 →
  ∃ regular_count : ℕ, 
    regular_count * regular_price + large_count * large_price = total ∧
    regular_count = 221 := by
  sorry

end NUMINAMATH_CALUDE_rubber_duck_race_l513_51310


namespace NUMINAMATH_CALUDE_intersection_equality_l513_51318

-- Define the sets M and N
def M : Set ℝ := {x | Real.sqrt x < 4}
def N : Set ℝ := {x | 3 * x ≥ 1}

-- Define the intersection set
def intersection_set : Set ℝ := {x | 1/3 ≤ x ∧ x < 16}

-- State the theorem
theorem intersection_equality : M ∩ N = intersection_set := by
  sorry

end NUMINAMATH_CALUDE_intersection_equality_l513_51318


namespace NUMINAMATH_CALUDE_interest_percentage_approx_l513_51337

def purchase_price : ℚ := 2345
def down_payment : ℚ := 385
def num_monthly_payments : ℕ := 18
def monthly_payment : ℚ := 125

def total_paid : ℚ := down_payment + num_monthly_payments * monthly_payment

def interest_paid : ℚ := total_paid - purchase_price

def interest_percentage : ℚ := (interest_paid / purchase_price) * 100

theorem interest_percentage_approx :
  ∃ (ε : ℚ), ε > 0 ∧ ε < 0.1 ∧ |interest_percentage - 12.4| < ε :=
sorry

end NUMINAMATH_CALUDE_interest_percentage_approx_l513_51337


namespace NUMINAMATH_CALUDE_length_of_AF_l513_51394

/-- Given a plot ABCD with known dimensions, prove the length of AF --/
theorem length_of_AF (CE ED AE : ℝ) (area_ABCD : ℝ) 
  (h1 : CE = 40)
  (h2 : ED = 50)
  (h3 : AE = 120)
  (h4 : area_ABCD = 7200) :
  ∃ AF : ℝ, AF = 128 := by
  sorry

end NUMINAMATH_CALUDE_length_of_AF_l513_51394


namespace NUMINAMATH_CALUDE_digimon_pack_cost_is_445_l513_51368

/-- The cost of a pack of Digimon cards -/
def digimon_pack_cost : ℝ := 4.45

/-- The number of Digimon card packs bought -/
def num_digimon_packs : ℕ := 4

/-- The cost of the baseball card deck -/
def baseball_deck_cost : ℝ := 6.06

/-- The total amount spent on cards -/
def total_spent : ℝ := 23.86

/-- Theorem stating that the cost of each Digimon card pack is $4.45 -/
theorem digimon_pack_cost_is_445 :
  digimon_pack_cost * num_digimon_packs + baseball_deck_cost = total_spent :=
by sorry

end NUMINAMATH_CALUDE_digimon_pack_cost_is_445_l513_51368


namespace NUMINAMATH_CALUDE_valid_a_values_l513_51321

def A (a : ℝ) : Set ℝ := {1, 3, a}
def B (a : ℝ) : Set ℝ := {1, a^2 - a + 1}

theorem valid_a_values :
  ∀ a : ℝ, (A a ⊇ B a) → (a = -1 ∨ a = 2) :=
by sorry

end NUMINAMATH_CALUDE_valid_a_values_l513_51321


namespace NUMINAMATH_CALUDE_card_selection_count_l513_51353

def total_cards : ℕ := 12
def red_cards : ℕ := 4
def yellow_cards : ℕ := 4
def blue_cards : ℕ := 4
def cards_to_select : ℕ := 3
def max_red_cards : ℕ := 1

theorem card_selection_count :
  (Nat.choose (yellow_cards + blue_cards) cards_to_select) +
  (Nat.choose red_cards max_red_cards * Nat.choose (yellow_cards + blue_cards) (cards_to_select - max_red_cards)) = 168 := by
  sorry

end NUMINAMATH_CALUDE_card_selection_count_l513_51353


namespace NUMINAMATH_CALUDE_train_length_l513_51312

/-- The length of a train given specific passing times -/
theorem train_length : ∃ (L : ℝ), 
  (∀ (V : ℝ), V = L / 24 → V = (L + 650) / 89) → L = 240 :=
by
  sorry

end NUMINAMATH_CALUDE_train_length_l513_51312


namespace NUMINAMATH_CALUDE_M_equals_m_plus_one_l513_51330

/-- Given natural numbers n, m, h, and b, where n ≥ h(m+1) and h ≥ 1,
    M_{(n, n m, b)} represents a certain combinatorial property. -/
def M (n m h b : ℕ) : ℕ := sorry

/-- Theorem stating that M_{(n, n m, b)} = m + 1 under given conditions -/
theorem M_equals_m_plus_one (n m h b : ℕ) (h1 : n ≥ h * (m + 1)) (h2 : h ≥ 1) :
  M n m h b = m + 1 := by
  sorry

end NUMINAMATH_CALUDE_M_equals_m_plus_one_l513_51330


namespace NUMINAMATH_CALUDE_complex_division_simplification_l513_51383

theorem complex_division_simplification :
  (2 - Complex.I) / (3 + 4 * Complex.I) = 2/25 - 11/25 * Complex.I := by
  sorry

end NUMINAMATH_CALUDE_complex_division_simplification_l513_51383


namespace NUMINAMATH_CALUDE_first_sequence_6th_7th_terms_l513_51304

def first_sequence : ℕ → ℕ
  | 0 => 3
  | n + 1 => 2 * first_sequence n + 1

theorem first_sequence_6th_7th_terms :
  first_sequence 5 = 127 ∧ first_sequence 6 = 255 := by
  sorry

end NUMINAMATH_CALUDE_first_sequence_6th_7th_terms_l513_51304


namespace NUMINAMATH_CALUDE_quadratic_root_difference_l513_51364

theorem quadratic_root_difference : 
  let a : ℝ := 6 + 3 * Real.sqrt 5
  let b : ℝ := -(3 + Real.sqrt 5)
  let c : ℝ := 1
  let discriminant := b^2 - 4*a*c
  let root_difference := Real.sqrt discriminant / a
  root_difference = (Real.sqrt 6 - Real.sqrt 5) / 3 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_root_difference_l513_51364


namespace NUMINAMATH_CALUDE_subset_of_intersection_eq_union_l513_51392

theorem subset_of_intersection_eq_union {A B C : Set α} 
  (hA : A.Nonempty) (hB : B.Nonempty) (hC : C.Nonempty) 
  (h : A ∩ B = B ∪ C) : C ⊆ B := by
  sorry

end NUMINAMATH_CALUDE_subset_of_intersection_eq_union_l513_51392


namespace NUMINAMATH_CALUDE_carpet_cost_proof_l513_51333

theorem carpet_cost_proof (floor_length floor_width carpet_side_length carpet_cost : ℝ) 
  (h1 : floor_length = 24)
  (h2 : floor_width = 64)
  (h3 : carpet_side_length = 8)
  (h4 : carpet_cost = 24) : 
  (floor_length * floor_width) / (carpet_side_length * carpet_side_length) * carpet_cost = 576 := by
  sorry

end NUMINAMATH_CALUDE_carpet_cost_proof_l513_51333


namespace NUMINAMATH_CALUDE_simplify_expression_l513_51382

theorem simplify_expression (x : ℝ) : (3*x + 25) - (2*x - 5) = x + 30 := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l513_51382


namespace NUMINAMATH_CALUDE_train_average_speed_l513_51342

theorem train_average_speed (d1 d2 t1 t2 : ℝ) (h1 : d1 = 225) (h2 : d2 = 370) (h3 : t1 = 3.5) (h4 : t2 = 5) :
  (d1 + d2) / (t1 + t2) = 70 :=
by
  sorry

end NUMINAMATH_CALUDE_train_average_speed_l513_51342


namespace NUMINAMATH_CALUDE_courtyard_paving_l513_51340

-- Define the courtyard dimensions in meters
def courtyard_length : ℝ := 42
def courtyard_width : ℝ := 22

-- Define the brick dimensions in centimeters
def brick_length : ℝ := 16
def brick_width : ℝ := 10

-- Define the conversion factor from square meters to square centimeters
def sq_m_to_sq_cm : ℝ := 10000

-- Theorem statement
theorem courtyard_paving (courtyard_length courtyard_width brick_length brick_width sq_m_to_sq_cm : ℝ) :
  courtyard_length = 42 →
  courtyard_width = 22 →
  brick_length = 16 →
  brick_width = 10 →
  sq_m_to_sq_cm = 10000 →
  (courtyard_length * courtyard_width * sq_m_to_sq_cm) / (brick_length * brick_width) = 57750 :=
by
  sorry


end NUMINAMATH_CALUDE_courtyard_paving_l513_51340


namespace NUMINAMATH_CALUDE_cos_555_degrees_l513_51345

theorem cos_555_degrees : 
  Real.cos (555 * Real.pi / 180) = -(Real.sqrt 6 + Real.sqrt 2) / 4 := by
  sorry

end NUMINAMATH_CALUDE_cos_555_degrees_l513_51345


namespace NUMINAMATH_CALUDE_f_composition_of_one_l513_51344

def f (x : ℤ) : ℤ :=
  if x % 2 = 0 then x / 3 else 4 * x + 1

theorem f_composition_of_one : f (f (f (f 1))) = 341 := by
  sorry

end NUMINAMATH_CALUDE_f_composition_of_one_l513_51344


namespace NUMINAMATH_CALUDE_circle_diameter_from_area_l513_51326

theorem circle_diameter_from_area :
  ∀ (A r d : ℝ),
  A = 81 * Real.pi →
  A = Real.pi * r^2 →
  d = 2 * r →
  d = 18 := by sorry

end NUMINAMATH_CALUDE_circle_diameter_from_area_l513_51326


namespace NUMINAMATH_CALUDE_sue_nuts_count_l513_51334

theorem sue_nuts_count (bill_nuts harry_nuts sue_nuts : ℕ) : 
  bill_nuts = 6 * harry_nuts →
  harry_nuts = 2 * sue_nuts →
  bill_nuts + harry_nuts = 672 →
  sue_nuts = 48 := by
sorry

end NUMINAMATH_CALUDE_sue_nuts_count_l513_51334


namespace NUMINAMATH_CALUDE_sector_max_area_l513_51341

/-- Given a sector with circumference 20, prove that its area is maximized when the central angle is 2 radians. -/
theorem sector_max_area (r : ℝ) (l : ℝ) (α : ℝ) :
  l + 2 * r = 20 →  -- Circumference condition
  l = r * α →       -- Arc length formula
  α = 2 →           -- Proposed maximum angle
  ∀ (r' : ℝ) (l' : ℝ) (α' : ℝ),
    l' + 2 * r' = 20 →
    l' = r' * α' →
    (1/2) * r * l ≥ (1/2) * r' * l' :=
by sorry

end NUMINAMATH_CALUDE_sector_max_area_l513_51341


namespace NUMINAMATH_CALUDE_number_of_children_l513_51375

theorem number_of_children (pencils_per_child : ℕ) (total_pencils : ℕ) (h1 : pencils_per_child = 2) (h2 : total_pencils = 22) :
  total_pencils / pencils_per_child = 11 := by
  sorry

end NUMINAMATH_CALUDE_number_of_children_l513_51375


namespace NUMINAMATH_CALUDE_new_member_amount_l513_51313

theorem new_member_amount (group : Finset ℕ) (group_sum : ℕ) (new_member : ℕ) : 
  Finset.card group = 7 →
  group_sum / 7 = 20 →
  (group_sum + new_member) / 8 = 14 →
  new_member = 756 := by
sorry

end NUMINAMATH_CALUDE_new_member_amount_l513_51313


namespace NUMINAMATH_CALUDE_cone_height_from_circular_sector_l513_51355

theorem cone_height_from_circular_sector (r : ℝ) (h : r = 10) :
  let circumference := 2 * Real.pi * r
  let sector_arc_length := circumference / 3
  let base_radius := sector_arc_length / (2 * Real.pi)
  let height := Real.sqrt (r^2 - base_radius^2)
  height = 20 * Real.sqrt 2 / 3 := by sorry

end NUMINAMATH_CALUDE_cone_height_from_circular_sector_l513_51355


namespace NUMINAMATH_CALUDE_spinner_points_north_l513_51350

/-- Represents the four cardinal directions -/
inductive Direction
  | North
  | East
  | South
  | West

/-- Represents a rotation of the spinner -/
def rotate (initial : Direction) (revolutions : ℚ) : Direction :=
  sorry

/-- Theorem stating that after the described rotations, the spinner points north -/
theorem spinner_points_north :
  let initial_direction := Direction.North
  let clockwise_rotation := 7/2
  let counterclockwise_rotation := 5/2
  rotate (rotate initial_direction clockwise_rotation) (-counterclockwise_rotation) = Direction.North :=
by sorry

end NUMINAMATH_CALUDE_spinner_points_north_l513_51350


namespace NUMINAMATH_CALUDE_intersection_chord_length_l513_51314

/-- The line l in the xy-plane -/
def line_l (x y : ℝ) : Prop :=
  2 * x - 2 * Real.sqrt 3 * y + 2 * Real.sqrt 3 - 1 = 0

/-- The circle C in the xy-plane -/
def circle_C (x y : ℝ) : Prop :=
  (x - 1/2)^2 + (y - 1/2)^2 = 1/2

/-- The theorem stating that the length of the chord formed by the intersection of line l and circle C is √5/2 -/
theorem intersection_chord_length :
  ∃ (A B : ℝ × ℝ),
    line_l A.1 A.2 ∧ line_l B.1 B.2 ∧
    circle_C A.1 A.2 ∧ circle_C B.1 B.2 ∧
    Real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2) = Real.sqrt 5 / 2 :=
sorry

end NUMINAMATH_CALUDE_intersection_chord_length_l513_51314


namespace NUMINAMATH_CALUDE_no_odd_multiples_of_18_24_36_between_1500_3000_l513_51302

theorem no_odd_multiples_of_18_24_36_between_1500_3000 :
  ∀ n : ℕ, 1500 < n ∧ n < 3000 ∧ n % 2 = 1 →
    ¬(18 ∣ n ∧ 24 ∣ n ∧ 36 ∣ n) :=
by sorry

end NUMINAMATH_CALUDE_no_odd_multiples_of_18_24_36_between_1500_3000_l513_51302


namespace NUMINAMATH_CALUDE_eight_coin_flips_sequences_l513_51309

/-- The number of distinct sequences for n coin flips -/
def coin_sequences (n : ℕ) : ℕ := 2^n

/-- Theorem stating that the number of distinct sequences for 8 coin flips is 256 -/
theorem eight_coin_flips_sequences : coin_sequences 8 = 256 := by
  sorry

end NUMINAMATH_CALUDE_eight_coin_flips_sequences_l513_51309


namespace NUMINAMATH_CALUDE_arrangement_count_l513_51319

/-- The number of white pieces -/
def white_pieces : ℕ := 5

/-- The number of black pieces -/
def black_pieces : ℕ := 10

/-- The number of different arrangements of white and black pieces
    satisfying the given conditions -/
def num_arrangements : ℕ := Nat.choose black_pieces white_pieces

theorem arrangement_count :
  num_arrangements = 252 :=
by sorry

end NUMINAMATH_CALUDE_arrangement_count_l513_51319


namespace NUMINAMATH_CALUDE_calculate_responses_needed_l513_51398

/-- The percentage of people who respond to a questionnaire -/
def response_rate : ℝ := 0.60

/-- The minimum number of questionnaires that should be mailed -/
def min_questionnaires : ℕ := 1250

/-- The number of responses needed -/
def responses_needed : ℕ := 750

/-- Theorem: Given the response rate and minimum number of questionnaires,
    the number of responses needed is 750 -/
theorem calculate_responses_needed : 
  ⌊response_rate * min_questionnaires⌋ = responses_needed := by
  sorry

end NUMINAMATH_CALUDE_calculate_responses_needed_l513_51398


namespace NUMINAMATH_CALUDE_hyperbola_eccentricity_l513_51316

/-- Given a hyperbola and a parabola, if the asymptote of the hyperbola
    intersects the parabola at only one point, then the eccentricity
    of the hyperbola is √5. -/
theorem hyperbola_eccentricity (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  (∃ k : ℝ, ∀ x y : ℝ,
    (x^2/a^2 - y^2/b^2 = 1 → y = k*x) ∧
    (x^2 = y - 1 → y = k*x) →
    (∀ z : ℝ, z ≠ x → x^2 = z - 1 → y ≠ k*z)) →
  let c := Real.sqrt (a^2 + b^2)
  c/a = Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_hyperbola_eccentricity_l513_51316


namespace NUMINAMATH_CALUDE_equal_powers_implies_equality_l513_51379

theorem equal_powers_implies_equality (a b : ℝ) : 
  0 < a → 0 < b → a^b = b^a → a < 1 → a = b := by
sorry

end NUMINAMATH_CALUDE_equal_powers_implies_equality_l513_51379


namespace NUMINAMATH_CALUDE_strawberry_weight_calculation_l513_51339

def total_fruit_weight : ℕ := 10
def apple_weight : ℕ := 3
def orange_weight : ℕ := 1
def grape_weight : ℕ := 3

theorem strawberry_weight_calculation :
  total_fruit_weight - (apple_weight + orange_weight + grape_weight) = 3 :=
by
  sorry

end NUMINAMATH_CALUDE_strawberry_weight_calculation_l513_51339


namespace NUMINAMATH_CALUDE_systematic_sampling_interval_l513_51396

theorem systematic_sampling_interval
  (population_size : ℕ)
  (sample_size : ℕ)
  (h1 : population_size = 800)
  (h2 : sample_size = 40)
  : population_size / sample_size = 20 := by
  sorry

end NUMINAMATH_CALUDE_systematic_sampling_interval_l513_51396


namespace NUMINAMATH_CALUDE_june_election_win_l513_51331

theorem june_election_win (total_students : ℕ) (boy_percentage : ℚ) (june_male_vote_percentage : ℚ) 
  (h_total : total_students = 200)
  (h_boy : boy_percentage = 3/5)
  (h_june_male : june_male_vote_percentage = 27/40)
  : ∃ (min_female_vote_percentage : ℚ), 
    min_female_vote_percentage ≥ 1/4 ∧ 
    (boy_percentage * june_male_vote_percentage + (1 - boy_percentage) * min_female_vote_percentage) * total_students > total_students / 2 := by
  sorry

end NUMINAMATH_CALUDE_june_election_win_l513_51331


namespace NUMINAMATH_CALUDE_bakery_pie_division_l513_51372

theorem bakery_pie_division (total_pie : ℚ) (num_people : ℕ) : 
  total_pie = 5/6 ∧ num_people = 4 → total_pie / num_people = 5/24 := by
  sorry

end NUMINAMATH_CALUDE_bakery_pie_division_l513_51372


namespace NUMINAMATH_CALUDE_average_age_increase_l513_51393

/-- Proves that adding a 28-year-old student to a class of 9 students with an average age of 8 years increases the overall average age by 2 years -/
theorem average_age_increase (total_students : ℕ) (initial_students : ℕ) (initial_average : ℝ) (new_student_age : ℕ) :
  total_students = 10 →
  initial_students = 9 →
  initial_average = 8 →
  new_student_age = 28 →
  (initial_students * initial_average + new_student_age) / total_students - initial_average = 2 := by
  sorry

end NUMINAMATH_CALUDE_average_age_increase_l513_51393


namespace NUMINAMATH_CALUDE_difference_of_squares_l513_51351

theorem difference_of_squares (a b : ℝ) : (a + b) * (a - b) = a^2 - b^2 := by
  sorry

end NUMINAMATH_CALUDE_difference_of_squares_l513_51351


namespace NUMINAMATH_CALUDE_exactly_one_shot_probability_l513_51391

/-- The probability that exactly one person makes a shot given the probabilities of A and B making shots. -/
theorem exactly_one_shot_probability (p_a p_b : ℝ) (h_a : p_a = 0.8) (h_b : p_b = 0.6) :
  p_a * (1 - p_b) + (1 - p_a) * p_b = 0.44 := by
  sorry

end NUMINAMATH_CALUDE_exactly_one_shot_probability_l513_51391


namespace NUMINAMATH_CALUDE_work_completion_time_l513_51354

/-- The time it takes to complete a work with two workers working sequentially -/
def total_work_time (mahesh_full_time : ℕ) (mahesh_work_time : ℕ) (rajesh_finish_time : ℕ) : ℕ :=
  mahesh_work_time + rajesh_finish_time

/-- Theorem stating that under given conditions, the total work time is 50 days -/
theorem work_completion_time :
  total_work_time 45 20 30 = 50 := by
  sorry

end NUMINAMATH_CALUDE_work_completion_time_l513_51354


namespace NUMINAMATH_CALUDE_inequality_proof_l513_51352

theorem inequality_proof (x y k : ℝ) 
  (h1 : x > 0) 
  (h2 : y > 0) 
  (h3 : x ≠ y) 
  (h4 : k > 0) 
  (h5 : k < 2) : 
  ((x + y) / 2) ^ k > (Real.sqrt (x * y)) ^ k ∧ 
  (Real.sqrt (x * y)) ^ k > (2 * x * y / (x + y)) ^ k := by
sorry

end NUMINAMATH_CALUDE_inequality_proof_l513_51352


namespace NUMINAMATH_CALUDE_chocolate_price_proof_l513_51343

/-- Proves that if a chocolate's price is reduced by 57 cents and the resulting price is $1.43, then the original price was $2.00. -/
theorem chocolate_price_proof (original_price : ℝ) : 
  (original_price - 0.57 = 1.43) → original_price = 2.00 := by
  sorry

end NUMINAMATH_CALUDE_chocolate_price_proof_l513_51343


namespace NUMINAMATH_CALUDE_difference_of_squares_73_47_l513_51317

theorem difference_of_squares_73_47 : 73^2 - 47^2 = 3120 := by
  sorry

end NUMINAMATH_CALUDE_difference_of_squares_73_47_l513_51317


namespace NUMINAMATH_CALUDE_min_colors_for_four_color_rect_l513_51356

/-- Represents a coloring of an n × n board using k colors. -/
structure Coloring (n k : ℕ) :=
  (colors : Fin n → Fin n → Fin k)
  (all_used : ∀ c : Fin k, ∃ i j : Fin n, colors i j = c)

/-- Checks if four cells at the intersections of two rows and two columns have different colors. -/
def hasFourColorRect (n k : ℕ) (c : Coloring n k) : Prop :=
  ∃ i₁ i₂ j₁ j₂ : Fin n, i₁ ≠ i₂ ∧ j₁ ≠ j₂ ∧
    c.colors i₁ j₁ ≠ c.colors i₁ j₂ ∧
    c.colors i₁ j₁ ≠ c.colors i₂ j₁ ∧
    c.colors i₁ j₁ ≠ c.colors i₂ j₂ ∧
    c.colors i₁ j₂ ≠ c.colors i₂ j₁ ∧
    c.colors i₁ j₂ ≠ c.colors i₂ j₂ ∧
    c.colors i₂ j₁ ≠ c.colors i₂ j₂

/-- The main theorem stating that 2n is the smallest number of colors
    that guarantees a four-color rectangle in any coloring. -/
theorem min_colors_for_four_color_rect (n : ℕ) (h : n ≥ 2) :
  (∀ k : ℕ, k ≥ 2*n → ∀ c : Coloring n k, hasFourColorRect n k c) ∧
  (∃ c : Coloring n (2*n - 1), ¬hasFourColorRect n (2*n - 1) c) :=
sorry

end NUMINAMATH_CALUDE_min_colors_for_four_color_rect_l513_51356


namespace NUMINAMATH_CALUDE_expression_simplification_l513_51306

theorem expression_simplification (x : ℝ) (h : x = 3) :
  (x / (x - 2) - x / (x + 2)) / (4 * x / (x - 2)) = 1 / 5 := by
  sorry

end NUMINAMATH_CALUDE_expression_simplification_l513_51306


namespace NUMINAMATH_CALUDE_school_average_difference_l513_51361

theorem school_average_difference : 
  let total_students : ℕ := 120
  let total_teachers : ℕ := 6
  let class_sizes : List ℕ := [60, 30, 15, 10, 3, 2]
  let t : ℚ := (total_students : ℚ) / total_teachers
  let s : ℚ := (class_sizes.map (λ size => (size : ℚ) * size / total_students)).sum
  t - s = -20316 / 1000 := by
sorry

end NUMINAMATH_CALUDE_school_average_difference_l513_51361


namespace NUMINAMATH_CALUDE_no_positive_integer_solutions_l513_51349

theorem no_positive_integer_solutions : 
  ¬∃ (x y : ℕ), x > 0 ∧ y > 0 ∧ (x^2 * y^2)^2 - 14 * (x^2 * y^2) + 49 = 0 :=
by sorry

end NUMINAMATH_CALUDE_no_positive_integer_solutions_l513_51349


namespace NUMINAMATH_CALUDE_factor_expression_l513_51376

theorem factor_expression (z : ℝ) :
  75 * z^24 + 225 * z^48 = 75 * z^24 * (1 + 3 * z^24) := by
  sorry

end NUMINAMATH_CALUDE_factor_expression_l513_51376


namespace NUMINAMATH_CALUDE_solve_for_k_l513_51323

theorem solve_for_k : ∃ k : ℚ, 
  (let x : ℚ := -3
   k * (x - 2) - 4 = k - 2 * x) ∧ 
  k = -5/3 := by
  sorry

end NUMINAMATH_CALUDE_solve_for_k_l513_51323


namespace NUMINAMATH_CALUDE_contractor_daily_wage_l513_51373

/-- Contractor's daily wage problem -/
theorem contractor_daily_wage (total_days : ℕ) (absent_days : ℕ) (fine_per_day : ℚ) (total_pay : ℚ) :
  total_days = 30 →
  absent_days = 2 →
  fine_per_day = 15/2 →
  total_pay = 685 →
  ∃ (daily_wage : ℚ), 
    daily_wage * (total_days - absent_days : ℚ) - fine_per_day * absent_days = total_pay ∧
    daily_wage = 25 :=
by sorry

end NUMINAMATH_CALUDE_contractor_daily_wage_l513_51373


namespace NUMINAMATH_CALUDE_existence_of_special_set_l513_51381

theorem existence_of_special_set (n : ℕ) (hn : n ≥ 3) :
  ∃ (S : Finset ℕ),
    (Finset.card S = 2 * n) ∧
    (∀ m : ℕ, 2 ≤ m ∧ m ≤ n →
      ∃ (A : Finset ℕ),
        A ⊆ S ∧
        Finset.card A = m ∧
        2 * (A.sum id) = S.sum id) :=
  sorry

end NUMINAMATH_CALUDE_existence_of_special_set_l513_51381


namespace NUMINAMATH_CALUDE_min_sum_m_n_l513_51359

theorem min_sum_m_n (m n : ℕ+) (h : 98 * m = n^3) : 
  (∀ (m' n' : ℕ+), 98 * m' = n'^3 → m' + n' ≥ m + n) → m + n = 42 :=
by sorry

end NUMINAMATH_CALUDE_min_sum_m_n_l513_51359


namespace NUMINAMATH_CALUDE_share_price_calculation_l513_51377

/-- Proves the price of shares given dividend rate, face value, and return on investment -/
theorem share_price_calculation (dividend_rate : ℝ) (face_value : ℝ) (roi : ℝ) : 
  dividend_rate = 0.125 →
  face_value = 60 →
  roi = 0.25 →
  dividend_rate * face_value = roi * (face_value * dividend_rate / roi) := by
  sorry

#check share_price_calculation

end NUMINAMATH_CALUDE_share_price_calculation_l513_51377


namespace NUMINAMATH_CALUDE_g_in_M_l513_51303

-- Define the set M
def M : Set (ℝ → ℝ) :=
  {f | ∀ x₁ x₂, |x₁| ≤ 1 → |x₂| ≤ 1 → |f x₁ - f x₂| ≤ 4 * |x₁ - x₂|}

-- Define the function g
def g : ℝ → ℝ := λ x ↦ x^2 + 2*x - 1

-- Theorem statement
theorem g_in_M : g ∈ M := by
  sorry

end NUMINAMATH_CALUDE_g_in_M_l513_51303


namespace NUMINAMATH_CALUDE_triangle_rectangle_ratio_l513_51335

theorem triangle_rectangle_ratio : 
  ∀ (triangle_leg : ℝ) (rect_short_side : ℝ),
  triangle_leg > 0 ∧ rect_short_side > 0 →
  2 * triangle_leg + Real.sqrt 2 * triangle_leg = 48 →
  2 * (rect_short_side + 2 * rect_short_side) = 48 →
  (Real.sqrt 2 * triangle_leg) / rect_short_side = 3 * (2 * Real.sqrt 2 - 2) :=
by sorry

end NUMINAMATH_CALUDE_triangle_rectangle_ratio_l513_51335


namespace NUMINAMATH_CALUDE_circle_equation_radius_8_l513_51327

/-- The equation x^2 + 14x + y^2 + 10y - k = 0 represents a circle of radius 8 if and only if k = 10 -/
theorem circle_equation_radius_8 (x y k : ℝ) : 
  (∃ h₁ h₂ : ℝ, ∀ x y : ℝ, x^2 + 14*x + y^2 + 10*y - k = 0 ↔ (x - h₁)^2 + (y - h₂)^2 = 64) ↔ 
  k = 10 := by
sorry

end NUMINAMATH_CALUDE_circle_equation_radius_8_l513_51327


namespace NUMINAMATH_CALUDE_subsets_of_three_element_set_l513_51325

theorem subsets_of_three_element_set : 
  Finset.card (Finset.powerset {1, 2, 3}) = 8 := by sorry

end NUMINAMATH_CALUDE_subsets_of_three_element_set_l513_51325


namespace NUMINAMATH_CALUDE_raja_savings_l513_51329

def monthly_income : ℝ := 37500

def household_percentage : ℝ := 35
def clothes_percentage : ℝ := 20
def medicines_percentage : ℝ := 5

def total_expenditure_percentage : ℝ := household_percentage + clothes_percentage + medicines_percentage

def savings_percentage : ℝ := 100 - total_expenditure_percentage

theorem raja_savings : (savings_percentage / 100) * monthly_income = 15000 := by
  sorry

end NUMINAMATH_CALUDE_raja_savings_l513_51329


namespace NUMINAMATH_CALUDE_domestic_tourists_scientific_notation_l513_51308

/-- Represents a number in scientific notation -/
structure ScientificNotation where
  coefficient : ℝ
  exponent : ℤ
  coefficient_range : 1 ≤ coefficient ∧ coefficient < 10

/-- Converts a real number to scientific notation -/
def toScientificNotation (x : ℝ) : ScientificNotation :=
  sorry

theorem domestic_tourists_scientific_notation :
  toScientificNotation 274000000 =
    ScientificNotation.mk 2.74 8 (by norm_num) :=
by sorry

end NUMINAMATH_CALUDE_domestic_tourists_scientific_notation_l513_51308


namespace NUMINAMATH_CALUDE_quadratic_completion_l513_51360

theorem quadratic_completion (x : ℝ) :
  ∃ (d e : ℝ), x^2 - 24*x + 45 = (x + d)^2 + e ∧ d + e = -111 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_completion_l513_51360


namespace NUMINAMATH_CALUDE_code_cracker_combinations_l513_51365

/-- The number of different colors of pegs in the CodeCracker game -/
def num_colors : ℕ := 6

/-- The number of slots for pegs in the CodeCracker game -/
def num_slots : ℕ := 5

/-- The total number of possible secret codes in the CodeCracker game -/
def total_codes : ℕ := num_colors ^ num_slots

/-- Theorem stating that the total number of possible secret codes in the CodeCracker game is 7776 -/
theorem code_cracker_combinations : total_codes = 7776 := by
  sorry

end NUMINAMATH_CALUDE_code_cracker_combinations_l513_51365


namespace NUMINAMATH_CALUDE_january_salary_l513_51315

/-- Given the average salaries for two four-month periods and the salary for May,
    prove that the salary for January is 5700. -/
theorem january_salary
  (avg_jan_to_apr : (jan + feb + mar + apr) / 4 = 8000)
  (avg_feb_to_may : (feb + mar + apr + may) / 4 = 8200)
  (may_salary : may = 6500)
  : jan = 5700 := by
  sorry

end NUMINAMATH_CALUDE_january_salary_l513_51315


namespace NUMINAMATH_CALUDE_corner_triangles_area_l513_51357

/-- Given a square with side length 16 units, if we remove four isosceles right triangles 
    from its corners, where the leg of each triangle is 1/4 of the square's side length, 
    the total area of the removed triangles is 32 square units. -/
theorem corner_triangles_area (square_side : ℝ) (triangle_leg : ℝ) : 
  square_side = 16 → 
  triangle_leg = square_side / 4 → 
  4 * (1/2 * triangle_leg^2) = 32 :=
by sorry

end NUMINAMATH_CALUDE_corner_triangles_area_l513_51357


namespace NUMINAMATH_CALUDE_imaginary_part_of_z_l513_51374

theorem imaginary_part_of_z (z : ℂ) (h : (1 + 2 * Complex.I) * z = Complex.I) : 
  z.im = 1 / 5 := by
sorry

end NUMINAMATH_CALUDE_imaginary_part_of_z_l513_51374

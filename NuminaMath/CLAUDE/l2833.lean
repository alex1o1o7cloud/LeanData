import Mathlib

namespace NUMINAMATH_CALUDE_hyperbola_asymptotes_l2833_283368

/-- Given a hyperbola with equation x²/m - y²/n = 1 where mn ≠ 0, 
    eccentricity 2, and one focus at (1, 0), 
    prove that its asymptotes are √3x ± y = 0 -/
theorem hyperbola_asymptotes 
  (m n : ℝ) 
  (h1 : m * n ≠ 0) 
  (h2 : ∀ x y : ℝ, x^2 / m - y^2 / n = 1) 
  (h3 : (Real.sqrt (m + n)) / (Real.sqrt m) = 2) 
  (h4 : ∃ x y : ℝ, x^2 / m - y^2 / n = 1 ∧ x = 1 ∧ y = 0) : 
  ∃ k : ℝ, k = Real.sqrt 3 ∧ 
    (∀ x y : ℝ, (k * x = y ∨ k * x = -y) ↔ x^2 / m - y^2 / n = 0) :=
sorry

end NUMINAMATH_CALUDE_hyperbola_asymptotes_l2833_283368


namespace NUMINAMATH_CALUDE_quadratic_root_problem_l2833_283337

theorem quadratic_root_problem (m : ℝ) : 
  (1 : ℝ) ^ 2 + m * 1 - 4 = 0 → 
  ∃ (x : ℝ), x ≠ 1 ∧ x ^ 2 + m * x - 4 = 0 ∧ x = -4 := by
sorry

end NUMINAMATH_CALUDE_quadratic_root_problem_l2833_283337


namespace NUMINAMATH_CALUDE_xy_sum_over_five_l2833_283323

theorem xy_sum_over_five (x y : ℝ) (h1 : x * y > 0) (h2 : 1 / x + 1 / y = 15) (h3 : 1 / (x * y) = 5) :
  (x + y) / 5 = 3 / 5 := by
  sorry

end NUMINAMATH_CALUDE_xy_sum_over_five_l2833_283323


namespace NUMINAMATH_CALUDE_system_equation_ratio_l2833_283303

theorem system_equation_ratio (x y c d : ℝ) (h1 : 4 * x - 3 * y = c) (h2 : 2 * y - 8 * x = d) (h3 : d ≠ 0) :
  c / d = -1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_system_equation_ratio_l2833_283303


namespace NUMINAMATH_CALUDE_fraction_sum_inequality_l2833_283352

theorem fraction_sum_inequality (a b c d : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (hd : d > 0) :
  (a / (b + 2*c + 3*d)) + (b / (c + 2*d + 3*a)) + (c / (d + 2*a + 3*b)) + (d / (a + 2*b + 3*c)) ≥ 2/3 := by
  sorry

end NUMINAMATH_CALUDE_fraction_sum_inequality_l2833_283352


namespace NUMINAMATH_CALUDE_complex_fraction_equality_l2833_283327

theorem complex_fraction_equality : Complex.I ^ 2 = -1 → (2 : ℂ) / (1 - Complex.I) = 1 + Complex.I := by
  sorry

end NUMINAMATH_CALUDE_complex_fraction_equality_l2833_283327


namespace NUMINAMATH_CALUDE_inequality_and_equality_condition_l2833_283309

theorem inequality_and_equality_condition (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  1 / a^2 + 1 / b^2 + 8 * a * b ≥ 8 ∧
  (1 / a^2 + 1 / b^2 + 8 * a * b = 8 ↔ a = 1/2 ∧ b = 1/2) := by
  sorry

end NUMINAMATH_CALUDE_inequality_and_equality_condition_l2833_283309


namespace NUMINAMATH_CALUDE_simplify_expression_l2833_283384

theorem simplify_expression (a : ℝ) (h : a < 2) : 
  Real.sqrt ((a - 2)^2) + a - 1 = 1 := by
sorry

end NUMINAMATH_CALUDE_simplify_expression_l2833_283384


namespace NUMINAMATH_CALUDE_number_of_values_in_calculation_l2833_283330

theorem number_of_values_in_calculation 
  (initial_average : ℝ)
  (correct_average : ℝ)
  (incorrect_value : ℝ)
  (correct_value : ℝ)
  (h1 : initial_average = 46)
  (h2 : correct_average = 51)
  (h3 : incorrect_value = 25)
  (h4 : correct_value = 75) :
  ∃ (n : ℕ), n > 0 ∧ 
    n * initial_average + (correct_value - incorrect_value) = n * correct_average ∧
    n = 10 := by
sorry

end NUMINAMATH_CALUDE_number_of_values_in_calculation_l2833_283330


namespace NUMINAMATH_CALUDE_remainder_theorem_l2833_283353

theorem remainder_theorem (A B : ℕ) (h : A = B * 9 + 13) : A % 9 = 4 := by
  sorry

end NUMINAMATH_CALUDE_remainder_theorem_l2833_283353


namespace NUMINAMATH_CALUDE_coin_identification_possible_l2833_283395

/-- Represents the expert's response, which is always an overestimate -/
structure ExpertResponse :=
  (reported : ℕ)
  (actual : ℕ)
  (overestimate : ℕ)
  (h : reported = actual + overestimate)

/-- Represents the coin identification process -/
def can_identify_counterfeit (total_coins : ℕ) (max_presentation : ℕ) : Prop :=
  ∀ (counterfeit : Finset ℕ) (overestimate : ℕ),
    counterfeit.card ≤ total_coins →
    (∀ subset : Finset ℕ, subset.card ≤ max_presentation →
      ∃ response : ExpertResponse,
        response.actual = (subset ∩ counterfeit).card ∧
        response.overestimate = overestimate) →
    ∃ process : ℕ → Bool,
      ∀ coin, coin < total_coins → (process coin ↔ coin ∈ counterfeit)

theorem coin_identification_possible :
  can_identify_counterfeit 100 20 :=
sorry

end NUMINAMATH_CALUDE_coin_identification_possible_l2833_283395


namespace NUMINAMATH_CALUDE_integer_roots_of_polynomial_l2833_283357

theorem integer_roots_of_polynomial (a : ℤ) : 
  a = -4 →
  (∀ x : ℤ, x^4 - 16*x^3 + (81-2*a)*x^2 + (16*a-142)*x + a^2 - 21*a + 68 = 0 ↔ 
    x = 2 ∨ x = 3 ∨ x = 4 ∨ x = 7) := by
  sorry

#check integer_roots_of_polynomial

end NUMINAMATH_CALUDE_integer_roots_of_polynomial_l2833_283357


namespace NUMINAMATH_CALUDE_chocolate_profit_example_l2833_283349

/-- Calculates the profit from selling chocolates given the following conditions:
  * Number of chocolate bars
  * Cost price per bar
  * Total selling price
  * Packaging cost per bar
-/
def chocolate_profit (num_bars : ℕ) (cost_price : ℚ) (total_selling_price : ℚ) (packaging_cost : ℚ) : ℚ :=
  total_selling_price - (num_bars * (cost_price + packaging_cost))

theorem chocolate_profit_example :
  chocolate_profit 5 5 90 2 = 55 := by
  sorry

end NUMINAMATH_CALUDE_chocolate_profit_example_l2833_283349


namespace NUMINAMATH_CALUDE_product_of_one_fourth_and_one_half_l2833_283381

theorem product_of_one_fourth_and_one_half : (1 / 4 : ℚ) * (1 / 2 : ℚ) = 1 / 8 := by
  sorry

end NUMINAMATH_CALUDE_product_of_one_fourth_and_one_half_l2833_283381


namespace NUMINAMATH_CALUDE_max_min_values_l2833_283310

-- Define the conditions
def positive_xy (x y : ℝ) : Prop := x > 0 ∧ y > 0
def constraint (x y : ℝ) : Prop := 3 * x + 2 * y = 10

-- Define the theorem
theorem max_min_values (x y : ℝ) 
  (h1 : positive_xy x y) (h2 : constraint x y) : 
  (∃ (m : ℝ), m = Real.sqrt (3 * x) + Real.sqrt (2 * y) ∧ 
    m ≤ 2 * Real.sqrt 5 ∧ 
    ∀ (x' y' : ℝ), positive_xy x' y' → constraint x' y' → 
      Real.sqrt (3 * x') + Real.sqrt (2 * y') ≤ m) ∧
  (∃ (n : ℝ), n = 3 / x + 2 / y ∧ 
    n ≥ 5 / 2 ∧ 
    ∀ (x' y' : ℝ), positive_xy x' y' → constraint x' y' → 
      3 / x' + 2 / y' ≥ n) :=
by sorry

end NUMINAMATH_CALUDE_max_min_values_l2833_283310


namespace NUMINAMATH_CALUDE_least_addition_to_perfect_square_l2833_283398

theorem least_addition_to_perfect_square : ∃ (x : ℝ), 
  (x ≥ 0) ∧ 
  (∃ (n : ℕ), (0.0320 + x) = n^2) ∧
  (∀ (y : ℝ), y ≥ 0 → (∃ (m : ℕ), (0.0320 + y) = m^2) → y ≥ x) ∧
  (x = 0.9680) := by
sorry

end NUMINAMATH_CALUDE_least_addition_to_perfect_square_l2833_283398


namespace NUMINAMATH_CALUDE_more_boys_than_girls_l2833_283369

theorem more_boys_than_girls (total : ℕ) (boys : ℕ) (girls : ℕ) : 
  total = 100 →
  boys + girls = total →
  3 * girls = 2 * boys →
  boys - girls = 20 :=
by
  sorry

end NUMINAMATH_CALUDE_more_boys_than_girls_l2833_283369


namespace NUMINAMATH_CALUDE_ellipse_line_intersection_l2833_283366

def ellipse (x y : ℝ) : Prop := x^2 / 2 + y^2 = 1

def line (k m x y : ℝ) : Prop := y = k * x + m

def perpendicular_bisector (x₁ y₁ x₂ y₂ x y : ℝ) : Prop :=
  (y - (y₁ + y₂) / 2) / (x - (x₁ + x₂) / 2) = -(x₂ - x₁) / (y₂ - y₁)

theorem ellipse_line_intersection (k m : ℝ) :
  (∃ x₁ y₁ x₂ y₂ : ℝ,
    ellipse x₁ y₁ ∧ ellipse x₂ y₂ ∧
    line k m x₁ y₁ ∧ line k m x₂ y₂ ∧
    perpendicular_bisector x₁ y₁ x₂ y₂ 0 (-1/2)) →
  2 * k^2 + 1 = 2 * m := by
sorry

end NUMINAMATH_CALUDE_ellipse_line_intersection_l2833_283366


namespace NUMINAMATH_CALUDE_unique_product_sum_relation_l2833_283331

theorem unique_product_sum_relation (a b c : ℕ+) :
  (a * b * c = 8 * (a + b + c)) ∧ 
  ((c = 2 * a + b) ∨ (b = 2 * a + c) ∨ (a = 2 * b + c)) →
  a * b * c = 136 :=
sorry

end NUMINAMATH_CALUDE_unique_product_sum_relation_l2833_283331


namespace NUMINAMATH_CALUDE_number_satisfying_condition_l2833_283307

theorem number_satisfying_condition : ∃ x : ℝ, (0.1 * x = 0.2 * 650 + 190) ∧ (x = 3200) := by
  sorry

end NUMINAMATH_CALUDE_number_satisfying_condition_l2833_283307


namespace NUMINAMATH_CALUDE_sum_squares_plus_product_lower_bound_l2833_283315

theorem sum_squares_plus_product_lower_bound 
  (a b c : ℝ) 
  (h1 : 0 ≤ a) (h2 : 0 ≤ b) (h3 : 0 ≤ c) 
  (h4 : a + b + c = 3) : 
  a^2 + b^2 + c^2 + a*b*c ≥ 4 := by
  sorry

end NUMINAMATH_CALUDE_sum_squares_plus_product_lower_bound_l2833_283315


namespace NUMINAMATH_CALUDE_quadratic_inequality_always_positive_l2833_283359

theorem quadratic_inequality_always_positive (m : ℝ) :
  (∀ x : ℝ, x^2 - (m - 4)*x - m + 7 > 0) ↔ m > -2 ∧ m < 6 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_inequality_always_positive_l2833_283359


namespace NUMINAMATH_CALUDE_three_digit_sum_reduction_l2833_283386

def is_valid_number (n : ℕ) : Prop :=
  100 ≤ n ∧ n ≤ 999 ∧
  let d1 := n / 100
  let d2 := (n / 10) % 10
  let d3 := n % 10
  let sum := d1 + d2 + d3
  let n_plus_3 := n + 3
  let d1_new := n_plus_3 / 100
  let d2_new := (n_plus_3 / 10) % 10
  let d3_new := n_plus_3 % 10
  let sum_new := d1_new + d2_new + d3_new
  sum_new = sum / 3

theorem three_digit_sum_reduction :
  ∀ n : ℕ, is_valid_number n ↔ n = 117 ∨ n = 207 ∨ n = 108 :=
sorry

end NUMINAMATH_CALUDE_three_digit_sum_reduction_l2833_283386


namespace NUMINAMATH_CALUDE_min_value_of_f_l2833_283344

def f (x : ℕ+) : ℚ := (x.val^2 + 33) / x.val

theorem min_value_of_f :
  (∀ x : ℕ+, f x ≥ 23/2) ∧ (∃ x : ℕ+, f x = 23/2) := by sorry

end NUMINAMATH_CALUDE_min_value_of_f_l2833_283344


namespace NUMINAMATH_CALUDE_three_prime_divisors_of_nine_power_minus_one_l2833_283354

theorem three_prime_divisors_of_nine_power_minus_one (n : ℕ) (x : ℕ) 
  (h1 : x = 9^n - 1)
  (h2 : (Nat.factors x).toFinset.card = 3)
  (h3 : 7 ∈ Nat.factors x) :
  x = 728 := by sorry

end NUMINAMATH_CALUDE_three_prime_divisors_of_nine_power_minus_one_l2833_283354


namespace NUMINAMATH_CALUDE_nacl_formed_l2833_283348

-- Define the chemical reaction
structure Reaction where
  reactant1 : String
  reactant2 : String
  product1 : String
  product2 : String
  product3 : String

-- Define the moles of substances
structure Moles where
  nh4cl : ℝ
  naoh : ℝ
  nacl : ℝ

-- Define the reaction and initial moles
def reaction : Reaction :=
  { reactant1 := "NH4Cl"
  , reactant2 := "NaOH"
  , product1 := "NaCl"
  , product2 := "NH3"
  , product3 := "H2O" }

def initial_moles : Moles :=
  { nh4cl := 2
  , naoh := 2
  , nacl := 0 }

-- Theorem statement
theorem nacl_formed (r : Reaction) (m : Moles) :
  r = reaction ∧ m = initial_moles →
  m.nacl + 2 = 2 :=
by sorry

end NUMINAMATH_CALUDE_nacl_formed_l2833_283348


namespace NUMINAMATH_CALUDE_smallest_n_divisibility_l2833_283333

theorem smallest_n_divisibility : ∃ (n : ℕ), n > 0 ∧ 
  36 ∣ n^2 ∧ 1024 ∣ n^3 ∧ 
  ∀ (m : ℕ), m > 0 → 36 ∣ m^2 → 1024 ∣ m^3 → n ≤ m :=
by
  -- Proof goes here
  sorry

end NUMINAMATH_CALUDE_smallest_n_divisibility_l2833_283333


namespace NUMINAMATH_CALUDE_retail_price_calculation_l2833_283380

def calculate_retail_price (wholesale : ℝ) (profit_percent : ℝ) (discount_percent : ℝ) : ℝ :=
  let intended_price := wholesale * (1 + profit_percent)
  intended_price * (1 - discount_percent)

def overall_retail_price (w1 w2 w3 : ℝ) (p1 p2 p3 : ℝ) (d1 d2 d3 : ℝ) : ℝ :=
  calculate_retail_price w1 p1 d1 +
  calculate_retail_price w2 p2 d2 +
  calculate_retail_price w3 p3 d3

theorem retail_price_calculation :
  overall_retail_price 90 120 200 0.20 0.30 0.25 0.10 0.15 0.05 = 467.30 := by
  sorry

end NUMINAMATH_CALUDE_retail_price_calculation_l2833_283380


namespace NUMINAMATH_CALUDE_treble_double_plus_five_l2833_283397

theorem treble_double_plus_five (initial_number : ℕ) : initial_number = 15 → 
  3 * (2 * initial_number + 5) = 105 := by
  sorry

end NUMINAMATH_CALUDE_treble_double_plus_five_l2833_283397


namespace NUMINAMATH_CALUDE_age_difference_l2833_283324

-- Define variables for ages
variable (a b c d : ℕ)

-- Define the conditions
def condition1 : Prop := a + b = b + c + 15
def condition2 : Prop := a + d = c + d + 12
def condition3 : Prop := a = d + 3

-- Theorem statement
theorem age_difference (h1 : condition1 a b c) (h2 : condition2 a c d) (h3 : condition3 a d) :
  a - c = 12 := by sorry

end NUMINAMATH_CALUDE_age_difference_l2833_283324


namespace NUMINAMATH_CALUDE_cost_price_calculation_l2833_283396

/-- Proves that if an article is sold at 800 with a profit of 25%, then its cost price is 640. -/
theorem cost_price_calculation (selling_price : ℝ) (profit_percentage : ℝ) 
  (h1 : selling_price = 800)
  (h2 : profit_percentage = 25) :
  let cost_price := selling_price / (1 + profit_percentage / 100)
  cost_price = 640 := by sorry

end NUMINAMATH_CALUDE_cost_price_calculation_l2833_283396


namespace NUMINAMATH_CALUDE_ship_storm_problem_l2833_283379

/-- A problem about a ship's journey and a storm -/
theorem ship_storm_problem (initial_speed : ℝ) (initial_time : ℝ) 
  (h1 : initial_speed = 30)
  (h2 : initial_time = 20)
  (h3 : initial_speed * initial_time = (1/2) * (total_distance : ℝ))
  (h4 : distance_after_storm = (1/3) * total_distance) : 
  initial_speed * initial_time - distance_after_storm = 200 := by
  sorry

#check ship_storm_problem

end NUMINAMATH_CALUDE_ship_storm_problem_l2833_283379


namespace NUMINAMATH_CALUDE_probability_one_boy_one_girl_l2833_283343

/-- The probability of selecting one boy and one girl from a group of 3 boys and 2 girls, when choosing 2 students out of 5 -/
theorem probability_one_boy_one_girl :
  let total_students : ℕ := 5
  let num_boys : ℕ := 3
  let num_girls : ℕ := 2
  let students_to_select : ℕ := 2
  let total_combinations := Nat.choose total_students students_to_select
  let favorable_outcomes := Nat.choose num_boys 1 * Nat.choose num_girls 1
  (favorable_outcomes : ℚ) / total_combinations = 3 / 5 :=
by sorry

end NUMINAMATH_CALUDE_probability_one_boy_one_girl_l2833_283343


namespace NUMINAMATH_CALUDE_min_boxes_for_treat_bags_l2833_283313

/-- Represents the number of items in each box -/
structure BoxSizes where
  chocolate : Nat
  mint : Nat
  caramel : Nat

/-- Represents the number of boxes of each item -/
structure Boxes where
  chocolate : Nat
  mint : Nat
  caramel : Nat

/-- Calculates the total number of boxes -/
def totalBoxes (b : Boxes) : Nat :=
  b.chocolate + b.mint + b.caramel

/-- Checks if the given number of boxes results in complete treat bags with no leftovers -/
def isValidDistribution (sizes : BoxSizes) (boxes : Boxes) : Prop :=
  sizes.chocolate * boxes.chocolate = sizes.mint * boxes.mint ∧
  sizes.chocolate * boxes.chocolate = sizes.caramel * boxes.caramel

/-- The main theorem stating the minimum number of boxes needed -/
theorem min_boxes_for_treat_bags : ∃ (boxes : Boxes),
  let sizes : BoxSizes := ⟨50, 40, 25⟩
  isValidDistribution sizes boxes ∧ 
  totalBoxes boxes = 17 ∧
  (∀ (other : Boxes), isValidDistribution sizes other → totalBoxes other ≥ totalBoxes boxes) := by
  sorry

end NUMINAMATH_CALUDE_min_boxes_for_treat_bags_l2833_283313


namespace NUMINAMATH_CALUDE_first_part_second_part_l2833_283356

-- Define the function f
def f (a x : ℝ) : ℝ := x^2 + 2*a*x - 3

-- Theorem for the first part of the problem
theorem first_part (a : ℝ) : f a (a + 1) - f a a = 9 → a = 2 := by
  sorry

-- Theorem for the second part of the problem
theorem second_part (a : ℝ) : 
  (∀ x, f a x ≥ -4) ∧ (∃ x, f a x = -4) ↔ (a = 1 ∨ a = -1) := by
  sorry

end NUMINAMATH_CALUDE_first_part_second_part_l2833_283356


namespace NUMINAMATH_CALUDE_triangle_angle_not_all_greater_60_l2833_283377

theorem triangle_angle_not_all_greater_60 :
  ∀ (a b c : ℝ), 
  (a > 0 ∧ b > 0 ∧ c > 0) →  -- Angles are positive
  (a + b + c = 180) →        -- Sum of angles in a triangle is 180°
  ¬(a > 60 ∧ b > 60 ∧ c > 60) :=
by
  sorry

end NUMINAMATH_CALUDE_triangle_angle_not_all_greater_60_l2833_283377


namespace NUMINAMATH_CALUDE_remaining_flight_time_l2833_283304

/-- Calculates the remaining time on a flight given the total flight duration and activity durations. -/
theorem remaining_flight_time (total_duration activity1 activity2 activity3 : ℕ) :
  total_duration = 360 ∧ 
  activity1 = 90 ∧ 
  activity2 = 40 ∧ 
  activity3 = 120 →
  total_duration - (activity1 + activity2 + activity3) = 110 := by
  sorry

#check remaining_flight_time

end NUMINAMATH_CALUDE_remaining_flight_time_l2833_283304


namespace NUMINAMATH_CALUDE_no_ab_term_when_m_is_neg_six_l2833_283394

-- Define the polynomial as a function of a, b, and m
def polynomial (a b m : ℝ) : ℝ := 3 * (a^2 - 2*a*b - b^2) - (a^2 + m*a*b + 2*b^2)

-- Theorem stating that the polynomial has no ab term when m = -6
theorem no_ab_term_when_m_is_neg_six :
  ∀ a b : ℝ, (∀ m : ℝ, polynomial a b m = 2*a^2 - (6+m)*a*b - 5*b^2) →
  (∃! m : ℝ, ∀ a b : ℝ, polynomial a b m = 2*a^2 - 5*b^2) →
  (∃ m : ℝ, m = -6 ∧ ∀ a b : ℝ, polynomial a b m = 2*a^2 - 5*b^2) :=
by sorry

end NUMINAMATH_CALUDE_no_ab_term_when_m_is_neg_six_l2833_283394


namespace NUMINAMATH_CALUDE_nephews_count_l2833_283371

/-- The number of nephews Alden had 10 years ago -/
def alden_nephews_10_years_ago : ℕ := 50

/-- The number of nephews Alden has now -/
def alden_nephews_now : ℕ := 2 * alden_nephews_10_years_ago

/-- The number of additional nephews Vihaan has compared to Alden -/
def vihaan_additional_nephews : ℕ := 60

/-- The number of nephews Vihaan has -/
def vihaan_nephews : ℕ := alden_nephews_now + vihaan_additional_nephews

/-- The total number of nephews Alden and Vihaan have together -/
def total_nephews : ℕ := alden_nephews_now + vihaan_nephews

theorem nephews_count : total_nephews = 260 := by
  sorry

end NUMINAMATH_CALUDE_nephews_count_l2833_283371


namespace NUMINAMATH_CALUDE_vowel_soup_combinations_l2833_283308

/-- The number of vowels available -/
def num_vowels : ℕ := 5

/-- The length of the words to be formed -/
def word_length : ℕ := 6

/-- The number of times each vowel appears in the bowl -/
def vowel_count : ℕ := 7

/-- The total number of six-letter words that can be formed -/
def total_combinations : ℕ := num_vowels ^ word_length

theorem vowel_soup_combinations :
  total_combinations = 15625 :=
sorry

end NUMINAMATH_CALUDE_vowel_soup_combinations_l2833_283308


namespace NUMINAMATH_CALUDE_leisure_park_ticket_cost_l2833_283393

/-- The cost of tickets for a family visit to a leisure park -/
theorem leisure_park_ticket_cost :
  ∀ (child_ticket : ℕ),
  child_ticket * 5 + (child_ticket + 8) * 2 + (child_ticket + 4) * 2 = 150 →
  child_ticket + 8 = 22 := by
  sorry

end NUMINAMATH_CALUDE_leisure_park_ticket_cost_l2833_283393


namespace NUMINAMATH_CALUDE_counterexample_exists_l2833_283360

theorem counterexample_exists : ∃ n : ℕ, Nat.Prime n ∧ ¬(Nat.Prime (n - 2)) :=
sorry

end NUMINAMATH_CALUDE_counterexample_exists_l2833_283360


namespace NUMINAMATH_CALUDE_contrapositive_true_l2833_283388

theorem contrapositive_true : 
  (∀ a : ℝ, a > 2 → a^2 > 4) ↔ (∀ a : ℝ, a ≤ 2 → a^2 ≤ 4) :=
by sorry

end NUMINAMATH_CALUDE_contrapositive_true_l2833_283388


namespace NUMINAMATH_CALUDE_extreme_value_at_negative_three_l2833_283347

/-- The function f(x) -/
def f (a : ℝ) (x : ℝ) : ℝ := x^3 + a*x + 3*x - 9

/-- The derivative of f(x) -/
def f' (a : ℝ) (x : ℝ) : ℝ := 3*x^2 + a + 3

theorem extreme_value_at_negative_three (a : ℝ) :
  (∃ (x : ℝ), f' a x = 0) ∧ f' a (-3) = 0 → a = 5 := by sorry

end NUMINAMATH_CALUDE_extreme_value_at_negative_three_l2833_283347


namespace NUMINAMATH_CALUDE_yellow_peaches_undetermined_l2833_283392

def basket_peaches (red green yellow : ℕ) : Prop :=
  red = 7 ∧ green = 8 ∧ green = red + 1

theorem yellow_peaches_undetermined :
  ∀ (red green yellow : ℕ),
    basket_peaches red green yellow →
    ¬∃ (n : ℕ), ∀ (y : ℕ), basket_peaches red green y → y = n :=
by
  sorry

end NUMINAMATH_CALUDE_yellow_peaches_undetermined_l2833_283392


namespace NUMINAMATH_CALUDE_cool_drink_solution_l2833_283314

/-- Proves that 12 liters of water were added to achieve the given conditions -/
theorem cool_drink_solution (initial_volume : ℝ) (initial_concentration : ℝ) 
  (jasmine_added : ℝ) (final_concentration : ℝ) :
  initial_volume = 80 →
  initial_concentration = 0.1 →
  jasmine_added = 8 →
  final_concentration = 0.16 →
  ∃ (water_added : ℝ),
    water_added = 12 ∧
    (initial_volume * initial_concentration + jasmine_added) / 
    (initial_volume + jasmine_added + water_added) = final_concentration :=
by
  sorry


end NUMINAMATH_CALUDE_cool_drink_solution_l2833_283314


namespace NUMINAMATH_CALUDE_problem_solution_l2833_283367

theorem problem_solution (x : ℚ) : 
  2 + 1 / (1 + 1 / (2 + 2 / (3 + x))) = 144 / 53 → x = 3 / 4 := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l2833_283367


namespace NUMINAMATH_CALUDE_special_function_at_1001_l2833_283361

/-- A function satisfying the given conditions -/
def special_function (f : ℝ → ℝ) : Prop :=
  (∀ x > 0, f x > 0) ∧
  (∀ x y, x > y ∧ y > 0 → f (x - y) = Real.sqrt (f (x * y) + 3))

/-- The main theorem stating that f(1001) = 3 for any function satisfying the conditions -/
theorem special_function_at_1001 (f : ℝ → ℝ) (h : special_function f) : f 1001 = 3 := by
  sorry

end NUMINAMATH_CALUDE_special_function_at_1001_l2833_283361


namespace NUMINAMATH_CALUDE_S_infinite_l2833_283373

/-- The expression 2^(n^3+1) - 3^(n^2+1) + 5^(n+1) for positive integer n -/
def f (n : ℕ+) : ℤ := 2^(n.val^3+1) - 3^(n.val^2+1) + 5^(n.val+1)

/-- The set of prime numbers that divide f(n) for some positive integer n -/
def S : Set ℕ := {p : ℕ | Nat.Prime p ∧ ∃ (n : ℕ+), ∃ (k : ℤ), f n = k * p}

/-- Theorem stating that the set S is infinite -/
theorem S_infinite : Set.Infinite S := by sorry

end NUMINAMATH_CALUDE_S_infinite_l2833_283373


namespace NUMINAMATH_CALUDE_circumcircle_radius_isosceles_triangle_l2833_283332

/-- Given a triangle with two sides of length a and a third side of length b,
    the radius of its circumcircle is a²/√(4a² - b²) -/
theorem circumcircle_radius_isosceles_triangle (a b : ℝ) (ha : a > 0) (hb : b > 0) (hc : b < 2*a) :
  ∃ R : ℝ, R = a^2 / Real.sqrt (4*a^2 - b^2) ∧ 
  R > 0 ∧ 
  R * Real.sqrt (4*a^2 - b^2) = a^2 :=
sorry

end NUMINAMATH_CALUDE_circumcircle_radius_isosceles_triangle_l2833_283332


namespace NUMINAMATH_CALUDE_circle_area_above_line_is_zero_l2833_283362

/-- The circle equation: x^2 - 8x + y^2 - 10y + 29 = 0 -/
def circle_equation (x y : ℝ) : Prop :=
  x^2 - 8*x + y^2 - 10*y + 29 = 0

/-- The line equation: y = x - 2 -/
def line_equation (x y : ℝ) : Prop :=
  y = x - 2

/-- The area of the circle above the line -/
def area_above_line (circle : (ℝ × ℝ) → Prop) (line : (ℝ × ℝ) → Prop) : ℝ :=
  sorry -- Definition of area calculation

theorem circle_area_above_line_is_zero :
  area_above_line (λ (x, y) ↦ circle_equation x y) (λ (x, y) ↦ line_equation x y) = 0 := by
  sorry

end NUMINAMATH_CALUDE_circle_area_above_line_is_zero_l2833_283362


namespace NUMINAMATH_CALUDE_max_quad_area_l2833_283320

-- Define the circle
def Circle (x y : ℝ) : Prop := x^2 + y^2 = 4

-- Define the lines
def Line1 (m x y : ℝ) : Prop := m*x - y + 1 = 0
def Line2 (m x y : ℝ) : Prop := x + m*y - m = 0

-- Define the quadrilateral area
def QuadArea (A B C D : ℝ × ℝ) : ℝ := sorry

-- Theorem statement
theorem max_quad_area :
  ∀ (m : ℝ) (A B C D : ℝ × ℝ),
    Circle A.1 A.2 ∧ Circle B.1 B.2 ∧ Circle C.1 C.2 ∧ Circle D.1 D.2 →
    Line1 m A.1 A.2 ∧ Line1 m C.1 C.2 →
    Line2 m B.1 B.2 ∧ Line2 m D.1 D.2 →
    QuadArea A B C D ≤ 7 :=
sorry

end NUMINAMATH_CALUDE_max_quad_area_l2833_283320


namespace NUMINAMATH_CALUDE_canada_trip_problem_l2833_283365

/-- Represents the exchange rate from US dollars to Canadian dollars -/
def exchange_rate : ℚ := 15 / 9

/-- Calculates the sum of digits of a natural number -/
def sum_of_digits (n : ℕ) : ℕ := sorry

/-- The main theorem -/
theorem canada_trip_problem (d : ℕ) :
  (exchange_rate * d - 120 = d) → 
  d = 180 ∧ sum_of_digits d = 9 := by
  sorry

end NUMINAMATH_CALUDE_canada_trip_problem_l2833_283365


namespace NUMINAMATH_CALUDE_each_student_receives_six_apples_l2833_283326

/-- The number of apples Anita has -/
def total_apples : ℕ := 360

/-- The number of students in Anita's class -/
def num_students : ℕ := 60

/-- The number of apples each student should receive -/
def apples_per_student : ℕ := total_apples / num_students

/-- Theorem stating that each student should receive 6 apples -/
theorem each_student_receives_six_apples : apples_per_student = 6 := by
  sorry

end NUMINAMATH_CALUDE_each_student_receives_six_apples_l2833_283326


namespace NUMINAMATH_CALUDE_largest_valid_number_l2833_283389

def is_valid_number (n : ℕ) : Prop :=
  ∃ (a b c d e f : ℕ),
    n = a * 100000 + b * 10000 + c * 1000 + d * 100 + e * 10 + f ∧
    a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ a ≠ e ∧ a ≠ f ∧
    b ≠ c ∧ b ≠ d ∧ b ≠ e ∧ b ≠ f ∧
    c ≠ d ∧ c ≠ e ∧ c ≠ f ∧
    d ≠ e ∧ d ≠ f ∧
    e ≠ f ∧
    (c = a + b ∨ c = a - b ∨ c = b - a) ∧
    (d = b + c ∨ d = b - c ∨ d = c - b) ∧
    (e = c + d ∨ e = c - d ∨ e = d - c) ∧
    0 ≤ a ∧ a ≤ 9 ∧
    0 ≤ b ∧ b ≤ 9 ∧
    0 ≤ c ∧ c ≤ 9 ∧
    0 ≤ d ∧ d ≤ 9 ∧
    0 ≤ e ∧ e ≤ 9 ∧
    0 ≤ f ∧ f ≤ 9

theorem largest_valid_number :
  ∀ n : ℕ, is_valid_number n → n ≤ 972538 :=
by sorry

end NUMINAMATH_CALUDE_largest_valid_number_l2833_283389


namespace NUMINAMATH_CALUDE_units_digit_periodicity_l2833_283399

theorem units_digit_periodicity (k : ℕ) : 
  (k * (k + 1) * (k + 2)) % 10 = ((k + 10) * (k + 11) * (k + 12)) % 10 := by
  sorry

end NUMINAMATH_CALUDE_units_digit_periodicity_l2833_283399


namespace NUMINAMATH_CALUDE_sum_interior_angles_heptagon_l2833_283336

/-- The sum of interior angles of a polygon with n sides -/
def sum_interior_angles (n : ℕ) : ℝ := (n - 2) * 180

/-- A heptagon is a polygon with 7 sides -/
def heptagon_sides : ℕ := 7

/-- Theorem: The sum of the interior angles of a heptagon is 900 degrees -/
theorem sum_interior_angles_heptagon :
  sum_interior_angles heptagon_sides = 900 := by
  sorry

end NUMINAMATH_CALUDE_sum_interior_angles_heptagon_l2833_283336


namespace NUMINAMATH_CALUDE_residue_mod_29_l2833_283345

theorem residue_mod_29 : ∃ k : ℤ, -1237 = 29 * k + 10 := by sorry

end NUMINAMATH_CALUDE_residue_mod_29_l2833_283345


namespace NUMINAMATH_CALUDE_clock_hand_positions_l2833_283346

/-- Represents a clock with hour and minute hands -/
structure Clock :=
  (hours : ℕ)
  (minutes : ℕ)

/-- The number of times the hour and minute hands coincide in 24 hours -/
def coincidences : ℕ := 22

/-- The number of times the hour and minute hands form a straight angle in 24 hours -/
def straight_angles : ℕ := 24

/-- The number of times the hour and minute hands form a right angle in 24 hours -/
def right_angles : ℕ := 48

/-- The number of full rotations the minute hand makes in 24 hours -/
def minute_rotations : ℕ := 24

/-- The number of full rotations the hour hand makes in 24 hours -/
def hour_rotations : ℕ := 2

theorem clock_hand_positions (c : Clock) :
  coincidences = 22 ∧
  straight_angles = 24 ∧
  right_angles = 48 :=
sorry

end NUMINAMATH_CALUDE_clock_hand_positions_l2833_283346


namespace NUMINAMATH_CALUDE_inverse_proportion_percentage_change_l2833_283375

/-- Proves the percentage decrease in b when a increases by q% for inversely proportional variables -/
theorem inverse_proportion_percentage_change 
  (a b : ℝ) (q : ℝ) (c : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : q > 0) :
  (a * b = c) →  -- inverse proportionality condition
  let a' := a * (1 + q / 100)  -- a increased by q%
  let b' := c / a'  -- new b value
  (b - b') / b * 100 = 100 * q / (100 + q) :=
by sorry

end NUMINAMATH_CALUDE_inverse_proportion_percentage_change_l2833_283375


namespace NUMINAMATH_CALUDE_relationship_significance_l2833_283339

/-- The critical value for a 2x2 contingency table at 0.05 significance level -/
def critical_value : ℝ := 3.841

/-- The observed K^2 value from a 2x2 contingency table -/
def observed_value : ℝ := 4.013

/-- The maximum probability of making a mistake -/
def max_error_probability : ℝ := 0.05

/-- Theorem stating the relationship between the observed value, critical value, and maximum error probability -/
theorem relationship_significance (h : observed_value > critical_value) :
  max_error_probability = 0.05 := by
  sorry

end NUMINAMATH_CALUDE_relationship_significance_l2833_283339


namespace NUMINAMATH_CALUDE_probability_one_head_in_three_tosses_l2833_283383

theorem probability_one_head_in_three_tosses :
  let n : ℕ := 3  -- number of tosses
  let k : ℕ := 1  -- number of heads we want
  let p : ℚ := 1/2  -- probability of heads on a single toss
  Nat.choose n k * p^k * (1-p)^(n-k) = 3/8 := by
sorry

end NUMINAMATH_CALUDE_probability_one_head_in_three_tosses_l2833_283383


namespace NUMINAMATH_CALUDE_optimal_triangle_height_l2833_283351

/-- Given two parallel lines with distance b between them, and a segment of length a on one of the lines,
    the sum of areas of two triangles formed by connecting a point on the line segment to a point on the other parallel line
    is minimized when the height of one triangle is b√2/2. -/
theorem optimal_triangle_height (a b : ℝ) (h : 0 < a ∧ 0 < b) :
  let height := b * Real.sqrt 2 / 2
  let area (h : ℝ) := a * h / 2 + a * (b - h) ^ 2 / (2 * h)
  ∀ h, 0 < h ∧ h < b → area height ≤ area h := by
sorry

end NUMINAMATH_CALUDE_optimal_triangle_height_l2833_283351


namespace NUMINAMATH_CALUDE_polynomial_divisibility_l2833_283305

theorem polynomial_divisibility (a b c d e : ℤ) :
  (∀ x : ℤ, (7 : ℤ) ∣ (a * x^4 + b * x^3 + c * x^2 + d * x + e)) →
  ((7 : ℤ) ∣ a) ∧ ((7 : ℤ) ∣ b) ∧ ((7 : ℤ) ∣ c) ∧ ((7 : ℤ) ∣ d) ∧ ((7 : ℤ) ∣ e) :=
by sorry

end NUMINAMATH_CALUDE_polynomial_divisibility_l2833_283305


namespace NUMINAMATH_CALUDE_average_after_12th_innings_l2833_283382

/-- Represents a batsman's performance -/
structure Batsman where
  innings : ℕ
  lastScore : ℕ
  averageIncrease : ℕ
  neverNotOut : Bool

/-- Calculates the average score after the latest innings -/
def calculateAverage (b : Batsman) : ℚ :=
  if b.innings = 0 then 0
  else (b.innings * (b.averageIncrease : ℚ) + b.lastScore) / b.innings

/-- Theorem stating the average after 12th innings -/
theorem average_after_12th_innings (b : Batsman) 
  (h1 : b.innings = 12)
  (h2 : b.lastScore = 55)
  (h3 : b.averageIncrease = 1)
  (h4 : b.neverNotOut = true) :
  calculateAverage b = 44 := by
  sorry


end NUMINAMATH_CALUDE_average_after_12th_innings_l2833_283382


namespace NUMINAMATH_CALUDE_jade_tower_solution_l2833_283317

/-- The number of Lego pieces in Jade's tower problem -/
def jade_tower_problem (width_per_level : ℕ) (num_levels : ℕ) (pieces_left : ℕ) : Prop :=
  width_per_level * num_levels + pieces_left = 100

/-- Theorem stating the solution to Jade's Lego tower problem -/
theorem jade_tower_solution : jade_tower_problem 7 11 23 := by
  sorry

end NUMINAMATH_CALUDE_jade_tower_solution_l2833_283317


namespace NUMINAMATH_CALUDE_blocks_per_friend_l2833_283322

theorem blocks_per_friend (total_blocks : ℕ) (num_friends : ℕ) (blocks_per_friend : ℕ) : 
  total_blocks = 28 → num_friends = 4 → blocks_per_friend = total_blocks / num_friends → blocks_per_friend = 7 := by
  sorry

end NUMINAMATH_CALUDE_blocks_per_friend_l2833_283322


namespace NUMINAMATH_CALUDE_five_points_in_unit_triangle_close_pair_l2833_283364

-- Define an equilateral triangle
structure EquilateralTriangle where
  side_length : ℝ
  is_positive : side_length > 0

-- Define a point in 2D space
structure Point where
  x : ℝ
  y : ℝ

-- Define a function to check if a point is inside the triangle
def is_inside_triangle (t : EquilateralTriangle) (p : Point) : Prop :=
  sorry -- Actual implementation would go here

-- Define a function to calculate distance between two points
def distance (p1 p2 : Point) : ℝ :=
  sorry -- Actual implementation would go here

-- Theorem statement
theorem five_points_in_unit_triangle_close_pair 
  (t : EquilateralTriangle) 
  (h_side : t.side_length = 1) 
  (points : Fin 5 → Point) 
  (h_inside : ∀ i, is_inside_triangle t (points i)) :
  ∃ i j, i ≠ j ∧ distance (points i) (points j) < 0.5 :=
sorry

end NUMINAMATH_CALUDE_five_points_in_unit_triangle_close_pair_l2833_283364


namespace NUMINAMATH_CALUDE_multiple_of_p_l2833_283321

theorem multiple_of_p (p q : ℚ) (k : ℚ) : 
  p / q = 3 / 5 → kp + q = 11 → k = 2 := by sorry

end NUMINAMATH_CALUDE_multiple_of_p_l2833_283321


namespace NUMINAMATH_CALUDE_running_preference_related_to_gender_certainty_running_preference_related_to_gender_l2833_283350

/-- Represents the data from the survey about running preferences among university students. -/
structure RunningPreferenceSurvey where
  total_students : ℕ
  boys : ℕ
  boys_not_liking : ℕ
  girls_liking : ℕ

/-- Calculates the chi-square value for the given survey data. -/
def calculate_chi_square (survey : RunningPreferenceSurvey) : ℚ :=
  let girls := survey.total_students - survey.boys
  let boys_liking := survey.boys - survey.boys_not_liking
  let girls_not_liking := girls - survey.girls_liking
  let n := survey.total_students
  let a := boys_liking
  let b := survey.boys_not_liking
  let c := survey.girls_liking
  let d := girls_not_liking
  (n * (a * d - b * c)^2 : ℚ) / ((a + b) * (c + d) * (a + c) * (b + d))

/-- Theorem stating that the chi-square value for the given survey data is greater than 6.635,
    indicating a 99% certainty that liking running is related to gender. -/
theorem running_preference_related_to_gender (survey : RunningPreferenceSurvey)
  (h1 : survey.total_students = 200)
  (h2 : survey.boys = 120)
  (h3 : survey.boys_not_liking = 50)
  (h4 : survey.girls_liking = 30) :
  calculate_chi_square survey > 6635 / 1000 :=
sorry

/-- Corollary stating that there is a 99% certainty that liking running is related to gender. -/
theorem certainty_running_preference_related_to_gender (survey : RunningPreferenceSurvey)
  (h1 : survey.total_students = 200)
  (h2 : survey.boys = 120)
  (h3 : survey.boys_not_liking = 50)
  (h4 : survey.girls_liking = 30) :
  ∃ (certainty : ℚ), certainty = 99 / 100 ∧
  calculate_chi_square survey > 6635 / 1000 :=
sorry

end NUMINAMATH_CALUDE_running_preference_related_to_gender_certainty_running_preference_related_to_gender_l2833_283350


namespace NUMINAMATH_CALUDE_solution_set_abs_inequality_l2833_283391

theorem solution_set_abs_inequality :
  {x : ℝ | |2 - x| < 5} = {x : ℝ | -3 < x ∧ x < 7} := by sorry

end NUMINAMATH_CALUDE_solution_set_abs_inequality_l2833_283391


namespace NUMINAMATH_CALUDE_tape_length_calculation_l2833_283374

/-- Calculate the total length of overlapping tape sheets -/
def totalTapeLength (sheetLength : ℝ) (overlap : ℝ) (numSheets : ℕ) : ℝ :=
  sheetLength + (numSheets - 1 : ℝ) * (sheetLength - overlap)

/-- Theorem: The total length of 64 sheets of tape, each 25 cm long, 
    with a 3 cm overlap between consecutive sheets, is 1411 cm -/
theorem tape_length_calculation :
  totalTapeLength 25 3 64 = 1411 := by
  sorry

end NUMINAMATH_CALUDE_tape_length_calculation_l2833_283374


namespace NUMINAMATH_CALUDE_monthly_growth_rate_correct_max_daily_tourists_may_correct_l2833_283390

-- Define the number of tourists in February and April
def tourists_february : ℝ := 16000
def tourists_april : ℝ := 25000

-- Define the number of tourists from May 1st to May 21st
def tourists_may_21 : ℝ := 21250

-- Define the monthly average growth rate
def monthly_growth_rate : ℝ := 0.25

-- Define the function to calculate the growth over two months
def two_month_growth (initial : ℝ) (rate : ℝ) : ℝ :=
  initial * (1 + rate) ^ 2

-- Define the function to calculate the maximum number of tourists in May
def max_tourists_may (rate : ℝ) : ℝ :=
  tourists_april * (1 + rate)

-- Theorem 1: Prove the monthly average growth rate
theorem monthly_growth_rate_correct :
  two_month_growth tourists_february monthly_growth_rate = tourists_april :=
sorry

-- Theorem 2: Prove the maximum average number of tourists per day in the last 10 days of May
theorem max_daily_tourists_may_correct :
  (max_tourists_may monthly_growth_rate - tourists_may_21) / 10 = 100000 :=
sorry

end NUMINAMATH_CALUDE_monthly_growth_rate_correct_max_daily_tourists_may_correct_l2833_283390


namespace NUMINAMATH_CALUDE_pancake_fundraiser_l2833_283335

/-- The civic league's pancake breakfast fundraiser problem -/
theorem pancake_fundraiser
  (pancake_price : ℝ)
  (bacon_price : ℝ)
  (pancake_stacks : ℕ)
  (bacon_slices : ℕ)
  (h1 : pancake_price = 4)
  (h2 : bacon_price = 2)
  (h3 : pancake_stacks = 60)
  (h4 : bacon_slices = 90) :
  pancake_price * (pancake_stacks : ℝ) + bacon_price * (bacon_slices : ℝ) = 420 := by
  sorry

end NUMINAMATH_CALUDE_pancake_fundraiser_l2833_283335


namespace NUMINAMATH_CALUDE_characterization_of_representable_numbers_l2833_283363

/-- Two natural numbers are relatively prime if their greatest common divisor is 1 -/
def RelativelyPrime (a b : ℕ) : Prop := Nat.gcd a b = 1

/-- A natural number k can be represented as the sum of two relatively prime numbers greater than 1 -/
def RepresentableAsSumOfRelativelyPrime (k : ℕ) : Prop :=
  ∃ a b : ℕ, a > 1 ∧ b > 1 ∧ RelativelyPrime a b ∧ a + b = k

/-- Theorem stating the characterization of numbers representable as sum of two relatively prime numbers greater than 1 -/
theorem characterization_of_representable_numbers :
  ∀ k : ℕ, RepresentableAsSumOfRelativelyPrime k ↔ k = 5 ∨ k ≥ 7 :=
sorry

end NUMINAMATH_CALUDE_characterization_of_representable_numbers_l2833_283363


namespace NUMINAMATH_CALUDE_mistaken_division_multiplication_l2833_283300

/-- Given a number x and another number n, where x is mistakenly divided by n instead of being multiplied,
    and the percentage error in the result is 99%, prove that n = 10. -/
theorem mistaken_division_multiplication (x : ℝ) (n : ℝ) (h : x ≠ 0) :
  (x / n) / (x * n) = 1 / 100 → n = 10 := by
  sorry

end NUMINAMATH_CALUDE_mistaken_division_multiplication_l2833_283300


namespace NUMINAMATH_CALUDE_matthew_crackers_l2833_283341

theorem matthew_crackers (total_crackers : ℕ) (crackers_per_friend : ℕ) (num_friends : ℕ) :
  total_crackers = 8 →
  crackers_per_friend = 2 →
  total_crackers = num_friends * crackers_per_friend →
  num_friends = 4 := by
sorry

end NUMINAMATH_CALUDE_matthew_crackers_l2833_283341


namespace NUMINAMATH_CALUDE_metal_detector_time_busier_days_is_30_l2833_283387

/-- Represents the time Mark spends on courthouse activities in a week -/
structure CourthouseTime where
  totalWeeklyTime : ℕ
  parkingTime : ℕ
  walkingTime : ℕ
  workDays : ℕ
  lessCrowdedDays : ℕ
  metalDetectorTimeLessCrowded : ℕ

/-- Calculates the time spent on metal detector on busier days -/
def metalDetectorTimeBusierDays (ct : CourthouseTime) : ℕ :=
  let totalParkingWalkingTime := ct.workDays * (ct.parkingTime + ct.walkingTime)
  let totalMetalDetectorTime := ct.totalWeeklyTime - totalParkingWalkingTime
  let metalDetectorTimeLessCrowdedTotal := ct.lessCrowdedDays * ct.metalDetectorTimeLessCrowded
  let metalDetectorTimeBusierTotal := totalMetalDetectorTime - metalDetectorTimeLessCrowdedTotal
  metalDetectorTimeBusierTotal / (ct.workDays - ct.lessCrowdedDays)

theorem metal_detector_time_busier_days_is_30 (ct : CourthouseTime) :
  ct.totalWeeklyTime = 130 ∧
  ct.parkingTime = 5 ∧
  ct.walkingTime = 3 ∧
  ct.workDays = 5 ∧
  ct.lessCrowdedDays = 3 ∧
  ct.metalDetectorTimeLessCrowded = 10 →
  metalDetectorTimeBusierDays ct = 30 := by
  sorry


end NUMINAMATH_CALUDE_metal_detector_time_busier_days_is_30_l2833_283387


namespace NUMINAMATH_CALUDE_inequality_proof_l2833_283329

theorem inequality_proof (α : ℝ) (m : ℕ) (a b : ℝ) 
  (h1 : 0 < α) (h2 : α < π / 2) (h3 : m ≥ 1) (h4 : 0 < a) (h5 : 0 < b) : 
  a / (Real.cos α)^m + b / (Real.sin α)^m ≥ (a^(2/(m+2)) + b^(2/(m+2)))^((m+2)/2) := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l2833_283329


namespace NUMINAMATH_CALUDE_quadratic_intersection_l2833_283328

/-- Represents a quadratic function of the form y = x^2 + px + q -/
structure QuadraticFunction where
  p : ℝ
  q : ℝ
  h : -2 * p + q = 2023

/-- The x-coordinate of the intersection point -/
def intersection_x : ℝ := -2

/-- The y-coordinate of the intersection point -/
def intersection_y : ℝ := 2027

/-- Theorem stating that all quadratic functions satisfying the condition intersect at a single point -/
theorem quadratic_intersection (f : QuadraticFunction) : 
  (intersection_x^2 + f.p * intersection_x + f.q) = intersection_y := by
  sorry

end NUMINAMATH_CALUDE_quadratic_intersection_l2833_283328


namespace NUMINAMATH_CALUDE_art_show_earnings_l2833_283385

def extra_large_price : ℕ := 150
def large_price : ℕ := 100
def medium_price : ℕ := 80
def small_price : ℕ := 60

def extra_large_sold : ℕ := 3
def large_sold : ℕ := 5
def medium_sold : ℕ := 8
def small_sold : ℕ := 10

def large_discount : ℚ := 0.1
def sales_tax : ℚ := 0.05

def total_earnings : ℚ := 2247

theorem art_show_earnings :
  let extra_large_total := extra_large_price * extra_large_sold
  let large_total := large_price * large_sold * (1 - large_discount)
  let medium_total := medium_price * medium_sold
  let small_total := small_price * small_sold
  let subtotal := extra_large_total + large_total + medium_total + small_total
  let tax := subtotal * sales_tax
  (subtotal + tax : ℚ) = total_earnings := by
sorry

end NUMINAMATH_CALUDE_art_show_earnings_l2833_283385


namespace NUMINAMATH_CALUDE_max_value_theorem_l2833_283334

theorem max_value_theorem (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : 1/x + 2/y = 1) :
  2*x + y ≤ 8 ∧ ∃ (x₀ y₀ : ℝ), x₀ > 0 ∧ y₀ > 0 ∧ 1/x₀ + 2/y₀ = 1 ∧ 2*x₀ + y₀ = 8 :=
sorry

end NUMINAMATH_CALUDE_max_value_theorem_l2833_283334


namespace NUMINAMATH_CALUDE_marble_fraction_after_tripling_l2833_283301

theorem marble_fraction_after_tripling (total : ℚ) (h1 : total > 0) : 
  let blue := (2/3) * total
  let red := total - blue
  let new_red := 3 * red
  let new_total := blue + new_red
  new_red / new_total = 3/5 := by
sorry

end NUMINAMATH_CALUDE_marble_fraction_after_tripling_l2833_283301


namespace NUMINAMATH_CALUDE_average_first_21_multiples_of_5_l2833_283316

theorem average_first_21_multiples_of_5 : 
  let n : ℕ := 21
  let multiples : List ℕ := List.range n |>.map (fun i => (i + 1) * 5)
  (multiples.sum : ℚ) / n = 55 := by
  sorry

end NUMINAMATH_CALUDE_average_first_21_multiples_of_5_l2833_283316


namespace NUMINAMATH_CALUDE_probability_select_A_l2833_283342

/-- The probability of selecting a specific person when choosing 2 from 5 -/
def probability_select_person (total : ℕ) (choose : ℕ) : ℚ :=
  (total - 1).choose (choose - 1) / total.choose choose

/-- The group size -/
def group_size : ℕ := 5

/-- The number of people to choose -/
def choose_size : ℕ := 2

theorem probability_select_A :
  probability_select_person group_size choose_size = 2/5 := by
  sorry

end NUMINAMATH_CALUDE_probability_select_A_l2833_283342


namespace NUMINAMATH_CALUDE_brads_running_speed_l2833_283378

/-- Proves that Brad's running speed is 6 km/h given the conditions of the problem -/
theorem brads_running_speed (maxwell_speed : ℝ) (total_distance : ℝ) (maxwell_time : ℝ) 
  (h1 : maxwell_speed = 4)
  (h2 : total_distance = 94)
  (h3 : maxwell_time = 10) : 
  let brad_time := maxwell_time - 1
  let maxwell_distance := maxwell_speed * maxwell_time
  let brad_distance := total_distance - maxwell_distance
  brad_distance / brad_time = 6 := by
  sorry

end NUMINAMATH_CALUDE_brads_running_speed_l2833_283378


namespace NUMINAMATH_CALUDE_painted_cells_theorem_l2833_283355

/-- Represents a rectangular grid with painted cells -/
structure PaintedGrid where
  rows : ℕ
  cols : ℕ
  painted_cells : ℕ

/-- Calculates the number of painted cells in a grid with the given painting pattern -/
def calculate_painted_cells (k l : ℕ) : ℕ :=
  (2 * k + 1) * (2 * l + 1) - k * l

/-- Theorem stating the possible numbers of painted cells given the conditions -/
theorem painted_cells_theorem :
  ∀ (grid : PaintedGrid),
  (∃ (k l : ℕ), 
    grid.rows = 2 * k + 1 ∧ 
    grid.cols = 2 * l + 1 ∧ 
    k * l = 74) →
  grid.painted_cells = 373 ∨ grid.painted_cells = 301 :=
by sorry

end NUMINAMATH_CALUDE_painted_cells_theorem_l2833_283355


namespace NUMINAMATH_CALUDE_bus_driver_max_regular_hours_l2833_283338

/-- Proves that the maximum number of regular hours is 40 given the conditions --/
theorem bus_driver_max_regular_hours : 
  let regular_rate : ℚ := 16
  let overtime_rate : ℚ := regular_rate * (1 + 3/4)
  let total_compensation : ℚ := 1340
  let total_hours : ℕ := 65
  let max_regular_hours : ℕ := 40
  regular_rate * max_regular_hours + 
  overtime_rate * (total_hours - max_regular_hours) = total_compensation := by
sorry


end NUMINAMATH_CALUDE_bus_driver_max_regular_hours_l2833_283338


namespace NUMINAMATH_CALUDE_not_perfect_square_l2833_283302

theorem not_perfect_square : 
  ¬ ∃ n : ℕ, 5^2023 = n^2 ∧ 
  ∃ a : ℕ, 3^2021 = a^2 ∧
  ∃ b : ℕ, 7^2024 = b^2 ∧
  ∃ c : ℕ, 6^2025 = c^2 ∧
  ∃ d : ℕ, 8^2026 = d^2 :=
by sorry

end NUMINAMATH_CALUDE_not_perfect_square_l2833_283302


namespace NUMINAMATH_CALUDE_adidas_cost_l2833_283370

/-- The cost of Adidas shoes given the sales information -/
theorem adidas_cost (total_goal : ℝ) (nike_cost reebok_cost : ℝ) 
  (nike_sold adidas_sold reebok_sold : ℕ) (excess : ℝ) 
  (h1 : total_goal = 1000)
  (h2 : nike_cost = 60)
  (h3 : reebok_cost = 35)
  (h4 : nike_sold = 8)
  (h5 : adidas_sold = 6)
  (h6 : reebok_sold = 9)
  (h7 : excess = 65)
  : ∃ (adidas_cost : ℝ), 
    nike_cost * nike_sold + adidas_cost * adidas_sold + reebok_cost * reebok_sold 
    = total_goal + excess ∧ adidas_cost = 45 := by
  sorry

end NUMINAMATH_CALUDE_adidas_cost_l2833_283370


namespace NUMINAMATH_CALUDE_sqrt_equation_solution_l2833_283319

theorem sqrt_equation_solution (x : ℝ) :
  (Real.sqrt (4 * x + 6) / Real.sqrt (8 * x + 2) = 2 / Real.sqrt 3) → x = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_equation_solution_l2833_283319


namespace NUMINAMATH_CALUDE_bug_probability_l2833_283372

/-- The probability of the bug being at its starting vertex after n moves -/
def Q : ℕ → ℚ
  | 0 => 1
  | n + 1 => (1 / 3) * (1 - Q n)

/-- The problem statement -/
theorem bug_probability : Q 8 = 547 / 2187 := by
  sorry

end NUMINAMATH_CALUDE_bug_probability_l2833_283372


namespace NUMINAMATH_CALUDE_min_points_10th_game_l2833_283325

def points_6_to_9 : List ℕ := [18, 15, 16, 19]

def total_points_6_to_9 : ℕ := points_6_to_9.sum

def average_greater_after_9_than_5 (first_5_total : ℕ) : Prop :=
  (first_5_total + total_points_6_to_9) / 9 > first_5_total / 5

def first_5_not_exceed_85 (first_5_total : ℕ) : Prop :=
  first_5_total ≤ 85

theorem min_points_10th_game (first_5_total : ℕ) 
  (h1 : average_greater_after_9_than_5 first_5_total)
  (h2 : first_5_not_exceed_85 first_5_total) :
  ∃ (points_10th : ℕ), 
    (first_5_total + total_points_6_to_9 + points_10th) / 10 > 17 ∧
    ∀ (x : ℕ), x < points_10th → 
      (first_5_total + total_points_6_to_9 + x) / 10 ≤ 17 :=
by sorry

end NUMINAMATH_CALUDE_min_points_10th_game_l2833_283325


namespace NUMINAMATH_CALUDE_book_chapters_l2833_283376

/-- The number of chapters in a book, given the number of chapters read per day and the number of days taken to finish the book. -/
def total_chapters (chapters_per_day : ℕ) (days_to_finish : ℕ) : ℕ :=
  chapters_per_day * days_to_finish

/-- Theorem stating that the total number of chapters in the book is 220,448. -/
theorem book_chapters :
  total_chapters 332 664 = 220448 := by
  sorry

end NUMINAMATH_CALUDE_book_chapters_l2833_283376


namespace NUMINAMATH_CALUDE_string_cheese_calculation_l2833_283358

/-- The number of string cheeses in each package for Kelly's kids' lunches. -/
def string_cheeses_per_package : ℕ := by sorry

theorem string_cheese_calculation (days_per_week : ℕ) (oldest_daily : ℕ) (youngest_daily : ℕ) 
  (weeks : ℕ) (packages : ℕ) (h1 : days_per_week = 5) (h2 : oldest_daily = 2) 
  (h3 : youngest_daily = 1) (h4 : weeks = 4) (h5 : packages = 2) : 
  string_cheeses_per_package = 30 := by sorry

end NUMINAMATH_CALUDE_string_cheese_calculation_l2833_283358


namespace NUMINAMATH_CALUDE_arithmetic_mean_value_l2833_283311

/-- A normal distribution with given properties -/
structure NormalDistribution where
  σ : ℝ  -- standard deviation
  x : ℝ  -- value 2 standard deviations below the mean
  h : x = μ - 2 * σ  -- relation between x, μ, and σ

/-- The arithmetic mean of a normal distribution satisfying given conditions -/
def arithmetic_mean (d : NormalDistribution) : ℝ := 
  d.x + 2 * d.σ

/-- Theorem stating the arithmetic mean of the specific normal distribution -/
theorem arithmetic_mean_value (d : NormalDistribution) 
  (h_σ : d.σ = 1.5) (h_x : d.x = 11) : arithmetic_mean d = 14 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_mean_value_l2833_283311


namespace NUMINAMATH_CALUDE_solve_inequality_system_simplify_expression_l2833_283312

-- Problem 1
theorem solve_inequality_system (x : ℝ) :
  (10 - 3 * x < -5 ∧ x / 3 ≥ 4 - (x - 2) / 2) ↔ x ≥ 6 := by sorry

-- Problem 2
theorem simplify_expression (a : ℝ) (h : a ≠ 0 ∧ a ≠ 1 ∧ a ≠ -1) :
  2 / (a + 1) - (a - 2) / (a^2 - 1) / (a * (a - 2) / (a^2 - 2*a + 1)) = 1 / a := by sorry

end NUMINAMATH_CALUDE_solve_inequality_system_simplify_expression_l2833_283312


namespace NUMINAMATH_CALUDE_rectangle_area_l2833_283318

theorem rectangle_area (short_side : ℝ) (perimeter : ℝ) 
  (h1 : short_side = 11) 
  (h2 : perimeter = 52) : 
  short_side * (perimeter / 2 - short_side) = 165 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_area_l2833_283318


namespace NUMINAMATH_CALUDE_max_gcd_value_l2833_283306

theorem max_gcd_value (m : ℕ+) : 
  (Nat.gcd (15 * m.val + 4) (14 * m.val + 3) ≤ 11) ∧ 
  (∃ m : ℕ+, Nat.gcd (15 * m.val + 4) (14 * m.val + 3) = 11) := by
  sorry

end NUMINAMATH_CALUDE_max_gcd_value_l2833_283306


namespace NUMINAMATH_CALUDE_union_of_A_and_B_l2833_283340

def A : Set ℕ := {1, 2, 3, 5, 7}
def B : Set ℕ := {3, 4, 5}

theorem union_of_A_and_B : A ∪ B = {1, 2, 3, 4, 5, 7} := by sorry

end NUMINAMATH_CALUDE_union_of_A_and_B_l2833_283340

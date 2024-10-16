import Mathlib

namespace NUMINAMATH_CALUDE_modern_pentathlon_theorem_l559_55935

/-- Represents a competitor in the Modern Pentathlon --/
inductive Competitor
| A
| B
| C

/-- Represents an event in the Modern Pentathlon --/
inductive Event
| Shooting
| Fencing
| Swimming
| Equestrian
| CrossCountryRunning

/-- Represents the place a competitor finished in an event --/
inductive Place
| First
| Second
| Third

/-- The scoring system for the Modern Pentathlon --/
structure ScoringSystem where
  a : ℕ+
  b : ℕ+
  c : ℕ+
  a_gt_b : a > b
  b_gt_c : b > c

/-- The results of the Modern Pentathlon --/
def ModernPentathlonResults (s : ScoringSystem) :=
  Competitor → Event → Place

/-- Calculate the total score for a competitor given the results --/
def totalScore (s : ScoringSystem) (results : ModernPentathlonResults s) (competitor : Competitor) : ℕ :=
  sorry

/-- The theorem to be proved --/
theorem modern_pentathlon_theorem (s : ScoringSystem) 
  (results : ModernPentathlonResults s)
  (total_A : totalScore s results Competitor.A = 22)
  (total_B : totalScore s results Competitor.B = 9)
  (total_C : totalScore s results Competitor.C = 9)
  (B_first_equestrian : results Competitor.B Event.Equestrian = Place.First) :
  (∃ r : ModernPentathlonResults s, 
    (totalScore s r Competitor.A = 22) ∧ 
    (totalScore s r Competitor.B = 9) ∧ 
    (totalScore s r Competitor.C = 9) ∧
    (r Competitor.B Event.Equestrian = Place.First) ∧
    (r Competitor.B Event.Swimming = Place.Third)) ∧
  (∃ r : ModernPentathlonResults s, 
    (totalScore s r Competitor.A = 22) ∧ 
    (totalScore s r Competitor.B = 9) ∧ 
    (totalScore s r Competitor.C = 9) ∧
    (r Competitor.B Event.Equestrian = Place.First) ∧
    (r Competitor.C Event.Swimming = Place.Third)) :=
  sorry

end NUMINAMATH_CALUDE_modern_pentathlon_theorem_l559_55935


namespace NUMINAMATH_CALUDE_shepherd_problem_l559_55963

theorem shepherd_problem :
  ∃! (a b c : ℕ), 
    a + b + 10 * c = 100 ∧
    20 * a + 10 * b + 10 * c = 200 ∧
    a = 1 ∧ b = 9 ∧ c = 9 := by
  sorry

end NUMINAMATH_CALUDE_shepherd_problem_l559_55963


namespace NUMINAMATH_CALUDE_fraction_is_positive_integer_l559_55934

theorem fraction_is_positive_integer (q : ℕ+) :
  (∃ k : ℕ+, (5 * q + 40 : ℚ) / (3 * q - 8 : ℚ) = k) ↔ 3 ≤ q ∧ q ≤ 28 := by
  sorry

end NUMINAMATH_CALUDE_fraction_is_positive_integer_l559_55934


namespace NUMINAMATH_CALUDE_optimal_profit_profit_function_correct_l559_55957

/-- Represents the daily profit function for a factory -/
def daily_profit (x : ℝ) : ℝ := -50 * x^2 + 400 * x + 9000

/-- Represents the optimal price reduction -/
def optimal_reduction : ℝ := 4

/-- Represents the maximum daily profit -/
def max_profit : ℝ := 9800

/-- Theorem stating the optimal price reduction and maximum profit -/
theorem optimal_profit :
  (∀ x : ℝ, daily_profit x ≤ daily_profit optimal_reduction) ∧
  daily_profit optimal_reduction = max_profit := by
  sorry

/-- Theorem stating the correctness of the daily profit function -/
theorem profit_function_correct
  (cost_per_kg : ℝ)
  (initial_price : ℝ)
  (initial_sales : ℝ)
  (sales_increase_rate : ℝ)
  (h1 : cost_per_kg = 30)
  (h2 : initial_price = 48)
  (h3 : initial_sales = 500)
  (h4 : sales_increase_rate = 50) :
  ∀ x : ℝ, daily_profit x =
    (initial_price - x - cost_per_kg) * (initial_sales + sales_increase_rate * x) := by
  sorry

end NUMINAMATH_CALUDE_optimal_profit_profit_function_correct_l559_55957


namespace NUMINAMATH_CALUDE_inequality_solution_set_l559_55962

theorem inequality_solution_set (a : ℝ) (x₁ x₂ : ℝ) : 
  a > 0 →
  (∀ x, x^2 - a*x - 6*a^2 < 0 ↔ x₁ < x ∧ x < x₂) →
  x₂ - x₁ = 10 →
  a = 2 := by
sorry

end NUMINAMATH_CALUDE_inequality_solution_set_l559_55962


namespace NUMINAMATH_CALUDE_perfect_square_conditions_l559_55973

theorem perfect_square_conditions (a b c d e f : ℝ) :
  (∃ (p q r : ℝ), ∀ (x y z : ℝ),
    a * x^2 + b * y^2 + c * z^2 + 2 * d * x * y + 2 * e * y * z + 2 * f * z * x = (p * x + q * y + r * z)^2)
  ↔
  (a * b = d^2 ∧ b * c = e^2 ∧ c * a = f^2 ∧ a * e = d * f ∧ b * f = d * e ∧ c * d = e * f) :=
by sorry

end NUMINAMATH_CALUDE_perfect_square_conditions_l559_55973


namespace NUMINAMATH_CALUDE_simplify_expression_l559_55982

theorem simplify_expression : 1 - 1 / (1 + Real.sqrt 5) + 1 / (1 - Real.sqrt 5) = 1 := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l559_55982


namespace NUMINAMATH_CALUDE_star_op_specific_value_l559_55926

-- Define the * operation for non-zero integers
def star_op (a b : ℤ) : ℚ := 1 / a + 1 / b

-- Theorem statement
theorem star_op_specific_value :
  ∀ a b : ℕ+, 
  (a : ℤ) + (b : ℤ) = 15 → 
  (a : ℤ) * (b : ℤ) = 36 → 
  star_op a b = 5 / 12 := by
  sorry

end NUMINAMATH_CALUDE_star_op_specific_value_l559_55926


namespace NUMINAMATH_CALUDE_carol_cupcakes_theorem_l559_55960

/-- Calculates the number of cupcakes made after selling the first batch -/
def cupcakes_made_after (initial : ℕ) (sold : ℕ) (final_total : ℕ) : ℕ :=
  final_total - (initial - sold)

/-- Proves that Carol made 28 cupcakes after selling the first batch -/
theorem carol_cupcakes_theorem (initial : ℕ) (sold : ℕ) (final_total : ℕ)
    (h1 : initial = 30)
    (h2 : sold = 9)
    (h3 : final_total = 49) :
    cupcakes_made_after initial sold final_total = 28 := by
  sorry

end NUMINAMATH_CALUDE_carol_cupcakes_theorem_l559_55960


namespace NUMINAMATH_CALUDE_intersection_of_A_and_B_l559_55909

def A : Set ℝ := {x | 0 ≤ x ∧ x ≤ 3}
def B : Set ℝ := {x | 1 ≤ x ∧ x < 4}

theorem intersection_of_A_and_B :
  A ∩ B = {x : ℝ | 1 ≤ x ∧ x ≤ 3} := by sorry

end NUMINAMATH_CALUDE_intersection_of_A_and_B_l559_55909


namespace NUMINAMATH_CALUDE_estimate_value_l559_55990

theorem estimate_value : 6 < (2 * Real.sqrt 2 + Real.sqrt 3) * Real.sqrt 2 ∧ 
                         (2 * Real.sqrt 2 + Real.sqrt 3) * Real.sqrt 2 < 7 := by
  sorry

end NUMINAMATH_CALUDE_estimate_value_l559_55990


namespace NUMINAMATH_CALUDE_fountain_area_l559_55972

theorem fountain_area (AB DC : ℝ) (h1 : AB = 20) (h2 : DC = 12) : ∃ (area : ℝ), area = 244 * Real.pi := by
  sorry

end NUMINAMATH_CALUDE_fountain_area_l559_55972


namespace NUMINAMATH_CALUDE_car_rental_rates_l559_55993

/-- The daily rate of the first car rental company -/
def daily_rate : ℝ := 21.95

/-- The per-mile rate of the first car rental company -/
def first_company_per_mile : ℝ := 0.19

/-- The fixed rate of City Rentals -/
def city_rentals_fixed : ℝ := 18.95

/-- The per-mile rate of City Rentals -/
def city_rentals_per_mile : ℝ := 0.21

/-- The number of miles at which the costs are equal -/
def equal_cost_miles : ℝ := 150.0

theorem car_rental_rates :
  daily_rate + first_company_per_mile * equal_cost_miles =
  city_rentals_fixed + city_rentals_per_mile * equal_cost_miles :=
by sorry

end NUMINAMATH_CALUDE_car_rental_rates_l559_55993


namespace NUMINAMATH_CALUDE_family_reunion_attendance_l559_55943

/-- The number of male adults at the family reunion -/
def male_adults : ℕ := 100

/-- The number of female adults at the family reunion -/
def female_adults : ℕ := male_adults + 50

/-- The total number of adults at the family reunion -/
def total_adults : ℕ := male_adults + female_adults

/-- The number of children at the family reunion -/
def children : ℕ := 2 * total_adults

/-- The total number of attendees at the family reunion -/
def total_attendees : ℕ := total_adults + children

theorem family_reunion_attendance : 
  female_adults = male_adults + 50 ∧ 
  children = 2 * total_adults ∧ 
  total_attendees = 750 → 
  male_adults = 100 := by
  sorry

end NUMINAMATH_CALUDE_family_reunion_attendance_l559_55943


namespace NUMINAMATH_CALUDE_teacher_zhao_masks_l559_55921

theorem teacher_zhao_masks (n : ℕ) : 
  (n / 2 * 5 + n / 2 * 7 + 25 = n / 3 * 10 + 2 * n / 3 * 7 - 35) →
  (n / 2 * 5 + n / 2 * 7 + 25 = 205) := by
  sorry

end NUMINAMATH_CALUDE_teacher_zhao_masks_l559_55921


namespace NUMINAMATH_CALUDE_triangle_perimeter_l559_55915

theorem triangle_perimeter (a b c : ℕ) (α β γ : ℝ) : 
  a > 0 ∧ b = a + 1 ∧ c = b + 1 →  -- Consecutive positive integer sides
  α > 0 ∧ β > 0 ∧ γ > 0 →  -- Positive angles
  α + β + γ = π →  -- Sum of angles in a triangle
  max γ (max α β) = 2 * min γ (min α β) →  -- Largest angle is twice the smallest
  a + b + c = 15 :=  -- Perimeter is 15
by sorry

end NUMINAMATH_CALUDE_triangle_perimeter_l559_55915


namespace NUMINAMATH_CALUDE_min_value_and_range_l559_55912

theorem min_value_and_range (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a + b = 1) :
  (∃ (min : ℝ), min = 9 ∧ ∀ (a' b' : ℝ), a' > 0 → b' > 0 → a' + b' = 1 → 1 / a' + 4 / b' ≥ min) ∧
  (∀ (x : ℝ), (∀ (a' b' : ℝ), a' > 0 → b' > 0 → a' + b' = 1 → 1 / a' + 4 / b' ≥ |2*x - 1| - |x + 1|) ↔ 
    -7 ≤ x ∧ x ≤ 11) := by
  sorry

end NUMINAMATH_CALUDE_min_value_and_range_l559_55912


namespace NUMINAMATH_CALUDE_square_sum_reciprocal_l559_55901

theorem square_sum_reciprocal (x : ℝ) (h : x + 1/x = 5) : x^2 + 1/x^2 = 23 := by
  sorry

end NUMINAMATH_CALUDE_square_sum_reciprocal_l559_55901


namespace NUMINAMATH_CALUDE_fraction_less_than_sqrt_l559_55996

theorem fraction_less_than_sqrt (x : ℝ) (h : x > 0) : x / (1 + x) < Real.sqrt x := by
  sorry

end NUMINAMATH_CALUDE_fraction_less_than_sqrt_l559_55996


namespace NUMINAMATH_CALUDE_largest_common_number_l559_55928

def is_in_first_sequence (x : ℕ) : Prop := ∃ n : ℕ, x = 3 + 8 * n

def is_in_second_sequence (x : ℕ) : Prop := ∃ m : ℕ, x = 5 + 9 * m

def is_in_range (x : ℕ) : Prop := 1 ≤ x ∧ x ≤ 150

theorem largest_common_number :
  (is_in_first_sequence 131) ∧
  (is_in_second_sequence 131) ∧
  (is_in_range 131) ∧
  (∀ y : ℕ, y > 131 →
    ¬(is_in_first_sequence y ∧ is_in_second_sequence y ∧ is_in_range y)) :=
by sorry

end NUMINAMATH_CALUDE_largest_common_number_l559_55928


namespace NUMINAMATH_CALUDE_equation_solution_l559_55923

theorem equation_solution : 
  ∀ x : ℝ, x ≠ 5 → (x + 60 / (x - 5) = -12 ↔ x = 0 ∨ x = -7) :=
by sorry

end NUMINAMATH_CALUDE_equation_solution_l559_55923


namespace NUMINAMATH_CALUDE_chess_pawn_loss_l559_55914

theorem chess_pawn_loss (total_pawns_start : ℕ) (pawns_per_player : ℕ) 
  (kennedy_lost : ℕ) (pawns_left : ℕ) : 
  total_pawns_start = 2 * pawns_per_player →
  pawns_per_player = 8 →
  kennedy_lost = 4 →
  pawns_left = 11 →
  pawns_per_player - (pawns_left - (pawns_per_player - kennedy_lost)) = 1 := by
  sorry

end NUMINAMATH_CALUDE_chess_pawn_loss_l559_55914


namespace NUMINAMATH_CALUDE_quadratic_factorization_sum_l559_55913

theorem quadratic_factorization_sum (d e f : ℤ) : 
  (∀ x, x^2 + 9*x + 20 = (x + d) * (x + e)) →
  (∀ x, x^2 + 11*x - 60 = (x + e) * (x - f)) →
  d + e + f = 23 := by
sorry

end NUMINAMATH_CALUDE_quadratic_factorization_sum_l559_55913


namespace NUMINAMATH_CALUDE_trig_inequality_l559_55989

theorem trig_inequality (x y : Real) 
  (hx : 0 < x ∧ x < Real.pi / 2)
  (hy : 0 < y ∧ y < Real.pi / 2)
  (h_eq : Real.sin x = x * Real.cos y) : 
  x / 2 < y ∧ y < x :=
by sorry

end NUMINAMATH_CALUDE_trig_inequality_l559_55989


namespace NUMINAMATH_CALUDE_d_share_is_300_l559_55925

/-- Calculates the share of profit for an investor given the investments and total profit -/
def calculate_share (investment_c : ℚ) (investment_d : ℚ) (total_profit : ℚ) : ℚ :=
  (investment_d / (investment_c + investment_d)) * total_profit

/-- Theorem stating that D's share of the profit is 300 given the specified investments and total profit -/
theorem d_share_is_300 
  (investment_c : ℚ) 
  (investment_d : ℚ) 
  (total_profit : ℚ) 
  (h1 : investment_c = 1000)
  (h2 : investment_d = 1500)
  (h3 : total_profit = 500) :
  calculate_share investment_c investment_d total_profit = 300 := by
  sorry

#eval calculate_share 1000 1500 500

end NUMINAMATH_CALUDE_d_share_is_300_l559_55925


namespace NUMINAMATH_CALUDE_log_2_base_10_bound_l559_55955

theorem log_2_base_10_bound (h1 : 2^11 = 2048) (h2 : 2^12 = 4096) (h3 : 10^4 = 10000) :
  Real.log 2 / Real.log 10 < 4/11 := by
sorry

end NUMINAMATH_CALUDE_log_2_base_10_bound_l559_55955


namespace NUMINAMATH_CALUDE_cost_price_calculation_l559_55968

/-- Proves that the cost price of an article is 78.944 given the specified conditions --/
theorem cost_price_calculation (marked_price : ℝ) (discount_rate : ℝ) (profit_rate : ℝ) :
  marked_price = 98.68 →
  discount_rate = 0.05 →
  profit_rate = 0.25 →
  ∃ (cost_price : ℝ),
    (1 - discount_rate) * marked_price = cost_price * (1 + profit_rate) ∧
    cost_price = 78.944 := by
  sorry

end NUMINAMATH_CALUDE_cost_price_calculation_l559_55968


namespace NUMINAMATH_CALUDE_scientific_notation_correct_scientific_notation_format_l559_55929

/-- Represents the value in billions -/
def billion_value : ℝ := 57.44

/-- Represents the coefficient in scientific notation -/
def scientific_coefficient : ℝ := 5.744

/-- Represents the exponent in scientific notation -/
def scientific_exponent : ℤ := 9

/-- Asserts that the scientific notation is correct for the given value -/
theorem scientific_notation_correct :
  billion_value * 10^9 = scientific_coefficient * 10^scientific_exponent :=
by sorry

/-- Asserts that the coefficient in scientific notation is between 1 and 10 -/
theorem scientific_notation_format :
  1 ≤ scientific_coefficient ∧ scientific_coefficient < 10 :=
by sorry

end NUMINAMATH_CALUDE_scientific_notation_correct_scientific_notation_format_l559_55929


namespace NUMINAMATH_CALUDE_roots_product_minus_three_l559_55905

theorem roots_product_minus_three (x₁ x₂ : ℝ) : 
  (3 * x₁^2 - 7 * x₁ - 6 = 0) → 
  (3 * x₂^2 - 7 * x₂ - 6 = 0) → 
  (x₁ - 3) * (x₂ - 3) = 0 := by
  sorry

end NUMINAMATH_CALUDE_roots_product_minus_three_l559_55905


namespace NUMINAMATH_CALUDE_sum_a_d_l559_55945

theorem sum_a_d (a b c d : ℝ) 
  (h1 : a * b + a * c + b * d + c * d = 48) 
  (h2 : b + c = 6) : 
  a + d = 8 := by
sorry

end NUMINAMATH_CALUDE_sum_a_d_l559_55945


namespace NUMINAMATH_CALUDE_y_derivative_l559_55958

noncomputable def y (x : ℝ) : ℝ := Real.tan (Real.sqrt (Real.cos (1/3))) + (Real.sin (31*x))^2 / (31 * Real.cos (62*x))

theorem y_derivative (x : ℝ) :
  deriv y x = (2 * (Real.sin (31*x) * Real.cos (31*x) * Real.cos (62*x) + Real.sin (31*x)^2 * Real.sin (62*x))) / Real.cos (62*x)^2 :=
sorry

end NUMINAMATH_CALUDE_y_derivative_l559_55958


namespace NUMINAMATH_CALUDE_carmen_cookie_sales_l559_55944

/-- Represents the number of boxes of each type of cookie sold --/
structure CookieSales where
  samoas : ℕ
  thinMints : ℕ
  fudgeDelights : ℕ
  sugarCookies : ℕ

/-- Represents the price of each type of cookie --/
structure CookiePrices where
  samoas : ℚ
  thinMints : ℚ
  fudgeDelights : ℚ
  sugarCookies : ℚ

/-- Calculates the total revenue from cookie sales --/
def totalRevenue (sales : CookieSales) (prices : CookiePrices) : ℚ :=
  sales.samoas * prices.samoas +
  sales.thinMints * prices.thinMints +
  sales.fudgeDelights * prices.fudgeDelights +
  sales.sugarCookies * prices.sugarCookies

/-- The main theorem representing Carmen's cookie sales --/
theorem carmen_cookie_sales 
  (sales : CookieSales)
  (prices : CookiePrices)
  (h1 : sales.samoas = 3)
  (h2 : sales.thinMints = 2)
  (h3 : sales.fudgeDelights = 1)
  (h4 : prices.samoas = 4)
  (h5 : prices.thinMints = 7/2)
  (h6 : prices.fudgeDelights = 5)
  (h7 : prices.sugarCookies = 2)
  (h8 : totalRevenue sales prices = 42) :
  sales.sugarCookies = 9 := by
  sorry

end NUMINAMATH_CALUDE_carmen_cookie_sales_l559_55944


namespace NUMINAMATH_CALUDE_sign_determination_l559_55931

theorem sign_determination (a b : ℝ) (h1 : a + b < 0) (h2 : b / a > 0) : a < 0 ∧ b < 0 := by
  sorry

end NUMINAMATH_CALUDE_sign_determination_l559_55931


namespace NUMINAMATH_CALUDE_average_price_approx_1_645_l559_55906

/-- Calculate the average price per bottle given the number and prices of large and small bottles, and a discount rate for large bottles. -/
def averagePricePerBottle (largeBottles smallBottles : ℕ) (largePricePerBottle smallPricePerBottle : ℚ) (discountRate : ℚ) : ℚ :=
  let largeCost := largeBottles * largePricePerBottle
  let largeDiscount := largeCost * discountRate
  let discountedLargeCost := largeCost - largeDiscount
  let smallCost := smallBottles * smallPricePerBottle
  let totalCost := discountedLargeCost + smallCost
  let totalBottles := largeBottles + smallBottles
  totalCost / totalBottles

/-- The average price per bottle is approximately $1.645 given the specific conditions. -/
theorem average_price_approx_1_645 :
  let largeBattles := 1325
  let smallBottles := 750
  let largePricePerBottle := 189/100  -- $1.89
  let smallPricePerBottle := 138/100  -- $1.38
  let discountRate := 5/100  -- 5%
  abs (averagePricePerBottle largeBattles smallBottles largePricePerBottle smallPricePerBottle discountRate - 1645/1000) < 1/1000 := by
  sorry


end NUMINAMATH_CALUDE_average_price_approx_1_645_l559_55906


namespace NUMINAMATH_CALUDE_population_scientific_notation_l559_55946

def population : ℝ := 1370000000

theorem population_scientific_notation :
  ∃ (a : ℝ) (n : ℤ), population = a * (10 : ℝ) ^ n ∧ 1 ≤ |a| ∧ |a| < 10 ∧ a = 1.37 ∧ n = 9 := by
  sorry

end NUMINAMATH_CALUDE_population_scientific_notation_l559_55946


namespace NUMINAMATH_CALUDE_simplify_expression_l559_55939

theorem simplify_expression (x : ℝ) (hx : x ≠ 0) :
  Real.sqrt (4 + ((x^3 - 2) / (3*x))^2) = (Real.sqrt (x^6 - 4*x^3 + 36*x^2 + 4)) / (3*x) := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l559_55939


namespace NUMINAMATH_CALUDE_pentagonal_field_fencing_cost_l559_55999

/-- Calculates the cost of fencing for an irregular pentagonal field -/
def fencing_cost (side1 side2 side3 side4 side5 : ℕ) 
                 (rate1 rate2 rate3 : ℕ) : ℕ :=
  rate1 * (side1 + side2) + rate2 * side3 + rate3 * (side4 + side5)

/-- Theorem stating the total cost of fencing for the given pentagonal field -/
theorem pentagonal_field_fencing_cost :
  fencing_cost 42 37 52 65 48 7 5 10 = 1943 := by sorry

end NUMINAMATH_CALUDE_pentagonal_field_fencing_cost_l559_55999


namespace NUMINAMATH_CALUDE_system_solution_l559_55919

theorem system_solution :
  ∃ (x y z : ℝ),
    (x + y + z = 26) ∧
    (3 * x - 2 * y + z = 3) ∧
    (x - 4 * y - 2 * z = -13) ∧
    (x = -32.2) ∧
    (y = -13.8) ∧
    (z = 72) := by
  sorry

end NUMINAMATH_CALUDE_system_solution_l559_55919


namespace NUMINAMATH_CALUDE_imaginary_cube_l559_55956

theorem imaginary_cube (i : ℂ) : i^2 = -1 → 1 + i^3 = 1 - i := by
  sorry

end NUMINAMATH_CALUDE_imaginary_cube_l559_55956


namespace NUMINAMATH_CALUDE_rita_remaining_money_l559_55966

def initial_amount : ℕ := 400
def dress_cost : ℕ := 20
def pants_cost : ℕ := 12
def jacket_cost : ℕ := 30
def transportation_cost : ℕ := 5
def dress_quantity : ℕ := 5
def pants_quantity : ℕ := 3
def jacket_quantity : ℕ := 4

theorem rita_remaining_money :
  initial_amount - 
  (dress_cost * dress_quantity + 
   pants_cost * pants_quantity + 
   jacket_cost * jacket_quantity + 
   transportation_cost) = 139 := by
sorry

end NUMINAMATH_CALUDE_rita_remaining_money_l559_55966


namespace NUMINAMATH_CALUDE_impossible_coloring_l559_55932

theorem impossible_coloring : ¬∃(color : ℕ → Bool),
  (∀ n : ℕ, color n ≠ color (n + 5)) ∧
  (∀ n : ℕ, color n ≠ color (2 * n)) :=
by sorry

end NUMINAMATH_CALUDE_impossible_coloring_l559_55932


namespace NUMINAMATH_CALUDE_reflected_beam_angle_l559_55983

/-- Given a fixed beam of light falling on a mirror at an acute angle α with its projection
    on the mirror plane, and the mirror rotated by an acute angle β around this projection,
    the angle θ between the two reflected beams (before and after rotation) is given by
    θ = arccos(1 - 2 * sin²α * sin²β) -/
theorem reflected_beam_angle (α β : Real) (h_α : 0 < α ∧ α < π/2) (h_β : 0 < β ∧ β < π/2) :
  ∃ θ : Real, θ = Real.arccos (1 - 2 * Real.sin α ^ 2 * Real.sin β ^ 2) :=
sorry

end NUMINAMATH_CALUDE_reflected_beam_angle_l559_55983


namespace NUMINAMATH_CALUDE_min_sum_consecutive_multiples_l559_55903

theorem min_sum_consecutive_multiples : 
  ∃ (a b c d : ℕ), 
    (b = a + 1) ∧ 
    (c = b + 1) ∧ 
    (d = c + 1) ∧
    (∃ k : ℕ, a = 11 * k) ∧
    (∃ l : ℕ, b = 7 * l) ∧
    (∃ m : ℕ, c = 5 * m) ∧
    (∃ n : ℕ, d = 3 * n) ∧
    (∀ w x y z : ℕ, 
      (x = w + 1) ∧ 
      (y = x + 1) ∧ 
      (z = y + 1) ∧
      (∃ p : ℕ, w = 11 * p) ∧
      (∃ q : ℕ, x = 7 * q) ∧
      (∃ r : ℕ, y = 5 * r) ∧
      (∃ s : ℕ, z = 3 * s) →
      (a + b + c + d ≤ w + x + y + z)) ∧
    (a + b + c + d = 1458) :=
by sorry

end NUMINAMATH_CALUDE_min_sum_consecutive_multiples_l559_55903


namespace NUMINAMATH_CALUDE_line_l_properties_l559_55930

/-- A line that passes through (3,2) and has equal intercepts on both axes -/
def line_l (x y : ℝ) : Prop :=
  y = -x + 5

theorem line_l_properties :
  (∃ a : ℝ, line_l a 2 ∧ a = 3) ∧
  (∃ b : ℝ, line_l b 0 ∧ line_l 0 b ∧ b > 0) :=
by sorry

end NUMINAMATH_CALUDE_line_l_properties_l559_55930


namespace NUMINAMATH_CALUDE_emergency_vehicle_reachable_area_l559_55908

/-- The area reachable by an emergency vehicle in a desert -/
theorem emergency_vehicle_reachable_area 
  (road_speed : ℝ) 
  (sand_speed : ℝ) 
  (time : ℝ) 
  (h_road_speed : road_speed = 60) 
  (h_sand_speed : sand_speed = 10) 
  (h_time : time = 5/60) : 
  ∃ (area : ℝ), area = 25 + 25 * Real.pi / 36 ∧ 
  area = (road_speed * time)^2 + 4 * (Real.pi * (sand_speed * time)^2 / 4) := by
sorry

end NUMINAMATH_CALUDE_emergency_vehicle_reachable_area_l559_55908


namespace NUMINAMATH_CALUDE_jims_estimate_l559_55985

theorem jims_estimate (x y ε : ℝ) (hx : x > y) (hy : y > 0) (hε : ε > 0) :
  (x^2 + ε) - (y^2 - ε) > x^2 - y^2 := by
  sorry

end NUMINAMATH_CALUDE_jims_estimate_l559_55985


namespace NUMINAMATH_CALUDE_four_lines_equal_angles_l559_55967

/-- A point in 3D space -/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- A line in 3D space -/
structure Line3D where
  point : Point3D
  direction : Point3D

/-- A rectangular box in 3D space -/
structure RectangularBox where
  corner : Point3D
  width : ℝ
  length : ℝ
  height : ℝ

/-- The angle between a line and an edge of the box -/
def angleWithEdge (l : Line3D) (b : RectangularBox) (edge : Fin 12) : ℝ :=
  sorry

/-- A line forms equal angles with all edges of the box -/
def formsEqualAngles (l : Line3D) (b : RectangularBox) : Prop :=
  ∀ (e1 e2 : Fin 12), angleWithEdge l b e1 = angleWithEdge l b e2

/-- The main theorem -/
theorem four_lines_equal_angles (P : Point3D) (b : RectangularBox) :
  ∃! (lines : Finset Line3D), lines.card = 4 ∧ 
    ∀ l ∈ lines, l.point = P ∧ formsEqualAngles l b :=
  sorry

end NUMINAMATH_CALUDE_four_lines_equal_angles_l559_55967


namespace NUMINAMATH_CALUDE_min_bushes_for_pumpkins_l559_55991

/-- Represents the number of containers of raspberries per bush -/
def containers_per_bush : ℕ := 10

/-- Represents the number of containers of raspberries needed to trade for 3 pumpkins -/
def containers_per_trade : ℕ := 6

/-- Represents the number of pumpkins obtained from one trade -/
def pumpkins_per_trade : ℕ := 3

/-- Represents the target number of pumpkins -/
def target_pumpkins : ℕ := 72

/-- 
Proves that the minimum number of bushes needed to obtain at least the target number of pumpkins
is 15, given the defined ratios of containers per bush and pumpkins per trade.
-/
theorem min_bushes_for_pumpkins :
  ∃ (n : ℕ), n * containers_per_bush * pumpkins_per_trade ≥ target_pumpkins * containers_per_trade ∧
  ∀ (m : ℕ), m * containers_per_bush * pumpkins_per_trade ≥ target_pumpkins * containers_per_trade → m ≥ n :=
by sorry

end NUMINAMATH_CALUDE_min_bushes_for_pumpkins_l559_55991


namespace NUMINAMATH_CALUDE_sequence_max_value_l559_55959

def a (n : ℕ+) : ℚ := n / (n^2 + 156)

theorem sequence_max_value :
  (∃ (k : ℕ+), a k = 1/25 ∧ 
   ∀ (n : ℕ+), a n ≤ 1/25) ∧
  (∀ (n : ℕ+), a n = 1/25 → (n = 12 ∨ n = 13)) :=
sorry

end NUMINAMATH_CALUDE_sequence_max_value_l559_55959


namespace NUMINAMATH_CALUDE_special_line_equation_l559_55995

/-- A line passing through (9, 4) with x-intercept 5 units greater than y-intercept -/
def SpecialLine : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | ∃ (m b : ℝ), p.2 = m * p.1 + b ∧ (9, 4) ∈ {q : ℝ × ℝ | q.2 = m * q.1 + b} ∧
    ∃ (x y : ℝ), x = y + 5 ∧ 0 = m * x + b ∧ y = b}

/-- The three possible equations of the special line -/
def PossibleEquations : Set (ℝ × ℝ → Prop) :=
  {(λ p : ℝ × ℝ ↦ 2 * p.1 + 3 * p.2 - 30 = 0),
   (λ p : ℝ × ℝ ↦ 2 * p.1 - 3 * p.2 - 6 = 0),
   (λ p : ℝ × ℝ ↦ p.1 - p.2 - 5 = 0)}

theorem special_line_equation :
  ∃ (eq : ℝ × ℝ → Prop), eq ∈ PossibleEquations ∧ ∀ p : ℝ × ℝ, p ∈ SpecialLine ↔ eq p :=
by sorry

end NUMINAMATH_CALUDE_special_line_equation_l559_55995


namespace NUMINAMATH_CALUDE_difference_largest_smallest_l559_55922

/-- The set of available digits --/
def available_digits : Finset Nat := {2, 0, 3, 5, 8}

/-- A four-digit number formed from the available digits --/
structure FourDigitNumber where
  digits : Finset Nat
  size_eq : digits.card = 4
  subset : digits ⊆ available_digits

/-- The largest four-digit number that can be formed --/
def largest_number : Nat := 8532

/-- The smallest four-digit number that can be formed --/
def smallest_number : Nat := 2035

/-- Theorem: The difference between the largest and smallest four-digit numbers is 6497 --/
theorem difference_largest_smallest :
  largest_number - smallest_number = 6497 := by sorry

end NUMINAMATH_CALUDE_difference_largest_smallest_l559_55922


namespace NUMINAMATH_CALUDE_specific_frustum_lateral_surface_area_l559_55978

/-- The lateral surface area of a frustum of a cone --/
def lateralSurfaceArea (slantHeight : ℝ) (radiusRatio : ℝ) (centralAngle : ℝ) : ℝ :=
  sorry

/-- Theorem: The lateral surface area of a specific frustum of a cone --/
theorem specific_frustum_lateral_surface_area :
  lateralSurfaceArea 10 (2/5) 216 = 252 * Real.pi / 5 := by
  sorry

end NUMINAMATH_CALUDE_specific_frustum_lateral_surface_area_l559_55978


namespace NUMINAMATH_CALUDE_three_planes_division_l559_55987

/-- A plane in 3D space -/
structure Plane3D where
  -- We don't need to define the internal structure of a plane for this problem

/-- The number of parts that a set of planes divides 3D space into -/
def num_parts (planes : List Plane3D) : ℕ := sorry

theorem three_planes_division :
  ∀ (p1 p2 p3 : Plane3D),
  4 ≤ num_parts [p1, p2, p3] ∧ num_parts [p1, p2, p3] ≤ 8 := by sorry

end NUMINAMATH_CALUDE_three_planes_division_l559_55987


namespace NUMINAMATH_CALUDE_faye_earnings_l559_55904

/-- Calculates the earnings from selling necklaces at a garage sale -/
def necklace_earnings (bead_necklaces gem_stone_necklaces price_per_necklace : ℕ) : ℕ :=
  (bead_necklaces + gem_stone_necklaces) * price_per_necklace

/-- Proves that Faye's earnings from selling necklaces are 70 dollars -/
theorem faye_earnings : necklace_earnings 3 7 7 = 70 := by
  sorry

end NUMINAMATH_CALUDE_faye_earnings_l559_55904


namespace NUMINAMATH_CALUDE_integer_solutions_of_equation_l559_55942

theorem integer_solutions_of_equation :
  ∀ x y : ℤ, x^2 + x*y - y = 2 ↔ (x = 2 ∧ y = -2) ∨ (x = 0 ∧ y = -2) := by
  sorry

end NUMINAMATH_CALUDE_integer_solutions_of_equation_l559_55942


namespace NUMINAMATH_CALUDE_card_distribution_implies_square_l559_55902

theorem card_distribution_implies_square (n : ℕ) (m : ℕ) (h_n : n ≥ 3) 
  (h_m : m = n * (n - 1) / 2) (h_m_even : Even m) 
  (a : Fin n → ℕ) (h_a_range : ∀ i, 1 ≤ a i ∧ a i ≤ m) 
  (h_a_distinct : ∀ i j, i ≠ j → a i ≠ a j) 
  (h_sums_distinct : ∀ i j k l, (i ≠ j ∧ k ≠ l) → (i, j) ≠ (k, l) → 
    (a i + a j) % m ≠ (a k + a l) % m) :
  ∃ k : ℕ, n = k^2 := by
  sorry

end NUMINAMATH_CALUDE_card_distribution_implies_square_l559_55902


namespace NUMINAMATH_CALUDE_john_mary_chess_tournament_l559_55969

theorem john_mary_chess_tournament : 
  ¬∃ (n : ℕ), (n % 16 = 0 ∧ (n + 1) % 25 = 0) ∨ (n % 25 = 0 ∧ (n + 1) % 16 = 0) :=
by sorry

end NUMINAMATH_CALUDE_john_mary_chess_tournament_l559_55969


namespace NUMINAMATH_CALUDE_max_candy_leftover_l559_55949

theorem max_candy_leftover (x : ℕ) : ∃ (q r : ℕ), x = 12 * q + r ∧ r < 12 ∧ r ≤ 11 :=
sorry

end NUMINAMATH_CALUDE_max_candy_leftover_l559_55949


namespace NUMINAMATH_CALUDE_find_g_l559_55994

-- Define the functions f and g
def f : ℝ → ℝ := λ x ↦ 2 * x + 3

-- Define the property of g
def g_property (g : ℝ → ℝ) : Prop := ∀ x, g (x + 2) = f x

-- Theorem statement
theorem find_g : ∃ g : ℝ → ℝ, g_property g ∧ (∀ x, g x = 2 * x - 1) := by
  sorry

end NUMINAMATH_CALUDE_find_g_l559_55994


namespace NUMINAMATH_CALUDE_empty_solution_set_l559_55965

theorem empty_solution_set (a : ℝ) :
  (∀ x : ℝ, ¬(|x - 1| - |x + 2| < a)) ↔ a ≤ -3 :=
by sorry

end NUMINAMATH_CALUDE_empty_solution_set_l559_55965


namespace NUMINAMATH_CALUDE_binomial_coefficient_equality_l559_55938

theorem binomial_coefficient_equality (n : ℕ) (r : ℕ) : 
  (Nat.choose n (4*r - 1) = Nat.choose n (r + 1)) → 
  (n = 20 ∧ r = 4) := by
  sorry

end NUMINAMATH_CALUDE_binomial_coefficient_equality_l559_55938


namespace NUMINAMATH_CALUDE_flight_chess_starting_position_l559_55937

theorem flight_chess_starting_position (x : ℤ) :
  x - 5 + 4 + 2 - 3 + 1 = 6 → x = 7 := by
  sorry

end NUMINAMATH_CALUDE_flight_chess_starting_position_l559_55937


namespace NUMINAMATH_CALUDE_spelling_bee_initial_students_l559_55998

theorem spelling_bee_initial_students :
  ∀ (initial_students : ℕ),
    (initial_students : ℝ) * 0.3 * 0.5 = 18 →
    initial_students = 120 :=
by
  sorry

end NUMINAMATH_CALUDE_spelling_bee_initial_students_l559_55998


namespace NUMINAMATH_CALUDE_polynomial_division_quotient_l559_55975

theorem polynomial_division_quotient :
  ∀ (x : ℝ),
  ∃ (r : ℝ),
  8 * x^3 + 5 * x^2 - 4 * x - 7 = (x + 3) * (8 * x^2 - 19 * x + 53) + r :=
by
  sorry

end NUMINAMATH_CALUDE_polynomial_division_quotient_l559_55975


namespace NUMINAMATH_CALUDE_circle_graph_percentage_l559_55954

theorem circle_graph_percentage (total_degrees : ℝ) (total_percentage : ℝ) 
  (manufacturing_degrees : ℝ) (manufacturing_percentage : ℝ) : 
  total_degrees = 360 →
  total_percentage = 100 →
  manufacturing_degrees = 108 →
  manufacturing_percentage / total_percentage = manufacturing_degrees / total_degrees →
  manufacturing_percentage = 30 := by
sorry

end NUMINAMATH_CALUDE_circle_graph_percentage_l559_55954


namespace NUMINAMATH_CALUDE_cone_cylinder_volume_ratio_l559_55927

/-- The ratio of the volume of a cone to the volume of a cylinder with the same base radius,
    where the cone's height is one-third of the cylinder's height, is 1/9. -/
theorem cone_cylinder_volume_ratio (r h : ℝ) (hr : r > 0) (hh : h > 0) :
  (1 / 3 * π * r^2 * (h / 3)) / (π * r^2 * h) = 1 / 9 := by
  sorry


end NUMINAMATH_CALUDE_cone_cylinder_volume_ratio_l559_55927


namespace NUMINAMATH_CALUDE_restaurant_cooks_count_l559_55988

/-- Proves that the number of cooks is 9 given the initial and final ratios of cooks to waiters --/
theorem restaurant_cooks_count (initial_cooks : ℕ) (initial_waiters : ℕ) 
  (h1 : initial_cooks * 8 = initial_waiters * 3)
  (h2 : initial_cooks * 4 = (initial_waiters + 12) * 1) : 
  initial_cooks = 9 := by
  sorry

end NUMINAMATH_CALUDE_restaurant_cooks_count_l559_55988


namespace NUMINAMATH_CALUDE_complex_point_equivalence_l559_55911

theorem complex_point_equivalence : 
  let z : ℂ := (Complex.I) / (1 + 3 * Complex.I)
  z = (3 : ℝ) / 10 + ((1 : ℝ) / 10) * Complex.I :=
by sorry

end NUMINAMATH_CALUDE_complex_point_equivalence_l559_55911


namespace NUMINAMATH_CALUDE_school_students_count_l559_55951

theorem school_students_count (total : ℕ) 
  (chess_ratio : Real) 
  (swimming_ratio : Real) 
  (swimming_count : ℕ) 
  (h1 : chess_ratio = 0.25)
  (h2 : swimming_ratio = 0.50)
  (h3 : swimming_count = 125)
  (h4 : ↑swimming_count = swimming_ratio * (chess_ratio * total)) :
  total = 1000 := by
sorry

end NUMINAMATH_CALUDE_school_students_count_l559_55951


namespace NUMINAMATH_CALUDE_last_group_markers_theorem_l559_55997

/-- Calculates the number of markers each student in the last group receives --/
def markers_per_last_student (total_students : ℕ) (marker_boxes : ℕ) (markers_per_box : ℕ)
  (first_group_students : ℕ) (first_group_markers_per_student : ℕ)
  (second_group_students : ℕ) (second_group_markers_per_student : ℕ) : ℕ :=
  let total_markers := marker_boxes * markers_per_box
  let first_group_markers := first_group_students * first_group_markers_per_student
  let second_group_markers := second_group_students * second_group_markers_per_student
  let remaining_markers := total_markers - first_group_markers - second_group_markers
  let last_group_students := total_students - first_group_students - second_group_students
  remaining_markers / last_group_students

/-- Theorem stating that under the given conditions, each student in the last group receives 6 markers --/
theorem last_group_markers_theorem :
  markers_per_last_student 30 22 5 10 2 15 4 = 6 := by
  sorry

end NUMINAMATH_CALUDE_last_group_markers_theorem_l559_55997


namespace NUMINAMATH_CALUDE_laundry_detergent_problem_l559_55979

def standard_weight : ℕ := 450
def price_per_bag : ℕ := 3
def weight_deviations : List ℤ := [-6, -3, -2, 0, 1, 4, 5, -1]
def qualification_criterion : ℤ → Bool := λ x => x.natAbs ≤ 4

theorem laundry_detergent_problem :
  let total_weight := (weight_deviations.sum + standard_weight * weight_deviations.length : ℤ)
  let qualified_bags := weight_deviations.filter qualification_criterion
  let total_sales := qualified_bags.length * price_per_bag
  (total_weight = 3598 ∧ total_sales = 18) := by sorry

end NUMINAMATH_CALUDE_laundry_detergent_problem_l559_55979


namespace NUMINAMATH_CALUDE_circular_section_area_l559_55992

theorem circular_section_area (r : ℝ) (d : ℝ) (h : r = 5 ∧ d = 3) :
  let section_radius : ℝ := Real.sqrt (r^2 - d^2)
  π * section_radius^2 = 16 * π :=
by sorry

end NUMINAMATH_CALUDE_circular_section_area_l559_55992


namespace NUMINAMATH_CALUDE_factorial_fraction_simplification_l559_55924

theorem factorial_fraction_simplification (N : ℕ) (h : N > 2) :
  (Nat.factorial (N - 2) * (N - 1)^2) / Nat.factorial (N + 2) =
  (N - 1) / (N * (N + 1) * (N + 2)) :=
by sorry

end NUMINAMATH_CALUDE_factorial_fraction_simplification_l559_55924


namespace NUMINAMATH_CALUDE_average_candies_sigyeong_group_l559_55984

def sigyeong_group : List Nat := [16, 22, 30, 26, 18, 20]

theorem average_candies_sigyeong_group : 
  (sigyeong_group.sum / sigyeong_group.length : ℚ) = 22 := by
  sorry

end NUMINAMATH_CALUDE_average_candies_sigyeong_group_l559_55984


namespace NUMINAMATH_CALUDE_optimus_prime_distance_l559_55947

/-- Prove that the distance between points A and B is 750 km given the conditions in the problem --/
theorem optimus_prime_distance : ∀ (D S : ℝ),
  (D / S - D / (S * (1 + 1/4)) = 1) →
  (150 / S + (D - 150) / S - (150 / S + (D - 150) / (S * (1 + 1/5))) = 2/3) →
  D = 750 := by
  sorry

end NUMINAMATH_CALUDE_optimus_prime_distance_l559_55947


namespace NUMINAMATH_CALUDE_sum_first_100_even_integers_l559_55981

/-- The sum of the first n positive even integers -/
def sumFirstNEvenIntegers (n : ℕ) : ℕ := n * (n + 1)

/-- Theorem: The sum of the first 100 positive even integers is 10100 -/
theorem sum_first_100_even_integers : sumFirstNEvenIntegers 100 = 10100 := by
  sorry

end NUMINAMATH_CALUDE_sum_first_100_even_integers_l559_55981


namespace NUMINAMATH_CALUDE_min_discriminant_quadratic_trinomial_l559_55900

theorem min_discriminant_quadratic_trinomial (a b c : ℝ) :
  (∀ x, a * x^2 + b * x + c ≥ 0) →
  (∀ x, abs x < 1 → a * x^2 + b * x + c ≤ 1 / Real.sqrt (1 - x^2)) →
  b^2 - 4*a*c ≥ -4 ∧ ∃ a' b' c', b'^2 - 4*a'*c' = -4 :=
by sorry

end NUMINAMATH_CALUDE_min_discriminant_quadratic_trinomial_l559_55900


namespace NUMINAMATH_CALUDE_company_employees_l559_55933

/-- Calculates the initial number of employees in a company given the following conditions:
  * Hourly wage
  * Hours worked per day
  * Days worked per week
  * Weeks worked per month
  * Number of new hires
  * Total monthly payroll after hiring
-/
def initial_employees (
  hourly_wage : ℕ
  ) (hours_per_day : ℕ
  ) (days_per_week : ℕ
  ) (weeks_per_month : ℕ
  ) (new_hires : ℕ
  ) (total_payroll : ℕ
  ) : ℕ :=
  let monthly_hours := hours_per_day * days_per_week * weeks_per_month
  let monthly_wage := hourly_wage * monthly_hours
  (total_payroll / monthly_wage) - new_hires

theorem company_employees :
  initial_employees 12 10 5 4 200 1680000 = 500 := by
  sorry

end NUMINAMATH_CALUDE_company_employees_l559_55933


namespace NUMINAMATH_CALUDE_quadratic_inequality_solution_set_is_real_line_l559_55952

/-- The solution set of a quadratic inequality is the entire real line -/
theorem quadratic_inequality_solution_set_is_real_line 
  (a b c : ℝ) (h : a ≠ 0) :
  (∀ x, a * x^2 + b * x + c < 0) ↔ (a < 0 ∧ b^2 - 4*a*c < 0) :=
sorry

end NUMINAMATH_CALUDE_quadratic_inequality_solution_set_is_real_line_l559_55952


namespace NUMINAMATH_CALUDE_line_tangent_to_parabola_l559_55976

/-- The value of c for which the line y = 3x + c is tangent to the parabola y² = 12x -/
def tangent_line_c : ℝ := 1

/-- The line equation: y = 3x + c -/
def line_equation (x y c : ℝ) : Prop := y = 3 * x + c

/-- The parabola equation: y² = 12x -/
def parabola_equation (x y : ℝ) : Prop := y^2 = 12 * x

/-- The line y = 3x + c is tangent to the parabola y² = 12x when c = tangent_line_c -/
theorem line_tangent_to_parabola :
  ∃! (x y : ℝ), line_equation x y tangent_line_c ∧ parabola_equation x y :=
sorry

end NUMINAMATH_CALUDE_line_tangent_to_parabola_l559_55976


namespace NUMINAMATH_CALUDE_lassis_from_nine_mangoes_l559_55961

/-- The number of lassis that can be made from a given number of mangoes -/
def lassis_from_mangoes (mangoes : ℕ) : ℕ :=
  5 * mangoes

/-- The cost of a given number of mangoes -/
def mango_cost (mangoes : ℕ) : ℕ :=
  2 * mangoes

theorem lassis_from_nine_mangoes :
  lassis_from_mangoes 9 = 45 :=
by sorry

end NUMINAMATH_CALUDE_lassis_from_nine_mangoes_l559_55961


namespace NUMINAMATH_CALUDE_rectangular_field_fence_l559_55980

theorem rectangular_field_fence (area : ℝ) (fence_length : ℝ) (uncovered_side : ℝ) :
  area = 600 →
  fence_length = 130 →
  uncovered_side * (fence_length - uncovered_side) / 2 = area →
  uncovered_side = 120 :=
by
  sorry

end NUMINAMATH_CALUDE_rectangular_field_fence_l559_55980


namespace NUMINAMATH_CALUDE_intersection_of_sets_l559_55918

/-- Given sets A and B, prove their intersection -/
theorem intersection_of_sets :
  let A : Set ℝ := {x | x < 1}
  let B : Set ℝ := {x | x^2 - x - 6 < 0}
  A ∩ B = {x | -2 < x ∧ x < 1} :=
by sorry

end NUMINAMATH_CALUDE_intersection_of_sets_l559_55918


namespace NUMINAMATH_CALUDE_polynomial_identity_l559_55936

theorem polynomial_identity : 
  102^5 - 5 * 102^4 + 10 * 102^3 - 10 * 102^2 + 5 * 102 - 1 = 1051012301 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_identity_l559_55936


namespace NUMINAMATH_CALUDE_sum_of_digits_of_greatest_prime_divisor_l559_55948

-- Define the number we're working with
def n : Nat := 32767

-- Define a function to get the greatest prime divisor
def greatestPrimeDivisor (m : Nat) : Nat :=
  sorry

-- Define a function to sum the digits of a number
def sumOfDigits (m : Nat) : Nat :=
  sorry

-- The theorem to prove
theorem sum_of_digits_of_greatest_prime_divisor :
  sumOfDigits (greatestPrimeDivisor n) = 13 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_digits_of_greatest_prime_divisor_l559_55948


namespace NUMINAMATH_CALUDE_quadratic_equation_solution_l559_55907

theorem quadratic_equation_solution :
  let f : ℝ → ℝ := λ x ↦ x^2 - 2*x - 8
  ∃ x₁ x₂ : ℝ, x₁ = 4 ∧ x₂ = -2 ∧ f x₁ = 0 ∧ f x₂ = 0 ∧
  ∀ x : ℝ, f x = 0 → x = x₁ ∨ x = x₂ :=
by sorry

end NUMINAMATH_CALUDE_quadratic_equation_solution_l559_55907


namespace NUMINAMATH_CALUDE_acute_angle_alpha_l559_55953

theorem acute_angle_alpha (α : Real) (h : 0 < α ∧ α < Real.pi / 2) 
  (eq : Real.cos (Real.pi / 6) * Real.sin α = Real.sqrt 3 / 4) : 
  α = Real.pi / 6 := by
  sorry

end NUMINAMATH_CALUDE_acute_angle_alpha_l559_55953


namespace NUMINAMATH_CALUDE_quadratic_equation_solution_l559_55940

theorem quadratic_equation_solution (c : ℝ) :
  (∃ x : ℝ, x^2 - 3*x + c = 0 ∧ (-x)^2 + 3*(-x) - c = 0) →
  (∀ x : ℝ, x^2 - 3*x + c = 0 ↔ (x = 0 ∨ x = 3)) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_equation_solution_l559_55940


namespace NUMINAMATH_CALUDE_good_quality_sufficient_for_not_cheap_l559_55917

-- Define the propositions
variable (good_quality : Prop)
variable (not_cheap : Prop)

-- Define the given equivalence
axiom you_get_what_you_pay_for : (good_quality → not_cheap) ↔ (¬not_cheap → ¬good_quality)

-- Theorem to prove
theorem good_quality_sufficient_for_not_cheap : good_quality → not_cheap := by
  sorry

end NUMINAMATH_CALUDE_good_quality_sufficient_for_not_cheap_l559_55917


namespace NUMINAMATH_CALUDE_parallel_line_through_point_l559_55916

/-- Given two lines in the form ax + by + c = 0, they are parallel if and only if they have the same slope (a/b ratio) -/
def parallel_lines (a₁ b₁ c₁ a₂ b₂ c₂ : ℝ) : Prop :=
  a₁ / b₁ = a₂ / b₂

/-- A point (x, y) lies on a line ax + by + c = 0 if and only if the equation is satisfied -/
def point_on_line (a b c x y : ℝ) : Prop :=
  a * x + b * y + c = 0

/-- The theorem states that the line 3x + 4y - 11 = 0 is parallel to 3x + 4y + 1 = 0 and passes through (1, 2) -/
theorem parallel_line_through_point :
  parallel_lines 3 4 (-11) 3 4 1 ∧ point_on_line 3 4 (-11) 1 2 :=
by sorry

end NUMINAMATH_CALUDE_parallel_line_through_point_l559_55916


namespace NUMINAMATH_CALUDE_vector_addition_proof_l559_55971

def a : Fin 2 → ℝ := ![1, -2]
def b : Fin 2 → ℝ := ![3, 5]

theorem vector_addition_proof : 
  (2 • a + b) = ![5, 1] := by sorry

end NUMINAMATH_CALUDE_vector_addition_proof_l559_55971


namespace NUMINAMATH_CALUDE_joe_count_l559_55941

theorem joe_count (barry_count kevin_count julie_count : ℕ) 
  (nice_count : ℕ) (joe_nice_ratio : ℚ) :
  barry_count = 24 →
  kevin_count = 20 →
  julie_count = 80 →
  nice_count = 99 →
  joe_nice_ratio = 1/10 →
  ∃ (joe_count : ℕ),
    joe_count = 50 ∧
    nice_count = barry_count + 
                 (kevin_count / 2) + 
                 (julie_count * 3 / 4) + 
                 (joe_count * joe_nice_ratio) :=
by sorry

end NUMINAMATH_CALUDE_joe_count_l559_55941


namespace NUMINAMATH_CALUDE_cookie_is_circle_with_radius_nine_l559_55910

/-- The cookie's boundary equation -/
def cookie_boundary (x y : ℝ) : Prop :=
  x^2 + y^2 + 28 = 6*x + 20*y

/-- The circle equation with center (3, 10) and radius 9 -/
def circle_equation (x y : ℝ) : Prop :=
  (x - 3)^2 + (y - 10)^2 = 81

/-- Theorem stating that the cookie boundary is equivalent to a circle with radius 9 -/
theorem cookie_is_circle_with_radius_nine :
  ∀ x y : ℝ, cookie_boundary x y ↔ circle_equation x y :=
by sorry

end NUMINAMATH_CALUDE_cookie_is_circle_with_radius_nine_l559_55910


namespace NUMINAMATH_CALUDE_abs_of_negative_three_l559_55920

theorem abs_of_negative_three :
  ∀ x : ℝ, x = -3 → |x| = 3 := by
  sorry

end NUMINAMATH_CALUDE_abs_of_negative_three_l559_55920


namespace NUMINAMATH_CALUDE_inequality_implies_k_bound_l559_55950

theorem inequality_implies_k_bound :
  (∃ x : ℝ, |x + 1| - |x - 2| < k) → k > -3 := by
  sorry

end NUMINAMATH_CALUDE_inequality_implies_k_bound_l559_55950


namespace NUMINAMATH_CALUDE_employee_savings_l559_55974

/-- Calculate the combined savings of three employees after four weeks -/
theorem employee_savings (hourly_wage : ℚ) (hours_per_day : ℕ) (days_per_week : ℕ) (num_weeks : ℕ)
  (robby_save_ratio : ℚ) (jaylen_save_ratio : ℚ) (miranda_save_ratio : ℚ)
  (h1 : hourly_wage = 10)
  (h2 : hours_per_day = 10)
  (h3 : days_per_week = 5)
  (h4 : num_weeks = 4)
  (h5 : robby_save_ratio = 2/5)
  (h6 : jaylen_save_ratio = 3/5)
  (h7 : miranda_save_ratio = 1/2) :
  (hourly_wage * hours_per_day * days_per_week * num_weeks) *
  (robby_save_ratio + jaylen_save_ratio + miranda_save_ratio) = 3000 := by
  sorry


end NUMINAMATH_CALUDE_employee_savings_l559_55974


namespace NUMINAMATH_CALUDE_rectangle_placement_l559_55986

theorem rectangle_placement (a b c d : ℝ) 
  (h1 : a < c) (h2 : c ≤ d) (h3 : d < b) (h4 : a * b < c * d) :
  (∃ (x y : ℝ), x ≤ c ∧ y ≤ d ∧ x * y = a * b) ↔ 
  (b^2 - a^2)^2 ≤ (b*c - a*d)^2 + (b*d - a*c)^2 := by
sorry

end NUMINAMATH_CALUDE_rectangle_placement_l559_55986


namespace NUMINAMATH_CALUDE_intersection_range_l559_55970

-- Define the curve C
def C (x y : ℝ) : Prop :=
  (x ≥ 0 ∧ y^2 = 4*x) ∨ (x ≤ 0 ∧ y = 0)

-- Define the line segment AB
def lineAB (a x y : ℝ) : Prop :=
  y = x + 1 ∧ x ≥ a - 1 ∧ x ≤ a

-- Theorem statement
theorem intersection_range (a : ℝ) :
  (∃! p : ℝ × ℝ, C p.1 p.2 ∧ lineAB a p.1 p.2) →
  a ∈ Set.Icc (-1) 0 ∪ Set.Icc 1 2 :=
sorry

end NUMINAMATH_CALUDE_intersection_range_l559_55970


namespace NUMINAMATH_CALUDE_fraction_simplification_l559_55964

theorem fraction_simplification (N : ℕ) :
  (Nat.factorial (N - 2) * (N - 1) * N) / Nat.factorial (N + 2) = 1 / ((N + 1) * (N + 2)) :=
by sorry

end NUMINAMATH_CALUDE_fraction_simplification_l559_55964


namespace NUMINAMATH_CALUDE_kylie_coins_problem_l559_55977

/-- The number of coins Kylie's father gave her -/
def coins_from_father : ℕ := sorry

theorem kylie_coins_problem : coins_from_father = 8 := by
  have piggy_bank : ℕ := 15
  have from_brother : ℕ := 13
  have given_to_laura : ℕ := 21
  have left_with : ℕ := 15
  
  have total_before_father : ℕ := piggy_bank + from_brother
  have total_after_father : ℕ := total_before_father + coins_from_father
  have after_giving_to_laura : ℕ := total_after_father - given_to_laura
  
  have : after_giving_to_laura = left_with := by sorry
  
  sorry

end NUMINAMATH_CALUDE_kylie_coins_problem_l559_55977

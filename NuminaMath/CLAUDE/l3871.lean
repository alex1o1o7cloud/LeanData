import Mathlib

namespace NUMINAMATH_CALUDE_system_solutions_l3871_387150

theorem system_solutions :
  let solutions := [(2, 1), (0, -3), (-6, 9)]
  ∀ (x y : ℝ),
    (x + |y| = 3 ∧ 2*|x| - y = 3) ↔ (x, y) ∈ solutions := by
  sorry

end NUMINAMATH_CALUDE_system_solutions_l3871_387150


namespace NUMINAMATH_CALUDE_fuel_consumption_rate_l3871_387183

/-- Given a plane with a certain amount of fuel and remaining flight time,
    calculate the rate of fuel consumption per hour. -/
theorem fuel_consumption_rate (fuel_left : ℝ) (time_left : ℝ) :
  fuel_left = 6.3333 →
  time_left = 0.6667 →
  ∃ (rate : ℝ), abs (rate - (fuel_left / time_left)) < 0.01 ∧ abs (rate - 9.5) < 0.01 :=
by sorry

end NUMINAMATH_CALUDE_fuel_consumption_rate_l3871_387183


namespace NUMINAMATH_CALUDE_new_girl_weight_l3871_387112

/-- Proves that the weight of a new girl is 80 kg given the conditions of the problem -/
theorem new_girl_weight (n : ℕ) (initial_weight total_weight : ℝ) :
  n = 20 →
  initial_weight = 40 →
  (total_weight - initial_weight + 80) / n = total_weight / n + 2 →
  80 = total_weight - initial_weight + 40 :=
by sorry

end NUMINAMATH_CALUDE_new_girl_weight_l3871_387112


namespace NUMINAMATH_CALUDE_max_y_value_l3871_387199

theorem max_y_value (x y : ℤ) (h : x * y + 7 * x + 6 * y = -8) :
  y ≤ 27 ∧ ∃ (x₀ : ℤ), x₀ * 27 + 7 * x₀ + 6 * 27 = -8 :=
sorry

end NUMINAMATH_CALUDE_max_y_value_l3871_387199


namespace NUMINAMATH_CALUDE_intersection_with_complement_l3871_387148

def U : Set ℕ := {1, 3, 5, 7}
def A : Set ℕ := {3, 5}
def B : Set ℕ := {1, 3, 7}

theorem intersection_with_complement : A ∩ (U \ B) = {5} := by sorry

end NUMINAMATH_CALUDE_intersection_with_complement_l3871_387148


namespace NUMINAMATH_CALUDE_mirror_area_l3871_387109

theorem mirror_area (frame_width frame_height frame_thickness : ℕ) : 
  frame_width = 100 ∧ frame_height = 120 ∧ frame_thickness = 15 →
  (frame_width - 2 * frame_thickness) * (frame_height - 2 * frame_thickness) = 6300 :=
by sorry

end NUMINAMATH_CALUDE_mirror_area_l3871_387109


namespace NUMINAMATH_CALUDE_brother_money_distribution_l3871_387110

theorem brother_money_distribution (older_initial younger_initial difference transfer : ℕ) :
  older_initial = 2800 →
  younger_initial = 1500 →
  difference = 360 →
  transfer = 470 →
  (older_initial - transfer) = (younger_initial + transfer + difference) :=
by
  sorry

end NUMINAMATH_CALUDE_brother_money_distribution_l3871_387110


namespace NUMINAMATH_CALUDE_triangle_properties_l3871_387175

/-- Theorem about a specific triangle ABC -/
theorem triangle_properties (A B C : ℝ) (a b c : ℝ) :
  -- Triangle ABC exists with sides a, b, c opposite to angles A, B, C
  0 < A ∧ A < π ∧ 0 < B ∧ B < π ∧ 0 < C ∧ C < π →
  0 < a ∧ 0 < b ∧ 0 < c →
  a + b > c ∧ b + c > a ∧ c + a > b →
  -- Law of sines
  a / Real.sin A = b / Real.sin B →
  b / Real.sin B = c / Real.sin C →
  -- Given conditions
  (2 * c - a) * Real.cos B = b * Real.cos A →
  3 * a + b = 2 * c →
  b = 2 →
  1 / Real.sin A + 1 / Real.sin C = 4 * Real.sqrt 3 / 3 →
  -- Conclusions
  Real.cos C = -1/7 ∧ 
  (1/2 * a * c * Real.sin B : ℝ) = Real.sqrt 3 :=
by sorry

end NUMINAMATH_CALUDE_triangle_properties_l3871_387175


namespace NUMINAMATH_CALUDE_greatest_digit_sum_base_nine_l3871_387185

/-- 
Given a positive integer n less than 5000, returns the sum of digits
in its base-nine representation.
-/
def sumOfDigitsBaseNine (n : ℕ) : ℕ := sorry

/-- 
The greatest possible sum of the digits in the base-nine representation
of a positive integer less than 5000.
-/
def maxDigitSum : ℕ := 26

theorem greatest_digit_sum_base_nine :
  ∀ n : ℕ, n < 5000 → sumOfDigitsBaseNine n ≤ maxDigitSum :=
sorry

end NUMINAMATH_CALUDE_greatest_digit_sum_base_nine_l3871_387185


namespace NUMINAMATH_CALUDE_fraction_of_powers_equals_3125_l3871_387129

theorem fraction_of_powers_equals_3125 : (125000 ^ 5) / (25000 ^ 5) = 3125 := by
  sorry

end NUMINAMATH_CALUDE_fraction_of_powers_equals_3125_l3871_387129


namespace NUMINAMATH_CALUDE_quadratic_decreasing_after_vertex_l3871_387142

def f (x : ℝ) : ℝ := -(x - 2)^2 - 7

theorem quadratic_decreasing_after_vertex :
  ∀ (x1 x2 : ℝ), x1 > 2 → x2 > x1 → f x2 < f x1 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_decreasing_after_vertex_l3871_387142


namespace NUMINAMATH_CALUDE_max_distance_complex_circle_l3871_387168

theorem max_distance_complex_circle (z : ℂ) (h : Complex.abs (z + 2 - 2*I) = 1) :
  ∃ (max_val : ℝ), max_val = 4 ∧ ∀ w : ℂ, Complex.abs (w + 2 - 2*I) = 1 → Complex.abs (w - 1 - 2*I) ≤ max_val :=
sorry

end NUMINAMATH_CALUDE_max_distance_complex_circle_l3871_387168


namespace NUMINAMATH_CALUDE_cookies_remaining_cookies_remaining_result_l3871_387166

/-- Calculates the number of cookies remaining in Cristian's jar --/
theorem cookies_remaining (initial_white : ℕ) (black_white_diff : ℕ) : ℕ :=
  let initial_black := initial_white + black_white_diff
  let remaining_white := initial_white - (3 * initial_white / 4)
  let remaining_black := initial_black - (initial_black / 2)
  remaining_white + remaining_black

/-- Proves that the number of cookies remaining is 85 --/
theorem cookies_remaining_result : cookies_remaining 80 50 = 85 := by
  sorry

end NUMINAMATH_CALUDE_cookies_remaining_cookies_remaining_result_l3871_387166


namespace NUMINAMATH_CALUDE_square_area_proof_l3871_387173

theorem square_area_proof (x : ℝ) : 
  (5 * x - 21 = 36 - 4 * x) → 
  (5 * x - 21)^2 = 113.4225 := by
  sorry

end NUMINAMATH_CALUDE_square_area_proof_l3871_387173


namespace NUMINAMATH_CALUDE_pen_price_relationship_l3871_387152

/-- Represents the relationship between the number of pens and their selling price. -/
theorem pen_price_relationship (x y : ℝ) : 
  (∀ (box_pens : ℝ) (box_price : ℝ), box_pens = 10 ∧ box_price = 16 → 
    y = (box_price / box_pens) * x) → 
  y = 1.6 * x := by
  sorry

end NUMINAMATH_CALUDE_pen_price_relationship_l3871_387152


namespace NUMINAMATH_CALUDE_moles_of_HCN_l3871_387117

-- Define the reaction components
structure Reaction where
  CuSO4 : ℝ
  HCN : ℝ
  Cu_CN_2 : ℝ
  H2SO4 : ℝ

-- Define the balanced equation
def balanced_equation (r : Reaction) : Prop :=
  r.CuSO4 = r.Cu_CN_2 ∧ r.HCN = 4 * r.CuSO4 ∧ r.H2SO4 = r.CuSO4

-- Define the given conditions
def given_conditions (r : Reaction) : Prop :=
  r.CuSO4 = 1 ∧ r.Cu_CN_2 = 1

-- Theorem to prove
theorem moles_of_HCN (r : Reaction) 
  (h1 : balanced_equation r) 
  (h2 : given_conditions r) : 
  r.HCN = 4 :=
sorry

end NUMINAMATH_CALUDE_moles_of_HCN_l3871_387117


namespace NUMINAMATH_CALUDE_integral_evaluation_l3871_387119

open Real

theorem integral_evaluation : 
  ∫ (x : ℝ) in Real.arccos (1 / Real.sqrt 10)..Real.arccos (1 / Real.sqrt 26), 
    12 / ((6 + 5 * tan x) * sin (2 * x)) = log (105 / 93) := by
  sorry

end NUMINAMATH_CALUDE_integral_evaluation_l3871_387119


namespace NUMINAMATH_CALUDE_dhoni_leftover_earnings_l3871_387159

def rent_percent : ℝ := 20
def dishwasher_percent : ℝ := 15
def bills_percent : ℝ := 10
def car_percent : ℝ := 8
def grocery_percent : ℝ := 12
def tax_percent : ℝ := 5
def savings_percent : ℝ := 40

theorem dhoni_leftover_earnings : 
  let total_expenses := rent_percent + dishwasher_percent + bills_percent + car_percent + grocery_percent + tax_percent
  let remaining_after_expenses := 100 - total_expenses
  let savings := (savings_percent / 100) * remaining_after_expenses
  let leftover := remaining_after_expenses - savings
  leftover = 18 := by sorry

end NUMINAMATH_CALUDE_dhoni_leftover_earnings_l3871_387159


namespace NUMINAMATH_CALUDE_range_of_a_l3871_387163

theorem range_of_a (a : ℝ) : 
  (∀ x, x ∈ {x : ℝ | -4*x + 4*a < 0} → x ≠ 2) → 
  a ∈ {x : ℝ | x ≥ 2} := by
  sorry

end NUMINAMATH_CALUDE_range_of_a_l3871_387163


namespace NUMINAMATH_CALUDE_objects_meet_time_l3871_387167

/-- Two objects moving towards each other meet after 10 seconds -/
theorem objects_meet_time : ∃ t : ℝ, t = 10 ∧ 
  390 = 3 * t^2 + 0.012 * (t - 5) := by sorry

end NUMINAMATH_CALUDE_objects_meet_time_l3871_387167


namespace NUMINAMATH_CALUDE_remainder_of_second_division_l3871_387195

def p (x : ℝ) : ℝ := x^6 - 4*x^5 + 6*x^4 - 4*x^3 + x^2

def s1 (x : ℝ) : ℝ := x^5 - 3*x^4 + 3*x^3 - x^2

def t1 : ℝ := p 1

def t2 : ℝ := s1 1

theorem remainder_of_second_division (x : ℝ) : t2 = 0 := by
  sorry

end NUMINAMATH_CALUDE_remainder_of_second_division_l3871_387195


namespace NUMINAMATH_CALUDE_log_equation_solution_l3871_387136

theorem log_equation_solution (y : ℝ) (h : y > 0) :
  (Real.log y^3 / Real.log 3) + (Real.log y / Real.log (1/3)) = 6 → y = 27 := by
  sorry

end NUMINAMATH_CALUDE_log_equation_solution_l3871_387136


namespace NUMINAMATH_CALUDE_reciprocal_inequality_l3871_387156

theorem reciprocal_inequality (a b : ℝ) (h1 : a < b) (h2 : b < 0) : 1 / a > 1 / b := by
  sorry

end NUMINAMATH_CALUDE_reciprocal_inequality_l3871_387156


namespace NUMINAMATH_CALUDE_sodaCans_theorem_l3871_387189

/-- The number of cans of soda that can be bought for a given amount of euros -/
def sodaCans (S Q E : ℚ) : ℚ :=
  10 * E * S / Q

/-- Theorem stating that the number of cans of soda that can be bought for E euros
    is equal to 10ES/Q, given that S cans can be purchased for Q dimes and
    1 euro is equivalent to 10 dimes -/
theorem sodaCans_theorem (S Q E : ℚ) (hS : S > 0) (hQ : Q > 0) (hE : E ≥ 0) :
  sodaCans S Q E = 10 * E * S / Q :=
by sorry

end NUMINAMATH_CALUDE_sodaCans_theorem_l3871_387189


namespace NUMINAMATH_CALUDE_cloth_cost_price_calculation_l3871_387143

/-- The cost price of one metre of cloth, given the selling details --/
def cost_price_per_metre (cloth_length : ℕ) (selling_price : ℚ) (profit_per_metre : ℚ) : ℚ :=
  (selling_price - cloth_length * profit_per_metre) / cloth_length

theorem cloth_cost_price_calculation :
  let cloth_length : ℕ := 92
  let selling_price : ℚ := 9890
  let profit_per_metre : ℚ := 24
  cost_price_per_metre cloth_length selling_price profit_per_metre = 83.5 := by
sorry


end NUMINAMATH_CALUDE_cloth_cost_price_calculation_l3871_387143


namespace NUMINAMATH_CALUDE_positive_integer_solutions_inequality_l3871_387170

theorem positive_integer_solutions_inequality (x : ℕ+) : 
  (2 * x.val - 3 ≤ 5) ↔ x ∈ ({1, 2, 3, 4} : Set ℕ+) := by
  sorry

end NUMINAMATH_CALUDE_positive_integer_solutions_inequality_l3871_387170


namespace NUMINAMATH_CALUDE_intersection_M_N_l3871_387155

def M : Set ℝ := {x | x^2 - 2*x < 0}
def N : Set ℝ := {x | |x| < 1}

theorem intersection_M_N : M ∩ N = Set.Ioo 0 1 := by sorry

end NUMINAMATH_CALUDE_intersection_M_N_l3871_387155


namespace NUMINAMATH_CALUDE_sphere_surface_area_l3871_387153

theorem sphere_surface_area (V : ℝ) (h : V = 72 * Real.pi) :
  ∃ (r : ℝ), V = (4 / 3) * Real.pi * r^3 ∧
              4 * Real.pi * r^2 = 36 * Real.pi * 2^(2/3) := by
  sorry

end NUMINAMATH_CALUDE_sphere_surface_area_l3871_387153


namespace NUMINAMATH_CALUDE_golden_ratio_solution_l3871_387107

theorem golden_ratio_solution (x : ℝ) :
  x > 0 ∧ x = Real.sqrt (x - 1 / x) + Real.sqrt (1 - 1 / x) ↔ x = (1 + Real.sqrt 5) / 2 := by
  sorry

end NUMINAMATH_CALUDE_golden_ratio_solution_l3871_387107


namespace NUMINAMATH_CALUDE_compute_expression_l3871_387102

theorem compute_expression : 6^2 - 4*5 + 4^2 = 32 := by
  sorry

end NUMINAMATH_CALUDE_compute_expression_l3871_387102


namespace NUMINAMATH_CALUDE_prob_not_green_correct_l3871_387192

/-- Given odds for pulling a green marble from a bag -/
def green_marble_odds : ℚ := 5 / 6

/-- The probability of not pulling a green marble -/
def prob_not_green : ℚ := 6 / 11

/-- Theorem stating that given the odds for pulling a green marble,
    the probability of not pulling a green marble is correct -/
theorem prob_not_green_correct :
  green_marble_odds = 5 / 6 →
  prob_not_green = 6 / 11 := by
sorry

end NUMINAMATH_CALUDE_prob_not_green_correct_l3871_387192


namespace NUMINAMATH_CALUDE_ellipse_hyperbola_coinciding_foci_l3871_387174

/-- The value of b^2 for an ellipse and hyperbola with coinciding foci -/
theorem ellipse_hyperbola_coinciding_foci (b : ℝ) : 
  (∃ (x y : ℝ), x^2/25 + y^2/b^2 = 1) ∧ 
  (∃ (x y : ℝ), x^2/169 - y^2/144 = 1/36) ∧
  (∀ (x y : ℝ), x^2/25 + y^2/b^2 = 1 ↔ x^2/169 - y^2/144 = 1/36) →
  b^2 = 587/36 := by
sorry

end NUMINAMATH_CALUDE_ellipse_hyperbola_coinciding_foci_l3871_387174


namespace NUMINAMATH_CALUDE_intersection_with_complement_l3871_387121

open Set

def U : Finset ℕ := {0, 1, 2, 3, 4, 5}
def A : Finset ℕ := {0, 1, 3}
def B : Finset ℕ := {2, 3, 5}

theorem intersection_with_complement :
  A ∩ (U \ B) = {0, 1} := by sorry

end NUMINAMATH_CALUDE_intersection_with_complement_l3871_387121


namespace NUMINAMATH_CALUDE_commute_time_difference_l3871_387190

/-- A set of five commute times with specific properties -/
structure CommuteTimes where
  x : ℝ
  y : ℝ
  average : ℝ
  variance : ℝ
  average_eq : (x + y + 10 + 11 + 9) / 5 = average
  variance_eq : ((x - average)^2 + (y - average)^2 + (10 - average)^2 + (11 - average)^2 + (9 - average)^2) / 5 = variance

/-- The theorem stating that for the given commute times, |x-y| = 4 -/
theorem commute_time_difference (ct : CommuteTimes) (h1 : ct.average = 10) (h2 : ct.variance = 2) : 
  |ct.x - ct.y| = 4 := by
  sorry

end NUMINAMATH_CALUDE_commute_time_difference_l3871_387190


namespace NUMINAMATH_CALUDE_point_on_graph_l3871_387160

def f (x : ℝ) : ℝ := |x^3 + 1| + |x^3 - 1|

theorem point_on_graph (a : ℝ) : (a, f (-a)) ∈ {p : ℝ × ℝ | p.2 = f p.1} := by
  sorry

end NUMINAMATH_CALUDE_point_on_graph_l3871_387160


namespace NUMINAMATH_CALUDE_overall_gain_percentage_is_10_51_l3871_387115

/-- Represents a transaction with quantity, buy price, and sell price -/
structure Transaction where
  quantity : ℕ
  buyPrice : ℚ
  sellPrice : ℚ

/-- Calculates the profit or loss for a single transaction -/
def transactionProfit (t : Transaction) : ℚ :=
  t.quantity * (t.sellPrice - t.buyPrice)

/-- Calculates the cost for a single transaction -/
def transactionCost (t : Transaction) : ℚ :=
  t.quantity * t.buyPrice

/-- Calculates the overall gain percentage for a list of transactions -/
def overallGainPercentage (transactions : List Transaction) : ℚ :=
  let totalProfit := (transactions.map transactionProfit).sum
  let totalCost := (transactions.map transactionCost).sum
  totalProfit / totalCost * 100

/-- The main theorem stating that the overall gain percentage for the given transactions is 10.51% -/
theorem overall_gain_percentage_is_10_51 :
  let transactions := [
    ⟨10, 8, 10⟩,
    ⟨7, 15, 18⟩,
    ⟨5, 22, 20⟩
  ]
  overallGainPercentage transactions = 10.51 := by
  sorry

end NUMINAMATH_CALUDE_overall_gain_percentage_is_10_51_l3871_387115


namespace NUMINAMATH_CALUDE_bowling_shoe_rental_cost_l3871_387138

/-- The cost to rent bowling shoes for a day, given the following conditions:
  1. The cost per game is $1.75.
  2. A person has $12.80 in total.
  3. The person can bowl a maximum of 7 complete games. -/
theorem bowling_shoe_rental_cost :
  let cost_per_game : ℚ := 175 / 100
  let total_money : ℚ := 1280 / 100
  let max_games : ℕ := 7
  let shoe_rental_cost : ℚ := total_money - (cost_per_game * max_games)
  shoe_rental_cost = 55 / 100 := by sorry

end NUMINAMATH_CALUDE_bowling_shoe_rental_cost_l3871_387138


namespace NUMINAMATH_CALUDE_smallest_next_divisor_l3871_387128

theorem smallest_next_divisor (m : ℕ) : 
  m % 2 = 0 ∧ 1000 ≤ m ∧ m < 10000 ∧ m % 437 = 0 → 
  ∃ (d : ℕ), d > 437 ∧ m % d = 0 ∧ d ≥ 874 ∧ 
  ∀ (d' : ℕ), d' > 437 ∧ m % d' = 0 → d' ≥ 874 :=
by sorry

end NUMINAMATH_CALUDE_smallest_next_divisor_l3871_387128


namespace NUMINAMATH_CALUDE_sum_of_repeating_decimals_l3871_387137

/-- Represents a repeating decimal with a single repeating digit -/
def single_repeat (n : ℕ) : ℚ := n / 9

/-- Represents a repeating decimal with two repeating digits -/
def double_repeat (a b : ℕ) : ℚ := (10 * a + b) / 99

/-- The sum of 0.777... and 0.131313... is equal to 10/11 -/
theorem sum_of_repeating_decimals : 
  single_repeat 7 + double_repeat 1 3 = 10 / 11 := by sorry

end NUMINAMATH_CALUDE_sum_of_repeating_decimals_l3871_387137


namespace NUMINAMATH_CALUDE_square_root_b_minus_a_l3871_387161

/-- Given that the square roots of a positive number are 2-3a and a+2,
    and the cube root of 5a+3b-1 is 3, prove that the square root of b-a is 2. -/
theorem square_root_b_minus_a (a b : ℝ) : 
  (∃ (k : ℝ), k > 0 ∧ (2 - 3*a)^2 = k ∧ (a + 2)^2 = k) →  -- square roots condition
  (5*a + 3*b - 1)^(1/3) = 3 →                             -- cube root condition
  Real.sqrt (b - a) = 2 := by
sorry

end NUMINAMATH_CALUDE_square_root_b_minus_a_l3871_387161


namespace NUMINAMATH_CALUDE_golden_ratio_from_logarithms_l3871_387106

theorem golden_ratio_from_logarithms (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  (Real.log a / Real.log 4 = Real.log b / Real.log 18) ∧ 
  (Real.log b / Real.log 18 = Real.log (a + b) / Real.log 32) →
  b / a = (1 + Real.sqrt 5) / 2 := by
  sorry

end NUMINAMATH_CALUDE_golden_ratio_from_logarithms_l3871_387106


namespace NUMINAMATH_CALUDE_probability_5_heart_ace_l3871_387162

/-- Represents a standard deck of 52 playing cards. -/
def StandardDeck : ℕ := 52

/-- Represents the number of 5s in a standard deck. -/
def NumberOf5s : ℕ := 4

/-- Represents the number of hearts in a standard deck. -/
def NumberOfHearts : ℕ := 13

/-- Represents the number of Aces in a standard deck. -/
def NumberOfAces : ℕ := 4

/-- Theorem stating the probability of drawing a 5 as the first card, 
    a heart as the second card, and an Ace as the third card from a standard 52-card deck. -/
theorem probability_5_heart_ace : 
  (NumberOf5s : ℚ) / StandardDeck * 
  NumberOfHearts / (StandardDeck - 1) * 
  NumberOfAces / (StandardDeck - 2) = 1 / 650 := by
  sorry

end NUMINAMATH_CALUDE_probability_5_heart_ace_l3871_387162


namespace NUMINAMATH_CALUDE_sqrt_product_sqrt_three_times_sqrt_two_equals_sqrt_six_l3871_387197

theorem sqrt_product (a b : ℝ) (ha : 0 ≤ a) (hb : 0 ≤ b) : 
  Real.sqrt (a * b) = Real.sqrt a * Real.sqrt b :=
by sorry

theorem sqrt_three_times_sqrt_two_equals_sqrt_six : 
  Real.sqrt 3 * Real.sqrt 2 = Real.sqrt 6 :=
by sorry

end NUMINAMATH_CALUDE_sqrt_product_sqrt_three_times_sqrt_two_equals_sqrt_six_l3871_387197


namespace NUMINAMATH_CALUDE_negation_of_proposition_negation_of_greater_than_sin_l3871_387147

theorem negation_of_proposition (p : ℝ → Prop) : 
  (¬ ∀ x : ℝ, p x) ↔ (∃ x : ℝ, ¬ p x) :=
by sorry

theorem negation_of_greater_than_sin :
  (¬ ∀ x : ℝ, x > Real.sin x) ↔ (∃ x : ℝ, x ≤ Real.sin x) :=
by sorry

end NUMINAMATH_CALUDE_negation_of_proposition_negation_of_greater_than_sin_l3871_387147


namespace NUMINAMATH_CALUDE_prime_condition_implies_result_l3871_387140

theorem prime_condition_implies_result (p : ℕ) 
  (h1 : Nat.Prime p) 
  (h2 : Nat.Prime (p^4 + 3)) : 
  p^5 + 3 = 35 := by
  sorry

end NUMINAMATH_CALUDE_prime_condition_implies_result_l3871_387140


namespace NUMINAMATH_CALUDE_buying_problem_equations_l3871_387130

theorem buying_problem_equations (x y : ℕ) : 
  x > 0 → y > 0 → (8 * x - y = 3 ∧ y - 7 * x = 4) → True := by
  sorry

end NUMINAMATH_CALUDE_buying_problem_equations_l3871_387130


namespace NUMINAMATH_CALUDE_backpack_profit_equation_l3871_387182

/-- Represents the profit calculation for a backpack sale -/
theorem backpack_profit_equation (x : ℝ) : 
  (1 + 0.5) * x * 0.8 - x = 8 ↔ 
  (x > 0 ∧ 
   (1 + 0.5) * x * 0.8 = x + 8) :=
by sorry

#check backpack_profit_equation

end NUMINAMATH_CALUDE_backpack_profit_equation_l3871_387182


namespace NUMINAMATH_CALUDE_not_tangent_implies_a_less_than_one_third_l3871_387194

/-- The function f(x) = x³ - 3ax --/
def f (a : ℝ) (x : ℝ) : ℝ := x^3 - 3*a*x

/-- The derivative of f(x) --/
def f_derivative (a : ℝ) (x : ℝ) : ℝ := 3*x^2 - 3*a

/-- Theorem stating that if the line x + y + m = 0 is not a tangent to y = f(x) for any m,
    then a < 1/3 --/
theorem not_tangent_implies_a_less_than_one_third (a : ℝ) :
  (∀ m : ℝ, ¬∃ x : ℝ, f_derivative a x = -1 ∧ f a x = -(x + m)) →
  a < 1/3 :=
sorry

end NUMINAMATH_CALUDE_not_tangent_implies_a_less_than_one_third_l3871_387194


namespace NUMINAMATH_CALUDE_set_intersection_proof_l3871_387139

def M : Set ℤ := {-1, 1, 2}
def N : Set ℤ := {1, 2, 3}

theorem set_intersection_proof : M ∩ N = {1, 2} := by
  sorry

end NUMINAMATH_CALUDE_set_intersection_proof_l3871_387139


namespace NUMINAMATH_CALUDE_x_value_when_y_is_14_l3871_387132

theorem x_value_when_y_is_14 (t : ℝ) (x y : ℝ) 
  (h1 : x = 3 - 2*t) 
  (h2 : y = 3*t + 5) 
  (h3 : y = 14) : 
  x = -3 := by
  sorry

end NUMINAMATH_CALUDE_x_value_when_y_is_14_l3871_387132


namespace NUMINAMATH_CALUDE_triangle_area_proof_l3871_387103

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

end NUMINAMATH_CALUDE_triangle_area_proof_l3871_387103


namespace NUMINAMATH_CALUDE_quadrilateral_diagonal_sum_lower_bound_l3871_387111

theorem quadrilateral_diagonal_sum_lower_bound (x y : ℝ) (α : ℝ) :
  x > 0 → y > 0 → 0 < α → α < π →
  x * y * Real.sin α = 2 →
  x + y ≥ 2 * Real.sqrt 2 := by
sorry

end NUMINAMATH_CALUDE_quadrilateral_diagonal_sum_lower_bound_l3871_387111


namespace NUMINAMATH_CALUDE_remainder_equality_l3871_387134

theorem remainder_equality (P P' D Q R R' : ℕ) 
  (h1 : P > P') 
  (h2 : R = P % D) 
  (h3 : R' = P' % D) : 
  ((P + Q) * P') % D = (R * R') % D := by
sorry

end NUMINAMATH_CALUDE_remainder_equality_l3871_387134


namespace NUMINAMATH_CALUDE_intersection_A_B_l3871_387149

def A : Set ℝ := {x | x^2 - 3*x - 4 < 0}

def B : Set ℝ := {-2, -1, 1, 2, 4}

theorem intersection_A_B : A ∩ B = {1, 2} := by sorry

end NUMINAMATH_CALUDE_intersection_A_B_l3871_387149


namespace NUMINAMATH_CALUDE_triangle_radii_relations_l3871_387180

/-- Given a triangle ABC with sides a, b, c, inradius r, exradii r_a, r_b, r_c, semi-perimeter p, and area S -/
theorem triangle_radii_relations (a b c r r_a r_b r_c p S : ℝ) 
  (h_positive : a > 0 ∧ b > 0 ∧ c > 0 ∧ r > 0 ∧ r_a > 0 ∧ r_b > 0 ∧ r_c > 0 ∧ p > 0 ∧ S > 0)
  (h_semiperimeter : p = (a + b + c) / 2)
  (h_area_inradius : S = p * r)
  (h_area_exradius_a : S = (p - a) * r_a)
  (h_area_exradius_b : S = (p - b) * r_b)
  (h_area_exradius_c : S = (p - c) * r_c) :
  (1 / r = 1 / r_a + 1 / r_b + 1 / r_c) ∧ 
  (S = Real.sqrt (r * r_a * r_b * r_c)) := by
  sorry

end NUMINAMATH_CALUDE_triangle_radii_relations_l3871_387180


namespace NUMINAMATH_CALUDE_adolfo_tower_blocks_l3871_387126

-- Define the variables
def initial_blocks : ℕ := sorry
def added_blocks : ℝ := 65.0
def total_blocks : ℕ := 100

-- State the theorem
theorem adolfo_tower_blocks : initial_blocks = 35 := by
  sorry

end NUMINAMATH_CALUDE_adolfo_tower_blocks_l3871_387126


namespace NUMINAMATH_CALUDE_train_length_l3871_387108

/-- The length of a train given its speed, time to pass a platform, and platform length. -/
theorem train_length (train_speed : ℝ) (time_to_pass : ℝ) (platform_length : ℝ) :
  train_speed = 60 →
  time_to_pass = 23.998080153587715 →
  platform_length = 260 →
  (train_speed * 1000 / 3600) * time_to_pass - platform_length = 140 := by
  sorry

#check train_length

end NUMINAMATH_CALUDE_train_length_l3871_387108


namespace NUMINAMATH_CALUDE_square_of_binomial_l3871_387120

theorem square_of_binomial (x : ℝ) : (7 - Real.sqrt (x^2 - 33))^2 = x^2 - 14 * Real.sqrt (x^2 - 33) + 16 := by
  sorry

end NUMINAMATH_CALUDE_square_of_binomial_l3871_387120


namespace NUMINAMATH_CALUDE_three_fifths_of_ten_x_minus_three_l3871_387104

theorem three_fifths_of_ten_x_minus_three (x : ℝ) : 
  (3 / 5) * (10 * x - 3) = 6 * x - 9 / 5 := by
  sorry

end NUMINAMATH_CALUDE_three_fifths_of_ten_x_minus_three_l3871_387104


namespace NUMINAMATH_CALUDE_smallest_nonneg_minus_opposite_largest_neg_l3871_387172

theorem smallest_nonneg_minus_opposite_largest_neg : ∃ a b : ℤ,
  (∀ x : ℤ, x ≥ 0 → a ≤ x) ∧
  (∀ y : ℤ, y < 0 → y ≤ -b) ∧
  (a - b = 1) := by
  sorry

end NUMINAMATH_CALUDE_smallest_nonneg_minus_opposite_largest_neg_l3871_387172


namespace NUMINAMATH_CALUDE_power_three_mod_five_l3871_387127

theorem power_three_mod_five : 3^2023 % 5 = 2 := by
  sorry

end NUMINAMATH_CALUDE_power_three_mod_five_l3871_387127


namespace NUMINAMATH_CALUDE_soda_crates_count_l3871_387154

def bridge_weight_limit : ℕ := 20000
def empty_truck_weight : ℕ := 12000
def soda_crate_weight : ℕ := 50
def num_dryers : ℕ := 3
def dryer_weight : ℕ := 3000
def loaded_truck_weight : ℕ := 24000

def calculate_soda_crates (bridge_weight_limit empty_truck_weight soda_crate_weight 
                           num_dryers dryer_weight loaded_truck_weight : ℕ) : ℕ := 
  let total_dryer_weight := num_dryers * dryer_weight
  let remaining_weight := loaded_truck_weight - empty_truck_weight - total_dryer_weight
  let soda_weight := remaining_weight / 3
  soda_weight / soda_crate_weight

theorem soda_crates_count : 
  calculate_soda_crates bridge_weight_limit empty_truck_weight soda_crate_weight 
                         num_dryers dryer_weight loaded_truck_weight = 20 := by
  sorry

end NUMINAMATH_CALUDE_soda_crates_count_l3871_387154


namespace NUMINAMATH_CALUDE_pentagon_triangle_side_ratio_l3871_387158

theorem pentagon_triangle_side_ratio :
  ∀ (t p : ℝ),
    t > 0 ∧ p > 0 →
    3 * t = 15 →
    5 * p = 15 →
    t / p = 5 / 3 := by
  sorry

end NUMINAMATH_CALUDE_pentagon_triangle_side_ratio_l3871_387158


namespace NUMINAMATH_CALUDE_stratified_sampling_problem_l3871_387118

/-- Represents the number of students to be drawn from a stratum -/
structure SampleSize (total : ℕ) (stratum : ℕ) (drawn : ℕ) where
  size : ℕ
  proportional : size * total = stratum * drawn

/-- The problem statement -/
theorem stratified_sampling_problem :
  let total_students : ℕ := 1400
  let male_students : ℕ := 800
  let female_students : ℕ := 600
  let male_drawn : ℕ := 40
  ∃ (female_sample : SampleSize total_students female_students male_drawn),
    female_sample.size = 30 := by
  sorry

end NUMINAMATH_CALUDE_stratified_sampling_problem_l3871_387118


namespace NUMINAMATH_CALUDE_factory_production_time_l3871_387198

/-- The number of dolls produced by the factory -/
def num_dolls : ℕ := 12000

/-- The number of shoes per doll -/
def shoes_per_doll : ℕ := 2

/-- The number of bags per doll -/
def bags_per_doll : ℕ := 3

/-- The number of cosmetics sets per doll -/
def cosmetics_per_doll : ℕ := 1

/-- The number of hats per doll -/
def hats_per_doll : ℕ := 5

/-- The time in seconds to make one doll -/
def time_per_doll : ℕ := 45

/-- The time in seconds to make one accessory -/
def time_per_accessory : ℕ := 10

/-- The total combined machine operation time for manufacturing all dolls and accessories -/
def total_time : ℕ := 1860000

theorem factory_production_time : 
  num_dolls * time_per_doll + 
  num_dolls * (shoes_per_doll + bags_per_doll + cosmetics_per_doll + hats_per_doll) * time_per_accessory = 
  total_time := by sorry

end NUMINAMATH_CALUDE_factory_production_time_l3871_387198


namespace NUMINAMATH_CALUDE_quadratic_roots_condition_l3871_387100

theorem quadratic_roots_condition (a : ℝ) :
  (∃ x y : ℝ, x ≠ y ∧ a * x^2 - x + 2 = 0 ∧ a * y^2 - y + 2 = 0) ↔ (a < 1/8 ∧ a ≠ 0) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_roots_condition_l3871_387100


namespace NUMINAMATH_CALUDE_sum_odd_integers_mod_12_l3871_387124

/-- The sum of the first n odd positive integers -/
def sum_odd_integers (n : ℕ) : ℕ := n * n

/-- The theorem stating that the remainder when the sum of the first 10 odd positive integers 
    is divided by 12 is equal to 4 -/
theorem sum_odd_integers_mod_12 : sum_odd_integers 10 % 12 = 4 := by
  sorry

end NUMINAMATH_CALUDE_sum_odd_integers_mod_12_l3871_387124


namespace NUMINAMATH_CALUDE_hyperbola_equation_l3871_387191

-- Define the ellipse D
def ellipse_D (x y : ℝ) : Prop := x^2 / 50 + y^2 / 25 = 1

-- Define the circle M
def circle_M (x y : ℝ) : Prop := x^2 + (y - 5)^2 = 9

-- Define the hyperbola G
def hyperbola_G (x y : ℝ) : Prop := x^2 / 9 - y^2 / 16 = 1

-- Define the foci of ellipse D
def foci_D : Set (ℝ × ℝ) := {(-5, 0), (5, 0)}

-- Theorem statement
theorem hyperbola_equation :
  ∀ x y : ℝ,
  (∀ x' y' : ℝ, ellipse_D x' y' → foci_D = {(-5, 0), (5, 0)}) →
  (∀ x' y' : ℝ, hyperbola_G x' y' → foci_D = {(-5, 0), (5, 0)}) →
  (∃ a b : ℝ, ∀ x' y' : ℝ, (b * x' = a * y' ∨ b * x' = -a * y') →
    ∃ t : ℝ, circle_M (x' + t) (y' + t)) →
  hyperbola_G x y := by sorry

end NUMINAMATH_CALUDE_hyperbola_equation_l3871_387191


namespace NUMINAMATH_CALUDE_mary_hourly_wage_l3871_387144

def mary_long_day_hours : ℕ := 9
def mary_short_day_hours : ℕ := 5
def mary_long_days_per_week : ℕ := 3
def mary_short_days_per_week : ℕ := 2
def mary_weekly_earnings : ℕ := 407

def mary_total_weekly_hours : ℕ :=
  mary_long_day_hours * mary_long_days_per_week +
  mary_short_day_hours * mary_short_days_per_week

theorem mary_hourly_wage :
  mary_weekly_earnings / mary_total_weekly_hours = 11 := by
  sorry

end NUMINAMATH_CALUDE_mary_hourly_wage_l3871_387144


namespace NUMINAMATH_CALUDE_valid_base5_number_l3871_387114

def is_base5_digit (d : Nat) : Prop := d ≤ 4

def is_base5_number (n : Nat) : Prop :=
  ∀ d, d ∈ n.digits 5 → is_base5_digit d

theorem valid_base5_number : is_base5_number 2134 := by sorry

end NUMINAMATH_CALUDE_valid_base5_number_l3871_387114


namespace NUMINAMATH_CALUDE_marked_percentage_above_cost_price_l3871_387176

/-- Proves that for an article with given cost price, selling price, and discount percentage,
    the marked percentage above the cost price is correct. -/
theorem marked_percentage_above_cost_price
  (cost_price : ℝ)
  (selling_price : ℝ)
  (discount_percentage : ℝ)
  (h1 : cost_price = 540)
  (h2 : selling_price = 496.80)
  (h3 : discount_percentage = 19.999999999999996)
  : (((selling_price / (1 - discount_percentage / 100) - cost_price) / cost_price) * 100 = 15) := by
  sorry

end NUMINAMATH_CALUDE_marked_percentage_above_cost_price_l3871_387176


namespace NUMINAMATH_CALUDE_computer_table_price_l3871_387133

/-- The selling price of an item given its cost price and markup percentage -/
def selling_price (cost : ℚ) (markup : ℚ) : ℚ :=
  cost * (1 + markup / 100)

/-- Theorem: The selling price of a computer table with cost price 6925 and 24% markup is 8587 -/
theorem computer_table_price : selling_price 6925 24 = 8587 := by
  sorry

end NUMINAMATH_CALUDE_computer_table_price_l3871_387133


namespace NUMINAMATH_CALUDE_fertilizer_mixture_problem_l3871_387164

/-- Given two fertilizer solutions, one with unknown percentage P and another with 53%,
    mixed to form 42 liters of 63% solution, where 20 liters of the first solution were used,
    prove that the percentage of fertilizer in the first solution is 74%. -/
theorem fertilizer_mixture_problem (P : ℝ) : 
  (20 * P / 100 + 22 * 53 / 100 = 42 * 63 / 100) → P = 74 := by
  sorry

end NUMINAMATH_CALUDE_fertilizer_mixture_problem_l3871_387164


namespace NUMINAMATH_CALUDE_jackson_sandwiches_l3871_387141

/-- The number of peanut butter and jelly sandwiches Jackson ate during the school year -/
def sandwiches_eaten (weeks : ℕ) (missed_wednesdays : ℕ) (missed_fridays : ℕ) : ℕ :=
  (weeks - missed_wednesdays) + (weeks - missed_fridays)

/-- Theorem stating that Jackson ate 69 sandwiches given the problem conditions -/
theorem jackson_sandwiches : 
  sandwiches_eaten 36 1 2 = 69 := by
  sorry

end NUMINAMATH_CALUDE_jackson_sandwiches_l3871_387141


namespace NUMINAMATH_CALUDE_hyperbola_eccentricity_range_l3871_387179

-- Define the hyperbola and its properties
structure Hyperbola where
  a : ℝ
  b : ℝ
  c : ℝ
  equation : ∀ x y, x^2 / a^2 - y^2 / b^2 = 1

-- Define the asymptotes and intersection points
def asymptote_intersections (h : Hyperbola) (x : ℝ) : Set (ℝ × ℝ) :=
  {(x, y) | x = h.a^2 / h.c ∧ (y = h.b * x / h.a ∨ y = -h.b * x / h.a)}

-- Define the angle AFB
def angle_AFB (h : Hyperbola) (A B : ℝ × ℝ) : ℝ := sorry

-- Define the eccentricity
def eccentricity (h : Hyperbola) : ℝ := sorry

-- State the theorem
theorem hyperbola_eccentricity_range (h : Hyperbola) 
  (A B : ℝ × ℝ) (hA : A ∈ asymptote_intersections h (h.a^2 / h.c)) 
  (hB : B ∈ asymptote_intersections h (h.a^2 / h.c))
  (hAngle : π/3 < angle_AFB h A B ∧ angle_AFB h A B < π/2) :
  Real.sqrt 2 < eccentricity h ∧ eccentricity h < 2 := by
  sorry

end NUMINAMATH_CALUDE_hyperbola_eccentricity_range_l3871_387179


namespace NUMINAMATH_CALUDE_green_balls_count_l3871_387165

theorem green_balls_count (blue_count : ℕ) (ratio_blue : ℕ) (ratio_green : ℕ) (green_count : ℕ) : 
  blue_count = 20 → 
  ratio_blue = 5 → 
  ratio_green = 3 → 
  blue_count * ratio_green = green_count * ratio_blue → 
  green_count = 12 := by
sorry

end NUMINAMATH_CALUDE_green_balls_count_l3871_387165


namespace NUMINAMATH_CALUDE_ittymangnark_catch_l3871_387187

/-- Represents the number of fish each family member and pet receives -/
structure FishDistribution where
  ittymangnark : ℕ
  kingnook : ℕ
  oomyapeck : ℕ
  yurraknalik : ℕ
  ankaq : ℕ
  nanuq : ℕ

/-- Represents the distribution of fish eyes -/
structure EyeDistribution where
  oomyapeck : ℕ
  yurraknalik : ℕ
  ankaq : ℕ
  nanuq : ℕ

/-- Theorem stating that given the fish and eye distribution, Ittymangnark caught 21 fish -/
theorem ittymangnark_catch (fish : FishDistribution) (eyes : EyeDistribution) :
  fish.ittymangnark = 3 →
  fish.kingnook = 4 →
  fish.oomyapeck = 1 →
  fish.yurraknalik = 2 →
  fish.ankaq = 1 →
  fish.nanuq = 3 →
  eyes.oomyapeck = 24 →
  eyes.yurraknalik = 4 →
  eyes.ankaq = 6 →
  eyes.nanuq = 8 →
  fish.ittymangnark + fish.kingnook + fish.oomyapeck + fish.yurraknalik + fish.ankaq + fish.nanuq = 21 :=
by
  sorry

end NUMINAMATH_CALUDE_ittymangnark_catch_l3871_387187


namespace NUMINAMATH_CALUDE_solution_set_l3871_387116

theorem solution_set (m : ℤ) 
  (h1 : ∃! (x : ℤ), |2*x - m| ≤ 1 ∧ x = 2) :
  {x : ℝ | |x - 1| + |x - 3| ≥ m} = 
    {x : ℝ | x ≤ 0} ∪ {x : ℝ | x ≥ 4} := by
  sorry

end NUMINAMATH_CALUDE_solution_set_l3871_387116


namespace NUMINAMATH_CALUDE_line_parabola_intersection_l3871_387171

/-- A line with equation y = kx + 1 -/
structure Line (k : ℝ) where
  eq : ℝ → ℝ
  h : ∀ x, eq x = k * x + 1

/-- A parabola with equation y² = 4x -/
structure Parabola where
  eq : ℝ → ℝ → Prop
  h : ∀ x y, eq x y ↔ y^2 = 4*x

/-- The number of intersection points between a line and a parabola -/
def intersectionCount (l : Line k) (p : Parabola) : ℕ :=
  sorry

theorem line_parabola_intersection (k : ℝ) (l : Line k) (p : Parabola) :
  intersectionCount l p = 1 → k = 0 ∨ k = 1 :=
sorry

end NUMINAMATH_CALUDE_line_parabola_intersection_l3871_387171


namespace NUMINAMATH_CALUDE_ratio_of_sum_and_difference_l3871_387188

theorem ratio_of_sum_and_difference (x y : ℝ) : 
  x > 0 → y > 0 → x > y → x + y = 7 * (x - y) → x / y = 4 / 3 := by
  sorry

end NUMINAMATH_CALUDE_ratio_of_sum_and_difference_l3871_387188


namespace NUMINAMATH_CALUDE_AB_squared_is_8_l3871_387113

/-- The parabola equation -/
def parabola (x y : ℝ) : Prop := y = 3 * x^2 + 4 * x + 2

/-- Point A on the parabola -/
def A : ℝ × ℝ := sorry

/-- Point B on the parabola -/
def B : ℝ × ℝ := sorry

/-- The origin is the midpoint of AB -/
axiom origin_is_midpoint : (A.1 + B.1) / 2 = 0 ∧ (A.2 + B.2) / 2 = 0

/-- A and B are on the parabola -/
axiom A_on_parabola : parabola A.1 A.2
axiom B_on_parabola : parabola B.1 B.2

/-- The square of the length of AB -/
def AB_squared : ℝ := (A.1 - B.1)^2 + (A.2 - B.2)^2

/-- Theorem: The square of the length of AB is 8 -/
theorem AB_squared_is_8 : AB_squared = 8 := sorry

end NUMINAMATH_CALUDE_AB_squared_is_8_l3871_387113


namespace NUMINAMATH_CALUDE_number_of_factors_27648_l3871_387186

theorem number_of_factors_27648 : Nat.card (Nat.divisors 27648) = 44 := by
  sorry

end NUMINAMATH_CALUDE_number_of_factors_27648_l3871_387186


namespace NUMINAMATH_CALUDE_binomial_coefficient_n_1_l3871_387122

theorem binomial_coefficient_n_1 (n : ℕ+) : (n.val : ℕ).choose 1 = n.val := by sorry

end NUMINAMATH_CALUDE_binomial_coefficient_n_1_l3871_387122


namespace NUMINAMATH_CALUDE_find_r_l3871_387135

theorem find_r (k : ℝ) (r : ℝ) 
  (h1 : 5 = k * 3^r) 
  (h2 : 45 = k * 9^r) : 
  r = 2 := by
sorry

end NUMINAMATH_CALUDE_find_r_l3871_387135


namespace NUMINAMATH_CALUDE_election_votes_theorem_l3871_387184

/-- Theorem: In an election with 3 candidates, where one candidate received 71.42857142857143% 
    of the total votes, and the other two candidates received 3000 and 5000 votes respectively, 
    the winning candidate received 20,000 votes. -/
theorem election_votes_theorem : 
  let total_votes : ℝ := (20000 + 3000 + 5000 : ℝ)
  let winning_percentage : ℝ := 71.42857142857143
  let other_votes_1 : ℝ := 3000
  let other_votes_2 : ℝ := 5000
  let winning_votes : ℝ := 20000
  (winning_votes / total_votes) * 100 = winning_percentage ∧
  winning_votes + other_votes_1 + other_votes_2 = total_votes :=
by
  sorry

#check election_votes_theorem

end NUMINAMATH_CALUDE_election_votes_theorem_l3871_387184


namespace NUMINAMATH_CALUDE_friends_money_distribution_l3871_387157

theorem friends_money_distribution (x : ℚ) :
  x > 0 →
  let total := 6*x + 5*x + 4*x + 7*x + 0
  let pete_received := x + x + x + x
  pete_received / total = 2 / 11 := by
sorry

end NUMINAMATH_CALUDE_friends_money_distribution_l3871_387157


namespace NUMINAMATH_CALUDE_distance_to_center_squared_l3871_387177

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Calculates the square of the distance between two points -/
def distanceSquared (p1 p2 : Point) : ℝ :=
  (p1.x - p2.x)^2 + (p1.y - p2.y)^2

/-- Theorem: The square of the distance from B to the center of the circle is 50 -/
theorem distance_to_center_squared (O A B C : Point) : 
  O.x = 0 ∧ O.y = 0 →  -- Center at origin
  distanceSquared O A = 100 →  -- A is on the circle
  distanceSquared O C = 100 →  -- C is on the circle
  distanceSquared A B = 64 →  -- AB = 8
  distanceSquared B C = 9 →  -- BC = 3
  (B.x - A.x) * (C.y - B.y) = (B.y - A.y) * (C.x - B.x) →  -- ABC is a right angle
  distanceSquared O B = 50 := by
  sorry


end NUMINAMATH_CALUDE_distance_to_center_squared_l3871_387177


namespace NUMINAMATH_CALUDE_intersection_points_form_equilateral_triangle_l3871_387131

/-- The common points of the circle x^2 + (y - 1)^2 = 1 and the ellipse 9x^2 + (y + 1)^2 = 9 form an equilateral triangle -/
theorem intersection_points_form_equilateral_triangle :
  ∀ (A B C : ℝ × ℝ),
  (A ∈ {p : ℝ × ℝ | p.1^2 + (p.2 - 1)^2 = 1} ∩ {p : ℝ × ℝ | 9*p.1^2 + (p.2 + 1)^2 = 9}) →
  (B ∈ {p : ℝ × ℝ | p.1^2 + (p.2 - 1)^2 = 1} ∩ {p : ℝ × ℝ | 9*p.1^2 + (p.2 + 1)^2 = 9}) →
  (C ∈ {p : ℝ × ℝ | p.1^2 + (p.2 - 1)^2 = 1} ∩ {p : ℝ × ℝ | 9*p.1^2 + (p.2 + 1)^2 = 9}) →
  A ≠ B → B ≠ C → A ≠ C →
  dist A B = dist B C ∧ dist B C = dist C A :=
by sorry


end NUMINAMATH_CALUDE_intersection_points_form_equilateral_triangle_l3871_387131


namespace NUMINAMATH_CALUDE_table_runner_coverage_l3871_387125

theorem table_runner_coverage (total_runner_area : ℝ) (table_area : ℝ) (coverage_percentage : ℝ) (two_layer_area : ℝ) :
  total_runner_area = 204 →
  table_area = 175 →
  coverage_percentage = 0.8 →
  two_layer_area = 24 →
  ∃ (one_layer_area three_layer_area : ℝ),
    one_layer_area + two_layer_area + three_layer_area = coverage_percentage * table_area ∧
    one_layer_area + 2 * two_layer_area + 3 * three_layer_area = total_runner_area ∧
    three_layer_area = 20 :=
by sorry

end NUMINAMATH_CALUDE_table_runner_coverage_l3871_387125


namespace NUMINAMATH_CALUDE_coffee_shop_solution_l3871_387151

/-- Represents the coffee shop scenario with Alice and Bob -/
def coffee_shop_scenario (x : ℝ) : Prop :=
  let alice_initial := x
  let bob_initial := 1.25 * x
  let alice_consumed := 0.75 * x
  let bob_consumed := 0.75 * (1.25 * x)
  let alice_remaining := 0.25 * x
  let bob_remaining := 1.25 * x - 0.75 * (1.25 * x)
  let alice_gives := 0.5 * alice_remaining + 1
  let alice_final := alice_consumed - alice_gives
  let bob_final := bob_consumed + alice_gives
  (alice_final = bob_final) ∧
  (alice_initial + bob_initial = 9)

/-- Theorem stating that there exists a solution to the coffee shop scenario -/
theorem coffee_shop_solution : ∃ x : ℝ, coffee_shop_scenario x := by
  sorry


end NUMINAMATH_CALUDE_coffee_shop_solution_l3871_387151


namespace NUMINAMATH_CALUDE_f_derivative_sum_l3871_387145

-- Define the function f and its derivative f'
variable (f : ℝ → ℝ) (f' : ℝ → ℝ)

-- State the theorem
theorem f_derivative_sum (h1 : ∀ x, HasDerivAt f (f' x) x)
                         (h2 : ∀ x, f x = 3 * x^2 + 2 * x * f' 2) :
  f' 5 + f' 2 = -6 := by
  sorry

end NUMINAMATH_CALUDE_f_derivative_sum_l3871_387145


namespace NUMINAMATH_CALUDE_unique_solution_range_l3871_387178

theorem unique_solution_range (a : ℝ) : 
  (∃! x : ℝ, 1 < x ∧ x < 3 ∧ Real.log (x - 1) + Real.log (3 - x) = Real.log (x - a)) ↔ 
  (3/4 ≤ a ∧ a < 3) :=
sorry

end NUMINAMATH_CALUDE_unique_solution_range_l3871_387178


namespace NUMINAMATH_CALUDE_cubic_equation_solutions_l3871_387101

theorem cubic_equation_solutions :
  ∀ m n : ℤ, m^3 - n^3 = 2*m*n + 8 ↔ (m = 0 ∧ n = -2) ∨ (m = 2 ∧ n = 0) :=
by sorry

end NUMINAMATH_CALUDE_cubic_equation_solutions_l3871_387101


namespace NUMINAMATH_CALUDE_g_at_negative_two_l3871_387169

def g (x : ℝ) : ℝ := 3*x^4 - 20*x^3 + 35*x^2 - 28*x - 84

theorem g_at_negative_two : g (-2) = 320 := by
  sorry

end NUMINAMATH_CALUDE_g_at_negative_two_l3871_387169


namespace NUMINAMATH_CALUDE_overlapping_triangles_angle_sum_l3871_387196

/-- Given two overlapping triangles ABC and DEF where B and E are the same point,
    prove that the sum of angles A, B, C, D, and F is 290 degrees. -/
theorem overlapping_triangles_angle_sum
  (A B C D F : Real)
  (h1 : A = 40)
  (h2 : C = 70)
  (h3 : D = 50)
  (h4 : F = 60)
  (h5 : A + B + C = 180)  -- Sum of angles in triangle ABC
  (h6 : D + B + F = 180)  -- Sum of angles in triangle DEF (B is used instead of E)
  : A + B + C + D + F = 290 := by
  sorry

end NUMINAMATH_CALUDE_overlapping_triangles_angle_sum_l3871_387196


namespace NUMINAMATH_CALUDE_multiply_b_equals_five_l3871_387181

theorem multiply_b_equals_five (a b x : ℝ) 
  (h1 : 4 * a = x * b) 
  (h2 : a * b ≠ 0) 
  (h3 : (a / 5) / (b / 4) = 1) : 
  x = 5 := by sorry

end NUMINAMATH_CALUDE_multiply_b_equals_five_l3871_387181


namespace NUMINAMATH_CALUDE_negation_equivalence_l3871_387105

theorem negation_equivalence :
  (¬ ∃ x : ℝ, x > 2 ∨ x ≤ -1) ↔ (∀ x : ℝ, -1 < x ∧ x ≤ 2) :=
by sorry

end NUMINAMATH_CALUDE_negation_equivalence_l3871_387105


namespace NUMINAMATH_CALUDE_chosen_number_proof_l3871_387146

theorem chosen_number_proof : ∃ x : ℚ, (x / 8 : ℚ) - 100 = 6 ∧ x = 848 := by
  sorry

end NUMINAMATH_CALUDE_chosen_number_proof_l3871_387146


namespace NUMINAMATH_CALUDE_polynomial_equation_solution_l3871_387193

open Real Polynomial

/-- A polynomial satisfying the given functional equation -/
def SatisfiesEquation (P : ℝ[X]) : Prop :=
  ∀ (x : ℝ), x ≠ 0 → P.eval x + P.eval (1/x) = (P.eval (x + 1/x) + P.eval (x - 1/x)) / 2

/-- The theorem statement -/
theorem polynomial_equation_solution :
  ∀ (P : ℝ[X]), SatisfiesEquation P →
    ∃ (a b : ℝ), P = a • X^4 + b • X^2 + 6*a • 1 :=
sorry

end NUMINAMATH_CALUDE_polynomial_equation_solution_l3871_387193


namespace NUMINAMATH_CALUDE_choir_group_size_l3871_387123

theorem choir_group_size (total : ℕ) (group2 : ℕ) (group3 : ℕ) (h1 : total = 70) (h2 : group2 = 30) (h3 : group3 = 15) :
  total - group2 - group3 = 25 := by
  sorry

end NUMINAMATH_CALUDE_choir_group_size_l3871_387123

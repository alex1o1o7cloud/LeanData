import Mathlib

namespace intersection_slope_range_l1480_148026

/-- Given two points A and B, and a line l that intersects the line segment AB,
    prove that the slope k of line l is within a specific range. -/
theorem intersection_slope_range (A B : ℝ × ℝ) (k : ℝ) : 
  A = (1, 3) →
  B = (-2, -1) →
  (∃ x y : ℝ, x ∈ Set.Icc (min A.1 B.1) (max A.1 B.1) ∧ 
              y ∈ Set.Icc (min A.2 B.2) (max A.2 B.2) ∧
              y = k * (x - 2) + 1) →
  -2 ≤ k ∧ k ≤ 1/2 :=
by sorry

end intersection_slope_range_l1480_148026


namespace smallest_n_congruence_l1480_148083

theorem smallest_n_congruence : ∃ (n : ℕ), n > 0 ∧ 635 * n ≡ 1251 * n [ZMOD 30] ∧ 
  ∀ (m : ℕ), m > 0 → 635 * m ≡ 1251 * m [ZMOD 30] → n ≤ m :=
by
  use 15
  sorry

end smallest_n_congruence_l1480_148083


namespace factor_x_squared_minus_144_l1480_148071

theorem factor_x_squared_minus_144 (x : ℝ) : x^2 - 144 = (x - 12) * (x + 12) := by
  sorry

end factor_x_squared_minus_144_l1480_148071


namespace lcm_consecutive_sum_l1480_148034

theorem lcm_consecutive_sum (a b c : ℕ) : 
  (a + 1 = b) → (b + 1 = c) → (Nat.lcm a (Nat.lcm b c) = 168) → (a + b + c = 21) := by
  sorry

end lcm_consecutive_sum_l1480_148034


namespace number_in_interval_l1480_148059

theorem number_in_interval (x : ℝ) (h : x = (1/x) * (-x) + 2) :
  x = 1 ∧ 0 < x ∧ x ≤ 2 := by
  sorry

end number_in_interval_l1480_148059


namespace negative_a_sign_l1480_148027

theorem negative_a_sign (a : ℝ) : ∃ (x y : ℝ), x < 0 ∧ y > 0 ∧ (-a = x ∨ -a = y) :=
  sorry

end negative_a_sign_l1480_148027


namespace cube_surface_area_equal_volume_l1480_148030

theorem cube_surface_area_equal_volume (l w h : ℝ) (cube_edge : ℝ) :
  l = 10 ∧ w = 5 ∧ h = 24 →
  cube_edge^3 = l * w * h →
  6 * cube_edge^2 = 6 * (1200^(2/3)) := by
sorry

end cube_surface_area_equal_volume_l1480_148030


namespace divisors_of_2_pow_48_minus_1_l1480_148001

theorem divisors_of_2_pow_48_minus_1 :
  ∃! (a b : ℕ), 60 < a ∧ a < 70 ∧ 60 < b ∧ b < 70 ∧
  (2^48 - 1) % a = 0 ∧ (2^48 - 1) % b = 0 ∧
  a = 63 ∧ b = 65 := by sorry

end divisors_of_2_pow_48_minus_1_l1480_148001


namespace angle_sum_result_l1480_148017

theorem angle_sum_result (a b : Real) (h1 : 0 < a ∧ a < π/2) (h2 : 0 < b ∧ b < π/2)
  (h3 : 5 * (Real.cos a)^2 + 3 * (Real.cos b)^2 = 2)
  (h4 : 5 * Real.sin (2*a) + 3 * Real.sin (2*b) = 0) :
  2*a + b = π/2 := by
sorry

end angle_sum_result_l1480_148017


namespace overlap_length_l1480_148032

theorem overlap_length (total_length edge_to_edge_distance : ℝ) 
  (h1 : total_length = 98) 
  (h2 : edge_to_edge_distance = 83) 
  (h3 : ∃ x : ℝ, total_length = edge_to_edge_distance + 6 * x) :
  ∃ x : ℝ, x = 2.5 ∧ total_length = edge_to_edge_distance + 6 * x := by
sorry

end overlap_length_l1480_148032


namespace escalator_speed_calculation_l1480_148076

/-- The speed of the escalator in feet per second. -/
def escalator_speed : ℝ := 12

/-- The length of the escalator in feet. -/
def escalator_length : ℝ := 160

/-- The walking speed of the person in feet per second. -/
def walking_speed : ℝ := 8

/-- The time taken to cover the entire length of the escalator in seconds. -/
def time_taken : ℝ := 8

theorem escalator_speed_calculation :
  (walking_speed + escalator_speed) * time_taken = escalator_length :=
by sorry

end escalator_speed_calculation_l1480_148076


namespace gcd_power_two_minus_one_l1480_148009

theorem gcd_power_two_minus_one : 
  Nat.gcd (2^1023 - 1) (2^1034 - 1) = 2^11 - 1 := by
  sorry

end gcd_power_two_minus_one_l1480_148009


namespace homework_time_reduction_l1480_148002

theorem homework_time_reduction (initial_time final_time : ℝ) (x : ℝ) :
  initial_time = 100 →
  final_time = 70 →
  0 < x →
  x < 1 →
  initial_time * (1 - x)^2 = final_time :=
by
  sorry

end homework_time_reduction_l1480_148002


namespace cubic_factorization_l1480_148029

theorem cubic_factorization (t : ℝ) : t^3 - 144 = (t - 12) * (t^2 + 12*t + 144) := by
  sorry

end cubic_factorization_l1480_148029


namespace sundae_price_l1480_148096

/-- Given a caterer's order of ice-cream bars and sundaes, calculate the price of each sundae. -/
theorem sundae_price
  (num_ice_cream_bars : ℕ)
  (num_sundaes : ℕ)
  (total_price : ℚ)
  (ice_cream_bar_price : ℚ)
  (h1 : num_ice_cream_bars = 225)
  (h2 : num_sundaes = 125)
  (h3 : total_price = 200)
  (h4 : ice_cream_bar_price = 0.60) :
  (total_price - (↑num_ice_cream_bars * ice_cream_bar_price)) / ↑num_sundaes = 0.52 := by
  sorry

end sundae_price_l1480_148096


namespace complex_sum_theorem_l1480_148042

theorem complex_sum_theorem (a b c d : ℝ) (ω : ℂ) : 
  a ≠ -2 → b ≠ -2 → c ≠ -2 → d ≠ -2 →
  ω^4 = 1 →
  ω ≠ 1 →
  1 / (a + ω) + 1 / (b + ω) + 1 / (c + ω) + 1 / (d + ω) = 2 / ω^2 →
  1 / (a + 2) + 1 / (b + 2) + 1 / (c + 2) + 1 / (d + 2) = 2 := by
sorry


end complex_sum_theorem_l1480_148042


namespace pencil_profit_proof_l1480_148087

/-- Proves that selling 1500 pencils results in a profit of exactly $150.00 -/
theorem pencil_profit_proof (total_pencils : ℕ) (buy_price sell_price : ℚ) (profit_target : ℚ) 
  (h1 : total_pencils = 2000)
  (h2 : buy_price = 15/100)
  (h3 : sell_price = 30/100)
  (h4 : profit_target = 150) :
  (1500 : ℚ) * sell_price - (total_pencils : ℚ) * buy_price = profit_target := by
  sorry

end pencil_profit_proof_l1480_148087


namespace min_framing_for_specific_picture_l1480_148097

/-- Calculates the minimum number of linear feet of framing needed for an enlarged and bordered picture. -/
def min_framing_feet (original_width original_height border_width : ℕ) : ℕ :=
  let enlarged_width := 2 * original_width
  let enlarged_height := 2 * original_height
  let total_width := enlarged_width + 2 * border_width
  let total_height := enlarged_height + 2 * border_width
  let perimeter_inches := 2 * (total_width + total_height)
  (perimeter_inches + 11) / 12

/-- Theorem stating that for a 5-inch by 7-inch picture, enlarged and bordered as described, 
    the minimum framing needed is 6 feet. -/
theorem min_framing_for_specific_picture : 
  min_framing_feet 5 7 3 = 6 := by
  sorry

#eval min_framing_feet 5 7 3

end min_framing_for_specific_picture_l1480_148097


namespace right_trapezoid_with_inscribed_circle_sides_l1480_148068

/-- A right trapezoid with an inscribed circle -/
structure RightTrapezoidWithInscribedCircle where
  R : ℝ
  shorter_base : ℝ
  longer_base : ℝ
  longer_leg : ℝ
  shorter_base_eq : shorter_base = 4/3 * R

/-- Theorem: In a right trapezoid with an inscribed circle of radius R and shorter base 4/3 R, 
    the longer base is 4R and the longer leg is 10/3 R -/
theorem right_trapezoid_with_inscribed_circle_sides 
  (t : RightTrapezoidWithInscribedCircle) : 
  t.longer_base = 4 * t.R ∧ t.longer_leg = 10/3 * t.R := by
  sorry

end right_trapezoid_with_inscribed_circle_sides_l1480_148068


namespace no_solution_for_system_l1480_148077

theorem no_solution_for_system : ¬∃ x : ℝ, 
  (1 / (x + 2) + 8 / (x + 6) ≥ 2) ∧ (5 / (x + 1) - 2 ≤ 1) := by
  sorry

end no_solution_for_system_l1480_148077


namespace perimeter_increase_is_237_point_5_percent_l1480_148081

/-- Represents the side length ratio between consecutive triangles -/
def ratio : ℝ := 1.5

/-- Calculates the percent increase in perimeter from the first to the fourth triangle -/
def perimeter_increase : ℝ :=
  (ratio^3 - 1) * 100

/-- Theorem stating that the percent increase in perimeter is 237.5% -/
theorem perimeter_increase_is_237_point_5_percent :
  ∃ ε > 0, |perimeter_increase - 237.5| < ε :=
sorry

end perimeter_increase_is_237_point_5_percent_l1480_148081


namespace felix_distance_covered_l1480_148018

/-- The initial speed in miles per hour -/
def initial_speed : ℝ := 66

/-- The number of hours Felix wants to drive -/
def drive_hours : ℝ := 4

/-- The factor by which Felix wants to increase his speed -/
def speed_increase_factor : ℝ := 2

/-- Calculates the distance covered given a speed and time -/
def distance_covered (speed : ℝ) (time : ℝ) : ℝ := speed * time

/-- Theorem stating the distance Felix will cover -/
theorem felix_distance_covered : 
  distance_covered (initial_speed * speed_increase_factor) drive_hours = 528 := by
  sorry

end felix_distance_covered_l1480_148018


namespace min_balls_for_twenty_of_one_color_l1480_148025

/-- Represents the number of balls of each color in the box -/
structure BallCounts where
  red : Nat
  green : Nat
  yellow : Nat
  blue : Nat
  white : Nat
  black : Nat

/-- The minimum number of balls needed to guarantee at least n balls of a single color -/
def minBallsForGuarantee (counts : BallCounts) (n : Nat) : Nat :=
  sorry

/-- The specific ball counts in our problem -/
def problemCounts : BallCounts :=
  { red := 35, green := 30, yellow := 25, blue := 15, white := 12, black := 10 }

theorem min_balls_for_twenty_of_one_color :
  minBallsForGuarantee problemCounts 20 = 95 := by sorry

end min_balls_for_twenty_of_one_color_l1480_148025


namespace f_difference_180_90_l1480_148049

-- Define the sum of divisors function
def sum_of_divisors (n : ℕ+) : ℕ := sorry

-- Define the function f
def f (n : ℕ+) : ℚ := (sum_of_divisors n : ℚ) / n

-- Theorem statement
theorem f_difference_180_90 : f 180 - f 90 = 13 / 30 := by sorry

end f_difference_180_90_l1480_148049


namespace power_of_product_with_negative_l1480_148006

theorem power_of_product_with_negative (m n : ℝ) : (-2 * m^3 * n^2)^2 = 4 * m^6 * n^4 := by
  sorry

end power_of_product_with_negative_l1480_148006


namespace octagon_angle_property_l1480_148092

theorem octagon_angle_property (n : ℕ) : 
  (n - 2) * 180 = 3 * 360 ↔ n = 8 := by
  sorry

end octagon_angle_property_l1480_148092


namespace evaluate_f_l1480_148004

def f (x : ℝ) : ℝ := 2 * x^2 - 4 * x + 9

theorem evaluate_f : 3 * f 5 + 4 * f (-2) = 217 := by
  sorry

end evaluate_f_l1480_148004


namespace imaginary_sum_equals_negative_i_l1480_148043

theorem imaginary_sum_equals_negative_i (i : ℂ) (hi : i^2 = -1) :
  i^11 + i^16 + i^21 + i^26 + i^31 = -i := by
  sorry

end imaginary_sum_equals_negative_i_l1480_148043


namespace remaining_balance_proof_l1480_148085

def gift_card_balance (initial_balance : ℚ) (latte_price : ℚ) (croissant_price : ℚ) 
  (days : ℕ) (cookie_price : ℚ) (num_cookies : ℕ) : ℚ :=
  initial_balance - (latte_price + croissant_price) * days - cookie_price * num_cookies

theorem remaining_balance_proof :
  gift_card_balance 100 3.75 3.50 7 1.25 5 = 43 := by
  sorry

end remaining_balance_proof_l1480_148085


namespace cube_root_of_64_l1480_148084

theorem cube_root_of_64 (m : ℝ) : (64 : ℝ)^(1/3) = 2^m → m = 2 := by
  sorry

end cube_root_of_64_l1480_148084


namespace intersection_nonempty_implies_m_value_l1480_148000

theorem intersection_nonempty_implies_m_value (m : ℤ) : 
  let P : Set ℤ := {0, m}
  let Q : Set ℤ := {x | 2 * x^2 - 5 * x < 0}
  (P ∩ Q).Nonempty → m = 1 ∨ m = 2 := by
sorry

end intersection_nonempty_implies_m_value_l1480_148000


namespace average_and_differences_l1480_148078

theorem average_and_differences (y : ℝ) : 
  (50 + y) / 2 = 60 →
  y = 70 ∧ 
  |50 - y| = 20 ∧ 
  50 - y = -20 := by
sorry

end average_and_differences_l1480_148078


namespace max_portfolios_is_six_l1480_148021

/-- Represents the number of items Stacy purchases -/
structure Purchase where
  pens : ℕ
  pads : ℕ
  portfolios : ℕ

/-- Calculates the total cost of a purchase -/
def totalCost (p : Purchase) : ℕ :=
  2 * p.pens + 5 * p.pads + 15 * p.portfolios

/-- Checks if a purchase is valid according to the problem constraints -/
def isValidPurchase (p : Purchase) : Prop :=
  p.pens ≥ 1 ∧ p.pads ≥ 1 ∧ p.portfolios ≥ 1 ∧ totalCost p = 100

/-- The maximum number of portfolios that can be purchased -/
def maxPortfolios : ℕ := 6

/-- Theorem stating that 6 is the maximum number of portfolios that can be purchased -/
theorem max_portfolios_is_six :
  (∀ p : Purchase, isValidPurchase p → p.portfolios ≤ maxPortfolios) ∧
  (∃ p : Purchase, isValidPurchase p ∧ p.portfolios = maxPortfolios) := by
  sorry


end max_portfolios_is_six_l1480_148021


namespace x_intercept_ratio_l1480_148086

/-- Two lines with the same non-zero y-intercept -/
structure TwoLines where
  b : ℝ
  s : ℝ
  t : ℝ
  b_nonzero : b ≠ 0
  line1_equation : 0 = 8 * s + b
  line2_equation : 0 = 4 * t + b

/-- The ratio of x-intercepts is 1/2 -/
theorem x_intercept_ratio (lines : TwoLines) : lines.s / lines.t = 1 / 2 := by
  sorry

end x_intercept_ratio_l1480_148086


namespace sandwich_combinations_l1480_148019

/-- The number of available toppings -/
def num_toppings : ℕ := 10

/-- The number of slice options -/
def num_slice_options : ℕ := 4

/-- The total number of sandwich combinations -/
def total_combinations : ℕ := num_slice_options * 2^num_toppings

/-- Theorem: The total number of sandwich combinations is 4096 -/
theorem sandwich_combinations :
  total_combinations = 4096 := by
  sorry

end sandwich_combinations_l1480_148019


namespace algebraic_expression_value_l1480_148011

theorem algebraic_expression_value (x y : ℝ) (h : x + 2*y - 1 = 0) :
  (2*x + 4*y) / (x^2 + 4*x*y + 4*y^2) = 2 := by
  sorry

end algebraic_expression_value_l1480_148011


namespace breakfast_cost_l1480_148055

def toast_price : ℕ := 1
def egg_price : ℕ := 3

def dale_toast : ℕ := 2
def dale_eggs : ℕ := 2

def andrew_toast : ℕ := 1
def andrew_eggs : ℕ := 2

theorem breakfast_cost : 
  (dale_toast * toast_price + dale_eggs * egg_price) +
  (andrew_toast * toast_price + andrew_eggs * egg_price) = 15 := by
sorry

end breakfast_cost_l1480_148055


namespace three_correct_deliveries_l1480_148053

def num_houses : ℕ := 5
def num_packages : ℕ := 5

def probability_three_correct : ℚ := 1 / 6

theorem three_correct_deliveries :
  let total_arrangements := num_houses.factorial
  let correct_three_ways := num_houses.choose 3
  let incorrect_two_ways := 1  -- derangement of 2
  let prob_three_correct := correct_three_ways * incorrect_two_ways / total_arrangements
  prob_three_correct = probability_three_correct := by sorry

end three_correct_deliveries_l1480_148053


namespace fifth_month_sale_l1480_148014

/-- Proves that the sale in the 5th month is 6029, given the conditions of the problem -/
theorem fifth_month_sale (
  average_sale : ℕ)
  (first_month_sale : ℕ)
  (second_month_sale : ℕ)
  (third_month_sale : ℕ)
  (fourth_month_sale : ℕ)
  (sixth_month_sale : ℕ)
  (h1 : average_sale = 5600)
  (h2 : first_month_sale = 5266)
  (h3 : second_month_sale = 5768)
  (h4 : third_month_sale = 5922)
  (h5 : fourth_month_sale = 5678)
  (h6 : sixth_month_sale = 4937) :
  first_month_sale + second_month_sale + third_month_sale + fourth_month_sale + 6029 + sixth_month_sale = 6 * average_sale :=
by sorry

#eval 5266 + 5768 + 5922 + 5678 + 6029 + 4937
#eval 6 * 5600

end fifth_month_sale_l1480_148014


namespace a_plus_b_value_l1480_148064

def A : Set ℝ := {x | x^2 - 2*x - 3 > 0}
def B (a b : ℝ) : Set ℝ := {x | x^2 + a*x + b ≤ 0}

theorem a_plus_b_value (a b : ℝ) : 
  A ∪ B a b = Set.univ ∧ A ∩ B a b = Set.Ioc 3 4 → a + b = -7 :=
by sorry

end a_plus_b_value_l1480_148064


namespace cube_sum_equals_one_l1480_148035

theorem cube_sum_equals_one (x y : ℝ) 
  (h1 : x * (x^4 + y^4) = y^5) 
  (h2 : x^2 * (x + y) ≠ y^3) : 
  x^3 + y^3 = 1 := by
sorry

end cube_sum_equals_one_l1480_148035


namespace work_left_fraction_l1480_148098

theorem work_left_fraction (a_days b_days work_days : ℕ) 
  (ha : a_days > 0) (hb : b_days > 0) (hw : work_days > 0) : 
  let total_work : ℚ := 1
  let a_rate : ℚ := 1 / a_days
  let b_rate : ℚ := 1 / b_days
  let combined_rate : ℚ := a_rate + b_rate
  let work_done : ℚ := combined_rate * work_days
  let work_left : ℚ := total_work - work_done
  (a_days = 15 ∧ b_days = 20 ∧ work_days = 5) → work_left = 5 / 12 := by
  sorry

end work_left_fraction_l1480_148098


namespace expression_evaluation_l1480_148013

theorem expression_evaluation :
  let f (x : ℚ) := (2*x - 3) / (x + 2)
  let g (x : ℚ) := (2*(f x) - 3) / (f x + 2)
  g 2 = -10/9 := by
  sorry

end expression_evaluation_l1480_148013


namespace average_beef_sold_example_l1480_148057

/-- Calculates the average amount of beef sold per day over three days -/
def average_beef_sold (day1 : ℕ) (day2_multiplier : ℕ) (day3 : ℕ) : ℚ :=
  (day1 + day1 * day2_multiplier + day3) / 3

theorem average_beef_sold_example :
  average_beef_sold 210 2 150 = 260 := by
  sorry

end average_beef_sold_example_l1480_148057


namespace john_jenny_meeting_point_l1480_148005

/-- Represents the running scenario of John and Jenny -/
structure RunningScenario where
  total_distance : ℝ
  uphill_distance : ℝ
  downhill_distance : ℝ
  john_start_time_diff : ℝ
  john_uphill_speed : ℝ
  john_downhill_speed : ℝ
  jenny_uphill_speed : ℝ
  jenny_downhill_speed : ℝ

/-- Calculates the meeting point of John and Jenny -/
def meeting_point (scenario : RunningScenario) : ℝ :=
  sorry

/-- Theorem stating that John and Jenny meet 45/32 km from the top of the hill -/
theorem john_jenny_meeting_point :
  let scenario : RunningScenario := {
    total_distance := 12,
    uphill_distance := 6,
    downhill_distance := 6,
    john_start_time_diff := 1/4,
    john_uphill_speed := 12,
    john_downhill_speed := 18,
    jenny_uphill_speed := 14,
    jenny_downhill_speed := 21
  }
  meeting_point scenario = 45/32 := by sorry

end john_jenny_meeting_point_l1480_148005


namespace inequality_proof_l1480_148070

theorem inequality_proof (x : ℝ) 
  (h : (abs x ≤ 1) ∨ (abs x ≥ 2)) : 
  Real.cos (2*x^3 - x^2 - 5*x - 2) + 
  Real.cos (2*x^3 + 3*x^2 - 3*x - 2) - 
  Real.cos ((2*x + 1) * Real.sqrt (x^4 - 5*x^2 + 4)) < 3 := by
  sorry

end inequality_proof_l1480_148070


namespace loan_amount_l1480_148074

/-- Proves that the amount lent is 2000 rupees given the specified conditions -/
theorem loan_amount (P : ℚ) 
  (h1 : P * (17/100 * 4 - 15/100 * 4) = 160) : P = 2000 := by
  sorry

#check loan_amount

end loan_amount_l1480_148074


namespace sum_equals_fraction_l1480_148047

/-- Given a real number k > 2 such that the infinite sum of (6n-2)/k^n from n=1 to infinity
    equals 31/9, prove that k = 147/62. -/
theorem sum_equals_fraction (k : ℝ) 
  (h1 : k > 2)
  (h2 : ∑' n, (6 * n - 2) / k^n = 31/9) : 
  k = 147/62 := by
  sorry

end sum_equals_fraction_l1480_148047


namespace tangency_point_proof_l1480_148066

-- Define the two parabolas
def parabola1 (x y : ℚ) : Prop := y = x^2 + 20*x + 70
def parabola2 (x y : ℚ) : Prop := x = y^2 + 70*y + 1225

-- Define the point of tangency
def point_of_tangency : ℚ × ℚ := (-19/2, -69/2)

-- Theorem statement
theorem tangency_point_proof :
  let (x, y) := point_of_tangency
  parabola1 x y ∧ parabola2 x y ∧
  ∀ (x' y' : ℚ), x' ≠ x ∨ y' ≠ y →
    ¬(parabola1 x' y' ∧ parabola2 x' y') :=
by sorry

end tangency_point_proof_l1480_148066


namespace gcf_lcm_sum_theorem_l1480_148048

def numbers : List Nat := [15, 20, 30]

theorem gcf_lcm_sum_theorem :
  (Nat.gcd (Nat.gcd 15 20) 30) + (Nat.lcm (Nat.lcm 15 20) 30) = 65 := by
  sorry

end gcf_lcm_sum_theorem_l1480_148048


namespace intersection_in_first_quadrant_l1480_148088

-- Define the lines
def line1 (x y : ℝ) : Prop := 2 * x + 7 * y = 14
def line2 (k x y : ℝ) : Prop := k * x - y = k + 1

-- Define the intersection point
def intersection (k : ℝ) : Prop :=
  ∃ x y : ℝ, line1 x y ∧ line2 k x y

-- Define the first quadrant condition
def first_quadrant (x y : ℝ) : Prop := x > 0 ∧ y > 0

-- Theorem statement
theorem intersection_in_first_quadrant :
  ∀ k : ℝ, (∃ x y : ℝ, intersection k ∧ first_quadrant x y) → k > 0 :=
by sorry

end intersection_in_first_quadrant_l1480_148088


namespace prism_cone_properties_l1480_148093

/-- Regular triangular prism with a point T on edge BB₁ forming a cone --/
structure PrismWithCone where
  -- Base edge length of the prism
  a : ℝ
  -- Height of the prism
  h : ℝ
  -- Distance BT
  bt : ℝ
  -- Distance B₁T
  b₁t : ℝ
  -- Constraint on BT:B₁T ratio
  h_ratio : bt / b₁t = 2 / 3
  -- Constraint on prism height
  h_height : h = 5

/-- Theorem about the ratio of prism height to base edge and cone volume --/
theorem prism_cone_properties (p : PrismWithCone) :
  -- 1. Ratio of prism height to base edge is √5
  p.h / p.a = Real.sqrt 5 ∧
  -- 2. Volume of the cone
  ∃ (v : ℝ), v = (180 * Real.pi * Real.sqrt 3) / (23 * Real.sqrt 23) := by
  sorry

end prism_cone_properties_l1480_148093


namespace tangent_fraction_equality_l1480_148012

theorem tangent_fraction_equality (α : Real) (h : Real.tan α = 3) :
  (4 * Real.sin α - 2 * Real.cos α) / (5 * Real.cos α + 3 * Real.sin α) = 5 / 7 := by
  sorry

end tangent_fraction_equality_l1480_148012


namespace range_of_m_l1480_148038

theorem range_of_m (x m : ℝ) : 
  (∀ x, (2*m - 3 ≤ x ∧ x ≤ 2*m + 1) → x ≤ -5) → 
  m ≤ -3 :=
by sorry

end range_of_m_l1480_148038


namespace extra_large_posters_count_l1480_148015

def total_posters : ℕ := 200

def small_posters : ℕ := total_posters / 4
def medium_posters : ℕ := total_posters / 3
def large_posters : ℕ := total_posters / 5

def extra_large_posters : ℕ := total_posters - (small_posters + medium_posters + large_posters)

theorem extra_large_posters_count : extra_large_posters = 44 := by
  sorry

end extra_large_posters_count_l1480_148015


namespace triangle_properties_l1480_148094

theorem triangle_properties (a b c : ℝ) (A B C : ℝ) :
  (0 < a) → (0 < b) → (0 < c) →
  (0 < A) → (A < π) →
  (0 < B) → (B < π) →
  (0 < C) → (C < π/2) →
  (a * c * Real.cos B - b * c * Real.cos A = 3 * b^2) →
  (c = Real.sqrt 11) →
  (Real.sin C = 2 * Real.sqrt 2 / 3) →
  (Real.sin A / Real.sin B = Real.sqrt 7) ∧
  (1/2 * a * b * Real.sin C = Real.sqrt 14) := by
sorry

end triangle_properties_l1480_148094


namespace max_visible_sum_l1480_148040

/-- Represents a cube with six faces --/
structure Cube :=
  (faces : Fin 6 → ℕ)

/-- The set of numbers on each cube --/
def cube_numbers : Finset ℕ := {1, 3, 6, 12, 24, 48}

/-- A stack of three cubes --/
structure CubeStack :=
  (bottom : Cube)
  (middle : Cube)
  (top : Cube)

/-- The sum of visible numbers in a cube stack --/
def visible_sum (stack : CubeStack) : ℕ := sorry

/-- Theorem stating the maximum sum of visible numbers --/
theorem max_visible_sum :
  ∃ (stack : CubeStack),
    (∀ (c : Cube) (i : Fin 6), c.faces i ∈ cube_numbers) →
    (∀ (stack' : CubeStack), visible_sum stack' ≤ visible_sum stack) →
    visible_sum stack = 267 :=
sorry

end max_visible_sum_l1480_148040


namespace equilateral_triangle_coverage_l1480_148031

/-- An equilateral triangle -/
structure EquilateralTriangle where
  sideLength : ℝ
  sideLength_pos : sideLength > 0

/-- The union of a set of equilateral triangles -/
def UnionOfTriangles (triangles : Set EquilateralTriangle) : Set (ℝ × ℝ) := sorry

/-- A triangle is contained in a set of points -/
def TriangleContainedIn (t : EquilateralTriangle) (s : Set (ℝ × ℝ)) : Prop := sorry

theorem equilateral_triangle_coverage 
  (Δ : EquilateralTriangle) 
  (a b : ℝ)
  (h_a : Δ.sideLength = a)
  (h_b : b > 0)
  (h_five : ∃ (five_triangles : Finset EquilateralTriangle), 
    five_triangles.card = 5 ∧ 
    (∀ t ∈ five_triangles, t.sideLength = b) ∧
    TriangleContainedIn Δ (UnionOfTriangles five_triangles.toSet)) :
  ∃ (four_triangles : Finset EquilateralTriangle),
    four_triangles.card = 4 ∧
    (∀ t ∈ four_triangles, t.sideLength = b) ∧
    TriangleContainedIn Δ (UnionOfTriangles four_triangles.toSet) := by
  sorry

end equilateral_triangle_coverage_l1480_148031


namespace ken_released_three_fish_l1480_148054

/-- The number of fish Ken released -/
def fish_released (ken_caught : ℕ) (kendra_caught : ℕ) (brought_home : ℕ) : ℕ :=
  ken_caught + kendra_caught - brought_home

theorem ken_released_three_fish :
  ∀ (ken_caught kendra_caught brought_home : ℕ),
  ken_caught = 2 * kendra_caught →
  kendra_caught = 30 →
  brought_home = 87 →
  fish_released ken_caught kendra_caught brought_home = 3 :=
by
  sorry

end ken_released_three_fish_l1480_148054


namespace abc_product_l1480_148056

theorem abc_product (a b c : ℝ) 
  (h1 : 1/a + 1/b + 1/c = 4)
  (h2 : 4 * (1/(a+b) + 1/(b+c) + 1/(c+a)) = 4)
  (h3 : c/(a+b) + a/(b+c) + b/(c+a) = 4) :
  a * b * c = 49/23 := by
  sorry

end abc_product_l1480_148056


namespace ball_throw_height_difference_l1480_148080

/-- A proof of the height difference between Janice's final throw and Christine's first throw -/
theorem ball_throw_height_difference :
  let christine_first : ℕ := 20
  let janice_first : ℕ := christine_first - 4
  let christine_second : ℕ := christine_first + 10
  let janice_second : ℕ := janice_first * 2
  let christine_third : ℕ := christine_second + 4
  let highest_throw : ℕ := 37
  let janice_third : ℕ := highest_throw
  janice_third - christine_first = 17 :=
by
  sorry

end ball_throw_height_difference_l1480_148080


namespace yanni_money_problem_l1480_148039

/-- The amount of money Yanni's mother gave him -/
def mothers_gift : ℚ := 0.40

theorem yanni_money_problem :
  let initial_money : ℚ := 0.85
  let found_money : ℚ := 0.50
  let toy_cost : ℚ := 1.60
  let final_balance : ℚ := 0.15
  initial_money + mothers_gift + found_money - toy_cost = final_balance :=
by sorry

end yanni_money_problem_l1480_148039


namespace fourth_group_draw_l1480_148089

/-- Represents a systematic sampling setup -/
structure SystematicSampling where
  total_items : ℕ
  num_groups : ℕ
  first_draw : ℕ
  items_per_group : ℕ

/-- Calculates the number drawn in a given group for a systematic sampling -/
def draw_in_group (s : SystematicSampling) (group : ℕ) : ℕ :=
  s.first_draw + s.items_per_group * (group - 1)

/-- Theorem: In the given systematic sampling, the number drawn in the fourth group is 22 -/
theorem fourth_group_draw (s : SystematicSampling) 
  (h1 : s.total_items = 30)
  (h2 : s.num_groups = 5)
  (h3 : s.first_draw = 4)
  (h4 : s.items_per_group = 6) :
  draw_in_group s 4 = 22 := by
  sorry


end fourth_group_draw_l1480_148089


namespace brick_laying_time_l1480_148090

/-- Given that 2b men can lay 3f bricks in c days, prove that 4c men will take b^2 / f days to lay 6b bricks, assuming constant working rate. -/
theorem brick_laying_time 
  (b f c : ℝ) 
  (h : b > 0 ∧ f > 0 ∧ c > 0) 
  (rate : ℝ := (3 * f) / (2 * b * c)) : 
  (6 * b) / (4 * c * rate) = b^2 / f := by
sorry

end brick_laying_time_l1480_148090


namespace shopkeeper_ornaments_profit_least_possible_n_l1480_148008

theorem shopkeeper_ornaments_profit (n d : ℕ) (h1 : d > 0) : 
  (3 * (d / (3 * n)) + (n - 3) * (d / n + 10) - d = 150) → n ≥ 18 :=
by
  sorry

theorem least_possible_n : 
  ∃ (n d : ℕ), d > 0 ∧ 3 * (d / (3 * n)) + (n - 3) * (d / n + 10) - d = 150 ∧ n = 18 :=
by
  sorry

end shopkeeper_ornaments_profit_least_possible_n_l1480_148008


namespace toms_fruit_purchase_l1480_148073

/-- The problem of Tom's fruit purchase -/
theorem toms_fruit_purchase (apple_price : ℕ) (mango_price : ℕ) (apple_quantity : ℕ) (total_cost : ℕ) :
  apple_price = 70 →
  mango_price = 65 →
  apple_quantity = 8 →
  total_cost = 1145 →
  ∃ (mango_quantity : ℕ), 
    apple_price * apple_quantity + mango_price * mango_quantity = total_cost ∧ 
    mango_quantity = 9 := by
  sorry

#check toms_fruit_purchase

end toms_fruit_purchase_l1480_148073


namespace pants_bought_l1480_148045

def total_cost : ℕ := 1500
def tshirt_cost : ℕ := 100
def pants_cost : ℕ := 250
def num_tshirts : ℕ := 5

theorem pants_bought :
  (total_cost - num_tshirts * tshirt_cost) / pants_cost = 4 := by
sorry

end pants_bought_l1480_148045


namespace range_of_a_l1480_148091

/-- Proposition p: For all x in ℝ, ax^2 + ax + 1 > 0 always holds -/
def proposition_p (a : ℝ) : Prop :=
  ∀ x : ℝ, a * x^2 + a * x + 1 > 0

/-- Proposition q: The function f(x) = 4x^2 - ax is monotonically increasing on [1, +∞) -/
def proposition_q (a : ℝ) : Prop :=
  ∀ x y : ℝ, x ≥ 1 → y ≥ 1 → x < y → (4 * x^2 - a * x) < (4 * y^2 - a * y)

/-- The main theorem -/
theorem range_of_a :
  (∃ a : ℝ, (proposition_p a ∨ proposition_q a) ∧ ¬proposition_p a) →
  (∃ a : ℝ, a ≤ 0 ∨ (4 ≤ a ∧ a ≤ 8)) :=
by sorry

end range_of_a_l1480_148091


namespace function_inequality_implies_a_bound_sum_of_ratios_equals_zero_l1480_148095

-- Question 1
theorem function_inequality_implies_a_bound (a : ℝ) :
  (∀ p q : ℝ, 0 < p ∧ p < 2 ∧ 0 < q ∧ q < 2 ∧ p ≠ q →
    (a * Real.log (p + 2) - (p + 1)^2 - (a * Real.log (q + 2) - (q + 1)^2)) / (p - q) > 1) →
  a ≥ 28 :=
sorry

-- Question 2
theorem sum_of_ratios_equals_zero (a b c : ℝ) (h : a ≠ b ∧ b ≠ c ∧ a ≠ c) :
  let f := fun x => (x - a) * (x - b) * (x - c)
  let f' := fun x => 3 * x^2 - 2 * (a + b + c) * x + (a * b + b * c + c * a)
  a / (f' a) + b / (f' b) + c / (f' c) = 0 :=
sorry

end function_inequality_implies_a_bound_sum_of_ratios_equals_zero_l1480_148095


namespace decimal_38_to_binary_l1480_148033

-- Define a function to convert decimal to binary
def decimalToBinary (n : ℕ) : List Bool :=
  if n = 0 then [false] else
  let rec toBinary (m : ℕ) : List Bool :=
    if m = 0 then [] else (m % 2 = 1) :: toBinary (m / 2)
  toBinary n

-- Theorem statement
theorem decimal_38_to_binary :
  decimalToBinary 38 = [false, true, true, false, false, true] := by
  sorry

#eval decimalToBinary 38

end decimal_38_to_binary_l1480_148033


namespace light_2004_is_yellow_l1480_148050

def light_sequence : ℕ → Fin 4
  | n => match n % 7 with
    | 0 => 0  -- green
    | 1 => 1  -- yellow
    | 2 => 1  -- yellow
    | 3 => 2  -- red
    | 4 => 3  -- blue
    | 5 => 2  -- red
    | _ => 2  -- red

theorem light_2004_is_yellow : light_sequence 2003 = 1 := by
  sorry

end light_2004_is_yellow_l1480_148050


namespace initial_pens_count_prove_initial_pens_l1480_148082

theorem initial_pens_count : ℕ → Prop :=
  fun initial_pens =>
    let after_mike := initial_pens + 20
    let after_cindy := 2 * after_mike
    let after_sharon := after_cindy - 10
    after_sharon = initial_pens ∧ initial_pens = 30

theorem prove_initial_pens : ∃ (n : ℕ), initial_pens_count n := by
  sorry

end initial_pens_count_prove_initial_pens_l1480_148082


namespace distance_center_to_line_l1480_148037

-- Define the line l
def line_l (x y : ℝ) : Prop := x + y - 6 = 0

-- Define the circle C
def circle_C (x y θ : ℝ) : Prop :=
  x = 2 * Real.cos θ ∧ y = 2 * Real.sin θ + 2 ∧ 0 ≤ θ ∧ θ < 2 * Real.pi

-- Theorem statement
theorem distance_center_to_line :
  ∃ (x₀ y₀ : ℝ), 
    (∀ x y θ : ℝ, circle_C x y θ → (x - x₀)^2 + (y - y₀)^2 ≤ 4) ∧
    (|x₀ + y₀ - 6| / Real.sqrt 2 = 2 * Real.sqrt 2) :=
sorry

end distance_center_to_line_l1480_148037


namespace sqrt_expression_equality_algebraic_expression_equality_l1480_148067

-- Part 1
theorem sqrt_expression_equality : 2 * Real.sqrt 20 - Real.sqrt 5 + 2 * Real.sqrt (1/5) = (17 * Real.sqrt 5) / 5 := by sorry

-- Part 2
theorem algebraic_expression_equality : 
  (Real.sqrt 2 + Real.sqrt 3)^2 - (Real.sqrt 2 + Real.sqrt 3) * (Real.sqrt 2 - Real.sqrt 3) = 6 + 2 * Real.sqrt 6 := by sorry

end sqrt_expression_equality_algebraic_expression_equality_l1480_148067


namespace smallest_angle_measure_l1480_148052

/-- Represents the angles of a quadrilateral in arithmetic sequence -/
structure QuadrilateralAngles where
  a : ℝ  -- smallest angle
  d : ℝ  -- common difference

/-- Conditions for the quadrilateral angles -/
def quadrilateral_conditions (q : QuadrilateralAngles) : Prop :=
  q.a > 0 ∧
  q.d > 0 ∧
  q.a + (q.a + q.d) + (q.a + 2 * q.d) + (q.a + 3 * q.d) = 360 ∧
  q.a + (q.a + 2 * q.d) = 160

theorem smallest_angle_measure (q : QuadrilateralAngles) 
  (h : quadrilateral_conditions q) : q.a = 60 := by
  sorry

end smallest_angle_measure_l1480_148052


namespace square_difference_305_301_l1480_148022

theorem square_difference_305_301 : 305^2 - 301^2 = 2424 := by sorry

end square_difference_305_301_l1480_148022


namespace distribution_of_X_l1480_148065

/-- A discrete random variable with three possible values -/
structure DiscreteRV where
  x₁ : ℝ
  x₂ : ℝ
  x₃ : ℝ
  p₁ : ℝ
  p₂ : ℝ
  p₃ : ℝ
  x₁_lt_x₂ : x₁ < x₂
  x₂_lt_x₃ : x₂ < x₃
  prob_sum : p₁ + p₂ + p₃ = 1
  prob_nonneg : 0 ≤ p₁ ∧ 0 ≤ p₂ ∧ 0 ≤ p₃

/-- Expected value of a discrete random variable -/
def expectedValue (X : DiscreteRV) : ℝ :=
  X.x₁ * X.p₁ + X.x₂ * X.p₂ + X.x₃ * X.p₃

/-- Variance of a discrete random variable -/
def variance (X : DiscreteRV) : ℝ :=
  X.x₁^2 * X.p₁ + X.x₂^2 * X.p₂ + X.x₃^2 * X.p₃ - (expectedValue X)^2

/-- Theorem stating the distribution of the random variable X -/
theorem distribution_of_X (X : DiscreteRV) 
  (h₁ : X.x₁ = 1)
  (h₂ : X.p₁ = 0.3)
  (h₃ : X.p₂ = 0.2)
  (h₄ : expectedValue X = 2.2)
  (h₅ : variance X = 0.76) :
  X.x₂ = 2 ∧ X.x₃ = 3 ∧ X.p₃ = 0.5 := by
  sorry


end distribution_of_X_l1480_148065


namespace bottle_sales_revenue_l1480_148041

/-- Calculate the total revenue from bottle sales -/
theorem bottle_sales_revenue : 
  let small_bottles : ℕ := 6000
  let big_bottles : ℕ := 14000
  let medium_bottles : ℕ := 9000
  let small_price : ℚ := 2
  let big_price : ℚ := 4
  let medium_price : ℚ := 3
  let small_sold_percent : ℚ := 20 / 100
  let big_sold_percent : ℚ := 23 / 100
  let medium_sold_percent : ℚ := 15 / 100
  
  let small_revenue := (small_bottles : ℚ) * small_sold_percent * small_price
  let big_revenue := (big_bottles : ℚ) * big_sold_percent * big_price
  let medium_revenue := (medium_bottles : ℚ) * medium_sold_percent * medium_price
  
  let total_revenue := small_revenue + big_revenue + medium_revenue
  
  total_revenue = 19330 := by sorry

end bottle_sales_revenue_l1480_148041


namespace set_inclusion_implies_upper_bound_l1480_148060

-- Define the sets A and B
def A : Set ℝ := {x : ℝ | x > 1}
def B (a : ℝ) : Set ℝ := {x : ℝ | x < a}

-- Define the complement of B in ℝ
def C_R_B (a : ℝ) : Set ℝ := {x : ℝ | x ≥ a}

-- Theorem statement
theorem set_inclusion_implies_upper_bound (a : ℝ) :
  A ⊆ C_R_B a → a ≤ 1 := by
  sorry

end set_inclusion_implies_upper_bound_l1480_148060


namespace car_speed_problem_l1480_148072

/-- Given a car traveling for 2 hours with a speed of 40 km/h in the second hour
    and an average speed of 65 km/h, prove that the speed in the first hour is 90 km/h. -/
theorem car_speed_problem (first_hour_speed second_hour_speed average_speed : ℝ) :
  second_hour_speed = 40 →
  average_speed = 65 →
  (first_hour_speed + second_hour_speed) / 2 = average_speed →
  first_hour_speed = 90 := by
  sorry

end car_speed_problem_l1480_148072


namespace asian_games_survey_l1480_148063

theorem asian_games_survey (total students : ℕ) 
  (table_tennis badminton not_interested : ℕ) : 
  total = 50 → 
  table_tennis = 35 → 
  badminton = 30 → 
  not_interested = 5 → 
  table_tennis + badminton - (total - not_interested) = 20 := by
  sorry

end asian_games_survey_l1480_148063


namespace product_in_base9_l1480_148028

/-- Converts a base-9 number to its decimal (base-10) equivalent -/
def base9ToDecimal (n : ℕ) : ℕ := sorry

/-- Converts a decimal (base-10) number to its base-9 equivalent -/
def decimalToBase9 (n : ℕ) : ℕ := sorry

theorem product_in_base9 :
  decimalToBase9 (base9ToDecimal 327 * base9ToDecimal 6) = 2406 := by sorry

end product_in_base9_l1480_148028


namespace max_profit_plan_l1480_148069

-- Define the appliance types
inductive Appliance
| TV
| Refrigerator
| WashingMachine

-- Define the cost and selling prices
def cost_price (a : Appliance) : ℕ :=
  match a with
  | Appliance.TV => 2000
  | Appliance.Refrigerator => 1600
  | Appliance.WashingMachine => 1000

def selling_price (a : Appliance) : ℕ :=
  match a with
  | Appliance.TV => 2200
  | Appliance.Refrigerator => 1800
  | Appliance.WashingMachine => 1100

-- Define the purchasing plan
structure PurchasingPlan where
  tv_count : ℕ
  refrigerator_count : ℕ
  washing_machine_count : ℕ

-- Define the constraints
def is_valid_plan (p : PurchasingPlan) : Prop :=
  p.tv_count + p.refrigerator_count + p.washing_machine_count = 100 ∧
  p.tv_count = p.refrigerator_count ∧
  p.washing_machine_count ≤ p.tv_count ∧
  p.tv_count * cost_price Appliance.TV +
  p.refrigerator_count * cost_price Appliance.Refrigerator +
  p.washing_machine_count * cost_price Appliance.WashingMachine ≤ 160000

-- Define the profit calculation
def profit (p : PurchasingPlan) : ℕ :=
  p.tv_count * (selling_price Appliance.TV - cost_price Appliance.TV) +
  p.refrigerator_count * (selling_price Appliance.Refrigerator - cost_price Appliance.Refrigerator) +
  p.washing_machine_count * (selling_price Appliance.WashingMachine - cost_price Appliance.WashingMachine)

-- Theorem statement
theorem max_profit_plan :
  ∃ (p : PurchasingPlan),
    is_valid_plan p ∧
    profit p = 17400 ∧
    ∀ (q : PurchasingPlan), is_valid_plan q → profit q ≤ profit p :=
sorry

end max_profit_plan_l1480_148069


namespace trapezoid_perimeter_l1480_148046

/-- Represents a trapezoid EFGH with specific properties -/
structure Trapezoid where
  EF : ℝ
  angleEGF : ℝ
  angleFHE : ℝ
  height : ℝ
  EF_length : EF = 60
  angleEGF_value : angleEGF = 45 * π / 180
  angleFHE_value : angleFHE = 45 * π / 180
  height_value : height = 30 * Real.sqrt 2

/-- The perimeter of the trapezoid EFGH is 180 + 60√2 -/
theorem trapezoid_perimeter (t : Trapezoid) : 
  ∃ (perimeter : ℝ), perimeter = 180 + 60 * Real.sqrt 2 := by
  sorry

end trapezoid_perimeter_l1480_148046


namespace triangle_side_squares_sum_l1480_148058

theorem triangle_side_squares_sum (a b c : ℝ) (h : a + b + c = 4) (h1 : a > 0) (h2 : b > 0) (h3 : c > 0) :
  a^2 + b^2 + c^2 > 5 := by
sorry

end triangle_side_squares_sum_l1480_148058


namespace prob_select_one_from_two_out_of_four_prob_select_one_from_two_out_of_four_proof_l1480_148044

/-- The probability of selecting exactly one person from a group of two when randomly choosing two people from a group of four -/
theorem prob_select_one_from_two_out_of_four : ℚ :=
  2 / 3

/-- The total number of ways to select two people from four -/
def total_selections : ℕ := 6

/-- The number of ways to select exactly one person from a specific group of two when choosing two from four -/
def favorable_selections : ℕ := 4

/-- The probability is equal to the number of favorable outcomes divided by the total number of possible outcomes -/
theorem prob_select_one_from_two_out_of_four_proof :
  prob_select_one_from_two_out_of_four = favorable_selections / total_selections :=
sorry

end prob_select_one_from_two_out_of_four_prob_select_one_from_two_out_of_four_proof_l1480_148044


namespace checkerboard_squares_l1480_148079

/-- The number of squares of a given size on a rectangular grid -/
def count_squares (rows : ℕ) (cols : ℕ) (size : ℕ) : ℕ :=
  (rows - size + 1) * (cols - size + 1)

/-- The total number of squares on a 3x4 checkerboard -/
def total_squares : ℕ :=
  count_squares 3 4 1 + count_squares 3 4 2 + count_squares 3 4 3

/-- Theorem stating that the total number of squares on a 3x4 checkerboard is 20 -/
theorem checkerboard_squares :
  total_squares = 20 := by
  sorry

end checkerboard_squares_l1480_148079


namespace emilys_small_gardens_l1480_148023

/-- Given Emily's gardening scenario, prove the number of small gardens. -/
theorem emilys_small_gardens (total_seeds : ℕ) (big_garden_seeds : ℕ) (seeds_per_small_garden : ℕ) : 
  total_seeds = 41 →
  big_garden_seeds = 29 →
  seeds_per_small_garden = 4 →
  (total_seeds - big_garden_seeds) / seeds_per_small_garden = 3 := by
  sorry

end emilys_small_gardens_l1480_148023


namespace alphabet_size_l1480_148003

theorem alphabet_size :
  ∀ (dot_and_line dot_only line_only : ℕ),
    dot_and_line = 16 →
    line_only = 30 →
    dot_only = 4 →
    dot_and_line + dot_only + line_only = 50 := by
  sorry

end alphabet_size_l1480_148003


namespace cubic_factorization_l1480_148075

theorem cubic_factorization (x : ℝ) : x^3 - 2*x^2 + x - 2 = (x^2 + 1)*(x - 2) := by
  sorry

end cubic_factorization_l1480_148075


namespace stamp_arrangement_count_l1480_148024

/-- Represents the number of stamps of each denomination --/
def stamp_counts : List Nat := [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19]

/-- Represents the value of each stamp denomination --/
def stamp_values : List Nat := [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19]

/-- A function to calculate the number of unique arrangements --/
def count_arrangements (counts : List Nat) (values : List Nat) (target : Nat) : Nat :=
  sorry

theorem stamp_arrangement_count :
  count_arrangements stamp_counts stamp_values 20 = 76 :=
by sorry

end stamp_arrangement_count_l1480_148024


namespace sum_of_three_numbers_l1480_148036

theorem sum_of_three_numbers (a b c : ℝ) 
  (h1 : a^2 + b^2 + c^2 = 62) 
  (h2 : a*b + b*c + c*a = 131) : 
  a + b + c = 18 := by
sorry

end sum_of_three_numbers_l1480_148036


namespace gary_egg_collection_l1480_148010

/-- The number of chickens Gary starts with -/
def initial_chickens : ℕ := 4

/-- The factor by which the number of chickens increases after two years -/
def growth_factor : ℕ := 8

/-- The number of eggs each chicken lays per day -/
def eggs_per_chicken_per_day : ℕ := 6

/-- The number of days in a week -/
def days_in_week : ℕ := 7

/-- The number of eggs Gary collects every week after two years -/
def weekly_egg_collection : ℕ := initial_chickens * growth_factor * eggs_per_chicken_per_day * days_in_week

theorem gary_egg_collection : weekly_egg_collection = 1344 := by
  sorry

end gary_egg_collection_l1480_148010


namespace z_in_second_quadrant_l1480_148016

def complex_i : ℂ := Complex.I

def z : ℂ := complex_i + complex_i^2

def second_quadrant (c : ℂ) : Prop :=
  c.re < 0 ∧ c.im > 0

theorem z_in_second_quadrant : second_quadrant z := by
  sorry

end z_in_second_quadrant_l1480_148016


namespace sine_of_alpha_l1480_148061

-- Define the angle α
variable (α : Real)

-- Define the point on the terminal side of α
def point : ℝ × ℝ := (3, 4)

-- Define sine function
noncomputable def sine (θ : Real) : Real :=
  point.2 / Real.sqrt (point.1^2 + point.2^2)

-- Theorem statement
theorem sine_of_alpha : sine α = 4/5 := by
  sorry

end sine_of_alpha_l1480_148061


namespace parabola_intersects_x_axis_l1480_148062

/-- For a parabola y = x^2 + 2x + m - 1 to intersect with the x-axis, m must be less than or equal to 2 -/
theorem parabola_intersects_x_axis (m : ℝ) : 
  (∃ x : ℝ, x^2 + 2*x + m - 1 = 0) → m ≤ 2 := by
  sorry

end parabola_intersects_x_axis_l1480_148062


namespace sphere_intersection_circles_area_sum_l1480_148099

/-- Given a sphere of radius R and a point inside it at distance d from the center,
    the sum of the areas of three circles formed by the intersection of three
    mutually perpendicular planes passing through the point is equal to π(3R² - d²). -/
theorem sphere_intersection_circles_area_sum
  (R d : ℝ) (h_R : R > 0) (h_d : 0 ≤ d ∧ d < R) :
  ∃ (A : ℝ), A = π * (3 * R^2 - d^2) ∧
  ∀ (x y z : ℝ),
    x^2 + y^2 + z^2 = d^2 →
    A = π * ((R^2 - x^2) + (R^2 - y^2) + (R^2 - z^2)) :=
by sorry

end sphere_intersection_circles_area_sum_l1480_148099


namespace determinant_solution_l1480_148007

theorem determinant_solution (a : ℝ) (h : a ≠ 0) :
  ∃ x : ℝ, Matrix.det 
    ![![x + a, x, x],
      ![x, x + a, x],
      ![x, x, x + a]] = 0 ↔ x = -a / 3 := by
  sorry

end determinant_solution_l1480_148007


namespace intersection_M_N_l1480_148051

-- Define the sets M and N
def M : Set ℝ := {x | x^2 - 3*x - 4 ≤ 0}
def N : Set ℝ := {x | Real.log x ≥ 0}

-- State the theorem
theorem intersection_M_N : M ∩ N = {x | 1 ≤ x ∧ x ≤ 4} := by
  sorry

end intersection_M_N_l1480_148051


namespace least_positive_integer_congruence_l1480_148020

theorem least_positive_integer_congruence :
  ∃! x : ℕ+, x.val + 3649 ≡ 304 [ZMOD 15] ∧
  ∀ y : ℕ+, y.val + 3649 ≡ 304 [ZMOD 15] → x ≤ y :=
by sorry

end least_positive_integer_congruence_l1480_148020

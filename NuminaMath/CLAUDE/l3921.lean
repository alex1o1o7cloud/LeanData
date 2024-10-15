import Mathlib

namespace NUMINAMATH_CALUDE_total_apples_picked_l3921_392146

-- Define the number of apples picked by each person
def benny_apples : ℕ := 2 * 4
def dan_apples : ℕ := 9 * 5
def sarah_apples : ℕ := (dan_apples + 1) / 2  -- Rounding up
def lisa_apples : ℕ := ((3 * (benny_apples + dan_apples) + 4) / 5)  -- Rounding up

-- Theorem to prove
theorem total_apples_picked : 
  benny_apples + dan_apples + sarah_apples + lisa_apples = 108 := by
  sorry


end NUMINAMATH_CALUDE_total_apples_picked_l3921_392146


namespace NUMINAMATH_CALUDE_quadratic_function_properties_l3921_392167

/-- A quadratic function f(x) = ax^2 + bx + c where a ≠ 0 -/
noncomputable def f (a b c : ℝ) (x : ℝ) : ℝ := a * x^2 + b * x + c

theorem quadratic_function_properties
  (a b c : ℝ)
  (ha : a ≠ 0)
  (h_no_roots : ∀ x : ℝ, f a b c x ≠ x) :
  (∀ x : ℝ, f a b c (f a b c x) ≠ x) ∧
  (a > 0 → ∀ x : ℝ, f a b c (f a b c x) > x) :=
sorry

end NUMINAMATH_CALUDE_quadratic_function_properties_l3921_392167


namespace NUMINAMATH_CALUDE_initial_ball_count_is_three_l3921_392128

def bat_cost : ℕ := 500
def ball_cost : ℕ := 100

def initial_purchase_cost : ℕ := 3800
def initial_bat_count : ℕ := 7

def second_purchase_cost : ℕ := 1750
def second_bat_count : ℕ := 3
def second_ball_count : ℕ := 5

theorem initial_ball_count_is_three : 
  ∃ (x : ℕ), 
    initial_bat_count * bat_cost + x * ball_cost = initial_purchase_cost ∧
    second_bat_count * bat_cost + second_ball_count * ball_cost = second_purchase_cost ∧
    x = 3 := by
  sorry

end NUMINAMATH_CALUDE_initial_ball_count_is_three_l3921_392128


namespace NUMINAMATH_CALUDE_petya_vasya_game_l3921_392129

theorem petya_vasya_game (k : ℚ) : 
  ∃ (a b c : ℚ), ∃ (x y : ℚ), 
    x^3 + a*x^2 + b*x + c = 0 ∧ 
    y^3 + a*y^2 + b*y + c = 0 ∧ 
    y - x = 2014 :=
by sorry

end NUMINAMATH_CALUDE_petya_vasya_game_l3921_392129


namespace NUMINAMATH_CALUDE_smallest_k_for_f_divides_l3921_392160

/-- The polynomial z^12 + z^11 + z^7 + z^6 + z^5 + z + 1 -/
def f (z : ℂ) : ℂ := z^12 + z^11 + z^7 + z^6 + z^5 + z + 1

/-- Proposition: 91 is the smallest positive integer k such that f(z) divides z^k - 1 -/
theorem smallest_k_for_f_divides : ∀ z : ℂ, z ≠ 0 →
  (∀ k : ℕ, k > 0 → k < 91 → ¬(f z ∣ z^k - 1)) ∧
  (f z ∣ z^91 - 1) := by
  sorry

#check smallest_k_for_f_divides

end NUMINAMATH_CALUDE_smallest_k_for_f_divides_l3921_392160


namespace NUMINAMATH_CALUDE_jo_bob_balloon_ride_max_height_l3921_392138

/-- Represents the state of a hot air balloon ride -/
structure BalloonRide where
  ascent_rate : ℝ  -- Rate of ascent when chain is pulled (feet per minute)
  descent_rate : ℝ  -- Rate of descent when chain is released (feet per minute)
  first_pull_duration : ℝ  -- Duration of first chain pull (minutes)
  release_duration : ℝ  -- Duration of chain release (minutes)
  second_pull_duration : ℝ  -- Duration of second chain pull (minutes)

/-- Calculates the maximum height reached during a balloon ride -/
def max_height (ride : BalloonRide) : ℝ :=
  (ride.ascent_rate * ride.first_pull_duration) -
  (ride.descent_rate * ride.release_duration) +
  (ride.ascent_rate * ride.second_pull_duration)

/-- Theorem stating the maximum height reached during Jo-Bob's balloon ride -/
theorem jo_bob_balloon_ride_max_height :
  let ride : BalloonRide := {
    ascent_rate := 50,
    descent_rate := 10,
    first_pull_duration := 15,
    release_duration := 10,
    second_pull_duration := 15
  }
  max_height ride = 1400 := by sorry

end NUMINAMATH_CALUDE_jo_bob_balloon_ride_max_height_l3921_392138


namespace NUMINAMATH_CALUDE_factorization_coefficient_sum_l3921_392173

theorem factorization_coefficient_sum : 
  ∃ (A B C D E F G H J K : ℤ),
    (125 : ℤ) * X^9 - 216 * Y^9 = 
      (A * X + B * Y) * 
      (C * X^3 + D * X * Y^2 + E * Y^3) * 
      (F * X + G * Y) * 
      (H * X^3 + J * X * Y^2 + K * Y^3) ∧
    A + B + C + D + E + F + G + H + J + K = 24 :=
by sorry

end NUMINAMATH_CALUDE_factorization_coefficient_sum_l3921_392173


namespace NUMINAMATH_CALUDE_cody_dumplings_l3921_392113

def dumplings_problem (first_batch second_batch eaten_first shared_first shared_second additional_eaten : ℕ) : Prop :=
  let remaining_first := first_batch - eaten_first - shared_first
  let remaining_second := second_batch - shared_second
  let total_remaining := remaining_first + remaining_second - additional_eaten
  total_remaining = 10

theorem cody_dumplings :
  dumplings_problem 14 20 7 5 8 4 := by sorry

end NUMINAMATH_CALUDE_cody_dumplings_l3921_392113


namespace NUMINAMATH_CALUDE_point_on_h_graph_l3921_392139

-- Define the function g
def g : ℝ → ℝ := sorry

-- Define the function h in terms of g
def h (x : ℝ) : ℝ := (g x)^3

-- State the theorem
theorem point_on_h_graph :
  ∃ (x y : ℝ), g 2 = -5 ∧ h x = y ∧ x + y = -123 := by sorry

end NUMINAMATH_CALUDE_point_on_h_graph_l3921_392139


namespace NUMINAMATH_CALUDE_lottery_first_prize_probability_l3921_392149

/-- The number of balls in the MegaBall drawing -/
def megaBallCount : ℕ := 30

/-- The number of balls in the WinnerBalls drawing -/
def winnerBallCount : ℕ := 50

/-- The number of WinnerBalls picked -/
def winnerBallsPicked : ℕ := 5

/-- The probability of winning the first prize in the lottery game -/
def firstPrizeProbability : ℚ := 1 / 127125600

/-- Theorem stating the probability of winning the first prize in the lottery game -/
theorem lottery_first_prize_probability :
  firstPrizeProbability = 1 / (megaBallCount * 2 * Nat.choose winnerBallCount winnerBallsPicked) :=
by sorry

end NUMINAMATH_CALUDE_lottery_first_prize_probability_l3921_392149


namespace NUMINAMATH_CALUDE_computer_pricing_l3921_392101

theorem computer_pricing (selling_price_40 : ℝ) (profit_percentage_40 : ℝ) 
  (selling_price_50 : ℝ) (profit_percentage_50 : ℝ) :
  selling_price_40 = 2240 ∧ 
  profit_percentage_40 = 0.4 ∧ 
  selling_price_50 = 2400 ∧ 
  profit_percentage_50 = 0.5 →
  let cost := selling_price_40 / (1 + profit_percentage_40)
  selling_price_50 = cost * (1 + profit_percentage_50) := by
  sorry


end NUMINAMATH_CALUDE_computer_pricing_l3921_392101


namespace NUMINAMATH_CALUDE_range_of_a_l3921_392105

/-- A linear function y = (2a-3)x + a + 2 that is above the x-axis for -2 ≤ x ≤ 1 -/
def LinearFunction (a : ℝ) (x : ℝ) : ℝ := (2*a - 3)*x + a + 2

/-- The function is above the x-axis for -2 ≤ x ≤ 1 -/
def AboveXAxis (a : ℝ) : Prop :=
  ∀ x, -2 ≤ x ∧ x ≤ 1 → LinearFunction a x > 0

theorem range_of_a (a : ℝ) (h : AboveXAxis a) :
  1/3 < a ∧ a < 8/3 ∧ a ≠ 3/2 :=
sorry

end NUMINAMATH_CALUDE_range_of_a_l3921_392105


namespace NUMINAMATH_CALUDE_largest_valid_number_is_valid_853_largest_valid_number_is_853_l3921_392127

def is_valid_number (n : ℕ) : Prop :=
  100 ≤ n ∧ n < 1000 ∧  -- Three-digit integer
  (n / 100 = 8) ∧  -- Starts with 8
  (∀ d, d ≠ 0 → d ∣ n → n % d = 0) ∧  -- Divisible by each non-zero digit
  (n % (n / 100 + (n / 10) % 10 + n % 10) = 0)  -- Divisible by sum of digits

theorem largest_valid_number :
  ∀ m, is_valid_number m → m ≤ 853 :=
by sorry

theorem is_valid_853 : is_valid_number 853 :=
by sorry

theorem largest_valid_number_is_853 :
  ∀ n, is_valid_number n ∧ n ≠ 853 → n < 853 :=
by sorry

end NUMINAMATH_CALUDE_largest_valid_number_is_valid_853_largest_valid_number_is_853_l3921_392127


namespace NUMINAMATH_CALUDE_largest_integer_satisfying_inequality_l3921_392166

theorem largest_integer_satisfying_inequality :
  ∀ x : ℤ, (6*x - 5 < 3*x + 4) → x ≤ 2 ∧ (6*2 - 5 < 3*2 + 4) :=
by
  sorry

end NUMINAMATH_CALUDE_largest_integer_satisfying_inequality_l3921_392166


namespace NUMINAMATH_CALUDE_arithmetic_sequence_a7_l3921_392125

/-- An arithmetic sequence is a sequence where the difference between
    any two consecutive terms is constant. -/
def is_arithmetic_sequence (a : ℕ → ℚ) : Prop :=
  ∃ d : ℚ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_a7 (a : ℕ → ℚ) 
    (h_arith : is_arithmetic_sequence a)
    (h_a3 : a 3 = 2)
    (h_a5 : a 5 = 7) : 
  a 7 = 12 := by
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_a7_l3921_392125


namespace NUMINAMATH_CALUDE_floor_plus_self_unique_solution_l3921_392177

theorem floor_plus_self_unique_solution :
  ∃! r : ℝ, ⌊r⌋ + r = 20.7 :=
by sorry

end NUMINAMATH_CALUDE_floor_plus_self_unique_solution_l3921_392177


namespace NUMINAMATH_CALUDE_half_of_third_of_sixth_of_90_l3921_392133

theorem half_of_third_of_sixth_of_90 : (1 / 2 : ℚ) * (1 / 3) * (1 / 6) * 90 = 5 / 2 := by
  sorry

end NUMINAMATH_CALUDE_half_of_third_of_sixth_of_90_l3921_392133


namespace NUMINAMATH_CALUDE_percentage_increase_in_workers_l3921_392165

theorem percentage_increase_in_workers (original : ℕ) (new : ℕ) : 
  original = 852 → new = 1065 → (new - original) / original * 100 = 25 := by
  sorry

end NUMINAMATH_CALUDE_percentage_increase_in_workers_l3921_392165


namespace NUMINAMATH_CALUDE_original_deck_size_l3921_392174

/-- Represents a deck of cards with red and black cards only -/
structure Deck where
  red : ℕ
  black : ℕ

/-- The probability of selecting a red card from the deck -/
def redProbability (d : Deck) : ℚ :=
  d.red / (d.red + d.black)

theorem original_deck_size :
  ∀ d : Deck,
  redProbability d = 1/4 →
  redProbability {red := d.red, black := d.black + 6} = 1/5 →
  d.red + d.black = 24 := by
sorry

end NUMINAMATH_CALUDE_original_deck_size_l3921_392174


namespace NUMINAMATH_CALUDE_vector_sum_diff_magnitude_bounds_l3921_392194

theorem vector_sum_diff_magnitude_bounds (a b : ℝ × ℝ) 
  (ha : ‖a‖ = 1) (hb : ‖b‖ = 2) : 
  (∃ x : ℝ × ℝ, ‖x‖ = 1 ∧ ∃ y : ℝ × ℝ, ‖y‖ = 2 ∧ ‖x + y‖ + ‖x - y‖ = 4) ∧
  (∃ x : ℝ × ℝ, ‖x‖ = 1 ∧ ∃ y : ℝ × ℝ, ‖y‖ = 2 ∧ ‖x + y‖ + ‖x - y‖ = 2 * Real.sqrt 5) ∧
  (∀ x y : ℝ × ℝ, ‖x‖ = 1 → ‖y‖ = 2 → 4 ≤ ‖x + y‖ + ‖x - y‖ ∧ ‖x + y‖ + ‖x - y‖ ≤ 2 * Real.sqrt 5) :=
by sorry

end NUMINAMATH_CALUDE_vector_sum_diff_magnitude_bounds_l3921_392194


namespace NUMINAMATH_CALUDE_inequality_system_solution_set_l3921_392193

theorem inequality_system_solution_set : 
  ∀ x : ℝ, (abs x < 1 ∧ x * (x + 2) > 0) ↔ (0 < x ∧ x < 1) :=
by sorry

end NUMINAMATH_CALUDE_inequality_system_solution_set_l3921_392193


namespace NUMINAMATH_CALUDE_quadratic_equation_solution_l3921_392176

theorem quadratic_equation_solution (m : ℝ) : 
  (∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ 
    x₁^2 - 4*x₁ - 2*m + 5 = 0 ∧ 
    x₂^2 - 4*x₂ - 2*m + 5 = 0 ∧
    x₁*x₂ + x₁ + x₂ = m^2 + 6) →
  m = 1 := by
sorry

end NUMINAMATH_CALUDE_quadratic_equation_solution_l3921_392176


namespace NUMINAMATH_CALUDE_product_in_first_quadrant_l3921_392136

def complex_multiply (a b c d : ℝ) : ℂ :=
  Complex.mk (a * c - b * d) (a * d + b * c)

theorem product_in_first_quadrant :
  let z : ℂ := complex_multiply 1 3 3 (-1)
  0 < z.re ∧ 0 < z.im :=
by sorry

end NUMINAMATH_CALUDE_product_in_first_quadrant_l3921_392136


namespace NUMINAMATH_CALUDE_harrys_book_pages_l3921_392120

theorem harrys_book_pages (selenas_pages : ℕ) (harrys_pages : ℕ) : 
  selenas_pages = 400 →
  harrys_pages = selenas_pages / 2 - 20 →
  harrys_pages = 180 :=
by
  sorry

end NUMINAMATH_CALUDE_harrys_book_pages_l3921_392120


namespace NUMINAMATH_CALUDE_original_number_is_22_l3921_392135

theorem original_number_is_22 (N : ℕ) : 
  (∀ k < 6, ¬ (16 ∣ (N - k))) →  -- Condition 1: 6 is the least number
  (16 ∣ (N - 6)) →               -- Condition 2: N - 6 is divisible by 16
  N = 22 := by                   -- Conclusion: The original number is 22
sorry

end NUMINAMATH_CALUDE_original_number_is_22_l3921_392135


namespace NUMINAMATH_CALUDE_ned_remaining_games_l3921_392141

/-- The number of games Ned initially had -/
def initial_games : ℕ := 19

/-- The number of games Ned gave away -/
def games_given_away : ℕ := 13

/-- The number of games Ned has now -/
def remaining_games : ℕ := initial_games - games_given_away

theorem ned_remaining_games : remaining_games = 6 := by
  sorry

end NUMINAMATH_CALUDE_ned_remaining_games_l3921_392141


namespace NUMINAMATH_CALUDE_even_function_implies_a_equals_two_l3921_392159

/-- Given a function f(x) = (x * e^x) / (e^(ax) - 1), prove that if f is even, then a = 2 -/
theorem even_function_implies_a_equals_two (a : ℝ) :
  (∀ x : ℝ, x ≠ 0 → (x * Real.exp x) / (Real.exp (a * x) - 1) = 
    (-x * Real.exp (-x)) / (Real.exp (-a * x) - 1)) →
  a = 2 := by
sorry


end NUMINAMATH_CALUDE_even_function_implies_a_equals_two_l3921_392159


namespace NUMINAMATH_CALUDE_solution_set_g_range_of_a_l3921_392199

-- Define the functions f and g
def f (a x : ℝ) := |2*x - a| + |2*x + 3|
def g (x : ℝ) := |x - 1| + 2

-- Theorem for part (1)
theorem solution_set_g (x : ℝ) : 
  |g x| < 5 ↔ -2 < x ∧ x < 4 := by sorry

-- Theorem for part (2)
theorem range_of_a (a : ℝ) :
  (∀ x₁ : ℝ, ∃ x₂ : ℝ, f a x₁ = g x₂) → 
  (a ≥ -1 ∨ a ≤ -5) := by sorry

end NUMINAMATH_CALUDE_solution_set_g_range_of_a_l3921_392199


namespace NUMINAMATH_CALUDE_oil_to_add_l3921_392126

/-- The amount of oil Scarlett needs to add to her measuring cup -/
theorem oil_to_add (current : ℚ) (desired : ℚ) : 
  current = 0.16666666666666666 →
  desired = 0.8333333333333334 →
  desired - current = 0.6666666666666667 := by
  sorry

end NUMINAMATH_CALUDE_oil_to_add_l3921_392126


namespace NUMINAMATH_CALUDE_line_through_point_at_distance_l3921_392157

/-- A line passing through a point (x₀, y₀) and at a distance d from the origin -/
structure DistanceLine where
  x₀ : ℝ
  y₀ : ℝ
  d : ℝ

/-- Check if a line equation ax + by + c = 0 passes through a point (x₀, y₀) -/
def passesThrough (a b c x₀ y₀ : ℝ) : Prop :=
  a * x₀ + b * y₀ + c = 0

/-- Check if a line equation ax + by + c = 0 is at a distance d from the origin -/
def distanceFromOrigin (a b c d : ℝ) : Prop :=
  |c| / Real.sqrt (a^2 + b^2) = d

theorem line_through_point_at_distance (l : DistanceLine) :
  (passesThrough 1 0 (-3) l.x₀ l.y₀ ∧ distanceFromOrigin 1 0 (-3) l.d) ∨
  (passesThrough 8 (-15) 51 l.x₀ l.y₀ ∧ distanceFromOrigin 8 (-15) 51 l.d) :=
by sorry

#check line_through_point_at_distance

end NUMINAMATH_CALUDE_line_through_point_at_distance_l3921_392157


namespace NUMINAMATH_CALUDE_prime_sum_product_l3921_392184

theorem prime_sum_product (p q : ℕ) : 
  Prime p → Prime q → p + q = 91 → p * q = 178 := by sorry

end NUMINAMATH_CALUDE_prime_sum_product_l3921_392184


namespace NUMINAMATH_CALUDE_remainder_sum_l3921_392112

theorem remainder_sum (n : ℤ) : n % 20 = 11 → (n % 4 + n % 5 = 4) := by
  sorry

end NUMINAMATH_CALUDE_remainder_sum_l3921_392112


namespace NUMINAMATH_CALUDE_cheerleader_group_composition_l3921_392158

theorem cheerleader_group_composition :
  let total_males : ℕ := 10
  let males_chose_malt : ℕ := 6
  let females_chose_malt : ℕ := 8
  let total_chose_malt : ℕ := males_chose_malt + females_chose_malt
  let total_chose_coke : ℕ := total_chose_malt / 2
  let females_chose_coke : ℕ := total_chose_coke
  total_males = 10 →
  males_chose_malt = 6 →
  females_chose_malt = 8 →
  total_chose_malt = 2 * total_chose_coke →
  (females_chose_malt + females_chose_coke : ℕ) = 15
  := by sorry

end NUMINAMATH_CALUDE_cheerleader_group_composition_l3921_392158


namespace NUMINAMATH_CALUDE_cash_realized_before_brokerage_l3921_392132

/-- The cash realized on selling a stock before brokerage, given the total amount and brokerage rate -/
theorem cash_realized_before_brokerage 
  (total_amount : ℝ) 
  (brokerage_rate : ℝ) 
  (h1 : total_amount = 104)
  (h2 : brokerage_rate = 1 / 400) : 
  ∃ (cash_before_brokerage : ℝ), 
    cash_before_brokerage + cash_before_brokerage * brokerage_rate = total_amount ∧ 
    cash_before_brokerage = 41600 / 401 := by
  sorry

end NUMINAMATH_CALUDE_cash_realized_before_brokerage_l3921_392132


namespace NUMINAMATH_CALUDE_square_area_is_49_l3921_392106

-- Define the right triangle ABC
structure RightTriangle :=
  (AB : ℝ)
  (BC : ℝ)
  (is_right : True)  -- Placeholder for the right angle condition

-- Define the square BDEF
structure Square :=
  (side : ℝ)

-- Define the triangle EMN
structure TriangleEMN :=
  (EH : ℝ)

-- Main theorem
theorem square_area_is_49 
  (triangle : RightTriangle)
  (square : Square)
  (triangle_EMN : TriangleEMN)
  (h1 : triangle.AB = 15)
  (h2 : triangle.BC = 20)
  (h3 : triangle_EMN.EH = 2) :
  square.side ^ 2 = 49 := by
  sorry

end NUMINAMATH_CALUDE_square_area_is_49_l3921_392106


namespace NUMINAMATH_CALUDE_exam_scores_l3921_392110

theorem exam_scores (full_marks : ℝ) (a b c d : ℝ) : 
  full_marks = 500 →
  a = b * 0.9 →
  b = c * 1.25 →
  c = d * 0.8 →
  a = 360 →
  d / full_marks = 0.8 :=
by sorry

end NUMINAMATH_CALUDE_exam_scores_l3921_392110


namespace NUMINAMATH_CALUDE_abc_inequality_l3921_392190

theorem abc_inequality (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0)
  (h_sum : a + b + c = a^(1/7) + b^(1/7) + c^(1/7)) :
  a^a * b^b * c^c ≥ 1 := by
  sorry

end NUMINAMATH_CALUDE_abc_inequality_l3921_392190


namespace NUMINAMATH_CALUDE_percentage_equality_l3921_392114

theorem percentage_equality (x : ℝ) (h : 0.3 * (0.4 * x) = 60) : 0.4 * (0.3 * x) = 60 := by
  sorry

end NUMINAMATH_CALUDE_percentage_equality_l3921_392114


namespace NUMINAMATH_CALUDE_divisibility_criterion_37_l3921_392116

/-- Represents a function that divides a positive integer into three-digit segments from right to left -/
def segmentNumber (n : ℕ+) : List ℕ :=
  sorry

/-- Theorem: A positive integer is divisible by 37 if and only if the sum of its three-digit segments is divisible by 37 -/
theorem divisibility_criterion_37 (n : ℕ+) :
  37 ∣ n ↔ 37 ∣ (segmentNumber n).sum :=
by sorry

end NUMINAMATH_CALUDE_divisibility_criterion_37_l3921_392116


namespace NUMINAMATH_CALUDE_largest_angle_of_triangle_l3921_392185

theorem largest_angle_of_triangle (y : ℝ) : 
  45 + 60 + y = 180 →
  max (max 45 60) y = 75 := by
sorry

end NUMINAMATH_CALUDE_largest_angle_of_triangle_l3921_392185


namespace NUMINAMATH_CALUDE_cos_seven_pi_fourth_l3921_392197

theorem cos_seven_pi_fourth : Real.cos (7 * π / 4) = 1 / Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_cos_seven_pi_fourth_l3921_392197


namespace NUMINAMATH_CALUDE_asymptote_sum_l3921_392180

/-- Given an equation y = x / (x^3 + Dx^2 + Ex + F) where D, E, F are integers,
    if the graph has vertical asymptotes at x = -3, 0, and 3,
    then D + E + F = -9 -/
theorem asymptote_sum (D E F : ℤ) : 
  (∀ x : ℝ, x ≠ -3 ∧ x ≠ 0 ∧ x ≠ 3 → 
    ∃ y : ℝ, y = x / (x^3 + D*x^2 + E*x + F)) →
  D + E + F = -9 := by
  sorry

end NUMINAMATH_CALUDE_asymptote_sum_l3921_392180


namespace NUMINAMATH_CALUDE_rem_neg_five_sixths_three_fourths_l3921_392188

def rem (x y : ℚ) : ℚ := x - y * ⌊x / y⌋

theorem rem_neg_five_sixths_three_fourths :
  rem (-5/6) (3/4) = 2/3 := by sorry

end NUMINAMATH_CALUDE_rem_neg_five_sixths_three_fourths_l3921_392188


namespace NUMINAMATH_CALUDE_delivery_driver_boxes_l3921_392151

/-- Theorem: A delivery driver with 3 stops and 9 boxes per stop has 27 boxes in total. -/
theorem delivery_driver_boxes (stops : ℕ) (boxes_per_stop : ℕ) (h1 : stops = 3) (h2 : boxes_per_stop = 9) :
  stops * boxes_per_stop = 27 := by
  sorry

end NUMINAMATH_CALUDE_delivery_driver_boxes_l3921_392151


namespace NUMINAMATH_CALUDE_peters_pants_purchase_l3921_392109

theorem peters_pants_purchase (shirt_price : ℕ) (pants_price : ℕ) (total_cost : ℕ) :
  shirt_price * 2 = 20 →
  pants_price = 6 →
  ∃ (num_pants : ℕ), shirt_price * 5 + pants_price * num_pants = 62 →
  num_pants = 2 := by
sorry

end NUMINAMATH_CALUDE_peters_pants_purchase_l3921_392109


namespace NUMINAMATH_CALUDE_max_leftover_apples_l3921_392137

theorem max_leftover_apples (n : ℕ) (students : ℕ) (h : students = 8) :
  ∃ (apples_per_student : ℕ) (leftover : ℕ),
    n = students * apples_per_student + leftover ∧
    leftover < students ∧
    leftover ≤ 7 ∧
    (∀ k, k > leftover → ¬(∃ m, n = students * m + k)) :=
by sorry

end NUMINAMATH_CALUDE_max_leftover_apples_l3921_392137


namespace NUMINAMATH_CALUDE_reader_one_hour_ago_page_l3921_392196

/-- A reader who reads at a constant rate -/
structure Reader where
  rate : ℕ  -- pages per hour
  total_pages : ℕ
  current_page : ℕ
  remaining_hours : ℕ

/-- Calculates the page a reader was on one hour ago -/
def page_one_hour_ago (r : Reader) : ℕ :=
  r.current_page - r.rate

/-- Theorem: Given the specified conditions, the reader was on page 60 one hour ago -/
theorem reader_one_hour_ago_page :
  ∀ (r : Reader),
  r.total_pages = 210 →
  r.current_page = 90 →
  r.remaining_hours = 4 →
  (r.total_pages - r.current_page) = (r.rate * r.remaining_hours) →
  page_one_hour_ago r = 60 := by
  sorry


end NUMINAMATH_CALUDE_reader_one_hour_ago_page_l3921_392196


namespace NUMINAMATH_CALUDE_average_of_three_numbers_l3921_392117

theorem average_of_three_numbers : 
  let x : ℤ := -63
  let numbers : List ℤ := [2, 76, x]
  (numbers.sum : ℚ) / numbers.length = 5 := by sorry

end NUMINAMATH_CALUDE_average_of_three_numbers_l3921_392117


namespace NUMINAMATH_CALUDE_gcd_45736_123456_l3921_392181

theorem gcd_45736_123456 : Nat.gcd 45736 123456 = 352 := by
  sorry

end NUMINAMATH_CALUDE_gcd_45736_123456_l3921_392181


namespace NUMINAMATH_CALUDE_work_completion_time_l3921_392162

/-- Given that person A can complete a work in 30 days, and persons A and B together complete 1/9 of the work in 2 days, prove that person B can complete the work alone in 45 days. -/
theorem work_completion_time (a b : ℝ) (h1 : a = 30) 
  (h2 : 2 * (1 / a + 1 / b) = 1 / 9) : b = 45 := by
  sorry

end NUMINAMATH_CALUDE_work_completion_time_l3921_392162


namespace NUMINAMATH_CALUDE_min_value_a_squared_minus_b_l3921_392115

/-- The function f(x) = x^4 + ax^3 + bx^2 + ax + 1 -/
def f (a b x : ℝ) : ℝ := x^4 + a*x^3 + b*x^2 + a*x + 1

/-- Theorem: If f(x) has at least one root, then a^2 - b ≥ 1 -/
theorem min_value_a_squared_minus_b (a b : ℝ) :
  (∃ x, f a b x = 0) → a^2 - b ≥ 1 := by
  sorry

end NUMINAMATH_CALUDE_min_value_a_squared_minus_b_l3921_392115


namespace NUMINAMATH_CALUDE_area_AEHF_is_twelve_l3921_392111

/-- Rectangle ABCD with dimensions 5x6 -/
structure Rectangle :=
  (width : ℝ)
  (height : ℝ)

/-- Point on a 2D plane -/
structure Point :=
  (x : ℝ)
  (y : ℝ)

/-- Definition of rectangle ABCD -/
def rect_ABCD : Rectangle :=
  { width := 5, height := 6 }

/-- Point A at (0,0) -/
def point_A : Point :=
  { x := 0, y := 0 }

/-- Point E on CD, 3 units from D -/
def point_E : Point :=
  { x := 3, y := rect_ABCD.height }

/-- Point F on AB, 2 units from A -/
def point_F : Point :=
  { x := 2, y := 0 }

/-- Area of rectangle AEHF -/
def area_AEHF : ℝ :=
  (point_E.x - point_A.x) * (point_E.y - point_A.y)

/-- Theorem stating that the area of rectangle AEHF is 12 square units -/
theorem area_AEHF_is_twelve : area_AEHF = 12 := by
  sorry

end NUMINAMATH_CALUDE_area_AEHF_is_twelve_l3921_392111


namespace NUMINAMATH_CALUDE_bakery_pie_division_l3921_392121

theorem bakery_pie_division (pie_leftover : ℚ) (num_people : ℕ) : 
  pie_leftover = 8 / 9 → num_people = 3 → 
  pie_leftover / num_people = 8 / 27 := by
  sorry

end NUMINAMATH_CALUDE_bakery_pie_division_l3921_392121


namespace NUMINAMATH_CALUDE_degrees_to_radians_conversion_l3921_392195

theorem degrees_to_radians_conversion :
  ∀ (degrees : ℝ) (radians : ℝ),
  degrees * (π / 180) = radians →
  -630 * (π / 180) = -7 * π / 2 :=
by sorry

end NUMINAMATH_CALUDE_degrees_to_radians_conversion_l3921_392195


namespace NUMINAMATH_CALUDE_rosas_phone_calls_l3921_392171

/-- Rosa's phone book calling problem -/
theorem rosas_phone_calls (pages_last_week pages_this_week : ℝ) 
  (h1 : pages_last_week = 10.2)
  (h2 : pages_this_week = 8.6) : 
  pages_last_week + pages_this_week = 18.8 := by
  sorry

end NUMINAMATH_CALUDE_rosas_phone_calls_l3921_392171


namespace NUMINAMATH_CALUDE_expression_evaluation_l3921_392156

theorem expression_evaluation : ((-3)^2)^4 * (-3)^8 * 2 = 86093442 := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l3921_392156


namespace NUMINAMATH_CALUDE_absolute_value_inequality_l3921_392100

theorem absolute_value_inequality (x : ℝ) :
  |x - 1| - |x - 5| < 2 ↔ x < 4 := by sorry

end NUMINAMATH_CALUDE_absolute_value_inequality_l3921_392100


namespace NUMINAMATH_CALUDE_solve_congruence_l3921_392140

theorem solve_congruence :
  ∃ n : ℕ, 0 ≤ n ∧ n < 43 ∧ (11 * n) % 43 = 7 % 43 ∧ n = 28 := by
  sorry

end NUMINAMATH_CALUDE_solve_congruence_l3921_392140


namespace NUMINAMATH_CALUDE_xy_product_range_l3921_392183

theorem xy_product_range (x y : ℝ) : 
  x^2 * y^2 + x^2 - 10*x*y - 8*x + 16 = 0 → 0 ≤ x*y ∧ x*y ≤ 10 := by
  sorry

end NUMINAMATH_CALUDE_xy_product_range_l3921_392183


namespace NUMINAMATH_CALUDE_exam_selection_difference_l3921_392134

theorem exam_selection_difference (total_candidates : ℕ) 
  (selection_rate_A selection_rate_B : ℚ) : 
  total_candidates = 8200 →
  selection_rate_A = 6 / 100 →
  selection_rate_B = 7 / 100 →
  (selection_rate_B * total_candidates : ℚ).floor - 
  (selection_rate_A * total_candidates : ℚ).floor = 82 :=
by sorry

end NUMINAMATH_CALUDE_exam_selection_difference_l3921_392134


namespace NUMINAMATH_CALUDE_wolf_does_not_catch_hare_l3921_392198

/-- Prove that the wolf does not catch the hare given the initial conditions -/
theorem wolf_does_not_catch_hare (initial_distance : ℝ) (distance_to_refuge : ℝ) 
  (wolf_speed : ℝ) (hare_speed : ℝ) 
  (h1 : initial_distance = 30) 
  (h2 : distance_to_refuge = 250) 
  (h3 : wolf_speed = 600) 
  (h4 : hare_speed = 550) : 
  (distance_to_refuge / hare_speed) < ((initial_distance + distance_to_refuge) / wolf_speed) :=
by
  sorry

#check wolf_does_not_catch_hare

end NUMINAMATH_CALUDE_wolf_does_not_catch_hare_l3921_392198


namespace NUMINAMATH_CALUDE_cos_2alpha_value_l3921_392142

theorem cos_2alpha_value (α : Real) (h1 : 0 < α ∧ α < π/2) (h2 : Real.sin (α - π/4) = 1/4) : 
  Real.cos (2 * α) = -Real.sqrt 15 / 8 := by
  sorry

end NUMINAMATH_CALUDE_cos_2alpha_value_l3921_392142


namespace NUMINAMATH_CALUDE_complex_absolute_value_l3921_392187

/-- Given a complex number ω = 5 + 4i, prove that |ω^2 + 4ω + 41| = 2√2009 -/
theorem complex_absolute_value (ω : ℂ) (h : ω = 5 + 4 * I) : 
  Complex.abs (ω^2 + 4*ω + 41) = 2 * Real.sqrt 2009 := by
  sorry

end NUMINAMATH_CALUDE_complex_absolute_value_l3921_392187


namespace NUMINAMATH_CALUDE_total_cars_on_train_l3921_392148

/-- The number of cars Rita counted in the first 15 seconds -/
def initial_cars : ℕ := 9

/-- The time in seconds during which Rita counted the initial cars -/
def initial_time : ℕ := 15

/-- The total time in seconds for the train to pass -/
def total_time : ℕ := 195

/-- The rate of cars passing per second -/
def rate : ℚ := initial_cars / initial_time

/-- The theorem stating the total number of cars on the train -/
theorem total_cars_on_train : ⌊rate * total_time⌋ = 117 := by sorry

end NUMINAMATH_CALUDE_total_cars_on_train_l3921_392148


namespace NUMINAMATH_CALUDE_equation_solution_l3921_392192

theorem equation_solution :
  ∃ x : ℚ, x = -62/29 ∧ (Real.sqrt (7*x + 1) / Real.sqrt (4*(x + 2) - 1) = 3) := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l3921_392192


namespace NUMINAMATH_CALUDE_line_parallel_to_plane_l3921_392186

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the relations
variable (parallel : Line → Line → Prop)
variable (parallel_plane_line : Plane → Line → Prop)
variable (parallel_plane : Plane → Plane → Prop)
variable (intersect : Plane → Plane → Line → Prop)
variable (in_plane : Line → Plane → Prop)

-- State the theorem
theorem line_parallel_to_plane 
  (m n : Line) (α β : Plane) 
  (h1 : ¬ (m = n)) -- m and n are non-overlapping
  (h2 : ¬ (α = β)) -- α and β are non-overlapping
  (h3 : intersect α β n) -- α intersects β at n
  (h4 : ¬ in_plane m α) -- m is not in α
  (h5 : parallel m n) -- m is parallel to n
  : parallel_plane_line α m := by sorry

end NUMINAMATH_CALUDE_line_parallel_to_plane_l3921_392186


namespace NUMINAMATH_CALUDE_project_hours_theorem_l3921_392168

theorem project_hours_theorem (kate mark pat : ℕ) 
  (h1 : pat = 2 * kate)
  (h2 : 3 * pat = mark)
  (h3 : mark = kate + 120) :
  kate + mark + pat = 216 := by
  sorry

end NUMINAMATH_CALUDE_project_hours_theorem_l3921_392168


namespace NUMINAMATH_CALUDE_symmetric_points_y_axis_l3921_392119

theorem symmetric_points_y_axis (m n : ℝ) : 
  (m - 1 = -2 ∧ 4 = n + 2) → n^m = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_symmetric_points_y_axis_l3921_392119


namespace NUMINAMATH_CALUDE_jesses_room_difference_l3921_392107

/-- Jesse's room dimensions and length-width difference --/
theorem jesses_room_difference :
  ∀ (length width : ℝ),
  length = 20 →
  width = 19 →
  length - width = 1 :=
by
  sorry

end NUMINAMATH_CALUDE_jesses_room_difference_l3921_392107


namespace NUMINAMATH_CALUDE_pi_only_irrational_l3921_392123

-- Define a function to check if a number is rational
def is_rational (x : ℝ) : Prop := ∃ (p q : ℤ), q ≠ 0 ∧ x = p / q

-- State the theorem
theorem pi_only_irrational : 
  is_rational (1/7) ∧ 
  ¬(is_rational Real.pi) ∧ 
  is_rational (-1) ∧ 
  is_rational 0 :=
sorry

end NUMINAMATH_CALUDE_pi_only_irrational_l3921_392123


namespace NUMINAMATH_CALUDE_remaining_item_is_bead_l3921_392169

/-- Represents the three types of items --/
inductive Item
  | GoldBar
  | Pearl
  | Bead

/-- Represents the state of the tribe's possessions --/
structure TribeState where
  goldBars : Nat
  pearls : Nat
  beads : Nat

/-- Represents the possible exchanges --/
inductive Exchange
  | Cortes    -- 1 gold bar + 1 pearl → 1 bead
  | Montezuma -- 1 gold bar + 1 bead → 1 pearl
  | Totonacs  -- 1 pearl + 1 bead → 1 gold bar

def initialState : TribeState :=
  { goldBars := 24, pearls := 26, beads := 25 }

def applyExchange (state : TribeState) (exchange : Exchange) : TribeState :=
  match exchange with
  | Exchange.Cortes =>
      { goldBars := state.goldBars - 1, pearls := state.pearls - 1, beads := state.beads + 1 }
  | Exchange.Montezuma =>
      { goldBars := state.goldBars - 1, pearls := state.pearls + 1, beads := state.beads - 1 }
  | Exchange.Totonacs =>
      { goldBars := state.goldBars + 1, pearls := state.pearls - 1, beads := state.beads - 1 }

def remainingItem (state : TribeState) : Option Item :=
  if state.goldBars > 0 && state.pearls = 0 && state.beads = 0 then some Item.GoldBar
  else if state.goldBars = 0 && state.pearls > 0 && state.beads = 0 then some Item.Pearl
  else if state.goldBars = 0 && state.pearls = 0 && state.beads > 0 then some Item.Bead
  else none

/-- Theorem stating that if only one item type remains after any number of exchanges, it must be beads --/
theorem remaining_item_is_bead (exchanges : List Exchange) :
  let finalState := exchanges.foldl applyExchange initialState
  remainingItem finalState = some Item.Bead ∨ remainingItem finalState = none := by
  sorry

end NUMINAMATH_CALUDE_remaining_item_is_bead_l3921_392169


namespace NUMINAMATH_CALUDE_bug_meeting_point_l3921_392122

/-- Represents a triangle with side lengths -/
structure Triangle where
  pq : ℝ
  qr : ℝ
  pr : ℝ

/-- Represents a bug crawling on the triangle -/
structure Bug where
  speed : ℝ
  clockwise : Bool

/-- The meeting point of two bugs on a triangle -/
def meetingPoint (t : Triangle) (b1 b2 : Bug) : ℝ := sorry

theorem bug_meeting_point (t : Triangle) (b1 b2 : Bug) :
  t.pq = 8 ∧ t.qr = 10 ∧ t.pr = 12 ∧
  b1.speed = 2 ∧ b1.clockwise = true ∧
  b2.speed = 3 ∧ b2.clockwise = false →
  t.qr - meetingPoint t b1 b2 = 6 := by sorry

end NUMINAMATH_CALUDE_bug_meeting_point_l3921_392122


namespace NUMINAMATH_CALUDE_square_root_and_square_operations_l3921_392164

theorem square_root_and_square_operations : 
  (∃ (x : ℝ), x ^ 2 = 4 ∧ x = 2) ∧ 
  (∀ (a : ℝ), (-3 * a) ^ 2 = 9 * a ^ 2) := by
  sorry

end NUMINAMATH_CALUDE_square_root_and_square_operations_l3921_392164


namespace NUMINAMATH_CALUDE_parallelogram_roots_l3921_392175

theorem parallelogram_roots (a : ℝ) : 
  (∃ z₁ z₂ z₃ z₄ : ℂ, 
    z₁^4 - 8*z₁^3 + 13*a*z₁^2 - 2*(3*a^2 + 2*a - 4)*z₁ - 2 = 0 ∧
    z₂^4 - 8*z₂^3 + 13*a*z₂^2 - 2*(3*a^2 + 2*a - 4)*z₂ - 2 = 0 ∧
    z₃^4 - 8*z₃^3 + 13*a*z₃^2 - 2*(3*a^2 + 2*a - 4)*z₃ - 2 = 0 ∧
    z₄^4 - 8*z₄^3 + 13*a*z₄^2 - 2*(3*a^2 + 2*a - 4)*z₄ - 2 = 0 ∧
    (z₁ + z₃ = z₂ + z₄) ∧ (z₁ - z₂ = z₄ - z₃)) ↔
  a^2 + (2/3)*a - 49*(1/3) = 0 :=
sorry

end NUMINAMATH_CALUDE_parallelogram_roots_l3921_392175


namespace NUMINAMATH_CALUDE_computer_table_cost_calculation_l3921_392144

/-- The cost price of the computer table -/
def computer_table_cost : ℝ := 4813.58

/-- The cost price of the office chair -/
def office_chair_cost : ℝ := 5000

/-- The markup percentage -/
def markup_percentage : ℝ := 0.24

/-- The discount percentage -/
def discount_percentage : ℝ := 0.05

/-- The total amount paid by the customer -/
def total_paid : ℝ := 11560

theorem computer_table_cost_calculation :
  let total_before_discount := (1 + markup_percentage) * (computer_table_cost + office_chair_cost)
  (1 - discount_percentage) * total_before_discount = total_paid := by
  sorry

#eval computer_table_cost

end NUMINAMATH_CALUDE_computer_table_cost_calculation_l3921_392144


namespace NUMINAMATH_CALUDE_cubic_root_sum_cubes_l3921_392179

theorem cubic_root_sum_cubes (r s t : ℂ) : 
  (8 * r^3 + 2010 * r + 4016 = 0) →
  (8 * s^3 + 2010 * s + 4016 = 0) →
  (8 * t^3 + 2010 * t + 4016 = 0) →
  (r + s)^3 + (s + t)^3 + (t + r)^3 = 1506 := by
  sorry

end NUMINAMATH_CALUDE_cubic_root_sum_cubes_l3921_392179


namespace NUMINAMATH_CALUDE_max_value_expression_l3921_392191

theorem max_value_expression (a b c d : ℝ) 
  (ha : -6 ≤ a ∧ a ≤ 6)
  (hb : -6 ≤ b ∧ b ≤ 6)
  (hc : -6 ≤ c ∧ c ≤ 6)
  (hd : -6 ≤ d ∧ d ≤ 6) :
  (∀ x y z w, -6 ≤ x ∧ x ≤ 6 → -6 ≤ y ∧ y ≤ 6 → -6 ≤ z ∧ z ≤ 6 → -6 ≤ w ∧ w ≤ 6 →
    x + 2*y + z + 2*w - x*y - y*z - z*w - w*x ≤ a + 2*b + c + 2*d - a*b - b*c - c*d - d*a) →
  a + 2*b + c + 2*d - a*b - b*c - c*d - d*a = 156 :=
by sorry

end NUMINAMATH_CALUDE_max_value_expression_l3921_392191


namespace NUMINAMATH_CALUDE_smallest_natural_divisible_l3921_392118

theorem smallest_natural_divisible (n : ℕ) : 
  (∀ m : ℕ, m < n → ¬(∃ k : ℕ, m + 1 = 4 * k)) ∨
  (∀ m : ℕ, m < n → ¬(∃ k : ℕ, m + 1 = 6 * k)) ∨
  (∀ m : ℕ, m < n → ¬(∃ k : ℕ, m + 1 = 10 * k)) ∨
  (∀ m : ℕ, m < n → ¬(∃ k : ℕ, m + 1 = 12 * k)) →
  (∃ k1 k2 k3 k4 : ℕ, n + 1 = 4 * k1 ∧ n + 1 = 6 * k2 ∧ n + 1 = 10 * k3 ∧ n + 1 = 12 * k4) →
  n = 59 := by
sorry

end NUMINAMATH_CALUDE_smallest_natural_divisible_l3921_392118


namespace NUMINAMATH_CALUDE_monotone_increasing_implies_a_geq_5_l3921_392178

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := x^2 + 2*a*x + 2

-- State the theorem
theorem monotone_increasing_implies_a_geq_5 :
  ∀ a : ℝ, (∀ x y : ℝ, -5 ≤ x ∧ x < y ∧ y ≤ 5 → f a x < f a y) → a ≥ 5 := by
  sorry

end NUMINAMATH_CALUDE_monotone_increasing_implies_a_geq_5_l3921_392178


namespace NUMINAMATH_CALUDE_another_divisor_of_44404_l3921_392103

theorem another_divisor_of_44404 (n : Nat) (h1 : n = 44404) 
  (h2 : 12 ∣ n) (h3 : 48 ∣ n) (h4 : 74 ∣ n) (h5 : 100 ∣ n) : 
  199 ∣ n := by
  sorry

end NUMINAMATH_CALUDE_another_divisor_of_44404_l3921_392103


namespace NUMINAMATH_CALUDE_set_equality_through_double_complement_l3921_392172

universe u

theorem set_equality_through_double_complement 
  {U : Type u} [Nonempty U] (M N P : Set U) 
  (h1 : M = (Nᶜ : Set U)) 
  (h2 : N = (Pᶜ : Set U)) : 
  M = P := by
  sorry

end NUMINAMATH_CALUDE_set_equality_through_double_complement_l3921_392172


namespace NUMINAMATH_CALUDE_circle_radius_l3921_392130

theorem circle_radius (x y : ℝ) (h : x + y = 100 * Real.pi) :
  let r := Real.sqrt 101 - 1
  x = Real.pi * r^2 ∧ y = 2 * Real.pi * r := by
  sorry

end NUMINAMATH_CALUDE_circle_radius_l3921_392130


namespace NUMINAMATH_CALUDE_class_payment_problem_l3921_392153

theorem class_payment_problem (total_students : ℕ) (full_payment half_payment total_collected : ℚ) 
  (h1 : total_students = 25)
  (h2 : full_payment = 50)
  (h3 : half_payment = 25)
  (h4 : total_collected = 1150)
  (h5 : ∃ (full_payers half_payers : ℕ), 
    full_payers + half_payers = total_students ∧ 
    full_payers * full_payment + half_payers * half_payment = total_collected) :
  ∃ (half_payers : ℕ), half_payers = 4 := by
sorry

end NUMINAMATH_CALUDE_class_payment_problem_l3921_392153


namespace NUMINAMATH_CALUDE_goldfish_graph_is_finite_distinct_points_l3921_392189

def cost (n : ℕ) : ℕ := 20 * n + 500

def goldfish_points : Set (ℕ × ℕ) :=
  {p | ∃ n : ℕ, 1 ≤ n ∧ n ≤ 20 ∧ p = (n, cost n)}

theorem goldfish_graph_is_finite_distinct_points :
  Finite goldfish_points ∧ 
  (∀ p q : ℕ × ℕ, p ∈ goldfish_points → q ∈ goldfish_points → p ≠ q → p.2 ≠ q.2) :=
sorry

end NUMINAMATH_CALUDE_goldfish_graph_is_finite_distinct_points_l3921_392189


namespace NUMINAMATH_CALUDE_sin_addition_equality_l3921_392131

theorem sin_addition_equality (y : Real) : 
  (y ∈ Set.Icc 0 (Real.pi / 2)) → 
  (∀ x ∈ Set.Icc 0 (Real.pi / 2), Real.sin (x + y) = Real.sin x + Real.sin y) → 
  y = 0 :=
by sorry

end NUMINAMATH_CALUDE_sin_addition_equality_l3921_392131


namespace NUMINAMATH_CALUDE_solve_linear_equation_l3921_392152

theorem solve_linear_equation (x : ℝ) :
  3 * x - 8 = 4 * x + 5 → x = -13 := by
sorry

end NUMINAMATH_CALUDE_solve_linear_equation_l3921_392152


namespace NUMINAMATH_CALUDE_gina_money_to_mom_l3921_392170

theorem gina_money_to_mom (total : ℝ) (clothes_fraction : ℝ) (charity_fraction : ℝ) (kept : ℝ) :
  total = 400 →
  clothes_fraction = 1/8 →
  charity_fraction = 1/5 →
  kept = 170 →
  ∃ (mom_fraction : ℝ), 
    mom_fraction * total + clothes_fraction * total + charity_fraction * total + kept = total ∧
    mom_fraction = 1/4 :=
by sorry

end NUMINAMATH_CALUDE_gina_money_to_mom_l3921_392170


namespace NUMINAMATH_CALUDE_correct_product_l3921_392104

/-- Given two positive integers a and b, where a is a two-digit number,
    if the product of the reversed digits of a and b is 161,
    then the product of a and b is 224. -/
theorem correct_product (a b : ℕ) : 
  a ≥ 10 ∧ a ≤ 99 →  -- a is a two-digit number
  b > 0 →  -- b is positive
  (10 * (a % 10) + a / 10) * b = 161 →  -- reversed a * b = 161
  a * b = 224 :=
by sorry

end NUMINAMATH_CALUDE_correct_product_l3921_392104


namespace NUMINAMATH_CALUDE_half_merit_scholarship_percentage_l3921_392155

/-- Given a group of senior students, prove the percentage who received
    a half merit scholarship. -/
theorem half_merit_scholarship_percentage
  (total_students : ℕ)
  (full_scholarship_percentage : ℚ)
  (no_scholarship_count : ℕ)
  (h1 : total_students = 300)
  (h2 : full_scholarship_percentage = 5 / 100)
  (h3 : no_scholarship_count = 255) :
  (total_students - no_scholarship_count - 
   (full_scholarship_percentage * total_students).num) / total_students = 1 / 10 := by
  sorry

end NUMINAMATH_CALUDE_half_merit_scholarship_percentage_l3921_392155


namespace NUMINAMATH_CALUDE_decreasing_quadratic_implies_a_le_two_l3921_392182

/-- A quadratic function f(x) = -x² - 2(a-1)x + 5 -/
def f (a : ℝ) (x : ℝ) : ℝ := -x^2 - 2*(a-1)*x + 5

/-- The theorem states that if f(x) is decreasing on [-1, +∞), then a ≤ 2 -/
theorem decreasing_quadratic_implies_a_le_two (a : ℝ) :
  (∀ x₁ x₂ : ℝ, -1 ≤ x₁ ∧ x₁ < x₂ → f a x₂ < f a x₁) →
  a ≤ 2 :=
sorry

end NUMINAMATH_CALUDE_decreasing_quadratic_implies_a_le_two_l3921_392182


namespace NUMINAMATH_CALUDE_max_students_l3921_392124

/-- Represents the relationship between students -/
def knows (n : ℕ) := Fin n → Fin n → Prop

/-- At least two out of any three students know each other -/
def three_two_know (n : ℕ) (k : knows n) : Prop :=
  ∀ a b c : Fin n, a ≠ b ∧ b ≠ c ∧ a ≠ c →
    k a b ∨ k b c ∨ k a c

/-- At least two out of any four students do not know each other -/
def four_two_dont_know (n : ℕ) (k : knows n) : Prop :=
  ∀ a b c d : Fin n, a ≠ b ∧ b ≠ c ∧ c ≠ d ∧ a ≠ c ∧ a ≠ d ∧ b ≠ d →
    ¬(k a b) ∨ ¬(k a c) ∨ ¬(k a d) ∨ ¬(k b c) ∨ ¬(k b d) ∨ ¬(k c d)

/-- The maximum number of students satisfying the conditions is 8 -/
theorem max_students : 
  (∃ (k : knows 8), three_two_know 8 k ∧ four_two_dont_know 8 k) ∧
  (∀ n > 8, ¬∃ (k : knows n), three_two_know n k ∧ four_two_dont_know n k) :=
sorry

end NUMINAMATH_CALUDE_max_students_l3921_392124


namespace NUMINAMATH_CALUDE_prob_two_sixes_is_one_thirty_sixth_l3921_392163

/-- A fair six-sided die -/
structure FairDie :=
  (sides : Fin 6)

/-- The probability of rolling a specific number on a fair six-sided die -/
def prob_single_roll (d : FairDie) (n : Fin 6) : ℚ := 1 / 6

/-- The probability of rolling two consecutive sixes -/
def prob_two_sixes (d : FairDie) : ℚ :=
  (prob_single_roll d 5) * (prob_single_roll d 5)

/-- Theorem: The probability of rolling two consecutive sixes with a fair six-sided die is 1/36 -/
theorem prob_two_sixes_is_one_thirty_sixth (d : FairDie) :
  prob_two_sixes d = 1 / 36 := by
  sorry

end NUMINAMATH_CALUDE_prob_two_sixes_is_one_thirty_sixth_l3921_392163


namespace NUMINAMATH_CALUDE_product_equals_32_l3921_392102

theorem product_equals_32 : 
  (1 / 4) * 8 * (1 / 16) * 32 * (1 / 64) * 128 * (1 / 256) * 512 * (1 / 1024) * 2048 = 32 := by
  sorry

end NUMINAMATH_CALUDE_product_equals_32_l3921_392102


namespace NUMINAMATH_CALUDE_hexagon_sixth_angle_measure_l3921_392150

/-- The measure of the sixth angle in a hexagon, given the other five angles -/
theorem hexagon_sixth_angle_measure (a b c d e : ℝ) 
  (ha : a = 130)
  (hb : b = 95)
  (hc : c = 122)
  (hd : d = 108)
  (he : e = 114) :
  720 - (a + b + c + d + e) = 151 := by
  sorry

end NUMINAMATH_CALUDE_hexagon_sixth_angle_measure_l3921_392150


namespace NUMINAMATH_CALUDE_lords_partition_l3921_392147

/-- A graph with vertices of type α -/
structure Graph (α : Type) where
  adj : α → α → Prop

/-- The degree of a vertex in a graph -/
def degree {α : Type} (G : Graph α) (v : α) : ℕ := 
  sorry

/-- A partition of a set into two subsets -/
def Partition (α : Type) := (α → Bool)

/-- The number of adjacent vertices in the same partition -/
def samePartitionDegree {α : Type} (G : Graph α) (p : Partition α) (v : α) : ℕ := 
  sorry

theorem lords_partition {α : Type} (G : Graph α) :
  (∀ v : α, degree G v ≤ 3) →
  ∃ p : Partition α, ∀ v : α, samePartitionDegree G p v ≤ 1 := by
  sorry

end NUMINAMATH_CALUDE_lords_partition_l3921_392147


namespace NUMINAMATH_CALUDE_trig_inequality_implies_range_l3921_392161

open Real

theorem trig_inequality_implies_range (θ : ℝ) :
  θ ∈ Set.Icc 0 (2 * π) →
  (cos θ)^5 - (sin θ)^5 < 7 * ((sin θ)^3 - (cos θ)^3) →
  θ ∈ Set.Ioo (π / 4) (5 * π / 4) :=
by
  sorry

end NUMINAMATH_CALUDE_trig_inequality_implies_range_l3921_392161


namespace NUMINAMATH_CALUDE_raffle_probabilities_l3921_392154

structure Raffle :=
  (white_balls : ℕ)
  (black_balls : ℕ)
  (num_people : ℕ)

def first_person_wins (r : Raffle) : ℚ :=
  r.black_balls / (r.white_balls + r.black_balls)

def last_person_wins (r : Raffle) : ℚ :=
  (r.white_balls / (r.white_balls + r.black_balls)) *
  ((r.white_balls - 1) / (r.white_balls + r.black_balls - 1)) *
  ((r.white_balls - 2) / (r.white_balls + r.black_balls - 2)) *
  (r.black_balls / (r.white_balls + r.black_balls - 3))

def first_person_wins_continued (r : Raffle) : ℚ :=
  first_person_wins r +
  (r.white_balls / (r.white_balls + r.black_balls)) *
  ((r.white_balls - 1) / (r.white_balls + r.black_balls - 1)) *
  ((r.white_balls - 2) / (r.white_balls + r.black_balls - 2)) *
  ((r.white_balls - 3) / (r.white_balls + r.black_balls - 3)) *
  (r.black_balls / (r.white_balls + r.black_balls - 4))

def last_person_wins_continued (r : Raffle) : ℚ :=
  (r.white_balls / (r.white_balls + r.black_balls)) *
  ((r.white_balls - 1) / (r.white_balls + r.black_balls - 1)) *
  ((r.white_balls - 2) / (r.white_balls + r.black_balls - 2)) *
  (r.black_balls / (r.white_balls + r.black_balls - 3))

theorem raffle_probabilities :
  let r1 : Raffle := ⟨3, 1, 4⟩
  let r2 : Raffle := ⟨6, 2, 4⟩
  (first_person_wins r1 = 1/4) ∧
  (last_person_wins r1 = 1/4) ∧
  (first_person_wins_continued r2 = 5/14) ∧
  (last_person_wins_continued r2 = 1/7) :=
sorry

end NUMINAMATH_CALUDE_raffle_probabilities_l3921_392154


namespace NUMINAMATH_CALUDE_prayer_difference_l3921_392145

/-- Represents the number of prayers for a pastor in a week -/
structure WeeklyPrayers where
  regularDays : ℕ
  sunday : ℕ

/-- Calculates the total number of prayers in a week -/
def totalPrayers (wp : WeeklyPrayers) : ℕ :=
  wp.regularDays * 6 + wp.sunday

/-- Pastor Paul's prayer schedule -/
def paulPrayers : WeeklyPrayers where
  regularDays := 20
  sunday := 40

/-- Pastor Bruce's prayer schedule -/
def brucePrayers : WeeklyPrayers where
  regularDays := paulPrayers.regularDays / 2
  sunday := paulPrayers.sunday * 2

theorem prayer_difference : 
  totalPrayers paulPrayers - totalPrayers brucePrayers = 20 := by
  sorry


end NUMINAMATH_CALUDE_prayer_difference_l3921_392145


namespace NUMINAMATH_CALUDE_base_10_to_base_12_153_l3921_392108

def base_12_digit (n : ℕ) : Char :=
  if n < 10 then Char.ofNat (n + 48)
  else if n = 10 then 'A'
  else 'B'

def to_base_12 (n : ℕ) : String :=
  let d₁ := n / 12
  let d₀ := n % 12
  String.mk [base_12_digit d₁, base_12_digit d₀]

theorem base_10_to_base_12_153 :
  to_base_12 153 = "B9" := by
  sorry

end NUMINAMATH_CALUDE_base_10_to_base_12_153_l3921_392108


namespace NUMINAMATH_CALUDE_swimmer_speed_l3921_392143

/-- The speed of a swimmer in still water, given downstream and upstream distances and times. -/
theorem swimmer_speed (downstream_distance upstream_distance : ℝ) 
  (downstream_time upstream_time : ℝ) (h1 : downstream_distance = 36)
  (h2 : upstream_distance = 26) (h3 : downstream_time = 2) (h4 : upstream_time = 2) :
  ∃ (speed_still : ℝ), speed_still = 15.5 ∧ 
  downstream_distance / downstream_time = speed_still + (downstream_distance - upstream_distance) / (downstream_time + upstream_time) ∧
  upstream_distance / upstream_time = speed_still - (downstream_distance - upstream_distance) / (downstream_time + upstream_time) :=
by sorry

end NUMINAMATH_CALUDE_swimmer_speed_l3921_392143

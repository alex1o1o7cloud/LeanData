import Mathlib

namespace biased_coin_probability_l1213_121331

theorem biased_coin_probability (p : ℝ) : 
  p < (1 : ℝ) / 2 →
  6 * p^2 * (1 - p)^2 = (1 : ℝ) / 6 →
  p = (3 - Real.sqrt 3) / 6 := by
  sorry

end biased_coin_probability_l1213_121331


namespace squat_lift_loss_percentage_l1213_121381

/-- Calculates the percentage of squat lift lost given the original lifts and new total lift -/
theorem squat_lift_loss_percentage
  (orig_squat : ℝ)
  (orig_bench : ℝ)
  (orig_deadlift : ℝ)
  (deadlift_loss : ℝ)
  (new_total : ℝ)
  (h1 : orig_squat = 700)
  (h2 : orig_bench = 400)
  (h3 : orig_deadlift = 800)
  (h4 : deadlift_loss = 200)
  (h5 : new_total = 1490) :
  (orig_squat - (new_total - (orig_bench + (orig_deadlift - deadlift_loss)))) / orig_squat * 100 = 30 :=
by sorry

end squat_lift_loss_percentage_l1213_121381


namespace range_of_m_l1213_121306

-- Define p and q as functions of x and m
def p (x : ℝ) : Prop := |x - 3| ≤ 2

def q (x m : ℝ) : Prop := (x - m + 1) * (x - m - 1) ≤ 0

-- Define the necessary but not sufficient condition
def necessary_not_sufficient (p q : Prop) : Prop :=
  (q → p) ∧ ¬(p → q)

-- State the theorem
theorem range_of_m :
  ∀ m : ℝ, (∀ x : ℝ, necessary_not_sufficient (¬(p x)) (¬(q x m))) →
  (m ≥ 2 ∧ m ≤ 4) :=
sorry

end range_of_m_l1213_121306


namespace stable_performance_comparison_l1213_121340

/-- Represents a shooter's performance statistics -/
structure ShooterStats where
  average_score : ℝ
  variance : ℝ
  variance_nonneg : 0 ≤ variance

/-- Defines the concept of stability based on variance -/
def more_stable (a b : ShooterStats) : Prop :=
  a.variance < b.variance

theorem stable_performance_comparison 
  (A B : ShooterStats)
  (h_avg : A.average_score = B.average_score)
  (h_var_A : A.variance = 0.4)
  (h_var_B : B.variance = 3.2) :
  more_stable A B :=
sorry

end stable_performance_comparison_l1213_121340


namespace value_of_a_minus_b_l1213_121330

theorem value_of_a_minus_b (a b : ℝ) 
  (ha : |a| = 5) 
  (hb : |b| = 4) 
  (hab : a + b < 0) : 
  a - b = -9 ∨ a - b = -1 := by
  sorry

end value_of_a_minus_b_l1213_121330


namespace problem_statement_l1213_121349

theorem problem_statement : (-1)^49 + 2^(3^3 + 5^2 - 48^2) = -1 + 1 / 2^2252 := by
  sorry

end problem_statement_l1213_121349


namespace rakesh_distance_l1213_121345

/-- Represents the walking problem with four people: Hiro, Rakesh, Sanjay, and Charu -/
structure WalkingProblem where
  hiro_distance : ℝ
  total_distance : ℝ
  total_time : ℝ

/-- The conditions of the walking problem -/
def walking_conditions (wp : WalkingProblem) : Prop :=
  wp.total_distance = 85 ∧
  wp.total_time = 20 ∧
  ∃ (rakesh_time sanjay_time charu_time : ℝ),
    rakesh_time = wp.total_time - (wp.total_time - 2) - sanjay_time - charu_time ∧
    charu_time = wp.total_time - (wp.total_time - 2) ∧
    wp.total_distance = wp.hiro_distance + (4 * wp.hiro_distance - 10) + (2 * wp.hiro_distance + 3) +
      ((4 * wp.hiro_distance - 10) + (2 * wp.hiro_distance + 3)) / 2

/-- The theorem stating Rakesh's walking distance -/
theorem rakesh_distance (wp : WalkingProblem) (h : walking_conditions wp) :
    4 * wp.hiro_distance - 10 = 28.2 := by
  sorry


end rakesh_distance_l1213_121345


namespace least_repeating_digits_seven_thirteenths_l1213_121303

/-- The least number of digits in a repeating block of 7/13 -/
def leastRepeatingDigits : ℕ := 6

/-- 7/13 is a repeating decimal -/
axiom seven_thirteenths_repeating : ∃ (n : ℕ) (k : ℕ+), (7 : ℚ) / 13 = ↑n / (10 ^ k.val - 1)

theorem least_repeating_digits_seven_thirteenths :
  leastRepeatingDigits = 6 ∧
  ∀ m : ℕ, m < leastRepeatingDigits → ¬∃ (n : ℕ) (k : ℕ+), (7 : ℚ) / 13 = ↑n / (10 ^ m - 1) :=
sorry

end least_repeating_digits_seven_thirteenths_l1213_121303


namespace binomial_expansion_example_l1213_121312

theorem binomial_expansion_example : 
  8^4 + 4*(8^3)*2 + 6*(8^2)*(2^2) + 4*8*(2^3) + 2^4 = 10000 := by
  sorry

end binomial_expansion_example_l1213_121312


namespace fraction_equality_l1213_121344

theorem fraction_equality : (8 : ℝ) / (5 * 42) = 0.8 / (2.1 * 10) := by
  sorry

end fraction_equality_l1213_121344


namespace minimize_distance_sum_l1213_121319

/-- Given points P, Q, and R in a coordinate plane, prove that the value of m 
    that minimizes the sum of distances PR + QR is 7/2, under specific conditions. -/
theorem minimize_distance_sum (P Q R : ℝ × ℝ) (x m : ℝ) : 
  P = (7, 7) →
  Q = (3, 2) →
  R = (x, m) →
  ((-7 : ℝ), 7) ∈ {(x, y) | y = 3*x - 4} →
  (∀ m' : ℝ, 
    Real.sqrt ((7 - x)^2 + (7 - m')^2) + Real.sqrt ((3 - x)^2 + (2 - m')^2) ≥ 
    Real.sqrt ((7 - x)^2 + (7 - m)^2) + Real.sqrt ((3 - x)^2 + (2 - m)^2)) →
  m = 7/2 := by
sorry

end minimize_distance_sum_l1213_121319


namespace cookies_remaining_l1213_121398

-- Define the given conditions
def pieces_per_pack : ℕ := 3
def original_packs : ℕ := 226
def packs_given_away : ℕ := 3

-- Define the theorem
theorem cookies_remaining :
  (original_packs - packs_given_away) * pieces_per_pack = 669 := by
  sorry

end cookies_remaining_l1213_121398


namespace min_value_a_plus_b_l1213_121365

theorem min_value_a_plus_b (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a * b = 2 * a + b) :
  ∀ x y : ℝ, x > 0 → y > 0 → x * y = 2 * x + y → a + b ≤ x + y ∧ a + b = 2 * Real.sqrt 2 + 3 :=
by sorry

end min_value_a_plus_b_l1213_121365


namespace speed_difference_l1213_121350

theorem speed_difference (distance : ℝ) (emma_time lucas_time : ℝ) 
  (h1 : distance = 8)
  (h2 : emma_time = 12 / 60)
  (h3 : lucas_time = 40 / 60) :
  (distance / emma_time) - (distance / lucas_time) = 28 := by
  sorry

end speed_difference_l1213_121350


namespace polynomial_sum_of_coefficients_l1213_121367

theorem polynomial_sum_of_coefficients :
  ∀ (a a₁ a₂ a₃ a₄ a₅ : ℝ),
  (∀ x : ℝ, x^5 + 1 = a + a₁*(x-1) + a₂*(x-1)^2 + a₃*(x-1)^3 + a₄*(x-1)^4 + a₅*(x-1)^5) →
  a + a₁ + a₂ + a₃ + a₄ + a₅ = 33 := by
sorry

end polynomial_sum_of_coefficients_l1213_121367


namespace boat_distance_along_stream_l1213_121328

/-- The distance traveled by a boat along a stream in one hour -/
def distance_along_stream (boat_speed : ℝ) (against_stream_distance : ℝ) : ℝ :=
  boat_speed + (boat_speed - against_stream_distance)

/-- Theorem: The boat travels 11 km along the stream in one hour -/
theorem boat_distance_along_stream :
  distance_along_stream 9 7 = 11 := by
  sorry

end boat_distance_along_stream_l1213_121328


namespace gcd_power_two_minus_one_l1213_121347

theorem gcd_power_two_minus_one (a b : ℕ+) :
  Nat.gcd ((2 : ℕ) ^ a.val - 1) ((2 : ℕ) ^ b.val - 1) = (2 : ℕ) ^ (Nat.gcd a.val b.val) - 1 := by
  sorry

end gcd_power_two_minus_one_l1213_121347


namespace inscribed_equilateral_triangle_in_five_moves_l1213_121359

/-- Represents a point in the plane -/
structure Point :=
  (x : ℝ)
  (y : ℝ)

/-- Represents a circle in the plane -/
structure Circle :=
  (center : Point)
  (radius : ℝ)

/-- Represents a line in the plane -/
structure Line :=
  (a : ℝ)
  (b : ℝ)
  (c : ℝ)

/-- Represents the game state -/
structure GameState :=
  (knownPoints : Set Point)
  (lines : Set Line)
  (circles : Set Circle)

/-- Represents a move in the game -/
inductive Move
  | DrawLine (p1 p2 : Point)
  | DrawCircle (center : Point) (throughPoint : Point)

/-- Checks if a point is on a circle -/
def isOnCircle (p : Point) (c : Circle) : Prop :=
  (p.x - c.center.x)^2 + (p.y - c.center.y)^2 = c.radius^2

/-- Checks if three points form an equilateral triangle -/
def isEquilateralTriangle (p1 p2 p3 : Point) : Prop :=
  let d12 := ((p1.x - p2.x)^2 + (p1.y - p2.y)^2)
  let d23 := ((p2.x - p3.x)^2 + (p2.y - p3.y)^2)
  let d31 := ((p3.x - p1.x)^2 + (p3.y - p1.y)^2)
  d12 = d23 ∧ d23 = d31

/-- The main theorem -/
theorem inscribed_equilateral_triangle_in_five_moves 
  (initialCircle : Circle) (initialPoint : Point) 
  (h : isOnCircle initialPoint initialCircle) :
  ∃ (moves : List Move) (p1 p2 p3 : Point),
    moves.length = 5 ∧
    isEquilateralTriangle p1 p2 p3 ∧
    isOnCircle p1 initialCircle ∧
    isOnCircle p2 initialCircle ∧
    isOnCircle p3 initialCircle :=
  sorry

end inscribed_equilateral_triangle_in_five_moves_l1213_121359


namespace sqrt_difference_equals_two_sqrt_three_l1213_121366

theorem sqrt_difference_equals_two_sqrt_three :
  Real.sqrt (7 + 4 * Real.sqrt 3) - Real.sqrt (7 - 4 * Real.sqrt 3) = 2 * Real.sqrt 3 := by
  sorry

end sqrt_difference_equals_two_sqrt_three_l1213_121366


namespace s_1000_eq_720_l1213_121378

def s : ℕ → ℕ
  | 0 => 1
  | 1 => 1
  | n + 2 => 
    if n % 2 = 0 then s (n / 2)
    else if (n - 1) % 4 = 0 then s ((n - 1) / 2 + 1)
    else s ((n + 1) / 2 - 1) + (s ((n + 1) / 2 - 1))^2 / s ((n + 1) / 4 - 1)

theorem s_1000_eq_720 : s 1000 = 720 := by
  sorry

end s_1000_eq_720_l1213_121378


namespace hall_width_proof_l1213_121336

/-- Given a rectangular hall with specified dimensions and cost constraints, 
    prove that the width of the hall is 15 meters. -/
theorem hall_width_proof (length height : ℝ) (cost_per_sqm total_cost : ℝ) :
  length = 20 →
  height = 5 →
  cost_per_sqm = 30 →
  total_cost = 28500 →
  ∃ w : ℝ, w > 0 ∧ 
    (2 * length * w + 2 * length * height + 2 * w * height) * cost_per_sqm = total_cost ∧
    w = 15 := by
  sorry

#check hall_width_proof

end hall_width_proof_l1213_121336


namespace cans_in_cat_package_l1213_121368

/-- Represents the number of cans in each package of cat food -/
def cans_per_cat_package : ℕ := sorry

/-- The number of packages of cat food Adam bought -/
def cat_packages : ℕ := 9

/-- The number of packages of dog food Adam bought -/
def dog_packages : ℕ := 7

/-- The number of cans in each package of dog food -/
def cans_per_dog_package : ℕ := 5

/-- The difference between the total number of cat food cans and dog food cans -/
def can_difference : ℕ := 55

theorem cans_in_cat_package : 
  cans_per_cat_package * cat_packages = 
  cans_per_dog_package * dog_packages + can_difference ∧ 
  cans_per_cat_package = 10 := by sorry

end cans_in_cat_package_l1213_121368


namespace not_perfect_square_l1213_121397

-- Define the number with 300 ones followed by zeros
def number_with_300_ones : ℕ → ℕ 
  | n => 10^n * (10^300 - 1) / 9

-- Theorem statement
theorem not_perfect_square (n : ℕ) : 
  ¬ ∃ (m : ℕ), number_with_300_ones n = m^2 := by
  sorry


end not_perfect_square_l1213_121397


namespace percentage_with_no_conditions_is_22_5_l1213_121394

/-- Represents the survey results of teachers' health conditions -/
structure SurveyResults where
  total : ℕ
  highBloodPressure : ℕ
  heartTrouble : ℕ
  diabetes : ℕ
  highBloodPressureAndHeartTrouble : ℕ
  highBloodPressureAndDiabetes : ℕ
  heartTroubleAndDiabetes : ℕ
  allThree : ℕ

/-- Calculates the percentage of teachers with none of the health conditions -/
def percentageWithNoConditions (results : SurveyResults) : ℚ :=
  let withConditions :=
    results.highBloodPressure +
    results.heartTrouble +
    results.diabetes -
    results.highBloodPressureAndHeartTrouble -
    results.highBloodPressureAndDiabetes -
    results.heartTroubleAndDiabetes +
    results.allThree
  let withoutConditions := results.total - withConditions
  (withoutConditions : ℚ) / results.total * 100

/-- The survey results from the problem -/
def surveyData : SurveyResults :=
  { total := 200
  , highBloodPressure := 90
  , heartTrouble := 60
  , diabetes := 30
  , highBloodPressureAndHeartTrouble := 25
  , highBloodPressureAndDiabetes := 15
  , heartTroubleAndDiabetes := 10
  , allThree := 5 }

theorem percentage_with_no_conditions_is_22_5 :
  percentageWithNoConditions surveyData = 22.5 := by
  sorry

end percentage_with_no_conditions_is_22_5_l1213_121394


namespace alternating_arrangements_2_3_l1213_121355

/-- The number of ways to arrange m men and w women in a row, such that no two men or two women are adjacent -/
def alternating_arrangements (m : ℕ) (w : ℕ) : ℕ := sorry

theorem alternating_arrangements_2_3 :
  alternating_arrangements 2 3 = 24 := by sorry

end alternating_arrangements_2_3_l1213_121355


namespace indira_cricket_time_l1213_121392

/-- Sean's daily cricket playing time in minutes -/
def sean_daily_time : ℕ := 50

/-- Number of days Sean played cricket -/
def sean_days : ℕ := 14

/-- Total time Sean and Indira played cricket together in minutes -/
def total_time : ℕ := 1512

/-- Calculate Indira's cricket playing time -/
def indira_time : ℕ := total_time - (sean_daily_time * sean_days)

/-- Theorem stating Indira's cricket playing time -/
theorem indira_cricket_time : indira_time = 812 := by sorry

end indira_cricket_time_l1213_121392


namespace parallel_condition_l1213_121383

/-- Two lines in R² are parallel if and only if their slopes are equal -/
def are_parallel (a b c d e f : ℝ) : Prop :=
  (a * f = b * d) ∧ (a * e ≠ b * c ∨ c * f ≠ d * e)

/-- The condition for two lines to be parallel -/
theorem parallel_condition (a : ℝ) :
  (∀ x y : ℝ, are_parallel a 1 (-1) 1 a 1) ↔ a = 1 := by
  sorry

end parallel_condition_l1213_121383


namespace apartment_price_ratio_l1213_121389

theorem apartment_price_ratio :
  ∀ (a b : ℝ),
  a > 0 → b > 0 →
  1.21 * a + 1.11 * b = 1.15 * (a + b) →
  b / a = 1.5 := by
sorry

end apartment_price_ratio_l1213_121389


namespace segments_complete_circle_num_segments_minimal_l1213_121343

/-- The number of equal segments that can be drawn around a circle,
    where each segment subtends an arc of 120°. -/
def num_segments : ℕ := 3

/-- The measure of the arc subtended by each segment in degrees. -/
def arc_measure : ℕ := 120

/-- Theorem stating that the number of segments multiplied by the arc measure
    equals a full circle (360°). -/
theorem segments_complete_circle :
  num_segments * arc_measure = 360 := by sorry

/-- Theorem stating that num_segments is the smallest positive integer
    that satisfies the segments_complete_circle property. -/
theorem num_segments_minimal :
  ∀ n : ℕ, 0 < n → n * arc_measure = 360 → num_segments ≤ n := by sorry

end segments_complete_circle_num_segments_minimal_l1213_121343


namespace smallest_integer_satisfying_conditions_l1213_121300

theorem smallest_integer_satisfying_conditions :
  ∃ x : ℤ, (3 * |x| + 4 < 25) ∧ (x + 3 > 0) ∧
  (∀ y : ℤ, (3 * |y| + 4 < 25) ∧ (y + 3 > 0) → x ≤ y) ∧
  x = -3 := by
  sorry

end smallest_integer_satisfying_conditions_l1213_121300


namespace inequality_proof_l1213_121308

theorem inequality_proof (a b c : ℝ) 
  (ha : a > 0) (hb : b > 0) (hc : c > 0) 
  (sum_eq_one : a + b + c = 1) :
  1 / (1 - a) + 1 / (1 - b) + 1 / (1 - c) ≥ 
  2 / (1 + a) + 2 / (1 + b) + 2 / (1 + c) := by
  sorry

end inequality_proof_l1213_121308


namespace sum_of_digits_l1213_121332

/-- Given a three-digit number of the form 3a7 and another three-digit number 7c1,
    where a and c are single digits, prove that if 3a7 + 414 = 7c1 and 7c1 is
    divisible by 11, then a + c = 14. -/
theorem sum_of_digits (a c : ℕ) : 
  (a < 10) →
  (c < 10) →
  (300 + 10 * a + 7 + 414 = 700 + 10 * c + 1) →
  (700 + 10 * c + 1) % 11 = 0 →
  a + c = 14 := by
sorry

end sum_of_digits_l1213_121332


namespace boat_upstream_distance_l1213_121380

/-- Calculates the upstream distance traveled by a boat in one hour -/
def upstreamDistance (stillWaterSpeed : ℝ) (downstreamDistance : ℝ) : ℝ :=
  let streamSpeed := downstreamDistance - stillWaterSpeed
  stillWaterSpeed - streamSpeed

theorem boat_upstream_distance :
  upstreamDistance 5 8 = 2 := by
  sorry

#eval upstreamDistance 5 8

end boat_upstream_distance_l1213_121380


namespace differential_system_properties_l1213_121317

-- Define the system of differential equations
def system_ode (u : ℝ → ℝ) (x y : ℝ → ℝ) : Prop :=
  ∀ t, deriv x t = -2 * y t + u t ∧ deriv y t = -2 * x t + u t

-- Define the theorem
theorem differential_system_properties
  (u : ℝ → ℝ) (x y : ℝ → ℝ) (x₀ y₀ : ℝ)
  (h_cont : Continuous u)
  (h_system : system_ode u x y)
  (h_init : x 0 = x₀ ∧ y 0 = y₀) :
  (x₀ ≠ y₀ → ∀ t, x t - y t ≠ 0) ∧
  (x₀ = y₀ → ∀ T > 0, ∃ u : ℝ → ℝ, Continuous u ∧ x T = 0 ∧ y T = 0) :=
sorry

end differential_system_properties_l1213_121317


namespace sqrt_factorial_squared_l1213_121385

-- Define factorial function
def factorial (n : ℕ) : ℕ :=
  match n with
  | 0 => 1
  | n + 1 => (n + 1) * factorial n

-- State the theorem
theorem sqrt_factorial_squared :
  (((factorial 5 * factorial 4 : ℕ) : ℝ).sqrt ^ 2 : ℝ) = 2880 := by
  sorry

end sqrt_factorial_squared_l1213_121385


namespace smallest_b_in_arithmetic_series_l1213_121386

theorem smallest_b_in_arithmetic_series (a b c : ℝ) : 
  0 < a ∧ 0 < b ∧ 0 < c →  -- all terms are positive
  (∃ d : ℝ, a = b - d ∧ c = b + d) →  -- arithmetic series condition
  a * b * c = 125 →  -- product condition
  ∀ x : ℝ, (∃ y z : ℝ, 
    0 < y ∧ 0 < x ∧ 0 < z ∧  -- positivity for new terms
    (∃ e : ℝ, y = x - e ∧ z = x + e) ∧  -- arithmetic series for new terms
    y * x * z = 125) →  -- product condition for new terms
  x ≥ b →
  b ≥ 5 :=
sorry

end smallest_b_in_arithmetic_series_l1213_121386


namespace honey_jar_theorem_l1213_121384

def initial_honey : ℝ := 1.2499999999999998
def draw_percentage : ℝ := 0.20
def num_iterations : ℕ := 4

def honey_left (initial : ℝ) (draw : ℝ) (iterations : ℕ) : ℝ :=
  initial * (1 - draw) ^ iterations

theorem honey_jar_theorem :
  honey_left initial_honey draw_percentage num_iterations = 0.512 := by
  sorry

end honey_jar_theorem_l1213_121384


namespace xy_sum_difference_l1213_121304

theorem xy_sum_difference (x y : ℝ) 
  (h1 : x + Real.sqrt (x * y) + y = 9) 
  (h2 : x^2 + x*y + y^2 = 27) : 
  x - Real.sqrt (x * y) + y = 3 := by
sorry

end xy_sum_difference_l1213_121304


namespace train_sequence_count_l1213_121322

/-- The number of ways to choose k items from n items -/
def choose (n k : ℕ) : ℕ := sorry

/-- The factorial of a natural number -/
def factorial (n : ℕ) : ℕ := sorry

/-- The total number of departure sequences for 6 trains under given conditions -/
def train_sequences : ℕ :=
  let total_trains : ℕ := 6
  let trains_per_group : ℕ := 3
  let remaining_trains : ℕ := total_trains - 2  -- excluding A and B
  let ways_to_group : ℕ := choose remaining_trains (trains_per_group - 1)
  let ways_to_arrange_group : ℕ := factorial trains_per_group
  ways_to_group * ways_to_arrange_group * ways_to_arrange_group

theorem train_sequence_count : train_sequences = 216 := by sorry

end train_sequence_count_l1213_121322


namespace sanhat_integers_l1213_121309

theorem sanhat_integers (x y : ℤ) (h1 : 3 * x + 2 * y = 160) (h2 : x = 36 ∨ y = 36) :
  (x = 36 ∧ y = 26) ∨ (y = 36 ∧ x = 26) :=
sorry

end sanhat_integers_l1213_121309


namespace tangent_slope_angle_at_zero_l1213_121351

noncomputable def f (x : ℝ) : ℝ := Real.exp x

theorem tangent_slope_angle_at_zero :
  let slope := deriv f 0
  Real.arctan slope = π / 4 := by
  sorry

end tangent_slope_angle_at_zero_l1213_121351


namespace didi_fundraiser_amount_l1213_121374

/-- Calculates the total amount raised from cake sales and donations --/
def total_amount_raised (num_cakes : ℕ) (slices_per_cake : ℕ) (price_per_slice : ℚ) 
  (donation1_per_slice : ℚ) (donation2_per_slice : ℚ) : ℚ :=
  let total_slices := num_cakes * slices_per_cake
  let sales_amount := total_slices * price_per_slice
  let donation1_amount := total_slices * donation1_per_slice
  let donation2_amount := total_slices * donation2_per_slice
  sales_amount + donation1_amount + donation2_amount

/-- Theorem stating that under the given conditions, the total amount raised is $140 --/
theorem didi_fundraiser_amount :
  total_amount_raised 10 8 1 (1/2) (1/4) = 140 := by
  sorry

end didi_fundraiser_amount_l1213_121374


namespace jelly_bean_count_l1213_121333

/-- The number of jelly beans in the jar. -/
def total_jelly_beans : ℕ := 200

/-- Thomas's share of jelly beans as a fraction. -/
def thomas_share : ℚ := 1/10

/-- The ratio of Barry's share to Emmanuel's share. -/
def barry_emmanuel_ratio : ℚ := 4/5

/-- Emmanuel's share of jelly beans. -/
def emmanuel_share : ℕ := 100

/-- Theorem stating the total number of jelly beans in the jar. -/
theorem jelly_bean_count :
  total_jelly_beans = 200 ∧
  thomas_share = 1/10 ∧
  barry_emmanuel_ratio = 4/5 ∧
  emmanuel_share = 100 ∧
  emmanuel_share = (5/9 : ℚ) * ((1 - thomas_share) * total_jelly_beans) :=
by sorry

end jelly_bean_count_l1213_121333


namespace minotaur_returns_l1213_121391

/-- A room in the Minotaur's palace -/
structure Room where
  id : Nat

/-- A direction the Minotaur can turn -/
inductive Direction
  | Left
  | Right

/-- The state of the Minotaur's journey -/
structure State where
  room : Room
  enteredThrough : Nat
  nextTurn : Direction

/-- The palace with its room connections -/
structure Palace where
  rooms : Finset Room
  connections : Room → Finset (Nat × Room)
  room_count : rooms.card = 1000000
  three_corridors : ∀ r : Room, (connections r).card = 3

/-- The function that determines the next state based on the current state -/
def nextState (p : Palace) (s : State) : State :=
  sorry

/-- The theorem stating that the Minotaur will eventually return to the starting room -/
theorem minotaur_returns (p : Palace) (start : State) :
  ∃ n : Nat, (Nat.iterate (nextState p) n start).room = start.room :=
sorry

end minotaur_returns_l1213_121391


namespace data_set_average_l1213_121360

theorem data_set_average (x : ℝ) : 
  (2 + 1 + 4 + x + 6) / 5 = 4 → x = 7 := by
  sorry

end data_set_average_l1213_121360


namespace sum_divisible_by_ten_l1213_121357

theorem sum_divisible_by_ten : ∃ k : ℤ, 111^111 + 112^112 + 113^113 = 10 * k := by
  sorry

end sum_divisible_by_ten_l1213_121357


namespace lunas_budget_l1213_121301

/-- Luna's monthly budget problem -/
theorem lunas_budget (H F : ℝ) : 
  H + F = 240 →  -- Total budget for house rental and food
  H + F + 0.1 * F = 249 →  -- Total budget including phone bill
  F / H = 0.6  -- Food budget is 60% of house rental budget
:= by sorry

end lunas_budget_l1213_121301


namespace sixDigitPermutations_eq_90_l1213_121307

/-- The number of different positive, six-digit integers that can be formed using the digits 1, 1, 3, 3, 7, and 7 -/
def sixDigitPermutations : ℕ :=
  Nat.factorial 6 / (Nat.factorial 2 * Nat.factorial 2 * Nat.factorial 2)

/-- Theorem stating that the number of such permutations is 90 -/
theorem sixDigitPermutations_eq_90 : sixDigitPermutations = 90 := by
  sorry

end sixDigitPermutations_eq_90_l1213_121307


namespace xy_squared_sum_l1213_121311

theorem xy_squared_sum (x y : ℝ) (h1 : x + y = 2) (h2 : x * y = 3) :
  x^2 * y + x * y^2 = 6 := by
  sorry

end xy_squared_sum_l1213_121311


namespace symmetric_line_equation_l1213_121356

/-- A line in the xy-plane represented by its slope and y-intercept -/
structure Line where
  slope : ℝ
  intercept : ℝ

/-- The equation of a line in slope-intercept form -/
def Line.equation (l : Line) (x y : ℝ) : Prop :=
  y = l.slope * x + l.intercept

/-- Two lines are symmetric with respect to the y-axis -/
def symmetric_about_y_axis (l₁ l₂ : Line) : Prop :=
  l₁.slope = -l₂.slope ∧ l₁.intercept = l₂.intercept

theorem symmetric_line_equation (l₁ l₂ : Line) :
  l₁.equation x y = (y = 2 * x + 3) →
  symmetric_about_y_axis l₁ l₂ →
  l₂.equation x y = (y = -2 * x + 3) := by
  sorry

end symmetric_line_equation_l1213_121356


namespace simplify_fraction_l1213_121363

theorem simplify_fraction : 
  (1 / ((1 / (Real.sqrt 3 + 1)) + (3 / (Real.sqrt 5 - 2)))) = 
  (2 / (Real.sqrt 3 + 6 * Real.sqrt 5 + 11)) := by sorry

end simplify_fraction_l1213_121363


namespace smallest_block_size_l1213_121353

/-- 
Given a rectangular block with dimensions a × b × c formed by N congruent 1-cm cubes,
where (a-1)(b-1)(c-1) = 252, the smallest possible value of N is 224.
-/
theorem smallest_block_size (a b c N : ℕ) : 
  (a - 1) * (b - 1) * (c - 1) = 252 → 
  N = a * b * c → 
  (∀ a' b' c' N', (a' - 1) * (b' - 1) * (c' - 1) = 252 → N' = a' * b' * c' → N ≤ N') →
  N = 224 :=
by sorry

end smallest_block_size_l1213_121353


namespace admin_personnel_count_l1213_121329

/-- Represents the total number of employees in the unit -/
def total_employees : ℕ := 280

/-- Represents the sample size -/
def sample_size : ℕ := 56

/-- Represents the number of ordinary staff sampled -/
def ordinary_staff_sampled : ℕ := 49

/-- Calculates the number of administrative personnel -/
def admin_personnel : ℕ := total_employees - (total_employees * ordinary_staff_sampled / sample_size)

/-- Theorem stating that the number of administrative personnel is 35 -/
theorem admin_personnel_count : admin_personnel = 35 := by
  sorry

end admin_personnel_count_l1213_121329


namespace A_minus_2B_value_of_2B_minus_A_l1213_121388

/-- Given two expressions A and B in terms of a and b -/
def A (a b : ℝ) : ℝ := 2*a^2 + a*b + 3*b

def B (a b : ℝ) : ℝ := a^2 - a*b + a

/-- Theorem stating the equality of A - 2B and its simplified form -/
theorem A_minus_2B (a b : ℝ) : A a b - 2 * B a b = 3*a*b + 3*b - 2*a := by sorry

/-- Theorem stating the value of 2B - A under the given condition -/
theorem value_of_2B_minus_A (a b : ℝ) (h : (a + 1)^2 + |b - 3| = 0) : 
  2 * B a b - A a b = -2 := by sorry

end A_minus_2B_value_of_2B_minus_A_l1213_121388


namespace surface_area_of_specific_solid_l1213_121316

/-- A right prism with equilateral triangular bases -/
structure RightPrism where
  height : ℝ
  base_side : ℝ

/-- A solid formed by slicing off the top of the prism -/
structure SlicedSolid where
  prism : RightPrism

/-- The surface area of the sliced solid -/
def surface_area (solid : SlicedSolid) : ℝ :=
  sorry

/-- Theorem stating the surface area of the specific sliced solid -/
theorem surface_area_of_specific_solid :
  let prism := RightPrism.mk 20 10
  let solid := SlicedSolid.mk prism
  surface_area solid = 50 + (25 * Real.sqrt 3) / 4 + (5 * Real.sqrt 118.75) / 2 := by
  sorry

end surface_area_of_specific_solid_l1213_121316


namespace no_sum_of_150_consecutive_integers_l1213_121390

theorem no_sum_of_150_consecutive_integers : ¬ ∃ (k : ℤ),
  (150 * k + 11325 = 678900) ∨
  (150 * k + 11325 = 1136850) ∨
  (150 * k + 11325 = 1000000) ∨
  (150 * k + 11325 = 2251200) ∨
  (150 * k + 11325 = 1876800) :=
by sorry

end no_sum_of_150_consecutive_integers_l1213_121390


namespace colorings_theorem_l1213_121320

/-- The number of ways to color five cells in a 5x5 grid with one colored cell in each row and column. -/
def total_colorings : ℕ := 120

/-- The number of ways to color five cells in a 5x5 grid without one corner cell, 
    with one colored cell in each row and column. -/
def colorings_without_one_corner : ℕ := 96

/-- The number of ways to color five cells in a 5x5 grid without two corner cells, 
    with one colored cell in each row and column. -/
def colorings_without_two_corners : ℕ := 78

theorem colorings_theorem : 
  colorings_without_two_corners = total_colorings - 2 * (total_colorings - colorings_without_one_corner) + 6 :=
by sorry

end colorings_theorem_l1213_121320


namespace incorrect_statement_l1213_121346

theorem incorrect_statement : ¬(
  (∀ x : ℝ, x ∈ [0, 1] → Real.exp x ≥ 1) ∧
  (∃ x : ℝ, x^2 + x + 1 < 0)
) := by sorry

end incorrect_statement_l1213_121346


namespace window_area_theorem_l1213_121379

/-- Represents a rectangular glass pane with length and width in inches. -/
structure GlassPane where
  length : ℕ
  width : ℕ

/-- Calculates the area of a single glass pane in square inches. -/
def pane_area (pane : GlassPane) : ℕ :=
  pane.length * pane.width

/-- Represents a window composed of multiple identical glass panes. -/
structure Window where
  pane : GlassPane
  num_panes : ℕ

/-- Calculates the total area of a window in square inches. -/
def window_area (w : Window) : ℕ :=
  pane_area w.pane * w.num_panes

/-- Theorem: The area of a window with 8 panes, each 12 inches by 8 inches, is 768 square inches. -/
theorem window_area_theorem : 
  ∀ (w : Window), w.pane.length = 12 → w.pane.width = 8 → w.num_panes = 8 → 
  window_area w = 768 := by
  sorry

end window_area_theorem_l1213_121379


namespace cricket_team_age_difference_l1213_121321

theorem cricket_team_age_difference (team_size : ℕ) (captain_age : ℕ) (keeper_age_diff : ℕ) (team_avg_age : ℚ) :
  team_size = 11 →
  captain_age = 26 →
  keeper_age_diff = 3 →
  team_avg_age = 23 →
  let keeper_age := captain_age + keeper_age_diff
  let total_team_age := team_avg_age * team_size
  let remaining_players := team_size - 2
  let remaining_age := total_team_age - (captain_age + keeper_age)
  let remaining_avg_age := remaining_age / remaining_players
  (team_avg_age - remaining_avg_age) = 1 := by
sorry

end cricket_team_age_difference_l1213_121321


namespace opposite_of_negative_three_l1213_121358

-- Define the concept of opposite for integers
def opposite (n : Int) : Int := -n

-- Theorem stating that the opposite of -3 is 3
theorem opposite_of_negative_three : opposite (-3) = 3 := by
  sorry

end opposite_of_negative_three_l1213_121358


namespace factorization_a_squared_minus_2a_l1213_121370

theorem factorization_a_squared_minus_2a (a : ℝ) : a^2 - 2*a = a*(a - 2) := by
  sorry

end factorization_a_squared_minus_2a_l1213_121370


namespace power_of_product_l1213_121302

theorem power_of_product (a b : ℝ) : (a * b) ^ 3 = a ^ 3 * b ^ 3 := by
  sorry

end power_of_product_l1213_121302


namespace set_operations_and_subset_l1213_121396

open Set

def A : Set ℝ := {x | 1 ≤ x ∧ x ≤ 3}
def B : Set ℝ := {x | x > 2}
def C (a : ℝ) : Set ℝ := {x | 1 < x ∧ x < a}

theorem set_operations_and_subset :
  (A ∪ B = {x | 2 < x ∧ x ≤ 3}) ∧
  ((Bᶜ) ∩ A = {x | 1 ≤ x ∧ x ≤ 2}) ∧
  (∀ a : ℝ, C a ⊆ A → a ≤ 3) := by sorry

end set_operations_and_subset_l1213_121396


namespace yoongi_has_second_largest_number_l1213_121377

/-- Represents a student with their assigned number -/
structure Student where
  name : String
  number : Nat

/-- Checks if a student has the second largest number among a list of students -/
def hasSecondLargestNumber (s : Student) (students : List Student) : Prop :=
  ∃ (larger smaller : Student),
    larger ∈ students ∧
    smaller ∈ students ∧
    s ∈ students ∧
    larger.number > s.number ∧
    s.number > smaller.number ∧
    ∀ (other : Student), other ∈ students → other.number ≤ larger.number

theorem yoongi_has_second_largest_number :
  let yoongi := Student.mk "Yoongi" 7
  let jungkook := Student.mk "Jungkook" 6
  let yuna := Student.mk "Yuna" 9
  let students := [yoongi, jungkook, yuna]
  hasSecondLargestNumber yoongi students := by
  sorry

end yoongi_has_second_largest_number_l1213_121377


namespace largest_integer_with_remainder_l1213_121341

theorem largest_integer_with_remainder : 
  ∀ n : ℕ, n < 100 ∧ n % 7 = 2 → n ≤ 93 :=
by
  sorry

end largest_integer_with_remainder_l1213_121341


namespace class_size_l1213_121369

theorem class_size (total_average : ℝ) (group1_size : ℕ) (group1_average : ℝ)
                   (group2_size : ℕ) (group2_average : ℝ) (last_student_age : ℕ) :
  total_average = 15 →
  group1_size = 5 →
  group1_average = 12 →
  group2_size = 9 →
  group2_average = 16 →
  last_student_age = 21 →
  ∃ (n : ℕ), n = 15 ∧ n * total_average = group1_size * group1_average + group2_size * group2_average + last_student_age :=
by
  sorry

#check class_size

end class_size_l1213_121369


namespace cost_price_satisfies_profit_condition_l1213_121372

/-- The cost price of an article satisfies the given profit condition -/
theorem cost_price_satisfies_profit_condition (C : ℝ) : C > 0 → (0.27 * C) - (0.12 * C) = 108 ↔ C = 720 := by
  sorry

end cost_price_satisfies_profit_condition_l1213_121372


namespace ordering_of_trig_functions_l1213_121364

theorem ordering_of_trig_functions (a b c d : ℝ) : 
  a = Real.sin (Real.cos (2015 * π / 180)) →
  b = Real.sin (Real.sin (2015 * π / 180)) →
  c = Real.cos (Real.sin (2015 * π / 180)) →
  d = Real.cos (Real.cos (2015 * π / 180)) →
  c > d ∧ d > b ∧ b > a := by sorry

end ordering_of_trig_functions_l1213_121364


namespace dads_strawberry_weight_l1213_121395

/-- 
Given:
- The total initial weight of strawberries collected by Marco and his dad
- The weight of strawberries lost by Marco's dad
- The current weight of Marco's strawberries

Prove that the weight of Marco's dad's strawberries is equal to the difference between 
the total weight after loss and Marco's current weight of strawberries.
-/
theorem dads_strawberry_weight 
  (total_initial_weight : ℕ) 
  (weight_lost : ℕ) 
  (marcos_weight : ℕ) : 
  total_initial_weight - weight_lost - marcos_weight = 
    total_initial_weight - (weight_lost + marcos_weight) := by
  sorry

#check dads_strawberry_weight

end dads_strawberry_weight_l1213_121395


namespace cells_after_three_divisions_l1213_121313

/-- The number of cells after n divisions, starting with 1 cell -/
def cells_after_divisions (n : ℕ) : ℕ := 2^n

/-- Theorem: After 3 divisions, the number of cells is 8 -/
theorem cells_after_three_divisions : cells_after_divisions 3 = 8 := by
  sorry

end cells_after_three_divisions_l1213_121313


namespace price_per_deck_l1213_121399

def initial_decks : ℕ := 5
def remaining_decks : ℕ := 3
def total_earnings : ℕ := 4

theorem price_per_deck :
  (total_earnings : ℚ) / (initial_decks - remaining_decks) = 2 := by
  sorry

end price_per_deck_l1213_121399


namespace function_value_determination_l1213_121348

theorem function_value_determination (A : ℝ) (α : ℝ) 
  (h1 : A ≠ 0)
  (h2 : α ∈ Set.Icc 0 π)
  (h3 : A * Real.sin (α + π/4) = Real.cos (2*α))
  (h4 : Real.sin (2*α) = -7/9) :
  A = -4*Real.sqrt 2/3 := by
sorry

end function_value_determination_l1213_121348


namespace ellipse_theorem_l1213_121375

/-- Given an ellipse with semi-major axis a, semi-minor axis b, and eccentricity e -/
structure Ellipse where
  a : ℝ
  b : ℝ
  e : ℝ
  h_a_pos : 0 < a
  h_b_pos : 0 < b
  h_a_gt_b : b < a
  h_e_eq : e = 1/2
  h_e_def : e^2 = 1 - (b/a)^2

/-- The equation of the ellipse -/
def ellipse_equation (E : Ellipse) (x y : ℝ) : Prop :=
  x^2 / E.a^2 + y^2 / E.b^2 = 1

/-- The range of t for a line passing through (t,0) intersecting the ellipse -/
def t_range (t : ℝ) : Prop :=
  (t ≤ (4 - 6 * Real.sqrt 2) / 7 ∨ (4 + 6 * Real.sqrt 2) / 7 ≤ t) ∧ t ≠ 1

theorem ellipse_theorem (E : Ellipse) :
  (∀ x y, ellipse_equation E x y ↔ x^2 / 4 + y^2 / 3 = 1) ∧
  (∀ t, t_range t ↔
    ∃ A B : ℝ × ℝ,
      ellipse_equation E A.1 A.2 ∧
      ellipse_equation E B.1 B.2 ∧
      (A.1 - 1) * (B.1 - 1) + A.2 * B.2 = 0 ∧
      A.1 = t ∧ B.1 = t) := by
  sorry

end ellipse_theorem_l1213_121375


namespace shara_shell_count_l1213_121310

/-- Calculates the total number of shells Shara has after her vacation. -/
def total_shells (initial_shells : ℕ) (shells_per_day : ℕ) (days : ℕ) (fourth_day_shells : ℕ) : ℕ :=
  initial_shells + shells_per_day * days + fourth_day_shells

/-- Theorem stating that Shara has 41 shells after her vacation. -/
theorem shara_shell_count : 
  total_shells 20 5 3 6 = 41 := by
  sorry

end shara_shell_count_l1213_121310


namespace smallest_three_digit_multiple_of_6_with_digit_sum_12_l1213_121337

def is_three_digit (n : ℕ) : Prop := 100 ≤ n ∧ n ≤ 999

def digit_sum (n : ℕ) : ℕ :=
  (n / 100) + ((n / 10) % 10) + (n % 10)

theorem smallest_three_digit_multiple_of_6_with_digit_sum_12 :
  ∀ n : ℕ, is_three_digit n → n % 6 = 0 → digit_sum n = 12 → n ≥ 204 :=
by sorry

end smallest_three_digit_multiple_of_6_with_digit_sum_12_l1213_121337


namespace solution_set_f_leq_5_range_of_m_l1213_121339

-- Define the function f
def f (x : ℝ) : ℝ := |2*x - 1| + |2*x - 3|

-- Theorem for part (1)
theorem solution_set_f_leq_5 :
  {x : ℝ | f x ≤ 5} = {x : ℝ | -1/4 ≤ x ∧ x ≤ 9/4} :=
sorry

-- Theorem for part (2)
theorem range_of_m :
  {m : ℝ | ∀ x : ℝ, m^2 - m < f x} = {m : ℝ | -1 < m ∧ m < 2} :=
sorry

end solution_set_f_leq_5_range_of_m_l1213_121339


namespace fraction_of_fraction_of_fraction_main_theorem_l1213_121315

theorem fraction_of_fraction_of_fraction (a b c d : ℚ) :
  a * b * c * d = (a * b * c) * d := by sorry

theorem main_theorem : (1 / 2 : ℚ) * (1 / 3 : ℚ) * (1 / 6 : ℚ) * 72 = 2 := by sorry

end fraction_of_fraction_of_fraction_main_theorem_l1213_121315


namespace initial_money_calculation_l1213_121335

/-- Calculates the initial amount of money given the spending pattern and final amount --/
theorem initial_money_calculation (final_amount : ℚ) : 
  final_amount = 500 →
  ∃ initial_amount : ℚ,
    initial_amount * (1 - 1/3) * (1 - 1/5) * (1 - 1/4) = final_amount ∧
    initial_amount = 1250 :=
by sorry

end initial_money_calculation_l1213_121335


namespace vector_b_value_l1213_121352

/-- Given two vectors a and b in ℝ³, prove that b equals (-2, 4, -2) -/
theorem vector_b_value (a b : ℝ × ℝ × ℝ) : 
  a = (1, -2, 1) → a + b = (-1, 2, -1) → b = (-2, 4, -2) := by
  sorry

end vector_b_value_l1213_121352


namespace donut_circumference_ratio_l1213_121305

/-- The ratio of the outer circumference to the inner circumference of a donut-shaped object
    is equal to the ratio of their respective radii. -/
theorem donut_circumference_ratio (inner_radius outer_radius : ℝ)
  (h1 : inner_radius = 2)
  (h2 : outer_radius = 6) :
  (2 * Real.pi * outer_radius) / (2 * Real.pi * inner_radius) = outer_radius / inner_radius := by
  sorry

end donut_circumference_ratio_l1213_121305


namespace integral_x_squared_l1213_121362

theorem integral_x_squared : ∫ x in (0:ℝ)..(1:ℝ), x^2 = (1:ℝ)/3 := by sorry

end integral_x_squared_l1213_121362


namespace g_g_two_roots_l1213_121382

/-- The function g(x) defined as x^2 + 2x + c^2 -/
def g (c : ℝ) (x : ℝ) : ℝ := x^2 + 2*x + c^2

/-- The theorem stating that g(g(x)) has exactly two distinct real roots iff c = ±1 -/
theorem g_g_two_roots (c : ℝ) :
  (∃! (r₁ r₂ : ℝ), r₁ ≠ r₂ ∧ ∀ x, g c (g c x) = 0 ↔ x = r₁ ∨ x = r₂) ↔ c = 1 ∨ c = -1 :=
sorry

end g_g_two_roots_l1213_121382


namespace ellipse_major_axis_length_l1213_121318

/-- Given a right circular cylinder with radius 2 intersected by a plane forming an ellipse,
    if the major axis of the ellipse is 25% longer than the minor axis,
    then the length of the major axis is 5. -/
theorem ellipse_major_axis_length (cylinder_radius : ℝ) (minor_axis : ℝ) (major_axis : ℝ) :
  cylinder_radius = 2 →
  minor_axis = 2 * cylinder_radius →
  major_axis = 1.25 * minor_axis →
  major_axis = 5 := by
sorry

end ellipse_major_axis_length_l1213_121318


namespace square_area_on_parabola_l1213_121373

/-- The area of a square with one side on y = 7 and endpoints on y = x^2 + 4x + 3 is 32 -/
theorem square_area_on_parabola : ∃ (x₁ x₂ : ℝ),
  (x₁^2 + 4*x₁ + 3 = 7) ∧
  (x₂^2 + 4*x₂ + 3 = 7) ∧
  (x₁ ≠ x₂) ∧
  ((x₂ - x₁)^2 = 32) := by
  sorry

end square_area_on_parabola_l1213_121373


namespace quadratic_function_property_l1213_121342

/-- Given a quadratic function y = ax^2 + bx + 2 passing through (-1, 0), 
    prove that 2a - 2b = -4 -/
theorem quadratic_function_property (a b : ℝ) : 
  (∀ x y : ℝ, y = a * x^2 + b * x + 2) → 
  (0 = a * (-1)^2 + b * (-1) + 2) → 
  2 * a - 2 * b = -4 := by
  sorry

end quadratic_function_property_l1213_121342


namespace ratio_arithmetic_properties_l1213_121326

-- Define a ratio arithmetic sequence
def is_ratio_arithmetic_seq (p : ℕ → ℝ) (k : ℝ) :=
  ∀ n ≥ 2, p (n + 1) / p n - p n / p (n - 1) = k

-- Define a geometric sequence
def is_geometric_seq (a : ℕ → ℝ) (q : ℝ) :=
  ∀ n, a (n + 1) = q * a n

-- Define an arithmetic sequence
def is_arithmetic_seq (b : ℕ → ℝ) (d : ℝ) :=
  ∀ n, b (n + 1) = b n + d

-- Define the Fibonacci-like sequence
def fib_like (a : ℕ → ℝ) :=
  a 1 = 1 ∧ a 2 = 1 ∧ ∀ n ≥ 2, a (n + 1) = a n + a (n - 1)

theorem ratio_arithmetic_properties :
  (∀ a q, q ≠ 0 → is_geometric_seq a q → is_ratio_arithmetic_seq a 0) ∧
  (∃ b d, is_arithmetic_seq b d ∧ ∃ k, is_ratio_arithmetic_seq b k) ∧
  (∃ a b q d, is_arithmetic_seq a d ∧ is_geometric_seq b q ∧
    ¬∃ k, is_ratio_arithmetic_seq (fun n ↦ a n * b n) k) ∧
  (∀ a, fib_like a → ¬∃ k, is_ratio_arithmetic_seq a k) :=
sorry

end ratio_arithmetic_properties_l1213_121326


namespace purely_imaginary_condition_l1213_121323

theorem purely_imaginary_condition (θ : ℝ) : 
  (∃ (y : ℝ), Complex.mk (Real.sin (2 * θ) - 1) (Real.sqrt 2 * Real.cos θ + 1) = Complex.I * y) ↔ 
  (∃ (k : ℤ), θ = 2 * k * Real.pi + Real.pi / 4) :=
sorry

end purely_imaginary_condition_l1213_121323


namespace perfect_square_trinomial_l1213_121393

theorem perfect_square_trinomial (m : ℝ) : 
  (∃ a : ℝ, ∀ x : ℝ, x^2 + m*x + 25 = (x + a)^2) → m = 10 ∨ m = -10 := by
  sorry

end perfect_square_trinomial_l1213_121393


namespace power_multiplication_l1213_121354

theorem power_multiplication (x : ℝ) : x^5 * x^9 = x^14 := by sorry

end power_multiplication_l1213_121354


namespace total_pencils_l1213_121325

-- Define the number of pencils in each set
def pencils_set_a : ℕ := 10
def pencils_set_b : ℕ := 20
def pencils_set_c : ℕ := 30

-- Define the number of friends who bought each set
def friends_set_a : ℕ := 3
def friends_set_b : ℕ := 2
def friends_set_c : ℕ := 2

-- Define Chloe's purchase
def chloe_sets : ℕ := 1

-- Theorem statement
theorem total_pencils :
  (friends_set_a * pencils_set_a + 
   friends_set_b * pencils_set_b + 
   friends_set_c * pencils_set_c) +
  (chloe_sets * (pencils_set_a + pencils_set_b + pencils_set_c)) = 190 := by
  sorry

end total_pencils_l1213_121325


namespace sequence_a_correct_l1213_121361

def sequence_a (n : ℕ) : ℚ :=
  if n = 1 then 1
  else 1 / (2 * n - 1 : ℚ) - 1 / (2 * n - 3 : ℚ)

def sum_S (n : ℕ) : ℚ :=
  if n = 1 then 1
  else 1 / (2 * n - 1 : ℚ)

theorem sequence_a_correct :
  ∀ n : ℕ, n ≥ 1 →
    (n = 1 ∧ sequence_a n = 1) ∨
    (n ≥ 2 ∧ (sum_S n)^2 = sequence_a n * (sum_S n - 1/2)) :=
by sorry

end sequence_a_correct_l1213_121361


namespace no_integer_solution_for_1980_l1213_121334

theorem no_integer_solution_for_1980 : ∀ m n : ℤ, m^2 + n^2 ≠ 1980 := by
  sorry

end no_integer_solution_for_1980_l1213_121334


namespace simplified_ratio_l1213_121327

def sarah_apples : ℕ := 45
def brother_apples : ℕ := 9
def cousin_apples : ℕ := 27

def gcd_three (a b c : ℕ) : ℕ := Nat.gcd a (Nat.gcd b c)

theorem simplified_ratio :
  let common_divisor := gcd_three sarah_apples brother_apples cousin_apples
  (sarah_apples / common_divisor : ℕ) = 5 ∧
  (brother_apples / common_divisor : ℕ) = 1 ∧
  (cousin_apples / common_divisor : ℕ) = 3 := by
  sorry

end simplified_ratio_l1213_121327


namespace root_sum_reciprocal_l1213_121387

theorem root_sum_reciprocal (p q r A B C : ℝ) : 
  p ≠ q ∧ q ≠ r ∧ p ≠ r →
  (∀ x : ℝ, x^3 - 21*x^2 + 130*x - 210 = 0 ↔ x = p ∨ x = q ∨ x = r) →
  (∀ s : ℝ, s ≠ p ∧ s ≠ q ∧ s ≠ r → 
    1 / (s^3 - 21*s^2 + 130*s - 210) = A / (s - p) + B / (s - q) + C / (s - r)) →
  1 / A + 1 / B + 1 / C = 275 := by
sorry

end root_sum_reciprocal_l1213_121387


namespace multiplication_of_negative_half_and_two_l1213_121371

theorem multiplication_of_negative_half_and_two :
  (-1/2 : ℚ) * 2 = -1 := by sorry

end multiplication_of_negative_half_and_two_l1213_121371


namespace imaginary_unit_sum_l1213_121376

theorem imaginary_unit_sum : ∃ i : ℂ, i * i = -1 ∧ i + i^2 + i^3 = -1 := by sorry

end imaginary_unit_sum_l1213_121376


namespace book_arrangement_count_l1213_121338

/-- Represents the number of math books -/
def num_math_books : ℕ := 3

/-- Represents the number of physics books -/
def num_physics_books : ℕ := 2

/-- Represents the number of chemistry books -/
def num_chem_books : ℕ := 1

/-- Represents the total number of books -/
def total_books : ℕ := num_math_books + num_physics_books + num_chem_books

/-- Calculates the number of arrangements of books on a shelf -/
def num_arrangements : ℕ := 72

/-- Theorem stating that the number of arrangements of books on a shelf,
    where math books are adjacent and physics books are not adjacent,
    is equal to 72 -/
theorem book_arrangement_count :
  num_arrangements = 72 ∧
  num_math_books = 3 ∧
  num_physics_books = 2 ∧
  num_chem_books = 1 ∧
  total_books = num_math_books + num_physics_books + num_chem_books :=
by sorry

end book_arrangement_count_l1213_121338


namespace system_solution_l1213_121324

theorem system_solution (a : ℕ+) 
  (h_system : ∃ (x y : ℝ), a * x + y = -4 ∧ 2 * x + y = -2 ∧ x < 0 ∧ y > 0) :
  a = 3 := by
sorry

end system_solution_l1213_121324


namespace line_circle_separation_l1213_121314

theorem line_circle_separation (α β : ℝ) : 
  let m : ℝ × ℝ := (2 * Real.cos α, 2 * Real.sin α)
  let n : ℝ × ℝ := (3 * Real.cos β, 3 * Real.sin β)
  let angle_between := Real.arccos ((m.1 * n.1 + m.2 * n.2) / (Real.sqrt (m.1^2 + m.2^2) * Real.sqrt (n.1^2 + n.2^2)))
  let line_eq (x y : ℝ) := x * Real.cos α - y * Real.sin α + 1/2
  let circle_center : ℝ × ℝ := (Real.cos β, -Real.sin β)
  let circle_radius : ℝ := Real.sqrt 2 / 2
  let distance_to_line := |line_eq circle_center.1 circle_center.2|
  angle_between = π/3 → distance_to_line > circle_radius :=
by sorry

end line_circle_separation_l1213_121314

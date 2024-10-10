import Mathlib

namespace both_readers_count_l95_9550

def total_workers : ℕ := 72

def saramago_readers : ℕ := total_workers / 4
def kureishi_readers : ℕ := total_workers * 5 / 8

def both_readers : ℕ := 8

theorem both_readers_count :
  saramago_readers + kureishi_readers - both_readers + 
  (saramago_readers - both_readers - 1) = total_workers :=
by sorry

end both_readers_count_l95_9550


namespace employee_count_l95_9500

theorem employee_count :
  ∀ (E : ℕ) (M : ℝ),
    M = 0.99 * (E : ℝ) →
    M - 299.9999999999997 = 0.98 * (E : ℝ) →
    E = 30000 :=
by
  sorry

end employee_count_l95_9500


namespace bicycle_average_speed_l95_9515

theorem bicycle_average_speed (total_distance : ℝ) (first_distance : ℝ) (second_distance : ℝ)
  (first_speed : ℝ) (second_speed : ℝ) (h1 : total_distance = 250)
  (h2 : first_distance = 100) (h3 : second_distance = 150)
  (h4 : first_speed = 20) (h5 : second_speed = 15)
  (h6 : total_distance = first_distance + second_distance) :
  (total_distance / (first_distance / first_speed + second_distance / second_speed)) =
  (250 : ℝ) / ((100 : ℝ) / 20 + (150 : ℝ) / 15) := by
  sorry

end bicycle_average_speed_l95_9515


namespace arithmetic_sequence_properties_l95_9565

/-- An arithmetic sequence with specific terms -/
structure ArithmeticSequence where
  a : ℕ → ℤ
  is_arithmetic : ∀ n, a (n + 1) - a n = a (n + 2) - a (n + 1)
  a7_eq : a 7 = -2
  a20_eq : a 20 = -28

/-- The general term of the arithmetic sequence -/
def generalTerm (seq : ArithmeticSequence) (n : ℕ) : ℤ :=
  14 - 2 * n

/-- The sum of the first n terms of the arithmetic sequence -/
def sumFirstN (seq : ArithmeticSequence) (n : ℕ) : ℤ :=
  n * (seq.a 1 + seq.a n) / 2

theorem arithmetic_sequence_properties (seq : ArithmeticSequence) :
  (∀ n, seq.a n = generalTerm seq n) ∧
  (∃ n, sumFirstN seq n = 42 ∧ ∀ m, sumFirstN seq m ≤ 42) :=
sorry

end arithmetic_sequence_properties_l95_9565


namespace li_ming_on_time_probability_l95_9510

structure TransportationProbabilities where
  bike_prob : ℝ
  bus_prob : ℝ
  bike_on_time_prob : ℝ
  bus_on_time_prob : ℝ

def probability_on_time (p : TransportationProbabilities) : ℝ :=
  p.bike_prob * p.bike_on_time_prob + p.bus_prob * p.bus_on_time_prob

theorem li_ming_on_time_probability :
  ∀ (p : TransportationProbabilities),
    p.bike_prob = 0.7 →
    p.bus_prob = 0.3 →
    p.bike_on_time_prob = 0.9 →
    p.bus_on_time_prob = 0.8 →
    probability_on_time p = 0.87 := by
  sorry

end li_ming_on_time_probability_l95_9510


namespace no_divisibility_pairs_l95_9513

theorem no_divisibility_pairs : ¬∃ (m n : ℕ+), (m.val * n.val ∣ 3^m.val + 1) ∧ (m.val * n.val ∣ 3^n.val + 1) := by
  sorry

end no_divisibility_pairs_l95_9513


namespace sequence_non_positive_l95_9504

theorem sequence_non_positive
  (n : ℕ)
  (a : ℕ → ℝ)
  (h0 : a 0 = 0)
  (hn : a n = 0)
  (h_ineq : ∀ k : ℕ, 1 ≤ k ∧ k < n → a (k - 1) + a (k + 1) - 2 * a k ≥ 0) :
  ∀ k : ℕ, k ≤ n → a k ≤ 0 :=
by sorry

end sequence_non_positive_l95_9504


namespace prob_one_success_value_min_institutes_l95_9508

-- Define the probabilities of success for each institute
def prob_A : ℚ := 1/2
def prob_B : ℚ := 1/3
def prob_C : ℚ := 1/4

-- Define the probability of exactly one institute succeeding
def prob_one_success : ℚ := 
  prob_A * (1 - prob_B) * (1 - prob_C) + 
  (1 - prob_A) * prob_B * (1 - prob_C) + 
  (1 - prob_A) * (1 - prob_B) * prob_C

-- Define the function to calculate the probability of at least one success
-- given n institutes with probability p
def prob_at_least_one (n : ℕ) (p : ℚ) : ℚ := 1 - (1 - p)^n

-- Theorem 1: The probability of exactly one institute succeeding is 11/24
theorem prob_one_success_value : prob_one_success = 11/24 := by sorry

-- Theorem 2: The minimum number of institutes with success probability 1/3
-- needed to achieve at least 99/100 overall success probability is 12
theorem min_institutes : 
  (∀ n < 12, prob_at_least_one n (1/3) < 99/100) ∧ 
  prob_at_least_one 12 (1/3) ≥ 99/100 := by sorry

end prob_one_success_value_min_institutes_l95_9508


namespace slope_of_l3_l95_9529

/-- Line passing through two points -/
structure Line where
  slope : ℝ
  yIntercept : ℝ

/-- Point in 2D plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- Triangle defined by three points -/
structure Triangle where
  a : Point
  b : Point
  c : Point

/-- Calculate the area of a triangle -/
def triangleArea (t : Triangle) : ℝ := sorry

/-- Check if a point lies on a line -/
def pointOnLine (p : Point) (l : Line) : Prop := sorry

/-- Find the intersection point of two lines -/
def lineIntersection (l1 l2 : Line) : Point := sorry

/-- Calculate the slope between two points -/
def slopeBetweenPoints (p1 p2 : Point) : ℝ := sorry

theorem slope_of_l3 (l1 l2 l3 : Line) (A B C : Point) :
  l1.slope = 4/3 ∧ l1.yIntercept = 2/3 ∧
  pointOnLine A l1 ∧ A.x = -2 ∧ A.y = -3 ∧
  l2.slope = 0 ∧ l2.yIntercept = 2 ∧
  B = lineIntersection l1 l2 ∧
  pointOnLine A l3 ∧ pointOnLine C l3 ∧
  pointOnLine C l2 ∧
  l3.slope > 0 ∧
  triangleArea ⟨A, B, C⟩ = 5 →
  l3.slope = 5/6 := by sorry

end slope_of_l3_l95_9529


namespace halfway_between_fractions_l95_9530

theorem halfway_between_fractions :
  (1 / 12 + 1 / 20) / 2 = 1 / 15 := by sorry

end halfway_between_fractions_l95_9530


namespace calculate_b_amount_l95_9533

/-- Given a total amount and the ratio between two parts, calculate the second part -/
theorem calculate_b_amount (total : ℚ) (a b : ℚ) (h1 : a + b = total) (h2 : 2/3 * a = 1/2 * b) : 
  b = 691.43 := by
  sorry

end calculate_b_amount_l95_9533


namespace sum_of_four_digit_primes_and_multiples_of_three_l95_9545

def is_four_digit (n : ℕ) : Prop := 1000 ≤ n ∧ n ≤ 9999

def count_four_digit_primes : ℕ := sorry

def count_four_digit_multiples_of_three : ℕ := sorry

theorem sum_of_four_digit_primes_and_multiples_of_three :
  count_four_digit_primes + count_four_digit_multiples_of_three = 4061 := by sorry

end sum_of_four_digit_primes_and_multiples_of_three_l95_9545


namespace fraction_evaluation_l95_9506

theorem fraction_evaluation : 
  (((1 : ℚ) / 2 + (1 : ℚ) / 5) / ((3 : ℚ) / 7 - (1 : ℚ) / 14)) / ((3 : ℚ) / 4) = (196 : ℚ) / 75 := by
  sorry

end fraction_evaluation_l95_9506


namespace pentomino_tiling_l95_9539

-- Define the pentomino types
inductive Pentomino
| UShaped
| CrossShaped

-- Define a function to check if a rectangle can be tiled
def canTile (width height : ℕ) : Prop :=
  ∃ (arrangement : ℕ → ℕ → Pentomino), 
    (∀ x y, x < width ∧ y < height → ∃ (px py : ℕ) (p : Pentomino), 
      arrangement px py = p ∧ 
      (px ≤ x ∧ x < px + 5) ∧ 
      (py ≤ y ∧ y < py + 5))

-- State the theorem
theorem pentomino_tiling (n : ℕ) :
  n > 1 ∧ canTile 15 n ↔ n ≠ 2 ∧ n ≠ 4 ∧ n ≠ 7 := by
  sorry

end pentomino_tiling_l95_9539


namespace ellipse_condition_iff_l95_9542

-- Define the condition
def condition (m n : ℝ) : Prop := m > n ∧ n > 0

-- Define what it means for the equation to represent an ellipse with foci on the y-axis
def is_ellipse_with_foci_on_y_axis (m n : ℝ) : Prop :=
  ∃ (a b : ℝ), a > b ∧ b > 0 ∧ m = 1 / (a^2) ∧ n = 1 / (b^2)

-- State the theorem
theorem ellipse_condition_iff (m n : ℝ) :
  condition m n ↔ is_ellipse_with_foci_on_y_axis m n := by
  sorry

end ellipse_condition_iff_l95_9542


namespace quadratic_form_sum_l95_9587

theorem quadratic_form_sum (x : ℝ) : ∃ (a b c : ℝ), 
  (15 * x^2 + 90 * x + 405 = a * (x + b)^2 + c) ∧ (a + b + c = 288) := by
  sorry

end quadratic_form_sum_l95_9587


namespace inequality_range_l95_9570

theorem inequality_range (x y k : ℝ) : 
  x > 0 → y > 0 → x + y = k → 
  (∀ x y, x > 0 → y > 0 → x + y = k → (x + 1/x) * (y + 1/y) ≥ (k/2 + 2/k)^2) ↔ 
  (k > 0 ∧ k ≤ 2 * Real.sqrt (2 + Real.sqrt 5)) :=
by sorry

end inequality_range_l95_9570


namespace prime_fraction_sum_of_reciprocals_l95_9522

theorem prime_fraction_sum_of_reciprocals (p : ℕ) (h_prime : Nat.Prime p) (h_gt_3 : p > 3) :
  ∃ (m : ℕ) (x y : ℕ+), 3 ≤ m ∧ m ≤ p - 2 ∧ (m : ℚ) / (p^2 : ℚ) = (1 : ℚ) / (x : ℚ) + (1 : ℚ) / (y : ℚ) :=
sorry

end prime_fraction_sum_of_reciprocals_l95_9522


namespace log_equality_implies_golden_ratio_l95_9592

theorem log_equality_implies_golden_ratio (a b : ℝ) (ha : a > 0) (hb : b > 0) 
  (h : Real.log a / Real.log 4 = Real.log b / Real.log 6 ∧ 
       Real.log a / Real.log 4 = Real.log (a + b) / Real.log 9) : 
  b / a = (1 + Real.sqrt 5) / 2 := by
sorry

end log_equality_implies_golden_ratio_l95_9592


namespace thursday_rainfall_thursday_rainfall_proof_l95_9538

/-- Calculates the total rainfall on Thursday given the rainfall patterns of the week --/
theorem thursday_rainfall (monday_rain : Real) (tuesday_decrease : Real) 
  (wednesday_increase_percent : Real) (thursday_decrease_percent : Real) 
  (thursday_additional_rain : Real) : Real :=
  let tuesday_rain := monday_rain - tuesday_decrease
  let wednesday_rain := tuesday_rain * (1 + wednesday_increase_percent)
  let thursday_rain_before_system := wednesday_rain * (1 - thursday_decrease_percent)
  let thursday_total_rain := thursday_rain_before_system + thursday_additional_rain
  thursday_total_rain

/-- Proves that the total rainfall on Thursday is 0.54 inches given the specific conditions --/
theorem thursday_rainfall_proof :
  thursday_rainfall 0.9 0.7 0.5 0.2 0.3 = 0.54 := by
  sorry

end thursday_rainfall_thursday_rainfall_proof_l95_9538


namespace cube_root_of_three_times_five_to_seven_l95_9507

theorem cube_root_of_three_times_five_to_seven (x : ℝ) :
  x = (5^7 + 5^7 + 5^7)^(1/3) → x = 3^(1/3) * 5^(7/3) := by
sorry

end cube_root_of_three_times_five_to_seven_l95_9507


namespace polygon_30_sides_diagonals_l95_9552

/-- The number of diagonals in a convex polygon with n sides -/
def num_diagonals (n : ℕ) : ℕ := n * (n - 3) / 2

/-- Theorem: A convex polygon with 30 sides has 405 diagonals -/
theorem polygon_30_sides_diagonals :
  num_diagonals 30 = 405 := by
  sorry

end polygon_30_sides_diagonals_l95_9552


namespace quadratic_inequality_solution_range_l95_9502

theorem quadratic_inequality_solution_range (k : ℝ) : 
  (∀ x : ℝ, x = 1 → k^2 * x^2 - 6*k*x + 8 ≥ 0) → 
  (k ≥ 4 ∨ k ≤ 2) :=
by sorry

end quadratic_inequality_solution_range_l95_9502


namespace six_ronna_grams_scientific_notation_l95_9560

/-- Represents the number of zeros after a number for the 'ronna' prefix --/
def ronna_zeros : ℕ := 27

/-- Converts a number with the 'ronna' prefix to its scientific notation --/
def ronna_to_scientific (n : ℝ) : ℝ := n * (10 ^ ronna_zeros)

/-- Theorem stating that 6 ronna grams is equal to 6 × 10^27 grams --/
theorem six_ronna_grams_scientific_notation :
  ronna_to_scientific 6 = 6 * (10 ^ 27) := by sorry

end six_ronna_grams_scientific_notation_l95_9560


namespace runners_meet_again_l95_9591

def track_length : ℝ := 600

def runner_speeds : List ℝ := [3.6, 4.2, 5.4, 6.0]

def meeting_time : ℝ := 1000

theorem runners_meet_again :
  ∀ (speed : ℝ), speed ∈ runner_speeds →
  ∃ (n : ℕ), speed * meeting_time = n * track_length :=
by sorry

end runners_meet_again_l95_9591


namespace polynomial_remainder_theorem_l95_9523

theorem polynomial_remainder_theorem (c d : ℚ) : 
  let g : ℚ → ℚ := λ x => c * x^3 - 7 * x^2 + d * x - 8
  (g 2 = -8) ∧ (g (-3) = -80) → c = 107/7 ∧ d = -302/7 := by
sorry

end polynomial_remainder_theorem_l95_9523


namespace x_minus_y_equals_pi_over_three_l95_9544

theorem x_minus_y_equals_pi_over_three (x y : Real) 
  (h1 : 0 < y) (h2 : y < x) (h3 : x < π)
  (h4 : Real.tan x * Real.tan y = 2)
  (h5 : Real.sin x * Real.sin y = 1/3) : 
  x - y = π/3 := by
sorry

end x_minus_y_equals_pi_over_three_l95_9544


namespace fish_weight_l95_9546

theorem fish_weight : 
  ∀ w : ℝ, w = 2 + w / 3 → w = 3 := by sorry

end fish_weight_l95_9546


namespace cistern_fill_time_l95_9543

/-- Represents the time to fill a cistern with two taps -/
def time_to_fill_cistern (fill_rate : ℚ) (empty_rate : ℚ) : ℚ :=
  1 / (fill_rate - empty_rate)

/-- Theorem: The time to fill the cistern is 12 hours -/
theorem cistern_fill_time :
  let fill_rate : ℚ := 1/6
  let empty_rate : ℚ := 1/12
  time_to_fill_cistern fill_rate empty_rate = 12 := by
  sorry

end cistern_fill_time_l95_9543


namespace cos_2alpha_plus_2pi_3_l95_9537

theorem cos_2alpha_plus_2pi_3 (α : Real) (h : Real.sin (α - π/6) = 2/3) :
  Real.cos (2*α + 2*π/3) = -1/9 := by
  sorry

end cos_2alpha_plus_2pi_3_l95_9537


namespace inequality_proof_l95_9569

theorem inequality_proof (a b c : ℝ) 
  (h_pos_a : 0 < a) (h_pos_b : 0 < b) (h_pos_c : 0 < c)
  (h_sum : a + b + c = 1) : 
  1/(1-a) + 1/(1-b) + 1/(1-c) ≥ 2/(1+a) + 2/(1+b) + 2/(1+c) := by
  sorry

end inequality_proof_l95_9569


namespace second_divisor_problem_l95_9520

theorem second_divisor_problem : ∃ (D : ℕ+) (N : ℕ), N % 35 = 25 ∧ N % D = 4 ∧ D = 17 := by
  sorry

end second_divisor_problem_l95_9520


namespace sum_of_specific_numbers_l95_9577

theorem sum_of_specific_numbers : 12534 + 25341 + 53412 + 34125 = 125412 := by
  sorry

end sum_of_specific_numbers_l95_9577


namespace fountain_pen_price_l95_9583

theorem fountain_pen_price (num_fountain_pens : ℕ) (num_mechanical_pencils : ℕ)
  (total_cost : ℚ) (avg_price_mechanical_pencil : ℚ) :
  num_fountain_pens = 450 →
  num_mechanical_pencils = 3750 →
  total_cost = 11250 →
  avg_price_mechanical_pencil = 2.25 →
  (total_cost - (num_mechanical_pencils : ℚ) * avg_price_mechanical_pencil) / (num_fountain_pens : ℚ) = 6.25 := by
sorry

end fountain_pen_price_l95_9583


namespace track_circumference_is_720_l95_9598

/-- Represents the circumference of a circular track given specific meeting conditions of two travelers -/
def track_circumference (first_meeting_distance : ℝ) (second_meeting_remaining : ℝ) : ℝ :=
  let half_circumference := 360
  2 * half_circumference

/-- Theorem stating that under the given conditions, the track circumference is 720 yards -/
theorem track_circumference_is_720 :
  track_circumference 150 90 = 720 :=
by
  -- The proof would go here
  sorry

#eval track_circumference 150 90

end track_circumference_is_720_l95_9598


namespace stating_public_foundation_share_l95_9593

/-- Represents the charity donation problem -/
structure CharityDonation where
  X : ℝ  -- Total amount raised in dollars
  Y : ℝ  -- Percentage donated to public foundation
  Z : ℕ+  -- Number of organizations in public foundation
  W : ℕ+  -- Number of local non-profit groups
  A : ℝ  -- Amount received by each local non-profit group in dollars
  h1 : X > 0  -- Total amount raised is positive
  h2 : 0 < Y ∧ Y < 100  -- Percentage is between 0 and 100
  h3 : W * A = X * (100 - Y) / 100  -- Equation for local non-profit groups

/-- 
Theorem stating that each organization in the public foundation 
receives YX / (100Z) dollars
-/
theorem public_foundation_share (c : CharityDonation) :
  (c.Y * c.X) / (100 * c.Z) = 
  (c.X * c.Y / 100) / c.Z :=
sorry

end stating_public_foundation_share_l95_9593


namespace sum_of_coefficients_without_x_l95_9514

theorem sum_of_coefficients_without_x (x y : ℝ) : 
  (fun x y => (1 - x - 5*y)^5) 0 1 = -1024 := by sorry

end sum_of_coefficients_without_x_l95_9514


namespace max_value_implies_a_range_l95_9597

-- Define the function f
def f (x : ℝ) : ℝ := x^2 - 4*x + 8

-- State the theorem
theorem max_value_implies_a_range (a : ℝ) :
  (∀ x ∈ Set.Icc 1 a, f x ≤ f a) →
  a ∈ Set.Ici 3 :=
sorry

end max_value_implies_a_range_l95_9597


namespace wire_service_reporters_l95_9512

theorem wire_service_reporters (total_reporters : ℝ) 
  (local_politics_percentage : ℝ) (non_local_politics_percentage : ℝ) :
  local_politics_percentage = 18 / 100 →
  non_local_politics_percentage = 40 / 100 →
  total_reporters > 0 →
  (total_reporters - (total_reporters * local_politics_percentage / (1 - non_local_politics_percentage))) / total_reporters = 70 / 100 := by
  sorry

end wire_service_reporters_l95_9512


namespace cos_equation_solutions_l95_9582

theorem cos_equation_solutions :
  ∃! (S : Finset ℝ), 
    (∀ x ∈ S, x ∈ Set.Icc 0 Real.pi ∧ Real.cos (7 * x) = Real.cos (5 * x)) ∧
    S.card = 7 :=
sorry

end cos_equation_solutions_l95_9582


namespace system_solution_l95_9535

theorem system_solution (x y t : ℝ) :
  (x^2 + t = 1 ∧ (x + y) * t = 0 ∧ y^2 + t = 1) ↔
  ((t = 0 ∧ ((x = 1 ∧ y = 1) ∨ (x = 1 ∧ y = -1) ∨ (x = -1 ∧ y = 1) ∨ (x = -1 ∧ y = -1))) ∨
   (0 < t ∧ t < 1 ∧ ((x = Real.sqrt (1 - t) ∧ y = -Real.sqrt (1 - t)) ∨
                     (x = -Real.sqrt (1 - t) ∧ y = Real.sqrt (1 - t))))) :=
by sorry

end system_solution_l95_9535


namespace sports_club_intersection_l95_9525

theorem sports_club_intersection (total : ℕ) (badminton : ℕ) (tennis : ℕ) (neither : ℕ)
  (h1 : total = 30)
  (h2 : badminton = 17)
  (h3 : tennis = 17)
  (h4 : neither = 2) :
  badminton + tennis - (total - neither) = 6 :=
by sorry

end sports_club_intersection_l95_9525


namespace polygon_sides_l95_9573

theorem polygon_sides (n : ℕ) : n > 2 → (n - 2) * 180 = 2 * 360 → n = 6 := by
  sorry

end polygon_sides_l95_9573


namespace correct_num_teams_l95_9541

/-- The number of teams in a league where each team plays every other team exactly once -/
def num_teams : ℕ := 14

/-- The total number of games played in the league -/
def total_games : ℕ := 91

/-- Theorem stating that the number of teams is correct given the conditions -/
theorem correct_num_teams :
  (num_teams * (num_teams - 1)) / 2 = total_games :=
by sorry

end correct_num_teams_l95_9541


namespace profit_percentage_is_18_percent_l95_9574

def cost_price : ℝ := 460
def selling_price : ℝ := 542.8

theorem profit_percentage_is_18_percent :
  (selling_price - cost_price) / cost_price * 100 = 18 := by
sorry

end profit_percentage_is_18_percent_l95_9574


namespace arithmetic_calculation_l95_9564

theorem arithmetic_calculation : 4 * (8 - 3) - 7 = 13 := by
  sorry

end arithmetic_calculation_l95_9564


namespace square_field_area_l95_9509

theorem square_field_area (side : ℝ) (h1 : 4 * side = 36) 
  (h2 : 6 * (side * side) = 6 * (2 * (4 * side) + 9)) : side * side = 81 :=
by
  sorry

end square_field_area_l95_9509


namespace unique_solution_exponential_equation_l95_9518

theorem unique_solution_exponential_equation :
  ∃! x : ℝ, (2 : ℝ) ^ (7 * x + 2) * (4 : ℝ) ^ (2 * x + 5) = (8 : ℝ) ^ (5 * x + 3) :=
sorry

end unique_solution_exponential_equation_l95_9518


namespace necessary_but_not_sufficient_l95_9580

theorem necessary_but_not_sufficient (x y : ℝ) :
  (∀ x y : ℝ, x ≥ 1 ∧ y ≥ 1 → x^2 + y^2 ≥ 2) ∧
  (∃ x y : ℝ, x^2 + y^2 ≥ 2 ∧ ¬(x ≥ 1 ∧ y ≥ 1)) :=
by sorry

end necessary_but_not_sufficient_l95_9580


namespace total_days_on_orbius5_l95_9561

/-- Definition of the Orbius-5 calendar system -/
structure Orbius5Calendar where
  daysPerYear : Nat := 250
  regularSeasonDays : Nat := 49
  leapSeasonDays : Nat := 51
  regularSeasonsPerYear : Nat := 2
  leapSeasonsPerYear : Nat := 3
  cycleYears : Nat := 10

/-- Definition of the astronaut's visits -/
structure AstronautVisits where
  firstVisitRegularSeasons : Nat := 1
  secondVisitRegularSeasons : Nat := 2
  secondVisitLeapSeasons : Nat := 3
  thirdVisitYears : Nat := 3
  fourthVisitCycles : Nat := 1

/-- Function to calculate total days spent on Orbius-5 -/
def totalDaysOnOrbius5 (calendar : Orbius5Calendar) (visits : AstronautVisits) : Nat :=
  sorry

/-- Theorem stating the total days spent on Orbius-5 -/
theorem total_days_on_orbius5 (calendar : Orbius5Calendar) (visits : AstronautVisits) :
  totalDaysOnOrbius5 calendar visits = 3578 := by
  sorry

end total_days_on_orbius5_l95_9561


namespace seven_people_round_table_l95_9534

def factorial (n : ℕ) : ℕ := Nat.factorial n

def roundTableArrangements (n : ℕ) : ℕ := factorial (n - 1)

theorem seven_people_round_table :
  roundTableArrangements 7 = 720 := by
  sorry

end seven_people_round_table_l95_9534


namespace man_birth_year_l95_9547

theorem man_birth_year (x : ℕ) (h1 : x > 0) (h2 : x^2 - 10 - x > 1850) 
  (h3 : x^2 - 10 - x < 1900) : x^2 - 10 - x = 1882 := by
  sorry

end man_birth_year_l95_9547


namespace initial_group_size_l95_9572

/-- The number of men in the initial group -/
def initial_men_count : ℕ := sorry

/-- The average age increase when two women replace two men -/
def avg_age_increase : ℕ := 6

/-- The age of the first replaced man -/
def man1_age : ℕ := 18

/-- The age of the second replaced man -/
def man2_age : ℕ := 22

/-- The average age of the women -/
def women_avg_age : ℕ := 50

theorem initial_group_size : initial_men_count = 10 := by
  sorry

end initial_group_size_l95_9572


namespace point_on_axis_l95_9558

-- Define a point P in 2D space
def P (m : ℝ) : ℝ × ℝ := (m, 2 - m)

-- Define what it means for a point to lie on the coordinate axis
def lies_on_coordinate_axis (p : ℝ × ℝ) : Prop :=
  p.1 = 0 ∨ p.2 = 0

-- Theorem statement
theorem point_on_axis (m : ℝ) : 
  lies_on_coordinate_axis (P m) → m = 0 ∨ m = 2 := by
  sorry

end point_on_axis_l95_9558


namespace existence_of_hundredth_square_l95_9517

/-- Represents a square grid -/
structure Grid :=
  (size : ℕ)

/-- Represents a square that can be cut out from the grid -/
structure Square :=
  (size : ℕ)
  (position : ℕ × ℕ)

/-- The total number of 2×2 squares that can fit in a grid -/
def total_squares (g : Grid) : ℕ :=
  (g.size - 1) * (g.size - 1)

/-- Predicate to check if a square can be cut out from the grid -/
def can_cut_square (g : Grid) (s : Square) : Prop :=
  s.size = 2 ∧ 
  s.position.1 ≤ g.size - 1 ∧ 
  s.position.2 ≤ g.size - 1

theorem existence_of_hundredth_square (g : Grid) (cut_squares : Finset Square) :
  g.size = 29 →
  cut_squares.card = 99 →
  (∀ s ∈ cut_squares, can_cut_square g s) →
  ∃ s : Square, can_cut_square g s ∧ s ∉ cut_squares :=
sorry

end existence_of_hundredth_square_l95_9517


namespace hyperbola_asymptote_l95_9501

/-- Given a hyperbola with equation x²/a - y²/2 = 1 and one asymptote 2x - y = 0, 
    prove that a = 1/2 -/
theorem hyperbola_asymptote (a : ℝ) 
  (h1 : ∀ x y : ℝ, x^2 / a - y^2 / 2 = 1 → (2*x - y = 0 ∨ 2*x + y = 0)) : a = 1/2 := by
  sorry

end hyperbola_asymptote_l95_9501


namespace invitations_per_pack_is_four_l95_9559

/-- A structure representing the invitation problem --/
structure InvitationProblem where
  total_invitations : ℕ
  num_packs : ℕ
  invitations_per_pack : ℕ
  h1 : total_invitations = num_packs * invitations_per_pack

/-- Theorem stating that given the conditions of the problem, the number of invitations per pack is 4 --/
theorem invitations_per_pack_is_four (problem : InvitationProblem)
  (h2 : problem.total_invitations = 12)
  (h3 : problem.num_packs = 3) :
  problem.invitations_per_pack = 4 := by
  sorry

end invitations_per_pack_is_four_l95_9559


namespace square_side_length_l95_9596

theorem square_side_length (d : ℝ) (h : d = 2 * Real.sqrt 2) :
  ∃ s : ℝ, s > 0 ∧ s * s = d * d / 2 ∧ s = 2 := by
  sorry

end square_side_length_l95_9596


namespace refrigerator_temperature_l95_9588

/-- Given an initial temperature, a temperature decrease rate, and elapsed time,
    calculate the final temperature inside a refrigerator. -/
def final_temperature (initial_temp : ℝ) (decrease_rate : ℝ) (elapsed_time : ℝ) : ℝ :=
  initial_temp - decrease_rate * elapsed_time

/-- Theorem stating that under the given conditions, the final temperature is -8°C. -/
theorem refrigerator_temperature : 
  final_temperature 12 5 4 = -8 := by
  sorry

#eval final_temperature 12 5 4

end refrigerator_temperature_l95_9588


namespace sea_world_trip_savings_l95_9521

def trip_cost (parking : ℕ) (entrance : ℕ) (meal : ℕ) (souvenirs : ℕ) (hotel : ℕ) : ℕ :=
  parking + entrance + meal + souvenirs + hotel

def gas_cost (distance : ℕ) (mpg : ℕ) (price_per_gallon : ℕ) : ℕ :=
  (2 * distance / mpg) * price_per_gallon

def additional_savings (total_cost : ℕ) (current_savings : ℕ) : ℕ :=
  total_cost - current_savings

theorem sea_world_trip_savings : 
  let current_savings : ℕ := 28
  let parking : ℕ := 10
  let entrance : ℕ := 55
  let meal : ℕ := 25
  let souvenirs : ℕ := 40
  let hotel : ℕ := 80
  let distance : ℕ := 165
  let mpg : ℕ := 30
  let price_per_gallon : ℕ := 3
  
  let total_trip_cost := trip_cost parking entrance meal souvenirs hotel
  let total_gas_cost := gas_cost distance mpg price_per_gallon
  let total_cost := total_trip_cost + total_gas_cost
  
  additional_savings total_cost current_savings = 215 := by
  sorry

end sea_world_trip_savings_l95_9521


namespace quadratic_inequality_always_nonnegative_l95_9554

theorem quadratic_inequality_always_nonnegative :
  ∀ x : ℝ, 4 * x^2 - 4 * x + 1 ≥ 0 :=
by
  sorry

end quadratic_inequality_always_nonnegative_l95_9554


namespace first_part_distance_is_18_l95_9549

/-- Represents a cyclist's trip with given parameters -/
structure CyclistTrip where
  totalTime : ℝ
  speed1 : ℝ
  speed2 : ℝ
  distance2 : ℝ
  returnSpeed : ℝ

/-- Calculates the distance of the first part of the trip -/
def firstPartDistance (trip : CyclistTrip) : ℝ :=
  sorry

/-- Theorem stating that the first part of the trip is 18 miles long -/
theorem first_part_distance_is_18 (trip : CyclistTrip) 
  (h1 : trip.totalTime = 7.2)
  (h2 : trip.speed1 = 9)
  (h3 : trip.speed2 = 10)
  (h4 : trip.distance2 = 12)
  (h5 : trip.returnSpeed = 7.5) :
  firstPartDistance trip = 18 :=
sorry

end first_part_distance_is_18_l95_9549


namespace only_opening_window_is_translational_l95_9562

-- Define the type for phenomena
inductive Phenomenon
  | wipingCarWindows
  | openingClassroomDoor
  | openingClassroomWindow
  | swingingOnSwing

-- Define the property of being a translational motion
def isTranslationalMotion (p : Phenomenon) : Prop :=
  match p with
  | .wipingCarWindows => False
  | .openingClassroomDoor => False
  | .openingClassroomWindow => True
  | .swingingOnSwing => False

-- Theorem statement
theorem only_opening_window_is_translational :
  ∀ (p : Phenomenon), isTranslationalMotion p ↔ p = Phenomenon.openingClassroomWindow :=
by sorry

end only_opening_window_is_translational_l95_9562


namespace negation_equivalence_l95_9528

theorem negation_equivalence :
  (¬ ∀ x : ℝ, x > 0 → x / (x - 1) > 0) ↔ (∃ x : ℝ, x > 0 ∧ 0 ≤ x ∧ x < 1) := by
  sorry

end negation_equivalence_l95_9528


namespace problem_1_l95_9532

theorem problem_1 : 7 - (-3) + (-4) - |(-8)| = -2 := by sorry

end problem_1_l95_9532


namespace elections_with_past_officers_count_l95_9548

def total_candidates : ℕ := 16
def past_officers : ℕ := 7
def positions : ℕ := 5

def elections_with_past_officers : ℕ := Nat.choose total_candidates positions - Nat.choose (total_candidates - past_officers) positions

theorem elections_with_past_officers_count : elections_with_past_officers = 4242 := by
  sorry

end elections_with_past_officers_count_l95_9548


namespace flowchart_transformation_l95_9524

def transform (a b c : ℕ) : ℕ × ℕ × ℕ :=
  (c, a, b)

theorem flowchart_transformation :
  transform 21 32 75 = (75, 21, 32) := by
  sorry

end flowchart_transformation_l95_9524


namespace geometric_sequence_sum_l95_9594

/-- A geometric sequence with first term a and common ratio q -/
def geometric_sequence (a q : ℝ) : ℕ → ℝ
  | 0 => a
  | n + 1 => geometric_sequence a q n * q

theorem geometric_sequence_sum (a q : ℝ) (h1 : geometric_sequence a q 0 + geometric_sequence a q 1 = 20) 
  (h2 : geometric_sequence a q 2 + geometric_sequence a q 3 = 60) : 
  geometric_sequence a q 4 + geometric_sequence a q 5 = 180 := by
  sorry

end geometric_sequence_sum_l95_9594


namespace positive_expressions_l95_9571

theorem positive_expressions (U V W X Y : ℝ) 
  (h1 : U < V) (h2 : V < 0) (h3 : 0 < W) (h4 : W < X) (h5 : X < Y) : 
  (0 < U * V) ∧ 
  (0 < (X / V) * U) ∧ 
  (0 < W / (U * V)) ∧ 
  (0 < (X - Y) / W) := by
  sorry

end positive_expressions_l95_9571


namespace largest_k_for_2_pow_15_l95_9519

/-- The sum of k consecutive odd integers starting from 2m + 1 -/
def sumConsecutiveOdds (m k : ℕ) : ℕ := k * (2 * m + k)

/-- Proposition: The largest value of k for which 2^15 is expressible as the sum of k consecutive odd integers is 128 -/
theorem largest_k_for_2_pow_15 : 
  (∃ (m : ℕ), sumConsecutiveOdds m 128 = 2^15) ∧ 
  (∀ (k : ℕ), k > 128 → ¬∃ (m : ℕ), sumConsecutiveOdds m k = 2^15) := by
  sorry

end largest_k_for_2_pow_15_l95_9519


namespace sin_400_lt_cos_40_l95_9527

theorem sin_400_lt_cos_40 : 
  Real.sin (400 * Real.pi / 180) < Real.cos (40 * Real.pi / 180) := by
  sorry

end sin_400_lt_cos_40_l95_9527


namespace choir_size_proof_l95_9585

theorem choir_size_proof : Nat.lcm (Nat.lcm 9 10) 11 = 990 := by
  sorry

end choir_size_proof_l95_9585


namespace x_over_y_is_negative_one_l95_9557

theorem x_over_y_is_negative_one (x y : ℝ) 
  (h1 : 3 < (x - y) / (x + y)) 
  (h2 : (x - y) / (x + y) < 8) 
  (h3 : ∃ (n : ℤ), x / y = n) : 
  x / y = -1 := by sorry

end x_over_y_is_negative_one_l95_9557


namespace train_platform_crossing_time_l95_9595

/-- Represents the problem of a train crossing a platform --/
structure TrainProblem where
  train_speed_kmph : ℝ
  train_speed_ms : ℝ
  platform_length : ℝ
  time_to_cross_man : ℝ

/-- The theorem stating the time taken for the train to cross the platform --/
theorem train_platform_crossing_time (p : TrainProblem)
  (h1 : p.train_speed_kmph = 72)
  (h2 : p.train_speed_ms = p.train_speed_kmph / 3.6)
  (h3 : p.platform_length = 300)
  (h4 : p.time_to_cross_man = 15)
  : p.train_speed_ms * p.time_to_cross_man + p.platform_length = p.train_speed_ms * 30 := by
  sorry

end train_platform_crossing_time_l95_9595


namespace xy_equality_l95_9584

theorem xy_equality (x y : ℝ) : 4 * x * y - 3 * x * y = x * y := by
  sorry

end xy_equality_l95_9584


namespace dolls_count_l95_9575

/-- The total number of toys given -/
def total_toys : ℕ := 403

/-- The number of toy cars given to boys -/
def cars_to_boys : ℕ := 134

/-- The number of dolls given to girls -/
def dolls_to_girls : ℕ := total_toys - cars_to_boys

theorem dolls_count : dolls_to_girls = 269 := by
  sorry

end dolls_count_l95_9575


namespace correct_distribution_l95_9511

/-- The number of ways to distribute men and women into groups --/
def distribute_people (num_men num_women : ℕ) : ℕ :=
  let group1 := Nat.choose num_men 2 * Nat.choose num_women 1
  let group2 := Nat.choose (num_men - 2) 1 * Nat.choose (num_women - 1) 2
  let group3 := Nat.choose 1 1 * Nat.choose 2 2
  (group1 * group2 * group3) / 2

/-- Theorem stating the correct number of distributions --/
theorem correct_distribution : distribute_people 4 5 = 180 := by
  sorry

end correct_distribution_l95_9511


namespace decimal_to_fraction_l95_9581

theorem decimal_to_fraction : 
  (1.45 : ℚ) = 29 / 20 := by sorry

end decimal_to_fraction_l95_9581


namespace solution_set_fraction_inequality_l95_9586

theorem solution_set_fraction_inequality :
  {x : ℝ | (x - 2) / (x + 1) < 0} = {x : ℝ | -1 < x ∧ x < 2} := by sorry

end solution_set_fraction_inequality_l95_9586


namespace cistern_length_is_nine_l95_9503

/-- Represents a rectangular cistern with water -/
structure WaterCistern where
  length : ℝ
  width : ℝ
  depth : ℝ
  totalWetArea : ℝ

/-- Calculates the wet surface area of a cistern -/
def wetSurfaceArea (c : WaterCistern) : ℝ :=
  c.length * c.width + 2 * c.length * c.depth + 2 * c.width * c.depth

/-- Theorem: The length of the cistern with given parameters is 9 meters -/
theorem cistern_length_is_nine :
  ∃ (c : WaterCistern),
    c.width = 4 ∧
    c.depth = 1.25 ∧
    c.totalWetArea = 68.5 ∧
    wetSurfaceArea c = c.totalWetArea ∧
    c.length = 9 := by
  sorry

end cistern_length_is_nine_l95_9503


namespace greatest_consecutive_even_sum_180_l95_9566

/-- The sum of n consecutive even integers starting from 2a is n(2a + n - 1) -/
def sumConsecutiveEvenIntegers (n : ℕ) (a : ℤ) : ℤ := n * (2 * a + n - 1)

/-- 45 is the greatest number of consecutive even integers whose sum is 180 -/
theorem greatest_consecutive_even_sum_180 :
  ∀ n : ℕ, n > 45 → ¬∃ a : ℤ, sumConsecutiveEvenIntegers n a = 180 ∧
  ∃ a : ℤ, sumConsecutiveEvenIntegers 45 a = 180 :=
by sorry

#check greatest_consecutive_even_sum_180

end greatest_consecutive_even_sum_180_l95_9566


namespace share_ratio_l95_9576

/-- Proves that given a total amount of 527 and three shares A = 372, B = 93, and C = 62, 
    the ratio of A's share to B's share is 4:1. -/
theorem share_ratio (total : ℕ) (A B C : ℕ) 
  (h_total : total = 527)
  (h_A : A = 372)
  (h_B : B = 93)
  (h_C : C = 62)
  (h_sum : A + B + C = total) : 
  A / B = 4 := by
  sorry

end share_ratio_l95_9576


namespace min_students_both_l95_9555

-- Define the classroom
structure Classroom where
  total : ℕ
  glasses : ℕ
  blue_shirts : ℕ
  both : ℕ

-- Define the conditions
def valid_classroom (c : Classroom) : Prop :=
  c.glasses = (3 * c.total) / 7 ∧
  c.blue_shirts = (4 * c.total) / 9 ∧
  c.both ≤ min c.glasses c.blue_shirts ∧
  c.total ≥ c.glasses + c.blue_shirts - c.both

-- Theorem statement
theorem min_students_both (c : Classroom) (h : valid_classroom c) :
  ∃ (c_min : Classroom), valid_classroom c_min ∧ c_min.both = 8 ∧
  ∀ (c' : Classroom), valid_classroom c' → c'.both ≥ 8 :=
sorry

end min_students_both_l95_9555


namespace committee_selection_theorem_l95_9579

/-- The number of candidates nominated for the committee -/
def total_candidates : ℕ := 20

/-- The number of candidates who have previously served on the committee -/
def past_members : ℕ := 9

/-- The number of positions available in the new committee -/
def committee_size : ℕ := 6

/-- The number of ways to select the committee with at least one past member -/
def selections_with_past_member : ℕ := 38298

theorem committee_selection_theorem :
  (Nat.choose total_candidates committee_size) - 
  (Nat.choose (total_candidates - past_members) committee_size) = 
  selections_with_past_member :=
sorry

end committee_selection_theorem_l95_9579


namespace count_valid_numbers_l95_9556

/-- The set of digits to choose from -/
def digits : Finset Nat := {0, 1, 2, 3, 4, 5}

/-- Predicate to check if a number is odd -/
def is_odd (n : Nat) : Bool := n % 2 = 1

/-- Predicate to check if a number is even -/
def is_even (n : Nat) : Bool := n % 2 = 0

/-- The set of four-digit numbers formed from the given digits -/
def valid_numbers : Finset (Fin 10000) :=
  sorry

/-- Theorem stating the number of valid four-digit numbers -/
theorem count_valid_numbers : Finset.card valid_numbers = 180 := by
  sorry

end count_valid_numbers_l95_9556


namespace brothers_selection_probability_l95_9551

theorem brothers_selection_probability (p_x p_y p_both : ℚ) :
  p_x = 1/3 → p_y = 2/5 → p_both = p_x * p_y → p_both = 2/15 := by
  sorry

end brothers_selection_probability_l95_9551


namespace freshman_class_size_l95_9599

theorem freshman_class_size :
  ∃! n : ℕ,
    n < 600 ∧
    n % 17 = 16 ∧
    n % 19 = 18 ∧
    n = 322 := by
  sorry

end freshman_class_size_l95_9599


namespace parabola_coefficients_l95_9590

/-- A parabola with vertex (4, -1), vertical axis of symmetry, and passing through (0, -5) -/
structure Parabola where
  a : ℝ
  b : ℝ
  c : ℝ
  vertex_x : a ≠ 0 → -b / (2 * a) = 4
  vertex_y : a * 4^2 + b * 4 + c = -1
  symmetry : a ≠ 0 → -b / (2 * a) = 4
  point : a * 0^2 + b * 0 + c = -5

theorem parabola_coefficients (p : Parabola) : p.a = -1/4 ∧ p.b = 2 ∧ p.c = -5 := by
  sorry

end parabola_coefficients_l95_9590


namespace joans_grilled_cheese_sandwiches_l95_9578

/-- Calculates the number of grilled cheese sandwiches Joan makes given the conditions -/
theorem joans_grilled_cheese_sandwiches 
  (total_cheese : ℕ) 
  (ham_sandwiches : ℕ) 
  (cheese_per_ham : ℕ) 
  (cheese_per_grilled : ℕ) 
  (h1 : total_cheese = 50)
  (h2 : ham_sandwiches = 10)
  (h3 : cheese_per_ham = 2)
  (h4 : cheese_per_grilled = 3) :
  (total_cheese - ham_sandwiches * cheese_per_ham) / cheese_per_grilled = 10 := by
  sorry

end joans_grilled_cheese_sandwiches_l95_9578


namespace trigonometric_identities_l95_9563

theorem trigonometric_identities (α : ℝ) 
  (h1 : 0 < α) (h2 : α < π/2) 
  (h3 : 3 * Real.sin (π - α) = -2 * Real.cos (π + α)) : 
  ((4 * Real.sin α - 2 * Real.cos α) / (5 * Real.cos α + 3 * Real.sin α) = 2/21) ∧ 
  (Real.cos (2*α) + Real.sin (α + π/2) = (5 + 3 * Real.sqrt 13) / 13) := by
sorry

end trigonometric_identities_l95_9563


namespace optimal_rectangular_enclosure_area_l95_9553

theorem optimal_rectangular_enclosure_area
  (perimeter : ℝ)
  (min_length : ℝ)
  (min_width : ℝ)
  (h_perimeter : perimeter = 400)
  (h_min_length : min_length = 100)
  (h_min_width : min_width = 50) :
  ∃ (length width : ℝ),
    length ≥ min_length ∧
    width ≥ min_width ∧
    2 * (length + width) = perimeter ∧
    ∀ (l w : ℝ),
      l ≥ min_length →
      w ≥ min_width →
      2 * (l + w) = perimeter →
      l * w ≤ length * width ∧
      length * width = 10000 :=
by sorry

end optimal_rectangular_enclosure_area_l95_9553


namespace sum_nth_group_is_cube_l95_9505

/-- Returns the nth odd number -/
def nthOdd (n : ℕ) : ℕ := 2 * n - 1

/-- Returns the sum of the first n odd numbers -/
def sumFirstNOdds (n : ℕ) : ℕ := n^2

/-- Returns the sum of odd numbers in the nth group -/
def sumNthGroup (n : ℕ) : ℕ :=
  sumFirstNOdds (sumFirstNOdds n) - sumFirstNOdds (sumFirstNOdds (n - 1))

theorem sum_nth_group_is_cube (n : ℕ) (h : 1 ≤ n ∧ n ≤ 5) : sumNthGroup n = n^3 := by
  sorry

end sum_nth_group_is_cube_l95_9505


namespace bakers_cakes_l95_9531

/-- Baker's cake problem -/
theorem bakers_cakes (initial_cakes : ℕ) : 
  (initial_cakes - 91 + 154 = initial_cakes + 63) →
  initial_cakes = 182 := by
  sorry

#check bakers_cakes

end bakers_cakes_l95_9531


namespace square_to_rectangle_area_increase_l95_9540

theorem square_to_rectangle_area_increase (s : ℝ) (h : s > 0) :
  let original_area := s^2
  let new_length := 1.3 * s
  let new_width := 1.2 * s
  let new_area := new_length * new_width
  (new_area - original_area) / original_area = 0.56 := by
sorry

end square_to_rectangle_area_increase_l95_9540


namespace symmetry_sum_l95_9568

/-- Two points are symmetric with respect to the origin if the sum of their coordinates is (0, 0) -/
def symmetric_wrt_origin (p q : ℝ × ℝ) : Prop :=
  p.1 + q.1 = 0 ∧ p.2 + q.2 = 0

theorem symmetry_sum (a b : ℝ) :
  symmetric_wrt_origin (-2022, -1) (a, b) → a + b = 2023 := by
  sorry

end symmetry_sum_l95_9568


namespace cylinder_not_triangle_l95_9536

-- Define the possible shapes
inductive Shape
  | Cylinder
  | Cone
  | Prism
  | Pyramid

-- Define a function to check if a shape can appear as a triangle
def canAppearAsTriangle (s : Shape) : Prop :=
  match s with
  | Shape.Cylinder => False
  | _ => True

-- Theorem statement
theorem cylinder_not_triangle :
  ∀ s : Shape, canAppearAsTriangle s ↔ s ≠ Shape.Cylinder :=
by
  sorry


end cylinder_not_triangle_l95_9536


namespace gravelling_cost_l95_9567

/-- The cost of gravelling intersecting roads on a rectangular lawn. -/
theorem gravelling_cost 
  (lawn_length lawn_width road_width gravel_cost_per_sqm : ℝ)
  (h_lawn_length : lawn_length = 70)
  (h_lawn_width : lawn_width = 30)
  (h_road_width : road_width = 5)
  (h_gravel_cost : gravel_cost_per_sqm = 4) :
  (lawn_length * road_width + lawn_width * road_width - road_width * road_width) * gravel_cost_per_sqm = 1900 :=
by sorry

end gravelling_cost_l95_9567


namespace sector_area_l95_9589

/-- The area of a sector of a circle with radius 5 cm and arc length 4 cm is 10 cm². -/
theorem sector_area (r : ℝ) (arc_length : ℝ) (h1 : r = 5) (h2 : arc_length = 4) :
  (arc_length / (2 * π * r)) * (π * r^2) = 10 :=
by sorry

end sector_area_l95_9589


namespace ellipse_k_range_l95_9526

/-- An ellipse equation with parameter k -/
def ellipse_equation (x y k : ℝ) : Prop := x^2 + k*y^2 = 2

/-- Foci of the ellipse are on the y-axis -/
def foci_on_y_axis (k : ℝ) : Prop := sorry

/-- The equation represents an ellipse -/
def is_ellipse (k : ℝ) : Prop := sorry

theorem ellipse_k_range (k : ℝ) : 
  (∀ x y : ℝ, ellipse_equation x y k) → 
  is_ellipse k → 
  foci_on_y_axis k → 
  0 < k ∧ k < 1 :=
sorry

end ellipse_k_range_l95_9526


namespace star_one_neg_three_l95_9516

-- Define the ※ operation
def star (a b : ℝ) : ℝ := 2 * a * b - b^2

-- Theorem statement
theorem star_one_neg_three : star 1 (-3) = -15 := by sorry

end star_one_neg_three_l95_9516

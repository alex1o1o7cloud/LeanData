import Mathlib

namespace NUMINAMATH_CALUDE_guitar_price_l1506_150659

theorem guitar_price (upfront_percentage : ℝ) (upfront_payment : ℝ) (total_price : ℝ) :
  upfront_percentage = 0.20 →
  upfront_payment = 240 →
  upfront_percentage * total_price = upfront_payment →
  total_price = 1200 := by
  sorry

end NUMINAMATH_CALUDE_guitar_price_l1506_150659


namespace NUMINAMATH_CALUDE_shooting_mode_l1506_150610

def binomial_mode (n : ℕ) (p : ℝ) : Set ℕ :=
  {k : ℕ | ∀ i : ℕ, i ≤ n → (n.choose k) * p^k * (1-p)^(n-k) ≥ (n.choose i) * p^i * (1-p)^(n-i)}

theorem shooting_mode :
  binomial_mode 19 0.8 = {15, 16} := by
  sorry

end NUMINAMATH_CALUDE_shooting_mode_l1506_150610


namespace NUMINAMATH_CALUDE_change5_descent_l1506_150644

/-- Proof of Chang'e-5 lunar probe descent calculations -/
theorem change5_descent (initial_distance initial_speed final_speed time : ℝ) 
  (h1 : initial_distance = 1800)
  (h2 : initial_speed = 1800)
  (h3 : final_speed = 0)
  (h4 : time = 12 * 60) :
  let v := (0 - initial_distance) / time
  let a := (final_speed - initial_speed) / time
  v = -5/2 ∧ a = -5/2 := by sorry

end NUMINAMATH_CALUDE_change5_descent_l1506_150644


namespace NUMINAMATH_CALUDE_tom_reading_speed_increase_l1506_150675

/-- The factor by which Tom's reading speed increased -/
def reading_speed_increase_factor (normal_speed : ℕ) (increased_pages : ℕ) (hours : ℕ) : ℚ :=
  (increased_pages : ℚ) / ((normal_speed * hours) : ℚ)

/-- Theorem stating that Tom's reading speed increased by a factor of 3 -/
theorem tom_reading_speed_increase :
  reading_speed_increase_factor 12 72 2 = 3 := by
  sorry

end NUMINAMATH_CALUDE_tom_reading_speed_increase_l1506_150675


namespace NUMINAMATH_CALUDE_choose_three_from_nine_l1506_150671

theorem choose_three_from_nine : Nat.choose 9 3 = 84 := by
  sorry

end NUMINAMATH_CALUDE_choose_three_from_nine_l1506_150671


namespace NUMINAMATH_CALUDE_max_value_expression_l1506_150642

theorem max_value_expression (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) :
  (x^2 * y^2 * z^2 * (x^2 + y^2 + z^2)) / ((x + y)^3 * (y + z)^3) ≤ 1/24 ∧
  ∃ (a b c : ℝ), a > 0 ∧ b > 0 ∧ c > 0 ∧
    (a^2 * b^2 * c^2 * (a^2 + b^2 + c^2)) / ((a + b)^3 * (b + c)^3) = 1/24 :=
by sorry

end NUMINAMATH_CALUDE_max_value_expression_l1506_150642


namespace NUMINAMATH_CALUDE_tortoise_wins_l1506_150681

-- Define the race distance
def race_distance : ℝ := 100

-- Define the animals
inductive Animal
| tortoise
| hare

-- Define the speed function for each animal
def speed (a : Animal) (t : ℝ) : ℝ :=
  match a with
  | Animal.tortoise => sorry -- Increasing speed function
  | Animal.hare => sorry -- Piecewise function for hare's speed

-- Define the position function for each animal
def position (a : Animal) (t : ℝ) : ℝ :=
  sorry -- Integral of speed function

-- Define the finish time for each animal
def finish_time (a : Animal) : ℝ :=
  sorry -- Time when position equals race_distance

-- Theorem stating the tortoise wins
theorem tortoise_wins :
  finish_time Animal.tortoise < finish_time Animal.hare :=
sorry


end NUMINAMATH_CALUDE_tortoise_wins_l1506_150681


namespace NUMINAMATH_CALUDE_fourth_number_in_second_set_l1506_150652

theorem fourth_number_in_second_set (x y : ℝ) : 
  ((28 + x + 42 + 78 + 104) / 5 = 90) →
  ((128 + 255 + 511 + y + x) / 5 = 423) →
  y = 1023 := by
sorry

end NUMINAMATH_CALUDE_fourth_number_in_second_set_l1506_150652


namespace NUMINAMATH_CALUDE_xy_sum_greater_than_two_l1506_150653

theorem xy_sum_greater_than_two (x y : ℝ) (h1 : x^3 > y^2) (h2 : y^3 > x^2) : x + y > 2 := by
  sorry

end NUMINAMATH_CALUDE_xy_sum_greater_than_two_l1506_150653


namespace NUMINAMATH_CALUDE_muffin_combinations_l1506_150693

/-- Given four kinds of muffins, when purchasing eight muffins with at least one of each kind,
    there are 23 different possible combinations. -/
theorem muffin_combinations : ℕ :=
  let num_muffin_types : ℕ := 4
  let total_muffins : ℕ := 8
  let min_of_each_type : ℕ := 1
  23

#check muffin_combinations

end NUMINAMATH_CALUDE_muffin_combinations_l1506_150693


namespace NUMINAMATH_CALUDE_intersection_implies_a_value_l1506_150645

-- Define the sets A and B
def A : Set ℝ := {x | x^2 - 4 ≤ 0}
def B (a : ℝ) : Set ℝ := {x | 2*x + a ≤ 0}

-- State the theorem
theorem intersection_implies_a_value :
  ∃ a : ℝ, (A ∩ B a = {x | -2 ≤ x ∧ x ≤ 1}) → a = -2 := by
  sorry

end NUMINAMATH_CALUDE_intersection_implies_a_value_l1506_150645


namespace NUMINAMATH_CALUDE_sunday_visitors_theorem_l1506_150615

/-- Represents the average number of visitors on Sundays in a library -/
def average_sunday_visitors (
  total_days : ℕ)  -- Total number of days in the month
  (sunday_count : ℕ)  -- Number of Sundays in the month
  (non_sunday_average : ℕ)  -- Average number of visitors on non-Sundays
  (month_average : ℕ)  -- Average number of visitors per day for the entire month
  : ℕ :=
  ((month_average * total_days) - (non_sunday_average * (total_days - sunday_count))) / sunday_count

/-- Theorem stating that the average number of Sunday visitors is 510 given the problem conditions -/
theorem sunday_visitors_theorem :
  average_sunday_visitors 30 5 240 285 = 510 := by
  sorry

#eval average_sunday_visitors 30 5 240 285

end NUMINAMATH_CALUDE_sunday_visitors_theorem_l1506_150615


namespace NUMINAMATH_CALUDE_thirty_six_is_triangular_and_square_l1506_150631

/-- Definition of triangular numbers -/
def is_triangular (n : ℕ) : Prop :=
  ∃ k : ℕ, n = k * (k + 1) / 2

/-- Definition of square numbers -/
def is_square (n : ℕ) : Prop :=
  ∃ k : ℕ, n = k^2

/-- Theorem: 36 is both a triangular number and a square number -/
theorem thirty_six_is_triangular_and_square :
  is_triangular 36 ∧ is_square 36 :=
sorry

end NUMINAMATH_CALUDE_thirty_six_is_triangular_and_square_l1506_150631


namespace NUMINAMATH_CALUDE_hat_number_sum_l1506_150601

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ m : ℕ, m > 1 → m < n → ¬(n % m = 0)

theorem hat_number_sum : 
  ∀ (A B : ℕ),
  (A ≥ 2 ∧ A ≤ 49) →  -- Alice's number is between 2 and 49
  (B > 10 ∧ is_prime B) →  -- Bob's number is prime and greater than 10
  (∀ k : ℕ, k ≥ 2 ∧ k ≤ 49 → k ≠ A → ¬(k > B)) →  -- Alice can't tell who has the larger number
  (∀ k : ℕ, k ≥ 2 ∧ k ≤ 49 → k ≠ A → (k > B ∨ B > k)) →  -- Bob can tell who has the larger number
  (∃ (k : ℕ), 50 * B + A = k * k) →  -- The result is a perfect square
  A + B = 37 :=
by sorry

end NUMINAMATH_CALUDE_hat_number_sum_l1506_150601


namespace NUMINAMATH_CALUDE_final_black_area_l1506_150648

/-- The fraction of area that remains black after one change -/
def remaining_fraction : ℚ := 8 / 9

/-- The number of changes applied to the triangle -/
def num_changes : ℕ := 6

/-- The fraction of the original area that remains black after all changes -/
def final_black_fraction : ℚ := remaining_fraction ^ num_changes

/-- Theorem stating the final black fraction after the specified number of changes -/
theorem final_black_area :
  final_black_fraction = 262144 / 531441 := by
  sorry

end NUMINAMATH_CALUDE_final_black_area_l1506_150648


namespace NUMINAMATH_CALUDE_rearrangement_sum_not_all_nines_rearrangement_sum_power_ten_divisible_l1506_150656

/-- Represents a rearrangement of digits of a natural number -/
def rearrangement (n : ℕ) : ℕ → Prop :=
  sorry

/-- Sum of digits of a natural number -/
def sum_of_digits (n : ℕ) : ℕ :=
  sorry

theorem rearrangement_sum_not_all_nines (n : ℕ) :
  ∀ m : ℕ, rearrangement n m → n + m ≠ 10^1967 - 1 :=
sorry

theorem rearrangement_sum_power_ten_divisible (n : ℕ) :
  (∃ m : ℕ, rearrangement n m ∧ n + m = 10^10) → n % 10 = 0 :=
sorry

end NUMINAMATH_CALUDE_rearrangement_sum_not_all_nines_rearrangement_sum_power_ten_divisible_l1506_150656


namespace NUMINAMATH_CALUDE_arithmetic_sequence_problem_l1506_150616

theorem arithmetic_sequence_problem :
  ∀ a b c : ℤ,
  (∃ d : ℤ, b = a + d ∧ c = b + d) →  -- arithmetic sequence condition
  a + b + c = 6 →                    -- sum condition
  a * b * c = -10 →                  -- product condition
  ((a = 5 ∧ b = 2 ∧ c = -1) ∨ (a = -1 ∧ b = 2 ∧ c = 5)) :=
by sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_problem_l1506_150616


namespace NUMINAMATH_CALUDE_sphere_radius_from_great_circle_area_l1506_150654

theorem sphere_radius_from_great_circle_area (A : ℝ) (R : ℝ) :
  A = 4 * Real.pi → A = Real.pi * R^2 → R = 2 := by
  sorry

end NUMINAMATH_CALUDE_sphere_radius_from_great_circle_area_l1506_150654


namespace NUMINAMATH_CALUDE_rectangular_garden_area_l1506_150618

theorem rectangular_garden_area :
  ∀ (length width area : ℝ),
    length = 175 →
    width = 12 →
    area = length * width →
    area = 2100 := by
  sorry

end NUMINAMATH_CALUDE_rectangular_garden_area_l1506_150618


namespace NUMINAMATH_CALUDE_train_length_l1506_150612

/-- The length of a train given its speed and time to cross a pole -/
theorem train_length (speed : ℝ) (time : ℝ) : 
  speed = 300 → time = 15 → ∃ length : ℝ, 
  (abs (length - 1249.95) < 0.01) ∧ 
  (length = speed * 1000 / 3600 * time) := by
  sorry

end NUMINAMATH_CALUDE_train_length_l1506_150612


namespace NUMINAMATH_CALUDE_smallest_fraction_between_l1506_150696

theorem smallest_fraction_between (r s : ℕ+) : 
  (7 : ℚ)/11 < r/s ∧ r/s < (5 : ℚ)/8 ∧ 
  (∀ r' s' : ℕ+, (7 : ℚ)/11 < r'/s' ∧ r'/s' < (5 : ℚ)/8 → s ≤ s') →
  s - r = 10 := by
sorry

end NUMINAMATH_CALUDE_smallest_fraction_between_l1506_150696


namespace NUMINAMATH_CALUDE_emily_spent_twelve_dollars_l1506_150640

/-- The amount Emily spent on flowers -/
def emily_spent (price_per_flower : ℕ) (num_roses : ℕ) (num_daisies : ℕ) : ℕ :=
  price_per_flower * (num_roses + num_daisies)

/-- Theorem: Emily spent 12 dollars on flowers -/
theorem emily_spent_twelve_dollars :
  emily_spent 3 2 2 = 12 := by
  sorry

end NUMINAMATH_CALUDE_emily_spent_twelve_dollars_l1506_150640


namespace NUMINAMATH_CALUDE_empty_set_subset_of_all_l1506_150687

theorem empty_set_subset_of_all (A : Set α) : ∅ ⊆ A := by
  sorry

end NUMINAMATH_CALUDE_empty_set_subset_of_all_l1506_150687


namespace NUMINAMATH_CALUDE_f_max_min_on_interval_l1506_150666

-- Define the function f(x) = x³ - 2x² + 5
def f (x : ℝ) : ℝ := x^3 - 2*x^2 + 5

-- Define the interval [-2, 2]
def interval : Set ℝ := { x | -2 ≤ x ∧ x ≤ 2 }

-- Theorem stating the maximum and minimum values of f(x) on the interval [-2, 2]
theorem f_max_min_on_interval :
  ∃ (max min : ℝ),
    (∀ x ∈ interval, f x ≤ max) ∧
    (∃ x ∈ interval, f x = max) ∧
    (∀ x ∈ interval, min ≤ f x) ∧
    (∃ x ∈ interval, f x = min) ∧
    max = 5 ∧ min = -11 := by
  sorry

end NUMINAMATH_CALUDE_f_max_min_on_interval_l1506_150666


namespace NUMINAMATH_CALUDE_marble_probability_l1506_150667

/-- Given a box of 100 marbles with specified probabilities for white and green marbles,
    prove that the probability of drawing either a red or blue marble is 11/20. -/
theorem marble_probability (total : ℕ) (p_white p_green : ℚ) 
    (h_total : total = 100)
    (h_white : p_white = 1 / 4)
    (h_green : p_green = 1 / 5) :
    (total - (p_white * total + p_green * total)) / total = 11 / 20 := by
  sorry

end NUMINAMATH_CALUDE_marble_probability_l1506_150667


namespace NUMINAMATH_CALUDE_geometric_roots_difference_l1506_150664

theorem geometric_roots_difference (m n : ℝ) : 
  (∃ a r : ℝ, a = 1/2 ∧ r > 0 ∧ 
    (∀ x : ℝ, (x^2 - m*x + 2)*(x^2 - n*x + 2) = 0 ↔ 
      x = a ∨ x = a*r ∨ x = a*r^2 ∨ x = a*r^3)) →
  |m - n| = 3/2 := by sorry

end NUMINAMATH_CALUDE_geometric_roots_difference_l1506_150664


namespace NUMINAMATH_CALUDE_complement_intersection_theorem_l1506_150679

def U : Finset ℕ := {1, 2, 3, 4, 5, 6}
def M : Finset ℕ := {1, 4}
def N : Finset ℕ := {2, 3}

theorem complement_intersection_theorem :
  (U \ M) ∩ N = {2, 3} := by sorry

end NUMINAMATH_CALUDE_complement_intersection_theorem_l1506_150679


namespace NUMINAMATH_CALUDE_sin_C_in_right_triangle_l1506_150647

theorem sin_C_in_right_triangle (A B C : ℝ) : 
  0 ≤ A ∧ A ≤ π ∧ 
  0 ≤ B ∧ B ≤ π ∧ 
  0 ≤ C ∧ C ≤ π ∧ 
  A + B + C = π ∧ 
  A = π / 2 ∧ 
  Real.cos B = 4 / 5 → 
  Real.sin C = 4 / 5 := by
sorry

end NUMINAMATH_CALUDE_sin_C_in_right_triangle_l1506_150647


namespace NUMINAMATH_CALUDE_nisos_population_estimate_l1506_150684

/-- The initial population of Nisos in the year 2000 -/
def initial_population : ℕ := 400

/-- The number of years between 2000 and 2030 -/
def years_passed : ℕ := 30

/-- The number of years it takes for the population to double -/
def doubling_period : ℕ := 20

/-- The estimated population of Nisos in 2030 -/
def estimated_population_2030 : ℕ := 1131

/-- Theorem stating that the estimated population of Nisos in 2030 is approximately 1131 -/
theorem nisos_population_estimate :
  ∃ (ε : ℝ), ε > 0 ∧ ε < 1 ∧ 
  (initial_population : ℝ) * (2 : ℝ) ^ (years_passed / doubling_period : ℝ) ∈ 
  Set.Icc (estimated_population_2030 - ε) (estimated_population_2030 + ε) :=
sorry

end NUMINAMATH_CALUDE_nisos_population_estimate_l1506_150684


namespace NUMINAMATH_CALUDE_arithmetic_geometric_mean_inequality_l1506_150660

theorem arithmetic_geometric_mean_inequality 
  (a b c : ℝ) 
  (ha : a ≥ 0) 
  (hb : b ≥ 0) 
  (hc : c ≥ 0) : 
  (a + b + c) / 3 ≥ (a * b * c) ^ (1/3) :=
sorry

end NUMINAMATH_CALUDE_arithmetic_geometric_mean_inequality_l1506_150660


namespace NUMINAMATH_CALUDE_parabola_tangent_to_line_l1506_150678

/-- The parabola y = ax^2 + 10 is tangent to the line y = 2x + 3 if and only if a = 1/7 -/
theorem parabola_tangent_to_line (a : ℝ) : 
  (∃! x : ℝ, a * x^2 + 10 = 2 * x + 3) ↔ a = 1/7 := by
  sorry

end NUMINAMATH_CALUDE_parabola_tangent_to_line_l1506_150678


namespace NUMINAMATH_CALUDE_investment_loss_calculation_l1506_150694

/-- Represents the capital and loss of two investors -/
structure InvestmentScenario where
  capital_ratio : ℚ  -- Ratio of smaller capital to larger capital
  larger_loss : ℚ    -- Loss of the investor with larger capital
  total_loss : ℚ     -- Total loss of both investors

/-- Theorem stating the relationship between capital ratio, larger investor's loss, and total loss -/
theorem investment_loss_calculation (scenario : InvestmentScenario) 
  (h1 : scenario.capital_ratio = 1 / 9)
  (h2 : scenario.larger_loss = 1080) :
  scenario.total_loss = 1200 := by
  sorry

end NUMINAMATH_CALUDE_investment_loss_calculation_l1506_150694


namespace NUMINAMATH_CALUDE_watchtower_probability_l1506_150686

/-- Represents a searchlight with a given rotation speed in revolutions per minute -/
structure Searchlight where
  speed : ℝ
  speed_positive : speed > 0

/-- The setup of the watchtower problem -/
structure WatchtowerSetup where
  searchlight1 : Searchlight
  searchlight2 : Searchlight
  searchlight3 : Searchlight
  path_time : ℝ
  sl1_speed : searchlight1.speed = 2
  sl2_speed : searchlight2.speed = 3
  sl3_speed : searchlight3.speed = 4
  path_time_value : path_time = 30

/-- The probability of all searchlights not completing a revolution within the given time is 0 -/
theorem watchtower_probability (setup : WatchtowerSetup) :
  ∃ (s : Searchlight), s ∈ [setup.searchlight1, setup.searchlight2, setup.searchlight3] ∧
  (60 / s.speed ≤ setup.path_time) :=
sorry

end NUMINAMATH_CALUDE_watchtower_probability_l1506_150686


namespace NUMINAMATH_CALUDE_expression_is_integer_l1506_150658

theorem expression_is_integer (n : ℤ) : ∃ k : ℤ, (n / 3 : ℚ) + (n^2 / 2 : ℚ) + (n^3 / 6 : ℚ) = k := by
  sorry

end NUMINAMATH_CALUDE_expression_is_integer_l1506_150658


namespace NUMINAMATH_CALUDE_smallest_k_inequality_half_satisfies_inequality_l1506_150682

theorem smallest_k_inequality (k : ℝ) : 
  (∀ x y : ℝ, x ≥ 0 → y ≥ 0 → Real.sqrt (x * y) + k * Real.sqrt (|x - y|) ≥ (x + y) / 2) ↔ 
  k ≥ (1 / 2 : ℝ) :=
sorry

theorem half_satisfies_inequality : 
  ∀ x y : ℝ, x ≥ 0 → y ≥ 0 → 
    Real.sqrt (x * y) + (1 / 2 : ℝ) * Real.sqrt (|x - y|) ≥ (x + y) / 2 :=
sorry

end NUMINAMATH_CALUDE_smallest_k_inequality_half_satisfies_inequality_l1506_150682


namespace NUMINAMATH_CALUDE_drive_time_proof_l1506_150634

/-- Proves the time driven at 60 mph given the conditions of the problem -/
theorem drive_time_proof (total_distance : ℝ) (initial_speed : ℝ) (final_speed : ℝ) (total_time : ℝ)
  (h1 : total_distance = 120)
  (h2 : initial_speed = 60)
  (h3 : final_speed = 90)
  (h4 : total_time = 1.5) :
  ∃ t : ℝ, t + (total_distance - initial_speed * t) / final_speed = total_time ∧ t = 0.5 := by
  sorry

end NUMINAMATH_CALUDE_drive_time_proof_l1506_150634


namespace NUMINAMATH_CALUDE_min_value_of_sum_squares_l1506_150697

theorem min_value_of_sum_squares (x y z : ℝ) (h : x + y + z = 2) :
  x^2 + 2*y^2 + z^2 ≥ 4/3 ∧ 
  ∃ (a b c : ℝ), a + b + c = 2 ∧ a^2 + 2*b^2 + c^2 = 4/3 :=
by sorry

end NUMINAMATH_CALUDE_min_value_of_sum_squares_l1506_150697


namespace NUMINAMATH_CALUDE_no_simultaneous_perfect_squares_l1506_150676

theorem no_simultaneous_perfect_squares (n : ℕ+) :
  ¬∃ (a b : ℕ+), ((n + 1) * 2^n.val = a^2) ∧ ((n + 3) * 2^(n.val + 2) = b^2) := by
  sorry

end NUMINAMATH_CALUDE_no_simultaneous_perfect_squares_l1506_150676


namespace NUMINAMATH_CALUDE_clock_face_partition_l1506_150623

noncomputable def clockFaceAreas (r : ℝ) : (ℝ × ℝ × ℝ × ℝ) :=
  let t₁ := (Real.pi + 2 * Real.sqrt 3 - 6) / 12 * r^2
  let t₂ := (Real.pi - Real.sqrt 3) / 6 * r^2
  let t₃ := (7 * Real.pi + 2 * Real.sqrt 3 - 6) / 12 * r^2
  (t₁, t₂, t₂, t₃)

theorem clock_face_partition (r : ℝ) (h : r > 0) :
  let (t₁, t₂, t₂', t₃) := clockFaceAreas r
  t₁ + t₂ + t₂' + t₃ = Real.pi * r^2 ∧
  t₂ = t₂' ∧
  t₁ > 0 ∧ t₂ > 0 ∧ t₃ > 0 :=
by sorry

end NUMINAMATH_CALUDE_clock_face_partition_l1506_150623


namespace NUMINAMATH_CALUDE_at_least_one_white_certain_l1506_150655

-- Define the number of balls
def total_balls : ℕ := 6
def black_balls : ℕ := 2
def white_balls : ℕ := 4
def drawn_balls : ℕ := 3

-- Define the event of drawing at least one white ball
def at_least_one_white (drawn : Finset ℕ) : Prop :=
  ∃ b ∈ drawn, b > black_balls

-- Theorem statement
theorem at_least_one_white_certain :
  ∀ (drawn : Finset ℕ), drawn.card = drawn_balls → at_least_one_white drawn :=
sorry

end NUMINAMATH_CALUDE_at_least_one_white_certain_l1506_150655


namespace NUMINAMATH_CALUDE_sign_determination_l1506_150602

theorem sign_determination (a b : ℝ) (h1 : a * b > 0) (h2 : a + b < 0) : a < 0 ∧ b < 0 := by
  sorry

end NUMINAMATH_CALUDE_sign_determination_l1506_150602


namespace NUMINAMATH_CALUDE_club_advantage_l1506_150698

/-- Represents a fitness club with a monthly subscription cost -/
structure FitnessClub where
  name : String
  monthlyCost : ℕ

/-- Represents an attendance pattern -/
inductive AttendancePattern
  | Regular
  | MoodBased

/-- Calculates the yearly cost for a given club and attendance pattern -/
def yearlyCost (club : FitnessClub) (pattern : AttendancePattern) : ℕ :=
  match pattern with
  | AttendancePattern.Regular => club.monthlyCost * 12
  | AttendancePattern.MoodBased => 
      if club.name = "Beta" then club.monthlyCost * 8 else club.monthlyCost * 12

/-- Calculates the number of visits per year for a given attendance pattern -/
def yearlyVisits (pattern : AttendancePattern) : ℕ :=
  match pattern with
  | AttendancePattern.Regular => 96
  | AttendancePattern.MoodBased => 56

/-- Calculates the cost per visit for a given club and attendance pattern -/
def costPerVisit (club : FitnessClub) (pattern : AttendancePattern) : ℚ :=
  (yearlyCost club pattern : ℚ) / (yearlyVisits pattern : ℚ)

theorem club_advantage :
  let alpha : FitnessClub := { name := "Alpha", monthlyCost := 999 }
  let beta : FitnessClub := { name := "Beta", monthlyCost := 1299 }
  (costPerVisit alpha AttendancePattern.Regular < costPerVisit beta AttendancePattern.Regular) ∧
  (costPerVisit beta AttendancePattern.MoodBased < costPerVisit alpha AttendancePattern.MoodBased) := by
  sorry

end NUMINAMATH_CALUDE_club_advantage_l1506_150698


namespace NUMINAMATH_CALUDE_smallest_gcd_for_integer_solution_l1506_150609

theorem smallest_gcd_for_integer_solution : ∃ (n : ℕ), n > 0 ∧
  (∀ (a b c : ℤ), Int.gcd a (Int.gcd b c) = n →
    ∃ (x y z : ℤ), x + 2*y + 3*z = a ∧ 2*x + y - 2*z = b ∧ 3*x + y + 5*z = c) ∧
  (∀ (m : ℕ), 0 < m → m < n →
    ∃ (a b c : ℤ), Int.gcd a (Int.gcd b c) = m ∧
      ¬∃ (x y z : ℤ), x + 2*y + 3*z = a ∧ 2*x + y - 2*z = b ∧ 3*x + y + 5*z = c) ∧
  n = 28 :=
sorry

end NUMINAMATH_CALUDE_smallest_gcd_for_integer_solution_l1506_150609


namespace NUMINAMATH_CALUDE_divisible_by_eight_last_digits_l1506_150670

theorem divisible_by_eight_last_digits : 
  ∃! (S : Finset Nat), 
    (∀ n ∈ S, n < 10) ∧ 
    (∀ m : Nat, m % 8 = 0 → m % 10 ∈ S) ∧
    Finset.card S = 5 := by
  sorry

end NUMINAMATH_CALUDE_divisible_by_eight_last_digits_l1506_150670


namespace NUMINAMATH_CALUDE_trigonometric_expression_value_l1506_150683

theorem trigonometric_expression_value :
  Real.sin (315 * π / 180) * Real.sin (-1260 * π / 180) + 
  Real.cos (390 * π / 180) * Real.sin (-1020 * π / 180) = 3/4 := by
  sorry

end NUMINAMATH_CALUDE_trigonometric_expression_value_l1506_150683


namespace NUMINAMATH_CALUDE_translation_result_l1506_150608

/-- Translates a point in the 2D plane along the y-axis. -/
def translate_y (x y dy : ℝ) : ℝ × ℝ := (x, y + dy)

/-- The original point M. -/
def M : ℝ × ℝ := (-10, 1)

/-- The translation distance in the y-direction. -/
def dy : ℝ := 4

/-- The resulting point M₁ after translation. -/
def M₁ : ℝ × ℝ := translate_y M.1 M.2 dy

theorem translation_result :
  M₁ = (-10, 5) := by sorry

end NUMINAMATH_CALUDE_translation_result_l1506_150608


namespace NUMINAMATH_CALUDE_triangle_area_l1506_150628

/-- Given a triangle ABC where angle A is 30°, angle B is 45°, and side a is 2,
    prove that the area of the triangle is √3 + 1. -/
theorem triangle_area (A B C : ℝ) (a b c : ℝ) : 
  A = 30 * π / 180 →
  B = 45 * π / 180 →
  a = 2 →
  (1/2) * a * b * Real.sin (π - A - B) = Real.sqrt 3 + 1 :=
by sorry

end NUMINAMATH_CALUDE_triangle_area_l1506_150628


namespace NUMINAMATH_CALUDE_polynomial_simplification_l1506_150625

theorem polynomial_simplification (x : ℝ) :
  (x^3 + 4*x^2 - 7*x + 11) + (-4*x^4 - x^3 + x^2 + 7*x + 3) + (3*x^4 - 2*x^3 + 5*x - 1) =
  -x^4 - 2*x^3 + 5*x^2 + 5*x + 13 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_simplification_l1506_150625


namespace NUMINAMATH_CALUDE_max_value_on_ellipse_l1506_150632

-- Define the ellipse equation
def ellipse (x y : ℝ) : Prop :=
  x^2 + 3*x*y + 2*y^2 - 14*x - 21*y + 49 = 0

-- Define the function to be maximized
def f (x y : ℝ) : ℝ := x + y

-- Theorem statement
theorem max_value_on_ellipse :
  ∃ (x₀ y₀ : ℝ), ellipse x₀ y₀ ∧
  (∀ (x y : ℝ), ellipse x y → f x y ≤ f x₀ y₀) ∧
  f x₀ y₀ = 343 / 88 := by
  sorry

end NUMINAMATH_CALUDE_max_value_on_ellipse_l1506_150632


namespace NUMINAMATH_CALUDE_grapes_needed_theorem_l1506_150619

/-- The amount of grapes needed in a year after a 20% increase in production -/
def grapes_needed_after_increase (initial_usage : ℝ) : ℝ :=
  2 * (initial_usage * 1.2)

/-- Theorem stating that given an initial grape usage of 90 kg per 6 months 
    and a 20% increase in production, the total amount of grapes needed in a year is 216 kg -/
theorem grapes_needed_theorem :
  grapes_needed_after_increase 90 = 216 := by
  sorry

#eval grapes_needed_after_increase 90

end NUMINAMATH_CALUDE_grapes_needed_theorem_l1506_150619


namespace NUMINAMATH_CALUDE_circle_center_l1506_150663

/-- The center of a circle given by the equation (x-h)^2 + (y-k)^2 = r^2 is (h,k) -/
theorem circle_center (h k r : ℝ) : 
  (∀ x y : ℝ, (x - h)^2 + (y - k)^2 = r^2 ↔ ((x, y) ∈ {p : ℝ × ℝ | (p.1 - h)^2 + (p.2 - k)^2 = r^2})) → 
  (h, k) = (1, 1) → r^2 = 2 →
  (1, 1) ∈ {p : ℝ × ℝ | (p.1 - 1)^2 + (p.2 - 1)^2 = 2} :=
by sorry

end NUMINAMATH_CALUDE_circle_center_l1506_150663


namespace NUMINAMATH_CALUDE_prob_at_least_one_red_l1506_150635

/-- The probability of selecting at least one red ball when randomly choosing 2 balls out of 5 balls (2 red and 3 white) is 7/10. -/
theorem prob_at_least_one_red (total : ℕ) (red : ℕ) (white : ℕ) (select : ℕ) :
  total = 5 →
  red = 2 →
  white = 3 →
  select = 2 →
  (Nat.choose total select - Nat.choose white select : ℚ) / Nat.choose total select = 7 / 10 :=
by sorry

end NUMINAMATH_CALUDE_prob_at_least_one_red_l1506_150635


namespace NUMINAMATH_CALUDE_min_distances_2019_points_l1506_150695

/-- The minimum number of distinct distances between pairs of points in a set of n points in a plane -/
noncomputable def min_distinct_distances (n : ℕ) : ℝ :=
  Real.sqrt (n - 3/4 : ℝ) - 1/2

/-- Theorem: For 2019 distinct points in a plane, the number of distinct distances between pairs of points is at least 44 -/
theorem min_distances_2019_points :
  ⌈min_distinct_distances 2019⌉ ≥ 44 := by sorry

end NUMINAMATH_CALUDE_min_distances_2019_points_l1506_150695


namespace NUMINAMATH_CALUDE_matrix_multiplication_l1506_150620

def A : Matrix (Fin 2) (Fin 2) ℤ := !![3, 1; 4, -2]
def B : Matrix (Fin 2) (Fin 2) ℤ := !![5, -3; 2, 6]

theorem matrix_multiplication :
  A * B = !![17, -3; 16, -24] := by sorry

end NUMINAMATH_CALUDE_matrix_multiplication_l1506_150620


namespace NUMINAMATH_CALUDE_inequality_preservation_l1506_150622

theorem inequality_preservation (m n : ℝ) (h1 : m < n) (h2 : n < 0) :
  m + 2 < n + 2 := by
  sorry

end NUMINAMATH_CALUDE_inequality_preservation_l1506_150622


namespace NUMINAMATH_CALUDE_power_of_square_l1506_150669

theorem power_of_square (a : ℝ) : (a^2)^2 = a^4 := by
  sorry

end NUMINAMATH_CALUDE_power_of_square_l1506_150669


namespace NUMINAMATH_CALUDE_power_two_gt_two_n_plus_one_power_two_le_two_n_plus_one_for_small_n_smallest_n_for_inequality_l1506_150629

theorem power_two_gt_two_n_plus_one (n : ℕ) : n ≥ 3 → 2^n > 2*n + 1 :=
  sorry

theorem power_two_le_two_n_plus_one_for_small_n :
  (2^1 ≤ 2*1 + 1) ∧ (2^2 ≤ 2*2 + 1) :=
  sorry

theorem smallest_n_for_inequality : ∀ n : ℕ, n ≥ 3 ↔ 2^n > 2*n + 1 :=
  sorry

end NUMINAMATH_CALUDE_power_two_gt_two_n_plus_one_power_two_le_two_n_plus_one_for_small_n_smallest_n_for_inequality_l1506_150629


namespace NUMINAMATH_CALUDE_value_of_x_l1506_150680

theorem value_of_x (x y z d e f : ℝ) 
  (hd : d ≠ 0) (he : e ≠ 0) (hf : f ≠ 0)
  (h1 : x * y / (x + 2 * y) = d)
  (h2 : x * z / (2 * x + z) = e)
  (h3 : y * z / (y + 2 * z) = f) :
  x = 3 * d * e * f / (d * e - 2 * d * f + e * f) := by
  sorry

end NUMINAMATH_CALUDE_value_of_x_l1506_150680


namespace NUMINAMATH_CALUDE_selection_theorem_l1506_150699

/-- The number of ways to select k items from n items -/
def choose (n k : ℕ) : ℕ := sorry

/-- The number of graduates -/
def total_graduates : ℕ := 10

/-- The number of graduates to be selected -/
def selected_count : ℕ := 3

/-- The number of ways to select graduates with given conditions -/
def selection_ways : ℕ :=
  choose (total_graduates - 1) (selected_count - 1) +
  choose (total_graduates - 1) (selected_count - 1) -
  choose (total_graduates - 2) (selected_count - 2)

theorem selection_theorem : selection_ways = 49 := by sorry

end NUMINAMATH_CALUDE_selection_theorem_l1506_150699


namespace NUMINAMATH_CALUDE_probability_sum_14_correct_l1506_150691

/-- Represents a standard deck of 52 cards -/
def StandardDeck : Nat := 52

/-- Represents the number of cards with values 2 through 10 in a standard deck -/
def NumberCards : Nat := 36

/-- Represents the number of pairs of number cards that sum to 14 -/
def PairsSummingTo14 : Nat := 76

/-- The probability of selecting two number cards that sum to 14 from a standard deck -/
def probability_sum_14 : ℚ := 19 / 663

theorem probability_sum_14_correct : 
  (PairsSummingTo14 : ℚ) / (StandardDeck * (StandardDeck - 1)) = probability_sum_14 := by
  sorry

end NUMINAMATH_CALUDE_probability_sum_14_correct_l1506_150691


namespace NUMINAMATH_CALUDE_arithmetic_sequence_sum_modulo_l1506_150603

theorem arithmetic_sequence_sum_modulo (a₁ : ℕ) (aₙ : ℕ) (d : ℕ) (n : ℕ) :
  a₁ = 5 →
  aₙ = 145 →
  d = 5 →
  n = (aₙ - a₁) / d + 1 →
  (n * (a₁ + aₙ) / 2) % 12 = 3 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_sum_modulo_l1506_150603


namespace NUMINAMATH_CALUDE_sum_of_herds_equals_total_l1506_150665

/-- The total number of sheep on the farm -/
def total_sheep : ℕ := 149

/-- The number of herds on the farm -/
def num_herds : ℕ := 5

/-- The number of sheep in each herd -/
def herd_sizes : Fin num_herds → ℕ
  | ⟨0, _⟩ => 23
  | ⟨1, _⟩ => 37
  | ⟨2, _⟩ => 19
  | ⟨3, _⟩ => 41
  | ⟨4, _⟩ => 29
  | ⟨n+5, h⟩ => absurd h (Nat.not_lt_of_ge (Nat.le_add_left 5 n))

/-- The theorem stating that the sum of sheep in all herds equals the total number of sheep -/
theorem sum_of_herds_equals_total :
  (Finset.univ.sum fun i => herd_sizes i) = total_sheep := by
  sorry

end NUMINAMATH_CALUDE_sum_of_herds_equals_total_l1506_150665


namespace NUMINAMATH_CALUDE_lcm_of_ratio_two_three_l1506_150689

/-- Given two numbers a and b in the ratio 2:3, where a = 40 and b = 60, prove that their LCM is 60. -/
theorem lcm_of_ratio_two_three (a b : ℕ) (h1 : a = 40) (h2 : b = 60) (h3 : 3 * a = 2 * b) :
  Nat.lcm a b = 60 := by
  sorry

end NUMINAMATH_CALUDE_lcm_of_ratio_two_three_l1506_150689


namespace NUMINAMATH_CALUDE_find_S_l1506_150650

-- Define the sets A and B
def A : Set ℝ := {x | x^2 - 7*x + 10 ≤ 0}
def B : Set ℝ := {x | ∃ (a b : ℝ), x^2 + a*x + b < 0}

-- Define the union of A and B
def AUnionB : Set ℝ := {x | x - 3 < 4 ∧ 4 ≤ 2*x}

-- State the theorem
theorem find_S (a b : ℝ) : 
  A ∩ B = ∅ → 
  A ∪ B = AUnionB → 
  {x | x = a + b} = {23} := by sorry

end NUMINAMATH_CALUDE_find_S_l1506_150650


namespace NUMINAMATH_CALUDE_partition_product_ratio_l1506_150641

theorem partition_product_ratio (n : ℕ) (h : n > 2) :
  ∃ (A B : Finset ℕ), 
    A ∪ B = Finset.range n ∧ 
    A ∩ B = ∅ ∧ 
    max ((A.prod id) / (B.prod id)) ((B.prod id) / (A.prod id)) ≤ (n - 1) / (n - 2) := by
  sorry

end NUMINAMATH_CALUDE_partition_product_ratio_l1506_150641


namespace NUMINAMATH_CALUDE_motorcyclists_speeds_l1506_150611

/-- The length of the circular track in meters -/
def track_length : ℝ := 1000

/-- The time interval between overtakes in minutes -/
def overtake_interval : ℝ := 2

/-- The initial speed of motorcyclist A in meters per minute -/
def speed_A : ℝ := 1000

/-- The initial speed of motorcyclist B in meters per minute -/
def speed_B : ℝ := 1500

/-- Theorem stating the conditions and the conclusion about the motorcyclists' speeds -/
theorem motorcyclists_speeds :
  (speed_B - speed_A) * overtake_interval = track_length ∧
  (2 * speed_A - speed_B) * overtake_interval = track_length →
  speed_A = 1000 ∧ speed_B = 1500 := by
  sorry

end NUMINAMATH_CALUDE_motorcyclists_speeds_l1506_150611


namespace NUMINAMATH_CALUDE_grooms_age_l1506_150621

theorem grooms_age (bride_age groom_age : ℕ) : 
  bride_age = groom_age + 19 →
  bride_age + groom_age = 185 →
  groom_age = 83 := by
sorry

end NUMINAMATH_CALUDE_grooms_age_l1506_150621


namespace NUMINAMATH_CALUDE_deal_or_no_deal_probability_l1506_150674

theorem deal_or_no_deal_probability (total_boxes : ℕ) (high_value_boxes : ℕ) (eliminated_boxes : ℕ) :
  total_boxes = 30 →
  high_value_boxes = 8 →
  eliminated_boxes = 14 →
  (high_value_boxes : ℚ) / (total_boxes - eliminated_boxes : ℚ) ≥ 1/2 :=
by sorry

end NUMINAMATH_CALUDE_deal_or_no_deal_probability_l1506_150674


namespace NUMINAMATH_CALUDE_unique_solution_quadratic_l1506_150639

theorem unique_solution_quadratic (p : ℝ) : 
  p ≠ 0 ∧ (∃! x, p * x^2 - 8 * x + 2 = 0) ↔ p = 8 := by sorry

end NUMINAMATH_CALUDE_unique_solution_quadratic_l1506_150639


namespace NUMINAMATH_CALUDE_chess_group_players_l1506_150627

/-- The number of players in the chess group. -/
def n : ℕ := 20

/-- The total number of games played. -/
def total_games : ℕ := 190

/-- Theorem stating that the number of players is correct given the conditions. -/
theorem chess_group_players :
  (n * (n - 1) / 2 = total_games) ∧
  (∀ m : ℕ, m ≠ n → m * (m - 1) / 2 ≠ total_games) := by
  sorry

#check chess_group_players

end NUMINAMATH_CALUDE_chess_group_players_l1506_150627


namespace NUMINAMATH_CALUDE_johnny_pays_700_l1506_150646

/-- The price of a single ping pong ball in dollars -/
def price_per_ball : ℚ := 1 / 10

/-- The number of ping pong balls Johnny buys -/
def num_balls : ℕ := 10000

/-- The discount rate for buying in bulk -/
def discount_rate : ℚ := 30 / 100

/-- The amount Johnny pays for the ping pong balls -/
def amount_paid : ℚ := price_per_ball * num_balls * (1 - discount_rate)

theorem johnny_pays_700 : amount_paid = 700 := by
  sorry

end NUMINAMATH_CALUDE_johnny_pays_700_l1506_150646


namespace NUMINAMATH_CALUDE_probability_for_2x3x4_prism_l1506_150688

/-- Represents a rectangular prism with dimensions a, b, and c. -/
structure RectangularPrism where
  a : ℝ
  b : ℝ
  c : ℝ
  a_pos : 0 < a
  b_pos : 0 < b
  c_pos : 0 < c

/-- The probability that a plane determined by three randomly selected distinct vertices
    of a rectangular prism contains points inside the prism. -/
def probability_plane_intersects_interior (prism : RectangularPrism) : ℚ :=
  4/7

/-- Theorem stating that for a 2x3x4 rectangular prism, the probability of a plane
    determined by three randomly selected distinct vertices containing points inside
    the prism is 4/7. -/
theorem probability_for_2x3x4_prism :
  let prism : RectangularPrism := ⟨2, 3, 4, by norm_num, by norm_num, by norm_num⟩
  probability_plane_intersects_interior prism = 4/7 := by
  sorry


end NUMINAMATH_CALUDE_probability_for_2x3x4_prism_l1506_150688


namespace NUMINAMATH_CALUDE_sqrt_six_div_sqrt_two_eq_sqrt_three_l1506_150643

theorem sqrt_six_div_sqrt_two_eq_sqrt_three : 
  Real.sqrt 6 / Real.sqrt 2 = Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_six_div_sqrt_two_eq_sqrt_three_l1506_150643


namespace NUMINAMATH_CALUDE_triangle_problem_l1506_150600

/-- Given a triangle ABC with sides a, b, c opposite to angles A, B, C respectively -/
def Triangle (a b c : ℝ) (A B C : ℝ) := True

theorem triangle_problem 
  (a b c : ℝ) (A B C : ℝ) 
  (h_triangle : Triangle a b c A B C)
  (h_eq : 3 * a * Real.sin C = c * Real.cos A)
  (h_B : B = π / 4)
  (h_area : 1 / 2 * a * c * Real.sin B = 9) :
  Real.sin A = Real.sqrt 10 / 10 ∧ a = 3 := by
  sorry


end NUMINAMATH_CALUDE_triangle_problem_l1506_150600


namespace NUMINAMATH_CALUDE_connie_red_markers_l1506_150607

theorem connie_red_markers (total_markers blue_markers : ℕ) 
  (h1 : total_markers = 3343)
  (h2 : blue_markers = 1028) :
  total_markers - blue_markers = 2315 :=
by
  sorry

end NUMINAMATH_CALUDE_connie_red_markers_l1506_150607


namespace NUMINAMATH_CALUDE_stock_investment_fractions_l1506_150617

theorem stock_investment_fractions (initial_investment : ℝ) 
  (final_value : ℝ) (f : ℝ) : 
  initial_investment = 900 →
  final_value = 1350 →
  0 ≤ f →
  f ≤ 1/2 →
  2 * (2 * f * initial_investment) + (1/2 * (1 - 2*f) * initial_investment) = final_value →
  f = 1/3 := by
  sorry

end NUMINAMATH_CALUDE_stock_investment_fractions_l1506_150617


namespace NUMINAMATH_CALUDE_no_valid_assignment_l1506_150651

/-- Represents a mapping of characters to digits -/
def DigitAssignment := Char → Nat

/-- Checks if a DigitAssignment is valid for the given cryptarithmic problem -/
def is_valid_assignment (assignment : DigitAssignment) : Prop :=
  let s := assignment 'S'
  let t := assignment 'T'
  let i := assignment 'I'
  let k := assignment 'K'
  let m := assignment 'M'
  let a := assignment 'A'
  (s ≠ 0) ∧ 
  (m ≠ 0) ∧
  (s ≠ t) ∧ (s ≠ i) ∧ (s ≠ k) ∧ (s ≠ m) ∧ (s ≠ a) ∧
  (t ≠ i) ∧ (t ≠ k) ∧ (t ≠ m) ∧ (t ≠ a) ∧
  (i ≠ k) ∧ (i ≠ m) ∧ (i ≠ a) ∧
  (k ≠ m) ∧ (k ≠ a) ∧
  (m ≠ a) ∧
  (s < 10) ∧ (t < 10) ∧ (i < 10) ∧ (k < 10) ∧ (m < 10) ∧ (a < 10) ∧
  (10000 * s + 1000 * t + 100 * i + 10 * k + s +
   10000 * s + 1000 * t + 100 * i + 10 * k + s =
   100000 * m + 10000 * a + 1000 * s + 100 * t + 10 * i + k + s)

theorem no_valid_assignment : ¬∃ (assignment : DigitAssignment), is_valid_assignment assignment :=
sorry

end NUMINAMATH_CALUDE_no_valid_assignment_l1506_150651


namespace NUMINAMATH_CALUDE_remaining_payment_l1506_150661

/-- Given a product with a 10% deposit of $140, prove that the remaining amount to be paid is $1260 -/
theorem remaining_payment (deposit : ℝ) (deposit_percentage : ℝ) (full_price : ℝ) : 
  deposit = 140 ∧ 
  deposit_percentage = 0.1 ∧ 
  deposit = deposit_percentage * full_price → 
  full_price - deposit = 1260 :=
by sorry

end NUMINAMATH_CALUDE_remaining_payment_l1506_150661


namespace NUMINAMATH_CALUDE_carol_carrot_count_l1506_150672

/-- The number of carrots Carol picked -/
def carols_carrots : ℝ := 29.0

/-- The number of carrots Carol's mom picked -/
def moms_carrots : ℝ := 16.0

/-- The number of carrots they picked together -/
def joint_carrots : ℝ := 38.0

/-- The total number of carrots picked -/
def total_carrots : ℝ := 83.0

theorem carol_carrot_count : 
  carols_carrots + moms_carrots + joint_carrots = total_carrots :=
by sorry

end NUMINAMATH_CALUDE_carol_carrot_count_l1506_150672


namespace NUMINAMATH_CALUDE_min_value_2a_plus_b_min_value_2a_plus_b_equality_l1506_150604

theorem min_value_2a_plus_b (a b : ℝ) (ha : a > 0) (hb : b > 0) 
  (h : 1 / (a + 1) + 2 / (b - 2) = 1 / 2) : 
  2 * a + b ≥ 16 := by
  sorry

theorem min_value_2a_plus_b_equality (a b : ℝ) (ha : a > 0) (hb : b > 0) 
  (h : 1 / (a + 1) + 2 / (b - 2) = 1 / 2) : 
  (2 * a + b = 16) ↔ (a = 3 ∧ b = 10) := by
  sorry

end NUMINAMATH_CALUDE_min_value_2a_plus_b_min_value_2a_plus_b_equality_l1506_150604


namespace NUMINAMATH_CALUDE_sum_consecutive_odd_numbers_remainder_l1506_150637

theorem sum_consecutive_odd_numbers_remainder (start : ℕ) (h : start = 10999) :
  (List.sum (List.map (λ i => start + 2 * i) (List.range 7))) % 14 = 7 := by
  sorry

end NUMINAMATH_CALUDE_sum_consecutive_odd_numbers_remainder_l1506_150637


namespace NUMINAMATH_CALUDE_area_ratio_quad_to_decagon_l1506_150614

-- Define a regular decagon
structure RegularDecagon where
  vertices : Fin 10 → ℝ × ℝ
  is_regular : sorry

-- Define the area of a polygon
def area (polygon : List (ℝ × ℝ)) : ℝ := sorry

-- Define the quadrilateral ACEG within the decagon
def quadACEG (d : RegularDecagon) : List (ℝ × ℝ) :=
  [d.vertices 0, d.vertices 2, d.vertices 4, d.vertices 6]

-- Define the decagon as a list of points
def decagonPoints (d : RegularDecagon) : List (ℝ × ℝ) :=
  (List.range 10).map d.vertices

theorem area_ratio_quad_to_decagon (d : RegularDecagon) :
  area (quadACEG d) / area (decagonPoints d) = 1 / 4 := by
  sorry

end NUMINAMATH_CALUDE_area_ratio_quad_to_decagon_l1506_150614


namespace NUMINAMATH_CALUDE_largest_a_value_l1506_150662

/-- The equation has at least one integer root -/
def has_integer_root (a : ℤ) : Prop :=
  ∃ x : ℤ, (x^2 - (a+7)*x + 7*a)^(1/3) + 3^(1/3) = 0

/-- 11 is the largest integer value of a for which the equation has at least one integer root -/
theorem largest_a_value : (has_integer_root 11 ∧ ∀ a : ℤ, a > 11 → ¬has_integer_root a) :=
sorry

end NUMINAMATH_CALUDE_largest_a_value_l1506_150662


namespace NUMINAMATH_CALUDE_quadratic_always_positive_inequality_implication_existence_of_divisible_number_l1506_150668

-- Problem 1
theorem quadratic_always_positive : ∀ x : ℝ, x^2 - 8*x + 17 > 0 := by sorry

-- Problem 2
theorem inequality_implication : ∀ x : ℝ, (x+2)^2 - (x-3)^2 ≥ 0 → x ≥ 1/2 := by sorry

-- Problem 3
theorem existence_of_divisible_number : ∃ n : ℕ, 11 ∣ (6*n^2 - 7) := by sorry

end NUMINAMATH_CALUDE_quadratic_always_positive_inequality_implication_existence_of_divisible_number_l1506_150668


namespace NUMINAMATH_CALUDE_abcd_hex_binary_digits_l1506_150685

-- Define the hexadecimal number ABCD₁₆ as its decimal equivalent
def abcd_hex : ℕ := 43981

-- Theorem stating that the binary representation of ABCD₁₆ requires 16 bits
theorem abcd_hex_binary_digits : 
  (Nat.log 2 abcd_hex).succ = 16 := by sorry

end NUMINAMATH_CALUDE_abcd_hex_binary_digits_l1506_150685


namespace NUMINAMATH_CALUDE_inequality_proof_l1506_150636

theorem inequality_proof (a b c : ℝ) 
  (ha : 0 < a) (hb : 0 < b) (hc : 0 < c)
  (hab : a ≥ b) (hbc : b ≥ c)
  (hsum : a + b + c ≤ 1) :
  a^2 + 3*b^2 + 5*c^2 ≤ 1 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l1506_150636


namespace NUMINAMATH_CALUDE_infinitely_many_S_3n_geq_S_3n_plus_1_l1506_150605

-- Define the sum of digits function
def S (m : ℕ) : ℕ := sorry

-- Theorem statement
theorem infinitely_many_S_3n_geq_S_3n_plus_1 :
  ∀ N : ℕ, ∃ n : ℕ, n ≥ N ∧ S (3^n) ≥ S (3^(n+1)) :=
sorry

end NUMINAMATH_CALUDE_infinitely_many_S_3n_geq_S_3n_plus_1_l1506_150605


namespace NUMINAMATH_CALUDE_students_in_neither_art_nor_music_l1506_150673

theorem students_in_neither_art_nor_music
  (total : ℕ)
  (art : ℕ)
  (music : ℕ)
  (both : ℕ)
  (h1 : total = 60)
  (h2 : art = 40)
  (h3 : music = 30)
  (h4 : both = 15) :
  total - (art + music - both) = 5 := by
  sorry

end NUMINAMATH_CALUDE_students_in_neither_art_nor_music_l1506_150673


namespace NUMINAMATH_CALUDE_quadratic_rewrite_product_l1506_150649

/-- Given a quadratic equation 16x^2 - 40x - 24 that can be rewritten as (dx + e)^2 + f,
    where d, e, and f are integers, prove that de = -20 -/
theorem quadratic_rewrite_product (d e f : ℤ) : 
  (∀ x, 16 * x^2 - 40 * x - 24 = (d * x + e)^2 + f) → d * e = -20 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_rewrite_product_l1506_150649


namespace NUMINAMATH_CALUDE_forall_positive_implies_square_plus_greater_than_one_is_false_l1506_150638

theorem forall_positive_implies_square_plus_greater_than_one_is_false :
  ¬(∀ x : ℝ, x > 0 → x^2 + x > 1) :=
sorry

end NUMINAMATH_CALUDE_forall_positive_implies_square_plus_greater_than_one_is_false_l1506_150638


namespace NUMINAMATH_CALUDE_difference_of_squares_divisible_by_eight_l1506_150626

theorem difference_of_squares_divisible_by_eight (a b : ℤ) (h : a > b) :
  ∃ k : ℤ, (2 * a + 1)^2 - (2 * b + 1)^2 = 8 * k := by
  sorry

end NUMINAMATH_CALUDE_difference_of_squares_divisible_by_eight_l1506_150626


namespace NUMINAMATH_CALUDE_three_digit_congruence_count_l1506_150690

theorem three_digit_congruence_count :
  (∃ (S : Finset Nat), 
    (∀ x ∈ S, 100 ≤ x ∧ x ≤ 999) ∧
    (∀ x ∈ S, (4897 * x + 603) % 29 = 1427 % 29) ∧
    S.card = 28) := by
  sorry

end NUMINAMATH_CALUDE_three_digit_congruence_count_l1506_150690


namespace NUMINAMATH_CALUDE_digit_ratio_l1506_150657

theorem digit_ratio (x : ℕ) (a b c : ℕ) : 
  x ≥ 100 ∧ x < 1000 →  -- x is a 3-digit integer
  a > 0 →               -- a > 0
  x = 100 * a + 10 * b + c →  -- x is composed of digits a, b, c
  (999 : ℕ) - x = 241 →  -- difference between largest possible value and x is 241
  (b : ℚ) / c = 5 / 8 :=  -- ratio of b to c is 5:8
by sorry

end NUMINAMATH_CALUDE_digit_ratio_l1506_150657


namespace NUMINAMATH_CALUDE_article_cost_l1506_150613

theorem article_cost (decreased_price : ℝ) (decrease_percentage : ℝ) (actual_cost : ℝ) :
  decreased_price = 200 ∧
  decrease_percentage = 20 ∧
  decreased_price = actual_cost * (1 - decrease_percentage / 100) →
  actual_cost = 250 := by
  sorry

end NUMINAMATH_CALUDE_article_cost_l1506_150613


namespace NUMINAMATH_CALUDE_function_inequality_condition_l1506_150630

theorem function_inequality_condition (k : ℝ) : 
  (∀ (a x₁ x₂ : ℝ), 1 ≤ a ∧ a ≤ 2 ∧ 2 ≤ x₁ ∧ x₁ < x₂ ∧ x₂ ≤ 4 →
    |x₁ + a / x₁ - 4| - |x₂ + a / x₂ - 4| < k * (x₁ - x₂)) ↔
  k ≤ 6 - 4 * Real.sqrt 3 :=
by sorry

end NUMINAMATH_CALUDE_function_inequality_condition_l1506_150630


namespace NUMINAMATH_CALUDE_tom_candy_proof_l1506_150692

def initial_candy : ℕ := 2
def friend_candy : ℕ := 7
def bought_candy : ℕ := 10
def final_candy : ℕ := 19

theorem tom_candy_proof :
  initial_candy + friend_candy + bought_candy = final_candy :=
by sorry

end NUMINAMATH_CALUDE_tom_candy_proof_l1506_150692


namespace NUMINAMATH_CALUDE_online_price_is_6_l1506_150677

/-- The price of an item online -/
def online_price : ℝ := 6

/-- The price of an item in the regular store -/
def regular_price : ℝ := online_price + 2

/-- The total amount spent in the regular store -/
def regular_total : ℝ := 96

/-- The total amount spent online -/
def online_total : ℝ := 90

/-- The number of additional items bought online compared to the regular store -/
def additional_items : ℕ := 3

theorem online_price_is_6 :
  (online_total / online_price) = (regular_total / regular_price) + additional_items ∧
  online_price = 6 := by sorry

end NUMINAMATH_CALUDE_online_price_is_6_l1506_150677


namespace NUMINAMATH_CALUDE_arithmetic_sequence_formula_l1506_150606

/-- An arithmetic sequence with first three terms x-1, x+1, and 2x+3 has the general formula a_n = 2n - 3 -/
theorem arithmetic_sequence_formula (x : ℝ) (a : ℕ → ℝ) :
  (∀ n : ℕ, a (n + 1) - a n = a (n + 2) - a (n + 1)) →  -- arithmetic sequence condition
  a 1 = x - 1 →                                         -- first term
  a 2 = x + 1 →                                         -- second term
  a 3 = 2 * x + 3 →                                     -- third term
  ∀ n : ℕ, a n = 2 * n - 3 :=                           -- general formula
by sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_formula_l1506_150606


namespace NUMINAMATH_CALUDE_max_leftover_stickers_l1506_150624

theorem max_leftover_stickers (y : ℕ+) : 
  ∃ (q r : ℕ), y = 12 * q + r ∧ r < 12 ∧ r ≤ 11 ∧ 
  ∀ (q' r' : ℕ), y = 12 * q' + r' ∧ r' < 12 → r' ≤ r :=
by sorry

end NUMINAMATH_CALUDE_max_leftover_stickers_l1506_150624


namespace NUMINAMATH_CALUDE_largest_package_size_l1506_150633

theorem largest_package_size (ming_pencils catherine_pencils lucas_pencils : ℕ) 
  (h_ming : ming_pencils = 48)
  (h_catherine : catherine_pencils = 36)
  (h_lucas : lucas_pencils = 60) :
  Nat.gcd ming_pencils (Nat.gcd catherine_pencils lucas_pencils) = 12 := by
  sorry

end NUMINAMATH_CALUDE_largest_package_size_l1506_150633

import Mathlib

namespace NUMINAMATH_CALUDE_sufficient_not_necessary_l2910_291030

theorem sufficient_not_necessary (y : ℝ) (h : y > 2) :
  (∀ x, x > 1 → x + y > 3) ∧ 
  (∃ x, x + y > 3 ∧ ¬(x > 1)) := by
  sorry

end NUMINAMATH_CALUDE_sufficient_not_necessary_l2910_291030


namespace NUMINAMATH_CALUDE_correct_matching_probability_l2910_291039

theorem correct_matching_probability (n : ℕ) (hn : n = 4) :
  (1 : ℚ) / (n.factorial : ℚ) = 1 / 24 :=
sorry

end NUMINAMATH_CALUDE_correct_matching_probability_l2910_291039


namespace NUMINAMATH_CALUDE_complex_fraction_equality_l2910_291016

theorem complex_fraction_equality : ∃ (i : ℂ), i * i = -1 ∧ (2 * i) / (1 - i) = -1 + i := by
  sorry

end NUMINAMATH_CALUDE_complex_fraction_equality_l2910_291016


namespace NUMINAMATH_CALUDE_existence_of_counterexample_l2910_291052

theorem existence_of_counterexample : ∃ (a b c : ℤ), a > b ∧ b > c ∧ a + b ≤ c := by
  sorry

end NUMINAMATH_CALUDE_existence_of_counterexample_l2910_291052


namespace NUMINAMATH_CALUDE_unique_divisible_number_l2910_291028

def number (d : Nat) : Nat := 62684400 + d * 10

theorem unique_divisible_number :
  ∃! d : Nat, d < 10 ∧ (number d).mod 8 = 0 ∧ (number d).mod 5 = 0 :=
sorry

end NUMINAMATH_CALUDE_unique_divisible_number_l2910_291028


namespace NUMINAMATH_CALUDE_g_150_zeros_l2910_291070

-- Define g₀
def g₀ (x : ℝ) : ℝ := x + |x - 200| - |x + 200|

-- Define gₙ recursively
def g (n : ℕ) (x : ℝ) : ℝ :=
  match n with
  | 0 => g₀ x
  | n + 1 => |g n x| - 1

-- Theorem statement
theorem g_150_zeros :
  ∃! (s : Finset ℝ), s.card = 4 ∧ ∀ x : ℝ, x ∈ s ↔ g 150 x = 0 :=
sorry

end NUMINAMATH_CALUDE_g_150_zeros_l2910_291070


namespace NUMINAMATH_CALUDE_complex_number_range_l2910_291006

variable (z : ℂ) (a : ℝ)

theorem complex_number_range :
  (∃ (r : ℝ), z + 2*Complex.I = r) →
  (∃ (s : ℝ), z / (2 - Complex.I) = s) →
  (Complex.re ((z + a*Complex.I)^2) > 0) →
  (Complex.im ((z + a*Complex.I)^2) > 0) →
  2 < a ∧ a < 4 := by
sorry

end NUMINAMATH_CALUDE_complex_number_range_l2910_291006


namespace NUMINAMATH_CALUDE_gcd_seven_eight_factorial_l2910_291013

theorem gcd_seven_eight_factorial : Nat.gcd (Nat.factorial 7) (Nat.factorial 8) = Nat.factorial 7 := by
  sorry

end NUMINAMATH_CALUDE_gcd_seven_eight_factorial_l2910_291013


namespace NUMINAMATH_CALUDE_ellipse_equation_with_shared_focus_l2910_291029

/-- Given a parabola and an ellipse with shared focus, prove the equation of the ellipse -/
theorem ellipse_equation_with_shared_focus (a : ℝ) (h_a : a > 0) :
  (∃ (x y : ℝ), y^2 = 8*x) →  -- Parabola exists
  (∃ (x y : ℝ), x^2/a^2 + y^2 = 1) →  -- Ellipse exists
  (2 : ℝ) = a * (1 - 1/a^2).sqrt →  -- Focus of parabola is right focus of ellipse
  (∃ (x y : ℝ), x^2/5 + y^2 = 1) :=  -- Resulting ellipse equation
by sorry

end NUMINAMATH_CALUDE_ellipse_equation_with_shared_focus_l2910_291029


namespace NUMINAMATH_CALUDE_log_meaningful_range_l2910_291010

/-- The range of real number a for which log_(a-1)(5-a) is meaningful -/
def meaningful_log_range : Set ℝ :=
  {a : ℝ | a ∈ Set.Ioo 1 2 ∪ Set.Ioo 2 5}

theorem log_meaningful_range :
  ∀ a : ℝ, (∃ x : ℝ, (a - 1) ^ x = 5 - a) ↔ a ∈ meaningful_log_range := by
  sorry

end NUMINAMATH_CALUDE_log_meaningful_range_l2910_291010


namespace NUMINAMATH_CALUDE_range_of_a_l2910_291051

def p (x a : ℝ) : Prop := |x - a| < 4
def q (x : ℝ) : Prop := (x - 2) * (3 - x) > 0

theorem range_of_a : 
  (∀ x a : ℝ, (¬(p x a) → ¬(q x)) ∧ ∃ x, q x ∧ p x a) → 
  ∃ a : ℝ, -1 ≤ a ∧ a ≤ 6 :=
sorry

end NUMINAMATH_CALUDE_range_of_a_l2910_291051


namespace NUMINAMATH_CALUDE_best_fitting_model_has_largest_r_squared_model2_is_best_fitting_l2910_291096

/-- Represents a regression model with its coefficient of determination (R²) -/
structure RegressionModel where
  r_squared : ℝ
  r_squared_nonneg : 0 ≤ r_squared
  r_squared_le_one : r_squared ≤ 1

/-- Determines if a model is the best-fitting among a list of models -/
def is_best_fitting (models : List RegressionModel) (best : RegressionModel) : Prop :=
  best ∈ models ∧ ∀ m ∈ models, m.r_squared ≤ best.r_squared

theorem best_fitting_model_has_largest_r_squared 
  (models : List RegressionModel) (best : RegressionModel) : 
  is_best_fitting models best → 
  ∀ m ∈ models, m.r_squared ≤ best.r_squared :=
by sorry

/-- The four models from the problem -/
def model1 : RegressionModel := ⟨0.75, by norm_num, by norm_num⟩
def model2 : RegressionModel := ⟨0.90, by norm_num, by norm_num⟩
def model3 : RegressionModel := ⟨0.28, by norm_num, by norm_num⟩
def model4 : RegressionModel := ⟨0.55, by norm_num, by norm_num⟩

def problem_models : List RegressionModel := [model1, model2, model3, model4]

theorem model2_is_best_fitting : 
  is_best_fitting problem_models model2 :=
by sorry

end NUMINAMATH_CALUDE_best_fitting_model_has_largest_r_squared_model2_is_best_fitting_l2910_291096


namespace NUMINAMATH_CALUDE_cubic_identity_l2910_291054

theorem cubic_identity (a b : ℝ) : (a + b) * (a^2 - a*b + b^2) = a^3 + b^3 := by
  sorry

end NUMINAMATH_CALUDE_cubic_identity_l2910_291054


namespace NUMINAMATH_CALUDE_probability_at_least_one_from_A_l2910_291008

/-- Represents the number of classes in each school -/
structure SchoolClasses where
  A : ℕ
  B : ℕ
  C : ℕ

/-- Represents the number of classes sampled from each school -/
structure SampledClasses where
  A : ℕ
  B : ℕ
  C : ℕ

/-- The total number of classes to be sampled -/
def totalSampled : ℕ := 6

/-- The number of classes to be randomly selected for comparison -/
def comparisonClasses : ℕ := 2

/-- Calculate the probability of selecting at least one class from school A 
    when randomly choosing 2 out of 6 sampled classes -/
def probabilityAtLeastOneFromA (classes : SchoolClasses) (sampled : SampledClasses) : ℚ :=
  sorry

/-- Theorem stating the probability is 3/5 given the specific conditions -/
theorem probability_at_least_one_from_A : 
  let classes : SchoolClasses := ⟨12, 6, 18⟩
  let sampled : SampledClasses := ⟨2, 1, 3⟩
  probabilityAtLeastOneFromA classes sampled = 3/5 :=
sorry

end NUMINAMATH_CALUDE_probability_at_least_one_from_A_l2910_291008


namespace NUMINAMATH_CALUDE_right_triangle_hypotenuse_l2910_291042

theorem right_triangle_hypotenuse : ∀ (a b c : ℝ),
  a = 8 → b = 15 → c^2 = a^2 + b^2 → c = 17 := by
  sorry

end NUMINAMATH_CALUDE_right_triangle_hypotenuse_l2910_291042


namespace NUMINAMATH_CALUDE_forest_tree_density_l2910_291072

/-- Calculates the tree density in a rectangular forest given the logging parameters --/
theorem forest_tree_density
  (forest_length : ℕ)
  (forest_width : ℕ)
  (loggers : ℕ)
  (months : ℕ)
  (days_per_month : ℕ)
  (trees_per_logger_per_day : ℕ)
  (h1 : forest_length = 4)
  (h2 : forest_width = 6)
  (h3 : loggers = 8)
  (h4 : months = 10)
  (h5 : days_per_month = 30)
  (h6 : trees_per_logger_per_day = 6) :
  (loggers * months * days_per_month * trees_per_logger_per_day) / (forest_length * forest_width) = 600 := by
  sorry

#check forest_tree_density

end NUMINAMATH_CALUDE_forest_tree_density_l2910_291072


namespace NUMINAMATH_CALUDE_cos_240_degrees_l2910_291081

theorem cos_240_degrees : Real.cos (240 * π / 180) = -1/2 := by
  sorry

end NUMINAMATH_CALUDE_cos_240_degrees_l2910_291081


namespace NUMINAMATH_CALUDE_tetrahedron_edge_angle_relation_l2910_291020

/-- Theorem about the relationship between opposite edges and angles in a tetrahedron -/
theorem tetrahedron_edge_angle_relation 
  (a a₁ b b₁ c c₁ : ℝ) 
  (α β γ : ℝ) 
  (h_positive : a > 0 ∧ a₁ > 0 ∧ b > 0 ∧ b₁ > 0 ∧ c > 0 ∧ c₁ > 0)
  (h_angles : 0 ≤ α ∧ α ≤ Real.pi / 2 ∧ 0 ≤ β ∧ β ≤ Real.pi / 2 ∧ 0 ≤ γ ∧ γ ≤ Real.pi / 2) :
  (a * a₁ * Real.cos α = b * b₁ * Real.cos β + c * c₁ * Real.cos γ) ∨
  (b * b₁ * Real.cos β = a * a₁ * Real.cos α + c * c₁ * Real.cos γ) ∨
  (c * c₁ * Real.cos γ = a * a₁ * Real.cos α + b * b₁ * Real.cos β) := by
  sorry


end NUMINAMATH_CALUDE_tetrahedron_edge_angle_relation_l2910_291020


namespace NUMINAMATH_CALUDE_work_ratio_l2910_291095

/-- Given that A can finish a work in 12 days and A and B together can finish 0.25 part of the work in a day,
    prove that the ratio of time taken by B to finish the work alone to the time taken by A is 1:2 -/
theorem work_ratio (time_A : ℝ) (combined_rate : ℝ) :
  time_A = 12 →
  combined_rate = 0.25 →
  combined_rate = 1 / time_A + 1 / (time_A / 2) :=
by sorry

end NUMINAMATH_CALUDE_work_ratio_l2910_291095


namespace NUMINAMATH_CALUDE_kids_difference_l2910_291085

/-- The number of kids Julia played with on each day of the week. -/
structure WeeklyKids where
  monday : ℕ
  tuesday : ℕ
  wednesday : ℕ
  thursday : ℕ
  friday : ℕ
  saturday : ℕ
  sunday : ℕ

/-- Theorem stating the difference in the number of kids played with on specific days. -/
theorem kids_difference (w : WeeklyKids)
    (h1 : w.monday = 6)
    (h2 : w.tuesday = 17)
    (h3 : w.wednesday = 4)
    (h4 : w.thursday = 12)
    (h5 : w.friday = 10)
    (h6 : w.saturday = 15)
    (h7 : w.sunday = 9) :
    (w.tuesday + w.thursday) - (w.monday + w.wednesday + w.sunday) = 10 := by
  sorry


end NUMINAMATH_CALUDE_kids_difference_l2910_291085


namespace NUMINAMATH_CALUDE_smallest_difference_in_triangle_l2910_291011

theorem smallest_difference_in_triangle (PQ QR PR : ℕ) : 
  PQ + QR + PR = 2021 →
  PQ < QR →
  QR ≤ PR →
  PQ + QR > PR →
  PQ + PR > QR →
  QR + PR > PQ →
  ∃ (PQ' QR' PR' : ℕ), 
    PQ' + QR' + PR' = 2021 ∧
    PQ' < QR' ∧
    QR' ≤ PR' ∧
    PQ' + QR' > PR' ∧
    PQ' + PR' > QR' ∧
    QR' + PR' > PQ' ∧
    QR' - PQ' = 1 ∧
    ∀ (PQ'' QR'' PR'' : ℕ),
      PQ'' + QR'' + PR'' = 2021 →
      PQ'' < QR'' →
      QR'' ≤ PR'' →
      PQ'' + QR'' > PR'' →
      PQ'' + PR'' > QR'' →
      QR'' + PR'' > PQ'' →
      QR'' - PQ'' ≥ 1 :=
by sorry

end NUMINAMATH_CALUDE_smallest_difference_in_triangle_l2910_291011


namespace NUMINAMATH_CALUDE_firing_time_per_minute_l2910_291001

/-- Calculates the time spent firing per minute given the firing interval and duration -/
def timeSpentFiring (secondsPerMinute : ℕ) (firingInterval : ℕ) (fireDuration : ℕ) : ℕ :=
  (secondsPerMinute / firingInterval) * fireDuration

/-- Proves that given the specified conditions, the time spent firing per minute is 20 seconds -/
theorem firing_time_per_minute :
  timeSpentFiring 60 15 5 = 20 := by sorry

end NUMINAMATH_CALUDE_firing_time_per_minute_l2910_291001


namespace NUMINAMATH_CALUDE_cookie_eating_contest_l2910_291021

theorem cookie_eating_contest (first_student second_student : ℚ) : 
  first_student = 5/6 → second_student = 2/3 → first_student - second_student = 1/6 := by
  sorry

end NUMINAMATH_CALUDE_cookie_eating_contest_l2910_291021


namespace NUMINAMATH_CALUDE_exists_positive_integer_solution_l2910_291015

theorem exists_positive_integer_solution :
  ∃ (x y : ℕ), x > 0 ∧ y > 0 ∧ x + 2 * y = 7 :=
by sorry

end NUMINAMATH_CALUDE_exists_positive_integer_solution_l2910_291015


namespace NUMINAMATH_CALUDE_basketball_games_total_l2910_291098

theorem basketball_games_total (games_won games_lost : ℕ) : 
  games_won - games_lost = 28 → games_won = 45 → games_lost = 17 → 
  games_won + games_lost = 62 := by
  sorry

end NUMINAMATH_CALUDE_basketball_games_total_l2910_291098


namespace NUMINAMATH_CALUDE_uncle_lou_peanuts_l2910_291059

/-- Calculates the number of peanuts in each bag given the conditions of Uncle Lou's flight. -/
theorem uncle_lou_peanuts (bags : ℕ) (flight_duration : ℕ) (eating_rate : ℕ) : bags = 4 → flight_duration = 120 → eating_rate = 1 → (flight_duration / bags : ℕ) = 30 := by
  sorry

end NUMINAMATH_CALUDE_uncle_lou_peanuts_l2910_291059


namespace NUMINAMATH_CALUDE_shaded_shape_area_l2910_291025

/-- The area of a shape composed of a central square and four right triangles -/
theorem shaded_shape_area (grid_size : ℕ) (square_side : ℕ) (triangle_side : ℕ) : 
  grid_size = 10 → 
  square_side = 2 → 
  triangle_side = 5 → 
  (square_side * square_side + 4 * (triangle_side * triangle_side / 2 : ℚ)) = 54 := by
  sorry

#check shaded_shape_area

end NUMINAMATH_CALUDE_shaded_shape_area_l2910_291025


namespace NUMINAMATH_CALUDE_olympic_mascot_problem_l2910_291036

/-- Olympic Mascot Problem -/
theorem olympic_mascot_problem (m : ℝ) : 
  -- Conditions
  (3000 / m = 2400 / (m - 30)) →
  -- Definitions
  let bing_price := m
  let shuey_price := m - 30
  let bing_sell := 190
  let shuey_sell := 140
  let total_mascots := 200
  let profit (x : ℝ) := (bing_sell - bing_price) * x + (shuey_sell - shuey_price) * (total_mascots - x)
  -- Theorem statements
  (m = 150 ∧ 
   ∀ x : ℝ, 0 ≤ x ∧ x ≤ total_mascots ∧ (total_mascots - x ≥ (2/3) * x) →
     profit x ≤ profit 120) := by sorry

end NUMINAMATH_CALUDE_olympic_mascot_problem_l2910_291036


namespace NUMINAMATH_CALUDE_line_equation_l2910_291066

/-- Proves that the equation 4x + 3y - 13 = 0 represents the line passing through (1, 3)
    with a slope that is 1/3 of the slope of y = -4x -/
theorem line_equation (x y : ℝ) : 
  (∃ (k : ℝ), k = (-4 : ℝ) / 3 ∧ 
   y - 3 = k * (x - 1) ∧
   (∀ (x' y' : ℝ), y' = -4 * x' → k = (1 : ℝ) / 3 * (-4))) → 
  (4 * x + 3 * y - 13 = 0) := by
sorry

end NUMINAMATH_CALUDE_line_equation_l2910_291066


namespace NUMINAMATH_CALUDE_correct_propositions_l2910_291099

-- Define the type for propositions
inductive Proposition
  | one
  | two
  | three
  | four
  | five
  | six
  | seven

-- Define a function to check if a proposition is correct
def is_correct (p : Proposition) : Prop :=
  match p with
  | .two => True
  | .six => True
  | .seven => True
  | _ => False

-- Define the theorem
theorem correct_propositions :
  ∀ p : Proposition, is_correct p ↔ (p = .two ∨ p = .six ∨ p = .seven) :=
by sorry

end NUMINAMATH_CALUDE_correct_propositions_l2910_291099


namespace NUMINAMATH_CALUDE_smallest_k_sum_squares_div_150_l2910_291023

/-- The sum of squares from 1 to n -/
def sum_of_squares (n : ℕ) : ℕ := n * (n + 1) * (2 * n + 1) / 6

/-- 100 is the smallest positive integer k such that the sum of squares from 1 to k is divisible by 150 -/
theorem smallest_k_sum_squares_div_150 :
  ∀ k : ℕ, k > 0 → k < 100 → ¬(150 ∣ sum_of_squares k) ∧ (150 ∣ sum_of_squares 100) :=
by sorry

end NUMINAMATH_CALUDE_smallest_k_sum_squares_div_150_l2910_291023


namespace NUMINAMATH_CALUDE_chess_tournament_players_l2910_291058

/-- Represents a chess tournament with the given conditions -/
structure ChessTournament where
  n : ℕ  -- Number of players not in the lowest 12
  /-- Total number of players is n + 12 -/
  total_players : ℕ := n + 12
  /-- Each player played exactly one game against every other player -/
  total_games : ℕ := (total_players * (total_players - 1)) / 2
  /-- Points earned by n players not in the lowest 12 -/
  top_points : ℕ := n * (n - 1)
  /-- Points earned by 12 lowest-scoring players among themselves -/
  bottom_points : ℕ := 66
  /-- Total points earned in the tournament -/
  total_points : ℕ := total_games

/-- The theorem stating that the total number of players in the tournament is 34 -/
theorem chess_tournament_players : 
  ∀ t : ChessTournament, t.total_players = 34 := by
  sorry


end NUMINAMATH_CALUDE_chess_tournament_players_l2910_291058


namespace NUMINAMATH_CALUDE_couple_seating_arrangements_l2910_291055

/-- Represents a couple (a boy and a girl) -/
structure Couple :=
  (boy : Nat)
  (girl : Nat)

/-- Represents a seating arrangement on the bench -/
structure Arrangement :=
  (seat1 : Nat)
  (seat2 : Nat)
  (seat3 : Nat)
  (seat4 : Nat)

/-- Checks if a given arrangement is valid (each couple sits together) -/
def isValidArrangement (c1 c2 : Couple) (arr : Arrangement) : Prop :=
  (arr.seat1 = c1.boy ∧ arr.seat2 = c1.girl ∧ arr.seat3 = c2.boy ∧ arr.seat4 = c2.girl) ∨
  (arr.seat1 = c1.girl ∧ arr.seat2 = c1.boy ∧ arr.seat3 = c2.boy ∧ arr.seat4 = c2.girl) ∨
  (arr.seat1 = c1.boy ∧ arr.seat2 = c1.girl ∧ arr.seat3 = c2.girl ∧ arr.seat4 = c2.boy) ∨
  (arr.seat1 = c1.girl ∧ arr.seat2 = c1.boy ∧ arr.seat3 = c2.girl ∧ arr.seat4 = c2.boy) ∨
  (arr.seat1 = c2.boy ∧ arr.seat2 = c2.girl ∧ arr.seat3 = c1.boy ∧ arr.seat4 = c1.girl) ∨
  (arr.seat1 = c2.girl ∧ arr.seat2 = c2.boy ∧ arr.seat3 = c1.boy ∧ arr.seat4 = c1.girl) ∨
  (arr.seat1 = c2.boy ∧ arr.seat2 = c2.girl ∧ arr.seat3 = c1.girl ∧ arr.seat4 = c1.boy) ∨
  (arr.seat1 = c2.girl ∧ arr.seat2 = c2.boy ∧ arr.seat3 = c1.girl ∧ arr.seat4 = c1.boy)

/-- The main theorem: there are exactly 8 valid seating arrangements -/
theorem couple_seating_arrangements (c1 c2 : Couple) :
  ∃! (arrangements : Finset Arrangement), 
    (∀ arr ∈ arrangements, isValidArrangement c1 c2 arr) ∧
    arrangements.card = 8 :=
by
  sorry

end NUMINAMATH_CALUDE_couple_seating_arrangements_l2910_291055


namespace NUMINAMATH_CALUDE_six_digit_multiply_rearrange_l2910_291080

def is_valid_number (n : ℕ) : Prop :=
  100000 ≤ n ∧ n < 1000000 ∧ n / 100000 = 2

def rearranged (n m : ℕ) : Prop :=
  ∃ (a b c d e : ℕ),
    n = 200000 + 10000 * a + 1000 * b + 100 * c + 10 * d + e ∧
    m = 100000 * a + 10000 * b + 1000 * c + 100 * d + 10 * e + 2

def digit_sum (n : ℕ) : ℕ :=
  (n / 100000) + ((n / 10000) % 10) + ((n / 1000) % 10) +
  ((n / 100) % 10) + ((n / 10) % 10) + (n % 10)

theorem six_digit_multiply_rearrange (n : ℕ) :
  is_valid_number n → rearranged n (3 * n) → digit_sum n = 27 :=
by sorry

end NUMINAMATH_CALUDE_six_digit_multiply_rearrange_l2910_291080


namespace NUMINAMATH_CALUDE_min_sum_m_n_l2910_291084

theorem min_sum_m_n : ∃ (m n : ℕ+), 
  108 * (m : ℕ) = (n : ℕ)^3 ∧ 
  (∀ (m' n' : ℕ+), 108 * (m' : ℕ) = (n' : ℕ)^3 → (m : ℕ) + (n : ℕ) ≤ (m' : ℕ) + (n' : ℕ)) ∧
  (m : ℕ) + (n : ℕ) = 8 := by
  sorry

end NUMINAMATH_CALUDE_min_sum_m_n_l2910_291084


namespace NUMINAMATH_CALUDE_loss_equals_sixteen_pencils_l2910_291041

/-- Represents a pencil transaction with a loss -/
structure PencilTransaction where
  quantity : ℕ
  costMultiplier : ℝ
  sellingPrice : ℝ

/-- Calculates the number of pencils whose selling price equals the total loss -/
def lossInPencils (t : PencilTransaction) : ℝ :=
  t.quantity * (t.costMultiplier - 1)

/-- Theorem stating that for the given transaction, the loss equals the selling price of 16 pencils -/
theorem loss_equals_sixteen_pencils (t : PencilTransaction) 
  (h1 : t.quantity = 80)
  (h2 : t.costMultiplier = 1.2) : 
  lossInPencils t = 16 := by
  sorry

#eval lossInPencils { quantity := 80, costMultiplier := 1.2, sellingPrice := 1 }

end NUMINAMATH_CALUDE_loss_equals_sixteen_pencils_l2910_291041


namespace NUMINAMATH_CALUDE_bob_profit_is_1600_l2910_291092

/-- Calculates the total profit from selling puppies given the number of show dogs bought,
    cost per show dog, number of puppies, and selling price per puppy. -/
def calculate_profit (num_dogs : ℕ) (cost_per_dog : ℚ) (num_puppies : ℕ) (price_per_puppy : ℚ) : ℚ :=
  num_puppies * price_per_puppy - num_dogs * cost_per_dog

/-- Proves that Bob's total profit from selling puppies is $1,600.00 -/
theorem bob_profit_is_1600 :
  calculate_profit 2 250 6 350 = 1600 := by
  sorry

end NUMINAMATH_CALUDE_bob_profit_is_1600_l2910_291092


namespace NUMINAMATH_CALUDE_jack_hunting_frequency_l2910_291071

/-- Represents the hunting scenario for Jack --/
structure HuntingScenario where
  seasonLength : ℚ  -- Length of hunting season in quarters of a year
  deersPerTrip : ℕ  -- Number of deers caught per hunting trip
  deerWeight : ℕ    -- Weight of each deer in pounds
  keepRatio : ℚ     -- Ratio of deer weight kept per year
  keptWeight : ℕ    -- Total weight of deer kept in pounds

/-- Calculates the number of hunting trips per month --/
def tripsPerMonth (scenario : HuntingScenario) : ℚ :=
  let totalWeight := scenario.keptWeight / scenario.keepRatio
  let weightPerTrip := scenario.deersPerTrip * scenario.deerWeight
  let tripsPerYear := totalWeight / weightPerTrip
  let monthsInSeason := scenario.seasonLength * 12
  tripsPerYear / monthsInSeason

/-- Theorem stating that Jack goes hunting 6 times per month --/
theorem jack_hunting_frequency :
  let scenario : HuntingScenario := {
    seasonLength := 1/4,
    deersPerTrip := 2,
    deerWeight := 600,
    keepRatio := 1/2,
    keptWeight := 10800
  }
  tripsPerMonth scenario = 6 := by sorry

end NUMINAMATH_CALUDE_jack_hunting_frequency_l2910_291071


namespace NUMINAMATH_CALUDE_babysitting_earnings_l2910_291086

def final_balance (hourly_rate : ℕ) (hours_worked : ℕ) (initial_balance : ℕ) : ℕ :=
  initial_balance + hourly_rate * hours_worked

theorem babysitting_earnings : final_balance 5 7 20 = 55 := by
  sorry

end NUMINAMATH_CALUDE_babysitting_earnings_l2910_291086


namespace NUMINAMATH_CALUDE_santinos_fruits_l2910_291093

/-- The number of papaya trees Santino has -/
def papaya_trees : ℕ := 2

/-- The number of mango trees Santino has -/
def mango_trees : ℕ := 3

/-- The number of papayas produced by each papaya tree -/
def papayas_per_tree : ℕ := 10

/-- The number of mangos produced by each mango tree -/
def mangos_per_tree : ℕ := 20

/-- The total number of fruits Santino has -/
def total_fruits : ℕ := papaya_trees * papayas_per_tree + mango_trees * mangos_per_tree

theorem santinos_fruits : total_fruits = 80 := by
  sorry

end NUMINAMATH_CALUDE_santinos_fruits_l2910_291093


namespace NUMINAMATH_CALUDE_reflection_across_x_axis_l2910_291091

/-- Reflects a point across the x-axis -/
def reflect_x (p : ℝ × ℝ) : ℝ × ℝ := (p.1, -p.2)

/-- The original point A -/
def A : ℝ × ℝ := (-5, -2)

/-- The expected reflection of A across the x-axis -/
def A_reflected : ℝ × ℝ := (-5, 2)

theorem reflection_across_x_axis :
  reflect_x A = A_reflected := by sorry

end NUMINAMATH_CALUDE_reflection_across_x_axis_l2910_291091


namespace NUMINAMATH_CALUDE_race_head_start_l2910_291024

/-- Calculates the head start given in a race between two runners with different speeds -/
def headStart (cristinaSpeed nicky_speed : ℝ) (catchUpTime : ℝ) : ℝ :=
  nicky_speed * catchUpTime

theorem race_head_start :
  let cristinaSpeed : ℝ := 5
  let nickySpeed : ℝ := 3
  let catchUpTime : ℝ := 27
  headStart cristinaSpeed nickySpeed catchUpTime = 81 := by
sorry

end NUMINAMATH_CALUDE_race_head_start_l2910_291024


namespace NUMINAMATH_CALUDE_orchid_bushes_total_park_orchid_bushes_l2910_291038

/-- The total number of orchid bushes after planting is equal to the sum of the current number of bushes and the number of bushes planted over two days. -/
theorem orchid_bushes_total (current : ℕ) (today : ℕ) (tomorrow : ℕ) :
  current + today + tomorrow = current + today + tomorrow :=
by sorry

/-- Given the specific numbers from the problem -/
theorem park_orchid_bushes :
  let current : ℕ := 47
  let today : ℕ := 37
  let tomorrow : ℕ := 25
  current + today + tomorrow = 109 :=
by sorry

end NUMINAMATH_CALUDE_orchid_bushes_total_park_orchid_bushes_l2910_291038


namespace NUMINAMATH_CALUDE_reciprocal_comparison_l2910_291000

theorem reciprocal_comparison : 
  ((-1/3 : ℚ) < (-3 : ℚ) → False) ∧
  ((-3/2 : ℚ) < (-2/3 : ℚ)) ∧
  ((1/4 : ℚ) < (4 : ℚ)) ∧
  ((3/4 : ℚ) < (4/3 : ℚ) → False) ∧
  ((4/3 : ℚ) < (3/4 : ℚ) → False) := by
sorry

end NUMINAMATH_CALUDE_reciprocal_comparison_l2910_291000


namespace NUMINAMATH_CALUDE_line_parameterization_l2910_291034

/-- Given a line y = (2/3)x - 5 parameterized by (x, y) = (-6, p) + t(m, 7),
    prove that p = -9 and m = 21/2 -/
theorem line_parameterization (x y t : ℝ) (p m : ℝ) :
  (y = (2/3) * x - 5) →
  (∃ t, x = -6 + t * m ∧ y = p + t * 7) →
  (p = -9 ∧ m = 21/2) := by
  sorry

end NUMINAMATH_CALUDE_line_parameterization_l2910_291034


namespace NUMINAMATH_CALUDE_janet_has_five_dimes_l2910_291057

/-- Represents the number of coins of each type Janet has -/
structure CoinCount where
  nickels : ℕ
  dimes : ℕ
  quarters : ℕ

/-- The conditions of Janet's coin collection -/
def janet_coins (c : CoinCount) : Prop :=
  c.nickels + c.dimes + c.quarters = 10 ∧
  c.dimes + c.quarters = 7 ∧
  c.nickels + c.dimes = 8

/-- Theorem stating that Janet has 5 dimes -/
theorem janet_has_five_dimes :
  ∃ c : CoinCount, janet_coins c ∧ c.dimes = 5 := by
  sorry

end NUMINAMATH_CALUDE_janet_has_five_dimes_l2910_291057


namespace NUMINAMATH_CALUDE_sqrt_meaningful_range_l2910_291075

theorem sqrt_meaningful_range (a : ℝ) : 
  (∃ x : ℝ, x^2 = 2 - a) ↔ a ≤ 2 := by
sorry

end NUMINAMATH_CALUDE_sqrt_meaningful_range_l2910_291075


namespace NUMINAMATH_CALUDE_product_of_odd_primes_below_16_mod_32_l2910_291009

def odd_primes_below_16 : List Nat := [3, 5, 7, 11, 13]

theorem product_of_odd_primes_below_16_mod_32 :
  (List.prod odd_primes_below_16) % (2^5) = 7 := by
  sorry

end NUMINAMATH_CALUDE_product_of_odd_primes_below_16_mod_32_l2910_291009


namespace NUMINAMATH_CALUDE_range_of_a_l2910_291053

theorem range_of_a (a : ℝ) : 
  (∀ x : ℝ, x^2 + (a - 1)*x + 1 > 0) → (-1 < a ∧ a < 3) := by
  sorry

end NUMINAMATH_CALUDE_range_of_a_l2910_291053


namespace NUMINAMATH_CALUDE_part_one_part_two_l2910_291027

-- Define polynomials A, B, and C
def A (x y : ℝ) : ℝ := x^2 + x*y + 3*y
def B (x y : ℝ) : ℝ := x^2 - x*y

-- Theorem for part 1
theorem part_one (x y : ℝ) : 3 * A x y - B x y = 2*x^2 + 4*x*y + 9*y := by sorry

-- Theorem for part 2
theorem part_two (x y : ℝ) :
  (∃ C : ℝ → ℝ → ℝ, A x y + (1/3) * C x y = 2*x*y + 5*y) →
  (∃ C : ℝ → ℝ → ℝ, C x y = -3*x^2 + 3*x*y + 6*y) := by sorry

end NUMINAMATH_CALUDE_part_one_part_two_l2910_291027


namespace NUMINAMATH_CALUDE_petyas_number_l2910_291005

theorem petyas_number (x : ℝ) : x - x / 10 = 19.71 → x = 21.9 := by
  sorry

end NUMINAMATH_CALUDE_petyas_number_l2910_291005


namespace NUMINAMATH_CALUDE_monomial_count_l2910_291018

-- Define a type for algebraic expressions
inductive AlgebraicExpr
  | Constant (c : ℚ)
  | Variable (v : String)
  | Product (e1 e2 : AlgebraicExpr)
  | Sum (e1 e2 : AlgebraicExpr)
  | Fraction (num den : AlgebraicExpr)

-- Define what a monomial is
def isMonomial : AlgebraicExpr → Bool
  | AlgebraicExpr.Constant _ => true
  | AlgebraicExpr.Variable _ => true
  | AlgebraicExpr.Product e1 e2 => isMonomial e1 && isMonomial e2
  | _ => false

-- Define the list of given expressions
def givenExpressions : List AlgebraicExpr := [
  AlgebraicExpr.Product (AlgebraicExpr.Constant (-1/2)) (AlgebraicExpr.Product (AlgebraicExpr.Variable "m") (AlgebraicExpr.Variable "n")),
  AlgebraicExpr.Variable "m",
  AlgebraicExpr.Constant (1/2),
  AlgebraicExpr.Fraction (AlgebraicExpr.Variable "b") (AlgebraicExpr.Variable "a"),
  AlgebraicExpr.Sum (AlgebraicExpr.Product (AlgebraicExpr.Constant 2) (AlgebraicExpr.Variable "m")) (AlgebraicExpr.Constant 1),
  AlgebraicExpr.Fraction (AlgebraicExpr.Sum (AlgebraicExpr.Variable "x") (AlgebraicExpr.Product (AlgebraicExpr.Constant (-1)) (AlgebraicExpr.Variable "y"))) (AlgebraicExpr.Constant 5),
  AlgebraicExpr.Fraction 
    (AlgebraicExpr.Sum (AlgebraicExpr.Product (AlgebraicExpr.Constant 2) (AlgebraicExpr.Variable "x")) (AlgebraicExpr.Variable "y"))
    (AlgebraicExpr.Sum (AlgebraicExpr.Variable "x") (AlgebraicExpr.Product (AlgebraicExpr.Constant (-1)) (AlgebraicExpr.Variable "y"))),
  AlgebraicExpr.Sum 
    (AlgebraicExpr.Sum 
      (AlgebraicExpr.Product (AlgebraicExpr.Variable "x") (AlgebraicExpr.Variable "x")) 
      (AlgebraicExpr.Product (AlgebraicExpr.Constant 2) (AlgebraicExpr.Variable "x")))
    (AlgebraicExpr.Constant (3/2))
]

-- Theorem statement
theorem monomial_count : 
  (givenExpressions.filter isMonomial).length = 3 := by sorry

end NUMINAMATH_CALUDE_monomial_count_l2910_291018


namespace NUMINAMATH_CALUDE_friendly_numbers_solution_l2910_291088

/-- Two rational numbers are friendly if their sum is 66 -/
def friendly (m n : ℚ) : Prop := m + n = 66

/-- Given that 7x and -18 are friendly numbers, prove that x = 12 -/
theorem friendly_numbers_solution (x : ℚ) (h : friendly (7 * x) (-18)) : x = 12 := by
  sorry

end NUMINAMATH_CALUDE_friendly_numbers_solution_l2910_291088


namespace NUMINAMATH_CALUDE_compound_vs_simple_interest_l2910_291035

/-- Calculate compound interest given principal, rate, and time -/
def compound_interest (principal : ℝ) (rate : ℝ) (time : ℕ) : ℝ :=
  principal * (1 + rate) ^ time - principal

/-- Calculate simple interest given principal, rate, and time -/
def simple_interest (principal : ℝ) (rate : ℝ) (time : ℕ) : ℝ :=
  principal * rate * time

theorem compound_vs_simple_interest :
  ∀ P : ℝ,
  simple_interest P 0.1 2 = 600 →
  compound_interest P 0.1 2 = 630 := by
  sorry

end NUMINAMATH_CALUDE_compound_vs_simple_interest_l2910_291035


namespace NUMINAMATH_CALUDE_village_population_l2910_291048

theorem village_population (P : ℕ) : 
  (P : ℝ) * (1 - 0.05) * (1 - 0.15) = 3553 → P = 4400 := by
  sorry

end NUMINAMATH_CALUDE_village_population_l2910_291048


namespace NUMINAMATH_CALUDE_maria_final_amount_l2910_291089

def salary : ℝ := 2000
def tax_rate : ℝ := 0.20
def insurance_rate : ℝ := 0.05
def utility_rate : ℝ := 0.25

def remaining_after_deductions : ℝ := salary * (1 - tax_rate - insurance_rate)
def utility_bill : ℝ := remaining_after_deductions * utility_rate
def final_amount : ℝ := remaining_after_deductions - utility_bill

theorem maria_final_amount : final_amount = 1125 := by
  sorry

end NUMINAMATH_CALUDE_maria_final_amount_l2910_291089


namespace NUMINAMATH_CALUDE_fraction_equivalence_l2910_291012

theorem fraction_equivalence : 
  ∀ (n : ℚ), (3 + n) / (4 + n) = 4 / 5 → n = 1 := by sorry

end NUMINAMATH_CALUDE_fraction_equivalence_l2910_291012


namespace NUMINAMATH_CALUDE_quadratic_inequality_iff_abs_a_le_two_l2910_291083

theorem quadratic_inequality_iff_abs_a_le_two (a : ℝ) :
  (∀ x : ℝ, x^2 + a*x + 1 ≥ 0) ↔ |a| ≤ 2 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_inequality_iff_abs_a_le_two_l2910_291083


namespace NUMINAMATH_CALUDE_fraction_order_l2910_291087

theorem fraction_order : (21 : ℚ) / 17 < 23 / 18 ∧ 23 / 18 < 25 / 19 := by
  sorry

end NUMINAMATH_CALUDE_fraction_order_l2910_291087


namespace NUMINAMATH_CALUDE_square_ABCD_l2910_291047

/-- A point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- A quadrilateral defined by four points -/
structure Quadrilateral where
  A : Point
  B : Point
  C : Point
  D : Point

/-- Check if a quadrilateral is a square -/
def is_square (q : Quadrilateral) : Prop :=
  let AB := (q.B.x - q.A.x, q.B.y - q.A.y)
  let BC := (q.C.x - q.B.x, q.C.y - q.B.y)
  let CD := (q.D.x - q.C.x, q.D.y - q.C.y)
  let DA := (q.A.x - q.D.x, q.A.y - q.D.y)
  -- All sides have equal length
  AB.1^2 + AB.2^2 = BC.1^2 + BC.2^2 ∧
  BC.1^2 + BC.2^2 = CD.1^2 + CD.2^2 ∧
  CD.1^2 + CD.2^2 = DA.1^2 + DA.2^2 ∧
  -- Adjacent sides are perpendicular
  AB.1 * BC.1 + AB.2 * BC.2 = 0 ∧
  BC.1 * CD.1 + BC.2 * CD.2 = 0 ∧
  CD.1 * DA.1 + CD.2 * DA.2 = 0 ∧
  DA.1 * AB.1 + DA.2 * AB.2 = 0

theorem square_ABCD :
  let q := Quadrilateral.mk
    (Point.mk (-1) 3)
    (Point.mk 1 (-2))
    (Point.mk 6 0)
    (Point.mk 4 5)
  is_square q := by
  sorry

end NUMINAMATH_CALUDE_square_ABCD_l2910_291047


namespace NUMINAMATH_CALUDE_sum_of_squares_l2910_291069

theorem sum_of_squares (a b c : ℝ) : 
  a + b + c = 20 → 
  a * b + b * c + a * c = 131 → 
  a^2 + b^2 + c^2 = 138 := by
sorry

end NUMINAMATH_CALUDE_sum_of_squares_l2910_291069


namespace NUMINAMATH_CALUDE_collinear_points_k_l2910_291033

/-- A point in 2D space -/
structure Point2D where
  x : ℝ
  y : ℝ

/-- Check if three points are collinear -/
def collinear (p1 p2 p3 : Point2D) : Prop :=
  (p2.y - p1.y) * (p3.x - p1.x) = (p3.y - p1.y) * (p2.x - p1.x)

theorem collinear_points_k (k : ℝ) : 
  collinear ⟨1, -4⟩ ⟨3, 2⟩ ⟨6, k/3⟩ → k = 33 := by
  sorry


end NUMINAMATH_CALUDE_collinear_points_k_l2910_291033


namespace NUMINAMATH_CALUDE_cube_root_equation_solution_l2910_291073

theorem cube_root_equation_solution :
  ∀ x : ℝ, (((5 - x / 3) ^ (1/3 : ℝ) = -2) ↔ (x = 39)) :=
by sorry

end NUMINAMATH_CALUDE_cube_root_equation_solution_l2910_291073


namespace NUMINAMATH_CALUDE_orange_selling_gain_percentage_orange_selling_specific_gain_l2910_291045

/-- Calculates the gain percentage when changing selling rates of oranges -/
theorem orange_selling_gain_percentage 
  (initial_rate : ℝ) 
  (initial_loss_percentage : ℝ)
  (new_rate : ℝ) : ℝ :=
  let cost_price := 1 / (initial_rate * (1 - initial_loss_percentage / 100))
  let new_selling_price := 1 / new_rate
  let gain_percentage := (new_selling_price / cost_price - 1) * 100
  gain_percentage

/-- Proves that the specific change in orange selling rates results in a 44% gain -/
theorem orange_selling_specific_gain : 
  orange_selling_gain_percentage 12 10 7.5 = 44 := by
  sorry

end NUMINAMATH_CALUDE_orange_selling_gain_percentage_orange_selling_specific_gain_l2910_291045


namespace NUMINAMATH_CALUDE_coefficient_of_x_squared_l2910_291050

theorem coefficient_of_x_squared (k : ℝ) : 
  k = 1.7777777777777777 → 2 * k = 3.5555555555555554 := by
  sorry

end NUMINAMATH_CALUDE_coefficient_of_x_squared_l2910_291050


namespace NUMINAMATH_CALUDE_sticker_distribution_l2910_291097

theorem sticker_distribution (n : ℕ) (k : ℕ) (h1 : n = 10) (h2 : k = 3) :
  (Nat.choose (n - k + k - 1) (k - 1)) = 36 := by
  sorry

end NUMINAMATH_CALUDE_sticker_distribution_l2910_291097


namespace NUMINAMATH_CALUDE_probability_theorem_l2910_291040

def total_children : ℕ := 20
def num_girls : ℕ := 11
def num_boys : ℕ := 9

def probability_no_more_than_five_girls_between_first_last_boys : ℚ :=
  (Nat.choose 14 9 + 6 * Nat.choose 13 8) / Nat.choose 20 9

theorem probability_theorem :
  probability_no_more_than_five_girls_between_first_last_boys =
  (Nat.choose 14 9 + 6 * Nat.choose 13 8) / Nat.choose 20 9 :=
by sorry

end NUMINAMATH_CALUDE_probability_theorem_l2910_291040


namespace NUMINAMATH_CALUDE_point_translation_on_sine_curves_l2910_291031

theorem point_translation_on_sine_curves : ∃ (t s : ℝ),
  -- P(π/4, t) is on y = sin(x - π/12)
  t = Real.sin (π / 4 - π / 12) ∧
  -- s > 0
  s > 0 ∧
  -- P' is on y = sin(2x) after translation
  Real.sin (2 * (π / 4 - s)) = t ∧
  -- t = 1/2
  t = 1 / 2 ∧
  -- Minimum value of s = π/6
  s = π / 6 ∧
  -- s is the minimum positive value satisfying the conditions
  ∀ (s' : ℝ), s' > 0 → Real.sin (2 * (π / 4 - s')) = t → s ≤ s' := by
sorry

end NUMINAMATH_CALUDE_point_translation_on_sine_curves_l2910_291031


namespace NUMINAMATH_CALUDE_solve_jogging_problem_l2910_291065

def jogging_problem (daily_time : ℕ) (first_week_days : ℕ) (total_time : ℕ) : Prop :=
  let total_minutes : ℕ := total_time * 60
  let first_week_minutes : ℕ := first_week_days * daily_time
  let second_week_minutes : ℕ := total_minutes - first_week_minutes
  let second_week_days : ℕ := second_week_minutes / daily_time
  second_week_days = 5

theorem solve_jogging_problem :
  jogging_problem 30 3 4 := by sorry

end NUMINAMATH_CALUDE_solve_jogging_problem_l2910_291065


namespace NUMINAMATH_CALUDE_binomial_expansion_coefficient_relation_l2910_291032

theorem binomial_expansion_coefficient_relation (n : ℕ) : 
  (n.choose 2 * 3^(n-2) = 5 * n.choose 0 * 3^n) → n = 10 := by
  sorry

end NUMINAMATH_CALUDE_binomial_expansion_coefficient_relation_l2910_291032


namespace NUMINAMATH_CALUDE_tan_105_degrees_l2910_291007

theorem tan_105_degrees : Real.tan (105 * π / 180) = -2 - Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_tan_105_degrees_l2910_291007


namespace NUMINAMATH_CALUDE_alice_most_dogs_l2910_291049

-- Define the number of cats and dogs for each person
variable (Kc Ac Bc Kd Ad Bd : ℕ)

-- Define the conditions
variable (h1 : Kc > Ac)  -- Kathy owns more cats than Alice
variable (h2 : Kd > Bd)  -- Kathy owns more dogs than Bruce
variable (h3 : Ad > Kd)  -- Alice owns more dogs than Kathy
variable (h4 : Bc > Ac)  -- Bruce owns more cats than Alice

-- Theorem statement
theorem alice_most_dogs : Ad > Kd ∧ Ad > Bd := by
  sorry

end NUMINAMATH_CALUDE_alice_most_dogs_l2910_291049


namespace NUMINAMATH_CALUDE_max_value_quadratic_l2910_291090

theorem max_value_quadratic :
  (∀ r : ℝ, -3 * r^2 + 36 * r - 9 ≤ 99) ∧
  (∃ r : ℝ, -3 * r^2 + 36 * r - 9 = 99) :=
by sorry

end NUMINAMATH_CALUDE_max_value_quadratic_l2910_291090


namespace NUMINAMATH_CALUDE_fraction_value_at_four_l2910_291094

theorem fraction_value_at_four : 
  let x : ℝ := 4
  (x^6 - 64*x^3 + 512) / (x^3 - 16) = 48 := by
  sorry

end NUMINAMATH_CALUDE_fraction_value_at_four_l2910_291094


namespace NUMINAMATH_CALUDE_smallest_solution_floor_equation_l2910_291056

theorem smallest_solution_floor_equation :
  ∃ (x : ℝ), x > 0 ∧ 
    (⌊x^2⌋ - x * ⌊x⌋ = 6) ∧
    (∀ y : ℝ, y > 0 → (⌊y^2⌋ - y * ⌊y⌋ = 6) → x ≤ y) ∧
    x = 55 / 7 :=
by sorry

end NUMINAMATH_CALUDE_smallest_solution_floor_equation_l2910_291056


namespace NUMINAMATH_CALUDE_correct_calculation_l2910_291064

theorem correct_calculation (a b : ℝ) : 7 * a * b - 6 * a * b = a * b := by
  sorry

end NUMINAMATH_CALUDE_correct_calculation_l2910_291064


namespace NUMINAMATH_CALUDE_product_always_even_l2910_291002

theorem product_always_even (a b c : ℤ) : 
  Even ((a - b) * (b - c) * (c - a)) := by
  sorry

end NUMINAMATH_CALUDE_product_always_even_l2910_291002


namespace NUMINAMATH_CALUDE_circle_center_transformation_l2910_291060

def reflect_across_x_axis (p : ℝ × ℝ) : ℝ × ℝ :=
  (p.1, -p.2)

def translate_right (p : ℝ × ℝ) (d : ℝ) : ℝ × ℝ :=
  (p.1 + d, p.2)

theorem circle_center_transformation :
  let initial_center : ℝ × ℝ := (3, -4)
  let reflected_center := reflect_across_x_axis initial_center
  let final_center := translate_right reflected_center 5
  final_center = (8, 4) := by
sorry

end NUMINAMATH_CALUDE_circle_center_transformation_l2910_291060


namespace NUMINAMATH_CALUDE_trumpet_cost_l2910_291074

/-- The cost of the trumpet given the total spent and the costs of other items. -/
theorem trumpet_cost (total_spent music_tool_cost song_book_cost : ℚ) 
  (h1 : total_spent = 163.28)
  (h2 : music_tool_cost = 9.98)
  (h3 : song_book_cost = 4.14) :
  total_spent - (music_tool_cost + song_book_cost) = 149.16 := by
  sorry

end NUMINAMATH_CALUDE_trumpet_cost_l2910_291074


namespace NUMINAMATH_CALUDE_simplify_expression_l2910_291026

theorem simplify_expression : (256 : ℝ)^(1/4) * (125 : ℝ)^(1/3) = 20 := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l2910_291026


namespace NUMINAMATH_CALUDE_x_squared_gt_16_necessary_not_sufficient_for_x_gt_4_l2910_291022

theorem x_squared_gt_16_necessary_not_sufficient_for_x_gt_4 :
  (∃ x : ℝ, x^2 > 16 ∧ x ≤ 4) ∧
  (∀ x : ℝ, x > 4 → x^2 > 16) :=
by sorry

end NUMINAMATH_CALUDE_x_squared_gt_16_necessary_not_sufficient_for_x_gt_4_l2910_291022


namespace NUMINAMATH_CALUDE_project_budget_increase_l2910_291017

/-- Proves that the annual increase in budget for project Q is $50,000 --/
theorem project_budget_increase (initial_q initial_v annual_decrease_v : ℕ) 
  (h1 : initial_q = 540000)
  (h2 : initial_v = 780000)
  (h3 : annual_decrease_v = 10000)
  (h4 : ∃ (annual_increase_q : ℕ), 
    initial_q + 4 * annual_increase_q = initial_v - 4 * annual_decrease_v) :
  ∃ (annual_increase_q : ℕ), annual_increase_q = 50000 := by
  sorry

end NUMINAMATH_CALUDE_project_budget_increase_l2910_291017


namespace NUMINAMATH_CALUDE_rectangle_perimeter_l2910_291068

/-- Given a triangle with sides 10, 24, and 26 units, and a rectangle with width 8 units
    and area equal to the triangle's area, the perimeter of the rectangle is 46 units. -/
theorem rectangle_perimeter (a b c w : ℝ) (h1 : a = 10) (h2 : b = 24) (h3 : c = 26) (h4 : w = 8)
  (h5 : w * (a * b / 2 / w) = a * b / 2) : 2 * (w + (a * b / 2 / w)) = 46 :=
by sorry

end NUMINAMATH_CALUDE_rectangle_perimeter_l2910_291068


namespace NUMINAMATH_CALUDE_perfect_square_condition_l2910_291003

theorem perfect_square_condition (n : ℕ) : 
  (∃ k : ℕ, n^2 - 19*n + 91 = k^2) ↔ (n = 9 ∨ n = 10) :=
sorry

end NUMINAMATH_CALUDE_perfect_square_condition_l2910_291003


namespace NUMINAMATH_CALUDE_article_cost_price_l2910_291078

def cost_price : ℝ → Prop :=
  λ c => 
    ∃ s, 
      (s = 1.25 * c) ∧ 
      (s - 14.70 = 1.04 * c) ∧ 
      (c = 70)

theorem article_cost_price : 
  ∃ c, cost_price c :=
sorry

end NUMINAMATH_CALUDE_article_cost_price_l2910_291078


namespace NUMINAMATH_CALUDE_vector_dot_product_collinear_l2910_291077

/-- Two vectors are collinear if one is a scalar multiple of the other -/
def collinear (v w : ℝ × ℝ) : Prop :=
  ∃ (t : ℝ), v = (t * w.1, t * w.2)

/-- Dot product of two 2D vectors -/
def dot_product (v w : ℝ × ℝ) : ℝ :=
  v.1 * w.1 + v.2 * w.2

theorem vector_dot_product_collinear :
  ∀ (k : ℝ),
  let m : ℝ × ℝ := (2 * k - 1, k)
  let n : ℝ × ℝ := (4, 1)
  collinear m n → dot_product m n = -17/2 := by
  sorry

end NUMINAMATH_CALUDE_vector_dot_product_collinear_l2910_291077


namespace NUMINAMATH_CALUDE_first_month_sales_l2910_291063

def sales_second_month : ℤ := 8550
def sales_third_month : ℤ := 6855
def sales_fourth_month : ℤ := 3850
def sales_fifth_month : ℤ := 14045
def average_sale : ℤ := 7800
def num_months : ℤ := 5

theorem first_month_sales :
  (average_sale * num_months) - (sales_second_month + sales_third_month + sales_fourth_month + sales_fifth_month) = 8700 := by
  sorry

end NUMINAMATH_CALUDE_first_month_sales_l2910_291063


namespace NUMINAMATH_CALUDE_pokemon_card_difference_l2910_291062

-- Define the initial number of cards for Sally and Dan
def sally_initial : ℕ := 27
def dan_cards : ℕ := 41

-- Define the number of cards Sally bought
def sally_bought : ℕ := 20

-- Define Sally's total cards after buying
def sally_total : ℕ := sally_initial + sally_bought

-- Theorem to prove
theorem pokemon_card_difference : sally_total - dan_cards = 6 := by
  sorry

end NUMINAMATH_CALUDE_pokemon_card_difference_l2910_291062


namespace NUMINAMATH_CALUDE_distinct_selections_is_fifteen_l2910_291037

/-- The number of vowels in "MATHCOUNTS" -/
def num_vowels : ℕ := 3

/-- The number of distinct consonants in "MATHCOUNTS" excluding T -/
def num_distinct_consonants : ℕ := 5

/-- The number of T's in "MATHCOUNTS" -/
def num_t : ℕ := 2

/-- The total number of consonants in "MATHCOUNTS" -/
def total_consonants : ℕ := num_distinct_consonants + num_t

/-- The number of vowels to be selected -/
def vowels_to_select : ℕ := 3

/-- The number of consonants to be selected -/
def consonants_to_select : ℕ := 2

/-- The function to calculate the number of distinct ways to select letters -/
def distinct_selections : ℕ :=
  Nat.choose num_vowels vowels_to_select * Nat.choose num_distinct_consonants consonants_to_select +
  Nat.choose num_vowels vowels_to_select * Nat.choose (num_distinct_consonants - 1) (consonants_to_select - 1) +
  Nat.choose num_vowels vowels_to_select * Nat.choose (num_distinct_consonants - 2) (consonants_to_select - 2)

theorem distinct_selections_is_fifteen :
  distinct_selections = 15 :=
sorry

end NUMINAMATH_CALUDE_distinct_selections_is_fifteen_l2910_291037


namespace NUMINAMATH_CALUDE_altitude_angle_bisector_median_concurrent_l2910_291044

/-- Triangle ABC with sides a, b, c -/
structure Triangle :=
  (A B C : ℝ × ℝ)
  (a b c : ℝ)
  (side_a : dist B C = a)
  (side_b : dist C A = b)
  (side_c : dist A B = c)

/-- Altitude from A to BC -/
def altitude (t : Triangle) : ℝ × ℝ → ℝ × ℝ := sorry

/-- Angle bisector from B -/
def angle_bisector (t : Triangle) : ℝ × ℝ → ℝ × ℝ := sorry

/-- Median from C to AB -/
def median (t : Triangle) : ℝ × ℝ → ℝ × ℝ := sorry

/-- Three lines are concurrent -/
def concurrent (l₁ l₂ l₃ : ℝ × ℝ → ℝ × ℝ) : Prop := sorry

theorem altitude_angle_bisector_median_concurrent (t : Triangle) :
  concurrent (altitude t) (angle_bisector t) (median t) ↔
  t.a^2 * (t.a - t.c) = (t.b^2 - t.c^2) * (t.a + t.c) :=
sorry

end NUMINAMATH_CALUDE_altitude_angle_bisector_median_concurrent_l2910_291044


namespace NUMINAMATH_CALUDE_right_triangle_perimeter_l2910_291014

theorem right_triangle_perimeter (area : ℝ) (leg1 : ℝ) (leg2 : ℝ) (hypotenuse : ℝ) :
  area = 150 →
  leg1 = 30 →
  area = (1 / 2) * leg1 * leg2 →
  hypotenuse^2 = leg1^2 + leg2^2 →
  leg1 + leg2 + hypotenuse = 40 + 10 * Real.sqrt 10 :=
by sorry

end NUMINAMATH_CALUDE_right_triangle_perimeter_l2910_291014


namespace NUMINAMATH_CALUDE_problem_solution_l2910_291076

theorem problem_solution (a b : ℤ) 
  (h1 : 3015 * a + 3019 * b = 3023)
  (h2 : 3017 * a + 3021 * b = 3025) : 
  a - b = -3 := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l2910_291076


namespace NUMINAMATH_CALUDE_unique_fraction_sum_l2910_291043

theorem unique_fraction_sum (p : ℕ) (h_prime : Nat.Prime p) (h_odd : Odd p) :
  ∃! (n m : ℕ), n ≠ m ∧ 2/p = 1/n + 1/m ∧ n = (p + 1)/2 ∧ m = p * (p + 1)/2 :=
by sorry

end NUMINAMATH_CALUDE_unique_fraction_sum_l2910_291043


namespace NUMINAMATH_CALUDE_bank_queue_properties_l2910_291067

/-- Represents a queue of people with different operation times -/
structure BankQueue where
  total_people : Nat
  simple_ops : Nat
  long_ops : Nat
  simple_time : Nat
  long_time : Nat

/-- Calculates the minimum wasted person-minutes -/
def min_wasted_time (q : BankQueue) : Nat :=
  sorry

/-- Calculates the maximum wasted person-minutes -/
def max_wasted_time (q : BankQueue) : Nat :=
  sorry

/-- Calculates the expected wasted person-minutes for a random order -/
def expected_wasted_time (q : BankQueue) : Rat :=
  sorry

/-- Theorem stating the properties of the bank queue problem -/
theorem bank_queue_properties (q : BankQueue)
  (h1 : q.total_people = 8)
  (h2 : q.simple_ops = 5)
  (h3 : q.long_ops = 3)
  (h4 : q.simple_time = 1)
  (h5 : q.long_time = 5) :
  min_wasted_time q = 40 ∧
  max_wasted_time q = 100 ∧
  expected_wasted_time q = 84 := by
  sorry

end NUMINAMATH_CALUDE_bank_queue_properties_l2910_291067


namespace NUMINAMATH_CALUDE_fred_final_collection_l2910_291019

/-- Represents the types of coins Fred has --/
inductive Coin
  | Dime
  | Quarter
  | Nickel

/-- Represents Fred's coin collection --/
structure CoinCollection where
  dimes : ℕ
  quarters : ℕ
  nickels : ℕ

def initial_collection : CoinCollection :=
  { dimes := 7, quarters := 4, nickels := 12 }

def borrowed : CoinCollection :=
  { dimes := 3, quarters := 2, nickels := 0 }

def returned : CoinCollection :=
  { dimes := 0, quarters := 1, nickels := 5 }

def found_cents : ℕ := 50

def cents_per_dime : ℕ := 10

theorem fred_final_collection :
  ∃ (final : CoinCollection),
    final.dimes = 9 ∧
    final.quarters = 3 ∧
    final.nickels = 17 ∧
    final.dimes = initial_collection.dimes - borrowed.dimes + found_cents / cents_per_dime ∧
    final.quarters = initial_collection.quarters - borrowed.quarters + returned.quarters ∧
    final.nickels = initial_collection.nickels + returned.nickels :=
  sorry

end NUMINAMATH_CALUDE_fred_final_collection_l2910_291019


namespace NUMINAMATH_CALUDE_origin_midpoint_coordinates_l2910_291046

/-- Given two points A and B in a 2D Cartesian coordinate system, 
    this function returns true if the origin (0, 0) is the midpoint of AB. -/
def isOriginMidpoint (A B : ℝ × ℝ) : Prop :=
  (A.1 + B.1) / 2 = 0 ∧ (A.2 + B.2) / 2 = 0

/-- Theorem stating that if the origin is the midpoint of AB and 
    A has coordinates (-1, 2), then B has coordinates (1, -2). -/
theorem origin_midpoint_coordinates :
  let A : ℝ × ℝ := (-1, 2)
  let B : ℝ × ℝ := (1, -2)
  isOriginMidpoint A B → B = (1, -2) := by
  sorry


end NUMINAMATH_CALUDE_origin_midpoint_coordinates_l2910_291046


namespace NUMINAMATH_CALUDE_ericas_amount_l2910_291004

/-- The problem of calculating Erica's amount given the total and Sam's amount -/
theorem ericas_amount (total sam : ℚ) (h1 : total = 450.32) (h2 : sam = 325.67) :
  total - sam = 124.65 := by
  sorry

end NUMINAMATH_CALUDE_ericas_amount_l2910_291004


namespace NUMINAMATH_CALUDE_smallest_sum_with_same_probability_l2910_291061

/-- Represents a symmetrical die with faces numbered 1 to 6 -/
structure SymmetricalDie :=
  (faces : Fin 6)

/-- Represents a set of symmetrical dice -/
def DiceSet := List SymmetricalDie

/-- The probability of getting a specific sum when throwing the dice -/
def probability (d : DiceSet) (sum : Nat) : ℝ :=
  sorry

/-- The condition that the sum 2022 is possible with a positive probability -/
def sum_2022_possible (d : DiceSet) : Prop :=
  ∃ p : ℝ, p > 0 ∧ probability d 2022 = p

/-- The theorem stating the smallest possible sum with the same probability as 2022 -/
theorem smallest_sum_with_same_probability (d : DiceSet) 
  (h : sum_2022_possible d) : 
  ∃ p : ℝ, p > 0 ∧ 
    probability d 2022 = p ∧
    probability d 337 = p ∧
    ∀ (sum : Nat), sum < 337 → probability d sum < p :=
  sorry

end NUMINAMATH_CALUDE_smallest_sum_with_same_probability_l2910_291061


namespace NUMINAMATH_CALUDE_line_and_volume_proof_l2910_291082

-- Define the line l
def line_l (x y : ℝ) := x + y - 4 = 0

-- Define the parallel line
def parallel_line (x y : ℝ) := x + y - 1 = 0

-- Theorem statement
theorem line_and_volume_proof :
  -- Condition 1: Line l passes through (3,1)
  line_l 3 1 ∧
  -- Condition 2: Line l is parallel to x+y-1=0
  ∀ (x y : ℝ), line_l x y ↔ ∃ (k : ℝ), parallel_line (x + k) (y + k) →
  -- Conclusion 1: Equation of line l is x+y-4=0
  (∀ (x y : ℝ), line_l x y ↔ x + y - 4 = 0) ∧
  -- Conclusion 2: Volume of the geometric solid is (64/3)π
  (let volume := (64 / 3) * Real.pi
   volume = (1 / 3) * Real.pi * 4^2 * 4) :=
by sorry

end NUMINAMATH_CALUDE_line_and_volume_proof_l2910_291082


namespace NUMINAMATH_CALUDE_necessary_not_sufficient_l2910_291079

theorem necessary_not_sufficient (a : ℝ) :
  (∀ a, 1 / a > 1 → a < 1) ∧ (∃ a, a < 1 ∧ 1 / a ≤ 1) := by
  sorry

end NUMINAMATH_CALUDE_necessary_not_sufficient_l2910_291079

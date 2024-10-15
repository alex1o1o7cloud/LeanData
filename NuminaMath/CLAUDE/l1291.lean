import Mathlib

namespace NUMINAMATH_CALUDE_log_equality_implies_ratio_l1291_129116

theorem log_equality_implies_ratio (a b : ℝ) (ha : 0 < a) (hb : 0 < b) :
  (Real.log a / Real.log 9 = Real.log b / Real.log 12) ∧ 
  (Real.log a / Real.log 9 = Real.log (3*a + b) / Real.log 16) →
  b / a = (1 + Real.sqrt 13) / 2 := by
sorry

end NUMINAMATH_CALUDE_log_equality_implies_ratio_l1291_129116


namespace NUMINAMATH_CALUDE_fraction_sum_equality_l1291_129118

theorem fraction_sum_equality : 
  (1 : ℚ) / 3 + 1 / 2 - 5 / 6 + 1 / 5 + 1 / 4 - 9 / 20 - 5 / 6 = -5 / 6 := by
  sorry

end NUMINAMATH_CALUDE_fraction_sum_equality_l1291_129118


namespace NUMINAMATH_CALUDE_line_through_points_l1291_129140

/-- Given a line y = mx + c passing through points (3,2) and (7,14), prove that m - c = 10 -/
theorem line_through_points (m c : ℝ) : 
  (2 = m * 3 + c) → (14 = m * 7 + c) → m - c = 10 := by
  sorry

end NUMINAMATH_CALUDE_line_through_points_l1291_129140


namespace NUMINAMATH_CALUDE_smallest_divisible_by_1_to_12_l1291_129183

def is_divisible_by_range (n : ℕ) (a b : ℕ) : Prop :=
  ∀ i : ℕ, a ≤ i → i ≤ b → n % i = 0

theorem smallest_divisible_by_1_to_12 :
  ∃ (n : ℕ), n > 0 ∧ is_divisible_by_range n 1 12 ∧
  ∀ (m : ℕ), m > 0 → is_divisible_by_range m 1 12 → n ≤ m :=
by
  use 27720
  sorry

end NUMINAMATH_CALUDE_smallest_divisible_by_1_to_12_l1291_129183


namespace NUMINAMATH_CALUDE_multiple_of_four_is_multiple_of_two_l1291_129164

theorem multiple_of_four_is_multiple_of_two (n : ℕ) :
  (∀ k : ℕ, 4 ∣ k → 2 ∣ k) →
  4 ∣ n →
  2 ∣ n :=
by sorry

end NUMINAMATH_CALUDE_multiple_of_four_is_multiple_of_two_l1291_129164


namespace NUMINAMATH_CALUDE_plate_acceleration_l1291_129169

noncomputable def α : Real := Real.arccos 0.82
noncomputable def g : Real := 10

theorem plate_acceleration (R r m : Real) (h_R : R = 1) (h_r : r = 0.5) (h_m : m = 75) :
  let a := g * Real.sqrt ((1 - Real.cos α) / 2)
  let direction := α / 2
  a = 3 ∧ direction = Real.arcsin 0.2 := by sorry

end NUMINAMATH_CALUDE_plate_acceleration_l1291_129169


namespace NUMINAMATH_CALUDE_sqrt_seven_irrational_negative_one_third_rational_two_rational_decimal_rational_irrational_among_options_l1291_129182

theorem sqrt_seven_irrational :
  ¬ (∃ (p q : ℤ), q ≠ 0 ∧ Real.sqrt 7 = p / q) :=
by
  sorry

theorem negative_one_third_rational :
  ∃ (p q : ℤ), q ≠ 0 ∧ (-1 : ℚ) / 3 = p / q :=
by
  sorry

theorem two_rational :
  ∃ (p q : ℤ), q ≠ 0 ∧ (2 : ℚ) = p / q :=
by
  sorry

theorem decimal_rational :
  ∃ (p q : ℤ), q ≠ 0 ∧ (0.0101 : ℚ) = p / q :=
by
  sorry

theorem irrational_among_options :
  ¬ (∃ (p q : ℤ), q ≠ 0 ∧ Real.sqrt 7 = p / q) ∧
  (∃ (p q : ℤ), q ≠ 0 ∧ (-1 : ℚ) / 3 = p / q) ∧
  (∃ (p q : ℤ), q ≠ 0 ∧ (2 : ℚ) = p / q) ∧
  (∃ (p q : ℤ), q ≠ 0 ∧ (0.0101 : ℚ) = p / q) :=
by
  sorry

end NUMINAMATH_CALUDE_sqrt_seven_irrational_negative_one_third_rational_two_rational_decimal_rational_irrational_among_options_l1291_129182


namespace NUMINAMATH_CALUDE_amanda_quizzes_l1291_129142

/-- The number of quizzes Amanda has taken so far -/
def n : ℕ := sorry

/-- Amanda's average score on quizzes taken so far (as a percentage) -/
def current_average : ℚ := 92

/-- The required score on the final quiz to get an A (as a percentage) -/
def final_quiz_score : ℚ := 97

/-- The required average score over all quizzes to get an A (as a percentage) -/
def required_average : ℚ := 93

/-- The total number of quizzes including the final quiz -/
def total_quizzes : ℕ := 5

theorem amanda_quizzes : 
  n * current_average + final_quiz_score = required_average * total_quizzes ∧ n = 4 := by sorry

end NUMINAMATH_CALUDE_amanda_quizzes_l1291_129142


namespace NUMINAMATH_CALUDE_rhombus_area_l1291_129141

/-- The area of a rhombus with side length 4 and an interior angle of 45 degrees is 8√2 -/
theorem rhombus_area (s : ℝ) (θ : ℝ) (h1 : s = 4) (h2 : θ = π / 4) :
  s * s * Real.sin θ = 8 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_rhombus_area_l1291_129141


namespace NUMINAMATH_CALUDE_alex_hula_hoop_duration_l1291_129159

-- Define the hula hoop durations for each person
def nancy_duration : ℕ := 10

-- Casey's duration is 3 minutes less than Nancy's
def casey_duration : ℕ := nancy_duration - 3

-- Morgan's duration is three times Casey's duration
def morgan_duration : ℕ := casey_duration * 3

-- Alex's duration is the sum of Casey's and Morgan's durations minus 2 minutes
def alex_duration : ℕ := casey_duration + morgan_duration - 2

-- Theorem to prove Alex's hula hoop duration
theorem alex_hula_hoop_duration : alex_duration = 26 := by
  sorry

end NUMINAMATH_CALUDE_alex_hula_hoop_duration_l1291_129159


namespace NUMINAMATH_CALUDE_cos_alpha_value_l1291_129133

theorem cos_alpha_value (α : Real) (h : Real.sin (5 * Real.pi / 2 + α) = 1 / 5) :
  Real.cos α = 1 / 5 := by
  sorry

end NUMINAMATH_CALUDE_cos_alpha_value_l1291_129133


namespace NUMINAMATH_CALUDE_patio_table_cost_l1291_129158

/-- The cost of the patio table given the total cost and chair costs -/
theorem patio_table_cost (total_cost : ℕ) (chair_cost : ℕ) (num_chairs : ℕ) :
  total_cost = 135 →
  chair_cost = 20 →
  num_chairs = 4 →
  total_cost - (num_chairs * chair_cost) = 55 :=
by sorry

end NUMINAMATH_CALUDE_patio_table_cost_l1291_129158


namespace NUMINAMATH_CALUDE_two_digit_number_difference_divisibility_l1291_129194

theorem two_digit_number_difference_divisibility (A B : Nat) 
  (h1 : A ≠ B) (h2 : A > B) (h3 : A < 10) (h4 : B < 10) : 
  ∃ k : Int, (10 * A + B) - ((10 * B + A) - 5) = 3 * k := by
  sorry

end NUMINAMATH_CALUDE_two_digit_number_difference_divisibility_l1291_129194


namespace NUMINAMATH_CALUDE_solution_set_quadratic_inequality_l1291_129154

theorem solution_set_quadratic_inequality (x : ℝ) :
  x^2 - |x| - 2 < 0 ↔ -2 < x ∧ x < 2 := by
  sorry

end NUMINAMATH_CALUDE_solution_set_quadratic_inequality_l1291_129154


namespace NUMINAMATH_CALUDE_f_decreasing_interval_l1291_129100

/-- The function f(x) = x^2(ax + b) where a and b are real numbers. -/
def f (a b : ℝ) (x : ℝ) : ℝ := x^2 * (a * x + b)

/-- The derivative of f(x) -/
def f_prime (a b : ℝ) (x : ℝ) : ℝ := 3 * a * x^2 + 2 * b * x

theorem f_decreasing_interval (a b : ℝ) :
  (f_prime a b 2 = 0) →  -- f has an extremum at x = 2
  (f_prime a b 1 = -3) →  -- tangent line at (1, f(1)) is parallel to 3x + y = 0
  ∀ x, 0 < x → x < 2 → f_prime a b x < 0 :=
by sorry

end NUMINAMATH_CALUDE_f_decreasing_interval_l1291_129100


namespace NUMINAMATH_CALUDE_egg_distribution_l1291_129198

theorem egg_distribution (total_eggs : ℕ) (num_groups : ℕ) (eggs_per_group : ℕ) : 
  total_eggs = 8 → num_groups = 4 → eggs_per_group = total_eggs / num_groups → eggs_per_group = 2 := by
  sorry

end NUMINAMATH_CALUDE_egg_distribution_l1291_129198


namespace NUMINAMATH_CALUDE_water_bottles_used_second_game_l1291_129197

theorem water_bottles_used_second_game 
  (initial_cases : ℕ)
  (bottles_per_case : ℕ)
  (bottles_used_first_game : ℕ)
  (bottles_remaining_after_second_game : ℕ)
  (h1 : initial_cases = 10)
  (h2 : bottles_per_case = 20)
  (h3 : bottles_used_first_game = 70)
  (h4 : bottles_remaining_after_second_game = 20) :
  initial_cases * bottles_per_case - bottles_used_first_game - bottles_remaining_after_second_game = 110 :=
by sorry

end NUMINAMATH_CALUDE_water_bottles_used_second_game_l1291_129197


namespace NUMINAMATH_CALUDE_bill_apples_left_l1291_129151

/-- The number of apples Bill has left after distributing them -/
def apples_left (total : ℕ) (children : ℕ) (apples_per_child : ℕ) (pies : ℕ) (apples_per_pie : ℕ) : ℕ :=
  total - (children * apples_per_child + pies * apples_per_pie)

/-- Theorem: Bill has 24 apples left -/
theorem bill_apples_left :
  apples_left 50 2 3 2 10 = 24 := by
  sorry

end NUMINAMATH_CALUDE_bill_apples_left_l1291_129151


namespace NUMINAMATH_CALUDE_sandwich_combinations_l1291_129124

def num_meats : ℕ := 10
def num_cheeses : ℕ := 12
def num_condiments : ℕ := 5

theorem sandwich_combinations :
  (num_meats) * (num_cheeses.choose 2) * (num_condiments) = 3300 := by
  sorry

end NUMINAMATH_CALUDE_sandwich_combinations_l1291_129124


namespace NUMINAMATH_CALUDE_brownies_per_neighbor_l1291_129185

/-- Calculates the number of brownies each neighbor receives given the following conditions:
  * Melanie baked 15 batches of brownies
  * Each batch contains 30 brownies
  * She set aside 13/15 of the brownies in each batch for a bake sale
  * She placed 7/10 of the remaining brownies in a container
  * She donated 3/5 of what was left to a local charity
  * She wants to evenly distribute the rest among x neighbors
-/
theorem brownies_per_neighbor (x : ℕ) (x_pos : x > 0) : 
  let total_brownies := 15 * 30
  let bake_sale_brownies := (13 / 15 : ℚ) * total_brownies
  let remaining_after_bake_sale := total_brownies - bake_sale_brownies.floor
  let container_brownies := (7 / 10 : ℚ) * remaining_after_bake_sale
  let remaining_after_container := remaining_after_bake_sale - container_brownies.floor
  let charity_brownies := (3 / 5 : ℚ) * remaining_after_container
  let final_remaining := remaining_after_container - charity_brownies.floor
  (final_remaining / x : ℚ) = 8 / x := by
    sorry

#check brownies_per_neighbor

end NUMINAMATH_CALUDE_brownies_per_neighbor_l1291_129185


namespace NUMINAMATH_CALUDE_travel_group_combinations_l1291_129138

def total_friends : ℕ := 12
def friends_to_choose : ℕ := 5
def previously_traveled_friends : ℕ := 6

theorem travel_group_combinations : 
  (total_friends.choose friends_to_choose) - 
  ((total_friends - previously_traveled_friends).choose friends_to_choose) = 786 := by
  sorry

end NUMINAMATH_CALUDE_travel_group_combinations_l1291_129138


namespace NUMINAMATH_CALUDE_min_four_digit_quotient_l1291_129143

/-- A type representing a base-ten digit (1-9) -/
def Digit := { n : Nat // 1 ≤ n ∧ n ≤ 9 }

/-- The function to be minimized -/
def f (a b c d : Digit) : ℚ :=
  (1000 * a.val + 100 * b.val + 10 * c.val + d.val) / (a.val + b.val + c.val + d.val)

/-- The theorem stating the minimum value of the function -/
theorem min_four_digit_quotient :
  ∀ (a b c d : Digit),
    a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d →
    f a b c d ≥ 80.56 :=
sorry

end NUMINAMATH_CALUDE_min_four_digit_quotient_l1291_129143


namespace NUMINAMATH_CALUDE_negation_of_universal_quantifier_negation_of_quadratic_inequality_l1291_129175

theorem negation_of_universal_quantifier (p : ℝ → Prop) :
  (¬ ∀ x : ℝ, p x) ↔ (∃ x : ℝ, ¬ p x) := by sorry

theorem negation_of_quadratic_inequality :
  (¬ ∀ x : ℝ, x^2 - 3*x + 3 > 0) ↔ (∃ x : ℝ, x^2 - 3*x + 3 ≤ 0) := by sorry

end NUMINAMATH_CALUDE_negation_of_universal_quantifier_negation_of_quadratic_inequality_l1291_129175


namespace NUMINAMATH_CALUDE_quadratic_inequality_l1291_129104

theorem quadratic_inequality (m : ℝ) : (∃ x : ℝ, x^2 - x - m = 0) → m ≥ -1/4 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_inequality_l1291_129104


namespace NUMINAMATH_CALUDE_sibling_ages_theorem_l1291_129112

theorem sibling_ages_theorem :
  ∃ (a b c : ℕ+), 
    a * b * c = 72 ∧ 
    a + b + c = 13 ∧ 
    a > b ∧ b > c :=
by sorry

end NUMINAMATH_CALUDE_sibling_ages_theorem_l1291_129112


namespace NUMINAMATH_CALUDE_window_side_length_l1291_129111

/-- Represents the dimensions of a window pane -/
structure Pane where
  height : ℝ
  width : ℝ

/-- Represents the dimensions and properties of a window -/
structure Window where
  paneCount : ℕ
  rows : ℕ
  columns : ℕ
  pane : Pane
  borderWidth : ℝ

/-- The theorem stating that given the specified conditions, the window's side length is 27 inches -/
theorem window_side_length (w : Window) : 
  w.paneCount = 8 ∧ 
  w.rows = 2 ∧ 
  w.columns = 4 ∧ 
  w.pane.height = 3 * w.pane.width ∧
  w.borderWidth = 3 →
  (w.columns * w.pane.width + (w.columns + 1) * w.borderWidth : ℝ) = 27 :=
by sorry


end NUMINAMATH_CALUDE_window_side_length_l1291_129111


namespace NUMINAMATH_CALUDE_expression_value_l1291_129187

theorem expression_value : 103^4 - 4 * 103^3 + 6 * 103^2 - 4 * 103 + 1 = 108243216 := by
  sorry

end NUMINAMATH_CALUDE_expression_value_l1291_129187


namespace NUMINAMATH_CALUDE_typing_area_percentage_l1291_129123

/-- Calculates the percentage of a rectangular sheet used for typing, given the sheet dimensions and margins. -/
theorem typing_area_percentage (sheet_width sheet_length side_margin top_bottom_margin : ℝ) :
  sheet_width = 20 ∧ 
  sheet_length = 30 ∧ 
  side_margin = 2 ∧ 
  top_bottom_margin = 3 →
  (sheet_width - 2 * side_margin) * (sheet_length - 2 * top_bottom_margin) / (sheet_width * sheet_length) * 100 = 64 := by
  sorry

#check typing_area_percentage

end NUMINAMATH_CALUDE_typing_area_percentage_l1291_129123


namespace NUMINAMATH_CALUDE_riverdale_rangers_loss_percentage_l1291_129174

/-- Represents the statistics of a sports team --/
structure TeamStats where
  totalGames : ℕ
  winLossRatio : ℚ

/-- Calculates the percentage of games lost --/
def percentLost (stats : TeamStats) : ℚ :=
  let lostGames := stats.totalGames / (1 + stats.winLossRatio)
  (lostGames / stats.totalGames) * 100

/-- Theorem stating that for a team with given statistics, the percentage of games lost is 38% --/
theorem riverdale_rangers_loss_percentage :
  let stats : TeamStats := { totalGames := 65, winLossRatio := 8/5 }
  percentLost stats = 38 := by sorry


end NUMINAMATH_CALUDE_riverdale_rangers_loss_percentage_l1291_129174


namespace NUMINAMATH_CALUDE_division_problem_l1291_129155

theorem division_problem : (5 + 1/2) / (2/11) = 121/4 := by
  sorry

end NUMINAMATH_CALUDE_division_problem_l1291_129155


namespace NUMINAMATH_CALUDE_bookstore_max_revenue_l1291_129103

/-- The revenue function for the bookstore -/
def R (p : ℝ) : ℝ := p * (200 - 8 * p)

/-- The theorem stating the maximum revenue and optimal price -/
theorem bookstore_max_revenue :
  ∃ (p : ℝ), 0 ≤ p ∧ p ≤ 25 ∧
  R p = 1250 ∧
  ∀ (q : ℝ), 0 ≤ q ∧ q ≤ 25 → R q ≤ R p ∧
  p = 12.5 := by
  sorry

end NUMINAMATH_CALUDE_bookstore_max_revenue_l1291_129103


namespace NUMINAMATH_CALUDE_room_dimension_increase_l1291_129125

/-- Given the cost of painting a room and the cost of painting an enlarged version of the same room,
    calculate the factor by which the room's dimensions were increased. -/
theorem room_dimension_increase (original_cost enlarged_cost : ℝ) 
    (h1 : original_cost = 350)
    (h2 : enlarged_cost = 3150) :
    ∃ (n : ℝ), n = 3 ∧ enlarged_cost = n^2 * original_cost := by
  sorry

end NUMINAMATH_CALUDE_room_dimension_increase_l1291_129125


namespace NUMINAMATH_CALUDE_equation_solution_l1291_129122

theorem equation_solution (x y : ℝ) :
  x^5 + y^5 = 33 ∧ x + y = 3 →
  (x = 2 ∧ y = 1) ∨ (x = 1 ∧ y = 2) := by
sorry

end NUMINAMATH_CALUDE_equation_solution_l1291_129122


namespace NUMINAMATH_CALUDE_files_sorted_in_one_and_half_hours_l1291_129120

/-- Represents the number of files sorted by a group of clerks under specific conditions. -/
def filesSortedInOneAndHalfHours (totalFiles : ℕ) (filesPerHourPerClerk : ℕ) (totalTime : ℚ) : ℕ :=
  let initialClerks := 22  -- Derived from the problem conditions
  let reassignedClerks := 3  -- Derived from the problem conditions
  initialClerks * filesPerHourPerClerk + (initialClerks - reassignedClerks) * (filesPerHourPerClerk / 2)

/-- Proves that under the given conditions, the number of files sorted in 1.5 hours is 945. -/
theorem files_sorted_in_one_and_half_hours :
  filesSortedInOneAndHalfHours 1775 30 (157/60) = 945 := by
  sorry

#eval filesSortedInOneAndHalfHours 1775 30 (157/60)

end NUMINAMATH_CALUDE_files_sorted_in_one_and_half_hours_l1291_129120


namespace NUMINAMATH_CALUDE_set_operations_l1291_129150

def U : Set Nat := {1,2,3,4,5,6,7,8,9,10,11,13}
def A : Set Nat := {2,4,6,8}
def B : Set Nat := {3,4,5,6,8,9,11}

theorem set_operations :
  (A ∪ B = {2,3,4,5,6,8,9,11}) ∧
  (U \ A = {1,3,5,7,9,10,11,13}) ∧
  (U \ (A ∩ B) = {1,2,3,5,7,9,10,11,13}) ∧
  (A ∪ (U \ B) = {1,2,4,6,7,8,10,13}) := by
  sorry

end NUMINAMATH_CALUDE_set_operations_l1291_129150


namespace NUMINAMATH_CALUDE_slope_product_no_circle_through_A_l1291_129121

-- Define the ellipse
def E (x y : ℝ) : Prop := x^2/4 + y^2 = 1

-- Define points A and B
def A : ℝ × ℝ := (-2, 0)
def B : ℝ × ℝ := (2, 0)

-- Define a point P on the ellipse
def P (x₀ y₀ : ℝ) : Prop := E x₀ y₀ ∧ (x₀, y₀) ≠ A ∧ (x₀, y₀) ≠ B

-- Theorem: Product of slopes of PA and PB is -1/4
theorem slope_product (x₀ y₀ : ℝ) (h : P x₀ y₀) :
  (y₀ / (x₀ + 2)) * (y₀ / (x₀ - 2)) = -1/4 := by sorry

-- No circle with diameter MN passes through A
-- This part is more complex and would require additional definitions and theorems
-- We'll represent it as a proposition without proof
theorem no_circle_through_A (M N : ℝ × ℝ) (hM : E M.1 M.2) (hN : E N.1 N.2) :
  ¬∃ (center : ℝ × ℝ) (radius : ℝ), 
    (center.1 - M.1)^2 + (center.2 - M.2)^2 = radius^2 ∧
    (center.1 - N.1)^2 + (center.2 - N.2)^2 = radius^2 ∧
    (center.1 - A.1)^2 + (center.2 - A.2)^2 = radius^2 := by sorry

end NUMINAMATH_CALUDE_slope_product_no_circle_through_A_l1291_129121


namespace NUMINAMATH_CALUDE_motion_of_q_l1291_129105

/-- Point on a circle -/
structure Point2D where
  x : ℝ
  y : ℝ

/-- Motion of a point on a circle -/
structure CircularMotion where
  center : Point2D
  radius : ℝ
  angular_velocity : ℝ
  clockwise : Bool

/-- Given a point P moving counterclockwise on the unit circle with angular velocity ω,
    prove that the point Q(-2xy, y^2 - x^2) moves clockwise on the unit circle
    with angular velocity 2ω -/
theorem motion_of_q (ω : ℝ) (h_ω : ω > 0) :
  let p_motion : CircularMotion :=
    { center := ⟨0, 0⟩
    , radius := 1
    , angular_velocity := ω
    , clockwise := false }
  let q (p : Point2D) : Point2D :=
    ⟨-2 * p.x * p.y, p.y^2 - p.x^2⟩
  ∃ (q_motion : CircularMotion),
    q_motion.center = ⟨0, 0⟩ ∧
    q_motion.radius = 1 ∧
    q_motion.angular_velocity = 2 * ω ∧
    q_motion.clockwise = true :=
by sorry

end NUMINAMATH_CALUDE_motion_of_q_l1291_129105


namespace NUMINAMATH_CALUDE_function_equation_solution_l1291_129115

theorem function_equation_solution (f : ℝ → ℝ) 
  (h : ∀ x y : ℝ, x * f y + y * f x = (x + y) * f x * f y) : 
  (∀ x : ℝ, f x = 0) ∨ (∀ x : ℝ, f x = 1) := by
sorry

end NUMINAMATH_CALUDE_function_equation_solution_l1291_129115


namespace NUMINAMATH_CALUDE_y1_greater_than_y2_l1291_129176

/-- A quadratic function passing through points (0, y₁) and (4, y₂) -/
def quadratic_function (c y₁ y₂ : ℝ) : Prop :=
  y₁ = c ∧ y₂ = 16 - 24 + c

/-- Theorem stating that y₁ > y₂ for the given quadratic function -/
theorem y1_greater_than_y2 (c y₁ y₂ : ℝ) (h : quadratic_function c y₁ y₂) : y₁ > y₂ := by
  sorry

end NUMINAMATH_CALUDE_y1_greater_than_y2_l1291_129176


namespace NUMINAMATH_CALUDE_arithmetic_sequence_first_term_l1291_129145

/-- Sum of first n terms of an arithmetic sequence -/
def T (a₁ : ℚ) (d : ℚ) (n : ℕ) : ℚ :=
  n * (2 * a₁ + (n - 1) * d) / 2

/-- The problem statement -/
theorem arithmetic_sequence_first_term
  (h : ∃ (k : ℚ), ∀ (n : ℕ), n > 0 → T a₁ 5 (2*n) / T a₁ 5 n = k) :
  a₁ = 5/2 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_first_term_l1291_129145


namespace NUMINAMATH_CALUDE_digit_difference_quotient_l1291_129166

/-- Given that 524 in base 7 equals 3cd in base 10, where c and d are single digits,
    prove that (c - d) / 5 = -0.8 -/
theorem digit_difference_quotient (c d : ℕ) : 
  c < 10 → d < 10 → (5 * 7^2 + 2 * 7 + 4 : ℕ) = 300 + 10 * c + d → 
  (c - d : ℚ) / 5 = -4/5 := by sorry

end NUMINAMATH_CALUDE_digit_difference_quotient_l1291_129166


namespace NUMINAMATH_CALUDE_parallel_planes_transitivity_l1291_129148

structure Plane

/-- Two planes are parallel -/
def parallel (p q : Plane) : Prop := sorry

theorem parallel_planes_transitivity 
  (α β γ : Plane) 
  (h1 : α ≠ β) 
  (h2 : α ≠ γ) 
  (h3 : β ≠ γ) 
  (h4 : parallel α β) 
  (h5 : parallel α γ) : 
  parallel β γ := by sorry

end NUMINAMATH_CALUDE_parallel_planes_transitivity_l1291_129148


namespace NUMINAMATH_CALUDE_juelz_sisters_count_l1291_129184

theorem juelz_sisters_count (total_pieces : ℕ) (eaten_percentage : ℚ) (pieces_per_sister : ℕ) : 
  total_pieces = 240 →
  eaten_percentage = 60 / 100 →
  pieces_per_sister = 32 →
  (total_pieces - (eaten_percentage * total_pieces).num) / pieces_per_sister = 3 :=
by sorry

end NUMINAMATH_CALUDE_juelz_sisters_count_l1291_129184


namespace NUMINAMATH_CALUDE_infinite_solutions_imply_c_equals_three_l1291_129165

theorem infinite_solutions_imply_c_equals_three :
  (∀ y : ℝ, 3 * (3 + 2 * c * y) = 18 * y + 9) → c = 3 :=
by
  sorry

end NUMINAMATH_CALUDE_infinite_solutions_imply_c_equals_three_l1291_129165


namespace NUMINAMATH_CALUDE_success_rate_is_70_percent_l1291_129129

def games_played : ℕ := 15
def games_won : ℕ := 9
def remaining_games : ℕ := 5

def total_games : ℕ := games_played + remaining_games
def total_wins : ℕ := games_won + remaining_games

def success_rate : ℚ := (total_wins : ℚ) / (total_games : ℚ)

theorem success_rate_is_70_percent :
  success_rate = 7/10 :=
sorry

end NUMINAMATH_CALUDE_success_rate_is_70_percent_l1291_129129


namespace NUMINAMATH_CALUDE_biquadratic_equation_roots_l1291_129106

theorem biquadratic_equation_roots (x : ℝ) :
  x^4 - 8*x^2 + 4 = 0 ↔ x = Real.sqrt 3 - 1 ∨ x = Real.sqrt 3 + 1 ∨ x = -(Real.sqrt 3 - 1) ∨ x = -(Real.sqrt 3 + 1) :=
sorry

end NUMINAMATH_CALUDE_biquadratic_equation_roots_l1291_129106


namespace NUMINAMATH_CALUDE_consultant_decision_probability_l1291_129114

theorem consultant_decision_probability :
  let p : ℝ := 0.8  -- probability of each consultant being correct
  let n : ℕ := 3    -- number of consultants
  let k : ℕ := 2    -- minimum number of correct opinions for a correct decision
  -- probability of making the correct decision
  (Finset.sum (Finset.range (n + 1 - k)) (λ i => 
    (n.choose (n - i)) * p^(n - i) * (1 - p)^i)) = 0.896 := by
  sorry

end NUMINAMATH_CALUDE_consultant_decision_probability_l1291_129114


namespace NUMINAMATH_CALUDE_paco_ate_fifteen_sweet_cookies_l1291_129160

/-- The number of sweet cookies Paco ate -/
def sweet_cookies_eaten (initial_sweet : ℕ) (sweet_left : ℕ) : ℕ :=
  initial_sweet - sweet_left

/-- Theorem stating that Paco ate 15 sweet cookies -/
theorem paco_ate_fifteen_sweet_cookies : 
  sweet_cookies_eaten 34 19 = 15 := by
  sorry

end NUMINAMATH_CALUDE_paco_ate_fifteen_sweet_cookies_l1291_129160


namespace NUMINAMATH_CALUDE_cubic_equation_equivalence_l1291_129135

theorem cubic_equation_equivalence (y : ℝ) :
  6 * y^(1/3) - 3 * (y^2 / y^(2/3)) = 12 + y^(1/3) + y →
  ∃ z : ℝ, z = y^(1/3) ∧ 3 * z^4 + z^3 - 5 * z + 12 = 0 :=
by sorry

end NUMINAMATH_CALUDE_cubic_equation_equivalence_l1291_129135


namespace NUMINAMATH_CALUDE_two_numbers_with_ratio_and_square_difference_l1291_129146

theorem two_numbers_with_ratio_and_square_difference (p q : ℝ) (hp : p > 0) (hpn : p ≠ 1) (hq : q > 0) :
  let x : ℝ := q / (p - 1)
  let y : ℝ := p * q / (p - 1)
  y / x = p ∧ (y^2 - x^2) / (y + x) = q := by
  sorry

end NUMINAMATH_CALUDE_two_numbers_with_ratio_and_square_difference_l1291_129146


namespace NUMINAMATH_CALUDE_min_square_value_l1291_129161

theorem min_square_value (a b : ℕ+) 
  (h1 : ∃ r : ℕ, (15 * a + 16 * b : ℕ) = r^2)
  (h2 : ∃ s : ℕ, (16 * a - 15 * b : ℕ) = s^2) :
  min (15 * a + 16 * b) (16 * a - 15 * b) ≥ 231361 := by
sorry

end NUMINAMATH_CALUDE_min_square_value_l1291_129161


namespace NUMINAMATH_CALUDE_valid_sequences_of_length_21_l1291_129193

/-- Counts valid binary sequences of given length -/
def count_valid_sequences (n : ℕ) : ℕ :=
  if n < 3 then 0
  else if n = 3 then 1
  else if n = 4 then 1
  else if n = 5 then 1
  else if n = 6 then 2
  else count_valid_sequences (n - 4) + 2 * count_valid_sequences (n - 5) + 2 * count_valid_sequences (n - 6)

/-- The main theorem stating the number of valid sequences of length 21 -/
theorem valid_sequences_of_length_21 :
  count_valid_sequences 21 = 135 := by sorry

end NUMINAMATH_CALUDE_valid_sequences_of_length_21_l1291_129193


namespace NUMINAMATH_CALUDE_light_path_in_cube_l1291_129101

/-- Represents a point in 3D space -/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- Represents a cube -/
structure Cube where
  sideLength : ℝ

/-- Represents a light path in the cube -/
structure LightPath where
  start : Point3D
  reflection : Point3D
  length : ℝ

/-- Theorem stating the properties of the light path in the cube -/
theorem light_path_in_cube (cube : Cube) (path : LightPath) :
  cube.sideLength = 12 →
  path.start = Point3D.mk 0 0 0 →
  path.reflection = Point3D.mk 12 5 7 →
  ∃ (m n : ℕ), 
    path.length = m * Real.sqrt n ∧ 
    ¬ ∃ (p : ℕ), Prime p ∧ p^2 ∣ n ∧
    m + n = 230 := by
  sorry

end NUMINAMATH_CALUDE_light_path_in_cube_l1291_129101


namespace NUMINAMATH_CALUDE_ivan_dice_count_l1291_129191

theorem ivan_dice_count (x : ℕ) : 
  x + 2*x = 60 → x = 20 := by
  sorry

end NUMINAMATH_CALUDE_ivan_dice_count_l1291_129191


namespace NUMINAMATH_CALUDE_cone_sphere_ratio_l1291_129199

theorem cone_sphere_ratio (r h : ℝ) (h_pos : 0 < r) : 
  (1 / 3 : ℝ) * (4 / 3 * π * r^3) = (1 / 3 : ℝ) * π * r^2 * h → h / r = 4 / 3 := by
  sorry

end NUMINAMATH_CALUDE_cone_sphere_ratio_l1291_129199


namespace NUMINAMATH_CALUDE_pairwise_disjoint_sequences_l1291_129108

def largest_prime_power_divisor (n : ℕ) : ℕ := sorry

theorem pairwise_disjoint_sequences 
  (n : Fin 10000 → ℕ) 
  (h_distinct : ∀ i j, i ≠ j → n i ≠ n j) 
  (h_distinct_lpd : ∀ i j, i ≠ j → 
    largest_prime_power_divisor (n i) ≠ largest_prime_power_divisor (n j)) :
  ∃ a : Fin 10000 → ℤ, ∀ i j k l, i ≠ j → 
    a i + k * n i ≠ a j + l * n j :=
sorry

end NUMINAMATH_CALUDE_pairwise_disjoint_sequences_l1291_129108


namespace NUMINAMATH_CALUDE_bird_count_difference_l1291_129167

/-- Represents the count of birds on a single day -/
structure DailyCount where
  bluejays : ℕ
  cardinals : ℕ

/-- Calculates the difference between cardinals and blue jays for a single day -/
def dailyDifference (count : DailyCount) : ℤ :=
  count.cardinals - count.bluejays

/-- Theorem: The total difference between cardinals and blue jays over three days is 3 -/
theorem bird_count_difference (day1 day2 day3 : DailyCount)
  (h1 : day1 = { bluejays := 2, cardinals := 3 })
  (h2 : day2 = { bluejays := 3, cardinals := 3 })
  (h3 : day3 = { bluejays := 2, cardinals := 4 }) :
  dailyDifference day1 + dailyDifference day2 + dailyDifference day3 = 3 := by
  sorry

#eval dailyDifference { bluejays := 2, cardinals := 3 } +
      dailyDifference { bluejays := 3, cardinals := 3 } +
      dailyDifference { bluejays := 2, cardinals := 4 }

end NUMINAMATH_CALUDE_bird_count_difference_l1291_129167


namespace NUMINAMATH_CALUDE_ellipse_focus_k_value_l1291_129126

/-- Theorem: For an ellipse with equation x²/a² + y²/k = 1 and a focus at (0, √2), k = 2 -/
theorem ellipse_focus_k_value (a : ℝ) (k : ℝ) :
  (∀ x y : ℝ, x^2 / a^2 + y^2 / k = 1) →  -- Ellipse equation
  (0^2 / a^2 + (Real.sqrt 2)^2 / k = 1) →  -- Focus (0, √2) is on the ellipse
  k = 2 :=
by sorry

end NUMINAMATH_CALUDE_ellipse_focus_k_value_l1291_129126


namespace NUMINAMATH_CALUDE_ellipse_condition_l1291_129119

/-- The equation of the graph --/
def equation (x y k : ℝ) : Prop :=
  x^2 + 4*y^2 - 10*x + 56*y = k

/-- Definition of a non-degenerate ellipse --/
def is_non_degenerate_ellipse (k : ℝ) : Prop :=
  ∃ a b c d e : ℝ, a > 0 ∧ b > 0 ∧ 
    ∀ x y : ℝ, equation x y k ↔ (x - c)^2 / a + (y - d)^2 / b = e

/-- The main theorem --/
theorem ellipse_condition (k : ℝ) :
  is_non_degenerate_ellipse k ↔ k > -221 :=
sorry

end NUMINAMATH_CALUDE_ellipse_condition_l1291_129119


namespace NUMINAMATH_CALUDE_village_population_l1291_129190

/-- Represents the vampire population dynamics in a village --/
structure VampireVillage where
  initialVampires : ℕ
  initialPopulation : ℕ
  vampiresPerNight : ℕ
  nightsPassed : ℕ
  finalVampires : ℕ

/-- Theorem stating the initial population of the village --/
theorem village_population (v : VampireVillage) 
  (h1 : v.initialVampires = 2)
  (h2 : v.vampiresPerNight = 5)
  (h3 : v.nightsPassed = 2)
  (h4 : v.finalVampires = 72) :
  v.initialPopulation = 72 := by
  sorry

end NUMINAMATH_CALUDE_village_population_l1291_129190


namespace NUMINAMATH_CALUDE_article_cost_l1291_129131

/-- 
Given an article with two selling prices and a relationship between the gains,
prove that the cost of the article is 60.
-/
theorem article_cost (selling_price_1 selling_price_2 : ℝ) 
  (h1 : selling_price_1 = 360)
  (h2 : selling_price_2 = 340)
  (h3 : selling_price_1 - selling_price_2 = 0.05 * (selling_price_2 - cost)) :
  cost = 60 := by
  sorry

end NUMINAMATH_CALUDE_article_cost_l1291_129131


namespace NUMINAMATH_CALUDE_distance_travelled_l1291_129179

theorem distance_travelled (initial_speed : ℝ) (faster_speed : ℝ) (additional_distance : ℝ) :
  initial_speed = 12 →
  faster_speed = 18 →
  additional_distance = 30 →
  ∃ (actual_distance : ℝ),
    actual_distance / initial_speed = (actual_distance + additional_distance) / faster_speed ∧
    actual_distance = 60 := by
  sorry

end NUMINAMATH_CALUDE_distance_travelled_l1291_129179


namespace NUMINAMATH_CALUDE_factor_theorem_l1291_129189

def Q (d : ℝ) (x : ℝ) : ℝ := x^3 + 3*x^2 + d*x + 20

theorem factor_theorem (d : ℝ) : (∀ x, (x - 4) ∣ Q d x) → d = -33 := by
  sorry

end NUMINAMATH_CALUDE_factor_theorem_l1291_129189


namespace NUMINAMATH_CALUDE_difference_of_squares_l1291_129137

theorem difference_of_squares (x y : ℝ) : x^2 - 4*y^2 = (x - 2*y) * (x + 2*y) := by
  sorry

end NUMINAMATH_CALUDE_difference_of_squares_l1291_129137


namespace NUMINAMATH_CALUDE_cubic_roots_divisibility_l1291_129181

theorem cubic_roots_divisibility (p a b c : ℤ) (hp : Prime p) 
  (ha : p ∣ a) (hb : p ∣ b) (hc : p ∣ c)
  (hroots : ∃ (r s : ℤ), r ≠ s ∧ r^3 + a*r^2 + b*r + c = 0 ∧ s^3 + a*s^2 + b*s + c = 0) :
  p^3 ∣ c := by
sorry

end NUMINAMATH_CALUDE_cubic_roots_divisibility_l1291_129181


namespace NUMINAMATH_CALUDE_smallest_divisible_by_1_to_12_l1291_129196

theorem smallest_divisible_by_1_to_12 : ∃ n : ℕ, n > 0 ∧ (∀ k : ℕ, 1 ≤ k ∧ k ≤ 12 → k ∣ n) ∧ (∀ m : ℕ, m > 0 ∧ (∀ k : ℕ, 1 ≤ k ∧ k ≤ 12 → k ∣ m) → n ≤ m) ∧ n = 27720 := by
  sorry

end NUMINAMATH_CALUDE_smallest_divisible_by_1_to_12_l1291_129196


namespace NUMINAMATH_CALUDE_inequality_solution_l1291_129171

theorem inequality_solution (x : ℝ) : 
  (x / (x + 1) + (x - 3) / (2 * x) ≥ 4) ↔ (x ∈ Set.Icc (-3) (-1/5)) :=
sorry

end NUMINAMATH_CALUDE_inequality_solution_l1291_129171


namespace NUMINAMATH_CALUDE_max_distance_A_B_l1291_129113

def set_A : Set ℂ := {z : ℂ | z^4 - 16 = 0}
def set_B : Set ℂ := {z : ℂ | z^3 - 12*z^2 + 36*z - 64 = 0}

theorem max_distance_A_B : 
  ∃ (a : ℂ) (b : ℂ), a ∈ set_A ∧ b ∈ set_B ∧ 
    Complex.abs (a - b) = 10 ∧
    ∀ (x : ℂ) (y : ℂ), x ∈ set_A → y ∈ set_B → Complex.abs (x - y) ≤ 10 :=
sorry

end NUMINAMATH_CALUDE_max_distance_A_B_l1291_129113


namespace NUMINAMATH_CALUDE_sum_of_coordinates_l1291_129107

/-- Given a point C with coordinates (3, k), its reflection D over the y-axis
    with y-coordinate increased by 4, prove that the sum of all coordinates
    of C and D is 2k + 4. -/
theorem sum_of_coordinates (k : ℝ) : 
  let C : ℝ × ℝ := (3, k)
  let D : ℝ × ℝ := (-3, k + 4)
  (C.1 + C.2 + D.1 + D.2) = 2 * k + 4 :=
by sorry

end NUMINAMATH_CALUDE_sum_of_coordinates_l1291_129107


namespace NUMINAMATH_CALUDE_field_trip_adults_l1291_129168

theorem field_trip_adults (van_capacity : ℕ) (num_students : ℕ) (num_vans : ℕ) :
  van_capacity = 5 →
  num_students = 25 →
  num_vans = 6 →
  ∃ (num_adults : ℕ), num_adults = num_vans * van_capacity - num_students ∧ num_adults = 5 :=
by sorry

end NUMINAMATH_CALUDE_field_trip_adults_l1291_129168


namespace NUMINAMATH_CALUDE_park_area_l1291_129162

/-- Proves that a rectangular park with sides in ratio 3:2 and fencing cost of 125 at 50 ps per meter has an area of 3750 square meters -/
theorem park_area (length width : ℝ) (h1 : length / width = 3 / 2) 
  (h2 : 2 * (length + width) * 0.5 = 125) : length * width = 3750 := by
  sorry

end NUMINAMATH_CALUDE_park_area_l1291_129162


namespace NUMINAMATH_CALUDE_base2_10101010_equals_base4_2212_l1291_129139

def base2_to_decimal (b : List Bool) : ℕ :=
  b.foldl (fun acc x => 2 * acc + if x then 1 else 0) 0

def decimal_to_base4 (n : ℕ) : List (Fin 4) :=
  if n = 0 then [0] else
  let rec aux (m : ℕ) : List (Fin 4) :=
    if m = 0 then [] else (m % 4) :: aux (m / 4)
  aux n |>.reverse

theorem base2_10101010_equals_base4_2212 :
  decimal_to_base4 (base2_to_decimal [true, false, true, false, true, false, true, false]) =
  [2, 2, 1, 2] := by sorry

end NUMINAMATH_CALUDE_base2_10101010_equals_base4_2212_l1291_129139


namespace NUMINAMATH_CALUDE_salesman_profit_salesman_profit_is_442_l1291_129102

/-- Calculates the profit of a salesman selling backpacks --/
theorem salesman_profit (total_backpacks : ℕ) (total_cost : ℕ) 
  (first_sale_quantity : ℕ) (first_sale_price : ℕ)
  (second_sale_quantity : ℕ) (second_sale_price : ℕ)
  (remaining_price : ℕ) : ℕ :=
  let remaining_quantity := total_backpacks - first_sale_quantity - second_sale_quantity
  let total_revenue := 
    first_sale_quantity * first_sale_price +
    second_sale_quantity * second_sale_price +
    remaining_quantity * remaining_price
  total_revenue - total_cost

/-- The salesman's profit is $442 --/
theorem salesman_profit_is_442 : 
  salesman_profit 48 576 17 18 10 25 22 = 442 := by
  sorry

end NUMINAMATH_CALUDE_salesman_profit_salesman_profit_is_442_l1291_129102


namespace NUMINAMATH_CALUDE_quadratic_function_proof_l1291_129136

theorem quadratic_function_proof :
  let f : ℝ → ℝ := λ x => x^2 - 2*x - 3
  ∀ a b c : ℝ, a ≠ 0 →
  (∀ x : ℝ, f x = a * x^2 + b * x + c) →
  f (-2) = 5 ∧ f (-1) = 0 ∧ f 0 = -3 ∧ f 1 = -4 ∧ f 2 = -3 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_function_proof_l1291_129136


namespace NUMINAMATH_CALUDE_digit_product_is_30_l1291_129110

/-- Represents a 3x3 grid of digits -/
def Grid := Fin 3 → Fin 3 → Fin 9

/-- Check if all digits from 1 to 9 are used exactly once in the grid -/
def allDigitsUsedOnce (g : Grid) : Prop := ∀ d : Fin 9, ∃! (i j : Fin 3), g i j = d

/-- Product of digits in a row -/
def rowProduct (g : Grid) (row : Fin 3) : ℕ := (g row 0).val.succ * (g row 1).val.succ * (g row 2).val.succ

/-- Product of digits in a column -/
def colProduct (g : Grid) (col : Fin 3) : ℕ := (g 0 col).val.succ * (g 1 col).val.succ * (g 2 col).val.succ

/-- Product of digits in the shaded cells (top-left, center, bottom-right) -/
def shadedProduct (g : Grid) : ℕ := (g 0 0).val.succ * (g 1 1).val.succ * (g 2 2).val.succ

theorem digit_product_is_30 (g : Grid) 
  (h1 : allDigitsUsedOnce g)
  (h2 : rowProduct g 0 = 12)
  (h3 : rowProduct g 1 = 112)
  (h4 : colProduct g 0 = 216)
  (h5 : colProduct g 1 = 12) :
  shadedProduct g = 30 := by
  sorry

end NUMINAMATH_CALUDE_digit_product_is_30_l1291_129110


namespace NUMINAMATH_CALUDE_tangent_line_determines_b_l1291_129117

/-- A curve of the form y = x³ + ax + b -/
def curve (a b : ℝ) (x : ℝ) : ℝ := x^3 + a*x + b

/-- The derivative of the curve -/
def curve_derivative (a : ℝ) (x : ℝ) : ℝ := 3*x^2 + a

theorem tangent_line_determines_b (a b : ℝ) :
  curve a b 1 = 3 →
  curve_derivative a 1 = 2 →
  b = 3 := by
  sorry

#check tangent_line_determines_b

end NUMINAMATH_CALUDE_tangent_line_determines_b_l1291_129117


namespace NUMINAMATH_CALUDE_lemonade_glasses_l1291_129156

/-- The number of glasses of lemonade that can be made -/
def glasses_of_lemonade (total_lemons : ℕ) (lemons_per_glass : ℕ) : ℕ :=
  total_lemons / lemons_per_glass

/-- Theorem: Given 18 lemons and 2 lemons required per glass, 9 glasses of lemonade can be made -/
theorem lemonade_glasses : glasses_of_lemonade 18 2 = 9 := by
  sorry

end NUMINAMATH_CALUDE_lemonade_glasses_l1291_129156


namespace NUMINAMATH_CALUDE_ratio_to_nine_l1291_129178

theorem ratio_to_nine (x : ℝ) : (x / 9 = 5 / 1) → x = 45 := by
  sorry

end NUMINAMATH_CALUDE_ratio_to_nine_l1291_129178


namespace NUMINAMATH_CALUDE_identity_implies_a_minus_b_equals_one_l1291_129109

theorem identity_implies_a_minus_b_equals_one :
  ∀ (a b : ℚ),
  (∀ (y : ℚ), y > 0 → a / (y - 3) + b / (y + 5) = (3 * y + 7) / ((y - 3) * (y + 5))) →
  a - b = 1 := by
sorry

end NUMINAMATH_CALUDE_identity_implies_a_minus_b_equals_one_l1291_129109


namespace NUMINAMATH_CALUDE_p_investment_calculation_l1291_129195

def investment_ratio (p_investment q_investment : ℚ) : ℚ := p_investment / q_investment

theorem p_investment_calculation (q_investment : ℚ) (profit_ratio : ℚ) :
  q_investment = 30000 →
  profit_ratio = 3 / 5 →
  ∃ p_investment : ℚ, 
    investment_ratio p_investment q_investment = profit_ratio ∧
    p_investment = 18000 := by
  sorry

end NUMINAMATH_CALUDE_p_investment_calculation_l1291_129195


namespace NUMINAMATH_CALUDE_hexagon_areas_equal_l1291_129132

/-- Given a triangle T with sides of lengths r, g, and b, and area S,
    the area of both hexagons formed by extending the sides of T is equal to
    S * (4 + ((r^2 + g^2 + b^2)(r + g + b)) / (r * g * b)) -/
theorem hexagon_areas_equal (r g b S : ℝ) (hr : r > 0) (hg : g > 0) (hb : b > 0) (hS : S > 0) :
  let hexagon_area := S * (4 + ((r^2 + g^2 + b^2) * (r + g + b)) / (r * g * b))
  ∀ (area1 area2 : ℝ), area1 = hexagon_area ∧ area2 = hexagon_area → area1 = area2 := by
  sorry

end NUMINAMATH_CALUDE_hexagon_areas_equal_l1291_129132


namespace NUMINAMATH_CALUDE_shaded_area_semicircles_pattern_l1291_129152

/-- The area of the shaded region in a 1-foot length of alternating semicircles pattern --/
theorem shaded_area_semicircles_pattern (foot_to_inch : ℝ) (diameter : ℝ) (π : ℝ) : 
  foot_to_inch = 12 →
  diameter = 2 →
  (foot_to_inch / diameter) * (π * (diameter / 2)^2) = 6 * π := by
  sorry

end NUMINAMATH_CALUDE_shaded_area_semicircles_pattern_l1291_129152


namespace NUMINAMATH_CALUDE_teacher_assignment_theorem_l1291_129186

def number_of_teachers : ℕ := 4
def number_of_classes : ℕ := 3

-- Define a function that calculates the number of ways to assign teachers to classes
def ways_to_assign_teachers (teachers : ℕ) (classes : ℕ) : ℕ :=
  sorry -- The actual calculation goes here

theorem teacher_assignment_theorem :
  ways_to_assign_teachers number_of_teachers number_of_classes = 36 :=
by sorry

end NUMINAMATH_CALUDE_teacher_assignment_theorem_l1291_129186


namespace NUMINAMATH_CALUDE_f_sum_logs_l1291_129128

-- Define the function f
def f (x : ℝ) : ℝ := 1 + x^3

-- State the theorem
theorem f_sum_logs : f (Real.log 2) + f (Real.log (1/2)) = 2 := by
  sorry

end NUMINAMATH_CALUDE_f_sum_logs_l1291_129128


namespace NUMINAMATH_CALUDE_tangent_slope_point_A_l1291_129188

-- Define the function f(x)
def f (x : ℝ) : ℝ := x^2 + 3*x

-- Define the derivative of f(x)
def f_derivative (x : ℝ) : ℝ := 2*x + 3

-- Theorem statement
theorem tangent_slope_point_A :
  ∃ (x y : ℝ), 
    f_derivative x = 7 ∧ 
    f x = y ∧ 
    x = 2 ∧ 
    y = 10 := by sorry

end NUMINAMATH_CALUDE_tangent_slope_point_A_l1291_129188


namespace NUMINAMATH_CALUDE_sum_of_squares_problem_l1291_129163

theorem sum_of_squares_problem (a b c : ℝ) 
  (h_nonneg : a ≥ 0 ∧ b ≥ 0 ∧ c ≥ 0) 
  (h_sum_squares : a^2 + b^2 + c^2 = 48) 
  (h_sum_products : a*b + b*c + c*a = 26) : 
  a + b + c = 10 := by sorry

end NUMINAMATH_CALUDE_sum_of_squares_problem_l1291_129163


namespace NUMINAMATH_CALUDE_surface_area_difference_l1291_129144

/-- Calculates the difference between the sum of surface areas of smaller cubes
    and the surface area of a larger cube containing them. -/
theorem surface_area_difference (larger_volume : ℝ) (num_smaller_cubes : ℕ) (smaller_volume : ℝ) :
  larger_volume = 64 →
  num_smaller_cubes = 64 →
  smaller_volume = 1 →
  (num_smaller_cubes : ℝ) * (6 * smaller_volume ^ (2/3)) - 6 * larger_volume ^ (2/3) = 288 := by
  sorry

end NUMINAMATH_CALUDE_surface_area_difference_l1291_129144


namespace NUMINAMATH_CALUDE_speaker_sale_profit_l1291_129149

theorem speaker_sale_profit (selling_price : ℝ) 
  (profit_percentage : ℝ) (loss_percentage : ℝ) : 
  selling_price = 1.44 →
  profit_percentage = 0.2 →
  loss_percentage = 0.1 →
  let cost_price_1 := selling_price / (1 + profit_percentage)
  let cost_price_2 := selling_price / (1 - loss_percentage)
  let total_cost := cost_price_1 + cost_price_2
  let total_revenue := 2 * selling_price
  total_revenue - total_cost = 0.08 := by
sorry

end NUMINAMATH_CALUDE_speaker_sale_profit_l1291_129149


namespace NUMINAMATH_CALUDE_least_sum_of_equal_multiples_l1291_129177

theorem least_sum_of_equal_multiples (x y z : ℕ+) (h : (2 : ℕ) * x.val = (5 : ℕ) * y.val ∧ (5 : ℕ) * y.val = (6 : ℕ) * z.val) :
  ∃ (a b c : ℕ+), (2 : ℕ) * a.val = (5 : ℕ) * b.val ∧ (5 : ℕ) * b.val = (6 : ℕ) * c.val ∧
  (∀ (p q r : ℕ+), (2 : ℕ) * p.val = (5 : ℕ) * q.val ∧ (5 : ℕ) * q.val = (6 : ℕ) * r.val →
    a.val + b.val + c.val ≤ p.val + q.val + r.val) ∧
  a.val + b.val + c.val = 26 :=
sorry

end NUMINAMATH_CALUDE_least_sum_of_equal_multiples_l1291_129177


namespace NUMINAMATH_CALUDE_twenty_percent_greater_than_52_l1291_129134

theorem twenty_percent_greater_than_52 (x : ℝ) : x = 52 * (1 + 0.2) → x = 62.4 := by
  sorry

end NUMINAMATH_CALUDE_twenty_percent_greater_than_52_l1291_129134


namespace NUMINAMATH_CALUDE_flour_weight_qualified_l1291_129172

def nominal_weight : ℝ := 25
def tolerance : ℝ := 0.25
def flour_weight : ℝ := 24.80

theorem flour_weight_qualified :
  flour_weight ≥ nominal_weight - tolerance ∧
  flour_weight ≤ nominal_weight + tolerance := by
  sorry

end NUMINAMATH_CALUDE_flour_weight_qualified_l1291_129172


namespace NUMINAMATH_CALUDE_marias_age_l1291_129192

/-- 
Given that Jose is 12 years older than Maria and the sum of their ages is 40,
prove that Maria is 14 years old.
-/
theorem marias_age (maria jose : ℕ) 
  (h1 : jose = maria + 12) 
  (h2 : maria + jose = 40) : 
  maria = 14 := by
  sorry

end NUMINAMATH_CALUDE_marias_age_l1291_129192


namespace NUMINAMATH_CALUDE_apple_picking_ratio_l1291_129153

theorem apple_picking_ratio : 
  ∀ (frank_apples susan_apples : ℕ) (x : ℚ),
    frank_apples = 36 →
    susan_apples = frank_apples * x →
    (susan_apples / 2 + frank_apples * 2 / 3 : ℚ) = 78 →
    x = 3 := by
  sorry

end NUMINAMATH_CALUDE_apple_picking_ratio_l1291_129153


namespace NUMINAMATH_CALUDE_infinite_geometric_series_first_term_l1291_129130

theorem infinite_geometric_series_first_term
  (r : ℚ) (S : ℚ) (h1 : r = 1 / 8)
  (h2 : S = 60)
  (h3 : S = a / (1 - r)) :
  a = 105 / 2 :=
by sorry

end NUMINAMATH_CALUDE_infinite_geometric_series_first_term_l1291_129130


namespace NUMINAMATH_CALUDE_function_property_l1291_129170

theorem function_property (f : ℤ → ℤ) :
  (∀ a b c : ℤ, a + b + c = 0 → f a + f b + f c = a^2 + b^2 + c^2) →
  ∃ c : ℤ, ∀ x : ℤ, f x = x^2 + c * x :=
by sorry

end NUMINAMATH_CALUDE_function_property_l1291_129170


namespace NUMINAMATH_CALUDE_line_y_axis_intersection_l1291_129180

/-- A line passing through two points intersects the y-axis at a specific point -/
theorem line_y_axis_intersection (x₁ y₁ x₂ y₂ : ℝ) :
  x₁ = 3 →
  y₁ = 20 →
  x₂ = -7 →
  y₂ = 2 →
  ∃ y : ℝ, y = 14.6 ∧ (y - y₁) / (0 - x₁) = (y₂ - y₁) / (x₂ - x₁) :=
by sorry

end NUMINAMATH_CALUDE_line_y_axis_intersection_l1291_129180


namespace NUMINAMATH_CALUDE_percentage_of_juniors_l1291_129147

theorem percentage_of_juniors (total : ℕ) (seniors : ℕ) :
  total = 800 →
  seniors = 160 →
  let sophomores := (total : ℚ) * (1 / 4)
  let freshmen := sophomores + 16
  let juniors := total - (freshmen + sophomores + seniors)
  (juniors / total) * 100 = 28 := by
  sorry

end NUMINAMATH_CALUDE_percentage_of_juniors_l1291_129147


namespace NUMINAMATH_CALUDE_function_inequality_iff_a_geq_half_l1291_129173

/-- Given a function f(x) = ln x - a(x - 1), where a is a real number and x ≥ 1,
    prove that f(x) ≤ (ln x) / (x + 1) if and only if a ≥ 1/2 -/
theorem function_inequality_iff_a_geq_half (a : ℝ) :
  (∀ x : ℝ, x ≥ 1 → (Real.log x - a * (x - 1)) ≤ (Real.log x) / (x + 1)) ↔ a ≥ 1/2 := by
  sorry

end NUMINAMATH_CALUDE_function_inequality_iff_a_geq_half_l1291_129173


namespace NUMINAMATH_CALUDE_pigeons_eating_breadcrumbs_l1291_129127

theorem pigeons_eating_breadcrumbs (initial_pigeons : ℕ) (new_pigeons : ℕ) : 
  initial_pigeons = 1 → new_pigeons = 1 → initial_pigeons + new_pigeons = 2 := by
  sorry

end NUMINAMATH_CALUDE_pigeons_eating_breadcrumbs_l1291_129127


namespace NUMINAMATH_CALUDE_zero_not_identity_for_star_l1291_129157

-- Define the set S
def S : Set ℝ := {x : ℝ | x ≠ -1/3}

-- Define the * operation
def star (a b : ℝ) : ℝ := 3 * a * b + 1

-- Theorem statement
theorem zero_not_identity_for_star :
  ¬(∀ a ∈ S, (star 0 a = a ∧ star a 0 = a)) :=
sorry

end NUMINAMATH_CALUDE_zero_not_identity_for_star_l1291_129157

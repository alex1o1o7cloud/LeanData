import Mathlib

namespace value_of_x_when_sqrt_fraction_is_zero_l1890_189089

theorem value_of_x_when_sqrt_fraction_is_zero :
  ∀ x : ℝ, x ≠ 0 → (Real.sqrt (2 - x)) / x = 0 → x = 2 := by
  sorry

end value_of_x_when_sqrt_fraction_is_zero_l1890_189089


namespace large_power_of_two_appears_early_l1890_189049

/-- Represents the state of cards on the table at any given time -/
structure CardState where
  totalCards : Nat
  oddCards : Nat
  maxPowerOfTwo : Nat

/-- The initial state of cards -/
def initialState : CardState :=
  { totalCards := 100, oddCards := 43, maxPowerOfTwo := 0 }

/-- Function to calculate the next state after one minute -/
def nextState (state : CardState) : CardState :=
  { totalCards := state.totalCards + 1,
    oddCards := if state.oddCards = 43 then 44 else 44,
    maxPowerOfTwo := state.maxPowerOfTwo + 1 }

/-- Function to calculate the state after n minutes -/
def stateAfterMinutes (n : Nat) : CardState :=
  match n with
  | 0 => initialState
  | n + 1 => nextState (stateAfterMinutes n)

theorem large_power_of_two_appears_early (n : Nat) :
  (stateAfterMinutes n).maxPowerOfTwo ≥ 10000 →
  (stateAfterMinutes 1440).maxPowerOfTwo ≥ 10000 :=
by
  sorry

#check large_power_of_two_appears_early

end large_power_of_two_appears_early_l1890_189049


namespace parallel_line_slope_l1890_189084

/-- Given a line with equation 3x - 6y = 21, prove that any parallel line has slope 1/2 -/
theorem parallel_line_slope (x y : ℝ) :
  (3 * x - 6 * y = 21) → 
  (∃ (m b : ℝ), y = m * x + b ∧ m = (1 : ℝ) / 2) :=
by sorry

end parallel_line_slope_l1890_189084


namespace max_reflections_theorem_l1890_189053

/-- The angle between two intersecting lines in degrees -/
def angle_between_lines : ℝ := 10

/-- The maximum number of reflections before hitting perpendicularly -/
def max_reflections : ℕ := 18

/-- Theorem stating the maximum number of reflections -/
theorem max_reflections_theorem : 
  ∀ (n : ℕ), n * angle_between_lines ≤ 180 → n ≤ max_reflections :=
sorry

end max_reflections_theorem_l1890_189053


namespace all_same_number_probability_l1890_189009

/-- The number of faces on each die -/
def num_faces : ℕ := 6

/-- The number of dice thrown -/
def num_dice : ℕ := 5

/-- The total number of possible outcomes when throwing the dice -/
def total_outcomes : ℕ := num_faces ^ num_dice

/-- The number of favorable outcomes (all dice showing the same number) -/
def favorable_outcomes : ℕ := num_faces

/-- The probability of all dice showing the same number -/
def probability : ℚ := favorable_outcomes / total_outcomes

theorem all_same_number_probability :
  probability = 1 / 1296 := by sorry

end all_same_number_probability_l1890_189009


namespace solution_set_of_fraction_inequality_range_of_a_for_empty_solution_set_l1890_189094

-- Problem 1
theorem solution_set_of_fraction_inequality (x : ℝ) :
  (x - 3) / (x + 7) < 0 ↔ -7 < x ∧ x < 3 := by sorry

-- Problem 2
theorem range_of_a_for_empty_solution_set (a : ℝ) :
  (∀ x : ℝ, x^2 - 4*a*x + 4*a^2 + a > 0) → a > 0 := by sorry

end solution_set_of_fraction_inequality_range_of_a_for_empty_solution_set_l1890_189094


namespace quadratic_equation_coefficients_l1890_189036

theorem quadratic_equation_coefficients :
  ∀ (a b c : ℝ),
  (∀ x, 3 * x = x^2 - 2) →
  (∀ x, a * x^2 + b * x + c = 0) →
  (a = 1 ∧ b = -3 ∧ c = -2) :=
by sorry

end quadratic_equation_coefficients_l1890_189036


namespace divisibility_of_fifth_power_differences_l1890_189012

theorem divisibility_of_fifth_power_differences (x y z : ℤ) 
  (hxy : x ≠ y) (hyz : y ≠ z) (hzx : z ≠ x) : 
  ∃ k : ℤ, (x - y)^5 + (y - z)^5 + (z - x)^5 = k * (5 * (x - y) * (y - z) * (z - x)) := by
  sorry

end divisibility_of_fifth_power_differences_l1890_189012


namespace student_pairs_l1890_189062

theorem student_pairs (n : ℕ) (same_letter_pairs : ℕ) (total_pairs : ℕ) :
  n = 12 →
  same_letter_pairs = 3 →
  total_pairs = n.choose 2 →
  total_pairs - same_letter_pairs = 63 := by
  sorry

end student_pairs_l1890_189062


namespace certain_number_proof_l1890_189099

theorem certain_number_proof (y : ℕ) : (2^14) - (2^12) = 3 * (2^y) → y = 12 := by
  sorry

end certain_number_proof_l1890_189099


namespace two_sin_sixty_degrees_l1890_189023

theorem two_sin_sixty_degrees : 2 * Real.sin (π / 3) = Real.sqrt 3 := by
  sorry

end two_sin_sixty_degrees_l1890_189023


namespace sushi_father_lollipops_l1890_189077

/-- The number of lollipops Sushi's father bought -/
def initial_lollipops : ℕ := 12

/-- The number of lollipops eaten -/
def eaten_lollipops : ℕ := 5

/-- The number of lollipops left -/
def remaining_lollipops : ℕ := 7

theorem sushi_father_lollipops : 
  initial_lollipops = eaten_lollipops + remaining_lollipops :=
by sorry

end sushi_father_lollipops_l1890_189077


namespace arithmetic_sequence_problem_l1890_189011

-- Define an arithmetic sequence
def arithmetic_sequence (a : ℕ → ℝ) (d : ℝ) : Prop :=
  ∀ n, a (n + 1) = a n + d

-- Main theorem
theorem arithmetic_sequence_problem
  (a : ℕ → ℝ) (d m : ℝ) (h_d : d ≠ 0)
  (h_sum : a 3 + a 6 + a 10 + a 13 = 32)
  (h_seq : arithmetic_sequence a d)
  (h_m : ∃ m : ℕ, a m = 8) :
  ∃ m : ℕ, m = 8 ∧ a m = 8 :=
sorry

end arithmetic_sequence_problem_l1890_189011


namespace identity_proof_l1890_189026

theorem identity_proof (a b c x : ℝ) (hab : a ≠ b) (hac : a ≠ c) (hbc : b ≠ c) :
  (a^2 * (x - b) * (x - c)) / ((a - b) * (a - c)) +
  (b^2 * (x - a) * (x - c)) / ((b - a) * (b - c)) +
  (c^2 * (x - a) * (x - b)) / ((c - a) * (c - b)) = x^2 := by
  sorry

end identity_proof_l1890_189026


namespace polar_to_cartesian_circle_l1890_189061

theorem polar_to_cartesian_circle (x y ρ θ : ℝ) :
  (ρ = 4 * Real.sin θ) ∧ (x = ρ * Real.cos θ) ∧ (y = ρ * Real.sin θ) →
  x^2 + (y - 2)^2 = 4 :=
by sorry

end polar_to_cartesian_circle_l1890_189061


namespace seed_germination_percentage_l1890_189067

theorem seed_germination_percentage (seeds_plot1 seeds_plot2 : ℕ) 
  (germination_rate1 germination_rate2 : ℚ) :
  seeds_plot1 = 300 →
  seeds_plot2 = 200 →
  germination_rate1 = 20 / 100 →
  germination_rate2 = 35 / 100 →
  let total_seeds := seeds_plot1 + seeds_plot2
  let germinated_seeds1 := (seeds_plot1 : ℚ) * germination_rate1
  let germinated_seeds2 := (seeds_plot2 : ℚ) * germination_rate2
  let total_germinated := germinated_seeds1 + germinated_seeds2
  total_germinated / total_seeds = 26 / 100 := by
sorry

end seed_germination_percentage_l1890_189067


namespace f_composition_equals_pi_plus_one_l1890_189021

noncomputable def f (x : ℝ) : ℝ :=
  if x > 0 then x + 1
  else if x = 0 then Real.pi
  else 0

theorem f_composition_equals_pi_plus_one :
  f (f (f (-2))) = Real.pi + 1 := by sorry

end f_composition_equals_pi_plus_one_l1890_189021


namespace parallel_vectors_x_value_l1890_189096

/-- Two vectors in ℝ² are parallel if their cross product is zero -/
def parallel (a b : ℝ × ℝ) : Prop :=
  a.1 * b.2 - a.2 * b.1 = 0

theorem parallel_vectors_x_value :
  ∀ x : ℝ, parallel (1, 2) (x, -2) → x = -1 := by
  sorry

end parallel_vectors_x_value_l1890_189096


namespace least_integral_b_value_l1890_189016

theorem least_integral_b_value : 
  (∃ b : ℤ, (∀ x y : ℝ, (x^2 + y^2)^2 ≤ b * (x^4 + y^4)) ∧ 
   (∀ b' : ℤ, b' < b → ∃ x y : ℝ, (x^2 + y^2)^2 > b' * (x^4 + y^4))) → 
  (∃ b : ℤ, b = 2 ∧ 
   (∀ x y : ℝ, (x^2 + y^2)^2 ≤ b * (x^4 + y^4)) ∧ 
   (∀ b' : ℤ, b' < b → ∃ x y : ℝ, (x^2 + y^2)^2 > b' * (x^4 + y^4))) :=
by sorry

end least_integral_b_value_l1890_189016


namespace expression_simplification_l1890_189037

theorem expression_simplification (a : ℝ) (h : a = Real.sqrt 2 + 1) :
  (3 / (a - 1) + (a - 3) / (a^2 - 1)) / (a / (a + 1)) = 2 * Real.sqrt 2 := by
  sorry

end expression_simplification_l1890_189037


namespace sqrt_inequality_reciprocal_sum_inequality_l1890_189033

-- Part 1
theorem sqrt_inequality (b : ℝ) (h : b ≥ 2) :
  Real.sqrt (b + 1) - Real.sqrt b < Real.sqrt (b - 1) - Real.sqrt (b - 2) :=
sorry

-- Part 2
theorem reciprocal_sum_inequality (a b : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : a ≠ b) (h4 : a + b = 2) :
  1 / a + 1 / b > 2 :=
sorry

end sqrt_inequality_reciprocal_sum_inequality_l1890_189033


namespace fifteen_points_densified_thrice_equals_113_original_points_must_be_fifteen_l1890_189082

/-- Calculates the number of points after one densification -/
def densify (n : ℕ) : ℕ := 2 * n - 1

/-- Represents the process of densification repeated k times -/
def densify_k_times (n : ℕ) (k : ℕ) : ℕ :=
  match k with
  | 0 => n
  | k + 1 => densify (densify_k_times n k)

/-- The theorem stating that 3 densifications of 15 points results in 113 points -/
theorem fifteen_points_densified_thrice_equals_113 :
  densify_k_times 15 3 = 113 :=
by sorry

/-- The main theorem proving that starting with 15 points and applying 3 densifications
    is the only way to end up with 113 points -/
theorem original_points_must_be_fifteen (n : ℕ) :
  densify_k_times n 3 = 113 → n = 15 :=
by sorry

end fifteen_points_densified_thrice_equals_113_original_points_must_be_fifteen_l1890_189082


namespace family_chips_consumption_l1890_189017

/-- Calculates the number of chocolate chips each family member eats given the following conditions:
  - Each batch contains 12 cookies
  - The family has 4 total people
  - Kendra made three batches
  - Each cookie contains 2 chocolate chips
  - All family members get the same number of cookies
-/
def chips_per_person (cookies_per_batch : ℕ) (family_size : ℕ) (batches : ℕ) (chips_per_cookie : ℕ) : ℕ :=
  let total_cookies := cookies_per_batch * batches
  let cookies_per_person := total_cookies / family_size
  cookies_per_person * chips_per_cookie

/-- Proves that given the conditions in the problem, each family member eats 18 chocolate chips -/
theorem family_chips_consumption :
  chips_per_person 12 4 3 2 = 18 := by
  sorry

end family_chips_consumption_l1890_189017


namespace geometric_sequence_sum_l1890_189001

/-- A geometric sequence with positive terms -/
def GeometricSequence (a : ℕ → ℝ) : Prop :=
  ∃ (r : ℝ), r ≠ 0 ∧ ∀ n, a (n + 1) = a n * r ∧ a n > 0

theorem geometric_sequence_sum (a : ℕ → ℝ) :
  GeometricSequence a →
  a 2 * a 4 + 2 * a 3 * a 5 + a 4 * a 6 = 36 →
  a 3 + a 5 = 6 :=
by sorry

end geometric_sequence_sum_l1890_189001


namespace f_2015_value_l1890_189029

def is_odd_function (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = -f x

theorem f_2015_value (f : ℝ → ℝ) 
  (h_odd : is_odd_function f)
  (h_period : ∀ x, f (x + 2) = -f x)
  (h_01 : ∀ x, x ∈ Set.Icc 0 1 → f x = 3^x - 1) :
  f 2015 = -2 := by
sorry

end f_2015_value_l1890_189029


namespace complex_cube_root_of_unity_l1890_189069

theorem complex_cube_root_of_unity : (1/2 - Complex.I * (Real.sqrt 3)/2)^3 = -1 := by
  sorry

end complex_cube_root_of_unity_l1890_189069


namespace power_function_through_point_l1890_189048

-- Define a power function
def isPowerFunction (f : ℝ → ℝ) : Prop :=
  ∃ α : ℝ, ∀ x : ℝ, f x = x ^ α

-- Theorem statement
theorem power_function_through_point (f : ℝ → ℝ) 
  (h1 : isPowerFunction f) 
  (h2 : f 2 = Real.sqrt 2 / 2) : 
  f 4 = 1 / 2 := by
sorry

end power_function_through_point_l1890_189048


namespace angle_sum_ninety_degrees_l1890_189079

theorem angle_sum_ninety_degrees (α β : Real) 
  (acute_α : 0 < α ∧ α < Real.pi / 2)
  (acute_β : 0 < β ∧ β < Real.pi / 2)
  (eq1 : 3 * Real.sin α ^ 2 + 2 * Real.sin β ^ 2 = 1)
  (eq2 : 3 * Real.sin (2 * α) - 2 * Real.sin (2 * β) = 0) :
  α + 2 * β = Real.pi / 2 := by
sorry

end angle_sum_ninety_degrees_l1890_189079


namespace symmetric_arrangement_exists_l1890_189078

/-- Represents a grid figure -/
structure GridFigure where
  -- Add necessary fields to represent a grid figure
  asymmetric : Bool

/-- Represents an arrangement of grid figures -/
structure Arrangement where
  figures : List GridFigure
  symmetric : Bool

/-- Given three identical asymmetric grid figures, 
    there exists a symmetric arrangement -/
theorem symmetric_arrangement_exists : 
  ∀ (f : GridFigure), 
    f.asymmetric → 
    ∃ (a : Arrangement), 
      a.figures.length = 3 ∧ 
      (∀ fig ∈ a.figures, fig = f) ∧ 
      a.symmetric :=
by
  sorry


end symmetric_arrangement_exists_l1890_189078


namespace smallest_number_with_given_remainders_l1890_189024

theorem smallest_number_with_given_remainders : ∃ (b : ℕ), b > 0 ∧
  b % 4 = 2 ∧ b % 3 = 2 ∧ b % 5 = 3 ∧
  ∀ (n : ℕ), n > 0 ∧ n % 4 = 2 ∧ n % 3 = 2 ∧ n % 5 = 3 → b ≤ n :=
by
  use 38
  sorry

end smallest_number_with_given_remainders_l1890_189024


namespace busy_schedule_starts_26th_l1890_189002

/-- Represents the reading schedule for September --/
structure ReadingSchedule where
  total_pages : ℕ
  total_days : ℕ
  busy_days : ℕ
  special_day : ℕ
  special_day_pages : ℕ
  daily_pages : ℕ

/-- Calculates the start day of the busy schedule --/
def busy_schedule_start (schedule : ReadingSchedule) : ℕ :=
  schedule.total_days - 
  ((schedule.total_pages - schedule.special_day_pages) / schedule.daily_pages) - 
  1

/-- Theorem stating that the busy schedule starts on the 26th --/
theorem busy_schedule_starts_26th (schedule : ReadingSchedule) 
  (h1 : schedule.total_pages = 600)
  (h2 : schedule.total_days = 30)
  (h3 : schedule.busy_days = 4)
  (h4 : schedule.special_day = 23)
  (h5 : schedule.special_day_pages = 100)
  (h6 : schedule.daily_pages = 20) :
  busy_schedule_start schedule = 26 := by
  sorry

#eval busy_schedule_start {
  total_pages := 600,
  total_days := 30,
  busy_days := 4,
  special_day := 23,
  special_day_pages := 100,
  daily_pages := 20
}

end busy_schedule_starts_26th_l1890_189002


namespace shoebox_height_l1890_189019

/-- The height of a rectangular shoebox given specific conditions -/
theorem shoebox_height (width : ℝ) (block_side : ℝ) (uncovered_area : ℝ)
  (h_width : width = 6)
  (h_block : block_side = 4)
  (h_uncovered : uncovered_area = 8)
  : width * (block_side * block_side + uncovered_area) / width = 4 := by
  sorry

end shoebox_height_l1890_189019


namespace binomial_1294_2_l1890_189014

theorem binomial_1294_2 : Nat.choose 1294 2 = 836161 := by sorry

end binomial_1294_2_l1890_189014


namespace problem_solution_l1890_189071

-- Define the function f(x)
def f (a : ℝ) (x : ℝ) : ℝ := |x - a| + 3 * x

-- Theorem statement
theorem problem_solution (a : ℝ) (h : a > 0) :
  -- Part I
  (∀ x : ℝ, f 1 x ≥ 3 * x + 2 ↔ x ≥ 3 ∨ x ≤ -1) ∧
  -- Part II
  ((∀ x : ℝ, f a x ≤ 0 ↔ x ≤ -1) → a = 2) :=
by sorry

end problem_solution_l1890_189071


namespace other_root_of_quadratic_l1890_189035

theorem other_root_of_quadratic (m : ℝ) : 
  (2^2 - 2 + m = 0) → ((-1)^2 - (-1) + m = 0) := by sorry

end other_root_of_quadratic_l1890_189035


namespace decimal_point_shift_l1890_189098

theorem decimal_point_shift (x : ℝ) : 10 * x = x + 2.7 → x = 0.3 := by
  sorry

end decimal_point_shift_l1890_189098


namespace alice_has_winning_strategy_l1890_189063

/-- Represents a game on a complete graph -/
structure GraphGame where
  n : ℕ  -- number of vertices
  m : ℕ  -- maximum number of edges Bob can direct per turn

/-- Represents a strategy for Alice -/
def Strategy := GraphGame → Bool

/-- Checks if a strategy is winning for Alice -/
def is_winning_strategy (s : Strategy) (g : GraphGame) : Prop :=
  ∀ (bob_moves : ℕ → Fin g.m), ∃ (cycle : List (Fin g.n)), 
    cycle.length > 0 ∧ 
    cycle.Nodup ∧
    (∀ (i : Fin cycle.length), 
      ∃ (edge_directed_by_alice : Bool), 
        edge_directed_by_alice = true)

/-- The main theorem stating that Alice has a winning strategy -/
theorem alice_has_winning_strategy : 
  ∃ (s : Strategy), is_winning_strategy s ⟨2014, 1000⟩ := by
  sorry


end alice_has_winning_strategy_l1890_189063


namespace greatest_b_for_no_negative_seven_in_range_l1890_189034

theorem greatest_b_for_no_negative_seven_in_range : 
  ∃ (b : ℤ), b = 10 ∧ 
  (∀ (x : ℝ), x^2 + (b : ℝ) * x + 20 ≠ -7) ∧
  (∀ (b' : ℤ), b' > b → ∃ (x : ℝ), x^2 + (b' : ℝ) * x + 20 = -7) :=
by sorry

end greatest_b_for_no_negative_seven_in_range_l1890_189034


namespace power_of_power_l1890_189032

theorem power_of_power (a : ℝ) : (a^3)^4 = a^12 := by
  sorry

end power_of_power_l1890_189032


namespace cone_lateral_area_l1890_189013

/-- Given a cone with a central angle of 120° in its unfolded diagram and a base circle radius of 2 cm,
    prove that its lateral area is 12π cm². -/
theorem cone_lateral_area (central_angle : Real) (base_radius : Real) (lateral_area : Real) :
  central_angle = 120 * (π / 180) →
  base_radius = 2 →
  lateral_area = 12 * π →
  lateral_area = (1 / 2) * (2 * π * base_radius) * ((2 * π * base_radius) / (2 * π * (central_angle / (2 * π)))) :=
by sorry


end cone_lateral_area_l1890_189013


namespace z_in_fourth_quadrant_l1890_189054

/-- The complex number z defined as (2-i)^2 -/
def z : ℂ := (2 - Complex.I) ^ 2

/-- Theorem stating that z lies in the fourth quadrant of the complex plane -/
theorem z_in_fourth_quadrant : 
  z.re > 0 ∧ z.im < 0 := by sorry

end z_in_fourth_quadrant_l1890_189054


namespace nested_fraction_equality_l1890_189015

theorem nested_fraction_equality : 
  (1 : ℚ) / (3 - 1 / (3 - 1 / (3 - 1 / (3 - 1 / 3)))) = 21 / 55 := by
  sorry

end nested_fraction_equality_l1890_189015


namespace geometric_sequence_ratio_l1890_189007

/-- An infinite geometric sequence where any term is equal to the sum of all terms following it has a common ratio of 1/2 -/
theorem geometric_sequence_ratio (a : ℝ) (q : ℝ) (h : a ≠ 0) :
  (∀ n : ℕ, a * q^n = ∑' k, a * q^(n + k + 1)) → q = 1/2 := by
  sorry

end geometric_sequence_ratio_l1890_189007


namespace complex_magnitude_equation_l1890_189055

theorem complex_magnitude_equation (x : ℝ) : 
  (x > 0 ∧ Complex.abs (-3 + x * Complex.I) = 5 * Real.sqrt 5) → x = 2 * Real.sqrt 29 := by
  sorry

end complex_magnitude_equation_l1890_189055


namespace parallel_lines_imply_a_equals_7_l1890_189043

/-- A line in the 2D plane represented by ax + by + c = 0 -/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Two lines are parallel if their slopes are equal -/
def parallel (l1 l2 : Line) : Prop :=
  l1.a * l2.b = l2.a * l1.b

theorem parallel_lines_imply_a_equals_7 :
  let l1 : Line := { a := 2, b := 1, c := -1 }
  let l2 : Line := { a := a - 1, b := 3, c := -2 }
  parallel l1 l2 → a = 7 := by
  sorry

end parallel_lines_imply_a_equals_7_l1890_189043


namespace ball_exchange_game_theorem_l1890_189072

/-- Represents a game played by n girls exchanging balls. -/
def BallExchangeGame (n : ℕ) := Unit

/-- A game is nice if at the end nobody has her own ball. -/
def is_nice (game : BallExchangeGame n) : Prop := sorry

/-- A game is tiresome if at the end everybody has her initial ball. -/
def is_tiresome (game : BallExchangeGame n) : Prop := sorry

/-- There exists a nice game for n players. -/
def exists_nice_game (n : ℕ) : Prop :=
  ∃ (game : BallExchangeGame n), is_nice game

/-- There exists a tiresome game for n players. -/
def exists_tiresome_game (n : ℕ) : Prop :=
  ∃ (game : BallExchangeGame n), is_tiresome game

theorem ball_exchange_game_theorem (n : ℕ) (h : n ≥ 2) :
  (exists_nice_game n ↔ n ≠ 3) ∧
  (exists_tiresome_game n ↔ n % 4 = 0 ∨ n % 4 = 1) :=
sorry

end ball_exchange_game_theorem_l1890_189072


namespace replaced_person_weight_l1890_189028

/-- Given a group of 8 people, if replacing one person with a new person weighing 89 kg
    increases the average weight by 3 kg, then the weight of the replaced person is 65 kg. -/
theorem replaced_person_weight
  (n : ℕ)
  (new_weight : ℝ)
  (avg_increase : ℝ)
  (h1 : n = 8)
  (h2 : new_weight = 89)
  (h3 : avg_increase = 3)
  : ∃ (old_weight : ℝ), old_weight = new_weight - n * avg_increase :=
by
  sorry

end replaced_person_weight_l1890_189028


namespace sameTotalHeadsProbability_eq_565_2048_l1890_189066

/-- Represents the probability distribution of flipping four coins, 
    where three are fair and one has 5/8 probability of heads -/
def coinFlipDistribution : List ℚ :=
  [3/64, 14/64, 24/64, 18/64, 5/64]

/-- The probability of two people getting the same number of heads 
    when each flips four coins (three fair, one biased) -/
def sameTotalHeadsProbability : ℚ :=
  (coinFlipDistribution.map (λ x => x^2)).sum

theorem sameTotalHeadsProbability_eq_565_2048 :
  sameTotalHeadsProbability = 565/2048 := by
  sorry

end sameTotalHeadsProbability_eq_565_2048_l1890_189066


namespace dollar_four_negative_one_l1890_189018

-- Define the $ operation
def dollar (a b : ℤ) : ℤ := a * (b + 1) + a * b

-- Theorem statement
theorem dollar_four_negative_one :
  dollar 4 (-1) = -4 := by
  sorry

end dollar_four_negative_one_l1890_189018


namespace expense_representation_l1890_189052

-- Define income and expense as real numbers
def income : ℝ := 5
def expense : ℝ := 5

-- Define the representation of income
def income_representation : ℝ := 5

-- Theorem to prove
theorem expense_representation : 
  income_representation = income → -expense = -5 :=
by
  sorry

end expense_representation_l1890_189052


namespace min_value_theorem_l1890_189031

theorem min_value_theorem (x y : ℝ) (hx : x > 0) (hy : y > 0) (h_eq : 2 * x + y = 2) :
  ∃ (min_val : ℝ), min_val = 9/4 ∧ ∀ (z : ℝ), z = 2/(x + 1) + 1/y → z ≥ min_val :=
by sorry

end min_value_theorem_l1890_189031


namespace four_math_six_english_arrangements_l1890_189051

/-- The number of ways to arrange books and a trophy on a shelf -/
def shelfArrangements (mathBooks : ℕ) (englishBooks : ℕ) : ℕ :=
  2 * 2 * (Nat.factorial mathBooks) * (Nat.factorial englishBooks)

/-- Theorem stating the number of arrangements for 4 math books and 6 English books -/
theorem four_math_six_english_arrangements :
  shelfArrangements 4 6 = 69120 := by
  sorry

#eval shelfArrangements 4 6

end four_math_six_english_arrangements_l1890_189051


namespace total_rewards_distributed_l1890_189058

theorem total_rewards_distributed (students_A students_B students_C : ℕ)
  (rewards_per_student_A rewards_per_student_B rewards_per_student_C : ℕ) :
  students_A = students_B + 4 →
  students_B = students_C + 4 →
  rewards_per_student_A + 3 = rewards_per_student_B →
  rewards_per_student_B + 5 = rewards_per_student_C →
  students_A * rewards_per_student_A = students_B * rewards_per_student_B + 3 →
  students_B * rewards_per_student_B = students_C * rewards_per_student_C + 5 →
  students_A * rewards_per_student_A +
  students_B * rewards_per_student_B +
  students_C * rewards_per_student_C = 673 :=
by sorry

end total_rewards_distributed_l1890_189058


namespace intersection_points_theorem_l1890_189056

-- Define the curve C
def curve_C (x y : ℝ) : Prop := (x - 1)^2 + y^2 = 1

-- Define the line L
def line_L (x y m : ℝ) : Prop := x - Real.sqrt 2 * y - m = 0

-- Define the point P
def point_P (m : ℝ) : ℝ × ℝ := (m, 0)

-- Define the condition for intersection points A and B
def intersection_condition (m : ℝ) : Prop :=
  ∃ (x₁ y₁ x₂ y₂ : ℝ),
    curve_C x₁ y₁ ∧ curve_C x₂ y₂ ∧
    line_L x₁ y₁ m ∧ line_L x₂ y₂ m ∧
    ((x₁ - m)^2 + y₁^2) * ((x₂ - m)^2 + y₂^2) = 1

-- Theorem statement
theorem intersection_points_theorem :
  ∀ m : ℝ, intersection_condition m ↔ m = 1 + Real.sqrt 7 / 2 ∨ m = 1 - Real.sqrt 7 / 2 :=
sorry

end intersection_points_theorem_l1890_189056


namespace smallest_constant_inequality_l1890_189050

theorem smallest_constant_inequality (x y : ℝ) :
  (∀ D : ℝ, (∀ x y : ℝ, x^4 + y^4 + 1 ≥ D * (x^2 + y^2)) → D ≤ Real.sqrt 2) ∧
  (∀ x y : ℝ, x^4 + y^4 + 1 ≥ Real.sqrt 2 * (x^2 + y^2)) :=
by sorry

end smallest_constant_inequality_l1890_189050


namespace impossible_2018_after_2019_l1890_189081

/-- Represents a single step in the room occupancy change --/
inductive Step
  | Enter : Step  -- Two people enter (+2)
  | Exit : Step   -- One person exits (-1)

/-- Calculates the change in room occupancy for a given step --/
def stepChange (s : Step) : Int :=
  match s with
  | Step.Enter => 2
  | Step.Exit => -1

/-- Represents a sequence of steps over time --/
def Sequence := List Step

/-- Calculates the final room occupancy given a sequence of steps --/
def finalOccupancy (seq : Sequence) : Int :=
  seq.foldl (fun acc s => acc + stepChange s) 0

/-- Theorem: It's impossible to have 2018 people after 2019 steps --/
theorem impossible_2018_after_2019 :
  ∀ (seq : Sequence), seq.length = 2019 → finalOccupancy seq ≠ 2018 :=
by
  sorry


end impossible_2018_after_2019_l1890_189081


namespace original_statement_converse_not_always_true_inverse_not_always_true_neither_converse_nor_inverse_always_true_l1890_189010

-- Define the properties
def is_rectangle (q : Quadrilateral) : Prop := sorry
def has_opposite_sides_equal (q : Quadrilateral) : Prop := sorry

-- Define the original statement
theorem original_statement : 
  ∀ q : Quadrilateral, is_rectangle q → has_opposite_sides_equal q := sorry

-- Prove that the converse is not always true
theorem converse_not_always_true : 
  ¬(∀ q : Quadrilateral, has_opposite_sides_equal q → is_rectangle q) := sorry

-- Prove that the inverse is not always true
theorem inverse_not_always_true : 
  ¬(∀ q : Quadrilateral, ¬is_rectangle q → ¬has_opposite_sides_equal q) := sorry

-- Combine the results
theorem neither_converse_nor_inverse_always_true : 
  (¬(∀ q : Quadrilateral, has_opposite_sides_equal q → is_rectangle q)) ∧
  (¬(∀ q : Quadrilateral, ¬is_rectangle q → ¬has_opposite_sides_equal q)) := sorry

end original_statement_converse_not_always_true_inverse_not_always_true_neither_converse_nor_inverse_always_true_l1890_189010


namespace part1_part2_l1890_189022

/-- Definition of the function f -/
def f (a : ℝ) (x : ℝ) : ℝ := -3 * x^2 + a * (6 - a) * x + 6

/-- Theorem for part 1 -/
theorem part1 (a : ℝ) : 
  f a 1 > 0 ↔ (3 - 2 * Real.sqrt 3 < a ∧ a < 3 + 2 * Real.sqrt 3) := by sorry

/-- Theorem for part 2 -/
theorem part2 (a b : ℝ) : 
  (∀ x, f a x > b ↔ -1 < x ∧ x < 3) → 
  ((a = 3 - Real.sqrt 3 ∨ a = 3 + Real.sqrt 3) ∧ b = -3) := by sorry

end part1_part2_l1890_189022


namespace alternating_pair_sum_50_eq_2550_l1890_189000

def alternatingPairSum (n : Nat) : Int :=
  let f (k : Nat) : Int :=
    if k % 4 ≤ 1 then (n - k + 1)^2 else -(n - k + 1)^2
  (List.range n).map f |>.sum

theorem alternating_pair_sum_50_eq_2550 :
  alternatingPairSum 50 = 2550 := by
  sorry

end alternating_pair_sum_50_eq_2550_l1890_189000


namespace max_guests_correct_l1890_189020

/-- The maximum number of guests that can dine at a restaurant with n choices
    for each of starters, main dishes, desserts, and wines, such that:
    1) No two guests have the same order
    2) There is no collection of n guests whose orders coincide in three aspects
       but differ in the fourth -/
def max_guests (n : ℕ+) : ℕ :=
  if n = 1 then 1 else n^4 - n^3

theorem max_guests_correct (n : ℕ+) :
  (max_guests n = 1 ∧ n = 1) ∨
  (max_guests n = n^4 - n^3 ∧ n ≥ 2) :=
sorry

end max_guests_correct_l1890_189020


namespace bug_path_theorem_l1890_189042

/-- Represents a rectangular floor with a broken tile -/
structure Floor :=
  (width : ℕ)
  (length : ℕ)
  (broken_tile : ℕ × ℕ)

/-- Calculates the number of tiles a bug visits when walking diagonally across the floor -/
def tiles_visited (f : Floor) : ℕ :=
  f.width + f.length - Nat.gcd f.width f.length

/-- Theorem: A bug walking diagonally across a 12x25 floor with a broken tile visits 36 tiles -/
theorem bug_path_theorem (f : Floor) 
    (h_width : f.width = 12)
    (h_length : f.length = 25)
    (h_broken : f.broken_tile = (12, 18)) : 
  tiles_visited f = 36 := by
  sorry

end bug_path_theorem_l1890_189042


namespace electric_power_is_4_l1890_189041

-- Define the constants and variables
variable (k_star : ℝ) (e_tau : ℝ) (a_star : ℝ) (N_H : ℝ) (N_e : ℝ)

-- Define the conditions
axiom k_star_def : k_star = 1/3
axiom e_tau_a_star_def : e_tau * a_star = 0.15
axiom N_H_def : N_H = 80

-- Define the electric power equation
def electric_power (k_star e_tau a_star N_H : ℝ) : ℝ :=
  k_star * e_tau * a_star * N_H

-- State the theorem
theorem electric_power_is_4 :
  electric_power k_star e_tau a_star N_H = 4 :=
sorry

end electric_power_is_4_l1890_189041


namespace parabola_point_focus_distance_l1890_189093

theorem parabola_point_focus_distance (p : ℝ) (y : ℝ) (h1 : p > 0) :
  y^2 = 2*p*8 ∧ (8 + p/2)^2 + y^2 = 10^2 → p = 4 ∧ (y = 8 ∨ y = -8) := by
  sorry

end parabola_point_focus_distance_l1890_189093


namespace quadratic_root_difference_l1890_189008

theorem quadratic_root_difference (p : ℝ) : 
  let a := 1
  let b := -(2*p + 1)
  let c := p^2 - 5
  let discriminant := b^2 - 4*a*c
  let root_difference := Real.sqrt discriminant / (2*a)
  root_difference = Real.sqrt (2*p^2 + 4*p + 11) := by
  sorry

end quadratic_root_difference_l1890_189008


namespace green_peaches_count_l1890_189044

/-- Given a basket of fruits with the following properties:
  * There are p total fruits
  * There are r red peaches
  * The rest are green peaches
  * The sum of red peaches and twice the green peaches is 3 more than the total fruits
  Then the number of green peaches is always 3 -/
theorem green_peaches_count (p r : ℕ) (h1 : p = r + (p - r)) 
    (h2 : r + 2 * (p - r) = p + 3) : p - r = 3 := by
  sorry

#check green_peaches_count

end green_peaches_count_l1890_189044


namespace courtyard_width_is_20_l1890_189064

/-- Represents a rectangular paving stone -/
structure PavingStone where
  length : ℝ
  width : ℝ

/-- Represents a rectangular courtyard -/
structure Courtyard where
  length : ℝ
  width : ℝ

/-- Calculates the area of a paving stone -/
def area_paving_stone (stone : PavingStone) : ℝ :=
  stone.length * stone.width

/-- Calculates the area of a courtyard -/
def area_courtyard (yard : Courtyard) : ℝ :=
  yard.length * yard.width

/-- Theorem: The width of the courtyard is 20 meters -/
theorem courtyard_width_is_20 (stone : PavingStone) (yard : Courtyard) 
    (h1 : stone.length = 4)
    (h2 : stone.width = 2)
    (h3 : yard.length = 40)
    (h4 : area_courtyard yard = 100 * area_paving_stone stone) :
    yard.width = 20 := by
  sorry

#check courtyard_width_is_20

end courtyard_width_is_20_l1890_189064


namespace y_equals_five_l1890_189004

/-- A line passing through the origin with slope 1/2 -/
def line_k (x y : ℝ) : Prop := y = (1/2) * x

/-- The theorem stating that under the given conditions, y must equal 5 -/
theorem y_equals_five (x y : ℝ) :
  line_k x 6 →
  line_k 10 y →
  x * y = 60 →
  y = 5 := by sorry

end y_equals_five_l1890_189004


namespace blue_marble_difference_l1890_189006

theorem blue_marble_difference (jar1_blue jar1_green jar2_blue jar2_green : ℕ) :
  jar1_blue + jar1_green = jar2_blue + jar2_green →
  jar1_blue = 9 * jar1_green →
  jar2_blue = 8 * jar2_green →
  jar1_green + jar2_green = 95 →
  jar1_blue - jar2_blue = 5 := by
sorry

end blue_marble_difference_l1890_189006


namespace lucilles_earnings_l1890_189092

/-- Represents the earnings in cents for each type of weed -/
structure WeedEarnings where
  small : ℕ
  medium : ℕ
  large : ℕ

/-- Represents the number of weeds in a garden area -/
structure WeedCount where
  small : ℕ
  medium : ℕ
  large : ℕ

/-- Calculates the total earnings from weeding a garden area -/
def calculateEarnings (earnings : WeedEarnings) (count : WeedCount) : ℕ :=
  earnings.small * count.small + earnings.medium * count.medium + earnings.large * count.large

/-- Represents Lucille's weeding earnings problem -/
structure LucillesProblem where
  earnings : WeedEarnings
  flowerBed : WeedCount
  vegetablePatch : WeedCount
  grass : WeedCount
  sodaCost : ℕ
  salesTaxPercent : ℕ

/-- Theorem stating that Lucille has 130 cents left after buying the soda -/
theorem lucilles_earnings (problem : LucillesProblem)
  (h1 : problem.earnings = ⟨4, 8, 12⟩)
  (h2 : problem.flowerBed = ⟨6, 3, 2⟩)
  (h3 : problem.vegetablePatch = ⟨10, 2, 2⟩)
  (h4 : problem.grass = ⟨20, 10, 2⟩)
  (h5 : problem.sodaCost = 99)
  (h6 : problem.salesTaxPercent = 15) :
  let totalEarnings := calculateEarnings problem.earnings problem.flowerBed +
                       calculateEarnings problem.earnings problem.vegetablePatch +
                       calculateEarnings problem.earnings ⟨problem.grass.small / 2, problem.grass.medium / 2, problem.grass.large / 2⟩
  let sodaTotalCost := problem.sodaCost + (problem.sodaCost * problem.salesTaxPercent / 100 + 1)
  totalEarnings - sodaTotalCost = 130 := by sorry


end lucilles_earnings_l1890_189092


namespace lisa_expenses_l1890_189065

theorem lisa_expenses (B : ℝ) (book coffee : ℝ) : 
  book = 0.3 * (B - 2 * coffee) →
  coffee = 0.1 * (B - book) →
  book + coffee = (31 : ℝ) / 94 * B := by
sorry

end lisa_expenses_l1890_189065


namespace bakers_total_cost_l1890_189040

/-- Calculates the total cost of baker's ingredients --/
theorem bakers_total_cost : 
  let flour_boxes := 3
  let flour_price := 3
  let egg_trays := 3
  let egg_price := 10
  let milk_liters := 7
  let milk_price := 5
  let soda_boxes := 2
  let soda_price := 3
  
  flour_boxes * flour_price + 
  egg_trays * egg_price + 
  milk_liters * milk_price + 
  soda_boxes * soda_price = 80 := by
  sorry

end bakers_total_cost_l1890_189040


namespace quadratic_roots_l1890_189074

theorem quadratic_roots (m : ℝ) : 
  (∃ x : ℝ, x^2 - 4*x + m = 0 ∧ x = -1) → 
  (∃ y : ℝ, y^2 - 4*y + m = 0 ∧ y = 5) :=
by sorry

end quadratic_roots_l1890_189074


namespace option2_saves_money_at_80_l1890_189088

/-- The total charge for Option 1 given x participants -/
def option1_charge (x : ℝ) : ℝ := 1500 + 320 * x

/-- The total charge for Option 2 given x participants -/
def option2_charge (x : ℝ) : ℝ := 360 * x - 1800

/-- The original price per person -/
def original_price : ℝ := 400

theorem option2_saves_money_at_80 :
  ∀ x : ℝ, x > 50 → option2_charge 80 < option1_charge 80 := by
  sorry

end option2_saves_money_at_80_l1890_189088


namespace sum_of_digits_of_difference_of_squares_l1890_189075

def a : ℕ := 6666666
def b : ℕ := 3333333

-- Function to calculate the sum of digits
def sum_of_digits (n : ℕ) : ℕ :=
  if n < 10 then n else (n % 10) + sum_of_digits (n / 10)

theorem sum_of_digits_of_difference_of_squares :
  sum_of_digits ((a ^ 2) - (b ^ 2)) = 63 := by
  sorry

end sum_of_digits_of_difference_of_squares_l1890_189075


namespace total_books_calculation_l1890_189027

/-- The number of boxes containing children's books. -/
def num_boxes : ℕ := 5

/-- The number of children's books in each box. -/
def books_per_box : ℕ := 20

/-- The total number of children's books in all boxes. -/
def total_books : ℕ := num_boxes * books_per_box

theorem total_books_calculation : total_books = 100 := by
  sorry

end total_books_calculation_l1890_189027


namespace dice_probability_l1890_189045

/-- The number of possible outcomes when rolling 7 six-sided dice -/
def total_outcomes : ℕ := 6^7

/-- The number of ways to choose 2 numbers from 6 -/
def choose_two_from_six : ℕ := Nat.choose 6 2

/-- The number of ways to arrange 2 pairs in 7 positions -/
def arrange_two_pairs : ℕ := Nat.choose 7 2 * Nat.choose 5 2

/-- The number of ways to arrange remaining dice for two pairs case -/
def arrange_remaining_two_pairs : ℕ := 4 * 3 * 2

/-- The number of ways to arrange triplet and pair -/
def arrange_triplet_pair : ℕ := Nat.choose 7 3 * Nat.choose 4 2

/-- The number of ways to arrange remaining dice for triplet and pair case -/
def arrange_remaining_triplet_pair : ℕ := 4 * 3

/-- The total number of favorable outcomes -/
def favorable_outcomes : ℕ :=
  (choose_two_from_six * arrange_two_pairs * arrange_remaining_two_pairs) +
  (2 * choose_two_from_six * arrange_triplet_pair * arrange_remaining_triplet_pair)

theorem dice_probability :
  (favorable_outcomes : ℚ) / total_outcomes = 525 / 972 := by sorry

end dice_probability_l1890_189045


namespace encircling_stripe_probability_theorem_l1890_189085

/-- Represents a cube with 6 faces -/
structure Cube :=
  (faces : Fin 6 → Bool)

/-- The probability of a stripe on a single face -/
def stripe_prob : ℚ := 2/3

/-- The probability of a dot on a single face -/
def dot_prob : ℚ := 1/3

/-- The number of valid stripe configurations that encircle the cube -/
def valid_configurations : ℕ := 12

/-- The probability of a continuous stripe encircling the cube -/
def encircling_stripe_probability : ℚ := 768/59049

/-- Theorem stating the probability of a continuous stripe encircling the cube -/
theorem encircling_stripe_probability_theorem :
  encircling_stripe_probability = 
    (stripe_prob ^ 6) * valid_configurations :=
by sorry

end encircling_stripe_probability_theorem_l1890_189085


namespace quadrilateral_symmetry_theorem_l1890_189038

/-- Represents a quadrilateral in 2D space -/
structure Quadrilateral where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ
  D : ℝ × ℝ

/-- Represents the operation of replacing a vertex with its symmetric point -/
def symmetricOperation (q : Quadrilateral) : Quadrilateral :=
  sorry

/-- Checks if a quadrilateral is permissible (sides are pairwise different and it remains convex) -/
def isPermissible (q : Quadrilateral) : Prop :=
  sorry

/-- Checks if a quadrilateral is inscribed in a circle -/
def isInscribed (q : Quadrilateral) : Prop :=
  sorry

/-- Checks if two quadrilaterals are equal -/
def areEqual (q1 q2 : Quadrilateral) : Prop :=
  sorry

/-- Main theorem statement -/
theorem quadrilateral_symmetry_theorem (q : Quadrilateral) 
  (h_permissible : isPermissible q) :
  (∃ (q_inscribed : Quadrilateral), isInscribed q_inscribed ∧ 
    isPermissible q_inscribed ∧ 
    areEqual (symmetricOperation (symmetricOperation (symmetricOperation q_inscribed))) q_inscribed) ∧
  (areEqual (symmetricOperation (symmetricOperation (symmetricOperation 
    (symmetricOperation (symmetricOperation (symmetricOperation q)))))) q) :=
  sorry


end quadrilateral_symmetry_theorem_l1890_189038


namespace smallest_three_digit_even_in_pascal_l1890_189057

/-- Pascal's triangle coefficient -/
def pascal (n k : ℕ) : ℕ := 
  Nat.choose n k

/-- Check if a number is in Pascal's triangle -/
def inPascalTriangle (m : ℕ) : Prop :=
  ∃ n k : ℕ, pascal n k = m

/-- The smallest three-digit even number in Pascal's triangle -/
def smallestThreeDigitEvenInPascal : ℕ := 120

theorem smallest_three_digit_even_in_pascal :
  (inPascalTriangle smallestThreeDigitEvenInPascal) ∧
  (smallestThreeDigitEvenInPascal % 2 = 0) ∧
  (smallestThreeDigitEvenInPascal ≥ 100) ∧
  (smallestThreeDigitEvenInPascal < 1000) ∧
  (∀ m : ℕ, m < smallestThreeDigitEvenInPascal →
    m % 2 = 0 → m ≥ 100 → m < 1000 → ¬(inPascalTriangle m)) := by
  sorry

#check smallest_three_digit_even_in_pascal

end smallest_three_digit_even_in_pascal_l1890_189057


namespace solution_set_l1890_189087

-- Define the function f and its derivative f'
variable (f : ℝ → ℝ) (f' : ℝ → ℝ)

-- Define the condition that f' is the derivative of f
def is_derivative (f : ℝ → ℝ) (f' : ℝ → ℝ) : Prop :=
  ∀ x, deriv f x = f' x

-- Define the condition that 2f'(x) > f(x) for all x
def condition (f : ℝ → ℝ) (f' : ℝ → ℝ) : Prop :=
  ∀ x, 2 * f' x > f x

-- Define the inequality we want to solve
def inequality (f : ℝ → ℝ) (x : ℝ) : Prop :=
  Real.exp ((x - 1) / 2) * f x < f (2 * x - 1)

-- State the theorem
theorem solution_set (f : ℝ → ℝ) (f' : ℝ → ℝ) :
  is_derivative f f' → condition f f' →
  (∀ x, inequality f x ↔ x > 1) :=
sorry

end solution_set_l1890_189087


namespace max_juggling_time_max_juggling_time_value_l1890_189003

/-- Represents the time in seconds before Bobo drops a cow when juggling n cows -/
def drop_time (n : ℕ) : ℕ :=
  match n with
  | 1 => 64
  | 2 => 55
  | 3 => 47
  | 4 => 40
  | 5 => 33
  | 6 => 27
  | 7 => 22
  | 8 => 18
  | 9 => 14
  | 10 => 13
  | 11 => 12
  | 12 => 11
  | 13 => 10
  | 14 => 9
  | 15 => 8
  | 16 => 7
  | 17 => 6
  | 18 => 5
  | 19 => 4
  | 20 => 3
  | 21 => 2
  | 22 => 1
  | _ => 0

/-- Calculates the total juggling time for n cows -/
def total_time (n : ℕ) : ℕ := n * drop_time n

/-- The maximum number of cows Bobo can juggle -/
def max_cows : ℕ := 22

/-- Theorem: The maximum total juggling time is achieved with 5 cows -/
theorem max_juggling_time :
  ∀ n : ℕ, n ≤ max_cows → total_time 5 ≥ total_time n :=
by
  sorry

/-- Corollary: The maximum total juggling time is 165 seconds -/
theorem max_juggling_time_value : total_time 5 = 165 :=
by
  sorry

end max_juggling_time_max_juggling_time_value_l1890_189003


namespace pizza_coworkers_l1890_189025

theorem pizza_coworkers (num_pizzas : ℕ) (slices_per_pizza : ℕ) (slices_per_person : ℕ) :
  num_pizzas = 3 →
  slices_per_pizza = 8 →
  slices_per_person = 2 →
  (num_pizzas * slices_per_pizza) / slices_per_person = 12 := by
  sorry

end pizza_coworkers_l1890_189025


namespace modified_code_system_distinct_symbols_l1890_189080

/-- The number of possible symbols (dot, dash, or blank) -/
def num_symbols : ℕ := 3

/-- The maximum length of a sequence -/
def max_length : ℕ := 3

/-- The number of distinct symbols for a given sequence length -/
def distinct_symbols (length : ℕ) : ℕ := num_symbols ^ length

/-- The total number of distinct symbols for sequences of length 1 to max_length -/
def total_distinct_symbols : ℕ :=
  (Finset.range max_length).sum (λ i => distinct_symbols (i + 1))

theorem modified_code_system_distinct_symbols :
  total_distinct_symbols = 39 := by
  sorry

end modified_code_system_distinct_symbols_l1890_189080


namespace roots_equation_s_value_l1890_189047

theorem roots_equation_s_value (n r : ℝ) (c d : ℝ) :
  c^2 - n*c + 3 = 0 →
  d^2 - n*d + 3 = 0 →
  (c + 1/d)^2 - r*(c + 1/d) + s = 0 →
  (d + 1/c)^2 - r*(d + 1/c) + s = 0 →
  s = 16/3 := by
sorry

end roots_equation_s_value_l1890_189047


namespace initial_birds_in_tree_l1890_189097

theorem initial_birds_in_tree (additional_birds : ℕ) (total_birds : ℕ) 
  (h1 : additional_birds = 38) 
  (h2 : total_birds = 217) : 
  total_birds - additional_birds = 179 := by
  sorry

end initial_birds_in_tree_l1890_189097


namespace star_value_l1890_189091

def star (a b : ℤ) : ℚ := 1 / a + 1 / b

theorem star_value (a b : ℤ) (h1 : a ≠ 0) (h2 : b ≠ 0) (h3 : a + b = 10) (h4 : a * b = 24) :
  star a b = 5 / 12 := by
  sorry

end star_value_l1890_189091


namespace fraction_not_whole_number_l1890_189090

theorem fraction_not_whole_number : 
  (∃ n : ℕ, 60 / 12 = n) ∧ 
  (∀ n : ℕ, 60 / 8 ≠ n) ∧ 
  (∃ n : ℕ, 60 / 5 = n) ∧ 
  (∃ n : ℕ, 60 / 4 = n) ∧ 
  (∃ n : ℕ, 60 / 3 = n) := by
  sorry

end fraction_not_whole_number_l1890_189090


namespace flower_vase_problem_l1890_189030

/-- Calculates the number of vases needed to hold flowers given the vase capacity and flower counts. -/
def vases_needed (vase_capacity : ℕ) (carnations : ℕ) (roses : ℕ) : ℕ :=
  (carnations + roses + vase_capacity - 1) / vase_capacity

/-- Proves that given 9 flowers per vase, 4 carnations, and 23 roses, 3 vases are needed. -/
theorem flower_vase_problem : vases_needed 9 4 23 = 3 := by
  sorry

#eval vases_needed 9 4 23

end flower_vase_problem_l1890_189030


namespace textbook_ratio_l1890_189070

theorem textbook_ratio (initial : ℚ) (remaining : ℚ) 
  (h1 : initial = 960)
  (h2 : remaining = 360)
  (h3 : ∃ textbook_cost : ℚ, initial - textbook_cost - (1/4) * (initial - textbook_cost) = remaining) :
  ∃ textbook_cost : ℚ, textbook_cost / initial = 1/2 := by
sorry

end textbook_ratio_l1890_189070


namespace symmetric_point_wrt_origin_l1890_189073

/-- Given a point P with coordinates (3, -4), this theorem proves that its symmetric point
    with respect to the origin has coordinates (-3, 4). -/
theorem symmetric_point_wrt_origin :
  let P : ℝ × ℝ := (3, -4)
  let symmetric_point := (-P.1, -P.2)
  symmetric_point = (-3, 4) := by
  sorry

end symmetric_point_wrt_origin_l1890_189073


namespace subsets_sum_to_negative_eight_l1890_189059

def S : Finset Int := {-6, -4, -2, -1, 1, 2, 3, 4, 6}

theorem subsets_sum_to_negative_eight :
  ∃! (subsets : Finset (Finset Int)),
    (∀ subset ∈ subsets, subset ⊆ S ∧ (subset.sum id = -8)) ∧
    subsets.card = 6 :=
by sorry

end subsets_sum_to_negative_eight_l1890_189059


namespace student_arrangement_theorem_l1890_189005

/-- The number of ways to arrange 6 students (3 male and 3 female) with 3 female students adjacent -/
def adjacent_arrangement (n : ℕ) : ℕ := n

/-- The number of ways to arrange 6 students (3 male and 3 female) with 3 female students not adjacent -/
def not_adjacent_arrangement (n : ℕ) : ℕ := n

/-- The number of ways to arrange 6 students (3 male and 3 female) with one specific male student not at the beginning or end -/
def specific_male_arrangement (n : ℕ) : ℕ := n

theorem student_arrangement_theorem :
  adjacent_arrangement 144 = 144 ∧
  not_adjacent_arrangement 144 = 144 ∧
  specific_male_arrangement 480 = 480 :=
by sorry

end student_arrangement_theorem_l1890_189005


namespace fraction_of_lunch_eaten_l1890_189060

def total_calories : ℕ := 40
def recommended_calories : ℕ := 25
def extra_calories : ℕ := 5

def actual_calories : ℕ := recommended_calories + extra_calories

theorem fraction_of_lunch_eaten :
  (actual_calories : ℚ) / total_calories = 3 / 4 := by
  sorry

end fraction_of_lunch_eaten_l1890_189060


namespace largest_number_l1890_189095

-- Define a function to convert a number from base n to decimal
def to_decimal (digits : List Nat) (base : Nat) : Nat :=
  digits.enum.foldl (fun acc (i, d) => acc + d * base ^ i) 0

-- Define the numbers in their respective bases
def num_A : Nat := to_decimal [5, 8] 9
def num_B : Nat := to_decimal [0, 0, 2] 6
def num_C : Nat := to_decimal [8, 6] 11
def num_D : Nat := 70

-- Theorem statement
theorem largest_number :
  num_A = max num_A (max num_B (max num_C num_D)) :=
by sorry

end largest_number_l1890_189095


namespace bounded_function_satisfying_equation_l1890_189076

def is_bounded (f : ℤ → ℤ) : Prop :=
  ∃ M : ℤ, ∀ n : ℤ, |f n| ≤ M

def satisfies_equation (f : ℤ → ℤ) : Prop :=
  ∀ n k : ℤ, f (n + k) + f (k - n) = 2 * f k * f n

def is_zero_function (f : ℤ → ℤ) : Prop :=
  ∀ n : ℤ, f n = 0

def is_one_function (f : ℤ → ℤ) : Prop :=
  ∀ n : ℤ, f n = 1

def is_alternating_function (f : ℤ → ℤ) : Prop :=
  ∀ n : ℤ, f n = if n % 2 = 0 then 1 else -1

theorem bounded_function_satisfying_equation (f : ℤ → ℤ) 
  (h_bounded : is_bounded f) (h_satisfies : satisfies_equation f) :
  is_zero_function f ∨ is_one_function f ∨ is_alternating_function f :=
sorry

end bounded_function_satisfying_equation_l1890_189076


namespace distinguishable_cube_colorings_eq_30240_l1890_189086

/-- The number of colors available to paint the cube. -/
def num_colors : ℕ := 10

/-- The number of faces on the cube. -/
def num_faces : ℕ := 6

/-- The number of rotational symmetries of a cube. -/
def cube_rotations : ℕ := 24

/-- Calculates the number of distinguishable ways to paint a cube. -/
def distinguishable_cube_colorings : ℕ :=
  (num_colors * (num_colors - 1) * (num_colors - 2) * (num_colors - 3) * 
   (num_colors - 4) * (num_colors - 5)) / cube_rotations

/-- Theorem stating that the number of distinguishable ways to paint the cube is 30240. -/
theorem distinguishable_cube_colorings_eq_30240 :
  distinguishable_cube_colorings = 30240 := by
  sorry

end distinguishable_cube_colorings_eq_30240_l1890_189086


namespace proposition_truth_l1890_189039

theorem proposition_truth : (∃ x₀ : ℝ, x₀ - 2 > 0) ∧ ¬(∀ x : ℝ, 2^x > x^2) := by
  sorry

end proposition_truth_l1890_189039


namespace age_of_30th_student_l1890_189046

theorem age_of_30th_student 
  (total_students : Nat)
  (avg_age_all : ℝ)
  (num_group1 : Nat) (avg_age_group1 : ℝ)
  (num_group2 : Nat) (avg_age_group2 : ℝ)
  (num_group3 : Nat) (avg_age_group3 : ℝ)
  (age_single_student : ℝ)
  (h1 : total_students = 30)
  (h2 : avg_age_all = 23.5)
  (h3 : num_group1 = 9)
  (h4 : avg_age_group1 = 21.3)
  (h5 : num_group2 = 12)
  (h6 : avg_age_group2 = 19.7)
  (h7 : num_group3 = 7)
  (h8 : avg_age_group3 = 24.2)
  (h9 : age_single_student = 35)
  (h10 : num_group1 + num_group2 + num_group3 + 1 + 1 = total_students) :
  total_students * avg_age_all - 
  (num_group1 * avg_age_group1 + num_group2 * avg_age_group2 + 
   num_group3 * avg_age_group3 + age_single_student) = 72.5 := by
  sorry

end age_of_30th_student_l1890_189046


namespace divisor_of_number_minus_one_l1890_189068

theorem divisor_of_number_minus_one (n : ℕ) (h : n = 5026) : 5 ∣ (n - 1) := by
  sorry

end divisor_of_number_minus_one_l1890_189068


namespace laurent_series_expansion_l1890_189083

/-- Laurent series expansion of f(z) = 2ia / (z^2 + a^2) in the region 0 < |z - ia| < a -/
theorem laurent_series_expansion
  (a : ℝ) (z : ℂ) (ha : a > 0) (hz : 0 < Complex.abs (z - Complex.I * a) ∧ Complex.abs (z - Complex.I * a) < a) :
  (2 * Complex.I * a) / (z^2 + a^2) =
    1 / (z - Complex.I * a) - ∑' k, (z - Complex.I * a)^k / (Complex.I * a)^(k + 1) :=
by sorry

end laurent_series_expansion_l1890_189083

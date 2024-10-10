import Mathlib

namespace line_not_parallel_in_plane_l4030_403052

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the relations
variable (contained_in : Line → Plane → Prop)
variable (not_parallel : Line → Plane → Prop)
variable (coplanar : Line → Line → Plane → Prop)
variable (not_parallel_lines : Line → Line → Prop)

-- State the theorem
theorem line_not_parallel_in_plane 
  (m n : Line) (α β : Plane) 
  (h1 : m ≠ n) 
  (h2 : α ≠ β) 
  (h3 : contained_in m α) 
  (h4 : not_parallel n α) 
  (h5 : coplanar m n β) : 
  not_parallel_lines m n :=
sorry

end line_not_parallel_in_plane_l4030_403052


namespace factor_of_polynomial_l4030_403029

theorem factor_of_polynomial (x : ℝ) : 
  ∃ (q : ℝ → ℝ), (x^4 + 4*x^2 + 4) = (x^2 + 2) * q x := by
  sorry

end factor_of_polynomial_l4030_403029


namespace geometric_sequence_sixth_term_l4030_403028

/-- Given a geometric sequence {aₙ} where a₁a₃ = a₄ = 4, prove that a₆ = 8 -/
theorem geometric_sequence_sixth_term (a : ℕ → ℝ) :
  (∃ q : ℝ, ∀ n : ℕ, a (n + 1) = a n * q) →  -- a is a geometric sequence
  a 1 * a 3 = 4 →  -- given condition
  a 4 = 4 →        -- given condition
  a 6 = 8 :=
by sorry

end geometric_sequence_sixth_term_l4030_403028


namespace smallest_n_for_integer_sqrt_12n_l4030_403045

theorem smallest_n_for_integer_sqrt_12n :
  ∀ n : ℕ+, (∃ k : ℕ, k^2 = 12*n) → (∀ m : ℕ+, m < n → ¬∃ j : ℕ, j^2 = 12*m) → n = 3 :=
by sorry

end smallest_n_for_integer_sqrt_12n_l4030_403045


namespace residue_mod_16_l4030_403068

theorem residue_mod_16 : 260 * 18 - 21 * 8 + 4 ≡ 4 [ZMOD 16] := by
  sorry

end residue_mod_16_l4030_403068


namespace f_properties_l4030_403020

noncomputable def f (x : ℝ) : ℝ := ((Real.sin x - Real.cos x) * Real.sin (2 * x)) / Real.sin x

def is_periodic (f : ℝ → ℝ) (T : ℝ) : Prop :=
  ∀ x, f (x + T) = f x

def is_even (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = f x

theorem f_properties :
  (∀ x : ℝ, f x ≠ 0 ↔ x ∉ {y | ∃ k : ℤ, y = k * Real.pi}) ∧
  (∃ T : ℝ, T > 0 ∧ is_periodic f T ∧ ∀ S, (S > 0 ∧ is_periodic f S) → T ≤ S) ∧
  (∀ x : ℝ, f x ≥ 0 ↔ ∃ k : ℤ, x ∈ Set.Icc (Real.pi / 4 + k * Real.pi) (Real.pi / 2 + k * Real.pi)) ∧
  (∃ m : ℝ, m > 0 ∧ is_even (fun x ↦ f (x + m)) ∧
    ∀ n : ℝ, (n > 0 ∧ is_even (fun x ↦ f (x + n))) → m ≤ n) :=
by sorry

end f_properties_l4030_403020


namespace block_tower_combinations_l4030_403062

theorem block_tower_combinations :
  let initial_blocks : ℕ := 35
  let final_blocks : ℕ := 65
  let additional_blocks : ℕ := final_blocks - initial_blocks
  ∃! n : ℕ, n = (additional_blocks + 1) ∧
    n = (Finset.filter (fun p : ℕ × ℕ => p.1 + p.2 = additional_blocks)
      (Finset.product (Finset.range (additional_blocks + 1)) (Finset.range (additional_blocks + 1)))).card :=
by sorry

end block_tower_combinations_l4030_403062


namespace residue_negative_437_mod_13_l4030_403064

theorem residue_negative_437_mod_13 :
  ∃ (k : ℤ), -437 = 13 * k + 5 ∧ (0 : ℤ) ≤ 5 ∧ 5 < 13 := by
  sorry

end residue_negative_437_mod_13_l4030_403064


namespace baker_remaining_pastries_l4030_403010

theorem baker_remaining_pastries (pastries_made pastries_sold : ℕ) 
  (h1 : pastries_made = 148)
  (h2 : pastries_sold = 103) :
  pastries_made - pastries_sold = 45 := by
  sorry

end baker_remaining_pastries_l4030_403010


namespace xy_max_value_l4030_403084

theorem xy_max_value (x y : ℝ) (hx : x > 0) (hy : y > 0) (hsum : x + 2*y = 2) :
  ∃ (m : ℝ), m = 1/2 ∧ ∀ z, z = x*y → z ≤ m :=
sorry

end xy_max_value_l4030_403084


namespace sqrt_difference_equality_l4030_403077

theorem sqrt_difference_equality (p q : ℝ) 
  (h1 : p > 0) (h2 : 0 ≤ q) (h3 : q ≤ 5 * p) : 
  Real.sqrt (10 * p + 2 * Real.sqrt (25 * p^2 - q^2)) - 
  Real.sqrt (10 * p - 2 * Real.sqrt (25 * p^2 - q^2)) = 
  2 * Real.sqrt (5 * p - q) := by
  sorry

end sqrt_difference_equality_l4030_403077


namespace point_movement_on_number_line_l4030_403054

theorem point_movement_on_number_line (A : ℝ) : 
  A + 7 - 4 = 0 → A = -3 := by
  sorry

end point_movement_on_number_line_l4030_403054


namespace triangle_angle_solution_l4030_403026

/-- Given a triangle with angles measuring 60°, (5x)°, and (3x)°, prove that x = 15 -/
theorem triangle_angle_solution (x : ℝ) : 
  (60 : ℝ) + 5*x + 3*x = 180 → x = 15 := by
  sorry

#check triangle_angle_solution

end triangle_angle_solution_l4030_403026


namespace two_digit_sum_doubled_l4030_403040

theorem two_digit_sum_doubled (J L M K : ℕ) 
  (h_digits : J < 10 ∧ L < 10 ∧ M < 10 ∧ K < 10)
  (h_sum : (10 * J + M) + (10 * L + K) = 79) :
  2 * ((10 * J + M) + (10 * L + K)) = 158 := by
sorry

end two_digit_sum_doubled_l4030_403040


namespace book_page_digits_l4030_403033

/-- The total number of digits used to number pages in a book -/
def totalDigits (n : ℕ) : ℕ :=
  (min n 9) + 
  (2 * (min n 99 - 9)) + 
  (3 * (n - min n 99))

/-- Theorem: The total number of digits used in numbering the pages of a book with 346 pages is 930 -/
theorem book_page_digits : totalDigits 346 = 930 := by
  sorry

end book_page_digits_l4030_403033


namespace three_digit_cube_ending_777_l4030_403014

theorem three_digit_cube_ending_777 :
  ∃! x : ℕ, 100 ≤ x ∧ x < 1000 ∧ x^3 % 1000 = 777 :=
by
  use 753
  sorry

end three_digit_cube_ending_777_l4030_403014


namespace discount_amount_l4030_403096

/-- The cost of a spiral notebook in dollars -/
def spiral_notebook_cost : ℕ := 15

/-- The cost of a personal planner in dollars -/
def personal_planner_cost : ℕ := 10

/-- The number of spiral notebooks purchased -/
def notebooks_purchased : ℕ := 4

/-- The number of personal planners purchased -/
def planners_purchased : ℕ := 8

/-- The total cost after discount in dollars -/
def discounted_total : ℕ := 112

/-- Theorem stating that the discount amount is $28 -/
theorem discount_amount : 
  (notebooks_purchased * spiral_notebook_cost + planners_purchased * personal_planner_cost) - discounted_total = 28 := by
  sorry

end discount_amount_l4030_403096


namespace broker_income_slump_l4030_403048

/-- 
Proves that if a broker's income remains unchanged when the commission rate 
increases from 4% to 5%, then the percentage slump in business is 20%.
-/
theorem broker_income_slump (X : ℝ) (Y : ℝ) (h : X > 0) :
  (0.04 * X = 0.05 * Y) →  -- Income remains unchanged
  (Y / X = 0.8)            -- Percentage slump in business is 20%
  := by sorry

end broker_income_slump_l4030_403048


namespace system_solution_is_solution_set_l4030_403091

def system_solution (x y z : ℝ) : Prop :=
  x + y + z = 6 ∧ x*y + y*z + z*x = 11 ∧ x*y*z = 6

def solution_set : Set (ℝ × ℝ × ℝ) :=
  {(1, 2, 3), (1, 3, 2), (2, 1, 3), (2, 3, 1), (3, 1, 2), (3, 2, 1)}

theorem system_solution_is_solution_set :
  ∀ x y z, system_solution x y z ↔ (x, y, z) ∈ solution_set :=
by sorry

end system_solution_is_solution_set_l4030_403091


namespace properties_of_negative_23_l4030_403038

theorem properties_of_negative_23 :
  let x : ℝ := -23
  (∃ y : ℝ, x + y = 0 ∧ y = 23) ∧
  (∃ z : ℝ, x * z = 1 ∧ z = -1/23) ∧
  (abs x = 23) := by
  sorry

end properties_of_negative_23_l4030_403038


namespace zero_is_global_minimum_l4030_403044

-- Define the function f(x) = (x - 1)e^(x - 1)
noncomputable def f (x : ℝ) : ℝ := (x - 1) * Real.exp (x - 1)

-- Theorem statement
theorem zero_is_global_minimum :
  ∀ x : ℝ, f 0 ≤ f x :=
by sorry

end zero_is_global_minimum_l4030_403044


namespace malcolm_lights_theorem_l4030_403083

/-- The number of white lights Malcolm had initially --/
def initial_white_lights : ℕ := 59

/-- The number of red lights Malcolm bought --/
def red_lights : ℕ := 12

/-- The number of blue lights Malcolm bought --/
def blue_lights : ℕ := red_lights * 3

/-- The number of green lights Malcolm bought --/
def green_lights : ℕ := 6

/-- The number of colored lights Malcolm still needs to buy --/
def remaining_lights : ℕ := 5

/-- Theorem stating that the initial number of white lights is equal to 
    the sum of all colored lights bought and still to be bought --/
theorem malcolm_lights_theorem : 
  initial_white_lights = red_lights + blue_lights + green_lights + remaining_lights :=
by sorry

end malcolm_lights_theorem_l4030_403083


namespace dehydrated_men_fraction_l4030_403058

theorem dehydrated_men_fraction (total_men : ℕ) (finished_men : ℕ) 
  (h1 : total_men = 80)
  (h2 : finished_men = 52)
  (h3 : (1 : ℚ) / 4 * total_men = total_men - (3 : ℚ) / 4 * total_men)
  (h4 : (2 : ℚ) / 3 * ((3 : ℚ) / 4 * total_men) = total_men - finished_men - ((1 : ℚ) / 4 * total_men)) :
  (total_men - finished_men - (1 : ℚ) / 4 * total_men) / ((2 : ℚ) / 3 * ((3 : ℚ) / 4 * total_men)) = (1 : ℚ) / 5 := by
  sorry

end dehydrated_men_fraction_l4030_403058


namespace runners_speed_ratio_l4030_403021

/-- Represents a runner on a circular track -/
structure Runner where
  speed : ℝ
  direction : Bool  -- True for clockwise, False for counterclockwise

/-- Represents the circular track -/
structure Track where
  circumference : ℝ

/-- Represents the state of the runners on the track -/
structure RunnerState where
  track : Track
  runner1 : Runner
  runner2 : Runner
  meetingPoints : Finset ℝ  -- Set of points where runners meet

/-- The theorem statement -/
theorem runners_speed_ratio 
  (state : RunnerState) 
  (h1 : state.runner1.direction ≠ state.runner2.direction)  -- Runners move in opposite directions
  (h2 : state.runner1.speed ≠ 0 ∧ state.runner2.speed ≠ 0)  -- Both runners have non-zero speed
  (h3 : state.meetingPoints.card = 3)  -- There are exactly three meeting points
  (h4 : ∀ p ∈ state.meetingPoints, p < state.track.circumference)  -- Meeting points are on the track
  : state.runner2.speed / state.runner1.speed = 2 := by
  sorry

end runners_speed_ratio_l4030_403021


namespace fish_sharing_l4030_403037

/-- Represents the number of fish caught by each cat -/
structure CatCatch where
  white : ℕ
  black : ℕ
  calico : ℕ

/-- Represents the money transactions for each cat -/
structure CatMoney where
  white : ℚ
  black : ℚ
  calico : ℚ

def totalFish (c : CatCatch) : ℕ := c.white + c.black + c.calico

def averageShare (c : CatCatch) : ℚ := (totalFish c : ℚ) / 3

theorem fish_sharing (c : CatCatch) (m : CatMoney) : 
  c.white = 5 → c.black = 3 → c.calico = 0 → m.calico = -4/5 →
  (averageShare c = 8/3) ∧ 
  (m.white = 7) ∧ 
  (m.black = 1) ∧
  (m.white + m.black + m.calico = 0) ∧
  ((totalFish c : ℚ) * (3 : ℚ) / (totalFish c : ℚ) = 3) := by
  sorry

#check fish_sharing

end fish_sharing_l4030_403037


namespace solution_set_part1_solution_set_part2_l4030_403055

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := |x - a| + x

-- Part 1
theorem solution_set_part1 :
  {x : ℝ | f 1 x ≥ x + 2} = {x : ℝ | x ≥ 3 ∨ x ≤ -1} :=
sorry

-- Part 2
theorem solution_set_part2 (a : ℝ) (h : a > 0) :
  ({x : ℝ | f a x ≤ 3*x} = {x : ℝ | x ≥ 2}) → a = 6 :=
sorry

end solution_set_part1_solution_set_part2_l4030_403055


namespace weight_measurements_l4030_403019

/-- The set of available weights in pounds -/
def weights : List ℕ := [1, 3, 9, 27]

/-- The maximum weight that can be weighed using the given weights -/
def max_weight : ℕ := 40

/-- The number of different weights that can be measured -/
def different_weights : ℕ := 40

/-- Theorem stating the maximum weight and number of different weights -/
theorem weight_measurements :
  (weights.sum = max_weight) ∧
  (∀ w : ℕ, w > 0 ∧ w ≤ max_weight → ∃ combination : List ℕ, combination.all (· ∈ weights) ∧ combination.sum = w) ∧
  (different_weights = max_weight) :=
sorry

end weight_measurements_l4030_403019


namespace train_passing_jogger_time_l4030_403094

/-- Time for a train to pass a jogger given their speeds and initial positions -/
theorem train_passing_jogger_time
  (jogger_speed : ℝ)
  (train_speed : ℝ)
  (train_length : ℝ)
  (initial_distance : ℝ)
  (h1 : jogger_speed = 12 * (5 / 18))  -- Convert 12 km/hr to m/s
  (h2 : train_speed = 60 * (5 / 18))   -- Convert 60 km/hr to m/s
  (h3 : train_length = 300)
  (h4 : initial_distance = 300) :
  (train_length + initial_distance) / (train_speed - jogger_speed) = 15 := by
  sorry

#eval Float.ofScientific 15 0 1  -- Output: 15.0

end train_passing_jogger_time_l4030_403094


namespace number_wall_solve_l4030_403056

/-- Represents a row in the Number Wall -/
structure NumberWallRow :=
  (left : ℤ) (middle_left : ℤ) (middle_right : ℤ) (right : ℤ)

/-- Defines the Number Wall structure and rules -/
def NumberWall (bottom : NumberWallRow) : Prop :=
  ∃ (second : NumberWallRow) (third : NumberWallRow) (top : ℤ),
    second.left = bottom.left + bottom.middle_left
    ∧ second.middle_left = bottom.middle_left + bottom.middle_right
    ∧ second.middle_right = bottom.middle_right + bottom.right
    ∧ third.left = second.left + second.middle_left
    ∧ third.right = second.middle_right + second.right
    ∧ top = third.left + third.right
    ∧ top = 36

/-- The main theorem to prove -/
theorem number_wall_solve :
  ∀ m : ℤ, NumberWall ⟨m, 6, 12, 10⟩ → m = -28 :=
by sorry

end number_wall_solve_l4030_403056


namespace quadratic_root_transformation_l4030_403063

theorem quadratic_root_transformation (a b c r s : ℝ) :
  (∀ x, a * x^2 + b * x + c = 0 ↔ x = r ∨ x = s) →
  (∀ x, x^2 - b * x + a * c = 0 ↔ x = a * r + b ∨ x = a * s + b) :=
by sorry

end quadratic_root_transformation_l4030_403063


namespace complement_of_A_in_U_l4030_403076

-- Define the universal set U
def U : Set ℝ := {x | x^2 ≤ 4}

-- Define set A
def A : Set ℝ := {x | |x - 1| ≤ 1}

-- Theorem statement
theorem complement_of_A_in_U :
  (U \ A) = {x : ℝ | -2 ≤ x ∧ x < 0} :=
sorry

end complement_of_A_in_U_l4030_403076


namespace imaginary_part_of_complex_fraction_l4030_403016

theorem imaginary_part_of_complex_fraction :
  let z : ℂ := 5 / (1 + 2 * I)
  Complex.im z = -2 := by sorry

end imaginary_part_of_complex_fraction_l4030_403016


namespace total_lives_game_lives_calculation_l4030_403024

theorem total_lives (initial_lives : ℕ) (extra_lives_level1 : ℕ) (extra_lives_level2 : ℕ) :
  initial_lives + extra_lives_level1 + extra_lives_level2 =
  initial_lives + extra_lives_level1 + extra_lives_level2 :=
by sorry

theorem game_lives_calculation :
  let initial_lives : ℕ := 2
  let extra_lives_level1 : ℕ := 6
  let extra_lives_level2 : ℕ := 11
  initial_lives + extra_lives_level1 + extra_lives_level2 = 19 :=
by sorry

end total_lives_game_lives_calculation_l4030_403024


namespace susan_remaining_distance_l4030_403007

/-- The total number of spaces on the board game --/
def total_spaces : ℕ := 72

/-- Susan's movements over 5 turns --/
def susan_movements : List ℤ := [12, -3, 0, 4, -3]

/-- The theorem stating the remaining distance Susan needs to move --/
theorem susan_remaining_distance :
  total_spaces - (susan_movements.sum) = 62 := by sorry

end susan_remaining_distance_l4030_403007


namespace fraction_to_decimal_l4030_403039

theorem fraction_to_decimal : (3 : ℚ) / 40 = 0.075 := by sorry

end fraction_to_decimal_l4030_403039


namespace smallest_k_for_monochromatic_rectangle_l4030_403000

/-- A chessboard coloring is a function that assigns a color to each square of the board. -/
def Coloring (n k : ℕ) := Fin (2 * n) → Fin k → Fin n

/-- Predicate that checks if there exist 2 columns and 2 rows with 4 squares of the same color at their intersections. -/
def HasMonochromaticRectangle (n k : ℕ) (c : Coloring n k) : Prop :=
  ∃ (i j : Fin (2 * n)) (x y : Fin k),
    i ≠ j ∧ x ≠ y ∧ 
    c i x = c i y ∧ c j x = c j y ∧ c i x = c j x

/-- The main theorem stating the smallest k that guarantees a monochromatic rectangle for any n-coloring. -/
theorem smallest_k_for_monochromatic_rectangle (n : ℕ+) :
  ∃ (k : ℕ), k = 2 * n^2 - n + 1 ∧
  (∀ (m : ℕ), m ≥ k → ∀ (c : Coloring n m), HasMonochromaticRectangle n m c) ∧
  (∀ (m : ℕ), m < k → ∃ (c : Coloring n m), ¬HasMonochromaticRectangle n m c) :=
sorry

#check smallest_k_for_monochromatic_rectangle

end smallest_k_for_monochromatic_rectangle_l4030_403000


namespace sum_of_min_max_z_l4030_403009

-- Define the feasible region
def FeasibleRegion (x y : ℝ) : Prop :=
  2 * x - y + 2 ≥ 0 ∧ 2 * x + y - 2 ≥ 0 ∧ y ≥ 0

-- Define the function z
def z (x y : ℝ) : ℝ := x - y

-- Theorem statement
theorem sum_of_min_max_z :
  ∃ (min_z max_z : ℝ),
    (∀ (x y : ℝ), FeasibleRegion x y → z x y ≥ min_z) ∧
    (∃ (x y : ℝ), FeasibleRegion x y ∧ z x y = min_z) ∧
    (∀ (x y : ℝ), FeasibleRegion x y → z x y ≤ max_z) ∧
    (∃ (x y : ℝ), FeasibleRegion x y ∧ z x y = max_z) ∧
    min_z + max_z = -1 :=
by sorry

end sum_of_min_max_z_l4030_403009


namespace sarah_borrowed_l4030_403086

/-- Calculates the earnings for a given number of hours based on the described wage structure --/
def earnings (hours : ℕ) : ℕ :=
  let fullCycles := hours / 8
  let remainingHours := hours % 8
  fullCycles * (1 + 2 + 3 + 4 + 5 + 6 + 7 + 8) + 
    (List.range remainingHours).sum.succ

/-- The amount Sarah borrowed is equal to her earnings for 40 hours of work --/
theorem sarah_borrowed (borrowedAmount : ℕ) : borrowedAmount = earnings 40 := by
  sorry

#eval earnings 40  -- Should output 180

end sarah_borrowed_l4030_403086


namespace min_value_of_function_l4030_403095

theorem min_value_of_function (x : ℝ) (h : x > 0) : 
  x + 4 / x^2 ≥ 3 ∧ ∀ ε > 0, ∃ x₀ > 0, x₀ + 4 / x₀^2 < 3 + ε :=
sorry

end min_value_of_function_l4030_403095


namespace cubic_expression_value_l4030_403053

theorem cubic_expression_value (p q : ℝ) : 
  3 * p^2 - 5 * p - 12 = 0 →
  3 * q^2 - 5 * q - 12 = 0 →
  p ≠ q →
  (9 * p^3 - 9 * q^3) / (p - q) = 61 := by
sorry

end cubic_expression_value_l4030_403053


namespace parabola_translation_l4030_403071

/-- Represents a vertical translation of a parabola -/
def verticalTranslation (f : ℝ → ℝ) (k : ℝ) : ℝ → ℝ := λ x ↦ f x + k

/-- The original parabola function -/
def originalParabola : ℝ → ℝ := λ x ↦ x^2

theorem parabola_translation :
  verticalTranslation originalParabola 4 = λ x ↦ x^2 + 4 := by
  sorry

end parabola_translation_l4030_403071


namespace sharp_triple_30_l4030_403065

-- Define the function #
def sharp (N : ℝ) : ℝ := 0.6 * N + 2

-- Theorem statement
theorem sharp_triple_30 : sharp (sharp (sharp 30)) = 10.4 := by
  sorry

end sharp_triple_30_l4030_403065


namespace ordering_from_log_half_inequalities_l4030_403079

-- Define the logarithm function with base 1/2
noncomputable def log_half (x : ℝ) : ℝ := Real.log x / Real.log (1/2)

-- State the theorem
theorem ordering_from_log_half_inequalities 
  (a b c : ℝ) 
  (h1 : a > 0) (h2 : b > 0) (h3 : c > 0)
  (h4 : log_half b < log_half a) 
  (h5 : log_half a < log_half c) : 
  c < a ∧ a < b := by
  sorry

end ordering_from_log_half_inequalities_l4030_403079


namespace ac_price_is_1500_l4030_403018

-- Define the price ratios
def car_ratio : ℚ := 5
def ac_ratio : ℚ := 3
def scooter_ratio : ℚ := 2

-- Define the price difference between scooter and air conditioner
def price_difference : ℚ := 500

-- Define the tax rate for the car
def car_tax_rate : ℚ := 0.1

-- Define the discount rate for the air conditioner
def ac_discount_rate : ℚ := 0.15

-- Define the original price of the air conditioner
def original_ac_price : ℚ := 1500

-- Theorem statement
theorem ac_price_is_1500 :
  ∃ (x : ℚ),
    scooter_ratio * x = ac_ratio * x + price_difference ∧
    original_ac_price = ac_ratio * x :=
by sorry

end ac_price_is_1500_l4030_403018


namespace circle_C_is_correct_l4030_403069

-- Define the circle C
def circle_C (x y : ℝ) : Prop := (x - 1)^2 + (y - 6)^2 = 4

-- Define the tangent line
def tangent_line (x y : ℝ) : Prop := x - 2*y + 2 = 0

-- Define points A and B
def point_A : ℝ × ℝ := (-1, 0)
def point_B : ℝ × ℝ := (3, 0)

-- Theorem statement
theorem circle_C_is_correct :
  (∀ x y : ℝ, circle_C x y → tangent_line x y → False) ∧ 
  circle_C point_A.1 point_A.2 ∧
  circle_C point_B.1 point_B.2 := by
  sorry

end circle_C_is_correct_l4030_403069


namespace max_m_value_l4030_403072

-- Define the determinant function for a 2x2 matrix
def det (a b c d : ℝ) : ℝ := a * d - b * c

-- Define the inequality condition
def inequality_condition (m : ℝ) : Prop :=
  ∀ x : ℝ, 0 < x ∧ x ≤ 1 → det (x + 1) x m (x - 1) ≥ -2

-- Theorem statement
theorem max_m_value :
  ∃ m : ℝ, inequality_condition m ∧ ∀ m' : ℝ, inequality_condition m' → m' ≤ m :=
sorry

end max_m_value_l4030_403072


namespace power_fraction_evaluation_l4030_403003

theorem power_fraction_evaluation :
  ((2^1010)^2 - (2^1008)^2) / ((2^1009)^2 - (2^1007)^2) = 4 := by
  sorry

end power_fraction_evaluation_l4030_403003


namespace max_soap_boxes_in_carton_l4030_403012

/-- Represents the dimensions of a box -/
structure BoxDimensions where
  length : ℕ
  width : ℕ
  height : ℕ

/-- Calculates the maximum number of smaller boxes that can fit in a larger box -/
def maxBoxes (carton : BoxDimensions) (soapBox : BoxDimensions) : ℕ :=
  (carton.length / soapBox.length) * (carton.width / soapBox.height) * (carton.height / soapBox.width)

/-- Theorem stating the maximum number of soap boxes that can fit in the carton -/
theorem max_soap_boxes_in_carton :
  let carton : BoxDimensions := ⟨48, 25, 60⟩
  let soapBox : BoxDimensions := ⟨8, 6, 5⟩
  maxBoxes carton soapBox = 300 := by
  sorry

#eval maxBoxes ⟨48, 25, 60⟩ ⟨8, 6, 5⟩

end max_soap_boxes_in_carton_l4030_403012


namespace complex_number_magnitude_l4030_403097

theorem complex_number_magnitude (z w : ℂ) 
  (h1 : Complex.abs (3 * z - 2 * w) = 30)
  (h2 : Complex.abs (2 * z + 3 * w) = 19)
  (h3 : Complex.abs (z + w) = 5) :
  Complex.abs z = Real.sqrt 89 := by
  sorry

end complex_number_magnitude_l4030_403097


namespace x_satisfies_equation_x_is_approximately_69_28_l4030_403088

/-- The number that satisfies the given equation -/
def x : ℝ := 69.28

/-- The given approximation of q -/
def q_approx : ℝ := 9.237333333333334

/-- Theorem stating that x satisfies the equation within a small margin of error -/
theorem x_satisfies_equation : 
  abs ((x * 0.004) / 0.03 - q_approx) < 0.000001 := by
  sorry

/-- Theorem stating that x is approximately equal to 69.28 -/
theorem x_is_approximately_69_28 : 
  abs (x - 69.28) < 0.000001 := by
  sorry

end x_satisfies_equation_x_is_approximately_69_28_l4030_403088


namespace smallest_value_in_ratio_l4030_403036

theorem smallest_value_in_ratio (a b c d x y z : ℝ) 
  (h_pos : 0 < a ∧ 0 < b ∧ 0 < c)
  (h_order : a < b ∧ b < c)
  (h_ratio : ∃ k : ℝ, x = k * a ∧ y = k * b ∧ z = k * c)
  (h_sum : x + y + z = d) :
  min x (min y z) = d * a / (a + b + c) :=
by
  sorry

end smallest_value_in_ratio_l4030_403036


namespace expression_equals_one_l4030_403006

theorem expression_equals_one :
  (144^2 - 12^2) / (120^2 - 18^2) * ((120 - 18) * (120 + 18)) / ((144 - 12) * (144 + 12)) = 1 := by
  sorry

end expression_equals_one_l4030_403006


namespace isosceles_trapezoid_projections_imply_frustum_of_cone_l4030_403092

/-- A solid object in 3D space. -/
structure Solid :=
  (shape : Type)

/-- Represents a view of a solid. -/
inductive View
  | Front
  | Side

/-- Represents the shape of a 2D projection. -/
inductive ProjectionShape
  | IsoscelesTrapezoid
  | Other

/-- Returns the shape of the projection of a solid from a given view. -/
def projection (s : Solid) (v : View) : ProjectionShape :=
  sorry

/-- Represents a frustum of a cone. -/
def FrustumOfCone : Solid :=
  sorry

/-- Theorem stating that a solid with isosceles trapezoid projections
    in both front and side views is a frustum of a cone. -/
theorem isosceles_trapezoid_projections_imply_frustum_of_cone
  (s : Solid)
  (h1 : projection s View.Front = ProjectionShape.IsoscelesTrapezoid)
  (h2 : projection s View.Side = ProjectionShape.IsoscelesTrapezoid) :
  s = FrustumOfCone :=
sorry

end isosceles_trapezoid_projections_imply_frustum_of_cone_l4030_403092


namespace cos_2x_value_l4030_403051

theorem cos_2x_value (x : ℝ) (h : Real.sin (π / 4 + x / 2) = 3 / 5) : 
  Real.cos (2 * x) = -7 / 25 := by
  sorry

end cos_2x_value_l4030_403051


namespace sqrt_square_equality_implies_geq_l4030_403049

theorem sqrt_square_equality_implies_geq (a : ℝ) : 
  Real.sqrt ((a - 2)^2) = a - 2 → a ≥ 2 := by
  sorry

end sqrt_square_equality_implies_geq_l4030_403049


namespace soda_cost_calculation_l4030_403023

theorem soda_cost_calculation (regular_bottles : ℕ) (regular_price : ℚ) 
  (diet_bottles : ℕ) (diet_price : ℚ) (regular_discount : ℚ) (diet_tax : ℚ) :
  regular_bottles = 49 →
  regular_price = 120/100 →
  diet_bottles = 40 →
  diet_price = 110/100 →
  regular_discount = 10/100 →
  diet_tax = 8/100 →
  (regular_bottles : ℚ) * regular_price * (1 - regular_discount) + 
  (diet_bottles : ℚ) * diet_price * (1 + diet_tax) = 10044/100 := by
sorry

end soda_cost_calculation_l4030_403023


namespace simplify_expression_l4030_403025

/-- Proves that the simplified expression is equal to the original expression for all real x. -/
theorem simplify_expression (x : ℝ) : 3*x + 9*x^2 + 16 - (5 - 3*x - 9*x^2 + x^3) = -x^3 + 18*x^2 + 6*x + 11 := by
  sorry

end simplify_expression_l4030_403025


namespace storks_on_fence_l4030_403034

/-- The number of storks on a fence, given the initial number of birds,
    the number of birds that join, and the final difference between birds and storks. -/
def number_of_storks (initial_birds : ℕ) (joining_birds : ℕ) (final_difference : ℕ) : ℕ :=
  initial_birds + joining_birds - final_difference

/-- Theorem stating that the number of storks is 4 under the given conditions. -/
theorem storks_on_fence : number_of_storks 3 2 1 = 4 := by
  sorry

end storks_on_fence_l4030_403034


namespace parallel_line_equation_l4030_403082

/-- The equation of a line passing through (-1, 2) and parallel to 2x + y - 5 = 0 is 2x + y = 0 -/
theorem parallel_line_equation :
  let P : ℝ × ℝ := (-1, 2)
  let l₁ : ℝ → ℝ → Prop := λ x y ↦ 2 * x + y - 5 = 0
  let l₂ : ℝ → ℝ → Prop := λ x y ↦ 2 * x + y = 0
  (∀ x y, l₂ x y ↔ (2 * P.1 + P.2 = 2 * x + y ∧ ∀ x₁ y₁ x₂ y₂, l₁ x₁ y₁ → l₁ x₂ y₂ → 
    2 * (x₂ - x₁) = y₁ - y₂)) := by
  sorry

end parallel_line_equation_l4030_403082


namespace max_value_expression_l4030_403090

theorem max_value_expression :
  ∃ (M : ℝ), M = 27 ∧
  ∀ (x y : ℝ),
    (Real.sqrt (36 - 4 * Real.sqrt 5) * Real.sin x - Real.sqrt (2 * (1 + Real.cos (2 * x))) - 2) *
    (3 + 2 * Real.sqrt (10 - Real.sqrt 5) * Real.cos y - Real.cos (2 * y)) ≤ M :=
by sorry

end max_value_expression_l4030_403090


namespace min_value_of_expression_l4030_403005

theorem min_value_of_expression (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) 
  (sum_eq_3 : a + b + c = 3) : 
  let A := (a^3 + b^3) / (8*a*b + 9 - c^2) + 
           (b^3 + c^3) / (8*b*c + 9 - a^2) + 
           (c^3 + a^3) / (8*c*a + 9 - b^2)
  ∀ x, A ≥ x → x ≤ 3/8 :=
by sorry

end min_value_of_expression_l4030_403005


namespace ratio_problem_l4030_403042

theorem ratio_problem (a b c d : ℚ) 
  (h1 : a / b = 5)
  (h2 : c / b = 3)
  (h3 : c / d = 2) :
  d / a = 3 / 10 := by
sorry

end ratio_problem_l4030_403042


namespace birthday_45_days_later_l4030_403059

/-- Days of the week represented as integers mod 7 -/
inductive DayOfWeek : Type
| Sunday : DayOfWeek
| Monday : DayOfWeek
| Tuesday : DayOfWeek
| Wednesday : DayOfWeek
| Thursday : DayOfWeek
| Friday : DayOfWeek
| Saturday : DayOfWeek

/-- Function to add days to a given day of the week -/
def addDays (start : DayOfWeek) (days : Nat) : DayOfWeek :=
  match (start, days % 7) with
  | (DayOfWeek.Sunday, 0) => DayOfWeek.Sunday
  | (DayOfWeek.Sunday, 1) => DayOfWeek.Monday
  | (DayOfWeek.Sunday, 2) => DayOfWeek.Tuesday
  | (DayOfWeek.Sunday, 3) => DayOfWeek.Wednesday
  | (DayOfWeek.Sunday, 4) => DayOfWeek.Thursday
  | (DayOfWeek.Sunday, 5) => DayOfWeek.Friday
  | (DayOfWeek.Sunday, 6) => DayOfWeek.Saturday
  | (DayOfWeek.Monday, 0) => DayOfWeek.Monday
  | (DayOfWeek.Monday, 1) => DayOfWeek.Tuesday
  | (DayOfWeek.Monday, 2) => DayOfWeek.Wednesday
  | (DayOfWeek.Monday, 3) => DayOfWeek.Thursday
  | (DayOfWeek.Monday, 4) => DayOfWeek.Friday
  | (DayOfWeek.Monday, 5) => DayOfWeek.Saturday
  | (DayOfWeek.Monday, 6) => DayOfWeek.Sunday
  | (DayOfWeek.Tuesday, 0) => DayOfWeek.Tuesday
  | (DayOfWeek.Tuesday, 1) => DayOfWeek.Wednesday
  | (DayOfWeek.Tuesday, 2) => DayOfWeek.Thursday
  | (DayOfWeek.Tuesday, 3) => DayOfWeek.Friday
  | (DayOfWeek.Tuesday, 4) => DayOfWeek.Saturday
  | (DayOfWeek.Tuesday, 5) => DayOfWeek.Sunday
  | (DayOfWeek.Tuesday, 6) => DayOfWeek.Monday
  | (DayOfWeek.Wednesday, 0) => DayOfWeek.Wednesday
  | (DayOfWeek.Wednesday, 1) => DayOfWeek.Thursday
  | (DayOfWeek.Wednesday, 2) => DayOfWeek.Friday
  | (DayOfWeek.Wednesday, 3) => DayOfWeek.Saturday
  | (DayOfWeek.Wednesday, 4) => DayOfWeek.Sunday
  | (DayOfWeek.Wednesday, 5) => DayOfWeek.Monday
  | (DayOfWeek.Wednesday, 6) => DayOfWeek.Tuesday
  | (DayOfWeek.Thursday, 0) => DayOfWeek.Thursday
  | (DayOfWeek.Thursday, 1) => DayOfWeek.Friday
  | (DayOfWeek.Thursday, 2) => DayOfWeek.Saturday
  | (DayOfWeek.Thursday, 3) => DayOfWeek.Sunday
  | (DayOfWeek.Thursday, 4) => DayOfWeek.Monday
  | (DayOfWeek.Thursday, 5) => DayOfWeek.Tuesday
  | (DayOfWeek.Thursday, 6) => DayOfWeek.Wednesday
  | (DayOfWeek.Friday, 0) => DayOfWeek.Friday
  | (DayOfWeek.Friday, 1) => DayOfWeek.Saturday
  | (DayOfWeek.Friday, 2) => DayOfWeek.Sunday
  | (DayOfWeek.Friday, 3) => DayOfWeek.Monday
  | (DayOfWeek.Friday, 4) => DayOfWeek.Tuesday
  | (DayOfWeek.Friday, 5) => DayOfWeek.Wednesday
  | (DayOfWeek.Friday, 6) => DayOfWeek.Thursday
  | (DayOfWeek.Saturday, 0) => DayOfWeek.Saturday
  | (DayOfWeek.Saturday, 1) => DayOfWeek.Sunday
  | (DayOfWeek.Saturday, 2) => DayOfWeek.Monday
  | (DayOfWeek.Saturday, 3) => DayOfWeek.Tuesday
  | (DayOfWeek.Saturday, 4) => DayOfWeek.Wednesday
  | (DayOfWeek.Saturday, 5) => DayOfWeek.Thursday
  | (DayOfWeek.Saturday, 6) => DayOfWeek.Friday
  | _ => DayOfWeek.Sunday  -- This case should never happen

theorem birthday_45_days_later (birthday : DayOfWeek) :
  birthday = DayOfWeek.Tuesday → addDays birthday 45 = DayOfWeek.Friday :=
by sorry

end birthday_45_days_later_l4030_403059


namespace sum_inequality_l4030_403047

theorem sum_inequality (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (sum_abc : a + b + c = 1) :
  (1 / (b * c + a + 1 / a)) + (1 / (a * c + b + 1 / b)) + (1 / (a * b + c + 1 / c)) ≤ 27 / 31 := by
  sorry

end sum_inequality_l4030_403047


namespace davids_biology_marks_l4030_403041

def english_marks : ℕ := 86
def math_marks : ℕ := 89
def physics_marks : ℕ := 82
def chemistry_marks : ℕ := 87
def average_marks : ℕ := 85
def num_subjects : ℕ := 5

theorem davids_biology_marks :
  let known_subjects_total := english_marks + math_marks + physics_marks + chemistry_marks
  let all_subjects_total := average_marks * num_subjects
  all_subjects_total - known_subjects_total = 81 := by sorry

end davids_biology_marks_l4030_403041


namespace area_between_concentric_circles_l4030_403078

theorem area_between_concentric_circles (r : ℝ) (h1 : r > 0) :
  let R := 3 * r
  R - r = 3 →
  π * R^2 - π * r^2 = 18 * π :=
by sorry

end area_between_concentric_circles_l4030_403078


namespace necessary_not_sufficient_condition_l4030_403022

theorem necessary_not_sufficient_condition (a b : ℝ) :
  (∀ x y : ℝ, x > y → x + 1 > y) ∧
  (∃ x y : ℝ, x + 1 > y ∧ ¬(x > y)) :=
sorry

end necessary_not_sufficient_condition_l4030_403022


namespace problem_1_problem_2_problem_3_l4030_403080

-- Problem 1
theorem problem_1 : 2 * (Real.sqrt 5 - 1) - Real.sqrt 5 = Real.sqrt 5 - 2 := by sorry

-- Problem 2
theorem problem_2 : Real.sqrt 3 * (Real.sqrt 3 + 4 / Real.sqrt 3) = 7 := by sorry

-- Problem 3
theorem problem_3 : |Real.sqrt 3 - 2| + 3 - 27 + Real.sqrt ((-5)^2) = 3 - Real.sqrt 3 := by sorry

end problem_1_problem_2_problem_3_l4030_403080


namespace wednesday_bags_raked_l4030_403073

theorem wednesday_bags_raked (charge_per_bag : ℕ) (monday_bags : ℕ) (tuesday_bags : ℕ) (total_money : ℕ) :
  charge_per_bag = 4 →
  monday_bags = 5 →
  tuesday_bags = 3 →
  total_money = 68 →
  ∃ wednesday_bags : ℕ, wednesday_bags = 9 ∧ 
    total_money = charge_per_bag * (monday_bags + tuesday_bags + wednesday_bags) :=
by sorry

end wednesday_bags_raked_l4030_403073


namespace range_of_m_l4030_403074

theorem range_of_m (m : ℝ) : 
  (|m + 3| = m + 3) →
  (|3*m + 9| ≥ 4*m - 3 ↔ -3 ≤ m ∧ m ≤ 12) :=
by sorry

end range_of_m_l4030_403074


namespace remainder_of_base_12_num_div_9_l4030_403061

-- Define the base-12 number 1742₁₂
def base_12_num : ℕ := 1 * 12^3 + 7 * 12^2 + 4 * 12 + 2

-- Theorem statement
theorem remainder_of_base_12_num_div_9 :
  base_12_num % 9 = 5 := by
  sorry

end remainder_of_base_12_num_div_9_l4030_403061


namespace acceleration_at_two_l4030_403087

-- Define the distance function
def s (t : ℝ) : ℝ := 2 * t^3 - 5 * t^2

-- Define the velocity function as the derivative of the distance function
def v (t : ℝ) : ℝ := 6 * t^2 - 10 * t

-- Define the acceleration function as the derivative of the velocity function
def a (t : ℝ) : ℝ := 12 * t - 10

-- Theorem: The acceleration at t = 2 seconds is 14 units
theorem acceleration_at_two : a 2 = 14 := by
  sorry

-- Lemma: The velocity function is the derivative of the distance function
lemma velocity_is_derivative_of_distance (t : ℝ) : 
  deriv s t = v t := by
  sorry

-- Lemma: The acceleration function is the derivative of the velocity function
lemma acceleration_is_derivative_of_velocity (t : ℝ) : 
  deriv v t = a t := by
  sorry

end acceleration_at_two_l4030_403087


namespace first_month_sale_is_2500_l4030_403043

/-- Calculates the sale in the first month given the sales in other months and the average -/
def first_month_sale (second_month : ℕ) (third_month : ℕ) (fourth_month : ℕ) (average : ℕ) : ℕ :=
  4 * average - (second_month + third_month + fourth_month)

/-- Proves that the sale in the first month is 2500 given the conditions -/
theorem first_month_sale_is_2500 :
  first_month_sale 4000 3540 1520 2890 = 2500 := by
  sorry

end first_month_sale_is_2500_l4030_403043


namespace tangent_beta_l4030_403066

theorem tangent_beta (a b : ℝ) (α β γ : Real) 
  (h1 : (a + b) / (a - b) = Real.tan ((α + β) / 2) / Real.tan ((α - β) / 2))
  (h2 : (α + β) / 2 = π / 2 - γ / 2)
  (h3 : (α - β) / 2 = π / 2 - (β + γ / 2)) :
  Real.tan β = (2 * b * Real.tan (γ / 2)) / ((a + b) * Real.tan (γ / 2)^2 + (a - b)) := by
  sorry

end tangent_beta_l4030_403066


namespace six_years_passed_l4030_403008

/-- Represents a stem-and-leaf plot --/
structure StemAndLeafPlot where
  stem : List Nat
  leaves : List (List Nat)

/-- The initial stem-and-leaf plot --/
def initial_plot : StemAndLeafPlot := {
  stem := [0, 1, 2, 3, 4, 5],
  leaves := [[3], [0, 1, 2, 3, 4, 5], [2, 3, 5, 6, 8, 9], [4, 6], [0, 2], []]
}

/-- The final stem-and-leaf plot with obscured numbers --/
def final_plot : StemAndLeafPlot := {
  stem := [0, 1, 2, 3, 4, 5],
  leaves := [[], [6, 9], [4, 7], [0], [2, 8], []]
}

/-- Function to calculate the years passed --/
def years_passed (initial : StemAndLeafPlot) (final : StemAndLeafPlot) : Nat :=
  sorry

/-- Theorem stating that 6 years have passed --/
theorem six_years_passed :
  years_passed initial_plot final_plot = 6 := by
  sorry

end six_years_passed_l4030_403008


namespace abacus_problem_l4030_403001

def is_valid_abacus_division (upper lower : ℕ) : Prop :=
  upper ≥ 100 ∧ upper < 1000 ∧ lower ≥ 100 ∧ lower < 1000 ∧
  upper + lower = 1110 ∧
  (∃ k : ℕ, upper = k * lower) ∧
  (∃ a b c : ℕ, upper = 100 * a + 10 * b + c ∧ a ≠ b ∧ b ≠ c ∧ a ≠ c)

theorem abacus_problem : ∃ upper lower : ℕ, is_valid_abacus_division upper lower ∧ upper = 925 := by
  sorry

end abacus_problem_l4030_403001


namespace board_cut_ratio_l4030_403032

/-- Given a board of length 69 inches cut into two pieces, where the shorter piece is 23 inches long,
    the ratio of the longer piece to the shorter piece is 2:1. -/
theorem board_cut_ratio : 
  ∀ (short_piece long_piece : ℝ),
  short_piece = 23 →
  short_piece + long_piece = 69 →
  long_piece / short_piece = 2 := by
sorry

end board_cut_ratio_l4030_403032


namespace largest_digit_divisible_by_six_l4030_403046

theorem largest_digit_divisible_by_six :
  ∃ (N : ℕ), N ≤ 9 ∧ (5217 * 10 + N) % 6 = 0 ∧
  ∀ (M : ℕ), M ≤ 9 → (5217 * 10 + M) % 6 = 0 → M ≤ N :=
by sorry

end largest_digit_divisible_by_six_l4030_403046


namespace purely_imaginary_complex_number_l4030_403030

theorem purely_imaginary_complex_number (m : ℝ) : 
  (m^2 + 3*m - 4 = 0) ∧ (m + 4 ≠ 0) → m = 1 := by
  sorry

end purely_imaginary_complex_number_l4030_403030


namespace black_bears_count_l4030_403089

/-- Represents the number of bears of each color in the park -/
structure BearPopulation where
  white : ℕ
  black : ℕ
  brown : ℕ

/-- Conditions for the bear population in the park -/
def validBearPopulation (p : BearPopulation) : Prop :=
  p.black = 2 * p.white ∧
  p.brown = p.black + 40 ∧
  p.white + p.black + p.brown = 190

/-- Theorem stating that under the given conditions, there are 60 black bears -/
theorem black_bears_count (p : BearPopulation) (h : validBearPopulation p) : p.black = 60 := by
  sorry

end black_bears_count_l4030_403089


namespace martian_traffic_light_signals_l4030_403011

/-- Represents a Martian traffic light configuration -/
def MartianTrafficLight := Fin 6 → Bool

/-- The number of bulbs in the traffic light -/
def num_bulbs : Nat := 6

/-- Checks if two configurations are indistinguishable under the given conditions -/
def indistinguishable (c1 c2 : MartianTrafficLight) : Prop :=
  sorry

/-- Counts the number of distinguishable configurations -/
def count_distinguishable_configs : Nat :=
  sorry

/-- Theorem stating the number of distinguishable Martian traffic light signals -/
theorem martian_traffic_light_signals :
  count_distinguishable_configs = 44 :=
sorry

end martian_traffic_light_signals_l4030_403011


namespace distinct_triangles_in_3x2_grid_l4030_403085

/-- Represents a grid of dots -/
structure Grid :=
  (rows : Nat)
  (cols : Nat)

/-- Calculates the total number of dots in the grid -/
def Grid.totalDots (g : Grid) : Nat :=
  g.rows * g.cols

/-- Calculates the number of collinear groups in the grid -/
def Grid.collinearGroups (g : Grid) : Nat :=
  g.rows + g.cols

/-- Theorem: In a 3x2 grid, the number of distinct triangles is 15 -/
theorem distinct_triangles_in_3x2_grid :
  let g : Grid := { rows := 3, cols := 2 }
  let totalCombinations := Nat.choose (g.totalDots) 3
  let validTriangles := totalCombinations - g.collinearGroups
  validTriangles = 15 := by sorry


end distinct_triangles_in_3x2_grid_l4030_403085


namespace lcm_210_297_l4030_403075

theorem lcm_210_297 : Nat.lcm 210 297 = 20790 := by
  sorry

end lcm_210_297_l4030_403075


namespace min_value_theorem_l4030_403098

theorem min_value_theorem (x y : ℝ) (h1 : x > 0) (h2 : y > 0) (h3 : x + y = 4) :
  (1 / x + 4 / y) ≥ 9 / 4 ∧ 
  (1 / x + 4 / y = 9 / 4 ↔ y = 8 / 3 ∧ x = 4 / 3) :=
by sorry

end min_value_theorem_l4030_403098


namespace tiling_cost_theorem_l4030_403015

/-- Calculates the total cost of tiling a wall -/
def total_tiling_cost (wall_width wall_height tile_length tile_width tile_cost : ℕ) : ℕ :=
  let wall_area := wall_width * wall_height
  let tile_area := tile_length * tile_width
  let num_tiles := (wall_area + tile_area - 1) / tile_area  -- Ceiling division
  num_tiles * tile_cost

/-- Theorem: The total cost of tiling the given wall is 540,000 won -/
theorem tiling_cost_theorem : 
  total_tiling_cost 36 72 3 4 2500 = 540000 := by
  sorry

end tiling_cost_theorem_l4030_403015


namespace sin_pi_over_two_minus_pi_over_six_l4030_403081

theorem sin_pi_over_two_minus_pi_over_six :
  Real.sin (π / 2 - π / 6) = Real.sqrt 3 / 2 := by
  sorry

end sin_pi_over_two_minus_pi_over_six_l4030_403081


namespace proposition_evaluation_l4030_403067

theorem proposition_evaluation :
  (¬ ∀ m : ℝ, m > 0 → ∃ x : ℝ, x^2 - x + m = 0) ∧
  (¬ ∀ x y : ℝ, x + y > 2 → x > 1 ∧ y > 1) ∧
  (∃ x : ℝ, -2 < x ∧ x < 4 ∧ |x - 2| ≥ 3) ∧
  (¬ ∀ a b c : ℝ, a ≠ 0 →
    (b^2 - 4*a*c > 0 ↔ 
      ∃ x y : ℝ, x < 0 ∧ y > 0 ∧ 
      a*x^2 + b*x + c = 0 ∧ 
      a*y^2 + b*y + c = 0)) := by
  sorry

end proposition_evaluation_l4030_403067


namespace problem_solution_l4030_403057

theorem problem_solution (k : ℝ) (h1 : k ≠ 0) :
  (∀ x : ℝ, (x^2 - 3*k) * (x + 3*k) = x^3 + 3*k*(x^2 - x - 7)) →
  k = 7/3 := by
  sorry

end problem_solution_l4030_403057


namespace min_dot_product_sum_l4030_403093

/-- The ellipse on which point P moves --/
def ellipse (x y : ℝ) : Prop := x^2 / 4 + y^2 / 3 = 1

/-- Circle E --/
def circle_E (x y : ℝ) : Prop := (x - 1)^2 + y^2 = 1

/-- Circle F --/
def circle_F (x y : ℝ) : Prop := (x + 1)^2 + y^2 = 1

/-- The dot product of vectors PA and PB plus the dot product of vectors PC and PD --/
def dot_product_sum (a b : ℝ) : ℝ := 2 * (a^2 + b^2)

theorem min_dot_product_sum :
  ∀ a b : ℝ, ellipse a b → 
  (∀ x y : ℝ, circle_E x y → ∀ u v : ℝ, circle_F u v → 
    dot_product_sum a b ≥ 6) ∧ 
  (∃ x y u v : ℝ, circle_E x y ∧ circle_F u v ∧ dot_product_sum a b = 6) :=
sorry

end min_dot_product_sum_l4030_403093


namespace secant_slope_on_curve_l4030_403035

def f (x : ℝ) : ℝ := x^2 + x

theorem secant_slope_on_curve (Δx : ℝ) (Δy : ℝ) 
  (h1 : f 2 = 6)  -- P(2, 6) is on the curve
  (h2 : f (2 + Δx) = 6 + Δy)  -- Q(2 + Δx, 6 + Δy) is on the curve
  (h3 : Δx ≠ 0)  -- Ensure Δx is not zero for division
  : Δy / Δx = Δx + 5 := by
  sorry

#check secant_slope_on_curve

end secant_slope_on_curve_l4030_403035


namespace calligraphy_students_l4030_403013

theorem calligraphy_students (x : ℕ) : 
  (50 : ℕ) = (2 * x - 1) + x + (51 - 3 * x) :=
by sorry

end calligraphy_students_l4030_403013


namespace division_problem_l4030_403050

theorem division_problem (a b q : ℕ) 
  (h1 : a - b = 1200)
  (h2 : a = 1495)
  (h3 : a = b * q + 4) :
  q = 5 := by
sorry

end division_problem_l4030_403050


namespace natashas_journey_l4030_403099

/-- Natasha's hill climbing problem -/
theorem natashas_journey (time_up : ℝ) (time_down : ℝ) (speed_up : ℝ) :
  time_up = 4 →
  time_down = 2 →
  speed_up = 2.25 →
  (speed_up * time_up * 2) / (time_up + time_down) = 3 :=
by sorry

end natashas_journey_l4030_403099


namespace max_distance_for_given_tires_l4030_403004

/-- Represents the maximum distance a car can travel by switching tires -/
def max_distance (front_tire_life rear_tire_life : ℕ) : ℕ :=
  let swap_point := front_tire_life / 2
  swap_point + min (rear_tire_life - swap_point) (front_tire_life - swap_point)

/-- Theorem stating the maximum distance a car can travel with given tire lifespans -/
theorem max_distance_for_given_tires :
  max_distance 21000 28000 = 24000 := by
  sorry

end max_distance_for_given_tires_l4030_403004


namespace root_sum_eighth_power_l4030_403002

theorem root_sum_eighth_power (r s : ℝ) : 
  r^2 - r * Real.sqrt 5 + 1 = 0 ∧ 
  s^2 - s * Real.sqrt 5 + 1 = 0 → 
  r^8 + s^8 = 47 := by
  sorry

end root_sum_eighth_power_l4030_403002


namespace second_strongest_in_final_probability_l4030_403027

/-- Represents a player in the tournament -/
structure Player where
  strength : ℕ

/-- Represents a tournament with 8 players -/
structure Tournament where
  players : Fin 8 → Player
  strength_ordered : ∀ i j, i < j → (players i).strength > (players j).strength

/-- The probability that the second strongest player reaches the final -/
def probability_second_strongest_in_final (t : Tournament) : ℚ :=
  4 / 7

/-- Theorem stating that the probability of the second strongest player
    reaching the final is 4/7 -/
theorem second_strongest_in_final_probability (t : Tournament) :
  probability_second_strongest_in_final t = 4 / 7 := by
  sorry

end second_strongest_in_final_probability_l4030_403027


namespace isosceles_triangle_perimeter_l4030_403031

theorem isosceles_triangle_perimeter : ∀ x y : ℝ,
  x^2 - 9*x + 18 = 0 →
  y^2 - 9*y + 18 = 0 →
  x ≠ y →
  (x + 2*y = 15 ∨ y + 2*x = 15) :=
by sorry

end isosceles_triangle_perimeter_l4030_403031


namespace triangle_area_reduction_l4030_403070

theorem triangle_area_reduction (b h m : ℝ) (hb : b > 0) (hh : h > 0) (hm : m ≥ 0) :
  ∃ x : ℝ, 
    (1/2 : ℝ) * (b - x) * (h + m) = (1/2 : ℝ) * ((1/2 : ℝ) * b * h) ∧
    x = b * (2 * m + h) / (2 * (h + m)) := by
  sorry

end triangle_area_reduction_l4030_403070


namespace assign_roles_for_five_men_six_women_l4030_403017

/-- The number of ways to assign roles in a play --/
def assignRoles (numMen numWomen : ℕ) : ℕ :=
  let maleRoles := 2
  let femaleRoles := 2
  let eitherGenderRoles := 2
  let remainingActors := numMen + numWomen - maleRoles - femaleRoles
  (numMen.descFactorial maleRoles) *
  (numWomen.descFactorial femaleRoles) *
  (remainingActors.descFactorial eitherGenderRoles)

/-- Theorem stating the number of ways to assign roles for 5 men and 6 women --/
theorem assign_roles_for_five_men_six_women :
  assignRoles 5 6 = 25200 := by
  sorry

end assign_roles_for_five_men_six_women_l4030_403017


namespace no_solution_for_sqrt_equation_l4030_403060

theorem no_solution_for_sqrt_equation :
  ¬∃ x : ℝ, Real.sqrt (3*x - 2) + Real.sqrt (2*x - 2) + Real.sqrt (x - 1) = 3 :=
by sorry

end no_solution_for_sqrt_equation_l4030_403060

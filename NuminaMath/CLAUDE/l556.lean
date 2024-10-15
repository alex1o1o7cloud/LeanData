import Mathlib

namespace NUMINAMATH_CALUDE_total_increase_in_two_centuries_l556_55642

/-- Represents the increase in height per decade in meters -/
def increase_per_decade : ℝ := 90

/-- Represents the number of decades in 2 centuries -/
def decades_in_two_centuries : ℕ := 20

/-- Represents the total increase in height over 2 centuries in meters -/
def total_increase : ℝ := increase_per_decade * decades_in_two_centuries

/-- Theorem stating that the total increase in height over 2 centuries is 1800 meters -/
theorem total_increase_in_two_centuries : total_increase = 1800 := by
  sorry

end NUMINAMATH_CALUDE_total_increase_in_two_centuries_l556_55642


namespace NUMINAMATH_CALUDE_boy_speed_around_square_l556_55665

/-- The speed of a boy running around a square field -/
theorem boy_speed_around_square (side_length : ℝ) (time : ℝ) : 
  side_length = 20 → time = 24 → 
  (4 * side_length) / time * (3600 / 1000) = 12 := by
  sorry

end NUMINAMATH_CALUDE_boy_speed_around_square_l556_55665


namespace NUMINAMATH_CALUDE_circle_radii_theorem_l556_55628

/-- The configuration of circles as described in the problem -/
structure CircleConfiguration where
  r : ℝ  -- radius of white circles
  red_radius : ℝ  -- radius of Adam's red circle
  green_radius : ℝ  -- radius of Eva's green circle

/-- The theorem stating the radii of the red and green circles -/
theorem circle_radii_theorem (config : CircleConfiguration) :
  config.red_radius = (Real.sqrt 2 - 1) * config.r ∧
  config.green_radius = (2 * Real.sqrt 3 - 3) / 3 * config.r :=
by sorry

end NUMINAMATH_CALUDE_circle_radii_theorem_l556_55628


namespace NUMINAMATH_CALUDE_factors_of_72_l556_55667

def number_of_factors (n : ℕ) : ℕ := (Finset.filter (· ∣ n) (Finset.range (n + 1))).card

theorem factors_of_72 : number_of_factors 72 = 12 := by
  sorry

end NUMINAMATH_CALUDE_factors_of_72_l556_55667


namespace NUMINAMATH_CALUDE_chromium_54_neutrons_l556_55619

/-- The number of neutrons in an atom of chromium-54 -/
def neutrons_per_atom : ℕ := 54 - 24

/-- Avogadro's constant (atoms per mole) -/
def avogadro : ℝ := 6.022e23

/-- Amount of substance in moles -/
def amount : ℝ := 0.025

/-- Approximate number of neutrons in the given amount of chromium-54 -/
def total_neutrons : ℝ := amount * avogadro * neutrons_per_atom

theorem chromium_54_neutrons : 
  ∃ (ε : ℝ), ε > 0 ∧ ε < 0.1e23 ∧ |total_neutrons - 4.5e23| < ε :=
sorry

end NUMINAMATH_CALUDE_chromium_54_neutrons_l556_55619


namespace NUMINAMATH_CALUDE_min_value_theorem_l556_55664

theorem min_value_theorem (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : x + 2 / y = 3) :
  ∀ z, z = 2 / x + y → z ≥ 8 / 3 ∧ ∃ w, w = 2 / x + y ∧ w = 8 / 3 :=
sorry

end NUMINAMATH_CALUDE_min_value_theorem_l556_55664


namespace NUMINAMATH_CALUDE_no_real_roots_for_sqrt_equation_l556_55641

theorem no_real_roots_for_sqrt_equation :
  ¬ ∃ x : ℝ, Real.sqrt (x + 4) - Real.sqrt (x - 3) + 1 = 0 := by
  sorry

end NUMINAMATH_CALUDE_no_real_roots_for_sqrt_equation_l556_55641


namespace NUMINAMATH_CALUDE_current_velocity_velocity_of_current_l556_55610

/-- Calculates the velocity of the current given rowing conditions -/
theorem current_velocity (still_water_speed : ℝ) (total_time : ℝ) (distance : ℝ) : ℝ :=
  let v : ℝ := 2  -- The velocity of the current we want to prove
  have h1 : still_water_speed = 10 := by sorry
  have h2 : total_time = 30 := by sorry
  have h3 : distance = 144 := by sorry
  have h4 : (distance / (still_water_speed - v) + distance / (still_water_speed + v)) = total_time := by sorry
  v

/-- The main theorem stating the velocity of the current -/
theorem velocity_of_current : current_velocity 10 30 144 = 2 := by sorry

end NUMINAMATH_CALUDE_current_velocity_velocity_of_current_l556_55610


namespace NUMINAMATH_CALUDE_quadratic_divisibility_l556_55618

theorem quadratic_divisibility (p : ℕ) (a b c : ℕ) (h_prime : Nat.Prime p) 
  (h_a : 0 < a ∧ a ≤ p) (h_b : 0 < b ∧ b ≤ p) (h_c : 0 < c ∧ c ≤ p)
  (h_div : ∀ (x : ℕ), x > 0 → (p ∣ (a * x^2 + b * x + c))) :
  a + b + c = 3 * p := by
sorry

end NUMINAMATH_CALUDE_quadratic_divisibility_l556_55618


namespace NUMINAMATH_CALUDE_algebraic_expression_equality_l556_55651

theorem algebraic_expression_equality (x y : ℝ) (h : 2*x - 3*y = 1) : 
  6*y - 4*x + 8 = 6 := by
sorry

end NUMINAMATH_CALUDE_algebraic_expression_equality_l556_55651


namespace NUMINAMATH_CALUDE_wall_bricks_l556_55666

/-- The number of bricks in the wall -/
def num_bricks : ℕ := 720

/-- The time it takes Brenda to build the wall alone (in hours) -/
def brenda_time : ℕ := 12

/-- The time it takes Brandon to build the wall alone (in hours) -/
def brandon_time : ℕ := 15

/-- The decrease in combined output when working together (in bricks per hour) -/
def output_decrease : ℕ := 12

/-- The time it takes Brenda and Brandon to build the wall together (in hours) -/
def combined_time : ℕ := 6

/-- Theorem stating that the number of bricks in the wall is 720 -/
theorem wall_bricks : 
  (combined_time : ℚ) * ((num_bricks / brenda_time : ℚ) + (num_bricks / brandon_time : ℚ) - output_decrease) = num_bricks := by
  sorry

end NUMINAMATH_CALUDE_wall_bricks_l556_55666


namespace NUMINAMATH_CALUDE_range_of_a_l556_55631

open Real

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := exp (x - 1) + 2 * x - log a * x / log (sqrt 2)

theorem range_of_a (a : ℝ) (h1 : a > 0) :
  (∃ x y, 0 < x ∧ x < 2 ∧ 0 < y ∧ y < 2 ∧ x ≠ y ∧ f a x = 0 ∧ f a y = 0) →
  a > 2^(3/2) ∧ a < 2^((exp 1 + 4)/4) :=
sorry

end NUMINAMATH_CALUDE_range_of_a_l556_55631


namespace NUMINAMATH_CALUDE_last_defective_on_fifth_draw_l556_55621

def number_of_arrangements (n_total : ℕ) (n_genuine : ℕ) (n_defective : ℕ) : ℕ :=
  (n_total.choose (n_defective - 1)) * (n_defective.factorial) * n_genuine

theorem last_defective_on_fifth_draw :
  let n_total := 9
  let n_genuine := 5
  let n_defective := 4
  let n_draws := 5
  number_of_arrangements n_draws n_genuine n_defective = 480 :=
by sorry

end NUMINAMATH_CALUDE_last_defective_on_fifth_draw_l556_55621


namespace NUMINAMATH_CALUDE_red_marbles_fraction_l556_55637

theorem red_marbles_fraction (total : ℚ) (h : total > 0) : 
  let initial_blue := (2 / 3) * total
  let initial_red := total - initial_blue
  let new_red := 3 * initial_red
  let new_total := initial_blue + new_red
  new_red / new_total = 3 / 5 := by sorry

end NUMINAMATH_CALUDE_red_marbles_fraction_l556_55637


namespace NUMINAMATH_CALUDE_matias_grade_size_l556_55688

/-- Given a student's rank from best and worst in a group, calculate the total number of students -/
def totalStudents (rankBest : ℕ) (rankWorst : ℕ) : ℕ :=
  (rankBest - 1) + (rankWorst - 1) + 1

/-- Theorem: In a group where a student is both the 75th best and 75th worst, there are 149 students -/
theorem matias_grade_size :
  totalStudents 75 75 = 149 := by
  sorry

#eval totalStudents 75 75

end NUMINAMATH_CALUDE_matias_grade_size_l556_55688


namespace NUMINAMATH_CALUDE_ping_pong_probabilities_l556_55694

/-- Represents the probability of player A winning a point -/
def prob_A_wins (serving : Bool) : ℝ :=
  if serving then 0.5 else 0.4

/-- Represents the probability of player A winning the k-th point after a 10:10 tie -/
def prob_A_k (k : ℕ) : ℝ :=
  prob_A_wins (k % 2 = 1)

/-- The probability of the game ending in exactly 2 points after a 10:10 tie -/
def prob_X_2 : ℝ :=
  prob_A_k 1 * prob_A_k 2 + (1 - prob_A_k 1) * (1 - prob_A_k 2)

/-- The probability of the game ending in exactly 4 points after a 10:10 tie with A winning -/
def prob_X_4_A_wins : ℝ :=
  (1 - prob_A_k 1) * prob_A_k 2 * prob_A_k 3 * prob_A_k 4 +
  prob_A_k 1 * (1 - prob_A_k 2) * prob_A_k 3 * prob_A_k 4

theorem ping_pong_probabilities :
  (prob_X_2 = prob_A_k 1 * prob_A_k 2 + (1 - prob_A_k 1) * (1 - prob_A_k 2)) ∧
  (prob_X_4_A_wins = (1 - prob_A_k 1) * prob_A_k 2 * prob_A_k 3 * prob_A_k 4 +
                     prob_A_k 1 * (1 - prob_A_k 2) * prob_A_k 3 * prob_A_k 4) := by
  sorry

end NUMINAMATH_CALUDE_ping_pong_probabilities_l556_55694


namespace NUMINAMATH_CALUDE_minimum_heat_for_piston_ejection_l556_55627

/-- The minimum amount of heat required to shoot a piston out of a cylinder -/
theorem minimum_heat_for_piston_ejection
  (l₁ : Real) (l₂ : Real) (M : Real) (S : Real) (v : Real) (p₀ : Real) (g : Real)
  (h₁ : l₁ = 0.1) -- 10 cm in meters
  (h₂ : l₂ = 0.15) -- 15 cm in meters
  (h₃ : M = 10) -- 10 kg
  (h₄ : S = 0.001) -- 10 cm² in m²
  (h₅ : v = 1) -- 1 mole
  (h₆ : p₀ = 10^5) -- 10⁵ Pa
  (h₇ : g = 10) -- 10 m/s²
  : ∃ Q : Real, Q = 127.5 ∧ Q ≥ 0 := by
  sorry

end NUMINAMATH_CALUDE_minimum_heat_for_piston_ejection_l556_55627


namespace NUMINAMATH_CALUDE_alice_added_nineteen_plates_l556_55686

/-- The number of plates Alice added before the tower fell -/
def additional_plates (initial : ℕ) (second_addition : ℕ) (total : ℕ) : ℕ :=
  total - (initial + second_addition)

/-- Theorem stating that Alice added 19 more plates before the tower fell -/
theorem alice_added_nineteen_plates : 
  additional_plates 27 37 83 = 19 := by
  sorry

end NUMINAMATH_CALUDE_alice_added_nineteen_plates_l556_55686


namespace NUMINAMATH_CALUDE_cut_scene_length_is_8_minutes_l556_55697

/-- Calculates the length of a cut scene given the original and final movie lengths -/
def cut_scene_length (original_length final_length : ℕ) : ℕ :=
  original_length - final_length

theorem cut_scene_length_is_8_minutes :
  cut_scene_length 60 52 = 8 := by
  sorry

end NUMINAMATH_CALUDE_cut_scene_length_is_8_minutes_l556_55697


namespace NUMINAMATH_CALUDE_sum_of_digits_11_pow_2003_l556_55678

theorem sum_of_digits_11_pow_2003 : ∃ n : ℕ, 
  11^2003 = 100 * n + 31 ∧ 3 + 1 = 4 := by sorry

end NUMINAMATH_CALUDE_sum_of_digits_11_pow_2003_l556_55678


namespace NUMINAMATH_CALUDE_taxi_driver_theorem_l556_55682

def driving_distances : List Int := [-5, 3, 6, -4, 7, -2]

def base_fare : Nat := 8
def extra_fare_per_km : Nat := 2
def base_distance : Nat := 3

def cumulative_distance (n : Nat) : Int :=
  (driving_distances.take n).sum

def trip_fare (distance : Int) : Nat :=
  base_fare + max 0 (distance.natAbs - base_distance) * extra_fare_per_km

def total_earnings : Nat :=
  (driving_distances.map trip_fare).sum

theorem taxi_driver_theorem :
  (cumulative_distance 4 = 0) ∧
  (cumulative_distance driving_distances.length = 5) ∧
  (total_earnings = 68) := by
  sorry

end NUMINAMATH_CALUDE_taxi_driver_theorem_l556_55682


namespace NUMINAMATH_CALUDE_percentage_of_wax_used_l556_55648

def original_wax_20oz : ℕ := 5
def original_wax_5oz : ℕ := 5
def original_wax_1oz : ℕ := 25
def new_candles : ℕ := 3
def new_candle_size : ℕ := 5

def total_original_wax : ℕ := original_wax_20oz * 20 + original_wax_5oz * 5 + original_wax_1oz * 1
def wax_used_for_new_candles : ℕ := new_candles * new_candle_size

theorem percentage_of_wax_used (total_original_wax wax_used_for_new_candles : ℕ) :
  (wax_used_for_new_candles : ℚ) / (total_original_wax : ℚ) * 100 = 10 :=
sorry

end NUMINAMATH_CALUDE_percentage_of_wax_used_l556_55648


namespace NUMINAMATH_CALUDE_midpoint_property_l556_55685

/-- Given two points P and Q in the plane, their midpoint R satisfies 3x + 2y = 39 --/
theorem midpoint_property (P Q R : ℝ × ℝ) : 
  P = (12, 9) → Q = (4, 6) → R = ((P.1 + Q.1) / 2, (P.2 + Q.2) / 2) →
  3 * R.1 + 2 * R.2 = 39 := by
  sorry

#check midpoint_property

end NUMINAMATH_CALUDE_midpoint_property_l556_55685


namespace NUMINAMATH_CALUDE_intersection_with_complement_l556_55673

def U : Finset ℕ := {0,1,2,3,4,5,6}
def A : Finset ℕ := {0,1,3,5}
def B : Finset ℕ := {1,2,4}

theorem intersection_with_complement :
  A ∩ (U \ B) = {0,3,5} := by sorry

end NUMINAMATH_CALUDE_intersection_with_complement_l556_55673


namespace NUMINAMATH_CALUDE_count_numbers_with_nine_is_1848_l556_55615

/-- Counts the number of integers between 1000 and 9000 with four distinct digits, including at least one '9' -/
def count_numbers_with_nine : ℕ := 
  let first_digit_nine := 9 * 8 * 7
  let nine_in_other_positions := 3 * 8 * 8 * 7
  first_digit_nine + nine_in_other_positions

/-- Theorem stating that the count of integers between 1000 and 9000 
    with four distinct digits, including at least one '9', is 1848 -/
theorem count_numbers_with_nine_is_1848 : 
  count_numbers_with_nine = 1848 := by sorry

end NUMINAMATH_CALUDE_count_numbers_with_nine_is_1848_l556_55615


namespace NUMINAMATH_CALUDE_digit_1234_is_4_l556_55655

/-- The number of digits in the representation of an integer -/
def numDigits (n : ℕ) : ℕ := sorry

/-- The nth digit in the sequence of concatenated integers from 1 to 500 -/
def nthDigit (n : ℕ) : ℕ := sorry

theorem digit_1234_is_4 :
  nthDigit 1234 = 4 := by sorry

end NUMINAMATH_CALUDE_digit_1234_is_4_l556_55655


namespace NUMINAMATH_CALUDE_marvelous_class_size_l556_55635

theorem marvelous_class_size :
  ∀ (girls : ℕ) (boys : ℕ) (jelly_beans : ℕ),
    -- Each girl received twice as many jelly beans as there were girls
    (2 * girls * girls +
    -- Each boy received three times as many jelly beans as there were boys
    3 * boys * boys = 
    -- Total jelly beans given out
    jelly_beans) →
    -- She brought 645 jelly beans and had 3 left
    (jelly_beans = 645 - 3) →
    -- The number of boys was three more than twice the number of girls
    (boys = 2 * girls + 3) →
    -- The total number of students
    (girls + boys = 18) := by
  sorry

end NUMINAMATH_CALUDE_marvelous_class_size_l556_55635


namespace NUMINAMATH_CALUDE_parallel_tangents_theorem_l556_55674

-- Define the curve C
def C (a b d : ℝ) (x : ℝ) : ℝ := a * x^3 + b * x^2 + d

-- Define the derivative of C
def C_derivative (a b : ℝ) (x : ℝ) : ℝ := 3 * a * x^2 + 2 * b * x

theorem parallel_tangents_theorem (a b d : ℝ) :
  C a b d 1 = 1 →  -- Point A(1,1) is on the curve
  C a b d (-1) = -3 →  -- Point B(-1,-3) is on the curve
  C_derivative a b 1 = C_derivative a b (-1) →  -- Tangents at A and B are parallel
  a^3 + b^2 + d = 7 := by
  sorry

end NUMINAMATH_CALUDE_parallel_tangents_theorem_l556_55674


namespace NUMINAMATH_CALUDE_forty_seventh_digit_is_six_l556_55689

def sequence_digit (n : ℕ) : ℕ :=
  let start := 90
  let digit_pos := n - 1
  let num_index := digit_pos / 2
  let in_num_pos := digit_pos % 2
  let current_num := start - num_index
  (current_num / 10^(1 - in_num_pos)) % 10

theorem forty_seventh_digit_is_six :
  sequence_digit 47 = 6 := by
  sorry

end NUMINAMATH_CALUDE_forty_seventh_digit_is_six_l556_55689


namespace NUMINAMATH_CALUDE_number_puzzle_l556_55692

theorem number_puzzle : ∃ x : ℚ, x = (3/5) * (2*x) + 238 ∧ x = 1190 := by
  sorry

end NUMINAMATH_CALUDE_number_puzzle_l556_55692


namespace NUMINAMATH_CALUDE_rooks_attack_after_knight_moves_l556_55611

/-- Represents a position on the chess board -/
structure Position :=
  (row : Fin 15)
  (col : Fin 15)

/-- Represents a knight's move -/
inductive KnightMove
  | move1 : KnightMove  -- represents +2,+1 or -2,-1
  | move2 : KnightMove  -- represents +2,-1 or -2,+1
  | move3 : KnightMove  -- represents +1,+2 or -1,-2
  | move4 : KnightMove  -- represents +1,-2 or -1,+2

/-- Applies a knight's move to a position -/
def applyKnightMove (p : Position) (m : KnightMove) : Position :=
  sorry

/-- Checks if two positions are in the same row or column -/
def sameRowOrColumn (p1 p2 : Position) : Prop :=
  p1.row = p2.row ∨ p1.col = p2.col

theorem rooks_attack_after_knight_moves 
  (initial_positions : Fin 15 → Position)
  (h_no_initial_attack : ∀ i j, i ≠ j → ¬(sameRowOrColumn (initial_positions i) (initial_positions j)))
  (moves : Fin 15 → KnightMove) :
  ∃ i j, i ≠ j ∧ sameRowOrColumn (applyKnightMove (initial_positions i) (moves i)) (applyKnightMove (initial_positions j) (moves j)) :=
sorry

end NUMINAMATH_CALUDE_rooks_attack_after_knight_moves_l556_55611


namespace NUMINAMATH_CALUDE_solve_equation_l556_55636

theorem solve_equation (x : ℝ) : 
  Real.sqrt ((3 / x) + 3) = 4 / 3 → x = -27 / 11 := by
  sorry

end NUMINAMATH_CALUDE_solve_equation_l556_55636


namespace NUMINAMATH_CALUDE_notebook_distribution_l556_55629

theorem notebook_distribution (C : ℕ) (N : ℕ) : 
  (N / C = C / 8) → 
  (N = 16 * (C / 2)) → 
  (N = 512) :=
by
  sorry

end NUMINAMATH_CALUDE_notebook_distribution_l556_55629


namespace NUMINAMATH_CALUDE_solution_approximation_l556_55626

/-- A linear function f. -/
noncomputable def f (x : ℝ) : ℝ := x

/-- The equation to be solved. -/
def equation (x : ℝ) : Prop :=
  f (x * 0.004) / 0.03 = 9.237333333333334

/-- The theorem stating that the solution to the equation is approximately 69.3. -/
theorem solution_approximation :
  ∃ x : ℝ, equation x ∧ abs (x - 69.3) < 0.001 :=
sorry

end NUMINAMATH_CALUDE_solution_approximation_l556_55626


namespace NUMINAMATH_CALUDE_smallest_layer_sugar_l556_55600

/-- Represents a three-layer cake with sugar requirements -/
structure ThreeLayerCake where
  smallest_layer : ℝ
  second_layer : ℝ
  third_layer : ℝ
  second_is_twice_first : second_layer = 2 * smallest_layer
  third_is_thrice_second : third_layer = 3 * second_layer
  third_layer_sugar : third_layer = 12

/-- Proves that the smallest layer of the cake requires 2 cups of sugar -/
theorem smallest_layer_sugar (cake : ThreeLayerCake) : cake.smallest_layer = 2 := by
  sorry

#check smallest_layer_sugar

end NUMINAMATH_CALUDE_smallest_layer_sugar_l556_55600


namespace NUMINAMATH_CALUDE_divisibility_property_l556_55668

theorem divisibility_property (p : ℕ) (h1 : p > 1) (h2 : Odd p) :
  ∃ k : ℤ, (p - 1 : ℤ) ^ ((p - 1) / 2) - 1 = (p - 2) * k := by
  sorry

end NUMINAMATH_CALUDE_divisibility_property_l556_55668


namespace NUMINAMATH_CALUDE_inequality_solution_and_geometric_mean_l556_55693

theorem inequality_solution_and_geometric_mean (a b m : ℝ) : 
  (∀ x, (x - 2) / (a * x + b) > 0 ↔ -1 < x ∧ x < 2) →
  m^2 = a * b →
  (3 * m^2 * a) / (a^3 + 2 * b^3) = 1 := by sorry

end NUMINAMATH_CALUDE_inequality_solution_and_geometric_mean_l556_55693


namespace NUMINAMATH_CALUDE_max_profit_at_initial_price_l556_55622

/-- Represents the daily profit function for a clothing store -/
def daily_profit (x : ℝ) : ℝ :=
  (30 - x) * (30 + x)

/-- Theorem stating that the maximum daily profit occurs at the initial selling price -/
theorem max_profit_at_initial_price :
  ∀ x : ℝ, daily_profit 0 ≥ daily_profit x :=
sorry

end NUMINAMATH_CALUDE_max_profit_at_initial_price_l556_55622


namespace NUMINAMATH_CALUDE_period_1989_points_count_l556_55624

-- Define the unit circle
def UnitCircle : Set ℂ := {z : ℂ | Complex.abs z = 1}

-- Define the function f
def f (m : ℕ) (z : ℂ) : ℂ := z ^ m

-- Define the set of period n points
def PeriodPoints (m : ℕ) (n : ℕ) : Set ℂ :=
  {z ∈ UnitCircle | (f m)^[n] z = z ∧ ∀ k < n, (f m)^[k] z ≠ z}

-- Theorem statement
theorem period_1989_points_count (m : ℕ) (h : m > 1) :
  (PeriodPoints m 1989).ncard = m^1989 - m^663 - m^153 - m^117 + m^51 + m^39 + m^9 - m^3 := by
  sorry

end NUMINAMATH_CALUDE_period_1989_points_count_l556_55624


namespace NUMINAMATH_CALUDE_find_r_value_l556_55617

theorem find_r_value (x y k : ℝ) (h : y^2 + 4*y + 4 + Real.sqrt (x + y + k) = 0) :
  let r := |x * y|
  r = 2 := by
  sorry

end NUMINAMATH_CALUDE_find_r_value_l556_55617


namespace NUMINAMATH_CALUDE_expression_equality_l556_55634

/-- Proof that the given expression K is equal to 80xyz(x^2 + y^2 + z^2) -/
theorem expression_equality (x y z : ℝ) :
  (x + y + z)^5 - (-x + y + z)^5 - (x - y + z)^5 - (x + y - z)^5 = 80 * x * y * z * (x^2 + y^2 + z^2) := by
  sorry

end NUMINAMATH_CALUDE_expression_equality_l556_55634


namespace NUMINAMATH_CALUDE_total_cost_star_wars_toys_l556_55643

/-- The total cost of Star Wars toys, including a lightsaber, given the cost of other toys -/
theorem total_cost_star_wars_toys (other_toys_cost : ℕ) : 
  other_toys_cost = 1000 → 
  (2 * other_toys_cost + other_toys_cost) = 3 * other_toys_cost := by
  sorry

#check total_cost_star_wars_toys

end NUMINAMATH_CALUDE_total_cost_star_wars_toys_l556_55643


namespace NUMINAMATH_CALUDE_min_value_of_exponential_sum_l556_55612

theorem min_value_of_exponential_sum (x y : ℝ) 
  (hx : x > 0) (hy : y > 0) (h_sum : 2 * x + y = 6) :
  (9 : ℝ)^x + 3^y ≥ 54 ∧ ∃ x y, x > 0 ∧ y > 0 ∧ 2 * x + y = 6 ∧ (9 : ℝ)^x + 3^y = 54 := by
  sorry

end NUMINAMATH_CALUDE_min_value_of_exponential_sum_l556_55612


namespace NUMINAMATH_CALUDE_simple_interest_rate_problem_l556_55654

/-- The simple interest rate problem -/
theorem simple_interest_rate_problem 
  (simple_interest : ℝ) 
  (principal : ℝ) 
  (time : ℝ) 
  (h1 : simple_interest = 16.32)
  (h2 : principal = 34)
  (h3 : time = 8)
  (h4 : simple_interest = principal * (rate / 100) * time) :
  rate = 6 := by
  sorry

end NUMINAMATH_CALUDE_simple_interest_rate_problem_l556_55654


namespace NUMINAMATH_CALUDE_weight_of_b_l556_55632

/-- Given the average weights of three people (a, b, c) and two pairs (a, b) and (b, c),
    prove that the weight of b is 31 kg. -/
theorem weight_of_b (a b c : ℝ) 
  (h1 : (a + b + c) / 3 = 45)  -- average weight of a, b, and c is 45 kg
  (h2 : (a + b) / 2 = 40)      -- average weight of a and b is 40 kg
  (h3 : (b + c) / 2 = 43) :    -- average weight of b and c is 43 kg
  b = 31 := by
  sorry

end NUMINAMATH_CALUDE_weight_of_b_l556_55632


namespace NUMINAMATH_CALUDE_hill_climbing_speed_l556_55605

/-- Proves that given a round trip with specified conditions, 
    the average speed for the upward journey is 1.125 km/h -/
theorem hill_climbing_speed 
  (up_time : ℝ) 
  (down_time : ℝ) 
  (avg_speed : ℝ) 
  (h1 : up_time = 4) 
  (h2 : down_time = 2) 
  (h3 : avg_speed = 1.5) : 
  (avg_speed * (up_time + down_time)) / (2 * up_time) = 1.125 := by
  sorry

#check hill_climbing_speed

end NUMINAMATH_CALUDE_hill_climbing_speed_l556_55605


namespace NUMINAMATH_CALUDE_worker_days_calculation_l556_55695

theorem worker_days_calculation (wages_group1 wages_group2 : ℚ)
  (workers_group1 workers_group2 : ℕ) (days_group2 : ℕ) :
  wages_group1 = 9450 →
  wages_group2 = 9975 →
  workers_group1 = 15 →
  workers_group2 = 19 →
  days_group2 = 5 →
  ∃ (days_group1 : ℕ),
    (wages_group1 / (workers_group1 * days_group1 : ℚ)) =
    (wages_group2 / (workers_group2 * days_group2 : ℚ)) ∧
    days_group1 = 6 :=
by sorry

end NUMINAMATH_CALUDE_worker_days_calculation_l556_55695


namespace NUMINAMATH_CALUDE_sunshine_orchard_pumpkins_l556_55652

/-- The number of pumpkins at Moonglow Orchard -/
def moonglow_pumpkins : ℕ := 14

/-- The number of pumpkins at Sunshine Orchard -/
def sunshine_pumpkins : ℕ := 3 * moonglow_pumpkins + 12

theorem sunshine_orchard_pumpkins : sunshine_pumpkins = 54 := by
  sorry

end NUMINAMATH_CALUDE_sunshine_orchard_pumpkins_l556_55652


namespace NUMINAMATH_CALUDE_complex_equation_solution_l556_55679

theorem complex_equation_solution (z : ℂ) : (Complex.I * z = 4 + 3 * Complex.I) → z = 3 - 4 * Complex.I := by
  sorry

end NUMINAMATH_CALUDE_complex_equation_solution_l556_55679


namespace NUMINAMATH_CALUDE_platform_length_l556_55687

/-- The length of a platform given train specifications -/
theorem platform_length (train_length : ℝ) (train_speed_kmph : ℝ) (crossing_time : ℝ) : 
  train_length = 120 →
  train_speed_kmph = 72 →
  crossing_time = 25 →
  (train_speed_kmph * 1000 / 3600) * crossing_time - train_length = 380 := by
  sorry


end NUMINAMATH_CALUDE_platform_length_l556_55687


namespace NUMINAMATH_CALUDE_flight_duration_sum_main_flight_theorem_l556_55633

/-- Represents a time with hours and minutes -/
structure Time where
  hours : ℕ
  minutes : ℕ
  valid : minutes < 60

/-- Represents a flight with departure and arrival times -/
structure Flight where
  departure : Time
  arrival : Time

def Flight.duration (f : Flight) : ℕ × ℕ :=
  sorry

theorem flight_duration_sum (f : Flight) (time_zone_diff : ℕ) (daylight_saving : ℕ) :
  let (h, m) := f.duration
  h + m = 32 :=
by
  sorry

/-- The main theorem proving the flight duration sum -/
theorem main_flight_theorem : ∃ (f : Flight) (time_zone_diff daylight_saving : ℕ),
  f.departure = ⟨7, 15, sorry⟩ ∧
  f.arrival = ⟨17, 40, sorry⟩ ∧
  time_zone_diff = 3 ∧
  daylight_saving = 1 ∧
  (let (h, m) := f.duration
   0 < m ∧ m < 60 ∧ h + m = 32) :=
by
  sorry

end NUMINAMATH_CALUDE_flight_duration_sum_main_flight_theorem_l556_55633


namespace NUMINAMATH_CALUDE_fraction_equality_l556_55672

theorem fraction_equality (a b c d : ℚ) 
  (h1 : a / b = 1 / 4)
  (h2 : c / d = 1 / 4)
  (h3 : b + d ≠ 0) :
  (a + 2 * c) / (2 * b + 4 * d) = 1 / 8 := by
sorry

end NUMINAMATH_CALUDE_fraction_equality_l556_55672


namespace NUMINAMATH_CALUDE_not_equivalent_polar_points_l556_55662

/-- A point in polar coordinates -/
structure PolarPoint where
  r : ℝ
  θ : ℝ

/-- Check if two polar points are equivalent -/
def equivalentPolarPoints (p1 p2 : PolarPoint) : Prop :=
  p1.r = p2.r ∧ ∃ k : ℤ, p1.θ = p2.θ + 2 * k * Real.pi

theorem not_equivalent_polar_points :
  ¬ equivalentPolarPoints ⟨2, 11 * Real.pi / 6⟩ ⟨2, Real.pi / 6⟩ := by
  sorry

end NUMINAMATH_CALUDE_not_equivalent_polar_points_l556_55662


namespace NUMINAMATH_CALUDE_square_roots_of_specific_integers_l556_55607

theorem square_roots_of_specific_integers : ∃ (m₁ m₂ : ℕ),
  m₁^2 = 170569 ∧
  m₂^2 = 175561 ∧
  m₁ = 413 ∧
  m₂ = 419 := by
sorry

end NUMINAMATH_CALUDE_square_roots_of_specific_integers_l556_55607


namespace NUMINAMATH_CALUDE_ranking_sequences_count_l556_55680

/-- Represents a player in the chess tournament -/
inductive Player : Type
| A : Player
| B : Player
| C : Player
| D : Player

/-- Represents a match between two players -/
structure Match :=
(player1 : Player)
(player2 : Player)

/-- Represents the tournament structure -/
structure Tournament :=
(initial_match1 : Match)
(initial_match2 : Match)
(winners_match : Match)
(losers_match : Match)
(third_place_match : Match)

/-- A function to calculate the number of possible ranking sequences -/
def count_ranking_sequences (t : Tournament) : Nat :=
  sorry

/-- The theorem stating that the number of possible ranking sequences is 8 -/
theorem ranking_sequences_count :
  ∀ t : Tournament, count_ranking_sequences t = 8 :=
sorry

end NUMINAMATH_CALUDE_ranking_sequences_count_l556_55680


namespace NUMINAMATH_CALUDE_banana_arrangements_count_l556_55646

/-- The number of distinct arrangements of the letters in "BANANA" -/
def banana_arrangements : ℕ := 60

/-- The total number of letters in "BANANA" -/
def total_letters : ℕ := 6

/-- The number of occurrences of 'A' in "BANANA" -/
def count_A : ℕ := 3

/-- The number of occurrences of 'N' in "BANANA" -/
def count_N : ℕ := 2

/-- The number of occurrences of 'B' in "BANANA" -/
def count_B : ℕ := 1

/-- Theorem stating that the number of distinct arrangements of the letters in "BANANA" is 60 -/
theorem banana_arrangements_count :
  banana_arrangements = (Nat.factorial total_letters) / ((Nat.factorial count_A) * (Nat.factorial count_N)) :=
by sorry

end NUMINAMATH_CALUDE_banana_arrangements_count_l556_55646


namespace NUMINAMATH_CALUDE_pentagonal_prism_diagonals_l556_55601

/-- A regular pentagonal prism -/
structure RegularPentagonalPrism where
  /-- The number of vertices on each base -/
  base_vertices : ℕ
  /-- The total number of vertices -/
  total_vertices : ℕ
  /-- The number of base vertices is 5 -/
  base_is_pentagon : base_vertices = 5
  /-- The total number of vertices is twice the number of base vertices -/
  total_is_double_base : total_vertices = 2 * base_vertices

/-- A diagonal in a regular pentagonal prism -/
def is_diagonal (prism : RegularPentagonalPrism) (v1 v2 : ℕ) : Prop :=
  v1 ≠ v2 ∧ 
  v1 < prism.total_vertices ∧ 
  v2 < prism.total_vertices ∧
  (v1 < prism.base_vertices ↔ v2 ≥ prism.base_vertices)

/-- The total number of diagonals in a regular pentagonal prism -/
def total_diagonals (prism : RegularPentagonalPrism) : ℕ :=
  (prism.base_vertices * prism.base_vertices)

theorem pentagonal_prism_diagonals (prism : RegularPentagonalPrism) : 
  total_diagonals prism = 10 := by
  sorry

end NUMINAMATH_CALUDE_pentagonal_prism_diagonals_l556_55601


namespace NUMINAMATH_CALUDE_rectangle_length_equals_6_3_l556_55638

-- Define the parameters
def triangle_base : ℝ := 7.2
def triangle_height : ℝ := 7
def rectangle_width : ℝ := 4

-- Define the theorem
theorem rectangle_length_equals_6_3 :
  let triangle_area := (triangle_base * triangle_height) / 2
  let rectangle_length := triangle_area / rectangle_width
  rectangle_length = 6.3 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_length_equals_6_3_l556_55638


namespace NUMINAMATH_CALUDE_negation_square_positive_l556_55675

theorem negation_square_positive :
  ¬(∀ x : ℝ, x^2 > 0) ↔ ∃ x : ℝ, x^2 ≤ 0 := by sorry

end NUMINAMATH_CALUDE_negation_square_positive_l556_55675


namespace NUMINAMATH_CALUDE_jacob_ladder_price_l556_55616

/-- The price per rung for Jacob's ladders -/
def price_per_rung : ℚ := 2

/-- The number of ladders with 50 rungs -/
def ladders_50 : ℕ := 10

/-- The number of ladders with 60 rungs -/
def ladders_60 : ℕ := 20

/-- The number of rungs per ladder in the first group -/
def rungs_per_ladder_50 : ℕ := 50

/-- The number of rungs per ladder in the second group -/
def rungs_per_ladder_60 : ℕ := 60

/-- The total cost for all ladders -/
def total_cost : ℚ := 3400

theorem jacob_ladder_price : 
  price_per_rung * (ladders_50 * rungs_per_ladder_50 + ladders_60 * rungs_per_ladder_60) = total_cost := by
  sorry

end NUMINAMATH_CALUDE_jacob_ladder_price_l556_55616


namespace NUMINAMATH_CALUDE_not_divisible_by_n_plus_4_l556_55647

theorem not_divisible_by_n_plus_4 (n : ℕ+) : ¬ ∃ k : ℤ, (n.val^2 + 8*n.val + 15 : ℤ) = k * (n.val + 4) := by
  sorry

end NUMINAMATH_CALUDE_not_divisible_by_n_plus_4_l556_55647


namespace NUMINAMATH_CALUDE_prob_not_sold_is_one_fifth_expected_profit_four_batches_l556_55604

-- Define the probability of not passing each round
def prob_fail_first : ℚ := 1 / 9
def prob_fail_second : ℚ := 1 / 10

-- Define the profit/loss values
def profit_if_sold : ℤ := 400
def loss_if_not_sold : ℤ := 800

-- Define the number of batches
def num_batches : ℕ := 4

-- Define the probability of a batch being sold
def prob_sold : ℚ := (1 - prob_fail_first) * (1 - prob_fail_second)

-- Define the probability of a batch not being sold
def prob_not_sold : ℚ := 1 - prob_sold

-- Define the expected profit for a single batch
def expected_profit_single : ℚ := prob_sold * profit_if_sold - prob_not_sold * loss_if_not_sold

-- Theorem: Probability of a batch not being sold is 1/5
theorem prob_not_sold_is_one_fifth : prob_not_sold = 1 / 5 := by sorry

-- Theorem: Expected profit from 4 batches is 640 yuan
theorem expected_profit_four_batches : num_batches * expected_profit_single = 640 := by sorry

end NUMINAMATH_CALUDE_prob_not_sold_is_one_fifth_expected_profit_four_batches_l556_55604


namespace NUMINAMATH_CALUDE_continuity_at_nine_l556_55623

def f (x : ℝ) : ℝ := 4 * x^2 + 4

theorem continuity_at_nine :
  ∀ ε > 0, ∃ δ > 0, ∀ x, |x - 9| < δ → |f x - f 9| < ε :=
by sorry

end NUMINAMATH_CALUDE_continuity_at_nine_l556_55623


namespace NUMINAMATH_CALUDE_complex_subtraction_simplification_l556_55613

theorem complex_subtraction_simplification :
  (4 - 3 * Complex.I) - (7 - 5 * Complex.I) = -3 + 2 * Complex.I := by
  sorry

end NUMINAMATH_CALUDE_complex_subtraction_simplification_l556_55613


namespace NUMINAMATH_CALUDE_player_one_wins_l556_55644

/-- Represents the number of coins a player can take -/
def ValidMove (player : ℕ) (coins : ℕ) : Prop :=
  match player with
  | 1 => coins % 2 = 1 ∧ 1 ≤ coins ∧ coins ≤ 99
  | 2 => coins % 2 = 0 ∧ 2 ≤ coins ∧ coins ≤ 100
  | _ => False

/-- The game state -/
structure GameState where
  coins : ℕ
  currentPlayer : ℕ

/-- A winning strategy for a player -/
def WinningStrategy (player : ℕ) : Prop :=
  ∀ (state : GameState), state.currentPlayer = player →
    ∃ (move : ℕ), ValidMove player move ∧
      (state.coins < move ∨
       ¬∃ (opponentMove : ℕ), ValidMove (3 - player) opponentMove ∧
         state.coins - move - opponentMove ≥ 0)

/-- The main theorem: Player 1 has a winning strategy -/
theorem player_one_wins : WinningStrategy 1 := by
  sorry

#check player_one_wins

end NUMINAMATH_CALUDE_player_one_wins_l556_55644


namespace NUMINAMATH_CALUDE_car_ownership_l556_55657

theorem car_ownership (total : ℕ) (neither : ℕ) (both : ℕ) (bike_only : ℕ) 
  (h1 : total = 90)
  (h2 : neither = 11)
  (h3 : both = 20)
  (h4 : bike_only = 35) :
  total - neither - bike_only = 44 :=
by sorry

end NUMINAMATH_CALUDE_car_ownership_l556_55657


namespace NUMINAMATH_CALUDE_two_digit_reverse_pythagoras_sum_l556_55690

theorem two_digit_reverse_pythagoras_sum : ∃ (x y n : ℕ), 
  (10 ≤ x ∧ x < 100) ∧ 
  (10 ≤ y ∧ y < 100) ∧ 
  (∃ (a b : ℕ), x = 10 * a + b ∧ y = 10 * b + a ∧ a < 10 ∧ b < 10) ∧
  x^2 + y^2 = n^2 ∧
  x + y + n = 132 := by
  sorry

end NUMINAMATH_CALUDE_two_digit_reverse_pythagoras_sum_l556_55690


namespace NUMINAMATH_CALUDE_fraction_of_108_l556_55653

theorem fraction_of_108 : (1 / 2 : ℚ) * (1 / 3 : ℚ) * (1 / 6 : ℚ) * 108 = 3 := by
  sorry

end NUMINAMATH_CALUDE_fraction_of_108_l556_55653


namespace NUMINAMATH_CALUDE_pencil_length_l556_55649

/-- Prove that given the conditions of the pen, rubber, and pencil lengths, the pencil is 12 cm long -/
theorem pencil_length (rubber pen pencil : ℝ) 
  (pen_rubber : pen = rubber + 3)
  (pencil_pen : pencil = pen + 2)
  (total_length : rubber + pen + pencil = 29) :
  pencil = 12 := by
  sorry

end NUMINAMATH_CALUDE_pencil_length_l556_55649


namespace NUMINAMATH_CALUDE_focus_of_ellipse_l556_55609

/-- The equation of the ellipse -/
def ellipse_equation (x y : ℝ) : Prop := x^2 / 4 + y^2 / 5 = 1

/-- Definition of a focus of an ellipse -/
def is_focus (a b c : ℝ) (x y : ℝ) : Prop :=
  x^2 + y^2 = c^2 ∧ a^2 = b^2 + c^2 ∧ a > b ∧ b > 0

/-- Theorem: (0, 1) is a focus of the given ellipse -/
theorem focus_of_ellipse :
  ∃ (a b c : ℝ), a^2 = 5 ∧ b^2 = 4 ∧ 
  (∀ (x y : ℝ), ellipse_equation x y ↔ x^2 / a^2 + y^2 / b^2 = 1) ∧
  is_focus a b c 0 1 :=
sorry

end NUMINAMATH_CALUDE_focus_of_ellipse_l556_55609


namespace NUMINAMATH_CALUDE_perpendicular_vector_scalar_l556_55698

/-- Given two vectors a and b in ℝ², prove that if a + xb is perpendicular to b, then x = -2/5 -/
theorem perpendicular_vector_scalar (a b : ℝ × ℝ) (x : ℝ) 
    (h1 : a = (3, 4))
    (h2 : b = (2, -1))
    (h3 : (a.1 + x * b.1, a.2 + x * b.2) • b = 0) :
  x = -2/5 := by
  sorry

#check perpendicular_vector_scalar

end NUMINAMATH_CALUDE_perpendicular_vector_scalar_l556_55698


namespace NUMINAMATH_CALUDE_age_difference_is_nine_l556_55645

/-- The age difference between Bella's brother and Bella -/
def ageDifference (bellasAge : ℕ) (totalAge : ℕ) : ℕ :=
  totalAge - bellasAge - bellasAge

/-- Proof that the age difference is 9 years -/
theorem age_difference_is_nine :
  ageDifference 5 19 = 9 := by
  sorry

end NUMINAMATH_CALUDE_age_difference_is_nine_l556_55645


namespace NUMINAMATH_CALUDE_temperature_conversion_l556_55639

theorem temperature_conversion (t k a : ℝ) : 
  t = 5/9 * (k - 32) + a * k → t = 20 → a = 3 → k = 10.625 := by
  sorry

end NUMINAMATH_CALUDE_temperature_conversion_l556_55639


namespace NUMINAMATH_CALUDE_max_added_value_max_at_two_thirds_verify_half_a_l556_55660

/-- The added value function for a car factory's production line -/
def f (a : ℝ) (x : ℝ) : ℝ := 8 * (a - x) * x^2

/-- Theorem stating the maximum value of the added value function -/
theorem max_added_value (a : ℝ) (h_a : a > 0) :
  ∃ (x : ℝ), x ∈ Set.Ioo 0 (4 * a / 5) ∧
    f a x = (32 / 27) * a^3 ∧
    ∀ (y : ℝ), y ∈ Set.Ioo 0 (4 * a / 5) → f a y ≤ (32 / 27) * a^3 :=
by sorry

/-- Theorem stating that the maximum occurs at x = 2a/3 -/
theorem max_at_two_thirds (a : ℝ) (h_a : a > 0) :
  f a (2 * a / 3) = (32 / 27) * a^3 :=
by sorry

/-- Theorem verifying that f(a/2) = a^3 -/
theorem verify_half_a (a : ℝ) :
  f a (a / 2) = a^3 :=
by sorry

end NUMINAMATH_CALUDE_max_added_value_max_at_two_thirds_verify_half_a_l556_55660


namespace NUMINAMATH_CALUDE_remaining_gift_card_value_l556_55602

def bestBuyCardValue : ℕ := 500
def walmartCardValue : ℕ := 200

def initialBestBuyCards : ℕ := 6
def initialWalmartCards : ℕ := 9

def sentBestBuyCards : ℕ := 1
def sentWalmartCards : ℕ := 2

theorem remaining_gift_card_value :
  (initialBestBuyCards - sentBestBuyCards) * bestBuyCardValue +
  (initialWalmartCards - sentWalmartCards) * walmartCardValue = 3900 := by
  sorry

end NUMINAMATH_CALUDE_remaining_gift_card_value_l556_55602


namespace NUMINAMATH_CALUDE_simplify_sqrt_three_l556_55696

theorem simplify_sqrt_three : 3 * Real.sqrt 3 - 2 * Real.sqrt 3 = Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_simplify_sqrt_three_l556_55696


namespace NUMINAMATH_CALUDE_square_of_fraction_equals_4088484_l556_55669

theorem square_of_fraction_equals_4088484 :
  ((2023^2 - 2023) / 2023)^2 = 4088484 := by
  sorry

end NUMINAMATH_CALUDE_square_of_fraction_equals_4088484_l556_55669


namespace NUMINAMATH_CALUDE_divisible_by_77_l556_55656

theorem divisible_by_77 (n : ℤ) : ∃ k : ℤ, n^18 - n^12 - n^8 + n^2 = 77 * k := by
  sorry

end NUMINAMATH_CALUDE_divisible_by_77_l556_55656


namespace NUMINAMATH_CALUDE_equation_solutions_l556_55603

theorem equation_solutions :
  let f : ℝ → ℝ → ℝ := λ x y => y^4 + 4*y^2*x - 11*y^2 + 4*x*y - 8*y + 8*x^2 - 40*x + 52
  ∀ x y : ℝ, f x y = 0 ↔ (x = 1 ∧ y = 2) ∨ (x = 5/2 ∧ y = -1) :=
by sorry

end NUMINAMATH_CALUDE_equation_solutions_l556_55603


namespace NUMINAMATH_CALUDE_only_324_and_648_have_property_l556_55620

-- Define a function to calculate the sum of digits
def sumOfDigits (n : ℕ) : ℕ :=
  if n < 10 then n else (n % 10) + sumOfDigits (n / 10)

-- Define the property we're looking for
def hasProperty (x : ℕ) : Prop :=
  x = 36 * sumOfDigits x

-- State the theorem
theorem only_324_and_648_have_property :
  ∀ x : ℕ, hasProperty x ↔ x = 324 ∨ x = 648 :=
sorry

end NUMINAMATH_CALUDE_only_324_and_648_have_property_l556_55620


namespace NUMINAMATH_CALUDE_expression_equality_l556_55608

theorem expression_equality : 10 * 0.2 * 5 * 0.1 + 5 = 6 := by
  sorry

end NUMINAMATH_CALUDE_expression_equality_l556_55608


namespace NUMINAMATH_CALUDE_minimum_additional_games_minimum_additional_games_is_146_l556_55671

theorem minimum_additional_games : ℕ → Prop :=
  fun n =>
    let initial_games : ℕ := 4
    let initial_lions_wins : ℕ := 3
    let initial_eagles_wins : ℕ := 1
    let total_games : ℕ := initial_games + n
    let total_eagles_wins : ℕ := initial_eagles_wins + n
    (total_eagles_wins : ℚ) / (total_games : ℚ) ≥ 98 / 100 ∧
    ∀ m : ℕ, m < n →
      let total_games_m : ℕ := initial_games + m
      let total_eagles_wins_m : ℕ := initial_eagles_wins + m
      (total_eagles_wins_m : ℚ) / (total_games_m : ℚ) < 98 / 100

theorem minimum_additional_games_is_146 : minimum_additional_games 146 := by
  sorry

#check minimum_additional_games_is_146

end NUMINAMATH_CALUDE_minimum_additional_games_minimum_additional_games_is_146_l556_55671


namespace NUMINAMATH_CALUDE_carlos_july_reading_l556_55684

/-- The number of books Carlos read in June -/
def june_books : ℕ := 42

/-- The number of books Carlos read in August -/
def august_books : ℕ := 30

/-- The total number of books Carlos needed to read -/
def total_books : ℕ := 100

/-- The number of books Carlos read in July -/
def july_books : ℕ := total_books - (june_books + august_books)

theorem carlos_july_reading :
  july_books = 28 := by sorry

end NUMINAMATH_CALUDE_carlos_july_reading_l556_55684


namespace NUMINAMATH_CALUDE_midpoint_path_and_intersection_l556_55606

/-- The path C traced by the midpoint of PQ, where P is (0, 4) and Q moves on x^2 + y^2 = 8 -/
def path_C (x y : ℝ) : Prop := x^2 + (y - 2)^2 = 2

/-- The line l that intersects path C -/
def line_l (k x y : ℝ) : Prop := y = k * x

/-- The condition for point E on line segment AB -/
def point_E_condition (m n : ℝ) (OA OB : ℝ) : Prop :=
  3 / (m^2 + n^2) = 1 / OA^2 + 1 / OB^2

/-- The main theorem -/
theorem midpoint_path_and_intersection :
  ∀ (x y k m n OA OB : ℝ),
  path_C x y →
  line_l k x y →
  point_E_condition m n OA OB →
  -Real.sqrt 6 / 2 < m →
  m < Real.sqrt 6 / 2 →
  m ≠ 0 →
  n = Real.sqrt (3 * m^2 + 9) / 3 :=
by sorry

end NUMINAMATH_CALUDE_midpoint_path_and_intersection_l556_55606


namespace NUMINAMATH_CALUDE_max_sales_and_profit_l556_55676

-- Define the sales volume function for the first 4 days
def sales_volume_early (x : ℝ) : ℝ := 20 * x + 80

-- Define the sales volume function for days 6 to 20
def sales_volume_late (x : ℝ) : ℝ := -x^2 + 50*x - 100

-- Define the selling price function for the first 5 days
def selling_price (x : ℝ) : ℝ := 2 * x + 28

-- Define the cost price
def cost_price : ℝ := 22

-- Define the profit function for days 1 to 5
def profit_early (x : ℝ) : ℝ := (selling_price x - cost_price) * sales_volume_early x

-- Define the profit function for days 6 to 20
def profit_late (x : ℝ) : ℝ := (28 - cost_price) * sales_volume_late x

theorem max_sales_and_profit :
  (∀ x ∈ Set.Icc 6 20, sales_volume_late x ≤ sales_volume_late 20) ∧
  sales_volume_late 20 = 500 ∧
  (∀ x ∈ Set.Icc 1 20, profit_early x ≤ profit_late 20 ∧ profit_late x ≤ profit_late 20) ∧
  profit_late 20 = 3000 := by
  sorry

end NUMINAMATH_CALUDE_max_sales_and_profit_l556_55676


namespace NUMINAMATH_CALUDE_peters_height_is_96_inches_l556_55630

/-- Given a tree height, tree shadow length, and Peter's shadow length,
    calculate Peter's height in inches. -/
def peters_height_inches (tree_height foot_to_inch : ℕ) 
                         (tree_shadow peter_shadow : ℚ) : ℚ :=
  (tree_height : ℚ) / tree_shadow * peter_shadow * foot_to_inch

/-- Theorem stating that Peter's height is 96 inches given the problem conditions. -/
theorem peters_height_is_96_inches :
  peters_height_inches 60 12 15 2 = 96 := by
  sorry

#eval peters_height_inches 60 12 15 2

end NUMINAMATH_CALUDE_peters_height_is_96_inches_l556_55630


namespace NUMINAMATH_CALUDE_f_is_quadratic_l556_55683

/-- Definition of a quadratic function -/
def is_quadratic (f : ℝ → ℝ) : Prop :=
  ∃ a b c : ℝ, a ≠ 0 ∧ ∀ x, f x = a * x^2 + b * x + c

/-- The function f(x) = 3x^2 - 6 -/
def f (x : ℝ) : ℝ := 3 * x^2 - 6

/-- Theorem: f is a quadratic function -/
theorem f_is_quadratic : is_quadratic f := by
  sorry

end NUMINAMATH_CALUDE_f_is_quadratic_l556_55683


namespace NUMINAMATH_CALUDE_election_votes_theorem_l556_55691

theorem election_votes_theorem (candidate1_percent : ℝ) (candidate2_percent : ℝ) 
  (candidate3_percent : ℝ) (candidate4_percent : ℝ) (candidate4_votes : ℕ) :
  candidate1_percent = 42 →
  candidate2_percent = 30 →
  candidate3_percent = 20 →
  candidate4_percent = 8 →
  candidate4_votes = 720 →
  (candidate1_percent + candidate2_percent + candidate3_percent + candidate4_percent = 100) →
  ∃ (total_votes : ℕ), total_votes = 9000 ∧ 
    (candidate4_percent / 100 * total_votes : ℝ) = candidate4_votes :=
by sorry

end NUMINAMATH_CALUDE_election_votes_theorem_l556_55691


namespace NUMINAMATH_CALUDE_exam_score_theorem_l556_55681

/-- Represents an examination scoring system and a student's performance --/
structure ExamScore where
  correct_score : ℕ      -- Marks awarded for each correct answer
  wrong_penalty : ℕ      -- Marks deducted for each wrong answer
  total_score : ℤ        -- Total score achieved
  correct_answers : ℕ    -- Number of correct answers
  total_questions : ℕ    -- Total number of questions attempted

/-- Theorem stating that given the exam conditions, the total questions attempted is 75 --/
theorem exam_score_theorem (exam : ExamScore) 
  (h1 : exam.correct_score = 4)
  (h2 : exam.wrong_penalty = 1)
  (h3 : exam.total_score = 125)
  (h4 : exam.correct_answers = 40) :
  exam.total_questions = 75 := by
  sorry


end NUMINAMATH_CALUDE_exam_score_theorem_l556_55681


namespace NUMINAMATH_CALUDE_farm_sheep_count_l556_55625

/-- Represents the farm with sheep and horses -/
structure Farm where
  sheep : ℕ
  racehorses : ℕ
  draft_horses : ℕ

/-- The ratio of sheep to total horses is 7:8 -/
def sheep_horse_ratio (f : Farm) : Prop :=
  7 * (f.racehorses + f.draft_horses) = 8 * f.sheep

/-- Total horse food consumption per day -/
def total_horse_food (f : Farm) : ℕ :=
  250 * f.racehorses + 300 * f.draft_horses

/-- There is 1/3 more racehorses than draft horses -/
def racehorse_draft_ratio (f : Farm) : Prop :=
  f.racehorses = f.draft_horses + (f.draft_horses / 3)

/-- The farm satisfies all given conditions -/
def valid_farm (f : Farm) : Prop :=
  sheep_horse_ratio f ∧
  total_horse_food f = 21000 ∧
  racehorse_draft_ratio f

theorem farm_sheep_count :
  ∃ f : Farm, valid_farm f ∧ f.sheep = 67 :=
sorry

end NUMINAMATH_CALUDE_farm_sheep_count_l556_55625


namespace NUMINAMATH_CALUDE_video_game_enemies_l556_55650

/-- The number of points earned per enemy defeated -/
def points_per_enemy : ℕ := 5

/-- The number of enemies left undefeated -/
def enemies_left : ℕ := 6

/-- The total points earned when all but 6 enemies are defeated -/
def total_points : ℕ := 10

/-- The total number of enemies in the level -/
def total_enemies : ℕ := 8

theorem video_game_enemies :
  total_enemies = (total_points / points_per_enemy) + enemies_left := by
  sorry

end NUMINAMATH_CALUDE_video_game_enemies_l556_55650


namespace NUMINAMATH_CALUDE_probability_triangle_or_circle_l556_55661

def total_figures : ℕ := 10
def triangle_count : ℕ := 3
def circle_count : ℕ := 3

theorem probability_triangle_or_circle :
  (triangle_count + circle_count : ℚ) / total_figures = 3 / 5 :=
sorry

end NUMINAMATH_CALUDE_probability_triangle_or_circle_l556_55661


namespace NUMINAMATH_CALUDE_division_remainder_l556_55670

theorem division_remainder (dividend : ℕ) (divisor : ℕ) (quotient : ℕ) (remainder : ℕ) : 
  dividend = 190 →
  divisor = 21 →
  quotient = 9 →
  dividend = divisor * quotient + remainder →
  remainder = 1 := by
sorry

end NUMINAMATH_CALUDE_division_remainder_l556_55670


namespace NUMINAMATH_CALUDE_right_triangle_hypotenuse_l556_55677

theorem right_triangle_hypotenuse (a b c : ℝ) : 
  a = 15 → b = 36 → c^2 = a^2 + b^2 → c = 39 := by sorry

end NUMINAMATH_CALUDE_right_triangle_hypotenuse_l556_55677


namespace NUMINAMATH_CALUDE_find_number_l556_55663

theorem find_number : ∃ x : ℤ, 4 * x + 100 = 4100 := by
  sorry

end NUMINAMATH_CALUDE_find_number_l556_55663


namespace NUMINAMATH_CALUDE_james_total_score_l556_55658

/-- Calculates the total points scored by James in a basketball game -/
def total_points (field_goals three_pointers two_pointers free_throws : ℕ) : ℕ :=
  field_goals * 3 + three_pointers * 2 + two_pointers * 2 + free_throws * 1

theorem james_total_score :
  total_points 13 0 20 5 = 84 := by
  sorry

#eval total_points 13 0 20 5

end NUMINAMATH_CALUDE_james_total_score_l556_55658


namespace NUMINAMATH_CALUDE_trivia_game_score_l556_55699

/-- The final score of a trivia game given the scores of three rounds -/
def final_score (round1 : Int) (round2 : Int) (round3 : Int) : Int :=
  round1 + round2 + round3

/-- Theorem: Given the scores from three rounds of a trivia game (16, 33, and -48),
    the final score is equal to 1. -/
theorem trivia_game_score :
  final_score 16 33 (-48) = 1 := by
  sorry

end NUMINAMATH_CALUDE_trivia_game_score_l556_55699


namespace NUMINAMATH_CALUDE_solution_satisfies_equation_l556_55659

theorem solution_satisfies_equation (a b c : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0) 
  (hab : a ≠ b) (hbc : b ≠ c) (hac : a ≠ c) :
  let x := (a^2 - b^2) / (2*a)
  (x^2 + b^2 + c^2) = ((a - x)^2 + c^2) :=
by sorry

end NUMINAMATH_CALUDE_solution_satisfies_equation_l556_55659


namespace NUMINAMATH_CALUDE_arithmetic_geometric_inequality_l556_55614

/-- Arithmetic sequence -/
def arithmetic_sequence (a₁ d : ℝ) (n : ℕ) : ℝ := a₁ + (n - 1) * d

/-- Geometric sequence -/
def geometric_sequence (b₁ q : ℝ) (n : ℕ) : ℝ := b₁ * q^(n - 1)

theorem arithmetic_geometric_inequality (b₁ q : ℝ) (m : ℕ) 
  (h₁ : b₁ > 0) 
  (h₂ : m > 0) 
  (h₃ : 1 < q) 
  (h₄ : q < (2 : ℝ)^(1 / m)) :
  ∃ d : ℝ, ∀ n : ℕ, 2 ≤ n ∧ n ≤ m + 1 → 
    |arithmetic_sequence b₁ d n - geometric_sequence b₁ q n| ≤ b₁ ∧
    b₁ * (q^m - 2) / m ≤ d ∧ d ≤ b₁ * q^m / m :=
sorry

end NUMINAMATH_CALUDE_arithmetic_geometric_inequality_l556_55614


namespace NUMINAMATH_CALUDE_inequality_implies_range_l556_55640

theorem inequality_implies_range (a : ℝ) : 
  (∀ x : ℝ, |x - 1| + |2*x + 2| ≥ a^2 + (1/2)*a + 2) → 
  -1/2 ≤ a ∧ a ≤ 0 := by
sorry

end NUMINAMATH_CALUDE_inequality_implies_range_l556_55640

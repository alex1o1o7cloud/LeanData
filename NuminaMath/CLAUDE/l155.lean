import Mathlib

namespace NUMINAMATH_CALUDE_quadratic_function_inequality_max_l155_15551

theorem quadratic_function_inequality_max (a b c : ℝ) :
  (∀ x : ℝ, a * x^2 + b * x + c ≤ 2 * a * x + b) →
  (∃ M : ℝ, M = 2 * Real.sqrt 2 - 2 ∧
    (∀ a b c : ℝ, b^2 / (a^2 + c^2) ≤ M) ∧
    (∃ a b c : ℝ, b^2 / (a^2 + c^2) = M)) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_function_inequality_max_l155_15551


namespace NUMINAMATH_CALUDE_triathlon_speeds_correct_l155_15569

/-- Represents the minimum speeds required for Maria to complete the triathlon within the given time limit. -/
def triathlon_speeds (swim_dist : ℝ) (cycle_dist : ℝ) (run_dist : ℝ) (time_limit : ℝ) : ℝ × ℝ × ℝ :=
  let swim_speed : ℝ := 60
  let run_speed : ℝ := 3 * swim_speed
  let cycle_speed : ℝ := 2.5 * run_speed
  (swim_speed, cycle_speed, run_speed)

/-- Theorem stating that the calculated speeds are correct for the given triathlon conditions. -/
theorem triathlon_speeds_correct 
  (swim_dist : ℝ) (cycle_dist : ℝ) (run_dist : ℝ) (time_limit : ℝ)
  (h_swim : swim_dist = 800)
  (h_cycle : cycle_dist = 20000)
  (h_run : run_dist = 4000)
  (h_time : time_limit = 80) :
  let (swim_speed, cycle_speed, run_speed) := triathlon_speeds swim_dist cycle_dist run_dist time_limit
  swim_speed = 60 ∧ cycle_speed = 450 ∧ run_speed = 180 ∧
  swim_dist / swim_speed + cycle_dist / cycle_speed + run_dist / run_speed ≤ time_limit :=
by sorry

#check triathlon_speeds_correct

end NUMINAMATH_CALUDE_triathlon_speeds_correct_l155_15569


namespace NUMINAMATH_CALUDE_boat_upstream_speed_l155_15518

/-- Calculates the upstream speed of a boat given its still water speed and downstream speed -/
def upstream_speed (still_water_speed downstream_speed : ℝ) : ℝ :=
  2 * still_water_speed - downstream_speed

/-- Theorem: Given a boat with a speed of 8.5 km/hr in still water and a downstream speed of 13 km/hr, its upstream speed is 4 km/hr -/
theorem boat_upstream_speed :
  upstream_speed 8.5 13 = 4 := by
  sorry

end NUMINAMATH_CALUDE_boat_upstream_speed_l155_15518


namespace NUMINAMATH_CALUDE_base10_to_base12_144_l155_15542

/-- Converts a digit to its base 12 representation -/
def toBase12Digit (n : ℕ) : String :=
  if n < 10 then toString n
  else if n = 10 then "A"
  else if n = 11 then "B"
  else ""

/-- Converts a number from base 10 to base 12 -/
def toBase12 (n : ℕ) : String :=
  let d1 := n / 12
  let d0 := n % 12
  toBase12Digit d1 ++ toBase12Digit d0

theorem base10_to_base12_144 :
  toBase12 144 = "B10" := by sorry

end NUMINAMATH_CALUDE_base10_to_base12_144_l155_15542


namespace NUMINAMATH_CALUDE_gcd_factorial_seven_eight_l155_15531

theorem gcd_factorial_seven_eight : Nat.gcd (Nat.factorial 7) (Nat.factorial 8) = Nat.factorial 7 := by
  sorry

end NUMINAMATH_CALUDE_gcd_factorial_seven_eight_l155_15531


namespace NUMINAMATH_CALUDE_complex_number_calculation_l155_15599

/-- Given the complex number i where i^2 = -1, prove that (1+i)(1-i)+(-1+i) = 1+i -/
theorem complex_number_calculation : ∀ i : ℂ, i^2 = -1 → (1+i)*(1-i)+(-1+i) = 1+i := by
  sorry

end NUMINAMATH_CALUDE_complex_number_calculation_l155_15599


namespace NUMINAMATH_CALUDE_arithmetic_geometric_sequence_l155_15556

/-- An arithmetic sequence with common difference 2 -/
def arithmetic_seq (a : ℕ → ℝ) : Prop :=
  ∀ n, a (n + 1) = a n + 2

/-- a_1, a_2, a_5 form a geometric sequence -/
def geometric_subseq (a : ℕ → ℝ) : Prop :=
  (a 2) ^ 2 = a 1 * a 5

theorem arithmetic_geometric_sequence (a : ℕ → ℝ) 
  (h_arith : arithmetic_seq a) (h_geom : geometric_subseq a) : 
  a 2 = 3 := by
sorry

end NUMINAMATH_CALUDE_arithmetic_geometric_sequence_l155_15556


namespace NUMINAMATH_CALUDE_range_of_k_for_trigonometric_equation_l155_15565

theorem range_of_k_for_trigonometric_equation :
  ∀ k : ℝ, (∃ x : ℝ, x ∈ Set.Icc 0 (π/2) ∧ 
    Real.sqrt 3 * Real.sin (2*x) + Real.cos (2*x) = k + 1) ↔ 
  k ∈ Set.Icc (-2) 1 := by
sorry

end NUMINAMATH_CALUDE_range_of_k_for_trigonometric_equation_l155_15565


namespace NUMINAMATH_CALUDE_magnitude_of_z_l155_15505

theorem magnitude_of_z (z : ℂ) (h : z * (1 - 2*Complex.I) = 4 + 2*Complex.I) : 
  Complex.abs z = 2 := by
  sorry

end NUMINAMATH_CALUDE_magnitude_of_z_l155_15505


namespace NUMINAMATH_CALUDE_negation_equivalence_l155_15567

theorem negation_equivalence :
  (¬ ∃ x : ℝ, x^2 + 1 > 3*x) ↔ (∀ x : ℝ, x^2 + 1 ≤ 3*x) := by sorry

end NUMINAMATH_CALUDE_negation_equivalence_l155_15567


namespace NUMINAMATH_CALUDE_camel_cost_l155_15561

theorem camel_cost (camel horse ox elephant : ℕ → ℚ) 
  (h1 : 10 * camel 1 = 24 * horse 1)
  (h2 : 16 * horse 1 = 4 * ox 1)
  (h3 : 6 * ox 1 = 4 * elephant 1)
  (h4 : 10 * elephant 1 = 120000) :
  camel 1 = 4800 := by
  sorry

end NUMINAMATH_CALUDE_camel_cost_l155_15561


namespace NUMINAMATH_CALUDE_concentric_circles_radii_difference_l155_15566

theorem concentric_circles_radii_difference
  (r R : ℝ)
  (h_positive : r > 0)
  (h_ratio : π * R^2 / (π * r^2) = 4) :
  R - r = r :=
sorry

end NUMINAMATH_CALUDE_concentric_circles_radii_difference_l155_15566


namespace NUMINAMATH_CALUDE_f_of_one_plus_g_of_two_l155_15549

def f (x : ℝ) : ℝ := 2 * x - 3

def g (x : ℝ) : ℝ := x + 1

theorem f_of_one_plus_g_of_two : f (1 + g 2) = 5 := by sorry

end NUMINAMATH_CALUDE_f_of_one_plus_g_of_two_l155_15549


namespace NUMINAMATH_CALUDE_diagonal_game_winner_l155_15527

/-- Represents a player in the game -/
inductive Player
| First
| Second

/-- Represents the outcome of the game -/
inductive Outcome
| FirstPlayerWins
| SecondPlayerWins

/-- The number of diagonals in a polygon with s sides -/
def num_diagonals (s : ℕ) : ℕ := s * (s - 3) / 2

/-- The winner of the diagonal drawing game in a (2n+1)-gon -/
def winner (n : ℕ) : Outcome :=
  if n % 2 = 0 then Outcome.FirstPlayerWins else Outcome.SecondPlayerWins

/-- The main theorem about the winner of the diagonal drawing game -/
theorem diagonal_game_winner (n : ℕ) (h : n > 1) :
  winner n = (if n % 2 = 0 then Outcome.FirstPlayerWins else Outcome.SecondPlayerWins) :=
sorry

end NUMINAMATH_CALUDE_diagonal_game_winner_l155_15527


namespace NUMINAMATH_CALUDE_arithmetic_sequence_ninth_term_l155_15547

/-- Given an arithmetic sequence where the 3rd term is 23 and the 5th term is 43,
    the 9th term is 83. -/
theorem arithmetic_sequence_ninth_term
  (a : ℕ → ℝ)  -- The arithmetic sequence
  (h1 : a 3 = 23)  -- The 3rd term is 23
  (h2 : a 5 = 43)  -- The 5th term is 43
  (h_arith : ∀ n : ℕ, a (n + 1) - a n = a 2 - a 1)  -- The sequence is arithmetic
  : a 9 = 83 :=
by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_ninth_term_l155_15547


namespace NUMINAMATH_CALUDE_factors_of_36_l155_15564

theorem factors_of_36 : Nat.card (Nat.divisors 36) = 9 := by
  sorry

end NUMINAMATH_CALUDE_factors_of_36_l155_15564


namespace NUMINAMATH_CALUDE_line_through_three_points_l155_15538

/-- Given a line passing through points (4, 10), (-3, m), and (-12, 5), prove that m = 125/16 -/
theorem line_through_three_points (m : ℚ) : 
  (let slope1 := (m - 10) / (-7 : ℚ)
   let slope2 := (5 - m) / (-9 : ℚ)
   slope1 = slope2) →
  m = 125 / 16 := by
sorry

end NUMINAMATH_CALUDE_line_through_three_points_l155_15538


namespace NUMINAMATH_CALUDE_ellipse_minor_axis_length_l155_15587

/-- Given an ellipse with minimum distance 5 and maximum distance 15 from a point on the ellipse to a focus, 
    the length of its minor axis is 10√3. -/
theorem ellipse_minor_axis_length (min_dist max_dist : ℝ) (h1 : min_dist = 5) (h2 : max_dist = 15) :
  let a := (max_dist + min_dist) / 2
  let c := (max_dist - min_dist) / 2
  let b := Real.sqrt (a^2 - c^2)
  2 * b = 10 * Real.sqrt 3 := by sorry

end NUMINAMATH_CALUDE_ellipse_minor_axis_length_l155_15587


namespace NUMINAMATH_CALUDE_equation_solution_l155_15522

theorem equation_solution : ∃ x : ℝ, 1 - 1 / ((1 - x)^3) = 1 / (1 - x) ∧ x = 1 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l155_15522


namespace NUMINAMATH_CALUDE_initial_salmon_count_l155_15530

theorem initial_salmon_count (current_count : ℕ) (increase_factor : ℕ) (initial_count : ℕ) : 
  current_count = 5500 →
  increase_factor = 10 →
  current_count = (increase_factor + 1) * initial_count →
  initial_count = 550 := by
  sorry

end NUMINAMATH_CALUDE_initial_salmon_count_l155_15530


namespace NUMINAMATH_CALUDE_select_and_order_two_from_five_eq_twenty_l155_15533

/-- The number of ways to select and order 2 items from a set of 5 distinct items -/
def select_and_order_two_from_five : ℕ :=
  5 * 4

/-- Theorem: The number of ways to select and order 2 items from a set of 5 distinct items is 20 -/
theorem select_and_order_two_from_five_eq_twenty :
  select_and_order_two_from_five = 20 := by
  sorry

end NUMINAMATH_CALUDE_select_and_order_two_from_five_eq_twenty_l155_15533


namespace NUMINAMATH_CALUDE_dividend_calculation_l155_15519

theorem dividend_calculation (remainder quotient divisor : ℕ) 
  (h1 : remainder = 1)
  (h2 : quotient = 54)
  (h3 : divisor = 4) :
  divisor * quotient + remainder = 217 := by
  sorry

end NUMINAMATH_CALUDE_dividend_calculation_l155_15519


namespace NUMINAMATH_CALUDE_complete_square_quadratic_l155_15554

theorem complete_square_quadratic (x : ℝ) : 
  (∃ c d : ℝ, x^2 + 6*x - 3 = 0 ↔ (x + c)^2 = d) → 
  (∃ c : ℝ, (x + c)^2 = 12) :=
by sorry

end NUMINAMATH_CALUDE_complete_square_quadratic_l155_15554


namespace NUMINAMATH_CALUDE_sum_to_n_equals_91_l155_15574

theorem sum_to_n_equals_91 : ∃ n : ℕ, n * (n + 1) / 2 = 91 ∧ n = 13 := by
  sorry

end NUMINAMATH_CALUDE_sum_to_n_equals_91_l155_15574


namespace NUMINAMATH_CALUDE_anime_watching_problem_l155_15562

/-- The number of days from today to April 1, 2023 (exclusive) -/
def days_to_april_1 : ℕ := sorry

/-- The total number of episodes in the anime series -/
def total_episodes : ℕ := sorry

/-- Theorem stating the solution to the anime watching problem -/
theorem anime_watching_problem :
  (total_episodes - 2 * days_to_april_1 = 215) ∧
  (total_episodes - 5 * days_to_april_1 = 50) →
  (days_to_april_1 = 55 ∧ total_episodes = 325) :=
by sorry

end NUMINAMATH_CALUDE_anime_watching_problem_l155_15562


namespace NUMINAMATH_CALUDE_power_mod_seventeen_l155_15511

theorem power_mod_seventeen : 7^2023 % 17 = 15 := by sorry

end NUMINAMATH_CALUDE_power_mod_seventeen_l155_15511


namespace NUMINAMATH_CALUDE_smallest_base_for_mnmn_cube_mnmn_cube_in_base_seven_smallest_base_is_seven_l155_15523

def is_mnmn (n : ℕ) (b : ℕ) : Prop :=
  ∃ m n : ℕ, m < b ∧ n < b ∧ n = m * (b^3 + b) + n * (b^2 + 1)

def is_cube (n : ℕ) : Prop :=
  ∃ k : ℕ, n = k^3

theorem smallest_base_for_mnmn_cube :
  ∀ b : ℕ, b > 1 →
    (∃ n : ℕ, is_mnmn n b ∧ is_cube n) →
    b ≥ 7 :=
by sorry

theorem mnmn_cube_in_base_seven :
  ∃ n : ℕ, is_mnmn n 7 ∧ is_cube n :=
by sorry

theorem smallest_base_is_seven :
  (∀ b : ℕ, b > 1 → b < 7 → ¬∃ n : ℕ, is_mnmn n b ∧ is_cube n) ∧
  (∃ n : ℕ, is_mnmn n 7 ∧ is_cube n) :=
by sorry

end NUMINAMATH_CALUDE_smallest_base_for_mnmn_cube_mnmn_cube_in_base_seven_smallest_base_is_seven_l155_15523


namespace NUMINAMATH_CALUDE_jelly_bean_probability_l155_15592

theorem jelly_bean_probability (p_red p_orange p_green p_yellow : ℝ) :
  p_red = 0.15 →
  p_orange = 0.35 →
  p_green = 0.25 →
  p_red + p_orange + p_green + p_yellow = 1 →
  p_yellow = 0.25 := by
sorry

end NUMINAMATH_CALUDE_jelly_bean_probability_l155_15592


namespace NUMINAMATH_CALUDE_panda_babies_count_l155_15534

def total_couples : ℕ := 100
def young_percentage : ℚ := 1/5
def adult_percentage : ℚ := 3/5
def old_percentage : ℚ := 1/5

def young_pregnancy_chance : ℚ := 2/5
def adult_pregnancy_chance : ℚ := 1/4
def old_pregnancy_chance : ℚ := 1/10

def average_babies_per_pregnancy : ℚ := 3/2

def young_babies : ℕ := 12
def adult_babies : ℕ := 22
def old_babies : ℕ := 3

theorem panda_babies_count :
  young_babies + adult_babies + old_babies = 37 :=
by sorry

end NUMINAMATH_CALUDE_panda_babies_count_l155_15534


namespace NUMINAMATH_CALUDE_train_length_train_length_alt_l155_15578

/-- The length of a train given its speed and time to cross a pole -/
theorem train_length (speed : ℝ) (time : ℝ) : 
  speed = 72 → time = 9 → speed * time * (5 / 18) = 180 := by
  sorry

/-- Alternative formulation using more basic definitions -/
theorem train_length_alt (speed_kmh : ℝ) (time_s : ℝ) (length_m : ℝ) : 
  speed_kmh = 72 → time_s = 9 → 
  length_m = speed_kmh * (1000 / 3600) * time_s →
  length_m = 180 := by
  sorry

end NUMINAMATH_CALUDE_train_length_train_length_alt_l155_15578


namespace NUMINAMATH_CALUDE_circle_bounds_l155_15590

theorem circle_bounds (x y : ℝ) : 
  x^2 + (y - 1)^2 = 1 →
  (-Real.sqrt 3 / 3 ≤ (y - 1) / (x - 2) ∧ (y - 1) / (x - 2) ≤ Real.sqrt 3 / 3) ∧
  (1 - Real.sqrt 5 ≤ 2*x + y ∧ 2*x + y ≤ 1 + Real.sqrt 5) := by
  sorry

end NUMINAMATH_CALUDE_circle_bounds_l155_15590


namespace NUMINAMATH_CALUDE_complex_modulus_problem_l155_15579

theorem complex_modulus_problem (z : ℂ) (h : (1 + Complex.I) * z = 3 + Complex.I) :
  Complex.abs z = Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_complex_modulus_problem_l155_15579


namespace NUMINAMATH_CALUDE_quadratic_inequality_implies_a_geq_one_l155_15501

theorem quadratic_inequality_implies_a_geq_one (a : ℝ) :
  (∀ x : ℝ, a * x^2 + 2 * x + a ≥ 0) → a ≥ 1 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_inequality_implies_a_geq_one_l155_15501


namespace NUMINAMATH_CALUDE_point_in_second_quadrant_l155_15553

def second_quadrant (x y : ℝ) : Prop := x < 0 ∧ y > 0

theorem point_in_second_quadrant (m : ℝ) :
  second_quadrant (-1) (m^2 + 1) := by
  sorry

end NUMINAMATH_CALUDE_point_in_second_quadrant_l155_15553


namespace NUMINAMATH_CALUDE_isosceles_triangle_base_length_l155_15524

/-- An isosceles triangle with perimeter 16 and one side of length 6 has a base of either 6 or 4. -/
theorem isosceles_triangle_base_length (a b : ℝ) : 
  a > 0 → b > 0 → 
  a + b + b = 16 → 
  (a = 6 ∨ b = 6) → 
  (a = 6 ∧ b = 5) ∨ (a = 4 ∧ b = 6) :=
by sorry

end NUMINAMATH_CALUDE_isosceles_triangle_base_length_l155_15524


namespace NUMINAMATH_CALUDE_arithmetic_geometric_sequence_ratio_l155_15516

theorem arithmetic_geometric_sequence_ratio (a : ℕ → ℚ) (d : ℚ) (h1 : d ≠ 0) 
  (h2 : ∀ n : ℕ, a (n + 1) = a n + d)  -- arithmetic sequence definition
  (h3 : (a 3 - a 1) * (a 9 - a 3) = (a 3 - a 1)^2) :  -- geometric sequence condition
  (a 1 + a 3 + a 9) / (a 2 + a 4 + a 10) = 13/16 := by
sorry

end NUMINAMATH_CALUDE_arithmetic_geometric_sequence_ratio_l155_15516


namespace NUMINAMATH_CALUDE_impossibleToTileModifiedChessboard_l155_15525

/-- Represents a square on the chessboard -/
inductive Square
| Black
| White

/-- Represents the chessboard -/
def Chessboard := Array (Array Square)

/-- Creates a standard 8x8 chessboard -/
def createStandardChessboard : Chessboard :=
  sorry

/-- Removes the top-left and bottom-right squares from the chessboard -/
def removeCornerSquares (board : Chessboard) : Chessboard :=
  sorry

/-- Counts the number of black and white squares on the chessboard -/
def countSquares (board : Chessboard) : (Nat × Nat) :=
  sorry

/-- Represents a domino placement on the chessboard -/
structure DominoPlacement where
  position1 : Nat × Nat
  position2 : Nat × Nat

/-- Checks if a domino placement is valid on the given chessboard -/
def isValidPlacement (board : Chessboard) (placement : DominoPlacement) : Bool :=
  sorry

/-- Main theorem: It's impossible to tile the modified chessboard with dominos -/
theorem impossibleToTileModifiedChessboard :
  ∀ (placements : List DominoPlacement),
    let board := removeCornerSquares createStandardChessboard
    let (blackCount, whiteCount) := countSquares board
    (blackCount ≠ whiteCount) ∧
    (∀ p ∈ placements, isValidPlacement board p) →
    placements.length < 31 :=
  sorry

end NUMINAMATH_CALUDE_impossibleToTileModifiedChessboard_l155_15525


namespace NUMINAMATH_CALUDE_sphere_radius_when_area_equals_volume_l155_15540

theorem sphere_radius_when_area_equals_volume (r : ℝ) (h : r > 0) :
  (4 * Real.pi * r^2) = (4/3 * Real.pi * r^3) → r = 3 := by
  sorry

end NUMINAMATH_CALUDE_sphere_radius_when_area_equals_volume_l155_15540


namespace NUMINAMATH_CALUDE_milk_purchase_theorem_l155_15576

/-- The cost of one bag of milk in yuan -/
def milk_cost : ℕ := 3

/-- The number of bags paid for in the offer -/
def offer_paid : ℕ := 5

/-- The total number of bags received in the offer -/
def offer_total : ℕ := 6

/-- The amount of money mom has in yuan -/
def mom_money : ℕ := 20

/-- The maximum number of bags mom can buy -/
def max_bags : ℕ := 7

/-- The amount of money left after buying the maximum number of bags -/
def money_left : ℕ := 2

theorem milk_purchase_theorem :
  milk_cost * (offer_paid * (max_bags / offer_total) + max_bags % offer_total) ≤ mom_money ∧
  mom_money - milk_cost * (offer_paid * (max_bags / offer_total) + max_bags % offer_total) = money_left :=
by sorry

end NUMINAMATH_CALUDE_milk_purchase_theorem_l155_15576


namespace NUMINAMATH_CALUDE_largest_valid_number_l155_15583

def is_valid_number (n : ℕ) : Prop :=
  100 ≤ n ∧ n < 1000 ∧
  (n / 100 = 1 ∨ n / 100 = 7 ∨ n / 100 = 0) ∧
  ((n / 10) % 10 = 1 ∨ (n / 10) % 10 = 7 ∨ (n / 10) % 10 = 0) ∧
  (n % 10 = 1 ∨ n % 10 = 7 ∨ n % 10 = 0) ∧
  (n / 100 ≠ (n / 10) % 10) ∧
  (n / 100 ≠ n % 10) ∧
  ((n / 10) % 10 ≠ n % 10)

theorem largest_valid_number :
  ∀ n : ℕ, is_valid_number n → n ≤ 710 :=
sorry

end NUMINAMATH_CALUDE_largest_valid_number_l155_15583


namespace NUMINAMATH_CALUDE_cube_root_27_div_fourth_root_16_l155_15550

theorem cube_root_27_div_fourth_root_16 : (27 ^ (1/3)) / (16 ^ (1/4)) = 3/2 := by
  sorry

end NUMINAMATH_CALUDE_cube_root_27_div_fourth_root_16_l155_15550


namespace NUMINAMATH_CALUDE_volume_of_second_cylinder_l155_15528

/-- Given two cylinders with the same height and radii in the ratio 1:3, 
    if the volume of the first cylinder is 40 cc, 
    then the volume of the second cylinder is 360 cc. -/
theorem volume_of_second_cylinder 
  (h : ℝ) -- height of both cylinders
  (r₁ : ℝ) -- radius of the first cylinder
  (r₂ : ℝ) -- radius of the second cylinder
  (h_positive : h > 0)
  (r₁_positive : r₁ > 0)
  (ratio : r₂ = 3 * r₁) -- radii ratio condition
  (volume₁ : ℝ) -- volume of the first cylinder
  (h_volume₁ : volume₁ = Real.pi * r₁^2 * h) -- volume formula for the first cylinder
  (volume₁_value : volume₁ = 40) -- given volume of the first cylinder
  : Real.pi * r₂^2 * h = 360 := by
  sorry

end NUMINAMATH_CALUDE_volume_of_second_cylinder_l155_15528


namespace NUMINAMATH_CALUDE_A_prime_div_B_prime_l155_15536

/-- The series A' as defined in the problem -/
noncomputable def A' : ℝ := ∑' n, if n % 5 ≠ 0 ∧ n % 2 ≠ 0 then ((-1) ^ ((n - 1) / 2 : ℕ)) / n^2 else 0

/-- The series B' as defined in the problem -/
noncomputable def B' : ℝ := ∑' n, if n % 5 = 0 ∧ n % 2 ≠ 0 then ((-1) ^ ((n / 5 - 1) / 2 : ℕ)) / n^2 else 0

/-- The main theorem stating that A' / B' = 26 -/
theorem A_prime_div_B_prime : A' / B' = 26 := by
  sorry

end NUMINAMATH_CALUDE_A_prime_div_B_prime_l155_15536


namespace NUMINAMATH_CALUDE_no_real_solutions_l155_15509

theorem no_real_solutions : 
  ¬∃ (x : ℝ), (x ≠ 2) ∧ ((x^3 - 8) / (x - 2) = 3*x) := by sorry

end NUMINAMATH_CALUDE_no_real_solutions_l155_15509


namespace NUMINAMATH_CALUDE_line_parameterization_values_l155_15597

/-- A line parameterized by a point and a direction vector -/
structure ParametricLine (α : Type*) [Field α] where
  point : α × α
  direction : α × α

/-- The equation of a line in slope-intercept form -/
structure LineEquation (α : Type*) [Field α] where
  slope : α
  intercept : α

/-- Check if a point lies on a line given by slope-intercept equation -/
def LineEquation.contains_point {α : Type*} [Field α] (l : LineEquation α) (p : α × α) : Prop :=
  p.2 = l.slope * p.1 + l.intercept

/-- Check if a parametric line is equivalent to a line equation -/
def parametric_line_equiv_equation {α : Type*} [Field α] 
  (pl : ParametricLine α) (le : LineEquation α) : Prop :=
  ∀ t : α, le.contains_point (pl.point.1 + t * pl.direction.1, pl.point.2 + t * pl.direction.2)

theorem line_parameterization_values 
  (l : LineEquation ℝ) 
  (pl : ParametricLine ℝ) 
  (h_equiv : parametric_line_equiv_equation pl l) 
  (h_slope : l.slope = 2) 
  (h_intercept : l.intercept = -7) 
  (h_point : pl.point = (s, 2)) 
  (h_direction : pl.direction = (3, m)) : 
  s = 9/2 ∧ m = -1 := by
  sorry

end NUMINAMATH_CALUDE_line_parameterization_values_l155_15597


namespace NUMINAMATH_CALUDE_at_least_one_not_in_area_l155_15541

theorem at_least_one_not_in_area (p q : Prop) : 
  (¬p ∨ ¬q) ↔ (∃ trainee, trainee = "A" ∧ ¬p ∨ trainee = "B" ∧ ¬q) :=
sorry

end NUMINAMATH_CALUDE_at_least_one_not_in_area_l155_15541


namespace NUMINAMATH_CALUDE_largest_x_quadratic_inequality_l155_15520

theorem largest_x_quadratic_inequality :
  ∀ x : ℝ, x^2 - 10*x + 24 ≤ 0 → x ≤ 6 :=
by sorry

end NUMINAMATH_CALUDE_largest_x_quadratic_inequality_l155_15520


namespace NUMINAMATH_CALUDE_mary_age_proof_l155_15543

/-- Mary's age today -/
def mary_age : ℕ := 12

/-- Mary's father's age today -/
def father_age : ℕ := 4 * mary_age

theorem mary_age_proof :
  (father_age = 4 * mary_age) ∧
  (father_age - 3 = 5 * (mary_age - 3)) →
  mary_age = 12 :=
by sorry

end NUMINAMATH_CALUDE_mary_age_proof_l155_15543


namespace NUMINAMATH_CALUDE_hairs_to_grow_back_l155_15537

def hairs_lost_washing : ℕ := 32

def hairs_lost_brushing : ℕ := hairs_lost_washing / 2

def total_hairs_lost : ℕ := hairs_lost_washing + hairs_lost_brushing

theorem hairs_to_grow_back : total_hairs_lost + 1 = 49 := by sorry

end NUMINAMATH_CALUDE_hairs_to_grow_back_l155_15537


namespace NUMINAMATH_CALUDE_work_time_difference_l155_15570

def monday_minutes : ℕ := 450
def wednesday_minutes : ℕ := 300

def tuesday_minutes : ℕ := monday_minutes / 2

theorem work_time_difference : wednesday_minutes - tuesday_minutes = 75 := by
  sorry

end NUMINAMATH_CALUDE_work_time_difference_l155_15570


namespace NUMINAMATH_CALUDE_boat_current_speed_l155_15526

/-- Given a boat traveling downstream at 15 km/h, and the distance traveled downstream
    in 4 hours equals the distance traveled upstream in 5 hours, prove that the speed
    of the water current is 1.5 km/h. -/
theorem boat_current_speed (v_d : ℝ) (t_d t_u : ℝ) (h1 : v_d = 15)
    (h2 : t_d = 4) (h3 : t_u = 5) (h4 : v_d * t_d = (2 * v_d - 15) * t_u / 2) :
    ∃ v_c : ℝ, v_c = 1.5 ∧ v_d = v_c + (2 * v_d - 15) / 2 := by
  sorry

end NUMINAMATH_CALUDE_boat_current_speed_l155_15526


namespace NUMINAMATH_CALUDE_quadratic_completing_square_l155_15598

theorem quadratic_completing_square (x : ℝ) : 
  (4 * x^2 - 24 * x - 96 = 0) → 
  ∃ q t : ℝ, ((x + q)^2 = t) ∧ (t = 33) := by
sorry

end NUMINAMATH_CALUDE_quadratic_completing_square_l155_15598


namespace NUMINAMATH_CALUDE_exponential_function_through_point_l155_15504

theorem exponential_function_through_point (f : ℝ → ℝ) :
  (∀ x : ℝ, ∃ a : ℝ, a > 0 ∧ a ≠ 1 ∧ f x = a^x) →
  f 1 = 2 →
  f 2 = 4 := by
sorry

end NUMINAMATH_CALUDE_exponential_function_through_point_l155_15504


namespace NUMINAMATH_CALUDE_largest_fraction_l155_15571

theorem largest_fraction : 
  let fractions := [2/5, 3/7, 5/9, 4/11, 3/8]
  ∀ x ∈ fractions, (5:ℚ)/9 ≥ x := by
  sorry

end NUMINAMATH_CALUDE_largest_fraction_l155_15571


namespace NUMINAMATH_CALUDE_logarithm_sum_simplification_l155_15594

theorem logarithm_sum_simplification :
  let expr := (1 / (Real.log 3 / Real.log 12 + 1)) + 
              (1 / (Real.log 2 / Real.log 8 + 1)) + 
              (1 / (Real.log 3 / Real.log 9 + 1))
  expr = (5 * Real.log 2 + 2 * Real.log 3) / (4 * Real.log 2 + 3 * Real.log 3) := by
  sorry

end NUMINAMATH_CALUDE_logarithm_sum_simplification_l155_15594


namespace NUMINAMATH_CALUDE_splash_width_is_seven_l155_15559

/-- The width of a splash made by a pebble in meters -/
def pebble_splash : ℚ := 1/4

/-- The width of a splash made by a rock in meters -/
def rock_splash : ℚ := 1/2

/-- The width of a splash made by a boulder in meters -/
def boulder_splash : ℚ := 2

/-- The number of pebbles thrown -/
def pebbles_thrown : ℕ := 6

/-- The number of rocks thrown -/
def rocks_thrown : ℕ := 3

/-- The number of boulders thrown -/
def boulders_thrown : ℕ := 2

/-- The total width of splashes made by TreQuan's throws -/
def total_splash_width : ℚ := 
  pebble_splash * pebbles_thrown + 
  rock_splash * rocks_thrown + 
  boulder_splash * boulders_thrown

theorem splash_width_is_seven : total_splash_width = 7 := by
  sorry

end NUMINAMATH_CALUDE_splash_width_is_seven_l155_15559


namespace NUMINAMATH_CALUDE_min_value_of_w_l155_15512

theorem min_value_of_w :
  ∀ (x y z w : ℝ),
    -2 ≤ x ∧ x ≤ 5 →
    -3 ≤ y ∧ y ≤ 7 →
    4 ≤ z ∧ z ≤ 8 →
    w = x * y - z →
    w ≥ -23 ∧ ∃ (x₀ y₀ z₀ : ℝ),
      -2 ≤ x₀ ∧ x₀ ≤ 5 ∧
      -3 ≤ y₀ ∧ y₀ ≤ 7 ∧
      4 ≤ z₀ ∧ z₀ ≤ 8 ∧
      x₀ * y₀ - z₀ = -23 :=
by sorry


end NUMINAMATH_CALUDE_min_value_of_w_l155_15512


namespace NUMINAMATH_CALUDE_sequence_element_proof_l155_15585

theorem sequence_element_proof :
  (∃ n : ℕ+, n^2 + 2*n = 63) ∧
  (¬ ∃ n : ℕ+, n^2 + 2*n = 10) ∧
  (¬ ∃ n : ℕ+, n^2 + 2*n = 18) ∧
  (¬ ∃ n : ℕ+, n^2 + 2*n = 26) :=
by sorry

end NUMINAMATH_CALUDE_sequence_element_proof_l155_15585


namespace NUMINAMATH_CALUDE_cyclic_ratio_inequality_l155_15557

theorem cyclic_ratio_inequality (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  b^2 / a + c^2 / b + a^2 / c ≥ Real.sqrt (3 * (a^2 + b^2 + c^2)) := by
  sorry

end NUMINAMATH_CALUDE_cyclic_ratio_inequality_l155_15557


namespace NUMINAMATH_CALUDE_carter_red_velvet_cakes_l155_15560

/-- The number of red velvet cakes Carter usually bakes per week -/
def usual_red_velvet : ℕ := sorry

/-- The number of cheesecakes Carter usually bakes per week -/
def usual_cheesecakes : ℕ := 6

/-- The number of muffins Carter usually bakes per week -/
def usual_muffins : ℕ := 5

/-- The total number of additional cakes Carter baked this week -/
def additional_cakes : ℕ := 38

/-- The factor by which Carter increased his baking this week -/
def increase_factor : ℕ := 3

theorem carter_red_velvet_cakes :
  (usual_cheesecakes + usual_muffins + usual_red_velvet) + additional_cakes =
  increase_factor * (usual_cheesecakes + usual_muffins + usual_red_velvet) →
  usual_red_velvet = 8 := by
sorry

end NUMINAMATH_CALUDE_carter_red_velvet_cakes_l155_15560


namespace NUMINAMATH_CALUDE_percentage_problem_l155_15580

theorem percentage_problem (x : ℝ) : 0.25 * x = 0.20 * 1000 - 30 → x = 680 := by
  sorry

end NUMINAMATH_CALUDE_percentage_problem_l155_15580


namespace NUMINAMATH_CALUDE_area_Ω_bound_l155_15532

/-- Parabola C: y = (1/2)x^2 -/
def parabola_C (x y : ℝ) : Prop := y = (1/2) * x^2

/-- Circle D: x^2 + (y - 1/2)^2 = r^2, where r > 0 -/
def circle_D (x y r : ℝ) : Prop := x^2 + (y - 1/2)^2 = r^2 ∧ r > 0

/-- C and D have no common points -/
def no_intersection (r : ℝ) : Prop := ∀ x y : ℝ, parabola_C x y → ¬(circle_D x y r)

/-- Point A is on parabola C -/
def point_on_parabola (A : ℝ × ℝ) : Prop := parabola_C A.1 A.2

/-- Region Ω formed by tangents from A to D -/
def region_Ω (r : ℝ) (A : ℝ × ℝ) : Set (ℝ × ℝ) := sorry

/-- Area of region Ω -/
noncomputable def area_Ω (r : ℝ) (A : ℝ × ℝ) : ℝ := sorry

theorem area_Ω_bound (r : ℝ) (h : no_intersection r) :
  ∀ A : ℝ × ℝ, point_on_parabola A →
    0 < area_Ω r A ∧ area_Ω r A < π/16 := by sorry

end NUMINAMATH_CALUDE_area_Ω_bound_l155_15532


namespace NUMINAMATH_CALUDE_all_or_none_triangular_l155_15596

/-- A polynomial of degree 4 -/
structure Polynomial4 where
  p : ℝ
  q : ℝ
  r : ℝ
  s : ℝ

/-- Evaluate the polynomial at a given x -/
def eval (poly : Polynomial4) (x : ℝ) : ℝ :=
  x^4 + poly.p * x^3 + poly.q * x^2 + poly.r * x + poly.s

/-- Represents four points on a horizontal line intersecting the curve -/
structure FourPoints where
  x₁ : ℝ
  x₂ : ℝ
  x₃ : ℝ
  x₄ : ℝ
  h₁ : x₁ < x₂
  h₂ : x₂ < x₃
  h₃ : x₃ < x₄

/-- Check if three lengths can form a triangle -/
def isTriangle (a b c : ℝ) : Prop :=
  a + b > c ∧ b + c > a ∧ c + a > b

/-- Check if a set of four points is triangular -/
def isTriangular (pts : FourPoints) : Prop :=
  isTriangle (pts.x₂ - pts.x₁) (pts.x₃ - pts.x₁) (pts.x₄ - pts.x₁)

/-- The main theorem -/
theorem all_or_none_triangular (poly : Polynomial4) :
  (∀ y : ℝ, ∀ pts : FourPoints, eval poly pts.x₁ = y ∧ eval poly pts.x₂ = y ∧
    eval poly pts.x₃ = y ∧ eval poly pts.x₄ = y → isTriangular pts) ∨
  (∀ y : ℝ, ∀ pts : FourPoints, eval poly pts.x₁ = y ∧ eval poly pts.x₂ = y ∧
    eval poly pts.x₃ = y ∧ eval poly pts.x₄ = y → ¬isTriangular pts) :=
sorry

end NUMINAMATH_CALUDE_all_or_none_triangular_l155_15596


namespace NUMINAMATH_CALUDE_quadratic_equation_root_l155_15552

theorem quadratic_equation_root : ∃ (a b c : ℝ), a ≠ 0 ∧ 
  (∀ x, a * x^2 + b * x + c = 0 ↔ x^2 - 9 = 0) ∧
  ((-3 : ℝ)^2 - 9 = 0) := by
  sorry

end NUMINAMATH_CALUDE_quadratic_equation_root_l155_15552


namespace NUMINAMATH_CALUDE_mans_age_twice_sons_l155_15558

/-- Proves that the number of years until a man's age is twice his son's age is 2 -/
theorem mans_age_twice_sons (
  man_age_difference : ℕ → ℕ → ℕ)
  (son_current_age : ℕ)
  (h1 : man_age_difference son_current_age son_current_age = 34)
  (h2 : son_current_age = 32)
  : ∃ (years : ℕ), years = 2 ∧
    man_age_difference (son_current_age + years) (son_current_age + years) + years =
    2 * (son_current_age + years) :=
by sorry

end NUMINAMATH_CALUDE_mans_age_twice_sons_l155_15558


namespace NUMINAMATH_CALUDE_angle_at_point_l155_15510

theorem angle_at_point (y : ℝ) : 
  y > 0 ∧ y + y + 140 = 360 → y = 110 := by
  sorry

end NUMINAMATH_CALUDE_angle_at_point_l155_15510


namespace NUMINAMATH_CALUDE_max_value_range_l155_15573

-- Define the function f and its derivative
def f : ℝ → ℝ := sorry

-- Define the derivative of f
def f_derivative (a : ℝ) (x : ℝ) : ℝ := a * (x + 1) * (x - a)

-- State the theorem
theorem max_value_range (a : ℝ) :
  (∀ x, f_derivative a x = a * (x + 1) * (x - a)) →
  (∃ x₀, ∀ x, f x ≤ f x₀) →
  (∀ x, x < a → f_derivative a x > 0) →
  (∀ x, x > a → f_derivative a x < 0) →
  a ∈ Set.Ioo (-1 : ℝ) 0 :=
sorry

end NUMINAMATH_CALUDE_max_value_range_l155_15573


namespace NUMINAMATH_CALUDE_sheets_per_student_l155_15563

theorem sheets_per_student (num_classes : ℕ) (students_per_class : ℕ) (total_sheets : ℕ) :
  num_classes = 4 →
  students_per_class = 20 →
  total_sheets = 400 →
  total_sheets / (num_classes * students_per_class) = 5 := by
  sorry

end NUMINAMATH_CALUDE_sheets_per_student_l155_15563


namespace NUMINAMATH_CALUDE_termite_ridden_not_collapsing_l155_15586

/-- Given that 1/3 of homes are termite-ridden and 4/7 of termite-ridden homes are collapsing,
    prove that 3/21 of homes are termite-ridden but not collapsing. -/
theorem termite_ridden_not_collapsing 
  (total_homes : ℕ) 
  (termite_ridden : ℕ) 
  (collapsing : ℕ) 
  (h1 : termite_ridden = total_homes / 3) 
  (h2 : collapsing = termite_ridden * 4 / 7) : 
  (termite_ridden - collapsing) = total_homes * 3 / 21 := by
  sorry

end NUMINAMATH_CALUDE_termite_ridden_not_collapsing_l155_15586


namespace NUMINAMATH_CALUDE_chocolate_bar_calculation_l155_15545

/-- The number of chocolate bars in each box -/
def bars_per_box : ℕ := 5

/-- The number of boxes Tom needs to sell -/
def boxes_to_sell : ℕ := 170

/-- The total number of chocolate bars Tom needs to sell -/
def total_bars : ℕ := bars_per_box * boxes_to_sell

theorem chocolate_bar_calculation :
  total_bars = 850 := by sorry

end NUMINAMATH_CALUDE_chocolate_bar_calculation_l155_15545


namespace NUMINAMATH_CALUDE_range_of_a_in_fourth_quadrant_l155_15595

/-- Given a complex number z that corresponds to a point in the fourth quadrant,
    prove that the real parameter a in z = (a + 2i³) / (2 - i) is in the range (-1, 4) -/
theorem range_of_a_in_fourth_quadrant (a : ℝ) :
  let z : ℂ := (a + 2 * Complex.I ^ 3) / (2 - Complex.I)
  (z.re > 0 ∧ z.im < 0) → -1 < a ∧ a < 4 := by
  sorry

end NUMINAMATH_CALUDE_range_of_a_in_fourth_quadrant_l155_15595


namespace NUMINAMATH_CALUDE_angle_B_is_pi_third_b_range_l155_15548

open Real

/-- Triangle ABC with sides a, b, c opposite to angles A, B, C respectively --/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

/-- The given condition in the problem --/
def condition (t : Triangle) : Prop :=
  cos t.C + (cos t.A - Real.sqrt 3 * sin t.A) * cos t.B = 0

/-- Theorem 1: Given the condition, angle B is π/3 --/
theorem angle_B_is_pi_third (t : Triangle) (h : condition t) : t.B = π / 3 := by
  sorry

/-- Additional condition for part 2 --/
def sum_sides_is_one (t : Triangle) : Prop :=
  t.a + t.c = 1

/-- Theorem 2: Given sum_sides_is_one and B = π/3, b is in [1/2, 1) --/
theorem b_range (t : Triangle) (h1 : sum_sides_is_one t) (h2 : t.B = π / 3) :
  1 / 2 ≤ t.b ∧ t.b < 1 := by
  sorry

end NUMINAMATH_CALUDE_angle_B_is_pi_third_b_range_l155_15548


namespace NUMINAMATH_CALUDE_nancy_eats_indian_food_three_times_a_week_l155_15589

/-- Represents the number of times Nancy eats Indian food per week -/
def indian_food_times : ℕ := sorry

/-- Represents the number of times Nancy eats Mexican food per week -/
def mexican_food_times : ℕ := 2

/-- Represents the number of antacids Nancy takes when eating Indian food -/
def indian_food_antacids : ℕ := 3

/-- Represents the number of antacids Nancy takes when eating Mexican food -/
def mexican_food_antacids : ℕ := 2

/-- Represents the number of antacids Nancy takes on other days -/
def other_days_antacids : ℕ := 1

/-- Represents the total number of days in a week -/
def days_in_week : ℕ := 7

/-- Represents the number of weeks in a month (approximation) -/
def weeks_in_month : ℕ := 4

/-- Represents the total number of antacids Nancy takes per month -/
def total_antacids_per_month : ℕ := 60

/-- Theorem stating that Nancy eats Indian food 3 times a week -/
theorem nancy_eats_indian_food_three_times_a_week :
  indian_food_times = 3 :=
by sorry

end NUMINAMATH_CALUDE_nancy_eats_indian_food_three_times_a_week_l155_15589


namespace NUMINAMATH_CALUDE_max_chickens_and_chicks_max_chicks_no_chickens_l155_15508

/-- Represents the chicken coop problem -/
structure ChickenCoop where
  area : ℝ
  chicken_space : ℝ
  chick_space : ℝ
  chicken_feed : ℝ
  chick_feed : ℝ
  max_feed : ℝ

/-- Defines the specific chicken coop instance -/
def our_coop : ChickenCoop :=
  { area := 240
  , chicken_space := 4
  , chick_space := 2
  , chicken_feed := 160
  , chick_feed := 40
  , max_feed := 8000 }

/-- Theorem stating the maximum number of chickens and chicks -/
theorem max_chickens_and_chicks (coop : ChickenCoop) :
  ∃ (x y : ℕ), 
    x * coop.chicken_space + y * coop.chick_space = coop.area ∧
    x * coop.chicken_feed + y * coop.chick_feed ≤ coop.max_feed ∧
    x = 40 ∧ y = 40 ∧
    ∀ (a b : ℕ), 
      a * coop.chicken_space + b * coop.chick_space = coop.area →
      a * coop.chicken_feed + b * coop.chick_feed ≤ coop.max_feed →
      a ≤ x :=
by sorry

/-- Theorem stating the maximum number of chicks with no chickens -/
theorem max_chicks_no_chickens (coop : ChickenCoop) :
  ∃ (y : ℕ),
    y * coop.chick_space = coop.area ∧
    y * coop.chick_feed ≤ coop.max_feed ∧
    y = 120 ∧
    ∀ (b : ℕ),
      b * coop.chick_space = coop.area →
      b * coop.chick_feed ≤ coop.max_feed →
      b ≤ y :=
by sorry

end NUMINAMATH_CALUDE_max_chickens_and_chicks_max_chicks_no_chickens_l155_15508


namespace NUMINAMATH_CALUDE_smallest_number_with_unique_digits_divisible_by_990_l155_15582

theorem smallest_number_with_unique_digits_divisible_by_990 : ∃ (n : ℕ), 
  (n = 1234758690) ∧ 
  (∀ m : ℕ, m < n → ¬(∀ d : Fin 10, (m.digits 10).count d = 1)) ∧
  (∀ d : Fin 10, (n.digits 10).count d = 1) ∧
  (n % 990 = 0) := by
  sorry

end NUMINAMATH_CALUDE_smallest_number_with_unique_digits_divisible_by_990_l155_15582


namespace NUMINAMATH_CALUDE_range_of_f_l155_15502

noncomputable def f (x : ℝ) : ℝ := 3 * (x - 4)

theorem range_of_f :
  ∀ y : ℝ, y ≠ -27 → ∃ x : ℝ, x ≠ -5 ∧ f x = y :=
sorry

end NUMINAMATH_CALUDE_range_of_f_l155_15502


namespace NUMINAMATH_CALUDE_system_two_solutions_l155_15535

theorem system_two_solutions (a : ℝ) :
  (∃! x y, a^2 - 2*a*x - 6*y + x^2 + y^2 = 0 ∧ (|x| - 4)^2 + (|y| - 3)^2 = 25) ↔
  a ∈ Set.Ioo (-12) (-6) ∪ {0} ∪ Set.Ioo 6 12 :=
by sorry

end NUMINAMATH_CALUDE_system_two_solutions_l155_15535


namespace NUMINAMATH_CALUDE_select_duty_officers_l155_15517

def choose (n k : ℕ) : ℕ := 
  if k > n then 0
  else Nat.factorial n / (Nat.factorial k * Nat.factorial (n - k))

theorem select_duty_officers : choose 20 3 = 1140 := by
  sorry

end NUMINAMATH_CALUDE_select_duty_officers_l155_15517


namespace NUMINAMATH_CALUDE_total_count_theorem_l155_15555

/-- The total number of oysters and crabs counted over two days -/
def total_count (initial_oysters initial_crabs : ℕ) : ℕ :=
  let day1_total := initial_oysters + initial_crabs
  let day2_oysters := initial_oysters / 2
  let day2_crabs := initial_crabs * 2 / 3
  let day2_total := day2_oysters + day2_crabs
  day1_total + day2_total

/-- Theorem stating the total count of oysters and crabs over two days -/
theorem total_count_theorem (initial_oysters initial_crabs : ℕ) 
  (h1 : initial_oysters = 50) 
  (h2 : initial_crabs = 72) : 
  total_count initial_oysters initial_crabs = 195 := by
  sorry

end NUMINAMATH_CALUDE_total_count_theorem_l155_15555


namespace NUMINAMATH_CALUDE_jelly_bean_probability_l155_15500

theorem jelly_bean_probability 
  (red_prob : ℝ) 
  (orange_prob : ℝ) 
  (green_prob : ℝ) 
  (h1 : red_prob = 0.15)
  (h2 : orange_prob = 0.4)
  (h3 : green_prob = 0.1)
  (h4 : ∃ yellow_prob : ℝ, red_prob + orange_prob + yellow_prob + green_prob = 1) :
  ∃ yellow_prob : ℝ, yellow_prob = 0.35 ∧ red_prob + orange_prob + yellow_prob + green_prob = 1 := by
sorry

end NUMINAMATH_CALUDE_jelly_bean_probability_l155_15500


namespace NUMINAMATH_CALUDE_best_and_most_stable_values_l155_15572

/-- Represents a student's performance data -/
structure StudentPerformance where
  average : ℝ
  variance : ℝ

/-- The given data for students B, C, and D -/
def studentB : StudentPerformance := ⟨90, 12.5⟩
def studentC : StudentPerformance := ⟨91, 14.5⟩
def studentD : StudentPerformance := ⟨88, 11⟩

/-- Conditions for Student A to be the best-performing and most stable -/
def isBestAndMostStable (m n : ℝ) : Prop :=
  m > studentB.average ∧
  m > studentC.average ∧
  m > studentD.average ∧
  n < studentB.variance ∧
  n < studentC.variance ∧
  n < studentD.variance

/-- Theorem stating that m = 92 and n = 8.5 are the only values satisfying the conditions -/
theorem best_and_most_stable_values :
  ∀ m n : ℝ, isBestAndMostStable m n ↔ m = 92 ∧ n = 8.5 := by
  sorry

end NUMINAMATH_CALUDE_best_and_most_stable_values_l155_15572


namespace NUMINAMATH_CALUDE_company_picnic_volleyball_teams_l155_15521

theorem company_picnic_volleyball_teams 
  (managers : ℕ) 
  (employees : ℕ) 
  (teams : ℕ) 
  (h1 : managers = 23) 
  (h2 : employees = 7) 
  (h3 : teams = 6) : 
  (managers + employees) / teams = 5 := by
  sorry

end NUMINAMATH_CALUDE_company_picnic_volleyball_teams_l155_15521


namespace NUMINAMATH_CALUDE_probability_ratio_is_twenty_l155_15584

def total_balls : ℕ := 25
def num_bins : ℕ := 6

def distribution_A : List ℕ := [4, 4, 4, 5, 5, 2]
def distribution_B : List ℕ := [5, 5, 5, 5, 5, 0]

def probability_ratio : ℚ :=
  (Nat.choose num_bins 3 * Nat.choose 3 2 * Nat.choose 1 1 *
   (Nat.factorial total_balls / (Nat.factorial 4 * Nat.factorial 4 * Nat.factorial 4 * Nat.factorial 5 * Nat.factorial 5 * Nat.factorial 2))) /
  (Nat.choose num_bins 5 * Nat.choose 1 1 *
   (Nat.factorial total_balls / (Nat.factorial 5 * Nat.factorial 5 * Nat.factorial 5 * Nat.factorial 5 * Nat.factorial 5 * Nat.factorial 0)))

theorem probability_ratio_is_twenty :
  probability_ratio = 20 := by sorry

end NUMINAMATH_CALUDE_probability_ratio_is_twenty_l155_15584


namespace NUMINAMATH_CALUDE_edwards_toy_purchase_l155_15515

/-- Proves that given an initial amount of $17.80, after purchasing 4 items at $0.95 each
    and one item at $6.00, the remaining amount is $8.00. -/
theorem edwards_toy_purchase (initial_amount : ℚ) (toy_car_price : ℚ) (race_track_price : ℚ)
    (num_toy_cars : ℕ) (h1 : initial_amount = 17.8)
    (h2 : toy_car_price = 0.95) (h3 : race_track_price = 6)
    (h4 : num_toy_cars = 4) : 
    initial_amount - (toy_car_price * num_toy_cars + race_track_price) = 8 := by
  sorry

end NUMINAMATH_CALUDE_edwards_toy_purchase_l155_15515


namespace NUMINAMATH_CALUDE_quadratic_equation_solution_l155_15581

theorem quadratic_equation_solution :
  ∃! y : ℝ, y^2 + 6*y + 8 = -(y + 4)*(y + 6) :=
by
  -- The proof goes here
  sorry

end NUMINAMATH_CALUDE_quadratic_equation_solution_l155_15581


namespace NUMINAMATH_CALUDE_ball_selection_count_l155_15568

/-- Represents the set of colors available for the balls -/
inductive Color
| Red
| Yellow
| Blue

/-- Represents the set of letters used to mark the balls -/
inductive Letter
| A
| B
| C
| D
| E

/-- The total number of balls for each color -/
def ballsPerColor : Nat := 5

/-- The total number of colors -/
def numColors : Nat := 3

/-- The number of balls to be selected -/
def ballsToSelect : Nat := 5

/-- Calculates the number of ways to select the balls -/
def selectBalls : Nat := numColors ^ ballsToSelect

theorem ball_selection_count :
  selectBalls = 243 :=
sorry

end NUMINAMATH_CALUDE_ball_selection_count_l155_15568


namespace NUMINAMATH_CALUDE_flower_arrangement_count_l155_15513

def num_flowers : ℕ := 5
def num_vases : ℕ := 3

theorem flower_arrangement_count :
  (num_flowers * (num_flowers - 1) * (num_flowers - 2)) = 60 := by
  sorry

end NUMINAMATH_CALUDE_flower_arrangement_count_l155_15513


namespace NUMINAMATH_CALUDE_jewelry_sales_fraction_l155_15514

theorem jewelry_sales_fraction (total_sales : ℕ) (stationery_sales : ℕ) :
  total_sales = 36 →
  stationery_sales = 15 →
  (total_sales : ℚ) / 3 + stationery_sales + (total_sales : ℚ) / 4 = total_sales :=
by
  sorry

end NUMINAMATH_CALUDE_jewelry_sales_fraction_l155_15514


namespace NUMINAMATH_CALUDE_rachels_homework_l155_15575

/-- Rachel's homework problem -/
theorem rachels_homework (reading_pages : ℕ) (math_pages : ℕ) : 
  reading_pages = 2 → math_pages = reading_pages + 3 → math_pages = 5 :=
by sorry

end NUMINAMATH_CALUDE_rachels_homework_l155_15575


namespace NUMINAMATH_CALUDE_factorial_solutions_l155_15593

def factorial : ℕ → ℕ
  | 0 => 1
  | n + 1 => (n + 1) * factorial n

theorem factorial_solutions :
  ∀ x y z : ℕ, factorial x + 2^y = factorial z →
    ((x = 0 ∧ y = 0) ∨ (x = 1 ∧ y = 0) ∨ (x = 2 ∧ y = 2)) :=
by sorry

end NUMINAMATH_CALUDE_factorial_solutions_l155_15593


namespace NUMINAMATH_CALUDE_women_married_long_service_fraction_l155_15588

theorem women_married_long_service_fraction 
  (total_employees : ℕ) 
  (women_percentage : ℚ)
  (married_percentage : ℚ)
  (single_men_fraction : ℚ)
  (married_long_service_women_percentage : ℚ)
  (h1 : women_percentage = 76 / 100)
  (h2 : married_percentage = 60 / 100)
  (h3 : single_men_fraction = 2 / 3)
  (h4 : married_long_service_women_percentage = 70 / 100)
  : ℚ :=
by
  sorry

#check women_married_long_service_fraction

end NUMINAMATH_CALUDE_women_married_long_service_fraction_l155_15588


namespace NUMINAMATH_CALUDE_circumcenter_property_l155_15577

/-- A point in 3D space -/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- A plane in 3D space -/
structure Plane where
  a : ℝ
  b : ℝ
  c : ℝ
  d : ℝ

/-- A triangle in 3D space -/
structure Triangle3D where
  A : Point3D
  B : Point3D
  C : Point3D

/-- Check if a point is outside a plane -/
def isOutside (p : Point3D) (plane : Plane) : Prop := sorry

/-- Check if a line is perpendicular to a plane -/
def isPerpendicular (p1 p2 : Point3D) (plane : Plane) : Prop := sorry

/-- Check if a point is the foot of the perpendicular from another point to a plane -/
def isFootOfPerpendicular (o p : Point3D) (plane : Plane) : Prop := sorry

/-- Check if three distances are equal -/
def areDistancesEqual (p a b c : Point3D) : Prop := sorry

/-- Check if a point is the circumcenter of a triangle -/
def isCircumcenter (o : Point3D) (t : Triangle3D) : Prop := sorry

theorem circumcenter_property (P O : Point3D) (ABC : Triangle3D) (plane : Plane) :
  isOutside P plane →
  isPerpendicular P O plane →
  isFootOfPerpendicular O P plane →
  areDistancesEqual P ABC.A ABC.B ABC.C →
  isCircumcenter O ABC := by sorry

end NUMINAMATH_CALUDE_circumcenter_property_l155_15577


namespace NUMINAMATH_CALUDE_correct_seat_notation_l155_15591

/-- Represents a seat in a movie theater -/
structure Seat :=
  (row : ℕ)
  (seatNum : ℕ)

/-- Defines the notation for a seat -/
def seatNotation (s : Seat) : ℕ × ℕ := (s.row, s.seatNum)

theorem correct_seat_notation :
  let example_seat := Seat.mk 10 3
  let target_seat := Seat.mk 6 16
  (seatNotation example_seat = (10, 3)) →
  (seatNotation target_seat = (6, 16)) := by
  sorry

end NUMINAMATH_CALUDE_correct_seat_notation_l155_15591


namespace NUMINAMATH_CALUDE_fixed_distance_from_linear_combination_l155_15544

variable {E : Type*} [NormedAddCommGroup E] [InnerProductSpace ℝ E]

/-- Given vectors a and b, and a vector p satisfying the condition,
    prove that t = 9/8 and u = -1/8 make ‖p - (t*a + u*b)‖ constant. -/
theorem fixed_distance_from_linear_combination
  (a b p : E) (h : ‖p - b‖ = 3 * ‖p - a‖) :
  ∃ (c : ℝ), ∀ (q : E), ‖q - b‖ = 3 * ‖q - a‖ →
    ‖q - ((9/8 : ℝ) • a + (-1/8 : ℝ) • b)‖ = c :=
sorry

end NUMINAMATH_CALUDE_fixed_distance_from_linear_combination_l155_15544


namespace NUMINAMATH_CALUDE_rectangular_field_area_l155_15507

theorem rectangular_field_area : 
  let length : ℝ := 5.9
  let width : ℝ := 3
  length * width = 17.7 := by sorry

end NUMINAMATH_CALUDE_rectangular_field_area_l155_15507


namespace NUMINAMATH_CALUDE_solution_set_for_m_eq_2_m_range_for_solution_set_R_l155_15539

-- Define the function f
def f (m : ℝ) (x : ℝ) : ℝ := x^2 + (m-1)*x - m

-- Part 1
theorem solution_set_for_m_eq_2 :
  {x : ℝ | f 2 x < 0} = {x : ℝ | -2 < x ∧ x < 1} := by sorry

-- Part 2
theorem m_range_for_solution_set_R :
  (∀ x, f m x ≥ -1) → m ∈ Set.Icc (-3) 1 := by sorry

end NUMINAMATH_CALUDE_solution_set_for_m_eq_2_m_range_for_solution_set_R_l155_15539


namespace NUMINAMATH_CALUDE_gravelling_cost_theorem_l155_15546

/-- The cost of gravelling a path around a rectangular plot -/
theorem gravelling_cost_theorem 
  (plot_length : ℝ) 
  (plot_width : ℝ) 
  (path_width : ℝ) 
  (cost_per_sqm_paise : ℝ) 
  (h1 : plot_length = 110) 
  (h2 : plot_width = 65) 
  (h3 : path_width = 2.5) 
  (h4 : cost_per_sqm_paise = 60) : 
  (((plot_length * plot_width) - ((plot_length - 2 * path_width) * (plot_width - 2 * path_width))) * (cost_per_sqm_paise / 100)) = 510 := by
  sorry

end NUMINAMATH_CALUDE_gravelling_cost_theorem_l155_15546


namespace NUMINAMATH_CALUDE_no_consecutive_squares_arithmetic_sequence_l155_15506

theorem no_consecutive_squares_arithmetic_sequence :
  ∀ (x y z w : ℕ+), ¬∃ (d : ℝ),
    (y : ℝ)^2 = (x : ℝ)^2 + d ∧
    (z : ℝ)^2 = (y : ℝ)^2 + d ∧
    (w : ℝ)^2 = (z : ℝ)^2 + d :=
by sorry

end NUMINAMATH_CALUDE_no_consecutive_squares_arithmetic_sequence_l155_15506


namespace NUMINAMATH_CALUDE_circle_center_sum_l155_15529

/-- The sum of the x and y coordinates of the center of a circle
    described by the equation x^2 + y^2 = 6x + 6y - 30 is equal to 6 -/
theorem circle_center_sum (x y : ℝ) : 
  (x^2 + y^2 = 6*x + 6*y - 30) → ∃ h k : ℝ, (h + k = 6 ∧ (x - h)^2 + (y - k)^2 = (x^2 + y^2 - 6*x - 6*y + 30)) :=
by sorry

end NUMINAMATH_CALUDE_circle_center_sum_l155_15529


namespace NUMINAMATH_CALUDE_horner_method_difference_l155_15503

def f (x : ℝ) : ℝ := 1 - 5*x - 8*x^2 + 10*x^3 + 6*x^4 + 12*x^5 + 3*x^6

def v₀ : ℝ := 3
def v₁ (x : ℝ) : ℝ := v₀ * x + 12
def v₂ (x : ℝ) : ℝ := v₁ x * x + 6
def v₃ (x : ℝ) : ℝ := v₂ x * x + 10
def v₄ (x : ℝ) : ℝ := v₃ x * x - 8

theorem horner_method_difference (x : ℝ) (hx : x = -4) :
  (max v₀ (max (v₁ x) (max (v₂ x) (max (v₃ x) (v₄ x))))) -
  (min v₀ (min (v₁ x) (min (v₂ x) (min (v₃ x) (v₄ x))))) = 62 := by
  sorry

end NUMINAMATH_CALUDE_horner_method_difference_l155_15503

import Mathlib

namespace NUMINAMATH_CALUDE_work_rate_ab_together_days_for_ab_together_l1802_180277

-- Define the work rates for workers a, b, and c
variable (A B C : ℝ)

-- Define the conditions
variable (h1 : A + B + C = 1 / 5)  -- a, b, and c together finish in 5 days
variable (h2 : C = 1 / 7.5)        -- c alone finishes in 7.5 days

-- Theorem to prove
theorem work_rate_ab_together : A + B = 1 / 15 := by
  sorry

-- Theorem to prove the final result
theorem days_for_ab_together : 1 / (A + B) = 15 := by
  sorry

end NUMINAMATH_CALUDE_work_rate_ab_together_days_for_ab_together_l1802_180277


namespace NUMINAMATH_CALUDE_caiden_roofing_problem_l1802_180239

/-- Calculates the number of feet of free metal roofing given the total required roofing,
    cost per foot, and amount paid for the remaining roofing. -/
def free_roofing (total_required : ℕ) (cost_per_foot : ℕ) (amount_paid : ℕ) : ℕ :=
  total_required - (amount_paid / cost_per_foot)

/-- Theorem stating that given the specific conditions of Mr. Caiden's roofing problem,
    the amount of free roofing is 250 feet. -/
theorem caiden_roofing_problem :
  free_roofing 300 8 400 = 250 := by
  sorry

end NUMINAMATH_CALUDE_caiden_roofing_problem_l1802_180239


namespace NUMINAMATH_CALUDE_geometric_mean_minimum_l1802_180251

theorem geometric_mean_minimum (a b : ℝ) (ha : a > 0) (hb : b > 0)
  (h_geom_mean : 3 = Real.sqrt (9^a * 27^b)) :
  (3/a + 2/b) ≥ 12 ∧ ∃ (a₀ b₀ : ℝ), a₀ > 0 ∧ b₀ > 0 ∧ 3 = Real.sqrt (9^a₀ * 27^b₀) ∧ (3/a₀ + 2/b₀) = 12 := by
  sorry

end NUMINAMATH_CALUDE_geometric_mean_minimum_l1802_180251


namespace NUMINAMATH_CALUDE_waiting_room_ratio_l1802_180265

theorem waiting_room_ratio : 
  ∀ (initial_waiting : ℕ) (arrivals : ℕ) (interview : ℕ),
    initial_waiting = 22 →
    arrivals = 3 →
    (initial_waiting + arrivals) % interview = 0 →
    interview ≠ 1 →
    interview < initial_waiting + arrivals →
    (initial_waiting + arrivals) / interview = 5 :=
by sorry

end NUMINAMATH_CALUDE_waiting_room_ratio_l1802_180265


namespace NUMINAMATH_CALUDE_ellipse_equation_l1802_180227

/-- An ellipse passing through (3, 0) with eccentricity √6/3 has standard equations x²/9 + y²/3 = 1 or x²/9 + y²/27 = 1 -/
theorem ellipse_equation (x y : ℝ) : 
  let e : ℝ := Real.sqrt 6 / 3
  let passes_through : Prop := x^2 + y^2 = 9 ∧ y = 0
  let equation1 : Prop := x^2 / 9 + y^2 / 3 = 1
  let equation2 : Prop := x^2 / 9 + y^2 / 27 = 1
  passes_through → (equation1 ∨ equation2) :=
by sorry

end NUMINAMATH_CALUDE_ellipse_equation_l1802_180227


namespace NUMINAMATH_CALUDE_charitable_woman_purse_l1802_180252

/-- The charitable woman's purse problem -/
theorem charitable_woman_purse (P : ℚ) : 
  (P > 0) →
  (P - ((1/2) * P + 1) - ((1/2) * ((1/2) * P - 1) + 2) - ((1/2) * ((1/4) * P - 2.5) + 3) = 1) →
  P = 42 := by
  sorry

end NUMINAMATH_CALUDE_charitable_woman_purse_l1802_180252


namespace NUMINAMATH_CALUDE_fraction_equation_l1802_180266

theorem fraction_equation : ∃ (A B C : ℤ), 
  (A : ℚ) / 999 + (B : ℚ) / 1000 + (C : ℚ) / 1001 = 1 / (999 * 1000 * 1001) :=
by
  -- We claim that A = 500, B = -1, C = -500 satisfy the equation
  use 500, -1, -500
  -- The proof goes here
  sorry

end NUMINAMATH_CALUDE_fraction_equation_l1802_180266


namespace NUMINAMATH_CALUDE_distinct_colorings_l1802_180209

/-- The number of disks in the circle -/
def n : ℕ := 8

/-- The number of blue disks -/
def blue : ℕ := 4

/-- The number of red disks -/
def red : ℕ := 3

/-- The number of green disks -/
def green : ℕ := 1

/-- The number of rotational symmetries -/
def rotations : ℕ := 4

/-- The number of reflection symmetries -/
def reflections : ℕ := 4

/-- The total number of symmetries -/
def total_symmetries : ℕ := rotations + reflections + 1

/-- The number of colorings fixed by identity -/
def fixed_by_identity : ℕ := (n.choose blue) * ((n - blue).choose red)

/-- The number of colorings fixed by each reflection -/
def fixed_by_reflection : ℕ := 6

/-- The number of colorings fixed by each rotation (other than identity) -/
def fixed_by_rotation : ℕ := 0

/-- Theorem: The number of distinct colorings is 38 -/
theorem distinct_colorings : 
  (fixed_by_identity + reflections * fixed_by_reflection + (rotations - 1) * fixed_by_rotation) / total_symmetries = 38 := by
  sorry

end NUMINAMATH_CALUDE_distinct_colorings_l1802_180209


namespace NUMINAMATH_CALUDE_first_pair_sum_l1802_180232

-- Define the sequence rule
def sequence_rule (a b : ℕ) : ℕ := a + b - 1

-- Theorem statement
theorem first_pair_sum :
  (sequence_rule 8 9 = 16) →
  (sequence_rule 5 6 = 10) →
  (sequence_rule 7 8 = 14) →
  (sequence_rule 3 3 = 5) →
  sequence_rule 6 7 = 12 := by
sorry

end NUMINAMATH_CALUDE_first_pair_sum_l1802_180232


namespace NUMINAMATH_CALUDE_car_fuel_efficiency_l1802_180289

theorem car_fuel_efficiency 
  (x : ℝ) 
  (h1 : x > 0) 
  (tank_capacity : ℝ) 
  (h2 : tank_capacity = 12) 
  (efficiency_improvement : ℝ) 
  (h3 : efficiency_improvement = 0.8) 
  (distance_increase : ℝ) 
  (h4 : distance_increase = 96) :
  (tank_capacity * x / efficiency_improvement) - (tank_capacity * x) = distance_increase →
  x = 32 :=
by sorry

end NUMINAMATH_CALUDE_car_fuel_efficiency_l1802_180289


namespace NUMINAMATH_CALUDE_max_value_inequality_l1802_180282

theorem max_value_inequality (x y : ℝ) (h : x * y > 0) :
  (x / (x + y)) + (2 * y / (x + 2 * y)) ≤ 4 - 2 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_max_value_inequality_l1802_180282


namespace NUMINAMATH_CALUDE_cookies_left_l1802_180297

/-- The number of cookies in a dozen -/
def dozen : ℕ := 12

/-- The number of dozens of cookies Meena starts with -/
def initial_dozens : ℕ := 5

/-- The number of dozens of cookies sold to Mr. Stone -/
def sold_to_stone : ℕ := 2

/-- The number of cookies bought by Brock -/
def bought_by_brock : ℕ := 7

/-- Katy buys twice as many cookies as Brock -/
def bought_by_katy : ℕ := 2 * bought_by_brock

/-- The theorem stating that Meena has 15 cookies left -/
theorem cookies_left : 
  initial_dozens * dozen - sold_to_stone * dozen - bought_by_brock - bought_by_katy = 15 := by
  sorry


end NUMINAMATH_CALUDE_cookies_left_l1802_180297


namespace NUMINAMATH_CALUDE_arithmetic_expression_equality_l1802_180234

theorem arithmetic_expression_equality : 68 + (105 / 15) + (26 * 19) - 250 - (390 / 6) = 254 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_expression_equality_l1802_180234


namespace NUMINAMATH_CALUDE_expression_simplification_l1802_180253

theorem expression_simplification (x : ℝ) :
  (12 * x^4 + 6 * x^2) / (3 * x) - (-2 * x)^2 * (x + 1) = 2 * x - 4 * x^2 :=
by sorry

end NUMINAMATH_CALUDE_expression_simplification_l1802_180253


namespace NUMINAMATH_CALUDE_abs_add_ge_abs_add_x_range_when_eq_possible_values_l1802_180268

-- 1. Triangle inequality for absolute values
theorem abs_add_ge_abs_add (a b : ℚ) : |a| + |b| ≥ |a + b| := by sorry

-- 2. Range of x when |x|+2015=|x-2015|
theorem x_range_when_eq (x : ℝ) (h : |x| + 2015 = |x - 2015|) : x ≤ 0 := by sorry

-- 3. Possible values of a₁+a₂ given conditions
theorem possible_values (a₁ a₂ a₃ a₄ : ℝ) 
  (h1 : |a₁ + a₂| + |a₃ + a₄| = 15) 
  (h2 : |a₁ + a₂ + a₃ + a₄| = 5) : 
  (a₁ + a₂ = 10) ∨ (a₁ + a₂ = -10) ∨ (a₁ + a₂ = 5) ∨ (a₁ + a₂ = -5) := by sorry

end NUMINAMATH_CALUDE_abs_add_ge_abs_add_x_range_when_eq_possible_values_l1802_180268


namespace NUMINAMATH_CALUDE_cookie_count_l1802_180220

/-- Represents the dimensions of a rectangular object -/
structure Dimensions where
  length : ℕ
  width : ℕ

/-- Calculates the number of full smaller rectangles that can fit into a larger rectangle -/
def fullRectanglesFit (large : Dimensions) (small : Dimensions) : ℕ :=
  (large.length / small.length) * (large.width / small.width)

theorem cookie_count :
  let sheet := Dimensions.mk 30 24
  let cookie := Dimensions.mk 3 4
  fullRectanglesFit sheet cookie = 60 := by
  sorry

#eval fullRectanglesFit (Dimensions.mk 30 24) (Dimensions.mk 3 4)

end NUMINAMATH_CALUDE_cookie_count_l1802_180220


namespace NUMINAMATH_CALUDE_artist_paintings_l1802_180291

/-- Calculates the number of paintings an artist can complete in a given number of weeks. -/
def paintings_completed (hours_per_week : ℕ) (hours_per_painting : ℕ) (num_weeks : ℕ) : ℕ :=
  (hours_per_week / hours_per_painting) * num_weeks

/-- Proves that an artist spending 30 hours per week painting, taking 3 hours per painting,
    can complete 40 paintings in 4 weeks. -/
theorem artist_paintings : paintings_completed 30 3 4 = 40 := by
  sorry

end NUMINAMATH_CALUDE_artist_paintings_l1802_180291


namespace NUMINAMATH_CALUDE_probability_three_tails_one_head_l1802_180270

def coin_toss_probability : ℚ := 1/2

def number_of_coins : ℕ := 4

def number_of_tails : ℕ := 3

def number_of_heads : ℕ := 1

def number_of_favorable_outcomes : ℕ := 4

theorem probability_three_tails_one_head :
  (number_of_favorable_outcomes : ℚ) * coin_toss_probability ^ number_of_coins = 1/4 :=
sorry

end NUMINAMATH_CALUDE_probability_three_tails_one_head_l1802_180270


namespace NUMINAMATH_CALUDE_round_31083_58_to_two_sig_figs_l1802_180288

/-- Rounds a number to a specified number of significant figures -/
def roundToSignificantFigures (x : ℝ) (n : ℕ) : ℝ := sorry

/-- Theorem: Rounding 31,083.58 to two significant figures results in 3.1 × 10^4 -/
theorem round_31083_58_to_two_sig_figs :
  roundToSignificantFigures 31083.58 2 = 3.1 * 10^4 := by sorry

end NUMINAMATH_CALUDE_round_31083_58_to_two_sig_figs_l1802_180288


namespace NUMINAMATH_CALUDE_soccer_teams_count_l1802_180231

theorem soccer_teams_count : 
  ∃ (n : ℕ), 
    n > 0 ∧ 
    n * (n - 1) = 20 ∧ 
    n = 5 := by
  sorry

end NUMINAMATH_CALUDE_soccer_teams_count_l1802_180231


namespace NUMINAMATH_CALUDE_expression_one_value_expression_two_value_l1802_180269

-- Define the logarithm base 10 function
noncomputable def lg (x : ℝ) := Real.log x / Real.log 10

-- Theorem for the first expression
theorem expression_one_value :
  (-0.1)^0 + 32 * 2^(2/3) + (1/4)^(-(1/2)) = 5 := by sorry

-- Theorem for the second expression
theorem expression_two_value :
  lg 500 + lg (8/5) - (1/2) * lg 64 + 50 * (lg 2 + lg 5)^2 = 52 := by sorry

end NUMINAMATH_CALUDE_expression_one_value_expression_two_value_l1802_180269


namespace NUMINAMATH_CALUDE_paper_sheet_width_l1802_180249

theorem paper_sheet_width (sheet1_length sheet1_width sheet2_length : ℝ)
  (h1 : sheet1_length = 11)
  (h2 : sheet1_width = 17)
  (h3 : sheet2_length = 11)
  (h4 : 2 * sheet1_length * sheet1_width = 2 * sheet2_length * sheet2_width + 100) :
  sheet2_width = 12.45 := by
  sorry

end NUMINAMATH_CALUDE_paper_sheet_width_l1802_180249


namespace NUMINAMATH_CALUDE_arithmetic_sequence_properties_l1802_180294

/-- An arithmetic sequence with its properties -/
structure ArithmeticSequence where
  a : ℕ+ → ℝ  -- The sequence
  d : ℝ        -- Common difference
  S : ℕ+ → ℝ  -- Sum function
  sum_def : ∀ n : ℕ+, S n = n * a 1 + n * (n - 1) / 2 * d
  seq_def : ∀ n : ℕ+, a n = a 1 + (n - 1) * d

/-- Theorem about properties of an arithmetic sequence given certain conditions -/
theorem arithmetic_sequence_properties (seq : ArithmeticSequence)
    (h : seq.S 6 > seq.S 7 ∧ seq.S 7 > seq.S 5) :
    seq.d < 0 ∧ 
    seq.S 11 > 0 ∧ 
    seq.S 12 > 0 ∧ 
    seq.S 13 < 0 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_properties_l1802_180294


namespace NUMINAMATH_CALUDE_dart_board_probability_l1802_180219

/-- The probability of a dart landing in the center square of a regular hexagon dart board -/
theorem dart_board_probability (s : ℝ) (h : s > 0) : 
  let hexagon_area := 3 * Real.sqrt 3 / 2 * s^2
  let center_square_area := s^2 / 3
  center_square_area / hexagon_area = 2 * Real.sqrt 3 / 27 := by
  sorry

end NUMINAMATH_CALUDE_dart_board_probability_l1802_180219


namespace NUMINAMATH_CALUDE_amp_composition_l1802_180267

-- Define the & operations
def postfix_amp (x : ℤ) : ℤ := 8 - x
def prefix_amp (x : ℤ) : ℤ := x - 8

-- The theorem to prove
theorem amp_composition : prefix_amp (postfix_amp 15) = -15 := by
  sorry

end NUMINAMATH_CALUDE_amp_composition_l1802_180267


namespace NUMINAMATH_CALUDE_board_traversal_paths_bound_l1802_180286

/-- A piece on an n × n board that can move one step at a time (up, down, left, or right) --/
structure Piece (n : ℕ) where
  x : Fin n
  y : Fin n

/-- The number of unique paths to traverse the entire n × n board --/
def t (n : ℕ) : ℕ := sorry

/-- The theorem to be proved --/
theorem board_traversal_paths_bound {n : ℕ} (h : n ≥ 100) :
  (1.25 : ℝ) < (t n : ℝ) ^ (1 / (n^2 : ℝ)) ∧ (t n : ℝ) ^ (1 / (n^2 : ℝ)) < Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_board_traversal_paths_bound_l1802_180286


namespace NUMINAMATH_CALUDE_batsman_average_after_12th_innings_l1802_180290

/-- Represents a batsman's performance over a series of innings -/
structure BatsmanPerformance where
  innings : ℕ
  totalRuns : ℕ
  average : ℚ

/-- Calculates the new average after an additional innings -/
def newAverage (bp : BatsmanPerformance) (newScore : ℕ) : ℚ :=
  (bp.totalRuns + newScore) / (bp.innings + 1)

theorem batsman_average_after_12th_innings
  (bp : BatsmanPerformance)
  (h1 : bp.innings = 11)
  (h2 : newAverage bp 48 = bp.average + 2)
  : newAverage bp 48 = 26 := by
  sorry

end NUMINAMATH_CALUDE_batsman_average_after_12th_innings_l1802_180290


namespace NUMINAMATH_CALUDE_base9_multiplication_l1802_180200

/-- Converts a base 9 number represented as a list of digits to its decimal equivalent -/
def base9ToDecimal (digits : List Nat) : Nat :=
  digits.foldl (fun acc d => 9 * acc + d) 0

/-- Converts a decimal number to its base 9 representation as a list of digits -/
def decimalToBase9 (n : Nat) : List Nat :=
  if n = 0 then [0] else
  let rec aux (m : Nat) (acc : List Nat) : List Nat :=
    if m = 0 then acc else aux (m / 9) ((m % 9) :: acc)
  aux n []

/-- The main theorem statement -/
theorem base9_multiplication :
  let a := base9ToDecimal [3, 2, 7]
  let b := base9ToDecimal [1, 2]
  decimalToBase9 (a * b) = [4, 0, 4, 5] := by
  sorry


end NUMINAMATH_CALUDE_base9_multiplication_l1802_180200


namespace NUMINAMATH_CALUDE_olivias_wallet_l1802_180235

/-- Calculates the remaining money in Olivia's wallet after shopping --/
def remaining_money (initial : ℕ) (supermarket : ℕ) (showroom : ℕ) : ℕ :=
  initial - supermarket - showroom

/-- Theorem stating that Olivia has 26 dollars left after shopping --/
theorem olivias_wallet : remaining_money 106 31 49 = 26 := by
  sorry

end NUMINAMATH_CALUDE_olivias_wallet_l1802_180235


namespace NUMINAMATH_CALUDE_bug_position_after_2021_jumps_l1802_180215

/-- Represents the seven points on the circle -/
inductive Point
| One | Two | Three | Four | Five | Six | Seven

/-- Determines if a point is prime -/
def isPrime : Point → Bool
  | Point.Two => true
  | Point.Three => true
  | Point.Five => true
  | Point.Seven => true
  | _ => false

/-- Calculates the next point based on the current point -/
def nextPoint (p : Point) : Point :=
  match p with
  | Point.One => Point.Four
  | Point.Two => Point.Four
  | Point.Three => Point.Five
  | Point.Four => Point.Seven
  | Point.Five => Point.Seven
  | Point.Six => Point.Two
  | Point.Seven => Point.Two

/-- Calculates the bug's position after n jumps -/
def bugPosition (start : Point) (n : Nat) : Point :=
  match n with
  | 0 => start
  | n + 1 => nextPoint (bugPosition start n)

/-- The main theorem to prove -/
theorem bug_position_after_2021_jumps :
  bugPosition Point.Seven 2021 = Point.Two :=
sorry

end NUMINAMATH_CALUDE_bug_position_after_2021_jumps_l1802_180215


namespace NUMINAMATH_CALUDE_new_person_weight_l1802_180216

theorem new_person_weight (n : ℕ) (avg_increase : ℝ) (old_weight : ℝ) :
  n = 10 →
  avg_increase = 2.5 →
  old_weight = 65 →
  (n : ℝ) * avg_increase + old_weight = 90 :=
by sorry

end NUMINAMATH_CALUDE_new_person_weight_l1802_180216


namespace NUMINAMATH_CALUDE_sum_of_decimals_l1802_180298

/-- The sum of 123.45 and 678.90 is equal to 802.35 -/
theorem sum_of_decimals : (123.45 : ℝ) + 678.90 = 802.35 := by sorry

end NUMINAMATH_CALUDE_sum_of_decimals_l1802_180298


namespace NUMINAMATH_CALUDE_peters_claim_impossible_l1802_180271

/-- Represents the shooting scenario with initial bullets, shots made, and successful hits -/
structure ShootingScenario where
  initialBullets : ℕ
  shotsMade : ℕ
  successfulHits : ℕ

/-- Calculates the total number of bullets available after successful hits -/
def totalBullets (s : ShootingScenario) : ℕ :=
  s.initialBullets + s.successfulHits * 5

/-- Defines when a shooting scenario is possible -/
def isPossible (s : ShootingScenario) : Prop :=
  totalBullets s ≥ s.shotsMade

/-- Theorem stating that Peter's claim is impossible -/
theorem peters_claim_impossible :
  ¬ isPossible ⟨5, 50, 8⟩ := by
  sorry


end NUMINAMATH_CALUDE_peters_claim_impossible_l1802_180271


namespace NUMINAMATH_CALUDE_projection_sphere_existence_l1802_180275

-- Define a type for points in 3D space
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

-- Define a type for lines in 3D space
structure Line3D where
  point : Point3D
  direction : Point3D

-- Define a type for a set of lines
def LineSet := List Line3D

-- Function to check if lines are pairwise non-parallel
def pairwiseNonParallel (lines : LineSet) : Prop := sorry

-- Function to perform orthogonal projection
def orthogonalProject (p : Point3D) (l : Line3D) : Point3D := sorry

-- Function to generate all points from repeated projections
def allProjectionPoints (o : Point3D) (lines : LineSet) : Set Point3D := sorry

-- Theorem statement
theorem projection_sphere_existence 
  (o : Point3D) 
  (lines : LineSet) 
  (h : pairwiseNonParallel lines) :
  ∃ (r : ℝ), ∀ p ∈ allProjectionPoints o lines, 
    (p.x - o.x)^2 + (p.y - o.y)^2 + (p.z - o.z)^2 ≤ r^2 := by
  sorry

end NUMINAMATH_CALUDE_projection_sphere_existence_l1802_180275


namespace NUMINAMATH_CALUDE_power_function_through_point_l1802_180201

theorem power_function_through_point (a k : ℝ) : 
  (∀ x : ℝ, (a - 1) * x^k = ((a - 1):ℝ) * x^k) → 
  (a - 1) * (Real.sqrt 2)^k = 2 → 
  a + k = 4 := by
sorry

end NUMINAMATH_CALUDE_power_function_through_point_l1802_180201


namespace NUMINAMATH_CALUDE_polynomial_simplification_and_division_l1802_180208

theorem polynomial_simplification_and_division (x : ℝ) (h : x ≠ -1) :
  ((3 * x^3 + 4 * x^2 - 5 * x + 2) - (2 * x^3 - x^2 + 6 * x - 8)) / (x + 1) =
  x^2 + 4 * x - 15 + 25 / (x + 1) := by
  sorry

end NUMINAMATH_CALUDE_polynomial_simplification_and_division_l1802_180208


namespace NUMINAMATH_CALUDE_cars_meeting_time_l1802_180256

/-- The time taken for two cars to meet under specific conditions -/
theorem cars_meeting_time (distance : ℝ) (speed1 speed2 : ℝ) : 
  distance = 450 →
  speed1 = 45 →
  speed2 = 30 →
  (2 * distance) / (speed1 + speed2) = 12 := by
  sorry

end NUMINAMATH_CALUDE_cars_meeting_time_l1802_180256


namespace NUMINAMATH_CALUDE_hawks_lost_percentage_l1802_180284

/-- Represents a team's game statistics -/
structure TeamStats where
  total_games : ℕ
  win_ratio : ℚ
  loss_ratio : ℚ

/-- Calculates the percentage of games lost -/
def percent_lost (stats : TeamStats) : ℚ :=
  (stats.loss_ratio / (stats.win_ratio + stats.loss_ratio)) * 100

theorem hawks_lost_percentage :
  let hawks : TeamStats := {
    total_games := 64,
    win_ratio := 5,
    loss_ratio := 3
  }
  percent_lost hawks = 37.5 := by sorry

end NUMINAMATH_CALUDE_hawks_lost_percentage_l1802_180284


namespace NUMINAMATH_CALUDE_earth_inhabitable_fraction_l1802_180243

theorem earth_inhabitable_fraction :
  let earth_surface := 1
  let land_fraction := (1 : ℚ) / 3
  let inhabitable_land_fraction := (1 : ℚ) / 3
  (land_fraction * inhabitable_land_fraction) * earth_surface = (1 : ℚ) / 9 := by
  sorry

end NUMINAMATH_CALUDE_earth_inhabitable_fraction_l1802_180243


namespace NUMINAMATH_CALUDE_league_games_count_l1802_180217

/-- The number of games played in a league season -/
def games_in_season (n : ℕ) (k : ℕ) : ℕ :=
  (n.choose 2) * k

theorem league_games_count :
  games_in_season 20 7 = 1330 := by
  sorry

end NUMINAMATH_CALUDE_league_games_count_l1802_180217


namespace NUMINAMATH_CALUDE_instantaneous_velocity_at_4_seconds_l1802_180264

-- Define the displacement function
def s (t : ℝ) : ℝ := 3 * t^2 + t + 4

-- Define the velocity function as the derivative of displacement
def v (t : ℝ) : ℝ := 6 * t + 1

-- State the theorem
theorem instantaneous_velocity_at_4_seconds :
  v 4 = 25 := by sorry

end NUMINAMATH_CALUDE_instantaneous_velocity_at_4_seconds_l1802_180264


namespace NUMINAMATH_CALUDE_square_triangle_area_equality_l1802_180212

theorem square_triangle_area_equality (square_perimeter : ℝ) (triangle_height : ℝ) (x : ℝ) :
  square_perimeter = 64 →
  triangle_height = 64 →
  (square_perimeter / 4) ^ 2 = (1 / 2) * triangle_height * x →
  x = 8 := by
  sorry

end NUMINAMATH_CALUDE_square_triangle_area_equality_l1802_180212


namespace NUMINAMATH_CALUDE_problem_solution_l1802_180247

-- Define the functions f and g
def f (x : ℝ) := abs x
def g (x : ℝ) := 2 * f x + abs (2 * x - 1)

-- State the theorem
theorem problem_solution :
  (∃ m : ℝ, ∀ x : ℝ, g x ≥ m) →
  (∀ a b c : ℝ, a > 0 → b > 0 → c > 0 → a + 2 * b + c = 1 →
    ((∀ x : ℝ, f x < 2 * x - 1 ↔ x > 1) ∧
     1 / (a + b) + 1 / (b + c) ≥ 4)) :=
by sorry

end NUMINAMATH_CALUDE_problem_solution_l1802_180247


namespace NUMINAMATH_CALUDE_merchant_profit_percentage_l1802_180226

def markup_percentage : ℝ := 0.30
def discount_percentage : ℝ := 0.10

theorem merchant_profit_percentage :
  let marked_price := 1 + markup_percentage
  let discounted_price := marked_price * (1 - discount_percentage)
  (discounted_price - 1) * 100 = 17 := by
sorry

end NUMINAMATH_CALUDE_merchant_profit_percentage_l1802_180226


namespace NUMINAMATH_CALUDE_min_p_for_quadratic_roots_in_unit_interval_l1802_180224

theorem min_p_for_quadratic_roots_in_unit_interval :
  (∃ (p : ℕ+),
    (∀ p' : ℕ+, p' < p →
      ¬∃ (q r : ℕ+),
        (∃ (x y : ℝ),
          0 < x ∧ x < 1 ∧
          0 < y ∧ y < 1 ∧
          x ≠ y ∧
          p' * x^2 - q * x + r = 0 ∧
          p' * y^2 - q * y + r = 0)) ∧
    (∃ (q r : ℕ+),
      ∃ (x y : ℝ),
        0 < x ∧ x < 1 ∧
        0 < y ∧ y < 1 ∧
        x ≠ y ∧
        p * x^2 - q * x + r = 0 ∧
        p * y^2 - q * y + r = 0)) ∧
  (∀ p : ℕ+,
    (∀ p' : ℕ+, p' < p →
      ¬∃ (q r : ℕ+),
        (∃ (x y : ℝ),
          0 < x ∧ x < 1 ∧
          0 < y ∧ y < 1 ∧
          x ≠ y ∧
          p' * x^2 - q * x + r = 0 ∧
          p' * y^2 - q * y + r = 0)) ∧
    (∃ (q r : ℕ+),
      ∃ (x y : ℝ),
        0 < x ∧ x < 1 ∧
        0 < y ∧ y < 1 ∧
        x ≠ y ∧
        p * x^2 - q * x + r = 0 ∧
        p * y^2 - q * y + r = 0) →
    p = 5) :=
by sorry

end NUMINAMATH_CALUDE_min_p_for_quadratic_roots_in_unit_interval_l1802_180224


namespace NUMINAMATH_CALUDE_honor_students_count_l1802_180242

theorem honor_students_count 
  (total_students : ℕ) 
  (girls : ℕ) 
  (boys : ℕ) 
  (honor_girls : ℕ) 
  (honor_boys : ℕ) :
  total_students < 30 →
  total_students = girls + boys →
  (honor_girls : ℚ) / girls = 3 / 13 →
  (honor_boys : ℚ) / boys = 4 / 11 →
  honor_girls + honor_boys = 7 :=
by sorry

end NUMINAMATH_CALUDE_honor_students_count_l1802_180242


namespace NUMINAMATH_CALUDE_triangle_shape_l1802_180295

/-- Given a triangle ABC with sides a, b, c opposite to angles A, B, C respectively,
    prove that if (a² + b²)sin(A - B) = (a² - b²)sin(A + B),
    then the triangle is either isosceles (A = B) or right-angled (2A + 2B = 180°). -/
theorem triangle_shape (a b c A B C : ℝ) (h : (a^2 + b^2) * Real.sin (A - B) = (a^2 - b^2) * Real.sin (A + B)) :
  A = B ∨ 2*A + 2*B = Real.pi := by
  sorry

end NUMINAMATH_CALUDE_triangle_shape_l1802_180295


namespace NUMINAMATH_CALUDE_circle_equation_circle_properties_l1802_180299

theorem circle_equation (x y : ℝ) : 
  (x^2 + y^2 + 2*x - 4*y - 6 = 0) ↔ ((x + 1)^2 + (y - 2)^2 = 11) :=
by sorry

theorem circle_properties :
  ∃ (center : ℝ × ℝ) (radius : ℝ),
    center = (-1, 2) ∧ 
    radius = Real.sqrt 11 ∧
    ∀ (x y : ℝ), x^2 + y^2 + 2*x - 4*y - 6 = 0 ↔ 
      (x - center.1)^2 + (y - center.2)^2 = radius^2 :=
by sorry

end NUMINAMATH_CALUDE_circle_equation_circle_properties_l1802_180299


namespace NUMINAMATH_CALUDE_rectangle_area_l1802_180246

theorem rectangle_area (width : ℝ) (length : ℝ) : 
  length = 4 * width →
  2 * length + 2 * width = 200 →
  length * width = 1600 := by
sorry

end NUMINAMATH_CALUDE_rectangle_area_l1802_180246


namespace NUMINAMATH_CALUDE_shirt_cost_calculation_l1802_180258

def shorts_cost : ℚ := 13.99
def jacket_cost : ℚ := 7.43
def total_cost : ℚ := 33.56

theorem shirt_cost_calculation :
  total_cost - (shorts_cost + jacket_cost) = 12.14 := by
  sorry

end NUMINAMATH_CALUDE_shirt_cost_calculation_l1802_180258


namespace NUMINAMATH_CALUDE_sock_selection_theorem_l1802_180206

theorem sock_selection_theorem :
  let n : ℕ := 7  -- Total number of socks
  let k : ℕ := 4  -- Number of socks to choose
  Nat.choose n k = 35 := by
  sorry

end NUMINAMATH_CALUDE_sock_selection_theorem_l1802_180206


namespace NUMINAMATH_CALUDE_annual_interest_rate_is_33_point_33_percent_l1802_180285

/-- Represents the banker's gain in rupees -/
def bankers_gain : ℝ := 360

/-- Represents the banker's discount in rupees -/
def bankers_discount : ℝ := 1360

/-- Represents the time period in years -/
def time : ℝ := 3

/-- Calculates the true discount based on banker's discount and banker's gain -/
def true_discount : ℝ := bankers_discount - bankers_gain

/-- Calculates the present value based on banker's discount and banker's gain -/
def present_value : ℝ := bankers_discount - bankers_gain

/-- Calculates the face value as the sum of present value and true discount -/
def face_value : ℝ := present_value + true_discount

/-- Theorem stating that the annual interest rate is 100/3 percent -/
theorem annual_interest_rate_is_33_point_33_percent :
  ∃ (r : ℝ), r = 100 / 3 ∧ true_discount = (present_value * r * time) / 100 :=
sorry

end NUMINAMATH_CALUDE_annual_interest_rate_is_33_point_33_percent_l1802_180285


namespace NUMINAMATH_CALUDE_ella_sold_200_apples_l1802_180255

/-- The number of apples Ella sold -/
def apples_sold (bags_of_20 bags_of_25 apples_per_bag_20 apples_per_bag_25 apples_left : ℕ) : ℕ :=
  bags_of_20 * apples_per_bag_20 + bags_of_25 * apples_per_bag_25 - apples_left

/-- Theorem stating that Ella sold 200 apples -/
theorem ella_sold_200_apples :
  apples_sold 4 6 20 25 30 = 200 := by
  sorry

end NUMINAMATH_CALUDE_ella_sold_200_apples_l1802_180255


namespace NUMINAMATH_CALUDE_difference_divisible_by_99_l1802_180238

/-- Represents a three-digit number formed by digits a, b, and c -/
def three_digit_number (a b c : ℕ) : ℕ := 100 * a + 10 * b + c

/-- The largest three-digit number formed by digits a, b, and c where a > b > c -/
def largest_number (a b c : ℕ) : ℕ := three_digit_number a b c

/-- The smallest three-digit number formed by digits a, b, and c where a > b > c -/
def smallest_number (a b c : ℕ) : ℕ := three_digit_number c b a

theorem difference_divisible_by_99 (a b c : ℕ) 
  (h1 : a > b) (h2 : b > c) (h3 : c > 0) (h4 : a < 10) (h5 : b < 10) (h6 : c < 10) :
  ∃ k : ℕ, largest_number a b c - smallest_number a b c = 99 * k :=
sorry

end NUMINAMATH_CALUDE_difference_divisible_by_99_l1802_180238


namespace NUMINAMATH_CALUDE_corn_plants_per_row_l1802_180272

/-- Calculates the number of corn plants in each row given the water pumping conditions. -/
theorem corn_plants_per_row (
  pump_rate : ℝ)
  (pump_time : ℝ)
  (num_rows : ℕ)
  (water_per_plant : ℝ)
  (num_pigs : ℕ)
  (water_per_pig : ℝ)
  (num_ducks : ℕ)
  (water_per_duck : ℝ)
  (h_pump_rate : pump_rate = 3)
  (h_pump_time : pump_time = 25)
  (h_num_rows : num_rows = 4)
  (h_water_per_plant : water_per_plant = 0.5)
  (h_num_pigs : num_pigs = 10)
  (h_water_per_pig : water_per_pig = 4)
  (h_num_ducks : num_ducks = 20)
  (h_water_per_duck : water_per_duck = 0.25) :
  (pump_rate * pump_time - (num_pigs * water_per_pig + num_ducks * water_per_duck)) / (num_rows * water_per_plant) = 15 := by
sorry

end NUMINAMATH_CALUDE_corn_plants_per_row_l1802_180272


namespace NUMINAMATH_CALUDE_symmetric_points_sum_l1802_180240

/-- Given two points A and B that are symmetric about the y-axis, 
    prove that the sum of the offsets from A's coordinates equals 1. -/
theorem symmetric_points_sum (m n : ℝ) : 
  (∃ (A B : ℝ × ℝ), 
    A = (1 + m, 1 - n) ∧ 
    B = (-3, 2) ∧ 
    (A.1 = -B.1) ∧ 
    (A.2 = B.2)) →
  m + n = 1 := by
sorry

end NUMINAMATH_CALUDE_symmetric_points_sum_l1802_180240


namespace NUMINAMATH_CALUDE_fifteenth_term_ratio_l1802_180203

-- Define the sums of arithmetic sequences
def U (n : ℕ) (c f : ℚ) : ℚ := n * (2 * c + (n - 1) * f) / 2
def V (n : ℕ) (g h : ℚ) : ℚ := n * (2 * g + (n - 1) * h) / 2

-- Define the ratio condition
def ratio_condition (n : ℕ) (c f g h : ℚ) : Prop :=
  U n c f / V n g h = (5 * n^2 + 3 * n + 2) / (3 * n^2 + 2 * n + 30)

-- Define the 15th term of each sequence
def term_15 (c f : ℚ) : ℚ := c + 14 * f

-- Theorem statement
theorem fifteenth_term_ratio 
  (c f g h : ℚ) 
  (h1 : ∀ (n : ℕ), n > 0 → ratio_condition n c f g h) :
  term_15 c f / term_15 g h = 125 / 99 := by
  sorry

end NUMINAMATH_CALUDE_fifteenth_term_ratio_l1802_180203


namespace NUMINAMATH_CALUDE_angle_sum_equation_l1802_180210

theorem angle_sum_equation (α β : Real) (h : (1 + Real.sqrt 3 * Real.tan α) * (1 + Real.sqrt 3 * Real.tan β) = 4) :
  α + β = π / 3 := by
sorry

end NUMINAMATH_CALUDE_angle_sum_equation_l1802_180210


namespace NUMINAMATH_CALUDE_guaranteed_payoff_probability_l1802_180205

/-- Represents a fair six-sided die -/
def Die := Fin 6

/-- The score in the game is the sum of two die rolls -/
def score (roll1 roll2 : Die) : Nat := roll1.val + roll2.val + 2

/-- The maximum possible score in the game -/
def max_score : Nat := 12

/-- The number of players in the game -/
def num_players : Nat := 22

/-- The probability of rolling a specific number on a fair six-sided die -/
def prob_single_roll (n : Die) : Rat := 1 / 6

theorem guaranteed_payoff_probability :
  let guaranteed_score := max_score
  let prob_guaranteed_score := (prob_single_roll ⟨5, by norm_num⟩) * (prob_single_roll ⟨5, by norm_num⟩)
  (∀ s, s < guaranteed_score → ∃ (rolls : Fin num_players → Die × Die), 
    ∃ i, score (rolls i).1 (rolls i).2 ≥ s) ∧ 
  prob_guaranteed_score = 1 / 36 := by
  sorry

end NUMINAMATH_CALUDE_guaranteed_payoff_probability_l1802_180205


namespace NUMINAMATH_CALUDE_sheets_per_pack_calculation_l1802_180278

/-- Represents the number of sheets in a pack of notebook paper -/
def sheets_per_pack : ℕ := 100

/-- Represents the number of pages Chip takes per day per class -/
def pages_per_day_per_class : ℕ := 2

/-- Represents the number of days Chip takes notes per week -/
def days_per_week : ℕ := 5

/-- Represents the number of classes Chip has -/
def num_classes : ℕ := 5

/-- Represents the number of weeks Chip has been taking notes -/
def num_weeks : ℕ := 6

/-- Represents the number of packs Chip used -/
def packs_used : ℕ := 3

theorem sheets_per_pack_calculation :
  sheets_per_pack = 
    (pages_per_day_per_class * days_per_week * num_classes * num_weeks) / packs_used :=
by sorry

end NUMINAMATH_CALUDE_sheets_per_pack_calculation_l1802_180278


namespace NUMINAMATH_CALUDE_enclosed_area_theorem_l1802_180204

/-- The area enclosed by a curve composed of 9 congruent circular arcs, each with length π/3,
    centered at the vertices of a regular hexagon with side length 3 -/
def enclosed_area (arc_length : Real) (num_arcs : Nat) (hexagon_side : Real) : Real :=
  sorry

/-- The theorem stating the enclosed area for the given conditions -/
theorem enclosed_area_theorem :
  enclosed_area (π/3) 9 3 = (27 * Real.sqrt 3) / 2 + (3 * π) / 8 := by
  sorry

end NUMINAMATH_CALUDE_enclosed_area_theorem_l1802_180204


namespace NUMINAMATH_CALUDE_remainder_444_power_444_mod_13_l1802_180283

theorem remainder_444_power_444_mod_13 : 444^444 % 13 = 1 := by
  sorry

end NUMINAMATH_CALUDE_remainder_444_power_444_mod_13_l1802_180283


namespace NUMINAMATH_CALUDE_village_population_l1802_180263

theorem village_population (initial_population : ℕ) 
  (death_rate : ℚ) (leaving_rate : ℚ) (final_population : ℕ) : 
  initial_population = 4400 →
  death_rate = 5 / 100 →
  leaving_rate = 15 / 100 →
  final_population = 
    (initial_population - 
      (initial_population * death_rate).floor - 
      ((initial_population - (initial_population * death_rate).floor) * leaving_rate).floor) →
  final_population = 3553 := by
sorry

end NUMINAMATH_CALUDE_village_population_l1802_180263


namespace NUMINAMATH_CALUDE_apple_orchard_problem_l1802_180250

theorem apple_orchard_problem (total : ℝ) (fuji : ℝ) (gala : ℝ) : 
  (0.1 * total = total - fuji - gala) →
  (fuji + 0.1 * total = 238) →
  (fuji = 0.75 * total) →
  (gala = 42) := by
sorry

end NUMINAMATH_CALUDE_apple_orchard_problem_l1802_180250


namespace NUMINAMATH_CALUDE_exists_m_divisible_by_1988_l1802_180237

/-- Given f(x) = 3x + 2, prove that there exists a positive integer m 
    such that f^(100)(m) is divisible by 1988 -/
theorem exists_m_divisible_by_1988 :
  ∃ m : ℕ+, (3^100 : ℤ) * m.val + (3^100 - 1) ∣ 1988 := by
  sorry

end NUMINAMATH_CALUDE_exists_m_divisible_by_1988_l1802_180237


namespace NUMINAMATH_CALUDE_largest_solution_and_ratio_l1802_180274

theorem largest_solution_and_ratio (x : ℝ) (a b c d : ℤ) : 
  (7 * x / 4 - 2 = 5 / x) → 
  (x = (a + b * Real.sqrt c) / d) → 
  (∀ y : ℝ, (7 * y / 4 - 2 = 5 / y) → y ≤ x) →
  (x = (4 + 2 * Real.sqrt 39) / 7 ∧ a * c * d / b = 546) := by
  sorry

end NUMINAMATH_CALUDE_largest_solution_and_ratio_l1802_180274


namespace NUMINAMATH_CALUDE_quadratic_range_l1802_180261

theorem quadratic_range (a : ℝ) : 
  (∃ x : ℝ, x^2 + a*x + 1 < 0) ↔ a ∈ Set.Iio (-2) ∪ Set.Ioi 2 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_range_l1802_180261


namespace NUMINAMATH_CALUDE_units_digit_of_power_l1802_180279

theorem units_digit_of_power (n : ℕ) : (147 ^ 25) ^ 50 ≡ 9 [ZMOD 10] := by
  sorry

end NUMINAMATH_CALUDE_units_digit_of_power_l1802_180279


namespace NUMINAMATH_CALUDE_theater_ticket_price_l1802_180293

/-- The price of tickets at a theater with discounts for children and seniors -/
theorem theater_ticket_price :
  ∀ (adult_price : ℝ),
  (6 * adult_price + 5 * (adult_price / 2) + 3 * (adult_price * 0.75) = 42) →
  (10 * adult_price + 8 * (adult_price / 2) + 4 * (adult_price * 0.75) = 58.65) :=
by
  sorry

end NUMINAMATH_CALUDE_theater_ticket_price_l1802_180293


namespace NUMINAMATH_CALUDE_power_of_64_l1802_180225

theorem power_of_64 : (64 : ℝ) ^ (5/3) = 1024 := by
  sorry

end NUMINAMATH_CALUDE_power_of_64_l1802_180225


namespace NUMINAMATH_CALUDE_honey_barrel_problem_l1802_180222

theorem honey_barrel_problem (total_weight : ℝ) (half_removed_weight : ℝ) 
  (h1 : total_weight = 56)
  (h2 : half_removed_weight = 34) :
  ∃ (honey_weight barrel_weight : ℝ),
    honey_weight = 44 ∧
    barrel_weight = 12 ∧
    honey_weight + barrel_weight = total_weight ∧
    honey_weight / 2 + barrel_weight = half_removed_weight := by
  sorry

end NUMINAMATH_CALUDE_honey_barrel_problem_l1802_180222


namespace NUMINAMATH_CALUDE_river_speed_proof_l1802_180257

/-- Proves that the speed of the river is 1.2 kmph given the conditions of the rowing problem -/
theorem river_speed_proof 
  (rowing_speed : ℝ) 
  (total_time : ℝ) 
  (total_distance : ℝ) 
  (h1 : rowing_speed = 9) 
  (h2 : total_time = 1) 
  (h3 : total_distance = 8.84) : 
  ∃ (river_speed : ℝ), river_speed = 1.2 :=
by
  sorry

end NUMINAMATH_CALUDE_river_speed_proof_l1802_180257


namespace NUMINAMATH_CALUDE_vectors_not_collinear_l1802_180214

-- Define the vector space
variable {V : Type*} [AddCommGroup V] [Module ℝ V]

-- Define the proposition
def proposition (a b : V) : Prop :=
  ∀ k₁ k₂ : ℝ, k₁ • a + k₂ • b = 0 → k₁ ≠ 0 ∧ k₂ ≠ 0

-- State the theorem
theorem vectors_not_collinear (a b : V) :
  proposition a b → ¬(∃ (t : ℝ), a = t • b ∨ b = t • a) :=
by sorry

end NUMINAMATH_CALUDE_vectors_not_collinear_l1802_180214


namespace NUMINAMATH_CALUDE_simplify_sqrt_sum_l1802_180213

theorem simplify_sqrt_sum (a : ℝ) (h : 3 < a ∧ a < 5) : 
  Real.sqrt ((a - 2)^2) + Real.sqrt ((a - 8)^2) = 6 := by
  sorry

end NUMINAMATH_CALUDE_simplify_sqrt_sum_l1802_180213


namespace NUMINAMATH_CALUDE_functional_equation_properties_l1802_180241

/-- A function satisfying the given functional equation -/
def FunctionalEquation (f : ℝ → ℝ) : Prop :=
  ∀ x y : ℝ, f (x + y) + f (x - y) = 2 * f x * f y

theorem functional_equation_properties (f : ℝ → ℝ) 
  (h_eq : FunctionalEquation f) (h_nonzero : f 0 ≠ 0) : 
  (f 0 = 1) ∧ (∀ x : ℝ, f (-x) = f x) := by
  sorry

end NUMINAMATH_CALUDE_functional_equation_properties_l1802_180241


namespace NUMINAMATH_CALUDE_range_of_a_l1802_180202

theorem range_of_a (a : ℝ) : (∃ x : ℝ, Real.exp (2 * x) - (a - 3) * Real.exp x + 4 - 3 * a > 0) → a ≤ 4 / 3 := by
  sorry

end NUMINAMATH_CALUDE_range_of_a_l1802_180202


namespace NUMINAMATH_CALUDE_absolute_value_equation_solutions_l1802_180281

theorem absolute_value_equation_solutions :
  ∃! (s : Finset ℝ), s.card = 2 ∧ 
  (∀ x : ℝ, x ∈ s ↔ |x - 1| = |x - 2| + |x - 3| + |x - 4|) ∧
  (3 ∈ s ∧ 4 ∈ s) := by
  sorry

end NUMINAMATH_CALUDE_absolute_value_equation_solutions_l1802_180281


namespace NUMINAMATH_CALUDE_worker_efficiency_l1802_180207

/-- Given two workers p and q, where p can complete a work in 26 days,
    and p and q together can complete the same work in 16 days,
    prove that p is approximately 1.442% more efficient than q. -/
theorem worker_efficiency (p q : ℝ) (h1 : p > 0) (h2 : q > 0) 
  (h3 : p = 1 / 26) (h4 : p + q = 1 / 16) : 
  ∃ ε > 0, |((p - q) / q) * 100 - 1.442| < ε :=
sorry

end NUMINAMATH_CALUDE_worker_efficiency_l1802_180207


namespace NUMINAMATH_CALUDE_geometric_sequence_sum_l1802_180254

/-- Given a geometric sequence {a_n} with a_1 = 3 and a_1 + a_3 + a_5 = 21,
    prove that a_3 + a_5 + a_7 = 42 -/
theorem geometric_sequence_sum (a : ℕ → ℝ) (q : ℝ) : 
  (∀ n, a (n + 1) = a n * q) →  -- Geometric sequence definition
  a 1 = 3 →                    -- First condition
  a 1 + a 3 + a 5 = 21 →       -- Second condition
  a 3 + a 5 + a 7 = 42 :=      -- Conclusion to prove
by sorry

end NUMINAMATH_CALUDE_geometric_sequence_sum_l1802_180254


namespace NUMINAMATH_CALUDE_tangent_inclination_range_l1802_180221

open Real

theorem tangent_inclination_range (x : ℝ) : 
  let y := sin x
  let slope := cos x
  let θ := arctan slope
  0 ≤ θ ∧ θ < π ∧ (θ ≤ π/4 ∨ 3*π/4 ≤ θ) := by sorry

end NUMINAMATH_CALUDE_tangent_inclination_range_l1802_180221


namespace NUMINAMATH_CALUDE_committee_selection_l1802_180244

theorem committee_selection (n : ℕ) (k : ℕ) (h1 : n = 12) (h2 : k = 5) :
  Nat.choose n k = 792 := by
  sorry

end NUMINAMATH_CALUDE_committee_selection_l1802_180244


namespace NUMINAMATH_CALUDE_kilmer_park_tree_height_l1802_180228

/-- Calculates the height of a tree in inches after a given number of years -/
def tree_height_in_inches (initial_height : ℕ) (growth_rate : ℕ) (years : ℕ) (inches_per_foot : ℕ) : ℕ :=
  (initial_height + growth_rate * years) * inches_per_foot

/-- Proves that the height of the tree in Kilmer Park after 8 years is 1104 inches -/
theorem kilmer_park_tree_height : tree_height_in_inches 52 5 8 12 = 1104 := by
  sorry

end NUMINAMATH_CALUDE_kilmer_park_tree_height_l1802_180228


namespace NUMINAMATH_CALUDE_candy_remaining_l1802_180233

theorem candy_remaining (initial : ℝ) (eaten : ℝ) (h1 : initial = 67.5) (h2 : eaten = 64.3) :
  initial - eaten = 3.2 := by
sorry

end NUMINAMATH_CALUDE_candy_remaining_l1802_180233


namespace NUMINAMATH_CALUDE_quadratic_equation_from_means_l1802_180230

theorem quadratic_equation_from_means (a b : ℝ) : 
  (a + b) / 2 = 8 → 
  Real.sqrt (a * b) = 12 → 
  ∀ x, x^2 - 16*x + 144 = 0 ↔ (x = a ∨ x = b) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_equation_from_means_l1802_180230


namespace NUMINAMATH_CALUDE_common_chord_length_l1802_180273

-- Define the two circles
def circle1 (x y : ℝ) : Prop := x^2 + y^2 - 8 = 0
def circle2 (x y : ℝ) : Prop := x^2 + y^2 - 3*x + 4*y - 18 = 0

-- Define the common chord
def common_chord (x y : ℝ) : Prop := 3*x - 4*y + 10 = 0

-- Theorem statement
theorem common_chord_length :
  ∃ (length : ℝ), 
    (∀ (x y : ℝ), circle1 x y ∧ circle2 x y → common_chord x y) ∧
    length = 4 :=
sorry

end NUMINAMATH_CALUDE_common_chord_length_l1802_180273


namespace NUMINAMATH_CALUDE_equation_has_real_roots_l1802_180229

theorem equation_has_real_roots (a b : ℝ) : 
  ∃ x : ℝ, (x^2 / (x^2 - a^2) + x^2 / (x^2 - b^2) = 4) := by sorry

end NUMINAMATH_CALUDE_equation_has_real_roots_l1802_180229


namespace NUMINAMATH_CALUDE_cindy_paint_area_l1802_180259

/-- Given that Allen, Ben, and Cindy are painting a fence, where:
    1) The ratio of work done by Allen : Ben : Cindy is 3 : 5 : 2
    2) The total fence area to be painted is 300 square feet
    Prove that Cindy paints 60 square feet of the fence. -/
theorem cindy_paint_area (total_area : ℝ) (allen_ratio ben_ratio cindy_ratio : ℕ) :
  total_area = 300 ∧ 
  allen_ratio = 3 ∧ 
  ben_ratio = 5 ∧ 
  cindy_ratio = 2 →
  cindy_ratio * (total_area / (allen_ratio + ben_ratio + cindy_ratio)) = 60 :=
by sorry

end NUMINAMATH_CALUDE_cindy_paint_area_l1802_180259


namespace NUMINAMATH_CALUDE_hyperbola_asymptotes_l1802_180260

/-- The asymptotes of a hyperbola with equation (y^2 / 16) - (x^2 / 9) = 1
    shifted 5 units down along the y-axis are y = ± (4x/3) + 5 -/
theorem hyperbola_asymptotes (x y : ℝ) :
  let shifted_hyperbola := fun y => (y^2 / 16) - (x^2 / 9) = 1
  let asymptote₁ := fun x => (4 * x) / 3 + 5
  let asymptote₂ := fun x => -(4 * x) / 3 + 5
  shifted_hyperbola (y + 5) →
  (y = asymptote₁ x ∨ y = asymptote₂ x) :=
by sorry

end NUMINAMATH_CALUDE_hyperbola_asymptotes_l1802_180260


namespace NUMINAMATH_CALUDE_shekars_average_marks_l1802_180245

/-- Calculates the average marks given scores in five subjects -/
def averageMarks (math science socialStudies english biology : ℕ) : ℚ :=
  (math + science + socialStudies + english + biology : ℚ) / 5

/-- Theorem stating that Shekar's average marks are 75 -/
theorem shekars_average_marks :
  averageMarks 76 65 82 67 85 = 75 := by
  sorry

end NUMINAMATH_CALUDE_shekars_average_marks_l1802_180245


namespace NUMINAMATH_CALUDE_inequality_existence_l1802_180296

variable (a : ℝ)

theorem inequality_existence (h1 : a > 1) (h2 : a ≠ 2) :
  (¬ ∀ x : ℝ, (1 < x ∧ x < a) → (a < 2*x ∧ 2*x < a^2)) ∧
  (∃ x : ℝ, (a < 2*x ∧ 2*x < a^2) ∧ ¬(1 < x ∧ x < a)) := by
  sorry

end NUMINAMATH_CALUDE_inequality_existence_l1802_180296


namespace NUMINAMATH_CALUDE_specialNumberCount_is_70_l1802_180280

/-- The count of numbers between 200 and 899 (inclusive) with three different digits 
    that can be arranged in either strictly increasing or strictly decreasing order -/
def specialNumberCount : ℕ :=
  let lowerBound := 200
  let upperBound := 899
  let digitSet := {2, 3, 4, 5, 6, 7, 8}
  2 * (Finset.card digitSet).choose 3

theorem specialNumberCount_is_70 : specialNumberCount = 70 := by
  sorry

end NUMINAMATH_CALUDE_specialNumberCount_is_70_l1802_180280


namespace NUMINAMATH_CALUDE_geometric_sequence_sum_l1802_180236

theorem geometric_sequence_sum (a r : ℚ) (n : ℕ) (h1 : a = 1/3) (h2 : r = 1/3) :
  (a * (1 - r^n) / (1 - r) = 26/81) → n = 4 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_sum_l1802_180236


namespace NUMINAMATH_CALUDE_odd_function_property_l1802_180287

-- Define f as a function from ℝ to ℝ
variable (f : ℝ → ℝ)

-- Define g in terms of f
def g (x : ℝ) : ℝ := f x + 9

-- Theorem statement
theorem odd_function_property (h1 : ∀ x, f (-x) = -f x) 
                              (h2 : g (-2) = 3) : 
  f 2 = 6 := by
  sorry

end NUMINAMATH_CALUDE_odd_function_property_l1802_180287


namespace NUMINAMATH_CALUDE_change_in_f_l1802_180248

def f (x : ℝ) : ℝ := x^2 - 5*x

theorem change_in_f (x : ℝ) :
  (f (x + 2) - f x = 4*x - 6) ∧
  (f (x - 2) - f x = -4*x + 14) :=
by sorry

end NUMINAMATH_CALUDE_change_in_f_l1802_180248


namespace NUMINAMATH_CALUDE_ab_bounds_l1802_180262

theorem ab_bounds (a b c : ℝ) (h1 : a ≠ b) (h2 : c > 0)
  (h3 : a^4 - 2019*a = c) (h4 : b^4 - 2019*b = c) :
  -Real.sqrt c < a * b ∧ a * b < 0 := by
  sorry

end NUMINAMATH_CALUDE_ab_bounds_l1802_180262


namespace NUMINAMATH_CALUDE_equation_solution_l1802_180211

theorem equation_solution : ∃! x : ℚ, (x^2 + 2*x + 3) / (x + 4) = x + 5 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l1802_180211


namespace NUMINAMATH_CALUDE_quadratic_equation_roots_l1802_180218

theorem quadratic_equation_roots (m : ℝ) : 
  (∃ x : ℝ, 7 * x^2 + m * x = -6) ∧ (7 * 3^2 + m * 3 = -6) → 
  (7 * (2/7)^2 + m * (2/7) = -6) := by
sorry

end NUMINAMATH_CALUDE_quadratic_equation_roots_l1802_180218


namespace NUMINAMATH_CALUDE_chord_equation_l1802_180276

/-- Given a circle and a chord, prove the equation of the chord. -/
theorem chord_equation (x y : ℝ) :
  (x^2 + y^2 - 4*x - 5 = 0) →  -- Circle equation
  (∃ (a b : ℝ), (a + 3) / 2 = 3 ∧ (b + 1) / 2 = 1) →  -- Midpoint condition
  (x + y - 4 = 0) :=  -- Equation of line AB
by sorry

end NUMINAMATH_CALUDE_chord_equation_l1802_180276


namespace NUMINAMATH_CALUDE_solution_set_x_squared_leq_four_l1802_180292

theorem solution_set_x_squared_leq_four :
  {x : ℝ | x^2 ≤ 4} = {x : ℝ | -2 ≤ x ∧ x ≤ 2} := by
  sorry

end NUMINAMATH_CALUDE_solution_set_x_squared_leq_four_l1802_180292


namespace NUMINAMATH_CALUDE_exactly_one_hit_probability_l1802_180223

def probability_exactly_one_hit (p_a p_b : ℝ) : ℝ :=
  p_a * (1 - p_b) + (1 - p_a) * p_b

theorem exactly_one_hit_probability :
  let p_a : ℝ := 1/2
  let p_b : ℝ := 1/3
  probability_exactly_one_hit p_a p_b = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_exactly_one_hit_probability_l1802_180223

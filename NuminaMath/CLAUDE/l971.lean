import Mathlib

namespace NUMINAMATH_CALUDE_not_divisible_by_121_l971_97199

theorem not_divisible_by_121 (n : ℤ) : ¬(121 ∣ (n^2 + 3*n + 5)) := by
  sorry

end NUMINAMATH_CALUDE_not_divisible_by_121_l971_97199


namespace NUMINAMATH_CALUDE_max_students_above_average_l971_97147

theorem max_students_above_average (n : ℕ) (score1 score2 : ℚ) : 
  n = 150 →
  score1 > score2 →
  (n - 1) * score1 + score2 > n * ((n - 1) * score1 + score2) / n →
  ∃ (m : ℕ), m ≤ n ∧ m = 149 ∧ 
    (∀ (k : ℕ), k > m → 
      k * score1 + (n - k) * score2 ≤ n * (k * score1 + (n - k) * score2) / n) :=
by sorry

end NUMINAMATH_CALUDE_max_students_above_average_l971_97147


namespace NUMINAMATH_CALUDE_preimage_of_3_1_l971_97120

/-- The mapping f from ℝ² to ℝ² -/
def f (p : ℝ × ℝ) : ℝ × ℝ := (p.1 - p.2, p.1 + p.2)

/-- Theorem stating that (2, -1) is the pre-image of (3, 1) under f -/
theorem preimage_of_3_1 : f (2, -1) = (3, 1) := by
  sorry

end NUMINAMATH_CALUDE_preimage_of_3_1_l971_97120


namespace NUMINAMATH_CALUDE_complex_real_condition_l971_97174

/-- If z = m^2 - 1 + (m-1)i is a real number and m is real, then m = 1 -/
theorem complex_real_condition (m : ℝ) : 
  let z : ℂ := m^2 - 1 + (m - 1) * Complex.I
  (z.im = 0) → m = 1 := by
sorry

end NUMINAMATH_CALUDE_complex_real_condition_l971_97174


namespace NUMINAMATH_CALUDE_room_painting_cost_l971_97175

/-- Calculate the cost of painting a room's walls given its dimensions and openings. -/
def paintingCost (roomLength roomWidth roomHeight : ℝ)
                 (doorCount doorLength doorHeight : ℝ)
                 (largeWindowCount largeWindowLength largeWindowHeight : ℝ)
                 (smallWindowCount smallWindowLength smallWindowHeight : ℝ)
                 (costPerSqm : ℝ) : ℝ :=
  let wallArea := 2 * (roomLength * roomHeight + roomWidth * roomHeight)
  let doorArea := doorCount * doorLength * doorHeight
  let largeWindowArea := largeWindowCount * largeWindowLength * largeWindowHeight
  let smallWindowArea := smallWindowCount * smallWindowLength * smallWindowHeight
  let paintableArea := wallArea - (doorArea + largeWindowArea + smallWindowArea)
  paintableArea * costPerSqm

/-- Theorem stating that the cost of painting the room with given dimensions is 474 Rs. -/
theorem room_painting_cost :
  paintingCost 10 7 5 2 1 3 1 2 1.5 2 1 1.5 3 = 474 := by
  sorry

end NUMINAMATH_CALUDE_room_painting_cost_l971_97175


namespace NUMINAMATH_CALUDE_sergey_mistake_l971_97162

theorem sergey_mistake : ¬∃ a : ℤ, a % 15 = 8 ∧ a % 20 = 17 := by
  sorry

end NUMINAMATH_CALUDE_sergey_mistake_l971_97162


namespace NUMINAMATH_CALUDE_spherical_to_rectangular_conversion_l971_97143

/-- Proves that the conversion from spherical coordinates (10, 3π/4, π/4) to 
    rectangular coordinates results in (-5, 5, 5√2) -/
theorem spherical_to_rectangular_conversion :
  let ρ : ℝ := 10
  let θ : ℝ := 3 * Real.pi / 4
  let φ : ℝ := Real.pi / 4
  let x : ℝ := ρ * Real.sin φ * Real.cos θ
  let y : ℝ := ρ * Real.sin φ * Real.sin θ
  let z : ℝ := ρ * Real.cos φ
  (x = -5) ∧ (y = 5) ∧ (z = 5 * Real.sqrt 2) :=
by sorry

end NUMINAMATH_CALUDE_spherical_to_rectangular_conversion_l971_97143


namespace NUMINAMATH_CALUDE_digits_first_1500_even_integers_l971_97103

/-- The number of digits in a positive integer -/
def numDigits (n : ℕ) : ℕ := sorry

/-- The sum of digits for all positive even integers up to n -/
def sumDigitsEven (n : ℕ) : ℕ := sorry

/-- The 1500th positive even integer -/
def evenInteger1500 : ℕ := 3000

theorem digits_first_1500_even_integers :
  sumDigitsEven evenInteger1500 = 5448 := by sorry

end NUMINAMATH_CALUDE_digits_first_1500_even_integers_l971_97103


namespace NUMINAMATH_CALUDE_gcd_1729_587_l971_97106

theorem gcd_1729_587 : Nat.gcd 1729 587 = 1 := by
  sorry

end NUMINAMATH_CALUDE_gcd_1729_587_l971_97106


namespace NUMINAMATH_CALUDE_empty_solution_set_range_l971_97121

-- Define the inequality
def inequality (x m : ℝ) : Prop := |x - 1| + |x - m| < 2 * m

-- Define the theorem
theorem empty_solution_set_range :
  (∀ m : ℝ, (0 < m ∧ m < 1/3) ↔ ∀ x : ℝ, ¬(inequality x m)) :=
sorry

end NUMINAMATH_CALUDE_empty_solution_set_range_l971_97121


namespace NUMINAMATH_CALUDE_ned_weekly_earnings_l971_97160

/-- Calculates weekly earnings from selling left-handed mice --/
def weekly_earnings (normal_price : ℝ) (price_increase_percent : ℝ) 
                    (daily_sales : ℕ) (open_days : ℕ) : ℝ :=
  let left_handed_price := normal_price * (1 + price_increase_percent)
  let daily_earnings := left_handed_price * daily_sales
  daily_earnings * open_days

/-- Theorem stating Ned's weekly earnings --/
theorem ned_weekly_earnings :
  weekly_earnings 120 0.3 25 4 = 15600 := by
  sorry

#eval weekly_earnings 120 0.3 25 4

end NUMINAMATH_CALUDE_ned_weekly_earnings_l971_97160


namespace NUMINAMATH_CALUDE_car_b_speed_l971_97191

/-- Proves that given the initial conditions and final state, Car B's speed is 50 mph -/
theorem car_b_speed (initial_distance : ℝ) (car_a_speed : ℝ) (time : ℝ) (final_distance : ℝ) :
  initial_distance = 30 →
  car_a_speed = 58 →
  time = 4.75 →
  final_distance = 8 →
  (car_a_speed * time - initial_distance - final_distance) / time = 50 := by
sorry

end NUMINAMATH_CALUDE_car_b_speed_l971_97191


namespace NUMINAMATH_CALUDE_same_prime_divisors_same_outcome_l971_97133

/-- The number game as described in the problem -/
def NumberGame (k : ℕ) (n : ℕ) : Prop :=
  k > 2 ∧ n ≥ k

/-- A number is good if Banana has a winning strategy -/
def IsGood (k : ℕ) (n : ℕ) : Prop :=
  NumberGame k n ∧ sorry -- Definition of good number

/-- Two numbers have the same prime divisors up to k -/
def SamePrimeDivisorsUpTo (k : ℕ) (n n' : ℕ) : Prop :=
  ∀ p : ℕ, p ≤ k → Prime p → (p ∣ n ↔ p ∣ n')

/-- Main theorem: numbers with same prime divisors up to k have the same game outcome -/
theorem same_prime_divisors_same_outcome (k : ℕ) (n n' : ℕ) :
  NumberGame k n → NumberGame k n' → SamePrimeDivisorsUpTo k n n' →
  (IsGood k n ↔ IsGood k n') :=
sorry

end NUMINAMATH_CALUDE_same_prime_divisors_same_outcome_l971_97133


namespace NUMINAMATH_CALUDE_rational_function_value_l971_97189

structure RationalFunction where
  p : ℝ → ℝ
  q : ℝ → ℝ
  p_linear : ∃ a b : ℝ, ∀ x, p x = a * x + b
  q_quadratic : ∃ a b c : ℝ, ∀ x, q x = a * x^2 + b * x + c

def has_asymptotes (f : RationalFunction) (x₁ x₂ : ℝ) : Prop :=
  f.q x₁ = 0 ∧ f.q x₂ = 0

def passes_through (f : RationalFunction) (x y : ℝ) : Prop :=
  f.q x ≠ 0 ∧ f.p x / f.q x = y

theorem rational_function_value (f : RationalFunction) :
  has_asymptotes f (-4) 1 →
  passes_through f 0 0 →
  passes_through f 2 (-2) →
  f.p 3 / f.q 3 = -9/7 := by
    sorry

end NUMINAMATH_CALUDE_rational_function_value_l971_97189


namespace NUMINAMATH_CALUDE_no_four_digit_reverse_diff_1008_l971_97168

theorem no_four_digit_reverse_diff_1008 : 
  ¬ ∃ (a b c d : ℕ), 
    (1000 ≤ 1000 * a + 100 * b + 10 * c + d) ∧ 
    (1000 * a + 100 * b + 10 * c + d < 10000) ∧
    ((1000 * a + 100 * b + 10 * c + d) - (1000 * d + 100 * c + 10 * b + a) = 1008) :=
by sorry

end NUMINAMATH_CALUDE_no_four_digit_reverse_diff_1008_l971_97168


namespace NUMINAMATH_CALUDE_line_intersection_x_axis_l971_97172

/-- A line passing through two points (x₁, y₁) and (x₂, y₂) -/
structure Line where
  x₁ : ℝ
  y₁ : ℝ
  x₂ : ℝ
  y₂ : ℝ

/-- The x-coordinate of the intersection point of a line with the x-axis -/
def x_axis_intersection (l : Line) : ℝ :=
  sorry

theorem line_intersection_x_axis (l : Line) : 
  l.x₁ = 6 ∧ l.y₁ = 22 ∧ l.x₂ = -3 ∧ l.y₂ = 1 → x_axis_intersection l = -24/7 := by
  sorry

end NUMINAMATH_CALUDE_line_intersection_x_axis_l971_97172


namespace NUMINAMATH_CALUDE_sqrt_identity_l971_97115

theorem sqrt_identity : (Real.sqrt 3 + Real.sqrt 2) * (Real.sqrt 3 - Real.sqrt 2)^2 = Real.sqrt 3 - Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_identity_l971_97115


namespace NUMINAMATH_CALUDE_millicent_book_fraction_l971_97198

/-- Given that:
    - Harold has 1/2 as many books as Millicent
    - Harold brings 1/3 of his books to the new home
    - The new home's library capacity is 5/6 of Millicent's old library capacity
    Prove that Millicent brings 2/3 of her books to the new home -/
theorem millicent_book_fraction (M : ℝ) (F : ℝ) (H : ℝ) :
  H = (1 / 2 : ℝ) * M →
  (1 / 3 : ℝ) * H + F * M = (5 / 6 : ℝ) * M →
  F = (2 / 3 : ℝ) := by
  sorry

end NUMINAMATH_CALUDE_millicent_book_fraction_l971_97198


namespace NUMINAMATH_CALUDE_greatest_x_under_conditions_l971_97187

theorem greatest_x_under_conditions (x : ℕ) 
  (h1 : x > 0) 
  (h2 : ∃ k : ℕ, x = 5 * k) 
  (h3 : x^3 < 1331) : 
  ∀ y : ℕ, (y > 0 ∧ (∃ m : ℕ, y = 5 * m) ∧ y^3 < 1331) → y ≤ 10 :=
sorry

end NUMINAMATH_CALUDE_greatest_x_under_conditions_l971_97187


namespace NUMINAMATH_CALUDE_intersection_of_A_and_B_l971_97107

-- Define the sets A and B
def A : Set ℝ := {x | x^2 - 2*x - 3 > 0}
def B : Set ℝ := {x | x + 3 < 0}

-- State the theorem
theorem intersection_of_A_and_B : A ∩ B = {x | x < -3} := by sorry

end NUMINAMATH_CALUDE_intersection_of_A_and_B_l971_97107


namespace NUMINAMATH_CALUDE_one_prime_between_90_and_100_l971_97104

theorem one_prime_between_90_and_100 : 
  ∃! p, Prime p ∧ 90 < p ∧ p < 100 := by
sorry

end NUMINAMATH_CALUDE_one_prime_between_90_and_100_l971_97104


namespace NUMINAMATH_CALUDE_height_correction_percentage_l971_97119

/-- Proves that given a candidate's actual height of 5 feet 8 inches (68 inches),
    and an initial overstatement of 25%, the percentage correction from the
    stated height to the actual height is 20%. -/
theorem height_correction_percentage (actual_height : ℝ) (stated_height : ℝ) :
  actual_height = 68 →
  stated_height = actual_height * 1.25 →
  (stated_height - actual_height) / stated_height * 100 = 20 := by
  sorry

end NUMINAMATH_CALUDE_height_correction_percentage_l971_97119


namespace NUMINAMATH_CALUDE_paint_cans_used_l971_97105

/-- Represents the number of rooms that can be painted with a given amount of paint -/
structure PaintCapacity where
  rooms : ℕ

/-- Represents the number of paint cans -/
structure PaintCans where
  count : ℕ

/-- The painting scenario -/
structure PaintingScenario where
  initialCapacity : PaintCapacity
  lostCans : PaintCans
  finalCapacity : PaintCapacity

/-- The theorem to prove -/
theorem paint_cans_used (scenario : PaintingScenario) 
  (h1 : scenario.initialCapacity.rooms = 40)
  (h2 : scenario.lostCans.count = 5)
  (h3 : scenario.finalCapacity.rooms = 30) :
  ∃ (usedCans : PaintCans), usedCans.count = 15 ∧ 
    usedCans.count * (scenario.initialCapacity.rooms / (scenario.initialCapacity.rooms - scenario.finalCapacity.rooms)) = scenario.finalCapacity.rooms :=
by sorry

end NUMINAMATH_CALUDE_paint_cans_used_l971_97105


namespace NUMINAMATH_CALUDE_sin_sum_from_sin_cos_sums_l971_97148

theorem sin_sum_from_sin_cos_sums (x y : Real) 
  (h1 : Real.sin x + Real.sin y = Real.sqrt 2 / 2)
  (h2 : Real.cos x + Real.cos y = Real.sqrt 6 / 2) :
  Real.sin (x + y) = Real.sqrt 3 / 2 := by
sorry

end NUMINAMATH_CALUDE_sin_sum_from_sin_cos_sums_l971_97148


namespace NUMINAMATH_CALUDE_arithmetic_calculation_l971_97117

theorem arithmetic_calculation : 2 + 7 * 3 - 4 + 8 * 2 / 4 = 23 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_calculation_l971_97117


namespace NUMINAMATH_CALUDE_ellipse_foci_y_axis_l971_97159

theorem ellipse_foci_y_axis (k : ℝ) :
  (∀ x y : ℝ, x^2 / (9 - k) + y^2 / (k - 4) = 1) →
  (∃ a b c : ℝ, a > 0 ∧ b > 0 ∧ c > 0 ∧ b^2 = a^2 + c^2 ∧
    ∀ x y : ℝ, x^2 / a^2 + y^2 / b^2 = 1 ↔ x^2 / (9 - k) + y^2 / (k - 4) = 1) →
  13/2 < k ∧ k < 9 :=
by sorry

end NUMINAMATH_CALUDE_ellipse_foci_y_axis_l971_97159


namespace NUMINAMATH_CALUDE_selection_schemes_count_l971_97134

def total_people : ℕ := 6
def cities : ℕ := 4
def excluded_from_paris : ℕ := 2

theorem selection_schemes_count :
  (total_people.choose cities) *
  (cities.factorial) -
  (excluded_from_paris * (total_people - 1).choose (cities - 1) * (cities - 1).factorial) = 240 :=
sorry

end NUMINAMATH_CALUDE_selection_schemes_count_l971_97134


namespace NUMINAMATH_CALUDE_polynomial_value_l971_97129

/-- Given a polynomial G satisfying certain conditions, prove G(8) = 491/3 -/
theorem polynomial_value (G : ℝ → ℝ) : 
  (∀ x, G (4 * x) / G (x + 2) = 4 - (20 * x + 24) / (x^2 + 4 * x + 4)) →
  G 4 = 35 →
  G 8 = 491 / 3 := by
sorry

end NUMINAMATH_CALUDE_polynomial_value_l971_97129


namespace NUMINAMATH_CALUDE_boy_scout_percentage_l971_97139

/-- Represents the composition of a group of scouts -/
structure ScoutGroup where
  total : ℝ
  boys : ℝ
  girls : ℝ
  total_is_sum : total = boys + girls

/-- Represents the percentage of scouts with signed permission slips -/
structure PermissionSlips where
  total_percent : ℝ
  boys_percent : ℝ
  girls_percent : ℝ
  total_is_70_percent : total_percent = 0.7
  boys_is_75_percent : boys_percent = 0.75
  girls_is_62_5_percent : girls_percent = 0.625

theorem boy_scout_percentage 
  (group : ScoutGroup) 
  (slips : PermissionSlips) : 
  group.boys / group.total = 0.6 := by
  sorry

end NUMINAMATH_CALUDE_boy_scout_percentage_l971_97139


namespace NUMINAMATH_CALUDE_batsman_average_theorem_l971_97163

/-- Represents a batsman's performance -/
structure Batsman where
  innings : ℕ
  totalRuns : ℕ
  lastInningScore : ℕ
  averageIncrease : ℚ

/-- Calculates the new average of a batsman after their latest inning -/
def newAverage (b : Batsman) : ℚ :=
  (b.totalRuns + b.lastInningScore) / b.innings

theorem batsman_average_theorem (b : Batsman) 
  (h1 : b.innings = 17)
  (h2 : b.lastInningScore = 85)
  (h3 : b.averageIncrease = 3)
  (h4 : newAverage b = (b.totalRuns / (b.innings - 1) + b.averageIncrease)) :
  newAverage b = 37 := by
  sorry

end NUMINAMATH_CALUDE_batsman_average_theorem_l971_97163


namespace NUMINAMATH_CALUDE_rectangles_in_4x5_grid_l971_97196

/-- The number of rectangles in a horizontal strip of width n -/
def horizontalRectangles (n : ℕ) : ℕ := n * (n + 1) / 2

/-- The number of rectangles in a vertical strip of height m -/
def verticalRectangles (m : ℕ) : ℕ := m * (m + 1) / 2

/-- The total number of rectangles in an m×n grid -/
def totalRectangles (m n : ℕ) : ℕ :=
  m * horizontalRectangles n + n * verticalRectangles m - m * n

theorem rectangles_in_4x5_grid :
  totalRectangles 4 5 = 24 := by
  sorry

end NUMINAMATH_CALUDE_rectangles_in_4x5_grid_l971_97196


namespace NUMINAMATH_CALUDE_fraction_to_zero_power_l971_97183

theorem fraction_to_zero_power :
  let x : ℚ := -123456789 / 9876543210
  x ≠ 0 →
  x^0 = 1 := by sorry

end NUMINAMATH_CALUDE_fraction_to_zero_power_l971_97183


namespace NUMINAMATH_CALUDE_smallest_number_with_remainders_l971_97167

theorem smallest_number_with_remainders : ∃! x : ℕ,
  (x % 5 = 2) ∧ (x % 6 = 2) ∧ (x % 7 = 3) ∧
  (∀ y : ℕ, (y % 5 = 2) ∧ (y % 6 = 2) ∧ (y % 7 = 3) → x ≤ y) ∧
  x = 122 := by
sorry

end NUMINAMATH_CALUDE_smallest_number_with_remainders_l971_97167


namespace NUMINAMATH_CALUDE_subtraction_preserves_inequality_l971_97101

theorem subtraction_preserves_inequality (a b : ℝ) : a < b → a - 1 < b - 1 := by
  sorry

end NUMINAMATH_CALUDE_subtraction_preserves_inequality_l971_97101


namespace NUMINAMATH_CALUDE_polynomial_root_implies_h_value_l971_97116

theorem polynomial_root_implies_h_value : 
  ∀ h : ℚ, (3 : ℚ)^3 + h * 3 + 14 = 0 → h = -41/3 :=
by sorry

end NUMINAMATH_CALUDE_polynomial_root_implies_h_value_l971_97116


namespace NUMINAMATH_CALUDE_cubic_with_infinite_equal_pairs_has_integer_root_l971_97141

/-- A cubic polynomial with integer coefficients -/
structure CubicPolynomial where
  a : ℤ
  b : ℤ
  c : ℤ
  d : ℤ
  a_nonzero : a ≠ 0

/-- Evaluation of a cubic polynomial at a given integer -/
def eval (P : CubicPolynomial) (x : ℤ) : ℤ :=
  P.a * x^3 + P.b * x^2 + P.c * x + P.d

/-- The property that there are infinitely many pairs of distinct integers (x, y) such that xP(x) = yP(y) -/
def has_infinite_equal_pairs (P : CubicPolynomial) : Prop :=
  ∀ n : ℕ, ∃ x y : ℤ, x ≠ y ∧ x.natAbs > n ∧ y.natAbs > n ∧ x * eval P x = y * eval P y

/-- The main theorem: if a cubic polynomial has infinite equal pairs, then it has an integer root -/
theorem cubic_with_infinite_equal_pairs_has_integer_root (P : CubicPolynomial) 
  (h : has_infinite_equal_pairs P) : ∃ k : ℤ, eval P k = 0 := by
  sorry

end NUMINAMATH_CALUDE_cubic_with_infinite_equal_pairs_has_integer_root_l971_97141


namespace NUMINAMATH_CALUDE_inequality_solution_set_l971_97179

theorem inequality_solution_set :
  {x : ℝ | (x + 1) / (3 - x) < 0} = {x : ℝ | x < -1 ∨ x > 3} := by sorry

end NUMINAMATH_CALUDE_inequality_solution_set_l971_97179


namespace NUMINAMATH_CALUDE_problem1_l971_97145

theorem problem1 (a b : ℝ) (ha : a ≠ 0) :
  (a - b^2 / a) / ((a^2 + 2*a*b + b^2) / a) = (a - b) / (a + b) := by
  sorry

end NUMINAMATH_CALUDE_problem1_l971_97145


namespace NUMINAMATH_CALUDE_bob_winning_strategy_l971_97152

/-- Represents the state of the game with the number of beads -/
structure GameState where
  beads : Nat
  deriving Repr

/-- Represents a player in the game -/
inductive Player
  | Alice
  | Bob
  deriving Repr

/-- Defines a valid move in the game -/
def validMove (s : GameState) : Prop :=
  s.beads > 1

/-- Defines the next player's turn -/
def nextPlayer : Player → Player
  | Player.Alice => Player.Bob
  | Player.Bob => Player.Alice

/-- Theorem stating that Bob has a winning strategy -/
theorem bob_winning_strategy :
  ∀ (initialBeads : Nat),
    initialBeads % 2 = 1 →
    ∃ (strategy : Player → GameState → Nat),
      ∀ (game : GameState),
        game.beads = initialBeads →
        ¬(∃ (aliceStrategy : Player → GameState → Nat),
          ∀ (state : GameState),
            validMove state →
            (state.beads % 2 = 1 → 
              validMove {beads := state.beads - strategy Player.Bob state} ∧
              validMove {beads := strategy Player.Bob state}) ∧
            (state.beads % 2 = 0 →
              validMove {beads := state.beads - aliceStrategy Player.Alice state} ∧
              validMove {beads := aliceStrategy Player.Alice state})) :=
sorry

#check bob_winning_strategy

end NUMINAMATH_CALUDE_bob_winning_strategy_l971_97152


namespace NUMINAMATH_CALUDE_quadratic_equation_roots_range_l971_97108

theorem quadratic_equation_roots_range (m : ℝ) :
  (∃ x y : ℝ, x ≠ y ∧ x^2 + m*x + 1 = 0 ∧ y^2 + m*y + 1 = 0) ↔ 
  m < -2 ∨ m > 2 :=
sorry

end NUMINAMATH_CALUDE_quadratic_equation_roots_range_l971_97108


namespace NUMINAMATH_CALUDE_chess_tournament_matches_prime_l971_97165

/-- Represents a single-elimination chess tournament --/
structure ChessTournament where
  totalPlayers : ℕ
  byePlayers : ℕ
  initialPlayers : ℕ

/-- Calculates the number of matches in a single-elimination tournament --/
def matchesPlayed (t : ChessTournament) : ℕ := t.totalPlayers - 1

/-- Theorem: In the given chess tournament, 119 matches are played and this number is prime --/
theorem chess_tournament_matches_prime (t : ChessTournament) 
  (h1 : t.totalPlayers = 120) 
  (h2 : t.byePlayers = 40) 
  (h3 : t.initialPlayers = 80) : 
  matchesPlayed t = 119 ∧ Nat.Prime 119 := by
  sorry

#eval Nat.Prime 119  -- To verify that 119 is indeed prime

end NUMINAMATH_CALUDE_chess_tournament_matches_prime_l971_97165


namespace NUMINAMATH_CALUDE_expand_product_l971_97118

theorem expand_product (x : ℝ) : (4 * x + 2) * (3 * x - 1) * (x + 6) = 12 * x^3 + 74 * x^2 + 10 * x - 12 := by
  sorry

end NUMINAMATH_CALUDE_expand_product_l971_97118


namespace NUMINAMATH_CALUDE_jordans_mangoes_l971_97130

theorem jordans_mangoes (total : ℕ) (ripe : ℕ) (unripe : ℕ) (kept : ℕ) (jars : ℕ) (mangoes_per_jar : ℕ) : 
  ripe = total / 3 →
  unripe = 2 * total / 3 →
  kept = 16 →
  jars = 5 →
  mangoes_per_jar = 4 →
  unripe = kept + jars * mangoes_per_jar →
  total = ripe + unripe →
  total = 54 :=
by sorry

end NUMINAMATH_CALUDE_jordans_mangoes_l971_97130


namespace NUMINAMATH_CALUDE_combinations_theorem_l971_97112

/-- The number of choices in the art group -/
def art_choices : ℕ := 2

/-- The number of choices in the sports group -/
def sports_choices : ℕ := 3

/-- The number of choices in the music group -/
def music_choices : ℕ := 4

/-- The total number of possible combinations -/
def total_combinations : ℕ := art_choices * sports_choices * music_choices

theorem combinations_theorem : total_combinations = 24 := by
  sorry

end NUMINAMATH_CALUDE_combinations_theorem_l971_97112


namespace NUMINAMATH_CALUDE_arithmetic_sequence_sum_l971_97125

def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_sum (a : ℕ → ℝ) :
  arithmetic_sequence a →
  a 2 + a 5 + a 8 + a 11 = 48 →
  a 6 + a 7 = 24 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_sum_l971_97125


namespace NUMINAMATH_CALUDE_parabola_c_value_l971_97109

/-- Given a parabola y = x^2 + bx + c passing through (2,3) and (4,3), prove c = 11 -/
theorem parabola_c_value (b c : ℝ) 
  (eq1 : 3 = 2^2 + 2*b + c) 
  (eq2 : 3 = 4^2 + 4*b + c) : 
  c = 11 := by sorry

end NUMINAMATH_CALUDE_parabola_c_value_l971_97109


namespace NUMINAMATH_CALUDE_smallest_divisor_of_7614_l971_97186

def n : ℕ := 7614

theorem smallest_divisor_of_7614 :
  ∃ (d : ℕ), d > 1 ∧ d ∣ n ∧ ∀ (k : ℕ), 1 < k ∧ k ∣ n → d ≤ k :=
by sorry

end NUMINAMATH_CALUDE_smallest_divisor_of_7614_l971_97186


namespace NUMINAMATH_CALUDE_second_quadrant_angle_sum_l971_97136

theorem second_quadrant_angle_sum (θ : Real) : 
  (π / 2 < θ ∧ θ < π) →  -- θ is in the second quadrant
  (Real.tan (θ + π / 4) = 1 / 2) →  -- tan(θ + π/4) = 1/2
  Real.sin θ + Real.cos θ = -Real.sqrt 10 / 5 := by
  sorry

end NUMINAMATH_CALUDE_second_quadrant_angle_sum_l971_97136


namespace NUMINAMATH_CALUDE_solution_existence_l971_97131

theorem solution_existence (m : ℝ) : 
  (∃ x : ℝ, 3 * Real.sin x + 4 * Real.cos x = 2 * m - 1) ↔ 
  -2 ≤ m ∧ m ≤ 3 :=
by sorry

end NUMINAMATH_CALUDE_solution_existence_l971_97131


namespace NUMINAMATH_CALUDE_circle_distance_inequality_l971_97122

theorem circle_distance_inequality (r s AB : ℝ) (h1 : r > s) (h2 : AB > 0) : ¬(r - s > AB) := by
  sorry

end NUMINAMATH_CALUDE_circle_distance_inequality_l971_97122


namespace NUMINAMATH_CALUDE_polynomial_evaluation_l971_97171

theorem polynomial_evaluation :
  ∀ x : ℝ, x > 0 → x^2 - 3*x - 9 = 0 → x^3 - 3*x^2 - 9*x + 27 = 27 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_evaluation_l971_97171


namespace NUMINAMATH_CALUDE_smallest_number_divisible_plus_one_l971_97190

theorem smallest_number_divisible_plus_one (n : ℕ) : n = 1038239 ↔ 
  (∀ m : ℕ, m < n → ¬((m + 1) % 618 = 0 ∧ (m + 1) % 3648 = 0 ∧ (m + 1) % 60 = 0)) ∧
  ((n + 1) % 618 = 0 ∧ (n + 1) % 3648 = 0 ∧ (n + 1) % 60 = 0) :=
by sorry

end NUMINAMATH_CALUDE_smallest_number_divisible_plus_one_l971_97190


namespace NUMINAMATH_CALUDE_rectangle_triangle_area_equality_l971_97128

/-- Given a square ABCD with side length 2 and a rectangle EFGH within it,
    prove that if AE = x, EB = 1, EF = x, and the areas of EFGH and ABE are equal,
    then x = 3/2 -/
theorem rectangle_triangle_area_equality (x : ℝ) : 
  (∀ (A B C D E F G H : ℝ × ℝ),
    -- ABCD is a square with side length 2
    ‖B - A‖ = 2 ∧ ‖C - B‖ = 2 ∧ ‖D - C‖ = 2 ∧ ‖A - D‖ = 2 ∧
    -- EFGH is a rectangle within the square
    (E.1 ≥ A.1 ∧ E.1 ≤ B.1) ∧ (E.2 ≥ A.2 ∧ E.2 ≤ D.2) ∧
    (F.1 ≥ A.1 ∧ F.1 ≤ B.1) ∧ (F.2 ≥ A.2 ∧ F.2 ≤ D.2) ∧
    (G.1 ≥ A.1 ∧ G.1 ≤ B.1) ∧ (G.2 ≥ A.2 ∧ G.2 ≤ D.2) ∧
    (H.1 ≥ A.1 ∧ H.1 ≤ B.1) ∧ (H.2 ≥ A.2 ∧ H.2 ≤ D.2) ∧
    -- AE = x, EB = 1, EF = x
    ‖E - A‖ = x ∧ ‖B - E‖ = 1 ∧ ‖F - E‖ = x ∧
    -- Areas of rectangle EFGH and triangle ABE are equal
    ‖F - E‖ * ‖G - F‖ = (1/2) * ‖E - A‖ * ‖B - E‖) →
  x = 3/2 := by
  sorry


end NUMINAMATH_CALUDE_rectangle_triangle_area_equality_l971_97128


namespace NUMINAMATH_CALUDE_class1_participants_l971_97110

/-- The number of students in Class 1 -/
def class1_students : ℕ := 40

/-- The number of students in Class 2 -/
def class2_students : ℕ := 36

/-- The number of students in Class 3 -/
def class3_students : ℕ := 44

/-- The total number of students who did not participate in the competition -/
def non_participants : ℕ := 30

/-- The proportion of students participating in the competition -/
def participation_rate : ℚ := 3/4

theorem class1_participants :
  (class1_students : ℚ) * participation_rate = 30 :=
sorry

end NUMINAMATH_CALUDE_class1_participants_l971_97110


namespace NUMINAMATH_CALUDE_integral_proof_l971_97111

open Real

theorem integral_proof (x : ℝ) (h : x ≠ -2 ∧ x ≠ 8) : 
  deriv (fun x => (1/10) * log (abs ((x - 8) / (x + 2)))) x = 1 / (x^2 - 6*x - 16) :=
sorry

end NUMINAMATH_CALUDE_integral_proof_l971_97111


namespace NUMINAMATH_CALUDE_joan_apples_l971_97124

/-- The number of apples Joan has after picking, receiving, and having some taken away. -/
def final_apples (initial : ℕ) (added : ℕ) (taken : ℕ) : ℕ :=
  initial + added - taken

/-- Theorem stating that Joan's final number of apples is 55 given the problem conditions. -/
theorem joan_apples : final_apples 43 27 15 = 55 := by
  sorry

end NUMINAMATH_CALUDE_joan_apples_l971_97124


namespace NUMINAMATH_CALUDE_plan_a_is_lowest_l971_97137

/-- Represents a payment plan with monthly payment, duration, and interest rate -/
structure PaymentPlan where
  monthly_payment : ℝ
  duration : ℕ
  interest_rate : ℝ

/-- Calculates the total repayment amount for a given payment plan -/
def total_repayment (plan : PaymentPlan) : ℝ :=
  let principal := plan.monthly_payment * plan.duration
  principal + principal * plan.interest_rate

/-- The three payment plans available to Aaron -/
def plan_a : PaymentPlan := ⟨100, 12, 0.1⟩
def plan_b : PaymentPlan := ⟨90, 15, 0.08⟩
def plan_c : PaymentPlan := ⟨80, 18, 0.06⟩

/-- Theorem stating that Plan A has the lowest total repayment amount -/
theorem plan_a_is_lowest :
  total_repayment plan_a < total_repayment plan_b ∧
  total_repayment plan_a < total_repayment plan_c :=
by sorry

end NUMINAMATH_CALUDE_plan_a_is_lowest_l971_97137


namespace NUMINAMATH_CALUDE_complex_modulus_problem_l971_97142

theorem complex_modulus_problem (z : ℂ) (i : ℂ) : 
  i * i = -1 → z = (3 - i) / (1 + i) → Complex.abs (z + i) = Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_complex_modulus_problem_l971_97142


namespace NUMINAMATH_CALUDE_douglas_vote_percentage_l971_97144

theorem douglas_vote_percentage (total_voters : ℕ) (x_voters y_voters : ℕ) 
  (douglas_total_votes douglas_x_votes douglas_y_votes : ℕ) :
  x_voters = 2 * y_voters →
  douglas_total_votes = (66 * (x_voters + y_voters)) / 100 →
  douglas_x_votes = (74 * x_voters) / 100 →
  douglas_total_votes = douglas_x_votes + douglas_y_votes →
  (douglas_y_votes * 100) / y_voters = 50 :=
by sorry

end NUMINAMATH_CALUDE_douglas_vote_percentage_l971_97144


namespace NUMINAMATH_CALUDE_delta_value_l971_97164

theorem delta_value : ∃ Δ : ℤ, (5 * (-3) = Δ - 3) → (Δ = -12) := by
  sorry

end NUMINAMATH_CALUDE_delta_value_l971_97164


namespace NUMINAMATH_CALUDE_solution_set_inequality_l971_97100

/-- Given that the solution set of x^2 - ax + b < 0 is (1,2), 
    prove that the solution set of 1/x < b/a is (-∞, 0) ∪ (3/2, +∞) -/
theorem solution_set_inequality (a b : ℝ) : 
  (∀ x, x^2 - a*x + b < 0 ↔ 1 < x ∧ x < 2) →
  (∀ x, 1/x < b/a ↔ x < 0 ∨ 3/2 < x) := by
sorry

end NUMINAMATH_CALUDE_solution_set_inequality_l971_97100


namespace NUMINAMATH_CALUDE_coeff_x_cubed_in_expansion_coeff_x_cubed_proof_l971_97155

/-- The coefficient of x^3 in the expansion of (1-x)^10 is -120 -/
theorem coeff_x_cubed_in_expansion : Int :=
  -120

/-- The binomial coefficient (10 choose 3) -/
def binomial_coeff : Int :=
  120

theorem coeff_x_cubed_proof : coeff_x_cubed_in_expansion = -binomial_coeff := by
  sorry

end NUMINAMATH_CALUDE_coeff_x_cubed_in_expansion_coeff_x_cubed_proof_l971_97155


namespace NUMINAMATH_CALUDE_greatest_divisor_with_remainder_l971_97153

def a : ℕ := 3547
def b : ℕ := 12739
def c : ℕ := 21329
def r : ℕ := 17

theorem greatest_divisor_with_remainder (d : ℕ) : d > 0 → d.gcd (a - r) = d → d.gcd (b - r) = d → d.gcd (c - r) = d → 
  (∀ k : ℕ, k > d → (k.gcd (a - r) ≠ k ∨ k.gcd (b - r) ≠ k ∨ k.gcd (c - r) ≠ k)) → 
  (∀ n : ℕ, n > 0 → (a % n = r ∧ b % n = r ∧ c % n = r) → n ≤ d) :=
by sorry

end NUMINAMATH_CALUDE_greatest_divisor_with_remainder_l971_97153


namespace NUMINAMATH_CALUDE_wardrobe_cost_is_180_l971_97182

/-- Calculates the total cost of Marcia's wardrobe given the following conditions:
  - 3 skirts at $20.00 each
  - 5 blouses at $15.00 each
  - 2 pairs of pants at $30.00 each, with a sale: buy 1 pair, get 1 pair 1/2 off
-/
def wardrobeCost (skirtPrice blousePrice pantPrice : ℚ) : ℚ :=
  let skirtCost := 3 * skirtPrice
  let blouseCost := 5 * blousePrice
  let pantCost := pantPrice + (pantPrice / 2)
  skirtCost + blouseCost + pantCost

/-- Proves that the total cost of Marcia's wardrobe is $180.00 -/
theorem wardrobe_cost_is_180 :
  wardrobeCost 20 15 30 = 180 := by
  sorry

#eval wardrobeCost 20 15 30

end NUMINAMATH_CALUDE_wardrobe_cost_is_180_l971_97182


namespace NUMINAMATH_CALUDE_total_rackets_packed_l971_97149

/-- Proves that given the conditions of racket packaging, the total number of rackets is 100 -/
theorem total_rackets_packed (total_cartons : ℕ) (three_racket_cartons : ℕ) 
  (h1 : total_cartons = 38)
  (h2 : three_racket_cartons = 24) :
  3 * three_racket_cartons + 2 * (total_cartons - three_racket_cartons) = 100 := by
  sorry

end NUMINAMATH_CALUDE_total_rackets_packed_l971_97149


namespace NUMINAMATH_CALUDE_closed_set_properties_l971_97138

-- Define what it means for a set to be closed
def is_closed (M : Set ℤ) : Prop :=
  ∀ a b, a ∈ M → b ∈ M → (a + b) ∈ M ∧ (a - b) ∈ M

-- Define the set M = {-2, -1, 0, 1, 2}
def M : Set ℤ := {-2, -1, 0, 1, 2}

-- Define the set of positive integers
def positive_integers : Set ℤ := {n : ℤ | n > 0}

-- Define the set M = {n | n = 3k, k ∈ Z}
def M_3k : Set ℤ := {n : ℤ | ∃ k : ℤ, n = 3 * k}

theorem closed_set_properties :
  (¬ is_closed M) ∧
  (¬ is_closed positive_integers) ∧
  (is_closed M_3k) ∧
  (∃ A₁ A₂ : Set ℤ, is_closed A₁ ∧ is_closed A₂ ∧ ¬ is_closed (A₁ ∪ A₂)) := by
  sorry

end NUMINAMATH_CALUDE_closed_set_properties_l971_97138


namespace NUMINAMATH_CALUDE_chemical_reaction_result_l971_97123

-- Define the initial amounts
def initial_silver_nitrate : ℝ := 2
def initial_sodium_hydroxide : ℝ := 2
def initial_hydrochloric_acid : ℝ := 0.5

-- Define the reactions
def main_reaction (x : ℝ) : ℝ := x
def side_reaction (x : ℝ) : ℝ := x

-- Theorem statement
theorem chemical_reaction_result :
  let sodium_hydroxide_in_side_reaction := min initial_sodium_hydroxide initial_hydrochloric_acid
  let remaining_sodium_hydroxide := initial_sodium_hydroxide - sodium_hydroxide_in_side_reaction
  let reaction_limit := min remaining_sodium_hydroxide initial_silver_nitrate
  let sodium_nitrate_formed := main_reaction reaction_limit
  let silver_chloride_formed := main_reaction reaction_limit
  let unreacted_sodium_hydroxide := remaining_sodium_hydroxide - reaction_limit
  sodium_nitrate_formed = 1.5 ∧ 
  silver_chloride_formed = 1.5 ∧ 
  unreacted_sodium_hydroxide = 0 :=
by
  sorry


end NUMINAMATH_CALUDE_chemical_reaction_result_l971_97123


namespace NUMINAMATH_CALUDE_jacket_price_calculation_l971_97195

def calculate_final_price (initial_price : ℝ) (discount1 : ℝ) (discount2 : ℝ) (coupon : ℝ) (tax : ℝ) : ℝ :=
  let price_after_discount1 := initial_price * (1 - discount1)
  let price_after_discount2 := price_after_discount1 * (1 - discount2)
  let price_after_coupon := price_after_discount2 - coupon
  price_after_coupon * (1 + tax)

theorem jacket_price_calculation :
  calculate_final_price 150 0.30 0.10 10 0.05 = 88.725 := by
  sorry

end NUMINAMATH_CALUDE_jacket_price_calculation_l971_97195


namespace NUMINAMATH_CALUDE_square_side_length_l971_97166

/-- Given two identical overlapping squares where the upper square is moved 3 cm right and 5 cm down,
    resulting in a shaded area of 57 square centimeters, prove that the side length of each square is 9 cm. -/
theorem square_side_length (a : ℝ) 
  (h1 : 3 * a + 5 * (a - 3) = 57) : 
  a = 9 := by
  sorry

end NUMINAMATH_CALUDE_square_side_length_l971_97166


namespace NUMINAMATH_CALUDE_candies_per_packet_candies_per_packet_proof_l971_97140

/-- The number of candies in a packet given Bobby's eating habits and the time it takes to finish the packets. -/
theorem candies_per_packet : ℕ :=
  let packets : ℕ := 2
  let weekdays : ℕ := 5
  let weekend_days : ℕ := 2
  let candies_per_weekday : ℕ := 2
  let candies_per_weekend_day : ℕ := 1
  let weeks_to_finish : ℕ := 3
  18

/-- Proof that the number of candies in a packet is 18. -/
theorem candies_per_packet_proof :
  let packets : ℕ := 2
  let weekdays : ℕ := 5
  let weekend_days : ℕ := 2
  let candies_per_weekday : ℕ := 2
  let candies_per_weekend_day : ℕ := 1
  let weeks_to_finish : ℕ := 3
  candies_per_packet = 18 := by
  sorry

end NUMINAMATH_CALUDE_candies_per_packet_candies_per_packet_proof_l971_97140


namespace NUMINAMATH_CALUDE_monotonic_quadratic_function_l971_97135

/-- A function f is monotonic on an interval [a, b] if it is either
    non-decreasing or non-increasing on that interval. -/
def IsMonotonic (f : ℝ → ℝ) (a b : ℝ) : Prop :=
  (∀ x y, a ≤ x ∧ x ≤ y ∧ y ≤ b → f x ≤ f y) ∨
  (∀ x y, a ≤ x ∧ x ≤ y ∧ y ≤ b → f y ≤ f x)

/-- The theorem states that for the function f(x) = x^2 - kx + 1,
    f is monotonic on the interval [1, 2] if and only if
    k is in the set (-∞, 2] ∪ [4, +∞). -/
theorem monotonic_quadratic_function (k : ℝ) :
  IsMonotonic (fun x => x^2 - k*x + 1) 1 2 ↔ k ≤ 2 ∨ k ≥ 4 :=
sorry

end NUMINAMATH_CALUDE_monotonic_quadratic_function_l971_97135


namespace NUMINAMATH_CALUDE_quadratic_monotone_increasing_l971_97197

/-- A quadratic function f(x) = x^2 + 2ax + b is monotonically increasing
    on the interval [-1, +∞) if and only if a ≥ 1 -/
theorem quadratic_monotone_increasing (a b : ℝ) :
  (∀ x₁ x₂ : ℝ, -1 ≤ x₁ ∧ x₁ < x₂ → x₁^2 + 2*a*x₁ + b < x₂^2 + 2*a*x₂ + b) ↔ a ≥ 1 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_monotone_increasing_l971_97197


namespace NUMINAMATH_CALUDE_davids_homework_time_l971_97156

theorem davids_homework_time (math_time spelling_time reading_time : ℕ) 
  (h1 : math_time = 15)
  (h2 : spelling_time = 18)
  (h3 : reading_time = 27) :
  math_time + spelling_time + reading_time = 60 := by
  sorry

end NUMINAMATH_CALUDE_davids_homework_time_l971_97156


namespace NUMINAMATH_CALUDE_parabola_chord_length_l971_97114

/-- The length of a chord AB of a parabola y^2 = 8x intersected by a line y = kx - 2 -/
theorem parabola_chord_length (k : ℝ) : 
  (∃ A B : ℝ × ℝ, 
    (A.2^2 = 8 * A.1 ∧ A.2 = k * A.1 - 2) ∧ 
    (B.2^2 = 8 * B.1 ∧ B.2 = k * B.1 - 2) ∧
    A ≠ B ∧
    (A.1 + B.1) / 2 = 2) →
  ∃ A B : ℝ × ℝ, 
    (A.2^2 = 8 * A.1 ∧ A.2 = k * A.1 - 2) ∧ 
    (B.2^2 = 8 * B.1 ∧ B.2 = k * B.1 - 2) ∧
    A ≠ B ∧
    (A.1 + B.1) / 2 = 2 ∧
    Real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2) = 2 * Real.sqrt 15 :=
by sorry


end NUMINAMATH_CALUDE_parabola_chord_length_l971_97114


namespace NUMINAMATH_CALUDE_flash_fraction_is_one_l971_97192

/-- The fraction of an hour it takes for a light to flash 120 times, given that it flashes every 30 seconds -/
def flash_fraction : ℚ :=
  let flash_interval : ℚ := 30 / 3600  -- 30 seconds as a fraction of an hour
  let total_flashes : ℕ := 120
  total_flashes * flash_interval

theorem flash_fraction_is_one : flash_fraction = 1 := by
  sorry

end NUMINAMATH_CALUDE_flash_fraction_is_one_l971_97192


namespace NUMINAMATH_CALUDE_polynomial_equality_l971_97177

theorem polynomial_equality (a b : ℝ) : 
  (∀ x : ℝ, (x + a) * (x - 3) = x^2 + 2*x - b) → a - b = -10 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_equality_l971_97177


namespace NUMINAMATH_CALUDE_largest_package_size_l971_97157

theorem largest_package_size (juan_markers alicia_markers : ℕ) 
  (h1 : juan_markers = 36) (h2 : alicia_markers = 48) : 
  Nat.gcd juan_markers alicia_markers = 12 := by
  sorry

end NUMINAMATH_CALUDE_largest_package_size_l971_97157


namespace NUMINAMATH_CALUDE_arithmetic_sequence_10th_term_l971_97102

/-- An arithmetic sequence with 30 terms, first term 3, and last term 89 has its 10th term equal to 30 -/
theorem arithmetic_sequence_10th_term :
  ∀ (a : ℕ → ℚ), 
    (∀ i j : ℕ, i < j → a j - a i = (j - i) * (a 1 - a 0)) →  -- arithmetic sequence
    (a 0 = 3) →                                               -- first term is 3
    (a 29 = 89) →                                             -- last term is 89
    (a 9 = 30) :=                                             -- 10th term (index 9) is 30
by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_10th_term_l971_97102


namespace NUMINAMATH_CALUDE_corporation_employees_l971_97176

theorem corporation_employees (total : ℕ) (part_time : ℕ) (full_time : ℕ) :
  total = 65134 →
  part_time = 2041 →
  full_time = total - part_time →
  full_time = 63093 := by
sorry

end NUMINAMATH_CALUDE_corporation_employees_l971_97176


namespace NUMINAMATH_CALUDE_cube_sum_implies_sum_bound_l971_97151

theorem cube_sum_implies_sum_bound (a b : ℝ) :
  a > 0 → b > 0 → a^3 + b^3 = 2 → a + b ≤ 2 := by
  sorry

end NUMINAMATH_CALUDE_cube_sum_implies_sum_bound_l971_97151


namespace NUMINAMATH_CALUDE_ellipse_eccentricity_l971_97158

/-- Represents an ellipse with semi-major axis 'a' and semi-minor axis 2 -/
structure Ellipse where
  a : ℝ
  h_a : a > 2

/-- Represents a line with equation y = x - 2 -/
def Line : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.2 = p.1 - 2}

/-- The eccentricity of an ellipse -/
def eccentricity (e : Ellipse) : ℝ :=
  sorry

/-- A focus of the ellipse -/
def focus (e : Ellipse) : ℝ × ℝ :=
  sorry

theorem ellipse_eccentricity (e : Ellipse) :
  focus e ∈ Line →
  eccentricity e = Real.sqrt 2 / 2 :=
sorry

end NUMINAMATH_CALUDE_ellipse_eccentricity_l971_97158


namespace NUMINAMATH_CALUDE_intersection_difference_l971_97127

-- Define the two parabolas
def parabola1 (x : ℝ) : ℝ := 3 * x^2 - 2 * x + 1
def parabola2 (x : ℝ) : ℝ := -2 * x^2 + 4 * x + 1

-- Define the theorem
theorem intersection_difference (a b c d : ℝ) 
  (h1 : parabola1 a = parabola2 a) 
  (h2 : parabola1 c = parabola2 c) 
  (h3 : c ≥ a) : 
  c - a = 6/5 := by
  sorry

end NUMINAMATH_CALUDE_intersection_difference_l971_97127


namespace NUMINAMATH_CALUDE_lcm_of_20_45_75_l971_97170

theorem lcm_of_20_45_75 : Nat.lcm 20 (Nat.lcm 45 75) = 900 := by
  sorry

end NUMINAMATH_CALUDE_lcm_of_20_45_75_l971_97170


namespace NUMINAMATH_CALUDE_number_of_women_in_first_group_l971_97188

/-- The number of women in the first group -/
def W : ℕ := sorry

/-- The work rate of the first group in units per hour -/
def work_rate_1 : ℚ := 75 / (8 * 5)

/-- The work rate of the second group in units per hour -/
def work_rate_2 : ℚ := 30 / (3 * 8)

/-- Theorem stating that the number of women in the first group is 5 -/
theorem number_of_women_in_first_group : W = 5 := by sorry

end NUMINAMATH_CALUDE_number_of_women_in_first_group_l971_97188


namespace NUMINAMATH_CALUDE_distance_AB_on_parabola_l971_97178

-- Define the parabola
def parabola (x y : ℝ) : Prop := y^2 = 4*x

-- Define the focus of the parabola
def focus : ℝ × ℝ := (1, 0)

-- Define point B
def point_B : ℝ × ℝ := (3, 0)

-- Theorem statement
theorem distance_AB_on_parabola (A : ℝ × ℝ) :
  parabola A.1 A.2 →  -- A lies on the parabola
  ‖A - focus‖ = ‖point_B - focus‖ →  -- |AF| = |BF|
  ‖A - point_B‖ = 2 * Real.sqrt 2 :=  -- |AB| = 2√2
by sorry

end NUMINAMATH_CALUDE_distance_AB_on_parabola_l971_97178


namespace NUMINAMATH_CALUDE_johns_remaining_money_l971_97154

def remaining_money (initial amount_spent_on_sweets amount_given_to_each_friend : ℚ) : ℚ :=
  initial - (amount_spent_on_sweets + 2 * amount_given_to_each_friend)

theorem johns_remaining_money :
  remaining_money 10.50 2.25 2.20 = 3.85 := by
  sorry

end NUMINAMATH_CALUDE_johns_remaining_money_l971_97154


namespace NUMINAMATH_CALUDE_unique_solution_power_equation_l971_97181

theorem unique_solution_power_equation :
  ∀ x y : ℕ, x ≥ 1 → y ≥ 1 → (2^x : ℕ) + 3 = 11^y → x = 3 ∧ y = 1 := by
  sorry

end NUMINAMATH_CALUDE_unique_solution_power_equation_l971_97181


namespace NUMINAMATH_CALUDE_water_left_in_bathtub_water_left_is_7800_l971_97185

/-- Calculates the amount of water left in a bathtub given specific conditions. -/
theorem water_left_in_bathtub 
  (faucet_drip_rate : ℝ) 
  (evaporation_rate : ℝ) 
  (time_running : ℝ) 
  (water_dumped : ℝ) : ℝ :=
  let water_added_per_hour := faucet_drip_rate * 60 - evaporation_rate
  let total_water_added := water_added_per_hour * time_running
  let water_remaining := total_water_added - water_dumped * 1000
  water_remaining

/-- Proves that under the given conditions, 7800 ml of water are left in the bathtub. -/
theorem water_left_is_7800 :
  water_left_in_bathtub 40 200 9 12 = 7800 := by
  sorry

end NUMINAMATH_CALUDE_water_left_in_bathtub_water_left_is_7800_l971_97185


namespace NUMINAMATH_CALUDE_factoring_left_to_right_l971_97194

theorem factoring_left_to_right (m n : ℝ) : m^2 - 2*m*n + n^2 = (m - n)^2 := by
  sorry

end NUMINAMATH_CALUDE_factoring_left_to_right_l971_97194


namespace NUMINAMATH_CALUDE_x_value_proof_l971_97161

theorem x_value_proof (x : ℚ) 
  (eq1 : 8 * x^2 + 8 * x - 2 = 0) 
  (eq2 : 32 * x^2 + 68 * x - 8 = 0) : 
  x = 1/8 := by
sorry

end NUMINAMATH_CALUDE_x_value_proof_l971_97161


namespace NUMINAMATH_CALUDE_total_pencils_l971_97146

theorem total_pencils (num_people : ℕ) (pencils_per_person : ℕ) : 
  num_people = 5 → pencils_per_person = 15 → num_people * pencils_per_person = 75 := by
  sorry

end NUMINAMATH_CALUDE_total_pencils_l971_97146


namespace NUMINAMATH_CALUDE_A_specific_value_l971_97193

def A : ℕ → ℕ
  | 0 => 1
  | n + 1 => A (n / 2023) + A (n / 2023^2) + A (n / 2023^3)

theorem A_specific_value : A (2023^(3^2) + 20) = 653 := by
  sorry

end NUMINAMATH_CALUDE_A_specific_value_l971_97193


namespace NUMINAMATH_CALUDE_equation_solutions_l971_97126

theorem equation_solutions : 
  (∃ S₁ : Set ℝ, S₁ = {x : ℝ | x * (x - 2) + x - 2 = 0} ∧ S₁ = {2, -1}) ∧
  (∃ S₂ : Set ℝ, S₂ = {x : ℝ | 2 * x^2 + 5 * x + 3 = 0} ∧ S₂ = {-1, -3/2}) :=
by
  sorry


end NUMINAMATH_CALUDE_equation_solutions_l971_97126


namespace NUMINAMATH_CALUDE_common_ratio_of_geometric_sequence_l971_97132

-- Define the geometric sequence
def geometric_sequence (a : ℕ → ℝ) (q : ℝ) :=
  ∀ n : ℕ, a (n + 1) = a n * q

-- Define the arithmetic sequence property
def arithmetic_sequence_property (a : ℕ → ℝ) (q : ℝ) :=
  (3 / 2) * (a 1 * q) = (2 * a 0 + a 0 * q^2) / 2

-- Theorem statement
theorem common_ratio_of_geometric_sequence
  (a : ℕ → ℝ) (q : ℝ)
  (h1 : geometric_sequence a q)
  (h2 : arithmetic_sequence_property a q) :
  q = 1 ∨ q = 2 := by
  sorry

end NUMINAMATH_CALUDE_common_ratio_of_geometric_sequence_l971_97132


namespace NUMINAMATH_CALUDE_geometric_sequence_product_l971_97150

def geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = a n * r

theorem geometric_sequence_product (a : ℕ → ℝ) :
  geometric_sequence a →
  a 1 = 1 →
  a 10 = 3 →
  a 2 * a 3 * a 4 * a 5 * a 6 * a 7 * a 8 * a 9 = 81 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_product_l971_97150


namespace NUMINAMATH_CALUDE_new_student_weight_l971_97169

/-- Given a group of students and their weights, calculates the weight of a new student
    that changes the average weight of the group. -/
theorem new_student_weight
  (n : ℕ) -- number of students before new admission
  (w : ℝ) -- average weight before new admission
  (new_w : ℝ) -- new average weight after admission
  (h1 : n = 29) -- there are 29 students initially
  (h2 : w = 28) -- the initial average weight is 28 kg
  (h3 : new_w = 27.4) -- the new average weight is 27.4 kg
  : (n + 1) * new_w - n * w = 10 := by
  sorry

end NUMINAMATH_CALUDE_new_student_weight_l971_97169


namespace NUMINAMATH_CALUDE_c_prime_coordinates_l971_97180

/-- Triangle ABC with vertices A(1,2), B(2,1), and C(3,2) -/
structure Triangle :=
  (A : ℝ × ℝ)
  (B : ℝ × ℝ)
  (C : ℝ × ℝ)

/-- Similar triangle A'B'C' with similarity ratio 2 and origin as center of similarity -/
def similarTriangle (t : Triangle) : Triangle :=
  { A := (2 * t.A.1, 2 * t.A.2),
    B := (2 * t.B.1, 2 * t.B.2),
    C := (2 * t.C.1, 2 * t.C.2) }

/-- The original triangle ABC -/
def ABC : Triangle :=
  { A := (1, 2),
    B := (2, 1),
    C := (3, 2) }

/-- Theorem stating that C' has coordinates (6,4) or (-6,-4) -/
theorem c_prime_coordinates :
  let t' := similarTriangle ABC
  (t'.C = (6, 4) ∨ t'.C = (-6, -4)) :=
sorry

end NUMINAMATH_CALUDE_c_prime_coordinates_l971_97180


namespace NUMINAMATH_CALUDE_negation_equivalence_l971_97173

theorem negation_equivalence : 
  (¬ ∃ x : ℝ, x > 0 ∧ x^2 - 1 ≤ 0) ↔ (∀ x : ℝ, x > 0 → x^2 - 1 > 0) := by
  sorry

end NUMINAMATH_CALUDE_negation_equivalence_l971_97173


namespace NUMINAMATH_CALUDE_merry_go_round_revolutions_l971_97184

/-- The number of revolutions needed for two horses on a merry-go-round to travel the same distance -/
theorem merry_go_round_revolutions (r₁ r₂ n₁ : ℝ) (hr₁ : r₁ = 15) (hr₂ : r₂ = 5) (hn₁ : n₁ = 20) :
  ∃ n₂ : ℝ, n₂ = 60 ∧ n₁ * r₁ = n₂ * r₂ := by
  sorry


end NUMINAMATH_CALUDE_merry_go_round_revolutions_l971_97184


namespace NUMINAMATH_CALUDE_equation_roots_imply_m_range_l971_97113

theorem equation_roots_imply_m_range (m : ℝ) : 
  (∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ 
   4^x₁ - m * 2^(x₁+1) + 2 - m = 0 ∧
   4^x₂ - m * 2^(x₂+1) + 2 - m = 0) →
  1 < m ∧ m < 2 :=
by sorry

end NUMINAMATH_CALUDE_equation_roots_imply_m_range_l971_97113

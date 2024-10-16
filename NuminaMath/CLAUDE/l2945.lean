import Mathlib

namespace NUMINAMATH_CALUDE_intersection_A_B_l2945_294582

def A : Set ℝ := {x | -1 < x ∧ x ≤ 2}
def B : Set ℝ := {-1, 0, 1}

theorem intersection_A_B : A ∩ B = {0, 1} := by
  sorry

end NUMINAMATH_CALUDE_intersection_A_B_l2945_294582


namespace NUMINAMATH_CALUDE_third_to_second_ratio_l2945_294505

/-- Represents the number of questions solved in each hour -/
structure HourlyQuestions where
  first : ℕ
  second : ℕ
  third : ℕ

/-- Verifies if the given hourly questions satisfy the problem conditions -/
def satisfiesConditions (q : HourlyQuestions) : Prop :=
  q.third = 132 ∧
  q.third = 3 * q.first ∧
  q.first + q.second + q.third = 242

/-- Theorem stating that if the conditions are satisfied, the ratio of third to second hour questions is 2:1 -/
theorem third_to_second_ratio (q : HourlyQuestions) 
  (h : satisfiesConditions q) : q.third = 2 * q.second :=
by
  sorry

#check third_to_second_ratio

end NUMINAMATH_CALUDE_third_to_second_ratio_l2945_294505


namespace NUMINAMATH_CALUDE_duck_ratio_is_two_to_one_l2945_294533

/-- The ratio of ducks at North Pond to Lake Michigan -/
def duck_ratio (north_pond : ℕ) (lake_michigan : ℕ) : ℚ :=
  (north_pond : ℚ) / (lake_michigan : ℚ)

theorem duck_ratio_is_two_to_one :
  let lake_michigan : ℕ := 100
  let north_pond : ℕ := 206
  ∀ R : ℚ, north_pond = lake_michigan * R + 6 →
    duck_ratio north_pond lake_michigan = 2 := by
  sorry

end NUMINAMATH_CALUDE_duck_ratio_is_two_to_one_l2945_294533


namespace NUMINAMATH_CALUDE_juggling_show_balls_l2945_294571

/-- The number of balls needed for a juggling show -/
def balls_needed (jugglers : ℕ) (balls_per_juggler : ℕ) : ℕ :=
  jugglers * balls_per_juggler

/-- Theorem: 378 jugglers each juggling 6 balls require 2268 balls in total -/
theorem juggling_show_balls : balls_needed 378 6 = 2268 := by
  sorry

end NUMINAMATH_CALUDE_juggling_show_balls_l2945_294571


namespace NUMINAMATH_CALUDE_max_floor_length_l2945_294559

/-- Represents a rectangular tile with length and width in centimeters. -/
structure Tile where
  length : ℕ
  width : ℕ

/-- Represents a rectangular floor with length and width in centimeters. -/
structure Floor where
  length : ℕ
  width : ℕ

/-- Checks if a given number of tiles can fit on the floor without overlap or overshooting. -/
def canFitTiles (t : Tile) (f : Floor) (n : ℕ) : Prop :=
  (f.length % t.length = 0 ∧ f.width ≥ t.width ∧ (f.length / t.length) * (f.width / t.width) ≥ n) ∨
  (f.length % t.width = 0 ∧ f.width ≥ t.length ∧ (f.length / t.width) * (f.width / t.length) ≥ n)

theorem max_floor_length (t : Tile) (maxTiles : ℕ) :
  t.length = 50 →
  t.width = 40 →
  maxTiles = 9 →
  ∃ (f : Floor), canFitTiles t f maxTiles ∧
    ∀ (f' : Floor), canFitTiles t f' maxTiles → f'.length ≤ f.length ∧ f.length = 450 := by
  sorry

end NUMINAMATH_CALUDE_max_floor_length_l2945_294559


namespace NUMINAMATH_CALUDE_jogger_speed_l2945_294527

/-- The speed of the jogger in km/hr given the following conditions:
  1. The jogger is 200 m ahead of the train engine
  2. The train is 210 m long
  3. The train is running at 45 km/hr
  4. The train and jogger are moving in the same direction
  5. The train passes the jogger in 41 seconds
-/
theorem jogger_speed : ℝ := by
  -- Define the given conditions
  let initial_distance : ℝ := 200 -- meters
  let train_length : ℝ := 210 -- meters
  let train_speed : ℝ := 45 -- km/hr
  let passing_time : ℝ := 41 -- seconds

  -- Define the jogger's speed as a variable
  let jogger_speed : ℝ := 9 -- km/hr

  sorry -- Proof omitted

#check jogger_speed

end NUMINAMATH_CALUDE_jogger_speed_l2945_294527


namespace NUMINAMATH_CALUDE_min_distance_to_line_l2945_294558

theorem min_distance_to_line (x y : ℝ) :
  (8 * x + 15 * y = 120) →
  (∀ a b : ℝ, 8 * a + 15 * b = 120 → x^2 + y^2 ≤ a^2 + b^2) →
  Real.sqrt (x^2 + y^2) = 120 / 17 := by
sorry

end NUMINAMATH_CALUDE_min_distance_to_line_l2945_294558


namespace NUMINAMATH_CALUDE_symmetric_coloring_l2945_294536

/-- A number is red if it can be expressed as 81x + 100y for positive integers x and y -/
def IsRed (n : ℤ) : Prop :=
  ∃ (x y : ℕ), n = 81 * x + 100 * y ∧ x > 0 ∧ y > 0

/-- The theorem to be proved -/
theorem symmetric_coloring :
  ∃ (n : ℤ), (IsRed n ∧ ¬IsRed (8281 - n)) ∨ (¬IsRed n ∧ IsRed (8281 - n)) :=
sorry

end NUMINAMATH_CALUDE_symmetric_coloring_l2945_294536


namespace NUMINAMATH_CALUDE_equation_solution_l2945_294535

theorem equation_solution (a b c d : ℕ) : 
  0 < a ∧ a < 4 ∧ 
  0 < b ∧ b < 4 ∧ 
  0 < c ∧ c < 4 ∧ 
  0 < d ∧ d < 4 ∧ 
  4^a + 3^b + 2^c + 1^d = 78 → 
  b / c = 1 := by
sorry

end NUMINAMATH_CALUDE_equation_solution_l2945_294535


namespace NUMINAMATH_CALUDE_custom_op_result_l2945_294573

def custom_op (a b : ℤ) : ℤ := b^2 - a*b

theorem custom_op_result : custom_op (custom_op (-1) 2) 3 = -9 := by
  sorry

end NUMINAMATH_CALUDE_custom_op_result_l2945_294573


namespace NUMINAMATH_CALUDE_line_through_intersection_and_parallel_l2945_294534

-- Define the lines
def line1 (x y : ℝ) : Prop := 2*x + 3*y - 7 = 0
def line2 (x y : ℝ) : Prop := 7*x + 15*y + 1 = 0
def line3 (x y : ℝ) : Prop := x + 2*y - 3 = 0
def target_line (x y : ℝ) : Prop := 3*x + 6*y - 2 = 0

-- Define the intersection point
def intersection_point (x y : ℝ) : Prop := line1 x y ∧ line2 x y

-- Theorem statement
theorem line_through_intersection_and_parallel :
  ∃ (x y : ℝ), intersection_point x y ∧ target_line x y ∧
  ∃ (k : ℝ), k ≠ 0 ∧ ∀ (x y : ℝ), target_line x y ↔ line3 (k*x) (k*y) :=
sorry

end NUMINAMATH_CALUDE_line_through_intersection_and_parallel_l2945_294534


namespace NUMINAMATH_CALUDE_factor_theorem_cubic_l2945_294524

theorem factor_theorem_cubic (a : ℚ) :
  (∀ x, x^3 + 2*x^2 + a*x + 20 = 0 → x = 3) →
  a = -65/3 := by
sorry

end NUMINAMATH_CALUDE_factor_theorem_cubic_l2945_294524


namespace NUMINAMATH_CALUDE_bus_students_count_l2945_294539

theorem bus_students_count (initial_students : Real) (students_boarding : Real) : 
  initial_students = 10.0 → students_boarding = 3.0 → initial_students + students_boarding = 13.0 := by
  sorry

end NUMINAMATH_CALUDE_bus_students_count_l2945_294539


namespace NUMINAMATH_CALUDE_reverse_digit_integers_l2945_294538

theorem reverse_digit_integers (q r : ℕ) : 
  (10 ≤ q ∧ q < 100) →  -- q is a two-digit integer
  (10 ≤ r ∧ r < 100) →  -- r is a two-digit integer
  (∃ a b : ℕ, q = 10 * a + b ∧ r = 10 * b + a) →  -- q and r have reversed digits
  (q > r → q - r < 60) →  -- positive difference less than 60
  (r > q → r - q < 60) →  -- positive difference less than 60
  (∀ x y : ℕ, (10 ≤ x ∧ x < 100) → (10 ≤ y ∧ y < 100) → 
    (∃ c d : ℕ, x = 10 * c + d ∧ y = 10 * d + c) → 
    (x > y → x - y ≤ 54) ∧ (y > x → y - x ≤ 54)) →  -- greatest possible difference is 54
  (∃ a b : ℕ, q = 10 * a + b ∧ r = 10 * b + a ∧ a = b + 6) :=  -- conclusion: tens digit is 6 more than units digit
by sorry

end NUMINAMATH_CALUDE_reverse_digit_integers_l2945_294538


namespace NUMINAMATH_CALUDE_paving_cost_calculation_l2945_294531

-- Define the room dimensions and paving rate
def room_length : ℝ := 5.5
def room_width : ℝ := 4
def paving_rate : ℝ := 750

-- Define the function to calculate the cost of paving
def paving_cost (length width rate : ℝ) : ℝ :=
  length * width * rate

-- State the theorem
theorem paving_cost_calculation :
  paving_cost room_length room_width paving_rate = 16500 := by
  sorry

end NUMINAMATH_CALUDE_paving_cost_calculation_l2945_294531


namespace NUMINAMATH_CALUDE_least_integer_absolute_value_l2945_294578

theorem least_integer_absolute_value (x : ℤ) : 
  (∀ y : ℤ, |3*y - 4| ≤ 25 → y ≥ -7) ∧ |3*(-7) - 4| ≤ 25 := by
  sorry

end NUMINAMATH_CALUDE_least_integer_absolute_value_l2945_294578


namespace NUMINAMATH_CALUDE_orchard_fruit_count_l2945_294507

def apple_trees : ℕ := 50
def orange_trees : ℕ := 30
def apple_baskets_per_tree : ℕ := 25
def orange_baskets_per_tree : ℕ := 15
def apples_per_basket : ℕ := 18
def oranges_per_basket : ℕ := 12

theorem orchard_fruit_count :
  let total_apples := apple_trees * apple_baskets_per_tree * apples_per_basket
  let total_oranges := orange_trees * orange_baskets_per_tree * oranges_per_basket
  total_apples = 22500 ∧ total_oranges = 5400 := by
  sorry

end NUMINAMATH_CALUDE_orchard_fruit_count_l2945_294507


namespace NUMINAMATH_CALUDE_inequality_proof_l2945_294500

theorem inequality_proof (x : ℝ) : x > 0 ∧ |4*x - 5| < 8 → 0 < x ∧ x < 13/4 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l2945_294500


namespace NUMINAMATH_CALUDE_y_at_40_l2945_294504

/-- A line passing through three given points -/
structure Line where
  point1 : ℝ × ℝ
  point2 : ℝ × ℝ
  point3 : ℝ × ℝ

/-- The line passing through the given points -/
def exampleLine : Line :=
  { point1 := (2, 5)
  , point2 := (6, 17)
  , point3 := (10, 29) }

/-- Function to calculate y-coordinate for a given x-coordinate on the line -/
def yCoordinate (l : Line) (x : ℝ) : ℝ :=
  sorry

theorem y_at_40 (l : Line) : l = exampleLine → yCoordinate l 40 = 119 := by
  sorry

end NUMINAMATH_CALUDE_y_at_40_l2945_294504


namespace NUMINAMATH_CALUDE_mrs_white_carrot_yield_l2945_294522

/-- Calculates the expected carrot yield from a rectangular garden --/
def expected_carrot_yield (length_steps : ℕ) (width_steps : ℕ) (step_size : ℚ) (yield_per_sqft : ℚ) : ℚ :=
  (length_steps : ℚ) * step_size * (width_steps : ℚ) * step_size * yield_per_sqft

/-- Proves that the expected carrot yield for Mrs. White's garden is 1875 pounds --/
theorem mrs_white_carrot_yield : 
  expected_carrot_yield 18 25 (5/2) (2/3) = 1875 := by
  sorry

end NUMINAMATH_CALUDE_mrs_white_carrot_yield_l2945_294522


namespace NUMINAMATH_CALUDE_fifteen_percent_of_600_is_90_l2945_294523

theorem fifteen_percent_of_600_is_90 : ∃ x : ℝ, (15 / 100) * x = 90 ∧ x = 600 := by
  sorry

end NUMINAMATH_CALUDE_fifteen_percent_of_600_is_90_l2945_294523


namespace NUMINAMATH_CALUDE_set_A_equals_circle_B_l2945_294540

-- Define the circle D
def circle_D (Q : ℝ × ℝ) : Set (ℝ × ℝ) :=
  {P | dist P Q = 10}

-- Define point B
def point_B (Q : ℝ × ℝ) : ℝ × ℝ :=
  let v : ℝ × ℝ := (6, 0)  -- Arbitrary direction, 6 units from Q
  (Q.1 + v.1, Q.2 + v.2)

-- Define the set of points A satisfying the condition
def set_A (Q : ℝ × ℝ) : Set (ℝ × ℝ) :=
  {A | ∀ P ∈ circle_D Q, dist A (point_B Q) ≤ dist A P}

-- Define the circle with center B and radius 4
def circle_B (Q : ℝ × ℝ) : Set (ℝ × ℝ) :=
  {P | dist P (point_B Q) ≤ 4}

-- The theorem to prove
theorem set_A_equals_circle_B (Q : ℝ × ℝ) : set_A Q = circle_B Q := by
  sorry

end NUMINAMATH_CALUDE_set_A_equals_circle_B_l2945_294540


namespace NUMINAMATH_CALUDE_geometry_angle_probability_l2945_294599

def geometry_letters : Finset Char := {'G', 'E', 'O', 'M', 'T', 'R', 'Y'}
def angle_letters : Finset Char := {'A', 'N', 'G', 'L', 'E'}

theorem geometry_angle_probability : 
  (geometry_letters ∩ angle_letters).card / geometry_letters.card = 1 / 4 := by
sorry

end NUMINAMATH_CALUDE_geometry_angle_probability_l2945_294599


namespace NUMINAMATH_CALUDE_pet_shop_pricing_l2945_294593

theorem pet_shop_pricing (puppy_cost kitten_cost parakeet_cost : ℚ) : 
  puppy_cost = 3 * parakeet_cost →
  parakeet_cost = kitten_cost / 2 →
  2 * puppy_cost + 2 * kitten_cost + 3 * parakeet_cost = 130 →
  parakeet_cost = 10 := by
sorry

end NUMINAMATH_CALUDE_pet_shop_pricing_l2945_294593


namespace NUMINAMATH_CALUDE_tiffany_max_points_l2945_294541

/-- Represents the game setup and Tiffany's current state -/
structure GameState where
  initialMoney : ℕ
  costPerGame : ℕ
  ringsPerPlay : ℕ
  redBucketPoints : ℕ
  greenBucketPoints : ℕ
  gamesPlayed : ℕ
  redBucketsHit : ℕ
  greenBucketsHit : ℕ

/-- Calculates the maximum points achievable given a GameState -/
def maxPoints (state : GameState) : ℕ :=
  let pointsFromRed := state.redBucketsHit * state.redBucketPoints
  let pointsFromGreen := state.greenBucketsHit * state.greenBucketPoints
  let moneySpent := state.gamesPlayed * state.costPerGame
  let moneyLeft := state.initialMoney - moneySpent
  let gamesLeft := moneyLeft / state.costPerGame
  let maxPointsLastGame := state.ringsPerPlay * max state.redBucketPoints state.greenBucketPoints
  pointsFromRed + pointsFromGreen + gamesLeft * maxPointsLastGame

/-- Tiffany's game state -/
def tiffanyState : GameState where
  initialMoney := 3
  costPerGame := 1
  ringsPerPlay := 5
  redBucketPoints := 2
  greenBucketPoints := 3
  gamesPlayed := 2
  redBucketsHit := 4
  greenBucketsHit := 5

/-- Theorem stating that the maximum points Tiffany can achieve is 38 -/
theorem tiffany_max_points :
  maxPoints tiffanyState = 38 := by
  sorry

end NUMINAMATH_CALUDE_tiffany_max_points_l2945_294541


namespace NUMINAMATH_CALUDE_tap_b_fills_12_liters_l2945_294583

/-- Represents the water flow problem with two taps filling a bucket. -/
structure WaterFlow where
  bucket_volume : ℝ
  tap_a_rate : ℝ
  fill_time_both : ℝ

/-- The amount of water tap B fills in 20 minutes. -/
def tap_b_fill_20min (w : WaterFlow) : ℝ :=
  2 * (w.bucket_volume - w.tap_a_rate * w.fill_time_both)

/-- Theorem stating that tap B fills 12 liters in 20 minutes under given conditions. -/
theorem tap_b_fills_12_liters (w : WaterFlow)
  (h1 : w.bucket_volume = 36)
  (h2 : w.tap_a_rate = 3)
  (h3 : w.fill_time_both = 10) :
  tap_b_fill_20min w = 12 := by
  sorry

end NUMINAMATH_CALUDE_tap_b_fills_12_liters_l2945_294583


namespace NUMINAMATH_CALUDE_quadratic_factorization_sum_l2945_294521

theorem quadratic_factorization_sum (a b c : ℤ) : 
  (∀ x : ℝ, x^2 + 13*x + 30 = (x + a)*(x + b)) →
  (∀ x : ℝ, x^2 + 5*x - 50 = (x + b)*(x - c)) →
  a + b + c = 18 := by
sorry

end NUMINAMATH_CALUDE_quadratic_factorization_sum_l2945_294521


namespace NUMINAMATH_CALUDE_second_term_of_arithmetic_sequence_l2945_294586

/-- Given an arithmetic sequence where the sum of the first and third terms is 8,
    prove that the second term is 4. -/
theorem second_term_of_arithmetic_sequence (a d : ℝ) 
  (h : a + (a + 2 * d) = 8) : a + d = 4 := by
  sorry

end NUMINAMATH_CALUDE_second_term_of_arithmetic_sequence_l2945_294586


namespace NUMINAMATH_CALUDE_fraction_sum_from_hcf_lcm_and_sum_l2945_294551

theorem fraction_sum_from_hcf_lcm_and_sum (m n : ℕ+) 
  (hcf : Nat.gcd m n = 6)
  (lcm : Nat.lcm m n = 210)
  (sum : m + n = 80) :
  (1 : ℚ) / m + (1 : ℚ) / n = 2 / 31.5 := by
  sorry

end NUMINAMATH_CALUDE_fraction_sum_from_hcf_lcm_and_sum_l2945_294551


namespace NUMINAMATH_CALUDE_equation_solution_l2945_294508

theorem equation_solution (x : ℝ) (h : x ≠ -1) :
  (x^2 + x + 1) / (x + 1) = x + 3 ↔ x = -2/3 :=
by sorry

end NUMINAMATH_CALUDE_equation_solution_l2945_294508


namespace NUMINAMATH_CALUDE_gcf_of_84_112_210_l2945_294556

theorem gcf_of_84_112_210 : Nat.gcd 84 (Nat.gcd 112 210) = 14 := by
  sorry

end NUMINAMATH_CALUDE_gcf_of_84_112_210_l2945_294556


namespace NUMINAMATH_CALUDE_total_students_agreed_l2945_294591

def third_grade_students : ℕ := 256
def fourth_grade_students : ℕ := 525
def third_grade_agreement_rate : ℚ := 60 / 100
def fourth_grade_agreement_rate : ℚ := 45 / 100

theorem total_students_agreed :
  ⌊third_grade_agreement_rate * third_grade_students⌋ +
  ⌊fourth_grade_agreement_rate * fourth_grade_students⌋ = 390 := by
  sorry

end NUMINAMATH_CALUDE_total_students_agreed_l2945_294591


namespace NUMINAMATH_CALUDE_power_of_two_sum_l2945_294572

theorem power_of_two_sum : 2^4 + 2^4 + 2^5 + 2^5 = 96 := by
  sorry

end NUMINAMATH_CALUDE_power_of_two_sum_l2945_294572


namespace NUMINAMATH_CALUDE_polynomial_division_theorem_l2945_294569

theorem polynomial_division_theorem (x : ℝ) :
  x^6 + 5*x^4 + 3 = (x - 2) * (x^5 + 2*x^4 + 9*x^3 + 18*x^2 + 36*x + 72) + 147 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_division_theorem_l2945_294569


namespace NUMINAMATH_CALUDE_trailing_zeros_100_factorial_l2945_294561

/-- The number of trailing zeros in n! -/
def trailingZeros (n : ℕ) : ℕ :=
  (n / 5) + (n / 25)

/-- Theorem: The number of trailing zeros in 100! is 24 -/
theorem trailing_zeros_100_factorial : trailingZeros 100 = 24 := by
  sorry

end NUMINAMATH_CALUDE_trailing_zeros_100_factorial_l2945_294561


namespace NUMINAMATH_CALUDE_perpendicular_parallel_transitive_l2945_294588

/-- Represents a line in 3D space -/
structure Line3D where
  -- Add necessary fields/axioms for a line in 3D space
  -- This is a simplified representation

/-- Represents perpendicularity between two lines -/
def perpendicular (l1 l2 : Line3D) : Prop :=
  -- Define what it means for two lines to be perpendicular
  sorry

/-- Represents parallelism between two lines -/
def parallel (l1 l2 : Line3D) : Prop :=
  -- Define what it means for two lines to be parallel
  sorry

/-- Theorem: If line a is parallel to line b, and line l is perpendicular to a,
    then l is also perpendicular to b -/
theorem perpendicular_parallel_transitive (a b l : Line3D) :
  parallel a b → perpendicular l a → perpendicular l b := by
  sorry

end NUMINAMATH_CALUDE_perpendicular_parallel_transitive_l2945_294588


namespace NUMINAMATH_CALUDE_cafeteria_extra_apples_l2945_294532

/-- The number of extra apples in the cafeteria -/
def extra_apples (red_apples green_apples students_wanting_fruit : ℕ) : ℕ :=
  red_apples + green_apples - students_wanting_fruit

/-- Theorem: The cafeteria ends up with 40 extra apples -/
theorem cafeteria_extra_apples :
  extra_apples 42 7 9 = 40 := by
  sorry

end NUMINAMATH_CALUDE_cafeteria_extra_apples_l2945_294532


namespace NUMINAMATH_CALUDE_trinomial_perfect_fourth_power_l2945_294581

/-- A trinomial is a perfect fourth power for all integers if and only if its quadratic and linear coefficients are zero. -/
theorem trinomial_perfect_fourth_power (a b c : ℤ) :
  (∀ x : ℤ, ∃ y : ℤ, a * x^2 + b * x + c = y^4) → a = 0 ∧ b = 0 := by
  sorry

end NUMINAMATH_CALUDE_trinomial_perfect_fourth_power_l2945_294581


namespace NUMINAMATH_CALUDE_unique_satisfying_function_l2945_294560

/-- A function satisfying the given condition -/
def SatisfyingFunction (f : ℝ → ℝ) : Prop :=
  ∀ x y : ℝ, f (x + y) * f (x - y) = (f x - f y)^2 - (4 * x * y) * f y

/-- Theorem stating that the only function satisfying the condition is the zero function -/
theorem unique_satisfying_function :
  ∃! f : ℝ → ℝ, SatisfyingFunction f ∧ ∀ x : ℝ, f x = 0 := by
  sorry

end NUMINAMATH_CALUDE_unique_satisfying_function_l2945_294560


namespace NUMINAMATH_CALUDE_number_times_x_minus_3y_l2945_294513

/-- Given that 2x - y = 4 and kx - 3y = 12, prove that k = 6 -/
theorem number_times_x_minus_3y (x y k : ℝ) 
  (h1 : 2 * x - y = 4) 
  (h2 : k * x - 3 * y = 12) : 
  k = 6 := by
  sorry

end NUMINAMATH_CALUDE_number_times_x_minus_3y_l2945_294513


namespace NUMINAMATH_CALUDE_range_of_m_l2945_294509

-- Define the propositions p and q
def p (m : ℝ) : Prop := ∀ x y : ℝ, x^2 / (2*m) - y^2 / (m-2) = 1 → m > 2

def q (m : ℝ) : Prop := ∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ 
  x₁^2 + (2*m-3)*x₁ + 1 = 0 ∧ x₂^2 + (2*m-3)*x₂ + 1 = 0

-- State the theorem
theorem range_of_m : 
  (∀ m : ℝ, ¬(p m ∧ q m)) → 
  (∀ m : ℝ, p m ∨ q m) → 
  ∀ m : ℝ, (2 < m ∧ m ≤ 5/2) ∨ m < 1/2 :=
sorry

end NUMINAMATH_CALUDE_range_of_m_l2945_294509


namespace NUMINAMATH_CALUDE_train_crossing_time_l2945_294557

/-- Calculates the time for a train to cross a signal pole given its length, 
    the platform length, and the time to cross the platform. -/
theorem train_crossing_time 
  (train_length : ℝ) 
  (platform_length : ℝ) 
  (platform_crossing_time : ℝ) 
  (h1 : train_length = 300)
  (h2 : platform_length = 550.0000000000001)
  (h3 : platform_crossing_time = 51) :
  (train_length / ((train_length + platform_length) / platform_crossing_time)) = 18 := by
sorry

end NUMINAMATH_CALUDE_train_crossing_time_l2945_294557


namespace NUMINAMATH_CALUDE_circle_center_coordinates_l2945_294550

/-- The center of a circle satisfying given conditions -/
theorem circle_center_coordinates :
  ∃ (x y : ℝ),
    (x - 2*y = 0) ∧
    (3*x - 4*y = 20) ∧
    (x = 20 ∧ y = 10) := by
  sorry

end NUMINAMATH_CALUDE_circle_center_coordinates_l2945_294550


namespace NUMINAMATH_CALUDE_value_of_x_l2945_294589

theorem value_of_x : ∃ X : ℚ, (3/4 : ℚ) * (1/8 : ℚ) * X = (1/2 : ℚ) * (1/4 : ℚ) * 120 ∧ X = 160 := by
  sorry

end NUMINAMATH_CALUDE_value_of_x_l2945_294589


namespace NUMINAMATH_CALUDE_sequence_sum_zero_l2945_294529

-- Define the sequence type
def Sequence := Fin 12 → ℤ

-- Define the property of sum of three consecutive terms being 40
def ConsecutiveSum (seq : Sequence) : Prop :=
  ∀ i : Fin 10, seq i + seq (i + 1) + seq (i + 2) = 40

-- Define the theorem
theorem sequence_sum_zero (seq : Sequence) 
  (h1 : ConsecutiveSum seq) 
  (h2 : seq 2 = 9) : 
  seq 0 + seq 11 = 0 :=
sorry

end NUMINAMATH_CALUDE_sequence_sum_zero_l2945_294529


namespace NUMINAMATH_CALUDE_remaining_amount_with_taxes_remaining_amount_is_622_54_l2945_294587

/-- Calculates the remaining amount to be paid including taxes for a product purchase --/
theorem remaining_amount_with_taxes (deposit_percent : ℝ) (cash_deposit : ℝ) (reward_points : ℕ) 
  (point_value : ℝ) (tax_rate : ℝ) (luxury_tax_rate : ℝ) : ℝ :=
  let total_deposit := cash_deposit + (reward_points : ℝ) * point_value
  let total_price := total_deposit / deposit_percent
  let remaining_before_taxes := total_price - total_deposit
  let tax := remaining_before_taxes * tax_rate
  let luxury_tax := remaining_before_taxes * luxury_tax_rate
  remaining_before_taxes + tax + luxury_tax

/-- The remaining amount to be paid including taxes is $622.54 --/
theorem remaining_amount_is_622_54 :
  remaining_amount_with_taxes 0.30 150 800 0.10 0.12 0.04 = 622.54 := by
  sorry

end NUMINAMATH_CALUDE_remaining_amount_with_taxes_remaining_amount_is_622_54_l2945_294587


namespace NUMINAMATH_CALUDE_point_coordinate_product_l2945_294553

theorem point_coordinate_product : 
  ∀ y₁ y₂ : ℝ,
  (((4 - (-2))^2 + (y₁ - 5)^2 = 13^2) ∧
   ((4 - (-2))^2 + (y₂ - 5)^2 = 13^2) ∧
   (∀ y : ℝ, ((4 - (-2))^2 + (y - 5)^2 = 13^2) → (y = y₁ ∨ y = y₂))) →
  y₁ * y₂ = -108 := by
sorry

end NUMINAMATH_CALUDE_point_coordinate_product_l2945_294553


namespace NUMINAMATH_CALUDE_shelby_total_stars_l2945_294503

/-- The number of gold stars Shelby earned yesterday -/
def yesterdays_stars : ℕ := 4

/-- The number of gold stars Shelby earned today -/
def todays_stars : ℕ := 3

/-- The total number of gold stars Shelby earned -/
def total_stars : ℕ := yesterdays_stars + todays_stars

/-- Theorem: The total number of gold stars Shelby earned is 7 -/
theorem shelby_total_stars : total_stars = 7 := by
  sorry

end NUMINAMATH_CALUDE_shelby_total_stars_l2945_294503


namespace NUMINAMATH_CALUDE_four_foldable_positions_l2945_294548

/-- Represents a position where an additional square can be attached --/
inductive Position
| Top
| TopRight
| Right
| BottomRight
| Bottom
| BottomLeft
| Left
| TopLeft
| CenterTop
| CenterRight
| CenterBottom
| CenterLeft

/-- Represents the cross-shaped polygon --/
structure CrossPolygon :=
  (squares : Fin 5 → Unit)

/-- Represents the resulting polygon after attaching an additional square --/
structure ResultingPolygon :=
  (base : CrossPolygon)
  (additional : Position)

/-- Predicate to check if a resulting polygon can be folded into a cube with one face missing --/
def can_fold_to_cube (p : ResultingPolygon) : Prop :=
  sorry

/-- The main theorem --/
theorem four_foldable_positions :
  ∃ (valid_positions : Finset Position),
    (valid_positions.card = 4) ∧
    (∀ p : Position, p ∈ valid_positions ↔ 
      can_fold_to_cube ⟨CrossPolygon.mk (λ _ => Unit.unit), p⟩) :=
  sorry

end NUMINAMATH_CALUDE_four_foldable_positions_l2945_294548


namespace NUMINAMATH_CALUDE_e_value_l2945_294517

-- Define variables
variable (p j t b a e : ℝ)

-- Define conditions
def condition1 : Prop := j = 0.75 * p
def condition2 : Prop := j = 0.8 * t
def condition3 : Prop := t = p * (1 - e / 100)
def condition4 : Prop := b = 1.4 * j
def condition5 : Prop := a = 0.85 * b
def condition6 : Prop := e = 2 * ((p - a) / p) * 100

-- Theorem statement
theorem e_value (h1 : condition1 p j) (h2 : condition2 j t) (h3 : condition3 p t e)
                (h4 : condition4 j b) (h5 : condition5 b a) (h6 : condition6 p a e) :
  e = 21.5 := by
  sorry

end NUMINAMATH_CALUDE_e_value_l2945_294517


namespace NUMINAMATH_CALUDE_room_dimensions_l2945_294502

theorem room_dimensions (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (h_not_equal : b ≠ c) :
  let x := (b^2 * (b^2 - a^2) / (b^2 - c^2))^(1/4)
  let y := a / x
  let z := b / x
  let u := c * x / b
  ∃ (room_I room_II room_III : ℝ × ℝ),
    room_I.1 * room_I.2 = a ∧
    room_II.1 * room_II.2 = b ∧
    room_III.1 * room_III.2 = c ∧
    room_I.1 = room_II.1 ∧
    room_II.2 = room_III.2 ∧
    room_I.1^2 + room_I.2^2 = room_III.1^2 + room_III.2^2 ∧
    room_I = (x, y) ∧
    room_II = (x, z) ∧
    room_III = (u, z) := by
  sorry

#check room_dimensions

end NUMINAMATH_CALUDE_room_dimensions_l2945_294502


namespace NUMINAMATH_CALUDE_solve_system_l2945_294577

theorem solve_system (x y : ℝ) (eq1 : x - y = 8) (eq2 : x + 2*y = 14) : x = 10 := by
  sorry

end NUMINAMATH_CALUDE_solve_system_l2945_294577


namespace NUMINAMATH_CALUDE_statement_implies_innocence_statement_proves_innocence_l2945_294579

-- Define the possible roles of a person
inductive Role
  | Knight
  | Liar
  | Normal

-- Define the statement made by the defendant
def statement (role : Role) (guilty : Bool) : Prop :=
  (role = Role.Knight ∧ ¬guilty) ∨ (role = Role.Liar ∧ guilty)

-- Define what it means to be a criminal
def isCriminal (role : Role) : Prop :=
  role = Role.Knight ∨ role = Role.Liar

-- Theorem: The statement implies innocence for all possible roles
theorem statement_implies_innocence (role : Role) :
  (∀ r, isCriminal r → (statement r true ↔ ¬statement r false)) →
  statement role false →
  ¬isCriminal role ∨ ¬statement role true :=
by sorry

-- The main theorem: The statement proves innocence
theorem statement_proves_innocence :
  ∀ role, ¬isCriminal role ∨ ¬statement role true :=
by sorry

end NUMINAMATH_CALUDE_statement_implies_innocence_statement_proves_innocence_l2945_294579


namespace NUMINAMATH_CALUDE_cos_54_degrees_l2945_294530

theorem cos_54_degrees : Real.cos (54 * π / 180) = (-1 + Real.sqrt 5) / 4 := by
  sorry

end NUMINAMATH_CALUDE_cos_54_degrees_l2945_294530


namespace NUMINAMATH_CALUDE_annie_extracurricular_hours_l2945_294528

/-- Calculates the total extracurricular hours before midterms -/
def extracurricular_hours_before_midterms (
  chess_hours_per_week : ℕ)
  (drama_hours_per_week : ℕ)
  (glee_hours_per_week : ℕ)
  (weeks_in_semester : ℕ)
  (weeks_off_sick : ℕ) : ℕ :=
  let total_hours_per_week := chess_hours_per_week + drama_hours_per_week + glee_hours_per_week
  let weeks_before_midterms := weeks_in_semester / 2
  let active_weeks := weeks_before_midterms - weeks_off_sick
  total_hours_per_week * active_weeks

theorem annie_extracurricular_hours :
  extracurricular_hours_before_midterms 2 8 3 12 2 = 52 := by
  sorry

end NUMINAMATH_CALUDE_annie_extracurricular_hours_l2945_294528


namespace NUMINAMATH_CALUDE_perpendicular_iff_m_eq_two_l2945_294590

/-- Two vectors in R² -/
def Vector2 := ℝ × ℝ

/-- Dot product of two vectors in R² -/
def dot_product (v w : Vector2) : ℝ :=
  v.1 * w.1 + v.2 * w.2

/-- Definition of vector AB -/
def AB (m : ℝ) : Vector2 := (m + 3, 2 * m + 1)

/-- Definition of vector CD -/
def CD (m : ℝ) : Vector2 := (m + 3, -5)

/-- Theorem stating that AB and CD are perpendicular if and only if m = 2 -/
theorem perpendicular_iff_m_eq_two :
  ∀ m : ℝ, dot_product (AB m) (CD m) = 0 ↔ m = 2 :=
sorry

end NUMINAMATH_CALUDE_perpendicular_iff_m_eq_two_l2945_294590


namespace NUMINAMATH_CALUDE_remainder_after_adding_2024_l2945_294542

theorem remainder_after_adding_2024 (n : ℤ) (h : n % 8 = 3) : (n + 2024) % 8 = 3 := by
  sorry

end NUMINAMATH_CALUDE_remainder_after_adding_2024_l2945_294542


namespace NUMINAMATH_CALUDE_smallest_possible_n_l2945_294543

theorem smallest_possible_n (a b c n : ℕ) : 
  a < b → b < c → c < n → 
  a + b + c + n = 100 → 
  (∀ m : ℕ, m < n → ¬∃ x y z : ℕ, x < y ∧ y < z ∧ z < m ∧ x + y + z + m = 100) →
  n = 27 := by
  sorry

end NUMINAMATH_CALUDE_smallest_possible_n_l2945_294543


namespace NUMINAMATH_CALUDE_distinct_z_values_l2945_294544

/-- Given two integers x and y where:
    1. 200 ≤ x ≤ 999
    2. 100 ≤ y ≤ 999
    3. y is the number formed by reversing the digits of x
    4. z = x + y
    This theorem states that there are exactly 1878 distinct possible values for z. -/
theorem distinct_z_values (x y z : ℕ) 
  (hx : 200 ≤ x ∧ x ≤ 999)
  (hy : 100 ≤ y ∧ y ≤ 999)
  (hrev : y = 100 * (x % 10) + 10 * ((x / 10) % 10) + (x / 100))
  (hz : z = x + y) :
  ∃! (s : Finset ℕ), s = {z | ∃ (x y : ℕ), 
    200 ≤ x ∧ x ≤ 999 ∧
    100 ≤ y ∧ y ≤ 999 ∧
    y = 100 * (x % 10) + 10 * ((x / 10) % 10) + (x / 100) ∧
    z = x + y} ∧ 
  Finset.card s = 1878 :=
by sorry

end NUMINAMATH_CALUDE_distinct_z_values_l2945_294544


namespace NUMINAMATH_CALUDE_crayon_box_count_l2945_294563

theorem crayon_box_count : ∀ (total : ℕ),
  (total : ℚ) / 3 + (total : ℚ) * (1 / 5) + 56 = total →
  total = 120 := by
sorry

end NUMINAMATH_CALUDE_crayon_box_count_l2945_294563


namespace NUMINAMATH_CALUDE_min_sections_problem_l2945_294552

theorem min_sections_problem (num_boys num_girls max_per_section : ℕ) 
  (h_boys : num_boys = 408)
  (h_girls : num_girls = 240)
  (h_max : max_per_section = 24)
  (h_ratio : ∃ (x : ℕ), x > 0 ∧ num_boys ≤ 3 * x * max_per_section ∧ num_girls ≤ 2 * x * max_per_section) :
  ∃ (total_sections : ℕ), 
    total_sections = 30 ∧
    ∃ (boys_sections girls_sections : ℕ),
      boys_sections + girls_sections = total_sections ∧
      3 * girls_sections = 2 * boys_sections ∧
      num_boys ≤ boys_sections * max_per_section ∧
      num_girls ≤ girls_sections * max_per_section ∧
      ∀ (other_total : ℕ),
        (∃ (other_boys other_girls : ℕ),
          other_boys + other_girls = other_total ∧
          3 * other_girls = 2 * other_boys ∧
          num_boys ≤ other_boys * max_per_section ∧
          num_girls ≤ other_girls * max_per_section) →
        other_total ≥ total_sections :=
by sorry

end NUMINAMATH_CALUDE_min_sections_problem_l2945_294552


namespace NUMINAMATH_CALUDE_odd_terms_in_binomial_expansion_l2945_294545

theorem odd_terms_in_binomial_expansion (p q : ℤ) (hp : Odd p) (hq : Odd q) :
  (Finset.range 9).filter (fun k => Odd (Nat.choose 8 k * (p + q)^(8 - k) * p^k)) = {0, 8} := by
  sorry

end NUMINAMATH_CALUDE_odd_terms_in_binomial_expansion_l2945_294545


namespace NUMINAMATH_CALUDE_other_number_is_twenty_l2945_294568

theorem other_number_is_twenty (x y : ℤ) 
  (sum_eq : 3 * x + 2 * y = 145) 
  (one_is_35 : x = 35 ∨ y = 35) : 
  (x ≠ 35 → x = 20) ∧ (y ≠ 35 → y = 20) :=
sorry

end NUMINAMATH_CALUDE_other_number_is_twenty_l2945_294568


namespace NUMINAMATH_CALUDE_unique_valid_number_l2945_294546

def is_valid_number (n : ℕ) : Prop :=
  ∃ (a b c : ℕ),
    n = 100 * a + 10 * b + c ∧
    b = 3 ∧
    a + c = 5 ∧
    a ≠ b ∧ b ≠ c ∧ a ≠ c ∧
    ∃ (d : ℕ), n + 124 = 111 * d

theorem unique_valid_number : ∃! n : ℕ, is_valid_number n ∧ n = 431 :=
sorry

end NUMINAMATH_CALUDE_unique_valid_number_l2945_294546


namespace NUMINAMATH_CALUDE_function_value_at_inverse_point_l2945_294511

noncomputable def log_log_2_10 : ℝ := Real.log (Real.log 10 / Real.log 2)

theorem function_value_at_inverse_point 
  (a b : ℝ) 
  (f : ℝ → ℝ) 
  (h : ∀ x, f x = a * x^3 + b * Real.sin x + 4) 
  (h1 : f log_log_2_10 = 5) : 
  f (- log_log_2_10) = 3 := by
sorry

end NUMINAMATH_CALUDE_function_value_at_inverse_point_l2945_294511


namespace NUMINAMATH_CALUDE_tic_tac_toe_winnings_l2945_294564

theorem tic_tac_toe_winnings
  (total_games : ℕ)
  (tied_games : ℕ)
  (net_loss : ℤ)
  (h_total : total_games = 100)
  (h_tied : tied_games = 40)
  (h_loss : net_loss = 30)
  (h_win_value : ℤ)
  (h_tie_value : ℤ)
  (h_lose_value : ℤ)
  (h_win_val : h_win_value = 1)
  (h_tie_val : h_tie_value = 0)
  (h_lose_val : h_lose_value = -2)
  : ∃ (won_games : ℕ),
    won_games = 30 ∧
    won_games + tied_games + (total_games - won_games - tied_games) = total_games ∧
    h_win_value * won_games + h_tie_value * tied_games + h_lose_value * (total_games - won_games - tied_games) = -net_loss :=
by sorry

end NUMINAMATH_CALUDE_tic_tac_toe_winnings_l2945_294564


namespace NUMINAMATH_CALUDE_typing_service_problem_l2945_294585

/-- The typing service problem -/
theorem typing_service_problem 
  (total_pages : ℕ) 
  (twice_revised_pages : ℕ) 
  (initial_cost : ℕ) 
  (revision_cost : ℕ) 
  (total_cost : ℕ) 
  (h1 : total_pages = 100)
  (h2 : twice_revised_pages = 20)
  (h3 : initial_cost = 5)
  (h4 : revision_cost = 3)
  (h5 : total_cost = 710) :
  ∃ (once_revised_pages : ℕ),
    once_revised_pages = 30 ∧
    total_cost = 
      initial_cost * total_pages + 
      revision_cost * once_revised_pages + 
      2 * revision_cost * twice_revised_pages :=
by
  sorry


end NUMINAMATH_CALUDE_typing_service_problem_l2945_294585


namespace NUMINAMATH_CALUDE_sqrt_two_squared_cubed_l2945_294596

theorem sqrt_two_squared_cubed : (Real.sqrt (Real.sqrt 2)^2)^3 = 2 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_two_squared_cubed_l2945_294596


namespace NUMINAMATH_CALUDE_factorial_sum_square_solutions_l2945_294576

def factorial (n : ℕ) : ℕ := (List.range n).foldl (· * ·) 1

def factorial_sum (n : ℕ) : ℕ := (List.range n).map factorial |>.sum

theorem factorial_sum_square_solutions :
  ∀ m n : ℕ+, m^2 = factorial_sum n ↔ (m = 1 ∧ n = 1) ∨ (m = 3 ∧ n = 3) := by
  sorry

end NUMINAMATH_CALUDE_factorial_sum_square_solutions_l2945_294576


namespace NUMINAMATH_CALUDE_range_of_m_l2945_294512

-- Define the function f
def f (x : ℝ) : ℝ := x^2 - 4*x + 5

-- Define the theorem
theorem range_of_m (m : ℝ) :
  (∀ x ∈ Set.Icc 0 m, f x ≤ 5) ∧
  (∃ x ∈ Set.Icc 0 m, f x = 5) ∧
  (∀ x ∈ Set.Icc 0 m, f x ≥ 1) ∧
  (∃ x ∈ Set.Icc 0 m, f x = 1) →
  m ∈ Set.Icc 2 4 :=
by sorry

end NUMINAMATH_CALUDE_range_of_m_l2945_294512


namespace NUMINAMATH_CALUDE_sarahs_bowling_score_l2945_294520

theorem sarahs_bowling_score (g s : ℕ) : 
  s = g + 50 → (s + g) / 2 = 95 → s = 120 := by
sorry

end NUMINAMATH_CALUDE_sarahs_bowling_score_l2945_294520


namespace NUMINAMATH_CALUDE_paula_tickets_l2945_294562

/-- The number of times Paula wants to ride the go-karts -/
def go_kart_rides : ℕ := 1

/-- The number of times Paula wants to ride the bumper cars -/
def bumper_car_rides : ℕ := 4

/-- The number of tickets required for one go-kart ride -/
def go_kart_tickets : ℕ := 4

/-- The number of tickets required for one bumper car ride -/
def bumper_car_tickets : ℕ := 5

/-- The total number of tickets Paula needs -/
def total_tickets : ℕ := go_kart_rides * go_kart_tickets + bumper_car_rides * bumper_car_tickets

theorem paula_tickets : total_tickets = 24 := by
  sorry

end NUMINAMATH_CALUDE_paula_tickets_l2945_294562


namespace NUMINAMATH_CALUDE_horner_method_example_l2945_294595

def f (x : ℝ) : ℝ := 3 * x^3 + x - 3

theorem horner_method_example : f 3 = 28 := by
  sorry

end NUMINAMATH_CALUDE_horner_method_example_l2945_294595


namespace NUMINAMATH_CALUDE_marathon_distance_theorem_l2945_294501

/-- The length of a marathon in miles -/
def marathon_miles : ℕ := 26

/-- The additional length of a marathon in yards -/
def marathon_yards : ℕ := 312

/-- The number of yards in a mile -/
def yards_per_mile : ℕ := 1760

/-- The number of marathons Joanna has run -/
def num_marathons : ℕ := 8

/-- The total distance Joanna has run in yards -/
def total_distance : ℕ := num_marathons * (marathon_miles * yards_per_mile + marathon_yards)

theorem marathon_distance_theorem :
  ∃ (m : ℕ) (y : ℕ), total_distance = m * yards_per_mile + y ∧ y = 736 ∧ y < yards_per_mile :=
by sorry

end NUMINAMATH_CALUDE_marathon_distance_theorem_l2945_294501


namespace NUMINAMATH_CALUDE_cubic_equation_solution_l2945_294514

theorem cubic_equation_solution (x : ℝ) (hx : x ≠ 0) :
  1 - 6 / x + 9 / x^2 - 2 / x^3 = 0 →
  (3 / x = 3 / 2) ∨ (3 / x = 3 / (2 + Real.sqrt 3)) ∨ (3 / x = 3 / (2 - Real.sqrt 3)) :=
by sorry

end NUMINAMATH_CALUDE_cubic_equation_solution_l2945_294514


namespace NUMINAMATH_CALUDE_marbles_difference_l2945_294597

/-- The number of marbles Cindy and Lisa have after Cindy gives some to Lisa -/
def marbles_after_giving (cindy_initial : ℕ) (lisa_initial : ℕ) (marbles_given : ℕ) :
  ℕ × ℕ :=
  (cindy_initial - marbles_given, lisa_initial + marbles_given)

/-- The theorem stating the difference in marbles after Cindy gives some to Lisa -/
theorem marbles_difference
  (cindy_initial : ℕ)
  (lisa_initial : ℕ)
  (marbles_given : ℕ)
  (h1 : cindy_initial = 20)
  (h2 : cindy_initial = lisa_initial + 5)
  (h3 : marbles_given = 12) :
  (marbles_after_giving cindy_initial lisa_initial marbles_given).2 -
  (marbles_after_giving cindy_initial lisa_initial marbles_given).1 = 19 := by
  sorry

end NUMINAMATH_CALUDE_marbles_difference_l2945_294597


namespace NUMINAMATH_CALUDE_sports_league_games_l2945_294525

/-- Calculates the number of games in a sports league with two divisions -/
theorem sports_league_games (n₁ n₂ : ℕ) (intra_games : ℕ) (inter_games : ℕ) :
  n₁ = 5 →
  n₂ = 6 →
  intra_games = 3 →
  inter_games = 2 →
  (n₁ * (n₁ - 1) * intra_games / 2) +
  (n₂ * (n₂ - 1) * intra_games / 2) +
  (n₁ * n₂ * inter_games) = 135 :=
by
  sorry

#check sports_league_games

end NUMINAMATH_CALUDE_sports_league_games_l2945_294525


namespace NUMINAMATH_CALUDE_rational_square_decomposition_l2945_294575

theorem rational_square_decomposition (r : ℚ) :
  ∃ (S : Set (ℚ × ℚ)), (Set.Infinite S) ∧ (∀ (x y : ℚ), (x, y) ∈ S → x^2 + y^2 = r^2) :=
sorry

end NUMINAMATH_CALUDE_rational_square_decomposition_l2945_294575


namespace NUMINAMATH_CALUDE_circle_theorem_l2945_294566

-- Define the circle and points
def Circle (O : ℝ × ℝ) (r : ℝ) := {P : ℝ × ℝ | (P.1 - O.1)^2 + (P.2 - O.2)^2 = r^2}

def O : ℝ × ℝ := (0, 0)
def r : ℝ := 52

-- Define the points based on the given conditions
theorem circle_theorem (A B : ℝ × ℝ) (P Q : ℝ × ℝ) :
  A ∈ Circle O r →
  B ∈ Circle O r →
  P.1 = 28 ∧ P.2 = 0 →
  (Q.1 - A.1)^2 + (Q.2 - A.2)^2 = 15^2 →
  (Q.1 - P.1)^2 + (Q.2 - P.2)^2 = 15^2 →
  ∃ t : ℝ, Q = (t * B.1, t * B.2) →
  (B.1 - Q.1)^2 + (B.2 - Q.2)^2 = 11^2 :=
by sorry


end NUMINAMATH_CALUDE_circle_theorem_l2945_294566


namespace NUMINAMATH_CALUDE_farm_legs_l2945_294554

/-- The total number of animal legs on a farm with ducks, dogs, and spiders -/
def total_legs (num_ducks : ℕ) (num_dogs : ℕ) (num_spiders : ℕ) (num_three_legged_dogs : ℕ) : ℕ :=
  2 * num_ducks + 4 * (num_dogs - num_three_legged_dogs) + 3 * num_three_legged_dogs + 8 * num_spiders

/-- Theorem stating that the total number of animal legs on the farm is 55 -/
theorem farm_legs : total_legs 6 5 3 1 = 55 := by
  sorry

end NUMINAMATH_CALUDE_farm_legs_l2945_294554


namespace NUMINAMATH_CALUDE_madeline_work_hours_l2945_294518

def monthly_expenses : ℕ := 1200 + 400 + 200 + 60
def emergency_savings : ℕ := 200
def daytime_hourly_rate : ℕ := 15
def bakery_hourly_rate : ℕ := 12
def bakery_weekly_hours : ℕ := 5
def tax_rate : ℚ := 15 / 100

theorem madeline_work_hours :
  ∃ (h : ℕ), h ≥ 146 ∧
  (h * daytime_hourly_rate + 4 * bakery_weekly_hours * bakery_hourly_rate) * (1 - tax_rate) ≥ 
  (monthly_expenses + emergency_savings : ℚ) ∧
  ∀ (k : ℕ), k < h →
  (k * daytime_hourly_rate + 4 * bakery_weekly_hours * bakery_hourly_rate) * (1 - tax_rate) <
  (monthly_expenses + emergency_savings : ℚ) :=
by sorry

end NUMINAMATH_CALUDE_madeline_work_hours_l2945_294518


namespace NUMINAMATH_CALUDE_parallelogram_sum_l2945_294515

/-- A parallelogram with sides of lengths 5, 11, 3y+2, and 4x-1 -/
structure Parallelogram (x y : ℝ) :=
  (side1 : ℝ := 5)
  (side2 : ℝ := 11)
  (side3 : ℝ := 3 * y + 2)
  (side4 : ℝ := 4 * x - 1)

/-- Theorem: In a parallelogram with sides of lengths 5, 11, 3y+2, and 4x-1, x + y = 4 -/
theorem parallelogram_sum (x y : ℝ) (p : Parallelogram x y) : x + y = 4 := by
  sorry

end NUMINAMATH_CALUDE_parallelogram_sum_l2945_294515


namespace NUMINAMATH_CALUDE_partial_fraction_sum_l2945_294526

theorem partial_fraction_sum (p q r A B C : ℝ) : 
  p ≠ q ∧ q ≠ r ∧ p ≠ r →
  (∀ x : ℝ, x^3 - 20*x^2 + 125*x - 500 = (x - p)*(x - q)*(x - r)) →
  (∀ s : ℝ, s ≠ p ∧ s ≠ q ∧ s ≠ r → 
    1 / (s^3 - 20*s^2 + 125*s - 500) = A / (s - p) + B / (s - q) + C / (s - r)) →
  1 / A + 1 / B + 1 / C = 720 := by
sorry

end NUMINAMATH_CALUDE_partial_fraction_sum_l2945_294526


namespace NUMINAMATH_CALUDE_right_triangle_identification_l2945_294567

def is_right_triangle (a b c : ℕ) : Prop :=
  a * a + b * b = c * c ∨ a * a + c * c = b * b ∨ b * b + c * c = a * a

theorem right_triangle_identification :
  (is_right_triangle 5 12 13) ∧
  ¬(is_right_triangle 6 7 8) ∧
  ¬(is_right_triangle 3 4 6) ∧
  ¬(is_right_triangle 7 12 15) :=
by sorry

end NUMINAMATH_CALUDE_right_triangle_identification_l2945_294567


namespace NUMINAMATH_CALUDE_carpooling_distance_ratio_l2945_294598

def distance_to_first_friend : ℝ := 8

def distance_to_second_friend : ℝ := 4

def distance_to_work (d1 d2 : ℝ) : ℝ := 3 * (d1 + d2)

theorem carpooling_distance_ratio :
  distance_to_work distance_to_first_friend distance_to_second_friend = 36 →
  distance_to_second_friend / distance_to_first_friend = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_carpooling_distance_ratio_l2945_294598


namespace NUMINAMATH_CALUDE_tangent_line_equation_l2945_294584

/-- The equation of the tangent line to the circle (x-1)^2 + y^2 = 5 at the point (2, 2) is x + 2y - 6 = 0 -/
theorem tangent_line_equation (x y : ℝ) : 
  (∀ x y, (x - 1)^2 + y^2 = 5) →  -- Circle equation
  (2 - 1)^2 + 2^2 = 5 →           -- Point (2, 2) lies on the circle
  x + 2*y - 6 = 0                 -- Equation of the tangent line
    ↔ 
  ((x - 1)^2 + y^2 = 5 → (x - 2) + 2*(y - 2) = 0) -- Tangent line property
  :=
by sorry

end NUMINAMATH_CALUDE_tangent_line_equation_l2945_294584


namespace NUMINAMATH_CALUDE_twenty_new_homes_l2945_294592

/-- Calculates the number of new trailer homes added -/
def new_trailer_homes (initial_count : ℕ) (initial_avg_age : ℕ) (time_passed : ℕ) (current_avg_age : ℕ) : ℕ :=
  let total_age := initial_count * (initial_avg_age + time_passed)
  let k := (total_age - initial_count * current_avg_age) / (current_avg_age - time_passed)
  k

/-- Theorem stating that 20 new trailer homes were added -/
theorem twenty_new_homes :
  new_trailer_homes 30 15 3 12 = 20 := by
  sorry

#eval new_trailer_homes 30 15 3 12

end NUMINAMATH_CALUDE_twenty_new_homes_l2945_294592


namespace NUMINAMATH_CALUDE_f_properties_l2945_294574

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := Real.log (a * x + 1) + (1 - x) / (1 + x)

theorem f_properties (a : ℝ) (h_a : a > 0) :
  -- 1. If f'(1) = 0, then a = 1
  (∀ x : ℝ, x > 0 → (deriv (f a)) x = 0 → x = 1 → a = 1) ∧
  -- 2. For a ≥ 2, f'(x) > 0 for all x > 0
  (a ≥ 2 → ∀ x : ℝ, x > 0 → (deriv (f a)) x > 0) ∧
  -- 3. For 0 < a < 2, f'(x) < 0 for 0 < x < sqrt((2-a)/a) and f'(x) > 0 for x > sqrt((2-a)/a)
  (0 < a ∧ a < 2 → 
    (∀ x : ℝ, 0 < x ∧ x < Real.sqrt ((2 - a) / a) → (deriv (f a)) x < 0) ∧
    (∀ x : ℝ, x > Real.sqrt ((2 - a) / a) → (deriv (f a)) x > 0)) ∧
  -- 4. The minimum value of f(x) is 1 if and only if a ≥ 2
  (∃ x : ℝ, x ≥ 0 ∧ ∀ y : ℝ, y ≥ 0 → f a x ≤ f a y ∧ f a x = 1) ↔ a ≥ 2 :=
sorry

end NUMINAMATH_CALUDE_f_properties_l2945_294574


namespace NUMINAMATH_CALUDE_unique_solution_iff_a_equals_one_l2945_294519

-- Define the equation
def equation (a x : ℝ) : Prop :=
  3^(x^2 - 2*a*x + a^2) = a*x^2 - 2*a^2*x + a^3 + a^2 - 4*a + 4

-- Define the property of having exactly one solution
def has_exactly_one_solution (a : ℝ) : Prop :=
  ∃! x, equation a x

-- Theorem statement
theorem unique_solution_iff_a_equals_one :
  ∀ a : ℝ, has_exactly_one_solution a ↔ a = 1 := by sorry

end NUMINAMATH_CALUDE_unique_solution_iff_a_equals_one_l2945_294519


namespace NUMINAMATH_CALUDE_xyz_value_l2945_294580

theorem xyz_value (x y z : ℝ) 
  (hx : x > 0) (hy : y > 0) (hz : z > 0)
  (eq1 : x + 1/y = 5)
  (eq2 : y + 1/z = 3)
  (eq3 : z + 1/x = 2) :
  x * y * z = 10 + 3 * Real.sqrt 11 := by
sorry

end NUMINAMATH_CALUDE_xyz_value_l2945_294580


namespace NUMINAMATH_CALUDE_largest_expression_l2945_294516

def P : ℕ := 3 * 2024^2025
def Q : ℕ := 2024^2025
def R : ℕ := 2023 * 2024^2024
def S : ℕ := 3 * 2024^2024
def T : ℕ := 2024^2024
def U : ℕ := 2024^2023

theorem largest_expression :
  (P - Q ≥ Q - R) ∧
  (P - Q ≥ R - S) ∧
  (P - Q ≥ S - T) ∧
  (P - Q ≥ T - U) :=
by sorry

end NUMINAMATH_CALUDE_largest_expression_l2945_294516


namespace NUMINAMATH_CALUDE_final_output_is_127_l2945_294506

def flowchart_output : ℕ → ℕ
| 0 => 0
| (n + 1) => let a := flowchart_output n; if a < 100 then 2 * a + 1 else a

theorem final_output_is_127 : flowchart_output 7 = 127 := by
  sorry

end NUMINAMATH_CALUDE_final_output_is_127_l2945_294506


namespace NUMINAMATH_CALUDE_polynomial_simplification_l2945_294537

theorem polynomial_simplification (x : ℝ) : 
  (3 * x^2 + 8 * x - 5) - (2 * x^2 + 6 * x - 15) = x^2 + 2 * x + 10 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_simplification_l2945_294537


namespace NUMINAMATH_CALUDE_ellipse_product_l2945_294565

/-- A point in 2D space -/
structure Point := (x : ℝ) (y : ℝ)

/-- An ellipse defined by its center, major axis, minor axis, and a focus -/
structure Ellipse :=
  (center : Point)
  (majorAxis : ℝ)
  (minorAxis : ℝ)
  (focus : Point)

/-- Distance between two points -/
def distance (p1 p2 : Point) : ℝ := sorry

/-- Diameter of the incircle of a right triangle -/
def incircleDiameter (leg1 leg2 hypotenuse : ℝ) : ℝ := sorry

/-- Main theorem -/
theorem ellipse_product (e : Ellipse) :
  distance e.center e.focus = 8 →
  incircleDiameter e.minorAxis 8 e.majorAxis = 4 →
  e.majorAxis * e.minorAxis = 240 := by sorry

end NUMINAMATH_CALUDE_ellipse_product_l2945_294565


namespace NUMINAMATH_CALUDE_constant_term_binomial_expansion_l2945_294510

theorem constant_term_binomial_expansion :
  ∀ n : ℕ, n > 0 →
  ∃ k : ℕ, k > 0 ∧ k ≤ n + 1 ∧
  (∀ r : ℕ, r ≥ 0 ∧ r ≤ n →
    (Nat.choose n r * (1 : ℚ)) = 0 ∨ (2 * r = n → k = r + 1)) →
  k = 6 ∧ n = 10 :=
by sorry

end NUMINAMATH_CALUDE_constant_term_binomial_expansion_l2945_294510


namespace NUMINAMATH_CALUDE_rose_crystal_beads_l2945_294555

/-- The number of beads in each bracelet -/
def beads_per_bracelet : ℕ := 8

/-- The number of metal beads Nancy has -/
def nancy_metal_beads : ℕ := 40

/-- The number of pearl beads Nancy has more than metal beads -/
def nancy_extra_pearl_beads : ℕ := 20

/-- The number of bracelets they can make -/
def total_bracelets : ℕ := 20

/-- The relation between Rose's crystal and stone beads -/
def rose_stone_to_crystal_ratio : ℕ := 2

/-- Theorem: Rose has 20 crystal beads -/
theorem rose_crystal_beads :
  ∃ (crystal_beads : ℕ),
    crystal_beads = 20 ∧
    crystal_beads * (rose_stone_to_crystal_ratio + 1) =
      total_bracelets * beads_per_bracelet -
      (nancy_metal_beads + nancy_metal_beads + nancy_extra_pearl_beads) :=
by sorry

end NUMINAMATH_CALUDE_rose_crystal_beads_l2945_294555


namespace NUMINAMATH_CALUDE_employee_earnings_theorem_l2945_294549

/-- Calculates the total earnings for an employee based on their work schedule and pay rates -/
def calculate_earnings (task_a_rate : ℚ) (task_b_rate : ℚ) (overtime_multiplier : ℚ) 
                       (commission_rate : ℚ) (task_a_hours : List ℚ) (task_b_hours : List ℚ) : ℚ :=
  let task_a_total_hours := task_a_hours.sum
  let task_b_total_hours := task_b_hours.sum
  let task_a_regular_hours := min task_a_total_hours 40
  let task_a_overtime_hours := max (task_a_total_hours - 40) 0
  let task_a_earnings := task_a_regular_hours * task_a_rate + 
                         task_a_overtime_hours * task_a_rate * overtime_multiplier
  let task_b_earnings := task_b_total_hours * task_b_rate
  let total_before_commission := task_a_earnings + task_b_earnings
  let commission := if task_b_total_hours ≥ 10 then total_before_commission * commission_rate else 0
  total_before_commission + commission

/-- Theorem stating that the employee's earnings for the given work schedule and pay rates equal $2211 -/
theorem employee_earnings_theorem :
  let task_a_rate : ℚ := 30
  let task_b_rate : ℚ := 40
  let overtime_multiplier : ℚ := 1.5
  let commission_rate : ℚ := 0.1
  let task_a_hours : List ℚ := [6, 6, 6, 12, 12]
  let task_b_hours : List ℚ := [4, 4, 4, 3, 3]
  calculate_earnings task_a_rate task_b_rate overtime_multiplier commission_rate task_a_hours task_b_hours = 2211 := by
  sorry

end NUMINAMATH_CALUDE_employee_earnings_theorem_l2945_294549


namespace NUMINAMATH_CALUDE_hyperbola_standard_equation_l2945_294594

/-- A hyperbola passing through the point (4, √3) with asymptote equation y = 1/2x has the standard equation x²/4 - y² = 1 -/
theorem hyperbola_standard_equation (x y : ℝ) :
  (∃ k : ℝ, x^2 / 4 - y^2 = k) →  -- Assuming the general form of the hyperbola equation
  (∀ x, y = 1/2 * x) →           -- Asymptote equation
  (4^2 / 4 - (Real.sqrt 3)^2 = 1) →  -- The hyperbola passes through (4, √3)
  x^2 / 4 - y^2 = 1 :=            -- Standard equation of the hyperbola
by sorry

end NUMINAMATH_CALUDE_hyperbola_standard_equation_l2945_294594


namespace NUMINAMATH_CALUDE_cards_in_new_deck_l2945_294547

/-- The number of cards in a new deck -/
def cards_per_deck : ℕ := 55

/-- The number of cards that can be torn at once -/
def cards_per_tear : ℕ := 30

/-- The number of times cards are torn per week -/
def tears_per_week : ℕ := 3

/-- The number of decks purchased -/
def decks_purchased : ℕ := 18

/-- The number of weeks the tearing can continue -/
def weeks_of_tearing : ℕ := 11

/-- Theorem stating the number of cards in a new deck -/
theorem cards_in_new_deck : 
  cards_per_deck * decks_purchased = cards_per_tear * tears_per_week * weeks_of_tearing :=
by sorry

end NUMINAMATH_CALUDE_cards_in_new_deck_l2945_294547


namespace NUMINAMATH_CALUDE_absolute_difference_of_opposite_signs_l2945_294570

theorem absolute_difference_of_opposite_signs (a b : ℝ) :
  |a| = 4 → |b| = 2 → (a * b < 0) → |a - b| = 6 := by sorry

end NUMINAMATH_CALUDE_absolute_difference_of_opposite_signs_l2945_294570

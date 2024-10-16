import Mathlib

namespace NUMINAMATH_CALUDE_track_width_l2204_220429

theorem track_width (r₁ r₂ : ℝ) (h : 2 * Real.pi * r₁ - 2 * Real.pi * r₂ = 20 * Real.pi) : 
  r₁ - r₂ = 10 := by
sorry

end NUMINAMATH_CALUDE_track_width_l2204_220429


namespace NUMINAMATH_CALUDE_min_m_value_l2204_220483

/-- The minimum value of m that satisfies the given conditions -/
theorem min_m_value (m : ℝ) (h_m : m > 0) : 
  (∀ x₁ x₂ : ℝ, 
    let y₁ := Real.exp x₁
    let y₂ := 1 + Real.log (x₂ - m)
    y₁ = y₂ → |x₂ - x₁| ≥ Real.exp 1) → 
  m ≥ Real.exp 1 - 1 :=
sorry

end NUMINAMATH_CALUDE_min_m_value_l2204_220483


namespace NUMINAMATH_CALUDE_smallest_n_with_properties_l2204_220452

def is_terminating_decimal (n : ℕ) : Prop :=
  ∃ a b : ℕ, n = 2^a * 5^b

def contains_digit (n : ℕ) (d : ℕ) : Prop :=
  ∃ k m : ℕ, n = 10 * k + d + 10 * m

def contains_distinct_digits (n : ℕ) : Prop :=
  ∃ d₁ d₂ : ℕ, d₁ ≠ d₂ ∧ contains_digit n d₁ ∧ contains_digit n d₂

theorem smallest_n_with_properties : 
  (∀ m : ℕ, m < 128 → ¬(is_terminating_decimal m ∧ contains_digit m 9 ∧ contains_distinct_digits m)) ∧
  (is_terminating_decimal 128 ∧ contains_digit 128 9 ∧ contains_distinct_digits 128) :=
sorry

end NUMINAMATH_CALUDE_smallest_n_with_properties_l2204_220452


namespace NUMINAMATH_CALUDE_outfit_combinations_l2204_220400

/-- The number of shirts -/
def num_shirts : ℕ := 4

/-- The number of pants -/
def num_pants : ℕ := 5

/-- The number of items (shirts or pants) that have a unique color -/
def num_unique_colors : ℕ := num_shirts + num_pants - 1

/-- The number of different outfits that can be created -/
def num_outfits : ℕ := num_shirts * num_pants - 1

theorem outfit_combinations : num_outfits = 19 := by sorry

end NUMINAMATH_CALUDE_outfit_combinations_l2204_220400


namespace NUMINAMATH_CALUDE_coin_pile_impossibility_l2204_220402

/-- Represents a pile of coins -/
structure CoinPile :=
  (coins : ℕ)

/-- Represents the state of all coin piles -/
structure CoinState :=
  (piles : List CoinPile)

/-- Allowed operations on coin piles -/
inductive CoinOperation
  | Combine : CoinPile → CoinPile → CoinOperation
  | Split : CoinPile → CoinOperation

/-- Applies a coin operation to a coin state -/
def applyOperation (state : CoinState) (op : CoinOperation) : CoinState :=
  sorry

/-- Checks if a coin state matches the target configuration -/
def isTargetState (state : CoinState) : Prop :=
  ∃ (p1 p2 p3 : CoinPile),
    state.piles = [p1, p2, p3] ∧
    p1.coins = 52 ∧ p2.coins = 48 ∧ p3.coins = 5

/-- The main theorem stating the impossibility of reaching the target state -/
theorem coin_pile_impossibility :
  ∀ (initial : CoinState) (ops : List CoinOperation),
    initial.piles = [CoinPile.mk 51, CoinPile.mk 49, CoinPile.mk 5] →
    ¬(isTargetState (ops.foldl applyOperation initial)) :=
  sorry

end NUMINAMATH_CALUDE_coin_pile_impossibility_l2204_220402


namespace NUMINAMATH_CALUDE_smallest_sum_of_squares_and_cubes_infinitely_many_coprime_sums_l2204_220472

-- Define a function to check if a number is the sum of two squares
def isSumOfTwoSquares (n : ℕ) : Prop :=
  ∃ a b : ℕ, a^2 + b^2 = n ∧ a > 0 ∧ b > 0

-- Define a function to check if a number is the sum of two cubes
def isSumOfTwoCubes (n : ℕ) : Prop :=
  ∃ a b : ℕ, a^3 + b^3 = n ∧ a > 0 ∧ b > 0

-- Define a function to check if two numbers are coprime
def areCoprime (a b : ℕ) : Prop :=
  Nat.gcd a b = 1

theorem smallest_sum_of_squares_and_cubes :
  (∀ n : ℕ, n > 2 ∧ n < 65 → ¬(isSumOfTwoSquares n ∧ isSumOfTwoCubes n)) ∧
  (isSumOfTwoSquares 65 ∧ isSumOfTwoCubes 65) :=
sorry

theorem infinitely_many_coprime_sums :
  ∀ k : ℕ, ∃ n : ℕ,
    (∃ a b : ℕ, n = a^2 + b^2 ∧ areCoprime a b) ∧
    (∃ c d : ℕ, n = c^3 + d^3 ∧ areCoprime c d) ∧
    n > k :=
sorry

end NUMINAMATH_CALUDE_smallest_sum_of_squares_and_cubes_infinitely_many_coprime_sums_l2204_220472


namespace NUMINAMATH_CALUDE_timothy_initial_amount_matches_purchases_l2204_220414

/-- The amount of money Timothy had initially -/
def initial_amount : ℕ := 50

/-- The cost of a single t-shirt -/
def tshirt_cost : ℕ := 8

/-- The cost of a single bag -/
def bag_cost : ℕ := 10

/-- The number of t-shirts Timothy bought -/
def tshirts_bought : ℕ := 2

/-- The number of bags Timothy bought -/
def bags_bought : ℕ := 2

/-- The cost of a set of 3 key chains -/
def keychain_set_cost : ℕ := 2

/-- The number of key chains in a set -/
def keychains_per_set : ℕ := 3

/-- The number of key chains Timothy bought -/
def keychains_bought : ℕ := 21

/-- Theorem stating that Timothy's initial amount matches his purchases -/
theorem timothy_initial_amount_matches_purchases :
  initial_amount = 
    tshirts_bought * tshirt_cost + 
    bags_bought * bag_cost + 
    (keychains_bought / keychains_per_set) * keychain_set_cost :=
by
  sorry


end NUMINAMATH_CALUDE_timothy_initial_amount_matches_purchases_l2204_220414


namespace NUMINAMATH_CALUDE_distance_after_10_hours_l2204_220478

/-- The distance between two trains after a given time -/
def distance_between_trains (speed1 speed2 time : ℝ) : ℝ :=
  (speed2 - speed1) * time

/-- Theorem: The distance between two trains after 10 hours -/
theorem distance_after_10_hours :
  distance_between_trains 10 35 10 = 250 := by
  sorry

#eval distance_between_trains 10 35 10

end NUMINAMATH_CALUDE_distance_after_10_hours_l2204_220478


namespace NUMINAMATH_CALUDE_sum_ab_over_c_squared_plus_one_le_one_l2204_220473

theorem sum_ab_over_c_squared_plus_one_le_one 
  (a b c : ℝ) 
  (ha : a > 0) 
  (hb : b > 0) 
  (hc : c > 0) 
  (sum_eq_two : a + b + c = 2) :
  (a * b) / (c^2 + 1) + (b * c) / (a^2 + 1) + (c * a) / (b^2 + 1) ≤ 1 := by
  sorry

end NUMINAMATH_CALUDE_sum_ab_over_c_squared_plus_one_le_one_l2204_220473


namespace NUMINAMATH_CALUDE_linear_regression_transformation_l2204_220481

-- Define the variables and functions
variable (a b x : ℝ)
variable (y : ℝ)
variable (μ : ℝ)
variable (c : ℝ)
variable (v : ℝ)

-- Define the conditions
def condition_y : Prop := y = a * Real.exp (b / x)
def condition_μ : Prop := μ = Real.log y
def condition_c : Prop := c = Real.log a
def condition_v : Prop := v = 1 / x

-- State the theorem
theorem linear_regression_transformation 
  (h1 : condition_y a b x y)
  (h2 : condition_μ y μ)
  (h3 : condition_c a c)
  (h4 : condition_v x v) :
  μ = c + b * v :=
by sorry

end NUMINAMATH_CALUDE_linear_regression_transformation_l2204_220481


namespace NUMINAMATH_CALUDE_bulls_win_probability_l2204_220411

/-- The probability of the Knicks winning a single game -/
def p_knicks : ℚ := 3/5

/-- The probability of the Bulls winning a single game -/
def p_bulls : ℚ := 1 - p_knicks

/-- The number of ways to choose 3 games out of 6 -/
def ways_to_choose : ℕ := 20

/-- The probability of the Bulls winning the playoff series in exactly 7 games -/
def prob_bulls_win_in_seven : ℚ :=
  ways_to_choose * p_bulls^3 * p_knicks^3 * p_bulls

theorem bulls_win_probability :
  prob_bulls_win_in_seven = 864/15625 := by sorry

end NUMINAMATH_CALUDE_bulls_win_probability_l2204_220411


namespace NUMINAMATH_CALUDE_sum_20_terms_l2204_220428

/-- An arithmetic progression with the sum of its 4th and 12th terms equal to 20 -/
structure ArithmeticProgression where
  a : ℝ  -- First term
  d : ℝ  -- Common difference
  sum_4_12 : a + 3*d + a + 11*d = 20  -- Sum of 4th and 12th terms is 20

/-- Theorem about the sum of first 20 terms of the arithmetic progression -/
theorem sum_20_terms (ap : ArithmeticProgression) :
  ∃ k : ℝ, k = 200 + 120 * ap.d ∧ 
  (∀ n : ℕ, n ≤ 20 → (n : ℝ) / 2 * (2 * ap.a + (n - 1) * ap.d) ≤ k) ∧
  (∀ ε > 0, ∃ n : ℕ, n ≤ 20 ∧ k - (n : ℝ) / 2 * (2 * ap.a + (n - 1) * ap.d) < ε) :=
by sorry


end NUMINAMATH_CALUDE_sum_20_terms_l2204_220428


namespace NUMINAMATH_CALUDE_min_third_altitude_l2204_220482

/-- Represents a scalene triangle with specific altitude properties -/
structure ScaleneTriangle where
  -- Side lengths
  a : ℝ
  b : ℝ
  c : ℝ
  -- Altitudes
  h_D : ℝ
  h_E : ℝ
  h_F : ℝ
  -- Triangle inequality
  triangle_inequality : a + b > c ∧ b + c > a ∧ c + a > b
  -- Scalene property
  scalene : a ≠ b ∧ b ≠ c ∧ c ≠ a
  -- Given altitude values
  altitude_D : h_D = 18
  altitude_E : h_E = 8
  -- Relation between sides
  side_relation : b = 2 * a
  -- Area consistency
  area_consistency : a * h_D / 2 = b * h_E / 2

/-- The minimum possible integer length of the third altitude is 17 -/
theorem min_third_altitude (t : ScaleneTriangle) : 
  ∃ (n : ℕ), n ≥ 17 ∧ t.h_F = n ∧ ∀ (m : ℕ), m < 17 → t.h_F ≠ m :=
sorry

end NUMINAMATH_CALUDE_min_third_altitude_l2204_220482


namespace NUMINAMATH_CALUDE_rosas_phone_book_calling_l2204_220470

/-- Rosa's phone book calling problem -/
theorem rosas_phone_book_calling (pages_last_week pages_total : ℝ) 
  (h1 : pages_last_week = 10.2)
  (h2 : pages_total = 18.8) :
  pages_total - pages_last_week = 8.6 := by
  sorry

end NUMINAMATH_CALUDE_rosas_phone_book_calling_l2204_220470


namespace NUMINAMATH_CALUDE_C_share_of_profit_l2204_220401

def investment_A : ℕ := 8000
def investment_B : ℕ := 4000
def investment_C : ℕ := 2000
def total_profit : ℕ := 252000

theorem C_share_of_profit :
  (investment_C : ℚ) / (investment_A + investment_B + investment_C) * total_profit = 36000 :=
by sorry

end NUMINAMATH_CALUDE_C_share_of_profit_l2204_220401


namespace NUMINAMATH_CALUDE_hash_difference_seven_four_l2204_220417

-- Define the # operation
def hash (x y : ℤ) : ℤ := 2*x*y - 3*x - y

-- Theorem statement
theorem hash_difference_seven_four : hash 7 4 - hash 4 7 = -6 := by
  sorry

end NUMINAMATH_CALUDE_hash_difference_seven_four_l2204_220417


namespace NUMINAMATH_CALUDE_symmetric_point_yoz_plane_l2204_220469

/-- A point in 3D space -/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- The yOz plane in 3D space -/
def yOzPlane : Set Point3D := {p : Point3D | p.x = 0}

/-- Symmetry with respect to the yOz plane -/
def symmetricPointYOz (p : Point3D) : Point3D :=
  { x := -p.x, y := p.y, z := p.z }

/-- Theorem: The point (-1, -2, 3) is symmetric to (1, -2, 3) with respect to the yOz plane -/
theorem symmetric_point_yoz_plane :
  let p1 : Point3D := { x := 1, y := -2, z := 3 }
  let p2 : Point3D := { x := -1, y := -2, z := 3 }
  symmetricPointYOz p1 = p2 := by
  sorry

end NUMINAMATH_CALUDE_symmetric_point_yoz_plane_l2204_220469


namespace NUMINAMATH_CALUDE_partnership_investment_timing_l2204_220430

/-- A partnership problem where three partners invest at different times --/
theorem partnership_investment_timing 
  (x : ℝ) -- A's investment
  (annual_gain : ℝ) -- Total annual gain
  (a_share : ℝ) -- A's share of the gain
  (h1 : annual_gain = 12000) -- Given annual gain
  (h2 : a_share = 4000) -- Given A's share
  (h3 : a_share / annual_gain = 1/3) -- A's share ratio
  : ∃ (m : ℝ), -- The number of months after which C invests
    (x * 12) / (x * 12 + 2*x * 6 + 3*x * (12 - m)) = 1/3 ∧ 
    m = 8 := by
  sorry

end NUMINAMATH_CALUDE_partnership_investment_timing_l2204_220430


namespace NUMINAMATH_CALUDE_tennis_match_duration_l2204_220480

def minutes_per_hour : ℕ := 60

def hours : ℕ := 11
def additional_minutes : ℕ := 5

theorem tennis_match_duration : 
  hours * minutes_per_hour + additional_minutes = 665 := by
  sorry

end NUMINAMATH_CALUDE_tennis_match_duration_l2204_220480


namespace NUMINAMATH_CALUDE_sine_product_upper_bound_sine_product_upper_bound_achievable_l2204_220496

/-- Given points A, B, and C in a coordinate plane, where A = (-8, 0), B = (8, 0), and C = (t, 6) for some real number t, the product of sines of angles CAB and CBA is at most 3/8. -/
theorem sine_product_upper_bound (t : ℝ) :
  let A : ℝ × ℝ := (-8, 0)
  let B : ℝ × ℝ := (8, 0)
  let C : ℝ × ℝ := (t, 6)
  let angle_CAB := Real.arctan ((C.2 - A.2) / (C.1 - A.1)) - Real.arctan ((B.2 - A.2) / (B.1 - A.1))
  let angle_CBA := Real.arctan ((C.2 - B.2) / (C.1 - B.1)) - Real.arctan ((A.2 - B.2) / (A.1 - B.1))
  Real.sin angle_CAB * Real.sin angle_CBA ≤ 3/8 :=
by sorry

/-- The upper bound 3/8 for the product of sines is achievable. -/
theorem sine_product_upper_bound_achievable :
  ∃ t : ℝ,
  let A : ℝ × ℝ := (-8, 0)
  let B : ℝ × ℝ := (8, 0)
  let C : ℝ × ℝ := (t, 6)
  let angle_CAB := Real.arctan ((C.2 - A.2) / (C.1 - A.1)) - Real.arctan ((B.2 - A.2) / (B.1 - A.1))
  let angle_CBA := Real.arctan ((C.2 - B.2) / (C.1 - B.1)) - Real.arctan ((A.2 - B.2) / (A.1 - B.1))
  Real.sin angle_CAB * Real.sin angle_CBA = 3/8 :=
by sorry

end NUMINAMATH_CALUDE_sine_product_upper_bound_sine_product_upper_bound_achievable_l2204_220496


namespace NUMINAMATH_CALUDE_chessboard_repaint_theorem_l2204_220450

/-- Represents a chessboard of size n × n -/
structure Chessboard (n : ℕ) where
  size : n ≥ 3

/-- Represents an L-shaped tetromino and its rotations -/
inductive Tetromino
  | L
  | RotatedL90
  | RotatedL180
  | RotatedL270

/-- Represents a move that repaints a tetromino on the chessboard -/
def Move (n : ℕ) := Fin n → Fin n → Tetromino

/-- Predicate to check if a series of moves can repaint the entire chessboard -/
def CanRepaintEntireBoard (n : ℕ) (moves : List (Move n)) : Prop :=
  sorry

/-- Main theorem: The chessboard can be entirely repainted if and only if n is even and n ≥ 4 -/
theorem chessboard_repaint_theorem (n : ℕ) (b : Chessboard n) :
  (∃ (moves : List (Move n)), CanRepaintEntireBoard n moves) ↔ (Even n ∧ n ≥ 4) :=
sorry

end NUMINAMATH_CALUDE_chessboard_repaint_theorem_l2204_220450


namespace NUMINAMATH_CALUDE_parabola_properties_l2204_220449

-- Define the parabola
def parabola (x y : ℝ) : Prop := y^2 = 4*x

-- Define the line
def line (x y : ℝ) : Prop := y = x - 4

-- Define the origin
def origin : ℝ × ℝ := (0, 0)

-- Theorem statement
theorem parabola_properties :
  -- The parabola passes through (1, 2)
  parabola 1 2 ∧
  -- If A and B are intersection points of the line and parabola
  ∀ (A B : ℝ × ℝ), 
    (parabola A.1 A.2 ∧ line A.1 A.2) →
    (parabola B.1 B.2 ∧ line B.1 B.2) →
    A ≠ B →
    -- Then OA is perpendicular to OB
    (A.1 * B.1 + A.2 * B.2 = 0) :=
by sorry

end NUMINAMATH_CALUDE_parabola_properties_l2204_220449


namespace NUMINAMATH_CALUDE_jolene_raised_180_l2204_220440

/-- Represents Jolene's fundraising activities --/
structure JoleneFundraising where
  num_babysitting_families : ℕ
  babysitting_rate : ℕ
  num_cars_washed : ℕ
  car_wash_rate : ℕ

/-- Calculates the total amount Jolene raised --/
def total_raised (j : JoleneFundraising) : ℕ :=
  j.num_babysitting_families * j.babysitting_rate + j.num_cars_washed * j.car_wash_rate

/-- Theorem stating that Jolene raised $180 --/
theorem jolene_raised_180 :
  ∃ j : JoleneFundraising,
    j.num_babysitting_families = 4 ∧
    j.babysitting_rate = 30 ∧
    j.num_cars_washed = 5 ∧
    j.car_wash_rate = 12 ∧
    total_raised j = 180 :=
  sorry

end NUMINAMATH_CALUDE_jolene_raised_180_l2204_220440


namespace NUMINAMATH_CALUDE_quadratic_inequality_l2204_220415

/-- A quadratic polynomial with nonnegative coefficients -/
structure NonnegQuadratic where
  a : ℝ
  b : ℝ
  c : ℝ
  a_nonneg : 0 ≤ a
  b_nonneg : 0 ≤ b
  c_nonneg : 0 ≤ c

/-- The value of a quadratic polynomial at a given point -/
def evalQuadratic (P : NonnegQuadratic) (x : ℝ) : ℝ :=
  P.a * x^2 + P.b * x + P.c

/-- Theorem: For any quadratic polynomial with nonnegative coefficients and any real numbers x and y,
    the inequality P(xy)^2 ≤ P(x^2)P(y^2) holds -/
theorem quadratic_inequality (P : NonnegQuadratic) (x y : ℝ) :
    (evalQuadratic P (x * y))^2 ≤ (evalQuadratic P (x^2)) * (evalQuadratic P (y^2)) := by
  sorry

end NUMINAMATH_CALUDE_quadratic_inequality_l2204_220415


namespace NUMINAMATH_CALUDE_work_to_pump_liquid_l2204_220441

/-- Work required to pump liquid from a paraboloid cauldron -/
theorem work_to_pump_liquid (R H γ : ℝ) (h_R : R > 0) (h_H : H > 0) (h_γ : γ > 0) :
  ∃ (W : ℝ), W = 240 * π * H^3 * γ / 9810 ∧ W > 0 := by
  sorry

end NUMINAMATH_CALUDE_work_to_pump_liquid_l2204_220441


namespace NUMINAMATH_CALUDE_x_power_n_plus_inverse_l2204_220462

theorem x_power_n_plus_inverse (θ : ℝ) (x : ℂ) (n : ℕ) 
  (h1 : 0 < θ) (h2 : θ < π / 2) (h3 : x + 1 / x = -2 * Real.sin θ) (h4 : n > 0) :
  x^n + 1 / x^n = -2 * Real.sin (n * θ) := by
  sorry

end NUMINAMATH_CALUDE_x_power_n_plus_inverse_l2204_220462


namespace NUMINAMATH_CALUDE_ryan_marble_distribution_l2204_220488

theorem ryan_marble_distribution (total_marbles : ℕ) (marbles_per_friend : ℕ) (num_friends : ℕ) :
  total_marbles = 72 →
  marbles_per_friend = 8 →
  total_marbles = marbles_per_friend * num_friends →
  num_friends = 9 := by
sorry

end NUMINAMATH_CALUDE_ryan_marble_distribution_l2204_220488


namespace NUMINAMATH_CALUDE_batsman_highest_score_l2204_220404

theorem batsman_highest_score 
  (total_innings : ℕ) 
  (overall_average : ℚ) 
  (score_difference : ℕ) 
  (average_excluding_extremes : ℚ) 
  (h : total_innings = 46)
  (h1 : overall_average = 61)
  (h2 : score_difference = 150)
  (h3 : average_excluding_extremes = 58) :
  ∃ (highest_score lowest_score : ℕ),
    highest_score - lowest_score = score_difference ∧
    (highest_score + lowest_score : ℚ) = 
      total_innings * overall_average - (total_innings - 2) * average_excluding_extremes ∧
    highest_score = 202 :=
by sorry

end NUMINAMATH_CALUDE_batsman_highest_score_l2204_220404


namespace NUMINAMATH_CALUDE_biased_coin_probability_l2204_220409

def probability_of_heads (p : ℝ) (k : ℕ) (n : ℕ) : ℝ :=
  (n.choose k) * p^k * (1-p)^(n-k)

theorem biased_coin_probability : 
  ∀ p : ℝ, 
  0 < p → p < 1 →
  probability_of_heads p 1 7 = probability_of_heads p 2 7 →
  probability_of_heads p 1 7 ≠ 0 →
  probability_of_heads p 4 7 = 945 / 16384 := by
sorry

end NUMINAMATH_CALUDE_biased_coin_probability_l2204_220409


namespace NUMINAMATH_CALUDE_f_not_in_second_quadrant_l2204_220422

/-- The function f(x) = x - 2 -/
def f (x : ℝ) : ℝ := x - 2

/-- A point (x, y) is in the second quadrant if x < 0 and y > 0 -/
def in_second_quadrant (x y : ℝ) : Prop := x < 0 ∧ y > 0

theorem f_not_in_second_quadrant :
  ∀ x : ℝ, ¬(in_second_quadrant x (f x)) :=
by
  sorry

end NUMINAMATH_CALUDE_f_not_in_second_quadrant_l2204_220422


namespace NUMINAMATH_CALUDE_unique_solution_l2204_220463

/-- Jessica's work hours as a function of t -/
def jessica_hours (t : ℤ) : ℤ := 3 * t - 10

/-- Jessica's hourly rate as a function of t -/
def jessica_rate (t : ℤ) : ℤ := 4 * t - 9

/-- Bob's work hours as a function of t -/
def bob_hours (t : ℤ) : ℤ := t + 12

/-- Bob's hourly rate as a function of t -/
def bob_rate (t : ℤ) : ℤ := 2 * t + 1

/-- Predicate to check if t satisfies the equation -/
def satisfies_equation (t : ℤ) : Prop :=
  jessica_hours t * jessica_rate t = bob_hours t * bob_rate t

theorem unique_solution :
  ∃! t : ℤ, t > 3 ∧ satisfies_equation t := by sorry

end NUMINAMATH_CALUDE_unique_solution_l2204_220463


namespace NUMINAMATH_CALUDE_perfect_pair_122_14762_l2204_220455

/-- Two natural numbers form a perfect pair if their sum and product are perfect squares. -/
def isPerfectPair (a b : ℕ) : Prop :=
  ∃ (x y : ℕ), a + b = x^2 ∧ a * b = y^2

/-- Theorem stating that 122 and 14762 form a perfect pair. -/
theorem perfect_pair_122_14762 : isPerfectPair 122 14762 := by
  sorry

#check perfect_pair_122_14762

end NUMINAMATH_CALUDE_perfect_pair_122_14762_l2204_220455


namespace NUMINAMATH_CALUDE_arrangement_exists_iff_even_l2204_220484

/-- Represents a cell in the n × n table -/
structure Cell where
  row : Nat
  col : Nat

/-- Represents an arrangement of numbers in the n × n table -/
def Arrangement (n : Nat) := Cell → Nat

/-- Checks if two cells share a side -/
def adjacent (c1 c2 : Cell) : Prop :=
  (c1.row = c2.row ∧ c1.col.pred = c2.col ∨ c1.col.succ = c2.col) ∨
  (c1.col = c2.col ∧ c1.row.pred = c2.row ∨ c1.row.succ = c2.row)

/-- Checks if an arrangement is valid according to the problem conditions -/
def valid_arrangement (n : Nat) (arr : Arrangement n) : Prop :=
  n > 1 ∧
  (∀ c : Cell, c.row ≤ n ∧ c.col ≤ n → arr c ≤ n^2) ∧
  (∀ c1 c2 : Cell, c1 ≠ c2 → arr c1 ≠ arr c2) ∧
  (∀ k : Nat, k < n^2 → ∃ c1 c2 : Cell, adjacent c1 c2 ∧ arr c1 = k ∧ arr c2 = k + 1) ∧
  (∀ c1 c2 : Cell, arr c1 % n = arr c2 % n → c1.row ≠ c2.row ∧ c1.col ≠ c2.col)

theorem arrangement_exists_iff_even (n : Nat) :
  (∃ arr : Arrangement n, valid_arrangement n arr) ↔ Even n :=
sorry

end NUMINAMATH_CALUDE_arrangement_exists_iff_even_l2204_220484


namespace NUMINAMATH_CALUDE_polynomial_property_l2204_220477

def P (a b c : ℝ) (x : ℝ) : ℝ := 2*x^3 + a*x^2 + b*x + c

theorem polynomial_property (a b c : ℝ) :
  P a b c 0 = 8 →
  (∃ m : ℝ, m = (-(c / 2)) ∧ 
             m = -((a / 2) / 3) ∧ 
             m = 2 + a + b + c) →
  b = -38 := by sorry

end NUMINAMATH_CALUDE_polynomial_property_l2204_220477


namespace NUMINAMATH_CALUDE_z_in_second_quadrant_l2204_220476

-- Define the complex number z
def z : ℂ := -1 + 3 * Complex.I

-- Theorem stating that z is in the second quadrant
theorem z_in_second_quadrant : 
  z.re < 0 ∧ z.im > 0 :=
sorry

end NUMINAMATH_CALUDE_z_in_second_quadrant_l2204_220476


namespace NUMINAMATH_CALUDE_star_polygon_interior_angles_sum_l2204_220468

/-- A star polygon with n angles -/
structure StarPolygon where
  n : ℕ
  h_n : n ≥ 5

/-- The sum of interior angles of a star polygon -/
def sum_interior_angles (sp : StarPolygon) : ℝ :=
  180 * (sp.n - 4)

/-- Theorem: The sum of interior angles of a star polygon is 180° * (n - 4) -/
theorem star_polygon_interior_angles_sum (sp : StarPolygon) :
  sum_interior_angles sp = 180 * (sp.n - 4) := by
  sorry

end NUMINAMATH_CALUDE_star_polygon_interior_angles_sum_l2204_220468


namespace NUMINAMATH_CALUDE_doubled_average_l2204_220451

theorem doubled_average (n : ℕ) (initial_avg : ℝ) (h1 : n = 12) (h2 : initial_avg = 36) :
  let total := n * initial_avg
  let new_total := 2 * total
  let new_avg := new_total / n
  new_avg = 72 := by sorry

end NUMINAMATH_CALUDE_doubled_average_l2204_220451


namespace NUMINAMATH_CALUDE_unique_common_root_existence_l2204_220406

theorem unique_common_root_existence :
  ∃! m : ℝ, ∃! x : ℝ, 
    (x^2 + m*x + 2 = 0 ∧ x^2 + 2*x + m = 0) ∧
    (∀ y : ℝ, y^2 + m*y + 2 = 0 ∧ y^2 + 2*y + m = 0 → y = x) ∧
    m = -3 ∧ x = 1 :=
by sorry

end NUMINAMATH_CALUDE_unique_common_root_existence_l2204_220406


namespace NUMINAMATH_CALUDE_line_plane_parallelism_l2204_220458

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the parallelism relation between lines and planes
variable (parallel_line_plane : Line → Plane → Prop)

-- Define the parallelism relation between planes
variable (parallel_plane : Plane → Plane → Prop)

-- Define the intersection relation between lines
variable (intersect : Line → Line → Prop)

-- Define the relation for a line being outside a plane
variable (outside : Line → Plane → Prop)

-- Theorem statement
theorem line_plane_parallelism 
  (m n : Line) (α β : Plane) :
  intersect m n ∧ 
  outside m α ∧ outside m β ∧ 
  outside n α ∧ outside n β ∧
  parallel_line_plane m α ∧ parallel_line_plane m β ∧ 
  parallel_line_plane n α ∧ parallel_line_plane n β →
  parallel_plane α β :=
sorry

end NUMINAMATH_CALUDE_line_plane_parallelism_l2204_220458


namespace NUMINAMATH_CALUDE_interior_angle_of_17_sided_polygon_l2204_220479

theorem interior_angle_of_17_sided_polygon (S : ℝ) (x : ℝ) : 
  S = (17 - 2) * 180 ∧ S - x = 2570 → x = 130 := by
  sorry

end NUMINAMATH_CALUDE_interior_angle_of_17_sided_polygon_l2204_220479


namespace NUMINAMATH_CALUDE_pentagon_area_greater_than_third_square_l2204_220495

theorem pentagon_area_greater_than_third_square (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  a^2 + (a*b)/4 + (Real.sqrt 3/4)*b^2 > ((a+b)^2)/3 := by
  sorry

end NUMINAMATH_CALUDE_pentagon_area_greater_than_third_square_l2204_220495


namespace NUMINAMATH_CALUDE_fifth_term_of_geometric_sequence_l2204_220489

/-- A geometric sequence is a sequence where each term after the first is found by multiplying the previous term by a fixed, non-zero number called the common ratio. -/
def GeometricSequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, r ≠ 0 ∧ ∀ n : ℕ, a (n + 1) = a n * r

theorem fifth_term_of_geometric_sequence
  (a : ℕ → ℝ)
  (h_positive : ∀ n, a n > 0)
  (h_geometric : GeometricSequence a)
  (h_third_term : a 3 = 9)
  (h_seventh_term : a 7 = 1) :
  a 5 = 3 := by
sorry

end NUMINAMATH_CALUDE_fifth_term_of_geometric_sequence_l2204_220489


namespace NUMINAMATH_CALUDE_coin_and_die_prob_l2204_220439

/-- A fair coin -/
def FairCoin : Type := Bool

/-- A regular eight-sided die -/
def EightSidedDie : Type := Fin 8

/-- The event of getting heads on a fair coin -/
def headsEvent (c : FairCoin) : Prop := c = true

/-- The event of getting an even number on an eight-sided die -/
def evenDieEvent (d : EightSidedDie) : Prop := d.val % 2 = 0

/-- The probability of an event on a fair coin -/
axiom probCoin (event : FairCoin → Prop) : ℚ

/-- The probability of an event on an eight-sided die -/
axiom probDie (event : EightSidedDie → Prop) : ℚ

/-- The probability of getting heads on a fair coin -/
axiom prob_heads : probCoin headsEvent = 1/2

/-- The probability of getting an even number on an eight-sided die -/
axiom prob_even_die : probDie evenDieEvent = 1/2

/-- The main theorem: The probability of getting heads on a fair coin and an even number
    on a regular eight-sided die when flipped and rolled once is 1/4 -/
theorem coin_and_die_prob :
  probCoin headsEvent * probDie evenDieEvent = 1/4 := by sorry

end NUMINAMATH_CALUDE_coin_and_die_prob_l2204_220439


namespace NUMINAMATH_CALUDE_turtle_difference_l2204_220426

/-- The number of turtles Martha received -/
def martha_turtles : ℕ := 40

/-- The total number of turtles Marion and Martha received together -/
def total_turtles : ℕ := 100

/-- The number of turtles Marion received -/
def marion_turtles : ℕ := total_turtles - martha_turtles

/-- Marion received more turtles than Martha -/
axiom marion_more : marion_turtles > martha_turtles

theorem turtle_difference : marion_turtles - martha_turtles = 20 := by
  sorry

end NUMINAMATH_CALUDE_turtle_difference_l2204_220426


namespace NUMINAMATH_CALUDE_lunch_break_duration_l2204_220436

/-- Represents the painting scenario with Paul and his assistants --/
structure PaintingScenario where
  paul_rate : ℝ
  assistants_rate : ℝ
  lunch_break : ℝ

/-- Checks if the given scenario satisfies all conditions --/
def satisfies_conditions (s : PaintingScenario) : Prop :=
  -- Monday's condition
  (8 - s.lunch_break) * (s.paul_rate + s.assistants_rate) = 0.6 ∧
  -- Tuesday's condition
  (6 - s.lunch_break) * s.assistants_rate = 0.3 ∧
  -- Wednesday's condition
  (4 - s.lunch_break) * s.paul_rate = 0.1

/-- Theorem stating that the lunch break duration is 60 minutes --/
theorem lunch_break_duration :
  ∃ (s : PaintingScenario), s.lunch_break = 1 ∧ satisfies_conditions s :=
sorry

end NUMINAMATH_CALUDE_lunch_break_duration_l2204_220436


namespace NUMINAMATH_CALUDE_value_of_expression_l2204_220431

theorem value_of_expression (x : ℝ) (h : x = -2) : (3*x - 4)^2 = 100 := by
  sorry

end NUMINAMATH_CALUDE_value_of_expression_l2204_220431


namespace NUMINAMATH_CALUDE_tangent_line_at_one_f_lower_bound_l2204_220425

noncomputable section

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := x * Real.exp (x - a) - Real.log x - Real.log a

-- Define the derivative of f
def f_prime (a : ℝ) (x : ℝ) : ℝ := (x + 1) * Real.exp (x - a) - 1 / x

theorem tangent_line_at_one (a : ℝ) (ha : a > 0) :
  f_prime a 1 = 1 → ∃ m b : ℝ, m = 1 ∧ b = 0 ∧ ∀ x : ℝ, f a x = m * x + b := by sorry

theorem f_lower_bound (a : ℝ) (ha : 0 < a) (ha2 : a < (Real.sqrt 5 - 1) / 2) :
  ∀ x : ℝ, x > 0 → f a x > a / (a + 1) := by sorry

end NUMINAMATH_CALUDE_tangent_line_at_one_f_lower_bound_l2204_220425


namespace NUMINAMATH_CALUDE_division_problem_l2204_220454

theorem division_problem (divisor quotient remainder : ℕ) : 
  divisor = 10 * quotient → 
  divisor = 5 * remainder → 
  remainder = 46 → 
  divisor * quotient + remainder = 5336 := by
sorry

end NUMINAMATH_CALUDE_division_problem_l2204_220454


namespace NUMINAMATH_CALUDE_joes_speed_to_petes_speed_ratio_l2204_220418

/-- Prove that the ratio of Joe's speed to Pete's speed is 2:1 -/
theorem joes_speed_to_petes_speed_ratio (
  time : ℝ)
  (total_distance : ℝ)
  (joes_speed : ℝ)
  (h1 : time = 40)
  (h2 : total_distance = 16)
  (h3 : joes_speed = 0.266666666667)
  : joes_speed / ((total_distance - joes_speed * time) / time) = 2 := by
  sorry

end NUMINAMATH_CALUDE_joes_speed_to_petes_speed_ratio_l2204_220418


namespace NUMINAMATH_CALUDE_quadratic_one_solution_l2204_220408

theorem quadratic_one_solution (m : ℚ) : 
  (∃! x, 3 * x^2 - 7 * x + m = 0) → m = 49 / 12 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_one_solution_l2204_220408


namespace NUMINAMATH_CALUDE_square_divisibility_l2204_220465

theorem square_divisibility (n : ℕ+) (h : ∀ m : ℕ+, m ∣ n → m ≤ 12) : 144 ∣ n^2 := by
  sorry

end NUMINAMATH_CALUDE_square_divisibility_l2204_220465


namespace NUMINAMATH_CALUDE_compare_b_and_d_l2204_220460

theorem compare_b_and_d (a b c d : ℝ) 
  (ha : a > 0) (hb : b > 0) (hc : c > 0)
  (hab : a = b * 1.02)
  (hac : c = a * 0.99)
  (hcd : d = c * 0.99) : 
  b > d := by
sorry

end NUMINAMATH_CALUDE_compare_b_and_d_l2204_220460


namespace NUMINAMATH_CALUDE_pizza_tip_percentage_l2204_220446

/-- Calculates the tip percentage for Harry's pizza order --/
theorem pizza_tip_percentage
  (large_pizza_cost : ℝ)
  (topping_cost : ℝ)
  (num_pizzas : ℕ)
  (toppings_per_pizza : ℕ)
  (total_cost_with_tip : ℝ)
  (h1 : large_pizza_cost = 14)
  (h2 : topping_cost = 2)
  (h3 : num_pizzas = 2)
  (h4 : toppings_per_pizza = 3)
  (h5 : total_cost_with_tip = 50)
  : (total_cost_with_tip - (num_pizzas * large_pizza_cost + num_pizzas * toppings_per_pizza * topping_cost)) /
    (num_pizzas * large_pizza_cost + num_pizzas * toppings_per_pizza * topping_cost) = 0.25 := by
  sorry


end NUMINAMATH_CALUDE_pizza_tip_percentage_l2204_220446


namespace NUMINAMATH_CALUDE_prob_black_fourth_draw_l2204_220467

structure Box where
  red_balls : ℕ
  black_balls : ℕ

def initial_box : Box := { red_balls := 3, black_balls := 3 }

def total_balls (b : Box) : ℕ := b.red_balls + b.black_balls

def prob_black_first_draw (b : Box) : ℚ :=
  b.black_balls / (total_balls b)

theorem prob_black_fourth_draw (b : Box) :
  prob_black_first_draw b = 1/2 →
  (∃ (p : ℚ), p = prob_black_first_draw b ∧ p = 1/2) :=
by sorry

end NUMINAMATH_CALUDE_prob_black_fourth_draw_l2204_220467


namespace NUMINAMATH_CALUDE_magnitude_of_sum_l2204_220423

/-- Given two vectors a and b in ℝ², prove that the magnitude of a + 3b is 5√5 when a is parallel to b -/
theorem magnitude_of_sum (a b : ℝ × ℝ) (h_parallel : ∃ (k : ℝ), b = k • a) : 
  a.1 = 1 → a.2 = 2 → b.1 = -2 → 
  ‖(a.1 + 3 * b.1, a.2 + 3 * b.2)‖ = 5 * Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_magnitude_of_sum_l2204_220423


namespace NUMINAMATH_CALUDE_test_scores_l2204_220492

/-- Given a test with 50 questions, each worth 2 marks, prove that the total combined score
    for three students (Meghan, Jose, and Alisson) is 210 marks, given the following conditions:
    - Meghan scored 20 marks less than Jose
    - Jose scored 40 more marks than Alisson
    - Jose got 5 questions wrong -/
theorem test_scores (total_questions : Nat) (marks_per_question : Nat)
    (meghan_jose_diff : Nat) (jose_alisson_diff : Nat) (jose_wrong : Nat) :
  total_questions = 50 →
  marks_per_question = 2 →
  meghan_jose_diff = 20 →
  jose_alisson_diff = 40 →
  jose_wrong = 5 →
  ∃ (meghan_score jose_score alisson_score : Nat),
    meghan_score + jose_score + alisson_score = 210 :=
by sorry


end NUMINAMATH_CALUDE_test_scores_l2204_220492


namespace NUMINAMATH_CALUDE_sequence_property_l2204_220497

def is_arithmetic_progression (a b c d : ℝ) : Prop :=
  b - a = c - b ∧ c - b = d - c

def is_geometric_progression (a b c d : ℝ) : Prop :=
  b / a = c / b ∧ c / b = d / c

def has_common_difference (a b c d : ℝ) (diff : ℝ) : Prop :=
  b - a = diff ∧ c - b = diff ∧ d - c = diff

theorem sequence_property (a₁ a₂ a₃ a₄ a₅ a₆ a₇ a₈ : ℝ) :
  a₁ < a₂ ∧ a₂ < a₃ ∧ a₃ < a₄ ∧ a₄ < a₅ ∧ a₅ < a₆ ∧ a₆ < a₇ ∧ a₇ < a₈ →
  ((has_common_difference a₁ a₂ a₃ a₄ 4 ∧ has_common_difference a₅ a₆ a₇ a₈ 36) ∨
   (has_common_difference a₂ a₃ a₄ a₅ 4 ∧ has_common_difference a₅ a₆ a₇ a₈ 36) ∨
   (has_common_difference a₁ a₂ a₃ a₄ 4 ∧ has_common_difference a₄ a₅ a₆ a₇ 36) ∨
   (has_common_difference a₂ a₃ a₄ a₅ 4 ∧ has_common_difference a₄ a₅ a₆ a₇ 36) ∨
   (has_common_difference a₁ a₂ a₃ a₄ 36 ∧ has_common_difference a₅ a₆ a₇ a₈ 4)) →
  (is_geometric_progression a₂ a₃ a₄ a₅ ∨ is_geometric_progression a₃ a₄ a₅ a₆ ∨
   is_geometric_progression a₄ a₅ a₆ a₇) →
  a₈ = 126 ∨ a₈ = 6 :=
by sorry

end NUMINAMATH_CALUDE_sequence_property_l2204_220497


namespace NUMINAMATH_CALUDE_coefficient_of_a_is_one_l2204_220403

-- Define a monomial type
def Monomial := ℚ → ℕ → ℚ

-- Define the coefficient of a monomial
def coefficient (m : Monomial) : ℚ := m 1 0

-- Define the monomial 'a'
def a : Monomial := fun c n => if n = 1 then 1 else 0

-- Theorem statement
theorem coefficient_of_a_is_one : coefficient a = 1 := by
  sorry

end NUMINAMATH_CALUDE_coefficient_of_a_is_one_l2204_220403


namespace NUMINAMATH_CALUDE_angela_insects_l2204_220499

theorem angela_insects (dean_insects : ℕ) (jacob_insects : ℕ) (angela_insects : ℕ) (alex_insects : ℕ)
  (h1 : dean_insects = 30)
  (h2 : jacob_insects = 5 * dean_insects)
  (h3 : angela_insects = jacob_insects / 2)
  (h4 : alex_insects = 3 * dean_insects)
  (h5 : alex_insects = angela_insects - 10) :
  angela_insects = 75 := by
sorry

end NUMINAMATH_CALUDE_angela_insects_l2204_220499


namespace NUMINAMATH_CALUDE_sum_after_decrease_l2204_220413

theorem sum_after_decrease (a b : ℤ) : 
  a + b = 100 → (a - 48) + b = 52 := by
  sorry

end NUMINAMATH_CALUDE_sum_after_decrease_l2204_220413


namespace NUMINAMATH_CALUDE_product_real_implies_a_equals_two_l2204_220475

theorem product_real_implies_a_equals_two (a : ℝ) : 
  let z₁ : ℂ := 2 + Complex.I
  let z₂ : ℂ := a - Complex.I
  (z₁ * z₂).im = 0 → a = 2 := by
sorry

end NUMINAMATH_CALUDE_product_real_implies_a_equals_two_l2204_220475


namespace NUMINAMATH_CALUDE_beaus_age_proof_l2204_220444

/-- Represents Beau's age today -/
def beaus_age_today : ℕ := 42

/-- Represents the age of Beau's sons today -/
def sons_age_today : ℕ := 16

/-- The number of Beau's sons (triplets) -/
def number_of_sons : ℕ := 3

/-- The number of years ago when the sum of sons' ages equaled Beau's age -/
def years_ago : ℕ := 3

theorem beaus_age_proof :
  (sons_age_today - years_ago) * number_of_sons + years_ago = beaus_age_today :=
by sorry

end NUMINAMATH_CALUDE_beaus_age_proof_l2204_220444


namespace NUMINAMATH_CALUDE_quadratic_general_form_l2204_220445

/-- Given a quadratic equation x² = 3x + 1, its general form is x² - 3x - 1 = 0 -/
theorem quadratic_general_form :
  (fun x : ℝ => x^2) = (fun x : ℝ => 3*x + 1) →
  (fun x : ℝ => x^2 - 3*x - 1) = (fun x : ℝ => 0) := by
sorry

end NUMINAMATH_CALUDE_quadratic_general_form_l2204_220445


namespace NUMINAMATH_CALUDE_rotation_result_l2204_220461

/-- Given a point A(3, -4) rotated counterclockwise by π/2 around the origin,
    the resulting point B has a y-coordinate of 3. -/
theorem rotation_result : ∃ (B : ℝ × ℝ), 
  let A : ℝ × ℝ := (3, -4)
  let angle : ℝ := π / 2
  B.1 = A.1 * Real.cos angle - A.2 * Real.sin angle ∧
  B.2 = A.1 * Real.sin angle + A.2 * Real.cos angle ∧
  B.2 = 3 := by
  sorry

end NUMINAMATH_CALUDE_rotation_result_l2204_220461


namespace NUMINAMATH_CALUDE_parabola_hyperbola_intersection_l2204_220494

/-- Given a parabola and a hyperbola with specific properties, prove that the parameter 'a' of the hyperbola equals 1/4. -/
theorem parabola_hyperbola_intersection (p : ℝ) (m : ℝ) (a : ℝ) : 
  p > 0 → -- p is positive
  m^2 = 2*p -- point (1,m) is on the parabola y^2 = 2px
  → (1 - p/2)^2 + m^2 = 5^2 -- distance from (1,m) to focus (p/2, 0) is 5
  → ∃ (k : ℝ), k^2 * a = 1 ∧ k * m = 2 -- asymptote y = kx is perpendicular to AM (slope of AM is m/2)
  → a = 1/4 := by sorry

end NUMINAMATH_CALUDE_parabola_hyperbola_intersection_l2204_220494


namespace NUMINAMATH_CALUDE_intersection_equals_closed_ray_l2204_220491

-- Define the universal set U as ℝ
def U := ℝ

-- Define set A
def A : Set ℝ := {x : ℝ | ∃ y : ℝ, y = Real.log (x - 1) / Real.log 10}

-- Define set B
def B : Set ℝ := {y : ℝ | ∃ x : ℝ, y = Real.sqrt (x^2 + 2*x + 5)}

-- Define the intersection of A and B
def A_intersect_B : Set ℝ := A ∩ B

-- Theorem statement
theorem intersection_equals_closed_ray : 
  A_intersect_B = {x : ℝ | x ≥ 2} := by sorry

end NUMINAMATH_CALUDE_intersection_equals_closed_ray_l2204_220491


namespace NUMINAMATH_CALUDE_find_divisor_l2204_220466

theorem find_divisor (dividend quotient remainder : ℕ) 
  (h1 : dividend = 12401)
  (h2 : quotient = 76)
  (h3 : remainder = 13)
  (h4 : dividend = quotient * 163 + remainder) :
  163 = dividend / quotient :=
by sorry

end NUMINAMATH_CALUDE_find_divisor_l2204_220466


namespace NUMINAMATH_CALUDE_rectangle_area_l2204_220410

/-- The area of a rectangle bounded by y = a, y = -b, x = -c, and x = 2d, 
    where a, b, c, and d are positive numbers. -/
theorem rectangle_area (a b c d : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (hd : 0 < d) :
  (a + b) * (2 * d + c) = 2 * a * d + a * c + 2 * b * d + b * c := by
  sorry

#check rectangle_area

end NUMINAMATH_CALUDE_rectangle_area_l2204_220410


namespace NUMINAMATH_CALUDE_quadratic_equation_coefficients_l2204_220447

theorem quadratic_equation_coefficients :
  ∃ (a b c : ℝ), 
    (∀ x, 3 * x * (x - 1) = 2 * (x + 2) + 8 ↔ a * x^2 + b * x + c = 0) ∧
    a = 3 ∧ b = -5 ∧ c = -12 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_equation_coefficients_l2204_220447


namespace NUMINAMATH_CALUDE_max_candy_for_one_student_l2204_220485

/-- Calculates the maximum number of candy pieces one student can take -/
def maxCandyForOneStudent (totalStudents totalCandy minPerStudent : ℕ) : ℕ :=
  totalCandy - (totalStudents - 1) * minPerStudent

/-- Theorem: Given 40 students, 200 pieces of candy, and a minimum of 2 pieces per student,
    the maximum number of pieces one student can take is 122 -/
theorem max_candy_for_one_student :
  maxCandyForOneStudent 40 200 2 = 122 := by
  sorry

#eval maxCandyForOneStudent 40 200 2

end NUMINAMATH_CALUDE_max_candy_for_one_student_l2204_220485


namespace NUMINAMATH_CALUDE_euclidean_continued_fraction_connection_l2204_220443

/-- Euclidean algorithm steps -/
def euclidean_steps (m n : ℕ) : List (ℕ × ℕ) :=
  sorry

/-- Continued fraction representation -/
def continued_fraction (as : List ℕ) : ℚ :=
  sorry

/-- Theorem connecting Euclidean algorithm and continued fractions -/
theorem euclidean_continued_fraction_connection (m n : ℕ) (h : m < n) :
  let steps := euclidean_steps m n
  let as := steps.map Prod.fst
  ∀ k, k ≤ steps.length →
    continued_fraction (as.drop k) =
      (steps.get! k).snd / (steps.get! (k - 1)).snd :=
by sorry

end NUMINAMATH_CALUDE_euclidean_continued_fraction_connection_l2204_220443


namespace NUMINAMATH_CALUDE_cos_pi_half_plus_alpha_l2204_220416

theorem cos_pi_half_plus_alpha (α : Real) : 
  (∃ P : Real × Real, P.1 = -4/5 ∧ P.2 = 3/5 ∧ P.1^2 + P.2^2 = 1 ∧ 
   P.1 = Real.cos α ∧ P.2 = Real.sin α) → 
  Real.cos (π/2 + α) = -3/5 := by
sorry

end NUMINAMATH_CALUDE_cos_pi_half_plus_alpha_l2204_220416


namespace NUMINAMATH_CALUDE_circle_diameter_l2204_220464

/-- Given a circle with area M and circumference N, if M/N = 15, then the diameter is 60 -/
theorem circle_diameter (M N : ℝ) (h1 : M > 0) (h2 : N > 0) (h3 : M / N = 15) :
  let r := N / (2 * Real.pi)
  let d := 2 * r
  d = 60 := by sorry

end NUMINAMATH_CALUDE_circle_diameter_l2204_220464


namespace NUMINAMATH_CALUDE_sticker_probability_l2204_220437

def total_stickers : ℕ := 18
def selected_stickers : ℕ := 10
def missing_stickers : ℕ := 6

theorem sticker_probability :
  (Nat.choose missing_stickers missing_stickers * Nat.choose (total_stickers - missing_stickers) (selected_stickers - missing_stickers)) / 
  Nat.choose total_stickers selected_stickers = 5 / 442 := by
  sorry

end NUMINAMATH_CALUDE_sticker_probability_l2204_220437


namespace NUMINAMATH_CALUDE_min_value_of_reciprocal_sum_l2204_220471

/-- Given a line ax - by + 2 = 0 (a > 0, b > 0) intercepted by the circle x^2 + y^2 + 2x - 4y + 1 = 0
    with a chord length of 4, prove that the minimum value of 1/a + 1/b is 3/2 + √2 -/
theorem min_value_of_reciprocal_sum (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  (∃ (x y : ℝ), a * x - b * y + 2 = 0 ∧ x^2 + y^2 + 2*x - 4*y + 1 = 0) →
  (∃ (x₁ y₁ x₂ y₂ : ℝ), 
    a * x₁ - b * y₁ + 2 = 0 ∧ x₁^2 + y₁^2 + 2*x₁ - 4*y₁ + 1 = 0 ∧
    a * x₂ - b * y₂ + 2 = 0 ∧ x₂^2 + y₂^2 + 2*x₂ - 4*y₂ + 1 = 0 ∧
    (x₁ - x₂)^2 + (y₁ - y₂)^2 = 16) →
  (1 / a + 1 / b) ≥ 3/2 + Real.sqrt 2 :=
sorry

end NUMINAMATH_CALUDE_min_value_of_reciprocal_sum_l2204_220471


namespace NUMINAMATH_CALUDE_expression_equals_seventeen_l2204_220427

theorem expression_equals_seventeen : 3 + 4 * 5 - 6 = 17 := by
  sorry

end NUMINAMATH_CALUDE_expression_equals_seventeen_l2204_220427


namespace NUMINAMATH_CALUDE_farmer_tomatoes_l2204_220420

/-- Proves that if a farmer has 479 tomatoes and picks 364 of them, he will have 115 tomatoes left. -/
theorem farmer_tomatoes (initial : ℕ) (picked : ℕ) (remaining : ℕ) : 
  initial = 479 → picked = 364 → remaining = initial - picked → remaining = 115 :=
by sorry

end NUMINAMATH_CALUDE_farmer_tomatoes_l2204_220420


namespace NUMINAMATH_CALUDE_number_of_skirts_l2204_220412

theorem number_of_skirts (total_ways : ℕ) (num_pants : ℕ) : 
  total_ways = 7 → num_pants = 4 → total_ways - num_pants = 3 := by
  sorry

end NUMINAMATH_CALUDE_number_of_skirts_l2204_220412


namespace NUMINAMATH_CALUDE_leftover_pie_share_l2204_220498

theorem leftover_pie_share (total_leftover : ℚ) (num_people : ℕ) : 
  total_leftover = 6/7 ∧ num_people = 3 → 
  total_leftover / num_people = 2/7 := by
  sorry

end NUMINAMATH_CALUDE_leftover_pie_share_l2204_220498


namespace NUMINAMATH_CALUDE_green_percentage_approx_l2204_220407

/-- Represents the count of people preferring each color --/
structure ColorPreferences where
  red : ℕ
  blue : ℕ
  green : ℕ
  yellow : ℕ
  purple : ℕ
  orange : ℕ

/-- Calculates the percentage of people who preferred green --/
def greenPercentage (prefs : ColorPreferences) : ℚ :=
  (prefs.green : ℚ) / (prefs.red + prefs.blue + prefs.green + prefs.yellow + prefs.purple + prefs.orange) * 100

/-- Theorem stating that the percentage of people who preferred green is approximately 16.67% --/
theorem green_percentage_approx (prefs : ColorPreferences)
  (h1 : prefs.red = 70)
  (h2 : prefs.blue = 80)
  (h3 : prefs.green = 50)
  (h4 : prefs.yellow = 40)
  (h5 : prefs.purple = 30)
  (h6 : prefs.orange = 30) :
  ∃ ε > 0, |greenPercentage prefs - 50/3| < ε ∧ ε < 1/100 := by
  sorry

end NUMINAMATH_CALUDE_green_percentage_approx_l2204_220407


namespace NUMINAMATH_CALUDE_samantha_route_count_l2204_220421

/-- Represents the number of ways to arrange k items out of n items -/
def binomial (n k : ℕ) : ℕ := sorry

/-- The number of routes from Samantha's house to the southwest corner of City Park -/
def routes_to_park : ℕ := binomial 4 1

/-- The number of routes through City Park -/
def routes_through_park : ℕ := 1

/-- The number of routes from the northeast corner of City Park to school -/
def routes_to_school : ℕ := binomial 6 3

/-- The total number of possible routes Samantha can take -/
def total_routes : ℕ := routes_to_park * routes_through_park * routes_to_school

theorem samantha_route_count : total_routes = 80 := by sorry

end NUMINAMATH_CALUDE_samantha_route_count_l2204_220421


namespace NUMINAMATH_CALUDE_sharon_chris_angle_l2204_220490

/-- Represents a point on Earth's surface -/
structure EarthPoint where
  latitude : Real
  longitude : Real

/-- Calculates the central angle between two points on a spherical Earth -/
def centralAngle (p1 p2 : EarthPoint) : Real :=
  sorry

theorem sharon_chris_angle :
  let sharon : EarthPoint := { latitude := 0, longitude := 112 }
  let chris : EarthPoint := { latitude := 60, longitude := 10 }
  centralAngle sharon chris = 102 := by
  sorry

end NUMINAMATH_CALUDE_sharon_chris_angle_l2204_220490


namespace NUMINAMATH_CALUDE_f_decreasing_f_odd_implies_m_zero_l2204_220456

/-- The function f(x) = -2x + m -/
def f (m : ℝ) : ℝ → ℝ := fun x ↦ -2 * x + m

theorem f_decreasing (m : ℝ) : 
  ∀ x₁ x₂ : ℝ, x₁ < x₂ → f m x₁ > f m x₂ := by sorry

theorem f_odd_implies_m_zero (m : ℝ) : 
  (∀ x : ℝ, f m (-x) = -(f m x)) → m = 0 := by sorry

end NUMINAMATH_CALUDE_f_decreasing_f_odd_implies_m_zero_l2204_220456


namespace NUMINAMATH_CALUDE_quadratic_inequality_range_l2204_220432

theorem quadratic_inequality_range (k : ℝ) : 
  (∀ x : ℝ, x^2 - k*x + 1 > 0) ↔ -2 < k ∧ k < 2 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_inequality_range_l2204_220432


namespace NUMINAMATH_CALUDE_students_playing_soccer_l2204_220424

-- Define the total number of students
def total_students : ℕ := 450

-- Define the number of boys
def boys : ℕ := 320

-- Define the percentage of boys playing soccer
def boys_soccer_percentage : ℚ := 86 / 100

-- Define the number of girls not playing soccer
def girls_not_soccer : ℕ := 95

-- Theorem to prove
theorem students_playing_soccer : 
  ∃ (soccer_players : ℕ), 
    soccer_players = 250 ∧ 
    soccer_players ≤ total_students ∧
    (total_students - boys) - girls_not_soccer = 
      (1 - boys_soccer_percentage) * soccer_players :=
sorry

end NUMINAMATH_CALUDE_students_playing_soccer_l2204_220424


namespace NUMINAMATH_CALUDE_reciprocal_power_l2204_220405

theorem reciprocal_power (a : ℝ) (h : a⁻¹ = -1) : a^2023 = -1 := by
  sorry

end NUMINAMATH_CALUDE_reciprocal_power_l2204_220405


namespace NUMINAMATH_CALUDE_remainder_divisibility_l2204_220434

theorem remainder_divisibility (n : ℕ) : 
  (∃ k : ℕ, n = 3 * k + 2) ∧ 
  (∃ m : ℕ, k = 4 * m + 3) → 
  n % 6 = 5 := by
sorry

end NUMINAMATH_CALUDE_remainder_divisibility_l2204_220434


namespace NUMINAMATH_CALUDE_max_f_value_l2204_220474

theorem max_f_value (a b c d e f : ℝ) 
  (sum_condition : a + b + c + d + e + f = 10)
  (square_sum_condition : (a - 1)^2 + (b - 1)^2 + (c - 1)^2 + (d - 1)^2 + (e - 1)^2 + (f - 1)^2 = 6) :
  f ≤ 2 :=
by sorry

end NUMINAMATH_CALUDE_max_f_value_l2204_220474


namespace NUMINAMATH_CALUDE_screen_paper_difference_l2204_220487

/-- The perimeter of a square-shaped piece of paper is shorter than the height of a computer screen. 
    The height of the screen is 100 cm, and the side of the square paper is 20 cm. 
    This theorem proves that the difference between the screen height and the paper perimeter is 20 cm. -/
theorem screen_paper_difference (screen_height paper_side : ℝ) 
  (h1 : screen_height = 100)
  (h2 : paper_side = 20)
  (h3 : 4 * paper_side < screen_height) : 
  screen_height - 4 * paper_side = 20 := by
  sorry

end NUMINAMATH_CALUDE_screen_paper_difference_l2204_220487


namespace NUMINAMATH_CALUDE_ellipse_axis_endpoint_distance_l2204_220438

/-- The distance between an endpoint of the major axis and an endpoint of the minor axis of the ellipse 4(x+2)^2 + 16y^2 = 64 is 2√5. -/
theorem ellipse_axis_endpoint_distance : 
  ∃ (C D : ℝ × ℝ),
    (∀ (x y : ℝ), 4 * (x + 2)^2 + 16 * y^2 = 64 → 
      ((x = C.1 ∧ y = C.2) ∨ (x = -C.1 ∧ y = -C.2)) ∨ 
      ((x = D.1 ∧ y = D.2) ∨ (x = -D.1 ∧ y = -D.2))) →
    Real.sqrt ((C.1 - D.1)^2 + (C.2 - D.2)^2) = 2 * Real.sqrt 5 :=
by sorry

end NUMINAMATH_CALUDE_ellipse_axis_endpoint_distance_l2204_220438


namespace NUMINAMATH_CALUDE_cos_difference_formula_l2204_220419

theorem cos_difference_formula (α β : ℝ) 
  (h1 : Real.sin α + Real.sin β = 1/2) 
  (h2 : Real.cos α + Real.cos β = 1/3) : 
  Real.cos (α - β) = -59/72 := by
sorry

end NUMINAMATH_CALUDE_cos_difference_formula_l2204_220419


namespace NUMINAMATH_CALUDE_complex_fraction_power_eight_l2204_220453

theorem complex_fraction_power_eight :
  ((2 + 2 * Complex.I) / (2 - 2 * Complex.I)) ^ 8 = 1 := by
  sorry

end NUMINAMATH_CALUDE_complex_fraction_power_eight_l2204_220453


namespace NUMINAMATH_CALUDE_five_mondays_in_november_l2204_220459

/-- Represents the days of the week -/
inductive DayOfWeek
  | Monday
  | Tuesday
  | Wednesday
  | Thursday
  | Friday
  | Saturday
  | Sunday

/-- Represents a month -/
structure Month where
  days : Nat
  firstDay : DayOfWeek

/-- Given that a month starts on a certain day, 
    returns the number of occurrences of a specific day in that month -/
def countDayOccurrences (month : Month) (day : DayOfWeek) : Nat :=
  sorry

/-- October of year M -/
def october : Month :=
  { days := 31, firstDay := sorry }

/-- November of year M -/
def november : Month :=
  { days := 30, firstDay := sorry }

theorem five_mondays_in_november 
  (h : countDayOccurrences october DayOfWeek.Friday = 5) :
  countDayOccurrences november DayOfWeek.Monday = 5 := by
  sorry

end NUMINAMATH_CALUDE_five_mondays_in_november_l2204_220459


namespace NUMINAMATH_CALUDE_area_of_region_T_l2204_220435

/-- Represents a rhombus PQRS -/
structure Rhombus where
  side_length : ℝ
  angle_Q : ℝ

/-- Represents the region T inside the rhombus -/
def region_T (r : Rhombus) : Set (ℝ × ℝ) :=
  sorry

/-- The area of a region -/
def area (s : Set (ℝ × ℝ)) : ℝ :=
  sorry

/-- The theorem statement -/
theorem area_of_region_T (r : Rhombus) :
  r.side_length = 4 ∧ r.angle_Q = 150 * π / 180 →
  area (region_T r) = 8 * Real.sqrt 3 / 9 :=
sorry

end NUMINAMATH_CALUDE_area_of_region_T_l2204_220435


namespace NUMINAMATH_CALUDE_f_properties_l2204_220493

noncomputable def f (x : ℝ) : ℝ := 
  Real.sin x * Real.cos x + (1 + Real.tan x ^ 2) * Real.cos x ^ 2

theorem f_properties : 
  (∃ (T : ℝ), T > 0 ∧ ∀ (x : ℝ), f (x + T) = f x ∧ 
    ∀ (S : ℝ), S > 0 ∧ (∀ (x : ℝ), f (x + S) = f x) → T ≤ S) ∧
  (∀ (x : ℝ), f x ≤ 3/2) ∧ 
  (∃ (x : ℝ), f x = 3/2) := by
  sorry

end NUMINAMATH_CALUDE_f_properties_l2204_220493


namespace NUMINAMATH_CALUDE_factorial_divisibility_l2204_220433

theorem factorial_divisibility :
  ¬(57 ∣ Nat.factorial 18) ∧ (57 ∣ Nat.factorial 19) := by
  sorry

end NUMINAMATH_CALUDE_factorial_divisibility_l2204_220433


namespace NUMINAMATH_CALUDE_simon_candy_count_l2204_220442

def candy_problem (initial_candies : ℕ) : Prop :=
  let day1_remaining := initial_candies - (initial_candies / 4) - 3
  let day2_remaining := day1_remaining - (day1_remaining / 2) - 5
  let day3_remaining := day2_remaining - ((3 * day2_remaining) / 4) - 6
  day3_remaining = 4

theorem simon_candy_count : 
  ∃ (x : ℕ), candy_problem x ∧ x = 124 :=
sorry

end NUMINAMATH_CALUDE_simon_candy_count_l2204_220442


namespace NUMINAMATH_CALUDE_max_value_theorem_max_value_achieved_l2204_220448

theorem max_value_theorem (u v : ℝ) (hu : u > 0) (hv : v > 0) (h : 4 * u + 3 * v < 84) :
  u * v * (84 - 4 * u - 3 * v)^2 ≤ 259308 :=
sorry

theorem max_value_achieved (u v : ℝ) (hu : u > 0) (hv : v > 0) (h : 4 * u + 3 * v < 84) :
  ∃ (u₀ v₀ : ℝ), u₀ > 0 ∧ v₀ > 0 ∧ 4 * u₀ + 3 * v₀ < 84 ∧
    u₀ * v₀ * (84 - 4 * u₀ - 3 * v₀)^2 = 259308 :=
sorry

end NUMINAMATH_CALUDE_max_value_theorem_max_value_achieved_l2204_220448


namespace NUMINAMATH_CALUDE_prob_three_in_seven_thirteenths_l2204_220457

/-- The decimal representation of a rational number -/
def decimalRepresentation (q : ℚ) : List ℕ := sorry

/-- The probability of a digit occurring in a decimal representation -/
def digitProbability (d : ℕ) (q : ℚ) : ℚ := sorry

/-- Theorem: The probability of selecting 3 from the decimal representation of 7/13 is 1/6 -/
theorem prob_three_in_seven_thirteenths :
  digitProbability 3 (7/13) = 1/6 := by sorry

end NUMINAMATH_CALUDE_prob_three_in_seven_thirteenths_l2204_220457


namespace NUMINAMATH_CALUDE_luna_makes_seven_per_hour_l2204_220486

/-- The number of milkshakes Augustus can make per hour -/
def augustus_rate : ℕ := 3

/-- The number of hours Augustus and Luna work -/
def work_hours : ℕ := 8

/-- The total number of milkshakes made by Augustus and Luna -/
def total_milkshakes : ℕ := 80

/-- The number of milkshakes Luna can make per hour -/
def luna_rate : ℕ := (total_milkshakes - augustus_rate * work_hours) / work_hours

theorem luna_makes_seven_per_hour : luna_rate = 7 := by
  sorry

end NUMINAMATH_CALUDE_luna_makes_seven_per_hour_l2204_220486

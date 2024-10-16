import Mathlib

namespace NUMINAMATH_CALUDE_geometric_to_arithmetic_progression_l263_26353

-- Define the four numbers
def a : ℝ := 2
def b : ℝ := 6
def c : ℝ := 18
def d : ℝ := 54

-- Theorem statement
theorem geometric_to_arithmetic_progression :
  -- The numbers form a geometric progression
  (b / a = c / b) ∧ (c / b = d / c) ∧
  -- When transformed, they form an arithmetic progression
  ((b + 4) - a = c - (b + 4)) ∧ (c - (b + 4) = (d - 28) - c) :=
by sorry

end NUMINAMATH_CALUDE_geometric_to_arithmetic_progression_l263_26353


namespace NUMINAMATH_CALUDE_unique_phone_number_l263_26335

def is_valid_phone_number (n : ℕ) : Prop :=
  10000000 ≤ n ∧ n < 100000000

def first_four (n : ℕ) : ℕ := n / 10000

def last_four (n : ℕ) : ℕ := n % 10000

def first_three (n : ℕ) : ℕ := n / 100000

def last_five (n : ℕ) : ℕ := n % 100000

theorem unique_phone_number :
  ∃! n : ℕ, is_valid_phone_number n ∧
    first_four n + last_four n = 14405 ∧
    first_three n + last_five n = 16970 ∧
    n = 82616144 := by
  sorry

end NUMINAMATH_CALUDE_unique_phone_number_l263_26335


namespace NUMINAMATH_CALUDE_a_can_ensure_segments_l263_26307

/-- Represents a point on the circle -/
structure Point where
  has_piece : Bool

/-- Represents a segment between two points -/
structure Segment where
  point1 : Point
  point2 : Point

/-- Represents the state of the game -/
structure GameState where
  n : Nat
  points : List Point
  segments : List Segment

/-- Player A's strategy -/
def player_a_strategy (state : GameState) : GameState :=
  sorry

/-- Player B's strategy -/
def player_b_strategy (state : GameState) : GameState :=
  sorry

/-- Counts the number of segments connecting a point with a piece and a point without a piece -/
def count_valid_segments (state : GameState) : Nat :=
  sorry

/-- Main theorem -/
theorem a_can_ensure_segments (n : Nat) (h : n ≥ 2) :
  ∃ (initial_state : GameState),
    initial_state.n = n ∧
    initial_state.points.length = 3 * n ∧
    (∀ (b_strategy : GameState → GameState),
      let final_state := (player_a_strategy ∘ b_strategy)^[n] initial_state
      count_valid_segments final_state ≥ (n - 1) / 6) :=
  sorry

end NUMINAMATH_CALUDE_a_can_ensure_segments_l263_26307


namespace NUMINAMATH_CALUDE_least_perimeter_triangle_l263_26312

theorem least_perimeter_triangle (a b x : ℕ) (ha : a = 24) (hb : b = 37) : 
  (a + b > x ∧ a + x > b ∧ b + x > a) → (∀ y : ℕ, (a + b > y ∧ a + y > b ∧ b + y > a) → x ≤ y) →
  a + b + x = 75 :=
sorry

end NUMINAMATH_CALUDE_least_perimeter_triangle_l263_26312


namespace NUMINAMATH_CALUDE_total_cost_calculation_l263_26362

/-- The total cost of remaining balloons for Sam and Mary -/
def total_cost (s a m c : ℝ) : ℝ :=
  ((s - a) + m) * c

/-- Theorem stating the total cost of remaining balloons for Sam and Mary -/
theorem total_cost_calculation (s a m c : ℝ) 
  (hs : s = 6) (ha : a = 5) (hm : m = 7) (hc : c = 9) : 
  total_cost s a m c = 72 := by
  sorry

#eval total_cost 6 5 7 9

end NUMINAMATH_CALUDE_total_cost_calculation_l263_26362


namespace NUMINAMATH_CALUDE_function_inequality_implies_positive_c_l263_26373

/-- Given a function f(x) = x^2 + x + c, if f(f(x)) > x for all real x, then c > 0 -/
theorem function_inequality_implies_positive_c (c : ℝ) : 
  (∀ x : ℝ, (x^2 + x + c)^2 + (x^2 + x + c) + c > x) → c > 0 := by
  sorry

end NUMINAMATH_CALUDE_function_inequality_implies_positive_c_l263_26373


namespace NUMINAMATH_CALUDE_f_min_at_neg_three_p_half_l263_26342

/-- The function f(x) = x^2 + 3px + 2p^2 -/
def f (p : ℝ) (x : ℝ) : ℝ := x^2 + 3*p*x + 2*p^2

/-- Theorem: The minimum of f(x) occurs at x = -3p/2 when p > 0 -/
theorem f_min_at_neg_three_p_half (p : ℝ) (h : p > 0) :
  ∀ x : ℝ, f p (-3*p/2) ≤ f p x :=
sorry

end NUMINAMATH_CALUDE_f_min_at_neg_three_p_half_l263_26342


namespace NUMINAMATH_CALUDE_cos_45_sin_30_product_equation_general_form_l263_26389

-- Problem 1
theorem cos_45_sin_30_product : 4 * Real.cos (π / 4) * Real.sin (π / 6) = Real.sqrt 2 := by sorry

-- Problem 2
theorem equation_general_form (x : ℝ) : (x + 2) * (x - 3) = 2 * x - 6 ↔ x^2 - 3 * x = 0 := by sorry

end NUMINAMATH_CALUDE_cos_45_sin_30_product_equation_general_form_l263_26389


namespace NUMINAMATH_CALUDE_xiao_ming_score_l263_26356

/-- Calculates the comprehensive score based on individual scores and weights -/
def comprehensive_score (written : ℝ) (practical : ℝ) (publicity : ℝ) 
  (written_weight : ℝ) (practical_weight : ℝ) (publicity_weight : ℝ) : ℝ :=
  written * written_weight + practical * practical_weight + publicity * publicity_weight

/-- Theorem stating that Xiao Ming's comprehensive score is 97 -/
theorem xiao_ming_score : 
  comprehensive_score 96 98 96 0.3 0.5 0.2 = 97 := by
  sorry

#eval comprehensive_score 96 98 96 0.3 0.5 0.2

end NUMINAMATH_CALUDE_xiao_ming_score_l263_26356


namespace NUMINAMATH_CALUDE_expand_expression_l263_26352

theorem expand_expression (x : ℝ) : -3*x*(x^2 - x - 2) = -3*x^3 + 3*x^2 + 6*x := by
  sorry

end NUMINAMATH_CALUDE_expand_expression_l263_26352


namespace NUMINAMATH_CALUDE_inequality_and_equality_condition_l263_26365

theorem inequality_and_equality_condition (a b c : ℝ) 
  (h1 : a ≥ 0) (h2 : b ≥ 0) (h3 : c ≥ 0) 
  (h4 : ¬(a = b ∧ b = c)) : 
  (a - b * c)^2 + (b - c * a)^2 + (c - a * b)^2 ≥ 
    (1/2) * ((a - b)^2 + (b - c)^2 + (c - a)^2) ∧
  ((a - b * c)^2 + (b - c * a)^2 + (c - a * b)^2 = 
    (1/2) * ((a - b)^2 + (b - c)^2 + (c - a)^2) ↔ 
    ((a = 0 ∧ b = 0 ∧ c > 0) ∨ (a = 0 ∧ c = 0 ∧ b > 0) ∨ (b = 0 ∧ c = 0 ∧ a > 0))) :=
by sorry

end NUMINAMATH_CALUDE_inequality_and_equality_condition_l263_26365


namespace NUMINAMATH_CALUDE_fruit_difference_is_eight_l263_26360

/-- Represents the number of fruits in a basket -/
structure FruitBasket where
  redPeaches : ℕ
  yellowPeaches : ℕ
  greenPeaches : ℕ
  blueApples : ℕ
  purpleBananas : ℕ
  orangeKiwis : ℕ

/-- Calculates the difference between peaches and other fruits -/
def peachDifference (basket : FruitBasket) : ℕ :=
  (basket.greenPeaches + basket.yellowPeaches) - (basket.blueApples + basket.purpleBananas)

/-- The theorem to be proved -/
theorem fruit_difference_is_eight :
  ∃ (basket : FruitBasket),
    basket.redPeaches = 2 ∧
    basket.yellowPeaches = 6 ∧
    basket.greenPeaches = 14 ∧
    basket.blueApples = 4 ∧
    basket.purpleBananas = 8 ∧
    basket.orangeKiwis = 12 ∧
    peachDifference basket = 8 := by
  sorry

end NUMINAMATH_CALUDE_fruit_difference_is_eight_l263_26360


namespace NUMINAMATH_CALUDE_sum_of_squares_l263_26399

-- Define the triangle FAC
structure Triangle :=
  (F A C : ℝ × ℝ)

-- Define the property of right angle FAC
def isRightAngle (t : Triangle) : Prop :=
  -- This is a placeholder for the right angle condition
  sorry

-- Define the length of CF
def CF_length (t : Triangle) : ℝ := 12

-- Define the area of square ACDE
def area_ACDE (t : Triangle) : ℝ :=
  let (_, A) := t.A
  let (_, C) := t.C
  (A - C) ^ 2

-- Define the area of square AFGH
def area_AFGH (t : Triangle) : ℝ :=
  let (F, _) := t.F
  let (A, _) := t.A
  (F - A) ^ 2

-- The theorem to be proved
theorem sum_of_squares (t : Triangle) 
  (h1 : isRightAngle t) 
  (h2 : CF_length t = 12) : 
  area_ACDE t + area_AFGH t = 144 :=
sorry

end NUMINAMATH_CALUDE_sum_of_squares_l263_26399


namespace NUMINAMATH_CALUDE_closest_fraction_l263_26351

def medals_won : ℚ := 23 / 150

def options : List ℚ := [1/5, 1/6, 1/7, 1/8, 1/9]

theorem closest_fraction (closest : ℚ) :
  closest ∈ options ∧
  ∀ x ∈ options, |medals_won - closest| ≤ |medals_won - x| :=
by sorry

end NUMINAMATH_CALUDE_closest_fraction_l263_26351


namespace NUMINAMATH_CALUDE_sqrt_of_nine_l263_26338

theorem sqrt_of_nine : Real.sqrt 9 = 3 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_of_nine_l263_26338


namespace NUMINAMATH_CALUDE_chess_tournament_orders_l263_26383

/-- Represents the number of players in the tournament -/
def num_players : ℕ := 4

/-- Represents the number of possible outcomes for each match -/
def outcomes_per_match : ℕ := 2

/-- Represents the number of matches in the tournament -/
def num_matches : ℕ := num_players - 1

/-- Calculates the total number of possible finishing orders -/
def total_possible_orders : ℕ := outcomes_per_match ^ num_matches

/-- Theorem stating that there are exactly 8 different possible finishing orders -/
theorem chess_tournament_orders : total_possible_orders = 8 := by
  sorry

end NUMINAMATH_CALUDE_chess_tournament_orders_l263_26383


namespace NUMINAMATH_CALUDE_dividend_calculation_l263_26310

theorem dividend_calculation (divisor quotient remainder : ℕ) 
  (h1 : divisor = 20)
  (h2 : quotient = 8)
  (h3 : remainder = 6) :
  divisor * quotient + remainder = 166 := by
  sorry

end NUMINAMATH_CALUDE_dividend_calculation_l263_26310


namespace NUMINAMATH_CALUDE_harrison_extra_pages_l263_26398

def minimum_pages : ℕ := 25
def sam_pages : ℕ := 100

def pam_pages (sam : ℕ) : ℕ := sam / 2

def harrison_pages (pam : ℕ) : ℕ := pam - 15

theorem harrison_extra_pages :
  harrison_pages (pam_pages sam_pages) - minimum_pages = 10 :=
by sorry

end NUMINAMATH_CALUDE_harrison_extra_pages_l263_26398


namespace NUMINAMATH_CALUDE_expected_difference_coffee_tea_days_l263_26341

/-- Represents the outcome of rolling an 8-sided die -/
inductive DieOutcome
| One
| Two
| Three
| Four
| Five
| Six
| Seven
| Eight

/-- Represents the drink choice based on the die roll -/
inductive DrinkChoice
| Coffee
| Tea

/-- Function to determine the drink choice based on the die outcome -/
def choosedrink (outcome : DieOutcome) : DrinkChoice :=
  match outcome with
  | DieOutcome.Two | DieOutcome.Three | DieOutcome.Five | DieOutcome.Seven => DrinkChoice.Coffee
  | _ => DrinkChoice.Tea

/-- Number of days in a leap year -/
def leapYearDays : Nat := 366

/-- Probability of rolling a number that results in drinking coffee -/
def probCoffee : ℚ := 4 / 7

/-- Probability of rolling a number that results in drinking tea -/
def probTea : ℚ := 3 / 7

/-- Expected number of days drinking coffee in a leap year -/
def expectedCoffeeDays : ℚ := probCoffee * leapYearDays

/-- Expected number of days drinking tea in a leap year -/
def expectedTeaDays : ℚ := probTea * leapYearDays

/-- Theorem stating the expected difference between coffee and tea days -/
theorem expected_difference_coffee_tea_days :
  ⌊expectedCoffeeDays - expectedTeaDays⌋ = 52 := by sorry

end NUMINAMATH_CALUDE_expected_difference_coffee_tea_days_l263_26341


namespace NUMINAMATH_CALUDE_a_is_negative_one_l263_26367

def A (a : ℝ) : Set ℝ := {0, a, a^2}

theorem a_is_negative_one (a : ℝ) (h : 1 ∈ A a) : a = -1 := by
  sorry

end NUMINAMATH_CALUDE_a_is_negative_one_l263_26367


namespace NUMINAMATH_CALUDE_root_sum_reciprocals_l263_26327

theorem root_sum_reciprocals (p q r s t : ℂ) : 
  p^5 + 10*p^4 + 20*p^3 + 15*p^2 + 8*p + 5 = 0 →
  q^5 + 10*q^4 + 20*q^3 + 15*q^2 + 8*q + 5 = 0 →
  r^5 + 10*r^4 + 20*r^3 + 15*r^2 + 8*r + 5 = 0 →
  s^5 + 10*s^4 + 20*s^3 + 15*s^2 + 8*s + 5 = 0 →
  t^5 + 10*t^4 + 20*t^3 + 15*t^2 + 8*t + 5 = 0 →
  1/(p*q) + 1/(p*r) + 1/(p*s) + 1/(p*t) + 1/(q*r) + 1/(q*s) + 1/(q*t) + 1/(r*s) + 1/(r*t) + 1/(s*t) = 4 := by
sorry

end NUMINAMATH_CALUDE_root_sum_reciprocals_l263_26327


namespace NUMINAMATH_CALUDE_line_through_points_l263_26348

theorem line_through_points (a b : ℚ) : 
  (7 : ℚ) = a * 3 + b ∧ (19 : ℚ) = a * 10 + b → a - b = -1/7 := by
  sorry

end NUMINAMATH_CALUDE_line_through_points_l263_26348


namespace NUMINAMATH_CALUDE_fibonacci_square_property_l263_26332

/-- Fibonacci sequence -/
def fib : ℕ → ℕ
  | 0 => 0
  | 1 => 1
  | n + 2 => fib (n + 1) + fib n

/-- Proposition: N^2 = 1 + n(N + n) iff (N, n) are consecutive Fibonacci numbers -/
theorem fibonacci_square_property (N n : ℕ) (hN : N > 0) (hn : n > 0) :
  N^2 = 1 + n * (N + n) ↔ ∃ i : ℕ, i > 0 ∧ N = fib (i + 1) ∧ n = fib i :=
sorry

end NUMINAMATH_CALUDE_fibonacci_square_property_l263_26332


namespace NUMINAMATH_CALUDE_lindas_savings_l263_26395

-- Define the problem parameters
def furniture_ratio : ℚ := 5/8
def tv_ratio : ℚ := 1/4
def tv_discount : ℚ := 15/100
def furniture_discount : ℚ := 10/100
def initial_tv_cost : ℚ := 320
def exchange_rate : ℚ := 11/10

-- Define the theorem
theorem lindas_savings : 
  ∃ (savings : ℚ),
    savings * tv_ratio * (1 - tv_discount) = initial_tv_cost * (1 - tv_discount) ∧
    savings * furniture_ratio * (1 - furniture_discount) * exchange_rate = 
      savings * furniture_ratio * (1 - furniture_discount) ∧
    savings = 1088 :=
sorry

end NUMINAMATH_CALUDE_lindas_savings_l263_26395


namespace NUMINAMATH_CALUDE_whale_consumption_increase_l263_26392

/-- Represents the whale's plankton consumption pattern -/
structure WhaleConsumption where
  initial : ℕ  -- Initial consumption in the first hour
  increase : ℕ  -- Constant increase each hour after the first
  duration : ℕ  -- Duration of the feeding frenzy in hours
  total : ℕ     -- Total accumulated consumption
  sixth_hour : ℕ -- Consumption in the sixth hour

/-- Theorem stating the whale's consumption increase -/
theorem whale_consumption_increase 
  (w : WhaleConsumption) 
  (h1 : w.duration = 9)
  (h2 : w.total = 450)
  (h3 : w.sixth_hour = 54)
  (h4 : w.initial + 5 * w.increase = w.sixth_hour)
  (h5 : (w.duration : ℕ) * w.initial + 
        (w.duration * (w.duration - 1) / 2) * w.increase = w.total) : 
  w.increase = 4 := by
  sorry

end NUMINAMATH_CALUDE_whale_consumption_increase_l263_26392


namespace NUMINAMATH_CALUDE_lines_perpendicular_iff_b_eq_neg_ten_l263_26343

-- Define the slopes of the two lines
def slope1 : ℚ := -1/2
def slope2 (b : ℚ) : ℚ := -b/5

-- Define the perpendicularity condition
def perpendicular (b : ℚ) : Prop := slope1 * slope2 b = -1

-- Theorem statement
theorem lines_perpendicular_iff_b_eq_neg_ten :
  ∀ b : ℚ, perpendicular b ↔ b = -10 := by sorry

end NUMINAMATH_CALUDE_lines_perpendicular_iff_b_eq_neg_ten_l263_26343


namespace NUMINAMATH_CALUDE_rectangles_with_at_least_three_cells_l263_26361

/-- The number of rectangles containing at least three cells in a 6x6 grid -/
def rectanglesWithAtLeastThreeCells : ℕ := 345

/-- The size of the grid -/
def gridSize : ℕ := 6

/-- Total number of rectangles in an n x n grid -/
def totalRectangles (n : ℕ) : ℕ := (n + 1).choose 2 * (n + 1).choose 2

/-- Number of 1x1 rectangles in an n x n grid -/
def oneByOneRectangles (n : ℕ) : ℕ := n * n

/-- Number of 1x2 and 2x1 rectangles in an n x n grid -/
def oneBytwoRectangles (n : ℕ) : ℕ := 2 * n * (n - 1)

theorem rectangles_with_at_least_three_cells :
  rectanglesWithAtLeastThreeCells = 
    totalRectangles gridSize - oneByOneRectangles gridSize - oneBytwoRectangles gridSize :=
by sorry

end NUMINAMATH_CALUDE_rectangles_with_at_least_three_cells_l263_26361


namespace NUMINAMATH_CALUDE_trigonometric_product_equals_one_l263_26371

theorem trigonometric_product_equals_one :
  let cos30 : ℝ := Real.sqrt 3 / 2
  let sin30 : ℝ := 1 / 2
  let cos60 : ℝ := 1 / 2
  let sin60 : ℝ := Real.sqrt 3 / 2
  (1 - 1 / cos30) * (1 + 1 / sin60) * (1 - 1 / sin30) * (1 + 1 / cos60) = 1 := by
sorry

end NUMINAMATH_CALUDE_trigonometric_product_equals_one_l263_26371


namespace NUMINAMATH_CALUDE_nested_fraction_evaluation_l263_26302

theorem nested_fraction_evaluation : 
  (1 : ℚ) / (2 + 1 / (3 + 1 / 4)) = 13 / 30 := by
  sorry

end NUMINAMATH_CALUDE_nested_fraction_evaluation_l263_26302


namespace NUMINAMATH_CALUDE_power_fraction_simplification_l263_26358

theorem power_fraction_simplification :
  (3^1024 + 5 * 3^1022) / (3^1024 - 3^1022) = 7/4 := by
  sorry

end NUMINAMATH_CALUDE_power_fraction_simplification_l263_26358


namespace NUMINAMATH_CALUDE_units_digit_of_27_cubed_minus_17_cubed_units_digit_is_zero_l263_26323

theorem units_digit_of_27_cubed_minus_17_cubed : ℕ → Prop :=
  fun d => (27^3 - 17^3) % 10 = d

theorem units_digit_is_zero :
  units_digit_of_27_cubed_minus_17_cubed 0 := by
  sorry

end NUMINAMATH_CALUDE_units_digit_of_27_cubed_minus_17_cubed_units_digit_is_zero_l263_26323


namespace NUMINAMATH_CALUDE_sequence_sum_l263_26300

theorem sequence_sum (A B C D E F G H I : ℝ) : 
  D = 7 →
  I = 10 →
  B + C + D = 36 →
  C + D + E = 36 →
  D + E + F = 36 →
  E + F + G = 36 →
  F + G + H = 36 →
  G + H + I = 36 →
  A + I = 17 := by
sorry

end NUMINAMATH_CALUDE_sequence_sum_l263_26300


namespace NUMINAMATH_CALUDE_h1n1_spread_properties_l263_26337

/-- Represents the spread of H1N1 flu in a community -/
def H1N1Spread (x : ℝ) : Prop :=
  (1 + x)^2 = 36 ∧ x > 0

theorem h1n1_spread_properties (x : ℝ) (hx : H1N1Spread x) :
  x = 5 ∧ (1 + x)^3 > 200 := by
  sorry

#check h1n1_spread_properties

end NUMINAMATH_CALUDE_h1n1_spread_properties_l263_26337


namespace NUMINAMATH_CALUDE_x_minus_25_is_perfect_square_l263_26326

/-- Represents the number of zeros after the first 1 in the definition of x -/
def zeros_after_first_one : ℕ := 2011

/-- Represents the number of zeros after the second 1 in the definition of x -/
def zeros_after_second_one : ℕ := 2012

/-- Defines x as described in the problem -/
def x : ℕ := 
  10^(zeros_after_second_one + 3) + 
  10^(zeros_after_first_one + zeros_after_second_one + 2) + 
  50

/-- States that x - 25 is a perfect square -/
theorem x_minus_25_is_perfect_square : 
  ∃ n : ℕ, x - 25 = n^2 := by sorry

end NUMINAMATH_CALUDE_x_minus_25_is_perfect_square_l263_26326


namespace NUMINAMATH_CALUDE_seventh_eleventh_150th_decimal_l263_26333

/-- The decimal representation of a rational number -/
def decimalRepresentation (q : ℚ) : ℕ → ℕ := sorry

/-- The period length of a rational number's decimal representation -/
def periodLength (q : ℚ) : ℕ := sorry

theorem seventh_eleventh_150th_decimal :
  (7 : ℚ) / 11 ∈ {q : ℚ | periodLength q = 2 ∧ decimalRepresentation q 150 = 3} := by
  sorry

end NUMINAMATH_CALUDE_seventh_eleventh_150th_decimal_l263_26333


namespace NUMINAMATH_CALUDE_smallest_prime_with_digit_sum_23_l263_26305

def digit_sum (n : ℕ) : ℕ := sorry

def is_prime (n : ℕ) : Prop := sorry

theorem smallest_prime_with_digit_sum_23 :
  (∀ p : ℕ, is_prime p ∧ digit_sum p = 23 → p ≥ 599) ∧
  is_prime 599 ∧
  digit_sum 599 = 23 := by sorry

end NUMINAMATH_CALUDE_smallest_prime_with_digit_sum_23_l263_26305


namespace NUMINAMATH_CALUDE_point_P_coordinates_l263_26347

def C (x : ℝ) : ℝ := x^3 - 10*x + 3

theorem point_P_coordinates :
  ∃! (x y : ℝ), 
    y = C x ∧ 
    x < 0 ∧ 
    y > 0 ∧ 
    (3 * x^2 - 10 = 2) ∧ 
    x = -2 ∧ 
    y = 15 := by
  sorry

end NUMINAMATH_CALUDE_point_P_coordinates_l263_26347


namespace NUMINAMATH_CALUDE_arithmetic_progression_with_prime_condition_l263_26336

theorem arithmetic_progression_with_prime_condition :
  ∀ (a b c d : ℤ),
  (∃ (k : ℤ), b = a + k ∧ c = b + k ∧ d = c + k) →  -- arithmetic progression
  (∃ (p : ℕ), Nat.Prime p ∧ (d - c + 1 : ℤ) = p) →  -- d - c + 1 is prime
  a + b^2 + c^3 = d^2 * b →                        -- given equation
  (∃ (n : ℤ), a = n ∧ b = n + 1 ∧ c = n + 2 ∧ d = n + 3) :=
by sorry

end NUMINAMATH_CALUDE_arithmetic_progression_with_prime_condition_l263_26336


namespace NUMINAMATH_CALUDE_min_coach_handshakes_zero_l263_26308

/-- The total number of handshakes in the gymnastics competition -/
def total_handshakes : ℕ := 325

/-- The number of gymnasts -/
def num_gymnasts : ℕ := 26

/-- The number of handshakes between gymnasts -/
def gymnast_handshakes (n : ℕ) : ℕ := n * (n - 1) / 2

/-- The number of handshakes involving coaches -/
def coach_handshakes (total : ℕ) (n : ℕ) : ℕ := total - gymnast_handshakes n

theorem min_coach_handshakes_zero :
  coach_handshakes total_handshakes num_gymnasts = 0 :=
by sorry

end NUMINAMATH_CALUDE_min_coach_handshakes_zero_l263_26308


namespace NUMINAMATH_CALUDE_extremum_cubic_function_l263_26376

/-- Given a cubic function with an extremum at x = 1 and f(1) = 10, prove a = 4 -/
theorem extremum_cubic_function (a b : ℝ) : 
  let f : ℝ → ℝ := λ x ↦ x^3 + a*x^2 + b*x + a^2
  (∃ ε > 0, ∀ x ∈ Set.Ioo (1 - ε) (1 + ε), f x ≤ f 1 ∨ f x ≥ f 1) → 
  f 1 = 10 →
  a = 4 := by
sorry

end NUMINAMATH_CALUDE_extremum_cubic_function_l263_26376


namespace NUMINAMATH_CALUDE_net_sales_effect_l263_26355

/-- Calculates the net effect on sales after two consecutive price reductions and sales increases -/
theorem net_sales_effect (initial_price_reduction : ℝ) 
                         (initial_sales_increase : ℝ)
                         (second_price_reduction : ℝ)
                         (second_sales_increase : ℝ) :
  initial_price_reduction = 0.20 →
  initial_sales_increase = 0.80 →
  second_price_reduction = 0.15 →
  second_sales_increase = 0.60 →
  let first_quarter_sales := 1 + initial_sales_increase
  let second_quarter_sales := first_quarter_sales * (1 + second_sales_increase)
  let net_effect := (second_quarter_sales - 1) * 100
  net_effect = 188 := by
sorry

end NUMINAMATH_CALUDE_net_sales_effect_l263_26355


namespace NUMINAMATH_CALUDE_largest_sum_of_squared_differences_l263_26320

theorem largest_sum_of_squared_differences (a b c : ℕ) : 
  a ≠ b ∧ b ≠ c ∧ a ≠ c →
  a > 0 ∧ b > 0 ∧ c > 0 →
  ∃ x y z : ℕ, (b + c - a = x^2) ∧ (c + a - b = y^2) ∧ (a + b - c = z^2) →
  a + b + c < 100 →
  a + b + c ≤ 91 :=
by sorry

end NUMINAMATH_CALUDE_largest_sum_of_squared_differences_l263_26320


namespace NUMINAMATH_CALUDE_min_difference_triangle_sides_l263_26386

theorem min_difference_triangle_sides (a b c : ℕ) : 
  a < b → b < c → a + b + c = 2509 → 
  (∀ x y z : ℕ, x < y ∧ y < z ∧ x + y + z = 2509 → y - x ≥ b - a) → 
  b - a = 1 :=
sorry

end NUMINAMATH_CALUDE_min_difference_triangle_sides_l263_26386


namespace NUMINAMATH_CALUDE_simplify_nested_expression_l263_26382

theorem simplify_nested_expression (x : ℝ) : 2 - (3 - (4 - (5 - (2*x - 3)))) = -5 + 2*x := by
  sorry

end NUMINAMATH_CALUDE_simplify_nested_expression_l263_26382


namespace NUMINAMATH_CALUDE_probability_both_selected_l263_26324

theorem probability_both_selected (prob_ram : ℚ) (prob_ravi : ℚ) 
  (h1 : prob_ram = 5 / 7) 
  (h2 : prob_ravi = 1 / 5) : 
  prob_ram * prob_ravi = 1 / 7 := by
  sorry

end NUMINAMATH_CALUDE_probability_both_selected_l263_26324


namespace NUMINAMATH_CALUDE_cafe_benches_theorem_l263_26364

/-- Converts a number from base 5 to base 10 -/
def base5ToBase10 (n : Nat) : Nat :=
  let hundreds := n / 100
  let tens := (n / 10) % 10
  let ones := n % 10
  hundreds * 5^2 + tens * 5^1 + ones * 5^0

/-- Calculates the number of benches needed given a total number of people and people per bench -/
def benchesNeeded (totalPeople : Nat) (peoplePerBench : Nat) : Nat :=
  (totalPeople + peoplePerBench - 1) / peoplePerBench

theorem cafe_benches_theorem (cafeCapacity : Nat) (peoplePerBench : Nat) :
  cafeCapacity = 310 ∧ peoplePerBench = 3 →
  benchesNeeded (base5ToBase10 cafeCapacity) peoplePerBench = 27 := by
  sorry

#eval benchesNeeded (base5ToBase10 310) 3

end NUMINAMATH_CALUDE_cafe_benches_theorem_l263_26364


namespace NUMINAMATH_CALUDE_smallest_positive_d_for_inequality_l263_26345

theorem smallest_positive_d_for_inequality :
  (∃ d : ℝ, d > 0 ∧
    (∀ x y : ℝ, x ≥ y^2 → Real.sqrt x + d * |y - x| ≥ 2 * |y|)) ∧
  (∀ d : ℝ, d > 0 ∧
    (∀ x y : ℝ, x ≥ y^2 → Real.sqrt x + d * |y - x| ≥ 2 * |y|) →
    d ≥ 1) :=
by sorry

end NUMINAMATH_CALUDE_smallest_positive_d_for_inequality_l263_26345


namespace NUMINAMATH_CALUDE_max_ratio_squared_l263_26363

theorem max_ratio_squared (a b x y : ℝ) : 
  0 < a → 0 < b → a ≥ b →
  0 ≤ x → x < a →
  0 ≤ y → y < b →
  a^2 + y^2 = b^2 + x^2 →
  b^2 + x^2 = (a - x)^2 + (b - y)^2 →
  a^2 + b^2 = x^2 + b^2 →
  (∀ a' b' : ℝ, 0 < a' → 0 < b' → a' ≥ b' → (a' / b')^2 ≤ 4/3) ∧
  (∃ a' b' : ℝ, 0 < a' → 0 < b' → a' ≥ b' → (a' / b')^2 = 4/3) :=
by sorry

end NUMINAMATH_CALUDE_max_ratio_squared_l263_26363


namespace NUMINAMATH_CALUDE_condition_necessary_not_sufficient_l263_26309

-- Define a sequence type
def Sequence := ℕ → ℝ

-- Define the property of a sequence satisfying a_n = 2a_{n-1} for n ≥ 2
def SatisfiesCondition (a : Sequence) : Prop :=
  ∀ n : ℕ, n ≥ 2 → a n = 2 * a (n - 1)

-- Define a geometric sequence with common ratio 2
def IsGeometricSequenceWithRatio2 (a : Sequence) : Prop :=
  ∃ r : ℝ, r ≠ 0 ∧ ∀ n : ℕ, n ≥ 1 → a (n + 1) = 2 * a n

-- Theorem statement
theorem condition_necessary_not_sufficient :
  (∀ a : Sequence, IsGeometricSequenceWithRatio2 a → SatisfiesCondition a) ∧
  (∃ a : Sequence, SatisfiesCondition a ∧ ¬IsGeometricSequenceWithRatio2 a) := by
  sorry

end NUMINAMATH_CALUDE_condition_necessary_not_sufficient_l263_26309


namespace NUMINAMATH_CALUDE_consecutive_sum_iff_not_power_of_two_l263_26306

theorem consecutive_sum_iff_not_power_of_two (n : ℕ) (h : n ≥ 3) :
  (∃ (a k : ℕ), k > 0 ∧ n = (k * (2 * a + k - 1)) / 2) ↔ ¬∃ (m : ℕ), n = 2^m :=
sorry

end NUMINAMATH_CALUDE_consecutive_sum_iff_not_power_of_two_l263_26306


namespace NUMINAMATH_CALUDE_intersection_of_M_and_N_l263_26374

def M : Set ℝ := {x | Real.log x > 0}
def N : Set ℝ := {x | x^2 ≤ 4}

theorem intersection_of_M_and_N : M ∩ N = Set.Ioo 1 2 := by sorry

end NUMINAMATH_CALUDE_intersection_of_M_and_N_l263_26374


namespace NUMINAMATH_CALUDE_cos_angle_between_vectors_l263_26390

/-- Given vectors a and b in ℝ², where a = (2, 1) and a + 2b = (4, 5),
    the cosine of the angle between a and b is equal to 4/5. -/
theorem cos_angle_between_vectors (a b : ℝ × ℝ) :
  a = (2, 1) →
  a + 2 • b = (4, 5) →
  (a.1 * b.1 + a.2 * b.2) / (Real.sqrt (a.1^2 + a.2^2) * Real.sqrt (b.1^2 + b.2^2)) = 4/5 := by
  sorry

end NUMINAMATH_CALUDE_cos_angle_between_vectors_l263_26390


namespace NUMINAMATH_CALUDE_words_with_A_count_l263_26319

/-- The number of letters in our alphabet -/
def n : ℕ := 4

/-- The length of words we're considering -/
def k : ℕ := 3

/-- The number of letters in our alphabet excluding 'A' -/
def m : ℕ := 3

/-- The number of 3-letter words that can be made from the letters A, B, C, and D, 
    with at least one A being used and allowing repetition of letters -/
def words_with_A : ℕ := n^k - m^k

theorem words_with_A_count : words_with_A = 37 := by sorry

end NUMINAMATH_CALUDE_words_with_A_count_l263_26319


namespace NUMINAMATH_CALUDE_hat_shop_pricing_l263_26388

theorem hat_shop_pricing (x : ℝ) : 
  let increased_price := 1.30 * x
  let final_price := 0.75 * increased_price
  final_price = 0.975 * x := by
sorry

end NUMINAMATH_CALUDE_hat_shop_pricing_l263_26388


namespace NUMINAMATH_CALUDE_new_time_ratio_l263_26334

-- Define the distances and speed ratio
def first_trip_distance : ℝ := 100
def second_trip_distance : ℝ := 500
def speed_ratio : ℝ := 4

-- Theorem statement
theorem new_time_ratio (v : ℝ) (hv : v > 0) :
  let t1 := first_trip_distance / v
  let t2 := second_trip_distance / (speed_ratio * v)
  t2 / t1 = 1.25 := by
sorry

end NUMINAMATH_CALUDE_new_time_ratio_l263_26334


namespace NUMINAMATH_CALUDE_quadratic_equation_with_zero_sum_coefficients_l263_26315

theorem quadratic_equation_with_zero_sum_coefficients :
  ∃ (a b c : ℝ), a ≠ 0 ∧ a + b + c = 0 ∧ ∀ x, a * x^2 + b * x + c = 0 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_equation_with_zero_sum_coefficients_l263_26315


namespace NUMINAMATH_CALUDE_thursday_miles_proof_l263_26378

/-- The number of miles flown on Tuesday each week -/
def tuesday_miles : ℕ := 1134

/-- The total number of miles flown over 3 weeks -/
def total_miles : ℕ := 7827

/-- The number of weeks the pilot flies -/
def num_weeks : ℕ := 3

/-- The number of miles flown on Thursday each week -/
def thursday_miles : ℕ := (total_miles - num_weeks * tuesday_miles) / num_weeks

theorem thursday_miles_proof :
  thursday_miles = 1475 :=
by sorry

end NUMINAMATH_CALUDE_thursday_miles_proof_l263_26378


namespace NUMINAMATH_CALUDE_matrix_equation_proof_l263_26368

theorem matrix_equation_proof : 
  let N : Matrix (Fin 2) (Fin 2) ℝ := !![4, 4; 2, 4]
  N^4 - 3 • N^3 + 3 • N^2 - N = !![16, 24; 8, 12] := by
  sorry

end NUMINAMATH_CALUDE_matrix_equation_proof_l263_26368


namespace NUMINAMATH_CALUDE_average_multiples_of_10_l263_26385

/-- The average of multiples of 10 from 10 to 500 inclusive is 255 -/
theorem average_multiples_of_10 : 
  let first := 10
  let last := 500
  let step := 10
  (first + last) / 2 = 255 := by sorry

end NUMINAMATH_CALUDE_average_multiples_of_10_l263_26385


namespace NUMINAMATH_CALUDE_number_divided_by_24_is_19_l263_26331

theorem number_divided_by_24_is_19 (x : ℤ) : (x / 24 = 19) → x = 456 := by
  sorry

end NUMINAMATH_CALUDE_number_divided_by_24_is_19_l263_26331


namespace NUMINAMATH_CALUDE_chess_club_election_theorem_l263_26369

def total_candidates : ℕ := 20
def previous_board_members : ℕ := 10
def board_positions : ℕ := 6

theorem chess_club_election_theorem :
  (Nat.choose total_candidates board_positions) - 
  (Nat.choose (total_candidates - previous_board_members) board_positions) = 38550 := by
  sorry

end NUMINAMATH_CALUDE_chess_club_election_theorem_l263_26369


namespace NUMINAMATH_CALUDE_income_data_mean_difference_l263_26384

/-- Represents the income data for a group of families -/
structure IncomeData where
  num_families : ℕ
  min_income : ℕ
  max_income : ℕ
  incorrect_max_income : ℕ

/-- Calculates the difference between the mean of incorrect data and actual data -/
def mean_difference (data : IncomeData) : ℚ :=
  (data.incorrect_max_income - data.max_income : ℚ) / data.num_families

/-- Theorem stating the difference between means for the given problem -/
theorem income_data_mean_difference :
  ∀ (data : IncomeData),
  data.num_families = 800 →
  data.min_income = 10000 →
  data.max_income = 120000 →
  data.incorrect_max_income = 1200000 →
  mean_difference data = 1350 := by
  sorry

#eval mean_difference {
  num_families := 800,
  min_income := 10000,
  max_income := 120000,
  incorrect_max_income := 1200000
}

end NUMINAMATH_CALUDE_income_data_mean_difference_l263_26384


namespace NUMINAMATH_CALUDE_ellipse_line_intersection_l263_26304

/-- Given an ellipse and a line, this theorem states the conditions for their intersection at two distinct points. -/
theorem ellipse_line_intersection (m : ℝ) : 
  (∃ (x₁ y₁ x₂ y₂ : ℝ), 
    (x₁ ≠ x₂ ∨ y₁ ≠ y₂) ∧ 
    (x₁^2 / 3 + y₁^2 / m = 1) ∧
    (x₁ + 2*y₁ - 2 = 0) ∧
    (x₂^2 / 3 + y₂^2 / m = 1) ∧
    (x₂ + 2*y₂ - 2 = 0)) ↔ 
  (m > 1/12 ∧ m < 3) ∨ m > 3 :=
sorry

end NUMINAMATH_CALUDE_ellipse_line_intersection_l263_26304


namespace NUMINAMATH_CALUDE_orchids_after_planting_l263_26354

/-- The number of orchid bushes in the park after planting -/
def total_orchids (initial : ℕ) (planted : ℕ) : ℕ :=
  initial + planted

/-- Theorem: The park will have 6 orchid bushes after planting -/
theorem orchids_after_planting :
  total_orchids 2 4 = 6 := by
sorry

end NUMINAMATH_CALUDE_orchids_after_planting_l263_26354


namespace NUMINAMATH_CALUDE_lcm_of_20_25_30_l263_26325

theorem lcm_of_20_25_30 : Nat.lcm (Nat.lcm 20 25) 30 = 300 := by
  sorry

end NUMINAMATH_CALUDE_lcm_of_20_25_30_l263_26325


namespace NUMINAMATH_CALUDE_triangle_side_length_l263_26357

theorem triangle_side_length (a : ℤ) : 
  (4 < a ∧ a < 10) ∧ 
  (7 + 3 > a ∧ a + 3 > 7 ∧ a + 7 > 3) →
  a = 8 :=
by
  sorry

end NUMINAMATH_CALUDE_triangle_side_length_l263_26357


namespace NUMINAMATH_CALUDE_minimum_value_and_inequality_l263_26393

def f (x m : ℝ) : ℝ := |x - m| + |x + 1|

theorem minimum_value_and_inequality {m a b c : ℝ} (h_min : ∀ x, f x m ≥ 4) 
  (h_pos : a > 0 ∧ b > 0 ∧ c > 0) (h_sum : a + 2*b + 3*c = m) :
  (m = -5 ∨ m = 3) ∧ (1/a + 1/(2*b) + 1/(3*c) ≥ 3) := by
  sorry

end NUMINAMATH_CALUDE_minimum_value_and_inequality_l263_26393


namespace NUMINAMATH_CALUDE_ribbon_used_wendy_ribbon_problem_l263_26394

/-- Given the total amount of ribbon and the amount left after wrapping presents,
    prove that the amount used for wrapping is the difference between the two. -/
theorem ribbon_used (total : ℕ) (leftover : ℕ) (h : leftover ≤ total) :
  total - leftover = (total - leftover : ℕ) :=
by sorry

/-- Wendy's ribbon problem -/
theorem wendy_ribbon_problem :
  let total := 84
  let leftover := 38
  total - leftover = 46 :=
by sorry

end NUMINAMATH_CALUDE_ribbon_used_wendy_ribbon_problem_l263_26394


namespace NUMINAMATH_CALUDE_joes_test_scores_l263_26346

theorem joes_test_scores (scores : Fin 4 → ℝ) 
  (avg_before : (scores 0 + scores 1 + scores 2 + scores 3) / 4 = 35)
  (avg_after : ∃ i, (scores 0 + scores 1 + scores 2 + scores 3 - scores i) / 3 = 40)
  (lowest : ∃ i, ∀ j, scores i ≤ scores j) :
  ∃ i, scores i = 20 ∧ ∀ j, scores i ≤ scores j :=
by sorry

end NUMINAMATH_CALUDE_joes_test_scores_l263_26346


namespace NUMINAMATH_CALUDE_correct_propositions_l263_26381

-- Define the propositions
def proposition1 : Prop := ∀ p q : Prop, (¬(p ∨ q)) → (¬p ∧ ¬q)

def proposition2 : Prop := 
  (∃ x : ℝ, x^2 + 1 > 3*x) ∧ (∀ x : ℝ, x^2 - 1 < 3*x)

def proposition3 : Prop := 
  ∀ a b m : ℝ, (a < b) → (a*m^2 < b*m^2)

def proposition4 : Prop := 
  ∀ p q : Prop, (p → q) ∧ ¬(q → p) → (¬p → ¬q) ∧ ¬(¬q → ¬p)

-- Theorem stating which propositions are correct
theorem correct_propositions : 
  proposition1 ∧ ¬proposition2 ∧ ¬proposition3 ∧ proposition4 := by
  sorry

end NUMINAMATH_CALUDE_correct_propositions_l263_26381


namespace NUMINAMATH_CALUDE_cuboid_volume_doubled_l263_26314

/-- The volume of a cuboid after doubling its dimensions -/
theorem cuboid_volume_doubled (original_volume : ℝ) : 
  original_volume = 36 → 8 * original_volume = 288 := by
  sorry

end NUMINAMATH_CALUDE_cuboid_volume_doubled_l263_26314


namespace NUMINAMATH_CALUDE_hcl_formation_l263_26301

/-- Represents the number of moles of a substance -/
def Moles : Type := ℝ

/-- Represents a chemical reaction between NaCl and HNO3 -/
structure Reaction where
  nacl : Moles
  hno3 : Moles
  nano3 : Moles
  hcl : Moles

/-- Defines a balanced reaction where NaCl and HNO3 react in a 1:1 ratio -/
def balanced_reaction (r : Reaction) : Prop :=
  r.nacl = r.hno3 ∧ r.nacl = r.hcl ∧ r.nacl = r.nano3

/-- Theorem: In a balanced reaction, the number of moles of HCl formed
    is equal to the number of moles of NaCl used -/
theorem hcl_formation (r : Reaction) (h : balanced_reaction r) :
  r.hcl = r.nacl := by sorry

end NUMINAMATH_CALUDE_hcl_formation_l263_26301


namespace NUMINAMATH_CALUDE_boat_speed_in_still_water_l263_26375

/-- The speed of a boat in still water, given its upstream speed and maximum opposing current and wind resistance. -/
theorem boat_speed_in_still_water 
  (upstream_speed : ℝ) 
  (max_current : ℝ) 
  (max_wind_resistance : ℝ) :
  upstream_speed = 4 →
  max_current = 4 →
  max_wind_resistance = 1 →
  ∃ (still_water_speed : ℝ), 
    still_water_speed = 7 ∧ 
    upstream_speed = still_water_speed - max_current + max_wind_resistance :=
by sorry

end NUMINAMATH_CALUDE_boat_speed_in_still_water_l263_26375


namespace NUMINAMATH_CALUDE_sixth_graders_count_l263_26311

theorem sixth_graders_count (seventh_graders : ℕ) (seventh_percent : ℚ) (sixth_percent : ℚ) 
  (h1 : seventh_graders = 64)
  (h2 : seventh_percent = 32 / 100)
  (h3 : sixth_percent = 38 / 100)
  (h4 : seventh_graders = (seventh_percent * (seventh_graders / seventh_percent)).floor) :
  (sixth_percent * (seventh_graders / seventh_percent)).floor = 76 := by
  sorry

end NUMINAMATH_CALUDE_sixth_graders_count_l263_26311


namespace NUMINAMATH_CALUDE_trigonometric_identities_l263_26303

theorem trigonometric_identities (θ : Real) 
  (h : Real.sin θ + Real.cos θ = -Real.sqrt 10 / 5) : 
  (1 / Real.sin θ + 1 / Real.cos θ = 2 * Real.sqrt 10 / 3) ∧ 
  (Real.tan θ = -1/3 ∨ Real.tan θ = -3) := by
  sorry

end NUMINAMATH_CALUDE_trigonometric_identities_l263_26303


namespace NUMINAMATH_CALUDE_probability_not_snow_l263_26321

theorem probability_not_snow (p : ℚ) (h : p = 5/8) : 1 - p = 3/8 := by
  sorry

end NUMINAMATH_CALUDE_probability_not_snow_l263_26321


namespace NUMINAMATH_CALUDE_coin_radius_l263_26344

/-- Given a coin with diameter 14 millimeters, its radius is 7 millimeters. -/
theorem coin_radius (d : ℝ) (h : d = 14) : d / 2 = 7 := by
  sorry

end NUMINAMATH_CALUDE_coin_radius_l263_26344


namespace NUMINAMATH_CALUDE_greatest_three_digit_multiple_of_17_l263_26316

theorem greatest_three_digit_multiple_of_17 : 
  ∀ n : ℕ, 100 ≤ n ∧ n < 1000 ∧ 17 ∣ n → n ≤ 986 := by
  sorry

end NUMINAMATH_CALUDE_greatest_three_digit_multiple_of_17_l263_26316


namespace NUMINAMATH_CALUDE_hcf_of_ratio_numbers_l263_26387

def ratio_numbers (x : ℕ) : Fin 4 → ℕ
  | 0 => 2 * x
  | 1 => 3 * x
  | 2 => 4 * x
  | 3 => 5 * x

theorem hcf_of_ratio_numbers (x : ℕ) (h1 : Nat.lcm (ratio_numbers x 0) (Nat.lcm (ratio_numbers x 1) (Nat.lcm (ratio_numbers x 2) (ratio_numbers x 3))) = 3600)
  (h2 : Nat.gcd (ratio_numbers x 2) (ratio_numbers x 3) = 4) :
  Nat.gcd (ratio_numbers x 0) (Nat.gcd (ratio_numbers x 1) (Nat.gcd (ratio_numbers x 2) (ratio_numbers x 3))) = 4 :=
by sorry

end NUMINAMATH_CALUDE_hcf_of_ratio_numbers_l263_26387


namespace NUMINAMATH_CALUDE_intersection_of_A_and_B_l263_26322

-- Define the sets A and B
variable (A B : Set ℤ)

-- Define the function f
def f (x : ℤ) : ℤ := x^2

-- State the theorem
theorem intersection_of_A_and_B :
  (∀ x ∈ A, f x ∈ B) →
  B = {1, 2} →
  (A ∩ B = ∅ ∨ A ∩ B = {1}) :=
by sorry

end NUMINAMATH_CALUDE_intersection_of_A_and_B_l263_26322


namespace NUMINAMATH_CALUDE_population_after_five_years_l263_26379

/-- Represents the yearly change in organization population -/
def yearly_change (b : ℝ) : ℝ := 2.7 * b - 8.5

/-- Calculates the population after n years -/
def population_after_years (initial_population : ℝ) (n : ℕ) : ℝ :=
  match n with
  | 0 => initial_population
  | n + 1 => yearly_change (population_after_years initial_population n)

/-- Theorem stating the population after 5 years -/
theorem population_after_five_years :
  ∃ (ε : ℝ), ε > 0 ∧ ε < 1 ∧ 
  |population_after_years 25 5 - 2875| < ε :=
sorry

end NUMINAMATH_CALUDE_population_after_five_years_l263_26379


namespace NUMINAMATH_CALUDE_abc_divisibility_problem_l263_26349

theorem abc_divisibility_problem :
  ∀ a b c : ℕ,
    1 < a → a < b → b < c →
    (∃ n : ℕ, abc - 1 = n * ((a - 1) * (b - 1) * (c - 1))) →
    ((a = 2 ∧ b = 4 ∧ c = 8) ∨ (a = 3 ∧ b = 5 ∧ c = 15)) :=
by
  sorry

#check abc_divisibility_problem

end NUMINAMATH_CALUDE_abc_divisibility_problem_l263_26349


namespace NUMINAMATH_CALUDE_intersection_of_A_and_B_l263_26329

def A : Set (ℝ × ℝ) := {p | p.2 = -p.1}
def B : Set (ℝ × ℝ) := {p | p.2 = p.1^2 - 2}

theorem intersection_of_A_and_B :
  A ∩ B = {(-2, 2), (1, -1)} := by sorry

end NUMINAMATH_CALUDE_intersection_of_A_and_B_l263_26329


namespace NUMINAMATH_CALUDE_polynomial_evaluation_l263_26313

theorem polynomial_evaluation : 
  let a : ℝ := 2
  (3 * a^3 - 7 * a^2 + a - 5) * (4 * a - 6) = -14 := by
sorry

end NUMINAMATH_CALUDE_polynomial_evaluation_l263_26313


namespace NUMINAMATH_CALUDE_binomial_9_6_l263_26396

theorem binomial_9_6 : Nat.choose 9 6 = 84 := by
  sorry

end NUMINAMATH_CALUDE_binomial_9_6_l263_26396


namespace NUMINAMATH_CALUDE_eggs_distribution_l263_26380

/-- Given a total number of eggs and a number of groups, 
    calculate the number of eggs per group -/
def eggs_per_group (total_eggs : ℕ) (num_groups : ℕ) : ℕ :=
  total_eggs / num_groups

/-- Theorem stating that with 18 eggs split into 3 groups, 
    each group should have 6 eggs -/
theorem eggs_distribution :
  eggs_per_group 18 3 = 6 := by
  sorry

end NUMINAMATH_CALUDE_eggs_distribution_l263_26380


namespace NUMINAMATH_CALUDE_product_purchase_percentage_l263_26330

theorem product_purchase_percentage
  (price_increase : ℝ)
  (expenditure_difference : ℝ)
  (h1 : price_increase = 0.25)
  (h2 : expenditure_difference = 0.125) :
  (1 + price_increase) * ((1 + expenditure_difference) / (1 + price_increase)) = 0.9 :=
sorry

end NUMINAMATH_CALUDE_product_purchase_percentage_l263_26330


namespace NUMINAMATH_CALUDE_coprime_powers_of_primes_l263_26391

def valid_n : Set ℕ := {1, 2, 3, 4, 5, 6, 8, 9, 10, 12, 14, 18, 20, 24, 30, 42}

def is_power_of_prime (m : ℕ) : Prop :=
  ∃ p k, Nat.Prime p ∧ m = p ^ k

theorem coprime_powers_of_primes (n : ℕ) :
  (∀ m, 0 < m ∧ m < n ∧ Nat.Coprime m n → is_power_of_prime m) ↔ n ∈ valid_n := by
  sorry

end NUMINAMATH_CALUDE_coprime_powers_of_primes_l263_26391


namespace NUMINAMATH_CALUDE_specific_doctor_selection_mixed_team_selection_l263_26372

-- Define the number of doctors
def total_doctors : ℕ := 20
def internal_medicine_doctors : ℕ := 12
def surgeons : ℕ := 8

-- Define the number of doctors to be selected
def team_size : ℕ := 5

-- Theorem for part (1)
theorem specific_doctor_selection :
  Nat.choose (total_doctors - 2) (team_size - 1) = 3060 := by sorry

-- Theorem for part (2)
theorem mixed_team_selection :
  Nat.choose total_doctors team_size - 
  Nat.choose internal_medicine_doctors team_size - 
  Nat.choose surgeons team_size = 14656 := by sorry

end NUMINAMATH_CALUDE_specific_doctor_selection_mixed_team_selection_l263_26372


namespace NUMINAMATH_CALUDE_trucks_needed_l263_26317

def total_apples : ℕ := 42
def transported_apples : ℕ := 22
def truck_capacity : ℕ := 4

theorem trucks_needed : 
  (total_apples - transported_apples) / truck_capacity = 5 := by
  sorry

end NUMINAMATH_CALUDE_trucks_needed_l263_26317


namespace NUMINAMATH_CALUDE_investment_interest_rates_l263_26366

theorem investment_interest_rates 
  (P1 P2 : ℝ) 
  (r1 r2 r3 r4 r5 : ℝ) :
  P1 / P2 = 2 / 3 →
  P1 * 5 * 8 / 100 = 840 →
  P2 * (r1 + r2 + r3 + r4 + r5) / 100 = 840 →
  r1 + r2 + r3 + r4 + r5 = 26.67 :=
by sorry

end NUMINAMATH_CALUDE_investment_interest_rates_l263_26366


namespace NUMINAMATH_CALUDE_factor_x6_minus_81_l263_26350

theorem factor_x6_minus_81 (x : ℝ) : x^6 - 81 = (x^3 + 9) * (x - 3) * (x^2 + 3*x + 9) := by
  sorry

end NUMINAMATH_CALUDE_factor_x6_minus_81_l263_26350


namespace NUMINAMATH_CALUDE_equilateral_triangle_perimeter_l263_26339

/-- 
Given an equilateral triangle with area 100√3 cm², 
prove that its perimeter is 60 cm.
-/
theorem equilateral_triangle_perimeter (A : ℝ) (p : ℝ) : 
  A = 100 * Real.sqrt 3 → p = 60 := by
  sorry

end NUMINAMATH_CALUDE_equilateral_triangle_perimeter_l263_26339


namespace NUMINAMATH_CALUDE_library_wall_leftover_space_l263_26377

theorem library_wall_leftover_space 
  (wall_length : ℝ) 
  (desk_length : ℝ) 
  (bookcase_length : ℝ) 
  (h1 : wall_length = 15) 
  (h2 : desk_length = 2) 
  (h3 : bookcase_length = 1.5) : 
  ∃ (n : ℕ), 
    n * desk_length + n * bookcase_length ≤ wall_length ∧ 
    (n + 1) * desk_length + (n + 1) * bookcase_length > wall_length ∧
    wall_length - (n * desk_length + n * bookcase_length) = 1 := by
  sorry

#check library_wall_leftover_space

end NUMINAMATH_CALUDE_library_wall_leftover_space_l263_26377


namespace NUMINAMATH_CALUDE_total_necklaces_l263_26370

def necklaces_problem (boudreaux rhonda latch cecilia : ℕ) : Prop :=
  boudreaux = 12 ∧
  rhonda = boudreaux / 2 ∧
  latch = 3 * rhonda - 4 ∧
  cecilia = latch + 3 ∧
  boudreaux + rhonda + latch + cecilia = 49

theorem total_necklaces : ∃ (boudreaux rhonda latch cecilia : ℕ), 
  necklaces_problem boudreaux rhonda latch cecilia :=
by
  sorry

end NUMINAMATH_CALUDE_total_necklaces_l263_26370


namespace NUMINAMATH_CALUDE_statement_1_statement_4_main_theorem_l263_26340

-- Statement ①
theorem statement_1 : ∀ x y : ℝ, (x + y ≠ 3) → (x ≠ 2 ∨ y ≠ 1) := by sorry

-- Statement ④
theorem statement_4 : (¬ ∃ x : ℝ, x^2 = 1) ↔ (∀ x : ℝ, x^2 ≠ 1) := by sorry

-- Main theorem combining both statements
theorem main_theorem : (∀ x y : ℝ, (x + y ≠ 3) → (x ≠ 2 ∨ y ≠ 1)) ∧
                       ((¬ ∃ x : ℝ, x^2 = 1) ↔ (∀ x : ℝ, x^2 ≠ 1)) := by sorry

end NUMINAMATH_CALUDE_statement_1_statement_4_main_theorem_l263_26340


namespace NUMINAMATH_CALUDE_number_difference_l263_26318

/-- 
Theorem: Given a three-digit number x and an even two-digit number y, 
if their difference is 3, then x = 101 and y = 98.
-/
theorem number_difference (x y : ℕ) : 
  (100 ≤ x ∧ x ≤ 999) →  -- x is a three-digit number
  (10 ≤ y ∧ y ≤ 98) →    -- y is a two-digit number
  Even y →               -- y is even
  x - y = 3 →            -- difference is 3
  x = 101 ∧ y = 98 :=
by sorry

end NUMINAMATH_CALUDE_number_difference_l263_26318


namespace NUMINAMATH_CALUDE_oliver_vowel_learning_time_l263_26397

/-- The number of days Oliver takes to learn one alphabet -/
def days_per_alphabet : ℕ := 5

/-- The number of vowels in the English alphabet -/
def number_of_vowels : ℕ := 5

/-- The total number of days Oliver needs to finish learning all vowels -/
def total_days : ℕ := days_per_alphabet * number_of_vowels

theorem oliver_vowel_learning_time : total_days = 25 := by
  sorry

end NUMINAMATH_CALUDE_oliver_vowel_learning_time_l263_26397


namespace NUMINAMATH_CALUDE_min_value_of_f_l263_26359

-- Define the function f(x)
def f (x : ℝ) (m : ℝ) : ℝ := 2 * x^3 - 6 * x^2 + m

-- State the theorem
theorem min_value_of_f (m : ℝ) :
  (∃ x ∈ Set.Icc (-2 : ℝ) 2, ∀ y ∈ Set.Icc (-2 : ℝ) 2, f x m ≥ f y m) ∧
  (∃ x ∈ Set.Icc (-2 : ℝ) 2, f x m = 3) →
  (∃ x ∈ Set.Icc (-2 : ℝ) 2, ∀ y ∈ Set.Icc (-2 : ℝ) 2, f x m ≤ f y m) ∧
  (∃ x ∈ Set.Icc (-2 : ℝ) 2, f x m = -37) :=
by sorry

end NUMINAMATH_CALUDE_min_value_of_f_l263_26359


namespace NUMINAMATH_CALUDE_updated_mean_after_corrections_l263_26328

/-- Calculates the updated mean of a set of observations after correcting errors -/
theorem updated_mean_after_corrections (n : ℕ) (initial_mean : ℚ) 
  (n1 n2 n3 : ℕ) (error1 error2 error3 : ℚ) : 
  n = 50 → 
  initial_mean = 200 → 
  n1 = 20 → 
  n2 = 15 → 
  n3 = 15 → 
  error1 = -6 → 
  error2 = -5 → 
  error3 = 3 → 
  (initial_mean * n + n1 * error1 + n2 * error2 + n3 * error3) / n = 197 := by
  sorry

#eval (200 * 50 + 20 * (-6) + 15 * (-5) + 15 * 3) / 50

end NUMINAMATH_CALUDE_updated_mean_after_corrections_l263_26328

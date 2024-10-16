import Mathlib

namespace NUMINAMATH_CALUDE_gold_balance_fraction_is_one_third_l104_10413

/-- Represents a credit card with a spending limit and balance. -/
structure CreditCard where
  limit : ℝ
  balance : ℝ

/-- Represents Sally's credit cards and their properties. -/
structure SallysCards where
  gold : CreditCard
  platinum : CreditCard
  gold_balance_fraction : ℝ
  platinum_balance_fraction : ℝ
  remaining_platinum_fraction : ℝ

/-- Theorem stating the fraction of the gold card's limit that represents the current balance. -/
theorem gold_balance_fraction_is_one_third
  (cards : SallysCards)
  (h1 : cards.platinum.limit = 2 * cards.gold.limit)
  (h2 : cards.gold.balance = cards.gold_balance_fraction * cards.gold.limit)
  (h3 : cards.platinum.balance = (1/4) * cards.platinum.limit)
  (h4 : cards.remaining_platinum_fraction = 0.5833333333333334)
  (h5 : cards.platinum.limit - (cards.platinum.balance + cards.gold.balance) =
        cards.remaining_platinum_fraction * cards.platinum.limit) :
  cards.gold_balance_fraction = 1/3 := by
  sorry

end NUMINAMATH_CALUDE_gold_balance_fraction_is_one_third_l104_10413


namespace NUMINAMATH_CALUDE_square_ends_in_six_tens_digit_odd_l104_10421

theorem square_ends_in_six_tens_digit_odd (n : ℤ) : 
  n^2 % 100 = 6 → (n^2 / 10) % 2 = 1 := by
  sorry

end NUMINAMATH_CALUDE_square_ends_in_six_tens_digit_odd_l104_10421


namespace NUMINAMATH_CALUDE_quadratic_trinomial_prime_square_solution_l104_10486

/-- A quadratic trinomial function -/
def f (x : ℤ) : ℤ := 2 * x^2 - x - 36

/-- Predicate to check if a number is prime -/
def is_prime (p : ℕ) : Prop := Nat.Prime p

/-- The main theorem statement -/
theorem quadratic_trinomial_prime_square_solution :
  ∃! x : ℤ, ∃ p : ℕ, is_prime p ∧ f x = p^2 ∧ x = 13 := by sorry

end NUMINAMATH_CALUDE_quadratic_trinomial_prime_square_solution_l104_10486


namespace NUMINAMATH_CALUDE_quadratic_roots_order_l104_10405

/-- A quadratic function f(x) = x^2 + bx + c -/
def f (b c x : ℝ) : ℝ := x^2 + b*x + c

/-- The statement of the problem -/
theorem quadratic_roots_order (b c x₁ x₂ x₃ x₄ : ℝ) :
  (∀ x, f b c x - x = 0 ↔ x = x₁ ∨ x = x₂) →
  x₂ - x₁ > 2 →
  (∀ x, f b c (f b c x) = x ↔ x = x₁ ∨ x = x₂ ∨ x = x₃ ∨ x = x₄) →
  x₃ > x₄ →
  x₄ < x₁ ∧ x₁ < x₃ ∧ x₃ < x₂ := by
sorry

end NUMINAMATH_CALUDE_quadratic_roots_order_l104_10405


namespace NUMINAMATH_CALUDE_reciprocal_of_repeating_decimal_l104_10461

def repeating_decimal : ℚ := 36 / 99

theorem reciprocal_of_repeating_decimal : (repeating_decimal)⁻¹ = 11 / 4 := by
  sorry

end NUMINAMATH_CALUDE_reciprocal_of_repeating_decimal_l104_10461


namespace NUMINAMATH_CALUDE_not_54_after_60_operations_l104_10490

def Operation := Nat → Nat

def is_valid_operation (op : Operation) : Prop :=
  ∀ n, (op n = 2 * n) ∨ (op n = n / 2) ∨ (op n = 3 * n) ∨ (op n = n / 3)

def apply_operations (initial : Nat) (ops : List Operation) : Nat :=
  ops.foldl (λ acc op => op acc) initial

theorem not_54_after_60_operations (ops : List Operation) 
  (h_length : ops.length = 60) 
  (h_valid : ∀ op ∈ ops, is_valid_operation op) : 
  apply_operations 12 ops ≠ 54 := by
  sorry

end NUMINAMATH_CALUDE_not_54_after_60_operations_l104_10490


namespace NUMINAMATH_CALUDE_penny_nickel_dime_heads_prob_l104_10417

/-- Represents the outcome of a coin flip -/
inductive CoinOutcome
| Heads
| Tails

/-- Represents the set of coins being flipped -/
structure CoinSet :=
  (penny : CoinOutcome)
  (nickel : CoinOutcome)
  (dime : CoinOutcome)
  (quarter : CoinOutcome)
  (half_dollar : CoinOutcome)

/-- The probability of getting heads on the penny, nickel, and dime when flipping five coins -/
def prob_penny_nickel_dime_heads : ℚ :=
  1 / 8

/-- Theorem stating that the probability of getting heads on the penny, nickel, and dime
    when flipping five coins simultaneously is 1/8 -/
theorem penny_nickel_dime_heads_prob :
  prob_penny_nickel_dime_heads = 1 / 8 :=
by sorry

end NUMINAMATH_CALUDE_penny_nickel_dime_heads_prob_l104_10417


namespace NUMINAMATH_CALUDE_symmetric_point_wrt_y_axis_l104_10469

/-- Given a point A(3,1) in a Cartesian coordinate system, 
    its symmetric point with respect to the y-axis has coordinates (-3,1). -/
theorem symmetric_point_wrt_y_axis : 
  let A : ℝ × ℝ := (3, 1)
  let symmetric_point := (-A.1, A.2)
  symmetric_point = (-3, 1) := by sorry

end NUMINAMATH_CALUDE_symmetric_point_wrt_y_axis_l104_10469


namespace NUMINAMATH_CALUDE_unique_a_in_A_l104_10457

def A (a : ℝ) : Set ℝ := {a + 2, (a + 1)^2, a^2 + 3*a + 3}

theorem unique_a_in_A : ∃! a : ℝ, 1 ∈ A a := by sorry

end NUMINAMATH_CALUDE_unique_a_in_A_l104_10457


namespace NUMINAMATH_CALUDE_max_hands_for_54_coincidences_l104_10450

/-- Represents a clock with minute hands moving in opposite directions -/
structure Clock :=
  (hands_clockwise : ℕ)
  (hands_counterclockwise : ℕ)

/-- The number of coincidences between pairs of hands in one hour -/
def coincidences (c : Clock) : ℕ :=
  2 * c.hands_clockwise * c.hands_counterclockwise

/-- The total number of hands on the clock -/
def total_hands (c : Clock) : ℕ :=
  c.hands_clockwise + c.hands_counterclockwise

/-- Theorem stating that if there are 54 coincidences in an hour,
    the maximum number of hands is 28 -/
theorem max_hands_for_54_coincidences :
  ∀ c : Clock, coincidences c = 54 → total_hands c ≤ 28 :=
by sorry

end NUMINAMATH_CALUDE_max_hands_for_54_coincidences_l104_10450


namespace NUMINAMATH_CALUDE_seven_balls_four_boxes_l104_10416

/-- The number of ways to distribute n indistinguishable balls into k distinguishable boxes -/
def distribute_balls (n : ℕ) (k : ℕ) : ℕ := sorry

/-- Theorem: There are 104 ways to distribute 7 indistinguishable balls into 4 distinguishable boxes -/
theorem seven_balls_four_boxes : distribute_balls 7 4 = 104 := by sorry

end NUMINAMATH_CALUDE_seven_balls_four_boxes_l104_10416


namespace NUMINAMATH_CALUDE_linear_function_properties_l104_10446

def f (x : ℝ) := -2 * x + 1

theorem linear_function_properties :
  (∀ x y, x < y → f x > f y) ∧  -- decreasing
  (∀ x, f x - (-2 * x) = 1) ∧  -- parallel to y = -2x
  (f 0 = 1) ∧  -- intersection with y-axis
  (∃ x y z, x > 0 ∧ y < 0 ∧ z > 0 ∧ f x > 0 ∧ f y < 0 ∧ f z < 0) :=  -- passes through 1st, 2nd, and 4th quadrants
by sorry

end NUMINAMATH_CALUDE_linear_function_properties_l104_10446


namespace NUMINAMATH_CALUDE_julia_monday_kids_l104_10471

/-- The number of kids Julia played with on Tuesday -/
def tuesday_kids : ℕ := 10

/-- The additional number of kids Julia played with on Monday compared to Tuesday -/
def additional_monday_kids : ℕ := 8

/-- The number of kids Julia played with on Monday -/
def monday_kids : ℕ := tuesday_kids + additional_monday_kids

theorem julia_monday_kids : monday_kids = 18 := by
  sorry

end NUMINAMATH_CALUDE_julia_monday_kids_l104_10471


namespace NUMINAMATH_CALUDE_min_slope_tangent_line_l104_10498

-- Define the curve
def f (x : ℝ) : ℝ := x^3 + 3*x - 1

-- Define the derivative of the curve
def f' (x : ℝ) : ℝ := 3*x^2 + 3

-- Theorem statement
theorem min_slope_tangent_line :
  ∃ (x₀ y₀ : ℝ), 
    (∀ x : ℝ, f' x ≥ f' x₀) ∧ 
    y₀ = f x₀ ∧
    (∀ x : ℝ, 3*x - y₀ = 1 → f x = 3*x - 1) :=
sorry

end NUMINAMATH_CALUDE_min_slope_tangent_line_l104_10498


namespace NUMINAMATH_CALUDE_basketball_tournament_equation_l104_10435

/-- The number of games played in a basketball tournament -/
def num_games : ℕ := 28

/-- Theorem: In a basketball tournament with x teams, where each pair of teams plays exactly one game,
    and a total of 28 games are played, the equation ½x(x-1) = 28 holds true. -/
theorem basketball_tournament_equation (x : ℕ) (h : x > 1) :
  (x * (x - 1)) / 2 = num_games :=
sorry

end NUMINAMATH_CALUDE_basketball_tournament_equation_l104_10435


namespace NUMINAMATH_CALUDE_knights_on_red_chairs_l104_10420

structure Room where
  total_chairs : Nat
  knights : Nat
  liars : Nat
  knights_on_red : Nat
  liars_on_blue : Nat

/-- The room satisfies the initial conditions -/
def initial_condition (r : Room) : Prop :=
  r.total_chairs = 20 ∧ 
  r.knights + r.liars = r.total_chairs

/-- The room satisfies the conditions after switching seats -/
def after_switch_condition (r : Room) : Prop :=
  r.knights_on_red + (r.knights - r.knights_on_red) = r.total_chairs / 2 ∧
  (r.liars - r.liars_on_blue) + r.liars_on_blue = r.total_chairs / 2 ∧
  r.knights_on_red = r.liars_on_blue

theorem knights_on_red_chairs (r : Room) 
  (h1 : initial_condition r) 
  (h2 : after_switch_condition r) : 
  r.knights_on_red = 5 := by
  sorry

end NUMINAMATH_CALUDE_knights_on_red_chairs_l104_10420


namespace NUMINAMATH_CALUDE_inequality_solution_set_l104_10437

-- Define the function representing the left side of the inequality
def f (x : ℝ) : ℝ := -x^2 - 4*x + 5

-- Define the solution set
def solution_set : Set ℝ := {x | -5 < x ∧ x < 1}

-- Theorem stating that the solution set of the inequality is correct
theorem inequality_solution_set : 
  ∀ x : ℝ, f x > 0 ↔ x ∈ solution_set := by
  sorry

end NUMINAMATH_CALUDE_inequality_solution_set_l104_10437


namespace NUMINAMATH_CALUDE_all_measurements_correct_l104_10411

-- Define a structure for measurements
structure Measurement where
  value : Float
  unit : String

-- Define the measurements
def ruler_length : Measurement := { value := 2, unit := "decimeters" }
def truck_capacity : Measurement := { value := 5, unit := "tons" }
def bus_speed : Measurement := { value := 100, unit := "kilometers" }
def book_thickness : Measurement := { value := 7, unit := "millimeters" }
def backpack_weight : Measurement := { value := 4000, unit := "grams" }

-- Define propositions for correct units
def correct_ruler_unit (m : Measurement) : Prop := m.unit = "decimeters"
def correct_truck_unit (m : Measurement) : Prop := m.unit = "tons"
def correct_bus_unit (m : Measurement) : Prop := m.unit = "kilometers"
def correct_book_unit (m : Measurement) : Prop := m.unit = "millimeters"
def correct_backpack_unit (m : Measurement) : Prop := m.unit = "grams"

-- Theorem stating that all measurements have correct units
theorem all_measurements_correct : 
  correct_ruler_unit ruler_length ∧
  correct_truck_unit truck_capacity ∧
  correct_bus_unit bus_speed ∧
  correct_book_unit book_thickness ∧
  correct_backpack_unit backpack_weight :=
by sorry


end NUMINAMATH_CALUDE_all_measurements_correct_l104_10411


namespace NUMINAMATH_CALUDE_tangent_length_specific_circle_l104_10428

/-- A circle passing through three points -/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

/-- The circle passing through three given points -/
def circleThrough (A B C : ℝ × ℝ) : Circle :=
  sorry

/-- The length of a tangent segment from a point to a circle -/
def tangentLength (P : ℝ × ℝ) (c : Circle) : ℝ :=
  sorry

/-- The theorem stating the length of the tangent segment -/
theorem tangent_length_specific_circle :
  let A : ℝ × ℝ := (4, 5)
  let B : ℝ × ℝ := (7, 9)
  let C : ℝ × ℝ := (6, 14)
  let P : ℝ × ℝ := (1, 1)
  let c := circleThrough A B C
  tangentLength P c = 5 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_tangent_length_specific_circle_l104_10428


namespace NUMINAMATH_CALUDE_clippings_per_friend_l104_10466

theorem clippings_per_friend 
  (num_friends : ℕ) 
  (total_glue_drops : ℕ) 
  (glue_drops_per_clipping : ℕ) 
  (h1 : num_friends = 7)
  (h2 : total_glue_drops = 126)
  (h3 : glue_drops_per_clipping = 6) :
  (total_glue_drops / glue_drops_per_clipping) / num_friends = 3 :=
by sorry

end NUMINAMATH_CALUDE_clippings_per_friend_l104_10466


namespace NUMINAMATH_CALUDE_arithmetic_sum_equals_180_l104_10491

/-- The sum of an arithmetic sequence with first term 30, common difference 10, and 4 terms -/
def arithmeticSum : ℕ := sorry

/-- The first term of the sequence -/
def firstTerm : ℕ := 30

/-- The common difference between consecutive terms -/
def commonDifference : ℕ := 10

/-- The number of terms in the sequence -/
def numberOfTerms : ℕ := 4

/-- Theorem stating that the sum of the arithmetic sequence is 180 -/
theorem arithmetic_sum_equals_180 : arithmeticSum = 180 := by sorry

end NUMINAMATH_CALUDE_arithmetic_sum_equals_180_l104_10491


namespace NUMINAMATH_CALUDE_square_plus_reciprocal_square_l104_10477

theorem square_plus_reciprocal_square (x : ℝ) (h : x + 1/x = 3.5) : x^2 + 1/x^2 = 10.25 := by
  sorry

end NUMINAMATH_CALUDE_square_plus_reciprocal_square_l104_10477


namespace NUMINAMATH_CALUDE_triangle_side_length_l104_10495

/-- Given a triangle ABC with sides a, b, c opposite to angles A, B, C respectively,
    if ∠B = 60°, ∠C = 75°, and a = 4, then b = 2√6. -/
theorem triangle_side_length (A B C : ℝ) (a b c : ℝ) : 
  B = 60 * π / 180 →
  C = 75 * π / 180 →
  a = 4 →
  (A + B + C = π) →
  (a / Real.sin A = b / Real.sin B) →
  b = 2 * Real.sqrt 6 := by
  sorry

end NUMINAMATH_CALUDE_triangle_side_length_l104_10495


namespace NUMINAMATH_CALUDE_quadratic_equation_real_root_l104_10456

theorem quadratic_equation_real_root (m : ℝ) : 
  ∃ x : ℝ, x^2 - (m + 1) * x + (3 * m - 6) = 0 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_equation_real_root_l104_10456


namespace NUMINAMATH_CALUDE_fraction_division_problem_l104_10439

theorem fraction_division_problem : (5 : ℚ) / (8 / 13) = 65 / 8 := by
  sorry

end NUMINAMATH_CALUDE_fraction_division_problem_l104_10439


namespace NUMINAMATH_CALUDE_anthony_pencils_l104_10430

theorem anthony_pencils (initial final added : ℕ) 
  (h1 : added = 56)
  (h2 : final = 65)
  (h3 : final = initial + added) :
  initial = 9 := by
sorry

end NUMINAMATH_CALUDE_anthony_pencils_l104_10430


namespace NUMINAMATH_CALUDE_f_min_at_4_l104_10487

/-- The quadratic function f(x) = x^2 - 8x + 15 -/
def f (x : ℝ) : ℝ := x^2 - 8*x + 15

/-- Theorem: The function f(x) = x^2 - 8x + 15 has a minimum value when x = 4 -/
theorem f_min_at_4 : ∀ y : ℝ, f 4 ≤ f y := by
  sorry

end NUMINAMATH_CALUDE_f_min_at_4_l104_10487


namespace NUMINAMATH_CALUDE_trevor_coin_conversion_l104_10404

/-- Represents the types of coins in the problem -/
inductive Coin
  | Quarter
  | Dime
  | Nickel
  | Penny

/-- Calculates the value of a coin in cents -/
def coinValue (c : Coin) : ℕ :=
  match c with
  | Coin.Quarter => 25
  | Coin.Dime => 10
  | Coin.Nickel => 5
  | Coin.Penny => 1

/-- Represents the coin count in Trevor's bank -/
structure CoinCount where
  total : ℕ
  quarters : ℕ
  dimes : ℕ
  nickels : ℕ
  pennies : ℕ

/-- Calculates the total value of coins in cents -/
def totalValue (cc : CoinCount) : ℕ :=
  cc.quarters * coinValue Coin.Quarter +
  cc.dimes * coinValue Coin.Dime +
  cc.nickels * coinValue Coin.Nickel +
  cc.pennies * coinValue Coin.Penny

/-- Converts total value to $5 bills and $1 coins -/
def convertToBillsAndCoins (value : ℕ) : (ℕ × ℕ) :=
  (value / 500, (value % 500) / 100)

theorem trevor_coin_conversion :
  let cc : CoinCount := {
    total := 153,
    quarters := 45,
    dimes := 34,
    nickels := 19,
    pennies := 153 - 45 - 34 - 19
  }
  let (fiveBills, oneDollars) := convertToBillsAndCoins (totalValue cc)
  fiveBills - oneDollars = 2 := by sorry

end NUMINAMATH_CALUDE_trevor_coin_conversion_l104_10404


namespace NUMINAMATH_CALUDE_koala_fiber_intake_l104_10452

/-- Given that koalas absorb 40% of the fiber they eat and a particular koala
    absorbed 16 ounces of fiber in one day, prove that it ate 40 ounces of fiber. -/
theorem koala_fiber_intake (absorption_rate : ℝ) (absorbed_amount : ℝ) (total_intake : ℝ) :
  absorption_rate = 0.40 →
  absorbed_amount = 16 →
  absorbed_amount = absorption_rate * total_intake →
  total_intake = 40 := by
  sorry

end NUMINAMATH_CALUDE_koala_fiber_intake_l104_10452


namespace NUMINAMATH_CALUDE_dillar_dallar_never_equal_l104_10459

/-- Represents the state of the financier's money -/
structure MoneyState :=
  (dillars : ℕ)
  (dallars : ℕ)

/-- Represents a currency exchange operation -/
inductive ExchangeOp
  | DillarToDallar : ExchangeOp
  | DallarToDillar : ExchangeOp

/-- Applies an exchange operation to a money state -/
def applyExchange (state : MoneyState) (op : ExchangeOp) : MoneyState :=
  match op with
  | ExchangeOp.DillarToDallar => 
      ⟨state.dillars - 1, state.dallars + 10⟩
  | ExchangeOp.DallarToDillar => 
      ⟨state.dillars + 10, state.dallars - 1⟩

/-- Applies a sequence of exchange operations to an initial state -/
def applyExchanges (initial : MoneyState) (ops : List ExchangeOp) : MoneyState :=
  ops.foldl applyExchange initial

theorem dillar_dallar_never_equal :
  ∀ (ops : List ExchangeOp),
    let finalState := applyExchanges ⟨1, 0⟩ ops
    finalState.dillars ≠ finalState.dallars :=
by sorry

end NUMINAMATH_CALUDE_dillar_dallar_never_equal_l104_10459


namespace NUMINAMATH_CALUDE_walking_rate_ratio_l104_10468

theorem walking_rate_ratio (usual_time new_time : ℝ) 
  (h1 : usual_time = 16)
  (h2 : new_time = usual_time - 4) :
  new_time / usual_time = 3 / 4 :=
by sorry

end NUMINAMATH_CALUDE_walking_rate_ratio_l104_10468


namespace NUMINAMATH_CALUDE_train_length_l104_10479

/-- The length of a train given its speed and the time it takes to cross a platform -/
theorem train_length (train_speed : ℝ) (platform_length : ℝ) (crossing_time : ℝ) :
  train_speed = 72 * (5/18) → 
  platform_length = 260 →
  crossing_time = 26 →
  train_speed * crossing_time - platform_length = 260 := by
  sorry

end NUMINAMATH_CALUDE_train_length_l104_10479


namespace NUMINAMATH_CALUDE_ten_factorial_mod_thirteen_l104_10499

def factorial (n : ℕ) : ℕ := 
  match n with
  | 0 => 1
  | n + 1 => (n + 1) * factorial n

theorem ten_factorial_mod_thirteen : 
  factorial 10 % 13 = 7 := by
  sorry

end NUMINAMATH_CALUDE_ten_factorial_mod_thirteen_l104_10499


namespace NUMINAMATH_CALUDE_floor_ceiling_sum_l104_10483

theorem floor_ceiling_sum : ⌊(0.999 : ℝ)⌋ + ⌈(2.001 : ℝ)⌉ = 3 := by sorry

end NUMINAMATH_CALUDE_floor_ceiling_sum_l104_10483


namespace NUMINAMATH_CALUDE_rectangle_area_l104_10400

theorem rectangle_area (perimeter : ℝ) (width : ℝ) (length : ℝ) : 
  perimeter = 40 →
  length = 2 * width →
  width * length = 800 / 9 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_area_l104_10400


namespace NUMINAMATH_CALUDE_roots_sum_cube_plus_linear_l104_10455

theorem roots_sum_cube_plus_linear (α β : ℝ) : 
  (α^2 + 2*α - 1 = 0) → 
  (β^2 + 2*β - 1 = 0) → 
  α^3 + 5*β + 10 = -2 := by
sorry

end NUMINAMATH_CALUDE_roots_sum_cube_plus_linear_l104_10455


namespace NUMINAMATH_CALUDE_painting_time_theorem_l104_10489

def painter_a_rate : ℝ := 50
def painter_b_rate : ℝ := 40
def painter_c_rate : ℝ := 30

def room_7_area : ℝ := 220
def room_8_area : ℝ := 320
def room_9_area : ℝ := 420
def room_10_area : ℝ := 270

def total_area : ℝ := room_7_area + room_8_area + room_9_area + room_10_area
def combined_rate : ℝ := painter_a_rate + painter_b_rate + painter_c_rate

theorem painting_time_theorem : 
  total_area / combined_rate = 10.25 := by sorry

end NUMINAMATH_CALUDE_painting_time_theorem_l104_10489


namespace NUMINAMATH_CALUDE_ways_without_first_grade_ways_with_all_grades_l104_10438

/-- Represents the number of products of each grade -/
structure ProductCounts where
  total : Nat
  firstGrade : Nat
  secondGrade : Nat
  thirdGrade : Nat

/-- The given product counts in the problem -/
def givenCounts : ProductCounts :=
  { total := 8
  , firstGrade := 3
  , secondGrade := 3
  , thirdGrade := 2 }

/-- Number of products to draw -/
def drawCount : Nat := 4

/-- Calculates the number of ways to choose k items from n items -/
def choose (n k : Nat) : Nat :=
  if k > n then 0
  else Nat.factorial n / (Nat.factorial k * Nat.factorial (n - k))

/-- Theorem for the first question -/
theorem ways_without_first_grade (counts : ProductCounts) :
  choose (counts.secondGrade + counts.thirdGrade) drawCount = 5 :=
sorry

/-- Theorem for the second question -/
theorem ways_with_all_grades (counts : ProductCounts) :
  choose counts.firstGrade 2 * choose counts.secondGrade 1 * choose counts.thirdGrade 1 +
  choose counts.firstGrade 1 * choose counts.secondGrade 2 * choose counts.thirdGrade 1 +
  choose counts.firstGrade 1 * choose counts.secondGrade 1 * choose counts.thirdGrade 2 = 45 :=
sorry

end NUMINAMATH_CALUDE_ways_without_first_grade_ways_with_all_grades_l104_10438


namespace NUMINAMATH_CALUDE_f_min_value_l104_10492

-- Define the function f
def f (x : ℝ) : ℝ := |x - 2| + |5 - x|

-- State the theorem
theorem f_min_value :
  ∃ (m : ℝ), (∀ (x : ℝ), f x ≥ m) ∧ (∃ (x : ℝ), f x = m) ∧ (m = 3) :=
sorry

end NUMINAMATH_CALUDE_f_min_value_l104_10492


namespace NUMINAMATH_CALUDE_peanut_box_count_l104_10443

/-- Given an initial quantity of peanuts in a box and an additional quantity added,
    compute the final quantity of peanuts in the box. -/
def final_peanut_count (initial : ℕ) (added : ℕ) : ℕ := initial + added

/-- Theorem stating that given 4 initial peanuts and 6 added peanuts, 
    the final count is 10 peanuts. -/
theorem peanut_box_count : final_peanut_count 4 6 = 10 := by
  sorry

end NUMINAMATH_CALUDE_peanut_box_count_l104_10443


namespace NUMINAMATH_CALUDE_find_t_l104_10444

theorem find_t (s t : ℚ) (eq1 : 8 * s + 7 * t = 95) (eq2 : s = 2 * t - 3) : t = 119 / 23 := by
  sorry

end NUMINAMATH_CALUDE_find_t_l104_10444


namespace NUMINAMATH_CALUDE_police_emergency_number_prime_divisor_l104_10476

theorem police_emergency_number_prime_divisor (n : ℕ) (k : ℕ) (h : n = 100 * k + 133) :
  ∃ p : ℕ, p.Prime ∧ p > 7 ∧ p ∣ n := by
  sorry

end NUMINAMATH_CALUDE_police_emergency_number_prime_divisor_l104_10476


namespace NUMINAMATH_CALUDE_parametric_equation_solution_l104_10442

theorem parametric_equation_solution (a b : ℝ) (h1 : a ≠ 2 * b) (h2 : a ≠ -3 * b) :
  ∃! x : ℝ, (a * x - 3) / (b * x + 1) = 2 :=
by
  use 5 / (a - 2 * b)
  sorry

end NUMINAMATH_CALUDE_parametric_equation_solution_l104_10442


namespace NUMINAMATH_CALUDE_one_eighth_of_two_to_36_l104_10402

theorem one_eighth_of_two_to_36 (y : ℤ) :
  (1 / 8 : ℚ) * (2 : ℚ)^36 = (2 : ℚ)^y → y = 33 := by
  sorry

end NUMINAMATH_CALUDE_one_eighth_of_two_to_36_l104_10402


namespace NUMINAMATH_CALUDE_fraction_equality_l104_10481

theorem fraction_equality (a b : ℚ) (h : a / b = 2 / 3) : a / (a + b) = 2 / 5 := by
  sorry

end NUMINAMATH_CALUDE_fraction_equality_l104_10481


namespace NUMINAMATH_CALUDE_cooking_probability_l104_10480

-- Define the set of courses
def Courses := Finset.range 4

-- Define the probability of selecting a specific course
def prob_select (course : Courses) : ℚ :=
  1 / Courses.card

-- State the theorem
theorem cooking_probability :
  ∃ (cooking : Courses), prob_select cooking = 1 / 4 :=
sorry

end NUMINAMATH_CALUDE_cooking_probability_l104_10480


namespace NUMINAMATH_CALUDE_rectangle_perimeter_l104_10410

/-- Given a square with perimeter 100 units divided vertically into 4 congruent rectangles,
    the perimeter of one of these rectangles is 62.5 units. -/
theorem rectangle_perimeter (s : ℝ) (h1 : s > 0) (h2 : 4 * s = 100) : 
  2 * (s + s / 4) = 62.5 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_perimeter_l104_10410


namespace NUMINAMATH_CALUDE_quadratic_sum_of_constants_l104_10454

theorem quadratic_sum_of_constants (x : ℝ) : 
  ∃ (b c : ℝ), x^2 - 20*x + 49 = (x + b)^2 + c ∧ b + c = -61 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_sum_of_constants_l104_10454


namespace NUMINAMATH_CALUDE_trigonometric_simplification_l104_10445

theorem trigonometric_simplification (α : Real) :
  (1 - 2 * Real.sin α ^ 2) / (2 * Real.tan (5 * Real.pi / 4 + α) * Real.cos (Real.pi / 4 + α) ^ 2) -
  Real.tan α + Real.sin (Real.pi / 2 + α) - Real.cos (α - Real.pi / 2) =
  (2 * Real.sqrt 2 * Real.cos (Real.pi / 4 + α) * Real.cos (α / 2) ^ 2) / Real.cos α :=
by sorry

end NUMINAMATH_CALUDE_trigonometric_simplification_l104_10445


namespace NUMINAMATH_CALUDE_hexagon_triangle_join_l104_10415

/-- A regular polygon with n sides and side length 1 -/
structure RegularPolygon where
  sides : Nat
  sideLength : ℝ
  isRegular : sideLength = 1

/-- The number of edges in a shape formed by joining two regular polygons edge-to-edge -/
def joinedEdges (p1 p2 : RegularPolygon) (sharedEdges : Nat) : Nat :=
  p1.sides + p2.sides - sharedEdges

theorem hexagon_triangle_join :
  ∀ (hexagon : RegularPolygon) (triangle : RegularPolygon),
    hexagon.sides = 6 →
    triangle.sides = 3 →
    (joinedEdges hexagon triangle 3) = 5 := by
  sorry

end NUMINAMATH_CALUDE_hexagon_triangle_join_l104_10415


namespace NUMINAMATH_CALUDE_rectangle_side_increase_l104_10441

theorem rectangle_side_increase (increase_factor : Real) :
  increase_factor > 0 →
  (1 + increase_factor)^2 = 1.8225 →
  increase_factor = 0.35 := by
sorry

end NUMINAMATH_CALUDE_rectangle_side_increase_l104_10441


namespace NUMINAMATH_CALUDE_linear_equation_condition_l104_10453

theorem linear_equation_condition (m : ℝ) : 
  (|m - 1| = 1 ∧ m - 2 ≠ 0) ↔ m = 0 := by
sorry

end NUMINAMATH_CALUDE_linear_equation_condition_l104_10453


namespace NUMINAMATH_CALUDE_largest_k_exists_l104_10409

def X : ℕ → ℤ
  | 0 => 1
  | 1 => 1
  | (n + 2) => X (n + 1) + 2 * X n

def Y : ℕ → ℤ
  | 0 => 1
  | 1 => 1
  | (n + 2) => 3 * Y (n + 1) + 4 * Y n

theorem largest_k_exists : ∃! k : ℕ, k < 10^2007 ∧
  (∃ i : ℕ+, |X i - k| ≤ 2007) ∧
  (∃ j : ℕ+, |Y j - k| ≤ 2007) ∧
  ∀ m : ℕ, m > k → ¬(
    (∃ i : ℕ+, |X i - m| ≤ 2007) ∧
    (∃ j : ℕ+, |Y j - m| ≤ 2007) ∧
    m < 10^2007
  ) :=
by sorry

end NUMINAMATH_CALUDE_largest_k_exists_l104_10409


namespace NUMINAMATH_CALUDE_jaco_gift_budget_l104_10418

/-- Given a total budget, number of friends, and cost of parent gifts, 
    calculate the budget for each friend's gift -/
def friend_gift_budget (total_budget : ℕ) (num_friends : ℕ) (parent_gift_cost : ℕ) : ℕ :=
  (total_budget - 2 * parent_gift_cost) / num_friends

/-- Proof that Jaco's budget for each friend's gift is $9 -/
theorem jaco_gift_budget :
  friend_gift_budget 100 8 14 = 9 := by
  sorry

end NUMINAMATH_CALUDE_jaco_gift_budget_l104_10418


namespace NUMINAMATH_CALUDE_solve_systems_of_equations_l104_10433

theorem solve_systems_of_equations :
  -- System 1
  (∃ (x y : ℝ), x - y = 3 ∧ 3*x - 8*y = 14 ∧ x = 2 ∧ y = -1) ∧
  -- System 2
  (∃ (x y : ℝ), 3*x + y = 1 ∧ 5*x - 2*y = 9 ∧ x = 1 ∧ y = -2) :=
by sorry

end NUMINAMATH_CALUDE_solve_systems_of_equations_l104_10433


namespace NUMINAMATH_CALUDE_exists_multiple_sum_of_digits_divides_l104_10467

/-- Sum of digits function -/
def sum_of_digits (n : ℕ) : ℕ := sorry

/-- Theorem: For every positive integer n, there exists a multiple of n whose sum of digits divides it -/
theorem exists_multiple_sum_of_digits_divides (n : ℕ+) : 
  ∃ k : ℕ+, (sum_of_digits (k * n) ∣ (k * n)) := by sorry

end NUMINAMATH_CALUDE_exists_multiple_sum_of_digits_divides_l104_10467


namespace NUMINAMATH_CALUDE_sin_thirteen_pi_sixths_l104_10460

theorem sin_thirteen_pi_sixths : Real.sin (13 * π / 6) = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_sin_thirteen_pi_sixths_l104_10460


namespace NUMINAMATH_CALUDE_expression_perfect_square_iff_l104_10496

def is_perfect_square (x : ℕ) : Prop := ∃ y : ℕ, y * y = x

def expression (n : ℕ) : ℕ := (n^2 + 11*n - 4) * n.factorial + 33 * 13^n + 4

theorem expression_perfect_square_iff (n : ℕ) (hn : n > 0) :
  is_perfect_square (expression n) ↔ n = 1 ∨ n = 2 :=
sorry

end NUMINAMATH_CALUDE_expression_perfect_square_iff_l104_10496


namespace NUMINAMATH_CALUDE_arithmetic_geometric_sequence_common_difference_l104_10488

-- Define an arithmetic sequence
def is_arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

-- Define a geometric sequence
def is_geometric_sequence (a : ℕ → ℝ) (start finish : ℕ) : Prop :=
  ∃ r : ℝ, ∀ n ∈ Finset.range (finish - start), a (start + n + 1) = r * a (start + n)

theorem arithmetic_geometric_sequence_common_difference :
  ∀ a : ℕ → ℝ,
  is_arithmetic_sequence a →
  is_geometric_sequence a 1 3 →
  a 1 = 1 →
  ∃ d : ℝ, (∀ n : ℕ, a (n + 1) = a n + d) ∧ d = 0 :=
sorry

end NUMINAMATH_CALUDE_arithmetic_geometric_sequence_common_difference_l104_10488


namespace NUMINAMATH_CALUDE_sequence_ratio_l104_10463

def arithmetic_sum (a₁ : ℚ) (d : ℚ) (n : ℕ) : ℚ :=
  n * (2 * a₁ + (n - 1) * d) / 2

def arithmetic_square_sum (a₁ : ℚ) (d : ℚ) (n : ℕ) : ℚ :=
  (n * (2 * a₁^2 + (n - 1) * d^2 + (n - 1) * d * a₁)) / 2

theorem sequence_ratio :
  let n := (38 - 4) / 2 + 1
  arithmetic_sum 4 2 n / arithmetic_square_sum 3 3 n = 2 / 15 := by
  sorry

end NUMINAMATH_CALUDE_sequence_ratio_l104_10463


namespace NUMINAMATH_CALUDE_honey_harvest_increase_l104_10408

/-- Proves that the increase in honey harvest is 6085 pounds -/
theorem honey_harvest_increase 
  (last_year_harvest : ℕ) 
  (this_year_harvest : ℕ) 
  (h1 : last_year_harvest = 2479)
  (h2 : this_year_harvest = 8564) : 
  this_year_harvest - last_year_harvest = 6085 := by
  sorry

end NUMINAMATH_CALUDE_honey_harvest_increase_l104_10408


namespace NUMINAMATH_CALUDE_rectangle_dissection_theorem_l104_10497

/-- Represents a rectangle with width and height -/
structure Rectangle where
  width : ℝ
  height : ℝ

/-- Represents a triangle -/
structure Triangle

/-- Represents a pentagon -/
structure Pentagon

/-- Represents a set of shapes that can be rearranged -/
structure ShapeSet where
  triangles : Finset Triangle
  pentagon : Pentagon

theorem rectangle_dissection_theorem (initial : Rectangle) (final : Rectangle) 
  (h_initial : initial.width = 4 ∧ initial.height = 6)
  (h_final : final.width = 3 ∧ final.height = 8)
  (h_area_preservation : initial.width * initial.height = final.width * final.height) :
  ∃ (pieces : ShapeSet), 
    pieces.triangles.card = 2 ∧ 
    (∃ (arrangement : ShapeSet → Rectangle), arrangement pieces = final) :=
sorry

end NUMINAMATH_CALUDE_rectangle_dissection_theorem_l104_10497


namespace NUMINAMATH_CALUDE_weekly_reading_time_l104_10425

-- Define the daily meditation time
def daily_meditation_time : ℝ := 1

-- Define the daily reading time as twice the meditation time
def daily_reading_time : ℝ := 2 * daily_meditation_time

-- Define the number of days in a week
def days_in_week : ℕ := 7

-- Theorem to prove
theorem weekly_reading_time : daily_reading_time * days_in_week = 14 := by
  sorry

end NUMINAMATH_CALUDE_weekly_reading_time_l104_10425


namespace NUMINAMATH_CALUDE_min_cost_for_family_l104_10478

/-- Represents the ticket prices in rubles -/
structure TicketPrices where
  adult_single : ℕ
  child_single : ℕ
  day_pass_single : ℕ
  day_pass_group : ℕ
  three_day_pass_single : ℕ
  three_day_pass_group : ℕ

/-- Calculates the minimum amount spent on tickets for a family -/
def min_ticket_cost (prices : TicketPrices) (days : ℕ) (trips_per_day : ℕ) : ℕ :=
  sorry

/-- The theorem stating the minimum cost for the given scenario -/
theorem min_cost_for_family (prices : TicketPrices) :
  prices.adult_single = 40 →
  prices.child_single = 20 →
  prices.day_pass_single = 350 →
  prices.day_pass_group = 1500 →
  prices.three_day_pass_single = 900 →
  prices.three_day_pass_group = 3500 →
  min_ticket_cost prices 5 10 = 5200 := by
  sorry

end NUMINAMATH_CALUDE_min_cost_for_family_l104_10478


namespace NUMINAMATH_CALUDE_distance_walked_l104_10482

theorem distance_walked (x t : ℝ) 
  (h1 : (x + 1) * (3/4 * t) = x * t) 
  (h2 : (x - 1) * (t + 3) = x * t) : 
  x * t = 18 := by
  sorry

end NUMINAMATH_CALUDE_distance_walked_l104_10482


namespace NUMINAMATH_CALUDE_min_shots_for_battleship_l104_10474

/-- Represents a grid position --/
structure Position where
  row : Nat
  col : Nat

/-- Represents a ship placement --/
inductive ShipPlacement
  | Horizontal : Position → ShipPlacement
  | Vertical : Position → ShipPlacement

/-- The grid size --/
def gridSize : Nat := 5

/-- The ship length --/
def shipLength : Nat := 4

/-- Checks if a position is within the grid --/
def isValidPosition (p : Position) : Prop :=
  p.row ≥ 1 ∧ p.row ≤ gridSize ∧ p.col ≥ 1 ∧ p.col ≤ gridSize

/-- Checks if a ship placement is valid --/
def isValidPlacement (sp : ShipPlacement) : Prop :=
  match sp with
  | ShipPlacement.Horizontal p => isValidPosition p ∧ p.col + shipLength - 1 ≤ gridSize
  | ShipPlacement.Vertical p => isValidPosition p ∧ p.row + shipLength - 1 ≤ gridSize

/-- Checks if a shot hits a ship placement --/
def hitsShip (shot : Position) (sp : ShipPlacement) : Prop :=
  match sp with
  | ShipPlacement.Horizontal p =>
      shot.row = p.row ∧ shot.col ≥ p.col ∧ shot.col < p.col + shipLength
  | ShipPlacement.Vertical p =>
      shot.col = p.col ∧ shot.row ≥ p.row ∧ shot.row < p.row + shipLength

/-- The main theorem --/
theorem min_shots_for_battleship :
  ∃ (shots : List Position),
    shots.length = 6 ∧
    (∀ sp : ShipPlacement, isValidPlacement sp →
      ∃ shot ∈ shots, hitsShip shot sp) ∧
    (∀ (shots' : List Position),
      shots'.length < 6 →
      ∃ sp : ShipPlacement, isValidPlacement sp ∧
        ∀ shot ∈ shots', ¬hitsShip shot sp) :=
  sorry


end NUMINAMATH_CALUDE_min_shots_for_battleship_l104_10474


namespace NUMINAMATH_CALUDE_cos_pi_fourth_plus_alpha_l104_10427

theorem cos_pi_fourth_plus_alpha (α : Real) 
  (h : Real.sin (π / 4 - α) = 1 / 3) : 
  Real.cos (π / 4 + α) = 1 / 3 := by
  sorry

end NUMINAMATH_CALUDE_cos_pi_fourth_plus_alpha_l104_10427


namespace NUMINAMATH_CALUDE_teacup_cost_function_l104_10451

-- Define the cost of a single teacup
def teacup_cost : ℚ := 2.5

-- Define the function for the total cost
def total_cost (x : ℕ+) : ℚ := x.val * teacup_cost

-- Theorem statement
theorem teacup_cost_function (x : ℕ+) (y : ℚ) :
  y = total_cost x ↔ y = 2.5 * x.val := by sorry

end NUMINAMATH_CALUDE_teacup_cost_function_l104_10451


namespace NUMINAMATH_CALUDE_distinct_prime_factors_of_90_l104_10464

theorem distinct_prime_factors_of_90 : Nat.card (Nat.factors 90).toFinset = 3 := by
  sorry

end NUMINAMATH_CALUDE_distinct_prime_factors_of_90_l104_10464


namespace NUMINAMATH_CALUDE_intersection_of_A_and_B_l104_10424

def A : Set ℝ := {x | -1 < x ∧ x ≤ 5}
def B : Set ℝ := {-1, 2, 3, 6}

theorem intersection_of_A_and_B : A ∩ B = {2, 3} := by sorry

end NUMINAMATH_CALUDE_intersection_of_A_and_B_l104_10424


namespace NUMINAMATH_CALUDE_johns_tax_rate_l104_10472

theorem johns_tax_rate (john_income ingrid_income : ℝ)
  (ingrid_tax_rate combined_tax_rate : ℝ)
  (h1 : john_income = 56000)
  (h2 : ingrid_income = 74000)
  (h3 : ingrid_tax_rate = 0.4)
  (h4 : combined_tax_rate = 0.3569) :
  let total_income := john_income + ingrid_income
  let total_tax := combined_tax_rate * total_income
  let ingrid_tax := ingrid_tax_rate * ingrid_income
  let john_tax := total_tax - ingrid_tax
  john_tax / john_income = 0.3 := by
  sorry

end NUMINAMATH_CALUDE_johns_tax_rate_l104_10472


namespace NUMINAMATH_CALUDE_ratio_equality_l104_10432

theorem ratio_equality (a b c : ℝ) (h : a/2 = b/3 ∧ b/3 = c/4) : (a + b) / c = 5/4 := by
  sorry

end NUMINAMATH_CALUDE_ratio_equality_l104_10432


namespace NUMINAMATH_CALUDE_sphere_packing_ratio_l104_10462

/-- Configuration of four spheres with two radii -/
structure SpherePacking where
  r : ℝ  -- radius of smaller spheres
  R : ℝ  -- radius of larger spheres
  r_positive : r > 0
  R_positive : R > 0
  touch_plane : True  -- represents that all spheres touch the plane
  touch_others : True  -- represents that each sphere touches three others

/-- Theorem stating the ratio of radii in the sphere packing configuration -/
theorem sphere_packing_ratio (config : SpherePacking) : config.R / config.r = 1 + Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_sphere_packing_ratio_l104_10462


namespace NUMINAMATH_CALUDE_distance_BD_l104_10422

/-- Given three points B, C, and D in a 2D plane, prove that the distance between B and D is 13. -/
theorem distance_BD (B C D : ℝ × ℝ) : 
  B = (3, 9) → C = (3, -3) → D = (-2, -3) → 
  Real.sqrt ((B.1 - D.1)^2 + (B.2 - D.2)^2) = 13 := by
  sorry

end NUMINAMATH_CALUDE_distance_BD_l104_10422


namespace NUMINAMATH_CALUDE_area_of_overlapping_rotated_squares_exists_l104_10447

/-- Represents a square in 2D space -/
structure Square where
  sideLength : ℝ
  rotation : ℝ -- in radians

/-- Calculates the area of a polygon formed by overlapping squares -/
noncomputable def areaOfOverlappingSquares (squares : List Square) : ℝ :=
  sorry

theorem area_of_overlapping_rotated_squares_exists : 
  ∃ (A : ℝ), 
    let squares := [
      { sideLength := 4, rotation := 0 },
      { sideLength := 5, rotation := π/4 },
      { sideLength := 6, rotation := -π/6 }
    ]
    A = areaOfOverlappingSquares squares ∧ A > 0 := by
  sorry

end NUMINAMATH_CALUDE_area_of_overlapping_rotated_squares_exists_l104_10447


namespace NUMINAMATH_CALUDE_triangle_perimeter_l104_10440

theorem triangle_perimeter : ∀ x : ℝ,
  x^2 - 11*x + 30 = 0 →
  2 + x > 4 ∧ 4 + x > 2 ∧ 2 + 4 > x →
  2 + 4 + x = 11 :=
by
  sorry

end NUMINAMATH_CALUDE_triangle_perimeter_l104_10440


namespace NUMINAMATH_CALUDE_money_problem_l104_10458

theorem money_problem (c d : ℝ) 
  (h1 : 3 * c - 2 * d < 30)
  (h2 : 4 * c + d = 60) :
  c < 150 / 11 ∧ d > 60 / 11 := by
  sorry

end NUMINAMATH_CALUDE_money_problem_l104_10458


namespace NUMINAMATH_CALUDE_average_chocolate_pieces_per_cookie_l104_10475

/-- Proves that given 48 cookies, 108 chocolate chips, and 1/3 as many M&Ms as chocolate chips,
    the average number of chocolate pieces per cookie is 3. -/
theorem average_chocolate_pieces_per_cookie
  (num_cookies : ℕ)
  (num_choc_chips : ℕ)
  (num_mms : ℕ)
  (h1 : num_cookies = 48)
  (h2 : num_choc_chips = 108)
  (h3 : num_mms = num_choc_chips / 3)
  : (num_choc_chips + num_mms) / num_cookies = 3 := by
  sorry

#eval (108 + 108 / 3) / 48  -- Should output 3

end NUMINAMATH_CALUDE_average_chocolate_pieces_per_cookie_l104_10475


namespace NUMINAMATH_CALUDE_fraction_equality_l104_10403

theorem fraction_equality (a b c : ℝ) (h : a^2 = b*c) :
  (a + b) / (a - b) = (c + a) / (c - a) :=
by sorry

end NUMINAMATH_CALUDE_fraction_equality_l104_10403


namespace NUMINAMATH_CALUDE_f_3_range_l104_10407

-- Define the function f(x) = a x^2 - c
def f (a c x : ℝ) : ℝ := a * x^2 - c

-- State the theorem
theorem f_3_range (a c : ℝ) :
  (∀ x : ℝ, f a c x = a * x^2 - c) →
  (-4 ≤ f a c 1 ∧ f a c 1 ≤ -1) →
  (-1 ≤ f a c 2 ∧ f a c 2 ≤ 5) →
  (-1 ≤ f a c 3 ∧ f a c 3 ≤ 20) :=
by sorry

end NUMINAMATH_CALUDE_f_3_range_l104_10407


namespace NUMINAMATH_CALUDE_cistern_wet_surface_area_l104_10426

/-- The total wet surface area of a rectangular cistern -/
def total_wet_surface_area (length width height : ℝ) : ℝ :=
  length * width + 2 * length * height + 2 * width * height

/-- Theorem: The total wet surface area of a cistern with given dimensions -/
theorem cistern_wet_surface_area :
  total_wet_surface_area 9 6 2.25 = 121.5 := by
  sorry

end NUMINAMATH_CALUDE_cistern_wet_surface_area_l104_10426


namespace NUMINAMATH_CALUDE_parallelogram_area_example_l104_10414

/-- The area of a parallelogram with given base and height -/
def parallelogram_area (base height : ℝ) : ℝ := base * height

theorem parallelogram_area_example : 
  parallelogram_area 10 20 = 200 := by sorry

end NUMINAMATH_CALUDE_parallelogram_area_example_l104_10414


namespace NUMINAMATH_CALUDE_square_sum_from_product_and_sum_l104_10448

theorem square_sum_from_product_and_sum (r s : ℝ) 
  (h1 : r * s = 24) 
  (h2 : r + s = 10) : 
  r^2 + s^2 = 52 := by
sorry

end NUMINAMATH_CALUDE_square_sum_from_product_and_sum_l104_10448


namespace NUMINAMATH_CALUDE_second_person_average_pages_per_day_l104_10449

theorem second_person_average_pages_per_day 
  (summer_days : ℕ) 
  (books_read : ℕ) 
  (avg_pages_per_book : ℕ) 
  (second_person_percentage : ℚ) 
  (h1 : summer_days = 80)
  (h2 : books_read = 60)
  (h3 : avg_pages_per_book = 320)
  (h4 : second_person_percentage = 3/4) : 
  (books_read * avg_pages_per_book * second_person_percentage) / summer_days = 180 := by
  sorry

end NUMINAMATH_CALUDE_second_person_average_pages_per_day_l104_10449


namespace NUMINAMATH_CALUDE_equation_solution_l104_10431

theorem equation_solution : ∃ (x : ℝ), 
  Real.sqrt (9 + Real.sqrt (25 + 5*x)) + Real.sqrt (3 + Real.sqrt (5 + x)) = 3 + 3 * Real.sqrt 3 ∧ 
  x = 0.2 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l104_10431


namespace NUMINAMATH_CALUDE_leadership_assignment_theorem_l104_10470

def community_size : ℕ := 12
def chief_count : ℕ := 1
def supporting_chief_count : ℕ := 2
def senior_officer_count : ℕ := 2
def inferior_officer_count : ℕ := 2

def leadership_assignment_count : ℕ :=
  community_size *
  (community_size - chief_count).choose supporting_chief_count *
  (community_size - chief_count - supporting_chief_count).choose senior_officer_count *
  (community_size - chief_count - supporting_chief_count - senior_officer_count).choose inferior_officer_count

theorem leadership_assignment_theorem :
  leadership_assignment_count = 498960 := by
  sorry

end NUMINAMATH_CALUDE_leadership_assignment_theorem_l104_10470


namespace NUMINAMATH_CALUDE_conjugate_2023_l104_10473

/-- Conjugate point in 2D space -/
def conjugate (p : ℝ × ℝ) : ℝ × ℝ :=
  (-p.2 + 1, p.1 + 1)

/-- Sequence of conjugate points -/
def conjugateSequence : ℕ → ℝ × ℝ
  | 0 => (2, 2)
  | n + 1 => conjugate (conjugateSequence n)

theorem conjugate_2023 :
  conjugateSequence 2023 = (-2, 0) := by sorry

end NUMINAMATH_CALUDE_conjugate_2023_l104_10473


namespace NUMINAMATH_CALUDE_watch_cost_price_proof_l104_10436

/-- The cost price of a watch satisfying certain selling conditions -/
def watch_cost_price : ℝ := 1400

/-- The selling price at a 10% loss -/
def selling_price_loss (cost : ℝ) : ℝ := cost * 0.9

/-- The selling price at a 4% gain -/
def selling_price_gain (cost : ℝ) : ℝ := cost * 1.04

theorem watch_cost_price_proof :
  (selling_price_gain watch_cost_price - selling_price_loss watch_cost_price = 196) ∧
  (watch_cost_price = 1400) := by
  sorry

end NUMINAMATH_CALUDE_watch_cost_price_proof_l104_10436


namespace NUMINAMATH_CALUDE_initial_subscribers_count_l104_10465

/-- Represents the monthly income of a streamer based on their number of subscribers -/
def streamer_income (initial_subscribers : ℕ) (gift_subscribers : ℕ) (income_per_subscriber : ℕ) : ℕ :=
  (initial_subscribers + gift_subscribers) * income_per_subscriber

/-- Proves that the initial number of subscribers is 150 given the problem conditions -/
theorem initial_subscribers_count :
  ∃ (x : ℕ), streamer_income x 50 9 = 1800 ∧ x = 150 := by
  sorry

end NUMINAMATH_CALUDE_initial_subscribers_count_l104_10465


namespace NUMINAMATH_CALUDE_smallest_number_of_blocks_l104_10429

/-- Represents the dimensions of a wall --/
structure WallDimensions where
  length : ℕ
  height : ℕ

/-- Represents the dimensions of a block --/
structure BlockDimensions where
  length : ℚ
  height : ℕ

/-- Calculates the number of blocks needed for a wall with given conditions --/
def calculateBlocksNeeded (wall : WallDimensions) (blockHeight : ℕ) (evenRowBlocks : ℕ) (oddRowBlocks : ℕ) : ℕ :=
  let oddRows := (wall.height + 1) / 2
  let evenRows := wall.height / 2
  oddRows * oddRowBlocks + evenRows * evenRowBlocks

/-- Theorem stating the smallest number of blocks needed for the wall --/
theorem smallest_number_of_blocks 
  (wall : WallDimensions)
  (blockHeight : ℕ)
  (block2ft : BlockDimensions)
  (block1_5ft : BlockDimensions)
  (block1ft : BlockDimensions)
  (h1 : wall.length = 120)
  (h2 : wall.height = 7)
  (h3 : blockHeight = 1)
  (h4 : block2ft.length = 2)
  (h5 : block1_5ft.length = 3/2)
  (h6 : block1ft.length = 1)
  (h7 : block2ft.height = blockHeight)
  (h8 : block1_5ft.height = blockHeight)
  (h9 : block1ft.height = blockHeight) :
  calculateBlocksNeeded wall blockHeight 61 60 = 423 :=
sorry

end NUMINAMATH_CALUDE_smallest_number_of_blocks_l104_10429


namespace NUMINAMATH_CALUDE_complete_square_integer_l104_10401

theorem complete_square_integer (y : ℝ) : ∃ k : ℤ, y^2 + 12*y + 40 = (y + 6)^2 + k := by
  sorry

end NUMINAMATH_CALUDE_complete_square_integer_l104_10401


namespace NUMINAMATH_CALUDE_sqrt_x_minus_2_defined_l104_10406

theorem sqrt_x_minus_2_defined (x : ℝ) : 
  ∃ y : ℝ, y ^ 2 = x - 2 ↔ x ≥ 2 := by sorry

end NUMINAMATH_CALUDE_sqrt_x_minus_2_defined_l104_10406


namespace NUMINAMATH_CALUDE_system_consistency_l104_10434

/-- The system of equations is consistent if and only if a is 0, -2, or 54 -/
theorem system_consistency (x a : ℝ) : 
  (∃ x, (10 * x^2 + x - a - 11 = 0) ∧ (4 * x^2 + (a + 4) * x - 3 * a - 8 = 0)) ↔ 
  (a = 0 ∨ a = -2 ∨ a = 54) :=
by sorry

end NUMINAMATH_CALUDE_system_consistency_l104_10434


namespace NUMINAMATH_CALUDE_kate_change_l104_10493

/-- The amount Kate gave to the clerk in cents -/
def amount_given : ℕ := 100

/-- The cost of Kate's candy in cents -/
def candy_cost : ℕ := 54

/-- The change Kate should receive in cents -/
def change : ℕ := amount_given - candy_cost

/-- Theorem stating that Kate should receive 46 cents in change -/
theorem kate_change : change = 46 := by sorry

end NUMINAMATH_CALUDE_kate_change_l104_10493


namespace NUMINAMATH_CALUDE_krista_savings_exceed_target_l104_10419

/-- The sum of the first n terms of a geometric series with first term a and common ratio r -/
def geometricSum (a : ℚ) (r : ℚ) (n : ℕ) : ℚ :=
  a * (r^n - 1) / (r - 1)

/-- The first day Krista deposits money -/
def initialDeposit : ℚ := 3

/-- The ratio by which Krista increases her deposit each day -/
def depositRatio : ℚ := 3

/-- The amount Krista wants to exceed in cents -/
def targetAmount : ℚ := 2000

theorem krista_savings_exceed_target :
  (∀ k < 7, geometricSum initialDeposit depositRatio k ≤ targetAmount) ∧
  geometricSum initialDeposit depositRatio 7 > targetAmount :=
sorry

end NUMINAMATH_CALUDE_krista_savings_exceed_target_l104_10419


namespace NUMINAMATH_CALUDE_orange_trees_l104_10484

theorem orange_trees (total_fruits : ℕ) (fruits_per_tree : ℕ) (remaining_ratio : ℚ) : 
  total_fruits = 960 →
  fruits_per_tree = 200 →
  remaining_ratio = 3/5 →
  (total_fruits : ℚ) / (remaining_ratio * fruits_per_tree) = 8 :=
by sorry

end NUMINAMATH_CALUDE_orange_trees_l104_10484


namespace NUMINAMATH_CALUDE_positive_root_of_cubic_l104_10494

theorem positive_root_of_cubic (x : ℝ) :
  x = 2 + Real.sqrt 3 →
  x > 0 ∧ x^3 - 4*x^2 - 2*x - Real.sqrt 3 = 0 :=
by sorry

end NUMINAMATH_CALUDE_positive_root_of_cubic_l104_10494


namespace NUMINAMATH_CALUDE_expression_evaluation_l104_10423

theorem expression_evaluation (m n : ℤ) (hm : m = -1) (hn : n = 2) :
  3 * m^2 * n - 2 * m * n^2 - 4 * m^2 * n + m * n^2 = 2 := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l104_10423


namespace NUMINAMATH_CALUDE_power_boat_travel_time_l104_10412

/-- The time it takes for the power boat to travel from A to B -/
def travel_time_AB : ℝ := 4

/-- The distance between dock A and dock B in km -/
def distance_AB : ℝ := 20

/-- The original speed of the river current -/
def river_speed : ℝ := sorry

/-- The speed of the power boat relative to the river -/
def boat_speed : ℝ := sorry

/-- The total time of the journey in hours -/
def total_time : ℝ := 12

theorem power_boat_travel_time :
  let increased_river_speed := 1.5 * river_speed
  let downstream_speed := boat_speed + river_speed
  let upstream_speed := boat_speed - increased_river_speed
  distance_AB / downstream_speed = travel_time_AB ∧
  distance_AB + upstream_speed * (total_time - travel_time_AB) = river_speed * total_time :=
by sorry

end NUMINAMATH_CALUDE_power_boat_travel_time_l104_10412


namespace NUMINAMATH_CALUDE_largest_integer_negative_quadratic_l104_10485

theorem largest_integer_negative_quadratic :
  ∀ n : ℤ, n^2 - 11*n + 28 < 0 → n ≤ 6 :=
by sorry

end NUMINAMATH_CALUDE_largest_integer_negative_quadratic_l104_10485

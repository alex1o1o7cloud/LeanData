import Mathlib

namespace NUMINAMATH_CALUDE_dining_bill_share_l1090_109060

theorem dining_bill_share (total_bill : ℚ) (num_people : ℕ) (tip_percentage : ℚ) :
  total_bill = 139 ∧ num_people = 7 ∧ tip_percentage = 1/10 →
  (total_bill + total_bill * tip_percentage) / num_people = 2184/100 := by
  sorry

end NUMINAMATH_CALUDE_dining_bill_share_l1090_109060


namespace NUMINAMATH_CALUDE_intersection_slopes_sum_l1090_109022

/-- Given a line y = 2x - 3 and a parabola y² = 4x intersecting at points A and B,
    with O as the origin and k₁, k₂ as the slopes of OA and OB respectively,
    prove that the sum of the reciprocals of the slopes 1/k₁ + 1/k₂ = 1/2 -/
theorem intersection_slopes_sum (A B : ℝ × ℝ) (k₁ k₂ : ℝ) : 
  (A.2 = 2 * A.1 - 3) →
  (B.2 = 2 * B.1 - 3) →
  (A.2^2 = 4 * A.1) →
  (B.2^2 = 4 * B.1) →
  (k₁ = A.2 / A.1) →
  (k₂ = B.2 / B.1) →
  (1 / k₁ + 1 / k₂ = 1 / 2) := by
  sorry

end NUMINAMATH_CALUDE_intersection_slopes_sum_l1090_109022


namespace NUMINAMATH_CALUDE_jim_future_age_l1090_109030

/-- Tom's current age -/
def tom_current_age : ℕ := 37

/-- Tom's age 7 years ago -/
def tom_age_7_years_ago : ℕ := tom_current_age - 7

/-- Jim's age 7 years ago -/
def jim_age_7_years_ago : ℕ := tom_age_7_years_ago / 2 + 5

/-- Jim's current age -/
def jim_current_age : ℕ := jim_age_7_years_ago + 7

/-- Jim's age in 2 years -/
def jim_age_in_2_years : ℕ := jim_current_age + 2

theorem jim_future_age :
  tom_current_age = 37 →
  tom_age_7_years_ago = 30 →
  jim_age_7_years_ago = 20 →
  jim_current_age = 27 →
  jim_age_in_2_years = 29 := by
  sorry

end NUMINAMATH_CALUDE_jim_future_age_l1090_109030


namespace NUMINAMATH_CALUDE_ice_cream_box_problem_l1090_109033

/-- The number of ice cream bars in a box -/
def bars_per_box : ℕ := 3

/-- The cost of a box of ice cream bars in dollars -/
def box_cost : ℚ := 15/2

/-- The number of friends -/
def num_friends : ℕ := 6

/-- The number of bars each friend wants -/
def bars_per_friend : ℕ := 2

/-- The cost per person in dollars -/
def cost_per_person : ℚ := 5

theorem ice_cream_box_problem :
  bars_per_box = (num_friends * bars_per_friend) / ((num_friends * cost_per_person) / box_cost) :=
by sorry

end NUMINAMATH_CALUDE_ice_cream_box_problem_l1090_109033


namespace NUMINAMATH_CALUDE_equality_of_cyclic_powers_l1090_109035

theorem equality_of_cyclic_powers (x y z : ℝ) 
  (hx : x > 1) (hy : y > 1) (hz : z > 1)
  (h_eq1 : x^y = y^z) (h_eq2 : y^z = z^x) : x = y ∧ y = z :=
sorry

end NUMINAMATH_CALUDE_equality_of_cyclic_powers_l1090_109035


namespace NUMINAMATH_CALUDE_school_cinema_visit_payment_l1090_109063

/-- Represents the ticket pricing structure and student count for a cinema visit -/
structure CinemaVisit where
  individual_price : ℝ  -- Price of an individual ticket
  group_price : ℝ       -- Price of a group ticket for 10 people
  student_discount : ℝ  -- Discount rate for students (as a decimal)
  student_count : ℕ     -- Number of students

/-- Calculates the minimum amount to be paid for a school cinema visit -/
def minimum_payment (cv : CinemaVisit) : ℝ :=
  let group_size := 10
  let full_groups := cv.student_count / group_size
  let total_group_price := (full_groups * cv.group_price) * (1 - cv.student_discount)
  total_group_price

/-- The theorem stating the minimum payment for the given scenario -/
theorem school_cinema_visit_payment :
  let cv : CinemaVisit := {
    individual_price := 6,
    group_price := 40,
    student_discount := 0.1,
    student_count := 1258
  }
  minimum_payment cv = 4536 := by sorry

end NUMINAMATH_CALUDE_school_cinema_visit_payment_l1090_109063


namespace NUMINAMATH_CALUDE_multiplication_puzzle_l1090_109057

def is_digit (n : ℕ) : Prop := n ≥ 0 ∧ n ≤ 9

theorem multiplication_puzzle (a b : ℕ) (ha : is_digit a) (hb : is_digit b) 
  (h_mult : (30 + a) * (10 * b + 4) = 156) : a + b = 9 := by
  sorry

end NUMINAMATH_CALUDE_multiplication_puzzle_l1090_109057


namespace NUMINAMATH_CALUDE_car_cost_calculation_l1090_109016

/-- The total cost of a car given an initial payment, monthly payment, and number of months -/
theorem car_cost_calculation (initial_payment monthly_payment num_months : ℕ) :
  initial_payment = 5400 →
  monthly_payment = 420 →
  num_months = 19 →
  initial_payment + monthly_payment * num_months = 13380 := by
  sorry

end NUMINAMATH_CALUDE_car_cost_calculation_l1090_109016


namespace NUMINAMATH_CALUDE_repair_shop_earnings_l1090_109024

/-- Calculates the total earnings for a repair shop after applying discounts and taxes. -/
def totalEarnings (
  phoneRepairCost : ℚ)
  (laptopRepairCost : ℚ)
  (computerRepairCost : ℚ)
  (tabletRepairCost : ℚ)
  (smartwatchRepairCost : ℚ)
  (phoneRepairs : ℕ)
  (laptopRepairs : ℕ)
  (computerRepairs : ℕ)
  (tabletRepairs : ℕ)
  (smartwatchRepairs : ℕ)
  (computerRepairDiscount : ℚ)
  (salesTaxRate : ℚ) : ℚ :=
  sorry

theorem repair_shop_earnings :
  totalEarnings 11 15 18 12 8 9 5 4 6 8 (1/10) (1/20) = 393.54 := by
  sorry

end NUMINAMATH_CALUDE_repair_shop_earnings_l1090_109024


namespace NUMINAMATH_CALUDE_smallest_two_digit_prime_with_composite_reverse_l1090_109050

/-- A function that reverses the digits of a two-digit number -/
def reverse_digits (n : ℕ) : ℕ :=
  (n % 10) * 10 + (n / 10)

/-- A predicate that checks if a number is a two-digit number -/
def is_two_digit (n : ℕ) : Prop :=
  10 ≤ n ∧ n < 100

/-- The main theorem -/
theorem smallest_two_digit_prime_with_composite_reverse :
  ∀ p : ℕ, is_two_digit p → Nat.Prime p →
    (∀ q : ℕ, is_two_digit q → Nat.Prime q → q < p →
      Nat.Prime (reverse_digits q)) →
    ¬Nat.Prime (reverse_digits p) →
    p = 19 := by
  sorry

end NUMINAMATH_CALUDE_smallest_two_digit_prime_with_composite_reverse_l1090_109050


namespace NUMINAMATH_CALUDE_even_digits_in_512_base_8_l1090_109003

/-- Represents a natural number in base 8 as a list of digits -/
def BaseEightRepresentation : Type := List Nat

/-- Converts a natural number to its base-8 representation -/
def toBaseEight (n : Nat) : BaseEightRepresentation :=
  sorry

/-- Counts the number of even digits in a base-8 representation -/
def countEvenDigits (rep : BaseEightRepresentation) : Nat :=
  sorry

theorem even_digits_in_512_base_8 :
  countEvenDigits (toBaseEight 512) = 3 := by
  sorry

end NUMINAMATH_CALUDE_even_digits_in_512_base_8_l1090_109003


namespace NUMINAMATH_CALUDE_photo_collection_l1090_109028

theorem photo_collection (total : ℕ) (tim_less : ℕ) (paul_more : ℕ) : 
  total = 152 →
  tim_less = 100 →
  paul_more = 10 →
  ∃ (tim paul tom : ℕ), 
    tim = total - tim_less ∧
    paul = tim + paul_more ∧
    tom = total - (tim + paul) ∧
    tom = 38 := by
  sorry

end NUMINAMATH_CALUDE_photo_collection_l1090_109028


namespace NUMINAMATH_CALUDE_geometric_sequence_problem_l1090_109068

/-- 
A geometric sequence is defined by its first term and common ratio.
This theorem proves that for a geometric sequence where the third term is 18
and the sixth term is 162, the first term is 2 and the common ratio is 3.
-/
theorem geometric_sequence_problem (a r : ℝ) : 
  a * r^2 = 18 → a * r^5 = 162 → a = 2 ∧ r = 3 := by
sorry

end NUMINAMATH_CALUDE_geometric_sequence_problem_l1090_109068


namespace NUMINAMATH_CALUDE_midpoint_trajectory_l1090_109000

/-- The trajectory of the midpoint of a perpendicular from a point on the unit circle to the x-axis -/
theorem midpoint_trajectory (a b x y : ℝ) : 
  a^2 + b^2 = 1 →  -- P(a, b) is on the unit circle
  x = a →          -- x-coordinate of M is same as P
  y = b / 2 →      -- y-coordinate of M is half of P's
  x^2 + 4 * y^2 = 1 := by
sorry

end NUMINAMATH_CALUDE_midpoint_trajectory_l1090_109000


namespace NUMINAMATH_CALUDE_complex_magnitude_l1090_109081

theorem complex_magnitude (z : ℂ) (h : z * (1 + Complex.I) = 2) : Complex.abs z = Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_complex_magnitude_l1090_109081


namespace NUMINAMATH_CALUDE_squirrel_travel_time_l1090_109073

-- Define the speed of the squirrel in miles per hour
def speed : ℝ := 4

-- Define the distance to be traveled in miles
def distance : ℝ := 1

-- Define the conversion factor from hours to minutes
def minutes_per_hour : ℝ := 60

-- Theorem statement
theorem squirrel_travel_time :
  (distance / speed) * minutes_per_hour = 15 := by
  sorry

end NUMINAMATH_CALUDE_squirrel_travel_time_l1090_109073


namespace NUMINAMATH_CALUDE_fraction_multiplication_l1090_109034

theorem fraction_multiplication : (1 / 2) * (3 / 5) * (7 / 11) * (4 / 13) = 84 / 1430 := by
  sorry

end NUMINAMATH_CALUDE_fraction_multiplication_l1090_109034


namespace NUMINAMATH_CALUDE_average_weight_proof_l1090_109005

theorem average_weight_proof (rachel_weight jimmy_weight adam_weight : ℝ) : 
  rachel_weight = 75 ∧ 
  rachel_weight = jimmy_weight - 6 ∧ 
  rachel_weight = adam_weight + 15 → 
  (rachel_weight + jimmy_weight + adam_weight) / 3 = 72 := by
sorry

end NUMINAMATH_CALUDE_average_weight_proof_l1090_109005


namespace NUMINAMATH_CALUDE_stock_change_theorem_l1090_109017

theorem stock_change_theorem (x : ℝ) (h : x > 0) :
  let day1_value := x * (1 - 0.3)
  let day2_value := day1_value * (1 + 0.5)
  (day2_value - x) / x = 0.05 := by sorry

end NUMINAMATH_CALUDE_stock_change_theorem_l1090_109017


namespace NUMINAMATH_CALUDE_divisors_sum_8_implies_one_zero_l1090_109041

def has_three_smallest_distinct_divisors_sum_8 (A : ℕ+) : Prop :=
  ∃ d₁ d₂ d₃ : ℕ+, 
    d₁ < d₂ ∧ d₂ < d₃ ∧
    d₁ ∣ A ∧ d₂ ∣ A ∧ d₃ ∣ A ∧
    d₁.val + d₂.val + d₃.val = 8 ∧
    ∀ d : ℕ+, d ∣ A → d ≤ d₃ → d = d₁ ∨ d = d₂ ∨ d = d₃

def ends_with_one_zero (A : ℕ+) : Prop :=
  ∃ k : ℕ, A.val = 10 * k ∧ k % 10 ≠ 0

theorem divisors_sum_8_implies_one_zero (A : ℕ+) :
  has_three_smallest_distinct_divisors_sum_8 A → ends_with_one_zero A :=
sorry

end NUMINAMATH_CALUDE_divisors_sum_8_implies_one_zero_l1090_109041


namespace NUMINAMATH_CALUDE_lemming_average_distance_l1090_109058

/-- Given a square with side length 12, prove that a point moving 7 units along
    the diagonal from a corner and then 4 units perpendicular to the diagonal
    results in an average distance of 6 units to each side of the square. -/
theorem lemming_average_distance (square_side : ℝ) (diag_move : ℝ) (perp_move : ℝ) :
  square_side = 12 →
  diag_move = 7 →
  perp_move = 4 →
  let diag := square_side * Real.sqrt 2
  let diag_ratio := diag_move / diag
  let x := diag_ratio * square_side
  let y := diag_ratio * square_side
  let final_x := x + perp_move
  let final_y := y
  let dist_left := final_x
  let dist_bottom := final_y
  let dist_right := square_side - final_x
  let dist_top := square_side - final_y
  let avg_dist := (dist_left + dist_bottom + dist_right + dist_top) / 4
  avg_dist = 6 := by
sorry

end NUMINAMATH_CALUDE_lemming_average_distance_l1090_109058


namespace NUMINAMATH_CALUDE_split_investment_average_rate_l1090_109099

/-- The average interest rate for a split investment --/
theorem split_investment_average_rate (total_investment : ℝ) 
  (rate1 rate2 : ℝ) (fee : ℝ) : 
  total_investment > 0 →
  rate1 > 0 →
  rate2 > 0 →
  rate1 < rate2 →
  fee > 0 →
  ∃ (x : ℝ), 
    0 < x ∧ 
    x < total_investment ∧
    rate1 * (total_investment - x) - fee = rate2 * x →
  (rate2 * x + (rate1 * (total_investment - x) - fee)) / total_investment = 0.05133 :=
by sorry

end NUMINAMATH_CALUDE_split_investment_average_rate_l1090_109099


namespace NUMINAMATH_CALUDE_celias_budget_savings_percentage_l1090_109027

/-- Celia's budget problem --/
theorem celias_budget_savings_percentage :
  let weeks : ℕ := 4
  let food_budget_per_week : ℕ := 100
  let rent : ℕ := 1500
  let streaming_cost : ℕ := 30
  let phone_cost : ℕ := 50
  let savings : ℕ := 198
  let total_spending : ℕ := weeks * food_budget_per_week + rent + streaming_cost + phone_cost
  let savings_percentage : ℚ := (savings : ℚ) / (total_spending : ℚ) * 100
  savings_percentage = 10 := by sorry

end NUMINAMATH_CALUDE_celias_budget_savings_percentage_l1090_109027


namespace NUMINAMATH_CALUDE_pure_imaginary_condition_l1090_109006

theorem pure_imaginary_condition (a : ℝ) : 
  (Complex.I * (a^2 - a - 2) = (a^2 - 2*a) + Complex.I * (a^2 - a - 2)) → a = 0 := by
  sorry

end NUMINAMATH_CALUDE_pure_imaginary_condition_l1090_109006


namespace NUMINAMATH_CALUDE_product_of_divisors_60_l1090_109037

-- Define the number we're working with
def n : ℕ := 60

-- Define a function to get all divisors of a natural number
def divisors (m : ℕ) : Finset ℕ :=
  sorry

-- Define the product of all divisors
def product_of_divisors (m : ℕ) : ℕ :=
  (divisors m).prod id

-- Theorem statement
theorem product_of_divisors_60 :
  product_of_divisors n = 46656000000000 := by
  sorry

end NUMINAMATH_CALUDE_product_of_divisors_60_l1090_109037


namespace NUMINAMATH_CALUDE_charging_bull_rounds_in_hour_l1090_109059

/-- The time in seconds for the racing magic to complete one round -/
def racing_magic_time : ℕ := 60

/-- The time in minutes it takes for both to meet at the starting point for the second time -/
def meeting_time : ℕ := 6

/-- The number of rounds the racing magic completes in the meeting time -/
def racing_magic_rounds : ℕ := meeting_time

/-- The number of rounds the charging bull completes in the meeting time -/
def charging_bull_rounds : ℕ := racing_magic_rounds + 1

/-- The number of minutes in an hour -/
def minutes_per_hour : ℕ := 60

/-- The number of rounds the charging bull makes in an hour -/
def charging_bull_hourly_rounds : ℕ := 
  (charging_bull_rounds * minutes_per_hour) / meeting_time

theorem charging_bull_rounds_in_hour : 
  charging_bull_hourly_rounds = 70 := by sorry

end NUMINAMATH_CALUDE_charging_bull_rounds_in_hour_l1090_109059


namespace NUMINAMATH_CALUDE_complement_of_union_l1090_109093

def U : Set Int := {-2, -1, 0, 1, 2, 3}
def A : Set Int := {-1, 0, 1}
def B : Set Int := {1, 2}

theorem complement_of_union :
  (U \ (A ∪ B)) = {-2, 3} := by sorry

end NUMINAMATH_CALUDE_complement_of_union_l1090_109093


namespace NUMINAMATH_CALUDE_zorg_vamp_and_not_wook_l1090_109084

-- Define the types
variable (U : Type) -- Universe set
variable (Zorg Xyon Wook Vamp : Set U)

-- Define the conditions
variable (h1 : Zorg ⊆ Xyon)
variable (h2 : Wook ⊆ Xyon)
variable (h3 : Vamp ⊆ Zorg)
variable (h4 : Wook ⊆ Vamp)
variable (h5 : Zorg ∩ Wook = ∅)

-- Theorem to prove
theorem zorg_vamp_and_not_wook : 
  Zorg ⊆ Vamp ∧ Zorg ∩ Wook = ∅ :=
by sorry

end NUMINAMATH_CALUDE_zorg_vamp_and_not_wook_l1090_109084


namespace NUMINAMATH_CALUDE_pucks_not_in_original_position_l1090_109020

/-- Represents the arrangement of three pucks -/
inductive Arrangement
  | ABC
  | ACB
  | BAC
  | BCA
  | CAB
  | CBA

/-- Represents a single swap operation -/
def swap : Arrangement → Arrangement
  | Arrangement.ABC => Arrangement.BAC
  | Arrangement.ACB => Arrangement.CAB
  | Arrangement.BAC => Arrangement.ABC
  | Arrangement.BCA => Arrangement.CBA
  | Arrangement.CAB => Arrangement.ACB
  | Arrangement.CBA => Arrangement.BCA

/-- Applies n swaps to the initial arrangement -/
def applySwaps (n : Nat) (init : Arrangement) : Arrangement :=
  match n with
  | 0 => init
  | n + 1 => swap (applySwaps n init)

/-- Theorem stating that after 25 swaps, the arrangement cannot be the same as the initial one -/
theorem pucks_not_in_original_position (init : Arrangement) : 
  applySwaps 25 init ≠ init :=
sorry

end NUMINAMATH_CALUDE_pucks_not_in_original_position_l1090_109020


namespace NUMINAMATH_CALUDE_right_triangle_hypotenuse_l1090_109008

theorem right_triangle_hypotenuse (a b c : ℝ) : 
  a > 0 → b > 0 → c > 0 →
  a^2 + b^2 = c^2 →  -- right-angled triangle condition
  a^2 + b^2 + c^2 = 2500 →  -- sum of squares condition
  c = 25 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_right_triangle_hypotenuse_l1090_109008


namespace NUMINAMATH_CALUDE_score_ordering_l1090_109062

-- Define the set of people
inductive Person : Type
| M : Person  -- Marty
| Q : Person  -- Quay
| S : Person  -- Shana
| Z : Person  -- Zane
| K : Person  -- Kaleana

-- Define a function to represent the score of each person
variable (score : Person → ℕ)

-- Define the conditions from the problem
def marty_condition (score : Person → ℕ) : Prop :=
  ∃ p : Person, score Person.M > score p

def quay_condition (score : Person → ℕ) : Prop :=
  score Person.Q = score Person.K

def shana_condition (score : Person → ℕ) : Prop :=
  ∃ p : Person, score Person.S < score p

def zane_condition (score : Person → ℕ) : Prop :=
  (score Person.Z < score Person.S) ∨ (score Person.Z > score Person.M)

-- Theorem statement
theorem score_ordering (score : Person → ℕ) :
  marty_condition score →
  quay_condition score →
  shana_condition score →
  zane_condition score →
  (score Person.Z < score Person.S) ∧
  (score Person.S < score Person.Q) ∧
  (score Person.Q < score Person.M) :=
sorry

end NUMINAMATH_CALUDE_score_ordering_l1090_109062


namespace NUMINAMATH_CALUDE_alice_bob_meet_l1090_109056

/-- The number of points on the circle -/
def n : ℕ := 18

/-- Alice's clockwise movement per turn -/
def alice_move : ℕ := 7

/-- Bob's counterclockwise movement per turn -/
def bob_move : ℕ := 11

/-- Bob's extra skip every second turn -/
def bob_extra : ℕ := 1

/-- The number of turns after which Alice and Bob meet -/
def meeting_turns : ℕ := 36

/-- Function to calculate position on the circle after a number of moves -/
def position (start : ℕ) (moves : ℕ) : ℕ :=
  (start + moves - 1) % n + 1

/-- Alice's position after a given number of turns -/
def alice_position (turns : ℕ) : ℕ :=
  position n (alice_move * turns)

/-- Bob's position after a given number of turns -/
def bob_position (turns : ℕ) : ℕ :=
  position n (n * turns - bob_move * turns - bob_extra * (turns / 2))

/-- Theorem stating that Alice and Bob meet after the specified number of turns -/
theorem alice_bob_meet : alice_position meeting_turns = bob_position meeting_turns := by
  sorry


end NUMINAMATH_CALUDE_alice_bob_meet_l1090_109056


namespace NUMINAMATH_CALUDE_x_value_implies_y_value_l1090_109080

theorem x_value_implies_y_value :
  let x := (Real.sqrt 27 + Real.sqrt 243) / Real.sqrt 48
  x^3 - 2*x^2 + Real.sin (2*Real.pi*x) - Real.cos (Real.pi*x) = 10 := by
  sorry

end NUMINAMATH_CALUDE_x_value_implies_y_value_l1090_109080


namespace NUMINAMATH_CALUDE_inverse_cube_squared_l1090_109014

theorem inverse_cube_squared : (3⁻¹)^2 = (1 : ℚ) / 9 := by sorry

end NUMINAMATH_CALUDE_inverse_cube_squared_l1090_109014


namespace NUMINAMATH_CALUDE_least_divisible_by_1_to_9_halved_l1090_109070

def is_divisible_by_range (n : ℕ) (a b : ℕ) : Prop :=
  ∀ k : ℕ, a ≤ k → k ≤ b → k ∣ n

theorem least_divisible_by_1_to_9_halved :
  ∃ l : ℕ, (∀ m : ℕ, is_divisible_by_range m 1 9 → l ≤ m) ∧
           is_divisible_by_range l 1 9 ∧
           l / 2 = 1260 := by
  sorry

end NUMINAMATH_CALUDE_least_divisible_by_1_to_9_halved_l1090_109070


namespace NUMINAMATH_CALUDE_right_triangle_sets_l1090_109066

def is_right_triangle (a b c : ℝ) : Prop :=
  a > 0 ∧ b > 0 ∧ c > 0 ∧ a^2 + b^2 = c^2

theorem right_triangle_sets :
  ¬(is_right_triangle 6 7 8) ∧
  ¬(is_right_triangle 1 (Real.sqrt 2) 5) ∧
  is_right_triangle 6 8 10 ∧
  ¬(is_right_triangle (Real.sqrt 5) (2 * Real.sqrt 3) (Real.sqrt 15)) :=
by sorry

end NUMINAMATH_CALUDE_right_triangle_sets_l1090_109066


namespace NUMINAMATH_CALUDE_lines_intersect_at_point_l1090_109083

/-- Represents a 2D point --/
structure Point2D where
  x : ℚ
  y : ℚ

/-- Represents a line parameterization --/
structure LineParam where
  base : Point2D
  direction : Point2D

/-- First line parameterization --/
def line1 : LineParam := {
  base := { x := 2, y := 3 },
  direction := { x := 1, y := -4 }
}

/-- Second line parameterization --/
def line2 : LineParam := {
  base := { x := 4, y := -6 },
  direction := { x := 5, y := 3 }
}

/-- The intersection point of the two lines --/
def intersection_point : Point2D := {
  x := 185 / 23,
  y := 21 / 23
}

/-- Theorem stating that the given point is the intersection of the two lines --/
theorem lines_intersect_at_point :
  ∃ (t u : ℚ),
    (line1.base.x + t * line1.direction.x = intersection_point.x) ∧
    (line1.base.y + t * line1.direction.y = intersection_point.y) ∧
    (line2.base.x + u * line2.direction.x = intersection_point.x) ∧
    (line2.base.y + u * line2.direction.y = intersection_point.y) := by
  sorry

end NUMINAMATH_CALUDE_lines_intersect_at_point_l1090_109083


namespace NUMINAMATH_CALUDE_rectangular_solid_volume_l1090_109075

-- Define the rectangular solid
structure RectangularSolid where
  x : ℝ
  y : ℝ
  z : ℝ

-- Define the theorem
theorem rectangular_solid_volume 
  (r : RectangularSolid) 
  (h1 : r.x * r.y = 15) 
  (h2 : r.y * r.z = 10) 
  (h3 : r.x * r.z = 6) 
  (h4 : r.x = 3 * r.y) : 
  r.x * r.y * r.z = 6 * Real.sqrt 5 := by
  sorry


end NUMINAMATH_CALUDE_rectangular_solid_volume_l1090_109075


namespace NUMINAMATH_CALUDE_parallelepiped_properties_l1090_109095

/-- Represents a parallelepiped with given properties -/
structure Parallelepiped where
  height : ℝ
  lateral_edge_projection : ℝ
  rhombus_area : ℝ
  rhombus_diagonal : ℝ

/-- Calculate the lateral surface area of the parallelepiped -/
def lateral_surface_area (p : Parallelepiped) : ℝ := sorry

/-- Calculate the volume of the parallelepiped -/
def volume (p : Parallelepiped) : ℝ := sorry

/-- Theorem stating the properties of the specific parallelepiped -/
theorem parallelepiped_properties (p : Parallelepiped) 
  (h1 : p.height = 12)
  (h2 : p.lateral_edge_projection = 5)
  (h3 : p.rhombus_area = 24)
  (h4 : p.rhombus_diagonal = 8) : 
  lateral_surface_area p = 260 ∧ volume p = 312 := by sorry

end NUMINAMATH_CALUDE_parallelepiped_properties_l1090_109095


namespace NUMINAMATH_CALUDE_divisibility_condition_l1090_109079

theorem divisibility_condition (a b : ℕ+) :
  (a * b^2 + b + 7 ∣ a^2 * b + a + b) ↔
  ((a = 11 ∧ b = 1) ∨ (a = 49 ∧ b = 1) ∨ (∃ k : ℕ+, a = 7 * k^2 ∧ b = 7 * k)) :=
sorry

end NUMINAMATH_CALUDE_divisibility_condition_l1090_109079


namespace NUMINAMATH_CALUDE_valid_arrangements_l1090_109011

/-- Represents the number of ways to arrange 7 people in a line -/
def arrangement_count : ℕ := 72

/-- Represents the total number of people -/
def total_people : ℕ := 7

/-- Represents the number of students -/
def student_count : ℕ := 6

/-- Represents whether two people are at the ends of the line -/
def are_at_ends (coach : ℕ) (student_a : ℕ) : Prop :=
  (coach = 1 ∧ student_a = total_people) ∨ (coach = total_people ∧ student_a = 1)

/-- Represents whether two students are adjacent in the line -/
def are_adjacent (student1 : ℕ) (student2 : ℕ) : Prop :=
  student1 + 1 = student2 ∨ student2 + 1 = student1

/-- Represents whether two students are not adjacent in the line -/
def are_not_adjacent (student1 : ℕ) (student2 : ℕ) : Prop :=
  ¬(are_adjacent student1 student2)

/-- Theorem stating that the number of valid arrangements is 72 -/
theorem valid_arrangements :
  ∀ (coach student_a student_b student_c student_d : ℕ),
    are_at_ends coach student_a →
    are_adjacent student_b student_c →
    are_not_adjacent student_b student_d →
    arrangement_count = 72 := by sorry

end NUMINAMATH_CALUDE_valid_arrangements_l1090_109011


namespace NUMINAMATH_CALUDE_polynomial_remainder_theorem_l1090_109044

theorem polynomial_remainder_theorem (x : ℝ) : 
  (x^3 - 5*x^2 + 3*x - 7) % (x - 3) = -16 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_remainder_theorem_l1090_109044


namespace NUMINAMATH_CALUDE_inserted_numbers_sum_l1090_109042

theorem inserted_numbers_sum (x y : ℝ) : 
  10 < x ∧ x < y ∧ y < 39 ∧  -- x and y are between 10 and 39
  (x / 10 = y / x) ∧         -- 10, x, y form a geometric sequence
  (y - x = 39 - y) →         -- x, y, 39 form an arithmetic sequence
  x + y = 11.25 :=           -- sum of x and y is 11¼
by sorry

end NUMINAMATH_CALUDE_inserted_numbers_sum_l1090_109042


namespace NUMINAMATH_CALUDE_water_added_calculation_l1090_109055

def initial_volume : ℝ := 340
def water_percentage : ℝ := 0.88
def cola_percentage : ℝ := 0.05
def sugar_percentage : ℝ := 1 - water_percentage - cola_percentage
def added_sugar : ℝ := 3.2
def added_cola : ℝ := 6.8
def final_sugar_percentage : ℝ := 0.075

theorem water_added_calculation (water_added : ℝ) : 
  (sugar_percentage * initial_volume + added_sugar) / 
  (initial_volume + added_sugar + added_cola + water_added) = final_sugar_percentage → 
  water_added = 10 := by
  sorry

end NUMINAMATH_CALUDE_water_added_calculation_l1090_109055


namespace NUMINAMATH_CALUDE_odd_function_negative_domain_l1090_109004

def is_odd (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = -f x

theorem odd_function_negative_domain 
  (f : ℝ → ℝ) 
  (h_odd : is_odd f) 
  (h_nonneg : ∀ x ≥ 0, f x = x * (x + 1)) :
  ∀ x < 0, f x = x * (-x + 1) := by
sorry

end NUMINAMATH_CALUDE_odd_function_negative_domain_l1090_109004


namespace NUMINAMATH_CALUDE_min_value_theorem_l1090_109019

theorem min_value_theorem (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) 
  (hsum : a + b + c = 3) : 
  1 / (3 * a + 5 * b) + 1 / (3 * b + 5 * c) + 1 / (3 * c + 5 * a) ≥ 9 / 8 := by
  sorry

end NUMINAMATH_CALUDE_min_value_theorem_l1090_109019


namespace NUMINAMATH_CALUDE_driptown_rainfall_2011_l1090_109026

/-- The total rainfall in Driptown in 2011 -/
def total_rainfall_2011 (avg_2010 avg_increase : ℝ) : ℝ :=
  12 * (avg_2010 + avg_increase)

/-- Theorem: The total rainfall in Driptown in 2011 was 468 mm -/
theorem driptown_rainfall_2011 :
  total_rainfall_2011 37.2 1.8 = 468 := by
  sorry

end NUMINAMATH_CALUDE_driptown_rainfall_2011_l1090_109026


namespace NUMINAMATH_CALUDE_total_coins_l1090_109013

/-- Given a number of stacks and coins per stack, proves that the total number of coins
    is equal to the product of these two quantities. -/
theorem total_coins (num_stacks : ℕ) (coins_per_stack : ℕ) :
  num_stacks * coins_per_stack = num_stacks * coins_per_stack := by
  sorry

/-- Calculates the total number of coins Maria has. -/
def maria_coins : ℕ :=
  let num_stacks : ℕ := 10
  let coins_per_stack : ℕ := 6
  num_stacks * coins_per_stack

#eval maria_coins

end NUMINAMATH_CALUDE_total_coins_l1090_109013


namespace NUMINAMATH_CALUDE_remainder_98_102_div_8_l1090_109061

theorem remainder_98_102_div_8 : (98 * 102) % 8 = 4 := by
  sorry

end NUMINAMATH_CALUDE_remainder_98_102_div_8_l1090_109061


namespace NUMINAMATH_CALUDE_added_number_proof_l1090_109023

theorem added_number_proof (x : ℝ) : 
  (((2 * (62.5 + x)) / 5) - 5 = 22) → x = 5 := by
  sorry

end NUMINAMATH_CALUDE_added_number_proof_l1090_109023


namespace NUMINAMATH_CALUDE_system_solution_l1090_109012

theorem system_solution (x y z b : ℝ) : 
  (3 * x * y * z - x^3 - y^3 - z^3 = b^3) ∧
  (x + y + z = 2 * b) ∧
  (x^2 + y^2 + z^2 = b^2) →
  ((x = 0 ∧ y = -z) ∨ (y = 0 ∧ x = -z)) ∧ b = 0 :=
by sorry

end NUMINAMATH_CALUDE_system_solution_l1090_109012


namespace NUMINAMATH_CALUDE_lcm_sum_ratio_problem_l1090_109031

theorem lcm_sum_ratio_problem (A B x y : ℕ+) : 
  Nat.lcm A B = 60 →
  A + B = 50 →
  x > y →
  A * y = B * x →
  x = 3 ∧ y = 2 := by sorry

end NUMINAMATH_CALUDE_lcm_sum_ratio_problem_l1090_109031


namespace NUMINAMATH_CALUDE_intersection_of_A_and_B_l1090_109064

def A : Set ℝ := {x : ℝ | |x| > 1}
def B : Set ℝ := {x : ℝ | 0 < x ∧ x < 2}

theorem intersection_of_A_and_B :
  A ∩ B = {x : ℝ | 1 < x ∧ x < 2} := by sorry

end NUMINAMATH_CALUDE_intersection_of_A_and_B_l1090_109064


namespace NUMINAMATH_CALUDE_box_capacity_l1090_109067

-- Define the volumes and capacities
def small_box_volume : ℝ := 24
def small_box_paperclips : ℕ := 60
def large_box_volume : ℝ := 72
def large_box_staples : ℕ := 90

-- Define the theorem
theorem box_capacity :
  ∃ (large_box_paperclips large_box_mixed_staples : ℕ),
    large_box_paperclips = 90 ∧ 
    large_box_mixed_staples = 45 ∧
    (large_box_paperclips : ℝ) / (large_box_volume / 2) = (small_box_paperclips : ℝ) / small_box_volume ∧
    (large_box_mixed_staples : ℝ) / (large_box_volume / 2) = (large_box_staples : ℝ) / large_box_volume :=
by
  sorry

end NUMINAMATH_CALUDE_box_capacity_l1090_109067


namespace NUMINAMATH_CALUDE_triangle_area_problem_l1090_109040

noncomputable def triangle_area (a b c : ℝ) (A B C : ℝ) : ℝ :=
  (1/2) * a * b * Real.sin C

theorem triangle_area_problem (a b c : ℝ) (A B C : ℝ) 
  (h1 : c^2 = (a-b)^2 + 6)
  (h2 : C = π/3) :
  triangle_area a b c A B C = (3 * Real.sqrt 3) / 2 := by
  sorry

end NUMINAMATH_CALUDE_triangle_area_problem_l1090_109040


namespace NUMINAMATH_CALUDE_mango_rate_problem_l1090_109036

/-- Calculates the rate per kg of mangoes given the total amount paid, grape weight, grape rate, and mango weight -/
def mango_rate (total_paid : ℕ) (grape_weight : ℕ) (grape_rate : ℕ) (mango_weight : ℕ) : ℕ :=
  (total_paid - grape_weight * grape_rate) / mango_weight

theorem mango_rate_problem :
  mango_rate 1376 14 54 10 = 62 := by
  sorry

end NUMINAMATH_CALUDE_mango_rate_problem_l1090_109036


namespace NUMINAMATH_CALUDE_distinct_sums_theorem_l1090_109029

theorem distinct_sums_theorem (k n : ℕ) (a b c : Fin n → ℝ) :
  k ≥ 3 →
  n > Nat.choose k 3 →
  (∀ i j : Fin n, i ≠ j → a i ≠ a j ∧ b i ≠ b j ∧ c i ≠ c j) →
  ∃ S : Finset ℝ,
    S.card ≥ k + 1 ∧
    (∀ i : Fin n, (a i + b i) ∈ S ∧ (a i + c i) ∈ S ∧ (b i + c i) ∈ S) :=
by sorry

end NUMINAMATH_CALUDE_distinct_sums_theorem_l1090_109029


namespace NUMINAMATH_CALUDE_pizza_slices_per_person_l1090_109094

theorem pizza_slices_per_person 
  (coworkers : ℕ) 
  (pizzas : ℕ) 
  (slices_per_pizza : ℕ) 
  (h1 : coworkers = 12)
  (h2 : pizzas = 3)
  (h3 : slices_per_pizza = 8)
  : (pizzas * slices_per_pizza) / coworkers = 2 := by
  sorry

end NUMINAMATH_CALUDE_pizza_slices_per_person_l1090_109094


namespace NUMINAMATH_CALUDE_year2018_is_WuXu_l1090_109091

/-- Represents the Ten Heavenly Stems -/
inductive HeavenlyStem
| Jia | Yi | Bing | Ding | Wu | Ji | Geng | Xin | Ren | Gui

/-- Represents the Twelve Earthly Branches -/
inductive EarthlyBranch
| Zi | Chou | Yin | Mao | Chen | Si | Wu | Wei | Shen | You | Xu | Hai

/-- Represents a year in the Sexagenary Cycle -/
structure SexagenaryYear :=
  (stem : HeavenlyStem)
  (branch : EarthlyBranch)

/-- The Sexagenary Cycle -/
def SexagenaryCycle : List SexagenaryYear := sorry

/-- Function to get the next year in the Sexagenary Cycle -/
def nextYear (year : SexagenaryYear) : SexagenaryYear := sorry

/-- 2016 is the Bing Shen year -/
def year2016 : SexagenaryYear :=
  { stem := HeavenlyStem.Bing, branch := EarthlyBranch.Shen }

/-- Theorem: 2018 is the Wu Xu year in the Sexagenary Cycle -/
theorem year2018_is_WuXu :
  (nextYear (nextYear year2016)) = { stem := HeavenlyStem.Wu, branch := EarthlyBranch.Xu } := by
  sorry


end NUMINAMATH_CALUDE_year2018_is_WuXu_l1090_109091


namespace NUMINAMATH_CALUDE_notebook_cost_l1090_109039

theorem notebook_cost (total_students : Nat) (buyers : Nat) (notebooks_per_student : Nat) (cost_per_notebook : Nat) 
  (h1 : total_students = 36)
  (h2 : buyers > total_students / 2)
  (h3 : notebooks_per_student > 2)
  (h4 : cost_per_notebook > notebooks_per_student)
  (h5 : buyers * notebooks_per_student * cost_per_notebook = 2601) :
  cost_per_notebook = 289 := by
  sorry

end NUMINAMATH_CALUDE_notebook_cost_l1090_109039


namespace NUMINAMATH_CALUDE_complex_number_in_fourth_quadrant_l1090_109097

def i : ℂ := Complex.I

theorem complex_number_in_fourth_quadrant :
  let z : ℂ := 1 / (1 + i)
  (z.re > 0 ∧ z.im < 0) := by sorry

end NUMINAMATH_CALUDE_complex_number_in_fourth_quadrant_l1090_109097


namespace NUMINAMATH_CALUDE_price_reduction_equation_correct_option_is_c_l1090_109086

/-- Represents the price reduction scenario -/
structure PriceReduction where
  initial_price : ℝ
  final_price : ℝ
  reduction_percentage : ℝ

/-- The equation correctly represents the price reduction scenario -/
def correct_equation (pr : PriceReduction) : Prop :=
  pr.initial_price * (1 - pr.reduction_percentage)^2 = pr.final_price

/-- Theorem stating that the equation correctly represents the given scenario -/
theorem price_reduction_equation :
  ∀ (pr : PriceReduction),
    pr.initial_price = 150 →
    pr.final_price = 96 →
    correct_equation pr :=
by
  sorry

/-- The correct option is C -/
theorem correct_option_is_c : 
  ∃ (pr : PriceReduction),
    pr.initial_price = 150 ∧
    pr.final_price = 96 ∧
    correct_equation pr :=
by
  sorry

end NUMINAMATH_CALUDE_price_reduction_equation_correct_option_is_c_l1090_109086


namespace NUMINAMATH_CALUDE_new_drive_free_space_l1090_109052

/-- Calculates the free space on a new external drive after file operations and transfer. -/
theorem new_drive_free_space 
  (initial_free : ℝ) 
  (initial_used : ℝ) 
  (deleted_size : ℝ) 
  (new_files_size : ℝ) 
  (new_drive_size : ℝ)
  (h1 : initial_free = 2.4)
  (h2 : initial_used = 12.6)
  (h3 : deleted_size = 4.6)
  (h4 : new_files_size = 2)
  (h5 : new_drive_size = 20) :
  new_drive_size - (initial_used - deleted_size + new_files_size) = 10 := by
  sorry

#check new_drive_free_space

end NUMINAMATH_CALUDE_new_drive_free_space_l1090_109052


namespace NUMINAMATH_CALUDE_unique_divisiblity_solution_l1090_109092

theorem unique_divisiblity_solution : 
  ∀ a m n : ℕ+, 
    (a^n.val + 1 ∣ (a+1)^m.val) → 
    (a = 2 ∧ m = 2 ∧ n = 3) :=
by sorry

end NUMINAMATH_CALUDE_unique_divisiblity_solution_l1090_109092


namespace NUMINAMATH_CALUDE_min_value_sin_2x_minus_pi_4_l1090_109038

theorem min_value_sin_2x_minus_pi_4 :
  ∃ (min : ℝ), min = -Real.sqrt 2 / 2 ∧
  ∀ x : ℝ, x ∈ Set.Icc 0 (Real.pi / 2) →
  Real.sin (2 * x - Real.pi / 4) ≥ min :=
sorry

end NUMINAMATH_CALUDE_min_value_sin_2x_minus_pi_4_l1090_109038


namespace NUMINAMATH_CALUDE_benny_seashells_l1090_109071

theorem benny_seashells (initial_seashells given_away remaining : ℕ) : 
  initial_seashells = 66 → given_away = 52 → remaining = 14 →
  initial_seashells - given_away = remaining := by sorry

end NUMINAMATH_CALUDE_benny_seashells_l1090_109071


namespace NUMINAMATH_CALUDE_square_difference_square_sum_mental_math_strategy_l1090_109018

theorem square_difference (n : ℕ) : (n - 1)^2 = n^2 - (2*n - 1) :=
  sorry

theorem square_sum (n : ℕ) : (n + 1)^2 = n^2 + (2*n + 1) :=
  sorry

theorem mental_math_strategy :
  49^2 = 50^2 - 99 ∧ 51^2 = 50^2 + 101 :=
by
  have h1 : 49^2 = 50^2 - 99 := by
    calc
      49^2 = (50 - 1)^2 := by rfl
      _ = 50^2 - (2*50 - 1) := by apply square_difference
      _ = 50^2 - 99 := by ring
  
  have h2 : 51^2 = 50^2 + 101 := by
    calc
      51^2 = (50 + 1)^2 := by rfl
      _ = 50^2 + (2*50 + 1) := by apply square_sum
      _ = 50^2 + 101 := by ring
  
  exact ⟨h1, h2⟩

#check mental_math_strategy

end NUMINAMATH_CALUDE_square_difference_square_sum_mental_math_strategy_l1090_109018


namespace NUMINAMATH_CALUDE_space_needle_height_is_184_l1090_109077

-- Define the heights of the towers
def cn_tower_height : ℝ := 553
def height_difference : ℝ := 369

-- Define the height of the Space Needle
def space_needle_height : ℝ := cn_tower_height - height_difference

-- Theorem to prove
theorem space_needle_height_is_184 : space_needle_height = 184 := by
  sorry

end NUMINAMATH_CALUDE_space_needle_height_is_184_l1090_109077


namespace NUMINAMATH_CALUDE_sum_positive_implies_at_least_one_positive_l1090_109043

theorem sum_positive_implies_at_least_one_positive (a b : ℝ) :
  a + b > 0 → (a > 0 ∨ b > 0) := by
  sorry

end NUMINAMATH_CALUDE_sum_positive_implies_at_least_one_positive_l1090_109043


namespace NUMINAMATH_CALUDE_walking_distance_l1090_109015

/-- Given a person who walks at two speeds, prove that the actual distance traveled at the slower speed is 50 km. -/
theorem walking_distance (slow_speed fast_speed extra_distance : ℝ) 
  (h1 : slow_speed = 10)
  (h2 : fast_speed = 14)
  (h3 : extra_distance = 20)
  (h4 : slow_speed > 0)
  (h5 : fast_speed > slow_speed) :
  ∃ (actual_distance : ℝ),
    actual_distance / slow_speed = (actual_distance + extra_distance) / fast_speed ∧
    actual_distance = 50 := by
  sorry


end NUMINAMATH_CALUDE_walking_distance_l1090_109015


namespace NUMINAMATH_CALUDE_find_alpha_l1090_109046

theorem find_alpha (α β : ℝ) (h1 : α + β = 11) (h2 : α * β = 24) (h3 : α > β) : α = 8 := by
  sorry

end NUMINAMATH_CALUDE_find_alpha_l1090_109046


namespace NUMINAMATH_CALUDE_brookes_cows_solution_l1090_109065

/-- Represents the problem of determining the number of cows Brooke has --/
def brookes_cows (milk_price : ℚ) (butter_conversion : ℚ) (butter_price : ℚ) 
  (milk_per_cow : ℚ) (num_customers : ℕ) (milk_per_customer : ℚ) (total_earnings : ℚ) : Prop :=
  milk_price = 3 ∧
  butter_conversion = 2 ∧
  butter_price = 3/2 ∧
  milk_per_cow = 4 ∧
  num_customers = 6 ∧
  milk_per_customer = 6 ∧
  total_earnings = 144 ∧
  ∃ (num_cows : ℕ), 
    (↑num_cows : ℚ) * milk_per_cow = 
      (↑num_customers * milk_per_customer) + 
      ((total_earnings - (↑num_customers * milk_per_customer * milk_price)) / butter_price * (1 / butter_conversion))

theorem brookes_cows_solution :
  ∀ (milk_price butter_conversion butter_price milk_per_cow : ℚ)
    (num_customers : ℕ) (milk_per_customer total_earnings : ℚ),
  brookes_cows milk_price butter_conversion butter_price milk_per_cow num_customers milk_per_customer total_earnings →
  ∃ (num_cows : ℕ), num_cows = 12 :=
by sorry

end NUMINAMATH_CALUDE_brookes_cows_solution_l1090_109065


namespace NUMINAMATH_CALUDE_max_value_rational_function_max_value_attained_l1090_109010

theorem max_value_rational_function (x : ℝ) :
  x^6 / (x^12 + 3*x^9 - 5*x^6 + 15*x^3 + 27) ≤ 1/37 :=
by sorry

theorem max_value_attained :
  ∃ x : ℝ, x^6 / (x^12 + 3*x^9 - 5*x^6 + 15*x^3 + 27) = 1/37 :=
by sorry

end NUMINAMATH_CALUDE_max_value_rational_function_max_value_attained_l1090_109010


namespace NUMINAMATH_CALUDE_total_rectangles_is_eighteen_l1090_109025

/-- Represents a rectangle in the figure -/
structure Rectangle where
  size : Nat

/-- Represents the figure composed of rectangles -/
structure Figure where
  big_rectangle : Rectangle
  small_rectangles : Finset Rectangle
  middle_rectangles : Finset Rectangle

/-- Counts the total number of rectangles in the figure -/
def count_rectangles (f : Figure) : Nat :=
  1 + f.small_rectangles.card + f.middle_rectangles.card

/-- The theorem stating that the total number of rectangles is 18 -/
theorem total_rectangles_is_eighteen (f : Figure) 
  (h1 : f.big_rectangle.size = 1)
  (h2 : f.small_rectangles.card = 6)
  (h3 : f.middle_rectangles.card = 11) : 
  count_rectangles f = 18 := by
  sorry

#check total_rectangles_is_eighteen

end NUMINAMATH_CALUDE_total_rectangles_is_eighteen_l1090_109025


namespace NUMINAMATH_CALUDE_nina_reading_homework_multiplier_l1090_109049

-- Define the given conditions
def ruby_math_homework : ℕ := 6
def ruby_reading_homework : ℕ := 2
def nina_total_homework : ℕ := 48
def nina_math_homework_multiplier : ℕ := 4

-- Define Nina's math homework
def nina_math_homework : ℕ := ruby_math_homework * (nina_math_homework_multiplier + 1)

-- Define Nina's reading homework
def nina_reading_homework : ℕ := nina_total_homework - nina_math_homework

-- Theorem to prove
theorem nina_reading_homework_multiplier :
  nina_reading_homework / ruby_reading_homework = 9 := by
  sorry


end NUMINAMATH_CALUDE_nina_reading_homework_multiplier_l1090_109049


namespace NUMINAMATH_CALUDE_towel_shrinkage_l1090_109002

theorem towel_shrinkage (L B : ℝ) (h1 : L > 0) (h2 : B > 0) : 
  let new_length := 0.8 * L
  let new_area := 0.64 * (L * B)
  ∃ new_breadth : ℝ, 
    new_breadth = 0.8 * B ∧ 
    new_length * new_breadth = new_area := by
  sorry

end NUMINAMATH_CALUDE_towel_shrinkage_l1090_109002


namespace NUMINAMATH_CALUDE_part1_part2_l1090_109007

-- Define the sequences
def a : ℕ → ℝ := λ n => 2^n
def b : ℕ → ℝ := λ n => 3^n
def c : ℕ → ℝ := λ n => a n + b n

-- Part 1
theorem part1 (p : ℝ) :
  (∃ r : ℝ, ∀ n : ℕ, c (n + 2) - p * c (n + 1) = r * (c (n + 1) - p * c n)) →
  p = 2 ∨ p = 3 := by sorry

-- Part 2
theorem part2 {q1 q2 : ℝ} (hq : q1 ≠ q2) 
  (ha : ∀ n : ℕ, a (n + 1) = q1 * a n) 
  (hb : ∀ n : ℕ, b (n + 1) = q2 * b n) :
  ¬ (∃ r : ℝ, ∀ n : ℕ, c (n + 1) = r * c n) := by sorry

end NUMINAMATH_CALUDE_part1_part2_l1090_109007


namespace NUMINAMATH_CALUDE_square_area_ratio_l1090_109085

theorem square_area_ratio (a b : ℝ) (h : 4 * a = 16 * b) : a^2 = 16 * b^2 := by
  sorry

end NUMINAMATH_CALUDE_square_area_ratio_l1090_109085


namespace NUMINAMATH_CALUDE_sum_of_digits_of_B_l1090_109048

def digit_sum (n : ℕ) : ℕ := sorry

def A : ℕ := digit_sum (4444^4444)

def B : ℕ := digit_sum A

theorem sum_of_digits_of_B : digit_sum B = 7 := by sorry

end NUMINAMATH_CALUDE_sum_of_digits_of_B_l1090_109048


namespace NUMINAMATH_CALUDE_condition_relationship_l1090_109076

theorem condition_relationship (x : ℝ) :
  (∀ x, 2 * Real.sqrt 2 ≤ (x^2 + 2) / x ∧ (x^2 + 2) / x ≤ 3 → Real.sqrt 2 / 2 ≤ x ∧ x ≤ 2 * Real.sqrt 2) ∧
  (∃ x, Real.sqrt 2 / 2 ≤ x ∧ x ≤ 2 * Real.sqrt 2 ∧ (2 * Real.sqrt 2 > (x^2 + 2) / x ∨ (x^2 + 2) / x > 3)) :=
by sorry

end NUMINAMATH_CALUDE_condition_relationship_l1090_109076


namespace NUMINAMATH_CALUDE_banana_jar_candy_count_l1090_109089

/-- Given three jars of candy with specific relationships, prove the number of candy pieces in the banana jar. -/
theorem banana_jar_candy_count (peanut_butter grape banana : ℕ) 
  (h1 : peanut_butter = 4 * grape)
  (h2 : grape = banana + 5)
  (h3 : peanut_butter = 192) : 
  banana = 43 := by
  sorry

end NUMINAMATH_CALUDE_banana_jar_candy_count_l1090_109089


namespace NUMINAMATH_CALUDE_dales_peppers_theorem_l1090_109090

/-- The amount of green peppers bought by Dale's Vegetarian Restaurant in pounds -/
def green_peppers : ℝ := 2.8333333333333335

/-- The amount of red peppers bought by Dale's Vegetarian Restaurant in pounds -/
def red_peppers : ℝ := 2.8333333333333335

/-- The total amount of peppers bought by Dale's Vegetarian Restaurant in pounds -/
def total_peppers : ℝ := green_peppers + red_peppers

theorem dales_peppers_theorem : total_peppers = 5.666666666666667 := by
  sorry

end NUMINAMATH_CALUDE_dales_peppers_theorem_l1090_109090


namespace NUMINAMATH_CALUDE_sphere_in_cube_ratios_l1090_109032

/-- The ratio of volumes and surface areas for a sphere inscribed in a cube -/
theorem sphere_in_cube_ratios (s : ℝ) (h : s > 0) :
  let sphere_volume := (4 / 3) * Real.pi * s^3
  let cube_volume := (2 * s)^3
  let sphere_surface_area := 4 * Real.pi * s^2
  let cube_surface_area := 6 * (2 * s)^2
  (sphere_volume / cube_volume = Real.pi / 6) ∧
  (sphere_surface_area / cube_surface_area = Real.pi / 6) := by
  sorry

end NUMINAMATH_CALUDE_sphere_in_cube_ratios_l1090_109032


namespace NUMINAMATH_CALUDE_dot_product_range_l1090_109053

/-- A circle centered at the origin and tangent to the line x-√3y=4 -/
def Circle := {p : ℝ × ℝ | p.1^2 + p.2^2 = 4}

/-- The line x-√3y=4 -/
def TangentLine := {p : ℝ × ℝ | p.1 - Real.sqrt 3 * p.2 = 4}

/-- Point A where the circle intersects the negative x-axis -/
def A : ℝ × ℝ := (-2, 0)

/-- Point B where the circle intersects the positive x-axis -/
def B : ℝ × ℝ := (2, 0)

/-- A point P inside the circle satisfying the geometric sequence condition -/
def P := {p : ℝ × ℝ | p ∈ Circle ∧ p.1^2 = p.2^2 + 2}

/-- The dot product of PA and PB -/
def dotProduct (p : ℝ × ℝ) : ℝ := 
  (A.1 - p.1) * (B.1 - p.1) + (A.2 - p.2) * (B.2 - p.2)

theorem dot_product_range :
  ∀ p ∈ P, -2 ≤ dotProduct p ∧ dotProduct p < 0 := by sorry

end NUMINAMATH_CALUDE_dot_product_range_l1090_109053


namespace NUMINAMATH_CALUDE_conditional_without_else_l1090_109001

-- Define the structure of conditional statements
inductive ConditionalStatement
  | ifThen : ConditionalStatement
  | ifThenElse : ConditionalStatement

-- Define a property for conditional statements with one branch
def hasOneBranch (stmt : ConditionalStatement) : Prop :=
  match stmt with
  | ConditionalStatement.ifThen => true
  | ConditionalStatement.ifThenElse => false

-- Theorem: A conditional statement can be without the statement after ELSE
theorem conditional_without_else :
  ∃ (stmt : ConditionalStatement), hasOneBranch stmt ∧ stmt = ConditionalStatement.ifThen :=
sorry

end NUMINAMATH_CALUDE_conditional_without_else_l1090_109001


namespace NUMINAMATH_CALUDE_total_jewelry_is_83_l1090_109088

/-- Represents the initial jewelry counts and purchase rules --/
structure JewelryInventory where
  initial_necklaces : ℕ
  initial_earrings : ℕ
  initial_bracelets : ℕ
  initial_rings : ℕ
  store_a_necklaces : ℕ
  store_a_bracelets : ℕ
  store_b_necklaces : ℕ

/-- Calculates the total number of jewelry pieces after all additions --/
def totalJewelryPieces (inventory : JewelryInventory) : ℕ :=
  let store_a_earrings := (2 * inventory.initial_earrings) / 3
  let store_b_rings := 2 * inventory.initial_rings
  let mother_gift_earrings := store_a_earrings / 5
  
  inventory.initial_necklaces + inventory.initial_earrings + inventory.initial_bracelets + inventory.initial_rings +
  inventory.store_a_necklaces + store_a_earrings + inventory.store_a_bracelets +
  inventory.store_b_necklaces + store_b_rings +
  mother_gift_earrings

/-- Theorem stating that the total jewelry pieces is 83 given the specific inventory --/
theorem total_jewelry_is_83 :
  totalJewelryPieces ⟨10, 15, 5, 8, 10, 3, 4⟩ = 83 := by
  sorry

end NUMINAMATH_CALUDE_total_jewelry_is_83_l1090_109088


namespace NUMINAMATH_CALUDE_second_bush_pink_roses_l1090_109045

def rose_problem (red : ℕ) (yellow : ℕ) (orange : ℕ) (total_picked : ℕ) : ℕ :=
  let red_picked := red / 2
  let yellow_picked := yellow / 4
  let orange_picked := orange / 4
  let pink_picked := total_picked - red_picked - yellow_picked - orange_picked
  2 * pink_picked

theorem second_bush_pink_roses :
  rose_problem 12 20 8 22 = 18 := by
  sorry

end NUMINAMATH_CALUDE_second_bush_pink_roses_l1090_109045


namespace NUMINAMATH_CALUDE_lesser_number_l1090_109072

theorem lesser_number (a b : ℝ) (h1 : a + b = 55) (h2 : a - b = 7) : min a b = 24 := by
  sorry

end NUMINAMATH_CALUDE_lesser_number_l1090_109072


namespace NUMINAMATH_CALUDE_arithmetic_calculations_l1090_109021

theorem arithmetic_calculations :
  ((-2 : ℝ) + |3| + (-6) + |7| = 2) ∧
  (3.7 + (-1.3) + (-6.7) + 2.3 = -2) := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_calculations_l1090_109021


namespace NUMINAMATH_CALUDE_hyperbola_asymptotes_l1090_109074

/-- Given a hyperbola with equation y²/a² - x²/b² = 1 where a > 0 and b > 0,
    if the eccentricity is √3, then the equation of its asymptotes is x ± √2y = 0 -/
theorem hyperbola_asymptotes (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  (∃ (x y : ℝ), y^2 / a^2 - x^2 / b^2 = 1) →
  (∃ (c : ℝ), c^2 = a^2 + b^2 ∧ c / a = Real.sqrt 3) →
  (∃ (x y : ℝ), x^2 - 2 * y^2 = 0) :=
sorry

end NUMINAMATH_CALUDE_hyperbola_asymptotes_l1090_109074


namespace NUMINAMATH_CALUDE_inequality_proof_l1090_109009

theorem inequality_proof (a b : ℝ) (ha : 0 < a) (ha' : a < 1) (hb : 0 < b) (hb' : b < 1) :
  (a * b * (1 - a) * (1 - b)) / ((1 - a * b)^2) < (1 : ℝ) / 4 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l1090_109009


namespace NUMINAMATH_CALUDE_broken_marbles_count_l1090_109078

/-- The number of marbles in the first set -/
def set1_total : ℕ := 50

/-- The percentage of broken marbles in the first set -/
def set1_broken_percent : ℚ := 1/10

/-- The number of marbles in the second set -/
def set2_total : ℕ := 60

/-- The percentage of broken marbles in the second set -/
def set2_broken_percent : ℚ := 1/5

/-- The total number of broken marbles in both sets -/
def total_broken_marbles : ℕ := 17

theorem broken_marbles_count : 
  ⌊(set1_total : ℚ) * set1_broken_percent⌋ + ⌊(set2_total : ℚ) * set2_broken_percent⌋ = total_broken_marbles :=
by sorry

end NUMINAMATH_CALUDE_broken_marbles_count_l1090_109078


namespace NUMINAMATH_CALUDE_chessboard_game_stone_range_min_le_max_stones_l1090_109082

/-- A game on an n × n chessboard with k stones -/
def ChessboardGame (n : ℕ) (k : ℕ) : Prop :=
  n > 0 ∧ k ≥ 2 * n^2 - 2 * n ∧ k ≤ 3 * n^2 - 4 * n

/-- The theorem stating the range of stones for the game -/
theorem chessboard_game_stone_range (n : ℕ) :
  n > 0 → ∀ k, ChessboardGame n k ↔ 2 * n^2 - 2 * n ≤ k ∧ k ≤ 3 * n^2 - 4 * n :=
by sorry

/-- The minimum number of stones for the game -/
def min_stones (n : ℕ) : ℕ := 2 * n^2 - 2 * n

/-- The maximum number of stones for the game -/
def max_stones (n : ℕ) : ℕ := 3 * n^2 - 4 * n

/-- Theorem stating that the minimum number of stones is always less than or equal to the maximum -/
theorem min_le_max_stones (n : ℕ) : n > 0 → min_stones n ≤ max_stones n :=
by sorry

end NUMINAMATH_CALUDE_chessboard_game_stone_range_min_le_max_stones_l1090_109082


namespace NUMINAMATH_CALUDE_cloth_sale_price_l1090_109047

/-- Calculates the total selling price of cloth given the quantity, profit per meter, and cost price per meter. -/
def totalSellingPrice (quantity : ℕ) (profitPerMeter : ℕ) (costPricePerMeter : ℕ) : ℕ :=
  quantity * (costPricePerMeter + profitPerMeter)

/-- Proves that the total selling price of 85 meters of cloth with a profit of $15 per meter and a cost price of $90 per meter is $8925. -/
theorem cloth_sale_price : totalSellingPrice 85 15 90 = 8925 := by
  sorry

end NUMINAMATH_CALUDE_cloth_sale_price_l1090_109047


namespace NUMINAMATH_CALUDE_auditorium_seats_l1090_109069

theorem auditorium_seats (initial_seats : ℕ) (final_seats : ℕ) (seat_increase : ℕ) : 
  initial_seats = 320 →
  final_seats = 420 →
  seat_increase = 4 →
  ∃ (initial_rows : ℕ),
    initial_rows > 0 ∧
    initial_seats % initial_rows = 0 ∧
    (initial_seats / initial_rows + seat_increase) * (initial_rows + 1) = final_seats ∧
    initial_rows + 1 = 21 := by
  sorry

end NUMINAMATH_CALUDE_auditorium_seats_l1090_109069


namespace NUMINAMATH_CALUDE_items_left_in_cart_l1090_109098

def initial_items : ℕ := 18
def deleted_items : ℕ := 10

theorem items_left_in_cart : initial_items - deleted_items = 8 := by
  sorry

end NUMINAMATH_CALUDE_items_left_in_cart_l1090_109098


namespace NUMINAMATH_CALUDE_parabola_kite_sum_l1090_109051

/-- Represents a parabola of the form y = ax^2 + b -/
structure Parabola where
  a : ℝ
  b : ℝ

/-- Represents a kite formed by the intersection points of two parabolas with the coordinate axes -/
structure Kite where
  p1 : Parabola
  p2 : Parabola

theorem parabola_kite_sum (k : Kite) : 
  k.p1.a > 0 ∧ k.p2.a < 0 ∧  -- Ensure parabolas open in opposite directions
  k.p1.b = -4 ∧ k.p2.b = 6 ∧  -- Specific y-intercepts
  (∃ x y : ℝ, x ≠ 0 ∧ y ≠ 0 ∧ 
    k.p1.a * x^2 + k.p1.b = 0 ∧   -- x-intercepts of first parabola
    k.p2.a * x^2 + k.p2.b = 0 ∧   -- x-intercepts of second parabola
    k.p1.a * y^2 + k.p1.b = k.p2.a * y^2 + k.p2.b) ∧  -- Intersection point
  (1/2 * (2 * (k.p2.b - k.p1.b)) * (2 * Real.sqrt (k.p2.b / (-k.p2.a))) = 24) →  -- Area of kite
  k.p1.a + (-k.p2.a) = 125/72 := by
sorry

end NUMINAMATH_CALUDE_parabola_kite_sum_l1090_109051


namespace NUMINAMATH_CALUDE_black_to_white_area_ratio_l1090_109096

/-- The ratio of black to white area in concentric circles with radii 2, 4, 6, and 8 -/
theorem black_to_white_area_ratio : Real := by
  -- Define the radii of the circles
  let r1 : Real := 2
  let r2 : Real := 4
  let r3 : Real := 6
  let r4 : Real := 8

  -- Define the areas of the circles
  let A1 : Real := Real.pi * r1^2
  let A2 : Real := Real.pi * r2^2
  let A3 : Real := Real.pi * r3^2
  let A4 : Real := Real.pi * r4^2

  -- Define the areas of the black and white regions
  let black_area : Real := A1 + (A3 - A2)
  let white_area : Real := (A2 - A1) + (A4 - A3)

  -- Prove that the ratio of black area to white area is 3/5
  have h : black_area / white_area = 3 / 5 := by sorry

  exact 3 / 5

end NUMINAMATH_CALUDE_black_to_white_area_ratio_l1090_109096


namespace NUMINAMATH_CALUDE_iodine_mixture_theorem_l1090_109087

-- Define the given constants
def solution1_percentage : ℝ := 40
def solution2_volume : ℝ := 4.5
def final_mixture_volume : ℝ := 6
def final_mixture_percentage : ℝ := 50

-- Define the unknown percentage of the second solution
def solution2_percentage : ℝ := 26.67

-- Theorem statement
theorem iodine_mixture_theorem :
  solution1_percentage / 100 * solution2_volume + 
  solution2_percentage / 100 * solution2_volume = 
  final_mixture_percentage / 100 * final_mixture_volume := by
  sorry

end NUMINAMATH_CALUDE_iodine_mixture_theorem_l1090_109087


namespace NUMINAMATH_CALUDE_max_value_on_curves_l1090_109054

theorem max_value_on_curves (m n x y : ℝ) (α β : ℝ) : 
  m = Real.sqrt 6 * Real.cos α →
  n = Real.sqrt 6 * Real.sin α →
  x = Real.sqrt 24 * Real.cos β →
  y = Real.sqrt 24 * Real.sin β →
  (∀ m' n' x' y' α' β', 
    m' = Real.sqrt 6 * Real.cos α' →
    n' = Real.sqrt 6 * Real.sin α' →
    x' = Real.sqrt 24 * Real.cos β' →
    y' = Real.sqrt 24 * Real.sin β' →
    m' * x' + n' * y' ≤ 12) ∧
  (∃ m' n' x' y' α' β', 
    m' = Real.sqrt 6 * Real.cos α' ∧
    n' = Real.sqrt 6 * Real.sin α' ∧
    x' = Real.sqrt 24 * Real.cos β' ∧
    y' = Real.sqrt 24 * Real.sin β' ∧
    m' * x' + n' * y' = 12) :=
by sorry

end NUMINAMATH_CALUDE_max_value_on_curves_l1090_109054

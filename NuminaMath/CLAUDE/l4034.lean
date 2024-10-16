import Mathlib

namespace NUMINAMATH_CALUDE_parking_lot_rows_parking_lot_rows_example_l4034_403406

/-- Given a parking lot with the following properties:
    - A car is 5th from the right and 4th from the left in a row
    - The parking lot has 10 floors
    - There are 1600 cars in total
    The number of rows on each floor is 20. -/
theorem parking_lot_rows (car_position_right : Nat) (car_position_left : Nat)
                         (num_floors : Nat) (total_cars : Nat) : Nat :=
  let cars_in_row := car_position_right + car_position_left - 1
  let cars_per_floor := total_cars / num_floors
  cars_per_floor / cars_in_row

#check parking_lot_rows 5 4 10 1600 = 20

/-- The parking_lot_rows theorem holds for the given values. -/
theorem parking_lot_rows_example : parking_lot_rows 5 4 10 1600 = 20 := by
  sorry

end NUMINAMATH_CALUDE_parking_lot_rows_parking_lot_rows_example_l4034_403406


namespace NUMINAMATH_CALUDE_newspaper_conference_max_overlap_l4034_403491

theorem newspaper_conference_max_overlap (total : ℕ) (writers : ℕ) (editors : ℕ) (x : ℕ) :
  total = 100 →
  writers = 35 →
  editors > 38 →
  writers + editors + x = total →
  x ≤ 26 :=
by sorry

end NUMINAMATH_CALUDE_newspaper_conference_max_overlap_l4034_403491


namespace NUMINAMATH_CALUDE_initial_investment_l4034_403463

/-- Given an initial investment A at a simple annual interest rate r,
    prove that A = 5000 when the interest on A is $250 and
    the interest on $20,000 at the same rate is $1000. -/
theorem initial_investment (A r : ℝ) : 
  A > 0 →
  r > 0 →
  A * r / 100 = 250 →
  20000 * r / 100 = 1000 →
  A = 5000 := by
sorry

end NUMINAMATH_CALUDE_initial_investment_l4034_403463


namespace NUMINAMATH_CALUDE_cube_root_over_sixth_root_of_eight_l4034_403412

theorem cube_root_over_sixth_root_of_eight (x : ℝ) :
  (8 : ℝ) ^ (1/3) / (8 : ℝ) ^ (1/6) = (8 : ℝ) ^ (1/6) :=
by sorry

end NUMINAMATH_CALUDE_cube_root_over_sixth_root_of_eight_l4034_403412


namespace NUMINAMATH_CALUDE_expression_evaluation_l4034_403494

theorem expression_evaluation :
  let eight : ℕ := 2^3
  let sixteen : ℕ := 2^4
  (eight^5 / eight^3) * sixteen^4 / 2^3 = 524288 := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l4034_403494


namespace NUMINAMATH_CALUDE_problem_statement_l4034_403432

theorem problem_statement (x y : ℝ) (hx : x > 0) (hy : y > 0) (hsum : x + y = 1) :
  (1/6 ≤ x ∧ x < 1 ↔ |x + 2*y| + |x - y| ≤ 5/2) ∧
  (1/x^2 - 1) * (1/y^2 - 1) ≥ 9 := by
sorry

end NUMINAMATH_CALUDE_problem_statement_l4034_403432


namespace NUMINAMATH_CALUDE_expression_evaluation_l4034_403497

theorem expression_evaluation :
  let a : ℤ := -1
  let b : ℤ := 3
  2 * a * b^2 - (3 * a^2 * b - 2 * (3 * a^2 * b - a * b^2 - 1)) = 7 := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l4034_403497


namespace NUMINAMATH_CALUDE_julia_cd_shortage_l4034_403417

/-- The cost of a rock and roll CD -/
def rock_cost : ℕ := 5

/-- The cost of a pop CD -/
def pop_cost : ℕ := 10

/-- The cost of a dance CD -/
def dance_cost : ℕ := 3

/-- The cost of a country CD -/
def country_cost : ℕ := 7

/-- The number of CDs Julia wants to buy of each type -/
def cds_per_type : ℕ := 4

/-- Julia's budget -/
def julia_budget : ℕ := 75

/-- The amount Julia is short -/
def shortage : ℕ := 25

theorem julia_cd_shortage : 
  (rock_cost * cds_per_type + 
   pop_cost * cds_per_type + 
   dance_cost * cds_per_type + 
   country_cost * cds_per_type) - 
  julia_budget = shortage := by sorry

end NUMINAMATH_CALUDE_julia_cd_shortage_l4034_403417


namespace NUMINAMATH_CALUDE_fraction_ordering_l4034_403442

theorem fraction_ordering : (8 : ℚ) / 24 < 6 / 17 ∧ 6 / 17 < 10 / 27 := by
  sorry

end NUMINAMATH_CALUDE_fraction_ordering_l4034_403442


namespace NUMINAMATH_CALUDE_trigonometric_equation_solution_l4034_403430

theorem trigonometric_equation_solution (x y : ℝ) : 
  x = π / 6 → 
  Real.sin x * Real.cos x * y - 2 * Real.sin x * Real.sin x * y + Real.cos x * y = 1/2 → 
  y = (6 * Real.sqrt 3 + 4) / 23 := by
sorry

end NUMINAMATH_CALUDE_trigonometric_equation_solution_l4034_403430


namespace NUMINAMATH_CALUDE_total_paintable_area_l4034_403404

def bedroom_type1_length : ℝ := 14
def bedroom_type1_width : ℝ := 11
def bedroom_type1_height : ℝ := 9
def bedroom_type2_length : ℝ := 13
def bedroom_type2_width : ℝ := 12
def bedroom_type2_height : ℝ := 9
def num_bedrooms : ℕ := 4
def unpaintable_area : ℝ := 70

def wall_area (length width height : ℝ) : ℝ :=
  2 * (length * height + width * height)

def paintable_area (total_area unpaintable_area : ℝ) : ℝ :=
  total_area - unpaintable_area

theorem total_paintable_area :
  let type1_area := wall_area bedroom_type1_length bedroom_type1_width bedroom_type1_height
  let type2_area := wall_area bedroom_type2_length bedroom_type2_width bedroom_type2_height
  let total_area := (num_bedrooms / 2) * (paintable_area type1_area unpaintable_area + 
                                          paintable_area type2_area unpaintable_area)
  total_area = 1520 := by
  sorry

end NUMINAMATH_CALUDE_total_paintable_area_l4034_403404


namespace NUMINAMATH_CALUDE_april_plant_arrangement_l4034_403470

/-- The number of ways to arrange plants with specific conditions -/
def plant_arrangements (n_basil : ℕ) (n_tomato : ℕ) : ℕ :=
  (n_basil + n_tomato - 1).factorial * (n_basil - 1).factorial

/-- Theorem stating the number of arrangements for the given problem -/
theorem april_plant_arrangement :
  plant_arrangements 5 3 = 576 := by
  sorry

end NUMINAMATH_CALUDE_april_plant_arrangement_l4034_403470


namespace NUMINAMATH_CALUDE_three_digit_numbers_divisible_by_17_l4034_403465

theorem three_digit_numbers_divisible_by_17 : 
  (Finset.filter (fun k => 100 ≤ 17 * k ∧ 17 * k ≤ 999) (Finset.range 1000)).card = 53 := by
  sorry

end NUMINAMATH_CALUDE_three_digit_numbers_divisible_by_17_l4034_403465


namespace NUMINAMATH_CALUDE_world_cup_investment_scientific_notation_l4034_403439

/-- Scientific notation representation of a number -/
structure ScientificNotation where
  coefficient : ℝ
  exponent : ℤ
  is_valid : 1 ≤ coefficient ∧ coefficient < 10

/-- Convert a real number to scientific notation -/
def toScientificNotation (x : ℝ) : ScientificNotation :=
  sorry

theorem world_cup_investment_scientific_notation :
  toScientificNotation 220000000000 = ScientificNotation.mk 2.2 11 (by norm_num) :=
sorry

end NUMINAMATH_CALUDE_world_cup_investment_scientific_notation_l4034_403439


namespace NUMINAMATH_CALUDE_exists_monochromatic_triangle_l4034_403471

/-- A type representing the scientists -/
def Scientist : Type := Fin 17

/-- A type representing the topics -/
def Topic : Type := Fin 3

/-- A function representing the correspondence between scientists on a specific topic -/
def corresponds (s1 s2 : Scientist) : Topic :=
  sorry

/-- The main theorem stating that there exists a monochromatic triangle -/
theorem exists_monochromatic_triangle :
  ∃ (s1 s2 s3 : Scientist) (t : Topic),
    s1 ≠ s2 ∧ s2 ≠ s3 ∧ s1 ≠ s3 ∧
    corresponds s1 s2 = t ∧
    corresponds s2 s3 = t ∧
    corresponds s1 s3 = t :=
  sorry

end NUMINAMATH_CALUDE_exists_monochromatic_triangle_l4034_403471


namespace NUMINAMATH_CALUDE_three_lines_common_points_l4034_403405

/-- A line in 3D space --/
structure Line3D where
  -- Define a line using two points
  point1 : ℝ × ℝ × ℝ
  point2 : ℝ × ℝ × ℝ
  ne : point1 ≠ point2

/-- The number of common points of three lines in 3D space --/
def commonPointCount (l1 l2 l3 : Line3D) : Nat :=
  sorry

/-- Three lines determine three planes --/
def determineThreePlanes (l1 l2 l3 : Line3D) : Prop :=
  sorry

theorem three_lines_common_points 
  (l1 l2 l3 : Line3D) 
  (h : determineThreePlanes l1 l2 l3) : 
  commonPointCount l1 l2 l3 = 0 ∨ commonPointCount l1 l2 l3 = 1 :=
sorry

end NUMINAMATH_CALUDE_three_lines_common_points_l4034_403405


namespace NUMINAMATH_CALUDE_dragon_poker_ways_l4034_403411

/-- The number of points to be scored -/
def target_points : ℕ := 2018

/-- The number of suits in the deck -/
def num_suits : ℕ := 4

/-- Calculates the number of ways to partition a given number into a specified number of parts -/
def partition_ways (n : ℕ) (k : ℕ) : ℕ := Nat.choose (n + k - 1) (k - 1)

/-- The main theorem: The number of ways to score exactly 2018 points in Dragon Poker -/
theorem dragon_poker_ways : partition_ways target_points num_suits = 1373734330 := by
  sorry

end NUMINAMATH_CALUDE_dragon_poker_ways_l4034_403411


namespace NUMINAMATH_CALUDE_resort_tips_multiple_l4034_403441

theorem resort_tips_multiple (total_months : Nat) (special_month_fraction : Real) 
  (h1 : total_months = 7)
  (h2 : special_month_fraction = 0.5)
  (average_other_months : Real)
  (special_month_tips : Real)
  (h3 : special_month_tips = special_month_fraction * (average_other_months * (total_months - 1) + special_month_tips))
  (h4 : ∃ (m : Real), special_month_tips = m * average_other_months) :
  ∃ (m : Real), special_month_tips = 6 * average_other_months :=
by sorry

end NUMINAMATH_CALUDE_resort_tips_multiple_l4034_403441


namespace NUMINAMATH_CALUDE_log_ride_cost_l4034_403452

def ferris_wheel_cost : ℕ := 6
def roller_coaster_cost : ℕ := 5
def initial_tickets : ℕ := 2
def additional_tickets_needed : ℕ := 16

theorem log_ride_cost :
  ferris_wheel_cost + roller_coaster_cost + (additional_tickets_needed + initial_tickets - ferris_wheel_cost - roller_coaster_cost) = additional_tickets_needed + initial_tickets :=
by sorry

end NUMINAMATH_CALUDE_log_ride_cost_l4034_403452


namespace NUMINAMATH_CALUDE_money_distribution_l4034_403469

/-- Given three people A, B, and C with money, prove that B and C together have 320 Rs. -/
theorem money_distribution (A B C : ℕ) 
  (total : A + B + C = 500)
  (ac_sum : A + C = 200)
  (c_amount : C = 20) : 
  B + C = 320 := by
  sorry

end NUMINAMATH_CALUDE_money_distribution_l4034_403469


namespace NUMINAMATH_CALUDE_set_equality_l4034_403498

theorem set_equality : 
  let M : Set ℝ := {3, 2}
  let N : Set ℝ := {x | x^2 - 5*x + 6 = 0}
  M = N := by sorry

end NUMINAMATH_CALUDE_set_equality_l4034_403498


namespace NUMINAMATH_CALUDE_constant_value_c_l4034_403407

theorem constant_value_c (b c : ℝ) : 
  (∀ x : ℝ, (x + 3) * (x + b) = x^2 + c*x + 12) → c = 7 := by
  sorry

end NUMINAMATH_CALUDE_constant_value_c_l4034_403407


namespace NUMINAMATH_CALUDE_gcd_9013_4357_l4034_403486

theorem gcd_9013_4357 : Nat.gcd 9013 4357 = 1 := by
  sorry

end NUMINAMATH_CALUDE_gcd_9013_4357_l4034_403486


namespace NUMINAMATH_CALUDE_existence_of_m_l4034_403420

def x : ℕ → ℚ
  | 0 => 7
  | n + 1 => (x n ^ 2 + 7 * x n + 8) / (x n + 8)

theorem existence_of_m : ∃ m : ℕ, 
  123 ≤ m ∧ m ≤ 242 ∧ 
  x m ≤ 6 + 1 / (2^18) ∧
  ∀ k : ℕ, 0 < k ∧ k < m → x k > 6 + 1 / (2^18) := by
  sorry

end NUMINAMATH_CALUDE_existence_of_m_l4034_403420


namespace NUMINAMATH_CALUDE_expression_equality_l4034_403435

theorem expression_equality : 2⁻¹ - Real.sqrt 3 * Real.tan (60 * π / 180) + (π - 2011)^0 + |(-1/2)| = -1 := by
  sorry

end NUMINAMATH_CALUDE_expression_equality_l4034_403435


namespace NUMINAMATH_CALUDE_smallest_room_length_l4034_403447

/-- Given two rectangular rooms, where the larger room has dimensions 45 feet by 30 feet,
    and the smaller room has a width of 15 feet, if the difference in area between
    these two rooms is 1230 square feet, then the length of the smaller room is 8 feet. -/
theorem smallest_room_length
  (larger_width : ℝ) (larger_length : ℝ)
  (smaller_width : ℝ) (smaller_length : ℝ)
  (area_difference : ℝ) :
  larger_width = 45 →
  larger_length = 30 →
  smaller_width = 15 →
  area_difference = 1230 →
  larger_width * larger_length - smaller_width * smaller_length = area_difference →
  smaller_length = 8 :=
by sorry

end NUMINAMATH_CALUDE_smallest_room_length_l4034_403447


namespace NUMINAMATH_CALUDE_complex_number_properties_l4034_403484

/-- Given a complex number z and a real number m, where z = m^2 - m - 2 + (5m^2 - 20)i -/
theorem complex_number_properties (m : ℝ) (z : ℂ) 
  (h : z = (m^2 - m - 2 : ℝ) + (5 * m^2 - 20 : ℝ) * Complex.I) :
  (z.im = 0 ↔ m = 2 ∨ m = -2) ∧ 
  (z.re = 0 ∧ z.im ≠ 0 ↔ m = -1) := by
  sorry

end NUMINAMATH_CALUDE_complex_number_properties_l4034_403484


namespace NUMINAMATH_CALUDE_potato_harvest_problem_l4034_403488

theorem potato_harvest_problem :
  ∃! (x y : ℕ+), 
    x * y * 5 = 45715 ∧ 
    x ≤ 100 ∧  -- reasonable upper bound for number of students
    y ≤ 1000   -- reasonable upper bound for daily output per student
  := by sorry

end NUMINAMATH_CALUDE_potato_harvest_problem_l4034_403488


namespace NUMINAMATH_CALUDE_negation_or_implies_both_false_l4034_403462

theorem negation_or_implies_both_false (p q : Prop) :
  ¬(p ∨ q) → (¬p ∧ ¬q) := by
  sorry

end NUMINAMATH_CALUDE_negation_or_implies_both_false_l4034_403462


namespace NUMINAMATH_CALUDE_john_total_distance_l4034_403490

/-- Calculates the total distance driven given two separate trips with different speeds and durations. -/
def total_distance (speed1 : ℝ) (time1 : ℝ) (speed2 : ℝ) (time2 : ℝ) : ℝ :=
  speed1 * time1 + speed2 * time2

/-- Proves that John's total driving distance is 235 miles. -/
theorem john_total_distance :
  total_distance 35 2 55 3 = 235 := by
  sorry

end NUMINAMATH_CALUDE_john_total_distance_l4034_403490


namespace NUMINAMATH_CALUDE_inequality_implication_l4034_403443

theorem inequality_implication (a b : ℝ) (h : a > b) : b - a < 0 := by
  sorry

end NUMINAMATH_CALUDE_inequality_implication_l4034_403443


namespace NUMINAMATH_CALUDE_T_properties_l4034_403480

-- Define the operation T
def T (a b x y : ℚ) : ℚ := a * x * y + b * x - 4

-- State the theorem
theorem T_properties (a b : ℚ) (ha : a ≠ 0) (hb : b ≠ 0) 
  (h1 : T a b 2 1 = 2) (h2 : T a b (-1) 2 = -8) :
  (a = 1 ∧ b = 2) ∧ 
  (∀ m n, n ≠ -2 → T a b m n = 0 → m = 4 / (n + 2)) ∧
  (∀ k x y, (∀ k', T a b (k' * x) y = T a b (k * x) y) → y = -2) ∧
  (∀ x y : ℚ, (∀ k, T a b (k * x) y = T a b (k * y) x) → k = 0) :=
by sorry

end NUMINAMATH_CALUDE_T_properties_l4034_403480


namespace NUMINAMATH_CALUDE_remaining_amount_after_expenses_l4034_403401

def bonus : ℚ := 1496
def kitchen_fraction : ℚ := 1 / 22
def holiday_fraction : ℚ := 1 / 4
def gift_fraction : ℚ := 1 / 8

theorem remaining_amount_after_expenses : 
  bonus - (bonus * kitchen_fraction + bonus * holiday_fraction + bonus * gift_fraction) = 867 := by
  sorry

end NUMINAMATH_CALUDE_remaining_amount_after_expenses_l4034_403401


namespace NUMINAMATH_CALUDE_sara_marbles_l4034_403410

def marble_problem (initial : ℕ) (given : ℕ) (lost : ℕ) (traded : ℕ) : Prop :=
  initial + given - lost - traded = 5

theorem sara_marbles : marble_problem 10 5 7 3 := by
  sorry

end NUMINAMATH_CALUDE_sara_marbles_l4034_403410


namespace NUMINAMATH_CALUDE_arithmetic_sequence_property_l4034_403450

/-- An arithmetic sequence -/
def ArithmeticSequence (a : ℕ → ℤ) : Prop :=
  ∃ d : ℤ, ∀ n : ℕ, a (n + 1) = a n + d

/-- Theorem: In an arithmetic sequence where a₄ = -4 and a₈ = 4, a₁₂ = 12 -/
theorem arithmetic_sequence_property (a : ℕ → ℤ) 
  (h_arith : ArithmeticSequence a) 
  (h_a4 : a 4 = -4) 
  (h_a8 : a 8 = 4) : 
  a 12 = 12 := by
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_property_l4034_403450


namespace NUMINAMATH_CALUDE_fraction_simplification_l4034_403495

theorem fraction_simplification (x : ℝ) (h : x = 5) : 
  (x^4 + 12*x^2 + 36) / (x^2 + 6) = 31 := by
  sorry

end NUMINAMATH_CALUDE_fraction_simplification_l4034_403495


namespace NUMINAMATH_CALUDE_car_speed_l4034_403454

/-- Given a car traveling 810 km in 5 hours, its speed is 162 km/h -/
theorem car_speed (distance : ℝ) (time : ℝ) (speed : ℝ) 
  (h1 : distance = 810) 
  (h2 : time = 5) 
  (h3 : speed = distance / time) : 
  speed = 162 := by
sorry

end NUMINAMATH_CALUDE_car_speed_l4034_403454


namespace NUMINAMATH_CALUDE_complex_equation_roots_l4034_403483

theorem complex_equation_roots : 
  let z₁ : ℂ := (1 + 2 * Real.sqrt 7 - Complex.I * Real.sqrt 7) / 2
  let z₂ : ℂ := (1 - 2 * Real.sqrt 7 + Complex.I * Real.sqrt 7) / 2
  (z₁ ^ 2 - z₁ = 3 - 7 * Complex.I) ∧ (z₂ ^ 2 - z₂ = 3 - 7 * Complex.I) := by
  sorry

end NUMINAMATH_CALUDE_complex_equation_roots_l4034_403483


namespace NUMINAMATH_CALUDE_sum_lent_calculation_l4034_403421

/-- Calculates the sum lent given the interest rate, time period, and interest amount -/
theorem sum_lent_calculation (interest_rate : ℚ) (years : ℕ) (interest_difference : ℚ) : 
  interest_rate = 5 / 100 →
  years = 8 →
  interest_difference = 360 →
  (1 - years * interest_rate) * 600 = interest_difference :=
by
  sorry

end NUMINAMATH_CALUDE_sum_lent_calculation_l4034_403421


namespace NUMINAMATH_CALUDE_divisibility_by_six_l4034_403455

theorem divisibility_by_six (n : ℕ) 
  (div_by_two : ∃ k : ℕ, n = 2 * k) 
  (div_by_three : ∃ m : ℕ, n = 3 * m) : 
  ∃ p : ℕ, n = 6 * p := by
sorry

end NUMINAMATH_CALUDE_divisibility_by_six_l4034_403455


namespace NUMINAMATH_CALUDE_shoe_multiple_l4034_403464

/-- Given the following conditions:
  - Bonny has 13 pairs of shoes
  - Bonny's shoes are 5 less than a certain multiple of Becky's shoes
  - Bobby has 3 times as many shoes as Becky
  - Bobby has 27 pairs of shoes
  Prove that the multiple of Becky's shoes that is 5 more than Bonny's shoes is 2. -/
theorem shoe_multiple (bonny_shoes : ℕ) (bobby_shoes : ℕ) (becky_shoes : ℕ) (m : ℕ) :
  bonny_shoes = 13 →
  ∃ m, bonny_shoes + 5 = m * becky_shoes →
  bobby_shoes = 3 * becky_shoes →
  bobby_shoes = 27 →
  m = 2 :=
by sorry

end NUMINAMATH_CALUDE_shoe_multiple_l4034_403464


namespace NUMINAMATH_CALUDE_coin_payment_difference_l4034_403476

/-- Represents the available coin denominations in cents -/
inductive Coin : Type
  | OneCent : Coin
  | TenCent : Coin
  | TwentyCent : Coin

/-- The value of a coin in cents -/
def coin_value : Coin → ℕ
  | Coin.OneCent => 1
  | Coin.TenCent => 10
  | Coin.TwentyCent => 20

/-- A function that returns true if a list of coins sums to the target amount -/
def sum_to_target (coins : List Coin) (target : ℕ) : Prop :=
  (coins.map coin_value).sum = target

/-- The proposition to be proved -/
theorem coin_payment_difference (target : ℕ := 50) :
  ∃ (min_coins max_coins : List Coin),
    sum_to_target min_coins target ∧
    sum_to_target max_coins target ∧
    (max_coins.length - min_coins.length = 47) :=
  sorry

end NUMINAMATH_CALUDE_coin_payment_difference_l4034_403476


namespace NUMINAMATH_CALUDE_margies_change_l4034_403415

/-- Calculates the change Margie receives after buying oranges -/
def margieChange (numOranges : ℕ) (costPerOrange : ℚ) (amountPaid : ℚ) : ℚ :=
  amountPaid - (numOranges : ℚ) * costPerOrange

/-- Theorem stating that Margie's change is $8.50 -/
theorem margies_change :
  margieChange 5 (30 / 100) 10 = 17 / 2 := by
  sorry

#eval margieChange 5 (30 / 100) 10

end NUMINAMATH_CALUDE_margies_change_l4034_403415


namespace NUMINAMATH_CALUDE_negation_equivalence_l4034_403449

theorem negation_equivalence :
  (¬ ∃ x : ℝ, x ≥ 0 ∧ x^2 > 3) ↔ (∀ x : ℝ, x ≥ 0 → x^2 ≤ 3) :=
by sorry

end NUMINAMATH_CALUDE_negation_equivalence_l4034_403449


namespace NUMINAMATH_CALUDE_zero_of_f_l4034_403431

def f (x : ℝ) : ℝ := 4 * x - 2

theorem zero_of_f : ∃ x : ℝ, f x = 0 ∧ x = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_zero_of_f_l4034_403431


namespace NUMINAMATH_CALUDE_cube_difference_positive_l4034_403456

theorem cube_difference_positive {a b : ℝ} (h : a > b) : a^3 - b^3 > 0 := by
  sorry

end NUMINAMATH_CALUDE_cube_difference_positive_l4034_403456


namespace NUMINAMATH_CALUDE_divisibility_by_37_l4034_403467

theorem divisibility_by_37 (a b c : ℕ) :
  let p := 100 * a + 10 * b + c
  let q := 100 * b + 10 * c + a
  let r := 100 * c + 10 * a + b
  37 ∣ p → (37 ∣ q ∧ 37 ∣ r) := by
sorry

end NUMINAMATH_CALUDE_divisibility_by_37_l4034_403467


namespace NUMINAMATH_CALUDE_recurring_decimal_to_fraction_l4034_403479

theorem recurring_decimal_to_fraction :
  ∃ (x : ℚ), x = 3 + 145 / 999 ∧ x = 3142 / 999 := by
  sorry

end NUMINAMATH_CALUDE_recurring_decimal_to_fraction_l4034_403479


namespace NUMINAMATH_CALUDE_fraction_inequality_implies_inequality_l4034_403437

theorem fraction_inequality_implies_inequality (a b c : ℝ) :
  c ≠ 0 → (a / c^2 < b / c^2) → a < b := by
  sorry

end NUMINAMATH_CALUDE_fraction_inequality_implies_inequality_l4034_403437


namespace NUMINAMATH_CALUDE_sammy_remaining_problems_l4034_403477

theorem sammy_remaining_problems (total : ℕ) (fractions decimals multiplication division : ℕ)
  (completed_fractions completed_decimals completed_multiplication completed_division : ℕ)
  (h1 : total = 115)
  (h2 : fractions = 35)
  (h3 : decimals = 40)
  (h4 : multiplication = 20)
  (h5 : division = 20)
  (h6 : completed_fractions = 11)
  (h7 : completed_decimals = 17)
  (h8 : completed_multiplication = 9)
  (h9 : completed_division = 5)
  (h10 : total = fractions + decimals + multiplication + division) :
  total - (completed_fractions + completed_decimals + completed_multiplication + completed_division) = 73 := by
sorry

end NUMINAMATH_CALUDE_sammy_remaining_problems_l4034_403477


namespace NUMINAMATH_CALUDE_floor_times_x_eq_152_l4034_403499

theorem floor_times_x_eq_152 : ∃ x : ℝ, (⌊x⌋ : ℝ) * x = 152 ∧ x = 38 / 3 := by
  sorry

end NUMINAMATH_CALUDE_floor_times_x_eq_152_l4034_403499


namespace NUMINAMATH_CALUDE_triangle_inequality_l4034_403481

/-- Checks if three lengths can form a valid triangle --/
def is_valid_triangle (a b c : ℝ) : Prop :=
  a + b > c ∧ b + c > a ∧ c + a > b

theorem triangle_inequality (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  ¬(is_valid_triangle a b 15) ∧ is_valid_triangle a b 13 :=
by
  sorry

#check triangle_inequality 8 7

end NUMINAMATH_CALUDE_triangle_inequality_l4034_403481


namespace NUMINAMATH_CALUDE_complex_sum_magnitude_l4034_403427

theorem complex_sum_magnitude (a b c : ℂ) 
  (h1 : Complex.abs a = 1) 
  (h2 : Complex.abs b = 1) 
  (h3 : Complex.abs c = 1) 
  (h4 : a^3 / (b*c) + b^3 / (a*c) + c^3 / (a*b) = -3) : 
  Complex.abs (a + b + c) = 1 := by
  sorry

end NUMINAMATH_CALUDE_complex_sum_magnitude_l4034_403427


namespace NUMINAMATH_CALUDE_ceiling_negative_three_point_seven_l4034_403473

theorem ceiling_negative_three_point_seven :
  ⌈(-3.7 : ℝ)⌉ = -3 := by sorry

end NUMINAMATH_CALUDE_ceiling_negative_three_point_seven_l4034_403473


namespace NUMINAMATH_CALUDE_opposite_signs_inequality_l4034_403489

theorem opposite_signs_inequality (a b : ℝ) (h : a * b < 0) : |a + b| < |a - b| := by
  sorry

end NUMINAMATH_CALUDE_opposite_signs_inequality_l4034_403489


namespace NUMINAMATH_CALUDE_meeting_probability_l4034_403487

-- Define the time range in minutes
def timeRange : ℝ := 60

-- Define the waiting time in minutes
def waitTime : ℝ := 10

-- Define the probability of meeting function
def probabilityOfMeeting (arrivalRange1 : ℝ) (arrivalRange2 : ℝ) : ℚ :=
  sorry

theorem meeting_probability :
  (probabilityOfMeeting timeRange timeRange = 11/36) ∧
  (probabilityOfMeeting (timeRange/2) timeRange = 11/36) ∧
  (probabilityOfMeeting (5*timeRange/6) timeRange = 19/60) :=
sorry

end NUMINAMATH_CALUDE_meeting_probability_l4034_403487


namespace NUMINAMATH_CALUDE_nellie_legos_l4034_403434

theorem nellie_legos (initial : ℕ) (lost : ℕ) (given_away : ℕ) :
  initial ≥ lost + given_away →
  initial - (lost + given_away) = initial - lost - given_away :=
by
  sorry

#check nellie_legos 380 57 24

end NUMINAMATH_CALUDE_nellie_legos_l4034_403434


namespace NUMINAMATH_CALUDE_helium_lowest_liquefaction_temp_l4034_403472

-- Define the gases
inductive Gas : Type
| Oxygen
| Hydrogen
| Nitrogen
| Helium

-- Define the liquefaction temperature function
def liquefaction_temp : Gas → ℝ
| Gas.Oxygen => -183
| Gas.Hydrogen => -253
| Gas.Nitrogen => -195.8
| Gas.Helium => -268

-- Statement to prove
theorem helium_lowest_liquefaction_temp :
  ∀ g : Gas, liquefaction_temp Gas.Helium ≤ liquefaction_temp g :=
by sorry

end NUMINAMATH_CALUDE_helium_lowest_liquefaction_temp_l4034_403472


namespace NUMINAMATH_CALUDE_corn_acreage_l4034_403422

theorem corn_acreage (total_land : ℕ) (beans_ratio wheat_ratio corn_ratio : ℕ) 
  (h1 : total_land = 1034)
  (h2 : beans_ratio = 5)
  (h3 : wheat_ratio = 2)
  (h4 : corn_ratio = 4) : 
  (total_land * corn_ratio) / (beans_ratio + wheat_ratio + corn_ratio) = 376 := by
  sorry

end NUMINAMATH_CALUDE_corn_acreage_l4034_403422


namespace NUMINAMATH_CALUDE_bakery_sugar_amount_l4034_403418

/-- Given the ratios of ingredients in a bakery storage room, prove the amount of sugar. -/
theorem bakery_sugar_amount
  (sugar flour baking_soda : ℕ)
  (h1 : sugar = flour)  -- Sugar to flour ratio is 5:5, which simplifies to 1:1
  (h2 : flour = 10 * baking_soda)  -- Flour to baking soda ratio is 10:1
  (h3 : flour = 8 * (baking_soda + 60))  -- If 60 more pounds of baking soda were added, 
                                          -- the ratio of flour to baking soda would be 8:1
  : sugar = 2400 := by
  sorry


end NUMINAMATH_CALUDE_bakery_sugar_amount_l4034_403418


namespace NUMINAMATH_CALUDE_complex_distance_to_origin_l4034_403451

theorem complex_distance_to_origin : 
  let z : ℂ := (I^2016 - 2*I^2014) / (2 - I)^2
  Complex.abs z = 3/5 := by
  sorry

end NUMINAMATH_CALUDE_complex_distance_to_origin_l4034_403451


namespace NUMINAMATH_CALUDE_simplify_expression_l4034_403493

theorem simplify_expression (x : ℝ) : (3 * x)^3 + (2 * x) * (x^4) = 27 * x^3 + 2 * x^5 := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l4034_403493


namespace NUMINAMATH_CALUDE_statue_original_cost_l4034_403478

theorem statue_original_cost (selling_price : ℝ) (profit_percentage : ℝ) 
  (h1 : selling_price = 660)
  (h2 : profit_percentage = 20) :
  let original_cost := selling_price / (1 + profit_percentage / 100)
  original_cost = 550 := by
sorry

end NUMINAMATH_CALUDE_statue_original_cost_l4034_403478


namespace NUMINAMATH_CALUDE_lebesgue_decomposition_l4034_403457

variable (E : Type) [MeasurableSpace E]
variable (μ ν : Measure E)

/-- Lebesgue decomposition theorem -/
theorem lebesgue_decomposition :
  ∃ (f : E → ℝ) (D : Set E),
    MeasurableSet D ∧
    (∀ x, 0 ≤ f x) ∧
    Measurable f ∧
    ν D = 0 ∧
    (∀ (B : Set E), MeasurableSet B →
      μ B = ∫ x in B, f x ∂ν + μ (B ∩ D)) ∧
    (∀ (g : E → ℝ) (C : Set E),
      MeasurableSet C →
      (∀ x, 0 ≤ g x) →
      Measurable g →
      ν C = 0 →
      (∀ (B : Set E), MeasurableSet B →
        μ B = ∫ x in B, g x ∂ν + μ (B ∩ C)) →
      (μ (D Δ C) = 0 ∧ ν {x | f x ≠ g x} = 0)) :=
sorry

end NUMINAMATH_CALUDE_lebesgue_decomposition_l4034_403457


namespace NUMINAMATH_CALUDE_first_month_sale_is_5400_l4034_403438

/-- Calculates the sale in the first month given the sales for the next 5 months and the average sale for 6 months -/
def first_month_sale (sale2 sale3 sale4 sale5 sale6 average : ℕ) : ℕ :=
  6 * average - (sale2 + sale3 + sale4 + sale5 + sale6)

/-- Theorem stating that the sale in the first month is 5400 given the specific sales figures -/
theorem first_month_sale_is_5400 :
  first_month_sale 9000 6300 7200 4500 1200 5600 = 5400 := by
  sorry

#eval first_month_sale 9000 6300 7200 4500 1200 5600

end NUMINAMATH_CALUDE_first_month_sale_is_5400_l4034_403438


namespace NUMINAMATH_CALUDE_cistern_fill_time_l4034_403400

theorem cistern_fill_time (fill_time : ℝ) (empty_time : ℝ) (h1 : fill_time = 10) (h2 : empty_time = 12) :
  let net_fill_rate := 1 / fill_time - 1 / empty_time
  1 / net_fill_rate = 60 := by sorry

end NUMINAMATH_CALUDE_cistern_fill_time_l4034_403400


namespace NUMINAMATH_CALUDE_smallest_q_for_decimal_sequence_l4034_403429

theorem smallest_q_for_decimal_sequence (p q : ℕ+) : 
  (p : ℚ) / q = 0.123456789 → q ≥ 10989019 := by sorry

end NUMINAMATH_CALUDE_smallest_q_for_decimal_sequence_l4034_403429


namespace NUMINAMATH_CALUDE_julies_earnings_l4034_403426

/-- Calculates Julie's earnings for landscaping services --/
def calculate_earnings (
  lawn_rate : ℚ)
  (weed_rate : ℚ)
  (prune_rate : ℚ)
  (mulch_rate : ℚ)
  (lawn_hours_sept : ℚ)
  (weed_hours_sept : ℚ)
  (prune_hours_sept : ℚ)
  (mulch_hours_sept : ℚ) : ℚ :=
  let sept_earnings := 
    lawn_rate * lawn_hours_sept +
    weed_rate * weed_hours_sept +
    prune_rate * prune_hours_sept +
    mulch_rate * mulch_hours_sept
  let oct_earnings := 
    lawn_rate * (lawn_hours_sept * 1.5) +
    weed_rate * (weed_hours_sept * 1.5) +
    prune_rate * (prune_hours_sept * 1.5) +
    mulch_rate * (mulch_hours_sept * 1.5)
  sept_earnings + oct_earnings

/-- Theorem: Julie's total earnings for September and October --/
theorem julies_earnings : 
  calculate_earnings 4 8 10 12 25 3 10 5 = 710 := by
  sorry

end NUMINAMATH_CALUDE_julies_earnings_l4034_403426


namespace NUMINAMATH_CALUDE_max_value_of_sum_products_l4034_403419

theorem max_value_of_sum_products (a b c : ℝ) 
  (h1 : a ≥ 0) (h2 : b ≥ 0) (h3 : c ≥ 0) 
  (h4 : a^2 + b^2 + c^2 = 3) : 
  ∀ x y z : ℝ, x ≥ 0 → y ≥ 0 → z ≥ 0 → x^2 + y^2 + z^2 = 3 → 
  a*b + b*c + c*a ≥ x*y + y*z + z*x :=
sorry

end NUMINAMATH_CALUDE_max_value_of_sum_products_l4034_403419


namespace NUMINAMATH_CALUDE_johnsons_share_l4034_403445

/-- 
Given a profit-sharing ratio and Mike's total share, calculate Johnson's share.
-/
theorem johnsons_share 
  (mike_ratio : ℕ) 
  (johnson_ratio : ℕ) 
  (mike_total_share : ℕ) : 
  mike_ratio = 2 → 
  johnson_ratio = 5 → 
  mike_total_share = 1000 → 
  (mike_total_share * johnson_ratio) / mike_ratio = 2500 := by
  sorry

#check johnsons_share

end NUMINAMATH_CALUDE_johnsons_share_l4034_403445


namespace NUMINAMATH_CALUDE_ellipse_k_range_l4034_403475

/-- An ellipse with equation x^2 + ky^2 = 2 and foci on the y-axis -/
structure EllipseOnYAxis where
  k : ℝ
  is_ellipse : k > 0
  foci_on_y_axis : k < 1

/-- The range of k for an ellipse with equation x^2 + ky^2 = 2 and foci on the y-axis is (0, 1) -/
theorem ellipse_k_range (e : EllipseOnYAxis) : 0 < e.k ∧ e.k < 1 := by
  sorry

end NUMINAMATH_CALUDE_ellipse_k_range_l4034_403475


namespace NUMINAMATH_CALUDE_builder_project_l4034_403482

/-- A builder's project involving bolts and nuts -/
theorem builder_project (bolt_boxes : ℕ) (bolts_per_box : ℕ) (nut_boxes : ℕ) (nuts_per_box : ℕ)
  (leftover_bolts : ℕ) (leftover_nuts : ℕ) :
  bolt_boxes = 7 →
  bolts_per_box = 11 →
  nut_boxes = 3 →
  nuts_per_box = 15 →
  leftover_bolts = 3 →
  leftover_nuts = 6 →
  (bolt_boxes * bolts_per_box - leftover_bolts) + (nut_boxes * nuts_per_box - leftover_nuts) = 113 :=
by sorry

end NUMINAMATH_CALUDE_builder_project_l4034_403482


namespace NUMINAMATH_CALUDE_problem_statement_l4034_403474

theorem problem_statement (a b : ℝ) (h : a + b = 3) : 2*a^2 + 4*a*b + 2*b^2 - 4 = 14 := by
  sorry

end NUMINAMATH_CALUDE_problem_statement_l4034_403474


namespace NUMINAMATH_CALUDE_work_distribution_l4034_403423

theorem work_distribution (p : ℕ) (x : ℚ) (h1 : 0 < p) (h2 : 0 ≤ x) (h3 : x < 1) :
  p * 1 = (1 - x) * p * (3/2) → x = 1/3 := by
  sorry

end NUMINAMATH_CALUDE_work_distribution_l4034_403423


namespace NUMINAMATH_CALUDE_speed_conversion_equivalence_l4034_403485

/-- Conversion factor from meters per second to kilometers per hour -/
def mps_to_kmph : ℝ := 3.6

/-- The given speed in meters per second -/
def given_speed_mps : ℝ := 35.0028

/-- The calculated speed in kilometers per hour -/
def calculated_speed_kmph : ℝ := 126.01008

theorem speed_conversion_equivalence : 
  given_speed_mps * mps_to_kmph = calculated_speed_kmph := by
  sorry

end NUMINAMATH_CALUDE_speed_conversion_equivalence_l4034_403485


namespace NUMINAMATH_CALUDE_ellipse_and_midpoint_trajectory_l4034_403425

/-- Definition of the ellipse -/
def Ellipse (x y : ℝ) : Prop := x^2 / 4 + y^2 = 1

/-- Definition of the midpoint trajectory -/
def MidpointTrajectory (x y : ℝ) : Prop := (x - 1/2)^2 + 4 * (y - 1/4)^2 = 1

/-- Theorem: The standard equation of the ellipse and the midpoint trajectory -/
theorem ellipse_and_midpoint_trajectory :
  (∀ x y, Ellipse x y ↔ x^2 / 4 + y^2 = 1) ∧
  (∀ x₀ y₀ x y, Ellipse x₀ y₀ → x = (x₀ + 1) / 2 ∧ y = (y₀ + 1/2) / 2 → MidpointTrajectory x y) := by
  sorry

end NUMINAMATH_CALUDE_ellipse_and_midpoint_trajectory_l4034_403425


namespace NUMINAMATH_CALUDE_otimes_result_l4034_403468

/-- Definition of the ⊗ operation -/
def otimes (a b : ℚ) (x y : ℚ) : ℚ := a^2 * x + b * y - 3

/-- Theorem stating that 2 ⊗ (-6) = 7 given 1 ⊗ (-3) = 2 -/
theorem otimes_result (a b : ℚ) (h : otimes a b 1 (-3) = 2) : otimes a b 2 (-6) = 7 := by
  sorry

end NUMINAMATH_CALUDE_otimes_result_l4034_403468


namespace NUMINAMATH_CALUDE_average_roots_quadratic_l4034_403403

theorem average_roots_quadratic (a b c : ℝ) (h : b^2 - 4*a*c > 0) :
  let x₁ := (-b + Real.sqrt (b^2 - 4*a*c)) / (2*a)
  let x₂ := (-b - Real.sqrt (b^2 - 4*a*c)) / (2*a)
  a ≠ 0 → 3*x^2 + 4*x - 8 = 0 → (x₁ + x₂) / 2 = -2/3 :=
by sorry

end NUMINAMATH_CALUDE_average_roots_quadratic_l4034_403403


namespace NUMINAMATH_CALUDE_solution_set_when_a_is_one_a_value_when_minimum_is_four_l4034_403466

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := |2*x - 1| + |a*x - 5|

-- Part 1
theorem solution_set_when_a_is_one :
  let a : ℝ := 1
  {x : ℝ | f a x ≥ 9} = {x : ℝ | x ≤ -1 ∨ x > 5} := by sorry

-- Part 2
theorem a_value_when_minimum_is_four :
  ∃ (a : ℝ), 0 < a ∧ a < 5 ∧ 
  (∀ x : ℝ, f a x ≥ 4) ∧
  (∃ x : ℝ, f a x = 4) →
  a = 2 := by sorry

end NUMINAMATH_CALUDE_solution_set_when_a_is_one_a_value_when_minimum_is_four_l4034_403466


namespace NUMINAMATH_CALUDE_sum_of_reciprocals_l4034_403440

theorem sum_of_reciprocals (x y : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) 
  (h : x + y = 8 * x * y) : 1 / x + 1 / y = 8 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_reciprocals_l4034_403440


namespace NUMINAMATH_CALUDE_garage_sale_items_count_l4034_403448

theorem garage_sale_items_count 
  (prices : Finset ℕ) 
  (radio_price : ℕ) 
  (h_distinct : prices.card = prices.toList.length)
  (h_ninth_highest : (prices.filter (· > radio_price)).card = 8)
  (h_thirty_fifth_lowest : (prices.filter (· < radio_price)).card = 34)
  (h_radio_in_prices : radio_price ∈ prices) :
  prices.card = 43 := by
sorry

end NUMINAMATH_CALUDE_garage_sale_items_count_l4034_403448


namespace NUMINAMATH_CALUDE_apples_left_l4034_403460

theorem apples_left (initial_apples used_apples : ℕ) 
  (h1 : initial_apples = 43)
  (h2 : used_apples = 41) :
  initial_apples - used_apples = 2 := by
  sorry

end NUMINAMATH_CALUDE_apples_left_l4034_403460


namespace NUMINAMATH_CALUDE_triangle_side_length_l4034_403461

theorem triangle_side_length 
  (A B C : ℝ) 
  (h1 : Real.cos (A - 2*B) + Real.sin (2*A + B) = 2)
  (h2 : 0 < A ∧ A < π)
  (h3 : 0 < B ∧ B < π)
  (h4 : 0 < C ∧ C < π)
  (h5 : A + B + C = π)
  (h6 : BC = 6) :
  AB = 3 * (Real.sqrt 5 + 1) :=
sorry

end NUMINAMATH_CALUDE_triangle_side_length_l4034_403461


namespace NUMINAMATH_CALUDE_sample_product_l4034_403402

/-- Given a sample of five numbers (7, 8, 9, x, y) with an average of 8 
    and a standard deviation of √2, prove that xy = 60 -/
theorem sample_product (x y : ℝ) : 
  (7 + 8 + 9 + x + y) / 5 = 8 → 
  Real.sqrt (((7 - 8)^2 + (8 - 8)^2 + (9 - 8)^2 + (x - 8)^2 + (y - 8)^2) / 5) = Real.sqrt 2 →
  x * y = 60 := by
  sorry

end NUMINAMATH_CALUDE_sample_product_l4034_403402


namespace NUMINAMATH_CALUDE_part_one_part_two_l4034_403433

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := |x + 2*a|

-- Part 1
theorem part_one (a : ℝ) : 
  (∀ x : ℝ, f a x < 4 - 2*a ↔ -4 < x ∧ x < 4) → a = 0 := by sorry

-- Part 2
theorem part_two : 
  (∀ x : ℝ, f 1 x - f 1 (-2*x) ≤ x + 2) ∧ 
  (∀ m : ℝ, (∀ x : ℝ, f 1 x - f 1 (-2*x) ≤ x + m) → m ≥ 2) := by sorry

end NUMINAMATH_CALUDE_part_one_part_two_l4034_403433


namespace NUMINAMATH_CALUDE_johns_current_income_l4034_403413

/-- Calculates John's current yearly income based on tax rates and tax increase --/
theorem johns_current_income
  (initial_tax_rate : ℝ)
  (new_tax_rate : ℝ)
  (initial_income : ℝ)
  (tax_increase : ℝ)
  (h1 : initial_tax_rate = 0.20)
  (h2 : new_tax_rate = 0.30)
  (h3 : initial_income = 1000000)
  (h4 : tax_increase = 250000) :
  ∃ current_income : ℝ,
    current_income = 1500000 ∧
    new_tax_rate * current_income - initial_tax_rate * initial_income = tax_increase :=
by
  sorry


end NUMINAMATH_CALUDE_johns_current_income_l4034_403413


namespace NUMINAMATH_CALUDE_factor_expression_l4034_403458

theorem factor_expression (x : ℝ) : 81 * x^3 + 27 * x^2 = 27 * x^2 * (3 * x + 1) := by
  sorry

end NUMINAMATH_CALUDE_factor_expression_l4034_403458


namespace NUMINAMATH_CALUDE_root_value_range_l4034_403414

theorem root_value_range (a : ℝ) (h : a^2 + 3*a - 1 = 0) :
  2 < a^2 + 3*a + Real.sqrt 3 ∧ a^2 + 3*a + Real.sqrt 3 < 3 := by
  sorry

end NUMINAMATH_CALUDE_root_value_range_l4034_403414


namespace NUMINAMATH_CALUDE_total_profit_is_8640_l4034_403492

/-- Represents the investment and profit distribution of a business partnership --/
structure BusinessPartnership where
  total_investment : ℕ
  a_investment : ℕ
  b_investment : ℕ
  c_investment : ℕ
  a_profit_share : ℕ

/-- Calculates the total profit based on the given business partnership --/
def calculate_total_profit (bp : BusinessPartnership) : ℕ :=
  let investment_ratio := bp.a_investment + bp.b_investment + bp.c_investment
  let profit_per_ratio := bp.a_profit_share * investment_ratio / bp.a_investment
  profit_per_ratio

/-- Theorem stating the total profit for the given business scenario --/
theorem total_profit_is_8640 (bp : BusinessPartnership) 
  (h1 : bp.total_investment = 90000)
  (h2 : bp.a_investment = bp.b_investment + 6000)
  (h3 : bp.c_investment = bp.b_investment + 3000)
  (h4 : bp.a_investment + bp.b_investment + bp.c_investment = bp.total_investment)
  (h5 : bp.a_profit_share = 3168) :
  calculate_total_profit bp = 8640 := by
  sorry

end NUMINAMATH_CALUDE_total_profit_is_8640_l4034_403492


namespace NUMINAMATH_CALUDE_gift_bags_production_time_l4034_403446

theorem gift_bags_production_time (total_bags : ℕ) (rate_per_day : ℕ) (h1 : total_bags = 519) (h2 : rate_per_day = 42) :
  (total_bags + rate_per_day - 1) / rate_per_day = 13 :=
sorry

end NUMINAMATH_CALUDE_gift_bags_production_time_l4034_403446


namespace NUMINAMATH_CALUDE_square_sum_de_l4034_403459

theorem square_sum_de (a b c d e : ℕ+) 
  (eq1 : (a + 1) * (3 * b * c + 1) = d + 3 * e + 1)
  (eq2 : (b + 1) * (3 * c * a + 1) = 3 * d + e + 13)
  (eq3 : (c + 1) * (3 * a * b + 1) = 4 * (26 - d - e) - 1) :
  d ^ 2 + e ^ 2 = 146 := by
  sorry

end NUMINAMATH_CALUDE_square_sum_de_l4034_403459


namespace NUMINAMATH_CALUDE_factor_statements_l4034_403428

theorem factor_statements : 
  (∃ n : ℤ, 30 = 5 * n) ∧ (∃ m : ℤ, 180 = 9 * m) := by sorry

end NUMINAMATH_CALUDE_factor_statements_l4034_403428


namespace NUMINAMATH_CALUDE_thirty_percent_of_hundred_l4034_403496

theorem thirty_percent_of_hundred : ∃ x : ℝ, 30 = 0.30 * x ∧ x = 100 := by
  sorry

end NUMINAMATH_CALUDE_thirty_percent_of_hundred_l4034_403496


namespace NUMINAMATH_CALUDE_smallest_marble_count_l4034_403436

def is_valid_marble_count (n : ℕ) : Prop :=
  n > 1 ∧ n % 5 = 2 ∧ n % 7 = 2 ∧ n % 9 = 2

theorem smallest_marble_count :
  ∃ (n : ℕ), is_valid_marble_count n ∧
  ∀ (m : ℕ), is_valid_marble_count m → n ≤ m :=
by
  use 317
  sorry

end NUMINAMATH_CALUDE_smallest_marble_count_l4034_403436


namespace NUMINAMATH_CALUDE_order_of_integrals_l4034_403416

theorem order_of_integrals : 
  let a : ℝ := ∫ x in (0:ℝ)..2, x^2
  let b : ℝ := ∫ x in (0:ℝ)..2, x^3
  let c : ℝ := ∫ x in (0:ℝ)..2, Real.sin x
  c < a ∧ a < b := by sorry

end NUMINAMATH_CALUDE_order_of_integrals_l4034_403416


namespace NUMINAMATH_CALUDE_remove_seven_maintain_coverage_l4034_403408

/-- Represents a collection of objects covering a surface -/
structure CoveringSet (n : ℕ) :=
  (area : ℝ)
  (total_coverage : ℝ)
  (coverage : Fin n → ℝ)
  (covers_completely : total_coverage = area)
  (non_negative_coverage : ∀ i, coverage i ≥ 0)
  (sum_coverage : (Finset.sum Finset.univ coverage) = total_coverage)

/-- Theorem stating that it's possible to remove 7 objects from a set of 15
    such that the remaining 8 cover at least 8/15 of the total area -/
theorem remove_seven_maintain_coverage 
  (s : CoveringSet 15) : 
  ∃ (removed : Finset (Fin 15)), 
    Finset.card removed = 7 ∧ 
    (Finset.sum (Finset.univ \ removed) s.coverage) ≥ (8/15) * s.area := by
  sorry

end NUMINAMATH_CALUDE_remove_seven_maintain_coverage_l4034_403408


namespace NUMINAMATH_CALUDE_jenny_mother_age_problem_l4034_403424

/-- Given that Jenny is 10 years old in 2010 and her mother's age is five times Jenny's age,
    prove that the year when Jenny's mother's age will be twice Jenny's age is 2040. -/
theorem jenny_mother_age_problem (jenny_age_2010 : ℕ) (mother_age_2010 : ℕ) :
  jenny_age_2010 = 10 →
  mother_age_2010 = 5 * jenny_age_2010 →
  ∃ (years_after_2010 : ℕ),
    mother_age_2010 + years_after_2010 = 2 * (jenny_age_2010 + years_after_2010) ∧
    2010 + years_after_2010 = 2040 :=
by sorry

end NUMINAMATH_CALUDE_jenny_mother_age_problem_l4034_403424


namespace NUMINAMATH_CALUDE_x_greater_than_half_l4034_403409

theorem x_greater_than_half (x : ℝ) (h : (1/2) * x = 1) : 
  (x - 1/2) / (1/2) * 100 = 300 := by
  sorry

end NUMINAMATH_CALUDE_x_greater_than_half_l4034_403409


namespace NUMINAMATH_CALUDE_new_tv_cost_l4034_403444

/-- The cost of a new TV given the dimensions and price of an old TV, and the price difference per square inch. -/
theorem new_tv_cost (old_width old_height old_cost new_width new_height price_diff : ℝ) :
  old_width = 24 →
  old_height = 16 →
  old_cost = 672 →
  new_width = 48 →
  new_height = 32 →
  price_diff = 1 →
  (old_cost / (old_width * old_height) - price_diff) * (new_width * new_height) = 1152 := by
  sorry

end NUMINAMATH_CALUDE_new_tv_cost_l4034_403444


namespace NUMINAMATH_CALUDE_baker_cake_difference_l4034_403453

/-- Given the initial number of cakes, the number of cakes sold, and the number of cakes bought,
    prove that the difference between cakes sold and cakes bought is 47. -/
theorem baker_cake_difference (initial : ℕ) (sold : ℕ) (bought : ℕ) 
    (h1 : initial = 170) (h2 : sold = 78) (h3 : bought = 31) : 
    sold - bought = 47 := by
  sorry

end NUMINAMATH_CALUDE_baker_cake_difference_l4034_403453

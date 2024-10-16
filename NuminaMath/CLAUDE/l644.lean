import Mathlib

namespace NUMINAMATH_CALUDE_prob_at_least_two_green_l644_64487

/-- The probability of choosing at least two green apples when randomly selecting 3 apples
    from a set of 10 apples (6 red and 4 green) is equal to 1/3. -/
theorem prob_at_least_two_green (total : ℕ) (red : ℕ) (green : ℕ) (choose : ℕ) :
  total = 10 →
  red = 6 →
  green = 4 →
  choose = 3 →
  (Nat.choose green 2 * Nat.choose red 1 + Nat.choose green 3) / Nat.choose total choose = 1 / 3 :=
by sorry

end NUMINAMATH_CALUDE_prob_at_least_two_green_l644_64487


namespace NUMINAMATH_CALUDE_sum_of_squares_lower_bound_l644_64410

theorem sum_of_squares_lower_bound (a b c : ℝ) (h : a + 2*b + 3*c = 4) :
  a^2 + b^2 + c^2 ≥ 8/7 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_squares_lower_bound_l644_64410


namespace NUMINAMATH_CALUDE_product_of_integers_with_given_sum_and_difference_l644_64407

theorem product_of_integers_with_given_sum_and_difference :
  ∀ x y : ℕ+, 
    (x : ℤ) + (y : ℤ) = 72 → 
    (x : ℤ) - (y : ℤ) = 18 → 
    (x : ℤ) * (y : ℤ) = 1215 := by
  sorry

end NUMINAMATH_CALUDE_product_of_integers_with_given_sum_and_difference_l644_64407


namespace NUMINAMATH_CALUDE_book_reading_fraction_l644_64485

theorem book_reading_fraction (total_pages : ℕ) (pages_read : ℕ) : 
  total_pages = 60 →
  pages_read = (total_pages - pages_read) + 20 →
  (pages_read : ℚ) / total_pages = 2 / 3 := by
  sorry

end NUMINAMATH_CALUDE_book_reading_fraction_l644_64485


namespace NUMINAMATH_CALUDE_rectangle_count_l644_64483

/-- Given a rectangle with dimensions a and b where a < b, this theorem states that
    the number of rectangles with dimensions x and y satisfying the specified conditions
    is either 0 or 1. -/
theorem rectangle_count (a b x y : ℝ) (h1 : 0 < a) (h2 : a < b) : 
  (x < a ∧ y < a ∧ 
   2*(x + y) = (1/2)*(a + b) ∧ 
   x*y = (1/4)*a*b) → 
  (∃! p : ℝ × ℝ, p.1 < a ∧ p.2 < a ∧ 
                 2*(p.1 + p.2) = (1/2)*(a + b) ∧ 
                 p.1*p.2 = (1/4)*a*b) ∨
  (¬ ∃ p : ℝ × ℝ, p.1 < a ∧ p.2 < a ∧ 
                  2*(p.1 + p.2) = (1/2)*(a + b) ∧ 
                  p.1*p.2 = (1/4)*a*b) :=
by sorry

end NUMINAMATH_CALUDE_rectangle_count_l644_64483


namespace NUMINAMATH_CALUDE_last_toggled_locker_l644_64445

theorem last_toggled_locker (n : Nat) (h : n = 2048) :
  (Nat.sqrt n) ^ 2 = 1936 := by
  sorry

end NUMINAMATH_CALUDE_last_toggled_locker_l644_64445


namespace NUMINAMATH_CALUDE_seokjin_drank_least_l644_64435

/-- Represents the amount of milk drunk by each person in liters -/
structure MilkConsumption where
  jungkook : ℝ
  seokjin : ℝ
  yoongi : ℝ

/-- Given the milk consumption of Jungkook, Seokjin, and Yoongi, 
    proves that Seokjin drank the least amount of milk -/
theorem seokjin_drank_least (m : MilkConsumption) 
  (h1 : m.jungkook = 1.3)
  (h2 : m.seokjin = 11/10)
  (h3 : m.yoongi = 7/5) : 
  m.seokjin < m.jungkook ∧ m.seokjin < m.yoongi := by
  sorry

#check seokjin_drank_least

end NUMINAMATH_CALUDE_seokjin_drank_least_l644_64435


namespace NUMINAMATH_CALUDE_man_money_calculation_l644_64458

/-- Calculates the total amount of money given the number of 50 and 500 rupee notes -/
def total_amount (n_50 n_500 : ℕ) : ℕ :=
  50 * n_50 + 500 * n_500

/-- Proves that a man with 36 notes, 17 of which are 50 rupee notes and the rest are 500 rupee notes, has 10350 rupees in total -/
theorem man_money_calculation :
  let total_notes : ℕ := 36
  let n_50 : ℕ := 17
  let n_500 : ℕ := total_notes - n_50
  total_amount n_50 n_500 = 10350 := by
  sorry

#eval total_amount 17 19

end NUMINAMATH_CALUDE_man_money_calculation_l644_64458


namespace NUMINAMATH_CALUDE_sum_of_cubes_mod_6_l644_64446

theorem sum_of_cubes_mod_6 (h : ∀ n : ℕ, n^3 % 6 = n % 6) :
  (Finset.sum (Finset.range 150) (fun i => (i + 1)^3)) % 6 = 5 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_cubes_mod_6_l644_64446


namespace NUMINAMATH_CALUDE_tina_postcard_earnings_l644_64498

/-- Tina's postcard business earnings calculation --/
theorem tina_postcard_earnings :
  let postcards_per_day : ℕ := 30
  let price_per_postcard : ℕ := 5
  let days_worked : ℕ := 6
  let total_postcards : ℕ := postcards_per_day * days_worked
  let total_earnings : ℕ := total_postcards * price_per_postcard
  total_earnings = 900 :=
by
  sorry

#check tina_postcard_earnings

end NUMINAMATH_CALUDE_tina_postcard_earnings_l644_64498


namespace NUMINAMATH_CALUDE_arithmetic_sequence_formula_l644_64474

def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_formula (a : ℕ → ℝ) (S : ℕ → ℝ) :
  (∀ n : ℕ, 2 * S n = n * a n) →
  a 2 = 1 →
  (∀ n : ℕ, n ≥ 1 → a n = n - 1) ∧ arithmetic_sequence a :=
by sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_formula_l644_64474


namespace NUMINAMATH_CALUDE_slope_intercept_product_l644_64467

/-- Given points A, B, C in a plane, and D as the midpoint of AB,
    prove that the product of the slope and y-intercept of line CD is -5/2 -/
theorem slope_intercept_product (A B C D : ℝ × ℝ) : 
  A = (0, 10) →
  B = (0, 0) →
  C = (10, 0) →
  D = ((A.1 + B.1) / 2, (A.2 + B.2) / 2) →
  let m := (C.2 - D.2) / (C.1 - D.1)
  let b := D.2
  m * b = -5/2 := by sorry

end NUMINAMATH_CALUDE_slope_intercept_product_l644_64467


namespace NUMINAMATH_CALUDE_c_range_l644_64451

theorem c_range (c : ℝ) (h_c_pos : c > 0) : 
  (((∀ x y : ℝ, x < y → c^x > c^y) ↔ ¬(∀ x : ℝ, x + c > 0)) ∧ 
   ((∀ x : ℝ, x + c > 0) ↔ ¬(∀ x y : ℝ, x < y → c^x > c^y))) → 
  (c > 0 ∧ c ≠ 1) :=
by sorry

end NUMINAMATH_CALUDE_c_range_l644_64451


namespace NUMINAMATH_CALUDE_sufficient_not_necessary_negation_l644_64493

theorem sufficient_not_necessary_negation (p q : Prop) 
  (h_sufficient : p → q) 
  (h_not_necessary : ¬(q → p)) : 
  (¬q → ¬p) ∧ ¬(¬p → ¬q) := by
  sorry

end NUMINAMATH_CALUDE_sufficient_not_necessary_negation_l644_64493


namespace NUMINAMATH_CALUDE_part1_part2_part3_l644_64454

-- Define the quadratic function f(x)
def f (a : ℝ) (x : ℝ) : ℝ := a * x^2 + x + 1

-- Part 1
theorem part1 (a : ℝ) (h1 : a ≠ 0) :
  (∀ x, f a x > 0 ↔ -1/3 < x ∧ x < 1/2) → a = 1/5 := by sorry

-- Part 2
theorem part2 (a : ℝ) (h2 : a ∈ Set.Icc (-2 : ℝ) 0) :
  {x : ℝ | f a x > 0} = {x : ℝ | -1/2 < x ∧ x < 1} := by sorry

-- Part 3
theorem part3 (a : ℝ) :
  (∀ x ∈ Set.Icc 0 2, f a x > 0) → a > -3/4 ∧ a ≠ 0 := by sorry

end NUMINAMATH_CALUDE_part1_part2_part3_l644_64454


namespace NUMINAMATH_CALUDE_power_simplification_l644_64415

theorem power_simplification (m : ℕ) : 
  m = 8^126 → (m * 16) / 64 = 16^94 := by
  sorry

end NUMINAMATH_CALUDE_power_simplification_l644_64415


namespace NUMINAMATH_CALUDE_jason_earnings_l644_64472

/-- Calculates Jason's total earnings for the week given his work hours and rates --/
theorem jason_earnings (after_school_rate : ℝ) (saturday_rate : ℝ) (total_hours : ℝ) (saturday_hours : ℝ) :
  after_school_rate = 4 ∧ 
  saturday_rate = 6 ∧ 
  total_hours = 18 ∧ 
  saturday_hours = 8 →
  (total_hours - saturday_hours) * after_school_rate + saturday_hours * saturday_rate = 88 := by
  sorry

end NUMINAMATH_CALUDE_jason_earnings_l644_64472


namespace NUMINAMATH_CALUDE_spinner_final_direction_l644_64430

-- Define the four cardinal directions
inductive Direction
  | North
  | East
  | South
  | West

-- Define a function to rotate a direction
def rotate (d : Direction) (revolutions : ℚ) : Direction :=
  match (revolutions % 1).num.mod 4 with
  | 0 => d
  | 1 => match d with
         | Direction.North => Direction.East
         | Direction.East => Direction.South
         | Direction.South => Direction.West
         | Direction.West => Direction.North
  | 2 => match d with
         | Direction.North => Direction.South
         | Direction.East => Direction.West
         | Direction.South => Direction.North
         | Direction.West => Direction.East
  | 3 => match d with
         | Direction.North => Direction.West
         | Direction.East => Direction.North
         | Direction.South => Direction.East
         | Direction.West => Direction.South
  | _ => d  -- This case should never occur due to mod 4

-- Theorem statement
theorem spinner_final_direction :
  let initial_direction := Direction.North
  let clockwise_move := (7 : ℚ) / 2
  let counterclockwise_move := (17 : ℚ) / 4
  let final_direction := rotate initial_direction (clockwise_move - counterclockwise_move)
  final_direction = Direction.East := by
  sorry


end NUMINAMATH_CALUDE_spinner_final_direction_l644_64430


namespace NUMINAMATH_CALUDE_simplify_expression_l644_64453

theorem simplify_expression (x y z : ℝ) (hx : x = 3) (hy : y = 2) (hz : z = 4) :
  (12 * x^2 * y^3 * z) / (4 * x * y * z^2) = 9 := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l644_64453


namespace NUMINAMATH_CALUDE_gizmos_produced_l644_64438

/-- Represents the production scenario in a factory -/
structure ProductionScenario where
  a : ℝ  -- Time to produce a gadget
  b : ℝ  -- Time to produce a gizmo

/-- Checks if the production scenario satisfies the given conditions -/
def satisfies_conditions (s : ProductionScenario) : Prop :=
  s.a ≥ 0 ∧ s.b ≥ 0 ∧  -- Non-negative production times
  450 * s.a + 300 * s.b = 150 ∧  -- 150 workers in 1 hour
  360 * s.a + 450 * s.b = 180 ∧  -- 90 workers in 2 hours
  300 * s.a = 300  -- 75 workers produce 300 gadgets in 4 hours

/-- Theorem stating the number of gizmos produced by 75 workers in 4 hours -/
theorem gizmos_produced (s : ProductionScenario) 
  (h : satisfies_conditions s) : 
  75 * 4 / s.b = 150 := by
  sorry


end NUMINAMATH_CALUDE_gizmos_produced_l644_64438


namespace NUMINAMATH_CALUDE_student_count_l644_64462

theorem student_count (cost_per_student : ℕ) (total_cost : ℕ) (h1 : cost_per_student = 8) (h2 : total_cost = 184) :
  total_cost / cost_per_student = 23 :=
by
  sorry

end NUMINAMATH_CALUDE_student_count_l644_64462


namespace NUMINAMATH_CALUDE_symmetric_points_sum_l644_64417

/-- Two points are symmetric about the x-axis if their x-coordinates are the same
    and their y-coordinates are opposite numbers -/
def symmetric_about_x_axis (p q : ℝ × ℝ) : Prop :=
  p.1 = q.1 ∧ p.2 = -q.2

theorem symmetric_points_sum (b a : ℝ) :
  symmetric_about_x_axis (-2, b) (a, -3) → a + b = 1 := by
  sorry

end NUMINAMATH_CALUDE_symmetric_points_sum_l644_64417


namespace NUMINAMATH_CALUDE_initial_pencils_equals_sum_l644_64450

/-- The number of pencils Ken had initially -/
def initial_pencils : ℕ := sorry

/-- The number of pencils Ken gave to Manny -/
def pencils_to_manny : ℕ := 10

/-- The number of pencils Ken gave to Nilo -/
def pencils_to_nilo : ℕ := pencils_to_manny + 10

/-- The number of pencils Ken kept for himself -/
def pencils_kept : ℕ := 20

/-- Theorem stating that the initial number of pencils is equal to the sum of
    pencils given to Manny, Nilo, and kept by Ken -/
theorem initial_pencils_equals_sum :
  initial_pencils = pencils_to_manny + pencils_to_nilo + pencils_kept :=
by sorry

end NUMINAMATH_CALUDE_initial_pencils_equals_sum_l644_64450


namespace NUMINAMATH_CALUDE_shaded_area_regular_octagon_l644_64421

/-- The area of the shaded region in a regular octagon --/
theorem shaded_area_regular_octagon (s : ℝ) (h : s = 8) :
  let R := s / (2 * Real.sin (π / 8))
  let shaded_area := (R / 2) ^ 2
  shaded_area = 32 + 16 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_shaded_area_regular_octagon_l644_64421


namespace NUMINAMATH_CALUDE_product_modulo_l644_64465

theorem product_modulo : (1582 * 2031) % 600 = 42 := by
  sorry

end NUMINAMATH_CALUDE_product_modulo_l644_64465


namespace NUMINAMATH_CALUDE_clock_hand_speed_ratio_l644_64424

/-- Represents the number of degrees in a full rotation of a clock face. -/
def clock_degrees : ℕ := 360

/-- Represents the number of minutes it takes for the minute hand to complete a full rotation. -/
def minute_hand_period : ℕ := 60

/-- Represents the number of hours it takes for the hour hand to complete a full rotation. -/
def hour_hand_period : ℕ := 12

/-- Theorem stating that the ratio of the speeds of the hour hand to the minute hand is 2:24. -/
theorem clock_hand_speed_ratio :
  (clock_degrees / (hour_hand_period * minute_hand_period) : ℚ) / 
  (clock_degrees / minute_hand_period : ℚ) = 2 / 24 := by
  sorry

end NUMINAMATH_CALUDE_clock_hand_speed_ratio_l644_64424


namespace NUMINAMATH_CALUDE_cooperation_is_best_l644_64484

/-- Represents a factory with its daily processing capacity and fee -/
structure Factory where
  capacity : ℕ
  fee : ℕ

/-- Represents a processing plan with its duration and total cost -/
structure Plan where
  duration : ℕ
  cost : ℕ

/-- Calculates the plan for a single factory -/
def single_factory_plan (f : Factory) (total_products : ℕ) (engineer_fee : ℕ) : Plan :=
  let duration := total_products / f.capacity
  { duration := duration
  , cost := duration * (f.fee + engineer_fee) }

/-- Calculates the plan for two factories cooperating -/
def cooperation_plan (f1 f2 : Factory) (total_products : ℕ) (engineer_fee : ℕ) : Plan :=
  let duration := total_products / (f1.capacity + f2.capacity)
  { duration := duration
  , cost := duration * (f1.fee + f2.fee + engineer_fee) }

/-- Checks if one plan is better than another -/
def is_better_plan (p1 p2 : Plan) : Prop :=
  p1.duration < p2.duration ∧ p1.cost < p2.cost

theorem cooperation_is_best (total_products engineer_fee : ℕ) :
  let factory_a : Factory := { capacity := 16, fee := 80 }
  let factory_b : Factory := { capacity := 24, fee := 120 }
  let plan_a := single_factory_plan factory_a total_products engineer_fee
  let plan_b := single_factory_plan factory_b total_products engineer_fee
  let plan_coop := cooperation_plan factory_a factory_b total_products engineer_fee
  total_products = 960 ∧
  engineer_fee = 10 ∧
  factory_a.capacity * 3 = factory_b.capacity * 2 ∧
  factory_a.capacity + factory_b.capacity = 40 →
  is_better_plan plan_coop plan_a ∧ is_better_plan plan_coop plan_b :=
by sorry

end NUMINAMATH_CALUDE_cooperation_is_best_l644_64484


namespace NUMINAMATH_CALUDE_stating_solution_count_56_l644_64409

/-- 
Given a positive integer n, count_solutions n returns the number of solutions 
to the equation xy + z = n where x, y, and z are positive integers.
-/
def count_solutions (n : ℕ+) : ℕ := sorry

/-- 
Theorem stating that if count_solutions n = 56, then n = 34 or n = 35
-/
theorem solution_count_56 (n : ℕ+) : 
  count_solutions n = 56 → n = 34 ∨ n = 35 := by sorry

end NUMINAMATH_CALUDE_stating_solution_count_56_l644_64409


namespace NUMINAMATH_CALUDE_inequality_solution_set_l644_64447

theorem inequality_solution_set (x : ℝ) : -2 * x + 3 < 0 ↔ x > 3/2 := by
  sorry

end NUMINAMATH_CALUDE_inequality_solution_set_l644_64447


namespace NUMINAMATH_CALUDE_y_exceeds_x_by_100_percent_l644_64478

theorem y_exceeds_x_by_100_percent (x y : ℝ) (h : x = 0.5 * y) : 
  (y - x) / x = 1 := by sorry

end NUMINAMATH_CALUDE_y_exceeds_x_by_100_percent_l644_64478


namespace NUMINAMATH_CALUDE_arrangement_of_cards_l644_64491

def number_of_arrangements (total_cards : ℕ) (interchangeable_cards : ℕ) : ℕ :=
  (total_cards.factorial) / (interchangeable_cards.factorial)

theorem arrangement_of_cards : number_of_arrangements 15 13 = 210 := by
  sorry

end NUMINAMATH_CALUDE_arrangement_of_cards_l644_64491


namespace NUMINAMATH_CALUDE_difference_of_squares_l644_64496

theorem difference_of_squares (x y : ℝ) : x^2 - y^2 = (x + y) * (x - y) := by
  sorry

end NUMINAMATH_CALUDE_difference_of_squares_l644_64496


namespace NUMINAMATH_CALUDE_olaf_water_requirement_l644_64429

/-- Calculates the total water needed for a sailing trip -/
def water_needed_for_trip (crew_size : ℕ) (water_per_man_per_day : ℚ) 
  (boat_speed : ℕ) (total_distance : ℕ) : ℚ :=
  let trip_duration := total_distance / boat_speed
  let daily_water_requirement := crew_size * water_per_man_per_day
  daily_water_requirement * trip_duration

/-- Theorem: The total water needed for Olaf's sailing trip is 250 gallons -/
theorem olaf_water_requirement : 
  water_needed_for_trip 25 (1/2) 200 4000 = 250 := by
  sorry

end NUMINAMATH_CALUDE_olaf_water_requirement_l644_64429


namespace NUMINAMATH_CALUDE_max_sum_of_products_l644_64431

theorem max_sum_of_products (f g h j : ℕ) : 
  f ∈ ({4, 5, 9, 10} : Set ℕ) →
  g ∈ ({4, 5, 9, 10} : Set ℕ) →
  h ∈ ({4, 5, 9, 10} : Set ℕ) →
  j ∈ ({4, 5, 9, 10} : Set ℕ) →
  f ≠ g ∧ f ≠ h ∧ f ≠ j ∧ g ≠ h ∧ g ≠ j ∧ h ≠ j →
  f < g →
  f * g + g * h + h * j + f * j ≤ 196 :=
by sorry

end NUMINAMATH_CALUDE_max_sum_of_products_l644_64431


namespace NUMINAMATH_CALUDE_sum_of_digits_2023_base7_l644_64422

/-- Converts a natural number from base 10 to base 7 -/
def toBase7 (n : ℕ) : List ℕ :=
  sorry

/-- Sums the digits in a list -/
def sumDigits (digits : List ℕ) : ℕ :=
  sorry

theorem sum_of_digits_2023_base7 :
  sumDigits (toBase7 2023) = 13 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_digits_2023_base7_l644_64422


namespace NUMINAMATH_CALUDE_nancy_bills_denomination_l644_64441

/-- Given Nancy has 9 bills of equal denomination and a total of 45 dollars, 
    the denomination of each bill is $5. -/
theorem nancy_bills_denomination (num_bills : ℕ) (total_amount : ℕ) (denomination : ℕ) :
  num_bills = 9 →
  total_amount = 45 →
  num_bills * denomination = total_amount →
  denomination = 5 := by
sorry

end NUMINAMATH_CALUDE_nancy_bills_denomination_l644_64441


namespace NUMINAMATH_CALUDE_four_boys_three_girls_144_arrangements_l644_64416

/-- The number of ways to arrange alternating boys and girls in a row -/
def alternatingArrangements (boys girls : ℕ) : ℕ := boys.factorial * girls.factorial

/-- Theorem stating that if there are 3 girls and 144 alternating arrangements, there must be 4 boys -/
theorem four_boys_three_girls_144_arrangements :
  ∃ (boys : ℕ), boys > 0 ∧ alternatingArrangements boys 3 = 144 → boys = 4 := by
  sorry

#check four_boys_three_girls_144_arrangements

end NUMINAMATH_CALUDE_four_boys_three_girls_144_arrangements_l644_64416


namespace NUMINAMATH_CALUDE_stream_speed_l644_64411

/-- The speed of a stream given upstream and downstream canoe speeds -/
theorem stream_speed (upstream_speed downstream_speed : ℝ) 
  (h1 : upstream_speed = 4)
  (h2 : downstream_speed = 12) :
  (downstream_speed - upstream_speed) / 2 = 4 := by
  sorry

end NUMINAMATH_CALUDE_stream_speed_l644_64411


namespace NUMINAMATH_CALUDE_modular_inverse_87_mod_88_l644_64497

theorem modular_inverse_87_mod_88 : ∃ x : ℤ, 0 ≤ x ∧ x < 88 ∧ (87 * x) % 88 = 1 :=
by
  use 87
  sorry

end NUMINAMATH_CALUDE_modular_inverse_87_mod_88_l644_64497


namespace NUMINAMATH_CALUDE_quadratic_no_real_roots_l644_64423

theorem quadratic_no_real_roots : 
  {x : ℝ | x^2 + x + 1 = 0} = ∅ := by sorry

end NUMINAMATH_CALUDE_quadratic_no_real_roots_l644_64423


namespace NUMINAMATH_CALUDE_arithmetic_sequence_property_l644_64481

/-- For an arithmetic sequence with general term a_n = 3n - 4, 
    the difference between the first term and the common difference is -4. -/
theorem arithmetic_sequence_property : 
  ∀ (a : ℕ → ℤ), 
  (∀ n, a n = 3*n - 4) → 
  (a 1 - (a 2 - a 1) = -4) :=
by sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_property_l644_64481


namespace NUMINAMATH_CALUDE_award_distribution_l644_64469

theorem award_distribution (n : ℕ) (k : ℕ) :
  n = 6 ∧ k = 3 →
  (Finset.univ.powerset.filter (λ s : Finset (Fin n) => s.card = 2)).card.choose k = 15 :=
by sorry

end NUMINAMATH_CALUDE_award_distribution_l644_64469


namespace NUMINAMATH_CALUDE_english_test_average_l644_64443

theorem english_test_average (avg_two_months : ℝ) (third_month_score : ℝ) :
  avg_two_months = 86 →
  third_month_score = 98 →
  (2 * avg_two_months + third_month_score) / 3 = 90 := by
  sorry

end NUMINAMATH_CALUDE_english_test_average_l644_64443


namespace NUMINAMATH_CALUDE_correct_mean_calculation_l644_64444

theorem correct_mean_calculation (n : ℕ) (original_mean : ℚ) (incorrect_value correct_value : ℚ) :
  n = 20 ∧ original_mean = 150 ∧ incorrect_value = 135 ∧ correct_value = 160 →
  (n * original_mean - incorrect_value + correct_value) / n = 151.25 := by
sorry

end NUMINAMATH_CALUDE_correct_mean_calculation_l644_64444


namespace NUMINAMATH_CALUDE_basketball_success_rate_increase_success_rate_increase_approx_17_l644_64456

/-- Calculates the increase in success rate percentage for basketball free throws -/
theorem basketball_success_rate_increase 
  (initial_success : Nat) 
  (initial_attempts : Nat) 
  (subsequent_success_rate : Rat) 
  (subsequent_attempts : Nat) : ℝ :=
  let total_success := initial_success + ⌊subsequent_success_rate * subsequent_attempts⌋
  let total_attempts := initial_attempts + subsequent_attempts
  let new_rate := (total_success : ℝ) / total_attempts
  let initial_rate := (initial_success : ℝ) / initial_attempts
  let increase := (new_rate - initial_rate) * 100
  ⌊increase + 0.5⌋

/-- The increase in success rate percentage is approximately 17 percentage points -/
theorem success_rate_increase_approx_17 :
  ⌊basketball_success_rate_increase 7 15 (3/4) 18 + 0.5⌋ = 17 := by
  sorry

end NUMINAMATH_CALUDE_basketball_success_rate_increase_success_rate_increase_approx_17_l644_64456


namespace NUMINAMATH_CALUDE_orange_purchase_total_l644_64428

/-- The total quantity of oranges bought over three weeks -/
def totalOranges (initialPurchase additionalPurchase : ℕ) : ℕ :=
  let week1Total := initialPurchase + additionalPurchase
  let weeklyPurchaseAfter := 2 * week1Total
  week1Total + weeklyPurchaseAfter + weeklyPurchaseAfter

/-- Proof that the total quantity of oranges bought after three weeks is 75 kgs -/
theorem orange_purchase_total :
  totalOranges 10 5 = 75 := by
  sorry


end NUMINAMATH_CALUDE_orange_purchase_total_l644_64428


namespace NUMINAMATH_CALUDE_phil_cards_l644_64459

/-- Calculates the number of baseball cards remaining after buying for a year and losing half. -/
def remaining_cards (cards_per_week : ℕ) (weeks_per_year : ℕ) : ℕ :=
  (cards_per_week * weeks_per_year) / 2

/-- Theorem stating that buying 20 cards each week for 52 weeks and losing half results in 520 cards. -/
theorem phil_cards : remaining_cards 20 52 = 520 := by
  sorry

end NUMINAMATH_CALUDE_phil_cards_l644_64459


namespace NUMINAMATH_CALUDE_sequence_sum_values_l644_64499

def is_valid_sequence (a b : ℕ → ℕ) : Prop :=
  (∀ n, a (n + 1) > a n) ∧ 
  (∀ n, b (n + 1) > b n) ∧
  (a 10 = b 10) ∧ 
  (a 10 < 2017) ∧
  (∀ n, a (n + 2) = a (n + 1) + a n) ∧
  (∀ n, b (n + 1) = 2 * b n)

theorem sequence_sum_values (a b : ℕ → ℕ) :
  is_valid_sequence a b → (a 1 + b 1 = 13 ∨ a 1 + b 1 = 20) :=
by
  sorry

end NUMINAMATH_CALUDE_sequence_sum_values_l644_64499


namespace NUMINAMATH_CALUDE_smallest_sum_reciprocals_l644_64479

theorem smallest_sum_reciprocals (x y : ℕ+) (h1 : x ≠ y) (h2 : (1 : ℚ) / x + (1 : ℚ) / y = (1 : ℚ) / 15) :
  ∃ (a b : ℕ+), a ≠ b ∧ (1 : ℚ) / a + (1 : ℚ) / b = (1 : ℚ) / 15 ∧ (a : ℕ) + (b : ℕ) = 64 ∧
  ∀ (c d : ℕ+), c ≠ d → (1 : ℚ) / c + (1 : ℚ) / d = (1 : ℚ) / 15 → (c : ℕ) + (d : ℕ) ≥ 64 :=
by sorry

end NUMINAMATH_CALUDE_smallest_sum_reciprocals_l644_64479


namespace NUMINAMATH_CALUDE_burger_cost_is_12_l644_64473

/-- The cost of each burger Owen bought in June -/
def burger_cost (burgers_per_day : ℕ) (total_spent : ℕ) (days_in_june : ℕ) : ℚ :=
  total_spent / (burgers_per_day * days_in_june)

/-- Theorem stating that each burger costs 12 dollars -/
theorem burger_cost_is_12 :
  burger_cost 2 720 30 = 12 := by
  sorry

end NUMINAMATH_CALUDE_burger_cost_is_12_l644_64473


namespace NUMINAMATH_CALUDE_value_range_of_f_l644_64457

def f (x : Int) : Int := x + 1

theorem value_range_of_f :
  {y | ∃ x ∈ ({-1, 1} : Set Int), f x = y} = {0, 2} := by sorry

end NUMINAMATH_CALUDE_value_range_of_f_l644_64457


namespace NUMINAMATH_CALUDE_sin_cos_sum_27_18_l644_64426

theorem sin_cos_sum_27_18 :
  Real.sin (27 * π / 180) * Real.cos (18 * π / 180) +
  Real.cos (27 * π / 180) * Real.sin (18 * π / 180) =
  Real.sqrt 2 / 2 :=
by sorry

end NUMINAMATH_CALUDE_sin_cos_sum_27_18_l644_64426


namespace NUMINAMATH_CALUDE_centipede_sock_shoe_arrangements_l644_64470

def num_legs : ℕ := 10

def total_items : ℕ := 2 * num_legs

def valid_arrangements : ℕ := Nat.factorial total_items / (2^num_legs)

theorem centipede_sock_shoe_arrangements :
  valid_arrangements = Nat.factorial total_items / (2^num_legs) :=
by sorry

end NUMINAMATH_CALUDE_centipede_sock_shoe_arrangements_l644_64470


namespace NUMINAMATH_CALUDE_right_triangle_trig_l644_64414

/-- Given a right triangle XYZ with hypotenuse XY = 13 and YZ = 5, 
    prove that tan X = 5/12 and cos X = 12/13 -/
theorem right_triangle_trig (X Y Z : ℝ) (h1 : X^2 + Y^2 = Z^2) 
  (h2 : Z = 13) (h3 : Y = 5) : 
  Real.tan X = 5/12 ∧ Real.cos X = 12/13 := by
  sorry


end NUMINAMATH_CALUDE_right_triangle_trig_l644_64414


namespace NUMINAMATH_CALUDE_broken_line_length_lower_bound_l644_64480

/-- A broken line in a square -/
structure BrokenLine where
  -- The square containing the broken line
  square : Set (ℝ × ℝ)
  -- The broken line itself
  line : Set (ℝ × ℝ)
  -- The square has side length 50
  square_side : ∀ (x y : ℝ), (x, y) ∈ square → 0 ≤ x ∧ x ≤ 50 ∧ 0 ≤ y ∧ y ≤ 50
  -- The broken line is contained within the square
  line_in_square : line ⊆ square
  -- For any point in the square, there's a point on the line within distance 1
  close_point : ∀ (p : ℝ × ℝ), p ∈ square → ∃ (q : ℝ × ℝ), q ∈ line ∧ Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2) ≤ 1

/-- The length of a broken line -/
noncomputable def length (bl : BrokenLine) : ℝ := sorry

/-- Theorem: The length of the broken line is greater than 1248 -/
theorem broken_line_length_lower_bound (bl : BrokenLine) : length bl > 1248 := by
  sorry

end NUMINAMATH_CALUDE_broken_line_length_lower_bound_l644_64480


namespace NUMINAMATH_CALUDE_nail_fraction_sum_l644_64408

theorem nail_fraction_sum : 
  let size_2d : ℚ := 1/6
  let size_3d : ℚ := 2/15
  let size_4d : ℚ := 3/20
  let size_5d : ℚ := 1/10
  let size_6d : ℚ := 1/4
  let size_7d : ℚ := 1/12
  let size_8d : ℚ := 1/8
  let size_9d : ℚ := 1/30
  size_2d + size_3d + size_5d + size_8d = 21/40 := by
  sorry

end NUMINAMATH_CALUDE_nail_fraction_sum_l644_64408


namespace NUMINAMATH_CALUDE_age_ratio_equation_exists_l644_64437

/-- Represents the ages of three people in terms of a common multiplier -/
structure AgeRatio :=
  (x : ℝ)  -- Common multiplier
  (y : ℝ)  -- Number of years ago

/-- The equation relating the ages and the sum from y years ago -/
def ageEquation (r : AgeRatio) : Prop :=
  20 * r.x - 3 * r.y = 76

theorem age_ratio_equation_exists :
  ∃ r : AgeRatio, ageEquation r :=
sorry

end NUMINAMATH_CALUDE_age_ratio_equation_exists_l644_64437


namespace NUMINAMATH_CALUDE_inequality_preserved_under_subtraction_l644_64436

theorem inequality_preserved_under_subtraction (a b c : ℝ) : 
  a < b → a - 2*c < b - 2*c := by
  sorry

end NUMINAMATH_CALUDE_inequality_preserved_under_subtraction_l644_64436


namespace NUMINAMATH_CALUDE_optimal_candy_purchase_l644_64442

/-- Represents the number of candies in a purchase strategy -/
structure CandyPurchase where
  singles : ℕ
  packs : ℕ
  bulks : ℕ

/-- Calculates the total cost of a purchase strategy -/
def totalCost (p : CandyPurchase) : ℕ :=
  p.singles + 3 * p.packs + 4 * p.bulks

/-- Calculates the total number of candies in a purchase strategy -/
def totalCandies (p : CandyPurchase) : ℕ :=
  p.singles + 4 * p.packs + 7 * p.bulks

/-- Represents a valid purchase strategy within the $10 budget -/
def ValidPurchase (p : CandyPurchase) : Prop :=
  totalCost p ≤ 10

/-- The maximum number of candies that can be purchased with $10 -/
def maxCandies : ℕ := 16

theorem optimal_candy_purchase :
  ∀ p : CandyPurchase, ValidPurchase p → totalCandies p ≤ maxCandies ∧
  ∃ q : CandyPurchase, ValidPurchase q ∧ totalCandies q = maxCandies :=
by sorry

end NUMINAMATH_CALUDE_optimal_candy_purchase_l644_64442


namespace NUMINAMATH_CALUDE_arithmetic_mean_fractions_l644_64432

theorem arithmetic_mean_fractions (b c x : ℝ) (hbc : b ≠ c) (hx : x ≠ 0) :
  ((x + b) / x + (x - c) / x) / 2 = 1 + (b - c) / (2 * x) := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_mean_fractions_l644_64432


namespace NUMINAMATH_CALUDE_jakes_snake_length_l644_64401

/-- Given two snakes where one is 12 inches longer than the other,
    and their combined length is 70 inches, prove that the longer snake is 41 inches long. -/
theorem jakes_snake_length (penny_snake : ℕ) (jake_snake : ℕ)
  (h1 : jake_snake = penny_snake + 12)
  (h2 : penny_snake + jake_snake = 70) :
  jake_snake = 41 :=
by sorry

end NUMINAMATH_CALUDE_jakes_snake_length_l644_64401


namespace NUMINAMATH_CALUDE_ladder_angle_elevation_l644_64433

def ladder_foot_distance : ℝ := 4.6
def ladder_length : ℝ := 9.2

theorem ladder_angle_elevation :
  let cos_angle := ladder_foot_distance / ladder_length
  let angle := Real.arccos cos_angle
  ∃ ε > 0, abs (angle - Real.pi / 3) < ε :=
by
  sorry

end NUMINAMATH_CALUDE_ladder_angle_elevation_l644_64433


namespace NUMINAMATH_CALUDE_linear_equation_solution_l644_64488

theorem linear_equation_solution (a b : ℝ) (h1 : a - b = 0) (h2 : a ≠ 0) :
  ∃! x : ℝ, a * x + b = 0 ∧ x = -1 := by
  sorry

end NUMINAMATH_CALUDE_linear_equation_solution_l644_64488


namespace NUMINAMATH_CALUDE_actual_distance_is_82_l644_64403

/-- Represents the distance between two towns on a map --/
def map_distance : ℝ := 9

/-- Represents the initial scale of the map --/
def initial_scale : ℝ := 10

/-- Represents the subsequent scale of the map --/
def subsequent_scale : ℝ := 8

/-- Represents the distance on the map where the initial scale applies --/
def initial_scale_distance : ℝ := 5

/-- Calculates the actual distance between two towns given the map distance and scales --/
def actual_distance : ℝ :=
  initial_scale * initial_scale_distance +
  subsequent_scale * (map_distance - initial_scale_distance)

/-- Theorem stating that the actual distance between the towns is 82 miles --/
theorem actual_distance_is_82 : actual_distance = 82 := by
  sorry

end NUMINAMATH_CALUDE_actual_distance_is_82_l644_64403


namespace NUMINAMATH_CALUDE_rectangle_square_length_difference_l644_64468

/-- Given a square and a rectangle with specific perimeter and width relationships,
    prove that the length of the rectangle is 4 centimeters longer than the side of the square. -/
theorem rectangle_square_length_difference
  (s : ℝ) -- side length of the square
  (l w : ℝ) -- length and width of the rectangle
  (h1 : 2 * (l + w) = 4 * s + 4) -- perimeter relationship
  (h2 : w = s - 2) -- width relationship
  : l = s + 4 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_square_length_difference_l644_64468


namespace NUMINAMATH_CALUDE_amanda_hourly_rate_l644_64477

/-- Amanda's work scenario --/
structure AmandaWork where
  hours_per_day : ℕ
  pay_percentage : ℚ
  reduced_pay : ℚ

/-- Calculate Amanda's hourly rate --/
def hourly_rate (w : AmandaWork) : ℚ :=
  (w.reduced_pay / w.pay_percentage) / w.hours_per_day

/-- Theorem: Amanda's hourly rate is $50 --/
theorem amanda_hourly_rate (w : AmandaWork) 
  (h1 : w.hours_per_day = 10)
  (h2 : w.pay_percentage = 4/5)
  (h3 : w.reduced_pay = 400) :
  hourly_rate w = 50 := by
  sorry

#eval hourly_rate { hours_per_day := 10, pay_percentage := 4/5, reduced_pay := 400 }

end NUMINAMATH_CALUDE_amanda_hourly_rate_l644_64477


namespace NUMINAMATH_CALUDE_sphere_diameter_equal_volume_cone_l644_64413

/-- The diameter of a sphere with the same volume as a cone -/
theorem sphere_diameter_equal_volume_cone (r h : ℝ) (hr : r = 2) (hh : h = 8) :
  let cone_volume := (1/3) * Real.pi * r^2 * h
  let sphere_radius := (cone_volume * 3 / (4 * Real.pi))^(1/3)
  2 * sphere_radius = 4 := by sorry

end NUMINAMATH_CALUDE_sphere_diameter_equal_volume_cone_l644_64413


namespace NUMINAMATH_CALUDE_carrots_planted_per_hour_l644_64418

theorem carrots_planted_per_hour 
  (rows : ℕ) 
  (plants_per_row : ℕ) 
  (total_hours : ℕ) 
  (h1 : rows = 400) 
  (h2 : plants_per_row = 300) 
  (h3 : total_hours = 20) : 
  (rows * plants_per_row) / total_hours = 6000 := by
sorry

end NUMINAMATH_CALUDE_carrots_planted_per_hour_l644_64418


namespace NUMINAMATH_CALUDE_divisor_prime_ratio_l644_64425

def d (n : ℕ) : ℕ := (Nat.divisors n).card

theorem divisor_prime_ratio (n : ℕ) (p : ℕ) (h_prime : Nat.Prime p) :
  n / d n = p ↔ 
    n = 8 ∨ n = 9 ∨ n = 12 ∨ n = 18 ∨ n = 24 ∨
    (∃ q : ℕ, Nat.Prime q ∧ q > 3 ∧ (n = 8 * q ∨ n = 12 * q)) :=
by sorry

end NUMINAMATH_CALUDE_divisor_prime_ratio_l644_64425


namespace NUMINAMATH_CALUDE_original_bet_is_40_l644_64405

/-- Represents the payout ratio for a blackjack -/
def blackjack_ratio : ℚ := 3 / 2

/-- Represents the payout received by the player -/
def payout : ℚ := 60

/-- Calculates the original bet given the payout and the blackjack ratio -/
def original_bet (payout : ℚ) (ratio : ℚ) : ℚ := payout / ratio

/-- Proves that the original bet was $40 given the conditions -/
theorem original_bet_is_40 : 
  original_bet payout blackjack_ratio = 40 := by
  sorry

#eval original_bet payout blackjack_ratio

end NUMINAMATH_CALUDE_original_bet_is_40_l644_64405


namespace NUMINAMATH_CALUDE_bob_picked_450_apples_l644_64482

/-- The number of apples Bob picked for his family -/
def apples_picked (num_children : ℕ) (apples_per_child : ℕ) (num_adults : ℕ) (apples_per_adult : ℕ) : ℕ :=
  num_children * apples_per_child + num_adults * apples_per_adult

/-- Theorem stating that Bob picked 450 apples for his family -/
theorem bob_picked_450_apples : 
  apples_picked 33 10 40 3 = 450 := by
  sorry

end NUMINAMATH_CALUDE_bob_picked_450_apples_l644_64482


namespace NUMINAMATH_CALUDE_ending_number_is_67_l644_64476

-- Define the sum of first n odd integers
def sum_odd_integers (n : ℕ) : ℕ := n^2

-- Define the sum of odd integers from a to b inclusive
def sum_odd_range (a b : ℕ) : ℕ :=
  sum_odd_integers ((b - a) / 2 + 1) - sum_odd_integers ((a - 1) / 2)

-- The main theorem
theorem ending_number_is_67 :
  ∃ x : ℕ, x ≥ 11 ∧ sum_odd_range 11 x = 416 ∧ x = 67 :=
sorry

end NUMINAMATH_CALUDE_ending_number_is_67_l644_64476


namespace NUMINAMATH_CALUDE_angle_c_measure_l644_64404

theorem angle_c_measure (A B C : ℝ) (h1 : A + B = 80) (h2 : A + B + C = 180) : C = 100 := by
  sorry

end NUMINAMATH_CALUDE_angle_c_measure_l644_64404


namespace NUMINAMATH_CALUDE_bbq_ice_packs_l644_64486

/-- Given a BBQ scenario, calculate the number of 1-pound bags of ice in a pack -/
theorem bbq_ice_packs (people : ℕ) (ice_per_person : ℕ) (pack_price : ℚ) (total_spent : ℚ) :
  people = 15 →
  ice_per_person = 2 →
  pack_price = 3 →
  total_spent = 9 →
  (people * ice_per_person) / (total_spent / pack_price) = 10 := by
  sorry

#check bbq_ice_packs

end NUMINAMATH_CALUDE_bbq_ice_packs_l644_64486


namespace NUMINAMATH_CALUDE_solution_correctness_l644_64460

-- Define the equation
def equation (x : ℝ) : Prop := 2 * (x + 3) = 5 * x

-- Define the solution steps
def step1 (x : ℝ) : Prop := 2 * x + 6 = 5 * x
def step2 (x : ℝ) : Prop := 2 * x - 5 * x = -6
def step3 (x : ℝ) : Prop := -3 * x = -6
def step4 : ℝ := 2

-- Theorem stating the correctness of the solution and that step3 is not based on associative property
theorem solution_correctness :
  ∀ x : ℝ,
  equation x →
  step1 x ∧
  step2 x ∧
  step3 x ∧
  step4 = x ∧
  ¬(∃ a b c : ℝ, step3 x ↔ (a + b) + c = a + (b + c)) :=
by sorry

end NUMINAMATH_CALUDE_solution_correctness_l644_64460


namespace NUMINAMATH_CALUDE_kyle_paper_delivery_l644_64464

/-- The number of papers Kyle delivers in a week -/
def weekly_papers (weekday_houses : ℕ) (sunday_skip : ℕ) (sunday_extra : ℕ) : ℕ :=
  (weekday_houses * 6) + (weekday_houses - sunday_skip + sunday_extra)

/-- Theorem stating the total number of papers Kyle delivers in a week -/
theorem kyle_paper_delivery :
  weekly_papers 100 10 30 = 720 := by
  sorry

end NUMINAMATH_CALUDE_kyle_paper_delivery_l644_64464


namespace NUMINAMATH_CALUDE_foci_of_hyperbola_l644_64406

/-- The hyperbola equation -/
def hyperbola_equation (x y : ℝ) : Prop := 4 * y^2 - 25 * x^2 = 100

/-- The foci coordinates -/
def foci_coordinates : Set (ℝ × ℝ) := {(0, -Real.sqrt 29), (0, Real.sqrt 29)}

/-- Theorem: The given coordinates are the foci of the hyperbola -/
theorem foci_of_hyperbola :
  ∀ (x y : ℝ), hyperbola_equation x y → (x, y) ∈ foci_coordinates :=
sorry

end NUMINAMATH_CALUDE_foci_of_hyperbola_l644_64406


namespace NUMINAMATH_CALUDE_prob_2_to_4_value_l644_64449

/-- The probability distribution of a random variable ξ -/
def P (k : ℕ) : ℚ := 1 / 2^k

/-- The probability that 2 < ξ ≤ 4 -/
def prob_2_to_4 : ℚ := P 3 + P 4

theorem prob_2_to_4_value : prob_2_to_4 = 3/16 := by sorry

end NUMINAMATH_CALUDE_prob_2_to_4_value_l644_64449


namespace NUMINAMATH_CALUDE_arithmetic_sequences_common_terms_l644_64475

/-- The first arithmetic sequence -/
def seq1 (n : ℕ) : ℕ := 2 + 3 * n

/-- The second arithmetic sequence -/
def seq2 (n : ℕ) : ℕ := 4 + 5 * n

/-- The last term of the first sequence -/
def last1 : ℕ := 2015

/-- The last term of the second sequence -/
def last2 : ℕ := 2014

/-- The number of common terms between the two sequences -/
def commonTerms : ℕ := 134

theorem arithmetic_sequences_common_terms :
  (∃ (s : Finset ℕ), s.card = commonTerms ∧
    (∀ x ∈ s, ∃ n m : ℕ, seq1 n = x ∧ seq2 m = x ∧
      seq1 n ≤ last1 ∧ seq2 m ≤ last2)) :=
sorry

end NUMINAMATH_CALUDE_arithmetic_sequences_common_terms_l644_64475


namespace NUMINAMATH_CALUDE_integer_fraction_property_l644_64466

theorem integer_fraction_property (x y : ℤ) (h : ∃ k : ℤ, 3 * x + 4 * y = 5 * k) :
  ∃ m : ℤ, 4 * x - 3 * y = 5 * m := by
sorry

end NUMINAMATH_CALUDE_integer_fraction_property_l644_64466


namespace NUMINAMATH_CALUDE_ellipse_equation_l644_64455

/-- An ellipse with given properties -/
structure Ellipse where
  a : ℝ
  b : ℝ
  h0 : a > b
  h1 : b > 0

/-- The right focus of the ellipse -/
def right_focus (e : Ellipse) : ℝ × ℝ := (3, 0)

/-- The midpoint of the line segment AB -/
def midpoint_AB : ℝ × ℝ := (1, -1)

/-- Theorem: Given an ellipse with the specified properties, its equation is x²/18 + y²/9 = 1 -/
theorem ellipse_equation (e : Ellipse) 
  (h2 : right_focus e = (3, 0))
  (h3 : midpoint_AB = (1, -1)) :
  ∃ (x y : ℝ), x^2 / 18 + y^2 / 9 = 1 :=
sorry

end NUMINAMATH_CALUDE_ellipse_equation_l644_64455


namespace NUMINAMATH_CALUDE_parallelogram_area_l644_64461

def v : Fin 2 → ℝ := ![7, -4]
def w : Fin 2 → ℝ := ![13, -3]

theorem parallelogram_area : 
  abs (Matrix.det !![v 0, v 1; w 0, w 1]) = 31 := by sorry

end NUMINAMATH_CALUDE_parallelogram_area_l644_64461


namespace NUMINAMATH_CALUDE_complex_sum_imaginary_l644_64402

theorem complex_sum_imaginary (a : ℝ) : 
  let z₁ : ℂ := a^2 - 2 - 3*a*Complex.I
  let z₂ : ℂ := a + (a^2 + 2)*Complex.I
  (z₁ + z₂).re = 0 ∧ (z₁ + z₂).im ≠ 0 → a = -2 := by
  sorry

end NUMINAMATH_CALUDE_complex_sum_imaginary_l644_64402


namespace NUMINAMATH_CALUDE_circle_center_l644_64489

/-- Given a circle with equation (x-2)^2 + (y-3)^2 = 1, its center is at (2, 3) -/
theorem circle_center (x y : ℝ) : 
  ((x - 2)^2 + (y - 3)^2 = 1) → (2, 3) = (x, y) := by sorry

end NUMINAMATH_CALUDE_circle_center_l644_64489


namespace NUMINAMATH_CALUDE_four_digit_sum_l644_64412

theorem four_digit_sum (a b c d : ℕ) : 
  a ≠ 0 ∧ b ≠ 0 ∧ c ≠ 0 ∧ d ≠ 0 →
  a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d →
  a < 10 ∧ b < 10 ∧ c < 10 ∧ d < 10 →
  6 * (a + b + c + d) * 1111 = 73326 →
  ({a, b, c, d} : Finset ℕ) = {1, 2, 3, 5} :=
by sorry

end NUMINAMATH_CALUDE_four_digit_sum_l644_64412


namespace NUMINAMATH_CALUDE_jennas_profit_l644_64434

/-- Calculates the profit for Jenna's wholesale business --/
def calculate_profit (
  widget_cost : ℝ)
  (widget_price : ℝ)
  (rent : ℝ)
  (tax_rate : ℝ)
  (worker_salary : ℝ)
  (num_workers : ℕ)
  (widgets_sold : ℕ) : ℝ :=
  let revenue := widget_price * widgets_sold
  let cost_of_goods_sold := widget_cost * widgets_sold
  let gross_profit := revenue - cost_of_goods_sold
  let fixed_costs := rent + (worker_salary * num_workers)
  let profit_before_tax := gross_profit - fixed_costs
  let tax := tax_rate * profit_before_tax
  profit_before_tax - tax

/-- Theorem stating that Jenna's profit is $4000 given the specified conditions --/
theorem jennas_profit :
  calculate_profit 3 8 10000 0.2 2500 4 5000 = 4000 := by
  sorry

end NUMINAMATH_CALUDE_jennas_profit_l644_64434


namespace NUMINAMATH_CALUDE_stratified_sampling_l644_64494

/-- Represents the number of students in each grade and the sample size -/
structure SchoolSample where
  total : ℕ
  first_grade : ℕ
  second_grade : ℕ
  third_grade : ℕ
  sample_first : ℕ

/-- The conditions of the problem -/
def school_conditions (s : SchoolSample) : Prop :=
  s.total = 1290 ∧
  s.first_grade = 480 ∧
  s.second_grade = s.third_grade + 30 ∧
  s.total = s.first_grade + s.second_grade + s.third_grade ∧
  s.sample_first = 96

/-- The theorem to prove -/
theorem stratified_sampling (s : SchoolSample) 
  (h : school_conditions s) : 
  (s.sample_first * s.second_grade) / s.first_grade = 78 := by
  sorry


end NUMINAMATH_CALUDE_stratified_sampling_l644_64494


namespace NUMINAMATH_CALUDE_division_4863_by_97_l644_64463

theorem division_4863_by_97 : ∃ (q r : ℤ), 4863 = 97 * q + r ∧ 0 ≤ r ∧ r < 97 ∧ q = 50 ∧ r = 40 := by
  sorry

end NUMINAMATH_CALUDE_division_4863_by_97_l644_64463


namespace NUMINAMATH_CALUDE_two_students_same_type_l644_64471

-- Define the types of books
inductive BookType
  | History
  | Literature
  | Science

-- Define a type for a pair of books
def BookPair := BookType × BookType

-- Define the set of all possible book pairs
def allBookPairs : Finset BookPair :=
  sorry

-- Define the number of students
def numStudents : Nat := 7

-- Theorem statement
theorem two_students_same_type :
  ∃ (s₁ s₂ : Fin numStudents) (bp : BookPair),
    s₁ ≠ s₂ ∧ 
    (∀ (s : Fin numStudents), ∃ (bp : BookPair), bp ∈ allBookPairs) ∧
    (∃ (f : Fin numStudents → BookPair), f s₁ = bp ∧ f s₂ = bp) :=
  sorry

end NUMINAMATH_CALUDE_two_students_same_type_l644_64471


namespace NUMINAMATH_CALUDE_pet_food_price_l644_64419

/-- Given a manufacturer's suggested retail price and discount conditions, prove the price is $35 -/
theorem pet_food_price (M : ℝ) : 
  (M * (1 - 0.3) * (1 - 0.2) = 19.6) → M = 35 := by
  sorry

end NUMINAMATH_CALUDE_pet_food_price_l644_64419


namespace NUMINAMATH_CALUDE_max_value_theorem_l644_64420

theorem max_value_theorem (a b c : ℝ) 
  (h1 : 0 ≤ a) (h2 : 0 ≤ b) (h3 : 0 ≤ c) (h4 : a + b + c = 2) : 
  (a * b) / (2 * (a + b)) + (a * c) / (2 * (a + c)) + (b * c) / (2 * (b + c)) ≤ 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_max_value_theorem_l644_64420


namespace NUMINAMATH_CALUDE_eight_by_eight_tiling_ten_by_ten_no_tiling_l644_64492

-- Define a chessboard
structure Chessboard :=
  (size : Nat)
  (total_squares : Nat)
  (black_squares : Nat)
  (white_squares : Nat)

-- Define a pedestal shape
structure Pedestal :=
  (squares_covered : Nat)

-- Define the tiling property
def can_tile (b : Chessboard) (p : Pedestal) : Prop :=
  b.total_squares % p.squares_covered = 0

-- Define the color coverage property for 10x10 board
def color_coverage_property (b : Chessboard) (p : Pedestal) : Prop :=
  ∃ (k : Nat), 3 * k + k = b.black_squares ∧ 3 * k + k = b.white_squares

-- Theorem for 8x8 chessboard
theorem eight_by_eight_tiling :
  ∀ (b : Chessboard) (p : Pedestal),
    b.size = 8 →
    b.total_squares = 64 →
    p.squares_covered = 4 →
    can_tile b p :=
sorry

-- Theorem for 10x10 chessboard
theorem ten_by_ten_no_tiling :
  ∀ (b : Chessboard) (p : Pedestal),
    b.size = 10 →
    b.total_squares = 100 →
    b.black_squares = 50 →
    b.white_squares = 50 →
    p.squares_covered = 4 →
    ¬(can_tile b p ∧ color_coverage_property b p) :=
sorry

end NUMINAMATH_CALUDE_eight_by_eight_tiling_ten_by_ten_no_tiling_l644_64492


namespace NUMINAMATH_CALUDE_shaded_region_area_l644_64495

/-- Given a figure composed of 25 congruent squares, where the diagonal of a square 
    formed by 16 of these squares is 10 cm, the total area of the figure is 78.125 square cm. -/
theorem shaded_region_area (num_squares : ℕ) (diagonal : ℝ) (total_area : ℝ) : 
  num_squares = 25 → 
  diagonal = 10 → 
  total_area = 78.125 := by
  sorry

end NUMINAMATH_CALUDE_shaded_region_area_l644_64495


namespace NUMINAMATH_CALUDE_g_max_value_f_upper_bound_l644_64439

noncomputable def f (x : ℝ) := Real.log (x + 1)

noncomputable def g (x : ℝ) := f x - x / 4 - 1

theorem g_max_value :
  ∃ (x : ℝ), ∀ (y : ℝ), g y ≤ g x ∧ g x = 2 * Real.log 2 - 7 / 4 := by sorry

theorem f_upper_bound (x : ℝ) (hx : x > 0) :
  f x < (Real.exp x - 1) / x^2 := by sorry

end NUMINAMATH_CALUDE_g_max_value_f_upper_bound_l644_64439


namespace NUMINAMATH_CALUDE_ghee_mixture_problem_l644_64440

theorem ghee_mixture_problem (x : ℝ) : 
  (0.6 * x = x - 0.4 * x) →  -- 60% is pure ghee, 40% is vanaspati
  (0.4 * x = 0.2 * (x + 10)) →  -- After adding 10 kg, vanaspati becomes 20%
  (x = 10) :=  -- The original quantity was 10 kg
by sorry

end NUMINAMATH_CALUDE_ghee_mixture_problem_l644_64440


namespace NUMINAMATH_CALUDE_common_chord_length_is_sqrt55_div_5_l644_64452

noncomputable section

-- Define the circles
def circle_C1 (x y : ℝ) : Prop := (x - 3)^2 + y^2 = 9

def circle_C2_center : ℝ × ℝ := (Real.sqrt 2, Real.pi / 4)

def circle_C2_radius : ℝ := 1

-- Define the length of the common chord
def common_chord_length : ℝ := Real.sqrt 55 / 5

-- Theorem statement
theorem common_chord_length_is_sqrt55_div_5 :
  ∃ (A B : ℝ × ℝ),
    (circle_C1 A.1 A.2) ∧
    (circle_C1 B.1 B.2) ∧
    ((A.1 - circle_C2_center.1)^2 + (A.2 - circle_C2_center.2)^2 = circle_C2_radius^2) ∧
    ((B.1 - circle_C2_center.1)^2 + (B.2 - circle_C2_center.2)^2 = circle_C2_radius^2) ∧
    Real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2) = common_chord_length :=
by
  sorry

end NUMINAMATH_CALUDE_common_chord_length_is_sqrt55_div_5_l644_64452


namespace NUMINAMATH_CALUDE_probability_of_selecting_girl_l644_64448

-- Define the total number of candidates
def total_candidates : ℕ := 3 + 1

-- Define the number of girls
def number_of_girls : ℕ := 1

-- Theorem statement
theorem probability_of_selecting_girl :
  (number_of_girls : ℚ) / total_candidates = 1 / 4 := by
  sorry

end NUMINAMATH_CALUDE_probability_of_selecting_girl_l644_64448


namespace NUMINAMATH_CALUDE_function_formula_l644_64427

theorem function_formula (f : ℝ → ℝ) (h : ∀ x : ℝ, f (x - 1) = x^2) :
  ∀ x : ℝ, f x = (x + 1)^2 := by
  sorry

end NUMINAMATH_CALUDE_function_formula_l644_64427


namespace NUMINAMATH_CALUDE_complement_intersection_problem_l644_64490

open Set

theorem complement_intersection_problem (I A B : Set ℕ) : 
  I = {0, 1, 2, 3, 4} →
  A = {0, 2, 3} →
  B = {1, 3, 4} →
  (I \ A) ∩ B = {1, 4} := by
  sorry

end NUMINAMATH_CALUDE_complement_intersection_problem_l644_64490


namespace NUMINAMATH_CALUDE_school_section_problem_l644_64400

/-- Calculates the maximum number of equal-sized mixed-gender sections
    that can be formed given the number of boys and girls and the required ratio. -/
def max_sections (boys girls : ℕ) (boy_ratio girl_ratio : ℕ) : ℕ :=
  min (boys / boy_ratio) (girls / girl_ratio)

/-- Theorem stating the solution to the school section problem -/
theorem school_section_problem :
  max_sections 2040 1728 3 2 = 680 := by
  sorry

end NUMINAMATH_CALUDE_school_section_problem_l644_64400

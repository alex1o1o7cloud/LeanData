import Mathlib

namespace NUMINAMATH_CALUDE_hollow_circles_count_l3296_329679

/-- Represents the pattern of circles, where each number is the position of a hollow circle in the repeating sequence -/
def hollow_circle_positions : List Nat := [2, 5, 9]

/-- The length of the repeating sequence -/
def sequence_length : Nat := 9

/-- The total number of circles in the sequence -/
def total_circles : Nat := 2001

/-- Calculates the number of hollow circles in a sequence of given length -/
def count_hollow_circles (n : Nat) : Nat :=
  (n / sequence_length) * hollow_circle_positions.length + 
  (hollow_circle_positions.filter (· ≤ n % sequence_length)).length

theorem hollow_circles_count :
  count_hollow_circles total_circles = 667 := by
  sorry

end NUMINAMATH_CALUDE_hollow_circles_count_l3296_329679


namespace NUMINAMATH_CALUDE_quadratic_roots_property_l3296_329654

theorem quadratic_roots_property (p q : ℝ) : 
  (3 * p^2 - 5 * p - 8 = 0) → 
  (3 * q^2 - 5 * q - 8 = 0) → 
  p ≠ q →
  (3 * p^2 - 3 * q^2) * (p - q)⁻¹ = 5 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_roots_property_l3296_329654


namespace NUMINAMATH_CALUDE_jerry_trays_capacity_l3296_329605

def jerry_trays (trays_table1 trays_table2 num_trips : ℕ) : ℕ :=
  (trays_table1 + trays_table2) / num_trips

theorem jerry_trays_capacity :
  jerry_trays 9 7 2 = 8 := by
  sorry

end NUMINAMATH_CALUDE_jerry_trays_capacity_l3296_329605


namespace NUMINAMATH_CALUDE_rectangular_field_fence_l3296_329670

theorem rectangular_field_fence (area : ℝ) (fence_length : ℝ) (uncovered_side : ℝ) :
  area = 680 →
  fence_length = 146 →
  uncovered_side * (fence_length - uncovered_side) / 2 = area →
  uncovered_side = 136 :=
by
  sorry

end NUMINAMATH_CALUDE_rectangular_field_fence_l3296_329670


namespace NUMINAMATH_CALUDE_division_remainder_problem_l3296_329678

theorem division_remainder_problem (a b : ℕ) (h1 : a - b = 1365) (h2 : a = 1620)
  (h3 : ∃ (q : ℕ), q = 6 ∧ a = q * b + (a % b) ∧ a % b < b) : a % b = 90 := by
  sorry

end NUMINAMATH_CALUDE_division_remainder_problem_l3296_329678


namespace NUMINAMATH_CALUDE_cone_sphere_volume_ratio_l3296_329621

theorem cone_sphere_volume_ratio (r : ℝ) (h : r > 0) :
  let cone_volume := (1 / 3) * π * r^3
  let sphere_volume := (4 / 3) * π * r^3
  cone_volume / sphere_volume = 1 / 4 := by sorry

end NUMINAMATH_CALUDE_cone_sphere_volume_ratio_l3296_329621


namespace NUMINAMATH_CALUDE_father_son_age_ratio_l3296_329649

/-- Proves that given a father who is currently 45 years old, and after 15 years
    will be twice as old as his son, the current ratio of the father's age to
    the son's age is 3:1. -/
theorem father_son_age_ratio :
  ∀ (father_age son_age : ℕ),
    father_age = 45 →
    father_age + 15 = 2 * (son_age + 15) →
    father_age / son_age = 3 :=
by sorry

end NUMINAMATH_CALUDE_father_son_age_ratio_l3296_329649


namespace NUMINAMATH_CALUDE_arithmetic_sequence_problem_l3296_329631

/-- An arithmetic sequence with its sum sequence -/
structure ArithmeticSequence where
  a : ℕ → ℝ  -- The sequence
  S : ℕ → ℝ  -- The sum sequence
  is_arithmetic : ∀ n, a (n + 1) - a n = a (n + 2) - a (n + 1)
  sum_property : ∀ n, S n = (n : ℝ) * (a 1 + a n) / 2

/-- The main theorem -/
theorem arithmetic_sequence_problem (seq : ArithmeticSequence) 
    (h₁ : seq.S 3 = 9) (h₂ : seq.S 6 = 36) : 
  seq.a 6 + seq.a 7 + seq.a 8 = 39 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_problem_l3296_329631


namespace NUMINAMATH_CALUDE_tangent_line_condition_minimum_value_condition_min_value_case1_min_value_case2_min_value_case3_l3296_329650

noncomputable section

-- Define the function f(x)
def f (a : ℝ) (x : ℝ) : ℝ := x * Real.log x + a

-- Theorem for part 1
theorem tangent_line_condition (a : ℝ) :
  (∃ k, ∀ x, f a x = 2 * x + k - 2 * Real.exp 1) → a = Real.exp 1 :=
sorry

-- Theorem for part 2
theorem minimum_value_condition (a m : ℝ) (h : m > 0) :
  let min_value := min (f a (2 * m)) (min (f a (1 / Real.exp 1)) (f a m))
  ∀ x ∈ Set.Icc m (2 * m), f a x ≥ min_value :=
sorry

-- Additional theorems to specify the exact minimum value based on m
theorem min_value_case1 (a m : ℝ) (h1 : m > 0) (h2 : m ≤ 1 / (2 * Real.exp 1)) :
  ∀ x ∈ Set.Icc m (2 * m), f a x ≥ f a (2 * m) :=
sorry

theorem min_value_case2 (a m : ℝ) (h1 : m > 0) (h2 : 1 / (2 * Real.exp 1) < m) (h3 : m < 1 / Real.exp 1) :
  ∀ x ∈ Set.Icc m (2 * m), f a x ≥ f a (1 / Real.exp 1) :=
sorry

theorem min_value_case3 (a m : ℝ) (h1 : m > 0) (h2 : m ≥ 1 / Real.exp 1) :
  ∀ x ∈ Set.Icc m (2 * m), f a x ≥ f a m :=
sorry

end NUMINAMATH_CALUDE_tangent_line_condition_minimum_value_condition_min_value_case1_min_value_case2_min_value_case3_l3296_329650


namespace NUMINAMATH_CALUDE_complex_number_in_second_quadrant_l3296_329638

theorem complex_number_in_second_quadrant : 
  let z : ℂ := 2 * Complex.I * (Complex.I + 1) + 1
  (z.re < 0) ∧ (z.im > 0) := by sorry

end NUMINAMATH_CALUDE_complex_number_in_second_quadrant_l3296_329638


namespace NUMINAMATH_CALUDE_journey_distance_l3296_329658

/-- Proves that a journey with given conditions has a total distance of 224 km -/
theorem journey_distance (total_time : ℝ) (speed1 : ℝ) (speed2 : ℝ) 
  (h1 : total_time = 10)
  (h2 : speed1 = 21)
  (h3 : speed2 = 24) :
  (total_time * speed1 * speed2) / (speed1 + speed2) = 224 := by
  sorry

#check journey_distance

end NUMINAMATH_CALUDE_journey_distance_l3296_329658


namespace NUMINAMATH_CALUDE_grade_ratio_l3296_329695

/-- Proves that the ratio of Bob's grade to Jason's grade is 1:2 -/
theorem grade_ratio (jenny_grade jason_grade bob_grade : ℕ) : 
  jenny_grade = 95 →
  jason_grade = jenny_grade - 25 →
  bob_grade = 35 →
  (bob_grade : ℚ) / jason_grade = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_grade_ratio_l3296_329695


namespace NUMINAMATH_CALUDE_square_sequence_problem_l3296_329672

/-- The number of squares in the nth figure of the sequence -/
def g (n : ℕ) : ℕ :=
  2 * n^2 + 4 * n + 3

theorem square_sequence_problem :
  g 0 = 3 ∧ g 1 = 9 ∧ g 2 = 19 ∧ g 3 = 33 → g 100 = 20403 :=
by
  sorry

end NUMINAMATH_CALUDE_square_sequence_problem_l3296_329672


namespace NUMINAMATH_CALUDE_midpoint_coordinate_sum_l3296_329629

/-- The sum of the coordinates of the midpoint of a segment with endpoints (6, 12) and (0, -6) is 6. -/
theorem midpoint_coordinate_sum : 
  let p1 : ℝ × ℝ := (6, 12)
  let p2 : ℝ × ℝ := (0, -6)
  let midpoint := ((p1.1 + p2.1) / 2, (p1.2 + p2.2) / 2)
  midpoint.1 + midpoint.2 = 6 := by
  sorry

end NUMINAMATH_CALUDE_midpoint_coordinate_sum_l3296_329629


namespace NUMINAMATH_CALUDE_equivalent_expression_l3296_329666

theorem equivalent_expression (x y : ℝ) (h : x * y ≠ 0) :
  ((x^3 + 1) / x) * ((y^3 + 1) / y) - ((x^3 - 1) / y) * ((y^3 - 1) / x) = (2 * x^3 + 2 * y^3) / (x * y) := by
  sorry

end NUMINAMATH_CALUDE_equivalent_expression_l3296_329666


namespace NUMINAMATH_CALUDE_fraction_to_decimal_l3296_329618

theorem fraction_to_decimal : (67 : ℚ) / (2^3 * 5^4) = 0.0134 := by
  sorry

end NUMINAMATH_CALUDE_fraction_to_decimal_l3296_329618


namespace NUMINAMATH_CALUDE_different_grade_selections_l3296_329665

/-- The number of students in the first year -/
def first_year_students : ℕ := 4

/-- The number of students in the second year -/
def second_year_students : ℕ := 5

/-- The number of students in the third year -/
def third_year_students : ℕ := 4

/-- The total number of ways to select 2 students from different grades -/
def total_selections : ℕ := 56

theorem different_grade_selections :
  first_year_students * second_year_students +
  first_year_students * third_year_students +
  second_year_students * third_year_students = total_selections :=
by sorry

end NUMINAMATH_CALUDE_different_grade_selections_l3296_329665


namespace NUMINAMATH_CALUDE_square_of_sum_80_5_l3296_329644

theorem square_of_sum_80_5 : (80 + 5)^2 = 7225 := by
  sorry

end NUMINAMATH_CALUDE_square_of_sum_80_5_l3296_329644


namespace NUMINAMATH_CALUDE_expand_and_simplify_l3296_329639

theorem expand_and_simplify (x : ℝ) (hx : x ≠ 0) :
  (3 / 7) * ((14 / x^3) + 15*x - 6*x^5) = 6 / x^3 + (45*x) / 7 - (18*x^5) / 7 := by
  sorry

end NUMINAMATH_CALUDE_expand_and_simplify_l3296_329639


namespace NUMINAMATH_CALUDE_probability_total_more_than_seven_is_five_twelfths_l3296_329689

/-- The number of possible outcomes when throwing a pair of dice -/
def total_outcomes : ℕ := 36

/-- The number of favorable outcomes (total > 7) when throwing a pair of dice -/
def favorable_outcomes : ℕ := 15

/-- The probability of getting a total more than 7 when throwing a pair of dice -/
def probability_total_more_than_seven : ℚ := favorable_outcomes / total_outcomes

theorem probability_total_more_than_seven_is_five_twelfths :
  probability_total_more_than_seven = 5 / 12 := by sorry

end NUMINAMATH_CALUDE_probability_total_more_than_seven_is_five_twelfths_l3296_329689


namespace NUMINAMATH_CALUDE_abc_inequality_l3296_329602

theorem abc_inequality (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (habc : a * b * c = 1) :
  a / (1 + b) + b / (1 + c) + c / (1 + a) ≥ 3 / 2 := by
  sorry

end NUMINAMATH_CALUDE_abc_inequality_l3296_329602


namespace NUMINAMATH_CALUDE_cookfire_logs_remaining_l3296_329614

/-- Represents the number of logs remaining in a cookfire after a given number of hours -/
def logs_remaining (initial_logs : ℕ) (burn_rate : ℕ) (add_rate : ℕ) (hours : ℕ) : ℤ :=
  initial_logs + hours * (add_rate - burn_rate)

/-- Theorem stating the number of logs remaining after x hours for the given cookfire scenario -/
theorem cookfire_logs_remaining (x : ℕ) :
  logs_remaining 8 4 3 x = 8 - x :=
sorry

end NUMINAMATH_CALUDE_cookfire_logs_remaining_l3296_329614


namespace NUMINAMATH_CALUDE_problem_part1_problem_part2_l3296_329608

-- Part 1
theorem problem_part1 : (-2)^3 + |(-3)| - Real.tan (π/4) = -6 := by sorry

-- Part 2
theorem problem_part2 (a : ℝ) : (a + 2)^2 - a*(a - 4) = 8*a + 4 := by sorry

end NUMINAMATH_CALUDE_problem_part1_problem_part2_l3296_329608


namespace NUMINAMATH_CALUDE_not_concurrent_deduction_l3296_329612

/-- Represents a proof method -/
inductive ProofMethod
  | Synthetic
  | Analytic

/-- Represents the direction of reasoning -/
inductive ReasoningDirection
  | CauseToEffect
  | EffectToCause

/-- Maps a proof method to its reasoning direction -/
def methodDirection (m : ProofMethod) : ReasoningDirection :=
  match m with
  | ProofMethod.Synthetic => ReasoningDirection.CauseToEffect
  | ProofMethod.Analytic => ReasoningDirection.EffectToCause

/-- Theorem stating that synthetic and analytic methods do not concurrently deduce cause and effect -/
theorem not_concurrent_deduction :
  ∀ (m : ProofMethod), methodDirection m ≠ ReasoningDirection.CauseToEffect ∨ 
                       methodDirection m ≠ ReasoningDirection.EffectToCause :=
by
  sorry

end NUMINAMATH_CALUDE_not_concurrent_deduction_l3296_329612


namespace NUMINAMATH_CALUDE_ice_cream_sundaes_l3296_329656

theorem ice_cream_sundaes (n : ℕ) (h : n = 8) : 
  (n : ℕ) + n.choose 2 = 36 :=
by sorry

end NUMINAMATH_CALUDE_ice_cream_sundaes_l3296_329656


namespace NUMINAMATH_CALUDE_coefficient_of_x_cubed_l3296_329607

theorem coefficient_of_x_cubed (x : ℝ) : 
  let expression := 4*(x^3 - 2*x^4) + 3*(x^2 - 3*x^3 + 4*x^6) - (5*x^4 - 2*x^3)
  ∃ (a b c d e : ℝ), expression = -3*x^3 + a*x^2 + b*x^4 + c*x^6 + d*x + e :=
by
  sorry

end NUMINAMATH_CALUDE_coefficient_of_x_cubed_l3296_329607


namespace NUMINAMATH_CALUDE_function_property_l3296_329657

/-- Given a function f(x) = ax^3 - bx^(3/5) + 1, if f(-1) = 3, then f(1) = 1 -/
theorem function_property (a b : ℝ) :
  let f : ℝ → ℝ := λ x ↦ a * x^3 - b * x^(3/5) + 1
  f (-1) = 3 → f 1 = 1 := by
  sorry

end NUMINAMATH_CALUDE_function_property_l3296_329657


namespace NUMINAMATH_CALUDE_arithmetic_sequence_sum_base6_l3296_329696

/-- Represents a number in base 6 --/
structure Base6 :=
  (value : ℕ)
  (isValid : value < 6)

/-- Converts a natural number to its base 6 representation --/
def toBase6 (n : ℕ) : List Base6 :=
  sorry

/-- Arithmetic sequence in base 6 --/
def arithmeticSequenceBase6 (a l d : Base6) : List Base6 :=
  sorry

/-- Sum of a list of Base6 numbers --/
def sumBase6 (lst : List Base6) : List Base6 :=
  sorry

theorem arithmetic_sequence_sum_base6 :
  let a := Base6.mk 1 (by norm_num)
  let l := Base6.mk 5 (by norm_num) -- 41 in base 6 is 5 * 6 + 5 = 35
  let d := Base6.mk 2 (by norm_num)
  let sequence := arithmeticSequenceBase6 a l d
  sumBase6 sequence = toBase6 441 :=
by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_sum_base6_l3296_329696


namespace NUMINAMATH_CALUDE_transform_graph_point_l3296_329686

/-- Given a function g : ℝ → ℝ such that g(8) = 5, prove that (8/3, 14/9) is on the graph of
    3y = g(3x)/3 + 3 and the sum of its coordinates is 38/9 -/
theorem transform_graph_point (g : ℝ → ℝ) (h : g 8 = 5) :
  let f : ℝ → ℝ := λ x => (g (3 * x) / 3 + 3) / 3
  f (8/3) = 14/9 ∧ 8/3 + 14/9 = 38/9 := by
  sorry

end NUMINAMATH_CALUDE_transform_graph_point_l3296_329686


namespace NUMINAMATH_CALUDE_units_digit_of_n_squared_plus_two_to_n_l3296_329611

theorem units_digit_of_n_squared_plus_two_to_n (n : ℕ) :
  n = 2023^2 + 2^2023 →
  (n^2 + 2^n) % 10 = 7 := by
sorry

end NUMINAMATH_CALUDE_units_digit_of_n_squared_plus_two_to_n_l3296_329611


namespace NUMINAMATH_CALUDE_arithmetic_sequence_sum_l3296_329616

def is_arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_sum (a : ℕ → ℝ) :
  is_arithmetic_sequence a →
  a 3 + a 5 + a 12 - a 2 = 12 →
  a 7 + a 11 = 12 := by
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_sum_l3296_329616


namespace NUMINAMATH_CALUDE_rhombus_area_rhombus_area_is_88_l3296_329626

/-- The area of a rhombus with vertices at (0, 5.5), (8, 0), (0, -5.5), and (-8, 0) is 88 square units. -/
theorem rhombus_area : ℝ → Prop :=
  fun area =>
    let v1 : ℝ × ℝ := (0, 5.5)
    let v2 : ℝ × ℝ := (8, 0)
    let v3 : ℝ × ℝ := (0, -5.5)
    let v4 : ℝ × ℝ := (-8, 0)
    let d1 : ℝ := v1.2 - v3.2
    let d2 : ℝ := v2.1 - v4.1
    area = (d1 * d2) / 2 ∧ area = 88

theorem rhombus_area_is_88 : rhombus_area 88 := by
  sorry

end NUMINAMATH_CALUDE_rhombus_area_rhombus_area_is_88_l3296_329626


namespace NUMINAMATH_CALUDE_blurred_pages_frequency_l3296_329688

theorem blurred_pages_frequency 
  (total_pages : ℕ) 
  (crumpled_frequency : ℕ) 
  (neither_crumpled_nor_blurred : ℕ) 
  (h1 : total_pages = 42)
  (h2 : crumpled_frequency = 7)
  (h3 : neither_crumpled_nor_blurred = 24) :
  (total_pages - neither_crumpled_nor_blurred - (total_pages / crumpled_frequency)) / total_pages = 2 / 7 := by
sorry

end NUMINAMATH_CALUDE_blurred_pages_frequency_l3296_329688


namespace NUMINAMATH_CALUDE_inscribed_cube_volume_l3296_329660

/-- The volume of a cube inscribed in a sphere, which is itself inscribed in a larger cube -/
theorem inscribed_cube_volume (outer_cube_edge : ℝ) (h : outer_cube_edge = 12) :
  let sphere_diameter := outer_cube_edge
  let inner_cube_diagonal := sphere_diameter
  let inner_cube_edge := inner_cube_diagonal / Real.sqrt 3
  let inner_cube_volume := inner_cube_edge ^ 3
  inner_cube_volume = 192 * Real.sqrt 3 := by
  sorry

#check inscribed_cube_volume

end NUMINAMATH_CALUDE_inscribed_cube_volume_l3296_329660


namespace NUMINAMATH_CALUDE_polygon_20_vertices_has_170_diagonals_l3296_329675

/-- The number of diagonals in a polygon with n vertices -/
def num_diagonals (n : ℕ) : ℕ := n * (n - 3) / 2

/-- Theorem: A polygon with 20 vertices has 170 diagonals -/
theorem polygon_20_vertices_has_170_diagonals :
  num_diagonals 20 = 170 := by
  sorry

end NUMINAMATH_CALUDE_polygon_20_vertices_has_170_diagonals_l3296_329675


namespace NUMINAMATH_CALUDE_inequality_preservation_l3296_329630

theorem inequality_preservation (x y : ℝ) (h : x > y) : x / 5 > y / 5 := by
  sorry

end NUMINAMATH_CALUDE_inequality_preservation_l3296_329630


namespace NUMINAMATH_CALUDE_calculate_expression_l3296_329636

theorem calculate_expression : 
  50000 - ((37500 / 62.35)^2 + Real.sqrt 324) = -311752.222 := by
  sorry

end NUMINAMATH_CALUDE_calculate_expression_l3296_329636


namespace NUMINAMATH_CALUDE_plums_picked_total_l3296_329652

/-- The number of plums Melanie picked -/
def melanie_plums : ℕ := 4

/-- The number of plums Dan picked -/
def dan_plums : ℕ := 9

/-- The number of plums Sally picked -/
def sally_plums : ℕ := 3

/-- The total number of plums picked -/
def total_plums : ℕ := melanie_plums + dan_plums + sally_plums

theorem plums_picked_total : total_plums = 16 := by
  sorry

end NUMINAMATH_CALUDE_plums_picked_total_l3296_329652


namespace NUMINAMATH_CALUDE_zeta_sum_seventh_power_l3296_329690

theorem zeta_sum_seventh_power (ζ₁ ζ₂ ζ₃ : ℂ) 
  (h1 : ζ₁ + ζ₂ + ζ₃ = 2)
  (h2 : ζ₁^2 + ζ₂^2 + ζ₃^2 = 5)
  (h3 : ζ₁^3 + ζ₂^3 + ζ₃^3 = 14) :
  ζ₁^7 + ζ₂^7 + ζ₃^7 = 233 := by
  sorry

end NUMINAMATH_CALUDE_zeta_sum_seventh_power_l3296_329690


namespace NUMINAMATH_CALUDE_eggs_for_blueberry_and_pecan_pies_l3296_329600

theorem eggs_for_blueberry_and_pecan_pies 
  (total_eggs : ℕ) 
  (pumpkin_eggs : ℕ) 
  (apple_eggs : ℕ) 
  (cherry_eggs : ℕ) 
  (h1 : total_eggs = 1820)
  (h2 : pumpkin_eggs = 816)
  (h3 : apple_eggs = 384)
  (h4 : cherry_eggs = 120) :
  total_eggs - (pumpkin_eggs + apple_eggs + cherry_eggs) = 500 :=
by sorry

end NUMINAMATH_CALUDE_eggs_for_blueberry_and_pecan_pies_l3296_329600


namespace NUMINAMATH_CALUDE_stan_run_time_l3296_329606

/-- Calculates the total run time given the number of 3-minute songs, 2-minute songs, and additional time needed. -/
def total_run_time (three_min_songs : ℕ) (two_min_songs : ℕ) (additional_time : ℕ) : ℕ :=
  three_min_songs * 3 + two_min_songs * 2 + additional_time

/-- Proves that given 10 3-minute songs, 15 2-minute songs, and 40 minutes of additional time, the total run time is 100 minutes. -/
theorem stan_run_time :
  total_run_time 10 15 40 = 100 := by
  sorry

end NUMINAMATH_CALUDE_stan_run_time_l3296_329606


namespace NUMINAMATH_CALUDE_unique_n_value_l3296_329661

/-- Represents a round-robin golf tournament with the given conditions -/
structure GolfTournament where
  /-- Total number of players -/
  T : ℕ
  /-- Number of points scored by each player other than Simon and Garfunkle -/
  n : ℕ
  /-- Condition: Total number of matches equals total points distributed -/
  matches_eq_points : T * (T - 1) / 2 = 16 + n * (T - 2)
  /-- Condition: Tournament has at least 3 players -/
  min_players : T ≥ 3

/-- Theorem stating that the only possible value for n is 17 -/
theorem unique_n_value (tournament : GolfTournament) : tournament.n = 17 := by
  sorry

end NUMINAMATH_CALUDE_unique_n_value_l3296_329661


namespace NUMINAMATH_CALUDE_cloth_cost_price_calculation_l3296_329692

/-- Given the selling price and loss per metre for a certain length of cloth,
    calculate the cost price per metre. -/
def cost_price_per_metre (selling_price total_length loss_per_metre : ℚ) : ℚ :=
  (selling_price + loss_per_metre * total_length) / total_length

/-- Theorem stating that under the given conditions, 
    the cost price per metre of cloth is 95. -/
theorem cloth_cost_price_calculation :
  let selling_price : ℚ := 18000
  let total_length : ℚ := 200
  let loss_per_metre : ℚ := 5
  cost_price_per_metre selling_price total_length loss_per_metre = 95 :=
by
  sorry


end NUMINAMATH_CALUDE_cloth_cost_price_calculation_l3296_329692


namespace NUMINAMATH_CALUDE_homeless_donation_problem_l3296_329633

/-- The amount given to the last set of homeless families -/
def last_set_amount (total spent_on_first_four : ℝ) : ℝ :=
  total - spent_on_first_four

/-- The problem statement -/
theorem homeless_donation_problem (total first second third fourth : ℝ) 
  (h1 : total = 4500)
  (h2 : first = 725)
  (h3 : second = 1100)
  (h4 : third = 950)
  (h5 : fourth = 815) :
  last_set_amount total (first + second + third + fourth) = 910 := by
  sorry

end NUMINAMATH_CALUDE_homeless_donation_problem_l3296_329633


namespace NUMINAMATH_CALUDE_rectangle_diagonal_maximum_l3296_329659

theorem rectangle_diagonal_maximum (l w : ℝ) : 
  (2 * l + 2 * w = 40) → 
  (∀ l' w' : ℝ, (2 * l' + 2 * w' = 40) → (l'^2 + w'^2 ≤ l^2 + w^2)) →
  l^2 + w^2 = 200 :=
sorry

end NUMINAMATH_CALUDE_rectangle_diagonal_maximum_l3296_329659


namespace NUMINAMATH_CALUDE_total_football_games_l3296_329673

theorem total_football_games (games_this_year games_last_year : ℕ) 
  (h1 : games_this_year = 14)
  (h2 : games_last_year = 29) :
  games_this_year + games_last_year = 43 := by
  sorry

end NUMINAMATH_CALUDE_total_football_games_l3296_329673


namespace NUMINAMATH_CALUDE_min_value_complex_sum_l3296_329648

theorem min_value_complex_sum (a b c d : ℤ) (ζ : ℂ) 
  (h_distinct : a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d)
  (h_fourth_root : ζ^4 = 1)
  (h_not_one : ζ ≠ 1) :
  ∃ (m : ℝ), m = 2 ∧ ∀ (x y z w : ℤ), x ≠ y ∧ x ≠ z ∧ x ≠ w ∧ y ≠ z ∧ y ≠ w ∧ z ≠ w →
    Complex.abs (x + y*ζ + z*ζ^2 + w*ζ^3) ≥ m :=
by sorry

end NUMINAMATH_CALUDE_min_value_complex_sum_l3296_329648


namespace NUMINAMATH_CALUDE_paint_weight_l3296_329691

theorem paint_weight (total_weight : ℝ) (half_empty_weight : ℝ) 
  (h1 : total_weight = 24)
  (h2 : half_empty_weight = 14) :
  total_weight - half_empty_weight = 10 ∧ 
  2 * (total_weight - half_empty_weight) = 20 := by
  sorry

#check paint_weight

end NUMINAMATH_CALUDE_paint_weight_l3296_329691


namespace NUMINAMATH_CALUDE_set_equality_proof_l3296_329682

universe u

def U : Set Nat := {1, 2, 3, 4}
def M : Set Nat := {1, 3, 4}
def N : Set Nat := {1, 2}

theorem set_equality_proof :
  ({2, 3, 4} : Set Nat) = (U \ M) ∪ (U \ N) := by sorry

end NUMINAMATH_CALUDE_set_equality_proof_l3296_329682


namespace NUMINAMATH_CALUDE_souvenir_problem_l3296_329635

/-- Represents the number of ways to select souvenirs -/
def souvenir_selection_ways (total_types : ℕ) (expensive_types : ℕ) (cheap_types : ℕ) 
  (expensive_price : ℕ) (cheap_price : ℕ) (total_spent : ℕ) : ℕ :=
  (Nat.choose expensive_types 5) + 
  (Nat.choose expensive_types 4) * (Nat.choose cheap_types 2)

/-- The problem statement -/
theorem souvenir_problem : 
  souvenir_selection_ways 11 8 3 10 5 50 = 266 := by
  sorry

end NUMINAMATH_CALUDE_souvenir_problem_l3296_329635


namespace NUMINAMATH_CALUDE_lcm_and_gcd_of_36_and_48_l3296_329676

theorem lcm_and_gcd_of_36_and_48 :
  (Nat.lcm 36 48 = 144) ∧ (Nat.gcd 36 48 = 12) := by
  sorry

end NUMINAMATH_CALUDE_lcm_and_gcd_of_36_and_48_l3296_329676


namespace NUMINAMATH_CALUDE_fixed_point_of_exponential_function_l3296_329615

theorem fixed_point_of_exponential_function (a : ℝ) (h1 : a > 0) (h2 : a ≠ 1) :
  ∃ (x : ℝ), x = 1 ∧ (a^(x - 1) + 1 = x) := by
  sorry

end NUMINAMATH_CALUDE_fixed_point_of_exponential_function_l3296_329615


namespace NUMINAMATH_CALUDE_container_volume_ratio_l3296_329641

theorem container_volume_ratio : 
  ∀ (A B C : ℚ),
  (4/5 : ℚ) * A = (3/5 : ℚ) * B →
  (3/5 : ℚ) * B = (3/4 : ℚ) * C →
  A / C = (15/16 : ℚ) :=
by
  sorry

end NUMINAMATH_CALUDE_container_volume_ratio_l3296_329641


namespace NUMINAMATH_CALUDE_x_y_negative_l3296_329667

theorem x_y_negative (x y : ℝ) (h1 : x - y > x) (h2 : x + y < y) : x < 0 ∧ y < 0 := by
  sorry

end NUMINAMATH_CALUDE_x_y_negative_l3296_329667


namespace NUMINAMATH_CALUDE_dodecahedron_edge_probability_l3296_329693

/-- A regular dodecahedron -/
structure RegularDodecahedron where
  /-- The number of vertices in a regular dodecahedron -/
  num_vertices : ℕ
  /-- The number of edges connected to each vertex -/
  edges_per_vertex : ℕ
  /-- Properties of a regular dodecahedron -/
  vertex_count : num_vertices = 20
  edge_count : edges_per_vertex = 3

/-- The probability of randomly choosing two vertices that form an edge in a regular dodecahedron -/
def edge_probability (d : RegularDodecahedron) : ℚ :=
  3 / 19

/-- Theorem stating the probability of randomly choosing two vertices that form an edge in a regular dodecahedron -/
theorem dodecahedron_edge_probability (d : RegularDodecahedron) :
  edge_probability d = 3 / 19 := by
  sorry

end NUMINAMATH_CALUDE_dodecahedron_edge_probability_l3296_329693


namespace NUMINAMATH_CALUDE_inequality_proof_l3296_329604

theorem inequality_proof (x y : ℝ) : x^2 + y^2 + 1 ≥ x*y + x + y := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l3296_329604


namespace NUMINAMATH_CALUDE_unique_number_l3296_329684

def is_three_digit_number (n : ℕ) : Prop := 100 ≤ n ∧ n < 1000

def has_distinct_digits (n : ℕ) : Prop :=
  let hundreds := n / 100
  let tens := (n / 10) % 10
  let units := n % 10
  hundreds ≠ tens ∧ hundreds ≠ units ∧ tens ≠ units

def is_perfect_square (n : ℕ) : Prop :=
  ∃ m : ℕ, n = m * m

def person_a_initially_unsure (n : ℕ) : Prop :=
  ∃ m : ℕ, m ≠ n ∧ 
    is_three_digit_number m ∧ 
    is_perfect_square m ∧ 
    has_distinct_digits m ∧ 
    m / 100 = n / 100

def person_b_knows_a_unsure (n : ℕ) : Prop :=
  ∀ m : ℕ, (m / 10) % 10 = (n / 10) % 10 → person_a_initially_unsure m

def person_c_knows_number (n : ℕ) : Prop :=
  ∀ m : ℕ, m ≠ n →
    ¬(is_three_digit_number m ∧ 
      is_perfect_square m ∧ 
      has_distinct_digits m ∧ 
      person_a_initially_unsure m ∧ 
      person_b_knows_a_unsure m ∧ 
      m % 10 = n % 10)

def person_a_knows_after_c (n : ℕ) : Prop :=
  ∀ m : ℕ, m ≠ n →
    ¬(is_three_digit_number m ∧ 
      is_perfect_square m ∧ 
      has_distinct_digits m ∧ 
      person_a_initially_unsure m ∧ 
      person_b_knows_a_unsure m ∧ 
      person_c_knows_number m ∧ 
      m / 100 = n / 100)

def person_b_knows_after_a (n : ℕ) : Prop :=
  ∀ m : ℕ, m ≠ n →
    ¬(is_three_digit_number m ∧ 
      is_perfect_square m ∧ 
      has_distinct_digits m ∧ 
      person_a_initially_unsure m ∧ 
      person_b_knows_a_unsure m ∧ 
      person_c_knows_number m ∧ 
      person_a_knows_after_c m ∧ 
      (m / 10) % 10 = (n / 10) % 10)

theorem unique_number : 
  ∃! n : ℕ, 
    is_three_digit_number n ∧ 
    is_perfect_square n ∧ 
    has_distinct_digits n ∧ 
    person_a_initially_unsure n ∧ 
    person_b_knows_a_unsure n ∧ 
    person_c_knows_number n ∧ 
    person_a_knows_after_c n ∧ 
    person_b_knows_after_a n ∧ 
    n = 289 := by sorry

end NUMINAMATH_CALUDE_unique_number_l3296_329684


namespace NUMINAMATH_CALUDE_sum_even_divisors_180_l3296_329653

/-- Sum of positive even divisors of a natural number n -/
def sumEvenDivisors (n : ℕ) : ℕ := sorry

/-- Theorem: The sum of positive even divisors of 180 is 468 -/
theorem sum_even_divisors_180 : sumEvenDivisors 180 = 468 := by sorry

end NUMINAMATH_CALUDE_sum_even_divisors_180_l3296_329653


namespace NUMINAMATH_CALUDE_geometric_progression_problem_l3296_329620

def geometric_progression (b₁ : ℝ) (q : ℝ) (n : ℕ) : ℝ := b₁ * q^(n - 1)

theorem geometric_progression_problem (b₁ b₅ : ℝ) (h₁ : b₁ = Real.sqrt 3) (h₅ : b₅ = Real.sqrt 243) :
  ∃ q : ℝ, (q = Real.sqrt 3 ∨ q = -Real.sqrt 3) ∧
    geometric_progression b₁ q 5 = b₅ ∧
    geometric_progression b₁ q 6 = 27 ∨ geometric_progression b₁ q 6 = -27 :=
by sorry

end NUMINAMATH_CALUDE_geometric_progression_problem_l3296_329620


namespace NUMINAMATH_CALUDE_classroom_ratio_l3296_329619

theorem classroom_ratio (num_boys num_girls : ℕ) 
  (h_positive : num_boys > 0 ∧ num_girls > 0) :
  let total := num_boys + num_girls
  let prob_boy := num_boys / total
  let prob_girl := num_girls / total
  prob_boy = (3/4 : ℚ) * prob_girl →
  (num_boys : ℚ) / total = 3/7 := by
sorry

end NUMINAMATH_CALUDE_classroom_ratio_l3296_329619


namespace NUMINAMATH_CALUDE_quadratic_real_root_condition_l3296_329637

theorem quadratic_real_root_condition (b : ℝ) : 
  (∃ x : ℝ, x^2 + b*x + 25 = 0) ↔ b ≤ -10 ∨ b ≥ 10 := by
sorry

end NUMINAMATH_CALUDE_quadratic_real_root_condition_l3296_329637


namespace NUMINAMATH_CALUDE_bouquet_cost_50_l3296_329645

/-- Represents the cost function for bouquets at Bella's Blossom Shop -/
def bouquet_cost (n : ℕ) : ℚ :=
  let base_price := (36 : ℚ) / 18 * n.min 40
  let extra_price := if n > 40 then (36 : ℚ) / 18 * (n - 40) else 0
  let total_price := base_price + extra_price
  if n > 40 then total_price * (9 / 10) else total_price

theorem bouquet_cost_50 : bouquet_cost 50 = 90 := by
  sorry

end NUMINAMATH_CALUDE_bouquet_cost_50_l3296_329645


namespace NUMINAMATH_CALUDE_mona_group_size_l3296_329627

/-- The number of groups Mona joined --/
def num_groups : ℕ := 9

/-- The number of unique players Mona grouped with --/
def unique_players : ℕ := 33

/-- The number of non-unique player slots --/
def non_unique_slots : ℕ := 3

/-- The number of players in each group, including Mona --/
def players_per_group : ℕ := 5

theorem mona_group_size :
  (num_groups * (players_per_group - 1)) - non_unique_slots = unique_players :=
by sorry

end NUMINAMATH_CALUDE_mona_group_size_l3296_329627


namespace NUMINAMATH_CALUDE_fixed_point_exponential_function_l3296_329668

theorem fixed_point_exponential_function (a : ℝ) (ha : a > 0) (hna : a ≠ 1) :
  let f : ℝ → ℝ := fun x ↦ a^(x - 1) + 1
  f 1 = 2 := by
  sorry

end NUMINAMATH_CALUDE_fixed_point_exponential_function_l3296_329668


namespace NUMINAMATH_CALUDE_remainder_sum_mod_eight_l3296_329669

theorem remainder_sum_mod_eight (a b c : ℕ) : 
  a < 8 → b < 8 → c < 8 → a > 0 → b > 0 → c > 0 →
  (a * b * c) % 8 = 1 →
  (7 * c) % 8 = 3 →
  (5 * b) % 8 = (4 + b) % 8 →
  (a + b + c) % 8 = 3 := by
sorry

end NUMINAMATH_CALUDE_remainder_sum_mod_eight_l3296_329669


namespace NUMINAMATH_CALUDE_max_training_cost_l3296_329685

def training_cost (x : ℕ) : ℕ :=
  if x ≤ 30 then 1400 * x
  else 2000 * x - 20 * x * x

theorem max_training_cost :
  ∃ (x : ℕ), x ≤ 60 ∧ ∀ (y : ℕ), y ≤ 60 → training_cost y ≤ training_cost x ∧ training_cost x = 50000 := by
  sorry

end NUMINAMATH_CALUDE_max_training_cost_l3296_329685


namespace NUMINAMATH_CALUDE_arithmetic_expression_evaluation_l3296_329683

theorem arithmetic_expression_evaluation :
  (80 / 16) + (100 / 25) + ((6^2) * 3) - 300 - ((324 / 9) * 2) = -255 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_expression_evaluation_l3296_329683


namespace NUMINAMATH_CALUDE_quadratic_equation_roots_l3296_329671

theorem quadratic_equation_roots (m : ℝ) (x₁ x₂ : ℝ) : 
  (∀ x, x^2 - 4*x - 2*m + 5 = 0 ↔ (x = x₁ ∨ x = x₂)) →
  (x₁ ≠ x₂) →
  (m ≥ 1/2) ∧ 
  (x₁ * x₂ + x₁ + x₂ = m^2 + 6 → m = 1) := by
sorry

end NUMINAMATH_CALUDE_quadratic_equation_roots_l3296_329671


namespace NUMINAMATH_CALUDE_polynomial_factor_l3296_329647

/-- The polynomial P(x) = x^3 - 3x^2 + cx - 8 -/
def P (c : ℝ) (x : ℝ) : ℝ := x^3 - 3*x^2 + c*x - 8

theorem polynomial_factor (c : ℝ) : 
  (∀ x, P c x = 0 ↔ (x + 2 = 0 ∨ ∃ q, P c x = (x + 2) * q)) → c = -14 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_factor_l3296_329647


namespace NUMINAMATH_CALUDE_system_not_unique_solution_l3296_329663

/-- The system of equations does not have a unique solution when k = 9 -/
theorem system_not_unique_solution (x y z : ℝ) (k m : ℝ) :
  (3 * (3 * x^2 + 4 * y^2) = 36) →
  (k * x^2 + 12 * y^2 = 30) →
  (m * x^3 - 2 * y^3 + z^2 = 24) →
  (k = 9 → ∃ (c : ℝ), c ≠ 0 ∧ (3 * x^2 + 4 * y^2 = c * (k * x^2 + 12 * y^2))) :=
by sorry

end NUMINAMATH_CALUDE_system_not_unique_solution_l3296_329663


namespace NUMINAMATH_CALUDE_sector_area_l3296_329622

theorem sector_area (α l : Real) (h1 : α = π / 6) (h2 : l = π / 3) :
  let r := l / α
  let s := (1 / 2) * l * r
  s = π / 3 := by
  sorry

end NUMINAMATH_CALUDE_sector_area_l3296_329622


namespace NUMINAMATH_CALUDE_bridge_extension_l3296_329643

/-- The width of the river in inches -/
def river_width : ℕ := 487

/-- The length of the existing bridge in inches -/
def existing_bridge_length : ℕ := 295

/-- The additional length needed for the bridge to cross the river -/
def additional_length : ℕ := river_width - existing_bridge_length

theorem bridge_extension :
  additional_length = 192 := by sorry

end NUMINAMATH_CALUDE_bridge_extension_l3296_329643


namespace NUMINAMATH_CALUDE_choose_four_different_suits_standard_deck_l3296_329681

/-- Represents a standard deck of cards -/
structure Deck :=
  (cards : Nat)
  (suits : Nat)
  (cards_per_suit : Nat)
  (h1 : cards = suits * cards_per_suit)

/-- The number of ways to choose 4 cards of different suits from a standard deck -/
def choose_four_different_suits (d : Deck) : Nat :=
  d.cards_per_suit ^ d.suits

/-- Theorem stating the number of ways to choose 4 cards of different suits from a standard deck -/
theorem choose_four_different_suits_standard_deck :
  ∃ (d : Deck), d.cards = 52 ∧ d.suits = 4 ∧ d.cards_per_suit = 13 ∧ choose_four_different_suits d = 28561 := by
  sorry

end NUMINAMATH_CALUDE_choose_four_different_suits_standard_deck_l3296_329681


namespace NUMINAMATH_CALUDE_triangle_properties_l3296_329634

-- Define a triangle ABC
structure Triangle where
  A : Real
  B : Real
  C : Real
  a : Real
  b : Real
  c : Real

-- Define the theorem
theorem triangle_properties (t : Triangle) 
  (h1 : t.b - (1/2) * t.c = t.a * Real.cos t.C)
  (h2 : 4 * (t.b + t.c) = 3 * t.b * t.c)
  (h3 : t.a = 2 * Real.sqrt 3) :
  t.A = Real.pi / 3 ∧ 
  (1/2) * t.b * t.c * Real.sin t.A = 2 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_triangle_properties_l3296_329634


namespace NUMINAMATH_CALUDE_symmetric_point_y_axis_l3296_329640

/-- A point in 3D space -/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- Symmetry with respect to the y-axis -/
def symmetricYAxis (p : Point3D) : Point3D :=
  { x := -p.x, y := p.y, z := -p.z }

/-- The theorem stating that the point symmetric to (2, -3, 5) with respect to the y-axis is (-2, -3, -5) -/
theorem symmetric_point_y_axis :
  let original := Point3D.mk 2 (-3) 5
  symmetricYAxis original = Point3D.mk (-2) (-3) (-5) := by
  sorry

end NUMINAMATH_CALUDE_symmetric_point_y_axis_l3296_329640


namespace NUMINAMATH_CALUDE_average_sale_is_6900_l3296_329680

def sales : List ℕ := [6435, 6927, 6855, 7230, 6562, 7391]

theorem average_sale_is_6900 :
  (sales.sum : ℚ) / sales.length = 6900 := by
  sorry

end NUMINAMATH_CALUDE_average_sale_is_6900_l3296_329680


namespace NUMINAMATH_CALUDE_garden_fencing_needed_l3296_329623

/-- Calculates the perimeter of a rectangular garden with given length and width. -/
def garden_perimeter (length width : ℝ) : ℝ := 2 * (length + width)

/-- Proves that a rectangular garden with length 300 yards and width half its length
    requires 900 yards of fencing. -/
theorem garden_fencing_needed :
  let length : ℝ := 300
  let width : ℝ := length / 2
  garden_perimeter length width = 900 := by
sorry

#eval garden_perimeter 300 150

end NUMINAMATH_CALUDE_garden_fencing_needed_l3296_329623


namespace NUMINAMATH_CALUDE_equivalent_root_equations_l3296_329601

theorem equivalent_root_equations (a : ℝ) :
  ∀ x : ℝ, x = a + Real.sqrt (a + Real.sqrt x) ↔ x = a + Real.sqrt x :=
by sorry

end NUMINAMATH_CALUDE_equivalent_root_equations_l3296_329601


namespace NUMINAMATH_CALUDE_charity_game_probability_l3296_329699

theorem charity_game_probability : 
  let p1 : ℝ := 0.9  -- Probability of correct answer for first picture
  let p2 : ℝ := 0.5  -- Probability of correct answer for second picture
  let p3 : ℝ := 0.4  -- Probability of correct answer for third picture
  let f1 : ℕ := 1000 -- Fund raised for first correct answer
  let f2 : ℕ := 2000 -- Fund raised for second correct answer
  let f3 : ℕ := 3000 -- Fund raised for third correct answer
  -- Probability of raising exactly 3000 yuan
  p1 * p2 * (1 - p3) = 0.27
  := by sorry

end NUMINAMATH_CALUDE_charity_game_probability_l3296_329699


namespace NUMINAMATH_CALUDE_square_area_increase_l3296_329603

theorem square_area_increase (s : ℝ) (k : ℝ) (h1 : s > 0) (h2 : k > 0) :
  (k * s)^2 = 25 * s^2 → k = 5 := by
  sorry

end NUMINAMATH_CALUDE_square_area_increase_l3296_329603


namespace NUMINAMATH_CALUDE_chocolates_remaining_chocolates_remaining_day6_l3296_329610

/-- Chocolates remaining after 5 days of eating with given conditions -/
theorem chocolates_remaining (total : ℕ) (day1 : ℕ) (day2 : ℕ) : ℕ :=
  let day3 := day1 - 3
  let day4 := 2 * day3 + 1
  let day5 := day2 / 2
  total - (day1 + day2 + day3 + day4 + day5)

/-- Proof that 14 chocolates remain on Day 6 given the problem conditions -/
theorem chocolates_remaining_day6 :
  chocolates_remaining 48 6 8 = 14 := by
  sorry

end NUMINAMATH_CALUDE_chocolates_remaining_chocolates_remaining_day6_l3296_329610


namespace NUMINAMATH_CALUDE_jane_nail_polish_drying_time_l3296_329697

/-- The total drying time for Jane's nail polish -/
def total_drying_time : ℕ :=
  let base_coat := 4
  let first_color := 5
  let second_color := 6
  let third_color := 7
  let first_nail_art := 8
  let second_nail_art := 10
  let top_coat := 9
  base_coat + first_color + second_color + third_color + first_nail_art + second_nail_art + top_coat

theorem jane_nail_polish_drying_time :
  total_drying_time = 49 := by sorry

end NUMINAMATH_CALUDE_jane_nail_polish_drying_time_l3296_329697


namespace NUMINAMATH_CALUDE_diamond_and_hearts_balance_l3296_329646

-- Define the symbols
variable (triangle diamond heart dot : ℕ)

-- Define the balance relation
def balances (left right : ℕ) : Prop := left = right

-- State the given conditions
axiom balance1 : balances (4 * triangle + 2 * diamond + heart) (21 * dot)
axiom balance2 : balances (2 * triangle) (diamond + heart + 5 * dot)

-- State the theorem to be proved
theorem diamond_and_hearts_balance : balances (diamond + 2 * heart) (11 * dot) := by sorry

end NUMINAMATH_CALUDE_diamond_and_hearts_balance_l3296_329646


namespace NUMINAMATH_CALUDE_problem_1_problem_2_problem_3_problem_4_problem_5_problem_6_l3296_329632

-- Problem 1
theorem problem_1 : (1 * -5) + 9 = 4 := by sorry

-- Problem 2
theorem problem_2 : 12 - (-16) + (-2) - 1 = 25 := by sorry

-- Problem 3
theorem problem_3 : 6 / (-2) * (-1/3) = 1 := by sorry

-- Problem 4
theorem problem_4 : (-15) * (1/3 + 1/5) = -8 := by sorry

-- Problem 5
theorem problem_5 : (-2)^3 - (-8) / |-(4/3)| = -2 := by sorry

-- Problem 6
theorem problem_6 : -(1^2022) - (1/2 - 1/3) * 3 = -3/2 := by sorry

end NUMINAMATH_CALUDE_problem_1_problem_2_problem_3_problem_4_problem_5_problem_6_l3296_329632


namespace NUMINAMATH_CALUDE_one_eighth_divided_by_one_fourth_l3296_329609

theorem one_eighth_divided_by_one_fourth (a b c : ℚ) :
  a = 1 / 8 → b = 1 / 4 → c = 1 / 2 → a / b = c := by
  sorry

end NUMINAMATH_CALUDE_one_eighth_divided_by_one_fourth_l3296_329609


namespace NUMINAMATH_CALUDE_bug_ends_on_two_l3296_329694

/-- Represents the points on the circle -/
inductive Point
| one
| two
| three
| four
| five
| six

/-- Defines the movement rules for the bug -/
def next_point (p : Point) : Point :=
  match p with
  | Point.one => Point.two
  | Point.two => Point.four
  | Point.three => Point.four
  | Point.four => Point.one
  | Point.five => Point.six
  | Point.six => Point.two

/-- Simulates the bug's movement for a given number of jumps -/
def bug_position (start : Point) (jumps : Nat) : Point :=
  match jumps with
  | 0 => start
  | n + 1 => next_point (bug_position start n)

/-- The main theorem to prove -/
theorem bug_ends_on_two :
  bug_position Point.six 2000 = Point.two := by
  sorry

end NUMINAMATH_CALUDE_bug_ends_on_two_l3296_329694


namespace NUMINAMATH_CALUDE_angle_inequality_l3296_329677

theorem angle_inequality (α β γ : Real) 
  (h1 : 0 < α) (h2 : α ≤ β) (h3 : β ≤ γ) (h4 : γ < π) :
  Real.sin (α / 2) + Real.sin (β / 2) > Real.sin (γ / 2) := by
  sorry

end NUMINAMATH_CALUDE_angle_inequality_l3296_329677


namespace NUMINAMATH_CALUDE_solve_equation_l3296_329674

theorem solve_equation (y : ℝ) : 7 - y = 10 → y = -3 := by
  sorry

end NUMINAMATH_CALUDE_solve_equation_l3296_329674


namespace NUMINAMATH_CALUDE_no_egyptian_fraction_for_seven_seventeenths_l3296_329625

theorem no_egyptian_fraction_for_seven_seventeenths :
  ¬ ∃ (a b : ℕ+), (7 : ℚ) / 17 = 1 / (a : ℚ) + 1 / (b : ℚ) := by
  sorry

end NUMINAMATH_CALUDE_no_egyptian_fraction_for_seven_seventeenths_l3296_329625


namespace NUMINAMATH_CALUDE_union_M_N_l3296_329624

def M : Set ℝ := {x | 1 / x > 1}
def N : Set ℝ := {x | x^2 + 2*x - 3 < 0}

theorem union_M_N : M ∪ N = Set.Ioo (-3) 1 := by
  sorry

end NUMINAMATH_CALUDE_union_M_N_l3296_329624


namespace NUMINAMATH_CALUDE_paula_shirts_bought_l3296_329662

def shirts_bought (initial_amount : ℕ) (shirt_cost : ℕ) (pants_cost : ℕ) (remaining_amount : ℕ) : ℕ :=
  (initial_amount - pants_cost - remaining_amount) / shirt_cost

theorem paula_shirts_bought :
  shirts_bought 109 11 13 74 = 2 := by
  sorry

end NUMINAMATH_CALUDE_paula_shirts_bought_l3296_329662


namespace NUMINAMATH_CALUDE_factorization_proof_l3296_329628

theorem factorization_proof (x : ℝ) : 2*x^3 - 8*x^2 + 8*x = 2*x*(x - 2)^2 := by
  sorry

end NUMINAMATH_CALUDE_factorization_proof_l3296_329628


namespace NUMINAMATH_CALUDE_complex_equation_solution_l3296_329642

theorem complex_equation_solution (z : ℂ) :
  z * (1 + 3 * Complex.I) = 4 + Complex.I →
  z = 7/10 - 11/10 * Complex.I := by
  sorry

end NUMINAMATH_CALUDE_complex_equation_solution_l3296_329642


namespace NUMINAMATH_CALUDE_smallest_valid_configuration_l3296_329698

/-- A configuration of lines in a plane -/
structure LineConfiguration where
  n : ℕ                   -- Total number of lines
  intersect_5 : ℕ         -- Index of line intersecting 5 others
  intersect_9 : ℕ         -- Index of line intersecting 9 others
  intersect_11 : ℕ        -- Index of line intersecting 11 others
  intersect_5_count : ℕ   -- Number of intersections for intersect_5
  intersect_9_count : ℕ   -- Number of intersections for intersect_9
  intersect_11_count : ℕ  -- Number of intersections for intersect_11

/-- Predicate to check if a line configuration is valid -/
def is_valid_configuration (config : LineConfiguration) : Prop :=
  config.n > 0 ∧
  config.intersect_5 < config.n ∧
  config.intersect_9 < config.n ∧
  config.intersect_11 < config.n ∧
  config.intersect_5_count = 5 ∧
  config.intersect_9_count = 9 ∧
  config.intersect_11_count = 11

/-- Theorem stating that 12 is the smallest number of lines satisfying the conditions -/
theorem smallest_valid_configuration :
  (∃ (config : LineConfiguration), is_valid_configuration config ∧ config.n = 12) ∧
  (∀ (config : LineConfiguration), is_valid_configuration config → config.n ≥ 12) :=
sorry

end NUMINAMATH_CALUDE_smallest_valid_configuration_l3296_329698


namespace NUMINAMATH_CALUDE_sum_first_105_remainder_l3296_329617

theorem sum_first_105_remainder (n : Nat) (d : Nat) : n = 105 → d = 5270 → (n * (n + 1) / 2) % d = 295 := by
  sorry

end NUMINAMATH_CALUDE_sum_first_105_remainder_l3296_329617


namespace NUMINAMATH_CALUDE_pie_point_returns_to_initial_position_l3296_329651

/-- Represents a point on a circular pie --/
structure PiePoint where
  angle : Real
  radius : Real

/-- Represents the operation of cutting, flipping, and rotating the pie --/
def pieOperation (α β : Real) (p : PiePoint) : PiePoint :=
  sorry

/-- The main theorem statement --/
theorem pie_point_returns_to_initial_position
  (α β : Real)
  (h1 : β < α)
  (h2 : α < 180)
  : ∃ N : ℕ, ∀ p : PiePoint,
    (pieOperation α β)^[N] p = p :=
  sorry

end NUMINAMATH_CALUDE_pie_point_returns_to_initial_position_l3296_329651


namespace NUMINAMATH_CALUDE_log_N_between_consecutive_integers_l3296_329687

theorem log_N_between_consecutive_integers 
  (N : ℝ) 
  (h : Real.log 2500 < Real.log N ∧ Real.log N < Real.log 10000) : 
  ∃ (m : ℤ), m + (m + 1) = 7 ∧ 
    (↑m : ℝ) < Real.log N ∧ Real.log N < (↑m + 1 : ℝ) := by
sorry

end NUMINAMATH_CALUDE_log_N_between_consecutive_integers_l3296_329687


namespace NUMINAMATH_CALUDE_building_floor_height_l3296_329664

/-- Proves that the height of each of the first 18 floors is 3 meters -/
theorem building_floor_height
  (total_floors : ℕ)
  (last_two_extra_height : ℝ)
  (total_height : ℝ)
  (h : ℝ)
  (h_total_floors : total_floors = 20)
  (h_last_two_extra : last_two_extra_height = 0.5)
  (h_total_height : total_height = 61)
  (h_height_equation : 18 * h + 2 * (h + last_two_extra_height) = total_height) :
  h = 3 := by
sorry

end NUMINAMATH_CALUDE_building_floor_height_l3296_329664


namespace NUMINAMATH_CALUDE_remainder_theorem_l3296_329655

theorem remainder_theorem (x y u v : ℕ) (h1 : y > 0) (h2 : x = u * y + v) (h3 : v < y) :
  (x + 3 * u * y) % y = v := by
sorry

end NUMINAMATH_CALUDE_remainder_theorem_l3296_329655


namespace NUMINAMATH_CALUDE_rachel_picked_apples_l3296_329613

def apples_picked (initial_apples remaining_apples : ℕ) : ℕ :=
  initial_apples - remaining_apples

theorem rachel_picked_apples (initial_apples remaining_apples : ℕ) 
  (h1 : initial_apples = 9)
  (h2 : remaining_apples = 7) :
  apples_picked initial_apples remaining_apples = 2 := by
sorry

end NUMINAMATH_CALUDE_rachel_picked_apples_l3296_329613

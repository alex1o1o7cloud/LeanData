import Mathlib

namespace NUMINAMATH_CALUDE_no_square_free_arithmetic_sequence_l2782_278261

theorem no_square_free_arithmetic_sequence :
  ∀ (a d : ℕ+), d ≠ 1 →
  ∃ (n : ℕ), ∃ (k : ℕ), k > 1 ∧ (k * k ∣ (a + n * d)) :=
by sorry

end NUMINAMATH_CALUDE_no_square_free_arithmetic_sequence_l2782_278261


namespace NUMINAMATH_CALUDE_complex_number_magnitude_squared_l2782_278246

theorem complex_number_magnitude_squared :
  ∀ (z : ℂ), z + Complex.abs z = 4 + 5*I → Complex.abs z^2 = 1681/64 := by
  sorry

end NUMINAMATH_CALUDE_complex_number_magnitude_squared_l2782_278246


namespace NUMINAMATH_CALUDE_certain_number_value_l2782_278267

theorem certain_number_value : ∃ x : ℝ,
  ((0.02 : ℝ)^2 + (0.52 : ℝ)^2 + (0.035 : ℝ)^2) / (x^2 + (0.052 : ℝ)^2 + (0.0035 : ℝ)^2) = 100 ∧
  x = 0.002 :=
by sorry

end NUMINAMATH_CALUDE_certain_number_value_l2782_278267


namespace NUMINAMATH_CALUDE_room_height_is_12_l2782_278287

/-- Represents the dimensions and cost information for a room whitewashing problem -/
structure RoomWhitewash where
  length : ℝ
  width : ℝ
  height : ℝ
  doorLength : ℝ
  doorWidth : ℝ
  windowLength : ℝ
  windowWidth : ℝ
  numWindows : ℕ
  costPerSqFt : ℝ
  totalCost : ℝ

/-- Theorem stating that given the room dimensions and whitewashing costs, 
    the height of the room is 12 feet -/
theorem room_height_is_12 (r : RoomWhitewash) 
    (h1 : r.length = 25)
    (h2 : r.width = 15)
    (h3 : r.doorLength = 6)
    (h4 : r.doorWidth = 3)
    (h5 : r.windowLength = 4)
    (h6 : r.windowWidth = 3)
    (h7 : r.numWindows = 3)
    (h8 : r.costPerSqFt = 5)
    (h9 : r.totalCost = 4530)
    : r.height = 12 := by
  sorry


end NUMINAMATH_CALUDE_room_height_is_12_l2782_278287


namespace NUMINAMATH_CALUDE_c_share_is_45_l2782_278251

/-- Represents the rent share calculation for a pasture -/
structure PastureRent where
  total_rent : ℕ
  a_oxen : ℕ
  a_months : ℕ
  b_oxen : ℕ
  b_months : ℕ
  c_oxen : ℕ
  c_months : ℕ

/-- Calculates C's share of the rent -/
def calculate_c_share (p : PastureRent) : ℕ :=
  let total_ox_months := p.a_oxen * p.a_months + p.b_oxen * p.b_months + p.c_oxen * p.c_months
  let c_ox_months := p.c_oxen * p.c_months
  (c_ox_months * p.total_rent) / total_ox_months

/-- Theorem stating that C's share of the rent is 45 -/
theorem c_share_is_45 (p : PastureRent) 
  (h1 : p.total_rent = 175)
  (h2 : p.a_oxen = 10) (h3 : p.a_months = 7)
  (h4 : p.b_oxen = 12) (h5 : p.b_months = 5)
  (h6 : p.c_oxen = 15) (h7 : p.c_months = 3) :
  calculate_c_share p = 45 := by
  sorry

end NUMINAMATH_CALUDE_c_share_is_45_l2782_278251


namespace NUMINAMATH_CALUDE_tetrahedron_fits_in_box_l2782_278245

theorem tetrahedron_fits_in_box : ∃ (x y z : ℝ),
  (x^2 + y^2 = 100) ∧
  (x^2 + z^2 = 81) ∧
  (y^2 + z^2 = 64) ∧
  (x < 8) ∧ (y < 8) ∧ (z < 5) := by
  sorry

end NUMINAMATH_CALUDE_tetrahedron_fits_in_box_l2782_278245


namespace NUMINAMATH_CALUDE_probability_is_one_fourth_l2782_278273

/-- The number of pennies Keiko tosses -/
def keiko_pennies : ℕ := 1

/-- The number of pennies Ephraim tosses -/
def ephraim_pennies : ℕ := 3

/-- The total number of possible outcomes when tossing n pennies -/
def total_outcomes (n : ℕ) : ℕ := 2^n

/-- The number of favorable outcomes where Ephraim gets the same number of heads as Keiko -/
def favorable_outcomes : ℕ := 4

/-- The probability that Ephraim gets the same number of heads as Keiko -/
def probability : ℚ := favorable_outcomes / (total_outcomes keiko_pennies * total_outcomes ephraim_pennies)

theorem probability_is_one_fourth : probability = 1/4 := by
  sorry

end NUMINAMATH_CALUDE_probability_is_one_fourth_l2782_278273


namespace NUMINAMATH_CALUDE_a_equals_3_sufficient_not_necessary_l2782_278265

/-- Two lines are parallel if their slopes are equal and they are not identical. -/
def are_parallel (m1 a1 b1 c1 m2 a2 b2 c2 : ℝ) : Prop :=
  m1 = m2 ∧ (a1, b1, c1) ≠ (a2, b2, c2)

/-- The condition for the given lines to be parallel. -/
def parallel_condition (a : ℝ) : Prop :=
  are_parallel (-a/2) a 2 1 (-3/(a-1)) 3 (a-1) (-2)

theorem a_equals_3_sufficient_not_necessary :
  (∃ a, a ≠ 3 ∧ parallel_condition a) ∧
  (parallel_condition 3) :=
sorry

end NUMINAMATH_CALUDE_a_equals_3_sufficient_not_necessary_l2782_278265


namespace NUMINAMATH_CALUDE_product_of_squares_l2782_278238

theorem product_of_squares (a b : ℝ) 
  (h1 : a^2 - b^2 = 10) 
  (h2 : a^4 + b^4 = 228) : 
  a * b = 8 := by
sorry

end NUMINAMATH_CALUDE_product_of_squares_l2782_278238


namespace NUMINAMATH_CALUDE_square_minus_reciprocal_square_l2782_278288

theorem square_minus_reciprocal_square (x : ℝ) (h1 : x > 1) (h2 : x + 1/x = Real.sqrt 22) :
  x^2 - 1/x^2 = 3 * Real.sqrt 2 := by
sorry

end NUMINAMATH_CALUDE_square_minus_reciprocal_square_l2782_278288


namespace NUMINAMATH_CALUDE_solve_for_y_l2782_278232

theorem solve_for_y (x y : ℝ) (h1 : x + 2 * y = 100) (h2 : x = 50) : y = 25 := by
  sorry

end NUMINAMATH_CALUDE_solve_for_y_l2782_278232


namespace NUMINAMATH_CALUDE_sodas_consumed_l2782_278278

def potluck_sodas (brought : ℕ) (taken_back : ℕ) : ℕ :=
  brought - taken_back

theorem sodas_consumed (brought : ℕ) (taken_back : ℕ) 
  (h : brought ≥ taken_back) : 
  potluck_sodas brought taken_back = brought - taken_back :=
by sorry

end NUMINAMATH_CALUDE_sodas_consumed_l2782_278278


namespace NUMINAMATH_CALUDE_distribute_balls_to_bags_correct_l2782_278297

/-- The number of ways to distribute n identical balls into m numbered bags, such that no bag is empty -/
def distribute_balls_to_bags (n m : ℕ) : ℕ :=
  Nat.choose (n - 1) (m - 1)

/-- Theorem: The number of ways to distribute n identical balls into m numbered bags, 
    such that no bag is empty, is equal to (n-1) choose (m-1) -/
theorem distribute_balls_to_bags_correct (n m : ℕ) (h1 : n > m) (h2 : m > 0) : 
  distribute_balls_to_bags n m = Nat.choose (n - 1) (m - 1) := by
  sorry

#check distribute_balls_to_bags_correct

end NUMINAMATH_CALUDE_distribute_balls_to_bags_correct_l2782_278297


namespace NUMINAMATH_CALUDE_lemonade_water_amount_l2782_278286

/-- Proves that making 2 gallons of lemonade with a 5:2 water to lemon juice ratio requires 40/7 quarts of water -/
theorem lemonade_water_amount :
  let water_parts : ℚ := 5
  let lemon_juice_parts : ℚ := 2
  let total_parts : ℚ := water_parts + lemon_juice_parts
  let gallons_to_make : ℚ := 2
  let quarts_per_gallon : ℚ := 4
  let total_quarts : ℚ := gallons_to_make * quarts_per_gallon
  water_parts * (total_quarts / total_parts) = 40 / 7 :=
by
  sorry

#check lemonade_water_amount

end NUMINAMATH_CALUDE_lemonade_water_amount_l2782_278286


namespace NUMINAMATH_CALUDE_kennedy_lost_four_pawns_l2782_278213

/-- Represents the number of pawns in a chess game -/
structure ChessGame where
  total_pawns : ℕ
  remaining_pawns : ℕ
  riley_lost_pawns : ℕ

/-- Calculates the number of pawns Kennedy has lost -/
def kennedy_lost_pawns (game : ChessGame) : ℕ :=
  game.total_pawns - game.remaining_pawns - game.riley_lost_pawns

/-- Theorem stating that Kennedy has lost 4 pawns -/
theorem kennedy_lost_four_pawns (game : ChessGame)
  (h1 : game.total_pawns = 16)
  (h2 : game.remaining_pawns = 11)
  (h3 : game.riley_lost_pawns = 1) :
  kennedy_lost_pawns game = 4 := by
  sorry

#eval kennedy_lost_pawns ⟨16, 11, 1⟩

end NUMINAMATH_CALUDE_kennedy_lost_four_pawns_l2782_278213


namespace NUMINAMATH_CALUDE_fourth_root_of_sum_of_cubes_l2782_278294

theorem fourth_root_of_sum_of_cubes : ∃ n : ℕ, n > 0 ∧ n^4 = 5508^3 + 5625^3 + 5742^3 ∧ n = 855 := by
  sorry

end NUMINAMATH_CALUDE_fourth_root_of_sum_of_cubes_l2782_278294


namespace NUMINAMATH_CALUDE_function_identity_l2782_278263

theorem function_identity (f : ℝ → ℝ) :
  (∀ x, f x + 2 * f (3 - x) = x^2) →
  (∀ x, f x = (1/3) * x^2 - 4 * x + 6) :=
by sorry

end NUMINAMATH_CALUDE_function_identity_l2782_278263


namespace NUMINAMATH_CALUDE_integers_between_cubes_l2782_278275

theorem integers_between_cubes : ∃ n : ℕ, n = (⌊(10.3 : ℝ)^3⌋ - ⌈(10.2 : ℝ)^3⌉ + 1) ∧ n = 155 := by
  sorry

end NUMINAMATH_CALUDE_integers_between_cubes_l2782_278275


namespace NUMINAMATH_CALUDE_sisters_phone_sale_total_l2782_278292

def phone_price : ℕ := 400

theorem sisters_phone_sale_total (vivienne_phones aliyah_extra_phones : ℕ) :
  vivienne_phones = 40 →
  aliyah_extra_phones = 10 →
  (vivienne_phones + (vivienne_phones + aliyah_extra_phones)) * phone_price = 36000 :=
by sorry

end NUMINAMATH_CALUDE_sisters_phone_sale_total_l2782_278292


namespace NUMINAMATH_CALUDE_john_weekly_income_l2782_278282

/-- Represents the number of crab baskets John reels in each time he collects crabs -/
def baskets_per_collection : ℕ := 3

/-- Represents the number of crabs each basket holds -/
def crabs_per_basket : ℕ := 4

/-- Represents the number of times John collects crabs per week -/
def collections_per_week : ℕ := 2

/-- Represents the selling price of each crab in dollars -/
def price_per_crab : ℕ := 3

/-- Calculates John's weekly income from selling crabs -/
def weekly_income : ℕ := baskets_per_collection * crabs_per_basket * collections_per_week * price_per_crab

/-- Theorem stating that John's weekly income from selling crabs is $72 -/
theorem john_weekly_income : weekly_income = 72 := by
  sorry

end NUMINAMATH_CALUDE_john_weekly_income_l2782_278282


namespace NUMINAMATH_CALUDE_chickens_egg_laying_rate_l2782_278250

/-- Proves that given the initial conditions, each chicken lays 6 eggs per day. -/
theorem chickens_egg_laying_rate 
  (initial_chickens : ℕ) 
  (growth_factor : ℕ) 
  (weekly_eggs : ℕ) 
  (h1 : initial_chickens = 4)
  (h2 : growth_factor = 8)
  (h3 : weekly_eggs = 1344) : 
  (weekly_eggs / 7) / (initial_chickens * growth_factor) = 6 := by
  sorry

#eval (1344 / 7) / (4 * 8)  -- Should output 6

end NUMINAMATH_CALUDE_chickens_egg_laying_rate_l2782_278250


namespace NUMINAMATH_CALUDE_square_difference_fraction_l2782_278262

theorem square_difference_fraction (x y : ℚ) 
  (sum_eq : x + y = 8/15) 
  (diff_eq : x - y = 1/35) : 
  x^2 - y^2 = 1/75 := by
sorry

end NUMINAMATH_CALUDE_square_difference_fraction_l2782_278262


namespace NUMINAMATH_CALUDE_center_student_coins_l2782_278217

/-- Represents the number of coins each student has in the network -/
structure CoinDistribution :=
  (center : ℕ)
  (first_ring : ℕ)
  (second_ring : ℕ)
  (outer_ring : ℕ)

/-- Defines the conditions of the coin distribution problem -/
def valid_distribution (d : CoinDistribution) : Prop :=
  -- Total number of coins is 3360
  d.center + 5 * d.first_ring + 5 * d.second_ring + 5 * d.outer_ring = 3360 ∧
  -- Center student exchanges with first ring
  d.center = d.first_ring ∧
  -- First ring exchanges with center, second ring, and other first ring students
  d.first_ring = d.center / 5 + d.second_ring / 2 ∧
  -- Second ring exchanges with first ring and outer ring
  d.second_ring = 2 * d.first_ring / 3 + d.outer_ring / 2 ∧
  -- Outer ring exchanges with second ring
  d.outer_ring = d.second_ring

/-- The main theorem stating that the center student must have 280 coins -/
theorem center_student_coins :
  ∀ d : CoinDistribution, valid_distribution d → d.center = 280 :=
by sorry

end NUMINAMATH_CALUDE_center_student_coins_l2782_278217


namespace NUMINAMATH_CALUDE_min_gumballs_for_four_same_color_gumball_problem_solution_l2782_278272

/-- Represents the number of gumballs of each color -/
structure GumballCounts where
  green : Nat
  red : Nat
  white : Nat
  blue : Nat

/-- The minimum number of gumballs needed to ensure getting four of the same color -/
def minGumballsForFourSameColor (counts : GumballCounts) : Nat :=
  13

/-- Theorem stating that for the given gumball counts, 
    the minimum number of gumballs needed to ensure 
    getting four of the same color is 13 -/
theorem min_gumballs_for_four_same_color 
  (counts : GumballCounts) 
  (h1 : counts.green = 12) 
  (h2 : counts.red = 10) 
  (h3 : counts.white = 9) 
  (h4 : counts.blue = 11) : 
  minGumballsForFourSameColor counts = 13 := by
  sorry

/-- Main theorem that proves the result for the specific problem instance -/
theorem gumball_problem_solution : 
  ∃ (counts : GumballCounts), 
    counts.green = 12 ∧ 
    counts.red = 10 ∧ 
    counts.white = 9 ∧ 
    counts.blue = 11 ∧ 
    minGumballsForFourSameColor counts = 13 := by
  sorry

end NUMINAMATH_CALUDE_min_gumballs_for_four_same_color_gumball_problem_solution_l2782_278272


namespace NUMINAMATH_CALUDE_product_mod_seven_l2782_278209

theorem product_mod_seven : (2023 * 2024 * 2025 * 2026) % 7 = 0 := by
  sorry

end NUMINAMATH_CALUDE_product_mod_seven_l2782_278209


namespace NUMINAMATH_CALUDE_temperature_decrease_l2782_278264

theorem temperature_decrease (current_temp : ℝ) (decrease_factor : ℝ) :
  current_temp = 84 →
  decrease_factor = 3/4 →
  current_temp - (decrease_factor * current_temp) = 21 := by
sorry

end NUMINAMATH_CALUDE_temperature_decrease_l2782_278264


namespace NUMINAMATH_CALUDE_triangle_angle_measure_l2782_278277

theorem triangle_angle_measure (D E F : ℝ) (h1 : D + E + F = 180)
  (h2 : F = 3 * E) (h3 : E = 15) : D = 120 := by
  sorry

end NUMINAMATH_CALUDE_triangle_angle_measure_l2782_278277


namespace NUMINAMATH_CALUDE_second_discount_percentage_l2782_278240

theorem second_discount_percentage (initial_price : ℝ) (first_discount : ℝ) (final_price : ℝ) :
  initial_price = 600 →
  first_discount = 10 →
  final_price = 513 →
  ∃ (second_discount : ℝ),
    final_price = initial_price * (1 - first_discount / 100) * (1 - second_discount / 100) ∧
    second_discount = 5 := by
  sorry

end NUMINAMATH_CALUDE_second_discount_percentage_l2782_278240


namespace NUMINAMATH_CALUDE_smallest_number_divisible_by_2022_starting_with_2023_l2782_278201

def starts_with (n m : ℕ) : Prop :=
  ∃ k : ℕ, n = m * 10^k + (n % 10^k) ∧ m * 10^k > n / 10

theorem smallest_number_divisible_by_2022_starting_with_2023 :
  ∀ n : ℕ, (n % 2022 = 0 ∧ starts_with n 2023) → n ≥ 20230110 := by
  sorry

end NUMINAMATH_CALUDE_smallest_number_divisible_by_2022_starting_with_2023_l2782_278201


namespace NUMINAMATH_CALUDE_angle_E_measure_l2782_278254

/-- 
Given a quadrilateral EFGH where ∠E = 2∠F = 4∠G = 5∠H, 
prove that the measure of ∠E is 150°.
-/
theorem angle_E_measure (E F G H : ℝ) : 
  E = 2 * F ∧ E = 4 * G ∧ E = 5 * H ∧ E + F + G + H = 360 → E = 150 := by
  sorry

end NUMINAMATH_CALUDE_angle_E_measure_l2782_278254


namespace NUMINAMATH_CALUDE_min_value_cos_sum_l2782_278293

theorem min_value_cos_sum (x y z : ℝ) (hx : 0 ≤ x ∧ x ≤ π/2) 
  (hy : 0 ≤ y ∧ y ≤ π/2) (hz : 0 ≤ z ∧ z ≤ π/2) :
  Real.cos (x - y) + Real.cos (y - z) + Real.cos (z - x) ≥ 1 := by
  sorry

end NUMINAMATH_CALUDE_min_value_cos_sum_l2782_278293


namespace NUMINAMATH_CALUDE_dilation_and_shift_result_l2782_278228

/-- Represents a complex number -/
structure ComplexNumber where
  re : ℝ
  im : ℝ

/-- Applies a dilation to a complex number -/
def dilate (center : ComplexNumber) (scale : ℝ) (z : ComplexNumber) : ComplexNumber :=
  { re := center.re + scale * (z.re - center.re),
    im := center.im + scale * (z.im - center.im) }

/-- Shifts a complex number by another complex number -/
def shift (z : ComplexNumber) (s : ComplexNumber) : ComplexNumber :=
  { re := z.re - s.re,
    im := z.im - s.im }

/-- The main theorem to be proved -/
theorem dilation_and_shift_result :
  let initial := ComplexNumber.mk 1 (-2)
  let center := ComplexNumber.mk 1 2
  let scale := 2
  let shiftAmount := ComplexNumber.mk 3 4
  let dilated := dilate center scale initial
  let final := shift dilated shiftAmount
  final = ComplexNumber.mk (-2) (-10) := by
  sorry

end NUMINAMATH_CALUDE_dilation_and_shift_result_l2782_278228


namespace NUMINAMATH_CALUDE_power_of_three_mod_seven_l2782_278235

theorem power_of_three_mod_seven : 3^20 % 7 = 2 := by
  sorry

end NUMINAMATH_CALUDE_power_of_three_mod_seven_l2782_278235


namespace NUMINAMATH_CALUDE_chris_money_before_birthday_l2782_278226

def chris_current_money : ℕ := 279
def grandmother_gift : ℕ := 25
def aunt_uncle_gift : ℕ := 20
def parents_gift : ℕ := 75

def total_birthday_gifts : ℕ := grandmother_gift + aunt_uncle_gift + parents_gift

theorem chris_money_before_birthday :
  chris_current_money - total_birthday_gifts = 159 := by
  sorry

end NUMINAMATH_CALUDE_chris_money_before_birthday_l2782_278226


namespace NUMINAMATH_CALUDE_roses_given_l2782_278222

/-- The number of students in the class -/
def total_students : ℕ := 28

/-- The number of different types of flowers -/
def flower_types : ℕ := 3

/-- The relationship between daffodils and roses -/
def rose_daffodil_ratio : ℕ := 4

/-- The relationship between tulips and roses -/
def tulip_rose_ratio : ℕ := 3

/-- The number of boys in the class -/
def num_boys : ℕ := 11

/-- The number of girls in the class -/
def num_girls : ℕ := 17

/-- The number of daffodils given -/
def num_daffodils : ℕ := 11

/-- The number of roses given -/
def num_roses : ℕ := 44

/-- The number of tulips given -/
def num_tulips : ℕ := 132

theorem roses_given :
  num_roses = 44 ∧
  total_students = num_boys + num_girls ∧
  num_roses = rose_daffodil_ratio * num_daffodils ∧
  num_tulips = tulip_rose_ratio * num_roses ∧
  num_boys * num_girls = num_daffodils + num_roses + num_tulips :=
by sorry

end NUMINAMATH_CALUDE_roses_given_l2782_278222


namespace NUMINAMATH_CALUDE_fixed_point_on_line_l2782_278252

/-- The line equation mx+(1-m)y+m-2=0 always passes through the point (1,2) for all real m. -/
theorem fixed_point_on_line (m : ℝ) : m * 1 + (1 - m) * 2 + m - 2 = 0 := by
  sorry

end NUMINAMATH_CALUDE_fixed_point_on_line_l2782_278252


namespace NUMINAMATH_CALUDE_ellipse_and_slopes_l2782_278270

/-- Definition of the ellipse C -/
def ellipse_C (x y a b : ℝ) : Prop :=
  x^2 / a^2 + y^2 / b^2 = 1 ∧ a > b ∧ b > 0

/-- Definition of the foci F1 and F2 -/
def foci (F1 F2 : ℝ × ℝ) (a b : ℝ) : Prop :=
  F1.1 < 0 ∧ F2.1 > 0 ∧ F1.2 = 0 ∧ F2.2 = 0 ∧ F1.1^2 = F2.1^2 ∧ F2.1^2 = a^2 - b^2

/-- Definition of the circles intersecting on C -/
def circles_intersect_on_C (F1 F2 : ℝ × ℝ) (a b : ℝ) : Prop :=
  ∃ P : ℝ × ℝ, ellipse_C P.1 P.2 a b ∧ 
    (P.1 - F1.1)^2 + (P.2 - F1.2)^2 = 9 ∧
    (P.1 - F2.1)^2 + (P.2 - F2.2)^2 = 1

/-- Definition of point A -/
def point_A (a b : ℝ) : ℝ × ℝ := (0, b)

/-- Definition of angle F1AF2 -/
def angle_F1AF2 (F1 F2 : ℝ × ℝ) (a b : ℝ) : Prop :=
  let A := point_A a b
  Real.cos (2 * Real.pi / 3) = 
    ((F1.1 - A.1) * (F2.1 - A.1) + (F1.2 - A.2) * (F2.2 - A.2)) /
    (Real.sqrt ((F1.1 - A.1)^2 + (F1.2 - A.2)^2) * Real.sqrt ((F2.1 - A.1)^2 + (F2.2 - A.2)^2))

/-- Definition of line l -/
def line_l (k : ℝ) (x y : ℝ) : Prop :=
  y + 1 = k * (x - 2)

/-- Definition of points M and N -/
def points_M_N (a b k : ℝ) : Prop :=
  ∃ M N : ℝ × ℝ, ellipse_C M.1 M.2 a b ∧ ellipse_C N.1 N.2 a b ∧
    line_l k M.1 M.2 ∧ line_l k N.1 N.2 ∧ M ≠ N

/-- Main theorem -/
theorem ellipse_and_slopes (a b : ℝ) (F1 F2 : ℝ × ℝ) (k : ℝ) :
  ellipse_C 0 b a b →
  foci F1 F2 a b →
  circles_intersect_on_C F1 F2 a b →
  angle_F1AF2 F1 F2 a b →
  points_M_N a b k →
  (a = 2 ∧ b = 1) ∧
  (∃ k1 k2 : ℝ, k1 + k2 = -1) :=
sorry

end NUMINAMATH_CALUDE_ellipse_and_slopes_l2782_278270


namespace NUMINAMATH_CALUDE_fishing_loss_fraction_l2782_278227

theorem fishing_loss_fraction (jordan_catch : ℕ) (perry_catch : ℕ) (remaining : ℕ) : 
  jordan_catch = 4 →
  perry_catch = 2 * jordan_catch →
  remaining = 9 →
  (jordan_catch + perry_catch - remaining : ℚ) / (jordan_catch + perry_catch) = 1/4 :=
by sorry

end NUMINAMATH_CALUDE_fishing_loss_fraction_l2782_278227


namespace NUMINAMATH_CALUDE_log_order_l2782_278296

theorem log_order (a b c : ℝ) (ha : a = Real.log 3 / Real.log 4)
  (hb : b = Real.log 4 / Real.log 3) (hc : c = Real.log (4/3) / Real.log (3/4)) :
  b > a ∧ a > c :=
by sorry

end NUMINAMATH_CALUDE_log_order_l2782_278296


namespace NUMINAMATH_CALUDE_fifth_row_solution_l2782_278218

/-- Represents the possible values in the grid -/
inductive GridValue
  | Two
  | Zero
  | One
  | Five
  | Blank

/-- Represents a 5x5 grid -/
def Grid := Fin 5 → Fin 5 → GridValue

/-- Check if a grid satisfies the row constraint -/
def satisfiesRowConstraint (g : Grid) : Prop :=
  ∀ row, ∃! i, g row i = GridValue.Two ∧
         ∃! i, g row i = GridValue.Zero ∧
         ∃! i, g row i = GridValue.One ∧
         ∃! i, g row i = GridValue.Five

/-- Check if a grid satisfies the column constraint -/
def satisfiesColumnConstraint (g : Grid) : Prop :=
  ∀ col, ∃! i, g i col = GridValue.Two ∧
         ∃! i, g i col = GridValue.Zero ∧
         ∃! i, g i col = GridValue.One ∧
         ∃! i, g i col = GridValue.Five

/-- Check if a grid satisfies the diagonal constraint -/
def satisfiesDiagonalConstraint (g : Grid) : Prop :=
  ∀ i j, i < 4 → j < 4 →
    (g i j ≠ GridValue.Blank → g (i+1) (j+1) ≠ g i j) ∧
    (g i (j+1) ≠ GridValue.Blank → g (i+1) j ≠ g i (j+1))

/-- The main theorem stating the solution for the fifth row -/
theorem fifth_row_solution (g : Grid) 
  (hrow : satisfiesRowConstraint g)
  (hcol : satisfiesColumnConstraint g)
  (hdiag : satisfiesDiagonalConstraint g) :
  g 4 0 = GridValue.One ∧
  g 4 1 = GridValue.Five ∧
  g 4 2 = GridValue.Blank ∧
  g 4 3 = GridValue.Blank ∧
  g 4 4 = GridValue.Two :=
sorry

end NUMINAMATH_CALUDE_fifth_row_solution_l2782_278218


namespace NUMINAMATH_CALUDE_tax_reduction_theorem_l2782_278239

theorem tax_reduction_theorem (T C : ℝ) (x : ℝ) 
  (h1 : T > 0) (h2 : C > 0) 
  (h3 : (T * (1 - x / 100)) * (C * 1.1) = T * C * 0.88) : x = 20 := by
  sorry

end NUMINAMATH_CALUDE_tax_reduction_theorem_l2782_278239


namespace NUMINAMATH_CALUDE_june_found_seventeen_eggs_l2782_278243

/-- The number of eggs June found -/
def total_eggs : ℕ :=
  let nest1_eggs := 2 * 5  -- 2 nests with 5 eggs each in 1 tree
  let nest2_eggs := 1 * 3  -- 1 nest with 3 eggs in another tree
  let nest3_eggs := 1 * 4  -- 1 nest with 4 eggs in the front yard
  nest1_eggs + nest2_eggs + nest3_eggs

/-- Theorem stating that June found 17 eggs in total -/
theorem june_found_seventeen_eggs : total_eggs = 17 := by
  sorry

end NUMINAMATH_CALUDE_june_found_seventeen_eggs_l2782_278243


namespace NUMINAMATH_CALUDE_sum_external_angles_hexagon_l2782_278257

/-- A regular hexagon is a polygon with 6 sides of equal length and 6 equal angles -/
def RegularHexagon : Type := Unit

/-- The external angle of a polygon is the angle between one side and the extension of an adjacent side -/
def ExternalAngle (p : RegularHexagon) : ℝ := sorry

/-- The sum of external angles of a regular hexagon -/
def SumExternalAngles (p : RegularHexagon) : ℝ := sorry

/-- Theorem: The sum of the external angles of a regular hexagon is 360° -/
theorem sum_external_angles_hexagon (p : RegularHexagon) :
  SumExternalAngles p = 360 := by sorry

end NUMINAMATH_CALUDE_sum_external_angles_hexagon_l2782_278257


namespace NUMINAMATH_CALUDE_no_solution_implies_a_leq_8_l2782_278291

theorem no_solution_implies_a_leq_8 (a : ℝ) :
  (∀ x : ℝ, ¬(|x - 5| + |x + 3| < a)) → a ≤ 8 := by
  sorry

end NUMINAMATH_CALUDE_no_solution_implies_a_leq_8_l2782_278291


namespace NUMINAMATH_CALUDE_binomial_8_3_l2782_278203

theorem binomial_8_3 : Nat.choose 8 3 = 56 := by
  sorry

end NUMINAMATH_CALUDE_binomial_8_3_l2782_278203


namespace NUMINAMATH_CALUDE_total_berets_is_eleven_l2782_278256

def spools_per_beret : ℕ := 3

def red_spools : ℕ := 12
def black_spools : ℕ := 15
def blue_spools : ℕ := 6

def berets_from_spools (spools : ℕ) : ℕ := spools / spools_per_beret

theorem total_berets_is_eleven :
  berets_from_spools red_spools + berets_from_spools black_spools + berets_from_spools blue_spools = 11 := by
  sorry

end NUMINAMATH_CALUDE_total_berets_is_eleven_l2782_278256


namespace NUMINAMATH_CALUDE_pull_up_median_mode_l2782_278241

def pull_up_data : List ℕ := [6, 8, 7, 7, 8, 9, 8, 9]

def median (l : List ℕ) : ℚ := sorry

def mode (l : List ℕ) : ℕ := sorry

theorem pull_up_median_mode :
  median pull_up_data = 8 ∧ mode pull_up_data = 8 := by sorry

end NUMINAMATH_CALUDE_pull_up_median_mode_l2782_278241


namespace NUMINAMATH_CALUDE_f_properties_g_property_l2782_278298

noncomputable def f (ω : ℝ) (x : ℝ) : ℝ := 2 * Real.sqrt 3 * Real.sin (ω * x / 2) * Real.cos (ω * x / 2) + 6 * Real.cos (ω * x / 2)^2 - 3

def is_even (f : ℝ → ℝ) : Prop := ∀ x, f x = f (-x)

def has_period (f : ℝ → ℝ) (p : ℝ) : Prop := ∀ x, f (x + p) = f x

def is_increasing_on (f : ℝ → ℝ) (a b : ℝ) : Prop := ∀ x y, a < x ∧ x < y ∧ y < b → f x < f y

theorem f_properties (ω : ℝ) (θ : ℝ) (h_ω : ω > 0) (h_θ : 0 < θ ∧ θ < Real.pi / 2) :
  (is_even (fun x ↦ f ω (x + θ)) ∧ 
   has_period (fun x ↦ f ω (x + θ)) Real.pi ∧
   ∀ p, has_period (fun x ↦ f ω (x + θ)) p → p ≥ Real.pi) →
  ω = 2 ∧ θ = Real.pi / 12 :=
sorry

theorem g_property (ω : ℝ) (h_ω : ω > 0) :
  is_increasing_on (fun x ↦ f ω (3 * x)) 0 (Real.pi / 3) →
  ω ≤ 1 / 6 :=
sorry

end NUMINAMATH_CALUDE_f_properties_g_property_l2782_278298


namespace NUMINAMATH_CALUDE_integer_pair_solution_l2782_278242

theorem integer_pair_solution :
  ∀ x y : ℕ+,
  (2 * x.val * y.val = 2 * x.val + y.val + 21) →
  ((x.val = 1 ∧ y.val = 23) ∨ (x.val = 6 ∧ y.val = 3)) :=
by
  sorry

end NUMINAMATH_CALUDE_integer_pair_solution_l2782_278242


namespace NUMINAMATH_CALUDE_real_solutions_condition_l2782_278247

theorem real_solutions_condition (a : ℝ) :
  (∃ x y : ℝ, x + y^2 = a ∧ y + x^2 = a) ↔ a ≥ 3/4 := by
  sorry

end NUMINAMATH_CALUDE_real_solutions_condition_l2782_278247


namespace NUMINAMATH_CALUDE_triangle_side_length_l2782_278253

theorem triangle_side_length (A B C : ℝ) (a b c : ℝ) :
  (0 < A) ∧ (A < π) ∧ (0 < B) ∧ (B < π) ∧ (0 < C) ∧ (C < π) →
  (A + B + C = π) →
  (a + b + 10 * c = 2 * (Real.sin A + Real.sin B + 10 * Real.sin C)) →
  (A = π / 3) →
  (a = Real.sin A / Real.sin B * b) →
  (a = Real.sin A / Real.sin C * c) →
  a = Real.sqrt 3 := by
sorry

end NUMINAMATH_CALUDE_triangle_side_length_l2782_278253


namespace NUMINAMATH_CALUDE_cost_price_percentage_l2782_278219

/-- Given a selling price and a profit percentage, calculates the cost price as a percentage of the selling price. -/
theorem cost_price_percentage (selling_price : ℝ) (profit_percentage : ℝ) :
  profit_percentage = 4.166666666666666 →
  (selling_price - (selling_price * profit_percentage / 100)) / selling_price * 100 = 95.83333333333334 :=
by
  sorry

end NUMINAMATH_CALUDE_cost_price_percentage_l2782_278219


namespace NUMINAMATH_CALUDE_non_tax_paying_percentage_is_six_percent_l2782_278208

/-- The number of customers shopping per day -/
def customers_per_day : ℕ := 1000

/-- The number of days in a week -/
def days_per_week : ℕ := 7

/-- The number of customers who pay taxes per week -/
def tax_paying_customers_per_week : ℕ := 6580

/-- The percentage of customers who do not pay tax -/
def non_tax_paying_percentage : ℚ :=
  (customers_per_day * days_per_week - tax_paying_customers_per_week : ℚ) /
  (customers_per_day * days_per_week : ℚ) * 100

theorem non_tax_paying_percentage_is_six_percent :
  non_tax_paying_percentage = 6 := by
  sorry

end NUMINAMATH_CALUDE_non_tax_paying_percentage_is_six_percent_l2782_278208


namespace NUMINAMATH_CALUDE_trampoline_jumps_l2782_278248

/-- The number of times Ronald jumped on the trampoline -/
def ronald_jumps : ℕ := 157

/-- The additional number of times Rupert jumped compared to Ronald -/
def rupert_additional_jumps : ℕ := 86

/-- The number of times Rupert jumped on the trampoline -/
def rupert_jumps : ℕ := ronald_jumps + rupert_additional_jumps

/-- The average number of jumps between the two brothers -/
def average_jumps : ℕ := (ronald_jumps + rupert_jumps) / 2

/-- The total number of jumps made by both Rupert and Ronald -/
def total_jumps : ℕ := ronald_jumps + rupert_jumps

theorem trampoline_jumps :
  average_jumps = 200 ∧ total_jumps = 400 := by
  sorry

end NUMINAMATH_CALUDE_trampoline_jumps_l2782_278248


namespace NUMINAMATH_CALUDE_jasmine_swimming_totals_l2782_278207

/-- Jasmine's weekly swimming routine -/
structure SwimmingRoutine where
  monday_laps : ℕ
  tuesday_laps : ℕ
  tuesday_aerobics : ℕ
  wednesday_laps : ℕ
  wednesday_time_per_lap : ℕ
  thursday_laps : ℕ
  friday_laps : ℕ

/-- Calculate total laps and partial time for a given number of weeks -/
def calculate_totals (routine : SwimmingRoutine) (weeks : ℕ) :
  (ℕ × ℕ) :=
  let weekly_laps := routine.monday_laps + routine.tuesday_laps +
                     routine.wednesday_laps + routine.thursday_laps +
                     routine.friday_laps
  let weekly_partial_time := routine.tuesday_aerobics +
                             (routine.wednesday_laps * routine.wednesday_time_per_lap)
  (weekly_laps * weeks, weekly_partial_time * weeks)

theorem jasmine_swimming_totals :
  let routine := SwimmingRoutine.mk 10 15 20 12 2 18 20
  let (total_laps, partial_time) := calculate_totals routine 5
  total_laps = 375 ∧ partial_time = 220 := by sorry

end NUMINAMATH_CALUDE_jasmine_swimming_totals_l2782_278207


namespace NUMINAMATH_CALUDE_partial_fraction_decomposition_l2782_278285

theorem partial_fraction_decomposition :
  ∃! (P Q R : ℚ), ∀ x : ℚ, x ≠ 1 ∧ x ≠ 4 ∧ x ≠ 6 →
    (x^2 + 2) / ((x - 1) * (x - 4) * (x - 6)) = 
    P / (x - 1) + Q / (x - 4) + R / (x - 6) ∧
    P = 1/5 ∧ Q = -3 ∧ R = 19/5 := by
  sorry

end NUMINAMATH_CALUDE_partial_fraction_decomposition_l2782_278285


namespace NUMINAMATH_CALUDE_solve_for_a_l2782_278220

def U (a : ℝ) : Set ℝ := {1, 2, a^2 + 2*a - 3}

def A (a : ℝ) : Set ℝ := {|a - 2|, 2}

theorem solve_for_a (a : ℝ) :
  U a = {1, 2, 0} ∧
  A a ∪ {0} = U a →
  a = 1 :=
sorry

end NUMINAMATH_CALUDE_solve_for_a_l2782_278220


namespace NUMINAMATH_CALUDE_sine_function_midpoint_l2782_278231

/-- Given a sine function y = A sin(Bx + C) + D that oscillates between 6 and 2, prove that D = 4 -/
theorem sine_function_midpoint (A B C D : ℝ) 
  (h_oscillation : ∀ x, 2 ≤ A * Real.sin (B * x + C) + D ∧ A * Real.sin (B * x + C) + D ≤ 6) : 
  D = 4 := by
  sorry

end NUMINAMATH_CALUDE_sine_function_midpoint_l2782_278231


namespace NUMINAMATH_CALUDE_consecutive_integers_not_representable_l2782_278212

theorem consecutive_integers_not_representable : ∀ k : ℤ, 12 ≤ k → k ≤ 19 → 
  ∀ x y : ℤ, |7 * x^2 + 9 * x * y - 5 * y^2| ≠ k := by
  sorry

end NUMINAMATH_CALUDE_consecutive_integers_not_representable_l2782_278212


namespace NUMINAMATH_CALUDE_exists_negative_fraction_abs_lt_four_l2782_278233

theorem exists_negative_fraction_abs_lt_four : ∃ (a b : ℤ), b ≠ 0 ∧ (a / b : ℚ) < 0 ∧ |a / b| < 4 := by
  sorry

end NUMINAMATH_CALUDE_exists_negative_fraction_abs_lt_four_l2782_278233


namespace NUMINAMATH_CALUDE_perpendicular_lines_sin_2alpha_l2782_278279

theorem perpendicular_lines_sin_2alpha (α : Real) :
  let l₁ : Real → Real → Real := λ x y => x * Real.sin α + y - 1
  let l₂ : Real → Real → Real := λ x y => x - 3 * y * Real.cos α + 1
  (∀ x y, l₁ x y = 0 → l₂ x y = 0 → (Real.sin α + 3 * Real.cos α) * (Real.sin α - 3 * Real.cos α) = 0) →
  Real.sin (2 * α) = 3 / 5 := by
sorry

end NUMINAMATH_CALUDE_perpendicular_lines_sin_2alpha_l2782_278279


namespace NUMINAMATH_CALUDE_excluded_students_count_l2782_278236

theorem excluded_students_count (total_students : ℕ) (initial_avg : ℚ) 
  (excluded_avg : ℚ) (new_avg : ℚ) (h1 : total_students = 10) 
  (h2 : initial_avg = 80) (h3 : excluded_avg = 70) (h4 : new_avg = 90) :
  ∃ (excluded : ℕ), 
    excluded = 5 ∧ 
    (initial_avg * total_students : ℚ) = 
      excluded_avg * excluded + new_avg * (total_students - excluded) :=
by sorry

end NUMINAMATH_CALUDE_excluded_students_count_l2782_278236


namespace NUMINAMATH_CALUDE_circle_area_from_circumference_l2782_278249

/-- Given a circle with circumference 87.98229536926875 cm, its area is approximately 615.75164 square centimeters. -/
theorem circle_area_from_circumference : 
  let circumference : ℝ := 87.98229536926875
  let radius : ℝ := circumference / (2 * Real.pi)
  let area : ℝ := Real.pi * radius ^ 2
  ∃ ε > 0, abs (area - 615.75164) < ε :=
by
  sorry

end NUMINAMATH_CALUDE_circle_area_from_circumference_l2782_278249


namespace NUMINAMATH_CALUDE_bus_ride_cost_l2782_278202

theorem bus_ride_cost (bus_cost train_cost : ℝ) : 
  train_cost = bus_cost + 6.85 →
  bus_cost + train_cost = 9.65 →
  bus_cost = 1.40 := by
sorry

end NUMINAMATH_CALUDE_bus_ride_cost_l2782_278202


namespace NUMINAMATH_CALUDE_tan_negative_405_degrees_l2782_278281

theorem tan_negative_405_degrees : Real.tan ((-405 : ℝ) * π / 180) = -1 := by sorry

end NUMINAMATH_CALUDE_tan_negative_405_degrees_l2782_278281


namespace NUMINAMATH_CALUDE_square_diagonal_perimeter_l2782_278259

theorem square_diagonal_perimeter (d : ℝ) (s : ℝ) (P : ℝ) :
  d = 2 * Real.sqrt 2 →  -- diagonal length
  d = s * Real.sqrt 2 →  -- relation between diagonal and side length
  P = 4 * s →           -- perimeter definition
  P = 8 := by
sorry

end NUMINAMATH_CALUDE_square_diagonal_perimeter_l2782_278259


namespace NUMINAMATH_CALUDE_odd_function_property_l2782_278260

def is_odd_function (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = -f x

def has_period_4 (f : ℝ → ℝ) : Prop :=
  ∀ x, f (x + 4) = f x

theorem odd_function_property (f : ℝ → ℝ) 
  (h_odd : is_odd_function f)
  (h_recursive : ∀ x, f (x + 4) = f x + 3 * f 2)
  (h_f_1 : f 1 = 1) :
  f 2015 + f 2016 = -1 := by
sorry

end NUMINAMATH_CALUDE_odd_function_property_l2782_278260


namespace NUMINAMATH_CALUDE_product_of_integers_with_lcm_and_gcd_l2782_278204

theorem product_of_integers_with_lcm_and_gcd (a b : ℕ+) : 
  Nat.lcm a b = 72 → 
  Nat.gcd a b = 8 → 
  (a = 4 * Nat.gcd a b ∨ b = 4 * Nat.gcd a b) → 
  a * b = 576 := by
sorry

end NUMINAMATH_CALUDE_product_of_integers_with_lcm_and_gcd_l2782_278204


namespace NUMINAMATH_CALUDE_vegetable_baskets_l2782_278283

/-- Calculates the number of baskets needed to store vegetables --/
theorem vegetable_baskets
  (keith_turnips : ℕ)
  (alyssa_turnips : ℕ)
  (sean_carrots : ℕ)
  (turnips_per_basket : ℕ)
  (carrots_per_basket : ℕ)
  (h1 : keith_turnips = 6)
  (h2 : alyssa_turnips = 9)
  (h3 : sean_carrots = 5)
  (h4 : turnips_per_basket = 5)
  (h5 : carrots_per_basket = 4) :
  (((keith_turnips + alyssa_turnips) + turnips_per_basket - 1) / turnips_per_basket) +
  ((sean_carrots + carrots_per_basket - 1) / carrots_per_basket) = 5 :=
by sorry

end NUMINAMATH_CALUDE_vegetable_baskets_l2782_278283


namespace NUMINAMATH_CALUDE_at_least_one_acute_angle_leq_45_l2782_278210

-- Define a right triangle
structure RightTriangle where
  angle1 : Real
  angle2 : Real
  angle3 : Real
  is_right_triangle : angle1 + angle2 + angle3 = 180
  has_right_angle : angle3 = 90

-- Theorem statement
theorem at_least_one_acute_angle_leq_45 (t : RightTriangle) :
  t.angle1 ≤ 45 ∨ t.angle2 ≤ 45 :=
sorry

end NUMINAMATH_CALUDE_at_least_one_acute_angle_leq_45_l2782_278210


namespace NUMINAMATH_CALUDE_employee_payment_l2782_278229

theorem employee_payment (total : ℝ) (x y : ℝ) (h1 : total = 572) (h2 : x = 1.2 * y) (h3 : total = x + y) : y = 260 :=
by
  sorry

end NUMINAMATH_CALUDE_employee_payment_l2782_278229


namespace NUMINAMATH_CALUDE_max_distinct_squares_sum_2100_l2782_278274

/-- The sum of squares of a list of natural numbers -/
def sum_of_squares (lst : List Nat) : Nat :=
  lst.map (· ^ 2) |>.sum

/-- A proposition stating that a list of natural numbers has distinct elements -/
def is_distinct (lst : List Nat) : Prop :=
  lst.Nodup

theorem max_distinct_squares_sum_2100 :
  (∃ (n : Nat) (lst : List Nat), 
    lst.length = n ∧ 
    is_distinct lst ∧ 
    sum_of_squares lst = 2100 ∧
    ∀ (m : Nat) (lst' : List Nat), 
      lst'.length = m ∧ 
      is_distinct lst' ∧ 
      sum_of_squares lst' = 2100 → 
      m ≤ n) ∧
  (∃ (lst : List Nat), 
    lst.length = 17 ∧ 
    is_distinct lst ∧ 
    sum_of_squares lst = 2100) :=
by
  sorry


end NUMINAMATH_CALUDE_max_distinct_squares_sum_2100_l2782_278274


namespace NUMINAMATH_CALUDE_expression_simplification_l2782_278225

theorem expression_simplification (x : ℝ) (h : x = Real.sqrt 2 - 1) :
  (x + 2) / (x^2 - 2*x) / ((8*x) / (x - 2) + x - 2) = 1 := by
  sorry

end NUMINAMATH_CALUDE_expression_simplification_l2782_278225


namespace NUMINAMATH_CALUDE_smallest_x_for_equation_l2782_278258

theorem smallest_x_for_equation : 
  ∃ (x : ℝ), x > 0 ∧ 
  (⌊x^2⌋ : ℝ) - x * (⌊x⌋ : ℝ) = 10 ∧ 
  (∀ y : ℝ, y > 0 ∧ (⌊y^2⌋ : ℝ) - y * (⌊y⌋ : ℝ) = 10 → y ≥ x) ∧
  x = 131 / 11 := by
sorry

end NUMINAMATH_CALUDE_smallest_x_for_equation_l2782_278258


namespace NUMINAMATH_CALUDE_isosceles_triangle_exists_l2782_278237

/-- A regular polygon with n sides -/
structure RegularPolygon (n : ℕ) where
  vertices : Fin n → ℝ × ℝ

/-- A coloring of the vertices of a polygon with two colors -/
def Coloring (n : ℕ) := Fin n → Bool

/-- An isosceles triangle in a regular polygon -/
def IsIsoscelesTriangle (p : RegularPolygon n) (v1 v2 v3 : Fin n) : Prop :=
  let d12 := dist (p.vertices v1) (p.vertices v2)
  let d23 := dist (p.vertices v2) (p.vertices v3)
  let d31 := dist (p.vertices v3) (p.vertices v1)
  d12 = d23 ∨ d23 = d31 ∨ d31 = d12

/-- The main theorem -/
theorem isosceles_triangle_exists (p : RegularPolygon 13) (c : Coloring 13) :
  ∃ (v1 v2 v3 : Fin 13), c v1 = c v2 ∧ c v2 = c v3 ∧ IsIsoscelesTriangle p v1 v2 v3 := by
  sorry


end NUMINAMATH_CALUDE_isosceles_triangle_exists_l2782_278237


namespace NUMINAMATH_CALUDE_inequality_proof_l2782_278289

theorem inequality_proof (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (h_abc : a * b * c = 1) :
  (1 / (a^3 * (b + c))) + (1 / (b^3 * (c + a))) + (1 / (c^3 * (a + b))) ≥ 3/2 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l2782_278289


namespace NUMINAMATH_CALUDE_expansion_term_count_l2782_278206

/-- The number of terms in the expansion of a product of sums of distinct variables -/
def expansion_terms (x y z : ℕ) : ℕ := x * y * z

/-- The first factor (a+b+c) has 3 terms -/
def factor1_terms : ℕ := 3

/-- The second factor (d+e+f+g) has 4 terms -/
def factor2_terms : ℕ := 4

/-- The third factor (h+i) has 2 terms -/
def factor3_terms : ℕ := 2

theorem expansion_term_count : 
  expansion_terms factor1_terms factor2_terms factor3_terms = 24 := by
  sorry

end NUMINAMATH_CALUDE_expansion_term_count_l2782_278206


namespace NUMINAMATH_CALUDE_five_fridays_in_august_l2782_278266

/-- Represents days of the week -/
inductive DayOfWeek
  | Sunday
  | Monday
  | Tuesday
  | Wednesday
  | Thursday
  | Friday
  | Saturday

/-- Returns the next day of the week -/
def nextDay (d : DayOfWeek) : DayOfWeek :=
  match d with
  | DayOfWeek.Sunday => DayOfWeek.Monday
  | DayOfWeek.Monday => DayOfWeek.Tuesday
  | DayOfWeek.Tuesday => DayOfWeek.Wednesday
  | DayOfWeek.Wednesday => DayOfWeek.Thursday
  | DayOfWeek.Thursday => DayOfWeek.Friday
  | DayOfWeek.Friday => DayOfWeek.Saturday
  | DayOfWeek.Saturday => DayOfWeek.Sunday

/-- Represents a calendar month -/
structure Month where
  days : Nat
  firstDay : DayOfWeek

/-- Returns the number of occurrences of a given day in a month -/
def countDayOccurrences (m : Month) (d : DayOfWeek) : Nat :=
  sorry

/-- Theorem: If July has five Tuesdays, then August has five Fridays -/
theorem five_fridays_in_august 
  (july : Month) 
  (h1 : july.days = 31)
  (h2 : countDayOccurrences july DayOfWeek.Tuesday = 5) :
  ∃ (august : Month), 
    august.days = 31 ∧ 
    august.firstDay = nextDay (nextDay (nextDay july.firstDay)) ∧
    countDayOccurrences august DayOfWeek.Friday = 5 :=
  sorry

end NUMINAMATH_CALUDE_five_fridays_in_august_l2782_278266


namespace NUMINAMATH_CALUDE_pie_chart_angle_l2782_278244

theorem pie_chart_angle (percentage : ℝ) (angle : ℝ) :
  percentage = 0.15 →
  angle = percentage * 360 →
  angle = 54 := by
  sorry

end NUMINAMATH_CALUDE_pie_chart_angle_l2782_278244


namespace NUMINAMATH_CALUDE_hyperbola_foci_l2782_278295

/-- The hyperbola equation -/
def hyperbola_equation (x y : ℝ) : Prop :=
  x^2 / 16 - y^2 / 9 = 1

/-- The foci of the hyperbola -/
def foci : Set (ℝ × ℝ) :=
  {(-5, 0), (5, 0)}

/-- Theorem: The given foci are the correct foci of the hyperbola -/
theorem hyperbola_foci :
  ∀ (x y : ℝ), hyperbola_equation x y →
  ∃ (f : ℝ × ℝ), f ∈ foci ∧
  (x - f.1)^2 + y^2 = (x + f.1)^2 + y^2 :=
sorry

end NUMINAMATH_CALUDE_hyperbola_foci_l2782_278295


namespace NUMINAMATH_CALUDE_arc_length_from_central_angle_l2782_278205

/-- Given a circle with circumference 80 feet and an arc subtended by a central angle of 120°,
    the length of this arc is 80/3 feet. -/
theorem arc_length_from_central_angle (circle : Real) (arc : Real) :
  (circle = 80) →  -- circumference of the circle is 80 feet
  (arc = 120 / 360 * circle) →  -- arc is subtended by a 120° angle
  (arc = 80 / 3) :=  -- length of the arc is 80/3 feet
by sorry

end NUMINAMATH_CALUDE_arc_length_from_central_angle_l2782_278205


namespace NUMINAMATH_CALUDE_computer_cost_computer_cost_proof_l2782_278234

theorem computer_cost (accessories_cost : ℕ) (playstation_worth : ℕ) (discount_percent : ℕ) (out_of_pocket : ℕ) : ℕ :=
  let playstation_sold := playstation_worth - (playstation_worth * discount_percent / 100)
  let total_paid := playstation_sold + out_of_pocket
  total_paid - accessories_cost

#check computer_cost 200 400 20 580 = 700

theorem computer_cost_proof :
  computer_cost 200 400 20 580 = 700 := by
  sorry

end NUMINAMATH_CALUDE_computer_cost_computer_cost_proof_l2782_278234


namespace NUMINAMATH_CALUDE_maintenance_scheduling_methods_l2782_278200

/-- Represents the days of the week --/
inductive Day
  | Monday
  | Tuesday
  | Wednesday
  | Thursday
  | Friday

/-- Represents the monitoring points --/
inductive MonitoringPoint
  | A
  | B
  | C
  | D
  | E
  | F
  | G
  | H

/-- A schedule is a function from MonitoringPoint to Day --/
def Schedule := MonitoringPoint → Day

/-- Checks if a schedule is valid according to the given conditions --/
def isValidSchedule (s : Schedule) : Prop :=
  (s MonitoringPoint.A = Day.Monday) ∧
  (s MonitoringPoint.B = Day.Tuesday) ∧
  (s MonitoringPoint.C = s MonitoringPoint.D) ∧
  (s MonitoringPoint.D = s MonitoringPoint.E) ∧
  (s MonitoringPoint.F ≠ Day.Friday) ∧
  (∀ d : Day, ∃ p : MonitoringPoint, s p = d)

/-- The total number of valid schedules --/
def totalValidSchedules : ℕ := sorry

theorem maintenance_scheduling_methods :
  totalValidSchedules = 60 := by sorry

end NUMINAMATH_CALUDE_maintenance_scheduling_methods_l2782_278200


namespace NUMINAMATH_CALUDE_can_measure_15_minutes_l2782_278216

/-- Represents an hourglass with a given duration in minutes -/
structure Hourglass where
  duration : ℕ

/-- Represents the state of the timing system -/
structure TimingSystem where
  hourglass1 : Hourglass
  hourglass2 : Hourglass

/-- Defines the initial state of the timing system -/
def initialState : TimingSystem :=
  { hourglass1 := { duration := 7 },
    hourglass2 := { duration := 11 } }

/-- Represents a sequence of operations on the hourglasses -/
inductive Operation
  | FlipHourglass1
  | FlipHourglass2
  | Wait (minutes : ℕ)

/-- Applies a sequence of operations to the timing system -/
def applyOperations (state : TimingSystem) (ops : List Operation) : ℕ :=
  sorry

/-- Theorem stating that 15 minutes can be measured using the given hourglasses -/
theorem can_measure_15_minutes :
  ∃ (ops : List Operation), applyOperations initialState ops = 15 :=
sorry

end NUMINAMATH_CALUDE_can_measure_15_minutes_l2782_278216


namespace NUMINAMATH_CALUDE_marble_leftover_l2782_278284

theorem marble_leftover (r p g : ℤ) 
  (hr : r % 6 = 5)
  (hp : p % 6 = 2)
  (hg : g % 6 = 3) :
  (r + p + g) % 6 = 4 := by
sorry

end NUMINAMATH_CALUDE_marble_leftover_l2782_278284


namespace NUMINAMATH_CALUDE_greatest_two_digit_prime_saturated_is_98_l2782_278255

/-- A number is prime saturated if the product of all its different positive prime factors
    is less than its square root -/
def IsPrimeSaturated (n : ℕ) : Prop :=
  (Finset.prod (Nat.factors n).toFinset id) < Real.sqrt (n : ℝ)

/-- The greatest two-digit prime saturated integer -/
def GreatestTwoDigitPrimeSaturated : ℕ := 98

theorem greatest_two_digit_prime_saturated_is_98 :
  IsPrimeSaturated GreatestTwoDigitPrimeSaturated ∧
  ∀ n : ℕ, 10 ≤ n ∧ n ≤ 99 ∧ IsPrimeSaturated n → n ≤ GreatestTwoDigitPrimeSaturated :=
by
  sorry

end NUMINAMATH_CALUDE_greatest_two_digit_prime_saturated_is_98_l2782_278255


namespace NUMINAMATH_CALUDE_existence_of_special_divisibility_pair_l2782_278271

theorem existence_of_special_divisibility_pair : 
  ∃ (a b : ℕ+), 
    a ∣ b^2 ∧ 
    b^2 ∣ a^3 ∧ 
    a^3 ∣ b^4 ∧ 
    b^4 ∣ a^5 ∧ 
    ¬(a^5 ∣ b^6) := by
  sorry

end NUMINAMATH_CALUDE_existence_of_special_divisibility_pair_l2782_278271


namespace NUMINAMATH_CALUDE_prob_sum_greater_than_8_is_5_18_l2782_278215

/-- The number of possible outcomes when rolling two dice -/
def total_outcomes : ℕ := 36

/-- The number of ways to get a sum of 8 or less when rolling two dice -/
def sum_8_or_less : ℕ := 26

/-- The probability of rolling two dice and getting a sum greater than eight -/
def prob_sum_greater_than_8 : ℚ :=
  1 - (sum_8_or_less : ℚ) / total_outcomes

theorem prob_sum_greater_than_8_is_5_18 :
  prob_sum_greater_than_8 = 5 / 18 := by
  sorry

end NUMINAMATH_CALUDE_prob_sum_greater_than_8_is_5_18_l2782_278215


namespace NUMINAMATH_CALUDE_work_completion_time_l2782_278269

/-- Given two workers a and b, where:
    1. a and b can finish the work together in 30 days
    2. a alone can finish the work in 60 days
    3. a and b worked together for 20 days before b left
    This theorem proves that a finishes the remaining work in 20 days after b left. -/
theorem work_completion_time 
  (total_work : ℝ) 
  (a_rate : ℝ) 
  (b_rate : ℝ) 
  (h1 : a_rate + b_rate = total_work / 30)
  (h2 : a_rate = total_work / 60)
  (h3 : (a_rate + b_rate) * 20 = 2 * total_work / 3) :
  (total_work / 3) / a_rate = 20 := by
  sorry

#check work_completion_time

end NUMINAMATH_CALUDE_work_completion_time_l2782_278269


namespace NUMINAMATH_CALUDE_even_function_property_l2782_278221

-- Define an even function
def EvenFunction (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = f x

-- State the theorem
theorem even_function_property (f : ℝ → ℝ) 
  (h_even : EvenFunction f) 
  (h_positive : ∀ x > 0, f x = 10^x) : 
  ∀ x < 0, f x = (1/10)^x := by
sorry

end NUMINAMATH_CALUDE_even_function_property_l2782_278221


namespace NUMINAMATH_CALUDE_ellipse_point_position_l2782_278280

theorem ellipse_point_position 
  (a b c : ℝ) 
  (x₁ x₂ : ℝ) 
  (h_a_pos : a > 0)
  (h_b_pos : b > 0)
  (h_c_pos : c > 0)
  (h_a_gt_b : a > b)
  (h_roots : x₁ + x₂ = -b/a ∧ x₁ * x₂ = -c/a) :
  1 < x₁^2 + x₂^2 ∧ x₁^2 + x₂^2 < 2 := by
sorry

end NUMINAMATH_CALUDE_ellipse_point_position_l2782_278280


namespace NUMINAMATH_CALUDE_value_of_d_l2782_278276

theorem value_of_d (a b c d : ℕ+) 
  (h1 : a^2 = c * (d + 29))
  (h2 : b^2 = c * (d - 29)) :
  d = 421 := by
  sorry

end NUMINAMATH_CALUDE_value_of_d_l2782_278276


namespace NUMINAMATH_CALUDE_apple_tree_production_l2782_278290

/-- Apple tree production over three years -/
theorem apple_tree_production : 
  let first_year : ℕ := 40
  let second_year : ℕ := 8 + 2 * first_year
  let third_year : ℕ := second_year - second_year / 4
  first_year + second_year + third_year = 194 := by
sorry

end NUMINAMATH_CALUDE_apple_tree_production_l2782_278290


namespace NUMINAMATH_CALUDE_equation_solution_l2782_278230

theorem equation_solution : ∃ x : ℝ, (0.82 : ℝ)^3 - (0.1 : ℝ)^3 / (0.82 : ℝ)^2 + x + (0.1 : ℝ)^2 = 0.72 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l2782_278230


namespace NUMINAMATH_CALUDE_sum_in_base5_l2782_278223

/-- Represents a number in base 5 --/
def Base5 : Type := ℕ

/-- Converts a base 5 number to its decimal representation --/
def to_decimal (n : Base5) : ℕ := sorry

/-- Converts a decimal number to its base 5 representation --/
def to_base5 (n : ℕ) : Base5 := sorry

/-- Addition operation for base 5 numbers --/
def base5_add (a b : Base5) : Base5 := to_base5 (to_decimal a + to_decimal b)

theorem sum_in_base5 :
  let a : Base5 := to_base5 231
  let b : Base5 := to_base5 414
  let c : Base5 := to_base5 123
  let result : Base5 := to_base5 1323
  base5_add (base5_add a b) c = result := by sorry

end NUMINAMATH_CALUDE_sum_in_base5_l2782_278223


namespace NUMINAMATH_CALUDE_imaginary_number_properties_l2782_278224

open Complex

theorem imaginary_number_properties (z : ℂ) (ω : ℝ) :
  z.im ≠ 0 →  -- z is an imaginary number
  ω = z.re + z.im * I + (z.re - z.im * I) / (z.re^2 + z.im^2) →  -- ω = z + 1/z
  -1 < ω ∧ ω < 2 →  -- -1 < ω < 2
  abs z = 1 ∧  -- |z| = 1
  -1/2 < z.re ∧ z.re < 1 ∧  -- real part of z is in (-1/2, 1)
  1 < abs (z - 2) ∧ abs (z - 2) < Real.sqrt 7  -- |z-2| is in (1, √7)
  := by sorry

end NUMINAMATH_CALUDE_imaginary_number_properties_l2782_278224


namespace NUMINAMATH_CALUDE_f_smallest_positive_period_l2782_278211

noncomputable def f (x : ℝ) : ℝ := 3 * Real.sin (2 * x) + 2^(|Real.sin (2 * x)|^2) + 5 * |Real.sin (2 * x)|

def is_periodic (f : ℝ → ℝ) (p : ℝ) : Prop :=
  ∀ x, f (x + p) = f x

def is_smallest_positive_period (f : ℝ → ℝ) (p : ℝ) : Prop :=
  p > 0 ∧ is_periodic f p ∧ ∀ q, 0 < q ∧ q < p → ¬is_periodic f q

theorem f_smallest_positive_period :
  is_smallest_positive_period f (Real.pi / 2) := by
  sorry

end NUMINAMATH_CALUDE_f_smallest_positive_period_l2782_278211


namespace NUMINAMATH_CALUDE_calculate_expression_1_calculate_expression_2_l2782_278268

-- Part 1
theorem calculate_expression_1 :
  (1) * (27 / 8) ^ (-2 / 3) - (49 / 9) ^ 0.5 + (0.008) ^ (-2 / 3) * (2 / 25) = 1 / 9 := by
  sorry

-- Part 2
theorem calculate_expression_2 :
  (2) * Real.log 25 / Real.log 10 + (2 / 3) * Real.log 8 / Real.log 10 +
  Real.log 5 / Real.log 10 * Real.log 20 / Real.log 10 + (Real.log 2 / Real.log 10) ^ 2 = 3 := by
  sorry

end NUMINAMATH_CALUDE_calculate_expression_1_calculate_expression_2_l2782_278268


namespace NUMINAMATH_CALUDE_nancy_carrots_l2782_278299

def initial_carrots : ℕ → ℕ → ℕ → Prop :=
  fun x thrown_out additional =>
    x - thrown_out + additional = 31

theorem nancy_carrots : initial_carrots 12 2 21 := by sorry

end NUMINAMATH_CALUDE_nancy_carrots_l2782_278299


namespace NUMINAMATH_CALUDE_fraction_calculation_l2782_278214

theorem fraction_calculation : 
  (8/9 - 5/6 + 2/3) / (-5/18) = -13/5 := by
  sorry

end NUMINAMATH_CALUDE_fraction_calculation_l2782_278214

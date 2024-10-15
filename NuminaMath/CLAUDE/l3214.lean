import Mathlib

namespace NUMINAMATH_CALUDE_parabola_transformation_l3214_321464

/-- Represents a parabola in the form y = (x + a)² + b -/
structure Parabola where
  a : ℝ
  b : ℝ

/-- Applies a horizontal shift to a parabola -/
def horizontal_shift (p : Parabola) (shift : ℝ) : Parabola :=
  { a := p.a - shift, b := p.b }

/-- Applies a vertical shift to a parabola -/
def vertical_shift (p : Parabola) (shift : ℝ) : Parabola :=
  { a := p.a, b := p.b + shift }

/-- The theorem to be proved -/
theorem parabola_transformation (p : Parabola) :
  p.a = 2 ∧ p.b = 3 →
  (vertical_shift (horizontal_shift p 3) (-2)) = { a := -1, b := 1 } := by
  sorry

end NUMINAMATH_CALUDE_parabola_transformation_l3214_321464


namespace NUMINAMATH_CALUDE_binary_1011_to_decimal_l3214_321457

/-- Converts a binary number represented as a list of bits (least significant bit first) to its decimal equivalent -/
def binary_to_decimal (bits : List Bool) : ℕ :=
  bits.enum.foldl (fun acc (i, b) => acc + if b then 2^i else 0) 0

/-- The binary representation of 1011 -/
def binary_1011 : List Bool := [true, true, false, true]

/-- Theorem stating that the decimal representation of binary 1011 is 11 -/
theorem binary_1011_to_decimal :
  binary_to_decimal binary_1011 = 11 := by sorry

end NUMINAMATH_CALUDE_binary_1011_to_decimal_l3214_321457


namespace NUMINAMATH_CALUDE_percentage_calculation_l3214_321489

theorem percentage_calculation (N : ℝ) (P : ℝ) : 
  N = 4800 → 
  (P / 100) * (30 / 100) * (50 / 100) * N = 108 → 
  P = 15 := by
sorry

end NUMINAMATH_CALUDE_percentage_calculation_l3214_321489


namespace NUMINAMATH_CALUDE_gcd_factorial_eight_ten_l3214_321488

theorem gcd_factorial_eight_ten : Nat.gcd (Nat.factorial 8) (Nat.factorial 10) = Nat.factorial 8 := by
  sorry

end NUMINAMATH_CALUDE_gcd_factorial_eight_ten_l3214_321488


namespace NUMINAMATH_CALUDE_reciprocal_of_repeating_decimal_l3214_321497

/-- The common fraction form of the repeating decimal 0.363636... -/
def repeating_decimal : ℚ := 4 / 11

/-- The reciprocal of the common fraction form of 0.363636... -/
def reciprocal : ℚ := 11 / 4

theorem reciprocal_of_repeating_decimal : (repeating_decimal)⁻¹ = reciprocal := by
  sorry

end NUMINAMATH_CALUDE_reciprocal_of_repeating_decimal_l3214_321497


namespace NUMINAMATH_CALUDE_divide_by_repeating_decimal_l3214_321430

def repeating_decimal_to_fraction (a b : ℕ) : ℚ :=
  (a : ℚ) / (99 : ℚ)

theorem divide_by_repeating_decimal (a b : ℕ) :
  (7 : ℚ) / (repeating_decimal_to_fraction a b) = 38.5 :=
sorry

end NUMINAMATH_CALUDE_divide_by_repeating_decimal_l3214_321430


namespace NUMINAMATH_CALUDE_two_integer_solutions_l3214_321438

theorem two_integer_solutions :
  ∃! (s : Finset ℤ), (∀ x ∈ s, |3*x - 4| + |3*x + 2| = 6) ∧ s.card = 2 :=
by sorry

end NUMINAMATH_CALUDE_two_integer_solutions_l3214_321438


namespace NUMINAMATH_CALUDE_sqrt_200_simplification_l3214_321471

theorem sqrt_200_simplification : Real.sqrt 200 = 10 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_200_simplification_l3214_321471


namespace NUMINAMATH_CALUDE_absolute_value_fraction_less_than_one_l3214_321485

theorem absolute_value_fraction_less_than_one (x y : ℝ) 
  (hx : |x| < 1) (hy : |y| < 1) : |(x - y) / (1 - x * y)| < 1 := by
  sorry

end NUMINAMATH_CALUDE_absolute_value_fraction_less_than_one_l3214_321485


namespace NUMINAMATH_CALUDE_opposite_of_abs_one_over_2023_l3214_321402

theorem opposite_of_abs_one_over_2023 :
  -(|1 / 2023|) = -1 / 2023 := by sorry

end NUMINAMATH_CALUDE_opposite_of_abs_one_over_2023_l3214_321402


namespace NUMINAMATH_CALUDE_cambridge_population_l3214_321474

-- Define the number of people in Cambridge
variable (n : ℕ)

-- Define the total amount of water and apple juice consumed
variable (W A : ℝ)

-- Define the mayor's drink
variable (L : ℝ)

-- Each person drinks 12 ounces
axiom total_drink : W + A = 12 * n

-- The mayor's drink is 12 ounces
axiom mayor_drink : L = 12

-- The mayor drinks 1/6 of total water and 1/8 of total apple juice
axiom mayor_portions : L = (1/6) * W + (1/8) * A

-- All drinks have positive amounts of both liquids
axiom positive_amounts : W > 0 ∧ A > 0

-- Theorem: The number of people in Cambridge is 7
theorem cambridge_population : n = 7 :=
sorry

end NUMINAMATH_CALUDE_cambridge_population_l3214_321474


namespace NUMINAMATH_CALUDE_picnic_cost_l3214_321455

def sandwich_price : ℚ := 6
def fruit_salad_price : ℚ := 4
def cheese_platter_price : ℚ := 8
def soda_price : ℚ := 2.5
def snack_bag_price : ℚ := 4.5

def num_people : ℕ := 6
def num_sandwiches : ℕ := 6
def num_fruit_salads : ℕ := 4
def num_cheese_platters : ℕ := 3
def num_sodas : ℕ := 12
def num_snack_bags : ℕ := 5

def sandwich_discount (n : ℕ) : ℕ := n / 6
def cheese_platter_discount (n : ℕ) : ℚ := if n ≥ 2 then 0.1 else 0
def soda_discount (n : ℕ) : ℕ := (n / 10) * 2
def snack_bag_discount (n : ℕ) : ℕ := n / 2

def total_cost : ℚ :=
  (num_sandwiches - sandwich_discount num_sandwiches) * sandwich_price +
  num_fruit_salads * fruit_salad_price +
  (num_cheese_platters * cheese_platter_price) * (1 - cheese_platter_discount num_cheese_platters) +
  (num_sodas - soda_discount num_sodas) * soda_price +
  num_snack_bags * snack_bag_price - snack_bag_discount num_snack_bags

theorem picnic_cost : total_cost = 113.1 := by
  sorry

end NUMINAMATH_CALUDE_picnic_cost_l3214_321455


namespace NUMINAMATH_CALUDE_gcd_lcm_sum_l3214_321463

theorem gcd_lcm_sum : Nat.gcd 45 75 + Nat.lcm 48 18 = 159 := by sorry

end NUMINAMATH_CALUDE_gcd_lcm_sum_l3214_321463


namespace NUMINAMATH_CALUDE_natural_fraction_pairs_l3214_321467

def is_valid_pair (x y : ℕ) : Prop :=
  (∃ k : ℕ, (x + 1) = k * y) ∧ (∃ m : ℕ, (y + 1) = m * x)

theorem natural_fraction_pairs :
  ∀ x y : ℕ, is_valid_pair x y ↔ 
    ((x = 2 ∧ y = 1) ∨ (x = 3 ∧ y = 2) ∨ (x = 1 ∧ y = 2) ∨ (x = 2 ∧ y = 3)) :=
by sorry

end NUMINAMATH_CALUDE_natural_fraction_pairs_l3214_321467


namespace NUMINAMATH_CALUDE_battery_price_l3214_321416

theorem battery_price (total_cost tire_cost : ℕ) (h1 : total_cost = 224) (h2 : tire_cost = 42) :
  total_cost - 4 * tire_cost = 56 := by
  sorry

end NUMINAMATH_CALUDE_battery_price_l3214_321416


namespace NUMINAMATH_CALUDE_painted_cube_probability_l3214_321460

/-- Represents a cube with side length 5 and two adjacent faces painted --/
structure PaintedCube :=
  (side_length : ℕ)
  (painted_faces : ℕ)

/-- Calculates the total number of unit cubes in the large cube --/
def total_cubes (c : PaintedCube) : ℕ :=
  c.side_length ^ 3

/-- Calculates the number of unit cubes with exactly two painted faces --/
def two_painted_faces (c : PaintedCube) : ℕ :=
  (c.side_length - 2) ^ 2

/-- Calculates the number of unit cubes with no painted faces --/
def no_painted_faces (c : PaintedCube) : ℕ :=
  total_cubes c - (2 * c.side_length ^ 2 - c.side_length)

/-- Calculates the probability of selecting one cube with two painted faces
    and one cube with no painted faces --/
def probability (c : PaintedCube) : ℚ :=
  (two_painted_faces c * no_painted_faces c : ℚ) /
  (total_cubes c * (total_cubes c - 1) / 2 : ℚ)

/-- The main theorem stating the probability for a 5x5x5 cube with two painted faces --/
theorem painted_cube_probability :
  let c := PaintedCube.mk 5 2
  probability c = 24 / 258 := by
  sorry


end NUMINAMATH_CALUDE_painted_cube_probability_l3214_321460


namespace NUMINAMATH_CALUDE_adjacent_knights_probability_l3214_321401

def number_of_knights : ℕ := 30
def chosen_knights : ℕ := 4

def probability_adjacent_knights : ℚ :=
  1 - (Nat.choose (number_of_knights - chosen_knights) chosen_knights : ℚ) /
      (Nat.choose number_of_knights chosen_knights : ℚ)

theorem adjacent_knights_probability :
  probability_adjacent_knights = 250 / 549 := by sorry

end NUMINAMATH_CALUDE_adjacent_knights_probability_l3214_321401


namespace NUMINAMATH_CALUDE_chocolate_bar_cost_l3214_321483

/-- Proves that the cost of each bar of chocolate is $5 given the conditions of the problem. -/
theorem chocolate_bar_cost
  (num_bars : ℕ)
  (total_selling_price : ℚ)
  (packaging_cost_per_bar : ℚ)
  (total_profit : ℚ)
  (h1 : num_bars = 5)
  (h2 : total_selling_price = 90)
  (h3 : packaging_cost_per_bar = 2)
  (h4 : total_profit = 55) :
  ∃ (cost_per_bar : ℚ), cost_per_bar = 5 ∧
    total_selling_price = num_bars * cost_per_bar + num_bars * packaging_cost_per_bar + total_profit :=
by sorry

end NUMINAMATH_CALUDE_chocolate_bar_cost_l3214_321483


namespace NUMINAMATH_CALUDE_negative_reciprocal_positive_l3214_321436

theorem negative_reciprocal_positive (x : ℝ) (h : x < 0) : -x⁻¹ > 0 := by
  sorry

end NUMINAMATH_CALUDE_negative_reciprocal_positive_l3214_321436


namespace NUMINAMATH_CALUDE_power_multiplication_l3214_321465

theorem power_multiplication (a : ℝ) : a ^ 2 * a ^ 5 = a ^ 7 := by
  sorry

end NUMINAMATH_CALUDE_power_multiplication_l3214_321465


namespace NUMINAMATH_CALUDE_rectangle_side_equality_l3214_321478

theorem rectangle_side_equality (X : ℝ) : 
  (∀ (top bottom : ℝ), top = 5 + X ∧ bottom = 10 ∧ top = bottom) → X = 5 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_side_equality_l3214_321478


namespace NUMINAMATH_CALUDE_square_side_length_l3214_321492

theorem square_side_length (rectangle_width rectangle_length : ℝ) 
  (h1 : rectangle_width = 4)
  (h2 : rectangle_length = 9)
  (h3 : rectangle_width > 0)
  (h4 : rectangle_length > 0) :
  ∃ (square_side : ℝ), 
    square_side * square_side = rectangle_width * rectangle_length ∧ 
    square_side = 6 := by
  sorry

end NUMINAMATH_CALUDE_square_side_length_l3214_321492


namespace NUMINAMATH_CALUDE_volunteer_selection_probabilities_l3214_321423

/-- Represents the number of calligraphy competition winners -/
def calligraphy_winners : ℕ := 4

/-- Represents the number of painting competition winners -/
def painting_winners : ℕ := 2

/-- Represents the total number of winners -/
def total_winners : ℕ := calligraphy_winners + painting_winners

/-- Represents the number of volunteers to be selected -/
def volunteers_needed : ℕ := 2

/-- The probability of selecting both volunteers from calligraphy winners -/
def prob_both_calligraphy : ℚ := 2 / 5

/-- The probability of selecting one volunteer from each competition -/
def prob_one_each : ℚ := 8 / 15

theorem volunteer_selection_probabilities :
  (Nat.choose calligraphy_winners volunteers_needed) / (Nat.choose total_winners volunteers_needed) = prob_both_calligraphy ∧
  (calligraphy_winners * painting_winners) / (Nat.choose total_winners volunteers_needed) = prob_one_each :=
sorry

end NUMINAMATH_CALUDE_volunteer_selection_probabilities_l3214_321423


namespace NUMINAMATH_CALUDE_fine_on_fifth_day_l3214_321486

/-- Calculates the fine for a given day based on the previous day's fine -/
def nextDayFine (prevFine : ℚ) : ℚ :=
  min (prevFine + 0.3) (prevFine * 2)

/-- Calculates the fine for a specified number of days -/
def fineAfterDays (days : ℕ) : ℚ :=
  match days with
  | 0 => 0
  | 1 => 0.07
  | n + 1 => nextDayFine (fineAfterDays n)

theorem fine_on_fifth_day :
  fineAfterDays 5 = 0.86 := by sorry

end NUMINAMATH_CALUDE_fine_on_fifth_day_l3214_321486


namespace NUMINAMATH_CALUDE_race_speed_ratio_l3214_321451

/-- The ratio of runner a's speed to runner b's speed in a race -/
def speed_ratio (head_start_percent : ℚ) (winning_distance_percent : ℚ) : ℚ :=
  (1 + head_start_percent) / (1 + winning_distance_percent)

/-- Theorem stating that the speed ratio is 37/35 given the specified conditions -/
theorem race_speed_ratio :
  speed_ratio (48/100) (40/100) = 37/35 := by
  sorry

end NUMINAMATH_CALUDE_race_speed_ratio_l3214_321451


namespace NUMINAMATH_CALUDE_equation_solution_l3214_321414

theorem equation_solution (x : ℝ) : (2*x - 3)^(x + 3) = 1 ↔ x = -3 ∨ x = 2 ∨ x = -1 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l3214_321414


namespace NUMINAMATH_CALUDE_exactly_one_event_probability_l3214_321473

theorem exactly_one_event_probability (p₁ p₂ : ℝ) 
  (h₁ : 0 ≤ p₁ ∧ p₁ ≤ 1) (h₂ : 0 ≤ p₂ ∧ p₂ ≤ 1) : 
  p₁ * (1 - p₂) + p₂ * (1 - p₁) = 
  (p₁ + p₂) - (p₁ * p₂) := by
  sorry

end NUMINAMATH_CALUDE_exactly_one_event_probability_l3214_321473


namespace NUMINAMATH_CALUDE_fifth_term_of_arithmetic_sequence_l3214_321426

/-- 
Given an arithmetic sequence starting with 3, 7, 11, ..., 
prove that its fifth term is 19.
-/
theorem fifth_term_of_arithmetic_sequence : 
  ∀ (a : ℕ → ℕ), 
  (a 0 = 3) → 
  (a 1 = 7) → 
  (a 2 = 11) → 
  (∀ n, a (n + 1) - a n = a 1 - a 0) → 
  a 4 = 19 := by
sorry

end NUMINAMATH_CALUDE_fifth_term_of_arithmetic_sequence_l3214_321426


namespace NUMINAMATH_CALUDE_triangle_problem_l3214_321449

open Real

theorem triangle_problem (a b c : ℝ) (A B C : ℝ) :
  0 < a ∧ 0 < b ∧ 0 < c →
  0 < A ∧ 0 < B ∧ 0 < C →
  A + B + C = π →
  a * cos B + b * cos A = 2 * c * cos C →
  c = 2 * Real.sqrt 3 →
  C = π / 3 ∧
  (∃ (area : ℝ), area ≤ 3 * Real.sqrt 3 ∧
    ∀ (area' : ℝ), area' = 1/2 * a * b * sin C → area' ≤ area) :=
by sorry

end NUMINAMATH_CALUDE_triangle_problem_l3214_321449


namespace NUMINAMATH_CALUDE_consecutive_squares_difference_l3214_321417

theorem consecutive_squares_difference (n : ℕ) : (n + 1)^2 - n^2 = 2*n + 1 := by
  sorry

end NUMINAMATH_CALUDE_consecutive_squares_difference_l3214_321417


namespace NUMINAMATH_CALUDE_necessary_but_not_sufficient_condition_l3214_321413

theorem necessary_but_not_sufficient_condition (p q : Prop) :
  (p ∧ q → p ∨ q) ∧ ¬(p ∨ q → p ∧ q) := by
  sorry

end NUMINAMATH_CALUDE_necessary_but_not_sufficient_condition_l3214_321413


namespace NUMINAMATH_CALUDE_xy_bounds_l3214_321475

theorem xy_bounds (x y a : ℝ) (h1 : x + y = a) (h2 : x^2 + y^2 = -a^2 + 2) :
  -1 ≤ x * y ∧ x * y ≤ 1/3 := by
  sorry

end NUMINAMATH_CALUDE_xy_bounds_l3214_321475


namespace NUMINAMATH_CALUDE_capital_after_18_years_l3214_321482

def initial_investment : ℝ := 2000
def increase_rate : ℝ := 0.5
def years_per_period : ℕ := 3
def total_years : ℕ := 18

theorem capital_after_18_years :
  let periods : ℕ := total_years / years_per_period
  let growth_factor : ℝ := 1 + increase_rate
  let final_capital : ℝ := initial_investment * growth_factor ^ periods
  final_capital = 22781.25 := by sorry

end NUMINAMATH_CALUDE_capital_after_18_years_l3214_321482


namespace NUMINAMATH_CALUDE_nested_expression_value_l3214_321424

theorem nested_expression_value : (3 * (3 * (3 * (3 * (3 * (3 + 2) + 2) + 2) + 2) + 2) + 2) = 1457 := by
  sorry

end NUMINAMATH_CALUDE_nested_expression_value_l3214_321424


namespace NUMINAMATH_CALUDE_race_result_l3214_321444

/-- Represents a runner in the race -/
structure Runner where
  speed : ℝ
  position : ℝ → ℝ

/-- The race setup -/
structure Race where
  sasha : Runner
  lesha : Runner
  kolya : Runner
  length : ℝ

def Race.valid (race : Race) : Prop :=
  race.sasha.speed > 0 ∧ 
  race.lesha.speed > 0 ∧ 
  race.kolya.speed > 0 ∧
  race.length > 0 ∧
  -- When Sasha finishes, Lesha is 10 meters behind
  race.sasha.position (race.length / race.sasha.speed) = race.length ∧
  race.lesha.position (race.length / race.sasha.speed) = race.length - 10 ∧
  -- When Lesha finishes, Kolya is 10 meters behind
  race.lesha.position (race.length / race.lesha.speed) = race.length ∧
  race.kolya.position (race.length / race.lesha.speed) = race.length - 10 ∧
  -- All runners have constant speeds
  ∀ t, race.sasha.position t = race.sasha.speed * t ∧
       race.lesha.position t = race.lesha.speed * t ∧
       race.kolya.position t = race.kolya.speed * t

theorem race_result (race : Race) (h : race.valid) : 
  race.kolya.position (race.length / race.sasha.speed) = race.length - 19 := by
  sorry

end NUMINAMATH_CALUDE_race_result_l3214_321444


namespace NUMINAMATH_CALUDE_ripe_oranges_per_day_l3214_321499

theorem ripe_oranges_per_day :
  ∀ (daily_ripe_oranges : ℕ),
    daily_ripe_oranges * 73 = 365 →
    daily_ripe_oranges = 5 := by
  sorry

end NUMINAMATH_CALUDE_ripe_oranges_per_day_l3214_321499


namespace NUMINAMATH_CALUDE_unique_palindrome_square_l3214_321419

/-- A function that returns true if a number is a three-digit palindrome with an even middle digit -/
def is_valid_palindrome (n : ℕ) : Prop :=
  100 ≤ n ∧ n < 1000 ∧  -- three-digit number
  (n / 100 = n % 10) ∧  -- first and last digits are the same
  (n / 10 % 10) % 2 = 0  -- middle digit is even

/-- The main theorem stating that there is exactly one number satisfying the conditions -/
theorem unique_palindrome_square : ∃! n : ℕ, 
  is_valid_palindrome n ∧ ∃ m : ℕ, n = m^2 :=
sorry

end NUMINAMATH_CALUDE_unique_palindrome_square_l3214_321419


namespace NUMINAMATH_CALUDE_max_candy_consumption_l3214_321495

theorem max_candy_consumption (n : ℕ) (h : n = 45) : 
  (n * (n - 1)) / 2 = 990 :=
by sorry

end NUMINAMATH_CALUDE_max_candy_consumption_l3214_321495


namespace NUMINAMATH_CALUDE_initial_milk_water_ratio_l3214_321452

theorem initial_milk_water_ratio 
  (total_initial_volume : ℝ)
  (additional_water : ℝ)
  (final_ratio : ℝ)
  (h1 : total_initial_volume = 45)
  (h2 : additional_water = 23)
  (h3 : final_ratio = 1.125)
  : ∃ (initial_milk initial_water : ℝ),
    initial_milk + initial_water = total_initial_volume ∧
    initial_milk / (initial_water + additional_water) = final_ratio ∧
    initial_milk / initial_water = 4 := by
  sorry

end NUMINAMATH_CALUDE_initial_milk_water_ratio_l3214_321452


namespace NUMINAMATH_CALUDE_sign_determination_l3214_321454

theorem sign_determination (a b : ℝ) (h1 : a > b) (h2 : 1 / a > 1 / b) : a > 0 ∧ b < 0 := by
  sorry

end NUMINAMATH_CALUDE_sign_determination_l3214_321454


namespace NUMINAMATH_CALUDE_function_always_negative_m_range_l3214_321434

theorem function_always_negative_m_range
  (f : ℝ → ℝ)
  (m : ℝ)
  (h1 : ∀ x, f x = m * x^2 - m * x - 1)
  (h2 : ∀ x, f x < 0) :
  -4 < m ∧ m ≤ 0 :=
by sorry

end NUMINAMATH_CALUDE_function_always_negative_m_range_l3214_321434


namespace NUMINAMATH_CALUDE_rotate_right_triangle_surface_area_l3214_321411

/-- The surface area of a solid formed by rotating a right triangle with sides 3, 4, and 5 around its shortest side -/
theorem rotate_right_triangle_surface_area :
  let triangle : Fin 3 → ℝ := ![3, 4, 5]
  let shortest_side := triangle 0
  let hypotenuse := triangle 2
  let height := triangle 1
  let base_area := π * height ^ 2
  let lateral_area := π * height * hypotenuse
  base_area + lateral_area = 36 * π :=
by sorry

end NUMINAMATH_CALUDE_rotate_right_triangle_surface_area_l3214_321411


namespace NUMINAMATH_CALUDE_min_value_for_four_digit_product_l3214_321421

theorem min_value_for_four_digit_product (n : ℕ) : 
  (341 * n ≥ 1000 ∧ ∀ m < n, 341 * m < 1000) → n = 3 := by
  sorry

end NUMINAMATH_CALUDE_min_value_for_four_digit_product_l3214_321421


namespace NUMINAMATH_CALUDE_stream_speed_l3214_321431

/-- Represents the speed of a boat in a stream -/
structure BoatSpeed where
  boatStillWater : ℝ  -- Speed of the boat in still water
  stream : ℝ          -- Speed of the stream

/-- Calculates the effective speed of the boat -/
def effectiveSpeed (b : BoatSpeed) (downstream : Bool) : ℝ :=
  if downstream then b.boatStillWater + b.stream else b.boatStillWater - b.stream

/-- Theorem: Given the conditions, the speed of the stream is 3 km/h -/
theorem stream_speed (b : BoatSpeed) 
  (h1 : effectiveSpeed b true * 4 = 84)  -- Downstream condition
  (h2 : effectiveSpeed b false * 4 = 60) -- Upstream condition
  : b.stream = 3 := by
  sorry


end NUMINAMATH_CALUDE_stream_speed_l3214_321431


namespace NUMINAMATH_CALUDE_journey_speed_proof_l3214_321445

/-- Proves that given a journey of 120 miles in 120 minutes, with average speeds of 50 mph and 60 mph
for the first and second 40-minute segments respectively, the average speed for the last 40-minute
segment is 70 mph. -/
theorem journey_speed_proof (total_distance : ℝ) (total_time : ℝ) (speed1 : ℝ) (speed2 : ℝ) :
  total_distance = 120 →
  total_time = 120 →
  speed1 = 50 →
  speed2 = 60 →
  ∃ (speed3 : ℝ), speed3 = 70 ∧ (speed1 + speed2 + speed3) / 3 = total_distance / (total_time / 60) :=
by sorry

end NUMINAMATH_CALUDE_journey_speed_proof_l3214_321445


namespace NUMINAMATH_CALUDE_parallelogram_side_sum_l3214_321403

/-- 
A parallelogram has side lengths of 10, 12, 10y-2, and 4x+6. 
This theorem proves that x+y = 2.7.
-/
theorem parallelogram_side_sum (x y : ℝ) : 
  (4*x + 6 = 12) → (10*y - 2 = 10) → x + y = 2.7 := by sorry

end NUMINAMATH_CALUDE_parallelogram_side_sum_l3214_321403


namespace NUMINAMATH_CALUDE_cubic_quartic_relation_l3214_321448

theorem cubic_quartic_relation (x y : ℝ) 
  (h1 : x^3 + y^3 + 1 / (x^3 + y^3) = 3) 
  (h2 : x + y = 2) : 
  x^4 + y^4 + 1 / (x^4 + y^4) = 257/16 := by
  sorry

end NUMINAMATH_CALUDE_cubic_quartic_relation_l3214_321448


namespace NUMINAMATH_CALUDE_seed_mixture_percentage_l3214_321458

/-- Given two seed mixtures X and Y, and a final mixture containing both, 
    this theorem proves the percentage of mixture X in the final mixture. -/
theorem seed_mixture_percentage 
  (x_ryegrass : ℚ) (x_bluegrass : ℚ) (y_ryegrass : ℚ) (y_fescue : ℚ) 
  (final_ryegrass : ℚ) : 
  x_ryegrass = 40 / 100 →
  x_bluegrass = 60 / 100 →
  y_ryegrass = 25 / 100 →
  y_fescue = 75 / 100 →
  final_ryegrass = 38 / 100 →
  x_ryegrass + x_bluegrass = 1 →
  y_ryegrass + y_fescue = 1 →
  ∃ (p : ℚ), p * x_ryegrass + (1 - p) * y_ryegrass = final_ryegrass ∧ 
             p = 260 / 3 :=
by sorry

end NUMINAMATH_CALUDE_seed_mixture_percentage_l3214_321458


namespace NUMINAMATH_CALUDE_sum_mod_eleven_l3214_321494

theorem sum_mod_eleven : (10555 + 10556 + 10557 + 10558 + 10559) % 11 = 4 := by
  sorry

end NUMINAMATH_CALUDE_sum_mod_eleven_l3214_321494


namespace NUMINAMATH_CALUDE_min_sum_of_squares_l3214_321442

/-- Given that a + 2b + 3c + 4d = 12, prove that a^2 + b^2 + c^2 + d^2 ≥ 24/5 -/
theorem min_sum_of_squares (a b c d : ℝ) (h : a + 2*b + 3*c + 4*d = 12) :
  a^2 + b^2 + c^2 + d^2 ≥ 24/5 := by
  sorry

end NUMINAMATH_CALUDE_min_sum_of_squares_l3214_321442


namespace NUMINAMATH_CALUDE_locus_of_vertices_is_parabola_l3214_321466

/-- The locus of vertices of a family of parabolas forms a parabola -/
theorem locus_of_vertices_is_parabola (a c : ℝ) (ha : a > 0) (hc : c > 0) :
  ∃ (A B C : ℝ), A ≠ 0 ∧
    (∀ t : ℝ, ∃ (x y : ℝ),
      (y = a * x^2 + (2 * t + 1) * x + c) ∧
      (x = -(2 * t + 1) / (2 * a)) ∧
      (y = A * x^2 + B * x + C)) :=
by sorry

end NUMINAMATH_CALUDE_locus_of_vertices_is_parabola_l3214_321466


namespace NUMINAMATH_CALUDE_distance_to_focus_l3214_321480

/-- Given a parabola y^2 = 4x and a point M(x₀, 2√3) on it, 
    the distance from M to the focus of the parabola is 4. -/
theorem distance_to_focus (x₀ : ℝ) : 
  (2 * Real.sqrt 3)^2 = 4 * x₀ →   -- Point M is on the parabola
  x₀ + 1 = 4 :=                    -- Distance to focus
by sorry

end NUMINAMATH_CALUDE_distance_to_focus_l3214_321480


namespace NUMINAMATH_CALUDE_best_fit_r_squared_l3214_321427

def r_squared_values : List ℝ := [0.27, 0.85, 0.96, 0.5]

theorem best_fit_r_squared (best_fit : ℝ) (h : best_fit ∈ r_squared_values) :
  (∀ x ∈ r_squared_values, x ≤ best_fit) ∧ best_fit = 0.96 := by
  sorry

end NUMINAMATH_CALUDE_best_fit_r_squared_l3214_321427


namespace NUMINAMATH_CALUDE_negation_of_exponential_inequality_l3214_321447

theorem negation_of_exponential_inequality :
  (¬ ∀ a : ℝ, a > 0 → Real.exp a ≥ 1) ↔ (∃ a : ℝ, a > 0 ∧ Real.exp a < 1) := by
  sorry

end NUMINAMATH_CALUDE_negation_of_exponential_inequality_l3214_321447


namespace NUMINAMATH_CALUDE_rahul_salary_calculation_l3214_321429

def calculate_remaining_salary (initial_salary : ℕ) : ℕ :=
  let after_rent := initial_salary - initial_salary * 20 / 100
  let after_education := after_rent - after_rent * 10 / 100
  let after_clothes := after_education - after_education * 10 / 100
  after_clothes

theorem rahul_salary_calculation :
  calculate_remaining_salary 2125 = 1377 := by
  sorry

end NUMINAMATH_CALUDE_rahul_salary_calculation_l3214_321429


namespace NUMINAMATH_CALUDE_triangle_angle_sum_l3214_321400

theorem triangle_angle_sum (a b c : ℝ) (h1 : 0 < a) (h2 : 0 < b) (h3 : 0 < c) 
  (h4 : a + b + c = 180) (h5 : a = 37) (h6 : b = 53) : c = 90 := by
  sorry

end NUMINAMATH_CALUDE_triangle_angle_sum_l3214_321400


namespace NUMINAMATH_CALUDE_point_coordinates_l3214_321491

-- Define a point in 2D space
def Point := ℝ × ℝ

-- Define the distance from a point to the x-axis
def distToXAxis (p : Point) : ℝ := |p.2|

-- Define the distance from a point to the y-axis
def distToYAxis (p : Point) : ℝ := |p.1|

-- Theorem statement
theorem point_coordinates (P : Point) :
  P.2 > 0 →  -- P is above the x-axis
  P.1 < 0 →  -- P is to the left of the y-axis
  distToXAxis P = 4 →  -- P is 4 units away from x-axis
  distToYAxis P = 4 →  -- P is 4 units away from y-axis
  P = (-4, 4) := by
sorry

end NUMINAMATH_CALUDE_point_coordinates_l3214_321491


namespace NUMINAMATH_CALUDE_gcd_1248_1001_l3214_321408

theorem gcd_1248_1001 : Nat.gcd 1248 1001 = 13 := by
  sorry

end NUMINAMATH_CALUDE_gcd_1248_1001_l3214_321408


namespace NUMINAMATH_CALUDE_power_function_sum_l3214_321493

/-- A function f is a power function if it can be written as f(x) = k * x^c, 
    where k and c are constants, and c is not zero. -/
def isPowerFunction (f : ℝ → ℝ) : Prop :=
  ∃ (k c : ℝ), c ≠ 0 ∧ ∀ x, f x = k * x^c

/-- Given that f(x) = a*x^(2a+1) - b + 1 is a power function, prove that a + b = 2 -/
theorem power_function_sum (a b : ℝ) :
  isPowerFunction (fun x ↦ a * x^(2*a+1) - b + 1) → a + b = 2 := by
  sorry

end NUMINAMATH_CALUDE_power_function_sum_l3214_321493


namespace NUMINAMATH_CALUDE_inscribed_circle_radius_l3214_321456

theorem inscribed_circle_radius (a b c : ℝ) (ha : a = 5) (hb : b = 12) (hc : c = 15) :
  let r := (1 / a + 1 / b + 1 / c + 2 * Real.sqrt (1 / (a * b) + 1 / (a * c) + 1 / (b * c)))⁻¹
  r = 20 / 19 := by sorry

end NUMINAMATH_CALUDE_inscribed_circle_radius_l3214_321456


namespace NUMINAMATH_CALUDE_hybrid_cars_with_full_headlights_l3214_321490

/-- Proves that in a car dealership with 600 cars, where 60% are hybrids and 40% of hybrids have only one headlight, the number of hybrids with full headlights is 216. -/
theorem hybrid_cars_with_full_headlights (total_cars : ℕ) (hybrid_percentage : ℚ) (one_headlight_percentage : ℚ) :
  total_cars = 600 →
  hybrid_percentage = 60 / 100 →
  one_headlight_percentage = 40 / 100 →
  (total_cars : ℚ) * hybrid_percentage * (1 - one_headlight_percentage) = 216 := by
  sorry

end NUMINAMATH_CALUDE_hybrid_cars_with_full_headlights_l3214_321490


namespace NUMINAMATH_CALUDE_gcd_245_1001_l3214_321405

theorem gcd_245_1001 : Nat.gcd 245 1001 = 7 := by sorry

end NUMINAMATH_CALUDE_gcd_245_1001_l3214_321405


namespace NUMINAMATH_CALUDE_factor_theorem_application_l3214_321477

theorem factor_theorem_application (x t : ℝ) : 
  (∃ k : ℝ, 4 * x^2 + 9 * x - 2 = (x - t) * k) ↔ (t = -1/4 ∨ t = -2) :=
by sorry

end NUMINAMATH_CALUDE_factor_theorem_application_l3214_321477


namespace NUMINAMATH_CALUDE_reciprocal_sum_identity_l3214_321498

theorem reciprocal_sum_identity (x y z : ℝ) (h : x ≠ 0 ∧ y ≠ 0 ∧ z ≠ 0) :
  1 / x + 1 / y = 1 / z → z = x * y / (y + x) := by
  sorry

end NUMINAMATH_CALUDE_reciprocal_sum_identity_l3214_321498


namespace NUMINAMATH_CALUDE_max_S_value_l3214_321418

-- Define a triangle
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  -- Triangle inequality constraints
  pos_a : a > 0
  pos_b : b > 0
  pos_c : c > 0
  ineq_ab : a + b > c
  ineq_bc : b + c > a
  ineq_ca : c + a > b

-- Define the area function S
def S (t : Triangle) : ℝ := (t.a - t.b + t.c) * (t.a + t.b - t.c)

-- Theorem statement
theorem max_S_value (t : Triangle) (h : t.b + t.c = 8) :
  S t ≤ 64 / 17 :=
sorry

end NUMINAMATH_CALUDE_max_S_value_l3214_321418


namespace NUMINAMATH_CALUDE_cat_in_bag_change_l3214_321459

theorem cat_in_bag_change (p : ℕ) (h : 0 < p ∧ p ≤ 1000) : 
  ∃ (change : ℕ), change = 1000 - p := by
  sorry

end NUMINAMATH_CALUDE_cat_in_bag_change_l3214_321459


namespace NUMINAMATH_CALUDE_book_cost_problem_l3214_321487

theorem book_cost_problem (total_cost : ℝ) (loss_percent : ℝ) (gain_percent : ℝ) 
  (h1 : total_cost = 360)
  (h2 : loss_percent = 0.15)
  (h3 : gain_percent = 0.19)
  (h4 : ∃ (c1 c2 : ℝ), c1 + c2 = total_cost ∧ 
                       c1 * (1 - loss_percent) = c2 * (1 + gain_percent)) :
  ∃ (loss_book_cost : ℝ), loss_book_cost = 210 ∧ 
    ∃ (c2 : ℝ), loss_book_cost + c2 = total_cost ∧ 
    loss_book_cost * (1 - loss_percent) = c2 * (1 + gain_percent) :=
sorry

end NUMINAMATH_CALUDE_book_cost_problem_l3214_321487


namespace NUMINAMATH_CALUDE_ellipse_equation_with_shared_focus_l3214_321440

/-- Given a parabola and an ellipse with shared focus, prove the equation of the ellipse -/
theorem ellipse_equation_with_shared_focus (a : ℝ) (h_a : a > 0) :
  (∃ (x y : ℝ), y^2 = 8*x) →  -- Parabola exists
  (∃ (x y : ℝ), x^2/a^2 + y^2 = 1) →  -- Ellipse exists
  (2 : ℝ) = a * (1 - 1/a^2).sqrt →  -- Focus of parabola is right focus of ellipse
  (∃ (x y : ℝ), x^2/5 + y^2 = 1) :=  -- Resulting ellipse equation
by sorry

end NUMINAMATH_CALUDE_ellipse_equation_with_shared_focus_l3214_321440


namespace NUMINAMATH_CALUDE_distinct_divisors_lower_bound_l3214_321432

theorem distinct_divisors_lower_bound (n : ℕ) (A : ℕ) (factors : Finset ℕ) 
  (h1 : factors.card = n)
  (h2 : ∀ x ∈ factors, x > 1)
  (h3 : A = factors.prod id) :
  (Finset.filter (· ∣ A) (Finset.range (A + 1))).card ≥ n * (n - 1) / 2 + 1 := by
  sorry

end NUMINAMATH_CALUDE_distinct_divisors_lower_bound_l3214_321432


namespace NUMINAMATH_CALUDE_interior_angles_sum_l3214_321425

theorem interior_angles_sum (n : ℕ) (h : 180 * (n - 2) = 2340) :
  180 * ((n + 3) - 2) = 2880 := by
  sorry

end NUMINAMATH_CALUDE_interior_angles_sum_l3214_321425


namespace NUMINAMATH_CALUDE_min_distance_circle_line_l3214_321468

theorem min_distance_circle_line : 
  ∃ (d : ℝ), d = 2 ∧ 
  (∀ (x₁ y₁ x₂ y₂ : ℝ), 
    ((x₁ - Real.sqrt 3 / 2)^2 + (y₁ - 1/2)^2 = 1) →
    (Real.sqrt 3 * x₂ + y₂ = 8) →
    d ≤ Real.sqrt ((x₁ - x₂)^2 + (y₁ - y₂)^2)) ∧
  (∃ (x₁ y₁ x₂ y₂ : ℝ), 
    ((x₁ - Real.sqrt 3 / 2)^2 + (y₁ - 1/2)^2 = 1) ∧
    (Real.sqrt 3 * x₂ + y₂ = 8) ∧
    d = Real.sqrt ((x₁ - x₂)^2 + (y₁ - y₂)^2)) :=
by
  sorry

end NUMINAMATH_CALUDE_min_distance_circle_line_l3214_321468


namespace NUMINAMATH_CALUDE_video_cassettes_in_second_set_l3214_321435

-- Define the cost of a video cassette
def video_cassette_cost : ℕ := 300

-- Define the equations from the problem
def equation1 (audio_cost video_count : ℕ) : Prop :=
  5 * audio_cost + video_count * video_cassette_cost = 1350

def equation2 (audio_cost : ℕ) : Prop :=
  7 * audio_cost + 3 * video_cassette_cost = 1110

-- Theorem to prove
theorem video_cassettes_in_second_set :
  ∃ (audio_cost video_count : ℕ),
    equation1 audio_cost video_count ∧
    equation2 audio_cost →
    3 = 3 :=
sorry

end NUMINAMATH_CALUDE_video_cassettes_in_second_set_l3214_321435


namespace NUMINAMATH_CALUDE_opposite_of_2021_l3214_321428

theorem opposite_of_2021 : -(2021 : ℤ) = -2021 := by
  sorry

end NUMINAMATH_CALUDE_opposite_of_2021_l3214_321428


namespace NUMINAMATH_CALUDE_initial_friends_count_l3214_321412

theorem initial_friends_count (initial_group : ℕ) (additional_friends : ℕ) (total_people : ℕ) : 
  initial_group + additional_friends = total_people ∧ 
  additional_friends = 3 ∧ 
  total_people = 7 → 
  initial_group = 4 := by
  sorry

end NUMINAMATH_CALUDE_initial_friends_count_l3214_321412


namespace NUMINAMATH_CALUDE_triangle_angle_weighted_average_bounds_l3214_321441

theorem triangle_angle_weighted_average_bounds 
  (A B C a b c : ℝ) 
  (h_triangle : A + B + C = π) 
  (h_positive : a > 0 ∧ b > 0 ∧ c > 0) : 
  π / 3 ≤ (a * A + b * B + c * C) / (a + b + c) ∧ 
  (a * A + b * B + c * C) / (a + b + c) < π / 2 := by
sorry

end NUMINAMATH_CALUDE_triangle_angle_weighted_average_bounds_l3214_321441


namespace NUMINAMATH_CALUDE_second_vessel_capacity_l3214_321479

/-- Proves that the capacity of the second vessel is 3.625 liters given the conditions of the problem -/
theorem second_vessel_capacity :
  let vessel1_capacity : ℝ := 3
  let vessel1_alcohol_percentage : ℝ := 0.25
  let vessel2_alcohol_percentage : ℝ := 0.40
  let total_liquid : ℝ := 8
  let new_concentration : ℝ := 0.275
  ∃ vessel2_capacity : ℝ,
    vessel2_capacity > 0 ∧
    vessel1_capacity * vessel1_alcohol_percentage + 
    vessel2_capacity * vessel2_alcohol_percentage = 
    total_liquid * new_concentration ∧
    vessel2_capacity = 3.625 := by
  sorry


end NUMINAMATH_CALUDE_second_vessel_capacity_l3214_321479


namespace NUMINAMATH_CALUDE_sam_initial_watermelons_l3214_321406

/-- The number of watermelons Sam grew initially -/
def initial_watermelons : ℕ := sorry

/-- The number of additional watermelons Sam grew -/
def additional_watermelons : ℕ := 3

/-- The total number of watermelons Sam has now -/
def total_watermelons : ℕ := 7

/-- Theorem stating that Sam grew 4 watermelons initially -/
theorem sam_initial_watermelons : 
  initial_watermelons + additional_watermelons = total_watermelons → initial_watermelons = 4 := by
  sorry

end NUMINAMATH_CALUDE_sam_initial_watermelons_l3214_321406


namespace NUMINAMATH_CALUDE_boys_share_is_14_l3214_321420

/-- The amount of money each boy makes from selling shrimp -/
def boys_share (victor_shrimp : ℕ) (austin_diff : ℕ) (price : ℚ) (per_shrimp : ℕ) : ℚ :=
  let austin_shrimp := victor_shrimp - austin_diff
  let victor_austin_total := victor_shrimp + austin_shrimp
  let brian_shrimp := victor_austin_total / 2
  let total_shrimp := victor_shrimp + austin_shrimp + brian_shrimp
  let total_money := (total_shrimp / per_shrimp) * price
  total_money / 3

/-- Theorem stating that each boy's share is $14 given the problem conditions -/
theorem boys_share_is_14 :
  boys_share 26 8 7 11 = 14 := by
  sorry

end NUMINAMATH_CALUDE_boys_share_is_14_l3214_321420


namespace NUMINAMATH_CALUDE_red_marbles_taken_away_l3214_321410

/-- Proves that the number of red marbles taken away is 3 --/
theorem red_marbles_taken_away :
  let initial_red : ℕ := 20
  let initial_blue : ℕ := 30
  let total_left : ℕ := 35
  ∃ (red_taken : ℕ),
    (initial_red - red_taken) + (initial_blue - 4 * red_taken) = total_left ∧
    red_taken = 3 :=
by sorry

end NUMINAMATH_CALUDE_red_marbles_taken_away_l3214_321410


namespace NUMINAMATH_CALUDE_madeline_and_brother_total_money_l3214_321415

def madeline_money : ℕ := 48

theorem madeline_and_brother_total_money :
  madeline_money + (madeline_money / 2) = 72 := by
  sorry

end NUMINAMATH_CALUDE_madeline_and_brother_total_money_l3214_321415


namespace NUMINAMATH_CALUDE_concentric_circles_area_ratio_l3214_321433

/-- Given two concentric circles D and C, where C is inside D, 
    this theorem proves the diameter of C when the ratio of 
    the area between the circles to the area of C is 4:1 -/
theorem concentric_circles_area_ratio (d_diameter : ℝ) 
  (h_d_diameter : d_diameter = 24) 
  (c_diameter : ℝ) 
  (h_inside : c_diameter < d_diameter) 
  (h_ratio : (π * (d_diameter/2)^2 - π * (c_diameter/2)^2) / (π * (c_diameter/2)^2) = 4) :
  c_diameter = 24 * Real.sqrt 5 / 5 := by
  sorry

#check concentric_circles_area_ratio

end NUMINAMATH_CALUDE_concentric_circles_area_ratio_l3214_321433


namespace NUMINAMATH_CALUDE_target_seat_representation_l3214_321476

/-- Represents a seat in a cinema -/
structure CinemaSeat where
  row : Nat
  seatNumber : Nat

/-- Given representation for seat number 4 in row 6 -/
def givenSeat : CinemaSeat := ⟨6, 4⟩

/-- The seat we want to represent (seat number 1 in row 5) -/
def targetSeat : CinemaSeat := ⟨5, 1⟩

/-- Theorem stating that the target seat is correctly represented -/
theorem target_seat_representation : targetSeat = ⟨5, 1⟩ := by
  sorry

end NUMINAMATH_CALUDE_target_seat_representation_l3214_321476


namespace NUMINAMATH_CALUDE_probability_between_lines_in_first_quadrant_l3214_321481

/-- Line represented by a linear equation y = mx + b -/
structure Line where
  m : ℝ
  b : ℝ

/-- Point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

def Line.eval (l : Line) (x : ℝ) : ℝ := l.m * x + l.b

def is_below (p : Point) (l : Line) : Prop := p.y ≤ l.eval p.x

def is_in_first_quadrant (p : Point) : Prop := p.x ≥ 0 ∧ p.y ≥ 0

def is_between_lines (p : Point) (l1 l2 : Line) : Prop :=
  is_below p l1 ∧ ¬is_below p l2

theorem probability_between_lines_in_first_quadrant
  (l m : Line)
  (h1 : l.m = -3 ∧ l.b = 9)
  (h2 : m.m = -1 ∧ m.b = 3)
  (h3 : ∀ (p : Point), is_in_first_quadrant p → is_below p l → is_below p m) :
  (∀ (p : Point), is_in_first_quadrant p → is_below p l → is_between_lines p l m) :=
sorry

end NUMINAMATH_CALUDE_probability_between_lines_in_first_quadrant_l3214_321481


namespace NUMINAMATH_CALUDE_condition_necessary_not_sufficient_l3214_321462

-- Define the equation of an ellipse
def is_ellipse_equation (a b : ℝ) : Prop :=
  ∀ x y : ℝ, x^2 / a + y^2 / b = 1 → (a > 0 ∧ b > 0 ∧ a ≠ b)

-- Define the condition ab > 0
def condition (a b : ℝ) : Prop := a * b > 0

-- Theorem stating that the condition is necessary but not sufficient
theorem condition_necessary_not_sufficient :
  (∀ a b : ℝ, is_ellipse_equation a b → condition a b) ∧
  (∃ a b : ℝ, condition a b ∧ ¬is_ellipse_equation a b) := by
  sorry

end NUMINAMATH_CALUDE_condition_necessary_not_sufficient_l3214_321462


namespace NUMINAMATH_CALUDE_half_circle_roll_distance_l3214_321461

/-- The length of the path traveled by the center of a half-circle when rolled along a straight line -/
theorem half_circle_roll_distance (r : ℝ) (h : r = 3 / Real.pi) : 
  let roll_distance := r * Real.pi + r
  roll_distance = 3 + 3 / Real.pi := by sorry

end NUMINAMATH_CALUDE_half_circle_roll_distance_l3214_321461


namespace NUMINAMATH_CALUDE_angle_is_135_degrees_l3214_321439

def angle_between_vectors (a b : ℝ × ℝ) : ℝ := sorry

theorem angle_is_135_degrees (a b : ℝ × ℝ) 
  (sum_condition : a.1 + b.1 = 2 ∧ a.2 + b.2 = -1)
  (a_condition : a = (1, 2)) :
  angle_between_vectors a b = 135 * (π / 180) := by sorry

end NUMINAMATH_CALUDE_angle_is_135_degrees_l3214_321439


namespace NUMINAMATH_CALUDE_unique_non_representable_expression_l3214_321443

/-- Represents an algebraic expression that may or may not be
    representable as a square of a binomial or difference of squares. -/
inductive BinomialExpression
  | Representable (a b : ℤ) : BinomialExpression
  | NotRepresentable (a b : ℤ) : BinomialExpression

/-- Determines if a given expression can be represented as a 
    square of a binomial or difference of squares. -/
def is_representable (expr : BinomialExpression) : Prop :=
  match expr with
  | BinomialExpression.Representable _ _ => True
  | BinomialExpression.NotRepresentable _ _ => False

/-- The four expressions from the original problem. -/
def expr1 : BinomialExpression := BinomialExpression.Representable 1 (-2)
def expr2 : BinomialExpression := BinomialExpression.Representable 1 (-2)
def expr3 : BinomialExpression := BinomialExpression.Representable 2 (-1)
def expr4 : BinomialExpression := BinomialExpression.NotRepresentable 1 2

theorem unique_non_representable_expression :
  is_representable expr1 ∧
  is_representable expr2 ∧
  is_representable expr3 ∧
  ¬is_representable expr4 :=
sorry

end NUMINAMATH_CALUDE_unique_non_representable_expression_l3214_321443


namespace NUMINAMATH_CALUDE_centroid_triangle_area_l3214_321472

-- Define the rectangle ABCD
structure Rectangle where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ
  D : ℝ × ℝ

-- Define point E on side CD
def E (rect : Rectangle) : ℝ × ℝ :=
  sorry

-- Define the area of a triangle
def triangleArea (p q r : ℝ × ℝ) : ℝ :=
  sorry

-- Define the centroid of a triangle
def centroid (p q r : ℝ × ℝ) : ℝ × ℝ :=
  sorry

theorem centroid_triangle_area (rect : Rectangle) :
  let G₁ := centroid rect.A rect.D (E rect)
  let G₂ := centroid rect.A rect.B (E rect)
  let G₃ := centroid rect.B rect.C (E rect)
  triangleArea G₁ G₂ G₃ = 1/18 :=
by
  sorry

end NUMINAMATH_CALUDE_centroid_triangle_area_l3214_321472


namespace NUMINAMATH_CALUDE_smallest_y_l3214_321484

theorem smallest_y (y : ℕ) 
  (h1 : y % 6 = 5) 
  (h2 : y % 7 = 6) 
  (h3 : y % 8 = 7) : 
  y ≥ 167 ∧ ∃ (z : ℕ), z % 6 = 5 ∧ z % 7 = 6 ∧ z % 8 = 7 ∧ z = 167 :=
sorry

end NUMINAMATH_CALUDE_smallest_y_l3214_321484


namespace NUMINAMATH_CALUDE_ellipse_intersection_length_l3214_321409

-- Define the ellipse C
def ellipse_C (a b : ℝ) (x y : ℝ) : Prop :=
  (x^2 / a^2) + (y^2 / b^2) = 1

-- Define the conditions for the rhombus formed by the vertices
def rhombus_condition (a b : ℝ) : Prop :=
  a > b ∧ b > 0 ∧ 2 * a * b = 2 * Real.sqrt 2 ∧ a^2 + b^2 = 3

-- Define the line l
def line_l (k : ℝ) (x y : ℝ) : Prop :=
  y = k * x - 2

-- Define the slope product condition
def slope_product_condition (x₁ y₁ x₂ y₂ : ℝ) : Prop :=
  (y₁ / x₁) * (y₂ / x₂) = -1

-- Main theorem
theorem ellipse_intersection_length :
  ∀ (a b : ℝ),
  rhombus_condition a b →
  (∀ (x y : ℝ), ellipse_C a b x y ↔ x^2 / 2 + y^2 = 1) ∧
  (∀ (k x₁ y₁ x₂ y₂ : ℝ),
    ellipse_C a b x₁ y₁ ∧ 
    ellipse_C a b x₂ y₂ ∧
    line_l k x₁ y₁ ∧ 
    line_l k x₂ y₂ ∧
    slope_product_condition x₁ y₁ x₂ y₂ →
    Real.sqrt ((x₁ - x₂)^2 + (y₁ - y₂)^2) = 4 * Real.sqrt 21 / 11) :=
sorry

end NUMINAMATH_CALUDE_ellipse_intersection_length_l3214_321409


namespace NUMINAMATH_CALUDE_displeased_polynomial_at_one_is_zero_l3214_321496

-- Define a polynomial p(x) = x^2 - (m+n)x + mn
def p (m n : ℝ) (x : ℝ) : ℝ := x^2 - (m + n) * x + m * n

-- Define what it means for a polynomial to be displeased
def isDispleased (m n : ℝ) : Prop :=
  ∃ (x₁ x₂ x₃ x₄ : ℝ), x₁ ≠ x₂ ∧ x₁ ≠ x₃ ∧ x₁ ≠ x₄ ∧ x₂ ≠ x₃ ∧ x₂ ≠ x₄ ∧ x₃ ≠ x₄ ∧
  (∀ x : ℝ, p m n (p m n x) = 0 ↔ x = x₁ ∨ x = x₂ ∨ x = x₃ ∨ x = x₄)

-- Define the theorem
theorem displeased_polynomial_at_one_is_zero :
  ∃! (a : ℝ), isDispleased a a ∧
  (∀ m n : ℝ, isDispleased m n → m * n ≤ a * a) ∧
  p a a 1 = 0 :=
sorry

end NUMINAMATH_CALUDE_displeased_polynomial_at_one_is_zero_l3214_321496


namespace NUMINAMATH_CALUDE_fraction_simplification_l3214_321437

theorem fraction_simplification :
  (1 / 2 + 1 / 5) / (3 / 7 - 1 / 14) = 49 / 25 := by
  sorry

end NUMINAMATH_CALUDE_fraction_simplification_l3214_321437


namespace NUMINAMATH_CALUDE_greatest_prime_factor_of_154_l3214_321422

theorem greatest_prime_factor_of_154 :
  ∃ (p : ℕ), Nat.Prime p ∧ p ∣ 154 ∧ ∀ q, Nat.Prime q → q ∣ 154 → q ≤ p ∧ p = 11 :=
by sorry

end NUMINAMATH_CALUDE_greatest_prime_factor_of_154_l3214_321422


namespace NUMINAMATH_CALUDE_circus_tent_capacity_l3214_321470

theorem circus_tent_capacity (total_capacity : ℕ) (num_sections : ℕ) (section_capacity : ℕ) : 
  total_capacity = 984 → 
  num_sections = 4 → 
  total_capacity = num_sections * section_capacity → 
  section_capacity = 246 := by
sorry

end NUMINAMATH_CALUDE_circus_tent_capacity_l3214_321470


namespace NUMINAMATH_CALUDE_max_truck_speed_l3214_321446

theorem max_truck_speed (distance : ℝ) (hourly_cost : ℝ) (fixed_cost : ℝ) (max_total_cost : ℝ) :
  distance = 125 →
  hourly_cost = 30 →
  fixed_cost = 1000 →
  max_total_cost = 1200 →
  ∃ (max_speed : ℝ),
    max_speed = 75 ∧
    ∀ (speed : ℝ),
      speed > 0 →
      (distance / speed) * hourly_cost + fixed_cost + 2 * speed ≤ max_total_cost →
      speed ≤ max_speed :=
sorry

end NUMINAMATH_CALUDE_max_truck_speed_l3214_321446


namespace NUMINAMATH_CALUDE_cat_teeth_count_l3214_321453

theorem cat_teeth_count (dog_teeth : ℕ) (pig_teeth : ℕ) (num_dogs : ℕ) (num_cats : ℕ) (num_pigs : ℕ) (total_teeth : ℕ) :
  dog_teeth = 42 →
  pig_teeth = 28 →
  num_dogs = 5 →
  num_cats = 10 →
  num_pigs = 7 →
  total_teeth = 706 →
  (total_teeth - num_dogs * dog_teeth - num_pigs * pig_teeth) / num_cats = 30 := by
sorry

end NUMINAMATH_CALUDE_cat_teeth_count_l3214_321453


namespace NUMINAMATH_CALUDE_log_product_telescoping_l3214_321450

theorem log_product_telescoping (z : ℝ) : 
  z = (Real.log 4 / Real.log 3) * (Real.log 5 / Real.log 4) * 
      (Real.log 6 / Real.log 5) * (Real.log 7 / Real.log 6) * 
      (Real.log 8 / Real.log 7) * (Real.log 9 / Real.log 8) * 
      (Real.log 10 / Real.log 9) * (Real.log 11 / Real.log 10) * 
      (Real.log 12 / Real.log 11) * (Real.log 13 / Real.log 12) * 
      (Real.log 14 / Real.log 13) * (Real.log 15 / Real.log 14) * 
      (Real.log 16 / Real.log 15) * (Real.log 17 / Real.log 16) * 
      (Real.log 18 / Real.log 17) * (Real.log 19 / Real.log 18) * 
      (Real.log 20 / Real.log 19) * (Real.log 21 / Real.log 20) * 
      (Real.log 22 / Real.log 21) * (Real.log 23 / Real.log 22) * 
      (Real.log 24 / Real.log 23) * (Real.log 25 / Real.log 24) * 
      (Real.log 26 / Real.log 25) * (Real.log 27 / Real.log 26) * 
      (Real.log 28 / Real.log 27) * (Real.log 29 / Real.log 28) * 
      (Real.log 30 / Real.log 29) * (Real.log 31 / Real.log 30) * 
      (Real.log 32 / Real.log 31) * (Real.log 33 / Real.log 32) * 
      (Real.log 34 / Real.log 33) * (Real.log 35 / Real.log 34) * 
      (Real.log 36 / Real.log 35) * (Real.log 37 / Real.log 36) * 
      (Real.log 38 / Real.log 37) * (Real.log 39 / Real.log 38) * 
      (Real.log 40 / Real.log 39) →
  z = (3 * Real.log 2 + Real.log 5) / Real.log 3 := by
  sorry

end NUMINAMATH_CALUDE_log_product_telescoping_l3214_321450


namespace NUMINAMATH_CALUDE_count_odd_integers_between_fractions_l3214_321404

theorem count_odd_integers_between_fractions :
  let lower_bound : ℚ := 17 / 4
  let upper_bound : ℚ := 35 / 2
  (Finset.filter (fun n => n % 2 = 1)
    (Finset.Icc (Int.ceil lower_bound) (Int.floor upper_bound))).card = 7 := by
  sorry

end NUMINAMATH_CALUDE_count_odd_integers_between_fractions_l3214_321404


namespace NUMINAMATH_CALUDE_correct_parentheses_removal_l3214_321469

theorem correct_parentheses_removal (x : ℝ) :
  2 - 4 * ((1/4) * x + 1) = 2 - x - 4 := by
sorry

end NUMINAMATH_CALUDE_correct_parentheses_removal_l3214_321469


namespace NUMINAMATH_CALUDE_subtracted_value_l3214_321407

theorem subtracted_value (x y : ℤ) : x = 60 ∧ 4 * x - y = 102 → y = 138 := by
  sorry

end NUMINAMATH_CALUDE_subtracted_value_l3214_321407

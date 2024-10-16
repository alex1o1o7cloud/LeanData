import Mathlib

namespace NUMINAMATH_CALUDE_coin_draw_probability_l803_80386

-- Define the coin types and their quantities
def pennies : ℕ := 3
def nickels : ℕ := 5
def dimes : ℕ := 4
def quarters : ℕ := 2

-- Define the total number of coins
def total_coins : ℕ := pennies + nickels + dimes + quarters

-- Define the number of coins drawn
def coins_drawn : ℕ := 8

-- Define the probability function
def probability_at_least_one_dollar : ℚ :=
  1596 / 3003

-- Theorem statement
theorem coin_draw_probability :
  probability_at_least_one_dollar = 
    (Nat.choose total_coins coins_drawn).pred / Nat.choose total_coins coins_drawn :=
by sorry

end NUMINAMATH_CALUDE_coin_draw_probability_l803_80386


namespace NUMINAMATH_CALUDE_arithmetic_sequence_count_l803_80318

theorem arithmetic_sequence_count :
  ∀ (a₁ last d : ℕ) (n : ℕ),
    a₁ = 1 →
    last = 2025 →
    d = 4 →
    last = a₁ + d * (n - 1) →
    n = 507 :=
by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_count_l803_80318


namespace NUMINAMATH_CALUDE_system_solution_l803_80388

theorem system_solution (x y k : ℝ) : 
  (4 * x + 2 * y = 5 * k - 4) → 
  (2 * x + 4 * y = -1) → 
  (x - y = 1) → 
  (k = 1) := by
sorry

end NUMINAMATH_CALUDE_system_solution_l803_80388


namespace NUMINAMATH_CALUDE_pta_funds_remaining_l803_80313

def initial_amount : ℚ := 600

def amount_after_supplies (initial : ℚ) : ℚ :=
  initial - (2 / 5) * initial

def amount_after_food (after_supplies : ℚ) : ℚ :=
  after_supplies - (30 / 100) * after_supplies

def final_amount (after_food : ℚ) : ℚ :=
  after_food - (1 / 3) * after_food

theorem pta_funds_remaining :
  final_amount (amount_after_food (amount_after_supplies initial_amount)) = 168 := by
  sorry

end NUMINAMATH_CALUDE_pta_funds_remaining_l803_80313


namespace NUMINAMATH_CALUDE_brothers_ages_theorem_l803_80302

/-- Represents the ages of the three brothers -/
structure BrothersAges where
  kolya : ℕ
  vanya : ℕ
  petya : ℕ

/-- The conditions given in the problem -/
def satisfiesConditions (ages : BrothersAges) : Prop :=
  ages.petya = 10 ∧
  ages.kolya = ages.petya + 3 ∧
  ages.vanya = ages.petya - 1

/-- The theorem to be proved -/
theorem brothers_ages_theorem (ages : BrothersAges) :
  satisfiesConditions ages → ages.vanya = 9 ∧ ages.kolya = 13 := by
  sorry

#check brothers_ages_theorem

end NUMINAMATH_CALUDE_brothers_ages_theorem_l803_80302


namespace NUMINAMATH_CALUDE_parabola_directrix_l803_80376

/-- The parabola defined by y = 8x^2 + 2 has a directrix y = 63/32 -/
theorem parabola_directrix : ∀ (x y : ℝ), y = 8 * x^2 + 2 → 
  ∃ (f d : ℝ), f = -d ∧ f - d = 1/16 ∧ d = -1/32 ∧ 
  (∀ (p : ℝ × ℝ), p.2 = 8 * p.1^2 + 2 → 
    (p.1^2 + (p.2 - (f + 2))^2 = (p.2 - (d + 2))^2)) ∧
  63/32 = d + 2 := by
  sorry


end NUMINAMATH_CALUDE_parabola_directrix_l803_80376


namespace NUMINAMATH_CALUDE_annual_mischief_convention_handshakes_l803_80301

/-- The number of handshakes at the Annual Mischief Convention -/
theorem annual_mischief_convention_handshakes (n_gremlins : ℕ) (n_imps : ℕ) : 
  n_gremlins = 30 → n_imps = 15 → 
  (n_gremlins * (n_gremlins - 1)) / 2 + n_imps * (n_gremlins / 2) = 660 := by
  sorry

end NUMINAMATH_CALUDE_annual_mischief_convention_handshakes_l803_80301


namespace NUMINAMATH_CALUDE_pentagon_diagonal_equality_l803_80360

/-- A point in a 2D plane -/
structure Point :=
  (x : ℝ) (y : ℝ)

/-- A pentagon defined by five points -/
structure Pentagon :=
  (A B C D E : Point)

/-- Checks if a pentagon is convex -/
def is_convex (p : Pentagon) : Prop := sorry

/-- Checks if a line segment bisects an angle -/
def bisects_angle (P Q R S : Point) : Prop := sorry

/-- Finds the intersection point of two line segments -/
def intersection (P Q R S : Point) : Point := sorry

/-- Calculates the distance between two points -/
def distance (P Q : Point) : ℝ := sorry

theorem pentagon_diagonal_equality (p : Pentagon) 
  (h_convex : is_convex p)
  (h_bd_bisect1 : bisects_angle p.C p.B p.E p.D)
  (h_bd_bisect2 : bisects_angle p.C p.D p.A p.B)
  (h_ce_bisect1 : bisects_angle p.A p.C p.D p.E)
  (h_ce_bisect2 : bisects_angle p.B p.E p.D p.C)
  (K : Point) (h_K : K = intersection p.B p.E p.A p.C)
  (L : Point) (h_L : L = intersection p.B p.E p.A p.D) :
  distance p.C K = distance p.D L := by sorry

end NUMINAMATH_CALUDE_pentagon_diagonal_equality_l803_80360


namespace NUMINAMATH_CALUDE_star_two_three_solve_equation_l803_80330

-- Define the new operation
def star (a b : ℝ) : ℝ := a^2 + 2*a*b

-- Theorem 1
theorem star_two_three : star 2 3 = 16 := by sorry

-- Theorem 2
theorem solve_equation (x : ℝ) : star (-2) x = -2 + x → x = 6/5 := by sorry

end NUMINAMATH_CALUDE_star_two_three_solve_equation_l803_80330


namespace NUMINAMATH_CALUDE_william_washed_two_normal_cars_l803_80312

/-- The time William spends washing a normal car's windows -/
def window_time : ℕ := 4

/-- The time William spends washing a normal car's body -/
def body_time : ℕ := 7

/-- The time William spends cleaning a normal car's tires -/
def tire_time : ℕ := 4

/-- The time William spends waxing a normal car -/
def wax_time : ℕ := 9

/-- The total time William spends on one normal car -/
def normal_car_time : ℕ := window_time + body_time + tire_time + wax_time

/-- The time William spends on one big SUV -/
def suv_time : ℕ := 2 * normal_car_time

/-- The total time William spent washing all vehicles -/
def total_time : ℕ := 96

/-- The number of normal cars William washed -/
def normal_cars : ℕ := (total_time - suv_time) / normal_car_time

theorem william_washed_two_normal_cars : normal_cars = 2 := by
  sorry

end NUMINAMATH_CALUDE_william_washed_two_normal_cars_l803_80312


namespace NUMINAMATH_CALUDE_path_area_l803_80333

/-- Calculates the area of a path surrounding a rectangular field -/
theorem path_area (field_length field_width path_width : ℝ) :
  field_length = 85 ∧ 
  field_width = 55 ∧ 
  path_width = 2.5 → 
  (field_length + 2 * path_width) * (field_width + 2 * path_width) - 
  field_length * field_width = 725 := by
  sorry

end NUMINAMATH_CALUDE_path_area_l803_80333


namespace NUMINAMATH_CALUDE_xyz_product_absolute_value_l803_80337

theorem xyz_product_absolute_value (x y z : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) (hz : z ≠ 0)
  (hdistinct : x ≠ y ∧ y ≠ z ∧ x ≠ z)
  (heq1 : x + 1 / y = y + 1 / z)
  (heq2 : y + 1 / z = z + 1 / x + 1) :
  |x * y * z| = 1 := by
sorry

end NUMINAMATH_CALUDE_xyz_product_absolute_value_l803_80337


namespace NUMINAMATH_CALUDE_stratified_sampling_problem_l803_80340

theorem stratified_sampling_problem (total : ℕ) (sample_size : ℕ) 
  (stratum_A : ℕ) (stratum_B : ℕ) (h1 : total = 1200) (h2 : sample_size = 120) 
  (h3 : stratum_A = 380) (h4 : stratum_B = 420) : 
  let stratum_C := total - stratum_A - stratum_B
  (sample_size * stratum_C) / total = 40 := by
sorry

end NUMINAMATH_CALUDE_stratified_sampling_problem_l803_80340


namespace NUMINAMATH_CALUDE_sheep_count_l803_80320

/-- Given 3 herds of sheep with 20 sheep in each herd, the total number of sheep is 60. -/
theorem sheep_count (num_herds : ℕ) (sheep_per_herd : ℕ) 
  (h1 : num_herds = 3) 
  (h2 : sheep_per_herd = 20) : 
  num_herds * sheep_per_herd = 60 := by
  sorry

end NUMINAMATH_CALUDE_sheep_count_l803_80320


namespace NUMINAMATH_CALUDE_concert_duration_13h25m_l803_80382

/-- Calculates the total duration in minutes of a concert given its length in hours and minutes. -/
def concert_duration (hours : ℕ) (minutes : ℕ) : ℕ :=
  hours * 60 + minutes

/-- Theorem stating that a concert lasting 13 hours and 25 minutes has a total duration of 805 minutes. -/
theorem concert_duration_13h25m :
  concert_duration 13 25 = 805 := by
  sorry

end NUMINAMATH_CALUDE_concert_duration_13h25m_l803_80382


namespace NUMINAMATH_CALUDE_probability_a_equals_one_l803_80335

theorem probability_a_equals_one (a b c : ℕ+) (sum_constraint : a + b + c = 6) :
  (Finset.filter (fun x => x.1 = 1) (Finset.product (Finset.range 6) (Finset.product (Finset.range 6) (Finset.range 6)))).card /
  (Finset.filter (fun x => x.1 + x.2.1 + x.2.2 = 6) (Finset.product (Finset.range 6) (Finset.product (Finset.range 6) (Finset.range 6)))).card
  = 2 / 5 := by
  sorry

end NUMINAMATH_CALUDE_probability_a_equals_one_l803_80335


namespace NUMINAMATH_CALUDE_abc12_paths_l803_80341

/-- Represents the number of adjacent letters or numerals --/
def adjacent_count (letter : Char) : Nat :=
  match letter with
  | 'A' => 2  -- Number of B's adjacent to A
  | 'B' => 3  -- Number of C's adjacent to each B
  | 'C' => 2  -- Number of 1's adjacent to each C
  | '1' => 1  -- Number of 2's adjacent to each 1
  | _   => 0  -- For any other character

/-- Calculates the total number of paths to spell ABC12 --/
def total_paths : Nat :=
  adjacent_count 'A' * adjacent_count 'B' * adjacent_count 'C' * adjacent_count '1'

/-- Theorem stating that the number of paths to spell ABC12 is 12 --/
theorem abc12_paths : total_paths = 12 := by
  sorry

end NUMINAMATH_CALUDE_abc12_paths_l803_80341


namespace NUMINAMATH_CALUDE_oliver_final_amount_l803_80346

def oliver_money_left (initial_amount savings frisbee_cost puzzle_cost birthday_gift : ℕ) : ℕ :=
  initial_amount + savings - frisbee_cost - puzzle_cost + birthday_gift

theorem oliver_final_amount :
  oliver_money_left 9 5 4 3 8 = 15 := by
  sorry

end NUMINAMATH_CALUDE_oliver_final_amount_l803_80346


namespace NUMINAMATH_CALUDE_trapezoid_ab_length_l803_80348

/-- Represents a trapezoid ABCD with specific properties -/
structure Trapezoid where
  -- Length of side AB
  ab : ℝ
  -- Length of side CD
  cd : ℝ
  -- Ratio of areas of triangles ABC and ADC
  area_ratio : ℝ
  -- The sum of AB and CD is 280
  sum_sides : ab + cd = 280
  -- The ratio of areas is 5:2
  ratio_constraint : area_ratio = 5 / 2

/-- Theorem: In a trapezoid with given properties, AB = 200 -/
theorem trapezoid_ab_length (t : Trapezoid) : t.ab = 200 := by
  sorry


end NUMINAMATH_CALUDE_trapezoid_ab_length_l803_80348


namespace NUMINAMATH_CALUDE_intersection_circles_properties_l803_80306

/-- Given two circles O₁ and O₂ with equations x² + y² - 2x = 0 and x² + y² + 2x - 4y = 0 respectively,
    prove that their intersection points A and B satisfy:
    1. The line AB has equation x - y = 0
    2. The perpendicular bisector of AB has equation x + y - 1 = 0 -/
theorem intersection_circles_properties (x y : ℝ) :
  let O₁ := {(x, y) | x^2 + y^2 - 2*x = 0}
  let O₂ := {(x, y) | x^2 + y^2 + 2*x - 4*y = 0}
  let A := (x₀, y₀)
  let B := (x₁, y₁)
  ∀ x₀ y₀ x₁ y₁,
    (x₀, y₀) ∈ O₁ ∧ (x₀, y₀) ∈ O₂ ∧
    (x₁, y₁) ∈ O₁ ∧ (x₁, y₁) ∈ O₂ ∧
    (x₀, y₀) ≠ (x₁, y₁) →
    (x - y = 0 ↔ ∃ t, x = (1-t)*x₀ + t*x₁ ∧ y = (1-t)*y₀ + t*y₁) ∧
    (x + y - 1 = 0 ↔ (x - 1)^2 + y^2 = (x + 1)^2 + (y - 2)^2) :=
by sorry

end NUMINAMATH_CALUDE_intersection_circles_properties_l803_80306


namespace NUMINAMATH_CALUDE_complex_square_ratio_real_l803_80368

theorem complex_square_ratio_real (z₁ z₂ : ℂ) (hz₁ : z₁ ≠ 0) (hz₂ : z₂ ≠ 0) 
  (h : Complex.abs (z₁ + z₂) = Complex.abs (z₁ - z₂)) : 
  ∃ (r : ℝ), (z₁ / z₂)^2 = r := by
  sorry

end NUMINAMATH_CALUDE_complex_square_ratio_real_l803_80368


namespace NUMINAMATH_CALUDE_augmented_matrix_solution_l803_80372

/-- Given an augmented matrix and its solution, prove that c₁ - c₂ = -1 -/
theorem augmented_matrix_solution (c₁ c₂ : ℝ) : 
  (2 * 2 + 3 * 1 = c₁) → 
  (3 * 2 + 2 * 1 = c₂) → 
  c₁ - c₂ = -1 := by
sorry

end NUMINAMATH_CALUDE_augmented_matrix_solution_l803_80372


namespace NUMINAMATH_CALUDE_opposite_of_2023_l803_80314

theorem opposite_of_2023 : Int.neg 2023 = -2023 := by
  sorry

end NUMINAMATH_CALUDE_opposite_of_2023_l803_80314


namespace NUMINAMATH_CALUDE_apple_ratio_l803_80399

theorem apple_ratio (jim_apples jane_apples jerry_apples : ℕ) 
  (h1 : jim_apples = 20)
  (h2 : jane_apples = 60)
  (h3 : jerry_apples = 40) :
  (jim_apples + jane_apples + jerry_apples) / 3 / jim_apples = 2 := by
  sorry

end NUMINAMATH_CALUDE_apple_ratio_l803_80399


namespace NUMINAMATH_CALUDE_fraction_equals_d_minus_one_l803_80367

theorem fraction_equals_d_minus_one (n d : ℕ) (h : d ∣ n) :
  ∃ i : ℕ, i < n ∧ (i : ℚ) / (n - i : ℚ) = d - 1 := by
  sorry

end NUMINAMATH_CALUDE_fraction_equals_d_minus_one_l803_80367


namespace NUMINAMATH_CALUDE_consecutive_odd_integers_sum_and_product_l803_80356

theorem consecutive_odd_integers_sum_and_product :
  ∀ x : ℚ,
  (x + 4 = 4 * x) →
  (x + (x + 4) = 20 / 3) ∧
  (x * (x + 4) = 64 / 9) := by
  sorry

end NUMINAMATH_CALUDE_consecutive_odd_integers_sum_and_product_l803_80356


namespace NUMINAMATH_CALUDE_least_integer_satisfying_inequality_l803_80319

theorem least_integer_satisfying_inequality :
  ∃ (x : ℤ), (∀ (y : ℤ), |y^2 + 3*y + 10| ≤ 25 - y → x ≤ y) ∧
             |x^2 + 3*x + 10| ≤ 25 - x ∧
             x = -5 := by
  sorry

end NUMINAMATH_CALUDE_least_integer_satisfying_inequality_l803_80319


namespace NUMINAMATH_CALUDE_problem_solution_l803_80366

theorem problem_solution : (12 : ℝ) ^ 1 * 6 ^ 4 / 432 = 36 := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l803_80366


namespace NUMINAMATH_CALUDE_jack_and_toddlers_time_l803_80390

/-- The time it takes for Jack and his toddlers to get ready -/
def total_time (jack_shoe_time : ℕ) (toddler_extra_time : ℕ) (num_toddlers : ℕ) : ℕ :=
  jack_shoe_time + num_toddlers * (jack_shoe_time + toddler_extra_time)

/-- Theorem: The total time for Jack and his toddlers to get ready is 18 minutes -/
theorem jack_and_toddlers_time : total_time 4 3 2 = 18 := by
  sorry

end NUMINAMATH_CALUDE_jack_and_toddlers_time_l803_80390


namespace NUMINAMATH_CALUDE_round_37_396_to_nearest_tenth_l803_80371

/-- Represents a repeating decimal with an integer part and a repeating fractional part -/
structure RepeatingDecimal where
  integerPart : ℤ
  repeatingPart : ℕ

/-- Rounds a RepeatingDecimal to the nearest tenth -/
def roundToNearestTenth (x : RepeatingDecimal) : ℚ :=
  sorry

/-- The repeating decimal 37.396396... -/
def x : RepeatingDecimal :=
  { integerPart := 37, repeatingPart := 396 }

theorem round_37_396_to_nearest_tenth :
  roundToNearestTenth x = 37.4 := by
  sorry

end NUMINAMATH_CALUDE_round_37_396_to_nearest_tenth_l803_80371


namespace NUMINAMATH_CALUDE_gauss_family_mean_age_is_correct_l803_80331

/-- The mean age of the Gauss family children -/
def gauss_family_mean_age : ℚ :=
  let ages : List ℕ := [7, 7, 8, 14, 12, 15, 16]
  (ages.sum : ℚ) / ages.length

/-- Theorem stating that the mean age of the Gauss family children is 79/7 -/
theorem gauss_family_mean_age_is_correct : gauss_family_mean_age = 79 / 7 := by
  sorry

end NUMINAMATH_CALUDE_gauss_family_mean_age_is_correct_l803_80331


namespace NUMINAMATH_CALUDE_range_of_f_l803_80342

-- Define the function f
def f (x : ℝ) : ℝ := x - x^3

-- State the theorem
theorem range_of_f :
  ∃ (a b : ℝ), a = -6 ∧ b = 2 * Real.sqrt 3 / 9 ∧
  (∀ y, (∃ x ∈ Set.Icc 0 2, f x = y) ↔ a ≤ y ∧ y ≤ b) :=
by sorry

end NUMINAMATH_CALUDE_range_of_f_l803_80342


namespace NUMINAMATH_CALUDE_remainder_theorem_l803_80332

-- Define the polynomial
def p (x : ℝ) : ℝ := x^4 - x^2 + 3*x + 4

-- State the theorem
theorem remainder_theorem :
  ∃ q : ℝ → ℝ, ∀ x : ℝ, p x = (x + 2) * q x + 10 := by
  sorry

end NUMINAMATH_CALUDE_remainder_theorem_l803_80332


namespace NUMINAMATH_CALUDE_photo_arrangement_count_l803_80351

/-- The number of ways to arrange 5 people in a row for a photo, with one person fixed in the middle -/
def photo_arrangements : ℕ := 24

/-- The number of people in the photo -/
def total_people : ℕ := 5

/-- The number of people who can be arranged in non-middle positions -/
def non_middle_people : ℕ := total_people - 1

theorem photo_arrangement_count : photo_arrangements = non_middle_people! := by
  sorry

end NUMINAMATH_CALUDE_photo_arrangement_count_l803_80351


namespace NUMINAMATH_CALUDE_interval_of_decrease_l803_80305

-- Define the function f(x)
def f (x : ℝ) : ℝ := x^3 - 15*x^2 - 33*x + 6

-- Define the derivative of f(x)
def f' (x : ℝ) : ℝ := 3*x^2 - 30*x - 33

-- Theorem statement
theorem interval_of_decrease :
  ∀ x ∈ Set.Ioo (-1 : ℝ) 11, f' x < 0 ∧
  ∀ y ∈ Set.Iic (-1 : ℝ) ∪ Set.Ici 11, f' y ≥ 0 :=
sorry

end NUMINAMATH_CALUDE_interval_of_decrease_l803_80305


namespace NUMINAMATH_CALUDE_sqrt_sum_equals_two_l803_80384

theorem sqrt_sum_equals_two : 
  Real.sqrt (4 + 2 * Real.sqrt 3) + Real.sqrt (4 - 2 * Real.sqrt 3) = 2 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_sum_equals_two_l803_80384


namespace NUMINAMATH_CALUDE_cube_root_of_three_cubes_of_three_to_fifth_l803_80389

theorem cube_root_of_three_cubes_of_three_to_fifth (x : ℝ) : 
  x = (3^5 + 3^5 + 3^5)^(1/3) → x = 9 := by
  sorry

end NUMINAMATH_CALUDE_cube_root_of_three_cubes_of_three_to_fifth_l803_80389


namespace NUMINAMATH_CALUDE_percentage_subtraction_l803_80370

theorem percentage_subtraction (total : ℕ) (difference : ℕ) : 
  total = 7000 →
  difference = 700 →
  ∃ (p : ℚ), (1 / 10 : ℚ) * total - p * total = difference ∧ p = 0 :=
by sorry

end NUMINAMATH_CALUDE_percentage_subtraction_l803_80370


namespace NUMINAMATH_CALUDE_characterization_of_M_inequality_for_product_one_l803_80373

-- Define the function f
def f (x : ℝ) : ℝ := |2*x + 1| - |x - 2|

-- Define the set M
def M : Set ℝ := {x | f x ≤ 2}

-- Theorem 1: Characterization of set M
theorem characterization_of_M : M = {x | -5 ≤ x ∧ x ≤ 1} := by sorry

-- Theorem 2: Inequality for positive numbers with product 1
theorem inequality_for_product_one (a b c : ℝ) 
  (ha : a > 0) (hb : b > 0) (hc : c > 0) (h_prod : a * b * c = 1) :
  Real.sqrt a + Real.sqrt b + Real.sqrt c ≤ 1/a + 1/b + 1/c := by sorry

end NUMINAMATH_CALUDE_characterization_of_M_inequality_for_product_one_l803_80373


namespace NUMINAMATH_CALUDE_min_copies_discount_proof_l803_80334

/-- The minimum number of photocopies required for a discount -/
def min_copies_for_discount : ℕ := 160

/-- The cost of one photocopy in dollars -/
def cost_per_copy : ℚ := 2 / 100

/-- The discount rate offered -/
def discount_rate : ℚ := 25 / 100

/-- The total savings when ordering 160 copies -/
def total_savings : ℚ := 80 / 100

theorem min_copies_discount_proof :
  (min_copies_for_discount : ℚ) * cost_per_copy * (1 - discount_rate) =
  (min_copies_for_discount : ℚ) * cost_per_copy - total_savings :=
by sorry

end NUMINAMATH_CALUDE_min_copies_discount_proof_l803_80334


namespace NUMINAMATH_CALUDE_bleached_towel_breadth_decrease_l803_80369

/-- Represents the properties of a towel before and after bleaching. -/
structure Towel where
  length_decrease : ℝ
  area_decrease : ℝ
  breadth_decrease : ℝ

/-- Theorem stating the relationship between length, area, and breadth decrease for a bleached towel. -/
theorem bleached_towel_breadth_decrease (t : Towel) 
  (h1 : t.length_decrease = 0.3)
  (h2 : t.area_decrease = 0.475) :
  t.breadth_decrease = 0.25 := by
  sorry

#check bleached_towel_breadth_decrease

end NUMINAMATH_CALUDE_bleached_towel_breadth_decrease_l803_80369


namespace NUMINAMATH_CALUDE_sum_thirteen_is_156_l803_80350

/-- An arithmetic sequence with specific properties -/
structure ArithmeticSequence where
  a : ℕ → ℝ  -- The sequence
  d : ℝ      -- Common difference
  sum : ℕ → ℝ -- Sum function
  is_arithmetic : ∀ n, a (n + 1) = a n + d
  sum_formula : ∀ n, sum n = n * (2 * a 1 + (n - 1) * d) / 2
  third_sum : sum 3 = 6
  specific_sum : a 9 + a 11 + a 13 = 60

/-- The sum of the first 13 terms of the arithmetic sequence is 156 -/
theorem sum_thirteen_is_156 (seq : ArithmeticSequence) : seq.sum 13 = 156 := by
  sorry

end NUMINAMATH_CALUDE_sum_thirteen_is_156_l803_80350


namespace NUMINAMATH_CALUDE_exists_k_sum_of_digits_equal_l803_80374

/-- Sum of digits function -/
def sumOfDigits (n : ℕ) : ℕ := sorry

/-- Predicate to check if a number contains the digit 9 -/
def hasNoNine (n : ℕ) : Prop := sorry

/-- Main theorem -/
theorem exists_k_sum_of_digits_equal : 
  ∃ k : ℕ, k > 0 ∧ hasNoNine k ∧ sumOfDigits k = sumOfDigits (2^(24^2017) * k) := by sorry

end NUMINAMATH_CALUDE_exists_k_sum_of_digits_equal_l803_80374


namespace NUMINAMATH_CALUDE_figure_squares_l803_80339

-- Define the sequence function
def f (n : ℕ) : ℕ := 2 * n^2 + 2 * n + 1

-- State the theorem
theorem figure_squares (n : ℕ) : 
  f 0 = 1 ∧ f 1 = 5 ∧ f 2 = 13 ∧ f 3 = 25 → f 100 = 20201 := by
  sorry


end NUMINAMATH_CALUDE_figure_squares_l803_80339


namespace NUMINAMATH_CALUDE_employee_relocation_l803_80311

theorem employee_relocation (E : ℝ) 
  (prefer_Y : ℝ) (prefer_X : ℝ) (max_preferred : ℝ) 
  (h1 : prefer_Y = 0.4 * E)
  (h2 : prefer_X = 0.6 * E)
  (h3 : max_preferred = 140)
  (h4 : prefer_Y + prefer_X = max_preferred) :
  prefer_X / E = 0.6 := by
sorry

end NUMINAMATH_CALUDE_employee_relocation_l803_80311


namespace NUMINAMATH_CALUDE_quadratic_distinct_roots_l803_80394

theorem quadratic_distinct_roots (c : ℝ) :
  (∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ x₁^2 + 2*x₁ + 4*c = 0 ∧ x₂^2 + 2*x₂ + 4*c = 0) ↔ c < (1/4 : ℝ) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_distinct_roots_l803_80394


namespace NUMINAMATH_CALUDE_santa_claus_candy_distribution_l803_80343

theorem santa_claus_candy_distribution :
  ∃ (n b g c m : ℕ),
    n = b + g ∧
    n > 0 ∧
    b * c + g * (c + 1) = 47 ∧
    b * (m + 1) + g * m = 74 ∧
    n = 11 :=
by sorry

end NUMINAMATH_CALUDE_santa_claus_candy_distribution_l803_80343


namespace NUMINAMATH_CALUDE_expression_factorization_l803_80308

theorem expression_factorization (x y : ℝ) :
  (3 * x^3 + 28 * x^2 * y + 4 * x) - (-4 * x^3 + 5 * x^2 * y - 4 * x) = x * (x + 8) * (7 * x + 1) := by
  sorry

end NUMINAMATH_CALUDE_expression_factorization_l803_80308


namespace NUMINAMATH_CALUDE_min_value_of_expression_min_value_attained_l803_80364

theorem min_value_of_expression (x : ℝ) :
  (15 - x) * (9 - x) * (15 + x) * (9 + x) ≥ -5184 :=
by sorry

theorem min_value_attained :
  ∃ x : ℝ, (15 - x) * (9 - x) * (15 + x) * (9 + x) = -5184 :=
by sorry

end NUMINAMATH_CALUDE_min_value_of_expression_min_value_attained_l803_80364


namespace NUMINAMATH_CALUDE_expression_simplification_l803_80377

theorem expression_simplification (a : ℕ) (h : a = 2023) :
  (a^3 - 2*a^2*(a+1) + 3*a*(a+1)^2 - (a+1)^3 + 2) / (a*(a+1)) = a + 1 / (a*(a+1)) := by
  sorry

end NUMINAMATH_CALUDE_expression_simplification_l803_80377


namespace NUMINAMATH_CALUDE_magic_card_profit_theorem_l803_80325

/-- Calculates the profit from selling a Magic card that increases in value -/
def magic_card_profit (initial_price : ℝ) (value_multiplier : ℝ) : ℝ :=
  initial_price * value_multiplier - initial_price

/-- Theorem: The profit from selling a Magic card that triples in value from $100 is $200 -/
theorem magic_card_profit_theorem :
  magic_card_profit 100 3 = 200 := by
  sorry

end NUMINAMATH_CALUDE_magic_card_profit_theorem_l803_80325


namespace NUMINAMATH_CALUDE_third_quadrant_condition_l803_80379

-- Define the complex number z as a function of m
def z (m : ℝ) : ℂ := Complex.mk (m + 3) (m - 1)

-- Define what it means for a complex number to be in the third quadrant
def in_third_quadrant (w : ℂ) : Prop := w.re < 0 ∧ w.im < 0

-- The theorem statement
theorem third_quadrant_condition (m : ℝ) :
  in_third_quadrant (z m) ↔ m < -3 := by
  sorry

end NUMINAMATH_CALUDE_third_quadrant_condition_l803_80379


namespace NUMINAMATH_CALUDE_option_C_equals_nine_l803_80375

theorem option_C_equals_nine : 3 * 3 - 3 + 3 = 9 := by
  sorry

end NUMINAMATH_CALUDE_option_C_equals_nine_l803_80375


namespace NUMINAMATH_CALUDE_equal_division_of_sweets_and_candies_l803_80353

theorem equal_division_of_sweets_and_candies :
  let num_sweets : ℕ := 72
  let num_candies : ℕ := 56
  let num_people : ℕ := 4
  let sweets_per_person : ℕ := num_sweets / num_people
  let candies_per_person : ℕ := num_candies / num_people
  let total_per_person : ℕ := sweets_per_person + candies_per_person
  total_per_person = 32 := by
  sorry

end NUMINAMATH_CALUDE_equal_division_of_sweets_and_candies_l803_80353


namespace NUMINAMATH_CALUDE_train_speed_problem_l803_80321

/-- Proves that given a train journey of 5x km, where 4x km is traveled at 20 kmph,
    and the average speed for the entire journey is 40/3 kmph,
    the speed for the initial x km is 40/7 kmph. -/
theorem train_speed_problem (x : ℝ) (h : x > 0) :
  let total_distance : ℝ := 5 * x
  let second_leg_distance : ℝ := 4 * x
  let second_leg_speed : ℝ := 20
  let average_speed : ℝ := 40 / 3
  let initial_speed : ℝ := 40 / 7
  (total_distance / (x / initial_speed + second_leg_distance / second_leg_speed) = average_speed) :=
by
  sorry


end NUMINAMATH_CALUDE_train_speed_problem_l803_80321


namespace NUMINAMATH_CALUDE_angle_E_measure_l803_80324

-- Define the heptagon and its angles
structure Heptagon where
  A : ℝ
  B : ℝ
  C : ℝ
  D : ℝ
  E : ℝ
  F : ℝ
  G : ℝ

-- Define the properties of the heptagon
def is_valid_heptagon (h : Heptagon) : Prop :=
  h.A > 0 ∧ h.B > 0 ∧ h.C > 0 ∧ h.D > 0 ∧ h.E > 0 ∧ h.F > 0 ∧ h.G > 0 ∧
  h.A + h.B + h.C + h.D + h.E + h.F + h.G = 900

-- Define the conditions given in the problem
def satisfies_conditions (h : Heptagon) : Prop :=
  h.A = h.B ∧ h.A = h.C ∧ h.A = h.D ∧  -- A, B, C, D are congruent
  h.E = h.F ∧                          -- E and F are congruent
  h.A = h.E - 50 ∧                     -- A is 50° less than E
  h.G = 180 - h.E                      -- G is supplementary to E

-- The theorem to prove
theorem angle_E_measure (h : Heptagon) 
  (hvalid : is_valid_heptagon h) 
  (hcond : satisfies_conditions h) : 
  h.E = 184 := by
  sorry  -- The proof would go here


end NUMINAMATH_CALUDE_angle_E_measure_l803_80324


namespace NUMINAMATH_CALUDE_range_of_a_l803_80344

def p (a : ℝ) : Prop := ∀ x ∈ Set.Icc 1 2, x^2 - a ≥ 0

def q (a : ℝ) : Prop := ∃ x : ℝ, x^2 + (a-1)*x + 1 < 0

theorem range_of_a (a : ℝ) 
  (h1 : p a ∨ q a) 
  (h2 : ¬(p a ∧ q a)) : 
  (a ∈ Set.Icc (-1) 1) ∨ (a > 3) := by
  sorry

end NUMINAMATH_CALUDE_range_of_a_l803_80344


namespace NUMINAMATH_CALUDE_books_bought_at_fair_l803_80393

theorem books_bought_at_fair (initial_books final_books : ℕ) 
  (h1 : initial_books = 9)
  (h2 : final_books = 12) :
  final_books - initial_books = 3 := by
sorry

end NUMINAMATH_CALUDE_books_bought_at_fair_l803_80393


namespace NUMINAMATH_CALUDE_laundry_problem_solution_l803_80359

/-- Represents the laundromat problem setup -/
structure LaundryProblem where
  washer_cost : ℚ  -- Cost per washer load in dollars
  dryer_cost : ℚ   -- Cost per 10 minutes of dryer use in dollars
  wash_loads : ℕ   -- Number of wash loads
  num_dryers : ℕ   -- Number of dryers used
  total_spent : ℚ  -- Total amount spent in dollars

/-- Calculates the time each dryer ran in minutes -/
def dryer_time (p : LaundryProblem) : ℚ :=
  let washing_cost := p.washer_cost * p.wash_loads
  let drying_cost := p.total_spent - washing_cost
  let total_drying_time := (drying_cost / p.dryer_cost) * 10
  total_drying_time / p.num_dryers

/-- Theorem stating that for the given problem setup, each dryer ran for 40 minutes -/
theorem laundry_problem_solution (p : LaundryProblem) 
  (h1 : p.washer_cost = 4)
  (h2 : p.dryer_cost = 1/4)
  (h3 : p.wash_loads = 2)
  (h4 : p.num_dryers = 3)
  (h5 : p.total_spent = 11) :
  dryer_time p = 40 := by
  sorry


end NUMINAMATH_CALUDE_laundry_problem_solution_l803_80359


namespace NUMINAMATH_CALUDE_two_lines_with_45_degree_angle_l803_80328

/-- Represents a line in 3D space -/
structure Line3D where
  -- Add necessary fields

/-- Represents a plane in 3D space -/
structure Plane3D where
  -- Add necessary fields

/-- Represents a point in 3D space -/
structure Point3D where
  -- Add necessary fields

/-- Calculates the angle between a line and a plane -/
def angle_line_plane (l : Line3D) (p : Plane3D) : Real :=
  sorry

/-- Calculates the angle between two lines -/
def angle_between_lines (l1 l2 : Line3D) : Real :=
  sorry

/-- Checks if a line passes through a point -/
def line_passes_through (l : Line3D) (p : Point3D) : Prop :=
  sorry

theorem two_lines_with_45_degree_angle 
  (a : Line3D) (α : Plane3D) (P : Point3D) 
  (h : angle_line_plane a α = 30) : 
  ∃! (l1 l2 : Line3D), 
    l1 ≠ l2 ∧
    line_passes_through l1 P ∧
    line_passes_through l2 P ∧
    angle_between_lines l1 a = 45 ∧
    angle_between_lines l2 a = 45 ∧
    angle_line_plane l1 α = 45 ∧
    angle_line_plane l2 α = 45 ∧
    (∀ l : Line3D, 
      line_passes_through l P ∧ 
      angle_between_lines l a = 45 ∧ 
      angle_line_plane l α = 45 → 
      l = l1 ∨ l = l2) :=
by
  sorry

end NUMINAMATH_CALUDE_two_lines_with_45_degree_angle_l803_80328


namespace NUMINAMATH_CALUDE_purple_part_length_l803_80381

/-- The length of the purple part of a pencil -/
def purple_length : ℝ := 1.5

/-- The length of the black part of a pencil -/
def black_length : ℝ := 0.5

/-- The length of the blue part of a pencil -/
def blue_length : ℝ := 2

/-- The total length of the pencil -/
def total_length : ℝ := 4

/-- Theorem stating that the length of the purple part of the pencil is 1.5 cm -/
theorem purple_part_length :
  purple_length = total_length - (black_length + blue_length) :=
by sorry

end NUMINAMATH_CALUDE_purple_part_length_l803_80381


namespace NUMINAMATH_CALUDE_jeff_fills_ten_boxes_l803_80363

/-- The number of boxes Jeff can fill with his donuts -/
def boxes_filled (donuts_per_day : ℕ) (days : ℕ) (jeff_eats_per_day : ℕ) (chris_eats : ℕ) (donuts_per_box : ℕ) : ℕ :=
  ((donuts_per_day * days) - (jeff_eats_per_day * days) - chris_eats) / donuts_per_box

/-- Proof that Jeff can fill 10 boxes with his donuts -/
theorem jeff_fills_ten_boxes :
  boxes_filled 10 12 1 8 10 = 10 := by
  sorry

end NUMINAMATH_CALUDE_jeff_fills_ten_boxes_l803_80363


namespace NUMINAMATH_CALUDE_inequality_proof_l803_80358

theorem inequality_proof (a b : ℝ) (ha : a > 0) (hb : b > 0) (h : a^3 + b^3 = 2) :
  ((a + b) * (a^5 + b^5) ≥ 4) ∧ (a + b ≤ 2) := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l803_80358


namespace NUMINAMATH_CALUDE_rectangle_area_l803_80361

/-- A rectangle divided into three identical squares with a perimeter of 120 cm has an area of 675 square centimeters. -/
theorem rectangle_area (side : ℝ) : 
  (8 * side = 120) →  -- perimeter condition
  (3 * side * side = 675) -- area calculation
  := by sorry

end NUMINAMATH_CALUDE_rectangle_area_l803_80361


namespace NUMINAMATH_CALUDE_frequency_problem_l803_80354

theorem frequency_problem (sample_size : ℕ) (num_groups : ℕ) 
  (common_diff : ℚ) (last_seven_sum : ℚ) : 
  sample_size = 1000 →
  num_groups = 10 →
  common_diff = 0.05 →
  last_seven_sum = 0.79 →
  ∃ (x : ℚ), 
    x > 0 ∧ 
    x + common_diff > 0 ∧ 
    x + 2 * common_diff > 0 ∧
    x + (x + common_diff) + (x + 2 * common_diff) + last_seven_sum = 1 →
    (x * sample_size : ℚ) = 20 :=
by sorry

end NUMINAMATH_CALUDE_frequency_problem_l803_80354


namespace NUMINAMATH_CALUDE_mother_age_is_40_l803_80357

/-- The age of the mother -/
def mother_age : ℕ := sorry

/-- The sum of the ages of the 7 children -/
def children_ages_sum : ℕ := sorry

/-- The age of the mother is equal to the sum of the ages of her 7 children -/
axiom mother_age_eq_children_sum : mother_age = children_ages_sum

/-- After 20 years, the sum of the ages of the children will be three times the age of the mother -/
axiom future_age_relation : children_ages_sum + 7 * 20 = 3 * (mother_age + 20)

theorem mother_age_is_40 : mother_age = 40 := by sorry

end NUMINAMATH_CALUDE_mother_age_is_40_l803_80357


namespace NUMINAMATH_CALUDE_meaningful_expression_range_l803_80345

-- Define the condition for the expression to be meaningful
def is_meaningful (x : ℝ) : Prop := x > 3

-- Theorem statement
theorem meaningful_expression_range :
  ∀ x : ℝ, is_meaningful x ↔ x > 3 := by
  sorry

end NUMINAMATH_CALUDE_meaningful_expression_range_l803_80345


namespace NUMINAMATH_CALUDE_function_transformation_l803_80398

theorem function_transformation (f : ℝ → ℝ) (h : f 1 = 3) : 
  f 1 + 1 = 4 := by sorry

end NUMINAMATH_CALUDE_function_transformation_l803_80398


namespace NUMINAMATH_CALUDE_coefficient_x_cubed_in_expansion_l803_80329

theorem coefficient_x_cubed_in_expansion :
  let n : ℕ := 5
  let a : ℚ := 1
  let b : ℚ := 2
  let r : ℕ := 3
  let binomial_coeff := Nat.choose n r
  binomial_coeff * b^r = 80 := by
  sorry

end NUMINAMATH_CALUDE_coefficient_x_cubed_in_expansion_l803_80329


namespace NUMINAMATH_CALUDE_total_score_is_40_l803_80365

def game1_score : ℕ := 10
def game2_score : ℕ := 14
def game3_score : ℕ := 6

def first_three_games_total : ℕ := game1_score + game2_score + game3_score
def first_three_games_average : ℕ := first_three_games_total / 3
def game4_score : ℕ := first_three_games_average

def total_score : ℕ := first_three_games_total + game4_score

theorem total_score_is_40 : total_score = 40 := by
  sorry

end NUMINAMATH_CALUDE_total_score_is_40_l803_80365


namespace NUMINAMATH_CALUDE_negative_cube_root_of_negative_square_minus_one_l803_80326

theorem negative_cube_root_of_negative_square_minus_one (a : ℝ) :
  ∃ x : ℝ, x < 0 ∧ x^3 = -a^2 - 1 := by
  sorry

end NUMINAMATH_CALUDE_negative_cube_root_of_negative_square_minus_one_l803_80326


namespace NUMINAMATH_CALUDE_race_head_start_l803_80309

theorem race_head_start (Va Vb D H : ℝ) :
  Va = (30 / 17) * Vb →
  D / Va = (D - H) / Vb →
  H = (13 / 30) * D :=
by sorry

end NUMINAMATH_CALUDE_race_head_start_l803_80309


namespace NUMINAMATH_CALUDE_line_vector_proof_l803_80383

def line_vector (t : ℝ) : ℝ × ℝ × ℝ := sorry

theorem line_vector_proof :
  (line_vector 1 = (2, -3, 5) ∧ line_vector 4 = (-2, 9, -11)) →
  line_vector 5 = (-10/3, 13, -49/3) := by
  sorry

end NUMINAMATH_CALUDE_line_vector_proof_l803_80383


namespace NUMINAMATH_CALUDE_mixture_replacement_l803_80378

/-- Given a mixture of liquids A and B with an initial ratio of 4:1 and a final ratio of 2:3 after
    replacing some mixture with pure B, prove that 60 liters of mixture were replaced when the
    initial amount of liquid A was 48 liters. -/
theorem mixture_replacement (initial_A : ℝ) (initial_B : ℝ) (replaced : ℝ) :
  initial_A = 48 →
  initial_A / initial_B = 4 / 1 →
  initial_A / (initial_B + replaced) = 2 / 3 →
  replaced = 60 :=
by sorry

end NUMINAMATH_CALUDE_mixture_replacement_l803_80378


namespace NUMINAMATH_CALUDE_three_numbers_sum_l803_80304

theorem three_numbers_sum : ∀ (a b c : ℝ),
  a ≤ b ∧ b ≤ c →  -- Arrange numbers in ascending order
  b = 10 →  -- Median is 10
  (a + b + c) / 3 = a + 8 →  -- Mean is 8 more than least
  (a + b + c) / 3 = c - 20 →  -- Mean is 20 less than greatest
  a + b + c = 66 := by
sorry

end NUMINAMATH_CALUDE_three_numbers_sum_l803_80304


namespace NUMINAMATH_CALUDE_correct_ring_arrangements_l803_80310

/-- The number of ways to arrange rings on fingers -/
def ring_arrangements (total_rings : ℕ) (rings_used : ℕ) (fingers : ℕ) (max_per_finger : ℕ) : ℕ :=
  sorry

/-- Theorem stating the correct number of ring arrangements -/
theorem correct_ring_arrangements :
  ring_arrangements 10 6 5 3 = 145152000 :=
sorry

end NUMINAMATH_CALUDE_correct_ring_arrangements_l803_80310


namespace NUMINAMATH_CALUDE_find_M_l803_80322

theorem find_M : ∃ (M : ℕ+), (12^2 * 45^2 : ℕ) = 15^2 * M^2 ∧ M = 36 := by
  sorry

end NUMINAMATH_CALUDE_find_M_l803_80322


namespace NUMINAMATH_CALUDE_symmetry_center_l803_80387

open Real

/-- Given a function f and its symmetric function g, prove that (π/4, 0) is a center of symmetry of g -/
theorem symmetry_center (f g : ℝ → ℝ) : 
  (∀ x, f x = sin (2*x + π/6)) →
  (∀ x, f (π/6 - x) = g x) →
  (π/4, 0) ∈ {p : ℝ × ℝ | ∀ x, g (p.1 + x) = g (p.1 - x)} :=
by sorry

end NUMINAMATH_CALUDE_symmetry_center_l803_80387


namespace NUMINAMATH_CALUDE_complex_division_equality_l803_80316

theorem complex_division_equality : Complex.I * 2 / (1 - Complex.I) = -1 + Complex.I := by sorry

end NUMINAMATH_CALUDE_complex_division_equality_l803_80316


namespace NUMINAMATH_CALUDE_polynomial_remainder_l803_80391

def polynomial (x : ℝ) : ℝ := 6*x^8 - 2*x^7 - 10*x^6 + 3*x^4 + 5*x^3 - 15

def divisor (x : ℝ) : ℝ := 3*x - 6

theorem polynomial_remainder :
  ∃ (q : ℝ → ℝ), ∀ x, polynomial x = q x * divisor x + 713 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_remainder_l803_80391


namespace NUMINAMATH_CALUDE_f_range_l803_80327

def f (x : ℝ) : ℝ := 256 * x^9 - 576 * x^7 + 432 * x^5 - 120 * x^3 + 9 * x

theorem f_range :
  (∀ x ∈ Set.Icc (-1 : ℝ) 1, f x ∈ Set.Icc (-1 : ℝ) 1) ∧
  (∃ x₁ ∈ Set.Icc (-1 : ℝ) 1, f x₁ = -1) ∧
  (∃ x₂ ∈ Set.Icc (-1 : ℝ) 1, f x₂ = 1) :=
sorry

end NUMINAMATH_CALUDE_f_range_l803_80327


namespace NUMINAMATH_CALUDE_min_value_absolute_sum_l803_80355

theorem min_value_absolute_sum (x y : ℝ) : 
  |x - 1| + |x| + |y - 1| + |y + 1| ≥ 3 := by
  sorry

end NUMINAMATH_CALUDE_min_value_absolute_sum_l803_80355


namespace NUMINAMATH_CALUDE_sum_less_than_six_for_735_l803_80307

def is_less_than_six (n : ℕ) : Bool :=
  n < 6

def sum_less_than_six (cards : List ℕ) : ℕ :=
  (cards.filter is_less_than_six).sum

theorem sum_less_than_six_for_735 : 
  ∃ (cards : List ℕ), 
    cards.length = 3 ∧ 
    (∀ n ∈ cards, 1 ≤ n ∧ n ≤ 9) ∧
    cards.foldl (λ acc d => acc * 10 + d) 0 = 735 ∧
    sum_less_than_six cards = 8 :=
by
  sorry

end NUMINAMATH_CALUDE_sum_less_than_six_for_735_l803_80307


namespace NUMINAMATH_CALUDE_cosine_sum_17th_roots_l803_80362

theorem cosine_sum_17th_roots : 
  Real.cos (2 * Real.pi / 17) + Real.cos (6 * Real.pi / 17) + Real.cos (10 * Real.pi / 17) = (Real.sqrt 13 - 1) / 4 := by
  sorry

end NUMINAMATH_CALUDE_cosine_sum_17th_roots_l803_80362


namespace NUMINAMATH_CALUDE_cos_ninety_degrees_l803_80392

theorem cos_ninety_degrees : Real.cos (π / 2) = 0 := by
  sorry

end NUMINAMATH_CALUDE_cos_ninety_degrees_l803_80392


namespace NUMINAMATH_CALUDE_yahs_to_bahs_conversion_l803_80300

/-- Represents the number of bahs equivalent to 36 rahs -/
def bahs_per_36_rahs : ℕ := 24

/-- Represents the number of rahs equivalent to 18 yahs -/
def rahs_per_18_yahs : ℕ := 12

/-- Represents the number of yahs we want to convert to bahs -/
def yahs_to_convert : ℕ := 1500

/-- Theorem stating the equivalence between 1500 yahs and 667 bahs -/
theorem yahs_to_bahs_conversion :
  ∃ (bahs : ℕ), bahs = 667 ∧
  (bahs * bahs_per_36_rahs * rahs_per_18_yahs : ℚ) / 36 / 18 = yahs_to_convert / 1 :=
sorry

end NUMINAMATH_CALUDE_yahs_to_bahs_conversion_l803_80300


namespace NUMINAMATH_CALUDE_card_draw_probability_l803_80315

/-- A standard deck of cards -/
structure Deck :=
  (cards : Nat)

/-- The probability of drawing a specific sequence of cards -/
def draw_probability (d : Deck) (spades : Nat) (tens : Nat) (queens : Nat) : Rat :=
  let p1 := spades / d.cards
  let p2 := tens / (d.cards - 1)
  let p3 := queens / (d.cards - 2)
  p1 * p2 * p3

/-- The theorem to prove -/
theorem card_draw_probability :
  let d := Deck.mk 52
  let spades := 13
  let tens := 4
  let queens := 4
  draw_probability d spades tens queens = 17 / 11050 :=
by
  sorry


end NUMINAMATH_CALUDE_card_draw_probability_l803_80315


namespace NUMINAMATH_CALUDE_division_equality_l803_80323

theorem division_equality : 815472 / 6630 = 123 := by
  sorry

end NUMINAMATH_CALUDE_division_equality_l803_80323


namespace NUMINAMATH_CALUDE_class_average_theorem_l803_80385

theorem class_average_theorem (boy_percentage : ℝ) (girl_percentage : ℝ) 
  (boy_score : ℝ) (girl_score : ℝ) :
  boy_percentage = 0.4 →
  girl_percentage = 0.6 →
  boy_score = 80 →
  girl_score = 90 →
  boy_percentage + girl_percentage = 1 →
  boy_percentage * boy_score + girl_percentage * girl_score = 86 := by
sorry

end NUMINAMATH_CALUDE_class_average_theorem_l803_80385


namespace NUMINAMATH_CALUDE_largest_number_problem_l803_80380

theorem largest_number_problem (a b c : ℝ) 
  (h_order : a < b ∧ b < c)
  (h_sum : a + b + c = 75)
  (h_diff_large : c - b = 5)
  (h_diff_small : b - a = 4) :
  c = 89 / 3 := by
sorry

end NUMINAMATH_CALUDE_largest_number_problem_l803_80380


namespace NUMINAMATH_CALUDE_A_and_D_independent_l803_80317

def num_balls : ℕ := 5

def prob_A : ℚ := 1 / num_balls
def prob_B : ℚ := 1 / num_balls
def prob_C : ℚ := 3 / (num_balls * num_balls)
def prob_D : ℚ := 1 / num_balls

def prob_AD : ℚ := 1 / (num_balls * num_balls)

theorem A_and_D_independent : prob_AD = prob_A * prob_D := by sorry

end NUMINAMATH_CALUDE_A_and_D_independent_l803_80317


namespace NUMINAMATH_CALUDE_base_conversion_3500_to_base_7_l803_80338

theorem base_conversion_3500_to_base_7 :
  (1 * 7^4 + 3 * 7^3 + 1 * 7^2 + 3 * 7^1 + 0 * 7^0 : ℕ) = 3500 := by
  sorry

end NUMINAMATH_CALUDE_base_conversion_3500_to_base_7_l803_80338


namespace NUMINAMATH_CALUDE_hotel_loss_calculation_l803_80303

/-- Calculates the loss incurred by a hotel given its operations expenses and client payments --/
def hotel_loss (expenses : ℝ) (client_payment_ratio : ℝ) : ℝ :=
  expenses - (client_payment_ratio * expenses)

/-- Theorem: A hotel with $100 expenses and client payments of 3/4 of expenses incurs a $25 loss --/
theorem hotel_loss_calculation :
  hotel_loss 100 (3/4) = 25 := by
  sorry

end NUMINAMATH_CALUDE_hotel_loss_calculation_l803_80303


namespace NUMINAMATH_CALUDE_equation_solution_l803_80396

theorem equation_solution (n : ℚ) : 
  (2 / (n + 2) + 3 / (n + 2) + (2 * n) / (n + 2) = 4) → n = -3/2 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l803_80396


namespace NUMINAMATH_CALUDE_expression_evaluation_l803_80352

theorem expression_evaluation :
  let x : ℤ := -2
  (x^2 + 7*x - 8) = -18 := by sorry

end NUMINAMATH_CALUDE_expression_evaluation_l803_80352


namespace NUMINAMATH_CALUDE_function_properties_l803_80395

-- Define the function f(x)
def f (x : ℝ) : ℝ := |3*x + 3| - |x - 5|

-- Define the solution set M
def M : Set ℝ := {x | f x > 0}

-- State the theorem
theorem function_properties :
  (M = {x | x < -4 ∨ x > 1/2}) ∧
  (∃ m : ℝ, ∀ x : ℝ, f x ≥ m ∧ ∃ x₀ : ℝ, f x₀ = m) ∧
  (∀ a b c : ℝ, a > 0 → b > 0 → c > 0 → a + b + c = 6 →
    1 / (a + b) + 1 / (b + c) + 1 / (c + a) ≥ 3/4) := by
  sorry

end NUMINAMATH_CALUDE_function_properties_l803_80395


namespace NUMINAMATH_CALUDE_range_of_x_when_a_is_one_range_of_a_l803_80347

-- Define the conditions
def p (a x : ℝ) : Prop := x^2 - 5*a*x + 4*a^2 < 0
def q (x : ℝ) : Prop := 2 < x ∧ x ≤ 5

-- Part 1
theorem range_of_x_when_a_is_one :
  ∀ x : ℝ, p 1 x ∧ q x → 2 < x ∧ x < 4 := by sorry

-- Part 2
theorem range_of_a :
  (∀ x : ℝ, p a x → q x) ∧ (∃ x : ℝ, q x ∧ ¬(p a x)) →
  5/4 < a ∧ a ≤ 2 := by sorry

#check range_of_x_when_a_is_one
#check range_of_a

end NUMINAMATH_CALUDE_range_of_x_when_a_is_one_range_of_a_l803_80347


namespace NUMINAMATH_CALUDE_randy_piggy_bank_theorem_l803_80397

/-- Calculates the amount in Randy's piggy bank after a year -/
def piggy_bank_after_year (initial_amount : ℕ) (store_trip_cost : ℕ) (store_trips_per_month : ℕ)
  (internet_bill : ℕ) (extra_cost_third_trip : ℕ) (weekly_earnings : ℕ) (birthday_gift : ℕ) : ℕ :=
  let months_in_year : ℕ := 12
  let weeks_in_year : ℕ := 52
  let regular_store_expenses := store_trip_cost * store_trips_per_month * months_in_year
  let extra_expenses := extra_cost_third_trip * (months_in_year / 3)
  let internet_expenses := internet_bill * months_in_year
  let job_income := weekly_earnings * weeks_in_year
  let total_expenses := regular_store_expenses + extra_expenses + internet_expenses
  let total_income := job_income + birthday_gift
  initial_amount + total_income - total_expenses

theorem randy_piggy_bank_theorem :
  piggy_bank_after_year 200 2 4 20 1 15 100 = 740 := by
  sorry

end NUMINAMATH_CALUDE_randy_piggy_bank_theorem_l803_80397


namespace NUMINAMATH_CALUDE_find_a_value_l803_80336

def f (x a : ℝ) : ℝ := |x + 1| + |x - a|

theorem find_a_value (a : ℝ) (h1 : a > 0) 
  (h2 : ∀ x, f x a ≥ 5 ↔ x ≤ -2 ∨ x > 3) : a = 2 := by
  sorry

end NUMINAMATH_CALUDE_find_a_value_l803_80336


namespace NUMINAMATH_CALUDE_sin_pi_plus_alpha_l803_80349

theorem sin_pi_plus_alpha (α : Real) :
  (∃ (x y : Real), x = Real.sqrt 5 ∧ y = -2 ∧ x = Real.cos α * Real.sqrt (x^2 + y^2) ∧ y = Real.sin α * Real.sqrt (x^2 + y^2)) →
  Real.sin (Real.pi + α) = 2/3 := by
  sorry

end NUMINAMATH_CALUDE_sin_pi_plus_alpha_l803_80349

import Mathlib

namespace NUMINAMATH_CALUDE_divisible_by_24_l3532_353243

theorem divisible_by_24 (n : ℤ) : ∃ k : ℤ, n * (n + 2) * (5 * n - 1) * (5 * n + 1) = 24 * k := by
  sorry

end NUMINAMATH_CALUDE_divisible_by_24_l3532_353243


namespace NUMINAMATH_CALUDE_quadratic_two_roots_l3532_353279

theorem quadratic_two_roots (b : ℝ) : 
  ∃ (x₁ x₂ : ℝ), x₁ ≠ x₂ ∧ 
  (∀ x : ℝ, x^2 + b*x - 3 = 0 ↔ x = x₁ ∨ x = x₂) := by
sorry

end NUMINAMATH_CALUDE_quadratic_two_roots_l3532_353279


namespace NUMINAMATH_CALUDE_shopping_expense_l3532_353294

theorem shopping_expense (initial_amount : ℝ) (amount_left : ℝ) : 
  initial_amount = 158 →
  amount_left = 78 →
  ∃ (shoes_price bag_price lunch_price : ℝ),
    bag_price = shoes_price - 17 ∧
    lunch_price = bag_price / 4 ∧
    initial_amount = shoes_price + bag_price + lunch_price + amount_left ∧
    shoes_price = 45 := by
  sorry

end NUMINAMATH_CALUDE_shopping_expense_l3532_353294


namespace NUMINAMATH_CALUDE_unique_solution_modular_equation_l3532_353260

theorem unique_solution_modular_equation :
  ∃! n : ℤ, 0 ≤ n ∧ n < 103 ∧ (99 * n) % 103 = 72 % 103 ∧ n = 52 := by
  sorry

end NUMINAMATH_CALUDE_unique_solution_modular_equation_l3532_353260


namespace NUMINAMATH_CALUDE_flower_shop_utilities_percentage_l3532_353208

/-- Calculates the percentage of rent paid for utilities in James' flower shop --/
theorem flower_shop_utilities_percentage
  (weekly_rent : ℝ)
  (store_hours_per_day : ℝ)
  (store_days_per_week : ℝ)
  (employees_per_shift : ℝ)
  (employee_hourly_wage : ℝ)
  (total_weekly_expenses : ℝ)
  (h1 : weekly_rent = 1200)
  (h2 : store_hours_per_day = 16)
  (h3 : store_days_per_week = 5)
  (h4 : employees_per_shift = 2)
  (h5 : employee_hourly_wage = 12.5)
  (h6 : total_weekly_expenses = 3440)
  : (((total_weekly_expenses - (store_hours_per_day * store_days_per_week * employees_per_shift * employee_hourly_wage)) - weekly_rent) / weekly_rent) * 100 = 20 := by
  sorry

end NUMINAMATH_CALUDE_flower_shop_utilities_percentage_l3532_353208


namespace NUMINAMATH_CALUDE_john_needs_more_money_l3532_353230

/-- Given that John needs $2.5 in total and has $0.75, prove that he needs $1.75 more. -/
theorem john_needs_more_money (total_needed : ℝ) (amount_has : ℝ) 
  (h1 : total_needed = 2.5)
  (h2 : amount_has = 0.75) :
  total_needed - amount_has = 1.75 := by
sorry

end NUMINAMATH_CALUDE_john_needs_more_money_l3532_353230


namespace NUMINAMATH_CALUDE_max_n_value_l3532_353207

theorem max_n_value (a b c d : ℝ) (n : ℕ) 
  (h1 : a > b) (h2 : b > c) (h3 : c > d)
  (h4 : (1 / (a - b)) + (1 / (b - c)) + (1 / (c - d)) ≥ n / (a - d)) :
  n ≤ 9 :=
sorry

end NUMINAMATH_CALUDE_max_n_value_l3532_353207


namespace NUMINAMATH_CALUDE_num_non_mult_6_divisors_l3532_353211

/-- The smallest integer satisfying the given conditions -/
def m : ℕ :=
  2^3 * 3^4 * 5^6

/-- m/2 is a perfect square -/
axiom m_div_2_is_square : ∃ k : ℕ, m / 2 = k^2

/-- m/3 is a perfect cube -/
axiom m_div_3_is_cube : ∃ k : ℕ, m / 3 = k^3

/-- m/5 is a perfect fifth -/
axiom m_div_5_is_fifth : ∃ k : ℕ, m / 5 = k^5

/-- The number of divisors of m -/
def num_divisors (n : ℕ) : ℕ :=
  (Finset.filter (· ∣ n) (Finset.range (n + 1))).card

/-- The number of divisors of m that are multiples of 6 -/
def num_divisors_mult_6 (n : ℕ) : ℕ :=
  (Finset.filter (λ x => x ∣ n ∧ 6 ∣ x) (Finset.range (n + 1))).card

/-- The main theorem -/
theorem num_non_mult_6_divisors :
    num_divisors m - num_divisors_mult_6 m = 56 := by
  sorry

end NUMINAMATH_CALUDE_num_non_mult_6_divisors_l3532_353211


namespace NUMINAMATH_CALUDE_polynomial_not_equal_77_l3532_353254

theorem polynomial_not_equal_77 (x y : ℤ) : 
  x^5 - 4*x^4*y - 5*y^2*x^3 + 20*y^3*x^2 + 4*y^4*x - 16*y^5 ≠ 77 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_not_equal_77_l3532_353254


namespace NUMINAMATH_CALUDE_a_investment_l3532_353297

/-- Calculates the investment of partner A in a business partnership --/
def calculate_investment_A (investment_B investment_C total_profit profit_share_A : ℚ) : ℚ :=
  let total_investment := investment_B + investment_C + profit_share_A * (investment_B + investment_C) / (total_profit - profit_share_A)
  profit_share_A * total_investment / total_profit

/-- Theorem stating that A's investment is 6300 given the problem conditions --/
theorem a_investment (investment_B investment_C total_profit profit_share_A : ℚ)
  (hB : investment_B = 4200)
  (hC : investment_C = 10500)
  (hProfit : total_profit = 12100)
  (hShareA : profit_share_A = 3630) :
  calculate_investment_A investment_B investment_C total_profit profit_share_A = 6300 := by
  sorry

#eval calculate_investment_A 4200 10500 12100 3630

end NUMINAMATH_CALUDE_a_investment_l3532_353297


namespace NUMINAMATH_CALUDE_largest_gcd_of_sum_1023_l3532_353241

theorem largest_gcd_of_sum_1023 :
  ∃ (c d : ℕ+), c + d = 1023 ∧
  ∀ (a b : ℕ+), a + b = 1023 → Nat.gcd a b ≤ Nat.gcd c d ∧
  Nat.gcd c d = 341 :=
sorry

end NUMINAMATH_CALUDE_largest_gcd_of_sum_1023_l3532_353241


namespace NUMINAMATH_CALUDE_strawberry_jelly_amount_l3532_353205

/-- The amount of strawberry jelly in grams -/
def strawberry_jelly : ℕ := sorry

/-- The amount of blueberry jelly in grams -/
def blueberry_jelly : ℕ := 4518

/-- The total amount of jelly in grams -/
def total_jelly : ℕ := 6310

/-- Theorem stating that the amount of strawberry jelly is 1792 grams -/
theorem strawberry_jelly_amount : strawberry_jelly = 1792 :=
  by
    sorry

/-- Lemma stating that the sum of strawberry and blueberry jelly equals the total jelly -/
lemma jelly_sum : strawberry_jelly + blueberry_jelly = total_jelly :=
  by
    sorry

end NUMINAMATH_CALUDE_strawberry_jelly_amount_l3532_353205


namespace NUMINAMATH_CALUDE_bug_path_tiles_l3532_353215

/-- The number of tiles a bug visits when walking diagonally across a rectangular floor -/
def tiles_visited (width length : ℕ) : ℕ :=
  width + length - Nat.gcd width length

theorem bug_path_tiles : tiles_visited 12 18 = 24 := by
  sorry

end NUMINAMATH_CALUDE_bug_path_tiles_l3532_353215


namespace NUMINAMATH_CALUDE_alice_additional_spend_l3532_353255

/-- Represents the grocery store cart with various items and their prices -/
structure GroceryCart where
  chicken : Float
  lettuce : Float
  cherryTomatoes : Float
  sweetPotatoes : Float
  broccoli : Float
  brusselSprouts : Float
  strawberries : Float
  cereal : Float
  groundBeef : Float

/-- Calculates the pre-tax total of the grocery cart -/
def calculatePreTaxTotal (cart : GroceryCart) : Float :=
  cart.chicken + cart.lettuce + cart.cherryTomatoes + cart.sweetPotatoes +
  cart.broccoli + cart.brusselSprouts + cart.strawberries + cart.cereal +
  cart.groundBeef

/-- Theorem: The difference between the minimum spend for free delivery and
    Alice's pre-tax total is $3.02 -/
theorem alice_additional_spend (minSpend : Float) (cart : GroceryCart)
    (h1 : minSpend = 50.00)
    (h2 : cart.chicken = 10.80)
    (h3 : cart.lettuce = 3.50)
    (h4 : cart.cherryTomatoes = 5.00)
    (h5 : cart.sweetPotatoes = 3.75)
    (h6 : cart.broccoli = 6.00)
    (h7 : cart.brusselSprouts = 2.50)
    (h8 : cart.strawberries = 4.80)
    (h9 : cart.cereal = 4.00)
    (h10 : cart.groundBeef = 5.63) :
    minSpend - calculatePreTaxTotal cart = 3.02 := by
  sorry

end NUMINAMATH_CALUDE_alice_additional_spend_l3532_353255


namespace NUMINAMATH_CALUDE_parabola_translation_l3532_353288

/-- Original parabola function -/
def f (x : ℝ) : ℝ := x^2 + 1

/-- Translated parabola function -/
def g (x : ℝ) : ℝ := x^2 + 4*x + 5

/-- Translation function -/
def translate (x : ℝ) : ℝ := x + 2

theorem parabola_translation :
  ∀ x : ℝ, g x = f (translate x) :=
by sorry

end NUMINAMATH_CALUDE_parabola_translation_l3532_353288


namespace NUMINAMATH_CALUDE_calculate_gross_profit_l3532_353282

/-- Calculate the gross profit given the sales price, gross profit margin, sales tax, and initial discount --/
theorem calculate_gross_profit (sales_price : ℝ) (gross_profit_margin : ℝ) (sales_tax : ℝ) (initial_discount : ℝ) :
  sales_price = 81 →
  gross_profit_margin = 1.7 →
  sales_tax = 0.07 →
  initial_discount = 0.15 →
  ∃ (gross_profit : ℝ), abs (gross_profit - 56.07) < 0.01 := by
  sorry


end NUMINAMATH_CALUDE_calculate_gross_profit_l3532_353282


namespace NUMINAMATH_CALUDE_floor_plus_self_unique_solution_l3532_353214

theorem floor_plus_self_unique_solution : 
  ∃! r : ℝ, (⌊r⌋ : ℝ) + r = 18.75 := by sorry

end NUMINAMATH_CALUDE_floor_plus_self_unique_solution_l3532_353214


namespace NUMINAMATH_CALUDE_cost_formula_l3532_353239

def cost (P : ℕ) : ℕ :=
  15 + 4 * (P - 1) - 10 * (if P > 5 then 1 else 0)

theorem cost_formula (P : ℕ) :
  cost P = 15 + 4 * (P - 1) - 10 * (if P > 5 then 1 else 0) :=
by sorry

end NUMINAMATH_CALUDE_cost_formula_l3532_353239


namespace NUMINAMATH_CALUDE_power_of_power_three_l3532_353283

theorem power_of_power_three : (3^2)^4 = 6561 := by
  sorry

end NUMINAMATH_CALUDE_power_of_power_three_l3532_353283


namespace NUMINAMATH_CALUDE_triangle_shape_l3532_353237

theorem triangle_shape (a b : ℝ) (A B C : ℝ) (h_triangle : A + B + C = π) 
  (h_positive : 0 < a ∧ 0 < b) (h_condition : a * Real.cos A = b * Real.cos B) :
  A = B ∨ A + B = π / 2 := by sorry

end NUMINAMATH_CALUDE_triangle_shape_l3532_353237


namespace NUMINAMATH_CALUDE_square_sum_pattern_l3532_353280

theorem square_sum_pattern : 
  (1^2 + 3^2 = 10) → (2^2 + 4^2 = 20) → (3^2 + 5^2 = 34) → (4^2 + 6^2 = 52) := by
  sorry

end NUMINAMATH_CALUDE_square_sum_pattern_l3532_353280


namespace NUMINAMATH_CALUDE_negation_equivalence_l3532_353245

theorem negation_equivalence :
  ¬(∀ (x : ℝ), ∃ (n : ℕ+), (n : ℝ) ≥ x) ↔ 
  ∃ (x : ℝ), ∀ (n : ℕ+), (n : ℝ) < x^2 :=
sorry

end NUMINAMATH_CALUDE_negation_equivalence_l3532_353245


namespace NUMINAMATH_CALUDE_tangent_point_coordinates_l3532_353291

/-- The curve y = x^4 - x -/
def f (x : ℝ) : ℝ := x^4 - x

/-- The derivative of f -/
def f' (x : ℝ) : ℝ := 4 * x^3 - 1

theorem tangent_point_coordinates :
  ∀ x y : ℝ,
  y = f x →
  f' x = 3 →
  x = 1 ∧ y = 0 :=
by sorry

end NUMINAMATH_CALUDE_tangent_point_coordinates_l3532_353291


namespace NUMINAMATH_CALUDE_max_both_writers_and_editors_l3532_353284

/-- Conference attendees -/
structure Conference where
  total : ℕ
  writers : ℕ
  editors : ℕ
  both : ℕ
  neither : ℕ

/-- Conference constraints -/
def valid_conference (c : Conference) : Prop :=
  c.total = 100 ∧
  c.writers = 35 ∧
  c.editors > 38 ∧
  c.neither = 2 * c.both ∧
  c.total = c.writers + c.editors - c.both + c.neither

/-- Theorem: The maximum number of people who can be both writers and editors is 26 -/
theorem max_both_writers_and_editors (c : Conference) (h : valid_conference c) :
  c.both ≤ 26 := by
  sorry

end NUMINAMATH_CALUDE_max_both_writers_and_editors_l3532_353284


namespace NUMINAMATH_CALUDE_monitor_pixel_count_l3532_353277

/-- Calculates the total number of pixels on a monitor given its dimensions and pixel density. -/
def total_pixels (width : ℕ) (height : ℕ) (pixel_density : ℕ) : ℕ :=
  (width * pixel_density) * (height * pixel_density)

/-- Theorem: A monitor that is 21 inches wide and 12 inches tall with a pixel density of 100 dots per inch has 2,520,000 pixels. -/
theorem monitor_pixel_count :
  total_pixels 21 12 100 = 2520000 := by
  sorry

end NUMINAMATH_CALUDE_monitor_pixel_count_l3532_353277


namespace NUMINAMATH_CALUDE_first_year_growth_rate_l3532_353290

/-- Proves that given the initial and final populations, and the second year's growth rate,
    the first year's growth rate is 22%. -/
theorem first_year_growth_rate (initial_pop : ℕ) (final_pop : ℕ) (second_year_rate : ℚ) :
  initial_pop = 800 →
  final_pop = 1220 →
  second_year_rate = 25 / 100 →
  ∃ (first_year_rate : ℚ),
    first_year_rate = 22 / 100 ∧
    final_pop = initial_pop * (1 + first_year_rate) * (1 + second_year_rate) :=
by sorry

end NUMINAMATH_CALUDE_first_year_growth_rate_l3532_353290


namespace NUMINAMATH_CALUDE_simplest_form_fraction_l3532_353265

/-- A fraction is in simplest form if its numerator and denominator have no common factors
    other than 1 and -1, and neither the numerator nor denominator can be factored further. -/
def IsSimplestForm (n d : ℝ → ℝ → ℝ) : Prop :=
  ∀ x y, (n x y ≠ 0 ∨ d x y ≠ 0) →
    ∀ f : ℝ → ℝ → ℝ, (f x y ∣ n x y) ∧ (f x y ∣ d x y) → f x y = 1 ∨ f x y = -1

/-- The fraction (x^2 + y^2) / (x + y) is in simplest form. -/
theorem simplest_form_fraction (x y : ℝ) :
    IsSimplestForm (fun x y => x^2 + y^2) (fun x y => x + y) := by
  sorry

#check simplest_form_fraction

end NUMINAMATH_CALUDE_simplest_form_fraction_l3532_353265


namespace NUMINAMATH_CALUDE_city_population_problem_l3532_353278

theorem city_population_problem (p : ℝ) : 
  0.85 * (p + 800) = p + 824 ↔ p = 960 :=
by sorry

end NUMINAMATH_CALUDE_city_population_problem_l3532_353278


namespace NUMINAMATH_CALUDE_divisors_of_power_minus_one_l3532_353268

/-- The number of distinct positive divisors of a positive integer -/
def num_divisors (n : ℕ) : ℕ := (Nat.divisors n).card

/-- Main theorem -/
theorem divisors_of_power_minus_one (a n : ℕ) (ha : a > 1) (hn : n > 0) 
  (h_prime : Nat.Prime (a^n + 1)) : num_divisors (a^n - 1) ≥ n := by
  sorry


end NUMINAMATH_CALUDE_divisors_of_power_minus_one_l3532_353268


namespace NUMINAMATH_CALUDE_hexagon_angle_sum_l3532_353201

theorem hexagon_angle_sum (x y : ℝ) : 
  x ≥ 0 → y ≥ 0 → 
  34 + 80 + 30 + 90 + x + y = 720 → 
  x + y = 36 := by
sorry

end NUMINAMATH_CALUDE_hexagon_angle_sum_l3532_353201


namespace NUMINAMATH_CALUDE_base3_to_base10_conversion_l3532_353229

/-- Converts a base 3 number to base 10 -/
def base3ToBase10 (digits : List Nat) : Nat :=
  digits.enum.foldl (fun acc (i, d) => acc + d * (3^i)) 0

/-- The base 3 representation of the number -/
def base3Number : List Nat := [1, 2, 1, 0, 2]

theorem base3_to_base10_conversion :
  base3ToBase10 base3Number = 178 := by
  sorry

end NUMINAMATH_CALUDE_base3_to_base10_conversion_l3532_353229


namespace NUMINAMATH_CALUDE_disneyland_arrangement_l3532_353298

def number_of_arrangements (total : ℕ) (type_a : ℕ) (type_b : ℕ) : ℕ :=
  (Nat.factorial type_a) * (Nat.factorial type_b)

theorem disneyland_arrangement :
  let total := 6
  let type_a := 2
  let type_b := 4
  number_of_arrangements total type_a type_b = 48 := by
  sorry

end NUMINAMATH_CALUDE_disneyland_arrangement_l3532_353298


namespace NUMINAMATH_CALUDE_michelle_initial_ride_fee_l3532_353261

/-- A taxi ride with an initial fee and per-mile charge. -/
structure TaxiRide where
  distance : ℝ
  chargePerMile : ℝ
  totalPaid : ℝ

/-- Calculate the initial ride fee for a taxi ride. -/
def initialRideFee (ride : TaxiRide) : ℝ :=
  ride.totalPaid - ride.distance * ride.chargePerMile

/-- Theorem: The initial ride fee for Michelle's taxi ride is $2. -/
theorem michelle_initial_ride_fee :
  let ride : TaxiRide := {
    distance := 4,
    chargePerMile := 2.5,
    totalPaid := 12
  }
  initialRideFee ride = 2 := by
  sorry

end NUMINAMATH_CALUDE_michelle_initial_ride_fee_l3532_353261


namespace NUMINAMATH_CALUDE_negative_a_cubed_times_a_squared_l3532_353225

theorem negative_a_cubed_times_a_squared (a : ℝ) : (-a)^3 * a^2 = -a^5 := by
  sorry

end NUMINAMATH_CALUDE_negative_a_cubed_times_a_squared_l3532_353225


namespace NUMINAMATH_CALUDE_midpoint_ratio_range_l3532_353234

/-- Given two lines and a point M that is the midpoint of two points on these lines,
    prove that the ratio of y₀/x₀ falls within a specific range. -/
theorem midpoint_ratio_range (P Q : ℝ × ℝ) (x₀ y₀ : ℝ) :
  (P.1 + 2 * P.2 - 1 = 0) →  -- P is on the line x + 2y - 1 = 0
  (Q.1 + 2 * Q.2 + 3 = 0) →  -- Q is on the line x + 2y + 3 = 0
  ((x₀, y₀) = ((P.1 + Q.1) / 2, (P.2 + Q.2) / 2)) →  -- M(x₀, y₀) is the midpoint of PQ
  (y₀ > x₀ + 2) →  -- Given condition
  (-1/2 < y₀ / x₀) ∧ (y₀ / x₀ < -1/5) :=  -- The range of y₀/x₀
by sorry

end NUMINAMATH_CALUDE_midpoint_ratio_range_l3532_353234


namespace NUMINAMATH_CALUDE_probability_at_least_one_white_ball_l3532_353218

theorem probability_at_least_one_white_ball (total_balls : ℕ) (red_balls : ℕ) (white_balls : ℕ) 
  (drawn_balls : ℕ) (h1 : total_balls = red_balls + white_balls) (h2 : total_balls = 5) 
  (h3 : red_balls = 3) (h4 : white_balls = 2) (h5 : drawn_balls = 3) : 
  1 - (Nat.choose red_balls drawn_balls : ℚ) / (Nat.choose total_balls drawn_balls : ℚ) = 9 / 10 := by
  sorry

end NUMINAMATH_CALUDE_probability_at_least_one_white_ball_l3532_353218


namespace NUMINAMATH_CALUDE_perimeter_of_triangle_cos_A_minus_C_l3532_353226

-- Define the triangle ABC
def triangle_ABC (a b c : ℝ) (A B C : ℝ) : Prop :=
  a = 1 ∧ b = 2 ∧ Real.cos C = 1/4

-- Theorem for the perimeter
theorem perimeter_of_triangle (a b c : ℝ) (A B C : ℝ) 
  (h : triangle_ABC a b c A B C) : a + b + c = 5 := by
  sorry

-- Theorem for cos(A-C)
theorem cos_A_minus_C (a b c : ℝ) (A B C : ℝ) 
  (h : triangle_ABC a b c A B C) : Real.cos (A - C) = 11/16 := by
  sorry

end NUMINAMATH_CALUDE_perimeter_of_triangle_cos_A_minus_C_l3532_353226


namespace NUMINAMATH_CALUDE_complex_magnitude_l3532_353217

theorem complex_magnitude (z : ℂ) (h : z * (1 - Complex.I)^2 = 1 + Complex.I) :
  Complex.abs z = Real.sqrt 2 / 2 := by
  sorry

end NUMINAMATH_CALUDE_complex_magnitude_l3532_353217


namespace NUMINAMATH_CALUDE_geometric_sequence_terms_l3532_353272

def geometric_sequence (a₁ : ℚ) (r : ℚ) (n : ℕ) : ℚ :=
  a₁ * r^(n - 1)

theorem geometric_sequence_terms
  (a₁ : ℚ) (a₂ : ℚ) (h₁ : a₁ = 4) (h₂ : a₂ = 16/3) :
  let r := a₂ / a₁
  (geometric_sequence a₁ r 10 = 1048576/19683) ∧
  (geometric_sequence a₁ r 5 = 1024/81) := by
sorry

end NUMINAMATH_CALUDE_geometric_sequence_terms_l3532_353272


namespace NUMINAMATH_CALUDE_total_tiles_needed_l3532_353224

def room_length : ℕ := 12
def room_width : ℕ := 16
def small_tile_size : ℕ := 1
def large_tile_size : ℕ := 2

theorem total_tiles_needed : 
  (room_length * room_width - (room_length - 2 * small_tile_size) * (room_width - 2 * small_tile_size)) + 
  ((room_length - 2 * small_tile_size) * (room_width - 2 * small_tile_size) / (large_tile_size * large_tile_size)) = 87 := by
  sorry

end NUMINAMATH_CALUDE_total_tiles_needed_l3532_353224


namespace NUMINAMATH_CALUDE_final_top_number_is_16_l3532_353258

/-- Represents the state of the paper after folding operations -/
structure PaperState :=
  (top_number : Nat)

/-- Represents a folding operation -/
inductive FoldOperation
  | FoldBottomUp
  | FoldTopDown
  | FoldLeftRight

/-- The initial configuration of the paper -/
def initial_paper : List (List Nat) :=
  [[1, 2], [3, 4], [5, 6], [7, 8], [9, 10], [11, 12], [13, 14], [15, 16]]

/-- Perform a single fold operation -/
def fold (state : PaperState) (op : FoldOperation) : PaperState :=
  match op with
  | FoldOperation.FoldBottomUp => { top_number := 15 }
  | FoldOperation.FoldTopDown => { top_number := 9 }
  | FoldOperation.FoldLeftRight => { top_number := state.top_number + 1 }

/-- Perform a sequence of fold operations -/
def fold_sequence (initial : PaperState) (ops : List FoldOperation) : PaperState :=
  ops.foldl fold initial

/-- The theorem to be proved -/
theorem final_top_number_is_16 :
  (fold_sequence { top_number := 1 }
    [FoldOperation.FoldBottomUp,
     FoldOperation.FoldTopDown,
     FoldOperation.FoldBottomUp,
     FoldOperation.FoldLeftRight]).top_number = 16 := by
  sorry


end NUMINAMATH_CALUDE_final_top_number_is_16_l3532_353258


namespace NUMINAMATH_CALUDE_circle_inequality_l3532_353266

theorem circle_inequality (c : ℝ) : 
  (∀ x y : ℝ, x^2 + (y - 1)^2 = 1 → x + y + c ≥ 0) ↔ c ≥ Real.sqrt 2 - 1 := by
  sorry

end NUMINAMATH_CALUDE_circle_inequality_l3532_353266


namespace NUMINAMATH_CALUDE_equation_solution_l3532_353212

theorem equation_solution (x : ℚ) : 5 * x + 3 = 2 * x - 4 → 3 * (x^2 + 6) = 103 / 3 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l3532_353212


namespace NUMINAMATH_CALUDE_tim_pencils_l3532_353222

theorem tim_pencils (tyrah_pencils : ℕ) (sarah_pencils : ℕ) (tim_pencils : ℕ)
  (h1 : tyrah_pencils = 6 * sarah_pencils)
  (h2 : tim_pencils = 8 * sarah_pencils)
  (h3 : tyrah_pencils = 12) :
  tim_pencils = 16 := by
sorry

end NUMINAMATH_CALUDE_tim_pencils_l3532_353222


namespace NUMINAMATH_CALUDE_days_worked_by_c_l3532_353244

/-- The number of days worked by person a -/
def days_a : ℕ := 6

/-- The number of days worked by person b -/
def days_b : ℕ := 9

/-- The daily wage of person c in rupees -/
def wage_c : ℕ := 115

/-- The total earnings of all three persons in rupees -/
def total_earnings : ℕ := 1702

/-- The ratio of daily wages for persons a, b, and c -/
def wage_ratio : Fin 3 → ℕ
| 0 => 3
| 1 => 4
| 2 => 5

/-- Theorem stating that person c worked for 4 days -/
theorem days_worked_by_c : 
  ∃ (days_c : ℕ), 
    days_c * wage_c + 
    days_a * (wage_ratio 0 * wage_c / wage_ratio 2) + 
    days_b * (wage_ratio 1 * wage_c / wage_ratio 2) = 
    total_earnings ∧ days_c = 4 := by
  sorry

end NUMINAMATH_CALUDE_days_worked_by_c_l3532_353244


namespace NUMINAMATH_CALUDE_aq_length_is_112_over_35_l3532_353264

/-- Represents a triangle with side lengths a, b, and c -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Represents an inscribed right triangle within another triangle -/
structure InscribedRightTriangle where
  outer : Triangle
  pc : ℝ
  bp : ℝ
  cq : ℝ

/-- The length of AQ in the described configuration -/
def aq_length (t : InscribedRightTriangle) : ℝ :=
  -- Definition of aq_length goes here
  sorry

/-- Theorem stating that AQ = 112/35 in the given configuration -/
theorem aq_length_is_112_over_35 :
  let t : InscribedRightTriangle := {
    outer := { a := 6, b := 7, c := 8 },
    pc := 4,
    bp := 3,
    cq := 3
  }
  aq_length t = 112 / 35 := by sorry

end NUMINAMATH_CALUDE_aq_length_is_112_over_35_l3532_353264


namespace NUMINAMATH_CALUDE_solve_for_b_l3532_353204

-- Define the functions p and q
def p (x : ℝ) := 3 * x + 5
def q (x b : ℝ) := 4 * x - b

-- State the theorem
theorem solve_for_b :
  ∀ b : ℝ, p (q 3 b) = 29 → b = 4 := by
  sorry

end NUMINAMATH_CALUDE_solve_for_b_l3532_353204


namespace NUMINAMATH_CALUDE_sum_of_squared_residuals_l3532_353240

theorem sum_of_squared_residuals 
  (total_sum_squared_deviations : ℝ) 
  (correlation_coefficient : ℝ) 
  (h1 : total_sum_squared_deviations = 100) 
  (h2 : correlation_coefficient = 0.818) : 
  total_sum_squared_deviations * (1 - correlation_coefficient ^ 2) = 33.0876 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_squared_residuals_l3532_353240


namespace NUMINAMATH_CALUDE_ratio_equality_l3532_353293

theorem ratio_equality (a b : ℝ) (h : 4 * a = 5 * b) : a / b = 5 / 4 := by
  sorry

end NUMINAMATH_CALUDE_ratio_equality_l3532_353293


namespace NUMINAMATH_CALUDE_spring_math_camp_attendance_l3532_353285

theorem spring_math_camp_attendance : ∃ (total boys girls : ℕ),
  total = boys + girls ∧
  50 ≤ total ∧ total ≤ 70 ∧
  3 * boys + 9 * girls = 8 * boys + 2 * girls ∧
  total = 60 := by
  sorry

end NUMINAMATH_CALUDE_spring_math_camp_attendance_l3532_353285


namespace NUMINAMATH_CALUDE_function_values_l3532_353247

-- Define the functions f and g
def f (x : ℝ) : ℝ := 4 * x^2 - 2 * x + 1
def g (x : ℝ) : ℝ := 3 * x^2 + 1

-- State the theorem
theorem function_values :
  f 2 = 9 ∧ f (-2) = 25 ∧ g (-1) = 4 := by
  sorry

end NUMINAMATH_CALUDE_function_values_l3532_353247


namespace NUMINAMATH_CALUDE_fraction_simplification_l3532_353252

theorem fraction_simplification (a b : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) :
  a^2 / (a * b) = a / b := by
  sorry

end NUMINAMATH_CALUDE_fraction_simplification_l3532_353252


namespace NUMINAMATH_CALUDE_range_of_sum_on_circle_l3532_353271

theorem range_of_sum_on_circle (x y : ℝ) (h : x^2 + y^2 - 4*x + 3 = 0) :
  ∃ (min max : ℝ), min = 2 - Real.sqrt 2 ∧ max = 2 + Real.sqrt 2 ∧
  min ≤ x + y ∧ x + y ≤ max :=
sorry

end NUMINAMATH_CALUDE_range_of_sum_on_circle_l3532_353271


namespace NUMINAMATH_CALUDE_episode_filming_time_increase_l3532_353269

/-- The percentage increase in filming time compared to episode duration -/
theorem episode_filming_time_increase (episode_duration : ℕ) (episodes_per_week : ℕ) (filming_time : ℕ) : 
  episode_duration = 20 →
  episodes_per_week = 5 →
  filming_time = 600 →
  (((filming_time / (episodes_per_week * 4)) - episode_duration) / episode_duration) * 100 = 50 := by
sorry

end NUMINAMATH_CALUDE_episode_filming_time_increase_l3532_353269


namespace NUMINAMATH_CALUDE_walking_distance_problem_l3532_353236

theorem walking_distance_problem (x t d : ℝ) 
  (h1 : d = (x + 1) * (3/4 * t))
  (h2 : d = (x - 1) * (t + 3)) :
  d = 18 := by
  sorry

end NUMINAMATH_CALUDE_walking_distance_problem_l3532_353236


namespace NUMINAMATH_CALUDE_childless_count_bertha_l3532_353216

structure Family :=
  (daughters : ℕ)
  (total_descendants : ℕ)
  (grandchildren_per_daughter : ℕ)

def childless_count (f : Family) : ℕ :=
  f.total_descendants - f.daughters

theorem childless_count_bertha (f : Family) 
  (h1 : f.daughters = 8)
  (h2 : f.total_descendants = 40)
  (h3 : f.grandchildren_per_daughter = 4)
  (h4 : f.total_descendants = f.daughters + f.daughters * f.grandchildren_per_daughter) :
  childless_count f = 32 := by
  sorry


end NUMINAMATH_CALUDE_childless_count_bertha_l3532_353216


namespace NUMINAMATH_CALUDE_cos_75_degrees_l3532_353235

theorem cos_75_degrees : Real.cos (75 * π / 180) = (Real.sqrt 6 - Real.sqrt 2) / 4 := by
  sorry

end NUMINAMATH_CALUDE_cos_75_degrees_l3532_353235


namespace NUMINAMATH_CALUDE_complex_number_problem_l3532_353259

theorem complex_number_problem (a : ℝ) : 
  let z : ℂ := Complex.I * (2 + a * Complex.I)
  (Complex.re z = -Complex.im z) → a = 2 := by
  sorry

end NUMINAMATH_CALUDE_complex_number_problem_l3532_353259


namespace NUMINAMATH_CALUDE_min_value_a_plus_2b_min_value_equals_3_plus_2sqrt2_l3532_353299

theorem min_value_a_plus_2b (a b : ℝ) (ha : a > 0) (hb : b > 0) (h : 1/a + 1/b = 1) :
  ∀ x y : ℝ, x > 0 → y > 0 → 1/x + 1/y = 1 → a + 2*b ≤ x + 2*y :=
by sorry

theorem min_value_equals_3_plus_2sqrt2 (a b : ℝ) (ha : a > 0) (hb : b > 0) (h : 1/a + 1/b = 1) :
  a + 2*b = 3 + 2*Real.sqrt 2 :=
by sorry

end NUMINAMATH_CALUDE_min_value_a_plus_2b_min_value_equals_3_plus_2sqrt2_l3532_353299


namespace NUMINAMATH_CALUDE_quartet_performances_theorem_l3532_353206

/-- Represents the number of performances for each friend -/
structure Performances where
  sarah : ℕ
  lily : ℕ
  emma : ℕ
  nora : ℕ
  kate : ℕ

/-- The total number of quartet performances -/
def total_performances (p : Performances) : ℕ :=
  (p.sarah + p.lily + p.emma + p.nora + p.kate) / 4

theorem quartet_performances_theorem (p : Performances) :
  p.nora = 10 →
  p.sarah = 6 →
  p.lily > 6 →
  p.emma > 6 →
  p.kate > 6 →
  p.lily < 10 →
  p.emma < 10 →
  p.kate < 10 →
  (p.sarah + p.lily + p.emma + p.nora + p.kate) % 4 = 0 →
  total_performances p = 10 := by
  sorry

#check quartet_performances_theorem

end NUMINAMATH_CALUDE_quartet_performances_theorem_l3532_353206


namespace NUMINAMATH_CALUDE_tiles_per_row_l3532_353227

-- Define the room area in square feet
def room_area : ℝ := 144

-- Define the tile size in inches
def tile_size : ℝ := 8

-- Define the number of inches in a foot
def inches_per_foot : ℝ := 12

-- Theorem to prove
theorem tiles_per_row : 
  ⌊(inches_per_foot * (room_area ^ (1/2 : ℝ))) / tile_size⌋ = 18 := by
  sorry

end NUMINAMATH_CALUDE_tiles_per_row_l3532_353227


namespace NUMINAMATH_CALUDE_comic_stacking_arrangements_l3532_353248

def spiderman_comics : ℕ := 8
def archie_comics : ℕ := 5
def garfield_comics : ℕ := 3

def total_comics : ℕ := spiderman_comics + archie_comics + garfield_comics

def garfield_group_positions : ℕ := 3

theorem comic_stacking_arrangements :
  (spiderman_comics.factorial * archie_comics.factorial * garfield_comics.factorial * garfield_group_positions) = 8669760 := by
  sorry

end NUMINAMATH_CALUDE_comic_stacking_arrangements_l3532_353248


namespace NUMINAMATH_CALUDE_water_break_frequency_l3532_353242

theorem water_break_frequency
  (total_work_time : ℕ)
  (sitting_break_interval : ℕ)
  (water_break_excess : ℕ)
  (h1 : total_work_time = 240)
  (h2 : sitting_break_interval = 120)
  (h3 : water_break_excess = 10)
  : ℕ :=
  by
  -- Proof goes here
  sorry

#check water_break_frequency

end NUMINAMATH_CALUDE_water_break_frequency_l3532_353242


namespace NUMINAMATH_CALUDE_hyperbola_iff_m_in_range_l3532_353233

/-- The equation represents a hyperbola -/
def is_hyperbola (m : ℝ) : Prop :=
  ∃ (x y : ℝ), x^2 / (2 + m) + y^2 / (m + 1) = 1

/-- The range of m for which the equation represents a hyperbola -/
def m_range : Set ℝ := {m | -2 < m ∧ m < -1}

/-- Theorem: The equation represents a hyperbola if and only if m is in the range (-2, -1) -/
theorem hyperbola_iff_m_in_range :
  ∀ m : ℝ, is_hyperbola m ↔ m ∈ m_range :=
by sorry

end NUMINAMATH_CALUDE_hyperbola_iff_m_in_range_l3532_353233


namespace NUMINAMATH_CALUDE_two_numbers_difference_l3532_353267

theorem two_numbers_difference (a b : ℕ) : 
  a + b = 23976 →
  b % 8 = 0 →
  a = b - b / 8 →
  b - a = 1598 := by
sorry

end NUMINAMATH_CALUDE_two_numbers_difference_l3532_353267


namespace NUMINAMATH_CALUDE_isosceles_right_triangle_locus_l3532_353292

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Calculates the square of the distance between two points -/
def distanceSquared (p1 p2 : Point) : ℝ :=
  (p1.x - p2.x)^2 + (p1.y - p2.y)^2

/-- Theorem: Locus of points for isosceles right triangle -/
theorem isosceles_right_triangle_locus (s : ℝ) (h : s > 0) :
  let A : Point := ⟨0, 0⟩
  let B : Point := ⟨s, 0⟩
  let C : Point := ⟨0, s⟩
  let center : Point := ⟨s/3, s/3⟩
  let radius : ℝ := Real.sqrt (s^2/3)
  ∀ P : Point, 
    (distanceSquared P A + distanceSquared P B + distanceSquared P C = 4 * s^2) ↔ 
    (distanceSquared P center = radius^2) := by
  sorry

end NUMINAMATH_CALUDE_isosceles_right_triangle_locus_l3532_353292


namespace NUMINAMATH_CALUDE_cu_atom_count_l3532_353275

/-- Represents the number of atoms of an element in a compound -/
structure AtomCount where
  count : ℕ

/-- Represents the atomic weight of an element in amu -/
structure AtomicWeight where
  weight : ℝ

/-- Represents a chemical compound -/
structure Compound where
  cu : AtomCount
  c : AtomCount
  o : AtomCount
  molecularWeight : ℝ

def cuAtomicWeight : AtomicWeight := ⟨63.55⟩
def cAtomicWeight : AtomicWeight := ⟨12.01⟩
def oAtomicWeight : AtomicWeight := ⟨16.00⟩

def compoundWeight (cpd : Compound) : ℝ :=
  cpd.cu.count * cuAtomicWeight.weight +
  cpd.c.count * cAtomicWeight.weight +
  cpd.o.count * oAtomicWeight.weight

theorem cu_atom_count (cpd : Compound) :
  cpd.c = ⟨1⟩ ∧ cpd.o = ⟨3⟩ ∧ cpd.molecularWeight = 124 →
  cpd.cu = ⟨1⟩ :=
by sorry

end NUMINAMATH_CALUDE_cu_atom_count_l3532_353275


namespace NUMINAMATH_CALUDE_dinner_bill_proof_l3532_353270

theorem dinner_bill_proof (total_friends : ℕ) (paying_friends : ℕ) (extra_payment : ℚ) : 
  total_friends = 10 → 
  paying_friends = 9 → 
  extra_payment = 3 → 
  ∃ (bill : ℚ), bill = 270 ∧ 
    paying_friends * (bill / total_friends + extra_payment) = bill :=
by sorry

end NUMINAMATH_CALUDE_dinner_bill_proof_l3532_353270


namespace NUMINAMATH_CALUDE_completing_square_equivalence_l3532_353274

theorem completing_square_equivalence :
  ∀ x : ℝ, (x^2 + 2*x - 5 = 0) ↔ ((x + 1)^2 = 6) := by
  sorry

end NUMINAMATH_CALUDE_completing_square_equivalence_l3532_353274


namespace NUMINAMATH_CALUDE_square_sum_value_l3532_353256

theorem square_sum_value (a b : ℝ) (h1 : a - b = 3) (h2 : a * b = 10) : a^2 + b^2 = 29 := by
  sorry

end NUMINAMATH_CALUDE_square_sum_value_l3532_353256


namespace NUMINAMATH_CALUDE_joan_apples_l3532_353231

/-- The number of apples Joan picked -/
def apples_picked : ℕ := 43

/-- The number of apples Joan gave to Melanie -/
def apples_given : ℕ := 27

/-- The number of apples Joan has now -/
def apples_remaining : ℕ := apples_picked - apples_given

theorem joan_apples : apples_remaining = 16 := by
  sorry

end NUMINAMATH_CALUDE_joan_apples_l3532_353231


namespace NUMINAMATH_CALUDE_card_distribution_l3532_353219

theorem card_distribution (n : ℕ) : 
  (Finset.sum (Finset.range (n - 1)) (λ k => Nat.choose n (k + 1))) = 2 * (2^(n - 1) - 1) :=
by sorry

end NUMINAMATH_CALUDE_card_distribution_l3532_353219


namespace NUMINAMATH_CALUDE_sin_sum_inverse_sin_tan_l3532_353246

theorem sin_sum_inverse_sin_tan (x y : ℝ) 
  (hx : x = 4 / 5) (hy : y = 1 / 2) : 
  Real.sin (Real.arcsin x + Real.arctan y) = 11 * Real.sqrt 5 / 25 := by
  sorry

end NUMINAMATH_CALUDE_sin_sum_inverse_sin_tan_l3532_353246


namespace NUMINAMATH_CALUDE_max_odd_sums_for_given_range_l3532_353276

def max_odd_sums (n : ℕ) (start : ℕ) : ℕ :=
  if n ≤ 2 then 0
  else ((n - 2) / 2) + 1

theorem max_odd_sums_for_given_range :
  max_odd_sums 998 1000 = 499 := by
  sorry

end NUMINAMATH_CALUDE_max_odd_sums_for_given_range_l3532_353276


namespace NUMINAMATH_CALUDE_largest_number_with_equal_quotient_and_remainder_l3532_353232

theorem largest_number_with_equal_quotient_and_remainder :
  ∀ (A B C : ℕ),
    (A = 7 * B + C) →
    (B = C) →
    (C < 7) →
    A ≤ 48 :=
by
  sorry

end NUMINAMATH_CALUDE_largest_number_with_equal_quotient_and_remainder_l3532_353232


namespace NUMINAMATH_CALUDE_shanghai_population_aging_l3532_353228

/-- Represents a city's demographic characteristics -/
structure CityDemographics where
  location : String
  economy : String
  inMigrationRate : String
  mechanicalGrowthRate : String
  naturalGrowthRate : String

/-- Represents possible population issues -/
inductive PopulationIssue
  | SatelliteTownPopulation
  | PopulationAging
  | LargePopulationBase
  | YoungPopulationStructure

/-- Determines the most significant population issue for a given city -/
def mostSignificantIssue (city : CityDemographics) : PopulationIssue :=
  sorry

/-- Shanghai's demographic characteristics -/
def shanghai : CityDemographics := {
  location := "eastern coast of China",
  economy := "developed",
  inMigrationRate := "high",
  mechanicalGrowthRate := "high",
  naturalGrowthRate := "low"
}

/-- Theorem stating that Shanghai's most significant population issue is aging -/
theorem shanghai_population_aging :
  mostSignificantIssue shanghai = PopulationIssue.PopulationAging :=
  sorry

end NUMINAMATH_CALUDE_shanghai_population_aging_l3532_353228


namespace NUMINAMATH_CALUDE_age_of_B_l3532_353253

/-- Given the initial ratio of ages and the ratio after 2 years, prove B's age is 6 years -/
theorem age_of_B (k : ℚ) (x : ℚ) : 
  (5 * k : ℚ) / (3 * k : ℚ) = 5 / 3 →
  (4 * k : ℚ) / (3 * k : ℚ) = 4 / 3 →
  ((5 * k + 2) : ℚ) / ((3 * k + 2) : ℚ) = 3 / 2 →
  ((3 * k + 2) : ℚ) / ((2 * k + 2) : ℚ) = 2 / x →
  (3 * k : ℚ) = 6 := by
  sorry

#check age_of_B

end NUMINAMATH_CALUDE_age_of_B_l3532_353253


namespace NUMINAMATH_CALUDE_problem_statement_l3532_353289

theorem problem_statement : 5 * 301 + 4 * 301 + 3 * 301 + 300 = 3912 := by
  sorry

end NUMINAMATH_CALUDE_problem_statement_l3532_353289


namespace NUMINAMATH_CALUDE_division_remainder_problem_l3532_353296

theorem division_remainder_problem : ∃ (x : ℕ+), 
  19250 % x.val = 11 ∧ 
  20302 % x.val = 3 ∧ 
  x.val = 53 := by
  sorry

end NUMINAMATH_CALUDE_division_remainder_problem_l3532_353296


namespace NUMINAMATH_CALUDE_quadratic_inequality_equivalence_l3532_353209

theorem quadratic_inequality_equivalence :
  ∃ d : ℝ, ∀ x : ℝ, x * (2 * x + 4) < d ↔ x ∈ Set.Ioo (-4) 1 :=
by
  use 8
  sorry

end NUMINAMATH_CALUDE_quadratic_inequality_equivalence_l3532_353209


namespace NUMINAMATH_CALUDE_mikes_basketball_games_l3532_353273

theorem mikes_basketball_games (points_per_game : ℕ) (total_points : ℕ) (h1 : points_per_game = 4) (h2 : total_points = 24) :
  total_points / points_per_game = 6 := by
  sorry

end NUMINAMATH_CALUDE_mikes_basketball_games_l3532_353273


namespace NUMINAMATH_CALUDE_y_increase_for_x_increase_l3532_353238

/-- Given a line with the following properties:
    1. When x increases by 2 units, y increases by 5 units.
    2. The line passes through the point (1, 1).
    3. We consider an x-value increase of 8 units.
    
    This theorem proves that the y-value will increase by 20 units. -/
theorem y_increase_for_x_increase (slope : ℚ) (x_increase y_increase : ℚ) :
  slope = 5 / 2 →
  x_increase = 8 →
  y_increase = slope * x_increase →
  y_increase = 20 := by
  sorry

end NUMINAMATH_CALUDE_y_increase_for_x_increase_l3532_353238


namespace NUMINAMATH_CALUDE_percentage_materialB_in_final_mixture_l3532_353220

/-- Represents a mixture of oil and material B -/
structure Mixture where
  total : ℝ
  oil : ℝ
  materialB : ℝ

/-- The initial mixture A -/
def initialMixtureA : Mixture :=
  { total := 8
    oil := 8 * 0.2
    materialB := 8 * 0.8 }

/-- The mixture after adding 2 kg of oil -/
def mixtureAfterOil : Mixture :=
  { total := initialMixtureA.total + 2
    oil := initialMixtureA.oil + 2
    materialB := initialMixtureA.materialB }

/-- The additional 6 kg of mixture A -/
def additionalMixtureA : Mixture :=
  { total := 6
    oil := 6 * 0.2
    materialB := 6 * 0.8 }

/-- The final mixture -/
def finalMixture : Mixture :=
  { total := mixtureAfterOil.total + additionalMixtureA.total
    oil := mixtureAfterOil.oil + additionalMixtureA.oil
    materialB := mixtureAfterOil.materialB + additionalMixtureA.materialB }

theorem percentage_materialB_in_final_mixture :
  finalMixture.materialB / finalMixture.total = 0.7 := by
  sorry

end NUMINAMATH_CALUDE_percentage_materialB_in_final_mixture_l3532_353220


namespace NUMINAMATH_CALUDE_find_divisor_l3532_353295

theorem find_divisor (dividend : Nat) (quotient : Nat) (h1 : dividend = 62976) (h2 : quotient = 123) :
  dividend / quotient = 512 := by
  sorry

end NUMINAMATH_CALUDE_find_divisor_l3532_353295


namespace NUMINAMATH_CALUDE_circle_area_difference_l3532_353286

theorem circle_area_difference : ∀ (π : ℝ), 
  let r1 : ℝ := 30
  let d2 : ℝ := 30
  let area1 : ℝ := π * r1^2
  let area2 : ℝ := π * (d2/2)^2
  area1 - area2 = 675 * π :=
by sorry

end NUMINAMATH_CALUDE_circle_area_difference_l3532_353286


namespace NUMINAMATH_CALUDE_employed_males_percentage_l3532_353287

theorem employed_males_percentage (total_population : ℝ) (employed_population : ℝ) (employed_females : ℝ) :
  employed_population = 0.64 * total_population →
  employed_females = 0.21875 * employed_population →
  0.4996 * total_population = employed_population - employed_females :=
by sorry

end NUMINAMATH_CALUDE_employed_males_percentage_l3532_353287


namespace NUMINAMATH_CALUDE_intersection_of_A_and_B_l3532_353249

def set_A : Set ℝ := {x | ∃ y, y = Real.sqrt (x^2 - 1)}
def set_B : Set ℝ := {y | ∃ x, y = Real.sqrt (x^2 - 1)}

theorem intersection_of_A_and_B : set_A ∩ set_B = Set.Ici 1 := by sorry

end NUMINAMATH_CALUDE_intersection_of_A_and_B_l3532_353249


namespace NUMINAMATH_CALUDE_tan_75_degrees_l3532_353257

theorem tan_75_degrees (h1 : 75 = 60 + 15) 
                        (h2 : Real.tan (60 * π / 180) = Real.sqrt 3) 
                        (h3 : Real.tan (15 * π / 180) = 2 - Real.sqrt 3) : 
  Real.tan (75 * π / 180) = 2 + Real.sqrt 3 := by sorry

end NUMINAMATH_CALUDE_tan_75_degrees_l3532_353257


namespace NUMINAMATH_CALUDE_endpoint_of_vector_l3532_353210

def vector_a : Fin 3 → ℝ := ![3, -4, 2]
def point_A : Fin 3 → ℝ := ![2, -1, 1]
def point_B : Fin 3 → ℝ := ![5, -5, 3]

theorem endpoint_of_vector (i : Fin 3) : 
  point_B i = point_A i + vector_a i :=
by
  sorry

end NUMINAMATH_CALUDE_endpoint_of_vector_l3532_353210


namespace NUMINAMATH_CALUDE_max_value_inequality_l3532_353213

theorem max_value_inequality (x y z : ℝ) :
  ∃ (A : ℝ), A > 0 ∧ 
  (∀ (B : ℝ), B > A → 
    ∃ (a b c : ℝ), a^4 + b^4 + c^4 + a^2*b*c + a*b^2*c + a*b*c^2 - B*(a*b + b*c + c*a)^2 < 0) ∧
  (x^4 + y^4 + z^4 + x^2*y*z + x*y^2*z + x*y*z^2 - A*(x*y + y*z + z*x)^2 ≥ 0) ∧
  A = 2/3 :=
sorry

end NUMINAMATH_CALUDE_max_value_inequality_l3532_353213


namespace NUMINAMATH_CALUDE_max_a_for_monotonous_l3532_353250

/-- The function f(x) = -x^3 + ax is monotonous (non-increasing) on [1, +∞) -/
def is_monotonous (a : ℝ) : Prop :=
  ∀ x y, x ≥ 1 → y ≥ 1 → x ≤ y → (-x^3 + a*x) ≥ (-y^3 + a*y)

/-- The maximum value of a for which f(x) = -x^3 + ax is monotonous on [1, +∞) is 3 -/
theorem max_a_for_monotonous : (∃ a_max : ℝ, a_max = 3 ∧ 
  (∀ a : ℝ, is_monotonous a → a ≤ a_max) ∧ 
  is_monotonous a_max) :=
sorry

end NUMINAMATH_CALUDE_max_a_for_monotonous_l3532_353250


namespace NUMINAMATH_CALUDE_units_digit_of_sum_of_products_l3532_353281

def consecutive_product (n : ℕ) (count : ℕ) : ℕ :=
  (List.range count).foldl (λ acc _ => acc * n) 1

def units_digit (n : ℕ) : ℕ :=
  n % 10

theorem units_digit_of_sum_of_products : 
  units_digit (consecutive_product 2017 2016 + consecutive_product 2016 2017) = 7 := by
  sorry

end NUMINAMATH_CALUDE_units_digit_of_sum_of_products_l3532_353281


namespace NUMINAMATH_CALUDE_total_rulers_problem_solution_l3532_353200

/-- Given an initial number of rulers and a number of rulers added, 
    the total number of rulers is equal to the sum of these two numbers. -/
theorem total_rulers (initial_rulers added_rulers : ℕ) :
  initial_rulers + added_rulers = 
    initial_rulers + added_rulers := by sorry

/-- The specific case for the problem -/
theorem problem_solution : 
  11 + 14 = 25 := by sorry

end NUMINAMATH_CALUDE_total_rulers_problem_solution_l3532_353200


namespace NUMINAMATH_CALUDE_one_eighth_of_number_l3532_353203

theorem one_eighth_of_number (n : ℚ) (h : 6/11 * n = 48) : 1/8 * n = 11 := by
  sorry

end NUMINAMATH_CALUDE_one_eighth_of_number_l3532_353203


namespace NUMINAMATH_CALUDE_impossibleEggDivision_l3532_353263

/-- Represents the number of eggs of each type -/
structure EggCounts where
  typeA : ℕ
  typeB : ℕ
  typeC : ℕ

/-- Represents the ratio of eggs in each group -/
structure EggRatio where
  typeA : ℕ
  typeB : ℕ
  typeC : ℕ

/-- Function to check if it's possible to divide eggs into groups with a given ratio -/
def canDivideEggs (counts : EggCounts) (ratio : EggRatio) (numGroups : ℕ) : Prop :=
  ∃ (groupSize : ℕ),
    counts.typeA = numGroups * groupSize * ratio.typeA ∧
    counts.typeB = numGroups * groupSize * ratio.typeB ∧
    counts.typeC = numGroups * groupSize * ratio.typeC

/-- Theorem stating that it's impossible to divide the given eggs into 5 groups with the specified ratio -/
theorem impossibleEggDivision : 
  let counts : EggCounts := ⟨15, 12, 8⟩
  let ratio : EggRatio := ⟨2, 3, 1⟩
  let numGroups : ℕ := 5
  ¬(canDivideEggs counts ratio numGroups) := by
  sorry


end NUMINAMATH_CALUDE_impossibleEggDivision_l3532_353263


namespace NUMINAMATH_CALUDE_pens_taken_after_first_month_l3532_353202

theorem pens_taken_after_first_month 
  (total_pens : ℕ) 
  (pens_taken_second_month : ℕ) 
  (remaining_pens : ℕ) : 
  total_pens = 315 → 
  pens_taken_second_month = 41 → 
  remaining_pens = 237 → 
  total_pens - (total_pens - remaining_pens - pens_taken_second_month) - pens_taken_second_month = remaining_pens → 
  total_pens - remaining_pens - pens_taken_second_month = 37 := by
  sorry

end NUMINAMATH_CALUDE_pens_taken_after_first_month_l3532_353202


namespace NUMINAMATH_CALUDE_complex_sum_equals_negative_two_l3532_353251

/-- Given that z = cos(6π/11) + i sin(6π/11), prove that z/(1 + z²) + z²/(1 + z⁴) + z³/(1 + z⁶) = -2 -/
theorem complex_sum_equals_negative_two (z : ℂ) (h : z = Complex.exp (Complex.I * (6 * Real.pi / 11))) :
  z / (1 + z^2) + z^2 / (1 + z^4) + z^3 / (1 + z^6) = -2 := by
  sorry

end NUMINAMATH_CALUDE_complex_sum_equals_negative_two_l3532_353251


namespace NUMINAMATH_CALUDE_dining_bill_share_l3532_353262

def total_bill : ℝ := 211.00
def num_people : ℕ := 9
def tip_percentage : ℝ := 0.15

theorem dining_bill_share :
  let tip := total_bill * tip_percentage
  let total_with_tip := total_bill + tip
  let share_per_person := total_with_tip / num_people
  ∃ ε > 0, |share_per_person - 26.96| < ε :=
by sorry

end NUMINAMATH_CALUDE_dining_bill_share_l3532_353262


namespace NUMINAMATH_CALUDE_chosen_number_l3532_353221

theorem chosen_number (x : ℝ) : (x / 12)^2 - 240 = 8 → x = 24 * Real.sqrt 62 := by
  sorry

end NUMINAMATH_CALUDE_chosen_number_l3532_353221


namespace NUMINAMATH_CALUDE_max_m_value_l3532_353223

theorem max_m_value (p q : ℝ → Prop) (m : ℝ) : 
  (∀ x, p x ↔ (x^2 - 4*x - 5 > 0)) →
  (∀ x, q x ↔ (x^2 - 2*x + 1 - m^2 > 0)) →
  (m > 0) →
  (∀ x, p x → q x) →
  (∃ x, q x ∧ ¬(p x)) →
  (∀ m' > m, ∃ x, p x ∧ ¬(q x)) →
  m = 2 :=
by sorry

end NUMINAMATH_CALUDE_max_m_value_l3532_353223

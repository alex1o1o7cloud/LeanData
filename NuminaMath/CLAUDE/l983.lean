import Mathlib

namespace NUMINAMATH_CALUDE_power_of_64_three_fourths_l983_98314

theorem power_of_64_three_fourths : (64 : ℝ) ^ (3/4) = 16 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_power_of_64_three_fourths_l983_98314


namespace NUMINAMATH_CALUDE_compute_expression_l983_98338

theorem compute_expression : 75 * 1313 - 25 * 1313 + 50 * 1313 = 131300 := by
  sorry

end NUMINAMATH_CALUDE_compute_expression_l983_98338


namespace NUMINAMATH_CALUDE_distance_swam_against_current_l983_98345

/-- Calculates the distance swam against a river current -/
theorem distance_swam_against_current
  (speed_still_water : ℝ)
  (current_speed : ℝ)
  (time : ℝ)
  (h1 : speed_still_water = 5)
  (h2 : current_speed = 1.2)
  (h3 : time = 3.1578947368421053)
  : (speed_still_water - current_speed) * time = 12 := by
  sorry

end NUMINAMATH_CALUDE_distance_swam_against_current_l983_98345


namespace NUMINAMATH_CALUDE_probability_distribution_problem_l983_98367

theorem probability_distribution_problem (m n : ℝ) 
  (sum_prob : 0.1 + m + n + 0.1 = 1)
  (condition : m + 2 * n = 1.2) : n = 0.4 := by
  sorry

end NUMINAMATH_CALUDE_probability_distribution_problem_l983_98367


namespace NUMINAMATH_CALUDE_unsold_bars_l983_98331

theorem unsold_bars (total_bars : ℕ) (price_per_bar : ℕ) (total_amount : ℕ) : 
  total_bars = 8 → price_per_bar = 4 → total_amount = 20 → 
  total_bars - (total_amount / price_per_bar) = 3 :=
by sorry

end NUMINAMATH_CALUDE_unsold_bars_l983_98331


namespace NUMINAMATH_CALUDE_keyboard_printer_cost_l983_98328

/-- The total cost of keyboards and printers -/
def total_cost (num_keyboards : ℕ) (num_printers : ℕ) (keyboard_price : ℕ) (printer_price : ℕ) : ℕ :=
  num_keyboards * keyboard_price + num_printers * printer_price

/-- Theorem stating that the total cost of 15 keyboards at $20 each and 25 printers at $70 each is $2050 -/
theorem keyboard_printer_cost : total_cost 15 25 20 70 = 2050 := by
  sorry

end NUMINAMATH_CALUDE_keyboard_printer_cost_l983_98328


namespace NUMINAMATH_CALUDE_initial_blocks_l983_98362

theorem initial_blocks (initial final added : ℕ) : 
  final = initial + added → 
  final = 65 → 
  added = 30 → 
  initial = 35 := by sorry

end NUMINAMATH_CALUDE_initial_blocks_l983_98362


namespace NUMINAMATH_CALUDE_parabola_c_value_l983_98399

/-- A parabola with vertex (h, k) passing through point (x₀, y₀) has c = 12.5 -/
theorem parabola_c_value (a b c h k x₀ y₀ : ℝ) : 
  (∀ x y, y = a * x^2 + b * x + c) →  -- parabola equation
  (h = 3 ∧ k = -1) →                  -- vertex at (3, -1)
  (x₀ = 1 ∧ y₀ = 5) →                 -- point (1, 5) on parabola
  (∀ x, a * (x - h)^2 + k = a * x^2 + b * x + c) →  -- vertex form equals general form
  (y₀ = a * x₀^2 + b * x₀ + c) →      -- point (1, 5) satisfies equation
  c = 12.5 := by
sorry

end NUMINAMATH_CALUDE_parabola_c_value_l983_98399


namespace NUMINAMATH_CALUDE_milk_consumption_l983_98346

/-- The amount of regular milk consumed in a week -/
def regular_milk : ℝ := 0.5

/-- The amount of soy milk consumed in a week -/
def soy_milk : ℝ := 0.1

/-- The total amount of milk consumed in a week -/
def total_milk : ℝ := regular_milk + soy_milk

theorem milk_consumption : total_milk = 0.6 := by
  sorry

end NUMINAMATH_CALUDE_milk_consumption_l983_98346


namespace NUMINAMATH_CALUDE_cookies_sold_in_morning_l983_98383

/-- Proves the number of cookies sold in the morning given the total cookies,
    cookies sold during lunch and afternoon, and cookies left at the end of the day. -/
theorem cookies_sold_in_morning 
  (total : ℕ) 
  (lunch_sold : ℕ) 
  (afternoon_sold : ℕ) 
  (left_at_end : ℕ) 
  (h1 : total = 120) 
  (h2 : lunch_sold = 57) 
  (h3 : afternoon_sold = 16) 
  (h4 : left_at_end = 11) : 
  total - lunch_sold - afternoon_sold - left_at_end = 36 := by
  sorry

end NUMINAMATH_CALUDE_cookies_sold_in_morning_l983_98383


namespace NUMINAMATH_CALUDE_thirty_five_million_scientific_notation_l983_98351

/-- Represents a number in scientific notation -/
structure ScientificNotation where
  coefficient : ℝ
  exponent : ℤ
  is_valid : 1 ≤ coefficient ∧ coefficient < 10

/-- Converts a positive integer to scientific notation -/
def to_scientific_notation (n : ℕ+) : ScientificNotation :=
  sorry

theorem thirty_five_million_scientific_notation :
  to_scientific_notation 35000000 = ScientificNotation.mk 3.5 7 (by norm_num) :=
sorry

end NUMINAMATH_CALUDE_thirty_five_million_scientific_notation_l983_98351


namespace NUMINAMATH_CALUDE_paintings_distribution_l983_98355

theorem paintings_distribution (total_paintings : ℕ) (paintings_per_room : ℕ) (num_rooms : ℕ) :
  total_paintings = 32 →
  paintings_per_room = 8 →
  total_paintings = paintings_per_room * num_rooms →
  num_rooms = 4 := by
sorry

end NUMINAMATH_CALUDE_paintings_distribution_l983_98355


namespace NUMINAMATH_CALUDE_nth_equation_l983_98352

/-- The product of consecutive integers from n+1 to 2n -/
def leftSide (n : ℕ) : ℕ := Finset.prod (Finset.range n) (λ i => n + i + 1)

/-- The product of odd numbers from 1 to 2n-1 -/
def oddProduct (n : ℕ) : ℕ := Finset.prod (Finset.range n) (λ i => 2 * i + 1)

/-- The nth equation in the pattern -/
theorem nth_equation (n : ℕ) : leftSide n = 2^n * oddProduct n := by
  sorry

end NUMINAMATH_CALUDE_nth_equation_l983_98352


namespace NUMINAMATH_CALUDE_a_equals_fibonacci_ratio_l983_98356

def fibonacci : ℕ → ℕ
  | 0 => 1
  | 1 => 1
  | (n + 2) => fibonacci (n + 1) + fibonacci n

def a : ℕ → ℕ
  | 0 => 3
  | (n + 1) => (a n)^2 - 2

theorem a_equals_fibonacci_ratio (n : ℕ) :
  a n = fibonacci (2^(n+1)) / fibonacci 4 := by
  sorry

end NUMINAMATH_CALUDE_a_equals_fibonacci_ratio_l983_98356


namespace NUMINAMATH_CALUDE_marks_tanks_l983_98305

/-- The number of tanks Mark has for pregnant fish -/
def num_tanks : ℕ := sorry

/-- The number of pregnant fish in each tank -/
def fish_per_tank : ℕ := 4

/-- The number of young fish each pregnant fish gives birth to -/
def young_per_fish : ℕ := 20

/-- The total number of young fish Mark has -/
def total_young : ℕ := 240

/-- Theorem stating that the number of tanks Mark has is 3 -/
theorem marks_tanks : num_tanks = 3 := by sorry

end NUMINAMATH_CALUDE_marks_tanks_l983_98305


namespace NUMINAMATH_CALUDE_largest_prime_divisor_of_1202102_base5_l983_98343

/-- Converts a number from base 5 to base 10 -/
def base5ToBase10 (n : ℕ) : ℕ := sorry

/-- Checks if a number is prime -/
def isPrime (n : ℕ) : Prop := sorry

/-- Finds the largest prime divisor of a natural number -/
def largestPrimeDivisor (n : ℕ) : ℕ := sorry

theorem largest_prime_divisor_of_1202102_base5 :
  largestPrimeDivisor (base5ToBase10 1202102) = 307 := by sorry

end NUMINAMATH_CALUDE_largest_prime_divisor_of_1202102_base5_l983_98343


namespace NUMINAMATH_CALUDE_complex_magnitude_problem_l983_98300

theorem complex_magnitude_problem (z : ℂ) : z = (2 + I) / (1 - I) → Complex.abs z = Real.sqrt 10 / 2 := by
  sorry

end NUMINAMATH_CALUDE_complex_magnitude_problem_l983_98300


namespace NUMINAMATH_CALUDE_min_value_theorem_l983_98303

theorem min_value_theorem (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : 3 / x + 1 / y = 1) :
  3 * x + 4 * y ≥ 25 ∧ ∃ (x₀ y₀ : ℝ), 3 * x₀ + 4 * y₀ = 25 ∧ 3 / x₀ + 1 / y₀ = 1 :=
by sorry

end NUMINAMATH_CALUDE_min_value_theorem_l983_98303


namespace NUMINAMATH_CALUDE_subset_implies_m_equals_one_l983_98341

def A (m : ℝ) : Set ℝ := {-1, 3, 2*m-1}
def B (m : ℝ) : Set ℝ := {3, m^2}

theorem subset_implies_m_equals_one (m : ℝ) :
  B m ⊆ A m → m = 1 :=
by sorry

end NUMINAMATH_CALUDE_subset_implies_m_equals_one_l983_98341


namespace NUMINAMATH_CALUDE_recipe_total_cups_l983_98361

/-- Represents the ratio of ingredients in a recipe -/
structure RecipeRatio where
  butter : ℕ
  flour : ℕ
  sugar : ℕ

/-- Calculates the total cups of ingredients given a ratio and the amount of butter -/
def totalIngredients (ratio : RecipeRatio) (butterCups : ℕ) : ℕ :=
  butterCups * (ratio.butter + ratio.flour + ratio.sugar) / ratio.butter

theorem recipe_total_cups (ratio : RecipeRatio) (butterCups : ℕ) 
    (h1 : ratio.butter = 1) 
    (h2 : ratio.flour = 5) 
    (h3 : ratio.sugar = 3) 
    (h4 : butterCups = 9) : 
  totalIngredients ratio butterCups = 81 := by
  sorry

end NUMINAMATH_CALUDE_recipe_total_cups_l983_98361


namespace NUMINAMATH_CALUDE_star_property_l983_98332

/-- Custom binary operation ※ -/
def star (a b : ℝ) (x y : ℝ) : ℝ := a * x - b * y

theorem star_property (a b : ℝ) (h : star a b 1 2 = 8) :
  star a b (-2) (-4) = -16 := by sorry

end NUMINAMATH_CALUDE_star_property_l983_98332


namespace NUMINAMATH_CALUDE_pyramid_volume_theorem_l983_98395

/-- A regular hexagon -/
structure RegularHexagon where
  vertices : Fin 6 → ℝ × ℝ

/-- A right pyramid with a regular hexagon base -/
structure RightPyramid where
  base : RegularHexagon
  apex : ℝ × ℝ × ℝ

/-- An equilateral triangle -/
structure EquilateralTriangle where
  vertices : Fin 3 → ℝ × ℝ

/-- Calculate the volume of a right pyramid -/
def pyramidVolume (p : RightPyramid) : ℝ := sorry

/-- Check if a triangle is equilateral with given side length -/
def isEquilateralWithSideLength (t : EquilateralTriangle) (s : ℝ) : Prop := sorry

theorem pyramid_volume_theorem (p : RightPyramid) (t : EquilateralTriangle) :
  isEquilateralWithSideLength t 10 →
  pyramidVolume p = 187.5 := by
  sorry

end NUMINAMATH_CALUDE_pyramid_volume_theorem_l983_98395


namespace NUMINAMATH_CALUDE_asha_win_probability_l983_98397

theorem asha_win_probability (p_lose p_tie : ℚ) : 
  p_lose = 3/8 → p_tie = 1/4 → 1 - p_lose - p_tie = 3/8 := by
  sorry

end NUMINAMATH_CALUDE_asha_win_probability_l983_98397


namespace NUMINAMATH_CALUDE_possible_average_82_l983_98347

def test_scores : List Nat := [71, 77, 80, 87]

theorem possible_average_82 (last_score : Nat) 
  (h1 : last_score ≥ 0)
  (h2 : last_score ≤ 100) :
  ∃ (avg : Rat), 
    avg = (test_scores.sum + last_score) / 5 ∧ 
    avg = 82 := by
  sorry

end NUMINAMATH_CALUDE_possible_average_82_l983_98347


namespace NUMINAMATH_CALUDE_soccer_committee_combinations_l983_98389

theorem soccer_committee_combinations : Nat.choose 6 4 = 15 := by
  sorry

end NUMINAMATH_CALUDE_soccer_committee_combinations_l983_98389


namespace NUMINAMATH_CALUDE_pie_chart_most_appropriate_for_milk_powder_l983_98392

/-- Represents different types of statistical charts -/
inductive ChartType
  | Line
  | Bar
  | Pie

/-- Represents a substance in milk powder -/
structure Substance where
  name : String
  percentage : Float

/-- Represents the composition of milk powder -/
def MilkPowderComposition := List Substance

/-- Determines if a chart type is appropriate for displaying percentage composition -/
def is_appropriate_for_percentage_composition (chart : ChartType) (composition : MilkPowderComposition) : Prop :=
  chart = ChartType.Pie

/-- Theorem stating that a pie chart is the most appropriate for displaying milk powder composition -/
theorem pie_chart_most_appropriate_for_milk_powder (composition : MilkPowderComposition) :
  is_appropriate_for_percentage_composition ChartType.Pie composition :=
by sorry

end NUMINAMATH_CALUDE_pie_chart_most_appropriate_for_milk_powder_l983_98392


namespace NUMINAMATH_CALUDE_y1_value_l983_98318

theorem y1_value (y1 y2 y3 : ℝ) 
  (h_order : 0 ≤ y3 ∧ y3 ≤ y2 ∧ y2 ≤ y1 ∧ y1 ≤ 1)
  (h_eq : (1 - y1)^2 + 2*(y1 - y2)^2 + 3*(y2 - y3)^2 + 4*y3^2 = 1/2) :
  y1 = (3 * Real.sqrt 6 - 6) / 6 := by
  sorry

end NUMINAMATH_CALUDE_y1_value_l983_98318


namespace NUMINAMATH_CALUDE_sphere_to_cube_volume_ratio_l983_98306

/-- The ratio of the volume of a sphere with diameter 12 inches to the volume of a cube with edge length 6 inches is 4π/3. -/
theorem sphere_to_cube_volume_ratio : 
  let sphere_diameter : ℝ := 12
  let cube_edge : ℝ := 6
  let sphere_volume := (4 / 3) * Real.pi * (sphere_diameter / 2) ^ 3
  let cube_volume := cube_edge ^ 3
  sphere_volume / cube_volume = (4 * Real.pi) / 3 := by
sorry

end NUMINAMATH_CALUDE_sphere_to_cube_volume_ratio_l983_98306


namespace NUMINAMATH_CALUDE_billy_soda_distribution_l983_98365

/-- Represents the number of sodas Billy can give to each sibling -/
def sodas_per_sibling (total_sodas : ℕ) (num_sisters : ℕ) : ℕ :=
  total_sodas / (num_sisters + 2 * num_sisters)

/-- Theorem stating that Billy can give 2 sodas to each sibling -/
theorem billy_soda_distribution :
  sodas_per_sibling 12 2 = 2 := by
  sorry

end NUMINAMATH_CALUDE_billy_soda_distribution_l983_98365


namespace NUMINAMATH_CALUDE_units_digit_7_pow_million_l983_98377

def units_digit_cycle_7 : List Nat := [7, 9, 3, 1]

theorem units_digit_7_pow_million :
  ∃ (n : Nat), n < 10 ∧ (7^(10^6 : Nat)) % 10 = n ∧ n = 1 :=
by
  sorry

#check units_digit_7_pow_million

end NUMINAMATH_CALUDE_units_digit_7_pow_million_l983_98377


namespace NUMINAMATH_CALUDE_geometric_arithmetic_sequence_ratio_l983_98374

theorem geometric_arithmetic_sequence_ratio (a : ℕ → ℝ) (q : ℝ) :
  (∀ n, a (n + 1) = a n * q) →  -- geometric sequence condition
  (2 * a 3 = a 1 + a 2) →       -- arithmetic sequence condition
  (q = 1 ∨ q = -1/2) :=
by sorry

end NUMINAMATH_CALUDE_geometric_arithmetic_sequence_ratio_l983_98374


namespace NUMINAMATH_CALUDE_new_total_cucumber_weight_l983_98398

/-- Calculates the new weight of cucumbers after evaporation -/
def new_cucumber_weight (initial_weight : ℝ) (water_percentage : ℝ) (evaporation_rate : ℝ) : ℝ :=
  let water_weight := initial_weight * water_percentage
  let dry_weight := initial_weight * (1 - water_percentage)
  let evaporated_water := water_weight * evaporation_rate
  (water_weight - evaporated_water) + dry_weight

/-- Theorem stating the new total weight of cucumbers after evaporation -/
theorem new_total_cucumber_weight :
  let batch1 := new_cucumber_weight 50 0.99 0.01
  let batch2 := new_cucumber_weight 30 0.98 0.02
  let batch3 := new_cucumber_weight 20 0.97 0.03
  batch1 + batch2 + batch3 = 98.335 := by
  sorry

#eval new_cucumber_weight 50 0.99 0.01 +
      new_cucumber_weight 30 0.98 0.02 +
      new_cucumber_weight 20 0.97 0.03

end NUMINAMATH_CALUDE_new_total_cucumber_weight_l983_98398


namespace NUMINAMATH_CALUDE_yacht_distance_squared_bounds_l983_98372

theorem yacht_distance_squared_bounds (θ : Real) 
  (h1 : 30 * Real.pi / 180 ≤ θ) 
  (h2 : θ ≤ 75 * Real.pi / 180) : 
  ∃ (AC : Real), 200 ≤ AC^2 ∧ AC^2 ≤ 656 := by
  sorry

end NUMINAMATH_CALUDE_yacht_distance_squared_bounds_l983_98372


namespace NUMINAMATH_CALUDE_f_properties_l983_98315

-- Define the function f(x)
def f (a : ℝ) (x : ℝ) : ℝ := a * x^2 - 2 * a * x - 3

-- State the theorem
theorem f_properties (a : ℝ) (h_a_pos : a > 0) :
  (∀ x ≥ 3, f a x ≥ 0) → a ≥ 1 ∧
  (∃ x₁ x₂ : ℝ, x₁ > 0 ∧ x₂ > 0 ∧ x₁ ≠ x₂ ∧ f a x₁ = 0 ∧ f a x₂ = 0) →
  ∃ s : ℝ, 2 < s ∧ s < 4 ∧ ∃ x₁ x₂ : ℝ, x₁ > 0 ∧ x₂ > 0 ∧ x₁ ≠ x₂ ∧ f a x₁ = 0 ∧ f a x₂ = 0 ∧ x₁^2 + x₂^2 = s :=
by sorry

end NUMINAMATH_CALUDE_f_properties_l983_98315


namespace NUMINAMATH_CALUDE_min_value_theorem_l983_98391

theorem min_value_theorem (x y : ℝ) (h1 : x * y + 3 * x = 3) (h2 : 0 < x) (h3 : x < 1/2) :
  ∀ z, z = (3 / x) + (1 / (y - 3)) → z ≥ 8 :=
by sorry

end NUMINAMATH_CALUDE_min_value_theorem_l983_98391


namespace NUMINAMATH_CALUDE_certain_number_problem_l983_98310

theorem certain_number_problem (n m : ℕ+) : 
  Nat.lcm n m = 48 →
  Nat.gcd n m = 8 →
  n = 24 →
  m = 16 := by
sorry

end NUMINAMATH_CALUDE_certain_number_problem_l983_98310


namespace NUMINAMATH_CALUDE_line_through_point_l983_98396

/-- Given a line equation -3/4 - 3kx = 7y and a point (1/3, -8) on this line,
    prove that k = 55.25 is the unique value satisfying these conditions. -/
theorem line_through_point (k : ℝ) : 
  (-3/4 - 3*k*(1/3) = 7*(-8)) ↔ k = 55.25 := by sorry

end NUMINAMATH_CALUDE_line_through_point_l983_98396


namespace NUMINAMATH_CALUDE_largest_prime_divisor_of_factorial_sum_l983_98359

def factorial (n : ℕ) : ℕ := (List.range n).foldl (· * ·) 1

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ d : ℕ, d > 1 → d < n → ¬(n % d = 0)

theorem largest_prime_divisor_of_factorial_sum :
  ∃ (p : ℕ), is_prime p ∧ 
    (p ∣ (factorial 13 + factorial 14)) ∧
    (∀ q : ℕ, is_prime q → q ∣ (factorial 13 + factorial 14) → q ≤ p) ∧
    p = 5 := by
  sorry

end NUMINAMATH_CALUDE_largest_prime_divisor_of_factorial_sum_l983_98359


namespace NUMINAMATH_CALUDE_sum_reciprocals_of_three_integers_l983_98342

theorem sum_reciprocals_of_three_integers (a b c : ℕ+) :
  a < b ∧ b < c ∧ a + b + c = 11 →
  (1 : ℚ) / a + 1 / b + 1 / c = 31 / 21 := by
  sorry

end NUMINAMATH_CALUDE_sum_reciprocals_of_three_integers_l983_98342


namespace NUMINAMATH_CALUDE_smallest_integer_with_20_divisors_l983_98354

theorem smallest_integer_with_20_divisors : 
  ∃ n : ℕ+, (n = 240) ∧ 
  (∀ m : ℕ+, m < n → (Finset.card (Nat.divisors m) ≠ 20)) ∧ 
  (Finset.card (Nat.divisors n) = 20) := by
  sorry

end NUMINAMATH_CALUDE_smallest_integer_with_20_divisors_l983_98354


namespace NUMINAMATH_CALUDE_units_digit_of_M_M7_l983_98360

def M : ℕ → ℕ
  | 0 => 3
  | 1 => 1
  | (n + 2) => 2 * M (n + 1) + M n

theorem units_digit_of_M_M7 : M (M 7) % 10 = 3 := by
  sorry

end NUMINAMATH_CALUDE_units_digit_of_M_M7_l983_98360


namespace NUMINAMATH_CALUDE_polar_equation_defines_parabola_l983_98393

/-- The polar equation r = 1 / (1 + cos θ) defines a parabola. -/
theorem polar_equation_defines_parabola :
  ∃ (a b c : ℝ), a ≠ 0 ∧
  (∀ (x y : ℝ), (∃ (r θ : ℝ), r > 0 ∧ 
    r = 1 / (1 + Real.cos θ) ∧
    x = r * Real.cos θ ∧
    y = r * Real.sin θ) ↔
    a * y^2 + b * x + c = 0) :=
sorry

end NUMINAMATH_CALUDE_polar_equation_defines_parabola_l983_98393


namespace NUMINAMATH_CALUDE_fill_three_positions_from_fifteen_l983_98358

/-- The number of ways to fill positions from a pool of candidates -/
def fill_positions (n : ℕ) (k : ℕ) : ℕ :=
  if k = 0 then 1
  else if n < k then 0
  else n * fill_positions (n - 1) (k - 1)

/-- Theorem: There are 2730 ways to fill 3 positions from 15 candidates -/
theorem fill_three_positions_from_fifteen :
  fill_positions 15 3 = 2730 := by
  sorry

end NUMINAMATH_CALUDE_fill_three_positions_from_fifteen_l983_98358


namespace NUMINAMATH_CALUDE_remainder_problem_l983_98317

theorem remainder_problem (N : ℕ) : N % 751 = 53 → N % 29 = 24 := by
  sorry

end NUMINAMATH_CALUDE_remainder_problem_l983_98317


namespace NUMINAMATH_CALUDE_ladder_slip_l983_98322

theorem ladder_slip (initial_length initial_distance slip_down slide_out : ℝ) 
  (h_length : initial_length = 30)
  (h_distance : initial_distance = 9)
  (h_slip : slip_down = 5)
  (h_slide : slide_out = 3) :
  let final_distance := initial_distance + slide_out
  final_distance = 12 := by sorry

end NUMINAMATH_CALUDE_ladder_slip_l983_98322


namespace NUMINAMATH_CALUDE_distance_difference_l983_98370

theorem distance_difference (john_distance nina_distance : ℝ) 
  (h1 : john_distance = 0.7)
  (h2 : nina_distance = 0.4) :
  john_distance - nina_distance = 0.3 := by
  sorry

end NUMINAMATH_CALUDE_distance_difference_l983_98370


namespace NUMINAMATH_CALUDE_greatest_multiple_under_1000_l983_98326

theorem greatest_multiple_under_1000 :
  ∃ n : ℕ, n < 1000 ∧ 5 ∣ n ∧ 6 ∣ n ∧ ∀ m : ℕ, m < 1000 ∧ 5 ∣ m ∧ 6 ∣ m → m ≤ n :=
by
  -- The proof goes here
  sorry

end NUMINAMATH_CALUDE_greatest_multiple_under_1000_l983_98326


namespace NUMINAMATH_CALUDE_birthday_money_calculation_l983_98327

/-- The amount of money Sam spent on baseball gear -/
def amount_spent : ℕ := 64

/-- The amount of money Sam had left over -/
def amount_left : ℕ := 23

/-- The total amount of money Sam received for his birthday -/
def total_amount : ℕ := amount_spent + amount_left

/-- Theorem stating that the total amount Sam received is the sum of what he spent and what he had left -/
theorem birthday_money_calculation : total_amount = 87 := by
  sorry

end NUMINAMATH_CALUDE_birthday_money_calculation_l983_98327


namespace NUMINAMATH_CALUDE_treasure_hunt_probability_l983_98333

def num_islands : ℕ := 6
def num_treasure_islands : ℕ := 3

def prob_treasure : ℚ := 1/4
def prob_traps : ℚ := 1/12
def prob_neither : ℚ := 2/3

theorem treasure_hunt_probability :
  (Nat.choose num_islands num_treasure_islands : ℚ) *
  prob_treasure ^ num_treasure_islands *
  prob_neither ^ (num_islands - num_treasure_islands) =
  5/54 := by sorry

end NUMINAMATH_CALUDE_treasure_hunt_probability_l983_98333


namespace NUMINAMATH_CALUDE_quadratic_solution_range_l983_98369

/-- The range of t for which the quadratic equation x^2 - 4x + 1 - t = 0 has solutions in (0, 7/2) -/
theorem quadratic_solution_range (t : ℝ) : 
  (∃ x : ℝ, 0 < x ∧ x < 7/2 ∧ x^2 - 4*x + 1 - t = 0) ↔ -3 ≤ t ∧ t < 1 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_solution_range_l983_98369


namespace NUMINAMATH_CALUDE_dealer_profit_is_sixty_percent_l983_98364

/-- Calculates the dealer's profit percentage given the purchase and sale information. -/
def dealer_profit_percentage (purchase_quantity : ℕ) (purchase_price : ℚ) 
  (sale_quantity : ℕ) (sale_price : ℚ) : ℚ :=
  let cost_price_per_article := purchase_price / purchase_quantity
  let selling_price_per_article := sale_price / sale_quantity
  let profit_per_article := selling_price_per_article - cost_price_per_article
  (profit_per_article / cost_price_per_article) * 100

/-- Theorem stating that the dealer's profit percentage is 60% given the specified conditions. -/
theorem dealer_profit_is_sixty_percent :
  dealer_profit_percentage 15 25 12 32 = 60 := by
  sorry

end NUMINAMATH_CALUDE_dealer_profit_is_sixty_percent_l983_98364


namespace NUMINAMATH_CALUDE_quadratic_sum_l983_98329

/-- A quadratic function f(x) = ax^2 + bx + c with a minimum value of 36
    and roots at x = 1 and x = 5 has the property that a + b + c = 0 -/
theorem quadratic_sum (a b c : ℝ) : 
  (∀ x, a * x^2 + b * x + c ≥ 36) ∧ 
  (∃ x₀, ∀ x, a * x^2 + b * x + c ≥ a * x₀^2 + b * x₀ + c ∧ a * x₀^2 + b * x₀ + c = 36) ∧
  (a * 1^2 + b * 1 + c = 0) ∧
  (a * 5^2 + b * 5 + c = 0) →
  a + b + c = 0 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_sum_l983_98329


namespace NUMINAMATH_CALUDE_calcium_oxide_molecular_weight_l983_98385

/-- The atomic weight of calcium in g/mol -/
def atomic_weight_Ca : ℝ := 40.08

/-- The atomic weight of oxygen in g/mol -/
def atomic_weight_O : ℝ := 16.00

/-- The number of calcium atoms in a molecule of calcium oxide -/
def num_Ca_atoms : ℕ := 1

/-- The number of oxygen atoms in a molecule of calcium oxide -/
def num_O_atoms : ℕ := 1

/-- The molecular weight of calcium oxide in g/mol -/
def molecular_weight_CaO : ℝ := atomic_weight_Ca * num_Ca_atoms + atomic_weight_O * num_O_atoms

theorem calcium_oxide_molecular_weight :
  molecular_weight_CaO = 56.08 := by sorry

end NUMINAMATH_CALUDE_calcium_oxide_molecular_weight_l983_98385


namespace NUMINAMATH_CALUDE_book_price_calculation_l983_98382

def initial_price : ℝ := 250

def week1_decrease : ℝ := 0.125
def week1_increase : ℝ := 0.30
def week2_decrease : ℝ := 0.20
def week3_increase : ℝ := 0.50

def conversion_rate : ℝ := 3
def sales_tax_rate : ℝ := 0.05

def price_after_fluctuations : ℝ :=
  initial_price * (1 - week1_decrease) * (1 + week1_increase) * (1 - week2_decrease) * (1 + week3_increase)

def price_in_currency_b : ℝ := price_after_fluctuations * conversion_rate

def final_price : ℝ := price_in_currency_b * (1 + sales_tax_rate)

theorem book_price_calculation :
  final_price = 1074.9375 := by sorry

end NUMINAMATH_CALUDE_book_price_calculation_l983_98382


namespace NUMINAMATH_CALUDE_alyssa_cookies_l983_98380

-- Define the number of cookies Aiyanna has
def aiyanna_cookies : ℕ := 140

-- Define the difference between Alyssa's and Aiyanna's cookies
def cookie_difference : ℕ := 11

-- Theorem stating Alyssa's number of cookies
theorem alyssa_cookies : 
  ∃ (a : ℕ), a = aiyanna_cookies + cookie_difference :=
by
  sorry

end NUMINAMATH_CALUDE_alyssa_cookies_l983_98380


namespace NUMINAMATH_CALUDE_teddy_pillows_l983_98363

/-- The number of pounds in a ton -/
def pounds_per_ton : ℕ := 2000

/-- The amount of fluffy foam material Teddy has, in tons -/
def teddy_material : ℕ := 3

/-- The amount of fluffy foam material used for each pillow, in pounds -/
def material_per_pillow : ℕ := 5 - 3

/-- The number of pillows Teddy can make -/
def pillows_made : ℕ := (teddy_material * pounds_per_ton) / material_per_pillow

theorem teddy_pillows :
  pillows_made = 3000 := by sorry

end NUMINAMATH_CALUDE_teddy_pillows_l983_98363


namespace NUMINAMATH_CALUDE_game_points_total_l983_98330

theorem game_points_total (layla_points nahima_points : ℕ) : 
  layla_points = 70 → 
  layla_points = nahima_points + 28 → 
  layla_points + nahima_points = 112 := by
sorry

end NUMINAMATH_CALUDE_game_points_total_l983_98330


namespace NUMINAMATH_CALUDE_james_sticker_cost_l983_98350

theorem james_sticker_cost (packs : ℕ) (stickers_per_pack : ℕ) (cost_per_sticker : ℚ) : 
  packs = 4 → 
  stickers_per_pack = 30 → 
  cost_per_sticker = 1/10 → 
  (packs * stickers_per_pack * cost_per_sticker) / 2 = 6 := by
  sorry

end NUMINAMATH_CALUDE_james_sticker_cost_l983_98350


namespace NUMINAMATH_CALUDE_slope_of_line_l983_98375

theorem slope_of_line (x y : ℝ) :
  (4 * y = -5 * x + 8) → (y = (-5/4) * x + 2) :=
by sorry

end NUMINAMATH_CALUDE_slope_of_line_l983_98375


namespace NUMINAMATH_CALUDE_three_equal_differences_l983_98386

theorem three_equal_differences (n : ℕ) (a : Fin (2 * n) → ℕ) 
  (h1 : n > 2)
  (h2 : ∀ i j, i ≠ j → a i ≠ a j)
  (h3 : ∀ i, a i ≤ n^2)
  (h4 : ∀ i, a i > 0) :
  ∃ (i1 j1 i2 j2 i3 j3 : Fin (2 * n)), 
    (i1 > j1 ∧ i2 > j2 ∧ i3 > j3) ∧ 
    (i1 ≠ i2 ∨ j1 ≠ j2) ∧ 
    (i1 ≠ i3 ∨ j1 ≠ j3) ∧ 
    (i2 ≠ i3 ∨ j2 ≠ j3) ∧
    (a i1 - a j1 = a i2 - a j2) ∧ 
    (a i1 - a j1 = a i3 - a j3) :=
by sorry

end NUMINAMATH_CALUDE_three_equal_differences_l983_98386


namespace NUMINAMATH_CALUDE_roots_sum_reciprocals_l983_98379

theorem roots_sum_reciprocals (α β : ℝ) : 
  3 * α^2 + α - 1 = 0 →
  3 * β^2 + β - 1 = 0 →
  α > β →
  (α / β) + (β / α) = -7/3 := by
sorry

end NUMINAMATH_CALUDE_roots_sum_reciprocals_l983_98379


namespace NUMINAMATH_CALUDE_remaining_fruit_cost_is_eight_l983_98316

/-- Represents the cost of fruit remaining in Tanya's bag after half fell out --/
def remaining_fruit_cost (pear_count : ℕ) (pear_price : ℚ) 
                         (apple_count : ℕ) (apple_price : ℚ)
                         (pineapple_count : ℕ) (pineapple_price : ℚ) : ℚ :=
  ((pear_count : ℚ) * pear_price + 
   (apple_count : ℚ) * apple_price + 
   (pineapple_count : ℚ) * pineapple_price) / 2

/-- Theorem stating the cost of remaining fruit excluding plums --/
theorem remaining_fruit_cost_is_eight :
  remaining_fruit_cost 6 1.5 4 0.75 2 2 = 8 := by
  sorry

end NUMINAMATH_CALUDE_remaining_fruit_cost_is_eight_l983_98316


namespace NUMINAMATH_CALUDE_forest_trees_count_l983_98376

/-- The side length of the square-shaped street in meters -/
def street_side_length : ℝ := 100

/-- The area of the square-shaped street in square meters -/
def street_area : ℝ := street_side_length ^ 2

/-- The area of the forest in square meters -/
def forest_area : ℝ := 3 * street_area

/-- The number of trees per square meter in the forest -/
def trees_per_square_meter : ℝ := 4

/-- The total number of trees in the forest -/
def total_trees : ℝ := forest_area * trees_per_square_meter

theorem forest_trees_count : total_trees = 120000 := by
  sorry

end NUMINAMATH_CALUDE_forest_trees_count_l983_98376


namespace NUMINAMATH_CALUDE_root_equality_condition_l983_98340

theorem root_equality_condition (m n p : ℕ) 
  (hm : Even m) (hn : Even n) (hp : Even p) 
  (hm_pos : m > 0) (hn_pos : n > 0) (hp_pos : p > 0) :
  (m - p : ℝ) ^ (1 / n) = (n - p : ℝ) ^ (1 / m) ↔ m = n ∧ m ≥ p :=
sorry

end NUMINAMATH_CALUDE_root_equality_condition_l983_98340


namespace NUMINAMATH_CALUDE_sprint_medal_awarding_ways_l983_98387

/-- The number of ways to award medals in an international sprint final --/
def medalAwardingWays (totalSprinters : ℕ) (americanSprinters : ℕ) (medals : ℕ) : ℕ :=
  -- We'll define this function without implementation
  sorry

/-- Theorem stating the number of ways to award medals under given conditions --/
theorem sprint_medal_awarding_ways :
  medalAwardingWays 10 4 3 = 696 :=
by
  sorry

end NUMINAMATH_CALUDE_sprint_medal_awarding_ways_l983_98387


namespace NUMINAMATH_CALUDE_cos_sin_identity_l983_98308

theorem cos_sin_identity (α : Real) (h : Real.cos (π / 6 - α) = Real.sqrt 3 / 3) :
  Real.cos (5 * π / 6 + α) - Real.sin (α - π / 6) ^ 2 = -(2 + Real.sqrt 3) / 3 := by
  sorry

end NUMINAMATH_CALUDE_cos_sin_identity_l983_98308


namespace NUMINAMATH_CALUDE_max_russian_score_l983_98301

/-- Represents a chess player -/
structure Player where
  country : String
  score : ℚ

/-- Represents a chess tournament -/
structure Tournament where
  players : Finset Player
  russianPlayers : Finset Player
  winner : Player
  runnerUp : Player

/-- The scoring system for the tournament -/
def scoringSystem : ℚ × ℚ × ℚ := (1, 1/2, 0)

/-- Theorem statement for the maximum score of Russian players -/
theorem max_russian_score (t : Tournament) : 
  t.players.card = 20 ∧ 
  t.russianPlayers.card = 6 ∧
  t.winner.country = "Russia" ∧
  t.runnerUp.country = "Armenia" ∧
  t.winner.score > t.runnerUp.score ∧
  (∀ p ∈ t.players, p ≠ t.winner → p ≠ t.runnerUp → t.runnerUp.score > p.score) →
  (t.russianPlayers.sum (λ p => p.score)) ≤ 96 := by
  sorry

end NUMINAMATH_CALUDE_max_russian_score_l983_98301


namespace NUMINAMATH_CALUDE_floor_sum_inequality_l983_98339

-- Define the floor function
noncomputable def floor (x : ℝ) : ℤ := 
  Int.floor x

-- Define the fractional part function
noncomputable def frac (x : ℝ) : ℝ := 
  x - floor x

-- State the theorem
theorem floor_sum_inequality (x y : ℝ) : 
  (floor x + floor y ≤ floor (x + y)) ∧ 
  (floor (x + y) ≤ floor x + floor y + 1) ∧ 
  (floor x + floor y = floor (x + y) ∨ floor (x + y) = floor x + floor y + 1) :=
sorry

end NUMINAMATH_CALUDE_floor_sum_inequality_l983_98339


namespace NUMINAMATH_CALUDE_second_class_average_l983_98320

theorem second_class_average (n₁ n₂ : ℕ) (avg₁ avg_total : ℚ) : 
  n₁ = 12 → 
  n₂ = 28 → 
  avg₁ = 40 → 
  avg_total = 54 → 
  ∃ avg₂ : ℚ, 
    (n₁ * avg₁ + n₂ * avg₂) / (n₁ + n₂) = avg_total ∧ 
    avg₂ = 60 :=
by sorry

end NUMINAMATH_CALUDE_second_class_average_l983_98320


namespace NUMINAMATH_CALUDE_two_bedroom_units_count_l983_98323

theorem two_bedroom_units_count 
  (total_units : ℕ) 
  (one_bedroom_cost two_bedroom_cost : ℕ) 
  (total_cost : ℕ) 
  (h1 : total_units = 12)
  (h2 : one_bedroom_cost = 360)
  (h3 : two_bedroom_cost = 450)
  (h4 : total_cost = 4950) :
  ∃ (one_bedroom_count two_bedroom_count : ℕ),
    one_bedroom_count + two_bedroom_count = total_units ∧
    one_bedroom_count * one_bedroom_cost + two_bedroom_count * two_bedroom_cost = total_cost ∧
    two_bedroom_count = 7 := by
  sorry

end NUMINAMATH_CALUDE_two_bedroom_units_count_l983_98323


namespace NUMINAMATH_CALUDE_scientific_notation_of_175_billion_l983_98384

theorem scientific_notation_of_175_billion : ∃ (a : ℝ) (n : ℤ), 
  175000000000 = a * (10 : ℝ) ^ n ∧ 1 ≤ a ∧ a < 10 ∧ a = 1.75 ∧ n = 11 :=
sorry

end NUMINAMATH_CALUDE_scientific_notation_of_175_billion_l983_98384


namespace NUMINAMATH_CALUDE_spelling_bee_initial_students_l983_98378

theorem spelling_bee_initial_students :
  ∀ (initial_students : ℕ),
    (initial_students : ℝ) * 0.3 * 0.5 = 18 →
    initial_students = 120 :=
by
  sorry

end NUMINAMATH_CALUDE_spelling_bee_initial_students_l983_98378


namespace NUMINAMATH_CALUDE_cylinder_cone_sphere_volume_l983_98307

/-- Given a cylinder with volume 150π cm³, prove that the sum of the volumes of a cone 
    with the same base radius and height as the cylinder, and a sphere with the same 
    radius as the cylinder, is equal to 50π + (4/3)π * (∛150)² cm³. -/
theorem cylinder_cone_sphere_volume 
  (r h : ℝ) 
  (h_cylinder_volume : π * r^2 * h = 150 * π) :
  (1/3 * π * r^2 * h) + (4/3 * π * r^3) = 50 * π + 4/3 * π * (150^(2/3)) := by
  sorry

end NUMINAMATH_CALUDE_cylinder_cone_sphere_volume_l983_98307


namespace NUMINAMATH_CALUDE_gold_bars_per_row_l983_98373

/-- Represents the arrangement of gold bars in a safe -/
structure GoldSafe where
  rows : Nat
  totalWorth : Nat
  barValue : Nat

/-- Calculates the number of gold bars per row in a safe -/
def barsPerRow (safe : GoldSafe) : Nat :=
  (safe.totalWorth / safe.barValue) / safe.rows

/-- Theorem: If a safe has 4 rows, total worth of $1,600,000, and each bar is worth $40,000,
    then there are 10 gold bars in each row -/
theorem gold_bars_per_row :
  let safe : GoldSafe := { rows := 4, totalWorth := 1600000, barValue := 40000 }
  barsPerRow safe = 10 := by
  sorry


end NUMINAMATH_CALUDE_gold_bars_per_row_l983_98373


namespace NUMINAMATH_CALUDE_medium_supermarkets_in_sample_l983_98311

/-- Represents the number of supermarkets in each category -/
structure SupermarketCounts where
  large : ℕ
  medium : ℕ
  small : ℕ

/-- Calculates the total number of supermarkets -/
def total_supermarkets (counts : SupermarketCounts) : ℕ :=
  counts.large + counts.medium + counts.small

/-- Calculates the number of supermarkets of a given category in a stratified sample -/
def stratified_sample_count (counts : SupermarketCounts) (sample_size : ℕ) (category : ℕ) : ℕ :=
  (category * sample_size) / (total_supermarkets counts)

/-- Theorem stating the number of medium-sized supermarkets in the stratified sample -/
theorem medium_supermarkets_in_sample 
  (counts : SupermarketCounts) 
  (sample_size : ℕ) : 
  counts.large = 200 → 
  counts.medium = 400 → 
  counts.small = 1400 → 
  sample_size = 100 → 
  stratified_sample_count counts sample_size counts.medium = 20 := by
  sorry

end NUMINAMATH_CALUDE_medium_supermarkets_in_sample_l983_98311


namespace NUMINAMATH_CALUDE_pascal_triangle_interior_sum_l983_98321

/-- Sum of interior numbers in a row of Pascal's Triangle -/
def interior_sum (n : ℕ) : ℕ := 2^(n-1) - 2

/-- The problem statement -/
theorem pascal_triangle_interior_sum :
  interior_sum 6 = 30 →
  interior_sum 8 = 126 := by
sorry

end NUMINAMATH_CALUDE_pascal_triangle_interior_sum_l983_98321


namespace NUMINAMATH_CALUDE_two_digit_product_digits_l983_98344

theorem two_digit_product_digits :
  ∀ a b : ℕ,
  10 ≤ a ∧ a ≤ 99 →
  10 ≤ b ∧ b ≤ 99 →
  (100 ≤ a * b ∧ a * b ≤ 9999) :=
by sorry

end NUMINAMATH_CALUDE_two_digit_product_digits_l983_98344


namespace NUMINAMATH_CALUDE_oblique_square_area_l983_98312

/-- The area of an oblique two-dimensional drawing of a unit square -/
theorem oblique_square_area :
  ∀ (S_oblique : ℝ),
  (1 : ℝ) ^ 2 = 1 →  -- Side length of original square is 1
  S_oblique / 1 = Real.sqrt 2 / 4 →  -- Ratio of areas
  S_oblique = Real.sqrt 2 / 4 := by
sorry

end NUMINAMATH_CALUDE_oblique_square_area_l983_98312


namespace NUMINAMATH_CALUDE_quadratic_discriminant_perfect_square_l983_98335

theorem quadratic_discriminant_perfect_square 
  (a b c t : ℝ) 
  (h1 : a ≠ 0) 
  (h2 : a * t^2 + b * t + c = 0) : 
  b^2 - 4*a*c = (2*a*t + b)^2 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_discriminant_perfect_square_l983_98335


namespace NUMINAMATH_CALUDE_inequality_proof_l983_98357

/-- If for all real x, 1 - a cos x - b sin x - A cos 2x - B sin 2x ≥ 0, 
    then a² + b² ≤ 2 and A² + B² ≤ 1 -/
theorem inequality_proof (a b A B : ℝ) 
  (h : ∀ x : ℝ, 1 - a * Real.cos x - b * Real.sin x - A * Real.cos (2 * x) - B * Real.sin (2 * x) ≥ 0) : 
  a^2 + b^2 ≤ 2 ∧ A^2 + B^2 ≤ 1 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l983_98357


namespace NUMINAMATH_CALUDE_base_conversion_theorem_l983_98304

theorem base_conversion_theorem (n : ℕ+) (A B : ℕ) : 
  (A < 8 ∧ B < 5) →
  (8 * A + B = n) →
  (5 * B + A = n) →
  (n : ℕ) = 33 := by
  sorry

end NUMINAMATH_CALUDE_base_conversion_theorem_l983_98304


namespace NUMINAMATH_CALUDE_nineteenth_term_is_zero_l983_98309

/-- A sequence with specific properties -/
def special_sequence (a : ℕ → ℝ) : Prop :=
  a 3 = 2 ∧ 
  a 7 = 1 ∧ 
  ∃ d : ℝ, ∀ n : ℕ, 1 / (a (n + 1) + 1) - 1 / (a n + 1) = d

/-- Theorem stating that for a special sequence, the 19th term is 0 -/
theorem nineteenth_term_is_zero (a : ℕ → ℝ) (h : special_sequence a) : 
  a 19 = 0 := by sorry

end NUMINAMATH_CALUDE_nineteenth_term_is_zero_l983_98309


namespace NUMINAMATH_CALUDE_negation_relationship_l983_98348

theorem negation_relationship (a : ℝ) : 
  ¬(∀ a, a > 0 → a^2 > a) ∧ ¬(∀ a, a^2 ≤ a → a ≤ 0) := by
  sorry

end NUMINAMATH_CALUDE_negation_relationship_l983_98348


namespace NUMINAMATH_CALUDE_roots_form_parallelogram_l983_98324

/-- The polynomial whose roots we're investigating -/
def P (b : ℝ) (z : ℂ) : ℂ := z^4 - 8*z^3 + 17*b*z^2 - 2*(3*b^2 + 4*b - 4)*z + 9

/-- A function that checks if four complex numbers form a parallelogram -/
def isParallelogram (z₁ z₂ z₃ z₄ : ℂ) : Prop := 
  (z₁ + z₃ = z₂ + z₄) ∧ (z₁ - z₂ = z₄ - z₃)

/-- The main theorem stating the values of b for which the roots form a parallelogram -/
theorem roots_form_parallelogram : 
  ∀ b : ℝ, (∃ z₁ z₂ z₃ z₄ : ℂ, 
    (P b z₁ = 0) ∧ (P b z₂ = 0) ∧ (P b z₃ = 0) ∧ (P b z₄ = 0) ∧ 
    isParallelogram z₁ z₂ z₃ z₄) ↔ 
  (b = 7/3 ∨ b = 2) :=
sorry

end NUMINAMATH_CALUDE_roots_form_parallelogram_l983_98324


namespace NUMINAMATH_CALUDE_gcd_plus_binary_sum_l983_98371

def binary_to_decimal (b : List Bool) : Nat :=
  b.enum.foldl (fun acc (i, bit) => acc + if bit then 2^i else 0) 0

theorem gcd_plus_binary_sum : 
  let a := Nat.gcd 98 63
  let b := binary_to_decimal [true, true, false, false, true, true]
  a + b = 58 := by sorry

end NUMINAMATH_CALUDE_gcd_plus_binary_sum_l983_98371


namespace NUMINAMATH_CALUDE_square_of_product_l983_98325

theorem square_of_product (a b : ℝ) : (a * b)^2 = a^2 * b^2 := by
  sorry

end NUMINAMATH_CALUDE_square_of_product_l983_98325


namespace NUMINAMATH_CALUDE_necklace_price_calculation_l983_98388

def polo_shirt_price : ℕ := 26
def polo_shirt_quantity : ℕ := 3
def necklace_quantity : ℕ := 2
def computer_game_price : ℕ := 90
def computer_game_quantity : ℕ := 1
def rebate : ℕ := 12
def total_cost_after_rebate : ℕ := 322

theorem necklace_price_calculation (necklace_price : ℕ) : 
  polo_shirt_price * polo_shirt_quantity + 
  necklace_price * necklace_quantity + 
  computer_game_price * computer_game_quantity - 
  rebate = total_cost_after_rebate → 
  necklace_price = 83 := by sorry

end NUMINAMATH_CALUDE_necklace_price_calculation_l983_98388


namespace NUMINAMATH_CALUDE_triangle_division_perimeter_l983_98319

/-- A structure representing a triangle division scenario -/
structure TriangleDivision where
  large_perimeter : ℝ
  num_small_triangles : ℕ
  small_perimeter : ℝ

/-- The theorem statement -/
theorem triangle_division_perimeter 
  (td : TriangleDivision) 
  (h1 : td.large_perimeter = 120)
  (h2 : td.num_small_triangles = 9)
  (h3 : td.small_perimeter * 3 = td.large_perimeter) :
  td.small_perimeter = 40 := by
  sorry

end NUMINAMATH_CALUDE_triangle_division_perimeter_l983_98319


namespace NUMINAMATH_CALUDE_binomial_coefficient_six_choose_two_l983_98366

theorem binomial_coefficient_six_choose_two : 
  Nat.choose 6 2 = 15 := by
  sorry

end NUMINAMATH_CALUDE_binomial_coefficient_six_choose_two_l983_98366


namespace NUMINAMATH_CALUDE_arithmetic_sequence_special_condition_l983_98394

/-- An arithmetic sequence with positive terms -/
structure ArithmeticSequence where
  a : ℕ → ℝ
  positive : ∀ n, a n > 0
  is_arithmetic : ∀ n, a (n + 1) - a n = a (n + 2) - a (n + 1)

/-- Theorem stating that if 3a_6 - a_7^2 + 3a_8 = 0 in an arithmetic sequence with positive terms, then a_7 = 6 -/
theorem arithmetic_sequence_special_condition
  (seq : ArithmeticSequence)
  (h : 3 * seq.a 6 - (seq.a 7)^2 + 3 * seq.a 8 = 0) :
  seq.a 7 = 6 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_special_condition_l983_98394


namespace NUMINAMATH_CALUDE_unique_solution_square_equation_l983_98381

theorem unique_solution_square_equation :
  ∃! y : ℤ, (2010 + y)^2 = y^2 ∧ y = -1005 := by sorry

end NUMINAMATH_CALUDE_unique_solution_square_equation_l983_98381


namespace NUMINAMATH_CALUDE_full_price_tickets_count_l983_98349

/-- Represents the number of tickets sold at each price point -/
structure TicketSales where
  full : ℕ
  half : ℕ
  double : ℕ

/-- Represents the price of a full-price ticket -/
def FullPrice : ℕ := 30

/-- The total number of tickets sold -/
def TotalTickets : ℕ := 200

/-- The total revenue from all ticket sales -/
def TotalRevenue : ℕ := 3600

/-- Calculates the total number of tickets sold -/
def totalTicketCount (sales : TicketSales) : ℕ :=
  sales.full + sales.half + sales.double

/-- Calculates the total revenue from all ticket sales -/
def totalRevenue (sales : TicketSales) : ℕ :=
  sales.full * FullPrice + sales.half * (FullPrice / 2) + sales.double * (2 * FullPrice)

/-- Theorem stating that the number of full-price tickets sold is 80 -/
theorem full_price_tickets_count :
  ∃ (sales : TicketSales),
    totalTicketCount sales = TotalTickets ∧
    totalRevenue sales = TotalRevenue ∧
    sales.full = 80 :=
by sorry

end NUMINAMATH_CALUDE_full_price_tickets_count_l983_98349


namespace NUMINAMATH_CALUDE_function_properties_l983_98334

noncomputable def f (ω θ : ℝ) (x : ℝ) : ℝ := 2 * Real.cos (ω * x + θ)

theorem function_properties (ω θ : ℝ) (h_ω : ω > 0) (h_θ : 0 ≤ θ ∧ θ ≤ π/2)
  (h_intersect : f ω θ 0 = Real.sqrt 3)
  (h_period : ∀ x, f ω θ (x + π) = f ω θ x) :
  (θ = π/6 ∧ ω = 2) ∧
  (∃ x₀ ∈ Set.Icc (π/2) π,
    let y₀ := Real.sqrt 3 / 2
    let x₁ := 2 * x₀ - π/2
    let y₁ := f ω θ x₁
    y₀ = (y₁ + 0) / 2 ∧ (x₀ = 2*π/3 ∨ x₀ = 3*π/4)) :=
by sorry

end NUMINAMATH_CALUDE_function_properties_l983_98334


namespace NUMINAMATH_CALUDE_even_function_implies_a_equals_two_l983_98368

/-- A function f: ℝ → ℝ is even if f(-x) = f(x) for all x ∈ ℝ -/
def IsEven (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f (-x) = f x

/-- The function f(x) = (x+a)(x-2) -/
def f (a : ℝ) (x : ℝ) : ℝ := (x + a) * (x - 2)

/-- If f(x) = (x+a)(x-2) is an even function, then a = 2 -/
theorem even_function_implies_a_equals_two :
  ∀ a : ℝ, IsEven (f a) → a = 2 := by
  sorry

end NUMINAMATH_CALUDE_even_function_implies_a_equals_two_l983_98368


namespace NUMINAMATH_CALUDE_min_dot_product_on_ellipse_l983_98302

-- Define the ellipse
def is_on_ellipse (x y : ℝ) : Prop := x^2 / 9 + y^2 / 8 = 1

-- Define the center and focus
def O : ℝ × ℝ := (0, 0)
def F : ℝ × ℝ := (-1, 0)

-- Define the dot product of vectors
def dot_product (v1 v2 : ℝ × ℝ) : ℝ :=
  v1.1 * v2.1 + v1.2 * v2.2

-- State the theorem
theorem min_dot_product_on_ellipse :
  ∀ P : ℝ × ℝ, is_on_ellipse P.1 P.2 →
    ∃ min_value : ℝ, min_value = 6 ∧
      ∀ Q : ℝ × ℝ, is_on_ellipse Q.1 Q.2 →
        dot_product (Q.1 - O.1, Q.2 - O.2) (Q.1 - F.1, Q.2 - F.2) ≥ min_value :=
by sorry

end NUMINAMATH_CALUDE_min_dot_product_on_ellipse_l983_98302


namespace NUMINAMATH_CALUDE_problem_statement_l983_98336

theorem problem_statement (x y : ℝ) (h : |2*x - y| + Real.sqrt (x + 3*y - 7) = 0) :
  (Real.sqrt ((x - y)^2)) / (y - x) = 1 := by
sorry

end NUMINAMATH_CALUDE_problem_statement_l983_98336


namespace NUMINAMATH_CALUDE_smallest_total_blocks_smallest_total_blocks_exist_l983_98337

/-- Represents the dimensions of a cubic block -/
structure Block where
  length : ℕ
  width : ℕ
  height : ℕ

/-- Represents a cubic pedestal -/
structure Pedestal where
  sideLength : ℕ

/-- Represents a square foundation -/
structure Foundation where
  sideLength : ℕ
  thickness : ℕ

/-- Calculates the volume of a pedestal in terms of blocks -/
def pedestalVolume (p : Pedestal) : ℕ :=
  p.sideLength ^ 3

/-- Calculates the volume of a foundation in terms of blocks -/
def foundationVolume (f : Foundation) : ℕ :=
  f.sideLength ^ 2 * f.thickness

theorem smallest_total_blocks : ℕ × ℕ → Prop
  | (pedestal_side, foundation_side) =>
    let block : Block := ⟨1, 1, 1⟩
    let pedestal : Pedestal := ⟨pedestal_side⟩
    let foundation : Foundation := ⟨foundation_side, 1⟩
    (pedestalVolume pedestal = foundationVolume foundation) ∧
    (foundation_side = pedestal_side ^ (3/2)) ∧
    (pedestalVolume pedestal + foundationVolume foundation = 128) ∧
    ∀ (p : Pedestal) (f : Foundation),
      (pedestalVolume p = foundationVolume f) →
      (f.sideLength = p.sideLength ^ (3/2)) →
      (pedestalVolume p + foundationVolume f ≥ 128)

theorem smallest_total_blocks_exist :
  ∃ (pedestal_side foundation_side : ℕ),
    smallest_total_blocks (pedestal_side, foundation_side) :=
  sorry

end NUMINAMATH_CALUDE_smallest_total_blocks_smallest_total_blocks_exist_l983_98337


namespace NUMINAMATH_CALUDE_max_absolute_value_quadratic_l983_98390

theorem max_absolute_value_quadratic (a b : ℝ) :
  (∃ m : ℝ, ∀ t ∈ Set.Icc 0 4, ∃ t' ∈ Set.Icc 0 4, |t'^2 + a*t' + b| ≥ m) ∧
  (∀ m : ℝ, (∀ t ∈ Set.Icc 0 4, ∃ t' ∈ Set.Icc 0 4, |t'^2 + a*t' + b| ≥ m) → m ≤ 2) :=
by sorry

end NUMINAMATH_CALUDE_max_absolute_value_quadratic_l983_98390


namespace NUMINAMATH_CALUDE_regular_polygon_interior_angle_sum_l983_98313

theorem regular_polygon_interior_angle_sum (n : ℕ) (h1 : n > 2) : 
  (360 / n = 45) → (n - 2) * 180 = 1080 := by
  sorry

end NUMINAMATH_CALUDE_regular_polygon_interior_angle_sum_l983_98313


namespace NUMINAMATH_CALUDE_elvis_songwriting_time_l983_98353

theorem elvis_songwriting_time (total_songs : ℕ) (studio_time : ℕ) (recording_time : ℕ) (editing_time : ℕ)
  (h1 : total_songs = 10)
  (h2 : studio_time = 5 * 60)  -- 5 hours in minutes
  (h3 : recording_time = 12)   -- 12 minutes per song
  (h4 : editing_time = 30)     -- 30 minutes for all songs
  : (studio_time - (total_songs * recording_time + editing_time)) / total_songs = 15 := by
  sorry

end NUMINAMATH_CALUDE_elvis_songwriting_time_l983_98353

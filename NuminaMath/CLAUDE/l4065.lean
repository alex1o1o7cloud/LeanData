import Mathlib

namespace NUMINAMATH_CALUDE_closest_integer_to_seven_times_three_fourths_l4065_406577

theorem closest_integer_to_seven_times_three_fourths : 
  ∃ (n : ℤ), ∀ (m : ℤ), |n - (7 * 3 / 4)| ≤ |m - (7 * 3 / 4)| ∧ n = 5 :=
by sorry

end NUMINAMATH_CALUDE_closest_integer_to_seven_times_three_fourths_l4065_406577


namespace NUMINAMATH_CALUDE_union_and_subset_conditions_l4065_406524

def A : Set ℝ := {x | -2 ≤ x ∧ x ≤ 5}
def B (m : ℝ) : Set ℝ := {x | m + 1 ≤ x ∧ x ≤ 2*m - 1}

theorem union_and_subset_conditions :
  (∀ m : ℝ, m = 4 → A ∪ B m = {x | -2 ≤ x ∧ x ≤ 7}) ∧
  (∀ m : ℝ, B m ⊆ A ↔ m ≤ 3) := by sorry

end NUMINAMATH_CALUDE_union_and_subset_conditions_l4065_406524


namespace NUMINAMATH_CALUDE_complex_sum_l4065_406552

theorem complex_sum (z : ℂ) (h : z^2 + z + 1 = 0) : 
  z^101 + z^102 + z^103 + z^104 + z^105 = -2 := by
  sorry

end NUMINAMATH_CALUDE_complex_sum_l4065_406552


namespace NUMINAMATH_CALUDE_series_sum_l4065_406537

/-- The sum of the infinite series Σ(n=1 to ∞) ((3n - 2) / (n(n+1)(n+3))) equals 61/24 -/
theorem series_sum : ∑' n, (3 * n - 2) / (n * (n + 1) * (n + 3)) = 61 / 24 := by
  sorry

end NUMINAMATH_CALUDE_series_sum_l4065_406537


namespace NUMINAMATH_CALUDE_negative_one_powers_equality_l4065_406597

theorem negative_one_powers_equality : -1^2022 - (-1)^2023 - (-1)^0 = -1 := by
  sorry

end NUMINAMATH_CALUDE_negative_one_powers_equality_l4065_406597


namespace NUMINAMATH_CALUDE_rationalize_denominator_l4065_406518

theorem rationalize_denominator :
  (Real.sqrt 12 + Real.sqrt 5) / (Real.sqrt 3 + Real.sqrt 5) = (Real.sqrt 15 - 1) / 2 := by
  sorry

end NUMINAMATH_CALUDE_rationalize_denominator_l4065_406518


namespace NUMINAMATH_CALUDE_inequality_of_powers_l4065_406568

theorem inequality_of_powers (α : Real) (h : α ∈ Set.Ioo (π/4) (π/2)) :
  (Real.cos α) ^ (Real.sin α) < (Real.cos α) ^ (Real.cos α) ∧
  (Real.cos α) ^ (Real.cos α) < (Real.sin α) ^ (Real.cos α) := by
  sorry

end NUMINAMATH_CALUDE_inequality_of_powers_l4065_406568


namespace NUMINAMATH_CALUDE_planes_parallel_if_perpendicular_to_same_line_l4065_406504

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the perpendicular relation between a line and a plane
variable (perpendicular : Line → Plane → Prop)

-- Define the parallel relation between two planes
variable (parallel : Plane → Plane → Prop)

-- State the theorem
theorem planes_parallel_if_perpendicular_to_same_line 
  (a : Line) (α β : Plane) :
  perpendicular a α → perpendicular a β → parallel α β :=
sorry

end NUMINAMATH_CALUDE_planes_parallel_if_perpendicular_to_same_line_l4065_406504


namespace NUMINAMATH_CALUDE_solve_for_d_l4065_406514

theorem solve_for_d (x d : ℝ) (h1 : x = 0.3) 
  (h2 : (10 * x + 2) / 4 - (d * x - 6) / 18 = (2 * x + 4) / 3) : d = 3 := by
  sorry

end NUMINAMATH_CALUDE_solve_for_d_l4065_406514


namespace NUMINAMATH_CALUDE_right_triangle_hypotenuse_l4065_406517

theorem right_triangle_hypotenuse (a b c : ℝ) : 
  a = 6 → b = 8 → c^2 = a^2 + b^2 → c = 10 := by sorry

end NUMINAMATH_CALUDE_right_triangle_hypotenuse_l4065_406517


namespace NUMINAMATH_CALUDE_sum_of_squares_is_384_l4065_406561

/-- Represents the rates of cycling, jogging, and swimming -/
structure Rates where
  cycling : ℕ
  jogging : ℕ
  swimming : ℕ

/-- The conditions of the problem -/
def satisfies_conditions (r : Rates) : Prop :=
  -- Rates are even
  r.cycling % 2 = 0 ∧ r.jogging % 2 = 0 ∧ r.swimming % 2 = 0 ∧
  -- Ed's distance equation
  3 * r.cycling + 4 * r.jogging + 2 * r.swimming = 88 ∧
  -- Sue's distance equation
  2 * r.cycling + 3 * r.jogging + 4 * r.swimming = 104

/-- The theorem to be proved -/
theorem sum_of_squares_is_384 :
  ∃ r : Rates, satisfies_conditions r ∧ 
    r.cycling^2 + r.jogging^2 + r.swimming^2 = 384 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_squares_is_384_l4065_406561


namespace NUMINAMATH_CALUDE_max_real_part_sum_l4065_406527

theorem max_real_part_sum (z : Fin 18 → ℂ) (w : Fin 18 → ℂ) : 
  (∀ j : Fin 18, z j ^ 18 = (2 : ℂ) ^ 54) →
  (∀ j : Fin 18, w j = z j ∨ w j = Complex.I * z j ∨ w j = -z j) →
  (∃ w_choice : Fin 18 → ℂ, 
    (∀ j : Fin 18, w_choice j = z j ∨ w_choice j = Complex.I * z j ∨ w_choice j = -z j) ∧
    (Finset.sum Finset.univ (λ j => (w_choice j).re) = 
      8 + 8 * (2 * (1 + Real.sqrt 3 + Real.sqrt 2 + 
        Real.cos (π / 9) + Real.cos (2 * π / 9) + Real.cos (4 * π / 9) + 
        Real.cos (5 * π / 9) + Real.cos (7 * π / 9) + Real.cos (8 * π / 9))))) ∧
  (∀ w_alt : Fin 18 → ℂ, 
    (∀ j : Fin 18, w_alt j = z j ∨ w_alt j = Complex.I * z j ∨ w_alt j = -z j) →
    Finset.sum Finset.univ (λ j => (w_alt j).re) ≤ 
      8 + 8 * (2 * (1 + Real.sqrt 3 + Real.sqrt 2 + 
        Real.cos (π / 9) + Real.cos (2 * π / 9) + Real.cos (4 * π / 9) + 
        Real.cos (5 * π / 9) + Real.cos (7 * π / 9) + Real.cos (8 * π / 9)))) := by
  sorry

end NUMINAMATH_CALUDE_max_real_part_sum_l4065_406527


namespace NUMINAMATH_CALUDE_binomial_coefficient_19_12_l4065_406566

theorem binomial_coefficient_19_12 : 
  (Nat.choose 20 12 = 125970) → 
  (Nat.choose 19 11 = 75582) → 
  (Nat.choose 18 11 = 31824) → 
  (Nat.choose 19 12 = 50388) := by
sorry

end NUMINAMATH_CALUDE_binomial_coefficient_19_12_l4065_406566


namespace NUMINAMATH_CALUDE_function_property_l4065_406571

/-- Given a function f(x) = ax² - bx where a and b are positive constants,
    if f(f(1)) = -1 and √(ab) = 3, then a = 1 or a = 2. -/
theorem function_property (a b : ℝ) (ha : 0 < a) (hb : 0 < b) :
  let f : ℝ → ℝ := λ x => a * x^2 - b * x
  (f (f 1) = -1) ∧ (Real.sqrt (a * b) = 3) → a = 1 ∨ a = 2 := by
  sorry

end NUMINAMATH_CALUDE_function_property_l4065_406571


namespace NUMINAMATH_CALUDE_final_selling_price_approx_1949_l4065_406501

/-- Calculate the final selling price of a cycle, helmet, and safety lights --/
def calculate_final_selling_price (cycle_cost helmet_cost safety_light_cost : ℚ) 
  (num_safety_lights : ℕ) (cycle_discount tax_rate cycle_loss helmet_profit transaction_fee : ℚ) : ℚ :=
  let cycle_discounted := cycle_cost * (1 - cycle_discount)
  let total_cost := cycle_discounted + helmet_cost + (safety_light_cost * num_safety_lights)
  let total_with_tax := total_cost * (1 + tax_rate)
  let cycle_selling := cycle_discounted * (1 - cycle_loss)
  let helmet_selling := helmet_cost * (1 + helmet_profit)
  let safety_lights_selling := safety_light_cost * num_safety_lights
  let total_selling := cycle_selling + helmet_selling + safety_lights_selling
  let final_price := total_selling * (1 - transaction_fee)
  final_price

/-- Theorem stating that the final selling price is approximately 1949 --/
theorem final_selling_price_approx_1949 : 
  ∃ (ε : ℚ), ε > 0 ∧ ε < 1 ∧ 
  |calculate_final_selling_price 1400 400 200 2 (10/100) (5/100) (12/100) (25/100) (3/100) - 1949| < ε :=
sorry

end NUMINAMATH_CALUDE_final_selling_price_approx_1949_l4065_406501


namespace NUMINAMATH_CALUDE_final_price_after_two_reductions_l4065_406531

/-- Given an original price and two identical percentage reductions, 
    calculate the final price after the reductions. -/
def final_price (original_price : ℝ) (reduction_percentage : ℝ) : ℝ :=
  original_price * (1 - reduction_percentage)^2

/-- Theorem stating that for a product with original price $100 and 
    two reductions of percentage m, the final price is 100(1-m)^2 -/
theorem final_price_after_two_reductions (m : ℝ) :
  final_price 100 m = 100 * (1 - m)^2 := by
  sorry

end NUMINAMATH_CALUDE_final_price_after_two_reductions_l4065_406531


namespace NUMINAMATH_CALUDE_pentagon_area_l4065_406522

/-- The area of a specific pentagon -/
theorem pentagon_area : 
  ∀ (pentagon_sides : List ℝ) 
    (trapezoid_bases : List ℝ) 
    (trapezoid_height : ℝ) 
    (triangle_base : ℝ) 
    (triangle_height : ℝ),
  pentagon_sides = [18, 25, 30, 28, 25] →
  trapezoid_bases = [25, 28] →
  trapezoid_height = 30 →
  triangle_base = 18 →
  triangle_height = 24 →
  (1/2 * (trapezoid_bases.sum) * trapezoid_height) + (1/2 * triangle_base * triangle_height) = 1011 := by
sorry


end NUMINAMATH_CALUDE_pentagon_area_l4065_406522


namespace NUMINAMATH_CALUDE_expression_evaluation_l4065_406596

theorem expression_evaluation :
  let x : ℤ := -2
  let y : ℤ := 1
  5 * x^2 - 2*x*y + 3*(x*y + 2) - 1 = 23 :=
by sorry

end NUMINAMATH_CALUDE_expression_evaluation_l4065_406596


namespace NUMINAMATH_CALUDE_lune_area_minus_square_l4065_406539

theorem lune_area_minus_square (r1 r2 s : ℝ) : r1 = 2 → r2 = 1 → s = 1 →
  (π * r1^2 / 2 - π * r2^2 / 2) - s^2 = 3 * π / 2 - 1 := by
  sorry

end NUMINAMATH_CALUDE_lune_area_minus_square_l4065_406539


namespace NUMINAMATH_CALUDE_lara_chips_count_l4065_406528

theorem lara_chips_count :
  ∀ (total_chips : ℕ),
  (total_chips / 6 : ℚ) + 34 + 16 = total_chips →
  total_chips = 60 := by
sorry

end NUMINAMATH_CALUDE_lara_chips_count_l4065_406528


namespace NUMINAMATH_CALUDE_power_product_squared_l4065_406505

theorem power_product_squared (a b : ℝ) : (2 * a * b^2)^2 = 4 * a^2 * b^4 := by sorry

end NUMINAMATH_CALUDE_power_product_squared_l4065_406505


namespace NUMINAMATH_CALUDE_system_consistent_iff_k_equals_four_l4065_406598

theorem system_consistent_iff_k_equals_four 
  (x y u : ℝ) (k : ℝ) : 
  (x + y = 1 ∧ k * x + y = 2 ∧ x + k * u = 3) ↔ k = 4 := by
  sorry

end NUMINAMATH_CALUDE_system_consistent_iff_k_equals_four_l4065_406598


namespace NUMINAMATH_CALUDE_garlic_cloves_left_is_600_l4065_406581

/-- The number of garlic cloves Maria has left after using some for a feast -/
def garlic_cloves_left : ℕ :=
  let kitchen_initial := 750
  let pantry_initial := 450
  let basement_initial := 300
  let kitchen_used := 500
  let pantry_used := 230
  let basement_used := 170
  (kitchen_initial - kitchen_used) + (pantry_initial - pantry_used) + (basement_initial - basement_used)

theorem garlic_cloves_left_is_600 : garlic_cloves_left = 600 := by
  sorry

end NUMINAMATH_CALUDE_garlic_cloves_left_is_600_l4065_406581


namespace NUMINAMATH_CALUDE_tonya_large_lemonade_sales_l4065_406523

/-- Represents the sales data for Tonya's lemonade stand. -/
structure LemonadeSales where
  small_price : ℕ
  medium_price : ℕ
  large_price : ℕ
  total_revenue : ℕ
  small_revenue : ℕ
  medium_revenue : ℕ

/-- Calculates the number of large lemonade cups sold. -/
def large_cups_sold (sales : LemonadeSales) : ℕ :=
  (sales.total_revenue - sales.small_revenue - sales.medium_revenue) / sales.large_price

/-- Theorem stating that Tonya sold 5 cups of large lemonade. -/
theorem tonya_large_lemonade_sales :
  let sales : LemonadeSales := {
    small_price := 1,
    medium_price := 2,
    large_price := 3,
    total_revenue := 50,
    small_revenue := 11,
    medium_revenue := 24
  }
  large_cups_sold sales = 5 := by sorry

end NUMINAMATH_CALUDE_tonya_large_lemonade_sales_l4065_406523


namespace NUMINAMATH_CALUDE_smallest_n_for_integer_root_l4065_406519

theorem smallest_n_for_integer_root : ∃ (n : ℕ), n > 0 ∧ 
  (∀ (m : ℕ), m > 0 → m < n → ¬ ∃ (k : ℕ), k^2 = 2019 - m) ∧
  ∃ (k : ℕ), k^2 = 2019 - n :=
by sorry

end NUMINAMATH_CALUDE_smallest_n_for_integer_root_l4065_406519


namespace NUMINAMATH_CALUDE_largest_four_digit_divisible_by_98_l4065_406530

theorem largest_four_digit_divisible_by_98 : 
  ∀ n : ℕ, n ≤ 9998 ∧ n ≥ 1000 ∧ n % 98 = 0 → n ≤ 9998 :=
by
  sorry

end NUMINAMATH_CALUDE_largest_four_digit_divisible_by_98_l4065_406530


namespace NUMINAMATH_CALUDE_pool_water_rates_l4065_406508

/-- Represents the water delivery rates for two pools -/
structure PoolRates :=
  (first : ℝ)
  (second : ℝ)

/-- Proves that the water delivery rates for two pools satisfy the given conditions -/
theorem pool_water_rates :
  ∃ (rates : PoolRates),
    rates.first = 90 ∧
    rates.second = 60 ∧
    rates.first = rates.second + 30 ∧
    ∃ (t : ℝ),
      (rates.first * t + rates.second * t = 2 * rates.first * t) ∧
      (rates.first * (t + 8/3) = rates.first * t) ∧
      (rates.second * (t + 10/3) = rates.second * t) :=
by sorry

end NUMINAMATH_CALUDE_pool_water_rates_l4065_406508


namespace NUMINAMATH_CALUDE_perfect_square_properties_l4065_406584

theorem perfect_square_properties (a : ℕ) :
  (∀ n : ℕ, n > 0 → a ∈ ({1, 2, 4} : Set ℕ) → ¬∃ m : ℕ, n * (a + n) = m^2) ∧
  ((∃ k : ℕ, k ≥ 3 ∧ a = 2^k) → ∃ n : ℕ, n > 0 ∧ ∃ m : ℕ, n * (a + n) = m^2) ∧
  (a ∉ ({1, 2, 4} : Set ℕ) → ∃ n : ℕ, n > 0 ∧ ∃ m : ℕ, n * (a + n) = m^2) :=
by sorry

end NUMINAMATH_CALUDE_perfect_square_properties_l4065_406584


namespace NUMINAMATH_CALUDE_ratio_problem_l4065_406526

theorem ratio_problem (a b c : ℝ) (h1 : b / a = 4) (h2 : c / b = 5) : 
  (a + b) / (b + c) = 5 / 24 := by
sorry

end NUMINAMATH_CALUDE_ratio_problem_l4065_406526


namespace NUMINAMATH_CALUDE_max_value_of_a_l4065_406510

theorem max_value_of_a (a b c : ℝ) 
  (sum_zero : a + b + c = 0) 
  (sum_squares_one : a^2 + b^2 + c^2 = 1) : 
  a ≤ Real.sqrt 6 / 3 ∧ ∃ (a₀ : ℝ), a₀ = Real.sqrt 6 / 3 ∧ 
  ∃ (b₀ c₀ : ℝ), a₀ + b₀ + c₀ = 0 ∧ a₀^2 + b₀^2 + c₀^2 = 1 :=
sorry

end NUMINAMATH_CALUDE_max_value_of_a_l4065_406510


namespace NUMINAMATH_CALUDE_max_monthly_profit_l4065_406509

/-- Represents the monthly profit function for a product with given cost and pricing conditions. -/
def monthly_profit (x : ℕ) : ℚ :=
  -10 * x^2 + 110 * x + 2100

/-- Theorem stating the maximum monthly profit and the optimal selling prices. -/
theorem max_monthly_profit :
  (∀ x : ℕ, 0 < x → x ≤ 15 → monthly_profit x ≤ 2400) ∧
  monthly_profit 5 = 2400 ∧
  monthly_profit 6 = 2400 :=
sorry

end NUMINAMATH_CALUDE_max_monthly_profit_l4065_406509


namespace NUMINAMATH_CALUDE_intersection_and_non_membership_l4065_406565

-- Define the lines
def line1 (x y : ℚ) : Prop := y = -3 * x
def line2 (x y : ℚ) : Prop := y + 3 = 9 * x
def line3 (x y : ℚ) : Prop := y = 2 * x - 1

-- Define the intersection point
def intersection_point : ℚ × ℚ := (1/4, -3/4)

-- Theorem statement
theorem intersection_and_non_membership :
  let (x, y) := intersection_point
  (line1 x y ∧ line2 x y) ∧ ¬(line3 x y) := by sorry

end NUMINAMATH_CALUDE_intersection_and_non_membership_l4065_406565


namespace NUMINAMATH_CALUDE_det2_trig_equality_l4065_406595

-- Define the second-order determinant
def det2 (a b d c : ℝ) : ℝ := a * c - b * d

-- State the theorem
theorem det2_trig_equality : det2 (Real.sin (50 * π / 180)) (Real.cos (40 * π / 180)) (-Real.sqrt 3 * Real.tan (10 * π / 180)) 1 = 1 := by sorry

end NUMINAMATH_CALUDE_det2_trig_equality_l4065_406595


namespace NUMINAMATH_CALUDE_martha_butterflies_l4065_406511

theorem martha_butterflies (total : ℕ) (blue : ℕ) (yellow : ℕ) (black : ℕ) : 
  total = 19 → 
  blue = 2 * yellow → 
  blue = 6 → 
  black = total - (blue + yellow) → 
  black = 10 := by
sorry

end NUMINAMATH_CALUDE_martha_butterflies_l4065_406511


namespace NUMINAMATH_CALUDE_determine_y_from_one_point_determine_y_from_k_one_additional_data_necessary_and_sufficient_l4065_406570

/-- A structure representing a proportional relationship between x and y --/
structure ProportionalRelationship where
  k : ℝ  -- Constant of proportionality
  proportional : ∀ (x y : ℝ), y = k * x

/-- Given a proportional relationship and one point, we can determine y for any x --/
theorem determine_y_from_one_point 
  (rel : ProportionalRelationship) (x₀ y₀ : ℝ) (h : y₀ = rel.k * x₀) :
  ∀ (x : ℝ), ∃! (y : ℝ), y = rel.k * x :=
sorry

/-- Given a proportional relationship and k, we can determine y for any x --/
theorem determine_y_from_k (rel : ProportionalRelationship) :
  ∀ (x : ℝ), ∃! (y : ℝ), y = rel.k * x :=
sorry

/-- One additional piece of data (either k or a point) is necessary and sufficient --/
theorem one_additional_data_necessary_and_sufficient :
  ∀ (x y : ℝ → ℝ), (∃ (k : ℝ), ∀ (t : ℝ), y t = k * x t) →
  ((∃ (k : ℝ), ∀ (t : ℝ), y t = k * x t) ∨ 
   (∃ (x₀ y₀ : ℝ), y x₀ = y₀ ∧ ∀ (t : ℝ), y t = (y₀ / x₀) * x t)) ∧
  (∀ (t : ℝ), ∃! (yt : ℝ), y t = yt) :=
sorry

end NUMINAMATH_CALUDE_determine_y_from_one_point_determine_y_from_k_one_additional_data_necessary_and_sufficient_l4065_406570


namespace NUMINAMATH_CALUDE_tank_plastering_cost_l4065_406532

/-- Calculates the cost of plastering a tank's walls and bottom -/
def plasteringCost (length width depth : ℝ) (costPerSqMeter : ℝ) : ℝ :=
  let bottomArea := length * width
  let longWallsArea := 2 * (length * depth)
  let shortWallsArea := 2 * (width * depth)
  let totalArea := bottomArea + longWallsArea + shortWallsArea
  totalArea * costPerSqMeter

/-- Theorem stating the cost of plastering the given tank -/
theorem tank_plastering_cost :
  plasteringCost 25 12 6 0.25 = 186 := by
  sorry

end NUMINAMATH_CALUDE_tank_plastering_cost_l4065_406532


namespace NUMINAMATH_CALUDE_two_truth_tellers_l4065_406516

/-- Represents the four Knaves -/
inductive Knave : Type
  | Hearts
  | Clubs
  | Diamonds
  | Spades

/-- Represents whether a Knave is telling the truth or lying -/
def Truthfulness : Type := Knave → Bool

/-- A consistent arrangement of truthfulness satisfies the interdependence of Knaves' statements -/
def is_consistent (t : Truthfulness) : Prop :=
  t Knave.Hearts = (t Knave.Clubs = false ∧ t Knave.Diamonds = true ∧ t Knave.Spades = false)

/-- Counts the number of truth-telling Knaves -/
def count_truth_tellers (t : Truthfulness) : Nat :=
  (if t Knave.Hearts then 1 else 0) +
  (if t Knave.Clubs then 1 else 0) +
  (if t Knave.Diamonds then 1 else 0) +
  (if t Knave.Spades then 1 else 0)

/-- Main theorem: Any consistent arrangement has exactly two truth-tellers -/
theorem two_truth_tellers (t : Truthfulness) (h : is_consistent t) :
  count_truth_tellers t = 2 := by
  sorry

end NUMINAMATH_CALUDE_two_truth_tellers_l4065_406516


namespace NUMINAMATH_CALUDE_quadratic_equation_real_roots_l4065_406592

theorem quadratic_equation_real_roots (m : ℝ) : 
  ∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ x₁^2 + (3*m - 1)*x₁ + (2*m^2 - m) = 0 ∧
                x₂^2 + (3*m - 1)*x₂ + (2*m^2 - m) = 0 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_equation_real_roots_l4065_406592


namespace NUMINAMATH_CALUDE_total_pictures_l4065_406542

/-- 
Given that:
- Randy drew 5 pictures
- Peter drew 3 more pictures than Randy
- Quincy drew 20 more pictures than Peter

Prove that the total number of pictures drawn by Randy, Peter, and Quincy is 41.
-/
theorem total_pictures (randy peter quincy : ℕ) : 
  randy = 5 → 
  peter = randy + 3 → 
  quincy = peter + 20 → 
  randy + peter + quincy = 41 := by
  sorry

end NUMINAMATH_CALUDE_total_pictures_l4065_406542


namespace NUMINAMATH_CALUDE_largest_five_digit_palindrome_divisible_by_127_l4065_406557

/-- A function that checks if a number is a 5-digit palindrome -/
def is_five_digit_palindrome (n : ℕ) : Prop :=
  10000 ≤ n ∧ n ≤ 99999 ∧ 
  (n / 10000 = n % 10) ∧
  ((n / 1000) % 10 = (n / 10) % 10)

/-- The largest 5-digit palindrome divisible by 127 -/
def largest_palindrome : ℕ := 99399

theorem largest_five_digit_palindrome_divisible_by_127 :
  is_five_digit_palindrome largest_palindrome ∧
  largest_palindrome % 127 = 0 ∧
  ∀ n : ℕ, is_five_digit_palindrome n ∧ n % 127 = 0 → n ≤ largest_palindrome :=
by sorry

end NUMINAMATH_CALUDE_largest_five_digit_palindrome_divisible_by_127_l4065_406557


namespace NUMINAMATH_CALUDE_cubic_sum_l4065_406575

theorem cubic_sum (a b c : ℝ) 
  (h1 : a + b + c = 5) 
  (h2 : a * b + a * c + b * c = 7) 
  (h3 : a * b * c = -18) : 
  a^3 + b^3 + c^3 = 29 := by
sorry


end NUMINAMATH_CALUDE_cubic_sum_l4065_406575


namespace NUMINAMATH_CALUDE_cat_whiskers_correct_l4065_406587

structure Cat where
  name : String
  whiskers : ℕ

def princess_puff : Cat := { name := "Princess Puff", whiskers := 14 }

def catman_do : Cat := { 
  name := "Catman Do", 
  whiskers := 2 * princess_puff.whiskers - 6 
}

def sir_whiskerson : Cat := { 
  name := "Sir Whiskerson", 
  whiskers := princess_puff.whiskers + catman_do.whiskers + 8 
}

def lady_flufflepuff : Cat := { 
  name := "Lady Flufflepuff", 
  whiskers := sir_whiskerson.whiskers / 2 + 4 
}

def mr_mittens : Cat := { 
  name := "Mr. Mittens", 
  whiskers := Int.natAbs (catman_do.whiskers - lady_flufflepuff.whiskers)
}

theorem cat_whiskers_correct : 
  princess_puff.whiskers = 14 ∧ 
  catman_do.whiskers = 22 ∧ 
  sir_whiskerson.whiskers = 44 ∧ 
  lady_flufflepuff.whiskers = 26 ∧ 
  mr_mittens.whiskers = 4 := by
  sorry

end NUMINAMATH_CALUDE_cat_whiskers_correct_l4065_406587


namespace NUMINAMATH_CALUDE_managers_salary_l4065_406547

def employee_count : ℕ := 24
def initial_average_salary : ℕ := 1500
def average_increase : ℕ := 400

theorem managers_salary (total_salary : ℕ) (managers_salary : ℕ) :
  total_salary = employee_count * initial_average_salary ∧
  (total_salary + managers_salary) / (employee_count + 1) = initial_average_salary + average_increase →
  managers_salary = 11500 := by
  sorry

end NUMINAMATH_CALUDE_managers_salary_l4065_406547


namespace NUMINAMATH_CALUDE_greatest_difference_l4065_406551

theorem greatest_difference (n m : ℕ+) (h : 1023 = 17 * n + m) : 
  ∃ (n' m' : ℕ+), 1023 = 17 * n' + m' ∧ ∀ (a b : ℕ+), 1023 = 17 * a + b → (n' : ℤ) - m' ≥ (a : ℤ) - b :=
by sorry

end NUMINAMATH_CALUDE_greatest_difference_l4065_406551


namespace NUMINAMATH_CALUDE_sum_of_digits_b_l4065_406545

/-- The number with 2^n digits '9' in base 10 -/
def a (n : ℕ) : ℕ := 10^(2^n) - 1

/-- The product of the first n+1 terms of a_k -/
def b : ℕ → ℕ
  | 0 => a 0
  | n+1 => (a (n+1)) * (b n)

/-- The sum of digits of a natural number -/
def sum_of_digits : ℕ → ℕ := sorry

theorem sum_of_digits_b (n : ℕ) : sum_of_digits (b n) = 9 * 2^n := by sorry

end NUMINAMATH_CALUDE_sum_of_digits_b_l4065_406545


namespace NUMINAMATH_CALUDE_pyramid_height_equals_cube_volume_l4065_406529

/-- The height of a square-based pyramid with the same volume as a cube -/
theorem pyramid_height_equals_cube_volume (cube_edge : ℝ) (pyramid_base : ℝ) (h : ℝ) : 
  cube_edge = 5 →
  pyramid_base = 10 →
  (1 / 3) * pyramid_base^2 * h = cube_edge^3 →
  h = 3.75 := by
sorry

end NUMINAMATH_CALUDE_pyramid_height_equals_cube_volume_l4065_406529


namespace NUMINAMATH_CALUDE_max_digit_sum_two_digit_primes_l4065_406553

def is_two_digit (n : ℕ) : Prop := 10 ≤ n ∧ n ≤ 99

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ m : ℕ, m > 1 → m < n → ¬(n % m = 0)

def digit_sum (n : ℕ) : ℕ := (n / 10) + (n % 10)

theorem max_digit_sum_two_digit_primes :
  ∃ (p : ℕ), is_two_digit p ∧ is_prime p ∧
    digit_sum p = 17 ∧
    ∀ (q : ℕ), is_two_digit q → is_prime q → digit_sum q ≤ 17 :=
sorry

end NUMINAMATH_CALUDE_max_digit_sum_two_digit_primes_l4065_406553


namespace NUMINAMATH_CALUDE_factor_sum_l4065_406549

theorem factor_sum (P Q : ℝ) : 
  (∃ b c : ℝ, (X^2 + 3*X + 4) * (X^2 + b*X + c) = X^4 + P*X^2 + Q) → 
  P + Q = 15 := by
sorry

end NUMINAMATH_CALUDE_factor_sum_l4065_406549


namespace NUMINAMATH_CALUDE_negation_equivalence_l4065_406582

-- Define the set of non-negative real numbers
def nonNegativeReals : Set ℝ := {x : ℝ | x ≥ 0}

-- Define the original proposition
def originalProposition : Prop :=
  ∀ x ∈ nonNegativeReals, Real.exp x ≥ 1

-- Define the negation of the proposition
def negationProposition : Prop :=
  ∃ x ∈ nonNegativeReals, Real.exp x < 1

-- Theorem stating that the negation is correct
theorem negation_equivalence :
  ¬originalProposition ↔ negationProposition :=
by sorry

end NUMINAMATH_CALUDE_negation_equivalence_l4065_406582


namespace NUMINAMATH_CALUDE_sum_of_three_consecutive_integers_l4065_406590

theorem sum_of_three_consecutive_integers :
  ∃ n : ℤ, n - 1 + n + (n + 1) = 21 ∧
  (n - 1 + n + (n + 1) = 17 ∨
   n - 1 + n + (n + 1) = 11 ∨
   n - 1 + n + (n + 1) = 25 ∨
   n - 1 + n + (n + 1) = 21 ∨
   n - 1 + n + (n + 1) = 8) :=
by sorry

end NUMINAMATH_CALUDE_sum_of_three_consecutive_integers_l4065_406590


namespace NUMINAMATH_CALUDE_gcf_275_180_l4065_406541

theorem gcf_275_180 : Nat.gcd 275 180 = 5 := by
  sorry

end NUMINAMATH_CALUDE_gcf_275_180_l4065_406541


namespace NUMINAMATH_CALUDE_brad_lemonade_profit_l4065_406556

/-- Calculates the net profit from a lemonade stand operation -/
def lemonade_stand_profit (
  glasses_per_gallon : ℕ)
  (cost_per_gallon : ℚ)
  (gallons_made : ℕ)
  (price_per_glass : ℚ)
  (glasses_drunk : ℕ)
  (glasses_unsold : ℕ) : ℚ :=
  let total_glasses := glasses_per_gallon * gallons_made
  let glasses_for_sale := total_glasses - glasses_drunk
  let glasses_sold := glasses_for_sale - glasses_unsold
  let revenue := glasses_sold * price_per_glass
  let cost := gallons_made * cost_per_gallon
  revenue - cost

/-- Theorem stating that Brad's net profit is $14.00 -/
theorem brad_lemonade_profit :
  lemonade_stand_profit 16 3.5 2 1 5 6 = 14 := by
  sorry

end NUMINAMATH_CALUDE_brad_lemonade_profit_l4065_406556


namespace NUMINAMATH_CALUDE_jack_driving_distance_l4065_406512

/-- Calculates the number of miles driven every four months given the total years of driving and total miles driven. -/
def miles_per_four_months (years : ℕ) (total_miles : ℕ) : ℕ :=
  total_miles / (years * 3)

/-- Theorem stating that driving for 9 years and covering 999,000 miles results in 37,000 miles every four months. -/
theorem jack_driving_distance :
  miles_per_four_months 9 999000 = 37000 := by
  sorry

end NUMINAMATH_CALUDE_jack_driving_distance_l4065_406512


namespace NUMINAMATH_CALUDE_almas_test_score_l4065_406543

/-- Given two people, Alma and Melina, where Melina's age is 60 and three times Alma's age,
    and the sum of their ages is twice Alma's test score, prove that Alma's test score is 40. -/
theorem almas_test_score (alma_age melina_age alma_score : ℕ) : 
  melina_age = 60 →
  melina_age = 3 * alma_age →
  alma_age + melina_age = 2 * alma_score →
  alma_score = 40 := by
  sorry

end NUMINAMATH_CALUDE_almas_test_score_l4065_406543


namespace NUMINAMATH_CALUDE_chord_length_l4065_406520

theorem chord_length (r d : ℝ) (hr : r = 5) (hd : d = 4) :
  let chord_length := 2 * Real.sqrt (r^2 - d^2)
  chord_length = 6 := by
  sorry

end NUMINAMATH_CALUDE_chord_length_l4065_406520


namespace NUMINAMATH_CALUDE_definite_integral_2x_minus_3x_squared_l4065_406503

theorem definite_integral_2x_minus_3x_squared : 
  ∫ x in (0 : ℝ)..2, (2 * x - 3 * x^2) = -4 := by sorry

end NUMINAMATH_CALUDE_definite_integral_2x_minus_3x_squared_l4065_406503


namespace NUMINAMATH_CALUDE_sum_of_coefficients_eq_64_l4065_406546

/-- The sum of the numerical coefficients in the complete expansion of (x^2 - 3xy + y^2)^6 -/
def sum_of_coefficients : ℕ :=
  (1 - 3)^6

theorem sum_of_coefficients_eq_64 : sum_of_coefficients = 64 := by
  sorry

#eval sum_of_coefficients

end NUMINAMATH_CALUDE_sum_of_coefficients_eq_64_l4065_406546


namespace NUMINAMATH_CALUDE_peaches_in_basket_l4065_406591

/-- Represents the number of peaches in a basket -/
structure Basket :=
  (red : ℕ)
  (green : ℕ)

/-- The total number of peaches in a basket is the sum of red and green peaches -/
def total_peaches (b : Basket) : ℕ := b.red + b.green

/-- Given a basket with 7 red peaches and 3 green peaches, prove that the total number of peaches is 10 -/
theorem peaches_in_basket :
  ∀ b : Basket, b.red = 7 ∧ b.green = 3 → total_peaches b = 10 :=
by
  sorry

#check peaches_in_basket

end NUMINAMATH_CALUDE_peaches_in_basket_l4065_406591


namespace NUMINAMATH_CALUDE_necessary_and_sufficient_condition_l4065_406576

theorem necessary_and_sufficient_condition (a : ℝ) :
  let f := fun x => x * (x - a) * (x - 2)
  let f' := fun x => 3 * x^2 - 2 * (a + 2) * x + 2 * a
  (0 < a ∧ a < 2) ↔ f' a < 0 := by
  sorry

end NUMINAMATH_CALUDE_necessary_and_sufficient_condition_l4065_406576


namespace NUMINAMATH_CALUDE_trig_expression_equals_32_l4065_406535

theorem trig_expression_equals_32 : 
  3 / (Real.sin (20 * π / 180))^2 - 1 / (Real.cos (20 * π / 180))^2 + 64 * (Real.sin (20 * π / 180))^2 = 32 := by
  sorry

end NUMINAMATH_CALUDE_trig_expression_equals_32_l4065_406535


namespace NUMINAMATH_CALUDE_solution_set_f_range_of_m_l4065_406578

-- Define the function f
def f (x : ℝ) : ℝ := |x - 1|

-- Part I
theorem solution_set_f (x : ℝ) : 
  (|f x - 3| ≤ 4) ↔ (-6 ≤ x ∧ x ≤ 8) := by sorry

-- Part II
theorem range_of_m (m : ℝ) : 
  (∀ x : ℝ, f x + f (x + 3) ≥ m^2 - 2*m) ↔ (-1 ≤ m ∧ m ≤ 3) := by sorry

end NUMINAMATH_CALUDE_solution_set_f_range_of_m_l4065_406578


namespace NUMINAMATH_CALUDE_equilateral_triangle_area_perimeter_ratio_l4065_406536

/-- The ratio of the area to the perimeter of an equilateral triangle with side length 10 -/
theorem equilateral_triangle_area_perimeter_ratio :
  let side_length : ℝ := 10
  let height : ℝ := side_length * (Real.sqrt 3 / 2)
  let area : ℝ := (1 / 2) * side_length * height
  let perimeter : ℝ := 3 * side_length
  (area / perimeter) = (5 * Real.sqrt 3) / 6 := by
sorry

end NUMINAMATH_CALUDE_equilateral_triangle_area_perimeter_ratio_l4065_406536


namespace NUMINAMATH_CALUDE_solve_equation_l4065_406548

theorem solve_equation (x : ℝ) (h : 61 + 5 * 12 / (x / 3) = 62) : x = 180 := by
  sorry

end NUMINAMATH_CALUDE_solve_equation_l4065_406548


namespace NUMINAMATH_CALUDE_complex_number_location_l4065_406567

theorem complex_number_location (z : ℂ) (h : (z + 3*Complex.I) * (3 + Complex.I) = 7 - Complex.I) :
  (z.re > 0) ∧ (z.im < 0) := by
  sorry

end NUMINAMATH_CALUDE_complex_number_location_l4065_406567


namespace NUMINAMATH_CALUDE_football_progress_l4065_406534

def yard_changes : List Int := [-5, 9, -12, 17, -15, 24, -7]

theorem football_progress : yard_changes.sum = 11 := by
  sorry

end NUMINAMATH_CALUDE_football_progress_l4065_406534


namespace NUMINAMATH_CALUDE_system_solution_l4065_406586

theorem system_solution (x y k : ℝ) : 
  (2 * x + y = 1) → 
  (x + 2 * y = k - 2) → 
  (x - y = 2) → 
  (k = 1) := by
sorry

end NUMINAMATH_CALUDE_system_solution_l4065_406586


namespace NUMINAMATH_CALUDE_odd_digits_345_base5_l4065_406562

/-- Counts the number of odd digits in a base-5 number -/
def countOddDigitsBase5 (n : ℕ) : ℕ := sorry

/-- Converts a base-10 number to base-5 -/
def toBase5 (n : ℕ) : ℕ := sorry

theorem odd_digits_345_base5 :
  countOddDigitsBase5 (toBase5 345) = 2 := by sorry

end NUMINAMATH_CALUDE_odd_digits_345_base5_l4065_406562


namespace NUMINAMATH_CALUDE_alexander_pencil_count_l4065_406560

/-- The number of pencils Alexander uses for all exhibitions -/
def total_pencils (initial_pictures : ℕ) (new_galleries : ℕ) (pictures_per_new_gallery : ℕ) 
  (pencils_per_picture : ℕ) (pencils_for_signing : ℕ) : ℕ :=
  let total_pictures := initial_pictures + new_galleries * pictures_per_new_gallery
  let pencils_for_drawing := total_pictures * pencils_per_picture
  let total_exhibitions := 1 + new_galleries
  let pencils_for_all_signings := total_exhibitions * pencils_for_signing
  pencils_for_drawing + pencils_for_all_signings

/-- Theorem stating that Alexander uses 88 pencils in total -/
theorem alexander_pencil_count : 
  total_pencils 9 5 2 4 2 = 88 := by
  sorry


end NUMINAMATH_CALUDE_alexander_pencil_count_l4065_406560


namespace NUMINAMATH_CALUDE_computer_price_problem_l4065_406558

theorem computer_price_problem (P : ℝ) : 
  1.30 * P = 351 ∧ 2 * P = 540 → P = 270 := by
  sorry

end NUMINAMATH_CALUDE_computer_price_problem_l4065_406558


namespace NUMINAMATH_CALUDE_matrix_commutation_l4065_406579

open Matrix

theorem matrix_commutation (A B : Matrix (Fin 2) (Fin 2) ℝ) 
  (h1 : A + B = A * B) 
  (h2 : A * B = ![![5, 1], ![-2, 2]]) : 
  B * A = A * B := by sorry

end NUMINAMATH_CALUDE_matrix_commutation_l4065_406579


namespace NUMINAMATH_CALUDE_volleyball_team_size_l4065_406583

/-- The number of people on each team in a volleyball game -/
def peoplePerTeam (managers : ℕ) (employees : ℕ) (teams : ℕ) : ℕ :=
  (managers + employees) / teams

/-- Theorem: In a volleyball game with 3 managers, 3 employees, and 3 teams, there are 2 people per team -/
theorem volleyball_team_size : peoplePerTeam 3 3 3 = 2 := by
  sorry

end NUMINAMATH_CALUDE_volleyball_team_size_l4065_406583


namespace NUMINAMATH_CALUDE_inequality_proof_l4065_406550

theorem inequality_proof (x y : ℝ) (h1 : x^2 ≥ y) (h2 : y^2 ≥ x) :
  x / (y^2 + 1) + y / (x^2 + 1) ≤ 1 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l4065_406550


namespace NUMINAMATH_CALUDE_notebook_length_l4065_406515

/-- Given a rectangular notebook with area 1.77 cm² and width 3 cm, prove its length is 0.59 cm -/
theorem notebook_length (area : ℝ) (width : ℝ) (length : ℝ) :
  area = 1.77 ∧ width = 3 ∧ area = length * width → length = 0.59 := by
  sorry

end NUMINAMATH_CALUDE_notebook_length_l4065_406515


namespace NUMINAMATH_CALUDE_cubic_equation_only_trivial_solution_l4065_406599

theorem cubic_equation_only_trivial_solution (x y z : ℤ) :
  x^3 - 2*y^3 - 4*z^3 = 0 → x = 0 ∧ y = 0 ∧ z = 0 := by
  sorry

end NUMINAMATH_CALUDE_cubic_equation_only_trivial_solution_l4065_406599


namespace NUMINAMATH_CALUDE_cost_to_fill_can_n_l4065_406580

/-- Represents a right circular cylinder -/
structure Cylinder where
  radius : ℝ
  height : ℝ

/-- The cost in dollars to fill a given volume of gasoline -/
def fillCost (volume : ℝ) : ℝ := sorry

theorem cost_to_fill_can_n (can_b can_n : Cylinder) (half_b_cost : ℝ) : 
  can_n.radius = 2 * can_b.radius →
  can_n.height = can_b.height / 2 →
  fillCost (π * can_b.radius^2 * can_b.height / 2) = 4 →
  fillCost (π * can_n.radius^2 * can_n.height) = 16 := by sorry

end NUMINAMATH_CALUDE_cost_to_fill_can_n_l4065_406580


namespace NUMINAMATH_CALUDE_factor_tree_proof_l4065_406569

theorem factor_tree_proof (A B C D E : ℕ) 
  (hB : B = 4 * D)
  (hC : C = 7 * E)
  (hA : A = B * C)
  (hD : D = 4 * 3)
  (hE : E = 7 * 3) :
  A = 7056 := by
  sorry

end NUMINAMATH_CALUDE_factor_tree_proof_l4065_406569


namespace NUMINAMATH_CALUDE_no_natural_solutions_for_equation_l4065_406544

theorem no_natural_solutions_for_equation : 
  ∀ (x y z : ℕ), x^x + 2*y^y ≠ z^z := by
  sorry

end NUMINAMATH_CALUDE_no_natural_solutions_for_equation_l4065_406544


namespace NUMINAMATH_CALUDE_A_B_symmetric_about_x_axis_l4065_406513

/-- Two points are symmetric about the x-axis if their x-coordinates are equal
    and their y-coordinates are negatives of each other -/
def symmetric_about_x_axis (A B : ℝ × ℝ) : Prop :=
  A.1 = B.1 ∧ A.2 = -B.2

/-- Point A in the coordinate plane -/
def A : ℝ × ℝ := (-1, 3)

/-- Point B in the coordinate plane -/
def B : ℝ × ℝ := (-1, -3)

/-- Theorem stating that points A and B are symmetric about the x-axis -/
theorem A_B_symmetric_about_x_axis : symmetric_about_x_axis A B := by
  sorry


end NUMINAMATH_CALUDE_A_B_symmetric_about_x_axis_l4065_406513


namespace NUMINAMATH_CALUDE_largest_number_l4065_406540

def a : ℝ := 8.12334
def b : ℝ := 8.123333333 -- Approximation of 8.123̅3
def c : ℝ := 8.123333333 -- Approximation of 8.12̅33
def d : ℝ := 8.123323323 -- Approximation of 8.1̅233
def e : ℝ := 8.123312331 -- Approximation of 8.̅1233

theorem largest_number : 
  (b = c) ∧ (b ≥ a) ∧ (b ≥ d) ∧ (b ≥ e) := by sorry

end NUMINAMATH_CALUDE_largest_number_l4065_406540


namespace NUMINAMATH_CALUDE_expected_yolks_in_carton_l4065_406555

/-- Represents a carton of eggs with various yolk counts -/
structure EggCarton where
  total_eggs : ℕ
  double_yolk_eggs : ℕ
  triple_yolk_eggs : ℕ
  extra_yolk_probability : ℝ

/-- Calculates the expected number of yolks in a carton of eggs -/
def expected_yolks (carton : EggCarton) : ℝ :=
  let single_yolk_eggs := carton.total_eggs - carton.double_yolk_eggs - carton.triple_yolk_eggs
  let base_yolks := single_yolk_eggs + 2 * carton.double_yolk_eggs + 3 * carton.triple_yolk_eggs
  let extra_yolks := carton.extra_yolk_probability * (carton.double_yolk_eggs + carton.triple_yolk_eggs)
  base_yolks + extra_yolks

/-- Theorem stating the expected number of yolks in the given carton -/
theorem expected_yolks_in_carton :
  let carton : EggCarton := {
    total_eggs := 15,
    double_yolk_eggs := 5,
    triple_yolk_eggs := 3,
    extra_yolk_probability := 0.1
  }
  expected_yolks carton = 26.8 := by sorry

end NUMINAMATH_CALUDE_expected_yolks_in_carton_l4065_406555


namespace NUMINAMATH_CALUDE_nate_running_distance_l4065_406506

/-- The total distance Nate ran in meters -/
def total_distance (field_length : ℝ) (additional_distance : ℝ) : ℝ :=
  4 * field_length + additional_distance

/-- Theorem stating the total distance Nate ran -/
theorem nate_running_distance :
  let field_length : ℝ := 168
  let additional_distance : ℝ := 500
  total_distance field_length additional_distance = 1172 := by
  sorry

end NUMINAMATH_CALUDE_nate_running_distance_l4065_406506


namespace NUMINAMATH_CALUDE_students_not_in_chorus_or_band_l4065_406593

theorem students_not_in_chorus_or_band 
  (total : ℕ) (chorus : ℕ) (band : ℕ) (both : ℕ) 
  (h1 : total = 50)
  (h2 : chorus = 18)
  (h3 : band = 26)
  (h4 : both = 2) :
  total - (chorus + band - both) = 8 := by
  sorry

end NUMINAMATH_CALUDE_students_not_in_chorus_or_band_l4065_406593


namespace NUMINAMATH_CALUDE_charlie_prob_different_colors_l4065_406502

/-- Represents the number of marbles of each color -/
def num_marbles : ℕ := 3

/-- Represents the total number of marbles -/
def total_marbles : ℕ := 3 * num_marbles

/-- Represents the number of marbles each person takes -/
def marbles_per_person : ℕ := 3

/-- Calculates the number of ways to choose k items from n items -/
def choose (n k : ℕ) : ℕ := (Nat.factorial n) / ((Nat.factorial k) * (Nat.factorial (n - k)))

/-- Calculates the total number of ways the marbles can be drawn -/
def total_ways : ℕ := 
  (choose total_marbles marbles_per_person) * 
  (choose (total_marbles - marbles_per_person) marbles_per_person) * 
  (choose marbles_per_person marbles_per_person)

/-- Calculates the number of favorable outcomes for Charlie -/
def favorable_outcomes : ℕ := 
  (choose num_marbles 2) * (choose num_marbles 2) * (choose num_marbles 2)

/-- The probability of Charlie drawing three different colored marbles -/
theorem charlie_prob_different_colors : 
  (favorable_outcomes : ℚ) / total_ways = 5 / 8 := by sorry

end NUMINAMATH_CALUDE_charlie_prob_different_colors_l4065_406502


namespace NUMINAMATH_CALUDE_sum_g_79_l4065_406564

-- Define the functions f and g
def f (x : ℝ) : ℝ := 4 * x^2 + 3 * x - 1
def g (y : ℝ) : ℝ := y^2 - 2 * y + 2

-- Define the equation f(x) = 79
def f_eq_79 (x : ℝ) : Prop := f x = 79

-- Theorem statement
theorem sum_g_79 (x₁ x₂ : ℝ) (h₁ : f_eq_79 x₁) (h₂ : f_eq_79 x₂) (h₃ : x₁ ≠ x₂) :
  ∃ (s : ℝ), s = g (f x₁) + g (f x₂) ∧ 
  (∀ (y : ℝ), g y = s ↔ y = 79) :=
sorry

end NUMINAMATH_CALUDE_sum_g_79_l4065_406564


namespace NUMINAMATH_CALUDE_farm_egg_yolks_l4065_406574

/-- Represents the number of yolks in an egg carton -/
def yolks_in_carton (total_eggs : ℕ) (double_yolk_eggs : ℕ) : ℕ :=
  2 * double_yolk_eggs + (total_eggs - double_yolk_eggs)

/-- Theorem: A carton of 12 eggs with 5 double-yolk eggs has 17 yolks in total -/
theorem farm_egg_yolks : yolks_in_carton 12 5 = 17 := by
  sorry

end NUMINAMATH_CALUDE_farm_egg_yolks_l4065_406574


namespace NUMINAMATH_CALUDE_largest_6k_plus_1_factor_of_11_factorial_l4065_406588

def factorial (n : ℕ) : ℕ := (List.range n).foldl (· * ·) 1

def is_factor (a b : ℕ) : Prop := b % a = 0

def is_of_form_6k_plus_1 (n : ℕ) : Prop := ∃ k : ℕ, n = 6 * k + 1

theorem largest_6k_plus_1_factor_of_11_factorial :
  ∀ n : ℕ, is_factor n (factorial 11) → is_of_form_6k_plus_1 n → n ≤ 385 :=
by sorry

end NUMINAMATH_CALUDE_largest_6k_plus_1_factor_of_11_factorial_l4065_406588


namespace NUMINAMATH_CALUDE_square_to_rectangle_perimeter_l4065_406572

theorem square_to_rectangle_perimeter (n : ℕ) (a : ℝ) : 
  a > 0 →
  n > 0 →
  ∃ k : ℕ, k > 0 ∧ k < n ∧
  (k : ℝ) * 6 * a = (n - 2 * k : ℝ) * 4 * a ∧
  4 * n * a - (4 * n * a - 40) = 40 →
  4 * n * a = 280 :=
by sorry

end NUMINAMATH_CALUDE_square_to_rectangle_perimeter_l4065_406572


namespace NUMINAMATH_CALUDE_jason_pears_l4065_406500

theorem jason_pears (total pears_keith pears_mike : ℕ) 
  (h_total : total = 105)
  (h_keith : pears_keith = 47)
  (h_mike : pears_mike = 12) :
  total - (pears_keith + pears_mike) = 46 := by
  sorry

end NUMINAMATH_CALUDE_jason_pears_l4065_406500


namespace NUMINAMATH_CALUDE_lindas_savings_l4065_406507

theorem lindas_savings (savings : ℝ) : 
  (2 / 3 : ℝ) * savings + 250 = savings → savings = 750 := by
  sorry

end NUMINAMATH_CALUDE_lindas_savings_l4065_406507


namespace NUMINAMATH_CALUDE_sufficient_not_necessary_l4065_406525

theorem sufficient_not_necessary :
  (∀ x : ℝ, x > 3 → x^2 - 5*x + 6 > 0) ∧
  (∃ x : ℝ, x^2 - 5*x + 6 > 0 ∧ ¬(x > 3)) := by
  sorry

end NUMINAMATH_CALUDE_sufficient_not_necessary_l4065_406525


namespace NUMINAMATH_CALUDE_jack_needs_five_rocks_l4065_406554

/-- The number of rocks needed to equalize weights on a see-saw -/
def rocks_needed (jack_weight anna_weight rock_weight : ℕ) : ℕ :=
  (jack_weight - anna_weight) / rock_weight

/-- Theorem: Jack needs 5 rocks to equalize weights with Anna -/
theorem jack_needs_five_rocks :
  rocks_needed 60 40 4 = 5 := by
  sorry

end NUMINAMATH_CALUDE_jack_needs_five_rocks_l4065_406554


namespace NUMINAMATH_CALUDE_tiffany_bags_next_day_l4065_406559

/-- The number of bags Tiffany had on Monday -/
def monday_bags : ℕ := 7

/-- The additional number of bags Tiffany found on the next day compared to Monday -/
def additional_bags : ℕ := 5

/-- The total number of bags Tiffany found on the next day -/
def next_day_bags : ℕ := monday_bags + additional_bags

theorem tiffany_bags_next_day : next_day_bags = 12 := by
  sorry

end NUMINAMATH_CALUDE_tiffany_bags_next_day_l4065_406559


namespace NUMINAMATH_CALUDE_no_solutions_absolute_value_equation_l4065_406573

theorem no_solutions_absolute_value_equation :
  ¬ ∃ x : ℝ, |x - 2| = |x - 1| + |x - 5| := by
sorry

end NUMINAMATH_CALUDE_no_solutions_absolute_value_equation_l4065_406573


namespace NUMINAMATH_CALUDE_return_time_is_11pm_l4065_406533

structure Journey where
  startTime : Nat
  totalDistance : Nat
  speedLevel : Nat
  speedUphill : Nat
  speedDownhill : Nat
  terrainDistribution : Nat

def calculateReturnTime (j : Journey) : Nat :=
  let oneWayTime := j.terrainDistribution / j.speedLevel +
                    j.terrainDistribution / j.speedUphill +
                    j.terrainDistribution / j.speedDownhill +
                    j.terrainDistribution / j.speedLevel
  let totalTime := 2 * oneWayTime
  j.startTime + totalTime

theorem return_time_is_11pm (j : Journey) 
  (h1 : j.startTime = 15) -- 3 pm in 24-hour format
  (h2 : j.totalDistance = 12)
  (h3 : j.speedLevel = 4)
  (h4 : j.speedUphill = 3)
  (h5 : j.speedDownhill = 6)
  (h6 : j.terrainDistribution = 4) -- Assumption of equal distribution
  : calculateReturnTime j = 23 := by -- 11 pm in 24-hour format
  sorry


end NUMINAMATH_CALUDE_return_time_is_11pm_l4065_406533


namespace NUMINAMATH_CALUDE_cube_volume_doubling_l4065_406589

theorem cube_volume_doubling (a : ℝ) (h : a > 0) :
  (2 * a) ^ 3 = 8 * a ^ 3 :=
by sorry

end NUMINAMATH_CALUDE_cube_volume_doubling_l4065_406589


namespace NUMINAMATH_CALUDE_quadratic_two_roots_l4065_406585

-- Define the quadratic equation
def quadratic (x k : ℝ) : ℝ := x^2 + 2*x + k

-- Define the condition for two distinct real roots
def has_two_distinct_real_roots (k : ℝ) : Prop :=
  ∃ x y : ℝ, x ≠ y ∧ quadratic x k = 0 ∧ quadratic y k = 0

-- State the theorem
theorem quadratic_two_roots :
  has_two_distinct_real_roots 0 ∧ 
  ∀ k : ℝ, has_two_distinct_real_roots k → k = 0 :=
sorry

end NUMINAMATH_CALUDE_quadratic_two_roots_l4065_406585


namespace NUMINAMATH_CALUDE_knife_percentage_after_trade_l4065_406521

/-- Represents a silverware set with knives, forks, and spoons -/
structure Silverware where
  knives : ℕ
  forks : ℕ
  spoons : ℕ

/-- Represents a trade of silverware -/
structure Trade where
  knivesReceived : ℕ
  spoonsGiven : ℕ

def initialSet : Silverware :=
  { knives := 6
  , forks := 12
  , spoons := 6 * 3 }

def trade : Trade :=
  { knivesReceived := 10
  , spoonsGiven := 6 }

def finalSet (initial : Silverware) (t : Trade) : Silverware :=
  { knives := initial.knives + t.knivesReceived
  , forks := initial.forks
  , spoons := initial.spoons - t.spoonsGiven }

def totalPieces (s : Silverware) : ℕ :=
  s.knives + s.forks + s.spoons

def knifePercentage (s : Silverware) : ℚ :=
  s.knives / totalPieces s

theorem knife_percentage_after_trade :
  knifePercentage (finalSet initialSet trade) = 2/5 := by
  sorry

end NUMINAMATH_CALUDE_knife_percentage_after_trade_l4065_406521


namespace NUMINAMATH_CALUDE_soda_consumption_theorem_l4065_406538

/-- The number of bottles of soda left after a given period -/
def bottles_left (bottles_per_pack : ℕ) (packs_bought : ℕ) (bottles_per_day : ℚ) (days : ℕ) : ℚ :=
  (bottles_per_pack * packs_bought : ℚ) - (bottles_per_day * days)

/-- Theorem stating that given the conditions, 4 bottles will be left after 4 weeks -/
theorem soda_consumption_theorem :
  bottles_left 6 3 (1/2) (4 * 7) = 4 := by
  sorry

end NUMINAMATH_CALUDE_soda_consumption_theorem_l4065_406538


namespace NUMINAMATH_CALUDE_triathlete_average_speed_l4065_406594

/-- The average speed of a triathlete for swimming and running events -/
theorem triathlete_average_speed 
  (swim_speed : ℝ) 
  (run_speed : ℝ) 
  (h1 : swim_speed = 1) 
  (h2 : run_speed = 6) : 
  (2 * swim_speed * run_speed) / (swim_speed + run_speed) = 12 / 7 := by
  sorry

end NUMINAMATH_CALUDE_triathlete_average_speed_l4065_406594


namespace NUMINAMATH_CALUDE_investment_growth_approx_l4065_406563

/-- Approximates the future value of an investment with compound interest -/
def future_value (principal : ℝ) (rate : ℝ) (time : ℕ) : ℝ :=
  principal * (1 + rate) ^ time

/-- Theorem: An investment of $1500 at 8% annual interest grows to approximately $13500 in 28 years -/
theorem investment_growth_approx :
  ∃ ε > 0, abs (future_value 1500 0.08 28 - 13500) < ε :=
by
  sorry

end NUMINAMATH_CALUDE_investment_growth_approx_l4065_406563

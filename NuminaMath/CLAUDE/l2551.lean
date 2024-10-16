import Mathlib

namespace NUMINAMATH_CALUDE_not_divisible_by_seven_l2551_255110

theorem not_divisible_by_seven (a b : ℕ) : 
  ¬(7 ∣ (a * b)) → ¬(7 ∣ a ∨ 7 ∣ b) := by
  sorry

end NUMINAMATH_CALUDE_not_divisible_by_seven_l2551_255110


namespace NUMINAMATH_CALUDE_dunbar_bouquets_l2551_255141

/-- The number of table decorations needed --/
def num_table_decorations : ℕ := 7

/-- The number of white roses used in each table decoration --/
def roses_per_table_decoration : ℕ := 12

/-- The number of white roses used in each bouquet --/
def roses_per_bouquet : ℕ := 5

/-- The total number of white roses needed for all bouquets and table decorations --/
def total_roses : ℕ := 109

/-- The number of bouquets Mrs. Dunbar needs to make --/
def num_bouquets : ℕ := (total_roses - num_table_decorations * roses_per_table_decoration) / roses_per_bouquet

theorem dunbar_bouquets : num_bouquets = 5 := by
  sorry

end NUMINAMATH_CALUDE_dunbar_bouquets_l2551_255141


namespace NUMINAMATH_CALUDE_additional_income_needed_l2551_255103

/-- Calculate the additional annual income needed to reach a target amount after expenses --/
theorem additional_income_needed
  (current_income : ℝ)
  (rent : ℝ)
  (groceries : ℝ)
  (gas : ℝ)
  (target_amount : ℝ)
  (h1 : current_income = 65000)
  (h2 : rent = 20000)
  (h3 : groceries = 5000)
  (h4 : gas = 8000)
  (h5 : target_amount = 42000) :
  current_income + 10000 - (rent + groceries + gas) ≥ target_amount ∧
  ∀ x : ℝ, x < 10000 → current_income + x - (rent + groceries + gas) < target_amount :=
by
  sorry

#check additional_income_needed

end NUMINAMATH_CALUDE_additional_income_needed_l2551_255103


namespace NUMINAMATH_CALUDE_exponent_division_l2551_255116

theorem exponent_division (a : ℝ) : a^4 / a^2 = a^2 := by
  sorry

end NUMINAMATH_CALUDE_exponent_division_l2551_255116


namespace NUMINAMATH_CALUDE_pen_gain_percentage_l2551_255193

/-- 
Given that the selling price of 5 pens equals the cost price of 10 pens, 
prove that the gain percentage is 100%.
-/
theorem pen_gain_percentage (cost selling : ℝ) 
  (h : 5 * selling = 10 * cost) : 
  (selling - cost) / cost * 100 = 100 := by
  sorry

end NUMINAMATH_CALUDE_pen_gain_percentage_l2551_255193


namespace NUMINAMATH_CALUDE_fraction_inequality_l2551_255130

theorem fraction_inequality (a b c : ℝ) (h1 : a > b) (h2 : b > 0) (h3 : c < 0) :
  a / (a - c) > b / (b - c) := by
  sorry

end NUMINAMATH_CALUDE_fraction_inequality_l2551_255130


namespace NUMINAMATH_CALUDE_max_value_quadratic_inequality_l2551_255139

theorem max_value_quadratic_inequality :
  let f : ℝ → ℝ := λ x => -2 * x^2 + 9 * x - 7
  ∃ (max_x : ℝ), max_x = 3.5 ∧
    (∀ x : ℝ, f x ≤ 0 → x ≤ max_x) ∧
    f max_x ≤ 0 :=
by sorry

end NUMINAMATH_CALUDE_max_value_quadratic_inequality_l2551_255139


namespace NUMINAMATH_CALUDE_red_apples_count_l2551_255149

theorem red_apples_count (red : ℕ) (green : ℕ) : 
  green = red + 12 →
  red + green = 44 →
  red = 16 := by
sorry

end NUMINAMATH_CALUDE_red_apples_count_l2551_255149


namespace NUMINAMATH_CALUDE_sum_of_digits_8_pow_1502_l2551_255127

/-- The sum of the tens digit and the units digit in the decimal representation of 8^1502 is 10 -/
theorem sum_of_digits_8_pow_1502 : ∃ n : ℕ, 8^1502 = 100 * n + 64 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_digits_8_pow_1502_l2551_255127


namespace NUMINAMATH_CALUDE_semester_days_l2551_255101

/-- Calculates the number of days given daily distance and total distance -/
def calculate_days (daily_distance : ℕ) (total_distance : ℕ) : ℕ :=
  total_distance / daily_distance

/-- Theorem stating that given the specific conditions, the number of days is 160 -/
theorem semester_days : calculate_days 10 1600 = 160 := by
  sorry

end NUMINAMATH_CALUDE_semester_days_l2551_255101


namespace NUMINAMATH_CALUDE_max_product_sum_300_l2551_255115

theorem max_product_sum_300 : 
  (∀ x y : ℤ, x + y = 300 → x * y ≤ 22500) ∧ 
  (∃ x y : ℤ, x + y = 300 ∧ x * y = 22500) := by
  sorry

end NUMINAMATH_CALUDE_max_product_sum_300_l2551_255115


namespace NUMINAMATH_CALUDE_parabola_focus_distance_l2551_255198

/-- Theorem: For a parabola y^2 = 2px (p > 0) with vertex at origin, passing through (x₀, 2),
    if the distance from A to focus is 3 times the distance from origin to focus, then p = √2 -/
theorem parabola_focus_distance (p : ℝ) (x₀ : ℝ) (h_p_pos : p > 0) :
  (2 : ℝ)^2 = 2 * p * x₀ →  -- parabola passes through (x₀, 2)
  x₀ + p / 2 = 3 * (p / 2) →  -- |AF| = 3|OF|
  p = Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_parabola_focus_distance_l2551_255198


namespace NUMINAMATH_CALUDE_unique_polynomial_satisfying_conditions_l2551_255164

/-- A polynomial function of degree at most 3 -/
def Polynomial3 := ℝ → ℝ

/-- The conditions that g must satisfy -/
def SatisfiesConditions (g : Polynomial3) : Prop :=
  (∀ x, g (x^2) = (g x)^2) ∧
  (∀ x, g (x^2) = g (g x)) ∧
  (g 1 = 1)

/-- The theorem stating that there exists exactly one polynomial of degree at most 3 satisfying the conditions -/
theorem unique_polynomial_satisfying_conditions :
  ∃! g : Polynomial3, SatisfiesConditions g :=
sorry

end NUMINAMATH_CALUDE_unique_polynomial_satisfying_conditions_l2551_255164


namespace NUMINAMATH_CALUDE_remainder_of_1394_divided_by_2535_l2551_255168

theorem remainder_of_1394_divided_by_2535 : Int.mod 1394 2535 = 1394 := by
  sorry

end NUMINAMATH_CALUDE_remainder_of_1394_divided_by_2535_l2551_255168


namespace NUMINAMATH_CALUDE_suraj_innings_l2551_255181

/-- 
Proves that the number of innings Suraj played before the last one is 16,
given the conditions of the problem.
-/
theorem suraj_innings : 
  ∀ (n : ℕ) (A : ℚ),
  (A + 4 = 28) →                             -- New average after increase
  (n * A + 92 = (n + 1) * 28) →              -- Total runs equation
  (n = 16) := by
sorry

end NUMINAMATH_CALUDE_suraj_innings_l2551_255181


namespace NUMINAMATH_CALUDE_min_second_longest_side_unit_area_triangle_l2551_255135

theorem min_second_longest_side_unit_area_triangle (a b c : ℝ) (h_area : (1/2) * a * b * Real.sin γ = 1) (h_order : a ≤ b ∧ b ≤ c) (γ : ℝ) :
  b ≥ Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_min_second_longest_side_unit_area_triangle_l2551_255135


namespace NUMINAMATH_CALUDE_unique_solution_xy_l2551_255154

theorem unique_solution_xy (x y : ℝ) 
  (h1 : x^2 + y^2 = 2)
  (h2 : x^2 / (2 - y) + y^2 / (2 - x) = 2) :
  x = 1 ∧ y = 1 :=
by sorry

end NUMINAMATH_CALUDE_unique_solution_xy_l2551_255154


namespace NUMINAMATH_CALUDE_difference_largest_negative_smallest_positive_not_two_l2551_255175

theorem difference_largest_negative_smallest_positive_not_two : ¬(∃ n m : ℤ, 
  (∀ k : ℤ, k < 0 → k ≤ n) ∧ 
  (∀ k : ℤ, k > 0 → m ≤ k) ∧ 
  n - m = 2) :=
sorry

end NUMINAMATH_CALUDE_difference_largest_negative_smallest_positive_not_two_l2551_255175


namespace NUMINAMATH_CALUDE_positive_real_inequality_l2551_255132

theorem positive_real_inequality (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  (a - b) * (a - c) / (2 * a^2 + (b + c)^2) +
  (b - c) * (b - a) / (2 * b^2 + (c + a)^2) +
  (c - a) * (c - b) / (2 * c^2 + (a + b)^2) ≥ 0 := by
  sorry

end NUMINAMATH_CALUDE_positive_real_inequality_l2551_255132


namespace NUMINAMATH_CALUDE_i_power_sum_l2551_255167

-- Define the imaginary unit i
def i : ℂ := Complex.I

-- Define the property that i^2 = -1
axiom i_squared : i^2 = -1

-- Define the property that powers of i repeat every 4 powers
axiom i_power_cycle (n : ℤ) : i^n = i^(n % 4)

-- State the theorem
theorem i_power_sum : i^17 + i^2023 = 0 := by
  sorry

end NUMINAMATH_CALUDE_i_power_sum_l2551_255167


namespace NUMINAMATH_CALUDE_figure_areas_l2551_255150

theorem figure_areas (total_squares : ℕ) (black_area : ℝ) 
  (h1 : total_squares = 8)
  (h2 : black_area = 7.5) : 
  ∃ (square_side : ℝ) (white_area dark_gray_area light_gray_area shaded_area : ℝ),
    square_side > 0 ∧
    white_area = 1.5 ∧
    dark_gray_area = 6 ∧
    light_gray_area = 5.25 ∧
    shaded_area = 3.75 ∧
    black_area = 2.5 * square_side^2 ∧
    white_area = 0.5 * square_side^2 ∧
    dark_gray_area = 2 * square_side^2 ∧
    light_gray_area = 1.75 * square_side^2 ∧
    shaded_area = 1.25 * square_side^2 ∧
    (white_area + dark_gray_area + light_gray_area + shaded_area + black_area) = 
      (total_squares : ℝ) * square_side^2 := by
  sorry

#check figure_areas

end NUMINAMATH_CALUDE_figure_areas_l2551_255150


namespace NUMINAMATH_CALUDE_smallest_angle_solution_l2551_255156

theorem smallest_angle_solution (y : Real) : 
  (∀ θ : Real, θ > 0 ∧ θ < y → 10 * Real.sin θ * Real.cos θ ^ 3 - 10 * Real.sin θ ^ 3 * Real.cos θ ≠ Real.sqrt 2) ∧
  (10 * Real.sin y * Real.cos y ^ 3 - 10 * Real.sin y ^ 3 * Real.cos y = Real.sqrt 2) →
  y = 11.25 * π / 180 := by
  sorry

end NUMINAMATH_CALUDE_smallest_angle_solution_l2551_255156


namespace NUMINAMATH_CALUDE_range_of_function_range_tight_l2551_255194

theorem range_of_function (x : ℝ) :
  ∃ (y : ℝ), y = |2 * Real.sin x + 3 * Real.cos x + 4| ∧
  4 - Real.sqrt 13 ≤ y ∧ y ≤ 4 + Real.sqrt 13 :=
by sorry

theorem range_tight :
  ∃ (x₁ x₂ : ℝ), 
    |2 * Real.sin x₁ + 3 * Real.cos x₁ + 4| = 4 - Real.sqrt 13 ∧
    |2 * Real.sin x₂ + 3 * Real.cos x₂ + 4| = 4 + Real.sqrt 13 :=
by sorry

end NUMINAMATH_CALUDE_range_of_function_range_tight_l2551_255194


namespace NUMINAMATH_CALUDE_emily_dimes_l2551_255180

/-- Represents the number of coins of each type -/
structure CoinCount where
  pennies : ℕ
  nickels : ℕ
  dimes : ℕ

/-- The problem constraints -/
def validCoinCount (c : CoinCount) : Prop :=
  c.pennies + c.nickels + c.dimes = 50 ∧
  c.pennies + 5 * c.nickels + 10 * c.dimes = 200

/-- The main theorem -/
theorem emily_dimes : ∃ (c : CoinCount), validCoinCount c ∧ c.dimes = 10 :=
  sorry

end NUMINAMATH_CALUDE_emily_dimes_l2551_255180


namespace NUMINAMATH_CALUDE_angle_measure_proof_l2551_255147

theorem angle_measure_proof (AOB BOC : Real) : 
  AOB + BOC = 180 →  -- adjacent supplementary angles
  AOB = BOC + 18 →   -- AOB is 18° larger than BOC
  AOB = 99 :=        -- prove that AOB is 99°
by sorry

end NUMINAMATH_CALUDE_angle_measure_proof_l2551_255147


namespace NUMINAMATH_CALUDE_brick_width_calculation_l2551_255153

/-- Calculates the width of a brick given the wall dimensions, brick dimensions, and number of bricks --/
theorem brick_width_calculation (wall_length wall_height wall_thickness : ℝ)
                                (brick_length brick_height : ℝ)
                                (num_bricks : ℕ) :
  wall_length = 800 →
  wall_height = 600 →
  wall_thickness = 22.5 →
  brick_length = 125 →
  brick_height = 6 →
  num_bricks = 1280 →
  ∃ (brick_width : ℝ),
    brick_width = 11.25 ∧
    wall_length * wall_height * wall_thickness =
    num_bricks * brick_length * brick_width * brick_height :=
by
  sorry

#check brick_width_calculation

end NUMINAMATH_CALUDE_brick_width_calculation_l2551_255153


namespace NUMINAMATH_CALUDE_hexagon_three_circles_area_l2551_255159

/-- The area of the region inside a regular hexagon but outside three inscribed circles --/
theorem hexagon_three_circles_area (s : ℝ) (h : s = 4) : 
  let hexagon_area := 3 * Real.sqrt 3 / 2 * s^2
  let circle_radius := s / 2
  let circle_area := π * circle_radius^2
  let total_circle_area := 3 * circle_area
  hexagon_area - total_circle_area = 24 * Real.sqrt 3 - 12 * π :=
by sorry

end NUMINAMATH_CALUDE_hexagon_three_circles_area_l2551_255159


namespace NUMINAMATH_CALUDE_leftover_eggs_l2551_255191

/-- Given that there are 119 eggs to be packaged into cartons of 12 eggs each,
    prove that the number of eggs left over is 11. -/
theorem leftover_eggs : Int.mod 119 12 = 11 := by
  sorry

end NUMINAMATH_CALUDE_leftover_eggs_l2551_255191


namespace NUMINAMATH_CALUDE_card_sum_difference_l2551_255162

theorem card_sum_difference (n : ℕ) (a : ℕ → ℝ) 
  (h_n : n > 4)
  (h_a : ∀ m ∈ Finset.range (2*n + 5), ⌊a m⌋ = m) :
  ∃ (i j k l : ℕ), i ∈ Finset.range (2*n + 5) ∧ 
                   j ∈ Finset.range (2*n + 5) ∧ 
                   k ∈ Finset.range (2*n + 5) ∧ 
                   l ∈ Finset.range (2*n + 5) ∧
                   i ≠ j ∧ i ≠ k ∧ i ≠ l ∧ j ≠ k ∧ j ≠ l ∧ k ≠ l ∧
                   |a i + a j - a k - a l| < 1 / (n - Real.sqrt (n / 2)) :=
sorry

end NUMINAMATH_CALUDE_card_sum_difference_l2551_255162


namespace NUMINAMATH_CALUDE_biathlon_average_speed_l2551_255185

def cycling_speed : ℝ := 18
def running_speed : ℝ := 8

theorem biathlon_average_speed :
  let harmonic_mean := 2 / (1 / cycling_speed + 1 / running_speed)
  harmonic_mean = 144 / 13 := by
  sorry

end NUMINAMATH_CALUDE_biathlon_average_speed_l2551_255185


namespace NUMINAMATH_CALUDE_pizza_payment_difference_l2551_255107

/-- Represents the cost structure and consumption of a pizza --/
structure PizzaOrder where
  total_slices : Nat
  plain_cost : Int
  cheese_slices : Nat
  veggie_slices : Nat
  topping_cost : Int
  jerry_plain_slices : Nat

/-- Calculates the difference in payment between Jerry and Tom --/
def payment_difference (order : PizzaOrder) : Int :=
  let total_cost := order.plain_cost + 2 * order.topping_cost
  let slice_cost := total_cost / order.total_slices
  let jerry_slices := order.cheese_slices + order.veggie_slices + order.jerry_plain_slices
  let tom_slices := order.total_slices - jerry_slices
  slice_cost * (jerry_slices - tom_slices)

/-- Theorem stating the difference in payment between Jerry and Tom --/
theorem pizza_payment_difference :
  ∃ (order : PizzaOrder),
    order.total_slices = 12 ∧
    order.plain_cost = 12 ∧
    order.cheese_slices = 4 ∧
    order.veggie_slices = 4 ∧
    order.topping_cost = 3 ∧
    order.jerry_plain_slices = 2 ∧
    payment_difference order = 12 := by
  sorry

end NUMINAMATH_CALUDE_pizza_payment_difference_l2551_255107


namespace NUMINAMATH_CALUDE_lines_planes_perpendicular_l2551_255148

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the relations between lines and planes
variable (parallel : Line → Line → Prop)
variable (perpendicular : Line → Plane → Prop)
variable (contains : Plane → Line → Prop)
variable (plane_perpendicular : Plane → Plane → Prop)

-- State the theorem
theorem lines_planes_perpendicular 
  (m n : Line) (α β : Plane) :
  parallel m n →
  contains α m →
  perpendicular n β →
  plane_perpendicular α β :=
sorry

end NUMINAMATH_CALUDE_lines_planes_perpendicular_l2551_255148


namespace NUMINAMATH_CALUDE_inequality_proof_l2551_255144

theorem inequality_proof (k l m n : ℕ) 
  (h1 : k < l) (h2 : l < m) (h3 : m < n) (h4 : l * m = k * n) : 
  ((n - k) / 2 : ℚ)^2 ≥ k + 2 := by sorry

end NUMINAMATH_CALUDE_inequality_proof_l2551_255144


namespace NUMINAMATH_CALUDE_lindas_savings_l2551_255136

theorem lindas_savings (savings : ℕ) : 
  (3 : ℚ) / 4 * savings + 250 = savings → savings = 1000 := by
  sorry

end NUMINAMATH_CALUDE_lindas_savings_l2551_255136


namespace NUMINAMATH_CALUDE_shaded_area_is_eleven_l2551_255179

/-- Given a grid with rectangles of dimensions 2x3, 3x4, and 4x5, and two unshaded right-angled triangles
    with dimensions (base 12, height 4) and (base 3, height 2), the shaded area is 11. -/
theorem shaded_area_is_eleven :
  let grid_area := 2 * 3 + 3 * 4 + 4 * 5
  let triangle1_area := (12 * 4) / 2
  let triangle2_area := (3 * 2) / 2
  let shaded_area := grid_area - triangle1_area - triangle2_area
  shaded_area = 11 := by
sorry


end NUMINAMATH_CALUDE_shaded_area_is_eleven_l2551_255179


namespace NUMINAMATH_CALUDE_max_value_expression_l2551_255192

theorem max_value_expression (x y z : ℝ) (h₁ : 0 ≤ x) (h₂ : 0 ≤ y) (h₃ : 0 ≤ z) (h₄ : x^2 + y^2 + z^2 = 1) :
  3 * x * y * Real.sqrt 3 + 9 * y * z ≤ Real.sqrt 255 :=
by sorry

end NUMINAMATH_CALUDE_max_value_expression_l2551_255192


namespace NUMINAMATH_CALUDE_complex_product_real_l2551_255129

theorem complex_product_real (a : ℝ) : 
  let z₁ : ℂ := 1 + a * Complex.I
  let z₂ : ℂ := 3 + 2 * Complex.I
  (z₁ * z₂).im = 0 → a = -2/3 := by
sorry

end NUMINAMATH_CALUDE_complex_product_real_l2551_255129


namespace NUMINAMATH_CALUDE_second_month_sale_l2551_255177

/-- Proves that the sale in the second month is 10500 given the conditions of the problem -/
theorem second_month_sale (sales : Fin 6 → ℕ)
  (h1 : sales 0 = 2500)
  (h3 : sales 2 = 9855)
  (h4 : sales 3 = 7230)
  (h5 : sales 4 = 7000)
  (h6 : sales 5 = 11915)
  (avg : (sales 0 + sales 1 + sales 2 + sales 3 + sales 4 + sales 5) / 6 = 7500) :
  sales 1 = 10500 := by
  sorry

#check second_month_sale

end NUMINAMATH_CALUDE_second_month_sale_l2551_255177


namespace NUMINAMATH_CALUDE_existence_of_solution_l2551_255100

/-- Floor function -/
noncomputable def floor (x : ℝ) : ℤ :=
  Int.floor x

/-- Condition: For any positive integers k₁ and k₂, ⌊k₁α⌋ ≠ ⌊k₂β⌋ -/
def condition (α β : ℝ) : Prop :=
  ∀ (k₁ k₂ : ℕ), k₁ > 0 ∧ k₂ > 0 → floor (k₁ * α) ≠ floor (k₂ * β)

/-- Theorem: If the condition holds for positive real numbers α and β,
    then there exist positive integers m₁ and m₂ such that (m₁/α) + (m₂/β) = 1 -/
theorem existence_of_solution (α β : ℝ) (hα : α > 0) (hβ : β > 0) 
    (h : condition α β) : 
    ∃ (m₁ m₂ : ℕ), m₁ > 0 ∧ m₂ > 0 ∧ (m₁ : ℝ) / α + (m₂ : ℝ) / β = 1 :=
  sorry

end NUMINAMATH_CALUDE_existence_of_solution_l2551_255100


namespace NUMINAMATH_CALUDE_quotient_in_third_quadrant_l2551_255158

/-- Given complex numbers z₁ and z₂ where z₁ = 1 - 2i and the points corresponding to z₁ and z₂ 
    are symmetric about the imaginary axis, the point corresponding to z₂/z₁ lies in the third 
    quadrant of the complex plane. -/
theorem quotient_in_third_quadrant (z₁ z₂ : ℂ) 
    (h₁ : z₁ = 1 - 2*I) 
    (h₂ : z₂.re = -z₁.re ∧ z₂.im = z₁.im) : 
    (z₂ / z₁).re < 0 ∧ (z₂ / z₁).im < 0 := by
  sorry

end NUMINAMATH_CALUDE_quotient_in_third_quadrant_l2551_255158


namespace NUMINAMATH_CALUDE_triangle_median_inequality_l2551_255143

/-- Given a triangle with side lengths a, b, c, medians m_a, m_b, m_c, 
    and circumscribed circle diameter D, the following inequality holds. -/
theorem triangle_median_inequality 
  (a b c m_a m_b m_c D : ℝ) 
  (h_pos_a : 0 < a) (h_pos_b : 0 < b) (h_pos_c : 0 < c)
  (h_pos_m_a : 0 < m_a) (h_pos_m_b : 0 < m_b) (h_pos_m_c : 0 < m_c)
  (h_pos_D : 0 < D)
  (h_triangle : a + b > c ∧ b + c > a ∧ c + a > b)
  (h_median_a : 4 * m_a^2 = 2 * b^2 + 2 * c^2 - a^2)
  (h_median_b : 4 * m_b^2 = 2 * c^2 + 2 * a^2 - b^2)
  (h_median_c : 4 * m_c^2 = 2 * a^2 + 2 * b^2 - c^2)
  (h_circumradius : D = 2 * (a * b * c) / (4 * area))
  (h_area : area = Real.sqrt ((a + b + c) * (b + c - a) * (c + a - b) * (a + b - c)) / 4) :
  (a^2 + b^2) / m_c + (b^2 + c^2) / m_a + (c^2 + a^2) / m_b ≤ 6 * D := by
  sorry

end NUMINAMATH_CALUDE_triangle_median_inequality_l2551_255143


namespace NUMINAMATH_CALUDE_weight_11_25m_l2551_255178

/-- Represents the weight of a uniform rod given its length -/
def rod_weight (length : ℝ) : ℝ := sorry

/-- The rod is uniform, meaning its weight is proportional to its length -/
axiom rod_uniform (l₁ l₂ : ℝ) : l₁ * rod_weight l₂ = l₂ * rod_weight l₁

/-- The weight of 6 meters of the rod is 22.8 kg -/
axiom weight_6m : rod_weight 6 = 22.8

/-- Theorem: If 6 m of a uniform rod weighs 22.8 kg, then 11.25 m weighs 42.75 kg -/
theorem weight_11_25m : rod_weight 11.25 = 42.75 := by sorry

end NUMINAMATH_CALUDE_weight_11_25m_l2551_255178


namespace NUMINAMATH_CALUDE_system_solution_l2551_255146

theorem system_solution (x y z : ℝ) : 
  (Real.sqrt (2 * x^2 + 2) = y + 1 ∧
   Real.sqrt (2 * y^2 + 2) = z + 1 ∧
   Real.sqrt (2 * z^2 + 2) = x + 1) →
  (x = 1 ∧ y = 1 ∧ z = 1) :=
by sorry

end NUMINAMATH_CALUDE_system_solution_l2551_255146


namespace NUMINAMATH_CALUDE_prob_at_most_one_red_l2551_255131

/-- The probability of drawing at most 1 red ball from a bag of 8 balls (3 red, 2 white, 3 black) when randomly selecting 3 balls. -/
theorem prob_at_most_one_red (total : ℕ) (red : ℕ) (white : ℕ) (black : ℕ) 
  (h_total : total = 8)
  (h_red : red = 3)
  (h_white : white = 2)
  (h_black : black = 3)
  (h_sum : red + white + black = total)
  (draw : ℕ)
  (h_draw : draw = 3) :
  (Nat.choose (total - red) draw + Nat.choose red 1 * Nat.choose (total - red) (draw - 1)) / Nat.choose total draw = 5 / 7 := by
  sorry

end NUMINAMATH_CALUDE_prob_at_most_one_red_l2551_255131


namespace NUMINAMATH_CALUDE_no_convex_polygon_partition_into_non_convex_quadrilaterals_l2551_255106

/-- A polygon is a closed planar figure bounded by straight line segments. -/
structure Polygon where
  vertices : Set (ℝ × ℝ)
  is_closed : Bool
  is_planar : Bool

/-- A polygon is convex if all its interior angles are less than or equal to 180 degrees. -/
def is_convex (p : Polygon) : Prop :=
  sorry

/-- A quadrilateral is a polygon with exactly four sides. -/
def is_quadrilateral (p : Polygon) : Prop :=
  sorry

/-- A quadrilateral is non-convex if at least one of its interior angles is greater than 180 degrees. -/
def is_non_convex_quadrilateral (q : Polygon) : Prop :=
  is_quadrilateral q ∧ ¬(is_convex q)

/-- A partition of a polygon is a set of smaller polygons that completely cover the original polygon without overlapping. -/
def is_partition (p : Polygon) (parts : Set Polygon) : Prop :=
  sorry

/-- The main theorem: It is impossible to partition a convex polygon into non-convex quadrilaterals. -/
theorem no_convex_polygon_partition_into_non_convex_quadrilaterals :
  ∀ (p : Polygon) (parts : Set Polygon),
    is_convex p →
    is_partition p parts →
    (∀ q ∈ parts, is_non_convex_quadrilateral q) →
    False :=
  sorry

end NUMINAMATH_CALUDE_no_convex_polygon_partition_into_non_convex_quadrilaterals_l2551_255106


namespace NUMINAMATH_CALUDE_diamond_olivine_difference_l2551_255117

theorem diamond_olivine_difference (agate olivine diamond : ℕ) : 
  agate = 30 →
  olivine = agate + 5 →
  diamond > olivine →
  agate + olivine + diamond = 111 →
  diamond - olivine = 11 :=
by sorry

end NUMINAMATH_CALUDE_diamond_olivine_difference_l2551_255117


namespace NUMINAMATH_CALUDE_expected_digits_is_31_20_l2551_255176

/-- A fair 20-sided die with numbers 1 through 20 -/
def icosahedralDie : Finset ℕ := Finset.range 20

/-- The number of digits for a given number on the die -/
def numDigits (n : ℕ) : ℕ :=
  if n < 10 then 1 else 2

/-- The expected number of digits when rolling the die -/
def expectedDigits : ℚ :=
  (icosahedralDie.sum (λ i => numDigits (i + 1))) / icosahedralDie.card

theorem expected_digits_is_31_20 : expectedDigits = 31 / 20 := by
  sorry

end NUMINAMATH_CALUDE_expected_digits_is_31_20_l2551_255176


namespace NUMINAMATH_CALUDE_distance_between_towns_l2551_255160

/-- The distance between two towns given two trains traveling towards each other -/
theorem distance_between_towns
  (total_distance : ℝ)
  (speed_difference : ℝ)
  (time : ℝ)
  (remaining_distance : ℝ)
  (h1 : total_distance = 300)
  (h2 : speed_difference = 10)
  (h3 : time = 2)
  (h4 : remaining_distance = 40) :
  total_distance = 300 := by sorry

end NUMINAMATH_CALUDE_distance_between_towns_l2551_255160


namespace NUMINAMATH_CALUDE_cylinder_surface_area_l2551_255161

/-- Given a rectangle with length 4π cm and width 2 cm that is rolled into a cylinder
    using the longer side as the circumference of the base, prove that the total
    surface area of the resulting cylinder is 16π cm². -/
theorem cylinder_surface_area (π : ℝ) (h : π > 0) :
  let rectangle_length : ℝ := 4 * π
  let rectangle_width : ℝ := 2
  let base_circumference : ℝ := rectangle_length
  let base_radius : ℝ := base_circumference / (2 * π)
  let cylinder_height : ℝ := rectangle_width
  let total_surface_area : ℝ := 2 * π * base_radius^2 + 2 * π * base_radius * cylinder_height
  total_surface_area = 16 * π :=
by sorry

end NUMINAMATH_CALUDE_cylinder_surface_area_l2551_255161


namespace NUMINAMATH_CALUDE_dad_vacuum_time_l2551_255157

theorem dad_vacuum_time (downstairs upstairs : ℕ) : 
  upstairs = 2 * downstairs + 5 →
  downstairs + upstairs = 38 →
  upstairs = 27 := by
sorry

end NUMINAMATH_CALUDE_dad_vacuum_time_l2551_255157


namespace NUMINAMATH_CALUDE_salary_restoration_l2551_255122

theorem salary_restoration (original_salary : ℝ) (original_salary_positive : original_salary > 0) :
  let reduced_salary := 0.8 * original_salary
  let restored_salary := reduced_salary * 1.25
  restored_salary = original_salary := by
sorry

end NUMINAMATH_CALUDE_salary_restoration_l2551_255122


namespace NUMINAMATH_CALUDE_geometric_sequence_fourth_term_l2551_255126

/-- A geometric sequence with first term 1024 and sixth term 125 has its fourth term equal to 2000 -/
theorem geometric_sequence_fourth_term : ∀ (a : ℕ → ℝ), 
  (∃ r : ℝ, ∀ n : ℕ, a n = 1024 * r ^ (n - 1)) →  -- Geometric sequence definition
  a 1 = 1024 →                                   -- First term condition
  a 6 = 125 →                                    -- Sixth term condition
  a 4 = 2000 :=                                  -- Fourth term (to prove)
by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_fourth_term_l2551_255126


namespace NUMINAMATH_CALUDE_complex_equation_sum_squares_l2551_255133

theorem complex_equation_sum_squares (a b : ℝ) :
  (a + Complex.I) / Complex.I = b + Complex.I * Real.sqrt 2 →
  a^2 + b^2 = 3 := by
sorry

end NUMINAMATH_CALUDE_complex_equation_sum_squares_l2551_255133


namespace NUMINAMATH_CALUDE_third_month_sales_l2551_255183

def sales_1 : ℕ := 5400
def sales_2 : ℕ := 9000
def sales_4 : ℕ := 7200
def sales_5 : ℕ := 4500
def sales_6 : ℕ := 1200
def average_sale : ℕ := 5600
def num_months : ℕ := 6

theorem third_month_sales :
  ∃ (sales_3 : ℕ),
    sales_3 = num_months * average_sale - (sales_1 + sales_2 + sales_4 + sales_5 + sales_6) ∧
    sales_3 = 6300 := by
  sorry

end NUMINAMATH_CALUDE_third_month_sales_l2551_255183


namespace NUMINAMATH_CALUDE_like_terms_exponent_sum_l2551_255195

theorem like_terms_exponent_sum (a b : ℝ) (x y : ℤ) : 
  (∃ k : ℝ, k ≠ 0 ∧ -3 * a^(x + 2*y) * b^9 = k * (2 * a^3 * b^(2*x + y))) → 
  x + y = 4 := by sorry

end NUMINAMATH_CALUDE_like_terms_exponent_sum_l2551_255195


namespace NUMINAMATH_CALUDE_warm_up_puzzle_time_l2551_255165

/-- Represents the time taken for the warm-up puzzle in minutes -/
def warm_up_time : ℝ := 10

/-- Represents the total number of puzzles solved -/
def total_puzzles : ℕ := 3

/-- Represents the total time spent solving all puzzles in minutes -/
def total_time : ℝ := 70

/-- Represents the time multiplier for the longer puzzles compared to the warm-up puzzle -/
def longer_puzzle_multiplier : ℝ := 3

/-- Represents the number of longer puzzles solved -/
def longer_puzzles : ℕ := 2

theorem warm_up_puzzle_time :
  warm_up_time * (1 + longer_puzzle_multiplier * longer_puzzles) = total_time :=
by sorry

end NUMINAMATH_CALUDE_warm_up_puzzle_time_l2551_255165


namespace NUMINAMATH_CALUDE_fib_units_digit_periodic_fib_15_value_units_digit_of_fib_fib_15_l2551_255104

def fib : ℕ → ℕ
  | 0 => 0
  | 1 => 1
  | (n + 2) => fib (n + 1) + fib n

theorem fib_units_digit_periodic (n : ℕ) : fib n % 10 = fib (n % 60) % 10 := by sorry

theorem fib_15_value : fib 15 = 610 := by sorry

theorem units_digit_of_fib_fib_15 : fib (fib 15) % 10 = 5 := by sorry

end NUMINAMATH_CALUDE_fib_units_digit_periodic_fib_15_value_units_digit_of_fib_fib_15_l2551_255104


namespace NUMINAMATH_CALUDE_new_students_average_age_l2551_255188

/-- Calculates the average age of new students joining a class --/
theorem new_students_average_age
  (original_average : ℝ)
  (original_strength : ℕ)
  (new_students : ℕ)
  (average_decrease : ℝ)
  (h1 : original_average = 40)
  (h2 : original_strength = 12)
  (h3 : new_students = 12)
  (h4 : average_decrease = 4) :
  let new_average := original_average - average_decrease
  let total_new_strength := original_strength + new_students
  let total_age_after := (original_strength + new_students) * new_average
  let total_age_before := original_strength * original_average
  let total_age_new_students := total_age_after - total_age_before
  (total_age_new_students / new_students : ℝ) = 32 := by
  sorry

end NUMINAMATH_CALUDE_new_students_average_age_l2551_255188


namespace NUMINAMATH_CALUDE_yellow_marbles_count_l2551_255125

/-- The number of yellow marbles Mary has -/
def mary_marbles : ℕ := 9

/-- The number of yellow marbles Joan has -/
def joan_marbles : ℕ := 3

/-- The total number of yellow marbles Mary and Joan have together -/
def total_marbles : ℕ := mary_marbles + joan_marbles

theorem yellow_marbles_count : total_marbles = 12 := by
  sorry

end NUMINAMATH_CALUDE_yellow_marbles_count_l2551_255125


namespace NUMINAMATH_CALUDE_equal_tabletops_and_legs_l2551_255123

/-- Represents the amount of wood used for tabletops -/
def wood_for_tabletops : ℝ := 3

/-- Represents the amount of wood used for legs -/
def wood_for_legs : ℝ := 5 - wood_for_tabletops

/-- Represents the number of tabletops that can be made from 1 cubic meter of wood -/
def tabletops_per_cubic_meter : ℝ := 50

/-- Represents the number of legs that can be made from 1 cubic meter of wood -/
def legs_per_cubic_meter : ℝ := 300

/-- Represents the number of legs per table -/
def legs_per_table : ℝ := 4

theorem equal_tabletops_and_legs :
  wood_for_tabletops * tabletops_per_cubic_meter = 
  wood_for_legs * legs_per_cubic_meter / legs_per_table := by
  sorry

end NUMINAMATH_CALUDE_equal_tabletops_and_legs_l2551_255123


namespace NUMINAMATH_CALUDE_group_size_proof_l2551_255173

/-- Proves that the number of members in a group is 54, given the conditions of the problem -/
theorem group_size_proof (n : ℕ) : 
  (n : ℚ) * n = 2916 → n = 54 := by
  sorry

end NUMINAMATH_CALUDE_group_size_proof_l2551_255173


namespace NUMINAMATH_CALUDE_min_bailing_rate_is_8_l2551_255145

/-- Represents the fishing scenario with Steve and LeRoy -/
structure FishingScenario where
  distance_to_shore : ℝ
  water_intake_rate : ℝ
  max_water_capacity : ℝ
  rowing_speed : ℝ

/-- Calculates the minimum bailing rate required to reach the shore without sinking -/
def min_bailing_rate (scenario : FishingScenario) : ℝ :=
  -- The actual calculation is not implemented here
  sorry

/-- Theorem stating that the minimum bailing rate for the given scenario is 8 gallons per minute -/
theorem min_bailing_rate_is_8 (scenario : FishingScenario) 
  (h1 : scenario.distance_to_shore = 1)
  (h2 : scenario.water_intake_rate = 10)
  (h3 : scenario.max_water_capacity = 30)
  (h4 : scenario.rowing_speed = 4) :
  min_bailing_rate scenario = 8 := by
  sorry

end NUMINAMATH_CALUDE_min_bailing_rate_is_8_l2551_255145


namespace NUMINAMATH_CALUDE_min_vertices_is_six_l2551_255186

/-- A graph where each vertex knows exactly three others -/
def KnowledgeGraph (V : Type*) := V → Finset V

/-- Predicate to check if a vertex has exactly 3 neighbors -/
def has_three_neighbors (G : KnowledgeGraph V) (v : V) : Prop :=
  (G v).card = 3

/-- Predicate to check if among any three vertices, two are not connected -/
def has_non_connected_pair (G : KnowledgeGraph V) : Prop :=
  ∀ (a b c : V), a ≠ b ∧ b ≠ c ∧ a ≠ c →
    ¬(a ∈ G b ∧ b ∈ G c ∧ c ∈ G a)

/-- The main theorem stating the minimum number of vertices is 6 -/
theorem min_vertices_is_six (V : Type*) [Fintype V] :
  (∃ (G : KnowledgeGraph V), (∀ v, has_three_neighbors G v) ∧ has_non_connected_pair G) →
  Fintype.card V ≥ 6 :=
sorry

end NUMINAMATH_CALUDE_min_vertices_is_six_l2551_255186


namespace NUMINAMATH_CALUDE_rattlesnake_count_l2551_255124

theorem rattlesnake_count (total : ℕ) (pythons boa_constrictors rattlesnakes vipers : ℕ) :
  total = 350 ∧
  total = pythons + boa_constrictors + rattlesnakes + vipers ∧
  pythons = 2 * boa_constrictors ∧
  vipers = rattlesnakes / 2 ∧
  boa_constrictors = 60 ∧
  pythons + vipers = (40 * total) / 100 →
  rattlesnakes = 40 := by
sorry

end NUMINAMATH_CALUDE_rattlesnake_count_l2551_255124


namespace NUMINAMATH_CALUDE_bread_slice_cost_l2551_255182

/-- Calculates the cost per slice of bread in cents -/
def cost_per_slice (num_loaves : ℕ) (slices_per_loaf : ℕ) (amount_paid : ℕ) (change : ℕ) : ℕ :=
  let total_cost := amount_paid - change
  let total_slices := num_loaves * slices_per_loaf
  (total_cost * 100) / total_slices

/-- Proves that the cost per slice is 40 cents given the problem conditions -/
theorem bread_slice_cost :
  cost_per_slice 3 20 40 16 = 40 := by
  sorry

#eval cost_per_slice 3 20 40 16

end NUMINAMATH_CALUDE_bread_slice_cost_l2551_255182


namespace NUMINAMATH_CALUDE_sin_alpha_value_l2551_255137

theorem sin_alpha_value (α : Real) :
  let P : Real × Real := (-2 * Real.sin (60 * π / 180), 2 * Real.cos (30 * π / 180))
  (∃ k : Real, k > 0 ∧ P = (k * Real.cos α, k * Real.sin α)) →
  Real.sin α = Real.sqrt 2 / 2 := by
sorry

end NUMINAMATH_CALUDE_sin_alpha_value_l2551_255137


namespace NUMINAMATH_CALUDE_power_of_power_l2551_255184

theorem power_of_power : (2^3)^3 = 512 := by
  sorry

end NUMINAMATH_CALUDE_power_of_power_l2551_255184


namespace NUMINAMATH_CALUDE_problem_solution_l2551_255112

theorem problem_solution (t : ℝ) (x y : ℝ) 
    (h1 : x = 3 - 2*t) 
    (h2 : y = 3*t + 6) 
    (h3 : x = 0) : 
  y = 21/2 := by
sorry

end NUMINAMATH_CALUDE_problem_solution_l2551_255112


namespace NUMINAMATH_CALUDE_isosceles_triangle_leg_length_l2551_255134

theorem isosceles_triangle_leg_length 
  (base : ℝ) 
  (leg : ℝ) 
  (h1 : base = 8) 
  (h2 : leg^2 - 9*leg + 20 = 0) 
  (h3 : leg > base/2) : 
  leg = 5 := by
sorry

end NUMINAMATH_CALUDE_isosceles_triangle_leg_length_l2551_255134


namespace NUMINAMATH_CALUDE_negation_of_exists_lt_one_squared_leq_one_l2551_255108

theorem negation_of_exists_lt_one_squared_leq_one :
  (¬ ∃ x : ℝ, x < 1 ∧ x^2 ≤ 1) ↔ (∀ x : ℝ, x < 1 → x^2 > 1) := by sorry

end NUMINAMATH_CALUDE_negation_of_exists_lt_one_squared_leq_one_l2551_255108


namespace NUMINAMATH_CALUDE_cosine_sum_squared_l2551_255128

theorem cosine_sum_squared : 
  (Real.cos (42 * π / 180) + Real.cos (102 * π / 180) + 
   Real.cos (114 * π / 180) + Real.cos (174 * π / 180))^2 = 3/4 := by
  sorry

end NUMINAMATH_CALUDE_cosine_sum_squared_l2551_255128


namespace NUMINAMATH_CALUDE_basketball_play_time_l2551_255151

/-- Calculates the time spent playing basketball given the total play time and time spent playing football. -/
theorem basketball_play_time 
  (total_time : Real) 
  (football_time : Nat) 
  (h1 : total_time = 1.5) 
  (h2 : football_time = 60) : 
  (total_time * 60 - football_time : Real) = 30 := by
  sorry

end NUMINAMATH_CALUDE_basketball_play_time_l2551_255151


namespace NUMINAMATH_CALUDE_turn_on_all_in_four_moves_l2551_255102

/-- Represents a light bulb on a 2D grid -/
structure Bulb where
  x : ℕ
  y : ℕ
  is_on : Bool

/-- Represents a line on a 2D plane -/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Represents the state of the grid -/
def GridState := List Bulb

/-- Checks if a bulb is on the specified side of a line -/
def is_on_side (b : Bulb) (l : Line) (positive_side : Bool) : Bool :=
  sorry

/-- Applies a move to the grid state -/
def apply_move (state : GridState) (l : Line) (positive_side : Bool) : GridState :=
  sorry

/-- Checks if all bulbs are on -/
def all_on (state : GridState) : Bool :=
  sorry

/-- Theorem: It's possible to turn on all bulbs in exactly four moves -/
theorem turn_on_all_in_four_moves :
  ∃ (moves : List (Line × Bool)),
    moves.length = 4 ∧
    let initial_state : GridState := [
      {x := 0, y := 0, is_on := false},
      {x := 0, y := 1, is_on := false},
      {x := 1, y := 0, is_on := false},
      {x := 1, y := 1, is_on := false}
    ]
    let final_state := moves.foldl (λ state move => apply_move state move.1 move.2) initial_state
    all_on final_state :=
  sorry

end NUMINAMATH_CALUDE_turn_on_all_in_four_moves_l2551_255102


namespace NUMINAMATH_CALUDE_point_b_coordinates_l2551_255138

/-- Given a circle with center (0,0) and radius 2, points A(2,2) and B(a,b),
    if for any point P on the circle, |PA|/|PB| = √2, then B = (1,1) -/
theorem point_b_coordinates (a b : ℝ) : 
  (∀ x y : ℝ, x^2 + y^2 = 4 → 
    ((x - 2)^2 + (y - 2)^2) / ((x - a)^2 + (y - b)^2) = 2) → 
  a = 1 ∧ b = 1 := by sorry

end NUMINAMATH_CALUDE_point_b_coordinates_l2551_255138


namespace NUMINAMATH_CALUDE_arithmetic_sequence_property_l2551_255114

/-- An arithmetic sequence with the given properties -/
structure ArithmeticSequence where
  a : ℕ → ℝ
  d : ℝ
  d_nonzero : d ≠ 0
  sum_condition : a 2 + a 4 = 10
  geometric_condition : (a 2) ^ 2 = a 1 * a 5
  arithmetic_property : ∀ n : ℕ, a (n + 1) = a n + d

/-- The theorem statement -/
theorem arithmetic_sequence_property (seq : ArithmeticSequence) :
  seq.a 1 = 1 ∧ ∀ n : ℕ, seq.a n = 2 * n - 1 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_property_l2551_255114


namespace NUMINAMATH_CALUDE_trader_profit_l2551_255142

theorem trader_profit (C : ℝ) (C_pos : C > 0) : 
  let markup := 0.12
  let discount := 0.09821428571428571
  let marked_price := C * (1 + markup)
  let final_price := marked_price * (1 - discount)
  (final_price - C) / C = 0.01 := by
sorry

end NUMINAMATH_CALUDE_trader_profit_l2551_255142


namespace NUMINAMATH_CALUDE_cube_edge_length_l2551_255199

/-- A prism made up of six squares -/
structure Cube where
  edge_length : ℝ
  edge_sum : ℝ

/-- The sum of the lengths of all edges is 72 cm -/
def total_edge_length (c : Cube) : Prop :=
  c.edge_sum = 72

/-- Theorem: If the sum of the lengths of all edges is 72 cm, 
    then the length of one edge is 6 cm -/
theorem cube_edge_length (c : Cube) 
    (h : total_edge_length c) : c.edge_length = 6 := by
  sorry

end NUMINAMATH_CALUDE_cube_edge_length_l2551_255199


namespace NUMINAMATH_CALUDE_student_addition_mistake_l2551_255121

theorem student_addition_mistake (a b : ℤ) :
  (a + 10 * b = 7182) ∧ (a + b = 3132) → (a = 2682 ∧ b = 450) := by
  sorry

end NUMINAMATH_CALUDE_student_addition_mistake_l2551_255121


namespace NUMINAMATH_CALUDE_triangle_3_4_5_l2551_255197

/-- A function that checks if three numbers can form a triangle -/
def can_form_triangle (a b c : ℝ) : Prop :=
  a + b > c ∧ b + c > a ∧ c + a > b

/-- Theorem stating that the line segments 3, 4, and 5 can form a triangle -/
theorem triangle_3_4_5 : can_form_triangle 3 4 5 := by
  sorry

end NUMINAMATH_CALUDE_triangle_3_4_5_l2551_255197


namespace NUMINAMATH_CALUDE_tv_price_decrease_l2551_255189

theorem tv_price_decrease (x : ℝ) : 
  (1 - x / 100) * (1 + 55 / 100) = 1 + 24 / 100 → x = 20 := by
  sorry

end NUMINAMATH_CALUDE_tv_price_decrease_l2551_255189


namespace NUMINAMATH_CALUDE_divisibility_implies_equality_l2551_255170

theorem divisibility_implies_equality (a b n : ℕ) 
  (h : ∀ (k : ℕ), k ≠ b → (b - k) ∣ (a - k^n)) : 
  a = b^n := by
  sorry

end NUMINAMATH_CALUDE_divisibility_implies_equality_l2551_255170


namespace NUMINAMATH_CALUDE_equity_investment_l2551_255113

def total_investment : ℝ := 250000

theorem equity_investment (debt : ℝ) 
  (h1 : debt + 3 * debt = total_investment) : 
  3 * debt = 187500 := by
  sorry

#check equity_investment

end NUMINAMATH_CALUDE_equity_investment_l2551_255113


namespace NUMINAMATH_CALUDE_lake_fish_population_l2551_255111

/-- Represents the fish population in a lake --/
structure FishPopulation where
  initial_tagged : ℕ
  second_catch : ℕ
  tagged_in_second_catch : ℕ
  new_migrants : ℕ

/-- Calculates the approximate total number of fish in the lake --/
def approximate_total_fish (fp : FishPopulation) : ℕ :=
  (fp.initial_tagged * fp.second_catch) / fp.tagged_in_second_catch

/-- The main theorem stating the approximate number of fish in the lake --/
theorem lake_fish_population (fp : FishPopulation) 
  (h1 : fp.initial_tagged = 500)
  (h2 : fp.second_catch = 300)
  (h3 : fp.tagged_in_second_catch = 6)
  (h4 : fp.new_migrants = 250) :
  approximate_total_fish fp = 25000 := by
  sorry

#eval approximate_total_fish { initial_tagged := 500, second_catch := 300, tagged_in_second_catch := 6, new_migrants := 250 }

end NUMINAMATH_CALUDE_lake_fish_population_l2551_255111


namespace NUMINAMATH_CALUDE_sisters_get_five_bars_l2551_255140

/-- Calculates the number of granola bars each sister receives when splitting the remaining bars evenly -/
def granola_bars_per_sister (total : ℕ) (set_aside : ℕ) (traded : ℕ) (num_sisters : ℕ) : ℕ :=
  (total - set_aside - traded) / num_sisters

/-- Proves that given the specific conditions, each sister receives 5 granola bars -/
theorem sisters_get_five_bars :
  let total := 20
  let set_aside := 7
  let traded := 3
  let num_sisters := 2
  granola_bars_per_sister total set_aside traded num_sisters = 5 := by
  sorry

#eval granola_bars_per_sister 20 7 3 2

end NUMINAMATH_CALUDE_sisters_get_five_bars_l2551_255140


namespace NUMINAMATH_CALUDE_inverse_of_proposition_l2551_255155

theorem inverse_of_proposition :
  (∀ a b : ℝ, a = -2*b → a^2 = 4*b^2) →
  (∀ a b : ℝ, a^2 = 4*b^2 → a = -2*b) :=
by sorry

end NUMINAMATH_CALUDE_inverse_of_proposition_l2551_255155


namespace NUMINAMATH_CALUDE_square_circle_area_ratio_l2551_255171

theorem square_circle_area_ratio (s : ℝ) (r : ℝ) (h1 : s > 0) (h2 : r > 0) (h3 : s = 2 * r) :
  (s^2) / (π * r^2) = 4 / π :=
sorry

end NUMINAMATH_CALUDE_square_circle_area_ratio_l2551_255171


namespace NUMINAMATH_CALUDE_quadrangular_prism_has_12_edges_l2551_255119

/-- Number of edges in a prism with n sides -/
def prism_edges (n : ℕ) : ℕ := 3 * n

/-- Number of edges in a pyramid with n sides -/
def pyramid_edges (n : ℕ) : ℕ := 2 * n

theorem quadrangular_prism_has_12_edges :
  prism_edges 4 = 12 ∧
  pyramid_edges 4 ≠ 12 ∧
  pyramid_edges 5 ≠ 12 ∧
  prism_edges 5 ≠ 12 :=
by sorry

end NUMINAMATH_CALUDE_quadrangular_prism_has_12_edges_l2551_255119


namespace NUMINAMATH_CALUDE_repeating_decimal_ratio_l2551_255190

/-- Represents a repeating decimal with a 3-digit repetend -/
def RepeatingDecimal (whole : ℕ) (repetend : ℕ) : ℚ :=
  whole + (repetend : ℚ) / 999

theorem repeating_decimal_ratio : 
  (RepeatingDecimal 0 833) / (RepeatingDecimal 1 666) = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_repeating_decimal_ratio_l2551_255190


namespace NUMINAMATH_CALUDE_special_permutations_l2551_255166

def word_length : ℕ := 7
def num_vowels : ℕ := 3
def num_consonants : ℕ := 4

theorem special_permutations :
  (word_length.choose num_vowels) * (num_consonants.factorial) = 840 := by
  sorry

end NUMINAMATH_CALUDE_special_permutations_l2551_255166


namespace NUMINAMATH_CALUDE_cherries_cost_correct_l2551_255109

/-- The amount Alyssa paid for cherries -/
def cherries_cost : ℚ := 985 / 100

/-- The amount Alyssa paid for grapes -/
def grapes_cost : ℚ := 1208 / 100

/-- The total amount Alyssa spent -/
def total_spent : ℚ := 2193 / 100

/-- Theorem stating that the amount Alyssa paid for cherries is correct -/
theorem cherries_cost_correct : cherries_cost = total_spent - grapes_cost := by
  sorry

end NUMINAMATH_CALUDE_cherries_cost_correct_l2551_255109


namespace NUMINAMATH_CALUDE_attendee_difference_l2551_255152

/-- The number of attendees from Company A -/
def company_A : ℕ := 30

/-- The number of attendees from Company B -/
def company_B : ℕ := 2 * company_A

/-- The number of attendees from Company C -/
def company_C : ℕ := company_A + 10

/-- The number of attendees from Company D -/
def company_D : ℕ := company_C - 25

/-- The number of attendees not from companies A, B, C, or D -/
def other_attendees : ℕ := 20

/-- The total number of attendees -/
def total_attendees : ℕ := 185

theorem attendee_difference :
  company_A + company_B + company_C + company_D + other_attendees = total_attendees ∧
  company_C - company_D = 25 := by
  sorry

end NUMINAMATH_CALUDE_attendee_difference_l2551_255152


namespace NUMINAMATH_CALUDE_quadratic_root_implies_s_value_l2551_255174

theorem quadratic_root_implies_s_value 
  (r s : ℝ) 
  (h : (4 + 3*I : ℂ) = -r/(2*2) + (r^2/(2*2)^2 - s/2).sqrt) : 
  s = 50 := by
sorry

end NUMINAMATH_CALUDE_quadratic_root_implies_s_value_l2551_255174


namespace NUMINAMATH_CALUDE_reciprocal_of_abs_neg_three_l2551_255172

theorem reciprocal_of_abs_neg_three (x : ℝ) : x = |(-3)| → 1 / x = 1 / 3 := by
  sorry

end NUMINAMATH_CALUDE_reciprocal_of_abs_neg_three_l2551_255172


namespace NUMINAMATH_CALUDE_negation_equivalence_l2551_255118

theorem negation_equivalence :
  (¬ ∀ x : ℝ, |x - 2| + |x - 4| > 3) ↔ (∃ x : ℝ, |x - 2| + |x - 4| ≤ 3) := by
  sorry

end NUMINAMATH_CALUDE_negation_equivalence_l2551_255118


namespace NUMINAMATH_CALUDE_least_positive_integer_congruence_l2551_255187

theorem least_positive_integer_congruence :
  ∃ (x : ℕ), x > 0 ∧ (x + 5419 : ℤ) ≡ 3789 [ZMOD 15] ∧
  ∀ (y : ℕ), y > 0 ∧ (y + 5419 : ℤ) ≡ 3789 [ZMOD 15] → x ≤ y :=
by sorry

end NUMINAMATH_CALUDE_least_positive_integer_congruence_l2551_255187


namespace NUMINAMATH_CALUDE_ben_owes_rachel_l2551_255105

theorem ben_owes_rachel (rate : ℚ) (lawns_mowed : ℚ) 
  (h1 : rate = 13 / 3) 
  (h2 : lawns_mowed = 8 / 5) : 
  rate * lawns_mowed = 104 / 15 := by
  sorry

end NUMINAMATH_CALUDE_ben_owes_rachel_l2551_255105


namespace NUMINAMATH_CALUDE_component_is_unqualified_l2551_255120

-- Define the nominal diameter and tolerance
def nominal_diameter : ℝ := 20
def tolerance : ℝ := 0.02

-- Define the measured diameter
def measured_diameter : ℝ := 19.9

-- Define what it means for a component to be qualified
def is_qualified (d : ℝ) : Prop :=
  nominal_diameter - tolerance ≤ d ∧ d ≤ nominal_diameter + tolerance

-- Theorem stating that the component is unqualified
theorem component_is_unqualified : ¬ is_qualified measured_diameter := by
  sorry

end NUMINAMATH_CALUDE_component_is_unqualified_l2551_255120


namespace NUMINAMATH_CALUDE_unique_positive_number_l2551_255196

theorem unique_positive_number : ∃! x : ℝ, x > 0 ∧ x - 4 = 21 / x := by
  sorry

end NUMINAMATH_CALUDE_unique_positive_number_l2551_255196


namespace NUMINAMATH_CALUDE_extreme_values_of_f_l2551_255169

def f (x : ℝ) : ℝ := 3 * x^5 - 5 * x^3

theorem extreme_values_of_f :
  ∃ (a b : ℝ), (∀ x : ℝ, f x ≤ f a ∨ f x ≥ f b) ∧
               (∀ c : ℝ, (∀ x : ℝ, f x ≤ f c ∨ f x ≥ f c) → c = a ∨ c = b) :=
sorry

end NUMINAMATH_CALUDE_extreme_values_of_f_l2551_255169


namespace NUMINAMATH_CALUDE_problem_statements_l2551_255163

theorem problem_statements :
  -- Statement 1
  (∀ x : ℝ, (x^2 - 3*x + 2 = 0 → x = 1) ↔ (x ≠ 1 → x^2 - 3*x + 2 ≠ 0)) ∧
  -- Statement 2
  (∀ x : ℝ, x > 2 → x^2 - 3*x + 2 > 0) ∧
  (∃ x : ℝ, x ≤ 2 ∧ x^2 - 3*x + 2 > 0) ∧
  -- Statement 3
  (∃ p q : Prop, ¬(p ∧ q) ∧ (p ∨ q)) ∧
  -- Statement 4
  (¬(∃ x : ℝ, x^2 + x + 1 < 0) ↔ (∀ x : ℝ, x^2 + x + 1 ≥ 0)) :=
by sorry

end NUMINAMATH_CALUDE_problem_statements_l2551_255163

import Mathlib

namespace NUMINAMATH_CALUDE_prime_roots_range_l1372_137204

theorem prime_roots_range (p : ℕ) (h_prime : Nat.Prime p) 
  (h_roots : ∃ x y : ℤ, x^2 + p*x - 444*p = 0 ∧ y^2 + p*y - 444*p = 0) : 
  31 < p ∧ p ≤ 41 :=
sorry

end NUMINAMATH_CALUDE_prime_roots_range_l1372_137204


namespace NUMINAMATH_CALUDE_rectangle_area_l1372_137270

/-- Given a rectangle where the length is four times the width and the perimeter is 200 cm,
    prove that its area is 1600 square centimeters. -/
theorem rectangle_area (w : ℝ) (h1 : w > 0) (h2 : 8 * w + 2 * w = 200) : w * (4 * w) = 1600 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_area_l1372_137270


namespace NUMINAMATH_CALUDE_exists_quadratic_polynomial_with_constant_negative_two_l1372_137240

/-- A quadratic polynomial in x and y with constant term -2 -/
def quadratic_polynomial (x y : ℝ) : ℝ := 15 * x^2 - y - 2

/-- Theorem stating the existence of a quadratic polynomial in x and y with constant term -2 -/
theorem exists_quadratic_polynomial_with_constant_negative_two :
  ∃ (f : ℝ → ℝ → ℝ), (∃ (a b c d e : ℝ), ∀ (x y : ℝ), 
    f x y = a * x^2 + b * x * y + c * y^2 + d * x + e * y - 2) :=
sorry

end NUMINAMATH_CALUDE_exists_quadratic_polynomial_with_constant_negative_two_l1372_137240


namespace NUMINAMATH_CALUDE_kate_age_problem_l1372_137221

theorem kate_age_problem :
  ∃! n : ℕ, n > 0 ∧ n.factorial = 1307674368000 :=
by
  -- The proof goes here
  sorry

end NUMINAMATH_CALUDE_kate_age_problem_l1372_137221


namespace NUMINAMATH_CALUDE_rice_distribution_theorem_l1372_137245

/-- Represents the amount of rice in a container after dividing the total rice equally -/
def rice_per_container (total_pounds : ℚ) (num_containers : ℕ) : ℚ :=
  (total_pounds * 16) / num_containers

/-- Theorem stating that dividing 49 and 3/4 pounds of rice equally among 7 containers 
    results in approximately 114 ounces of rice per container -/
theorem rice_distribution_theorem :
  ∃ (ε : ℚ), ε > 0 ∧ ε < 1 ∧ 
  |rice_per_container (49 + 3/4) 7 - 114| < ε :=
sorry

end NUMINAMATH_CALUDE_rice_distribution_theorem_l1372_137245


namespace NUMINAMATH_CALUDE_solve_problem_l1372_137248

/-- The number of Adidas shoes Alice sold to meet her quota -/
def problem : Prop :=
  let quota : ℕ := 1000
  let adidas_price : ℕ := 45
  let nike_price : ℕ := 60
  let reebok_price : ℕ := 35
  let nike_sold : ℕ := 8
  let reebok_sold : ℕ := 9
  let above_goal : ℕ := 65
  ∃ adidas_sold : ℕ,
    adidas_sold * adidas_price + nike_sold * nike_price + reebok_sold * reebok_price = quota + above_goal ∧
    adidas_sold = 6

theorem solve_problem : problem := by
  sorry

end NUMINAMATH_CALUDE_solve_problem_l1372_137248


namespace NUMINAMATH_CALUDE_angle_between_quito_and_kampala_l1372_137235

/-- The angle at the center of a spherical Earth between two points on the equator -/
def angle_at_center (west_longitude east_longitude : ℝ) : ℝ :=
  west_longitude + east_longitude

/-- Theorem: The angle at the center of a spherical Earth between two points,
    one at 78° W and the other at 32° E, both on the equator, is 110°. -/
theorem angle_between_quito_and_kampala :
  angle_at_center 78 32 = 110 := by
  sorry

end NUMINAMATH_CALUDE_angle_between_quito_and_kampala_l1372_137235


namespace NUMINAMATH_CALUDE_price_per_drawing_l1372_137229

-- Define the variables
def saturday_sales : ℕ := 24
def sunday_sales : ℕ := 16
def total_revenue : ℕ := 800

-- Define the theorem
theorem price_per_drawing : 
  ∃ (price : ℚ), price * (saturday_sales + sunday_sales) = total_revenue ∧ price = 20 := by
  sorry

end NUMINAMATH_CALUDE_price_per_drawing_l1372_137229


namespace NUMINAMATH_CALUDE_x_equals_one_sufficient_not_necessary_l1372_137209

theorem x_equals_one_sufficient_not_necessary (x : ℝ) :
  (x = 1 → x * (x - 1) = 0) ∧ (∃ y : ℝ, y ≠ 1 ∧ y * (y - 1) = 0) := by
  sorry

end NUMINAMATH_CALUDE_x_equals_one_sufficient_not_necessary_l1372_137209


namespace NUMINAMATH_CALUDE_greatest_distance_between_circle_centers_l1372_137222

/-- The greatest distance between centers of two circles in a rectangle -/
theorem greatest_distance_between_circle_centers
  (rectangle_width : ℝ)
  (rectangle_height : ℝ)
  (circle_diameter : ℝ)
  (h_width : rectangle_width = 24)
  (h_height : rectangle_height = 18)
  (h_diameter : circle_diameter = 8)
  (h_nonneg_width : 0 ≤ rectangle_width)
  (h_nonneg_height : 0 ≤ rectangle_height)
  (h_nonneg_diameter : 0 ≤ circle_diameter)
  (h_fit : circle_diameter ≤ min rectangle_width rectangle_height) :
  ∃ d : ℝ, d = Real.sqrt 356 ∧
    ∀ d' : ℝ, d' ≤ d ∧
      ∃ (x₁ y₁ x₂ y₂ : ℝ),
        0 ≤ x₁ ∧ x₁ ≤ rectangle_width ∧
        0 ≤ y₁ ∧ y₁ ≤ rectangle_height ∧
        0 ≤ x₂ ∧ x₂ ≤ rectangle_width ∧
        0 ≤ y₂ ∧ y₂ ≤ rectangle_height ∧
        circle_diameter / 2 ≤ x₁ ∧ x₁ ≤ rectangle_width - circle_diameter / 2 ∧
        circle_diameter / 2 ≤ y₁ ∧ y₁ ≤ rectangle_height - circle_diameter / 2 ∧
        circle_diameter / 2 ≤ x₂ ∧ x₂ ≤ rectangle_width - circle_diameter / 2 ∧
        circle_diameter / 2 ≤ y₂ ∧ y₂ ≤ rectangle_height - circle_diameter / 2 ∧
        d' = Real.sqrt ((x₁ - x₂)^2 + (y₁ - y₂)^2) :=
by
  sorry

end NUMINAMATH_CALUDE_greatest_distance_between_circle_centers_l1372_137222


namespace NUMINAMATH_CALUDE_payment_is_two_l1372_137280

/-- The amount Edmund needs to save -/
def saving_goal : ℕ := 75

/-- The number of chores Edmund normally does per week -/
def normal_chores_per_week : ℕ := 12

/-- The number of chores Edmund does per day during the saving period -/
def chores_per_day : ℕ := 4

/-- The number of days Edmund works during the saving period -/
def working_days : ℕ := 14

/-- The total amount Edmund earns for extra chores -/
def total_earned : ℕ := 64

/-- Calculates the number of extra chores Edmund does -/
def extra_chores : ℕ := chores_per_day * working_days - normal_chores_per_week * 2

/-- The payment per extra chore -/
def payment_per_extra_chore : ℚ := total_earned / extra_chores

theorem payment_is_two :
  payment_per_extra_chore = 2 := by sorry

end NUMINAMATH_CALUDE_payment_is_two_l1372_137280


namespace NUMINAMATH_CALUDE_inequality_system_solution_l1372_137287

theorem inequality_system_solution (a : ℝ) : 
  (∀ x : ℝ, (5*x - 3 < 3*x + 5 ∧ x < a) ↔ x < 4) → a ≥ 4 := by
  sorry

end NUMINAMATH_CALUDE_inequality_system_solution_l1372_137287


namespace NUMINAMATH_CALUDE_eggs_per_box_l1372_137273

theorem eggs_per_box (total_eggs : ℝ) (num_boxes : ℝ) 
  (h1 : total_eggs = 3.0) 
  (h2 : num_boxes = 2.0) 
  (h3 : num_boxes ≠ 0) : 
  total_eggs / num_boxes = 1.5 := by
  sorry

end NUMINAMATH_CALUDE_eggs_per_box_l1372_137273


namespace NUMINAMATH_CALUDE_fractional_equation_simplification_l1372_137220

theorem fractional_equation_simplification (x : ℝ) (h : x ≠ 2) : 
  (x / (x - 2) - 2 = 3 / (2 - x)) ↔ (x - 2 * (x - 2) = -3) :=
sorry

end NUMINAMATH_CALUDE_fractional_equation_simplification_l1372_137220


namespace NUMINAMATH_CALUDE_sum_of_roots_quadratic_undefined_expression_sum_l1372_137293

theorem sum_of_roots_quadratic : ∀ (a b c : ℝ), a ≠ 0 →
  let roots := {x : ℝ | a * x^2 + b * x + c = 0}
  (∃ x₁ x₂, roots = {x₁, x₂}) →
  (∃ s, ∀ x ∈ roots, ∃ y ∈ roots, x + y = s) →
  (∃ s, ∀ x ∈ roots, ∃ y ∈ roots, x + y = s) → s = -b / a :=
by sorry

theorem undefined_expression_sum : 
  let roots := {x : ℝ | x^2 - 7*x + 12 = 0}
  (∃ x₁ x₂, roots = {x₁, x₂}) →
  (∃ s, ∀ x ∈ roots, ∃ y ∈ roots, x + y = s) →
  s = 7 :=
by sorry

end NUMINAMATH_CALUDE_sum_of_roots_quadratic_undefined_expression_sum_l1372_137293


namespace NUMINAMATH_CALUDE_ratio_S4_a3_l1372_137207

/-- Given a sequence {a_n} where S_n is the sum of its first n terms -/
def S (n : ℕ) : ℚ := 2^n - 1

/-- The n-th term of the sequence -/
def a (n : ℕ) : ℚ := S n - S (n-1)

/-- The theorem to prove -/
theorem ratio_S4_a3 : S 4 / a 3 = 15/4 := by sorry

end NUMINAMATH_CALUDE_ratio_S4_a3_l1372_137207


namespace NUMINAMATH_CALUDE_linear_regression_at_25_l1372_137292

/-- Linear regression function -/
def linear_regression (x : ℝ) : ℝ := 0.50 * x - 0.81

/-- Theorem: The linear regression equation y = 0.50x - 0.81 yields y = 11.69 when x = 25 -/
theorem linear_regression_at_25 : linear_regression 25 = 11.69 := by
  sorry

end NUMINAMATH_CALUDE_linear_regression_at_25_l1372_137292


namespace NUMINAMATH_CALUDE_intersection_complement_when_m_2_intersection_complement_empty_iff_l1372_137219

-- Define the sets A and B
def A : Set ℝ := {x | 1 < x ∧ x < 4}
def B (m : ℝ) : Set ℝ := {x | x ≤ 3*m - 4 ∨ x ≥ 8 + m}

-- Theorem for part 1
theorem intersection_complement_when_m_2 :
  A ∩ (Set.univ \ B 2) = {x | 2 < x ∧ x < 4} := by sorry

-- Theorem for part 2
theorem intersection_complement_empty_iff (m : ℝ) :
  m < 6 →
  (A ∩ (Set.univ \ B m) = ∅ ↔ m ≤ -7 ∨ (8/3 ≤ m ∧ m < 6)) := by sorry

end NUMINAMATH_CALUDE_intersection_complement_when_m_2_intersection_complement_empty_iff_l1372_137219


namespace NUMINAMATH_CALUDE_same_color_probability_l1372_137208

/-- The probability of drawing two balls of the same color from a box of 6 balls -/
theorem same_color_probability : ℝ := by
  -- Define the number of balls of each color
  let red_balls : ℕ := 3
  let yellow_balls : ℕ := 2
  let blue_balls : ℕ := 1

  -- Define the total number of balls
  let total_balls : ℕ := red_balls + yellow_balls + blue_balls

  -- Define the probability of drawing two balls of the same color
  let prob : ℝ := 4 / 15

  -- Proof goes here
  sorry

end NUMINAMATH_CALUDE_same_color_probability_l1372_137208


namespace NUMINAMATH_CALUDE_parabola_intersection_difference_l1372_137299

-- Define the two parabolas
def f (x : ℝ) : ℝ := 3 * x^2 - 6 * x + 6
def g (x : ℝ) : ℝ := -2 * x^2 - 4 * x + 6

-- Define the set of intersection points
def intersection_points : Set ℝ := {x : ℝ | f x = g x}

-- Theorem statement
theorem parabola_intersection_difference :
  ∃ (a c : ℝ), a ∈ intersection_points ∧ c ∈ intersection_points ∧ c ≥ a ∧ c - a = 2/5 :=
sorry

end NUMINAMATH_CALUDE_parabola_intersection_difference_l1372_137299


namespace NUMINAMATH_CALUDE_degree_of_g_l1372_137217

-- Define the polynomial f(x)
def f (x : ℝ) : ℝ := -9 * x^4 + 2 * x^3 - 7 * x + 8

-- State the theorem
theorem degree_of_g (g : ℝ → ℝ) :
  (∃ (a b : ℝ), ∀ x, f x + g x = a * x + b) →  -- degree of f(x) + g(x) is 1
  (∃ (a b c d e : ℝ), a ≠ 0 ∧ ∀ x, g x = a * x^4 + b * x^3 + c * x^2 + d * x + e) :=  -- g(x) is a polynomial of degree 4
by sorry

end NUMINAMATH_CALUDE_degree_of_g_l1372_137217


namespace NUMINAMATH_CALUDE_beef_weight_loss_percentage_l1372_137227

theorem beef_weight_loss_percentage 
  (initial_weight : ℝ) 
  (processed_weight : ℝ) 
  (h1 : initial_weight = 1500) 
  (h2 : processed_weight = 750) : 
  (initial_weight - processed_weight) / initial_weight * 100 = 50 := by
sorry

end NUMINAMATH_CALUDE_beef_weight_loss_percentage_l1372_137227


namespace NUMINAMATH_CALUDE_cheese_cost_for_order_l1372_137202

/-- Represents the cost of cheese for a Mexican restaurant order --/
def cheese_cost (burrito_count : ℕ) (taco_count : ℕ) (enchilada_count : ℕ) : ℚ :=
  let cheddar_per_burrito : ℚ := 4
  let cheddar_per_taco : ℚ := 9
  let mozzarella_per_enchilada : ℚ := 5
  let cheddar_cost_per_ounce : ℚ := 4/5
  let mozzarella_cost_per_ounce : ℚ := 1
  (burrito_count * cheddar_per_burrito + taco_count * cheddar_per_taco) * cheddar_cost_per_ounce +
  (enchilada_count * mozzarella_per_enchilada) * mozzarella_cost_per_ounce

/-- Theorem stating the total cost of cheese for a specific order --/
theorem cheese_cost_for_order :
  cheese_cost 7 1 3 = 446/10 :=
sorry

end NUMINAMATH_CALUDE_cheese_cost_for_order_l1372_137202


namespace NUMINAMATH_CALUDE_angle_A_is_pi_over_two_l1372_137223

/-- 
Given a triangle ABC where:
- The sides opposite to angles A, B, C are a, b, c respectively
- b = √5
- c = 2
- cos B = 2/3

Prove that the measure of angle A is π/2
-/
theorem angle_A_is_pi_over_two (a b c : ℝ) (A B C : ℝ) : 
  b = Real.sqrt 5 → 
  c = 2 → 
  Real.cos B = 2/3 → 
  A + B + C = π → 
  0 < A ∧ A < π → 
  0 < B ∧ B < π → 
  0 < C ∧ C < π → 
  a * Real.sin B = b * Real.sin A → 
  b^2 = a^2 + c^2 - 2*a*c * Real.cos B → 
  A = π/2 := by
sorry

end NUMINAMATH_CALUDE_angle_A_is_pi_over_two_l1372_137223


namespace NUMINAMATH_CALUDE_alpha_value_l1372_137285

theorem alpha_value (α : Real) (h1 : α ∈ Set.Ioo 0 Real.pi) (h2 : Real.cos α = -1/2) : 
  α = 2 * Real.pi / 3 := by
  sorry

end NUMINAMATH_CALUDE_alpha_value_l1372_137285


namespace NUMINAMATH_CALUDE_necessary_condition_for_false_proposition_l1372_137211

theorem necessary_condition_for_false_proposition (a : ℝ) :
  (¬ ∃ x₀ : ℝ, x₀^2 - a*x₀ + 1 ≤ 0) → (-2 ≤ a ∧ a ≤ 2) :=
by sorry

end NUMINAMATH_CALUDE_necessary_condition_for_false_proposition_l1372_137211


namespace NUMINAMATH_CALUDE_congruent_mod_divisor_congruent_mod_polynomial_l1372_137265

/-- Definition of congruence modulo m -/
def congruent_mod (a b m : ℤ) : Prop :=
  ∃ k : ℤ, a - b = m * k

/-- Statement 1 -/
theorem congruent_mod_divisor (a b m d : ℤ) (hm : 0 < m) (hd : 0 < d) (hdiv : d ∣ m) 
    (h : congruent_mod a b m) : congruent_mod a b d := by
  sorry

/-- Definition of the polynomial f(x) = x³ - 2x + 5 -/
def f (x : ℤ) : ℤ := x^3 - 2*x + 5

/-- Statement 4 -/
theorem congruent_mod_polynomial (a b m : ℤ) (hm : 0 < m) 
    (h : congruent_mod a b m) : congruent_mod (f a) (f b) m := by
  sorry

end NUMINAMATH_CALUDE_congruent_mod_divisor_congruent_mod_polynomial_l1372_137265


namespace NUMINAMATH_CALUDE_room_length_calculation_l1372_137275

/-- Given a rectangular room with width 4 meters and a floor paving cost resulting in a total cost of 18700, prove that the length of the room is 5.5 meters. -/
theorem room_length_calculation (width : ℝ) (cost_per_sqm : ℝ) (total_cost : ℝ) (length : ℝ) : 
  width = 4 →
  cost_per_sqm = 850 →
  total_cost = 18700 →
  total_cost = cost_per_sqm * (length * width) →
  length = 5.5 := by
  sorry

end NUMINAMATH_CALUDE_room_length_calculation_l1372_137275


namespace NUMINAMATH_CALUDE_mikey_leaves_left_l1372_137271

/-- The number of leaves Mikey has left after some blow away -/
def leaves_left (initial : ℕ) (blown_away : ℕ) : ℕ :=
  initial - blown_away

/-- Theorem stating that Mikey has 112 leaves left -/
theorem mikey_leaves_left :
  leaves_left 356 244 = 112 := by
  sorry

end NUMINAMATH_CALUDE_mikey_leaves_left_l1372_137271


namespace NUMINAMATH_CALUDE_a_minus_b_equals_negative_nine_l1372_137206

theorem a_minus_b_equals_negative_nine
  (a b : ℝ)
  (h : |a + 5| + Real.sqrt (2 * b - 8) = 0) :
  a - b = -9 := by
sorry

end NUMINAMATH_CALUDE_a_minus_b_equals_negative_nine_l1372_137206


namespace NUMINAMATH_CALUDE_billion_two_hundred_million_scientific_notation_l1372_137236

/-- Scientific notation representation of a number -/
structure ScientificNotation where
  coefficient : ℝ
  exponent : ℤ
  is_valid : 1 ≤ coefficient ∧ coefficient < 10

/-- Converts a real number to scientific notation -/
noncomputable def toScientificNotation (x : ℝ) : ScientificNotation :=
  sorry

theorem billion_two_hundred_million_scientific_notation :
  toScientificNotation 1200000000 = ScientificNotation.mk 1.2 9 (by norm_num) :=
sorry

end NUMINAMATH_CALUDE_billion_two_hundred_million_scientific_notation_l1372_137236


namespace NUMINAMATH_CALUDE_trajectory_of_product_slopes_l1372_137251

/-- The trajectory of a moving point P whose product of slopes to fixed points A(-1,0) and B(1,0) is -1 -/
theorem trajectory_of_product_slopes (x y : ℝ) (hx : x ≠ 1 ∧ x ≠ -1) :
  (y / (x + 1)) * (y / (x - 1)) = -1 → x^2 + y^2 = 1 := by
sorry

end NUMINAMATH_CALUDE_trajectory_of_product_slopes_l1372_137251


namespace NUMINAMATH_CALUDE_parallel_vectors_m_value_l1372_137212

/-- 
Given two parallel vectors a and b in ℝ², where a = (-2, 1) and b = (1, m),
prove that m = -1/2.
-/
theorem parallel_vectors_m_value (a b : ℝ × ℝ) (m : ℝ) 
  (h1 : a = (-2, 1)) 
  (h2 : b = (1, m)) 
  (h3 : ∃ (k : ℝ), a = k • b) : 
  m = -1/2 := by
sorry

end NUMINAMATH_CALUDE_parallel_vectors_m_value_l1372_137212


namespace NUMINAMATH_CALUDE_product_real_implies_b_value_l1372_137295

/-- Given complex numbers z₁ and z₂, if their product is real, then b = -2 -/
theorem product_real_implies_b_value (z₁ z₂ : ℂ) (b : ℝ) 
  (h₁ : z₁ = 1 + I) 
  (h₂ : z₂ = 2 + b * I) 
  (h₃ : (z₁ * z₂).im = 0) : 
  b = -2 := by
  sorry

end NUMINAMATH_CALUDE_product_real_implies_b_value_l1372_137295


namespace NUMINAMATH_CALUDE_smallest_number_divisible_by_multiple_l1372_137244

def isDivisibleBy (a b : ℕ) : Prop := ∃ k : ℕ, a = b * k

theorem smallest_number_divisible_by_multiple : 
  ∃! n : ℕ, (∀ m : ℕ, m < n → 
    ¬(isDivisibleBy (m - 6) 12 ∧ 
      isDivisibleBy (m - 6) 16 ∧ 
      isDivisibleBy (m - 6) 18 ∧ 
      isDivisibleBy (m - 6) 21 ∧ 
      isDivisibleBy (m - 6) 28)) ∧ 
    isDivisibleBy (n - 6) 12 ∧ 
    isDivisibleBy (n - 6) 16 ∧ 
    isDivisibleBy (n - 6) 18 ∧ 
    isDivisibleBy (n - 6) 21 ∧ 
    isDivisibleBy (n - 6) 28 ∧
    n = 1014 :=
by sorry

end NUMINAMATH_CALUDE_smallest_number_divisible_by_multiple_l1372_137244


namespace NUMINAMATH_CALUDE_odd_function_sum_l1372_137234

def is_odd (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = -f x

theorem odd_function_sum (f : ℝ → ℝ) (h_odd : is_odd f) (h_pos : ∀ x > 0, f x = 2*x - 3) :
  f (-2) + f 0 = -1 := by
  sorry

end NUMINAMATH_CALUDE_odd_function_sum_l1372_137234


namespace NUMINAMATH_CALUDE_unique_positive_solution_l1372_137262

theorem unique_positive_solution :
  ∃! (y : ℝ), y > 0 ∧ (y - 6) / 12 = 6 / (y - 12) ∧ y = 18 := by
  sorry

end NUMINAMATH_CALUDE_unique_positive_solution_l1372_137262


namespace NUMINAMATH_CALUDE_complex_equation_solution_l1372_137203

theorem complex_equation_solution (z : ℂ) :
  (1 + 2*I) * z = 4 + 3*I → z = 2 - I :=
by sorry

end NUMINAMATH_CALUDE_complex_equation_solution_l1372_137203


namespace NUMINAMATH_CALUDE_permutation_inequalities_l1372_137291

/-- Given a set X of n elements and 0 ≤ k ≤ n, a_{n,k} is the maximum number of permutations
    acting on X such that every two of them have at least k components in common -/
def a (n k : ℕ) : ℕ := sorry

/-- Given a set X of n elements and 0 ≤ k ≤ n, b_{n,k} is the maximum number of permutations
    acting on X such that every two of them have at most k components in common -/
def b (n k : ℕ) : ℕ := sorry

theorem permutation_inequalities (n k : ℕ) (h : k ≤ n) :
  a n k * b n (k - 1) ≤ n! ∧ ∀ p : ℕ, Nat.Prime p → a p 2 = p! / 2 := by
  sorry

end NUMINAMATH_CALUDE_permutation_inequalities_l1372_137291


namespace NUMINAMATH_CALUDE_complex_power_sum_l1372_137267

theorem complex_power_sum (z : ℂ) (h : z = -Complex.I) : z^100 + z^50 + 1 = -Complex.I := by
  sorry

end NUMINAMATH_CALUDE_complex_power_sum_l1372_137267


namespace NUMINAMATH_CALUDE_correct_num_technicians_l1372_137266

/-- Represents the number of technicians in the workshop -/
def num_technicians : ℕ := 7

/-- Represents the total number of workers in the workshop -/
def total_workers : ℕ := 21

/-- Represents the average salary of all workers -/
def avg_salary_all : ℕ := 8000

/-- Represents the average salary of technicians -/
def avg_salary_technicians : ℕ := 12000

/-- Represents the average salary of non-technicians -/
def avg_salary_rest : ℕ := 6000

/-- Theorem stating that the number of technicians is correct given the conditions -/
theorem correct_num_technicians : 
  num_technicians * avg_salary_technicians + 
  (total_workers - num_technicians) * avg_salary_rest = 
  total_workers * avg_salary_all :=
sorry

#check correct_num_technicians

end NUMINAMATH_CALUDE_correct_num_technicians_l1372_137266


namespace NUMINAMATH_CALUDE_dusty_single_layer_purchase_l1372_137255

/-- Represents the cost and quantity of cake slices purchased by Dusty -/
structure CakePurchase where
  single_layer_price : ℕ
  double_layer_price : ℕ
  double_layer_quantity : ℕ
  payment : ℕ
  change : ℕ

/-- Calculates the number of single layer cake slices purchased -/
def single_layer_quantity (purchase : CakePurchase) : ℕ :=
  (purchase.payment - purchase.change - purchase.double_layer_price * purchase.double_layer_quantity) / purchase.single_layer_price

/-- Theorem stating that Dusty bought 7 single layer cake slices -/
theorem dusty_single_layer_purchase :
  let purchase := CakePurchase.mk 4 7 5 100 37
  single_layer_quantity purchase = 7 := by
  sorry

end NUMINAMATH_CALUDE_dusty_single_layer_purchase_l1372_137255


namespace NUMINAMATH_CALUDE_solve_equation_l1372_137284

theorem solve_equation (x : ℝ) : 3 * x = (36 - x) + 16 → x = 13 := by
  sorry

end NUMINAMATH_CALUDE_solve_equation_l1372_137284


namespace NUMINAMATH_CALUDE_sum_22_probability_l1372_137214

/-- Represents a 20-faced die with some numbered faces and some blank faces -/
structure Die where
  numbered_faces : Finset ℕ
  blank_faces : ℕ
  total_faces : numbered_faces.card + blank_faces = 20

/-- The first die with faces 1 through 18 and two blank faces -/
def die1 : Die where
  numbered_faces := Finset.range 18
  blank_faces := 2
  total_faces := sorry

/-- The second die with faces 2 through 9 and 11 through 20 and two blank faces -/
def die2 : Die where
  numbered_faces := (Finset.range 8).image (λ x => x + 2) ∪ (Finset.range 10).image (λ x => x + 11)
  blank_faces := 2
  total_faces := sorry

/-- The probability of an event given the number of favorable outcomes and total outcomes -/
def probability (favorable : ℕ) (total : ℕ) : ℚ := favorable / total

/-- The theorem to be proved -/
theorem sum_22_probability :
  probability (die1.numbered_faces.card * die2.numbered_faces.card) (20 * 20) = 1 / 40 := by
  sorry

end NUMINAMATH_CALUDE_sum_22_probability_l1372_137214


namespace NUMINAMATH_CALUDE_exam_maximum_marks_l1372_137259

/-- Given an exam where:
  1. The passing mark is 80% of the maximum marks.
  2. A student got 200 marks.
  3. The student failed by 200 marks (i.e., needs 200 more marks to pass).
  Prove that the maximum marks for the exam is 500. -/
theorem exam_maximum_marks :
  ∀ (max_marks : ℕ),
  (max_marks : ℚ) * (80 : ℚ) / (100 : ℚ) = (200 : ℚ) + (200 : ℚ) →
  max_marks = 500 := by
sorry

end NUMINAMATH_CALUDE_exam_maximum_marks_l1372_137259


namespace NUMINAMATH_CALUDE_max_perpendicular_faces_theorem_l1372_137297

/-- The maximum number of faces perpendicular to the base in an n-sided pyramid -/
def max_perpendicular_faces (n : ℕ) : ℕ :=
  if n % 2 = 0 then n / 2 else (n + 1) / 2

/-- Theorem stating the maximum number of faces perpendicular to the base in an n-sided pyramid -/
theorem max_perpendicular_faces_theorem (n : ℕ) (h : n > 2) :
  max_perpendicular_faces n = if n % 2 = 0 then n / 2 else (n + 1) / 2 :=
by sorry

end NUMINAMATH_CALUDE_max_perpendicular_faces_theorem_l1372_137297


namespace NUMINAMATH_CALUDE_integral_equality_l1372_137272

theorem integral_equality : ∫ (x : ℝ) in Set.Icc π (2*π), (1 - Real.cos x) / (x - Real.sin x)^2 = 1 / (2*π) := by
  sorry

end NUMINAMATH_CALUDE_integral_equality_l1372_137272


namespace NUMINAMATH_CALUDE_addition_of_decimals_l1372_137263

theorem addition_of_decimals : 7.56 + 4.29 = 11.85 := by
  sorry

end NUMINAMATH_CALUDE_addition_of_decimals_l1372_137263


namespace NUMINAMATH_CALUDE_sufficient_condition_for_inequality_l1372_137294

theorem sufficient_condition_for_inequality (a : ℝ) :
  a ≥ 5 → ∀ x ∈ Set.Icc 1 2, x^2 - a ≤ 0 := by sorry

end NUMINAMATH_CALUDE_sufficient_condition_for_inequality_l1372_137294


namespace NUMINAMATH_CALUDE_candy_box_original_price_l1372_137225

/-- Given a candy box with an original price, which after a 25% increase becomes 10 pounds,
    prove that the original price was 8 pounds. -/
theorem candy_box_original_price (original_price : ℝ) : 
  (original_price * 1.25 = 10) → original_price = 8 := by
  sorry

end NUMINAMATH_CALUDE_candy_box_original_price_l1372_137225


namespace NUMINAMATH_CALUDE_paint_per_statue_l1372_137278

theorem paint_per_statue (total_paint : ℚ) (num_statues : ℕ) 
  (h1 : total_paint = 3/6)
  (h2 : num_statues = 3) :
  total_paint / num_statues = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_paint_per_statue_l1372_137278


namespace NUMINAMATH_CALUDE_impossible_equal_sum_distribution_l1372_137269

theorem impossible_equal_sum_distribution : ∀ n : ℕ, 2 ≤ n → n ≤ 14 →
  ¬ ∃ (partition : List (List ℕ)), 
    (∀ group ∈ partition, ∀ x ∈ group, 1 ≤ x ∧ x ≤ 14) ∧
    (partition.length = n) ∧
    (∀ group ∈ partition, group.sum = 105 / n) ∧
    (partition.join.toFinset = Finset.range 14) :=
by sorry

end NUMINAMATH_CALUDE_impossible_equal_sum_distribution_l1372_137269


namespace NUMINAMATH_CALUDE_lowest_score_within_two_std_dev_l1372_137264

/-- Given a mean score and standard deviation, calculate the lowest score within a certain number of standard deviations from the mean. -/
def lowest_score (mean : ℝ) (std_dev : ℝ) (num_std_dev : ℝ) : ℝ :=
  mean - num_std_dev * std_dev

/-- Theorem stating that for a mean of 60 and standard deviation of 10, the lowest score within 2 standard deviations is 40. -/
theorem lowest_score_within_two_std_dev :
  lowest_score 60 10 2 = 40 := by
  sorry

#eval lowest_score 60 10 2

end NUMINAMATH_CALUDE_lowest_score_within_two_std_dev_l1372_137264


namespace NUMINAMATH_CALUDE_integer_solutions_count_l1372_137216

theorem integer_solutions_count :
  let f : ℤ → ℤ → ℤ := λ x y => 6 * y^2 + 3 * x * y + x + 2 * y - 72
  ∃! s : Finset (ℤ × ℤ), (∀ (x y : ℤ), (x, y) ∈ s ↔ f x y = 0) ∧ Finset.card s = 4 :=
by sorry

end NUMINAMATH_CALUDE_integer_solutions_count_l1372_137216


namespace NUMINAMATH_CALUDE_waiter_problem_l1372_137290

/-- Given an initial number of customers and two groups of customers leaving,
    calculate the final number of customers remaining. -/
def remaining_customers (initial : ℝ) (first_group : ℝ) (second_group : ℝ) : ℝ :=
  initial - first_group - second_group

/-- Theorem stating that for the given problem, the number of remaining customers is 3.0 -/
theorem waiter_problem (initial : ℝ) (first_group : ℝ) (second_group : ℝ)
    (h1 : initial = 36.0)
    (h2 : first_group = 19.0)
    (h3 : second_group = 14.0) :
    remaining_customers initial first_group second_group = 3.0 := by
  sorry

end NUMINAMATH_CALUDE_waiter_problem_l1372_137290


namespace NUMINAMATH_CALUDE_platform_length_l1372_137282

/-- Given a train and platform with specific properties, prove the length of the platform -/
theorem platform_length 
  (train_length : ℝ)
  (time_cross_platform : ℝ)
  (time_cross_pole : ℝ)
  (h1 : train_length = 300)
  (h2 : time_cross_platform = 36)
  (h3 : time_cross_pole = 18) :
  let train_speed := train_length / time_cross_pole
  let platform_length := train_speed * time_cross_platform - train_length
  platform_length = 300 := by
sorry


end NUMINAMATH_CALUDE_platform_length_l1372_137282


namespace NUMINAMATH_CALUDE_stable_table_configurations_l1372_137277

def stableConfigurations (n : ℕ+) : ℕ :=
  (1/3) * (n+1) * (2*n^2 + 4*n + 3)

theorem stable_table_configurations (n : ℕ+) :
  (stableConfigurations n) =
  (Finset.sum (Finset.range (2*n+1)) (λ k =>
    (if k ≤ n then k + 1 else 2*n - k + 1)^2)) :=
sorry

end NUMINAMATH_CALUDE_stable_table_configurations_l1372_137277


namespace NUMINAMATH_CALUDE_geometric_sequence_308th_term_l1372_137243

def geometric_sequence (a₁ : ℝ) (r : ℝ) (n : ℕ) : ℝ :=
  a₁ * r ^ (n - 1)

theorem geometric_sequence_308th_term :
  let a₁ := 12
  let a₂ := -24
  let r := a₂ / a₁
  geometric_sequence a₁ r 308 = -2^307 * 12 :=
by sorry

end NUMINAMATH_CALUDE_geometric_sequence_308th_term_l1372_137243


namespace NUMINAMATH_CALUDE_geometric_sequence_third_term_l1372_137258

-- Define a geometric sequence
def geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = a n * r

-- State the theorem
theorem geometric_sequence_third_term
  (a : ℕ → ℝ)
  (h_geom : geometric_sequence a)
  (h_a1 : a 1 = 1)
  (h_a5 : a 5 = 16) :
  a 3 = 4 := by
sorry

end NUMINAMATH_CALUDE_geometric_sequence_third_term_l1372_137258


namespace NUMINAMATH_CALUDE_f_inequality_l1372_137224

/-- An odd function f: ℝ → ℝ with specific properties -/
def f_properties (f : ℝ → ℝ) : Prop :=
  (∀ x, f (-x) = -f x) ∧ 
  (∀ x, f (x - 4) = -f x) ∧
  (∀ x y, 0 ≤ x ∧ x < y ∧ y ≤ 2 → f x < f y)

/-- Theorem stating the inequality for the given function -/
theorem f_inequality (f : ℝ → ℝ) (h : f_properties f) : 
  f (-1) < f 4 ∧ f 4 < f 3 := by
  sorry

end NUMINAMATH_CALUDE_f_inequality_l1372_137224


namespace NUMINAMATH_CALUDE_cistern_theorem_l1372_137232

/-- Represents the cistern problem -/
def cistern_problem (capacity : ℝ) (leak_time : ℝ) (tap_rate : ℝ) : Prop :=
  let leak_rate : ℝ := capacity / leak_time
  let net_rate : ℝ := leak_rate - tap_rate
  let emptying_time : ℝ := capacity / net_rate
  emptying_time = 24

/-- The theorem statement for the cistern problem -/
theorem cistern_theorem :
  cistern_problem 480 20 4 := by sorry

end NUMINAMATH_CALUDE_cistern_theorem_l1372_137232


namespace NUMINAMATH_CALUDE_sum_u_v_equals_negative_42_over_77_l1372_137252

theorem sum_u_v_equals_negative_42_over_77 
  (u v : ℚ) 
  (eq1 : 3 * u - 7 * v = 17) 
  (eq2 : 5 * u + 3 * v = 1) : 
  u + v = -42 / 77 := by
sorry

end NUMINAMATH_CALUDE_sum_u_v_equals_negative_42_over_77_l1372_137252


namespace NUMINAMATH_CALUDE_characterize_function_l1372_137286

theorem characterize_function (n : ℕ) (hn : n ≥ 1) (hodd : Odd n) :
  ∃ (ε : Int) (d : ℕ) (c : Int),
    ε = 1 ∨ ε = -1 ∧
    d > 0 ∧
    d ∣ n ∧
    ∀ (f : ℤ → ℤ),
      (∀ (x y : ℤ), (f x - f y) ∣ (x^n - y^n)) →
      ∃ (ε' : Int) (d' : ℕ) (c' : Int),
        (ε' = 1 ∨ ε' = -1) ∧
        d' > 0 ∧
        d' ∣ n ∧
        ∀ (x : ℤ), f x = ε' * x^d' + c' :=
by sorry

end NUMINAMATH_CALUDE_characterize_function_l1372_137286


namespace NUMINAMATH_CALUDE_probability_all_black_is_correct_l1372_137210

def urn_black_balls : ℕ := 10
def urn_white_balls : ℕ := 5
def total_balls : ℕ := urn_black_balls + urn_white_balls
def drawn_balls : ℕ := 2

def probability_all_black : ℚ := (urn_black_balls.choose drawn_balls) / (total_balls.choose drawn_balls)

theorem probability_all_black_is_correct :
  probability_all_black = 3 / 7 :=
sorry

end NUMINAMATH_CALUDE_probability_all_black_is_correct_l1372_137210


namespace NUMINAMATH_CALUDE_elsas_final_marbles_l1372_137239

/-- Calculates the number of marbles Elsa has at the end of the day -/
def elsas_marbles : ℕ :=
  let initial := 150
  let after_breakfast := initial - (initial * 5 / 100)
  let after_lunch := after_breakfast - (after_breakfast * 2 / 5)
  let after_mom_gift := after_lunch + 25
  let after_susie_return := after_mom_gift + (after_breakfast * 2 / 5 * 150 / 100)
  let peter_exchange := 15
  let elsa_gives := peter_exchange * 3 / 5
  let elsa_receives := peter_exchange * 2 / 5
  let after_peter := after_susie_return - elsa_gives + elsa_receives
  let final := after_peter - (after_peter / 4)
  final

theorem elsas_final_marbles :
  elsas_marbles = 145 := by
  sorry

end NUMINAMATH_CALUDE_elsas_final_marbles_l1372_137239


namespace NUMINAMATH_CALUDE_tangency_distance_value_l1372_137288

/-- Configuration of four circles where three small circles of radius 2 are externally
    tangent to each other and internally tangent to a larger circle -/
structure CircleConfiguration where
  -- Radius of each small circle
  small_radius : ℝ
  -- Center of the large circle
  large_center : ℝ × ℝ
  -- Centers of the three small circles
  small_centers : Fin 3 → ℝ × ℝ
  -- The three small circles are externally tangent to each other
  small_circles_tangent : ∀ (i j : Fin 3), i ≠ j →
    ‖small_centers i - small_centers j‖ = 2 * small_radius
  -- The three small circles are internally tangent to the large circle
  large_circle_tangent : ∀ (i : Fin 3),
    ‖large_center - small_centers i‖ = ‖large_center - small_centers 0‖

/-- The distance from the center of the large circle to the point of tangency
    on one of the small circles in the given configuration -/
def tangency_distance (config : CircleConfiguration) : ℝ :=
  ‖config.large_center - config.small_centers 0‖ - config.small_radius

/-- Theorem stating that the tangency distance is equal to 2√3 - 2 -/
theorem tangency_distance_value (config : CircleConfiguration) 
    (h : config.small_radius = 2) : 
    tangency_distance config = 2 * Real.sqrt 3 - 2 := by
  sorry

end NUMINAMATH_CALUDE_tangency_distance_value_l1372_137288


namespace NUMINAMATH_CALUDE_intersection_P_Q_l1372_137233

def P : Set ℝ := {-3, 0, 2, 4}
def Q : Set ℝ := {x : ℝ | -1 < x ∧ x < 3}

theorem intersection_P_Q : P ∩ Q = {0, 2} := by sorry

end NUMINAMATH_CALUDE_intersection_P_Q_l1372_137233


namespace NUMINAMATH_CALUDE_boys_in_class_l1372_137241

/-- Proves that in a class of 20 students, if exactly one-third of the boys sit with a girl
    and exactly one-half of the girls sit with a boy, then there are 12 boys in the class. -/
theorem boys_in_class (total_students : ℕ) (boys : ℕ) (girls : ℕ) :
  total_students = 20 →
  boys + girls = total_students →
  (boys / 3 : ℚ) = (girls / 2 : ℚ) →
  boys = 12 := by
  sorry

end NUMINAMATH_CALUDE_boys_in_class_l1372_137241


namespace NUMINAMATH_CALUDE_product_of_radicals_l1372_137249

theorem product_of_radicals (q : ℝ) (hq : q > 0) :
  Real.sqrt (42 * q) * Real.sqrt (7 * q) * Real.sqrt (14 * q^3) = 14 * q^2 * Real.sqrt (42 * q) :=
by sorry

end NUMINAMATH_CALUDE_product_of_radicals_l1372_137249


namespace NUMINAMATH_CALUDE_simplify_expression_l1372_137257

theorem simplify_expression : 
  5 * Real.sqrt 3 + (Real.sqrt 4 + 2 * Real.sqrt 3) = 7 * Real.sqrt 3 + 2 :=
by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l1372_137257


namespace NUMINAMATH_CALUDE_john_wallet_dimes_l1372_137274

def total_amount : ℚ := 680 / 100  -- $6.80 as a rational number

theorem john_wallet_dimes :
  ∀ (d q : ℕ),  -- d: number of dimes, q: number of quarters
  d = q + 4 →  -- four more dimes than quarters
  (d : ℚ) * (10 / 100) + (q : ℚ) * (25 / 100) = total_amount →  -- total amount equation
  d = 22 :=
by sorry

end NUMINAMATH_CALUDE_john_wallet_dimes_l1372_137274


namespace NUMINAMATH_CALUDE_fencing_cost_calculation_l1372_137298

-- Define the plot dimensions
def length : ℝ := 60
def breadth : ℝ := 40

-- Define the cost per meter of fencing
def cost_per_meter : ℝ := 26.50

-- Calculate the perimeter
def perimeter : ℝ := 2 * (length + breadth)

-- Calculate the total cost
def total_cost : ℝ := perimeter * cost_per_meter

-- Theorem to prove
theorem fencing_cost_calculation :
  total_cost = 5300 :=
by sorry

end NUMINAMATH_CALUDE_fencing_cost_calculation_l1372_137298


namespace NUMINAMATH_CALUDE_divisibility_condition_l1372_137261

theorem divisibility_condition (n : ℤ) : 
  (n^5 + 3) % (n^2 + 1) = 0 ↔ n ∈ ({-3, -1, 0, 1, 2} : Set ℤ) := by sorry

end NUMINAMATH_CALUDE_divisibility_condition_l1372_137261


namespace NUMINAMATH_CALUDE_paint_coats_calculation_l1372_137201

/-- Proves the number of coats of paint that can be applied given the wall area,
    paint coverage, paint cost, and individual contributions. -/
theorem paint_coats_calculation (wall_area : ℝ) (paint_coverage : ℝ) (paint_cost : ℝ) (contribution : ℝ)
    (h_wall : wall_area = 1600)
    (h_coverage : paint_coverage = 400)
    (h_cost : paint_cost = 45)
    (h_contribution : contribution = 180) :
    ⌊(2 * contribution) / (paint_cost * (wall_area / paint_coverage))⌋ = 2 := by
  sorry

#check paint_coats_calculation

end NUMINAMATH_CALUDE_paint_coats_calculation_l1372_137201


namespace NUMINAMATH_CALUDE_function_condition_l1372_137237

noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  if x < 1 then 2^x + 1 else x^2 + a*x

theorem function_condition (a : ℝ) : f a (f a 0) = 3 * a → a = 4 := by
  sorry

end NUMINAMATH_CALUDE_function_condition_l1372_137237


namespace NUMINAMATH_CALUDE_sin_minus_cos_105_deg_l1372_137253

theorem sin_minus_cos_105_deg : 
  Real.sin (105 * π / 180) - Real.cos (105 * π / 180) = Real.sqrt 6 / 2 := by
  sorry

end NUMINAMATH_CALUDE_sin_minus_cos_105_deg_l1372_137253


namespace NUMINAMATH_CALUDE_watson_class_composition_l1372_137268

/-- Represents the number of students in each grade level in Ms. Watson's class -/
structure ClassComposition where
  kindergartners : Nat
  first_graders : Nat
  second_graders : Nat

/-- The total number of students in Ms. Watson's class -/
def total_students (c : ClassComposition) : Nat :=
  c.kindergartners + c.first_graders + c.second_graders

/-- Theorem stating that given the conditions of Ms. Watson's class, 
    there are 4 second graders -/
theorem watson_class_composition :
  ∃ (c : ClassComposition),
    c.kindergartners = 14 ∧
    c.first_graders = 24 ∧
    total_students c = 42 ∧
    c.second_graders = 4 := by
  sorry

end NUMINAMATH_CALUDE_watson_class_composition_l1372_137268


namespace NUMINAMATH_CALUDE_game_ends_in_six_rounds_l1372_137238

/-- Represents a player in the token game -/
inductive Player : Type
| A
| B
| C

/-- Represents the state of the game at any given round -/
structure GameState :=
  (tokens : Player → ℕ)

/-- Determines if the game has ended (any player has 0 tokens) -/
def game_ended (state : GameState) : Prop :=
  ∃ p : Player, state.tokens p = 0

/-- Simulates one round of the game -/
def play_round (state : GameState) : GameState :=
  sorry

/-- The initial state of the game -/
def initial_state : GameState :=
  { tokens := λ p => match p with
    | Player.A => 16
    | Player.B => 14
    | Player.C => 12 }

/-- Theorem stating that the game ends after exactly 6 rounds -/
theorem game_ends_in_six_rounds :
  let final_state := (play_round^[6]) initial_state
  game_ended final_state ∧ ¬game_ended ((play_round^[5]) initial_state) :=
sorry

end NUMINAMATH_CALUDE_game_ends_in_six_rounds_l1372_137238


namespace NUMINAMATH_CALUDE_new_average_weight_l1372_137213

/-- Given 6 people with an average weight of 154 lbs and a 7th person weighing 133 lbs,
    prove that the new average weight of all 7 people is 151 lbs. -/
theorem new_average_weight 
  (initial_people : Nat) 
  (initial_avg_weight : ℚ) 
  (new_person_weight : ℚ) : 
  initial_people = 6 → 
  initial_avg_weight = 154 → 
  new_person_weight = 133 → 
  ((initial_people : ℚ) * initial_avg_weight + new_person_weight) / (initial_people + 1) = 151 := by
  sorry

#check new_average_weight

end NUMINAMATH_CALUDE_new_average_weight_l1372_137213


namespace NUMINAMATH_CALUDE_unique_b_value_l1372_137205

theorem unique_b_value (b h a : ℕ) (hb_pos : 0 < b) (hh_pos : 0 < h) (hb_lt_h : b < h)
  (heq : b^2 + h^2 = b*(a + h) + a*h) : b = 2 := by
  sorry

end NUMINAMATH_CALUDE_unique_b_value_l1372_137205


namespace NUMINAMATH_CALUDE_inclination_angle_range_l1372_137215

/-- Given a line with equation x*sin(α) + y + 2 = 0, 
    the range of the inclination angle α is [0, π/4] ∪ [3π/4, π) -/
theorem inclination_angle_range (x y : ℝ) (α : ℝ) :
  (x * Real.sin α + y + 2 = 0) →
  α ∈ Set.Icc 0 (Real.pi / 4) ∪ Set.Ico (3 * Real.pi / 4) Real.pi :=
by sorry

end NUMINAMATH_CALUDE_inclination_angle_range_l1372_137215


namespace NUMINAMATH_CALUDE_max_value_3a_plus_b_l1372_137279

theorem max_value_3a_plus_b (a b : ℝ) :
  (∀ x ∈ Set.Icc 1 2, |a * x^2 + b * x + a| ≤ x) →
  (∃ a₀ b₀ : ℝ, (∀ x ∈ Set.Icc 1 2, |a₀ * x^2 + b₀ * x + a₀| ≤ x) ∧ 3 * a₀ + b₀ = 3) ∧
  (∀ a₁ b₁ : ℝ, (∀ x ∈ Set.Icc 1 2, |a₁ * x^2 + b₁ * x + a₁| ≤ x) → 3 * a₁ + b₁ ≤ 3) :=
by sorry

#check max_value_3a_plus_b

end NUMINAMATH_CALUDE_max_value_3a_plus_b_l1372_137279


namespace NUMINAMATH_CALUDE_paint_usage_correct_l1372_137218

/-- Represents the amount of paint used for a canvas size -/
structure PaintUsage where
  red : ℕ
  blue : ℕ
  yellow : ℕ
  green : ℕ

/-- Calculates the total paint used for a given canvas size and count -/
def totalPaintUsed (usage : PaintUsage) (count : ℕ) : PaintUsage :=
  { red := usage.red * count
  , blue := usage.blue * count
  , yellow := usage.yellow * count
  , green := usage.green * count
  }

/-- Adds two PaintUsage structures -/
def addPaintUsage (a b : PaintUsage) : PaintUsage :=
  { red := a.red + b.red
  , blue := a.blue + b.blue
  , yellow := a.yellow + b.yellow
  , green := a.green + b.green
  }

theorem paint_usage_correct : 
  let extraLarge : PaintUsage := { red := 5, blue := 3, yellow := 2, green := 1 }
  let large : PaintUsage := { red := 4, blue := 2, yellow := 3, green := 1 }
  let medium : PaintUsage := { red := 3, blue := 1, yellow := 2, green := 1 }
  let small : PaintUsage := { red := 1, blue := 1, yellow := 1, green := 1 }
  
  let totalUsage := addPaintUsage
    (addPaintUsage
      (addPaintUsage
        (totalPaintUsed extraLarge 3)
        (totalPaintUsed large 5))
      (totalPaintUsed medium 6))
    (totalPaintUsed small 8)

  totalUsage.red = 61 ∧
  totalUsage.blue = 33 ∧
  totalUsage.yellow = 41 ∧
  totalUsage.green = 22 :=
by sorry


end NUMINAMATH_CALUDE_paint_usage_correct_l1372_137218


namespace NUMINAMATH_CALUDE_initial_fish_l1372_137246

def fish_bought : ℝ := 280.0
def fish_now : ℕ := 492

theorem initial_fish : ℕ := by
  sorry

#check initial_fish = 212

end NUMINAMATH_CALUDE_initial_fish_l1372_137246


namespace NUMINAMATH_CALUDE_binary_representation_of_41_l1372_137281

/-- Converts a natural number to its binary representation as a list of bits -/
def toBinary (n : ℕ) : List ℕ :=
  if n = 0 then [0]
  else
    let rec go (m : ℕ) (acc : List ℕ) : List ℕ :=
      if m = 0 then acc
      else go (m / 2) ((m % 2) :: acc)
    go n []

/-- The binary representation of 41 -/
def binary41 : List ℕ := [1, 0, 1, 0, 0, 1]

/-- Theorem stating that the binary representation of 41 is [1, 0, 1, 0, 0, 1] -/
theorem binary_representation_of_41 : toBinary 41 = binary41 := by
  sorry

end NUMINAMATH_CALUDE_binary_representation_of_41_l1372_137281


namespace NUMINAMATH_CALUDE_andy_remaining_demerits_l1372_137250

/-- The maximum number of demerits allowed in a month before firing -/
def max_demerits : ℕ := 50

/-- The number of demerits per late instance -/
def demerits_per_late : ℕ := 2

/-- The number of times Andy was late -/
def times_late : ℕ := 6

/-- The number of demerits for the inappropriate joke -/
def joke_demerits : ℕ := 15

/-- The number of additional demerits Andy can get before being fired -/
def remaining_demerits : ℕ := max_demerits - (demerits_per_late * times_late + joke_demerits)

theorem andy_remaining_demerits : remaining_demerits = 23 := by
  sorry

end NUMINAMATH_CALUDE_andy_remaining_demerits_l1372_137250


namespace NUMINAMATH_CALUDE_grocery_bill_calculation_l1372_137230

/-- Represents the denomination of a bill or coin in pesos -/
inductive Denomination
| bill : Denomination  -- 20-peso bill
| coin : Denomination  -- 5-peso coin

/-- The value of a given denomination in pesos -/
def value (d : Denomination) : ℕ :=
  match d with
  | .bill => 20
  | .coin => 5

/-- The total number of bills and coins used -/
def total_count : ℕ := 24

/-- The number of each denomination used -/
def count_each : ℕ := 11

/-- Calculates the total amount of the grocery bill in pesos -/
def grocery_bill : ℕ :=
  count_each * value Denomination.bill + count_each * value Denomination.coin

theorem grocery_bill_calculation :
  grocery_bill = 275 ∧
  count_each + count_each = total_count :=
by sorry

end NUMINAMATH_CALUDE_grocery_bill_calculation_l1372_137230


namespace NUMINAMATH_CALUDE_asterisk_replacement_l1372_137256

/-- The expression after substituting 2x for * and expanding -/
def expanded_expression (x : ℝ) : ℝ := x^6 + x^4 + 4*x^2 + 4

/-- The number of terms in the expanded expression -/
def num_terms (x : ℝ) : ℕ := 4

theorem asterisk_replacement :
  ∀ x : ℝ, (x^3 - 2)^2 + (x^2 + 2*x)^2 = expanded_expression x ∧
           num_terms x = 4 :=
by sorry

end NUMINAMATH_CALUDE_asterisk_replacement_l1372_137256


namespace NUMINAMATH_CALUDE_license_plate_increase_l1372_137283

theorem license_plate_increase : 
  let old_plates := 26 * (10 ^ 3)
  let new_plates := (26 ^ 4) * (10 ^ 4)
  (new_plates / old_plates : ℚ) = 175760 := by
sorry

end NUMINAMATH_CALUDE_license_plate_increase_l1372_137283


namespace NUMINAMATH_CALUDE_b_value_l1372_137242

theorem b_value (a b c m : ℝ) (h : m = (c * a * b) / (a + b)) : 
  b = (m * a) / (c * a - m) :=
sorry

end NUMINAMATH_CALUDE_b_value_l1372_137242


namespace NUMINAMATH_CALUDE_sum_of_solutions_quadratic_l1372_137296

theorem sum_of_solutions_quadratic (x : ℝ) : 
  let equation := -48 * x^2 + 108 * x - 27 = 0
  let sum_of_solutions := -108 / (-48)
  sum_of_solutions = 9/4 := by
sorry

end NUMINAMATH_CALUDE_sum_of_solutions_quadratic_l1372_137296


namespace NUMINAMATH_CALUDE_dividend_calculation_l1372_137276

theorem dividend_calculation (divisor quotient remainder : ℕ) 
  (h1 : divisor = 17)
  (h2 : quotient = 9)
  (h3 : remainder = 7) :
  divisor * quotient + remainder = 160 := by
  sorry

end NUMINAMATH_CALUDE_dividend_calculation_l1372_137276


namespace NUMINAMATH_CALUDE_female_puppies_count_l1372_137289

theorem female_puppies_count (total : ℕ) (male : ℕ) (ratio : ℚ) : ℕ :=
  let female := total - male
  have h1 : total = 12 := by sorry
  have h2 : male = 10 := by sorry
  have h3 : ratio = 1/5 := by sorry
  have h4 : (female : ℚ) / male = ratio := by sorry
  2

#check female_puppies_count

end NUMINAMATH_CALUDE_female_puppies_count_l1372_137289


namespace NUMINAMATH_CALUDE_ace_of_hearts_probability_l1372_137200

def standard_deck : ℕ := 52
def jokers : ℕ := 2
def total_cards : ℕ := standard_deck + jokers
def ace_of_hearts : ℕ := 1

theorem ace_of_hearts_probability :
  (ace_of_hearts : ℚ) / total_cards = 1 / 54 :=
sorry

end NUMINAMATH_CALUDE_ace_of_hearts_probability_l1372_137200


namespace NUMINAMATH_CALUDE_exponent_difference_l1372_137228

theorem exponent_difference (a m n : ℝ) (h1 : a^m = 12) (h2 : a^n = 3) : a^(m-n) = 4 := by
  sorry

end NUMINAMATH_CALUDE_exponent_difference_l1372_137228


namespace NUMINAMATH_CALUDE_max_value_of_s_l1372_137254

-- Define the function s
def s (x y : ℝ) : ℝ := x + y

-- State the theorem
theorem max_value_of_s :
  ∃ (M : ℝ), M = 9 ∧ ∀ (x y : ℝ), s x y ≤ M :=
sorry

end NUMINAMATH_CALUDE_max_value_of_s_l1372_137254


namespace NUMINAMATH_CALUDE_distance_between_points_l1372_137226

theorem distance_between_points : 
  let p1 : ℝ × ℝ := (1, 3)
  let p2 : ℝ × ℝ := (6, 7)
  Real.sqrt ((p2.1 - p1.1)^2 + (p2.2 - p1.2)^2) = Real.sqrt 41 := by
  sorry

end NUMINAMATH_CALUDE_distance_between_points_l1372_137226


namespace NUMINAMATH_CALUDE_smallest_n_square_cube_l1372_137247

theorem smallest_n_square_cube : ∃ (n : ℕ), n > 0 ∧ 
  (∃ (k : ℕ), 4 * n = k^2) ∧ 
  (∃ (m : ℕ), 5 * n = m^3) ∧ 
  (∀ (x : ℕ), x > 0 → 
    (∃ (y : ℕ), 4 * x = y^2) → 
    (∃ (z : ℕ), 5 * x = z^3) → 
    n ≤ x) ∧
  n = 25 :=
sorry

end NUMINAMATH_CALUDE_smallest_n_square_cube_l1372_137247


namespace NUMINAMATH_CALUDE_pizza_ingredients_calculation_l1372_137260

/-- Pizza ingredients calculation -/
theorem pizza_ingredients_calculation 
  (water : ℕ) 
  (flour : ℕ) 
  (salt : ℚ) 
  (h1 : water = 10)
  (h2 : flour = 16)
  (h3 : salt = (1/2 : ℚ) * flour) :
  (water + flour : ℕ) = 26 ∧ salt = 8 := by
  sorry

end NUMINAMATH_CALUDE_pizza_ingredients_calculation_l1372_137260


namespace NUMINAMATH_CALUDE_train_bridge_crossing_time_l1372_137231

/-- The time taken for a train to cross a bridge -/
theorem train_bridge_crossing_time (train_length : ℝ) (train_speed_kmh : ℝ) (bridge_length : ℝ)
  (h1 : train_length = 110)
  (h2 : train_speed_kmh = 36)
  (h3 : bridge_length = 132) :
  (train_length + bridge_length) / (train_speed_kmh * 1000 / 3600) = 24.2 := by
  sorry

end NUMINAMATH_CALUDE_train_bridge_crossing_time_l1372_137231

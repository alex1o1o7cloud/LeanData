import Mathlib

namespace NUMINAMATH_CALUDE_parallel_line_through_point_l1602_160273

/-- Given a line with slope m and a point P(x1, y1), prove that the line
    y = mx + (y1 - mx1) passes through P and is parallel to the original line. -/
theorem parallel_line_through_point (m x1 y1 : ℝ) :
  let L2 : ℝ → ℝ := λ x => m * x + (y1 - m * x1)
  (L2 x1 = y1) ∧ (∀ x y, y = L2 x ↔ y - y1 = m * (x - x1)) := by sorry

end NUMINAMATH_CALUDE_parallel_line_through_point_l1602_160273


namespace NUMINAMATH_CALUDE_f2_form_l1602_160290

/-- A quadratic function with coefficients a, b, and c. -/
def f (a b c : ℝ) (x : ℝ) : ℝ := a * x^2 + b * x + c

/-- The reflection of a function about the y-axis. -/
def reflect_y (g : ℝ → ℝ) : ℝ → ℝ := λ x ↦ g (-x)

/-- The reflection of a function about the line y = 1. -/
def reflect_y_eq_1 (g : ℝ → ℝ) : ℝ → ℝ := λ x ↦ 2 - g x

/-- Theorem stating the form of f2 after two reflections of f. -/
theorem f2_form (a b c : ℝ) (ha : a ≠ 0) :
  let f1 := reflect_y (f a b c)
  let f2 := reflect_y_eq_1 f1
  ∀ x, f2 x = -a * x^2 + b * x + (2 - c) :=
sorry

end NUMINAMATH_CALUDE_f2_form_l1602_160290


namespace NUMINAMATH_CALUDE_rent_split_l1602_160248

theorem rent_split (total_rent : ℕ) (num_people : ℕ) (individual_rent : ℕ) :
  total_rent = 490 →
  num_people = 7 →
  individual_rent = total_rent / num_people →
  individual_rent = 70 := by
  sorry

end NUMINAMATH_CALUDE_rent_split_l1602_160248


namespace NUMINAMATH_CALUDE_inverse_g_sum_l1602_160232

noncomputable def g (x : ℝ) : ℝ :=
  if x ≤ 2 then 3 - x else 3*x - x^2

theorem inverse_g_sum : 
  ∃ (a b c : ℝ), g a = -2 ∧ g b = 0 ∧ g c = 4 ∧ a + b + c = 6 :=
by sorry

end NUMINAMATH_CALUDE_inverse_g_sum_l1602_160232


namespace NUMINAMATH_CALUDE_trapezoid_segment_length_l1602_160224

/-- Represents a trapezoid with the given properties -/
structure Trapezoid where
  shorter_base : ℝ
  longer_base : ℝ
  height : ℝ
  midline_ratio : ℝ
  equal_area_segment : ℝ
  base_difference : ℝ
  area_ratio : ℚ

/-- The properties of the trapezoid as described in the problem -/
axiom trapezoid_properties (t : Trapezoid) :
  t.longer_base = t.shorter_base + t.base_difference ∧
  t.base_difference = 150 ∧
  t.midline_ratio = (t.shorter_base + t.longer_base) / 2 ∧
  t.area_ratio = 3 / 2 ∧
  (t.midline_ratio - t.shorter_base) / (t.longer_base - t.midline_ratio) = t.area_ratio

/-- The theorem to be proved -/
theorem trapezoid_segment_length (t : Trapezoid) :
  ⌊(t.equal_area_segment ^ 2) / 150⌋ = 550 :=
sorry

end NUMINAMATH_CALUDE_trapezoid_segment_length_l1602_160224


namespace NUMINAMATH_CALUDE_algebraic_expression_value_l1602_160256

theorem algebraic_expression_value (x y : ℝ) 
  (h1 : x - y = 5) 
  (h2 : x * y = -3) : 
  x^2 * y - x * y^2 = -15 := by
  sorry

end NUMINAMATH_CALUDE_algebraic_expression_value_l1602_160256


namespace NUMINAMATH_CALUDE_evaluate_expression_l1602_160277

theorem evaluate_expression : 3 * (-3)^4 + 3 * (-3)^3 + 3 * (-3)^2 + 3 * 3^2 + 3 * 3^3 + 3 * 3^4 = 540 := by
  sorry

end NUMINAMATH_CALUDE_evaluate_expression_l1602_160277


namespace NUMINAMATH_CALUDE_vector_parallel_to_a_l1602_160267

/-- Given a vector a = (-5, 4), prove that (-5k, 4k) is parallel to a for any scalar k. -/
theorem vector_parallel_to_a (k : ℝ) : 
  ∃ (t : ℝ), ((-5 : ℝ), (4 : ℝ)) = t • ((-5*k : ℝ), (4*k : ℝ)) := by
  sorry

end NUMINAMATH_CALUDE_vector_parallel_to_a_l1602_160267


namespace NUMINAMATH_CALUDE_triangle_perimeter_from_average_side_length_l1602_160212

/-- Given a triangle with three sides where the average length of the sides is 12,
    prove that the perimeter of the triangle is 36. -/
theorem triangle_perimeter_from_average_side_length :
  ∀ (a b c : ℝ), (a + b + c) / 3 = 12 → a + b + c = 36 := by
  sorry

end NUMINAMATH_CALUDE_triangle_perimeter_from_average_side_length_l1602_160212


namespace NUMINAMATH_CALUDE_pizza_slices_l1602_160257

theorem pizza_slices (total_pizzas : ℕ) (total_slices : ℕ) (h1 : total_pizzas = 21) (h2 : total_slices = 168) :
  total_slices / total_pizzas = 8 := by
sorry

end NUMINAMATH_CALUDE_pizza_slices_l1602_160257


namespace NUMINAMATH_CALUDE_inequality_proof_l1602_160252

theorem inequality_proof (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  (a + b + c) * (a^2 + b^2 + c^2) ≥ 9 * a * b * c := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l1602_160252


namespace NUMINAMATH_CALUDE_yellow_marbles_count_l1602_160260

theorem yellow_marbles_count (total : ℕ) (red : ℕ) (blue : ℕ) (yellow : ℕ) : 
  total = 85 →
  blue = 3 * red →
  red = 14 →
  total = red + blue + yellow →
  yellow = 29 := by
sorry

end NUMINAMATH_CALUDE_yellow_marbles_count_l1602_160260


namespace NUMINAMATH_CALUDE_percentage_needed_is_35_l1602_160274

/-- The percentage of total marks needed to pass, given Pradeep's score, 
    the marks he fell short by, and the maximum marks. -/
def percentage_to_pass (pradeep_score : ℕ) (marks_short : ℕ) (max_marks : ℕ) : ℚ :=
  ((pradeep_score + marks_short : ℚ) / max_marks) * 100

/-- Theorem stating that the percentage needed to pass is 35% -/
theorem percentage_needed_is_35 (pradeep_score marks_short max_marks : ℕ) 
  (h1 : pradeep_score = 185)
  (h2 : marks_short = 25)
  (h3 : max_marks = 600) :
  percentage_to_pass pradeep_score marks_short max_marks = 35 := by
  sorry

#eval percentage_to_pass 185 25 600

end NUMINAMATH_CALUDE_percentage_needed_is_35_l1602_160274


namespace NUMINAMATH_CALUDE_exists_line_with_perpendicular_chord_l1602_160209

-- Define the ellipse C
def C (x y : ℝ) : Prop := x^2 + 2*y^2 = 8

-- Define the line l
def l (x y m : ℝ) : Prop := y = x + m

-- Define the condition for A and B being on the ellipse C and line l
def on_ellipse_and_line (x₁ y₁ x₂ y₂ m : ℝ) : Prop :=
  C x₁ y₁ ∧ C x₂ y₂ ∧ l x₁ y₁ m ∧ l x₂ y₂ m

-- Define the condition for AB being perpendicular to OA and OB
def perpendicular_chord (x₁ y₁ x₂ y₂ : ℝ) : Prop :=
  x₁ * x₂ + y₁ * y₂ = 0

-- Main theorem
theorem exists_line_with_perpendicular_chord :
  ∃ m : ℝ, m = 4 * Real.sqrt 3 / 3 ∨ m = -4 * Real.sqrt 3 / 3 ∧
  ∃ x₁ y₁ x₂ y₂ : ℝ, 
    on_ellipse_and_line x₁ y₁ x₂ y₂ m ∧
    perpendicular_chord x₁ y₁ x₂ y₂ :=
  sorry

end NUMINAMATH_CALUDE_exists_line_with_perpendicular_chord_l1602_160209


namespace NUMINAMATH_CALUDE_lines_cannot_form_triangle_l1602_160254

-- Define the lines
def l₁ (x y : ℝ) : Prop := 4 * x + y = 4
def l₂ (m x y : ℝ) : Prop := m * x + y = 0
def l₃ (m x y : ℝ) : Prop := 2 * x - 3 * m * y = 4

-- Define when lines are parallel
def parallel (m₁ m₂ : ℝ) : Prop := m₁ = m₂

-- Define when three lines intersect at a single point
def intersect_at_point (m : ℝ) : Prop :=
  ∃ x y : ℝ, l₁ x y ∧ l₂ m x y ∧ l₃ m x y

-- Theorem statement
theorem lines_cannot_form_triangle (m : ℝ) : 
  (¬∃ x₁ y₁ x₂ y₂ x₃ y₃ : ℝ, 
    l₁ x₁ y₁ ∧ l₂ m x₂ y₂ ∧ l₃ m x₃ y₃ ∧ 
    (x₁ ≠ x₂ ∨ y₁ ≠ y₂) ∧ (x₂ ≠ x₃ ∨ y₂ ≠ y₃) ∧ (x₃ ≠ x₁ ∨ y₃ ≠ y₁)) ↔ 
  (m = 4 ∨ m = -1/6 ∨ m = -1 ∨ m = 2/3) :=
sorry

end NUMINAMATH_CALUDE_lines_cannot_form_triangle_l1602_160254


namespace NUMINAMATH_CALUDE_sum_of_digits_9ab_l1602_160242

/-- The sum of digits of a natural number in base 10 -/
def sumOfDigits (n : ℕ) : ℕ := sorry

/-- A number consisting of n repetitions of a digit d in base 10 -/
def repeatedDigit (d : ℕ) (n : ℕ) : ℕ := sorry

theorem sum_of_digits_9ab : 
  let a := repeatedDigit 6 2023
  let b := repeatedDigit 4 2023
  sumOfDigits (9 * a * b) = 20225 := by sorry

end NUMINAMATH_CALUDE_sum_of_digits_9ab_l1602_160242


namespace NUMINAMATH_CALUDE_equilateral_triangle_area_l1602_160283

/-- For an equilateral triangle where the square of each side's length
    is equal to the perimeter, the area of the triangle is 9√3/4 square units. -/
theorem equilateral_triangle_area (s : ℝ) (h1 : s > 0) (h2 : s^2 = 3*s) :
  (s^2 * Real.sqrt 3) / 4 = 9 * Real.sqrt 3 / 4 := by
  sorry

end NUMINAMATH_CALUDE_equilateral_triangle_area_l1602_160283


namespace NUMINAMATH_CALUDE_binomial_coefficient_divisible_by_prime_l1602_160278

theorem binomial_coefficient_divisible_by_prime (p k : ℕ) : 
  Prime p → 0 < k → k < p → ∃ m : ℕ, Nat.choose p k = p * m := by
  sorry

end NUMINAMATH_CALUDE_binomial_coefficient_divisible_by_prime_l1602_160278


namespace NUMINAMATH_CALUDE_goldfish_count_correct_l1602_160230

/-- The number of Goldfish Layla has -/
def num_goldfish : ℕ := 2

/-- The total amount of food Layla gives to all her fish -/
def total_food : ℕ := 12

/-- The number of Swordtails Layla has -/
def num_swordtails : ℕ := 3

/-- The amount of food each Swordtail gets -/
def food_per_swordtail : ℕ := 2

/-- The number of Guppies Layla has -/
def num_guppies : ℕ := 8

/-- The amount of food each Guppy gets -/
def food_per_guppy : ℚ := 1/2

/-- The amount of food each Goldfish gets -/
def food_per_goldfish : ℕ := 1

/-- Theorem stating that the number of Goldfish is correct given the conditions -/
theorem goldfish_count_correct : 
  total_food = num_swordtails * food_per_swordtail + 
               num_guppies * food_per_guppy + 
               num_goldfish * food_per_goldfish :=
by sorry

end NUMINAMATH_CALUDE_goldfish_count_correct_l1602_160230


namespace NUMINAMATH_CALUDE_right_triangle_from_sine_condition_l1602_160228

theorem right_triangle_from_sine_condition (A B C : ℝ) :
  (0 < A) → (A < π) →
  (0 < B) → (B < π) →
  (0 < C) → (C < π) →
  A + B + C = π →
  (Real.sin A + Real.sin B) * (Real.sin A - Real.sin B) = (Real.sin C)^2 →
  ∃ (a b c : ℝ), a > 0 ∧ b > 0 ∧ c > 0 ∧ a^2 + b^2 = c^2 :=
by sorry

end NUMINAMATH_CALUDE_right_triangle_from_sine_condition_l1602_160228


namespace NUMINAMATH_CALUDE_dot_product_is_2020_l1602_160215

/-- A trapezoid with perpendicular diagonals -/
structure PerpendicularDiagonalTrapezoid where
  -- Points of the trapezoid
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ
  D : ℝ × ℝ
  -- AB is a base of length 101
  AB_length : dist A B = 101
  -- CD is a base of length 20
  CD_length : dist C D = 20
  -- ABCD is a trapezoid (parallel sides)
  is_trapezoid : (B.1 - A.1) * (D.2 - C.2) = (D.1 - C.1) * (B.2 - A.2)
  -- Diagonals are perpendicular
  diagonals_perpendicular : (C.1 - A.1) * (D.1 - B.1) + (C.2 - A.2) * (D.2 - B.2) = 0

/-- The dot product of vectors AD and BC in a trapezoid with perpendicular diagonals -/
def dot_product_AD_BC (t : PerpendicularDiagonalTrapezoid) : ℝ :=
  let AD := (t.D.1 - t.A.1, t.D.2 - t.A.2)
  let BC := (t.C.1 - t.B.1, t.C.2 - t.B.2)
  AD.1 * BC.1 + AD.2 * BC.2

/-- Theorem: The dot product of AD and BC is 2020 -/
theorem dot_product_is_2020 (t : PerpendicularDiagonalTrapezoid) :
  dot_product_AD_BC t = 2020 := by
  sorry

end NUMINAMATH_CALUDE_dot_product_is_2020_l1602_160215


namespace NUMINAMATH_CALUDE_slope_of_line_l1602_160204

/-- The slope of a line passing through two points is 1 -/
theorem slope_of_line (M N : ℝ × ℝ) (h1 : M = (-Real.sqrt 3, Real.sqrt 2)) 
  (h2 : N = (-Real.sqrt 2, Real.sqrt 3)) : 
  (N.2 - M.2) / (N.1 - M.1) = 1 := by
  sorry

end NUMINAMATH_CALUDE_slope_of_line_l1602_160204


namespace NUMINAMATH_CALUDE_square_field_area_l1602_160200

theorem square_field_area (side : ℝ) (h1 : 4 * side = 36) 
  (h2 : 6 * (side * side) = 6 * (2 * (4 * side) + 9)) : side * side = 81 :=
by
  sorry

end NUMINAMATH_CALUDE_square_field_area_l1602_160200


namespace NUMINAMATH_CALUDE_count_valid_pairs_l1602_160246

def has_one_solution (b c : ℕ) : Prop :=
  b^2 = 4*c ∨ c^2 = 4*b

def valid_pair (b c : ℕ) : Prop :=
  1 ≤ b ∧ b ≤ 6 ∧ 1 ≤ c ∧ c ≤ 6 ∧ has_one_solution b c

theorem count_valid_pairs :
  ∃ (S : Finset (ℕ × ℕ)), (∀ (p : ℕ × ℕ), p ∈ S ↔ valid_pair p.1 p.2) ∧ Finset.card S = 3 :=
sorry

end NUMINAMATH_CALUDE_count_valid_pairs_l1602_160246


namespace NUMINAMATH_CALUDE_product_terminal_zeros_l1602_160288

/-- The number of terminal zeros in a natural number -/
def terminalZeros (n : ℕ) : ℕ := sorry

/-- The product of 75 and 480 -/
def product : ℕ := 75 * 480

/-- Theorem: The number of terminal zeros in the product of 75 and 480 is 3 -/
theorem product_terminal_zeros : terminalZeros product = 3 := by sorry

end NUMINAMATH_CALUDE_product_terminal_zeros_l1602_160288


namespace NUMINAMATH_CALUDE_chord_length_l1602_160205

-- Define the curve C in polar coordinates
def curve_C (ρ θ : ℝ) : Prop := ρ - ρ * Real.cos (2 * θ) - 12 * Real.cos θ = 0

-- Define the line l in parametric form
def line_l (t x y : ℝ) : Prop := x = -4/5 * t + 2 ∧ y = 3/5 * t

-- Define the curve C in rectangular coordinates
def curve_C_rect (x y : ℝ) : Prop := y^2 = 6 * x

-- Define the line l in normal form
def line_l_normal (x y : ℝ) : Prop := 3 * x + 4 * y - 6 = 0

-- Theorem statement
theorem chord_length :
  ∀ (x₁ y₁ x₂ y₂ : ℝ),
  curve_C_rect x₁ y₁ ∧ curve_C_rect x₂ y₂ ∧
  line_l_normal x₁ y₁ ∧ line_l_normal x₂ y₂ ∧
  x₁ ≠ x₂ →
  Real.sqrt ((x₁ - x₂)^2 + (y₁ - y₂)^2) = (20 * Real.sqrt 7) / 3 :=
sorry

end NUMINAMATH_CALUDE_chord_length_l1602_160205


namespace NUMINAMATH_CALUDE_absolute_value_inequality_l1602_160291

theorem absolute_value_inequality (x : ℝ) :
  abs (x - 2) + abs (x + 3) < 8 ↔ -9/2 < x ∧ x < 7/2 := by
  sorry

end NUMINAMATH_CALUDE_absolute_value_inequality_l1602_160291


namespace NUMINAMATH_CALUDE_optimal_solution_l1602_160263

/-- Represents the factory worker allocation problem -/
structure FactoryProblem where
  total_workers : ℕ
  salary_a : ℕ
  salary_b : ℕ
  job_b_constraint : ℕ → Prop

/-- The specific factory problem instance -/
def factory_instance : FactoryProblem where
  total_workers := 120
  salary_a := 800
  salary_b := 1000
  job_b_constraint := fun x => (120 - x) ≥ 3 * x

/-- Calculate the total monthly salary -/
def total_salary (p : FactoryProblem) (workers_a : ℕ) : ℕ :=
  p.salary_a * workers_a + p.salary_b * (p.total_workers - workers_a)

/-- Theorem stating the optimal solution -/
theorem optimal_solution (p : FactoryProblem) :
  p = factory_instance →
  (∀ x : ℕ, x ≤ p.total_workers → p.job_b_constraint x → total_salary p x ≥ 114000) ∧
  total_salary p 30 = 114000 ∧
  p.job_b_constraint 30 := by
  sorry

end NUMINAMATH_CALUDE_optimal_solution_l1602_160263


namespace NUMINAMATH_CALUDE_tom_has_nine_balloons_l1602_160270

/-- The number of yellow balloons Sara has -/
def sara_balloons : ℕ := 8

/-- The total number of yellow balloons Tom and Sara have together -/
def total_balloons : ℕ := 17

/-- The number of yellow balloons Tom has -/
def tom_balloons : ℕ := total_balloons - sara_balloons

theorem tom_has_nine_balloons : tom_balloons = 9 := by
  sorry

end NUMINAMATH_CALUDE_tom_has_nine_balloons_l1602_160270


namespace NUMINAMATH_CALUDE_square_sum_given_sum_and_product_l1602_160284

theorem square_sum_given_sum_and_product (x y : ℝ) 
  (h1 : (x + y)^2 = 16) 
  (h2 : x * y = -8) : 
  x^2 + y^2 = 32 := by
sorry

end NUMINAMATH_CALUDE_square_sum_given_sum_and_product_l1602_160284


namespace NUMINAMATH_CALUDE_rays_grocery_bill_l1602_160216

/-- Calculates the total grocery bill for Ray's purchase with a store rewards discount --/
theorem rays_grocery_bill :
  let meat_price : ℚ := 5
  let crackers_price : ℚ := 3.5
  let vegetable_price : ℚ := 2
  let vegetable_quantity : ℕ := 4
  let cheese_price : ℚ := 3.5
  let discount_rate : ℚ := 0.1

  let total_before_discount : ℚ := 
    meat_price + crackers_price + (vegetable_price * vegetable_quantity) + cheese_price
  
  let discount_amount : ℚ := total_before_discount * discount_rate
  
  let final_bill : ℚ := total_before_discount - discount_amount

  final_bill = 18 := by sorry

end NUMINAMATH_CALUDE_rays_grocery_bill_l1602_160216


namespace NUMINAMATH_CALUDE_normal_distribution_probability_l1602_160227

/-- A random variable following a normal distribution with mean 2 and standard deviation 1. -/
def ξ : Real → Real := sorry

/-- The probability density function of the standard normal distribution. -/
noncomputable def φ : Real → Real := sorry

/-- The cumulative distribution function of the standard normal distribution. -/
noncomputable def Φ : Real → Real := sorry

/-- The probability that ξ is greater than 3. -/
def P_gt_3 : Real := 0.023

theorem normal_distribution_probability (h : P_gt_3 = 1 - Φ 1) : 
  Φ 1 - Φ (-1) = 0.954 := by sorry

end NUMINAMATH_CALUDE_normal_distribution_probability_l1602_160227


namespace NUMINAMATH_CALUDE_pitcher_juice_distribution_l1602_160237

theorem pitcher_juice_distribution :
  ∀ (C : ℝ),
  C > 0 →
  let pineapple_juice := (1/2 : ℝ) * C
  let orange_juice := (1/4 : ℝ) * C
  let total_juice := pineapple_juice + orange_juice
  let cups := 4
  let juice_per_cup := total_juice / cups
  (juice_per_cup / C) * 100 = 18.75 :=
by
  sorry

end NUMINAMATH_CALUDE_pitcher_juice_distribution_l1602_160237


namespace NUMINAMATH_CALUDE_fraction_inequality_l1602_160299

theorem fraction_inequality (a b c d : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (hd : 0 < d) :
  1 / (1 / a + 1 / b) + 1 / (1 / c + 1 / d) ≤ 1 / (1 / (a + c) + 1 / (b + d)) := by
  sorry

end NUMINAMATH_CALUDE_fraction_inequality_l1602_160299


namespace NUMINAMATH_CALUDE_odd_product_pattern_l1602_160255

theorem odd_product_pattern (n : ℕ) (h : Odd n) : n * (n + 2) = (n + 1)^2 - 1 := by
  sorry

end NUMINAMATH_CALUDE_odd_product_pattern_l1602_160255


namespace NUMINAMATH_CALUDE_max_value_log_product_l1602_160259

/-- The maximum value of lg a · lg c given the conditions -/
theorem max_value_log_product (a b c : ℝ) 
  (ha : a > 1) (hb : b > 1) (hc : c > 1)
  (eq1 : Real.log a / Real.log 10 + Real.log c / Real.log b = 3)
  (eq2 : Real.log b / Real.log 10 + Real.log c / Real.log a = 4) :
  (∃ (x : ℝ), (Real.log a / Real.log 10) * (Real.log c / Real.log 10) ≤ x) ∧
  (∀ (y : ℝ), (Real.log a / Real.log 10) * (Real.log c / Real.log 10) ≤ y → 16/3 ≤ y) :=
sorry

end NUMINAMATH_CALUDE_max_value_log_product_l1602_160259


namespace NUMINAMATH_CALUDE_find_set_A_l1602_160296

def U : Set ℕ := {1,2,3,4,5,6,7,8}

theorem find_set_A (A B : Set ℕ)
  (h1 : A ∩ (U \ B) = {1,8})
  (h2 : (U \ A) ∩ B = {2,6})
  (h3 : (U \ A) ∩ (U \ B) = {4,7}) :
  A = {1,3,5,8} := by
  sorry

end NUMINAMATH_CALUDE_find_set_A_l1602_160296


namespace NUMINAMATH_CALUDE_distribution_combinations_l1602_160244

/-- The number of ways to distribute 2 objects among 4 categories -/
def distributionCount : ℕ := 10

/-- The number of categories -/
def categoryCount : ℕ := 4

/-- The number of objects to distribute -/
def objectCount : ℕ := 2

theorem distribution_combinations :
  (categoryCount : ℕ) + (categoryCount * (categoryCount - 1) / 2) = distributionCount :=
sorry

end NUMINAMATH_CALUDE_distribution_combinations_l1602_160244


namespace NUMINAMATH_CALUDE_unreachable_zero_l1602_160206

/-- Represents a point in 2D space -/
structure Point where
  x : ℤ
  y : ℤ

/-- The set of possible moves -/
inductive Move where
  | swap : Move
  | scale : Move
  | negate : Move
  | increment : Move
  | decrement : Move

/-- Apply a move to a point -/
def applyMove (p : Point) (m : Move) : Point :=
  match m with
  | Move.swap => ⟨p.y, p.x⟩
  | Move.scale => ⟨3 * p.x, -2 * p.y⟩
  | Move.negate => ⟨-2 * p.x, 3 * p.y⟩
  | Move.increment => ⟨p.x + 1, p.y + 4⟩
  | Move.decrement => ⟨p.x - 1, p.y - 4⟩

/-- The sum of coordinates modulo 5 -/
def sumMod5 (p : Point) : ℤ :=
  (p.x + p.y) % 5

/-- Theorem: It's impossible to reach (0, 0) from (0, 1) using the given moves -/
theorem unreachable_zero : 
  ∀ (moves : List Move), 
    let finalPoint := moves.foldl applyMove ⟨0, 1⟩
    sumMod5 finalPoint ≠ 0 := by
  sorry


end NUMINAMATH_CALUDE_unreachable_zero_l1602_160206


namespace NUMINAMATH_CALUDE_smaller_circle_radius_l1602_160282

theorem smaller_circle_radius (r_large : ℝ) (A₁ A₂ : ℝ) : 
  r_large = 4 →
  A₁ + A₂ = π * (2 * r_large)^2 →
  2 * A₂ = A₁ + (A₁ + A₂) →
  A₁ = π * r_small^2 →
  r_small = 4 := by
  sorry

end NUMINAMATH_CALUDE_smaller_circle_radius_l1602_160282


namespace NUMINAMATH_CALUDE_bobs_remaining_funds_l1602_160219

/-- Converts a number from octal to decimal --/
def octal_to_decimal (n : ℕ) : ℕ := sorry

/-- Calculates the remaining funds after expenses --/
def remaining_funds (savings : ℕ) (ticket_cost : ℕ) (meal_cost : ℕ) : ℕ :=
  savings - (ticket_cost + meal_cost)

theorem bobs_remaining_funds :
  let bobs_savings : ℕ := octal_to_decimal 7777
  let ticket_cost : ℕ := 1500
  let meal_cost : ℕ := 250
  remaining_funds bobs_savings ticket_cost meal_cost = 2345 := by sorry

end NUMINAMATH_CALUDE_bobs_remaining_funds_l1602_160219


namespace NUMINAMATH_CALUDE_david_homework_hours_l1602_160287

/-- Calculates the weekly homework hours for a course -/
def weekly_homework_hours (total_weeks : ℕ) (class_hours_per_week : ℕ) (total_course_hours : ℕ) : ℕ :=
  (total_course_hours - (total_weeks * class_hours_per_week)) / total_weeks

theorem david_homework_hours :
  let total_weeks : ℕ := 24
  let three_hour_classes : ℕ := 2
  let four_hour_classes : ℕ := 1
  let class_hours_per_week : ℕ := three_hour_classes * 3 + four_hour_classes * 4
  let total_course_hours : ℕ := 336
  weekly_homework_hours total_weeks class_hours_per_week total_course_hours = 4 := by
  sorry

end NUMINAMATH_CALUDE_david_homework_hours_l1602_160287


namespace NUMINAMATH_CALUDE_tom_candy_pieces_l1602_160218

theorem tom_candy_pieces (initial_boxes : ℕ) (given_away : ℕ) (pieces_per_box : ℕ) : 
  initial_boxes = 14 → given_away = 8 → pieces_per_box = 3 →
  (initial_boxes - given_away) * pieces_per_box = 18 := by
sorry

end NUMINAMATH_CALUDE_tom_candy_pieces_l1602_160218


namespace NUMINAMATH_CALUDE_average_difference_l1602_160258

theorem average_difference (a b c : ℝ) 
  (h1 : (a + b) / 2 = 115)
  (h2 : (b + c) / 2 = 160) :
  a - c = -90 := by
sorry

end NUMINAMATH_CALUDE_average_difference_l1602_160258


namespace NUMINAMATH_CALUDE_min_value_of_x_l1602_160220

theorem min_value_of_x (x : ℝ) (h1 : x > 0) (h2 : Real.log x ≥ Real.log 2 + (1/2) * Real.log x) : x ≥ 4 := by
  sorry

end NUMINAMATH_CALUDE_min_value_of_x_l1602_160220


namespace NUMINAMATH_CALUDE_chef_potato_problem_l1602_160213

theorem chef_potato_problem (cooked : ℕ) (cook_time : ℕ) (remaining_time : ℕ) : 
  cooked = 7 → 
  cook_time = 5 → 
  remaining_time = 45 → 
  cooked + remaining_time / cook_time = 16 := by
sorry

end NUMINAMATH_CALUDE_chef_potato_problem_l1602_160213


namespace NUMINAMATH_CALUDE_smallest_square_side_l1602_160285

/-- Represents a square with integer side length -/
structure Square where
  side : ℕ

/-- Represents a partition of a square into smaller squares -/
structure Partition where
  total : ℕ
  unit_squares : ℕ
  other_squares : List ℕ

/-- Checks if a partition is valid for a given square -/
def is_valid_partition (s : Square) (p : Partition) : Prop :=
  p.total = 15 ∧
  p.unit_squares = 12 ∧
  p.other_squares.length = 3 ∧
  (p.unit_squares + p.other_squares.sum) = s.side * s.side ∧
  ∀ x ∈ p.other_squares, x > 0

/-- The theorem stating the smallest possible square side length -/
theorem smallest_square_side : 
  ∃ (s : Square) (p : Partition), 
    is_valid_partition s p ∧ 
    (∀ (s' : Square) (p' : Partition), is_valid_partition s' p' → s.side ≤ s'.side) ∧
    s.side = 5 := by
  sorry

end NUMINAMATH_CALUDE_smallest_square_side_l1602_160285


namespace NUMINAMATH_CALUDE_geometric_mean_of_45_and_80_l1602_160203

theorem geometric_mean_of_45_and_80 : 
  ∃ x : ℝ, (x ^ 2 = 45 * 80) ∧ (x = 60 ∨ x = -60) := by
  sorry

end NUMINAMATH_CALUDE_geometric_mean_of_45_and_80_l1602_160203


namespace NUMINAMATH_CALUDE_sqrt_product_sqrt_l1602_160234

theorem sqrt_product_sqrt : Real.sqrt (49 * Real.sqrt 25) = 7 * Real.sqrt 5 := by sorry

end NUMINAMATH_CALUDE_sqrt_product_sqrt_l1602_160234


namespace NUMINAMATH_CALUDE_storm_damage_conversion_l1602_160261

/-- Converts Canadian dollars to American dollars given exchange rates -/
def storm_damage_in_usd (damage_cad : ℝ) (cad_to_eur : ℝ) (eur_to_usd : ℝ) : ℝ :=
  damage_cad * cad_to_eur * eur_to_usd

/-- Theorem: The storm damage in USD is 40.5 million given the conditions -/
theorem storm_damage_conversion :
  storm_damage_in_usd 45000000 0.75 1.2 = 40500000 := by
  sorry

end NUMINAMATH_CALUDE_storm_damage_conversion_l1602_160261


namespace NUMINAMATH_CALUDE_salon_average_customers_l1602_160222

def customers_per_day : List ℕ := [10, 12, 15, 13, 18, 16, 11]

def days_per_week : ℕ := 7

def average_daily_customers : ℚ :=
  (customers_per_day.sum : ℚ) / days_per_week

theorem salon_average_customers :
  average_daily_customers = 13.57 := by
  sorry

end NUMINAMATH_CALUDE_salon_average_customers_l1602_160222


namespace NUMINAMATH_CALUDE_inverse_not_in_M_log_in_M_iff_exp_plus_square_in_M_l1602_160207

-- Define the set M
def M : Set (ℝ → ℝ) :=
  {f | ∃ x₀ : ℝ, f (x₀ + 1) = f x₀ + f 1}

-- Theorem 1
theorem inverse_not_in_M :
  (fun x => 1 / x) ∉ M := sorry

-- Theorem 2
theorem log_in_M_iff (a : ℝ) :
  (fun x => Real.log (a / (x^2 + 1))) ∈ M ↔ 
  3 - Real.sqrt 5 ≤ a ∧ a ≤ 3 + Real.sqrt 5 := sorry

-- Theorem 3
theorem exp_plus_square_in_M :
  (fun x => 2^x + x^2) ∈ M := sorry

end NUMINAMATH_CALUDE_inverse_not_in_M_log_in_M_iff_exp_plus_square_in_M_l1602_160207


namespace NUMINAMATH_CALUDE_p_sufficient_not_necessary_for_not_q_l1602_160202

/-- Given conditions p and q, prove that p is a sufficient but not necessary condition for ¬q -/
theorem p_sufficient_not_necessary_for_not_q :
  ∀ x : ℝ,
  (0 < x ∧ x ≤ 1) →  -- condition p
  ((1 / x < 1) → False) →  -- ¬q
  ∃ y : ℝ, ((1 / y < 1) → False) ∧ ¬(0 < y ∧ y ≤ 1) :=
by sorry


end NUMINAMATH_CALUDE_p_sufficient_not_necessary_for_not_q_l1602_160202


namespace NUMINAMATH_CALUDE_molecular_weight_C4H10_is_58_12_l1602_160226

/-- The atomic weight of carbon in atomic mass units (amu) -/
def carbon_weight : ℝ := 12.01

/-- The atomic weight of hydrogen in atomic mass units (amu) -/
def hydrogen_weight : ℝ := 1.008

/-- The number of carbon atoms in C4H10 -/
def carbon_count : ℕ := 4

/-- The number of hydrogen atoms in C4H10 -/
def hydrogen_count : ℕ := 10

/-- The molecular weight of C4H10 in atomic mass units (amu) -/
def molecular_weight_C4H10 : ℝ := carbon_weight * carbon_count + hydrogen_weight * hydrogen_count

/-- Theorem stating that the molecular weight of C4H10 is 58.12 amu -/
theorem molecular_weight_C4H10_is_58_12 : 
  molecular_weight_C4H10 = 58.12 := by sorry

end NUMINAMATH_CALUDE_molecular_weight_C4H10_is_58_12_l1602_160226


namespace NUMINAMATH_CALUDE_sqrt_equation_solution_l1602_160231

theorem sqrt_equation_solution :
  ∀ x : ℝ, Real.sqrt (x + 15) = 12 → x = 129 := by sorry

end NUMINAMATH_CALUDE_sqrt_equation_solution_l1602_160231


namespace NUMINAMATH_CALUDE_circus_receipts_l1602_160289

theorem circus_receipts (total_tickets : ℕ) (adult_ticket_cost : ℕ) (child_ticket_cost : ℕ) (adult_tickets_sold : ℕ) :
  total_tickets = 522 →
  adult_ticket_cost = 15 →
  child_ticket_cost = 8 →
  adult_tickets_sold = 130 →
  (adult_tickets_sold * adult_ticket_cost + (total_tickets - adult_tickets_sold) * child_ticket_cost) = 5086 :=
by sorry

end NUMINAMATH_CALUDE_circus_receipts_l1602_160289


namespace NUMINAMATH_CALUDE_greatest_rational_root_l1602_160264

-- Define the quadratic equation type
structure QuadraticEquation where
  a : Nat
  b : Nat
  c : Nat
  h_a : a ≤ 100
  h_b : b ≤ 100
  h_c : c ≤ 100

-- Define a rational root
def RationalRoot (q : QuadraticEquation) (x : ℚ) : Prop :=
  q.a * x^2 + q.b * x + q.c = 0

-- State the theorem
theorem greatest_rational_root (q : QuadraticEquation) :
  ∃ (x : ℚ), RationalRoot q x ∧ 
  ∀ (y : ℚ), RationalRoot q y → y ≤ x ∧ x = -1/99 :=
sorry

end NUMINAMATH_CALUDE_greatest_rational_root_l1602_160264


namespace NUMINAMATH_CALUDE_g_sum_symmetric_l1602_160245

/-- Given a function g(x) = ax^8 + bx^6 - cx^4 + 5 where g(10) = 3,
    prove that g(10) + g(-10) = 6 -/
theorem g_sum_symmetric (a b c : ℝ) : 
  let g : ℝ → ℝ := λ x ↦ a * x^8 + b * x^6 - c * x^4 + 5
  g 10 = 3 → g 10 + g (-10) = 6 := by sorry

end NUMINAMATH_CALUDE_g_sum_symmetric_l1602_160245


namespace NUMINAMATH_CALUDE_triangle_side_length_range_l1602_160253

theorem triangle_side_length_range : ∃ (min max : ℤ),
  (∀ x : ℤ, (x + 8 > 10 ∧ x + 10 > 8 ∧ 8 + 10 > x) → min ≤ x ∧ x ≤ max) ∧
  min = 3 ∧ max = 17 ∧ max - min = 14 := by
  sorry

end NUMINAMATH_CALUDE_triangle_side_length_range_l1602_160253


namespace NUMINAMATH_CALUDE_M_intersect_P_equals_singleton_l1602_160292

-- Define the sets M and P
def M : Set (ℝ × ℝ) := {(x, y) | 4 * x + y = 6}
def P : Set (ℝ × ℝ) := {(x, y) | 3 * x + 2 * y = 7}

-- Theorem statement
theorem M_intersect_P_equals_singleton : M ∩ P = {(1, 2)} := by
  sorry

end NUMINAMATH_CALUDE_M_intersect_P_equals_singleton_l1602_160292


namespace NUMINAMATH_CALUDE_necessary_not_sufficient_condition_l1602_160238

/-- A quadratic equation x^2 + x + c = 0 has two real roots of opposite signs -/
def has_two_real_roots_opposite_signs (c : ℝ) : Prop :=
  ∃ x y : ℝ, x ≠ y ∧ x * y < 0 ∧ x^2 + x + c = 0 ∧ y^2 + y + c = 0

theorem necessary_not_sufficient_condition :
  (∀ c : ℝ, has_two_real_roots_opposite_signs c → c < 0) ∧
  (∃ c : ℝ, c < 0 ∧ ¬has_two_real_roots_opposite_signs c) :=
by sorry

end NUMINAMATH_CALUDE_necessary_not_sufficient_condition_l1602_160238


namespace NUMINAMATH_CALUDE_mean_of_four_integers_l1602_160211

theorem mean_of_four_integers (x : ℤ) : 
  (78 + 83 + 82 + x) / 4 = 80 → x = 77 ∧ x = 80 - 3 := by
  sorry

end NUMINAMATH_CALUDE_mean_of_four_integers_l1602_160211


namespace NUMINAMATH_CALUDE_geometric_sequence_ratio_l1602_160240

theorem geometric_sequence_ratio (a : ℕ → ℝ) (q : ℝ) :
  (∀ n : ℕ, a (n + 1) = q * a n) →  -- {a_n} is a geometric sequence with ratio q
  a 1 * a 2 * a 3 = 2 →             -- First condition
  a 2 * a 3 * a 4 = 16 →            -- Second condition
  q = 2 :=                          -- Conclusion to prove
by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_ratio_l1602_160240


namespace NUMINAMATH_CALUDE_complex_number_in_third_quadrant_l1602_160286

theorem complex_number_in_third_quadrant :
  let z : ℂ := -Complex.I / (1 + Complex.I)
  (z.re < 0) ∧ (z.im < 0) := by
  sorry

end NUMINAMATH_CALUDE_complex_number_in_third_quadrant_l1602_160286


namespace NUMINAMATH_CALUDE_geometric_sequence_sum_l1602_160272

/-- Given a geometric sequence {a_n} where a₁ = 3 and a₄ = 24, prove that a₃ + a₄ + a₅ = 84 -/
theorem geometric_sequence_sum (a : ℕ → ℝ) : 
  (∀ n, a (n + 1) / a n = a 2 / a 1) →  -- Condition for geometric sequence
  a 1 = 3 →
  a 4 = 24 →
  a 3 + a 4 + a 5 = 84 := by
sorry

end NUMINAMATH_CALUDE_geometric_sequence_sum_l1602_160272


namespace NUMINAMATH_CALUDE_investment_income_is_500_l1602_160208

/-- Calculates the total yearly income from a set of investments -/
def totalYearlyIncome (totalAmount : ℝ) (firstInvestment : ℝ) (firstRate : ℝ) 
                      (secondInvestment : ℝ) (secondRate : ℝ) (remainderRate : ℝ) : ℝ :=
  let remainderInvestment := totalAmount - firstInvestment - secondInvestment
  firstInvestment * firstRate + secondInvestment * secondRate + remainderInvestment * remainderRate

/-- Theorem: The total yearly income from the given investment strategy is $500 -/
theorem investment_income_is_500 : 
  totalYearlyIncome 10000 4000 0.05 3500 0.04 0.064 = 500 := by
  sorry

end NUMINAMATH_CALUDE_investment_income_is_500_l1602_160208


namespace NUMINAMATH_CALUDE_problem_solution_l1602_160281

theorem problem_solution (a b : ℝ) (h1 : a - b = 3) (h2 : a * b = 2) : 
  (a + b)^2 = 17 ∧ a^2 - 6*a*b + b^2 = 1 := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l1602_160281


namespace NUMINAMATH_CALUDE_quadratic_second_root_l1602_160247

theorem quadratic_second_root 
  (p q r : ℝ) 
  (h : 2*p*(q-r)*2^2 + 3*q*(r-p)*2 + 4*r*(p-q) = 0) :
  ∃ x : ℝ, 
    x ≠ 2 ∧ 
    2*p*(q-r)*x^2 + 3*q*(r-p)*x + 4*r*(p-q) = 0 ∧ 
    x = (r*(p-q)) / (p*(q-r)) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_second_root_l1602_160247


namespace NUMINAMATH_CALUDE_fraction_powers_equality_l1602_160214

theorem fraction_powers_equality : (0.5 ^ 4) / (0.05 ^ 3) = 500 := by
  sorry

end NUMINAMATH_CALUDE_fraction_powers_equality_l1602_160214


namespace NUMINAMATH_CALUDE_li_ming_on_time_probability_l1602_160201

structure TransportationProbabilities where
  bike_prob : ℝ
  bus_prob : ℝ
  bike_on_time_prob : ℝ
  bus_on_time_prob : ℝ

def probability_on_time (p : TransportationProbabilities) : ℝ :=
  p.bike_prob * p.bike_on_time_prob + p.bus_prob * p.bus_on_time_prob

theorem li_ming_on_time_probability :
  ∀ (p : TransportationProbabilities),
    p.bike_prob = 0.7 →
    p.bus_prob = 0.3 →
    p.bike_on_time_prob = 0.9 →
    p.bus_on_time_prob = 0.8 →
    probability_on_time p = 0.87 := by
  sorry

end NUMINAMATH_CALUDE_li_ming_on_time_probability_l1602_160201


namespace NUMINAMATH_CALUDE_pipeA_rate_correct_l1602_160223

/-- Represents the rate at which Pipe A fills the tank -/
def pipeA_rate : ℝ := 40

/-- The capacity of the tank in liters -/
def tank_capacity : ℝ := 750

/-- The rate at which Pipe B fills the tank in liters per minute -/
def pipeB_rate : ℝ := 30

/-- The rate at which Pipe C drains the tank in liters per minute -/
def pipeC_rate : ℝ := 20

/-- The time in minutes it takes to fill the tank -/
def fill_time : ℝ := 45

/-- The duration of one cycle in minutes -/
def cycle_duration : ℝ := 3

/-- Theorem stating that the rate of Pipe A is correct given the conditions -/
theorem pipeA_rate_correct : 
  tank_capacity = (fill_time / cycle_duration) * (pipeA_rate + pipeB_rate - pipeC_rate) :=
by sorry

end NUMINAMATH_CALUDE_pipeA_rate_correct_l1602_160223


namespace NUMINAMATH_CALUDE_even_increasing_negative_inequality_l1602_160269

/-- A function that is even and increasing on (-∞, -1] -/
def EvenIncreasingNegative (f : ℝ → ℝ) : Prop :=
  (∀ x, f (-x) = f x) ∧ 
  (∀ x y, x ≤ y → y ≤ -1 → f x ≤ f y)

/-- Theorem stating the inequality for functions that are even and increasing on (-∞, -1] -/
theorem even_increasing_negative_inequality (f : ℝ → ℝ) 
  (h : EvenIncreasingNegative f) : 
  f 2 < f (-3/2) ∧ f (-3/2) < f (-1) := by
  sorry

end NUMINAMATH_CALUDE_even_increasing_negative_inequality_l1602_160269


namespace NUMINAMATH_CALUDE_unique_representation_l1602_160229

theorem unique_representation (A : ℕ) : 
  ∃! (x y : ℕ), A = ((x + y)^2 + 3*x + y) / 2 := by
  sorry

end NUMINAMATH_CALUDE_unique_representation_l1602_160229


namespace NUMINAMATH_CALUDE_collision_count_theorem_l1602_160265

/-- Represents the physical properties and conditions of the ball collision problem -/
structure BallCollisionProblem where
  tubeLength : ℝ
  numBalls : ℕ
  ballVelocity : ℝ
  timePeriod : ℝ

/-- Calculates the number of collisions for a given BallCollisionProblem -/
def calculateCollisions (problem : BallCollisionProblem) : ℕ :=
  sorry

/-- Theorem stating that the number of collisions for the given problem is 505000 -/
theorem collision_count_theorem (problem : BallCollisionProblem) 
  (h1 : problem.tubeLength = 1)
  (h2 : problem.numBalls = 100)
  (h3 : problem.ballVelocity = 10)
  (h4 : problem.timePeriod = 10) :
  calculateCollisions problem = 505000 := by
  sorry

end NUMINAMATH_CALUDE_collision_count_theorem_l1602_160265


namespace NUMINAMATH_CALUDE_fencing_calculation_l1602_160249

/-- Calculates the fencing required for a rectangular field -/
theorem fencing_calculation (area : ℝ) (uncovered_side : ℝ) : 
  area > 0 → uncovered_side > 0 → area / uncovered_side + 2 * uncovered_side = 76 := by
  sorry

#check fencing_calculation

end NUMINAMATH_CALUDE_fencing_calculation_l1602_160249


namespace NUMINAMATH_CALUDE_necessary_but_not_sufficient_l1602_160268

theorem necessary_but_not_sufficient (a b : ℝ) (ha : a > 0) :
  (∀ b, a > |b| → a + b > 0) ∧ (∃ b, a + b > 0 ∧ ¬(a > |b|)) := by
  sorry

end NUMINAMATH_CALUDE_necessary_but_not_sufficient_l1602_160268


namespace NUMINAMATH_CALUDE_phi_difference_bound_l1602_160251

/-- The n-th iterate of a function -/
def iterate (f : ℝ → ℝ) (n : ℕ) : ℝ → ℝ :=
  match n with
  | 0 => id
  | n + 1 => f ∘ (iterate f n)

/-- The main theorem -/
theorem phi_difference_bound
  (f : ℝ → ℝ)
  (h_mono : ∀ x y, x ≤ y → f x ≤ f y)
  (h_period : ∀ x, f (x + 1) = f x + 1)
  (n : ℕ)
  (φ : ℝ → ℝ)
  (h_phi : ∀ x, φ x = iterate f n x - x) :
  ∀ x y, |φ x - φ y| < 1 :=
sorry

end NUMINAMATH_CALUDE_phi_difference_bound_l1602_160251


namespace NUMINAMATH_CALUDE_min_distance_curve_line_l1602_160293

/-- Given real numbers a, b, c, d satisfying the conditions,
    prove that the minimum value of (a-c)^2 + (b-d)^2 is (9/5) * (ln(e/3))^2 -/
theorem min_distance_curve_line (a b c d : ℝ) 
    (h1 : (a + 3 * Real.log a) / b = 1)
    (h2 : (d - 3) / (2 * c) = 1) :
    ∃ (min : ℝ), min = (9/5) * (Real.log (Real.exp 1 / 3))^2 ∧
    ∀ (x y z w : ℝ), 
    (x + 3 * Real.log x) / y = 1 → 
    (w - 3) / (2 * z) = 1 → 
    (x - z)^2 + (y - w)^2 ≥ min :=
by sorry

end NUMINAMATH_CALUDE_min_distance_curve_line_l1602_160293


namespace NUMINAMATH_CALUDE_system_solution_l1602_160297

theorem system_solution : ∃! (x y : ℝ), x - y = -5 ∧ 3 * x + 2 * y = 10 := by
  sorry

end NUMINAMATH_CALUDE_system_solution_l1602_160297


namespace NUMINAMATH_CALUDE_range_of_x_plus_y_l1602_160236

theorem range_of_x_plus_y (x y : ℝ) (h : x^2 + 2*x*y + 4*y^2 = 6) :
  ∃ (z : ℝ), z = x + y ∧ -Real.sqrt 6 ≤ z ∧ z ≤ Real.sqrt 6 ∧
  ∀ (w : ℝ), (∃ (a b : ℝ), w = a + b ∧ a^2 + 2*a*b + 4*b^2 = 6) →
    -Real.sqrt 6 ≤ w ∧ w ≤ Real.sqrt 6 :=
sorry

end NUMINAMATH_CALUDE_range_of_x_plus_y_l1602_160236


namespace NUMINAMATH_CALUDE_candy_boxes_l1602_160298

theorem candy_boxes (pieces_per_box : ℕ) (total_pieces : ℕ) (h1 : pieces_per_box = 500) (h2 : total_pieces = 3000) :
  total_pieces / pieces_per_box = 6 := by
sorry

end NUMINAMATH_CALUDE_candy_boxes_l1602_160298


namespace NUMINAMATH_CALUDE_harmonic_sum_equals_one_third_l1602_160210

-- Define the harmonic number sequence
def H : ℕ → ℚ
  | 0 => 0
  | n + 1 => H n + 1 / (n + 1)

-- Define the summand of the series
def summand (n : ℕ) : ℚ := 1 / ((n + 2 : ℚ) * H (n + 1) * H (n + 2))

-- State the theorem
theorem harmonic_sum_equals_one_third :
  ∑' n, summand n = 1 / 3 := by sorry

end NUMINAMATH_CALUDE_harmonic_sum_equals_one_third_l1602_160210


namespace NUMINAMATH_CALUDE_sin_equality_theorem_l1602_160243

theorem sin_equality_theorem (n : ℤ) :
  -180 ≤ n ∧ n ≤ 180 →
  (Real.sin (n * π / 180) = Real.sin (720 * π / 180)) ↔ (n = 0 ∨ n = 180) := by
sorry

end NUMINAMATH_CALUDE_sin_equality_theorem_l1602_160243


namespace NUMINAMATH_CALUDE_square_of_real_number_proposition_l1602_160295

theorem square_of_real_number_proposition :
  ∃ (p q : Prop), (∀ x : ℝ, x^2 > 0 ∨ x^2 = 0) ↔ (p ∨ q) :=
by sorry

end NUMINAMATH_CALUDE_square_of_real_number_proposition_l1602_160295


namespace NUMINAMATH_CALUDE_two_valid_configurations_l1602_160250

-- Define a 4x4 table as a function from (Fin 4 × Fin 4) to Char
def Table := Fin 4 → Fin 4 → Char

-- Define the swap operations
def swapFirstTwoRows (t : Table) : Table :=
  fun i j => if i = 0 then t 1 j else if i = 1 then t 0 j else t i j

def swapFirstTwoCols (t : Table) : Table :=
  fun i j => if j = 0 then t i 1 else if j = 1 then t i 0 else t i j

def swapLastTwoCols (t : Table) : Table :=
  fun i j => if j = 2 then t i 3 else if j = 3 then t i 2 else t i j

-- Define the property of identical letters in corresponding quadrants
def maintainsQuadrantProperty (t1 t2 : Table) : Prop :=
  ∀ i j, (t1 i j = t1 (i + 2) j ∧ t1 i j = t1 i (j + 2) ∧ t1 i j = t1 (i + 2) (j + 2)) →
         (t2 i j = t2 (i + 2) j ∧ t2 i j = t2 i (j + 2) ∧ t2 i j = t2 (i + 2) (j + 2))

-- Define the initial table
def initialTable : Table :=
  fun i j => match (i, j) with
  | (0, 0) => 'A' | (0, 1) => 'B' | (0, 2) => 'C' | (0, 3) => 'D'
  | (1, 0) => 'D' | (1, 1) => 'C' | (1, 2) => 'B' | (1, 3) => 'A'
  | (2, 0) => 'C' | (2, 1) => 'A' | (2, 2) => 'C' | (2, 3) => 'A'
  | (3, 0) => 'B' | (3, 1) => 'D' | (3, 2) => 'B' | (3, 3) => 'D'

-- The main theorem
theorem two_valid_configurations :
  ∃! (validConfigs : Finset Table),
    validConfigs.card = 2 ∧
    (∀ t ∈ validConfigs,
      maintainsQuadrantProperty initialTable
        (swapLastTwoCols (swapFirstTwoCols (swapFirstTwoRows t)))) := by
  sorry

end NUMINAMATH_CALUDE_two_valid_configurations_l1602_160250


namespace NUMINAMATH_CALUDE_division_remainder_l1602_160241

theorem division_remainder (dividend : ℕ) (divisor : ℕ) (quotient : ℕ) (h1 : dividend = 3086) (h2 : divisor = 85) (h3 : quotient = 36) :
  dividend - divisor * quotient = 26 := by
  sorry

end NUMINAMATH_CALUDE_division_remainder_l1602_160241


namespace NUMINAMATH_CALUDE_circle_tangent_to_line_circle_through_origin_l1602_160276

-- Define the circle C
def circle_C (x y m : ℝ) : Prop := x^2 + y^2 + x - 6*y + m = 0

-- Define the line l
def line_l (x y : ℝ) : Prop := x + y - 3 = 0

-- Theorem for part I
theorem circle_tangent_to_line :
  ∃ m : ℝ, ∀ x y : ℝ, circle_C x y m ∧ line_l x y →
    (x + 1/2)^2 + (y - 3)^2 = 1/8 :=
sorry

-- Theorem for part II
theorem circle_through_origin :
  ∃ m : ℝ, m = -3/2 ∧
    (∀ x1 y1 x2 y2 : ℝ,
      (circle_C x1 y1 m ∧ line_l x1 y1) ∧
      (circle_C x2 y2 m ∧ line_l x2 y2) ∧
      x1 ≠ x2 →
      x1 * x2 + y1 * y2 = 0) :=
sorry

end NUMINAMATH_CALUDE_circle_tangent_to_line_circle_through_origin_l1602_160276


namespace NUMINAMATH_CALUDE_root_equation_implies_expression_value_l1602_160279

theorem root_equation_implies_expression_value (m : ℝ) : 
  m^2 + 2*m - 1 = 0 → 2*m^2 + 4*m + 2021 = 2023 := by
  sorry

end NUMINAMATH_CALUDE_root_equation_implies_expression_value_l1602_160279


namespace NUMINAMATH_CALUDE_jason_borrowed_amount_l1602_160233

/-- Represents the value of a chore based on its position in the cycle -/
def chore_value (n : ℕ) : ℕ :=
  match n % 6 with
  | 1 => 1
  | 2 => 3
  | 3 => 5
  | 4 => 7
  | 5 => 9
  | 0 => 11
  | _ => 0  -- This case should never occur

/-- Calculates the total value of a complete cycle of 6 chores -/
def cycle_value : ℕ := 
  (chore_value 1) + (chore_value 2) + (chore_value 3) + 
  (chore_value 4) + (chore_value 5) + (chore_value 6)

/-- Theorem: Jason borrowed $288 -/
theorem jason_borrowed_amount : 
  (cycle_value * (48 / 6) = 288) := by
  sorry

end NUMINAMATH_CALUDE_jason_borrowed_amount_l1602_160233


namespace NUMINAMATH_CALUDE_smallest_number_1755_more_than_sum_of_digits_l1602_160225

def sum_of_digits (n : ℕ) : ℕ :=
  if n < 10 then n else (n % 10) + sum_of_digits (n / 10)

theorem smallest_number_1755_more_than_sum_of_digits :
  (∀ m : ℕ, m < 1770 → m ≠ sum_of_digits m + 1755) ∧
  1770 = sum_of_digits 1770 + 1755 :=
sorry

end NUMINAMATH_CALUDE_smallest_number_1755_more_than_sum_of_digits_l1602_160225


namespace NUMINAMATH_CALUDE_difference_of_squares_l1602_160221

theorem difference_of_squares (m : ℝ) : m^2 - 1 = (m + 1) * (m - 1) := by sorry

end NUMINAMATH_CALUDE_difference_of_squares_l1602_160221


namespace NUMINAMATH_CALUDE_incorrect_stability_statement_l1602_160294

/-- Represents the variance of an individual's high jump scores -/
structure JumpVariance where
  value : ℝ
  is_positive : value > 0

/-- Represents the stability of an individual's high jump scores -/
def more_stable (a b : JumpVariance) : Prop :=
  a.value < b.value

theorem incorrect_stability_statement :
  ∃ (a b : JumpVariance),
    a.value = 1.1 ∧
    b.value = 2.5 ∧
    ¬(more_stable a b) :=
sorry

end NUMINAMATH_CALUDE_incorrect_stability_statement_l1602_160294


namespace NUMINAMATH_CALUDE_perpendicular_unit_vector_l1602_160262

/-- Given a vector a = (2, 1), prove that (√5/5, -2√5/5) is a unit vector perpendicular to a. -/
theorem perpendicular_unit_vector (a : ℝ × ℝ) (h : a = (2, 1)) :
  let b : ℝ × ℝ := (Real.sqrt 5 / 5, -2 * Real.sqrt 5 / 5)
  (a.1 * b.1 + a.2 * b.2 = 0) ∧ (b.1^2 + b.2^2 = 1) :=
by sorry

end NUMINAMATH_CALUDE_perpendicular_unit_vector_l1602_160262


namespace NUMINAMATH_CALUDE_geometric_sequence_sum_l1602_160275

/-- Given a geometric sequence {a_n} with positive terms, where 4a_3, a_5, and 2a_4 form an arithmetic sequence, and a_1 = 1, prove that S_4 = 15 -/
theorem geometric_sequence_sum (a : ℕ → ℝ) (q : ℝ) :
  (∀ n, a n > 0) →  -- All terms are positive
  (∀ n, a (n + 1) = q * a n) →  -- Geometric sequence
  4 * a 3 + a 5 = 2 * (2 * a 4) →  -- Arithmetic sequence condition
  a 1 = 1 →  -- First term is 1
  a 1 + a 2 + a 3 + a 4 = 15 :=  -- S_4 = 15
by sorry

end NUMINAMATH_CALUDE_geometric_sequence_sum_l1602_160275


namespace NUMINAMATH_CALUDE_rectangle_perimeter_problem_l1602_160239

/-- The perimeter of a rectangle given its width and height -/
def perimeter (width height : ℝ) : ℝ := 2 * (width + height)

/-- The width of the large rectangle in terms of small rectangles -/
def large_width : ℕ := 5

/-- The height of the large rectangle in terms of small rectangles -/
def large_height : ℕ := 4

theorem rectangle_perimeter_problem (x y : ℝ) 
  (hA : perimeter (6 * x) y = 56)
  (hB : perimeter (4 * x) (3 * y) = 56) :
  perimeter (2 * x) (3 * y) = 40 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_perimeter_problem_l1602_160239


namespace NUMINAMATH_CALUDE_four_point_circle_theorem_l1602_160266

/-- A type representing a point in a plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- A type representing a circle in a plane -/
structure Circle where
  center : Point
  radius : ℝ

/-- Check if a point is on or inside a circle -/
def Point.onOrInside (p : Point) (c : Circle) : Prop :=
  (p.x - c.center.x)^2 + (p.y - c.center.y)^2 ≤ c.radius^2

/-- Check if a point is on the circumference of a circle -/
def Point.onCircumference (p : Point) (c : Circle) : Prop :=
  (p.x - c.center.x)^2 + (p.y - c.center.y)^2 = c.radius^2

/-- Check if three points are collinear -/
def collinear (p q r : Point) : Prop :=
  (q.y - p.y) * (r.x - p.x) = (r.y - p.y) * (q.x - p.x)

/-- The main theorem -/
theorem four_point_circle_theorem (a b c d : Point) 
  (h : ¬collinear a b c ∧ ¬collinear a b d ∧ ¬collinear a c d ∧ ¬collinear b c d) :
  ∃ (circ : Circle), 
    (Point.onCircumference a circ ∧ Point.onCircumference b circ ∧ Point.onCircumference c circ ∧ Point.onOrInside d circ) ∨
    (Point.onCircumference a circ ∧ Point.onCircumference b circ ∧ Point.onCircumference d circ ∧ Point.onOrInside c circ) ∨
    (Point.onCircumference a circ ∧ Point.onCircumference c circ ∧ Point.onCircumference d circ ∧ Point.onOrInside b circ) ∨
    (Point.onCircumference b circ ∧ Point.onCircumference c circ ∧ Point.onCircumference d circ ∧ Point.onOrInside a circ) :=
sorry

end NUMINAMATH_CALUDE_four_point_circle_theorem_l1602_160266


namespace NUMINAMATH_CALUDE_inverse_f_at_3_l1602_160217

-- Define the function f
def f (x : ℝ) : ℝ := x^2 + 2

-- Define the domain of f
def f_domain (x : ℝ) : Prop := -2 ≤ x ∧ x < 0

-- Statement to prove
theorem inverse_f_at_3 :
  ∃ (f_inv : ℝ → ℝ), 
    (∀ x, f_domain x → f_inv (f x) = x) ∧
    (∀ y, ∃ x, f_domain x ∧ f x = y → f (f_inv y) = y) ∧
    f_inv 3 = -1 :=
sorry

end NUMINAMATH_CALUDE_inverse_f_at_3_l1602_160217


namespace NUMINAMATH_CALUDE_g_12_equals_155_l1602_160280

/-- The function g defined for all integers n -/
def g (n : ℤ) : ℤ := n^2 - n + 23

/-- Theorem stating that g(12) equals 155 -/
theorem g_12_equals_155 : g 12 = 155 := by
  sorry

end NUMINAMATH_CALUDE_g_12_equals_155_l1602_160280


namespace NUMINAMATH_CALUDE_new_energy_vehicle_analysis_l1602_160271

def daily_distances : List Int := [-8, -12, -16, 0, 22, 31, 33]
def standard_distance : Int := 50
def gasoline_consumption : Rat := 5.5
def gasoline_price : Rat := 8.4
def electric_consumption : Rat := 15
def electricity_price : Rat := 0.5

theorem new_energy_vehicle_analysis :
  let max_distance := daily_distances.foldl max (daily_distances.head!)
  let min_distance := daily_distances.foldl min (daily_distances.head!)
  let total_distance := daily_distances.sum
  let gasoline_cost := (total_distance : Rat) / 100 * gasoline_consumption * gasoline_price
  let electric_cost := (total_distance : Rat) / 100 * electric_consumption * electricity_price
  (max_distance - min_distance = 49) ∧
  (total_distance = 50) ∧
  (gasoline_cost - electric_cost = 154.8) := by
  sorry


end NUMINAMATH_CALUDE_new_energy_vehicle_analysis_l1602_160271


namespace NUMINAMATH_CALUDE_multiply_divide_example_l1602_160235

theorem multiply_divide_example : (3.242 * 10) / 100 = 0.3242 := by
  sorry

end NUMINAMATH_CALUDE_multiply_divide_example_l1602_160235

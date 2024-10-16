import Mathlib

namespace NUMINAMATH_CALUDE_caterpillar_length_difference_l3452_345284

theorem caterpillar_length_difference :
  let green_length : ℝ := 3
  let orange_length : ℝ := 1.17
  green_length - orange_length = 1.83 := by
sorry

end NUMINAMATH_CALUDE_caterpillar_length_difference_l3452_345284


namespace NUMINAMATH_CALUDE_cakes_slices_problem_l3452_345258

theorem cakes_slices_problem (total_slices : ℕ) (friends_fraction : ℚ) 
  (family_fraction : ℚ) (eaten_slices : ℕ) (remaining_slices : ℕ) :
  total_slices = 16 →
  family_fraction = 1/3 →
  eaten_slices = 3 →
  remaining_slices = 5 →
  (1 - friends_fraction) * (1 - family_fraction) * total_slices - eaten_slices = remaining_slices →
  friends_fraction = 1/4 := by
sorry

end NUMINAMATH_CALUDE_cakes_slices_problem_l3452_345258


namespace NUMINAMATH_CALUDE_negation_equivalence_l3452_345256

theorem negation_equivalence (a : ℝ) :
  (¬ ∃ x₀ : ℝ, x₀ > 0 ∧ 2^x₀ * (x₀ - a) > 1) ↔
  (∀ x : ℝ, x > 0 → 2^x * (x - a) ≤ 1) :=
by sorry

end NUMINAMATH_CALUDE_negation_equivalence_l3452_345256


namespace NUMINAMATH_CALUDE_wrong_mark_calculation_l3452_345237

theorem wrong_mark_calculation (correct_mark : ℕ) (num_pupils : ℕ) :
  correct_mark = 45 →
  num_pupils = 44 →
  ∃ (wrong_mark : ℕ),
    (wrong_mark - correct_mark : ℚ) / num_pupils = 1/2 ∧
    wrong_mark = 67 :=
by sorry

end NUMINAMATH_CALUDE_wrong_mark_calculation_l3452_345237


namespace NUMINAMATH_CALUDE_quadratic_root_difference_l3452_345260

/-- Given a quadratic equation x^2 + px - q = 0 where p and q are positive real numbers,
    if the difference between its roots is 2, then p = √(4 - 4q) -/
theorem quadratic_root_difference (p q : ℝ) (hp : p > 0) (hq : q > 0) :
  let r₁ := (-p + Real.sqrt (p^2 + 4*q)) / 2
  let r₂ := (-p - Real.sqrt (p^2 + 4*q)) / 2
  (r₁ - r₂ = 2) → p = Real.sqrt (4 - 4*q) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_root_difference_l3452_345260


namespace NUMINAMATH_CALUDE_function_max_implies_a_range_l3452_345259

/-- Given a function f(x) = (ax^2)/2 - (1+2a)x + 2ln(x) where a > 0,
    if f(x) has a maximum value in the interval (1/2, 1),
    then 1 < a < 2. -/
theorem function_max_implies_a_range (a : ℝ) (f : ℝ → ℝ) :
  a > 0 →
  (∀ x, f x = (a * x^2) / 2 - (1 + 2*a) * x + 2 * Real.log x) →
  (∃ x₀ ∈ Set.Ioo (1/2) 1, ∀ x ∈ Set.Ioo (1/2) 1, f x ≤ f x₀) →
  1 < a ∧ a < 2 :=
by sorry

end NUMINAMATH_CALUDE_function_max_implies_a_range_l3452_345259


namespace NUMINAMATH_CALUDE_spaghetti_dinner_cost_l3452_345271

/-- Calculates the cost per serving of a meal given the costs of ingredients and number of servings -/
def cost_per_serving (pasta_cost sauce_cost meatballs_cost : ℚ) (servings : ℕ) : ℚ :=
  (pasta_cost + sauce_cost + meatballs_cost) / servings

/-- Theorem: Given the specific costs and number of servings, the cost per serving is $1.00 -/
theorem spaghetti_dinner_cost :
  cost_per_serving 1 2 5 8 = 1 := by
  sorry

#eval cost_per_serving 1 2 5 8

end NUMINAMATH_CALUDE_spaghetti_dinner_cost_l3452_345271


namespace NUMINAMATH_CALUDE_ellipse_theorem_l3452_345225

-- Define the ellipse
def ellipse (a b : ℝ) (x y : ℝ) : Prop := x^2 / a^2 + y^2 / b^2 = 1

-- Define the conditions
def ellipse_conditions (a b : ℝ) : Prop :=
  a > b ∧ b > 0 ∧ 2 * b = 2 ∧ ellipse a b 1 (Real.sqrt 3 / 2)

-- Theorem statement
theorem ellipse_theorem (a b : ℝ) (h : ellipse_conditions a b) :
  (∀ x y : ℝ, ellipse a b x y ↔ x^2 / 4 + y^2 = 1) ∧
  (∃ C : ℝ × ℝ, C.1^2 / 4 + C.2^2 = 1 ∧
    ∃ A B : ℝ × ℝ, A.1^2 / 4 + A.2^2 = 1 ∧ B.1^2 / 4 + B.2^2 = 1 ∧
      (A.1 * B.1 + A.2 * B.2 = 0) ∧
      (C.1 = A.1 + B.1) ∧ (C.2 = A.2 + B.2) ∧
      (abs (A.1 * B.2 - A.2 * B.1) = Real.sqrt 3)) :=
by
  sorry


end NUMINAMATH_CALUDE_ellipse_theorem_l3452_345225


namespace NUMINAMATH_CALUDE_touching_spheres_bounds_l3452_345272

/-- Represents a tetrahedron -/
structure Tetrahedron where
  faces : Fin 4 → Real
  volume : Real

/-- Represents a sphere touching all face planes of a tetrahedron -/
structure TouchingSphere where
  radius : Real
  center : Real × Real × Real

/-- Returns the number of spheres touching all face planes of a given tetrahedron -/
def countTouchingSpheres (t : Tetrahedron) : ℕ :=
  sorry

/-- Theorem stating the bounds on the number of touching spheres for any tetrahedron -/
theorem touching_spheres_bounds (t : Tetrahedron) :
  5 ≤ countTouchingSpheres t ∧ countTouchingSpheres t ≤ 8 := by
  sorry

end NUMINAMATH_CALUDE_touching_spheres_bounds_l3452_345272


namespace NUMINAMATH_CALUDE_john_hired_twenty_lessons_l3452_345243

/-- Given the cost of a piano, the original price of a lesson, the discount percentage,
    and the total cost, calculate the number of lessons hired. -/
def number_of_lessons (piano_cost lesson_price discount_percent total_cost : ℚ) : ℚ :=
  let discounted_price := lesson_price * (1 - discount_percent / 100)
  let lesson_cost := total_cost - piano_cost
  lesson_cost / discounted_price

/-- Prove that given the specified costs and discount, John hired 20 lessons. -/
theorem john_hired_twenty_lessons :
  number_of_lessons 500 40 25 1100 = 20 := by
  sorry

end NUMINAMATH_CALUDE_john_hired_twenty_lessons_l3452_345243


namespace NUMINAMATH_CALUDE_tangent_line_equation_l3452_345297

noncomputable def f (x : ℝ) : ℝ := (x - 2) * (x^3 - 1)

theorem tangent_line_equation :
  let p : ℝ × ℝ := (1, 0)
  let m : ℝ := deriv f p.1
  (λ (x y : ℝ) ↦ m * (x - p.1) - (y - p.2)) = (λ (x y : ℝ) ↦ 3 * x + y - 3) :=
by sorry

end NUMINAMATH_CALUDE_tangent_line_equation_l3452_345297


namespace NUMINAMATH_CALUDE_range_of_a_l3452_345282

-- Define the propositions p and q
def p (x : ℝ) : Prop := x^2 - 8*x - 20 ≤ 0
def q (x a : ℝ) : Prop := x^2 - 2*x + 1 - a^2 ≥ 0

-- Define the set A corresponding to ¬p
def A : Set ℝ := {x | x < -2 ∨ x > 10}

-- Define the set B corresponding to q
def B (a : ℝ) : Set ℝ := {x | x ≤ 1 - a ∨ x ≥ 1 + a}

-- State the theorem
theorem range_of_a :
  ∀ a : ℝ, (a > 0 ∧ A ⊆ B a ∧ A ≠ B a) → (0 < a ∧ a ≤ 3) :=
by sorry

end NUMINAMATH_CALUDE_range_of_a_l3452_345282


namespace NUMINAMATH_CALUDE_constant_fifth_term_binomial_expansion_l3452_345214

theorem constant_fifth_term_binomial_expansion (a x : ℝ) (n : ℕ) :
  (∃ k : ℝ, k ≠ 0 ∧ (Nat.choose n 4) * a^(n-4) * (-1)^4 * x^(n-8) = k) →
  n = 8 := by
sorry

end NUMINAMATH_CALUDE_constant_fifth_term_binomial_expansion_l3452_345214


namespace NUMINAMATH_CALUDE_christmas_gifts_theorem_l3452_345219

/-- The number of gifts left under the Christmas tree -/
def gifts_left (initial_gifts additional_gifts gifts_sent : ℕ) : ℕ :=
  initial_gifts + additional_gifts - gifts_sent

/-- Theorem: Given the initial gifts, additional gifts, and gifts sent,
    prove that the number of gifts left under the tree is 44. -/
theorem christmas_gifts_theorem :
  gifts_left 77 33 66 = 44 := by
  sorry

end NUMINAMATH_CALUDE_christmas_gifts_theorem_l3452_345219


namespace NUMINAMATH_CALUDE_triangle_properties_l3452_345212

/-- Given a triangle ABC with specific properties, prove that A = π/3 and AB = 2 -/
theorem triangle_properties (A B C : ℝ) (AB BC AC : ℝ) :
  (0 < A) ∧ (A < π) →
  (0 < B) ∧ (B < π) →
  (0 < C) ∧ (C < π) →
  A + B + C = π →
  2 * Real.sin B * Real.cos A = Real.sin (A + C) →
  BC = 2 →
  (1/2) * AB * AC * Real.sin A = Real.sqrt 3 →
  A = π/3 ∧ AB = 2 := by
sorry

end NUMINAMATH_CALUDE_triangle_properties_l3452_345212


namespace NUMINAMATH_CALUDE_prime_even_intersection_l3452_345292

def isPrime (n : ℕ) : Prop := sorry

def isEven (n : ℕ) : Prop := sorry

def P : Set ℕ := {n | isPrime n}
def Q : Set ℕ := {n | isEven n}

theorem prime_even_intersection : P ∩ Q = {2} := by sorry

end NUMINAMATH_CALUDE_prime_even_intersection_l3452_345292


namespace NUMINAMATH_CALUDE_modular_inverse_of_5_mod_33_l3452_345279

theorem modular_inverse_of_5_mod_33 :
  ∃ x : ℕ, x ≥ 0 ∧ x ≤ 32 ∧ (5 * x) % 33 = 1 ∧ x = 20 := by sorry

end NUMINAMATH_CALUDE_modular_inverse_of_5_mod_33_l3452_345279


namespace NUMINAMATH_CALUDE_quadratic_discriminant_l3452_345251

/-- The discriminant of a quadratic equation ax^2 + bx + c is b^2 - 4ac -/
def discriminant (a b c : ℚ) : ℚ := b^2 - 4*a*c

theorem quadratic_discriminant : 
  let a : ℚ := 5
  let b : ℚ := 5 + 1/5
  let c : ℚ := 1/5
  discriminant a b c = 576/25 := by sorry

end NUMINAMATH_CALUDE_quadratic_discriminant_l3452_345251


namespace NUMINAMATH_CALUDE_base_conversion_1850_to_base_7_l3452_345223

theorem base_conversion_1850_to_base_7 :
  (5 * 7^3 + 2 * 7^2 + 5 * 7^1 + 2 * 7^0 : ℕ) = 1850 := by
  sorry

end NUMINAMATH_CALUDE_base_conversion_1850_to_base_7_l3452_345223


namespace NUMINAMATH_CALUDE_cubic_polynomial_problem_l3452_345261

-- Define the cubic equation
def cubic_equation (x : ℝ) : ℝ := x^3 + 4*x^2 + 7*x + 10

-- Define the roots a, b, c
noncomputable def a : ℝ := sorry
noncomputable def b : ℝ := sorry
noncomputable def c : ℝ := sorry

-- Define P(x)
noncomputable def P : ℝ → ℝ := sorry

-- Theorem statement
theorem cubic_polynomial_problem :
  (cubic_equation a = 0) ∧ 
  (cubic_equation b = 0) ∧ 
  (cubic_equation c = 0) ∧
  (P a = 2*(b + c)) ∧
  (P b = 2*(a + c)) ∧
  (P c = 2*(a + b)) ∧
  (P (a + b + c) = -20) →
  ∀ x, P x = (4*x^3 + 16*x^2 + 55*x - 16) / 9 := by
  sorry

end NUMINAMATH_CALUDE_cubic_polynomial_problem_l3452_345261


namespace NUMINAMATH_CALUDE_average_cost_is_two_l3452_345246

/-- Calculates the average cost per fruit given the costs and quantities of apples, bananas, and oranges. -/
def average_cost_per_fruit (apple_cost banana_cost orange_cost : ℚ) 
                           (apple_qty banana_qty orange_qty : ℕ) : ℚ :=
  let total_cost := apple_cost * apple_qty + banana_cost * banana_qty + orange_cost * orange_qty
  let total_qty := apple_qty + banana_qty + orange_qty
  total_cost / total_qty

/-- Proves that the average cost per fruit is $2 given the specific costs and quantities. -/
theorem average_cost_is_two :
  average_cost_per_fruit 2 1 3 12 4 4 = 2 := by
  sorry

#eval average_cost_per_fruit 2 1 3 12 4 4

end NUMINAMATH_CALUDE_average_cost_is_two_l3452_345246


namespace NUMINAMATH_CALUDE_slope_of_line_parallel_lines_coefficient_l3452_345201

/-- Two lines are parallel if and only if their slopes are equal -/
axiom parallel_lines_equal_slopes {m₁ m₂ b₁ b₂ : ℝ} :
  (∀ x y : ℝ, m₁ * x + y = b₁ ↔ m₂ * x + y = b₂) ↔ m₁ = m₂

/-- The slope of a line in the form ax + by + c = 0 is -a/b -/
theorem slope_of_line (a b c : ℝ) (hb : b ≠ 0) :
  ∀ x y : ℝ, a * x + b * y + c = 0 ↔ y = (-a / b) * x - c / b :=
sorry

theorem parallel_lines_coefficient (a : ℝ) :
  (∀ x y : ℝ, a * x + 2 * y + 2 = 0 ↔ 3 * x - y - 2 = 0) → a = -6 :=
sorry

end NUMINAMATH_CALUDE_slope_of_line_parallel_lines_coefficient_l3452_345201


namespace NUMINAMATH_CALUDE_no_four_tangents_for_different_radii_l3452_345285

/-- Represents a circle in a plane --/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

/-- Represents the number of common tangents between two circles --/
def commonTangents (c1 c2 : Circle) : ℕ := sorry

/-- Two circles with different radii cannot have exactly 4 common tangents --/
theorem no_four_tangents_for_different_radii (c1 c2 : Circle) 
  (h : c1.radius ≠ c2.radius) : commonTangents c1 c2 ≠ 4 := by sorry

end NUMINAMATH_CALUDE_no_four_tangents_for_different_radii_l3452_345285


namespace NUMINAMATH_CALUDE_only_odd_solution_is_one_l3452_345296

theorem only_odd_solution_is_one :
  ∀ y : ℤ, ∃ x : ℤ, x^2 + 2*y^2 = y*x^2 + y + 1 ∧ Odd y → y = 1 :=
by sorry

end NUMINAMATH_CALUDE_only_odd_solution_is_one_l3452_345296


namespace NUMINAMATH_CALUDE_problem_statement_l3452_345220

theorem problem_statement (a b : ℝ) 
  (h_pos_a : a > 0) (h_pos_b : b > 0)
  (h_a : a ≥ 1/a + 2/b) (h_b : b ≥ 3/a + 2/b) :
  (a + b ≥ 4) ∧ 
  (a^2 + b^2 ≥ 3 + 2*Real.sqrt 6) ∧ 
  (1/a + 1/b < 1 + Real.sqrt 2 / 2) := by
sorry

end NUMINAMATH_CALUDE_problem_statement_l3452_345220


namespace NUMINAMATH_CALUDE_repeated_sequence_result_l3452_345204

/-- Represents one cycle of operations -/
def cycle_increment : Int := 15 - 12 + 3

/-- Calculates the number of complete cycles in n steps -/
def complete_cycles (n : Nat) : Nat := n / 3

/-- Calculates the number of remaining steps after complete cycles -/
def remaining_steps (n : Nat) : Nat := n % 3

/-- Calculates the increment from remaining steps -/
def remaining_increment (steps : Nat) : Int :=
  if steps = 1 then 15
  else if steps = 2 then 15 - 12
  else 0

/-- Theorem stating the result of the repeated operation sequence -/
theorem repeated_sequence_result :
  let initial_value : Int := 100
  let total_steps : Nat := 26
  let cycles : Nat := complete_cycles total_steps
  let remaining : Nat := remaining_steps total_steps
  initial_value + cycles * cycle_increment + remaining_increment remaining = 151 := by
  sorry

end NUMINAMATH_CALUDE_repeated_sequence_result_l3452_345204


namespace NUMINAMATH_CALUDE_water_height_in_cylinder_l3452_345263

/-- Given a cone with base radius 10 cm and height 15 cm, when its volume of water is poured into a cylinder with base radius 20 cm, the height of water in the cylinder is 1.25 cm. -/
theorem water_height_in_cylinder (π : ℝ) : 
  let cone_radius : ℝ := 10
  let cone_height : ℝ := 15
  let cylinder_radius : ℝ := 20
  let cone_volume : ℝ := (1/3) * π * cone_radius^2 * cone_height
  let cylinder_height : ℝ := cone_volume / (π * cylinder_radius^2)
  cylinder_height = 1.25 := by
  sorry

end NUMINAMATH_CALUDE_water_height_in_cylinder_l3452_345263


namespace NUMINAMATH_CALUDE_mall_price_change_loss_l3452_345281

theorem mall_price_change_loss : ∀ (a b : ℝ),
  a * (1.2 : ℝ)^2 = 23.04 →
  b * (0.8 : ℝ)^2 = 23.04 →
  (a + b) - 2 * 23.04 = 5.92 := by
sorry

end NUMINAMATH_CALUDE_mall_price_change_loss_l3452_345281


namespace NUMINAMATH_CALUDE_excavation_time_equality_l3452_345233

/-- Represents the dimensions of an excavation site -/
structure Dimensions where
  depth : ℝ
  length : ℝ
  breadth : ℝ

/-- Calculates the volume of an excavation site given its dimensions -/
def volume (d : Dimensions) : ℝ := d.depth * d.length * d.breadth

/-- The number of days required to dig an excavation site is directly proportional to its volume when the number of laborers is constant -/
axiom days_proportional_to_volume {d1 d2 : Dimensions} {days1 : ℝ} (h : volume d1 = volume d2) :
  days1 = days1 * (volume d2 / volume d1)

theorem excavation_time_equality (initial : Dimensions) (new : Dimensions) (initial_days : ℝ) 
    (h_initial : initial = { depth := 100, length := 25, breadth := 30 })
    (h_new : new = { depth := 75, length := 20, breadth := 50 })
    (h_initial_days : initial_days = 12) :
    initial_days = initial_days * (volume new / volume initial) := by
  sorry

end NUMINAMATH_CALUDE_excavation_time_equality_l3452_345233


namespace NUMINAMATH_CALUDE_equation_solution_l3452_345200

theorem equation_solution : 
  ∃ y : ℝ, (2 / y + (3 / y) / (6 / y) = 1.2) ∧ y = 20 / 7 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l3452_345200


namespace NUMINAMATH_CALUDE_complex_modulus_problem_l3452_345254

theorem complex_modulus_problem (z : ℂ) : 
  z = 3 + (3 + 4*I) / (4 - 3*I) → Complex.abs z = Real.sqrt 10 := by
sorry

end NUMINAMATH_CALUDE_complex_modulus_problem_l3452_345254


namespace NUMINAMATH_CALUDE_min_value_fraction_sum_l3452_345229

theorem min_value_fraction_sum (a d b c : ℝ) 
  (ha : a ≥ 0) (hd : d ≥ 0) (hb : b > 0) (hc : c > 0) 
  (h_sum : b + c ≥ a + d) : 
  b / (c + d) + c / (a + b) ≥ Real.sqrt 2 - 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_min_value_fraction_sum_l3452_345229


namespace NUMINAMATH_CALUDE_f_inequality_range_l3452_345216

noncomputable def f (x : ℝ) : ℝ :=
  if x ≤ 0 then x + 1 else 2^x

theorem f_inequality_range :
  ∀ x : ℝ, (f x + f (x - 1/2) > 1) ↔ (x > -1/4) :=
by sorry

end NUMINAMATH_CALUDE_f_inequality_range_l3452_345216


namespace NUMINAMATH_CALUDE_birthday_candles_ratio_l3452_345231

theorem birthday_candles_ratio (ambika_candles : ℕ) (total_candles : ℕ) : 
  ambika_candles = 4 → total_candles = 14 → 
  ∃ (aniyah_ratio : ℚ), aniyah_ratio = 2.5 ∧ 
  ambika_candles * (1 + aniyah_ratio) = total_candles :=
sorry

end NUMINAMATH_CALUDE_birthday_candles_ratio_l3452_345231


namespace NUMINAMATH_CALUDE_license_plate_difference_l3452_345230

/-- The number of possible letters in a license plate. -/
def num_letters : ℕ := 26

/-- The number of possible digits in a license plate. -/
def num_digits : ℕ := 10

/-- The number of possible license plates for Georgia (LLDLLL format). -/
def georgia_plates : ℕ := num_letters^4 * num_digits^2

/-- The number of possible license plates for Nebraska (LLDDDDD format). -/
def nebraska_plates : ℕ := num_letters^2 * num_digits^5

/-- The difference between the number of possible license plates for Nebraska and Georgia. -/
theorem license_plate_difference : nebraska_plates - georgia_plates = 21902400 := by
  sorry

end NUMINAMATH_CALUDE_license_plate_difference_l3452_345230


namespace NUMINAMATH_CALUDE_system_solution_l3452_345293

theorem system_solution (x y : ℝ) : 
  (x^2 + y^2 ≤ 1 ∧ 
   16 * x^4 - 8 * x^2 * y^2 + y^4 - 40 * x^2 - 10 * y^2 + 25 = 0) ↔ 
  ((x = -2 / Real.sqrt 5 ∧ y = 1 / Real.sqrt 5) ∨
   (x = -2 / Real.sqrt 5 ∧ y = -1 / Real.sqrt 5) ∨
   (x = 2 / Real.sqrt 5 ∧ y = 1 / Real.sqrt 5) ∨
   (x = 2 / Real.sqrt 5 ∧ y = -1 / Real.sqrt 5)) :=
by sorry

end NUMINAMATH_CALUDE_system_solution_l3452_345293


namespace NUMINAMATH_CALUDE_tangent_line_is_correct_l3452_345224

-- Define the curve
def f (x : ℝ) : ℝ := x^2 + 2*x - 1

-- Define the point of tangency
def point : ℝ × ℝ := (1, 2)

-- Define the proposed tangent line
def tangent_line (x y : ℝ) : Prop := 4*x - y - 2 = 0

-- Theorem statement
theorem tangent_line_is_correct :
  let (x₀, y₀) := point
  (∀ x, tangent_line x (f x)) ∧
  (tangent_line x₀ y₀) ∧
  (∀ x, x ≠ x₀ → ¬(tangent_line x (f x) ∧ tangent_line x₀ y₀)) :=
sorry

end NUMINAMATH_CALUDE_tangent_line_is_correct_l3452_345224


namespace NUMINAMATH_CALUDE_intersection_equals_B_implies_a_is_one_l3452_345291

def A : Set ℝ := {-1, 1}

def B (a : ℝ) : Set ℝ := {x : ℝ | a * x^2 = 1}

theorem intersection_equals_B_implies_a_is_one (a : ℝ) : A ∩ B a = B a → a = 1 := by
  sorry

end NUMINAMATH_CALUDE_intersection_equals_B_implies_a_is_one_l3452_345291


namespace NUMINAMATH_CALUDE_angle_terminal_side_value_l3452_345266

theorem angle_terminal_side_value (m : ℝ) (α : ℝ) (h : m ≠ 0) :
  (∃ (x y : ℝ), x = -4 * m ∧ y = 3 * m ∧ 
    x = Real.cos α * Real.sqrt (x^2 + y^2) ∧ 
    y = Real.sin α * Real.sqrt (x^2 + y^2)) →
  2 * Real.sin α + Real.cos α = 2/5 ∨ 2 * Real.sin α + Real.cos α = -2/5 :=
by sorry

end NUMINAMATH_CALUDE_angle_terminal_side_value_l3452_345266


namespace NUMINAMATH_CALUDE_journey_problem_l3452_345241

theorem journey_problem (total_distance : ℝ) (days : ℕ) (q : ℝ) :
  total_distance = 378 ∧ days = 6 ∧ q = 1/2 →
  ∃ a : ℝ, a * (1 - q^days) / (1 - q) = total_distance ∧ a * q^(days - 1) = 6 := by
  sorry

end NUMINAMATH_CALUDE_journey_problem_l3452_345241


namespace NUMINAMATH_CALUDE_arithmetic_geometric_mean_inequality_l3452_345252

theorem arithmetic_geometric_mean_inequality {x y : ℝ} (hx : x ≥ 0) (hy : y ≥ 0) :
  (x + y) / 2 ≥ Real.sqrt (x * y) ∧
  ((x + y) / 2 = Real.sqrt (x * y) ↔ x = y) := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_geometric_mean_inequality_l3452_345252


namespace NUMINAMATH_CALUDE_symmetry_about_x_axis_l3452_345222

/-- A point in a 2D plane -/
structure Point2D where
  x : ℝ
  y : ℝ

/-- Symmetry about the x-axis -/
def symmetricAboutXAxis (p : Point2D) : Point2D :=
  { x := p.x, y := -p.y }

theorem symmetry_about_x_axis :
  let P : Point2D := { x := -1, y := 5 }
  symmetricAboutXAxis P = { x := -1, y := -5 } := by
  sorry

end NUMINAMATH_CALUDE_symmetry_about_x_axis_l3452_345222


namespace NUMINAMATH_CALUDE_square_of_r_minus_three_l3452_345232

theorem square_of_r_minus_three (r : ℝ) (h : r^2 - 6*r + 5 = 0) : (r - 3)^2 = 4 := by
  sorry

end NUMINAMATH_CALUDE_square_of_r_minus_three_l3452_345232


namespace NUMINAMATH_CALUDE_geometric_sequence_product_l3452_345289

theorem geometric_sequence_product (x y z : ℝ) : 
  1 < x ∧ x < y ∧ y < z ∧ z < 4 →
  (∃ r : ℝ, r > 0 ∧ x = r * 1 ∧ y = r * x ∧ z = r * y ∧ 4 = r * z) →
  1 * x * y * z * 4 = 32 := by
sorry

end NUMINAMATH_CALUDE_geometric_sequence_product_l3452_345289


namespace NUMINAMATH_CALUDE_parabola_intersection_l3452_345253

/-- Proves that (-3, 55) and (4, -8) are the only intersection points of the parabolas
    y = 3x^2 - 12x - 8 and y = 2x^2 - 10x + 4 -/
theorem parabola_intersection :
  let f (x : ℝ) := 3 * x^2 - 12 * x - 8
  let g (x : ℝ) := 2 * x^2 - 10 * x + 4
  ∀ x y : ℝ, f x = g x ∧ y = f x ↔ (x = -3 ∧ y = 55) ∨ (x = 4 ∧ y = -8) := by
  sorry

end NUMINAMATH_CALUDE_parabola_intersection_l3452_345253


namespace NUMINAMATH_CALUDE_time_addition_sum_l3452_345248

/-- Represents time in hours, minutes, and seconds -/
structure Time where
  hours : Nat
  minutes : Nat
  seconds : Nat

/-- Adds a duration to a given time and returns the resulting time -/
def addTime (start : Time) (dHours dMinutes dSeconds : Nat) : Time :=
  sorry

/-- Converts a 24-hour time to 12-hour format -/
def to12Hour (t : Time) : Time :=
  sorry

theorem time_addition_sum (startTime : Time) :
  let endTime := to12Hour (addTime startTime 145 50 15)
  endTime.hours + endTime.minutes + endTime.seconds = 69 := by
  sorry

end NUMINAMATH_CALUDE_time_addition_sum_l3452_345248


namespace NUMINAMATH_CALUDE_total_rope_length_l3452_345202

def rope_lengths : List ℝ := [8, 20, 2, 2, 2, 7]
def knot_loss : ℝ := 1.2

theorem total_rope_length :
  let initial_length := rope_lengths.sum
  let num_knots := rope_lengths.length - 1
  let total_loss := num_knots * knot_loss
  initial_length - total_loss = 35 := by
  sorry

end NUMINAMATH_CALUDE_total_rope_length_l3452_345202


namespace NUMINAMATH_CALUDE_exists_A_for_monomial_l3452_345227

-- Define what a monomial is
def is_monomial (e : ℝ → ℝ) : Prop :=
  ∃ (c : ℝ) (n : ℕ), ∀ x, e x = c * x^n

-- Define the expression -3x + A
def expr (A : ℝ → ℝ) (x : ℝ) : ℝ := -3*x + A x

-- Theorem statement
theorem exists_A_for_monomial :
  ∃ (A : ℝ → ℝ), is_monomial (expr A) :=
sorry

end NUMINAMATH_CALUDE_exists_A_for_monomial_l3452_345227


namespace NUMINAMATH_CALUDE_line_intersection_y_axis_l3452_345213

-- Define a point in 2D space
structure Point2D where
  x : ℝ
  y : ℝ

-- Define a line by two points
structure Line where
  p1 : Point2D
  p2 : Point2D

-- Define the y-axis
def yAxis : Line := { p1 := ⟨0, 0⟩, p2 := ⟨0, 1⟩ }

-- Function to check if a point is on a line
def isPointOnLine (p : Point2D) (l : Line) : Prop :=
  (p.y - l.p1.y) * (l.p2.x - l.p1.x) = (p.x - l.p1.x) * (l.p2.y - l.p1.y)

-- Function to check if a point is on the y-axis
def isPointOnYAxis (p : Point2D) : Prop :=
  p.x = 0

-- Theorem statement
theorem line_intersection_y_axis :
  let l : Line := { p1 := ⟨2, 9⟩, p2 := ⟨4, 13⟩ }
  let intersection : Point2D := ⟨0, 5⟩
  isPointOnLine intersection l ∧ isPointOnYAxis intersection := by
  sorry

end NUMINAMATH_CALUDE_line_intersection_y_axis_l3452_345213


namespace NUMINAMATH_CALUDE_six_balls_four_boxes_l3452_345205

/-- The number of ways to distribute n indistinguishable balls into k distinguishable boxes -/
def distribute_balls (n : ℕ) (k : ℕ) : ℕ :=
  sorry

/-- The theorem stating that there are 9 ways to distribute 6 indistinguishable balls into 4 distinguishable boxes -/
theorem six_balls_four_boxes : distribute_balls 6 4 = 9 := by
  sorry

end NUMINAMATH_CALUDE_six_balls_four_boxes_l3452_345205


namespace NUMINAMATH_CALUDE_log_base_2_derivative_l3452_345203

theorem log_base_2_derivative (x : ℝ) (h : x > 0) : 
  deriv (λ x => Real.log x / Real.log 2) x = 1 / (x * Real.log 2) :=
by sorry

end NUMINAMATH_CALUDE_log_base_2_derivative_l3452_345203


namespace NUMINAMATH_CALUDE_golden_ratio_expression_l3452_345276

theorem golden_ratio_expression (R : ℝ) (h1 : R^2 + R - 1 = 0) (h2 : R > 0) :
  R^(R^(R^2 + 1/R) + 1/R) + 1/R = 2 := by
  sorry

end NUMINAMATH_CALUDE_golden_ratio_expression_l3452_345276


namespace NUMINAMATH_CALUDE_triangle_square_ratio_l3452_345288

/-- A triangle in a 2D plane --/
structure Triangle :=
  (a b c : ℝ)
  (hpos : 0 < a ∧ 0 < b ∧ 0 < c)
  (hineq : a + b > c ∧ b + c > a ∧ c + a > b)

/-- The side length of the largest square inscribed in a triangle --/
noncomputable def maxInscribedSquareSide (t : Triangle) : ℝ := sorry

/-- The side length of the smallest square circumscribed around a triangle --/
noncomputable def minCircumscribedSquareSide (t : Triangle) : ℝ := sorry

/-- A triangle is right-angled if one of its angles is 90 degrees --/
def isRightTriangle (t : Triangle) : Prop := sorry

theorem triangle_square_ratio (t : Triangle) :
  minCircumscribedSquareSide t / maxInscribedSquareSide t ≥ 2 ∧
  (minCircumscribedSquareSide t / maxInscribedSquareSide t = 2 ↔ isRightTriangle t) :=
sorry

end NUMINAMATH_CALUDE_triangle_square_ratio_l3452_345288


namespace NUMINAMATH_CALUDE_smallest_two_digit_with_digit_product_12_l3452_345257

def is_two_digit (n : ℕ) : Prop := 10 ≤ n ∧ n ≤ 99

def digit_product (n : ℕ) : ℕ :=
  (n / 10) * (n % 10)

theorem smallest_two_digit_with_digit_product_12 :
  ∀ n : ℕ, is_two_digit n → digit_product n = 12 → 26 ≤ n :=
sorry

end NUMINAMATH_CALUDE_smallest_two_digit_with_digit_product_12_l3452_345257


namespace NUMINAMATH_CALUDE_polynomial_simplification_l3452_345209

theorem polynomial_simplification (x : ℝ) : 
  (3*x^2 + 4*x + 6)*(x-2) - (x-2)*(x^2 + 5*x - 72) + (2*x - 7)*(x-2)*(x+4) = 
  4*x^3 - 8*x^2 + 50*x - 100 := by
sorry

end NUMINAMATH_CALUDE_polynomial_simplification_l3452_345209


namespace NUMINAMATH_CALUDE_gcd_9009_14014_l3452_345234

theorem gcd_9009_14014 : Nat.gcd 9009 14014 = 1001 := by
  sorry

end NUMINAMATH_CALUDE_gcd_9009_14014_l3452_345234


namespace NUMINAMATH_CALUDE_one_absent_out_of_three_l3452_345210

def probability_absent : ℚ := 1 / 40

def probability_present : ℚ := 1 - probability_absent

def probability_one_absent_two_present : ℚ :=
  3 * probability_absent * probability_present * probability_present

theorem one_absent_out_of_three (ε : ℚ) (h : ε > 0) :
  |probability_one_absent_two_present - 4563 / 64000| < ε :=
sorry

end NUMINAMATH_CALUDE_one_absent_out_of_three_l3452_345210


namespace NUMINAMATH_CALUDE_room_area_from_carpet_l3452_345244

/-- Given a rectangular carpet covering 30% of a room's floor area, 
    if the carpet measures 4 feet by 9 feet, 
    then the total floor area of the room is 120 square feet. -/
theorem room_area_from_carpet (carpet_length carpet_width : ℝ) 
  (carpet_coverage_percent : ℝ) (total_area : ℝ) :
  carpet_length = 4 →
  carpet_width = 9 →
  carpet_coverage_percent = 30 →
  carpet_length * carpet_width / total_area = carpet_coverage_percent / 100 →
  total_area = 120 :=
by sorry

end NUMINAMATH_CALUDE_room_area_from_carpet_l3452_345244


namespace NUMINAMATH_CALUDE_white_triangle_pairs_count_l3452_345211

/-- Represents the number of triangles of each color in each half of the figure -/
structure TriangleCounts where
  red : Nat
  blue : Nat
  white : Nat

/-- Represents the number of coinciding pairs of each type when the figure is folded -/
structure CoincidingPairs where
  red_red : Nat
  blue_blue : Nat
  red_white : Nat
  white_white : Nat

/-- The main theorem statement -/
theorem white_triangle_pairs_count 
  (counts : TriangleCounts)
  (pairs : CoincidingPairs)
  (h1 : counts.red = 3)
  (h2 : counts.blue = 5)
  (h3 : counts.white = 8)
  (h4 : pairs.red_red = 2)
  (h5 : pairs.blue_blue = 3)
  (h6 : pairs.red_white = 2) :
  pairs.white_white = 5 := by
  sorry

end NUMINAMATH_CALUDE_white_triangle_pairs_count_l3452_345211


namespace NUMINAMATH_CALUDE_rectangle_area_below_line_l3452_345206

/-- Given a rectangle bounded by y = 2a, y = -b, x = -2c, and x = d, 
    where a, b, c, and d are positive real numbers, and a line y = x + a 
    intersecting the rectangle, this theorem states the area of the 
    rectangle below the line. -/
theorem rectangle_area_below_line 
  (a b c d : ℝ) 
  (ha : a > 0) (hb : b > 0) (hc : c > 0) (hd : d > 0) :
  let rectangle_area := (2*a + b) * (d + 2*c)
  let triangle_area := (1/2) * (d + 2*c + b + a) * (a + b + 2*c)
  rectangle_area - triangle_area = 
    (2*a + b) * (d + 2*c) - (1/2) * (d + 2*c + b + a) * (a + b + 2*c) := by
  sorry

end NUMINAMATH_CALUDE_rectangle_area_below_line_l3452_345206


namespace NUMINAMATH_CALUDE_min_value_sum_reciprocal_squares_l3452_345265

/-- Two circles in the xy-plane -/
structure TwoCircles where
  a : ℝ
  b : ℝ
  h1 : a ≠ 0
  h2 : b ≠ 0

/-- The property that the two circles have exactly three common tangents -/
def has_three_common_tangents (c : TwoCircles) : Prop :=
  ∃ (x y : ℝ), x^2 + y^2 + 2 * c.a * x + c.a^2 - 4 = 0 ∧
                x^2 + y^2 - 4 * c.b * y - 1 + 4 * c.b^2 = 0

/-- The theorem stating the minimum value of 1/a^2 + 1/b^2 -/
theorem min_value_sum_reciprocal_squares (c : TwoCircles) 
  (h : has_three_common_tangents c) : 
  (∀ ε > 0, ∃ (c' : TwoCircles), has_three_common_tangents c' ∧ 
    1 / c'.a^2 + 1 / c'.b^2 < 1 + ε) ∧
  (∀ (c' : TwoCircles), has_three_common_tangents c' → 
    1 / c'.a^2 + 1 / c'.b^2 ≥ 1) :=
sorry

end NUMINAMATH_CALUDE_min_value_sum_reciprocal_squares_l3452_345265


namespace NUMINAMATH_CALUDE_minimal_plums_after_tricks_l3452_345270

/-- Represents the quantities of fruits -/
structure Fruits :=
  (apples : ℕ)
  (pears : ℕ)
  (plums : ℕ)

/-- Represents a trick that can be performed -/
inductive Trick
| trick1 : Trick  -- Switch 1 plum and 1 pear with 2 apples
| trick2 : Trick  -- Switch 1 pear and 1 apple with 3 plums
| trick3 : Trick  -- Switch 1 apple and 1 plum with 4 pears

/-- Applies a trick to the current quantities -/
def applyTrick (f : Fruits) (t : Trick) : Fruits :=
  match t with
  | Trick.trick1 => Fruits.mk (f.apples + 2) (f.pears - 1) (f.plums - 1)
  | Trick.trick2 => Fruits.mk (f.apples - 1) (f.pears - 1) (f.plums + 3)
  | Trick.trick3 => Fruits.mk (f.apples - 1) (f.pears + 4) (f.plums - 1)

/-- Applies a sequence of tricks to the initial quantities -/
def applyTricks (initial : Fruits) (tricks : List Trick) : Fruits :=
  tricks.foldl applyTrick initial

theorem minimal_plums_after_tricks (initial : Fruits) 
  (h_initial : initial = Fruits.mk 2012 2012 2012) :
  ∃ (tricks : List Trick), 
    let final := applyTricks initial tricks
    final.apples = 2012 ∧ 
    final.pears = 2012 ∧ 
    final.plums > 2012 ∧
    ∀ (other_tricks : List Trick), 
      let other_final := applyTricks initial other_tricks
      other_final.apples = 2012 → 
      other_final.pears = 2012 → 
      other_final.plums ≥ 2025 :=
by
  sorry

end NUMINAMATH_CALUDE_minimal_plums_after_tricks_l3452_345270


namespace NUMINAMATH_CALUDE_max_movies_watched_l3452_345290

def movie_duration : ℕ := 90
def tuesday_watch_time : ℕ := 270

theorem max_movies_watched (wednesday_multiplier : ℕ) (h : wednesday_multiplier = 2) :
  let tuesday_movies := tuesday_watch_time / movie_duration
  let wednesday_movies := wednesday_multiplier * tuesday_movies
  tuesday_movies + wednesday_movies = 9 :=
by sorry

end NUMINAMATH_CALUDE_max_movies_watched_l3452_345290


namespace NUMINAMATH_CALUDE_triangle_side_length_l3452_345249

theorem triangle_side_length (a b c : ℝ) (A : ℝ) :
  a = 2 →
  c = 2 * Real.sqrt 3 →
  Real.cos A = Real.sqrt 3 / 2 →
  b < c →
  b^2 - 6*b + 8 = 0 →
  b = 2 := by sorry

end NUMINAMATH_CALUDE_triangle_side_length_l3452_345249


namespace NUMINAMATH_CALUDE_correct_categorization_l3452_345247

def numbers : List ℚ := [15, -3/8, 0, 0.15, -30, -12.8, 22/5, 20]

def is_integer (q : ℚ) : Prop := ∃ (n : ℤ), q = n
def is_fraction (q : ℚ) : Prop := ¬(is_integer q)
def is_positive_integer (q : ℚ) : Prop := is_integer q ∧ q > 0
def is_negative_fraction (q : ℚ) : Prop := is_fraction q ∧ q < 0
def is_non_negative (q : ℚ) : Prop := q ≥ 0

def integer_set : Set ℚ := {q ∈ numbers | is_integer q}
def fraction_set : Set ℚ := {q ∈ numbers | is_fraction q}
def positive_integer_set : Set ℚ := {q ∈ numbers | is_positive_integer q}
def negative_fraction_set : Set ℚ := {q ∈ numbers | is_negative_fraction q}
def non_negative_set : Set ℚ := {q ∈ numbers | is_non_negative q}

theorem correct_categorization :
  integer_set = {15, 0, -30, 20} ∧
  fraction_set = {-3/8, 0.15, -12.8, 22/5} ∧
  positive_integer_set = {15, 20} ∧
  negative_fraction_set = {-3/8, -12.8} ∧
  non_negative_set = {15, 0, 0.15, 22/5, 20} := by
  sorry

end NUMINAMATH_CALUDE_correct_categorization_l3452_345247


namespace NUMINAMATH_CALUDE_total_leaves_l3452_345269

/-- The number of ferns Karen hangs around her house. -/
def num_ferns : ℕ := 12

/-- The number of fronds each fern has. -/
def fronds_per_fern : ℕ := 15

/-- The number of leaves each frond has. -/
def leaves_per_frond : ℕ := 45

/-- Theorem stating the total number of leaves on all ferns. -/
theorem total_leaves : num_ferns * fronds_per_fern * leaves_per_frond = 8100 := by
  sorry

end NUMINAMATH_CALUDE_total_leaves_l3452_345269


namespace NUMINAMATH_CALUDE_pencils_given_to_dorothy_l3452_345280

/-- Given that Josh had a certain number of pencils initially and was left with
    a smaller number after giving some to Dorothy, prove that the number of
    pencils he gave to Dorothy is the difference between the initial and final amounts. -/
theorem pencils_given_to_dorothy
  (initial_pencils : ℕ)
  (remaining_pencils : ℕ)
  (h1 : initial_pencils = 142)
  (h2 : remaining_pencils = 111)
  (h3 : remaining_pencils < initial_pencils) :
  initial_pencils - remaining_pencils = 31 :=
by sorry

end NUMINAMATH_CALUDE_pencils_given_to_dorothy_l3452_345280


namespace NUMINAMATH_CALUDE_cloth_cost_price_l3452_345208

/-- Given a cloth sale scenario, prove the cost price per meter -/
theorem cloth_cost_price
  (total_meters : ℕ)
  (selling_price : ℕ)
  (profit_per_meter : ℕ)
  (h1 : total_meters = 85)
  (h2 : selling_price = 8925)
  (h3 : profit_per_meter = 35) :
  (selling_price - total_meters * profit_per_meter) / total_meters = 70 := by
  sorry

end NUMINAMATH_CALUDE_cloth_cost_price_l3452_345208


namespace NUMINAMATH_CALUDE_distinct_lines_count_l3452_345277

/-- Represents a 4-by-4 grid of lattice points -/
def Grid := Fin 4 × Fin 4

/-- A line in the grid is defined by two distinct points it passes through -/
def Line := { pair : Grid × Grid // pair.1 ≠ pair.2 }

/-- Counts the number of distinct lines passing through at least two points in the grid -/
def countDistinctLines : Nat :=
  sorry

/-- The main theorem stating that the number of distinct lines is 84 -/
theorem distinct_lines_count : countDistinctLines = 84 :=
  sorry

end NUMINAMATH_CALUDE_distinct_lines_count_l3452_345277


namespace NUMINAMATH_CALUDE_angle_C_value_l3452_345298

-- Define a triangle ABC
structure Triangle :=
  (A B C : Real)
  (a b c : Real)

-- Define the theorem
theorem angle_C_value (t : Triangle) 
  (h1 : t.b = Real.sqrt 2)
  (h2 : t.c = 1)
  (h3 : t.B = π / 4) : 
  t.C = π / 6 := by
  sorry

end NUMINAMATH_CALUDE_angle_C_value_l3452_345298


namespace NUMINAMATH_CALUDE_baseball_team_selection_l3452_345221

/-- The number of ways to choose k items from n items -/
def choose (n k : ℕ) : ℕ := Nat.choose n k

/-- The total number of players in the team -/
def total_players : ℕ := 18

/-- The number of quadruplets that must be included -/
def quadruplets : ℕ := 4

/-- The number of starters to be chosen -/
def starters : ℕ := 9

theorem baseball_team_selection :
  choose (total_players - quadruplets) (starters - quadruplets) = 2002 := by
  sorry

end NUMINAMATH_CALUDE_baseball_team_selection_l3452_345221


namespace NUMINAMATH_CALUDE_chess_playoff_orders_l3452_345287

/-- Represents the structure of a chess playoff tournament --/
structure ChessPlayoff where
  numPlayers : Nat
  numMatches : Nat
  firstMatchPlayers : Fin 3 × Fin 3
  secondMatchPlayer : Fin 3

/-- Calculates the number of possible prize orders in a chess playoff tournament --/
def numPossibleOrders (tournament : ChessPlayoff) : Nat :=
  2^tournament.numMatches

/-- Theorem stating that the number of possible prize orders in the given tournament structure is 4 --/
theorem chess_playoff_orders (tournament : ChessPlayoff) 
  (h1 : tournament.numPlayers = 3)
  (h2 : tournament.numMatches = 2)
  (h3 : tournament.firstMatchPlayers = (⟨2, by norm_num⟩, ⟨1, by norm_num⟩))
  (h4 : tournament.secondMatchPlayer = ⟨0, by norm_num⟩) :
  numPossibleOrders tournament = 4 := by
  sorry


end NUMINAMATH_CALUDE_chess_playoff_orders_l3452_345287


namespace NUMINAMATH_CALUDE_no_linear_factor_with_integer_coefficients_l3452_345278

theorem no_linear_factor_with_integer_coefficients :
  ∀ (a b c d : ℤ), (∀ (x y z : ℝ), 
    a*x + b*y + c*z + d ≠ 0 ∨ 
    x^2 - y^2 - z^2 + 3*y*z + x + 2*y - z ≠ (a*x + b*y + c*z + d) * 
      ((x^2 - y^2 - z^2 + 3*y*z + x + 2*y - z) / (a*x + b*y + c*z + d))) :=
by sorry

end NUMINAMATH_CALUDE_no_linear_factor_with_integer_coefficients_l3452_345278


namespace NUMINAMATH_CALUDE_intersection_right_triangle_l3452_345217

/-- Given a line and a circle in the Cartesian plane, if they intersect at two points
    forming a right triangle with the circle's center, then the parameter of the line
    and circle equations must be -1. -/
theorem intersection_right_triangle (a : ℝ) : 
  (∃ (A B : ℝ × ℝ), 
    (a * A.1 + A.2 - 2 = 0 ∧ (A.1 - 1)^2 + (A.2 - a)^2 = 16) ∧
    (a * B.1 + B.2 - 2 = 0 ∧ (B.1 - 1)^2 + (B.2 - a)^2 = 16) ∧
    A ≠ B ∧
    let C : ℝ × ℝ := (1, a)
    (A.1 - C.1) * (B.1 - C.1) + (A.2 - C.2) * (B.2 - C.2) = 0) →
  a = -1 := by
sorry


end NUMINAMATH_CALUDE_intersection_right_triangle_l3452_345217


namespace NUMINAMATH_CALUDE_polynomial_integer_roots_l3452_345299

def polynomial (x a : ℤ) : ℤ := x^3 + 5*x^2 + a*x + 12

def has_integer_root (a : ℤ) : Prop :=
  ∃ x : ℤ, polynomial x a = 0

def valid_a_values : Set ℤ := {-18, 16, -20, 12, -16, 8, -11, 5, -4, 2, 0, -1}

theorem polynomial_integer_roots :
  ∀ a : ℤ, has_integer_root a ↔ a ∈ valid_a_values := by sorry

end NUMINAMATH_CALUDE_polynomial_integer_roots_l3452_345299


namespace NUMINAMATH_CALUDE_cos_sin_sum_l3452_345226

theorem cos_sin_sum (φ : Real) (h : Real.cos (π / 2 + φ) = Real.sqrt 3 / 2) :
  Real.cos (3 * π / 2 - φ) + Real.sin (φ - π) = Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_cos_sin_sum_l3452_345226


namespace NUMINAMATH_CALUDE_complex_modulus_equation_l3452_345239

theorem complex_modulus_equation (t : ℝ) (h1 : t > 0) :
  Complex.abs (Complex.mk (-3) t) = 5 * Real.sqrt 2 → t = Real.sqrt 41 := by
  sorry

end NUMINAMATH_CALUDE_complex_modulus_equation_l3452_345239


namespace NUMINAMATH_CALUDE_corn_preference_percentage_l3452_345294

theorem corn_preference_percentage (peas carrots corn : ℕ) : 
  peas = 6 → carrots = 9 → corn = 5 → 
  (corn : ℚ) / (peas + carrots + corn : ℚ) * 100 = 25 := by
  sorry

end NUMINAMATH_CALUDE_corn_preference_percentage_l3452_345294


namespace NUMINAMATH_CALUDE_parabola_area_l3452_345235

-- Define the two parabolas
def f (x : ℝ) : ℝ := x^2
def g (x : ℝ) : ℝ := 8 - x^2

-- Define the region
def R : Set ℝ := {x | -2 ≤ x ∧ x ≤ 2}

-- State the theorem
theorem parabola_area : 
  (∫ (x : ℝ) in R, g x - f x) = 64/3 := by sorry

end NUMINAMATH_CALUDE_parabola_area_l3452_345235


namespace NUMINAMATH_CALUDE_initial_ratio_is_four_to_one_l3452_345228

/-- Represents a mixture of milk and water -/
structure Mixture where
  milk : ℝ
  water : ℝ

/-- The initial mixture before adding water -/
def initial_mixture : Mixture :=
  { milk := 48, water := 12 }

/-- The final mixture after adding 60 litres of water -/
def final_mixture : Mixture :=
  { milk := initial_mixture.milk, water := initial_mixture.water + 60 }

/-- The total volume of the initial mixture -/
def initial_volume : ℝ := 60

theorem initial_ratio_is_four_to_one :
  initial_mixture.milk / initial_mixture.water = 4 ∧
  initial_mixture.milk + initial_mixture.water = initial_volume ∧
  final_mixture.milk / final_mixture.water = 1 / 2 := by
  sorry

#check initial_ratio_is_four_to_one

end NUMINAMATH_CALUDE_initial_ratio_is_four_to_one_l3452_345228


namespace NUMINAMATH_CALUDE_investment_rate_calculation_l3452_345264

theorem investment_rate_calculation 
  (total_investment : ℝ) 
  (first_investment : ℝ) 
  (second_investment : ℝ) 
  (first_rate : ℝ) 
  (second_rate : ℝ) 
  (desired_income : ℝ) :
  total_investment = 15000 →
  first_investment = 5000 →
  second_investment = 6000 →
  first_rate = 0.03 →
  second_rate = 0.045 →
  desired_income = 800 →
  let remaining_investment := total_investment - first_investment - second_investment
  let income_from_first := first_investment * first_rate
  let income_from_second := second_investment * second_rate
  let remaining_income := desired_income - income_from_first - income_from_second
  remaining_income / remaining_investment = 0.095 := by sorry

end NUMINAMATH_CALUDE_investment_rate_calculation_l3452_345264


namespace NUMINAMATH_CALUDE_cube_difference_fifty_l3452_345295

/-- The sum of cubes of the first n positive integers -/
def sumOfPositiveCubes (n : ℕ) : ℕ := (n * (n + 1) / 2) ^ 2

/-- The sum of cubes of the first n negative integers -/
def sumOfNegativeCubes (n : ℕ) : ℤ := -(sumOfPositiveCubes n)

/-- The difference between the sum of cubes of the first n positive integers
    and the sum of cubes of the first n negative integers -/
def cubeDifference (n : ℕ) : ℤ := (sumOfPositiveCubes n : ℤ) - sumOfNegativeCubes n

theorem cube_difference_fifty : cubeDifference 50 = 3251250 := by sorry

end NUMINAMATH_CALUDE_cube_difference_fifty_l3452_345295


namespace NUMINAMATH_CALUDE_isosceles_triangle_not_unique_l3452_345207

/-- An isosceles triangle -/
structure IsoscelesTriangle where
  base_angle : ℝ
  other_angle : ℝ

/-- A function that attempts to construct an isosceles triangle from given angles -/
noncomputable def construct_isosceles_triangle (ba oa : ℝ) : Option IsoscelesTriangle := sorry

/-- Theorem stating that an isosceles triangle is not uniquely determined by a base angle and another angle -/
theorem isosceles_triangle_not_unique :
  ∃ (ba₁ oa₁ ba₂ oa₂ : ℝ),
    ba₁ = ba₂ ∧
    oa₁ = oa₂ ∧
    construct_isosceles_triangle ba₁ oa₁ ≠ construct_isosceles_triangle ba₂ oa₂ :=
sorry

end NUMINAMATH_CALUDE_isosceles_triangle_not_unique_l3452_345207


namespace NUMINAMATH_CALUDE_exam_average_marks_l3452_345242

theorem exam_average_marks (total_boys : ℕ) (total_avg : ℚ) (passed_avg : ℚ) (passed_boys : ℕ) :
  total_boys = 120 →
  total_avg = 35 →
  passed_avg = 39 →
  passed_boys = 100 →
  let failed_boys := total_boys - passed_boys
  let total_marks := total_avg * total_boys
  let passed_marks := passed_avg * passed_boys
  let failed_marks := total_marks - passed_marks
  (failed_marks / failed_boys : ℚ) = 15 := by
  sorry

end NUMINAMATH_CALUDE_exam_average_marks_l3452_345242


namespace NUMINAMATH_CALUDE_five_thousand_five_hundred_scientific_notation_l3452_345215

/-- Scientific notation representation of a number -/
structure ScientificNotation where
  coefficient : ℝ
  exponent : ℤ
  one_le_coeff_lt_ten : 1 ≤ coefficient ∧ coefficient < 10

/-- Converts a real number to scientific notation -/
def toScientificNotation (x : ℝ) : ScientificNotation :=
  sorry

theorem five_thousand_five_hundred_scientific_notation :
  toScientificNotation 5500 = ScientificNotation.mk 5.5 3 (by norm_num) :=
sorry

end NUMINAMATH_CALUDE_five_thousand_five_hundred_scientific_notation_l3452_345215


namespace NUMINAMATH_CALUDE_line_slope_l3452_345286

/-- The slope of a line given by the equation 3y + 4x = 12 is -4/3 -/
theorem line_slope (x y : ℝ) : 3 * y + 4 * x = 12 → (y - 4) / (x - 0) = -4/3 := by
  sorry

end NUMINAMATH_CALUDE_line_slope_l3452_345286


namespace NUMINAMATH_CALUDE_mike_work_hours_l3452_345262

def wash_time : ℕ := 10
def oil_change_time : ℕ := 15
def tire_change_time : ℕ := 30
def cars_washed : ℕ := 9
def cars_oil_changed : ℕ := 6
def tire_sets_changed : ℕ := 2

theorem mike_work_hours : 
  (cars_washed * wash_time + cars_oil_changed * oil_change_time + tire_sets_changed * tire_change_time) / 60 = 4 := by
  sorry

end NUMINAMATH_CALUDE_mike_work_hours_l3452_345262


namespace NUMINAMATH_CALUDE_oblique_projection_properties_l3452_345250

/-- Represents a shape in 2D space -/
inductive Shape
  | Triangle
  | Parallelogram
  | Square
  | Rhombus

/-- Represents the result of an oblique projection on a shape -/
def obliqueProjection (s : Shape) : Shape :=
  match s with
  | Shape.Triangle => Shape.Triangle
  | Shape.Parallelogram => Shape.Parallelogram
  | Shape.Square => Shape.Parallelogram  -- Assuming it becomes a general parallelogram
  | Shape.Rhombus => Shape.Parallelogram -- Assuming it becomes a general parallelogram

theorem oblique_projection_properties :
  (∀ s : Shape, s = Shape.Triangle → obliqueProjection s = Shape.Triangle) ∧
  (∀ s : Shape, s = Shape.Parallelogram → obliqueProjection s = Shape.Parallelogram) ∧
  (∃ s : Shape, s = Shape.Square ∧ obliqueProjection s ≠ Shape.Square) ∧
  (∃ s : Shape, s = Shape.Rhombus ∧ obliqueProjection s ≠ Shape.Rhombus) :=
by
  sorry

#check oblique_projection_properties

end NUMINAMATH_CALUDE_oblique_projection_properties_l3452_345250


namespace NUMINAMATH_CALUDE_ratio_chain_l3452_345268

theorem ratio_chain (a b c d e f : ℝ) 
  (h1 : a / b = 1 / 4)
  (h2 : b / c = 13 / 9)
  (h3 : c / d = 5 / 13)
  (h4 : d / e = 2 / 3)
  (h5 : e / f = 7 / 5) :
  a / f = 7 / 6 := by
sorry

end NUMINAMATH_CALUDE_ratio_chain_l3452_345268


namespace NUMINAMATH_CALUDE_prob_all_fives_four_dice_l3452_345236

-- Define a standard six-sided die
def standard_die := Finset.range 6

-- Define the probability of getting a specific number on a standard die
def prob_specific_number (die : Finset Nat) : ℚ :=
  1 / die.card

-- Define the number of dice
def num_dice : Nat := 4

-- Define the desired outcome (all fives)
def all_fives (n : Nat) : Bool := n = 5

-- Theorem: The probability of getting all fives on four standard six-sided dice is 1/1296
theorem prob_all_fives_four_dice : 
  (prob_specific_number standard_die) ^ num_dice = 1 / 1296 :=
sorry

end NUMINAMATH_CALUDE_prob_all_fives_four_dice_l3452_345236


namespace NUMINAMATH_CALUDE_inconsistent_division_problem_l3452_345245

theorem inconsistent_division_problem 
  (x y q : ℕ+) 
  (h1 : x = 9 * y + 4)
  (h2 : 2 * x = 7 * q + 1)
  (h3 : 5 * y - x = 3) :
  False :=
sorry

end NUMINAMATH_CALUDE_inconsistent_division_problem_l3452_345245


namespace NUMINAMATH_CALUDE_isosceles_right_triangle_vector_sum_l3452_345275

/-- An isosceles right triangle with hypotenuse of length 6 -/
structure IsoscelesRightTriangle where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ
  isRight : (B.1 - A.1) * (C.1 - A.1) + (B.2 - A.2) * (C.2 - A.2) = 0
  isIsosceles : (B.1 - A.1)^2 + (B.2 - A.2)^2 = (C.1 - A.1)^2 + (C.2 - A.2)^2
  hypotenuseLength : (C.1 - B.1)^2 + (C.2 - B.2)^2 = 36

def dotProduct (v w : ℝ × ℝ) : ℝ :=
  v.1 * w.1 + v.2 * w.2

def vectorSum (t : IsoscelesRightTriangle) : ℝ :=
  let AB := (t.B.1 - t.A.1, t.B.2 - t.A.2)
  let AC := (t.C.1 - t.A.1, t.C.2 - t.A.2)
  let BC := (t.C.1 - t.B.1, t.C.2 - t.B.2)
  let BA := (-AB.1, -AB.2)
  let CA := (-AC.1, -AC.2)
  let CB := (-BC.1, -BC.2)
  dotProduct AB AC + dotProduct BC BA + dotProduct CA CB

theorem isosceles_right_triangle_vector_sum (t : IsoscelesRightTriangle) :
  vectorSum t = 36 := by
  sorry

end NUMINAMATH_CALUDE_isosceles_right_triangle_vector_sum_l3452_345275


namespace NUMINAMATH_CALUDE_milford_lake_algae_increase_l3452_345240

/-- The increase in algae plants in Milford Lake -/
def algae_increase (original current : ℕ) : ℕ := current - original

/-- Theorem stating the increase in algae plants in Milford Lake -/
theorem milford_lake_algae_increase :
  algae_increase 809 3263 = 2454 := by
  sorry

end NUMINAMATH_CALUDE_milford_lake_algae_increase_l3452_345240


namespace NUMINAMATH_CALUDE_ellipse_and_chord_properties_l3452_345238

-- Define the ellipse C₁
def ellipse (x y : ℝ) : Prop := x^2 / 6 + y^2 / 2 = 1

-- Define the line l₂
def line (x y : ℝ) : Prop := y = Real.sqrt 3 * (x - 2)

-- Define the intersection points
def intersection_points (x₁ y₁ x₂ y₂ : ℝ) : Prop :=
  ellipse x₁ y₁ ∧ ellipse x₂ y₂ ∧ line x₁ y₁ ∧ line x₂ y₂

theorem ellipse_and_chord_properties :
  -- The ellipse equation is correct
  (∀ x y, ellipse x y ↔ x^2 / 6 + y^2 / 2 = 1) ∧
  -- The chord length is correct
  (∀ x₁ y₁ x₂ y₂, intersection_points x₁ y₁ x₂ y₂ →
    Real.sqrt ((x₁ - x₂)^2 + (y₁ - y₂)^2) = 4 * Real.sqrt 6 / 5) :=
by sorry

end NUMINAMATH_CALUDE_ellipse_and_chord_properties_l3452_345238


namespace NUMINAMATH_CALUDE_abs_negative_two_thirds_equals_two_thirds_l3452_345273

theorem abs_negative_two_thirds_equals_two_thirds : 
  |(-2 : ℚ) / 3| = 2 / 3 := by sorry

end NUMINAMATH_CALUDE_abs_negative_two_thirds_equals_two_thirds_l3452_345273


namespace NUMINAMATH_CALUDE_hockey_players_count_l3452_345267

theorem hockey_players_count (cricket football softball total : ℕ) 
  (h_cricket : cricket = 16)
  (h_football : football = 18)
  (h_softball : softball = 13)
  (h_total : total = 59)
  : total - (cricket + football + softball) = 12 := by
  sorry

end NUMINAMATH_CALUDE_hockey_players_count_l3452_345267


namespace NUMINAMATH_CALUDE_right_triangle_perimeter_l3452_345283

theorem right_triangle_perimeter (a b c : ℝ) : 
  a > 0 → b > 0 → c > 0 →
  a * b / 2 = 150 →
  a = 15 →
  a^2 + b^2 = c^2 →
  a + b + c = 60 := by
sorry

end NUMINAMATH_CALUDE_right_triangle_perimeter_l3452_345283


namespace NUMINAMATH_CALUDE_f_properties_l3452_345255

def f (x : ℝ) : ℝ := x^3 - 3*x^2

theorem f_properties :
  (∀ x y, x < y ∧ y < 0 → f x < f y) ∧
  (∀ x y, 2 < x ∧ x < y → f x < f y) ∧
  (∀ x y, 0 < x ∧ x < y ∧ y < 2 → f x > f y) ∧
  (∀ x, f x ≤ 0) ∧
  (f 0 = 0) ∧
  (∀ x, f x ≥ -4) ∧
  (f 2 = -4) :=
by sorry

end NUMINAMATH_CALUDE_f_properties_l3452_345255


namespace NUMINAMATH_CALUDE_eight_solutions_for_triple_f_l3452_345218

def f (x : ℝ) : ℝ := |1 - 2*x|

theorem eight_solutions_for_triple_f (x : ℝ) :
  x ∈ Set.Icc 0 1 →
  ∃! (solutions : Finset ℝ),
    (∀ s ∈ solutions, f (f (f s)) = (1/2) * s) ∧
    Finset.card solutions = 8 :=
sorry

end NUMINAMATH_CALUDE_eight_solutions_for_triple_f_l3452_345218


namespace NUMINAMATH_CALUDE_airport_distance_airport_distance_proof_l3452_345274

theorem airport_distance : ℝ → Prop :=
  fun d : ℝ =>
    let initial_speed : ℝ := 45
    let speed_increase : ℝ := 20
    let late_time : ℝ := 0.75  -- 45 minutes in hours
    let early_time : ℝ := 0.25  -- 15 minutes in hours
    let t : ℝ := (d / initial_speed) - late_time  -- Time if he continued at initial speed
    
    (d = initial_speed * (t + late_time)) ∧
    (d - initial_speed = (initial_speed + speed_increase) * (t - early_time)) →
    d = 61.875

-- The proof would go here
theorem airport_distance_proof : airport_distance 61.875 := by
  sorry

end NUMINAMATH_CALUDE_airport_distance_airport_distance_proof_l3452_345274

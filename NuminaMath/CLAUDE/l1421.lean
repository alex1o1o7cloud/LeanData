import Mathlib

namespace NUMINAMATH_CALUDE_gift_spending_theorem_l1421_142179

def num_siblings : ℕ := 3
def num_parents : ℕ := 2
def cost_per_sibling : ℕ := 30
def cost_per_parent : ℕ := 30

def total_cost : ℕ := num_siblings * cost_per_sibling + num_parents * cost_per_parent

theorem gift_spending_theorem : total_cost = 150 := by
  sorry

end NUMINAMATH_CALUDE_gift_spending_theorem_l1421_142179


namespace NUMINAMATH_CALUDE_florist_roses_problem_l1421_142124

/-- A florist problem involving roses -/
theorem florist_roses_problem (initial_roses : ℕ) (picked_roses : ℕ) (final_roses : ℕ) :
  initial_roses = 50 →
  picked_roses = 21 →
  final_roses = 56 →
  ∃ (sold_roses : ℕ), initial_roses - sold_roses + picked_roses = final_roses ∧ sold_roses = 15 :=
by sorry

end NUMINAMATH_CALUDE_florist_roses_problem_l1421_142124


namespace NUMINAMATH_CALUDE_M_intersect_N_eq_N_l1421_142130

def M : Set Int := {-1, 0, 1}

def N : Set Int := {x | ∃ a b, a ∈ M ∧ b ∈ M ∧ a ≠ b ∧ x = a * b}

theorem M_intersect_N_eq_N : M ∩ N = N := by sorry

end NUMINAMATH_CALUDE_M_intersect_N_eq_N_l1421_142130


namespace NUMINAMATH_CALUDE_power_function_through_point_l1421_142186

/-- A power function passing through (3, √3) evaluates to 1/2 at x = 1/4 -/
theorem power_function_through_point (f : ℝ → ℝ) (α : ℝ) :
  (∀ x > 0, f x = x^α) →  -- f is a power function
  f 3 = Real.sqrt 3 →     -- f passes through (3, √3)
  f (1/4) = 1/2 := by
sorry

end NUMINAMATH_CALUDE_power_function_through_point_l1421_142186


namespace NUMINAMATH_CALUDE_complex_fraction_simplification_l1421_142167

theorem complex_fraction_simplification :
  (2 : ℂ) / (1 + Complex.I) = 1 - Complex.I := by sorry

end NUMINAMATH_CALUDE_complex_fraction_simplification_l1421_142167


namespace NUMINAMATH_CALUDE_circle_area_equality_l1421_142116

theorem circle_area_equality (r₁ r₂ : ℝ) (h₁ : r₁ = 25) (h₂ : r₂ = 17) :
  ∃ r : ℝ, π * r^2 = π * r₁^2 - π * r₂^2 ∧ r = 4 * Real.sqrt 21 := by
  sorry

end NUMINAMATH_CALUDE_circle_area_equality_l1421_142116


namespace NUMINAMATH_CALUDE_complex_expression_simplification_l1421_142141

theorem complex_expression_simplification :
  ∀ (i : ℂ), i^2 = -1 →
  7 * (4 - i) + 4 * i * (7 - i) + 2 * (3 + i) = 38 + 23 * i := by
sorry

end NUMINAMATH_CALUDE_complex_expression_simplification_l1421_142141


namespace NUMINAMATH_CALUDE_watch_cost_price_l1421_142148

theorem watch_cost_price (selling_price loss_percent gain_percent additional_amount : ℝ) 
  (h1 : selling_price = (1 - loss_percent / 100) * 1500)
  (h2 : selling_price + additional_amount = (1 + gain_percent / 100) * 1500)
  (h3 : loss_percent = 10)
  (h4 : gain_percent = 5)
  (h5 : additional_amount = 225) : 
  1500 = 1500 := by sorry

end NUMINAMATH_CALUDE_watch_cost_price_l1421_142148


namespace NUMINAMATH_CALUDE_pgcd_and_divisibility_properties_l1421_142103

/-- Given a ≥ 2 and m ≥ n ≥ 1, prove three statements about PGCD and divisibility -/
theorem pgcd_and_divisibility_properties (a m n : ℕ) (ha : a ≥ 2) (hmn : m ≥ n) (hn : n ≥ 1) :
  (gcd (a^m - 1) (a^n - 1) = gcd (a^(m-n) - 1) (a^n - 1)) ∧
  (gcd (a^m - 1) (a^n - 1) = a^(gcd m n) - 1) ∧
  ((a^m - 1) ∣ (a^n - 1) ↔ m ∣ n) := by
  sorry


end NUMINAMATH_CALUDE_pgcd_and_divisibility_properties_l1421_142103


namespace NUMINAMATH_CALUDE_vector_norm_difference_l1421_142171

theorem vector_norm_difference (a b : ℝ × ℝ) :
  (‖a‖ = 2) → (‖b‖ = 1) → (‖a + b‖ = Real.sqrt 3) → ‖a - b‖ = Real.sqrt 7 := by
  sorry

end NUMINAMATH_CALUDE_vector_norm_difference_l1421_142171


namespace NUMINAMATH_CALUDE_count_five_digit_numbers_without_five_is_52488_l1421_142142

/-- The count of five-digit numbers not containing the digit 5 -/
def count_five_digit_numbers_without_five : ℕ :=
  8 * 9^4

/-- Theorem stating that the count of five-digit numbers not containing the digit 5 is 52488 -/
theorem count_five_digit_numbers_without_five_is_52488 :
  count_five_digit_numbers_without_five = 52488 := by
  sorry

end NUMINAMATH_CALUDE_count_five_digit_numbers_without_five_is_52488_l1421_142142


namespace NUMINAMATH_CALUDE_thirty_percent_of_hundred_l1421_142157

theorem thirty_percent_of_hundred : (30 : ℝ) / 100 * 100 = 30 := by
  sorry

end NUMINAMATH_CALUDE_thirty_percent_of_hundred_l1421_142157


namespace NUMINAMATH_CALUDE_max_value_of_trig_function_l1421_142109

theorem max_value_of_trig_function :
  let f : ℝ → ℝ := λ x ↦ Real.cos (2 * x) + 6 * Real.cos (Real.pi / 2 - x)
  ∃ M : ℝ, M = 5 ∧ ∀ x : ℝ, f x ≤ M :=
by sorry

end NUMINAMATH_CALUDE_max_value_of_trig_function_l1421_142109


namespace NUMINAMATH_CALUDE_intersection_point_unique_l1421_142119

/-- The line equation -/
def line_equation (x y z : ℝ) : Prop :=
  (x - 1) / 6 = (y - 3) / 1 ∧ (y - 3) / 1 = (z + 5) / 3

/-- The plane equation -/
def plane_equation (x y z : ℝ) : Prop :=
  3 * x - 2 * y + 5 * z - 3 = 0

/-- The intersection point -/
def intersection_point : ℝ × ℝ × ℝ := (7, 4, -2)

/-- Theorem stating that the intersection_point is the unique point satisfying both equations -/
theorem intersection_point_unique :
  line_equation intersection_point.1 intersection_point.2.1 intersection_point.2.2 ∧
  plane_equation intersection_point.1 intersection_point.2.1 intersection_point.2.2 ∧
  ∀ x y z : ℝ, line_equation x y z ∧ plane_equation x y z → (x, y, z) = intersection_point :=
by sorry


end NUMINAMATH_CALUDE_intersection_point_unique_l1421_142119


namespace NUMINAMATH_CALUDE_root_cubic_value_l1421_142120

theorem root_cubic_value (m : ℝ) (h : m^2 + m - 1 = 0) : m^3 + 2*m^2 + 2014 = 2015 := by
  sorry

end NUMINAMATH_CALUDE_root_cubic_value_l1421_142120


namespace NUMINAMATH_CALUDE_ellipse_circle_tangent_relation_l1421_142191

-- Define the ellipse
def is_on_ellipse (x y a b : ℝ) : Prop := x^2 / a^2 + y^2 / b^2 = 1

-- Define the circle
def is_on_circle (x y r : ℝ) : Prop := x^2 + y^2 = r^2

-- Define the tangent line
def is_on_line (x y k m : ℝ) : Prop := y = k * x + m

-- Define the condition that the line is tangent to the circle
def is_tangent_to_circle (k m r : ℝ) : Prop := m^2 = (1 + k^2) * r^2

-- Main theorem
theorem ellipse_circle_tangent_relation 
  (a b r k m x₁ y₁ x₂ y₂ : ℝ) 
  (ha : a > 0) (hb : b > 0) (hr : 0 < r) (hrb : r < b) :
  is_on_ellipse x₁ y₁ a b ∧ 
  is_on_ellipse x₂ y₂ a b ∧
  is_on_line x₁ y₁ k m ∧
  is_on_line x₂ y₂ k m ∧
  is_tangent_to_circle k m r ∧
  x₁ * x₂ + y₁ * y₂ = 0 →
  r^2 * (a^2 + b^2) = a^2 * b^2 :=
by sorry

end NUMINAMATH_CALUDE_ellipse_circle_tangent_relation_l1421_142191


namespace NUMINAMATH_CALUDE_pear_banana_weight_equality_l1421_142121

/-- Given that 10 pears weigh the same as 6 bananas, 
    prove that 50 pears weigh the same as 30 bananas. -/
theorem pear_banana_weight_equality :
  ∀ (pear_weight banana_weight : ℕ → ℝ),
  (∀ n : ℕ, pear_weight (10 * n) = banana_weight (6 * n)) →
  pear_weight 50 = banana_weight 30 :=
by
  sorry

end NUMINAMATH_CALUDE_pear_banana_weight_equality_l1421_142121


namespace NUMINAMATH_CALUDE_sequence_sum_property_l1421_142152

theorem sequence_sum_property (a : ℕ → ℝ) (S : ℕ → ℝ) 
  (h : ∀ n : ℕ, n ≥ 1 → S n = 2 * a n - n) :
  2 / (a 1 * a 2) + 4 / (a 2 * a 3) + 8 / (a 3 * a 4) + 16 / (a 4 * a 5) = 30 / 31 :=
by sorry

end NUMINAMATH_CALUDE_sequence_sum_property_l1421_142152


namespace NUMINAMATH_CALUDE_hyperbola_asymptotes_l1421_142139

/-- The hyperbola equation -/
def hyperbola (x y : ℝ) : Prop := x^2 - 4*y^2 = -1

/-- The asymptote equations -/
def asymptotes (x y : ℝ) : Prop := x + 2*y = 0 ∨ x - 2*y = 0

/-- Theorem: The asymptotes of the given hyperbola are x ± 2y = 0 -/
theorem hyperbola_asymptotes : 
  ∀ x y : ℝ, hyperbola x y → asymptotes x y :=
sorry

end NUMINAMATH_CALUDE_hyperbola_asymptotes_l1421_142139


namespace NUMINAMATH_CALUDE_range_of_a_l1421_142153

-- Define the propositions p and q
def p (x a : ℝ) : Prop := x^2 - 4*a*x + 3*a^2 < 0

def q (x : ℝ) : Prop := x^2 - x - 6 ≤ 0

-- Define the theorem
theorem range_of_a (a : ℝ) :
  (a < 0) →
  (∀ x, ¬(p x a) → ¬(q x)) →
  (∃ x, ¬(p x a) ∧ (q x)) →
  -2/3 ≤ a ∧ a < 0 :=
by sorry

end NUMINAMATH_CALUDE_range_of_a_l1421_142153


namespace NUMINAMATH_CALUDE_john_notebook_duration_l1421_142178

/-- The number of days notebooks last given specific conditions -/
def notebook_duration (
  num_notebooks : ℕ
  ) (pages_per_notebook : ℕ
  ) (pages_per_weekday : ℕ
  ) (pages_per_weekend_day : ℕ
  ) : ℕ :=
  let total_pages := num_notebooks * pages_per_notebook
  let pages_per_week := 5 * pages_per_weekday + 2 * pages_per_weekend_day
  let full_weeks := total_pages / pages_per_week
  let remaining_pages := total_pages % pages_per_week
  let full_days := full_weeks * 7
  let extra_days := 
    if remaining_pages ≤ 5 * pages_per_weekday
    then remaining_pages / pages_per_weekday
    else 5 + (remaining_pages - 5 * pages_per_weekday + pages_per_weekend_day - 1) / pages_per_weekend_day
  full_days + extra_days

theorem john_notebook_duration :
  notebook_duration 5 40 4 6 = 43 := by
  sorry

end NUMINAMATH_CALUDE_john_notebook_duration_l1421_142178


namespace NUMINAMATH_CALUDE_quadratic_inequality_solution_set_l1421_142137

theorem quadratic_inequality_solution_set :
  {x : ℝ | 3 * x^2 - 5 * x - 8 > 0} = Set.Iio (-4/3) ∪ Set.Ioi 2 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_inequality_solution_set_l1421_142137


namespace NUMINAMATH_CALUDE_repeating_decimal_division_l1421_142128

theorem repeating_decimal_division :
  let a : ℚ := 64 / 99
  let b : ℚ := 16 / 99
  a / b = 4 := by sorry

end NUMINAMATH_CALUDE_repeating_decimal_division_l1421_142128


namespace NUMINAMATH_CALUDE_complex_expression_equality_l1421_142122

theorem complex_expression_equality : 
  let a : ℂ := 3 - 2*I
  let b : ℂ := 2 + 3*I
  3*a + 4*b = 17 + 6*I := by sorry

end NUMINAMATH_CALUDE_complex_expression_equality_l1421_142122


namespace NUMINAMATH_CALUDE_division_problem_l1421_142181

theorem division_problem (dividend : ℕ) (divisor : ℕ) (remainder : ℕ) (quotient : ℕ) : 
  dividend = 1565 → divisor = 24 → remainder = 5 → quotient = 65 →
  dividend = divisor * quotient + remainder :=
by sorry

end NUMINAMATH_CALUDE_division_problem_l1421_142181


namespace NUMINAMATH_CALUDE_complex_exponentiation_165_deg_60_l1421_142170

theorem complex_exponentiation_165_deg_60 : 
  (Complex.exp (Complex.I * Real.pi * 165 / 180)) ^ 60 = -1 := by
  sorry

end NUMINAMATH_CALUDE_complex_exponentiation_165_deg_60_l1421_142170


namespace NUMINAMATH_CALUDE_shift_down_quadratic_l1421_142127

-- Define the original quadratic function
def original_function (x : ℝ) : ℝ := x^2

-- Define the transformation (shift down by 2 units)
def shift_down (y : ℝ) : ℝ := y - 2

-- Define the resulting function after transformation
def resulting_function (x : ℝ) : ℝ := x^2 - 2

-- Theorem stating that shifting the original function down by 2 units
-- results in the resulting function
theorem shift_down_quadratic :
  ∀ x : ℝ, shift_down (original_function x) = resulting_function x :=
by
  sorry


end NUMINAMATH_CALUDE_shift_down_quadratic_l1421_142127


namespace NUMINAMATH_CALUDE_inscribed_rectangle_delta_l1421_142134

/-- Triangle with side lengths a, b, c -/
structure Triangle (a b c : ℝ) where
  side_a : a > 0
  side_b : b > 0
  side_c : c > 0

/-- Rectangle inscribed in a triangle -/
structure InscribedRectangle (T : Triangle a b c) where
  area : ℝ → ℝ  -- Area as a function of the rectangle's width

/-- The coefficient δ in the quadratic area formula of an inscribed rectangle -/
def delta (T : Triangle 15 39 36) (R : InscribedRectangle T) : ℚ :=
  60 / 169

theorem inscribed_rectangle_delta :
  ∀ (T : Triangle 15 39 36) (R : InscribedRectangle T),
  ∃ (γ : ℝ), ∀ (ω : ℝ), R.area ω = γ * ω - (delta T R : ℝ) * ω^2 := by
  sorry

end NUMINAMATH_CALUDE_inscribed_rectangle_delta_l1421_142134


namespace NUMINAMATH_CALUDE_equilateral_triangles_are_similar_l1421_142145

/-- An equilateral triangle is a triangle with all sides equal -/
structure EquilateralTriangle where
  side : ℝ
  side_positive : side > 0

/-- Similarity of two equilateral triangles -/
def are_similar (t1 t2 : EquilateralTriangle) : Prop :=
  ∃ k : ℝ, k > 0 ∧ t2.side = k * t1.side

/-- Theorem: Any two equilateral triangles are similar -/
theorem equilateral_triangles_are_similar (t1 t2 : EquilateralTriangle) :
  are_similar t1 t2 := by
  sorry


end NUMINAMATH_CALUDE_equilateral_triangles_are_similar_l1421_142145


namespace NUMINAMATH_CALUDE_vector_problem_l1421_142144

/-- Given two vectors a and b in ℝ², prove that:
    1) a = (-3, 4) and b = (5, -12)
    2) The dot product of a and b is -63
    3) The cosine of the angle between a and b is -63/65
-/
theorem vector_problem (a b : ℝ × ℝ) :
  (a.1 + b.1 = 2 ∧ a.2 + b.2 = -8) ∧  -- a + b = (2, -8)
  (a.1 - b.1 = -8 ∧ a.2 - b.2 = 16) → -- a - b = (-8, 16)
  (a = (-3, 4) ∧ b = (5, -12)) ∧      -- Part 1
  (a.1 * b.1 + a.2 * b.2 = -63) ∧     -- Part 2
  ((a.1 * b.1 + a.2 * b.2) / (Real.sqrt (a.1^2 + a.2^2) * Real.sqrt (b.1^2 + b.2^2)) = -63/65) -- Part 3
  := by sorry

end NUMINAMATH_CALUDE_vector_problem_l1421_142144


namespace NUMINAMATH_CALUDE_mystery_number_proof_l1421_142138

theorem mystery_number_proof : ∃ x : ℝ, x * 6 = 72 ∧ x = 12 := by
  sorry

end NUMINAMATH_CALUDE_mystery_number_proof_l1421_142138


namespace NUMINAMATH_CALUDE_smallest_volume_is_180_l1421_142129

/-- Represents the dimensions and cube counts of a rectangular box. -/
structure BoxDimensions where
  a : ℕ+  -- length
  b : ℕ+  -- width
  c : ℕ+  -- height
  red_in_bc : ℕ+  -- number of red cubes in each 1×b×c layer
  green_in_bc : ℕ+  -- number of green cubes in each 1×b×c layer
  green_in_ac : ℕ+  -- number of green cubes in each a×1×c layer
  yellow_in_ac : ℕ+  -- number of yellow cubes in each a×1×c layer

/-- Checks if the given box dimensions satisfy the problem conditions. -/
def valid_box_dimensions (box : BoxDimensions) : Prop :=
  box.red_in_bc = 9 ∧
  box.green_in_bc = 12 ∧
  box.green_in_ac = 20 ∧
  box.yellow_in_ac = 25

/-- Calculates the volume of the box. -/
def box_volume (box : BoxDimensions) : ℕ :=
  box.a * box.b * box.c

/-- The main theorem stating that the smallest possible volume is 180. -/
theorem smallest_volume_is_180 :
  ∀ box : BoxDimensions, valid_box_dimensions box → box_volume box ≥ 180 :=
by sorry

end NUMINAMATH_CALUDE_smallest_volume_is_180_l1421_142129


namespace NUMINAMATH_CALUDE_nine_eat_both_veg_and_non_veg_l1421_142154

/-- Represents the number of people in different dietary categories in a family -/
structure FamilyDiet where
  only_veg : ℕ
  only_non_veg : ℕ
  total_veg : ℕ

/-- Calculates the number of people who eat both veg and non-veg -/
def both_veg_and_non_veg (f : FamilyDiet) : ℕ :=
  f.total_veg - f.only_veg

/-- Theorem stating that 9 people eat both veg and non-veg in the given family -/
theorem nine_eat_both_veg_and_non_veg (f : FamilyDiet)
    (h1 : f.only_veg = 11)
    (h2 : f.only_non_veg = 6)
    (h3 : f.total_veg = 20) :
    both_veg_and_non_veg f = 9 := by
  sorry

end NUMINAMATH_CALUDE_nine_eat_both_veg_and_non_veg_l1421_142154


namespace NUMINAMATH_CALUDE_range_of_a_l1421_142173

theorem range_of_a (a : ℝ) : 
  (∀ x : ℝ, a * x^2 + 2 * a * x + 1 > 0) ↔ (0 ≤ a ∧ a < 1) :=
sorry

end NUMINAMATH_CALUDE_range_of_a_l1421_142173


namespace NUMINAMATH_CALUDE_second_project_depth_l1421_142174

/-- Represents a digging project with its dimensions and duration -/
structure DiggingProject where
  depth : ℝ
  length : ℝ
  breadth : ℝ
  days : ℝ

/-- Calculates the volume of a digging project -/
def volume (p : DiggingProject) : ℝ := p.depth * p.length * p.breadth

/-- The first digging project -/
def project1 : DiggingProject := {
  depth := 100,
  length := 25,
  breadth := 30,
  days := 12
}

/-- The second digging project with unknown depth -/
def project2 (depth : ℝ) : DiggingProject := {
  depth := depth,
  length := 20,
  breadth := 50,
  days := 12
}

/-- Theorem stating that the depth of the second project is 75 meters -/
theorem second_project_depth : 
  ∃ (depth : ℝ), volume project1 = volume (project2 depth) ∧ depth = 75 := by
  sorry


end NUMINAMATH_CALUDE_second_project_depth_l1421_142174


namespace NUMINAMATH_CALUDE_parabola_fixed_point_l1421_142185

/-- The parabola passes through the point (3, 36) for all real t -/
theorem parabola_fixed_point :
  ∀ t : ℝ, 36 = 4 * (3 : ℝ)^2 + t * 3 - t^2 - 3 * t := by
  sorry

end NUMINAMATH_CALUDE_parabola_fixed_point_l1421_142185


namespace NUMINAMATH_CALUDE_sequence_sum_l1421_142166

theorem sequence_sum (n : ℕ) (y : ℕ → ℕ) (h1 : y 1 = 2) 
  (h2 : ∀ k ∈ Finset.range (n - 1), y (k + 1) = y k + k + 1) : 
  Finset.sum (Finset.range n) (λ k => y (k + 1)) = 2 * n + (n - 1) * n * (n + 1) / 6 := by
  sorry

end NUMINAMATH_CALUDE_sequence_sum_l1421_142166


namespace NUMINAMATH_CALUDE_cyclic_sum_inequality_l1421_142168

theorem cyclic_sum_inequality (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) (h_xyz : x * y * z ≥ 1) :
  (x^5 - x^2) / (x^5 + y^2 + z^2) + (y^5 - y^2) / (y^5 + z^2 + x^2) + (z^5 - z^2) / (z^5 + x^2 + y^2) ≥ 0 :=
sorry

end NUMINAMATH_CALUDE_cyclic_sum_inequality_l1421_142168


namespace NUMINAMATH_CALUDE_complex_number_location_l1421_142115

theorem complex_number_location (z : ℂ) (h : (1 - Complex.I) * z = 2 * Complex.I) :
  z.re < 0 ∧ z.im > 0 := by
  sorry

end NUMINAMATH_CALUDE_complex_number_location_l1421_142115


namespace NUMINAMATH_CALUDE_total_goats_l1421_142163

theorem total_goats (washington_goats : ℕ) (paddington_extra : ℕ) : 
  washington_goats = 180 → 
  paddington_extra = 70 → 
  washington_goats + (washington_goats + paddington_extra) = 430 := by
sorry

end NUMINAMATH_CALUDE_total_goats_l1421_142163


namespace NUMINAMATH_CALUDE_impossible_configuration_l1421_142113

theorem impossible_configuration : ¬ ∃ (arrangement : List ℕ) (sum : ℕ),
  (arrangement.toFinset = {1, 4, 9, 16, 25, 36, 49}) ∧
  (∀ radial_line : List ℕ, radial_line.sum = sum) ∧
  (∀ triangle : List ℕ, triangle.sum = sum) :=
by sorry

end NUMINAMATH_CALUDE_impossible_configuration_l1421_142113


namespace NUMINAMATH_CALUDE_expression_simplification_l1421_142131

theorem expression_simplification :
  (((3 + 4 + 5 + 6) / 3) + ((3 * 6 + 9) / 4)) = 12.75 := by sorry

end NUMINAMATH_CALUDE_expression_simplification_l1421_142131


namespace NUMINAMATH_CALUDE_equation_solution_l1421_142100

theorem equation_solution :
  ∃ n : ℚ, (22 + Real.sqrt (-4 + 18 * n) = 24) ∧ n = 4/9 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l1421_142100


namespace NUMINAMATH_CALUDE_quadruple_reappearance_l1421_142175

/-- The transformation function that generates the next quadruple -/
def transform (q : ℝ × ℝ × ℝ × ℝ) : ℝ × ℝ × ℝ × ℝ :=
  let (a, b, c, d) := q
  (a * b, b * c, c * d, d * a)

/-- The sequence of quadruples generated by repeatedly applying the transformation -/
def quadruple_sequence (initial : ℝ × ℝ × ℝ × ℝ) : ℕ → ℝ × ℝ × ℝ × ℝ
  | 0 => initial
  | n + 1 => transform (quadruple_sequence initial n)

theorem quadruple_reappearance (a b c d : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0) (hd : d ≠ 0) :
  (∃ (n : ℕ), n > 0 ∧ quadruple_sequence (a, b, c, d) n = (a, b, c, d)) →
  a = 1 ∧ b = 1 ∧ c = 1 ∧ d = 1 :=
sorry

end NUMINAMATH_CALUDE_quadruple_reappearance_l1421_142175


namespace NUMINAMATH_CALUDE_parabola_directrix_l1421_142150

/-- Given a parabola with equation y = 2x^2, its directrix equation is y = -1/8 -/
theorem parabola_directrix (x y : ℝ) : 
  (y = 2 * x^2) → (∃ (k : ℝ), k = -1/8 ∧ k = y) :=
by sorry

end NUMINAMATH_CALUDE_parabola_directrix_l1421_142150


namespace NUMINAMATH_CALUDE_probability_below_8_l1421_142133

theorem probability_below_8 (p_10 p_9 p_8 : ℝ) 
  (h1 : p_10 = 0.24)
  (h2 : p_9 = 0.28)
  (h3 : p_8 = 0.19) :
  1 - (p_10 + p_9 + p_8) = 0.29 := by
  sorry

end NUMINAMATH_CALUDE_probability_below_8_l1421_142133


namespace NUMINAMATH_CALUDE_quadratic_distinct_roots_l1421_142112

theorem quadratic_distinct_roots (m : ℝ) : 
  (∃ x y : ℝ, x ≠ y ∧ x^2 + m*x + 9 = 0 ∧ y^2 + m*y + 9 = 0) ↔ 
  (m < -6 ∨ m > 6) := by
sorry

end NUMINAMATH_CALUDE_quadratic_distinct_roots_l1421_142112


namespace NUMINAMATH_CALUDE_sharon_salary_increase_l1421_142190

theorem sharon_salary_increase (S : ℝ) (h1 : S + 0.20 * S = 600) (h2 : S + x * S = 575) : x = 0.15 := by
  sorry

end NUMINAMATH_CALUDE_sharon_salary_increase_l1421_142190


namespace NUMINAMATH_CALUDE_equal_paper_distribution_l1421_142197

theorem equal_paper_distribution (total_sheets : ℕ) (num_friends : ℕ) (sheets_per_friend : ℕ) :
  total_sheets = 15 →
  num_friends = 3 →
  total_sheets = num_friends * sheets_per_friend →
  sheets_per_friend = 5 := by
  sorry

end NUMINAMATH_CALUDE_equal_paper_distribution_l1421_142197


namespace NUMINAMATH_CALUDE_guy_has_sixty_cents_l1421_142111

/-- The amount of money each person has in cents -/
structure Money where
  lance : ℕ
  margaret : ℕ
  bill : ℕ
  guy : ℕ

/-- The total amount of money in cents -/
def total (m : Money) : ℕ := m.lance + m.margaret + m.bill + m.guy

/-- Theorem: Given the conditions, Guy has 60 cents -/
theorem guy_has_sixty_cents (m : Money) 
  (h1 : m.lance = 70)
  (h2 : m.margaret = 75)  -- Three-fourths of a dollar is 75 cents
  (h3 : m.bill = 60)      -- Six dimes is 60 cents
  (h4 : total m = 265) : 
  m.guy = 60 := by
  sorry


end NUMINAMATH_CALUDE_guy_has_sixty_cents_l1421_142111


namespace NUMINAMATH_CALUDE_quadratic_function_m_value_l1421_142118

/-- A function y of x is quadratic if it can be written in the form y = ax² + bx + c, where a ≠ 0 -/
def IsQuadratic (f : ℝ → ℝ) : Prop :=
  ∃ a b c : ℝ, a ≠ 0 ∧ ∀ x, f x = a * x^2 + b * x + c

/-- The main theorem stating that m = -1 is the only value satisfying the given conditions -/
theorem quadratic_function_m_value :
  ∃! m : ℝ, IsQuadratic (fun x ↦ (m - 1) * x^(m^2 + 1) + 3 * x) ∧ m - 1 ≠ 0 :=
by
  sorry


end NUMINAMATH_CALUDE_quadratic_function_m_value_l1421_142118


namespace NUMINAMATH_CALUDE_polynomial_factorization_l1421_142164

theorem polynomial_factorization (b : ℝ) : 
  (8 * b^4 - 100 * b^3 + 18) - (3 * b^4 - 11 * b^3 + 18) = b^3 * (5 * b - 89) := by
  sorry

end NUMINAMATH_CALUDE_polynomial_factorization_l1421_142164


namespace NUMINAMATH_CALUDE_simplify_fraction_l1421_142165

theorem simplify_fraction : 21 * (8 / 15) * (1 / 14) = 4 / 5 := by
  sorry

end NUMINAMATH_CALUDE_simplify_fraction_l1421_142165


namespace NUMINAMATH_CALUDE_solve_for_m_l1421_142117

theorem solve_for_m (x y m : ℝ) : 
  x = 2 → 
  y = m → 
  3 * x + 2 * y = 10 → 
  m = 2 := by
sorry

end NUMINAMATH_CALUDE_solve_for_m_l1421_142117


namespace NUMINAMATH_CALUDE_no_prime_sum_53_l1421_142147

-- Define what it means for a number to be prime
def isPrime (n : ℕ) : Prop := n > 1 ∧ ∀ d : ℕ, d ∣ n → d = 1 ∨ d = n

-- Theorem statement
theorem no_prime_sum_53 : ¬∃ (p q : ℕ), isPrime p ∧ isPrime q ∧ p + q = 53 := by
  sorry

end NUMINAMATH_CALUDE_no_prime_sum_53_l1421_142147


namespace NUMINAMATH_CALUDE_smallest_valid_fourth_number_l1421_142151

def is_valid_fourth_number (n : ℕ) : Prop :=
  n ≥ 10 ∧ n < 100 ∧
  (6 + 8 + 2 + 4 + 8 + 5 + (n / 10) + (n % 10)) * 4 = 68 + 24 + 85 + n

theorem smallest_valid_fourth_number :
  ∀ m : ℕ, m ≥ 10 ∧ m < 57 → ¬(is_valid_fourth_number m) ∧ is_valid_fourth_number 57 := by
  sorry

end NUMINAMATH_CALUDE_smallest_valid_fourth_number_l1421_142151


namespace NUMINAMATH_CALUDE_pet_store_dogs_l1421_142104

/-- Calculates the number of dogs in a pet store after a series of events --/
def final_dog_count (initial : ℕ) 
  (sunday_received sunday_sold : ℕ)
  (monday_received monday_returned : ℕ)
  (tuesday_received tuesday_sold : ℕ) : ℕ :=
  initial + sunday_received - sunday_sold + 
  monday_received + monday_returned +
  tuesday_received - tuesday_sold

/-- Theorem stating the final number of dogs in the pet store --/
theorem pet_store_dogs : 
  final_dog_count 2 5 2 3 1 4 3 = 10 := by
  sorry

end NUMINAMATH_CALUDE_pet_store_dogs_l1421_142104


namespace NUMINAMATH_CALUDE_base_conversion_512_to_base_7_l1421_142108

theorem base_conversion_512_to_base_7 :
  (1 * 7^3 + 3 * 7^2 + 3 * 7^1 + 1 * 7^0) = 512 := by
  sorry

end NUMINAMATH_CALUDE_base_conversion_512_to_base_7_l1421_142108


namespace NUMINAMATH_CALUDE_jim_sara_equal_savings_l1421_142132

/-- The number of weeks in the saving period -/
def weeks : ℕ := 820

/-- Sara's initial savings in dollars -/
def sara_initial : ℕ := 4100

/-- Sara's weekly savings in dollars -/
def sara_weekly : ℕ := 10

/-- Jim's weekly savings in dollars -/
def jim_weekly : ℕ := 15

/-- Total savings after the given period -/
def total_savings (initial weekly : ℕ) : ℕ :=
  initial + weekly * weeks

theorem jim_sara_equal_savings :
  total_savings 0 jim_weekly = total_savings sara_initial sara_weekly := by
  sorry

end NUMINAMATH_CALUDE_jim_sara_equal_savings_l1421_142132


namespace NUMINAMATH_CALUDE_box_has_four_balls_l1421_142177

/-- A color of a ball -/
inductive Color
| Red
| Blue
| Other

/-- A box containing balls of different colors -/
structure Box where
  balls : List Color

/-- Checks if a list of colors contains at least one red and one blue -/
def hasRedAndBlue (colors : List Color) : Prop :=
  Color.Red ∈ colors ∧ Color.Blue ∈ colors

/-- The main theorem stating that the box must contain exactly 4 balls -/
theorem box_has_four_balls (box : Box) : 
  (∀ (a b c : Color), a ∈ box.balls → b ∈ box.balls → c ∈ box.balls → 
    a ≠ b → b ≠ c → a ≠ c → hasRedAndBlue [a, b, c]) →
  (3 < box.balls.length) →
  box.balls.length = 4 := by
  sorry


end NUMINAMATH_CALUDE_box_has_four_balls_l1421_142177


namespace NUMINAMATH_CALUDE_complex_division_result_l1421_142110

theorem complex_division_result : (10 * Complex.I) / (1 - 2 * Complex.I) = -4 + 2 * Complex.I := by
  sorry

end NUMINAMATH_CALUDE_complex_division_result_l1421_142110


namespace NUMINAMATH_CALUDE_ariella_savings_after_two_years_l1421_142146

/-- Calculates the final amount in a savings account after simple interest is applied. -/
def final_amount (initial_amount : ℝ) (interest_rate : ℝ) (years : ℝ) : ℝ :=
  initial_amount * (1 + interest_rate * years)

/-- Proves that Ariella will have $720 after two years given the problem conditions. -/
theorem ariella_savings_after_two_years 
  (daniella_amount : ℝ)
  (ariella_excess : ℝ)
  (interest_rate : ℝ)
  (years : ℝ)
  (h1 : daniella_amount = 400)
  (h2 : ariella_excess = 200)
  (h3 : interest_rate = 0.1)
  (h4 : years = 2) :
  final_amount (daniella_amount + ariella_excess) interest_rate years = 720 :=
by
  sorry

#check ariella_savings_after_two_years

end NUMINAMATH_CALUDE_ariella_savings_after_two_years_l1421_142146


namespace NUMINAMATH_CALUDE_x_power_ten_equals_one_l1421_142176

theorem x_power_ten_equals_one (x : ℂ) (h : x + 1/x = Real.sqrt 5) : x^10 = 1 := by
  sorry

end NUMINAMATH_CALUDE_x_power_ten_equals_one_l1421_142176


namespace NUMINAMATH_CALUDE_value_in_scientific_notation_l1421_142140

-- Define a billion
def billion : ℝ := 10^9

-- Define the value in question
def value : ℝ := 101.49 * billion

-- Theorem statement
theorem value_in_scientific_notation : value = 1.0149 * 10^10 := by
  sorry

end NUMINAMATH_CALUDE_value_in_scientific_notation_l1421_142140


namespace NUMINAMATH_CALUDE_geometric_sequence_fourth_term_l1421_142183

/-- Given a geometric sequence {a_n}, if a_2 and a_6 are roots of x^2 - 34x + 81 = 0, then a_4 = 9 -/
theorem geometric_sequence_fourth_term
  (a : ℕ → ℝ)
  (h_geometric : ∀ n, a (n + 1) / a n = a (n + 2) / a (n + 1))
  (h_roots : a 2 * a 6 = 81 ∧ a 2 + a 6 = 34) :
  a 4 = 9 := by sorry

end NUMINAMATH_CALUDE_geometric_sequence_fourth_term_l1421_142183


namespace NUMINAMATH_CALUDE_units_digit_of_special_two_digit_number_l1421_142125

/-- 
Given a two-digit number M = 10a + b, where a and b are single digits,
if M = ab + (a + b) + 5, then b = 8.
-/
theorem units_digit_of_special_two_digit_number (a b : ℕ) : 
  a ≥ 1 ∧ a ≤ 9 ∧ b ≥ 0 ∧ b ≤ 9 →
  (10 * a + b = a * b + a + b + 5) →
  b = 8 := by
sorry

end NUMINAMATH_CALUDE_units_digit_of_special_two_digit_number_l1421_142125


namespace NUMINAMATH_CALUDE_fraction_equals_zero_l1421_142101

theorem fraction_equals_zero (x : ℝ) :
  x = 3 → (2 * x - 6) / (5 * x + 10) = 0 :=
by
  sorry

end NUMINAMATH_CALUDE_fraction_equals_zero_l1421_142101


namespace NUMINAMATH_CALUDE_fraction_of_powers_equals_five_thirds_l1421_142199

theorem fraction_of_powers_equals_five_thirds :
  (2^2014 + 2^2012) / (2^2014 - 2^2012) = 5/3 := by
  sorry

end NUMINAMATH_CALUDE_fraction_of_powers_equals_five_thirds_l1421_142199


namespace NUMINAMATH_CALUDE_max_area_rectangle_l1421_142156

theorem max_area_rectangle (l w : ℕ) : 
  (2 * (l + w) = 120) →  -- perimeter condition
  (∀ a b : ℕ, 2 * (a + b) = 120 → l * w ≥ a * b) →  -- maximum area condition
  l * w = 900 := by
sorry

end NUMINAMATH_CALUDE_max_area_rectangle_l1421_142156


namespace NUMINAMATH_CALUDE_crescent_moon_area_l1421_142182

/-- The area of a crescent moon formed by two circles -/
theorem crescent_moon_area :
  let large_circle_radius : ℝ := 4
  let small_circle_radius : ℝ := 2
  let large_quarter_circle_area : ℝ := π * large_circle_radius^2 / 4
  let small_half_circle_area : ℝ := π * small_circle_radius^2 / 2
  large_quarter_circle_area - small_half_circle_area = 2 * π := by
sorry

end NUMINAMATH_CALUDE_crescent_moon_area_l1421_142182


namespace NUMINAMATH_CALUDE_erroneous_product_equals_correct_l1421_142187

/-- Given a positive integer, reverse its digits --/
def reverse_digits (n : ℕ) : ℕ := sorry

/-- Check if a number is two-digit --/
def is_two_digit (n : ℕ) : Prop := 10 ≤ n ∧ n < 100

theorem erroneous_product_equals_correct (a b : ℕ) :
  a > 0 ∧ b > 0 ∧ is_two_digit b ∧ a * (reverse_digits b) = 180 → a * b = 180 := by
  sorry

end NUMINAMATH_CALUDE_erroneous_product_equals_correct_l1421_142187


namespace NUMINAMATH_CALUDE_candy_distribution_l1421_142136

theorem candy_distribution (n : ℕ) (h1 : n > 0) : 
  (100 % n = 1) ↔ n = 11 := by sorry

end NUMINAMATH_CALUDE_candy_distribution_l1421_142136


namespace NUMINAMATH_CALUDE_cube_surface_area_increase_l1421_142123

theorem cube_surface_area_increase (L : ℝ) (L_new : ℝ) (h : L > 0) :
  L_new = 1.3 * L →
  (6 * L_new^2 - 6 * L^2) / (6 * L^2) = 0.69 := by
sorry

end NUMINAMATH_CALUDE_cube_surface_area_increase_l1421_142123


namespace NUMINAMATH_CALUDE_percentage_relation_l1421_142162

theorem percentage_relation (A B C x y : ℝ) 
  (h1 : A > 0) (h2 : B > 0) (h3 : C > 0)
  (h4 : A > B) (h5 : B > C)
  (h6 : A = B * (1 + x / 100))
  (h7 : B = C * (1 + y / 100)) : 
  x = 100 * (A / (C * (1 + y / 100)) - 1) := by
  sorry

end NUMINAMATH_CALUDE_percentage_relation_l1421_142162


namespace NUMINAMATH_CALUDE_concentric_circles_intersection_l1421_142135

theorem concentric_circles_intersection (r_outer r_inner : ℝ) (h_outer : r_outer * 2 * Real.pi = 24 * Real.pi) (h_inner : r_inner * 2 * Real.pi = 14 * Real.pi) : r_outer - r_inner = 5 := by
  sorry

end NUMINAMATH_CALUDE_concentric_circles_intersection_l1421_142135


namespace NUMINAMATH_CALUDE_cosine_translation_monotonicity_l1421_142192

/-- Given a function g(x) = 2cos(2x - π/3) that is monotonically increasing
    in the intervals [0, a/3] and [2a, 7π/6], prove that π/3 ≤ a ≤ π/2. -/
theorem cosine_translation_monotonicity (a : ℝ) :
  (∀ x ∈ Set.Icc 0 (a / 3), Monotone (fun x => 2 * Real.cos (2 * x - π / 3))) ∧
  (∀ x ∈ Set.Icc (2 * a) (7 * π / 6), Monotone (fun x => 2 * Real.cos (2 * x - π / 3))) →
  π / 3 ≤ a ∧ a ≤ π / 2 :=
by sorry

end NUMINAMATH_CALUDE_cosine_translation_monotonicity_l1421_142192


namespace NUMINAMATH_CALUDE_line_inclination_angle_l1421_142189

/-- The angle of inclination of the line x - √3y + 6 = 0 is 30°. -/
theorem line_inclination_angle (x y : ℝ) :
  x - Real.sqrt 3 * y + 6 = 0 →
  Real.arctan (Real.sqrt 3 / 3) = 30 * π / 180 :=
by sorry

end NUMINAMATH_CALUDE_line_inclination_angle_l1421_142189


namespace NUMINAMATH_CALUDE_ellipse_and_triangle_area_l1421_142105

-- Define the ellipse C
def ellipse_C (a b : ℝ) (x y : ℝ) : Prop :=
  (x^2 / a^2) + (y^2 / b^2) = 1 ∧ a > b ∧ b > 0

-- Define the inscribed circle
def inscribed_circle (x y : ℝ) : Prop :=
  x^2 + y^2 = 2

-- Define the parabola E
def parabola_E (p : ℝ) (x y : ℝ) : Prop :=
  y^2 = 2 * p * x ∧ p > 0

-- Define the line l
def line_l (m : ℝ) (x y : ℝ) : Prop :=
  y = x + m ∧ 0 ≤ m ∧ m ≤ 1

-- State the theorem
theorem ellipse_and_triangle_area :
  ∀ (a b c p m : ℝ) (x y : ℝ),
  ellipse_C a b x y →
  inscribed_circle x y →
  parabola_E p x y →
  line_l m x y →
  (∃ (x₁ y₁ x₂ y₂ : ℝ), line_l m x₁ y₁ ∧ line_l m x₂ y₂ ∧ parabola_E p x₁ y₁ ∧ parabola_E p x₂ y₂) →
  (∃ (F : ℝ × ℝ), F.1 = c ∧ F.2 = 0 ∧ c^2 = a^2 - b^2) →
  (b = c) →
  (a^2 = 8 ∧ b^2 = 4) ∧
  (∃ (S : ℝ), S = (32 * Real.sqrt 6) / 9 ∧
    ∀ (S' : ℝ), S' ≤ S) :=
sorry

end NUMINAMATH_CALUDE_ellipse_and_triangle_area_l1421_142105


namespace NUMINAMATH_CALUDE_total_coughs_after_20_minutes_l1421_142161

/-- The number of coughs per minute for Georgia -/
def georgia_coughs_per_minute : ℕ := 5

/-- The number of coughs per minute for Robert -/
def robert_coughs_per_minute : ℕ := 2 * georgia_coughs_per_minute

/-- The time period in minutes -/
def time_period : ℕ := 20

/-- The total number of coughs by Georgia and Robert after the given time period -/
def total_coughs : ℕ := (georgia_coughs_per_minute * time_period) + (robert_coughs_per_minute * time_period)

theorem total_coughs_after_20_minutes : total_coughs = 300 := by
  sorry

end NUMINAMATH_CALUDE_total_coughs_after_20_minutes_l1421_142161


namespace NUMINAMATH_CALUDE_removed_triangles_area_l1421_142193

theorem removed_triangles_area (s : ℝ) (x : ℝ) : 
  s = 16 → 
  (s - 2*x)^2 + (s - 2*x)^2 = s^2 →
  2 * x^2 = 768 - 512 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_removed_triangles_area_l1421_142193


namespace NUMINAMATH_CALUDE_unique_solution_absolute_value_equation_l1421_142143

theorem unique_solution_absolute_value_equation :
  ∃! x : ℝ, |x - 1| = |x - 2| + |x + 3| + 1 := by
  sorry

end NUMINAMATH_CALUDE_unique_solution_absolute_value_equation_l1421_142143


namespace NUMINAMATH_CALUDE_negation_of_universal_proposition_l1421_142188

theorem negation_of_universal_proposition :
  (¬ (∀ x : ℝ, x > Real.sin x)) ↔ (∃ x : ℝ, x ≤ Real.sin x) := by sorry

end NUMINAMATH_CALUDE_negation_of_universal_proposition_l1421_142188


namespace NUMINAMATH_CALUDE_average_speed_calculation_l1421_142184

theorem average_speed_calculation (total_distance : ℝ) (first_half_speed : ℝ) (second_half_time_factor : ℝ) :
  total_distance = 640 →
  first_half_speed = 80 →
  second_half_time_factor = 3 →
  let first_half_distance := total_distance / 2
  let first_half_time := first_half_distance / first_half_speed
  let second_half_time := first_half_time * second_half_time_factor
  let total_time := first_half_time + second_half_time
  total_distance / total_time = 40 := by
  sorry

end NUMINAMATH_CALUDE_average_speed_calculation_l1421_142184


namespace NUMINAMATH_CALUDE_largest_common_divisor_of_consecutive_odds_l1421_142126

theorem largest_common_divisor_of_consecutive_odds (n : ℕ) (h : Even n) (h_pos : 0 < n) :
  ∃ (k : ℕ), k = 315 ∧ 
  (∀ (d : ℕ), d ∣ ((n+1)*(n+3)*(n+5)*(n+7)*(n+9)*(n+11)*(n+13)) → d ≤ k) ∧
  k ∣ ((n+1)*(n+3)*(n+5)*(n+7)*(n+9)*(n+11)*(n+13)) :=
sorry

end NUMINAMATH_CALUDE_largest_common_divisor_of_consecutive_odds_l1421_142126


namespace NUMINAMATH_CALUDE_cheryl_material_problem_l1421_142114

theorem cheryl_material_problem (x : ℚ) : 
  (x + 2/3 : ℚ) - 8/18 = 2/3 → x = 4/9 := by sorry

end NUMINAMATH_CALUDE_cheryl_material_problem_l1421_142114


namespace NUMINAMATH_CALUDE_cosine_problem_l1421_142107

theorem cosine_problem (α β : Real) (h1 : 0 < α ∧ α < π/2) (h2 : 0 < β ∧ β < π/2)
  (h3 : Real.cos (α + β) = 12/13) (h4 : Real.cos (2*α + β) = 3/5) :
  Real.cos α = 56/65 := by
  sorry

end NUMINAMATH_CALUDE_cosine_problem_l1421_142107


namespace NUMINAMATH_CALUDE_solution_of_f_1001_l1421_142195

def f₁ (x : ℚ) : ℚ := 2/3 - 3/(3*x+1)

def f (n : ℕ) (x : ℚ) : ℚ :=
  match n with
  | 0 => x
  | 1 => f₁ x
  | n+1 => f₁ (f n x)

theorem solution_of_f_1001 :
  ∃ x : ℚ, f 1001 x = x - 3 ∧ x = 5/3 := by sorry

end NUMINAMATH_CALUDE_solution_of_f_1001_l1421_142195


namespace NUMINAMATH_CALUDE_original_number_l1421_142155

theorem original_number (x : ℝ) : 3 * (2 * x + 5) = 135 → x = 20 := by
  sorry

end NUMINAMATH_CALUDE_original_number_l1421_142155


namespace NUMINAMATH_CALUDE_inequality_proof_l1421_142169

theorem inequality_proof (a b c : ℝ) 
  (h_pos_a : 0 < a) (h_pos_b : 0 < b) (h_pos_c : 0 < c) 
  (h_sum : a + b + c = 1) : 
  (a * b + b * c + c * a ≤ 1 / 3) ∧ 
  (a^2 / b + b^2 / c + c^2 / a ≥ 1) := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l1421_142169


namespace NUMINAMATH_CALUDE_sqrt_75_plus_30sqrt3_form_l1421_142194

def is_square_free (n : ℕ) : Prop :=
  ∀ m : ℕ, m > 1 → m.sqrt ^ 2 ∣ n → m.sqrt ^ 2 = 1

theorem sqrt_75_plus_30sqrt3_form :
  ∃ (a b c : ℤ), (c : ℝ) > 0 ∧ is_square_free c.toNat ∧
  Real.sqrt (75 + 30 * Real.sqrt 3) = a + b * Real.sqrt c ∧
  a + b + c = 12 :=
sorry

end NUMINAMATH_CALUDE_sqrt_75_plus_30sqrt3_form_l1421_142194


namespace NUMINAMATH_CALUDE_workshop_pairing_probability_l1421_142172

theorem workshop_pairing_probability (n : ℕ) (h : n = 24) :
  let total_participants := n
  let pairing_probability := (1 : ℚ) / (n - 1 : ℚ)
  pairing_probability = (1 : ℚ) / (23 : ℚ) :=
by sorry

end NUMINAMATH_CALUDE_workshop_pairing_probability_l1421_142172


namespace NUMINAMATH_CALUDE_set_equality_l1421_142159

def M : Set ℝ := {x | x^2 - 3*x + 2 = 0}
def N : Set ℝ := {1, 2}

theorem set_equality : M = N := by
  sorry

end NUMINAMATH_CALUDE_set_equality_l1421_142159


namespace NUMINAMATH_CALUDE_largest_quantity_l1421_142149

theorem largest_quantity : 
  (2008 / 2007 + 2008 / 2009 : ℚ) > (2009 / 2008 + 2009 / 2010 : ℚ) ∧ 
  (2009 / 2008 + 2009 / 2010 : ℚ) > (2008 / 2009 + 2010 / 2009 : ℚ) := by
  sorry

end NUMINAMATH_CALUDE_largest_quantity_l1421_142149


namespace NUMINAMATH_CALUDE_function_extrema_implies_a_range_l1421_142102

/-- If f(x) = x^3 + 2ax^2 + 3(a+2)x + 1 has both a maximum and a minimum value, then a > 2 or a < -1 -/
theorem function_extrema_implies_a_range (a : ℝ) : 
  (∃ (max min : ℝ), ∀ x, (x^3 + 2*a*x^2 + 3*(a+2)*x + 1 ≤ max ∧ x^3 + 2*a*x^2 + 3*(a+2)*x + 1 ≥ min)) →
  (a > 2 ∨ a < -1) := by
  sorry


end NUMINAMATH_CALUDE_function_extrema_implies_a_range_l1421_142102


namespace NUMINAMATH_CALUDE_negation_of_forall_inequality_l1421_142158

theorem negation_of_forall_inequality :
  (¬ ∀ x : ℝ, x^2 - x > x + 1) ↔ (∃ x : ℝ, x^2 - x ≤ x + 1) := by
  sorry

end NUMINAMATH_CALUDE_negation_of_forall_inequality_l1421_142158


namespace NUMINAMATH_CALUDE_first_term_of_geometric_series_l1421_142160

/-- The first term of an infinite geometric series with common ratio -1/3 and sum 24 is 32 -/
theorem first_term_of_geometric_series (a : ℝ) : 
  (∑' n, a * (-1/3)^n : ℝ) = 24 → a = 32 := by
  sorry

end NUMINAMATH_CALUDE_first_term_of_geometric_series_l1421_142160


namespace NUMINAMATH_CALUDE_P_root_characteristics_l1421_142198

-- Define the polynomial P(x)
def P (x : ℝ) : ℝ := x^7 - 4*x^5 - 8*x^3 - x + 12

-- Theorem statement
theorem P_root_characteristics :
  (∀ x < 0, P x ≠ 0) ∧ (∃ x > 0, P x = 0) := by sorry

end NUMINAMATH_CALUDE_P_root_characteristics_l1421_142198


namespace NUMINAMATH_CALUDE_cornbread_pieces_l1421_142196

def pan_length : ℕ := 24
def pan_width : ℕ := 20
def piece_size : ℕ := 3

theorem cornbread_pieces :
  (pan_length * pan_width) / (piece_size * piece_size) = 53 :=
by sorry

end NUMINAMATH_CALUDE_cornbread_pieces_l1421_142196


namespace NUMINAMATH_CALUDE_divisible_by_nine_l1421_142106

theorem divisible_by_nine (n : ℕ) : ∃ k : ℤ, (4 : ℤ)^n + 15*n - 1 = 9*k := by
  sorry

end NUMINAMATH_CALUDE_divisible_by_nine_l1421_142106


namespace NUMINAMATH_CALUDE_arithmetic_geometric_sequence_a6_l1421_142180

def arithmetic_sequence (a : ℕ → ℝ) (d : ℝ) :=
  ∀ n, a (n + 1) = a n + d

def geometric_sequence (a b c : ℝ) :=
  b * b = a * c

theorem arithmetic_geometric_sequence_a6 (a : ℕ → ℝ) :
  arithmetic_sequence a 2 →
  geometric_sequence (a 1) (a 3) (a 4) →
  a 6 = 2 := by
sorry

end NUMINAMATH_CALUDE_arithmetic_geometric_sequence_a6_l1421_142180

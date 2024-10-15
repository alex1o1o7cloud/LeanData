import Mathlib

namespace NUMINAMATH_CALUDE_inequality_proof_l341_34169

theorem inequality_proof (x y : ℝ) (p q : ℕ) 
  (hx : x > 0) (hy : y > 0) (hxy : x ≠ y) :
  (x^(-(p:ℝ)/q) - y^((p:ℝ)/q) * x^(-(2*p:ℝ)/q)) / 
  (x^((1-2*p:ℝ)/q) - y^((1:ℝ)/q) * x^(-(2*p:ℝ)/q)) > 
  p * (x*y)^((p-1:ℝ)/(2*q)) :=
sorry

end NUMINAMATH_CALUDE_inequality_proof_l341_34169


namespace NUMINAMATH_CALUDE_rectangle_area_with_circles_l341_34137

/-- The area of a rectangle surrounded by four circles -/
theorem rectangle_area_with_circles (r : ℝ) (h1 : r = 3) : ∃ (length width : ℝ),
  length = 2 * r * 2 ∧ 
  width = 2 * r ∧
  length * width = 72 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_area_with_circles_l341_34137


namespace NUMINAMATH_CALUDE_indeterminate_product_at_opposite_points_l341_34117

-- Define a continuous function on an open interval
def ContinuousOnOpenInterval (f : ℝ → ℝ) (a b : ℝ) : Prop :=
  ∀ x, a < x ∧ x < b → ContinuousAt f x

-- Define the property of having a single root at 0 in the interval (-2, 2)
def SingleRootAtZero (f : ℝ → ℝ) : Prop :=
  (∀ x, -2 < x ∧ x < 2 ∧ f x = 0 → x = 0) ∧
  (f 0 = 0)

-- Theorem statement
theorem indeterminate_product_at_opposite_points
  (f : ℝ → ℝ)
  (h_cont : ContinuousOnOpenInterval f (-2) 2)
  (h_root : SingleRootAtZero f) :
  ∃ (f₁ f₂ f₃ : ℝ → ℝ),
    (ContinuousOnOpenInterval f₁ (-2) 2 ∧ SingleRootAtZero f₁ ∧ f₁ (-1) * f₁ 1 > 0) ∧
    (ContinuousOnOpenInterval f₂ (-2) 2 ∧ SingleRootAtZero f₂ ∧ f₂ (-1) * f₂ 1 < 0) ∧
    (ContinuousOnOpenInterval f₃ (-2) 2 ∧ SingleRootAtZero f₃ ∧ f₃ (-1) * f₃ 1 = 0) :=
  sorry

end NUMINAMATH_CALUDE_indeterminate_product_at_opposite_points_l341_34117


namespace NUMINAMATH_CALUDE_sum_of_max_and_min_is_two_l341_34199

-- Define the function f
def f (x : ℝ) : ℝ := |x - 3| + |x - 5| - |2*x - 8|

-- Define the domain
def domain : Set ℝ := { x | 3 ≤ x ∧ x ≤ 10 }

-- State the theorem
theorem sum_of_max_and_min_is_two :
  ∃ (max min : ℝ), 
    (∀ x ∈ domain, f x ≤ max) ∧
    (∃ x ∈ domain, f x = max) ∧
    (∀ x ∈ domain, min ≤ f x) ∧
    (∃ x ∈ domain, f x = min) ∧
    max + min = 2 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_max_and_min_is_two_l341_34199


namespace NUMINAMATH_CALUDE_smallest_marble_count_l341_34150

/-- Represents the number of marbles of each color in the urn -/
structure MarbleCount where
  red : ℕ
  white : ℕ
  blue : ℕ
  green : ℕ
  yellow : ℕ

/-- Calculates the total number of marbles in the urn -/
def totalMarbles (mc : MarbleCount) : ℕ :=
  mc.red + mc.white + mc.blue + mc.green + mc.yellow

/-- Represents the probability of drawing a specific combination of marbles -/
def drawProbability (mc : MarbleCount) (r w b g y : ℕ) : ℚ :=
  (mc.red.choose r * mc.white.choose w * mc.blue.choose b * mc.green.choose g * mc.yellow.choose y : ℚ) /
  (totalMarbles mc).choose 4

/-- Checks if the four specified events are equally likely -/
def eventsEquallyLikely (mc : MarbleCount) : Prop :=
  drawProbability mc 4 0 0 0 0 = drawProbability mc 3 1 0 0 0 ∧
  drawProbability mc 4 0 0 0 0 = drawProbability mc 1 1 1 0 1 ∧
  drawProbability mc 4 0 0 0 0 = drawProbability mc 1 1 1 1 0

/-- The main theorem stating the smallest number of marbles satisfying the conditions -/
theorem smallest_marble_count : ∃ (mc : MarbleCount), 
  eventsEquallyLikely mc ∧ 
  totalMarbles mc = 11 ∧ 
  (∀ (mc' : MarbleCount), eventsEquallyLikely mc' → totalMarbles mc' ≥ totalMarbles mc) :=
sorry

end NUMINAMATH_CALUDE_smallest_marble_count_l341_34150


namespace NUMINAMATH_CALUDE_solution_equation1_solution_equation2_l341_34171

-- Define the equations
def equation1 (x : ℝ) : Prop := 2 * x^2 = 8 * x
def equation2 (y : ℝ) : Prop := y^2 - 10 * y - 1 = 0

-- Theorem for the first equation
theorem solution_equation1 : 
  (∃ x : ℝ, equation1 x) ∧ 
  (∀ x : ℝ, equation1 x ↔ (x = 0 ∨ x = 4)) :=
sorry

-- Theorem for the second equation
theorem solution_equation2 : 
  (∃ y : ℝ, equation2 y) ∧ 
  (∀ y : ℝ, equation2 y ↔ (y = 5 + Real.sqrt 26 ∨ y = 5 - Real.sqrt 26)) :=
sorry

end NUMINAMATH_CALUDE_solution_equation1_solution_equation2_l341_34171


namespace NUMINAMATH_CALUDE_middle_marble_radius_l341_34180

/-- Given a sequence of five marbles with radii forming a geometric sequence,
    where the smallest radius is 8 and the largest radius is 18,
    prove that the middle (third) marble has a radius of 12. -/
theorem middle_marble_radius 
  (r : Fin 5 → ℝ)  -- r is a function mapping the index of each marble to its radius
  (h_geom_seq : ∀ i j k, i < j → j < k → r j ^ 2 = r i * r k)  -- geometric sequence condition
  (h_smallest : r 0 = 8)  -- radius of the smallest marble
  (h_largest : r 4 = 18)  -- radius of the largest marble
  : r 2 = 12 := by  -- radius of the middle (third) marble
sorry


end NUMINAMATH_CALUDE_middle_marble_radius_l341_34180


namespace NUMINAMATH_CALUDE_min_a_for_subset_l341_34161

theorem min_a_for_subset (a : ℝ) : 
  (∀ x ∈ Set.Icc 1 5, x^2 - 6*x ≤ a + 2) ↔ a ≥ -7 :=
sorry

end NUMINAMATH_CALUDE_min_a_for_subset_l341_34161


namespace NUMINAMATH_CALUDE_exponent_division_l341_34141

theorem exponent_division (a : ℝ) (m n : ℕ) :
  a ^ m / a ^ n = a ^ (m - n) :=
sorry

end NUMINAMATH_CALUDE_exponent_division_l341_34141


namespace NUMINAMATH_CALUDE_function_composition_equality_l341_34123

/-- Given a function f(x) = ax^2 - √3 where a > 0, prove that f(f(√3)) = -√3 implies a = √3/3 -/
theorem function_composition_equality (a : ℝ) (ha : a > 0) :
  let f : ℝ → ℝ := λ x ↦ a * x^2 - Real.sqrt 3
  f (f (Real.sqrt 3)) = -Real.sqrt 3 → a = Real.sqrt 3 / 3 := by
  sorry

end NUMINAMATH_CALUDE_function_composition_equality_l341_34123


namespace NUMINAMATH_CALUDE_f_composition_value_l341_34191

noncomputable def f (x : ℝ) : ℝ :=
  if x < 0 then 2^(x + 2) else x^3

theorem f_composition_value : f (f (-1)) = 8 := by sorry

end NUMINAMATH_CALUDE_f_composition_value_l341_34191


namespace NUMINAMATH_CALUDE_collinear_vectors_imply_fixed_point_l341_34157

/-- Two 2D vectors are collinear if one is a scalar multiple of the other -/
def collinear (v w : ℝ × ℝ) : Prop :=
  ∃ t : ℝ, v.1 = t * w.1 ∧ v.2 = t * w.2

/-- A point (x, y) is on a line y = mx + c if y = mx + c -/
def on_line (m c : ℝ) (p : ℝ × ℝ) : Prop :=
  p.2 = m * p.1 + c

theorem collinear_vectors_imply_fixed_point (k b : ℝ) :
  collinear (k + 2, 1) (-b, 1) →
  on_line k b (1, -2) :=
by sorry

end NUMINAMATH_CALUDE_collinear_vectors_imply_fixed_point_l341_34157


namespace NUMINAMATH_CALUDE_average_score_calculation_l341_34177

theorem average_score_calculation (total_students : ℕ) (male_students : ℕ) (female_students : ℕ)
  (male_avg_score : ℝ) (female_avg_score : ℝ) :
  male_students = (0.4 : ℝ) * total_students →
  female_students = total_students - male_students →
  male_avg_score = 75 →
  female_avg_score = 80 →
  (male_avg_score * male_students + female_avg_score * female_students) / total_students = 78 :=
by
  sorry

#check average_score_calculation

end NUMINAMATH_CALUDE_average_score_calculation_l341_34177


namespace NUMINAMATH_CALUDE_second_car_rate_l341_34181

/-- Given two cars starting at the same point, with the first car traveling at 50 mph,
    and after 3 hours the distance between them is 30 miles,
    prove that the rate of the second car is 40 mph. -/
theorem second_car_rate (v : ℝ) : 
  v > 0 →  -- The rate of the second car is positive
  50 * 3 - v * 3 = 30 →  -- After 3 hours, the distance between the cars is 30 miles
  v = 40 := by
sorry

end NUMINAMATH_CALUDE_second_car_rate_l341_34181


namespace NUMINAMATH_CALUDE_point_placement_theorem_l341_34138

theorem point_placement_theorem : ∃ n : ℕ+, 9 * n - 8 = 82 := by
  sorry

end NUMINAMATH_CALUDE_point_placement_theorem_l341_34138


namespace NUMINAMATH_CALUDE_problem_statement_problem_statement_2_l341_34136

def f (a : ℝ) (x : ℝ) : ℝ := x^2 - 2*a*x + 5

theorem problem_statement (a : ℝ) (h1 : a > 1) :
  (∀ x ∈ Set.Icc 1 a, f a x ∈ Set.Icc 1 a ∧ ∀ y ∈ Set.Icc 1 a, ∃ x ∈ Set.Icc 1 a, f a x = y) →
  a = 2 :=
sorry

theorem problem_statement_2 (a : ℝ) (h1 : a > 1) :
  (∀ x y : ℝ, x < y ∧ y ≤ 2 → f a x > f a y) ∧
  (∀ x ∈ Set.Icc 1 2, f a x ≤ 0) →
  a ≥ 3 :=
sorry

end NUMINAMATH_CALUDE_problem_statement_problem_statement_2_l341_34136


namespace NUMINAMATH_CALUDE_tangent_line_equation_l341_34104

-- Define the curve
def f (x : ℝ) : ℝ := x^2 + 2*x

-- Define the derivative of the curve
def f' (x : ℝ) : ℝ := 2*x + 2

-- Define the point of tangency
def point : ℝ × ℝ := (1, 3)

-- Theorem statement
theorem tangent_line_equation :
  let (x₀, y₀) := point
  let m := f' x₀
  (4 : ℝ) * x - y - 1 = 0 ↔ y - y₀ = m * (x - x₀) :=
by sorry

end NUMINAMATH_CALUDE_tangent_line_equation_l341_34104


namespace NUMINAMATH_CALUDE_trigonometric_identities_l341_34184

theorem trigonometric_identities (α : Real) 
  (h : Real.sin α + Real.cos α = Real.sqrt 3 / 3) : 
  (Real.sin α)^4 + (Real.cos α)^4 = 7/9 ∧ 
  Real.tan α / (1 + (Real.tan α)^2) = -1/3 := by
  sorry

end NUMINAMATH_CALUDE_trigonometric_identities_l341_34184


namespace NUMINAMATH_CALUDE_expected_outcome_is_negative_two_thirds_l341_34173

/-- Represents the sides of the die --/
inductive DieSide
| A
| B
| C

/-- The probability of rolling each side of the die --/
def probability (side : DieSide) : ℚ :=
  match side with
  | DieSide.A => 1/3
  | DieSide.B => 1/2
  | DieSide.C => 1/6

/-- The monetary outcome of rolling each side of the die --/
def monetaryOutcome (side : DieSide) : ℚ :=
  match side with
  | DieSide.A => 2
  | DieSide.B => -4
  | DieSide.C => 6

/-- The expected monetary outcome of rolling the die --/
def expectedOutcome : ℚ :=
  (probability DieSide.A * monetaryOutcome DieSide.A) +
  (probability DieSide.B * monetaryOutcome DieSide.B) +
  (probability DieSide.C * monetaryOutcome DieSide.C)

/-- Theorem stating that the expected monetary outcome is -2/3 --/
theorem expected_outcome_is_negative_two_thirds :
  expectedOutcome = -2/3 := by
  sorry

end NUMINAMATH_CALUDE_expected_outcome_is_negative_two_thirds_l341_34173


namespace NUMINAMATH_CALUDE_complex_magnitude_equation_l341_34115

theorem complex_magnitude_equation (x : ℝ) :
  x > 0 ∧ Complex.abs (3 + x * Complex.I) = 7 ↔ x = 2 * Real.sqrt 10 := by
  sorry

end NUMINAMATH_CALUDE_complex_magnitude_equation_l341_34115


namespace NUMINAMATH_CALUDE_multiply_powers_of_a_l341_34170

theorem multiply_powers_of_a (a : ℝ) : -2 * a^3 * (3 * a^2) = -6 * a^5 := by
  sorry

end NUMINAMATH_CALUDE_multiply_powers_of_a_l341_34170


namespace NUMINAMATH_CALUDE_discount_percent_calculation_l341_34153

theorem discount_percent_calculation (MP : ℝ) (CP : ℝ) (h1 : CP = 0.64 * MP) (h2 : 34.375 = (CP * 1.34375 - CP) / CP * 100) :
  (MP - CP * 1.34375) / MP * 100 = 14 := by
  sorry

end NUMINAMATH_CALUDE_discount_percent_calculation_l341_34153


namespace NUMINAMATH_CALUDE_arthurs_dinner_cost_l341_34158

def dinner_cost (appetizer steak wine_glass dessert : ℚ) (wine_glasses : ℕ) (discount_percent tip_percent : ℚ) : ℚ :=
  let full_cost := appetizer + steak + (wine_glass * wine_glasses) + dessert
  let discount := steak * discount_percent
  let discounted_cost := full_cost - discount
  let tip := full_cost * tip_percent
  discounted_cost + tip

theorem arthurs_dinner_cost :
  dinner_cost 8 20 3 6 2 (1/2) (1/5) = 38 := by
  sorry

end NUMINAMATH_CALUDE_arthurs_dinner_cost_l341_34158


namespace NUMINAMATH_CALUDE_isosceles_triangle_relationship_l341_34142

/-- Represents an isosceles triangle with given perimeter and slant length -/
structure IsoscelesTriangle where
  perimeter : ℝ
  slantLength : ℝ

/-- The base length of an isosceles triangle given its perimeter and slant length -/
def baseLength (triangle : IsoscelesTriangle) : ℝ :=
  triangle.perimeter - 2 * triangle.slantLength

/-- Theorem stating the functional relationship and valid range for an isosceles triangle -/
theorem isosceles_triangle_relationship (triangle : IsoscelesTriangle)
    (h_perimeter : triangle.perimeter = 12)
    (h_valid_slant : 3 < triangle.slantLength ∧ triangle.slantLength < 6) :
    baseLength triangle = 12 - 2 * triangle.slantLength ∧
    3 < triangle.slantLength ∧ triangle.slantLength < 6 := by
  sorry

end NUMINAMATH_CALUDE_isosceles_triangle_relationship_l341_34142


namespace NUMINAMATH_CALUDE_complex_number_equality_l341_34135

theorem complex_number_equality : (7 : ℂ) - 3*I - 3*(2 - 5*I) + 4*I = 1 + 16*I := by
  sorry

end NUMINAMATH_CALUDE_complex_number_equality_l341_34135


namespace NUMINAMATH_CALUDE_room_width_calculation_l341_34198

/-- Given a room with specified dimensions and paving costs, calculate its width -/
theorem room_width_calculation (length : ℝ) (cost_per_sqm : ℝ) (total_cost : ℝ) 
  (h1 : length = 5.5)
  (h2 : cost_per_sqm = 600)
  (h3 : total_cost = 12375) :
  total_cost / cost_per_sqm / length = 3.75 := by
  sorry

#check room_width_calculation

end NUMINAMATH_CALUDE_room_width_calculation_l341_34198


namespace NUMINAMATH_CALUDE_pet_store_dogs_l341_34140

/-- The number of dogs in a pet store after receiving additional dogs over two days -/
def total_dogs (initial : ℕ) (sunday_addition : ℕ) (monday_addition : ℕ) : ℕ :=
  initial + sunday_addition + monday_addition

/-- Theorem stating that starting with 2 dogs, adding 5 on Sunday and 3 on Monday results in 10 dogs -/
theorem pet_store_dogs : total_dogs 2 5 3 = 10 := by
  sorry

end NUMINAMATH_CALUDE_pet_store_dogs_l341_34140


namespace NUMINAMATH_CALUDE_cell_phone_customers_l341_34174

theorem cell_phone_customers (total : ℕ) (us_customers : ℕ) 
  (h1 : total = 7422) 
  (h2 : us_customers = 723) : 
  total - us_customers = 6699 := by
  sorry

end NUMINAMATH_CALUDE_cell_phone_customers_l341_34174


namespace NUMINAMATH_CALUDE_expression_simplification_and_evaluation_l341_34122

theorem expression_simplification_and_evaluation (x : ℝ) (h1 : x ≠ -2) (h2 : x ≠ -1) (h3 : x ≠ 1) :
  (1 - 3 / (x + 2)) / ((x^2 - 1) / (x + 2)) = 1 / (x + 1) ∧
  (2 - 3 / (2 + 2)) / ((2^2 - 1) / (2 + 2)) = 1 / 3 :=
by sorry

end NUMINAMATH_CALUDE_expression_simplification_and_evaluation_l341_34122


namespace NUMINAMATH_CALUDE_part_one_part_two_l341_34194

-- Define the function f
def f (m : ℝ) (x : ℝ) : ℝ := |x + m| + |2 * x + 1|

-- Part I: Prove that when m = -1, f(x) ≤ 3 is equivalent to -1 ≤ x ≤ 1
theorem part_one : 
  ∀ x : ℝ, f (-1) x ≤ 3 ↔ -1 ≤ x ∧ x ≤ 1 := by sorry

-- Part II: Prove that the minimum value of f(x) is |m - 1/2|
theorem part_two (m : ℝ) : 
  ∃ x : ℝ, ∀ y : ℝ, f m x ≤ f m y ∧ f m x = |m - 1/2| := by sorry

end NUMINAMATH_CALUDE_part_one_part_two_l341_34194


namespace NUMINAMATH_CALUDE_time_calculation_correct_l341_34146

/-- Represents a date and time -/
structure DateTime where
  year : Nat
  month : Nat
  day : Nat
  hour : Nat
  minute : Nat

/-- Adds minutes to a DateTime -/
def addMinutes (dt : DateTime) (minutes : Nat) : DateTime :=
  sorry

/-- The start date and time -/
def startDateTime : DateTime :=
  { year := 2021, month := 1, day := 5, hour := 15, minute := 0 }

/-- The number of minutes to add -/
def minutesToAdd : Nat := 5050

/-- The expected end date and time -/
def expectedEndDateTime : DateTime :=
  { year := 2021, month := 1, day := 9, hour := 3, minute := 10 }

theorem time_calculation_correct :
  addMinutes startDateTime minutesToAdd = expectedEndDateTime := by sorry

end NUMINAMATH_CALUDE_time_calculation_correct_l341_34146


namespace NUMINAMATH_CALUDE_parallelogram_properties_l341_34188

/-- Represents a parallelogram with given dimensions -/
structure Parallelogram where
  base : ℝ
  height : ℝ
  total_side : ℝ

/-- Calculate the area of a parallelogram -/
def area (p : Parallelogram) : ℝ := p.base * p.height

/-- Calculate the slant height of a parallelogram -/
def slant_height (p : Parallelogram) : ℝ := p.total_side - p.height

theorem parallelogram_properties (p : Parallelogram) 
  (h_base : p.base = 20)
  (h_height : p.height = 6)
  (h_total_side : p.total_side = 9) :
  area p = 120 ∧ slant_height p = 3 := by
  sorry


end NUMINAMATH_CALUDE_parallelogram_properties_l341_34188


namespace NUMINAMATH_CALUDE_quadratic_roots_l341_34185

theorem quadratic_roots : ∃ (x₁ x₂ : ℝ), x₁ = 2 ∧ x₂ = 3 ∧ 
  (∀ x : ℝ, x^2 - 5*x + 6 = 0 ↔ (x = x₁ ∨ x = x₂)) := by
  sorry

end NUMINAMATH_CALUDE_quadratic_roots_l341_34185


namespace NUMINAMATH_CALUDE_binomial_coefficient_equality_l341_34179

theorem binomial_coefficient_equality (n : ℕ) : 
  (Nat.choose 18 (3*n + 6) = Nat.choose 18 (4*n - 2)) ↔ n = 2 :=
sorry

end NUMINAMATH_CALUDE_binomial_coefficient_equality_l341_34179


namespace NUMINAMATH_CALUDE_completing_square_equivalence_l341_34139

theorem completing_square_equivalence :
  ∀ x : ℝ, (x^2 - 6*x - 1 = 0) ↔ ((x - 3)^2 = 10) := by sorry

end NUMINAMATH_CALUDE_completing_square_equivalence_l341_34139


namespace NUMINAMATH_CALUDE_circle_angle_constraint_l341_34164

-- Define the circle C
def C (x y : ℝ) : Prop := (x - 6)^2 + (y - 8)^2 = 1

-- Define points A and B
def A (m : ℝ) : ℝ × ℝ := (-m, 0)
def B (m : ℝ) : ℝ × ℝ := (m, 0)

-- Define the angle APB
def angle_APB (m : ℝ) (P : ℝ × ℝ) : ℝ := sorry

-- Main theorem
theorem circle_angle_constraint (m : ℝ) :
  m > 0 →
  (∀ P : ℝ × ℝ, C P.1 P.2 → angle_APB m P < 90) →
  9 < m ∧ m < 11 :=
sorry

end NUMINAMATH_CALUDE_circle_angle_constraint_l341_34164


namespace NUMINAMATH_CALUDE_prop_p_iff_prop_q_l341_34102

theorem prop_p_iff_prop_q (m : ℝ) :
  (∀ x : ℝ, |x + 2| + |x - 1| ≥ m) ↔
  (∃ x : ℝ, x^2 - 2*m*x + m^2 + m - 3 = 0) :=
by sorry

end NUMINAMATH_CALUDE_prop_p_iff_prop_q_l341_34102


namespace NUMINAMATH_CALUDE_set_equality_l341_34134

open Set

def U : Set ℝ := univ
def M : Set ℝ := {x | x < 1}
def N : Set ℝ := {x | -1 < x ∧ x < 2}

theorem set_equality : {x : ℝ | x ≥ 2} = (M ∪ N)ᶜ := by sorry

end NUMINAMATH_CALUDE_set_equality_l341_34134


namespace NUMINAMATH_CALUDE_at_least_one_genuine_certain_l341_34108

theorem at_least_one_genuine_certain (total : ℕ) (genuine : ℕ) (defective : ℕ) (selected : ℕ)
  (h1 : total = 8)
  (h2 : genuine = 5)
  (h3 : defective = 3)
  (h4 : total = genuine + defective)
  (h5 : selected = 4) :
  ∀ (selection : Finset (Fin total)),
    selection.card = selected →
    ∃ (i : Fin total), i ∈ selection ∧ i.val < genuine :=
by sorry

end NUMINAMATH_CALUDE_at_least_one_genuine_certain_l341_34108


namespace NUMINAMATH_CALUDE_inequality_solution_range_l341_34106

theorem inequality_solution_range (a : ℝ) : 
  (∃ x : ℝ, x + |x - 1| ≤ a) → a ≥ 1 := by sorry

end NUMINAMATH_CALUDE_inequality_solution_range_l341_34106


namespace NUMINAMATH_CALUDE_product_of_primes_l341_34165

theorem product_of_primes (a b c d : ℕ) : 
  Prime a ∧ Prime b ∧ Prime c ∧ Prime d ∧  -- a, b, c, d are prime
  a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d ∧  -- a, b, c, d are distinct
  a + c = d ∧  -- condition (i)
  a * (a + b + c + d) = c * (d - b) ∧  -- condition (ii)
  1 + b * c + d = b * d  -- condition (iii)
  → a * b * c * d = 2002 := by
sorry

end NUMINAMATH_CALUDE_product_of_primes_l341_34165


namespace NUMINAMATH_CALUDE_fence_cost_per_foot_l341_34192

theorem fence_cost_per_foot 
  (area : ℝ) 
  (total_cost : ℝ) 
  (h1 : area = 81) 
  (h2 : total_cost = 2088) : 
  (total_cost / (4 * Real.sqrt area)) = 58 := by
  sorry

end NUMINAMATH_CALUDE_fence_cost_per_foot_l341_34192


namespace NUMINAMATH_CALUDE_vector_c_value_l341_34176

def a : ℝ × ℝ := (2, -1)
def b : ℝ × ℝ := (-3, 5)

theorem vector_c_value (c : ℝ × ℝ) : 
  3 • a + (4 • b - a) + 2 • c = (0, 0) → c = (4, -9) := by
  sorry

end NUMINAMATH_CALUDE_vector_c_value_l341_34176


namespace NUMINAMATH_CALUDE_negation_of_all_teachers_generous_l341_34125

-- Define the universe of discourse
variable (U : Type)

-- Define predicates for being a teacher and being generous
variable (teacher : U → Prop)
variable (generous : U → Prop)

-- State the theorem
theorem negation_of_all_teachers_generous :
  (¬ ∀ x, teacher x → generous x) ↔ (∃ x, teacher x ∧ ¬ generous x) :=
by sorry

end NUMINAMATH_CALUDE_negation_of_all_teachers_generous_l341_34125


namespace NUMINAMATH_CALUDE_speed_conversion_l341_34155

/-- Conversion factor from meters per second to kilometers per hour -/
def mps_to_kmph : ℝ := 3.6

/-- Given speed in meters per second -/
def given_speed : ℝ := 20

/-- Theorem: Converting 20 mps to kmph results in 72 kmph -/
theorem speed_conversion :
  given_speed * mps_to_kmph = 72 := by sorry

end NUMINAMATH_CALUDE_speed_conversion_l341_34155


namespace NUMINAMATH_CALUDE_factor_calculation_l341_34114

theorem factor_calculation (f : ℚ) : f * (2 * 16 + 5) = 111 ↔ f = 3 := by sorry

end NUMINAMATH_CALUDE_factor_calculation_l341_34114


namespace NUMINAMATH_CALUDE_polar_to_rectangular_coordinates_l341_34162

theorem polar_to_rectangular_coordinates :
  let r : ℝ := 2
  let θ : ℝ := π / 3
  let x : ℝ := r * Real.cos θ
  let y : ℝ := r * Real.sin θ
  (x = 1 ∧ y = Real.sqrt 3) :=
by sorry

end NUMINAMATH_CALUDE_polar_to_rectangular_coordinates_l341_34162


namespace NUMINAMATH_CALUDE_quadratic_roots_property_l341_34128

theorem quadratic_roots_property (k : ℝ) : 
  (∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ 
    x₁^2 - 4*x₁ + k + 1 = 0 ∧ 
    x₂^2 - 4*x₂ + k + 1 = 0 ∧
    3/x₁ + 3/x₂ = x₁*x₂ - 4) →
  k = -3 ∧ k ≤ 3 := by
sorry

end NUMINAMATH_CALUDE_quadratic_roots_property_l341_34128


namespace NUMINAMATH_CALUDE_jane_score_is_12_l341_34143

/-- Represents the score calculation for a modified AMC 8 contest --/
def modified_amc_score (correct : ℕ) (incorrect : ℕ) (unanswered : ℕ) : ℚ :=
  (correct : ℚ) - (incorrect : ℚ) / 2

/-- Theorem stating that Jane's score in the modified AMC 8 contest is 12 --/
theorem jane_score_is_12 :
  let total_questions : ℕ := 35
  let correct_answers : ℕ := 18
  let incorrect_answers : ℕ := 12
  let unanswered_questions : ℕ := 5
  modified_amc_score correct_answers incorrect_answers unanswered_questions = 12 := by
  sorry

#eval modified_amc_score 18 12 5

end NUMINAMATH_CALUDE_jane_score_is_12_l341_34143


namespace NUMINAMATH_CALUDE_lines_parallel_implies_a_eq_one_lines_perpendicular_implies_a_eq_zero_l341_34112

/-- Two lines in the xy-plane -/
structure Line2D where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Define the first line l₁: x + ay - 2a - 2 = 0 -/
def l₁ (a : ℝ) : Line2D := ⟨1, a, -2*a - 2⟩

/-- Define the second line l₂: ax + y - 1 - a = 0 -/
def l₂ (a : ℝ) : Line2D := ⟨a, 1, -1 - a⟩

/-- Two lines are parallel if their slopes are equal -/
def parallel (l₁ l₂ : Line2D) : Prop := l₁.a * l₂.b = l₂.a * l₁.b

/-- Two lines are perpendicular if the product of their slopes is -1 -/
def perpendicular (l₁ l₂ : Line2D) : Prop := l₁.a * l₂.a + l₁.b * l₂.b = 0

theorem lines_parallel_implies_a_eq_one :
  ∀ a : ℝ, parallel (l₁ a) (l₂ a) → a = 1 := by sorry

theorem lines_perpendicular_implies_a_eq_zero :
  ∀ a : ℝ, perpendicular (l₁ a) (l₂ a) → a = 0 := by sorry

end NUMINAMATH_CALUDE_lines_parallel_implies_a_eq_one_lines_perpendicular_implies_a_eq_zero_l341_34112


namespace NUMINAMATH_CALUDE_probability_identical_value_l341_34189

/-- Represents the colors that can be used to paint a cube face -/
inductive Color
| Red
| Blue

/-- Represents a cube with painted faces -/
def Cube := Fin 6 → Color

/-- Checks if two cubes are identical after rotation -/
def identical_after_rotation (cube1 cube2 : Cube) : Prop := sorry

/-- The set of all possible cube paintings -/
def all_cubes : Set Cube := sorry

/-- The set of pairs of cubes that are identical after rotation -/
def identical_pairs : Set (Cube × Cube) := sorry

/-- The probability of two independently painted cubes being identical after rotation -/
def probability_identical : ℚ := sorry

theorem probability_identical_value :
  probability_identical = 459 / 4096 := by sorry

end NUMINAMATH_CALUDE_probability_identical_value_l341_34189


namespace NUMINAMATH_CALUDE_intersection_equals_open_closed_interval_l341_34175

def A : Set ℝ := {x | x^2 - 4*x + 3 < 0}

def B : Set ℝ := {x | ∃ y, y = Real.log (x - 2)}

def C_R_B : Set ℝ := (Set.univ : Set ℝ) \ B

theorem intersection_equals_open_closed_interval : (C_R_B ∩ A) = {x | 1 < x ∧ x ≤ 2} := by sorry

end NUMINAMATH_CALUDE_intersection_equals_open_closed_interval_l341_34175


namespace NUMINAMATH_CALUDE_west_notation_l341_34119

-- Define a type for direction
inductive Direction
  | East
  | West

-- Define a function that represents the notation for walking in a given direction
def walkNotation (dir : Direction) (distance : ℝ) : ℝ :=
  match dir with
  | Direction.East => distance
  | Direction.West => -distance

-- Theorem statement
theorem west_notation (d : ℝ) :
  walkNotation Direction.East d = d →
  walkNotation Direction.West d = -d :=
by sorry

end NUMINAMATH_CALUDE_west_notation_l341_34119


namespace NUMINAMATH_CALUDE_student_height_correction_l341_34166

theorem student_height_correction (n : ℕ) (initial_avg : ℝ) (incorrect_height : ℝ) (actual_avg : ℝ) :
  n = 20 →
  initial_avg = 175 →
  incorrect_height = 151 →
  actual_avg = 174.25 →
  ∃ (actual_height : ℝ), 
    actual_height = 166 ∧
    n * initial_avg = (n - 1) * actual_avg + incorrect_height ∧
    n * actual_avg = (n - 1) * actual_avg + actual_height :=
by sorry

end NUMINAMATH_CALUDE_student_height_correction_l341_34166


namespace NUMINAMATH_CALUDE_card_ratio_proof_l341_34172

theorem card_ratio_proof (total_cards baseball_cards : ℕ) 
  (h1 : total_cards = 125)
  (h2 : baseball_cards = 95) : 
  (baseball_cards : ℚ) / (total_cards - baseball_cards) = 19 / 6 := by
  sorry

end NUMINAMATH_CALUDE_card_ratio_proof_l341_34172


namespace NUMINAMATH_CALUDE_treys_chores_l341_34160

theorem treys_chores (clean_house_tasks : ℕ) (shower_tasks : ℕ) (dinner_tasks : ℕ) 
  (total_time_hours : ℕ) (h1 : clean_house_tasks = 7) (h2 : shower_tasks = 1) 
  (h3 : dinner_tasks = 4) (h4 : total_time_hours = 2) : 
  (total_time_hours * 60) / (clean_house_tasks + shower_tasks + dinner_tasks) = 10 := by
  sorry

end NUMINAMATH_CALUDE_treys_chores_l341_34160


namespace NUMINAMATH_CALUDE_wang_trip_distance_l341_34147

/-- The distance between Mr. Wang's home and location A -/
def distance : ℝ := 330

theorem wang_trip_distance : 
  ∀ x : ℝ, 
  x > 0 → 
  (x / 100 + x / 120) - (x / 150 + 2 * x / 198) = 31 / 60 → 
  x = distance := by
sorry

end NUMINAMATH_CALUDE_wang_trip_distance_l341_34147


namespace NUMINAMATH_CALUDE_log3_one_third_l341_34196

-- Define the logarithm function for base 3
noncomputable def log3 (x : ℝ) := Real.log x / Real.log 3

-- State the theorem
theorem log3_one_third : log3 (1/3) = -1 := by
  sorry

end NUMINAMATH_CALUDE_log3_one_third_l341_34196


namespace NUMINAMATH_CALUDE_triangle_side_value_l341_34118

/-- A triangle with sides a, b, and c satisfies the triangle inequality -/
def is_triangle (a b c : ℝ) : Prop :=
  a + b > c ∧ b + c > a ∧ c + a > b

/-- The set of possible values for x -/
def possible_values : Set ℕ := {2, 4, 6, 8}

/-- The theorem statement -/
theorem triangle_side_value (x : ℕ) (hx : x ∈ possible_values) :
  is_triangle 2 x 6 ↔ x = 6 := by
  sorry


end NUMINAMATH_CALUDE_triangle_side_value_l341_34118


namespace NUMINAMATH_CALUDE_paintbrush_cost_l341_34126

/-- The amount spent on paintbrushes given the total spent and costs of other items. -/
theorem paintbrush_cost (total_spent : ℝ) (canvas_cost : ℝ) (paint_cost : ℝ) (easel_cost : ℝ) 
  (h1 : total_spent = 90)
  (h2 : canvas_cost = 40)
  (h3 : paint_cost = canvas_cost / 2)
  (h4 : easel_cost = 15) :
  total_spent - (canvas_cost + paint_cost + easel_cost) = 15 :=
by sorry

end NUMINAMATH_CALUDE_paintbrush_cost_l341_34126


namespace NUMINAMATH_CALUDE_parity_of_F_l341_34127

/-- F(n) is the number of ways to express n as the sum of three different positive integers -/
def F (n : ℕ) : ℕ := sorry

/-- Main theorem about the parity of F(n) -/
theorem parity_of_F (n : ℕ) (hn : n > 0) :
  (n % 6 = 2 ∨ n % 6 = 4 → F n % 2 = 0) ∧
  (n % 6 = 0 → F n % 2 = 1) := by
  sorry

end NUMINAMATH_CALUDE_parity_of_F_l341_34127


namespace NUMINAMATH_CALUDE_total_rent_is_435_l341_34178

/-- Represents the rent calculation for a pasture shared by multiple parties -/
structure PastureRent where
  a_horses : ℕ
  a_months : ℕ
  b_horses : ℕ
  b_months : ℕ
  c_horses : ℕ
  c_months : ℕ
  b_payment : ℕ

/-- Calculates the total rent for the pasture -/
def calculate_total_rent (pr : PastureRent) : ℕ :=
  let total_horse_months := pr.a_horses * pr.a_months + pr.b_horses * pr.b_months + pr.c_horses * pr.c_months
  let b_horse_months := pr.b_horses * pr.b_months
  (pr.b_payment * total_horse_months) / b_horse_months

/-- Theorem stating that the total rent for the given conditions is 435 -/
theorem total_rent_is_435 (pr : PastureRent) 
  (h1 : pr.a_horses = 12) (h2 : pr.a_months = 8)
  (h3 : pr.b_horses = 16) (h4 : pr.b_months = 9)
  (h5 : pr.c_horses = 18) (h6 : pr.c_months = 6)
  (h7 : pr.b_payment = 180) : 
  calculate_total_rent pr = 435 := by
  sorry

end NUMINAMATH_CALUDE_total_rent_is_435_l341_34178


namespace NUMINAMATH_CALUDE_inequality_proof_l341_34145

theorem inequality_proof (a b c : ℝ) (h : a ≠ b) :
  Real.sqrt ((a - c)^2 + b^2) + Real.sqrt (a^2 + (b - c)^2) > Real.sqrt 2 * abs (a - b) := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l341_34145


namespace NUMINAMATH_CALUDE_complement_A_equals_negative_reals_l341_34129

-- Define the universe U as the set of real numbers
def U : Set ℝ := Set.univ

-- Define set A as the set of non-negative real numbers
def A : Set ℝ := { x : ℝ | x ≥ 0 }

-- Define the complement of A in U
def complement_A : Set ℝ := U \ A

-- Theorem statement
theorem complement_A_equals_negative_reals :
  complement_A = { x : ℝ | x < 0 } :=
sorry

end NUMINAMATH_CALUDE_complement_A_equals_negative_reals_l341_34129


namespace NUMINAMATH_CALUDE_function_maximum_implies_a_range_l341_34152

/-- Given a function f(x) = 4x³ - 3x with a maximum in the interval (a, a+2), prove that a is in the range (-5/2, -1]. -/
theorem function_maximum_implies_a_range 
  (f : ℝ → ℝ) 
  (a : ℝ) 
  (h₁ : ∀ x, f x = 4 * x^3 - 3 * x)
  (h₂ : ∃ x₀ ∈ Set.Ioo a (a + 2), ∀ x ∈ Set.Ioo a (a + 2), f x ≤ f x₀) :
  a ∈ Set.Ioc (-5/2) (-1) :=
sorry

end NUMINAMATH_CALUDE_function_maximum_implies_a_range_l341_34152


namespace NUMINAMATH_CALUDE_complex_modulus_problem_l341_34190

theorem complex_modulus_problem (i : ℂ) (h : i * i = -1) :
  Complex.abs (1 / (1 - i) + i) = Real.sqrt 10 / 2 := by
  sorry

end NUMINAMATH_CALUDE_complex_modulus_problem_l341_34190


namespace NUMINAMATH_CALUDE_min_value_x_plus_y_l341_34195

theorem min_value_x_plus_y (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : x^2 + 3*x*y - 2 = 0) :
  ∃ (m : ℝ), m = 4/3 ∧ ∀ (a b : ℝ), a > 0 → b > 0 → a^2 + 3*a*b - 2 = 0 → x + y ≤ a + b :=
sorry

end NUMINAMATH_CALUDE_min_value_x_plus_y_l341_34195


namespace NUMINAMATH_CALUDE_brick_length_l341_34101

/-- The surface area of a rectangular prism -/
def surface_area (l w h : ℝ) : ℝ := 2 * l * w + 2 * l * h + 2 * w * h

/-- Theorem: For a brick with width 4 cm, height 2 cm, and surface area 136 square centimeters, the length of the brick is 10 cm -/
theorem brick_length : 
  ∃ (l : ℝ), surface_area l 4 2 = 136 ∧ l = 10 :=
sorry

end NUMINAMATH_CALUDE_brick_length_l341_34101


namespace NUMINAMATH_CALUDE_problem_solution_l341_34167

theorem problem_solution (α β : Real) 
  (h1 : α ∈ Set.Ioo 0 (Real.pi / 2))
  (h2 : Real.cos (2 * α) = 4 / 5)
  (h3 : β ∈ Set.Ioo (Real.pi / 2) Real.pi)
  (h4 : 5 * Real.sin (2 * α + β) = Real.sin β) : 
  (Real.sin α + Real.cos α = 2 * Real.sqrt 10 / 5) ∧ 
  (β = 3 * Real.pi / 4) := by
sorry


end NUMINAMATH_CALUDE_problem_solution_l341_34167


namespace NUMINAMATH_CALUDE_exists_continuous_surjective_non_monotonic_l341_34193

/-- A continuous function from ℝ to ℝ with full range that is not monotonic -/
theorem exists_continuous_surjective_non_monotonic :
  ∃ f : ℝ → ℝ, Continuous f ∧ Function.Surjective f ∧ ¬Monotone f := by
  sorry

end NUMINAMATH_CALUDE_exists_continuous_surjective_non_monotonic_l341_34193


namespace NUMINAMATH_CALUDE_geometric_series_common_ratio_l341_34111

theorem geometric_series_common_ratio :
  ∀ (a r : ℚ),
    a = 4/7 →
    a * r = 16/21 →
    r = 4/3 :=
by
  sorry

end NUMINAMATH_CALUDE_geometric_series_common_ratio_l341_34111


namespace NUMINAMATH_CALUDE_stone_pile_impossibility_l341_34113

theorem stone_pile_impossibility :
  ∀ (n : ℕ) (stones piles : ℕ → ℕ),
  (stones 0 = 1001 ∧ piles 0 = 1) →
  (∀ k, stones (k + 1) + piles (k + 1) = stones k + piles k) →
  (∀ k, stones (k + 1) = stones k - 1) →
  (∀ k, piles (k + 1) = piles k + 1) →
  ¬∃ k, stones k = 3 * piles k :=
by sorry

end NUMINAMATH_CALUDE_stone_pile_impossibility_l341_34113


namespace NUMINAMATH_CALUDE_middle_number_problem_l341_34182

theorem middle_number_problem (x y z : ℕ) : 
  x < y → y < z → x + y = 20 → x + z = 25 → y + z = 29 → y = 12 := by
  sorry

end NUMINAMATH_CALUDE_middle_number_problem_l341_34182


namespace NUMINAMATH_CALUDE_max_2012_gons_sharing_vertices_not_sides_target_2012_gons_impossible_l341_34132

/-- The number of vertices in each polygon -/
def n : ℕ := 2012

/-- The maximum number of different polygons we want to prove is impossible -/
def target_polygons : ℕ := 1006

/-- The actual maximum number of polygons possible -/
def max_polygons : ℕ := 1005

theorem max_2012_gons_sharing_vertices_not_sides :
  ∀ (num_polygons : ℕ),
    (∀ (v : Fin n), num_polygons * 2 ≤ n - 1) →
    num_polygons ≤ max_polygons :=
by sorry

theorem target_2012_gons_impossible :
  ¬(∀ (v : Fin n), target_polygons * 2 ≤ n - 1) :=
by sorry

end NUMINAMATH_CALUDE_max_2012_gons_sharing_vertices_not_sides_target_2012_gons_impossible_l341_34132


namespace NUMINAMATH_CALUDE_john_travel_time_l341_34168

/-- Proves that given a distance of 24 km and a normal travel time of 44 minutes,
    if a speed of 40 kmph results in arriving 8 minutes early,
    then a speed of 30 kmph will result in arriving 4 minutes late. -/
theorem john_travel_time (distance : ℝ) (normal_time : ℝ) (early_speed : ℝ) (late_speed : ℝ) :
  distance = 24 →
  normal_time = 44 / 60 →
  early_speed = 40 →
  late_speed = 30 →
  distance / early_speed = normal_time - 8 / 60 →
  distance / late_speed = normal_time + 4 / 60 :=
by sorry

end NUMINAMATH_CALUDE_john_travel_time_l341_34168


namespace NUMINAMATH_CALUDE_largest_prime_factors_difference_l341_34121

theorem largest_prime_factors_difference (n : Nat) (h : n = 180181) :
  ∃ (p q : Nat), Nat.Prime p ∧ Nat.Prime q ∧ p > q ∧
  p ∣ n ∧ q ∣ n ∧
  (∀ r : Nat, Nat.Prime r → r ∣ n → r ≤ p) ∧
  (∀ r : Nat, Nat.Prime r → r ∣ n → r ≠ p → r ≤ q) ∧
  p - q = 2 :=
by sorry

end NUMINAMATH_CALUDE_largest_prime_factors_difference_l341_34121


namespace NUMINAMATH_CALUDE_f_bounds_f_inequality_solution_set_l341_34159

def f (x : ℝ) : ℝ := |x - 2| - |x - 5|

theorem f_bounds : ∀ x : ℝ, -3 ≤ f x ∧ f x ≤ 3 := by sorry

theorem f_inequality_solution_set :
  {x : ℝ | f x ≥ x^2 - 8*x + 15} = {x : ℝ | 5 - Real.sqrt 3 ≤ x ∧ x ≤ 6} := by sorry

end NUMINAMATH_CALUDE_f_bounds_f_inequality_solution_set_l341_34159


namespace NUMINAMATH_CALUDE_custom_mul_solution_l341_34187

/-- Custom multiplication operation -/
def custom_mul (a b : ℝ) : ℝ := 2*a - b^2

/-- Theorem stating that if a * 3 = 3 under the custom multiplication, then a = 6 -/
theorem custom_mul_solution :
  ∀ a : ℝ, custom_mul a 3 = 3 → a = 6 := by
  sorry

end NUMINAMATH_CALUDE_custom_mul_solution_l341_34187


namespace NUMINAMATH_CALUDE_plate_arrangement_theorem_l341_34105

/-- The number of ways to arrange plates around a circular table. -/
def circularArrangements (blue red green yellow : ℕ) : ℕ :=
  Nat.factorial (blue + red + green + yellow - 1) /
  (Nat.factorial blue * Nat.factorial red * Nat.factorial green * Nat.factorial yellow)

/-- The number of ways to arrange plates around a circular table with adjacent green plates. -/
def circularArrangementsWithAdjacentGreen (blue red green yellow : ℕ) : ℕ :=
  Nat.factorial (blue + red + 1 + yellow - 1) /
  (Nat.factorial blue * Nat.factorial red * Nat.factorial yellow) *
  Nat.factorial green

/-- The number of ways to arrange plates around a circular table without adjacent green plates. -/
def circularArrangementsWithoutAdjacentGreen (blue red green yellow : ℕ) : ℕ :=
  circularArrangements blue red green yellow -
  circularArrangementsWithAdjacentGreen blue red green yellow

theorem plate_arrangement_theorem :
  circularArrangementsWithoutAdjacentGreen 4 3 3 1 = 2520 :=
by sorry

end NUMINAMATH_CALUDE_plate_arrangement_theorem_l341_34105


namespace NUMINAMATH_CALUDE_function_nonnegative_implies_inequalities_l341_34109

/-- Given real constants a, b, A, B, and a function f(θ) = 1 - a cos θ - b sin θ - A sin 2θ - B cos 2θ,
    if f(θ) ≥ 0 for all real θ, then a² + b² ≤ 2 and A² + B² ≤ 1. -/
theorem function_nonnegative_implies_inequalities (a b A B : ℝ) :
  (∀ θ : ℝ, 1 - a * Real.cos θ - b * Real.sin θ - A * Real.sin (2 * θ) - B * Real.cos (2 * θ) ≥ 0) →
  a^2 + b^2 ≤ 2 ∧ A^2 + B^2 ≤ 1 := by
  sorry

end NUMINAMATH_CALUDE_function_nonnegative_implies_inequalities_l341_34109


namespace NUMINAMATH_CALUDE_isosceles_not_unique_l341_34154

/-- Represents a triangle with side lengths a, b, c and angles A, B, C --/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

/-- Predicate to check if a triangle is isosceles --/
def IsIsosceles (t : Triangle) : Prop :=
  t.a = t.b ∨ t.b = t.c ∨ t.c = t.a

/-- Predicate to check if two triangles are non-congruent --/
def AreNonCongruent (t1 t2 : Triangle) : Prop :=
  t1.a ≠ t2.a ∨ t1.b ≠ t2.b ∨ t1.c ≠ t2.c

/-- Theorem stating that a base angle and opposite side do not uniquely determine an isosceles triangle --/
theorem isosceles_not_unique (θ : ℝ) (s : ℝ) :
  ∃ (t1 t2 : Triangle), IsIsosceles t1 ∧ IsIsosceles t2 ∧ 
  AreNonCongruent t1 t2 ∧
  ((t1.A = θ ∧ t1.a = s) ∨ (t1.B = θ ∧ t1.b = s) ∨ (t1.C = θ ∧ t1.c = s)) ∧
  ((t2.A = θ ∧ t2.a = s) ∨ (t2.B = θ ∧ t2.b = s) ∨ (t2.C = θ ∧ t2.c = s)) :=
sorry

end NUMINAMATH_CALUDE_isosceles_not_unique_l341_34154


namespace NUMINAMATH_CALUDE_lemonade_stand_problem_l341_34144

theorem lemonade_stand_problem (bea_price dawn_price : ℚ) (bea_glasses : ℕ) (earnings_difference : ℚ) :
  bea_price = 25 / 100 →
  dawn_price = 28 / 100 →
  bea_glasses = 10 →
  earnings_difference = 26 / 100 →
  ∃ dawn_glasses : ℕ,
    dawn_glasses = 8 ∧
    bea_price * bea_glasses = dawn_price * dawn_glasses + earnings_difference :=
by
  sorry

#check lemonade_stand_problem

end NUMINAMATH_CALUDE_lemonade_stand_problem_l341_34144


namespace NUMINAMATH_CALUDE_min_blocks_for_wall_l341_34133

/-- Represents a block in the wall -/
inductive Block
| OneFootBlock
| TwoFootBlock

/-- Represents a row of blocks in the wall -/
def Row := List Block

/-- The wall specification -/
structure WallSpec where
  length : Nat
  height : Nat
  blockHeight : Nat
  evenEnds : Bool
  staggeredJoins : Bool

/-- Checks if a row of blocks is valid according to the wall specification -/
def isValidRow (spec : WallSpec) (row : Row) : Prop := sorry

/-- Checks if a list of rows forms a valid wall according to the specification -/
def isValidWall (spec : WallSpec) (rows : List Row) : Prop := sorry

/-- Counts the total number of blocks in a list of rows -/
def countBlocks (rows : List Row) : Nat := sorry

/-- The main theorem to be proved -/
theorem min_blocks_for_wall (spec : WallSpec) : 
  spec.length = 102 ∧ 
  spec.height = 8 ∧ 
  spec.blockHeight = 1 ∧ 
  spec.evenEnds = true ∧ 
  spec.staggeredJoins = true → 
  ∃ (rows : List Row), 
    isValidWall spec rows ∧ 
    countBlocks rows = 416 ∧ 
    ∀ (otherRows : List Row), 
      isValidWall spec otherRows → 
      countBlocks otherRows ≥ 416 := by sorry

end NUMINAMATH_CALUDE_min_blocks_for_wall_l341_34133


namespace NUMINAMATH_CALUDE_inverse_g_at_neg_138_l341_34120

-- Define the function g
def g (x : ℝ) : ℝ := 5 * x^3 - 3

-- State the theorem
theorem inverse_g_at_neg_138 :
  g⁻¹ (-138) = -3 :=
sorry

end NUMINAMATH_CALUDE_inverse_g_at_neg_138_l341_34120


namespace NUMINAMATH_CALUDE_skew_edge_prob_is_4_11_l341_34149

/-- A cube with 12 edges -/
structure Cube :=
  (edges : Finset (Fin 12))
  (edge_count : edges.card = 12)

/-- Two edges of a cube are skew if they don't intersect and are not in the same plane -/
def are_skew (c : Cube) (e1 e2 : Fin 12) : Prop := sorry

/-- The number of edges skew to any given edge in a cube -/
def skew_edge_count (c : Cube) : ℕ := 4

/-- The probability of selecting two skew edges from a cube -/
def skew_edge_probability (c : Cube) : ℚ :=
  (skew_edge_count c : ℚ) / (c.edges.card - 1 : ℚ)

/-- Theorem: The probability of selecting two skew edges from a cube is 4/11 -/
theorem skew_edge_prob_is_4_11 (c : Cube) : 
  skew_edge_probability c = 4 / 11 := by sorry

end NUMINAMATH_CALUDE_skew_edge_prob_is_4_11_l341_34149


namespace NUMINAMATH_CALUDE_fixed_point_unique_l341_34151

/-- The line l passes through the point (x, y) for all real values of m -/
def passes_through (x y : ℝ) : Prop :=
  ∀ m : ℝ, (2 + m) * x + (1 - 2*m) * y + (4 - 3*m) = 0

/-- The point M is the unique point that the line l passes through for all m -/
theorem fixed_point_unique :
  ∃! p : ℝ × ℝ, passes_through p.1 p.2 ∧ p = (-1, -2) :=
sorry

end NUMINAMATH_CALUDE_fixed_point_unique_l341_34151


namespace NUMINAMATH_CALUDE_tailor_cut_difference_l341_34103

theorem tailor_cut_difference (skirt_cut pants_cut : ℝ) : 
  skirt_cut = 0.75 → pants_cut = 0.5 → skirt_cut - pants_cut = 0.25 := by
  sorry

end NUMINAMATH_CALUDE_tailor_cut_difference_l341_34103


namespace NUMINAMATH_CALUDE_allStarSeatingArrangements_l341_34156

def factorial (n : ℕ) : ℕ := (List.range n).foldl (·*·+1) 1

-- Define the number of All-Stars for each team
def cubs : ℕ := 4
def redSox : ℕ := 3
def yankees : ℕ := 2

-- Define the total number of All-Stars
def totalAllStars : ℕ := cubs + redSox + yankees

-- Define the number of team blocks (excluding the fixed block)
def remainingTeamBlocks : ℕ := 2

theorem allStarSeatingArrangements :
  factorial remainingTeamBlocks * factorial cubs * factorial redSox * factorial yankees = 576 := by
  sorry

end NUMINAMATH_CALUDE_allStarSeatingArrangements_l341_34156


namespace NUMINAMATH_CALUDE_searchlight_configuration_exists_l341_34124

/-- Represents a searchlight with its position and direction --/
structure Searchlight where
  position : ℝ × ℝ
  direction : ℝ

/-- Checks if a point is within the illuminated region of a searchlight --/
def isIlluminated (s : Searchlight) (p : ℝ × ℝ) : Prop :=
  sorry

/-- Calculates the shadow length of a searchlight given a configuration --/
def shadowLength (s : Searchlight) (config : List Searchlight) : ℝ :=
  sorry

/-- Theorem: There exists a configuration of 7 searchlights where each casts a 7km shadow --/
theorem searchlight_configuration_exists : 
  ∃ (config : List Searchlight), 
    config.length = 7 ∧ 
    ∀ s ∈ config, shadowLength s config = 7 :=
  sorry

end NUMINAMATH_CALUDE_searchlight_configuration_exists_l341_34124


namespace NUMINAMATH_CALUDE_perimeter_AEC_l341_34163

/-- A square with side length 2 and vertices A, B, C, D (in order) is folded so that C meets AB at C'.
    AC' = 1/4, and BC intersects AD at E. -/
structure FoldedSquare where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ
  D : ℝ × ℝ
  C' : ℝ × ℝ
  E : ℝ × ℝ
  h_square : A = (0, 2) ∧ B = (0, 0) ∧ C = (2, 0) ∧ D = (2, 2)
  h_C'_on_AB : C'.1 = 1/4 ∧ C'.2 = 0
  h_E_on_AD : E = (0, 2)

/-- The perimeter of triangle AEC' in a folded square is (√65 + 1)/4 -/
theorem perimeter_AEC'_folded_square (fs : FoldedSquare) :
  let d (p q : ℝ × ℝ) := Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2)
  d fs.A fs.E + d fs.E fs.C' + d fs.C' fs.A = (Real.sqrt 65 + 1) / 4 := by
  sorry


end NUMINAMATH_CALUDE_perimeter_AEC_l341_34163


namespace NUMINAMATH_CALUDE_inequality_proof_l341_34197

theorem inequality_proof (x : ℝ) (h : x > 0) : Real.exp (2 * x^3) - 2*x > 2*(x+1)*Real.log x := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l341_34197


namespace NUMINAMATH_CALUDE_repeating_decimal_equality_l341_34148

/-- Represents a repeating decimal with an integer part and a repeating fractional part. -/
structure RepeatingDecimal where
  integerPart : ℚ
  repeatingPart : ℚ
  repeatingPartLength : ℕ

/-- Converts a RepeatingDecimal to a rational number. -/
def RepeatingDecimal.toRational (x : RepeatingDecimal) : ℚ :=
  x.integerPart + x.repeatingPart / (1 - (1 / 10 ^ x.repeatingPartLength))

/-- Theorem stating that 0.3̅206̅ is equal to 5057/9990 -/
theorem repeating_decimal_equality : 
  let x : RepeatingDecimal := ⟨3/10, 206/1000, 3⟩
  x.toRational = 5057 / 9990 := by
  sorry

end NUMINAMATH_CALUDE_repeating_decimal_equality_l341_34148


namespace NUMINAMATH_CALUDE_C_share_of_profit_l341_34186

def investment_A : ℕ := 24000
def investment_B : ℕ := 32000
def investment_C : ℕ := 36000
def total_profit : ℕ := 92000

theorem C_share_of_profit :
  (investment_C : ℚ) / (investment_A + investment_B + investment_C) * total_profit = 36000 := by
  sorry

end NUMINAMATH_CALUDE_C_share_of_profit_l341_34186


namespace NUMINAMATH_CALUDE_hyperbola_eccentricity_l341_34116

/-- Represents a hyperbola with semi-major axis a and semi-minor axis b -/
structure Hyperbola where
  a : ℝ
  b : ℝ
  h_pos_a : 0 < a
  h_pos_b : 0 < b

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- The eccentricity of a hyperbola with the given properties is 2 + √3 -/
theorem hyperbola_eccentricity (h : Hyperbola) (F₁ F₂ A B : Point) :
  -- F₁ and F₂ are the left and right foci respectively
  -- A line passes through F₁ at a 60° angle
  -- The line intersects the y-axis at A and the right branch of the hyperbola at B
  -- A is the midpoint of F₁B
  (∃ (θ : ℝ), θ = Real.pi / 3 ∧ 
    A.x = 0 ∧
    B.x > 0 ∧
    (B.x - F₁.x) * Real.cos θ = (B.y - F₁.y) * Real.sin θ ∧
    A.x = (F₁.x + B.x) / 2 ∧
    A.y = (F₁.y + B.y) / 2) →
  -- The eccentricity of the hyperbola is 2 + √3
  h.a / Real.sqrt (h.a^2 + h.b^2) = 2 + Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_hyperbola_eccentricity_l341_34116


namespace NUMINAMATH_CALUDE_list_number_relation_l341_34130

theorem list_number_relation (l : List ℝ) (n : ℝ) : 
  l.length = 21 ∧ 
  n ∈ l ∧ 
  n = (l.sum / 6) →
  n = 4 * ((l.sum - n) / 20) := by
sorry

end NUMINAMATH_CALUDE_list_number_relation_l341_34130


namespace NUMINAMATH_CALUDE_computer_printer_price_l341_34110

/-- The total price of a basic computer and printer -/
def total_price (basic_computer_price printer_price : ℝ) : ℝ :=
  basic_computer_price + printer_price

/-- The price of an enhanced computer -/
def enhanced_computer_price (basic_computer_price : ℝ) : ℝ :=
  basic_computer_price + 500

/-- Condition for printer price with enhanced computer -/
def printer_price_condition (basic_computer_price printer_price : ℝ) : Prop :=
  printer_price = (1/3) * (enhanced_computer_price basic_computer_price + printer_price)

theorem computer_printer_price :
  ∃ (printer_price : ℝ),
    let basic_computer_price := 1500
    printer_price_condition basic_computer_price printer_price ∧
    total_price basic_computer_price printer_price = 2500 := by
  sorry

end NUMINAMATH_CALUDE_computer_printer_price_l341_34110


namespace NUMINAMATH_CALUDE_sum_of_powers_l341_34183

theorem sum_of_powers (x : ℂ) (h1 : x^7 = 1) (h2 : x ≠ 1) :
  x^2 / (x - 1) + x^4 / (x^2 - 1) + x^6 / (x^3 - 1) = 1 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_powers_l341_34183


namespace NUMINAMATH_CALUDE_derivative_value_l341_34107

theorem derivative_value (f : ℝ → ℝ) (f' : ℝ → ℝ) (h : ∀ x, f x = 2 * x * f' 2 + x^3) :
  f' 2 = -12 := by
  sorry

end NUMINAMATH_CALUDE_derivative_value_l341_34107


namespace NUMINAMATH_CALUDE_matchstick_pattern_l341_34100

/-- 
Given a sequence where:
- The first term is 5
- Each subsequent term increases by 3
Prove that the 20th term is 62
-/
theorem matchstick_pattern (a : ℕ → ℕ) 
  (h1 : a 1 = 5)
  (h2 : ∀ n : ℕ, n ≥ 2 → a n = a (n-1) + 3) :
  a 20 = 62 := by
  sorry

end NUMINAMATH_CALUDE_matchstick_pattern_l341_34100


namespace NUMINAMATH_CALUDE_shaded_area_of_semicircle_l341_34131

theorem shaded_area_of_semicircle (total_area : ℝ) (h : total_area > 0) :
  let num_parts : ℕ := 6
  let excluded_fraction : ℝ := 2 / 3
  let shaded_area : ℝ := total_area * (1 - excluded_fraction)
  shaded_area = total_area / 3 :=
by sorry

end NUMINAMATH_CALUDE_shaded_area_of_semicircle_l341_34131

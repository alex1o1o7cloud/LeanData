import Mathlib

namespace NUMINAMATH_CALUDE_six_doctors_three_days_l861_86157

/-- The number of ways for a given number of doctors to each choose one rest day from a given number of days. -/
def restDayChoices (numDoctors : ℕ) (numDays : ℕ) : ℕ :=
  numDays ^ numDoctors

/-- Theorem stating that for 6 doctors choosing from 3 days, the number of choices is 3^6. -/
theorem six_doctors_three_days : 
  restDayChoices 6 3 = 3^6 := by
  sorry

end NUMINAMATH_CALUDE_six_doctors_three_days_l861_86157


namespace NUMINAMATH_CALUDE_popped_kernel_probability_l861_86101

theorem popped_kernel_probability (total : ℝ) (h_total : total > 0) :
  let white := (2 / 3) * total
  let yellow := (1 / 3) * total
  let white_popped := (1 / 2) * white
  let yellow_popped := (2 / 3) * yellow
  let total_popped := white_popped + yellow_popped
  (white_popped / total_popped) = (3 / 5) := by
  sorry

end NUMINAMATH_CALUDE_popped_kernel_probability_l861_86101


namespace NUMINAMATH_CALUDE_negation_of_existence_l861_86171

theorem negation_of_existence (l : ℝ) : 
  (¬ ∃ x : ℝ, x + l ≥ 0) ↔ (∀ x : ℝ, x + l < 0) :=
sorry

end NUMINAMATH_CALUDE_negation_of_existence_l861_86171


namespace NUMINAMATH_CALUDE_direction_vector_x_component_l861_86116

/-- Given a line passing through two points with a specific direction vector form, prove the value of the direction vector's x-component. -/
theorem direction_vector_x_component
  (p1 : ℝ × ℝ)
  (p2 : ℝ × ℝ)
  (h1 : p1 = (-3, 6))
  (h2 : p2 = (2, -1))
  (h3 : ∃ (a : ℝ), (a, -1) = (p2.1 - p1.1, p2.2 - p1.2)) :
  ∃ (a : ℝ), (a, -1) = (p2.1 - p1.1, p2.2 - p1.2) ∧ a = -5/7 := by
sorry


end NUMINAMATH_CALUDE_direction_vector_x_component_l861_86116


namespace NUMINAMATH_CALUDE_intersection_line_canonical_equations_l861_86167

/-- Given two planes in 3D space, this theorem proves that the canonical equations
    of the line formed by their intersection have a specific form. -/
theorem intersection_line_canonical_equations
  (plane1 : ℝ → ℝ → ℝ → Prop)
  (plane2 : ℝ → ℝ → ℝ → Prop)
  (h1 : ∀ x y z, plane1 x y z ↔ 4*x + y - 3*z + 2 = 0)
  (h2 : ∀ x y z, plane2 x y z ↔ 2*x - y + z - 8 = 0) :
  ∃ (line : ℝ → ℝ → ℝ → Prop),
    (∀ x y z, line x y z ↔ (x - 1) / (-2) = (y + 6) / (-10) ∧ (y + 6) / (-10) = z / (-6)) ∧
    (∀ x y z, line x y z ↔ plane1 x y z ∧ plane2 x y z) :=
by sorry

end NUMINAMATH_CALUDE_intersection_line_canonical_equations_l861_86167


namespace NUMINAMATH_CALUDE_sum_pairwise_reciprocal_sums_geq_three_halves_l861_86184

theorem sum_pairwise_reciprocal_sums_geq_three_halves 
  (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) : 
  a / (b + c) + b / (a + c) + c / (a + b) ≥ 3 / 2 := by
  sorry

end NUMINAMATH_CALUDE_sum_pairwise_reciprocal_sums_geq_three_halves_l861_86184


namespace NUMINAMATH_CALUDE_part_dimensions_l861_86193

/-- Given a base dimension with upper and lower tolerances, 
    prove the maximum and minimum allowable dimensions. -/
theorem part_dimensions 
  (base : ℝ) 
  (upper_tolerance : ℝ) 
  (lower_tolerance : ℝ) 
  (h_base : base = 7) 
  (h_upper : upper_tolerance = 0.05) 
  (h_lower : lower_tolerance = 0.02) : 
  (base + upper_tolerance = 7.05) ∧ (base - lower_tolerance = 6.98) := by
  sorry

end NUMINAMATH_CALUDE_part_dimensions_l861_86193


namespace NUMINAMATH_CALUDE_probability_at_least_five_consecutive_heads_l861_86172

def num_flips : ℕ := 8
def min_consecutive_heads : ℕ := 5

def favorable_outcomes : ℕ := 10
def total_outcomes : ℕ := 2^num_flips

theorem probability_at_least_five_consecutive_heads :
  (favorable_outcomes : ℚ) / total_outcomes = 5 / 128 := by sorry

end NUMINAMATH_CALUDE_probability_at_least_five_consecutive_heads_l861_86172


namespace NUMINAMATH_CALUDE_x_y_relation_l861_86181

theorem x_y_relation (Q : ℝ) (x y : ℝ) (hx : x = Real.sqrt (Q/2 + Real.sqrt (Q/2)))
  (hy : y = Real.sqrt (Q/2 - Real.sqrt (Q/2))) :
  (x^6 + y^6) / 40 = 10 := by
  sorry

end NUMINAMATH_CALUDE_x_y_relation_l861_86181


namespace NUMINAMATH_CALUDE_combined_salaries_l861_86112

/-- Given the salary of E and the average salary of A, B, C, D, and E,
    calculate the combined salaries of A, B, C, and D. -/
theorem combined_salaries 
  (salary_E : ℕ) 
  (average_salary : ℕ) 
  (h1 : salary_E = 9000)
  (h2 : average_salary = 8600) :
  (5 * average_salary) - salary_E = 34000 :=
by sorry

end NUMINAMATH_CALUDE_combined_salaries_l861_86112


namespace NUMINAMATH_CALUDE_brothers_age_sum_l861_86186

/-- Two brothers with an age difference of 4 years -/
structure Brothers where
  older_age : ℕ
  younger_age : ℕ
  age_difference : older_age = younger_age + 4

/-- The sum of the brothers' ages -/
def age_sum (b : Brothers) : ℕ := b.older_age + b.younger_age

/-- Theorem: The sum of the ages of two brothers who are 4 years apart,
    where the older one is 16 years old, is 28 years. -/
theorem brothers_age_sum :
  ∀ (b : Brothers), b.older_age = 16 → age_sum b = 28 := by
  sorry

end NUMINAMATH_CALUDE_brothers_age_sum_l861_86186


namespace NUMINAMATH_CALUDE_perpendicular_vector_equation_l861_86146

/-- Given two vectors a and b in ℝ², find the value of t such that a is perpendicular to (t * a + b) -/
theorem perpendicular_vector_equation (a b : ℝ × ℝ) (h : a = (1, 2) ∧ b = (4, 3)) :
  ∃ t : ℝ, a.1 * (t * a.1 + b.1) + a.2 * (t * a.2 + b.2) = 0 ∧ t = -2 := by
  sorry

#check perpendicular_vector_equation

end NUMINAMATH_CALUDE_perpendicular_vector_equation_l861_86146


namespace NUMINAMATH_CALUDE_square_root_expression_values_l861_86107

theorem square_root_expression_values :
  ∀ (x y z : ℝ),
  (x^2 = 25) →
  (y = 4) →
  (z^2 = 9) →
  (2*x + y - 5*z = -1) ∨ (2*x + y - 5*z = 29) :=
by
  sorry

end NUMINAMATH_CALUDE_square_root_expression_values_l861_86107


namespace NUMINAMATH_CALUDE_fgh_supermarkets_in_us_l861_86118

theorem fgh_supermarkets_in_us (total : ℕ) (difference : ℕ) 
  (h_total : total = 84)
  (h_difference : difference = 14) :
  let us := (total + difference) / 2
  us = 49 := by
sorry

end NUMINAMATH_CALUDE_fgh_supermarkets_in_us_l861_86118


namespace NUMINAMATH_CALUDE_rectangle_to_cylinder_surface_area_l861_86159

/-- The surface area of a cylinder formed by rolling a rectangle -/
def cylinderSurfaceArea (length width : Real) : Set Real :=
  let baseArea1 := Real.pi * (length / (2 * Real.pi))^2
  let baseArea2 := Real.pi * (width / (2 * Real.pi))^2
  let lateralArea := length * width
  {lateralArea + 2 * baseArea1, lateralArea + 2 * baseArea2}

theorem rectangle_to_cylinder_surface_area :
  cylinderSurfaceArea (4 * Real.pi) (8 * Real.pi) = {32 * Real.pi^2 + 8 * Real.pi, 32 * Real.pi^2 + 32 * Real.pi} := by
  sorry

#check rectangle_to_cylinder_surface_area

end NUMINAMATH_CALUDE_rectangle_to_cylinder_surface_area_l861_86159


namespace NUMINAMATH_CALUDE_jane_yellow_sheets_l861_86103

/-- The number of old, yellow sheets of drawing paper Jane has -/
def yellowSheets (totalSheets brownSheets : ℕ) : ℕ :=
  totalSheets - brownSheets

theorem jane_yellow_sheets : 
  let totalSheets : ℕ := 55
  let brownSheets : ℕ := 28
  yellowSheets totalSheets brownSheets = 27 := by
  sorry

end NUMINAMATH_CALUDE_jane_yellow_sheets_l861_86103


namespace NUMINAMATH_CALUDE_quadratic_inequality_solution_sets_l861_86111

-- Define the solution set of ax^2 - 5x + b > 0
def solution_set_1 : Set ℝ := {x | -3 < x ∧ x < -2}

-- Define the quadratic expression ax^2 - 5x + b
def quadratic_1 (a b x : ℝ) : ℝ := a * x^2 - 5 * x + b

-- Define the quadratic expression bx^2 - 5x + a
def quadratic_2 (a b x : ℝ) : ℝ := b * x^2 - 5 * x + a

-- Define the solution set of bx^2 - 5x + a < 0
def solution_set_2 : Set ℝ := {x | x < -1/2 ∨ x > -1/3}

theorem quadratic_inequality_solution_sets 
  (a b : ℝ) :
  (∀ x, x ∈ solution_set_1 ↔ quadratic_1 a b x > 0) →
  (∀ x, x ∈ solution_set_2 ↔ quadratic_2 a b x < 0) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_inequality_solution_sets_l861_86111


namespace NUMINAMATH_CALUDE_price_reduction_correct_l861_86183

/-- The final price of a medication after two price reductions -/
def final_price (m : ℝ) (x : ℝ) : ℝ := m * (1 - x)^2

/-- Theorem stating that the final price after two reductions is correct -/
theorem price_reduction_correct (m : ℝ) (x : ℝ) (y : ℝ) 
  (hm : m > 0) (hx : 0 ≤ x ∧ x < 1) :
  y = final_price m x ↔ y = m * (1 - x)^2 := by sorry

end NUMINAMATH_CALUDE_price_reduction_correct_l861_86183


namespace NUMINAMATH_CALUDE_fixed_point_on_circle_l861_86169

-- Define the ellipse C
def ellipse (x y : ℝ) : Prop := x^2/4 + y^2 = 1

-- Define the line L passing through M(0, -1/3)
def line (x y k : ℝ) : Prop := y = k*x - 1/3

-- Define a point on the ellipse
def point_on_ellipse (x y : ℝ) : Prop := ellipse x y

-- Define the intersection points A and B
def intersection_points (x1 y1 x2 y2 k : ℝ) : Prop :=
  point_on_ellipse x1 y1 ∧ point_on_ellipse x2 y2 ∧
  line x1 y1 k ∧ line x2 y2 k

-- Define the circle with diameter AB
def circle_AB (x y x1 y1 x2 y2 : ℝ) : Prop :=
  (x - (x1 + x2)/2)^2 + (y - (y1 + y2)/2)^2 = ((x1 - x2)^2 + (y1 - y2)^2)/4

-- Theorem statement
theorem fixed_point_on_circle (k : ℝ) :
  ∀ x1 y1 x2 y2,
  intersection_points x1 y1 x2 y2 k →
  circle_AB 0 1 x1 y1 x2 y2 :=
sorry

end NUMINAMATH_CALUDE_fixed_point_on_circle_l861_86169


namespace NUMINAMATH_CALUDE_fraction_equality_implies_numerator_equality_l861_86182

theorem fraction_equality_implies_numerator_equality 
  (a b c : ℝ) (h1 : c ≠ 0) (h2 : a / c = b / c) : a = b := by
  sorry

end NUMINAMATH_CALUDE_fraction_equality_implies_numerator_equality_l861_86182


namespace NUMINAMATH_CALUDE_a_range_theorem_l861_86178

/-- Proposition p: (a-2)x^2 + 2(a-2)x - 4 < 0 for all x ∈ ℝ -/
def prop_p (a : ℝ) : Prop :=
  ∀ x : ℝ, (a - 2) * x^2 + 2 * (a - 2) * x - 4 < 0

/-- Proposition q: One root of x^2 + (a-1)x + 1 = 0 is in (0,1), and the other is in (1,2) -/
def prop_q (a : ℝ) : Prop :=
  ∃ x y : ℝ, x^2 + (a - 1) * x + 1 = 0 ∧ y^2 + (a - 1) * y + 1 = 0 ∧
    0 < x ∧ x < 1 ∧ 1 < y ∧ y < 2

/-- The range of values for a -/
def a_range (a : ℝ) : Prop :=
  (a > -2 ∧ a ≤ -3/2) ∨ (a ≥ -1 ∧ a ≤ 2)

theorem a_range_theorem (a : ℝ) :
  (prop_p a ∨ prop_q a) ∧ ¬(prop_p a ∧ prop_q a) → a_range a := by
  sorry

end NUMINAMATH_CALUDE_a_range_theorem_l861_86178


namespace NUMINAMATH_CALUDE_profit_percentage_l861_86113

/-- Given that the cost price of 58 articles equals the selling price of 50 articles, 
    the percent profit is 16%. -/
theorem profit_percentage (C S : ℝ) (h : 58 * C = 50 * S) : 
  (S - C) / C * 100 = 16 := by
  sorry

end NUMINAMATH_CALUDE_profit_percentage_l861_86113


namespace NUMINAMATH_CALUDE_one_third_vector_AB_l861_86158

/-- Given two vectors OA and OB in 2D space, prove that 1/3 of vector AB equals (-3, -2) -/
theorem one_third_vector_AB (OA OB : ℝ × ℝ) : 
  OA = (2, 8) → OB = (-7, 2) → (1 / 3 : ℝ) • (OB - OA) = (-3, -2) := by sorry

end NUMINAMATH_CALUDE_one_third_vector_AB_l861_86158


namespace NUMINAMATH_CALUDE_fraction_relationship_l861_86141

theorem fraction_relationship (a b c : ℝ) 
  (h1 : a / b = 3 / 5) 
  (h2 : b / c = 2 / 7) : 
  c / a = 35 / 6 := by
  sorry

end NUMINAMATH_CALUDE_fraction_relationship_l861_86141


namespace NUMINAMATH_CALUDE_sin_2alpha_value_l861_86134

theorem sin_2alpha_value (α : ℝ) (h : Real.cos (α + π/4) = 3*Real.sqrt 2/5) : 
  Real.sin (2*α) = -11/25 := by
sorry

end NUMINAMATH_CALUDE_sin_2alpha_value_l861_86134


namespace NUMINAMATH_CALUDE_number_relationship_l861_86102

-- Define the numbers in their respective bases
def a : ℕ := 33
def b : ℕ := 5 * 6 + 2  -- 52 in base 6
def c : ℕ := 16 + 8 + 4 + 2 + 1  -- 11111 in base 2

-- Theorem statement
theorem number_relationship : a > b ∧ b > c := by
  sorry

end NUMINAMATH_CALUDE_number_relationship_l861_86102


namespace NUMINAMATH_CALUDE_number_of_girls_l861_86173

/-- Given a group of children with specific characteristics, prove the number of girls. -/
theorem number_of_girls (total_children happy_children sad_children neutral_children boys happy_boys sad_girls neutral_boys : ℕ) 
  (h1 : total_children = 60)
  (h2 : happy_children = 30)
  (h3 : sad_children = 10)
  (h4 : neutral_children = 20)
  (h5 : boys = 19)
  (h6 : happy_boys = 6)
  (h7 : sad_girls = 4)
  (h8 : neutral_boys = 7)
  (h9 : happy_children + sad_children + neutral_children = total_children) :
  total_children - boys = 41 := by
  sorry

end NUMINAMATH_CALUDE_number_of_girls_l861_86173


namespace NUMINAMATH_CALUDE_rogers_expenses_l861_86144

theorem rogers_expenses (A : ℝ) (m s p : ℝ) : 
  (m = 0.25 * (A - s - p)) →
  (s = 0.1 * (A - m - p)) →
  (p = 0.05 * (A - m - s)) →
  (A > 0) →
  (m > 0) →
  (s > 0) →
  (p > 0) →
  (abs ((m + s + p) / A - 0.32) < 0.005) := by
sorry

end NUMINAMATH_CALUDE_rogers_expenses_l861_86144


namespace NUMINAMATH_CALUDE_overtime_pay_fraction_l861_86142

/-- Represents the overtime pay calculation problem --/
theorem overtime_pay_fraction (regular_wage : ℝ) (hours_per_day : ℝ) (days : ℕ) 
  (total_pay : ℝ) (regular_hours : ℝ) (overtime_fraction : ℝ) : 
  regular_wage = 18 →
  hours_per_day = 10 →
  days = 5 →
  total_pay = 990 →
  regular_hours = 8 →
  total_pay = (regular_wage * regular_hours * days) + 
    (regular_wage * (1 + overtime_fraction) * (hours_per_day - regular_hours) * days) →
  overtime_fraction = 1/2 := by
  sorry


end NUMINAMATH_CALUDE_overtime_pay_fraction_l861_86142


namespace NUMINAMATH_CALUDE_rectangle_rotation_volume_l861_86148

/-- The volume of a solid formed by rotating a rectangle around one of its sides -/
theorem rectangle_rotation_volume (length width : ℝ) (h_length : length = 6) (h_width : width = 4) :
  ∃ (volume : ℝ), (volume = 96 * Real.pi ∨ volume = 144 * Real.pi) ∧
  (∃ (axis : ℝ), (axis = length ∨ axis = width) ∧
    volume = Real.pi * (axis / 2) ^ 2 * (if axis = length then width else length)) := by
  sorry

end NUMINAMATH_CALUDE_rectangle_rotation_volume_l861_86148


namespace NUMINAMATH_CALUDE_parabola_points_l861_86145

/-- A point on a parabola with equation y² = 4x that is 3 units away from its focus -/
structure ParabolaPoint where
  x : ℝ
  y : ℝ
  on_parabola : y^2 = 4*x
  distance_from_focus : (x - 1)^2 + y^2 = 3^2

/-- The theorem stating that (2, 2√2) and (2, -2√2) are the points on the parabola y² = 4x
    that are 3 units away from its focus -/
theorem parabola_points : 
  (∃ (p : ParabolaPoint), p.x = 2 ∧ p.y = 2 * Real.sqrt 2) ∧
  (∃ (p : ParabolaPoint), p.x = 2 ∧ p.y = -2 * Real.sqrt 2) :=
by sorry

end NUMINAMATH_CALUDE_parabola_points_l861_86145


namespace NUMINAMATH_CALUDE_louisa_travel_l861_86138

/-- Louisa's vacation travel problem -/
theorem louisa_travel (average_speed : ℝ) (second_day_distance : ℝ) (time_difference : ℝ) :
  average_speed = 33.333333333333336 →
  second_day_distance = 350 →
  time_difference = 3 →
  ∃ (first_day_distance : ℝ),
    first_day_distance = average_speed * (second_day_distance / average_speed - time_difference) ∧
    first_day_distance = 250 :=
by sorry

end NUMINAMATH_CALUDE_louisa_travel_l861_86138


namespace NUMINAMATH_CALUDE_cost_of_lettuce_cost_of_lettuce_is_one_dollar_l861_86151

/-- The cost of the head of lettuce in Lauren's grocery purchase --/
theorem cost_of_lettuce : ℝ := by
  -- Define the known costs
  let meat_cost : ℝ := 2 * 3.5
  let buns_cost : ℝ := 1.5
  let tomato_cost : ℝ := 1.5 * 2
  let pickles_cost : ℝ := 2.5 - 1

  -- Define the total bill and change
  let total_paid : ℝ := 20
  let change : ℝ := 6

  -- Define the actual spent amount
  let actual_spent : ℝ := total_paid - change

  -- Define the sum of known costs
  let known_costs : ℝ := meat_cost + buns_cost + tomato_cost + pickles_cost

  -- The cost of lettuce is the difference between actual spent and known costs
  have lettuce_cost : ℝ := actual_spent - known_costs

  -- Prove that the cost of lettuce is 1.00
  sorry

/-- The cost of the head of lettuce is $1.00 --/
theorem cost_of_lettuce_is_one_dollar : cost_of_lettuce = 1 := by sorry

end NUMINAMATH_CALUDE_cost_of_lettuce_cost_of_lettuce_is_one_dollar_l861_86151


namespace NUMINAMATH_CALUDE_sector_angle_l861_86199

/-- Given a circular sector with area 1 and radius 1, prove that its central angle in radians is 2 -/
theorem sector_angle (area : ℝ) (radius : ℝ) (angle : ℝ) 
  (h_area : area = 1) 
  (h_radius : radius = 1) 
  (h_sector : area = 1/2 * radius^2 * angle) : angle = 2 := by
  sorry

end NUMINAMATH_CALUDE_sector_angle_l861_86199


namespace NUMINAMATH_CALUDE_angle_PQR_is_60_degrees_l861_86197

-- Define the points
def P : ℝ × ℝ × ℝ := (-3, 1, 7)
def Q : ℝ × ℝ × ℝ := (-4, 0, 3)
def R : ℝ × ℝ × ℝ := (-5, 0, 4)

-- Define the angle PQR in radians
def angle_PQR : ℝ := sorry

theorem angle_PQR_is_60_degrees :
  angle_PQR = π / 3 := by sorry

end NUMINAMATH_CALUDE_angle_PQR_is_60_degrees_l861_86197


namespace NUMINAMATH_CALUDE_geometric_sequence_minimum_l861_86136

theorem geometric_sequence_minimum (a : ℕ → ℝ) (q : ℝ) (h_q : q ≠ 1) :
  (∃ s t : ℕ, s ≠ 0 ∧ t ≠ 0 ∧ a s * a t = (a 5)^2) →
  (∃ s t : ℕ, s ≠ 0 ∧ t ≠ 0 ∧ a s * a t = (a 5)^2 ∧
    ∀ u v : ℕ, u ≠ 0 → v ≠ 0 → a u * a v = (a 5)^2 →
      4/s + 1/(4*t) ≤ 4/u + 1/(4*v)) →
  (∃ s t : ℕ, s ≠ 0 ∧ t ≠ 0 ∧ a s * a t = (a 5)^2 ∧ 4/s + 1/(4*t) = 5/8) :=
by sorry

end NUMINAMATH_CALUDE_geometric_sequence_minimum_l861_86136


namespace NUMINAMATH_CALUDE_remainder_of_division_l861_86127

theorem remainder_of_division (n : ℕ) : 
  (3^302 + 302) % (3^151 + 3^101 + 1) = 302 := by
  sorry

#check remainder_of_division

end NUMINAMATH_CALUDE_remainder_of_division_l861_86127


namespace NUMINAMATH_CALUDE_tangent_slope_implies_a_l861_86161

-- Define the curve
def curve (a : ℝ) (x : ℝ) : ℝ := x^4 + a*x^2 + 1

-- Define the derivative of the curve
def curve_derivative (a : ℝ) (x : ℝ) : ℝ := 4*x^3 + 2*a*x

theorem tangent_slope_implies_a (a : ℝ) :
  curve a (-1) = a + 2 →
  curve_derivative a (-1) = 8 →
  a = -6 := by sorry

end NUMINAMATH_CALUDE_tangent_slope_implies_a_l861_86161


namespace NUMINAMATH_CALUDE_quadrilateral_properties_l861_86121

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a quadrilateral defined by four points -/
structure Quadrilateral where
  O : Point
  P : Point
  Q : Point
  R : Point

/-- Checks if a quadrilateral is a parallelogram -/
def isParallelogram (quad : Quadrilateral) : Prop :=
  (quad.R.x - quad.P.x = quad.Q.x - quad.O.x) ∧
  (quad.R.y - quad.P.y = quad.Q.y - quad.O.y)

/-- Checks if a quadrilateral is a rhombus -/
def isRhombus (quad : Quadrilateral) : Prop :=
  let OP := (quad.P.x - quad.O.x)^2 + (quad.P.y - quad.O.y)^2
  let OQ := (quad.Q.x - quad.O.x)^2 + (quad.Q.y - quad.O.y)^2
  let OR := (quad.R.x - quad.O.x)^2 + (quad.R.y - quad.O.y)^2
  let PQ := (quad.Q.x - quad.P.x)^2 + (quad.Q.y - quad.P.y)^2
  OP = OQ ∧ OQ = OR ∧ OR = PQ

theorem quadrilateral_properties (x₁ y₁ x₂ y₂ : ℝ) :
  let quad := Quadrilateral.mk
    (Point.mk 0 0)
    (Point.mk x₁ y₁)
    (Point.mk x₂ y₂)
    (Point.mk (2*x₁ - x₂) (2*y₁ - y₂))
  isParallelogram quad ∧ (∃ (x₁ y₁ x₂ y₂ : ℝ), isRhombus quad) := by
  sorry

end NUMINAMATH_CALUDE_quadrilateral_properties_l861_86121


namespace NUMINAMATH_CALUDE_sum_of_arithmetic_sequences_l861_86100

def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) - a n = d

theorem sum_of_arithmetic_sequences
  (a b : ℕ → ℝ)
  (ha : arithmetic_sequence a)
  (hb : arithmetic_sequence b)
  (h1 : a 1 = 25)
  (h2 : b 1 = 75)
  (h3 : a 2 + b 2 = 100) :
  a 37 + b 37 = 100 := by
sorry

end NUMINAMATH_CALUDE_sum_of_arithmetic_sequences_l861_86100


namespace NUMINAMATH_CALUDE_regular_polygon_diagonals_l861_86137

theorem regular_polygon_diagonals (n : ℕ) : n > 2 →
  (n * (n - 3)) / 2 = 20 → n = 8 := by sorry

end NUMINAMATH_CALUDE_regular_polygon_diagonals_l861_86137


namespace NUMINAMATH_CALUDE_banana_bread_theorem_l861_86104

/-- The number of bananas needed to make one loaf of banana bread -/
def bananas_per_loaf : ℕ := 4

/-- The number of loaves made on Monday -/
def monday_loaves : ℕ := 3

/-- The number of loaves made on Tuesday -/
def tuesday_loaves : ℕ := 2 * monday_loaves

/-- The total number of bananas used over two days -/
def total_bananas : ℕ := bananas_per_loaf * (monday_loaves + tuesday_loaves)

theorem banana_bread_theorem : total_bananas = 36 := by
  sorry

end NUMINAMATH_CALUDE_banana_bread_theorem_l861_86104


namespace NUMINAMATH_CALUDE_intersection_A_complement_B_l861_86129

def U : Set ℝ := Set.univ

def A : Set ℝ := {x | x^2 - 2*x < 0}

def B : Set ℝ := {x | x > 1}

theorem intersection_A_complement_B :
  A ∩ (U \ B) = {x : ℝ | 0 < x ∧ x ≤ 1} := by sorry

end NUMINAMATH_CALUDE_intersection_A_complement_B_l861_86129


namespace NUMINAMATH_CALUDE_complex_power_225_deg_18_l861_86106

theorem complex_power_225_deg_18 : 
  (Complex.exp (Complex.I * Real.pi * (5 / 4)))^18 = Complex.I := by sorry

end NUMINAMATH_CALUDE_complex_power_225_deg_18_l861_86106


namespace NUMINAMATH_CALUDE_bus_wheel_radius_proof_l861_86196

/-- The speed of the bus in km/h -/
def bus_speed : ℝ := 66

/-- The revolutions per minute of the wheel -/
def wheel_rpm : ℝ := 125.11373976342128

/-- The radius of the wheel in centimeters -/
def wheel_radius : ℝ := 140.007

/-- Theorem stating that given the bus speed and wheel rpm, the wheel radius is approximately 140.007 cm -/
theorem bus_wheel_radius_proof :
  ∃ (ε : ℝ), ε > 0 ∧ ε < 0.001 ∧ 
  |wheel_radius - (bus_speed * 100000 / (60 * wheel_rpm * 2 * Real.pi))| < ε :=
sorry

end NUMINAMATH_CALUDE_bus_wheel_radius_proof_l861_86196


namespace NUMINAMATH_CALUDE_length_AC_l861_86155

-- Define the right triangle ABC
def triangle_ABC (A B C : ℝ × ℝ) : Prop :=
  -- Condition 1 and 2: ABC is a right triangle with angle C = 90°
  (C.1 - A.1) * (B.1 - A.1) + (C.2 - A.2) * (B.2 - A.2) = 0 ∧
  -- Condition 3: AB = 9
  Real.sqrt ((B.1 - A.1)^2 + (B.2 - A.2)^2) = 9 ∧
  -- Condition 4: cos B = 2/3
  (C.1 - B.1) / Real.sqrt ((C.1 - B.1)^2 + (C.2 - B.2)^2) = 2/3

-- Theorem statement
theorem length_AC (A B C : ℝ × ℝ) (h : triangle_ABC A B C) :
  Real.sqrt ((C.1 - A.1)^2 + (C.2 - A.2)^2) = 3 * Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_length_AC_l861_86155


namespace NUMINAMATH_CALUDE_largest_n_for_product_l861_86154

/-- Arithmetic sequence (a_n) with initial value 1 and common difference x -/
def a (n : ℕ) (x : ℤ) : ℤ := 1 + (n - 1 : ℤ) * x

/-- Arithmetic sequence (b_n) with initial value 1 and common difference y -/
def b (n : ℕ) (y : ℤ) : ℤ := 1 + (n - 1 : ℤ) * y

theorem largest_n_for_product (x y : ℤ) (hx : x > 0) (hy : y > 0) 
  (h_a2_b2 : 1 < a 2 x ∧ a 2 x ≤ b 2 y) :
  (∃ n : ℕ, a n x * b n y = 1764) →
  (∀ m : ℕ, a m x * b m y = 1764 → m ≤ 44) ∧
  (∃ n : ℕ, a n x * b n y = 1764 ∧ n = 44) :=
by sorry

end NUMINAMATH_CALUDE_largest_n_for_product_l861_86154


namespace NUMINAMATH_CALUDE_matt_work_time_l861_86128

theorem matt_work_time (total_time together_time matt_remaining_time : ℝ) 
  (h1 : total_time = 20)
  (h2 : together_time = 12)
  (h3 : matt_remaining_time = 10) : 
  (total_time * matt_remaining_time) / (total_time - together_time) = 25 := by
  sorry

#check matt_work_time

end NUMINAMATH_CALUDE_matt_work_time_l861_86128


namespace NUMINAMATH_CALUDE_museum_ticket_price_l861_86164

theorem museum_ticket_price 
  (group_size : ℕ) 
  (total_paid : ℚ) 
  (tax_rate : ℚ) 
  (h1 : group_size = 25) 
  (h2 : total_paid = 945) 
  (h3 : tax_rate = 5 / 100) : 
  ∃ (face_value : ℚ), 
    face_value = 36 ∧ 
    total_paid = group_size * face_value * (1 + tax_rate) := by
  sorry

end NUMINAMATH_CALUDE_museum_ticket_price_l861_86164


namespace NUMINAMATH_CALUDE_aubreys_garden_aubreys_garden_proof_l861_86135

/-- Aubrey's Garden Planting Problem -/
theorem aubreys_garden (tomato_cucumber_ratio : Nat) (plants_per_row : Nat) (tomatoes_per_plant : Nat) (total_tomatoes : Nat) : Nat :=
  let tomato_rows := total_tomatoes / (plants_per_row * tomatoes_per_plant)
  let cucumber_rows := tomato_rows * tomato_cucumber_ratio
  tomato_rows + cucumber_rows

/-- Proof of Aubrey's Garden Planting Problem -/
theorem aubreys_garden_proof :
  aubreys_garden 2 8 3 120 = 15 := by
  sorry

end NUMINAMATH_CALUDE_aubreys_garden_aubreys_garden_proof_l861_86135


namespace NUMINAMATH_CALUDE_sphere_configuration_exists_l861_86117

-- Define a sphere in 3D space
structure Sphere where
  center : ℝ × ℝ × ℝ
  radius : ℝ

-- Define a plane in 3D space
structure Plane where
  normal : ℝ × ℝ × ℝ
  point : ℝ × ℝ × ℝ

-- Function to check if a plane is tangent to a sphere
def is_tangent_plane (p : Plane) (s : Sphere) : Prop :=
  -- Implementation details omitted
  sorry

-- Function to check if a plane touches a sphere
def touches_sphere (p : Plane) (s : Sphere) : Prop :=
  -- Implementation details omitted
  sorry

-- Main theorem
theorem sphere_configuration_exists : ∃ (spheres : Fin 5 → Sphere),
  ∀ i : Fin 5, ∃ (p : Plane),
    is_tangent_plane p (spheres i) ∧
    (∀ j : Fin 5, j ≠ i → touches_sphere p (spheres j)) :=
  sorry

end NUMINAMATH_CALUDE_sphere_configuration_exists_l861_86117


namespace NUMINAMATH_CALUDE_min_bottles_for_27_people_min_bottles_sufficient_l861_86188

/-- The minimum number of bottles needed to be purchased for a given number of people,
    given that 3 empty bottles can be exchanged for 1 full bottle -/
def min_bottles_to_purchase (num_people : ℕ) : ℕ :=
  (2 * num_people + 2) / 3

/-- Proof that for 27 people, the minimum number of bottles to purchase is 18 -/
theorem min_bottles_for_27_people :
  min_bottles_to_purchase 27 = 18 := by
  sorry

/-- Proof that the calculated minimum number of bottles is sufficient for all people -/
theorem min_bottles_sufficient (num_people : ℕ) :
  min_bottles_to_purchase num_people + (min_bottles_to_purchase num_people) / 2 ≥ num_people := by
  sorry

end NUMINAMATH_CALUDE_min_bottles_for_27_people_min_bottles_sufficient_l861_86188


namespace NUMINAMATH_CALUDE_divisibility_by_seven_l861_86114

theorem divisibility_by_seven (n : ℕ) : 
  7 ∣ n ↔ 7 ∣ ((n / 10) - 2 * (n % 10)) := by
  sorry

end NUMINAMATH_CALUDE_divisibility_by_seven_l861_86114


namespace NUMINAMATH_CALUDE_game_ends_in_41_rounds_l861_86131

/-- Represents a player in the token game -/
structure Player where
  name : String
  tokens : ℕ

/-- Represents the state of the game -/
structure GameState where
  players : List Player
  rounds : ℕ

/-- Simulates one round of the game -/
def playRound (state : GameState) : GameState :=
  sorry

/-- Checks if the game has ended (i.e., a player has run out of tokens) -/
def isGameOver (state : GameState) : Bool :=
  sorry

/-- Simulates the entire game until it ends -/
def playGame (initialState : GameState) : GameState :=
  sorry

/-- Theorem stating that the game ends after exactly 41 rounds -/
theorem game_ends_in_41_rounds :
  let initialState : GameState :=
    { players := [
        { name := "D", tokens := 16 },
        { name := "E", tokens := 15 },
        { name := "F", tokens := 13 }
      ],
      rounds := 0
    }
  let finalState := playGame initialState
  finalState.rounds = 41 ∧ isGameOver finalState := by
  sorry

end NUMINAMATH_CALUDE_game_ends_in_41_rounds_l861_86131


namespace NUMINAMATH_CALUDE_mario_garden_flowers_l861_86160

/-- The number of flowers on Mario's first hibiscus plant -/
def first_plant : ℕ := 2

/-- The number of flowers on Mario's second hibiscus plant -/
def second_plant : ℕ := 2 * first_plant

/-- The number of flowers on Mario's third hibiscus plant -/
def third_plant : ℕ := 4 * second_plant

/-- The total number of flowers in Mario's garden -/
def total_flowers : ℕ := first_plant + second_plant + third_plant

theorem mario_garden_flowers : total_flowers = 22 := by
  sorry

end NUMINAMATH_CALUDE_mario_garden_flowers_l861_86160


namespace NUMINAMATH_CALUDE_range_of_even_power_function_l861_86176

theorem range_of_even_power_function (k : ℕ) (hk : Even k) (hk_pos : k > 0) :
  Set.range (fun x : ℝ => x ^ k) = Set.Ici (0 : ℝ) := by
  sorry

end NUMINAMATH_CALUDE_range_of_even_power_function_l861_86176


namespace NUMINAMATH_CALUDE_small_bottles_initial_count_small_bottles_initial_count_proof_l861_86162

theorem small_bottles_initial_count : ℝ → Prop :=
  fun S =>
    let big_bottles : ℝ := 12000
    let small_bottles_remaining_ratio : ℝ := 0.85
    let big_bottles_remaining_ratio : ℝ := 0.82
    let total_remaining : ℝ := 14090
    S * small_bottles_remaining_ratio + big_bottles * big_bottles_remaining_ratio = total_remaining →
    S = 5000

-- The proof goes here
theorem small_bottles_initial_count_proof : small_bottles_initial_count 5000 := by
  sorry

end NUMINAMATH_CALUDE_small_bottles_initial_count_small_bottles_initial_count_proof_l861_86162


namespace NUMINAMATH_CALUDE_violet_family_ticket_cost_l861_86119

/-- The cost of separate tickets for Violet's family -/
def separate_ticket_cost (adult_price children_price : ℕ) (num_adults num_children : ℕ) : ℕ :=
  adult_price * num_adults + children_price * num_children

/-- Theorem: The total cost of separate tickets for Violet's family is $155 -/
theorem violet_family_ticket_cost :
  separate_ticket_cost 35 20 1 6 = 155 :=
by sorry

end NUMINAMATH_CALUDE_violet_family_ticket_cost_l861_86119


namespace NUMINAMATH_CALUDE_arithmetic_sequence_property_l861_86153

-- Define an arithmetic sequence
def is_arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) - a n = d

-- Define the theorem
theorem arithmetic_sequence_property
  (a : ℕ → ℝ) (l m n : ℕ) (a' b c : ℝ) 
  (h_arithmetic : is_arithmetic_sequence a)
  (h_l : a l = 1 / a')
  (h_m : a m = 1 / b)
  (h_n : a n = 1 / c) :
  (l - m : ℝ) * a' * b + (m - n : ℝ) * b * c + (n - l : ℝ) * c * a' = 0 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_property_l861_86153


namespace NUMINAMATH_CALUDE_multiply_by_48_equals_173_times_240_l861_86185

theorem multiply_by_48_equals_173_times_240 : 48 * 865 = 173 * 240 := by
  sorry

end NUMINAMATH_CALUDE_multiply_by_48_equals_173_times_240_l861_86185


namespace NUMINAMATH_CALUDE_number_of_divisors_2310_l861_86170

/-- The number of positive divisors of 2310 is 32. -/
theorem number_of_divisors_2310 : Nat.card (Nat.divisors 2310) = 32 := by
  sorry

end NUMINAMATH_CALUDE_number_of_divisors_2310_l861_86170


namespace NUMINAMATH_CALUDE_inequality_proof_l861_86166

theorem inequality_proof (a b c : ℝ) (h1 : a > b) (h2 : b > c) (h3 : c > 0) :
  (b / (a - b) > c / (a - c)) ∧
  (a / (a + b) < (a + c) / (a + b + c)) ∧
  (1 / (a - b) + 1 / (b - c) ≥ 4 / (a - c)) := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l861_86166


namespace NUMINAMATH_CALUDE_right_triangle_arm_square_l861_86152

/-- In a right triangle with hypotenuse c and arms a and b, where c = a + 2,
    the square of b is equal to 4a + 4. -/
theorem right_triangle_arm_square (a c : ℝ) (h1 : c = a + 2) :
  ∃ b : ℝ, b^2 = 4*a + 4 ∧ a^2 + b^2 = c^2 := by sorry

end NUMINAMATH_CALUDE_right_triangle_arm_square_l861_86152


namespace NUMINAMATH_CALUDE_tower_configurations_mod_1000_l861_86163

/-- Recursively calculates the number of valid tower configurations for m cubes -/
def tower_configurations (m : ℕ) : ℕ :=
  match m with
  | 0 => 0
  | 1 => 1
  | 2 => 2
  | 3 => 6
  | n + 1 => (n + 2) * tower_configurations n

/-- Represents the conditions for building towers with 9 cubes -/
def valid_tower_conditions (n : ℕ) : Prop :=
  n ≤ 9 ∧ 
  ∀ k : ℕ, k ≤ n → ∃ cube : ℕ, cube = k ∧
  ∀ i j : ℕ, i < j → j - i ≤ 3

/-- The main theorem stating that the number of different towers is congruent to 200 modulo 1000 -/
theorem tower_configurations_mod_1000 :
  valid_tower_conditions 9 →
  tower_configurations 9 % 1000 = 200 := by
  sorry


end NUMINAMATH_CALUDE_tower_configurations_mod_1000_l861_86163


namespace NUMINAMATH_CALUDE_coin_value_problem_l861_86147

theorem coin_value_problem :
  ∃ (n d q : ℕ),
    n + d + q = 30 ∧
    5 * n + 10 * d + 25 * q = 315 ∧
    10 * n + 25 * d + 5 * q = 5 * n + 10 * d + 25 * q + 120 :=
by sorry

end NUMINAMATH_CALUDE_coin_value_problem_l861_86147


namespace NUMINAMATH_CALUDE_inequality_equivalence_l861_86156

theorem inequality_equivalence (x : ℝ) : 
  (1/3 : ℝ)^(x^2 - 8) > 3^(-2*x) ↔ -2 < x ∧ x < 4 := by sorry

end NUMINAMATH_CALUDE_inequality_equivalence_l861_86156


namespace NUMINAMATH_CALUDE_missing_number_proof_l861_86194

theorem missing_number_proof (x : ℝ) : (4 + x) + (8 - 3 - 1) = 11 → x = 3 := by
  sorry

end NUMINAMATH_CALUDE_missing_number_proof_l861_86194


namespace NUMINAMATH_CALUDE_acute_triangle_existence_l861_86174

theorem acute_triangle_existence (d : Fin 12 → ℝ) 
  (h_range : ∀ i, 1 < d i ∧ d i < 12) :
  ∃ i j k, i ≠ j ∧ j ≠ k ∧ i ≠ k ∧ 
    (d i < d j + d k) ∧ 
    (d j < d i + d k) ∧ 
    (d k < d i + d j) ∧
    (d i)^2 + (d j)^2 > (d k)^2 :=
by sorry

end NUMINAMATH_CALUDE_acute_triangle_existence_l861_86174


namespace NUMINAMATH_CALUDE_book_width_average_l861_86108

theorem book_width_average : 
  let book_widths : List ℝ := [3, 3/4, 1.2, 4, 9, 0.5, 8]
  let total_width : ℝ := book_widths.sum
  let num_books : ℕ := book_widths.length
  let average_width : ℝ := total_width / num_books
  ∃ ε > 0, |average_width - 3.8| < ε := by
sorry

end NUMINAMATH_CALUDE_book_width_average_l861_86108


namespace NUMINAMATH_CALUDE_product_and_sum_of_integers_l861_86123

theorem product_and_sum_of_integers : ∃ (n m : ℕ), 
  m = n + 2 ∧ 
  n * m = 2720 ∧ 
  n > 0 ∧ 
  n + m = 104 := by
sorry

end NUMINAMATH_CALUDE_product_and_sum_of_integers_l861_86123


namespace NUMINAMATH_CALUDE_points_collinear_iff_b_eq_neg_one_over_44_l861_86130

/-- Three points are collinear if and only if the slopes between any two pairs of points are equal. -/
def collinear (x₁ y₁ x₂ y₂ x₃ y₃ : ℝ) : Prop :=
  (y₂ - y₁) * (x₃ - x₁) = (y₃ - y₁) * (x₂ - x₁)

/-- The theorem states that the given points are collinear if and only if b = -1/44. -/
theorem points_collinear_iff_b_eq_neg_one_over_44 :
  ∀ b : ℝ, collinear 4 (-6) (2*b + 1) 4 (-3*b + 2) 1 ↔ b = -1/44 :=
by sorry

end NUMINAMATH_CALUDE_points_collinear_iff_b_eq_neg_one_over_44_l861_86130


namespace NUMINAMATH_CALUDE_circle_and_line_properties_l861_86126

-- Define the circles and line
def circle_M (x y : ℝ) := 2*x^2 + 2*y^2 - 8*x - 8*y - 1 = 0
def circle_N (x y : ℝ) := x^2 + y^2 + 2*x + 2*y - 6 = 0
def line_l (x y : ℝ) := x + y - 9 = 0

-- Define the angle condition
def angle_BAC : ℝ := 45

-- Theorem statement
theorem circle_and_line_properties :
  ∃ (x y : ℝ),
    -- 1. Equation of circle through intersection of M and N, and origin
    (x^2 + y^2 - (50/11)*x - (50/11)*y = 0) ∧
    -- 2a. Equations of line AC when x-coordinate of A is 4
    ((5*x + y - 25 = 0) ∨ (x - 5*y + 21 = 0)) ∧
    -- 2b. Range of possible x-coordinates for point A
    (∀ (m : ℝ), (m ∈ Set.Icc 3 6) ↔ 
      (∃ (y : ℝ), line_l m y ∧ 
        ∃ (B C : ℝ × ℝ), 
          circle_M B.1 B.2 ∧ 
          circle_M C.1 C.2 ∧
          (angle_BAC = 45))) :=
sorry

end NUMINAMATH_CALUDE_circle_and_line_properties_l861_86126


namespace NUMINAMATH_CALUDE_max_piles_l861_86177

/-- Represents a configuration of stone piles -/
structure StonePiles :=
  (piles : List Nat)
  (total_stones : Nat)
  (h_total : piles.sum = total_stones)
  (h_factor : ∀ (p q : Nat), p ∈ piles → q ∈ piles → p < 2 * q)

/-- Defines a valid split operation on stone piles -/
def split (sp : StonePiles) (i : Nat) (n : Nat) : Option StonePiles :=
  sorry

/-- Theorem: The maximum number of piles that can be formed is 30 -/
theorem max_piles (sp : StonePiles) (h_initial : sp.total_stones = 660) :
  (∀ sp' : StonePiles, ∃ (i j : Nat), split sp i j = some sp') →
  sp.piles.length ≤ 30 :=
sorry

end NUMINAMATH_CALUDE_max_piles_l861_86177


namespace NUMINAMATH_CALUDE_quadratic_real_solutions_range_l861_86191

theorem quadratic_real_solutions_range (m : ℝ) : 
  (∃ x : ℝ, (m - 2) * x^2 - 2 * x + 1 = 0) → m ≤ 3 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_real_solutions_range_l861_86191


namespace NUMINAMATH_CALUDE_evaluate_expression_l861_86132

theorem evaluate_expression : (24^36) / (72^18) = 8^18 := by
  sorry

end NUMINAMATH_CALUDE_evaluate_expression_l861_86132


namespace NUMINAMATH_CALUDE_point_on_line_l861_86124

/-- A complex number z represented as (a-1) + 3i, where a is a real number -/
def z (a : ℝ) : ℂ := Complex.mk (a - 1) 3

/-- The line y = x + 2 in the complex plane -/
def line (x : ℝ) : ℝ := x + 2

/-- Theorem: If z(a) is on the line y = x + 2, then a = 2 -/
theorem point_on_line (a : ℝ) : z a = Complex.mk (z a).re (line (z a).re) → a = 2 := by
  sorry

end NUMINAMATH_CALUDE_point_on_line_l861_86124


namespace NUMINAMATH_CALUDE_school_field_trip_cost_l861_86168

/-- Calculates the total cost for a school field trip to a farm -/
theorem school_field_trip_cost (num_students : ℕ) (num_adults : ℕ) 
  (student_fee : ℕ) (adult_fee : ℕ) : 
  num_students = 35 → num_adults = 4 → student_fee = 5 → adult_fee = 6 →
  num_students * student_fee + num_adults * adult_fee = 199 :=
by sorry

end NUMINAMATH_CALUDE_school_field_trip_cost_l861_86168


namespace NUMINAMATH_CALUDE_sand_pile_volume_l861_86190

/-- The volume of a cylindrical pile of sand -/
theorem sand_pile_volume :
  ∀ (r h d : ℝ),
  d = 8 →                -- diameter is 8 feet
  r = d / 2 →            -- radius is half the diameter
  h = 2 * r →            -- height is twice the radius
  π * r^2 * h = 128 * π  -- volume is 128π cubic feet
  := by sorry

end NUMINAMATH_CALUDE_sand_pile_volume_l861_86190


namespace NUMINAMATH_CALUDE_number_comparison_l861_86109

theorem number_comparison (A B : ℝ) (h : (3/4) * A = (2/3) * B) : A < B := by
  sorry

end NUMINAMATH_CALUDE_number_comparison_l861_86109


namespace NUMINAMATH_CALUDE_joshua_share_l861_86105

def total_amount : ℚ := 123.50
def joshua_multiplier : ℚ := 3.5
def jasmine_multiplier : ℚ := 0.75

theorem joshua_share :
  ∃ (justin_share : ℚ),
    justin_share + joshua_multiplier * justin_share + jasmine_multiplier * justin_share = total_amount ∧
    joshua_multiplier * justin_share = 82.32 :=
by sorry

end NUMINAMATH_CALUDE_joshua_share_l861_86105


namespace NUMINAMATH_CALUDE_largest_difference_l861_86179

def A : ℕ := 3 * 2005^2006
def B : ℕ := 2005^2006
def C : ℕ := 2004 * 2005^2005
def D : ℕ := 3 * 2005^2005
def E : ℕ := 2005^2005
def F : ℕ := 2005^2004

theorem largest_difference : A - B > max (B - C) (max (C - D) (max (D - E) (E - F))) := by
  sorry

end NUMINAMATH_CALUDE_largest_difference_l861_86179


namespace NUMINAMATH_CALUDE_shape_area_theorem_l861_86110

/-- Represents a shape in a grid --/
structure GridShape where
  wholeSquares : ℕ
  halfSquares : ℕ

/-- Calculates the area of a GridShape --/
def calculateArea (shape : GridShape) : ℚ :=
  shape.wholeSquares + shape.halfSquares / 2

theorem shape_area_theorem (shape : GridShape) :
  shape.wholeSquares = 5 → shape.halfSquares = 6 → calculateArea shape = 8 := by
  sorry

end NUMINAMATH_CALUDE_shape_area_theorem_l861_86110


namespace NUMINAMATH_CALUDE_white_sox_games_lost_l861_86195

theorem white_sox_games_lost (total_games won_games : ℕ) 
  (h1 : total_games = 162)
  (h2 : won_games = 99)
  (h3 : won_games = lost_games + 36) : lost_games = 63 :=
by
  sorry

end NUMINAMATH_CALUDE_white_sox_games_lost_l861_86195


namespace NUMINAMATH_CALUDE_initial_guppies_count_l861_86122

/-- Represents the fish tank scenario --/
structure FishTank where
  initialGuppies : ℕ
  initialAngelfish : ℕ
  initialTigerSharks : ℕ
  initialOscarFish : ℕ
  soldGuppies : ℕ
  soldAngelfish : ℕ
  soldTigerSharks : ℕ
  soldOscarFish : ℕ
  remainingFish : ℕ

/-- Theorem stating the initial number of guppies in Danny's fish tank --/
theorem initial_guppies_count (tank : FishTank)
    (h1 : tank.initialAngelfish = 76)
    (h2 : tank.initialTigerSharks = 89)
    (h3 : tank.initialOscarFish = 58)
    (h4 : tank.soldGuppies = 30)
    (h5 : tank.soldAngelfish = 48)
    (h6 : tank.soldTigerSharks = 17)
    (h7 : tank.soldOscarFish = 24)
    (h8 : tank.remainingFish = 198)
    (h9 : tank.remainingFish = 
      (tank.initialGuppies - tank.soldGuppies) +
      (tank.initialAngelfish - tank.soldAngelfish) +
      (tank.initialTigerSharks - tank.soldTigerSharks) +
      (tank.initialOscarFish - tank.soldOscarFish)) :
    tank.initialGuppies = 94 := by
  sorry

end NUMINAMATH_CALUDE_initial_guppies_count_l861_86122


namespace NUMINAMATH_CALUDE_ella_video_game_spending_l861_86150

/-- Proves that Ella spent $100 on video games last year given her current salary and spending habits -/
theorem ella_video_game_spending (new_salary : ℝ) (raise_percentage : ℝ) (video_game_percentage : ℝ) :
  new_salary = 275 →
  raise_percentage = 0.1 →
  video_game_percentage = 0.4 →
  (new_salary / (1 + raise_percentage)) * video_game_percentage = 100 := by
  sorry

end NUMINAMATH_CALUDE_ella_video_game_spending_l861_86150


namespace NUMINAMATH_CALUDE_shortest_distance_to_quadratic_curve_l861_86125

/-- The shortest distance from a point to a quadratic curve -/
theorem shortest_distance_to_quadratic_curve
  (m k a b : ℝ) :
  let curve := fun (x : ℝ) => m * x^2 + k
  let P := (a, b)
  let Q := fun (c : ℝ) => (c, curve c)
  ∃ (c : ℝ), ∀ (x : ℝ),
    dist P (Q c) ≤ dist P (Q x) ∧
    dist P (Q c) = |m * a^2 + k - b| :=
by sorry

end NUMINAMATH_CALUDE_shortest_distance_to_quadratic_curve_l861_86125


namespace NUMINAMATH_CALUDE_min_m_plus_n_l861_86165

/-- The sum of interior angles of a regular n-gon -/
def interior_angle_sum (n : ℕ) : ℕ := 180 * (n - 2)

/-- The sum of interior angles of m regular n-gons -/
def total_interior_angle_sum (m n : ℕ) : ℕ := m * interior_angle_sum n

/-- Predicate to check if the sum of interior angles is divisible by 27 -/
def is_divisible_by_27 (m n : ℕ) : Prop :=
  (total_interior_angle_sum m n) % 27 = 0

/-- The main theorem stating the minimum value of m + n -/
theorem min_m_plus_n :
  ∃ (m₀ n₀ : ℕ), is_divisible_by_27 m₀ n₀ ∧
    ∀ (m n : ℕ), is_divisible_by_27 m n → m₀ + n₀ ≤ m + n :=
sorry

end NUMINAMATH_CALUDE_min_m_plus_n_l861_86165


namespace NUMINAMATH_CALUDE_units_digit_34_pow_30_l861_86140

theorem units_digit_34_pow_30 : (34^30) % 10 = 6 := by
  sorry

end NUMINAMATH_CALUDE_units_digit_34_pow_30_l861_86140


namespace NUMINAMATH_CALUDE_g_derivative_at_midpoint_sign_l861_86187

/-- The function g(x) defined as x + a * ln(x) - k * x^2 --/
noncomputable def g (a k x : ℝ) : ℝ := x + a * Real.log x - k * x^2

/-- The derivative of g(x) --/
noncomputable def g' (a k x : ℝ) : ℝ := 1 + a / x - 2 * k * x

theorem g_derivative_at_midpoint_sign (a k x₁ x₂ : ℝ) 
  (hk : k > 0) 
  (hx : x₁ ≠ x₂) 
  (hz₁ : g a k x₁ = 0) 
  (hz₂ : g a k x₂ = 0) :
  (a > 0 → g' a k ((x₁ + x₂) / 2) < 0) ∧
  (a < 0 → g' a k ((x₁ + x₂) / 2) > 0) :=
by sorry

end NUMINAMATH_CALUDE_g_derivative_at_midpoint_sign_l861_86187


namespace NUMINAMATH_CALUDE_yarn_crochet_length_l861_86120

def yarn_problem (total_length : ℝ) (num_parts : ℕ) (parts_used : ℕ) : Prop :=
  total_length = 10 ∧ 
  num_parts = 5 ∧ 
  parts_used = 3 ∧ 
  (total_length / num_parts) * parts_used = 6

theorem yarn_crochet_length : 
  ∀ (total_length : ℝ) (num_parts : ℕ) (parts_used : ℕ),
  yarn_problem total_length num_parts parts_used :=
by
  sorry

end NUMINAMATH_CALUDE_yarn_crochet_length_l861_86120


namespace NUMINAMATH_CALUDE_parallel_vectors_trig_expression_l861_86143

/-- Given vectors a and b that are parallel, prove that the given expression equals 3√2 -/
theorem parallel_vectors_trig_expression (x : ℝ) 
  (a : ℝ × ℝ) (b : ℝ × ℝ) 
  (ha : a = (Real.sin x, 2)) 
  (hb : b = (Real.cos x, 1)) 
  (h_parallel : ∃ (k : ℝ), a = k • b) : 
  (2 * Real.sin (x + π/4)) / (Real.sin x - Real.cos x) = 3 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_parallel_vectors_trig_expression_l861_86143


namespace NUMINAMATH_CALUDE_original_savings_amount_l861_86175

/-- Proves that the original savings amount was $11,000 given the spending pattern and remaining balance. -/
theorem original_savings_amount (initial_savings : ℝ) : 
  initial_savings * (1 - 0.2 - 0.4) - 1500 = 2900 → 
  initial_savings = 11000 := by
sorry

end NUMINAMATH_CALUDE_original_savings_amount_l861_86175


namespace NUMINAMATH_CALUDE_legs_of_special_triangle_l861_86189

/-- A right triangle with an inscribed circle -/
structure RightTriangleWithInscribedCircle where
  /-- The length of one leg of the triangle -/
  leg1 : ℝ
  /-- The length of the other leg of the triangle -/
  leg2 : ℝ
  /-- The radius of the inscribed circle -/
  radius : ℝ
  /-- The distance from the center of the inscribed circle to one end of the hypotenuse -/
  dist1 : ℝ
  /-- The distance from the center of the inscribed circle to the other end of the hypotenuse -/
  dist2 : ℝ
  /-- The leg1 is positive -/
  leg1_pos : 0 < leg1
  /-- The leg2 is positive -/
  leg2_pos : 0 < leg2
  /-- The radius is positive -/
  radius_pos : 0 < radius
  /-- The dist1 is positive -/
  dist1_pos : 0 < dist1
  /-- The dist2 is positive -/
  dist2_pos : 0 < dist2
  /-- The triangle satisfies the Pythagorean theorem -/
  pythagorean : leg1^2 + leg2^2 = (dist1 + dist2)^2
  /-- The radius is related to the legs and distances as per the properties of an inscribed circle -/
  inscribed_circle : radius = (leg1 + leg2 - dist1 - dist2) / 2

/-- 
If the center of the inscribed circle in a right triangle is at distances √5 and √10 
from the ends of the hypotenuse, then the legs of the triangle are 3 and 4.
-/
theorem legs_of_special_triangle (t : RightTriangleWithInscribedCircle) 
  (h1 : t.dist1 = Real.sqrt 5) (h2 : t.dist2 = Real.sqrt 10) : 
  (t.leg1 = 3 ∧ t.leg2 = 4) ∨ (t.leg1 = 4 ∧ t.leg2 = 3) := by
  sorry

end NUMINAMATH_CALUDE_legs_of_special_triangle_l861_86189


namespace NUMINAMATH_CALUDE_mans_speed_with_current_l861_86133

/-- Given a man's speed against the current and the speed of the current,
    calculate the man's speed with the current. -/
def speed_with_current (speed_against_current : ℝ) (current_speed : ℝ) : ℝ :=
  speed_against_current + 2 * current_speed

/-- Theorem stating that given the specific conditions in the problem,
    the man's speed with the current is 20 kmph. -/
theorem mans_speed_with_current :
  let speed_against_current := 18
  let current_speed := 1
  speed_with_current speed_against_current current_speed = 20 := by
  sorry

#eval speed_with_current 18 1

end NUMINAMATH_CALUDE_mans_speed_with_current_l861_86133


namespace NUMINAMATH_CALUDE_circular_sector_angle_l861_86139

/-- Given a circular sector with arc length 30 and diameter 16, 
    prove that its central angle in radians is 15/4 -/
theorem circular_sector_angle (arc_length : ℝ) (diameter : ℝ) 
  (h1 : arc_length = 30) (h2 : diameter = 16) :
  arc_length / (diameter / 2) = 15 / 4 := by
  sorry

end NUMINAMATH_CALUDE_circular_sector_angle_l861_86139


namespace NUMINAMATH_CALUDE_fraction_comparison_l861_86180

theorem fraction_comparison : (10^1984 + 1) / (10^1985) > (10^1985 + 1) / (10^1986) := by
  sorry

end NUMINAMATH_CALUDE_fraction_comparison_l861_86180


namespace NUMINAMATH_CALUDE_largest_integral_solution_l861_86192

theorem largest_integral_solution : ∃ x : ℤ, (1 : ℚ) / 4 < x / 6 ∧ x / 6 < 7 / 9 ∧ ∀ y : ℤ, (1 : ℚ) / 4 < y / 6 ∧ y / 6 < 7 / 9 → y ≤ x := by
  sorry

end NUMINAMATH_CALUDE_largest_integral_solution_l861_86192


namespace NUMINAMATH_CALUDE_modular_exponentiation_difference_l861_86115

theorem modular_exponentiation_difference (n : ℕ) :
  (51 : ℤ) ^ n - (9 : ℤ) ^ n ≡ 0 [ZMOD 6] :=
by
  sorry

end NUMINAMATH_CALUDE_modular_exponentiation_difference_l861_86115


namespace NUMINAMATH_CALUDE_polygon_exterior_angles_l861_86149

theorem polygon_exterior_angles (n : ℕ) (exterior_angle : ℝ) : 
  (n : ℝ) * exterior_angle = 360 → exterior_angle = 30 → n = 12 := by
  sorry

end NUMINAMATH_CALUDE_polygon_exterior_angles_l861_86149


namespace NUMINAMATH_CALUDE_triangle_most_stable_l861_86198

-- Define the possible shapes
inductive Shape
  | Heptagon
  | Hexagon
  | Pentagon
  | Triangle

-- Define a property for stability
def is_stable (s : Shape) : Prop :=
  match s with
  | Shape.Triangle => true
  | _ => false

-- Theorem stating that the triangle is the most stable shape
theorem triangle_most_stable :
  ∀ s : Shape, is_stable s → s = Shape.Triangle :=
by
  sorry

end NUMINAMATH_CALUDE_triangle_most_stable_l861_86198

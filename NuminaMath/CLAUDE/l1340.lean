import Mathlib

namespace NUMINAMATH_CALUDE_quadratic_inequality_l1340_134037

theorem quadratic_inequality (a b c : ℝ) 
  (h : ∀ x : ℝ, a * x^2 + b * x + c < 0) : 
  b / a < c / a + 1 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_inequality_l1340_134037


namespace NUMINAMATH_CALUDE_negation_equivalence_l1340_134060

theorem negation_equivalence :
  (¬ ∃ x : ℝ, x^2 - x - 1 > 0) ↔ (∀ x : ℝ, x^2 - x - 1 ≤ 0) :=
by sorry

end NUMINAMATH_CALUDE_negation_equivalence_l1340_134060


namespace NUMINAMATH_CALUDE_isosceles_triangle_perimeter_l1340_134089

/-- An isosceles triangle with sides a, b, and c, where a = b -/
structure IsoscelesTriangle where
  a : ℝ
  b : ℝ
  c : ℝ
  isosceles : a = b
  positive_a : 0 < a
  positive_b : 0 < b
  positive_c : 0 < c
  triangle_inequality : a + b > c ∧ b + c > a ∧ c + a > b

/-- The perimeter of a triangle -/
def perimeter (t : IsoscelesTriangle) : ℝ := t.a + t.b + t.c

/-- Theorem: The perimeter of an isosceles triangle with sides 2 and 5 is 12 -/
theorem isosceles_triangle_perimeter :
  ∃ (t : IsoscelesTriangle), t.a = 2 ∧ t.c = 5 ∧ perimeter t = 12 :=
by sorry

end NUMINAMATH_CALUDE_isosceles_triangle_perimeter_l1340_134089


namespace NUMINAMATH_CALUDE_arithmetic_sequence_middle_term_l1340_134036

/-- Three real numbers form an arithmetic sequence if the middle term is the arithmetic mean of the other two terms -/
def is_arithmetic_sequence (a b c : ℝ) : Prop := b = (a + c) / 2

theorem arithmetic_sequence_middle_term :
  ∀ m : ℝ, is_arithmetic_sequence 2 m 6 → m = 4 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_middle_term_l1340_134036


namespace NUMINAMATH_CALUDE_distinct_positive_factors_of_81_l1340_134075

theorem distinct_positive_factors_of_81 : 
  Finset.card (Nat.divisors 81) = 5 := by
  sorry

end NUMINAMATH_CALUDE_distinct_positive_factors_of_81_l1340_134075


namespace NUMINAMATH_CALUDE_square_of_99_l1340_134032

theorem square_of_99 : (99 : ℕ) ^ 2 = 9801 := by
  sorry

end NUMINAMATH_CALUDE_square_of_99_l1340_134032


namespace NUMINAMATH_CALUDE_bennett_brothers_count_l1340_134025

theorem bennett_brothers_count :
  ∀ (aaron_brothers bennett_brothers : ℕ),
    aaron_brothers = 4 →
    bennett_brothers = 2 * aaron_brothers - 2 →
    bennett_brothers = 6 :=
by
  sorry

end NUMINAMATH_CALUDE_bennett_brothers_count_l1340_134025


namespace NUMINAMATH_CALUDE_polynomial_root_sum_l1340_134097

theorem polynomial_root_sum (c d : ℝ) : 
  (Complex.I * Real.sqrt 2 + 2 : ℂ) ^ 3 + c * (Complex.I * Real.sqrt 2 + 2) + d = 0 → 
  c + d = 14 := by
sorry

end NUMINAMATH_CALUDE_polynomial_root_sum_l1340_134097


namespace NUMINAMATH_CALUDE_solve_for_z_l1340_134053

theorem solve_for_z (x y z : ℤ) (h1 : x^2 = y - 4) (h2 : x = -6) (h3 : y = z + 2) : z = 38 := by
  sorry

end NUMINAMATH_CALUDE_solve_for_z_l1340_134053


namespace NUMINAMATH_CALUDE_legos_in_box_l1340_134043

theorem legos_in_box (total : ℕ) (used : ℕ) (missing : ℕ) (in_box : ℕ) : 
  total = 500 → 
  used = total / 2 → 
  missing = 5 → 
  in_box = total - used - missing → 
  in_box = 245 := by
sorry

end NUMINAMATH_CALUDE_legos_in_box_l1340_134043


namespace NUMINAMATH_CALUDE_linear_function_values_l1340_134033

/-- A linear function y = kx + b passing through (-1, 0) and (2, 1/2) -/
def linear_function (x : ℚ) : ℚ :=
  let k : ℚ := 6
  let b : ℚ := -1
  k * x + b

theorem linear_function_values :
  linear_function 0 = -1 ∧
  linear_function (1/2) = 2 ∧
  linear_function (-1/2) = -4 := by
  sorry


end NUMINAMATH_CALUDE_linear_function_values_l1340_134033


namespace NUMINAMATH_CALUDE_square_side_length_l1340_134035

theorem square_side_length (area : ℝ) (side : ℝ) : 
  area = 1/9 → side^2 = area → side = 1/3 := by
  sorry

end NUMINAMATH_CALUDE_square_side_length_l1340_134035


namespace NUMINAMATH_CALUDE_vertex_coordinates_l1340_134022

def f (x : ℝ) := (x - 1)^2 - 2

theorem vertex_coordinates :
  ∃ (x y : ℝ), (x = 1 ∧ y = -2) ∧
  ∀ (t : ℝ), f t ≥ f x :=
by sorry

end NUMINAMATH_CALUDE_vertex_coordinates_l1340_134022


namespace NUMINAMATH_CALUDE_quadratic_inequality_l1340_134021

-- Define the quadratic function f
def f (b c x : ℝ) : ℝ := x^2 + b*x + c

-- State the theorem
theorem quadratic_inequality (b c : ℝ) 
  (h : ∀ x, f b c (3 + x) = f b c (3 - x)) : 
  f b c 4 < f b c 1 ∧ f b c 1 < f b c (-1) := by
  sorry

end NUMINAMATH_CALUDE_quadratic_inequality_l1340_134021


namespace NUMINAMATH_CALUDE_sqrt_calculation_and_algebraic_expression_l1340_134009

theorem sqrt_calculation_and_algebraic_expression :
  (∃ x : ℝ, x^2 = 18) ∧ 
  (∃ y : ℝ, y^2 = 8) ∧ 
  (∃ z : ℝ, z^2 = 1/2) ∧
  (∃ a : ℝ, a^2 = 5) ∧
  (∃ b : ℝ, b^2 = 3) ∧
  (∃ c : ℝ, c^2 = 12) ∧
  (∃ d : ℝ, d^2 = 27) →
  (∃ x y z : ℝ, x^2 = 18 ∧ y^2 = 8 ∧ z^2 = 1/2 ∧ x - y + z = 3 * Real.sqrt 2 / 2) ∧
  (∃ a b c d : ℝ, a^2 = 5 ∧ b^2 = 3 ∧ c^2 = 12 ∧ d^2 = 27 ∧
    (2*a - 1) * (1 + 2*a) + b * (c - d) = 16) := by
  sorry

end NUMINAMATH_CALUDE_sqrt_calculation_and_algebraic_expression_l1340_134009


namespace NUMINAMATH_CALUDE_arithmetic_sequence_inequality_l1340_134081

-- Define an arithmetic sequence
def arithmetic_sequence (a : ℕ → ℝ) (d : ℝ) : Prop :=
  ∀ n, a (n + 1) = a n + d

theorem arithmetic_sequence_inequality
  (a : ℕ → ℝ) (d : ℝ) (n : ℕ)
  (h_arithmetic : arithmetic_sequence a d)
  (h_positive : d > 0)
  (h_n : n > 1) :
  a 1 * a (n + 1) < a 2 * a n :=
by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_inequality_l1340_134081


namespace NUMINAMATH_CALUDE_reciprocal_sum_l1340_134024

theorem reciprocal_sum (a b : ℝ) (h1 : a ≠ b) (h2 : a / b + a = b / a + b) : 1 / a + 1 / b = -1 := by
  sorry

end NUMINAMATH_CALUDE_reciprocal_sum_l1340_134024


namespace NUMINAMATH_CALUDE_at_least_one_real_root_l1340_134091

theorem at_least_one_real_root (c : ℝ) : 
  ∃ x : ℝ, (x^2 + c*x + 2 = 0) ∨ (x^2 + 2*x + c = 2) := by
  sorry

end NUMINAMATH_CALUDE_at_least_one_real_root_l1340_134091


namespace NUMINAMATH_CALUDE_runner_problem_l1340_134064

/-- Proves that given the conditions of the runner's problem, the time taken for the second half is 10 hours -/
theorem runner_problem (v : ℝ) (h1 : v > 0) : 
  (40 / v = 20 / v + 5) → (40 / (v / 2) = 10) :=
by
  sorry

end NUMINAMATH_CALUDE_runner_problem_l1340_134064


namespace NUMINAMATH_CALUDE_no_function_satisfies_condition_l1340_134092

theorem no_function_satisfies_condition : ¬∃ f : ℝ → ℝ, ∀ x y : ℝ, f x * f y - f (x * y) = 2 * x + 2 * y := by
  sorry

end NUMINAMATH_CALUDE_no_function_satisfies_condition_l1340_134092


namespace NUMINAMATH_CALUDE_spade_calculation_l1340_134030

def spade (x y : ℝ) : ℝ := (x + y) * (x - y)

theorem spade_calculation : spade 5 (spade 6 7 + 2) = -96 := by
  sorry

end NUMINAMATH_CALUDE_spade_calculation_l1340_134030


namespace NUMINAMATH_CALUDE_inscribed_circle_radius_l1340_134070

/-- An isosceles trapezoid circumscribed around a circle -/
structure IsoscelesTrapezoid where
  /-- The area of the trapezoid -/
  area : ℝ
  /-- The angle at the base of the trapezoid in radians -/
  baseAngle : ℝ
  /-- The radius of the inscribed circle -/
  radius : ℝ
  /-- The area is positive -/
  area_pos : 0 < area
  /-- The base angle is 30° (π/6 radians) -/
  angle_is_30deg : baseAngle = Real.pi / 6
  /-- The radius is positive -/
  radius_pos : 0 < radius

/-- Theorem: The radius of the inscribed circle in an isosceles trapezoid -/
theorem inscribed_circle_radius (t : IsoscelesTrapezoid) : 
  t.radius = Real.sqrt (2 * t.area) / 4 := by
  sorry

end NUMINAMATH_CALUDE_inscribed_circle_radius_l1340_134070


namespace NUMINAMATH_CALUDE_quadratic_roots_and_inequality_l1340_134000

-- Define the quadratic equation
def quadratic (a x : ℝ) : ℝ := x^2 - 2*a*x + a^2 + a - 2

-- Define the proposition p
def p (a : ℝ) : Prop := ∃ x : ℝ, quadratic a x = 0

-- Define the proposition q
def q (m a : ℝ) : Prop := m - 1 ≤ a ∧ a ≤ m + 3

theorem quadratic_roots_and_inequality (a m : ℝ) :
  (¬ p a → a > 2) ∧
  ((∀ m, p a → q m a) ∧ (∃ m, q m a ∧ ¬ p a) → m ≤ -1) :=
sorry

end NUMINAMATH_CALUDE_quadratic_roots_and_inequality_l1340_134000


namespace NUMINAMATH_CALUDE_nut_mixture_weight_l1340_134090

/-- Given a mixture of nuts with 5 parts almonds to 2 parts walnuts by weight,
    and 250 pounds of almonds, the total weight of the mixture is 350 pounds. -/
theorem nut_mixture_weight (almond_parts : ℕ) (walnut_parts : ℕ) (almond_weight : ℝ) :
  almond_parts = 5 →
  walnut_parts = 2 →
  almond_weight = 250 →
  (almond_weight / almond_parts) * (almond_parts + walnut_parts) = 350 := by
  sorry

#check nut_mixture_weight

end NUMINAMATH_CALUDE_nut_mixture_weight_l1340_134090


namespace NUMINAMATH_CALUDE_geometric_sequence_product_l1340_134008

-- Define a geometric sequence
def geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = a n * r

-- State the theorem
theorem geometric_sequence_product (a : ℕ → ℝ) :
  geometric_sequence a → a 4 = 4 → a 2 * a 6 = 16 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_product_l1340_134008


namespace NUMINAMATH_CALUDE_third_number_in_set_l1340_134014

theorem third_number_in_set (x : ℝ) : 
  let set1 := [10, 70, 28]
  let set2 := [20, 40, x]
  (set2.sum / set2.length : ℝ) = (set1.sum / set1.length : ℝ) + 4 →
  x = 60 := by
sorry

end NUMINAMATH_CALUDE_third_number_in_set_l1340_134014


namespace NUMINAMATH_CALUDE_two_x_plus_y_equals_seven_l1340_134020

theorem two_x_plus_y_equals_seven 
  (h1 : (x + y) / 3 = 1.6666666666666667)
  (h2 : x + 2 * y = 8) : 
  2 * x + y = 7 := by
  sorry

end NUMINAMATH_CALUDE_two_x_plus_y_equals_seven_l1340_134020


namespace NUMINAMATH_CALUDE_max_product_of_radii_l1340_134071

/-- Two circles C₁ and C₂ are externally tangent -/
def externally_tangent (a b : ℝ) : Prop :=
  a + b = 3

/-- The equation of circle C₁ -/
def circle_C₁ (a : ℝ) (x y : ℝ) : Prop :=
  (x + a)^2 + (y - 2)^2 = 1

/-- The equation of circle C₂ -/
def circle_C₂ (b : ℝ) (x y : ℝ) : Prop :=
  (x - b)^2 + (y - 2)^2 = 4

theorem max_product_of_radii (a b : ℝ) 
  (ha : a > 0) (hb : b > 0) 
  (h_tangent : externally_tangent a b) :
  a * b ≤ 9/4 ∧ ∃ (a₀ b₀ : ℝ), a₀ * b₀ = 9/4 ∧ externally_tangent a₀ b₀ := by
  sorry

end NUMINAMATH_CALUDE_max_product_of_radii_l1340_134071


namespace NUMINAMATH_CALUDE_quadratic_always_nonnegative_implies_m_range_l1340_134059

theorem quadratic_always_nonnegative_implies_m_range (m : ℝ) :
  (∀ x : ℝ, x^2 + m*x + 2*m - 3 ≥ 0) → 2 ≤ m ∧ m ≤ 6 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_always_nonnegative_implies_m_range_l1340_134059


namespace NUMINAMATH_CALUDE_third_line_product_l1340_134067

/-- Given two positive real numbers a and b, prove that 
    x = -a/2 + √(a²/4 + b²) satisfies x(x + a) = b² -/
theorem third_line_product (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  let x := -a/2 + Real.sqrt (a^2/4 + b^2)
  x * (x + a) = b^2 := by
  sorry

end NUMINAMATH_CALUDE_third_line_product_l1340_134067


namespace NUMINAMATH_CALUDE_set_inclusion_implies_m_range_l1340_134031

theorem set_inclusion_implies_m_range (m : ℝ) :
  let P : Set ℝ := {x | -2 ≤ x ∧ x ≤ 10}
  let S : Set ℝ := {x | 1 - m ≤ x ∧ x ≤ 1 + m}
  (S.Nonempty) → (S ⊆ P) → (0 ≤ m ∧ m ≤ 3) :=
by
  sorry

end NUMINAMATH_CALUDE_set_inclusion_implies_m_range_l1340_134031


namespace NUMINAMATH_CALUDE_inequality_proof_l1340_134018

theorem inequality_proof (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (h_abc : a * b * c = 1) :
  1 / (a^3 * (b + c)) + 1 / (b^3 * (c + a)) + 1 / (c^3 * (a + b)) ≥ 
  3/2 + 1/4 * (a * (c - b)^2 / (c + b) + b * (c - a)^2 / (c + a) + c * (b - a)^2 / (b + a)) := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l1340_134018


namespace NUMINAMATH_CALUDE_sugar_cube_weight_l1340_134040

/-- The weight of sugar cubes in the first group -/
def weight_first_group : ℝ := 10

/-- The number of ants in the first group -/
def ants_first : ℕ := 15

/-- The number of sugar cubes moved by the first group -/
def cubes_first : ℕ := 600

/-- The time taken by the first group (in hours) -/
def time_first : ℝ := 5

/-- The number of ants in the second group -/
def ants_second : ℕ := 20

/-- The number of sugar cubes moved by the second group -/
def cubes_second : ℕ := 960

/-- The time taken by the second group (in hours) -/
def time_second : ℝ := 3

/-- The weight of sugar cubes in the second group -/
def weight_second : ℝ := 5

theorem sugar_cube_weight :
  (ants_first : ℝ) * cubes_second * time_first * weight_second =
  (ants_second : ℝ) * cubes_first * time_second * weight_first_group :=
by sorry

end NUMINAMATH_CALUDE_sugar_cube_weight_l1340_134040


namespace NUMINAMATH_CALUDE_students_spend_two_dollars_l1340_134041

/-- The price of one pencil in cents -/
def pencil_price : ℕ := 20

/-- The number of pencils Tolu wants -/
def tolu_pencils : ℕ := 3

/-- The number of pencils Robert wants -/
def robert_pencils : ℕ := 5

/-- The number of pencils Melissa wants -/
def melissa_pencils : ℕ := 2

/-- The total amount spent by the students in dollars -/
def total_spent : ℚ := (pencil_price * (tolu_pencils + robert_pencils + melissa_pencils)) / 100

theorem students_spend_two_dollars : total_spent = 2 := by
  sorry

end NUMINAMATH_CALUDE_students_spend_two_dollars_l1340_134041


namespace NUMINAMATH_CALUDE_range_of_a_l1340_134055

-- Define the line l: 2x - 3y + 1 = 0
def line (x y : ℝ) : Prop := 2 * x - 3 * y + 1 = 0

-- Define the points M and N
def point_M (a : ℝ) : ℝ × ℝ := (1, -a)
def point_N (a : ℝ) : ℝ × ℝ := (a, 1)

-- Define the condition for points being on opposite sides of the line
def opposite_sides (a : ℝ) : Prop :=
  (2 * (point_M a).1 - 3 * (point_M a).2 + 1) * (2 * (point_N a).1 - 3 * (point_N a).2 + 1) < 0

-- Theorem statement
theorem range_of_a (a : ℝ) : opposite_sides a → -1 < a ∧ a < 1 := by
  sorry

end NUMINAMATH_CALUDE_range_of_a_l1340_134055


namespace NUMINAMATH_CALUDE_cubic_factor_sum_l1340_134093

/-- Given a cubic polynomial x^3 + ax^2 + bx + 8 with factors (x+1) and (x+2),
    prove that a + b = 21 -/
theorem cubic_factor_sum (a b : ℝ) : 
  (∃ c : ℝ, ∀ x : ℝ, x^3 + a*x^2 + b*x + 8 = (x+1)*(x+2)*(x+c)) →
  a + b = 21 := by
sorry

end NUMINAMATH_CALUDE_cubic_factor_sum_l1340_134093


namespace NUMINAMATH_CALUDE_box_dimensions_l1340_134061

theorem box_dimensions (x y z : ℝ) 
  (volume : x * y * z = 160)
  (face1 : y * z = 80)
  (face2 : x * z = 40)
  (face3 : x * y = 32) :
  x = 4 ∧ y = 8 ∧ z = 10 := by
  sorry

end NUMINAMATH_CALUDE_box_dimensions_l1340_134061


namespace NUMINAMATH_CALUDE_expected_boy_girl_pairs_l1340_134062

theorem expected_boy_girl_pairs (n_boys n_girls : ℕ) (h_boys : n_boys = 8) (h_girls : n_girls = 12) :
  let total := n_boys + n_girls
  let inner_boys := n_boys - 2
  let inner_pairs := total - 1
  let inner_prob := (inner_boys * n_girls) / ((inner_boys + n_girls) * (inner_boys + n_girls - 1))
  let end_prob := n_girls / total
  (inner_pairs - 2) * (2 * inner_prob) + 2 * end_prob = 144/17 + 24/19 := by
  sorry

end NUMINAMATH_CALUDE_expected_boy_girl_pairs_l1340_134062


namespace NUMINAMATH_CALUDE_function_properties_l1340_134050

def is_even_function (f : ℝ → ℝ) : Prop := ∀ x, f x = f (-x)

def is_periodic_function (f : ℝ → ℝ) (p : ℝ) : Prop := ∀ x, f (x + p) = f x

theorem function_properties (f : ℝ → ℝ) 
  (h_not_constant : ∃ x y, f x ≠ f y)
  (h_periodic : ∀ x, f (x - 1) = f (x + 1))
  (h_symmetry : ∀ x, f (x + 1) = f (1 - x)) :
  is_even_function f ∧ is_periodic_function f 2 :=
sorry

end NUMINAMATH_CALUDE_function_properties_l1340_134050


namespace NUMINAMATH_CALUDE_savings_calculation_l1340_134082

/-- Calculates the total savings given a daily savings amount and number of days -/
def totalSavings (dailySavings : ℕ) (days : ℕ) : ℕ :=
  dailySavings * days

/-- Theorem: If a person saves $24 every day for 365 days, the total savings will be $8,760 -/
theorem savings_calculation :
  totalSavings 24 365 = 8760 := by
  sorry

end NUMINAMATH_CALUDE_savings_calculation_l1340_134082


namespace NUMINAMATH_CALUDE_knights_and_knaves_solution_l1340_134004

-- Define the types for residents and their statements
inductive Resident : Type
| A
| B
| C

inductive Status : Type
| Knight
| Knave

-- Define the statement made by A
def statement_A (status : Resident → Status) : Prop :=
  status Resident.C = Status.Knight → status Resident.B = Status.Knave

-- Define the statement made by C
def statement_C (status : Resident → Status) : Prop :=
  status Resident.A ≠ status Resident.C ∧
  ((status Resident.A = Status.Knight ∧ status Resident.C = Status.Knave) ∨
   (status Resident.A = Status.Knave ∧ status Resident.C = Status.Knight))

-- Define the truthfulness of statements based on the speaker's status
def is_truthful (status : Resident → Status) (r : Resident) (stmt : Prop) : Prop :=
  (status r = Status.Knight ∧ stmt) ∨ (status r = Status.Knave ∧ ¬stmt)

-- Theorem stating the solution
theorem knights_and_knaves_solution :
  ∃ (status : Resident → Status),
    is_truthful status Resident.A (statement_A status) ∧
    is_truthful status Resident.C (statement_C status) ∧
    status Resident.A = Status.Knave ∧
    status Resident.B = Status.Knight ∧
    status Resident.C = Status.Knight :=
sorry

end NUMINAMATH_CALUDE_knights_and_knaves_solution_l1340_134004


namespace NUMINAMATH_CALUDE_technician_average_salary_l1340_134011

/-- Calculates the average salary of technicians in a workshop --/
theorem technician_average_salary
  (total_workers : ℕ)
  (total_average : ℚ)
  (num_technicians : ℕ)
  (non_technician_average : ℚ)
  (h1 : total_workers = 30)
  (h2 : total_average = 8000)
  (h3 : num_technicians = 10)
  (h4 : non_technician_average = 6000)
  : (total_average * total_workers - non_technician_average * (total_workers - num_technicians)) / num_technicians = 12000 := by
  sorry

end NUMINAMATH_CALUDE_technician_average_salary_l1340_134011


namespace NUMINAMATH_CALUDE_green_shirt_pairs_l1340_134017

theorem green_shirt_pairs (red_students green_students total_students total_pairs red_red_pairs : ℕ)
  (h1 : red_students = 63)
  (h2 : green_students = 69)
  (h3 : total_students = red_students + green_students)
  (h4 : total_pairs = 66)
  (h5 : red_red_pairs = 25)
  (h6 : total_students = 2 * total_pairs) :
  2 * red_red_pairs + (red_students - 2 * red_red_pairs) + 2 * ((green_students - (red_students - 2 * red_red_pairs)) / 2) = total_students :=
by sorry

end NUMINAMATH_CALUDE_green_shirt_pairs_l1340_134017


namespace NUMINAMATH_CALUDE_perpendicular_lines_from_perpendicular_planes_l1340_134087

-- Define the types for planes and lines
variable (Plane : Type) (Line : Type)

-- Define the relationships between planes and lines
variable (is_perpendicular_line_plane : Line → Plane → Prop)
variable (is_perpendicular_plane_plane : Plane → Plane → Prop)
variable (is_perpendicular_line_line : Line → Line → Prop)
variable (are_distinct : Plane → Plane → Prop)
variable (are_non_intersecting : Line → Line → Prop)

-- State the theorem
theorem perpendicular_lines_from_perpendicular_planes 
  (α β : Plane) (l m : Line)
  (h_distinct : are_distinct α β)
  (h_non_intersecting : are_non_intersecting l m)
  (h1 : is_perpendicular_line_plane l α)
  (h2 : is_perpendicular_line_plane m β)
  (h3 : is_perpendicular_plane_plane α β) :
  is_perpendicular_line_line l m :=
sorry

end NUMINAMATH_CALUDE_perpendicular_lines_from_perpendicular_planes_l1340_134087


namespace NUMINAMATH_CALUDE_interior_angles_ratio_l1340_134046

-- Define a triangle type
structure Triangle where
  -- Define exterior angles
  ext_angle1 : ℝ
  ext_angle2 : ℝ
  ext_angle3 : ℝ
  -- Condition: exterior angles sum to 360°
  sum_ext_angles : ext_angle1 + ext_angle2 + ext_angle3 = 360
  -- Condition: ratio of exterior angles is 3:4:5
  ratio_ext_angles : ∃ (x : ℝ), ext_angle1 = 3*x ∧ ext_angle2 = 4*x ∧ ext_angle3 = 5*x

-- Define interior angles
def interior_angle1 (t : Triangle) : ℝ := 180 - t.ext_angle1
def interior_angle2 (t : Triangle) : ℝ := 180 - t.ext_angle2
def interior_angle3 (t : Triangle) : ℝ := 180 - t.ext_angle3

-- Theorem statement
theorem interior_angles_ratio (t : Triangle) :
  ∃ (k : ℝ), interior_angle1 t = 3*k ∧ interior_angle2 t = 2*k ∧ interior_angle3 t = k := by
  sorry

end NUMINAMATH_CALUDE_interior_angles_ratio_l1340_134046


namespace NUMINAMATH_CALUDE_hari_joined_after_five_months_l1340_134013

/-- Represents the business scenario with two partners --/
structure Business where
  praveen_investment : ℕ
  hari_investment : ℕ
  profit_ratio_praveen : ℕ
  profit_ratio_hari : ℕ
  total_duration : ℕ

/-- Calculates the number of months after which Hari joined the business --/
def months_until_hari_joined (b : Business) : ℕ :=
  let x := b.total_duration - (b.praveen_investment * b.total_duration * b.profit_ratio_hari) / 
           (b.hari_investment * b.profit_ratio_praveen)
  x

/-- Theorem stating that Hari joined 5 months after Praveen started the business --/
theorem hari_joined_after_five_months (b : Business) 
  (h1 : b.praveen_investment = 3220)
  (h2 : b.hari_investment = 8280)
  (h3 : b.profit_ratio_praveen = 2)
  (h4 : b.profit_ratio_hari = 3)
  (h5 : b.total_duration = 12) :
  months_until_hari_joined b = 5 := by
  sorry

#eval months_until_hari_joined ⟨3220, 8280, 2, 3, 12⟩

end NUMINAMATH_CALUDE_hari_joined_after_five_months_l1340_134013


namespace NUMINAMATH_CALUDE_claire_balloons_l1340_134079

theorem claire_balloons (initial : ℕ) : 
  initial - 12 - 9 + 11 = 39 → initial = 49 := by
  sorry

end NUMINAMATH_CALUDE_claire_balloons_l1340_134079


namespace NUMINAMATH_CALUDE_principal_amount_is_875_l1340_134047

/-- Calculates the principal amount given the interest rate, time, and total interest. -/
def calculate_principal (interest_rate : ℚ) (time : ℕ) (total_interest : ℚ) : ℚ :=
  (total_interest * 100) / (interest_rate * time)

/-- Theorem stating that given the specified conditions, the principal amount is 875. -/
theorem principal_amount_is_875 :
  let interest_rate : ℚ := 12
  let time : ℕ := 20
  let total_interest : ℚ := 2100
  calculate_principal interest_rate time total_interest = 875 := by sorry

end NUMINAMATH_CALUDE_principal_amount_is_875_l1340_134047


namespace NUMINAMATH_CALUDE_two_color_theorem_l1340_134026

-- Define a type for regions in the plane
def Region : Type := ℕ

-- Define a type for colors
inductive Color
| Red : Color
| Blue : Color

-- Define a function type for coloring the map
def Coloring := Region → Color

-- Define a relation for adjacent regions
def Adjacent : Region → Region → Prop := sorry

-- Define a property for a valid coloring
def ValidColoring (coloring : Coloring) : Prop :=
  ∀ r1 r2 : Region, Adjacent r1 r2 → coloring r1 ≠ coloring r2

-- Define a type for the map configuration
structure MapConfiguration :=
  (num_lines : ℕ)
  (num_circles : ℕ)

-- State the theorem
theorem two_color_theorem :
  ∀ (config : MapConfiguration), ∃ (coloring : Coloring), ValidColoring coloring :=
sorry

end NUMINAMATH_CALUDE_two_color_theorem_l1340_134026


namespace NUMINAMATH_CALUDE_smallest_n_perfect_square_and_fifth_power_l1340_134085

theorem smallest_n_perfect_square_and_fifth_power : 
  ∃ (n : ℕ), n > 0 ∧ 
  (∃ (k : ℕ), 4 * n = k^2) ∧ 
  (∃ (m : ℕ), 5 * n = m^5) ∧
  (∀ (x : ℕ), x > 0 → 
    ((∃ (y : ℕ), 4 * x = y^2) ∧ (∃ (z : ℕ), 5 * x = z^5)) → 
    x ≥ 625) ∧
  n = 625 :=
sorry

end NUMINAMATH_CALUDE_smallest_n_perfect_square_and_fifth_power_l1340_134085


namespace NUMINAMATH_CALUDE_negation_theorem1_negation_theorem2_l1340_134073

-- Define a type for triangles
structure Triangle where
  -- You might add more properties here if needed
  interiorAngleSum : ℝ

-- Define the propositions
def proposition1 : Prop := ∃ t : Triangle, t.interiorAngleSum ≠ 180

def proposition2 : Prop := ∀ x : ℝ, |x| + x^2 ≥ 0

-- State the theorems
theorem negation_theorem1 : 
  (¬ proposition1) ↔ (∀ t : Triangle, t.interiorAngleSum = 180) :=
sorry

theorem negation_theorem2 : 
  (¬ proposition2) ↔ (∃ x : ℝ, |x| + x^2 < 0) :=
sorry

end NUMINAMATH_CALUDE_negation_theorem1_negation_theorem2_l1340_134073


namespace NUMINAMATH_CALUDE_intersecting_circles_equal_chords_l1340_134065

/-- Given two intersecting circles with radii 10 and 8 units, whose centers are 15 units apart,
    if a line is drawn through their intersection point P such that it creates equal chords QP and PR,
    then the square of the length of chord QP is 250. -/
theorem intersecting_circles_equal_chords (r₁ r₂ d : ℝ) (P Q R : ℝ × ℝ) :
  r₁ = 10 →
  r₂ = 8 →
  d = 15 →
  (P.1 - Q.1)^2 + (P.2 - Q.2)^2 = (P.1 - R.1)^2 + (P.2 - R.2)^2 →
  (P.1 - Q.1)^2 + (P.2 - Q.2)^2 = 250 :=
by sorry

end NUMINAMATH_CALUDE_intersecting_circles_equal_chords_l1340_134065


namespace NUMINAMATH_CALUDE_intersection_implies_z_value_l1340_134027

-- Define the complex number i
def i : ℂ := Complex.I

-- Define the sets M and N
def M (z : ℂ) : Set ℂ := {1, 2, z * i}
def N : Set ℂ := {3, 4}

-- State the theorem
theorem intersection_implies_z_value (z : ℂ) : 
  M z ∩ N = {4} → z = -4 * i :=
by
  sorry

end NUMINAMATH_CALUDE_intersection_implies_z_value_l1340_134027


namespace NUMINAMATH_CALUDE_problem_statement_l1340_134069

theorem problem_statement (x y : ℝ) (h : 1 ≤ x^2 - x*y + y^2 ∧ x^2 - x*y + y^2 ≤ 2) :
  (2/9 ≤ x^4 + y^4 ∧ x^4 + y^4 ≤ 8) ∧
  (∀ n : ℕ, n ≥ 3 → x^(2*n) + y^(2*n) ≥ 2/3^n) := by
  sorry

end NUMINAMATH_CALUDE_problem_statement_l1340_134069


namespace NUMINAMATH_CALUDE_total_shells_formula_l1340_134042

/-- The total number of shells picked up in two hours -/
def total_shells (x : ℚ) : ℚ :=
  x + (x + 32)

/-- Theorem stating that the total number of shells is equal to 2x + 32 -/
theorem total_shells_formula (x : ℚ) : total_shells x = 2 * x + 32 := by
  sorry

end NUMINAMATH_CALUDE_total_shells_formula_l1340_134042


namespace NUMINAMATH_CALUDE_sum_geq_three_cube_root_three_l1340_134086

theorem sum_geq_three_cube_root_three (a b c : ℝ) 
  (h_pos_a : a > 0) (h_pos_b : b > 0) (h_pos_c : c > 0)
  (h_eq : a^3 + b^3 + c^3 = a^2 * b^2 * c^2) : 
  a + b + c ≥ 3 * (3 : ℝ)^(1/3) := by
  sorry

end NUMINAMATH_CALUDE_sum_geq_three_cube_root_three_l1340_134086


namespace NUMINAMATH_CALUDE_specific_stick_displacement_l1340_134084

/-- Represents a uniform stick leaning against a support -/
structure LeaningStick where
  length : ℝ
  projection : ℝ

/-- Calculates the final horizontal displacement of a leaning stick after falling -/
def finalDisplacement (stick : LeaningStick) : ℝ :=
  sorry

/-- Theorem stating the final displacement of a specific stick configuration -/
theorem specific_stick_displacement :
  let stick : LeaningStick := { length := 120, projection := 70 }
  finalDisplacement stick = 25 := by sorry

end NUMINAMATH_CALUDE_specific_stick_displacement_l1340_134084


namespace NUMINAMATH_CALUDE_female_students_count_l1340_134098

theorem female_students_count (total_students sample_size : ℕ) 
  (h1 : total_students = 1600)
  (h2 : sample_size = 200)
  (h3 : ∃ (sampled_girls sampled_boys : ℕ), 
    sampled_girls + sampled_boys = sample_size ∧ 
    sampled_boys = sampled_girls + 10) :
  ∃ (female_students : ℕ), female_students = 760 ∧ 
    female_students * sample_size = sampled_girls * total_students :=
by
  sorry


end NUMINAMATH_CALUDE_female_students_count_l1340_134098


namespace NUMINAMATH_CALUDE_work_completion_time_l1340_134003

/-- The number of days it takes for A to complete the work alone -/
def days_for_A : ℕ := 6

/-- The number of days it takes for B to complete the work alone -/
def days_for_B : ℕ := 8

/-- The number of days it takes for A, B, and C to complete the work together -/
def days_for_ABC : ℕ := 3

/-- The total payment for the work in rupees -/
def total_payment : ℕ := 1200

/-- C's share of the payment in rupees -/
def C_share : ℕ := 150

theorem work_completion_time :
  (1 : ℚ) / days_for_A + (1 : ℚ) / days_for_B + 
  ((C_share : ℚ) / total_payment) * ((1 : ℚ) / days_for_ABC) = 
  (1 : ℚ) / days_for_ABC := by sorry

end NUMINAMATH_CALUDE_work_completion_time_l1340_134003


namespace NUMINAMATH_CALUDE_f_range_characterization_l1340_134074

noncomputable def f (x : ℝ) := Real.sqrt 3 * Real.sin x - Real.cos x

theorem f_range_characterization :
  ∀ x : ℝ, f x ≥ 1 ↔ ∃ k : ℤ, 2 * k * Real.pi + Real.pi / 3 ≤ x ∧ x ≤ 2 * k * Real.pi + Real.pi :=
sorry

end NUMINAMATH_CALUDE_f_range_characterization_l1340_134074


namespace NUMINAMATH_CALUDE_fixed_point_on_line_l1340_134039

theorem fixed_point_on_line (m : ℝ) : 
  (3 * m - 2) * (-3/4 : ℝ) - (m - 2) * (-13/4 : ℝ) - (m - 5) = 0 := by
  sorry

end NUMINAMATH_CALUDE_fixed_point_on_line_l1340_134039


namespace NUMINAMATH_CALUDE_least_five_digit_congruent_to_7_mod_21_l1340_134002

theorem least_five_digit_congruent_to_7_mod_21 :
  ∀ n : ℕ, 
    n ≥ 10000 ∧ n ≤ 99999 ∧ n % 21 = 7 → n ≥ 10003 :=
by
  sorry

end NUMINAMATH_CALUDE_least_five_digit_congruent_to_7_mod_21_l1340_134002


namespace NUMINAMATH_CALUDE_new_person_age_l1340_134015

/-- Given a group of 10 persons where replacing a 46-year-old person with a new person
    decreases the average age by 3 years, prove that the age of the new person is 16 years. -/
theorem new_person_age (T : ℝ) (A : ℝ) : 
  (T / 10 = (T - 46 + A) / 10 + 3) → A = 16 := by
  sorry

end NUMINAMATH_CALUDE_new_person_age_l1340_134015


namespace NUMINAMATH_CALUDE_double_arccos_three_fifths_equals_arcsin_twentyfour_twentyfifths_l1340_134007

theorem double_arccos_three_fifths_equals_arcsin_twentyfour_twentyfifths :
  2 * Real.arccos (3/5) = Real.arcsin (24/25) := by
  sorry

end NUMINAMATH_CALUDE_double_arccos_three_fifths_equals_arcsin_twentyfour_twentyfifths_l1340_134007


namespace NUMINAMATH_CALUDE_quarter_sector_area_l1340_134080

/-- The area of a quarter sector of a circle with diameter 10 meters -/
theorem quarter_sector_area (d : ℝ) (h : d = 10) : 
  (π * (d / 2)^2) / 4 = 6.25 * π := by
  sorry

end NUMINAMATH_CALUDE_quarter_sector_area_l1340_134080


namespace NUMINAMATH_CALUDE_polynomial_root_property_l1340_134083

/-- Given a polynomial x^3 + ax^2 + bx + 18b with nonzero integer coefficients a and b,
    if it has two coinciding integer roots and all three roots are integers,
    then |ab| = 1440 -/
theorem polynomial_root_property (a b : ℤ) (ha : a ≠ 0) (hb : b ≠ 0) :
  (∃ r s : ℤ, (∀ x : ℝ, x^3 + a*x^2 + b*x + 18*b = (x - r)^2 * (x - s)) ∧
              r ≠ s) →
  |a * b| = 1440 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_root_property_l1340_134083


namespace NUMINAMATH_CALUDE_expression_is_equation_l1340_134052

/-- Definition of an equation -/
def is_equation (e : Prop) : Prop :=
  ∃ (x : ℝ), ∃ (f g : ℝ → ℝ), e = (f x = g x)

/-- The expression 2x - 1 = 3 is an equation -/
theorem expression_is_equation : is_equation (∃ x : ℝ, 2 * x - 1 = 3) := by
  sorry

end NUMINAMATH_CALUDE_expression_is_equation_l1340_134052


namespace NUMINAMATH_CALUDE_square_side_length_l1340_134029

theorem square_side_length (area : ℝ) (side : ℝ) (h1 : area = 1 / 9) (h2 : side * side = area) : 
  side = 1 / 3 := by
sorry

end NUMINAMATH_CALUDE_square_side_length_l1340_134029


namespace NUMINAMATH_CALUDE_triangle_area_ratio_l1340_134019

/-- Given two triangles AEF and AFC sharing a common vertex A, 
    where EF:FC = 3:5 and the area of AEF is 27, 
    prove that the area of AFC is 45. -/
theorem triangle_area_ratio (EF FC : ℝ) (area_AEF area_AFC : ℝ) : 
  EF / FC = 3 / 5 → 
  area_AEF = 27 → 
  area_AEF / area_AFC = EF / FC → 
  area_AFC = 45 := by
sorry

end NUMINAMATH_CALUDE_triangle_area_ratio_l1340_134019


namespace NUMINAMATH_CALUDE_unfenced_length_l1340_134023

theorem unfenced_length 
  (field_side : ℝ) 
  (wire_cost : ℝ) 
  (budget : ℝ) 
  (h1 : field_side = 5000)
  (h2 : wire_cost = 30)
  (h3 : budget = 120000) : 
  field_side * 4 - (budget / wire_cost) = 1000 := by
  sorry

end NUMINAMATH_CALUDE_unfenced_length_l1340_134023


namespace NUMINAMATH_CALUDE_linear_regression_point_difference_l1340_134077

theorem linear_regression_point_difference (x₀ y₀ : ℝ) : 
  let data_points : List (ℝ × ℝ) := [(1, 2), (3, 5), (6, 8), (x₀, y₀)]
  let x_mean : ℝ := (1 + 3 + 6 + x₀) / 4
  let y_mean : ℝ := (2 + 5 + 8 + y₀) / 4
  let regression_line (x : ℝ) : ℝ := x + 2
  regression_line x_mean = y_mean →
  x₀ - y₀ = -3 := by
sorry

end NUMINAMATH_CALUDE_linear_regression_point_difference_l1340_134077


namespace NUMINAMATH_CALUDE_pure_imaginary_fraction_l1340_134057

theorem pure_imaginary_fraction (a : ℝ) : 
  (∃ (y : ℝ), (Complex.I : ℂ) * y = (1 - a * Complex.I) / (1 + Complex.I)) → a = 1 := by
  sorry

end NUMINAMATH_CALUDE_pure_imaginary_fraction_l1340_134057


namespace NUMINAMATH_CALUDE_tangent_ratio_bounds_l1340_134001

noncomputable def f (x : ℝ) : ℝ := |Real.exp x - 1|

theorem tangent_ratio_bounds (x₁ x₂ : ℝ) (h₁ : x₁ < 0) (h₂ : x₂ > 0) :
  let A := (x₁, f x₁)
  let B := (x₂, f x₂)
  let M := (0, (1 - Real.exp x₁) + x₁ * Real.exp x₁)
  let N := (0, (Real.exp x₂ - 1) - x₂ * Real.exp x₂)
  let tangent_slope_A := -Real.exp x₁
  let tangent_slope_B := Real.exp x₂
  tangent_slope_A * tangent_slope_B = -1 →
  let AM := Real.sqrt ((x₁ - 0)^2 + (f x₁ - M.2)^2)
  let BN := Real.sqrt ((x₂ - 0)^2 + (f x₂ - N.2)^2)
  0 < AM / BN ∧ AM / BN < 1 :=
by sorry

end NUMINAMATH_CALUDE_tangent_ratio_bounds_l1340_134001


namespace NUMINAMATH_CALUDE_profit_percentage_invariant_l1340_134066

/-- Represents the profit percentage as a real number between 0 and 1 -/
def ProfitPercentage := { p : ℝ // 0 ≤ p ∧ p ≤ 1 }

/-- Represents the discount percentage as a real number between 0 and 1 -/
def DiscountPercentage := { d : ℝ // 0 ≤ d ∧ d ≤ 1 }

/-- The profit percentage remains the same regardless of the discount -/
theorem profit_percentage_invariant (profit_with_discount : ProfitPercentage) 
  (discount : DiscountPercentage) :
  ∃ (profit_without_discount : ProfitPercentage), 
  profit_without_discount = profit_with_discount :=
sorry

end NUMINAMATH_CALUDE_profit_percentage_invariant_l1340_134066


namespace NUMINAMATH_CALUDE_no_real_roots_for_equation_l1340_134063

theorem no_real_roots_for_equation : ¬∃ x : ℝ, x + Real.sqrt (2*x - 5) = 5 := by
  sorry

end NUMINAMATH_CALUDE_no_real_roots_for_equation_l1340_134063


namespace NUMINAMATH_CALUDE_four_digit_cube_square_sum_multiple_of_seven_l1340_134028

theorem four_digit_cube_square_sum_multiple_of_seven :
  ∃ (x y : ℕ), 
    1000 ≤ x ∧ x < 10000 ∧ 
    7 ∣ x ∧
    x = (y^3 + y^2) / 7 ∧
    (x = 1386 ∨ x = 1200) :=
sorry

end NUMINAMATH_CALUDE_four_digit_cube_square_sum_multiple_of_seven_l1340_134028


namespace NUMINAMATH_CALUDE_x_plus_p_in_terms_of_p_l1340_134054

theorem x_plus_p_in_terms_of_p (x p : ℝ) (h1 : |x - 3| = p) (h2 : x > 3) : x + p = 2*p + 3 := by
  sorry

end NUMINAMATH_CALUDE_x_plus_p_in_terms_of_p_l1340_134054


namespace NUMINAMATH_CALUDE_rice_distribution_l1340_134038

theorem rice_distribution (R : ℚ) : 
  (7/10 : ℚ) * R - (3/10 : ℚ) * R = 20 → R = 50 := by
sorry

end NUMINAMATH_CALUDE_rice_distribution_l1340_134038


namespace NUMINAMATH_CALUDE_back_seat_capacity_is_twelve_l1340_134049

/-- Represents the seating arrangement and capacity of a bus -/
structure BusSeating where
  left_seats : ℕ
  right_seats : ℕ
  people_per_seat : ℕ
  total_capacity : ℕ

/-- Calculates the number of people who can sit in the back seat of the bus -/
def back_seat_capacity (bus : BusSeating) : ℕ :=
  bus.total_capacity - (bus.left_seats + bus.right_seats) * bus.people_per_seat

/-- Theorem stating the number of people who can sit in the back seat -/
theorem back_seat_capacity_is_twelve :
  ∀ (bus : BusSeating),
    bus.left_seats = 15 →
    bus.right_seats = bus.left_seats - 3 →
    bus.people_per_seat = 3 →
    bus.total_capacity = 93 →
    back_seat_capacity bus = 12 := by
  sorry


end NUMINAMATH_CALUDE_back_seat_capacity_is_twelve_l1340_134049


namespace NUMINAMATH_CALUDE_scarlet_savings_l1340_134078

/-- The amount of money Scarlet saved initially -/
def initial_savings : ℕ := sorry

/-- The cost of the earrings Scarlet bought -/
def earrings_cost : ℕ := 23

/-- The cost of the necklace Scarlet bought -/
def necklace_cost : ℕ := 48

/-- The amount of money Scarlet has left -/
def money_left : ℕ := 9

/-- Theorem stating that Scarlet's initial savings equals the sum of her purchases and remaining money -/
theorem scarlet_savings : initial_savings = earrings_cost + necklace_cost + money_left :=
by sorry

end NUMINAMATH_CALUDE_scarlet_savings_l1340_134078


namespace NUMINAMATH_CALUDE_angle_KJG_measure_l1340_134012

-- Define the geometric configuration
structure GeometricConfig where
  -- JKL is a 45-45-90 right triangle
  JKL_is_45_45_90 : Bool
  -- GHIJ is a square
  GHIJ_is_square : Bool
  -- JKLK is a square
  JKLK_is_square : Bool

-- Define the theorem
theorem angle_KJG_measure (config : GeometricConfig) 
  (h1 : config.JKL_is_45_45_90 = true)
  (h2 : config.GHIJ_is_square = true)
  (h3 : config.JKLK_is_square = true) :
  ∃ (angle_KJG : ℝ), angle_KJG = 135 := by
  sorry


end NUMINAMATH_CALUDE_angle_KJG_measure_l1340_134012


namespace NUMINAMATH_CALUDE_factorial_divides_theorem_l1340_134072

def divides (a b : ℤ) : Prop := ∃ k : ℤ, b = a * k

theorem factorial_divides_theorem (a : ℤ) :
  (∃ S : Set ℕ, Set.Infinite S ∧ ∀ n ∈ S, divides (n.factorial + a) ((2*n).factorial)) →
  a = 0 := by sorry

end NUMINAMATH_CALUDE_factorial_divides_theorem_l1340_134072


namespace NUMINAMATH_CALUDE_other_sides_equations_l1340_134005

/-- An isosceles right triangle with one leg on the line 2x - y = 0 and hypotenuse midpoint (4, 2) -/
structure IsoscelesRightTriangle where
  /-- The line containing one leg of the triangle -/
  leg_line : Set (ℝ × ℝ)
  /-- The midpoint of the hypotenuse -/
  hypotenuse_midpoint : ℝ × ℝ
  /-- The triangle is isosceles and right-angled -/
  is_isosceles_right : Bool
  /-- The leg line equation is 2x - y = 0 -/
  leg_line_eq : leg_line = {(x, y) : ℝ × ℝ | 2 * x - y = 0}
  /-- The hypotenuse midpoint is (4, 2) -/
  midpoint_coords : hypotenuse_midpoint = (4, 2)

/-- The theorem stating the equations of the other two sides -/
theorem other_sides_equations (t : IsoscelesRightTriangle) :
  ∃ (side1 side2 : Set (ℝ × ℝ)),
    side1 = {(x, y) : ℝ × ℝ | x + 2 * y - 2 = 0} ∧
    side2 = {(x, y) : ℝ × ℝ | x + 2 * y - 14 = 0} :=
  sorry

end NUMINAMATH_CALUDE_other_sides_equations_l1340_134005


namespace NUMINAMATH_CALUDE_correct_seating_arrangements_l1340_134010

/-- The number of seats in the row -/
def num_seats : ℕ := 8

/-- The number of people to be seated -/
def num_people : ℕ := 3

/-- A function that calculates the number of valid seating arrangements -/
def seating_arrangements (seats : ℕ) (people : ℕ) : ℕ :=
  sorry  -- The actual implementation would go here

/-- Theorem stating that the number of seating arrangements is 24 -/
theorem correct_seating_arrangements :
  seating_arrangements num_seats num_people = 24 := by sorry


end NUMINAMATH_CALUDE_correct_seating_arrangements_l1340_134010


namespace NUMINAMATH_CALUDE_peanuts_lost_l1340_134058

def initial_peanuts : ℕ := 74
def final_peanuts : ℕ := 15

theorem peanuts_lost : initial_peanuts - final_peanuts = 59 := by
  sorry

end NUMINAMATH_CALUDE_peanuts_lost_l1340_134058


namespace NUMINAMATH_CALUDE_modified_cube_edges_l1340_134068

/-- A cube with equilateral triangular pyramids extended from its edge midpoints. -/
structure ModifiedCube where
  /-- The number of edges in the original cube. -/
  cube_edges : ℕ
  /-- The number of vertices in the original cube. -/
  cube_vertices : ℕ
  /-- The number of new edges added by each pyramid (excluding the base). -/
  pyramid_new_edges : ℕ
  /-- The number of new edges added to each original cube edge. -/
  new_edges_per_cube_edge : ℕ

/-- The total number of edges in the modified cube. -/
def total_edges (c : ModifiedCube) : ℕ :=
  c.cube_edges + 
  (c.cube_edges * c.new_edges_per_cube_edge) + 
  c.cube_edges

theorem modified_cube_edges :
  ∀ (c : ModifiedCube),
  c.cube_edges = 12 →
  c.cube_vertices = 8 →
  c.pyramid_new_edges = 4 →
  c.new_edges_per_cube_edge = 2 →
  total_edges c = 48 := by
  sorry

end NUMINAMATH_CALUDE_modified_cube_edges_l1340_134068


namespace NUMINAMATH_CALUDE_parabola_focus_distance_l1340_134095

/-- Theorem: For a parabola y² = 2px where p > 0, if the distance from its focus 
    to the line y = x + 1 is √2, then p = 2. -/
theorem parabola_focus_distance (p : ℝ) : 
  p > 0 → 
  (let focus : ℝ × ℝ := (p/2, 0)
   let distance_to_line (x y : ℝ) := |(-1:ℝ)*x + y - 1| / Real.sqrt 2
   distance_to_line (p/2) 0 = Real.sqrt 2) → 
  p = 2 := by
sorry

end NUMINAMATH_CALUDE_parabola_focus_distance_l1340_134095


namespace NUMINAMATH_CALUDE_sum_of_solutions_l1340_134034

theorem sum_of_solutions (y : ℝ) : (∃ y₁ y₂ : ℝ, y₁ + 16 / y₁ = 12 ∧ y₂ + 16 / y₂ = 12 ∧ y₁ ≠ y₂ ∧ y₁ + y₂ = 12) := by
  sorry

end NUMINAMATH_CALUDE_sum_of_solutions_l1340_134034


namespace NUMINAMATH_CALUDE_a_range_l1340_134099

def p (a : ℝ) : Prop := ∃ (x y : ℝ), x^2 + y^2 - a*x + y + 1 = 0

def q (a : ℝ) : Prop := ∃ (x y : ℝ), 2*a*x + (1-a)*y + 1 = 0 ∧ (2*a)/(a-1) > 1

def range_of_a (a : ℝ) : Prop := a ∈ Set.Icc (-Real.sqrt 3) (-1) ∪ Set.Ioc 1 (Real.sqrt 3)

theorem a_range (a : ℝ) :
  (∀ a, p a ∨ q a) ∧ (∀ a, ¬(p a ∧ q a)) →
  range_of_a a :=
sorry

end NUMINAMATH_CALUDE_a_range_l1340_134099


namespace NUMINAMATH_CALUDE_quadratic_points_relationship_l1340_134076

/-- A quadratic function f(x) = -x² + 2x + 3 -/
def f (x : ℝ) : ℝ := -x^2 + 2*x + 3

/-- Point P₁ on the graph of f -/
def P₁ : ℝ × ℝ := (-1, f (-1))

/-- Point P₂ on the graph of f -/
def P₂ : ℝ × ℝ := (3, f 3)

/-- Point P₃ on the graph of f -/
def P₃ : ℝ × ℝ := (5, f 5)

theorem quadratic_points_relationship :
  P₁.2 = P₂.2 ∧ P₂.2 > P₃.2 := by sorry

end NUMINAMATH_CALUDE_quadratic_points_relationship_l1340_134076


namespace NUMINAMATH_CALUDE_geometric_sequence_common_ratio_l1340_134016

/-- A geometric sequence with the given properties has a common ratio of 1/2 -/
theorem geometric_sequence_common_ratio 
  (a : ℕ → ℝ) 
  (h_geometric : ∀ n : ℕ, a (n + 1) = a n * (a 2 / a 1)) 
  (h_sum_1_3 : a 1 + a 3 = 10) 
  (h_sum_4_6 : a 4 + a 6 = 5/4) : 
  a 2 / a 1 = 1/2 := by
sorry

end NUMINAMATH_CALUDE_geometric_sequence_common_ratio_l1340_134016


namespace NUMINAMATH_CALUDE_set_operations_and_subsets_l1340_134096

def U : Finset ℕ := {4, 5, 6, 7, 8, 9, 10, 11, 12}
def A : Finset ℕ := {6, 8, 10, 12}
def B : Finset ℕ := {1, 6, 8}

theorem set_operations_and_subsets :
  (A ∪ B = {1, 6, 8, 10, 12}) ∧
  (U \ A = {4, 5, 7, 9, 11}) ∧
  (Finset.powerset (A ∩ B)).card = 4 := by sorry

end NUMINAMATH_CALUDE_set_operations_and_subsets_l1340_134096


namespace NUMINAMATH_CALUDE_max_value_of_expression_l1340_134006

theorem max_value_of_expression :
  (∀ x : ℝ, |x - 1| - |x + 4| - 5 ≤ 0) ∧
  (∃ x : ℝ, |x - 1| - |x + 4| - 5 = 0) :=
by sorry

end NUMINAMATH_CALUDE_max_value_of_expression_l1340_134006


namespace NUMINAMATH_CALUDE_factorization_proof_l1340_134051

theorem factorization_proof (z : ℝ) :
  45 * z^12 + 180 * z^24 = 45 * z^12 * (1 + 4 * z^12) := by
  sorry

end NUMINAMATH_CALUDE_factorization_proof_l1340_134051


namespace NUMINAMATH_CALUDE_zeros_of_f_l1340_134044

def f (x : ℝ) : ℝ := x * (x^2 - 16)

theorem zeros_of_f :
  ∀ x : ℝ, f x = 0 ↔ x = -4 ∨ x = 0 ∨ x = 4 := by
  sorry

end NUMINAMATH_CALUDE_zeros_of_f_l1340_134044


namespace NUMINAMATH_CALUDE_race_distance_l1340_134048

theorem race_distance (a_time b_time : ℝ) (beat_distance : ℝ) : 
  a_time = 28 → b_time = 32 → beat_distance = 16 → 
  ∃ d : ℝ, d = 128 ∧ d / a_time * b_time = d - beat_distance :=
by
  sorry

end NUMINAMATH_CALUDE_race_distance_l1340_134048


namespace NUMINAMATH_CALUDE_max_value_of_x_minus_x_squared_l1340_134056

theorem max_value_of_x_minus_x_squared (x : ℝ) :
  0 < x → x < 1 → ∃ (y : ℝ), y = 1/2 ∧ ∀ z, 0 < z → z < 1 → x * (1 - x) ≤ y * (1 - y) :=
by sorry

end NUMINAMATH_CALUDE_max_value_of_x_minus_x_squared_l1340_134056


namespace NUMINAMATH_CALUDE_right_triangle_hypotenuse_l1340_134045

theorem right_triangle_hypotenuse (PQ PR PS SQ PT TR QT SR : ℝ) :
  PS / SQ = 1 / 3 →
  PT / TR = 1 / 3 →
  QT = 20 →
  SR = 36 →
  PQ^2 + PR^2 = 1085.44 :=
by sorry

end NUMINAMATH_CALUDE_right_triangle_hypotenuse_l1340_134045


namespace NUMINAMATH_CALUDE_triangle_abc_properties_l1340_134088

/-- Triangle ABC with sides a, b, c opposite to angles A, B, C respectively -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

/-- The theorem statement -/
theorem triangle_abc_properties (t : Triangle) 
  (h1 : t.a = t.b * Real.cos t.C + (Real.sqrt 3 / 3) * t.c * Real.sin t.B)
  (h2 : t.a + t.c = 6)
  (h3 : (1/2) * t.a * t.c * Real.sin t.B = 3 * Real.sqrt 3 / 2) : 
  t.B = π/3 ∧ t.b = 3 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_triangle_abc_properties_l1340_134088


namespace NUMINAMATH_CALUDE_sqrt_expression_simplification_l1340_134094

theorem sqrt_expression_simplification :
  Real.sqrt 27 / (Real.sqrt 3 / 2) * (2 * Real.sqrt 2) - 6 * Real.sqrt 2 = 6 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_expression_simplification_l1340_134094

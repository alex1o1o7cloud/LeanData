import Mathlib

namespace NUMINAMATH_CALUDE_polar_circle_equation_l2333_233308

/-- A circle in a polar coordinate system with radius 1 and center at (1, 0) -/
structure PolarCircle where
  center : ℝ × ℝ := (1, 0)
  radius : ℝ := 1

/-- A point in polar coordinates -/
structure PolarPoint where
  ρ : ℝ
  θ : ℝ

/-- Predicate to check if a point is on the circle -/
def IsOnCircle (c : PolarCircle) (p : PolarPoint) : Prop :=
  p.ρ = 2 * c.radius * Real.cos p.θ

theorem polar_circle_equation (c : PolarCircle) (p : PolarPoint) 
  (h : IsOnCircle c p) : p.ρ = 2 * Real.cos p.θ := by
  sorry

end NUMINAMATH_CALUDE_polar_circle_equation_l2333_233308


namespace NUMINAMATH_CALUDE_intersection_of_A_and_B_l2333_233334

def A : Set ℝ := {x | |x| ≤ 2}
def B : Set ℝ := {x | x ≤ 1}

theorem intersection_of_A_and_B : A ∩ B = {x : ℝ | -2 ≤ x ∧ x ≤ 1} := by sorry

end NUMINAMATH_CALUDE_intersection_of_A_and_B_l2333_233334


namespace NUMINAMATH_CALUDE_nested_cubes_properties_l2333_233396

/-- Represents a cube with an inscribed sphere, which contains another inscribed cube. -/
structure NestedCubes where
  outer_surface_area : ℝ
  outer_side_length : ℝ
  sphere_diameter : ℝ
  inner_side_length : ℝ

/-- The surface area of a cube given its side length. -/
def cube_surface_area (side_length : ℝ) : ℝ := 6 * side_length^2

/-- The volume of a cube given its side length. -/
def cube_volume (side_length : ℝ) : ℝ := side_length^3

/-- Theorem stating the properties of the nested cubes structure. -/
theorem nested_cubes_properties (nc : NestedCubes) 
  (h1 : nc.outer_surface_area = 54)
  (h2 : nc.outer_side_length^2 = 54 / 6)
  (h3 : nc.sphere_diameter = nc.outer_side_length)
  (h4 : nc.inner_side_length * Real.sqrt 3 = nc.sphere_diameter) :
  cube_surface_area nc.inner_side_length = 18 ∧ 
  cube_volume nc.inner_side_length = 3 * Real.sqrt 3 := by
  sorry

#check nested_cubes_properties

end NUMINAMATH_CALUDE_nested_cubes_properties_l2333_233396


namespace NUMINAMATH_CALUDE_unique_root_condition_l2333_233385

/-- The equation √(ax² + ax + 2) = ax + 2 has a unique real root if and only if a = -8 or a ≥ 1 -/
theorem unique_root_condition (a : ℝ) : 
  (∃! x : ℝ, Real.sqrt (a * x^2 + a * x + 2) = a * x + 2) ↔ (a = -8 ∨ a ≥ 1) :=
by sorry

end NUMINAMATH_CALUDE_unique_root_condition_l2333_233385


namespace NUMINAMATH_CALUDE_stream_speed_is_one_l2333_233362

/-- Represents the speed of a boat in still water and the speed of a stream. -/
structure BoatProblem where
  boat_speed : ℝ
  stream_speed : ℝ

/-- Given the conditions of the problem, proves that the stream speed is 1 km/h. -/
theorem stream_speed_is_one
  (bp : BoatProblem)
  (h1 : bp.boat_speed + bp.stream_speed = 100 / 10)
  (h2 : bp.boat_speed - bp.stream_speed = 200 / 25) :
  bp.stream_speed = 1 := by
  sorry

end NUMINAMATH_CALUDE_stream_speed_is_one_l2333_233362


namespace NUMINAMATH_CALUDE_remainder_3_pow_2040_mod_11_l2333_233314

theorem remainder_3_pow_2040_mod_11 : 3^2040 % 11 = 1 := by
  sorry

end NUMINAMATH_CALUDE_remainder_3_pow_2040_mod_11_l2333_233314


namespace NUMINAMATH_CALUDE_vector_expression_l2333_233394

/-- Given vectors a, b, and c in ℝ², prove that c = a - 2b --/
theorem vector_expression (a b c : ℝ × ℝ) :
  a = (3, -2) → b = (-2, 1) → c = (7, -4) → c = a - 2 • b := by
  sorry

end NUMINAMATH_CALUDE_vector_expression_l2333_233394


namespace NUMINAMATH_CALUDE_linear_equation_solution_l2333_233357

theorem linear_equation_solution (x y : ℝ) : 3 * x + y = 1 → y = -3 * x + 1 := by
  sorry

end NUMINAMATH_CALUDE_linear_equation_solution_l2333_233357


namespace NUMINAMATH_CALUDE_subcommittees_count_l2333_233323

def total_members : ℕ := 12
def teacher_count : ℕ := 5
def subcommittee_size : ℕ := 4

def subcommittees_with_teacher : ℕ := Nat.choose total_members subcommittee_size - Nat.choose (total_members - teacher_count) subcommittee_size

theorem subcommittees_count : subcommittees_with_teacher = 460 := by
  sorry

end NUMINAMATH_CALUDE_subcommittees_count_l2333_233323


namespace NUMINAMATH_CALUDE_f_monotonicity_and_zeros_l2333_233304

def f (k : ℝ) (x : ℝ) : ℝ := x^3 - k*x + k^2

theorem f_monotonicity_and_zeros (k : ℝ) :
  (∀ x y, x < y → k ≤ 0 → f k x < f k y) ∧
  (k > 0 → ∀ x y, (x < y ∧ y < -Real.sqrt (k/3)) ∨ (x < y ∧ x > Real.sqrt (k/3)) → f k x < f k y) ∧
  (k > 0 → ∀ x y, -Real.sqrt (k/3) < x ∧ x < y ∧ y < Real.sqrt (k/3) → f k x > f k y) ∧
  (∃ x y z, x < y ∧ y < z ∧ f k x = 0 ∧ f k y = 0 ∧ f k z = 0 ↔ 0 < k ∧ k < 4/27) :=
by sorry

end NUMINAMATH_CALUDE_f_monotonicity_and_zeros_l2333_233304


namespace NUMINAMATH_CALUDE_sphere_always_circular_cross_section_l2333_233325

-- Define the basic geometric shapes
inductive GeometricShape
  | Cone
  | Sphere
  | Cylinder
  | Prism

-- Define a predicate for having a circular cross section
def hasCircularCrossSection (shape : GeometricShape) : Prop :=
  match shape with
  | GeometricShape.Sphere => True
  | _ => False

-- Theorem statement
theorem sphere_always_circular_cross_section :
  ∀ (shape : GeometricShape),
    hasCircularCrossSection shape ↔ shape = GeometricShape.Sphere := by
  sorry

end NUMINAMATH_CALUDE_sphere_always_circular_cross_section_l2333_233325


namespace NUMINAMATH_CALUDE_initial_speed_proof_l2333_233384

/-- The acceleration due to gravity in m/s² -/
def g : ℝ := 10

/-- The height of the building in meters -/
def h : ℝ := 180

/-- The time taken to fall the last 60 meters in seconds -/
def t : ℝ := 1

/-- The distance fallen in the last second in meters -/
def d : ℝ := 60

/-- The initial downward speed of the object in m/s -/
def v₀ : ℝ := 25

theorem initial_speed_proof : 
  ∃ (v : ℝ), v = v₀ ∧ 
  d = (v + v₀) / 2 * t ∧ 
  v^2 = v₀^2 + 2 * g * (h - d) :=
sorry

end NUMINAMATH_CALUDE_initial_speed_proof_l2333_233384


namespace NUMINAMATH_CALUDE_sum_of_four_numbers_l2333_233305

theorem sum_of_four_numbers : 1432 + 3214 + 2143 + 4321 = 11110 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_four_numbers_l2333_233305


namespace NUMINAMATH_CALUDE_triangular_prism_surface_area_l2333_233342

/-- The surface area of a triangular-based prism created by vertically cutting a rectangular prism -/
theorem triangular_prism_surface_area (l w h : ℝ) (h_l : l = 3) (h_w : w = 5) (h_h : h = 12) :
  let front_area := l * h
  let side_area := l * w
  let triangle_area := w * h / 2
  let back_diagonal := Real.sqrt (w^2 + h^2)
  let back_area := l * back_diagonal
  front_area + side_area + 2 * triangle_area + back_area = 150 :=
sorry

end NUMINAMATH_CALUDE_triangular_prism_surface_area_l2333_233342


namespace NUMINAMATH_CALUDE_g_invertible_interval_largest_invertible_interval_l2333_233318

-- Define the function g(x)
def g (x : ℝ) : ℝ := 3 * x^2 - 9 * x + 4

-- Define the interval [3/2, ∞)
def interval : Set ℝ := {x | x ≥ 3/2}

theorem g_invertible_interval :
  ∀ (a b : ℝ), a ≥ 3/2 ∧ b ≥ 3/2 ∧ a ≠ b → g a ≠ g b :=
by sorry

theorem largest_invertible_interval :
  ∀ (I : Set ℝ), 2 ∈ I ∧ (∀ (x y : ℝ), x ∈ I ∧ y ∈ I ∧ x ≠ y → g x ≠ g y) →
  I ⊆ interval :=
by sorry

#check g_invertible_interval
#check largest_invertible_interval

end NUMINAMATH_CALUDE_g_invertible_interval_largest_invertible_interval_l2333_233318


namespace NUMINAMATH_CALUDE_arithmetic_equations_l2333_233321

theorem arithmetic_equations : 
  (12 * 12 / (12 + 12) = 6) ∧ ((12 * 12 + 12) / 12 = 13) := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_equations_l2333_233321


namespace NUMINAMATH_CALUDE_man_work_time_l2333_233315

theorem man_work_time (total_work : ℝ) (man_rate : ℝ) (son_rate : ℝ) 
  (h1 : man_rate + son_rate = total_work / 3)
  (h2 : son_rate = total_work / 5.25) :
  man_rate = total_work / 7 := by
sorry

end NUMINAMATH_CALUDE_man_work_time_l2333_233315


namespace NUMINAMATH_CALUDE_quadratic_inequality_l2333_233367

theorem quadratic_inequality (x : ℝ) : x^2 - 6*x > 15 ↔ x < -1.5 ∨ x > 7.5 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_inequality_l2333_233367


namespace NUMINAMATH_CALUDE_builder_problem_l2333_233386

/-- Calculate the minimum number of packs needed given the total items and items per pack -/
def minPacks (total : ℕ) (perPack : ℕ) : ℕ :=
  (total + perPack - 1) / perPack

/-- The problem statement -/
theorem builder_problem :
  let totalBrackets := 42
  let bracketsPerPack := 5
  minPacks totalBrackets bracketsPerPack = 9 := by
  sorry

end NUMINAMATH_CALUDE_builder_problem_l2333_233386


namespace NUMINAMATH_CALUDE_congruence_system_solution_l2333_233320

theorem congruence_system_solution (x : ℤ) :
  (9 * x + 3) % 15 = 6 →
  x % 5 = 2 →
  x % 5 = 2 := by
sorry

end NUMINAMATH_CALUDE_congruence_system_solution_l2333_233320


namespace NUMINAMATH_CALUDE_triangle_tan_c_l2333_233387

/-- Given a triangle ABC with sides a, b, c opposite to angles A, B, C respectively,
    if the area S satisfies 2S = (a + b)² - c², then tan C = -4/3 -/
theorem triangle_tan_c (a b c : ℝ) (h : a > 0 ∧ b > 0 ∧ c > 0) :
  let S := (1 / 2) * a * b * Real.sin C
  2 * S = (a + b)^2 - c^2 →
  Real.tan C = -4/3 :=
by sorry

end NUMINAMATH_CALUDE_triangle_tan_c_l2333_233387


namespace NUMINAMATH_CALUDE_f_value_at_2_l2333_233302

/-- A function f(x) = x^5 + ax^3 + bx - 8 where f(-2) = 10 -/
def f (a b : ℝ) (x : ℝ) : ℝ := x^5 + a*x^3 + b*x - 8

/-- Theorem: If f(-2) = 10, then f(2) = -26 -/
theorem f_value_at_2 (a b : ℝ) (h : f a b (-2) = 10) : f a b 2 = -26 := by
  sorry

end NUMINAMATH_CALUDE_f_value_at_2_l2333_233302


namespace NUMINAMATH_CALUDE_earliest_retirement_year_l2333_233370

/-- Represents the retirement eligibility rule -/
def rule_of_70 (age : ℕ) (years_employed : ℕ) : Prop :=
  age + years_employed ≥ 70

/-- Represents the employee's age in a given year -/
def age_in_year (hire_year : ℕ) (hire_age : ℕ) (current_year : ℕ) : ℕ :=
  hire_age + (current_year - hire_year)

/-- Represents the employee's years of employment in a given year -/
def years_employed (hire_year : ℕ) (current_year : ℕ) : ℕ :=
  current_year - hire_year

/-- Theorem stating the earliest retirement year for the employee -/
theorem earliest_retirement_year 
  (hire_year : ℕ) 
  (hire_age : ℕ) 
  (retirement_year : ℕ) :
  hire_year = 1987 →
  hire_age = 32 →
  retirement_year = 2006 →
  (∀ y : ℕ, y < retirement_year → 
    ¬(rule_of_70 (age_in_year hire_year hire_age y) (years_employed hire_year y))) →
  rule_of_70 (age_in_year hire_year hire_age retirement_year) (years_employed hire_year retirement_year) :=
by
  sorry


end NUMINAMATH_CALUDE_earliest_retirement_year_l2333_233370


namespace NUMINAMATH_CALUDE_monomial_sum_implies_mn_value_l2333_233379

theorem monomial_sum_implies_mn_value 
  (m n : ℤ) 
  (h : ∃ (a : ℚ), 3 * X^(m+6) * Y^(2*n+1) + X * Y^7 = a * X^(m+6) * Y^(2*n+1)) : 
  m * n = -15 := by
  sorry

end NUMINAMATH_CALUDE_monomial_sum_implies_mn_value_l2333_233379


namespace NUMINAMATH_CALUDE_common_chord_equation_length_AB_l2333_233398

-- Define the circles C and M
def C (x y : ℝ) : Prop := x^2 + y^2 - 10*x - 10*y = 0
def M (x y : ℝ) : Prop := x^2 + y^2 + 6*x + 2*y - 40 = 0

-- Define the intersection points A and B
def A : ℝ × ℝ := (-2, 6)
def B : ℝ × ℝ := (4, -2)

-- Theorem for the equation of the common chord
theorem common_chord_equation : 
  ∀ (x y : ℝ), C x y ∧ M x y → 4*x + 2*y - 10 = 0 :=
sorry

-- Theorem for the length of AB
theorem length_AB : 
  Real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2) = 10 :=
sorry

end NUMINAMATH_CALUDE_common_chord_equation_length_AB_l2333_233398


namespace NUMINAMATH_CALUDE_complement_of_A_in_U_l2333_233395

-- Define the universal set U as the real numbers
def U : Set ℝ := Set.univ

-- Define set A
def A : Set ℝ := {x | x^2 - 2*x - 3 ≤ 0}

-- Define the complement of A in U
def complementA : Set ℝ := {x | x < -1 ∨ x > 3}

-- Theorem statement
theorem complement_of_A_in_U : Set.compl A = complementA := by sorry

end NUMINAMATH_CALUDE_complement_of_A_in_U_l2333_233395


namespace NUMINAMATH_CALUDE_trigonometric_expressions_equal_half_l2333_233381

theorem trigonometric_expressions_equal_half :
  let expr1 := Real.sin (15 * π / 180) * Real.cos (15 * π / 180)
  let expr2 := Real.cos (π / 8)^2 - Real.sin (π / 8)^2
  let expr3 := Real.tan (22.5 * π / 180) / (1 - Real.tan (22.5 * π / 180)^2)
  (expr1 ≠ 1/2 ∧ expr2 ≠ 1/2 ∧ expr3 = 1/2) :=
by sorry

end NUMINAMATH_CALUDE_trigonometric_expressions_equal_half_l2333_233381


namespace NUMINAMATH_CALUDE_log_power_difference_l2333_233390

theorem log_power_difference (x : ℝ) (h1 : x < 1) 
  (h2 : (Real.log x / Real.log 10)^2 - Real.log (x^2) / Real.log 10 = 48) :
  (Real.log x / Real.log 10)^5 - Real.log (x^5) / Real.log 10 = -7746 := by
  sorry

end NUMINAMATH_CALUDE_log_power_difference_l2333_233390


namespace NUMINAMATH_CALUDE_elberta_has_45_dollars_l2333_233373

-- Define the amounts for each person
def granny_smith_amount : ℕ := 100
def anjou_amount : ℕ := (2 * granny_smith_amount) / 5
def elberta_amount : ℕ := anjou_amount + 5

-- Theorem to prove
theorem elberta_has_45_dollars : elberta_amount = 45 := by
  sorry

end NUMINAMATH_CALUDE_elberta_has_45_dollars_l2333_233373


namespace NUMINAMATH_CALUDE_triangle_area_l2333_233372

theorem triangle_area (p A B : Real) (h_positive : p > 0) (h_angles : 0 < A ∧ 0 < B ∧ A + B < π) : 
  let C := π - A - B
  let S := (2 * p^2 * Real.sin A * Real.sin B * Real.sin C) / (Real.sin A + Real.sin B + Real.sin C)^2
  S > 0 ∧ S < p^2 := by sorry

end NUMINAMATH_CALUDE_triangle_area_l2333_233372


namespace NUMINAMATH_CALUDE_min_colors_shapes_for_distribution_centers_l2333_233348

theorem min_colors_shapes_for_distribution_centers :
  ∃ (C S : ℕ),
    (C = 3 ∧ S = 3) ∧
    (∀ (C' S' : ℕ),
      C' + C' * (C' - 1) / 2 + S' + S' * (S' - 1) ≥ 12 →
      C' ≥ C ∧ S' ≥ S) ∧
    C + C * (C - 1) / 2 + S + S * (S - 1) ≥ 12 :=
by sorry

end NUMINAMATH_CALUDE_min_colors_shapes_for_distribution_centers_l2333_233348


namespace NUMINAMATH_CALUDE_sufficient_not_necessary_condition_l2333_233356

theorem sufficient_not_necessary_condition :
  (∀ x : ℝ, x = -2 → x^2 = 4) ∧
  ¬(∀ x : ℝ, x^2 = 4 → x = -2) :=
by sorry

end NUMINAMATH_CALUDE_sufficient_not_necessary_condition_l2333_233356


namespace NUMINAMATH_CALUDE_perimeter_of_square_b_l2333_233300

/-- Given a square A with perimeter 40 cm and a square B with area equal to one-third the area of square A, 
    the perimeter of square B is (40√3)/3 cm. -/
theorem perimeter_of_square_b (square_a square_b : Real → Real → Prop) : 
  (∃ side_a, square_a side_a side_a ∧ 4 * side_a = 40) →
  (∃ side_b, square_b side_b side_b ∧ side_b^2 = (side_a^2) / 3) →
  (∃ perimeter_b, perimeter_b = 40 * Real.sqrt 3 / 3) :=
by sorry

end NUMINAMATH_CALUDE_perimeter_of_square_b_l2333_233300


namespace NUMINAMATH_CALUDE_solution_set_of_inequality_l2333_233316

theorem solution_set_of_inequality (x : ℝ) :
  (|2 * x^2 - 1| ≤ 1) ↔ (-1 ≤ x ∧ x ≤ 1) := by
  sorry

end NUMINAMATH_CALUDE_solution_set_of_inequality_l2333_233316


namespace NUMINAMATH_CALUDE_complex_fraction_simplification_l2333_233377

theorem complex_fraction_simplification :
  let z : ℂ := (3 - I) / (1 - I)
  z = 2 + I :=
by
  sorry

end NUMINAMATH_CALUDE_complex_fraction_simplification_l2333_233377


namespace NUMINAMATH_CALUDE_quadratic_completion_of_square_l2333_233399

theorem quadratic_completion_of_square (x : ℝ) : 
  2 * x^2 - 4 * x - 3 = 0 ↔ (x - 1)^2 = 5/2 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_completion_of_square_l2333_233399


namespace NUMINAMATH_CALUDE_diophantine_equation_solutions_l2333_233388

theorem diophantine_equation_solutions (a b c : ℤ) :
  (1 : ℚ) / a + (1 : ℚ) / b + (1 : ℚ) / c = 1 ↔
  ((a = 3 ∧ b = 3 ∧ c = 3) ∨
   (a = 2 ∧ b = 3 ∧ c = 6) ∨
   (a = 2 ∧ b = 4 ∧ c = 4) ∨
   (∃ t : ℤ, a = 1 ∧ b = t ∧ c = -t)) :=
by sorry

end NUMINAMATH_CALUDE_diophantine_equation_solutions_l2333_233388


namespace NUMINAMATH_CALUDE_tangent_range_l2333_233374

/-- The circle C in the Cartesian coordinate system -/
def Circle (x y : ℝ) : Prop := x^2 + y^2 - 4*x = 0

/-- The line equation -/
def Line (k x y : ℝ) : Prop := y = k*(x + 1)

/-- Point P is on the line -/
def PointOnLine (P : ℝ × ℝ) (k : ℝ) : Prop :=
  Line k P.1 P.2

/-- Two tangents from P to the circle are perpendicular -/
def PerpendicularTangents (P : ℝ × ℝ) : Prop :=
  ∃ A B : ℝ × ℝ, Circle A.1 A.2 ∧ Circle B.1 B.2 ∧
    (A.1 - P.1) * (B.1 - P.1) + (A.2 - P.2) * (B.2 - P.2) = 0

/-- The main theorem -/
theorem tangent_range (k : ℝ) :
  (∃ P : ℝ × ℝ, PointOnLine P k ∧ PerpendicularTangents P) →
  k ∈ Set.Icc (-2 * Real.sqrt 2) (2 * Real.sqrt 2) :=
sorry

end NUMINAMATH_CALUDE_tangent_range_l2333_233374


namespace NUMINAMATH_CALUDE_five_ounce_letter_cost_l2333_233368

/-- Postage fee structure -/
structure PostageFee where
  baseRate : ℚ  -- Base rate in dollars
  additionalRate : ℚ  -- Additional rate per ounce in dollars
  handlingFee : ℚ  -- Handling fee in dollars
  handlingFeeThreshold : ℕ  -- Threshold in ounces for applying handling fee

/-- Calculate the total postage fee for a given weight -/
def calculatePostageFee (fee : PostageFee) (weight : ℕ) : ℚ :=
  fee.baseRate +
  fee.additionalRate * (weight - 1) +
  if weight > fee.handlingFeeThreshold then fee.handlingFee else 0

/-- Theorem: The cost to send a 5-ounce letter is $1.45 -/
theorem five_ounce_letter_cost :
  let fee : PostageFee := {
    baseRate := 35 / 100,
    additionalRate := 25 / 100,
    handlingFee := 10 / 100,
    handlingFeeThreshold := 2
  }
  calculatePostageFee fee 5 = 145 / 100 := by
  sorry

end NUMINAMATH_CALUDE_five_ounce_letter_cost_l2333_233368


namespace NUMINAMATH_CALUDE_AgOH_formation_l2333_233383

-- Define the chemical reaction
structure Reaction where
  reactant1 : String
  reactant2 : String
  product1 : String
  product2 : String
  ratio : ℚ

-- Define the initial conditions
def initial_AgNO3 : ℚ := 3
def initial_NaOH : ℚ := 3

-- Define the reaction
def silver_hydroxide_reaction : Reaction := {
  reactant1 := "AgNO3"
  reactant2 := "NaOH"
  product1 := "AgOH"
  product2 := "NaNO3"
  ratio := 1
}

-- Theorem statement
theorem AgOH_formation (r : Reaction) (h1 : r = silver_hydroxide_reaction) 
  (h2 : initial_AgNO3 = initial_NaOH) : 
  let moles_AgOH := min initial_AgNO3 initial_NaOH
  moles_AgOH = 3 := by sorry

end NUMINAMATH_CALUDE_AgOH_formation_l2333_233383


namespace NUMINAMATH_CALUDE_decimal_equivalent_of_one_fifth_squared_l2333_233376

theorem decimal_equivalent_of_one_fifth_squared : (1 / 5 : ℚ) ^ 2 = 0.04 := by
  sorry

end NUMINAMATH_CALUDE_decimal_equivalent_of_one_fifth_squared_l2333_233376


namespace NUMINAMATH_CALUDE_quadratic_function_coefficients_l2333_233345

/-- A quadratic function f(x) = x^2 + ax + b satisfying f(f(x) + x) / f(x) = x^2 + 2023x + 1776 
    has coefficients a = 2021 and b = -246. -/
theorem quadratic_function_coefficients (a b : ℝ) : 
  (∀ x, (((x^2 + a*x + b)^2 + a*(x^2 + a*x + b) + b) / (x^2 + a*x + b) = x^2 + 2023*x + 1776)) → 
  (a = 2021 ∧ b = -246) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_function_coefficients_l2333_233345


namespace NUMINAMATH_CALUDE_soda_comparison_l2333_233369

theorem soda_comparison (J : ℝ) (L A : ℝ) 
  (h1 : L = J * 1.5)  -- Liliane has 50% more soda than Jacqueline
  (h2 : A = J * 1.25) -- Alice has 25% more soda than Jacqueline
  : L = A * 1.2       -- Liliane has 20% more soda than Alice
:= by sorry

end NUMINAMATH_CALUDE_soda_comparison_l2333_233369


namespace NUMINAMATH_CALUDE_problem_statement_l2333_233360

theorem problem_statement (x y : ℝ) (h : 1 ≤ x^2 - x*y + y^2 ∧ x^2 - x*y + y^2 ≤ 2) :
  (2/9 ≤ x^4 + y^4 ∧ x^4 + y^4 ≤ 8) ∧
  (∀ n : ℕ, n ≥ 3 → x^(2*n) + y^(2*n) ≥ 2/3^n) := by
  sorry

end NUMINAMATH_CALUDE_problem_statement_l2333_233360


namespace NUMINAMATH_CALUDE_problem_part_1_problem_part_2_l2333_233303

def f (a : ℝ) (x : ℝ) : ℝ := x^2 + a
def g (x : ℝ) : ℝ := |x + 1| + |x - 2|

theorem problem_part_1 :
  {x : ℝ | f (-4) x ≥ g x} = {x : ℝ | x ≤ -1 - Real.sqrt 6 ∨ x ≥ 3} := by sorry

theorem problem_part_2 (a : ℝ) :
  (∀ x ∈ Set.Icc 0 3, f a x ≤ g x) → a ≤ -4 := by sorry

end NUMINAMATH_CALUDE_problem_part_1_problem_part_2_l2333_233303


namespace NUMINAMATH_CALUDE_sum_of_m_and_n_l2333_233365

theorem sum_of_m_and_n (m n : ℝ) (h : |m - 2| + |n - 6| = 0) : m + n = 8 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_m_and_n_l2333_233365


namespace NUMINAMATH_CALUDE_perimeter_area_ratio_not_always_equal_l2333_233327

/-- Represents an isosceles triangle -/
structure IsoscelesTriangle where
  base : ℝ
  height : ℝ
  leg : ℝ
  perimeter : ℝ
  area : ℝ

/-- The theorem states that the ratio of perimeters is not always equal to the ratio of areas for two different isosceles triangles -/
theorem perimeter_area_ratio_not_always_equal
  (triangle1 triangle2 : IsoscelesTriangle)
  (h_base_neq : triangle1.base ≠ triangle2.base)
  (h_leg_neq : triangle1.leg ≠ triangle2.leg) :
  ¬ ∀ (triangle1 triangle2 : IsoscelesTriangle),
    triangle1.perimeter / triangle2.perimeter = triangle1.area / triangle2.area :=
by sorry

end NUMINAMATH_CALUDE_perimeter_area_ratio_not_always_equal_l2333_233327


namespace NUMINAMATH_CALUDE_max_distance_to_line_l2333_233307

/-- Given m ∈ ℝ, prove that the maximum distance from a point P(x,y) satisfying both
    x + m*y = 0 and m*x - y - 2*m + 4 = 0 to the line (x-1)*cos θ + (y-2)*sin θ = 3 is 3 + √5 -/
theorem max_distance_to_line (m : ℝ) :
  let P : ℝ × ℝ := (x, y) 
  ∃ x y : ℝ, x + m*y = 0 ∧ m*x - y - 2*m + 4 = 0 →
  (∀ θ : ℝ, (x - 1)*Real.cos θ + (y - 2)*Real.sin θ ≤ 3 + Real.sqrt 5) ∧
  (∃ θ : ℝ, (x - 1)*Real.cos θ + (y - 2)*Real.sin θ = 3 + Real.sqrt 5) :=
by sorry

end NUMINAMATH_CALUDE_max_distance_to_line_l2333_233307


namespace NUMINAMATH_CALUDE_modified_cube_edges_l2333_233359

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

end NUMINAMATH_CALUDE_modified_cube_edges_l2333_233359


namespace NUMINAMATH_CALUDE_cartons_per_box_l2333_233326

/-- Given information about gum packaging and distribution, prove the number of cartons per box -/
theorem cartons_per_box 
  (packs_per_carton : ℕ) 
  (sticks_per_pack : ℕ)
  (total_sticks : ℕ)
  (total_boxes : ℕ)
  (h1 : packs_per_carton = 5)
  (h2 : sticks_per_pack = 3)
  (h3 : total_sticks = 480)
  (h4 : total_boxes = 8)
  : (total_sticks / total_boxes) / (packs_per_carton * sticks_per_pack) = 4 := by
  sorry

end NUMINAMATH_CALUDE_cartons_per_box_l2333_233326


namespace NUMINAMATH_CALUDE_quadratic_residue_mod_prime_l2333_233301

theorem quadratic_residue_mod_prime (p : Nat) (h_prime : Nat.Prime p) (h_odd : p % 2 = 1) :
  (∃ a : Int, (a ^ 2) % p = (p - 1) % p) ↔ p % 4 = 1 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_residue_mod_prime_l2333_233301


namespace NUMINAMATH_CALUDE_triangle_inequality_constant_l2333_233352

theorem triangle_inequality_constant (a b c : ℝ) (h : a > 0 ∧ b > 0 ∧ c > 0) :
  (a^2 + b^2 + c^2) / (a*b + b*c + c*a) < 2 ∧
  ∀ N : ℝ, (∀ a b c : ℝ, a > 0 → b > 0 → c > 0 →
    (a^2 + b^2 + c^2) / (a*b + b*c + c*a) < N) →
  2 ≤ N :=
sorry

end NUMINAMATH_CALUDE_triangle_inequality_constant_l2333_233352


namespace NUMINAMATH_CALUDE_min_colors_theorem_l2333_233324

/-- The minimum number of colors needed for a regular n-gon -/
def min_colors (n : ℕ) : ℕ :=
  if n % 2 = 1 then n else n - 1

/-- Theorem stating the minimum number of colors needed for a regular n-gon -/
theorem min_colors_theorem (n : ℕ) (h : n ≥ 3) :
  ∃ (m : ℕ), m = min_colors n ∧
  (∀ (k : ℕ), k < m → ¬(∀ (coloring : Fin n → Fin n → Fin k),
    ∀ (v : Fin n), ∀ (i j : Fin n), i ≠ j → coloring v i ≠ coloring v j)) ∧
  (∃ (coloring : Fin n → Fin n → Fin m),
    ∀ (v : Fin n), ∀ (i j : Fin n), i ≠ j → coloring v i ≠ coloring v j) :=
by sorry

#check min_colors_theorem

end NUMINAMATH_CALUDE_min_colors_theorem_l2333_233324


namespace NUMINAMATH_CALUDE_min_value_and_solution_set_l2333_233364

def f (a : ℝ) (x : ℝ) : ℝ := |x + 2| - |x - a|

theorem min_value_and_solution_set (a : ℝ) (h1 : a > 0) :
  (∃ (m : ℝ), m = -3 ∧ ∀ x, f a x ≥ m) →
  (a = 1 ∧ 
   ∀ x, |f a x| ≤ 2 ↔ a / 2 - 2 ≤ x ∧ x < a / 2) :=
by sorry

end NUMINAMATH_CALUDE_min_value_and_solution_set_l2333_233364


namespace NUMINAMATH_CALUDE_mika_birthday_stickers_l2333_233393

/-- The number of stickers Mika gets for her birthday -/
def birthday_stickers : ℕ := sorry

/-- The number of stickers Mika initially had -/
def initial_stickers : ℕ := 20

/-- The number of stickers Mika bought -/
def bought_stickers : ℕ := 26

/-- The number of stickers Mika gave to her sister -/
def given_stickers : ℕ := 6

/-- The number of stickers Mika used for the greeting card -/
def used_stickers : ℕ := 58

/-- The number of stickers Mika has left -/
def remaining_stickers : ℕ := 2

theorem mika_birthday_stickers :
  initial_stickers + bought_stickers + birthday_stickers - given_stickers - used_stickers = remaining_stickers ∧
  birthday_stickers = 20 :=
sorry

end NUMINAMATH_CALUDE_mika_birthday_stickers_l2333_233393


namespace NUMINAMATH_CALUDE_max_product_decomposition_l2333_233336

theorem max_product_decomposition (a : ℝ) (ha : a > 0) :
  ∀ x y : ℝ, x ≥ 0 → y ≥ 0 → x + y = a →
  x * y ≤ (a / 2) * (a / 2) ∧
  (x * y = (a / 2) * (a / 2) ↔ x = a / 2 ∧ y = a / 2) :=
by sorry

end NUMINAMATH_CALUDE_max_product_decomposition_l2333_233336


namespace NUMINAMATH_CALUDE_function_property_l2333_233313

theorem function_property (f : ℝ → ℝ) :
  (∀ (x y : ℝ), x > 0 → y > 0 → f (x * y) = f x / y) →
  f 400 = 4 →
  f 800 = 2 := by
  sorry

end NUMINAMATH_CALUDE_function_property_l2333_233313


namespace NUMINAMATH_CALUDE_simplify_expressions_l2333_233338

theorem simplify_expressions :
  (1 + (-0.5) = 0.5) ∧
  (2 - 10.1 = -10.1) ∧
  (3 + 7 = 10) ∧
  (4 - (-20) = 24) ∧
  (5 + |-(2/3)| = 17/3) ∧
  (6 - |-(4/5)| = 26/5) ∧
  (7 + (-(-10)) = 17) ∧
  (8 - (-(-20/7)) = -12/7) := by
  sorry

end NUMINAMATH_CALUDE_simplify_expressions_l2333_233338


namespace NUMINAMATH_CALUDE_system_solution_l2333_233375

theorem system_solution (x y : ℤ) (h1 : 7 - x = 15) (h2 : y - 3 = 4 + x) :
  x = -8 ∧ y = -1 := by
  sorry

end NUMINAMATH_CALUDE_system_solution_l2333_233375


namespace NUMINAMATH_CALUDE_largest_c_for_negative_three_in_range_l2333_233343

-- Define the function g(x)
def g (c : ℝ) (x : ℝ) : ℝ := x^2 + 4*x + c

-- State the theorem
theorem largest_c_for_negative_three_in_range :
  (∃ (c : ℝ), ∀ (c' : ℝ), (∃ (x : ℝ), g c' x = -3) → c' ≤ c) ∧
  (∃ (x : ℝ), g 1 x = -3) :=
sorry

end NUMINAMATH_CALUDE_largest_c_for_negative_three_in_range_l2333_233343


namespace NUMINAMATH_CALUDE_parabola_equation_l2333_233392

/-- A parabola is defined by its axis equation and standard form equation. -/
structure Parabola where
  /-- The x-coordinate of the axis of the parabola -/
  axis : ℝ
  /-- The coefficient in the standard form equation y² = 2px -/
  p : ℝ
  /-- Condition that p is positive -/
  p_pos : p > 0

/-- The standard form equation of a parabola is y² = 2px -/
def standard_form (para : Parabola) : Prop :=
  ∀ x y : ℝ, y^2 = 2 * para.p * x

/-- The axis equation of a parabola is x = -p/2 -/
def axis_equation (para : Parabola) : Prop :=
  para.axis = -para.p / 2

/-- Theorem: Given a parabola with axis equation x = -2, its standard form equation is y² = 8x -/
theorem parabola_equation (para : Parabola) 
  (h : axis_equation para) 
  (h_axis : para.axis = -2) : 
  standard_form para ∧ para.p = 4 := by
  sorry

end NUMINAMATH_CALUDE_parabola_equation_l2333_233392


namespace NUMINAMATH_CALUDE_binary_sum_equals_638_l2333_233363

def binary_to_decimal (b : ℕ) : ℕ := 2^b - 1

theorem binary_sum_equals_638 :
  (binary_to_decimal 9) + (binary_to_decimal 7) = 638 := by
  sorry

end NUMINAMATH_CALUDE_binary_sum_equals_638_l2333_233363


namespace NUMINAMATH_CALUDE_locus_of_angle_bisector_intersection_l2333_233329

-- Define the points and constants
variable (a : ℝ) -- distance between A and B
variable (x₀ y₀ : ℝ) -- coordinates of point C
variable (x y : ℝ) -- coordinates of point P

-- State the theorem
theorem locus_of_angle_bisector_intersection 
  (h1 : a > 0) -- A and B are distinct points
  (h2 : x₀^2 + y₀^2 = 1) -- C is on the unit circle centered at A
  (h3 : x = (a * x₀) / (1 + a)) -- x-coordinate of P
  (h4 : y = (a * y₀) / (1 + a)) -- y-coordinate of P
  : x^2 + y^2 = (a^2) / ((1 + a)^2) := by
  sorry

end NUMINAMATH_CALUDE_locus_of_angle_bisector_intersection_l2333_233329


namespace NUMINAMATH_CALUDE_cab_driver_income_l2333_233350

/-- Given a cab driver's income for 5 days, prove that the income on the third day is $450 -/
theorem cab_driver_income (income : Fin 5 → ℕ) 
  (day1 : income 0 = 600)
  (day2 : income 1 = 250)
  (day4 : income 3 = 400)
  (day5 : income 4 = 800)
  (avg_income : (income 0 + income 1 + income 2 + income 3 + income 4) / 5 = 500) :
  income 2 = 450 := by
  sorry

end NUMINAMATH_CALUDE_cab_driver_income_l2333_233350


namespace NUMINAMATH_CALUDE_alcohol_mixture_problem_l2333_233353

/-- Proves that given a mixture of x litres with 20% alcohol, when 5 litres of water are added 
    resulting in a new mixture with 15% alcohol, the value of x is 15 litres. -/
theorem alcohol_mixture_problem (x : ℝ) 
  (h1 : x > 0)  -- Ensure x is positive
  (h2 : 0.20 * x = 0.15 * (x + 5)) : x = 15 := by
  sorry

end NUMINAMATH_CALUDE_alcohol_mixture_problem_l2333_233353


namespace NUMINAMATH_CALUDE_burger_non_filler_percentage_l2333_233358

/-- Given a burger with total weight and filler weight, calculate the percentage that is not filler -/
theorem burger_non_filler_percentage 
  (total_weight : ℝ) 
  (filler_weight : ℝ) 
  (h1 : total_weight = 120)
  (h2 : filler_weight = 30) : 
  (total_weight - filler_weight) / total_weight * 100 = 75 := by
  sorry

end NUMINAMATH_CALUDE_burger_non_filler_percentage_l2333_233358


namespace NUMINAMATH_CALUDE_problem_solution_l2333_233335

theorem problem_solution (x y : ℝ) 
  (h1 : x + y = 5) 
  (h2 : x * y = 3) 
  (h3 : x^3 - x^2 - 4*x + 4 = 0) 
  (h4 : y^3 - y^2 - 4*y + 4 = 0) : 
  x + x^3/y^2 + y^3/x^2 + y = 174 := by sorry

end NUMINAMATH_CALUDE_problem_solution_l2333_233335


namespace NUMINAMATH_CALUDE_jimmy_garden_servings_l2333_233378

/-- Represents the number of plants in each plot -/
def plants_per_plot : ℕ := 9

/-- Represents the number of servings produced by each carrot plant -/
def carrot_servings : ℕ := 4

/-- Represents the number of servings produced by each corn plant -/
def corn_servings : ℕ := 5 * carrot_servings

/-- Represents the number of servings produced by each green bean plant -/
def green_bean_servings : ℕ := corn_servings / 2

/-- Calculates the total number of servings from all three plots -/
def total_servings : ℕ := 
  plants_per_plot * carrot_servings +
  plants_per_plot * corn_servings +
  plants_per_plot * green_bean_servings

/-- Theorem stating that the total number of servings is 306 -/
theorem jimmy_garden_servings : total_servings = 306 := by
  sorry

end NUMINAMATH_CALUDE_jimmy_garden_servings_l2333_233378


namespace NUMINAMATH_CALUDE_intersection_of_sets_l2333_233347

theorem intersection_of_sets : 
  let A : Set ℕ := {1, 2, 3}
  let B : Set ℕ := {1, 2, 5}
  A ∩ B = {1, 2} := by
sorry

end NUMINAMATH_CALUDE_intersection_of_sets_l2333_233347


namespace NUMINAMATH_CALUDE_angela_insect_count_l2333_233337

theorem angela_insect_count (dean_insects jacob_insects angela_insects : ℕ) : 
  dean_insects = 30 →
  jacob_insects = 5 * dean_insects →
  angela_insects = jacob_insects / 2 →
  angela_insects = 75 := by
  sorry

end NUMINAMATH_CALUDE_angela_insect_count_l2333_233337


namespace NUMINAMATH_CALUDE_sum_max_min_f_on_interval_l2333_233322

def f (x : ℝ) : ℝ := x^2 - 4*x + 1

theorem sum_max_min_f_on_interval : 
  ∃ (max min : ℝ), 
    (∀ x ∈ Set.Icc 0 5, f x ≤ max) ∧ 
    (∃ x ∈ Set.Icc 0 5, f x = max) ∧
    (∀ x ∈ Set.Icc 0 5, min ≤ f x) ∧ 
    (∃ x ∈ Set.Icc 0 5, f x = min) ∧
    max + min = 3 :=
by sorry

end NUMINAMATH_CALUDE_sum_max_min_f_on_interval_l2333_233322


namespace NUMINAMATH_CALUDE_arithmetic_sequence_sum_l2333_233344

def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_sum (a : ℕ → ℝ) :
  arithmetic_sequence a →
  (a 1 + a 2 + a 3 = 9) →
  (a 4 + a 5 + a 6 = 27) →
  (a 7 + a 8 + a 9 = 45) :=
by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_sum_l2333_233344


namespace NUMINAMATH_CALUDE_fraction_equality_l2333_233391

theorem fraction_equality (a : ℕ+) : 
  (a : ℚ) / (a + 37) = 875 / 1000 → a = 259 := by
  sorry

end NUMINAMATH_CALUDE_fraction_equality_l2333_233391


namespace NUMINAMATH_CALUDE_find_n_l2333_233366

theorem find_n : ∃ n : ℤ, n + (n + 1) + (n + 2) = 9 ∧ n = 2 := by
  sorry

end NUMINAMATH_CALUDE_find_n_l2333_233366


namespace NUMINAMATH_CALUDE_line_shift_theorem_l2333_233339

-- Define the original line
def original_line (x : ℝ) : ℝ := -2 * x + 1

-- Define the shift amount
def shift : ℝ := 2

-- Define the shifted line
def shifted_line (x : ℝ) : ℝ := original_line (x + shift)

-- Theorem statement
theorem line_shift_theorem :
  ∀ x : ℝ, shifted_line x = -2 * x - 3 := by
  sorry

end NUMINAMATH_CALUDE_line_shift_theorem_l2333_233339


namespace NUMINAMATH_CALUDE_larger_number_proof_l2333_233354

theorem larger_number_proof (x y : ℝ) (h1 : x - y = 3) (h2 : x^2 - y^2 = 39) : x = 8 := by
  sorry

end NUMINAMATH_CALUDE_larger_number_proof_l2333_233354


namespace NUMINAMATH_CALUDE_solution_set_for_a_equals_one_solution_range_for_a_l2333_233340

-- Define the function f(x) = |x-a|
def f (a : ℝ) (x : ℝ) : ℝ := |x - a|

-- Part 1
theorem solution_set_for_a_equals_one :
  {x : ℝ | |x - 1| > (1/2) * (x + 1)} = {x : ℝ | x > 3 ∨ x < 1/3} := by sorry

-- Part 2
theorem solution_range_for_a :
  ∀ a : ℝ, (∃ x : ℝ, |x - a| + |x - 2| ≤ 3) ↔ -1 ≤ a ∧ a ≤ 5 := by sorry

end NUMINAMATH_CALUDE_solution_set_for_a_equals_one_solution_range_for_a_l2333_233340


namespace NUMINAMATH_CALUDE_common_area_rectangle_ellipse_l2333_233371

/-- Represents a rectangle with width and height -/
structure Rectangle where
  width : ℝ
  height : ℝ

/-- Represents an ellipse with semi-major and semi-minor axis lengths -/
structure Ellipse where
  semiMajor : ℝ
  semiMinor : ℝ

/-- Calculates the area of the region common to a rectangle and an ellipse that share the same center -/
def commonArea (r : Rectangle) (e : Ellipse) : ℝ := sorry

theorem common_area_rectangle_ellipse :
  let r := Rectangle.mk 10 4
  let e := Ellipse.mk 3 2
  commonArea r e = 6 * Real.pi := by sorry

end NUMINAMATH_CALUDE_common_area_rectangle_ellipse_l2333_233371


namespace NUMINAMATH_CALUDE_binary_representation_of_106_l2333_233380

/-- Converts a natural number to its binary representation as a list of bits -/
def toBinary (n : ℕ) : List Bool :=
  if n = 0 then [false]
  else
    let rec go (m : ℕ) (acc : List Bool) : List Bool :=
      if m = 0 then acc
      else go (m / 2) ((m % 2 = 1) :: acc)
    go n []

/-- Converts a list of bits to a string representation of a binary number -/
def binaryToString (bits : List Bool) : String :=
  bits.map (fun b => if b then '1' else '0') |> String.mk

theorem binary_representation_of_106 :
  binaryToString (toBinary 106) = "1101010" := by
  sorry

#eval binaryToString (toBinary 106)

end NUMINAMATH_CALUDE_binary_representation_of_106_l2333_233380


namespace NUMINAMATH_CALUDE_ascending_order_for_negative_x_l2333_233328

theorem ascending_order_for_negative_x (x : ℝ) (h : -1 < x ∧ x < 0) : 
  5 * x < 0.5 * x ∧ 0.5 * x < 5 - x := by
  sorry

end NUMINAMATH_CALUDE_ascending_order_for_negative_x_l2333_233328


namespace NUMINAMATH_CALUDE_triangle_side_length_l2333_233355

theorem triangle_side_length (a : ℝ) : 
  (5 : ℝ) > 0 ∧ (8 : ℝ) > 0 ∧ a > 0 →
  (5 + 8 > a ∧ 5 + a > 8 ∧ 8 + a > 5) ↔ (3 < a ∧ a < 13) :=
by sorry

end NUMINAMATH_CALUDE_triangle_side_length_l2333_233355


namespace NUMINAMATH_CALUDE_fraction_simplification_l2333_233306

theorem fraction_simplification : (10^9 : ℕ) / (2 * 10^5) = 5000 := by
  sorry

end NUMINAMATH_CALUDE_fraction_simplification_l2333_233306


namespace NUMINAMATH_CALUDE_power_23_2023_mod_29_l2333_233397

theorem power_23_2023_mod_29 : 23^2023 % 29 = 24 := by
  sorry

end NUMINAMATH_CALUDE_power_23_2023_mod_29_l2333_233397


namespace NUMINAMATH_CALUDE_tan_product_special_angles_l2333_233331

theorem tan_product_special_angles :
  let A : Real := 30 * π / 180
  let B : Real := 60 * π / 180
  (1 + Real.tan A) * (1 + Real.tan B) = 4 + 2 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_tan_product_special_angles_l2333_233331


namespace NUMINAMATH_CALUDE_hockey_league_games_l2333_233319

theorem hockey_league_games (n : ℕ) (games_per_pair : ℕ) : n = 10 ∧ games_per_pair = 4 →
  (n * (n - 1) / 2) * games_per_pair = 180 := by
  sorry

end NUMINAMATH_CALUDE_hockey_league_games_l2333_233319


namespace NUMINAMATH_CALUDE_octahedron_cube_volume_ratio_l2333_233312

/-- Given a cube, an octahedron is formed by joining the centers of adjoining faces. -/
structure OctahedronFromCube where
  cube_side : ℝ
  cube_volume : ℝ
  octahedron_side : ℝ
  octahedron_volume : ℝ

/-- The ratio of the volume of the octahedron to the volume of the cube is 1/6. -/
theorem octahedron_cube_volume_ratio (o : OctahedronFromCube) :
  o.octahedron_volume / o.cube_volume = 1 / 6 := by
  sorry

end NUMINAMATH_CALUDE_octahedron_cube_volume_ratio_l2333_233312


namespace NUMINAMATH_CALUDE_statement_A_incorrect_l2333_233309

/-- Represents the process of meiosis and fertilization -/
structure MeiosisFertilization where
  sperm_transformation : Bool
  egg_metabolism_increase : Bool
  homologous_chromosomes_appearance : Bool
  fertilization_randomness : Bool

/-- Represents the correctness of statements about meiosis and fertilization -/
structure Statements where
  A : Bool
  B : Bool
  C : Bool
  D : Bool

/-- The given information about meiosis and fertilization -/
def given_info : MeiosisFertilization :=
  { sperm_transformation := true
  , egg_metabolism_increase := true
  , homologous_chromosomes_appearance := true
  , fertilization_randomness := true }

/-- The correctness of statements based on the given information -/
def statement_correctness (info : MeiosisFertilization) : Statements :=
  { A := false  -- Statement A is incorrect
  , B := info.sperm_transformation && info.egg_metabolism_increase
  , C := info.homologous_chromosomes_appearance
  , D := info.fertilization_randomness }

/-- Theorem stating that statement A is incorrect -/
theorem statement_A_incorrect (info : MeiosisFertilization) :
  (statement_correctness info).A = false := by
  sorry

end NUMINAMATH_CALUDE_statement_A_incorrect_l2333_233309


namespace NUMINAMATH_CALUDE_hotel_room_pricing_and_schemes_l2333_233382

theorem hotel_room_pricing_and_schemes :
  ∀ (price_A price_B : ℕ) (schemes : List (ℕ × ℕ)),
  (∃ n : ℕ, 6000 = n * price_A ∧ 4400 = n * price_B) →
  price_A = price_B + 80 →
  (∀ (a b : ℕ), (a, b) ∈ schemes → a + b = 30) →
  (∀ (a b : ℕ), (a, b) ∈ schemes → 2 * a ≥ b) →
  (∀ (a b : ℕ), (a, b) ∈ schemes → a * price_A + b * price_B ≤ 7600) →
  price_A = 300 ∧ price_B = 220 ∧ schemes = [(10, 20), (11, 19), (12, 18)] := by
  sorry

end NUMINAMATH_CALUDE_hotel_room_pricing_and_schemes_l2333_233382


namespace NUMINAMATH_CALUDE_max_demand_decrease_l2333_233351

theorem max_demand_decrease (price_increase : ℝ) (revenue_increase : ℝ) : 
  price_increase = 0.20 →
  revenue_increase = 0.10 →
  (1 + price_increase) * (1 - (1 / 12 : ℝ)) ≥ 1 + revenue_increase :=
by sorry

end NUMINAMATH_CALUDE_max_demand_decrease_l2333_233351


namespace NUMINAMATH_CALUDE_medal_award_ways_l2333_233333

def total_sprinters : ℕ := 10
def american_sprinters : ℕ := 4
def medals : ℕ := 3

def ways_to_award_medals : ℕ := 
  let non_american_sprinters := total_sprinters - american_sprinters
  let no_american_medals := non_american_sprinters * (non_american_sprinters - 1) * (non_american_sprinters - 2)
  let one_american_medal := american_sprinters * medals * non_american_sprinters * (non_american_sprinters - 1)
  no_american_medals + one_american_medal

theorem medal_award_ways :
  ways_to_award_medals = 480 :=
by sorry

end NUMINAMATH_CALUDE_medal_award_ways_l2333_233333


namespace NUMINAMATH_CALUDE_defective_part_probability_l2333_233349

theorem defective_part_probability (p : ℝ) : 
  (0 ≤ p) ∧ (p ≤ 1) →
  (1 - 0.01) * (1 - p) = 0.9603 →
  p = 0.03 := by
sorry

end NUMINAMATH_CALUDE_defective_part_probability_l2333_233349


namespace NUMINAMATH_CALUDE_max_value_product_l2333_233361

theorem max_value_product (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : 5 * x + 6 * y < 90) :
  x * y * (90 - 5 * x - 6 * y) ≤ 900 ∧
  ∃ x₀ y₀ : ℝ, x₀ > 0 ∧ y₀ > 0 ∧ 5 * x₀ + 6 * y₀ < 90 ∧ x₀ * y₀ * (90 - 5 * x₀ - 6 * y₀) = 900 :=
by sorry

end NUMINAMATH_CALUDE_max_value_product_l2333_233361


namespace NUMINAMATH_CALUDE_characterization_of_solutions_l2333_233389

/-- Sum of digits of a non-negative integer -/
def sum_of_digits (n : ℕ) : ℕ :=
  if n < 10 then n else (n % 10) + sum_of_digits (n / 10)

/-- The set of solutions -/
def solution_set : Set ℕ := {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 18, 19}

/-- Main theorem: n ≤ 2s(n) iff n is in the solution set -/
theorem characterization_of_solutions (n : ℕ) :
  n ≤ 2 * sum_of_digits n ↔ n ∈ solution_set :=
sorry

end NUMINAMATH_CALUDE_characterization_of_solutions_l2333_233389


namespace NUMINAMATH_CALUDE_no_real_roots_l2333_233346

def f (x : ℝ) : ℝ := x^2 - 2*x + 3

theorem no_real_roots : ∀ x : ℝ, f x ≠ 0 := by
  sorry

end NUMINAMATH_CALUDE_no_real_roots_l2333_233346


namespace NUMINAMATH_CALUDE_seating_arrangements_l2333_233311

/-- The number of ways to arrange n people in a row. -/
def arrangements (n : ℕ) : ℕ := Nat.factorial n

/-- The number of ways to arrange n people in a row, 
    where a group of k people must sit consecutively. -/
def arrangementsWithConsecutiveGroup (n k : ℕ) : ℕ := 
  arrangements (n - k + 1) * arrangements k

/-- The number of ways to arrange 10 people in a row, 
    where 4 specific people cannot sit in 4 consecutive seats. -/
theorem seating_arrangements : 
  arrangements 10 - arrangementsWithConsecutiveGroup 10 4 = 3507840 := by
  sorry

#eval arrangements 10 - arrangementsWithConsecutiveGroup 10 4

end NUMINAMATH_CALUDE_seating_arrangements_l2333_233311


namespace NUMINAMATH_CALUDE_max_roses_is_316_l2333_233341

/-- The price of an individual rose in cents -/
def individual_price : ℕ := 730

/-- The price of one dozen roses in cents -/
def dozen_price : ℕ := 3600

/-- The price of two dozen roses in cents -/
def two_dozen_price : ℕ := 5000

/-- The total budget in cents -/
def budget : ℕ := 68000

/-- The function to calculate the maximum number of roses that can be purchased -/
def max_roses : ℕ :=
  let two_dozen_sets := budget / two_dozen_price
  let remaining := budget % two_dozen_price
  let individual_roses := remaining / individual_price
  two_dozen_sets * 24 + individual_roses

/-- Theorem stating that the maximum number of roses that can be purchased is 316 -/
theorem max_roses_is_316 : max_roses = 316 := by
  sorry

end NUMINAMATH_CALUDE_max_roses_is_316_l2333_233341


namespace NUMINAMATH_CALUDE_total_fruits_is_107_l2333_233332

-- Define the number of oranges and apples picked by George and Amelia
def george_oranges : ℕ := 45
def amelia_apples : ℕ := 15
def george_amelia_apple_diff : ℕ := 5
def george_amelia_orange_diff : ℕ := 18

-- Define the number of apples picked by George
def george_apples : ℕ := amelia_apples + george_amelia_apple_diff

-- Define the number of oranges picked by Amelia
def amelia_oranges : ℕ := george_oranges - george_amelia_orange_diff

-- Define the total number of fruits picked
def total_fruits : ℕ := george_oranges + george_apples + amelia_oranges + amelia_apples

-- Theorem statement
theorem total_fruits_is_107 : total_fruits = 107 := by sorry

end NUMINAMATH_CALUDE_total_fruits_is_107_l2333_233332


namespace NUMINAMATH_CALUDE_sets_intersection_union_l2333_233317

-- Define the sets A, B, and C
def A : Set ℝ := {x | x^2 + x - 2 ≤ 0}
def B : Set ℝ := {x | 2 < x + 1 ∧ x + 1 ≤ 4}
def C (b c : ℝ) : Set ℝ := {x | x^2 + b*x + c > 0}

-- State the theorem
theorem sets_intersection_union (b c : ℝ) :
  (A ∪ B) ∩ C b c = ∅ ∧ (A ∪ B) ∪ C b c = Set.univ →
  b = -1 ∧ c = -6 := by
  sorry

end NUMINAMATH_CALUDE_sets_intersection_union_l2333_233317


namespace NUMINAMATH_CALUDE_total_distance_jogged_l2333_233330

/-- The total distance jogged by Kyle and Sarah -/
def total_distance (inner_track_length outer_track_length : ℝ)
  (kyle_inner_laps kyle_outer_laps : ℝ)
  (sarah_inner_laps sarah_outer_laps : ℝ) : ℝ :=
  (kyle_inner_laps * inner_track_length + kyle_outer_laps * outer_track_length) +
  (sarah_inner_laps * inner_track_length + sarah_outer_laps * outer_track_length)

/-- Theorem stating the total distance jogged by Kyle and Sarah -/
theorem total_distance_jogged :
  total_distance 250 400 1.12 1.78 2.73 1.36 = 2218.5 := by
  sorry

end NUMINAMATH_CALUDE_total_distance_jogged_l2333_233330


namespace NUMINAMATH_CALUDE_three_digit_numbers_count_l2333_233310

/-- Represents a card with two distinct numbers -/
structure Card where
  front : Nat
  back : Nat
  distinct : front ≠ back

/-- The set of cards given in the problem -/
def cards : Finset Card := sorry

/-- The number of cards -/
def num_cards : Nat := Finset.card cards

/-- The number of cards used to form a number -/
def cards_used : Nat := 3

/-- Calculates the number of different three-digit numbers that can be formed -/
def num_three_digit_numbers : Nat :=
  (num_cards.choose cards_used) * (2^cards_used) * (cards_used.factorial)

theorem three_digit_numbers_count :
  num_three_digit_numbers = 192 := by sorry

end NUMINAMATH_CALUDE_three_digit_numbers_count_l2333_233310

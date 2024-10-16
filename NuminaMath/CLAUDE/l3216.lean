import Mathlib

namespace NUMINAMATH_CALUDE_sqrt_112_between_consecutive_integers_product_l3216_321603

theorem sqrt_112_between_consecutive_integers_product : ∃ (n : ℕ), 
  n > 0 ∧ 
  n^2 < 112 ∧ 
  (n + 1)^2 > 112 ∧ 
  n * (n + 1) = 110 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_112_between_consecutive_integers_product_l3216_321603


namespace NUMINAMATH_CALUDE_tetrahedron_PQRS_volume_l3216_321661

/-- The volume of a tetrahedron given its edge lengths -/
noncomputable def tetrahedronVolume (a b c d e f : ℝ) : ℝ :=
  (1 / 6) * Real.sqrt (
    a^2 * b^2 * c^2 + a^2 * d^2 * e^2 + b^2 * d^2 * f^2 + c^2 * e^2 * f^2
    - a^2 * (d^2 * e^2 + d^2 * f^2 + e^2 * f^2)
    - b^2 * (c^2 * e^2 + c^2 * f^2 + e^2 * f^2)
    - c^2 * (b^2 * d^2 + b^2 * f^2 + d^2 * f^2)
    - d^2 * (a^2 * e^2 + a^2 * f^2 + e^2 * f^2)
    - e^2 * (a^2 * d^2 + a^2 * f^2 + d^2 * f^2)
    - f^2 * (a^2 * b^2 + a^2 * c^2 + b^2 * c^2)
  )

theorem tetrahedron_PQRS_volume :
  let PQ : ℝ := 6
  let PR : ℝ := 4
  let PS : ℝ := (12 / 5) * Real.sqrt 2
  let QR : ℝ := 3
  let QS : ℝ := 4
  let RS : ℝ := (12 / 5) * Real.sqrt 5
  tetrahedronVolume PQ PR PS QR QS RS = 24 / 5 := by
  sorry

end NUMINAMATH_CALUDE_tetrahedron_PQRS_volume_l3216_321661


namespace NUMINAMATH_CALUDE_guys_with_bullets_l3216_321609

theorem guys_with_bullets (n : ℕ) (h : n > 0) : 
  (∀ (guy : Fin n), 25 - 4 = (n * 25 - n * 4) / n) → n ≥ 1 :=
by sorry

end NUMINAMATH_CALUDE_guys_with_bullets_l3216_321609


namespace NUMINAMATH_CALUDE_f_is_periodic_l3216_321602

/-- Given two functions f and g on ℝ satisfying certain conditions, 
    prove that f is periodic -/
theorem f_is_periodic 
  (f g : ℝ → ℝ)
  (h₁ : f 0 = 1)
  (h₂ : ∃ a : ℝ, a ≠ 0 ∧ g a = 1)
  (h₃ : ∀ x, g (-x) = -g x)
  (h₄ : ∀ x y, f (x - y) = f x * f y + g x * g y) :
  ∃ p : ℝ, p ≠ 0 ∧ ∀ x, f (x + p) = f x :=
sorry

end NUMINAMATH_CALUDE_f_is_periodic_l3216_321602


namespace NUMINAMATH_CALUDE_polyhedron_20_faces_l3216_321637

/-- A polyhedron with triangular faces -/
structure Polyhedron where
  faces : ℕ
  edges : ℕ
  vertices : ℕ

/-- The Euler characteristic for polyhedra -/
def euler_characteristic (p : Polyhedron) : ℕ :=
  p.vertices - p.edges + p.faces

/-- Theorem: A polyhedron with 20 triangular faces has 30 edges and 12 vertices -/
theorem polyhedron_20_faces (p : Polyhedron) 
  (h_faces : p.faces = 20) 
  (h_triangular : p.edges * 2 = p.faces * 3) 
  (h_euler : euler_characteristic p = 2) : 
  p.edges = 30 ∧ p.vertices = 12 := by
  sorry


end NUMINAMATH_CALUDE_polyhedron_20_faces_l3216_321637


namespace NUMINAMATH_CALUDE_triangulation_reconstruction_l3216_321693

/-- A convex polygon represented by its vertices -/
structure ConvexPolygon where
  vertices : List ℝ × ℝ
  is_convex : sorry

/-- A triangulation of a convex polygon -/
structure Triangulation (P : ConvexPolygon) where
  diagonals : List (ℕ × ℕ)
  is_valid : sorry

/-- The number of triangles adjacent to each vertex in a triangulation -/
def adjacentTriangles (P : ConvexPolygon) (T : Triangulation P) : List ℕ :=
  sorry

/-- Theorem stating that a triangulation can be uniquely reconstructed from adjacent triangle counts -/
theorem triangulation_reconstruction
  (P : ConvexPolygon)
  (T1 T2 : Triangulation P)
  (h : adjacentTriangles P T1 = adjacentTriangles P T2) :
  T1 = T2 :=
sorry

end NUMINAMATH_CALUDE_triangulation_reconstruction_l3216_321693


namespace NUMINAMATH_CALUDE_problem_solution_l3216_321634

theorem problem_solution (a b x y m : ℝ) 
  (h1 : a + b = 0) 
  (h2 : x * y = 1) 
  (h3 : |m| = 2) : 
  m^2 + (a + b) / 2 + (-x * y)^2023 = 3 := by
sorry

end NUMINAMATH_CALUDE_problem_solution_l3216_321634


namespace NUMINAMATH_CALUDE_orange_juice_percentage_l3216_321673

/-- Represents the composition and pricing of a drink made from milk and orange juice -/
structure DrinkComposition where
  milk_mass : ℝ
  juice_mass : ℝ
  initial_milk_price : ℝ
  initial_juice_price : ℝ
  milk_price_change : ℝ
  juice_price_change : ℝ

/-- The theorem stating the mass percentage of orange juice in the drink -/
theorem orange_juice_percentage (drink : DrinkComposition) 
  (h_price_ratio : drink.initial_juice_price = 6 * drink.initial_milk_price)
  (h_milk_change : drink.milk_price_change = -0.15)
  (h_juice_change : drink.juice_price_change = 0.1)
  (h_cost_unchanged : 
    drink.milk_mass * drink.initial_milk_price * (1 + drink.milk_price_change) + 
    drink.juice_mass * drink.initial_juice_price * (1 + drink.juice_price_change) = 
    drink.milk_mass * drink.initial_milk_price + 
    drink.juice_mass * drink.initial_juice_price) :
  drink.juice_mass / (drink.milk_mass + drink.juice_mass) = 0.2 := by
  sorry

end NUMINAMATH_CALUDE_orange_juice_percentage_l3216_321673


namespace NUMINAMATH_CALUDE_quadratic_function_domain_range_conditions_l3216_321649

/-- Given a quadratic function f(x) = -1/2 * x^2 + x with domain [m, n] and range [k*m, k*n],
    prove that m = 2(1 - k) and n = 0 must be satisfied. -/
theorem quadratic_function_domain_range_conditions
  (f : ℝ → ℝ)
  (m n k : ℝ)
  (h_f : ∀ x, f x = -1/2 * x^2 + x)
  (h_domain : Set.Icc m n = {x | f x ∈ Set.Icc (k * m) (k * n)})
  (h_m_lt_n : m < n)
  (h_k_gt_1 : k > 1) :
  m = 2 * (1 - k) ∧ n = 0 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_function_domain_range_conditions_l3216_321649


namespace NUMINAMATH_CALUDE_line_parameterization_l3216_321683

/-- Given a line y = (3/4)x + 2 parameterized by [x; y] = [-8; s] + t[l; -6],
    prove that s = -4 and l = -8 -/
theorem line_parameterization (s l : ℝ) : 
  (∀ x y t : ℝ, y = (3/4) * x + 2 ↔ ∃ t, (x, y) = (-8 + t * l, s + t * (-6))) →
  s = -4 ∧ l = -8 := by
  sorry

end NUMINAMATH_CALUDE_line_parameterization_l3216_321683


namespace NUMINAMATH_CALUDE_complex_equation_solution_l3216_321659

theorem complex_equation_solution :
  ∀ x : ℝ, (1 - 2*I) * (x + I) = 4 - 3*I → x = 2 := by
  sorry

end NUMINAMATH_CALUDE_complex_equation_solution_l3216_321659


namespace NUMINAMATH_CALUDE_opposite_number_l3216_321618

theorem opposite_number (a : ℤ) : (∀ b : ℤ, a + b = 0 → b = -2022) → a = 2022 := by
  sorry

end NUMINAMATH_CALUDE_opposite_number_l3216_321618


namespace NUMINAMATH_CALUDE_min_value_xy_plus_two_over_xy_l3216_321670

theorem min_value_xy_plus_two_over_xy (x y : ℝ) 
  (hx : x > 0) (hy : y > 0) (hsum : x + y = 1) :
  (∀ z w : ℝ, z > 0 → w > 0 → z + w = 1 → x * y + 2 / (x * y) ≤ z * w + 2 / (z * w)) ∧ 
  x * y + 2 / (x * y) = 33 / 4 :=
sorry

end NUMINAMATH_CALUDE_min_value_xy_plus_two_over_xy_l3216_321670


namespace NUMINAMATH_CALUDE_chosen_number_l3216_321662

theorem chosen_number (x : ℝ) : (x / 12)^2 - 240 = 8 → x = 24 * Real.sqrt 62 := by
  sorry

end NUMINAMATH_CALUDE_chosen_number_l3216_321662


namespace NUMINAMATH_CALUDE_sum_squares_inequality_l3216_321642

theorem sum_squares_inequality (a b c : ℝ) 
  (h_pos : 0 < a ∧ 0 < b ∧ 0 < c) 
  (h_sum : a^2 + b^2 + c^2 = 3) : 
  a + b + c ≥ a^2*b^2 + b^2*c^2 + c^2*a^2 := by
sorry

end NUMINAMATH_CALUDE_sum_squares_inequality_l3216_321642


namespace NUMINAMATH_CALUDE_line_ellipse_intersection_slopes_l3216_321641

-- Define the y-intercept
def y_intercept : ℝ := 8

-- Define the ellipse equation
def ellipse_eq (x y : ℝ) : Prop := 4 * x^2 + 25 * y^2 = 100

-- Define the line equation
def line_eq (m x : ℝ) : ℝ := m * x + y_intercept

-- Theorem statement
theorem line_ellipse_intersection_slopes :
  ∀ m : ℝ, (∃ x : ℝ, ellipse_eq x (line_eq m x)) ↔ m^2 ≥ 2.4 :=
by sorry

end NUMINAMATH_CALUDE_line_ellipse_intersection_slopes_l3216_321641


namespace NUMINAMATH_CALUDE_stating_adjacent_probability_in_grid_l3216_321638

/-- The number of students -/
def num_students : ℕ := 8

/-- The number of rows in the seating arrangement -/
def num_rows : ℕ := 2

/-- The number of columns in the seating arrangement -/
def num_columns : ℕ := 4

/-- The probability of two specific students being adjacent -/
def adjacent_probability : ℚ := 5/14

/-- 
Theorem stating that the probability of two specific students 
being adjacent in a random seating arrangement is 5/14
-/
theorem adjacent_probability_in_grid : 
  let total_arrangements := Nat.factorial num_students
  let row_adjacent_pairs := num_rows * (num_columns - 1)
  let column_adjacent_pairs := num_columns
  let ways_to_arrange_pair := 2
  let remaining_arrangements := Nat.factorial (num_students - 2)
  let favorable_outcomes := (row_adjacent_pairs + column_adjacent_pairs) * 
                            ways_to_arrange_pair * 
                            remaining_arrangements
  (favorable_outcomes : ℚ) / total_arrangements = adjacent_probability := by
  sorry

end NUMINAMATH_CALUDE_stating_adjacent_probability_in_grid_l3216_321638


namespace NUMINAMATH_CALUDE_converse_and_inverse_false_l3216_321635

-- Define the universe of polygons
variable (Polygon : Type)

-- Define properties of polygons
variable (is_rhombus : Polygon → Prop)
variable (is_parallelogram : Polygon → Prop)

-- Original statement
axiom original_statement : ∀ p : Polygon, is_rhombus p → is_parallelogram p

-- Theorem to prove
theorem converse_and_inverse_false :
  (¬ ∀ p : Polygon, is_parallelogram p → is_rhombus p) ∧
  (¬ ∀ p : Polygon, ¬is_rhombus p → ¬is_parallelogram p) :=
by sorry

end NUMINAMATH_CALUDE_converse_and_inverse_false_l3216_321635


namespace NUMINAMATH_CALUDE_point_move_result_l3216_321605

def point_move (initial_position : ℤ) (move_distance : ℤ) : Set ℤ :=
  {initial_position - move_distance, initial_position + move_distance}

theorem point_move_result :
  point_move (-5) 3 = {-8, -2} := by sorry

end NUMINAMATH_CALUDE_point_move_result_l3216_321605


namespace NUMINAMATH_CALUDE_functional_equation_solution_l3216_321616

/-- A function satisfying the given functional equation -/
def SatisfiesFunctionalEquation (f : ℝ → ℝ) : Prop :=
  ∀ x y : ℝ, f (x * (1 + y)) = f x * (1 + f y)

/-- The main theorem stating that any function satisfying the functional equation
    is either the identity function or the zero function -/
theorem functional_equation_solution (f : ℝ → ℝ) 
  (h : SatisfiesFunctionalEquation f) : 
  (∀ x : ℝ, f x = x) ∨ (∀ x : ℝ, f x = 0) := by
  sorry

end NUMINAMATH_CALUDE_functional_equation_solution_l3216_321616


namespace NUMINAMATH_CALUDE_cube_volume_partition_l3216_321675

/-- Represents a point in 3D space -/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- Represents a cube in 3D space -/
structure Cube where
  sideLength : ℝ

/-- Represents a plane in 3D space -/
structure Plane where
  a : ℝ
  b : ℝ
  c : ℝ
  d : ℝ

/-- Given a cube and a plane passing through the midpoint of one edge and two points
    on opposite edges with the ratio 1:7 from the vertices, the smaller part of the
    volume separated by this plane is 25/192 of the cube's volume. -/
theorem cube_volume_partition (cube : Cube) (plane : Plane)
  (h1 : plane.a * (cube.sideLength / 2) + plane.b * 0 + plane.c * 0 = plane.d)
  (h2 : plane.a * 0 + plane.b * 0 + plane.c * (cube.sideLength / 8) = plane.d)
  (h3 : plane.a * cube.sideLength + plane.b * cube.sideLength + plane.c * (cube.sideLength / 8) = plane.d) :
  ∃ (smallerVolume : ℝ), smallerVolume = (25 / 192) * cube.sideLength ^ 3 := by
  sorry

end NUMINAMATH_CALUDE_cube_volume_partition_l3216_321675


namespace NUMINAMATH_CALUDE_count_solutions_l3216_321631

open Complex

/-- The number of complex solutions to e^z = (z + i) / (z - i) with |z| < 20 -/
def num_solutions : ℕ := 14

/-- The equation e^z = (z + i) / (z - i) -/
def satisfies_equation (z : ℂ) : Prop :=
  exp z = (z + I) / (z - I)

/-- The condition |z| < 20 -/
def within_bounds (z : ℂ) : Prop :=
  abs z < 20

theorem count_solutions :
  ∃ (S : Finset ℂ), S.card = num_solutions ∧
    (∀ z ∈ S, satisfies_equation z ∧ within_bounds z) ∧
    (∀ z : ℂ, satisfies_equation z ∧ within_bounds z → z ∈ S) :=
sorry

end NUMINAMATH_CALUDE_count_solutions_l3216_321631


namespace NUMINAMATH_CALUDE_inscribed_circle_radius_l3216_321690

/-- An isosceles triangle with an inscribed circle -/
structure IsoscelesTriangleWithInscribedCircle where
  /-- The length of the equal sides of the isosceles triangle -/
  a : ℝ
  /-- The radius of the inscribed circle -/
  r : ℝ
  /-- The area of the triangle -/
  area : ℝ
  /-- The ratio of AN to AB, where N is the point where a line parallel to BC 
      and tangent to the inscribed circle intersects AB -/
  an_ratio : ℝ
  /-- Condition that the triangle is isosceles -/
  isosceles : a > 0
  /-- Condition that AN = 3/8 * AB -/
  an_condition : an_ratio = 3/8
  /-- Condition that the area of the triangle is 12 -/
  area_condition : area = 12

/-- Theorem: If the conditions are met, the radius of the inscribed circle is 3/2 -/
theorem inscribed_circle_radius 
  (t : IsoscelesTriangleWithInscribedCircle) : t.r = 3/2 := by
  sorry

end NUMINAMATH_CALUDE_inscribed_circle_radius_l3216_321690


namespace NUMINAMATH_CALUDE_fruit_problem_equations_l3216_321678

/-- Represents the ancient Chinese fruit problem --/
structure FruitProblem where
  totalFruits : ℕ
  totalCost : ℕ
  bitterFruitCount : ℕ
  bitterFruitCost : ℕ
  sweetFruitCount : ℕ
  sweetFruitCost : ℕ

/-- The system of equations for the fruit problem --/
def fruitEquations (p : FruitProblem) (x y : ℚ) : Prop :=
  x + y = p.totalFruits ∧
  (4 / 7 : ℚ) * x + (11 / 9 : ℚ) * y = p.totalCost

/-- Theorem stating that the given system of equations correctly represents the fruit problem --/
theorem fruit_problem_equations (p : FruitProblem) 
  (h1 : p.totalFruits = 1000)
  (h2 : p.totalCost = 999)
  (h3 : p.bitterFruitCount = 7)
  (h4 : p.bitterFruitCost = 4)
  (h5 : p.sweetFruitCount = 9)
  (h6 : p.sweetFruitCost = 11) :
  ∃ x y : ℚ, fruitEquations p x y :=
sorry

end NUMINAMATH_CALUDE_fruit_problem_equations_l3216_321678


namespace NUMINAMATH_CALUDE_deposit_calculation_l3216_321658

theorem deposit_calculation (remaining_amount : ℝ) (deposit_percentage : ℝ) : 
  remaining_amount = 1260 ∧ deposit_percentage = 0.1 → 
  (remaining_amount / (1 - deposit_percentage)) * deposit_percentage = 140 := by
sorry

end NUMINAMATH_CALUDE_deposit_calculation_l3216_321658


namespace NUMINAMATH_CALUDE_two_numbers_sum_and_quotient_l3216_321698

theorem two_numbers_sum_and_quotient (x y : ℝ) : 
  x > 0 → y > 0 → x + y = 432 → y / x = 5 → x = 72 ∧ y = 360 := by
  sorry

end NUMINAMATH_CALUDE_two_numbers_sum_and_quotient_l3216_321698


namespace NUMINAMATH_CALUDE_tulip_area_l3216_321679

/-- Given a flower bed with roses and tulips, calculate the area occupied by tulips -/
theorem tulip_area (total_area : Real) (rose_fraction : Real) (tulip_fraction : Real) 
  (h1 : total_area = 2.4)
  (h2 : rose_fraction = 1/3)
  (h3 : tulip_fraction = 1/4) :
  tulip_fraction * (total_area - rose_fraction * total_area) = 0.4 := by
  sorry

#check tulip_area

end NUMINAMATH_CALUDE_tulip_area_l3216_321679


namespace NUMINAMATH_CALUDE_employment_agency_payroll_l3216_321644

/-- Calculates the total payroll for an employment agency --/
theorem employment_agency_payroll
  (total_hired : ℕ)
  (num_laborers : ℕ)
  (operator_pay : ℕ)
  (laborer_pay : ℕ)
  (h_total : total_hired = 35)
  (h_laborers : num_laborers = 19)
  (h_operator_pay : operator_pay = 140)
  (h_laborer_pay : laborer_pay = 90) :
  let num_operators := total_hired - num_laborers
  let operator_total := num_operators * operator_pay
  let laborer_total := num_laborers * laborer_pay
  operator_total + laborer_total = 3950 := by
  sorry

#check employment_agency_payroll

end NUMINAMATH_CALUDE_employment_agency_payroll_l3216_321644


namespace NUMINAMATH_CALUDE_greatest_digit_sum_base9_under_2500_l3216_321677

/-- Represents a positive integer in base 9 --/
structure Base9 where
  digits : List Nat
  positive : digits ≠ []
  valid : ∀ d ∈ digits, d < 9

/-- Converts a Base9 number to its decimal representation --/
def toDecimal (n : Base9) : Nat := sorry

/-- Computes the sum of digits of a Base9 number --/
def digitSum (n : Base9) : Nat := sorry

theorem greatest_digit_sum_base9_under_2500 :
  (∃ (n : Base9), toDecimal n < 2500 ∧ digitSum n = 24) ∧
  (∀ (m : Base9), toDecimal m < 2500 → digitSum m ≤ 24) := by
  sorry

end NUMINAMATH_CALUDE_greatest_digit_sum_base9_under_2500_l3216_321677


namespace NUMINAMATH_CALUDE_at_least_one_equation_has_two_roots_l3216_321686

theorem at_least_one_equation_has_two_roots (p q₁ q₂ : ℝ) (h : p = q₁ + q₂ + 1) :
  (∃ x y : ℝ, x ≠ y ∧ x^2 + x + q₁ = 0 ∧ y^2 + y + q₁ = 0) ∨
  (∃ x y : ℝ, x ≠ y ∧ x^2 + p*x + q₂ = 0 ∧ y^2 + p*y + q₂ = 0) :=
by sorry

end NUMINAMATH_CALUDE_at_least_one_equation_has_two_roots_l3216_321686


namespace NUMINAMATH_CALUDE_cos_squared_alpha_minus_pi_fourth_l3216_321627

theorem cos_squared_alpha_minus_pi_fourth (α : ℝ) 
  (h : Real.sin (2 * α) = 1 / 3) : 
  Real.cos (α - π / 4) ^ 2 = 2 / 3 := by
  sorry

end NUMINAMATH_CALUDE_cos_squared_alpha_minus_pi_fourth_l3216_321627


namespace NUMINAMATH_CALUDE_potato_rows_count_l3216_321697

/-- Represents the farmer's crop situation -/
structure FarmCrops where
  corn_rows : ℕ
  potato_rows : ℕ
  corn_per_row : ℕ
  potatoes_per_row : ℕ
  intact_crops : ℕ

/-- Theorem stating the number of potato rows given the problem conditions -/
theorem potato_rows_count (farm : FarmCrops)
    (h_corn_rows : farm.corn_rows = 10)
    (h_corn_per_row : farm.corn_per_row = 9)
    (h_potatoes_per_row : farm.potatoes_per_row = 30)
    (h_intact_crops : farm.intact_crops = 120)
    (h_half_destroyed : farm.intact_crops = (farm.corn_rows * farm.corn_per_row + farm.potato_rows * farm.potatoes_per_row) / 2) :
  farm.potato_rows = 2 := by
  sorry


end NUMINAMATH_CALUDE_potato_rows_count_l3216_321697


namespace NUMINAMATH_CALUDE_binomial_square_constant_l3216_321681

theorem binomial_square_constant (a : ℝ) : 
  (∃ b : ℝ, ∀ x : ℝ, 16 * x^2 + 40 * x + a = (4 * x + b)^2) → a = 25 := by
  sorry

end NUMINAMATH_CALUDE_binomial_square_constant_l3216_321681


namespace NUMINAMATH_CALUDE_diophantine_equation_7z_squared_l3216_321629

theorem diophantine_equation_7z_squared (x y z : ℕ) : 
  x^2 + y^2 = 7 * z^2 → x = 0 ∧ y = 0 ∧ z = 0 := by
  sorry

end NUMINAMATH_CALUDE_diophantine_equation_7z_squared_l3216_321629


namespace NUMINAMATH_CALUDE_violinists_count_l3216_321617

/-- Represents the number of people playing each instrument in an orchestra -/
structure Orchestra where
  total : ℕ
  drums : ℕ
  trombone : ℕ
  trumpet : ℕ
  frenchHorn : ℕ
  cello : ℕ
  contrabass : ℕ
  clarinet : ℕ
  flute : ℕ
  maestro : ℕ

/-- Calculates the number of violinists in the orchestra -/
def violinists (o : Orchestra) : ℕ :=
  o.total - (o.drums + o.trombone + o.trumpet + o.frenchHorn + o.cello + o.contrabass + o.clarinet + o.flute + o.maestro)

/-- Theorem stating that the number of violinists in the given orchestra is 3 -/
theorem violinists_count (o : Orchestra) 
  (h1 : o.total = 21)
  (h2 : o.drums = 1)
  (h3 : o.trombone = 4)
  (h4 : o.trumpet = 2)
  (h5 : o.frenchHorn = 1)
  (h6 : o.cello = 1)
  (h7 : o.contrabass = 1)
  (h8 : o.clarinet = 3)
  (h9 : o.flute = 4)
  (h10 : o.maestro = 1) :
  violinists o = 3 := by
  sorry


end NUMINAMATH_CALUDE_violinists_count_l3216_321617


namespace NUMINAMATH_CALUDE_equation_to_general_form_l3216_321633

theorem equation_to_general_form :
  ∀ x : ℝ, (2 * x^2 - 1 = 6 * x) ↔ (2 * x^2 - 6 * x - 1 = 0) :=
by
  sorry

end NUMINAMATH_CALUDE_equation_to_general_form_l3216_321633


namespace NUMINAMATH_CALUDE_watch_payment_l3216_321657

theorem watch_payment (original_price : ℚ) (discount_rate : ℚ) (dime_value : ℚ) (quarter_value : ℚ) :
  original_price = 15 →
  discount_rate = 1/5 →
  dime_value = 1/10 →
  quarter_value = 1/4 →
  ∃ (num_dimes num_quarters : ℕ),
    (num_dimes : ℚ) = 2 * (num_quarters : ℚ) ∧
    (original_price * (1 - discount_rate) = dime_value * num_dimes + quarter_value * num_quarters) ∧
    num_dimes = 52 :=
by sorry

end NUMINAMATH_CALUDE_watch_payment_l3216_321657


namespace NUMINAMATH_CALUDE_bean_garden_rows_l3216_321612

/-- Given a garden with bean plants arranged in rows and columns,
    prove that with 15 columns and 780 total plants, there are 52 rows. -/
theorem bean_garden_rows (total_plants : ℕ) (columns : ℕ) (rows : ℕ) : 
  total_plants = 780 → columns = 15 → total_plants = rows * columns → rows = 52 := by
  sorry

end NUMINAMATH_CALUDE_bean_garden_rows_l3216_321612


namespace NUMINAMATH_CALUDE_mans_age_twice_sons_l3216_321643

theorem mans_age_twice_sons (son_age : ℕ) (age_difference : ℕ) : 
  son_age = 26 → age_difference = 28 → 
  ∃ (years : ℕ), (son_age + years + age_difference) = 2 * (son_age + years) ∧ years = 2 :=
by sorry

end NUMINAMATH_CALUDE_mans_age_twice_sons_l3216_321643


namespace NUMINAMATH_CALUDE_exterior_angle_smaller_implies_obtuse_l3216_321601

/-- A triangle with vertices A, B, and C -/
structure Triangle where
  A : Point
  B : Point
  C : Point

/-- Predicate to check if a triangle is obtuse -/
def is_obtuse_triangle (t : Triangle) : Prop := sorry

/-- Predicate to check if an exterior angle is smaller than its adjacent interior angle -/
def exterior_angle_smaller_than_interior (t : Triangle) : Prop := sorry

/-- Theorem: If an exterior angle of a triangle is smaller than its adjacent interior angle, 
    then the triangle is obtuse -/
theorem exterior_angle_smaller_implies_obtuse (t : Triangle) :
  exterior_angle_smaller_than_interior t → is_obtuse_triangle t := by
  sorry

end NUMINAMATH_CALUDE_exterior_angle_smaller_implies_obtuse_l3216_321601


namespace NUMINAMATH_CALUDE_amusement_park_trip_distance_l3216_321688

/-- Calculates the distance traveled given speed and time -/
def distance (speed : ℝ) (time : ℝ) : ℝ := speed * time

/-- The total distance covered by Amanda and her friends -/
theorem amusement_park_trip_distance : 
  let d1 := distance 40 1.5
  let d2 := distance 50 1
  let d3 := distance 30 2.25
  d1 + d2 + d3 = 177.5 := by sorry

end NUMINAMATH_CALUDE_amusement_park_trip_distance_l3216_321688


namespace NUMINAMATH_CALUDE_price_difference_l3216_321653

/-- The original price of Liz's old car -/
def original_price : ℝ := 32500

/-- The selling price of Liz's old car as a percentage of the original price -/
def selling_percentage : ℝ := 0.80

/-- The additional amount Liz needs to buy the new car -/
def additional_amount : ℝ := 4000

/-- The price of the new car -/
def new_car_price : ℝ := 30000

/-- The theorem stating the difference between the original price of the old car and the price of the new car -/
theorem price_difference : original_price - new_car_price = 2500 := by
  sorry

end NUMINAMATH_CALUDE_price_difference_l3216_321653


namespace NUMINAMATH_CALUDE_library_books_calculation_l3216_321699

/-- Proves that given a library with a capacity of 400 books, if adding 240 books
    makes it 90% full, then the initial number of books is 120. -/
theorem library_books_calculation (capacity : ℕ) (to_buy : ℕ) (final_percentage : ℚ)
    (h1 : capacity = 400)
    (h2 : to_buy = 240)
    (h3 : final_percentage = 9/10)
    (h4 : (initial_books + to_buy : ℚ) / capacity = final_percentage) :
    initial_books = 120 := by
  sorry

#check library_books_calculation

end NUMINAMATH_CALUDE_library_books_calculation_l3216_321699


namespace NUMINAMATH_CALUDE_emily_dresses_l3216_321651

theorem emily_dresses (melissa : ℕ) (debora : ℕ) (emily : ℕ) : 
  debora = melissa + 12 →
  melissa = emily / 2 →
  melissa + debora + emily = 44 →
  emily = 16 := by sorry

end NUMINAMATH_CALUDE_emily_dresses_l3216_321651


namespace NUMINAMATH_CALUDE_prime_product_theorem_l3216_321672

def largest_one_digit_prime : ℕ := 7
def second_largest_one_digit_prime : ℕ := 5
def second_largest_two_digit_prime : ℕ := 89

theorem prime_product_theorem :
  largest_one_digit_prime * second_largest_one_digit_prime * second_largest_two_digit_prime = 3115 := by
  sorry

end NUMINAMATH_CALUDE_prime_product_theorem_l3216_321672


namespace NUMINAMATH_CALUDE_largest_divisor_of_four_consecutive_odd_integers_l3216_321665

theorem largest_divisor_of_four_consecutive_odd_integers (n : ℤ) : 
  ∃ (d : ℤ), d > 0 ∧ 
  (∀ (k : ℤ), (d ∣ (2*k-3)*(2*k-1)*(2*k+1)*(2*k+3))) ∧ 
  (∀ (m : ℤ), m > d → ∃ (l : ℤ), ¬(m ∣ (2*l-3)*(2*l-1)*(2*l+1)*(2*l+3))) →
  d = 3 :=
by sorry

end NUMINAMATH_CALUDE_largest_divisor_of_four_consecutive_odd_integers_l3216_321665


namespace NUMINAMATH_CALUDE_prob_both_counterfeit_given_at_least_one_l3216_321666

def total_banknotes : ℕ := 20
def counterfeit_banknotes : ℕ := 5
def selected_banknotes : ℕ := 2

def prob_both_counterfeit : ℚ := (counterfeit_banknotes.choose 2) / (total_banknotes.choose 2)
def prob_at_least_one_counterfeit : ℚ := 
  ((counterfeit_banknotes.choose 2) + (counterfeit_banknotes.choose 1) * ((total_banknotes - counterfeit_banknotes).choose 1)) / 
  (total_banknotes.choose 2)

theorem prob_both_counterfeit_given_at_least_one : 
  prob_both_counterfeit / prob_at_least_one_counterfeit = 2 / 17 := by sorry

end NUMINAMATH_CALUDE_prob_both_counterfeit_given_at_least_one_l3216_321666


namespace NUMINAMATH_CALUDE_max_value_expression_l3216_321608

theorem max_value_expression (a b c d : ℝ) 
  (h_pos : a > 0 ∧ b > 0 ∧ c > 0 ∧ d > 0) 
  (h_sum : a + b + c + d ≤ 4) :
  (a * (b + 2 * c)) ^ (1/4) + 
  (b * (c + 2 * d)) ^ (1/4) + 
  (c * (d + 2 * a)) ^ (1/4) + 
  (d * (a + 2 * b)) ^ (1/4) ≤ 4 * 3 ^ (1/4) := by
sorry

end NUMINAMATH_CALUDE_max_value_expression_l3216_321608


namespace NUMINAMATH_CALUDE_matt_points_l3216_321685

/-- Calculates the total points scored in basketball given the number of successful 2-point and 3-point shots -/
def total_points (two_point_shots : ℕ) (three_point_shots : ℕ) : ℕ :=
  2 * two_point_shots + 3 * three_point_shots

/-- Theorem stating that four 2-point shots and two 3-point shots result in 14 points -/
theorem matt_points : total_points 4 2 = 14 := by
  sorry

end NUMINAMATH_CALUDE_matt_points_l3216_321685


namespace NUMINAMATH_CALUDE_triangle_properties_l3216_321655

/-- Theorem about properties of an acute triangle ABC --/
theorem triangle_properties 
  (A B C : Real) -- Angles of the triangle
  (a b c : Real) -- Sides of the triangle opposite to A, B, C respectively
  (h_acute : A > 0 ∧ B > 0 ∧ C > 0 ∧ A + B + C = π) -- Triangle is acute
  (h_sine : Real.sqrt 3 * a = 2 * c * Real.sin A) -- Given condition
  (h_side : a = 2) -- Given side length
  (h_area : (1/2) * a * c * Real.sin B = (3 * Real.sqrt 3) / 2) -- Given area
  : C = π/3 ∧ c = Real.sqrt 7 := by
  sorry


end NUMINAMATH_CALUDE_triangle_properties_l3216_321655


namespace NUMINAMATH_CALUDE_max_m_value_l3216_321664

theorem max_m_value (p q : ℝ → Prop) (m : ℝ) : 
  (∀ x, p x ↔ (x^2 - 4*x - 5 > 0)) →
  (∀ x, q x ↔ (x^2 - 2*x + 1 - m^2 > 0)) →
  (m > 0) →
  (∀ x, p x → q x) →
  (∃ x, q x ∧ ¬(p x)) →
  (∀ m' > m, ∃ x, p x ∧ ¬(q x)) →
  m = 2 :=
by sorry

end NUMINAMATH_CALUDE_max_m_value_l3216_321664


namespace NUMINAMATH_CALUDE_geometric_sequence_third_term_l3216_321600

/-- A geometric sequence with a negative common ratio -/
def GeometricSequence (a : ℕ → ℝ) : Prop :=
  ∃ q : ℝ, q < 0 ∧ ∀ n : ℕ, a (n + 1) = a n * q

theorem geometric_sequence_third_term
  (a : ℕ → ℝ)
  (h_geo : GeometricSequence a)
  (h_first : a 1 = 2)
  (h_relation : a 3 - 4 = a 2) :
  a 3 = 2 := by
  sorry


end NUMINAMATH_CALUDE_geometric_sequence_third_term_l3216_321600


namespace NUMINAMATH_CALUDE_first_discount_percentage_l3216_321660

theorem first_discount_percentage (original_price : ℝ) (final_price : ℝ) 
  (second_discount : ℝ) (first_discount : ℝ) : 
  original_price = 200 →
  final_price = 152 →
  second_discount = 0.05 →
  final_price = original_price * (1 - first_discount) * (1 - second_discount) →
  first_discount = 0.20 := by
  sorry

#check first_discount_percentage

end NUMINAMATH_CALUDE_first_discount_percentage_l3216_321660


namespace NUMINAMATH_CALUDE_calculate_interest_rate_l3216_321639

/-- Calculate the rate of interest per annum given the principal, time, and simple interest -/
theorem calculate_interest_rate (principal time simple_interest : ℝ) :
  principal > 0 →
  time > 0 →
  simple_interest > 0 →
  principal = 6693.75 →
  time = 5 →
  simple_interest = 4016.25 →
  (simple_interest * 100) / (principal * time) = 12 := by
sorry

end NUMINAMATH_CALUDE_calculate_interest_rate_l3216_321639


namespace NUMINAMATH_CALUDE_prove_num_sodas_l3216_321687

def sandwich_cost : ℚ := 149/100
def soda_cost : ℚ := 87/100
def total_cost : ℚ := 646/100
def num_sandwiches : ℕ := 2

def num_sodas : ℕ := 4

theorem prove_num_sodas : 
  sandwich_cost * num_sandwiches + soda_cost * num_sodas = total_cost := by
  sorry

#eval num_sodas

end NUMINAMATH_CALUDE_prove_num_sodas_l3216_321687


namespace NUMINAMATH_CALUDE_tim_pencils_l3216_321663

theorem tim_pencils (tyrah_pencils : ℕ) (sarah_pencils : ℕ) (tim_pencils : ℕ)
  (h1 : tyrah_pencils = 6 * sarah_pencils)
  (h2 : tim_pencils = 8 * sarah_pencils)
  (h3 : tyrah_pencils = 12) :
  tim_pencils = 16 := by
sorry

end NUMINAMATH_CALUDE_tim_pencils_l3216_321663


namespace NUMINAMATH_CALUDE_matrix_inverse_equality_l3216_321695

/-- Given a 3x3 matrix B with a variable d in the (2,3) position, prove that if B^(-1) = k * B, then d = 13/9 and k = -329/52 -/
theorem matrix_inverse_equality (d k : ℚ) : 
  let B : Matrix (Fin 3) (Fin 3) ℚ := !![1, 2, 3; 4, 5, d; 6, 7, 8]
  (B⁻¹ = k • B) → (d = 13/9 ∧ k = -329/52) := by
  sorry

end NUMINAMATH_CALUDE_matrix_inverse_equality_l3216_321695


namespace NUMINAMATH_CALUDE_prism_surface_area_l3216_321604

/-- A right square prism with all vertices on the surface of a sphere -/
structure PrismOnSphere where
  -- The diameter of the sphere
  sphere_diameter : ℝ
  -- The side length of the base of the prism
  base_side_length : ℝ
  -- The height of the prism
  height : ℝ
  -- All vertices are on the sphere surface
  vertices_on_sphere : sphere_diameter^2 = base_side_length^2 + base_side_length^2 + height^2

/-- The surface area of a right square prism -/
def surface_area (p : PrismOnSphere) : ℝ :=
  2 * p.base_side_length^2 + 4 * p.base_side_length * p.height

/-- Theorem: The surface area of the specific prism is 2 + 4√2 -/
theorem prism_surface_area :
  ∃ (p : PrismOnSphere),
    p.sphere_diameter = 2 ∧
    p.base_side_length = 1 ∧
    surface_area p = 2 + 4 * Real.sqrt 2 :=
by sorry

end NUMINAMATH_CALUDE_prism_surface_area_l3216_321604


namespace NUMINAMATH_CALUDE_factorization_xy_squared_l3216_321689

theorem factorization_xy_squared (x y : ℝ) : x^2*y + x*y^2 = x*y*(x + y) := by
  sorry

end NUMINAMATH_CALUDE_factorization_xy_squared_l3216_321689


namespace NUMINAMATH_CALUDE_total_soccer_balls_donated_l3216_321694

-- Define the given conditions
def soccer_balls_per_class : ℕ := 5
def number_of_schools : ℕ := 2
def elementary_classes_per_school : ℕ := 4
def middle_classes_per_school : ℕ := 5

-- Define the theorem
theorem total_soccer_balls_donated : 
  soccer_balls_per_class * number_of_schools * (elementary_classes_per_school + middle_classes_per_school) = 90 := by
  sorry


end NUMINAMATH_CALUDE_total_soccer_balls_donated_l3216_321694


namespace NUMINAMATH_CALUDE_system_solution_l3216_321647

-- Define the system of equations
def system (x y : ℝ) : Prop :=
  x + 3 * y + 14 ≤ 0 ∧
  x^4 + 2 * x^2 * y^2 + y^4 + 64 - 20 * x^2 - 20 * y^2 = 8 * x * y

-- Theorem stating that the solution to the system is (-2, -4)
theorem system_solution :
  ∃! p : ℝ × ℝ, system p.1 p.2 ∧ p = (-2, -4) := by
  sorry


end NUMINAMATH_CALUDE_system_solution_l3216_321647


namespace NUMINAMATH_CALUDE_quadratic_equation_properties_specific_root_condition_l3216_321684

/-- Represents a quadratic equation of the form x^2 + 2(m+1)x + m^2 - 1 = 0 -/
def quadratic_equation (m : ℝ) (x : ℝ) : Prop :=
  x^2 + 2*(m+1)*x + m^2 - 1 = 0

/-- The discriminant of the quadratic equation -/
def discriminant (m : ℝ) : ℝ :=
  8*m + 8

/-- Condition for the roots of the quadratic equation -/
def root_condition (x₁ x₂ : ℝ) : Prop :=
  (x₁ - x₂)^2 = 16 - x₁*x₂

theorem quadratic_equation_properties (m : ℝ) :
  (∃ x₁ x₂ : ℝ, quadratic_equation m x₁ ∧ quadratic_equation m x₂ ∧ x₁ ≠ x₂) ↔ m ≥ -1 :=
sorry

theorem specific_root_condition (m : ℝ) :
  (∃ x₁ x₂ : ℝ, quadratic_equation m x₁ ∧ quadratic_equation m x₂ ∧ 
   x₁ ≠ x₂ ∧ root_condition x₁ x₂) → m = 1 :=
sorry

end NUMINAMATH_CALUDE_quadratic_equation_properties_specific_root_condition_l3216_321684


namespace NUMINAMATH_CALUDE_three_integer_chords_l3216_321630

/-- Represents a circle with a given radius and a point at a given distance from its center -/
structure CircleWithPoint where
  radius : ℝ
  distanceFromCenter : ℝ

/-- Counts the number of integer-length chords containing the given point -/
def countIntegerChords (c : CircleWithPoint) : ℕ :=
  sorry

theorem three_integer_chords :
  let c := CircleWithPoint.mk 13 5
  countIntegerChords c = 3 := by
  sorry

end NUMINAMATH_CALUDE_three_integer_chords_l3216_321630


namespace NUMINAMATH_CALUDE_complex_equation_solution_l3216_321610

theorem complex_equation_solution (z : ℂ) : z * (1 - I) = 3 - I → z = 2 + I := by
  sorry

end NUMINAMATH_CALUDE_complex_equation_solution_l3216_321610


namespace NUMINAMATH_CALUDE_fraction_inequality_l3216_321668

theorem fraction_inequality (a b c : ℝ) (h1 : a > b) (h2 : b > 0) (h3 : c > 0) :
  a / (a + c) > b / (b + c) := by
  sorry

end NUMINAMATH_CALUDE_fraction_inequality_l3216_321668


namespace NUMINAMATH_CALUDE_marys_friends_l3216_321632

theorem marys_friends (total_stickers : ℕ) (stickers_per_friend : ℕ) (stickers_per_other : ℕ) 
  (stickers_left : ℕ) (total_students : ℕ) :
  total_stickers = 50 →
  stickers_per_friend = 4 →
  stickers_per_other = 2 →
  stickers_left = 8 →
  total_students = 17 →
  ∃ (num_friends : ℕ),
    num_friends * stickers_per_friend + 
    (total_students - 1 - num_friends) * stickers_per_other + 
    stickers_left = total_stickers ∧
    num_friends = 5 := by
  sorry

end NUMINAMATH_CALUDE_marys_friends_l3216_321632


namespace NUMINAMATH_CALUDE_dad_caught_more_trouts_l3216_321676

/-- The number of trouts Caleb caught -/
def caleb_trouts : ℕ := 2

/-- The number of trouts Caleb's dad caught -/
def dad_trouts : ℕ := 3 * caleb_trouts

/-- The difference in trouts caught between Caleb's dad and Caleb -/
def trout_difference : ℕ := dad_trouts - caleb_trouts

theorem dad_caught_more_trouts : trout_difference = 4 := by
  sorry

end NUMINAMATH_CALUDE_dad_caught_more_trouts_l3216_321676


namespace NUMINAMATH_CALUDE_length_of_DH_l3216_321646

-- Define the triangle and points
structure Triangle :=
  (A B C D E F G H : ℝ × ℝ)

-- Define the properties of the triangle and points
def EquilateralTriangle (t : Triangle) : Prop :=
  let d := Real.sqrt 3
  t.A = (0, 0) ∧ t.B = (2, 0) ∧ t.C = (1, d)

def PointsOnSides (t : Triangle) : Prop :=
  ∃ x y z w : ℝ,
    0 ≤ x ∧ x ≤ 2 ∧
    0 ≤ y ∧ y ≤ 2 ∧
    0 ≤ z ∧ z ≤ 2 ∧
    0 ≤ w ∧ w ≤ 2 ∧
    t.D = (x, 0) ∧
    t.F = (y, 0) ∧
    t.E = (1 - z/2, z * Real.sqrt 3 / 2) ∧
    t.G = (1 - w/2, w * Real.sqrt 3 / 2)

def ParallelLines (t : Triangle) : Prop :=
  (t.E.2 - t.D.2) / (t.E.1 - t.D.1) = Real.sqrt 3 ∧
  (t.G.2 - t.F.2) / (t.G.1 - t.F.1) = Real.sqrt 3

def SpecificLengths (t : Triangle) : Prop :=
  t.D.1 - t.A.1 = 0.5 ∧
  Real.sqrt ((t.E.1 - t.D.1)^2 + (t.E.2 - t.D.2)^2) = 1 ∧
  t.F.1 - t.D.1 = 0.5 ∧
  Real.sqrt ((t.G.1 - t.F.1)^2 + (t.G.2 - t.F.2)^2) = 1 ∧
  t.B.1 - t.F.1 = 0.5

def ParallelDH (t : Triangle) : Prop :=
  ∃ k : ℝ, t.H = (k * t.C.1 + (1 - k) * t.A.1, k * t.C.2 + (1 - k) * t.A.2)

-- State the theorem
theorem length_of_DH (t : Triangle) :
  EquilateralTriangle t →
  PointsOnSides t →
  ParallelLines t →
  SpecificLengths t →
  ParallelDH t →
  Real.sqrt ((t.H.1 - t.D.1)^2 + (t.H.2 - t.D.2)^2) = 1 :=
by sorry

end NUMINAMATH_CALUDE_length_of_DH_l3216_321646


namespace NUMINAMATH_CALUDE_meaningful_expression_l3216_321636

theorem meaningful_expression (x : ℝ) : 
  (∃ y : ℝ, y = 1 / Real.sqrt (x + 2)) ↔ x > -2 := by sorry

end NUMINAMATH_CALUDE_meaningful_expression_l3216_321636


namespace NUMINAMATH_CALUDE_absolute_value_inequality_solution_set_l3216_321614

theorem absolute_value_inequality_solution_set : 
  {x : ℝ | |x| > -1} = Set.univ :=
sorry

end NUMINAMATH_CALUDE_absolute_value_inequality_solution_set_l3216_321614


namespace NUMINAMATH_CALUDE_dot_product_sum_l3216_321606

theorem dot_product_sum (a b : ℝ × ℝ × ℝ) (h1 : a = (0, 2, 0)) (h2 : b = (1, 0, -1)) :
  (a.1 + b.1, a.2.1 + b.2.1, a.2.2 + b.2.2) • b = 2 := by
  sorry

end NUMINAMATH_CALUDE_dot_product_sum_l3216_321606


namespace NUMINAMATH_CALUDE_boys_average_score_l3216_321656

theorem boys_average_score (num_boys num_girls : ℕ) (girls_avg class_avg : ℝ) :
  num_boys = 12 →
  num_girls = 4 →
  girls_avg = 92 →
  class_avg = 86 →
  (num_boys * (class_avg * (num_boys + num_girls) - num_girls * girls_avg)) / (num_boys * (num_boys + num_girls)) = 84 :=
by sorry

end NUMINAMATH_CALUDE_boys_average_score_l3216_321656


namespace NUMINAMATH_CALUDE_largest_geometric_three_digit_l3216_321623

/-- Represents a three-digit number as a tuple of its digits -/
def ThreeDigitNumber := (Nat × Nat × Nat)

/-- Checks if all digits in a ThreeDigitNumber are distinct -/
def distinct_digits (n : ThreeDigitNumber) : Prop :=
  n.1 ≠ n.2.1 ∧ n.1 ≠ n.2.2 ∧ n.2.1 ≠ n.2.2

/-- Checks if the digits of a ThreeDigitNumber form a geometric sequence -/
def geometric_sequence (n : ThreeDigitNumber) : Prop :=
  ∃ r : Rat, r ≠ 0 ∧ n.2.1 = n.1 * r ∧ n.2.2 = n.2.1 * r

/-- Checks if a ThreeDigitNumber has no zero digits -/
def no_zero_digits (n : ThreeDigitNumber) : Prop :=
  n.1 ≠ 0 ∧ n.2.1 ≠ 0 ∧ n.2.2 ≠ 0

/-- Converts a ThreeDigitNumber to its integer representation -/
def to_int (n : ThreeDigitNumber) : Nat :=
  100 * n.1 + 10 * n.2.1 + n.2.2

/-- The main theorem stating that 842 is the largest number satisfying all conditions -/
theorem largest_geometric_three_digit :
  ∀ n : ThreeDigitNumber,
    distinct_digits n ∧ 
    geometric_sequence n ∧ 
    no_zero_digits n →
    to_int n ≤ 842 :=
  sorry

end NUMINAMATH_CALUDE_largest_geometric_three_digit_l3216_321623


namespace NUMINAMATH_CALUDE_exam_candidates_count_l3216_321619

theorem exam_candidates_count : ∀ (x : ℝ), 
  (0.07 * x = 0.06 * x + 80) →
  x = 8000 := by
  sorry

end NUMINAMATH_CALUDE_exam_candidates_count_l3216_321619


namespace NUMINAMATH_CALUDE_total_cost_of_materials_l3216_321625

/-- The total cost of materials for a construction company -/
theorem total_cost_of_materials
  (gravel_quantity : ℝ)
  (gravel_price : ℝ)
  (sand_quantity : ℝ)
  (sand_price : ℝ)
  (h1 : gravel_quantity = 5.91)
  (h2 : gravel_price = 30.50)
  (h3 : sand_quantity = 8.11)
  (h4 : sand_price = 40.50) :
  gravel_quantity * gravel_price + sand_quantity * sand_price = 508.71 := by
  sorry

end NUMINAMATH_CALUDE_total_cost_of_materials_l3216_321625


namespace NUMINAMATH_CALUDE_parabola_equation_part1_parabola_equation_part2_l3216_321624

-- Part 1
theorem parabola_equation_part1 (a b c : ℝ) (h : a ≠ 0) :
  (∀ x y : ℝ, y = a * x^2 + b * x + c) →
  (1, 10) = (- b / (2 * a), a * (- b / (2 * a))^2 + b * (- b / (2 * a)) + c) →
  (-1, -2) = (-1, a * (-1)^2 + b * (-1) + c) →
  (∀ x y : ℝ, y = -3 * (x - 1)^2 + 10) := by sorry

-- Part 2
theorem parabola_equation_part2 (a b c : ℝ) (h : a ≠ 0) :
  (∀ x y : ℝ, y = a * x^2 + b * x + c) →
  (0 = a * (-1)^2 + b * (-1) + c) →
  (0 = a * 3^2 + b * 3 + c) →
  (3 = c) →
  (∀ x y : ℝ, y = -x^2 + 2 * x + 3) := by sorry

end NUMINAMATH_CALUDE_parabola_equation_part1_parabola_equation_part2_l3216_321624


namespace NUMINAMATH_CALUDE_discrete_rv_distribution_l3216_321674

/-- A discrete random variable with two possible values -/
structure DiscreteRV where
  x₁ : ℝ
  x₂ : ℝ
  p₁ : ℝ
  h₁ : x₂ > x₁
  h₂ : p₁ = 0.6
  h₃ : p₁ * x₁ + (1 - p₁) * x₂ = 1.4  -- Expected value
  h₄ : p₁ * (x₁ - 1.4)^2 + (1 - p₁) * (x₂ - 1.4)^2 = 0.24  -- Variance

/-- The probability distribution of the discrete random variable -/
def probability_distribution (X : DiscreteRV) : Prop :=
  X.x₁ = 1 ∧ X.x₂ = 2

theorem discrete_rv_distribution (X : DiscreteRV) :
  probability_distribution X := by
  sorry

end NUMINAMATH_CALUDE_discrete_rv_distribution_l3216_321674


namespace NUMINAMATH_CALUDE_trapezoid_segment_length_l3216_321648

-- Define a trapezoid PQRS
structure Trapezoid :=
  (P Q R S : ℝ × ℝ)

-- Define the length function
def length (A B : ℝ × ℝ) : ℝ := sorry

-- Define the area of a triangle
def area_triangle (A B C : ℝ × ℝ) : ℝ := sorry

theorem trapezoid_segment_length (PQRS : Trapezoid) :
  length PQRS.P PQRS.S + length PQRS.R PQRS.Q = 270 →
  area_triangle PQRS.P PQRS.Q PQRS.R / area_triangle PQRS.P PQRS.S PQRS.R = 5 / 4 →
  length PQRS.P PQRS.S = 150 := by
  sorry

end NUMINAMATH_CALUDE_trapezoid_segment_length_l3216_321648


namespace NUMINAMATH_CALUDE_smallest_prime_sum_of_three_primes_l3216_321654

-- Define a function to check if a number is prime
def isPrime (n : Nat) : Prop := n > 1 ∧ ∀ d : Nat, d > 1 → d < n → ¬(n % d = 0)

-- Define a function to check if a number is the sum of three different primes
def isSumOfThreeDifferentPrimes (n : Nat) : Prop :=
  ∃ (p q r : Nat), isPrime p ∧ isPrime q ∧ isPrime r ∧ p ≠ q ∧ q ≠ r ∧ p ≠ r ∧ p + q + r = n

-- State the theorem
theorem smallest_prime_sum_of_three_primes :
  isPrime 19 ∧ 
  isSumOfThreeDifferentPrimes 19 ∧ 
  ∀ n : Nat, n < 19 → ¬(isPrime n ∧ isSumOfThreeDifferentPrimes n) :=
by sorry

end NUMINAMATH_CALUDE_smallest_prime_sum_of_three_primes_l3216_321654


namespace NUMINAMATH_CALUDE_remainder_problem_l3216_321652

theorem remainder_problem : 123456789012 % 252 = 144 := by
  sorry

end NUMINAMATH_CALUDE_remainder_problem_l3216_321652


namespace NUMINAMATH_CALUDE_newspaper_cost_difference_l3216_321615

/-- Calculates the annual cost difference between Juanita's newspaper purchases and Grant's subscription --/
theorem newspaper_cost_difference : 
  let grant_base_cost : ℝ := 200
  let grant_loyalty_discount : ℝ := 0.1
  let grant_summer_discount : ℝ := 0.05
  let juanita_mon_wed_price : ℝ := 0.5
  let juanita_thu_fri_price : ℝ := 0.6
  let juanita_sat_price : ℝ := 0.8
  let juanita_sun_price : ℝ := 3
  let juanita_monthly_coupon : ℝ := 0.25
  let juanita_holiday_surcharge : ℝ := 0.5
  let weeks_per_year : ℕ := 52
  let months_per_year : ℕ := 12
  let summer_months : ℕ := 2

  let grant_annual_cost := grant_base_cost * (1 - grant_loyalty_discount) - 
    (grant_base_cost / months_per_year) * summer_months * grant_summer_discount

  let juanita_weekly_cost := 3 * juanita_mon_wed_price + 2 * juanita_thu_fri_price + 
    juanita_sat_price + juanita_sun_price

  let juanita_annual_cost := juanita_weekly_cost * weeks_per_year - 
    juanita_monthly_coupon * months_per_year + juanita_holiday_surcharge * months_per_year

  juanita_annual_cost - grant_annual_cost = 162.5 := by sorry

end NUMINAMATH_CALUDE_newspaper_cost_difference_l3216_321615


namespace NUMINAMATH_CALUDE_aq_length_is_112_over_35_l3216_321667

/-- Represents a triangle with side lengths a, b, and c -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Represents an inscribed right triangle within another triangle -/
structure InscribedRightTriangle where
  outer : Triangle
  pc : ℝ
  bp : ℝ
  cq : ℝ

/-- The length of AQ in the described configuration -/
def aq_length (t : InscribedRightTriangle) : ℝ :=
  -- Definition of aq_length goes here
  sorry

/-- Theorem stating that AQ = 112/35 in the given configuration -/
theorem aq_length_is_112_over_35 :
  let t : InscribedRightTriangle := {
    outer := { a := 6, b := 7, c := 8 },
    pc := 4,
    bp := 3,
    cq := 3
  }
  aq_length t = 112 / 35 := by sorry

end NUMINAMATH_CALUDE_aq_length_is_112_over_35_l3216_321667


namespace NUMINAMATH_CALUDE_peaches_at_stand_l3216_321650

/-- The total number of peaches at the stand after picking more is equal to the sum of the initial number of peaches and the number of peaches picked. -/
theorem peaches_at_stand (initial_peaches picked_peaches : ℕ) :
  initial_peaches + picked_peaches = initial_peaches + picked_peaches :=
by sorry

end NUMINAMATH_CALUDE_peaches_at_stand_l3216_321650


namespace NUMINAMATH_CALUDE_equal_angles_not_always_opposite_l3216_321692

-- Define the basic geometric concepts
variable (Line : Type) (Point : Type) (Angle : Type)
variable (opposite : Angle → Angle → Prop)
variable (equal : Angle → Angle → Prop)
variable (perpendicular : Line → Line → Prop)
variable (parallel : Line → Line → Prop)
variable (corresponding : Angle → Angle → Prop)

-- State the propositions
axiom opposite_angles_equal : ∀ (a b : Angle), opposite a b → equal a b
axiom perpendicular_lines_parallel : ∀ (l1 l2 l3 : Line), perpendicular l1 l3 → perpendicular l2 l3 → parallel l1 l2
axiom corresponding_angles_equal : ∀ (a b : Angle), corresponding a b → equal a b

-- State the theorem to be proved
theorem equal_angles_not_always_opposite : ¬(∀ (a b : Angle), equal a b → opposite a b) :=
sorry

end NUMINAMATH_CALUDE_equal_angles_not_always_opposite_l3216_321692


namespace NUMINAMATH_CALUDE_only_one_correct_probability_l3216_321620

theorem only_one_correct_probability (p_a p_b : ℝ) : 
  p_a = 1/5 → p_b = 1/4 → 
  p_a * (1 - p_b) + (1 - p_a) * p_b = 7/20 := by
  sorry

end NUMINAMATH_CALUDE_only_one_correct_probability_l3216_321620


namespace NUMINAMATH_CALUDE_robot_tracing_time_l3216_321680

/-- Represents a rectangular grid with width and height -/
structure Grid where
  width : ℕ
  height : ℕ

/-- Calculates the total length of lines in a grid -/
def totalLength (g : Grid) : ℕ :=
  (g.width + 1) * g.height + (g.height + 1) * g.width

/-- Represents the robot's tracing speed in grid units per minute -/
def robotSpeed (g : Grid) (time : ℚ) : ℚ :=
  (totalLength g : ℚ) / time

theorem robot_tracing_time 
  (g1 g2 : Grid) 
  (t1 : ℚ) 
  (hg1 : g1 = ⟨3, 7⟩) 
  (hg2 : g2 = ⟨5, 5⟩) 
  (ht1 : t1 = 26) :
  robotSpeed g1 t1 * (totalLength g2 : ℚ) = 30 := by
  sorry

end NUMINAMATH_CALUDE_robot_tracing_time_l3216_321680


namespace NUMINAMATH_CALUDE_molecular_weight_h2o_is_18_l3216_321640

/-- The molecular weight of dihydrogen monoxide in grams per mole -/
def molecular_weight_h2o : ℝ := 18

/-- The number of moles of dihydrogen monoxide -/
def moles_h2o : ℝ := 7

/-- The total weight of dihydrogen monoxide in grams -/
def total_weight_h2o : ℝ := 126

/-- Theorem: The molecular weight of dihydrogen monoxide is 18 grams per mole -/
theorem molecular_weight_h2o_is_18 :
  molecular_weight_h2o = total_weight_h2o / moles_h2o :=
by sorry

end NUMINAMATH_CALUDE_molecular_weight_h2o_is_18_l3216_321640


namespace NUMINAMATH_CALUDE_edmund_earns_64_dollars_l3216_321613

/-- Calculates the amount Edmund earns for extra chores over two weeks -/
def edmunds_earnings (normal_chores_per_week : ℕ) (chores_per_day : ℕ) 
  (days : ℕ) (payment_per_extra_chore : ℕ) : ℕ :=
  let total_chores := chores_per_day * days
  let normal_total_chores := normal_chores_per_week * (days / 7)
  let extra_chores := total_chores - normal_total_chores
  extra_chores * payment_per_extra_chore

/-- Theorem stating that Edmund earns $64 for extra chores over two weeks -/
theorem edmund_earns_64_dollars :
  edmunds_earnings 12 4 14 2 = 64 := by
  sorry


end NUMINAMATH_CALUDE_edmund_earns_64_dollars_l3216_321613


namespace NUMINAMATH_CALUDE_parking_lot_cars_remaining_l3216_321621

theorem parking_lot_cars_remaining (initial_cars : ℕ) 
  (first_group_left : ℕ) (second_group_left : ℕ) : 
  initial_cars = 24 → first_group_left = 8 → second_group_left = 6 →
  initial_cars - first_group_left - second_group_left = 10 := by
  sorry

end NUMINAMATH_CALUDE_parking_lot_cars_remaining_l3216_321621


namespace NUMINAMATH_CALUDE_quadratic_roots_condition_l3216_321691

theorem quadratic_roots_condition (m : ℝ) :
  (∃ x₁ x₂ : ℝ, x₁ > 0 ∧ x₂ > 0 ∧ 
   x₁^2 - (m+2)*x₁ + 1 = 0 ∧ 
   x₂^2 - (m+2)*x₂ + 1 = 0 ∧ 
   x₁ ≠ x₂) →
  m ≥ 0 := by
sorry

end NUMINAMATH_CALUDE_quadratic_roots_condition_l3216_321691


namespace NUMINAMATH_CALUDE_line_segment_endpoint_l3216_321669

/-- Given a line segment in the Cartesian plane with midpoint (2020, 11), 
    one endpoint at (a, 0), and the other endpoint on the line y = x, 
    prove that a = 4018 -/
theorem line_segment_endpoint (a : ℝ) : 
  (∃ t : ℝ, (a + t) / 2 = 2020 ∧ t / 2 = 11 ∧ t = t) → a = 4018 := by
  sorry

end NUMINAMATH_CALUDE_line_segment_endpoint_l3216_321669


namespace NUMINAMATH_CALUDE_algebraic_expression_value_l3216_321696

theorem algebraic_expression_value (x y : ℝ) (h : x - y - 3 = 0) :
  x^2 - y^2 - 6*y = 9 := by sorry

end NUMINAMATH_CALUDE_algebraic_expression_value_l3216_321696


namespace NUMINAMATH_CALUDE_turban_price_turban_price_proof_l3216_321682

/-- The price of a turban given the following conditions:
  - The total salary for one year is Rs. 90 plus one turban
  - The servant works for 9 months (3/4 of a year)
  - The servant receives Rs. 60 plus the turban for 9 months of work
-/
theorem turban_price : ℝ :=
  let yearly_salary : ℝ → ℝ := λ t => 90 + t
  let worked_fraction : ℝ := 3 / 4
  let received_salary : ℝ → ℝ := λ t => 60 + t
  30

theorem turban_price_proof (t : ℝ) : 
  (let yearly_salary : ℝ → ℝ := λ t => 90 + t
   let worked_fraction : ℝ := 3 / 4
   let received_salary : ℝ → ℝ := λ t => 60 + t
   worked_fraction * yearly_salary t = received_salary t) →
  t = 30 := by
sorry

end NUMINAMATH_CALUDE_turban_price_turban_price_proof_l3216_321682


namespace NUMINAMATH_CALUDE_floor_pi_minus_e_l3216_321611

theorem floor_pi_minus_e : ⌊π - Real.exp 1⌋ = 0 := by sorry

end NUMINAMATH_CALUDE_floor_pi_minus_e_l3216_321611


namespace NUMINAMATH_CALUDE_inequalities_satisfied_l3216_321626

theorem inequalities_satisfied
  (x y z : ℝ) (a b c : ℕ)
  (hx : x ≥ 0) (hy : y ≥ 0) (hz : z ≥ 0)
  (hxa : x < a) (hyb : y < b) (hzc : z < c) :
  (x * y + y * z + z * x < a * b + b * c + c * a) ∧
  (x^2 + y^2 + z^2 < a^2 + b^2 + c^2) ∧
  (x * y * z < a * b * c) := by
  sorry

end NUMINAMATH_CALUDE_inequalities_satisfied_l3216_321626


namespace NUMINAMATH_CALUDE_probability_theorem_l3216_321671

/-- Represents the number of guests -/
def num_guests : ℕ := 4

/-- Represents the number of roll types -/
def num_roll_types : ℕ := 4

/-- Represents the number of each roll type prepared -/
def rolls_per_type : ℕ := 3

/-- Represents the total number of rolls -/
def total_rolls : ℕ := num_roll_types * rolls_per_type

/-- Represents the number of rolls each guest receives -/
def rolls_per_guest : ℕ := num_roll_types

/-- Calculates the probability of each guest receiving one roll of each type -/
def probability_one_of_each : ℚ :=
  (rolls_per_type ^ num_roll_types * (rolls_per_type - 1) ^ num_roll_types * (rolls_per_type - 2) ^ num_roll_types) /
  (Nat.choose total_rolls rolls_per_guest * Nat.choose (total_rolls - rolls_per_guest) rolls_per_guest * Nat.choose (total_rolls - 2*rolls_per_guest) rolls_per_guest)

theorem probability_theorem :
  probability_one_of_each = 12 / 321 :=
by sorry

end NUMINAMATH_CALUDE_probability_theorem_l3216_321671


namespace NUMINAMATH_CALUDE_matrix_A_properties_l3216_321607

/-- The line l: 2x - y = 3 -/
def line_l (x y : ℝ) : Prop := 2 * x - y = 3

/-- The transformation matrix A -/
def matrix_A : Matrix (Fin 2) (Fin 2) ℝ :=
  ![![-1, 1],
    ![-4, 3]]

/-- The inverse of matrix A -/
def matrix_A_inv : Matrix (Fin 2) (Fin 2) ℝ :=
  ![![3, -1],
    ![4, -1]]

/-- The transformation σ maps the line l onto itself -/
def transformation_preserves_line (A : Matrix (Fin 2) (Fin 2) ℝ) : Prop :=
  ∀ x y : ℝ, line_l x y → line_l (A 0 0 * x + A 0 1 * y) (A 1 0 * x + A 1 1 * y)

theorem matrix_A_properties :
  transformation_preserves_line matrix_A ∧
  matrix_A * matrix_A_inv = 1 ∧
  matrix_A_inv * matrix_A = 1 := by
  sorry

end NUMINAMATH_CALUDE_matrix_A_properties_l3216_321607


namespace NUMINAMATH_CALUDE_scientific_notation_equality_l3216_321645

theorem scientific_notation_equality : 122254 = 1.22254 * (10 ^ 5) := by
  sorry

end NUMINAMATH_CALUDE_scientific_notation_equality_l3216_321645


namespace NUMINAMATH_CALUDE_subtract_negatives_example_l3216_321628

theorem subtract_negatives_example : (-3) - (-5) = 2 := by
  sorry

end NUMINAMATH_CALUDE_subtract_negatives_example_l3216_321628


namespace NUMINAMATH_CALUDE_prime_factor_puzzle_l3216_321622

theorem prime_factor_puzzle (a b c d w x y z : ℕ) : 
  w.Prime → x.Prime → y.Prime → z.Prime →
  w < x → x < y → y < z →
  (w^a) * (x^b) * (y^c) * (z^d) = 660 →
  (a + b) - (c + d) = 1 →
  b = 1 := by sorry

end NUMINAMATH_CALUDE_prime_factor_puzzle_l3216_321622

import Mathlib

namespace NUMINAMATH_CALUDE_modified_star_angle_sum_l574_57496

/-- A modified n-pointed star --/
structure ModifiedStar where
  n : ℕ
  is_valid : n ≥ 6

/-- The sum of interior angles of the modified star --/
def interior_angle_sum (star : ModifiedStar) : ℝ :=
  180 * (star.n - 2)

/-- Theorem: The sum of interior angles of a modified n-pointed star is 180(n-2) degrees --/
theorem modified_star_angle_sum (star : ModifiedStar) :
  interior_angle_sum star = 180 * (star.n - 2) := by
  sorry

end NUMINAMATH_CALUDE_modified_star_angle_sum_l574_57496


namespace NUMINAMATH_CALUDE_sum_of_exponential_equality_l574_57465

theorem sum_of_exponential_equality (a b : ℝ) (h : (2 : ℝ) ^ b = (2 : ℝ) ^ (6 - a)) : a + b = 6 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_exponential_equality_l574_57465


namespace NUMINAMATH_CALUDE_binomial_15_12_l574_57402

theorem binomial_15_12 : Nat.choose 15 12 = 455 := by
  sorry

end NUMINAMATH_CALUDE_binomial_15_12_l574_57402


namespace NUMINAMATH_CALUDE_geometric_sequence_inequality_l574_57415

/-- Given a geometric sequence with positive terms and common ratio not equal to 1,
    prove that the arithmetic mean of the 3rd and 9th terms is greater than
    the geometric mean of the 5th and 7th terms. -/
theorem geometric_sequence_inequality (a : ℕ → ℝ) (q : ℝ) :
  (∀ n, a n > 0) →  -- All terms are positive
  q ≠ 1 →           -- Common ratio is not 1
  (∀ n, a (n + 1) = q * a n) →  -- Geometric sequence property
  (a 3 + a 9) / 2 > Real.sqrt (a 5 * a 7) := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_inequality_l574_57415


namespace NUMINAMATH_CALUDE_hot_dog_consumption_l574_57491

theorem hot_dog_consumption (x : ℕ) : 
  x + (x + 2) + (x + 4) = 36 → x = 10 := by
  sorry

end NUMINAMATH_CALUDE_hot_dog_consumption_l574_57491


namespace NUMINAMATH_CALUDE_cos_2alpha_plus_pi_3_l574_57464

theorem cos_2alpha_plus_pi_3 (α : Real) 
  (h : Real.sin (π / 6 - α) - Real.cos α = 1 / 3) : 
  Real.cos (2 * α + π / 3) = 7 / 9 := by
  sorry

end NUMINAMATH_CALUDE_cos_2alpha_plus_pi_3_l574_57464


namespace NUMINAMATH_CALUDE_length_of_AB_l574_57455

-- Define the hyperbola
def hyperbola (x y : ℝ) : Prop := x^2 / 3 - y^2 / 6 = 1

-- Define the line with slope tan(30°) passing through (3, 0)
def line (x y : ℝ) : Prop := y = (Real.sqrt 3 / 3) * (x - 3)

-- Define the intersection points A and B
def intersection_points (A B : ℝ × ℝ) : Prop :=
  hyperbola A.1 A.2 ∧ hyperbola B.1 B.2 ∧
  line A.1 A.2 ∧ line B.1 B.2 ∧
  A ≠ B

-- Theorem statement
theorem length_of_AB (A B : ℝ × ℝ) :
  intersection_points A B →
  Real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2) = (16 / 5) * Real.sqrt 3 :=
sorry

end NUMINAMATH_CALUDE_length_of_AB_l574_57455


namespace NUMINAMATH_CALUDE_area_of_triangle_DCE_l574_57434

/-- Given a rectangle BDEF with AB = 24 and EF = 15, and triangle BCE with area 60,
    prove that the area of triangle DCE is 30 -/
theorem area_of_triangle_DCE (AB EF : ℝ) (area_BCE : ℝ) :
  AB = 24 →
  EF = 15 →
  area_BCE = 60 →
  let BC := (2 * area_BCE) / EF
  let DC := EF - BC
  let DE := (2 * area_BCE) / BC
  (1/2) * DC * DE = 30 := by
  sorry

end NUMINAMATH_CALUDE_area_of_triangle_DCE_l574_57434


namespace NUMINAMATH_CALUDE_quadratic_roots_opposite_l574_57421

theorem quadratic_roots_opposite (k : ℝ) : 
  (∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ 
    x₁^2 + (k^2 - 4)*x₁ + k - 1 = 0 ∧
    x₂^2 + (k^2 - 4)*x₂ + k - 1 = 0 ∧
    x₁ = -x₂) →
  k = -2 := by
sorry

end NUMINAMATH_CALUDE_quadratic_roots_opposite_l574_57421


namespace NUMINAMATH_CALUDE_circular_cross_section_solids_l574_57451

-- Define the geometric solids
inductive GeometricSolid
  | Cube
  | Cylinder
  | Cone
  | TriangularPrism

-- Define a predicate for having a circular cross-section
def has_circular_cross_section (solid : GeometricSolid) : Prop :=
  match solid with
  | GeometricSolid.Cylinder => true
  | GeometricSolid.Cone => true
  | _ => false

-- Theorem statement
theorem circular_cross_section_solids :
  ∀ (solid : GeometricSolid),
    has_circular_cross_section solid ↔
      (solid = GeometricSolid.Cylinder ∨ solid = GeometricSolid.Cone) :=
by sorry

end NUMINAMATH_CALUDE_circular_cross_section_solids_l574_57451


namespace NUMINAMATH_CALUDE_absolute_value_inequality_solution_set_l574_57413

theorem absolute_value_inequality_solution_set :
  {x : ℝ | |x + 1| > 3} = {x : ℝ | x < -4 ∨ x > 2} := by
  sorry

end NUMINAMATH_CALUDE_absolute_value_inequality_solution_set_l574_57413


namespace NUMINAMATH_CALUDE_solve_equation_l574_57414

theorem solve_equation : ∃ y : ℚ, (2 * y + 3 * y = 500 - (4 * y + 6 * y)) ∧ y = 100 / 3 := by
  sorry

end NUMINAMATH_CALUDE_solve_equation_l574_57414


namespace NUMINAMATH_CALUDE_problem_1_problem_2_problem_3_problem_4_l574_57419

-- Problem 1
theorem problem_1 : -3 + 8 - 7 - 15 = -17 := by sorry

-- Problem 2
theorem problem_2 : 23 - 6 * (-3) + 2 * (-4) = 33 := by sorry

-- Problem 3
theorem problem_3 : -8 / (4/5) * (-2/3) = 20/3 := by sorry

-- Problem 4
theorem problem_4 : -(2^2) - 9 * ((-1/3)^2) + |(-4)| = -1 := by sorry

end NUMINAMATH_CALUDE_problem_1_problem_2_problem_3_problem_4_l574_57419


namespace NUMINAMATH_CALUDE_quadratic_root_value_l574_57485

theorem quadratic_root_value (m : ℝ) : 
  (∃ x : ℝ, x^2 - 2*x + m = 0 ∧ x = 1) → m = 1 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_root_value_l574_57485


namespace NUMINAMATH_CALUDE_marks_deck_cost_l574_57408

/-- The total cost of a rectangular deck with sealant -/
def deck_cost (length width base_cost sealant_cost : ℝ) : ℝ :=
  let area := length * width
  let total_cost_per_sqft := base_cost + sealant_cost
  area * total_cost_per_sqft

/-- Theorem: The cost of Mark's deck is $4800 -/
theorem marks_deck_cost :
  deck_cost 30 40 3 1 = 4800 := by
  sorry

end NUMINAMATH_CALUDE_marks_deck_cost_l574_57408


namespace NUMINAMATH_CALUDE_sqrt_300_approximation_l574_57448

theorem sqrt_300_approximation (ε δ : ℝ) (ε_pos : ε > 0) (δ_pos : δ > 0) 
  (h : |Real.sqrt 3 - 1.732| < δ) : 
  |Real.sqrt 300 - 17.32| < ε := by
  sorry

end NUMINAMATH_CALUDE_sqrt_300_approximation_l574_57448


namespace NUMINAMATH_CALUDE_v_closed_under_multiplication_l574_57458

def v : Set ℕ := {n : ℕ | ∃ m : ℕ, m > 0 ∧ n = m^3}

theorem v_closed_under_multiplication :
  ∀ a b : ℕ, a ∈ v → b ∈ v → (a * b) ∈ v :=
by sorry

end NUMINAMATH_CALUDE_v_closed_under_multiplication_l574_57458


namespace NUMINAMATH_CALUDE_oplus_four_two_l574_57471

def oplus (a b : ℝ) : ℝ := 4 * a + 3 * b

theorem oplus_four_two : oplus 4 2 = 22 := by
  sorry

end NUMINAMATH_CALUDE_oplus_four_two_l574_57471


namespace NUMINAMATH_CALUDE_incorrect_multiplication_result_l574_57483

theorem incorrect_multiplication_result 
  (x : ℝ) 
  (h1 : ∃ a b : ℕ, 987 * x = 500000 + 10000 * a + 700 + b / 100 + 0.0989999999)
  (h2 : 987 * x ≠ 555707.2899999999)
  (h3 : 555707.2899999999 = 987 * x) : 
  987 * x = 598707.2989999999 := by
sorry

end NUMINAMATH_CALUDE_incorrect_multiplication_result_l574_57483


namespace NUMINAMATH_CALUDE_modular_arithmetic_problem_l574_57479

theorem modular_arithmetic_problem :
  ∃ (a b : ℤ), 
    (7 * a) % 72 = 1 ∧ 
    (13 * b) % 72 = 1 ∧ 
    ((3 * a + 9 * b) % 72) % 72 = 6 :=
by sorry

end NUMINAMATH_CALUDE_modular_arithmetic_problem_l574_57479


namespace NUMINAMATH_CALUDE_fibonacci_problem_l574_57490

theorem fibonacci_problem (x : ℕ) (h : x > 0) :
  (10 : ℝ) / x = 40 / (x + 6) →
  ∃ (y : ℕ), y > 0 ∧
    (10 : ℝ) / x = 10 / y ∧
    40 / (x + 6) = 40 / (y + 6) ∧
    (10 : ℝ) / y = 40 / (y + 6) :=
by sorry

end NUMINAMATH_CALUDE_fibonacci_problem_l574_57490


namespace NUMINAMATH_CALUDE_triangle_perimeter_with_tangent_circles_l574_57417

-- Define the circle type
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

-- Define the triangle type
structure Triangle where
  vertices : (ℝ × ℝ) × (ℝ × ℝ) × (ℝ × ℝ)

-- Define a function to check if circles are tangent to each other
def areTangent (c1 c2 : Circle) : Prop :=
  let (x1, y1) := c1.center
  let (x2, y2) := c2.center
  (x2 - x1)^2 + (y2 - y1)^2 = (c1.radius + c2.radius)^2

-- Define a function to check if a circle is tangent to two sides of a triangle
def isTangentToTriangleSides (c : Circle) (t : Triangle) : Prop :=
  sorry -- Implementation details omitted for brevity

-- Theorem statement
theorem triangle_perimeter_with_tangent_circles 
  (X Y Z : Circle) (DEF : Triangle) :
  X.radius = 2 ∧ Y.radius = 2 ∧ Z.radius = 2 →
  areTangent X Y ∧ areTangent Y Z ∧ areTangent Z X →
  isTangentToTriangleSides X DEF ∧ 
  isTangentToTriangleSides Y DEF ∧ 
  isTangentToTriangleSides Z DEF →
  let (D, E, F) := DEF.vertices
  let perimeter := Real.sqrt ((D.1 - E.1)^2 + (D.2 - E.2)^2) +
                   Real.sqrt ((E.1 - F.1)^2 + (E.2 - F.2)^2) +
                   Real.sqrt ((F.1 - D.1)^2 + (F.2 - D.2)^2)
  perimeter = 12 * Real.sqrt 3 :=
by
  sorry -- Proof omitted

end NUMINAMATH_CALUDE_triangle_perimeter_with_tangent_circles_l574_57417


namespace NUMINAMATH_CALUDE_tea_sales_revenue_l574_57446

/-- Represents the sales data for tea leaves over two years -/
structure TeaSalesData where
  price_ratio : ℝ  -- Ratio of this year's price to last year's
  yield_this_year : ℝ  -- Yield in kg this year
  yield_difference : ℝ  -- Difference in yield compared to last year
  revenue_increase : ℝ  -- Increase in revenue compared to last year

/-- Calculates the sales revenue for this year given the tea sales data -/
def calculate_revenue (data : TeaSalesData) : ℝ :=
  let yield_last_year := data.yield_this_year + data.yield_difference
  let revenue_last_year := yield_last_year
  revenue_last_year + data.revenue_increase

/-- Theorem stating that given the specific conditions, the sales revenue this year is 9930 yuan -/
theorem tea_sales_revenue 
  (data : TeaSalesData)
  (h1 : data.price_ratio = 10)
  (h2 : data.yield_this_year = 198.6)
  (h3 : data.yield_difference = 87.4)
  (h4 : data.revenue_increase = 8500) :
  calculate_revenue data = 9930 := by
  sorry

#eval calculate_revenue ⟨10, 198.6, 87.4, 8500⟩

end NUMINAMATH_CALUDE_tea_sales_revenue_l574_57446


namespace NUMINAMATH_CALUDE_circle_theorem_l574_57450

-- Define a circle
def Circle : Type := {p : ℝ × ℝ // ∃ (center : ℝ × ℝ) (radius : ℝ), (p.1 - center.1)^2 + (p.2 - center.2)^2 = radius^2}

-- Define points on the circle
variable (ω₁ : Circle)
variable (A B C D : Circle)

-- Define the order of points on the circle
def InOrder (A C B D : Circle) : Prop := sorry

-- Define the distance between two points
def Distance (p q : Circle) : ℝ := sorry

-- Define the midpoint of an arc
def IsMidpointOfArc (M A B : Circle) : Prop := sorry

-- The main theorem
theorem circle_theorem (h_order : InOrder A C B D) :
  (Distance C D)^2 = (Distance A C) * (Distance B C) + (Distance A D) * (Distance B D) ↔
  (IsMidpointOfArc C A B ∨ IsMidpointOfArc D A B) :=
sorry

end NUMINAMATH_CALUDE_circle_theorem_l574_57450


namespace NUMINAMATH_CALUDE_line_parallel_to_intersection_of_parallel_planes_l574_57494

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the parallel relation between lines and planes
variable (parallelLinePlane : Line → Plane → Prop)

-- Define the parallel relation between lines
variable (parallelLine : Line → Line → Prop)

-- Define the intersection of two planes
variable (intersection : Plane → Plane → Line)

-- Theorem statement
theorem line_parallel_to_intersection_of_parallel_planes
  (a b : Line) (α β : Plane)
  (h1 : parallelLinePlane a α)
  (h2 : parallelLinePlane a β)
  (h3 : intersection α β = b) :
  parallelLine a b :=
sorry

end NUMINAMATH_CALUDE_line_parallel_to_intersection_of_parallel_planes_l574_57494


namespace NUMINAMATH_CALUDE_range_of_sum_of_reciprocals_l574_57489

theorem range_of_sum_of_reciprocals (x y : ℝ) (hx : x > 0) (hy : y > 0) 
  (h : x + 4*y + 1/x + 1/y = 10) : 
  1 ≤ 1/x + 1/y ∧ 1/x + 1/y ≤ 9 := by
  sorry

end NUMINAMATH_CALUDE_range_of_sum_of_reciprocals_l574_57489


namespace NUMINAMATH_CALUDE_isabel_ds_games_l574_57478

theorem isabel_ds_games (initial_games : ℕ) (remaining_games : ℕ) (given_games : ℕ) : 
  initial_games = 90 → remaining_games = 3 → given_games = initial_games - remaining_games → given_games = 87 := by
  sorry

end NUMINAMATH_CALUDE_isabel_ds_games_l574_57478


namespace NUMINAMATH_CALUDE_fraction_simplification_l574_57412

theorem fraction_simplification (a b : ℝ) (x : ℝ) (h1 : x = a / b) (h2 : a ≠ b) (h3 : b ≠ 0) :
  (a + b) / (a - b) = (x + 1) / (x - 1) := by
  sorry

end NUMINAMATH_CALUDE_fraction_simplification_l574_57412


namespace NUMINAMATH_CALUDE_f_min_value_l574_57463

/-- The function f(x) = |x + 3| + |x + 6| + |x + 8| + |x + 10| -/
def f (x : ℝ) : ℝ := |x + 3| + |x + 6| + |x + 8| + |x + 10|

/-- Theorem stating that f(x) has a minimum value of 9 at x = -8 -/
theorem f_min_value :
  (∀ x : ℝ, f x ≥ 9) ∧ f (-8) = 9 := by sorry

end NUMINAMATH_CALUDE_f_min_value_l574_57463


namespace NUMINAMATH_CALUDE_solution_set_is_open_interval_l574_57449

-- Define the function f
variable (f : ℝ → ℝ)

-- Define the properties of f
def increasing_f := ∀ x y, x < y → f x < f y
def f_zero_is_neg_one := f 0 = -1
def f_three_is_one := f 3 = 1

-- Define the solution set
def solution_set (f : ℝ → ℝ) := {x : ℝ | |f x| < 1}

-- State the theorem
theorem solution_set_is_open_interval
  (h_increasing : increasing_f f)
  (h_zero : f_zero_is_neg_one f)
  (h_three : f_three_is_one f) :
  solution_set f = Set.Ioo 0 3 :=
sorry

end NUMINAMATH_CALUDE_solution_set_is_open_interval_l574_57449


namespace NUMINAMATH_CALUDE_pieces_from_rod_l574_57423

/-- The number of pieces of a given length that can be cut from a rod. -/
def number_of_pieces (rod_length_m : ℕ) (piece_length_cm : ℕ) : ℕ :=
  (rod_length_m * 100) / piece_length_cm

/-- Theorem: The number of 85 cm pieces that can be cut from a 34-meter rod is 40. -/
theorem pieces_from_rod : number_of_pieces 34 85 = 40 := by
  sorry

end NUMINAMATH_CALUDE_pieces_from_rod_l574_57423


namespace NUMINAMATH_CALUDE_julie_can_print_100_newspapers_l574_57400

/-- The number of boxes of paper Julie bought -/
def boxes : ℕ := 2

/-- The number of packages in each box -/
def packages_per_box : ℕ := 5

/-- The number of sheets in each package -/
def sheets_per_package : ℕ := 250

/-- The number of sheets required to print one newspaper -/
def sheets_per_newspaper : ℕ := 25

/-- The total number of newspapers Julie can print -/
def newspapers_printed : ℕ := 
  (boxes * packages_per_box * sheets_per_package) / sheets_per_newspaper

theorem julie_can_print_100_newspapers : newspapers_printed = 100 := by
  sorry

end NUMINAMATH_CALUDE_julie_can_print_100_newspapers_l574_57400


namespace NUMINAMATH_CALUDE_sin_cos_sum_equivalent_l574_57409

theorem sin_cos_sum_equivalent (x : ℝ) : 
  Real.sin (3 * x) + Real.cos (3 * x) = Real.sqrt 2 * Real.sin (3 * (x + π / 12)) := by
  sorry

end NUMINAMATH_CALUDE_sin_cos_sum_equivalent_l574_57409


namespace NUMINAMATH_CALUDE_extended_twelve_basketball_conference_games_l574_57406

/-- Calculates the number of games in a basketball conference with specific rules --/
def conference_games (teams_per_division : ℕ) (divisions : ℕ) (intra_division_games : ℕ) : ℕ :=
  let total_teams := teams_per_division * divisions
  let games_per_team := (teams_per_division - 1) * intra_division_games + teams_per_division * (divisions - 1)
  total_teams * games_per_team / 2

/-- Theorem stating the number of games in the Extended Twelve Basketball Conference --/
theorem extended_twelve_basketball_conference_games :
  conference_games 8 2 3 = 232 := by
  sorry

end NUMINAMATH_CALUDE_extended_twelve_basketball_conference_games_l574_57406


namespace NUMINAMATH_CALUDE_geometric_sequence_inequality_l574_57474

/-- A geometric sequence with positive terms and common ratio not equal to 1 -/
structure GeometricSequence where
  a : ℕ → ℝ
  q : ℝ
  h_positive : ∀ n, a n > 0
  h_q_positive : q > 0
  h_q_not_one : q ≠ 1
  h_geometric : ∀ n, a (n + 1) = q * a n

/-- 
For a geometric sequence with positive terms and common ratio q where q > 0 and q ≠ 1,
a_n + a_{n+3} > a_{n+1} + a_{n+2} for all n
-/
theorem geometric_sequence_inequality (seq : GeometricSequence) :
  ∀ n, seq.a n + seq.a (n + 3) > seq.a (n + 1) + seq.a (n + 2) :=
sorry

end NUMINAMATH_CALUDE_geometric_sequence_inequality_l574_57474


namespace NUMINAMATH_CALUDE_intersection_of_A_and_B_l574_57428

-- Define set A
def A : Set ℝ := {x : ℝ | |x| ≤ 1}

-- Define set B
def B : Set ℝ := {y : ℝ | ∃ x : ℝ, y = x^2}

-- Theorem statement
theorem intersection_of_A_and_B :
  A ∩ B = {x : ℝ | 0 ≤ x ∧ x ≤ 1} :=
by sorry

end NUMINAMATH_CALUDE_intersection_of_A_and_B_l574_57428


namespace NUMINAMATH_CALUDE_cylinder_in_sphere_volume_l574_57459

theorem cylinder_in_sphere_volume (r h R : ℝ) (hr : r = 4) (hR : R = 7) 
  (hh : h^2 = 180) : 
  (4/3 * π * R^3 - π * r^2 * h) = (728/3) * π := by
  sorry

end NUMINAMATH_CALUDE_cylinder_in_sphere_volume_l574_57459


namespace NUMINAMATH_CALUDE_new_person_weight_l574_57495

/-- The weight of the new person given the conditions of the problem -/
theorem new_person_weight (n : ℕ) (weight_increase : ℝ) (replaced_weight : ℝ) :
  n = 15 ∧ weight_increase = 3.8 ∧ replaced_weight = 75 →
  n * weight_increase + replaced_weight = 132 := by
sorry

end NUMINAMATH_CALUDE_new_person_weight_l574_57495


namespace NUMINAMATH_CALUDE_disjunction_true_l574_57457

theorem disjunction_true : 
  (∀ x : ℝ, x < 0 → 2^x > x) ∨ (∃ x : ℝ, x^2 + x + 1 < 0) := by sorry

end NUMINAMATH_CALUDE_disjunction_true_l574_57457


namespace NUMINAMATH_CALUDE_odd_functions_identification_l574_57401

-- Define a general function type
def RealFunction := ℝ → ℝ

-- Define what it means for a function to be odd
def IsOdd (f : RealFunction) : Prop := ∀ x : ℝ, f (-x) = -f x

-- Define the given functions
def F1 (f : RealFunction) : RealFunction := fun x ↦ -|f x|
def F2 (f : RealFunction) : RealFunction := fun x ↦ x * f (x^2)
def F3 (f : RealFunction) : RealFunction := fun x ↦ -f (-x)
def F4 (f : RealFunction) : RealFunction := fun x ↦ f x - f (-x)

-- State the theorem
theorem odd_functions_identification (f : RealFunction) :
  ¬IsOdd (F1 f) ∧ IsOdd (F2 f) ∧ IsOdd (F4 f) :=
sorry

end NUMINAMATH_CALUDE_odd_functions_identification_l574_57401


namespace NUMINAMATH_CALUDE_vector_dot_product_equality_iff_collinear_l574_57425

theorem vector_dot_product_equality_iff_collinear 
  {n : Type*} [NormedAddCommGroup n] [InnerProductSpace ℝ n] 
  (a b : n) (ha : a ≠ 0) (hb : b ≠ 0) :
  |inner a b| = ‖a‖ * ‖b‖ ↔ ∃ (t : ℝ), a = t • b :=
sorry

end NUMINAMATH_CALUDE_vector_dot_product_equality_iff_collinear_l574_57425


namespace NUMINAMATH_CALUDE_cross_product_example_l574_57456

/-- The cross product of two 3D vectors -/
def cross_product (u v : ℝ × ℝ × ℝ) : ℝ × ℝ × ℝ :=
  (u.2.1 * v.2.2 - u.2.2 * v.2.1,
   u.2.2 * v.1 - u.1 * v.2.2,
   u.1 * v.2.1 - u.2.1 * v.1)

theorem cross_product_example :
  let u : ℝ × ℝ × ℝ := (3, 2, 4)
  let v : ℝ × ℝ × ℝ := (4, 3, -1)
  cross_product u v = (-14, 19, 1) := by
  sorry

end NUMINAMATH_CALUDE_cross_product_example_l574_57456


namespace NUMINAMATH_CALUDE_parallel_planes_theorem_l574_57453

structure Plane :=
  (p : Set (ℝ × ℝ × ℝ))

structure Line :=
  (l : Set (ℝ × ℝ × ℝ))

def perpendicular (l : Line) (p : Plane) : Prop :=
  sorry

def parallel (l : Line) (p : Plane) : Prop :=
  sorry

def skew (l1 l2 : Line) : Prop :=
  sorry

def contained_in (l : Line) (p : Plane) : Prop :=
  sorry

theorem parallel_planes_theorem 
  (α β : Plane) 
  (a b : Line) 
  (h_diff : α ≠ β) 
  (h_perp : perpendicular a α ∧ perpendicular a β)
  (h_skew : skew a b)
  (h_contained : contained_in a α ∧ contained_in b β)
  (h_parallel : parallel b α) :
  parallel a β :=
sorry

end NUMINAMATH_CALUDE_parallel_planes_theorem_l574_57453


namespace NUMINAMATH_CALUDE_right_triangle_hypotenuse_l574_57468

theorem right_triangle_hypotenuse (a b c : ℝ) (h1 : a = 60) (h2 : b = 80) 
  (h3 : c^2 = a^2 + b^2) : c = 100 := by
  sorry

end NUMINAMATH_CALUDE_right_triangle_hypotenuse_l574_57468


namespace NUMINAMATH_CALUDE_repeating_decimal_equals_fraction_product_is_222_l574_57403

/-- The repeating decimal 0.018018018... as a real number -/
def repeating_decimal : ℚ := 18 / 999

/-- The fraction 2/111 -/
def target_fraction : ℚ := 2 / 111

/-- Theorem stating that the repeating decimal 0.018018018... is equal to 2/111 -/
theorem repeating_decimal_equals_fraction : repeating_decimal = target_fraction := by
  sorry

/-- The product of the numerator and denominator of the fraction -/
def numerator_denominator_product : ℕ := 2 * 111

/-- Theorem stating that the product of the numerator and denominator is 222 -/
theorem product_is_222 : numerator_denominator_product = 222 := by
  sorry

end NUMINAMATH_CALUDE_repeating_decimal_equals_fraction_product_is_222_l574_57403


namespace NUMINAMATH_CALUDE_iphone_price_calculation_l574_57431

def calculate_final_price (initial_price : ℝ) 
                          (discount1 : ℝ) (tax1 : ℝ) 
                          (discount2 : ℝ) (tax2 : ℝ) : ℝ :=
  let price_after_discount1 := initial_price * (1 - discount1)
  let price_after_tax1 := price_after_discount1 * (1 + tax1)
  let price_after_discount2 := price_after_tax1 * (1 - discount2)
  let final_price := price_after_discount2 * (1 + tax2)
  final_price

theorem iphone_price_calculation :
  let initial_price : ℝ := 1000
  let discount1 : ℝ := 0.1
  let tax1 : ℝ := 0.08
  let discount2 : ℝ := 0.2
  let tax2 : ℝ := 0.06
  let final_price := calculate_final_price initial_price discount1 tax1 discount2 tax2
  ∃ ε > 0, |final_price - 824.26| < ε :=
sorry

end NUMINAMATH_CALUDE_iphone_price_calculation_l574_57431


namespace NUMINAMATH_CALUDE_interest_calculation_l574_57486

/-- Given a principal amount P, calculate the compound interest for 2 years at 5% per year -/
def compound_interest (P : ℝ) : ℝ :=
  P * (1 + 0.05)^2 - P

/-- Given a principal amount P, calculate the simple interest for 2 years at 5% per year -/
def simple_interest (P : ℝ) : ℝ :=
  P * 0.05 * 2

/-- Theorem stating that if the compound interest is $615, then the simple interest is $600 -/
theorem interest_calculation (P : ℝ) :
  compound_interest P = 615 → simple_interest P = 600 := by
  sorry

end NUMINAMATH_CALUDE_interest_calculation_l574_57486


namespace NUMINAMATH_CALUDE_interest_rate_problem_l574_57433

theorem interest_rate_problem (R T : ℝ) : 
  900 * (1 + R * T / 100) = 956 ∧
  900 * (1 + (R + 4) * T / 100) = 1064 →
  T = 3 := by
sorry

end NUMINAMATH_CALUDE_interest_rate_problem_l574_57433


namespace NUMINAMATH_CALUDE_line_equation_sum_l574_57436

/-- Given a line with slope -4 passing through the point (5, 2), 
    prove that if its equation is of the form y = mx + b, then m + b = 18 -/
theorem line_equation_sum (m b : ℝ) : 
  m = -4 → 
  2 = m * 5 + b → 
  m + b = 18 :=
by sorry

end NUMINAMATH_CALUDE_line_equation_sum_l574_57436


namespace NUMINAMATH_CALUDE_intersection_sum_l574_57488

/-- Given two lines y = 2x + c and y = 4x + d intersecting at (3, 11), prove that c + d = 4 -/
theorem intersection_sum (c d : ℝ) 
  (h1 : 11 = 2 * 3 + c) 
  (h2 : 11 = 4 * 3 + d) : 
  c + d = 4 := by sorry

end NUMINAMATH_CALUDE_intersection_sum_l574_57488


namespace NUMINAMATH_CALUDE_sundress_price_problem_l574_57473

theorem sundress_price_problem (P : ℝ) : 
  P - (P * 0.85 * 1.25) = 4.5 → P * 0.85 = 61.2 := by
  sorry

end NUMINAMATH_CALUDE_sundress_price_problem_l574_57473


namespace NUMINAMATH_CALUDE_intersection_of_M_and_N_l574_57440

-- Define sets M and N
def M : Set ℝ := {x | x < 1}
def N : Set ℝ := {x | x^2 < 4}

-- Theorem statement
theorem intersection_of_M_and_N : M ∩ N = {x : ℝ | -2 < x ∧ x < 1} := by
  sorry

end NUMINAMATH_CALUDE_intersection_of_M_and_N_l574_57440


namespace NUMINAMATH_CALUDE_mika_initial_stickers_l574_57492

/-- Represents the number of stickers Mika has at different stages --/
structure StickerCount where
  initial : ℕ
  after_buying : ℕ
  after_birthday : ℕ
  after_giving : ℕ
  after_decorating : ℕ
  final : ℕ

/-- Defines the sticker transactions Mika goes through --/
def sticker_transactions (s : StickerCount) : Prop :=
  s.after_buying = s.initial + 26 ∧
  s.after_birthday = s.after_buying + 20 ∧
  s.after_giving = s.after_birthday - 6 ∧
  s.after_decorating = s.after_giving - 58 ∧
  s.final = s.after_decorating ∧
  s.final = 2

/-- Theorem stating that Mika initially had 20 stickers --/
theorem mika_initial_stickers :
  ∃ (s : StickerCount), sticker_transactions s ∧ s.initial = 20 := by
  sorry

end NUMINAMATH_CALUDE_mika_initial_stickers_l574_57492


namespace NUMINAMATH_CALUDE_sequence_property_l574_57445

theorem sequence_property (a : ℕ → ℝ) (c : ℝ) 
  (h1 : ∀ n : ℕ, a (n + 3) = a n)
  (h2 : ∀ n : ℕ, a n * a (n + 3) - a (n + 1) * a (n + 2) = c) :
  (∀ n : ℕ, a (n + 1) = a n ∧ c = 0) ∨ 
  (∀ n : ℕ, a n + a (n + 1) + a (n + 2) = 0 ∧ 4 * c - 3 * (a n)^2 > 0) :=
sorry

end NUMINAMATH_CALUDE_sequence_property_l574_57445


namespace NUMINAMATH_CALUDE_largest_ball_radius_l574_57420

/-- The radius of the torus circle -/
def torus_radius : ℝ := 2

/-- The x-coordinate of the torus circle center -/
def torus_center_x : ℝ := 4

/-- The z-coordinate of the torus circle center -/
def torus_center_z : ℝ := 1

/-- The theorem stating that the radius of the largest spherical ball that can sit on top of the center of the torus and touch the horizontal plane is 4 -/
theorem largest_ball_radius : 
  ∃ (r : ℝ), r = 4 ∧ 
  (torus_center_x ^ 2 + (r - torus_center_z) ^ 2 = (r + torus_radius) ^ 2) ∧
  r > 0 :=
sorry

end NUMINAMATH_CALUDE_largest_ball_radius_l574_57420


namespace NUMINAMATH_CALUDE_largest_number_problem_l574_57469

theorem largest_number_problem (a b c d e : ℕ) 
  (sum1 : a + b + c + d = 350)
  (sum2 : a + b + c + e = 370)
  (sum3 : a + b + d + e = 390)
  (sum4 : a + c + d + e = 410)
  (sum5 : b + c + d + e = 430) :
  max a (max b (max c (max d e))) = 138 := by
sorry

end NUMINAMATH_CALUDE_largest_number_problem_l574_57469


namespace NUMINAMATH_CALUDE_halloween_candy_theorem_l574_57460

/-- The number of candy pieces eaten on Halloween night -/
def candy_eaten (katie_candy sister_candy remaining_candy : ℕ) : ℕ :=
  katie_candy + sister_candy - remaining_candy

/-- Theorem: Given the conditions, the number of candy pieces eaten is 9 -/
theorem halloween_candy_theorem :
  candy_eaten 10 6 7 = 9 := by
  sorry

end NUMINAMATH_CALUDE_halloween_candy_theorem_l574_57460


namespace NUMINAMATH_CALUDE_special_dog_food_ounces_per_pound_l574_57487

/-- Represents the number of ounces in a pound of special dog food -/
def ounces_per_pound : ℕ := 16

/-- Represents the number of days in a year -/
def days_in_year : ℕ := 365

/-- Represents the number of days the puppy eats 2 ounces per day -/
def initial_feeding_days : ℕ := 60

/-- Represents the number of ounces the puppy eats per day during the initial feeding period -/
def initial_feeding_ounces : ℕ := 2

/-- Represents the number of ounces the puppy eats per day after the initial feeding period -/
def later_feeding_ounces : ℕ := 4

/-- Represents the number of pounds in each bag of special dog food -/
def pounds_per_bag : ℕ := 5

/-- Represents the number of bags the family needs to buy -/
def bags_needed : ℕ := 17

theorem special_dog_food_ounces_per_pound :
  ounces_per_pound = 16 :=
by sorry

end NUMINAMATH_CALUDE_special_dog_food_ounces_per_pound_l574_57487


namespace NUMINAMATH_CALUDE_m_range_equivalence_l574_57427

theorem m_range_equivalence (m : ℝ) : 
  Real.sqrt (2 * m + 1) > Real.sqrt (m^2 + m - 1) ↔ 
  m ≥ (Real.sqrt 5 - 1) / 2 ∧ m < 2 := by
  sorry

end NUMINAMATH_CALUDE_m_range_equivalence_l574_57427


namespace NUMINAMATH_CALUDE_double_inequality_proof_l574_57410

theorem double_inequality_proof (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) :
  let f := (1 / (x + y + z + 1)) - (1 / ((x + 1) * (y + 1) * (z + 1)))
  (0 < f) ∧ 
  (f ≤ 1/8) ∧ 
  (f = 1/8 ↔ x = 1 ∧ y = 1 ∧ z = 1) :=
by sorry


end NUMINAMATH_CALUDE_double_inequality_proof_l574_57410


namespace NUMINAMATH_CALUDE_function_root_property_l574_57454

/-- Given a function f(x) = m · 2^x + x^2 + nx, if the set of roots of f(x) is equal to 
    the set of roots of f(f(x)) and is non-empty, then m+n is in the interval [0, 4). -/
theorem function_root_property (m n : ℝ) : 
  let f : ℝ → ℝ := λ x ↦ m * (2^x) + x^2 + n*x
  (∃ x, f x = 0) ∧ 
  (∀ x, f x = 0 ↔ f (f x) = 0) →
  0 ≤ m + n ∧ m + n < 4 := by
sorry

end NUMINAMATH_CALUDE_function_root_property_l574_57454


namespace NUMINAMATH_CALUDE_undefined_values_sum_l574_57404

theorem undefined_values_sum (f : ℝ → ℝ) (h : f = λ x => 5*x / (3*x^2 - 9*x + 6)) : 
  ∃ C D : ℝ, (3*C^2 - 9*C + 6 = 0) ∧ (3*D^2 - 9*D + 6 = 0) ∧ (C + D = 3) := by
  sorry

end NUMINAMATH_CALUDE_undefined_values_sum_l574_57404


namespace NUMINAMATH_CALUDE_coloring_books_bought_l574_57426

theorem coloring_books_bought (initial books_given_away final : ℕ) : 
  initial = 45 → books_given_away = 6 → final = 59 → 
  final - (initial - books_given_away) = 20 := by
  sorry

end NUMINAMATH_CALUDE_coloring_books_bought_l574_57426


namespace NUMINAMATH_CALUDE_joan_balloon_count_l574_57462

/-- The number of orange balloons Joan has after receiving more from a friend -/
def total_balloons (initial : ℕ) (received : ℕ) : ℕ :=
  initial + received

/-- Theorem: Given Joan has 8 orange balloons initially and receives 2 more from a friend,
    she now has 10 orange balloons in total. -/
theorem joan_balloon_count : total_balloons 8 2 = 10 := by
  sorry

end NUMINAMATH_CALUDE_joan_balloon_count_l574_57462


namespace NUMINAMATH_CALUDE_parabola_through_point_l574_57482

/-- A parabola passing through the point (4, -2) has either the equation y^2 = x or x^2 = -8y -/
theorem parabola_through_point (x y : ℝ) : 
  (x = 4 ∧ y = -2) → (y^2 = x ∨ x^2 = -8*y) := by
  sorry

end NUMINAMATH_CALUDE_parabola_through_point_l574_57482


namespace NUMINAMATH_CALUDE_set_inclusion_implies_a_range_l574_57429

/-- Given sets A, B, and C defined as follows:
    A = {x | 2 ≤ x < 7}
    B = {x | 3 < x ≤ 10}
    C = {x | a-5 < x < a}
    Prove that if C is a non-empty subset of A ∪ B, then 7 ≤ a ≤ 10. -/
theorem set_inclusion_implies_a_range (a : ℝ) :
  let A : Set ℝ := {x | 2 ≤ x ∧ x < 7}
  let B : Set ℝ := {x | 3 < x ∧ x ≤ 10}
  let C : Set ℝ := {x | a - 5 < x ∧ x < a}
  C.Nonempty → C ⊆ A ∪ B → 7 ≤ a ∧ a ≤ 10 := by
  sorry


end NUMINAMATH_CALUDE_set_inclusion_implies_a_range_l574_57429


namespace NUMINAMATH_CALUDE_angle_y_measure_l574_57437

/-- An isosceles triangle with specific angle relationships -/
structure IsoscelesTriangle where
  -- Angle X in degrees
  x : ℝ
  -- Triangle sum theorem
  sum_theorem : x + 3*x + 3*x = 180
  -- Non-negativity of angles
  x_nonneg : x ≥ 0

/-- The measure of angle Y in the isosceles triangle is 540/7 degrees -/
theorem angle_y_measure (t : IsoscelesTriangle) : 3 * t.x = 540 / 7 := by
  sorry

end NUMINAMATH_CALUDE_angle_y_measure_l574_57437


namespace NUMINAMATH_CALUDE_equation_solution_l574_57435

theorem equation_solution : ∃! x : ℚ, (x^2 + 3*x + 5) / (x + 6) = x + 7 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l574_57435


namespace NUMINAMATH_CALUDE_franks_age_l574_57424

theorem franks_age (frank gabriel lucy : ℕ) : 
  gabriel = frank - 3 →
  frank + gabriel = 17 →
  lucy = gabriel + 5 →
  lucy = gabriel + frank →
  frank = 10 := by
sorry

end NUMINAMATH_CALUDE_franks_age_l574_57424


namespace NUMINAMATH_CALUDE_total_weight_of_CaO_l574_57493

/-- The atomic weight of calcium in g/mol -/
def atomic_weight_Ca : ℝ := 40.08

/-- The atomic weight of oxygen in g/mol -/
def atomic_weight_O : ℝ := 16.00

/-- The number of moles of CaO -/
def moles_CaO : ℝ := 7

/-- The molecular weight of CaO in g/mol -/
def molecular_weight_CaO : ℝ := atomic_weight_Ca + atomic_weight_O

/-- The total weight of CaO in grams -/
def total_weight_CaO : ℝ := molecular_weight_CaO * moles_CaO

/-- Theorem stating the total weight of 7 moles of CaO -/
theorem total_weight_of_CaO : total_weight_CaO = 392.56 := by
  sorry

end NUMINAMATH_CALUDE_total_weight_of_CaO_l574_57493


namespace NUMINAMATH_CALUDE_repeating_decimal_to_fraction_l574_57430

/-- Represents a repeating decimal with a two-digit repeating part -/
def repeating_decimal (a b : ℕ) : ℚ :=
  (10 * a + b) / 99

theorem repeating_decimal_to_fraction :
  repeating_decimal 2 7 = 3 / 11 ∧
  3 + 11 = 14 :=
sorry

end NUMINAMATH_CALUDE_repeating_decimal_to_fraction_l574_57430


namespace NUMINAMATH_CALUDE_custom_op_value_l574_57475

-- Define the custom operation *
def custom_op (a b : ℤ) : ℚ := 1 / a + 1 / b

-- State the theorem
theorem custom_op_value (a b : ℤ) (ha : a ≠ 0) (hb : b ≠ 0) 
  (sum_eq : a + b = 9) (prod_eq : a * b = 20) : 
  custom_op a b = 9 / 20 := by
  sorry

end NUMINAMATH_CALUDE_custom_op_value_l574_57475


namespace NUMINAMATH_CALUDE_carol_goal_impossible_l574_57461

theorem carol_goal_impossible (total_quizzes : ℕ) (goal_percentage : ℚ) 
  (quizzes_taken : ℕ) (as_earned : ℕ) : 
  total_quizzes = 60 → 
  goal_percentage = 85 / 100 → 
  quizzes_taken = 40 → 
  as_earned = 26 → 
  ¬ ∃ (future_as : ℕ), 
    (as_earned + future_as : ℚ) / total_quizzes ≥ goal_percentage ∧ 
    future_as ≤ total_quizzes - quizzes_taken :=
by sorry

end NUMINAMATH_CALUDE_carol_goal_impossible_l574_57461


namespace NUMINAMATH_CALUDE_faster_by_plane_l574_57441

-- Define the driving time in minutes
def driving_time : ℕ := 3 * 60 + 15

-- Define the components of the airplane trip
def airport_drive_time : ℕ := 10
def boarding_wait_time : ℕ := 20
def offboarding_time : ℕ := 10

-- Define the flight time as one-third of the driving time
def flight_time : ℕ := driving_time / 3

-- Define the total airplane trip time
def airplane_trip_time : ℕ := airport_drive_time + boarding_wait_time + flight_time + offboarding_time

-- Theorem to prove
theorem faster_by_plane : driving_time - airplane_trip_time = 90 := by
  sorry

end NUMINAMATH_CALUDE_faster_by_plane_l574_57441


namespace NUMINAMATH_CALUDE_geometric_sequence_min_a1_l574_57444

theorem geometric_sequence_min_a1 (a : ℕ+ → ℕ+) (r : ℕ+) :
  (∀ i : ℕ+, a (i + 1) = a i * r) →  -- Geometric sequence condition
  (a 20 + a 21 = 20^21) →            -- Given condition
  (∃ x y : ℕ+, (∀ k : ℕ+, a 1 ≤ 2^(x:ℕ) * 5^(y:ℕ)) ∧ 
               a 1 = 2^(x:ℕ) * 5^(y:ℕ) ∧ 
               x + y = 24) :=
by sorry

end NUMINAMATH_CALUDE_geometric_sequence_min_a1_l574_57444


namespace NUMINAMATH_CALUDE_problem_statement_l574_57418

theorem problem_statement :
  (¬ (∃ x₀ : ℝ, x₀^2 + x₀ + 1 < 0) ∨ (∀ a b c : ℝ, b > c → a * b > a * c)) ∧
  (¬ (∃ x₀ : ℝ, x₀^2 + x₀ + 1 < 0) ∧ ¬ (∀ a b c : ℝ, b > c → a * b > a * c)) :=
by sorry

end NUMINAMATH_CALUDE_problem_statement_l574_57418


namespace NUMINAMATH_CALUDE_find_x_l574_57438

-- Define the binary operation
def binary_op (n : ℤ) (x : ℤ) : ℤ := n - (n * x)

-- State the theorem
theorem find_x : ∃ x : ℤ, 
  (∀ n : ℕ, n > 2 → binary_op n x ≥ 10) ∧ 
  (binary_op 2 x < 10) ∧
  x = -3 := by
  sorry

end NUMINAMATH_CALUDE_find_x_l574_57438


namespace NUMINAMATH_CALUDE_school_picnic_volunteers_l574_57476

/-- The problem of calculating overlapping volunteers at a school picnic -/
theorem school_picnic_volunteers
  (total_parents : ℕ)
  (supervise_volunteers : ℕ)
  (refreshment_volunteers : ℕ)
  (h1 : total_parents = 84)
  (h2 : supervise_volunteers = 25)
  (h3 : refreshment_volunteers = 42)
  (h4 : refreshment_volunteers = (3/2 : ℚ) * (total_parents - supervise_volunteers - refreshment_volunteers + overlap_volunteers)) :
  overlap_volunteers = 11 :=
by sorry

end NUMINAMATH_CALUDE_school_picnic_volunteers_l574_57476


namespace NUMINAMATH_CALUDE_union_condition_implies_a_range_l574_57416

def A (a : ℝ) := {x : ℝ | a ≤ x ∧ x ≤ a + 3}
def B := {x : ℝ | x < -1 ∨ x > 5}

theorem union_condition_implies_a_range (a : ℝ) :
  A a ∪ B = B → a < -4 ∨ a > 5 := by
  sorry

end NUMINAMATH_CALUDE_union_condition_implies_a_range_l574_57416


namespace NUMINAMATH_CALUDE_work_left_after_collaboration_l574_57497

/-- Represents the fraction of work completed in one day -/
def work_rate (days : ℕ) : ℚ := 1 / days

/-- Represents the total work completed by two people in a given number of days -/
def total_work (rate_a rate_b : ℚ) (days : ℕ) : ℚ := (rate_a + rate_b) * days

theorem work_left_after_collaboration (days_a days_b collab_days : ℕ) 
  (h1 : days_a = 15) (h2 : days_b = 20) (h3 : collab_days = 4) : 
  1 - total_work (work_rate days_a) (work_rate days_b) collab_days = 8 / 15 := by
  sorry

#check work_left_after_collaboration

end NUMINAMATH_CALUDE_work_left_after_collaboration_l574_57497


namespace NUMINAMATH_CALUDE_expected_teachers_with_masters_l574_57439

def total_teachers : ℕ := 320
def masters_degree_ratio : ℚ := 1 / 4

theorem expected_teachers_with_masters :
  (total_teachers : ℚ) * masters_degree_ratio = 80 := by
  sorry

end NUMINAMATH_CALUDE_expected_teachers_with_masters_l574_57439


namespace NUMINAMATH_CALUDE_probability_one_red_one_white_l574_57405

/-- The probability of drawing 1 red ball and 1 white ball when drawing two balls with replacement 
    from a bag containing 2 red balls and 3 white balls is equal to 2/5. -/
theorem probability_one_red_one_white (total_balls : ℕ) (red_balls : ℕ) (white_balls : ℕ)
  (h_total : total_balls = red_balls + white_balls)
  (h_red : red_balls = 2)
  (h_white : white_balls = 3) :
  (red_balls / total_balls) * (white_balls / total_balls) * 2 = 2 / 5 := by
  sorry

end NUMINAMATH_CALUDE_probability_one_red_one_white_l574_57405


namespace NUMINAMATH_CALUDE_cricket_equipment_cost_l574_57477

theorem cricket_equipment_cost (bat_cost : ℕ) (ball_cost : ℕ) : 
  (7 * bat_cost + 6 * ball_cost = 3800) →
  (3 * bat_cost + 5 * ball_cost = 1750) →
  (bat_cost = 500) →
  ball_cost = 50 := by
sorry

end NUMINAMATH_CALUDE_cricket_equipment_cost_l574_57477


namespace NUMINAMATH_CALUDE_fraction_simplification_l574_57411

theorem fraction_simplification :
  (2 * Real.sqrt 7) / (Real.sqrt 2 + Real.sqrt 3 - Real.sqrt 6) =
  (2 * Real.sqrt 14 + 8 * Real.sqrt 21 + 2 * Real.sqrt 42 + 8 * Real.sqrt 63) / 23 := by
  sorry

end NUMINAMATH_CALUDE_fraction_simplification_l574_57411


namespace NUMINAMATH_CALUDE_steves_return_speed_l574_57467

/-- Proves that given a round trip with specified conditions, the return speed is 10 km/h -/
theorem steves_return_speed (total_distance : ℝ) (total_time : ℝ) (outbound_distance : ℝ) :
  total_distance = 40 →
  total_time = 6 →
  outbound_distance = 20 →
  let outbound_speed := outbound_distance / (total_time / 2)
  let return_speed := 2 * outbound_speed
  return_speed = 10 := by
  sorry

end NUMINAMATH_CALUDE_steves_return_speed_l574_57467


namespace NUMINAMATH_CALUDE_base7_addition_multiplication_l574_57498

/-- Converts a base 7 number represented as a list of digits to its decimal equivalent -/
def base7ToDecimal (digits : List Nat) : Nat :=
  digits.foldl (fun acc d => 7 * acc + d) 0

/-- Converts a decimal number to its base 7 representation as a list of digits -/
def decimalToBase7 (n : Nat) : List Nat :=
  if n < 7 then [n]
  else (n % 7) :: decimalToBase7 (n / 7)

/-- Adds two base 7 numbers -/
def addBase7 (a b : List Nat) : List Nat :=
  decimalToBase7 (base7ToDecimal a + base7ToDecimal b)

/-- Multiplies a base 7 number by another base 7 number -/
def mulBase7 (a b : List Nat) : List Nat :=
  decimalToBase7 (base7ToDecimal a * base7ToDecimal b)

theorem base7_addition_multiplication :
  mulBase7 (addBase7 [5, 2] [4, 3, 3]) [2] = [4, 6, 6] := by sorry

end NUMINAMATH_CALUDE_base7_addition_multiplication_l574_57498


namespace NUMINAMATH_CALUDE_earth_inhabitable_fraction_l574_57447

theorem earth_inhabitable_fraction :
  let water_free_fraction : ℚ := 1/4
  let inhabitable_land_fraction : ℚ := 1/3
  let inhabitable_fraction : ℚ := water_free_fraction * inhabitable_land_fraction
  inhabitable_fraction = 1/12 := by
sorry

end NUMINAMATH_CALUDE_earth_inhabitable_fraction_l574_57447


namespace NUMINAMATH_CALUDE_max_plus_min_equals_zero_l574_57484

def f (x : ℝ) := x^3 - 3*x

theorem max_plus_min_equals_zero :
  ∀ m n : ℝ,
  (∀ x : ℝ, f x ≤ m) →
  (∃ x : ℝ, f x = m) →
  (∀ x : ℝ, n ≤ f x) →
  (∃ x : ℝ, f x = n) →
  m + n = 0 :=
by sorry

end NUMINAMATH_CALUDE_max_plus_min_equals_zero_l574_57484


namespace NUMINAMATH_CALUDE_pythagorean_triple_3_4_5_l574_57466

theorem pythagorean_triple_3_4_5 : 
  ∃ (x : ℕ), x > 0 ∧ 3^2 + 4^2 = x^2 :=
by
  use 5
  sorry

#check pythagorean_triple_3_4_5

end NUMINAMATH_CALUDE_pythagorean_triple_3_4_5_l574_57466


namespace NUMINAMATH_CALUDE_largest_two_digit_prime_factor_l574_57432

/-- The binomial coefficient (n choose k) -/
def binomial (n k : ℕ) : ℕ := (Nat.factorial n) / (Nat.factorial k * Nat.factorial (n - k))

/-- Checks if a number is prime -/
def isPrime (n : ℕ) : Prop := Nat.Prime n

/-- Checks if a number is a two-digit number -/
def isTwoDigit (n : ℕ) : Prop := 10 ≤ n ∧ n < 100

/-- Checks if a number is a factor of another number -/
def isFactor (a b : ℕ) : Prop := b % a = 0

theorem largest_two_digit_prime_factor :
  ∃ (p : ℕ), isPrime p ∧ 
             isTwoDigit p ∧ 
             isFactor p (binomial 300 150) ∧
             (∀ (q : ℕ), isPrime q → isTwoDigit q → isFactor q (binomial 300 150) → q ≤ p) ∧
             p = 97 := by sorry

end NUMINAMATH_CALUDE_largest_two_digit_prime_factor_l574_57432


namespace NUMINAMATH_CALUDE_f_diff_max_min_eq_one_l574_57499

/-- The function f(x) = x^2 - 2bx - 1 -/
def f (b : ℝ) (x : ℝ) : ℝ := x^2 - 2*b*x - 1

/-- The closed interval [0, 1] -/
def I : Set ℝ := Set.Icc 0 1

/-- The statement that the difference between the maximum and minimum values of f(x) on [0, 1] is 1 -/
def diffMaxMin (b : ℝ) : Prop :=
  ∃ (max min : ℝ), (∀ x ∈ I, f b x ≤ max) ∧
                   (∀ x ∈ I, min ≤ f b x) ∧
                   (max - min = 1)

/-- The main theorem -/
theorem f_diff_max_min_eq_one :
  ∀ b : ℝ, diffMaxMin b ↔ (b = 0 ∨ b = 1) :=
sorry

end NUMINAMATH_CALUDE_f_diff_max_min_eq_one_l574_57499


namespace NUMINAMATH_CALUDE_factorization_problem_1_factorization_problem_2_l574_57480

-- Problem 1
theorem factorization_problem_1 (x y : ℝ) :
  5 * x^2 + 6 * x * y - 8 * y^2 = (5 * x - 4 * y) * (x + 2 * y) := by
  sorry

-- Problem 2
theorem factorization_problem_2 (x a : ℝ) :
  x^2 + 2 * x - 15 - a * x - 5 * a = (x + 5) * (x - (3 + a)) := by
  sorry

end NUMINAMATH_CALUDE_factorization_problem_1_factorization_problem_2_l574_57480


namespace NUMINAMATH_CALUDE_village_population_l574_57481

theorem village_population (P : ℕ) : 
  (P : ℝ) * (1 - 0.1) * (1 - 0.25) * (1 - 0.12) * (1 - 0.15) = 4136 → 
  P = 8192 := by
sorry

end NUMINAMATH_CALUDE_village_population_l574_57481


namespace NUMINAMATH_CALUDE_sqrt_sum_equals_seven_l574_57470

theorem sqrt_sum_equals_seven (y : ℝ) (h : Real.sqrt (64 - y^2) - Real.sqrt (36 - y^2) = 4) :
  Real.sqrt (64 - y^2) + Real.sqrt (36 - y^2) = 7 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_sum_equals_seven_l574_57470


namespace NUMINAMATH_CALUDE_town_population_l574_57407

theorem town_population (increase_rate : ℝ) (future_population : ℕ) :
  increase_rate = 0.1 →
  future_population = 242 →
  ∃ present_population : ℕ,
    present_population * (1 + increase_rate) = future_population ∧
    present_population = 220 := by
  sorry

end NUMINAMATH_CALUDE_town_population_l574_57407


namespace NUMINAMATH_CALUDE_different_graphs_l574_57472

-- Define the three equations
def equation_I (x y : ℝ) : Prop := y = x + 3
def equation_II (x y : ℝ) : Prop := y = (x^2 - 1) / (x - 1)
def equation_III (x y : ℝ) : Prop := (x - 1) * y = x^2 - 1

-- Define what it means for two equations to have the same graph
def same_graph (eq1 eq2 : ℝ → ℝ → Prop) : Prop :=
  ∀ x y, eq1 x y ↔ eq2 x y

-- Theorem statement
theorem different_graphs :
  ¬(same_graph equation_I equation_II) ∧
  ¬(same_graph equation_I equation_III) ∧
  ¬(same_graph equation_II equation_III) :=
sorry

end NUMINAMATH_CALUDE_different_graphs_l574_57472


namespace NUMINAMATH_CALUDE_smallest_integer_x_l574_57442

theorem smallest_integer_x : ∃ x : ℤ, (∀ y : ℤ, 3 * |y|^3 + 5 < 56 → x ≤ y) ∧ (3 * |x|^3 + 5 < 56) := by
  sorry

end NUMINAMATH_CALUDE_smallest_integer_x_l574_57442


namespace NUMINAMATH_CALUDE_club_truncator_probability_l574_57443

/-- Represents the outcome of a single match -/
inductive MatchResult
  | Win
  | Loss
  | Tie

/-- The total number of matches played by Club Truncator -/
def total_matches : ℕ := 5

/-- The probability of each match result -/
def match_probability : ℚ := 1 / 3

/-- Calculates the probability of having more wins than losses in the season -/
noncomputable def prob_more_wins_than_losses : ℚ := sorry

/-- The main theorem stating the probability of more wins than losses -/
theorem club_truncator_probability : prob_more_wins_than_losses = 32 / 81 := by sorry

end NUMINAMATH_CALUDE_club_truncator_probability_l574_57443


namespace NUMINAMATH_CALUDE_problem_solution_l574_57422

theorem problem_solution : 
  (((Real.sqrt 48) / (Real.sqrt 3) - (Real.sqrt (1/2)) * (Real.sqrt 12) + (Real.sqrt 24)) = 4 + Real.sqrt 6) ∧
  ((Real.sqrt 5 + Real.sqrt 2) * (Real.sqrt 5 - Real.sqrt 2) + (Real.sqrt 3 - 1)^2 = 7 - 2 * Real.sqrt 3) := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l574_57422


namespace NUMINAMATH_CALUDE_triangle_internal_point_theorem_l574_57452

/-- A triangle in a 2D plane --/
structure Triangle where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ

/-- Check if a point is inside a triangle (excluding the boundary) --/
def isInside (t : Triangle) (P : ℝ × ℝ) : Prop := sorry

/-- Check if a triangle is equilateral --/
def isEquilateral (t : Triangle) : Prop := sorry

/-- Check if three lengths can form a triangle --/
def canFormTriangle (a b c : ℝ) : Prop := sorry

/-- Main theorem --/
theorem triangle_internal_point_theorem (t : Triangle) (P : ℝ × ℝ) 
  (h : isInside t P) : 
  (isEquilateral t) ↔ 
  (canFormTriangle (dist P t.A) (dist P t.B) (dist P t.C)) := by
  sorry

#check triangle_internal_point_theorem

end NUMINAMATH_CALUDE_triangle_internal_point_theorem_l574_57452

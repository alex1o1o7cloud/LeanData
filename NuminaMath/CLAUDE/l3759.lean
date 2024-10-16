import Mathlib

namespace NUMINAMATH_CALUDE_min_circles_6x3_min_circles_5x3_l3759_375949

-- Define a rectangle
structure Rectangle where
  width : ℝ
  height : ℝ

-- Define a circle
structure Circle where
  radius : ℝ

-- Define a function to calculate the minimum number of circles needed to cover a rectangle
def minCircles (r : Rectangle) (c : Circle) : ℕ :=
  sorry

-- Theorem for 6 × 3 rectangle
theorem min_circles_6x3 :
  let r := Rectangle.mk 6 3
  let c := Circle.mk (Real.sqrt 2)
  minCircles r c = 6 :=
sorry

-- Theorem for 5 × 3 rectangle
theorem min_circles_5x3 :
  let r := Rectangle.mk 5 3
  let c := Circle.mk (Real.sqrt 2)
  minCircles r c = 5 :=
sorry

end NUMINAMATH_CALUDE_min_circles_6x3_min_circles_5x3_l3759_375949


namespace NUMINAMATH_CALUDE_square_equality_implies_four_l3759_375982

theorem square_equality_implies_four (x : ℝ) : (8 - x)^2 = x^2 → x = 4 := by
  sorry

end NUMINAMATH_CALUDE_square_equality_implies_four_l3759_375982


namespace NUMINAMATH_CALUDE_trisection_point_intersection_l3759_375910

noncomputable section

def f (x : ℝ) := Real.log x / Real.log 2

theorem trisection_point_intersection
  (x₁ x₂ : ℝ)
  (h_order : 0 < x₁ ∧ x₁ < x₂)
  (h_x₁ : x₁ = 4)
  (h_x₂ : x₂ = 16) :
  ∃ x₄ : ℝ, f x₄ = (2 * f x₁ + f x₂) / 3 ∧ x₄ = 2^(8/3) :=
sorry


end NUMINAMATH_CALUDE_trisection_point_intersection_l3759_375910


namespace NUMINAMATH_CALUDE_problem_solution_l3759_375989

theorem problem_solution (a b c : ℝ) 
  (h1 : |a - 4| + |b + 5| = 0) 
  (h2 : a + c = 0) : 
  3*a + 2*b - 4*c = 18 := by
sorry

end NUMINAMATH_CALUDE_problem_solution_l3759_375989


namespace NUMINAMATH_CALUDE_DL_length_l3759_375948

-- Define the triangle DEF
structure Triangle :=
  (DE : ℝ)
  (EF : ℝ)
  (FD : ℝ)

-- Define the circles ω3 and ω4
structure Circle :=
  (center : ℝ × ℝ)
  (radius : ℝ)

-- Define the point L
def L : ℝ × ℝ := sorry

-- Define the given triangle
def givenTriangle : Triangle :=
  { DE := 6
  , EF := 10
  , FD := 8 }

-- Define circle ω3
def ω3 : Circle := sorry

-- Define circle ω4
def ω4 : Circle := sorry

-- State the theorem
theorem DL_length (t : Triangle) (ω3 ω4 : Circle) :
  t = givenTriangle →
  (ω3.center.1 - L.1)^2 + (ω3.center.2 - L.2)^2 = ω3.radius^2 →
  (ω4.center.1 - L.1)^2 + (ω4.center.2 - L.2)^2 = ω4.radius^2 →
  (0 - L.1)^2 + (0 - L.2)^2 = 4^2 := by
  sorry

end NUMINAMATH_CALUDE_DL_length_l3759_375948


namespace NUMINAMATH_CALUDE_hyperbola_eccentricity_l3759_375945

/-- Represents a hyperbola with semi-major axis a and semi-minor axis b -/
structure Hyperbola (a b : ℝ) where
  eq : (x y : ℝ) → x^2 / a^2 - y^2 / b^2 = 1
  a_pos : a > 0
  b_pos : b > 0

/-- Represents a point in 2D space -/
structure Point (x y : ℝ)

/-- The left focus of the hyperbola -/
def leftFocus (h : Hyperbola a b) : Point c 0 := sorry

/-- The right vertex of the hyperbola -/
def rightVertex (h : Hyperbola a b) : Point a 0 := sorry

/-- The upper endpoint of the imaginary axis -/
def upperImaginaryEndpoint (h : Hyperbola a b) : Point 0 b := sorry

/-- The point where AB intersects the asymptote -/
def intersectionPoint (h : Hyperbola a b) : Point (a/2) (b/2) := sorry

/-- FM bisects ∠BFA -/
def fmBisectsAngle (h : Hyperbola a b) : Prop := sorry

/-- Eccentricity of the hyperbola -/
def eccentricity (h : Hyperbola a b) : ℝ := sorry

/-- Main theorem: The eccentricity of the hyperbola is 1 + √3 -/
theorem hyperbola_eccentricity (a b : ℝ) (h : Hyperbola a b) 
  (bisect : fmBisectsAngle h) : eccentricity h = 1 + Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_hyperbola_eccentricity_l3759_375945


namespace NUMINAMATH_CALUDE_perpendicular_lines_intersection_l3759_375961

/-- Two lines are perpendicular if and only if the product of their slopes is -1 -/
def perpendicular (m1 m2 : ℝ) : Prop := m1 * m2 = -1

/-- The equation of a line in the form ax + by = c -/
def line_equation (a b c : ℝ) (x y : ℝ) : Prop := a * x + b * y = c

theorem perpendicular_lines_intersection (a b c : ℝ) :
  line_equation a (-2) c 1 (-5) ∧
  line_equation 2 b (-c) 1 (-5) ∧
  perpendicular (a / 2) (-2 / b) →
  c = 13 := by sorry

end NUMINAMATH_CALUDE_perpendicular_lines_intersection_l3759_375961


namespace NUMINAMATH_CALUDE_impossible_erasure_l3759_375912

def arithmetic_sequence (a₁ d : ℕ) (n : ℕ) : List ℕ :=
  List.range n |>.map (fun i => a₁ + i * d)

def sum_list (l : List ℕ) : ℕ :=
  l.foldl (· + ·) 0

def is_divisible_by_11 (n : ℕ) : Prop :=
  n % 11 = 0

theorem impossible_erasure (a₁ d n : ℕ) (h₁ : a₁ = 4) (h₂ : d = 10) (h₃ : n = 11) :
  ¬ ∃ (e₁ : ℕ) (e₂ e₃ e₄ : List ℕ),
    (e₁ ∈ arithmetic_sequence a₁ d n) ∧
    (e₂.length = 2) ∧ (e₃.length = 3) ∧ (e₄.length = 4) ∧
    (∀ x ∈ e₂ ++ e₃ ++ e₄, x ∈ arithmetic_sequence a₁ d n) ∧
    (is_divisible_by_11 (sum_list (arithmetic_sequence a₁ d n) - e₁)) ∧
    (is_divisible_by_11 (sum_list (arithmetic_sequence a₁ d n) - e₁ - sum_list e₂)) ∧
    (is_divisible_by_11 (sum_list (arithmetic_sequence a₁ d n) - e₁ - sum_list e₂ - sum_list e₃)) ∧
    (is_divisible_by_11 (sum_list (arithmetic_sequence a₁ d n) - e₁ - sum_list e₂ - sum_list e₃ - sum_list e₄)) :=
by sorry

end NUMINAMATH_CALUDE_impossible_erasure_l3759_375912


namespace NUMINAMATH_CALUDE_carpet_innermost_length_l3759_375941

/-- Represents the dimensions of a rectangle --/
structure Rectangle where
  length : ℝ
  width : ℝ

/-- Calculates the area of a rectangle --/
def area (r : Rectangle) : ℝ := r.length * r.width

/-- Represents the carpet with three nested rectangles --/
structure Carpet where
  inner : Rectangle
  middle : Rectangle
  outer : Rectangle

/-- Checks if three numbers form an arithmetic progression --/
def isArithmeticProgression (a b c : ℝ) : Prop :=
  b - a = c - b

theorem carpet_innermost_length :
  ∀ (c : Carpet),
    c.inner.width = 2 →
    c.middle.length = c.inner.length + 4 →
    c.middle.width = c.inner.width + 4 →
    c.outer.length = c.middle.length + 4 →
    c.outer.width = c.middle.width + 4 →
    isArithmeticProgression (area c.inner) (area c.middle) (area c.outer) →
    c.inner.length = 4 := by
  sorry

#check carpet_innermost_length

end NUMINAMATH_CALUDE_carpet_innermost_length_l3759_375941


namespace NUMINAMATH_CALUDE_elise_initial_dog_food_l3759_375903

/-- The amount of dog food Elise already had -/
def initial_amount : ℕ := sorry

/-- The amount of dog food in the first bag Elise bought -/
def first_bag : ℕ := 15

/-- The amount of dog food in the second bag Elise bought -/
def second_bag : ℕ := 10

/-- The total amount of dog food Elise has after buying -/
def total_amount : ℕ := 40

theorem elise_initial_dog_food : initial_amount = 15 :=
  sorry

end NUMINAMATH_CALUDE_elise_initial_dog_food_l3759_375903


namespace NUMINAMATH_CALUDE_tangent_line_to_x_ln_x_l3759_375935

/-- A line y = 2x + m is tangent to the curve y = x ln x if and only if m = -e -/
theorem tangent_line_to_x_ln_x (m : ℝ) : 
  (∃ x₀ : ℝ, x₀ > 0 ∧ 
    (2 * x₀ + m = x₀ * Real.log x₀) ∧ 
    (2 = Real.log x₀ + 1)) ↔ 
  m = -Real.exp 1 := by
sorry

end NUMINAMATH_CALUDE_tangent_line_to_x_ln_x_l3759_375935


namespace NUMINAMATH_CALUDE_thirty_sided_polygon_diagonals_l3759_375931

/-- The number of diagonals in a convex polygon with n sides -/
def num_diagonals (n : ℕ) : ℕ := n * (n - 3) / 2

/-- Theorem: A convex polygon with 30 sides has 405 diagonals -/
theorem thirty_sided_polygon_diagonals :
  num_diagonals 30 = 405 := by
  sorry

end NUMINAMATH_CALUDE_thirty_sided_polygon_diagonals_l3759_375931


namespace NUMINAMATH_CALUDE_arithmetic_sequence_property_l3759_375955

def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_property (a : ℕ → ℝ) 
  (h_arithmetic : arithmetic_sequence a) 
  (h_sum : a 1 + a 13 = 12) : 
  a 7 = 6 := by
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_property_l3759_375955


namespace NUMINAMATH_CALUDE_min_value_f_f_decreasing_sum_lower_bound_l3759_375971

noncomputable section

-- Define the function f
def f (a b : ℝ) (x : ℝ) : ℝ := a * Real.log x + b * x^2 + x

-- Part 1
theorem min_value_f (x : ℝ) (h : x > 0) : 
  f (-1) 0 x ≥ 1 :=
sorry

-- Part 2
def f_special (x : ℝ) : ℝ := Real.log x - x^2 + x

theorem f_decreasing (x : ℝ) (h : x > 1) :
  ∀ y > x, f_special y < f_special x :=
sorry

-- Part 3
theorem sum_lower_bound (x₁ x₂ : ℝ) (h₁ : x₁ > 0) (h₂ : x₂ > 0)
  (h : f 1 1 x₁ + f 1 1 x₂ + x₁ * x₂ = 0) :
  x₁ + x₂ ≥ (Real.sqrt 5 - 1) / 2 :=
sorry

end NUMINAMATH_CALUDE_min_value_f_f_decreasing_sum_lower_bound_l3759_375971


namespace NUMINAMATH_CALUDE_square_difference_equality_l3759_375969

theorem square_difference_equality : (45 + 18)^2 - (45^2 + 18^2 + 10) = 1610 := by
  sorry

end NUMINAMATH_CALUDE_square_difference_equality_l3759_375969


namespace NUMINAMATH_CALUDE_rational_function_simplification_and_evaluation_l3759_375951

theorem rational_function_simplification_and_evaluation :
  ∀ x : ℝ, x ≠ 5 →
  (x^2 - 3*x - 10) / (x - 5) = x + 2 ∧
  (4^2 - 3*4 - 10) / (4 - 5) = 6 := by
sorry

end NUMINAMATH_CALUDE_rational_function_simplification_and_evaluation_l3759_375951


namespace NUMINAMATH_CALUDE_six_students_arrangement_l3759_375999

/-- The number of ways to arrange n elements -/
def factorial (n : ℕ) : ℕ := (List.range n).foldl (· * ·) 1

/-- The number of ways to arrange n elements, excluding arrangements where two specific elements are adjacent -/
def arrangementWithoutAdjacent (n : ℕ) : ℕ :=
  factorial n - (factorial (n - 1) * 2)

theorem six_students_arrangement :
  arrangementWithoutAdjacent 6 = 480 := by
  sorry

end NUMINAMATH_CALUDE_six_students_arrangement_l3759_375999


namespace NUMINAMATH_CALUDE_difference_of_squares_l3759_375921

theorem difference_of_squares (x : ℝ) : x^2 - 1 = (x + 1) * (x - 1) := by
  sorry

end NUMINAMATH_CALUDE_difference_of_squares_l3759_375921


namespace NUMINAMATH_CALUDE_multiplication_addition_equality_l3759_375984

theorem multiplication_addition_equality : 26 * 43 + 57 * 26 = 2600 := by
  sorry

end NUMINAMATH_CALUDE_multiplication_addition_equality_l3759_375984


namespace NUMINAMATH_CALUDE_set_intersection_and_union_l3759_375922

def A (x : ℝ) : Set ℝ := {x^2, 2*x - 1, -4}
def B (x : ℝ) : Set ℝ := {x - 5, 1 - x, 9}

theorem set_intersection_and_union :
  ∃ x : ℝ, (B x ∩ A x = {9}) ∧ 
           (x = -3) ∧ 
           (A x ∪ B x = {-8, -7, -4, 4, 9}) := by
  sorry

end NUMINAMATH_CALUDE_set_intersection_and_union_l3759_375922


namespace NUMINAMATH_CALUDE_commute_time_sum_of_squares_l3759_375904

theorem commute_time_sum_of_squares 
  (x y : ℝ) 
  (avg_eq : (x + y + 10 + 11 + 9) / 5 = 10) 
  (var_eq : ((x - 10)^2 + (y - 10)^2 + 0^2 + 1^2 + (-1)^2) / 5 = 2) : 
  x^2 + y^2 = 208 := by
sorry

end NUMINAMATH_CALUDE_commute_time_sum_of_squares_l3759_375904


namespace NUMINAMATH_CALUDE_dance_parity_l3759_375939

theorem dance_parity (n : ℕ) (h_odd : Odd n) (dances : Fin n → ℕ) : 
  ∃ i : Fin n, Even (dances i) := by
  sorry

end NUMINAMATH_CALUDE_dance_parity_l3759_375939


namespace NUMINAMATH_CALUDE_transportation_cost_calculation_l3759_375927

def transportation_cost (initial_amount dress_cost pants_cost jacket_cost dress_count pants_count jacket_count remaining_amount : ℕ) : ℕ :=
  let clothes_cost := dress_cost * dress_count + pants_cost * pants_count + jacket_cost * jacket_count
  let total_spent := initial_amount - remaining_amount
  total_spent - clothes_cost

theorem transportation_cost_calculation :
  transportation_cost 400 20 12 30 5 3 4 139 = 5 :=
by sorry

end NUMINAMATH_CALUDE_transportation_cost_calculation_l3759_375927


namespace NUMINAMATH_CALUDE_complex_equality_l3759_375953

theorem complex_equality (z₁ z₂ : ℂ) (a : ℝ) 
  (h1 : z₁ = 1 + Complex.I) 
  (h2 : z₂ = 3 + a * Complex.I) 
  (h3 : 3 * z₁ = z₂) : 
  a = 3 := by
sorry

end NUMINAMATH_CALUDE_complex_equality_l3759_375953


namespace NUMINAMATH_CALUDE_collinear_points_sum_l3759_375970

/-- Three points in 3D space are collinear if they lie on the same straight line. -/
def collinear (p1 p2 p3 : ℝ × ℝ × ℝ) : Prop :=
  ∃ t1 t2 : ℝ, p2 = p1 + t1 • (p3 - p1) ∧ p3 = p1 + t2 • (p3 - p1)

/-- Given three collinear points (2,x,y), (x,3,y), and (x,y,4), prove that x + y = 6. -/
theorem collinear_points_sum (x y : ℝ) :
  collinear (2, x, y) (x, 3, y) (x, y, 4) → x + y = 6 := by
  sorry

end NUMINAMATH_CALUDE_collinear_points_sum_l3759_375970


namespace NUMINAMATH_CALUDE_divisibility_of_expression_l3759_375964

theorem divisibility_of_expression (p : ℕ) (h_prime : Nat.Prime p) (h_gt_two : p > 2) :
  ∃ k : ℤ, (⌊(2 + Real.sqrt 5)^p⌋ : ℤ) - 2^(p + 1) = k * p :=
sorry

end NUMINAMATH_CALUDE_divisibility_of_expression_l3759_375964


namespace NUMINAMATH_CALUDE_fraction_equality_l3759_375993

theorem fraction_equality (x y : ℝ) (h1 : x ≠ 0) (h2 : y ≠ 0) 
  (h3 : (5 * x + 2 * y) / (x - 5 * y) = 3) : 
  (x + 5 * y) / (5 * x - y) = 7 / 87 := by
  sorry

end NUMINAMATH_CALUDE_fraction_equality_l3759_375993


namespace NUMINAMATH_CALUDE_parallel_vectors_dot_product_l3759_375954

/-- Given vectors a⃗(x,2), b⃗=(2,1), c⃗=(3,x), if a⃗ ∥ b⃗, then a⃗ ⋅ c⃗ = 20 -/
theorem parallel_vectors_dot_product (x : ℝ) :
  let a : ℝ × ℝ := (x, 2)
  let b : ℝ × ℝ := (2, 1)
  let c : ℝ × ℝ := (3, x)
  (∃ (k : ℝ), a = k • b) →
  a.1 * c.1 + a.2 * c.2 = 20 := by
sorry

end NUMINAMATH_CALUDE_parallel_vectors_dot_product_l3759_375954


namespace NUMINAMATH_CALUDE_power_equation_solution_l3759_375917

theorem power_equation_solution (n : ℕ) : 5^n = 5 * 25^3 * 125^2 → n = 13 := by
  sorry

end NUMINAMATH_CALUDE_power_equation_solution_l3759_375917


namespace NUMINAMATH_CALUDE_simplify_rational_expression_l3759_375983

theorem simplify_rational_expression (x : ℝ) (h1 : x ≠ 0) (h2 : x ≠ 2) (h3 : x ≠ 4) :
  ((x + 2) / (x^2 - 2*x) - (x - 1) / (x^2 - 4*x + 4)) / ((x - 4) / (x^2 - 2*x)) = 1 / (x - 2) :=
by sorry

end NUMINAMATH_CALUDE_simplify_rational_expression_l3759_375983


namespace NUMINAMATH_CALUDE_average_and_difference_l3759_375994

theorem average_and_difference (a b c : ℝ) 
  (h1 : (a + b) / 2 = 30)
  (h2 : (b + c) / 2 = 60)
  (h3 : c - a = 60) :
  c - a = 60 := by
  sorry

end NUMINAMATH_CALUDE_average_and_difference_l3759_375994


namespace NUMINAMATH_CALUDE_sin_300_degrees_l3759_375920

theorem sin_300_degrees : Real.sin (300 * Real.pi / 180) = -Real.sqrt 3 / 2 := by
  sorry

end NUMINAMATH_CALUDE_sin_300_degrees_l3759_375920


namespace NUMINAMATH_CALUDE_partially_symmetric_iff_l3759_375962

/-- A function is partially symmetric if it satisfies three specific conditions. -/
def PartiallySymmetric (f : ℝ → ℝ) : Prop :=
  (f 0 = 0) ∧
  (∀ x : ℝ, x ≠ 0 → x * (deriv f x) > 0) ∧
  (∀ x₁ x₂ : ℝ, x₁ < 0 ∧ 0 < x₂ ∧ abs x₁ < abs x₂ → f x₁ < f x₂)

/-- Theorem: A function is partially symmetric if and only if it satisfies the three conditions. -/
theorem partially_symmetric_iff (f : ℝ → ℝ) :
  PartiallySymmetric f ↔
    (f 0 = 0) ∧
    (∀ x : ℝ, x ≠ 0 → x * (deriv f x) > 0) ∧
    (∀ x₁ x₂ : ℝ, x₁ < 0 ∧ 0 < x₂ ∧ abs x₁ < abs x₂ → f x₁ < f x₂) :=
by
  sorry

end NUMINAMATH_CALUDE_partially_symmetric_iff_l3759_375962


namespace NUMINAMATH_CALUDE_five_balls_four_boxes_l3759_375975

/-- The number of ways to distribute indistinguishable balls into distinguishable boxes -/
def distribute_balls (n : ℕ) (k : ℕ) : ℕ := sorry

/-- Theorem: There are 56 ways to distribute 5 indistinguishable balls into 4 distinguishable boxes -/
theorem five_balls_four_boxes : distribute_balls 5 4 = 56 := by sorry

end NUMINAMATH_CALUDE_five_balls_four_boxes_l3759_375975


namespace NUMINAMATH_CALUDE_constant_term_expansion_l3759_375901

theorem constant_term_expansion (x : ℝ) : 
  (∃ c : ℝ, c = -160 ∧ 
   ∃ f : ℝ → ℝ, f x = (2*x - 1/x)^6 ∧ 
   ∀ ε > 0, ∃ δ > 0, ∀ x, |x| < δ → |f x - c| < ε) :=
sorry

end NUMINAMATH_CALUDE_constant_term_expansion_l3759_375901


namespace NUMINAMATH_CALUDE_union_of_A_and_B_l3759_375930

def A : Set ℕ := {0, 1, 2}

def B : Set ℕ := {x | ∃ a ∈ A, x = 2^a}

theorem union_of_A_and_B : A ∪ B = {0, 1, 2, 4} := by
  sorry

end NUMINAMATH_CALUDE_union_of_A_and_B_l3759_375930


namespace NUMINAMATH_CALUDE_solve_for_a_l3759_375900

def A (a b : ℝ) : Set ℝ := {x | x^2 + a*x + b = 0}

def B : Set ℝ := {3}

theorem solve_for_a (a b : ℝ) (h : A a b = B) : a = -6 := by
  sorry

end NUMINAMATH_CALUDE_solve_for_a_l3759_375900


namespace NUMINAMATH_CALUDE_cubic_equation_sum_l3759_375911

theorem cubic_equation_sum (a b c : ℝ) : 
  (a^3 - 6*a^2 + 11*a = 6) → 
  (b^3 - 6*b^2 + 11*b = 6) → 
  (c^3 - 6*c^2 + 11*c = 6) → 
  (a*b/c + b*c/a + c*a/b = 49/6) := by
sorry

end NUMINAMATH_CALUDE_cubic_equation_sum_l3759_375911


namespace NUMINAMATH_CALUDE_factorization_problem_l3759_375966

theorem factorization_problem (x y : ℝ) : (y + 2*x)^2 - (x + 2*y)^2 = 3*(x + y)*(x - y) := by
  sorry

end NUMINAMATH_CALUDE_factorization_problem_l3759_375966


namespace NUMINAMATH_CALUDE_leo_current_weight_l3759_375967

/-- Leo's current weight in pounds -/
def leo_weight : ℝ := 92

/-- Kendra's current weight in pounds -/
def kendra_weight : ℝ := 160 - leo_weight

/-- Theorem stating that Leo's current weight is 92 pounds -/
theorem leo_current_weight :
  (leo_weight + 10 = 1.5 * kendra_weight) ∧
  (leo_weight + kendra_weight = 160) →
  leo_weight = 92 := by
  sorry

end NUMINAMATH_CALUDE_leo_current_weight_l3759_375967


namespace NUMINAMATH_CALUDE_m_value_proof_l3759_375902

theorem m_value_proof (m : ℤ) (h : m < (Real.sqrt 11 - 1) / 2 ∧ (Real.sqrt 11 - 1) / 2 < m + 1) : m = 1 := by
  sorry

end NUMINAMATH_CALUDE_m_value_proof_l3759_375902


namespace NUMINAMATH_CALUDE_sandwich_combinations_l3759_375933

theorem sandwich_combinations (n_meat : Nat) (n_cheese : Nat) : 
  n_meat = 12 → n_cheese = 11 → (n_meat.choose 1) * (n_cheese.choose 3) = 1980 := by
  sorry

end NUMINAMATH_CALUDE_sandwich_combinations_l3759_375933


namespace NUMINAMATH_CALUDE_y_derivative_l3759_375958

noncomputable def y (x : ℝ) : ℝ := 
  (3 * x^2 - 4 * x + 2) * Real.sqrt (9 * x^2 - 12 * x + 3) + 
  (3 * x - 2)^4 * Real.arcsin (1 / (3 * x - 2))

theorem y_derivative (x : ℝ) (h : 3 * x - 2 > 0) : 
  deriv y x = 12 * (3 * x - 2)^3 * Real.arcsin (1 / (3 * x - 2)) := by
  sorry

end NUMINAMATH_CALUDE_y_derivative_l3759_375958


namespace NUMINAMATH_CALUDE_min_value_expression_l3759_375947

theorem min_value_expression (x y k : ℝ) : (x*y - k)^2 + (x + y - 1)^2 ≥ 1 := by
  sorry

end NUMINAMATH_CALUDE_min_value_expression_l3759_375947


namespace NUMINAMATH_CALUDE_business_join_time_l3759_375905

/-- Represents the number of months in a year -/
def months_in_year : ℕ := 12

/-- Represents A's investment in Rupees -/
def investment_A : ℕ := 36000

/-- Represents B's investment in Rupees -/
def investment_B : ℕ := 54000

/-- Represents the ratio of A's profit share to B's profit share -/
def profit_ratio : ℚ := 2 / 1

theorem business_join_time (x : ℕ) : 
  (investment_A * months_in_year : ℚ) / (investment_B * (months_in_year - x)) = profit_ratio →
  x = 8 :=
by sorry

end NUMINAMATH_CALUDE_business_join_time_l3759_375905


namespace NUMINAMATH_CALUDE_expression_equivalence_l3759_375959

theorem expression_equivalence (x y : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) (h : x = 2 * y) :
  (x - 2 / x) * (y + 2 / y) = (1 / 2) * (x^2 - 2*x + 8 - 16 / x) := by
  sorry

end NUMINAMATH_CALUDE_expression_equivalence_l3759_375959


namespace NUMINAMATH_CALUDE_exponent_equation_solution_l3759_375936

theorem exponent_equation_solution :
  ∃ x : ℝ, (4 : ℝ)^x * (4 : ℝ)^x * (4 : ℝ)^x * (4 : ℝ)^x = (256 : ℝ)^4 ∧ x = 4 := by
  sorry

end NUMINAMATH_CALUDE_exponent_equation_solution_l3759_375936


namespace NUMINAMATH_CALUDE_retailer_profit_percent_l3759_375973

/-- Calculates the profit percent given purchase price, overhead expenses, and selling price -/
def profit_percent (purchase_price overhead_expenses selling_price : ℚ) : ℚ :=
  let cost_price := purchase_price + overhead_expenses
  let profit := selling_price - cost_price
  (profit / cost_price) * 100

/-- Theorem: The profit percent for the given values is 25% -/
theorem retailer_profit_percent :
  profit_percent 225 15 300 = 25 := by
  sorry

end NUMINAMATH_CALUDE_retailer_profit_percent_l3759_375973


namespace NUMINAMATH_CALUDE_paper_folding_thickness_l3759_375968

/-- The thickness of the paper after n folds -/
def thickness (n : ℕ) : ℚ := (1 / 10) * 2^n

/-- The minimum number of folds required to exceed 12mm -/
def min_folds : ℕ := 7

theorem paper_folding_thickness :
  (∀ k < min_folds, thickness k ≤ 12) ∧ thickness min_folds > 12 := by
  sorry

end NUMINAMATH_CALUDE_paper_folding_thickness_l3759_375968


namespace NUMINAMATH_CALUDE_square_sum_equality_l3759_375978

theorem square_sum_equality (x y z : ℝ) 
  (h1 : x^2 + 4*y^2 + 16*z^2 = 48) 
  (h2 : x*y + 4*y*z + 2*z*x = 24) : 
  x^2 + y^2 + z^2 = 21 := by
  sorry

end NUMINAMATH_CALUDE_square_sum_equality_l3759_375978


namespace NUMINAMATH_CALUDE_goods_train_speed_l3759_375929

/-- The speed of the goods train given the conditions of the problem -/
theorem goods_train_speed 
  (man_train_speed : ℝ) 
  (goods_train_length : ℝ) 
  (passing_time : ℝ) 
  (h1 : man_train_speed = 60) 
  (h2 : goods_train_length = 0.3) -- 300 m converted to km
  (h3 : passing_time = 1/300) -- 12 seconds converted to hours
  : ∃ (goods_train_speed : ℝ), goods_train_speed = 30 :=
by sorry

end NUMINAMATH_CALUDE_goods_train_speed_l3759_375929


namespace NUMINAMATH_CALUDE_distance_AB_distance_AB_value_l3759_375923

def path_north : ℝ := 30 - 15 + 10
def path_east : ℝ := 80 - 30

theorem distance_AB : ℝ :=
  let north_south_distance := path_north
  let east_west_distance := path_east
  Real.sqrt (north_south_distance ^ 2 + east_west_distance ^ 2)

theorem distance_AB_value : distance_AB = 25 * Real.sqrt 5 := by sorry

end NUMINAMATH_CALUDE_distance_AB_distance_AB_value_l3759_375923


namespace NUMINAMATH_CALUDE_two_zeros_twelve_divisors_l3759_375924

def endsWithTwoZeros (n : ℕ) : Prop := n % 100 = 0

def countDivisors (n : ℕ) : ℕ := (Finset.filter (· ∣ n) (Finset.range (n + 1))).card

theorem two_zeros_twelve_divisors :
  ∀ n : ℕ, endsWithTwoZeros n ∧ countDivisors n = 12 ↔ n = 200 ∨ n = 500 := by
  sorry

end NUMINAMATH_CALUDE_two_zeros_twelve_divisors_l3759_375924


namespace NUMINAMATH_CALUDE_checkerboard_area_equality_l3759_375963

-- Define a convex quadrilateral
structure ConvexQuadrilateral where
  vertices : Fin 4 → ℝ × ℝ
  is_convex : sorry -- Condition for convexity

-- Define the division points on the sides
def division_points (q : ConvexQuadrilateral) : Fin 4 → Fin 8 → ℝ × ℝ :=
  sorry -- Function that returns the division points on each side

-- Define the cells formed by connecting corresponding division points
def cells (q : ConvexQuadrilateral) : List (List (ℝ × ℝ)) :=
  sorry -- List of cells, each cell represented by its vertices

-- Define the area of a cell
def cell_area (cell : List (ℝ × ℝ)) : ℝ :=
  sorry -- Function to calculate the area of a cell

-- Define the sum of areas of alternating cells (checkerboard pattern)
def alternating_sum (cells : List (List (ℝ × ℝ))) : ℝ :=
  sorry -- Sum of areas of alternating cells

-- The theorem to be proved
theorem checkerboard_area_equality (q : ConvexQuadrilateral) :
  let c := cells q
  alternating_sum c = alternating_sum (List.drop 1 c) :=
sorry

end NUMINAMATH_CALUDE_checkerboard_area_equality_l3759_375963


namespace NUMINAMATH_CALUDE_marble_probability_l3759_375991

theorem marble_probability (total red blue : ℕ) (h1 : total = 20) (h2 : red = 7) (h3 : blue = 5) :
  let white := total - (red + blue)
  (red + white : ℚ) / total = 3 / 4 := by sorry

end NUMINAMATH_CALUDE_marble_probability_l3759_375991


namespace NUMINAMATH_CALUDE_octal_calculation_l3759_375950

/-- Represents a number in base 8 --/
def OctalNumber := Nat

/-- Convert a decimal number to its octal representation --/
def toOctal (n : Nat) : OctalNumber :=
  sorry

/-- Add two octal numbers --/
def octalAdd (a b : OctalNumber) : OctalNumber :=
  sorry

/-- Subtract two octal numbers --/
def octalSub (a b : OctalNumber) : OctalNumber :=
  sorry

/-- Theorem: 72₈ - 45₈ + 23₈ = 50₈ in base 8 --/
theorem octal_calculation :
  octalAdd (octalSub (toOctal 72) (toOctal 45)) (toOctal 23) = toOctal 50 := by
  sorry

end NUMINAMATH_CALUDE_octal_calculation_l3759_375950


namespace NUMINAMATH_CALUDE_trig_identity_l3759_375960

theorem trig_identity : 
  Real.sin (20 * π / 180) * Real.sin (80 * π / 180) - 
  Real.cos (160 * π / 180) * Real.sin (10 * π / 180) = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_trig_identity_l3759_375960


namespace NUMINAMATH_CALUDE_sum_of_common_divisors_l3759_375942

def number_list : List Int := [48, 96, -16, 144, 192]

def is_common_divisor (d : Nat) : Bool :=
  number_list.all (fun n => n % d = 0)

def common_divisors : List Nat :=
  (List.range 193).filter is_common_divisor

theorem sum_of_common_divisors : (common_divisors.sum) = 15 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_common_divisors_l3759_375942


namespace NUMINAMATH_CALUDE_lcm_from_hcf_and_product_l3759_375944

theorem lcm_from_hcf_and_product (a b : ℕ+) : 
  Nat.gcd a b = 12 → a * b = 2460 → Nat.lcm a b = 205 := by
  sorry

end NUMINAMATH_CALUDE_lcm_from_hcf_and_product_l3759_375944


namespace NUMINAMATH_CALUDE_percent_of_a_is_4b_l3759_375972

theorem percent_of_a_is_4b (a b : ℝ) (h : a = 1.2 * b) : (4 * b) / a = 10/3 := by
  sorry

end NUMINAMATH_CALUDE_percent_of_a_is_4b_l3759_375972


namespace NUMINAMATH_CALUDE_possible_values_of_a_l3759_375988

def A (a : ℝ) : Set ℝ := {x | a * x + 1 = 0}
def B : Set ℝ := {x | x^2 - 3*x + 2 = 0}

theorem possible_values_of_a :
  ∀ a : ℝ, (A a ∪ B = B) ↔ (a = -1/2 ∨ a = 0 ∨ a = -1) :=
by sorry

end NUMINAMATH_CALUDE_possible_values_of_a_l3759_375988


namespace NUMINAMATH_CALUDE_ellipse_eccentricity_l3759_375943

/-- Definition of an ellipse with foci F₁ and F₂ -/
structure Ellipse where
  F₁ : ℝ × ℝ
  F₂ : ℝ × ℝ

/-- The maximum distance from a point on the ellipse to F₁ -/
def max_distance (E : Ellipse) : ℝ := 7

/-- The minimum distance from a point on the ellipse to F₁ -/
def min_distance (E : Ellipse) : ℝ := 1

/-- The eccentricity of an ellipse -/
def eccentricity (E : Ellipse) : ℝ := sorry

/-- Theorem: The square root of the eccentricity of the ellipse E is √3/2 -/
theorem ellipse_eccentricity (E : Ellipse) :
  Real.sqrt (eccentricity E) = Real.sqrt 3 / 2 := by sorry

end NUMINAMATH_CALUDE_ellipse_eccentricity_l3759_375943


namespace NUMINAMATH_CALUDE_sqrt_meaningful_range_l3759_375926

theorem sqrt_meaningful_range (x : ℝ) : 
  (∃ y : ℝ, y ^ 2 = 2 * x - 4) ↔ x ≥ 2 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_meaningful_range_l3759_375926


namespace NUMINAMATH_CALUDE_smallest_number_l3759_375919

theorem smallest_number (s : Set ℤ) (hs : s = {-2, 0, -1, 3}) : 
  ∃ m ∈ s, ∀ x ∈ s, m ≤ x ∧ m = -2 :=
by sorry

end NUMINAMATH_CALUDE_smallest_number_l3759_375919


namespace NUMINAMATH_CALUDE_satellite_upgraded_fraction_l3759_375928

/-- Represents a satellite with modular units and sensors -/
structure Satellite :=
  (units : ℕ)
  (non_upgraded_per_unit : ℕ)
  (upgraded_total : ℕ)

/-- The fraction of upgraded sensors on the satellite -/
def upgraded_fraction (s : Satellite) : ℚ :=
  s.upgraded_total / (s.units * s.non_upgraded_per_unit + s.upgraded_total)

theorem satellite_upgraded_fraction
  (s : Satellite)
  (h1 : s.units = 24)
  (h2 : s.non_upgraded_per_unit = s.upgraded_total / 6) :
  upgraded_fraction s = 1/5 := by
  sorry

end NUMINAMATH_CALUDE_satellite_upgraded_fraction_l3759_375928


namespace NUMINAMATH_CALUDE_buses_needed_l3759_375918

theorem buses_needed (students : ℕ) (seats_per_bus : ℕ) (h1 : students = 28) (h2 : seats_per_bus = 7) :
  (students + seats_per_bus - 1) / seats_per_bus = 4 := by
  sorry

end NUMINAMATH_CALUDE_buses_needed_l3759_375918


namespace NUMINAMATH_CALUDE_simplify_expression_l3759_375998

theorem simplify_expression (a b : ℝ) : (30*a + 70*b) + (15*a + 45*b) - (12*a + 60*b) = 33*a + 55*b := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l3759_375998


namespace NUMINAMATH_CALUDE_puppy_weight_l3759_375965

theorem puppy_weight (puppy smaller_cat larger_cat : ℝ) 
  (total_weight : puppy + smaller_cat + larger_cat = 24)
  (puppy_larger_cat : puppy + larger_cat = 2 * smaller_cat)
  (puppy_smaller_cat : puppy + smaller_cat = larger_cat) :
  puppy = 4 := by
  sorry

end NUMINAMATH_CALUDE_puppy_weight_l3759_375965


namespace NUMINAMATH_CALUDE_bankers_gain_specific_case_l3759_375985

/-- Calculates the banker's gain given the banker's discount, interest rate, and time period. -/
def bankers_gain (bankers_discount : ℚ) (interest_rate : ℚ) (time : ℚ) : ℚ :=
  (bankers_discount * interest_rate * time) / (100 + (interest_rate * time))

/-- Theorem stating that given the specific conditions, the banker's gain is 90. -/
theorem bankers_gain_specific_case :
  bankers_gain 340 12 3 = 90 := by
  sorry

end NUMINAMATH_CALUDE_bankers_gain_specific_case_l3759_375985


namespace NUMINAMATH_CALUDE_power_two_305_mod_9_l3759_375995

theorem power_two_305_mod_9 : 2^305 % 9 = 5 := by sorry

end NUMINAMATH_CALUDE_power_two_305_mod_9_l3759_375995


namespace NUMINAMATH_CALUDE_carla_class_size_l3759_375932

theorem carla_class_size :
  let students_in_restroom : ℕ := 2
  let absent_students : ℕ := 3 * students_in_restroom - 1
  let total_desks : ℕ := 4 * 6
  let occupied_desks : ℕ := (2 * total_desks) / 3
  let students_present : ℕ := occupied_desks
  students_in_restroom + absent_students + students_present = 23 := by
sorry

end NUMINAMATH_CALUDE_carla_class_size_l3759_375932


namespace NUMINAMATH_CALUDE_square_field_side_length_l3759_375914

theorem square_field_side_length (area : ℝ) (side : ℝ) :
  area = 225 →
  side * side = area →
  side = 15 := by
  sorry

end NUMINAMATH_CALUDE_square_field_side_length_l3759_375914


namespace NUMINAMATH_CALUDE_nth_power_divisors_l3759_375907

theorem nth_power_divisors (n : ℕ+) : 
  (∃ (d : ℕ), d = (Finset.card (Nat.divisors (n^n.val)))) → 
  d = 861 → 
  n = 20 := by
sorry

end NUMINAMATH_CALUDE_nth_power_divisors_l3759_375907


namespace NUMINAMATH_CALUDE_felicity_gas_usage_l3759_375934

/-- Proves that Felicity used 23 gallons of gas given the problem conditions -/
theorem felicity_gas_usage (adhira : ℝ) : 
  (adhira + (4 * adhira - 5) = 30) → (4 * adhira - 5 = 23) :=
by
  sorry

end NUMINAMATH_CALUDE_felicity_gas_usage_l3759_375934


namespace NUMINAMATH_CALUDE_santa_gifts_l3759_375906

theorem santa_gifts (x : ℕ) (h1 : x < 100) (h2 : x % 2 = 0) (h3 : x % 5 = 0) (h4 : x % 7 = 0) :
  x - (x / 2 + x / 5 + x / 7) = 11 :=
by sorry

end NUMINAMATH_CALUDE_santa_gifts_l3759_375906


namespace NUMINAMATH_CALUDE_ratio_equation_solution_sum_l3759_375992

theorem ratio_equation_solution_sum : 
  ∃! s : ℝ, ∀ x : ℝ, (3 * x + 4) / (5 * x + 4) = (5 * x + 6) / (8 * x + 6) → s = x :=
by
  sorry

end NUMINAMATH_CALUDE_ratio_equation_solution_sum_l3759_375992


namespace NUMINAMATH_CALUDE_reciprocal_squares_sum_of_product_five_l3759_375976

theorem reciprocal_squares_sum_of_product_five (a b : ℕ) (h : a * b = 5) :
  (1 : ℚ) / (a^2 : ℚ) + (1 : ℚ) / (b^2 : ℚ) = 26 / 25 := by
  sorry

end NUMINAMATH_CALUDE_reciprocal_squares_sum_of_product_five_l3759_375976


namespace NUMINAMATH_CALUDE_top_square_after_folds_l3759_375981

/-- Represents a 6x6 grid of numbers -/
def Grid := Fin 6 → Fin 6 → Nat

/-- Initial grid configuration -/
def initial_grid : Grid :=
  fun i j => 6 * i.val + j.val + 1

/-- Fold operation types -/
inductive FoldType
  | TopOver
  | BottomOver
  | RightOver
  | LeftOver

/-- Apply a single fold operation to the grid -/
def apply_fold (g : Grid) (ft : FoldType) : Grid :=
  sorry  -- Implementation of folding logic

/-- Sequence of folds as described in the problem -/
def fold_sequence : List FoldType :=
  [FoldType.TopOver, FoldType.BottomOver, FoldType.RightOver, 
   FoldType.LeftOver, FoldType.TopOver, FoldType.RightOver]

/-- Apply a sequence of folds to the grid -/
def apply_fold_sequence (g : Grid) (folds : List FoldType) : Grid :=
  sorry  -- Implementation of applying multiple folds

theorem top_square_after_folds (g : Grid) :
  g = initial_grid →
  (apply_fold_sequence g fold_sequence) 0 0 = 22 :=
sorry

end NUMINAMATH_CALUDE_top_square_after_folds_l3759_375981


namespace NUMINAMATH_CALUDE_negative_b_from_cubic_inequality_l3759_375938

theorem negative_b_from_cubic_inequality (a b : ℝ) 
  (h1 : a * b ≠ 0)
  (h2 : ∀ x : ℝ, x ≥ 0 → (x - a) * (x - b) * (x - 2*a - b) ≥ 0) :
  b < 0 := by
  sorry

end NUMINAMATH_CALUDE_negative_b_from_cubic_inequality_l3759_375938


namespace NUMINAMATH_CALUDE_base_conversion_sum_l3759_375916

/-- Converts a number from base 8 to base 10 -/
def base8ToBase10 (n : Nat) : Nat :=
  sorry

/-- Converts a number from base 14 to base 10 -/
def base14ToBase10 (n : Nat) (c d : Nat) : Nat :=
  sorry

theorem base_conversion_sum :
  let c : Nat := 12
  let d : Nat := 13
  base8ToBase10 356 + base14ToBase10 4 c d = 1203 := by
  sorry

end NUMINAMATH_CALUDE_base_conversion_sum_l3759_375916


namespace NUMINAMATH_CALUDE_solve_equation_l3759_375952

theorem solve_equation : ∃ x : ℝ, (7 - x = 9.5) ∧ (x = -2.5) := by sorry

end NUMINAMATH_CALUDE_solve_equation_l3759_375952


namespace NUMINAMATH_CALUDE_inverse_variation_problem_l3759_375996

theorem inverse_variation_problem (x y : ℝ) :
  (∀ (x y : ℝ), x > 0 ∧ y > 0) →
  (∃ (k : ℝ), ∀ (x y : ℝ), x^3 * y = k) →
  (2^3 * 5 = k) →
  (x^3 * 2000 = k) →
  x = 1 / Real.rpow 50 (1/3) :=
by sorry

end NUMINAMATH_CALUDE_inverse_variation_problem_l3759_375996


namespace NUMINAMATH_CALUDE_frog_hop_probability_l3759_375974

/-- Represents the possible positions on a 3x3 grid -/
inductive Position
  | Center
  | Edge
  | Corner

/-- Represents a single hop of the frog -/
def hop (pos : Position) : Position :=
  match pos with
  | Position.Center => Position.Edge
  | Position.Edge => sorry  -- Randomly choose between Center, Edge, or Corner
  | Position.Corner => Position.Corner

/-- Calculates the probability of landing on a corner exactly once in at most four hops -/
def prob_corner_once (hops : Nat) : ℚ :=
  sorry  -- Implement the probability calculation

/-- The main theorem stating the probability of landing on a corner exactly once in at most four hops -/
theorem frog_hop_probability : 
  prob_corner_once 4 = 25 / 32 := by
  sorry


end NUMINAMATH_CALUDE_frog_hop_probability_l3759_375974


namespace NUMINAMATH_CALUDE_no_double_application_function_l3759_375986

theorem no_double_application_function : ¬∃ f : ℕ → ℕ, ∀ x : ℕ, f (f x) = x + 1 := by
  sorry

end NUMINAMATH_CALUDE_no_double_application_function_l3759_375986


namespace NUMINAMATH_CALUDE_trailing_zeros_310_factorial_l3759_375990

/-- The number of trailing zeros in n! -/
def trailingZeros (n : ℕ) : ℕ :=
  (n / 5) + (n / 25) + (n / 125)

theorem trailing_zeros_310_factorial :
  trailingZeros 310 = 76 := by
  sorry

end NUMINAMATH_CALUDE_trailing_zeros_310_factorial_l3759_375990


namespace NUMINAMATH_CALUDE_initial_cooking_time_l3759_375987

/-- The initial cooking time for french fries given the recommended time and remaining time -/
theorem initial_cooking_time (recommended_time_minutes : ℕ) (remaining_time_seconds : ℕ) :
  let recommended_time_seconds := recommended_time_minutes * 60
  recommended_time_seconds - remaining_time_seconds = 45 :=
by
  sorry

#check initial_cooking_time 5 255

end NUMINAMATH_CALUDE_initial_cooking_time_l3759_375987


namespace NUMINAMATH_CALUDE_movie_group_composition_l3759_375915

-- Define the ticket prices and group information
def adult_price : ℚ := 9.5
def child_price : ℚ := 6.5
def total_people : ℕ := 7
def total_paid : ℚ := 54.5

-- Define the theorem
theorem movie_group_composition :
  ∃ (adults : ℕ) (children : ℕ),
    adults + children = total_people ∧
    (adult_price * adults + child_price * children : ℚ) = total_paid ∧
    adults = 3 := by
  sorry

end NUMINAMATH_CALUDE_movie_group_composition_l3759_375915


namespace NUMINAMATH_CALUDE_checkerboard_exists_l3759_375940

/-- Represents the color of a cell -/
inductive Color
| Black
| White

/-- Represents a 100x100 board -/
def Board := Fin 100 → Fin 100 → Color

/-- Checks if a cell is adjacent to the boundary -/
def isAdjacentToBoundary (i j : Fin 100) : Prop :=
  i = 0 ∨ i = 99 ∨ j = 0 ∨ j = 99

/-- Checks if a 2x2 square is monochromatic -/
def isMonochromatic (board : Board) (i j : Fin 100) : Prop :=
  ∃ c : Color,
    board i j = c ∧
    board (i + 1) j = c ∧
    board i (j + 1) = c ∧
    board (i + 1) (j + 1) = c

/-- Checks if a 2x2 square has a checkerboard pattern -/
def isCheckerboard (board : Board) (i j : Fin 100) : Prop :=
  (board i j = board (i + 1) (j + 1) ∧
   board i (j + 1) = board (i + 1) j ∧
   board i j ≠ board i (j + 1)) ∨
  (board i j = board (i + 1) (j + 1) ∧
   board i (j + 1) = board (i + 1) j ∧
   board i j ≠ board (i + 1) j)

theorem checkerboard_exists (board : Board)
  (boundary_black : ∀ i j : Fin 100, isAdjacentToBoundary i j → board i j = Color.Black)
  (no_monochromatic : ∀ i j : Fin 100, ¬isMonochromatic board i j) :
  ∃ i j : Fin 100, isCheckerboard board i j :=
sorry

end NUMINAMATH_CALUDE_checkerboard_exists_l3759_375940


namespace NUMINAMATH_CALUDE_correct_list_price_l3759_375925

/-- The list price of the item -/
def list_price : ℝ := 45

/-- Alice's selling price -/
def alice_price (x : ℝ) : ℝ := x - 15

/-- Bob's selling price -/
def bob_price (x : ℝ) : ℝ := x - 25

/-- Alice's commission rate -/
def alice_rate : ℝ := 0.1

/-- Bob's commission rate -/
def bob_rate : ℝ := 0.15

/-- Theorem stating that the list price is correct -/
theorem correct_list_price :
  alice_rate * alice_price list_price = bob_rate * bob_price list_price :=
by sorry

end NUMINAMATH_CALUDE_correct_list_price_l3759_375925


namespace NUMINAMATH_CALUDE_largest_multiple_12_negation_gt_neg_150_l3759_375946

theorem largest_multiple_12_negation_gt_neg_150 :
  ∀ n : ℤ, n ≥ 0 → 12 ∣ n → -n > -150 → n ≤ 144 :=
by
  sorry

end NUMINAMATH_CALUDE_largest_multiple_12_negation_gt_neg_150_l3759_375946


namespace NUMINAMATH_CALUDE_discount_problem_l3759_375979

theorem discount_problem (original_price : ℝ) : 
  original_price > 0 → 
  0.7 * original_price + 0.8 * original_price = 50 → 
  original_price = 100 / 3 := by
sorry

end NUMINAMATH_CALUDE_discount_problem_l3759_375979


namespace NUMINAMATH_CALUDE_expression_equality_l3759_375956

theorem expression_equality (y : ℝ) (c : ℝ) (h : y > 0) :
  (4 * y) / 20 + (c * y) / 10 = y / 2 → c = 3 := by
  sorry

end NUMINAMATH_CALUDE_expression_equality_l3759_375956


namespace NUMINAMATH_CALUDE_acute_triangle_properties_l3759_375957

theorem acute_triangle_properties (A B C : ℝ) (a b c : ℝ) :
  0 < A ∧ A < π / 2 →
  0 < B ∧ B < π / 2 →
  0 < C ∧ C < π / 2 →
  A + B + C = π →
  a = 2 * Real.sin A →
  b = 2 * Real.sin B →
  c = 2 * Real.sin C →
  a - b = 2 * b * Real.cos C →
  (C = 2 * B) ∧
  (π / 6 < B ∧ B < π / 4) ∧
  (Real.sqrt 2 < c / b ∧ c / b < Real.sqrt 3) := by
  sorry

end NUMINAMATH_CALUDE_acute_triangle_properties_l3759_375957


namespace NUMINAMATH_CALUDE_pure_imaginary_condition_l3759_375909

theorem pure_imaginary_condition (a : ℝ) : 
  (∃ b : ℝ, (Complex.I * b : ℂ) = (1 + a * Complex.I) / (1 - Complex.I)) → a = 1 := by
  sorry

end NUMINAMATH_CALUDE_pure_imaginary_condition_l3759_375909


namespace NUMINAMATH_CALUDE_cubic_factorization_l3759_375977

theorem cubic_factorization (m : ℝ) : m^3 - 4*m^2 + 4*m = m*(m-2)^2 := by
  sorry

end NUMINAMATH_CALUDE_cubic_factorization_l3759_375977


namespace NUMINAMATH_CALUDE_sqrt_inequality_solution_set_l3759_375913

theorem sqrt_inequality_solution_set (x : ℝ) :
  (x^3 - 8) / x ≥ 0 →
  (Real.sqrt ((x^3 - 8) / x) > x - 2 ↔ x ∈ Set.Ioi 2 ∪ Set.Iio 0) :=
by sorry

end NUMINAMATH_CALUDE_sqrt_inequality_solution_set_l3759_375913


namespace NUMINAMATH_CALUDE_consecutive_integers_coprime_l3759_375997

theorem consecutive_integers_coprime (n : ℤ) : 
  ∃ k ∈ Finset.range 10, ∀ m ∈ Finset.range 10, m ≠ k → Int.gcd (n + k) (n + m) = 1 := by
  sorry

end NUMINAMATH_CALUDE_consecutive_integers_coprime_l3759_375997


namespace NUMINAMATH_CALUDE_circle_symmetry_l3759_375908

-- Define the original circle C
def circle_C (x y : ℝ) : Prop := x^2 + y^2 - 2*x - 2*y + 1 = 0

-- Define the line l
def line_l (x y : ℝ) : Prop := x - y = 2

-- Define the symmetric circle
def symmetric_circle (x y : ℝ) : Prop := (x-3)^2 + (y+1)^2 = 1

-- Theorem statement
theorem circle_symmetry :
  ∀ x y : ℝ, 
  (∃ x₀ y₀ : ℝ, circle_C x₀ y₀ ∧ 
    (x + x₀)/2 - (y + y₀)/2 = 2 ∧ 
    (y - y₀)/(x - x₀) = -1) →
  symmetric_circle x y :=
sorry

end NUMINAMATH_CALUDE_circle_symmetry_l3759_375908


namespace NUMINAMATH_CALUDE_travis_cereal_weeks_l3759_375937

/-- Proves the number of weeks Travis eats cereal given his consumption and spending habits -/
theorem travis_cereal_weeks (boxes_per_week : ℕ) (cost_per_box : ℚ) (total_spent : ℚ) :
  boxes_per_week = 2 →
  cost_per_box = 3 →
  total_spent = 312 →
  (total_spent / (boxes_per_week * cost_per_box) : ℚ) = 52 := by
  sorry

end NUMINAMATH_CALUDE_travis_cereal_weeks_l3759_375937


namespace NUMINAMATH_CALUDE_area_of_grid_with_cutouts_l3759_375980

/-- The area of a square grid with triangular cutouts -/
theorem area_of_grid_with_cutouts (grid_side : ℕ) (cell_side : ℝ) 
  (dark_grey_area : ℝ) (light_grey_area : ℝ) : 
  grid_side = 6 → 
  cell_side = 1 → 
  dark_grey_area = 3 → 
  light_grey_area = 6 → 
  (grid_side : ℝ) * (grid_side : ℝ) * cell_side * cell_side - dark_grey_area - light_grey_area = 27 := by
sorry

end NUMINAMATH_CALUDE_area_of_grid_with_cutouts_l3759_375980

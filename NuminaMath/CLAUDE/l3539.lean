import Mathlib

namespace NUMINAMATH_CALUDE_difference_of_squares_factorization_l3539_353940

theorem difference_of_squares_factorization (m : ℤ) : m^2 - 1 = (m + 1) * (m - 1) := by
  sorry

end NUMINAMATH_CALUDE_difference_of_squares_factorization_l3539_353940


namespace NUMINAMATH_CALUDE_student_grade_problem_l3539_353905

theorem student_grade_problem (grade_history grade_third : ℝ) 
  (h1 : grade_history = 84)
  (h2 : grade_third = 69)
  (h3 : (grade_math + grade_history + grade_third) / 3 = 75) :
  grade_math = 72 := by
  sorry

end NUMINAMATH_CALUDE_student_grade_problem_l3539_353905


namespace NUMINAMATH_CALUDE_complex_equation_sum_l3539_353985

theorem complex_equation_sum (x y : ℝ) : 
  (Complex.mk (x - 1) (y + 1)) * (Complex.mk 2 1) = 0 → x + y = 0 := by
sorry

end NUMINAMATH_CALUDE_complex_equation_sum_l3539_353985


namespace NUMINAMATH_CALUDE_product_of_solutions_l3539_353984

theorem product_of_solutions (x : ℝ) : 
  (25 = 3 * x^2 + 10 * x) → 
  (∃ α β : ℝ, (3 * α^2 + 10 * α = 25) ∧ (3 * β^2 + 10 * β = 25) ∧ (α * β = -25/3)) :=
by sorry

end NUMINAMATH_CALUDE_product_of_solutions_l3539_353984


namespace NUMINAMATH_CALUDE_calculate_remaining_student_age_l3539_353946

/-- Given a class of students with known average ages for subgroups,
    calculate the age of the remaining student. -/
theorem calculate_remaining_student_age
  (total_students : ℕ)
  (total_average : ℕ)
  (group1_students : ℕ)
  (group1_average : ℕ)
  (group2_students : ℕ)
  (group2_average : ℕ)
  (h1 : total_students = 25)
  (h2 : total_average = 25)
  (h3 : group1_students = 10)
  (h4 : group1_average = 22)
  (h5 : group2_students = 14)
  (h6 : group2_average = 28)
  (h7 : group1_students + group2_students + 1 = total_students) :
  total_students * total_average =
    group1_students * group1_average +
    group2_students * group2_average + 13 :=
by sorry

end NUMINAMATH_CALUDE_calculate_remaining_student_age_l3539_353946


namespace NUMINAMATH_CALUDE_sum_of_k_for_minimum_area_l3539_353920

/-- The sum of k values that minimize the triangle area --/
def sum_of_k_values : ℤ := 24

/-- Point type --/
structure Point where
  x : ℚ
  y : ℚ

/-- Triangle defined by three points --/
structure Triangle where
  p1 : Point
  p2 : Point
  p3 : Point

/-- Function to calculate the area of a triangle --/
def triangle_area (t : Triangle) : ℚ :=
  sorry

/-- Function to check if a triangle has minimum area --/
def has_minimum_area (t : Triangle) : Prop :=
  sorry

/-- Theorem stating the sum of k values that minimize the triangle area --/
theorem sum_of_k_for_minimum_area :
  ∃ (k1 k2 : ℤ),
    k1 ≠ k2 ∧
    has_minimum_area (Triangle.mk
      (Point.mk 2 9)
      (Point.mk 14 18)
      (Point.mk 6 k1)) ∧
    has_minimum_area (Triangle.mk
      (Point.mk 2 9)
      (Point.mk 14 18)
      (Point.mk 6 k2)) ∧
    k1 + k2 = sum_of_k_values :=
  sorry

end NUMINAMATH_CALUDE_sum_of_k_for_minimum_area_l3539_353920


namespace NUMINAMATH_CALUDE_cubic_root_sum_product_l3539_353966

theorem cubic_root_sum_product (p q r : ℂ) : 
  (3 * p^3 - 5 * p^2 + 12 * p - 10 = 0) ∧
  (3 * q^3 - 5 * q^2 + 12 * q - 10 = 0) ∧
  (3 * r^3 - 5 * r^2 + 12 * r - 10 = 0) →
  p * q + q * r + r * p = 4 := by
sorry

end NUMINAMATH_CALUDE_cubic_root_sum_product_l3539_353966


namespace NUMINAMATH_CALUDE_simplify_expression_l3539_353974

theorem simplify_expression (z : ℝ) : (3 - 5 * z^2) - (4 + 3 * z^2) = -1 - 8 * z^2 := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l3539_353974


namespace NUMINAMATH_CALUDE_painted_cube_theorem_l3539_353911

/-- Represents the number of painted faces a small cube can have -/
inductive PaintedFaces
  | one
  | two
  | three

/-- Represents a large cube that is painted on the outside and cut into smaller cubes -/
structure PaintedCube where
  edge_length : ℕ
  small_cube_length : ℕ

/-- Counts the number of small cubes with a specific number of painted faces -/
def count_painted_faces (cube : PaintedCube) (faces : PaintedFaces) : ℕ :=
  match faces with
  | PaintedFaces.one => 0   -- Placeholder, actual calculation needed
  | PaintedFaces.two => 0   -- Placeholder, actual calculation needed
  | PaintedFaces.three => 0 -- Placeholder, actual calculation needed

/-- Theorem stating the correct count of small cubes with different numbers of painted faces -/
theorem painted_cube_theorem (cube : PaintedCube) 
    (h1 : cube.edge_length = 10)
    (h2 : cube.small_cube_length = 1) :
    count_painted_faces cube PaintedFaces.three = 8 ∧
    count_painted_faces cube PaintedFaces.two = 96 ∧
    count_painted_faces cube PaintedFaces.one = 384 := by
  sorry

end NUMINAMATH_CALUDE_painted_cube_theorem_l3539_353911


namespace NUMINAMATH_CALUDE_small_circle_radius_l3539_353919

theorem small_circle_radius (R : ℝ) (h : R = 5) :
  let d := Real.sqrt (2 * R^2)
  let r := (d - 2*R) / 2
  r = (Real.sqrt 200 - 10) / 2 := by sorry

end NUMINAMATH_CALUDE_small_circle_radius_l3539_353919


namespace NUMINAMATH_CALUDE_not_perfect_square_sum_of_squares_l3539_353933

theorem not_perfect_square_sum_of_squares (x y : ℤ) :
  ¬ ∃ (n : ℤ), (x^2 + x + 1)^2 + (y^2 + y + 1)^2 = n^2 := by
sorry

end NUMINAMATH_CALUDE_not_perfect_square_sum_of_squares_l3539_353933


namespace NUMINAMATH_CALUDE_hyperbola_properties_l3539_353912

/-- Represents a hyperbola with the equation (x^2 / a^2) - (y^2 / b^2) = 1 -/
structure Hyperbola (a b : ℝ) where
  equation : ∀ x y : ℝ, x^2 / a^2 - y^2 / b^2 = 1

/-- Represents an asymptote of a hyperbola -/
structure Asymptote (m : ℝ) where
  equation : ∀ x y : ℝ, y = m * x

/-- Represents a focus point of a hyperbola -/
structure Focus (x y : ℝ) where
  coordinates : ℝ × ℝ := (x, y)

theorem hyperbola_properties (h : Hyperbola 12 9) :
  (∃ a₁ : Asymptote (3/4), True) ∧
  (∃ a₂ : Asymptote (-3/4), True) ∧
  (∃ f₁ : Focus 15 0, True) ∧
  (∃ f₂ : Focus (-15) 0, True) := by
  sorry


end NUMINAMATH_CALUDE_hyperbola_properties_l3539_353912


namespace NUMINAMATH_CALUDE_disjunction_is_true_l3539_353959

-- Define the propositions p and q
def p : Prop := ∀ a b : ℝ, a > |b| → a^2 > b^2
def q : Prop := ∀ x : ℝ, x^2 = 4 → x = 2

-- State the theorem
theorem disjunction_is_true (hp : p) (hq : ¬q) : p ∨ q := by
  sorry

end NUMINAMATH_CALUDE_disjunction_is_true_l3539_353959


namespace NUMINAMATH_CALUDE_rectangle_shorter_side_l3539_353929

theorem rectangle_shorter_side
  (area : ℝ) (perimeter : ℝ)
  (h_area : area = 117)
  (h_perimeter : perimeter = 44)
  : ∃ (length width : ℝ),
    length * width = area ∧
    2 * (length + width) = perimeter ∧
    min length width = 9 :=
by sorry

end NUMINAMATH_CALUDE_rectangle_shorter_side_l3539_353929


namespace NUMINAMATH_CALUDE_factorization_a_squared_minus_ab_l3539_353980

theorem factorization_a_squared_minus_ab (a b : ℝ) : a^2 - a*b = a*(a - b) := by
  sorry

end NUMINAMATH_CALUDE_factorization_a_squared_minus_ab_l3539_353980


namespace NUMINAMATH_CALUDE_larger_cuboid_length_l3539_353993

/-- Proves that the length of a larger cuboid is 12 meters, given its width, height, and the number and dimensions of smaller cuboids it can be divided into. -/
theorem larger_cuboid_length (width height : ℝ) (num_small_cuboids : ℕ) 
  (small_length small_width small_height : ℝ) : 
  width = 14 →
  height = 10 →
  num_small_cuboids = 56 →
  small_length = 5 →
  small_width = 3 →
  small_height = 2 →
  (width * height * (num_small_cuboids * small_length * small_width * small_height) / (width * height)) = 12 :=
by sorry

end NUMINAMATH_CALUDE_larger_cuboid_length_l3539_353993


namespace NUMINAMATH_CALUDE_intersection_x_coordinate_l3539_353935

theorem intersection_x_coordinate (x y : ℤ) : 
  (y ≡ 3 * x + 4 [ZMOD 9]) → 
  (y ≡ 7 * x + 2 [ZMOD 9]) → 
  (x ≡ 5 [ZMOD 9]) := by
sorry

end NUMINAMATH_CALUDE_intersection_x_coordinate_l3539_353935


namespace NUMINAMATH_CALUDE_divisible_by_three_l3539_353937

theorem divisible_by_three (A B : ℤ) (h : A > B) :
  ∃ x : ℤ, (x = A ∨ x = B ∨ x = A + B ∨ x = A - B) ∧ x % 3 = 0 := by
  sorry

end NUMINAMATH_CALUDE_divisible_by_three_l3539_353937


namespace NUMINAMATH_CALUDE_imon_disentanglement_l3539_353939

/-- Represents the set of imons and their entanglements -/
structure ImonConfiguration where
  imons : Set Nat
  entangled : Nat → Nat → Bool

/-- Operation (i): Remove an imon entangled with an odd number of other imons -/
def removeOddEntangled (config : ImonConfiguration) : ImonConfiguration :=
  sorry

/-- Operation (ii): Double the set of imons -/
def doubleImons (config : ImonConfiguration) : ImonConfiguration :=
  sorry

/-- Checks if there are any entangled imons in the configuration -/
def hasEntangledImons (config : ImonConfiguration) : Bool :=
  sorry

/-- Represents a sequence of operations -/
inductive Operation
  | Remove
  | Double

theorem imon_disentanglement 
  (initial : ImonConfiguration) : 
  ∃ (ops : List Operation), 
    let final := ops.foldl (λ config op => 
      match op with
      | Operation.Remove => removeOddEntangled config
      | Operation.Double => doubleImons config
    ) initial
    ¬ hasEntangledImons final :=
  sorry

end NUMINAMATH_CALUDE_imon_disentanglement_l3539_353939


namespace NUMINAMATH_CALUDE_power_sum_difference_l3539_353915

theorem power_sum_difference : 3^(2+3+4) - (3^2 + 3^3 + 3^4 + 3^5) = 19323 := by
  sorry

end NUMINAMATH_CALUDE_power_sum_difference_l3539_353915


namespace NUMINAMATH_CALUDE_arctg_difference_bound_l3539_353964

theorem arctg_difference_bound (a b : ℝ) : 
  |Real.arctan a - Real.arctan b| ≤ |b - a| := by sorry

end NUMINAMATH_CALUDE_arctg_difference_bound_l3539_353964


namespace NUMINAMATH_CALUDE_min_omega_value_l3539_353906

theorem min_omega_value (ω : ℝ) (h1 : ω > 0) : 
  (∀ x, 2 * Real.cos (ω * (x - π/5) + π/5) = 2 * Real.sin (ω * x + π/5)) →
  ω ≥ 5/2 ∧ (∀ ω' > 0, (∀ x, 2 * Real.cos (ω' * (x - π/5) + π/5) = 2 * Real.sin (ω' * x + π/5)) → ω' ≥ ω) :=
by sorry

end NUMINAMATH_CALUDE_min_omega_value_l3539_353906


namespace NUMINAMATH_CALUDE_range_of_a_l3539_353991

-- Define the sets M and N
def M : Set ℝ := {x | x - 2 < 0}
def N (a : ℝ) : Set ℝ := {x | x < a}

-- Define the theorem
theorem range_of_a (a : ℝ) : M ⊆ N a ↔ a ∈ Set.Ici 2 := by
  sorry

-- Note: Set.Ici 2 represents the set [2, +∞) in Lean

end NUMINAMATH_CALUDE_range_of_a_l3539_353991


namespace NUMINAMATH_CALUDE_point_not_on_line_l3539_353914

/-- Given a line y = mx + b where m and b are real numbers satisfying mb > 0,
    prove that the point (2023, 0) cannot lie on this line. -/
theorem point_not_on_line (m b : ℝ) (h : m * b > 0) :
  ¬(0 = 2023 * m + b) := by
  sorry

end NUMINAMATH_CALUDE_point_not_on_line_l3539_353914


namespace NUMINAMATH_CALUDE_locus_is_ellipse_l3539_353938

/-- The locus of points (x, y) in the complex plane satisfying the given equation is an ellipse -/
theorem locus_is_ellipse (x y : ℝ) : 
  let z : ℂ := x + y * Complex.I
  (Complex.abs (z - (2 - Complex.I)) + Complex.abs (z - (-3 + Complex.I)) = 6) →
  ∃ (a b c d e f : ℝ), 
    a * x^2 + b * x * y + c * y^2 + d * x + e * y + f = 0 ∧ 
    b^2 - 4*a*c < 0 :=
by sorry

end NUMINAMATH_CALUDE_locus_is_ellipse_l3539_353938


namespace NUMINAMATH_CALUDE_max_value_implies_b_equals_two_l3539_353922

def is_valid_triple (a b c : ℕ) : Prop :=
  a ≠ b ∧ b ≠ c ∧ a ≠ c ∧ 
  a ∈ ({2, 3, 6} : Set ℕ) ∧ 
  b ∈ ({2, 3, 6} : Set ℕ) ∧ 
  c ∈ ({2, 3, 6} : Set ℕ)

theorem max_value_implies_b_equals_two (a b c : ℕ) :
  is_valid_triple a b c →
  (a : ℚ) / (b / c) ≤ 9 →
  (∀ x y z : ℕ, is_valid_triple x y z → (x : ℚ) / (y / z) ≤ 9) →
  b = 2 :=
sorry

end NUMINAMATH_CALUDE_max_value_implies_b_equals_two_l3539_353922


namespace NUMINAMATH_CALUDE_triangle_pairs_theorem_l3539_353926

def is_triangle (a b c : ℕ) : Prop :=
  a + b > c ∧ b + c > a ∧ c + a > b

def valid_triangle_pair (t1 t2 : ℕ × ℕ × ℕ) : Prop :=
  let (a, b, c) := t1
  let (d, e, f) := t2
  is_triangle a b c ∧ is_triangle d e f ∧ a + b + c + d + e + f = 16

theorem triangle_pairs_theorem :
  ∀ t1 t2 : ℕ × ℕ × ℕ,
  valid_triangle_pair t1 t2 →
  ((t1 = (4, 4, 3) ∧ t2 = (1, 2, 2)) ∨
   (t1 = (4, 4, 2) ∧ t2 = (2, 2, 2)) ∨
   (t1 = (4, 4, 1) ∧ t2 = (3, 2, 2)) ∨
   (t1 = (4, 4, 1) ∧ t2 = (3, 3, 1)) ∨
   (t2 = (4, 4, 3) ∧ t1 = (1, 2, 2)) ∨
   (t2 = (4, 4, 2) ∧ t1 = (2, 2, 2)) ∨
   (t2 = (4, 4, 1) ∧ t1 = (3, 2, 2)) ∨
   (t2 = (4, 4, 1) ∧ t1 = (3, 3, 1))) :=
by sorry

end NUMINAMATH_CALUDE_triangle_pairs_theorem_l3539_353926


namespace NUMINAMATH_CALUDE_trigonometric_identity_l3539_353997

theorem trigonometric_identity : 2 * Real.sin (390 * Real.pi / 180) - Real.tan (-45 * Real.pi / 180) + 5 * Real.cos (360 * Real.pi / 180) = 7 := by
  sorry

end NUMINAMATH_CALUDE_trigonometric_identity_l3539_353997


namespace NUMINAMATH_CALUDE_train_passing_jogger_time_l3539_353977

/-- Time for a train to pass a jogger given their speeds and initial positions -/
theorem train_passing_jogger_time
  (jogger_speed : ℝ)
  (train_speed : ℝ)
  (train_length : ℝ)
  (initial_distance : ℝ)
  (h_jogger_speed : jogger_speed = 9 * (1000 / 3600))
  (h_train_speed : train_speed = 45 * (1000 / 3600))
  (h_train_length : train_length = 120)
  (h_initial_distance : initial_distance = 250) :
  (initial_distance + train_length) / (train_speed - jogger_speed) = 37 := by
sorry

end NUMINAMATH_CALUDE_train_passing_jogger_time_l3539_353977


namespace NUMINAMATH_CALUDE_min_value_h_positive_m_l3539_353925

/-- The minimum value of ax - ln x for x > 0 and a ≥ 1 is 1 + ln a -/
theorem min_value_h (a : ℝ) (ha : a ≥ 1) :
  ∀ x > 0, a * x - Real.log x ≥ 1 + Real.log a := by sorry

/-- For all x > 0 and a ≥ 1, ax - ln(x + 1) > 0 -/
theorem positive_m (a : ℝ) (ha : a ≥ 1) :
  ∀ x > 0, a * x - Real.log (x + 1) > 0 := by sorry

end NUMINAMATH_CALUDE_min_value_h_positive_m_l3539_353925


namespace NUMINAMATH_CALUDE_monochromatic_rectangle_exists_l3539_353923

/-- A color type with four possible values -/
inductive Color
  | Red
  | Blue
  | Green
  | Yellow

/-- A point in the grid -/
structure Point where
  x : Fin 5
  y : Fin 41

/-- A coloring of the grid -/
def Coloring := Point → Color

/-- A rectangle in the grid -/
structure Rectangle where
  p1 : Point
  p2 : Point
  p3 : Point
  p4 : Point

/-- Predicate to check if four points form a valid rectangle with integer side lengths -/
def IsValidRectangle (r : Rectangle) : Prop :=
  (r.p1.x = r.p2.x ∧ r.p3.x = r.p4.x ∧ r.p1.y = r.p3.y ∧ r.p2.y = r.p4.y) ∨
  (r.p1.x = r.p3.x ∧ r.p2.x = r.p4.x ∧ r.p1.y = r.p2.y ∧ r.p3.y = r.p4.y)

/-- Main theorem: There exists a monochromatic rectangle with integer side lengths -/
theorem monochromatic_rectangle_exists (c : Coloring) : 
  ∃ (r : Rectangle), IsValidRectangle r ∧ 
    c r.p1 = c r.p2 ∧ c r.p2 = c r.p3 ∧ c r.p3 = c r.p4 := by
  sorry

end NUMINAMATH_CALUDE_monochromatic_rectangle_exists_l3539_353923


namespace NUMINAMATH_CALUDE_campaign_fundraising_l3539_353913

-- Define the problem parameters
def max_donation : ℕ := 1200
def max_donors : ℕ := 500
def half_donors_multiplier : ℕ := 3
def donation_percentage : ℚ := 40 / 100

-- Define the total money raised
def total_money_raised : ℚ := 3750000

-- Theorem statement
theorem campaign_fundraising :
  let max_donation_total := max_donation * max_donors
  let half_donation_total := (max_donation / 2) * (max_donors * half_donors_multiplier)
  let total_donations := max_donation_total + half_donation_total
  total_donations = donation_percentage * total_money_raised := by
  sorry


end NUMINAMATH_CALUDE_campaign_fundraising_l3539_353913


namespace NUMINAMATH_CALUDE_sales_tax_difference_l3539_353932

theorem sales_tax_difference (price : ℝ) (tax_rate1 : ℝ) (tax_rate2 : ℝ) : 
  price = 30 → tax_rate1 = 0.0675 → tax_rate2 = 0.055 → 
  price * tax_rate1 - price * tax_rate2 = 0.375 := by
  sorry

#eval (30 * 0.0675 - 30 * 0.055)

end NUMINAMATH_CALUDE_sales_tax_difference_l3539_353932


namespace NUMINAMATH_CALUDE_quadratic_equation_k_value_l3539_353996

/-- Given a quadratic equation with parameter k, prove that k = 1 under specific conditions -/
theorem quadratic_equation_k_value (k : ℝ) : 
  (∃ x1 x2 : ℝ, x1 ≠ x2 ∧ 
    x1^2 + (2*k + 1)*x1 + k^2 + 1 = 0 ∧ 
    x2^2 + (2*k + 1)*x2 + k^2 + 1 = 0 ∧
    x1 + x2 + 2*x1*x2 = 1) →
  k = 1 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_equation_k_value_l3539_353996


namespace NUMINAMATH_CALUDE_parabola_equation_points_with_y_neg_three_l3539_353963

/-- A parabola passing through (1,0) and (0,-3) with axis of symmetry x=2 -/
structure Parabola where
  -- Define the parabola using a function
  f : ℝ → ℝ
  -- The parabola passes through (1,0)
  passes_through_A : f 1 = 0
  -- The parabola passes through (0,-3)
  passes_through_B : f 0 = -3
  -- The axis of symmetry is x=2
  symmetry_axis : ∀ x, f (2 + x) = f (2 - x)

/-- The equation of the parabola is y = -(x-2)^2 + 1 -/
theorem parabola_equation (p : Parabola) : 
  ∀ x, p.f x = -(x - 2)^2 + 1 := by sorry

/-- The points (0,-3) and (4,-3) are the only points on the parabola with y-coordinate -3 -/
theorem points_with_y_neg_three (p : Parabola) :
  ∀ x, p.f x = -3 ↔ x = 0 ∨ x = 4 := by sorry

end NUMINAMATH_CALUDE_parabola_equation_points_with_y_neg_three_l3539_353963


namespace NUMINAMATH_CALUDE_leading_coefficient_of_P_l3539_353981

/-- The polynomial in question -/
def P (x : ℝ) : ℝ := -5 * (x^4 - 2*x^3 + 3*x) + 8 * (x^4 - x^2 + 1) - 3 * (3*x^4 + x^3 + x)

/-- The leading coefficient of a polynomial -/
def leadingCoefficient (p : ℝ → ℝ) : ℝ :=
  sorry -- Definition of leading coefficient

theorem leading_coefficient_of_P :
  leadingCoefficient P = -6 := by
  sorry

end NUMINAMATH_CALUDE_leading_coefficient_of_P_l3539_353981


namespace NUMINAMATH_CALUDE_indefinite_game_l3539_353953

/-- Represents the web structure --/
structure Web where
  rings : ℕ
  radii : ℕ
  rings_ge_two : rings ≥ 2
  radii_ge_three : radii ≥ 3

/-- Represents the game state --/
inductive GameState
  | Ongoing
  | ButterflyWins
  | SpiderWins

/-- Defines the game outcome --/
def gameOutcome (web : Web) : GameState :=
  if web.radii % 2 = 0 ∧ web.rings ≥ web.radii / 2 then
    GameState.Ongoing
  else if web.radii % 2 = 1 ∧ web.rings ≥ (web.radii - 1) / 2 then
    GameState.Ongoing
  else
    GameState.Ongoing -- We use Ongoing as a placeholder, as the actual outcome might depend on the players' strategies

/-- Theorem stating that under certain conditions, the game continues indefinitely --/
theorem indefinite_game (web : Web) :
  (web.radii % 2 = 0 → web.rings ≥ web.radii / 2) ∧
  (web.radii % 2 = 1 → web.rings ≥ (web.radii - 1) / 2) →
  gameOutcome web = GameState.Ongoing :=
by
  sorry

#check indefinite_game

end NUMINAMATH_CALUDE_indefinite_game_l3539_353953


namespace NUMINAMATH_CALUDE_total_sand_needed_l3539_353904

/-- The amount of sand in grams needed to fill one square inch -/
def sand_per_square_inch : ℕ := 3

/-- The length of the rectangular patch in inches -/
def rectangle_length : ℕ := 6

/-- The width of the rectangular patch in inches -/
def rectangle_width : ℕ := 7

/-- The side length of the square patch in inches -/
def square_side : ℕ := 5

/-- Calculates the area of a rectangle given its length and width -/
def rectangle_area (length width : ℕ) : ℕ := length * width

/-- Calculates the area of a square given its side length -/
def square_area (side : ℕ) : ℕ := side * side

/-- Calculates the amount of sand needed for a given area -/
def sand_needed (area : ℕ) : ℕ := area * sand_per_square_inch

/-- Theorem stating the total amount of sand needed for Jason's sand art -/
theorem total_sand_needed :
  sand_needed (rectangle_area rectangle_length rectangle_width) +
  sand_needed (square_area square_side) = 201 := by
  sorry


end NUMINAMATH_CALUDE_total_sand_needed_l3539_353904


namespace NUMINAMATH_CALUDE_relay_race_distance_l3539_353916

theorem relay_race_distance (total_distance : ℕ) (team_members : ℕ) (individual_distance : ℕ) : 
  total_distance = 150 ∧ team_members = 5 ∧ individual_distance * team_members = total_distance →
  individual_distance = 30 := by
  sorry

end NUMINAMATH_CALUDE_relay_race_distance_l3539_353916


namespace NUMINAMATH_CALUDE_sqrt_inequality_l3539_353924

theorem sqrt_inequality (a : ℝ) (h : a ≥ 2) : 
  Real.sqrt (a + 1) - Real.sqrt a < Real.sqrt (a - 1) - Real.sqrt (a - 2) := by
  sorry

end NUMINAMATH_CALUDE_sqrt_inequality_l3539_353924


namespace NUMINAMATH_CALUDE_common_divisors_9240_8820_l3539_353927

theorem common_divisors_9240_8820 : 
  (Finset.filter (fun d => d ∣ 9240 ∧ d ∣ 8820) (Finset.range 9241)).card = 32 := by
  sorry

end NUMINAMATH_CALUDE_common_divisors_9240_8820_l3539_353927


namespace NUMINAMATH_CALUDE_equal_numbers_product_l3539_353983

theorem equal_numbers_product (a b c d : ℝ) : 
  (a + b + c + d) / 4 = 12 ∧ 
  a = 8 ∧ 
  b = 22 ∧ 
  c = d → 
  c * d = 81 := by
sorry

end NUMINAMATH_CALUDE_equal_numbers_product_l3539_353983


namespace NUMINAMATH_CALUDE_stickers_on_bottles_elizabeth_stickers_l3539_353962

/-- Calculate the total number of stickers used on water bottles -/
theorem stickers_on_bottles 
  (initial_bottles : ℕ) 
  (lost_bottles : ℕ) 
  (stolen_bottles : ℕ) 
  (stickers_per_bottle : ℕ) 
  (h1 : initial_bottles ≥ lost_bottles + stolen_bottles) : 
  (initial_bottles - lost_bottles - stolen_bottles) * stickers_per_bottle = 
  (initial_bottles - (lost_bottles + stolen_bottles)) * stickers_per_bottle :=
by sorry

/-- Specific case for Elizabeth's water bottles -/
theorem elizabeth_stickers : 
  (10 : ℕ) - 2 - 1 = 7 ∧ 7 * 3 = 21 :=
by sorry

end NUMINAMATH_CALUDE_stickers_on_bottles_elizabeth_stickers_l3539_353962


namespace NUMINAMATH_CALUDE_sum_always_positive_l3539_353910

theorem sum_always_positive (b : ℝ) (h : b = 2) : 
  (∀ x : ℝ, (3*x^2 - 2*x + b) + (x^2 + b*x - 1) = 4*x^2 + 1) ∧
  (∀ x : ℝ, 4*x^2 + 1 > 0) := by
sorry

end NUMINAMATH_CALUDE_sum_always_positive_l3539_353910


namespace NUMINAMATH_CALUDE_haleys_concert_tickets_l3539_353902

theorem haleys_concert_tickets (ticket_price : ℕ) (extra_tickets : ℕ) (total_spent : ℕ) : 
  ticket_price = 4 → 
  extra_tickets = 5 → 
  total_spent = 32 → 
  ∃ (tickets_for_friends : ℕ), 
    ticket_price * (tickets_for_friends + extra_tickets) = total_spent ∧ 
    tickets_for_friends = 3 :=
by sorry

end NUMINAMATH_CALUDE_haleys_concert_tickets_l3539_353902


namespace NUMINAMATH_CALUDE_system_elimination_l3539_353978

theorem system_elimination (x y : ℝ) : 
  (x + y = 5 ∧ x - y = 2) → 
  (∃ k : ℝ, (x + y) + (x - y) = k ∧ y ≠ k / 2) ∧ 
  (∃ m : ℝ, (x + y) - (x - y) = m ∧ x ≠ m / 2) :=
by sorry

end NUMINAMATH_CALUDE_system_elimination_l3539_353978


namespace NUMINAMATH_CALUDE_weight_problem_l3539_353931

/-- Given three weights a, b, and c, prove that their average weights satisfy the given conditions --/
theorem weight_problem (a b c : ℝ) : 
  (a + b + c) / 3 = 45 →  -- average weight of a, b, and c is 45 kg
  (b + c) / 2 = 44 →      -- average weight of b and c is 44 kg
  b = 33 →                -- weight of b is 33 kg
  (a + b) / 2 = 40        -- average weight of a and b is 40 kg
:= by sorry

end NUMINAMATH_CALUDE_weight_problem_l3539_353931


namespace NUMINAMATH_CALUDE_marble_probability_l3539_353971

theorem marble_probability (total_marbles : ℕ) 
  (prob_both_black : ℚ) (prob_both_white : ℚ) :
  total_marbles = 30 →
  prob_both_black = 4/9 →
  prob_both_white = 4/25 :=
by sorry

end NUMINAMATH_CALUDE_marble_probability_l3539_353971


namespace NUMINAMATH_CALUDE_institutions_made_happy_l3539_353967

theorem institutions_made_happy (people_per_institution : ℕ) (total_people_happy : ℕ) : 
  people_per_institution = 80 → total_people_happy = 480 → 
  total_people_happy / people_per_institution = 6 := by
sorry

end NUMINAMATH_CALUDE_institutions_made_happy_l3539_353967


namespace NUMINAMATH_CALUDE_distance_between_parallel_lines_l3539_353918

/-- The distance between two parallel lines -/
theorem distance_between_parallel_lines 
  (m : ℝ) -- Parameter m in the second line equation
  (h1 : x + 2 * y - 1 = 0) -- First line equation
  (h2 : 2 * x + m * y + 4 = 0) -- Second line equation
  (h_parallel : m = 4) -- Condition for lines to be parallel
  : 
  -- The distance between the lines
  (|(-1) - 2| / Real.sqrt (1 + 4)) = 3 / Real.sqrt 5 :=
by sorry

end NUMINAMATH_CALUDE_distance_between_parallel_lines_l3539_353918


namespace NUMINAMATH_CALUDE_function_properties_l3539_353957

def isPeriodic (f : ℝ → ℝ) (T : ℝ) : Prop :=
  ∀ x, f (x + T) = f x

def isOddOn (f : ℝ → ℝ) (a b : ℝ) : Prop :=
  ∀ x, a ≤ x ∧ x ≤ b → f (-x) = -f x

def isLinearOn (f : ℝ → ℝ) (a b : ℝ) : Prop :=
  ∃ m k, ∀ x, a ≤ x ∧ x ≤ b → f x = m * x + k

def isQuadraticOn (f : ℝ → ℝ) (a b : ℝ) : Prop :=
  ∃ a₀ a₁ a₂, ∀ x, a ≤ x ∧ x ≤ b → f x = a₂ * x^2 + a₁ * x + a₀

theorem function_properties (f : ℝ → ℝ) 
    (h1 : isPeriodic f 5)
    (h2 : isOddOn f (-1) 1)
    (h3 : isLinearOn f 0 1)
    (h4 : isQuadraticOn f 1 4)
    (h5 : ∃ x, x = 2 ∧ f x = -5 ∧ ∀ y, f y ≥ -5) :
  (f 1 + f 4 = 0) ∧
  (∀ x, 1 ≤ x ∧ x ≤ 4 → f x = 5/3 * (x - 2)^2 - 5) ∧
  (∀ x, 4 ≤ x ∧ x ≤ 6 → f x = -10/3 * x + 50/3) ∧
  (∀ x, 6 < x ∧ x ≤ 9 → f x = 5/3 * (x - 7)^2 - 5) := by
  sorry

end NUMINAMATH_CALUDE_function_properties_l3539_353957


namespace NUMINAMATH_CALUDE_complex_number_quadrant_l3539_353954

theorem complex_number_quadrant : 
  let z : ℂ := (2 - Complex.I) ^ 2
  (z.re > 0 ∧ z.im < 0) := by sorry

end NUMINAMATH_CALUDE_complex_number_quadrant_l3539_353954


namespace NUMINAMATH_CALUDE_abs_neg_2023_l3539_353987

theorem abs_neg_2023 : |(-2023 : ℤ)| = 2023 := by
  sorry

end NUMINAMATH_CALUDE_abs_neg_2023_l3539_353987


namespace NUMINAMATH_CALUDE_corn_field_fraction_theorem_l3539_353948

/-- Represents a trapezoid field -/
structure TrapezoidField where
  short_side : ℝ
  long_side : ℝ
  angle : ℝ

/-- The fraction of a trapezoid field's area that is closer to its longest side -/
def fraction_closest_to_longest_side (field : TrapezoidField) : ℝ :=
  sorry

theorem corn_field_fraction_theorem (field : TrapezoidField) 
  (h1 : field.short_side = 120)
  (h2 : field.long_side = 240)
  (h3 : field.angle = 60) :
  fraction_closest_to_longest_side field = 1/3 := by
  sorry

end NUMINAMATH_CALUDE_corn_field_fraction_theorem_l3539_353948


namespace NUMINAMATH_CALUDE_money_distribution_problem_l3539_353936

/-- The number of people in the money distribution problem -/
def num_people : ℕ := 195

/-- The amount of coins the first person receives -/
def first_person_coins : ℕ := 3

/-- The amount of coins each person receives after redistribution -/
def redistribution_coins : ℕ := 100

theorem money_distribution_problem :
  ∃ (n : ℕ), n = num_people ∧
  first_person_coins * n + (n * (n - 1)) / 2 = redistribution_coins * n :=
by sorry

end NUMINAMATH_CALUDE_money_distribution_problem_l3539_353936


namespace NUMINAMATH_CALUDE_both_teasers_count_l3539_353988

/-- The number of brainiacs who like both rebus teasers and math teasers -/
def both_teasers (total : ℕ) (rebus : ℕ) (math : ℕ) (neither : ℕ) (math_only : ℕ) : ℕ :=
  total - rebus - math + (rebus + math - (total - neither))

theorem both_teasers_count :
  both_teasers 100 (2 * 50) 50 4 20 = 18 := by
  sorry

end NUMINAMATH_CALUDE_both_teasers_count_l3539_353988


namespace NUMINAMATH_CALUDE_point_symmetry_and_quadrant_l3539_353949

theorem point_symmetry_and_quadrant (a : ℤ) : 
  (-1 - 2*a > 0) ∧ (2*a - 1 > 0) → a = 0 := by
  sorry

end NUMINAMATH_CALUDE_point_symmetry_and_quadrant_l3539_353949


namespace NUMINAMATH_CALUDE_parabola_and_line_properties_l3539_353903

/-- Represents a parabola in the form y² = -2px --/
structure Parabola where
  p : ℝ

/-- Represents a point in 2D space --/
structure Point where
  x : ℝ
  y : ℝ

/-- Theorem about a specific parabola and line --/
theorem parabola_and_line_properties
  (C : Parabola)
  (A : Point)
  (h1 : A.y^2 = -2 * C.p * A.x) -- A lies on the parabola
  (h2 : A.x = -1 ∧ A.y = -2) -- A is (-1, -2)
  (h3 : ∃ (B : Point), B ≠ A ∧ 
    (B.y - A.y) / (B.x - A.x) = -Real.sqrt 3 ∧ -- Line AB has slope -√3
    B.y^2 = -2 * C.p * B.x) -- B also lies on the parabola
  : 
  (C.p = -2) ∧ -- Equation of parabola is y² = -4x
  (∀ (x y : ℝ), y^2 = -4*x ↔ y^2 = -2 * C.p * x) ∧ -- Equivalent form of parabola equation
  (1 = -C.p/2) ∧ -- Axis of symmetry is x = 1
  (∃ (B : Point), B ≠ A ∧
    (B.y - A.y)^2 + (B.x - A.x)^2 = (16/3)^2) -- Length of AB is 16/3
  := by sorry

end NUMINAMATH_CALUDE_parabola_and_line_properties_l3539_353903


namespace NUMINAMATH_CALUDE_theater_ticket_sales_l3539_353976

/-- Theorem: Theater Ticket Sales --/
theorem theater_ticket_sales
  (orchestra_price : ℕ)
  (balcony_price : ℕ)
  (total_revenue : ℕ)
  (balcony_excess : ℕ)
  (h1 : orchestra_price = 12)
  (h2 : balcony_price = 8)
  (h3 : total_revenue = 3320)
  (h4 : balcony_excess = 190)
  : ∃ (orchestra_tickets balcony_tickets : ℕ),
    orchestra_tickets * orchestra_price + balcony_tickets * balcony_price = total_revenue ∧
    balcony_tickets = orchestra_tickets + balcony_excess ∧
    orchestra_tickets + balcony_tickets = 370 :=
by
  sorry


end NUMINAMATH_CALUDE_theater_ticket_sales_l3539_353976


namespace NUMINAMATH_CALUDE_quadratic_root_implies_m_value_l3539_353950

theorem quadratic_root_implies_m_value (m : ℝ) : 
  (∃ x : ℝ, x^2 + (m+2)*x - 2 = 0 ∧ x = 1) → m = -1 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_root_implies_m_value_l3539_353950


namespace NUMINAMATH_CALUDE_alex_bike_trip_l3539_353909

/-- Alex's cross-country bike trip problem -/
theorem alex_bike_trip (total_distance : ℝ) (flat_speed : ℝ) (uphill_speed : ℝ) (uphill_time : ℝ)
                       (downhill_speed : ℝ) (downhill_time : ℝ) (walking_distance : ℝ) :
  total_distance = 164 →
  flat_speed = 20 →
  uphill_speed = 12 →
  uphill_time = 2.5 →
  downhill_speed = 24 →
  downhill_time = 1.5 →
  walking_distance = 8 →
  ∃ (flat_time : ℝ), 
    flat_time = 4.5 ∧ 
    total_distance = flat_speed * flat_time + uphill_speed * uphill_time + 
                     downhill_speed * downhill_time + walking_distance :=
by sorry


end NUMINAMATH_CALUDE_alex_bike_trip_l3539_353909


namespace NUMINAMATH_CALUDE_hyperbola_eccentricity_sqrt_5_l3539_353982

/-- Represents a hyperbola with semi-major axis a and semi-minor axis b -/
structure Hyperbola (a b : ℝ) where
  a_pos : a > 0
  b_pos : b > 0

/-- The eccentricity of a hyperbola -/
def eccentricity (h : Hyperbola a b) : ℝ := sorry

/-- The left focus of the hyperbola -/
def left_focus (h : Hyperbola a b) : ℝ × ℝ := sorry

theorem hyperbola_eccentricity_sqrt_5 
  {a b c : ℝ} (h : Hyperbola a b) 
  (focus_c : left_focus h = (-c, 0))
  (point_P : c^2/a^2 - 4 = 1) : 
  eccentricity h = Real.sqrt 5 := by sorry

end NUMINAMATH_CALUDE_hyperbola_eccentricity_sqrt_5_l3539_353982


namespace NUMINAMATH_CALUDE_fruit_arrangement_count_l3539_353960

def num_apples : ℕ := 4
def num_oranges : ℕ := 2
def num_bananas : ℕ := 2
def num_pears : ℕ := 1

def total_fruits : ℕ := num_apples + num_oranges + num_bananas + num_pears

theorem fruit_arrangement_count :
  (total_fruits.factorial) / (num_apples.factorial * num_oranges.factorial * num_bananas.factorial * num_pears.factorial) = 3780 := by
  sorry

end NUMINAMATH_CALUDE_fruit_arrangement_count_l3539_353960


namespace NUMINAMATH_CALUDE_exists_injection_with_property_l3539_353969

-- Define the set A as a finite type
variable {A : Type} [Finite A]

-- Define the set S as a predicate on triples of elements from A
variable (S : A → A → A → Prop)

-- State the conditions on S
variable (h1 : ∀ a b c : A, S a b c ↔ S b c a)
variable (h2 : ∀ a b c : A, S a b c ↔ ¬S c b a)
variable (h3 : ∀ a b c d : A, (S a b c ∧ S c d a) ↔ (S b c d ∧ S d a b))

-- State the theorem
theorem exists_injection_with_property :
  ∃ g : A → ℝ, Function.Injective g ∧
    ∀ a b c : A, g a < g b ∧ g b < g c → S a b c :=
sorry

end NUMINAMATH_CALUDE_exists_injection_with_property_l3539_353969


namespace NUMINAMATH_CALUDE_complex_equidistant_points_l3539_353973

theorem complex_equidistant_points (z : ℂ) : 
  (Complex.abs (z - 2) = Complex.abs (z + 4) ∧ 
   Complex.abs (z - 2) = Complex.abs (z + 2*I)) ↔ 
  z = -1 + I :=
by sorry

end NUMINAMATH_CALUDE_complex_equidistant_points_l3539_353973


namespace NUMINAMATH_CALUDE_product_sequence_mod_six_l3539_353945

theorem product_sequence_mod_six : ∃ (seq : List Nat), 
  (seq.length = 10) ∧ 
  (∀ i, i ∈ seq → ∃ k, i = 10 * k + 3) ∧
  (seq.prod % 6 = 3) := by
sorry

end NUMINAMATH_CALUDE_product_sequence_mod_six_l3539_353945


namespace NUMINAMATH_CALUDE_non_zero_digits_count_l3539_353934

/-- The fraction we're working with -/
def f : ℚ := 120 / (2^5 * 5^9)

/-- Count of non-zero digits after the decimal point in the decimal representation of a rational number -/
noncomputable def count_non_zero_digits_after_decimal (q : ℚ) : ℕ := sorry

/-- The main theorem: the count of non-zero digits after the decimal point for our fraction is 2 -/
theorem non_zero_digits_count : count_non_zero_digits_after_decimal f = 2 := by sorry

end NUMINAMATH_CALUDE_non_zero_digits_count_l3539_353934


namespace NUMINAMATH_CALUDE_f_le_g_l3539_353941

def f (n : ℕ+) : ℚ :=
  (Finset.range n).sum (fun i => 1 / ((i + 1) : ℚ) ^ 2) + 1

def g (n : ℕ+) : ℚ :=
  1/2 * (3 - 1 / (n : ℚ) ^ 2)

theorem f_le_g : ∀ n : ℕ+, f n ≤ g n := by
  sorry

end NUMINAMATH_CALUDE_f_le_g_l3539_353941


namespace NUMINAMATH_CALUDE_line_equations_l3539_353901

/-- Given point M -/
def M : ℝ × ℝ := (-1, 2)

/-- Given line equation -/
def L : ℝ → ℝ → ℝ := λ x y ↦ 2*x + y + 5

/-- Parallel line -/
def L_parallel : ℝ → ℝ → ℝ := λ x y ↦ 2*x + y

/-- Perpendicular line -/
def L_perpendicular : ℝ → ℝ → ℝ := λ x y ↦ x - 2*y + 5

theorem line_equations :
  (L_parallel M.1 M.2 = 0 ∧ 
   ∀ (x y : ℝ), L_parallel x y = 0 → L x y = L_parallel x y + 5) ∧
  (L_perpendicular M.1 M.2 = 0 ∧ 
   ∀ (x y : ℝ), L x y = 0 → L_perpendicular x y = 0 → x = y) := by
  sorry

end NUMINAMATH_CALUDE_line_equations_l3539_353901


namespace NUMINAMATH_CALUDE_expression_values_l3539_353998

theorem expression_values (a b c d : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0) (hd : d ≠ 0) :
  let expr := a / |a| + b / |b| + c / |c| + d / |d| + (a * b * c * d) / |a * b * c * d|
  expr = 5 ∨ expr = 1 ∨ expr = -3 := by
  sorry

end NUMINAMATH_CALUDE_expression_values_l3539_353998


namespace NUMINAMATH_CALUDE_J_specific_value_l3539_353995

/-- Definition of J function -/
def J (a b c : ℚ) : ℚ := a / b + b / c + c / a

/-- Theorem: J(3, 3/4, 4) equals 259/48 -/
theorem J_specific_value : J 3 (3/4) 4 = 259/48 := by
  sorry

/-- Lemma: Relationship between a, b, and c -/
lemma abc_relationship (a b c k : ℚ) (hk : k ≠ 0) : 
  b = a / k ∧ c = k * b → J a b c = J a (a / k) (k * (a / k)) := by
  sorry

end NUMINAMATH_CALUDE_J_specific_value_l3539_353995


namespace NUMINAMATH_CALUDE_difference_of_squares_l3539_353928

theorem difference_of_squares : 255^2 - 745^2 = -490000 := by
  sorry

end NUMINAMATH_CALUDE_difference_of_squares_l3539_353928


namespace NUMINAMATH_CALUDE_monogramming_cost_is_17_69_l3539_353972

/-- Calculates the monogramming cost per stocking --/
def monogramming_cost_per_stocking (grandchildren children : ℕ) 
  (stockings_per_grandchild : ℕ) (stocking_price : ℚ) (discount_percent : ℚ) 
  (total_cost : ℚ) : ℚ :=
  let total_stockings := grandchildren * stockings_per_grandchild + children
  let discounted_price := stocking_price * (1 - discount_percent / 100)
  let stockings_cost := total_stockings * discounted_price
  let total_monogramming_cost := total_cost - stockings_cost
  total_monogramming_cost / total_stockings

/-- Theorem stating that the monogramming cost per stocking is $17.69 --/
theorem monogramming_cost_is_17_69 :
  monogramming_cost_per_stocking 5 4 5 20 10 1035 = 1769 / 100 := by
  sorry

end NUMINAMATH_CALUDE_monogramming_cost_is_17_69_l3539_353972


namespace NUMINAMATH_CALUDE_subtracted_number_l3539_353968

theorem subtracted_number (x : ℕ) (some_number : ℕ) 
  (h1 : x = 88320) 
  (h2 : x + 1315 + 9211 - some_number = 11901) : 
  some_number = 86945 := by
  sorry

end NUMINAMATH_CALUDE_subtracted_number_l3539_353968


namespace NUMINAMATH_CALUDE_fraction_addition_l3539_353921

theorem fraction_addition (d : ℝ) : (3 + 4 * d) / 5 + 3 = (18 + 4 * d) / 5 := by
  sorry

end NUMINAMATH_CALUDE_fraction_addition_l3539_353921


namespace NUMINAMATH_CALUDE_number_multiplication_theorem_l3539_353955

theorem number_multiplication_theorem :
  ∃ x : ℝ, x * 9999 = 824777405 ∧ x = 82482.5 := by
  sorry

end NUMINAMATH_CALUDE_number_multiplication_theorem_l3539_353955


namespace NUMINAMATH_CALUDE_inequality_proof_l3539_353994

theorem inequality_proof (a b c : ℝ) 
  (h_pos_a : 0 < a) (h_pos_b : 0 < b) (h_pos_c : 0 < c) 
  (h_abc : a * b * c = 1) : 
  (a * b / (a^5 + a * b + b^5)) + 
  (b * c / (b^5 + b * c + c^5)) + 
  (c * a / (c^5 + c * a + a^5)) ≤ 1 := by
sorry

end NUMINAMATH_CALUDE_inequality_proof_l3539_353994


namespace NUMINAMATH_CALUDE_perpendicular_vectors_x_value_l3539_353990

/-- Given two vectors a and b in R², where a = (-2, 2) and b = (x, -3),
    if a is perpendicular to b, then x = -3. -/
theorem perpendicular_vectors_x_value :
  let a : Fin 2 → ℝ := ![-2, 2]
  let b : Fin 2 → ℝ := ![x, -3]
  (∀ (i : Fin 2), a i * b i = 0) → x = -3 := by
  sorry

end NUMINAMATH_CALUDE_perpendicular_vectors_x_value_l3539_353990


namespace NUMINAMATH_CALUDE_remainder_of_large_number_l3539_353970

theorem remainder_of_large_number (n : ℕ) (d : ℕ) (h : n = 123456789012 ∧ d = 126) :
  n % d = 18 := by
  sorry

end NUMINAMATH_CALUDE_remainder_of_large_number_l3539_353970


namespace NUMINAMATH_CALUDE_trees_on_path_l3539_353956

theorem trees_on_path (path_length : ℕ) (tree_spacing : ℕ) (h1 : path_length = 80) (h2 : tree_spacing = 4) :
  let trees_on_one_side := path_length / tree_spacing + 1
  2 * trees_on_one_side = 42 :=
by sorry

end NUMINAMATH_CALUDE_trees_on_path_l3539_353956


namespace NUMINAMATH_CALUDE_sin_5pi_6_minus_2alpha_l3539_353958

theorem sin_5pi_6_minus_2alpha (α : ℝ) 
  (h : Real.cos (π / 6 - α) = Real.sqrt 3 / 3) : 
  Real.sin (5 * π / 6 - 2 * α) = -1 / 3 := by
  sorry

end NUMINAMATH_CALUDE_sin_5pi_6_minus_2alpha_l3539_353958


namespace NUMINAMATH_CALUDE_lucy_earnings_l3539_353961

/-- Calculates the earnings for a single 6-hour cycle -/
def cycle_earnings : ℕ := 1 + 2 + 3 + 4 + 5 + 6

/-- Calculates the earnings for the remaining hours after complete cycles -/
def remaining_earnings (hours : ℕ) : ℕ :=
  match hours with
  | 0 => 0
  | 1 => 1
  | 2 => 1 + 2
  | _ => 1 + 2 + 3

/-- Calculates the total earnings for a given number of hours -/
def total_earnings (hours : ℕ) : ℕ :=
  let complete_cycles := hours / 6
  let remaining_hours := hours % 6
  complete_cycles * cycle_earnings + remaining_earnings remaining_hours

/-- The theorem stating that Lucy's earnings for 45 hours of work is $153 -/
theorem lucy_earnings : total_earnings 45 = 153 := by
  sorry

end NUMINAMATH_CALUDE_lucy_earnings_l3539_353961


namespace NUMINAMATH_CALUDE_quadratic_equations_solutions_l3539_353947

theorem quadratic_equations_solutions : ∃ (s1 s2 : Set ℝ),
  (∀ x : ℝ, x ∈ s1 ↔ 3 * x^2 = 6 * x) ∧
  (∀ x : ℝ, x ∈ s2 ↔ x^2 - 6 * x + 5 = 0) ∧
  s1 = {0, 2} ∧
  s2 = {5, 1} := by
sorry


end NUMINAMATH_CALUDE_quadratic_equations_solutions_l3539_353947


namespace NUMINAMATH_CALUDE_logarithm_bijection_l3539_353917

-- Define the set of positive real numbers
def PositiveReals := {x : ℝ | x > 0}

-- State the theorem
theorem logarithm_bijection (a : ℝ) (ha : a > 1) :
  ∃ f : PositiveReals → ℝ, Function.Bijective f :=
sorry

end NUMINAMATH_CALUDE_logarithm_bijection_l3539_353917


namespace NUMINAMATH_CALUDE_sunglasses_cap_probability_l3539_353992

theorem sunglasses_cap_probability (total_sunglasses : ℕ) (total_caps : ℕ) 
  (prob_cap_given_sunglasses : ℚ) :
  total_sunglasses = 80 →
  total_caps = 60 →
  prob_cap_given_sunglasses = 3/8 →
  (prob_cap_given_sunglasses * total_sunglasses : ℚ) / total_caps = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_sunglasses_cap_probability_l3539_353992


namespace NUMINAMATH_CALUDE_complex_product_magnitude_l3539_353951

theorem complex_product_magnitude : Complex.abs (3 - 5*Complex.I) * Complex.abs (3 + 5*Complex.I) = 34 := by
  sorry

end NUMINAMATH_CALUDE_complex_product_magnitude_l3539_353951


namespace NUMINAMATH_CALUDE_stewart_farm_horse_food_l3539_353979

/-- Represents a farm with sheep and horses -/
structure Farm where
  sheep : ℕ
  horses : ℕ
  total_horse_food : ℕ

/-- Calculates the amount of food each horse needs per day -/
def horse_food_per_day (f : Farm) : ℚ :=
  f.total_horse_food / f.horses

/-- The Stewart farm satisfies the given conditions -/
def stewart_farm : Farm :=
  { sheep := 24,
    horses := 56,
    total_horse_food := 12880 }

theorem stewart_farm_horse_food :
  horse_food_per_day stewart_farm = 230 :=
sorry

end NUMINAMATH_CALUDE_stewart_farm_horse_food_l3539_353979


namespace NUMINAMATH_CALUDE_jogger_train_distance_l3539_353965

/-- Proves that a jogger is 200 meters ahead of a train's engine given specific conditions --/
theorem jogger_train_distance (jogger_speed train_speed : ℝ) (train_length : ℝ) (passing_time : ℝ) :
  jogger_speed = 9 * (5 / 18) →
  train_speed = 45 * (5 / 18) →
  train_length = 200 →
  passing_time = 40 →
  (train_speed - jogger_speed) * passing_time = train_length + 200 :=
by
  sorry

#check jogger_train_distance

end NUMINAMATH_CALUDE_jogger_train_distance_l3539_353965


namespace NUMINAMATH_CALUDE_S_five_three_l3539_353989

-- Define the operation ∘
def S (a b : ℕ) : ℕ := 4 * a + 3 * b

-- Theorem statement
theorem S_five_three : S 5 3 = 29 := by
  sorry

end NUMINAMATH_CALUDE_S_five_three_l3539_353989


namespace NUMINAMATH_CALUDE_proof_by_contradiction_assumption_l3539_353952

theorem proof_by_contradiction_assumption (a b : ℕ) (h : 5 ∣ (a * b)) :
  (¬ (5 ∣ a) ∧ ¬ (5 ∣ b)) ↔ 
  ¬ (5 ∣ a ∨ 5 ∣ b) :=
by sorry

end NUMINAMATH_CALUDE_proof_by_contradiction_assumption_l3539_353952


namespace NUMINAMATH_CALUDE_correct_taobao_shopping_order_l3539_353986

-- Define the type for shopping steps
inductive ShoppingStep
| select_products
| buy_and_pay
| transfer_payment
| receive_and_confirm
| ship_goods

-- Define the shopping process
def shopping_process : List ShoppingStep :=
  [ShoppingStep.select_products, ShoppingStep.buy_and_pay, ShoppingStep.ship_goods, 
   ShoppingStep.receive_and_confirm, ShoppingStep.transfer_payment]

-- Define a function to check if the order is correct
def is_correct_order (order : List ShoppingStep) : Prop :=
  order = shopping_process

-- Theorem stating the correct order
theorem correct_taobao_shopping_order :
  is_correct_order [ShoppingStep.select_products, ShoppingStep.buy_and_pay, 
                    ShoppingStep.ship_goods, ShoppingStep.receive_and_confirm, 
                    ShoppingStep.transfer_payment] :=
by
  sorry

#check correct_taobao_shopping_order

end NUMINAMATH_CALUDE_correct_taobao_shopping_order_l3539_353986


namespace NUMINAMATH_CALUDE_different_gender_selection_l3539_353908

theorem different_gender_selection (total_members : ℕ) (boys : ℕ) (girls : ℕ) 
  (h1 : total_members = 24)
  (h2 : boys = 12)
  (h3 : girls = 12)
  (h4 : total_members = boys + girls) :
  (boys * girls) + (girls * boys) = 288 := by
sorry

end NUMINAMATH_CALUDE_different_gender_selection_l3539_353908


namespace NUMINAMATH_CALUDE_average_of_remaining_numbers_l3539_353942

theorem average_of_remaining_numbers
  (total : ℝ)
  (avg_all : ℝ)
  (avg_group1 : ℝ)
  (avg_group2 : ℝ)
  (h1 : total = 6 * avg_all)
  (h2 : avg_all = 3.95)
  (h3 : avg_group1 = 3.6)
  (h4 : avg_group2 = 3.85) :
  (total - 2 * avg_group1 - 2 * avg_group2) / 2 = 4.4 := by
sorry

end NUMINAMATH_CALUDE_average_of_remaining_numbers_l3539_353942


namespace NUMINAMATH_CALUDE_cube_and_sphere_volume_l3539_353907

theorem cube_and_sphere_volume (cube_volume : Real) (sphere_volume : Real) : 
  cube_volume = 8 → sphere_volume = 4 * Real.sqrt 3 * Real.pi := by
  sorry

end NUMINAMATH_CALUDE_cube_and_sphere_volume_l3539_353907


namespace NUMINAMATH_CALUDE_pencil_distribution_l3539_353975

theorem pencil_distribution (total_pencils : ℕ) (num_friends : ℕ) (pencils_per_friend : ℕ) :
  total_pencils = 24 →
  num_friends = 3 →
  total_pencils = num_friends * pencils_per_friend →
  pencils_per_friend = 8 := by
sorry

end NUMINAMATH_CALUDE_pencil_distribution_l3539_353975


namespace NUMINAMATH_CALUDE_f_2004_value_l3539_353943

/-- A function with the property that f(a) + f(b) = n^3 when a + b = 2^(n+1) -/
def special_function (f : ℕ → ℕ) : Prop :=
  ∀ (a b n : ℕ), a > 0 → b > 0 → n > 0 → a + b = 2^(n+1) → f a + f b = n^3

theorem f_2004_value (f : ℕ → ℕ) (h : special_function f) : f 2004 = 1320 := by
  sorry

end NUMINAMATH_CALUDE_f_2004_value_l3539_353943


namespace NUMINAMATH_CALUDE_mass_of_impurities_l3539_353900

/-- Given a sample of natural sulfur, prove that the mass of impurities
    is equal to the difference between the total mass and the mass of pure sulfur. -/
theorem mass_of_impurities (total_mass pure_sulfur_mass : ℝ) :
  total_mass ≥ pure_sulfur_mass →
  total_mass - pure_sulfur_mass = total_mass - pure_sulfur_mass :=
by sorry

end NUMINAMATH_CALUDE_mass_of_impurities_l3539_353900


namespace NUMINAMATH_CALUDE_cistern_filling_time_l3539_353944

/-- Given a cistern with two taps, one that can fill it in 5 hours and another that can empty it in 6 hours,
    calculate the time it takes to fill the cistern when both taps are opened simultaneously. -/
theorem cistern_filling_time (fill_time empty_time : ℝ) (h_fill : fill_time = 5) (h_empty : empty_time = 6) :
  (fill_time * empty_time) / (empty_time - fill_time) = 30 := by
  sorry

#check cistern_filling_time

end NUMINAMATH_CALUDE_cistern_filling_time_l3539_353944


namespace NUMINAMATH_CALUDE_biased_coin_probability_l3539_353930

def binomial (n k : ℕ) : ℕ := (Nat.choose n k)

theorem biased_coin_probability : ∃ (h : ℚ), 
  (0 < h ∧ h < 1) ∧ 
  (binomial 6 2 : ℚ) * h^2 * (1-h)^4 = (binomial 6 3 : ℚ) * h^3 * (1-h)^3 → 
  (binomial 6 4 : ℚ) * h^4 * (1-h)^2 = 19440 / 117649 :=
sorry

end NUMINAMATH_CALUDE_biased_coin_probability_l3539_353930


namespace NUMINAMATH_CALUDE_range_of_f_l3539_353999

-- Define the function f
def f (x : ℝ) : ℝ := 4 * x^3 - 3 * x - 2

-- State the theorem
theorem range_of_f :
  ∀ x ∈ Set.Icc 0 (Real.sqrt 3 / 2),
  ∃ y ∈ Set.Icc (-3) (-2),
  f x = y ∧
  ∀ z, f x = z → z ∈ Set.Icc (-3) (-2) :=
by sorry

-- Define the trigonometric identity
axiom cos_triple_angle (θ : ℝ) :
  Real.cos (3 * θ) = 4 * (Real.cos θ)^3 - 3 * (Real.cos θ)

end NUMINAMATH_CALUDE_range_of_f_l3539_353999

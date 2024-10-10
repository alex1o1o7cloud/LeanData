import Mathlib

namespace fraction_equality_l1032_103248

theorem fraction_equality (a b : ℝ) (h : (2*a - b) / (a + b) = 3/4) : b / a = 5/7 := by
  sorry

end fraction_equality_l1032_103248


namespace x_intercept_of_line_l1032_103253

/-- The x-intercept of the line 4x + 6y = 24 is (6, 0) -/
theorem x_intercept_of_line (x y : ℝ) : 
  4 * x + 6 * y = 24 → y = 0 → x = 6 := by sorry

end x_intercept_of_line_l1032_103253


namespace book_purchase_equations_l1032_103271

/-- Represents the problem of students pooling money to buy a book. -/
theorem book_purchase_equations (x y : ℝ) :
  (∀ (excess shortage : ℝ),
    excess = 4 ∧ shortage = 3 →
    (9 * x - y = excess ∧ y - 8 * x = shortage)) ↔
  (9 * x - y = 4 ∧ y - 8 * x = 3) :=
sorry

end book_purchase_equations_l1032_103271


namespace proportional_set_l1032_103276

/-- A set of four positive real numbers is proportional if the product of the outer terms equals the product of the inner terms. -/
def is_proportional (a b c d : ℝ) : Prop :=
  a > 0 ∧ b > 0 ∧ c > 0 ∧ d > 0 ∧ a * d = b * c

/-- The set {3, 6, 9, 18} is proportional. -/
theorem proportional_set : is_proportional 3 6 9 18 := by
  sorry

end proportional_set_l1032_103276


namespace binary_1100_is_12_l1032_103289

def binary_to_decimal (b : List Bool) : ℕ :=
  b.enum.foldl (fun acc (i, bit) => acc + (if bit then 2^i else 0)) 0

theorem binary_1100_is_12 : 
  binary_to_decimal [false, false, true, true] = 12 := by
  sorry

end binary_1100_is_12_l1032_103289


namespace array_sum_divisibility_l1032_103205

/-- Represents the sum of all terms in a 1/2011-array -/
def arraySum : ℚ :=
  (2011^2 : ℚ) / ((4011 : ℚ) * 2010)

/-- Numerator of the array sum when expressed as a simplified fraction -/
def m : ℕ := 2011^2

/-- Denominator of the array sum when expressed as a simplified fraction -/
def n : ℕ := 4011 * 2010

/-- Theorem stating that m + n is divisible by 2011 -/
theorem array_sum_divisibility : (m + n) % 2011 = 0 := by
  sorry

end array_sum_divisibility_l1032_103205


namespace officer_average_salary_l1032_103244

theorem officer_average_salary
  (total_avg : ℝ)
  (non_officer_avg : ℝ)
  (officer_count : ℕ)
  (non_officer_count : ℕ)
  (h1 : total_avg = 120)
  (h2 : non_officer_avg = 110)
  (h3 : officer_count = 15)
  (h4 : non_officer_count = 525) :
  let total_count := officer_count + non_officer_count
  let officer_total := total_avg * total_count - non_officer_avg * non_officer_count
  officer_total / officer_count = 470 := by
sorry

end officer_average_salary_l1032_103244


namespace leifs_apples_l1032_103299

def num_oranges : ℕ := 24 -- 2 dozen oranges

theorem leifs_apples :
  ∃ (num_apples : ℕ), num_apples = num_oranges - 10 ∧ num_apples = 14 :=
by sorry

end leifs_apples_l1032_103299


namespace dihedral_angle_relationship_not_determined_l1032_103246

/-- Two dihedral angles with perpendicular half-planes -/
structure PerpendicularDihedralAngles where
  angle1 : ℝ
  angle2 : ℝ
  perpendicular_half_planes : Bool

/-- The relationship between the sizes of two dihedral angles with perpendicular half-planes is not determined -/
theorem dihedral_angle_relationship_not_determined (angles : PerpendicularDihedralAngles) :
  angles.perpendicular_half_planes →
  ¬(∀ a : PerpendicularDihedralAngles, a.angle1 = a.angle2) ∧
  ¬(∀ a : PerpendicularDihedralAngles, a.angle1 + a.angle2 = π) ∧
  ¬(∀ a : PerpendicularDihedralAngles, a.angle1 = a.angle2 ∨ a.angle1 + a.angle2 = π) :=
by sorry

end dihedral_angle_relationship_not_determined_l1032_103246


namespace common_difference_is_two_l1032_103203

/-- An arithmetic sequence with given properties -/
structure ArithmeticSequence where
  a : ℕ → ℝ
  is_arithmetic : ∀ n, a (n + 1) - a n = a (n + 2) - a (n + 1)
  sum_condition : a 2 + a 6 = 8
  fifth_term : a 5 = 6

/-- The common difference of an arithmetic sequence -/
def common_difference (seq : ArithmeticSequence) : ℝ :=
  seq.a 2 - seq.a 1

theorem common_difference_is_two (seq : ArithmeticSequence) :
  common_difference seq = 2 := by
  sorry

end common_difference_is_two_l1032_103203


namespace overlap_area_of_rectangles_l1032_103221

theorem overlap_area_of_rectangles (a b x y : ℝ) : 
  a = 3 ∧ b = 9 ∧  -- Rectangle dimensions
  x^2 + a^2 = y^2 ∧ -- Pythagorean theorem for the corner triangle
  x + y = b ∧ -- Sum of triangle sides equals longer rectangle side
  0 < x ∧ 0 < y -- Positive lengths
  → (b * a - 2 * (x * a / 2)) = 15 := by sorry

end overlap_area_of_rectangles_l1032_103221


namespace inscribed_square_side_length_l1032_103279

/-- A right triangle with sides 5, 12, and 13 containing an inscribed square -/
structure InscribedSquare where
  /-- Side length of the inscribed square -/
  t : ℝ
  /-- The inscribed square has side length t -/
  is_square : t > 0
  /-- The triangle is a right triangle with sides 5, 12, and 13 -/
  is_right_triangle : 5^2 + 12^2 = 13^2
  /-- The square is inscribed in the triangle -/
  is_inscribed : ∃ (x y : ℝ), x > 0 ∧ y > 0 ∧ x + y = 13 ∧ t / x = 5 / 13 ∧ t / y = 12 / 13

/-- The side length of the inscribed square is 780/169 -/
theorem inscribed_square_side_length (s : InscribedSquare) : s.t = 780 / 169 := by
  sorry

end inscribed_square_side_length_l1032_103279


namespace expected_sides_formula_rectangle_limit_sides_l1032_103251

/-- Represents a polygon with a given number of sides -/
structure Polygon where
  sides : ℕ

/-- Represents the state after cutting a polygon -/
structure CutState where
  initialPolygon : Polygon
  numCuts : ℕ

/-- Calculates the expected number of sides after cuts -/
def expectedSides (state : CutState) : ℚ :=
  (state.initialPolygon.sides + 4 * state.numCuts) / (state.numCuts + 1)

/-- Theorem: The expected number of sides after cuts is (n + 4k) / (k + 1) -/
theorem expected_sides_formula (state : CutState) :
  expectedSides state = (state.initialPolygon.sides + 4 * state.numCuts) / (state.numCuts + 1) := by
  sorry

/-- Corollary: For a rectangle (4 sides), as cuts approach infinity, expected sides approach 4 -/
theorem rectangle_limit_sides (initialRect : Polygon) (h : initialRect.sides = 4) :
  ∀ ε > 0, ∃ N, ∀ k ≥ N,
    |expectedSides { initialPolygon := initialRect, numCuts := k } - 4| < ε := by
  sorry

end expected_sides_formula_rectangle_limit_sides_l1032_103251


namespace complex_modulus_squared_l1032_103294

theorem complex_modulus_squared : Complex.abs (3/4 + 3*Complex.I)^2 = 153/16 := by
  sorry

end complex_modulus_squared_l1032_103294


namespace ratio_problem_l1032_103278

theorem ratio_problem (a b c d : ℝ) 
  (h1 : a / b = 5)
  (h2 : b / c = 1 / 4)
  (h3 : c / d = 7) :
  d / a = 4 / 35 := by
sorry

end ratio_problem_l1032_103278


namespace percent_relation_l1032_103224

theorem percent_relation (a b c : ℝ) (h1 : c = 0.25 * a) (h2 : b = 2.5 * a) :
  c = 0.1 * b := by sorry

end percent_relation_l1032_103224


namespace percentage_men_is_seventy_l1032_103222

/-- The number of women in the engineering department -/
def num_women : ℕ := 180

/-- The number of men in the engineering department -/
def num_men : ℕ := 420

/-- The total number of students in the engineering department -/
def total_students : ℕ := num_women + num_men

/-- The percentage of men in the engineering department -/
def percentage_men : ℚ := (num_men : ℚ) / (total_students : ℚ) * 100

theorem percentage_men_is_seventy :
  percentage_men = 70 :=
sorry

end percentage_men_is_seventy_l1032_103222


namespace max_a_right_angle_circle_l1032_103273

/-- Given points A(-a, 0) and B(a, 0) where a > 0, and a point C on the circle (x-2)²+(y-2)²=2
    such that ∠ACB = 90°, the maximum value of a is 3√2. -/
theorem max_a_right_angle_circle (a : ℝ) (C : ℝ × ℝ) : 
  a > 0 → 
  (C.1 - 2)^2 + (C.2 - 2)^2 = 2 →
  (C.1 + a) * (C.1 - a) + C.2 * C.2 = 0 →
  a ≤ 3 * Real.sqrt 2 :=
sorry

end max_a_right_angle_circle_l1032_103273


namespace f_inequality_solution_l1032_103223

-- Define the function f
noncomputable def f (x : ℝ) : ℝ :=
  if x > 0 then -Real.log x - x else -Real.log (-x) + x

-- Define the solution set
def solution_set : Set ℝ := {m : ℝ | m ∈ Set.Ioo (-1/2) 0 ∪ Set.Ioo 0 (1/2)}

-- State the theorem
theorem f_inequality_solution :
  ∀ m : ℝ, m ≠ 0 → (f (1/m) < Real.log (1/2) - 2 ↔ m ∈ solution_set) :=
sorry

end f_inequality_solution_l1032_103223


namespace number_puzzle_l1032_103237

theorem number_puzzle (x : ℝ) : x / 3 = x - 42 → x = 63 := by
  sorry

end number_puzzle_l1032_103237


namespace school_meeting_attendance_l1032_103270

theorem school_meeting_attendance
  (seated_students : ℕ)
  (seated_teachers : ℕ)
  (standing_students : ℕ)
  (h1 : seated_students = 300)
  (h2 : seated_teachers = 30)
  (h3 : standing_students = 25) :
  seated_students + seated_teachers + standing_students = 355 :=
by sorry

end school_meeting_attendance_l1032_103270


namespace geometric_sequence_sum_property_l1032_103259

/-- A geometric sequence with its sum function -/
structure GeometricSequence where
  a : ℕ → ℝ  -- The sequence
  S : ℕ → ℝ  -- Sum function
  is_geometric : ∀ n : ℕ, n > 0 → a (n + 1) = a n * (a 2 / a 1)
  sum_property : ∀ n : ℕ, n > 0 → S n = (a 1) * (1 - (a 2 / a 1)^n) / (1 - (a 2 / a 1))

/-- The main theorem -/
theorem geometric_sequence_sum_property 
  (seq : GeometricSequence) 
  (h1 : seq.S 3 = 8) 
  (h2 : seq.S 6 = 7) : 
  seq.a 7 + seq.a 8 + seq.a 9 = 1/8 := by
sorry

end geometric_sequence_sum_property_l1032_103259


namespace smallest_m_partition_property_l1032_103240

def S (m : ℕ) : Set ℕ := {n : ℕ | 2 ≤ n ∧ n ≤ m}

def satisfies_condition (A : Set ℕ) : Prop :=
  ∃ a b : ℕ, a ∈ A ∧ b ∈ A ∧ a^b = b^a

theorem smallest_m_partition_property :
  ∀ m : ℕ, m ≥ 2 →
    (∀ A B : Set ℕ, A ∪ B = S m ∧ A ∩ B = ∅ →
      satisfies_condition A ∨ satisfies_condition B) ↔ m ≥ 16 :=
sorry

end smallest_m_partition_property_l1032_103240


namespace quadrilateral_is_trapezoid_or_parallelogram_l1032_103296

/-- A quadrilateral with angles A, B, C, and D, where the products of cosines of opposite angles are equal. -/
structure Quadrilateral where
  A : Real
  B : Real
  C : Real
  D : Real
  angle_sum : A + B + C + D = 2 * Real.pi
  cosine_product : Real.cos A * Real.cos C = Real.cos B * Real.cos D

/-- A quadrilateral is either a trapezoid or a parallelogram if the products of cosines of opposite angles are equal. -/
theorem quadrilateral_is_trapezoid_or_parallelogram (q : Quadrilateral) :
  (∃ (x y : Real), x + y = Real.pi ∧ (q.A = x ∧ q.C = x) ∨ (q.B = y ∧ q.D = y)) ∨
  (q.A = q.C ∧ q.B = q.D) := by
  sorry

end quadrilateral_is_trapezoid_or_parallelogram_l1032_103296


namespace least_positive_angle_theorem_l1032_103200

-- Define the equation
def equation (θ : Real) : Prop :=
  Real.cos (15 * Real.pi / 180) = Real.sin (35 * Real.pi / 180) + Real.sin (θ * Real.pi / 180)

-- State the theorem
theorem least_positive_angle_theorem :
  ∃ (θ : Real), θ > 0 ∧ equation θ ∧ ∀ (φ : Real), φ > 0 ∧ equation φ → θ ≤ φ ∧ θ = 35 :=
sorry

end least_positive_angle_theorem_l1032_103200


namespace spinner_probability_l1032_103295

theorem spinner_probability (p_largest p_next_largest p_smallest : ℝ) : 
  p_largest = (1 : ℝ) / 2 →
  p_next_largest = (1 : ℝ) / 3 →
  p_largest + p_next_largest + p_smallest = 1 →
  p_smallest = (1 : ℝ) / 6 := by
sorry

end spinner_probability_l1032_103295


namespace trapezoid_median_equilateral_triangles_l1032_103233

/-- The median of a trapezoid formed by sides of two equilateral triangles -/
theorem trapezoid_median_equilateral_triangles 
  (large_side : ℝ) 
  (small_side : ℝ) 
  (h1 : large_side = 4) 
  (h2 : small_side = large_side / 2) : 
  (large_side + small_side) / 2 = 3 := by
  sorry

#check trapezoid_median_equilateral_triangles

end trapezoid_median_equilateral_triangles_l1032_103233


namespace cubic_roots_from_known_root_l1032_103293

/-- Given a cubic polynomial P(x) = x^3 + ax^2 + bx + c and a known root α,
    the other roots of P(x) are the roots of the quadratic polynomial Q(x)
    obtained by dividing P(x) by (x - α). -/
theorem cubic_roots_from_known_root (a b c α : ℝ) :
  (α^3 + a*α^2 + b*α + c = 0) →
  ∃ (p q : ℝ),
    (∀ x, x^3 + a*x^2 + b*x + c = (x - α) * (x^2 + p*x + q)) ∧
    (∀ x, x ≠ α ∧ x^3 + a*x^2 + b*x + c = 0 ↔ x^2 + p*x + q = 0) :=
by sorry

end cubic_roots_from_known_root_l1032_103293


namespace equations_solutions_l1032_103267

-- Define the equations
def equation1 (x : ℝ) : Prop := x^2 - 4*x - 1 = 0
def equation2 (x : ℝ) : Prop := (x + 3)^2 = x + 3

-- Define the solution sets
def solutions1 : Set ℝ := {2 + Real.sqrt 5, 2 - Real.sqrt 5}
def solutions2 : Set ℝ := {-3, -2}

-- Theorem statement
theorem equations_solutions :
  (∀ x : ℝ, equation1 x ↔ x ∈ solutions1) ∧
  (∀ x : ℝ, equation2 x ↔ x ∈ solutions2) := by
  sorry

end equations_solutions_l1032_103267


namespace triangle_properties_l1032_103247

-- Define the triangle ABC
structure Triangle where
  A : Real
  B : Real
  C : Real
  a : Real
  b : Real
  c : Real

-- State the theorem
theorem triangle_properties (t : Triangle) 
  (h1 : t.b = 3)
  (h2 : (Real.cos t.A / Real.cos t.B) + (Real.sin t.A / Real.sin t.B) = 2 * t.c / t.b)
  (h3 : t.A + t.B + t.C = Real.pi)
  (h4 : t.A > 0 ∧ t.B > 0 ∧ t.C > 0)
  (h5 : t.a > 0 ∧ t.b > 0 ∧ t.c > 0) :
  t.B = Real.pi / 3 ∧ 
  (∀ (t' : Triangle), t'.b = 3 → t'.a + t'.b + t'.c ≤ 9) := by
  sorry

end triangle_properties_l1032_103247


namespace least_five_digit_square_cube_l1032_103285

theorem least_five_digit_square_cube : ∃ n : ℕ,
  (10000 ≤ n ∧ n < 100000) ∧  -- five-digit number
  (∃ a : ℕ, n = a^2) ∧        -- perfect square
  (∃ b : ℕ, n = b^3) ∧        -- perfect cube
  (∀ m : ℕ, 
    (10000 ≤ m ∧ m < 100000) ∧ 
    (∃ x : ℕ, m = x^2) ∧ 
    (∃ y : ℕ, m = y^3) → 
    n ≤ m) ∧                  -- least such number
  n = 15625 := by
sorry

end least_five_digit_square_cube_l1032_103285


namespace sequence_general_term_l1032_103239

/-- Given a sequence {a_n} where the sum of the first n terms is S_n = n^2 + 3n,
    prove that the general term a_n = 2n + 2 for all positive integers n. -/
theorem sequence_general_term (a : ℕ+ → ℝ) (S : ℕ+ → ℝ) 
    (h_S : ∀ n : ℕ+, S n = n^2 + 3*n) :
    ∀ n : ℕ+, a n = 2*n + 2 := by
  sorry

end sequence_general_term_l1032_103239


namespace size_relationship_l1032_103219

theorem size_relationship : ∀ a b c : ℝ,
  a = 2^(1/2) → b = 3^(1/3) → c = 5^(1/5) →
  b > a ∧ a > c := by
  sorry

end size_relationship_l1032_103219


namespace additional_discount_percentage_l1032_103226

def initial_budget : ℝ := 1000
def first_discount : ℝ := 100
def total_discount : ℝ := 280

def price_after_first_discount : ℝ := initial_budget - first_discount
def final_price : ℝ := initial_budget - total_discount
def additional_discount : ℝ := price_after_first_discount - final_price

theorem additional_discount_percentage : 
  (additional_discount / price_after_first_discount) * 100 = 20 := by
sorry

end additional_discount_percentage_l1032_103226


namespace factorization_equality_l1032_103277

theorem factorization_equality (m n : ℝ) : m^2 - n^2 + 2*m - 2*n = (m-n)*(m+n+2) := by
  sorry

end factorization_equality_l1032_103277


namespace car_speed_equality_l1032_103213

/-- Proves that given the conditions of the car problem, the average speed of Car Y is equal to the average speed of Car X -/
theorem car_speed_equality (speed_x : ℝ) (start_delay : ℝ) (distance_after_y_starts : ℝ) : 
  speed_x = 35 →
  start_delay = 72 / 60 →
  distance_after_y_starts = 98 →
  ∃ (speed_y : ℝ), speed_y = speed_x :=
by
  sorry

#check car_speed_equality

end car_speed_equality_l1032_103213


namespace race_difference_l1032_103266

/-- Given a race where A and B run 110 meters, with A finishing in 20 seconds
    and B finishing in 25 seconds, prove that A beats B by 22 meters. -/
theorem race_difference (race_distance : ℝ) (a_time b_time : ℝ) 
  (h_distance : race_distance = 110)
  (h_a_time : a_time = 20)
  (h_b_time : b_time = 25) :
  race_distance - (race_distance / b_time) * a_time = 22 :=
by sorry

end race_difference_l1032_103266


namespace arithmetic_mean_of_4_and_16_l1032_103255

theorem arithmetic_mean_of_4_and_16 (m : ℝ) : 
  m = (4 + 16) / 2 → m = 10 := by sorry

end arithmetic_mean_of_4_and_16_l1032_103255


namespace sphere_segment_height_ratio_l1032_103256

/-- Given a sphere of radius R and a plane cutting a segment from it, 
    if the ratio of the segment's volume to the volume of a cone with 
    the same base and height is n, then the height h of the segment 
    is given by h = R / (3 - n), where n < 3 -/
theorem sphere_segment_height_ratio 
  (R : ℝ) 
  (n : ℝ) 
  (h : ℝ) 
  (hn : n < 3) 
  (hR : R > 0) :
  (π * R^2 * (h - R/3)) / ((1/3) * π * R^2 * h) = n → 
  h = R / (3 - n) :=
by sorry

end sphere_segment_height_ratio_l1032_103256


namespace prism_18_edges_has_8_faces_l1032_103274

/-- A prism is a polyhedron with a specific structure. -/
structure Prism where
  edges : ℕ
  lateral_faces : ℕ
  total_faces : ℕ

/-- Theorem: A prism with 18 edges has 8 faces. -/
theorem prism_18_edges_has_8_faces :
  ∀ (p : Prism), p.edges = 18 → p.total_faces = 8 := by
  sorry


end prism_18_edges_has_8_faces_l1032_103274


namespace polynomial_division_degree_l1032_103249

theorem polynomial_division_degree (f d q r : Polynomial ℝ) : 
  Polynomial.degree f = 15 →
  f = d * q + r →
  Polynomial.degree q = 8 →
  r = 5 * X^2 + 3 * X - 9 →
  Polynomial.degree d = 7 := by
sorry

end polynomial_division_degree_l1032_103249


namespace subset_intersection_condition_solution_set_eq_interval_l1032_103232

-- Define the sets A and B
def A (a : ℝ) : Set ℝ := {x | 2*a + 1 ≤ x ∧ x ≤ 3*a - 5}
def B : Set ℝ := {x | 3 ≤ x ∧ x ≤ 22}

-- Define the theorem
theorem subset_intersection_condition (a : ℝ) :
  (A a).Nonempty ∧ (A a) ⊆ (A a) ∩ B ↔ 6 ≤ a ∧ a ≤ 9 := by
  sorry

-- Define the set of all 'a' that satisfies the condition
def solution_set : Set ℝ := {a | (A a).Nonempty ∧ (A a) ⊆ (A a) ∩ B}

-- Prove that the solution set is equal to the interval [6, 9]
theorem solution_set_eq_interval :
  solution_set = {a | 6 ≤ a ∧ a ≤ 9} := by
  sorry

end subset_intersection_condition_solution_set_eq_interval_l1032_103232


namespace cos_cube_decomposition_l1032_103245

theorem cos_cube_decomposition (b₁ b₂ b₃ : ℝ) :
  (∀ θ : ℝ, Real.cos θ ^ 3 = b₁ * Real.cos θ + b₂ * Real.cos (2 * θ) + b₃ * Real.cos (3 * θ)) →
  b₁^2 + b₂^2 + b₃^2 = 5/8 := by
  sorry

end cos_cube_decomposition_l1032_103245


namespace guayaquilean_sum_of_digits_l1032_103254

/-- Sum of digits of a natural number -/
def sum_of_digits (n : ℕ) : ℕ := sorry

/-- A natural number is guayaquilean if the sum of its digits equals the sum of digits of its square -/
def is_guayaquilean (n : ℕ) : Prop :=
  sum_of_digits n = sum_of_digits (n^2)

/-- The sum of digits of a guayaquilean number is either 9k or 9k + 1 for some k -/
theorem guayaquilean_sum_of_digits (n : ℕ) (h : is_guayaquilean n) :
  ∃ k : ℕ, sum_of_digits n = 9 * k ∨ sum_of_digits n = 9 * k + 1 :=
sorry

end guayaquilean_sum_of_digits_l1032_103254


namespace seq_of_nat_countable_l1032_103263

/-- The set of all sequences of n natural numbers -/
def SeqOfNat (n : ℕ) : Set (Fin n → ℕ) := Set.univ

/-- A set is countable if there exists an injection from the set to ℕ -/
def IsCountable (α : Type*) : Prop := ∃ f : α → ℕ, Function.Injective f

/-- For any natural number n, the set of all sequences of n natural numbers is countable -/
theorem seq_of_nat_countable (n : ℕ) : IsCountable (SeqOfNat n) := by sorry

end seq_of_nat_countable_l1032_103263


namespace will_uses_six_pages_l1032_103269

/-- The number of cards Will can put on each page -/
def cards_per_page : ℕ := 3

/-- The number of new cards Will has -/
def new_cards : ℕ := 8

/-- The number of old cards Will has -/
def old_cards : ℕ := 10

/-- The total number of cards Will has -/
def total_cards : ℕ := new_cards + old_cards

/-- The number of pages Will uses -/
def pages_used : ℕ := total_cards / cards_per_page

theorem will_uses_six_pages : pages_used = 6 := by
  sorry

end will_uses_six_pages_l1032_103269


namespace continuous_at_8_l1032_103268

def f (x : ℝ) : ℝ := 5 * x^2 + 5

theorem continuous_at_8 :
  ∀ ε > 0, ∃ δ > 0, ∀ x, |x - 8| < δ → |f x - f 8| < ε := by
sorry

end continuous_at_8_l1032_103268


namespace last_nonzero_digit_factorial_not_periodic_l1032_103282

/-- The last nonzero digit of a natural number -/
def lastNonzeroDigit (n : ℕ) : ℕ := sorry

/-- The sequence of last nonzero digits of factorials -/
def a (n : ℕ) : ℕ := lastNonzeroDigit (n.factorial)

/-- A sequence is eventually periodic if there exists some point after which it repeats with a fixed period -/
def EventuallyPeriodic (f : ℕ → ℕ) : Prop :=
  ∃ (N p : ℕ), p > 0 ∧ ∀ n ≥ N, f (n + p) = f n

theorem last_nonzero_digit_factorial_not_periodic :
  ¬ EventuallyPeriodic a := sorry

end last_nonzero_digit_factorial_not_periodic_l1032_103282


namespace hyperbola_m_range_l1032_103227

-- Define the hyperbola equation
def is_hyperbola (m : ℝ) : Prop :=
  ∃ (x y : ℝ), x^2 / (2 + m) + y^2 / (m + 1) = 1

-- Define the range of m
def m_range (m : ℝ) : Prop :=
  -2 < m ∧ m < -1

-- Theorem statement
theorem hyperbola_m_range :
  ∀ m : ℝ, is_hyperbola m → m_range m :=
by
  sorry

end hyperbola_m_range_l1032_103227


namespace smallest_positive_solution_l1032_103208

def equation (x : ℝ) : Prop :=
  (3 * x) / (x - 3) + (3 * x^2 - 45) / (x + 3) = 14

theorem smallest_positive_solution :
  ∃ (x : ℝ), x > 0 ∧ equation x ∧ ∀ (y : ℝ), y > 0 ∧ equation y → x ≤ y :=
by
  use 9
  sorry

end smallest_positive_solution_l1032_103208


namespace smallest_a_l1032_103242

/-- The polynomial with four positive integer roots -/
def p (a b c : ℤ) (x : ℤ) : ℤ := x^4 - a*x^3 + b*x^2 - c*x + 2520

/-- The proposition that the polynomial has four positive integer roots -/
def has_four_positive_integer_roots (a b c : ℤ) : Prop :=
  ∃ w x y z : ℤ, w > 0 ∧ x > 0 ∧ y > 0 ∧ z > 0 ∧
    ∀ t : ℤ, p a b c t = 0 ↔ t = w ∨ t = x ∨ t = y ∨ t = z

/-- The theorem stating that 29 is the smallest possible value of a -/
theorem smallest_a :
  ∀ a b c : ℤ, has_four_positive_integer_roots a b c →
  (∀ a' : ℤ, has_four_positive_integer_roots a' b c → a ≤ a') →
  a = 29 :=
sorry

end smallest_a_l1032_103242


namespace quadratic_polynomial_property_l1032_103297

-- Define the quadratic polynomial f
def f (p q r : ℝ) (x : ℝ) : ℝ := p * x^2 + q * x + r

-- State the theorem
theorem quadratic_polynomial_property 
  (a b c : ℝ) 
  (ha : a ≠ b) (hb : b ≠ c) (hc : a ≠ c)
  (hf : ∃ (p q r : ℝ), 
    f p q r a = b * c ∧ 
    f p q r b = c * a ∧ 
    f p q r c = a * b) : 
  ∃ (p q r : ℝ), f p q r (a + b + c) = a * b + b * c + a * c := by
sorry

end quadratic_polynomial_property_l1032_103297


namespace waiter_earnings_l1032_103260

theorem waiter_earnings (total_customers : ℕ) (non_tipping_customers : ℕ) (tip_amount : ℕ) : 
  total_customers = 7 → 
  non_tipping_customers = 5 → 
  tip_amount = 3 → 
  (total_customers - non_tipping_customers) * tip_amount = 6 := by
sorry

end waiter_earnings_l1032_103260


namespace distinct_arrangements_l1032_103229

/-- The number of distinct arrangements of 6 indistinguishable objects of one type
    and 4 indistinguishable objects of another type in a row of 10 positions -/
def arrangement_count : ℕ := Nat.choose 10 4

/-- Theorem stating that the number of distinct arrangements is 210 -/
theorem distinct_arrangements :
  arrangement_count = 210 := by sorry

end distinct_arrangements_l1032_103229


namespace inequality_equivalence_l1032_103230

theorem inequality_equivalence (x : ℝ) :
  (x - 2) * (2 * x + 3) ≠ 0 →
  ((10 * x^3 - x^2 - 38 * x + 40) / ((x - 2) * (2 * x + 3)) < 2) ↔ (x < 4/5) := by
  sorry

end inequality_equivalence_l1032_103230


namespace revenue_decrease_percentage_l1032_103211

theorem revenue_decrease_percentage (old_revenue new_revenue : ℝ) 
  (h1 : old_revenue = 69.0)
  (h2 : new_revenue = 52.0) :
  ∃ (ε : ℝ), abs ((old_revenue - new_revenue) / old_revenue * 100 - 24.64) < ε ∧ ε > 0 :=
by sorry

end revenue_decrease_percentage_l1032_103211


namespace perpendicular_lines_condition_l1032_103218

theorem perpendicular_lines_condition (m : ℝ) :
  (∀ x y, m * x + 2 * y - 1 = 0 → 3 * x + (m + 1) * y + 1 = 0 → 
    (m * 3 + 2 * (m + 1) = 0)) ↔ m = -2/5 := by
  sorry

end perpendicular_lines_condition_l1032_103218


namespace quadratic_inequality_solution_set_l1032_103292

theorem quadratic_inequality_solution_set :
  let f : ℝ → ℝ := λ x => 2 * x^2 + 7 * x + 3
  let solution_set : Set ℝ := {x | f x > 0}
  solution_set = {x | x < -3 ∨ x > -0.5} := by
sorry

end quadratic_inequality_solution_set_l1032_103292


namespace max_abs_z5_l1032_103283

theorem max_abs_z5 (z₁ z₂ z₃ z₄ z₅ : ℂ)
  (h1 : Complex.abs z₁ ≤ 1)
  (h2 : Complex.abs z₂ ≤ 1)
  (h3 : Complex.abs (2 * z₃ - (z₁ + z₂)) ≤ Complex.abs (z₁ - z₂))
  (h4 : Complex.abs (2 * z₄ - (z₁ + z₂)) ≤ Complex.abs (z₁ - z₂))
  (h5 : Complex.abs (2 * z₅ - (z₃ + z₄)) ≤ Complex.abs (z₃ - z₄)) :
  Complex.abs z₅ ≤ Real.sqrt 3 ∧ ∃ z₁ z₂ z₃ z₄ z₅ : ℂ, 
    Complex.abs z₁ ≤ 1 ∧
    Complex.abs z₂ ≤ 1 ∧
    Complex.abs (2 * z₃ - (z₁ + z₂)) ≤ Complex.abs (z₁ - z₂) ∧
    Complex.abs (2 * z₄ - (z₁ + z₂)) ≤ Complex.abs (z₁ - z₂) ∧
    Complex.abs (2 * z₅ - (z₃ + z₄)) ≤ Complex.abs (z₃ - z₄) ∧
    Complex.abs z₅ = Real.sqrt 3 :=
by sorry

end max_abs_z5_l1032_103283


namespace geometric_sequence_common_ratio_l1032_103214

theorem geometric_sequence_common_ratio 
  (a : ℕ → ℝ) 
  (h_geom : ∀ n, a (n + 1) = a n * q) 
  (h_sum1 : a 2 + a 3 = 2) 
  (h_sum2 : a 4 + a 5 = 32) : 
  q = 4 ∨ q = -4 := by
sorry

end geometric_sequence_common_ratio_l1032_103214


namespace power_zero_equations_l1032_103234

theorem power_zero_equations (a : ℝ) (h : a ≠ 0) :
  (∃ x, (x + 2)^0 ≠ 1) ∧
  ((a^2 + 1)^0 = 1) ∧
  ((-6*a)^0 = 1) ∧
  ((1/a)^0 = 1) :=
sorry

end power_zero_equations_l1032_103234


namespace granary_circumference_l1032_103280

/-- Represents the height of the granary in chi -/
def granary_height : ℝ := 13.325

/-- Represents the volume of the granary in cubic chi -/
def granary_volume : ℝ := 2000 * 1.62

/-- Approximation of π -/
def π_approx : ℝ := 3

theorem granary_circumference :
  let base_area := granary_volume / granary_height
  let radius := Real.sqrt (base_area / π_approx)
  2 * π_approx * radius = 54 := by sorry

end granary_circumference_l1032_103280


namespace pies_from_apples_l1032_103275

/-- Given the rate of pies per apples and a new number of apples, calculate the number of pies that can be made -/
def calculate_pies (initial_apples : ℕ) (initial_pies : ℕ) (new_apples : ℕ) : ℕ :=
  (new_apples * initial_pies) / initial_apples

/-- Theorem stating that given 3 pies can be made from 15 apples, 45 apples will yield 9 pies -/
theorem pies_from_apples :
  calculate_pies 15 3 45 = 9 := by
  sorry

end pies_from_apples_l1032_103275


namespace pepper_difference_l1032_103225

/-- Represents the types of curry based on spice level -/
inductive CurryType
| VerySpicy
| Spicy
| Mild

/-- Returns the number of peppers needed for a given curry type -/
def peppersNeeded (c : CurryType) : ℕ :=
  match c with
  | .VerySpicy => 3
  | .Spicy => 2
  | .Mild => 1

/-- Calculates the total number of peppers needed for a given number of curries of each type -/
def totalPeppers (verySpicy spicy mild : ℕ) : ℕ :=
  verySpicy * peppersNeeded CurryType.VerySpicy +
  spicy * peppersNeeded CurryType.Spicy +
  mild * peppersNeeded CurryType.Mild

/-- The main theorem stating the difference in peppers bought -/
theorem pepper_difference : 
  totalPeppers 30 30 10 - totalPeppers 0 15 90 = 40 := by
  sorry

#eval totalPeppers 30 30 10 - totalPeppers 0 15 90

end pepper_difference_l1032_103225


namespace inequalities_for_positive_reals_l1032_103210

theorem inequalities_for_positive_reals (a b : ℝ) 
  (h_pos_a : a > 0) (h_pos_b : b > 0) (h_sum : a + b = 2) : 
  a * b ≤ 1 ∧ a^2 + b^2 ≥ 2 ∧ Real.sqrt a + Real.sqrt b ≤ 2 := by
  sorry

end inequalities_for_positive_reals_l1032_103210


namespace factorization_result_quadratic_factorization_l1032_103235

-- Part 1
theorem factorization_result (a b : ℤ) :
  (∀ x, (2*x - 21) * (3*x - 7) - (3*x - 7) * (x - 13) = (3*x + a) * (x + b)) →
  a + 3*b = -31 := by sorry

-- Part 2
theorem quadratic_factorization :
  ∀ x, x^2 - 3*x + 2 = (x - 1) * (x - 2) := by sorry

end factorization_result_quadratic_factorization_l1032_103235


namespace right_triangle_hypotenuse_l1032_103284

theorem right_triangle_hypotenuse (a q : ℝ) (ha : a > 0) (hq : q > 0) :
  ∃ (b c : ℝ),
    c > 0 ∧
    a^2 + b^2 = c^2 ∧
    q * c = b^2 ∧
    c = q / 2 + Real.sqrt ((q / 2)^2 + a^2) :=
by sorry

end right_triangle_hypotenuse_l1032_103284


namespace merry_apples_sold_l1032_103207

/-- The number of apples Merry sold on Saturday and Sunday -/
def apples_sold (saturday_boxes : ℕ) (sunday_boxes : ℕ) (apples_per_box : ℕ) (boxes_left : ℕ) : ℕ :=
  (saturday_boxes - sunday_boxes + sunday_boxes - boxes_left) * apples_per_box

/-- Theorem stating that Merry sold 470 apples on Saturday and Sunday -/
theorem merry_apples_sold :
  apples_sold 50 25 10 3 = 470 := by
  sorry

end merry_apples_sold_l1032_103207


namespace negation_of_existence_negation_of_quadratic_inequality_l1032_103286

theorem negation_of_existence (P : ℝ → Prop) :
  (¬ ∃ x : ℝ, P x) ↔ (∀ x : ℝ, ¬ P x) :=
by sorry

theorem negation_of_quadratic_inequality :
  (¬ ∃ x : ℝ, x^2 + 2*x + 2 > 0) ↔ (∀ x : ℝ, x^2 + 2*x + 2 ≤ 0) :=
by sorry

end negation_of_existence_negation_of_quadratic_inequality_l1032_103286


namespace goldfish_ratio_l1032_103231

/-- Proves the ratio of goldfish Bexley brought to Hershel's initial goldfish -/
theorem goldfish_ratio :
  ∀ (hershel_betta hershel_goldfish bexley_goldfish : ℕ),
  hershel_betta = 10 →
  hershel_goldfish = 15 →
  ∃ (total_after_gift : ℕ),
    total_after_gift = 17 ∧
    (hershel_betta + (2 / 5 : ℚ) * hershel_betta + hershel_goldfish + bexley_goldfish) / 2 = total_after_gift →
    bexley_goldfish * 3 = hershel_goldfish :=
by
  sorry

end goldfish_ratio_l1032_103231


namespace bisection_method_calculations_l1032_103272

theorem bisection_method_calculations (a b : Real) (accuracy : Real) :
  a = 1.4 →
  b = 1.5 →
  accuracy = 0.001 →
  ∃ n : ℕ, (((b - a) / (2 ^ n : Real)) < accuracy) ∧ 
    (∀ m : ℕ, m < n → ((b - a) / (2 ^ m : Real)) ≥ accuracy) ∧
    n = 7 :=
by sorry

end bisection_method_calculations_l1032_103272


namespace triangle_inequality_arithmetic_sequence_l1032_103262

/-- An arithmetic sequence with positive terms and positive common difference -/
def ArithmeticSequence (a : ℕ → ℝ) (d : ℝ) : Prop :=
  (∀ n, a n > 0) ∧ d > 0 ∧ ∀ n, a (n + 1) = a n + d

theorem triangle_inequality_arithmetic_sequence 
  (a : ℕ → ℝ) (d : ℝ) (h : ArithmeticSequence a d) :
  a 2 + a 3 > a 4 ∧ a 2 + a 4 > a 3 ∧ a 3 + a 4 > a 2 := by
  sorry

end triangle_inequality_arithmetic_sequence_l1032_103262


namespace arithmetic_sequence_length_l1032_103206

theorem arithmetic_sequence_length (a₁ aₙ d : ℤ) (n : ℕ) 
  (h1 : a₁ = -33)
  (h2 : aₙ = 72)
  (h3 : d = 7)
  (h4 : aₙ = a₁ + (n - 1) * d) :
  n = 16 := by
sorry

end arithmetic_sequence_length_l1032_103206


namespace factorization_of_cubic_l1032_103290

theorem factorization_of_cubic (x : ℝ) : 2 * x^3 - 8 * x = 2 * x * (x + 2) * (x - 2) := by
  sorry

end factorization_of_cubic_l1032_103290


namespace abs_2x_plus_4_not_positive_l1032_103238

theorem abs_2x_plus_4_not_positive (x : ℝ) : |2*x + 4| ≤ 0 ↔ x = -2 := by
  sorry

end abs_2x_plus_4_not_positive_l1032_103238


namespace relation_abc_l1032_103298

theorem relation_abc : 
  let a := (2 : ℝ) ^ (1/5 : ℝ)
  let b := (2/5 : ℝ) ^ (1/5 : ℝ)
  let c := (2/5 : ℝ) ^ (3/5 : ℝ)
  a > b ∧ b > c := by sorry

end relation_abc_l1032_103298


namespace equation_solution_l1032_103250

theorem equation_solution (x y : ℕ) : x^y + y^x = 2408 ∧ x = 2407 → y = 1 := by
  sorry

end equation_solution_l1032_103250


namespace cake_segment_length_squared_l1032_103261

theorem cake_segment_length_squared (d : ℝ) (n : ℕ) (m : ℝ) : 
  d = 20 → n = 4 → m = (d / 2) * Real.sqrt 2 → m^2 = 200 := by
  sorry

end cake_segment_length_squared_l1032_103261


namespace magic_potion_cooking_time_l1032_103291

/-- Represents a time on a 24-hour digital clock --/
structure DigitalTime where
  hours : Fin 24
  minutes : Fin 60

/-- Checks if a given time is a magic moment --/
def isMagicMoment (t : DigitalTime) : Prop :=
  t.hours = t.minutes

/-- Calculates the time difference between two DigitalTimes in minutes --/
def timeDifference (start finish : DigitalTime) : ℕ :=
  sorry

/-- Theorem stating the existence of a valid cooking time for the magic potion --/
theorem magic_potion_cooking_time :
  ∃ (start finish : DigitalTime),
    isMagicMoment start ∧
    isMagicMoment finish ∧
    90 ≤ timeDifference start finish ∧
    timeDifference start finish ≤ 120 ∧
    timeDifference start finish = 98 :=
  sorry

end magic_potion_cooking_time_l1032_103291


namespace f_geq_a_iff_a_in_range_l1032_103243

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := x^2 - 2*a*x + 2

-- Define the domain
def domain : Set ℝ := {x : ℝ | x ≥ -1}

-- State the theorem
theorem f_geq_a_iff_a_in_range (a : ℝ) : 
  (∀ x ∈ domain, f a x ≥ a) ↔ a ∈ Set.Icc (-3) 1 := by sorry

end f_geq_a_iff_a_in_range_l1032_103243


namespace ratio_sum_quotient_l1032_103204

theorem ratio_sum_quotient (x y : ℚ) (h : x / y = 1 / 2) : (x + y) / y = 3 / 2 := by
  sorry

end ratio_sum_quotient_l1032_103204


namespace largest_B_term_l1032_103264

/-- The binomial coefficient -/
def binomial (n k : ℕ) : ℕ := sorry

/-- The term B_k in the binomial expansion -/
def B (k : ℕ) : ℝ := (binomial 500 k) * (0.3 ^ k)

/-- The theorem stating that B_k is largest when k = 125 -/
theorem largest_B_term : ∀ k : ℕ, k ≤ 500 → B k ≤ B 125 := by sorry

end largest_B_term_l1032_103264


namespace fraction_sum_l1032_103288

theorem fraction_sum (a b : ℚ) (h : a / b = 2 / 5) : (a + b) / b = 7 / 5 := by
  sorry

end fraction_sum_l1032_103288


namespace tomorrow_sunny_is_uncertain_l1032_103252

-- Define the type for events
inductive Event : Type
  | certain : Event
  | impossible : Event
  | inevitable : Event
  | uncertain : Event

-- Define the event "Tomorrow will be sunny"
def tomorrow_sunny : Event := Event.uncertain

-- Define the properties of events
def is_guaranteed (e : Event) : Prop :=
  e = Event.certain ∨ e = Event.inevitable

def cannot_happen (e : Event) : Prop :=
  e = Event.impossible

def is_not_guaranteed (e : Event) : Prop :=
  e = Event.uncertain

-- Theorem statement
theorem tomorrow_sunny_is_uncertain :
  is_not_guaranteed tomorrow_sunny ∧
  ¬is_guaranteed tomorrow_sunny ∧
  ¬cannot_happen tomorrow_sunny :=
by sorry

end tomorrow_sunny_is_uncertain_l1032_103252


namespace abs_sum_inequality_l1032_103217

theorem abs_sum_inequality (k : ℝ) : 
  (∀ x : ℝ, |x + 2| + |x + 1| > k) → k < 1 :=
by sorry

end abs_sum_inequality_l1032_103217


namespace fraction_equation_solution_l1032_103216

theorem fraction_equation_solution :
  ∀ y : ℚ, (2 / 5 : ℚ) - (1 / 3 : ℚ) = 4 / y → y = 60 := by
  sorry

end fraction_equation_solution_l1032_103216


namespace lcm_18_20_l1032_103202

theorem lcm_18_20 : Nat.lcm 18 20 = 180 := by
  sorry

end lcm_18_20_l1032_103202


namespace y_value_at_8_l1032_103209

-- Define the function y = k * x^(1/3)
def y (k x : ℝ) : ℝ := k * x^(1/3)

-- State the theorem
theorem y_value_at_8 (k : ℝ) :
  y k 64 = 4 * Real.sqrt 3 → y k 8 = 2 * Real.sqrt 3 := by
  sorry

end y_value_at_8_l1032_103209


namespace puppy_cost_l1032_103287

theorem puppy_cost (items_cost : ℝ) (discount_rate : ℝ) (total_spent : ℝ)
  (h1 : items_cost = 95)
  (h2 : discount_rate = 0.2)
  (h3 : total_spent = 96) :
  total_spent - items_cost * (1 - discount_rate) = 20 :=
by sorry

end puppy_cost_l1032_103287


namespace goat_difference_l1032_103258

theorem goat_difference (adam_goats andrew_goats ahmed_goats : ℕ) : 
  adam_goats = 7 →
  ahmed_goats = 13 →
  andrew_goats = ahmed_goats + 6 →
  andrew_goats - 2 * adam_goats = 5 := by
sorry

end goat_difference_l1032_103258


namespace greatest_k_for_inequality_l1032_103241

theorem greatest_k_for_inequality (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  ∃ k : ℝ, k > 0 ∧ 
  (∀ a b c : ℝ, a > 0 → b > 0 → c > 0 → 
    a/b + b/c + c/a - 3 ≥ k * (a/(b+c) + b/(c+a) + c/(a+b) - 3/2)) ∧
  (∀ k' : ℝ, k' > k → 
    ∃ a b c : ℝ, a > 0 ∧ b > 0 ∧ c > 0 ∧
    a/b + b/c + c/a - 3 < k' * (a/(b+c) + b/(c+a) + c/(a+b) - 3/2)) ∧
  k = 1 :=
sorry

end greatest_k_for_inequality_l1032_103241


namespace local_max_is_four_l1032_103257

/-- Given that x = 1 is a point of local minimum for f(x) = x³ - 3ax + 2,
    prove that the point of local maximum for f(x) is 4. -/
theorem local_max_is_four (a : ℝ) :
  let f : ℝ → ℝ := λ x ↦ x^3 - 3*a*x + 2
  (∀ h ∈ Set.Ioo (1 - ε) (1 + ε), f 1 ≤ f h) →
  ∃ ε > 0, ∀ h ∈ Set.Ioo (-1 - ε) (-1 + ε), f h ≤ f (-1) ∧ f (-1) = 4 :=
by sorry

end local_max_is_four_l1032_103257


namespace bear_census_l1032_103215

def total_bears (black_a : ℕ) : ℕ :=
  let black_b := 3 * black_a
  let black_c := 2 * black_b
  let white_a := black_a / 2
  let white_b := black_b / 2
  let white_c := black_c / 2
  let brown_a := black_a + 40
  let brown_b := black_b + 40
  let brown_c := black_c + 40
  black_a + black_b + black_c +
  white_a + white_b + white_c +
  brown_a + brown_b + brown_c

theorem bear_census (black_a : ℕ) (h1 : black_a = 60) :
  total_bears black_a = 1620 := by
  sorry

end bear_census_l1032_103215


namespace cow_chicken_leg_excess_l1032_103228

/-- Represents the number of legs more than twice the number of heads in a group of cows and chickens -/
def excess_legs (num_chickens : ℕ) : ℕ :=
  (4 * 10 + 2 * num_chickens) - 2 * (10 + num_chickens)

theorem cow_chicken_leg_excess :
  ∀ num_chickens : ℕ, excess_legs num_chickens = 20 := by
  sorry

end cow_chicken_leg_excess_l1032_103228


namespace well_depth_and_rope_length_l1032_103265

theorem well_depth_and_rope_length :
  ∃! (x y : ℝ),
    x / 4 - 3 = y ∧
    x / 5 + 1 = y ∧
    x = 80 ∧
    y = 17 := by
  sorry

end well_depth_and_rope_length_l1032_103265


namespace rosalina_gifts_l1032_103236

/-- The number of gifts Rosalina received from Emilio -/
def emilio_gifts : ℕ := 11

/-- The number of gifts Rosalina received from Jorge -/
def jorge_gifts : ℕ := 6

/-- The number of gifts Rosalina received from Pedro -/
def pedro_gifts : ℕ := 4

/-- The total number of gifts Rosalina received -/
def total_gifts : ℕ := emilio_gifts + jorge_gifts + pedro_gifts

theorem rosalina_gifts : total_gifts = 21 := by
  sorry

end rosalina_gifts_l1032_103236


namespace quadratic_increasing_condition_l1032_103212

/-- Given a quadratic function f(x) = x^2 - 2ax + 1 that is increasing on [1, +∞),
    prove that a ≤ 1 -/
theorem quadratic_increasing_condition (a : ℝ) : 
  (∀ x ≥ 1, ∀ y ≥ x, (y^2 - 2*a*y + 1) ≥ (x^2 - 2*a*x + 1)) → a ≤ 1 := by
sorry

end quadratic_increasing_condition_l1032_103212


namespace airplane_passengers_virginia_l1032_103281

/-- Calculates the number of people landing in Virginia given the flight conditions -/
theorem airplane_passengers_virginia
  (initial_passengers : ℕ)
  (texas_off texas_on : ℕ)
  (nc_off nc_on : ℕ)
  (crew : ℕ)
  (h1 : initial_passengers = 124)
  (h2 : texas_off = 58)
  (h3 : texas_on = 24)
  (h4 : nc_off = 47)
  (h5 : nc_on = 14)
  (h6 : crew = 10) :
  initial_passengers - texas_off + texas_on - nc_off + nc_on + crew = 67 :=
by sorry

end airplane_passengers_virginia_l1032_103281


namespace smallest_x_value_l1032_103220

theorem smallest_x_value : ∃ x : ℚ, 
  (∀ y : ℚ, 7 * (8 * y^2 + 8 * y + 11) = y * (8 * y - 35) → x ≤ y) ∧
  7 * (8 * x^2 + 8 * x + 11) = x * (8 * x - 35) ∧
  x = -7/3 := by
  sorry

end smallest_x_value_l1032_103220


namespace flower_cost_minimization_l1032_103201

/-- The cost of flowers given the number of carnations -/
def cost (x : ℕ) : ℕ := 55 - x

/-- The problem statement -/
theorem flower_cost_minimization :
  let total_flowers : ℕ := 11
  let min_lilies : ℕ := 2
  let carnation_cost : ℕ := 4
  let lily_cost : ℕ := 5
  (2 * lily_cost + carnation_cost = 14) →
  (3 * carnation_cost = 2 * lily_cost + 2) →
  (∀ x : ℕ, x ≤ total_flowers - min_lilies → cost x = 55 - x) →
  (∃ x : ℕ, x ≤ total_flowers - min_lilies ∧ cost x = 46 ∧ 
    ∀ y : ℕ, y ≤ total_flowers - min_lilies → cost y ≥ cost x) := by
  sorry

end flower_cost_minimization_l1032_103201

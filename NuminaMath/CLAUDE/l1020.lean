import Mathlib

namespace NUMINAMATH_CALUDE_rectangular_prism_volume_specific_prism_volume_l1020_102035

/-- A right rectangular prism with face areas a, b, and c has volume equal to the square root of their product. -/
theorem rectangular_prism_volume (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  ∃ l w h : ℝ, l > 0 ∧ w > 0 ∧ h > 0 ∧
  l * w = a ∧ w * h = b ∧ l * h = c ∧
  l * w * h = Real.sqrt (a * b * c) := by
  sorry

/-- The volume of a right rectangular prism with face areas 10, 14, and 35 square inches is 70 cubic inches. -/
theorem specific_prism_volume :
  ∃ l w h : ℝ, l > 0 ∧ w > 0 ∧ h > 0 ∧
  l * w = 10 ∧ w * h = 14 ∧ l * h = 35 ∧
  l * w * h = 70 := by
  sorry

end NUMINAMATH_CALUDE_rectangular_prism_volume_specific_prism_volume_l1020_102035


namespace NUMINAMATH_CALUDE_fraction_sum_and_product_l1020_102002

theorem fraction_sum_and_product : 
  (2 / 16 + 3 / 18 + 4 / 24) * (3 / 5) = 11 / 40 := by
  sorry

end NUMINAMATH_CALUDE_fraction_sum_and_product_l1020_102002


namespace NUMINAMATH_CALUDE_find_r_l1020_102079

theorem find_r (k r : ℝ) (h1 : 5 = k * 3^r) (h2 : 45 = k * 9^r) : r = 2 := by
  sorry

end NUMINAMATH_CALUDE_find_r_l1020_102079


namespace NUMINAMATH_CALUDE_tangent_dot_product_l1020_102021

/-- The circle with center at the origin and radius 1 -/
def unit_circle (x y : ℝ) : Prop := x^2 + y^2 = 1

/-- Point M outside the circle -/
def M : ℝ × ℝ := (2, 0)

/-- A point is on the circle -/
def on_circle (p : ℝ × ℝ) : Prop := unit_circle p.1 p.2

/-- A line is tangent to the circle at a point -/
def is_tangent (p q : ℝ × ℝ) : Prop :=
  on_circle p ∧ (p.1 * q.1 + p.2 * q.2 = 1)

/-- The dot product of two vectors -/
def dot_product (v w : ℝ × ℝ) : ℝ :=
  v.1 * w.1 + v.2 * w.2

theorem tangent_dot_product :
  ∃ (A B : ℝ × ℝ),
    on_circle A ∧
    on_circle B ∧
    is_tangent A M ∧
    is_tangent B M ∧
    dot_product (A.1 - M.1, A.2 - M.2) (B.1 - M.1, B.2 - M.2) = 3/2 := by
  sorry

end NUMINAMATH_CALUDE_tangent_dot_product_l1020_102021


namespace NUMINAMATH_CALUDE_max_value_expression_l1020_102041

theorem max_value_expression (x y z : ℝ) 
  (h1 : x ≥ 0) (h2 : y ≥ 0) (h3 : z ≥ 0) (h4 : x + y + z = 3) :
  (x^2 - 2*x*y + y^2) * (x^2 - 2*x*z + z^2) * (y^2 - 2*y*z + z^2) ≤ 1 := by
  sorry

end NUMINAMATH_CALUDE_max_value_expression_l1020_102041


namespace NUMINAMATH_CALUDE_six_people_charity_arrangements_l1020_102093

/-- The number of ways to distribute n people into 2 charity activities,
    with each activity accommodating no more than 4 people -/
def charity_arrangements (n : ℕ) : ℕ :=
  -- Definition to be implemented
  sorry

/-- Theorem stating that there are 50 ways to distribute 6 people
    into 2 charity activities with the given constraints -/
theorem six_people_charity_arrangements :
  charity_arrangements 6 = 50 := by
  sorry

end NUMINAMATH_CALUDE_six_people_charity_arrangements_l1020_102093


namespace NUMINAMATH_CALUDE_max_corner_sum_l1020_102058

/-- Represents a face of the cube -/
structure Face where
  value : Nat
  inv_value : Nat
  sum_eq_eight : value + inv_value = 8
  value_in_range : 1 ≤ value ∧ value ≤ 6

/-- Represents a cube with six faces -/
structure Cube where
  faces : Fin 6 → Face
  distinct : ∀ i j, i ≠ j → (faces i).value ≠ (faces j).value

/-- Represents a corner of the cube -/
structure Corner where
  cube : Cube
  face1 : Fin 6
  face2 : Fin 6
  face3 : Fin 6
  distinct : face1 ≠ face2 ∧ face2 ≠ face3 ∧ face1 ≠ face3
  adjacent : ¬ ((cube.faces face1).inv_value = (cube.faces face2).value ∨
                (cube.faces face1).inv_value = (cube.faces face3).value ∨
                (cube.faces face2).inv_value = (cube.faces face3).value)

/-- The sum of values at a corner -/
def cornerSum (c : Corner) : Nat :=
  (c.cube.faces c.face1).value + (c.cube.faces c.face2).value + (c.cube.faces c.face3).value

/-- The theorem to be proved -/
theorem max_corner_sum (c : Cube) : 
  ∀ corner : Corner, corner.cube = c → cornerSum corner ≤ 15 :=
sorry

end NUMINAMATH_CALUDE_max_corner_sum_l1020_102058


namespace NUMINAMATH_CALUDE_parallel_line_through_midpoint_l1020_102059

/-- Given two points A and B in ℝ², and a line L, 
    prove that the line passing through the midpoint of AB 
    and parallel to L has the equation 3x + y + 3 = 0 -/
theorem parallel_line_through_midpoint 
  (A B : ℝ × ℝ) 
  (hA : A = (-5, 2)) 
  (hB : B = (1, 4)) 
  (L : ℝ → ℝ) 
  (hL : ∀ x y, L x = y ↔ 3 * x + y - 2 = 0) :
  let M := ((A.1 + B.1) / 2, (A.2 + B.2) / 2)
  ∀ x y, 3 * x + y + 3 = 0 ↔ 
    ∃ k, y - M.2 = k * (x - M.1) ∧ 
         ∀ x' y', L x' = y' → y' - M.2 = k * (x' - M.1) :=
by sorry

end NUMINAMATH_CALUDE_parallel_line_through_midpoint_l1020_102059


namespace NUMINAMATH_CALUDE_vector_addition_and_scalar_multiplication_l1020_102054

/-- Given vectors a and b in ℝ³, prove that a + 2b equals the expected result. -/
theorem vector_addition_and_scalar_multiplication (a b : ℝ × ℝ × ℝ) :
  a = (3, -2, 1) →
  b = (-2, 4, 0) →
  a + 2 • b = (-1, 6, 1) := by
sorry

end NUMINAMATH_CALUDE_vector_addition_and_scalar_multiplication_l1020_102054


namespace NUMINAMATH_CALUDE_smallest_satisfying_number_l1020_102032

def satisfiesConditions (n : Nat) : Prop :=
  n % 43 = 0 ∧
  n < 43 * 9 ∧
  n % 2 = 1 ∧
  n % 3 = 1 ∧
  n % 4 = 1 ∧
  n % 5 = 1 ∧
  n % 6 = 1

theorem smallest_satisfying_number :
  satisfiesConditions 301 ∧
  ∀ m : Nat, m < 301 → ¬satisfiesConditions m :=
by sorry

end NUMINAMATH_CALUDE_smallest_satisfying_number_l1020_102032


namespace NUMINAMATH_CALUDE_largest_sum_simplification_l1020_102044

theorem largest_sum_simplification : 
  let sums : List ℚ := [1/3 + 1/4, 1/3 + 1/5, 1/3 + 1/2, 1/3 + 1/7, 1/3 + 1/8]
  (∀ s ∈ sums, s ≤ 1/3 + 1/2) ∧ 
  (1/3 + 1/2 = 5/6) := by
  sorry

end NUMINAMATH_CALUDE_largest_sum_simplification_l1020_102044


namespace NUMINAMATH_CALUDE_distance_between_points_l1020_102075

theorem distance_between_points : 
  let p1 : ℝ × ℝ := (2, -2)
  let p2 : ℝ × ℝ := (8, 8)
  Real.sqrt ((p2.1 - p1.1)^2 + (p2.2 - p1.2)^2) = Real.sqrt 136 :=
by sorry

end NUMINAMATH_CALUDE_distance_between_points_l1020_102075


namespace NUMINAMATH_CALUDE_inequality_proof_l1020_102070

theorem inequality_proof (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  a^2 / b + b^2 / c + c^2 / a ≥ a + b + c + 4 * (a - b)^2 / (a + b + c) := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l1020_102070


namespace NUMINAMATH_CALUDE_triangle_t_range_l1020_102073

theorem triangle_t_range (A B C : ℝ) (a b c : ℝ) (t : ℝ) :
  0 < B → B < π / 2 →  -- B is acute
  a > 0 → b > 0 → c > 0 →  -- sides are positive
  a * c = (1 / 4) * b ^ 2 →  -- given condition
  Real.sin A + Real.sin C = t * Real.sin B →  -- given condition
  t ∈ Set.Ioo (Real.sqrt 6 / 2) (Real.sqrt 2) :=
by
  sorry

end NUMINAMATH_CALUDE_triangle_t_range_l1020_102073


namespace NUMINAMATH_CALUDE_david_average_marks_l1020_102033

def david_marks : List ℕ := [74, 65, 82, 67, 90]

theorem david_average_marks :
  (david_marks.sum : ℚ) / david_marks.length = 75.6 := by sorry

end NUMINAMATH_CALUDE_david_average_marks_l1020_102033


namespace NUMINAMATH_CALUDE_three_statements_incorrect_l1020_102089

-- Define the four statements
def statement1 : Prop := ∀ (a : ℕ → ℝ) (S : ℕ → ℝ), 
  (∀ n, a (n+1) - a n = a (n+2) - a (n+1)) → 
  (a 6 + a 7 > 0 ↔ S 9 ≥ S 3)

def statement2 : Prop := 
  (¬ ∃ x : ℝ, x > 1) ↔ (∀ x : ℝ, x < 1)

def statement3 : Prop := 
  (∀ x : ℝ, x^2 - 4*x + 3 = 0 → (x = 1 ∨ x = 3)) ↔
  (∀ x : ℝ, (x ≠ 1 ∨ x ≠ 3) → x^2 - 4*x + 3 ≠ 0)

def statement4 : Prop := 
  ∀ p q : Prop, (¬(p ∨ q)) → (¬p ∧ ¬q)

-- Theorem stating that exactly 3 statements are incorrect
theorem three_statements_incorrect : 
  (¬statement1 ∧ ¬statement2 ∧ ¬statement3 ∧ statement4) :=
sorry

end NUMINAMATH_CALUDE_three_statements_incorrect_l1020_102089


namespace NUMINAMATH_CALUDE_opposite_of_sqrt3_minus_2_l1020_102092

theorem opposite_of_sqrt3_minus_2 :
  -(Real.sqrt 3 - 2) = 2 - Real.sqrt 3 := by sorry

end NUMINAMATH_CALUDE_opposite_of_sqrt3_minus_2_l1020_102092


namespace NUMINAMATH_CALUDE_rhombus_diagonals_perpendicular_rectangle_not_l1020_102084

-- Define a quadrilateral
structure Quadrilateral :=
  (A B C D : ℝ × ℝ)

-- Define a rhombus
def is_rhombus (q : Quadrilateral) : Prop :=
  let (x₁, y₁) := q.A
  let (x₂, y₂) := q.B
  let (x₃, y₃) := q.C
  let (x₄, y₄) := q.D
  -- All sides are equal
  (x₁ - x₂)^2 + (y₁ - y₂)^2 = (x₂ - x₃)^2 + (y₂ - y₃)^2 ∧
  (x₂ - x₃)^2 + (y₂ - y₃)^2 = (x₃ - x₄)^2 + (y₃ - y₄)^2 ∧
  (x₃ - x₄)^2 + (y₃ - y₄)^2 = (x₄ - x₁)^2 + (y₄ - y₁)^2

-- Define a rectangle
def is_rectangle (q : Quadrilateral) : Prop :=
  let (x₁, y₁) := q.A
  let (x₂, y₂) := q.B
  let (x₃, y₃) := q.C
  let (x₄, y₄) := q.D
  -- Opposite sides are equal and all angles are right angles
  (x₁ - x₂)^2 + (y₁ - y₂)^2 = (x₃ - x₄)^2 + (y₃ - y₄)^2 ∧
  (x₂ - x₃)^2 + (y₂ - y₃)^2 = (x₄ - x₁)^2 + (y₄ - y₁)^2 ∧
  (x₂ - x₁) * (x₃ - x₂) + (y₂ - y₁) * (y₃ - y₂) = 0 ∧
  (x₃ - x₂) * (x₄ - x₃) + (y₃ - y₂) * (y₄ - y₃) = 0 ∧
  (x₄ - x₃) * (x₁ - x₄) + (y₄ - y₃) * (y₁ - y₄) = 0 ∧
  (x₁ - x₄) * (x₂ - x₁) + (y₁ - y₄) * (y₂ - y₁) = 0

-- Define perpendicular diagonals
def perpendicular_diagonals (q : Quadrilateral) : Prop :=
  let (x₁, y₁) := q.A
  let (x₃, y₃) := q.C
  let (x₂, y₂) := q.B
  let (x₄, y₄) := q.D
  (x₃ - x₁) * (x₄ - x₂) + (y₃ - y₁) * (y₄ - y₂) = 0

-- Theorem statement
theorem rhombus_diagonals_perpendicular_rectangle_not :
  (∀ q : Quadrilateral, is_rhombus q → perpendicular_diagonals q) ∧
  ¬(∀ q : Quadrilateral, is_rectangle q → perpendicular_diagonals q) :=
sorry

end NUMINAMATH_CALUDE_rhombus_diagonals_perpendicular_rectangle_not_l1020_102084


namespace NUMINAMATH_CALUDE_problem_1_l1020_102065

theorem problem_1 : (-1/12) / (-1/2 + 2/3 + 3/4) = -1/11 := by
  sorry

end NUMINAMATH_CALUDE_problem_1_l1020_102065


namespace NUMINAMATH_CALUDE_inequality_proof_l1020_102048

theorem inequality_proof (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  1 / (b * (a + b)) + 1 / (c * (b + c)) + 1 / (a * (c + a)) ≥ 27 / (2 * (a + b + c)^2) := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l1020_102048


namespace NUMINAMATH_CALUDE_rectangular_solid_surface_area_l1020_102026

/-- A rectangular solid with prime edge lengths and volume 143 has surface area 382 -/
theorem rectangular_solid_surface_area : ∀ a b c : ℕ,
  Prime a → Prime b → Prime c →
  a * b * c = 143 →
  2 * (a * b + b * c + c * a) = 382 := by
  sorry

end NUMINAMATH_CALUDE_rectangular_solid_surface_area_l1020_102026


namespace NUMINAMATH_CALUDE_boxes_per_case_l1020_102097

theorem boxes_per_case (total_boxes : ℕ) (num_cases : ℕ) (h1 : total_boxes = 24) (h2 : num_cases = 3) :
  total_boxes / num_cases = 8 := by
sorry

end NUMINAMATH_CALUDE_boxes_per_case_l1020_102097


namespace NUMINAMATH_CALUDE_magnitude_of_vector_sum_l1020_102050

/-- The magnitude of the sum of two vectors given specific conditions -/
theorem magnitude_of_vector_sum (a b : ℝ × ℝ) :
  a = (1, 0) →
  ‖b‖ = Real.sqrt 2 →
  a • b = 1 →
  ‖2 • a + b‖ = Real.sqrt 10 := by
  sorry

end NUMINAMATH_CALUDE_magnitude_of_vector_sum_l1020_102050


namespace NUMINAMATH_CALUDE_skew_lines_distance_l1020_102095

/-- Given two skew lines a and b forming an angle θ, with their common perpendicular AA' of length d,
    and points E on a and F on b such that A'E = m and AF = n, the distance EF is given by
    √(d² + m² + n² ± 2mn cos θ). -/
theorem skew_lines_distance (θ d m n : ℝ) : ∃ (EF : ℝ), 
  EF = Real.sqrt (d^2 + m^2 + n^2 + 2*m*n*(Real.cos θ)) ∨
  EF = Real.sqrt (d^2 + m^2 + n^2 - 2*m*n*(Real.cos θ)) :=
by sorry


end NUMINAMATH_CALUDE_skew_lines_distance_l1020_102095


namespace NUMINAMATH_CALUDE_incorrect_statement_l1020_102010

-- Define the types for planes and lines
variable (Plane Line : Type)

-- Define the parallel relation for lines and planes
variable (parallelLine : Line → Line → Prop)
variable (parallelLinePlane : Line → Plane → Prop)

-- Define the intersection of two planes
variable (intersect : Plane → Plane → Line)

-- Define the containment relation for lines and planes
variable (contains : Plane → Line → Prop)

-- State the theorem
theorem incorrect_statement 
  (α β : Plane) (m n : Line) : 
  ¬(∀ α β m n, parallelLinePlane m α ∧ intersect α β = n → parallelLine m n) := by
  sorry

end NUMINAMATH_CALUDE_incorrect_statement_l1020_102010


namespace NUMINAMATH_CALUDE_circumcenter_property_l1020_102066

/-- A point in 3D space -/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- A plane in 3D space -/
structure Plane where
  a : ℝ
  b : ℝ
  c : ℝ
  d : ℝ

/-- A triangle in 3D space -/
structure Triangle3D where
  A : Point3D
  B : Point3D
  C : Point3D

/-- Check if a point is outside a plane -/
def isOutside (p : Point3D) (plane : Plane) : Prop := sorry

/-- Check if a line is perpendicular to a plane -/
def isPerpendicular (p1 p2 : Point3D) (plane : Plane) : Prop := sorry

/-- Check if a point is the foot of the perpendicular from another point to a plane -/
def isFootOfPerpendicular (o p : Point3D) (plane : Plane) : Prop := sorry

/-- Check if three distances are equal -/
def areDistancesEqual (p a b c : Point3D) : Prop := sorry

/-- Check if a point is the circumcenter of a triangle -/
def isCircumcenter (o : Point3D) (t : Triangle3D) : Prop := sorry

theorem circumcenter_property (P O : Point3D) (ABC : Triangle3D) (plane : Plane) :
  isOutside P plane →
  isPerpendicular P O plane →
  isFootOfPerpendicular O P plane →
  areDistancesEqual P ABC.A ABC.B ABC.C →
  isCircumcenter O ABC := by sorry

end NUMINAMATH_CALUDE_circumcenter_property_l1020_102066


namespace NUMINAMATH_CALUDE_parallelogram_base_length_l1020_102083

/-- Given a parallelogram with area 288 square centimeters and height 16 cm, 
    prove that its base length is 18 cm. -/
theorem parallelogram_base_length 
  (area : ℝ) 
  (height : ℝ) 
  (h1 : area = 288) 
  (h2 : height = 16) : 
  area / height = 18 := by
  sorry

end NUMINAMATH_CALUDE_parallelogram_base_length_l1020_102083


namespace NUMINAMATH_CALUDE_each_person_receives_eight_doughnuts_l1020_102086

/-- The number of doughnuts each person receives when Samuel and Cathy share their doughnuts. -/
def doughnuts_per_person : ℕ :=
  let samuel_doughnuts : ℕ := 2 * 12
  let cathy_doughnuts : ℕ := 4 * 12
  let total_doughnuts : ℕ := samuel_doughnuts + cathy_doughnuts
  let total_people : ℕ := 10
  let dieting_friends : ℕ := 1
  let sharing_people : ℕ := total_people - dieting_friends
  total_doughnuts / sharing_people

/-- Theorem stating that each person receives 8 doughnuts. -/
theorem each_person_receives_eight_doughnuts : doughnuts_per_person = 8 := by
  sorry

end NUMINAMATH_CALUDE_each_person_receives_eight_doughnuts_l1020_102086


namespace NUMINAMATH_CALUDE_sequence_element_proof_l1020_102098

theorem sequence_element_proof :
  (∃ n : ℕ+, n^2 + 2*n = 63) ∧
  (¬ ∃ n : ℕ+, n^2 + 2*n = 10) ∧
  (¬ ∃ n : ℕ+, n^2 + 2*n = 18) ∧
  (¬ ∃ n : ℕ+, n^2 + 2*n = 26) :=
by sorry

end NUMINAMATH_CALUDE_sequence_element_proof_l1020_102098


namespace NUMINAMATH_CALUDE_mark_bill_calculation_l1020_102024

def original_bill : ℝ := 500
def first_late_charge_rate : ℝ := 0.02
def second_late_charge_rate : ℝ := 0.03

def final_amount : ℝ := original_bill * (1 + first_late_charge_rate) * (1 + second_late_charge_rate)

theorem mark_bill_calculation : final_amount = 525.30 := by
  sorry

end NUMINAMATH_CALUDE_mark_bill_calculation_l1020_102024


namespace NUMINAMATH_CALUDE_perpendicular_lines_l1020_102018

-- Define the slopes of the lines
def m1 : ℚ := 3 / 4
def m2 : ℚ := -3 / 4
def m3 : ℚ := -3 / 4
def m4 : ℚ := -4 / 3
def m5 : ℚ := 12 / 5

-- Define the condition for perpendicularity
def are_perpendicular (m1 m2 : ℚ) : Prop := m1 * m2 = -1

-- Theorem statement
theorem perpendicular_lines :
  (are_perpendicular m1 m4) ∧
  (¬ are_perpendicular m1 m2) ∧
  (¬ are_perpendicular m1 m3) ∧
  (¬ are_perpendicular m1 m5) ∧
  (¬ are_perpendicular m2 m3) ∧
  (¬ are_perpendicular m2 m4) ∧
  (¬ are_perpendicular m2 m5) ∧
  (¬ are_perpendicular m3 m4) ∧
  (¬ are_perpendicular m3 m5) ∧
  (¬ are_perpendicular m4 m5) :=
by sorry

end NUMINAMATH_CALUDE_perpendicular_lines_l1020_102018


namespace NUMINAMATH_CALUDE_line_intercepts_sum_l1020_102046

/-- Given a line described by the equation y - 3 = -3(x + 2), 
    the sum of its x-intercept and y-intercept is -4 -/
theorem line_intercepts_sum : 
  ∀ (x y : ℝ), y - 3 = -3*(x + 2) → 
  ∃ (x_int y_int : ℝ), 
    (0 - 3 = -3*(x_int + 2)) ∧ 
    (y_int - 3 = -3*(0 + 2)) ∧ 
    (x_int + y_int = -4) := by
  sorry

end NUMINAMATH_CALUDE_line_intercepts_sum_l1020_102046


namespace NUMINAMATH_CALUDE_drug_price_reduction_l1020_102007

/-- Proves that given an initial price of 100 yuan and a final price of 81 yuan
    after two equal percentage reductions, the average percentage reduction each time is 10% -/
theorem drug_price_reduction (initial_price : ℝ) (final_price : ℝ) (reduction_percentage : ℝ) :
  initial_price = 100 →
  final_price = 81 →
  final_price = initial_price * (1 - reduction_percentage)^2 →
  reduction_percentage = 0.1 := by
sorry


end NUMINAMATH_CALUDE_drug_price_reduction_l1020_102007


namespace NUMINAMATH_CALUDE_installation_rate_one_each_possible_solutions_l1020_102060

/-- Represents the number of air conditioners installed by different worker combinations -/
structure InstallationRate where
  skilled : ℕ → ℕ
  new : ℕ → ℕ

/-- Represents the total number of air conditioners to be installed -/
def total_ac : ℕ := 500

/-- Represents the number of days to complete the installation -/
def days : ℕ := 20

/-- Given conditions on installation rates -/
axiom condition1 {r : InstallationRate} : r.skilled 1 + r.new 3 = 11
axiom condition2 {r : InstallationRate} : r.skilled 2 = r.new 5

/-- Theorem stating the installation rate of 1 skilled worker and 1 new worker -/
theorem installation_rate_one_each (r : InstallationRate) : 
  r.skilled 1 + r.new 1 = 7 := by sorry

/-- Theorem stating the possible solutions for m skilled workers and n new workers -/
theorem possible_solutions (m n : ℕ) : 
  (m ≠ 0 ∧ n ≠ 0 ∧ 5 * m + 2 * n = 25) ↔ (m = 1 ∧ n = 10) ∨ (m = 3 ∧ n = 5) := by sorry

end NUMINAMATH_CALUDE_installation_rate_one_each_possible_solutions_l1020_102060


namespace NUMINAMATH_CALUDE_product_of_terms_l1020_102022

/-- An arithmetic sequence -/
def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

/-- A geometric sequence -/
def geometric_sequence (b : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, r ≠ 0 ∧ ∀ n : ℕ, b (n + 1) = r * b n

/-- The main theorem -/
theorem product_of_terms (a b : ℕ → ℝ) :
  arithmetic_sequence a →
  geometric_sequence b →
  3 * a 1 - (a 8)^2 + 3 * a 15 = 0 →
  a 8 = b 10 →
  b 3 * b 17 = 36 := by
  sorry

end NUMINAMATH_CALUDE_product_of_terms_l1020_102022


namespace NUMINAMATH_CALUDE_cube_planes_divide_space_into_27_parts_l1020_102037

/-- Represents a cube in 3D space -/
structure Cube where
  -- Add necessary fields here
  mk :: -- Constructor

/-- Represents a plane in 3D space -/
structure Plane where
  -- Add necessary fields here
  mk :: -- Constructor

/-- Function to generate planes through each face of a cube -/
def planes_through_cube_faces (c : Cube) : List Plane :=
  sorry

/-- Function to count the number of parts the space is divided into by the planes -/
def count_divided_parts (planes : List Plane) : Nat :=
  sorry

/-- Theorem stating that planes through each face of a cube divide space into 27 parts -/
theorem cube_planes_divide_space_into_27_parts (c : Cube) :
  count_divided_parts (planes_through_cube_faces c) = 27 := by
  sorry

end NUMINAMATH_CALUDE_cube_planes_divide_space_into_27_parts_l1020_102037


namespace NUMINAMATH_CALUDE_misha_older_than_tanya_l1020_102039

/-- Represents a person's age in years and months -/
structure Age where
  years : ℕ
  months : ℕ
  inv : months < 12

/-- Compares two Ages -/
def Age.lt (a b : Age) : Prop :=
  a.years < b.years ∨ (a.years = b.years ∧ a.months < b.months)

/-- Adds months to an Age -/
def Age.addMonths (a : Age) (m : ℕ) : Age :=
  { years := a.years + (a.months + m) / 12,
    months := (a.months + m) % 12,
    inv := by sorry }

/-- Subtracts months from an Age -/
def Age.subMonths (a : Age) (m : ℕ) : Age :=
  { years := a.years - (m + 11) / 12,
    months := (a.months + 12 - (m % 12)) % 12,
    inv := by sorry }

theorem misha_older_than_tanya (tanya_past misha_future : Age) :
  tanya_past.addMonths 19 = tanya_past.addMonths 19 →
  misha_future.subMonths 16 = misha_future.subMonths 16 →
  tanya_past.years = 16 →
  misha_future.years = 19 →
  Age.lt (tanya_past.addMonths 19) (misha_future.subMonths 16) := by
  sorry

end NUMINAMATH_CALUDE_misha_older_than_tanya_l1020_102039


namespace NUMINAMATH_CALUDE_rice_bag_problem_l1020_102034

theorem rice_bag_problem (initial_stock : ℕ) : 
  initial_stock - 23 + 132 = 164 → initial_stock = 55 := by
  sorry

end NUMINAMATH_CALUDE_rice_bag_problem_l1020_102034


namespace NUMINAMATH_CALUDE_fourth_sample_number_l1020_102027

/-- Systematic sampling function -/
def systematicSample (totalStudents : ℕ) (sampleSize : ℕ) (firstSample : ℕ) : ℕ → ℕ :=
  fun i => firstSample + i * (totalStudents / sampleSize)

theorem fourth_sample_number
  (totalStudents : ℕ)
  (sampleSize : ℕ)
  (h_total : totalStudents = 52)
  (h_sample : sampleSize = 4)
  (h_first : systematicSample totalStudents sampleSize 7 0 = 7)
  (h_second : systematicSample totalStudents sampleSize 7 1 = 33)
  (h_third : systematicSample totalStudents sampleSize 7 2 = 46) :
  systematicSample totalStudents sampleSize 7 3 = 20 :=
sorry

end NUMINAMATH_CALUDE_fourth_sample_number_l1020_102027


namespace NUMINAMATH_CALUDE_intersection_singleton_implies_k_negative_one_intersection_and_union_when_k_is_two_l1020_102006

def M : Set ℝ := {x | -1 ≤ x ∧ x ≤ 5}
def N (k : ℝ) : Set ℝ := {x | x - k ≤ 0}

theorem intersection_singleton_implies_k_negative_one :
  (∃! x, x ∈ M ∩ N k) → k = -1 :=
by sorry

theorem intersection_and_union_when_k_is_two :
  M ∩ N 2 = {x | -1 ≤ x ∧ x ≤ 2} ∧ M ∪ N 2 = {x | x ≤ 5} :=
by sorry

end NUMINAMATH_CALUDE_intersection_singleton_implies_k_negative_one_intersection_and_union_when_k_is_two_l1020_102006


namespace NUMINAMATH_CALUDE_f_properties_l1020_102000

-- Define the function f
def f (x : ℝ) (a b c : ℝ) : ℝ := x^3 - 6*x^2 + 9*x - a*b*c

-- State the theorem
theorem f_properties (a b c : ℝ) (h1 : a < b) (h2 : b < c)
  (h3 : f a a b c = 0) (h4 : f b a b c = 0) (h5 : f c a b c = 0) :
  (f 0 a b c) * (f 1 a b c) < 0 ∧ (f 0 a b c) * (f 3 a b c) > 0 := by
  sorry

end NUMINAMATH_CALUDE_f_properties_l1020_102000


namespace NUMINAMATH_CALUDE_blue_bicycle_selection_count_l1020_102001

/-- The number of ways to select at least two blue shared bicycles -/
def select_blue_bicycles : ℕ :=
  (Nat.choose 4 2 * Nat.choose 6 2) + (Nat.choose 4 3 * Nat.choose 6 1) + Nat.choose 4 4

/-- Theorem stating that the number of ways to select at least two blue shared bicycles is 115 -/
theorem blue_bicycle_selection_count :
  select_blue_bicycles = 115 := by sorry

end NUMINAMATH_CALUDE_blue_bicycle_selection_count_l1020_102001


namespace NUMINAMATH_CALUDE_initial_weight_of_beef_l1020_102055

/-- The weight of a side of beef after five stages of processing --/
def final_weight (W : ℝ) : ℝ :=
  ((((W * 0.8) * 0.7) * 0.75) - 15) * 0.88

/-- Theorem stating the initial weight of the side of beef --/
theorem initial_weight_of_beef :
  ∃ W : ℝ, W > 0 ∧ final_weight W = 570 ∧ W = 1578 := by
  sorry

end NUMINAMATH_CALUDE_initial_weight_of_beef_l1020_102055


namespace NUMINAMATH_CALUDE_coefficient_x_squared_in_expansion_l1020_102023

theorem coefficient_x_squared_in_expansion : 
  let n : ℕ := 7
  let expansion := (1 - X : Polynomial ℚ) ^ n
  (expansion.coeff 2 : ℚ) = 21 := by
  sorry

end NUMINAMATH_CALUDE_coefficient_x_squared_in_expansion_l1020_102023


namespace NUMINAMATH_CALUDE_student_committee_candidates_l1020_102040

theorem student_committee_candidates : 
  ∃ n : ℕ, 
    n > 0 ∧ 
    n * (n - 1) = 132 ∧ 
    n = 12 := by
  sorry

end NUMINAMATH_CALUDE_student_committee_candidates_l1020_102040


namespace NUMINAMATH_CALUDE_tea_table_probability_l1020_102052

-- Define the number of people and tables
def total_people : ℕ := 6
def total_tables : ℕ := 3
def people_per_table : ℕ := 2
def coffee_drinkers : ℕ := 3
def tea_drinkers : ℕ := 3

-- Define the probability function
def probability_both_tea_drinkers : ℚ := 3 / 5

-- State the theorem
theorem tea_table_probability :
  probability_both_tea_drinkers = 3 / 5 := by sorry

end NUMINAMATH_CALUDE_tea_table_probability_l1020_102052


namespace NUMINAMATH_CALUDE_min_value_of_f_l1020_102015

def f (x : ℝ) (m : ℝ) := 2 * x^3 - 6 * x^2 + m

theorem min_value_of_f (m : ℝ) :
  (∃ x ∈ Set.Icc (-2 : ℝ) 2, ∀ y ∈ Set.Icc (-2 : ℝ) 2, f y m ≤ f x m) ∧
  (∃ x ∈ Set.Icc (-2 : ℝ) 2, f x m = 3) ∧
  (∀ x ∈ Set.Icc (-2 : ℝ) 2, f x m ≤ 3) →
  (∃ x ∈ Set.Icc (-2 : ℝ) 2, ∀ y ∈ Set.Icc (-2 : ℝ) 2, f x m ≤ f y m) ∧
  (∃ x ∈ Set.Icc (-2 : ℝ) 2, f x m = -37) :=
by sorry

end NUMINAMATH_CALUDE_min_value_of_f_l1020_102015


namespace NUMINAMATH_CALUDE_nonagon_non_adjacent_segments_l1020_102072

theorem nonagon_non_adjacent_segments (n : ℕ) (h : n = 9) : 
  (n * (n - 1)) / 2 - n = 27 := by
  sorry

end NUMINAMATH_CALUDE_nonagon_non_adjacent_segments_l1020_102072


namespace NUMINAMATH_CALUDE_lydia_almonds_l1020_102003

theorem lydia_almonds (lydia_almonds max_almonds : ℕ) : 
  lydia_almonds = max_almonds + 8 →
  max_almonds = lydia_almonds / 3 →
  lydia_almonds = 12 := by
sorry

end NUMINAMATH_CALUDE_lydia_almonds_l1020_102003


namespace NUMINAMATH_CALUDE_university_admission_problem_l1020_102074

theorem university_admission_problem :
  let n_universities : ℕ := 8
  let n_selected_universities : ℕ := 2
  let n_students : ℕ := 3
  
  (Nat.choose n_universities n_selected_universities) * (2 ^ n_students) = 224 :=
by sorry

end NUMINAMATH_CALUDE_university_admission_problem_l1020_102074


namespace NUMINAMATH_CALUDE_impossible_all_defective_l1020_102049

/-- Given 10 products with 2 defective ones, the probability of selecting 3 defective products
    when randomly choosing 3 is zero. -/
theorem impossible_all_defective (total : Nat) (defective : Nat) (selected : Nat)
    (h1 : total = 10)
    (h2 : defective = 2)
    (h3 : selected = 3) :
  Nat.choose defective selected / Nat.choose total selected = 0 := by
  sorry

end NUMINAMATH_CALUDE_impossible_all_defective_l1020_102049


namespace NUMINAMATH_CALUDE_ages_relationship_l1020_102043

/-- Given the ages of Katherine (K), Mel (M), and Lexi (L), with the relationships
    M = K - 3 and L = M + 2, prove that when K = 60, M = 57 and L = 59. -/
theorem ages_relationship (K M L : ℕ) 
    (h1 : M = K - 3) 
    (h2 : L = M + 2) 
    (h3 : K = 60) : 
  M = 57 ∧ L = 59 := by
sorry

end NUMINAMATH_CALUDE_ages_relationship_l1020_102043


namespace NUMINAMATH_CALUDE_arithmetic_sequence_problem_l1020_102045

/-- An arithmetic sequence with positive terms -/
def ArithmeticSequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d ∧ a n > 0

/-- The property that three terms form a geometric sequence -/
def GeometricSequence (x y z : ℝ) : Prop :=
  y * y = x * z

theorem arithmetic_sequence_problem (a : ℕ → ℝ) 
    (h_arith : ArithmeticSequence a)
    (h_sum : a 2 + a 3 + a 4 = 15)
    (h_geom : GeometricSequence (a 1 + 2) (a 3 + 4) (a 6 + 16)) :
    a 10 = 19 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_problem_l1020_102045


namespace NUMINAMATH_CALUDE_connie_marbles_problem_l1020_102030

/-- Proves that Connie started with 143 marbles given the conditions of the problem -/
theorem connie_marbles_problem :
  ∀ (initial : ℕ),
  initial - 73 = 70 →
  initial = 143 :=
by
  sorry

end NUMINAMATH_CALUDE_connie_marbles_problem_l1020_102030


namespace NUMINAMATH_CALUDE_square_root_of_four_l1020_102068

theorem square_root_of_four :
  ∀ x : ℝ, x^2 = 4 ↔ x = 2 ∨ x = -2 :=
sorry

end NUMINAMATH_CALUDE_square_root_of_four_l1020_102068


namespace NUMINAMATH_CALUDE_parabola_transformation_sum_l1020_102081

/-- Represents a quadratic function of the form ax^2 + bx + c -/
structure QuadraticFunction where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Represents a transformation of a quadratic function -/
inductive Transformation
  | Reflect
  | Translate (d : ℝ)

/-- Applies a transformation to a quadratic function -/
def applyTransformation (q : QuadraticFunction) (t : Transformation) : QuadraticFunction :=
  match t with
  | Transformation.Reflect => { a := q.a, b := -q.b, c := q.c }
  | Transformation.Translate d => { a := q.a, b := q.b - 2 * q.a * d, c := q.a * d^2 - q.b * d + q.c }

/-- Sums two quadratic functions -/
def sumQuadraticFunctions (q1 q2 : QuadraticFunction) : QuadraticFunction :=
  { a := q1.a + q2.a, b := q1.b + q2.b, c := q1.c + q2.c }

theorem parabola_transformation_sum (q : QuadraticFunction) :
  let f := applyTransformation (applyTransformation q Transformation.Reflect) (Transformation.Translate (-7))
  let g := applyTransformation q (Transformation.Translate 3)
  let sum := sumQuadraticFunctions f g
  sum.a = 2 * q.a ∧ sum.b = 8 * q.a - 2 * q.b ∧ sum.c = 58 * q.a - 4 * q.b + 2 * q.c :=
by sorry

end NUMINAMATH_CALUDE_parabola_transformation_sum_l1020_102081


namespace NUMINAMATH_CALUDE_bread_price_calculation_bread_price_proof_l1020_102057

theorem bread_price_calculation (initial_price : ℝ) 
  (thursday_increase : ℝ) (saturday_discount : ℝ) : ℝ :=
  let thursday_price := initial_price * (1 + thursday_increase)
  let saturday_price := thursday_price * (1 - saturday_discount)
  saturday_price

theorem bread_price_proof :
  bread_price_calculation 50 0.2 0.15 = 51 := by
  sorry

end NUMINAMATH_CALUDE_bread_price_calculation_bread_price_proof_l1020_102057


namespace NUMINAMATH_CALUDE_tangent_and_mean_value_theorem_l1020_102011

noncomputable section

/-- The function f(x) = x^2 + a(x + ln x) -/
def f (a : ℝ) (x : ℝ) : ℝ := x^2 + a * (x + Real.log x)

/-- The derivative of f with respect to x -/
def f_deriv (a : ℝ) (x : ℝ) : ℝ := 2*x + a * (1 + 1/x)

theorem tangent_and_mean_value_theorem (a : ℝ) :
  (∃ x₀ : ℝ, x₀ > 0 ∧ f_deriv a x₀ = 2*(a+1) ∧ f a x₀ = (a+1)*(2*x₀-1) - a - 1) ∧
  (∃ ξ : ℝ, 1 < ξ ∧ ξ < Real.exp 1 ∧ f_deriv a ξ = (f a (Real.exp 1) - f a 1) / (Real.exp 1 - 1)) :=
sorry

end

end NUMINAMATH_CALUDE_tangent_and_mean_value_theorem_l1020_102011


namespace NUMINAMATH_CALUDE_replacement_cost_theorem_l1020_102069

/-- The cost to replace all cardio machines in a chain of gyms -/
def total_replacement_cost (num_gyms : ℕ) (bikes_per_gym treadmills_per_gym ellipticals_per_gym : ℕ)
  (bike_cost : ℝ) : ℝ :=
  let treadmill_cost := 1.5 * bike_cost
  let elliptical_cost := 2 * treadmill_cost
  let total_bikes := num_gyms * bikes_per_gym
  let total_treadmills := num_gyms * treadmills_per_gym
  let total_ellipticals := num_gyms * ellipticals_per_gym
  total_bikes * bike_cost + total_treadmills * treadmill_cost + total_ellipticals * elliptical_cost

/-- Theorem stating the total cost to replace all cardio machines -/
theorem replacement_cost_theorem :
  total_replacement_cost 20 10 5 5 700 = 455000 := by
  sorry


end NUMINAMATH_CALUDE_replacement_cost_theorem_l1020_102069


namespace NUMINAMATH_CALUDE_units_digit_of_7_pow_2023_l1020_102017

/-- The units digit of a natural number -/
def unitsDigit (n : ℕ) : ℕ := n % 10

/-- The result of raising 7 to the power of 2023 -/
def power : ℕ := 7^2023

/-- Theorem: The units digit of 7^2023 is 3 -/
theorem units_digit_of_7_pow_2023 : unitsDigit power = 3 := by sorry

end NUMINAMATH_CALUDE_units_digit_of_7_pow_2023_l1020_102017


namespace NUMINAMATH_CALUDE_pirate_captain_age_l1020_102028

/-- Represents a health insurance card number -/
structure HealthInsuranceCard where
  main_number : Nat
  control_number : Nat
  h_main_digits : main_number < 10000000000000
  h_control_digits : control_number < 100

/-- Checks if a health insurance card number is valid -/
def is_valid_card (card : HealthInsuranceCard) : Prop :=
  (card.main_number + card.control_number) % 97 = 0

/-- Calculates the age based on birth year and current year -/
def calculate_age (birth_year : Nat) (current_year : Nat) : Nat :=
  current_year - birth_year

theorem pirate_captain_age :
  ∃ (card : HealthInsuranceCard),
    card.control_number = 67 ∧
    ∃ (x : Nat), x < 10 ∧ card.main_number = 1000000000000 * (10 + x) + 1271153044 ∧
    is_valid_card card ∧
    calculate_age (1900 + (10 + x)) 2011 = 65 := by
  sorry

end NUMINAMATH_CALUDE_pirate_captain_age_l1020_102028


namespace NUMINAMATH_CALUDE_downstream_distance_is_36_l1020_102076

/-- Represents the swimming scenario --/
structure SwimmingScenario where
  still_water_speed : ℝ
  upstream_distance : ℝ
  swim_time : ℝ

/-- Calculates the downstream distance given a swimming scenario --/
def downstream_distance (s : SwimmingScenario) : ℝ :=
  sorry

/-- Theorem stating that the downstream distance is 36 km for the given scenario --/
theorem downstream_distance_is_36 (s : SwimmingScenario) 
  (h1 : s.still_water_speed = 9)
  (h2 : s.upstream_distance = 18)
  (h3 : s.swim_time = 3) : 
  downstream_distance s = 36 :=
sorry

end NUMINAMATH_CALUDE_downstream_distance_is_36_l1020_102076


namespace NUMINAMATH_CALUDE_triangle_ratio_equality_l1020_102014

/-- Given a triangle with side length a, height h_a corresponding to side a,
    inradius r, and semiperimeter p, prove that (2p / a) = (h_a / r) -/
theorem triangle_ratio_equality (a h_a r p : ℝ) (h_positive : a > 0 ∧ h_a > 0 ∧ r > 0 ∧ p > 0)
  (h_area_inradius : p * r = (1/2) * a * h_a) : 
  (2 * p / a) = (h_a / r) := by
  sorry

end NUMINAMATH_CALUDE_triangle_ratio_equality_l1020_102014


namespace NUMINAMATH_CALUDE_no_double_application_increment_l1020_102080

theorem no_double_application_increment :
  ¬∃ f : ℤ → ℤ, ∀ x : ℤ, f (f x) = x + 1 := by sorry

end NUMINAMATH_CALUDE_no_double_application_increment_l1020_102080


namespace NUMINAMATH_CALUDE_arithmetic_sequence_sum_l1020_102004

/-- An arithmetic sequence is a sequence where the difference between
    any two consecutive terms is constant. -/
def is_arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

/-- Given an arithmetic sequence a where a_1 + a_5 = 6,
    prove that the sum of the first five terms is 15. -/
theorem arithmetic_sequence_sum (a : ℕ → ℝ) 
  (h_arith : is_arithmetic_sequence a) 
  (h_sum : a 1 + a 5 = 6) : 
  a 1 + a 2 + a 3 + a 4 + a 5 = 15 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_sum_l1020_102004


namespace NUMINAMATH_CALUDE_sequence_inequality_range_l1020_102099

/-- Given a sequence a_n with sum S_n, prove the range of t -/
theorem sequence_inequality_range (a : ℕ → ℝ) (S : ℕ → ℝ) (t : ℝ) : 
  (∀ n : ℕ, 2 * S n = (n + 1) * a n) →  -- Condition: 2S_n = (n+1)a_n
  (a 1 = 1) →  -- Condition: a_1 = 1
  (∀ n : ℕ, n ≥ 2 → a n = n) →  -- Derived from conditions
  (t > 0) →  -- Condition: t > 0
  (∃! n : ℕ, n > 0 ∧ a n^2 - t * a n - 2 * t^2 < 0) →  -- Condition: unique positive n satisfying inequality
  t ∈ Set.Ioo (1/2 : ℝ) 1 :=  -- Conclusion: t is in the open interval (1/2, 1]
sorry

end NUMINAMATH_CALUDE_sequence_inequality_range_l1020_102099


namespace NUMINAMATH_CALUDE_seed_mixture_problem_l1020_102025

/-- Represents the composition of a seed mixture -/
structure SeedMixture where
  ryegrass : ℝ
  bluegrass : ℝ
  fescue : ℝ

/-- The problem statement -/
theorem seed_mixture_problem (X Y : SeedMixture) (mixture_weight : ℝ) :
  X.ryegrass = 40 →
  Y.ryegrass = 25 →
  Y.fescue = 75 →
  X.ryegrass + X.bluegrass + X.fescue = 100 →
  Y.ryegrass + Y.bluegrass + Y.fescue = 100 →
  mixture_weight * 30 / 100 = X.ryegrass * (mixture_weight * 100 / 3 / 100) + Y.ryegrass * (mixture_weight * 200 / 3 / 100) →
  X.bluegrass = 60 := by
  sorry


end NUMINAMATH_CALUDE_seed_mixture_problem_l1020_102025


namespace NUMINAMATH_CALUDE_time_relationship_l1020_102038

theorem time_relationship (x : ℝ) (h : x > 0) :
  let time_left := (2 / 6) * x
  let total_time := x + time_left
  total_time = (4 / 3) * x :=
by sorry

end NUMINAMATH_CALUDE_time_relationship_l1020_102038


namespace NUMINAMATH_CALUDE_minimum_cats_with_stripes_and_black_ear_l1020_102056

theorem minimum_cats_with_stripes_and_black_ear (total_cats : ℕ) (mice_catchers : ℕ) 
  (striped_cats : ℕ) (black_ear_cats : ℕ) 
  (h1 : total_cats = 66) (h2 : mice_catchers = 21) 
  (h3 : striped_cats = 32) (h4 : black_ear_cats = 27) : 
  ∃ (x : ℕ), x = 14 ∧ 
  x ≤ striped_cats ∧ 
  x ≤ black_ear_cats ∧
  x ≤ total_cats - mice_catchers ∧
  ∀ (y : ℕ), y < x → 
    y > striped_cats + black_ear_cats - (total_cats - mice_catchers) := by
  sorry

end NUMINAMATH_CALUDE_minimum_cats_with_stripes_and_black_ear_l1020_102056


namespace NUMINAMATH_CALUDE_age_problem_l1020_102096

/-- Given three people a, b, and c, where:
    - a is two years older than b
    - b is twice as old as c
    - The sum of their ages is 72
    Prove that b is 28 years old. -/
theorem age_problem (a b c : ℕ) 
  (h1 : a = b + 2)
  (h2 : b = 2 * c)
  (h3 : a + b + c = 72) : 
  b = 28 := by
  sorry

end NUMINAMATH_CALUDE_age_problem_l1020_102096


namespace NUMINAMATH_CALUDE_negation_of_all_exponential_are_monotonic_l1020_102051

-- Define exponential function
def ExponentialFunction (f : ℝ → ℝ) : Prop := sorry

-- Define monotonic function
def MonotonicFunction (f : ℝ → ℝ) : Prop := sorry

-- Theorem statement
theorem negation_of_all_exponential_are_monotonic :
  (¬ ∀ f : ℝ → ℝ, ExponentialFunction f → MonotonicFunction f) ↔
  (∃ f : ℝ → ℝ, ExponentialFunction f ∧ ¬MonotonicFunction f) :=
by sorry

end NUMINAMATH_CALUDE_negation_of_all_exponential_are_monotonic_l1020_102051


namespace NUMINAMATH_CALUDE_simplify_expression_l1020_102082

theorem simplify_expression (a b c : ℝ) (h : b^2 = c^2) :
  -|b| - |a-b| + |a-c| - |b+c| = - |a-b| + |a-c| - |b+c| := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l1020_102082


namespace NUMINAMATH_CALUDE_birdhouse_charge_for_two_l1020_102016

/-- The cost of a birdhouse for Denver -/
def birdhouse_cost (wood_pieces : ℕ) (wood_price paint_cost labor_cost : ℚ) : ℚ :=
  wood_pieces * wood_price + paint_cost + labor_cost

/-- The selling price of a birdhouse -/
def birdhouse_price (cost profit : ℚ) : ℚ :=
  cost + profit

/-- The total charge for multiple birdhouses -/
def total_charge (price : ℚ) (quantity : ℕ) : ℚ :=
  price * quantity

theorem birdhouse_charge_for_two :
  let wood_pieces : ℕ := 7
  let wood_price : ℚ := 3/2  -- $1.50
  let paint_cost : ℚ := 3
  let labor_cost : ℚ := 9/2  -- $4.50
  let profit : ℚ := 11/2  -- $5.50
  let cost := birdhouse_cost wood_pieces wood_price paint_cost labor_cost
  let price := birdhouse_price cost profit
  let quantity : ℕ := 2
  total_charge price quantity = 47
  := by sorry

end NUMINAMATH_CALUDE_birdhouse_charge_for_two_l1020_102016


namespace NUMINAMATH_CALUDE_horse_journey_l1020_102071

/-- Given a geometric sequence with common ratio 1/2 and sum of first 7 terms equal to 700,
    the sum of the first 14 terms is 22575/32 -/
theorem horse_journey (a : ℝ) (S : ℕ → ℝ) : 
  (∀ n, S (n + 1) = S n + a * (1/2)^n) → 
  S 0 = 0 →
  S 7 = 700 →
  S 14 = 22575/32 := by
sorry

end NUMINAMATH_CALUDE_horse_journey_l1020_102071


namespace NUMINAMATH_CALUDE_probability_multiple_2_3_7_l1020_102012

/-- The number of integers from 1 to n that are divisible by at least one of a, b, or c -/
def countMultiples (n : ℕ) (a b c : ℕ) : ℕ :=
  (n / a + n / b + n / c) - (n / lcm a b + n / lcm a c + n / lcm b c) + n / lcm a (lcm b c)

/-- The probability of selecting a multiple of 2, 3, or 7 from the first 150 positive integers -/
theorem probability_multiple_2_3_7 : 
  countMultiples 150 2 3 7 = 107 := by sorry

end NUMINAMATH_CALUDE_probability_multiple_2_3_7_l1020_102012


namespace NUMINAMATH_CALUDE_ball_selection_count_l1020_102042

/-- Represents the number of balls of each color -/
def ballsPerColor : ℕ := 7

/-- Represents the number of colors -/
def numberOfColors : ℕ := 3

/-- Represents the total number of balls -/
def totalBalls : ℕ := ballsPerColor * numberOfColors

/-- Checks if three numbers are non-consecutive -/
def areNonConsecutive (a b c : ℕ) : Prop :=
  (a + 1 ≠ b ∧ b + 1 ≠ c) ∧ (b + 1 ≠ a ∧ c + 1 ≠ b) ∧ (c + 1 ≠ a ∧ a + 1 ≠ c)

/-- Counts the number of ways to select 3 non-consecutive numbers from 1 to 7 -/
def nonConsecutiveSelections : ℕ := 35

/-- The main theorem to be proved -/
theorem ball_selection_count :
  (∃ (f : Fin totalBalls → ℕ × Fin numberOfColors),
    (∀ i j, i ≠ j → f i ≠ f j) ∧
    (∀ i, (f i).1 ∈ Finset.range ballsPerColor) ∧
    (∃ a b c, a ≠ b ∧ b ≠ c ∧ a ≠ c ∧
      areNonConsecutive (f a).1 (f b).1 (f c).1 ∧
      (f a).2 ≠ (f b).2 ∧ (f b).2 ≠ (f c).2 ∧ (f a).2 ≠ (f c).2)) →
  nonConsecutiveSelections * numberOfColors = 60 :=
sorry

end NUMINAMATH_CALUDE_ball_selection_count_l1020_102042


namespace NUMINAMATH_CALUDE_sum_of_solutions_quadratic_l1020_102085

theorem sum_of_solutions_quadratic (x : ℝ) : 
  let a : ℝ := -24
  let b : ℝ := 72
  let c : ℝ := -120
  let sum_of_roots := -b / a
  (a * x^2 + b * x + c = 0) → sum_of_roots = 3 :=
by sorry

end NUMINAMATH_CALUDE_sum_of_solutions_quadratic_l1020_102085


namespace NUMINAMATH_CALUDE_largest_integer_difference_in_triangle_l1020_102063

theorem largest_integer_difference_in_triangle (n : ℕ) (hn : n ≥ 4) :
  (∃ k : ℕ, k > 0 ∧
    (∀ k' : ℕ, k' > k →
      ¬∃ a b c : ℕ, 1 ≤ a ∧ a < b ∧ b < c ∧ c ≤ n ∧
        c - b ≥ k' ∧ b - a ≥ k' ∧ a + b ≥ c + 1) ∧
    (∃ a b c : ℕ, 1 ≤ a ∧ a < b ∧ b < c ∧ c ≤ n ∧
      c - b ≥ k ∧ b - a ≥ k ∧ a + b ≥ c + 1)) ∧
  (∀ k : ℕ, (∃ a b c : ℕ, 1 ≤ a ∧ a < b ∧ b < c ∧ c ≤ n ∧
    c - b ≥ k ∧ b - a ≥ k ∧ a + b ≥ c + 1) →
    k ≤ (n - 1) / 3) :=
by sorry

end NUMINAMATH_CALUDE_largest_integer_difference_in_triangle_l1020_102063


namespace NUMINAMATH_CALUDE_division_result_l1020_102087

theorem division_result : (4.036 : ℝ) / 0.02 = 201.8 := by
  sorry

end NUMINAMATH_CALUDE_division_result_l1020_102087


namespace NUMINAMATH_CALUDE_average_waiting_time_for_first_bite_l1020_102005

theorem average_waiting_time_for_first_bite 
  (rod1_bites : ℝ) 
  (rod2_bites : ℝ) 
  (total_bites : ℝ) 
  (time_interval : ℝ) 
  (h1 : rod1_bites = 3)
  (h2 : rod2_bites = 2)
  (h3 : total_bites = rod1_bites + rod2_bites)
  (h4 : time_interval = 6) :
  (time_interval / total_bites) = 6/5 := by
  sorry

end NUMINAMATH_CALUDE_average_waiting_time_for_first_bite_l1020_102005


namespace NUMINAMATH_CALUDE_distance_on_line_l1020_102091

/-- Given two points on a line y = kx + b, prove that their distance is |x₁ - x₂|√(1 + k²) -/
theorem distance_on_line (k b x₁ x₂ : ℝ) :
  let y₁ := k * x₁ + b
  let y₂ := k * x₂ + b
  ((x₁ - x₂)^2 + (y₁ - y₂)^2).sqrt = |x₁ - x₂| * (1 + k^2).sqrt := by
  sorry

end NUMINAMATH_CALUDE_distance_on_line_l1020_102091


namespace NUMINAMATH_CALUDE_smallest_three_digit_multiple_of_13_l1020_102009

theorem smallest_three_digit_multiple_of_13 : 
  ∀ n : ℕ, n ≥ 100 ∧ 13 ∣ n → n ≥ 104 :=
by
  sorry

end NUMINAMATH_CALUDE_smallest_three_digit_multiple_of_13_l1020_102009


namespace NUMINAMATH_CALUDE_madeline_water_goal_l1020_102047

/-- The amount of water Madeline wants to drink in a day -/
def waterGoal (bottleCapacity : ℕ) (refills : ℕ) (additionalWater : ℕ) : ℕ :=
  bottleCapacity * refills + additionalWater

/-- Proves that Madeline's water goal is 100 ounces -/
theorem madeline_water_goal :
  waterGoal 12 7 16 = 100 := by
  sorry

end NUMINAMATH_CALUDE_madeline_water_goal_l1020_102047


namespace NUMINAMATH_CALUDE_min_value_product_l1020_102029

theorem min_value_product (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) (h_prod : x * y * z = 1) :
  (3 * x + y) * (x + 3 * z) * (y + z + 1) ≥ 48 ∧
  ∃ (x₀ y₀ z₀ : ℝ), x₀ > 0 ∧ y₀ > 0 ∧ z₀ > 0 ∧ x₀ * y₀ * z₀ = 1 ∧
    (3 * x₀ + y₀) * (x₀ + 3 * z₀) * (y₀ + z₀ + 1) = 48 :=
by sorry

end NUMINAMATH_CALUDE_min_value_product_l1020_102029


namespace NUMINAMATH_CALUDE_ms_hatcher_students_l1020_102020

theorem ms_hatcher_students (third_graders : ℕ) (fourth_graders : ℕ) (fifth_graders : ℕ) : 
  third_graders = 20 →
  fourth_graders = 2 * third_graders →
  fifth_graders = third_graders / 2 →
  third_graders + fourth_graders + fifth_graders = 70 := by
  sorry

end NUMINAMATH_CALUDE_ms_hatcher_students_l1020_102020


namespace NUMINAMATH_CALUDE_contractor_absent_days_l1020_102090

/-- Proves that given the contract conditions, the number of absent days is 6 -/
theorem contractor_absent_days
  (total_days : ℕ)
  (pay_per_day : ℚ)
  (fine_per_day : ℚ)
  (total_amount : ℚ)
  (h1 : total_days = 30)
  (h2 : pay_per_day = 25)
  (h3 : fine_per_day = 7.5)
  (h4 : total_amount = 555) :
  ∃ (absent_days : ℕ),
    absent_days = 6 ∧
    (pay_per_day * (total_days - absent_days) - fine_per_day * absent_days = total_amount) :=
by sorry

end NUMINAMATH_CALUDE_contractor_absent_days_l1020_102090


namespace NUMINAMATH_CALUDE_factorial_of_factorial_divided_by_factorial_l1020_102036

theorem factorial_of_factorial_divided_by_factorial :
  (Nat.factorial (Nat.factorial 4)) / (Nat.factorial 4) = Nat.factorial 23 := by
  sorry

end NUMINAMATH_CALUDE_factorial_of_factorial_divided_by_factorial_l1020_102036


namespace NUMINAMATH_CALUDE_product_of_roots_l1020_102062

theorem product_of_roots (x : ℝ) : 
  (x^3 - 9*x^2 + 27*x - 8 = 0) → 
  (∃ p q r : ℝ, x^3 - 9*x^2 + 27*x - 8 = (x - p) * (x - q) * (x - r) ∧ p * q * r = 8) :=
by sorry

end NUMINAMATH_CALUDE_product_of_roots_l1020_102062


namespace NUMINAMATH_CALUDE_cos_sixty_degrees_equals_half_l1020_102088

theorem cos_sixty_degrees_equals_half : Real.cos (π / 3) = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_cos_sixty_degrees_equals_half_l1020_102088


namespace NUMINAMATH_CALUDE_sqrt_sum_inequality_l1020_102053

theorem sqrt_sum_inequality (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a + b = 1) :
  Real.sqrt (2 * a + 1) + Real.sqrt (2 * b + 1) ≤ 2 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_sum_inequality_l1020_102053


namespace NUMINAMATH_CALUDE_quadratic_ratio_l1020_102031

/-- Given a quadratic function f(x) = x^2 + 1500x + 1500, 
    prove that when expressed as (x + b)^2 + c, 
    the ratio c/b equals -748 -/
theorem quadratic_ratio (f : ℝ → ℝ) (b c : ℝ) : 
  (∀ x, f x = x^2 + 1500*x + 1500) → 
  (∀ x, f x = (x + b)^2 + c) → 
  c / b = -748 := by
sorry

end NUMINAMATH_CALUDE_quadratic_ratio_l1020_102031


namespace NUMINAMATH_CALUDE_symmetrical_function_is_two_minus_ln_l1020_102008

/-- A function whose graph is symmetrical to y = e^(2-x) with respect to y = x -/
def SymmetricalToExp (f : ℝ → ℝ) : Prop :=
  ∀ x y, f x = y ↔ f y = x

/-- The main theorem stating that f(x) = 2 - ln(x) -/
theorem symmetrical_function_is_two_minus_ln (f : ℝ → ℝ) 
    (h : SymmetricalToExp f) : 
    ∀ x > 0, f x = 2 - Real.log x := by
  sorry

end NUMINAMATH_CALUDE_symmetrical_function_is_two_minus_ln_l1020_102008


namespace NUMINAMATH_CALUDE_wire_service_reporters_l1020_102094

theorem wire_service_reporters (total : ℝ) (local_politics : ℝ) (politics : ℝ) : 
  local_politics = 0.2 * total →
  local_politics = 0.8 * politics →
  (total - politics) / total = 0.75 :=
by sorry

end NUMINAMATH_CALUDE_wire_service_reporters_l1020_102094


namespace NUMINAMATH_CALUDE_perpendicular_sufficient_condition_perpendicular_not_necessary_condition_l1020_102019

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the relations
variable (perpendicular : Line → Plane → Prop)
variable (parallel : Plane → Plane → Prop)

-- Define the given conditions
variable (m n : Line)
variable (α β : Plane)
variable (h_diff_lines : m ≠ n)
variable (h_diff_planes : α ≠ β)

-- State the theorem
theorem perpendicular_sufficient_condition 
  (h_parallel : parallel α β) 
  (h_perp : perpendicular n β) : 
  perpendicular n α :=
sorry

-- State that the condition is not necessary
theorem perpendicular_not_necessary_condition :
  ¬(∀ (n : Line) (α β : Plane), 
    perpendicular n α → 
    (parallel α β ∧ perpendicular n β)) :=
sorry

end NUMINAMATH_CALUDE_perpendicular_sufficient_condition_perpendicular_not_necessary_condition_l1020_102019


namespace NUMINAMATH_CALUDE_cleo_utility_equality_l1020_102061

/-- Utility function for Cleo's activities -/
def utility (reading : ℝ) (painting : ℝ) : ℝ := reading * painting

/-- Time spent painting on Saturday -/
def saturday_painting (t : ℝ) : ℝ := t

/-- Time spent reading on Saturday -/
def saturday_reading (t : ℝ) : ℝ := 10 - 2 * t

/-- Time spent painting on Sunday -/
def sunday_painting (t : ℝ) : ℝ := 5 - t

/-- Time spent reading on Sunday -/
def sunday_reading (t : ℝ) : ℝ := 2 * t + 4

theorem cleo_utility_equality :
  ∃ t : ℝ, utility (saturday_reading t) (saturday_painting t) = utility (sunday_reading t) (sunday_painting t) ∧ t = 0 := by
  sorry

end NUMINAMATH_CALUDE_cleo_utility_equality_l1020_102061


namespace NUMINAMATH_CALUDE_paintable_area_theorem_l1020_102064

/-- Calculates the total paintable area of rooms with given dimensions and unpaintable area -/
def totalPaintableArea (numRooms length width height unpaintableArea : ℕ) : ℕ :=
  let wallArea := 2 * (length * height + width * height)
  let paintableAreaPerRoom := wallArea - unpaintableArea
  numRooms * paintableAreaPerRoom

/-- Theorem stating that the total paintable area of 4 rooms with given dimensions is 1644 sq ft -/
theorem paintable_area_theorem :
  totalPaintableArea 4 15 12 9 75 = 1644 := by
  sorry

end NUMINAMATH_CALUDE_paintable_area_theorem_l1020_102064


namespace NUMINAMATH_CALUDE_first_floor_rooms_l1020_102067

theorem first_floor_rooms : ∃ (x : ℕ), x > 0 ∧ 
  (6 * (x - 1) = 5 * x + 4) ∧ x = 10 := by
  sorry

end NUMINAMATH_CALUDE_first_floor_rooms_l1020_102067


namespace NUMINAMATH_CALUDE_equality_relations_l1020_102077

theorem equality_relations (a b c d : ℝ) (h : a * b = c * d) : 
  (a / c = d / b) ∧ 
  (a / d = c / b) ∧ 
  ((a + c) / c = (d + b) / b) ∧ 
  ¬ ∀ (a b c d : ℝ), a * b = c * d → (a + 1) / (c + 1) = (d + 1) / (b + 1) :=
by sorry

end NUMINAMATH_CALUDE_equality_relations_l1020_102077


namespace NUMINAMATH_CALUDE_calculation_proof_l1020_102013

theorem calculation_proof : 20062006 * 2007 + 20072007 * 2008 - 2006 * 20072007 - 2007 * 20082008 = 0 := by
  sorry

end NUMINAMATH_CALUDE_calculation_proof_l1020_102013


namespace NUMINAMATH_CALUDE_midpoint_minus_eighth_l1020_102078

theorem midpoint_minus_eighth (a b c : ℚ) : 
  a = 1/4 → b = 1/2 → c = 1/8 → 
  ((a + b) / 2) - c = 1/4 := by sorry

end NUMINAMATH_CALUDE_midpoint_minus_eighth_l1020_102078

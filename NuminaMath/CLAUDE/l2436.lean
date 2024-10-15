import Mathlib

namespace NUMINAMATH_CALUDE_line_equation_correct_l2436_243668

/-- The y-intercept of the line 2x + y + 2 = 0 -/
def y_intercept : ℝ := -2

/-- The point A through which line l passes -/
def point_A : ℝ × ℝ := (2, 0)

/-- The equation of line l -/
def line_equation (x y : ℝ) : Prop := x - y - 2 = 0

theorem line_equation_correct :
  (line_equation point_A.1 point_A.2) ∧
  (line_equation 0 y_intercept) ∧
  (∀ x y : ℝ, line_equation x y → (2 * x + y + 2 = 0 → y = y_intercept)) :=
sorry

end NUMINAMATH_CALUDE_line_equation_correct_l2436_243668


namespace NUMINAMATH_CALUDE_system_solution_l2436_243629

theorem system_solution :
  ∃ (x y : ℝ), x + y = 1 ∧ 4 * x + y = 10 ∧ x = 3 ∧ y = -2 :=
by sorry

end NUMINAMATH_CALUDE_system_solution_l2436_243629


namespace NUMINAMATH_CALUDE_remove_one_for_avg_eight_point_five_l2436_243602

theorem remove_one_for_avg_eight_point_five (n : Nat) (h : n = 15) :
  let list := List.range n
  let sum := n * (n + 1) / 2
  let removed := 1
  let remaining_sum := sum - removed
  let remaining_count := n - 1
  (remaining_sum : ℚ) / remaining_count = 17/2 := by
  sorry

end NUMINAMATH_CALUDE_remove_one_for_avg_eight_point_five_l2436_243602


namespace NUMINAMATH_CALUDE_parallel_lines_theorem_l2436_243677

/-- A line in a 3D space --/
structure Line3D where
  -- We don't need to define the internal structure of a line
  -- for this problem, so we leave it empty

/-- Two lines are parallel --/
def parallel (l1 l2 : Line3D) : Prop :=
  sorry

/-- Two lines form equal angles with a third line --/
def equal_angles (l1 l2 l3 : Line3D) : Prop :=
  sorry

/-- A line is perpendicular to another line --/
def perpendicular (l1 l2 : Line3D) : Prop :=
  sorry

/-- Main theorem: Exactly two of the given propositions about parallel lines are false --/
theorem parallel_lines_theorem :
  ∃ (prop1 prop2 prop3 : Prop),
    prop1 = (∀ l1 l2 l3 : Line3D, equal_angles l1 l2 l3 → parallel l1 l2) ∧
    prop2 = (∀ l1 l2 l3 : Line3D, perpendicular l1 l3 → perpendicular l2 l3 → parallel l1 l2) ∧
    prop3 = (∀ l1 l2 l3 : Line3D, parallel l1 l3 → parallel l2 l3 → parallel l1 l2) ∧
    (¬prop1 ∧ ¬prop2 ∧ prop3) :=
  sorry

end NUMINAMATH_CALUDE_parallel_lines_theorem_l2436_243677


namespace NUMINAMATH_CALUDE_number_puzzle_l2436_243634

theorem number_puzzle (x : ℤ) : x - 13 = 31 → x + 11 = 55 := by
  sorry

end NUMINAMATH_CALUDE_number_puzzle_l2436_243634


namespace NUMINAMATH_CALUDE_xyz_inequality_l2436_243606

theorem xyz_inequality (x y z : ℝ) (h : x^2 + y^2 + z^2 + 2*x*y*z = 1) : 
  8*x*y*z ≤ 1 ∧ 
  (8*x*y*z = 1 ↔ 
    ((x, y, z) = (1/2, 1/2, 1/2) ∨ 
     (x, y, z) = (-1/2, -1/2, 1/2) ∨ 
     (x, y, z) = (-1/2, 1/2, -1/2) ∨ 
     (x, y, z) = (1/2, -1/2, -1/2))) := by
  sorry

end NUMINAMATH_CALUDE_xyz_inequality_l2436_243606


namespace NUMINAMATH_CALUDE_coordinates_wrt_origin_l2436_243630

/-- A point in a 2D plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- The origin of the coordinate system -/
def origin : Point := ⟨0, 0⟩

/-- The coordinate system -/
structure CoordinateSystem where
  origin : Point

/-- A point's coordinates with respect to a coordinate system -/
def coordinates (p : Point) (cs : CoordinateSystem) : Point :=
  ⟨p.x - cs.origin.x, p.y - cs.origin.y⟩

theorem coordinates_wrt_origin (A : Point) (cs : CoordinateSystem) :
  A.x = -1 ∧ A.y = 2 → coordinates A cs = A :=
by sorry

end NUMINAMATH_CALUDE_coordinates_wrt_origin_l2436_243630


namespace NUMINAMATH_CALUDE_number_relationship_l2436_243690

theorem number_relationship (a b c d : ℝ) 
  (ha : a > 0) (hb : b > 0) (hc : c > 0) (hd : d > 0)
  (ha2 : a^2 = 2) (hb3 : b^3 = 3) (hc4 : c^4 = 4) (hd5 : d^5 = 5) :
  a = c ∧ c < d ∧ d < b :=
sorry

end NUMINAMATH_CALUDE_number_relationship_l2436_243690


namespace NUMINAMATH_CALUDE_quadratic_inequality_solution_set_l2436_243624

theorem quadratic_inequality_solution_set 
  (a b c : ℝ) 
  (h : Set.Ioo (-3 : ℝ) (1/2) = {x : ℝ | c * x^2 + b * x + a < 0}) :
  {x : ℝ | a * x^2 + b * x + c ≥ 0} = Set.Icc (-1/3 : ℝ) 2 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_inequality_solution_set_l2436_243624


namespace NUMINAMATH_CALUDE_smallest_equivalent_angle_l2436_243618

theorem smallest_equivalent_angle (x : ℝ) (h : x = -11/4 * Real.pi) :
  ∃ (θ : ℝ) (k : ℤ),
    x = θ + 2 * ↑k * Real.pi ∧
    θ ∈ Set.Icc (-Real.pi) Real.pi ∧
    ∀ (φ : ℝ) (m : ℤ),
      x = φ + 2 * ↑m * Real.pi →
      φ ∈ Set.Icc (-Real.pi) Real.pi →
      |θ| ≤ |φ| ∧
    θ = -3/4 * Real.pi :=
sorry

end NUMINAMATH_CALUDE_smallest_equivalent_angle_l2436_243618


namespace NUMINAMATH_CALUDE_T_coprime_and_sum_reciprocals_l2436_243641

def T : ℕ → ℕ
  | 0 => 2
  | n + 1 => T n^2 - T n + 1

theorem T_coprime_and_sum_reciprocals :
  (∀ m n, m ≠ n → Nat.gcd (T m) (T n) = 1) ∧
  (∑' i, (T i)⁻¹ : ℝ) = 1 := by
  sorry

end NUMINAMATH_CALUDE_T_coprime_and_sum_reciprocals_l2436_243641


namespace NUMINAMATH_CALUDE_intersection_parallel_line_exists_specific_intersection_parallel_line_l2436_243694

/-- Given two lines l₁ and l₂ in the plane, and a third line l₃,
    this theorem states that there exists a line l that passes through
    the intersection of l₁ and l₂ and is parallel to l₃. -/
theorem intersection_parallel_line_exists (a₁ b₁ c₁ a₂ b₂ c₂ a₃ b₃ c₃ : ℝ) :
  ∃ (a b c : ℝ),
    -- l₁: a₁x + b₁y + c₁ = 0
    -- l₂: a₂x + b₂y + c₂ = 0
    -- l₃: a₃x + b₃y + c₃ = 0
    -- l: ax + by + c = 0
    -- l passes through the intersection of l₁ and l₂
    (∀ (x y : ℝ), a₁ * x + b₁ * y + c₁ = 0 ∧ a₂ * x + b₂ * y + c₂ = 0 → a * x + b * y + c = 0) ∧
    -- l is parallel to l₃
    (∃ (k : ℝ), k ≠ 0 ∧ a = k * a₃ ∧ b = k * b₃) :=
by
  sorry

/-- The specific instance of the theorem for the given problem -/
theorem specific_intersection_parallel_line :
  ∃ (a b c : ℝ),
    -- l₁: 2x + 3y - 5 = 0
    -- l₂: 3x - 2y - 3 = 0
    -- l₃: 2x + y - 3 = 0
    -- l: ax + by + c = 0
    (∀ (x y : ℝ), 2 * x + 3 * y - 5 = 0 ∧ 3 * x - 2 * y - 3 = 0 → a * x + b * y + c = 0) ∧
    (∃ (k : ℝ), k ≠ 0 ∧ a = k * 2 ∧ b = k * 1) ∧
    a = 26 ∧ b = -13 ∧ c = -29 :=
by
  sorry

end NUMINAMATH_CALUDE_intersection_parallel_line_exists_specific_intersection_parallel_line_l2436_243694


namespace NUMINAMATH_CALUDE_product_41_reciprocal_squares_sum_l2436_243684

theorem product_41_reciprocal_squares_sum :
  ∀ a b : ℕ+,
  (a.val : ℕ) * (b.val : ℕ) = 41 →
  (1 : ℚ) / (a.val^2 : ℚ) + (1 : ℚ) / (b.val^2 : ℚ) = 1682 / 1681 :=
by sorry

end NUMINAMATH_CALUDE_product_41_reciprocal_squares_sum_l2436_243684


namespace NUMINAMATH_CALUDE_triangle_ratio_l2436_243603

/-- Given an acute triangle ABC with a point D inside it, 
    if ∠ADB = ∠ACB + 90° and AC * BD = AD * BC, 
    then (AB * CD) / (AC * BD) = √2 -/
theorem triangle_ratio (A B C D : ℂ) : 
  A ≠ B ∧ B ≠ C ∧ C ≠ A ∧  -- A, B, C form a triangle
  (∃ t : ℝ, 0 < t ∧ t < 1 ∧ D = t*B + (1-t)*C) ∧  -- D is inside triangle ABC
  Complex.arg ((D - B) / (D - A)) = Complex.arg ((C - B) / (C - A)) + Real.pi / 2 ∧  -- ∠ADB = ∠ACB + 90°
  Complex.abs (C - A) * Complex.abs (D - B) = Complex.abs (D - A) * Complex.abs (C - B) →  -- AC * BD = AD * BC
  Complex.abs ((B - A) * (D - C)) / (Complex.abs (C - A) * Complex.abs (D - B)) = Real.sqrt 2 := by
sorry

end NUMINAMATH_CALUDE_triangle_ratio_l2436_243603


namespace NUMINAMATH_CALUDE_summit_conference_attendance_l2436_243656

/-- The number of diplomats who attended the summit conference -/
def D : ℕ := 120

/-- The number of diplomats who spoke French -/
def french_speakers : ℕ := 20

/-- The number of diplomats who did not speak Hindi -/
def non_hindi_speakers : ℕ := 32

/-- The proportion of diplomats who spoke neither French nor Hindi -/
def neither_french_nor_hindi : ℚ := 1/5

/-- The proportion of diplomats who spoke both French and Hindi -/
def both_french_and_hindi : ℚ := 1/10

theorem summit_conference_attendance :
  D = 120 ∧
  french_speakers = 20 ∧
  non_hindi_speakers = 32 ∧
  neither_french_nor_hindi = 1/5 ∧
  both_french_and_hindi = 1/10 ∧
  (D : ℚ) * neither_french_nor_hindi + (D : ℚ) * both_french_and_hindi + french_speakers = D :=
sorry

end NUMINAMATH_CALUDE_summit_conference_attendance_l2436_243656


namespace NUMINAMATH_CALUDE_equations_represent_parabola_and_ellipse_l2436_243682

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents the equations mx + ny² = 0 and mx² + ny² = 1 -/
def Equations (m n : ℝ) : Prop :=
  m ≠ 0 ∧ n ≠ 0 ∧
  ∃ (p : Point), m * p.x + n * p.y^2 = 0 ∧ m * p.x^2 + n * p.y^2 = 1

/-- Represents a parabola opening to the right -/
def ParabolaOpeningRight (m n : ℝ) : Prop :=
  m < 0 ∧ n > 0 ∧
  ∀ (p : Point), m * p.x + n * p.y^2 = 0 → p.x = -n / m * p.y^2

/-- Represents an ellipse centered at the origin -/
def Ellipse (m n : ℝ) : Prop :=
  m ≠ 0 ∧ n ≠ 0 ∧
  ∀ (p : Point), m * p.x^2 + n * p.y^2 = 1

/-- Theorem stating that the equations represent a parabola opening right and an ellipse -/
theorem equations_represent_parabola_and_ellipse (m n : ℝ) :
  Equations m n → ParabolaOpeningRight m n ∧ Ellipse m n :=
by sorry

end NUMINAMATH_CALUDE_equations_represent_parabola_and_ellipse_l2436_243682


namespace NUMINAMATH_CALUDE_square_divided_into_triangles_even_count_l2436_243636

theorem square_divided_into_triangles_even_count (a : ℕ) (h : a > 0) :
  let triangle_area : ℚ := 3 * 4 / 2
  let square_area : ℚ := a^2
  let num_triangles : ℚ := square_area / triangle_area
  (∃ k : ℕ, num_triangles = k ∧ k % 2 = 0) :=
sorry

end NUMINAMATH_CALUDE_square_divided_into_triangles_even_count_l2436_243636


namespace NUMINAMATH_CALUDE_tilde_result_bounds_l2436_243610

def tilde (a b : ℚ) : ℚ := |a - b|

def consecutive_integers (n : ℕ) : List ℚ := List.range n

def perform_tilde (l : List ℚ) : ℚ :=
  l.foldl tilde (l.head!)

def max_tilde_result (n : ℕ) : ℚ :=
  if n % 4 == 1 then n - 1 else n

def min_tilde_result (n : ℕ) : ℚ :=
  if n % 4 == 2 || n % 4 == 3 then 1 else 0

theorem tilde_result_bounds (n : ℕ) (l : List ℚ) :
  l.length = n ∧ l.toFinset = (consecutive_integers n).toFinset →
  perform_tilde l ≤ max_tilde_result n ∧
  perform_tilde l ≥ min_tilde_result n :=
sorry

end NUMINAMATH_CALUDE_tilde_result_bounds_l2436_243610


namespace NUMINAMATH_CALUDE_apple_basket_theorem_l2436_243635

/-- Represents the capacity of an apple basket -/
structure Basket where
  capacity : ℕ

/-- Represents the current state of Jack's basket -/
structure JackBasket extends Basket where
  current : ℕ
  space_left : ℕ

/-- Theorem about apple baskets -/
theorem apple_basket_theorem (jack : JackBasket) (jill : Basket) : 
  jack.capacity = 12 →
  jack.space_left = 4 →
  jill.capacity = 2 * jack.capacity →
  (jill.capacity / (jack.capacity - jack.space_left) : ℕ) = 3 := by
  sorry

end NUMINAMATH_CALUDE_apple_basket_theorem_l2436_243635


namespace NUMINAMATH_CALUDE_root_product_cubic_l2436_243639

theorem root_product_cubic (a b c : ℂ) : 
  (∀ x : ℂ, 3 * x^3 - 8 * x^2 + 5 * x - 9 = 0 ↔ x = a ∨ x = b ∨ x = c) →
  a * b * c = 3 := by
sorry

end NUMINAMATH_CALUDE_root_product_cubic_l2436_243639


namespace NUMINAMATH_CALUDE_quadratic_factorization_l2436_243652

theorem quadratic_factorization (a b : ℤ) :
  (∀ y : ℝ, 4 * y^2 - 3 * y - 28 = (4 * y + a) * (y + b)) →
  a - b = -11 := by
sorry

end NUMINAMATH_CALUDE_quadratic_factorization_l2436_243652


namespace NUMINAMATH_CALUDE_tims_contribution_l2436_243670

-- Define the initial number of bales
def initial_bales : ℕ := 28

-- Define the final number of bales
def final_bales : ℕ := 54

-- Define Tim's contribution
def tims_bales : ℕ := final_bales - initial_bales

-- Theorem to prove
theorem tims_contribution : tims_bales = 26 := by sorry

end NUMINAMATH_CALUDE_tims_contribution_l2436_243670


namespace NUMINAMATH_CALUDE_intersection_equality_l2436_243611

def A : Set ℝ := {x | |x - 4| < 2 * x}
def B (a : ℝ) : Set ℝ := {x | x * (x - a) ≥ (a + 6) * (x - a)}

theorem intersection_equality (a : ℝ) : A ∩ B a = A ↔ a ≤ -14/3 := by
  sorry

end NUMINAMATH_CALUDE_intersection_equality_l2436_243611


namespace NUMINAMATH_CALUDE_boat_speed_ratio_l2436_243647

def boat_speed : ℝ := 20
def current_speed : ℝ := 4
def distance : ℝ := 2

theorem boat_speed_ratio :
  let downstream_speed := boat_speed + current_speed
  let upstream_speed := boat_speed - current_speed
  let downstream_time := distance / downstream_speed
  let upstream_time := distance / upstream_speed
  let total_time := downstream_time + upstream_time
  let total_distance := 2 * distance
  let average_speed := total_distance / total_time
  (average_speed / boat_speed) = 24 / 25 := by
sorry

end NUMINAMATH_CALUDE_boat_speed_ratio_l2436_243647


namespace NUMINAMATH_CALUDE_fraction_equality_l2436_243657

theorem fraction_equality (a b : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) 
  (h : (5 * a + 2 * b) / (2 * a - 5 * b) = 3) : 
  (2 * a + 5 * b) / (5 * a - 2 * b) = 39 / 83 := by
  sorry

end NUMINAMATH_CALUDE_fraction_equality_l2436_243657


namespace NUMINAMATH_CALUDE_sum_fusion_2020_l2436_243631

/-- A number is a sum fusion number if it's equal to the square difference of two consecutive even numbers. -/
def IsSumFusionNumber (n : ℕ) : Prop :=
  ∃ k : ℕ, n = (2*k + 2)^2 - (2*k)^2

/-- 2020 is a sum fusion number. -/
theorem sum_fusion_2020 : IsSumFusionNumber 2020 := by
  sorry

#check sum_fusion_2020

end NUMINAMATH_CALUDE_sum_fusion_2020_l2436_243631


namespace NUMINAMATH_CALUDE_vehicle_speed_problem_l2436_243648

/-- Represents the problem of determining initial and final speeds of a vehicle --/
theorem vehicle_speed_problem
  (total_distance : ℝ)
  (initial_distance : ℝ)
  (initial_time : ℝ)
  (late_time : ℝ)
  (early_time : ℝ)
  (h1 : total_distance = 280)
  (h2 : initial_distance = 112)
  (h3 : initial_time = 2)
  (h4 : late_time = 0.5)
  (h5 : early_time = 0.5)
  : ∃ (initial_speed final_speed : ℝ),
    initial_speed = initial_distance / initial_time ∧
    final_speed = (total_distance - initial_distance) / (
      (total_distance / initial_speed - late_time) - initial_time
    ) ∧
    initial_speed = 56 ∧
    final_speed = 84 := by
  sorry


end NUMINAMATH_CALUDE_vehicle_speed_problem_l2436_243648


namespace NUMINAMATH_CALUDE_max_digit_sum_for_reciprocal_l2436_243691

theorem max_digit_sum_for_reciprocal (a b c z : ℕ) : 
  (a ≤ 9 ∧ b ≤ 9 ∧ c ≤ 9) →  -- a, b, and c are digits
  (100 * a + 10 * b + c = 1000 / z) →  -- 0.abc = 1/z
  (0 < z ∧ z ≤ 15) →  -- 0 < z ≤ 15
  (∀ w, (w ≤ 9 ∧ w ≤ 9 ∧ w ≤ 9) → 
        (100 * w + 10 * w + w = 1000 / z) → 
        (0 < z ∧ z ≤ 15) → 
        a + b + c ≥ w + w + w) →
  a + b + c = 8 :=
by sorry

end NUMINAMATH_CALUDE_max_digit_sum_for_reciprocal_l2436_243691


namespace NUMINAMATH_CALUDE_open_box_volume_proof_l2436_243612

/-- Calculates the volume of an open box formed by cutting squares from the corners of a rectangular sheet. -/
def open_box_volume (sheet_length sheet_width cut_size : ℝ) : ℝ :=
  (sheet_length - 2 * cut_size) * (sheet_width - 2 * cut_size) * cut_size

/-- Proves that the volume of the open box formed from a 48 m x 36 m sheet with 6 m x 6 m corner cuts is 5184 m³. -/
theorem open_box_volume_proof :
  open_box_volume 48 36 6 = 5184 := by
  sorry

#eval open_box_volume 48 36 6

end NUMINAMATH_CALUDE_open_box_volume_proof_l2436_243612


namespace NUMINAMATH_CALUDE_larger_number_proof_l2436_243646

theorem larger_number_proof (a b : ℕ) 
  (hcf_cond : Nat.gcd a b = 23)
  (lcm_cond : Nat.lcm a b = 23 * 13 * 16) :
  max a b = 368 := by
sorry

end NUMINAMATH_CALUDE_larger_number_proof_l2436_243646


namespace NUMINAMATH_CALUDE_jeremy_age_l2436_243663

/-- Represents the ages of three people -/
structure Ages where
  jeremy : ℕ
  sebastian : ℕ
  sophia : ℕ

/-- The conditions of the problem -/
def problem_conditions (ages : Ages) : Prop :=
  (ages.jeremy + 3) + (ages.sebastian + 3) + (ages.sophia + 3) = 150 ∧
  ages.sebastian = ages.jeremy + 4 ∧
  ages.sophia + 3 = 60

/-- The theorem stating Jeremy's current age -/
theorem jeremy_age (ages : Ages) (h : problem_conditions ages) : ages.jeremy = 40 := by
  sorry

end NUMINAMATH_CALUDE_jeremy_age_l2436_243663


namespace NUMINAMATH_CALUDE_abs_of_nonnegative_l2436_243651

theorem abs_of_nonnegative (x : ℝ) : x ≥ 0 → |x| = x := by
  sorry

end NUMINAMATH_CALUDE_abs_of_nonnegative_l2436_243651


namespace NUMINAMATH_CALUDE_may_salary_is_6500_l2436_243628

/-- Calculates the salary for May given the average salaries and January's salary -/
def salary_may (avg_jan_to_apr avg_feb_to_may jan_salary : ℚ) : ℚ :=
  4 * avg_feb_to_may - (4 * avg_jan_to_apr - jan_salary)

/-- Proves that the salary for May is 6500 given the conditions -/
theorem may_salary_is_6500 :
  let avg_jan_to_apr : ℚ := 8000
  let avg_feb_to_may : ℚ := 8200
  let jan_salary : ℚ := 5700
  salary_may avg_jan_to_apr avg_feb_to_may jan_salary = 6500 :=
by
  sorry

#eval salary_may 8000 8200 5700

end NUMINAMATH_CALUDE_may_salary_is_6500_l2436_243628


namespace NUMINAMATH_CALUDE_circumcircle_equation_l2436_243660

/-- Given a triangle ABC with vertices A(0,4), B(0,0), and C(3,0),
    prove that (x-3/2)^2 + (y-2)^2 = 25/4 is the equation of its circumcircle. -/
theorem circumcircle_equation (x y : ℝ) : 
  let A : ℝ × ℝ := (0, 4)
  let B : ℝ × ℝ := (0, 0)
  let C : ℝ × ℝ := (3, 0)
  (x - 3/2)^2 + (y - 2)^2 = 25/4 ↔ 
    ∃ (center : ℝ × ℝ) (radius : ℝ), 
      (center.1 - A.1)^2 + (center.2 - A.2)^2 = radius^2 ∧
      (center.1 - B.1)^2 + (center.2 - B.2)^2 = radius^2 ∧
      (center.1 - C.1)^2 + (center.2 - C.2)^2 = radius^2 :=
by sorry


end NUMINAMATH_CALUDE_circumcircle_equation_l2436_243660


namespace NUMINAMATH_CALUDE_rectangle_area_18_l2436_243619

def rectangle_area (w l : ℕ+) : ℕ := w.val * l.val

theorem rectangle_area_18 :
  {p : ℕ+ × ℕ+ | rectangle_area p.1 p.2 = 18} =
  {(1, 18), (2, 9), (3, 6), (6, 3), (9, 2), (18, 1)} := by
  sorry

end NUMINAMATH_CALUDE_rectangle_area_18_l2436_243619


namespace NUMINAMATH_CALUDE_product_of_roots_quadratic_l2436_243626

theorem product_of_roots_quadratic (x : ℝ) : 
  (8 = -2 * x^2 - 6 * x) → (∃ α β : ℝ, (α * β = 4 ∧ 8 = -2 * α^2 - 6 * α ∧ 8 = -2 * β^2 - 6 * β)) :=
by sorry

end NUMINAMATH_CALUDE_product_of_roots_quadratic_l2436_243626


namespace NUMINAMATH_CALUDE_same_color_probability_l2436_243678

theorem same_color_probability (total_balls green_balls red_balls : ℕ) 
  (h_total : total_balls = green_balls + red_balls)
  (h_green : green_balls = 6)
  (h_red : red_balls = 4) : 
  (green_balls / total_balls) ^ 2 + (red_balls / total_balls) ^ 2 = 13 / 25 := by
  sorry

end NUMINAMATH_CALUDE_same_color_probability_l2436_243678


namespace NUMINAMATH_CALUDE_number_difference_l2436_243609

theorem number_difference (a b : ℕ) (h1 : a + b = 22904) (h2 : b % 5 = 0) (h3 : b = 7 * a) : 
  b - a = 17178 := by
sorry

end NUMINAMATH_CALUDE_number_difference_l2436_243609


namespace NUMINAMATH_CALUDE_modulus_of_complex_expression_l2436_243675

theorem modulus_of_complex_expression : 
  Complex.abs ((1 + Complex.I) / (1 - Complex.I) + Complex.I) = 2 := by sorry

end NUMINAMATH_CALUDE_modulus_of_complex_expression_l2436_243675


namespace NUMINAMATH_CALUDE_teal_more_blue_l2436_243654

/-- The number of people surveyed -/
def total_surveyed : ℕ := 150

/-- The number of people who believe teal is "more green" -/
def more_green : ℕ := 80

/-- The number of people who believe teal is both "more green" and "more blue" -/
def both : ℕ := 40

/-- The number of people who think teal is neither "more green" nor "more blue" -/
def neither : ℕ := 20

/-- The number of people who believe teal is "more blue" -/
def more_blue : ℕ := total_surveyed - (more_green - both) - both - neither

theorem teal_more_blue : more_blue = 90 := by
  sorry

end NUMINAMATH_CALUDE_teal_more_blue_l2436_243654


namespace NUMINAMATH_CALUDE_apple_pyramid_sum_l2436_243645

/-- Calculates the number of apples in a single layer of the pyramid --/
def layer_apples (base_width : ℕ) (base_length : ℕ) (layer : ℕ) : ℕ :=
  (base_width - layer + 1) * (base_length - layer + 1)

/-- Calculates the total number of apples in the pyramid --/
def total_apples (base_width : ℕ) (base_length : ℕ) : ℕ :=
  (List.range (min base_width base_length)).foldl (λ sum layer => sum + layer_apples base_width base_length layer) 0

theorem apple_pyramid_sum :
  total_apples 6 9 = 154 :=
by sorry

end NUMINAMATH_CALUDE_apple_pyramid_sum_l2436_243645


namespace NUMINAMATH_CALUDE_red_marbles_in_bag_l2436_243600

theorem red_marbles_in_bag (total_marbles : ℕ) 
  (prob_two_non_red : ℚ) (red_marbles : ℕ) : 
  total_marbles = 48 →
  prob_two_non_red = 9/16 →
  (((total_marbles - red_marbles : ℚ) / total_marbles) ^ 2 = prob_two_non_red) →
  red_marbles = 12 := by
  sorry

end NUMINAMATH_CALUDE_red_marbles_in_bag_l2436_243600


namespace NUMINAMATH_CALUDE_min_paper_toys_l2436_243664

/-- Represents the number of paper toys that can be made from one sheet -/
structure PaperToys where
  boats : Nat
  planes : Nat

/-- The number of paper toys that can be made from one sheet -/
def sheet_capacity : PaperToys :=
  { boats := 8, planes := 6 }

/-- The minimum number of paper toys that can be made -/
def min_toys : Nat :=
  sheet_capacity.boats

theorem min_paper_toys :
  ∀ (n : Nat), n ≥ min_toys →
  ∃ (b p : Nat), b > 0 ∧ n = b * sheet_capacity.boats + p * sheet_capacity.planes :=
by sorry

end NUMINAMATH_CALUDE_min_paper_toys_l2436_243664


namespace NUMINAMATH_CALUDE_log_25_between_1_and_2_l2436_243661

theorem log_25_between_1_and_2 :
  ∃ (a b : ℤ), a + 1 = b ∧ (a : ℝ) < Real.log 25 / Real.log 10 ∧ Real.log 25 / Real.log 10 < b :=
sorry

end NUMINAMATH_CALUDE_log_25_between_1_and_2_l2436_243661


namespace NUMINAMATH_CALUDE_sin_135_degrees_l2436_243667

theorem sin_135_degrees : Real.sin (135 * π / 180) = 1 / Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_sin_135_degrees_l2436_243667


namespace NUMINAMATH_CALUDE_solution_set_theorem_l2436_243613

def increasing_function (f : ℝ → ℝ) : Prop :=
  ∀ x y, x < y → f x < f y

theorem solution_set_theorem (f : ℝ → ℝ) 
  (h_increasing : increasing_function f) 
  (h_f_0 : f 0 = -1) 
  (h_f_3 : f 3 = 1) :
  {x : ℝ | |f x| < 1} = Set.Ioo 0 3 := by
  sorry

end NUMINAMATH_CALUDE_solution_set_theorem_l2436_243613


namespace NUMINAMATH_CALUDE_probability_all_white_balls_l2436_243693

theorem probability_all_white_balls (total_balls : ℕ) (white_balls : ℕ) (drawn_balls : ℕ) :
  total_balls = 11 →
  white_balls = 5 →
  drawn_balls = 5 →
  (Nat.choose white_balls drawn_balls : ℚ) / (Nat.choose total_balls drawn_balls : ℚ) = 1 / 462 := by
sorry

end NUMINAMATH_CALUDE_probability_all_white_balls_l2436_243693


namespace NUMINAMATH_CALUDE_boutique_packaging_combinations_l2436_243616

theorem boutique_packaging_combinations :
  let wrapping_paper_designs : ℕ := 10
  let ribbon_colors : ℕ := 5
  let gift_card_varieties : ℕ := 6
  let decorative_tag_types : ℕ := 2
  wrapping_paper_designs * ribbon_colors * gift_card_varieties * decorative_tag_types = 600 :=
by sorry

end NUMINAMATH_CALUDE_boutique_packaging_combinations_l2436_243616


namespace NUMINAMATH_CALUDE_geometric_progression_proof_l2436_243686

theorem geometric_progression_proof (b q : ℝ) : 
  b + b*q + b*q^2 + b*q^3 = -40 ∧ 
  b^2 + (b*q)^2 + (b*q^2)^2 + (b*q^3)^2 = 3280 → 
  b = 2 ∧ q = -3 := by
sorry

end NUMINAMATH_CALUDE_geometric_progression_proof_l2436_243686


namespace NUMINAMATH_CALUDE_inequality_ratio_l2436_243699

theorem inequality_ratio (a b : ℝ) (h1 : a < b) (h2 : b < 0) : b / a < a / b := by
  sorry

end NUMINAMATH_CALUDE_inequality_ratio_l2436_243699


namespace NUMINAMATH_CALUDE_three_cubes_volume_and_area_l2436_243605

/-- Calculates the volume of a cube given its edge length -/
def cube_volume (edge_length : ℝ) : ℝ := edge_length ^ 3

/-- Calculates the surface area of a cube given its edge length -/
def cube_surface_area (edge_length : ℝ) : ℝ := 6 * edge_length ^ 2

/-- Theorem about the total volume and surface area of three cubic boxes -/
theorem three_cubes_volume_and_area (edge1 edge2 edge3 : ℝ) 
  (h1 : edge1 = 3) (h2 : edge2 = 5) (h3 : edge3 = 6) : 
  cube_volume edge1 + cube_volume edge2 + cube_volume edge3 = 368 ∧ 
  cube_surface_area edge1 + cube_surface_area edge2 + cube_surface_area edge3 = 420 := by
  sorry

#check three_cubes_volume_and_area

end NUMINAMATH_CALUDE_three_cubes_volume_and_area_l2436_243605


namespace NUMINAMATH_CALUDE_ravon_has_card_4_l2436_243637

structure Player where
  name : String
  score : Nat
  cards : Finset Nat

def card_set : Finset Nat := Finset.range 10

theorem ravon_has_card_4 (players : Finset Player)
  (h1 : players.card = 5)
  (h2 : ∀ p ∈ players, p.cards ⊆ card_set)
  (h3 : ∀ p ∈ players, p.cards.card = 2)
  (h4 : ∀ p ∈ players, p.score = (p.cards.sum id))
  (h5 : ∃ p ∈ players, p.name = "Ravon" ∧ p.score = 11)
  (h6 : ∃ p ∈ players, p.name = "Oscar" ∧ p.score = 4)
  (h7 : ∃ p ∈ players, p.name = "Aditi" ∧ p.score = 7)
  (h8 : ∃ p ∈ players, p.name = "Tyrone" ∧ p.score = 16)
  (h9 : ∃ p ∈ players, p.name = "Kim" ∧ p.score = 17)
  (h10 : ∀ c ∈ card_set, (players.filter (λ p => c ∈ p.cards)).card = 1) :
  ∃ p ∈ players, p.name = "Ravon" ∧ 4 ∈ p.cards :=
sorry

end NUMINAMATH_CALUDE_ravon_has_card_4_l2436_243637


namespace NUMINAMATH_CALUDE_fifth_subject_score_l2436_243650

/-- Given a student with 5 subject scores, prove that if 4 scores are known
    and the average of all 5 scores is 73, then the fifth score must be 85. -/
theorem fifth_subject_score
  (scores : Fin 5 → ℕ)
  (known_scores : scores 0 = 55 ∧ scores 1 = 67 ∧ scores 2 = 76 ∧ scores 3 = 82)
  (average : (scores 0 + scores 1 + scores 2 + scores 3 + scores 4) / 5 = 73) :
  scores 4 = 85 := by
sorry

end NUMINAMATH_CALUDE_fifth_subject_score_l2436_243650


namespace NUMINAMATH_CALUDE_f_properties_l2436_243685

noncomputable def f (x : ℝ) : ℝ := Real.log x - x^2 - x

theorem f_properties :
  -- Part 1: Monotonicity
  (∀ x ≥ 1, ∀ y ≥ x, f y ≤ f x) ∧
  -- Part 2: Inequality for a ≥ 2
  (∀ a ≥ 2, ∀ x > 0, f x < (a/2 - 1) * x^2 + a * x - 1) ∧
  -- Part 3: Inequality for x1 and x2
  (∀ x1 > 0, ∀ x2 > 0,
    f x1 + f x2 + 2 * (x1^2 + x2^2) + x1 * x2 = 0 →
    x1 + x2 ≥ (Real.sqrt 5 - 1) / 2) :=
by sorry

end NUMINAMATH_CALUDE_f_properties_l2436_243685


namespace NUMINAMATH_CALUDE_min_sum_given_product_l2436_243683

theorem min_sum_given_product (a b : ℝ) : 
  a > 0 → b > 0 → a * b = a + b + 3 → (∀ x y : ℝ, x > 0 → y > 0 → x * y = x + y + 3 → a + b ≤ x + y) → a + b = 6 :=
by sorry

end NUMINAMATH_CALUDE_min_sum_given_product_l2436_243683


namespace NUMINAMATH_CALUDE_solve_equation_l2436_243643

theorem solve_equation (x : ℚ) : (3 * x + 5) / 7 = 13 → x = 86 / 3 := by
  sorry

end NUMINAMATH_CALUDE_solve_equation_l2436_243643


namespace NUMINAMATH_CALUDE_bounds_on_y_l2436_243687

-- Define the equations
def eq1 (x y : ℝ) : Prop := x^2 - 6*x + 2*y = 0
def eq2 (x y : ℝ) : Prop := 3*x^2 + 12*x - 2*y - 4 = 0
def eq3 (x y : ℝ) : Prop := y = 2*x / (1 + x^2)
def eq4 (x y : ℝ) : Prop := y = (2*x - 1) / (x^2 + 2*x + 1)

-- Define the theorem
theorem bounds_on_y :
  ∀ x y : ℝ,
  eq1 x y ∧ eq2 x y ∧ eq3 x y ∧ eq4 x y →
  y ≤ 4.5 ∧ y ≥ -8 ∧ -1 ≤ y ∧ y ≤ 1 ∧ y ≤ 1/3 :=
by sorry

end NUMINAMATH_CALUDE_bounds_on_y_l2436_243687


namespace NUMINAMATH_CALUDE_enclosing_polygon_sides_l2436_243662

/-- Represents a regular polygon -/
structure RegularPolygon :=
  (sides : ℕ)

/-- Represents the enclosing arrangement -/
structure EnclosingArrangement :=
  (central : RegularPolygon)
  (enclosing : RegularPolygon)
  (num_enclosing : ℕ)

/-- Checks if the arrangement is symmetrical and without gaps or overlaps -/
def is_valid_arrangement (arr : EnclosingArrangement) : Prop :=
  arr.central.sides = arr.num_enclosing ∧
  arr.num_enclosing * (180 / arr.enclosing.sides) = arr.central.sides * (180 - (arr.central.sides - 2) * 180 / arr.central.sides) / 2

theorem enclosing_polygon_sides
  (arr : EnclosingArrangement)
  (h_valid : is_valid_arrangement arr)
  (h_central_sides : arr.central.sides = 15) :
  arr.enclosing.sides = 15 :=
sorry

end NUMINAMATH_CALUDE_enclosing_polygon_sides_l2436_243662


namespace NUMINAMATH_CALUDE_angle_D_is_60_l2436_243688

-- Define a quadrilateral
structure Quadrilateral :=
  (A B C D : Real)

-- Define the properties of the quadrilateral
def is_valid_quadrilateral (q : Quadrilateral) : Prop :=
  q.A + q.B + q.C + q.D = 360

-- Define the specific conditions of our quadrilateral
def special_quadrilateral (q : Quadrilateral) : Prop :=
  q.A + q.B = 180 ∧ q.C = 2 * q.D

-- Theorem statement
theorem angle_D_is_60 (q : Quadrilateral) 
  (h1 : is_valid_quadrilateral q) 
  (h2 : special_quadrilateral q) : 
  q.D = 60 := by
  sorry

end NUMINAMATH_CALUDE_angle_D_is_60_l2436_243688


namespace NUMINAMATH_CALUDE_expression_equals_zero_l2436_243638

theorem expression_equals_zero :
  (-1)^2023 - |1 - Real.sqrt 3| + Real.sqrt 6 * Real.sqrt (1/2) = 0 := by
  sorry

end NUMINAMATH_CALUDE_expression_equals_zero_l2436_243638


namespace NUMINAMATH_CALUDE_woman_birth_year_l2436_243666

/-- A woman born in the first half of the twentieth century was x years old in the year x^2. This theorem proves her birth year was 1892. -/
theorem woman_birth_year :
  ∃ (x : ℕ),
    (x^2 : ℕ) < 2000 ∧  -- Born in the first half of the 20th century
    (x^2 : ℕ) ≥ 1900 ∧  -- Born in the 20th century
    (x^2 - x : ℕ) = 1892  -- Birth year calculation
  := by sorry

#check woman_birth_year

end NUMINAMATH_CALUDE_woman_birth_year_l2436_243666


namespace NUMINAMATH_CALUDE_jewelry_store_profit_l2436_243659

/-- Calculates the gross profit for a pair of earrings -/
def earrings_gross_profit (purchase_price : ℚ) (markup_percentage : ℚ) (price_decrease_percentage : ℚ) : ℚ :=
  let initial_selling_price := purchase_price / (1 - markup_percentage)
  let price_decrease := initial_selling_price * price_decrease_percentage
  let final_selling_price := initial_selling_price - price_decrease
  final_selling_price - purchase_price

/-- Theorem stating the gross profit for the given scenario -/
theorem jewelry_store_profit :
  earrings_gross_profit 240 (25/100) (20/100) = 16 := by
  sorry

end NUMINAMATH_CALUDE_jewelry_store_profit_l2436_243659


namespace NUMINAMATH_CALUDE_sin_negative_600_degrees_l2436_243698

theorem sin_negative_600_degrees : Real.sin ((-600 : ℝ) * π / 180) = Real.sqrt 3 / 2 := by
  sorry

end NUMINAMATH_CALUDE_sin_negative_600_degrees_l2436_243698


namespace NUMINAMATH_CALUDE_polynomial_division_remainder_l2436_243674

theorem polynomial_division_remainder : ∃ q : Polynomial ℤ, 
  3 * X^4 + 16 * X^3 + 5 * X^2 - 36 * X + 58 = 
  (X^2 + 5 * X + 3) * q + (-28 * X + 55) := by
  sorry

end NUMINAMATH_CALUDE_polynomial_division_remainder_l2436_243674


namespace NUMINAMATH_CALUDE_fixed_point_theorem_l2436_243633

universe u

theorem fixed_point_theorem (S : Type u) [Nonempty S] (f : Set S → Set S) 
  (h : ∀ (X Y : Set S), X ⊆ Y → f X ⊆ f Y) :
  ∃ (A : Set S), f A = A := by
sorry

end NUMINAMATH_CALUDE_fixed_point_theorem_l2436_243633


namespace NUMINAMATH_CALUDE_quadratic_inequality_domain_l2436_243617

theorem quadratic_inequality_domain (a : ℝ) :
  (∀ x : ℝ, (x < 1 ∨ x > 5) → x^2 - 2*(a-2)*x + a > 0) ↔ a ∈ Set.Ioo 1 5 ∪ Set.Ioc 5 5 :=
sorry

end NUMINAMATH_CALUDE_quadratic_inequality_domain_l2436_243617


namespace NUMINAMATH_CALUDE_person_height_from_shadows_l2436_243625

/-- Given a tree and a person casting shadows under the same lighting conditions,
    calculate the person's height based on the tree's height and shadow lengths. -/
theorem person_height_from_shadows 
  (tree_height : ℝ) (tree_shadow : ℝ) (person_shadow : ℝ) 
  (tree_height_pos : tree_height > 0)
  (tree_shadow_pos : tree_shadow > 0)
  (person_shadow_pos : person_shadow > 0)
  (h_tree : tree_height = 40 ∧ tree_shadow = 10)
  (h_person_shadow : person_shadow = 15 / 12) -- Convert 15 inches to feet
  : (tree_height / tree_shadow) * person_shadow = 5 := by
  sorry

#check person_height_from_shadows

end NUMINAMATH_CALUDE_person_height_from_shadows_l2436_243625


namespace NUMINAMATH_CALUDE_min_tosses_for_heads_l2436_243695

theorem min_tosses_for_heads (p : ℝ) (h_p : p = 1/2) :
  ∃ n : ℕ, n ≥ 1 ∧
  (∀ k : ℕ, k ≥ n → 1 - p^k ≥ 15/16) ∧
  (∀ k : ℕ, k < n → 1 - p^k < 15/16) ∧
  n = 4 :=
sorry

end NUMINAMATH_CALUDE_min_tosses_for_heads_l2436_243695


namespace NUMINAMATH_CALUDE_base_of_equation_l2436_243671

theorem base_of_equation (x y : ℕ) (base : ℝ) : 
  x = 9 → 
  x - y = 9 → 
  (base ^ x) * (4 ^ y) = 19683 → 
  base = 3 := by
sorry

end NUMINAMATH_CALUDE_base_of_equation_l2436_243671


namespace NUMINAMATH_CALUDE_cylinder_not_unique_l2436_243672

theorem cylinder_not_unique (S V : ℝ) (h_pos_S : S > 0) (h_pos_V : V > 0)
  (h_inequality : S > 3 * (2 * π * V^2)^(1/3)) :
  ∃ (r₁ r₂ h₁ h₂ : ℝ),
    r₁ ≠ r₂ ∧
    2 * π * r₁ * h₁ + 2 * π * r₁^2 = S ∧
    2 * π * r₂ * h₂ + 2 * π * r₂^2 = S ∧
    π * r₁^2 * h₁ = V ∧
    π * r₂^2 * h₂ = V :=
by sorry

end NUMINAMATH_CALUDE_cylinder_not_unique_l2436_243672


namespace NUMINAMATH_CALUDE_new_books_bought_l2436_243655

/-- Given Kaleb's initial number of books, the number of books he sold, and his final number of books,
    prove that the number of new books he bought is equal to the difference between his final number
    of books and the number of books he had after selling some. -/
theorem new_books_bought (initial_books sold_books final_books : ℕ) :
  initial_books = 34 →
  sold_books = 17 →
  final_books = 24 →
  final_books - (initial_books - sold_books) = 7 :=
by sorry

end NUMINAMATH_CALUDE_new_books_bought_l2436_243655


namespace NUMINAMATH_CALUDE_cricket_team_throwers_l2436_243601

/-- Represents a cricket team with throwers and non-throwers -/
structure CricketTeam where
  total_players : ℕ
  throwers : ℕ
  right_handed : ℕ
  left_handed : ℕ

/-- Conditions for the cricket team problem -/
def valid_cricket_team (team : CricketTeam) : Prop :=
  team.total_players = 58 ∧
  team.throwers + team.right_handed + team.left_handed = team.total_players ∧
  team.throwers + team.right_handed = 51 ∧
  team.left_handed = (team.total_players - team.throwers) / 3

theorem cricket_team_throwers :
  ∀ team : CricketTeam, valid_cricket_team team → team.throwers = 37 := by
  sorry

end NUMINAMATH_CALUDE_cricket_team_throwers_l2436_243601


namespace NUMINAMATH_CALUDE_gaeun_taller_than_nana_l2436_243608

/-- Proves that Gaeun is taller than Nana by 0.5 centimeters -/
theorem gaeun_taller_than_nana :
  let nana_height_m : ℝ := 1.618
  let gaeun_height_cm : ℝ := 162.3
  let m_to_cm : ℝ := 100
  gaeun_height_cm - (nana_height_m * m_to_cm) = 0.5 := by sorry

end NUMINAMATH_CALUDE_gaeun_taller_than_nana_l2436_243608


namespace NUMINAMATH_CALUDE_arithmetic_sequence_term_count_l2436_243622

/-- 
Given an arithmetic sequence with:
- First term a = 2
- Last term l = 2008
- Common difference d = 3

Prove that the number of terms in the sequence is 669.
-/
theorem arithmetic_sequence_term_count : 
  ∀ (a l d n : ℕ), 
    a = 2 → 
    l = 2008 → 
    d = 3 → 
    l = a + (n - 1) * d → 
    n = 669 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_term_count_l2436_243622


namespace NUMINAMATH_CALUDE_inequality_proof_l2436_243665

theorem inequality_proof (a b : ℝ) 
  (h : ∀ x : ℝ, Real.cos (a * Real.sin x) > Real.sin (b * Real.cos x)) : 
  a^2 + b^2 < (Real.pi^2) / 4 := by
sorry

end NUMINAMATH_CALUDE_inequality_proof_l2436_243665


namespace NUMINAMATH_CALUDE_three_As_theorem_l2436_243692

-- Define the set of students
inductive Student : Type
  | Alan : Student
  | Beth : Student
  | Carlos : Student
  | Diana : Student
  | Emma : Student

-- Define a function to represent whether a student got an A
def gotA : Student → Prop := sorry

-- Define the implications stated by each student
axiom alan_implication : gotA Student.Alan → gotA Student.Beth
axiom beth_implication : gotA Student.Beth → (gotA Student.Carlos ∧ gotA Student.Emma)
axiom carlos_implication : gotA Student.Carlos → gotA Student.Diana
axiom diana_implication : gotA Student.Diana → gotA Student.Emma

-- Define a function to count how many students got an A
def count_A : (Student → Prop) → Nat := sorry

-- State the theorem
theorem three_As_theorem :
  (count_A gotA = 3) →
  ((gotA Student.Beth ∧ gotA Student.Carlos ∧ gotA Student.Emma) ∨
   (gotA Student.Carlos ∧ gotA Student.Diana ∧ gotA Student.Emma)) :=
by sorry

end NUMINAMATH_CALUDE_three_As_theorem_l2436_243692


namespace NUMINAMATH_CALUDE_range_of_c_l2436_243680

-- Define the propositions p and q
def p (c : ℝ) : Prop := ∀ x y : ℝ, x < y → c^x > c^y
def q (c : ℝ) : Prop := 1 - 2*c < 0

-- State the theorem
theorem range_of_c (c : ℝ) (h1 : c > 0) (h2 : c ≠ 1) 
  (h3 : (p c ∨ q c) ∧ ¬(p c ∧ q c)) : 
  (c ∈ Set.Ioc 0 (1/2)) ∨ (c ∈ Set.Ioi 1) := by
  sorry

end NUMINAMATH_CALUDE_range_of_c_l2436_243680


namespace NUMINAMATH_CALUDE_sum_of_common_ratios_is_seven_l2436_243627

/-- Given two nonconstant geometric sequences with first term k and different common ratios p and r,
    if a₃ - b₃ = 7(a₂ - b₂), then p + r = 7. -/
theorem sum_of_common_ratios_is_seven
  (k : ℝ) (p r : ℝ) (hp : p ≠ 1) (hr : r ≠ 1) (hpr : p ≠ r) :
  k * p^2 - k * r^2 = 7 * (k * p - k * r) → p + r = 7 := by
  sorry


end NUMINAMATH_CALUDE_sum_of_common_ratios_is_seven_l2436_243627


namespace NUMINAMATH_CALUDE_largest_quantity_l2436_243658

theorem largest_quantity (A B C : ℚ) : 
  A = 2006/2005 + 2006/2007 →
  B = 2006/2007 + 2008/2007 →
  C = 2007/2006 + 2007/2008 →
  A > B ∧ A > C := by
  sorry

end NUMINAMATH_CALUDE_largest_quantity_l2436_243658


namespace NUMINAMATH_CALUDE_impossible_cube_permutation_l2436_243604

/-- Represents a position in the 3x3x3 cube -/
structure Position :=
  (x y z : Fin 3)

/-- Represents a labeling of the 27 unit cubes -/
def Labeling := Fin 27 → Position

/-- Represents a move: swapping cube 27 with a neighbor -/
inductive Move
  | swap : Position → Move

/-- The parity of a position (even sum of coordinates is black, odd is white) -/
def Position.parity (p : Position) : Bool :=
  (p.x + p.y + p.z) % 2 = 0

/-- The final permutation required by the problem -/
def finalPermutation (n : Fin 27) : Fin 27 :=
  if n = 27 then 27 else 27 - n

/-- Theorem stating the impossibility of the required sequence of moves -/
theorem impossible_cube_permutation (initial : Labeling) :
  ¬ ∃ (moves : List Move), 
    (∀ n : Fin 27, 
      (initial n).parity = (initial (finalPermutation n)).parity) ∧
    (moves.length % 2 = 0) :=
  sorry

end NUMINAMATH_CALUDE_impossible_cube_permutation_l2436_243604


namespace NUMINAMATH_CALUDE_ellipse_equation_l2436_243649

-- Define the hyperbola E1
def E1 (x y : ℝ) : Prop := x^2 / 4 - y^2 / 5 = 1

-- Define the ellipse E2
def E2 (x y a b : ℝ) : Prop := x^2 / a^2 + y^2 / b^2 = 1

-- Define the condition that a > b > 0
def ellipse_condition (a b : ℝ) : Prop := a > b ∧ b > 0

-- Define the common focus condition
def common_focus (E1 E2 : ℝ → ℝ → Prop) : Prop := 
  ∃ (x y : ℝ), E1 x y ∧ E2 x y

-- Define the intersection condition
def intersect_in_quadrants (E1 E2 : ℝ → ℝ → Prop) : Prop :=
  ∃ (x1 y1 x2 y2 : ℝ), E1 x1 y1 ∧ E2 x1 y1 ∧ E1 x2 y2 ∧ E2 x2 y2 ∧
    x1 > 0 ∧ y1 > 0 ∧ x2 > 0 ∧ y2 < 0

-- Define the condition that chord MN passes through focus F2
def chord_through_focus (E1 E2 : ℝ → ℝ → Prop) : Prop :=
  ∃ (x1 y1 x2 y2 : ℝ), E1 x1 y1 ∧ E2 x1 y1 ∧ E1 x2 y2 ∧ E2 x2 y2 ∧
    (y1 - y2) / (x1 - x2) = (y1 - 0) / (x1 - 3)

theorem ellipse_equation :
  ∀ (a b : ℝ),
    ellipse_condition a b →
    common_focus E1 (E2 · · a b) →
    intersect_in_quadrants E1 (E2 · · a b) →
    chord_through_focus E1 (E2 · · a b) →
    a^2 = 81/4 ∧ b^2 = 45/4 :=
sorry

end NUMINAMATH_CALUDE_ellipse_equation_l2436_243649


namespace NUMINAMATH_CALUDE_special_geometric_sequence_a0_l2436_243623

/-- A geometric sequence with a special sum property -/
structure SpecialGeometricSequence where
  a : ℕ → ℝ
  isGeometric : ∀ n : ℕ, a (n + 2) * a n = (a (n + 1))^2
  sumProperty : ∀ n : ℕ, (Finset.range n).sum a = 5^(n + 1) + a 0 - 5

/-- The value of a₀ in a SpecialGeometricSequence is -5 -/
theorem special_geometric_sequence_a0 (seq : SpecialGeometricSequence) : seq.a 0 = -5 := by
  sorry


end NUMINAMATH_CALUDE_special_geometric_sequence_a0_l2436_243623


namespace NUMINAMATH_CALUDE_ceiling_squared_fraction_l2436_243676

theorem ceiling_squared_fraction : ⌈(-7/4)^2⌉ = 4 := by
  sorry

end NUMINAMATH_CALUDE_ceiling_squared_fraction_l2436_243676


namespace NUMINAMATH_CALUDE_length_AB_given_P_Q_positions_AB_length_is_189_l2436_243620

/-- Represents a point on a line segment -/
structure PointOnSegment (A B : ℝ) where
  x : ℝ
  h1 : A ≤ x
  h2 : x ≤ B

/-- Theorem: Length of AB given P and Q positions -/
theorem length_AB_given_P_Q_positions
  (A B : ℝ)
  (P : PointOnSegment A B)
  (Q : PointOnSegment A B)
  (h_same_side : (P.x - (A + B) / 2) * (Q.x - (A + B) / 2) > 0)
  (h_P_ratio : P.x - A = 3 / 7 * (B - A))
  (h_Q_ratio : Q.x - A = 4 / 9 * (B - A))
  (h_PQ_distance : |Q.x - P.x| = 3)
  : B - A = 189 := by
  sorry

/-- Corollary: AB length is 189 -/
theorem AB_length_is_189 : ∃ A B : ℝ, B - A = 189 ∧ 
  ∃ (P Q : PointOnSegment A B), 
    (P.x - (A + B) / 2) * (Q.x - (A + B) / 2) > 0 ∧
    P.x - A = 3 / 7 * (B - A) ∧
    Q.x - A = 4 / 9 * (B - A) ∧
    |Q.x - P.x| = 3 := by
  sorry

end NUMINAMATH_CALUDE_length_AB_given_P_Q_positions_AB_length_is_189_l2436_243620


namespace NUMINAMATH_CALUDE_probability_at_least_two_correct_l2436_243615

theorem probability_at_least_two_correct (n : ℕ) (p : ℚ) : 
  n = 6 → p = 1/6 → 
  1 - (Nat.choose n 0 * p^0 * (1-p)^n + Nat.choose n 1 * p^1 * (1-p)^(n-1)) = 34369/58420 := by
  sorry

end NUMINAMATH_CALUDE_probability_at_least_two_correct_l2436_243615


namespace NUMINAMATH_CALUDE_birch_count_l2436_243621

/-- Represents the number of trees of each species in the forest --/
structure ForestComposition where
  oak : ℕ
  pine : ℕ
  spruce : ℕ
  birch : ℕ

/-- Theorem stating the number of birch trees in the forest --/
theorem birch_count (forest : ForestComposition) : forest.birch = 2160 :=
  by
  have total_trees : forest.oak + forest.pine + forest.spruce + forest.birch = 4000 := by sorry
  have spruce_percentage : forest.spruce = 4000 * 10 / 100 := by sorry
  have pine_percentage : forest.pine = 4000 * 13 / 100 := by sorry
  have oak_count : forest.oak = forest.spruce + forest.pine := by sorry
  sorry


end NUMINAMATH_CALUDE_birch_count_l2436_243621


namespace NUMINAMATH_CALUDE_income_ratio_l2436_243607

/-- Proves that the ratio of A's monthly income to B's monthly income is 2.5:1 -/
theorem income_ratio (c_monthly : ℕ) (a_annual : ℕ) : 
  c_monthly = 15000 →
  a_annual = 504000 →
  (a_annual / 12 : ℚ) / ((1 + 12/100) * c_monthly) = 5/2 := by
  sorry

end NUMINAMATH_CALUDE_income_ratio_l2436_243607


namespace NUMINAMATH_CALUDE_triangle_area_l2436_243697

/-- The area of a triangle with vertices A(2, 2), B(8, 2), and C(4, 10) is 24 square units. -/
theorem triangle_area : 
  let A : ℝ × ℝ := (2, 2)
  let B : ℝ × ℝ := (8, 2)
  let C : ℝ × ℝ := (4, 10)
  let area := abs ((A.1 * (B.2 - C.2) + B.1 * (C.2 - A.2) + C.1 * (A.2 - B.2)) / 2)
  area = 24 := by sorry

end NUMINAMATH_CALUDE_triangle_area_l2436_243697


namespace NUMINAMATH_CALUDE_quadratic_discriminant_l2436_243642

/-- The discriminant of a quadratic equation ax² + bx + c is b² - 4ac -/
def discriminant (a b c : ℚ) : ℚ := b^2 - 4*a*c

/-- The coefficients of the quadratic equation 2x² + (2 + 1/2)x + 1/2 -/
def a : ℚ := 2
def b : ℚ := 5/2
def c : ℚ := 1/2

theorem quadratic_discriminant :
  discriminant a b c = 9/4 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_discriminant_l2436_243642


namespace NUMINAMATH_CALUDE_inequality_holds_l2436_243679

/-- An equilateral triangle with height 1 -/
structure EquilateralTriangle :=
  (height : ℝ)
  (height_eq_one : height = 1)

/-- A point inside the equilateral triangle -/
structure PointInTriangle (t : EquilateralTriangle) :=
  (x : ℝ)
  (y : ℝ)
  (z : ℝ)
  (sum_eq_height : x + y + z = t.height)
  (all_positive : x > 0 ∧ y > 0 ∧ z > 0)

/-- The inequality holds for any point inside the equilateral triangle -/
theorem inequality_holds (t : EquilateralTriangle) (p : PointInTriangle t) :
  p.x^2 + p.y^2 + p.z^2 ≥ p.x^3 + p.y^3 + p.z^3 + 6*p.x*p.y*p.z :=
sorry

end NUMINAMATH_CALUDE_inequality_holds_l2436_243679


namespace NUMINAMATH_CALUDE_product_sum_ratio_l2436_243673

theorem product_sum_ratio : (1 * 2 * 3 * 4 * 5 * 6) / (1 + 2 + 3 + 4 + 5 + 6) = 240 / 7 := by
  sorry

end NUMINAMATH_CALUDE_product_sum_ratio_l2436_243673


namespace NUMINAMATH_CALUDE_cosine_sum_lower_bound_l2436_243696

theorem cosine_sum_lower_bound (a b c : ℝ) :
  Real.cos (a - b) + Real.cos (b - c) + Real.cos (c - a) ≥ -3/2 := by
  sorry

end NUMINAMATH_CALUDE_cosine_sum_lower_bound_l2436_243696


namespace NUMINAMATH_CALUDE_max_large_chips_l2436_243640

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ m : ℕ, m > 1 → m < n → ¬(n % m = 0)

theorem max_large_chips :
  ∀ (small large prime : ℕ),
    small + large = 61 →
    small = large + prime →
    is_prime prime →
    large ≤ 29 :=
by sorry

end NUMINAMATH_CALUDE_max_large_chips_l2436_243640


namespace NUMINAMATH_CALUDE_convex_pentagon_arithmetic_angles_l2436_243681

/-- A convex pentagon with angles in arithmetic progression has each angle greater than 36° -/
theorem convex_pentagon_arithmetic_angles (α γ : ℝ) (h_convex : α + 4*γ < π) 
  (h_sum : 5*α + 10*γ = 3*π) : α > π/5 := by
  sorry

end NUMINAMATH_CALUDE_convex_pentagon_arithmetic_angles_l2436_243681


namespace NUMINAMATH_CALUDE_number_comparisons_l2436_243653

theorem number_comparisons : 
  (97430 < 100076) ∧ 
  (67500000 > 65700000) ∧ 
  (2648050 > 2648005) ∧ 
  (45000000 = 45000000) := by
  sorry

end NUMINAMATH_CALUDE_number_comparisons_l2436_243653


namespace NUMINAMATH_CALUDE_first_run_rate_l2436_243644

theorem first_run_rate (first_run_distance : ℝ) (second_run_distance : ℝ) 
  (second_run_rate : ℝ) (total_time : ℝ) :
  first_run_distance = 5 →
  second_run_distance = 4 →
  second_run_rate = 9.5 →
  total_time = 88 →
  first_run_distance * (total_time - second_run_distance * second_run_rate) / first_run_distance = 10 :=
by sorry

end NUMINAMATH_CALUDE_first_run_rate_l2436_243644


namespace NUMINAMATH_CALUDE_equation_solution_l2436_243614

theorem equation_solution : 
  ∃ x : ℝ, ((0.02^2 + 0.52^2 + 0.035^2) / (0.002^2 + 0.052^2 + x^2) = 100) ∧ x = 0.0035 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l2436_243614


namespace NUMINAMATH_CALUDE_inscribed_octagon_area_is_200_l2436_243669

/-- The area of an octagon inscribed in a square, where each vertex of the octagon
    bisects the sides of the square and the perimeter of the square is 80 centimeters. -/
def inscribedOctagonArea (square_perimeter : ℝ) (octagon_bisects_square : Prop) : ℝ :=
  sorry

/-- Theorem stating that the area of the inscribed octagon is 200 square centimeters. -/
theorem inscribed_octagon_area_is_200 :
  inscribedOctagonArea 80 true = 200 := by sorry

end NUMINAMATH_CALUDE_inscribed_octagon_area_is_200_l2436_243669


namespace NUMINAMATH_CALUDE_chess_games_ratio_l2436_243632

/-- Given a chess player who played 44 games and won 16 of them, 
    prove that the ratio of games lost to games won is 7:4 -/
theorem chess_games_ratio (total_games : ℕ) (games_won : ℕ) 
  (h1 : total_games = 44) (h2 : games_won = 16) :
  (total_games - games_won) / games_won = 7 / 4 := by
  sorry

end NUMINAMATH_CALUDE_chess_games_ratio_l2436_243632


namespace NUMINAMATH_CALUDE_jump_rope_total_l2436_243689

theorem jump_rope_total (taehyung_jumps_per_day : ℕ) (taehyung_days : ℕ) 
                        (namjoon_jumps_per_day : ℕ) (namjoon_days : ℕ) :
  taehyung_jumps_per_day = 56 →
  taehyung_days = 3 →
  namjoon_jumps_per_day = 35 →
  namjoon_days = 4 →
  taehyung_jumps_per_day * taehyung_days + namjoon_jumps_per_day * namjoon_days = 308 :=
by
  sorry

end NUMINAMATH_CALUDE_jump_rope_total_l2436_243689

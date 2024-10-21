import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_curve_and_line_equations_l813_81373

/-- Point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Line in 2D space -/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Circle in 2D space -/
structure Circle where
  center : Point
  radius : ℝ

noncomputable def distance (p1 p2 : Point) : ℝ :=
  Real.sqrt ((p1.x - p2.x)^2 + (p1.y - p2.y)^2)

def CircleMembership (p : Point) (c : Circle) : Prop :=
  (p.x - c.center.x)^2 + (p.y - c.center.y)^2 = c.radius^2

theorem curve_and_line_equations 
  (P : Point)
  (C : Circle)
  (l : Line)
  (A B : Point)
  (h1 : P.x = -3 ∧ P.y = -3/2)
  (h2 : C.center.x = 0 ∧ C.center.y = 0 ∧ C.radius = 5)
  (h3 : (l.a * P.x + l.b * P.y + l.c = 0) ∧ 
        (l.a * A.x + l.b * A.y + l.c = 0) ∧ 
        (l.a * B.x + l.b * B.y + l.c = 0))
  (h4 : A.x^2 + A.y^2 = C.radius^2 ∧ B.x^2 + B.y^2 = C.radius^2)
  (h5 : distance A B = 8) :
  (∀ (p : Point), p.x^2 + p.y^2 = 25 ↔ CircleMembership p C) ∧
  (l.a = 3 ∧ l.b = 0 ∧ l.c = 9) ∨ (l.a = 3 ∧ l.b = 4 ∧ l.c = 15) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_curve_and_line_equations_l813_81373


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_n_for_divisibility_l813_81300

theorem smallest_n_for_divisibility :
  (∃ n : ℕ, ∀ x y z : ℕ, x > 0 → y > 0 → z > 0 → x ∣ y^3 → y ∣ z^3 → z ∣ x^3 → x * y * z ∣ (x + y + z)^n) ∧
  (∀ m : ℕ, m < 13 → ∃ x y z : ℕ, x > 0 ∧ y > 0 ∧ z > 0 ∧ x ∣ y^3 ∧ y ∣ z^3 ∧ z ∣ x^3 ∧ ¬(x * y * z ∣ (x + y + z)^m)) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_n_for_divisibility_l813_81300


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_angle_B_l813_81314

theorem triangle_angle_B (A B C : ℝ) (a b c : ℝ) : 
  0 < A ∧ A < Real.pi → 
  0 < B ∧ B < Real.pi → 
  0 < C ∧ C < Real.pi → 
  a > 0 → b > 0 → c > 0 →
  Real.sin C = 2 * Real.sin A →
  b = Real.sqrt 3 * a →
  B = Real.pi / 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_angle_B_l813_81314


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_polynomial_value_at_neg_two_l813_81330

/-- A polynomial of degree 4 satisfying specific conditions -/
noncomputable def P : ℝ → ℝ :=
  fun x => (1 / 24) * (x - 1) * (x - 2) * (x - 3) * (x - 4) + x^2

/-- The theorem stating the properties of P and its value at -2 -/
theorem polynomial_value_at_neg_two :
  (∀ x : ℝ, P x = (1 / 24) * (x - 1) * (x - 2) * (x - 3) * (x - 4) + x^2) →
  P 0 = 1 →
  P 1 = 1 →
  P 2 = 4 →
  P 3 = 9 →
  P 4 = 16 →
  P (-2) = 19 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_polynomial_value_at_neg_two_l813_81330


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_negative_division_division_negative_81_27_l813_81368

theorem negative_division (a b : Int) (h : b ≠ 0) : 
  ((-a) / (-b) : Int) = a / b := by
  -- This proof is left as sorry for now, but it can be completed later if needed
  sorry

theorem division_negative_81_27 : ((-81) / (-27) : Int) = 3 := by
  have h : -27 ≠ 0 := by norm_num
  rw [negative_division 81 27]
  · norm_num
  · norm_num  -- Prove 27 ≠ 0


end NUMINAMATH_CALUDE_ERRORFEEDBACK_negative_division_division_negative_81_27_l813_81368


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_T_is_ellipse_l813_81360

/-- A point in a 2D plane -/
structure Point2D where
  x : ℝ
  y : ℝ

/-- Calculate the area of a triangle given its three vertices -/
noncomputable def area_triangle (A B C : Point2D) : ℝ := sorry

/-- Calculate the perimeter of a triangle given its three vertices -/
noncomputable def perimeter_triangle (A B C : Point2D) : ℝ := sorry

/-- The set T of points A forming triangles ABC with fixed area and perimeter -/
def T (B C : Point2D) : Set Point2D :=
  {A : Point2D | 
    area_triangle A B C = 2 ∧ 
    perimeter_triangle A B C = 10}

/-- Define what it means for a set to be an ellipse -/
def is_ellipse (S : Set Point2D) : Prop := sorry

theorem T_is_ellipse (B C : Point2D) : 
  is_ellipse (T B C) := by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_T_is_ellipse_l813_81360


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_point_ordinate_l813_81349

/-- The parabola is defined by the equation x^2 = 4y -/
def is_on_parabola (x y : ℝ) : Prop := x^2 = 4*y

/-- The focus of the parabola is at (0, 1) -/
def focus : ℝ × ℝ := (0, 1)

/-- The distance between two points -/
noncomputable def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

theorem parabola_point_ordinate (x y : ℝ) :
  is_on_parabola x y →
  distance (x, y) focus = 5 →
  y = 4 := by
  sorry

#check parabola_point_ordinate

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_point_ordinate_l813_81349


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_common_difference_l813_81378

/-- Given an arithmetic sequence {a_n} with first term a₁ and common difference d,
    S_n denotes the sum of the first n terms. -/
noncomputable def S (a₁ d : ℝ) (n : ℕ) : ℝ := n * a₁ + (n * (n - 1) / 2) * d

/-- Theorem stating that if S_3/3 - S_2/2 = 1 for an arithmetic sequence,
    then the common difference d equals 2. -/
theorem arithmetic_sequence_common_difference (a₁ d : ℝ) :
  S a₁ d 3 / 3 - S a₁ d 2 / 2 = 1 → d = 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_common_difference_l813_81378


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trajectory_is_parabola_l813_81321

/-- A point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- The distance between two points -/
noncomputable def distance (p1 p2 : Point) : ℝ :=
  Real.sqrt ((p1.x - p2.x)^2 + (p1.y - p2.y)^2)

/-- The distance from a point to a vertical line -/
def distanceToVerticalLine (p : Point) (lineX : ℝ) : ℝ :=
  |p.x - lineX|

/-- The condition that defines the trajectory of point P -/
noncomputable def satisfiesCondition (p : Point) : Prop :=
  distance p ⟨3, 0⟩ = distanceToVerticalLine p (-2) + 1

/-- The equation of the parabola -/
def isOnParabola (p : Point) : Prop :=
  p.y^2 = 12 * p.x

theorem trajectory_is_parabola :
  ∀ p : Point, satisfiesCondition p → isOnParabola p :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_trajectory_is_parabola_l813_81321


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_reciprocal_points_theorem_l813_81367

-- Define the concept of a reciprocal point
def has_reciprocal_point (f : ℝ → ℝ) (domain : Set ℝ) : Prop :=
  ∃ x ∈ domain, x * f x = 1

-- Define the given functions
noncomputable def f1 : ℝ → ℝ := λ x => -2 * x + 2 * Real.sqrt 2
noncomputable def f2 : ℝ → ℝ := λ x => Real.sin x
noncomputable def f3 : ℝ → ℝ := λ x => x + 1 / x
noncomputable def f4 : ℝ → ℝ := λ x => Real.exp x
noncomputable def f5 : ℝ → ℝ := λ x => -2 * Real.log x

-- Define the domains
def domain1 : Set ℝ := Set.univ
def domain2 : Set ℝ := Set.Icc 0 (2 * Real.pi)
def domain3 : Set ℝ := Set.Ioi 0
def domain4 : Set ℝ := Set.univ
def domain5 : Set ℝ := Set.Ioi 0

-- State the theorem
theorem reciprocal_points_theorem :
  has_reciprocal_point f1 domain1 ∧
  has_reciprocal_point f2 domain2 ∧
  ¬has_reciprocal_point f3 domain3 ∧
  has_reciprocal_point f4 domain4 ∧
  ¬has_reciprocal_point f5 domain5 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_reciprocal_points_theorem_l813_81367


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_decreasing_interval_l813_81390

-- Define the function f
def f (x : ℝ) : ℝ := x^2 - 2*x + 3

-- Theorem statement
theorem f_decreasing_interval :
  {x : ℝ | ∀ y, y < x → f y > f x} = Set.Iic 1 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_decreasing_interval_l813_81390


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_enclosed_area_is_nine_halves_l813_81305

-- Define the two curves
def curve1 (x : ℝ) : ℝ := 9 - x^2
def curve2 (x : ℝ) : ℝ := x + 7

-- Define the enclosed area
noncomputable def enclosed_area : ℝ := ∫ x in Set.Icc (-2) 1, curve1 x - curve2 x

-- Theorem statement
theorem enclosed_area_is_nine_halves : enclosed_area = 9/2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_enclosed_area_is_nine_halves_l813_81305


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_polar_to_cartesian_parabola_l813_81301

/-- The curve defined by r = 3 sin θ sec θ in polar coordinates is equivalent to x² = 3y in Cartesian coordinates -/
theorem polar_to_cartesian_parabola : 
  ∀ (r θ x y : ℝ), 
    r = 3 * Real.sin θ * (1 / Real.cos θ) → 
    x = r * Real.cos θ → 
    y = r * Real.sin θ → 
    x^2 = 3*y := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_polar_to_cartesian_parabola_l813_81301


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_has_zero_iff_f_greater_than_exp_neg_l813_81398

-- Define the function f(x)
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := Real.log x + a / x

-- State the theorems
theorem f_has_zero_iff (a : ℝ) :
  (∃ x > 0, f a x = 0) ↔ (0 < a ∧ a ≤ Real.exp (-1)) := by
  sorry

theorem f_greater_than_exp_neg (a : ℝ) :
  a ≥ 2 / Real.exp 1 → ∀ x > 0, f a x > Real.exp (-x) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_has_zero_iff_f_greater_than_exp_neg_l813_81398


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rectangle_ellipse_perimeter_l813_81351

/-- Area of a rectangle given its vertices -/
def area_rectangle (E F G H : ℝ × ℝ) : ℝ := sorry

/-- Area of an ellipse given its semi-major and semi-minor axes -/
def area_ellipse (a b : ℝ) : ℝ := sorry

/-- Predicate to check if a point is on an ellipse -/
def on_ellipse (P : ℝ × ℝ) (a b : ℝ) : Prop := sorry

/-- Predicate to check if two points are foci of an ellipse -/
def are_foci (F H : ℝ × ℝ) (a b : ℝ) : Prop := sorry

/-- Perimeter of a rectangle given its vertices -/
def perimeter_rectangle (E F G H : ℝ × ℝ) : ℝ := sorry

/-- Given a rectangle EFGH and an ellipse with specific properties, 
    prove that the perimeter of the rectangle is 8√2016 -/
theorem rectangle_ellipse_perimeter :
  ∀ (E F G H : ℝ × ℝ) (a b : ℝ),
    -- Rectangle area is 4032
    area_rectangle E F G H = 4032 →
    -- Ellipse area is 4032π
    area_ellipse a b = 4032 * Real.pi →
    -- E and G are on the ellipse
    on_ellipse E a b ∧ on_ellipse G a b →
    -- F and H are foci of the ellipse
    are_foci F H a b →
    -- Perimeter of rectangle EFGH is 8√2016
    perimeter_rectangle E F G H = 8 * Real.sqrt 2016 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_rectangle_ellipse_perimeter_l813_81351


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sandcastle_height_differences_l813_81304

/-- Calculates the absolute difference between two numbers -/
def heightDifference (a b : ℝ) : ℝ := abs (a - b)

/-- Proves that the height differences between sandcastles are correctly calculated -/
theorem sandcastle_height_differences 
  (miki_height sister_height sam_height : ℝ)
  (h1 : miki_height = 0.83)
  (h2 : sister_height = 0.5)
  (h3 : sam_height = 1.2) :
  (heightDifference miki_height sister_height = 0.33) ∧
  (heightDifference miki_height sam_height = 0.37) ∧
  (heightDifference sister_height sam_height = 0.7) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_sandcastle_height_differences_l813_81304


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l813_81361

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := 1/2 - (2^x) / (2^x + 1)

-- State the theorem
theorem f_properties :
  -- f is odd
  (∀ x, f (-x) = -f x) →
  -- f is decreasing
  (∀ x y, x < y → f x > f y) ∧
  -- Inequality holds for m ≥ 3
  (∀ m : ℝ, m ≥ 3 →
    ∀ x ∈ Set.Icc 3 9,
      f ((Real.log 3)^2 * x) + f (2 - m * Real.log 3 * x) ≥ 0) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l813_81361


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l813_81395

noncomputable def f (x : ℝ) := Real.sin (2 * x) - Real.sqrt 3 * (Real.cos x ^ 2 - Real.sin x ^ 2)

theorem f_properties :
  (∀ t, f ((11 * Real.pi) / 12 + t) = f ((11 * Real.pi) / 12 - t)) ∧
  (∀ t, f ((2 * Real.pi) / 3 + t) = -f ((2 * Real.pi) / 3 - t)) ∧
  (∀ x, f (x + Real.pi / 3) = f x) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l813_81395


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_det_special_matrix_l813_81386

theorem det_special_matrix (α β : ℝ) : 
  Matrix.det !![0, Real.cos α, Real.sin α; 
                -Real.cos α, 0, Real.cos β; 
                -Real.sin α, -Real.cos β, 0] = 0 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_det_special_matrix_l813_81386


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_quadrilateral_area_is_10_5_computed_area_l813_81354

/-- The area of a quadrilateral given its vertices -/
noncomputable def quadrilateralArea (v1 v2 v3 v4 : ℝ × ℝ) : ℝ :=
  let (x1, y1) := v1
  let (x2, y2) := v2
  let (x3, y3) := v3
  let (x4, y4) := v4
  (1/2) * abs ((x1*y2 + x2*y3 + x3*y4 + x4*y1) - (y1*x2 + y2*x3 + y3*x4 + y4*x1))

/-- Theorem: The area of the quadrilateral with vertices (2,1), (1,6), (5,5), and (7,7) is 10.5 -/
theorem quadrilateral_area_is_10_5 :
  quadrilateralArea (2, 1) (1, 6) (5, 5) (7, 7) = 10.5 := by
  sorry

-- Note: We can't use #eval here because the function is noncomputable
-- Instead, we can state a theorem about the computed value
theorem computed_area :
  ∃ (ε : ℝ), ε > 0 ∧ abs (quadrilateralArea (2, 1) (1, 6) (5, 5) (7, 7) - 10.5) < ε := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_quadrilateral_area_is_10_5_computed_area_l813_81354


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_stewart_farm_ratio_l813_81355

-- Define the given constants
def horse_food_per_day : ℕ := 230
def total_horse_food : ℕ := 12880
def num_sheep : ℕ := 48

-- Define the number of horses
def num_horses : ℕ := total_horse_food / horse_food_per_day

-- Define the ratio of sheep to horses
def sheep_to_horse_ratio : ℚ := (num_sheep : ℚ) / (num_horses : ℚ)

-- Theorem to prove
theorem stewart_farm_ratio : sheep_to_horse_ratio = 6 / 7 := by
  -- Unfold the definitions
  unfold sheep_to_horse_ratio
  unfold num_horses
  -- Perform the calculation
  norm_num
  -- The proof is complete
  rfl

end NUMINAMATH_CALUDE_ERRORFEEDBACK_stewart_farm_ratio_l813_81355


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_statement_l813_81353

open Set

def A : Set ℝ := {x | x^2 - 3*x + 2 > 0}

def B (a : ℝ) : Set ℝ := {x | x^2 - (a+1)*x + a ≤ 0}

theorem problem_statement (a : ℝ) (h : a > 1) :
  (Set.univ \ A) ∪ B a = B a → a ≥ 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_statement_l813_81353


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_blue_chips_l813_81339

/-- Represents the number of chips in the box -/
def total_chips : ℕ := 75

/-- Represents the number of blue chips -/
def blue_chips : ℕ → ℕ := sorry

/-- Represents the number of red chips -/
def red_chips : ℕ → ℕ := sorry

/-- Represents the product of two distinct prime numbers -/
def prime_product : ℕ → ℕ := sorry

/-- The main theorem stating the maximum number of blue chips -/
theorem max_blue_chips :
  ∀ n : ℕ,
  (blue_chips n + red_chips n = total_chips) →
  (∃ p q : ℕ, Nat.Prime p ∧ Nat.Prime q ∧ p ≠ q ∧ prime_product n = p * q) →
  (red_chips n = blue_chips n + prime_product n) →
  blue_chips n ≤ 20 := by
  sorry

#check max_blue_chips

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_blue_chips_l813_81339


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_range_MN_l813_81303

-- Define the curves C1 and C2
noncomputable def C1 (φ : ℝ) : ℝ × ℝ := (2 * Real.cos φ, Real.sin φ)

def C2_center : ℝ × ℝ := (0, 3)
def C2_radius : ℝ := 1

-- Define a point on C1
noncomputable def M (φ : ℝ) : ℝ × ℝ := C1 φ

-- Define a point on C2
noncomputable def N (θ : ℝ) : ℝ × ℝ :=
  (C2_center.1 + C2_radius * Real.cos θ, C2_center.2 + C2_radius * Real.sin θ)

-- Define the distance between two points
noncomputable def distance (p q : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2)

-- Theorem statement
theorem distance_range_MN :
  ∀ φ θ : ℝ, 1 ≤ distance (M φ) (N θ) ∧ distance (M φ) (N θ) ≤ 5 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_range_MN_l813_81303


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_shales_method_l813_81379

-- Define the types for points and circles
variable (Point Circle : Type)

-- Define the necessary geometric relations
variable (on_circle : Point → Circle → Prop)
variable (center : Circle → Point)
variable (intersect : Circle → Circle → Point)
variable (line_intersect : Point → Point → Circle → Point)
variable (distance : Point → Point → ℝ)

-- Define the given conditions
variable (C : Circle)
variable (O P Q R L : Point)
variable (D : Circle)

-- State the theorem
theorem shales_method 
  (h1 : on_circle O C)
  (h2 : center D = O)
  (h3 : intersect D C = P)
  (h4 : intersect D C = Q)
  (h5 : ∃ E : Circle, center E = Q ∧ 
        ∃ S : Point, intersect D E = S ∧ 
        on_circle S C ∧ 
        R = S)
  (h6 : line_intersect P R C = L) :
  ∃ X : Point, center C = X ∧ 
  distance X L = distance X Q :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_shales_method_l813_81379


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_value_in_second_quadrant_l813_81310

theorem tan_value_in_second_quadrant (x : ℝ) 
  (h1 : Real.sin x = Real.sqrt 5 / 5) 
  (h2 : x ∈ Set.Ioo (Real.pi / 2) (3 * Real.pi / 2)) : 
  Real.tan x = -1/2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_value_in_second_quadrant_l813_81310


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_log_difference_implies_a_l813_81362

theorem log_difference_implies_a (a : ℝ) (h1 : 0 < a) (h2 : a < 1) 
  (h3 : Real.log 2 / Real.log a - Real.log 4 / Real.log a = 2) : a = Real.sqrt 2 / 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_log_difference_implies_a_l813_81362


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_only_group3_is_obtuse_l813_81363

/-- Check if three numbers can form an obtuse triangle -/
def is_obtuse_triangle (a b c : ℝ) : Prop :=
  let sides := [a, b, c]
  let max_side := max (max a b) c
  let other_sides := sides.filter (· ≠ max_side)
  let x := other_sides[0]!
  let y := other_sides[1]!
  (x^2 + y^2 - max_side^2) / (2 * x * y) < 0

/-- The given groups of numbers -/
def group1 : List ℝ := [6, 7, 8]
def group2 : List ℝ := [7, 8, 10]
def group3 : List ℝ := [2, 6, 7]
def group4 : List ℝ := [5, 12, 13]

/-- Theorem: Only group3 forms an obtuse triangle -/
theorem only_group3_is_obtuse :
  (¬ is_obtuse_triangle group1[0]! group1[1]! group1[2]!) ∧
  (¬ is_obtuse_triangle group2[0]! group2[1]! group2[2]!) ∧
  (is_obtuse_triangle group3[0]! group3[1]! group3[2]!) ∧
  (¬ is_obtuse_triangle group4[0]! group4[1]! group4[2]!) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_only_group3_is_obtuse_l813_81363


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_imaginary_part_of_specific_complex_max_distance_from_unit_circle_l813_81384

-- Define the complex number type
variable (z w : ℂ)

-- Theorem B
theorem imaginary_part_of_specific_complex :
  z = Complex.I * (1 - 2 * Complex.I) → Complex.im z = 1 :=
by sorry

-- Theorem D
theorem max_distance_from_unit_circle :
  Complex.abs z = 1 → 
  (∀ w, Complex.abs w = 1 → Complex.abs (w + 1) ≤ 2) ∧
  (∃ w, Complex.abs w = 1 ∧ Complex.abs (w + 1) = 2) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_imaginary_part_of_specific_complex_max_distance_from_unit_circle_l813_81384


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cube_face_product_sum_max_l813_81392

def cube_face_sum (faces : Finset (Fin 6 → ℕ)) : Prop :=
  ∀ f ∈ faces, (f 0 + f 1 = 10 ∨ f 0 + f 1 = 11) ∧
               (f 2 + f 3 = 10 ∨ f 2 + f 3 = 11) ∧
               (f 4 + f 5 = 10 ∨ f 4 + f 5 = 11)

def cube_face_values (faces : Finset (Fin 6 → ℕ)) : Prop :=
  ∀ f ∈ faces, (Finset.image f (Finset.univ : Finset (Fin 6))).toSet = {2, 3, 4, 6, 7, 8}

theorem cube_face_product_sum_max (faces : Finset (Fin 6 → ℕ))
  (h1 : cube_face_sum faces) (h2 : cube_face_values faces) :
  (faces.sup fun f => (f 0 + f 1) * (f 2 + f 3) * (f 4 + f 5)) ≤ 1210 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cube_face_product_sum_max_l813_81392


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_lines_intersect_lines_not_parallel_or_coincident_l813_81338

/-- Two lines in the xy-plane, parameterized by m -/
noncomputable def line1 (m : ℝ) := {(x, y) : ℝ × ℝ | 3 * x + 2 * y + m = 0}
noncomputable def line2 (m : ℝ) := {(x, y) : ℝ × ℝ | (m^2 + 1) * x - 3 * y - 3 * m = 0}

/-- The slope of line1 -/
noncomputable def slope1 : ℝ := -3 / 2

/-- The slope of line2 -/
noncomputable def slope2 (m : ℝ) : ℝ := (m^2 + 1) / 3

/-- Theorem: The two lines are intersecting for all real values of m -/
theorem lines_intersect (m : ℝ) : slope1 ≠ slope2 m := by
  sorry

/-- Corollary: The two lines are not parallel or coincident for any real value of m -/
theorem lines_not_parallel_or_coincident (m : ℝ) : 
  ¬(line1 m = line2 m ∨ ∃ k, ∀ (x y : ℝ), (x, y) ∈ line1 m ↔ (x + k, y) ∈ line2 m) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_lines_intersect_lines_not_parallel_or_coincident_l813_81338


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ferry_distance_proof_l813_81344

-- Define the ferry speed and current speed
noncomputable def ferry_speed : ℝ := 40
noncomputable def current_speed : ℝ := 24

-- Define the distance between A and B
noncomputable def distance_AB : ℝ := 70

-- Define the midpoint C
noncomputable def distance_AC : ℝ := distance_AB / 2

-- Define the downstream and upstream speeds
noncomputable def downstream_speed : ℝ := ferry_speed + current_speed
noncomputable def upstream_speed : ℝ := ferry_speed - current_speed

-- Define the usual travel times
noncomputable def usual_downstream_time : ℝ := distance_AB / downstream_speed
noncomputable def usual_upstream_time : ℝ := distance_AB / upstream_speed

-- Define the odd day incident time
noncomputable def odd_day_incident_time : ℝ := distance_AC / downstream_speed + (distance_AB - distance_AC) / current_speed

-- Define the even day incident time
noncomputable def even_day_incident_time : ℝ := (distance_AB - distance_AC) / upstream_speed + 1 + distance_AC / (2 * upstream_speed)

-- Theorem to prove
theorem ferry_distance_proof :
  (odd_day_incident_time = 43/18 * usual_downstream_time) ∧
  (even_day_incident_time = usual_upstream_time) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ferry_distance_proof_l813_81344


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sine_function_properties_l813_81320

noncomputable def sine_function (ω : ℕ+) (x : ℝ) : ℝ := Real.sin (ω * x)

theorem sine_function_properties (ω : ℕ+) :
  (∃ (a b : ℝ), a < b ∧ b - a = 1 ∧ 
    (∀ x ∈ Set.Icc a b, sine_function ω x ≤ 1) ∧
    (∃ (x₁ x₂ : ℝ), x₁ ∈ Set.Icc a b ∧ x₂ ∈ Set.Icc a b ∧ x₁ ≠ x₂ ∧
      sine_function ω x₁ = 1 ∧ sine_function ω x₂ = 1)) ∧
  (∀ x y : ℝ, -π/16 ≤ x ∧ x < y ∧ y ≤ π/15 → sine_function ω x < sine_function ω y) →
  ω = 8 := by sorry

#check sine_function_properties

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sine_function_properties_l813_81320


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_notebooks_given_to_tom_approx_l813_81397

-- Define the initial number of notebooks
noncomputable def initial_red : ℝ := 15
noncomputable def initial_blue : ℝ := 17
noncomputable def initial_white : ℝ := 19

-- Define the actions taken on each day
noncomputable def day1_red_to_tom : ℝ := 4.5
noncomputable def day1_blue_fraction_to_tom : ℝ := 1/3

noncomputable def day2_white_fraction_given : ℝ := 1/2
noncomputable def day2_blue_fraction_given : ℝ := 1/4

noncomputable def day3_red_given : ℝ := 3.5
noncomputable def day3_blue_fraction_given : ℝ := 2/5
noncomputable def day3_white_fraction_kept : ℝ := 1/4

-- Define the final number of notebooks left
noncomputable def final_notebooks_left : ℝ := 5

-- Theorem to prove
theorem notebooks_given_to_tom_approx (ε : ℝ) (hε : ε > 0) :
  ∃ (x : ℝ), abs (x - 10.17) < ε ∧ 
  x = day1_red_to_tom + day1_blue_fraction_to_tom * initial_blue :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_notebooks_given_to_tom_approx_l813_81397


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_all_students_same_canteen_l813_81369

/-- The probability that all three students choose the same canteen -/
noncomputable def prob_same_canteen : ℝ := 1/4

/-- The number of canteens -/
def num_canteens : ℕ := 2

/-- The probability of a student choosing a specific canteen -/
noncomputable def prob_choose_canteen : ℝ := 1/num_canteens

theorem all_students_same_canteen :
  prob_same_canteen = prob_choose_canteen^3 * num_canteens := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_all_students_same_canteen_l813_81369


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_barrel_filling_trips_l813_81329

noncomputable section

/-- The radius of both the barrel and the bucket in inches -/
def r : ℝ := 10

/-- The height of the barrel in inches -/
def h : ℝ := 15

/-- The volume of the hemispherical bucket -/
noncomputable def bucket_volume : ℝ := (2/3) * Real.pi * r^3

/-- The volume of the cylindrical barrel -/
noncomputable def barrel_volume : ℝ := Real.pi * r^2 * h

/-- The number of trips needed to fill the barrel -/
def trips_needed : ℕ := 3

theorem barrel_filling_trips :
  ⌈barrel_volume / bucket_volume⌉ = trips_needed :=
sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_barrel_filling_trips_l813_81329


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_y_intercept_l813_81399

-- Define the circle
def circle_eq (x y : ℝ) : Prop := x^2 + y^2 + 4*x - 2*y + 3 = 0

-- Define the point P
def P : ℝ × ℝ := (-1, 0)

-- Define a line passing through P
def line_through_P (m : ℝ) (x y : ℝ) : Prop := y - P.2 = m * (x - P.1)

-- Define the tangent condition
def is_tangent (m : ℝ) : Prop := ∃ (x y : ℝ), circle_eq x y ∧ line_through_P m x y

-- Main theorem
theorem tangent_y_intercept :
  ∃ (m : ℝ), is_tangent m → 
    ∃ (b : ℝ), b = 1 ∧ line_through_P m 0 b := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_y_intercept_l813_81399


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_of_f_range_of_a_l813_81317

-- Define the function f
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a * Real.exp x - x + Real.log a - 2

-- Define the function g
noncomputable def g (a : ℝ) (x : ℝ) : ℝ := f a x + x - Real.log (x + 2)

-- Theorem for part (I)
theorem min_value_of_f (a : ℝ) :
  (∃ x : ℝ, x = 0 ∧ (deriv (f a)) x = 0) →
  ∃ x : ℝ, ∀ y : ℝ, f a x ≤ f a y ∧ f a x = -1 :=
by
  sorry

-- Theorem for part (II)
theorem range_of_a :
  (∃ a : ℝ, ∃ x y : ℝ, x ≠ y ∧ g a x = 0 ∧ g a y = 0) →
  ∃ a : ℝ, 0 < a ∧ a < Real.exp 1 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_of_f_range_of_a_l813_81317


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_equilateral_triangle_side_length_l813_81393

/-- An equilateral triangle with a point inside it -/
structure EquilateralTriangleWithPoint where
  /-- The side length of the equilateral triangle -/
  side_length : ℝ
  /-- The point inside the triangle -/
  point : ℝ × ℝ
  /-- The perpendicular distance from the point to the first side -/
  dist1 : ℝ
  /-- The perpendicular distance from the point to the second side -/
  dist2 : ℝ
  /-- The perpendicular distance from the point to the third side -/
  dist3 : ℝ
  /-- The triangle is equilateral -/
  equilateral : side_length > 0
  /-- The point is inside the triangle -/
  point_inside : dist1 > 0 ∧ dist2 > 0 ∧ dist3 > 0
  /-- The sum of areas of three smaller triangles equals the area of the whole triangle -/
  area_equality : (1/2) * side_length * (dist1 + dist2 + dist3) = (Real.sqrt 3 / 4) * side_length^2

/-- The main theorem -/
theorem equilateral_triangle_side_length 
  (triangle : EquilateralTriangleWithPoint)
  (h1 : triangle.dist1 = 2)
  (h2 : triangle.dist2 = 4)
  (h3 : triangle.dist3 = 5) :
  triangle.side_length = 6 * Real.sqrt 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_equilateral_triangle_side_length_l813_81393


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_heroes_total_games_l813_81340

theorem heroes_total_games : ℕ := by
  -- Define the given percentages and game counts
  let initial_win_percentage : ℚ := 40 / 100
  let final_win_percentage : ℚ := 52 / 100
  let district_wins : ℕ := 8
  let district_losses : ℕ := 3

  -- Introduce variables for the initial and total games and wins
  let initial_games : ℕ := sorry
  let initial_wins : ℕ := sorry
  let total_games : ℕ := sorry
  let total_wins : ℕ := sorry

  -- State the relationships between the variables
  have h1 : (initial_wins : ℚ) / initial_games = initial_win_percentage := sorry
  have h2 : total_games = initial_games + district_wins + district_losses := sorry
  have h3 : total_wins = initial_wins + district_wins := sorry
  have h4 : (total_wins : ℚ) / total_games = final_win_percentage := sorry

  -- The final answer
  exact 30

-- Add this line to make the theorem opaque
attribute [local instance] heroes_total_games

end NUMINAMATH_CALUDE_ERRORFEEDBACK_heroes_total_games_l813_81340


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_area_AOB_l813_81374

/-- Definition of the ellipse C -/
def ellipse_C (x y : ℝ) : Prop := y^2 / 4 + x^2 = 1

/-- Definition of the accompanying circle -/
def accompanying_circle (x y : ℝ) : Prop := x^2 + y^2 = 1

/-- Area of triangle AOB as a function of m -/
noncomputable def S_AOB (m : ℝ) : ℝ := 2 * Real.sqrt 3 * |m| / (m^2 + 3)

/-- Theorem stating the maximum area of triangle AOB -/
theorem max_area_AOB :
  ∀ m : ℝ, |m| ≥ 1 →
  S_AOB m ≤ 1 ∧
  (S_AOB m = 1 ↔ m = Real.sqrt 3 ∨ m = -Real.sqrt 3) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_area_AOB_l813_81374


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rotated_semicircle_area_theorem_l813_81377

/-- The area of a figure formed by rotating a semicircle of radius R around one of its endpoints by an angle of 30° -/
noncomputable def rotated_semicircle_area (R : ℝ) : ℝ :=
  (Real.pi * R^2) / 3

/-- Theorem stating that the area of the rotated semicircle figure is πR²/3 -/
theorem rotated_semicircle_area_theorem (R : ℝ) (R_pos : R > 0) :
  rotated_semicircle_area R = (Real.pi * R^2) / 3 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_rotated_semicircle_area_theorem_l813_81377


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_race_result_l813_81306

/-- Represents a runner in the race -/
structure Runner where
  speed : ℝ
  time : ℝ

/-- The race scenario -/
structure RaceScenario where
  distance : ℝ
  runner_a : Runner
  runner_b : Runner
  lead_distance : ℝ

theorem race_result (race : RaceScenario) :
  race.distance = 80 ∧
  race.runner_a.time = 3 ∧
  race.lead_distance = 56 →
  race.runner_b.time - race.runner_a.time = 7 := by
  intro h
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_race_result_l813_81306


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_arctan_equation_l813_81325

theorem sin_arctan_equation (x : ℝ) (h1 : x > 0) (h2 : Real.sin (Real.arctan (2 * x)) = x) : x^2 = 3/4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_arctan_equation_l813_81325


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_multiplication_properties_l813_81358

class MultiplicationStructure (S : Type) where
  mul : S → S → S
  add : S → S → S
  one : S

variable {S : Type} [MultiplicationStructure S]

theorem multiplication_properties :
  (∀ (x y z : S), (MultiplicationStructure.mul (MultiplicationStructure.mul x y) z) = MultiplicationStructure.mul x (MultiplicationStructure.mul y z)) ∧
  (∀ (x y : S), MultiplicationStructure.mul x y = MultiplicationStructure.mul y x) ∧
  (∀ (x y z : S), MultiplicationStructure.mul x (MultiplicationStructure.add y z) = 
    MultiplicationStructure.add (MultiplicationStructure.mul x y) (MultiplicationStructure.mul x z)) ∧
  (∃ (e : S), ∀ (x : S), MultiplicationStructure.mul x e = x ∧ MultiplicationStructure.mul e x = x) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_multiplication_properties_l813_81358


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_portfolio_weighted_average_yield_l813_81315

/-- Represents a stock with its face value, market price, and yield -/
structure Stock where
  faceValue : ℚ
  marketPrice : ℚ
  yield : ℚ

/-- Represents an investment in a stock -/
structure Investment where
  stock : Stock
  amount : ℚ

/-- Calculates the weighted average yield of a portfolio -/
noncomputable def weightedAverageYield (investments : List Investment) : ℚ :=
  let totalInvestment := investments.foldl (fun acc inv => acc + inv.amount) 0
  investments.foldl (fun acc inv => acc + (inv.amount / totalInvestment) * inv.stock.yield) 0

theorem portfolio_weighted_average_yield :
  let stockA : Stock := { faceValue := 1000, marketPrice := 1200, yield := 18/100 }
  let stockB : Stock := { faceValue := 1000, marketPrice := 800, yield := 22/100 }
  let stockC : Stock := { faceValue := 1000, marketPrice := 1000, yield := 15/100 }
  let investments : List Investment := [
    { stock := stockA, amount := 5000 },
    { stock := stockB, amount := 3000 },
    { stock := stockC, amount := 2000 }
  ]
  weightedAverageYield investments = 93/500 := by
  sorry

#eval (93:ℚ)/500 -- To verify the decimal representation

end NUMINAMATH_CALUDE_ERRORFEEDBACK_portfolio_weighted_average_yield_l813_81315


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_midpoint_slope_theorem_l813_81389

-- Define the midpoint of a segment
def my_midpoint (x₁ y₁ x₂ y₂ : ℚ) : ℚ × ℚ :=
  ((x₁ + x₂) / 2, (y₁ + y₂) / 2)

-- Define the slope between two points
def my_slope (x₁ y₁ x₂ y₂ : ℚ) : ℚ :=
  (y₂ - y₁) / (x₂ - x₁)

-- Theorem statement
theorem midpoint_slope_theorem :
  let m₁ := my_midpoint 1 2 3 5
  let m₂ := my_midpoint 4 1 7 7
  my_slope m₁.1 m₁.2 m₂.1 m₂.2 = 1 / 7 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_midpoint_slope_theorem_l813_81389


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_mean_of_sequence_l813_81328

def sequenceNum (n : ℕ) : ℕ := 
  (10^n - 1) * 8 / 9

def arithmetic_mean (s : Finset ℕ) (f : ℕ → ℕ) : ℚ :=
  (s.sum f) / s.card

theorem arithmetic_mean_of_sequence :
  let s := Finset.range 9
  let mean := arithmetic_mean s sequenceNum
  (Int.floor mean).toNat.digits 10 = [2, 1, 0, 9, 2, 6, 9, 0, 1] := by sorry

#eval (Int.floor (arithmetic_mean (Finset.range 9) sequenceNum)).toNat.digits 10

end NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_mean_of_sequence_l813_81328


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_has_two_extrema_l813_81371

/-- The function f(x) defined in terms of parameter a -/
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := Real.exp x * (x - a * Real.exp x)

/-- The derivative of f(x) with respect to x -/
noncomputable def f_deriv (a : ℝ) (x : ℝ) : ℝ := Real.exp x * (1 + x - 2 * a * Real.exp x)

/-- Theorem stating the condition for f to have exactly two extrema points -/
theorem f_has_two_extrema (a : ℝ) : 
  (∃ x₁ x₂ : ℝ, x₁ < x₂ ∧ 
    f_deriv a x₁ = 0 ∧ 
    f_deriv a x₂ = 0 ∧ 
    (∀ x : ℝ, f_deriv a x = 0 → x = x₁ ∨ x = x₂)) ↔ 
  (0 < a ∧ a < 1/2) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_has_two_extrema_l813_81371


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ten_faucets_fill_time_l813_81334

/-- The time it takes for ten faucets to fill a 50-gallon tub, given that five faucets fill a 125-gallon tub in 8 minutes, assuming all faucets dispense water at the same rate. -/
theorem ten_faucets_fill_time : ℝ := by
  -- Define the given conditions
  let tub_size_five : ℝ := 125  -- Size of tub filled by five faucets (in gallons)
  let time_five : ℝ := 8        -- Time taken by five faucets (in minutes)
  let num_faucets_five : ℝ := 5 -- Number of faucets in the first scenario
  let tub_size_ten : ℝ := 50    -- Size of tub to be filled by ten faucets (in gallons)
  let num_faucets_ten : ℝ := 10 -- Number of faucets in the second scenario

  -- Calculate the fill time for ten faucets
  let fill_time := (tub_size_ten * time_five * num_faucets_five) / (tub_size_five * num_faucets_ten)

  -- Prove that the fill time is equal to 1.6 minutes
  have h : fill_time = 1.6 := by
    -- Proof steps would go here
    sorry

  exact fill_time


end NUMINAMATH_CALUDE_ERRORFEEDBACK_ten_faucets_fill_time_l813_81334


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_properties_l813_81323

def a : Fin 3 → ℝ := ![1, -1, 2]
def b : Fin 3 → ℝ := ![-2, 2, -4]

theorem vector_properties :
  let magnitude := Real.sqrt (a 0^2 + a 1^2 + a 2^2)
  let collinear := ∃ k : ℝ, ∀ i : Fin 3, b i = k * a i
  let symmetric_x := ![a 0, -a 1, -a 2]
  let symmetric_yOz := ![-a 0, a 1, a 2]
  (magnitude = Real.sqrt 6) ∧
  collinear ∧
  symmetric_x = ![1, 1, -2] ∧
  symmetric_yOz = ![-1, -1, 2] := by
  sorry

#eval a
#eval b

end NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_properties_l813_81323


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_domain_l813_81365

noncomputable def f (x : ℝ) := Real.log (3 - 4 * Real.sin x ^ 2)

def domain (x : ℝ) : Prop :=
  ∃ k : ℤ, (2 * k * Real.pi - Real.pi / 3 < x ∧ x < 2 * k * Real.pi + Real.pi / 3) ∨
           (2 * k * Real.pi + 2 * Real.pi / 3 < x ∧ x < 2 * k * Real.pi + 4 * Real.pi / 3)

theorem f_domain :
  ∀ x : ℝ, f x ∈ Set.univ ↔ domain x :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_domain_l813_81365


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_lawn_mowing_l813_81316

theorem lawn_mowing (mary_time tom_time tom_work_time : ℚ) 
  (h1 : mary_time = 5)
  (h2 : tom_time = 6)
  (h3 : tom_work_time = 3) : 
  1 - (tom_work_time / tom_time) = (1 : ℚ) / 2 := by
  sorry

#check lawn_mowing

end NUMINAMATH_CALUDE_ERRORFEEDBACK_lawn_mowing_l813_81316


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_theorem_l813_81311

/-- Definition of the ellipse C -/
def ellipse_C (x y a b : ℝ) : Prop := x^2 / a^2 + y^2 / b^2 = 1

/-- Conditions for the ellipse -/
structure EllipseConditions where
  a : ℝ
  b : ℝ
  c : ℝ
  h1 : a > b
  h2 : b > 0
  h3 : b > c
  h4 : 2 * a = 4
  h5 : c * b = Real.sqrt 3

/-- The main theorem -/
theorem ellipse_theorem (ec : EllipseConditions) :
  (∀ x y, ellipse_C x y ec.a ec.b ↔ x^2 / 4 + y^2 / 3 = 1) ∧
  ∃ P : ℝ × ℝ, P = (1, 0) ∧
    ∀ k m x₀ y₀,
      (ellipse_C x₀ y₀ ec.a ec.b ∧ y₀ = k * x₀ + m) →
      let N := (4, k * 4 + m)
      (P.fst - x₀) * (N.fst - x₀) + (P.snd - y₀) * (N.snd - y₀) = 0 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_theorem_l813_81311


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trigonometric_values_for_given_cosine_l813_81347

theorem trigonometric_values_for_given_cosine (θ : Real) 
  (h1 : Real.cos θ = 12/13) 
  (h2 : θ ∈ Set.Ioo π (2*π)) : 
  Real.sin (θ - π/6) = -(5*Real.sqrt 3 + 12)/26 ∧ 
  Real.tan (θ + π/4) = 7/17 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_trigonometric_values_for_given_cosine_l813_81347


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cricket_bat_profit_percentage_l813_81335

/-- Calculate the profit percentage given the selling price and profit -/
noncomputable def profit_percentage (selling_price profit : ℝ) : ℝ :=
  let cost_price := selling_price - profit
  (profit / cost_price) * 100

/-- Theorem: The profit percentage for a cricket bat sold at $850 with a profit of $255 is approximately 42.86% -/
theorem cricket_bat_profit_percentage :
  abs (profit_percentage 850 255 - 42.86) < 0.01 := by
  -- Unfold the definition of profit_percentage
  unfold profit_percentage
  -- Simplify the expression
  simp
  -- Use numerical approximation
  norm_num
  -- Complete the proof
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_cricket_bat_profit_percentage_l813_81335


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_solve_trigonometric_equation_l813_81364

theorem solve_trigonometric_equation (a : ℝ) : 
  (∃ x : ℝ, x = -π/6 ∧ 3 * Real.tan (x + a) = Real.sqrt 3) → 
  a ∈ Set.Ioo (-π) 0 → 
  a = -2*π/3 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_solve_trigonometric_equation_l813_81364


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_similar_triangle_double_area_l813_81359

/-- Represents a point on the grid --/
structure GridPoint where
  x : ℤ
  y : ℤ

/-- Represents a triangle on the grid --/
structure GridTriangle where
  a : GridPoint
  b : GridPoint
  c : GridPoint

/-- The area of a triangle --/
noncomputable def triangleArea (t : GridTriangle) : ℝ := sorry

/-- The transformation function --/
def F (p : GridPoint) : GridPoint :=
  { x := p.x - p.y, y := p.x + p.y }

/-- Theorem: For any triangle on the grid, there exists a similar triangle with double the area --/
theorem similar_triangle_double_area 
  (X : GridTriangle) 
  (S : ℝ) 
  (h : triangleArea X = S) :
  ∃ (X' : GridTriangle), 
    (triangleArea X' = 2 * S) ∧ 
    (∀ (p : GridPoint), p ∈ ({X.a, X.b, X.c} : Set GridPoint) → 
      F p ∈ ({X'.a, X'.b, X'.c} : Set GridPoint)) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_similar_triangle_double_area_l813_81359


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_extrema_on_interval_l813_81385

open Set
open Function

theorem function_extrema_on_interval (f : ℝ → ℝ) (f' : ℝ → ℝ) (a b : ℝ) (h : a ≤ b) :
  (∀ x ∈ Icc a b, HasDerivAt f (f' x) x) →
  (∀ x ∈ Icc a b, f' x > 0) →
  (∀ x ∈ Icc a b, f a ≤ f x) ∧
  (∀ x ∈ Icc a b, f x ≤ f b) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_extrema_on_interval_l813_81385


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_between_specific_points_l813_81346

noncomputable def point := ℝ × ℝ

noncomputable def distance (p1 p2 : point) : ℝ :=
  Real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

theorem distance_between_specific_points :
  distance (2, 5) (5, -1) = 3 * Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_between_specific_points_l813_81346


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_inequality_l813_81313

/-- A strictly increasing function on real numbers -/
noncomputable def StrictlyIncreasing (f : ℝ → ℝ) : Prop :=
  ∀ x y, x < y → f x < f y

/-- The function g as defined in the problem -/
noncomputable def g (f : ℝ → ℝ) (x t : ℝ) : ℝ :=
  (f (x + t) - f x) / (f x - f (x - t))

/-- The theorem to be proved -/
theorem function_inequality (f : ℝ → ℝ) (h : StrictlyIncreasing f)
  (h1 : ∀ t > 0, (1 / 2 : ℝ) < g f 0 t ∧ g f 0 t < 2)
  (h2 : ∀ x ≠ 0, ∀ t > 0, t ≤ |x| → (1 / 2 : ℝ) < g f x t ∧ g f x t < 2) :
  ∀ x t, t > 0 → (1 / 14 : ℝ) < g f x t ∧ g f x t < 14 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_inequality_l813_81313


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_zero_in_interval_l813_81326

-- Define the function f(x) = ln x - 6 + 2x
noncomputable def f (x : ℝ) : ℝ := Real.log x - 6 + 2 * x

-- State the theorem
theorem zero_in_interval :
  ∃! x₀ : ℝ, x₀ ∈ Set.Ioo 2 3 ∧ f x₀ = 0 :=
by
  sorry

-- Note: Set.Ioo 2 3 represents the open interval (2, 3)

end NUMINAMATH_CALUDE_ERRORFEEDBACK_zero_in_interval_l813_81326


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_pyramid_volume_is_2_sqrt_29_l813_81380

/-- Represents a rectangular parallelepiped -/
structure RectangularParallelepiped where
  ab : ℝ
  bc : ℝ
  cg : ℝ

/-- Represents a point M on the edge FG of the parallelepiped -/
structure PointM where
  fm : ℝ
  mg : ℝ

/-- Calculates the volume of a rectangular pyramid -/
noncomputable def pyramidVolume (p : RectangularParallelepiped) (m : PointM) : ℝ :=
  (1/3) * p.bc * (Real.sqrt (p.ab^2 + p.bc^2)) * m.mg

theorem pyramid_volume_is_2_sqrt_29 (p : RectangularParallelepiped) (m : PointM) :
  p.ab = 5 → p.bc = 2 → p.cg = 4 → m.fm = 3 * m.mg →
  pyramidVolume p m = 2 * Real.sqrt 29 := by
  sorry

#check pyramid_volume_is_2_sqrt_29

end NUMINAMATH_CALUDE_ERRORFEEDBACK_pyramid_volume_is_2_sqrt_29_l813_81380


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_modified_octagon_perimeter_is_19_5_l813_81324

/-- The perimeter of a modified regular octagon -/
noncomputable def modified_octagon_perimeter (original_side_length : ℝ) : ℝ :=
  5 * original_side_length + 3 * (original_side_length / 2)

/-- Theorem: The perimeter of the modified octagon is 19.5 units -/
theorem modified_octagon_perimeter_is_19_5 :
  modified_octagon_perimeter 3 = 19.5 := by
  -- Unfold the definition of modified_octagon_perimeter
  unfold modified_octagon_perimeter
  -- Simplify the expression
  simp
  -- Perform the calculation
  norm_num


end NUMINAMATH_CALUDE_ERRORFEEDBACK_modified_octagon_perimeter_is_19_5_l813_81324


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_pi_4_implies_sin_cos_ratio_l813_81370

theorem tan_pi_4_implies_sin_cos_ratio (θ : Real) :
  Real.tan (θ + Real.pi / 4) = 2 →
  (Real.sin θ + Real.cos θ) / (Real.sin θ - Real.cos θ) = -2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_pi_4_implies_sin_cos_ratio_l813_81370


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_double_angle_special_case_l813_81307

theorem tan_double_angle_special_case (α : Real) 
  (h1 : α > π / 2) (h2 : α < π) (h3 : Real.sin (α - π / 2) = 3 / 5) : 
  Real.tan (2 * α) = 24 / 7 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_double_angle_special_case_l813_81307


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_x_prime_factorization_l813_81337

/-- Given positive integers x and y satisfying 3x^2 = 5y^4, 
    the smallest possible value of x is 3^1 * 5^4, 
    and its prime factorization a^b * c^d satisfies a + b + c + d = 13 -/
theorem smallest_x_prime_factorization (x y : ℕ+) (h : 3 * x^2 = 5 * y^4) :
  ∃ (a b c d : ℕ), 
    x = 3^1 * 5^4 ∧ 
    x = a^b * c^d ∧ 
    (∀ z : ℕ+, 3 * z^2 = 5 * y^4 → x ≤ z) ∧ 
    a + b + c + d = 13 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_x_prime_factorization_l813_81337


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_eccentricity_l813_81308

-- Define the hyperbola
noncomputable def hyperbola (a b : ℝ) (x y : ℝ) : Prop := x^2 / a^2 - y^2 / b^2 = 1

-- Define the eccentricity of a hyperbola
noncomputable def eccentricity (a b : ℝ) : ℝ := Real.sqrt (1 + b^2 / a^2)

-- Theorem statement
theorem hyperbola_eccentricity 
  (a b : ℝ) 
  (h1 : a > 0) 
  (h2 : b > 0) 
  (h3 : ∃ (k : ℝ), k * 3 = a ∧ k * (-4) = b) : 
  eccentricity a b = 5/3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_eccentricity_l813_81308


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_M_N_l813_81318

-- Define set M
def M : Set ℝ := {x : ℝ | -1 ≤ x ∧ x ≤ 2}

-- Define set N
def N : Set ℝ := {y : ℝ | ∃ x, y = 2^x}

-- Theorem statement
theorem intersection_M_N : M ∩ N = Set.Ioo 0 2 ∪ {2} := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_M_N_l813_81318


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_n_power_greater_than_square_l813_81345

theorem smallest_n_power_greater_than_square : ∃ N : ℕ,
  (N = 5) ∧
  (∀ n : ℕ, n ∈ Finset.range 5 → 2^(n + N) > (n + N)^2) ∧
  (∀ m : ℕ, m < N → ∃ k : ℕ, k ∈ Finset.range 5 ∧ 2^(k + m) ≤ (k + m)^2) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_n_power_greater_than_square_l813_81345


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_eccentricity_implies_a_l813_81394

/-- Represents an ellipse with semi-major axis a and semi-minor axis sqrt(5) -/
structure Ellipse where
  a : ℝ
  h_a_pos : a > 0

/-- The eccentricity of an ellipse -/
noncomputable def eccentricity (e : Ellipse) : ℝ := 
  Real.sqrt (1 - 5 / (e.a ^ 2))

/-- Theorem stating that for an ellipse with semi-major axis a and semi-minor axis sqrt(5),
    if the eccentricity is 2/3, then a = 3 -/
theorem ellipse_eccentricity_implies_a (e : Ellipse) 
    (h_ecc : eccentricity e = 2/3) : e.a = 3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_eccentricity_implies_a_l813_81394


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_johnny_hourly_wage_l813_81383

/-- Johnny's work scenario -/
structure WorkScenario where
  hours_worked : ℚ
  total_earnings : ℚ

/-- Calculate hourly wage given a work scenario -/
def hourly_wage (scenario : WorkScenario) : ℚ :=
  scenario.total_earnings / scenario.hours_worked

/-- Johnny's specific work scenario -/
def johnny_scenario : WorkScenario :=
  { hours_worked := 6
    total_earnings := 285/10 }

/-- Theorem: Johnny's hourly wage is $4.75 -/
theorem johnny_hourly_wage :
  hourly_wage johnny_scenario = 475/100 := by
  -- Unfold the definitions and perform the calculation
  unfold hourly_wage johnny_scenario
  -- The rest of the proof would go here
  sorry

#eval hourly_wage johnny_scenario

end NUMINAMATH_CALUDE_ERRORFEEDBACK_johnny_hourly_wage_l813_81383


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_undefined_values_count_l813_81312

theorem undefined_values_count : 
  ∃! (S : Finset ℝ), (∀ x ∈ S, (x^2 + x - 6) * (x - 4) = 0) ∧ Finset.card S = 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_undefined_values_count_l813_81312


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sine_cosine_subtraction_l813_81309

theorem sine_cosine_subtraction (x y : ℝ) : 
  Real.sin (x + y) * Real.cos y - Real.cos (x + y) * Real.sin y = Real.sin x := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sine_cosine_subtraction_l813_81309


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_max_value_tan_theta_value_l813_81302

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := 2 * Real.sin x * Real.cos x + Real.cos (2 * x)

-- Theorem for the maximum value of f
theorem f_max_value : 
  (∃ (k : ℤ), ∀ (x : ℝ), f x ≤ f (k * π + π / 8) ∧ 
  f (k * π + π / 8) = Real.sqrt 2) := by sorry

-- Theorem for the tangent value
theorem tan_theta_value (θ : ℝ) (h1 : 0 < θ) (h2 : θ < π / 2) 
  (h3 : f (θ + π / 8) = Real.sqrt 2 / 3) : 
  Real.tan θ = Real.sqrt 2 / 2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_max_value_tan_theta_value_l813_81302


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_interest_rate_calculation_l813_81336

/-- Calculates the simple interest rate given the principal, interest, and time. -/
noncomputable def simple_interest_rate (principal : ℝ) (interest : ℝ) (time : ℝ) : ℝ :=
  (interest / (principal * time)) * 100

/-- Theorem: Given the specified principal, interest, and time, the simple interest rate is 3.5% -/
theorem interest_rate_calculation :
  let principal : ℝ := 535.7142857142857
  let interest : ℝ := 75
  let time : ℝ := 4
  simple_interest_rate principal interest time = 3.5 := by
  unfold simple_interest_rate
  norm_num
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_interest_rate_calculation_l813_81336


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_larger_ball_radius_l813_81332

theorem larger_ball_radius (r : ℝ) :
  (4 / 3 * Real.pi * r^3 = 10 * (4 / 3 * Real.pi * 2^3)) → r = Real.rpow 240 (1/3) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_larger_ball_radius_l813_81332


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sugar_price_theorem_l813_81333

/-- Represents the original price of sugar per kg -/
noncomputable def original_price : ℝ := 8

/-- Represents the price reduction percentage -/
noncomputable def price_reduction : ℝ := 25 / 4

/-- Represents the amount of money spent -/
noncomputable def money_spent : ℝ := 120

/-- Theorem stating that given the conditions, the original price of sugar is 8 Rs per kg -/
theorem sugar_price_theorem : 
  let new_price := original_price * (1 - price_reduction / 100)
  let original_amount := money_spent / original_price
  let new_amount := money_spent / new_price
  new_amount - original_amount = 1 → original_price = 8 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_sugar_price_theorem_l813_81333


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_specific_isosceles_triangle_l813_81343

/-- An isosceles triangle with given side lengths -/
structure IsoscelesTriangle where
  /-- Length of equal sides -/
  side : ℝ
  /-- Length of base -/
  base : ℝ

/-- Calculate the area of an isosceles triangle -/
noncomputable def areaIsoscelesTriangle (t : IsoscelesTriangle) : ℝ :=
  let height := Real.sqrt (t.side ^ 2 - (t.base / 2) ^ 2)
  (t.base * height) / 2

/-- Theorem: Area of specific isosceles triangle is 60 -/
theorem area_specific_isosceles_triangle :
  let t : IsoscelesTriangle := { side := 13, base := 10 }
  areaIsoscelesTriangle t = 60 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_specific_isosceles_triangle_l813_81343


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_point_coordinates_l813_81366

/-- Parabola type representing y^2 = 4x -/
structure Parabola where
  equation : ℝ → ℝ → Prop
  focus : ℝ × ℝ

/-- Point on a parabola -/
structure ParabolaPoint (p : Parabola) where
  x : ℝ
  y : ℝ
  on_parabola : p.equation x y

/-- Distance between two points in 2D space -/
noncomputable def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

theorem parabola_point_coordinates 
  (p : Parabola) 
  (h_eq : p.equation = fun x y => y^2 = 4*x) 
  (h_focus : p.focus = (1, 0)) 
  (A : ParabolaPoint p) 
  (h_dist : distance (A.x, A.y) p.focus = 3) : 
  A.x = 2 ∧ (A.y = 2 * Real.sqrt 2 ∨ A.y = -2 * Real.sqrt 2) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_point_coordinates_l813_81366


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distributive_analogous_to_division_l813_81331

-- Define the distributive property
def distributive (a b c : ℝ) : Prop := (a + b) * c = a * c + b * c

-- Define the division property
def division_preserves_equality (a b c : ℝ) : Prop :=
  c ≠ 0 → (a / c = b / c → a = b)

-- State the theorem about the analogy
theorem distributive_analogous_to_division :
  ∀ (a b c : ℝ), (distributive a b c ↔ division_preserves_equality a b c) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_distributive_analogous_to_division_l813_81331


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_first_sales_amount_l813_81396

/-- The amount of the first sales in millions of dollars -/
def S : ℝ := sorry

/-- The royalties received on the first sales in millions of dollars -/
def first_royalties : ℝ := 6

/-- The royalties received on the next $108 million in sales in millions of dollars -/
def second_royalties : ℝ := 9

/-- The amount of the second sales in millions of dollars -/
def second_sales : ℝ := 108

/-- The decrease in the ratio of royalties to sales as a decimal -/
def ratio_decrease : ℝ := 0.7222222222222221

theorem first_sales_amount : 
  (first_royalties / S) - (second_royalties / second_sales) = 
  (first_royalties / S) * ratio_decrease → S = 20 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_first_sales_amount_l813_81396


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_water_transfer_height_l813_81381

/-- Represents the dimensions of a cuboid -/
structure CuboidDimensions where
  length : ℝ
  width : ℝ
  height : ℝ

/-- Represents the dimensions of a square pyramid -/
structure PyramidDimensions where
  baseSide : ℝ
  height : ℝ

/-- Calculates the volume of a square pyramid -/
noncomputable def pyramidVolume (d : PyramidDimensions) : ℝ :=
  (1 / 3) * d.baseSide^2 * d.height

/-- Calculates the height of water in a cuboid given its base dimensions and volume -/
noncomputable def waterHeight (base : CuboidDimensions) (volume : ℝ) : ℝ :=
  volume / (base.length * base.width)

theorem water_transfer_height : 
  let pyramid := PyramidDimensions.mk 10 15
  let cuboid := CuboidDimensions.mk 15 20 0
  let pyramidWaterVolume := pyramidVolume pyramid
  waterHeight cuboid pyramidWaterVolume = 5/3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_water_transfer_height_l813_81381


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_sine_identity_l813_81376

theorem triangle_sine_identity (A B C : ℝ) (h : A + B + C = Real.pi) :
  Real.sin (4 * A) + Real.sin (4 * B) + Real.sin (4 * C) = 
  -4 * Real.sin (2 * A) * Real.sin (2 * B) * Real.sin (2 * C) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_sine_identity_l813_81376


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rounding_division_result_l813_81387

noncomputable def roundDown (x : ℝ) : ℝ :=
  ⌊x * 10000⌋ / 10000

theorem rounding_division_result (x : ℝ) (hx : x > 0) :
  ∃ (y : ℝ), y ∈ Set.Icc (0 : ℝ) 1 ∧
             ∃ (k : ℕ), y = k / 10000 ∧
             y = roundDown (roundDown x / x) := by
  sorry

#check rounding_division_result

end NUMINAMATH_CALUDE_ERRORFEEDBACK_rounding_division_result_l813_81387


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_not_in_third_quadrant_l813_81319

/-- Given a line ax + by + c = 0 where ac < 0 and bc < 0, 
    the line does not pass through the third quadrant. -/
theorem line_not_in_third_quadrant 
  (a b c : ℝ) 
  (h1 : a * c < 0) 
  (h2 : b * c < 0) : 
  ∃ (x y : ℝ), a * x + b * y + c = 0 ∧ ¬(x < 0 ∧ y < 0) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_not_in_third_quadrant_l813_81319


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_C_value_triangle_area_l813_81352

-- Define the triangle ABC
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

-- Define the conditions of the problem
def triangle_conditions (t : Triangle) : Prop :=
  t.A = Real.pi/3 ∧ t.c = 3/7 * t.a

-- Part I: Prove that sin C = 3√3/14
theorem sin_C_value (t : Triangle) (h : triangle_conditions t) : 
  Real.sin t.C = 3 * Real.sqrt 3 / 14 := by
  sorry

-- Part II: Prove that if a = 7, the area of the triangle is 6√3
theorem triangle_area (t : Triangle) (h : triangle_conditions t) (h_a : t.a = 7) :
  (1/2) * t.a * t.c * Real.sin t.B = 6 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_C_value_triangle_area_l813_81352


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_right_triangles_and_angle_l813_81375

/-- Triangle with side lengths a, b, and c --/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Check if a triangle is a right triangle using the Pythagorean theorem --/
def isRightTriangle (t : Triangle) : Prop :=
  t.a^2 + t.b^2 = t.c^2

/-- Calculate the angle between sides a and b using the Law of Cosines --/
noncomputable def angleBetweenSides (t : Triangle) : ℝ :=
  Real.arccos ((t.a^2 + t.b^2 - t.c^2) / (2 * t.a * t.b))

theorem right_triangles_and_angle :
  let t1 : Triangle := ⟨15, 36, 39⟩
  let t2 : Triangle := ⟨40, 42, 58⟩
  (isRightTriangle t1 ∧ isRightTriangle t2) ∧
  (angleBetweenSides t1 = Real.pi / 2) := by
  sorry

#check right_triangles_and_angle

end NUMINAMATH_CALUDE_ERRORFEEDBACK_right_triangles_and_angle_l813_81375


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_set_of_inequality_l813_81357

-- Define the function f
noncomputable def f (x : ℝ) : ℝ :=
  if x ≥ 0 then x^3 + 2*x else ((-x)^3 + 2*(-x))

-- State the theorem
theorem solution_set_of_inequality (f : ℝ → ℝ) :
  (∀ x, f x = f (-x)) →  -- f is even
  (∀ x ≥ 0, f x = x^3 + 2*x) →  -- definition of f for x ≥ 0
  {x | f (x - 2) < 3} = Set.Ioo 1 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_set_of_inequality_l813_81357


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_fescue_percentage_in_Y_l813_81327

/-- Represents a seed mixture -/
structure SeedMixture where
  ryegrass : ℚ
  bluegrass : ℚ
  fescue : ℚ

/-- The composition of seed mixture X -/
def X : SeedMixture :=
  { ryegrass := 2/5
    bluegrass := 3/5
    fescue := 0 }

/-- The composition of seed mixture Y -/
def Y : SeedMixture :=
  { ryegrass := 1/4
    bluegrass := 0
    fescue := 3/4 }  -- This is what we want to prove

/-- The proportion of X in the final mixture -/
def x_proportion : ℚ := 1 / 3

/-- The proportion of Y in the final mixture -/
def y_proportion : ℚ := 2 / 3

/-- The percentage of ryegrass in the final mixture -/
def final_ryegrass_percentage : ℚ := 3 / 10

theorem fescue_percentage_in_Y :
  Y.fescue = 3/4 :=
by
  sorry

#check fescue_percentage_in_Y

end NUMINAMATH_CALUDE_ERRORFEEDBACK_fescue_percentage_in_Y_l813_81327


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_uniform_transformation_l813_81350

-- Define the uniform distribution on [0, 1]
def uniform_01 : Set ℝ := { x | 0 ≤ x ∧ x ≤ 1 }

-- Define the uniform distribution on [-3, 3]
def uniform_neg3_3 : Set ℝ := { x | -3 ≤ x ∧ x ≤ 3 }

-- Define the transformation function
def transform (x : ℝ) : ℝ := 6 * x - 3

-- Theorem statement
theorem uniform_transformation :
  ∀ (x : ℝ), x ∈ uniform_01 → transform x ∈ uniform_neg3_3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_uniform_transformation_l813_81350


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_dot_product_AB_AC_l813_81388

noncomputable section

open Real

def f (x : ℝ) : ℝ := 
  (3 : ℝ).sqrt * sin (π - x) * cos (-x) + sin (π + x) * cos (π / 2 - x)

def A : ℝ × ℝ := (2 * π / 3, -3 / 2)
def B : ℝ × ℝ := (π / 6, 1 / 2)
def C : ℝ × ℝ := (7 * π / 6, 1 / 2)

theorem dot_product_AB_AC : 
  let AB : ℝ × ℝ := (B.1 - A.1, B.2 - A.2)
  let AC : ℝ × ℝ := (C.1 - A.1, C.2 - A.2)
  AB.1 * AC.1 + AB.2 * AC.2 = 4 - π^2 / 4 := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_dot_product_AB_AC_l813_81388


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_odd_and_consistent_l813_81348

noncomputable def f (x : ℝ) : ℝ := 
  if x ≤ 0 then Real.log ((-x + 1) : ℝ) / Real.log (1/2 : ℝ)
  else -Real.log ((x + 1) : ℝ) / Real.log (1/2 : ℝ)

theorem f_odd_and_consistent : 
  (∀ x, f (-x) = -f x) ∧ 
  (∀ x ≤ 0, f x = Real.log ((-x + 1) : ℝ) / Real.log (1/2 : ℝ)) ∧
  (∀ x > 0, f x = -Real.log ((x + 1) : ℝ) / Real.log (1/2 : ℝ)) := by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_odd_and_consistent_l813_81348


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_b_2016_equals_negative_four_l813_81391

def b : ℕ → ℤ
  | 0 => 1  -- Added case for 0
  | 1 => 1
  | 2 => 5
  | (n + 3) => b (n + 2) - b (n + 1)

theorem b_2016_equals_negative_four : b 2016 = -4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_b_2016_equals_negative_four_l813_81391


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_to_line_l813_81372

/-- The minimum distance from the origin to a point on the line 3x + 4y - 20 = 0 is 4. -/
theorem min_distance_to_line : 
  ∃ (d : ℝ), d = 4 ∧ 
  ∀ (a b : ℝ), (3 * a + 4 * b - 20 = 0) → Real.sqrt (a^2 + b^2) ≥ d :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_to_line_l813_81372


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_batsman_sixes_l813_81356

theorem batsman_sixes (total_runs : ℕ) (boundaries : ℕ) (boundary_value : ℕ) (running_percentage : ℚ) : 
  total_runs = 120 →
  boundaries = 3 →
  boundary_value = 4 →
  running_percentage = 1/2 →
  ∃ (sixes : ℕ), 
    total_runs = boundaries * boundary_value + (running_percentage * ↑total_runs).floor + sixes * 6 ∧
    sixes = 8 :=
by
  intro h_total h_boundaries h_boundary_value h_running_percentage
  use 8
  apply And.intro
  · rw [h_total, h_boundaries, h_boundary_value, h_running_percentage]
    simp
    norm_num
    rfl
  · rfl


end NUMINAMATH_CALUDE_ERRORFEEDBACK_batsman_sixes_l813_81356


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_pyramid_distances_l813_81322

/-- Sequence of maximum distances in nested pyramids --/
noncomputable def D : ℕ → ℝ
  | 0 => 2019
  | n + 1 => 2019 * Real.sqrt 3 / (2 ^ (n + 1))

/-- The theorem stating the formula for maximum distances --/
theorem pyramid_distances : 
  ∀ n : ℕ, D n = if n = 0 then 2019 else 2019 * Real.sqrt 3 / (2 ^ n) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_pyramid_distances_l813_81322


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hexagonal_pyramid_surface_area_l813_81341

/-- The total surface area of a regular hexagonal pyramid -/
noncomputable def totalSurfaceArea (h : ℝ) : ℝ := (3 * h^2 * Real.sqrt 3) / 2

/-- Theorem: The total surface area of a regular hexagonal pyramid with apothem h
    and dihedral angle at the base 60° is (3h² √3) / 2 -/
theorem hexagonal_pyramid_surface_area (h : ℝ) (h_pos : h > 0) :
  let apothem := h
  let dihedral_angle := π / 3  -- 60° in radians
  totalSurfaceArea apothem = (3 * h^2 * Real.sqrt 3) / 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_hexagonal_pyramid_surface_area_l813_81341


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_arrival_time_difference_l813_81382

/-- The distance to the park in miles -/
noncomputable def distance_to_park : ℝ := 2

/-- Jill's speed in miles per hour -/
noncomputable def jill_speed : ℝ := 10

/-- Jack's speed in miles per hour -/
noncomputable def jack_speed : ℝ := 5

/-- Convert hours to minutes -/
noncomputable def hours_to_minutes (hours : ℝ) : ℝ := hours * 60

/-- Calculate travel time in hours given distance and speed -/
noncomputable def travel_time (distance : ℝ) (speed : ℝ) : ℝ := distance / speed

theorem arrival_time_difference : 
  hours_to_minutes (travel_time distance_to_park jack_speed - travel_time distance_to_park jill_speed) = 12 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_arrival_time_difference_l813_81382


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_lateral_area_inscribed_cylinder_l813_81342

-- Define the hemisphere
def hemisphere_radius : ℝ := 2

-- Define the inscribed cylinder
def inscribed_cylinder (r h : ℝ) : Prop :=
  r ≥ 0 ∧ h ≥ 0 ∧ r^2 + h^2 = hemisphere_radius^2

-- Define the lateral area of a cylinder
noncomputable def lateral_area (r h : ℝ) : ℝ := 2 * Real.pi * r * h

-- Theorem statement
theorem max_lateral_area_inscribed_cylinder :
  ∃ (max_area : ℝ), max_area = 4 * Real.pi ∧
  ∀ (r h : ℝ), inscribed_cylinder r h →
  lateral_area r h ≤ max_area := by
  sorry

#check max_lateral_area_inscribed_cylinder

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_lateral_area_inscribed_cylinder_l813_81342

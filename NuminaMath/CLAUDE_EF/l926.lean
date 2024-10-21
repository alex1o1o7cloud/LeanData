import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_wedge_volume_approximation_l926_92631

-- Define the cylinder dimensions
noncomputable def cylinder_radius : ℝ := 4
noncomputable def cylinder_height : ℝ := 10

-- Define the wedge angle in degrees
noncomputable def wedge_angle : ℝ := 60

-- Define pi as a constant (since we're using an approximation in the final result)
noncomputable def π : ℝ := Real.pi

-- Define the volume of the wedge
noncomputable def wedge_volume : ℝ :=
  (wedge_angle / 360) * π * cylinder_radius^2 * cylinder_height

-- Theorem statement
theorem wedge_volume_approximation :
  ∃ ε > 0, |wedge_volume - 83.77| < ε := by
  -- The proof goes here
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_wedge_volume_approximation_l926_92631


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_equations_l926_92640

-- Define the circle
def circle_equation (x y : ℝ) : Prop := (x - 3)^2 + (y - 2)^2 = 1

-- Define the point through which the tangent line passes
def tangent_point : ℝ × ℝ := (4, 0)

-- Define the possible tangent lines
def tangent_line_1 (x y : ℝ) : Prop := 3*x + 4*y - 12 = 0
def tangent_line_2 (x : ℝ) : Prop := x = 4

theorem tangent_line_equations :
  ∀ (k m : ℝ), (∀ (x y : ℝ), (y = k*x + m) → 
    ((x = tangent_point.1 ∧ y = tangent_point.2) ∨ 
     (∃ (t : ℝ), circle_equation x y ∧ 
       (y - tangent_point.2 = k * (x - tangent_point.1)) ∧ 
       ((x - 3) * (x - tangent_point.1) + (y - 2) * (y - tangent_point.2) = 0)))) 
  → ((k = -3/4 ∧ m = 3) ∨ (k = 0 ∧ m = 4)) := by sorry

#check tangent_line_equations

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_equations_l926_92640


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_lakers_win_in_seven_l926_92635

def nba_finals_probability (lakers_win_prob : ℚ) : ℚ :=
  let celtics_win_prob := 1 - lakers_win_prob
  let ways_to_reach_game7 := (Nat.choose 6 3 : ℚ)
  let prob_3_3_tie := ways_to_reach_game7 * lakers_win_prob^3 * celtics_win_prob^3
  prob_3_3_tie * lakers_win_prob

theorem lakers_win_in_seven :
  nba_finals_probability (1/4) = 540/16384 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_lakers_win_in_seven_l926_92635


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_FYH_in_specific_trapezoid_l926_92612

/-- Represents a trapezoid with given properties -/
structure Trapezoid where
  ef : ℝ
  gh : ℝ
  area : ℝ

/-- Calculates the area of triangle FYH in a trapezoid with given properties -/
noncomputable def areaFYH (t : Trapezoid) : ℝ :=
  (t.ef * (t.area / (t.ef + t.gh))) / 5

/-- Theorem stating the area of triangle FYH in the specific trapezoid -/
theorem area_FYH_in_specific_trapezoid :
  let t : Trapezoid := { ef := 24, gh := 36, area := 360 }
  areaFYH t = 57.6 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_FYH_in_specific_trapezoid_l926_92612


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_set_equivalence_l926_92668

-- Define the function f(x) = 0.5^x
noncomputable def f (x : ℝ) : ℝ := (1/2) ^ x

-- State that f is decreasing on ℝ
axiom f_decreasing : ∀ x y : ℝ, x < y → f y < f x

-- Theorem statement
theorem solution_set_equivalence :
  ∀ x : ℝ, f (2 * x) > f (x⁻¹) ↔ x < -1 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_set_equivalence_l926_92668


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_x_lt_2y_is_two_fifths_l926_92685

/-- A rectangle in the 2D plane -/
structure Rectangle where
  x_min : ℝ
  x_max : ℝ
  y_min : ℝ
  y_max : ℝ
  h_x : x_min < x_max
  h_y : y_min < y_max

/-- The probability that a randomly selected point (x,y) from a rectangle satisfies x < 2y -/
noncomputable def probability_x_lt_2y (r : Rectangle) : ℝ :=
  let triangle_area := (r.y_max - r.y_min) * (2 * r.y_max - r.x_min) / 2
  let rectangle_area := (r.x_max - r.x_min) * (r.y_max - r.y_min)
  triangle_area / rectangle_area

/-- The specific rectangle in the problem -/
def problem_rectangle : Rectangle where
  x_min := 0
  x_max := 5
  y_min := 0
  y_max := 2
  h_x := by norm_num
  h_y := by norm_num

theorem probability_x_lt_2y_is_two_fifths :
  probability_x_lt_2y problem_rectangle = 2 / 5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_x_lt_2y_is_two_fifths_l926_92685


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_inequality_and_monotonicity_l926_92670

-- Define the function f(x)
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := Real.sqrt (x^2 + 1) - a * x

-- State the theorem
theorem f_inequality_and_monotonicity (a : ℝ) (h : a > 0) :
  (∀ x : ℝ, f a x ≤ 1 ↔ 
    ((0 < a ∧ a < 1 ∧ 0 ≤ x ∧ x ≤ 2*a/(1-a^2)) ∨
     (a ≥ 1 ∧ x ≥ 0))) ∧
  (∀ x y : ℝ, 0 ≤ x → 0 ≤ y → x < y → f a x > f a y ↔ a ≥ 1) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_inequality_and_monotonicity_l926_92670


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_max_sum_l926_92673

/-- Triangle XYZ in the Cartesian plane -/
structure Triangle where
  X : ℝ × ℝ
  Y : ℝ × ℝ
  Z : ℝ × ℝ

/-- Area of a triangle given its vertices -/
noncomputable def triangleArea (t : Triangle) : ℝ :=
  (1/2) * abs ((t.X.1 * (t.Y.2 - t.Z.2) + t.Y.1 * (t.Z.2 - t.X.2) + t.Z.1 * (t.X.2 - t.Y.2)))

/-- Slope of a line given two points -/
noncomputable def slopeOfLine (p1 p2 : ℝ × ℝ) : ℝ :=
  (p2.2 - p1.2) / (p2.1 - p1.1)

/-- Midpoint of a line segment -/
noncomputable def midpointOfLine (p1 p2 : ℝ × ℝ) : ℝ × ℝ :=
  ((p1.1 + p2.1) / 2, (p1.2 + p2.2) / 2)

/-- The main theorem -/
theorem triangle_max_sum (t : Triangle) (h1 : t.X = (10, 15)) (h2 : t.Y = (20, 17))
    (h3 : triangleArea t = 50) (h4 : slopeOfLine (midpointOfLine t.X t.Y) t.Z = -3) :
    t.Z.1 + t.Z.2 ≤ 1460/92 + 13/92 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_max_sum_l926_92673


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inradius_circumradius_product_l926_92658

/-- Given a triangle ABC with side lengths a, b, c that are roots of x^3 - 27x^2 + 222x - 540,
    the product of its inradius and circumradius is 10. -/
theorem inradius_circumradius_product (a b c : ℝ) : 
  a^3 - 27*a^2 + 222*a - 540 = 0 →
  b^3 - 27*b^2 + 222*b - 540 = 0 →
  c^3 - 27*c^2 + 222*c - 540 = 0 →
  a > 0 → b > 0 → c > 0 →
  a + b > c → b + c > a → c + a > b →
  let s := (a + b + c) / 2
  let area := Real.sqrt (s * (s - a) * (s - b) * (s - c))
  let inradius := area / s
  let circumradius := (a * b * c) / (4 * area)
  inradius * circumradius = 10 := by
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_inradius_circumradius_product_l926_92658


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_a_plus_b_l926_92614

-- Define points A, B, C
def A : ℝ × ℝ := (1, -1)
def B : ℝ × ℝ := (4, 0)
def C : ℝ × ℝ := (2, 2)

-- Define vectors AB and AC
def AB : ℝ × ℝ := (B.1 - A.1, B.2 - A.2)
def AC : ℝ × ℝ := (C.1 - A.1, C.2 - A.2)

-- Define the region D
def D (a b : ℝ) : Set (ℝ × ℝ) :=
  {P | ∃ (l m : ℝ), 1 < l ∧ l ≤ a ∧ 1 < m ∧ m ≤ b ∧
    (P.1 - A.1, P.2 - A.2) = (l * AB.1 + m * AC.1, l * AB.2 + m * AC.2)}

-- Define the area of region D
def area_D (a b : ℝ) : ℝ := (a - 1) * (b - 1) * 8

-- Theorem statement
theorem min_a_plus_b :
  ∀ a b : ℝ, a > 1 ∧ b > 1 ∧ area_D a b = 8 → a + b ≥ 4 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_a_plus_b_l926_92614


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_k_for_two_zeros_l926_92678

/-- The function f(x) defined in the problem -/
noncomputable def f (k : ℝ) (x : ℝ) : ℝ := x^2 - x * Real.log x - k * (x + 2) + 2

/-- Theorem stating the range of k for which f(x) has two zeros in (1/2, +∞) -/
theorem range_of_k_for_two_zeros (k : ℝ) :
  (∃ x₁ x₂, 1/2 < x₁ ∧ x₁ < x₂ ∧ f k x₁ = 0 ∧ f k x₂ = 0) →
  1 < k ∧ k ≤ (9 + 2 * Real.log 2) / 10 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_k_for_two_zeros_l926_92678


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_T_max_value_l926_92690

noncomputable def T (n : ℕ) : ℝ := n * (2021 / 2022) ^ (n - 1)

theorem T_max_value : 
  ∀ k : ℕ, k ≠ 2021 ∧ k ≠ 2022 → T k ≤ max (T 2021) (T 2022) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_T_max_value_l926_92690


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_exactly_one_true_l926_92664

-- Define proposition p
def p : Prop := ∀ (A B C : Real) (AB BC : Real),
  AB < BC → Real.sin C < Real.sin A

-- Define proposition q
def q : Prop := ∀ a : Real,
  (a > 1 → (1 / a < 1)) ∧ ¬(∀ a : Real, (1 / a < 1) → a > 1)

theorem exactly_one_true : 
  (p ∨ q) ∧
  ¬(p ∧ q) ∧
  ¬(¬p ∨ q) ∧
  ¬(¬p ∧ q) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_exactly_one_true_l926_92664


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_sum_l926_92681

def b : ℕ → ℚ
  | 0 => 2
  | 1 => 3
  | (n + 2) => b (n + 1) + b n

theorem sequence_sum : ∑' n, b n / 9^(n + 1) = 1 / 29 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_sum_l926_92681


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_line_intersection_circle_constant_value_l926_92637

/-- Circle with equation x^2 + y^2 - 2ax - 2y + 2 = 0 -/
def Circle (a : ℝ) : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.1^2 + p.2^2 - 2*a*p.1 - 2*p.2 + 2 = 0}

/-- Line with equation y = x -/
def Line : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.2 = p.1}

/-- The center of the circle -/
def CircleCenter (a : ℝ) : ℝ × ℝ := (a, 1)

/-- The angle ACB, where A and B are intersection points of the circle and line -/
noncomputable def AngleACB (a : ℝ) : ℝ := Real.pi / 3

theorem circle_line_intersection (a : ℝ) :
  ∃ A B : ℝ × ℝ, A ∈ Circle a ∧ A ∈ Line ∧ B ∈ Circle a ∧ B ∈ Line ∧ A ≠ B :=
sorry

theorem circle_constant_value :
  ∃ a : ℝ, (∀ A B : ℝ × ℝ, A ∈ Circle a ∧ A ∈ Line ∧ B ∈ Circle a ∧ B ∈ Line ∧ A ≠ B →
    AngleACB a = Real.pi / 3) → a = -5 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_line_intersection_circle_constant_value_l926_92637


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_no_zeros_implies_a_geq_neg_one_l926_92609

/-- The function f(x) with parameter a -/
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := x^2 - (1/2) * Real.log x + a*x

/-- Theorem stating that if f(x) has no zeros in (1, +∞), then a ≥ -1 -/
theorem f_no_zeros_implies_a_geq_neg_one (a : ℝ) :
  (∀ x > 1, f a x ≠ 0) → a ≥ -1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_no_zeros_implies_a_geq_neg_one_l926_92609


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_same_color_right_triangle_l926_92662

/-- A color type with two possibilities: red or blue -/
inductive Color
  | red
  | blue

/-- A point on the side of a triangle -/
structure Point where
  x : ℝ
  y : ℝ

/-- An equilateral triangle -/
structure EquilateralTriangle where
  A : Point
  B : Point
  C : Point
  is_equilateral : sorry  -- Condition for equilateral triangle

/-- A coloring function that assigns a color to each point on the sides of a triangle -/
def coloring (t : EquilateralTriangle) : Point → Color := sorry

/-- A right triangle formed by three points -/
structure RightTriangle where
  P : Point
  Q : Point
  R : Point
  is_right : sorry  -- Condition for right triangle

/-- Check if a point is on the side of an equilateral triangle -/
def on_side (t : EquilateralTriangle) (p : Point) : Prop := sorry

/-- Main theorem: There exists a right triangle with vertices of the same color -/
theorem exists_same_color_right_triangle (t : EquilateralTriangle) :
  ∃ (rt : RightTriangle), 
    (on_side t rt.P ∧ on_side t rt.Q ∧ on_side t rt.R) ∧ 
    (coloring t rt.P = coloring t rt.Q ∧ coloring t rt.Q = coloring t rt.R) :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_same_color_right_triangle_l926_92662


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_centroid_distance_approx_l926_92608

/-- Triangle XYZ with given side lengths --/
structure Triangle (X Y Z : ℝ × ℝ) where
  xy_length : dist X Y = 30
  xz_length : dist X Z = 29
  yz_length : dist Y Z = 31

/-- Centroid of a triangle --/
noncomputable def centroid (X Y Z : ℝ × ℝ) : ℝ × ℝ :=
  ((X.1 + Y.1 + Z.1) / 3, (X.2 + Y.2 + Z.2) / 3)

/-- Theorem statement --/
theorem centroid_distance_approx (X Y Z : ℝ × ℝ) (t : Triangle X Y Z) :
  let G := centroid X Y Z
  abs (dist X G - 16.7) < 0.1 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_centroid_distance_approx_l926_92608


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_range_l926_92624

noncomputable def linear_function (x : ℝ) : ℝ := -2 * x + 3

noncomputable def inverse_proportion (k : ℝ) (x : ℝ) : ℝ := k / x

def intersect (k : ℝ) : Prop :=
  ∃ x : ℝ, x ≠ 0 ∧ linear_function x = inverse_proportion k x

theorem intersection_range :
  ∀ k : ℝ, k > 0 → (intersect k ↔ k ≤ 9/8) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_range_l926_92624


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hillside_apartment_size_l926_92600

/-- Given a rental rate and a monthly budget, calculates the largest affordable apartment size. -/
noncomputable def largest_affordable_size (rate : ℚ) (budget : ℚ) : ℚ :=
  budget / rate

/-- Theorem stating that given the specific rental rate and budget, the largest affordable size is 600 sq ft. -/
theorem hillside_apartment_size :
  let rate : ℚ := 120 / 100
  let budget : ℚ := 720
  largest_affordable_size rate budget = 600 := by
  -- Unfold the definitions
  unfold largest_affordable_size
  -- Simplify the expression
  simp
  -- The proof is complete
  rfl


end NUMINAMATH_CALUDE_ERRORFEEDBACK_hillside_apartment_size_l926_92600


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_dog_hare_speed_ratio_l926_92663

/-- Represents the distance covered in one leap -/
structure LeapDistance where
  distance : ℝ
  positive : distance > 0

/-- Represents an animal with a leap distance -/
structure Animal where
  leap : LeapDistance

/-- Defines the relationship between dog and hare leaps -/
def dog_hare_leap_relation (dog : Animal) (hare : Animal) : Prop :=
  dog.leap.distance = 2 * hare.leap.distance

/-- Defines the leap count relationship between dog and hare -/
def dog_hare_leap_count_relation (dog_leaps : ℕ) (hare_leaps : ℕ) : Prop :=
  dog_leaps = 10 ∧ hare_leaps = 2

/-- Calculates the speed ratio between two animals -/
noncomputable def speed_ratio (a b : Animal) (a_leaps b_leaps : ℕ) : ℝ :=
  (a_leaps : ℝ) * a.leap.distance / ((b_leaps : ℝ) * b.leap.distance)

/-- Theorem stating the speed ratio between dog and hare -/
theorem dog_hare_speed_ratio (dog hare : Animal) (dog_leaps hare_leaps : ℕ) :
  dog_hare_leap_relation dog hare →
  dog_hare_leap_count_relation dog_leaps hare_leaps →
  speed_ratio dog hare dog_leaps hare_leaps = 10 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_dog_hare_speed_ratio_l926_92663


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_apple_trees_l926_92675

/-- Represents a grid for planting apple trees -/
structure AppleGrid :=
  (total_cells : Nat)
  (border_trees : Nat)
  (max_illumination : Nat)

/-- Predicate to check if it's possible to plant n trees in the grid -/
def can_plant_trees (grid : AppleGrid) (n : Nat) : Prop :=
  ∃ (light_windows : Nat),
    n ≤ grid.border_trees + (grid.total_cells - grid.border_trees - light_windows) ∧
    grid.total_cells - grid.border_trees - light_windows ≤ light_windows * grid.max_illumination

/-- Theorem: Maximum number of apple trees in the grid -/
theorem max_apple_trees (grid : AppleGrid) 
  (h1 : grid.total_cells = 69)
  (h2 : grid.border_trees = 28)
  (h3 : grid.max_illumination = 4) : 
  ∃ (max_trees : Nat), max_trees = 60 ∧ 
  ∀ (n : Nat), n > max_trees → ¬(can_plant_trees grid n) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_apple_trees_l926_92675


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_floor_sin_equal_in_interval_l926_92674

theorem floor_sin_equal_in_interval (x y : ℝ) (hx : 0 < x ∧ x < Real.pi/2) (hy : 0 < y ∧ y < Real.pi/2) :
  ⌊Real.sin x⌋ = ⌊Real.sin y⌋ ∧ ⌊Real.sin x⌋ = 0 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_floor_sin_equal_in_interval_l926_92674


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_is_smallest_for_three_pairwise_coprime_l926_92682

/-- The function f(n) that gives the smallest number of elements needed to guarantee
    at least 3 pairwise coprime elements in any subset of {m, m+1, ..., m+n-1} -/
def f (n : ℕ) : ℕ :=
  (n + 1) / 2 + (n + 1) / 3 - (n + 1) / 6 + 1

/-- The property that any f(n)-element subset of {m, m+1, ..., m+n-1}
    contains at least 3 pairwise coprime elements -/
def has_three_pairwise_coprime (n m k : ℕ) : Prop :=
  ∀ (S : Finset ℕ), S ⊆ Finset.range n ∧ S.card = k →
    ∃ (a b c : ℕ), a ∈ S ∧ b ∈ S ∧ c ∈ S ∧
      Nat.Coprime (m + a) (m + b) ∧
      Nat.Coprime (m + b) (m + c) ∧
      Nat.Coprime (m + c) (m + a)

theorem f_is_smallest_for_three_pairwise_coprime (n : ℕ) (h : n ≥ 4) :
  (∀ m : ℕ, has_three_pairwise_coprime n m (f n)) ∧
  (∀ k : ℕ, k < f n → ∃ m : ℕ, ¬has_three_pairwise_coprime n m k) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_is_smallest_for_three_pairwise_coprime_l926_92682


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_is_smallest_f_is_minimal_l926_92647

/-- Given a positive integer n ≥ 4, f(n) is the smallest integer such that
    for any positive integer m, any subset of f(n) elements from the set
    {m, m+1, ..., m+n-1} contains at least 3 pairwise coprime elements. -/
def f (n : ℕ) : ℕ :=
  (n + 1) / 2 + (n + 1) / 3 - (n + 1) / 6 + 1

/-- Theorem stating that f(n) is the smallest integer satisfying the given property
    for any integer n ≥ 4. -/
theorem f_is_smallest (n : ℕ) (hn : n ≥ 4) :
  ∀ (m : ℕ), m > 0 →
    ∀ (S : Finset ℕ),
      S ⊆ Finset.range n →
      S.card = f n →
      ∃ (a b c : ℕ), a ∈ S ∧ b ∈ S ∧ c ∈ S ∧
        a ≠ b ∧ b ≠ c ∧ a ≠ c ∧
        Nat.Coprime a b ∧ Nat.Coprime b c ∧ Nat.Coprime a c :=
by sorry

/-- Theorem stating that no smaller integer than f(n) satisfies the property
    for any integer n ≥ 4. -/
theorem f_is_minimal (n : ℕ) (hn : n ≥ 4) :
  ∀ (k : ℕ), k < f n →
    ∃ (m : ℕ), m > 0 ∧
      ∃ (S : Finset ℕ),
        S ⊆ Finset.range n ∧
        S.card = k ∧
        ∀ (a b c : ℕ), a ∈ S → b ∈ S → c ∈ S →
          a ≠ b → b ≠ c → a ≠ c →
          ¬(Nat.Coprime a b ∧ Nat.Coprime b c ∧ Nat.Coprime a c) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_is_smallest_f_is_minimal_l926_92647


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_cos_equation_solution_l926_92683

theorem sin_cos_equation_solution :
  ∃ x : ℝ, x = π / 2 ∧ Real.sin (4 * x) * Real.sin (5 * x) = Real.cos (4 * x) * Real.cos (5 * x) :=
by
  use π / 2
  constructor
  · rfl
  · sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_cos_equation_solution_l926_92683


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_sphere_radius_for_specific_pyramid_l926_92645

/-- The smallest radius of a sphere that can contain a regular quadrilateral pyramid with given dimensions -/
noncomputable def smallest_sphere_radius (base_edge : ℝ) (apothem : ℝ) : ℝ :=
  (base_edge * Real.sqrt 2) / 2

/-- Theorem stating that the smallest radius of a sphere that can contain a regular quadrilateral pyramid
    with a base edge of 14 and an apothem of 12 is 7√2 -/
theorem smallest_sphere_radius_for_specific_pyramid :
  smallest_sphere_radius 14 12 = 7 * Real.sqrt 2 := by
  sorry

-- Remove the #eval statement as it's not computable

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_sphere_radius_for_specific_pyramid_l926_92645


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_between_circles_l926_92689

/-- The radius of the larger circle D -/
noncomputable def R : ℝ := 40

/-- The number of smaller circles in the ring -/
def n : ℕ := 8

/-- The radius of each smaller circle in the ring -/
noncomputable def s : ℝ := R * (Real.sqrt 2 - 1)

/-- The area of the region inside the larger circle D and outside all smaller circles -/
noncomputable def M : ℝ := -36800 * Real.pi + 25600 * Real.sqrt 2 * Real.pi

/-- Theorem stating that the floor of M is 1859 -/
theorem area_between_circles : ⌊M⌋ = 1859 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_between_circles_l926_92689


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rhombus_area_l926_92659

/-- The area of a rhombus with vertices (0, 3.5), (9, 0), (0, -3.5), and (-9, 0) is 63 square units. -/
theorem rhombus_area (x₁ y₁ x₂ y₂ x₃ y₃ x₄ y₄ : ℝ)
  (h₁ : x₁ = 0 ∧ y₁ = 3.5)
  (h₂ : x₂ = 9 ∧ y₂ = 0)
  (h₃ : x₃ = 0 ∧ y₃ = -3.5)
  (h₄ : x₄ = -9 ∧ y₄ = 0) :
  let area := (|x₂ - x₄| * |y₁ - y₃|) / 2
  area = 63 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_rhombus_area_l926_92659


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_discounted_price_theorem_l926_92672

/-- Given a good with an original price, prove that after applying successive discounts
    of 20%, 10%, and 5%, if the final price is Rs. 6500, then the original price is
    approximately Rs. 9502.92. -/
theorem discounted_price_theorem (P : ℝ) : 
  P * (1 - 0.2) * (1 - 0.1) * (1 - 0.05) = 6500 → P = 9502.92 := by
  intro h
  have : P = 6500 / (0.8 * 0.9 * 0.95) := by
    rw [← h]
    field_simp
    ring
  rw [this]
  norm_num
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_discounted_price_theorem_l926_92672


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_equilibrium_forces_l926_92601

-- Define the force type as a 2D vector
def Force := ℝ × ℝ

-- Define the magnitude of a force
noncomputable def magnitude (f : Force) : ℝ := Real.sqrt (f.1^2 + f.2^2)

-- State the theorem
theorem equilibrium_forces (F₁ F₂ F₃ : Force) :
  -- F₁ and F₂ have magnitude 6
  magnitude F₁ = 6 ∧ magnitude F₂ = 6 ∧
  -- F₁ and F₂ form a 120° angle
  F₁.1 * F₂.1 + F₁.2 * F₂.2 = -18 ∧
  -- The forces are in equilibrium
  F₁.1 + F₂.1 + F₃.1 = 0 ∧ F₁.2 + F₂.2 + F₃.2 = 0 →
  -- The magnitude of F₃ is 6
  magnitude F₃ = 6 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_equilibrium_forces_l926_92601


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_theorem_l926_92648

noncomputable def triangle_problem (A B C : ℝ) (a b c : ℝ) : Prop :=
  let S := 5 * Real.sqrt 3
  Real.cos (2 * A) - 3 * Real.cos (B + C) = 1 ∧
  b = 5 ∧
  (1 / 2) * b * c * Real.sin A = S ∧
  Real.sin B * Real.sin C = 5 / 7

theorem triangle_theorem :
  ∀ (A B C : ℝ) (a b c : ℝ),
    0 < A ∧ A < Real.pi ∧
    0 < B ∧ B < Real.pi ∧
    0 < C ∧ C < Real.pi ∧
    a > 0 ∧ b > 0 ∧ c > 0 ∧
    A + B + C = Real.pi →
    triangle_problem A B C a b c := by
  sorry

#check triangle_theorem

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_theorem_l926_92648


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circumscribed_circle_area_eq_48pi_l926_92665

/-- The area of a circle circumscribed about an equilateral triangle with side length s units -/
noncomputable def circumscribedCircleArea (s : ℝ) : ℝ :=
  Real.pi * (2 * s / (3 * Real.sqrt 3))^2

/-- Theorem: The area of a circle circumscribed about an equilateral triangle with side length 12 units is 48π -/
theorem circumscribed_circle_area_eq_48pi :
  circumscribedCircleArea 12 = 48 * Real.pi := by
  sorry

-- Remove the #eval statement as it's not computable

end NUMINAMATH_CALUDE_ERRORFEEDBACK_circumscribed_circle_area_eq_48pi_l926_92665


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_candy_theorem_l926_92688

def candy_problem (initial_candy initial_chocolate eaten_candy1 eaten_candy2 shared_candy eaten_chocolate bought_chocolate : ℤ) : ℕ :=
  let remaining_candy := initial_candy - eaten_candy1 - eaten_candy2 - shared_candy
  let remaining_chocolate := initial_chocolate - eaten_chocolate + bought_chocolate
  (remaining_chocolate - remaining_candy).natAbs

theorem candy_theorem :
  candy_problem 250 175 38 36 12 16 28 = 23 := by
  rfl

#eval candy_problem 250 175 38 36 12 16 28

end NUMINAMATH_CALUDE_ERRORFEEDBACK_candy_theorem_l926_92688


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_wire_length_for_specific_rods_l926_92639

/-- The minimum length of wire required to encircle two rods -/
noncomputable def wire_length (d₁ d₂ : ℝ) : ℝ :=
  2 * Real.sqrt ((d₁/2 + d₂/2)^2 - (d₂/2 - d₁/2)^2) + 
  (5 * Real.pi / 3) * d₁ +
  (2 * Real.pi / 3) * d₂

/-- Theorem stating the minimum length of wire for rods with diameters 10 and 20 inches -/
theorem wire_length_for_specific_rods : 
  wire_length 10 20 = 20 * Real.sqrt 2 + 170 * Real.pi / 6 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_wire_length_for_specific_rods_l926_92639


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_point_coordinates_l926_92695

noncomputable def fourth_quadrant (x y : ℝ) : Prop := x > 0 ∧ y < 0

noncomputable def distance_to_axes (x y : ℝ) : ℝ := max (|x|) (|y|)

theorem point_coordinates :
  ∀ (x y : ℝ), fourth_quadrant x y → distance_to_axes x y = 4 → x = 4 ∧ y = -4 :=
by
  intros x y h1 h2
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_point_coordinates_l926_92695


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_expression_equals_five_l926_92697

-- Define the new operations
noncomputable def new_add (a b : ℝ) : ℝ := a * b
noncomputable def new_sub (a b : ℝ) : ℝ := a + b
noncomputable def new_mul (a b : ℝ) : ℝ := a / b
noncomputable def new_div (a b : ℝ) : ℝ := a - b

-- Define the expression using the new operations
noncomputable def expression : ℝ := new_div (new_sub (new_add 6 (new_mul 8 3)) 9) 25

-- Theorem statement
theorem expression_equals_five : expression = 5 := by
  -- Expand the definition of expression
  unfold expression
  -- Expand the definitions of new operations
  unfold new_div new_sub new_add new_mul
  -- Simplify the arithmetic
  simp [div_eq_mul_inv]
  -- The proof is omitted
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_expression_equals_five_l926_92697


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_gold_tokens_l926_92602

/-- Represents the number of tokens of each color --/
structure Tokens where
  red : ℕ
  blue : ℕ
  gold : ℕ

/-- Represents the exchange rules at the booths --/
inductive Exchange
  | redToGold : Exchange  -- 3 red → 1 gold + 2 blue
  | blueToGold : Exchange -- 2 blue → 1 gold + 1 red

/-- Applies an exchange to the current token state --/
def applyExchange (t : Tokens) (e : Exchange) : Tokens :=
  match e with
  | Exchange.redToGold => 
      if t.red ≥ 3 then { red := t.red - 3, blue := t.blue + 2, gold := t.gold + 1 }
      else t
  | Exchange.blueToGold => 
      if t.blue ≥ 2 then { red := t.red + 1, blue := t.blue - 2, gold := t.gold + 1 }
      else t

/-- Checks if any more exchanges can be made --/
def canExchange (t : Tokens) : Bool :=
  t.red ≥ 3 ∨ t.blue ≥ 2

/-- Applies a list of exchanges to the initial token state --/
def applyExchanges (initialTokens : Tokens) (exchanges : List Exchange) : Tokens :=
  exchanges.foldl applyExchange initialTokens

/-- The main theorem to prove --/
theorem max_gold_tokens : ∃ (finalTokens : Tokens),
  (∃ (exchanges : List Exchange),
    (applyExchanges { red := 90, blue := 60, gold := 0 } exchanges = finalTokens) ∧
    ¬(canExchange finalTokens)) ∧
  finalTokens.gold = 148 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_gold_tokens_l926_92602


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_of_function_l926_92642

/-- The function f(t) = (4^t - 2t)t / 16^t has a maximum value of 1/8 for real t -/
theorem max_value_of_function :
  ∃ (t : ℝ), (λ t : ℝ ↦ (4^t - 2*t)*t / 16^t) t = 1/8 ∧
  ∀ (s : ℝ), (4^s - 2*s)*s / 16^s ≤ 1/8 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_of_function_l926_92642


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_shaded_area_is_one_fifteenth_l926_92638

/-- The sum of the infinite geometric series representing the shaded area of the square -/
noncomputable def shaded_area_sum : ℝ := 1 / 4 * (1 / (1 - 1 / 16))

/-- The initial shaded area is 1/4 of the total area -/
axiom initial_shaded_area : (1 : ℝ) / 4 = 1 / 4

/-- In each iteration, 1/4 of the previously shaded area is shaded again -/
axiom shading_ratio : (1 : ℝ) / 16 = 1 / 16

/-- The theorem stating that the shaded area sum equals 1/15 -/
theorem shaded_area_is_one_fifteenth : shaded_area_sum = 1 / 15 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_shaded_area_is_one_fifteenth_l926_92638


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_work_completion_time_l926_92610

/-- The time taken to complete a work given the following conditions:
  * Amit can complete the work in 15 days
  * Ananthu can complete the work in 30 days
  * Amit works for 3 days and then leaves
  * Ananthu completes the remaining work
-/
noncomputable def total_time (amit_days : ℝ) (ananthu_days : ℝ) (amit_worked : ℝ) : ℝ :=
  let amit_rate := 1 / amit_days
  let ananthu_rate := 1 / ananthu_days
  let amit_work := amit_rate * amit_worked
  let remaining_work := 1 - amit_work
  let ananthu_time := remaining_work / ananthu_rate
  amit_worked + ananthu_time

theorem work_completion_time :
  total_time 15 30 3 = 7.8 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_work_completion_time_l926_92610


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_of_function_l926_92680

theorem min_value_of_function (x : ℝ) (h : x > 1) : 
  (∀ y : ℝ, y = 3*x + 1/(x-1) → y ≥ 2*Real.sqrt 3 + 3) ∧ 
  (∃ y : ℝ, y = 3*x + 1/(x-1) ∧ y = 2*Real.sqrt 3 + 3) :=
by
  sorry

#check min_value_of_function

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_of_function_l926_92680


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_positive_omega_l926_92646

theorem min_positive_omega : ∃ (ω : ℝ), ω > 0 ∧
  (∀ (x : ℝ), Real.sin (ω * (x + Real.pi / 2) - Real.pi / 4) = Real.cos (ω * x)) ∧
  (∀ (ω' : ℝ), ω' > 0 → 
    (∀ (x : ℝ), Real.sin (ω' * (x + Real.pi / 2) - Real.pi / 4) = Real.cos (ω' * x)) → 
    ω ≤ ω') ∧
  ω = 3 / 2 := by
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_positive_omega_l926_92646


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_intersection_and_cyclic_quadrilateral_l926_92696

/-- A straight line in the plane -/
structure Line where

/-- A circle in the plane -/
structure Circle where

/-- A point in the plane -/
structure Point where

/-- Defines the relationship of a circle touching a line at a point -/
def touches_at_point (c : Circle) (l : Line) (p : Point) : Prop := sorry

/-- Defines the relationship of a circle touching another circle at a point -/
def touches_circle_at_point (c1 c2 : Circle) (p : Point) : Prop := sorry

/-- Defines a cyclic quadrilateral -/
def is_cyclic_quadrilateral (a b c d : Point) : Prop := sorry

/-- Defines the intersection of two lines -/
def lines_intersect_at (p1 p2 p3 p4 : Point) (p : Point) : Prop := sorry

/-- Defines a point lying on a circle -/
def point_on_circle (p : Point) (c : Circle) : Prop := sorry

theorem circle_intersection_and_cyclic_quadrilateral 
  (g : Line) (k₁ k₂ k₃ : Circle) (A B C D : Point) :
  touches_at_point k₁ g A →
  touches_at_point k₂ g B →
  touches_circle_at_point k₃ k₁ D →
  touches_circle_at_point k₃ k₂ C →
  ∃ X : Point, 
    is_cyclic_quadrilateral A B C D ∧ 
    lines_intersect_at A D B C X ∧
    point_on_circle X k₃ := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_intersection_and_cyclic_quadrilateral_l926_92696


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_infinite_solutions_l926_92623

/-- A sequence of real numbers -/
def Sequence := ℕ+ → ℝ

/-- Definition of a geometric progression -/
def IsGeometricProgression (a : Sequence) : Prop :=
  ∃ q : ℝ, ∀ n : ℕ+, a (n + 1) = q * a n

/-- The system of equations represented by the matrix -/
def SystemOfEquations (a : Sequence) : Set (ℝ × ℝ) :=
  { (x, y) | a 1 * x + a 2 * y = a 4 ∧ a 5 * x + a 6 * y = a 8 }

/-- The theorem statement -/
theorem infinite_solutions
  (a : Sequence)
  (h : IsGeometricProgression a) :
  Infinite (SystemOfEquations a) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_infinite_solutions_l926_92623


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_train_crossing_time_l926_92661

/-- Calculates the time taken for a train to cross a platform -/
noncomputable def time_to_cross (train_length : ℝ) (platform_length : ℝ) (speed : ℝ) : ℝ :=
  (train_length + platform_length) / speed

theorem train_crossing_time 
  (train_length : ℝ) 
  (first_platform_length : ℝ) 
  (second_platform_length : ℝ) 
  (time_second_platform : ℝ) 
  (h1 : train_length = 150)
  (h2 : first_platform_length = 150)
  (h3 : second_platform_length = 250)
  (h4 : time_second_platform = 20)
  : time_to_cross train_length first_platform_length 
      ((train_length + second_platform_length) / time_second_platform) = 15 := by
  sorry

#check train_crossing_time

end NUMINAMATH_CALUDE_ERRORFEEDBACK_train_crossing_time_l926_92661


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_ratio_is_10_9_l926_92676

-- Define the squares and their properties
def square_WXYZ : ℝ → (ℝ × ℝ) × (ℝ × ℝ) × (ℝ × ℝ) × (ℝ × ℝ) := sorry
def square_IJKL : ℝ → (ℝ × ℝ) × (ℝ × ℝ) × (ℝ × ℝ) × (ℝ × ℝ) := sorry

-- Define the side length of square WXYZ
def side_length_WXYZ : ℝ := 10

-- Define the condition for point I
def point_I_condition (square : ℝ → (ℝ × ℝ) × (ℝ × ℝ) × (ℝ × ℝ) × (ℝ × ℝ)) : Prop :=
  let ((w₁, w₂), (x₁, x₂), _, _) := square side_length_WXYZ
  2 * (w₁ - x₁) = x₁ - w₁

-- Define the area ratio
noncomputable def area_ratio (square1 square2 : ℝ → (ℝ × ℝ) × (ℝ × ℝ) × (ℝ × ℝ) × (ℝ × ℝ)) : ℝ :=
  let ((i₁, i₂), (j₁, j₂), _, _) := square1 side_length_WXYZ
  let area_IJKL := (j₁ - i₁) ^ 2 + (j₂ - i₂) ^ 2
  let area_WXYZ := side_length_WXYZ ^ 2
  area_IJKL / area_WXYZ

-- Theorem statement
theorem area_ratio_is_10_9 :
  point_I_condition square_IJKL →
  area_ratio square_IJKL square_WXYZ = 10 / 9 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_ratio_is_10_9_l926_92676


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_limit_proof_l926_92657

/-- The limit of (e^x - e^(3x)) / (sin(3x) - tan(2x)) as x approaches 0 is -2 -/
theorem limit_proof : 
  ∀ ε > 0, ∃ δ > 0, ∀ x : ℝ, x ≠ 0 → |x| < δ → 
    |(Real.exp x - Real.exp (3*x)) / (Real.sin (3*x) - Real.tan (2*x)) + 2| < ε := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_limit_proof_l926_92657


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_time_difference_proportional_to_180_over_vc_l926_92618

/-- Represents a contestant in the race -/
structure Contestant where
  speed : ℝ
  headStart : ℝ

/-- Represents the race setup -/
structure Race where
  distance : ℝ
  contestants : Fin 3 → Contestant

/-- Calculate the time difference between two contestants finishing the race -/
noncomputable def timeDifference (race : Race) (i j : Fin 3) : ℝ :=
  let ci := race.contestants i
  let cj := race.contestants j
  (race.distance - cj.headStart) / cj.speed - (race.distance - ci.headStart) / ci.speed

/-- The main theorem about the time difference between contestants A and C -/
theorem time_difference_proportional_to_180_over_vc (race : Race) :
  race.distance = 1200 →
  (race.contestants 0).speed = (5 / 3) * (race.contestants 2).speed →
  (race.contestants 0).headStart = 100 →
  (race.contestants 2).headStart = 0 →
  ∃ k : ℝ, timeDifference race 0 2 = k * (180 / (race.contestants 2).speed) := by
  sorry

#check time_difference_proportional_to_180_over_vc

end NUMINAMATH_CALUDE_ERRORFEEDBACK_time_difference_proportional_to_180_over_vc_l926_92618


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l926_92604

noncomputable def is_odd_function (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = -f x

noncomputable def f : ℝ → ℝ := fun x =>
  if x < 0 then x^2 - 2/x
  else if x = 0 then 0
  else -(x^2) - 2/x

theorem f_properties :
  is_odd_function f ∧ f 0 = 0 ∧ f 1 = -3 ∧
  (∀ x, x < 0 → f x = x^2 - 2/x) ∧
  (∀ x, x > 0 → f x = -(x^2) - 2/x) := by
  sorry

#check f_properties

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l926_92604


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_points_range_l926_92649

open Real Set

theorem intersection_points_range (ω φ : ℝ) : 
  φ > 0 ∧ 
  abs φ < π / 2 ∧
  ω = π / sqrt 2 →
  (∃ (P M N : ℝ × ℝ), 
    P.1 ∈ Icc 0 (5 * sqrt 2 / 2) ∧
    M.1 ∈ Icc 0 (5 * sqrt 2 / 2) ∧
    N.1 ∈ Icc 0 (5 * sqrt 2 / 2) ∧
    P.2 = sin (ω * P.1 + φ) ∧
    P.2 = cos (ω * P.1 + φ) ∧
    M.2 = sin (ω * M.1 + φ) ∧
    M.2 = cos (ω * M.1 + φ) ∧
    N.2 = sin (ω * N.1 + φ) ∧
    N.2 = cos (ω * N.1 + φ) ∧
    (N.1 - P.1) * (M.1 - P.1) + (N.2 - P.2) * (M.2 - P.2) = 0) →
  φ ∈ Icc (-π/4) (π/4) := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_points_range_l926_92649


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_median_perpendicular_area_l926_92636

-- Define the triangle PQR
structure Triangle (P Q R : ℝ × ℝ) : Prop where
  is_triangle : P ≠ Q ∧ Q ≠ R ∧ R ≠ P

-- Define a median of a triangle
def is_median (P Q R S : ℝ × ℝ) : Prop :=
  ∃ (M : ℝ × ℝ), M = ((Q.1 + R.1) / 2, (Q.2 + R.2) / 2) ∧ S = M

-- Define perpendicularity of two line segments
def are_perpendicular (P Q R S : ℝ × ℝ) : Prop :=
  (Q.1 - P.1) * (S.1 - R.1) + (Q.2 - P.2) * (S.2 - R.2) = 0

-- Define the length of a line segment
noncomputable def length (P Q : ℝ × ℝ) : ℝ :=
  Real.sqrt ((Q.1 - P.1)^2 + (Q.2 - P.2)^2)

-- Define the area of a triangle
noncomputable def triangle_area (P Q R : ℝ × ℝ) : ℝ :=
  abs ((Q.1 - P.1) * (R.2 - P.2) - (R.1 - P.1) * (Q.2 - P.2)) / 2

-- State the theorem
theorem median_perpendicular_area 
  (P Q R S T : ℝ × ℝ) 
  (h_triangle : Triangle P Q R)
  (h_median_PS : is_median P Q R S)
  (h_median_QT : is_median Q R P T)
  (h_perpendicular : are_perpendicular P S Q T)
  (h_PS_length : length P S = 18)
  (h_QT_length : length Q T = 24) :
  triangle_area P Q R = 576 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_median_perpendicular_area_l926_92636


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parallel_vectors_lambda_l926_92641

/-- Prove that if vector a = (2, 5) is parallel to vector b = (l, 4), then l = 8/5 -/
theorem parallel_vectors_lambda (l : ℚ) : 
  let a : Fin 2 → ℚ := ![2, 5]
  let b : Fin 2 → ℚ := ![l, 4]
  (∃ (k : ℚ), k ≠ 0 ∧ a = k • b) → l = 8/5 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parallel_vectors_lambda_l926_92641


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_chord_length_l926_92671

/-- The curve C in the x-y plane -/
def C (x y : ℝ) : Prop := (x - 2)^2 + y^2 = 1

/-- The line l in the x-y plane -/
def l (x y : ℝ) : Prop := 4*x + 3*y - 10 = 0

/-- The length of the chord formed by the intersection of C and l -/
noncomputable def chord_length : ℝ := 16/15

/-- Theorem stating that the length of the chord formed by the intersection of C and l is 16/15 -/
theorem intersection_chord_length :
  ∃ (A B : ℝ × ℝ),
    C A.1 A.2 ∧ C B.1 B.2 ∧
    l A.1 A.2 ∧ l B.1 B.2 ∧
    A ≠ B ∧
    Real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2) = chord_length := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_chord_length_l926_92671


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_is_three_fourths_l926_92651

def K : Finset ℤ := {-2, -1, 1, -3}

def not_passes_third_quadrant (k : ℤ) : Prop := k < 0

instance : DecidablePred not_passes_third_quadrant :=
  fun k => Int.decLt k 0

def probability_not_passing_third_quadrant : ℚ :=
  (K.filter not_passes_third_quadrant).card / K.card

theorem probability_is_three_fourths :
  probability_not_passing_third_quadrant = 3/4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_is_three_fourths_l926_92651


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_P_Q_no_relationship_l926_92613

-- Define the sets P and Q
def P : Set ℝ := {x : ℝ | ∃ y : ℝ, y = x^2}
def Q : Set (ℝ × ℝ) := {p : ℝ × ℝ | p.2 = p.1^2}

-- Helper function to convert elements of P to elements of Q
def P_to_Q (x : ℝ) : ℝ × ℝ := (x, x^2)

-- Helper function to convert elements of Q to elements of P
def Q_to_P (p : ℝ × ℝ) : ℝ := p.1

-- Theorem statement
theorem P_Q_no_relationship : 
  (¬∀ x ∈ P, P_to_Q x ∈ Q) ∧ 
  (¬∀ p ∈ Q, Q_to_P p ∈ P) ∧ 
  P ≠ Set.range Q_to_P :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_P_Q_no_relationship_l926_92613


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_orthogonal_vectors_l926_92628

theorem min_value_orthogonal_vectors (x y : ℝ) : 
  let a : Fin 2 → ℝ := ![x - 1, 2]
  let b : Fin 2 → ℝ := ![4, y]
  (∀ i, i < 2 → a i * b i = 0) →
  8 ≤ (16 : ℝ)^x + (4 : ℝ)^y ∧ ∃ x y, (16 : ℝ)^x + (4 : ℝ)^y = 8 := by
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_orthogonal_vectors_l926_92628


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cosine_ratio_theorem_l926_92622

theorem cosine_ratio_theorem (α : ℝ) 
  (h1 : α ∈ Set.Ioo 0 (π/4))
  (h2 : Real.cos (α - π/4) = 4/5) :
  Real.cos (2*α) / Real.sin (α + π/4) = 6/5 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cosine_ratio_theorem_l926_92622


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_alpha_for_point_l926_92655

/-- If point P(-3, 4) lies on the terminal side of angle α, then cos α = -3/5 -/
theorem cos_alpha_for_point (α : ℝ) : 
  (∃ (P : ℝ × ℝ), P = (-3, 4) ∧ P.1 = -3 * Real.cos α ∧ P.2 = 4 * Real.sin α) →
  Real.cos α = -3/5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_alpha_for_point_l926_92655


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_positive_period_abs_sin_l926_92656

noncomputable def f (x : ℝ) := |Real.sin x|

theorem smallest_positive_period_abs_sin :
  ∃ (p : ℝ), p > 0 ∧ (∀ x, f (x + p) = f x) ∧
  (∀ q, q > 0 → (∀ x, f (x + q) = f x) → p ≤ q) ∧
  p = Real.pi := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_positive_period_abs_sin_l926_92656


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_segment_division_l926_92644

/-- The midpoint of two points -/
def is_midpoint (M A B : ℝ × ℝ) : Prop :=
  M.1 = (A.1 + B.1) / 2 ∧ M.2 = (A.2 + B.2) / 2

/-- Given a segment AB divided into three equal parts by points C(3,4) and D(5,6),
    prove that the coordinates of A are (1,2) and the coordinates of B are (7,8). -/
theorem segment_division (A B C D : ℝ × ℝ) :
  C = (3, 4) →
  D = (5, 6) →
  is_midpoint C A D →
  is_midpoint D C B →
  A = (1, 2) ∧ B = (7, 8) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_segment_division_l926_92644


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_store_profit_is_ten_yuan_l926_92692

/-- Represents a calculator with its cost and selling price -/
structure Calculator where
  cost : ℚ
  price : ℚ

/-- Calculates the profit percentage for a calculator -/
def profitPercentage (c : Calculator) : ℚ :=
  (c.price - c.cost) / c.cost * 100

/-- The store's transaction with two calculators -/
def storeTransaction (c1 c2 : Calculator) : ℚ :=
  c1.price + c2.price - c1.cost - c2.cost

theorem store_profit_is_ten_yuan :
  ∀ (c1 c2 : Calculator),
    c1.price = 80 ∧
    c2.price = 80 ∧
    profitPercentage c1 = 60 ∧
    profitPercentage c2 = -20 →
    storeTransaction c1 c2 = 10 := by
  sorry

#eval profitPercentage { cost := 50, price := 80 }
#eval profitPercentage { cost := 100, price := 80 }
#eval storeTransaction { cost := 50, price := 80 } { cost := 100, price := 80 }

end NUMINAMATH_CALUDE_ERRORFEEDBACK_store_profit_is_ten_yuan_l926_92692


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_odot_problem_l926_92603

-- Define the operation ⊙
noncomputable def odot (a b : ℝ) : ℝ := a^3 / (b + 1)

-- State the theorem
theorem odot_problem :
  let x := odot (odot 3 2) 1
  let y := odot 3 (odot 2 1)
  x - y = 3591 / 10 := by
  -- Expand the definitions of x and y
  unfold odot
  -- Perform algebraic manipulations
  simp [pow_three]
  -- The rest of the proof would go here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_odot_problem_l926_92603


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_thirteen_sequences_l926_92616

-- Define the triangle T
def T : Set (ℝ × ℝ) := {(0, 0), (4, 1), (0, 2)}

-- Define the transformations
def rotate180 (p : ℝ × ℝ) : ℝ × ℝ := (4 - p.1, 2 - p.2)
def reflectX (p : ℝ × ℝ) : ℝ × ℝ := (p.1, -p.2)
def reflectY (p : ℝ × ℝ) : ℝ × ℝ := (-p.1, p.2)
def translate (p : ℝ × ℝ) : ℝ × ℝ := (p.1 + 2, p.2)

-- Define the set of transformations
def transformations : List (ℝ × ℝ → ℝ × ℝ) := [rotate180, reflectX, reflectY, translate]

-- Define a sequence of three transformations
def transformation_sequence := Fin 3 → (ℝ × ℝ → ℝ × ℝ)

-- Define the identity transformation on the set T
def is_identity (seq : transformation_sequence) : Prop :=
  ∀ p ∈ T, (seq 2 (seq 1 (seq 0 p))) = p

-- The theorem to prove
theorem thirteen_sequences :
  ∃ (S : List transformation_sequence), 
    (∀ seq ∈ S, (∀ i, seq i ∈ transformations) ∧ is_identity seq) ∧ 
    S.length = 13 :=
sorry

#check thirteen_sequences

end NUMINAMATH_CALUDE_ERRORFEEDBACK_thirteen_sequences_l926_92616


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_train_length_calculation_l926_92677

/-- Calculates the length of a train given its speed, the speed of a man walking in the opposite direction, and the time it takes for the train to pass the man. -/
theorem train_length_calculation (train_speed_kmph : ℝ) (man_speed_kmph : ℝ) (passing_time_sec : ℝ) :
  train_speed_kmph = 54.99520038396929 →
  man_speed_kmph = 5 →
  passing_time_sec = 6 →
  let relative_speed_kmph := train_speed_kmph + man_speed_kmph
  let relative_speed_mps := relative_speed_kmph * (5 / 18)
  let train_length_m := relative_speed_mps * passing_time_sec
  train_length_m = 99.99180063994882 := by
  sorry

-- Remove the #eval line as it's not necessary for the theorem proof

end NUMINAMATH_CALUDE_ERRORFEEDBACK_train_length_calculation_l926_92677


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_projection_problem_l926_92686

noncomputable def vector_projection (u v : ℝ × ℝ) : ℝ × ℝ :=
  let dot_product := u.1 * v.1 + u.2 * v.2
  let v_norm_squared := v.1 * v.1 + v.2 * v.2
  (dot_product / v_norm_squared * v.1, dot_product / v_norm_squared * v.2)

theorem projection_problem (v : ℝ × ℝ) 
  (h : vector_projection (1, 4) v = (2/5, -8/5)) :
  vector_projection (3, -2) v = (11/17, -44/17) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_projection_problem_l926_92686


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hall_volume_l926_92687

/-- Represents a rectangular hall with given dimensions -/
structure RectangularHall where
  length : ℝ
  width : ℝ
  height : ℝ

/-- Calculates the volume of a rectangular hall -/
def volume (hall : RectangularHall) : ℝ :=
  hall.length * hall.width * hall.height

/-- Calculates the sum of the areas of the floor and ceiling -/
def floorCeilingArea (hall : RectangularHall) : ℝ :=
  2 * hall.length * hall.width

/-- Calculates the sum of the areas of the four walls -/
def wallsArea (hall : RectangularHall) : ℝ :=
  2 * hall.height * (hall.length + hall.width)

/-- Theorem stating the volume of the hall given the conditions -/
theorem hall_volume (hall : RectangularHall) 
    (h1 : hall.length = 15)
    (h2 : hall.width = 12)
    (h3 : floorCeilingArea hall = wallsArea hall) :
    ∃ ε > 0, |volume hall - 8004| < ε := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_hall_volume_l926_92687


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_ratio_range_l926_92626

-- Define a triangle ABC
structure Triangle where
  A : Real
  B : Real
  C : Real
  a : Real
  b : Real
  c : Real
  sum_angles : A + B + C = Real.pi
  positive_sides : a > 0 ∧ b > 0 ∧ c > 0
  sine_law : a / (Real.sin A) = b / (Real.sin B)

-- Define the theorem
theorem triangle_ratio_range {t : Triangle} (h : t.A = 2 * t.B) :
  2 < (t.c / t.b) + (2 * t.b / t.a) ∧ (t.c / t.b) + (2 * t.b / t.a) < 4 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_ratio_range_l926_92626


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_g_expression_g_formula_l926_92679

-- Define the functions f and g
def f : ℝ → ℝ := λ x ↦ 2 * x + 3
def g : ℝ → ℝ := λ x ↦ 2 * x - 1

-- State the theorem
theorem g_expression :
  (∀ x, g (x + 2) = f x) := by
  intro x
  calc
    g (x + 2) = 2 * (x + 2) - 1 := rfl
    _         = 2 * x + 4 - 1   := by ring
    _         = 2 * x + 3       := by ring
    _         = f x             := rfl

-- Verify that g(x) = 2x - 1
theorem g_formula : ∀ x, g x = 2 * x - 1 := by
  intro x
  rfl


end NUMINAMATH_CALUDE_ERRORFEEDBACK_g_expression_g_formula_l926_92679


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_larger_number_is_ten_l926_92660

theorem larger_number_is_ten (x y : ℤ) : 
  x * y = 40 →
  x + y = 14 →
  x ≥ 0 →
  y ≥ 0 →
  abs (x - y) ≤ 6 →
  max x y = 10 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_larger_number_is_ten_l926_92660


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_projection_theorem_l926_92669

def v : ℝ → Fin 3 → ℝ := λ z i => match i with
  | 0 => 0
  | 1 => 4
  | 2 => z

def u : Fin 3 → ℝ := λ i => match i with
  | 0 => -4
  | 1 => 6
  | 2 => -2

def projection (y : Fin 3 → ℝ) : ℝ → Fin 3 → ℝ := λ c i =>
  c * y i

def dot_product (x y : Fin 3 → ℝ) : ℝ :=
  (x 0 * y 0) + (x 1 * y 1) + (x 2 * y 2)

theorem projection_theorem (z : ℝ) : 
  (∃ c : ℝ, c = 16/56 ∧ projection u c = projection u ((dot_product (v z) u) / (dot_product u u))) → z = 4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_projection_theorem_l926_92669


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_dihedral_angle_at_apex_is_45_degrees_l926_92629

/-- A regular square pyramid with coinciding centers of inscribed and circumscribed spheres -/
structure RegularSquarePyramid where
  /-- The centers of the inscribed and circumscribed spheres coincide -/
  centers_coincide : Bool

/-- The dihedral angle at the apex of a regular square pyramid -/
def dihedral_angle_at_apex (p : RegularSquarePyramid) : Real := 
  sorry -- Placeholder definition

/-- Theorem: The dihedral angle at the apex of a regular square pyramid
    with coinciding centers of inscribed and circumscribed spheres is 45° -/
theorem dihedral_angle_at_apex_is_45_degrees (p : RegularSquarePyramid)
  (h : p.centers_coincide = true) :
  dihedral_angle_at_apex p = 45 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_dihedral_angle_at_apex_is_45_degrees_l926_92629


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_baker_cakes_sold_l926_92634

theorem baker_cakes_sold (total_cakes : ℕ) (total_pastries : ℕ) (pastries_sold : ℕ) (difference : ℕ) : 
  pastries_sold + difference = 97 :=
  by
  -- Define the given conditions
  have h1 : total_cakes = 14 := by sorry
  have h2 : total_pastries = 153 := by sorry
  have h3 : pastries_sold = 8 := by sorry
  have h4 : difference = 89 := by sorry

  -- Define the number of cakes sold
  let cakes_sold := pastries_sold + difference

  -- Prove that the number of cakes sold is 97
  calc
    cakes_sold = pastries_sold + difference := rfl
    _ = 8 + 89 := by rw [h3, h4]
    _ = 97 := by rfl


end NUMINAMATH_CALUDE_ERRORFEEDBACK_baker_cakes_sold_l926_92634


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_digits_of_A_l926_92632

def A : ℕ := 2^63 * 4^25 * 5^106 - 2^22 * 4^44 * 5^105 - 1

theorem sum_of_digits_of_A : (A.digits 10).sum = 959 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_digits_of_A_l926_92632


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_a_l926_92605

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := x + Real.sin x

-- State the theorem
theorem range_of_a (a : ℝ) : 
  (f (a - 1) + f (2 * a^2) ≤ 0) ↔ (-1 ≤ a ∧ a ≤ 1/2) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_a_l926_92605


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trapezium_area_is_55_over_4_l926_92698

/-- Represents the side lengths of the three squares in centimeters -/
structure SquareSides where
  small : ℚ
  medium : ℚ
  large : ℚ

/-- Calculates the area of the trapezium formed by parts of three squares -/
noncomputable def trapeziumArea (sides : SquareSides) : ℚ :=
  let p := sides.small * (sides.large / (sides.small + sides.medium + sides.large))
  let q := (sides.small + sides.medium) * (sides.large / (sides.small + sides.medium + sides.large))
  let height := sides.medium
  (p + q) * height / 2

/-- Theorem stating that the area of the trapezium is 55/4 cm² -/
theorem trapezium_area_is_55_over_4 (sides : SquareSides) 
    (h1 : sides.small = 3)
    (h2 : sides.medium = 5)
    (h3 : sides.large = 8) : 
  trapeziumArea sides = 55 / 4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_trapezium_area_is_55_over_4_l926_92698


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_BC_length_l926_92654

/-- Triangle ABC with given properties -/
structure Triangle where
  -- Side lengths
  AB : ℝ
  AC : ℝ
  BC : ℝ
  -- Median from A to midpoint of BC
  AM : ℝ
  -- Height from A to BC
  AH : ℝ
  -- Properties
  AB_eq : AB = 5
  AC_eq : AC = 8
  AM_eq : AM = 5
  AH_eq : AH = 4
  -- Median formula
  median_formula : AM^2 = (1/2) * (2*AB^2 + 2*AC^2 - BC^2)
  -- Area formula using height
  area_formula : (1/2) * BC * AH = (1/2) * AB * AC * Real.sin (Real.arccos ((AB^2 + AC^2 - BC^2) / (2 * AB * AC)))

/-- The length of BC in the given triangle is √78 -/
theorem BC_length (t : Triangle) : t.BC = Real.sqrt 78 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_BC_length_l926_92654


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_equation_l926_92699

-- Define the curve
noncomputable def f (x : ℝ) : ℝ := -5 * Real.exp x + 3

-- Define the point of tangency
def point : ℝ × ℝ := (0, -2)

-- Define the derivative of the curve
noncomputable def f' (x : ℝ) : ℝ := -5 * Real.exp x

-- State the theorem
theorem tangent_line_equation :
  ∃ (a b c : ℝ), 
    (a * point.1 + b * point.2 + c = 0) ∧ 
    (∀ x y : ℝ, y = f x → (a * x + b * y + c = 0) ↔ 
      (y - point.2 = f' point.1 * (x - point.1))) ∧
    (a = 5 ∧ b = 1 ∧ c = 2) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_equation_l926_92699


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_nSn_l926_92617

/-- Represents an arithmetic sequence -/
structure ArithmeticSequence where
  a : ℕ → ℚ
  d : ℚ
  arithmetic_property : ∀ n, a (n + 1) = a n + d

/-- Sum of first n terms of an arithmetic sequence -/
def S (seq : ArithmeticSequence) (n : ℕ) : ℚ :=
  n * (seq.a 1 + seq.a n) / 2

theorem min_nSn (seq : ArithmeticSequence) (m : ℕ) (h_m : m ≥ 2) 
    (h_Sm1 : S seq (m - 1) = -2)
    (h_Sm : S seq m = 0)
    (h_Sp1 : S seq (m + 1) = 3) :
  ∃ n : ℕ, ∀ k : ℕ, k * S seq k ≥ n * S seq n ∧ n * S seq n = -9 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_nSn_l926_92617


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_difference_l926_92630

noncomputable def floor (x : ℝ) : ℤ := ⌊x⌋
noncomputable def frac (x : ℝ) : ℝ := x - ⌊x⌋

theorem solution_difference (x y : ℝ) 
  (eq1 : floor x + frac y = 3.2)
  (eq2 : frac x + floor y = 4.7) :
  |x - y| = 0.5 := by
  sorry

#check solution_difference

end NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_difference_l926_92630


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_slope_from_inclination_l926_92619

noncomputable def inclination_angle : ℝ := 30 * Real.pi / 180

theorem slope_from_inclination :
  Real.tan inclination_angle = Real.sqrt 3 / 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_slope_from_inclination_l926_92619


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_herd_division_l926_92620

theorem herd_division (herd : ℕ) : 
  (herd / 3 + herd / 6 + herd / 9 + 8 = herd) ∧ 
  (herd % 3 = 0) ∧ (herd % 6 = 0) ∧ (herd % 9 = 0) →
  herd = 144 := by
  intro h
  -- The proof steps would go here
  sorry

#check herd_division

end NUMINAMATH_CALUDE_ERRORFEEDBACK_herd_division_l926_92620


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_length_l926_92693

/-- The circle C with equation (x-4)²+(y+2)²=5 -/
def circle_C : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | (p.1 - 4)^2 + (p.2 + 2)^2 = 5}

/-- The line y = x + 2 -/
def line_L : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.2 = p.1 + 2}

/-- The center of the circle C -/
def center_C : ℝ × ℝ := (4, -2)

/-- The radius of the circle C -/
noncomputable def radius_C : ℝ := Real.sqrt 5

/-- The point M on line_L closest to center_C -/
noncomputable def point_M : ℝ × ℝ :=
  let d := (|4 - (-2) + 2|) / Real.sqrt 2
  (4 - d / Real.sqrt 2, -2 + d / Real.sqrt 2)

/-- The length of the tangent line segment from point_M to circle_C -/
noncomputable def length_MN : ℝ :=
  let d := (|4 - (-2) + 2|) / Real.sqrt 2
  Real.sqrt (d^2 - radius_C^2)

theorem tangent_length :
  length_MN = 3 * Real.sqrt 3 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_length_l926_92693


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_sum_arithmetic_sequence_l926_92611

noncomputable section

/-- An arithmetic sequence with common difference d -/
def arithmetic_sequence (a₁ : ℝ) (d : ℝ) : ℕ → ℝ :=
  λ n => a₁ + (n - 1) * d

/-- Sum of the first n terms of an arithmetic sequence -/
def sum_arithmetic_sequence (a₁ : ℝ) (d : ℝ) (n : ℕ) : ℝ :=
  n * (2 * a₁ + (n - 1) * d) / 2

/-- The quadratic function associated with the inequality -/
def quadratic_function (a₁ d c : ℝ) (x : ℝ) : ℝ :=
  (d / 2) * x^2 + (a₁ - d / 2) * x + c

theorem max_sum_arithmetic_sequence
  (a₁ d c : ℝ)
  (h_solution_set : ∀ x, 0 ≤ x ∧ x ≤ 22 ↔ quadratic_function a₁ d c x ≥ 0) :
  ∃ (n : ℕ), n = 11 ∧
    ∀ (m : ℕ), m > 11 →
      sum_arithmetic_sequence a₁ d n ≥ sum_arithmetic_sequence a₁ d m :=
by sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_sum_arithmetic_sequence_l926_92611


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_limit_sin_3x_over_x_l926_92650

theorem limit_sin_3x_over_x :
  ∀ ε > 0, ∃ δ > 0, ∀ x : ℝ, 
    0 < |x| ∧ |x| < δ → |((Real.sin (3 * x)) / x) - 3| < ε :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_limit_sin_3x_over_x_l926_92650


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_alpha_plus_beta_l926_92615

theorem sin_alpha_plus_beta (α β : ℝ) 
  (h1 : Real.cos (π/4 - α) = 3/5)
  (h2 : Real.sin (5*π/4 + β) = -12/13)
  (h3 : α ∈ Set.Ioo (π/4) (3*π/4))
  (h4 : β ∈ Set.Ioo 0 (π/4)) :
  Real.sin (α + β) = 56/65 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_alpha_plus_beta_l926_92615


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_line_intersection_distance_difference_l926_92621

-- Define the parabola
def parabola (x y : ℝ) : Prop := y^2 = 4*x

-- Define the line
def line (x y : ℝ) : Prop := y = x - 1

-- Define the focus
def focus : ℝ × ℝ := (1, 0)

-- Define the intersection points
def intersection_points : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | parabola p.1 p.2 ∧ line p.1 p.2}

-- State the theorem
theorem parabola_line_intersection_distance_difference :
  ∃ A B : ℝ × ℝ, A ∈ intersection_points ∧ B ∈ intersection_points ∧
    |dist focus B - dist focus A| = 4 * Real.sqrt 2 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_line_intersection_distance_difference_l926_92621


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trailingZeros_2006_factorial_l926_92653

/-- The number of factors of 5 in n! -/
def factorsOf5 (n : ℕ) : ℕ :=
  Finset.sum (Finset.range (n + 1)) (fun k => k / 5)

/-- The number of consecutive zeros at the end of n! in base 10 -/
def trailingZeros (n : ℕ) : ℕ := factorsOf5 n

theorem trailingZeros_2006_factorial : trailingZeros 2006 = 500 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_trailingZeros_2006_factorial_l926_92653


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_mySequence_general_term_l926_92643

def mySequence (n : ℕ) : ℚ :=
  match n with
  | 0 => 3
  | n + 1 => mySequence n + n + 1

theorem mySequence_general_term (n : ℕ) :
  mySequence n = (n^2 + n + 4) / 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_mySequence_general_term_l926_92643


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_work_completion_time_l926_92652

noncomputable def ab_time : ℝ := 12

noncomputable def a_time : ℝ := 20

noncomputable def c_time : ℝ := 30

noncomputable def b_fraction : ℝ := 1/2

noncomputable def c_fraction : ℝ := 1/3

noncomputable def abc_time : ℝ := 90/7

theorem work_completion_time :
  (1 / a_time) + (1 / ab_time - 1 / a_time) * b_fraction + (1 / c_time) * c_fraction = 1 / abc_time := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_work_completion_time_l926_92652


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_of_A_and_B_l926_92666

-- Define set A
def A : Set ℝ := {y : ℝ | ∃ x : ℝ, y = 3^x}

-- Define set B
def B : Set ℝ := {x : ℝ | x^2 - 4 ≤ 0}

-- Theorem statement
theorem intersection_of_A_and_B : A ∩ B = {x : ℝ | 0 < x ∧ x ≤ 2} := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_of_A_and_B_l926_92666


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_no_solutions_sin_equation_l926_92694

theorem no_solutions_sin_equation :
  ¬ ∃ (x y : ℝ), 0 < x ∧ x < π/2 ∧ 0 < y ∧ y < π/2 ∧ Real.sin x + Real.sin y = Real.sin (x * y) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_no_solutions_sin_equation_l926_92694


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_operations_correspondence_l926_92625

-- Define the operations
noncomputable def multiplication (a b : ℝ) : ℝ := a * b

noncomputable def division (a b : ℝ) : ℝ := a / b

noncomputable def exponentiation (a b : ℝ) : ℝ := a ^ b

noncomputable def square_root (a : ℝ) : ℝ := Real.sqrt a

def absolute_value (a : ℝ) : ℝ := abs a

-- Theorem stating that these operations correspond to the given symbols
theorem operations_correspondence :
  ∃ (mul div exp sqr abs : ℝ → ℝ → ℝ),
    (mul = multiplication) ∧
    (div = division) ∧
    (exp = exponentiation) ∧
    (∀ x, sqr x x = square_root x) ∧
    (∀ x, abs x x = absolute_value x) := by
  sorry

#check operations_correspondence

end NUMINAMATH_CALUDE_ERRORFEEDBACK_operations_correspondence_l926_92625


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_valid_set_size_l926_92667

def isValidSet (S : Finset ℕ) : Prop :=
  S ⊆ Finset.range 10 ∧
  ∃ N : ℕ, ∀ n > N, ∃ a b : ℕ,
    a > 0 ∧ b > 0 ∧
    n = a + b ∧
    (∀ d ∈ Nat.digits 10 a, d ∈ S) ∧
    (∀ d ∈ Nat.digits 10 b, d ∈ S)

theorem smallest_valid_set_size :
  (∃ S : Finset ℕ, isValidSet S ∧ S.card = 5) ∧
  (∀ S : Finset ℕ, isValidSet S → S.card ≥ 5) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_valid_set_size_l926_92667


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_properties_l926_92627

noncomputable def arithmetic_sequence (a₁ d : ℝ) (n : ℕ) : ℝ := a₁ + (n - 1 : ℝ) * d

noncomputable def arithmetic_sum (a₁ d : ℝ) (n : ℕ) : ℝ := n * (2 * a₁ + (n - 1 : ℝ) * d) / 2

theorem arithmetic_sequence_properties :
  ∃ (a₁ d : ℝ),
    -- Given conditions
    arithmetic_sequence a₁ d 3 = 24 ∧
    arithmetic_sequence a₁ d 6 = 18 ∧
    -- Prove the general term
    (∀ n : ℕ, arithmetic_sequence a₁ d n = 30 - 2 * ↑n) ∧
    -- Prove the sum formula
    (∀ n : ℕ, arithmetic_sum a₁ d n = -(↑n)^2 + 29 * ↑n) ∧
    -- Prove the maximum value and where it occurs
    (∀ n : ℕ, arithmetic_sum a₁ d n ≤ 210) ∧
    arithmetic_sum a₁ d 14 = 210 ∧
    arithmetic_sum a₁ d 15 = 210 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_properties_l926_92627


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_distance_to_upper_vertex_l926_92684

/-- The ellipse C defined by x²/5 + y² = 1 -/
def C : Set (ℝ × ℝ) := {p | p.1^2 / 5 + p.2^2 = 1}

/-- The upper vertex B of the ellipse C -/
def B : ℝ × ℝ := (0, 1)

/-- The distance between two points in ℝ² -/
noncomputable def distance (p q : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2)

theorem max_distance_to_upper_vertex :
  ∃ (max_dist : ℝ), max_dist = 5/2 ∧
    ∀ (P : ℝ × ℝ), P ∈ C → distance P B ≤ max_dist :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_distance_to_upper_vertex_l926_92684


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_monotonic_decreasing_interval_l926_92633

-- Define the function (marked as noncomputable due to Real.log)
noncomputable def f (x : ℝ) : ℝ := Real.log (x^2 - 4*x + 3)

-- Define the domain of the function
def dom (x : ℝ) : Prop := x < 1 ∨ x > 3

-- Theorem statement
theorem monotonic_decreasing_interval :
  ∀ x ∈ {x : ℝ | dom x}, 
    (∀ y ∈ {y : ℝ | dom y}, x < y → f x > f y) ↔ x ∈ Set.Iio 1 :=
by
  sorry

#check monotonic_decreasing_interval

end NUMINAMATH_CALUDE_ERRORFEEDBACK_monotonic_decreasing_interval_l926_92633


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_providers_assignment_l926_92606

/-- The number of ways to assign different service providers to children -/
def assign_providers (total_providers : ℕ) (num_children : ℕ) : ℕ :=
  (List.range num_children).foldl (λ acc i => acc * (total_providers - i)) 1

/-- Theorem: Assigning 4 different providers from 25 to 4 children results in 303600 ways -/
theorem providers_assignment :
  assign_providers 25 4 = 303600 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_providers_assignment_l926_92606


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_inequality_triangle_equality_l926_92607

-- Define the Triangle structure
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  h_positive : 0 < a ∧ 0 < b ∧ 0 < c
  h_triangle_inequality : a + b > c ∧ b + c > a ∧ c + a > b

-- Define the necessary functions
def Triangle.inradius (t : Triangle) : ℝ := sorry
def Triangle.area (t : Triangle) : ℝ := sorry
def Triangle.semiperimeter (t : Triangle) : ℝ := sorry
def Triangle.circumradius (t : Triangle) : ℝ := sorry
def Triangle.isEquilateral (t : Triangle) : Prop := sorry

theorem triangle_inequality (ABC : Triangle) (r : ℝ) (S : ℝ) (p : ℝ) (R : ℝ) 
  (h_r : r = ABC.inradius)
  (h_S : S = ABC.area)
  (h_p : p = ABC.semiperimeter)
  (h_R : R = ABC.circumradius) :
  r ≤ (Real.sqrt (Real.sqrt (3 * S))) / 3 ∧ 
  (Real.sqrt (Real.sqrt (3 * S))) / 3 ≤ (Real.sqrt 3 / 9) * p ∧ 
  (Real.sqrt 3 / 9) * p ≤ R / 2 :=
sorry

theorem triangle_equality (ABC : Triangle) (r : ℝ) (S : ℝ) (p : ℝ) (R : ℝ) 
  (h_r : r = ABC.inradius)
  (h_S : S = ABC.area)
  (h_p : p = ABC.semiperimeter)
  (h_R : R = ABC.circumradius) :
  (r = (Real.sqrt (Real.sqrt (3 * S))) / 3 ∧ 
   (Real.sqrt (Real.sqrt (3 * S))) / 3 = (Real.sqrt 3 / 9) * p ∧ 
   (Real.sqrt 3 / 9) * p = R / 2) ↔ 
  ABC.isEquilateral :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_inequality_triangle_equality_l926_92607


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hexagon_area_inequality_l926_92691

open Real

theorem hexagon_area_inequality (a b c : ℝ) (α β γ : ℝ) (t : ℝ) 
  (h_positive : a > 0 ∧ b > 0 ∧ c > 0)
  (h_angles : α > 0 ∧ β > 0 ∧ γ > 0)
  (h_sum_angles : α + β + γ = π)
  (h_area : t = (1/2) * a * b * sin γ)
  (h_sides : a = b * sin γ / sin α ∧ c = a * sin β / sin α) :
  (1/2) * (a^2 + b^2 + c^2) * (sin α + sin β + sin γ) + 4 * t ≥ 13 * t := by
  sorry

#check hexagon_area_inequality

end NUMINAMATH_CALUDE_ERRORFEEDBACK_hexagon_area_inequality_l926_92691

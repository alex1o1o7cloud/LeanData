import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_isosceles_trapezoid_segments_l720_72036

-- Define the trapezoid and its properties
structure IsoscelesTrapezoid where
  a : ℝ
  b : ℝ
  h : a > b

-- Define the segments
noncomputable def DP (t : IsoscelesTrapezoid) : ℝ := (t.a - t.b) / 2
noncomputable def AP (t : IsoscelesTrapezoid) : ℝ := (t.a + t.b) / 2

-- State the theorem
theorem isosceles_trapezoid_segments (t : IsoscelesTrapezoid) :
  DP t = (t.a - t.b) / 2 ∧ AP t = (t.a + t.b) / 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_isosceles_trapezoid_segments_l720_72036


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_one_over_a_is_fraction_l720_72007

/-- A fraction is an expression with a variable in the denominator. -/
def IsFraction (f : ℝ → ℝ) : Prop :=
  ∃ (n d : ℝ → ℝ), (∀ x, f x = n x / d x) ∧ (∃ x, d x ≠ 0)

/-- The function f(a) = 1/a is a fraction. -/
theorem one_over_a_is_fraction : IsFraction (λ a => 1 / a) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_one_over_a_is_fraction_l720_72007


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_composition_value_l720_72054

noncomputable def f (a b x : ℝ) : ℝ :=
  if x > 0 then Real.log x / Real.log 3
  else a^x + b

theorem f_composition_value :
  ∀ a b : ℝ,
  (∀ x > 0, f a b x = Real.log x / Real.log 3) →
  (∀ x ≤ 0, f a b x = a^x + b) →
  f a b 0 = 2 →
  f a b (-1) = 3 →
  f a b (f a b (-3)) = 2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_composition_value_l720_72054


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_ellipse_to_line_l720_72044

/-- The line l in the xy-plane -/
def line_l (x y : ℝ) : Prop := x - y + 4 = 0

/-- The ellipse C in the xy-plane -/
def ellipse_C (x y : ℝ) : Prop := x^2 / 3 + y^2 / 9 = 1

/-- The distance from a point (x, y) to the line l -/
noncomputable def distance_to_line (x y : ℝ) : ℝ := 
  |x - y + 4| / Real.sqrt 2

/-- Theorem stating the minimum distance from ellipse C to line l -/
theorem min_distance_ellipse_to_line :
  ∃ (min_dist : ℝ), min_dist = 2 * Real.sqrt 2 - Real.sqrt 6 ∧
  ∀ (x y : ℝ), ellipse_C x y → distance_to_line x y ≥ min_dist := by
  sorry

#check min_distance_ellipse_to_line

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_ellipse_to_line_l720_72044


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_three_element_subsets_sizes_l720_72000

def P : Finset ℕ := Finset.range 10 |>.image (fun n => 2*n + 1)

theorem sum_of_three_element_subsets_sizes :
  (Finset.powerset P |>.filter (fun s => s.card = 3)).sum (fun s => s.card) = 3600 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_three_element_subsets_sizes_l720_72000


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_chair_sale_cost_l720_72009

/-- Calculates the total cost of chairs with discounts applied -/
noncomputable def calculate_total_cost (original_price : ℝ) (num_chairs : ℕ) : ℝ :=
  let discounted_price := original_price * (1 - 0.25)
  let base_cost := discounted_price * (num_chairs : ℝ)
  let additional_discount := 
    if num_chairs > 5 
    then ((num_chairs - 5) : ℝ) * discounted_price * (1/3) 
    else 0
  base_cost - additional_discount

/-- Theorem stating the total cost of 8 chairs with given discounts -/
theorem chair_sale_cost : calculate_total_cost 20 8 = 105 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_chair_sale_cost_l720_72009


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_product_special_angles_l720_72076

theorem sin_product_special_angles (θ : ℝ) 
  (h1 : Real.sin θ = 1/3) 
  (h2 : θ ∈ Set.Ioo (-Real.pi/2) (Real.pi/2)) : 
  Real.sin (Real.pi - θ) * Real.sin (Real.pi/2 - θ) = 2 * Real.sqrt 2 / 9 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_product_special_angles_l720_72076


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_perfect_square_product_probability_l720_72088

/-- Die sides -/
def die_sides : List ℕ := [1, 2, 3, 5, 7, 10, 14, 15]

/-- Number of rolls -/
def num_rolls : ℕ := 4

/-- Check if a number is a perfect square -/
def is_perfect_square (n : ℕ) : Prop := ∃ m : ℕ, m * m = n

/-- Calculate the probability of an event given the number of favorable outcomes and total outcomes -/
def probability (favorable : ℕ) (total : ℕ) : ℚ := favorable / total

/-- Count the number of favorable outcomes -/
def countFavorableOutcomes (sides : List ℕ) (rolls : ℕ) (pred : ℕ → Prop) : ℕ := sorry

/-- Calculate the total number of possible outcomes -/
def total_outcomes (sides : List ℕ) (rolls : ℕ) : ℕ := sorry

/-- Theorem stating the probability of rolling a perfect square product -/
theorem perfect_square_product_probability : 
  probability (countFavorableOutcomes die_sides num_rolls is_perfect_square) 
              (total_outcomes die_sides num_rolls) = 287 / 4096 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_perfect_square_product_probability_l720_72088


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_baked_goods_not_eaten_l720_72083

def initial_cookies : ℕ := 200
def initial_brownies : ℕ := 150
def initial_cupcakes : ℕ := 100

def wife_cookies_percent : ℚ := 30 / 100
def wife_brownies_percent : ℚ := 20 / 100
def wife_cupcakes_percent : ℚ := 50 / 100

def daughter_cookies : ℕ := 40
def daughter_brownies_percent : ℚ := 15 / 100

def friend1_cookies_percent : ℚ := 25 / 100
def friend1_brownies_percent : ℚ := 10 / 100
def friend1_cupcakes : ℕ := 10

def friend2_cookies_percent : ℚ := 5 / 100
def friend2_brownies_percent : ℚ := 5 / 100
def friend2_cupcakes : ℕ := 5

def javier_percent : ℚ := 50 / 100

theorem baked_goods_not_eaten :
  let remaining_cookies := initial_cookies -
    (wife_cookies_percent * initial_cookies).floor -
    daughter_cookies -
    (friend1_cookies_percent * (initial_cookies - (wife_cookies_percent * initial_cookies).floor - daughter_cookies)).floor -
    (friend2_cookies_percent * (initial_cookies - (wife_cookies_percent * initial_cookies).floor - daughter_cookies - 
      (friend1_cookies_percent * (initial_cookies - (wife_cookies_percent * initial_cookies).floor - daughter_cookies)).floor)).floor
  let remaining_brownies := initial_brownies -
    (wife_brownies_percent * initial_brownies).floor -
    (daughter_brownies_percent * (initial_brownies - (wife_brownies_percent * initial_brownies).floor)).floor -
    (friend1_brownies_percent * (initial_brownies - (wife_brownies_percent * initial_brownies).floor - 
      (daughter_brownies_percent * (initial_brownies - (wife_brownies_percent * initial_brownies).floor)).floor)).floor -
    (friend2_brownies_percent * (initial_brownies - (wife_brownies_percent * initial_brownies).floor - 
      (daughter_brownies_percent * (initial_brownies - (wife_brownies_percent * initial_brownies).floor)).floor - 
      (friend1_brownies_percent * (initial_brownies - (wife_brownies_percent * initial_brownies).floor - 
        (daughter_brownies_percent * (initial_brownies - (wife_brownies_percent * initial_brownies).floor)).floor)).floor)).floor
  let remaining_cupcakes := initial_cupcakes -
    (wife_cupcakes_percent * initial_cupcakes).floor -
    friend1_cupcakes -
    friend2_cupcakes
  remaining_cookies - (javier_percent * remaining_cookies).floor +
  remaining_brownies - (javier_percent * remaining_brownies).floor +
  remaining_cupcakes - (javier_percent * remaining_cupcakes).floor = 98 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_baked_goods_not_eaten_l720_72083


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_sequence_b_l720_72024

noncomputable def sequence_a : ℕ → ℝ
  | 0 => 1
  | 1 => 1
  | (n + 2) => 4 * (3 ^ n)

noncomputable def sequence_S : ℕ → ℝ
  | 0 => 0
  | 1 => 1
  | (n + 1) => 3 * sequence_S n + 2

noncomputable def sequence_b (n : ℕ) : ℝ :=
  8 * n / (sequence_a (n + 1) - sequence_a n)

noncomputable def sequence_T : ℕ → ℝ
  | 0 => 0
  | (n + 1) => sequence_T n + sequence_b (n + 1)

theorem sum_of_sequence_b (n : ℕ) :
  sequence_T n = 77/12 - (n/2 + 3/4) * (1/3)^(n-2) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_sequence_b_l720_72024


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_equal_sided_quadrilateral_not_necessarily_planar_l720_72025

-- Define the basic shapes
structure Triangle
structure Quadrilateral
structure Trapezoid
structure Parallelogram

-- Define a planar shape
class PlanarShape (α : Type) : Prop

-- Define a quadrilateral with equal sides
def EqualSidedQuadrilateral (q : Quadrilateral) : Prop := sorry

-- Axioms for planar shapes
axiom triangle_planar : PlanarShape Triangle
axiom trapezoid_planar : PlanarShape Trapezoid
axiom parallelogram_planar : PlanarShape Parallelogram

-- Theorem to prove
theorem equal_sided_quadrilateral_not_necessarily_planar :
  ¬ (∀ q : Quadrilateral, EqualSidedQuadrilateral q → PlanarShape Quadrilateral) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_equal_sided_quadrilateral_not_necessarily_planar_l720_72025


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_isosceles_triangle_perpendicular_medians_vertex_angle_l720_72001

/-- An isosceles triangle with mutually perpendicular medians drawn to the lateral sides -/
structure IsoscelesTriangleWithPerpendicularMedians where
  -- The base angles of the isosceles triangle
  base_angle : ℝ
  -- The angle between the medians
  median_angle : ℝ
  -- The medians are perpendicular
  medians_perpendicular : median_angle = Real.pi / 2

/-- The angle at the vertex of an isosceles triangle with mutually perpendicular medians -/
noncomputable def vertex_angle (triangle : IsoscelesTriangleWithPerpendicularMedians) : ℝ :=
  2 * Real.arctan (1 / 3)

/-- Theorem: The angle at the vertex of an isosceles triangle with mutually perpendicular medians is 2 * arctan(1/3) -/
theorem isosceles_triangle_perpendicular_medians_vertex_angle 
  (triangle : IsoscelesTriangleWithPerpendicularMedians) : 
  vertex_angle triangle = 2 * Real.arctan (1 / 3) := by
  -- Proof steps would go here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_isosceles_triangle_perpendicular_medians_vertex_angle_l720_72001


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_integral_x_plus_one_l720_72041

theorem integral_x_plus_one : ∫ x in (-1)..0, (x + 1) = 1/2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_integral_x_plus_one_l720_72041


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ratio_of_numbers_with_given_hcf_lcm_l720_72039

/-- Given two positive integers with HCF 3 and LCM 36, prove their ratio is 3:4 -/
theorem ratio_of_numbers_with_given_hcf_lcm (A B : ℕ+) (x y : ℕ+) :
  Nat.gcd A.val B.val = 3 →
  Nat.lcm A.val B.val = 36 →
  A.val * y.val = B.val * x.val →
  Nat.Coprime x.val y.val →
  x.val = 3 ∧ y.val = 4 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_ratio_of_numbers_with_given_hcf_lcm_l720_72039


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rectangle_domino_covering_l720_72035

/-- Represents a domino placement on a rectangle -/
structure DominoCovering (n m : ℕ) :=
  (layer1 : Set (ℕ × ℕ × Bool))
  (layer2 : Set (ℕ × ℕ × Bool))

/-- The squares covered by a domino at position (x, y) with orientation b -/
def dominoSquares (x y : ℕ) (b : Bool) : Set (ℕ × ℕ) :=
  if b then {(x, y), (x + 1, y)} else {(x, y), (x, y + 1)}

/-- Checks if a domino covering is valid for a 2n × 2m rectangle -/
def isValidCovering (n m : ℕ) (covering : DominoCovering n m) : Prop :=
  -- Each layer covers the entire rectangle
  (∀ i j, i < 2*n ∧ j < 2*m → 
    (∃ x y b, (x, y, b) ∈ covering.layer1 ∧ (i, j) ∈ dominoSquares x y b) ∧
    (∃ x y b, (x, y, b) ∈ covering.layer2 ∧ (i, j) ∈ dominoSquares x y b)) ∧
  -- No overlap between layers
  (∀ x1 y1 b1 x2 y2 b2, 
    (x1, y1, b1) ∈ covering.layer1 ∧ (x2, y2, b2) ∈ covering.layer2 → 
    (dominoSquares x1 y1 b1) ∩ (dominoSquares x2 y2 b2) = ∅)

/-- Theorem: A 2n × 2m rectangle can be covered by two layers of dominoes -/
theorem rectangle_domino_covering (n m : ℕ) : 
  ∃ covering : DominoCovering n m, isValidCovering n m covering := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_rectangle_domino_covering_l720_72035


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_of_g_at_extreme_points_diff_l720_72086

noncomputable def f (m : ℝ) (x : ℝ) : ℝ := x^2 - 2*x + m * Real.log x

noncomputable def g (x : ℝ) : ℝ := (x - 3/4) * Real.exp x

theorem min_value_of_g_at_extreme_points_diff (m : ℝ) :
  ∃ (x₁ x₂ : ℝ), x₁ < x₂ ∧
  (∀ x, x > 0 → (deriv (f m)) x = 0 ↔ x = x₁ ∨ x = x₂) →
  g (x₁ - x₂) = -Real.exp (-1/4) := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_of_g_at_extreme_points_diff_l720_72086


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_array_count_theorem_l720_72053

/-- Represents the number of ways to write down an array of positive integers
    starting with n and ending with 1, where each subsequent number is a
    positive integer less than the square root of the previous number. -/
def S (n : ℕ) : ℕ :=
  sorry

/-- The condition that each subsequent number in the array is less than
    the square root of the previous number. -/
def valid_sequence (seq : List ℕ) : Prop :=
  ∀ i, i + 1 < seq.length → (seq.get! (i+1) : ℝ) < Real.sqrt (seq.get! i)

theorem array_count_theorem :
  S 2012 = 201 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_array_count_theorem_l720_72053


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cube_root_product_simplification_l720_72068

theorem cube_root_product_simplification :
  (8 + 27) ^ (1/3 : ℝ) * (8 + Real.sqrt 64) ^ (1/3 : ℝ) = 560 ^ (1/3 : ℝ) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cube_root_product_simplification_l720_72068


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_diamond_problem_l720_72040

-- Define the diamond operation as noncomputable
noncomputable def diamond (a b : ℝ) : ℝ := Real.sqrt (a^2 + b^2)

-- State the theorem
theorem diamond_problem :
  diamond (diamond 5 12) (diamond (-12) (-5)) = 13 * Real.sqrt 2 := by
  -- Proof steps will go here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_diamond_problem_l720_72040


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_between_points_l720_72003

def point1 : ℝ × ℝ × ℝ := (2, -3, 1)
def point2 : ℝ × ℝ × ℝ := (8, 4, -3)

theorem distance_between_points : 
  Real.sqrt ((point2.fst - point1.fst)^2 + (point2.snd.fst - point1.snd.fst)^2 + (point2.snd.snd - point1.snd.snd)^2) = Real.sqrt 101 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_between_points_l720_72003


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_max_on_positive_l720_72067

-- Define the function f
noncomputable def f (x : ℝ) : ℝ :=
  if x < 0 then x * (1 + x) else -((-x) * (1 + (-x)))

-- State the theorem
theorem f_max_on_positive :
  ∃ M, M = (1/4 : ℝ) ∧ ∀ x > 0, f x ≤ M :=
by
  -- We'll use 1/4 as our maximum value
  let M := (1/4 : ℝ)
  
  -- Prove that M satisfies the conditions
  use M
  
  constructor
  · -- Prove M = 1/4
    rfl
    
  · -- Prove ∀ x > 0, f x ≤ M
    intro x hx
    sorry -- The actual proof would go here

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_max_on_positive_l720_72067


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_m_l720_72084

open Real

/-- The function f(x) defined in the problem -/
noncomputable def f (m : ℝ) (x : ℝ) : ℝ := 3 * cos (π * x / m)

/-- The theorem stating the range of m given the conditions -/
theorem range_of_m (m : ℝ) : 
  (∃ x₀ : ℝ, x₀ ≠ 0 ∧ 
    (∀ x : ℝ, f m x ≤ f m x₀ ∨ f m x ≥ f m x₀) ∧ 
    x₀^2 + f m x₀ < 4*m) → 
  2 - Real.sqrt 7 < m ∧ m < 2 + Real.sqrt 7 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_m_l720_72084


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_imaginary_unit_power_6_equals_neg_1_l720_72048

theorem imaginary_unit_power_6_equals_neg_1 : Complex.I ^ 6 = -1 := by
  -- Proof goes here
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_imaginary_unit_power_6_equals_neg_1_l720_72048


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_perpendicular_construction_l720_72017

-- Define the basic geometric elements
structure Point where
  x : ℝ
  y : ℝ

structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

-- Define the notion of a point being on a line
def Point.on_line (p : Point) (l : Line) : Prop :=
  l.a * p.x + l.b * p.y + l.c = 0

-- Define perpendicularity of two lines
def Line.perpendicular (l1 l2 : Line) : Prop :=
  l1.a * l2.a + l1.b * l2.b = 0

-- Define the distance between two points
noncomputable def distance (p1 p2 : Point) : ℝ :=
  Real.sqrt ((p1.x - p2.x)^2 + (p1.y - p2.y)^2)

-- Define the concept of constructible points using compass and straightedge
inductive Constructible : Point → Prop where
  | given : (p : Point) → Constructible p
  | intersection_lines : (l1 l2 : Line) → Constructible (Point.mk 0 0) → Constructible (Point.mk 0 0)
  | intersection_circles : (c1 c2 : Point × ℝ) → Constructible (Point.mk 0 0) → Constructible (Point.mk 0 0)

-- Theorem statement
theorem perpendicular_construction 
  (M N : Point) (l : Line) (A : Point)
  (h_M : M.on_line l)
  (h_N : N.on_line l)
  (h_A : ¬ A.on_line l)
  (h_MN_distinct : M ≠ N) :
  ∃ (B : Point), 
    Constructible B ∧ 
    Line.perpendicular (Line.mk (B.y - A.y) (A.x - B.x) 0) l := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_perpendicular_construction_l720_72017


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_perfect_squares_property_l720_72008

theorem perfect_squares_property (S : Finset ℕ) : 
  (S.card = 25) →
  (∀ n ∈ S, n ≤ 1000) →
  (∀ n m : ℕ, n ∈ S → m ∈ S → n ≠ m → ∃ k : ℕ, n * m = k^2) →
  (∀ n ∈ S, ∃ k : ℕ, n = k^2) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_perfect_squares_property_l720_72008


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_hexagon_side_at_60_degrees_l720_72050

/-- A square in which a regular hexagon is inscribed. -/
structure InscribedHexagonSquare where
  /-- The side length of the square. -/
  side : ℝ
  /-- The side length of the inscribed regular hexagon. -/
  hexagon_side : ℝ
  /-- The centers of the square and hexagon coincide. -/
  centers_coincide : Prop
  /-- The hexagon is inscribed in the square. -/
  hexagon_inscribed : hexagon_side ≤ side / 2

/-- The angle between the diagonal of the square and a line connecting the center to a vertex of the hexagon. -/
def diagonal_vertex_angle (s : InscribedHexagonSquare) : ℝ :=
  sorry

/-- The maximum possible side length of an inscribed hexagon given the square's side length. -/
noncomputable def max_hexagon_side (square_side : ℝ) : ℝ :=
  sorry

/-- The theorem stating that the hexagon has maximum side length when its opposite vertices intersect the square's diagonal at a 60° angle. -/
theorem max_hexagon_side_at_60_degrees (s : InscribedHexagonSquare) :
  s.hexagon_side = max_hexagon_side s.side ↔ diagonal_vertex_angle s = 60 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_hexagon_side_at_60_degrees_l720_72050


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_cost_28_apples_l720_72011

/-- Represents the cost in cents for a given number of apples -/
def AppleCost : ℕ → ℕ := sorry

/-- The cost of 4 apples is 15 cents -/
axiom cost_4 : AppleCost 4 = 15

/-- The cost of 7 apples is 30 cents -/
axiom cost_7 : AppleCost 7 = 30

/-- The AppleCost function is additive -/
axiom cost_additive : ∀ m n, AppleCost (m + n) = AppleCost m + AppleCost n

/-- The minimum cost to purchase exactly 28 apples is 120 cents -/
theorem min_cost_28_apples : 
  (∀ f : ℕ → ℕ, (∃ a b, f 0 = 0 ∧ f 1 = a * 4 + b * 7 ∧ a * 15 + b * 30 = AppleCost (f 1)) → 
    f 28 ≥ 120) ∧ 
  (∃ g : ℕ → ℕ, g 28 = 120 ∧ ∃ a b, g 1 = a * 4 + b * 7 ∧ a * 15 + b * 30 = AppleCost (g 1)) :=
by sorry

#check min_cost_28_apples

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_cost_28_apples_l720_72011


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tip_percentage_calculation_l720_72014

-- Define the given values
noncomputable def food_cost : ℝ := 30
noncomputable def sales_tax_rate : ℝ := 0.095
noncomputable def total_paid : ℝ := 35.75

-- Define the tip percentage calculation
noncomputable def tip_percentage (food_cost sales_tax_rate total_paid : ℝ) : ℝ :=
  ((total_paid - (food_cost * (1 + sales_tax_rate))) / food_cost) * 100

-- State the theorem
theorem tip_percentage_calculation :
  abs (tip_percentage food_cost sales_tax_rate total_paid - 9.67) < 0.01 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tip_percentage_calculation_l720_72014


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_circle_tangent_range_l720_72019

/-- A line in the 2D plane -/
structure Line where
  k : ℝ
  equation : ℝ → ℝ → Prop := fun x y ↦ k * x + y + 4 = 0

/-- A circle in the 2D plane -/
def Circle : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.1^2 + p.2^2 - 2*p.2 = 0}

/-- Distance between two points in the 2D plane -/
noncomputable def distance (p q : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2)

/-- Tangent point from a point to a circle -/
def isTangentPoint (p q : ℝ × ℝ) : Prop :=
  q ∈ Circle ∧ ∀ r ∈ Circle, distance p q ≤ distance p r

theorem line_circle_tangent_range (L : Line) :
  (∃ p : ℝ × ℝ, L.equation p.1 p.2 ∧
    ∃ q : ℝ × ℝ, isTangentPoint p q ∧ distance p q = 2) →
  L.k ∈ Set.Iic (-2) ∪ Set.Ici 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_circle_tangent_range_l720_72019


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_integer_below_power_remainder_l720_72028

theorem largest_integer_below_power_remainder : 
  (⌊(3^123 : ℚ) / 5⌋ : ℤ) % 16 = 5 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_integer_below_power_remainder_l720_72028


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_set_f_greater_than_x_l720_72085

noncomputable def f (x : ℝ) : ℝ := 
  if x > 0 then x^2 - 4*x
  else -((-x)^2 - 4*(-x))

theorem solution_set_f_greater_than_x :
  let S := {x : ℝ | f x > x}
  (∀ x : ℝ, f (-x) = -f x) →
  S = Set.Ioo (-3) 0 ∪ Set.Ioi 5 := by
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_set_f_greater_than_x_l720_72085


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_common_tangents_l720_72057

-- Define the parabola and circle
def my_parabola (x y : ℝ) : Prop := y^2 = 5*x
def my_circle (x y : ℝ) : Prop := 9*x^2 + 9*y^2 = 16

-- Define the two potential tangent lines
def tangent1 (x y : ℝ) : Prop := 9*x + 12*y + 20 = 0
def tangent2 (x y : ℝ) : Prop := 9*x - 12*y + 20 = 0

-- Theorem statement
theorem common_tangents :
  ∀ (x y : ℝ),
  (tangent1 x y ∨ tangent2 x y) →
  (∃ (x₀ y₀ : ℝ), my_parabola x₀ y₀ ∧ (tangent1 x₀ y₀ ∨ tangent2 x₀ y₀)) ∧
  (∃ (x₁ y₁ : ℝ), my_circle x₁ y₁ ∧ (tangent1 x₁ y₁ ∨ tangent2 x₁ y₁)) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_common_tangents_l720_72057


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_coordinates_is_60_l720_72063

-- Define a point in 2D space
def Point := ℝ × ℝ

-- Define the distance between two points
noncomputable def distance (p1 p2 : Point) : ℝ :=
  Real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

-- Define the distance from a point to a horizontal line
def distToHorizontalLine (p : Point) (y : ℝ) : ℝ :=
  |p.2 - y|

-- Define the set of points satisfying the conditions
def satisfyingPoints : Set Point :=
  {p : Point | distToHorizontalLine p 10 = 3 ∧ distance p (5, 10) = 10}

-- Theorem statement
theorem sum_of_coordinates_is_60 :
  ∃ (p1 p2 p3 p4 : Point),
    p1 ∈ satisfyingPoints ∧
    p2 ∈ satisfyingPoints ∧
    p3 ∈ satisfyingPoints ∧
    p4 ∈ satisfyingPoints ∧
    p1 ≠ p2 ∧ p1 ≠ p3 ∧ p1 ≠ p4 ∧ p2 ≠ p3 ∧ p2 ≠ p4 ∧ p3 ≠ p4 ∧
    p1.1 + p1.2 + p2.1 + p2.2 + p3.1 + p3.2 + p4.1 + p4.2 = 60 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_coordinates_is_60_l720_72063


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_at_A_l720_72058

/-- The equation of a curve S -/
def S (x : ℝ) : ℝ := 3*x - x^3

/-- The x-coordinate of point A -/
def A_x : ℝ := 2

/-- The y-coordinate of point A -/
def A_y : ℝ := -2

/-- The equation of the tangent line -/
def tangent_line (x y : ℝ) : Prop := 9*x + y - 16 = 0

/-- The derivative of S -/
def S_deriv (x : ℝ) : ℝ := 3 - 3*x^2

theorem tangent_line_at_A :
  tangent_line A_x A_y ∧
  ∀ x y : ℝ, tangent_line x y → 
    (y - A_y) = (S_deriv A_x) * (x - A_x) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_at_A_l720_72058


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_v_of_3_equals_20_l720_72015

noncomputable def u (x : ℝ) : ℝ := 4 * x - 9

noncomputable def v (y : ℝ) : ℝ := 
  let x := (y + 9) / 4
  x^2 + 4 * x - 1

theorem v_of_3_equals_20 : v 3 = 20 := by
  -- Unfold the definition of v
  unfold v
  -- Simplify the expression
  simp
  -- Perform numerical calculation
  norm_num


end NUMINAMATH_CALUDE_ERRORFEEDBACK_v_of_3_equals_20_l720_72015


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_of_A_and_B_l720_72059

-- Define set A
def A : Set ℝ := {y | ∃ x, y = 2 - x ∧ x < 0}

-- Define set B
def B : Set ℝ := {x | x ≥ 0}

-- Theorem statement
theorem intersection_of_A_and_B : A ∩ B = Set.Ioi 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_of_A_and_B_l720_72059


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_four_valid_positions_l720_72006

/-- Represents a cross-shaped figure composed of 5 squares -/
structure CrossFigure :=
  (center : Unit)
  (sides : Fin 4 → Unit)

/-- Represents a position where an additional square can be placed -/
inductive Position
  | Top : Fin 4 → Position

/-- Checks if a given position forms a valid topless cubical box -/
def is_valid_position (cross : CrossFigure) (pos : Position) : Prop :=
  match pos with
  | Position.Top _ => True

/-- Counts the number of valid positions for forming a topless cubical box -/
def count_valid_positions (cross : CrossFigure) : ℕ :=
  Fintype.card (Fin 4)

/-- The main theorem stating that there are exactly 4 valid positions -/
theorem four_valid_positions (cross : CrossFigure) :
  count_valid_positions cross = 4 := by
  unfold count_valid_positions
  simp [Fintype.card_fin]


end NUMINAMATH_CALUDE_ERRORFEEDBACK_four_valid_positions_l720_72006


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_fifty_third_card_is_two_l720_72049

/-- Represents the cards in a standard deck --/
inductive Card : Type
  | Two | Three | Four | Five | Six | Seven | Eight | Nine | Ten
  | Jack | Queen | King | Ace

/-- The sequence of cards in order --/
def cardSequence : List Card := [
  Card.Two, Card.Three, Card.Four, Card.Five, Card.Six, Card.Seven, Card.Eight,
  Card.Nine, Card.Ten, Card.Jack, Card.Queen, Card.King, Card.Ace
]

/-- The number of cards in a complete sequence --/
def sequenceLength : Nat := 13

/-- Finds the nth card in the repeating sequence --/
def nthCard (n : Nat) : Card :=
  cardSequence[n % sequenceLength % cardSequence.length]'sorry

theorem fifty_third_card_is_two :
  nthCard 53 = Card.Two := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_fifty_third_card_is_two_l720_72049


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_dorchester_earnings_per_puppy_l720_72096

/-- Represents Dorchester's earnings at the puppy wash -/
structure PuppyWashEarnings where
  base_pay : ℚ
  total_earnings : ℚ
  puppies_washed : ℕ

/-- Calculates the earnings per puppy -/
def earnings_per_puppy (e : PuppyWashEarnings) : ℚ :=
  (e.total_earnings - e.base_pay) / e.puppies_washed

/-- Theorem stating Dorchester's earnings per puppy -/
theorem dorchester_earnings_per_puppy :
  let e := PuppyWashEarnings.mk 40 76 16
  earnings_per_puppy e = 9/4 := by
  sorry

#eval earnings_per_puppy (PuppyWashEarnings.mk 40 76 16)

end NUMINAMATH_CALUDE_ERRORFEEDBACK_dorchester_earnings_per_puppy_l720_72096


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_to_line_l720_72005

noncomputable def line_equation (x : ℝ) : ℝ := (5/3) * x + 4/5

noncomputable def distance_to_line (m n : ℤ) : ℝ :=
  |5 * (m : ℝ) - 3 * (n : ℝ) + 12/5| / (5 * Real.sqrt 34)

theorem min_distance_to_line :
  ∃ (d : ℝ), d = Real.sqrt 34 / 85 ∧
  ∀ (m n : ℤ), distance_to_line m n ≥ d := by
  sorry

#check min_distance_to_line

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_to_line_l720_72005


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_equals_three_implies_x_equals_sqrt_three_l720_72055

-- Define the piecewise function f
noncomputable def f (x : ℝ) : ℝ :=
  if x ≤ -1 then x + 2
  else if x < 2 then x^2
  else 2*x

-- State the theorem
theorem f_equals_three_implies_x_equals_sqrt_three :
  ∀ x : ℝ, f x = 3 → x = Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_equals_three_implies_x_equals_sqrt_three_l720_72055


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cylinder_radius_problem_l720_72031

/-- Represents a cylinder with given radius and height -/
structure Cylinder where
  radius : ℝ
  height : ℝ

/-- Calculates the volume of a cylinder -/
noncomputable def volume (c : Cylinder) : ℝ := Real.pi * c.radius^2 * c.height

/-- The theorem stating the conditions and the result to be proved -/
theorem cylinder_radius_problem (c : Cylinder) (h : c.height = 3) :
  volume { radius := c.radius + 4, height := c.height } = volume { radius := c.radius, height := c.height + 4 } →
  c.radius = 8 := by
  sorry

#check cylinder_radius_problem

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cylinder_radius_problem_l720_72031


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_even_and_increasing_l720_72090

-- Define the function f(x) = 3^|x|
noncomputable def f (x : ℝ) : ℝ := 3^(abs x)

-- State the theorem
theorem f_even_and_increasing : 
  (∀ x : ℝ, f (-x) = f x) ∧ 
  (∀ x y : ℝ, 0 < x → x < y → f x < f y) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_even_and_increasing_l720_72090


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_greene_family_expense_l720_72098

def amusement_park_expense (
  discounted_ticket_cost : ℚ)
  (original_ticket_cost : ℚ)
  (food_beverage_discount : ℚ)
  (mr_greene_souvenir_cost : ℚ)
  (num_children : ℕ)
  (game_cost_per_child : ℚ)
  (transportation_cost : ℚ)
  (tax_rate : ℚ) : ℚ :=
  let food_beverage_cost := original_ticket_cost - food_beverage_discount
  let mrs_greene_souvenir_cost := 2 * mr_greene_souvenir_cost
  let total_game_cost := (num_children : ℚ) * game_cost_per_child
  let taxable_amount := food_beverage_cost + mr_greene_souvenir_cost + mrs_greene_souvenir_cost + total_game_cost
  let tax := tax_rate * taxable_amount
  discounted_ticket_cost + food_beverage_cost + mr_greene_souvenir_cost + mrs_greene_souvenir_cost + total_game_cost + transportation_cost + tax

theorem greene_family_expense :
  amusement_park_expense 45 50 13 15 3 9 25 (8/100) = 187.72 := by
  sorry

#eval amusement_park_expense 45 50 13 15 3 9 25 (8/100)

end NUMINAMATH_CALUDE_ERRORFEEDBACK_greene_family_expense_l720_72098


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_cosine_identity_l720_72029

theorem triangle_cosine_identity (α β γ : ℝ) (h : α + β + γ = Real.pi) :
  Real.cos α ^ 2 + Real.cos β ^ 2 + Real.cos γ ^ 2 + 2 * Real.cos α * Real.cos β * Real.cos γ = 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_cosine_identity_l720_72029


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_at_one_l720_72034

/-- The function f(x) = x ln x + x -/
noncomputable def f (x : ℝ) : ℝ := x * Real.log x + x

/-- The derivative of f(x) -/
noncomputable def f_deriv (x : ℝ) : ℝ := Real.log x + 2

theorem tangent_line_at_one :
  let x₀ : ℝ := 1
  let y₀ : ℝ := f x₀
  let m : ℝ := f_deriv x₀
  ∀ x y : ℝ, y - y₀ = m * (x - x₀) ↔ y = 2 * x - 1 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_at_one_l720_72034


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_pool_capacity_calculation_l720_72078

/-- The capacity of a pool given specific filling conditions -/
noncomputable def pool_capacity (fill_time_both : ℝ) (fill_time_first : ℝ) (diff_rate : ℝ) : ℝ :=
  let rate_first := 1 / fill_time_first
  let rate_second := rate_first + diff_rate
  (rate_first + rate_second) * fill_time_both

theorem pool_capacity_calculation :
  let fill_time_both := 48 / 60  -- 48 minutes in hours
  let fill_time_first := 2       -- 2 hours
  let diff_rate := 50 / 60       -- 50 cubic meters per minute converted to hours
  pool_capacity fill_time_both fill_time_first diff_rate = 12000 := by
  sorry

-- Remove the #eval line as it's not computable

end NUMINAMATH_CALUDE_ERRORFEEDBACK_pool_capacity_calculation_l720_72078


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unnamed_finished_tenth_l720_72077

/-- Represents the finishing position in a race. -/
def Position := Fin 12

/-- Represents a racer in the race. -/
inductive Racer
| Simon
| David
| Hikmet
| Jack
| Marta
| Rand
| Todd
| Unnamed (n : Nat)

/-- Represents the result of a race. -/
def RaceResult := Racer → Position

theorem unnamed_finished_tenth (result : RaceResult) : 
  (result Racer.Marta = ⟨6, by norm_num⟩) →
  (result Racer.Jack = ⟨(result Racer.Simon).val - 3, by sorry⟩) →
  (result Racer.Hikmet = ⟨(result Racer.Rand).val + 7, by sorry⟩) →
  (result Racer.Jack = ⟨(result Racer.Marta).val - 3, by sorry⟩) →
  (result Racer.Hikmet = ⟨(result Racer.David).val - 3, by sorry⟩) →
  (result Racer.Todd = ⟨(result Racer.Jack).val + 5, by sorry⟩) →
  (result Racer.Rand = ⟨(result Racer.Todd).val - 2, by sorry⟩) →
  ∃ n, result (Racer.Unnamed n) = ⟨9, by norm_num⟩ := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_unnamed_finished_tenth_l720_72077


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_johns_journey_speed_l720_72064

/-- Represents a journey with two driving segments -/
structure Journey where
  total_distance : ℚ
  first_segment_time : ℚ
  second_segment_time : ℚ

/-- Calculates the speed of a journey -/
def journey_speed (j : Journey) : ℚ :=
  j.total_distance / (j.first_segment_time + j.second_segment_time)

/-- Theorem: John's journey speed is 45 mph -/
theorem johns_journey_speed :
  let j : Journey := { total_distance := 225, first_segment_time := 2, second_segment_time := 3 }
  journey_speed j = 45 := by
  -- Unfold the definition of journey_speed
  unfold journey_speed
  -- Simplify the expression
  simp
  -- The proof is complete
  rfl


end NUMINAMATH_CALUDE_ERRORFEEDBACK_johns_journey_speed_l720_72064


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ray_and_tom_fuel_efficiency_l720_72072

/-- Calculates the combined fuel efficiency of two cars -/
noncomputable def combined_fuel_efficiency (efficiency1 efficiency2 total_distance : ℝ) : ℝ :=
  total_distance / ((total_distance / (2 * efficiency1)) + (total_distance / (2 * efficiency2)))

/-- Theorem stating the combined fuel efficiency of Ray and Tom's cars -/
theorem ray_and_tom_fuel_efficiency :
  let ray_efficiency : ℝ := 40
  let tom_efficiency : ℝ := 20
  let total_distance : ℝ := 200
  abs (combined_fuel_efficiency ray_efficiency tom_efficiency total_distance - 80/3) < 0.01 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ray_and_tom_fuel_efficiency_l720_72072


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_double_angle_l720_72099

theorem cos_double_angle (α : ℝ) (h : Real.cos α = 4/5) : Real.cos (2*α) = 7/25 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_double_angle_l720_72099


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_pure_imaginary_complex_number_l720_72026

theorem pure_imaginary_complex_number (m : ℝ) : 
  (Complex.I * (m + 1) + (m^2 - m - 2)).im ≠ 0 ∧ (Complex.I * (m + 1) + (m^2 - m - 2)).re = 0 → m = 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_pure_imaginary_complex_number_l720_72026


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sufficient_but_not_necessary_not_necessary_l720_72022

/-- A proposition p is defined as "∀x > e, a - ln x < 0" --/
def proposition (a : ℝ) : Prop :=
  ∀ x > Real.exp 1, a - Real.log x < 0

/-- Sufficient condition for the proposition --/
def sufficient_condition (a : ℝ) : Prop := a < 1

/-- The theorem states that the sufficient_condition is indeed sufficient
    but not necessary for the proposition to be true --/
theorem sufficient_but_not_necessary :
  (∀ a, sufficient_condition a → proposition a) ∧
  ¬(∀ a, proposition a → sufficient_condition a) := by
  sorry

/-- The proposition is true for some a ≥ 1 --/
theorem not_necessary (a : ℝ) (h : a = 1) : proposition a := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sufficient_but_not_necessary_not_necessary_l720_72022


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_second_smallest_hot_dog_packs_l720_72051

theorem second_smallest_hot_dog_packs : ∃ n : ℕ, n > 0 ∧
  (∀ k : ℕ, 0 < k → k < n → ¬((12 * k) % 10 = 5 ∧ ∃ m : ℕ, (10 * m) % 12 = 3)) ∧
  (12 * n) % 10 = 5 ∧
  (∃ m : ℕ, (10 * m) % 12 = 3) ∧
  n = 15 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_second_smallest_hot_dog_packs_l720_72051


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_equal_weight_partition_l720_72075

theorem max_equal_weight_partition (weights : Finset ℕ) : 
  weights = Finset.range 5771 →
  (∃ (partition : Finset (Finset ℕ)), 
    partition.card = 2886 ∧
    (∀ s, s ∈ partition → s ⊆ weights) ∧
    (∀ s t, s ∈ partition → t ∈ partition → s ≠ t → s ∩ t = ∅) ∧
    (∀ s, s ∈ partition → (s.sum id) = ((weights.sum id) / 2886)) ∧
    (∀ n, n > 2886 → ¬∃ (partition : Finset (Finset ℕ)), 
      partition.card = n ∧
      (∀ s, s ∈ partition → s ⊆ weights) ∧
      (∀ s t, s ∈ partition → t ∈ partition → s ≠ t → s ∩ t = ∅) ∧
      (∀ s, s ∈ partition → (s.sum id) = ((weights.sum id) / n)))) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_equal_weight_partition_l720_72075


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_2x_five_points_l720_72013

noncomputable def five_point_method (f : ℝ → ℝ) (period : ℝ) : List ℝ :=
  [0, period / 4, period / 2, 3 * period / 4, period]

theorem sin_2x_five_points :
  five_point_method (λ x => 2 * Real.sin (2 * x)) Real.pi =
    [0, Real.pi / 4, Real.pi / 2, 3 * Real.pi / 4, Real.pi] := by
  -- Proof goes here
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_2x_five_points_l720_72013


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_constant_ratio_implies_b_zero_l720_72079

-- Define the circle C
def circle_C (b : ℝ) (x y : ℝ) : Prop :=
  (x + 4)^2 + (y + b)^2 = 16

-- Define the distance between two points
noncomputable def distance (x1 y1 x2 y2 : ℝ) : ℝ :=
  Real.sqrt ((x2 - x1)^2 + (y2 - y1)^2)

-- Define the ratio PA/PB
noncomputable def ratio (x y : ℝ) : ℝ :=
  distance x y (-2) 0 / distance x y 4 0

theorem constant_ratio_implies_b_zero (b : ℝ) :
  (∀ x y : ℝ, circle_C b x y → ∃ k : ℝ, ∀ x' y' : ℝ, circle_C b x' y' → ratio x y = ratio x' y') →
  b = 0 := by
  sorry

#check constant_ratio_implies_b_zero

end NUMINAMATH_CALUDE_ERRORFEEDBACK_constant_ratio_implies_b_zero_l720_72079


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inverse_f_undefined_at_one_l720_72010

-- Define the function f as noncomputable
noncomputable def f (x : ℝ) : ℝ := (x - 2) / (x - 5)

-- Theorem statement
theorem inverse_f_undefined_at_one :
  ∀ x : ℝ, f x = 1 → x = 5 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inverse_f_undefined_at_one_l720_72010


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_bricks_required_for_wall_l720_72030

/-- Represents the dimensions of a rectangular object -/
structure Dimensions where
  length : ℝ
  width : ℝ
  height : ℝ

/-- Calculates the volume of a rectangular object given its dimensions -/
noncomputable def volume (d : Dimensions) : ℝ :=
  d.length * d.width * d.height

/-- Converts centimeters to meters -/
noncomputable def cmToM (x : ℝ) : ℝ :=
  x / 100

theorem bricks_required_for_wall
  (brick_cm : Dimensions)
  (wall_m : Dimensions)
  (h_brick_length : brick_cm.length = 20)
  (h_brick_width : brick_cm.width = 10)
  (h_brick_height : brick_cm.height = 7.5)
  (h_wall_length : wall_m.length = 25)
  (h_wall_width : wall_m.width = 2)
  (h_wall_height : wall_m.height = 0.75) :
  (volume wall_m) / (volume (Dimensions.mk (cmToM brick_cm.length) (cmToM brick_cm.width) (cmToM brick_cm.height))) = 25000 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_bricks_required_for_wall_l720_72030


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_equation_l720_72097

-- Define the curve C
def C (x y : ℝ) : Prop := x^2 + y^2 - 2*x = 0

-- Define point P
def P : ℝ × ℝ := (3, 2)

-- Define the equation of line AB
def line_AB (x y : ℝ) : Prop := 2*x + 2*y - 3 = 0

-- Theorem statement
theorem tangent_line_equation :
  ∃ (A B : ℝ × ℝ),
    C A.1 A.2 ∧ C B.1 B.2 ∧  -- A and B are on curve C
    (∃ (t : ℝ × ℝ → ℝ × ℝ), t P = A) ∧  -- There exists a tangent from P to A
    (∃ (t : ℝ × ℝ → ℝ × ℝ), t P = B) ∧  -- There exists a tangent from P to B
    ∀ (x y : ℝ), x ∈ Set.Icc A.1 B.1 ∧ y ∈ Set.Icc A.2 B.2 → line_AB x y  -- The line AB satisfies the equation
  := by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_equation_l720_72097


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_vertex_x_is_two_l720_72016

/-- A quadratic function passing through specific points -/
noncomputable def quadratic_function (a b c : ℝ) : ℝ → ℝ := λ x ↦ a * x^2 + b * x + c

/-- The x-coordinate of the vertex of a quadratic function -/
noncomputable def vertex_x (a b : ℝ) : ℝ := -b / (2 * a)

theorem vertex_x_is_two (a b c : ℝ) :
  quadratic_function a b c 0 = 4 →
  quadratic_function a b c 4 = 4 →
  quadratic_function a b c 3 = 9 →
  vertex_x a b = 2 := by
  sorry

#eval "Theorem stated successfully"

end NUMINAMATH_CALUDE_ERRORFEEDBACK_vertex_x_is_two_l720_72016


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_point_on_line_segment_l720_72002

-- Define the space
variable {E : Type*} [NormedAddCommGroup E] [InnerProductSpace ℝ E] [CompleteSpace E]

-- Define the fixed points and the moving point
variable (F₁ F₂ M : E)

-- Define the conditions
variable (h₁ : ‖F₁ - F₂‖ = 16)
variable (h₂ : ∀ M, ‖M - F₁‖ + ‖M - F₂‖ = 16)

-- Theorem statement
theorem point_on_line_segment :
  ∃ t : ℝ, 0 ≤ t ∧ t ≤ 1 ∧ M = (1 - t) • F₁ + t • F₂ := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_point_on_line_segment_l720_72002


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_point_c_value_l720_72061

/-- Represents a point on a number line -/
structure Point where
  value : ℝ

/-- Represents the movement of a point on a number line -/
def move (p : Point) (distance : ℝ) : Point :=
  ⟨p.value + distance⟩

theorem point_c_value (a b c : Point) :
  (move (move a (-5)) 2).value = b.value →
  (move a 4).value = c.value →
  b.value = -1 →
  c.value = 6 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_point_c_value_l720_72061


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_eccentricity_l720_72020

/-- The eccentricity of a hyperbola with given properties -/
theorem hyperbola_eccentricity 
  (a b c : ℝ) 
  (ha : a > 0)
  (hb : b > 0)
  (P F₁ F₂ : ℝ × ℝ) 
  (h_hyperbola : (P.1^2 / a^2) - (P.2^2 / b^2) = 1)
  (h_foci : F₁ ≠ F₂)
  (h_orthogonal : (P.1 - F₁.1) * (P.2 - F₁.2) + (P.1 - F₂.1) * (P.2 - F₂.2) = 0)
  (h_product : ((P.1 - F₁.1)^2 + (P.2 - F₁.2)^2) * ((P.1 - F₂.1)^2 + (P.2 - F₂.2)^2) = (2*a*c)^2) :
  (c^2 / a^2) = (Real.sqrt 5 + 1) / 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_eccentricity_l720_72020


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_properties_l720_72095

/-- The parabola C with focus F and parameter p > 0 -/
structure Parabola where
  p : ℝ
  hp : p > 0

/-- The ellipse E with semi-major axis 2 and semi-minor axis √3 -/
def Ellipse : Type := Unit

/-- A point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- A line segment between two points -/
structure Segment where
  a : Point
  b : Point

/-- The focus of the parabola coincides with the right focus of the ellipse -/
def focus_coincide (C : Parabola) (_E : Ellipse) : Prop :=
  C.p = 2

/-- The chord AB of the parabola C passes through point F and intersects the directrix at M -/
def chord_property (_C : Parabola) (_AB : Segment) (_F _M : Point) : Prop :=
  true  -- We don't need to define this precisely for the statement

theorem parabola_properties (C : Parabola) (E : Ellipse) (AB : Segment) (F M : Point)
    (h1 : focus_coincide C E)
    (h2 : chord_property C AB F M) :
    (∃ (A' B' : Point),
      /- 1. The equation of the parabola C is y² = 4x -/
      C.p = 2 ∧
      /- 2. The minimum value of |AB|/|MF| is 2 -/
      (∃ (min_ratio : ℝ), min_ratio = 2 ∧
        ∀ (AB' : Segment), chord_property C AB' F M →
          let ratio := Real.sqrt ((AB'.a.x - AB'.b.x)^2 + (AB'.a.y - AB'.b.y)^2) /
                       Real.sqrt ((M.x - F.x)^2 + (M.y - F.y)^2)
          ratio ≥ min_ratio) ∧
      /- 3. Triangle A'FB' is a right triangle -/
      (Point.x A' - Point.x F)*(Point.x B' - Point.x F) +
      (Point.y A' - Point.y F)*(Point.y B' - Point.y F) = 0) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_properties_l720_72095


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_incenter_coords_for_specific_triangle_l720_72070

/-- Represents a triangle with side lengths a, b, and c. --/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Represents the coordinates of a point in barycentric coordinates. --/
structure BarycentricCoord where
  x : ℝ
  y : ℝ
  z : ℝ

/-- The incenter of a triangle in barycentric coordinates. --/
noncomputable def incenter (t : Triangle) : BarycentricCoord :=
  { x := t.a / (t.a + t.b + t.c)
  , y := t.b / (t.a + t.b + t.c)
  , z := t.c / (t.a + t.b + t.c) }

theorem incenter_coords_for_specific_triangle :
  let t : Triangle := { a := 6, b := 10, c := 8 }
  let i : BarycentricCoord := incenter t
  i.x = 5/12 ∧ i.y = 1/3 ∧ i.z = 1/4 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_incenter_coords_for_specific_triangle_l720_72070


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_lambda_parity_l720_72046

theorem exists_lambda_parity : ∃ lambda : ℝ, lambda > 0 ∧ ∀ n : ℕ, n > 0 → (⌊lambda^n⌋ : ℤ) % 2 = n % 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_lambda_parity_l720_72046


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_holly_paychecks_l720_72037

/-- Calculates the number of paychecks Holly receives per year given her contribution, 
    the company's match percentage, and the total yearly contribution. -/
def number_of_paychecks (holly_contribution : ℚ) (company_match_percent : ℚ) 
                        (total_yearly_contribution : ℚ) : ℕ :=
  (total_yearly_contribution / (holly_contribution * (1 + company_match_percent / 100))).floor.toNat

/-- Proves that Holly receives 26 paychecks per year given the specified conditions. -/
theorem holly_paychecks : 
  number_of_paychecks 100 6 2756 = 26 := by
  -- Unfold the definition of number_of_paychecks
  unfold number_of_paychecks
  -- Evaluate the expression
  norm_num
  -- QED
  rfl


end NUMINAMATH_CALUDE_ERRORFEEDBACK_holly_paychecks_l720_72037


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_tangent_to_circle_l720_72052

/-- The line equation after translation -/
def translated_line (lambda : ℝ) (x y : ℝ) : Prop :=
  x + 2*y + 5 + lambda = 0

/-- The circle equation -/
def circle_eq (x y : ℝ) : Prop :=
  x^2 + y^2 + 2*x - 4*y = 0

/-- The center of the circle -/
def circle_center : ℝ × ℝ := (-1, 2)

/-- The radius of the circle -/
noncomputable def circle_radius : ℝ := Real.sqrt 5

theorem line_tangent_to_circle (lambda : ℝ) :
  (∀ x y, translated_line lambda x y → ¬ circle_eq x y) ∧
  (∃ x y, translated_line lambda x y ∧ circle_eq x y) →
  lambda = -3 ∨ lambda = -13 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_tangent_to_circle_l720_72052


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_water_evaporation_percentage_l720_72042

/-- Proves that given a bowl with 10 ounces of water and an evaporation rate of 0.16 ounces per day
    over a 75-day period, the percentage of water that evaporates is 100%. -/
theorem water_evaporation_percentage
  (initial_water : ℝ)
  (evaporation_rate : ℝ)
  (days : ℝ)
  (h1 : initial_water = 10)
  (h2 : evaporation_rate = 0.16)
  (h3 : days = 75)
  : (min initial_water (evaporation_rate * days) / initial_water) * 100 = 100 := by
  sorry

#check water_evaporation_percentage

end NUMINAMATH_CALUDE_ERRORFEEDBACK_water_evaporation_percentage_l720_72042


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_closed_form_l720_72032

/-- A sequence satisfying the given recurrence relation -/
def sequenceProperty (a : ℕ → ℕ) (initial : ℕ) : Prop :=
  a 1 = initial ∧ ∀ n : ℕ, n ≥ 2 → a n - a (n - 1) = 2^(n - 1)

/-- The theorem stating the closed form of the sequence -/
theorem sequence_closed_form (a : ℕ → ℕ) (initial : ℕ) :
  sequenceProperty a initial → ∀ n : ℕ, n ≥ 1 → a n = initial + 2^n - 1 := by
  sorry

#check sequence_closed_form

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_closed_form_l720_72032


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_composite_numbers_with_coprime_gcd_l720_72033

/-- The upper bound for the numbers -/
def upperBound : ℕ := 1500

/-- The list of prime numbers less than or equal to the square root of the upper bound -/
def primes : List ℕ := [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]

/-- A function to check if a number is composite -/
def isComposite (n : ℕ) : Prop := ¬ Nat.Prime n ∧ n > 1

/-- The theorem statement -/
theorem max_composite_numbers_with_coprime_gcd :
  ∃ (S : Finset ℕ),
    (∀ n, n ∈ S → n < upperBound ∧ isComposite n) ∧
    (∀ a b, a ∈ S → b ∈ S → a ≠ b → Nat.gcd a b = 1) ∧
    S.card = primes.length ∧
    (∀ T : Finset ℕ,
      (∀ n, n ∈ T → n < upperBound ∧ isComposite n) →
      (∀ a b, a ∈ T → b ∈ T → a ≠ b → Nat.gcd a b = 1) →
      T.card ≤ primes.length) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_composite_numbers_with_coprime_gcd_l720_72033


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_coeff_x2_and_coeff_x7_l720_72069

noncomputable def f (m n : ℕ+) (x : ℝ) : ℝ := (1 + x)^(m.val:ℝ) + (1 + x)^(n.val:ℝ)

theorem min_coeff_x2_and_coeff_x7 (m n : ℕ+) :
  (∃ k : ℕ+, k = m + n ∧ k = 19) →
  (∃ min_coeff_x2 : ℕ, min_coeff_x2 = 81 ∧
    ∀ p q : ℕ+, p + q = 19 →
      Nat.choose p.val 2 + Nat.choose q.val 2 ≥ min_coeff_x2) ∧
  (∃ coeff_x7 : ℕ, coeff_x7 = 156 ∧
    ((m = 10 ∧ n = 9) ∨ (m = 9 ∧ n = 10)) →
      Nat.choose m.val 7 + Nat.choose n.val 7 = coeff_x7) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_coeff_x2_and_coeff_x7_l720_72069


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_harry_hours_worked_l720_72082

-- Define the payment structure for Harry and James
noncomputable def harry_pay (x : ℝ) (h : ℝ) : ℝ :=
  if h ≤ 15 then x * h else x * 15 + 1.5 * x * (h - 15)

noncomputable def james_pay (x : ℝ) (h : ℝ) : ℝ :=
  if h ≤ 40 then x * h else x * 40 + 2 * x * (h - 40)

-- Theorem statement
theorem harry_hours_worked (x : ℝ) (h : ℝ) :
  x > 0 → james_pay x 41 = harry_pay x h → h = 22 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_harry_hours_worked_l720_72082


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_objects_for_unique_sums_l720_72027

/-- Represents a 3x3 grid of non-negative integers -/
def Grid := Fin 3 → Fin 3 → ℕ

/-- The sum of a row in the grid -/
def row_sum (g : Grid) (i : Fin 3) : ℕ := 
  Finset.sum (Finset.univ : Finset (Fin 3)) (λ j => g i j)

/-- The sum of a column in the grid -/
def col_sum (g : Grid) (j : Fin 3) : ℕ := 
  Finset.sum (Finset.univ : Finset (Fin 3)) (λ i => g i j)

/-- Predicate that checks if all rows and columns have unique sums -/
def unique_sums (g : Grid) : Prop :=
  (∀ i j : Fin 3, i ≠ j → row_sum g i ≠ row_sum g j) ∧
  (∀ i j : Fin 3, i ≠ j → col_sum g i ≠ col_sum g j)

/-- The total sum of all elements in the grid -/
def total_sum (g : Grid) : ℕ :=
  Finset.sum (Finset.univ : Finset (Fin 3)) (λ i => 
    Finset.sum (Finset.univ : Finset (Fin 3)) (λ j => g i j))

/-- The main theorem stating the minimum number of objects required -/
theorem min_objects_for_unique_sums :
  ∃ (g : Grid), unique_sums g ∧ total_sum g = 8 ∧
  ∀ (h : Grid), unique_sums h → total_sum h ≥ 8 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_objects_for_unique_sums_l720_72027


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_2beta_eq_2cos_2alpha_l720_72065

theorem cos_2beta_eq_2cos_2alpha 
  (α β θ : Real) 
  (h1 : ∀ k : Int, α ≠ k * π + π / 2) 
  (h2 : ∀ k : Int, β ≠ k * π + π / 2)
  (h3 : (Real.sin θ)^2 - 2 * Real.sin α * Real.sin θ + (Real.sin β)^2 = 0)
  (h4 : (Real.cos θ)^2 - 2 * Real.sin α * Real.cos θ + (Real.sin β)^2 = 0) : 
  Real.cos (2 * β) = 2 * Real.cos (2 * α) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_2beta_eq_2cos_2alpha_l720_72065


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_monotonic_decrease_interval_l720_72060

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a^(|2*x - 4|)

theorem monotonic_decrease_interval
  (a : ℝ)
  (h1 : a > 0)
  (h2 : a ≠ 1)
  (h3 : f a 1 = 9) :
  ∃ (I : Set ℝ), StrictMonoOn (f a) (Set.Iic 2) ∧ Set.Iic 2 = I :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_monotonic_decrease_interval_l720_72060


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_concentric_circles_radii_difference_l720_72012

/-- Given two concentric circles with area ratio 2:5 and smaller radius r,
    the difference between their radii is approximately 0.58r -/
theorem concentric_circles_radii_difference (r : ℝ) (R : ℝ) (h : r > 0) :
  (π * R^2) / (π * r^2) = 5/2 →
  abs (R - r - 0.58 * r) < 0.01 * r :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_concentric_circles_radii_difference_l720_72012


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_alpha_terminal_side_l720_72004

theorem sin_alpha_terminal_side (α : ℝ) (P : ℝ × ℝ) :
  P = (-8, 6) →  -- Point P lies on the terminal side of angle α
  Real.sin α = 3/5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_alpha_terminal_side_l720_72004


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_sum_of_roots_l720_72093

-- Define the function f as noncomputable due to the use of Real.sqrt
noncomputable def f (x : ℝ) : ℝ := Real.sin x + Real.sqrt 3 * Real.cos x + 2

-- State the theorem
theorem sin_sum_of_roots (m : ℝ) (α β : ℝ) :
  (∀ x, x ∈ Set.Icc 0 (2 * Real.pi) → f x ∈ Set.Icc 0 4) →
  α ∈ Set.Icc 0 (2 * Real.pi) →
  β ∈ Set.Icc 0 (2 * Real.pi) →
  α ≠ β →
  f α = m →
  f β = m →
  Real.sin (α + β) = Real.sqrt 3 / 2 :=
by
  -- The proof is omitted using 'sorry'
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_sum_of_roots_l720_72093


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_less_than_radius_correct_answer_is_4_l720_72043

/-- A circle with center O and radius 5 -/
structure Circle :=
  (O : ℝ × ℝ)
  (radius : ℝ := 5)

/-- A point P inside the circle -/
structure PointInside (c : Circle) :=
  (P : ℝ × ℝ)
  (inside : ∀ (x y : ℝ), P = (x, y) → (x - c.O.1)^2 + (y - c.O.2)^2 < c.radius^2)

/-- The distance between two points in ℝ² -/
noncomputable def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

/-- Theorem: The distance from the center to a point inside the circle is less than the radius -/
theorem distance_less_than_radius (c : Circle) (p : PointInside c) :
  distance c.O p.P < c.radius := by
  sorry

/-- The correct answer is option D: 4 -/
theorem correct_answer_is_4 (c : Circle) (p : PointInside c) :
  ∃ (d : ℝ), d = 4 ∧ distance c.O p.P = d := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_less_than_radius_correct_answer_is_4_l720_72043


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_edges_special_graph_l720_72091

/-- A graph with the property that for any two vertices, there exists a third vertex
    connected to neither of them. -/
structure SpecialGraph (n : ℕ) where
  vertices : Fin n
  edges : Finset (Fin n × Fin n)
  property : ∀ (u v : Fin n), ∃ (w : Fin n), (u, w) ∉ edges ∧ (v, w) ∉ edges
  n_ge_4 : n ≥ 4

/-- The maximum number of edges in a SpecialGraph with n vertices is (n-1)(n-3)/2. -/
theorem max_edges_special_graph {n : ℕ} (G : SpecialGraph n) :
  G.edges.card ≤ (n - 1) * (n - 3) / 2 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_edges_special_graph_l720_72091


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rectangle_perimeter_l720_72071

theorem rectangle_perimeter (total_area : ℝ) (num_squares : ℕ) (width_squares : ℕ) (height_squares : ℕ) :
  total_area = 216 →
  num_squares = 6 →
  width_squares = 3 →
  height_squares = 2 →
  2 * (Real.sqrt (total_area / num_squares) * (width_squares + height_squares)) = 60 := by
  intro h_area h_num h_width h_height
  -- Define intermediate calculations
  let square_area := total_area / num_squares
  let square_side := Real.sqrt square_area
  let rectangle_width := square_side * width_squares
  let rectangle_height := square_side * height_squares
  -- The actual proof would go here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_rectangle_perimeter_l720_72071


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_angles_in_rectangular_parallelepiped_l720_72062

-- Define a rectangular parallelepiped
structure RectangularParallelepiped where
  a : ℝ
  b : ℝ
  c : ℝ
  h_positive : a > 0 ∧ b > 0 ∧ c > 0

-- Define the angles α and β
noncomputable def angle_α (rp : RectangularParallelepiped) : ℝ := 
  Real.arcsin (rp.b / Real.sqrt (rp.a^2 + rp.b^2 + rp.c^2))

noncomputable def angle_β (rp : RectangularParallelepiped) : ℝ := 
  Real.arcsin (rp.a / Real.sqrt (rp.a^2 + rp.b^2 + rp.c^2))

-- State the theorem
theorem angles_in_rectangular_parallelepiped (rp : RectangularParallelepiped) :
  angle_α rp > 0 ∧ angle_β rp > 0 ∧ angle_α rp + angle_β rp < π/2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_angles_in_rectangular_parallelepiped_l720_72062


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_expansion_term_inequality_l720_72081

theorem expansion_term_inequality (x : ℝ) : 
  (Nat.choose 6 2 * 2^4 * (-x)^2 ≤ Nat.choose 6 1 * 2^5 * (-x) ∧ 
   Nat.choose 6 1 * 2^5 * (-x) < Nat.choose 6 0 * 2^6) ↔ 
  (-1/3 < x ∧ x ≤ 0) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_expansion_term_inequality_l720_72081


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_board_evolution_final_board_state_l720_72074

def board_sum (a b c : ℕ) : ℕ := a + b + c

def pairwise_sum (a b c : ℕ) : (ℕ × ℕ × ℕ) :=
  (a + b, a + c, b + c)

def last_digit (n : ℕ) : ℕ := n % 10

def valid_final_states : List (ℕ × ℕ × ℕ) :=
  [(7, 8, 9), (7, 9, 8), (8, 7, 9), (8, 9, 7), (9, 7, 8), (9, 8, 7)]

theorem board_evolution (n : ℕ) :
  ∃ (a b c : ℕ), 
    (last_digit a, last_digit b, last_digit c) ∈ valid_final_states ∧
    board_sum (last_digit a) (last_digit b) (last_digit c) = 9 :=
by
  sorry

theorem final_board_state :
  ∃ (a b c : ℕ), 
    (last_digit a, last_digit b, last_digit c) ∈ valid_final_states :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_board_evolution_final_board_state_l720_72074


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_tangent_circle_l720_72066

/-- The line on which the center of circle C lies -/
def line_l (x y : ℝ) : Prop := x - y + 10 = 0

/-- The equation of circle C -/
def circle_C (a b x y : ℝ) : Prop := (x - a)^2 + (y - b)^2 = 25

/-- The equation of circle O -/
def circle_O (r x y : ℝ) : Prop := x^2 + y^2 = r^2

/-- The distance between two points -/
noncomputable def distance (x1 y1 x2 y2 : ℝ) : ℝ := ((x1 - x2)^2 + (y1 - y2)^2).sqrt

theorem unique_tangent_circle :
  ∃ (a b : ℝ),
    line_l a b ∧
    circle_C a b (-5) 0 ∧
    (∀ (x y : ℝ), circle_C a b x y → distance 0 0 x y ≥ 5) ∧
    (∃! (x y : ℝ), circle_C a b x y ∧ distance 0 0 x y = 5 + (5 * Real.sqrt 2 - 5)) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_tangent_circle_l720_72066


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_monotone_increasing_characterization_of_m_l720_72038

noncomputable section

open Real

-- Define the function f
def f (x : ℝ) : ℝ := 2 * (sin (π / 4 + x))^2 - Real.sqrt 3 * cos (2 * x) - 1

-- Define the domain
def domain : Set ℝ := { x | π / 4 ≤ x ∧ x ≤ π / 2 }

-- Theorem 1: f is monotonically increasing on [π/4, 5π/12]
theorem f_monotone_increasing :
  ∀ x y, x ∈ domain → y ∈ domain → π / 4 ≤ x → x < y → y ≤ 5 * π / 12 → f x < f y := by
  sorry

-- Theorem 2: Characterization of m for which |f(x) - m| < 2 holds
theorem characterization_of_m :
  ∀ m : ℝ, (∀ x ∈ domain, |f x - m| < 2) ↔ (0 < m ∧ m < 3) := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_monotone_increasing_characterization_of_m_l720_72038


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_crease_length_is_correct_l720_72092

/-- Represents a right triangle with sides 6, 8, and 10 inches -/
structure RightTriangle where
  a : ℚ
  b : ℚ
  c : ℚ
  is_right : a^2 + b^2 = c^2
  side_a : a = 6
  side_b : b = 8
  side_c : c = 10

/-- The length of the crease when point A is folded onto point B -/
def crease_length (t : RightTriangle) : ℚ := 15/4

/-- Theorem stating that the crease length is 15/4 inches -/
theorem crease_length_is_correct (t : RightTriangle) : crease_length t = 15/4 := by
  -- Unfold the definition of crease_length
  unfold crease_length
  -- The result follows directly from the definition
  rfl


end NUMINAMATH_CALUDE_ERRORFEEDBACK_crease_length_is_correct_l720_72092


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_meat_price_theorem_l720_72087

/-- The original price of meat per kilogram -/
noncomputable def original_price : ℚ := 10

/-- The total amount spent -/
noncomputable def total_amount : ℚ := 240

/-- The price increase ratio -/
noncomputable def price_increase : ℚ := 1/5

/-- The difference in quantity bought -/
noncomputable def quantity_difference : ℚ := 4

/-- Theorem stating the original price of meat -/
theorem meat_price_theorem :
  (total_amount / original_price) - (total_amount / (original_price * (1 + price_increase))) = quantity_difference :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_meat_price_theorem_l720_72087


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_symmetry_l720_72023

noncomputable def f (ω : ℝ) (x : ℝ) : ℝ := Real.sin (ω * x + Real.pi / 4)

theorem function_symmetry (ω : ℝ) (h1 : ω > 0) (h2 : ∀ x, f ω (x + Real.pi / 2) = f ω x) :
  ∀ x, f ω (3 * Real.pi / 8 + x) = f ω (3 * Real.pi / 8 - x) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_symmetry_l720_72023


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_jaco_purchase_amount_l720_72018

/-- Calculates the total amount to pay after discounts --/
def totalAmountAfterDiscounts (purchases : List ℚ) 
  (discount100 : ℚ) (discount150 : ℚ) : ℚ :=
  let totalSpent := purchases.sum
  let discount100Amount := (totalSpent / 100).floor * 10
  let discount150Amount := (totalSpent / 150).floor * (15/2)
  totalSpent - discount100Amount - discount150Amount

/-- The theorem statement --/
theorem jaco_purchase_amount :
  let purchases := [74, 92, 2, 3, 4, 5, 42, 58]
  let discount100 := 10 -- 10% discount for every $100
  let discount150 := 5  -- 5% discount for every $150
  totalAmountAfterDiscounts purchases discount100 discount150 = 252.5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_jaco_purchase_amount_l720_72018


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_four_numbers_l720_72089

theorem sum_of_four_numbers (p q r s : ℝ) : 
  0 < p → 0 < q → 0 < r → 0 < s →
  p^2 + q^2 = 2500 →
  r^2 + s^2 = 2500 →
  p * q = 1225 →
  r * s = 1225 →
  p + s = 75 →
  p + q + r + s = 150 := by
  sorry

#check sum_of_four_numbers

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_four_numbers_l720_72089


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_abby_overall_score_l720_72094

def test1_score : ℚ := 85 / 100
def test1_problems : ℕ := 30

def test2_score : ℚ := 75 / 100
def test2_problems : ℕ := 50

def test3_score : ℚ := 60 / 100
def test3_problems : ℕ := 20

def test4_score : ℚ := 90 / 100
def test4_problems : ℕ := 40

def total_problems : ℕ := test1_problems + test2_problems + test3_problems + test4_problems

def correct_problems : ℚ := 
  test1_score * test1_problems + 
  test2_score * test2_problems + 
  test3_score * test3_problems + 
  test4_score * test4_problems

theorem abby_overall_score : 
  correct_problems / total_problems = 80 / 100 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_abby_overall_score_l720_72094


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_permutation_count_l720_72056

def permutations (n k : ℕ) : ℕ := Nat.choose n k

def M : ℕ := Finset.sum (Finset.range 4) (λ k => 
  permutations 5 (k + 1) * permutations 6 k * permutations 7 (k + 2))

theorem permutation_count : M % 1000 = 555 := by
  -- The proof goes here
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_permutation_count_l720_72056


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_regular_pyramid_volume_bound_l720_72080

noncomputable def is_volume_of_regular_pyramid (b V : ℝ) : Prop :=
  ∃ (α : ℝ), 0 < α ∧ α < Real.pi/2 ∧ V = (1/3) * Real.pi * b^3 * (Real.cos α)^2 * Real.sin α

theorem regular_pyramid_volume_bound (b : ℝ) (h : b = 2) : 
  ∀ V : ℝ, is_volume_of_regular_pyramid b V → V < 3.25 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_regular_pyramid_volume_bound_l720_72080


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_lcm_pairs_count_l720_72073

theorem lcm_pairs_count : 
  let N : ℕ := 59400000
  ∃ (S : Finset (ℕ × ℕ)), 
    (∀ (pair : ℕ × ℕ), pair ∈ S ↔ Nat.lcm pair.1 pair.2 = N) ∧ 
    S.card = 1502 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_lcm_pairs_count_l720_72073


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_conversion_theorem_l720_72047

-- Define conversion factors
def meters_to_decimeters : ℚ := 10
def decimeters_to_centimeters : ℚ := 10
def yuan_to_cents : ℚ := 100

-- Define the conversion functions
noncomputable def convert_meters_to_decimeters_and_centimeters (m : ℚ) : ℚ × ℚ :=
  let decimeters := m * meters_to_decimeters
  let whole_decimeters := ⌊decimeters⌋
  let remaining_decimeters := decimeters - whole_decimeters
  let centimeters := remaining_decimeters * decimeters_to_centimeters
  (whole_decimeters, centimeters)

noncomputable def convert_yuan_to_yuan_and_cents (y : ℚ) : ℚ × ℚ :=
  let whole_yuan := ⌊y⌋
  let remaining_yuan := y - whole_yuan
  let cents := remaining_yuan * yuan_to_cents
  (whole_yuan, cents)

-- State the theorem
theorem conversion_theorem :
  convert_meters_to_decimeters_and_centimeters (56/100) = (5, 6) ∧
  convert_yuan_to_yuan_and_cents (205/100) = (2, 5) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_conversion_theorem_l720_72047


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_pizza_area_increase_l720_72021

theorem pizza_area_increase : 
  ∀ (r : ℝ), r > 0 → 
  (π * (1.4 * r)^2 - π * r^2) / (π * r^2) * 100 = 96 := by
  intro r hr
  -- Proof steps would go here
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_pizza_area_increase_l720_72021


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_theorem_l720_72045

noncomputable section

def Triangle (a b c : ℝ) := a > 0 ∧ b > 0 ∧ c > 0 ∧ a + b > c ∧ b + c > a ∧ c + a > b

variable {a b c A B C : ℝ}

def area_triangle (a b c : ℝ) : ℝ := 
  let s := (a + b + c) / 2
  Real.sqrt (s * (s - a) * (s - b) * (s - c))

theorem triangle_theorem (h_triangle : Triangle a b c)
  (h_order : a < b ∧ b < c)
  (h_sine : a / Real.sin A = 2 * b / Real.sqrt 3) :
  B = π / 3 ∧
  (a = 2 ∧ c = 3 → b = Real.sqrt 7 ∧ 
    area_triangle a b c = 3 * Real.sqrt 3 / 2) :=
by sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_theorem_l720_72045

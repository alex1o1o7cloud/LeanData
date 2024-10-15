import Mathlib

namespace NUMINAMATH_CALUDE_cubic_root_product_l3291_329187

theorem cubic_root_product (p q r : ℝ) : 
  (3 * p^3 - 9 * p^2 + 5 * p - 15 = 0) ∧ 
  (3 * q^3 - 9 * q^2 + 5 * q - 15 = 0) ∧ 
  (3 * r^3 - 9 * r^2 + 5 * r - 15 = 0) → 
  p * q * r = 5 := by
sorry

end NUMINAMATH_CALUDE_cubic_root_product_l3291_329187


namespace NUMINAMATH_CALUDE_tangent_product_theorem_l3291_329145

theorem tangent_product_theorem : 
  Real.tan (20 * π / 180) * Real.tan (40 * π / 180) * Real.tan (60 * π / 180) * Real.tan (80 * π / 180) = 3 := by
  sorry

end NUMINAMATH_CALUDE_tangent_product_theorem_l3291_329145


namespace NUMINAMATH_CALUDE_unique_solution_cube_equation_l3291_329151

theorem unique_solution_cube_equation (x y : ℕ) :
  y^6 + 2*y^3 - y^2 + 1 = x^3 → x = 1 ∧ y = 0 := by
  sorry

end NUMINAMATH_CALUDE_unique_solution_cube_equation_l3291_329151


namespace NUMINAMATH_CALUDE_spade_calculation_l3291_329147

/-- Define the ⊙ operation for real numbers -/
def spade (a b : ℝ) : ℝ := (a + b) * (a - b)

/-- Theorem stating that 7 ⊙ (2 ⊙ 3) = 24 -/
theorem spade_calculation : spade 7 (spade 2 3) = 24 := by
  sorry

end NUMINAMATH_CALUDE_spade_calculation_l3291_329147


namespace NUMINAMATH_CALUDE_complex_square_l3291_329114

theorem complex_square : (1 + Complex.I) ^ 2 = 2 * Complex.I := by
  sorry

end NUMINAMATH_CALUDE_complex_square_l3291_329114


namespace NUMINAMATH_CALUDE_ratio_range_l3291_329156

-- Define the condition for the point (x,y)
def satisfies_condition (x y : ℝ) : Prop :=
  x^2 - 8*x + y^2 - 4*y + 16 ≤ 0

-- Define the range for y/x
def in_range (r : ℝ) : Prop :=
  0 ≤ r ∧ r ≤ 4/3

-- Theorem statement
theorem ratio_range (x y : ℝ) (h : satisfies_condition x y) (hx : x ≠ 0) :
  in_range (y / x) :=
sorry

end NUMINAMATH_CALUDE_ratio_range_l3291_329156


namespace NUMINAMATH_CALUDE_balloon_difference_balloon_difference_proof_l3291_329136

/-- Proves that the difference between the combined total of Amy, Felix, and Olivia's balloons
    and James' balloons is 373. -/
theorem balloon_difference : ℕ → ℕ → ℕ → ℕ → Prop :=
  fun james_balloons amy_balloons felix_balloons olivia_balloons =>
    james_balloons = 1222 ∧
    amy_balloons = 513 ∧
    felix_balloons = 687 ∧
    olivia_balloons = 395 →
    (amy_balloons + felix_balloons + olivia_balloons) - james_balloons = 373

-- The proof is omitted
theorem balloon_difference_proof :
  balloon_difference 1222 513 687 395 := by sorry

end NUMINAMATH_CALUDE_balloon_difference_balloon_difference_proof_l3291_329136


namespace NUMINAMATH_CALUDE_fraction_equality_l3291_329192

theorem fraction_equality : (1 : ℝ) / (2 - Real.sqrt 3) = 2 + Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_fraction_equality_l3291_329192


namespace NUMINAMATH_CALUDE_border_tile_difference_l3291_329138

/-- Represents an octagonal figure made of tiles -/
structure OctagonalFigure where
  white_tiles : ℕ
  black_tiles : ℕ

/-- Creates a new figure by adding a border of black tiles -/
def add_border (figure : OctagonalFigure) : OctagonalFigure :=
  { white_tiles := figure.white_tiles,
    black_tiles := figure.black_tiles + 8 }

/-- The difference between black and white tiles in a figure -/
def tile_difference (figure : OctagonalFigure) : ℤ :=
  figure.black_tiles - figure.white_tiles

theorem border_tile_difference (original : OctagonalFigure) 
  (h1 : original.white_tiles = 16)
  (h2 : original.black_tiles = 9) :
  tile_difference (add_border original) = 1 := by
  sorry

end NUMINAMATH_CALUDE_border_tile_difference_l3291_329138


namespace NUMINAMATH_CALUDE_triangle_angle_relation_l3291_329127

theorem triangle_angle_relation (P Q R : Real) (h1 : 5 * Real.sin P + 2 * Real.cos Q = 5) 
  (h2 : 2 * Real.sin Q + 5 * Real.cos P = 3) (h3 : P + Q + R = π) : Real.sin R = 1/20 := by
  sorry

end NUMINAMATH_CALUDE_triangle_angle_relation_l3291_329127


namespace NUMINAMATH_CALUDE_tourist_distribution_l3291_329110

theorem tourist_distribution (total_tourists : ℕ) (h1 : total_tourists = 737) :
  ∃! (num_cars tourists_per_car : ℕ),
    num_cars * tourists_per_car = total_tourists ∧
    num_cars > 0 ∧
    tourists_per_car > 0 :=
by
  sorry

end NUMINAMATH_CALUDE_tourist_distribution_l3291_329110


namespace NUMINAMATH_CALUDE_sum_of_smaller_radii_eq_twice_original_radius_l3291_329117

/-- Represents a tetrahedron with an insphere and four smaller tetrahedrons -/
structure Tetrahedron where
  r : ℝ  -- radius of the insphere of the original tetrahedron
  r₁ : ℝ  -- radius of the insphere of the first smaller tetrahedron
  r₂ : ℝ  -- radius of the insphere of the second smaller tetrahedron
  r₃ : ℝ  -- radius of the insphere of the third smaller tetrahedron
  r₄ : ℝ  -- radius of the insphere of the fourth smaller tetrahedron

/-- The sum of the radii of the inspheres of the four smaller tetrahedrons is equal to twice the radius of the insphere of the original tetrahedron -/
theorem sum_of_smaller_radii_eq_twice_original_radius (t : Tetrahedron) :
  t.r₁ + t.r₂ + t.r₃ + t.r₄ = 2 * t.r := by
  sorry

end NUMINAMATH_CALUDE_sum_of_smaller_radii_eq_twice_original_radius_l3291_329117


namespace NUMINAMATH_CALUDE_nonDefectiveEnginesCount_l3291_329164

/-- Given a number of batches and engines per batch, calculates the number of non-defective engines
    when one fourth of the total engines are defective. -/
def nonDefectiveEngines (batches : ℕ) (enginesPerBatch : ℕ) : ℕ :=
  let totalEngines := batches * enginesPerBatch
  let defectiveEngines := totalEngines / 4
  totalEngines - defectiveEngines

/-- Proves that given 5 batches of 80 engines each, with one fourth being defective,
    the number of non-defective engines is 300. -/
theorem nonDefectiveEnginesCount :
  nonDefectiveEngines 5 80 = 300 := by
  sorry

#eval nonDefectiveEngines 5 80

end NUMINAMATH_CALUDE_nonDefectiveEnginesCount_l3291_329164


namespace NUMINAMATH_CALUDE_find_A_in_subtraction_l3291_329124

/-- Given that AB82 - 9C9 = 493D and A, B, C, D are different digits, prove that A = 5 -/
theorem find_A_in_subtraction (A B C D : ℕ) : 
  A * 1000 + B * 100 + 82 - (9 * 100 + C * 10 + 9) = 4 * 100 + 9 * 10 + D →
  A < 10 ∧ B < 10 ∧ C < 10 ∧ D < 10 →
  A ≠ B ∧ A ≠ C ∧ A ≠ D ∧ B ≠ C ∧ B ≠ D ∧ C ≠ D →
  A = 5 := by
sorry

end NUMINAMATH_CALUDE_find_A_in_subtraction_l3291_329124


namespace NUMINAMATH_CALUDE_complex_fraction_evaluation_l3291_329111

theorem complex_fraction_evaluation (a b : ℂ) (ha : a ≠ 0) (hb : b ≠ 0) 
  (h : a^2 + a*b + b^2 = 0) : 
  (a^15 + b^15) / (a + b)^15 = -2 := by
  sorry

end NUMINAMATH_CALUDE_complex_fraction_evaluation_l3291_329111


namespace NUMINAMATH_CALUDE_triangle_inequality_triangle_inequality_theorem_l3291_329183

-- Define a triangle as a structure with three sides
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  pos_a : a > 0
  pos_b : b > 0
  pos_c : c > 0

-- State the Triangle Inequality Theorem
theorem triangle_inequality (t : Triangle) : 
  t.a + t.b > t.c ∧ t.b + t.c > t.a ∧ t.c + t.a > t.b := by
  sorry

-- Define the property we want to prove
def sum_of_two_sides_greater_than_third (t : Triangle) : Prop :=
  (t.a + t.b > t.c) ∧ (t.b + t.c > t.a) ∧ (t.c + t.a > t.b)

-- Prove that the Triangle Inequality Theorem holds for all triangles
theorem triangle_inequality_theorem :
  ∀ t : Triangle, sum_of_two_sides_greater_than_third t := by
  sorry

end NUMINAMATH_CALUDE_triangle_inequality_triangle_inequality_theorem_l3291_329183


namespace NUMINAMATH_CALUDE_shortest_distance_specific_rectangle_l3291_329122

/-- A rectangle on a cube face with given dimensions -/
structure RectangleOnCube where
  pq : ℝ
  qr : ℝ
  is_vertex_q : Bool
  is_vertex_s : Bool
  on_adjacent_faces : Bool

/-- The shortest distance between two points through a cube -/
def shortest_distance_through_cube (r : RectangleOnCube) : ℝ :=
  sorry

/-- Theorem stating the shortest distance for the given rectangle -/
theorem shortest_distance_specific_rectangle :
  let r : RectangleOnCube := {
    pq := 20,
    qr := 15,
    is_vertex_q := true,
    is_vertex_s := true,
    on_adjacent_faces := true
  }
  shortest_distance_through_cube r = Real.sqrt 337 := by
  sorry

end NUMINAMATH_CALUDE_shortest_distance_specific_rectangle_l3291_329122


namespace NUMINAMATH_CALUDE_yield_fertilization_correlation_l3291_329162

/-- Represents the yield of crops -/
def CropYield : Type := ℝ

/-- Represents the amount of fertilization -/
def Fertilization : Type := ℝ

/-- Defines the relationship between crop yield and fertilization -/
def dependsOn (y : CropYield) (f : Fertilization) : Prop := ∃ (g : Fertilization → CropYield), y = g f

/-- Defines correlation between two variables -/
def correlated (X Y : Type) : Prop := ∃ (f : X → Y), Function.Injective f ∨ Function.Surjective f

/-- Theorem stating that if crop yield depends on fertilization, then they are correlated -/
theorem yield_fertilization_correlation :
  (∀ y : CropYield, ∀ f : Fertilization, dependsOn y f) →
  correlated Fertilization CropYield :=
by sorry

end NUMINAMATH_CALUDE_yield_fertilization_correlation_l3291_329162


namespace NUMINAMATH_CALUDE_enclosed_area_is_3600_l3291_329148

/-- The equation defining the graph -/
def graph_equation (x y : ℝ) : Prop :=
  |x - 120| + |y| = |x/3|

/-- The set of points satisfying the graph equation -/
def graph_set : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | graph_equation p.1 p.2}

/-- The area enclosed by the graph -/
noncomputable def enclosed_area : ℝ := sorry

/-- Theorem stating that the enclosed area is 3600 -/
theorem enclosed_area_is_3600 : enclosed_area = 3600 := by sorry

end NUMINAMATH_CALUDE_enclosed_area_is_3600_l3291_329148


namespace NUMINAMATH_CALUDE_ellipse_equation_l3291_329178

/-- The standard equation of an ellipse given its properties -/
theorem ellipse_equation (f1 f2 p : ℝ × ℝ) (other_ellipse : ℝ → ℝ → Prop) :
  f1 = (0, -4) →
  f2 = (0, 4) →
  p = (-3, 2) →
  (∀ x y, other_ellipse x y ↔ x^2/9 + y^2/4 = 1) →
  (∀ x y, (x^2/15 + y^2/10 = 1) ↔
    (∃ d1 d2 : ℝ,
      d1 + d2 = 10 ∧
      d1^2 = (x - f1.1)^2 + (y - f1.2)^2 ∧
      d2^2 = (x - f2.1)^2 + (y - f2.2)^2 ∧
      x^2/15 + y^2/10 = 1 ∧
      other_ellipse x y)) :=
sorry

end NUMINAMATH_CALUDE_ellipse_equation_l3291_329178


namespace NUMINAMATH_CALUDE_jason_added_erasers_l3291_329133

/-- Given an initial number of erasers and a final number of erasers after Jason adds some,
    calculate how many erasers Jason placed in the drawer. -/
def erasers_added (initial_erasers final_erasers : ℕ) : ℕ :=
  final_erasers - initial_erasers

/-- Theorem stating that Jason added 131 erasers to the drawer. -/
theorem jason_added_erasers :
  erasers_added 139 270 = 131 := by sorry

end NUMINAMATH_CALUDE_jason_added_erasers_l3291_329133


namespace NUMINAMATH_CALUDE_f_value_at_7_l3291_329171

-- Define the function f
def f (a b : ℝ) (x : ℝ) : ℝ := a * x^3 + b * x - 5

-- State the theorem
theorem f_value_at_7 (a b : ℝ) :
  f a b (-7) = 7 → f a b 7 = -17 := by
  sorry

end NUMINAMATH_CALUDE_f_value_at_7_l3291_329171


namespace NUMINAMATH_CALUDE_quadratic_roots_sum_product_l3291_329105

theorem quadratic_roots_sum_product (m n : ℝ) : 
  (m^2 + 4*m - 1 = 0) → 
  (n^2 + 4*n - 1 = 0) → 
  m + n + m*n = -5 := by
sorry

end NUMINAMATH_CALUDE_quadratic_roots_sum_product_l3291_329105


namespace NUMINAMATH_CALUDE_crayons_per_box_l3291_329134

/-- Given an industrial machine that makes 321 crayons a day and 45 full boxes a day,
    prove that there are 7 crayons in each box. -/
theorem crayons_per_box :
  ∀ (total_crayons : ℕ) (total_boxes : ℕ),
    total_crayons = 321 →
    total_boxes = 45 →
    ∃ (crayons_per_box : ℕ),
      crayons_per_box * total_boxes ≤ total_crayons ∧
      (crayons_per_box + 1) * total_boxes > total_crayons ∧
      crayons_per_box = 7 :=
by sorry

end NUMINAMATH_CALUDE_crayons_per_box_l3291_329134


namespace NUMINAMATH_CALUDE_g_continuity_condition_l3291_329112

/-- The function g(x) = 5x - 3 -/
def g (x : ℝ) : ℝ := 5 * x - 3

/-- The statement is true if and only if d ≤ c/5 -/
theorem g_continuity_condition (c d : ℝ) (hc : c > 0) (hd : d > 0) :
  (∀ x : ℝ, |x - 1| < d → |g x - 1| < c) ↔ d ≤ c / 5 := by
  sorry

end NUMINAMATH_CALUDE_g_continuity_condition_l3291_329112


namespace NUMINAMATH_CALUDE_line_slope_l3291_329159

/-- The slope of the line (x/2) + (y/3) = 2 is -3/2 -/
theorem line_slope (x y : ℝ) :
  (x / 2 + y / 3 = 2) → (∃ b : ℝ, y = (-3/2) * x + b) := by
  sorry

end NUMINAMATH_CALUDE_line_slope_l3291_329159


namespace NUMINAMATH_CALUDE_set_A_equality_l3291_329140

def A : Set ℕ := {x | x ≤ 4}

theorem set_A_equality : A = {0, 1, 2, 3, 4} := by sorry

end NUMINAMATH_CALUDE_set_A_equality_l3291_329140


namespace NUMINAMATH_CALUDE_circle_center_coordinate_sum_l3291_329197

/-- Given a circle with equation x^2 + y^2 + 4x - 12y + 20 = 0, 
    the sum of the x and y coordinates of its center is 4 -/
theorem circle_center_coordinate_sum :
  ∀ (x y : ℝ), x^2 + y^2 + 4*x - 12*y + 20 = 0 →
  ∃ (h k : ℝ), (∀ (x y : ℝ), (x - h)^2 + (y - k)^2 = (x^2 + y^2 + 4*x - 12*y + 20)) ∧ 
                h + k = 4 := by
  sorry

end NUMINAMATH_CALUDE_circle_center_coordinate_sum_l3291_329197


namespace NUMINAMATH_CALUDE_complement_P_subset_Q_l3291_329119

open Set Real

theorem complement_P_subset_Q : 
  let P : Set ℝ := {x | x < 1}
  let Q : Set ℝ := {x | x > -1}
  (compl P : Set ℝ) ⊆ Q := by
  sorry

end NUMINAMATH_CALUDE_complement_P_subset_Q_l3291_329119


namespace NUMINAMATH_CALUDE_scenic_spot_probabilities_l3291_329195

def total_spots : ℕ := 10
def five_a_spots : ℕ := 4
def four_a_spots : ℕ := 6

def spots_after_yuntai : ℕ := 4

theorem scenic_spot_probabilities :
  (five_a_spots : ℚ) / total_spots = 2 / 5 ∧
  (2 : ℚ) / (spots_after_yuntai * (spots_after_yuntai - 1)) = 1 / 6 := by
  sorry


end NUMINAMATH_CALUDE_scenic_spot_probabilities_l3291_329195


namespace NUMINAMATH_CALUDE_team_combinations_l3291_329115

theorem team_combinations (n k : ℕ) (h1 : n = 7) (h2 : k = 4) :
  Nat.choose n k = 35 := by
  sorry

end NUMINAMATH_CALUDE_team_combinations_l3291_329115


namespace NUMINAMATH_CALUDE_log_equation_solution_l3291_329196

theorem log_equation_solution (x : ℝ) (h : x > 0) :
  Real.log 216 / Real.log (2 * x) = x →
  x = 3 ∧ ¬∃ (n : ℕ), x = n^2 ∧ ¬∃ (n : ℕ), x = n^3 ∧ ∃ (n : ℕ), x = n := by
  sorry

end NUMINAMATH_CALUDE_log_equation_solution_l3291_329196


namespace NUMINAMATH_CALUDE_range_of_u_l3291_329184

theorem range_of_u (x y : ℝ) (h : x^2 / 3 + y^2 = 1) :
  let u := |2*x + y - 4| + |3 - x - 2*y|
  1 ≤ u ∧ u ≤ 13 :=
sorry

end NUMINAMATH_CALUDE_range_of_u_l3291_329184


namespace NUMINAMATH_CALUDE_regression_line_equation_l3291_329139

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a linear equation y = mx + b -/
structure LinearEquation where
  m : ℝ
  b : ℝ

/-- Check if a point lies on a line given by a linear equation -/
def pointOnLine (p : Point) (eq : LinearEquation) : Prop :=
  p.y = eq.m * p.x + eq.b

theorem regression_line_equation 
  (slope : ℝ) 
  (center : Point) 
  (h_slope : slope = 1.23)
  (h_center : center = ⟨4, 5⟩) :
  ∃ (eq : LinearEquation), 
    eq.m = slope ∧ 
    pointOnLine center eq ∧ 
    eq = ⟨1.23, 0.08⟩ := by
  sorry

end NUMINAMATH_CALUDE_regression_line_equation_l3291_329139


namespace NUMINAMATH_CALUDE_zeros_of_f_l3291_329129

noncomputable section

-- Define the piecewise function f
def f (a : ℝ) (x : ℝ) : ℝ :=
  if x > 1/2 then x - 2/x else x^2 + 2*x + a - 1

-- Define the set of zeros for f
def zeros (a : ℝ) : Set ℝ :=
  {x : ℝ | f a x = 0}

-- Theorem statement
theorem zeros_of_f (a : ℝ) (h : a > 0) :
  zeros a = 
    if a > 2 then {Real.sqrt 2}
    else if a = 2 then {Real.sqrt 2, -1}
    else {Real.sqrt 2, -1 + Real.sqrt (2-a), -1 - Real.sqrt (2-a)} :=
by sorry

end

end NUMINAMATH_CALUDE_zeros_of_f_l3291_329129


namespace NUMINAMATH_CALUDE_distance_between_vertices_parabolas_distance_l3291_329185

-- Define the equation
def equation (x y : ℝ) : Prop :=
  Real.sqrt (x^2 + y^2) + abs (y - 2) = 4

-- Define the two parabolas
def parabola1 (x y : ℝ) : Prop :=
  y = -(1/12) * x^2 + 3

def parabola2 (x y : ℝ) : Prop :=
  y = (1/4) * x^2 - 1

-- Define the vertices of the parabolas
def vertex1 : ℝ × ℝ := (0, 3)
def vertex2 : ℝ × ℝ := (0, -1)

-- Theorem: The distance between the vertices is 4
theorem distance_between_vertices : 
  Real.sqrt ((vertex1.1 - vertex2.1)^2 + (vertex1.2 - vertex2.2)^2) = 4 := by
  sorry

-- Main theorem
theorem parabolas_distance : ∃ (x1 y1 x2 y2 : ℝ),
  equation x1 y1 ∧ equation x2 y2 ∧
  parabola1 x1 y1 ∧ parabola2 x2 y2 ∧
  Real.sqrt ((x1 - x2)^2 + (y1 - y2)^2) = 4 := by
  sorry

end NUMINAMATH_CALUDE_distance_between_vertices_parabolas_distance_l3291_329185


namespace NUMINAMATH_CALUDE_HE_in_possible_values_l3291_329113

/-- A quadrilateral with side lengths satisfying certain conditions -/
structure Quadrilateral :=
  (EF : ℝ)
  (FG : ℝ)
  (GH : ℝ)
  (HE : ℤ)
  (ef_eq : EF = 7)
  (fg_eq : FG = 21)
  (gh_eq : GH = 7)

/-- The possible values for HE in the quadrilateral -/
def possible_HE (q : Quadrilateral) : Set ℤ :=
  {n : ℤ | 15 ≤ n ∧ n ≤ 27}

/-- The theorem stating that HE must be in the set of possible values -/
theorem HE_in_possible_values (q : Quadrilateral) : q.HE ∈ possible_HE q := by
  sorry

end NUMINAMATH_CALUDE_HE_in_possible_values_l3291_329113


namespace NUMINAMATH_CALUDE_painter_problem_l3291_329137

theorem painter_problem (total_rooms : ℕ) (time_per_room : ℕ) (time_left : ℕ) 
  (h1 : total_rooms = 9)
  (h2 : time_per_room = 8)
  (h3 : time_left = 32) :
  total_rooms - (time_left / time_per_room) = 5 := by
sorry

end NUMINAMATH_CALUDE_painter_problem_l3291_329137


namespace NUMINAMATH_CALUDE_max_a_value_l3291_329125

/-- An even function f defined on ℝ such that f(x) = e^x for x ≥ 0 -/
noncomputable def f : ℝ → ℝ :=
  fun x => if x ≥ 0 then Real.exp x else Real.exp (-x)

theorem max_a_value :
  (∃ a : ℝ, ∀ x ∈ Set.Icc a (a + 1), f (x + a) ≥ f x ^ 2) ∧
  (∀ a : ℝ, a > -3/4 → ∃ x ∈ Set.Icc a (a + 1), f (x + a) < f x ^ 2) :=
by sorry

end NUMINAMATH_CALUDE_max_a_value_l3291_329125


namespace NUMINAMATH_CALUDE_total_turtles_l3291_329180

def turtle_problem (lucas rebecca miguel tran pedro kristen kris trey : ℕ) : Prop :=
  lucas = 8 ∧
  rebecca = 2 * lucas ∧
  miguel = rebecca + 10 ∧
  tran = miguel + 5 ∧
  pedro = 2 * tran ∧
  kristen = 3 * pedro ∧
  kris = kristen / 4 ∧
  trey = 5 * kris ∧
  lucas + rebecca + miguel + tran + pedro + kristen + kris + trey = 605

theorem total_turtles :
  ∃ (lucas rebecca miguel tran pedro kristen kris trey : ℕ),
    turtle_problem lucas rebecca miguel tran pedro kristen kris trey :=
by
  sorry

end NUMINAMATH_CALUDE_total_turtles_l3291_329180


namespace NUMINAMATH_CALUDE_product_approx_six_times_number_l3291_329144

-- Define a function to check if two numbers are approximately equal
def approx_equal (x y : ℝ) : Prop := abs (x - y) ≤ 1

-- Theorem 1: The product of 198 × 2 is approximately 400
theorem product_approx : approx_equal (198 * 2) 400 := by sorry

-- Theorem 2: If twice a number is 78, then six times that number is 240
theorem six_times_number (x : ℝ) (h : 2 * x = 78) : 6 * x = 240 := by sorry

end NUMINAMATH_CALUDE_product_approx_six_times_number_l3291_329144


namespace NUMINAMATH_CALUDE_a₉₉_eq_182_l3291_329194

/-- An arithmetic sequence with specified properties -/
structure ArithmeticSequence where
  -- First term
  a₁ : ℝ
  -- Common difference
  d : ℝ
  -- Sum of first 17 terms is 34
  sum_17 : 17 * a₁ + (17 * 16 / 2) * d = 34
  -- Third term is -10
  a₃ : a₁ + 2 * d = -10

/-- The 99th term of the arithmetic sequence -/
def a₉₉ (seq : ArithmeticSequence) : ℝ := seq.a₁ + 98 * seq.d

/-- Theorem stating that a₉₉ = 182 for the given arithmetic sequence -/
theorem a₉₉_eq_182 (seq : ArithmeticSequence) : a₉₉ seq = 182 := by
  sorry

end NUMINAMATH_CALUDE_a₉₉_eq_182_l3291_329194


namespace NUMINAMATH_CALUDE_three_numbers_problem_l3291_329181

theorem three_numbers_problem :
  ∃ (x y z : ℝ), x > 0 ∧ y > 0 ∧ z > 0 ∧
  x = 1.4 * y ∧
  x / z = 14 / 11 ∧
  z - y = 0.125 * (x + y) - 40 ∧
  x = 280 ∧ y = 200 ∧ z = 220 := by
sorry

end NUMINAMATH_CALUDE_three_numbers_problem_l3291_329181


namespace NUMINAMATH_CALUDE_divisibility_implies_one_l3291_329167

theorem divisibility_implies_one (n : ℕ+) (h : n ∣ 2^n.val - 1) : n = 1 := by
  sorry

end NUMINAMATH_CALUDE_divisibility_implies_one_l3291_329167


namespace NUMINAMATH_CALUDE_pet_store_theorem_l3291_329143

/-- Given a ratio of cats to dogs to birds and the number of cats, 
    calculate the number of dogs and birds -/
def pet_store_count (cat_ratio dog_ratio bird_ratio num_cats : ℕ) : ℕ × ℕ :=
  let scale_factor := num_cats / cat_ratio
  (dog_ratio * scale_factor, bird_ratio * scale_factor)

/-- Theorem: Given the ratio 2:3:4 for cats:dogs:birds and 20 cats, 
    there are 30 dogs and 40 birds -/
theorem pet_store_theorem : 
  pet_store_count 2 3 4 20 = (30, 40) := by
  sorry

end NUMINAMATH_CALUDE_pet_store_theorem_l3291_329143


namespace NUMINAMATH_CALUDE_interest_rate_calculation_l3291_329155

/-- Proves that the interest rate at which A lent to B is 15% given the conditions --/
theorem interest_rate_calculation (principal : ℝ) (rate_B_to_C : ℝ) (time : ℝ) (B_gain : ℝ) 
  (h_principal : principal = 2000)
  (h_rate_B_to_C : rate_B_to_C = 17)
  (h_time : time = 4)
  (h_B_gain : B_gain = 160)
  : ∃ R : ℝ, R = 15 ∧ 
    principal * (rate_B_to_C / 100) * time - principal * (R / 100) * time = B_gain :=
by
  sorry

#check interest_rate_calculation

end NUMINAMATH_CALUDE_interest_rate_calculation_l3291_329155


namespace NUMINAMATH_CALUDE_algorithm_uniqueness_false_l3291_329182

-- Define the concept of an algorithm
structure Algorithm where
  finite : Bool
  determinate : Bool
  outputProperty : Bool

-- Define the property of uniqueness for algorithms
def isUnique (problemClass : Type) (alg : Algorithm) : Prop :=
  ∀ (otherAlg : Algorithm), alg = otherAlg

-- Theorem statement
theorem algorithm_uniqueness_false :
  ∃ (problemClass : Type) (alg1 alg2 : Algorithm),
    alg1.finite ∧ alg1.determinate ∧ alg1.outputProperty ∧
    alg2.finite ∧ alg2.determinate ∧ alg2.outputProperty ∧
    alg1 ≠ alg2 :=
sorry

end NUMINAMATH_CALUDE_algorithm_uniqueness_false_l3291_329182


namespace NUMINAMATH_CALUDE_circle_symmetry_l3291_329161

/-- Given a circle with center (1,2) and radius 1, symmetric about the line y = x + b,
    prove that b = 1 -/
theorem circle_symmetry (b : ℝ) : 
  (∀ x y : ℝ, (x - 1)^2 + (y - 2)^2 = 1 ↔ (y - x = b ∧ (x + y - 3)^2 + (y - x - b)^2 / 4 = 1)) →
  b = 1 := by
sorry

end NUMINAMATH_CALUDE_circle_symmetry_l3291_329161


namespace NUMINAMATH_CALUDE_melanie_has_41_balloons_l3291_329176

/-- The number of blue balloons Joan has -/
def joan_balloons : ℕ := 40

/-- The total number of blue balloons Joan and Melanie have together -/
def total_balloons : ℕ := 81

/-- The number of blue balloons Melanie has -/
def melanie_balloons : ℕ := total_balloons - joan_balloons

/-- Theorem stating that Melanie has 41 blue balloons -/
theorem melanie_has_41_balloons : melanie_balloons = 41 := by
  sorry

end NUMINAMATH_CALUDE_melanie_has_41_balloons_l3291_329176


namespace NUMINAMATH_CALUDE_good_price_after_discounts_l3291_329100

theorem good_price_after_discounts (P : ℝ) : 
  P * (1 - 0.20) * (1 - 0.10) * (1 - 0.05) = 6700 → P = 9798.25 := by
  sorry

end NUMINAMATH_CALUDE_good_price_after_discounts_l3291_329100


namespace NUMINAMATH_CALUDE_geometric_sequence_implies_geometric_subsequences_exists_non_geometric_sequence_with_geometric_subsequences_l3291_329126

def is_geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = r * a n

def odd_subsequence (a : ℕ → ℝ) : ℕ → ℝ :=
  λ k => a (2 * k - 1)

def even_subsequence (a : ℕ → ℝ) : ℕ → ℝ :=
  λ k => a (2 * k)

theorem geometric_sequence_implies_geometric_subsequences
  (a : ℕ → ℝ) (h : is_geometric_sequence a) :
  is_geometric_sequence (odd_subsequence a) ∧
  is_geometric_sequence (even_subsequence a) :=
sorry

theorem exists_non_geometric_sequence_with_geometric_subsequences :
  ∃ a : ℕ → ℝ,
    is_geometric_sequence (odd_subsequence a) ∧
    is_geometric_sequence (even_subsequence a) ∧
    ¬is_geometric_sequence a :=
sorry

end NUMINAMATH_CALUDE_geometric_sequence_implies_geometric_subsequences_exists_non_geometric_sequence_with_geometric_subsequences_l3291_329126


namespace NUMINAMATH_CALUDE_root_magnitude_of_quadratic_l3291_329102

theorem root_magnitude_of_quadratic (z : ℂ) : z^2 + z + 1 = 0 → Complex.abs z = 1 := by
  sorry

end NUMINAMATH_CALUDE_root_magnitude_of_quadratic_l3291_329102


namespace NUMINAMATH_CALUDE_min_four_dollar_frisbees_l3291_329135

theorem min_four_dollar_frisbees :
  ∀ (x y : ℕ),
  x + y = 64 →
  3 * x + 4 * y = 200 →
  y ≥ 8 :=
by
  sorry

end NUMINAMATH_CALUDE_min_four_dollar_frisbees_l3291_329135


namespace NUMINAMATH_CALUDE_cracker_ratio_is_one_l3291_329123

/-- The number of crackers Marcus has -/
def marcus_crackers : ℕ := 27

/-- The number of crackers Mona has -/
def mona_crackers : ℕ := marcus_crackers

/-- The ratio of Marcus's crackers to Mona's crackers -/
def cracker_ratio : ℚ := marcus_crackers / mona_crackers

theorem cracker_ratio_is_one : cracker_ratio = 1 := by
  sorry

end NUMINAMATH_CALUDE_cracker_ratio_is_one_l3291_329123


namespace NUMINAMATH_CALUDE_deepak_third_period_profit_l3291_329168

def anand_investment : ℕ := 22500
def deepak_investment : ℕ := 35000
def total_investment : ℕ := anand_investment + deepak_investment

def first_period_profit : ℕ := 9600
def second_period_profit : ℕ := 12800
def third_period_profit : ℕ := 18000

def profit_share (investment : ℕ) (total_profit : ℕ) : ℚ :=
  (investment : ℚ) / (total_investment : ℚ) * (total_profit : ℚ)

theorem deepak_third_period_profit :
  profit_share deepak_investment third_period_profit = 10960 := by
  sorry

end NUMINAMATH_CALUDE_deepak_third_period_profit_l3291_329168


namespace NUMINAMATH_CALUDE_power_function_m_value_l3291_329106

/-- A power function that passes through (2, 16) and (1/2, m) -/
def power_function (x : ℝ) : ℝ := x ^ 4

theorem power_function_m_value :
  let f := power_function
  (f 2 = 16) ∧ (∃ m, f (1/2) = m) →
  f (1/2) = 1/16 := by
sorry

end NUMINAMATH_CALUDE_power_function_m_value_l3291_329106


namespace NUMINAMATH_CALUDE_equal_area_rectangles_l3291_329128

/-- Represents the dimensions of a rectangle -/
structure Rectangle where
  length : ℝ
  width : ℝ

/-- Calculates the area of a rectangle -/
def area (r : Rectangle) : ℝ := r.length * r.width

theorem equal_area_rectangles (r1 r2 : Rectangle) 
  (h1 : r1.length = 4)
  (h2 : r1.width = 30)
  (h3 : r2.width = 15)
  (h4 : area r1 = area r2) :
  r2.length = 8 := by
  sorry

end NUMINAMATH_CALUDE_equal_area_rectangles_l3291_329128


namespace NUMINAMATH_CALUDE_dulce_has_three_points_l3291_329172

-- Define the points for each person and the team
def max_points : ℕ := 5
def dulce_points : ℕ := 3  -- This is what we want to prove
def val_points (d : ℕ) : ℕ := 2 * (max_points + d)
def team_total (d : ℕ) : ℕ := max_points + d + val_points d

-- Define the opponent's points and the point difference
def opponent_points : ℕ := 40
def point_difference : ℕ := 16

-- Theorem to prove
theorem dulce_has_three_points : 
  team_total dulce_points = opponent_points - point_difference := by
  sorry


end NUMINAMATH_CALUDE_dulce_has_three_points_l3291_329172


namespace NUMINAMATH_CALUDE_factorial_ratio_72_l3291_329170

theorem factorial_ratio_72 : ∃! (n : ℕ), (Nat.factorial (n + 2)) / (Nat.factorial n) = 72 :=
by
  -- Proof goes here
  sorry

end NUMINAMATH_CALUDE_factorial_ratio_72_l3291_329170


namespace NUMINAMATH_CALUDE_grisha_remaining_money_l3291_329101

-- Define the given constants
def initial_money : ℕ := 5000
def bunny_price : ℕ := 45
def bag_price : ℕ := 30
def bunnies_per_bag : ℕ := 30

-- Define the function to calculate the remaining money
def remaining_money : ℕ :=
  let full_bag_cost := bag_price + bunnies_per_bag * bunny_price
  let full_bags := initial_money / full_bag_cost
  let money_after_full_bags := initial_money - full_bags * full_bag_cost
  let additional_bag_cost := bag_price
  let money_for_extra_bunnies := money_after_full_bags - additional_bag_cost
  let extra_bunnies := money_for_extra_bunnies / bunny_price
  initial_money - (full_bags * full_bag_cost + additional_bag_cost + extra_bunnies * bunny_price)

-- The theorem to prove
theorem grisha_remaining_money :
  remaining_money = 20 := by sorry

end NUMINAMATH_CALUDE_grisha_remaining_money_l3291_329101


namespace NUMINAMATH_CALUDE_sufficient_but_not_necessary_l3291_329120

theorem sufficient_but_not_necessary (x : ℝ) :
  (∀ x, -1 ≤ x ∧ x ≤ 5 → (x - 5) * (x + 1) < 0) ∧
  (∃ x, (x - 5) * (x + 1) < 0 ∧ (x < -1 ∨ x > 5)) :=
by sorry

end NUMINAMATH_CALUDE_sufficient_but_not_necessary_l3291_329120


namespace NUMINAMATH_CALUDE_min_blue_eyes_and_backpack_proof_l3291_329103

def min_blue_eyes_and_backpack (total_students blue_eyes backpacks glasses : ℕ) : ℕ :=
  blue_eyes - (total_students - backpacks)

theorem min_blue_eyes_and_backpack_proof 
  (total_students : ℕ) 
  (blue_eyes : ℕ) 
  (backpacks : ℕ) 
  (glasses : ℕ) 
  (h1 : total_students = 35)
  (h2 : blue_eyes = 18)
  (h3 : backpacks = 25)
  (h4 : glasses = 10)
  (h5 : ∃ (x : ℕ), x ≥ 2 ∧ x ≤ glasses ∧ x ≤ blue_eyes) :
  min_blue_eyes_and_backpack total_students blue_eyes backpacks glasses = 10 := by
  sorry

#eval min_blue_eyes_and_backpack 35 18 25 10

end NUMINAMATH_CALUDE_min_blue_eyes_and_backpack_proof_l3291_329103


namespace NUMINAMATH_CALUDE_function_non_negative_iff_k_geq_neg_one_l3291_329199

/-- The function f(x) = |x^2 - 1| + x^2 + kx is non-negative on (0, +∞) if and only if k ≥ -1 -/
theorem function_non_negative_iff_k_geq_neg_one (k : ℝ) :
  (∀ x > 0, |x^2 - 1| + x^2 + k*x ≥ 0) ↔ k ≥ -1 :=
by sorry

end NUMINAMATH_CALUDE_function_non_negative_iff_k_geq_neg_one_l3291_329199


namespace NUMINAMATH_CALUDE_mark_travel_distance_l3291_329132

/-- Represents the time in minutes to travel one mile on day 1 -/
def initial_time : ℕ := 3

/-- Calculates the time in minutes to travel one mile on a given day -/
def time_for_mile (day : ℕ) : ℕ :=
  initial_time + 3 * (day - 1)

/-- Calculates the distance traveled in miles on a given day -/
def distance_per_day (day : ℕ) : ℕ :=
  if 60 % (time_for_mile day) = 0 then 60 / (time_for_mile day) else 0

/-- Calculates the total distance traveled over 6 days -/
def total_distance : ℕ :=
  (List.range 6).map (fun i => distance_per_day (i + 1)) |> List.sum

theorem mark_travel_distance :
  total_distance = 39 := by sorry

end NUMINAMATH_CALUDE_mark_travel_distance_l3291_329132


namespace NUMINAMATH_CALUDE_brother_d_payment_l3291_329130

theorem brother_d_payment (n : ℕ) (a₁ d : ℚ) (h₁ : n = 5) (h₂ : a₁ = 300) 
  (h₃ : n / 2 * (2 * a₁ + (n - 1) * d) = 1000) : a₁ + 3 * d = 450 := by
  sorry

end NUMINAMATH_CALUDE_brother_d_payment_l3291_329130


namespace NUMINAMATH_CALUDE_circle_slope_range_l3291_329158

theorem circle_slope_range (x y : ℝ) (h : x^2 + (y - 3)^2 = 1) :
  ∃ (k : ℝ), k ∈ Set.Icc (-2 * Real.sqrt 2) (2 * Real.sqrt 2) ∧ y = k * x :=
sorry

end NUMINAMATH_CALUDE_circle_slope_range_l3291_329158


namespace NUMINAMATH_CALUDE_no_positive_rational_solution_l3291_329179

theorem no_positive_rational_solution (n : ℕ+) :
  ¬∃ (x y : ℚ), 0 < x ∧ 0 < y ∧ x + y + 1/x + 1/y = 3*n := by
  sorry

end NUMINAMATH_CALUDE_no_positive_rational_solution_l3291_329179


namespace NUMINAMATH_CALUDE_pattern_repeats_proof_l3291_329118

/-- The number of beads in one pattern -/
def beads_per_pattern : ℕ := 14

/-- The number of beads in one bracelet -/
def beads_per_bracelet : ℕ := 42

/-- The total number of beads for 1 bracelet and 10 necklaces -/
def total_beads : ℕ := 742

/-- The number of times the pattern repeats per necklace -/
def pattern_repeats_per_necklace : ℕ := 5

/-- Theorem stating that the pattern repeats 5 times per necklace -/
theorem pattern_repeats_proof : 
  beads_per_bracelet + 10 * pattern_repeats_per_necklace * beads_per_pattern = total_beads :=
by sorry

end NUMINAMATH_CALUDE_pattern_repeats_proof_l3291_329118


namespace NUMINAMATH_CALUDE_reciprocal_sum_theorem_l3291_329121

theorem reciprocal_sum_theorem (x y : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) 
  (h : x + y = 3 * x * y) : 1 / x + 1 / y = 3 := by
  sorry

end NUMINAMATH_CALUDE_reciprocal_sum_theorem_l3291_329121


namespace NUMINAMATH_CALUDE_systematic_sampling_solution_l3291_329131

/-- Represents a systematic sampling problem -/
structure SystematicSampling where
  population_size : ℕ
  sample_size : ℕ

/-- Represents a solution to a systematic sampling problem -/
structure SystematicSamplingSolution where
  excluded : ℕ
  interval : ℕ

/-- Checks if a solution is valid for a given systematic sampling problem -/
def is_valid_solution (problem : SystematicSampling) (solution : SystematicSamplingSolution) : Prop :=
  (problem.population_size - solution.excluded) % problem.sample_size = 0 ∧
  (problem.population_size - solution.excluded) / problem.sample_size = solution.interval

theorem systematic_sampling_solution 
  (problem : SystematicSampling) 
  (h_pop : problem.population_size = 102) 
  (h_sample : problem.sample_size = 9) :
  ∃ (solution : SystematicSamplingSolution), 
    solution.excluded = 3 ∧ 
    solution.interval = 11 ∧ 
    is_valid_solution problem solution :=
sorry

end NUMINAMATH_CALUDE_systematic_sampling_solution_l3291_329131


namespace NUMINAMATH_CALUDE_complete_square_ratio_l3291_329173

/-- Represents a quadratic expression in the form ak² + bk + c -/
structure QuadraticExpression (α : Type*) [Ring α] where
  a : α
  b : α
  c : α

/-- Represents a quadratic expression in completed square form a(k + b)² + c -/
structure CompletedSquareForm (α : Type*) [Ring α] where
  a : α
  b : α
  c : α

/-- Function to convert a QuadraticExpression to CompletedSquareForm -/
def completeSquare {α : Type*} [Field α] (q : QuadraticExpression α) : CompletedSquareForm α :=
  sorry

theorem complete_square_ratio {α : Type*} [Field α] :
  let q : QuadraticExpression α := ⟨4, -8, 16⟩
  let csf := completeSquare q
  csf.c / csf.b = -12 := by sorry

end NUMINAMATH_CALUDE_complete_square_ratio_l3291_329173


namespace NUMINAMATH_CALUDE_min_blocks_correct_l3291_329174

/-- A list of positive integer weights representing ice blocks -/
def IceBlocks := List Nat

/-- Predicate to check if a list of weights can satisfy any demand (p, q) where p + q ≤ 2016 -/
def CanSatisfyDemand (blocks : IceBlocks) : Prop :=
  ∀ p q : Nat, p + q ≤ 2016 → ∃ (subsetP subsetQ : List Nat),
    subsetP.Disjoint subsetQ ∧
    subsetP.sum = p ∧
    subsetQ.sum = q ∧
    (subsetP ++ subsetQ).Sublist blocks

/-- The minimum number of ice blocks needed -/
def MinBlocks : Nat := 18

/-- Theorem stating that MinBlocks is the minimum number of ice blocks needed -/
theorem min_blocks_correct :
  (∃ (blocks : IceBlocks), blocks.length = MinBlocks ∧ blocks.all (· > 0) ∧ CanSatisfyDemand blocks) ∧
  (∀ (blocks : IceBlocks), blocks.length < MinBlocks → ¬CanSatisfyDemand blocks) := by
  sorry

#check min_blocks_correct

end NUMINAMATH_CALUDE_min_blocks_correct_l3291_329174


namespace NUMINAMATH_CALUDE_trig_identity_on_line_l3291_329165

/-- If the terminal side of angle α lies on the line y = 2x, 
    then sin²α - cos²α + sin α * cos α = 1 -/
theorem trig_identity_on_line (α : Real) 
  (h : Real.tan α = 2) : 
  Real.sin α ^ 2 - Real.cos α ^ 2 + Real.sin α * Real.cos α = 1 := by
  sorry

end NUMINAMATH_CALUDE_trig_identity_on_line_l3291_329165


namespace NUMINAMATH_CALUDE_sqrt_x_plus_sqrt_x_equals_y_infinitely_many_pairs_l3291_329186

theorem sqrt_x_plus_sqrt_x_equals_y (m : ℕ) :
  ∃ (x y : ℚ), (x + Real.sqrt x).sqrt = y ∧
    y = Real.sqrt (m * (m + 1)) ∧
    x = (2 * y^2 + 1 + Real.sqrt (4 * y^2 + 1)) / 2 :=
by sorry

theorem infinitely_many_pairs :
  ∀ n : ℕ, ∃ (S : Finset (ℚ × ℚ)), S.card = n ∧
    ∀ (x y : ℚ), (x, y) ∈ S → (x + Real.sqrt x).sqrt = y :=
by sorry

end NUMINAMATH_CALUDE_sqrt_x_plus_sqrt_x_equals_y_infinitely_many_pairs_l3291_329186


namespace NUMINAMATH_CALUDE_two_digit_number_five_times_sum_of_digits_l3291_329146

theorem two_digit_number_five_times_sum_of_digits : 
  ∃! n : ℕ, 10 ≤ n ∧ n < 100 ∧ n = 5 * (n / 10 + n % 10) :=
by
  sorry

end NUMINAMATH_CALUDE_two_digit_number_five_times_sum_of_digits_l3291_329146


namespace NUMINAMATH_CALUDE_apartment_complex_households_l3291_329189

/-- Calculates the total number of households in an apartment complex. -/
def total_households (num_buildings : ℕ) (num_floors : ℕ) 
  (households_first_floor : ℕ) (households_other_floors : ℕ) : ℕ :=
  num_buildings * (households_first_floor + (num_floors - 1) * households_other_floors)

/-- Theorem stating that the total number of households in the given apartment complex is 68. -/
theorem apartment_complex_households : 
  total_households 4 6 2 3 = 68 := by
  sorry

#eval total_households 4 6 2 3

end NUMINAMATH_CALUDE_apartment_complex_households_l3291_329189


namespace NUMINAMATH_CALUDE_system_solution_l3291_329160

theorem system_solution (x y z t : ℤ) :
  (3 * x - 2 * y + 4 * z + 2 * t = 19) →
  (5 * x + 6 * y - 2 * z + 3 * t = 23) →
  (x = 16 * z - 18 * y - 11 ∧ t = 28 * y - 26 * z + 26) :=
by sorry

end NUMINAMATH_CALUDE_system_solution_l3291_329160


namespace NUMINAMATH_CALUDE_minimum_value_theorem_l3291_329108

theorem minimum_value_theorem (a b : ℝ) (ha : a > 0) (hb : b > 0) 
  (h_geometric_mean : Real.sqrt 3 = Real.sqrt (3^a * 3^(2*b))) :
  (∀ x y, x > 0 → y > 0 → 2/x + 1/y ≥ 2/a + 1/b) → 2/a + 1/b = 8 :=
by sorry

end NUMINAMATH_CALUDE_minimum_value_theorem_l3291_329108


namespace NUMINAMATH_CALUDE_root_fraction_to_power_l3291_329169

theorem root_fraction_to_power : (81 ^ (1/3)) / (81 ^ (1/4)) = 81 ^ (1/12) := by
  sorry

end NUMINAMATH_CALUDE_root_fraction_to_power_l3291_329169


namespace NUMINAMATH_CALUDE_game_not_fair_l3291_329190

/-- Represents the game described in the problem -/
structure Game where
  deck_size : ℕ
  named_cards : ℕ
  win_amount : ℚ
  lose_amount : ℚ

/-- Calculates the expected winnings for the guessing player -/
def expected_winnings (g : Game) : ℚ :=
  let p_named := g.named_cards / g.deck_size
  let p_not_named := 1 - p_named
  let max_cards_per_suit := g.deck_size / 4
  let p_correct_guess_not_named := max_cards_per_suit / (g.deck_size - g.named_cards)
  let expected_case1 := p_named * g.win_amount
  let expected_case2 := p_not_named * (p_correct_guess_not_named * g.win_amount - (1 - p_correct_guess_not_named) * g.lose_amount)
  expected_case1 + expected_case2

/-- The theorem stating that the expected winnings for the guessing player are 1/8 Ft -/
theorem game_not_fair (g : Game) (h1 : g.deck_size = 32) (h2 : g.named_cards = 4) 
    (h3 : g.win_amount = 2) (h4 : g.lose_amount = 1) : 
  expected_winnings g = 1/8 := by
  sorry

end NUMINAMATH_CALUDE_game_not_fair_l3291_329190


namespace NUMINAMATH_CALUDE_cubic_sum_minus_product_l3291_329116

theorem cubic_sum_minus_product (a b c : ℝ) 
  (sum_eq : a + b + c = 12) 
  (sum_product_eq : a * b + a * c + b * c = 30) : 
  a^3 + b^3 + c^3 - 3*a*b*c = 648 := by sorry

end NUMINAMATH_CALUDE_cubic_sum_minus_product_l3291_329116


namespace NUMINAMATH_CALUDE_colonization_combinations_l3291_329104

def total_planets : ℕ := 15
def earth_like_planets : ℕ := 8
def mars_like_planets : ℕ := 7
def earth_like_cost : ℕ := 3
def mars_like_cost : ℕ := 1
def total_colonization_units : ℕ := 18

def valid_combination (earth_colonies mars_colonies : ℕ) : Prop :=
  earth_colonies ≤ earth_like_planets ∧
  mars_colonies ≤ mars_like_planets ∧
  earth_colonies * earth_like_cost + mars_colonies * mars_like_cost = total_colonization_units

def count_combinations : ℕ := sorry

theorem colonization_combinations :
  count_combinations = 2478 :=
sorry

end NUMINAMATH_CALUDE_colonization_combinations_l3291_329104


namespace NUMINAMATH_CALUDE_largest_number_less_than_150_divisible_by_3_l3291_329157

theorem largest_number_less_than_150_divisible_by_3 :
  ∃ (x : ℕ), x = 12 ∧
  (∀ (y : ℕ), 11 * y < 150 ∧ 3 ∣ y → y ≤ x) ∧
  11 * x < 150 ∧ 3 ∣ x :=
by sorry

end NUMINAMATH_CALUDE_largest_number_less_than_150_divisible_by_3_l3291_329157


namespace NUMINAMATH_CALUDE_number_equation_solution_l3291_329175

theorem number_equation_solution : 
  ∃ x : ℝ, (2 * x = 3 * x - 25) ∧ x = 25 := by
  sorry

end NUMINAMATH_CALUDE_number_equation_solution_l3291_329175


namespace NUMINAMATH_CALUDE_triangle_sine_cosine_identity_l3291_329107

/-- For angles A, B, and C of a triangle, sin A + sin B + sin C = 4 cos(A/2) cos(B/2) cos(C/2). -/
theorem triangle_sine_cosine_identity (A B C : ℝ) (h : A + B + C = Real.pi) :
  Real.sin A + Real.sin B + Real.sin C = 4 * Real.cos (A/2) * Real.cos (B/2) * Real.cos (C/2) := by
  sorry

end NUMINAMATH_CALUDE_triangle_sine_cosine_identity_l3291_329107


namespace NUMINAMATH_CALUDE_max_squirrel_attacks_l3291_329198

theorem max_squirrel_attacks (N : ℕ+) (a b c : ℤ) 
  (h1 : a ≥ b) (h2 : b ≥ c) (h3 : a - c = N) : 
  (∃ k : ℕ, k ≤ N ∧ 
    (∀ m : ℕ, m < k → ∃ a' b' c' : ℤ, 
      a' > b' ∧ b' ≥ c' ∧ a' - c' ≤ N - m) ∧
    (∃ a' b' c' : ℤ, a' = b' ∧ b' ≥ c' ∧ a' - c' ≤ N - k)) ∧
  (∀ k : ℕ, k > N → 
    ¬(∀ m : ℕ, m < k → ∃ a' b' c' : ℤ, 
      a' > b' ∧ b' ≥ c' ∧ a' - c' ≤ N - m)) :=
by sorry

end NUMINAMATH_CALUDE_max_squirrel_attacks_l3291_329198


namespace NUMINAMATH_CALUDE_domino_puzzle_l3291_329154

theorem domino_puzzle (visible_points : ℕ) (num_tiles : ℕ) (grid_size : ℕ) :
  visible_points = 37 →
  num_tiles = 8 →
  grid_size = 4 →
  ∃ (missing_points : ℕ),
    (visible_points + missing_points) % grid_size = 0 ∧
    missing_points ≤ 3 ∧
    ∀ (m : ℕ), m > missing_points →
      (visible_points + m) % grid_size ≠ 0 :=
by sorry


end NUMINAMATH_CALUDE_domino_puzzle_l3291_329154


namespace NUMINAMATH_CALUDE_chocolate_probability_theorem_not_always_between_probabilities_l3291_329166

structure ChocolateBox where
  white : ℕ
  total : ℕ
  h_total_pos : total > 0

def probability (box : ChocolateBox) : ℚ :=
  box.white / box.total

theorem chocolate_probability_theorem 
  (box1 box2 : ChocolateBox) :
  ∃ (combined : ChocolateBox),
    probability combined > min (probability box1) (probability box2) ∧
    probability combined < max (probability box1) (probability box2) ∧
    combined.white = box1.white + box2.white ∧
    combined.total = box1.total + box2.total :=
sorry

theorem not_always_between_probabilities 
  (box1 box2 : ChocolateBox) :
  ¬ ∀ (combined : ChocolateBox),
    (combined.white = box1.white + box2.white ∧
     combined.total = box1.total + box2.total) →
    (probability combined > min (probability box1) (probability box2) ∧
     probability combined < max (probability box1) (probability box2)) :=
sorry

end NUMINAMATH_CALUDE_chocolate_probability_theorem_not_always_between_probabilities_l3291_329166


namespace NUMINAMATH_CALUDE_paths_in_8x6_grid_l3291_329142

/-- The number of paths in a grid from bottom-left to top-right -/
def grid_paths (horizontal_steps : ℕ) (vertical_steps : ℕ) : ℕ :=
  Nat.choose (horizontal_steps + vertical_steps) vertical_steps

/-- Theorem: The number of paths in an 8x6 grid is 3003 -/
theorem paths_in_8x6_grid :
  grid_paths 8 6 = 3003 := by
  sorry

end NUMINAMATH_CALUDE_paths_in_8x6_grid_l3291_329142


namespace NUMINAMATH_CALUDE_joey_fraction_of_ethan_time_l3291_329191

def alexa_vacation_days : ℕ := 7 + 2  -- 1 week and 2 days

def joey_learning_days : ℕ := 6

def alexa_vacation_fraction : ℚ := 3/4

theorem joey_fraction_of_ethan_time : 
  (joey_learning_days : ℚ) / ((alexa_vacation_days : ℚ) / alexa_vacation_fraction) = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_joey_fraction_of_ethan_time_l3291_329191


namespace NUMINAMATH_CALUDE_root_power_sum_relation_l3291_329149

theorem root_power_sum_relation (t : ℕ → ℝ) (d e f : ℝ) : 
  (∃ (r₁ r₂ r₃ : ℝ), r₁^3 - 7*r₁^2 + 12*r₁ - 20 = 0 ∧ 
                      r₂^3 - 7*r₂^2 + 12*r₂ - 20 = 0 ∧ 
                      r₃^3 - 7*r₃^2 + 12*r₃ - 20 = 0 ∧ 
                      ∀ k, t k = r₁^k + r₂^k + r₃^k) →
  t 0 = 3 →
  t 1 = 7 →
  t 2 = 15 →
  (∀ k ≥ 2, t (k+1) = d * t k + e * t (k-1) + f * t (k-2) - 5) →
  d + e + f = 15 := by
sorry

end NUMINAMATH_CALUDE_root_power_sum_relation_l3291_329149


namespace NUMINAMATH_CALUDE_euler_family_mean_age_l3291_329153

def euler_family_ages : List ℕ := [6, 6, 6, 6, 12, 14, 14, 16]

theorem euler_family_mean_age : 
  (euler_family_ages.sum / euler_family_ages.length : ℚ) = 10 := by
  sorry

end NUMINAMATH_CALUDE_euler_family_mean_age_l3291_329153


namespace NUMINAMATH_CALUDE_polynomial_expansion_l3291_329109

theorem polynomial_expansion (x : ℝ) : 
  (3 * x^2 - 4 * x + 3) * (-4 * x^2 + 2 * x - 6) = 
  -12 * x^4 + 22 * x^3 - 38 * x^2 + 30 * x - 18 := by sorry

end NUMINAMATH_CALUDE_polynomial_expansion_l3291_329109


namespace NUMINAMATH_CALUDE_min_value_x_plus_y_l3291_329141

theorem min_value_x_plus_y (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : 1/x + 4/y = 1) :
  ∀ a b : ℝ, a > 0 → b > 0 → 1/a + 4/b = 1 → x + y ≤ a + b ∧ ∃ x y : ℝ, x > 0 ∧ y > 0 ∧ 1/x + 4/y = 1 ∧ x + y = 9 := by
  sorry

end NUMINAMATH_CALUDE_min_value_x_plus_y_l3291_329141


namespace NUMINAMATH_CALUDE_marcel_potatoes_l3291_329188

/-- Given the conditions of Marcel and Dale's grocery shopping, prove that Marcel bought 4 potatoes. -/
theorem marcel_potatoes :
  ∀ (marcel_corn dale_corn marcel_potatoes dale_potatoes total_vegetables : ℕ),
  marcel_corn = 10 →
  dale_corn = marcel_corn / 2 →
  dale_potatoes = 8 →
  total_vegetables = 27 →
  total_vegetables = marcel_corn + dale_corn + marcel_potatoes + dale_potatoes →
  marcel_potatoes = 4 :=
by
  sorry


end NUMINAMATH_CALUDE_marcel_potatoes_l3291_329188


namespace NUMINAMATH_CALUDE_trajectory_of_T_l3291_329193

-- Define the curve C
def C (x y : ℝ) : Prop := 4 * x^2 - y + 1 = 0

-- Define the fixed point M
def M : ℝ × ℝ := (-2, 0)

-- Define the relationship between A, T, and M
def AT_TM_relation (A T : ℝ × ℝ) : Prop :=
  let (xa, ya) := A
  let (xt, yt) := T
  (xa - xt, ya - yt) = (2 * (-2 - xt), 2 * (-yt))

-- Theorem statement
theorem trajectory_of_T (A T : ℝ × ℝ) :
  (∃ x y, A = (x, y) ∧ C x y) →  -- A is on curve C
  AT_TM_relation A T →           -- Relationship between A, T, and M holds
  4 * (3 * T.1 + 4)^2 - 3 * T.2 + 1 = 0 :=  -- Trajectory equation for T
by sorry

end NUMINAMATH_CALUDE_trajectory_of_T_l3291_329193


namespace NUMINAMATH_CALUDE_box_max_volume_l3291_329150

variable (a : ℝ) (x : ℝ)

-- Define the volume function
def V (a x : ℝ) : ℝ := (a - 2*x)^2 * x

-- State the theorem
theorem box_max_volume (h1 : a > 0) (h2 : 0 < x) (h3 : x < a/2) :
  ∃ (x_max : ℝ), x_max = a/6 ∧ 
  (∀ y, 0 < y → y < a/2 → V a y ≤ V a x_max) ∧
  V a x_max = 2*a^3/27 :=
sorry

end NUMINAMATH_CALUDE_box_max_volume_l3291_329150


namespace NUMINAMATH_CALUDE_unique_congruence_in_range_l3291_329177

theorem unique_congruence_in_range : ∃! n : ℕ, 3 ≤ n ∧ n ≤ 8 ∧ n % 8 = 123456 % 8 := by
  sorry

end NUMINAMATH_CALUDE_unique_congruence_in_range_l3291_329177


namespace NUMINAMATH_CALUDE_sequence_bounded_l3291_329152

/-- Given a sequence of positive real numbers satisfying a specific condition, prove that the sequence is bounded -/
theorem sequence_bounded (a : ℕ → ℝ) (h_pos : ∀ n, a n > 0) 
  (h_cond : ∀ k n m l, k + n = m + l → 
    (a k + a n) / (1 + a k * a n) = (a m + a l) / (1 + a m * a l)) :
  ∃ M, ∀ n, a n ≤ M :=
sorry

end NUMINAMATH_CALUDE_sequence_bounded_l3291_329152


namespace NUMINAMATH_CALUDE_ice_cream_volume_l3291_329163

/-- The volume of ice cream in a cone with a hemispherical top -/
theorem ice_cream_volume (h : ℝ) (r : ℝ) (h_pos : h > 0) (r_pos : r > 0) :
  let cone_volume := (1 / 3) * π * r^2 * h
  let hemisphere_volume := (2 / 3) * π * r^3
  h = 10 ∧ r = 3 → cone_volume + hemisphere_volume = 48 * π := by sorry

end NUMINAMATH_CALUDE_ice_cream_volume_l3291_329163

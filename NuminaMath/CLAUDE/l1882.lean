import Mathlib

namespace NUMINAMATH_CALUDE_journey_average_speed_l1882_188206

/-- Proves that the average speed of a two-segment journey is 54.4 miles per hour -/
theorem journey_average_speed :
  let distance1 : ℝ := 200  -- miles
  let time1 : ℝ := 4.5      -- hours
  let distance2 : ℝ := 480  -- miles
  let time2 : ℝ := 8        -- hours
  let total_distance := distance1 + distance2
  let total_time := time1 + time2
  let average_speed := total_distance / total_time
  average_speed = 54.4      -- miles per hour
:= by sorry

end NUMINAMATH_CALUDE_journey_average_speed_l1882_188206


namespace NUMINAMATH_CALUDE_line_equal_intercepts_l1882_188237

theorem line_equal_intercepts (a : ℝ) : 
  (∃ x y : ℝ, a * x + y - 2 - a = 0 ∧ 
   x = y ∧ 
   (x = 0 ∨ y = 0)) ↔ 
  (a = -2 ∨ a = 1) :=
sorry

end NUMINAMATH_CALUDE_line_equal_intercepts_l1882_188237


namespace NUMINAMATH_CALUDE_final_fish_count_l1882_188283

def fish_count (day : ℕ) : ℕ :=
  match day with
  | 0 => 10  -- Initial number of fish
  | 1 => 30  -- Day 1: 10 * 3
  | 2 => 90  -- Day 2: 30 * 3
  | 3 => 270 -- Day 3: 90 * 3
  | 4 => 162 -- Day 4: (270 * 3) - (270 * 3 * 2 / 5)
  | 5 => 486 -- Day 5: 162 * 3
  | 6 => 834 -- Day 6: (486 * 3) - (486 * 3 * 3 / 7)
  | 7 => 2502 -- Day 7: 834 * 3
  | 8 => 7531 -- Day 8: (2502 * 3) + 25
  | 9 => 22593 -- Day 9: 7531 * 3
  | 10 => 33890 -- Day 10: (22593 * 3) - (22593 * 3 / 2)
  | 11 => 101670 -- Day 11: 33890 * 3
  | _ => 305010 -- Day 12: 101670 * 3

theorem final_fish_count :
  fish_count 12 + (3 * fish_count 12 + 5) = 1220045 := by
  sorry

#eval fish_count 12 + (3 * fish_count 12 + 5)

end NUMINAMATH_CALUDE_final_fish_count_l1882_188283


namespace NUMINAMATH_CALUDE_cos_symmetry_l1882_188202

/-- The function f(x) = cos(2x + π/3) is symmetric about the line x = π/3 -/
theorem cos_symmetry (x : ℝ) : 
  let f : ℝ → ℝ := λ x ↦ Real.cos (2 * x + π / 3)
  ∀ y : ℝ, f (π / 3 + y) = f (π / 3 - y) := by
  sorry

end NUMINAMATH_CALUDE_cos_symmetry_l1882_188202


namespace NUMINAMATH_CALUDE_simplify_expression_1_expand_expression_2_l1882_188220

-- First expression
theorem simplify_expression_1 (a : ℝ) : (2 * a^2)^3 + (-3 * a^3)^2 = 17 * a^6 := by
  sorry

-- Second expression
theorem expand_expression_2 (x y : ℝ) : (x + 3*y) * (x - y) = x^2 + 2*x*y - 3*y^2 := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_1_expand_expression_2_l1882_188220


namespace NUMINAMATH_CALUDE_stating_count_numbers_with_five_or_six_in_base_eight_l1882_188243

/-- 
Given a positive integer n and a base b, returns the number of integers 
from 1 to n (inclusive) in base b that contain at least one digit d or e.
-/
def count_numbers_with_digits (n : ℕ) (b : ℕ) (d e : ℕ) : ℕ :=
  sorry

/-- 
Theorem stating that the number of integers from 1 to 512 (inclusive) 
in base 8 that contain at least one digit 5 or 6 is equal to 296.
-/
theorem count_numbers_with_five_or_six_in_base_eight : 
  count_numbers_with_digits 512 8 5 6 = 296 := by
  sorry

end NUMINAMATH_CALUDE_stating_count_numbers_with_five_or_six_in_base_eight_l1882_188243


namespace NUMINAMATH_CALUDE_book_cost_l1882_188241

/-- The cost of a book given partial payment and a condition on the remaining amount -/
theorem book_cost (paid : ℝ) (total_cost : ℝ) : 
  paid = 100 →
  (total_cost - paid) = (total_cost - (total_cost - paid)) →
  total_cost = 200 := by
  sorry

end NUMINAMATH_CALUDE_book_cost_l1882_188241


namespace NUMINAMATH_CALUDE_root_product_sum_l1882_188223

theorem root_product_sum (x₁ x₂ x₃ : ℝ) : 
  x₁ < x₂ ∧ x₂ < x₃ ∧ 
  (∀ x, Real.sqrt 2020 * x^3 - 4040 * x^2 + 4 = 0 ↔ x = x₁ ∨ x = x₂ ∨ x = x₃) →
  x₂ * (x₁ + x₃) = 2 := by
sorry

end NUMINAMATH_CALUDE_root_product_sum_l1882_188223


namespace NUMINAMATH_CALUDE_fraction_zero_implies_x_one_l1882_188266

theorem fraction_zero_implies_x_one :
  ∀ x : ℝ, (x - 1) / (x - 5) = 0 → x = 1 := by
  sorry

end NUMINAMATH_CALUDE_fraction_zero_implies_x_one_l1882_188266


namespace NUMINAMATH_CALUDE_circles_tangent_implies_a_value_l1882_188278

/-- Circle C1 with center (a, 0) and radius 2 -/
def C1 (a : ℝ) : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | (p.1 - a)^2 + p.2^2 = 4}

/-- Circle C2 with center (0, √5) and radius |a| -/
def C2 (a : ℝ) : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.1^2 + (p.2 - Real.sqrt 5)^2 = a^2}

/-- Two circles are externally tangent if the distance between their centers
    equals the sum of their radii -/
def externally_tangent (C1 C2 : Set (ℝ × ℝ)) : Prop :=
  ∃ (c1 c2 : ℝ × ℝ) (r1 r2 : ℝ),
    C1 = {p : ℝ × ℝ | (p.1 - c1.1)^2 + (p.2 - c1.2)^2 = r1^2} ∧
    C2 = {p : ℝ × ℝ | (p.1 - c2.1)^2 + (p.2 - c2.2)^2 = r2^2} ∧
    (c1.1 - c2.1)^2 + (c1.2 - c2.2)^2 = (r1 + r2)^2

theorem circles_tangent_implies_a_value :
  ∀ a : ℝ, externally_tangent (C1 a) (C2 a) → a = 1/4 ∨ a = -1/4 := by
  sorry

end NUMINAMATH_CALUDE_circles_tangent_implies_a_value_l1882_188278


namespace NUMINAMATH_CALUDE_trig_equation_solution_l1882_188246

theorem trig_equation_solution (t : ℝ) : 
  (2 * Real.cos (2 * t) + 5) * Real.cos t ^ 4 - (2 * Real.cos (2 * t) + 5) * Real.sin t ^ 4 = 3 ↔ 
  ∃ k : ℤ, t = π / 6 * (6 * ↑k + 1) ∨ t = π / 6 * (6 * ↑k - 1) :=
by sorry

end NUMINAMATH_CALUDE_trig_equation_solution_l1882_188246


namespace NUMINAMATH_CALUDE_binomial_expansion_example_l1882_188296

theorem binomial_expansion_example : 8^3 + 3*(8^2)*2 + 3*8*(2^2) + 2^3 = 1000 := by
  sorry

end NUMINAMATH_CALUDE_binomial_expansion_example_l1882_188296


namespace NUMINAMATH_CALUDE_sum_of_real_solutions_l1882_188215

theorem sum_of_real_solutions (b : ℝ) (h : b > 2) :
  ∃ y : ℝ, y ≥ 0 ∧ Real.sqrt (b - Real.sqrt (b + y)) = y ∧
  y = (Real.sqrt (4 * b - 3) - 1) / 2 := by
sorry

end NUMINAMATH_CALUDE_sum_of_real_solutions_l1882_188215


namespace NUMINAMATH_CALUDE_problem_solution_l1882_188228

theorem problem_solution (a b c : ℝ) : 
  |a - 1| + Real.sqrt (b + 2) + (c - 3)^2 = 0 → (a + b)^c = -1 := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l1882_188228


namespace NUMINAMATH_CALUDE_remaining_amount_after_buying_folders_l1882_188265

def initial_amount : ℕ := 19
def folder_cost : ℕ := 2

theorem remaining_amount_after_buying_folders :
  initial_amount - (initial_amount / folder_cost * folder_cost) = 1 := by
sorry

end NUMINAMATH_CALUDE_remaining_amount_after_buying_folders_l1882_188265


namespace NUMINAMATH_CALUDE_exists_inner_sum_greater_than_outer_sum_l1882_188210

/-- Represents a triangular pyramid (tetrahedron) --/
structure TriangularPyramid where
  base_edge_length : ℝ
  lateral_edge_length : ℝ

/-- Calculates the sum of edge lengths of a triangular pyramid --/
def sum_of_edges (pyramid : TriangularPyramid) : ℝ :=
  3 * pyramid.base_edge_length + 3 * pyramid.lateral_edge_length

/-- Represents two triangular pyramids with a common base, where one is inside the other --/
structure NestedPyramids where
  outer : TriangularPyramid
  inner : TriangularPyramid
  inner_inside_outer : inner.base_edge_length = outer.base_edge_length
  inner_lateral_edge_shorter : inner.lateral_edge_length < outer.lateral_edge_length

/-- Theorem: There exist nested pyramids where the sum of edges of the inner pyramid
    is greater than the sum of edges of the outer pyramid --/
theorem exists_inner_sum_greater_than_outer_sum :
  ∃ (np : NestedPyramids), sum_of_edges np.inner > sum_of_edges np.outer := by
  sorry


end NUMINAMATH_CALUDE_exists_inner_sum_greater_than_outer_sum_l1882_188210


namespace NUMINAMATH_CALUDE_exists_overlap_at_least_one_fifth_l1882_188239

/-- Represents a patch on the coat -/
structure Patch where
  area : ℝ
  area_nonneg : area ≥ 0

/-- Represents a coat with patches -/
structure Coat where
  total_area : ℝ
  patches : Finset Patch
  total_area_is_one : total_area = 1
  five_patches : patches.card = 5
  patch_area_at_least_half : ∀ p ∈ patches, p.area ≥ 1/2

/-- The theorem to be proved -/
theorem exists_overlap_at_least_one_fifth (coat : Coat) : 
  ∃ p1 p2 : Patch, p1 ∈ coat.patches ∧ p2 ∈ coat.patches ∧ p1 ≠ p2 ∧ 
    ∃ overlap_area : ℝ, overlap_area ≥ 1/5 ∧ 
      overlap_area ≤ min p1.area p2.area := by
  sorry

end NUMINAMATH_CALUDE_exists_overlap_at_least_one_fifth_l1882_188239


namespace NUMINAMATH_CALUDE_sine_of_angle_l1882_188268

theorem sine_of_angle (α : Real) (m : Real) (h1 : m ≠ 0) 
  (h2 : Real.sqrt 3 / Real.sqrt (3 + m^2) = m / 6) 
  (h3 : Real.cos α = m / 6) : 
  Real.sin α = Real.sqrt 3 / 2 := by
sorry

end NUMINAMATH_CALUDE_sine_of_angle_l1882_188268


namespace NUMINAMATH_CALUDE_symmetric_point_on_circle_l1882_188252

/-- The circle equation: x^2 + y^2 + 2x - 4y + 1 = 0 -/
def circle_equation (x y : ℝ) : Prop :=
  x^2 + y^2 + 2*x - 4*y + 1 = 0

/-- The line equation: x - ay + 2 = 0 -/
def line_equation (a x y : ℝ) : Prop :=
  x - a*y + 2 = 0

/-- The center of the circle -/
def circle_center : ℝ × ℝ := (-1, 2)

theorem symmetric_point_on_circle (a : ℝ) :
  line_equation a (circle_center.1) (circle_center.2) →
  a = -1/2 := by
  sorry

end NUMINAMATH_CALUDE_symmetric_point_on_circle_l1882_188252


namespace NUMINAMATH_CALUDE_angle_value_l1882_188271

-- Define the geometric sequence
def geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = r * a n

-- Define the theorem
theorem angle_value (a : ℕ → ℝ) (α : ℝ) :
  geometric_sequence a →
  (a 1 * a 1 - 2 * a 1 * Real.sin α - Real.sqrt 3 * Real.sin α = 0) →
  (a 8 * a 8 - 2 * a 8 * Real.sin α - Real.sqrt 3 * Real.sin α = 0) →
  ((a 1 + a 8) ^ 2 = 2 * a 3 * a 6 + 6) →
  (0 < α ∧ α < Real.pi / 2) →
  α = Real.pi / 3 := by
  sorry

end NUMINAMATH_CALUDE_angle_value_l1882_188271


namespace NUMINAMATH_CALUDE_f_2009_eq_zero_l1882_188230

/-- An even function on ℝ -/
def EvenFunction (f : ℝ → ℝ) : Prop :=
  ∀ x, f x = f (-x)

/-- An odd function on ℝ -/
def OddFunction (f : ℝ → ℝ) : Prop :=
  ∀ x, f x = -f (-x)

/-- Main theorem -/
theorem f_2009_eq_zero
  (f g : ℝ → ℝ)
  (h_even : EvenFunction f)
  (h_f_1 : f 1 = 0)
  (h_odd : OddFunction g)
  (h_g_def : ∀ x, g x = f (x - 1)) :
  f 2009 = 0 := by
sorry

end NUMINAMATH_CALUDE_f_2009_eq_zero_l1882_188230


namespace NUMINAMATH_CALUDE_compute_expression_l1882_188282

theorem compute_expression : 
  20 * (240 / 3 + 40 / 5 + 16 / 25 + 2) = 1772.8 := by
  sorry

end NUMINAMATH_CALUDE_compute_expression_l1882_188282


namespace NUMINAMATH_CALUDE_ball_selection_theorem_l1882_188253

def number_of_ways (total_red : ℕ) (total_white : ℕ) (balls_taken : ℕ) (min_score : ℕ) : ℕ :=
  let red_score := 2
  let white_score := 1
  (Finset.range (min total_red balls_taken + 1)).sum (fun red_taken =>
    let white_taken := balls_taken - red_taken
    if white_taken ≤ total_white ∧ red_taken * red_score + white_taken * white_score ≥ min_score
    then Nat.choose total_red red_taken * Nat.choose total_white white_taken
    else 0)

theorem ball_selection_theorem :
  number_of_ways 4 6 5 7 = 186 := by
  sorry

end NUMINAMATH_CALUDE_ball_selection_theorem_l1882_188253


namespace NUMINAMATH_CALUDE_IMO_2002_problem_l1882_188242

theorem IMO_2002_problem (A B C : ℕ) : 
  A > 0 → B > 0 → C > 0 → 
  A ≠ B → B ≠ C → A ≠ C →
  A * B * C = 2310 →
  (∀ X Y Z : ℕ, X > 0 → Y > 0 → Z > 0 → X ≠ Y → Y ≠ Z → X ≠ Z → X * Y * Z = 2310 → 
    A + B + C ≤ X + Y + Z) →
  (∀ X Y Z : ℕ, X > 0 → Y > 0 → Z > 0 → X ≠ Y → Y ≠ Z → X ≠ Z → X * Y * Z = 2310 → 
    A + B + C ≥ X + Y + Z) →
  A + B + C = 52 ∧ A + B + C = 390 :=
by sorry

end NUMINAMATH_CALUDE_IMO_2002_problem_l1882_188242


namespace NUMINAMATH_CALUDE_sphere_surface_area_with_cone_l1882_188256

/-- The surface area of a sphere containing a cone with base radius 1 and height √3 -/
theorem sphere_surface_area_with_cone (R : ℝ) : 
  (R : ℝ) > 0 → -- Radius is positive
  R^2 = (R - Real.sqrt 3)^2 + 1 → -- Cone geometry condition
  4 * π * R^2 = 16 * π / 3 := by
sorry

end NUMINAMATH_CALUDE_sphere_surface_area_with_cone_l1882_188256


namespace NUMINAMATH_CALUDE_bamboo_volume_proof_l1882_188236

/-- An arithmetic sequence of 9 terms -/
def ArithmeticSequence (a : Fin 9 → ℚ) : Prop :=
  ∃ d : ℚ, ∀ i j : Fin 9, a j - a i = (j - i : ℤ) • d

theorem bamboo_volume_proof (a : Fin 9 → ℚ) 
  (h_arith : ArithmeticSequence a)
  (h_bottom : a 0 + a 1 + a 2 = 4)
  (h_top : a 5 + a 6 + a 7 + a 8 = 3) :
  a 3 + a 4 = 2 + 3/22 := by
  sorry

end NUMINAMATH_CALUDE_bamboo_volume_proof_l1882_188236


namespace NUMINAMATH_CALUDE_angle_between_perpendicular_lines_to_dihedral_angle_l1882_188217

-- Define the dihedral angle
def dihedral_angle (α l β : Plane) : ℝ := sorry

-- Define perpendicularity between a line and a plane
def perpendicular (m : Line) (α : Plane) : Prop := sorry

-- Define the angle between two lines
def angle_between_lines (m n : Line) : ℝ := sorry

-- Define skew lines
def skew_lines (m n : Line) : Prop := sorry

theorem angle_between_perpendicular_lines_to_dihedral_angle 
  (α l β : Plane) (m n : Line) :
  dihedral_angle α l β = 60 ∧ 
  skew_lines m n ∧
  perpendicular m α ∧
  perpendicular n β →
  angle_between_lines m n = 60 := by sorry

end NUMINAMATH_CALUDE_angle_between_perpendicular_lines_to_dihedral_angle_l1882_188217


namespace NUMINAMATH_CALUDE_sphere_surface_area_l1882_188275

theorem sphere_surface_area (V : ℝ) (r : ℝ) (A : ℝ) : 
  V = 72 * Real.pi → 
  V = (4/3) * Real.pi * r^3 → 
  A = 4 * Real.pi * r^2 → 
  A = 36 * Real.pi * 2^(2/3) := by
sorry

end NUMINAMATH_CALUDE_sphere_surface_area_l1882_188275


namespace NUMINAMATH_CALUDE_f_even_and_increasing_l1882_188233

def f (x : ℝ) := x^2 + 1

theorem f_even_and_increasing : 
  (∀ x : ℝ, f x = f (-x)) ∧ 
  (∀ x y : ℝ, 0 < x → x < y → f x < f y) :=
sorry

end NUMINAMATH_CALUDE_f_even_and_increasing_l1882_188233


namespace NUMINAMATH_CALUDE_log_cutting_theorem_l1882_188257

/-- The number of pieces resulting from cutting a log -/
def num_pieces (n : ℕ) : ℕ := n + 1

/-- Theorem: Making 10 cuts on a log results in 11 pieces -/
theorem log_cutting_theorem : num_pieces 10 = 11 := by
  sorry

end NUMINAMATH_CALUDE_log_cutting_theorem_l1882_188257


namespace NUMINAMATH_CALUDE_alpha_necessary_not_sufficient_l1882_188292

-- Define the conditions α and β
def alpha (a b : ℝ) : Prop := b * (b - a) ≤ 0
def beta (a b : ℝ) : Prop := b ≠ 0 ∧ a / b ≥ 1

-- Theorem stating that α is a necessary but not sufficient condition for β
theorem alpha_necessary_not_sufficient :
  (∀ a b : ℝ, beta a b → alpha a b) ∧
  (∃ a b : ℝ, alpha a b ∧ ¬(beta a b)) :=
sorry

end NUMINAMATH_CALUDE_alpha_necessary_not_sufficient_l1882_188292


namespace NUMINAMATH_CALUDE_magnitude_a_plus_2b_l1882_188269

/-- Given two vectors a and b in ℝ², prove that |a + 2b| = √17 under certain conditions. -/
theorem magnitude_a_plus_2b (a b : ℝ × ℝ) 
  (h1 : ‖a‖ = 1)
  (h2 : ‖b‖ = 2)
  (h3 : a - b = (Real.sqrt 2, Real.sqrt 3)) :
  ‖a + 2 • b‖ = Real.sqrt 17 := by
  sorry

end NUMINAMATH_CALUDE_magnitude_a_plus_2b_l1882_188269


namespace NUMINAMATH_CALUDE_mean_height_is_70_625_l1882_188280

def heights : List ℝ := [58, 59, 60, 61, 64, 65, 68, 70, 73, 73, 75, 76, 77, 78, 78, 79]

theorem mean_height_is_70_625 :
  (heights.sum / heights.length : ℝ) = 70.625 := by
  sorry

end NUMINAMATH_CALUDE_mean_height_is_70_625_l1882_188280


namespace NUMINAMATH_CALUDE_point_quadrant_l1882_188255

/-- Given that point P(-4a, 2+b) is in the third quadrant, prove that point Q(a, b) is in the fourth quadrant -/
theorem point_quadrant (a b : ℝ) :
  (-4 * a < 0 ∧ 2 + b < 0) → (a > 0 ∧ b < 0) :=
by sorry

end NUMINAMATH_CALUDE_point_quadrant_l1882_188255


namespace NUMINAMATH_CALUDE_class_average_problem_l1882_188295

theorem class_average_problem (total_students : Nat) (high_scorers : Nat) (zero_scorers : Nat)
  (high_score : Nat) (class_average : Rat) :
  total_students = 27 →
  high_scorers = 5 →
  zero_scorers = 3 →
  high_score = 95 →
  class_average = 49.25925925925926 →
  let remaining_students := total_students - high_scorers - zero_scorers
  let total_score := class_average * total_students
  let high_scorers_total := high_scorers * high_score
  (total_score - high_scorers_total) / remaining_students = 45 := by
  sorry

end NUMINAMATH_CALUDE_class_average_problem_l1882_188295


namespace NUMINAMATH_CALUDE_xy_value_l1882_188232

theorem xy_value (x y : ℝ) (h : x^2 + y^2 + 4*x - 6*y + 13 = 0) : x^y = -8 := by
  sorry

end NUMINAMATH_CALUDE_xy_value_l1882_188232


namespace NUMINAMATH_CALUDE_track_length_is_630_l1882_188205

/-- The length of the circular track in meters -/
def track_length : ℝ := 630

/-- The angle between the starting positions of the two runners in degrees -/
def start_angle : ℝ := 120

/-- The distance run by the first runner (Tom) before the first meeting in meters -/
def first_meeting_distance : ℝ := 120

/-- The additional distance run by the second runner (Jerry) between the first and second meeting in meters -/
def second_meeting_distance : ℝ := 180

/-- Theorem stating that the given conditions imply the track length is 630 meters -/
theorem track_length_is_630 : 
  ∃ (speed_tom speed_jerry : ℝ), 
    speed_tom > 0 ∧ speed_jerry > 0 ∧
    first_meeting_distance / speed_tom = (track_length * start_angle / 360 - first_meeting_distance) / speed_jerry ∧
    (track_length - (track_length * start_angle / 360 - first_meeting_distance) - second_meeting_distance) / speed_tom = 
      (track_length * start_angle / 360 - first_meeting_distance + second_meeting_distance) / speed_jerry :=
by
  sorry


end NUMINAMATH_CALUDE_track_length_is_630_l1882_188205


namespace NUMINAMATH_CALUDE_point_on_transformed_plane_l1882_188279

/-- Represents a point in 3D space -/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- Represents a plane in 3D space -/
structure Plane where
  a : ℝ
  b : ℝ
  c : ℝ
  d : ℝ

/-- Applies a similarity transformation to a plane -/
def transformPlane (p : Plane) (k : ℝ) : Plane :=
  { a := p.a, b := p.b, c := p.c, d := k * p.d }

/-- Checks if a point lies on a plane -/
def pointOnPlane (pt : Point3D) (p : Plane) : Prop :=
  p.a * pt.x + p.b * pt.y + p.c * pt.z + p.d = 0

/-- The main theorem to prove -/
theorem point_on_transformed_plane :
  let A : Point3D := { x := -2, y := -1, z := 1 }
  let a : Plane := { a := 1, b := -2, c := 6, d := -10 }
  let k : ℝ := 3/5
  pointOnPlane A (transformPlane a k) := by sorry

end NUMINAMATH_CALUDE_point_on_transformed_plane_l1882_188279


namespace NUMINAMATH_CALUDE_infinite_series_sum_l1882_188213

/-- The sum of the infinite series Σ(n=1 to ∞) [2^(2n) / (1 + 2^n + 2^(2n) + 2^(3n) + 2^(3n+1))] is equal to 1/25. -/
theorem infinite_series_sum : 
  ∑' n : ℕ, (2^(2*n) : ℝ) / (1 + 2^n + 2^(2*n) + 2^(3*n) + 2^(3*n+1)) = 1/25 := by
  sorry

end NUMINAMATH_CALUDE_infinite_series_sum_l1882_188213


namespace NUMINAMATH_CALUDE_mechanic_work_hours_l1882_188234

theorem mechanic_work_hours (rate1 rate2 total_hours total_charge : ℕ) 
  (h1 : rate1 = 45)
  (h2 : rate2 = 85)
  (h3 : total_hours = 20)
  (h4 : total_charge = 1100) :
  ∃ (hours1 hours2 : ℕ), 
    hours1 + hours2 = total_hours ∧ 
    rate1 * hours1 + rate2 * hours2 = total_charge ∧
    hours2 = 5 := by
  sorry

end NUMINAMATH_CALUDE_mechanic_work_hours_l1882_188234


namespace NUMINAMATH_CALUDE_correct_statements_about_squares_l1882_188214

theorem correct_statements_about_squares :
  (∀ x : ℝ, x > 1 → x^2 > x) ∧
  (∀ x : ℝ, x < -1 → x^2 > x) :=
by sorry

end NUMINAMATH_CALUDE_correct_statements_about_squares_l1882_188214


namespace NUMINAMATH_CALUDE_triangle_problem_l1882_188286

theorem triangle_problem (a b c A B C : ℝ) : 
  0 < A ∧ A < π →
  0 < B ∧ B < π →
  0 < C ∧ C < π →
  a > 0 ∧ b > 0 ∧ c > 0 →
  a = b * Real.cos C + (Real.sqrt 3 / 3) * c * Real.sin B →
  b = 2 →
  (1/2) * a * c * Real.sin B = Real.sqrt 3 →
  B = π/3 ∧ a = 2 ∧ c = 2 := by sorry

end NUMINAMATH_CALUDE_triangle_problem_l1882_188286


namespace NUMINAMATH_CALUDE_katyas_age_l1882_188211

def insert_zero (n : ℕ) : ℕ :=
  (n / 10) * 100 + (n % 10)

theorem katyas_age :
  ∃! n : ℕ, n ≥ 10 ∧ n < 100 ∧ 6 * n = insert_zero n ∧ n = 18 :=
by sorry

end NUMINAMATH_CALUDE_katyas_age_l1882_188211


namespace NUMINAMATH_CALUDE_remaining_battery_life_is_eight_hours_l1882_188208

/-- Represents the battery life of a phone -/
structure PhoneBattery where
  inactiveLife : ℝ  -- Battery life when not in use (in hours)
  activeLife : ℝ    -- Battery life when used constantly (in hours)

/-- Calculates the remaining battery life -/
def remainingBatteryLife (battery : PhoneBattery) 
  (usedTime : ℝ)     -- Time the phone has been used (in hours)
  (totalTime : ℝ)    -- Total time since last charge (in hours)
  : ℝ :=
  sorry

/-- Theorem: Given the conditions, the remaining battery life is 8 hours -/
theorem remaining_battery_life_is_eight_hours 
  (battery : PhoneBattery)
  (h1 : battery.inactiveLife = 18)
  (h2 : battery.activeLife = 2)
  (h3 : remainingBatteryLife battery 0.5 6 = 8) :
  ∃ (t : ℝ), t = 8 ∧ remainingBatteryLife battery 0.5 6 = t := by
  sorry

end NUMINAMATH_CALUDE_remaining_battery_life_is_eight_hours_l1882_188208


namespace NUMINAMATH_CALUDE_repeating_decimal_sum_theorem_l1882_188204

/-- Represents a digit (0-9) -/
def Digit := Fin 10

/-- Converts a three-digit number represented by individual digits to an integer -/
def threeDigitToInt (a b c : Digit) : ℕ := 100 * a.val + 10 * b.val + c.val

/-- Converts a repeating decimal of the form 0.abab... to a rational number -/
def abRepeatingToRational (a b : Digit) : ℚ := (10 * a.val + b.val : ℚ) / 99

/-- Converts a repeating decimal of the form 0.abcabc... to a rational number -/
def abcRepeatingToRational (a b c : Digit) : ℚ := (100 * a.val + 10 * b.val + c.val : ℚ) / 999

theorem repeating_decimal_sum_theorem (a b c : Digit) :
  abRepeatingToRational a b + abcRepeatingToRational a b c = 17 / 37 →
  threeDigitToInt a b c = 270 := by
  sorry

end NUMINAMATH_CALUDE_repeating_decimal_sum_theorem_l1882_188204


namespace NUMINAMATH_CALUDE_triangle_area_l1882_188224

/-- The area of a triangle with vertices (5, -2), (10, 5), and (5, 5) is 17.5 square units. -/
theorem triangle_area : 
  let v1 : ℝ × ℝ := (5, -2)
  let v2 : ℝ × ℝ := (10, 5)
  let v3 : ℝ × ℝ := (5, 5)
  let area := (1/2) * abs ((v2.1 - v1.1) * (v3.2 - v1.2) - (v3.1 - v1.1) * (v2.2 - v1.2))
  area = 17.5 := by sorry

end NUMINAMATH_CALUDE_triangle_area_l1882_188224


namespace NUMINAMATH_CALUDE_sufficient_not_necessary_condition_l1882_188263

theorem sufficient_not_necessary_condition (x : ℝ) :
  (∀ x, x > 0 → x ≠ 0) ∧ (∃ x, x ≠ 0 ∧ ¬(x > 0)) := by
  sorry

end NUMINAMATH_CALUDE_sufficient_not_necessary_condition_l1882_188263


namespace NUMINAMATH_CALUDE_marbles_choice_count_l1882_188250

/-- The number of ways to choose marbles under specific conditions -/
def choose_marbles (total : ℕ) (red green blue : ℕ) (choose : ℕ) : ℕ :=
  let other := total - (red + green + blue)
  let color_pairs := (red * green + red * blue + green * blue)
  let remaining_choices := Nat.choose (other + red - 1 + green - 1 + blue - 1) (choose - 2)
  color_pairs * remaining_choices

/-- Theorem stating the number of ways to choose marbles under given conditions -/
theorem marbles_choice_count :
  choose_marbles 15 2 2 2 5 = 495 := by sorry

end NUMINAMATH_CALUDE_marbles_choice_count_l1882_188250


namespace NUMINAMATH_CALUDE_bacteria_growth_time_l1882_188287

theorem bacteria_growth_time (fill_time : ℕ) (initial_count : ℕ) : 
  (fill_time = 64 ∧ initial_count = 1) → 
  (∃ (new_fill_time : ℕ), new_fill_time = 62 ∧ 2^new_fill_time * initial_count * 4 = 2^fill_time) :=
by sorry

end NUMINAMATH_CALUDE_bacteria_growth_time_l1882_188287


namespace NUMINAMATH_CALUDE_geometric_sequence_ratio_sum_l1882_188247

theorem geometric_sequence_ratio_sum (k a₂ a₃ b₂ b₃ p r : ℝ) 
  (h1 : p ≠ 1)
  (h2 : r ≠ 1)
  (h3 : p ≠ r)
  (h4 : a₂ = k * p)
  (h5 : a₃ = k * p^2)
  (h6 : b₂ = k * r)
  (h7 : b₃ = k * r^2)
  (h8 : 3 * a₃ - 4 * b₃ = 5 * (3 * a₂ - 4 * b₂)) :
  p + r = 5 := by
sorry

end NUMINAMATH_CALUDE_geometric_sequence_ratio_sum_l1882_188247


namespace NUMINAMATH_CALUDE_divisibility_condition_l1882_188227

theorem divisibility_condition (m n : ℕ+) : (2*m^2 + n^2) ∣ (3*m*n + 3*m) ↔ (m = 1 ∧ n = 1) ∨ (m = 4 ∧ n = 2) ∨ (m = 4 ∧ n = 10) := by
  sorry

end NUMINAMATH_CALUDE_divisibility_condition_l1882_188227


namespace NUMINAMATH_CALUDE_hyperbola_condition_l1882_188294

/-- Defines the equation of a conic section -/
def is_hyperbola (m : ℝ) : Prop :=
  ∃ (x y : ℝ), x^2 / (m - 1) + y^2 / (m - 2) = 1 ∧ (m - 1) * (m - 2) < 0

/-- Theorem stating the necessary and sufficient condition for the equation to represent a hyperbola -/
theorem hyperbola_condition (m : ℝ) :
  is_hyperbola m ↔ 1 < m ∧ m < 2 :=
sorry

end NUMINAMATH_CALUDE_hyperbola_condition_l1882_188294


namespace NUMINAMATH_CALUDE_rectangle_area_l1882_188201

theorem rectangle_area (d : ℝ) (w : ℝ) (h : w > 0) : 
  d^2 = 10 * w^2 → 3 * w^2 = 3 * d^2 / 10 :=
by
  sorry

#check rectangle_area

end NUMINAMATH_CALUDE_rectangle_area_l1882_188201


namespace NUMINAMATH_CALUDE_triangle_inequality_l1882_188219

theorem triangle_inequality (a b m_a m_b : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : m_a > 0) (h4 : m_b > 0) (h5 : a > b) :
  a * m_a = b * m_b →
  a^2010 + m_a^2010 ≥ b^2010 + m_b^2010 := by
  sorry

end NUMINAMATH_CALUDE_triangle_inequality_l1882_188219


namespace NUMINAMATH_CALUDE_existence_of_m_l1882_188209

theorem existence_of_m (a b : ℝ) (h : a < b) :
  ∃ m : ℝ, m * a > m * b :=
sorry

end NUMINAMATH_CALUDE_existence_of_m_l1882_188209


namespace NUMINAMATH_CALUDE_second_month_sale_correct_l1882_188231

/-- Calculates the sale in the second month given the sales figures for other months and the average sale -/
def calculate_second_month_sale (first_month : ℕ) (third_month : ℕ) (fourth_month : ℕ) (fifth_month : ℕ) (average_sale : ℕ) : ℕ :=
  5 * average_sale - (first_month + third_month + fourth_month + fifth_month)

/-- Theorem stating that the calculated second month sale is correct -/
theorem second_month_sale_correct (first_month third_month fourth_month fifth_month average_sale : ℕ) :
  first_month = 5700 →
  third_month = 6855 →
  fourth_month = 3850 →
  fifth_month = 14045 →
  average_sale = 7800 →
  calculate_second_month_sale first_month third_month fourth_month fifth_month average_sale = 7550 :=
by
  sorry

#eval calculate_second_month_sale 5700 6855 3850 14045 7800

end NUMINAMATH_CALUDE_second_month_sale_correct_l1882_188231


namespace NUMINAMATH_CALUDE_range_of_a_perpendicular_case_l1882_188261

-- Define the line and hyperbola
def line (a : ℝ) (x : ℝ) : ℝ := a * x + 1
def hyperbola (x y : ℝ) : Prop := 3 * x^2 - y^2 = 1

-- Define the intersection condition
def intersects (a : ℝ) : Prop := ∃ x y, hyperbola x y ∧ y = line a x

-- Define the range of a
def valid_range (a : ℝ) : Prop := -Real.sqrt 6 < a ∧ a < Real.sqrt 6 ∧ a ≠ Real.sqrt 3 ∧ a ≠ -Real.sqrt 3

-- Define the perpendicularity condition
def perpendicular (a : ℝ) : Prop := ∃ x₁ y₁ x₂ y₂, 
  hyperbola x₁ y₁ ∧ y₁ = line a x₁ ∧
  hyperbola x₂ y₂ ∧ y₂ = line a x₂ ∧
  x₁ * x₂ + y₁ * y₂ = 0

-- Theorem 1: Range of a
theorem range_of_a : ∀ a : ℝ, intersects a ↔ valid_range a :=
sorry

-- Theorem 2: Perpendicular case
theorem perpendicular_case : ∀ a : ℝ, perpendicular a ↔ (a = 1 ∨ a = -1) :=
sorry

end NUMINAMATH_CALUDE_range_of_a_perpendicular_case_l1882_188261


namespace NUMINAMATH_CALUDE_five_cubic_yards_to_cubic_feet_l1882_188299

/-- Conversion factor from yards to feet -/
def yards_to_feet : ℝ := 3

/-- The number of cubic yards we want to convert -/
def cubic_yards : ℝ := 5

/-- The theorem states that 5 cubic yards is equal to 135 cubic feet -/
theorem five_cubic_yards_to_cubic_feet : 
  cubic_yards * (yards_to_feet ^ 3) = 135 := by
  sorry

end NUMINAMATH_CALUDE_five_cubic_yards_to_cubic_feet_l1882_188299


namespace NUMINAMATH_CALUDE_distance_to_x_axis_l1882_188207

theorem distance_to_x_axis (P : ℝ × ℝ) : P = (3, -2) → |P.2| = 2 := by
  sorry

end NUMINAMATH_CALUDE_distance_to_x_axis_l1882_188207


namespace NUMINAMATH_CALUDE_total_students_count_l1882_188277

/-- The number of students who wish to go on a scavenger hunting trip -/
def scavenger_hunting_students : ℕ := 4000

/-- The number of students who wish to go on a skiing trip -/
def skiing_students : ℕ := 2 * scavenger_hunting_students

/-- The total number of students -/
def total_students : ℕ := scavenger_hunting_students + skiing_students

theorem total_students_count : total_students = 12000 := by
  sorry

end NUMINAMATH_CALUDE_total_students_count_l1882_188277


namespace NUMINAMATH_CALUDE_computer_sticker_price_l1882_188270

theorem computer_sticker_price : 
  ∀ (x : ℝ), 
  (0.80 * x - 80 = 0.70 * x - 40 - 30) → 
  x = 700 := by
sorry

end NUMINAMATH_CALUDE_computer_sticker_price_l1882_188270


namespace NUMINAMATH_CALUDE_triangle_max_area_l1882_188238

open Real

theorem triangle_max_area (a b c : ℝ) (A B C : ℝ) :
  c = sqrt 2 →
  b = a * sin C + c * A →
  C = π / 4 →
  ∃ (S : ℝ), S ≤ (1 + sqrt 2) / 2 ∧
    ∀ (S' : ℝ), S' = 1 / 2 * a * b * sin C → S' ≤ S :=
by sorry

end NUMINAMATH_CALUDE_triangle_max_area_l1882_188238


namespace NUMINAMATH_CALUDE_cloth_cost_price_l1882_188229

/-- Given the following conditions for a cloth sale:
    - Total cloth length: 400 meters
    - Total selling price: Rs. 18,000
    - Loss per meter: Rs. 5
    Prove that the cost price for one meter of cloth is Rs. 50. -/
theorem cloth_cost_price 
  (total_length : ℝ) 
  (total_selling_price : ℝ) 
  (loss_per_meter : ℝ) 
  (h1 : total_length = 400)
  (h2 : total_selling_price = 18000)
  (h3 : loss_per_meter = 5) :
  (total_selling_price / total_length) + loss_per_meter = 50 := by
  sorry

end NUMINAMATH_CALUDE_cloth_cost_price_l1882_188229


namespace NUMINAMATH_CALUDE_even_function_range_l1882_188298

/-- A function f is even if f(-x) = f(x) for all x -/
def IsEven (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = f x

/-- A function f is increasing on (0, +∞) if f(x) ≤ f(y) for all 0 < x < y -/
def IncreasingOnPositive (f : ℝ → ℝ) : Prop :=
  ∀ x y, 0 < x → x < y → f x ≤ f y

theorem even_function_range (f : ℝ → ℝ) (a : ℝ) 
  (h_even : IsEven f)
  (h_incr : IncreasingOnPositive f)
  (h_cond : f a ≥ f 2) :
  a ∈ Set.Iic (-2) ∪ Set.Ici 2 := by
  sorry

end NUMINAMATH_CALUDE_even_function_range_l1882_188298


namespace NUMINAMATH_CALUDE_jay_and_paul_distance_l1882_188276

/-- The distance between two people walking in opposite directions -/
def distance_apart (jay_speed : ℚ) (paul_speed : ℚ) (time : ℚ) : ℚ :=
  jay_speed * time + paul_speed * time

/-- Theorem: Jay and Paul's distance apart after 2 hours -/
theorem jay_and_paul_distance : 
  let jay_speed : ℚ := 1 / 20 -- miles per minute
  let paul_speed : ℚ := 3 / 40 -- miles per minute
  let time : ℚ := 120 -- minutes (2 hours)
  distance_apart jay_speed paul_speed time = 15
  := by sorry

end NUMINAMATH_CALUDE_jay_and_paul_distance_l1882_188276


namespace NUMINAMATH_CALUDE_equilateral_triangle_max_area_l1882_188273

/-- The area of a triangle is maximum when it is equilateral, given a fixed perimeter -/
theorem equilateral_triangle_max_area 
  (a b c : ℝ) 
  (h_positive : a > 0 ∧ b > 0 ∧ c > 0) 
  (h_triangle : a + b > c ∧ b + c > a ∧ c + a > b) :
  let p := (a + b + c) / 2
  let S := Real.sqrt (p * (p - a) * (p - b) * (p - c))
  ∀ a' b' c' : ℝ, 
    a' > 0 → b' > 0 → c' > 0 →
    a' + b' + c' = a + b + c →
    let p' := (a' + b' + c') / 2
    let S' := Real.sqrt (p' * (p' - a') * (p' - b') * (p' - c'))
    S' ≤ S ∧ (S' = S → a' = b' ∧ b' = c') :=
by
  sorry


end NUMINAMATH_CALUDE_equilateral_triangle_max_area_l1882_188273


namespace NUMINAMATH_CALUDE_math_team_combinations_l1882_188249

/-- The number of ways to choose k elements from n elements -/
def choose (n k : ℕ) : ℕ := Nat.choose n k

/-- The number of girls in the math club -/
def num_girls : ℕ := 4

/-- The number of boys in the math club -/
def num_boys : ℕ := 6

/-- The number of girls to be chosen for the team -/
def girls_in_team : ℕ := 3

/-- The number of boys to be chosen for the team -/
def boys_in_team : ℕ := 2

theorem math_team_combinations :
  (choose num_girls girls_in_team) * (choose num_boys boys_in_team) = 60 := by
  sorry

end NUMINAMATH_CALUDE_math_team_combinations_l1882_188249


namespace NUMINAMATH_CALUDE_min_cost_all_B_trucks_l1882_188235

-- Define the capacities of trucks A and B
def truck_A_capacity : ℝ := 5
def truck_B_capacity : ℝ := 3

-- Define the cost per ton for trucks A and B
def cost_per_ton_A : ℝ := 100
def cost_per_ton_B : ℝ := 150

-- Define the total number of trucks
def total_trucks : ℕ := 5

-- Define the cost function
def cost_function (a : ℝ) : ℝ := 50 * a + 2250

-- Theorem statement
theorem min_cost_all_B_trucks :
  ∀ a : ℝ, 0 ≤ a ∧ a ≤ total_trucks →
  cost_function 0 ≤ cost_function a :=
by sorry

end NUMINAMATH_CALUDE_min_cost_all_B_trucks_l1882_188235


namespace NUMINAMATH_CALUDE_dice_probability_l1882_188262

/-- The number of sides on a standard die -/
def numSides : ℕ := 6

/-- The number of dice rolled -/
def numDice : ℕ := 5

/-- The number of dice re-rolled -/
def numReRolled : ℕ := 3

/-- The probability of a single re-rolled die matching the set-aside pair -/
def probSingleMatch : ℚ := 1 / numSides

/-- The probability of all re-rolled dice matching the set-aside pair -/
def probAllMatch : ℚ := probSingleMatch ^ numReRolled

theorem dice_probability : probAllMatch = 1 / 216 := by
  sorry

end NUMINAMATH_CALUDE_dice_probability_l1882_188262


namespace NUMINAMATH_CALUDE_small_tub_cost_l1882_188281

def total_cost : ℕ := 48
def num_large_tubs : ℕ := 3
def num_small_tubs : ℕ := 6
def cost_large_tub : ℕ := 6

theorem small_tub_cost : 
  ∃ (cost_small_tub : ℕ), 
    cost_small_tub * num_small_tubs + cost_large_tub * num_large_tubs = total_cost ∧
    cost_small_tub = 5 :=
by sorry

end NUMINAMATH_CALUDE_small_tub_cost_l1882_188281


namespace NUMINAMATH_CALUDE_complement_intersection_A_B_l1882_188267

def U : Set Nat := {1, 2, 3, 4, 5}
def A : Set Nat := {1, 3, 4}
def B : Set Nat := {3, 5}

theorem complement_intersection_A_B :
  (A ∩ B)ᶜ = {1, 2, 4, 5} :=
by sorry

end NUMINAMATH_CALUDE_complement_intersection_A_B_l1882_188267


namespace NUMINAMATH_CALUDE_circle_C_equation_and_OP_not_parallel_AB_l1882_188226

-- Define the circle M
def circle_M (r : ℝ) (x y : ℝ) : Prop := (x + 2)^2 + (y + 2)^2 = r^2

-- Define the line of symmetry
def symmetry_line (x y : ℝ) : Prop := x + y + 2 = 0

-- Define circle C
def circle_C (r : ℝ) (x y : ℝ) : Prop := (x - 2)^2 + (y - 2)^2 = r^2

-- Define point P
def point_P : ℝ × ℝ := (1, 1)

-- Define the origin O
def origin : ℝ × ℝ := (0, 0)

-- Define the slope of line OP
def slope_OP : ℝ := 1

-- Define the slope of line AB
def slope_AB : ℝ := 0

theorem circle_C_equation_and_OP_not_parallel_AB (r : ℝ) (h : r > 0) :
  (∀ x y : ℝ, circle_C r x y ↔ (x - 2)^2 + (y - 2)^2 = r^2) ∧ 
  slope_OP ≠ slope_AB :=
sorry

end NUMINAMATH_CALUDE_circle_C_equation_and_OP_not_parallel_AB_l1882_188226


namespace NUMINAMATH_CALUDE_school_play_seating_l1882_188245

theorem school_play_seating (rows : ℕ) (chairs_per_row : ℕ) (unoccupied : ℕ) : 
  rows = 40 → chairs_per_row = 20 → unoccupied = 10 → 
  rows * chairs_per_row - unoccupied = 790 := by
  sorry

end NUMINAMATH_CALUDE_school_play_seating_l1882_188245


namespace NUMINAMATH_CALUDE_f_lower_bound_f_condition_equivalent_l1882_188218

-- Define the function f
def f (x a : ℝ) : ℝ := |x - a^2| + |x + 2*a + 3|

-- Theorem 1: For all real x and a, f(x) ≥ 2
theorem f_lower_bound (x a : ℝ) : f x a ≥ 2 := by sorry

-- Theorem 2: f(-3/2) < 3 is equivalent to -1 < a < 0
theorem f_condition_equivalent (a : ℝ) : 
  (f (-3/2) a < 3) ↔ (-1 < a ∧ a < 0) := by sorry

end NUMINAMATH_CALUDE_f_lower_bound_f_condition_equivalent_l1882_188218


namespace NUMINAMATH_CALUDE_dice_sum_product_l1882_188254

theorem dice_sum_product (a b c d : ℕ) : 
  1 ≤ a ∧ a ≤ 6 → 
  1 ≤ b ∧ b ≤ 6 → 
  1 ≤ c ∧ c ≤ 6 → 
  1 ≤ d ∧ d ≤ 6 → 
  a * b * c * d = 216 → 
  a + b + c + d ≠ 19 := by
sorry

end NUMINAMATH_CALUDE_dice_sum_product_l1882_188254


namespace NUMINAMATH_CALUDE_largest_prime_factor_of_897_l1882_188216

theorem largest_prime_factor_of_897 : ∃ (p : ℕ), Prime p ∧ p ∣ 897 ∧ ∀ (q : ℕ), Prime q → q ∣ 897 → q ≤ p :=
by
  -- The proof goes here
  sorry

end NUMINAMATH_CALUDE_largest_prime_factor_of_897_l1882_188216


namespace NUMINAMATH_CALUDE_max_cables_cut_theorem_l1882_188244

/-- Represents a computer network with computers and cables -/
structure ComputerNetwork where
  num_computers : Nat
  num_cables : Nat
  num_clusters : Nat

/-- The initial state of the computer network -/
def initial_network : ComputerNetwork :=
  { num_computers := 200
  , num_cables := 345
  , num_clusters := 1 }

/-- The final state of the computer network after cable cutting -/
def final_network : ComputerNetwork :=
  { num_computers := 200
  , num_cables := initial_network.num_cables - 153
  , num_clusters := 8 }

/-- The maximum number of cables that can be cut -/
def max_cables_cut : Nat := 153

/-- Theorem stating the maximum number of cables that can be cut -/
theorem max_cables_cut_theorem :
  max_cables_cut = initial_network.num_cables - final_network.num_cables ∧
  final_network.num_clusters = 8 ∧
  final_network.num_cables ≥ final_network.num_computers - final_network.num_clusters :=
by sorry


end NUMINAMATH_CALUDE_max_cables_cut_theorem_l1882_188244


namespace NUMINAMATH_CALUDE_ratio_simplification_l1882_188291

theorem ratio_simplification : 
  (10^2001 + 10^2003) / (10^2002 + 10^2002) = 101 / 20 := by
  sorry

#eval Int.floor ((101 : ℚ) / 20)

end NUMINAMATH_CALUDE_ratio_simplification_l1882_188291


namespace NUMINAMATH_CALUDE_two_numbers_difference_l1882_188290

theorem two_numbers_difference (x y : ℝ) (h1 : x + y = 10) (h2 : x^2 - y^2 = 40) :
  |x - y| = 4 := by sorry

end NUMINAMATH_CALUDE_two_numbers_difference_l1882_188290


namespace NUMINAMATH_CALUDE_more_cars_difference_l1882_188258

/-- The number of cars Tommy has -/
def tommy_cars : ℕ := 3

/-- The number of cars Jessie has -/
def jessie_cars : ℕ := 3

/-- The total number of cars all three have -/
def total_cars : ℕ := 17

/-- The number of cars Jessie's older brother has -/
def brother_cars : ℕ := total_cars - (tommy_cars + jessie_cars)

theorem more_cars_difference : brother_cars - (tommy_cars + jessie_cars) = 5 := by
  sorry

end NUMINAMATH_CALUDE_more_cars_difference_l1882_188258


namespace NUMINAMATH_CALUDE_star_problem_l1882_188200

def star (x y : ℕ) : ℕ := x^2 + y

theorem star_problem : (3^(star 5 7)) ^ 2 + 4^(star 4 6) = 3^64 + 4^22 := by
  sorry

end NUMINAMATH_CALUDE_star_problem_l1882_188200


namespace NUMINAMATH_CALUDE_emily_phone_bill_l1882_188260

/-- Calculates the total cost of a cell phone plan based on usage. -/
def calculate_total_cost (base_cost : ℚ) (included_hours : ℚ) (text_cost : ℚ) 
  (extra_minute_cost : ℚ) (data_cost : ℚ) (texts_sent : ℚ) (hours_used : ℚ) 
  (data_used : ℚ) : ℚ :=
  base_cost + 
  (text_cost * texts_sent) + 
  (max (hours_used - included_hours) 0 * 60 * extra_minute_cost) + 
  (data_cost * data_used)

theorem emily_phone_bill :
  let base_cost : ℚ := 25
  let included_hours : ℚ := 25
  let text_cost : ℚ := 0.1
  let extra_minute_cost : ℚ := 0.15
  let data_cost : ℚ := 2
  let texts_sent : ℚ := 150
  let hours_used : ℚ := 26
  let data_used : ℚ := 3
  calculate_total_cost base_cost included_hours text_cost extra_minute_cost 
    data_cost texts_sent hours_used data_used = 55 := by
  sorry

end NUMINAMATH_CALUDE_emily_phone_bill_l1882_188260


namespace NUMINAMATH_CALUDE_arithmetic_sequence_property_l1882_188225

/-- An arithmetic sequence -/
def ArithmeticSequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

/-- The property that m + n = p + q implies a_m + a_n = a_p + a_q -/
def SufficientCondition (a : ℕ → ℝ) : Prop :=
  ∀ m n p q : ℕ, m + n = p + q → a m + a n = a p + a q

/-- The property that a_m + a_n = a_p + a_q does not always imply m + n = p + q -/
def NotNecessaryCondition (a : ℕ → ℝ) : Prop :=
  ∃ m n p q : ℕ, a m + a n = a p + a q ∧ m + n ≠ p + q

theorem arithmetic_sequence_property (a : ℕ → ℝ) :
  ArithmeticSequence a →
  SufficientCondition a ∧ NotNecessaryCondition a :=
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_property_l1882_188225


namespace NUMINAMATH_CALUDE_parabola_triangle_area_l1882_188288

/-- Given a parabola y² = 4x with focus F(1, 0) and directrix x = -1,
    and a line through F with slope √3 intersecting the parabola above
    the x-axis at point A, prove that the area of triangle AFK is 4√3,
    where K is the foot of the perpendicular from A to the directrix. -/
theorem parabola_triangle_area :
  let parabola : ℝ × ℝ → Prop := λ p => p.2^2 = 4 * p.1
  let F : ℝ × ℝ := (1, 0)
  let directrix : ℝ → Prop := λ x => x = -1
  let line : ℝ → ℝ := λ x => Real.sqrt 3 * (x - 1)
  let A : ℝ × ℝ := (3, 2 * Real.sqrt 3)
  let K : ℝ × ℝ := (-1, 2 * Real.sqrt 3)
  parabola A ∧
  (∀ x, line x = A.2 ↔ x = A.1) ∧
  directrix K.1 ∧
  (A.2 - K.2) / (A.1 - K.1) * (F.2 - A.2) / (F.1 - A.1) = -1 →
  (1/2) * abs (A.1 - F.1) * abs (A.2 - K.2) = 4 * Real.sqrt 3 := by
sorry

end NUMINAMATH_CALUDE_parabola_triangle_area_l1882_188288


namespace NUMINAMATH_CALUDE_probability_of_one_in_20_rows_l1882_188297

/-- The number of rows in Pascal's Triangle we're considering -/
def n : ℕ := 20

/-- The total number of elements in the first n rows of Pascal's Triangle -/
def total_elements (n : ℕ) : ℕ := n * (n + 1) / 2

/-- The number of 1's in the first n rows of Pascal's Triangle -/
def ones_count (n : ℕ) : ℕ := 2 * n - 1

/-- The probability of choosing a 1 from the first n rows of Pascal's Triangle -/
def probability_of_one (n : ℕ) : ℚ := ones_count n / total_elements n

theorem probability_of_one_in_20_rows : 
  probability_of_one n = 13 / 70 := by sorry

end NUMINAMATH_CALUDE_probability_of_one_in_20_rows_l1882_188297


namespace NUMINAMATH_CALUDE_max_value_of_a_l1882_188289

-- Define the sets A and B
def A (a : ℝ) : Set ℝ := {x | (x - 1) * (x - a) ≥ 0}
def B (a : ℝ) : Set ℝ := {x | x ≥ a - 1}

-- State the theorem
theorem max_value_of_a :
  ∀ a : ℝ, (A a ∪ B a = Set.univ) → (∀ b : ℝ, (A b ∪ B b = Set.univ) → b ≤ a) → a = 2 := by
  sorry

-- Note: Set.univ represents the entire real number line (ℝ)

end NUMINAMATH_CALUDE_max_value_of_a_l1882_188289


namespace NUMINAMATH_CALUDE_article_count_proof_l1882_188272

/-- The number of articles we are considering -/
def X : ℕ := 50

/-- The number of articles sold at selling price -/
def sold_articles : ℕ := 35

/-- The gain percentage -/
def gain_percentage : ℚ := 42857142857142854 / 100000000000000000

theorem article_count_proof :
  (∃ (C S : ℚ), C > 0 ∧ S > 0 ∧
    X * C = sold_articles * S ∧
    (S - C) / C = gain_percentage) →
  X = 50 :=
by sorry

end NUMINAMATH_CALUDE_article_count_proof_l1882_188272


namespace NUMINAMATH_CALUDE_car_trip_speed_l1882_188259

/-- Proves that given the conditions of the car trip, the speed for the remaining part is 20 mph -/
theorem car_trip_speed (D : ℝ) (h_D_pos : D > 0) : 
  let first_part := 0.8 * D
  let second_part := 0.2 * D
  let first_speed := 80
  let total_avg_speed := 50
  let v := (first_speed * total_avg_speed * second_part) / 
           (first_speed * D - total_avg_speed * first_part)
  v = 20 := by
  sorry

end NUMINAMATH_CALUDE_car_trip_speed_l1882_188259


namespace NUMINAMATH_CALUDE_function_value_implies_input_l1882_188222

/-- Given a function f(x) = (2x + 1) / (x - 1) and f(p) = 4, prove that p = 5/2 -/
theorem function_value_implies_input (f : ℝ → ℝ) (p : ℝ) 
  (h1 : ∀ x, x ≠ 1 → f x = (2 * x + 1) / (x - 1))
  (h2 : f p = 4) :
  p = 5/2 := by
sorry

end NUMINAMATH_CALUDE_function_value_implies_input_l1882_188222


namespace NUMINAMATH_CALUDE_diagonal_passes_through_600_cubes_l1882_188240

/-- The number of cubes an internal diagonal passes through in a rectangular solid -/
def cubes_passed_by_diagonal (a b c : ℕ) : ℕ :=
  a + b + c - (Nat.gcd a b + Nat.gcd b c + Nat.gcd c a) + Nat.gcd a (Nat.gcd b c)

/-- Theorem: For a 120 × 270 × 300 rectangular solid, the internal diagonal passes through 600 cubes -/
theorem diagonal_passes_through_600_cubes :
  cubes_passed_by_diagonal 120 270 300 = 600 := by
  sorry

end NUMINAMATH_CALUDE_diagonal_passes_through_600_cubes_l1882_188240


namespace NUMINAMATH_CALUDE_min_value_sum_squared_ratios_l1882_188248

theorem min_value_sum_squared_ratios (x y z : ℝ) 
  (hx : x > 0) (hy : y > 0) (hz : z > 0) (hsum : x + y + z = 9) :
  (x^2 + y^2) / (x + y) + (x^2 + z^2) / (x + z) + (y^2 + z^2) / (y + z) ≥ 9 := by
  sorry

end NUMINAMATH_CALUDE_min_value_sum_squared_ratios_l1882_188248


namespace NUMINAMATH_CALUDE_exists_expression_for_100_l1882_188293

/-- An arithmetic expression using only the number 3, parentheses, and basic arithmetic operations. -/
inductive Expr
  | three : Expr
  | add : Expr → Expr → Expr
  | sub : Expr → Expr → Expr
  | mul : Expr → Expr → Expr
  | div : Expr → Expr → Expr

/-- Evaluate an arithmetic expression. -/
def eval : Expr → ℚ
  | Expr.three => 3
  | Expr.add e1 e2 => eval e1 + eval e2
  | Expr.sub e1 e2 => eval e1 - eval e2
  | Expr.mul e1 e2 => eval e1 * eval e2
  | Expr.div e1 e2 => eval e1 / eval e2

/-- Count the number of threes in an expression. -/
def count_threes : Expr → ℕ
  | Expr.three => 1
  | Expr.add e1 e2 => count_threes e1 + count_threes e2
  | Expr.sub e1 e2 => count_threes e1 + count_threes e2
  | Expr.mul e1 e2 => count_threes e1 + count_threes e2
  | Expr.div e1 e2 => count_threes e1 + count_threes e2

/-- Theorem: There exists an arithmetic expression using fewer than ten threes that evaluates to 100. -/
theorem exists_expression_for_100 : ∃ e : Expr, eval e = 100 ∧ count_threes e < 10 := by
  sorry


end NUMINAMATH_CALUDE_exists_expression_for_100_l1882_188293


namespace NUMINAMATH_CALUDE_john_supermarket_spending_l1882_188212

def supermarket_spending (total : ℚ) : Prop :=
  let fruits_veg := (1 : ℚ) / 5 * total
  let meat := (1 : ℚ) / 3 * total
  let bakery := (1 : ℚ) / 10 * total
  let dairy := (1 : ℚ) / 6 * total
  let candy_magazine := total - (fruits_veg + meat + bakery + dairy)
  let magazine := (15 : ℚ) / 4  -- $3.75 as a rational number
  candy_magazine = (29 : ℚ) / 2 ∧  -- $14.50 as a rational number
  candy_magazine - magazine = (43 : ℚ) / 4  -- $10.75 as a rational number

theorem john_supermarket_spending :
  ∃ (total : ℚ), supermarket_spending total ∧ total = (145 : ℚ) / 2 :=
sorry

end NUMINAMATH_CALUDE_john_supermarket_spending_l1882_188212


namespace NUMINAMATH_CALUDE_count_even_multiples_of_three_squares_l1882_188203

theorem count_even_multiples_of_three_squares (n : Nat) : 
  (∃ k, k ∈ Finset.range n ∧ 36 * k * k < 3000) ↔ n = 10 :=
sorry

end NUMINAMATH_CALUDE_count_even_multiples_of_three_squares_l1882_188203


namespace NUMINAMATH_CALUDE_draw_with_min_black_balls_l1882_188251

def white_balls : ℕ := 6
def black_balls : ℕ := 4
def total_draw : ℕ := 4
def min_black : ℕ := 2

theorem draw_with_min_black_balls (white_balls black_balls total_draw min_black : ℕ) :
  (white_balls = 6) → (black_balls = 4) → (total_draw = 4) → (min_black = 2) →
  (Finset.sum (Finset.range (black_balls - min_black + 1))
    (λ i => Nat.choose black_balls (min_black + i) * Nat.choose white_balls (total_draw - (min_black + i)))) = 115 := by
  sorry

end NUMINAMATH_CALUDE_draw_with_min_black_balls_l1882_188251


namespace NUMINAMATH_CALUDE_nine_million_squared_zeros_l1882_188285

/-- For a positive integer n, represent a number composed of n nines -/
def all_nines (n : ℕ) : ℕ := 10^n - 1

/-- The number of zeros in the expansion of (all_nines n)² -/
def num_zeros (n : ℕ) : ℕ := n - 1

theorem nine_million_squared_zeros :
  ∃ (k : ℕ), all_nines 7 ^ 2 = k * 10^6 + m ∧ m < 10^6 :=
sorry

end NUMINAMATH_CALUDE_nine_million_squared_zeros_l1882_188285


namespace NUMINAMATH_CALUDE_count_divisible_integers_l1882_188264

theorem count_divisible_integers : 
  ∃! (S : Finset ℕ), 
    (∀ m ∈ S, m > 0 ∧ (2310 : ℤ) ∣ (m^2 - 2)) ∧ 
    (∀ m : ℕ, m > 0 ∧ (2310 : ℤ) ∣ (m^2 - 2) → m ∈ S) ∧
    S.card = 3 :=
by sorry

end NUMINAMATH_CALUDE_count_divisible_integers_l1882_188264


namespace NUMINAMATH_CALUDE_bhaskara_solution_l1882_188284

/-- The number of people in Bhaskara's money distribution problem -/
def bhaskara_problem (n : ℕ) : Prop :=
  let initial_sum := n * (2 * 3 + (n - 1) * 1) / 2
  let redistribution_sum := 100 * n
  initial_sum = redistribution_sum

theorem bhaskara_solution :
  ∃ n : ℕ, n > 0 ∧ bhaskara_problem n ∧ n = 195 := by
  sorry

end NUMINAMATH_CALUDE_bhaskara_solution_l1882_188284


namespace NUMINAMATH_CALUDE_parabola_properties_l1882_188221

/-- Parabola with vertex at origin and focus at (1,0) -/
structure Parabola where
  vertex : ℝ × ℝ := (0, 0)
  focus : ℝ × ℝ := (1, 0)

/-- Line passing through the focus of the parabola -/
structure Line (p : Parabola) where
  slope : ℝ

/-- Intersection points of the line with the parabola -/
def intersection_points (p : Parabola) (l : Line p) : Set (ℝ × ℝ) :=
  sorry

/-- Area of triangle formed by origin, focus, and two intersection points -/
def triangle_area (p : Parabola) (l : Line p) : ℝ :=
  sorry

theorem parabola_properties (p : Parabola) :
  (∀ x y : ℝ, (x, y) ∈ {(x, y) | y^2 = 4*x}) ∧
  (∃ min_area : ℝ, min_area = 2 ∧ 
    ∀ l : Line p, triangle_area p l ≥ min_area) :=
sorry

end NUMINAMATH_CALUDE_parabola_properties_l1882_188221


namespace NUMINAMATH_CALUDE_abc_inequality_l1882_188274

theorem abc_inequality (a b c : ℝ) (h1 : a > b) (h2 : b > c) (h3 : a + b + c = 0) :
  a * b > a * c := by
  sorry

end NUMINAMATH_CALUDE_abc_inequality_l1882_188274

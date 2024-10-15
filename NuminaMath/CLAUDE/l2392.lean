import Mathlib

namespace NUMINAMATH_CALUDE_intersection_A_complement_B_necessary_not_sufficient_condition_l2392_239203

-- Define sets A and B
def A : Set ℝ := {x | x^2 - 2*x - 8 ≤ 0}
def B (m : ℝ) : Set ℝ := {x | x^2 - 2*x + 1 - m^2 ≤ 0}

-- Theorem 1: Intersection of A and complement of B when m = 2
theorem intersection_A_complement_B :
  A ∩ (Set.univ \ B 2) = {x | -2 ≤ x ∧ x < -1 ∨ 3 < x ∧ x ≤ 4} := by sorry

-- Theorem 2: Necessary but not sufficient condition
theorem necessary_not_sufficient_condition :
  (∀ m > 0, (∀ x, x ∈ B m → x ∈ A) ∧ (∃ x, x ∈ A ∧ x ∉ B m)) ↔ 0 < m ∧ m ≤ 3 := by sorry

end NUMINAMATH_CALUDE_intersection_A_complement_B_necessary_not_sufficient_condition_l2392_239203


namespace NUMINAMATH_CALUDE_tangent_triangle_area_l2392_239205

/-- The area of the triangle formed by the tangent line at (1, e^(-1)) on y = e^(-x) and the axes is 2/e -/
theorem tangent_triangle_area :
  let f : ℝ → ℝ := fun x ↦ Real.exp (-x)
  let M : ℝ × ℝ := (1, Real.exp (-1))
  let tangent_line (x : ℝ) : ℝ := -Real.exp (-1) * (x - 1) + Real.exp (-1)
  let x_intercept : ℝ := 2
  let y_intercept : ℝ := 2 * Real.exp (-1)
  let triangle_area : ℝ := (1/2) * x_intercept * y_intercept
  triangle_area = 2 / Real.exp 1 :=
by sorry


end NUMINAMATH_CALUDE_tangent_triangle_area_l2392_239205


namespace NUMINAMATH_CALUDE_parabola_translation_l2392_239261

/-- A parabola in the Cartesian coordinate system. -/
structure Parabola where
  a : ℝ
  h : ℝ
  k : ℝ

/-- The equation of a parabola in the form y = a(x-h)^2 + k. -/
def parabola_equation (p : Parabola) (x : ℝ) : ℝ :=
  p.a * (x - p.h)^2 + p.k

/-- The translation of a parabola. -/
def translate (p : Parabola) (dx dy : ℝ) : Parabola :=
  { a := p.a, h := p.h + dx, k := p.k + dy }

theorem parabola_translation (p1 p2 : Parabola) :
  p1.a = 1/2 ∧ p1.h = 0 ∧ p1.k = -1 ∧
  p2.a = 1/2 ∧ p2.h = 4 ∧ p2.k = 2 →
  p2 = translate p1 4 3 :=
sorry

end NUMINAMATH_CALUDE_parabola_translation_l2392_239261


namespace NUMINAMATH_CALUDE_more_birch_than_fir_l2392_239254

/-- Represents a forest with fir and birch trees -/
structure Forest where
  fir_trees : ℕ
  birch_trees : ℕ

/-- A forest satisfies the Baron's condition if each fir tree has exactly 10 birch trees at 1 km distance -/
def satisfies_baron_condition (f : Forest) : Prop :=
  f.birch_trees = 10 * f.fir_trees

/-- Theorem: In a forest satisfying the Baron's condition, there are more birch trees than fir trees -/
theorem more_birch_than_fir (f : Forest) (h : satisfies_baron_condition f) : 
  f.birch_trees > f.fir_trees :=
sorry


end NUMINAMATH_CALUDE_more_birch_than_fir_l2392_239254


namespace NUMINAMATH_CALUDE_polynomial_evaluation_l2392_239272

def P (x : ℤ) : ℤ :=
  x^15 - 2008*x^14 + 2008*x^13 - 2008*x^12 + 2008*x^11 - 2008*x^10 + 2008*x^9 - 2008*x^8 + 2008*x^7 - 2008*x^6 + 2008*x^5 - 2008*x^4 + 2008*x^3 - 2008*x^2 + 2008*x

theorem polynomial_evaluation : P 2007 = 2007 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_evaluation_l2392_239272


namespace NUMINAMATH_CALUDE_work_days_calculation_l2392_239218

theorem work_days_calculation (total_days : ℕ) (work_pay : ℕ) (no_work_deduction : ℕ) (total_earnings : ℤ) :
  total_days = 30 ∧ 
  work_pay = 80 ∧ 
  no_work_deduction = 40 ∧ 
  total_earnings = 1600 →
  ∃ (days_not_worked : ℕ),
    days_not_worked = 20 ∧
    total_earnings = work_pay * (total_days - days_not_worked) - no_work_deduction * days_not_worked :=
by sorry


end NUMINAMATH_CALUDE_work_days_calculation_l2392_239218


namespace NUMINAMATH_CALUDE_secret_spread_theorem_l2392_239223

/-- The number of people who know the secret after n days -/
def secret_spread (n : ℕ) : ℕ :=
  (3^(n+1) - 1) / 2

/-- The day of the week given the number of days since Monday -/
def day_of_week (n : ℕ) : String :=
  match n % 7 with
  | 0 => "Monday"
  | 1 => "Tuesday"
  | 2 => "Wednesday"
  | 3 => "Thursday"
  | 4 => "Friday"
  | 5 => "Saturday"
  | _ => "Sunday"

theorem secret_spread_theorem :
  ∃ n : ℕ, secret_spread n ≥ 2186 ∧
           ∀ m : ℕ, m < n → secret_spread m < 2186 ∧
           day_of_week n = "Sunday" :=
by
  sorry

end NUMINAMATH_CALUDE_secret_spread_theorem_l2392_239223


namespace NUMINAMATH_CALUDE_binary_addition_subtraction_l2392_239234

/-- Represents a binary number as a list of bits (0 or 1) -/
def BinaryNumber := List Bool

/-- Converts a binary number to its decimal representation -/
def binaryToDecimal (b : BinaryNumber) : ℕ :=
  b.foldl (fun acc bit => 2 * acc + if bit then 1 else 0) 0

/-- The binary number 10110₂ -/
def b1 : BinaryNumber := [true, false, true, true, false]

/-- The binary number 1101₂ -/
def b2 : BinaryNumber := [true, true, false, true]

/-- The binary number 1010₂ -/
def b3 : BinaryNumber := [true, false, true, false]

/-- The binary number 1110₂ -/
def b4 : BinaryNumber := [true, true, true, false]

/-- The binary number 11111₂ (the expected result) -/
def result : BinaryNumber := [true, true, true, true, true]

theorem binary_addition_subtraction :
  binaryToDecimal b1 + binaryToDecimal b2 - binaryToDecimal b3 + binaryToDecimal b4 =
  binaryToDecimal result := by
  sorry

end NUMINAMATH_CALUDE_binary_addition_subtraction_l2392_239234


namespace NUMINAMATH_CALUDE_calculation_proof_no_solution_proof_l2392_239212

-- Part 1
theorem calculation_proof : Real.sqrt 3 ^ 2 - (2023 + Real.pi / 2) ^ 0 - (-1) ^ (-1 : Int) = 3 := by sorry

-- Part 2
theorem no_solution_proof :
  ¬∃ x : ℝ, (5 * x - 4 > 3 * x) ∧ ((2 * x - 1) / 3 < x / 2) := by sorry

end NUMINAMATH_CALUDE_calculation_proof_no_solution_proof_l2392_239212


namespace NUMINAMATH_CALUDE_find_divisor_l2392_239279

theorem find_divisor (dividend quotient remainder divisor : ℕ) 
  (h1 : dividend = 100)
  (h2 : quotient = 9)
  (h3 : remainder = 1)
  (h4 : dividend = divisor * quotient + remainder) :
  divisor = 11 := by
sorry

end NUMINAMATH_CALUDE_find_divisor_l2392_239279


namespace NUMINAMATH_CALUDE_integer_set_range_l2392_239238

theorem integer_set_range (a : ℝ) : 
  a ≤ 1 →
  (∃ (x y z : ℤ), x ≠ y ∧ y ≠ z ∧ x ≠ z ∧
    (↑x : ℝ) ∈ Set.Icc a (2 - a) ∧
    (↑y : ℝ) ∈ Set.Icc a (2 - a) ∧
    (↑z : ℝ) ∈ Set.Icc a (2 - a) ∧
    (∀ (w : ℤ), (↑w : ℝ) ∈ Set.Icc a (2 - a) → w = x ∨ w = y ∨ w = z)) →
  -1 < a ∧ a ≤ 0 :=
by sorry

end NUMINAMATH_CALUDE_integer_set_range_l2392_239238


namespace NUMINAMATH_CALUDE_lateralEdgeAngle_specific_pyramid_l2392_239260

/-- A regular truncated quadrangular pyramid -/
structure TruncatedPyramid where
  upperBaseSide : ℝ
  lowerBaseSide : ℝ
  height : ℝ
  lateralSurfaceArea : ℝ

/-- The angle between the lateral edge and the base plane of a truncated pyramid -/
def lateralEdgeAngle (p : TruncatedPyramid) : ℝ := sorry

/-- Theorem: The angle between the lateral edge and the base plane of a specific truncated pyramid -/
theorem lateralEdgeAngle_specific_pyramid :
  ∀ (p : TruncatedPyramid),
    p.lowerBaseSide = 5 * p.upperBaseSide →
    p.lateralSurfaceArea = p.height ^ 2 →
    lateralEdgeAngle p = Real.arctan (Real.sqrt (9 + 3 * Real.sqrt 10)) := by
  sorry

end NUMINAMATH_CALUDE_lateralEdgeAngle_specific_pyramid_l2392_239260


namespace NUMINAMATH_CALUDE_largest_common_remainder_l2392_239210

theorem largest_common_remainder :
  ∀ n : ℕ, 2013 ≤ n ∧ n ≤ 2156 →
  (∃ r : ℕ, n % 5 = r ∧ n % 11 = r ∧ n % 13 = r) →
  (∀ s : ℕ, (∃ m : ℕ, 2013 ≤ m ∧ m ≤ 2156 ∧ m % 5 = s ∧ m % 11 = s ∧ m % 13 = s) → s ≤ 4) :=
by sorry

#check largest_common_remainder

end NUMINAMATH_CALUDE_largest_common_remainder_l2392_239210


namespace NUMINAMATH_CALUDE_sphere_surface_area_with_inscribed_cube_l2392_239294

theorem sphere_surface_area_with_inscribed_cube (edge_length : ℝ) (radius : ℝ) : 
  edge_length = 2 → 
  radius^2 = 3 →
  4 * π * radius^2 = 12 * π := by
sorry

end NUMINAMATH_CALUDE_sphere_surface_area_with_inscribed_cube_l2392_239294


namespace NUMINAMATH_CALUDE_nested_fraction_equality_l2392_239274

theorem nested_fraction_equality : 
  2 + 1 / (2 + 1 / (2 + 1 / 2)) = 29 / 12 := by
  sorry

end NUMINAMATH_CALUDE_nested_fraction_equality_l2392_239274


namespace NUMINAMATH_CALUDE_mod_twelve_five_eleven_l2392_239293

theorem mod_twelve_five_eleven (m : ℕ) : 
  12^5 ≡ m [ZMOD 11] → 0 ≤ m → m < 11 → m = 1 := by
  sorry

end NUMINAMATH_CALUDE_mod_twelve_five_eleven_l2392_239293


namespace NUMINAMATH_CALUDE_lcm_problem_l2392_239263

theorem lcm_problem (a b : ℕ+) (h1 : b = 852) (h2 : Nat.lcm a b = 5964) : a = 852 := by
  sorry

end NUMINAMATH_CALUDE_lcm_problem_l2392_239263


namespace NUMINAMATH_CALUDE_f_max_min_on_interval_l2392_239228

noncomputable def f (x : ℝ) : ℝ := 2 * Real.sqrt 3 * Real.sin x * Real.cos x + 2 * (Real.cos x)^2 - 1

theorem f_max_min_on_interval :
  let a := 0
  let b := Real.pi / 2
  ∃ (x_max x_min : ℝ), x_max ∈ Set.Icc a b ∧ x_min ∈ Set.Icc a b ∧
    (∀ x ∈ Set.Icc a b, f x ≤ f x_max) ∧
    (∀ x ∈ Set.Icc a b, f x_min ≤ f x) ∧
    f x_max = 2 ∧ f x_min = -1 :=
sorry

end NUMINAMATH_CALUDE_f_max_min_on_interval_l2392_239228


namespace NUMINAMATH_CALUDE_sequence_first_term_l2392_239217

/-- Given a sequence {a_n} defined by a_n = (√2)^(n-2), prove that a_1 = √2/2 -/
theorem sequence_first_term (a : ℕ → ℝ) (h : ∀ n, a n = (Real.sqrt 2) ^ (n - 2)) :
  a 1 = Real.sqrt 2 / 2 := by
  sorry

end NUMINAMATH_CALUDE_sequence_first_term_l2392_239217


namespace NUMINAMATH_CALUDE_parallel_to_y_axis_second_quadrant_equal_distance_l2392_239233

-- Define point P
def P (a : ℝ) : ℝ × ℝ := (2*a - 2, a + 5)

-- Define point Q
def Q : ℝ × ℝ := (4, 5)

-- Part 1
theorem parallel_to_y_axis (a : ℝ) :
  (P a).1 = Q.1 → P a = (4, 8) := by sorry

-- Part 2
theorem second_quadrant_equal_distance (a : ℝ) :
  (P a).1 < 0 ∧ (P a).2 > 0 ∧ -(P a).1 = (P a).2 → a^2023 + a^(1/3) = -2 := by sorry

end NUMINAMATH_CALUDE_parallel_to_y_axis_second_quadrant_equal_distance_l2392_239233


namespace NUMINAMATH_CALUDE_rectangle_count_is_297_l2392_239278

/-- Represents a grid with a hole in the middle -/
structure Grid :=
  (size : ℕ)
  (hole_x : ℕ)
  (hole_y : ℕ)

/-- Counts the number of non-degenerate rectangles in a grid with a hole -/
def count_rectangles (g : Grid) : ℕ :=
  sorry

/-- The specific 7x7 grid with a hole at (4,4) -/
def specific_grid : Grid :=
  { size := 7, hole_x := 4, hole_y := 4 }

/-- Theorem stating that the number of non-degenerate rectangles in the specific grid is 297 -/
theorem rectangle_count_is_297 : count_rectangles specific_grid = 297 :=
  sorry

end NUMINAMATH_CALUDE_rectangle_count_is_297_l2392_239278


namespace NUMINAMATH_CALUDE_front_view_length_l2392_239237

/-- Given a line segment with length 5√2, side view 5, and top view √34, 
    its front view has length √41. -/
theorem front_view_length 
  (segment_length : ℝ) 
  (side_view : ℝ) 
  (top_view : ℝ) 
  (h1 : segment_length = 5 * Real.sqrt 2)
  (h2 : side_view = 5)
  (h3 : top_view = Real.sqrt 34) : 
  Real.sqrt (side_view^2 + top_view^2 + (Real.sqrt 41)^2) = segment_length :=
by sorry

end NUMINAMATH_CALUDE_front_view_length_l2392_239237


namespace NUMINAMATH_CALUDE_robotics_club_neither_cs_nor_electronics_l2392_239202

/-- The number of students in the robotics club who take neither computer science nor electronics -/
theorem robotics_club_neither_cs_nor_electronics :
  let total_students : ℕ := 60
  let cs_students : ℕ := 40
  let electronics_students : ℕ := 35
  let both_cs_and_electronics : ℕ := 25
  let neither_cs_nor_electronics : ℕ := total_students - (cs_students + electronics_students - both_cs_and_electronics)
  neither_cs_nor_electronics = 10 := by
sorry

end NUMINAMATH_CALUDE_robotics_club_neither_cs_nor_electronics_l2392_239202


namespace NUMINAMATH_CALUDE_sum_inequality_and_equality_condition_l2392_239216

theorem sum_inequality_and_equality_condition (a b c : ℝ) 
  (pos_a : a > 0) (pos_b : b > 0) (pos_c : c > 0)
  (h : (a + 1) * (b + 1) * (c + 1) = 8) : 
  a + b + c ≥ 3 ∧ (a + b + c = 3 ↔ a = 1 ∧ b = 1 ∧ c = 1) := by
  sorry

end NUMINAMATH_CALUDE_sum_inequality_and_equality_condition_l2392_239216


namespace NUMINAMATH_CALUDE_acid_dilution_l2392_239220

/-- Proves that adding 15 ounces of pure water to 30 ounces of a 30% acid solution yields a 20% acid solution. -/
theorem acid_dilution (initial_volume : ℝ) (initial_concentration : ℝ) (water_added : ℝ) (final_concentration : ℝ) :
  initial_volume = 30 →
  initial_concentration = 0.3 →
  water_added = 15 →
  final_concentration = 0.2 →
  (initial_volume * initial_concentration) / (initial_volume + water_added) = final_concentration :=
by sorry

end NUMINAMATH_CALUDE_acid_dilution_l2392_239220


namespace NUMINAMATH_CALUDE_square_area_from_perimeter_l2392_239239

/-- Theorem: The area of a square with perimeter 32 feet is 64 square feet. -/
theorem square_area_from_perimeter (perimeter : ℝ) (area : ℝ) : 
  perimeter = 32 → area = (perimeter / 4) ^ 2 → area = 64 := by
  sorry

end NUMINAMATH_CALUDE_square_area_from_perimeter_l2392_239239


namespace NUMINAMATH_CALUDE_angle_count_in_plane_l2392_239266

/-- Given n points in a plane, this theorem proves the number of 0° and 180° angles formed. -/
theorem angle_count_in_plane (n : ℕ) : 
  let zero_angles := n * (n - 1) * (n - 2) / 3
  let straight_angles := n * (n - 1) * (n - 2) / 6
  let total_angles := n * (n - 1) * (n - 2) / 2
  (zero_angles : ℚ) + (straight_angles : ℚ) = (total_angles : ℚ) :=
by sorry

/-- The total number of angles formed by n points in a plane. -/
def N (n : ℕ) : ℕ := n * (n - 1) * (n - 2) / 2

/-- The number of 0° angles formed by n points in a plane. -/
def zero_angles (n : ℕ) : ℕ := n * (n - 1) * (n - 2) / 3

/-- The number of 180° angles formed by n points in a plane. -/
def straight_angles (n : ℕ) : ℕ := n * (n - 1) * (n - 2) / 6

end NUMINAMATH_CALUDE_angle_count_in_plane_l2392_239266


namespace NUMINAMATH_CALUDE_max_intersections_three_circles_one_line_l2392_239206

/-- The maximum number of intersection points between three circles -/
def max_circle_intersections : ℕ := 6

/-- The maximum number of intersection points between three circles and a line -/
def max_circle_line_intersections : ℕ := 6

/-- The maximum number of intersection points between 3 different circles and 1 straight line -/
theorem max_intersections_three_circles_one_line :
  max_circle_intersections + max_circle_line_intersections = 12 := by sorry

end NUMINAMATH_CALUDE_max_intersections_three_circles_one_line_l2392_239206


namespace NUMINAMATH_CALUDE_probability_one_more_red_eq_three_eighths_l2392_239209

/-- Represents the color of a ball -/
inductive BallColor
  | Red
  | White

/-- Represents the outcome of three draws -/
def ThreeDraw := (BallColor × BallColor × BallColor)

/-- The set of all possible outcomes when drawing a ball three times with replacement -/
def allOutcomes : Finset ThreeDraw := sorry

/-- Predicate to check if an outcome has one more red ball than white balls -/
def hasOneMoreRed (draw : ThreeDraw) : Prop := sorry

/-- The set of favorable outcomes (one more red than white) -/
def favorableOutcomes : Finset ThreeDraw := sorry

/-- The probability of drawing the red ball one more time than the white ball -/
def probabilityOneMoreRed : ℚ := (favorableOutcomes.card : ℚ) / (allOutcomes.card : ℚ)

/-- Theorem: The probability of drawing the red ball one more time than the white ball is 3/8 -/
theorem probability_one_more_red_eq_three_eighths : 
  probabilityOneMoreRed = 3 / 8 := by sorry

end NUMINAMATH_CALUDE_probability_one_more_red_eq_three_eighths_l2392_239209


namespace NUMINAMATH_CALUDE_resultant_polyhedron_edges_l2392_239273

-- Define the convex polyhedron S
structure ConvexPolyhedron :=
  (vertices : ℕ)
  (edges : ℕ)

-- Define the operation of intersecting S with planes
def intersect_with_planes (S : ConvexPolyhedron) (num_planes : ℕ) : ℕ :=
  S.edges * 2 + S.edges

-- Theorem statement
theorem resultant_polyhedron_edges 
  (S : ConvexPolyhedron) 
  (h1 : S.vertices = S.vertices) 
  (h2 : S.edges = 150) :
  intersect_with_planes S S.vertices = 450 := by
  sorry

end NUMINAMATH_CALUDE_resultant_polyhedron_edges_l2392_239273


namespace NUMINAMATH_CALUDE_lock_problem_l2392_239241

def num_buttons : ℕ := 10
def buttons_to_press : ℕ := 3
def time_per_attempt : ℕ := 2

def total_combinations : ℕ := (num_buttons.choose buttons_to_press)

theorem lock_problem :
  let total_time : ℕ := total_combinations * time_per_attempt
  let avg_attempts : ℚ := (1 + total_combinations : ℚ) / 2
  let avg_time : ℚ := avg_attempts * time_per_attempt
  let max_attempts_in_minute : ℕ := 60 / time_per_attempt
  (total_time = 240) ∧
  (avg_time = 121) ∧
  (max_attempts_in_minute : ℚ) / total_combinations = 29 / 120 := by
  sorry

end NUMINAMATH_CALUDE_lock_problem_l2392_239241


namespace NUMINAMATH_CALUDE_equivalent_statements_l2392_239287

variable (P Q : Prop)

theorem equivalent_statements :
  ((P → Q) ↔ (¬Q → ¬P)) ∧ ((P → Q) ↔ (¬P ∨ Q)) :=
sorry

end NUMINAMATH_CALUDE_equivalent_statements_l2392_239287


namespace NUMINAMATH_CALUDE_prop_equivalence_l2392_239253

theorem prop_equivalence (p q : Prop) 
  (h1 : p ∨ q) 
  (h2 : ¬(p ∧ q)) : 
  p ↔ ¬q := by
  sorry

end NUMINAMATH_CALUDE_prop_equivalence_l2392_239253


namespace NUMINAMATH_CALUDE_red_jellybeans_count_l2392_239215

/-- Proves that the number of red jellybeans is 120 given the specified conditions -/
theorem red_jellybeans_count (total : ℕ) (blue : ℕ) (purple : ℕ) (orange : ℕ)
  (h_total : total = 200)
  (h_blue : blue = 14)
  (h_purple : purple = 26)
  (h_orange : orange = 40) :
  total - (blue + purple + orange) = 120 := by
  sorry

end NUMINAMATH_CALUDE_red_jellybeans_count_l2392_239215


namespace NUMINAMATH_CALUDE_inscribed_polyhedron_radius_gt_three_l2392_239219

/-- A polyhedron inscribed in a sphere -/
structure InscribedPolyhedron where
  radius : ℝ
  volume : ℝ
  surface_area : ℝ
  volume_eq_surface_area : volume = surface_area

/-- Theorem: For any polyhedron inscribed in a sphere, if its volume equals its surface area, then the radius of the sphere is greater than 3 -/
theorem inscribed_polyhedron_radius_gt_three (p : InscribedPolyhedron) : p.radius > 3 := by
  sorry

end NUMINAMATH_CALUDE_inscribed_polyhedron_radius_gt_three_l2392_239219


namespace NUMINAMATH_CALUDE_monomial_degree_l2392_239288

/-- Given that (a-2)x^2y^(|a|+1) is a monomial of degree 5 in x and y, and (a-2) ≠ 0, prove that a = -2 -/
theorem monomial_degree (a : ℤ) : 
  (∃ (x y : ℝ), (a - 2) * x^2 * y^(|a| + 1) ≠ 0) →  -- (a-2)x^2y^(|a|+1) is a monomial
  (2 + |a| + 1 = 5) →  -- The degree of the monomial in x and y is 5
  (a - 2 ≠ 0) →  -- (a-2) ≠ 0
  a = -2 := by
sorry


end NUMINAMATH_CALUDE_monomial_degree_l2392_239288


namespace NUMINAMATH_CALUDE_max_value_of_f_l2392_239246

noncomputable def f (x : ℝ) := 3 + Real.log x + 4 / Real.log x

theorem max_value_of_f :
  (∀ x : ℝ, 0 < x → x < 1 → f x ≤ -1) ∧
  (∃ x : ℝ, 0 < x ∧ x < 1 ∧ f x = -1) :=
sorry

end NUMINAMATH_CALUDE_max_value_of_f_l2392_239246


namespace NUMINAMATH_CALUDE_can_lids_per_box_l2392_239242

theorem can_lids_per_box (initial_lids : ℕ) (final_lids : ℕ) (num_boxes : ℕ) :
  initial_lids = 14 →
  final_lids = 53 →
  num_boxes = 3 →
  (final_lids - initial_lids) / num_boxes = 13 :=
by sorry

end NUMINAMATH_CALUDE_can_lids_per_box_l2392_239242


namespace NUMINAMATH_CALUDE_average_difference_l2392_239243

theorem average_difference (x : ℝ) : 
  (10 + 70 + x) / 3 = (20 + 40 + 60) / 3 - 7 → x = 19 := by
  sorry

end NUMINAMATH_CALUDE_average_difference_l2392_239243


namespace NUMINAMATH_CALUDE_zoo_feeding_theorem_l2392_239232

/-- Represents the number of bread and treats brought by each person -/
structure BreadAndTreats :=
  (bread : ℕ)
  (treats : ℕ)

/-- Calculates the total number of bread and treats -/
def totalItems (items : List BreadAndTreats) : ℕ :=
  (items.map (λ i => i.bread + i.treats)).sum

/-- Calculates the cost per pet -/
def costPerPet (totalBread totalTreats : ℕ) (x y : ℚ) (z : ℕ) : ℚ :=
  (totalBread * x + totalTreats * y) / z

theorem zoo_feeding_theorem 
  (jane_bread : ℕ) (jane_treats : ℕ)
  (wanda_bread : ℕ) (wanda_treats : ℕ)
  (carla_bread : ℕ) (carla_treats : ℕ)
  (peter_bread : ℕ) (peter_treats : ℕ)
  (x y : ℚ) (z : ℕ) :
  jane_bread = (75 * jane_treats) / 100 →
  wanda_treats = jane_treats / 2 →
  wanda_bread = 3 * wanda_treats →
  wanda_bread = 90 →
  carla_treats = (5 * carla_bread) / 2 →
  carla_bread = 40 →
  peter_bread = 2 * peter_treats →
  peter_bread + peter_treats = 140 →
  let items := [
    BreadAndTreats.mk jane_bread jane_treats,
    BreadAndTreats.mk wanda_bread wanda_treats,
    BreadAndTreats.mk carla_bread carla_treats,
    BreadAndTreats.mk peter_bread peter_treats
  ]
  totalItems items = 427 ∧
  costPerPet 235 192 x y z = (235 * x + 192 * y) / z :=
by sorry


end NUMINAMATH_CALUDE_zoo_feeding_theorem_l2392_239232


namespace NUMINAMATH_CALUDE_infinitely_many_primes_mod_3_eq_2_l2392_239250

theorem infinitely_many_primes_mod_3_eq_2 : 
  Set.Infinite {p : ℕ | Nat.Prime p ∧ p % 3 = 2} := by
  sorry

end NUMINAMATH_CALUDE_infinitely_many_primes_mod_3_eq_2_l2392_239250


namespace NUMINAMATH_CALUDE_part_i_part_ii_l2392_239247

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := |x - 1| + |x + a|

-- Define the function g
def g (a : ℝ) (x : ℝ) : ℝ := f a x - |3 + a|

-- Part I
theorem part_i : 
  {x : ℝ | f 3 x > 6} = {x : ℝ | x < -4 ∨ x > 2} :=
sorry

-- Part II
theorem part_ii :
  (∃ x : ℝ, g a x = 0) → a ≥ -2 :=
sorry

end NUMINAMATH_CALUDE_part_i_part_ii_l2392_239247


namespace NUMINAMATH_CALUDE_least_number_divisible_up_to_28_l2392_239230

def is_divisible_up_to (n : ℕ) (m : ℕ) : Prop :=
  ∀ k : ℕ, 1 ≤ k ∧ k ≤ m → n % k = 0

theorem least_number_divisible_up_to_28 :
  ∃ n : ℕ, n > 0 ∧ is_divisible_up_to n 28 ∧
  (∀ m : ℕ, 0 < m ∧ m < n → ¬is_divisible_up_to m 28) ∧
  n = 5348882400 := by
  sorry

end NUMINAMATH_CALUDE_least_number_divisible_up_to_28_l2392_239230


namespace NUMINAMATH_CALUDE_tower_painting_ways_level_painting_ways_l2392_239292

/-- Represents the number of ways to paint a single level of the tower -/
def paint_level : ℕ := 96

/-- Represents the number of colors available for painting -/
def num_colors : ℕ := 3

/-- Represents the number of levels in the tower -/
def num_levels : ℕ := 3

/-- Theorem stating the number of ways to paint the entire tower -/
theorem tower_painting_ways :
  num_colors * paint_level * paint_level = 27648 :=
by sorry

/-- Theorem stating the number of ways to paint a single level -/
theorem level_painting_ways :
  paint_level = 96 :=
by sorry

end NUMINAMATH_CALUDE_tower_painting_ways_level_painting_ways_l2392_239292


namespace NUMINAMATH_CALUDE_parallel_line_length_l2392_239296

/-- A triangle with a base of 20 inches and height of 10 inches, 
    divided into four equal areas by two parallel lines -/
structure DividedTriangle where
  base : ℝ
  height : ℝ
  baseParallel : ℝ
  base_eq : base = 20
  height_eq : height = 10
  equal_areas : baseParallel > 0 ∧ baseParallel < base

theorem parallel_line_length (t : DividedTriangle) : t.baseParallel = 10 := by
  sorry

end NUMINAMATH_CALUDE_parallel_line_length_l2392_239296


namespace NUMINAMATH_CALUDE_row_sum_is_odd_square_l2392_239251

/-- The sum of an arithmetic progression -/
def arithmetic_sum (a₁ : ℕ) (d : ℕ) (n : ℕ) : ℕ :=
  n * (2 * a₁ + (n - 1) * d) / 2

/-- The statement to be proved -/
theorem row_sum_is_odd_square (n : ℕ) (h : n > 0) :
  arithmetic_sum n 1 (2 * n - 1) = (2 * n - 1)^2 := by
  sorry

end NUMINAMATH_CALUDE_row_sum_is_odd_square_l2392_239251


namespace NUMINAMATH_CALUDE_greatest_four_digit_multiple_l2392_239280

theorem greatest_four_digit_multiple : ∃ n : ℕ, 
  (1000 ≤ n ∧ n < 10000) ∧ 
  (15 ∣ n) ∧ (25 ∣ n) ∧ (40 ∣ n) ∧ (75 ∣ n) ∧
  (∀ m : ℕ, (1000 ≤ m ∧ m < 10000) ∧ (15 ∣ m) ∧ (25 ∣ m) ∧ (40 ∣ m) ∧ (75 ∣ m) → m ≤ n) ∧
  n = 9600 :=
by sorry

end NUMINAMATH_CALUDE_greatest_four_digit_multiple_l2392_239280


namespace NUMINAMATH_CALUDE_barbara_candies_l2392_239211

/-- The number of candies Barbara has in total is 27, given her initial candies and additional purchase. -/
theorem barbara_candies : 
  ∀ (initial_candies additional_candies : ℕ),
    initial_candies = 9 →
    additional_candies = 18 →
    initial_candies + additional_candies = 27 :=
by
  sorry

end NUMINAMATH_CALUDE_barbara_candies_l2392_239211


namespace NUMINAMATH_CALUDE_complex_number_quadrant_l2392_239277

theorem complex_number_quadrant : ∃ (z : ℂ), z = (2 - I) / I ∧ z.re < 0 ∧ z.im > 0 := by
  sorry

end NUMINAMATH_CALUDE_complex_number_quadrant_l2392_239277


namespace NUMINAMATH_CALUDE_jimmy_lodging_cost_l2392_239286

/-- Calculates the total lodging cost for Jimmy's vacation --/
def total_lodging_cost (hostel_nights : ℕ) (hostel_cost_per_night : ℕ) 
  (cabin_nights : ℕ) (cabin_total_cost_per_night : ℕ) (cabin_friends : ℕ) : ℕ :=
  hostel_nights * hostel_cost_per_night + 
  cabin_nights * cabin_total_cost_per_night / (cabin_friends + 1)

/-- Theorem stating that Jimmy's total lodging cost is $75 --/
theorem jimmy_lodging_cost : 
  total_lodging_cost 3 15 2 45 2 = 75 := by
  sorry

end NUMINAMATH_CALUDE_jimmy_lodging_cost_l2392_239286


namespace NUMINAMATH_CALUDE_correct_statements_l2392_239269

theorem correct_statements :
  (∀ x : ℝ, x < 0 → x^3 < x) ∧
  (∀ x : ℝ, x^3 > 0 → x > 0) ∧
  (∀ x : ℝ, x > 1 → x^3 > x) :=
by sorry

end NUMINAMATH_CALUDE_correct_statements_l2392_239269


namespace NUMINAMATH_CALUDE_adjacent_supplementary_angles_l2392_239281

/-- Given two adjacent supplementary angles, if one is 60°, then the other is 120°. -/
theorem adjacent_supplementary_angles (angle1 angle2 : ℝ) : 
  angle1 = 60 → 
  angle1 + angle2 = 180 → 
  angle2 = 120 := by
sorry

end NUMINAMATH_CALUDE_adjacent_supplementary_angles_l2392_239281


namespace NUMINAMATH_CALUDE_squirrel_walnuts_l2392_239225

/-- The number of walnuts the boy squirrel effectively adds to the burrow -/
def boy_walnuts : ℕ := 5

/-- The number of walnuts the girl squirrel effectively adds to the burrow -/
def girl_walnuts : ℕ := 3

/-- The final number of walnuts in the burrow -/
def final_walnuts : ℕ := 20

/-- The initial number of walnuts in the burrow -/
def initial_walnuts : ℕ := 12

theorem squirrel_walnuts :
  initial_walnuts + boy_walnuts + girl_walnuts = final_walnuts :=
by
  sorry

end NUMINAMATH_CALUDE_squirrel_walnuts_l2392_239225


namespace NUMINAMATH_CALUDE_mountain_elevation_l2392_239240

/-- The relative elevation of a mountain given temperature information -/
theorem mountain_elevation (temp_decrease_rate : ℝ) (temp_summit temp_foot : ℝ) 
  (h1 : temp_decrease_rate = 0.7)
  (h2 : temp_summit = 14.1)
  (h3 : temp_foot = 26) :
  (temp_foot - temp_summit) / temp_decrease_rate * 100 = 1700 := by
  sorry

end NUMINAMATH_CALUDE_mountain_elevation_l2392_239240


namespace NUMINAMATH_CALUDE_second_train_length_l2392_239235

/-- The length of a train given crossing time and speeds -/
def train_length (l1 : ℝ) (v1 v2 : ℝ) (t : ℝ) : ℝ :=
  (v1 + v2) * t - l1

/-- Theorem: Given the conditions, the length of the second train is 210 meters -/
theorem second_train_length :
  let l1 : ℝ := 290  -- Length of first train in meters
  let v1 : ℝ := 120 * 1000 / 3600  -- Speed of first train in m/s
  let v2 : ℝ := 80 * 1000 / 3600   -- Speed of second train in m/s
  let t : ℝ := 9    -- Crossing time in seconds
  train_length l1 v1 v2 t = 210 := by
sorry


end NUMINAMATH_CALUDE_second_train_length_l2392_239235


namespace NUMINAMATH_CALUDE_slope_one_points_l2392_239256

theorem slope_one_points (a : ℝ) : 
  let A : ℝ × ℝ := (-a, 3)
  let B : ℝ × ℝ := (5, -a)
  let slope := (B.2 - A.2) / (B.1 - A.1)
  slope = 1 → a = -4 := by
sorry

end NUMINAMATH_CALUDE_slope_one_points_l2392_239256


namespace NUMINAMATH_CALUDE_solution_set_l2392_239282

-- Define the common logarithm (base 10)
noncomputable def lg (x : ℝ) : ℝ := Real.log x / Real.log 10

-- Define the equation
def equation (x : ℝ) : Prop := x^(lg x) = x^3 / 100

-- Theorem statement
theorem solution_set : 
  {x : ℝ | equation x} = {10, 100} :=
sorry

end NUMINAMATH_CALUDE_solution_set_l2392_239282


namespace NUMINAMATH_CALUDE_perp_planes_parallel_perp_plane_line_perp_l2392_239226

-- Define the types for lines and planes
variable (L : Type) [LinearOrderedField L]
variable (P : Type)

-- Define the relations
variable (parallel : L → L → Prop)
variable (perp : L → L → Prop)
variable (perp_plane : L → P → Prop)
variable (parallel_plane : P → P → Prop)
variable (contained : L → P → Prop)

-- Theorem 1
theorem perp_planes_parallel
  (m : L) (α β : P)
  (h1 : perp_plane m α)
  (h2 : perp_plane m β)
  : parallel_plane α β :=
sorry

-- Theorem 2
theorem perp_plane_line_perp
  (m n : L) (α : P)
  (h1 : perp_plane m α)
  (h2 : contained n α)
  : perp m n :=
sorry

end NUMINAMATH_CALUDE_perp_planes_parallel_perp_plane_line_perp_l2392_239226


namespace NUMINAMATH_CALUDE_intersection_of_A_and_B_l2392_239244

def A : Set ℝ := {x | x^2 - 2*x > 0}

def B : Set ℝ := {x | (x+1)/(x-1) ≤ 0}

theorem intersection_of_A_and_B : A ∩ B = {x : ℝ | -1 ≤ x ∧ x < 0} := by sorry

end NUMINAMATH_CALUDE_intersection_of_A_and_B_l2392_239244


namespace NUMINAMATH_CALUDE_complex_fraction_simplification_l2392_239289

theorem complex_fraction_simplification :
  (5 : ℂ) / (2 - Complex.I) = 2 + Complex.I := by sorry

end NUMINAMATH_CALUDE_complex_fraction_simplification_l2392_239289


namespace NUMINAMATH_CALUDE_square_sum_constant_l2392_239231

theorem square_sum_constant (x : ℝ) : (x + 2)^2 + 2*(x + 2)*(5 - x) + (5 - x)^2 = 49 := by
  sorry

end NUMINAMATH_CALUDE_square_sum_constant_l2392_239231


namespace NUMINAMATH_CALUDE_chosen_number_l2392_239262

theorem chosen_number (x : ℝ) : x / 5 - 154 = 6 → x = 800 := by
  sorry

end NUMINAMATH_CALUDE_chosen_number_l2392_239262


namespace NUMINAMATH_CALUDE_geometric_sequence_middle_term_l2392_239214

/-- Given a geometric sequence of real numbers 1, a₁, a₂, a₃, 4, prove that a₂ = 2 -/
theorem geometric_sequence_middle_term 
  (a₁ a₂ a₃ : ℝ) 
  (h : ∃ (r : ℝ), r ≠ 0 ∧ a₁ = r ∧ a₂ = r^2 ∧ a₃ = r^3 ∧ 4 = r^4) : 
  a₂ = 2 := by
sorry

end NUMINAMATH_CALUDE_geometric_sequence_middle_term_l2392_239214


namespace NUMINAMATH_CALUDE_inequality_implication_l2392_239295

theorem inequality_implication (a b : ℝ) (h : a > b) : -2 * a < -2 * b := by
  sorry

end NUMINAMATH_CALUDE_inequality_implication_l2392_239295


namespace NUMINAMATH_CALUDE_only_A_and_B_excellent_l2392_239298

/-- Represents a student's scores in three components -/
structure StudentScores where
  written : ℝ
  practical : ℝ
  growth : ℝ

/-- Calculates the total evaluation score for a student -/
def totalScore (s : StudentScores) : ℝ :=
  0.5 * s.written + 0.2 * s.practical + 0.3 * s.growth

/-- Determines if a score is excellent (over 90) -/
def isExcellent (score : ℝ) : Prop :=
  score > 90

/-- The scores of student A -/
def studentA : StudentScores :=
  { written := 90, practical := 83, growth := 95 }

/-- The scores of student B -/
def studentB : StudentScores :=
  { written := 98, practical := 90, growth := 95 }

/-- The scores of student C -/
def studentC : StudentScores :=
  { written := 80, practical := 88, growth := 90 }

/-- Theorem stating that only students A and B have excellent scores -/
theorem only_A_and_B_excellent :
  isExcellent (totalScore studentA) ∧
  isExcellent (totalScore studentB) ∧
  ¬isExcellent (totalScore studentC) := by
  sorry


end NUMINAMATH_CALUDE_only_A_and_B_excellent_l2392_239298


namespace NUMINAMATH_CALUDE_container_capacity_l2392_239227

theorem container_capacity (C : ℝ) (h : 0.35 * C + 48 = 0.75 * C) : C = 120 := by
  sorry

end NUMINAMATH_CALUDE_container_capacity_l2392_239227


namespace NUMINAMATH_CALUDE_inequality_solution_condition_l2392_239224

theorem inequality_solution_condition (a : ℝ) : 
  (∀ x : ℝ, (a + 1) * x > a + 1 ↔ x < 1) → a < -1 := by
  sorry

end NUMINAMATH_CALUDE_inequality_solution_condition_l2392_239224


namespace NUMINAMATH_CALUDE_history_not_statistics_l2392_239207

theorem history_not_statistics (total : ℕ) (history : ℕ) (statistics : ℕ) (history_or_statistics : ℕ) :
  total = 89 →
  history = 36 →
  statistics = 32 →
  history_or_statistics = 59 →
  history - (history + statistics - history_or_statistics) = 27 := by
  sorry

end NUMINAMATH_CALUDE_history_not_statistics_l2392_239207


namespace NUMINAMATH_CALUDE_square_area_ratio_l2392_239265

theorem square_area_ratio (a b : ℝ) (ha : a > 0) (hb : b > 0) 
  (h_perimeter : 4 * a = 4 * (4 * b)) : a^2 = 16 * b^2 := by
  sorry

end NUMINAMATH_CALUDE_square_area_ratio_l2392_239265


namespace NUMINAMATH_CALUDE_aquarium_water_volume_l2392_239297

/-- The initial volume of water in the aquarium -/
def initial_volume : ℝ := 36

/-- The volume of water after the cat spills half and Nancy triples the remainder -/
def final_volume : ℝ := 54

theorem aquarium_water_volume : 
  (3 * (initial_volume / 2)) = final_volume :=
by sorry

end NUMINAMATH_CALUDE_aquarium_water_volume_l2392_239297


namespace NUMINAMATH_CALUDE_jackson_pbj_sandwiches_l2392_239270

/-- Calculates the number of peanut butter and jelly sandwiches eaten in a school year --/
def pbj_sandwiches_eaten (weeks : ℕ) (wed_holidays : ℕ) (fri_holidays : ℕ) 
  (ham_cheese_interval : ℕ) (wed_missed : ℕ) (fri_missed : ℕ) : ℕ :=
  let total_wed := weeks
  let total_fri := weeks
  let wed_after_holidays := total_wed - wed_holidays
  let fri_after_holidays := total_fri - fri_holidays
  let wed_after_missed := wed_after_holidays - wed_missed
  let fri_after_missed := fri_after_holidays - fri_missed
  let ham_cheese_weeks := weeks / ham_cheese_interval
  let pbj_wed := wed_after_missed - ham_cheese_weeks
  let pbj_fri := fri_after_missed - (2 * ham_cheese_weeks)
  pbj_wed + pbj_fri

theorem jackson_pbj_sandwiches :
  pbj_sandwiches_eaten 36 2 3 4 1 2 = 37 := by
  sorry

#eval pbj_sandwiches_eaten 36 2 3 4 1 2

end NUMINAMATH_CALUDE_jackson_pbj_sandwiches_l2392_239270


namespace NUMINAMATH_CALUDE_inequality_solution_set_l2392_239258

theorem inequality_solution_set : 
  {x : ℝ | |x|^3 - 2*x^2 - 4*|x| + 3 < 0} = 
  {x : ℝ | -3 < x ∧ x < -1} ∪ {x : ℝ | 1 < x ∧ x < 3} := by
  sorry

end NUMINAMATH_CALUDE_inequality_solution_set_l2392_239258


namespace NUMINAMATH_CALUDE_snow_probability_l2392_239229

theorem snow_probability (p : ℝ) (h : p = 3 / 4) :
  1 - (1 - p)^4 = 255 / 256 := by
  sorry

end NUMINAMATH_CALUDE_snow_probability_l2392_239229


namespace NUMINAMATH_CALUDE_polynomial_with_geometric_zeros_l2392_239208

/-- A polynomial of the form x^4 + jx^2 + kx + 256 with four distinct real zeros in geometric progression has j = -32 -/
theorem polynomial_with_geometric_zeros (j k : ℝ) : 
  (∃ (a r : ℝ) (hr : r ≠ 1) (ha : a ≠ 0), 
    (∀ x : ℝ, x^4 + j*x^2 + k*x + 256 = (x - a*r^3) * (x - a*r^2) * (x - a*r) * (x - a)) ∧ 
    (a*r^3 ≠ a*r^2) ∧ (a*r^2 ≠ a*r) ∧ (a*r ≠ a)) → 
  j = -32 := by
sorry

end NUMINAMATH_CALUDE_polynomial_with_geometric_zeros_l2392_239208


namespace NUMINAMATH_CALUDE_subtracted_amount_l2392_239221

theorem subtracted_amount (x : ℝ) (h : x = 2.625) : 8 * x - 17 = 4 := by
  sorry

end NUMINAMATH_CALUDE_subtracted_amount_l2392_239221


namespace NUMINAMATH_CALUDE_carly_to_lisa_jeans_ratio_l2392_239264

/-- Represents the spending of a person on different items --/
structure Spending :=
  (tshirts : ℚ)
  (jeans : ℚ)
  (coats : ℚ)

/-- Calculate the total spending of a person --/
def totalSpending (s : Spending) : ℚ :=
  s.tshirts + s.jeans + s.coats

/-- Lisa's spending based on the given conditions --/
def lisa : Spending :=
  { tshirts := 40
  , jeans := 40 / 2
  , coats := 40 * 2 }

/-- Carly's spending based on the given conditions --/
def carly : Spending :=
  { tshirts := lisa.tshirts / 4
  , jeans := lisa.jeans * (230 - totalSpending lisa - (lisa.tshirts / 4) - (lisa.coats / 4)) / lisa.jeans
  , coats := lisa.coats / 4 }

/-- The main theorem to prove --/
theorem carly_to_lisa_jeans_ratio :
  carly.jeans / lisa.jeans = 3 := by sorry

end NUMINAMATH_CALUDE_carly_to_lisa_jeans_ratio_l2392_239264


namespace NUMINAMATH_CALUDE_problem_solution_l2392_239255

theorem problem_solution (k : ℝ) (h1 : k ≠ 0) :
  (∀ x : ℝ, (x^2 - k) * (x + k) = x^3 + k * (x^2 - x - 8)) → k = 8 := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l2392_239255


namespace NUMINAMATH_CALUDE_number_problem_l2392_239290

theorem number_problem (x : ℝ) (h : x - (3/4) * x = 100) : (1/4) * x = 100 := by
  sorry

end NUMINAMATH_CALUDE_number_problem_l2392_239290


namespace NUMINAMATH_CALUDE_geometric_series_sum_l2392_239271

theorem geometric_series_sum : ∑' i, (2/3:ℝ)^i = 2 := by sorry

end NUMINAMATH_CALUDE_geometric_series_sum_l2392_239271


namespace NUMINAMATH_CALUDE_product_of_three_numbers_l2392_239200

theorem product_of_three_numbers (x y z : ℝ) : 
  x > 0 → y > 0 → z > 0 → 
  x * y * z = 1 → 
  x + 1 / z = 8 → 
  y + 1 / x = 20 → 
  z + 1 / y = 10 / 53 := by
sorry

end NUMINAMATH_CALUDE_product_of_three_numbers_l2392_239200


namespace NUMINAMATH_CALUDE_parallel_planes_sufficient_not_necessary_l2392_239248

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the relations
variable (contains : Plane → Line → Prop)
variable (perpendicular : Line → Plane → Prop)
variable (parallel : Plane → Plane → Prop)
variable (perpendicularLines : Line → Line → Prop)

-- State the theorem
theorem parallel_planes_sufficient_not_necessary
  (a b : Line) (α β : Plane)
  (h1 : contains α a)
  (h2 : perpendicular b β) :
  (∀ a b α β, parallel α β → perpendicularLines a b) ∧
  (∃ a b α β, perpendicularLines a b ∧ ¬ parallel α β) :=
sorry

end NUMINAMATH_CALUDE_parallel_planes_sufficient_not_necessary_l2392_239248


namespace NUMINAMATH_CALUDE_consecutive_integers_with_prime_factors_l2392_239267

theorem consecutive_integers_with_prime_factors 
  (n s m : ℕ+) : 
  ∃ (x : ℕ), ∀ (j : ℕ), j ∈ Finset.range m → 
    (∃ (p : Finset ℕ), p.card = n ∧ 
      (∀ q ∈ p, Nat.Prime q ∧ 
        (∃ (k : ℕ), k ≥ s ∧ (q^k : ℕ) ∣ (x + j)))) :=
sorry

end NUMINAMATH_CALUDE_consecutive_integers_with_prime_factors_l2392_239267


namespace NUMINAMATH_CALUDE_decimal_to_fraction_l2392_239291

theorem decimal_to_fraction : (0.34 : ℚ) = 17 / 50 := by sorry

end NUMINAMATH_CALUDE_decimal_to_fraction_l2392_239291


namespace NUMINAMATH_CALUDE_spinner_area_l2392_239275

theorem spinner_area (r : ℝ) (p_win : ℝ) (p_bonus : ℝ) 
  (h_r : r = 15)
  (h_p_win : p_win = 1/3)
  (h_p_bonus : p_bonus = 1/6) :
  p_win * π * r^2 + p_bonus * π * r^2 = 112.5 * π := by
  sorry

end NUMINAMATH_CALUDE_spinner_area_l2392_239275


namespace NUMINAMATH_CALUDE_f_composition_inequality_l2392_239245

-- Define the piecewise function f
noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  if x ≥ 2 then x^2 + 2*a*x else 2^x + 1

-- State the theorem
theorem f_composition_inequality (a : ℝ) :
  (f a (f a 1) > 3 * a^2) ↔ (-1 < a ∧ a < 3) := by
  sorry

end NUMINAMATH_CALUDE_f_composition_inequality_l2392_239245


namespace NUMINAMATH_CALUDE_sin_2alpha_minus_pi_6_l2392_239204

theorem sin_2alpha_minus_pi_6 (α : Real) 
  (h : 2 * Real.sin α = 1 + 2 * Real.sqrt 3 * Real.cos α) : 
  Real.sin (2 * α - Real.pi / 6) = 7 / 8 := by
  sorry

end NUMINAMATH_CALUDE_sin_2alpha_minus_pi_6_l2392_239204


namespace NUMINAMATH_CALUDE_absolute_value_inequality_l2392_239236

theorem absolute_value_inequality (a : ℝ) :
  (∃ x : ℝ, |x + 1| + |x - 3| ≤ a) → a ≥ 4 := by
  sorry

end NUMINAMATH_CALUDE_absolute_value_inequality_l2392_239236


namespace NUMINAMATH_CALUDE_continuous_thin_stripe_probability_l2392_239257

-- Define the cube and its properties
def Cube := Fin 6

-- Define stripe properties
inductive StripeThickness
| thin
| thick

def StripeOrientation := Fin 4

structure Stripe :=
  (thickness : StripeThickness)
  (orientation : StripeOrientation)

def CubeConfiguration := Cube → Stripe

-- Define a function to check if a configuration has a continuous thin stripe
def hasContinuousThinStripe (config : CubeConfiguration) : Prop :=
  sorry -- Implementation details omitted

-- Define the probability space
def totalConfigurations : ℕ := 8^6

-- Define the number of favorable configurations
def favorableConfigurations : ℕ := 6144

-- Theorem statement
theorem continuous_thin_stripe_probability :
  (favorableConfigurations : ℚ) / totalConfigurations = 3 / 128 :=
sorry

end NUMINAMATH_CALUDE_continuous_thin_stripe_probability_l2392_239257


namespace NUMINAMATH_CALUDE_temp_increase_pressure_decrease_sea_water_heat_engine_possible_l2392_239268

-- Define an ideal gas
structure IdealGas where
  temperature : ℝ
  pressure : ℝ
  volume : ℝ
  particle_count : ℕ

-- Define the ideal gas law
axiom ideal_gas_law (gas : IdealGas) : gas.pressure * gas.volume = gas.particle_count * gas.temperature

-- Define average kinetic energy of molecules
def avg_kinetic_energy (gas : IdealGas) : ℝ := gas.temperature

-- Define a heat engine
structure HeatEngine where
  hot_reservoir : ℝ
  cold_reservoir : ℝ

-- Theorem 1: Temperature increase can lead to increased kinetic energy but decreased pressure
theorem temp_increase_pressure_decrease (gas1 gas2 : IdealGas) 
  (h_temp : gas2.temperature > gas1.temperature)
  (h_volume : gas2.volume = gas1.volume)
  (h_particles : gas2.particle_count = gas1.particle_count) :
  avg_kinetic_energy gas2 > avg_kinetic_energy gas1 ∧ 
  ∃ (p : ℝ), gas2.pressure = p ∧ p < gas1.pressure :=
sorry

-- Theorem 2: Heat engine using sea water temperature difference is theoretically possible
theorem sea_water_heat_engine_possible (shallow_temp deep_temp : ℝ) 
  (h_temp_diff : shallow_temp > deep_temp) :
  ∃ (engine : HeatEngine), engine.hot_reservoir = shallow_temp ∧ 
    engine.cold_reservoir = deep_temp ∧
    (∃ (work : ℝ), work > 0) :=
sorry

end NUMINAMATH_CALUDE_temp_increase_pressure_decrease_sea_water_heat_engine_possible_l2392_239268


namespace NUMINAMATH_CALUDE_kittens_sold_count_l2392_239283

/-- Represents the pet store's sales scenario -/
structure PetStoreSales where
  kitten_price : ℕ
  puppy_price : ℕ
  total_revenue : ℕ
  puppy_count : ℕ

/-- Calculates the number of kittens sold -/
def kittens_sold (s : PetStoreSales) : ℕ :=
  (s.total_revenue - s.puppy_price * s.puppy_count) / s.kitten_price

/-- Theorem stating the number of kittens sold -/
theorem kittens_sold_count (s : PetStoreSales) 
  (h1 : s.kitten_price = 6)
  (h2 : s.puppy_price = 5)
  (h3 : s.total_revenue = 17)
  (h4 : s.puppy_count = 1) :
  kittens_sold s = 2 := by
  sorry

end NUMINAMATH_CALUDE_kittens_sold_count_l2392_239283


namespace NUMINAMATH_CALUDE_quadratic_properties_l2392_239213

def quadratic_function (m : ℝ) (x : ℝ) : ℝ := m * x^2 - 4 * m * x + 3 * m

theorem quadratic_properties :
  ∀ m : ℝ,
  (∀ x : ℝ, quadratic_function m x = 0 ↔ x = 1 ∨ x = 3) ∧
  (m < 0 → 
    (∃ x₀ : ℝ, 1 ≤ x₀ ∧ x₀ ≤ 4 ∧ quadratic_function m x₀ = 2 ∧
      ∀ x : ℝ, 1 ≤ x ∧ x ≤ 4 → quadratic_function m x ≤ 2) →
    (∀ x : ℝ, 1 ≤ x ∧ x ≤ 4 → quadratic_function m x ≥ -6) ∧
    (∃ x₁ : ℝ, 1 ≤ x₁ ∧ x₁ ≤ 4 ∧ quadratic_function m x₁ = -6)) ∧
  (m ≤ -4/3 ∨ m ≥ 4/5 ↔
    ∃ x : ℝ, 2 ≤ x ∧ x ≤ 4 ∧ quadratic_function m x = (m + 4) / 2) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_properties_l2392_239213


namespace NUMINAMATH_CALUDE_inequality_of_means_l2392_239252

theorem inequality_of_means (a : ℝ) (h1 : a > 0) (h2 : a ≠ 1) :
  (a + a^2) / 2 > a^(3/2) ∧ a^(3/2) > 2 * a^2 / (1 + a) := by
  sorry

end NUMINAMATH_CALUDE_inequality_of_means_l2392_239252


namespace NUMINAMATH_CALUDE_gcd_lcm_product_90_150_l2392_239259

theorem gcd_lcm_product_90_150 : Nat.gcd 90 150 * Nat.lcm 90 150 = 13500 := by
  sorry

end NUMINAMATH_CALUDE_gcd_lcm_product_90_150_l2392_239259


namespace NUMINAMATH_CALUDE_school_sections_theorem_l2392_239276

/-- The number of sections formed when dividing students into equal groups -/
def number_of_sections (boys girls : Nat) : Nat :=
  (boys / Nat.gcd boys girls) + (girls / Nat.gcd boys girls)

/-- Theorem stating that the total number of sections for 408 boys and 216 girls is 26 -/
theorem school_sections_theorem :
  number_of_sections 408 216 = 26 := by
  sorry

end NUMINAMATH_CALUDE_school_sections_theorem_l2392_239276


namespace NUMINAMATH_CALUDE_taxi_charge_per_segment_l2392_239201

/-- Proves that the additional charge per 2/5 of a mile is $0.35 -/
theorem taxi_charge_per_segment (initial_fee : ℝ) (trip_distance : ℝ) (total_charge : ℝ) 
  (h1 : initial_fee = 2.35)
  (h2 : trip_distance = 3.6)
  (h3 : total_charge = 5.5) :
  (total_charge - initial_fee) / (trip_distance / (2/5)) = 0.35 := by
  sorry

end NUMINAMATH_CALUDE_taxi_charge_per_segment_l2392_239201


namespace NUMINAMATH_CALUDE_range_of_a_l2392_239285

-- Define the universal set U
def U : Set ℝ := {x : ℝ | 0 < x ∧ x < 9}

-- Define set A parameterized by a
def A (a : ℝ) : Set ℝ := {x : ℝ | 1 < x ∧ x < a}

-- Theorem statement
theorem range_of_a (a : ℝ) : 
  (∃ x, x ∈ A a) ∧ ¬(A a ⊆ U) ↔ 1 < a ∧ a ≤ 9 := by sorry

end NUMINAMATH_CALUDE_range_of_a_l2392_239285


namespace NUMINAMATH_CALUDE_chocolate_division_l2392_239284

theorem chocolate_division (total_chocolate : ℚ) (num_piles : ℕ) (friend_fraction : ℚ) :
  total_chocolate = 72 / 7 →
  num_piles = 8 →
  friend_fraction = 1 / 3 →
  friend_fraction * (total_chocolate / num_piles) = 3 / 7 := by
  sorry

end NUMINAMATH_CALUDE_chocolate_division_l2392_239284


namespace NUMINAMATH_CALUDE_toonie_is_two_dollar_coin_l2392_239299

/-- Represents the types of coins in Antonella's purse -/
inductive Coin
  | Loonie
  | Toonie

/-- The value of a coin in dollars -/
def coin_value (c : Coin) : ℕ :=
  match c with
  | Coin.Loonie => 1
  | Coin.Toonie => 2

/-- Antonella's coin situation -/
structure AntoniellaPurse where
  coins : List Coin
  initial_toonies : ℕ
  spent : ℕ
  remaining : ℕ

/-- The conditions of Antonella's coins -/
def antonellas_coins : AntoniellaPurse :=
  { coins := List.replicate 10 Coin.Toonie,  -- placeholder, actual distribution doesn't matter
    initial_toonies := 4,
    spent := 3,
    remaining := 11 }

theorem toonie_is_two_dollar_coin (purse : AntoniellaPurse := antonellas_coins) :
  ∃ (c : Coin), coin_value c = 2 ∧ c = Coin.Toonie :=
sorry

end NUMINAMATH_CALUDE_toonie_is_two_dollar_coin_l2392_239299


namespace NUMINAMATH_CALUDE_larger_number_proof_l2392_239222

theorem larger_number_proof (L S : ℕ) (h1 : L - S = 1395) (h2 : L = 6 * S + 15) : L = 1671 := by
  sorry

end NUMINAMATH_CALUDE_larger_number_proof_l2392_239222


namespace NUMINAMATH_CALUDE_fraction_difference_squared_l2392_239249

theorem fraction_difference_squared (a b : ℝ) (h : a ≠ 0 ∧ b ≠ 0) 
  (h1 : ∀ x y : ℝ, x ≠ 0 → y ≠ 0 → 1 / x - 1 / y = 1 / (x + y)) : 
  1 / a^2 - 1 / b^2 = 1 / (a * b) := by
  sorry

end NUMINAMATH_CALUDE_fraction_difference_squared_l2392_239249

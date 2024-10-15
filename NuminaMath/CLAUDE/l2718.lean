import Mathlib

namespace NUMINAMATH_CALUDE_parallel_line_existence_l2718_271871

-- Define the necessary structures
structure Point where
  x : ℝ
  y : ℝ

structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

-- Define parallelism
def parallel (l1 l2 : Line) : Prop :=
  l1.a * l2.b = l1.b * l2.a ∧ l1.a ≠ 0 ∧ l1.b ≠ 0 ∧ l2.a ≠ 0 ∧ l2.b ≠ 0

-- Define a line passing through a point
def passes_through (l : Line) (p : Point) : Prop :=
  l.a * p.x + l.b * p.y + l.c = 0

-- Theorem statement
theorem parallel_line_existence (A : Point) (l : Line) :
  ∃ (m : Line), passes_through m A ∧ parallel m l :=
sorry

end NUMINAMATH_CALUDE_parallel_line_existence_l2718_271871


namespace NUMINAMATH_CALUDE_smallest_value_of_fraction_sum_l2718_271848

theorem smallest_value_of_fraction_sum (a b : ℤ) (h : a > b) :
  (((a - b : ℚ) / (a + b)) + ((a + b : ℚ) / (a - b))) ≥ 2 ∧
  ∃ (a' b' : ℤ), a' > b' ∧ (((a' - b' : ℚ) / (a' + b')) + ((a' + b' : ℚ) / (a' - b'))) = 2 :=
by sorry

end NUMINAMATH_CALUDE_smallest_value_of_fraction_sum_l2718_271848


namespace NUMINAMATH_CALUDE_square_area_from_diagonal_l2718_271850

theorem square_area_from_diagonal (d : ℝ) (h : d = 12 * Real.sqrt 2) :
  let s := d / Real.sqrt 2
  s ^ 2 = 144 := by sorry

end NUMINAMATH_CALUDE_square_area_from_diagonal_l2718_271850


namespace NUMINAMATH_CALUDE_arithmetic_sequence_problem_l2718_271834

theorem arithmetic_sequence_problem (x : ℚ) : 
  let a₁ : ℚ := 1/3
  let a₂ : ℚ := x - 2
  let a₃ : ℚ := 4*x
  (a₂ - a₁ = a₃ - a₂) → x = -13/6 := by
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_problem_l2718_271834


namespace NUMINAMATH_CALUDE_cube_with_holes_surface_area_l2718_271882

/-- Calculates the total surface area of a cube with holes --/
def total_surface_area (cube_edge : ℝ) (hole_side : ℝ) : ℝ :=
  let original_surface_area := 6 * cube_edge^2
  let hole_area := 6 * hole_side^2
  let exposed_area := 6 * 4 * hole_side^2
  original_surface_area - hole_area + exposed_area

/-- Theorem: The total surface area of a cube with edge length 4 meters and square holes
    of side 2 meters cut through each face is 168 square meters --/
theorem cube_with_holes_surface_area :
  total_surface_area 4 2 = 168 := by
  sorry

end NUMINAMATH_CALUDE_cube_with_holes_surface_area_l2718_271882


namespace NUMINAMATH_CALUDE_largest_integer_less_than_150_over_7_l2718_271821

theorem largest_integer_less_than_150_over_7 : 
  (∀ n : ℤ, 7 * n < 150 → n ≤ 21) ∧ (7 * 21 < 150) := by sorry

end NUMINAMATH_CALUDE_largest_integer_less_than_150_over_7_l2718_271821


namespace NUMINAMATH_CALUDE_songs_added_l2718_271832

theorem songs_added (initial_songs deleted_songs final_songs : ℕ) : 
  initial_songs = 8 → deleted_songs = 5 → final_songs = 33 →
  final_songs - (initial_songs - deleted_songs) = 30 :=
by sorry

end NUMINAMATH_CALUDE_songs_added_l2718_271832


namespace NUMINAMATH_CALUDE_intersection_of_M_and_N_l2718_271873

open Set

def M : Set ℝ := {x | -1 < x ∧ x < 3}
def N : Set ℝ := {x | x < 1}

theorem intersection_of_M_and_N : M ∩ N = Ioo (-1) 1 := by
  sorry

end NUMINAMATH_CALUDE_intersection_of_M_and_N_l2718_271873


namespace NUMINAMATH_CALUDE_smallest_n_congruence_three_satisfies_congruence_three_is_smallest_l2718_271869

theorem smallest_n_congruence (n : ℕ) : n > 0 ∧ 23 * n ≡ 789 [MOD 8] → n ≥ 3 :=
by sorry

theorem three_satisfies_congruence : 23 * 3 ≡ 789 [MOD 8] :=
by sorry

theorem three_is_smallest (m : ℕ) : m > 0 ∧ 23 * m ≡ 789 [MOD 8] → m ≥ 3 :=
by sorry

end NUMINAMATH_CALUDE_smallest_n_congruence_three_satisfies_congruence_three_is_smallest_l2718_271869


namespace NUMINAMATH_CALUDE_add_inequality_preserves_order_l2718_271875

theorem add_inequality_preserves_order (a b c d : ℝ) 
  (h1 : a > b) (h2 : c > d) : a + c > b + d := by sorry

end NUMINAMATH_CALUDE_add_inequality_preserves_order_l2718_271875


namespace NUMINAMATH_CALUDE_string_length_around_cylinder_specific_string_length_l2718_271866

/-- 
Given a cylindrical post with circumference C, height H, and a string making n complete loops 
around it from bottom to top, the length of the string L is given by L = n * √(C² + (H/n)²)
-/
theorem string_length_around_cylinder (C H : ℝ) (n : ℕ) (h1 : C > 0) (h2 : H > 0) (h3 : n > 0) :
  let L := n * Real.sqrt (C^2 + (H/n)^2)
  L = n * Real.sqrt (C^2 + (H/n)^2) := by sorry

/-- 
For the specific case where C = 6, H = 18, and n = 3, prove that the string length is 18√2
-/
theorem specific_string_length :
  let C : ℝ := 6
  let H : ℝ := 18
  let n : ℕ := 3
  let L := n * Real.sqrt (C^2 + (H/n)^2)
  L = 18 * Real.sqrt 2 := by sorry

end NUMINAMATH_CALUDE_string_length_around_cylinder_specific_string_length_l2718_271866


namespace NUMINAMATH_CALUDE_jame_card_tearing_l2718_271879

/-- The number of cards in a new deck -/
def cards_per_deck : ℕ := 55

/-- The number of times Jame tears cards per week -/
def tear_times_per_week : ℕ := 3

/-- The number of decks Jame buys -/
def decks_bought : ℕ := 18

/-- The number of weeks Jame can go with the bought decks -/
def weeks_lasted : ℕ := 11

/-- The number of cards Jame can tear at a time -/
def cards_torn_at_once : ℕ := decks_bought * cards_per_deck / (weeks_lasted * tear_times_per_week)

theorem jame_card_tearing :
  cards_torn_at_once = 30 :=
sorry

end NUMINAMATH_CALUDE_jame_card_tearing_l2718_271879


namespace NUMINAMATH_CALUDE_family_ages_l2718_271835

theorem family_ages :
  ∀ (dad mom kolya tanya : ℕ),
    dad = mom + 4 →
    kolya = tanya + 4 →
    2 * kolya = dad →
    dad + mom + kolya + tanya = 130 →
    dad = 46 ∧ mom = 42 ∧ kolya = 23 ∧ tanya = 19 :=
by
  sorry

end NUMINAMATH_CALUDE_family_ages_l2718_271835


namespace NUMINAMATH_CALUDE_original_number_proof_l2718_271813

theorem original_number_proof (x : ℝ) (h : 1.40 * x = 700) : x = 500 := by
  sorry

end NUMINAMATH_CALUDE_original_number_proof_l2718_271813


namespace NUMINAMATH_CALUDE_triangle_lines_l2718_271893

-- Define the triangle ABC
def A : ℝ × ℝ := (3, -4)
def B : ℝ × ℝ := (6, 0)
def C : ℝ × ℝ := (-5, 2)

-- Define the altitude BD
def altitude_BD (x y : ℝ) : Prop := 4 * x - 3 * y - 24 = 0

-- Define the median BE
def median_BE (x y : ℝ) : Prop := x - 7 * y - 6 = 0

-- Theorem statement
theorem triangle_lines :
  (∀ x y : ℝ, altitude_BD x y ↔ 
    (x - B.1) * (C.2 - A.2) = (y - B.2) * (C.1 - A.1)) ∧
  (∀ x y : ℝ, median_BE x y ↔ 
    2 * (x - B.1) = (A.1 + C.1) - 2 * B.1 ∧
    2 * (y - B.2) = (A.2 + C.2) - 2 * B.2) :=
sorry

end NUMINAMATH_CALUDE_triangle_lines_l2718_271893


namespace NUMINAMATH_CALUDE_fish_tank_count_l2718_271840

theorem fish_tank_count : 
  ∀ (n : ℕ) (first_tank : ℕ) (other_tanks : ℕ),
    n = 3 →
    first_tank = 20 →
    other_tanks = 2 * first_tank →
    first_tank + (n - 1) * other_tanks = 100 := by
  sorry

end NUMINAMATH_CALUDE_fish_tank_count_l2718_271840


namespace NUMINAMATH_CALUDE_quadratic_inequality_solution_l2718_271824

-- Define the quadratic function
def f (x : ℝ) := x^2 - 3*x + 2

-- Define the solution set
def solution_set : Set ℝ := {x | f x < 0}

-- State the theorem
theorem quadratic_inequality_solution :
  solution_set = Set.Ioo 1 2 := by sorry

end NUMINAMATH_CALUDE_quadratic_inequality_solution_l2718_271824


namespace NUMINAMATH_CALUDE_vertex_at_max_min_l2718_271899

/-- The quadratic function f parameterized by k -/
def f (x k : ℝ) : ℝ := x^2 - 2*(2*k - 1)*x + 3*k^2 - 2*k + 6

/-- The x-coordinate of the vertex of f for a given k -/
def vertex_x (k : ℝ) : ℝ := 2*k - 1

/-- The minimum value of f for a given k -/
def min_value (k : ℝ) : ℝ := f (vertex_x k) k

/-- The theorem stating that the x-coordinate of the vertex when the minimum value is maximized is 1 -/
theorem vertex_at_max_min : 
  ∃ (k : ℝ), ∀ (k' : ℝ), min_value k ≥ min_value k' ∧ vertex_x k = 1 := by sorry

end NUMINAMATH_CALUDE_vertex_at_max_min_l2718_271899


namespace NUMINAMATH_CALUDE_red_balls_count_l2718_271825

theorem red_balls_count (total_balls : ℕ) (red_balls : ℕ) (white_balls : ℕ) :
  total_balls = red_balls + white_balls →
  white_balls = 4 →
  (red_balls : ℚ) / total_balls = 6 / 10 →
  red_balls = 6 := by
  sorry

end NUMINAMATH_CALUDE_red_balls_count_l2718_271825


namespace NUMINAMATH_CALUDE_max_intersections_theorem_l2718_271852

/-- A convex polygon inscribed in a circle -/
structure InscribedPolygon where
  sides : ℕ
  sides_ge_3 : sides ≥ 3

/-- The maximum number of intersections between two inscribed polygons -/
def max_intersections (P₁ P₂ : InscribedPolygon) : ℕ := P₁.sides * P₂.sides

/-- Theorem: Maximum intersections between two inscribed polygons -/
theorem max_intersections_theorem (P₁ P₂ : InscribedPolygon) 
  (h : P₁.sides ≤ P₂.sides) : 
  max_intersections P₁ P₂ = P₁.sides * P₂.sides := by sorry

end NUMINAMATH_CALUDE_max_intersections_theorem_l2718_271852


namespace NUMINAMATH_CALUDE_binomial_30_3_l2718_271817

theorem binomial_30_3 : Nat.choose 30 3 = 4060 := by
  sorry

end NUMINAMATH_CALUDE_binomial_30_3_l2718_271817


namespace NUMINAMATH_CALUDE_dog_bones_total_dog_bones_example_l2718_271843

/-- Given a dog with an initial number of bones and a number of bones dug up,
    the total number of bones is equal to the sum of the initial bones and dug up bones. -/
theorem dog_bones_total (initial_bones dug_up_bones : ℕ) :
  initial_bones + dug_up_bones = initial_bones + dug_up_bones := by
  sorry

/-- The specific case from the problem -/
theorem dog_bones_example : 
  493 + 367 = 860 := by
  sorry

end NUMINAMATH_CALUDE_dog_bones_total_dog_bones_example_l2718_271843


namespace NUMINAMATH_CALUDE_find_set_M_l2718_271859

def U : Set Nat := {0, 1, 2, 3}

theorem find_set_M (M : Set Nat) (h : Set.compl M = {2}) : M = {0, 1, 3} := by
  sorry

end NUMINAMATH_CALUDE_find_set_M_l2718_271859


namespace NUMINAMATH_CALUDE_wire_cutting_l2718_271880

theorem wire_cutting (total_length : ℝ) (ratio : ℚ) (shorter_piece : ℝ) :
  total_length = 60 →
  ratio = 2/4 →
  shorter_piece + ratio * shorter_piece = total_length →
  shorter_piece = 20 := by
  sorry

end NUMINAMATH_CALUDE_wire_cutting_l2718_271880


namespace NUMINAMATH_CALUDE_mary_needs_30_apples_l2718_271872

/-- Calculates the number of additional apples needed for baking pies -/
def additional_apples_needed (num_pies : ℕ) (apples_per_pie : ℕ) (apples_harvested : ℕ) : ℕ :=
  max ((num_pies * apples_per_pie) - apples_harvested) 0

/-- Proves that Mary needs to buy 30 more apples -/
theorem mary_needs_30_apples : additional_apples_needed 10 8 50 = 30 := by
  sorry

end NUMINAMATH_CALUDE_mary_needs_30_apples_l2718_271872


namespace NUMINAMATH_CALUDE_amoeba_count_after_ten_days_l2718_271837

/-- The number of amoebas after n days, given an initial population of 1 and a tripling growth rate each day. -/
def amoeba_count (n : ℕ) : ℕ := 3^n

/-- Theorem stating that the number of amoebas after 10 days is equal to 3^10. -/
theorem amoeba_count_after_ten_days : amoeba_count 10 = 3^10 := by
  sorry

end NUMINAMATH_CALUDE_amoeba_count_after_ten_days_l2718_271837


namespace NUMINAMATH_CALUDE_physics_marks_calculation_l2718_271811

def english_marks : ℕ := 74
def math_marks : ℕ := 65
def chemistry_marks : ℕ := 67
def biology_marks : ℕ := 90
def average_marks : ℚ := 75.6
def num_subjects : ℕ := 5

theorem physics_marks_calculation :
  ∃ (physics_marks : ℕ),
    (english_marks + math_marks + physics_marks + chemistry_marks + biology_marks : ℚ) / num_subjects = average_marks ∧
    physics_marks = 82 :=
by sorry

end NUMINAMATH_CALUDE_physics_marks_calculation_l2718_271811


namespace NUMINAMATH_CALUDE_pure_imaginary_complex_number_l2718_271803

theorem pure_imaginary_complex_number (a : ℝ) : 
  (∃ b : ℝ, a - (10 : ℂ) / (3 - Complex.I) = b * Complex.I) → a = 3 := by
  sorry

end NUMINAMATH_CALUDE_pure_imaginary_complex_number_l2718_271803


namespace NUMINAMATH_CALUDE_trees_in_yard_l2718_271836

/-- The number of trees planted along a yard with given specifications -/
def number_of_trees (yard_length : ℕ) (tree_distance : ℕ) : ℕ :=
  (yard_length / tree_distance) + 1

/-- Theorem stating the number of trees planted along the yard -/
theorem trees_in_yard :
  number_of_trees 273 21 = 14 := by
  sorry

end NUMINAMATH_CALUDE_trees_in_yard_l2718_271836


namespace NUMINAMATH_CALUDE_largest_multiple_18_with_9_0_is_correct_division_result_l2718_271895

/-- The largest multiple of 18 with digits 9 or 0 -/
def largest_multiple_18_with_9_0 : ℕ := 9990

/-- Check if a natural number consists only of digits 9 and 0 -/
def has_only_9_and_0_digits (n : ℕ) : Prop :=
  ∀ d : ℕ, d ∈ n.digits 10 → d = 9 ∨ d = 0

theorem largest_multiple_18_with_9_0_is_correct :
  largest_multiple_18_with_9_0 % 18 = 0 ∧
  has_only_9_and_0_digits largest_multiple_18_with_9_0 ∧
  ∀ m : ℕ, m > largest_multiple_18_with_9_0 →
    m % 18 ≠ 0 ∨ ¬(has_only_9_and_0_digits m) :=
by sorry

theorem division_result :
  largest_multiple_18_with_9_0 / 18 = 555 :=
by sorry

end NUMINAMATH_CALUDE_largest_multiple_18_with_9_0_is_correct_division_result_l2718_271895


namespace NUMINAMATH_CALUDE_tangent_lines_count_l2718_271805

/-- A circle in the 2D plane -/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

/-- A line in the 2D plane -/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Check if a line is tangent to a circle -/
def is_tangent (l : Line) (c : Circle) : Prop :=
  let (x₀, y₀) := c.center
  abs (l.a * x₀ + l.b * y₀ + l.c) / Real.sqrt (l.a^2 + l.b^2) = c.radius

/-- Check if a line has equal intercepts on both axes -/
def has_equal_intercepts (l : Line) : Prop :=
  l.a ≠ 0 ∧ l.b ≠ 0 ∧ l.c / l.a = l.c / l.b

/-- The main theorem -/
theorem tangent_lines_count : 
  let c : Circle := ⟨(0, -5), 3⟩
  ∃ (lines : Finset Line), 
    lines.card = 4 ∧ 
    (∀ l ∈ lines, is_tangent l c ∧ has_equal_intercepts l) ∧
    (∀ l : Line, is_tangent l c ∧ has_equal_intercepts l → l ∈ lines) :=
sorry

end NUMINAMATH_CALUDE_tangent_lines_count_l2718_271805


namespace NUMINAMATH_CALUDE_calculate_3Z5_l2718_271858

-- Define the Z operation
def Z (a b : ℝ) : ℝ := b + 15 * a - a^3

-- Theorem statement
theorem calculate_3Z5 : Z 3 5 = 23 := by
  sorry

end NUMINAMATH_CALUDE_calculate_3Z5_l2718_271858


namespace NUMINAMATH_CALUDE_polygon_contains_integer_different_points_l2718_271890

/-- A polygon on the coordinate plane. -/
structure Polygon where
  -- We don't need to define the full structure of a polygon,
  -- just its existence and area property
  area : ℝ

/-- Two points are integer-different if their coordinate differences are integers. -/
def integer_different (p₁ p₂ : ℝ × ℝ) : Prop :=
  ∃ (m n : ℤ), p₁.1 - p₂.1 = m ∧ p₁.2 - p₂.2 = n

/-- Main theorem: If a polygon has area greater than 1, it contains two integer-different points. -/
theorem polygon_contains_integer_different_points (P : Polygon) (h : P.area > 1) :
  ∃ (p₁ p₂ : ℝ × ℝ), p₁ ≠ p₂ ∧ integer_different p₁ p₂ := by
  sorry

end NUMINAMATH_CALUDE_polygon_contains_integer_different_points_l2718_271890


namespace NUMINAMATH_CALUDE_incorrect_complex_analogy_l2718_271839

def complex_square_property (z : ℂ) : Prop :=
  Complex.abs z ^ 2 = z ^ 2

theorem incorrect_complex_analogy :
  ∃ z : ℂ, ¬(complex_square_property z) :=
sorry

end NUMINAMATH_CALUDE_incorrect_complex_analogy_l2718_271839


namespace NUMINAMATH_CALUDE_det_inequality_and_equality_l2718_271808

open Complex Matrix

variable {n : ℕ}

theorem det_inequality_and_equality (A : Matrix (Fin n) (Fin n) ℂ) (a : ℂ) 
  (h : A - conjTranspose A = (2 * a) • 1) : 
  (Complex.abs (det A) ≥ Complex.abs a ^ n) ∧ 
  (Complex.abs (det A) = Complex.abs a ^ n → A = a • 1) := by
  sorry

end NUMINAMATH_CALUDE_det_inequality_and_equality_l2718_271808


namespace NUMINAMATH_CALUDE_jenga_blocks_removed_l2718_271814

def blocks_removed (num_players : ℕ) (num_rounds : ℕ) : ℕ :=
  (num_players * num_rounds * (num_rounds + 1)) / 2

def blocks_removed_sixth_round (num_players : ℕ) (num_rounds : ℕ) : ℕ :=
  blocks_removed num_players num_rounds + (num_rounds + 1)

theorem jenga_blocks_removed : 
  let num_players : ℕ := 5
  let num_rounds : ℕ := 5
  blocks_removed_sixth_round num_players num_rounds = 81 := by
  sorry

end NUMINAMATH_CALUDE_jenga_blocks_removed_l2718_271814


namespace NUMINAMATH_CALUDE_hyperbola_asymptotic_lines_l2718_271892

/-- The asymptotic lines of the hyperbola 3x^2 - y^2 = 3 are y = ± √3 x -/
theorem hyperbola_asymptotic_lines :
  let hyperbola := {(x, y) : ℝ × ℝ | 3 * x^2 - y^2 = 3}
  let asymptotic_lines := {(x, y) : ℝ × ℝ | y = Real.sqrt 3 * x ∨ y = -Real.sqrt 3 * x}
  (Set.range fun t : ℝ => (t, Real.sqrt 3 * t)) ∪ (Set.range fun t : ℝ => (t, -Real.sqrt 3 * t)) =
    {p | p ∈ asymptotic_lines ∧ p ∉ hyperbola ∧ ∀ ε > 0, ∃ q ∈ hyperbola, dist p q < ε} := by
  sorry


end NUMINAMATH_CALUDE_hyperbola_asymptotic_lines_l2718_271892


namespace NUMINAMATH_CALUDE_soccer_basketball_difference_l2718_271853

theorem soccer_basketball_difference :
  let soccer_boxes : ℕ := 8
  let basketball_boxes : ℕ := 5
  let balls_per_box : ℕ := 12
  let total_soccer_balls := soccer_boxes * balls_per_box
  let total_basketballs := basketball_boxes * balls_per_box
  total_soccer_balls - total_basketballs = 36 :=
by sorry

end NUMINAMATH_CALUDE_soccer_basketball_difference_l2718_271853


namespace NUMINAMATH_CALUDE_similar_triangle_longest_side_l2718_271816

theorem similar_triangle_longest_side 
  (a b c : ℝ) 
  (ha : a = 8) 
  (hb : b = 10) 
  (hc : c = 12) 
  (perimeter : ℝ) 
  (h_perimeter : perimeter = 150) 
  (h_similar : ∃ k : ℝ, k > 0 ∧ perimeter = k * (a + b + c)) :
  ∃ longest_side : ℝ, longest_side = 60 ∧ 
    longest_side = max (k * a) (max (k * b) (k * c)) :=
by sorry


end NUMINAMATH_CALUDE_similar_triangle_longest_side_l2718_271816


namespace NUMINAMATH_CALUDE_order_of_three_trig_expressions_l2718_271855

theorem order_of_three_trig_expressions :
  Real.arcsin (3/4) < Real.arccos (1/5) ∧ Real.arccos (1/5) < 1 + Real.arctan (2/3) := by
  sorry

end NUMINAMATH_CALUDE_order_of_three_trig_expressions_l2718_271855


namespace NUMINAMATH_CALUDE_extreme_value_and_range_l2718_271831

noncomputable def f (a x : ℝ) : ℝ := a * Real.exp x + (1 - Real.log x) / x

theorem extreme_value_and_range (a : ℝ) :
  (∀ x > 0, f 0 x ≥ -1 / Real.exp 2) ∧
  (∀ x > 0, f 0 x = -1 / Real.exp 2 → x = Real.exp 2) ∧
  (∀ x > 0, f a x ≥ 1 ↔ a ≥ 1 / Real.exp 2) :=
by sorry

end NUMINAMATH_CALUDE_extreme_value_and_range_l2718_271831


namespace NUMINAMATH_CALUDE_all_trains_return_to_initial_positions_city_n_trains_return_after_2016_minutes_l2718_271801

/-- Represents a metro line with its one-way travel time -/
structure MetroLine where
  one_way_time : ℕ

/-- Represents the metro system of city N -/
structure MetroSystem where
  red_line : MetroLine
  blue_line : MetroLine
  green_line : MetroLine

/-- Checks if a train returns to its initial position after given minutes -/
def returns_to_initial_position (line : MetroLine) (minutes : ℕ) : Prop :=
  minutes % (2 * line.one_way_time) = 0

/-- The theorem stating that all trains return to their initial positions after 2016 minutes -/
theorem all_trains_return_to_initial_positions (metro : MetroSystem) :
  returns_to_initial_position metro.red_line 2016 ∧
  returns_to_initial_position metro.blue_line 2016 ∧
  returns_to_initial_position metro.green_line 2016 :=
by
  sorry

/-- The metro system of city N -/
def city_n_metro : MetroSystem :=
  { red_line := { one_way_time := 7 }
  , blue_line := { one_way_time := 8 }
  , green_line := { one_way_time := 9 }
  }

/-- The main theorem proving that all trains in city N's metro return to their initial positions after 2016 minutes -/
theorem city_n_trains_return_after_2016_minutes :
  returns_to_initial_position city_n_metro.red_line 2016 ∧
  returns_to_initial_position city_n_metro.blue_line 2016 ∧
  returns_to_initial_position city_n_metro.green_line 2016 :=
by
  apply all_trains_return_to_initial_positions

end NUMINAMATH_CALUDE_all_trains_return_to_initial_positions_city_n_trains_return_after_2016_minutes_l2718_271801


namespace NUMINAMATH_CALUDE_correct_selection_ways_l2718_271847

/-- Represents the selection of athletes for a commendation meeting. -/
structure AthletesSelection where
  totalMales : Nat
  totalFemales : Nat
  maleCaptain : Nat
  femaleCaptain : Nat
  selectionSize : Nat

/-- Calculates the number of ways to select athletes under different conditions. -/
def selectionWays (s : AthletesSelection) : Nat × Nat × Nat × Nat :=
  let totalAthletes := s.totalMales + s.totalFemales
  let totalCaptains := s.maleCaptain + s.femaleCaptain
  let nonCaptains := totalAthletes - totalCaptains
  (
    Nat.choose s.totalMales 3 * Nat.choose s.totalFemales 2,
    Nat.choose totalCaptains 1 * Nat.choose nonCaptains 4 + Nat.choose totalCaptains 2 * Nat.choose nonCaptains 3,
    Nat.choose totalAthletes s.selectionSize - Nat.choose s.totalMales s.selectionSize,
    Nat.choose totalAthletes s.selectionSize - Nat.choose nonCaptains s.selectionSize - Nat.choose (s.totalMales - 1) (s.selectionSize - 1)
  )

/-- Theorem stating the correct number of ways to select athletes under different conditions. -/
theorem correct_selection_ways (s : AthletesSelection) 
  (h1 : s.totalMales = 6)
  (h2 : s.totalFemales = 4)
  (h3 : s.maleCaptain = 1)
  (h4 : s.femaleCaptain = 1)
  (h5 : s.selectionSize = 5) :
  selectionWays s = (120, 196, 246, 191) := by
  sorry

end NUMINAMATH_CALUDE_correct_selection_ways_l2718_271847


namespace NUMINAMATH_CALUDE_perimeter_of_modified_square_l2718_271865

/-- Given a square with perimeter 64 inches, cutting out an equilateral triangle
    with side length equal to the square's side and translating it to form a new figure
    results in a figure with perimeter 80 inches. -/
theorem perimeter_of_modified_square (square_perimeter : ℝ) (new_figure_perimeter : ℝ) :
  square_perimeter = 64 →
  new_figure_perimeter = square_perimeter + 2 * (square_perimeter / 4) - (square_perimeter / 4) →
  new_figure_perimeter = 80 :=
by sorry

end NUMINAMATH_CALUDE_perimeter_of_modified_square_l2718_271865


namespace NUMINAMATH_CALUDE_remainder_sum_l2718_271863

theorem remainder_sum (n : ℤ) : n % 20 = 13 → (n % 4 + n % 5 = 4) := by
  sorry

end NUMINAMATH_CALUDE_remainder_sum_l2718_271863


namespace NUMINAMATH_CALUDE_set_A_characterization_l2718_271829

def A : Set ℝ := {a | ∃! x, (x + a) / (x^2 - 1) = 1}

theorem set_A_characterization : A = {-1, 1, -5/4} := by sorry

end NUMINAMATH_CALUDE_set_A_characterization_l2718_271829


namespace NUMINAMATH_CALUDE_darry_climbed_152_steps_l2718_271841

/-- The number of steps Darry climbed today -/
def total_steps : ℕ :=
  let full_ladder_steps : ℕ := 11
  let full_ladder_climbs : ℕ := 10
  let small_ladder_steps : ℕ := 6
  let small_ladder_climbs : ℕ := 7
  full_ladder_steps * full_ladder_climbs + small_ladder_steps * small_ladder_climbs

theorem darry_climbed_152_steps : total_steps = 152 := by
  sorry

end NUMINAMATH_CALUDE_darry_climbed_152_steps_l2718_271841


namespace NUMINAMATH_CALUDE_dave_tickets_l2718_271827

/-- The number of tickets Dave has at the end of the scenario -/
def final_tickets (initial_win : ℕ) (spent : ℕ) (later_win : ℕ) : ℕ :=
  initial_win - spent + later_win

/-- Theorem stating that Dave ends up with 16 tickets -/
theorem dave_tickets : final_tickets 11 5 10 = 16 := by
  sorry

end NUMINAMATH_CALUDE_dave_tickets_l2718_271827


namespace NUMINAMATH_CALUDE_motel_flat_fee_calculation_l2718_271888

/-- A motel charging system with a flat fee for the first night and a fixed amount for additional nights. -/
structure MotelCharge where
  flatFee : ℕ  -- Flat fee for the first night
  nightlyRate : ℕ  -- Fixed amount for each additional night

/-- Calculates the total cost for a given number of nights -/
def totalCost (charge : MotelCharge) (nights : ℕ) : ℕ :=
  charge.flatFee + (nights - 1) * charge.nightlyRate

theorem motel_flat_fee_calculation (charge : MotelCharge) :
  totalCost charge 3 = 155 → totalCost charge 6 = 290 → charge.flatFee = 65 := by
  sorry

#check motel_flat_fee_calculation

end NUMINAMATH_CALUDE_motel_flat_fee_calculation_l2718_271888


namespace NUMINAMATH_CALUDE_lindsay_workout_weight_l2718_271830

/-- Represents the resistance of exercise bands in pounds -/
structure Band where
  resistance : ℕ

/-- Represents a workout exercise with associated weights -/
structure Exercise where
  bands : List Band
  legWeights : ℕ
  additionalWeight : ℕ

/-- Calculates the total weight for an exercise -/
def totalWeight (e : Exercise) : ℕ :=
  (e.bands.map (λ b => b.resistance)).sum + 2 * e.legWeights + e.additionalWeight

/-- Lindsey's workout session -/
def lindseyWorkout : Prop :=
  let bandA : Band := ⟨7⟩
  let bandB : Band := ⟨5⟩
  let bandC : Band := ⟨3⟩
  let squats : Exercise := ⟨[bandA, bandB, bandC], 10, 15⟩
  let lunges : Exercise := ⟨[bandA, bandC], 8, 18⟩
  totalWeight squats + totalWeight lunges = 94

theorem lindsay_workout_weight : lindseyWorkout := by
  sorry

end NUMINAMATH_CALUDE_lindsay_workout_weight_l2718_271830


namespace NUMINAMATH_CALUDE_negation_of_universal_proposition_l2718_271868

theorem negation_of_universal_proposition :
  (¬ ∀ x : ℝ, x^2 - 3*x + 3 > 0) ↔ (∃ x : ℝ, x^2 - 3*x + 3 ≤ 0) := by sorry

end NUMINAMATH_CALUDE_negation_of_universal_proposition_l2718_271868


namespace NUMINAMATH_CALUDE_large_glass_cost_l2718_271886

def cost_of_large_glass (initial_money : ℕ) (small_glass_cost : ℕ) (num_small_glasses : ℕ) (num_large_glasses : ℕ) (money_left : ℕ) : ℕ :=
  let money_after_small := initial_money - (small_glass_cost * num_small_glasses)
  let total_large_cost := money_after_small - money_left
  total_large_cost / num_large_glasses

theorem large_glass_cost :
  cost_of_large_glass 50 3 8 5 1 = 5 := by
  sorry

end NUMINAMATH_CALUDE_large_glass_cost_l2718_271886


namespace NUMINAMATH_CALUDE_min_gymnasts_is_30_l2718_271844

/-- Represents the total number of handshakes in a gymnastics meet -/
def total_handshakes : ℕ := 465

/-- Calculates the number of handshakes given the number of gymnasts -/
def handshakes (n : ℕ) : ℕ := (n * (n - 1)) / 2 + n

/-- Proves that 30 is the minimum number of gymnasts that satisfies the conditions -/
theorem min_gymnasts_is_30 :
  ∀ n : ℕ, n > 0 → n % 2 = 0 → handshakes n = total_handshakes → n ≥ 30 :=
by sorry

end NUMINAMATH_CALUDE_min_gymnasts_is_30_l2718_271844


namespace NUMINAMATH_CALUDE_max_discount_rate_l2718_271887

/-- The maximum discount rate that can be applied without incurring a loss,
    given an initial markup of 25% -/
theorem max_discount_rate : ∀ (m : ℝ) (x : ℝ),
  m > 0 →  -- Assuming positive cost
  (1.25 * m * (1 - x) ≥ m) ↔ (x ≤ 0.2) :=
by sorry

end NUMINAMATH_CALUDE_max_discount_rate_l2718_271887


namespace NUMINAMATH_CALUDE_unique_persistent_number_l2718_271806

/-- Definition of a persistent number -/
def isPersistent (T : ℝ) : Prop :=
  ∀ a b c d : ℝ, a ≠ 0 → a ≠ 1 → b ≠ 0 → b ≠ 1 → c ≠ 0 → c ≠ 1 → d ≠ 0 → d ≠ 1 →
    (a + b + c + d = T ∧ 1/a + 1/b + 1/c + 1/d = T) →
    1/(1-a) + 1/(1-b) + 1/(1-c) + 1/(1-d) = T

/-- Theorem: There exists a unique persistent number, and it equals 2 -/
theorem unique_persistent_number :
  ∃! T : ℝ, isPersistent T ∧ T = 2 := by
  sorry

end NUMINAMATH_CALUDE_unique_persistent_number_l2718_271806


namespace NUMINAMATH_CALUDE_yuan_jiao_conversion_meter_cm_conversion_l2718_271896

-- Define the conversion rates
def jiao_per_yuan : ℚ := 10
def cm_per_meter : ℚ := 100

-- Define the conversion functions
def jiao_to_yuan (j : ℚ) : ℚ := j / jiao_per_yuan
def meters_to_cm (m : ℚ) : ℚ := m * cm_per_meter

-- State the theorems
theorem yuan_jiao_conversion :
  5 + jiao_to_yuan 5 = 5.05 := by sorry

theorem meter_cm_conversion :
  meters_to_cm (12 * 0.1) = 120 := by sorry

end NUMINAMATH_CALUDE_yuan_jiao_conversion_meter_cm_conversion_l2718_271896


namespace NUMINAMATH_CALUDE_prime_from_phi_and_omega_l2718_271819

/-- Euler's totient function -/
def phi (n : ℕ) : ℕ := sorry

/-- Number of prime divisors of n -/
def omega (n : ℕ) : ℕ := sorry

/-- A number is prime if it has exactly two divisors -/
def is_prime (n : ℕ) : Prop := Nat.Prime n

theorem prime_from_phi_and_omega (n : ℕ) 
  (h1 : phi n ∣ (n - 1)) 
  (h2 : omega n ≤ 3) : 
  is_prime n :=
sorry

end NUMINAMATH_CALUDE_prime_from_phi_and_omega_l2718_271819


namespace NUMINAMATH_CALUDE_rock_collection_problem_l2718_271849

theorem rock_collection_problem (minerals_yesterday : ℕ) (gemstones : ℕ) (new_minerals : ℕ) :
  gemstones = minerals_yesterday / 2 →
  new_minerals = 6 →
  gemstones = 21 →
  minerals_yesterday + new_minerals = 48 :=
by sorry

end NUMINAMATH_CALUDE_rock_collection_problem_l2718_271849


namespace NUMINAMATH_CALUDE_product_of_h_at_roots_of_p_l2718_271822

theorem product_of_h_at_roots_of_p (y₁ y₂ y₃ y₄ y₅ : ℂ) : 
  (y₁^5 - y₁^3 + 1 = 0) → 
  (y₂^5 - y₂^3 + 1 = 0) → 
  (y₃^5 - y₃^3 + 1 = 0) → 
  (y₄^5 - y₄^3 + 1 = 0) → 
  (y₅^5 - y₅^3 + 1 = 0) → 
  ((y₁^2 + 1) * (y₂^2 + 1) * (y₃^2 + 1) * (y₄^2 + 1) * (y₅^2 + 1) = Complex.I) :=
by sorry

end NUMINAMATH_CALUDE_product_of_h_at_roots_of_p_l2718_271822


namespace NUMINAMATH_CALUDE_parallel_vectors_k_value_l2718_271802

def vector_parallel (a b : ℝ × ℝ) : Prop :=
  ∃ (t : ℝ), t ≠ 0 ∧ a.1 = t * b.1 ∧ a.2 = t * b.2

theorem parallel_vectors_k_value (k : ℝ) :
  vector_parallel (1, k) (2, 2) → k = 1 := by
  sorry

end NUMINAMATH_CALUDE_parallel_vectors_k_value_l2718_271802


namespace NUMINAMATH_CALUDE_profit_and_marginal_profit_max_not_equal_l2718_271812

/-- Revenue function --/
def R (x : ℕ+) : ℚ := 3000 * x - 20 * x^2

/-- Cost function --/
def C (x : ℕ+) : ℚ := 500 * x + 4000

/-- Profit function --/
def P (x : ℕ+) : ℚ := R x - C x

/-- Marginal profit function --/
def MP (x : ℕ+) : ℚ := P (x + 1) - P x

/-- The maximum allowed production --/
def max_production : ℕ+ := 100

theorem profit_and_marginal_profit_max_not_equal :
  (∃ x : ℕ+, x ≤ max_production ∧ ∀ y : ℕ+, y ≤ max_production → P y ≤ P x) ≠
  (∃ x : ℕ+, x ≤ max_production ∧ ∀ y : ℕ+, y ≤ max_production → MP y ≤ MP x) :=
by sorry

end NUMINAMATH_CALUDE_profit_and_marginal_profit_max_not_equal_l2718_271812


namespace NUMINAMATH_CALUDE_duck_pond_problem_l2718_271815

theorem duck_pond_problem (small_pond : ℕ) (large_pond : ℕ) 
  (green_small : ℚ) (green_large : ℚ) (total_green : ℚ) :
  large_pond = 50 →
  green_small = 1/5 →
  green_large = 3/25 →
  total_green = 3/20 →
  green_small * small_pond.cast + green_large * large_pond.cast = 
    total_green * (small_pond.cast + large_pond.cast) →
  small_pond = 30 := by
sorry

end NUMINAMATH_CALUDE_duck_pond_problem_l2718_271815


namespace NUMINAMATH_CALUDE_cottage_rental_cost_per_hour_l2718_271898

/-- Represents the cost of renting a cottage -/
structure CottageRental where
  hours : ℕ
  jack_payment : ℕ
  jill_payment : ℕ

/-- Calculates the cost per hour of renting a cottage -/
def cost_per_hour (rental : CottageRental) : ℚ :=
  (rental.jack_payment + rental.jill_payment : ℚ) / rental.hours

/-- Theorem: The cost per hour of the cottage rental is $5 -/
theorem cottage_rental_cost_per_hour :
  let rental := CottageRental.mk 8 20 20
  cost_per_hour rental = 5 := by
  sorry

end NUMINAMATH_CALUDE_cottage_rental_cost_per_hour_l2718_271898


namespace NUMINAMATH_CALUDE_bishop_white_invariant_l2718_271883

/-- Represents a position on a chessboard -/
structure Position where
  i : Nat
  j : Nat
  h_valid : i < 8 ∧ j < 8

/-- Checks if a position is on a white square -/
def isWhite (p : Position) : Prop :=
  (p.i + p.j) % 2 = 1

/-- Represents a valid bishop move -/
inductive BishopMove : Position → Position → Prop where
  | diag (p q : Position) (k : Int) :
      q.i = p.i + k ∧ q.j = p.j + k → BishopMove p q

theorem bishop_white_invariant (p q : Position) (h : BishopMove p q) :
  isWhite p → isWhite q := by
  sorry

end NUMINAMATH_CALUDE_bishop_white_invariant_l2718_271883


namespace NUMINAMATH_CALUDE_express_y_in_terms_of_x_l2718_271807

theorem express_y_in_terms_of_x (n : ℕ) (x y : ℝ) : 
  x = 3^n → y = 2 + 9^n → y = 2 + x^2 := by
sorry

end NUMINAMATH_CALUDE_express_y_in_terms_of_x_l2718_271807


namespace NUMINAMATH_CALUDE_min_cost_water_tank_l2718_271864

/-- Represents the dimensions and cost of a rectangular water tank. -/
structure WaterTank where
  length : ℝ
  width : ℝ
  depth : ℝ
  bottomCost : ℝ
  wallCost : ℝ

/-- Calculates the total cost of constructing the water tank. -/
def totalCost (tank : WaterTank) : ℝ :=
  tank.bottomCost * tank.length * tank.width +
  tank.wallCost * 2 * (tank.length + tank.width) * tank.depth

/-- Theorem stating the minimum cost configuration for the water tank. -/
theorem min_cost_water_tank :
  ∃ (tank : WaterTank),
    tank.depth = 3 ∧
    tank.length * tank.width * tank.depth = 48 ∧
    tank.bottomCost = 40 ∧
    tank.wallCost = 20 ∧
    tank.length = 4 ∧
    tank.width = 4 ∧
    totalCost tank = 1600 ∧
    (∀ (other : WaterTank),
      other.depth = 3 →
      other.length * other.width * other.depth = 48 →
      other.bottomCost = 40 →
      other.wallCost = 20 →
      totalCost other ≥ totalCost tank) := by
  sorry

end NUMINAMATH_CALUDE_min_cost_water_tank_l2718_271864


namespace NUMINAMATH_CALUDE_last_number_is_25_l2718_271884

theorem last_number_is_25 (numbers : Fin 7 → ℝ) : 
  (((numbers 0) + (numbers 1) + (numbers 2) + (numbers 3)) / 4 = 13) →
  (((numbers 3) + (numbers 4) + (numbers 5) + (numbers 6)) / 4 = 15) →
  ((numbers 4) + (numbers 5) + (numbers 6) = 55) →
  ((numbers 3) ^ 2 = numbers 6) →
  (numbers 6 = 25) := by
  sorry

end NUMINAMATH_CALUDE_last_number_is_25_l2718_271884


namespace NUMINAMATH_CALUDE_square_vertex_coordinates_l2718_271885

def is_vertex_of_centered_square (x y : ℤ) : Prop :=
  ∃ (s : ℝ), s > 0 ∧ x^2 + y^2 = 2 * s^2

theorem square_vertex_coordinates :
  ∀ x y : ℤ,
    is_vertex_of_centered_square x y →
    Nat.gcd x.natAbs y.natAbs = 2 →
    2 * (x^2 + y^2) = 10 * Nat.lcm x.natAbs y.natAbs →
    ((x = 2 ∧ y = 4) ∨ (x = 4 ∧ y = 2)) :=
by sorry

end NUMINAMATH_CALUDE_square_vertex_coordinates_l2718_271885


namespace NUMINAMATH_CALUDE_partner_b_share_l2718_271862

/-- Calculates the share of a partner in a partnership. -/
def calculate_share (total_profit : ℚ) (investment : ℚ) (total_investment : ℚ) : ℚ :=
  (investment / total_investment) * total_profit

theorem partner_b_share 
  (investment_a investment_b investment_c : ℚ)
  (share_a : ℚ)
  (h1 : investment_a = 7000)
  (h2 : investment_b = 11000)
  (h3 : investment_c = 18000)
  (h4 : share_a = 560) :
  calculate_share 
    ((share_a * (investment_a + investment_b + investment_c)) / investment_a)
    investment_b
    (investment_a + investment_b + investment_c) = 880 := by
  sorry

#eval calculate_share (560 * 36 / 7) 11000 36000

end NUMINAMATH_CALUDE_partner_b_share_l2718_271862


namespace NUMINAMATH_CALUDE_complex_number_purely_imaginary_l2718_271870

def is_purely_imaginary (z : ℂ) : Prop := z.re = 0 ∧ z.im ≠ 0

theorem complex_number_purely_imaginary (m : ℝ) :
  let z : ℂ := Complex.mk (m^2 - m) m
  is_purely_imaginary z → m = 1 := by
sorry

end NUMINAMATH_CALUDE_complex_number_purely_imaginary_l2718_271870


namespace NUMINAMATH_CALUDE_min_distance_Q_to_C_l2718_271809

noncomputable def A : ℝ × ℝ := (-1, 2)
noncomputable def B : ℝ × ℝ := (0, 1)

def C : Set (ℝ × ℝ) := {p : ℝ × ℝ | (p.1 - 1)^2 + p.2^2 = 4}

def l₁ : Set (ℝ × ℝ) := {q : ℝ × ℝ | 3 * q.1 - 4 * q.2 + 12 = 0}

theorem min_distance_Q_to_C :
  ∀ Q ∈ l₁, ∃ M ∈ C, ∀ M' ∈ C, dist Q M ≤ dist Q M' ∧ dist Q M ≥ Real.sqrt 5 :=
sorry

end NUMINAMATH_CALUDE_min_distance_Q_to_C_l2718_271809


namespace NUMINAMATH_CALUDE_max_area_rectangular_pen_l2718_271894

/-- Given 60 feet of fencing for a rectangular pen, the maximum possible area is 225 square feet -/
theorem max_area_rectangular_pen (perimeter : ℝ) (area : ℝ → ℝ → ℝ) :
  perimeter = 60 →
  (∀ x y, x > 0 → y > 0 → x + y = perimeter / 2 → area x y = x * y) →
  (∃ x y, x > 0 ∧ y > 0 ∧ x + y = perimeter / 2 ∧ area x y = 225) ∧
  (∀ x y, x > 0 → y > 0 → x + y = perimeter / 2 → area x y ≤ 225) :=
by sorry

end NUMINAMATH_CALUDE_max_area_rectangular_pen_l2718_271894


namespace NUMINAMATH_CALUDE_circle_properties_l2718_271876

/-- Circle with center (6,8) and radius 10 -/
def Circle := {p : ℝ × ℝ | (p.1 - 6)^2 + (p.2 - 8)^2 = 100}

/-- The circle passes through the origin -/
axiom origin_on_circle : (0, 0) ∈ Circle

/-- P is the point where the circle intersects the x-axis -/
def P : ℝ × ℝ := (12, 0)

/-- Q is the point on the circle with maximum y-coordinate -/
def Q : ℝ × ℝ := (6, 18)

/-- R is the point on the circle forming a right angle with P and Q -/
def R : ℝ × ℝ := (0, 16)

/-- S and T are the points on the circle forming 45-degree angles with P and Q -/
def S : ℝ × ℝ := (14, 14)
def T : ℝ × ℝ := (-2, 2)

theorem circle_properties :
  P ∈ Circle ∧
  Q ∈ Circle ∧
  R ∈ Circle ∧
  S ∈ Circle ∧
  T ∈ Circle ∧
  P.2 = 0 ∧
  ∀ p ∈ Circle, p.2 ≤ Q.2 ∧
  (R.1 - Q.1) * (P.1 - Q.1) + (R.2 - Q.2) * (P.2 - Q.2) = 0 ∧
  (S.1 - Q.1) * (P.1 - Q.1) + (S.2 - Q.2) * (P.2 - Q.2) =
    (T.1 - Q.1) * (P.1 - Q.1) + (T.2 - Q.2) * (P.2 - Q.2) :=
by sorry

end NUMINAMATH_CALUDE_circle_properties_l2718_271876


namespace NUMINAMATH_CALUDE_ice_rinks_and_ski_resorts_2019_l2718_271854

/-- The number of ice skating rinks in 2019 -/
def ice_rinks_2019 : ℕ := 830

/-- The number of ski resorts in 2019 -/
def ski_resorts_2019 : ℕ := 400

/-- The total number of ice skating rinks and ski resorts in 2019 -/
def total_2019 : ℕ := 1230

/-- The total number of ice skating rinks and ski resorts in 2022 -/
def total_2022 : ℕ := 2560

/-- The increase in ice skating rinks from 2019 to 2022 -/
def ice_rinks_increase : ℕ := 212

/-- The increase in ski resorts from 2019 to 2022 -/
def ski_resorts_increase : ℕ := 288

theorem ice_rinks_and_ski_resorts_2019 :
  ice_rinks_2019 + ski_resorts_2019 = total_2019 ∧
  2 * ice_rinks_2019 + ice_rinks_increase + ski_resorts_2019 + ski_resorts_increase = total_2022 := by
  sorry

end NUMINAMATH_CALUDE_ice_rinks_and_ski_resorts_2019_l2718_271854


namespace NUMINAMATH_CALUDE_gunther_working_time_l2718_271851

/-- Gunther's typing speed in words per minute -/
def typing_speed : ℚ := 160 / 3

/-- Total words Gunther types in a working day -/
def total_words : ℕ := 25600

/-- Gunther's working time in minutes -/
def working_time : ℕ := 480

theorem gunther_working_time :
  (total_words : ℚ) / typing_speed = working_time := by sorry

end NUMINAMATH_CALUDE_gunther_working_time_l2718_271851


namespace NUMINAMATH_CALUDE_vlad_score_in_competition_l2718_271818

/-- A video game competition between two players -/
structure VideoGameCompetition where
  rounds : ℕ
  points_per_win : ℕ
  taro_score : ℕ

/-- Calculate Vlad's score in the video game competition -/
def vlad_score (game : VideoGameCompetition) : ℕ :=
  game.rounds * game.points_per_win - game.taro_score

/-- Theorem stating Vlad's score in the specific competition described in the problem -/
theorem vlad_score_in_competition :
  let game : VideoGameCompetition := {
    rounds := 30,
    points_per_win := 5,
    taro_score := 3 * (30 * 5) / 5 - 4
  }
  vlad_score game = 64 := by sorry

end NUMINAMATH_CALUDE_vlad_score_in_competition_l2718_271818


namespace NUMINAMATH_CALUDE_number_and_percentage_problem_l2718_271846

theorem number_and_percentage_problem (N P : ℝ) : 
  (1/4 : ℝ) * (1/3 : ℝ) * (2/5 : ℝ) * N = 25 ∧ 
  (P/100 : ℝ) * N = 300 →
  N = 750 ∧ P = 40 := by
  sorry

end NUMINAMATH_CALUDE_number_and_percentage_problem_l2718_271846


namespace NUMINAMATH_CALUDE_all_vertices_integer_l2718_271826

/-- A cube in 3D space -/
structure Cube where
  vertices : Fin 8 → ℤ × ℤ × ℤ

/-- Predicate to check if four vertices form a valid cube face -/
def is_valid_face (v₁ v₂ v₃ v₄ : ℤ × ℤ × ℤ) : Prop := sorry

/-- Predicate to check if four vertices are non-coplanar -/
def are_non_coplanar (v₁ v₂ v₃ v₄ : ℤ × ℤ × ℤ) : Prop := sorry

/-- Theorem: If four non-coplanar vertices of a cube have integer coordinates, 
    then all vertices of the cube have integer coordinates -/
theorem all_vertices_integer (c : Cube) 
  (h₁ : is_valid_face (c.vertices 0) (c.vertices 1) (c.vertices 2) (c.vertices 3))
  (h₂ : are_non_coplanar (c.vertices 0) (c.vertices 1) (c.vertices 2) (c.vertices 3)) :
  ∀ i, ∃ (x y z : ℤ), c.vertices i = (x, y, z) := by
  sorry


end NUMINAMATH_CALUDE_all_vertices_integer_l2718_271826


namespace NUMINAMATH_CALUDE_hadassah_additional_paintings_l2718_271857

/-- Calculates the number of additional paintings given initial and total painting information -/
def additional_paintings (initial_paintings : ℕ) (initial_time : ℕ) (total_time : ℕ) : ℕ :=
  let painting_rate := initial_paintings / initial_time
  let additional_time := total_time - initial_time
  painting_rate * additional_time

/-- Proves that Hadassah painted 20 additional paintings -/
theorem hadassah_additional_paintings :
  additional_paintings 12 6 16 = 20 := by
  sorry

end NUMINAMATH_CALUDE_hadassah_additional_paintings_l2718_271857


namespace NUMINAMATH_CALUDE_lines_perp_to_plane_are_parallel_perp_line_to_parallel_planes_l2718_271810

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the relations between lines and planes
variable (perpendicular : Line → Plane → Prop)
variable (parallel_lines : Line → Line → Prop)
variable (parallel_planes : Plane → Plane → Prop)

-- Theorem 1: If two lines are perpendicular to the same plane, then they are parallel
theorem lines_perp_to_plane_are_parallel 
  (m n : Line) (α : Plane) 
  (h1 : perpendicular m α) (h2 : perpendicular n α) :
  parallel_lines m n := by sorry

-- Theorem 2: If two planes are parallel, and a line is perpendicular to one of them,
-- then it is perpendicular to the other
theorem perp_line_to_parallel_planes 
  (m : Line) (α β γ : Plane)
  (h1 : parallel_planes α β) (h2 : parallel_planes β γ) (h3 : perpendicular m α) :
  perpendicular m γ := by sorry

end NUMINAMATH_CALUDE_lines_perp_to_plane_are_parallel_perp_line_to_parallel_planes_l2718_271810


namespace NUMINAMATH_CALUDE_fifteenth_student_age_l2718_271860

theorem fifteenth_student_age 
  (total_students : Nat) 
  (group1_students : Nat) 
  (group2_students : Nat) 
  (total_average_age : ℝ) 
  (group1_average_age : ℝ) 
  (group2_average_age : ℝ) 
  (h1 : total_students = 15)
  (h2 : group1_students = 8)
  (h3 : group2_students = 6)
  (h4 : total_average_age = 15)
  (h5 : group1_average_age = 14)
  (h6 : group2_average_age = 16)
  (h7 : group1_students + group2_students + 1 = total_students) :
  (total_students : ℝ) * total_average_age - 
  ((group1_students : ℝ) * group1_average_age + (group2_students : ℝ) * group2_average_age) = 17 := by
  sorry


end NUMINAMATH_CALUDE_fifteenth_student_age_l2718_271860


namespace NUMINAMATH_CALUDE_min_visible_pairs_155_birds_l2718_271874

/-- The number of birds on the circle -/
def num_birds : ℕ := 155

/-- The visibility threshold in degrees -/
def visibility_threshold : ℝ := 10

/-- A function that calculates the minimum number of mutually visible bird pairs -/
def min_visible_pairs (n : ℕ) (threshold : ℝ) : ℕ :=
  sorry

/-- Theorem stating the minimum number of mutually visible bird pairs -/
theorem min_visible_pairs_155_birds :
  min_visible_pairs num_birds visibility_threshold = 270 :=
sorry

end NUMINAMATH_CALUDE_min_visible_pairs_155_birds_l2718_271874


namespace NUMINAMATH_CALUDE_select_blocks_count_l2718_271838

/-- The number of ways to select 4 blocks from a 6x6 grid such that no two blocks are in the same row or column -/
def select_blocks : ℕ :=
  Nat.choose 6 4 * Nat.choose 6 4 * Nat.factorial 4

/-- Theorem stating that the number of ways to select 4 blocks from a 6x6 grid
    such that no two blocks are in the same row or column is 5400 -/
theorem select_blocks_count : select_blocks = 5400 := by
  sorry

end NUMINAMATH_CALUDE_select_blocks_count_l2718_271838


namespace NUMINAMATH_CALUDE_cereal_eating_time_l2718_271804

/-- The time taken for two people to eat a certain amount of cereal together -/
def time_to_eat_together (fat_rate thin_rate total_amount : ℚ) : ℚ :=
  total_amount / (fat_rate + thin_rate)

theorem cereal_eating_time :
  let fat_rate : ℚ := 1 / 15  -- Mr. Fat's eating rate in pounds per minute
  let thin_rate : ℚ := 1 / 25 -- Mr. Thin's eating rate in pounds per minute
  let total_amount : ℚ := 4   -- Total amount of cereal in pounds
  time_to_eat_together fat_rate thin_rate total_amount = 75 / 2 := by
  sorry

#eval (75 : ℚ) / 2 -- Should output 37.5

end NUMINAMATH_CALUDE_cereal_eating_time_l2718_271804


namespace NUMINAMATH_CALUDE_product_of_sum_and_difference_l2718_271823

theorem product_of_sum_and_difference (x y : ℝ) (h1 : x > y) (h2 : x + y = 20) (h3 : x - y = 4) :
  (3 * x) * y = 288 := by
  sorry

end NUMINAMATH_CALUDE_product_of_sum_and_difference_l2718_271823


namespace NUMINAMATH_CALUDE_rectangle_ratio_is_two_l2718_271845

/-- Geometric arrangement of squares and rectangles -/
structure SquareFrame where
  inner_side : ℝ
  outer_side : ℝ
  rect_short : ℝ
  rect_long : ℝ
  area_ratio : outer_side^2 = 9 * inner_side^2
  outer_side_composition : outer_side = inner_side + 2 * rect_short
  inner_side_composition : inner_side + rect_long = outer_side

/-- Theorem: The ratio of the longer side to the shorter side of each rectangle is 2 -/
theorem rectangle_ratio_is_two (frame : SquareFrame) :
  frame.rect_long / frame.rect_short = 2 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_ratio_is_two_l2718_271845


namespace NUMINAMATH_CALUDE_complex_fraction_simplification_l2718_271897

theorem complex_fraction_simplification :
  let z₁ : ℂ := 4 + 7*I
  let z₂ : ℂ := 4 - 7*I
  (z₁ / z₂) - (z₂ / z₁) = 112 * I / 65 := by sorry

end NUMINAMATH_CALUDE_complex_fraction_simplification_l2718_271897


namespace NUMINAMATH_CALUDE_straight_lines_parabolas_disjoint_l2718_271867

-- Define the set of all straight lines
def StraightLines : Set (ℝ → ℝ) := {f | ∃ a b : ℝ, ∀ x, f x = a * x + b}

-- Define the set of all parabolas
def Parabolas : Set (ℝ → ℝ) := {f | ∃ a b c : ℝ, a ≠ 0 ∧ ∀ x, f x = a * x^2 + b * x + c}

-- Theorem statement
theorem straight_lines_parabolas_disjoint : StraightLines ∩ Parabolas = ∅ := by
  sorry

end NUMINAMATH_CALUDE_straight_lines_parabolas_disjoint_l2718_271867


namespace NUMINAMATH_CALUDE_white_circle_area_on_cube_l2718_271842

/-- Represents the problem of calculating the area of a white circle on a cube face --/
theorem white_circle_area_on_cube (edge_length : ℝ) (green_paint_area : ℝ) : 
  edge_length = 12 → 
  green_paint_area = 432 → 
  (6 * edge_length^2 - green_paint_area) / 6 = 72 := by
  sorry

#check white_circle_area_on_cube

end NUMINAMATH_CALUDE_white_circle_area_on_cube_l2718_271842


namespace NUMINAMATH_CALUDE_triangle_angle_B_l2718_271881

-- Define a triangle ABC
structure Triangle :=
  (A B C : Real)  -- Angles
  (a b c : Real)  -- Sides

-- Define the theorem
theorem triangle_angle_B (t : Triangle) :
  t.A = π/4 ∧ t.a = Real.sqrt 2 ∧ t.b = Real.sqrt 3 →
  t.B = π/3 ∨ t.B = 2*π/3 :=
by
  sorry

end NUMINAMATH_CALUDE_triangle_angle_B_l2718_271881


namespace NUMINAMATH_CALUDE_walking_delay_bus_miss_time_l2718_271820

/-- Given a usual walking time and a reduced speed factor, calculates the delay in reaching the destination. -/
theorem walking_delay (usual_time : ℝ) (speed_factor : ℝ) : 
  usual_time > 0 → 
  speed_factor > 0 → 
  speed_factor < 1 → 
  (usual_time / speed_factor) - usual_time = usual_time * (1 / speed_factor - 1) :=
by sorry

/-- Proves that walking at 4/5 of the usual speed, with a usual time of 24 minutes, results in a 6-minute delay. -/
theorem bus_miss_time (usual_time : ℝ) (h1 : usual_time = 24) : 
  (usual_time / (4/5)) - usual_time = 6 :=
by sorry

end NUMINAMATH_CALUDE_walking_delay_bus_miss_time_l2718_271820


namespace NUMINAMATH_CALUDE_total_shoes_l2718_271891

def scott_shoes : ℕ := 7

def anthony_shoes : ℕ := 3 * scott_shoes

def jim_shoes : ℕ := anthony_shoes - 2

def melissa_shoes : ℕ := jim_shoes / 2

def tim_shoes : ℕ := (anthony_shoes + melissa_shoes) / 2

theorem total_shoes : scott_shoes + anthony_shoes + jim_shoes + melissa_shoes + tim_shoes = 71 := by
  sorry

end NUMINAMATH_CALUDE_total_shoes_l2718_271891


namespace NUMINAMATH_CALUDE_complement_A_intersect_B_when_m_zero_A_subset_B_iff_m_leq_neg_three_l2718_271833

-- Define the sets A and B
def A : Set ℝ := {x : ℝ | -3 < x ∧ x < 2}
def B (m : ℝ) : Set ℝ := {x : ℝ | m < x ∧ x < 5}

-- Theorem for part (1)
theorem complement_A_intersect_B_when_m_zero :
  (Set.univ \ A) ∩ B 0 = Set.Icc 2 5 := by sorry

-- Theorem for part (2)
theorem A_subset_B_iff_m_leq_neg_three (m : ℝ) :
  A ⊆ B m ↔ m ≤ -3 := by sorry

end NUMINAMATH_CALUDE_complement_A_intersect_B_when_m_zero_A_subset_B_iff_m_leq_neg_three_l2718_271833


namespace NUMINAMATH_CALUDE_multiplicative_inverse_301_mod_401_l2718_271856

theorem multiplicative_inverse_301_mod_401 : ∃ x : ℤ, 0 ≤ x ∧ x < 401 ∧ (301 * x) % 401 = 1 :=
  by
  use 397
  sorry

end NUMINAMATH_CALUDE_multiplicative_inverse_301_mod_401_l2718_271856


namespace NUMINAMATH_CALUDE_unique_prime_sum_of_squares_l2718_271877

theorem unique_prime_sum_of_squares (p k x y a b : ℤ) : 
  Prime p → 
  p = 4 * k + 1 → 
  p = x^2 + y^2 → 
  p = a^2 + b^2 → 
  (x = a ∧ y = b) ∨ (x = -a ∧ y = -b) ∨ (x = b ∧ y = -a) ∨ (x = -b ∧ y = a) :=
sorry

end NUMINAMATH_CALUDE_unique_prime_sum_of_squares_l2718_271877


namespace NUMINAMATH_CALUDE_amount_left_after_spending_l2718_271878

def mildred_spent : ℕ := 25
def candice_spent : ℕ := 35
def total_given : ℕ := 100

theorem amount_left_after_spending :
  total_given - (mildred_spent + candice_spent) = 40 :=
by sorry

end NUMINAMATH_CALUDE_amount_left_after_spending_l2718_271878


namespace NUMINAMATH_CALUDE_banana_price_reduction_l2718_271889

/-- Given a 50% reduction in banana prices allows buying 80 more dozens for 60000.25 rupees,
    prove the reduced price per dozen is 375.0015625 rupees. -/
theorem banana_price_reduction (original_price : ℝ) : 
  (2 * 60000.25 / original_price - 60000.25 / original_price = 80) → 
  (original_price / 2 = 375.0015625) :=
by
  sorry

end NUMINAMATH_CALUDE_banana_price_reduction_l2718_271889


namespace NUMINAMATH_CALUDE_unique_prime_double_squares_l2718_271800

theorem unique_prime_double_squares : 
  ∃! (p : ℕ), 
    Prime p ∧ 
    (∃ (x y : ℕ), p + 7 = 2 * x^2 ∧ p^2 + 7 = 2 * y^2) ∧ 
    p = 11 := by
  sorry

end NUMINAMATH_CALUDE_unique_prime_double_squares_l2718_271800


namespace NUMINAMATH_CALUDE_calendar_puzzle_l2718_271828

def date_behind (letter : Char) (base : ℕ) : ℕ :=
  match letter with
  | 'A' => base
  | 'B' => base + 1
  | 'C' => base + 2
  | 'D' => base + 3
  | 'E' => base + 4
  | 'F' => base + 5
  | 'G' => base + 6
  | _ => base

theorem calendar_puzzle (base : ℕ) :
  ∃ (x : Char), (date_behind 'B' base + date_behind x base = 2 * date_behind 'A' base + 6) ∧ x = 'F' :=
by sorry

end NUMINAMATH_CALUDE_calendar_puzzle_l2718_271828


namespace NUMINAMATH_CALUDE_total_purchase_cost_l2718_271861

/-- Represents the price of a single small pack -/
def small_pack_price : ℚ := 387 / 100

/-- Represents the price of a single large pack -/
def large_pack_price : ℚ := 549 / 100

/-- Calculates the cost of small packs with bulk pricing -/
def small_pack_cost (n : ℕ) : ℚ :=
  if n ≥ 10 then n * small_pack_price * (1 - 1/10)
  else if n ≥ 5 then 5 * small_pack_price * (1 - 1/20) + (n - 5) * small_pack_price
  else n * small_pack_price

/-- Calculates the cost of large packs with bulk pricing -/
def large_pack_cost (n : ℕ) : ℚ :=
  if n ≥ 6 then n * large_pack_price * (1 - 3/20)
  else if n ≥ 3 then 3 * large_pack_price * (1 - 7/100) + (n - 3) * large_pack_price
  else n * large_pack_price

/-- Theorem stating the total cost of the purchase -/
theorem total_purchase_cost :
  (small_pack_cost 8 + large_pack_cost 4) * 100 = 5080 := by
  sorry

end NUMINAMATH_CALUDE_total_purchase_cost_l2718_271861

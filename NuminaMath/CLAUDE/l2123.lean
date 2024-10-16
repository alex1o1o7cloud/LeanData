import Mathlib

namespace NUMINAMATH_CALUDE_problem_solution_l2123_212307

-- Define the function f(x) = x^3 + ax^2 - x
def f (a : ℝ) (x : ℝ) : ℝ := x^3 + a*x^2 - x

-- Define the derivative of f
def f' (a : ℝ) (x : ℝ) : ℝ := 3*x^2 + 2*a*x - 1

theorem problem_solution (a : ℝ) (h : f' a 1 = 4) :
  a = 1 ∧
  ∃ (m b : ℝ), m = 4 ∧ b = -3 ∧ ∀ x y, y = f a x → (y - f a 1 = m * (x - 1) ↔ m*x - y - b = 0) ∧
  ∃ (lower upper : ℝ), lower = -5/27 ∧ upper = 10 ∧
    (∀ x, x ∈ Set.Icc 0 2 → f a x ∈ Set.Icc lower upper) ∧
    (∃ x₁ x₂, x₁ ∈ Set.Icc 0 2 ∧ x₂ ∈ Set.Icc 0 2 ∧ f a x₁ = lower ∧ f a x₂ = upper) :=
by
  sorry


end NUMINAMATH_CALUDE_problem_solution_l2123_212307


namespace NUMINAMATH_CALUDE_grace_constant_reading_rate_l2123_212384

/-- Grace's reading rate in pages per hour -/
def reading_rate (pages : ℕ) (hours : ℕ) : ℚ :=
  pages / hours

theorem grace_constant_reading_rate :
  let rate1 := reading_rate 200 20
  let rate2 := reading_rate 250 25
  rate1 = rate2 ∧ rate1 = 10 := by
  sorry

end NUMINAMATH_CALUDE_grace_constant_reading_rate_l2123_212384


namespace NUMINAMATH_CALUDE_inequality_theorem_l2123_212387

theorem inequality_theorem (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) 
  (h_prod : a * b * c = 1) 
  (h_ineq : a^2011 + b^2011 + c^2011 < 1/a^2011 + 1/b^2011 + 1/c^2011) : 
  a + b + c < 1/a + 1/b + 1/c := by
sorry

end NUMINAMATH_CALUDE_inequality_theorem_l2123_212387


namespace NUMINAMATH_CALUDE_sum_of_max_min_g_l2123_212329

def g (x : ℝ) : ℝ := |x - 5| + |x - 7| - |2*x - 12| + |3*x - 21|

theorem sum_of_max_min_g :
  ∃ (max min : ℝ),
    (∀ x : ℝ, 5 ≤ x → x ≤ 10 → g x ≤ max) ∧
    (∃ x : ℝ, 5 ≤ x ∧ x ≤ 10 ∧ g x = max) ∧
    (∀ x : ℝ, 5 ≤ x → x ≤ 10 → min ≤ g x) ∧
    (∃ x : ℝ, 5 ≤ x ∧ x ≤ 10 ∧ g x = min) ∧
    max + min = 11 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_max_min_g_l2123_212329


namespace NUMINAMATH_CALUDE_expression_evaluation_l2123_212344

theorem expression_evaluation : -20 + 12 * (8 / 4) - 5 = -1 := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l2123_212344


namespace NUMINAMATH_CALUDE_tower_surface_area_is_1207_l2123_212332

def cube_volumes : List ℕ := [1, 27, 64, 125, 216, 343, 512, 729]

def cube_side_lengths : List ℕ := [1, 3, 4, 5, 6, 7, 8, 9]

def visible_faces : List ℕ := [6, 4, 4, 4, 4, 4, 4, 5]

def tower_surface_area (volumes : List ℕ) (side_lengths : List ℕ) (faces : List ℕ) : ℕ :=
  (List.zip (List.zip side_lengths faces) volumes).foldr
    (fun ((s, f), v) acc => acc + f * s * s)
    0

theorem tower_surface_area_is_1207 :
  tower_surface_area cube_volumes cube_side_lengths visible_faces = 1207 := by
  sorry

end NUMINAMATH_CALUDE_tower_surface_area_is_1207_l2123_212332


namespace NUMINAMATH_CALUDE_icosikaipentagon_diagonals_from_vertex_l2123_212383

/-- The number of diagonals from a single vertex in a polygon with n sides -/
def diagonals_from_vertex (n : ℕ) : ℕ := n - 3

/-- An icosikaipentagon is a polygon with 25 sides -/
def icosikaipentagon_sides : ℕ := 25

theorem icosikaipentagon_diagonals_from_vertex : 
  diagonals_from_vertex icosikaipentagon_sides = 22 := by
  sorry

end NUMINAMATH_CALUDE_icosikaipentagon_diagonals_from_vertex_l2123_212383


namespace NUMINAMATH_CALUDE_yellow_face_probability_l2123_212370

/-- The probability of rolling a yellow face on a modified 10-sided die -/
theorem yellow_face_probability (total_faces : ℕ) (yellow_faces : ℕ) : 
  total_faces = 10 → yellow_faces = 4 → (yellow_faces : ℚ) / total_faces = 2 / 5 := by
  sorry

end NUMINAMATH_CALUDE_yellow_face_probability_l2123_212370


namespace NUMINAMATH_CALUDE_exists_integer_function_double_application_square_l2123_212336

theorem exists_integer_function_double_application_square :
  ∃ f : ℤ → ℤ, ∀ n : ℤ, f (f n) = n^2 := by
  sorry

end NUMINAMATH_CALUDE_exists_integer_function_double_application_square_l2123_212336


namespace NUMINAMATH_CALUDE_two_color_theorem_l2123_212374

/-- Represents a region in the plane --/
structure Region where
  id : Nat

/-- Represents the configuration of circles and lines --/
structure Configuration where
  regions : List Region
  adjacency : Region → Region → Bool

/-- Represents a coloring of regions --/
def Coloring := Region → Bool

/-- A valid coloring is one where adjacent regions have different colors --/
def is_valid_coloring (config : Configuration) (coloring : Coloring) : Prop :=
  ∀ r1 r2, config.adjacency r1 r2 → coloring r1 ≠ coloring r2

theorem two_color_theorem (config : Configuration) :
  ∃ (coloring : Coloring), is_valid_coloring config coloring := by
  sorry

end NUMINAMATH_CALUDE_two_color_theorem_l2123_212374


namespace NUMINAMATH_CALUDE_social_relationships_theorem_l2123_212303

/-- Represents the relationship between two people -/
inductive Relationship
  | Knows
  | DoesNotKnow

/-- A function representing the relationship between people -/
def relationship (people : ℕ) : (Fin people → Fin people → Relationship) :=
  sorry

theorem social_relationships_theorem (n : ℕ) :
  ∃ (A B : Fin (2*n+2)), ∃ (S : Finset (Fin (2*n+2))),
    S.card ≥ n ∧
    (∀ C ∈ S, C ≠ A ∧ C ≠ B) ∧
    (∀ C ∈ S, (relationship (2*n+2) A C = relationship (2*n+2) B C)) :=
  sorry

end NUMINAMATH_CALUDE_social_relationships_theorem_l2123_212303


namespace NUMINAMATH_CALUDE_ellipse_min_sum_l2123_212347

theorem ellipse_min_sum (m n : ℝ) (hm : m > 0) (hn : n > 0) : 
  (3 * Real.sqrt 3)^2 / m^2 + 1 / n^2 = 1 → m + n ≥ 8 := by sorry

end NUMINAMATH_CALUDE_ellipse_min_sum_l2123_212347


namespace NUMINAMATH_CALUDE_plate_on_square_table_l2123_212365

/-- Given a square table with a round plate, if the distances from the plate's edge
    to two adjacent sides of the table are a and b, and the distance from the plate's edge
    to the opposite side of the a measurement is c, then the distance from the plate's edge
    to the opposite side of the b measurement is a + c - b. -/
theorem plate_on_square_table (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  a + c = b + (a + c - b) :=
by sorry

end NUMINAMATH_CALUDE_plate_on_square_table_l2123_212365


namespace NUMINAMATH_CALUDE_cat_weight_problem_l2123_212340

theorem cat_weight_problem (total_weight : ℕ) (cat1_weight : ℕ) (cat2_weight : ℕ) 
  (h1 : total_weight = 13)
  (h2 : cat1_weight = 2)
  (h3 : cat2_weight = 7) :
  total_weight - cat1_weight - cat2_weight = 4 := by
  sorry

end NUMINAMATH_CALUDE_cat_weight_problem_l2123_212340


namespace NUMINAMATH_CALUDE_apple_difference_l2123_212341

theorem apple_difference (adam_apples jackie_apples : ℕ) 
  (adam_has : adam_apples = 9) 
  (jackie_has : jackie_apples = 10) : 
  jackie_apples - adam_apples = 1 := by
sorry

end NUMINAMATH_CALUDE_apple_difference_l2123_212341


namespace NUMINAMATH_CALUDE_x_less_than_negative_one_l2123_212346

theorem x_less_than_negative_one (a b : ℚ) 
  (ha : -2 < a ∧ a < -1.5) 
  (hb : 0.5 < b ∧ b < 1) : 
  let x := (a - 5*b) / (a + 5*b)
  x < -1 := by
sorry

end NUMINAMATH_CALUDE_x_less_than_negative_one_l2123_212346


namespace NUMINAMATH_CALUDE_valid_tree_arrangement_exists_l2123_212367

/-- Represents a tree type -/
inductive TreeType
| Apple
| Pear
| Plum
| Apricot
| Cherry
| Almond

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents an equilateral triangle -/
structure EquilateralTriangle where
  vertex1 : Point
  vertex2 : Point
  vertex3 : Point

/-- Represents the arrangement of trees -/
structure TreeArrangement where
  triangles : List EquilateralTriangle
  treeAssignment : Point → Option TreeType

/-- The main theorem stating that a valid tree arrangement exists -/
theorem valid_tree_arrangement_exists : ∃ (arrangement : TreeArrangement), 
  (arrangement.triangles.length = 6) ∧ 
  (∀ t ∈ arrangement.triangles, 
    ∃ (t1 t2 t3 : TreeType), t1 ≠ t2 ∧ t2 ≠ t3 ∧ t1 ≠ t3 ∧
    arrangement.treeAssignment t.vertex1 = some t1 ∧
    arrangement.treeAssignment t.vertex2 = some t2 ∧
    arrangement.treeAssignment t.vertex3 = some t3) ∧
  (∀ p : Point, (∃ t ∈ arrangement.triangles, p ∈ [t.vertex1, t.vertex2, t.vertex3]) →
    ∃! treeType : TreeType, arrangement.treeAssignment p = some treeType) :=
by sorry

end NUMINAMATH_CALUDE_valid_tree_arrangement_exists_l2123_212367


namespace NUMINAMATH_CALUDE_triangle_ratio_theorem_l2123_212353

/-- Represents a point in the plane -/
structure Point :=
  (x : ℝ)
  (y : ℝ)

/-- Represents a triangle with vertices A, B, and C -/
structure Triangle :=
  (A : Point)
  (B : Point)
  (C : Point)

/-- Checks if a point lies on a line segment between two other points -/
def lies_on (P Q R : Point) : Prop := sorry

/-- Checks if two line segments intersect -/
def intersect (P Q R S : Point) : Prop := sorry

/-- Represents the ratio of distances between points -/
def distance_ratio (P Q R S : Point) : ℚ := sorry

theorem triangle_ratio_theorem (ABC : Triangle) (D E T : Point) :
  lies_on D ABC.B ABC.C →
  lies_on E ABC.A ABC.B →
  intersect ABC.A D ABC.B E →
  (∃ (t : Point), intersect ABC.A D ABC.B E ∧ t = T) →
  distance_ratio ABC.A T T D = 2 →
  distance_ratio ABC.B T T E = 3 →
  distance_ratio ABC.C D D ABC.B = 2/9 := by
  sorry

end NUMINAMATH_CALUDE_triangle_ratio_theorem_l2123_212353


namespace NUMINAMATH_CALUDE_marcus_pebble_ratio_l2123_212395

def pebble_ratio (initial : ℕ) (received : ℕ) (final : ℕ) : Prop :=
  let skipped := initial + received - final
  (2 * skipped = initial) ∧ (skipped ≠ 0)

theorem marcus_pebble_ratio :
  pebble_ratio 18 30 39 := by
  sorry

end NUMINAMATH_CALUDE_marcus_pebble_ratio_l2123_212395


namespace NUMINAMATH_CALUDE_chapter_page_difference_l2123_212304

theorem chapter_page_difference (first_chapter_pages second_chapter_pages : ℕ) 
  (h1 : first_chapter_pages = 37)
  (h2 : second_chapter_pages = 80) : 
  second_chapter_pages - first_chapter_pages = 43 := by
sorry

end NUMINAMATH_CALUDE_chapter_page_difference_l2123_212304


namespace NUMINAMATH_CALUDE_smallest_n_with_perfect_square_sum_l2123_212366

def is_perfect_square (n : ℕ) : Prop := ∃ m : ℕ, n = m * m

def partition_with_perfect_square_sum (n : ℕ) : Prop :=
  ∀ (A B : Set ℕ),
    (A ∪ B = Finset.range n) →
    (A ∩ B = ∅) →
    (A ≠ ∅) →
    (B ≠ ∅) →
    (∃ (x y : ℕ), (x ∈ A ∧ y ∈ A ∧ is_perfect_square (x + y)) ∨
                  (x ∈ B ∧ y ∈ B ∧ is_perfect_square (x + y)))

theorem smallest_n_with_perfect_square_sum : 
  (∀ k < 15, ¬ partition_with_perfect_square_sum k) ∧ 
  partition_with_perfect_square_sum 15 :=
sorry

end NUMINAMATH_CALUDE_smallest_n_with_perfect_square_sum_l2123_212366


namespace NUMINAMATH_CALUDE_f_composition_value_l2123_212354

noncomputable def f (x : ℝ) : ℝ :=
  if x < 0 then Real.log (-x) else Real.tan x

theorem f_composition_value : f (f (3 * Real.pi / 4)) = 0 := by
  sorry

end NUMINAMATH_CALUDE_f_composition_value_l2123_212354


namespace NUMINAMATH_CALUDE_boys_equation_holds_l2123_212352

structure School where
  name : String
  total_students : ℕ

def calculate_boys (s : School) : ℚ :=
  s.total_students / (1 + s.total_students / 100)

theorem boys_equation_holds (s : School) :
  let x := calculate_boys s
  x + (x/100) * s.total_students = s.total_students :=
by sorry

def school_A : School := ⟨"A", 900⟩
def school_B : School := ⟨"B", 1200⟩
def school_C : School := ⟨"C", 1500⟩

#eval calculate_boys school_A
#eval calculate_boys school_B
#eval calculate_boys school_C

end NUMINAMATH_CALUDE_boys_equation_holds_l2123_212352


namespace NUMINAMATH_CALUDE_coloring_book_problem_l2123_212315

theorem coloring_book_problem (book1 : ℕ) (book2 : ℕ) (colored : ℕ) : 
  book1 = 23 → book2 = 32 → colored = 44 → 
  (book1 + book2) - colored = 11 := by
sorry

end NUMINAMATH_CALUDE_coloring_book_problem_l2123_212315


namespace NUMINAMATH_CALUDE_new_person_weight_l2123_212322

theorem new_person_weight (initial_total_weight : ℝ) : 
  let initial_avg := initial_total_weight / 10
  let new_avg := initial_avg + 5
  let new_total_weight := new_avg * 10
  new_total_weight - initial_total_weight + 60 = 110 := by
sorry

end NUMINAMATH_CALUDE_new_person_weight_l2123_212322


namespace NUMINAMATH_CALUDE_triangle_problem_l2123_212356

theorem triangle_problem (A B C : ℝ) (a b c : ℝ) :
  0 < A ∧ 0 < B ∧ 0 < C ∧ 0 < a ∧ 0 < b ∧ 0 < c →
  A + B + C = π →
  b * Real.cos A + (1/2) * a = c →
  (B = π/3 ∧
   (c = 5 → b = 7 → a = 8 ∧ (1/2) * a * c * Real.sin B = 10 * Real.sqrt 3) ∧
   (c = 5 → C = π/4 → a = (5 * Real.sqrt 3 + 5)/2 ∧ 
    (1/2) * a * c * Real.sin B = (75 + 25 * Real.sqrt 3)/8)) :=
by sorry

end NUMINAMATH_CALUDE_triangle_problem_l2123_212356


namespace NUMINAMATH_CALUDE_teacher_age_l2123_212301

theorem teacher_age (num_students : Nat) (student_avg_age : ℝ) (new_avg_age : ℝ) : 
  num_students = 30 →
  student_avg_age = 15 →
  new_avg_age = 16 →
  (num_students * student_avg_age + (num_students + 1) * new_avg_age - num_students * student_avg_age) = 46 := by
  sorry

end NUMINAMATH_CALUDE_teacher_age_l2123_212301


namespace NUMINAMATH_CALUDE_sqrt_50_between_consecutive_integers_product_l2123_212382

theorem sqrt_50_between_consecutive_integers_product : ∃ (n : ℕ), 
  n > 0 ∧ (n : ℝ) < Real.sqrt 50 ∧ Real.sqrt 50 < (n + 1) ∧ n * (n + 1) = 56 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_50_between_consecutive_integers_product_l2123_212382


namespace NUMINAMATH_CALUDE_chocolates_needed_to_fill_last_box_l2123_212380

def chocolates_per_box : ℕ := 30
def total_chocolates : ℕ := 254

theorem chocolates_needed_to_fill_last_box : 
  (chocolates_per_box - (total_chocolates % chocolates_per_box)) = 16 := by
  sorry

end NUMINAMATH_CALUDE_chocolates_needed_to_fill_last_box_l2123_212380


namespace NUMINAMATH_CALUDE_product_mod_23_l2123_212368

theorem product_mod_23 : (3001 * 3002 * 3003 * 3004 * 3005) % 23 = 9 := by
  sorry

end NUMINAMATH_CALUDE_product_mod_23_l2123_212368


namespace NUMINAMATH_CALUDE_xyz_minimum_l2123_212312

theorem xyz_minimum (x y z : ℝ) (h : x^3 + y^3 + z^3 - 3*x*y*z = 2) :
  ∀ (a b c : ℝ), a^3 + b^3 + c^3 - 3*a*b*c = 2 → x*y*z ≤ a*b*c := by
  sorry

end NUMINAMATH_CALUDE_xyz_minimum_l2123_212312


namespace NUMINAMATH_CALUDE_medical_team_selection_count_l2123_212359

theorem medical_team_selection_count : ∀ (m f k l : ℕ), 
  m = 6 → f = 5 → k = 2 → l = 1 →
  (m.choose k) * (f.choose l) = 75 :=
by sorry

end NUMINAMATH_CALUDE_medical_team_selection_count_l2123_212359


namespace NUMINAMATH_CALUDE_fraction_exponent_product_l2123_212399

theorem fraction_exponent_product : (1 / 3 : ℚ)^4 * (1 / 8 : ℚ) = 1 / 648 := by
  sorry

end NUMINAMATH_CALUDE_fraction_exponent_product_l2123_212399


namespace NUMINAMATH_CALUDE_sum_of_series_equals_three_fourths_l2123_212310

/-- The sum of the infinite series ∑(k=1 to ∞) k/3^k is equal to 3/4 -/
theorem sum_of_series_equals_three_fourths :
  (∑' k : ℕ, (k : ℝ) / 3^k) = 3/4 := by sorry

end NUMINAMATH_CALUDE_sum_of_series_equals_three_fourths_l2123_212310


namespace NUMINAMATH_CALUDE_good_pair_exists_l2123_212388

theorem good_pair_exists (m : ℕ) : ∃ n : ℕ, n > m ∧ 
  ∃ a b : ℕ, m * n = a^2 ∧ (m + 1) * (n + 1) = b^2 := by
  sorry

end NUMINAMATH_CALUDE_good_pair_exists_l2123_212388


namespace NUMINAMATH_CALUDE_smallest_square_area_l2123_212360

/-- Represents a rectangle with given width and height -/
structure Rectangle where
  width : ℕ
  height : ℕ

/-- Calculates the area of a square given its side length -/
def square_area (side : ℕ) : ℕ := side * side

/-- Checks if two rectangles can fit side by side within a given length -/
def can_fit_side_by_side (r1 r2 : Rectangle) (length : ℕ) : Prop :=
  r1.width + r2.width ≤ length ∨ r1.height + r2.height ≤ length

/-- Theorem: The smallest possible area of a square containing a 2×3 rectangle and a 4×5 rectangle
    without overlapping and with parallel sides is 49 square units -/
theorem smallest_square_area : ∃ (side : ℕ),
  let r1 : Rectangle := ⟨2, 3⟩
  let r2 : Rectangle := ⟨4, 5⟩
  (∀ (s : ℕ), s < side → ¬(can_fit_side_by_side r1 r2 s)) ∧
  (can_fit_side_by_side r1 r2 side) ∧
  (square_area side = 49) :=
sorry

end NUMINAMATH_CALUDE_smallest_square_area_l2123_212360


namespace NUMINAMATH_CALUDE_total_seeds_calculation_l2123_212323

/-- The number of rows of potatoes planted -/
def rows : ℕ := 6

/-- The number of seeds in each row -/
def seeds_per_row : ℕ := 9

/-- The total number of potato seeds planted -/
def total_seeds : ℕ := rows * seeds_per_row

theorem total_seeds_calculation : total_seeds = 54 := by
  sorry

end NUMINAMATH_CALUDE_total_seeds_calculation_l2123_212323


namespace NUMINAMATH_CALUDE_colored_dodecahedron_constructions_l2123_212334

/-- The number of faces in a dodecahedron -/
def num_faces : ℕ := 12

/-- The number of rotational symmetries considered for simplification -/
def rotational_symmetries : ℕ := 5

/-- The number of distinguishable ways to construct a colored dodecahedron -/
def distinguishable_constructions : ℕ := Nat.factorial (num_faces - 1) / rotational_symmetries

/-- Theorem stating the number of distinguishable ways to construct a colored dodecahedron -/
theorem colored_dodecahedron_constructions :
  distinguishable_constructions = 7983360 := by
  sorry

#eval distinguishable_constructions

end NUMINAMATH_CALUDE_colored_dodecahedron_constructions_l2123_212334


namespace NUMINAMATH_CALUDE_butterfly_collection_l2123_212311

/-- Given a collection of butterflies with specific conditions, prove the number of black butterflies. -/
theorem butterfly_collection (total : ℕ) (blue : ℕ) (yellow : ℕ) (black : ℕ)
  (h_total : total = 19)
  (h_blue : blue = 6)
  (h_yellow_ratio : yellow * 2 = blue)
  (h_sum : total = blue + yellow + black) :
  black = 10 := by
  sorry

end NUMINAMATH_CALUDE_butterfly_collection_l2123_212311


namespace NUMINAMATH_CALUDE_quadratic_opens_upwards_l2123_212305

/-- A quadratic function f(x) = ax² + bx + c -/
noncomputable def f (a b c : ℝ) (x : ℝ) : ℝ := a * x^2 + b * x + c

/-- The quadratic function opens upwards if a > 0 -/
def opens_upwards (a b c : ℝ) : Prop := a > 0

theorem quadratic_opens_upwards (a b c : ℝ) 
  (h1 : f a b c (-1) = 10)
  (h2 : f a b c 0 = 5)
  (h3 : f a b c 1 = 2)
  (h4 : f a b c 2 = 1)
  (h5 : f a b c 3 = 2) :
  opens_upwards a b c := by sorry

end NUMINAMATH_CALUDE_quadratic_opens_upwards_l2123_212305


namespace NUMINAMATH_CALUDE_geometric_sequence_a10_l2123_212338

def geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = r * a n

theorem geometric_sequence_a10 (a : ℕ → ℝ) 
  (h_geom : geometric_sequence a) 
  (h_a2 : a 2 = 2) 
  (h_a6 : a 6 = 162) : 
  a 10 = 13122 := by
sorry

end NUMINAMATH_CALUDE_geometric_sequence_a10_l2123_212338


namespace NUMINAMATH_CALUDE_simplify_expression_l2123_212393

theorem simplify_expression (x : ℝ) : 3*x + 2*x^2 + 5*x - x^2 + 7 = x^2 + 8*x + 7 := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l2123_212393


namespace NUMINAMATH_CALUDE_distinct_fraction_equality_l2123_212343

theorem distinct_fraction_equality (a b c : ℝ) (k : ℝ) : 
  a ≠ b ∧ b ≠ c ∧ a ≠ c →
  a / (1 + b) = k ∧ b / (1 + c) = k ∧ c / (1 + a) = k →
  k = -1 :=
sorry

end NUMINAMATH_CALUDE_distinct_fraction_equality_l2123_212343


namespace NUMINAMATH_CALUDE_wrong_number_correction_l2123_212333

theorem wrong_number_correction (n : ℕ) (initial_avg correct_avg : ℚ) 
  (first_error second_correct : ℤ) : 
  n = 10 → 
  initial_avg = 40.2 → 
  correct_avg = 40.3 → 
  first_error = 19 → 
  second_correct = 31 → 
  ∃ (second_error : ℤ), 
    (n : ℚ) * initial_avg - first_error - second_error + second_correct = (n : ℚ) * correct_avg ∧ 
    second_error = 11 := by
  sorry

end NUMINAMATH_CALUDE_wrong_number_correction_l2123_212333


namespace NUMINAMATH_CALUDE_sqrt_7_irrational_l2123_212314

theorem sqrt_7_irrational : ¬ ∃ (p q : ℕ), p > 0 ∧ q > 0 ∧ (p : ℚ) ^ 2 / (q : ℚ) ^ 2 = 7 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_7_irrational_l2123_212314


namespace NUMINAMATH_CALUDE_polynomial_coefficient_properties_l2123_212373

theorem polynomial_coefficient_properties (a : Fin 6 → ℝ) :
  (∀ x : ℝ, x^5 = a 0 + a 1 * (1 - x) + a 2 * (1 - x)^2 + a 3 * (1 - x)^3 + a 4 * (1 - x)^4 + a 5 * (1 - x)^5) →
  (a 3 = -10 ∧ a 1 + a 3 + a 5 = -16) := by
  sorry

end NUMINAMATH_CALUDE_polynomial_coefficient_properties_l2123_212373


namespace NUMINAMATH_CALUDE_order_of_expressions_l2123_212351

theorem order_of_expressions : 3^(1/2) > Real.log (1/2) / Real.log (1/3) ∧ 
  Real.log (1/2) / Real.log (1/3) > Real.log (1/3) / Real.log 2 := by
  sorry

end NUMINAMATH_CALUDE_order_of_expressions_l2123_212351


namespace NUMINAMATH_CALUDE_hexagon_percentage_l2123_212309

-- Define the tiling structure
structure Tiling where
  smallSquareArea : ℝ
  largeSquareArea : ℝ
  hexagonArea : ℝ
  smallSquaresPerLarge : ℕ
  hexagonsPerLarge : ℕ
  smallSquaresInHexagons : ℝ

-- Define the tiling conditions
def tilingConditions (t : Tiling) : Prop :=
  t.smallSquaresPerLarge = 16 ∧
  t.hexagonsPerLarge = 3 ∧
  t.largeSquareArea = 16 * t.smallSquareArea ∧
  t.hexagonArea = 2 * t.smallSquareArea ∧
  t.smallSquaresInHexagons = 3 * t.hexagonArea

-- Theorem to prove
theorem hexagon_percentage (t : Tiling) (h : tilingConditions t) :
  (t.smallSquaresInHexagons / t.largeSquareArea) * 100 = 37.5 := by
  sorry

end NUMINAMATH_CALUDE_hexagon_percentage_l2123_212309


namespace NUMINAMATH_CALUDE_trip_length_satisfies_conditions_l2123_212316

/-- Represents the total trip length in miles -/
def total_trip_length : ℝ := 180

/-- Represents the distance traveled on battery power in miles -/
def battery_distance : ℝ := 60

/-- Represents the fuel consumption rate in gallons per mile when using gasoline -/
def fuel_consumption_rate : ℝ := 0.03

/-- Represents the average fuel efficiency for the entire trip in miles per gallon -/
def average_fuel_efficiency : ℝ := 50

/-- Theorem stating that the total trip length satisfies the given conditions -/
theorem trip_length_satisfies_conditions :
  (total_trip_length / (fuel_consumption_rate * (total_trip_length - battery_distance)) = average_fuel_efficiency) ∧
  (total_trip_length > battery_distance) := by
  sorry

#check trip_length_satisfies_conditions

end NUMINAMATH_CALUDE_trip_length_satisfies_conditions_l2123_212316


namespace NUMINAMATH_CALUDE_number_properties_l2123_212396

def number : ℤ := 2023

theorem number_properties :
  (- number = -2023) ∧
  ((1 : ℚ) / number = 1 / 2023) ∧
  (|number| = 2023) := by
  sorry

end NUMINAMATH_CALUDE_number_properties_l2123_212396


namespace NUMINAMATH_CALUDE_hyperbola_eccentricity_l2123_212345

/-- The eccentricity of a hyperbola with equation mx² + y² = 1 and eccentricity √2 is -1 -/
theorem hyperbola_eccentricity (m : ℝ) : 
  (∀ x y : ℝ, m * x^2 + y^2 = 1) →  -- Equation of the hyperbola
  (∃ a c : ℝ, a > 0 ∧ c > a ∧ (c/a)^2 = 2) →  -- Eccentricity is √2
  m = -1 :=
by sorry

end NUMINAMATH_CALUDE_hyperbola_eccentricity_l2123_212345


namespace NUMINAMATH_CALUDE_last_digit_of_power_l2123_212321

theorem last_digit_of_power (a b : ℕ) (ha : a = 954950230952380948328708) (hb : b = 470128749397540235934750230) :
  (a^b) % 10 = 4 := by
  sorry

end NUMINAMATH_CALUDE_last_digit_of_power_l2123_212321


namespace NUMINAMATH_CALUDE_triangle_area_l2123_212372

theorem triangle_area (a c : ℝ) (A : ℝ) (h1 : a = 2) (h2 : c = 2 * Real.sqrt 3) (h3 : A = π / 6) :
  ∃ (area : ℝ), (area = 2 * Real.sqrt 3 ∨ area = Real.sqrt 3) ∧
  ∃ (B C : ℝ), 0 ≤ B ∧ B < 2 * π ∧ 0 ≤ C ∧ C < 2 * π ∧
  A + B + C = π ∧
  area = (1 / 2) * a * c * Real.sin B :=
by sorry

end NUMINAMATH_CALUDE_triangle_area_l2123_212372


namespace NUMINAMATH_CALUDE_cos_pi_4_plus_x_eq_4_5_l2123_212331

theorem cos_pi_4_plus_x_eq_4_5 (x : ℝ) (h1 : x ∈ Set.Ioo (-π/2) (-π/4)) 
  (h2 : Real.cos (π/4 + x) = 4/5) : 
  (Real.sin (2*x) - 2*(Real.sin x)^2) / (1 + Real.tan x) = 28/75 := by
  sorry

end NUMINAMATH_CALUDE_cos_pi_4_plus_x_eq_4_5_l2123_212331


namespace NUMINAMATH_CALUDE_min_value_mn_l2123_212319

def f (a x : ℝ) : ℝ := |x - a|

theorem min_value_mn (a m n : ℝ) : 
  (∀ x, f a x ≤ 1 ↔ 0 ≤ x ∧ x ≤ 2) →
  m > 0 →
  n > 0 →
  1/m + 1/(2*n) = a →
  ∀ k, m * n ≤ k → 2 ≤ k :=
by sorry

end NUMINAMATH_CALUDE_min_value_mn_l2123_212319


namespace NUMINAMATH_CALUDE_remainder_problem_l2123_212335

theorem remainder_problem (m : ℤ) (h : m % 5 = 3) : (4 * m + 5) % 5 = 2 := by
  sorry

end NUMINAMATH_CALUDE_remainder_problem_l2123_212335


namespace NUMINAMATH_CALUDE_fraction_simplification_l2123_212339

theorem fraction_simplification (y : ℝ) (h : y = 3) : 
  (y^8 + 18*y^4 + 81) / (y^4 + 9) = 90 := by
sorry

end NUMINAMATH_CALUDE_fraction_simplification_l2123_212339


namespace NUMINAMATH_CALUDE_handshake_problem_l2123_212337

theorem handshake_problem (n : ℕ) (total_handshakes : ℕ) 
  (h1 : n = 12) 
  (h2 : total_handshakes = 66) : 
  total_handshakes = n * (n - 1) / 2 ∧ 
  (total_handshakes / (n - 1) : ℚ) = 6 := by
sorry

end NUMINAMATH_CALUDE_handshake_problem_l2123_212337


namespace NUMINAMATH_CALUDE_expression_evaluation_l2123_212397

theorem expression_evaluation : 
  let x : ℤ := -2
  (-x^2 + 5 + 4*x) + (5*x - 4 + 2*x^2) = -13 :=
by sorry

end NUMINAMATH_CALUDE_expression_evaluation_l2123_212397


namespace NUMINAMATH_CALUDE_multiply_polynomials_l2123_212325

theorem multiply_polynomials (x : ℝ) : 
  (x^4 + 8*x^2 + 16) * (x^2 - 4) = x^4 + 8*x^2 + 12 := by
sorry

end NUMINAMATH_CALUDE_multiply_polynomials_l2123_212325


namespace NUMINAMATH_CALUDE_money_distribution_l2123_212364

theorem money_distribution (a b c : ℤ) 
  (total : a + b + c = 500)
  (ac_sum : a + c = 200)
  (bc_sum : b + c = 340) :
  c = 40 := by
  sorry

end NUMINAMATH_CALUDE_money_distribution_l2123_212364


namespace NUMINAMATH_CALUDE_bonaparte_execution_l2123_212378

def assassination_attempt (executed : ℕ) (killed : ℕ) (injured : ℕ) (deported : ℕ) : Prop :=
  injured = 2 * killed + (4 * executed) / 3 ∧
  killed + injured + executed < deported ∧
  killed = 2 * executed + 4 ∧
  deported = 98 ∧
  executed % 3 = 0 ∧
  executed < 11

theorem bonaparte_execution :
  ∃ (executed killed injured : ℕ),
    assassination_attempt executed killed injured 98 ∧
    executed = 9 := by sorry

end NUMINAMATH_CALUDE_bonaparte_execution_l2123_212378


namespace NUMINAMATH_CALUDE_f_satisfies_conditions_l2123_212350

open Complex

/-- The analytic function f(z) that satisfies the given conditions -/
noncomputable def f (z : ℂ) : ℂ := z^3 - 2*I*z + (2 + 3*I)

/-- The real part of f(z) -/
def u (x y : ℝ) : ℝ := x^3 - 3*x*y^2 + 2*y

theorem f_satisfies_conditions :
  (∀ x y : ℝ, (f (x + y*I)).re = u x y) ∧
  f I = 2 := by sorry

end NUMINAMATH_CALUDE_f_satisfies_conditions_l2123_212350


namespace NUMINAMATH_CALUDE_angle_q_sum_of_sin_cos_l2123_212302

theorem angle_q_sum_of_sin_cos (x : ℝ) (hx : x ≠ 0) :
  let P : ℝ × ℝ := (x, -1)
  let tan_q : ℝ := -x
  let sin_q : ℝ := -1 / Real.sqrt (1 + x^2)
  let cos_q : ℝ := x / Real.sqrt (1 + x^2)
  (sin_q + cos_q = 0) ∨ (sin_q + cos_q = -Real.sqrt 2) :=
by sorry

end NUMINAMATH_CALUDE_angle_q_sum_of_sin_cos_l2123_212302


namespace NUMINAMATH_CALUDE_sum_with_radical_conjugate_l2123_212300

-- Define the radical conjugate
def radical_conjugate (a b : ℝ) : ℝ := a - b

-- Theorem statement
theorem sum_with_radical_conjugate :
  let x : ℝ := 15 - Real.sqrt 500
  let y : ℝ := radical_conjugate 15 (Real.sqrt 500)
  x + y = 30 := by
  sorry

end NUMINAMATH_CALUDE_sum_with_radical_conjugate_l2123_212300


namespace NUMINAMATH_CALUDE_multiples_of_ten_l2123_212348

theorem multiples_of_ten (n : ℕ) : 
  100 + (n - 1) * 10 = 10000 ↔ n = 991 :=
by sorry

#check multiples_of_ten

end NUMINAMATH_CALUDE_multiples_of_ten_l2123_212348


namespace NUMINAMATH_CALUDE_simplify_and_evaluate_l2123_212386

theorem simplify_and_evaluate (a : ℤ) : 
  2 * (4 * a ^ 2 - a) - (3 * a ^ 2 - 2 * a + 5) = 40 ↔ a = -3 :=
by sorry

end NUMINAMATH_CALUDE_simplify_and_evaluate_l2123_212386


namespace NUMINAMATH_CALUDE_fraction_of_product_l2123_212369

theorem fraction_of_product (x : ℚ) : x * ((3 / 4 : ℚ) * (2 / 5 : ℚ) * 5040) = 756.0000000000001 → x = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_fraction_of_product_l2123_212369


namespace NUMINAMATH_CALUDE_monotonic_sequence_divisor_property_l2123_212308

def divisor_count (n : ℕ) : ℕ := (Finset.filter (· ∣ n) (Finset.range (n + 1))).card

def is_monotonic_increasing (a : ℕ → ℕ) : Prop :=
  ∀ i j : ℕ, i < j → a i < a j

theorem monotonic_sequence_divisor_property (a : ℕ → ℕ) :
  is_monotonic_increasing a →
  (∀ i j : ℕ, divisor_count (i + j) = divisor_count (a i + a j)) →
  ∀ n : ℕ, a n = n :=
sorry

end NUMINAMATH_CALUDE_monotonic_sequence_divisor_property_l2123_212308


namespace NUMINAMATH_CALUDE_function_comparison_l2123_212379

-- Define the function f
variable (f : ℝ → ℝ)

-- Define the properties of f
variable (h_even : ∀ x, f x = f (-x))
variable (h_decreasing : ∀ x y, x < y → y < 0 → f x > f y)

-- Define the theorem
theorem function_comparison (x₁ x₂ : ℝ) 
  (h₁ : x₁ < 0) (h₂ : x₁ + x₂ > 0) : f x₁ < f x₂ := by
  sorry

end NUMINAMATH_CALUDE_function_comparison_l2123_212379


namespace NUMINAMATH_CALUDE_new_person_weight_l2123_212349

theorem new_person_weight (initial_count : ℕ) (weight_increase : ℝ) (replaced_weight : ℝ) :
  initial_count = 10 →
  weight_increase = 4 →
  replaced_weight = 70 →
  ∃ (new_weight : ℝ),
    new_weight = initial_count * weight_increase + replaced_weight ∧
    new_weight = 110 := by
  sorry

end NUMINAMATH_CALUDE_new_person_weight_l2123_212349


namespace NUMINAMATH_CALUDE_remainder_77_pow_77_minus_15_mod_19_l2123_212313

theorem remainder_77_pow_77_minus_15_mod_19 : 77^77 - 15 ≡ 5 [MOD 19] := by
  sorry

end NUMINAMATH_CALUDE_remainder_77_pow_77_minus_15_mod_19_l2123_212313


namespace NUMINAMATH_CALUDE_complement_of_union_l2123_212355

def U : Finset ℕ := {1, 3, 5, 9}
def A : Finset ℕ := {1, 3, 9}
def B : Finset ℕ := {1, 9}

theorem complement_of_union :
  (U \ (A ∪ B)) = {5} := by sorry

end NUMINAMATH_CALUDE_complement_of_union_l2123_212355


namespace NUMINAMATH_CALUDE_pizza_piece_cost_l2123_212371

/-- Given that Luigi bought 4 pizzas for $80 and each pizza was cut into 5 pieces,
    prove that the cost of each pizza piece is $4. -/
theorem pizza_piece_cost (total_pizzas : ℕ) (total_cost : ℚ) (pieces_per_pizza : ℕ) :
  total_pizzas = 4 →
  total_cost = 80 →
  pieces_per_pizza = 5 →
  total_cost / (total_pizzas * pieces_per_pizza : ℚ) = 4 := by
  sorry

end NUMINAMATH_CALUDE_pizza_piece_cost_l2123_212371


namespace NUMINAMATH_CALUDE_sophies_daily_oranges_l2123_212376

/-- The number of oranges Sophie's mom gives her every day -/
def sophies_oranges : ℕ := 20

/-- The number of grapes Hannah eats per day -/
def hannahs_grapes : ℕ := 40

/-- The number of days in the observation period -/
def observation_days : ℕ := 30

/-- The total number of fruits eaten by Sophie and Hannah during the observation period -/
def total_fruits : ℕ := 1800

/-- Theorem stating that Sophie's mom gives her 20 oranges per day -/
theorem sophies_daily_oranges :
  sophies_oranges * observation_days + hannahs_grapes * observation_days = total_fruits :=
by sorry

end NUMINAMATH_CALUDE_sophies_daily_oranges_l2123_212376


namespace NUMINAMATH_CALUDE_even_quadratic_max_value_l2123_212362

/-- A quadratic function f(x) = ax^2 + bx + 1 defined on [-1-a, 2a] -/
def f (a b : ℝ) (x : ℝ) : ℝ := a * x^2 + b * x + 1

/-- The domain of the function -/
def domain (a : ℝ) : Set ℝ := Set.Icc (-1 - a) (2 * a)

/-- Theorem: If f is even on its domain, its maximum value is 5 -/
theorem even_quadratic_max_value (a b : ℝ) :
  (∀ x ∈ domain a, f a b x = f a b (-x)) →
  (∃ x ∈ domain a, ∀ y ∈ domain a, f a b y ≤ f a b x) →
  (∃ x ∈ domain a, f a b x = 5) :=
sorry

end NUMINAMATH_CALUDE_even_quadratic_max_value_l2123_212362


namespace NUMINAMATH_CALUDE_jakes_third_test_score_l2123_212342

/-- Proof that Jake scored 65 marks in the third test given the conditions -/
theorem jakes_third_test_score :
  ∀ (third_test_score : ℕ),
  (∃ (first_test : ℕ) (second_test : ℕ) (fourth_test : ℕ),
    first_test = 80 ∧
    second_test = first_test + 10 ∧
    fourth_test = third_test_score ∧
    (first_test + second_test + third_test_score + fourth_test) / 4 = 75) →
  third_test_score = 65 := by
sorry

end NUMINAMATH_CALUDE_jakes_third_test_score_l2123_212342


namespace NUMINAMATH_CALUDE_increasing_function_condition_l2123_212327

/-- A function f(x) = (1/2)mx^2 + ln x - 2x is increasing on its domain (x > 0) if and only if m ≥ 1 -/
theorem increasing_function_condition (m : ℝ) :
  (∀ x > 0, Monotone (fun x => (1/2) * m * x^2 + Real.log x - 2*x)) ↔ m ≥ 1 := by
  sorry

end NUMINAMATH_CALUDE_increasing_function_condition_l2123_212327


namespace NUMINAMATH_CALUDE_shaded_area_theorem_l2123_212324

def U : Set Nat := {1,2,3,4,5,6,7,8}
def M : Set Nat := {1,3,5,7}
def N : Set Nat := {5,6,7}

theorem shaded_area_theorem : U \ (M ∪ N) = {2,4,8} := by sorry

end NUMINAMATH_CALUDE_shaded_area_theorem_l2123_212324


namespace NUMINAMATH_CALUDE_division_multiplication_problem_l2123_212317

theorem division_multiplication_problem : (180 / 6) / 3 * 2 = 20 := by
  sorry

end NUMINAMATH_CALUDE_division_multiplication_problem_l2123_212317


namespace NUMINAMATH_CALUDE_magnitude_of_complex_number_l2123_212390

theorem magnitude_of_complex_number (z : ℂ) : z = Complex.I * (3 + 4 * Complex.I) → Complex.abs z = 5 := by
  sorry

end NUMINAMATH_CALUDE_magnitude_of_complex_number_l2123_212390


namespace NUMINAMATH_CALUDE_solution_volume_l2123_212361

/-- Given two solutions, one of 6 litres and another of V litres, 
    if 20% of the first solution is mixed with 60% of the second solution,
    and the resulting mixture is 36% of the total volume,
    then V equals 4 litres. -/
theorem solution_volume (V : ℝ) : 
  (0.2 * 6 + 0.6 * V) / (6 + V) = 0.36 → V = 4 := by
  sorry

end NUMINAMATH_CALUDE_solution_volume_l2123_212361


namespace NUMINAMATH_CALUDE_exponent_multiplication_l2123_212394

theorem exponent_multiplication (a : ℝ) : a^2 * a^3 = a^5 := by
  sorry

end NUMINAMATH_CALUDE_exponent_multiplication_l2123_212394


namespace NUMINAMATH_CALUDE_quadratic_inequality_range_l2123_212306

theorem quadratic_inequality_range : 
  ∀ x : ℝ, x^2 - 7*x + 12 < 0 → 
  ∃ y : ℝ, y ∈ Set.Ioo 42 56 ∧ y = x^2 + 7*x + 12 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_inequality_range_l2123_212306


namespace NUMINAMATH_CALUDE_five_digit_divisible_by_nine_l2123_212318

theorem five_digit_divisible_by_nine :
  ∃! d : ℕ, d < 10 ∧ (34700 + 10 * d + 9) % 9 = 0 := by
  sorry

end NUMINAMATH_CALUDE_five_digit_divisible_by_nine_l2123_212318


namespace NUMINAMATH_CALUDE_field_area_proof_l2123_212398

theorem field_area_proof (smaller_area larger_area : ℝ) : 
  smaller_area = 405 →
  larger_area - smaller_area = (1 / 5) * ((smaller_area + larger_area) / 2) →
  smaller_area + larger_area = 900 := by
sorry

end NUMINAMATH_CALUDE_field_area_proof_l2123_212398


namespace NUMINAMATH_CALUDE_ursula_hourly_wage_l2123_212389

/-- Calculates the hourly wage given annual salary and working hours --/
def hourly_wage (annual_salary : ℕ) (hours_per_day : ℕ) (days_per_month : ℕ) : ℚ :=
  (annual_salary : ℚ) / (12 * hours_per_day * days_per_month)

/-- Proves that Ursula's hourly wage is $8.50 given her work conditions --/
theorem ursula_hourly_wage :
  hourly_wage 16320 8 20 = 17/2 := by
  sorry

end NUMINAMATH_CALUDE_ursula_hourly_wage_l2123_212389


namespace NUMINAMATH_CALUDE_ball_placement_count_ball_placement_count_is_30_l2123_212385

/-- Number of ways to place 4 balls in 3 boxes with constraints -/
theorem ball_placement_count : ℕ :=
  let total_balls : ℕ := 4
  let num_boxes : ℕ := 3
  let ways_to_choose_two : ℕ := Nat.choose total_balls 2
  let ways_to_arrange_three : ℕ := Nat.factorial num_boxes
  let invalid_arrangements : ℕ := 6
  ways_to_choose_two * ways_to_arrange_three - invalid_arrangements

/-- Proof that the number of valid arrangements is 30 -/
theorem ball_placement_count_is_30 : ball_placement_count = 30 := by
  sorry

end NUMINAMATH_CALUDE_ball_placement_count_ball_placement_count_is_30_l2123_212385


namespace NUMINAMATH_CALUDE_solve_cubic_equation_l2123_212363

theorem solve_cubic_equation (m : ℝ) : (m - 3)^3 = (1/27)⁻¹ → m = 6 := by
  sorry

end NUMINAMATH_CALUDE_solve_cubic_equation_l2123_212363


namespace NUMINAMATH_CALUDE_range_of_3a_minus_2b_l2123_212328

theorem range_of_3a_minus_2b (a b : ℝ) 
  (h1 : 1 ≤ a - b) (h2 : a - b ≤ 2) 
  (h3 : 2 ≤ a + b) (h4 : a + b ≤ 4) : 
  ∃ (x : ℝ), (7/2 ≤ x ∧ x ≤ 7) ∧ (∃ (a' b' : ℝ), 
    (1 ≤ a' - b' ∧ a' - b' ≤ 2) ∧ 
    (2 ≤ a' + b' ∧ a' + b' ≤ 4) ∧ 
    (3 * a' - 2 * b' = x)) ∧
  (∀ (y : ℝ), (∃ (a'' b'' : ℝ), 
    (1 ≤ a'' - b'' ∧ a'' - b'' ≤ 2) ∧ 
    (2 ≤ a'' + b'' ∧ a'' + b'' ≤ 4) ∧ 
    (3 * a'' - 2 * b'' = y)) → 
    (7/2 ≤ y ∧ y ≤ 7)) := by
  sorry


end NUMINAMATH_CALUDE_range_of_3a_minus_2b_l2123_212328


namespace NUMINAMATH_CALUDE_arithmetic_sequence_property_l2123_212381

/-- An arithmetic sequence with specific properties -/
def ArithmeticSequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_property (a : ℕ → ℝ) 
  (h1 : ArithmeticSequence a) 
  (h2 : a 1 = 2) 
  (h3 : a 3 + a 5 = 10) : 
  a 7 = 8 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_property_l2123_212381


namespace NUMINAMATH_CALUDE_gcd_factorial_eight_and_factorial_six_squared_l2123_212357

theorem gcd_factorial_eight_and_factorial_six_squared :
  Nat.gcd (Nat.factorial 8) ((Nat.factorial 6)^2) = 7200 := by
  sorry

end NUMINAMATH_CALUDE_gcd_factorial_eight_and_factorial_six_squared_l2123_212357


namespace NUMINAMATH_CALUDE_george_amelia_apple_difference_l2123_212377

/-- Proves the difference in apple count between George and Amelia -/
theorem george_amelia_apple_difference :
  ∀ (george_oranges george_apples amelia_oranges amelia_apples : ℕ),
  george_oranges = 45 →
  amelia_oranges = george_oranges - 18 →
  amelia_apples = 15 →
  george_apples > amelia_apples →
  george_oranges + george_apples + amelia_oranges + amelia_apples = 107 →
  george_apples - amelia_apples = 5 := by
sorry

end NUMINAMATH_CALUDE_george_amelia_apple_difference_l2123_212377


namespace NUMINAMATH_CALUDE_factor_expression_l2123_212391

theorem factor_expression (x y : ℝ) : 231 * x^2 * y + 33 * x * y = 33 * x * y * (7 * x + 1) := by
  sorry

end NUMINAMATH_CALUDE_factor_expression_l2123_212391


namespace NUMINAMATH_CALUDE_gcd_n_cube_plus_27_and_n_plus_3_l2123_212375

theorem gcd_n_cube_plus_27_and_n_plus_3 (n : ℕ) (h : n > 9) :
  Nat.gcd (n^3 + 27) (n + 3) = n + 3 := by
  sorry

end NUMINAMATH_CALUDE_gcd_n_cube_plus_27_and_n_plus_3_l2123_212375


namespace NUMINAMATH_CALUDE_system_solution_unique_l2123_212358

theorem system_solution_unique :
  ∃! (x y : ℝ), 2 * x + 3 * y = 7 ∧ 4 * x - 3 * y = 5 :=
by
  -- Proof goes here
  sorry

end NUMINAMATH_CALUDE_system_solution_unique_l2123_212358


namespace NUMINAMATH_CALUDE_inscribed_squares_ratio_l2123_212330

/-- A right triangle with sides 6, 8, and 10 -/
structure RightTriangle where
  a : ℝ
  b : ℝ
  c : ℝ
  right_triangle : a^2 + b^2 = c^2
  sides : a = 6 ∧ b = 8 ∧ c = 10

/-- A square inscribed in the triangle with side along leg of length 6 -/
def inscribed_square_x (t : RightTriangle) (x : ℝ) : Prop :=
  x > 0 ∧ x ≤ t.a ∧ x / t.a = (t.b - x) / t.c

/-- A square inscribed in the triangle with side along leg of length 8 -/
def inscribed_square_y (t : RightTriangle) (y : ℝ) : Prop :=
  y > 0 ∧ y ≤ t.b ∧ y / t.b = (t.a - y) / t.c

theorem inscribed_squares_ratio (t : RightTriangle) (x y : ℝ) 
  (hx : inscribed_square_x t x) (hy : inscribed_square_y t y) : 
  x / y = 9 / 8 := by
  sorry

end NUMINAMATH_CALUDE_inscribed_squares_ratio_l2123_212330


namespace NUMINAMATH_CALUDE_initial_mean_calculation_l2123_212320

theorem initial_mean_calculation (n : ℕ) (wrong_value correct_value : ℝ) (corrected_mean : ℝ) :
  n = 50 ∧ 
  wrong_value = 21 ∧ 
  correct_value = 48 ∧ 
  corrected_mean = 36.54 →
  ∃ initial_mean : ℝ, 
    initial_mean * n + (correct_value - wrong_value) = corrected_mean * n ∧ 
    initial_mean = 36 :=
by sorry

end NUMINAMATH_CALUDE_initial_mean_calculation_l2123_212320


namespace NUMINAMATH_CALUDE_certain_number_proof_l2123_212326

/-- The certain number that, when multiplied by the smallest positive integer a
    that makes the product a square, equals 14 -/
def certain_number : ℕ := 14

theorem certain_number_proof (a : ℕ) (h1 : a = 14) 
  (h2 : ∀ k : ℕ, k < a → ¬∃ m : ℕ, k * certain_number = m * m) 
  (h3 : ∃ m : ℕ, a * certain_number = m * m) : 
  certain_number = 14 := by sorry

end NUMINAMATH_CALUDE_certain_number_proof_l2123_212326


namespace NUMINAMATH_CALUDE_angle_bisector_sum_l2123_212392

-- Define the triangle vertices
def P : ℝ × ℝ := (-8, 5)
def Q : ℝ × ℝ := (-15, -19)
def R : ℝ × ℝ := (1, -7)

-- Define the equation of the angle bisector
def angle_bisector_equation (a c : ℝ) (x y : ℝ) : Prop :=
  a * x + 2 * y + c = 0

-- Theorem statement
theorem angle_bisector_sum (a c : ℝ) :
  (∃ x y, angle_bisector_equation a c x y ∧
          (x, y) ≠ P ∧
          (∃ t : ℝ, (1 - t) • P + t • Q = (x, y) ∨ (1 - t) • P + t • R = (x, y))) →
  a + c = 89 :=
sorry

end NUMINAMATH_CALUDE_angle_bisector_sum_l2123_212392

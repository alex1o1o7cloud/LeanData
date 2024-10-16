import Mathlib

namespace NUMINAMATH_CALUDE_hemisphere_surface_area_l468_46819

theorem hemisphere_surface_area (diameter : ℝ) (h : diameter = 12) :
  let radius := diameter / 2
  let curved_surface_area := 2 * π * radius^2
  let base_area := π * radius^2
  curved_surface_area + base_area = 108 * π :=
by sorry

end NUMINAMATH_CALUDE_hemisphere_surface_area_l468_46819


namespace NUMINAMATH_CALUDE_isosceles_triangle_perimeter_l468_46828

/-- An isosceles triangle with two sides of length 9 and one side of length 2 has a perimeter of 20. -/
theorem isosceles_triangle_perimeter :
  ∀ (a b c : ℝ), 
    a = 9 → b = 9 → c = 2 →
    (a + b > c) ∧ (a + c > b) ∧ (b + c > a) →  -- Triangle inequality
    a = b →  -- Isosceles condition
    a + b + c = 20 := by
  sorry

end NUMINAMATH_CALUDE_isosceles_triangle_perimeter_l468_46828


namespace NUMINAMATH_CALUDE_shipping_cost_shipping_cost_cents_shipping_cost_proof_l468_46838

/-- Shipping cost calculation for a book -/
theorem shipping_cost (G : ℝ) : ℝ :=
  8 * ⌈G / 100⌉

/-- The shipping cost in cents for a book weighing G grams -/
theorem shipping_cost_cents (G : ℝ) : ℝ :=
  shipping_cost G

/-- Proof that the shipping cost in cents is equal to 8 * ⌈G / 100⌉ -/
theorem shipping_cost_proof (G : ℝ) : shipping_cost_cents G = 8 * ⌈G / 100⌉ := by
  sorry

end NUMINAMATH_CALUDE_shipping_cost_shipping_cost_cents_shipping_cost_proof_l468_46838


namespace NUMINAMATH_CALUDE_parallel_iff_a_eq_3_l468_46818

/-- Two lines are parallel if and only if their slopes are equal -/
def are_parallel (m1 n1 c1 m2 n2 c2 : ℝ) : Prop :=
  m1 * n2 = m2 * n1

/-- The first line: ax + 2y + 3a = 0 -/
def line1 (a : ℝ) (x y : ℝ) : Prop :=
  a * x + 2 * y + 3 * a = 0

/-- The second line: 3x + (a-1)y = a-7 -/
def line2 (a : ℝ) (x y : ℝ) : Prop :=
  3 * x + (a - 1) * y = a - 7

theorem parallel_iff_a_eq_3 :
  ∀ a : ℝ, are_parallel a 2 (3*a) 3 (a-1) (a-7) ↔ a = 3 :=
by sorry

end NUMINAMATH_CALUDE_parallel_iff_a_eq_3_l468_46818


namespace NUMINAMATH_CALUDE_clarissa_photos_eq_14_l468_46865

/-- The number of slots in the photo album -/
def total_slots : ℕ := 40

/-- The number of photos Cristina brings -/
def cristina_photos : ℕ := 7

/-- The number of photos John brings -/
def john_photos : ℕ := 10

/-- The number of photos Sarah brings -/
def sarah_photos : ℕ := 9

/-- The number of photos Clarissa needs to bring -/
def clarissa_photos : ℕ := total_slots - (cristina_photos + john_photos + sarah_photos)

theorem clarissa_photos_eq_14 : clarissa_photos = 14 := by
  sorry

end NUMINAMATH_CALUDE_clarissa_photos_eq_14_l468_46865


namespace NUMINAMATH_CALUDE_base16_A987B_bits_bits_count_A987B_l468_46883

def base16_to_decimal (n : String) : ℕ :=
  -- Implementation details omitted
  sorry

theorem base16_A987B_bits : 
  let decimal := base16_to_decimal "A987B"
  2^19 ≤ decimal ∧ decimal < 2^20 := by
  sorry

theorem bits_count_A987B : 
  (Nat.log 2 (base16_to_decimal "A987B") + 1 : ℕ) = 20 := by
  sorry

end NUMINAMATH_CALUDE_base16_A987B_bits_bits_count_A987B_l468_46883


namespace NUMINAMATH_CALUDE_solution_part1_solution_part2_l468_46813

-- Define the function f
def f (x a : ℝ) : ℝ := |x - 1| + |x - a|

-- Theorem for part 1
theorem solution_part1 :
  {x : ℝ | f x (-1) ≥ 3} = {x : ℝ | x ≤ -3/2 ∨ x ≥ 3/2} := by sorry

-- Theorem for part 2
theorem solution_part2 :
  {a : ℝ | ∀ x, f x a ≥ 2} = {a : ℝ | a ≤ -1 ∨ a ≥ 3} := by sorry

end NUMINAMATH_CALUDE_solution_part1_solution_part2_l468_46813


namespace NUMINAMATH_CALUDE_polynomial_simplification_l468_46873

theorem polynomial_simplification :
  ∀ x : ℝ, 3 - 5*x - 7*x^2 + 9 + 11*x - 13*x^2 - 15 + 17*x + 19*x^2 = -x^2 + 23*x - 3 :=
by
  sorry

end NUMINAMATH_CALUDE_polynomial_simplification_l468_46873


namespace NUMINAMATH_CALUDE_max_cylinder_radius_in_crate_l468_46877

/-- A rectangular crate with given dimensions. -/
structure Crate where
  length : ℝ
  width : ℝ
  height : ℝ

/-- A right circular cylinder. -/
structure Cylinder where
  radius : ℝ
  height : ℝ

/-- Check if a cylinder fits in a crate when placed upright. -/
def cylinderFitsInCrate (cyl : Cylinder) (crate : Crate) : Prop :=
  cyl.radius * 2 ≤ min crate.length crate.width ∧
  cyl.height ≤ max crate.length (max crate.width crate.height)

/-- The theorem stating the maximum radius of a cylinder that fits in the given crate. -/
theorem max_cylinder_radius_in_crate :
  let crate := Crate.mk 5 8 12
  ∃ (max_radius : ℝ),
    max_radius = 2.5 ∧
    (∀ (r : ℝ), r > max_radius → ∃ (h : ℝ),
      ¬cylinderFitsInCrate (Cylinder.mk r h) crate) ∧
    (∀ (r : ℝ), r ≤ max_radius → ∃ (h : ℝ),
      cylinderFitsInCrate (Cylinder.mk r h) crate) :=
by sorry

end NUMINAMATH_CALUDE_max_cylinder_radius_in_crate_l468_46877


namespace NUMINAMATH_CALUDE_noah_holidays_l468_46829

/-- Calculates the number of holidays taken in a year given monthly holidays. -/
def holidays_per_year (monthly_holidays : ℕ) : ℕ :=
  monthly_holidays * 12

/-- Theorem: Given 3 holidays per month for a full year, the total holidays is 36. -/
theorem noah_holidays :
  holidays_per_year 3 = 36 := by
  sorry

end NUMINAMATH_CALUDE_noah_holidays_l468_46829


namespace NUMINAMATH_CALUDE_union_of_sets_l468_46822

theorem union_of_sets (M N : Set ℕ) : 
  M = {0, 1, 3} → 
  N = {x | ∃ a ∈ M, x = 3 * a} → 
  M ∪ N = {0, 1, 3, 9} := by
sorry

end NUMINAMATH_CALUDE_union_of_sets_l468_46822


namespace NUMINAMATH_CALUDE_geometric_sequence_specific_form_l468_46843

/-- A geometric sequence is a sequence where the ratio of successive terms is constant. -/
def GeometricSequence (a : ℕ → ℝ) : Prop :=
  ∃ q : ℝ, ∀ n : ℕ, a (n + 1) = q * a n

/-- The theorem states that for a geometric sequence satisfying certain conditions,
    its general term has a specific form. -/
theorem geometric_sequence_specific_form (a : ℕ → ℝ) 
    (h_geo : GeometricSequence a) 
    (h_a2 : a 2 = 1)
    (h_relation : a 3 * a 5 = 2 * a 7) :
    ∀ n : ℕ, a n = 1 / 2^(n - 2) := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_specific_form_l468_46843


namespace NUMINAMATH_CALUDE_factorization_problem_1_factorization_problem_2_l468_46808

-- Problem 1
theorem factorization_problem_1 (x y : ℝ) :
  -3 * x^2 + 6 * x * y - 3 * y^2 = -3 * (x - y)^2 := by sorry

-- Problem 2
theorem factorization_problem_2 (m n : ℝ) :
  8 * m^2 * (m + n) - 2 * (m + n) = 2 * (m + n) * (2 * m + 1) * (2 * m - 1) := by sorry

end NUMINAMATH_CALUDE_factorization_problem_1_factorization_problem_2_l468_46808


namespace NUMINAMATH_CALUDE_triangle_proof_l468_46886

noncomputable section

open Real

/-- Triangle ABC with sides a, b, c opposite to angles A, B, C respectively -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

/-- The given equation for the triangle -/
def triangle_equation (t : Triangle) : Prop :=
  t.b^2 - (2 * Real.sqrt 3 / 3) * t.b * t.c * Real.sin t.A + t.c^2 = t.a^2

theorem triangle_proof (t : Triangle) 
  (h_eq : triangle_equation t) 
  (h_b : t.b = 2) 
  (h_c : t.c = 3) :
  t.A = π/3 ∧ t.a = Real.sqrt 7 ∧ Real.sin (2*t.B - t.A) = 3*Real.sqrt 3/14 := by
  sorry

end

end NUMINAMATH_CALUDE_triangle_proof_l468_46886


namespace NUMINAMATH_CALUDE_square_root_property_l468_46837

theorem square_root_property (x : ℝ) :
  Real.sqrt (x + 4) = 3 → (x + 4)^2 = 81 := by
  sorry

end NUMINAMATH_CALUDE_square_root_property_l468_46837


namespace NUMINAMATH_CALUDE_subtraction_of_decimals_l468_46823

theorem subtraction_of_decimals : 25.019 - 3.2663 = 21.7527 := by
  sorry

end NUMINAMATH_CALUDE_subtraction_of_decimals_l468_46823


namespace NUMINAMATH_CALUDE_train_speed_crossing_bridge_l468_46888

/-- The speed of a train crossing a bridge -/
theorem train_speed_crossing_bridge (train_length bridge_length : ℝ) (crossing_time : ℝ) 
  (h1 : train_length = 135)
  (h2 : bridge_length = 240)
  (h3 : crossing_time = 30) :
  (train_length + bridge_length) / crossing_time * 3.6 = 45 := by
  sorry

#check train_speed_crossing_bridge

end NUMINAMATH_CALUDE_train_speed_crossing_bridge_l468_46888


namespace NUMINAMATH_CALUDE_diagonal_grid_4x3_triangles_l468_46809

/-- Represents a rectangular grid with diagonals -/
structure DiagonalGrid :=
  (columns : Nat)
  (rows : Nat)

/-- Calculates the number of triangles in a diagonal grid -/
def count_triangles (grid : DiagonalGrid) : Nat :=
  let small_triangles := 2 * grid.columns * grid.rows
  let larger_rectangles := (grid.columns - 1) * (grid.rows - 1)
  let larger_triangles := 8 * larger_rectangles
  let additional_triangles := 6  -- Simplified count for larger configurations
  small_triangles + larger_triangles + additional_triangles

/-- Theorem stating that a 4x3 diagonal grid contains 78 triangles -/
theorem diagonal_grid_4x3_triangles :
  ∃ (grid : DiagonalGrid), grid.columns = 4 ∧ grid.rows = 3 ∧ count_triangles grid = 78 :=
by
  sorry

#eval count_triangles ⟨4, 3⟩

end NUMINAMATH_CALUDE_diagonal_grid_4x3_triangles_l468_46809


namespace NUMINAMATH_CALUDE_total_weight_of_clothes_l468_46825

/-- The total weight of clothes collected is 8.58 kg, given that male student's clothes weigh 2.6 kg and female student's clothes weigh 5.98 kg. -/
theorem total_weight_of_clothes (male_clothes : ℝ) (female_clothes : ℝ)
  (h1 : male_clothes = 2.6)
  (h2 : female_clothes = 5.98) :
  male_clothes + female_clothes = 8.58 := by
  sorry

end NUMINAMATH_CALUDE_total_weight_of_clothes_l468_46825


namespace NUMINAMATH_CALUDE_consecutive_odd_integers_count_l468_46802

/-- Represents a sequence of consecutive odd integers -/
structure ConsecutiveOddIntegers where
  n : ℕ  -- number of integers in the sequence
  first : ℤ  -- first (least) integer in the sequence
  avg : ℚ  -- average of the integers in the sequence

/-- Theorem: Given the conditions, the number of consecutive odd integers is 8 -/
theorem consecutive_odd_integers_count
  (seq : ConsecutiveOddIntegers)
  (h1 : seq.first = 407)
  (h2 : seq.avg = 414)
  : seq.n = 8 := by
  sorry

end NUMINAMATH_CALUDE_consecutive_odd_integers_count_l468_46802


namespace NUMINAMATH_CALUDE_rectangular_prism_volume_l468_46874

/-- A rectangular prism with given face areas has a volume of 24 cubic centimeters. -/
theorem rectangular_prism_volume (w h d : ℝ) 
  (front_area : w * h = 12)
  (side_area : d * h = 6)
  (top_area : d * w = 8) :
  w * h * d = 24 := by
  sorry

end NUMINAMATH_CALUDE_rectangular_prism_volume_l468_46874


namespace NUMINAMATH_CALUDE_paths_to_2005_l468_46817

/-- Represents a position on the 5x5 grid --/
inductive Position
| Center : Position
| Side : Position
| Corner : Position
| Edge : Position

/-- Represents the possible moves on the grid --/
def possibleMoves : Position → List Position
| Position.Center => [Position.Side, Position.Corner]
| Position.Side => [Position.Side, Position.Corner, Position.Edge]
| Position.Corner => [Position.Side, Position.Edge]
| Position.Edge => []

/-- Counts the number of paths to form 2005 on the given grid --/
def countPaths : ℕ :=
  let initialSideMoves := 4
  let initialCornerMoves := 4
  let sideToSideMoves := 2
  let sideToCornerMoves := 2
  let cornerToSideMoves := 2
  let sideToEdgeMoves := 3
  let cornerToEdgeMoves := 5

  let sideSidePaths := initialSideMoves * sideToSideMoves * sideToEdgeMoves
  let sideCornerPaths := initialSideMoves * sideToCornerMoves * cornerToEdgeMoves
  let cornerSidePaths := initialCornerMoves * cornerToSideMoves * sideToEdgeMoves

  sideSidePaths + sideCornerPaths + cornerSidePaths

/-- Theorem stating that there are 88 paths to form 2005 on the given grid --/
theorem paths_to_2005 : countPaths = 88 := by sorry

end NUMINAMATH_CALUDE_paths_to_2005_l468_46817


namespace NUMINAMATH_CALUDE_total_money_left_l468_46835

def monthly_income : ℕ := 1000

def savings_june : ℕ := monthly_income * 25 / 100
def savings_july : ℕ := monthly_income * 20 / 100
def savings_august : ℕ := monthly_income * 30 / 100

def expenses_june : ℕ := 200 + monthly_income * 5 / 100
def expenses_july : ℕ := 250 + monthly_income * 15 / 100
def expenses_august : ℕ := 300 + monthly_income * 10 / 100

def gift_august : ℕ := 50

def money_left_june : ℕ := monthly_income - savings_june - expenses_june
def money_left_july : ℕ := monthly_income - savings_july - expenses_july
def money_left_august : ℕ := monthly_income - savings_august - expenses_august + gift_august

theorem total_money_left : 
  money_left_june + money_left_july + money_left_august = 1250 := by
  sorry

end NUMINAMATH_CALUDE_total_money_left_l468_46835


namespace NUMINAMATH_CALUDE_second_discount_percentage_l468_46850

theorem second_discount_percentage (initial_price : ℝ) (first_discount : ℝ) (final_price : ℝ) : 
  initial_price = 200 →
  first_discount = 10 →
  final_price = 171 →
  (initial_price * (1 - first_discount / 100) * (1 - (initial_price * (1 - first_discount / 100) - final_price) / (initial_price * (1 - first_discount / 100))) = final_price) ∧
  ((initial_price * (1 - first_discount / 100) - final_price) / (initial_price * (1 - first_discount / 100)) * 100 = 5) :=
by sorry

end NUMINAMATH_CALUDE_second_discount_percentage_l468_46850


namespace NUMINAMATH_CALUDE_test_questions_l468_46875

theorem test_questions (score : ℕ) (correct : ℕ) (incorrect : ℕ) :
  score = correct - 2 * incorrect →
  score = 73 →
  correct = 91 →
  correct + incorrect = 100 :=
by sorry

end NUMINAMATH_CALUDE_test_questions_l468_46875


namespace NUMINAMATH_CALUDE_exponent_equality_l468_46866

theorem exponent_equality (a : ℝ) (m n : ℕ) (h : a ≠ 0) :
  a^12 = (a^3)^m ∧ a^12 = a^2 * a^n → m = 4 ∧ n = 10 := by
  sorry

end NUMINAMATH_CALUDE_exponent_equality_l468_46866


namespace NUMINAMATH_CALUDE_tenth_vertex_label_l468_46851

/-- Regular 2012-gon with vertices labeled according to specific conditions -/
structure Polygon2012 where
  /-- The labeling function for vertices -/
  label : Fin 2012 → Fin 2012
  /-- The first vertex is labeled A₁ -/
  first_vertex : label 0 = 0
  /-- The second vertex is labeled A₄ -/
  second_vertex : label 1 = 3
  /-- If k+ℓ and m+n have the same remainder mod 2012, then AₖAₗ and AₘAₙ don't intersect -/
  non_intersecting_chords : ∀ k ℓ m n : Fin 2012, 
    (k + ℓ) % 2012 = (m + n) % 2012 → 
    (label k + label ℓ) % 2012 ≠ (label m + label n) % 2012

/-- The label of the tenth vertex in a Polygon2012 is A₂₈ -/
theorem tenth_vertex_label (p : Polygon2012) : p.label 9 = 27 := by
  sorry

end NUMINAMATH_CALUDE_tenth_vertex_label_l468_46851


namespace NUMINAMATH_CALUDE_lights_remaining_on_l468_46860

def total_lights : ℕ := 2007

def lights_on_after_toggle (n : ℕ) : Prop :=
  let multiples_of_2 := (total_lights - 1) / 2
  let multiples_of_3 := total_lights / 3
  let multiples_of_5 := (total_lights - 2) / 5
  let multiples_of_6 := (total_lights - 3) / 6
  let multiples_of_10 := (total_lights - 7) / 10
  let multiples_of_15 := (total_lights - 12) / 15
  let multiples_of_30 := (total_lights - 27) / 30
  let toggled := multiples_of_2 + multiples_of_3 + multiples_of_5 - 
                 multiples_of_6 - multiples_of_10 - multiples_of_15 + 
                 multiples_of_30
  n = total_lights - toggled

theorem lights_remaining_on : lights_on_after_toggle 1004 := by sorry

end NUMINAMATH_CALUDE_lights_remaining_on_l468_46860


namespace NUMINAMATH_CALUDE_variable_value_proof_l468_46821

theorem variable_value_proof (x a k some_variable : ℝ) :
  (3 * x + 2) * (2 * x - 7) = a * x^2 + k * x + some_variable →
  a - some_variable + k = 3 →
  some_variable = -14 := by
  sorry

end NUMINAMATH_CALUDE_variable_value_proof_l468_46821


namespace NUMINAMATH_CALUDE_simplify_expression_l468_46840

theorem simplify_expression (x : ℝ) (h : x = Real.sqrt 2) :
  (x^2 - x) / (x^2 - 2*x + 1) = 2 + Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l468_46840


namespace NUMINAMATH_CALUDE_arithmetic_mean_proof_l468_46820

theorem arithmetic_mean_proof (x a : ℝ) (hx : x ≠ 0) :
  ((x + 2*a)/x - 1 + ((x - 3*a)/x + 1)) / 2 = 1 - a/(2*x) := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_mean_proof_l468_46820


namespace NUMINAMATH_CALUDE_quadratic_factorization_l468_46833

theorem quadratic_factorization (a b : ℕ) (h1 : a > b) 
  (h2 : ∀ x : ℝ, x^2 - 18*x + 77 = (x - a)*(x - b)) : 
  3*b - a = 10 := by
sorry

end NUMINAMATH_CALUDE_quadratic_factorization_l468_46833


namespace NUMINAMATH_CALUDE_quadratic_roots_property_l468_46841

theorem quadratic_roots_property (x₁ x₂ : ℝ) : 
  (x₁^2 + x₁ - 3 = 0) → 
  (x₂^2 + x₂ - 3 = 0) → 
  (x₁^3 - 4*x₂^2 + 19 = 0) := by
  sorry

end NUMINAMATH_CALUDE_quadratic_roots_property_l468_46841


namespace NUMINAMATH_CALUDE_hyperbola_focus_distance_l468_46889

/-- Represents a hyperbola with semi-major axis a -/
structure Hyperbola (a : ℝ) where
  equation : ℝ → ℝ → Prop := fun x y => x^2 / a^2 - y^2 / 9 = 1
  asymptote : ℝ → ℝ → Prop := fun x y => 3 * x - 2 * y = 0

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Distance between two points -/
def distance (p1 p2 : Point) : ℝ := sorry

/-- Left focus of the hyperbola -/
def F1 (h : Hyperbola 2) : Point := sorry

/-- Right focus of the hyperbola -/
def F2 (h : Hyperbola 2) : Point := sorry

/-- A point P on the hyperbola -/
def P (h : Hyperbola 2) : Point := sorry

theorem hyperbola_focus_distance (h : Hyperbola 2) (p : Point) 
  (hp : h.equation p.x p.y) 
  (ha : h.asymptote 3 2) 
  (hd : distance p (F1 h) = 3) : 
  distance p (F2 h) = 7 := by sorry

end NUMINAMATH_CALUDE_hyperbola_focus_distance_l468_46889


namespace NUMINAMATH_CALUDE_x_plus_y_value_l468_46824

theorem x_plus_y_value (x y : ℤ) (h1 : x - y = 200) (h2 : y = 245) : x + y = 690 := by
  sorry

end NUMINAMATH_CALUDE_x_plus_y_value_l468_46824


namespace NUMINAMATH_CALUDE_train_length_calculation_l468_46892

/-- Proves that the length of each train is 75 meters given the specified conditions. -/
theorem train_length_calculation (v1 v2 t : ℝ) (h1 : v1 = 46) (h2 : v2 = 36) (h3 : t = 54) :
  let relative_speed := (v1 - v2) * (5 / 18)
  let distance := relative_speed * t
  let train_length := distance / 2
  train_length = 75 := by sorry

end NUMINAMATH_CALUDE_train_length_calculation_l468_46892


namespace NUMINAMATH_CALUDE_geometric_arithmetic_sequence_l468_46868

/-- Proves that given a geometric sequence with common ratio q, if 16a₁, 4a₂, and a₃ form an arithmetic sequence, then q = 4 -/
theorem geometric_arithmetic_sequence (a : ℕ → ℝ) (q : ℝ) :
  (∀ n, a (n + 1) = q * a n) →  -- geometric sequence condition
  (16 * a 1 + a 3 = 2 * (4 * a 2)) →  -- arithmetic sequence condition
  q = 4 := by
sorry

end NUMINAMATH_CALUDE_geometric_arithmetic_sequence_l468_46868


namespace NUMINAMATH_CALUDE_set_relations_l468_46893

def A (a : ℝ) : Set ℝ := {x | 1 - a ≤ x ∧ x ≤ 1 + a}
def B : Set ℝ := {x | x < -1 ∨ x > 5}

theorem set_relations (a : ℝ) :
  (A a ∩ B = ∅ ↔ 0 ≤ a ∧ a ≤ 4) ∧
  (A a ∪ B = B ↔ a < -4) := by
  sorry

end NUMINAMATH_CALUDE_set_relations_l468_46893


namespace NUMINAMATH_CALUDE_polygonal_number_theorem_l468_46896

/-- The n-th k-sided polygonal number -/
def N (n k : ℕ) : ℚ :=
  (k - 2) / 2 * n^2 + (4 - k) / 2 * n

/-- Theorem stating the formula for the n-th k-sided polygonal number and the value of N(8,12) -/
theorem polygonal_number_theorem (n k : ℕ) (h1 : k ≥ 3) (h2 : n ≥ 1) : 
  N n k = (k - 2) / 2 * n^2 + (4 - k) / 2 * n ∧ N 8 12 = 288 := by
  sorry

end NUMINAMATH_CALUDE_polygonal_number_theorem_l468_46896


namespace NUMINAMATH_CALUDE_quadratic_equation_equivalence_l468_46826

/-- Given a quadratic equation (x + 3)² = x(3x - 1), prove it's equivalent to 2x² - 7x - 9 = 0 in general form -/
theorem quadratic_equation_equivalence (x : ℝ) : (x + 3)^2 = x * (3*x - 1) ↔ 2*x^2 - 7*x - 9 = 0 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_equation_equivalence_l468_46826


namespace NUMINAMATH_CALUDE_fraction_equality_l468_46853

theorem fraction_equality (x y : ℝ) (h : y / x = 3 / 4) : (x + y) / x = 7 / 4 := by
  sorry

end NUMINAMATH_CALUDE_fraction_equality_l468_46853


namespace NUMINAMATH_CALUDE_quadratic_inequality_solution_l468_46844

theorem quadratic_inequality_solution (a : ℝ) (h : 1 < a ∧ a < 2) :
  ∃ (x : ℝ), x^2 - (a^2 + 3*a + 2)*x + 3*a*(a^2 + 2) < 0 ↔ a^2 + 2 < x ∧ x < 3*a :=
by sorry

end NUMINAMATH_CALUDE_quadratic_inequality_solution_l468_46844


namespace NUMINAMATH_CALUDE_mean_home_runs_l468_46848

def total_players : ℕ := 6 + 4 + 3 + 1 + 1 + 1

def total_home_runs : ℕ := 6 * 6 + 7 * 4 + 8 * 3 + 10 * 1 + 11 * 1 + 12 * 1

theorem mean_home_runs : 
  (total_home_runs : ℚ) / (total_players : ℚ) = 7.5625 := by sorry

end NUMINAMATH_CALUDE_mean_home_runs_l468_46848


namespace NUMINAMATH_CALUDE_function_properties_l468_46810

-- Define the function f
def f (a b : ℝ) (x : ℝ) : ℝ := a * x^3 + x^2 + b * x

-- Define the function g
def g (a b : ℝ) (x : ℝ) : ℝ := f a b x + (3 * a * x^2 + 2 * x + b)

-- State the theorem
theorem function_properties (a b : ℝ) :
  (∀ x, g a b x = -g a b (-x)) →  -- g is an odd function
  (∃ c, ∀ x, f a b x = -1/3 * x^3 + x^2 + c) ∧
  (∀ x ∈ Set.Icc 1 2, g (-1/3) 0 x ≤ 4 * Real.sqrt 2 / 3) ∧
  (∀ x ∈ Set.Icc 1 2, g (-1/3) 0 x ≥ 4 / 3) ∧
  (∃ x ∈ Set.Icc 1 2, g (-1/3) 0 x = 4 * Real.sqrt 2 / 3) ∧
  (∃ x ∈ Set.Icc 1 2, g (-1/3) 0 x = 4 / 3) :=
by sorry

end NUMINAMATH_CALUDE_function_properties_l468_46810


namespace NUMINAMATH_CALUDE_fuel_station_problem_l468_46814

/-- Fuel station problem -/
theorem fuel_station_problem 
  (service_cost : ℝ) 
  (fuel_cost_per_liter : ℝ) 
  (total_cost : ℝ) 
  (num_minivans : ℕ) 
  (minivan_tank : ℝ) 
  (truck_tank : ℝ) 
  (h1 : service_cost = 2.20)
  (h2 : fuel_cost_per_liter = 0.70)
  (h3 : total_cost = 395.4)
  (h4 : num_minivans = 4)
  (h5 : minivan_tank = 65)
  (h6 : truck_tank = minivan_tank * 2.2)
  : ∃ (num_trucks : ℕ), num_trucks = 2 ∧ 
    total_cost = (num_minivans * (service_cost + fuel_cost_per_liter * minivan_tank)) + 
                 (num_trucks : ℝ) * (service_cost + fuel_cost_per_liter * truck_tank) :=
by sorry


end NUMINAMATH_CALUDE_fuel_station_problem_l468_46814


namespace NUMINAMATH_CALUDE_mixture_composition_l468_46890

/-- Represents the composition of a solution --/
structure Solution :=
  (a : ℝ)  -- Percentage of chemical A
  (b : ℝ)  -- Percentage of chemical B
  (sum_to_100 : a + b = 100)

/-- The problem statement --/
theorem mixture_composition 
  (X : Solution)
  (Y : Solution)
  (Z : Solution)
  (h_X : X.a = 40)
  (h_Y : Y.a = 50)
  (h_Z : Z.a = 30)
  : ∃ (x y z : ℝ),
    x + y + z = 100 ∧
    x * X.a / 100 + y * Y.a / 100 + z * Z.a / 100 = 46 ∧
    x = 40 ∧ y = 60 ∧ z = 0 := by
  sorry

end NUMINAMATH_CALUDE_mixture_composition_l468_46890


namespace NUMINAMATH_CALUDE_arithmetic_computation_l468_46846

theorem arithmetic_computation : 2 + 8 * 3 - 4 + 6 * 5 / 2 - 3^2 = 28 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_computation_l468_46846


namespace NUMINAMATH_CALUDE_cos_arcsin_eight_seventeenths_l468_46805

theorem cos_arcsin_eight_seventeenths : 
  Real.cos (Real.arcsin (8/17)) = 15/17 := by sorry

end NUMINAMATH_CALUDE_cos_arcsin_eight_seventeenths_l468_46805


namespace NUMINAMATH_CALUDE_parallel_vectors_component_l468_46849

/-- Given two parallel vectors a and b, prove that the second component of b is 5/3. -/
theorem parallel_vectors_component (a b : ℝ × ℝ) : 
  a = (3, 5) → b.1 = 1 → (∃ (k : ℝ), a = k • b) → b.2 = 5/3 := by
  sorry

end NUMINAMATH_CALUDE_parallel_vectors_component_l468_46849


namespace NUMINAMATH_CALUDE_abs_minus_self_nonneg_l468_46898

theorem abs_minus_self_nonneg (m : ℚ) : |m| - m ≥ 0 := by sorry

end NUMINAMATH_CALUDE_abs_minus_self_nonneg_l468_46898


namespace NUMINAMATH_CALUDE_lele_can_afford_cars_with_change_l468_46839

def price_a : ℚ := 46.5
def price_b : ℚ := 54.5
def lele_money : ℚ := 120

theorem lele_can_afford_cars_with_change : 
  price_a + price_b ≤ lele_money ∧ lele_money - (price_a + price_b) = 19 :=
by sorry

end NUMINAMATH_CALUDE_lele_can_afford_cars_with_change_l468_46839


namespace NUMINAMATH_CALUDE_sin_cos_sum_special_angle_l468_46871

theorem sin_cos_sum_special_angle : 
  Real.sin (5 * π / 180) * Real.cos (55 * π / 180) + 
  Real.cos (5 * π / 180) * Real.sin (55 * π / 180) = 
  Real.sqrt 3 / 2 := by
  sorry

end NUMINAMATH_CALUDE_sin_cos_sum_special_angle_l468_46871


namespace NUMINAMATH_CALUDE_average_weight_problem_l468_46891

theorem average_weight_problem (rachel_weight jimmy_weight adam_weight : ℝ) :
  rachel_weight = 75 ∧
  rachel_weight = jimmy_weight - 6 ∧
  rachel_weight = adam_weight + 15 →
  (rachel_weight + jimmy_weight + adam_weight) / 3 = 72 :=
by sorry

end NUMINAMATH_CALUDE_average_weight_problem_l468_46891


namespace NUMINAMATH_CALUDE_seven_numbers_divisible_by_three_l468_46830

theorem seven_numbers_divisible_by_three (S : Finset ℕ) (h : S.card = 7) :
  ∃ (a b c : ℕ), a ∈ S ∧ b ∈ S ∧ c ∈ S ∧ a ≠ b ∧ b ≠ c ∧ a ≠ c ∧ (a + b + c) % 3 = 0 := by
  sorry

end NUMINAMATH_CALUDE_seven_numbers_divisible_by_three_l468_46830


namespace NUMINAMATH_CALUDE_square_division_into_rectangles_l468_46899

theorem square_division_into_rectangles :
  ∃ (s : ℝ), s > 0 ∧
  ∃ (a : ℝ), a > 0 ∧
  7 * (2 * a^2) ≤ s^2 ∧
  2 * a ≤ s :=
sorry

end NUMINAMATH_CALUDE_square_division_into_rectangles_l468_46899


namespace NUMINAMATH_CALUDE_prob_same_color_is_69_200_l468_46859

def total_balls : ℕ := 8 + 5 + 7

def prob_blue : ℚ := 8 / total_balls
def prob_green : ℚ := 5 / total_balls
def prob_red : ℚ := 7 / total_balls

def prob_same_color : ℚ := prob_blue^2 + prob_green^2 + prob_red^2

theorem prob_same_color_is_69_200 : prob_same_color = 69 / 200 := by
  sorry

end NUMINAMATH_CALUDE_prob_same_color_is_69_200_l468_46859


namespace NUMINAMATH_CALUDE_trajectory_equation_l468_46858

/-- Given two points A and B symmetric about the origin, and a moving point P such that 
    the product of the slopes of AP and BP is -1/3, prove that the trajectory of P 
    is described by the equation x^2 + 3y^2 = 4, where x ≠ ±1 -/
theorem trajectory_equation (A B P : ℝ × ℝ) : 
  A = (-1, 1) →
  B = (1, -1) →
  (∀ x y, P = (x, y) → x ≠ 1 ∧ x ≠ -1 →
    ((y - 1) / (x + 1)) * ((y + 1) / (x - 1)) = -1/3) →
  ∃ x y, P = (x, y) ∧ x^2 + 3*y^2 = 4 ∧ x ≠ 1 ∧ x ≠ -1 :=
by sorry


end NUMINAMATH_CALUDE_trajectory_equation_l468_46858


namespace NUMINAMATH_CALUDE_intersection_count_possibilities_l468_46816

-- Define the ellipse
def Ellipse (x y : ℝ) : Prop := x^2/4 + y^2 = 1

-- Define a line
def Line (a b c : ℝ) (x y : ℝ) : Prop := a*x + b*y + c = 0

-- Define intersection of two lines
def LinesIntersect (l1 l2 : ℝ × ℝ × ℝ) : Prop :=
  ∃ x y, Line l1.1 l1.2.1 l1.2.2 x y ∧ Line l2.1 l2.2.1 l2.2.2 x y

-- Define when a line is not tangent to the ellipse
def NotTangent (l : ℝ × ℝ × ℝ) : Prop :=
  ∃ x1 y1 x2 y2, x1 ≠ x2 ∧ y1 ≠ y2 ∧
    Line l.1 l.2.1 l.2.2 x1 y1 ∧ Line l.1 l.2.1 l.2.2 x2 y2 ∧
    Ellipse x1 y1 ∧ Ellipse x2 y2

-- Define the number of intersection points
def IntersectionCount (l1 l2 : ℝ × ℝ × ℝ) : ℕ :=
  sorry

-- Theorem statement
theorem intersection_count_possibilities
  (l1 l2 : ℝ × ℝ × ℝ)
  (h1 : LinesIntersect l1 l2)
  (h2 : NotTangent l1)
  (h3 : NotTangent l2) :
  (IntersectionCount l1 l2 = 2) ∨
  (IntersectionCount l1 l2 = 3) ∨
  (IntersectionCount l1 l2 = 4) :=
sorry

end NUMINAMATH_CALUDE_intersection_count_possibilities_l468_46816


namespace NUMINAMATH_CALUDE_prob_both_odd_bounds_l468_46876

def range_start : ℕ := 1
def range_end : ℕ := 1000

def is_odd (n : ℕ) : Prop := n % 2 = 1

def count_odd : ℕ := (range_end - range_start + 1) / 2

def prob_first_odd : ℚ := count_odd / range_end

def prob_second_odd : ℚ := (count_odd - 1) / (range_end - 1)

def prob_both_odd : ℚ := prob_first_odd * prob_second_odd

theorem prob_both_odd_bounds : 1/6 < prob_both_odd ∧ prob_both_odd < 1/3 := by
  sorry

end NUMINAMATH_CALUDE_prob_both_odd_bounds_l468_46876


namespace NUMINAMATH_CALUDE_white_balls_count_l468_46856

/-- Proves that in a bag with 3 red balls and x white balls, 
    if the probability of drawing a red ball is 1/4, then x = 9. -/
theorem white_balls_count (x : ℕ) : 
  (3 : ℚ) / (3 + x) = 1 / 4 → x = 9 := by
  sorry

end NUMINAMATH_CALUDE_white_balls_count_l468_46856


namespace NUMINAMATH_CALUDE_random_selection_probability_l468_46878

theorem random_selection_probability (m : ℝ) : 
  (m > 0) → 
  (2 * m) / (4 - (-2)) = 1 / 3 → 
  m = 1 := by
sorry

end NUMINAMATH_CALUDE_random_selection_probability_l468_46878


namespace NUMINAMATH_CALUDE_unknown_number_proof_l468_46885

theorem unknown_number_proof (x : ℝ) : 
  (10 + 30 + 50) / 3 = ((x + 40 + 6) / 3) + 8 → x = 20 := by
  sorry

end NUMINAMATH_CALUDE_unknown_number_proof_l468_46885


namespace NUMINAMATH_CALUDE_range_of_a_l468_46882

theorem range_of_a (a : ℝ) : 
  (∀ x : ℝ, x ≠ 0 → |x + 1/x| > |a - 2| + 1) → 
  1 < a ∧ a < 3 := by
sorry

end NUMINAMATH_CALUDE_range_of_a_l468_46882


namespace NUMINAMATH_CALUDE_range_of_a_l468_46857

open Set Real

theorem range_of_a (p q : Prop) (h : p ∧ q) : 
  (∀ x ∈ Icc 1 2, x^2 ≥ a) ∧ (∃ x : ℝ, x^2 + 2*a*x + 2 - a = 0) → 
  a ≤ -2 ∨ a = 1 := by sorry

end NUMINAMATH_CALUDE_range_of_a_l468_46857


namespace NUMINAMATH_CALUDE_raisin_cost_fraction_l468_46894

theorem raisin_cost_fraction (raisin_cost : ℝ) : 
  let nut_cost : ℝ := 3 * raisin_cost
  let raisin_weight : ℝ := 3
  let nut_weight : ℝ := 3
  let total_raisin_cost : ℝ := raisin_cost * raisin_weight
  let total_nut_cost : ℝ := nut_cost * nut_weight
  let total_cost : ℝ := total_raisin_cost + total_nut_cost
  total_raisin_cost / total_cost = 1 / 4 :=
by sorry

end NUMINAMATH_CALUDE_raisin_cost_fraction_l468_46894


namespace NUMINAMATH_CALUDE_unique_number_property_l468_46806

theorem unique_number_property : ∃! (a : ℕ), a > 1 ∧
  ∀ (p : ℕ), Prime p → (p ∣ (a^6 - 1) → (p ∣ (a^3 - 1) ∨ p ∣ (a^2 - 1))) →
  a = 2 :=
by sorry

end NUMINAMATH_CALUDE_unique_number_property_l468_46806


namespace NUMINAMATH_CALUDE_sum_of_squares_l468_46854

theorem sum_of_squares (x y z a b c d : ℝ) 
  (h1 : x ≠ 0) (h2 : y ≠ 0) (h3 : z ≠ 0) 
  (h4 : a ≠ 0) (h5 : b ≠ 0) (h6 : c ≠ 0) (h7 : d ≠ 0)
  (h8 : x * y = a) (h9 : x * z = b) (h10 : y * z = c) (h11 : x + y + z = d) :
  x^2 + y^2 + z^2 = d^2 - 2*(a + b + c) := by
  sorry

end NUMINAMATH_CALUDE_sum_of_squares_l468_46854


namespace NUMINAMATH_CALUDE_simplify_expression_l468_46861

theorem simplify_expression (x : ℝ) : (3*x - 6)*(x + 8) - (x + 6)*(3*x - 2) = 2*x - 36 := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l468_46861


namespace NUMINAMATH_CALUDE_cone_from_sector_l468_46831

/-- Proves that a cone formed from a 300° sector of a circle with radius 8 
    has a base radius of 20/3 and a slant height of 8 -/
theorem cone_from_sector (sector_angle : Real) (circle_radius : Real) 
    (cone_base_radius : Real) (cone_slant_height : Real) : 
    sector_angle = 300 ∧ 
    circle_radius = 8 ∧ 
    cone_base_radius = 20 / 3 ∧ 
    cone_slant_height = circle_radius → 
    cone_base_radius * 2 * π = sector_angle / 360 * (2 * π * circle_radius) ∧
    cone_slant_height = circle_radius := by
  sorry

end NUMINAMATH_CALUDE_cone_from_sector_l468_46831


namespace NUMINAMATH_CALUDE_daps_equivalent_to_dips_l468_46832

/-- The number of daps equivalent to one dop -/
def daps_per_dop : ℚ := 5 / 4

/-- The number of dips equivalent to one dop -/
def dips_per_dop : ℚ := 3

/-- The number of dips we want to convert to daps -/
def target_dips : ℚ := 54

/-- Theorem stating the equivalence between daps and dips -/
theorem daps_equivalent_to_dips : 
  (target_dips * daps_per_dop) / dips_per_dop = 22.5 := by sorry

end NUMINAMATH_CALUDE_daps_equivalent_to_dips_l468_46832


namespace NUMINAMATH_CALUDE_absolute_value_fraction_l468_46804

theorem absolute_value_fraction : (|3| / |(-2)^3|) = -2 := by sorry

end NUMINAMATH_CALUDE_absolute_value_fraction_l468_46804


namespace NUMINAMATH_CALUDE_max_sum_constraint_l468_46897

theorem max_sum_constraint (x y : ℝ) (h : 3 * (x^2 + y^2) = x - y) :
  ∀ a b : ℝ, 3 * (a^2 + b^2) = a - b → x + y ≤ (1 : ℝ) / 3 :=
by sorry

end NUMINAMATH_CALUDE_max_sum_constraint_l468_46897


namespace NUMINAMATH_CALUDE_decoration_time_is_five_hours_l468_46881

/-- Represents the number of eggs Mia can decorate per hour -/
def mia_eggs_per_hour : ℕ := 24

/-- Represents the number of eggs Billy can decorate per hour -/
def billy_eggs_per_hour : ℕ := 10

/-- Represents the total number of eggs that need to be decorated -/
def total_eggs : ℕ := 170

/-- Calculates the time taken to decorate all eggs when Mia and Billy work together -/
def decoration_time : ℚ :=
  total_eggs / (mia_eggs_per_hour + billy_eggs_per_hour : ℚ)

/-- Theorem stating that the decoration time is 5 hours -/
theorem decoration_time_is_five_hours :
  decoration_time = 5 := by sorry

end NUMINAMATH_CALUDE_decoration_time_is_five_hours_l468_46881


namespace NUMINAMATH_CALUDE_tetrahedron_non_coplanar_choices_l468_46834

/-- Represents a point in 3D space -/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- Represents a tetrahedron -/
structure Tetrahedron where
  vertices : Fin 4 → Point3D
  midpoints : Fin 6 → Point3D

/-- Checks if four points are coplanar -/
def are_coplanar (p1 p2 p3 p4 : Point3D) : Prop := sorry

/-- The set of all points (vertices and midpoints) of a tetrahedron -/
def tetrahedron_points (t : Tetrahedron) : Finset Point3D := sorry

/-- The number of ways to choose 4 non-coplanar points from a tetrahedron's points -/
def non_coplanar_choices (t : Tetrahedron) : ℕ := sorry

theorem tetrahedron_non_coplanar_choices :
  ∀ t : Tetrahedron, non_coplanar_choices t = 141 := sorry

end NUMINAMATH_CALUDE_tetrahedron_non_coplanar_choices_l468_46834


namespace NUMINAMATH_CALUDE_periodic_function_l468_46847

/-- A function f is periodic with period 2c if it satisfies the given functional equation. -/
theorem periodic_function (f : ℝ → ℝ) (c : ℝ) :
  (∀ x, f (x + c) = 2 / (1 + f x) - 1) →
  (∀ x, f (x + 2*c) = f x) :=
by sorry

end NUMINAMATH_CALUDE_periodic_function_l468_46847


namespace NUMINAMATH_CALUDE_complex_modulus_of_iz_eq_one_l468_46887

theorem complex_modulus_of_iz_eq_one (z : ℂ) (h : Complex.I * z = 1) : Complex.abs z = 1 := by
  sorry

end NUMINAMATH_CALUDE_complex_modulus_of_iz_eq_one_l468_46887


namespace NUMINAMATH_CALUDE_darry_full_ladder_steps_l468_46880

/-- The number of times Darry climbs his full ladder -/
def full_ladder_climbs : ℕ := 10

/-- The number of steps in Darry's smaller ladder -/
def small_ladder_steps : ℕ := 6

/-- The number of times Darry climbs his smaller ladder -/
def small_ladder_climbs : ℕ := 7

/-- The total number of steps Darry climbed -/
def total_steps : ℕ := 152

/-- The number of steps in Darry's full ladder -/
def full_ladder_steps : ℕ := 11

theorem darry_full_ladder_steps :
  full_ladder_steps * full_ladder_climbs + small_ladder_steps * small_ladder_climbs = total_steps :=
by sorry

end NUMINAMATH_CALUDE_darry_full_ladder_steps_l468_46880


namespace NUMINAMATH_CALUDE_units_digit_factorial_25_l468_46800

def factorial (n : ℕ) : ℕ := (List.range n).foldl (· * ·) 1

theorem units_digit_factorial_25 : factorial 25 % 10 = 0 := by
  sorry

end NUMINAMATH_CALUDE_units_digit_factorial_25_l468_46800


namespace NUMINAMATH_CALUDE_max_participants_l468_46867

/-- Represents a round-robin chess tournament -/
structure ChessTournament where
  n : ℕ  -- number of players
  totalPoints : ℕ  -- total sum of all players' points

/-- The number of matches in a round-robin tournament with n players -/
def matchCount (n : ℕ) : ℕ := n * (n - 1) / 2

/-- Theorem stating the maximum number of participants in the tournament -/
theorem max_participants (t : ChessTournament) (h1 : t.totalPoints = 120) :
  t.n ≤ 11 ∧ ∃ (t' : ChessTournament), t'.n = 11 ∧ t'.totalPoints = 120 := by
  sorry

#check max_participants

end NUMINAMATH_CALUDE_max_participants_l468_46867


namespace NUMINAMATH_CALUDE_auditorium_seating_l468_46895

/-- The number of ways to seat people in an auditorium with the given conditions -/
def seatingArrangements (totalPeople : ℕ) (rowSeats : ℕ) : ℕ :=
  Nat.choose totalPeople rowSeats * 2^(totalPeople - 2)

/-- Theorem stating the number of seating arrangements for the given problem -/
theorem auditorium_seating :
  seatingArrangements 100 50 = Nat.choose 100 50 * 2^98 := by
  sorry

end NUMINAMATH_CALUDE_auditorium_seating_l468_46895


namespace NUMINAMATH_CALUDE_jupiter_properties_l468_46845

/-- Given orbital parameters of a moon, calculate properties of Jupiter -/
theorem jupiter_properties 
  (T : ℝ) -- Orbital period of the moon
  (R : ℝ) -- Orbital distance of the moon
  (f : ℝ) -- Gravitational constant
  (ρ : ℝ) -- Radius of Jupiter
  (V : ℝ) -- Volume of Jupiter
  (T_rot : ℝ) -- Rotational period of Jupiter
  (h₁ : T > 0)
  (h₂ : R > 0)
  (h₃ : f > 0)
  (h₄ : ρ > 0)
  (h₅ : V > 0)
  (h₆ : T_rot > 0) :
  ∃ (M σ g₁ Cf : ℝ),
    M = 4 * Real.pi^2 * R^3 / (f * T^2) ∧
    σ = M / V ∧
    g₁ = f * M / ρ^2 ∧
    Cf = 4 * Real.pi^2 * ρ / T_rot^2 :=
by
  sorry


end NUMINAMATH_CALUDE_jupiter_properties_l468_46845


namespace NUMINAMATH_CALUDE_solution_set_equality_l468_46879

theorem solution_set_equality : 
  {x : ℝ | (x + 3) * (1 - x) ≥ 0} = {x : ℝ | -3 ≤ x ∧ x ≤ 1} := by sorry

end NUMINAMATH_CALUDE_solution_set_equality_l468_46879


namespace NUMINAMATH_CALUDE_closed_equinumerous_to_halfopen_l468_46815

-- Define the closed interval [0,1]
def closedInterval : Set ℝ := Set.Icc 0 1

-- Define the half-open interval [0,1)
def halfOpenInterval : Set ℝ := Set.Ico 0 1

-- Statement: There exists a bijective function from [0,1] to [0,1)
theorem closed_equinumerous_to_halfopen :
  ∃ f : closedInterval → halfOpenInterval, Function.Bijective f := by
  sorry

end NUMINAMATH_CALUDE_closed_equinumerous_to_halfopen_l468_46815


namespace NUMINAMATH_CALUDE_fruit_ratio_l468_46884

def total_fruit : ℕ := 13
def remaining_fruit : ℕ := 9

def fruit_fell_out : ℕ := total_fruit - remaining_fruit

theorem fruit_ratio : 
  (fruit_fell_out : ℚ) / total_fruit = 4 / 13 := by sorry

end NUMINAMATH_CALUDE_fruit_ratio_l468_46884


namespace NUMINAMATH_CALUDE_map_distance_calculation_l468_46862

/-- Given a map scale and a measured distance on the map, calculate the actual distance -/
theorem map_distance_calculation (scale : ℚ) (map_distance : ℚ) (actual_distance : ℚ) :
  scale = 1 / 20000 →
  map_distance = 6 →
  actual_distance = map_distance * 20000 / 100 →
  actual_distance = 1200 :=
by sorry

end NUMINAMATH_CALUDE_map_distance_calculation_l468_46862


namespace NUMINAMATH_CALUDE_quadratic_equation_solution_l468_46863

theorem quadratic_equation_solution :
  ∃ (x₁ x₂ : ℝ), (∀ x : ℝ, x * (x + 1) = 3 * (x + 1) ↔ x = x₁ ∨ x = x₂) ∧ x₁ = -1 ∧ x₂ = 3 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_equation_solution_l468_46863


namespace NUMINAMATH_CALUDE_jamie_oliver_vacation_cost_l468_46811

def vacation_cost (num_people : ℕ) (num_days : ℕ) (ticket_cost : ℕ) (hotel_cost_per_day : ℕ) : ℕ :=
  num_people * ticket_cost + num_people * hotel_cost_per_day * num_days

theorem jamie_oliver_vacation_cost :
  vacation_cost 2 3 24 12 = 120 := by
  sorry

end NUMINAMATH_CALUDE_jamie_oliver_vacation_cost_l468_46811


namespace NUMINAMATH_CALUDE_optimal_arrangement_l468_46852

/-- Represents the arrangement of workers in a factory --/
structure WorkerArrangement where
  total_workers : ℕ
  type_a_workers : ℕ
  type_b_workers : ℕ
  type_a_production : ℕ
  type_b_production : ℕ
  set_a_units : ℕ
  set_b_units : ℕ

/-- Checks if the arrangement produces exact sets --/
def produces_exact_sets (arrangement : WorkerArrangement) : Prop :=
  arrangement.total_workers = arrangement.type_a_workers + arrangement.type_b_workers ∧
  arrangement.type_a_workers * arrangement.type_a_production / arrangement.set_a_units =
  arrangement.type_b_workers * arrangement.type_b_production / arrangement.set_b_units

/-- Theorem stating that the given arrangement produces exact sets --/
theorem optimal_arrangement :
  produces_exact_sets {
    total_workers := 104,
    type_a_workers := 72,
    type_b_workers := 32,
    type_a_production := 8,
    type_b_production := 12,
    set_a_units := 3,
    set_b_units := 2
  } := by sorry

end NUMINAMATH_CALUDE_optimal_arrangement_l468_46852


namespace NUMINAMATH_CALUDE_distance_between_points_l468_46869

/-- The distance between two points with the same x-coordinate in a Cartesian coordinate system. -/
def distance_same_x (y₁ y₂ : ℝ) : ℝ := |y₂ - y₁|

/-- Theorem stating that the distance between (3,-2) and (3,1) is 3. -/
theorem distance_between_points : distance_same_x (-2) 1 = 3 := by sorry

end NUMINAMATH_CALUDE_distance_between_points_l468_46869


namespace NUMINAMATH_CALUDE_triple_sharp_40_l468_46803

-- Define the # function
def sharp (N : ℝ) : ℝ := 0.7 * N + 2

-- State the theorem
theorem triple_sharp_40 : sharp (sharp (sharp 40)) = 18.1 := by
  sorry

end NUMINAMATH_CALUDE_triple_sharp_40_l468_46803


namespace NUMINAMATH_CALUDE_banana_bread_theorem_l468_46801

/-- The number of bananas needed to make one loaf of banana bread -/
def bananas_per_loaf : ℕ := 4

/-- The number of loaves of banana bread made on Monday -/
def monday_loaves : ℕ := 3

/-- The number of loaves of banana bread made on Tuesday -/
def tuesday_loaves : ℕ := 2 * monday_loaves

/-- The total number of bananas used for banana bread on both days -/
def total_bananas : ℕ := bananas_per_loaf * (monday_loaves + tuesday_loaves)

theorem banana_bread_theorem : total_bananas = 36 := by
  sorry

end NUMINAMATH_CALUDE_banana_bread_theorem_l468_46801


namespace NUMINAMATH_CALUDE_min_value_expression_l468_46870

theorem min_value_expression (a : ℝ) (ha : a > 0) : 
  ((a - 1) * (4 * a - 1)) / a ≥ -1 ∧ 
  ∃ (a₀ : ℝ), a₀ > 0 ∧ ((a₀ - 1) * (4 * a₀ - 1)) / a₀ = -1 := by
  sorry

#check min_value_expression

end NUMINAMATH_CALUDE_min_value_expression_l468_46870


namespace NUMINAMATH_CALUDE_optimal_config_is_minimum_l468_46812

/-- Represents the types of vans available --/
inductive VanType
  | A
  | B
  | C

/-- Capacity of each van type --/
def vanCapacity : VanType → ℕ
  | VanType.A => 7
  | VanType.B => 9
  | VanType.C => 12

/-- Available number of each van type --/
def availableVans : VanType → ℕ
  | VanType.A => 3
  | VanType.B => 4
  | VanType.C => 2

/-- Total number of people to transport --/
def totalPeople : ℕ := 40 + 14

/-- A configuration of vans --/
structure VanConfiguration where
  typeA : ℕ
  typeB : ℕ
  typeC : ℕ

/-- Calculate the total capacity of a given van configuration --/
def totalCapacity (config : VanConfiguration) : ℕ :=
  config.typeA * vanCapacity VanType.A +
  config.typeB * vanCapacity VanType.B +
  config.typeC * vanCapacity VanType.C

/-- Check if a van configuration is valid (within available vans) --/
def isValidConfiguration (config : VanConfiguration) : Prop :=
  config.typeA ≤ availableVans VanType.A ∧
  config.typeB ≤ availableVans VanType.B ∧
  config.typeC ≤ availableVans VanType.C

/-- The optimal van configuration --/
def optimalConfig : VanConfiguration :=
  { typeA := 0, typeB := 4, typeC := 2 }

/-- Theorem stating that the optimal configuration is the minimum number of vans needed --/
theorem optimal_config_is_minimum :
  isValidConfiguration optimalConfig ∧
  totalCapacity optimalConfig ≥ totalPeople ∧
  ∀ (config : VanConfiguration),
    isValidConfiguration config →
    totalCapacity config ≥ totalPeople →
    config.typeA + config.typeB + config.typeC ≥
    optimalConfig.typeA + optimalConfig.typeB + optimalConfig.typeC :=
by
  sorry


end NUMINAMATH_CALUDE_optimal_config_is_minimum_l468_46812


namespace NUMINAMATH_CALUDE_circle_center_point_distance_l468_46872

/-- The distance between the center of a circle and a point --/
theorem circle_center_point_distance (x y : ℝ) : 
  (x^2 + y^2 = 6*x - 2*y - 15) → 
  Real.sqrt ((3 - (-2))^2 + ((-1) - 5)^2) = Real.sqrt 61 := by
  sorry

end NUMINAMATH_CALUDE_circle_center_point_distance_l468_46872


namespace NUMINAMATH_CALUDE_perimeter_of_quadrilateral_l468_46827

-- Define the quadrilateral EFGH
structure Quadrilateral :=
  (E F G H : ℝ × ℝ)

-- Define the properties of the quadrilateral
def is_right_angled (q : Quadrilateral) : Prop :=
  (q.E.1 = q.F.1 ∧ q.F.2 = q.G.2) ∧ (q.G.1 = q.H.1 ∧ q.F.2 = q.G.2)

def side_lengths (q : Quadrilateral) : ℝ × ℝ × ℝ :=
  (15, 14, 7)

-- Theorem statement
theorem perimeter_of_quadrilateral (q : Quadrilateral) 
  (h1 : is_right_angled q) 
  (h2 : side_lengths q = (15, 14, 7)) : 
  ∃ (p : ℝ), p = 36 + 2 * Real.sqrt 65 ∧ 
  p = q.E.1 - q.F.1 + q.F.2 - q.G.2 + q.G.1 - q.H.1 + Real.sqrt ((q.E.1 - q.H.1)^2 + (q.E.2 - q.H.2)^2) :=
sorry

end NUMINAMATH_CALUDE_perimeter_of_quadrilateral_l468_46827


namespace NUMINAMATH_CALUDE_min_value_expression_l468_46807

theorem min_value_expression (p q r : ℝ) (hp : p > 0) (hq : q > 0) (hr : r > 0) :
  (9 * r) / (3 * p + 2 * q) + (9 * p) / (2 * q + 3 * r) + (2 * q) / (p + r) ≥ 2 ∧
  ((9 * r) / (3 * p + 2 * q) + (9 * p) / (2 * q + 3 * r) + (2 * q) / (p + r) = 2 ↔ 3 * p = 2 * q ∧ 2 * q = 3 * r) :=
by sorry

end NUMINAMATH_CALUDE_min_value_expression_l468_46807


namespace NUMINAMATH_CALUDE_solve_for_y_l468_46864

theorem solve_for_y (x y : ℝ) (h1 : x - y = 6) (h2 : x + y = 12) : y = 3 := by
  sorry

end NUMINAMATH_CALUDE_solve_for_y_l468_46864


namespace NUMINAMATH_CALUDE_solve_for_a_l468_46836

noncomputable def f (x : ℝ) : ℝ :=
  if x > 0 then 2^x else x + 1

theorem solve_for_a (a : ℝ) (h : f a + f 1 = 0) : a = -3 := by
  sorry

end NUMINAMATH_CALUDE_solve_for_a_l468_46836


namespace NUMINAMATH_CALUDE_factorial_equation_solutions_l468_46842

theorem factorial_equation_solutions :
  ∀ x y z : ℕ, 2^x + 5^y + 63 = z! → ((x = 5 ∧ y = 2 ∧ z = 5) ∨ (x = 4 ∧ y = 4 ∧ z = 6)) :=
by sorry

end NUMINAMATH_CALUDE_factorial_equation_solutions_l468_46842


namespace NUMINAMATH_CALUDE_no_solution_to_inequalities_l468_46855

theorem no_solution_to_inequalities : ¬∃ x : ℝ, (6*x - 2 < (x + 2)^2) ∧ ((x + 2)^2 < 9*x - 5) := by
  sorry

end NUMINAMATH_CALUDE_no_solution_to_inequalities_l468_46855

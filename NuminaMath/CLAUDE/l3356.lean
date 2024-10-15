import Mathlib

namespace NUMINAMATH_CALUDE_half_angle_quadrant_l3356_335621

-- Define a function to check if an angle is in the first quadrant
def is_first_quadrant (α : ℝ) : Prop :=
  ∃ k : ℤ, k * 360 < α ∧ α < k * 360 + 90

-- Define a function to check if an angle is in the first or third quadrant
def is_first_or_third_quadrant (α : ℝ) : Prop :=
  ∃ n : ℤ, (n * 360 < α ∧ α < n * 360 + 45) ∨ 
            (n * 360 + 180 < α ∧ α < n * 360 + 225)

-- Theorem statement
theorem half_angle_quadrant (α : ℝ) :
  is_first_quadrant α → is_first_or_third_quadrant (α / 2) :=
by sorry

end NUMINAMATH_CALUDE_half_angle_quadrant_l3356_335621


namespace NUMINAMATH_CALUDE_grade_assignment_count_l3356_335602

theorem grade_assignment_count (num_students : ℕ) (num_grades : ℕ) :
  num_students = 10 →
  num_grades = 4 →
  (num_grades ^ num_students : ℕ) = 1048576 := by
  sorry

end NUMINAMATH_CALUDE_grade_assignment_count_l3356_335602


namespace NUMINAMATH_CALUDE_y_squared_plus_7y_plus_12_bounds_l3356_335694

theorem y_squared_plus_7y_plus_12_bounds (y : ℝ) (h : y^2 - 7*y + 12 < 0) : 
  42 < y^2 + 7*y + 12 ∧ y^2 + 7*y + 12 < 56 := by
sorry

end NUMINAMATH_CALUDE_y_squared_plus_7y_plus_12_bounds_l3356_335694


namespace NUMINAMATH_CALUDE_triumphal_arch_proportion_l3356_335607

/-- Represents the number of photographs of each type of attraction -/
structure Photos where
  cathedrals : ℕ
  arches : ℕ
  waterfalls : ℕ
  castles : ℕ

/-- Represents the total number of each type of attraction seen -/
structure Attractions where
  cathedrals : ℕ
  arches : ℕ
  waterfalls : ℕ
  castles : ℕ

/-- The main theorem stating the proportion of photographs featuring triumphal arches -/
theorem triumphal_arch_proportion
  (p : Photos) (a : Attractions)
  (half_photographed : p.cathedrals + p.arches + p.waterfalls + p.castles = (a.cathedrals + a.arches + a.waterfalls + a.castles) / 2)
  (cathedral_arch_ratio : a.cathedrals = 3 * a.arches)
  (castle_waterfall_equal : a.castles = a.waterfalls)
  (quarter_castles : 4 * p.castles = p.cathedrals + p.arches + p.waterfalls + p.castles)
  (half_castles_photographed : 2 * p.castles = a.castles)
  (all_arches_photographed : p.arches = a.arches) :
  4 * p.arches = p.cathedrals + p.arches + p.waterfalls + p.castles :=
by sorry

end NUMINAMATH_CALUDE_triumphal_arch_proportion_l3356_335607


namespace NUMINAMATH_CALUDE_fib_pisano_period_l3356_335695

/-- Fibonacci sequence -/
def fib : ℕ → ℕ
  | 0 => 0
  | 1 => 1
  | (n + 2) => fib (n + 1) + fib n

/-- Pisano period for modulus 10 -/
def pisano_period : ℕ := 60

theorem fib_pisano_period :
  (∀ n : ℕ, n > 0 → fib n % 10 = fib (n + pisano_period) % 10) ∧
  (∀ t : ℕ, t > 0 → t < pisano_period →
    ∃ n : ℕ, n > 0 ∧ fib n % 10 ≠ fib (n + t) % 10) := by
  sorry

end NUMINAMATH_CALUDE_fib_pisano_period_l3356_335695


namespace NUMINAMATH_CALUDE_intersection_empty_implies_t_bound_l3356_335620

def M : Set (ℝ × ℝ) := {p | p.1^3 + 8*p.2^3 + 6*p.1*p.2 ≥ 1}

def D (t : ℝ) : Set (ℝ × ℝ) := {p | p.1^2 + p.2^2 ≤ t^2}

theorem intersection_empty_implies_t_bound 
  (t : ℝ) 
  (h : t ≠ 0) 
  (h_empty : D t ∩ M = ∅) : 
  -Real.sqrt 5 / 5 < t ∧ t < Real.sqrt 5 / 5 := by
  sorry

end NUMINAMATH_CALUDE_intersection_empty_implies_t_bound_l3356_335620


namespace NUMINAMATH_CALUDE_cleaning_staff_lcm_l3356_335692

theorem cleaning_staff_lcm : Nat.lcm (Nat.lcm (Nat.lcm 5 6) 8) 10 = 120 := by
  sorry

end NUMINAMATH_CALUDE_cleaning_staff_lcm_l3356_335692


namespace NUMINAMATH_CALUDE_eliminate_denominators_l3356_335656

theorem eliminate_denominators (x : ℚ) : 
  (2*x - 1) / 2 = 1 - (3 - x) / 3 ↔ 3*(2*x - 1) = 6 - 2*(3 - x) := by
sorry

end NUMINAMATH_CALUDE_eliminate_denominators_l3356_335656


namespace NUMINAMATH_CALUDE_select_gloves_count_l3356_335622

/-- The number of ways to select 4 gloves from 6 different pairs such that exactly two of the selected gloves are of the same color -/
def select_gloves : ℕ :=
  (Nat.choose 6 1) * (Nat.choose 10 2 - 5)

/-- Theorem stating that the number of ways to select the gloves is 240 -/
theorem select_gloves_count : select_gloves = 240 := by
  sorry

end NUMINAMATH_CALUDE_select_gloves_count_l3356_335622


namespace NUMINAMATH_CALUDE_binomial_probability_theorem_l3356_335626

/-- A random variable following a binomial distribution -/
structure BinomialRV where
  n : ℕ
  p : ℝ
  h_p_range : 0 ≤ p ∧ p ≤ 1

/-- The expected value of a binomial random variable -/
def expected_value (X : BinomialRV) : ℝ := X.n * X.p

/-- The probability mass function of a binomial random variable -/
def binomial_pmf (X : BinomialRV) (k : ℕ) : ℝ :=
  (Nat.choose X.n k : ℝ) * X.p^k * (1 - X.p)^(X.n - k)

/-- The main theorem -/
theorem binomial_probability_theorem (X : BinomialRV) 
  (h_p : X.p = 1/3) 
  (h_ev : expected_value X = 2) : 
  binomial_pmf X 2 = 80/243 := by
  sorry

end NUMINAMATH_CALUDE_binomial_probability_theorem_l3356_335626


namespace NUMINAMATH_CALUDE_tims_children_treats_l3356_335679

/-- The total number of treats Tim's children get while trick-or-treating -/
def total_treats (num_children : ℕ) (hours_out : ℕ) (houses_per_hour : ℕ) (treats_per_kid : ℕ) : ℕ :=
  num_children * hours_out * houses_per_hour * treats_per_kid

/-- Theorem stating that Tim's children get 180 treats in total -/
theorem tims_children_treats : 
  total_treats 3 4 5 3 = 180 := by
  sorry

end NUMINAMATH_CALUDE_tims_children_treats_l3356_335679


namespace NUMINAMATH_CALUDE_alaya_fruit_salads_l3356_335672

def fruit_salad_problem (alaya : ℕ) (angel : ℕ) : Prop :=
  angel = 2 * alaya ∧ alaya + angel = 600

theorem alaya_fruit_salads :
  ∃ (alaya : ℕ), fruit_salad_problem alaya (2 * alaya) ∧ alaya = 200 :=
by
  sorry

end NUMINAMATH_CALUDE_alaya_fruit_salads_l3356_335672


namespace NUMINAMATH_CALUDE_equation_roots_l3356_335619

theorem equation_roots : ∀ x : ℝ, (x - 2) * (x - 3) = x - 2 ↔ x = 2 ∨ x = 4 := by
  sorry

end NUMINAMATH_CALUDE_equation_roots_l3356_335619


namespace NUMINAMATH_CALUDE_matrix_addition_result_l3356_335623

def A : Matrix (Fin 2) (Fin 2) ℤ := !![4, -2; -3, 5]
def B : Matrix (Fin 2) (Fin 2) ℤ := !![-6, 0; 7, -8]

theorem matrix_addition_result : A + B = !![-2, -2; 4, -3] := by sorry

end NUMINAMATH_CALUDE_matrix_addition_result_l3356_335623


namespace NUMINAMATH_CALUDE_boat_current_rate_l3356_335683

/-- Proves that given a boat with a speed of 15 km/hr in still water, 
    traveling 3.6 km downstream in 12 minutes, the rate of the current is 3 km/hr. -/
theorem boat_current_rate 
  (boat_speed : ℝ) 
  (distance_downstream : ℝ) 
  (time_minutes : ℝ) 
  (h1 : boat_speed = 15) 
  (h2 : distance_downstream = 3.6) 
  (h3 : time_minutes = 12) : 
  ∃ (current_rate : ℝ), current_rate = 3 ∧ 
    distance_downstream = (boat_speed + current_rate) * (time_minutes / 60) := by
  sorry

end NUMINAMATH_CALUDE_boat_current_rate_l3356_335683


namespace NUMINAMATH_CALUDE_pascal_zero_property_l3356_335647

/-- Pascal's triangle binomial coefficient -/
def pascal (n k : ℕ) : ℕ := Nat.choose n k

/-- Property that all elements except extremes are zero in a row -/
def all_zero_except_extremes (n : ℕ) : Prop :=
  ∀ k, 1 ≤ k ∧ k < n → pascal n k = 0

/-- Theorem: If s-th row has all elements zero except extremes,
    then s^k-th rows also have this property for all k ≥ 2 -/
theorem pascal_zero_property (s : ℕ) (hs : s > 1) :
  all_zero_except_extremes s →
  ∀ k, k ≥ 2 → all_zero_except_extremes (s^k) :=
by sorry

end NUMINAMATH_CALUDE_pascal_zero_property_l3356_335647


namespace NUMINAMATH_CALUDE_rectangular_hyperbola_foci_distance_l3356_335685

/-- The distance between foci of a rectangular hyperbola xy = 4 -/
theorem rectangular_hyperbola_foci_distance : 
  ∀ (x y : ℝ), x * y = 4 → ∃ (f₁ f₂ : ℝ × ℝ), 
    (f₁.1 - f₂.1)^2 + (f₁.2 - f₂.2)^2 = 16 :=
by sorry

end NUMINAMATH_CALUDE_rectangular_hyperbola_foci_distance_l3356_335685


namespace NUMINAMATH_CALUDE_sin_shift_l3356_335699

theorem sin_shift (x : ℝ) : Real.sin (5 * π / 6 - x) = Real.sin (x + π / 6) := by
  sorry

end NUMINAMATH_CALUDE_sin_shift_l3356_335699


namespace NUMINAMATH_CALUDE_twelve_chairs_subsets_l3356_335617

/-- The number of chairs arranged in a circle -/
def n : ℕ := 12

/-- A function that calculates the number of subsets with at least four adjacent chairs -/
def subsets_with_adjacent_chairs (n : ℕ) : ℕ := sorry

/-- Theorem stating that for 12 chairs in a circle, there are 1704 subsets with at least four adjacent chairs -/
theorem twelve_chairs_subsets : subsets_with_adjacent_chairs n = 1704 := by sorry

end NUMINAMATH_CALUDE_twelve_chairs_subsets_l3356_335617


namespace NUMINAMATH_CALUDE_sinusoidal_function_translation_l3356_335608

/-- Given a function f(x) = sin(ωx + φ) with the following properties:
    - ω > 0
    - |φ| < π/2
    - Smallest positive period is π
    - Graph is translated left by π/6 units
    - Resulting function is odd
    Then f(x) = sin(2x - π/3) -/
theorem sinusoidal_function_translation (f : ℝ → ℝ) (ω φ : ℝ) 
    (h_omega : ω > 0)
    (h_phi : |φ| < π/2)
    (h_period : ∀ x, f (x + π) = f x)
    (h_smallest_period : ∀ T, T > 0 → (∀ x, f (x + T) = f x) → T ≥ π)
    (h_translation : ∀ x, f (x + π/6) = -f (-x + π/6)) :
  ∀ x, f x = Real.sin (2*x - π/3) := by
  sorry

end NUMINAMATH_CALUDE_sinusoidal_function_translation_l3356_335608


namespace NUMINAMATH_CALUDE_sum_x_coordinates_invariant_l3356_335674

/-- Represents a polygon in the Cartesian plane -/
structure Polygon where
  vertices : List (ℝ × ℝ)

/-- Creates a new polygon by finding midpoints of edges of the given polygon -/
def midpointTransform (p : Polygon) : Polygon :=
  sorry

/-- Calculates the sum of x-coordinates of a polygon's vertices -/
def sumXCoordinates (p : Polygon) : ℝ :=
  sorry

/-- Theorem: The sum of x-coordinates remains constant after two midpoint transformations -/
theorem sum_x_coordinates_invariant (Q₁ : Polygon) 
  (h : sumXCoordinates Q₁ = 132) 
  (h_vertices : Q₁.vertices.length = 44) : 
  sumXCoordinates (midpointTransform (midpointTransform Q₁)) = 132 := by
  sorry

end NUMINAMATH_CALUDE_sum_x_coordinates_invariant_l3356_335674


namespace NUMINAMATH_CALUDE_three_planes_seven_parts_intersection_lines_l3356_335659

/-- A plane in 3D space -/
structure Plane3D where
  -- Add necessary fields for a plane

/-- Represents the division of space by planes -/
structure SpaceDivision where
  planes : List Plane3D
  num_parts : Nat

/-- Counts the number of intersection lines between planes -/
def count_intersection_lines (division : SpaceDivision) : Nat :=
  sorry

theorem three_planes_seven_parts_intersection_lines 
  (division : SpaceDivision) 
  (h_planes : division.planes.length = 3)
  (h_parts : division.num_parts = 7) :
  count_intersection_lines division = 3 := by
  sorry

end NUMINAMATH_CALUDE_three_planes_seven_parts_intersection_lines_l3356_335659


namespace NUMINAMATH_CALUDE_cuttable_triangle_type_l3356_335676

/-- A triangle that can be cut into two parts formable into a rectangle -/
structure CuttableTriangle where
  /-- The triangle can be cut into two parts -/
  can_be_cut : Bool
  /-- The parts can be rearranged into a rectangle -/
  forms_rectangle : Bool

/-- Types of triangles -/
inductive TriangleType
  | Right
  | Isosceles
  | Other

/-- Theorem stating that a cuttable triangle must be right or isosceles -/
theorem cuttable_triangle_type (t : CuttableTriangle) :
  t.can_be_cut ∧ t.forms_rectangle →
  (∃ tt : TriangleType, tt = TriangleType.Right ∨ tt = TriangleType.Isosceles) :=
by
  sorry

end NUMINAMATH_CALUDE_cuttable_triangle_type_l3356_335676


namespace NUMINAMATH_CALUDE_opposite_of_negative_two_thirds_l3356_335665

theorem opposite_of_negative_two_thirds : 
  -(-(2/3 : ℚ)) = 2/3 := by sorry

end NUMINAMATH_CALUDE_opposite_of_negative_two_thirds_l3356_335665


namespace NUMINAMATH_CALUDE_rectangle_side_lengths_l3356_335640

/-- Represents a rectangle with side lengths a and b -/
structure Rectangle where
  a : ℕ
  b : ℕ

/-- Represents a square with side length s -/
structure Square where
  s : ℕ

theorem rectangle_side_lengths 
  (rect : Rectangle)
  (sq1 sq2 : Square)
  (h1 : sq1.s = sq2.s)
  (h2 : rect.a + rect.b = 19)
  (h3 : 2 * (rect.a + rect.b) + 2 * sq1.s = 48)
  (h4 : 2 * (rect.a + rect.b) + 4 * sq1.s = 58)
  (h5 : rect.a > rect.b)
  (h6 : rect.a ≤ 13) :
  rect.a = 12 ∧ rect.b = 7 := by
sorry

end NUMINAMATH_CALUDE_rectangle_side_lengths_l3356_335640


namespace NUMINAMATH_CALUDE_value_of_expression_l3356_335655

theorem value_of_expression (x : ℝ) (h : x = 5) : 3 * x + 4 = 19 := by
  sorry

end NUMINAMATH_CALUDE_value_of_expression_l3356_335655


namespace NUMINAMATH_CALUDE_diagonals_30_gon_skipping_2_l3356_335670

/-- A convex polygon with n sides --/
structure ConvexPolygon (n : ℕ) where
  -- Add necessary fields here

/-- The number of diagonals in a convex n-gon that skip exactly k adjacent vertices at each end --/
def diagonals_skipping (n k : ℕ) : ℕ :=
  (n * (n - 2*k - 1)) / 2

/-- Theorem: In a 30-sided convex polygon, there are 375 diagonals that skip exactly 2 adjacent vertices at each end --/
theorem diagonals_30_gon_skipping_2 :
  diagonals_skipping 30 2 = 375 := by
  sorry

end NUMINAMATH_CALUDE_diagonals_30_gon_skipping_2_l3356_335670


namespace NUMINAMATH_CALUDE_expression_simplification_l3356_335641

theorem expression_simplification (a b : ℝ) : 
  3 * a - 4 * b + 2 * a^2 - (7 * a - 2 * a^2 + 3 * b - 5) = -4 * a - 7 * b + 4 * a^2 + 5 := by
sorry

end NUMINAMATH_CALUDE_expression_simplification_l3356_335641


namespace NUMINAMATH_CALUDE_circle_center_polar_coordinates_l3356_335631

/-- The polar coordinates of the center of the circle ρ = √2(cos θ + sin θ) are (1, π/4) -/
theorem circle_center_polar_coordinates :
  let ρ : ℝ → ℝ := λ θ => Real.sqrt 2 * (Real.cos θ + Real.sin θ)
  ∃ r θ_c, r = 1 ∧ θ_c = π/4 ∧
    ∀ θ, ρ θ = 2 * Real.cos (θ - θ_c) :=
by sorry

end NUMINAMATH_CALUDE_circle_center_polar_coordinates_l3356_335631


namespace NUMINAMATH_CALUDE_triangle_altitude_and_median_l3356_335604

/-- Triangle ABC with vertices A(3,-4), B(6,0), and C(-5,2) -/
structure Triangle where
  A : ℝ × ℝ := (3, -4)
  B : ℝ × ℝ := (6, 0)
  C : ℝ × ℝ := (-5, 2)

/-- Line equation in the form ax + by + c = 0 -/
structure LineEquation where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Theorem stating the equations of altitude BD and median BE -/
theorem triangle_altitude_and_median (t : Triangle) :
  ∃ (altitude_bd median_be : LineEquation),
    altitude_bd = ⟨4, -3, -24⟩ ∧
    median_be = ⟨1, -7, -6⟩ := by
  sorry

end NUMINAMATH_CALUDE_triangle_altitude_and_median_l3356_335604


namespace NUMINAMATH_CALUDE_tree_planting_growth_rate_l3356_335677

/-- Represents the annual average growth rate of tree planting -/
def annual_growth_rate : ℝ → Prop :=
  λ x => 400 * (1 + x)^2 = 625

/-- Theorem stating the relationship between the number of trees planted
    in the first and third years, and the annual average growth rate -/
theorem tree_planting_growth_rate :
  ∃ x : ℝ, annual_growth_rate x :=
sorry

end NUMINAMATH_CALUDE_tree_planting_growth_rate_l3356_335677


namespace NUMINAMATH_CALUDE_pure_imaginary_implies_m_eq_neg_four_l3356_335680

def z (m : ℝ) : ℂ := Complex.mk (m^2 + 2*m - 8) (m - 2)

theorem pure_imaginary_implies_m_eq_neg_four :
  ∀ m : ℝ, (z m).re = 0 ∧ (z m).im ≠ 0 → m = -4 := by
  sorry

end NUMINAMATH_CALUDE_pure_imaginary_implies_m_eq_neg_four_l3356_335680


namespace NUMINAMATH_CALUDE_minimize_area_between_curves_l3356_335687

/-- The cubic function C(x) = x^3 - 3x^2 + 2x -/
def C (x : ℝ) : ℝ := x^3 - 3*x^2 + 2*x

/-- The linear function L(x, a) = ax -/
def L (x a : ℝ) : ℝ := a * x

/-- The area S(a) bounded by C and L -/
def S (a : ℝ) : ℝ := sorry

/-- The theorem stating that the value of a minimizing S(a) is 38 - 27√2 -/
theorem minimize_area_between_curves :
  ∃ (a : ℝ), a > -1/4 ∧ ∀ (b : ℝ), b > -1/4 → S a ≤ S b ∧ a = 38 - 27 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_minimize_area_between_curves_l3356_335687


namespace NUMINAMATH_CALUDE_apples_remaining_after_three_days_l3356_335690

/-- Represents the number of apples picked from a tree on a given day -/
structure Picking where
  treeA : ℕ
  treeB : ℕ
  treeC : ℕ

/-- Calculates the total number of apples remaining after three days of picking -/
def applesRemaining (initialA initialB initialC : ℕ) (day1 day2 day3 : Picking) : ℕ :=
  (initialA - day1.treeA - day2.treeA - day3.treeA) +
  (initialB - day1.treeB - day2.treeB - day3.treeB) +
  (initialC - day1.treeC - day2.treeC - day3.treeC)

theorem apples_remaining_after_three_days :
  let initialA := 200
  let initialB := 250
  let initialC := 300
  let day1 := Picking.mk 40 25 0
  let day2 := Picking.mk 0 80 38
  let day3 := Picking.mk 60 0 40
  applesRemaining initialA initialB initialC day1 day2 day3 = 467 := by
  sorry


end NUMINAMATH_CALUDE_apples_remaining_after_three_days_l3356_335690


namespace NUMINAMATH_CALUDE_perfect_square_two_pow_plus_three_l3356_335611

theorem perfect_square_two_pow_plus_three (n : ℕ) : 
  (∃ k : ℕ, 2^n + 3 = k^2) ↔ n = 0 := by sorry

end NUMINAMATH_CALUDE_perfect_square_two_pow_plus_three_l3356_335611


namespace NUMINAMATH_CALUDE_sum_of_solutions_quadratic_sum_of_solutions_specific_quadratic_l3356_335654

theorem sum_of_solutions_quadratic (a b c : ℝ) (h : a ≠ 0) :
  let r₁ := (-b + Real.sqrt (b^2 - 4*a*c)) / (2*a)
  let r₂ := (-b - Real.sqrt (b^2 - 4*a*c)) / (2*a)
  r₁ + r₂ = -b / a :=
by sorry

theorem sum_of_solutions_specific_quadratic :
  let a : ℝ := 3
  let b : ℝ := -24
  let c : ℝ := 98
  let r₁ := (-b + Real.sqrt (b^2 - 4*a*c)) / (2*a)
  let r₂ := (-b - Real.sqrt (b^2 - 4*a*c)) / (2*a)
  r₁ + r₂ = 8 :=
by sorry

end NUMINAMATH_CALUDE_sum_of_solutions_quadratic_sum_of_solutions_specific_quadratic_l3356_335654


namespace NUMINAMATH_CALUDE_fried_chicken_dinner_pieces_l3356_335663

/-- Represents the number of pieces of chicken in a family-size Fried Chicken Dinner -/
def fried_chicken_pieces : ℕ := sorry

/-- The number of pieces of chicken in a Chicken Pasta order -/
def chicken_pasta_pieces : ℕ := 2

/-- The number of pieces of chicken in a Barbecue Chicken order -/
def barbecue_chicken_pieces : ℕ := 3

/-- The number of Fried Chicken Dinner orders -/
def fried_chicken_orders : ℕ := 2

/-- The number of Chicken Pasta orders -/
def chicken_pasta_orders : ℕ := 6

/-- The number of Barbecue Chicken orders -/
def barbecue_chicken_orders : ℕ := 3

/-- The total number of pieces of chicken needed for all orders -/
def total_chicken_pieces : ℕ := 37

theorem fried_chicken_dinner_pieces :
  fried_chicken_pieces * fried_chicken_orders +
  chicken_pasta_pieces * chicken_pasta_orders +
  barbecue_chicken_pieces * barbecue_chicken_orders =
  total_chicken_pieces ∧
  fried_chicken_pieces = 8 := by sorry

end NUMINAMATH_CALUDE_fried_chicken_dinner_pieces_l3356_335663


namespace NUMINAMATH_CALUDE_cafeteria_pies_l3356_335653

/-- Given a cafeteria with initial apples, apples handed out, and apples required per pie,
    calculate the number of pies that can be made. -/
def calculate_pies (initial_apples : ℕ) (handed_out : ℕ) (apples_per_pie : ℕ) : ℕ :=
  (initial_apples - handed_out) / apples_per_pie

/-- Theorem stating that with 47 initial apples, 27 handed out, and 4 apples per pie,
    the number of pies that can be made is 5. -/
theorem cafeteria_pies :
  calculate_pies 47 27 4 = 5 := by
  sorry

end NUMINAMATH_CALUDE_cafeteria_pies_l3356_335653


namespace NUMINAMATH_CALUDE_cosine_of_angle_between_vectors_l3356_335612

def a : ℝ × ℝ := (2, -1)
def b : ℝ × ℝ := (3, 4)

theorem cosine_of_angle_between_vectors :
  let θ := Real.arccos ((a.1 * b.1 + a.2 * b.2) / (Real.sqrt (a.1^2 + a.2^2) * Real.sqrt (b.1^2 + b.2^2)))
  Real.cos θ = 2 * Real.sqrt 5 / 25 := by
  sorry

end NUMINAMATH_CALUDE_cosine_of_angle_between_vectors_l3356_335612


namespace NUMINAMATH_CALUDE_special_function_1988_l3356_335688

/-- A function from positive integers to positive integers satisfying the given property -/
def special_function (f : ℕ+ → ℕ+) : Prop :=
  ∀ m n : ℕ+, f (f m + f n) = m + n

/-- The main theorem stating that any function satisfying the special property maps 1988 to 1988 -/
theorem special_function_1988 (f : ℕ+ → ℕ+) (h : special_function f) : f 1988 = 1988 := by
  sorry

end NUMINAMATH_CALUDE_special_function_1988_l3356_335688


namespace NUMINAMATH_CALUDE_line_equation_a_value_l3356_335646

/-- Given a line passing through points (4,3) and (12,-3), if its equation is in the form (x/a) + (y/b) = 1, then a = 8 -/
theorem line_equation_a_value (a b : ℝ) : 
  (∀ x y : ℝ, (x / a + y / b = 1) ↔ (3 * x + 4 * y = 24)) →
  ((4 : ℝ) / a + (3 : ℝ) / b = 1) →
  ((12 : ℝ) / a + (-3 : ℝ) / b = 1) →
  a = 8 := by
sorry

end NUMINAMATH_CALUDE_line_equation_a_value_l3356_335646


namespace NUMINAMATH_CALUDE_find_k_l3356_335649

theorem find_k (x y k : ℝ) 
  (h1 : x = 2) 
  (h2 : y = -3) 
  (h3 : 2 * x^2 + k * x * y = 4) : 
  k = 2/3 := by
sorry

end NUMINAMATH_CALUDE_find_k_l3356_335649


namespace NUMINAMATH_CALUDE_welcoming_and_planning_committees_l3356_335629

theorem welcoming_and_planning_committees (n : ℕ) : 
  (Nat.choose n 2 = 6) → (Nat.choose n 4 = 1) :=
by sorry

end NUMINAMATH_CALUDE_welcoming_and_planning_committees_l3356_335629


namespace NUMINAMATH_CALUDE_set_equality_implies_a_equals_one_l3356_335648

theorem set_equality_implies_a_equals_one (a : ℝ) :
  let A : Set ℝ := {1, -2, a^2 - 1}
  let B : Set ℝ := {1, a^2 - 3*a, 0}
  A = B → a = 1 := by
sorry

end NUMINAMATH_CALUDE_set_equality_implies_a_equals_one_l3356_335648


namespace NUMINAMATH_CALUDE_percentage_multiplication_l3356_335634

theorem percentage_multiplication : (10 / 100 * 10) * (20 / 100 * 20) = 4 := by
  sorry

end NUMINAMATH_CALUDE_percentage_multiplication_l3356_335634


namespace NUMINAMATH_CALUDE_box_volume_is_four_cubic_feet_l3356_335630

/-- Calculates the internal volume of a box in cubic feet given its external dimensions in inches and wall thickness -/
def internal_volume (length width height wall_thickness : ℚ) : ℚ :=
  let internal_length := length - 2 * wall_thickness
  let internal_width := width - 2 * wall_thickness
  let internal_height := height - 2 * wall_thickness
  (internal_length * internal_width * internal_height) / 1728

/-- Proves that the internal volume of the specified box is 4 cubic feet -/
theorem box_volume_is_four_cubic_feet :
  internal_volume 26 26 14 1 = 4 := by
  sorry

end NUMINAMATH_CALUDE_box_volume_is_four_cubic_feet_l3356_335630


namespace NUMINAMATH_CALUDE_broken_beads_count_l3356_335633

/-- Calculates the number of necklaces with broken beads -/
def necklaces_with_broken_beads (initial_count : ℕ) (purchased : ℕ) (gifted : ℕ) (final_count : ℕ) : ℕ :=
  initial_count + purchased - gifted - final_count

theorem broken_beads_count :
  necklaces_with_broken_beads 50 5 15 37 = 3 := by
  sorry

end NUMINAMATH_CALUDE_broken_beads_count_l3356_335633


namespace NUMINAMATH_CALUDE_segment_length_l3356_335637

/-- The length of a segment with endpoints (1,2) and (9,16) is 2√65 -/
theorem segment_length : Real.sqrt ((9 - 1)^2 + (16 - 2)^2) = 2 * Real.sqrt 65 := by
  sorry

end NUMINAMATH_CALUDE_segment_length_l3356_335637


namespace NUMINAMATH_CALUDE_skateboard_distance_l3356_335652

/-- Sequence representing the distance covered by the skateboard in each second -/
def skateboardSequence (n : ℕ) : ℕ := 8 + 10 * (n - 1)

/-- The total distance traveled by the skateboard in 20 seconds -/
def totalDistance : ℕ := (Finset.range 20).sum skateboardSequence

/-- Theorem stating that the total distance traveled is 2060 inches -/
theorem skateboard_distance : totalDistance = 2060 := by
  sorry

end NUMINAMATH_CALUDE_skateboard_distance_l3356_335652


namespace NUMINAMATH_CALUDE_endpoint_coordinate_product_l3356_335661

/-- Given a line segment with midpoint (3, -5) and one endpoint at (7, -1),
    the product of the coordinates of the other endpoint is 9. -/
theorem endpoint_coordinate_product : 
  ∀ (x y : ℝ), 
  (3 = (x + 7) / 2) →  -- midpoint x-coordinate
  (-5 = (y + (-1)) / 2) →  -- midpoint y-coordinate
  x * y = 9 := by
sorry

end NUMINAMATH_CALUDE_endpoint_coordinate_product_l3356_335661


namespace NUMINAMATH_CALUDE_race_problem_l3356_335605

/-- The race problem on a circular lake -/
theorem race_problem (lake_circumference : ℝ) (serezha_speed : ℝ) (dima_run_speed : ℝ) 
  (serezha_time : ℝ) (dima_run_time : ℝ) :
  serezha_speed = 20 →
  dima_run_speed = 6 →
  serezha_time = 0.5 →
  dima_run_time = 0.25 →
  ∃ (total_time : ℝ), total_time = 37.5 / 60 := by
  sorry

#check race_problem

end NUMINAMATH_CALUDE_race_problem_l3356_335605


namespace NUMINAMATH_CALUDE_degree_three_iff_c_eq_neg_seven_fifteenths_l3356_335609

/-- The polynomial f(x) -/
def f (x : ℝ) : ℝ := 1 - 12*x + 3*x^2 - 4*x^3 + 7*x^4

/-- The polynomial g(x) -/
def g (x : ℝ) : ℝ := 5 - 2*x - 6*x^3 + 15*x^4

/-- The combined polynomial h(x, c) = f(x) + c*g(x) -/
def h (x c : ℝ) : ℝ := f x + c * g x

/-- Theorem stating that h(x, c) has degree 3 if and only if c = -7/15 -/
theorem degree_three_iff_c_eq_neg_seven_fifteenths :
  (∀ x, h x (-7/15) = 1 - 12*x + 3*x^2 - 6/5*x^3) ∧
  (∀ c, (∀ x, h x c = 1 - 12*x + 3*x^2 - 6/5*x^3) → c = -7/15) :=
by sorry

end NUMINAMATH_CALUDE_degree_three_iff_c_eq_neg_seven_fifteenths_l3356_335609


namespace NUMINAMATH_CALUDE_product_restoration_l3356_335696

theorem product_restoration (P : ℕ) : 
  P = (List.range 11).foldl (· * ·) 1 →
  ∃ (a b c : ℕ), 
    a < 10 ∧ b < 10 ∧ c < 10 ∧
    P = 399 * 100000 + a * 10000 + 68 * 100 + b * 10 + c →
  P = 39916800 := by sorry

end NUMINAMATH_CALUDE_product_restoration_l3356_335696


namespace NUMINAMATH_CALUDE_max_brownies_is_100_l3356_335657

/-- Represents the dimensions of a rectangular pan of brownies -/
structure BrowniePan where
  m : ℕ+  -- length
  n : ℕ+  -- width

/-- The total number of brownies in the pan -/
def totalBrownies (pan : BrowniePan) : ℕ := pan.m.val * pan.n.val

/-- The number of brownies along the perimeter of the pan -/
def perimeterBrownies (pan : BrowniePan) : ℕ := 2 * (pan.m.val + pan.n.val) - 4

/-- The condition that the total number of brownies is twice the perimeter brownies -/
def validCut (pan : BrowniePan) : Prop :=
  totalBrownies pan = 2 * perimeterBrownies pan

theorem max_brownies_is_100 :
  ∃ (pan : BrowniePan), validCut pan ∧
    (∀ (other : BrowniePan), validCut other → totalBrownies other ≤ totalBrownies pan) ∧
    totalBrownies pan = 100 :=
sorry

end NUMINAMATH_CALUDE_max_brownies_is_100_l3356_335657


namespace NUMINAMATH_CALUDE_tyson_total_score_l3356_335618

/-- Calculates the total points scored in a basketball game given the number of three-pointers, two-pointers, and one-pointers made. -/
def total_points (three_pointers two_pointers one_pointers : ℕ) : ℕ :=
  3 * three_pointers + 2 * two_pointers + one_pointers

/-- Theorem stating that given Tyson's scoring record, he scored a total of 75 points. -/
theorem tyson_total_score :
  total_points 15 12 6 = 75 := by
  sorry

#eval total_points 15 12 6

end NUMINAMATH_CALUDE_tyson_total_score_l3356_335618


namespace NUMINAMATH_CALUDE_cone_surface_area_l3356_335668

/-- The surface area of a cone with base radius 1 and slant height 3 is 4π. -/
theorem cone_surface_area : 
  let r : ℝ := 1  -- base radius
  let l : ℝ := 3  -- slant height
  let S : ℝ := π * r * (r + l)  -- surface area formula
  S = 4 * π :=
by sorry

end NUMINAMATH_CALUDE_cone_surface_area_l3356_335668


namespace NUMINAMATH_CALUDE_age_multiple_problem_l3356_335689

theorem age_multiple_problem (P_age Q_age : ℝ) (M : ℝ) : 
  P_age + Q_age = 100 →
  Q_age = 37.5 →
  37.5 = M * (Q_age - (P_age - Q_age)) →
  M = 3 := by
sorry

end NUMINAMATH_CALUDE_age_multiple_problem_l3356_335689


namespace NUMINAMATH_CALUDE_maggi_cupcakes_l3356_335606

/-- Proves that Maggi ate 0 cupcakes given the initial number of packages,
    cupcakes per package, and cupcakes left. -/
theorem maggi_cupcakes (initial_packages : ℕ) (cupcakes_per_package : ℕ) (cupcakes_left : ℕ)
    (h1 : initial_packages = 3)
    (h2 : cupcakes_per_package = 4)
    (h3 : cupcakes_left = 12) :
    initial_packages * cupcakes_per_package - cupcakes_left = 0 := by
  sorry

end NUMINAMATH_CALUDE_maggi_cupcakes_l3356_335606


namespace NUMINAMATH_CALUDE_spanish_test_score_difference_l3356_335673

theorem spanish_test_score_difference (average_score : ℝ) (marco_percentage : ℝ) (margaret_score : ℝ) :
  average_score = 90 ∧
  marco_percentage = 10 ∧
  margaret_score = 86 →
  margaret_score - (average_score * (1 - marco_percentage / 100)) = 5 := by
  sorry

end NUMINAMATH_CALUDE_spanish_test_score_difference_l3356_335673


namespace NUMINAMATH_CALUDE_complex_power_difference_zero_l3356_335681

-- Define the complex number i
def i : ℂ := Complex.I

-- State the theorem
theorem complex_power_difference_zero :
  (1 + i)^24 - (1 - i)^24 = 0 :=
by
  sorry

end NUMINAMATH_CALUDE_complex_power_difference_zero_l3356_335681


namespace NUMINAMATH_CALUDE_complex_set_forms_line_l3356_335686

/-- The set of complex numbers z such that (2-3i)z is real forms a line in the complex plane -/
theorem complex_set_forms_line : 
  ∃ (m : ℝ) (b : ℝ), 
    {z : ℂ | ∃ (r : ℝ), (2 - 3*I) * z = r} = 
    {z : ℂ | z.im = m * z.re + b} :=
by sorry

end NUMINAMATH_CALUDE_complex_set_forms_line_l3356_335686


namespace NUMINAMATH_CALUDE_min_value_ab_min_value_is_16_min_value_achieved_l3356_335625

theorem min_value_ab (a b : ℝ) (h_pos_a : 0 < a) (h_pos_b : 0 < b) 
  (h_eq : 1/a + 4/b = 1) : 
  ∀ x y : ℝ, 0 < x ∧ 0 < y ∧ 1/x + 4/y = 1 → a * b ≤ x * y := by
  sorry

theorem min_value_is_16 (a b : ℝ) (h_pos_a : 0 < a) (h_pos_b : 0 < b) 
  (h_eq : 1/a + 4/b = 1) : 
  a * b ≥ 16 := by
  sorry

theorem min_value_achieved (a b : ℝ) (h_pos_a : 0 < a) (h_pos_b : 0 < b) 
  (h_eq : 1/a + 4/b = 1) : 
  ∃ x y : ℝ, 0 < x ∧ 0 < y ∧ 1/x + 4/y = 1 ∧ x * y = 16 := by
  sorry

end NUMINAMATH_CALUDE_min_value_ab_min_value_is_16_min_value_achieved_l3356_335625


namespace NUMINAMATH_CALUDE_vector_subtraction_l3356_335614

/-- Given two vectors AB and AC in R², prove that CB = AB - AC -/
theorem vector_subtraction (AB AC : Fin 2 → ℝ) (h1 : AB = ![2, 3]) (h2 : AC = ![-1, 2]) :
  (fun i => AB i - AC i) = ![3, 1] := by
  sorry

end NUMINAMATH_CALUDE_vector_subtraction_l3356_335614


namespace NUMINAMATH_CALUDE_alan_cd_cost_l3356_335660

/-- The total cost of CDs Alan buys -/
def total_cost (price_avn : ℝ) (num_dark : ℕ) (num_90s : ℕ) : ℝ :=
  let price_dark := 2 * price_avn
  let cost_dark := num_dark * price_dark
  let cost_avn := price_avn
  let cost_others := cost_dark + cost_avn
  let cost_90s := 0.4 * cost_others
  cost_dark + cost_avn + cost_90s

/-- Theorem stating the total cost of Alan's CD purchase -/
theorem alan_cd_cost :
  total_cost 12 2 5 = 84 := by
  sorry

end NUMINAMATH_CALUDE_alan_cd_cost_l3356_335660


namespace NUMINAMATH_CALUDE_square_exterior_points_l3356_335669

/-- Given a square ABCD with side length 15 and exterior points G and H, prove that GH^2 = 1126 - 1030√2 -/
theorem square_exterior_points (A B C D G H K : ℝ × ℝ) : 
  let side_length : ℝ := 15
  let bg : ℝ := 7
  let dh : ℝ := 7
  let ag : ℝ := 13
  let ch : ℝ := 13
  let dk : ℝ := 8
  -- Square ABCD conditions
  (B.1 - A.1)^2 + (B.2 - A.2)^2 = side_length^2 ∧
  (C.1 - B.1)^2 + (C.2 - B.2)^2 = side_length^2 ∧
  (D.1 - C.1)^2 + (D.2 - C.2)^2 = side_length^2 ∧
  (A.1 - D.1)^2 + (A.2 - D.2)^2 = side_length^2 ∧
  -- Exterior points conditions
  (G.1 - B.1)^2 + (G.2 - B.2)^2 = bg^2 ∧
  (H.1 - D.1)^2 + (H.2 - D.2)^2 = dh^2 ∧
  (G.1 - A.1)^2 + (G.2 - A.2)^2 = ag^2 ∧
  (H.1 - C.1)^2 + (H.2 - C.2)^2 = ch^2 ∧
  -- K on extension of BD
  (K.1 - D.1)^2 + (K.2 - D.2)^2 = dk^2 ∧
  (K.1 - B.1) / (D.1 - B.1) = (K.2 - B.2) / (D.2 - B.2) ∧
  (K.1 - D.1) / (B.1 - D.1) > 1 →
  (G.1 - H.1)^2 + (G.2 - H.2)^2 = 1126 - 1030 * Real.sqrt 2 :=
by sorry

end NUMINAMATH_CALUDE_square_exterior_points_l3356_335669


namespace NUMINAMATH_CALUDE_least_integer_absolute_value_l3356_335638

theorem least_integer_absolute_value (x : ℤ) : (∀ y : ℤ, |2*y + 9| ≤ 20 → y ≥ -14) ∧ |2*(-14) + 9| ≤ 20 := by
  sorry

end NUMINAMATH_CALUDE_least_integer_absolute_value_l3356_335638


namespace NUMINAMATH_CALUDE_carpet_area_in_sq_yards_l3356_335639

def living_room_length : ℝ := 18
def living_room_width : ℝ := 9
def storage_room_side : ℝ := 3
def sq_feet_per_sq_yard : ℝ := 9

theorem carpet_area_in_sq_yards :
  (living_room_length * living_room_width + storage_room_side * storage_room_side) / sq_feet_per_sq_yard = 19 := by
  sorry

end NUMINAMATH_CALUDE_carpet_area_in_sq_yards_l3356_335639


namespace NUMINAMATH_CALUDE_seven_lines_regions_l3356_335643

/-- The number of regions formed by n lines in a plane, where no two are parallel and no three are concurrent -/
def num_regions (n : ℕ) : ℕ := 1 + n + (n * (n - 1)) / 2

/-- The property that no two lines are parallel and no three are concurrent -/
def general_position (n : ℕ) : Prop := n > 0

theorem seven_lines_regions :
  general_position 7 → num_regions 7 = 29 := by
  sorry

end NUMINAMATH_CALUDE_seven_lines_regions_l3356_335643


namespace NUMINAMATH_CALUDE_cube_sum_divisibility_l3356_335667

theorem cube_sum_divisibility (x y z : ℤ) :
  7 ∣ (x^3 + y^3 + z^3) → 7 ∣ (x * y * z) := by
  sorry

end NUMINAMATH_CALUDE_cube_sum_divisibility_l3356_335667


namespace NUMINAMATH_CALUDE_half_page_ad_cost_l3356_335684

/-- Calculates the cost of a half-page advertisement in Math Magazine -/
theorem half_page_ad_cost : 
  let page_length : ℝ := 9
  let page_width : ℝ := 12
  let cost_per_square_inch : ℝ := 8
  let full_page_area : ℝ := page_length * page_width
  let half_page_area : ℝ := full_page_area / 2
  let total_cost : ℝ := half_page_area * cost_per_square_inch
  total_cost = 432 :=
by sorry

end NUMINAMATH_CALUDE_half_page_ad_cost_l3356_335684


namespace NUMINAMATH_CALUDE_symmetric_circle_correct_l3356_335644

-- Define the original circle
def original_circle (x y : ℝ) : Prop :=
  (x - 2)^2 + (y - 3)^2 = 1

-- Define the line of symmetry
def symmetry_line (x y : ℝ) : Prop :=
  x + y - 1 = 0

-- Define the symmetric circle
def symmetric_circle (x y : ℝ) : Prop :=
  (x + 2)^2 + (y + 1)^2 = 1

-- Theorem stating that the symmetric circle is correct
theorem symmetric_circle_correct :
  ∀ (x y : ℝ), 
    (∃ (x₀ y₀ : ℝ), original_circle x₀ y₀ ∧ 
      (∀ (x' y' : ℝ), symmetry_line ((x + x₀)/2) ((y + y₀)/2) → 
        (x - x')^2 + (y - y')^2 = (x₀ - x')^2 + (y₀ - y')^2)) →
    symmetric_circle x y :=
by sorry

end NUMINAMATH_CALUDE_symmetric_circle_correct_l3356_335644


namespace NUMINAMATH_CALUDE_boxes_with_neither_l3356_335675

theorem boxes_with_neither (total : ℕ) (markers : ℕ) (erasers : ℕ) (both : ℕ)
  (h_total : total = 15)
  (h_markers : markers = 10)
  (h_erasers : erasers = 5)
  (h_both : both = 4) :
  total - (markers + erasers - both) = 4 := by
  sorry

end NUMINAMATH_CALUDE_boxes_with_neither_l3356_335675


namespace NUMINAMATH_CALUDE_erasers_in_box_l3356_335636

/-- The number of erasers left in the box after a series of operations -/
def erasers_left (initial : ℕ) (taken : ℕ) (added : ℕ) : ℕ :=
  let remaining := initial - taken
  let half_taken := remaining / 2
  remaining - half_taken + added

/-- Theorem stating the number of erasers left in the box -/
theorem erasers_in_box : erasers_left 320 67 30 = 157 := by
  sorry

end NUMINAMATH_CALUDE_erasers_in_box_l3356_335636


namespace NUMINAMATH_CALUDE_gcf_lcm_sum_15_20_30_l3356_335627

/-- The sum of the greatest common factor and the least common multiple of 15, 20, and 30 is 65 -/
theorem gcf_lcm_sum_15_20_30 : 
  (Nat.gcd 15 (Nat.gcd 20 30) + Nat.lcm 15 (Nat.lcm 20 30)) = 65 := by
  sorry

end NUMINAMATH_CALUDE_gcf_lcm_sum_15_20_30_l3356_335627


namespace NUMINAMATH_CALUDE_unique_positive_solution_l3356_335678

theorem unique_positive_solution : ∃! (x : ℝ), x > 0 ∧ (x - 5) / 10 = 5 / (x - 10) := by
  sorry

end NUMINAMATH_CALUDE_unique_positive_solution_l3356_335678


namespace NUMINAMATH_CALUDE_acute_angle_tangent_implies_a_equals_one_l3356_335635

/-- The curve C: y = x^3 - 2ax^2 + 2ax -/
def C (a : ℤ) (x : ℝ) : ℝ := x^3 - 2*a*x^2 + 2*a*x

/-- The derivative of C with respect to x -/
def C_derivative (a : ℤ) (x : ℝ) : ℝ := 3*x^2 - 4*a*x + 2*a

theorem acute_angle_tangent_implies_a_equals_one (a : ℤ) :
  (∀ x : ℝ, C_derivative a x > 0) →
  a = 1 := by
  sorry

end NUMINAMATH_CALUDE_acute_angle_tangent_implies_a_equals_one_l3356_335635


namespace NUMINAMATH_CALUDE_inequality_equivalence_l3356_335693

theorem inequality_equivalence (x y : ℝ) : 
  Real.sqrt (x^2 - 2*x*y) > Real.sqrt (1 - y^2) ↔ 
  ((x - y > 1 ∧ -1 < y ∧ y < 1) ∨ (x - y < -1 ∧ -1 < y ∧ y < 1)) :=
by sorry

end NUMINAMATH_CALUDE_inequality_equivalence_l3356_335693


namespace NUMINAMATH_CALUDE_binomial_square_condition_l3356_335662

/-- If ax^2 + 24x + 9 is the square of a binomial, then a = 16 -/
theorem binomial_square_condition (a : ℝ) : 
  (∃ r s : ℝ, ∀ x : ℝ, a * x^2 + 24 * x + 9 = (r * x + s)^2) → a = 16 := by
  sorry

end NUMINAMATH_CALUDE_binomial_square_condition_l3356_335662


namespace NUMINAMATH_CALUDE_ellipse_point_inside_circle_l3356_335645

theorem ellipse_point_inside_circle 
  (a b c : ℝ) 
  (h_ab : a > b) 
  (h_b_pos : b > 0) 
  (h_e : c / a = 1 / 2) 
  (x₁ x₂ : ℝ) 
  (h_roots : x₁ * x₂ = -c / a ∧ x₁ + x₂ = -b / a) : 
  x₁^2 + x₂^2 < 2 := by
sorry

end NUMINAMATH_CALUDE_ellipse_point_inside_circle_l3356_335645


namespace NUMINAMATH_CALUDE_largest_angle_convex_hexagon_l3356_335601

/-- The measure of the largest angle in a convex hexagon with specific angle measures -/
theorem largest_angle_convex_hexagon : 
  ∀ x : ℝ,
  (x + 2) + (2*x + 3) + (3*x - 1) + (4*x + 2) + (5*x - 4) + (6*x - 3) = 720 →
  max (x + 2) (max (2*x + 3) (max (3*x - 1) (max (4*x + 2) (max (5*x - 4) (6*x - 3))))) = 203 :=
by
  sorry

end NUMINAMATH_CALUDE_largest_angle_convex_hexagon_l3356_335601


namespace NUMINAMATH_CALUDE_two_disco_cassettes_probability_l3356_335603

/-- Represents the total number of cassettes -/
def total_cassettes : ℕ := 30

/-- Represents the number of disco cassettes -/
def disco_cassettes : ℕ := 12

/-- Represents the number of classical cassettes -/
def classical_cassettes : ℕ := 18

/-- Probability of selecting two disco cassettes when the first is returned -/
def prob_two_disco_with_return : ℚ := 4/25

/-- Probability of selecting two disco cassettes when the first is not returned -/
def prob_two_disco_without_return : ℚ := 22/145

/-- Theorem stating the probabilities of selecting two disco cassettes -/
theorem two_disco_cassettes_probability :
  (prob_two_disco_with_return = (disco_cassettes : ℚ) / total_cassettes * (disco_cassettes : ℚ) / total_cassettes) ∧
  (prob_two_disco_without_return = (disco_cassettes : ℚ) / total_cassettes * ((disco_cassettes - 1) : ℚ) / (total_cassettes - 1)) :=
by sorry

end NUMINAMATH_CALUDE_two_disco_cassettes_probability_l3356_335603


namespace NUMINAMATH_CALUDE_first_consignment_cost_price_l3356_335615

/-- Represents a consignment of cloth -/
structure Consignment where
  length : ℕ
  profit_per_meter : ℚ

/-- Calculates the cost price per meter for a given consignment -/
def cost_price_per_meter (c : Consignment) (selling_price : ℚ) : ℚ :=
  (selling_price - c.profit_per_meter * c.length) / c.length

theorem first_consignment_cost_price 
  (c1 : Consignment)
  (c2 : Consignment)
  (c3 : Consignment)
  (selling_price : ℚ) :
  c1.length = 92 ∧ 
  c1.profit_per_meter = 24 ∧
  c2.length = 120 ∧
  c2.profit_per_meter = 30 ∧
  c3.length = 75 ∧
  c3.profit_per_meter = 20 ∧
  selling_price = 9890 →
  cost_price_per_meter c1 selling_price = 83.50 := by
    sorry

end NUMINAMATH_CALUDE_first_consignment_cost_price_l3356_335615


namespace NUMINAMATH_CALUDE_ticket_sales_l3356_335671

theorem ticket_sales (total : ℕ) (reduced_first_week : ℕ) (full_price : ℕ) :
  total = 25200 →
  reduced_first_week = 5400 →
  full_price = 5 * reduced_first_week →
  total = reduced_first_week + full_price →
  full_price = 27000 := by
  sorry

end NUMINAMATH_CALUDE_ticket_sales_l3356_335671


namespace NUMINAMATH_CALUDE_pick_one_book_theorem_l3356_335632

/-- The number of ways to pick one book from a shelf with given numbers of math, physics, and chemistry books -/
def ways_to_pick_one_book (math_books : ℕ) (physics_books : ℕ) (chemistry_books : ℕ) : ℕ :=
  math_books + physics_books + chemistry_books

/-- Theorem stating that picking one book from a shelf with 5 math books, 4 physics books, and 5 chemistry books can be done in 14 ways -/
theorem pick_one_book_theorem : ways_to_pick_one_book 5 4 5 = 14 := by
  sorry

end NUMINAMATH_CALUDE_pick_one_book_theorem_l3356_335632


namespace NUMINAMATH_CALUDE_simplify_and_evaluate_l3356_335600

theorem simplify_and_evaluate (a b : ℚ) (ha : a = 2) (hb : b = 2/5) :
  (2*a + b)^2 - (3*b + 2*a) * (2*a - 3*b) = 24/5 := by
  sorry

end NUMINAMATH_CALUDE_simplify_and_evaluate_l3356_335600


namespace NUMINAMATH_CALUDE_units_digit_of_7_cubed_l3356_335650

theorem units_digit_of_7_cubed (n : ℕ) : n = 7^3 → n % 10 = 3 := by
  sorry

end NUMINAMATH_CALUDE_units_digit_of_7_cubed_l3356_335650


namespace NUMINAMATH_CALUDE_min_value_product_min_value_product_achieved_l3356_335682

theorem min_value_product (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (h_abc : a * b * c = 8) :
  (2 * a + b) * (a + 3 * c) * (b * c + 2) ≥ 128 :=
by sorry

theorem min_value_product_achieved (ε : ℝ) (hε : ε > 0) :
  ∃ (a b c : ℝ), a > 0 ∧ b > 0 ∧ c > 0 ∧ a * b * c = 8 ∧
    (2 * a + b) * (a + 3 * c) * (b * c + 2) < 128 + ε :=
by sorry

end NUMINAMATH_CALUDE_min_value_product_min_value_product_achieved_l3356_335682


namespace NUMINAMATH_CALUDE_difference_of_squares_81_49_l3356_335610

theorem difference_of_squares_81_49 : 81^2 - 49^2 = 4160 := by
  sorry

end NUMINAMATH_CALUDE_difference_of_squares_81_49_l3356_335610


namespace NUMINAMATH_CALUDE_fence_perimeter_l3356_335664

/-- The outer perimeter of a square fence given the number of posts, post width, and gap between posts. -/
def outerPerimeter (numPosts : ℕ) (postWidth : ℚ) (gapWidth : ℕ) : ℚ :=
  let postsPerSide : ℕ := numPosts / 4
  let numGaps : ℕ := postsPerSide - 1
  let gapLength : ℚ := numGaps * gapWidth
  let postLength : ℚ := postsPerSide * postWidth
  let sideLength : ℚ := gapLength + postLength
  4 * sideLength

/-- Theorem stating that the outer perimeter of the fence is 274 feet. -/
theorem fence_perimeter :
  outerPerimeter 36 (1/2) 8 = 274 := by sorry

end NUMINAMATH_CALUDE_fence_perimeter_l3356_335664


namespace NUMINAMATH_CALUDE_tan_sum_15_30_l3356_335616

-- Define the tangent function
noncomputable def tan (x : ℝ) : ℝ := Real.tan x

-- State the theorem
theorem tan_sum_15_30 :
  tan (15 * π / 180) + tan (30 * π / 180) + tan (15 * π / 180) * tan (30 * π / 180) = 1 :=
by
  -- Assume the trigonometric identity for the tangent of the sum of two angles
  have tan_sum_identity : ∀ A B : ℝ, tan (A + B) = (tan A + tan B) / (1 - tan A * tan B) := sorry
  -- Assume that tan 45° = 1
  have tan_45 : tan (45 * π / 180) = 1 := sorry
  sorry -- The proof goes here

end NUMINAMATH_CALUDE_tan_sum_15_30_l3356_335616


namespace NUMINAMATH_CALUDE_basketball_shot_expectation_l3356_335666

theorem basketball_shot_expectation 
  (a b c : ℝ) 
  (ha : 0 < a ∧ a < 1) 
  (hb : 0 < b ∧ b < 1) 
  (hc : 0 < c ∧ c < 1) 
  (h_sum : a + b + c = 1) 
  (h_expect : 3 * a + 2 * b = 2) : 
  (∀ x y, x > 0 ∧ y > 0 ∧ 3 * x + 2 * y = 2 → (2 / x + 1 / (3 * y)) ≥ 16 / 3) ∧ 
  (∃ x y, x > 0 ∧ y > 0 ∧ 3 * x + 2 * y = 2 ∧ 2 / x + 1 / (3 * y) = 16 / 3) :=
by sorry

end NUMINAMATH_CALUDE_basketball_shot_expectation_l3356_335666


namespace NUMINAMATH_CALUDE_solution_set_of_even_decreasing_function_l3356_335698

-- Define the function f
def f (a b x : ℝ) : ℝ := (x - 1) * (a * x + b)

-- State the theorem
theorem solution_set_of_even_decreasing_function (a b : ℝ) :
  (∀ x, f a b x = f a b (-x)) →  -- f is even
  (∀ x y, 0 < x ∧ x < y → f a b y < f a b x) →  -- f is decreasing on (0, +∞)
  {x : ℝ | f a b (2 - x) < 0} = {x : ℝ | x < 1 ∨ x > 3} :=
by sorry


end NUMINAMATH_CALUDE_solution_set_of_even_decreasing_function_l3356_335698


namespace NUMINAMATH_CALUDE_triangle_angle_x_l3356_335613

theorem triangle_angle_x (x : ℝ) : 
  x > 0 ∧ 3*x > 0 ∧ 40 > 0 ∧ 
  x + 3*x + 40 = 180 → 
  x = 35 := by
sorry

end NUMINAMATH_CALUDE_triangle_angle_x_l3356_335613


namespace NUMINAMATH_CALUDE_product_of_repeating_decimals_l3356_335642

/-- The first repeating decimal 0.030303... -/
def decimal1 : ℚ := 1 / 33

/-- The second repeating decimal 0.363636... -/
def decimal2 : ℚ := 4 / 11

/-- Theorem stating that the product of the two repeating decimals is 4/363 -/
theorem product_of_repeating_decimals :
  decimal1 * decimal2 = 4 / 363 := by sorry

end NUMINAMATH_CALUDE_product_of_repeating_decimals_l3356_335642


namespace NUMINAMATH_CALUDE_unique_zero_in_interval_l3356_335628

theorem unique_zero_in_interval (a : ℝ) (h : a > 3) :
  ∃! x : ℝ, x ∈ (Set.Ioo 0 2) ∧ x^2 - a*x + 1 = 0 :=
by sorry

end NUMINAMATH_CALUDE_unique_zero_in_interval_l3356_335628


namespace NUMINAMATH_CALUDE_discount_calculation_l3356_335658

/-- Proves that given a list price of 70, a final price of 59.85, and two successive discounts
    where one is 10%, the other discount percentage is 5%. -/
theorem discount_calculation (list_price : ℝ) (final_price : ℝ) (discount1 : ℝ) (discount2 : ℝ) :
  list_price = 70 →
  final_price = 59.85 →
  discount1 = 10 →
  final_price = list_price * (1 - discount1 / 100) * (1 - discount2 / 100) →
  discount2 = 5 := by
  sorry

end NUMINAMATH_CALUDE_discount_calculation_l3356_335658


namespace NUMINAMATH_CALUDE_sqrt_88_plus_42sqrt3_form_l3356_335624

theorem sqrt_88_plus_42sqrt3_form : ∃ (a b c : ℤ), 
  (Real.sqrt (88 + 42 * Real.sqrt 3) = a + b * Real.sqrt c) ∧ 
  (∀ (k : ℕ), k > 1 → ¬(∃ (m : ℕ), c = k^2 * m)) ∧
  (a + b + c = 13) := by
  sorry

end NUMINAMATH_CALUDE_sqrt_88_plus_42sqrt3_form_l3356_335624


namespace NUMINAMATH_CALUDE_walking_speed_problem_l3356_335651

/-- 
Given two people walking in opposite directions for 12 hours, 
with one walking at 3 km/hr and the distance between them after 12 hours being 120 km, 
prove that the speed of the other person is 7 km/hr.
-/
theorem walking_speed_problem (v : ℝ) : 
  v > 0 → -- Assuming positive speed
  (v + 3) * 12 = 120 → 
  v = 7 := by
sorry

end NUMINAMATH_CALUDE_walking_speed_problem_l3356_335651


namespace NUMINAMATH_CALUDE_bob_water_percentage_l3356_335691

def corn_water_usage : ℝ := 20
def cotton_water_usage : ℝ := 80
def bean_water_usage : ℝ := 2 * corn_water_usage

def bob_corn_acres : ℝ := 3
def bob_cotton_acres : ℝ := 9
def bob_bean_acres : ℝ := 12

def brenda_corn_acres : ℝ := 6
def brenda_cotton_acres : ℝ := 7
def brenda_bean_acres : ℝ := 14

def bernie_corn_acres : ℝ := 2
def bernie_cotton_acres : ℝ := 12

def bob_water_usage : ℝ := 
  bob_corn_acres * corn_water_usage + 
  bob_cotton_acres * cotton_water_usage + 
  bob_bean_acres * bean_water_usage

def total_water_usage : ℝ := 
  (bob_corn_acres + brenda_corn_acres + bernie_corn_acres) * corn_water_usage +
  (bob_cotton_acres + brenda_cotton_acres + bernie_cotton_acres) * cotton_water_usage +
  (bob_bean_acres + brenda_bean_acres) * bean_water_usage

theorem bob_water_percentage : 
  bob_water_usage / total_water_usage * 100 = 36 := by sorry

end NUMINAMATH_CALUDE_bob_water_percentage_l3356_335691


namespace NUMINAMATH_CALUDE_initial_sets_count_l3356_335697

/-- The number of letters available (A through J) -/
def n : ℕ := 10

/-- The number of letters in each set of initials -/
def k : ℕ := 3

/-- The number of different three-letter sets of initials possible -/
def num_initial_sets : ℕ := n * (n - 1) * (n - 2)

/-- Theorem stating that the number of different three-letter sets of initials
    using letters A through J, with no repeated letters, is equal to 720 -/
theorem initial_sets_count : num_initial_sets = 720 := by
  sorry

end NUMINAMATH_CALUDE_initial_sets_count_l3356_335697

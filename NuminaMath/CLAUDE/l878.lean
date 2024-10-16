import Mathlib

namespace NUMINAMATH_CALUDE_boxes_with_neither_l878_87853

/-- Represents the set of boxes in Christine's storage room. -/
def Boxes : Type := Unit

/-- The total number of boxes. -/
def total_boxes : ℕ := 15

/-- The number of boxes containing markers. -/
def boxes_with_markers : ℕ := 8

/-- The number of boxes containing sharpies. -/
def boxes_with_sharpies : ℕ := 5

/-- The number of boxes containing both markers and sharpies. -/
def boxes_with_both : ℕ := 4

/-- Theorem stating the number of boxes containing neither markers nor sharpies. -/
theorem boxes_with_neither (b : Boxes) :
  total_boxes - (boxes_with_markers + boxes_with_sharpies - boxes_with_both) = 6 := by
  sorry


end NUMINAMATH_CALUDE_boxes_with_neither_l878_87853


namespace NUMINAMATH_CALUDE_smallest_solution_abs_equation_l878_87885

theorem smallest_solution_abs_equation :
  ∃ (x : ℝ), x * |x| = 3 * x + 2 ∧
  ∀ (y : ℝ), y * |y| = 3 * y + 2 → x ≤ y ∧
  x = -2 := by
sorry

end NUMINAMATH_CALUDE_smallest_solution_abs_equation_l878_87885


namespace NUMINAMATH_CALUDE_mary_shop_visits_mary_shop_visits_proof_l878_87888

def shirt_cost : ℝ := 13.04
def jacket_cost : ℝ := 12.27
def total_cost : ℝ := 25.31

theorem mary_shop_visits : ℕ :=
  2

theorem mary_shop_visits_proof :
  (shirt_cost + jacket_cost = total_cost) →
  (∀ (shop : ℕ), shop ≤ mary_shop_visits → 
    (shop = 1 → shirt_cost > 0) ∧ 
    (shop = 2 → jacket_cost > 0)) →
  mary_shop_visits = 2 :=
by
  sorry

end NUMINAMATH_CALUDE_mary_shop_visits_mary_shop_visits_proof_l878_87888


namespace NUMINAMATH_CALUDE_max_servings_is_fifty_l878_87895

/-- Represents the number of chunks per serving for each fruit type -/
structure FruitRatio :=
  (cantaloupe : ℕ)
  (honeydew : ℕ)
  (pineapple : ℕ)
  (watermelon : ℕ)

/-- Represents the available chunks of each fruit type -/
structure AvailableFruit :=
  (cantaloupe : ℕ)
  (honeydew : ℕ)
  (pineapple : ℕ)
  (watermelon : ℕ)

/-- Calculates the maximum number of servings possible given a ratio and available fruit -/
def maxServings (ratio : FruitRatio) (available : AvailableFruit) : ℕ :=
  min
    (available.cantaloupe / ratio.cantaloupe)
    (min
      (available.honeydew / ratio.honeydew)
      (min
        (available.pineapple / ratio.pineapple)
        (available.watermelon / ratio.watermelon)))

theorem max_servings_is_fifty :
  let ratio : FruitRatio := ⟨3, 2, 1, 4⟩
  let available : AvailableFruit := ⟨150, 135, 60, 220⟩
  let minServings : ℕ := 50
  maxServings ratio available = 50 ∧ maxServings ratio available ≥ minServings :=
by sorry

end NUMINAMATH_CALUDE_max_servings_is_fifty_l878_87895


namespace NUMINAMATH_CALUDE_family_ages_solution_l878_87875

/-- Represents the current ages of Jennifer, Jordana, and James -/
structure FamilyAges where
  jennifer : ℕ
  jordana : ℕ
  james : ℕ

/-- Checks if the given ages satisfy the problem conditions -/
def satisfiesConditions (ages : FamilyAges) : Prop :=
  ages.jennifer + 20 = 40 ∧
  ages.jordana + 20 = 2 * (ages.jennifer + 20) ∧
  ages.james + 20 = (ages.jennifer + 20) + (ages.jordana + 20) - 10

theorem family_ages_solution :
  ∃ (ages : FamilyAges), satisfiesConditions ages ∧ ages.jordana = 60 ∧ ages.james = 90 :=
sorry

end NUMINAMATH_CALUDE_family_ages_solution_l878_87875


namespace NUMINAMATH_CALUDE_exists_valid_grid_l878_87863

/-- Represents a 3x3 grid with numbers -/
structure Grid :=
  (top_left top_right bottom_left bottom_right : ℕ)

/-- The sum of numbers along each side of the grid is 13 -/
def valid_sum (g : Grid) : Prop :=
  g.top_left + 4 + g.top_right = 13 ∧
  g.top_right + 2 + g.bottom_right = 13 ∧
  g.bottom_right + 1 + g.bottom_left = 13 ∧
  g.bottom_left + 3 + g.top_left = 13

/-- There exists a valid grid arrangement -/
theorem exists_valid_grid : ∃ (g : Grid), valid_sum g :=
sorry

end NUMINAMATH_CALUDE_exists_valid_grid_l878_87863


namespace NUMINAMATH_CALUDE_quadratic_one_root_l878_87838

theorem quadratic_one_root (m : ℝ) : 
  (∀ x : ℝ, x^2 - 8*m*x + 15*m = 0 → (∀ y : ℝ, y^2 - 8*m*y + 15*m = 0 → y = x)) → 
  m = 15/16 :=
sorry

end NUMINAMATH_CALUDE_quadratic_one_root_l878_87838


namespace NUMINAMATH_CALUDE_inequality_proof_l878_87821

theorem inequality_proof (x : ℝ) : 
  (4 * x^2 / (1 - Real.sqrt (1 + 2*x))^2 < 2*x + 9) → 
  (-1/2 ≤ x ∧ x < 45/8 ∧ x ≠ 0) := by
sorry

end NUMINAMATH_CALUDE_inequality_proof_l878_87821


namespace NUMINAMATH_CALUDE_group_size_proof_l878_87805

theorem group_size_proof (W : ℝ) (N : ℕ) : 
  ((W + 35) / N = W / N + 3.5) → N = 10 := by
  sorry

end NUMINAMATH_CALUDE_group_size_proof_l878_87805


namespace NUMINAMATH_CALUDE_final_racers_count_l878_87892

def race_elimination (initial_racers : ℕ) : ℕ :=
  let after_first := initial_racers - 10
  let after_second := after_first - (after_first / 3)
  let after_third := after_second - (after_second / 2)
  after_third

theorem final_racers_count :
  race_elimination 100 = 30 := by sorry

end NUMINAMATH_CALUDE_final_racers_count_l878_87892


namespace NUMINAMATH_CALUDE_hyperbola_properties_l878_87834

/-- The hyperbola defined by the equation x^2 - y^2 = 1 passes through (1, 0) and has asymptotes x ± y = 0 -/
theorem hyperbola_properties :
  ∃ (x y : ℝ), 
    (x^2 - y^2 = 1) ∧ 
    (x = 1 ∧ y = 0) ∧
    (∀ (t : ℝ), (x = t ∧ y = t) ∨ (x = t ∧ y = -t)) :=
by sorry

end NUMINAMATH_CALUDE_hyperbola_properties_l878_87834


namespace NUMINAMATH_CALUDE_sum_first_5_even_numbers_is_30_l878_87879

def first_n_even_numbers (n : ℕ) : List ℕ :=
  List.map (fun i => 2 * i) (List.range n)

theorem sum_first_5_even_numbers_is_30 :
  List.sum (first_n_even_numbers 5) = 30 :=
by
  sorry

#check sum_first_5_even_numbers_is_30

end NUMINAMATH_CALUDE_sum_first_5_even_numbers_is_30_l878_87879


namespace NUMINAMATH_CALUDE_convex_polyhedron_max_intersections_non_convex_polyhedron_96_intersections_no_full_intersection_l878_87881

/-- A polyhedron with a specified number of edges. -/
structure Polyhedron where
  edges : ℕ
  is_convex : Bool

/-- A plane that can intersect edges of a polyhedron. -/
structure IntersectingPlane where
  intersected_edges : ℕ
  passes_through_vertices : Bool

/-- Theorem about the maximum number of edges a plane can intersect in a convex polyhedron. -/
theorem convex_polyhedron_max_intersections
  (p : Polyhedron)
  (plane : IntersectingPlane)
  (h1 : p.edges = 100)
  (h2 : p.is_convex = true)
  (h3 : plane.passes_through_vertices = false) :
  plane.intersected_edges ≤ 66 :=
sorry

/-- Theorem about the existence of a non-convex polyhedron where a plane can intersect 96 edges. -/
theorem non_convex_polyhedron_96_intersections
  (p : Polyhedron)
  (h : p.edges = 100) :
  ∃ (plane : IntersectingPlane), plane.intersected_edges = 96 ∧ p.is_convex = false :=
sorry

/-- Theorem stating that it's impossible for a plane to intersect all 100 edges of a polyhedron. -/
theorem no_full_intersection
  (p : Polyhedron)
  (h : p.edges = 100) :
  ¬ ∃ (plane : IntersectingPlane), plane.intersected_edges = 100 :=
sorry

end NUMINAMATH_CALUDE_convex_polyhedron_max_intersections_non_convex_polyhedron_96_intersections_no_full_intersection_l878_87881


namespace NUMINAMATH_CALUDE_complex_modulus_from_square_l878_87873

theorem complex_modulus_from_square (z : ℂ) (h : z^2 = -48 + 64*I) : 
  Complex.abs z = 4 * Real.sqrt 5 := by
sorry

end NUMINAMATH_CALUDE_complex_modulus_from_square_l878_87873


namespace NUMINAMATH_CALUDE_condition_necessary_not_sufficient_l878_87810

theorem condition_necessary_not_sufficient 
  (a₁ a₂ b₁ b₂ : ℝ) 
  (h_nonzero : a₁ ≠ 0 ∧ a₂ ≠ 0 ∧ b₁ ≠ 0 ∧ b₂ ≠ 0) 
  (A : Set ℝ) 
  (hA : A = {x : ℝ | a₁ * x + b₁ > 0}) 
  (B : Set ℝ) 
  (hB : B = {x : ℝ | a₂ * x + b₂ > 0}) : 
  (∀ (A B : Set ℝ), A = B → a₁ / a₂ = b₁ / b₂) ∧ 
  ¬(∀ (A B : Set ℝ), a₁ / a₂ = b₁ / b₂ → A = B) :=
sorry

end NUMINAMATH_CALUDE_condition_necessary_not_sufficient_l878_87810


namespace NUMINAMATH_CALUDE_equation_solution_l878_87894

theorem equation_solution : 
  ∃ x : ℝ, (1 / 7 + 7 / x = 15 / x + 1 / 15) ∧ (x = 8) := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l878_87894


namespace NUMINAMATH_CALUDE_cookie_radius_l878_87865

/-- The equation of the cookie's boundary -/
def cookie_equation (x y : ℝ) : Prop :=
  x^2 + y^2 + 2*x - 4*y = 8

/-- The cookie is a circle -/
def is_circle (center : ℝ × ℝ) (radius : ℝ) : Prop :=
  ∀ x y, cookie_equation x y ↔ (x - center.1)^2 + (y - center.2)^2 = radius^2

/-- The radius of the cookie is √13 -/
theorem cookie_radius :
  ∃ center, is_circle center (Real.sqrt 13) :=
sorry

end NUMINAMATH_CALUDE_cookie_radius_l878_87865


namespace NUMINAMATH_CALUDE_midpoint_quadrilateral_area_l878_87843

/-- Given a rectangle with length 15 and width 8, the quadrilateral formed by
    connecting the midpoints of its sides has an area of 30 square units. -/
theorem midpoint_quadrilateral_area (l w : ℝ) (hl : l = 15) (hw : w = 8) :
  let midpoint_quad_area := (l / 2) * (w / 2)
  midpoint_quad_area = 30 := by sorry

end NUMINAMATH_CALUDE_midpoint_quadrilateral_area_l878_87843


namespace NUMINAMATH_CALUDE_mean_of_remaining_numbers_l878_87891

def numbers : List ℝ := [1924, 2057, 2170, 2229, 2301, 2365]

theorem mean_of_remaining_numbers (subset : List ℝ) (h1 : subset ⊆ numbers) 
  (h2 : subset.length = 4) (h3 : (subset.sum / subset.length) = 2187.25) :
  let remaining := numbers.filter (fun x => x ∉ subset)
  (remaining.sum / remaining.length) = 2148.5 := by
sorry

end NUMINAMATH_CALUDE_mean_of_remaining_numbers_l878_87891


namespace NUMINAMATH_CALUDE_polynomial_coefficient_problem_l878_87858

theorem polynomial_coefficient_problem (a b : ℝ) : 
  (0 < a) → 
  (0 < b) → 
  (a + b = 1) → 
  (21 * a^10 * b^4 = 35 * a^8 * b^6) → 
  a = 5 / (5 + Real.sqrt 15) := by
  sorry

end NUMINAMATH_CALUDE_polynomial_coefficient_problem_l878_87858


namespace NUMINAMATH_CALUDE_cashier_miscount_adjustment_l878_87820

/-- Represents the value of a coin in cents -/
def coin_value (coin : String) : ℕ :=
  match coin with
  | "penny" => 1
  | "nickel" => 5
  | "dime" => 10
  | "quarter" => 25
  | _ => 0

/-- Calculates the error when miscounting one coin as another -/
def miscount_error (actual : String) (counted_as : String) : ℤ :=
  (coin_value counted_as : ℤ) - (coin_value actual : ℤ)

/-- Theorem: The net error and correct adjustment for x miscounted coins -/
theorem cashier_miscount_adjustment (x : ℕ) :
  let penny_as_nickel_error := miscount_error "penny" "nickel"
  let quarter_as_dime_error := miscount_error "quarter" "dime"
  let net_error := x * penny_as_nickel_error + x * quarter_as_dime_error
  let adjustment := -net_error
  (net_error = -11 * x) ∧ (adjustment = 11 * x) := by
  sorry

end NUMINAMATH_CALUDE_cashier_miscount_adjustment_l878_87820


namespace NUMINAMATH_CALUDE_water_volume_calculation_l878_87832

/-- The volume of water in a container can be calculated by multiplying the number of small hemisphere containers required to hold the water by the volume of each small hemisphere container. -/
theorem water_volume_calculation (num_containers : ℕ) (hemisphere_volume : ℝ) (total_volume : ℝ) : 
  num_containers = 2735 →
  hemisphere_volume = 4 →
  total_volume = num_containers * hemisphere_volume →
  total_volume = 10940 := by
sorry

end NUMINAMATH_CALUDE_water_volume_calculation_l878_87832


namespace NUMINAMATH_CALUDE_sphere_volume_l878_87864

theorem sphere_volume (r : ℝ) (h : r > 0) :
  (∃ (d : ℝ), d > 0 ∧ d < r ∧
    4 = (r^2 - d^2).sqrt ∧
    d = 3) →
  (4 / 3 * Real.pi * r^3 = 500 * Real.pi / 3) :=
by sorry

end NUMINAMATH_CALUDE_sphere_volume_l878_87864


namespace NUMINAMATH_CALUDE_solve_pretzel_problem_l878_87824

def pretzel_problem (barry_pretzels : ℕ) : Prop :=
  let shelly_pretzels : ℕ := barry_pretzels / 2
  let angie_pretzels : ℕ := 3 * shelly_pretzels
  let dave_pretzels : ℕ := (angie_pretzels + shelly_pretzels) / 4
  let total_pretzels : ℕ := barry_pretzels + shelly_pretzels + angie_pretzels + dave_pretzels
  let price_per_pretzel : ℕ := 1
  let total_cost : ℕ := total_pretzels * price_per_pretzel
  (barry_pretzels = 12) →
  (total_cost = 42)

theorem solve_pretzel_problem :
  pretzel_problem 12 :=
by
  sorry

end NUMINAMATH_CALUDE_solve_pretzel_problem_l878_87824


namespace NUMINAMATH_CALUDE_function_with_two_symmetry_centers_decomposition_l878_87817

/-- A function has a center of symmetry at a if f(a-x) + f(a+x) = 2f(a) for all real x -/
def HasCenterOfSymmetry (f : ℝ → ℝ) (a : ℝ) : Prop :=
  ∀ x : ℝ, f (a - x) + f (a + x) = 2 * f a

/-- A function is linear if f(x) = mx + b for some real m and b -/
def IsLinear (f : ℝ → ℝ) : Prop :=
  ∃ m b : ℝ, ∀ x : ℝ, f x = m * x + b

/-- A function is periodic if there exists a non-zero real number p such that f(x + p) = f(x) for all real x -/
def IsPeriodic (f : ℝ → ℝ) : Prop :=
  ∃ p : ℝ, p ≠ 0 ∧ ∀ x : ℝ, f (x + p) = f x

/-- Main theorem: A function with at least two centers of symmetry can be written as the sum of a linear function and a periodic function -/
theorem function_with_two_symmetry_centers_decomposition (f : ℝ → ℝ) 
  (h1 : ∃ p q : ℝ, p ≠ q ∧ HasCenterOfSymmetry f p ∧ HasCenterOfSymmetry f q) :
  ∃ g h : ℝ → ℝ, IsLinear g ∧ IsPeriodic h ∧ ∀ x : ℝ, f x = g x + h x := by
  sorry


end NUMINAMATH_CALUDE_function_with_two_symmetry_centers_decomposition_l878_87817


namespace NUMINAMATH_CALUDE_remy_used_19_gallons_l878_87836

/-- Represents the water usage of three people taking showers -/
structure ShowerUsage where
  roman : ℕ
  remy : ℕ
  riley : ℕ

/-- Defines the conditions of the shower usage problem -/
def validShowerUsage (u : ShowerUsage) : Prop :=
  u.remy = 3 * u.roman + 1 ∧
  u.riley = u.roman + u.remy - 2 ∧
  u.roman + u.remy + u.riley = 48

/-- Theorem stating that if the shower usage is valid, Remy used 19 gallons -/
theorem remy_used_19_gallons (u : ShowerUsage) : 
  validShowerUsage u → u.remy = 19 := by
  sorry

#check remy_used_19_gallons

end NUMINAMATH_CALUDE_remy_used_19_gallons_l878_87836


namespace NUMINAMATH_CALUDE_exponential_equation_solution_l878_87867

theorem exponential_equation_solution :
  ∃ (y₁ y₂ : ℝ), y₁ ≠ y₂ ∧
  (∀ y : ℝ, (2 : ℝ) ^ (y^2 - 5*y - 6) = (4 : ℝ) ^ (y - 5) ↔ y = y₁ ∨ y = y₂) ∧
  y₁ + y₂ = 5 :=
sorry

end NUMINAMATH_CALUDE_exponential_equation_solution_l878_87867


namespace NUMINAMATH_CALUDE_probability_at_least_one_karnataka_l878_87861

theorem probability_at_least_one_karnataka (total_students : ℕ) 
  (maharashtra_students : ℕ) (karnataka_students : ℕ) (goa_students : ℕ) 
  (students_to_select : ℕ) : 
  total_students = 10 →
  maharashtra_students = 4 →
  karnataka_students = 3 →
  goa_students = 3 →
  students_to_select = 4 →
  (1 : ℚ) - (Nat.choose (total_students - karnataka_students) students_to_select : ℚ) / 
    (Nat.choose total_students students_to_select : ℚ) = 5 / 6 :=
by sorry

end NUMINAMATH_CALUDE_probability_at_least_one_karnataka_l878_87861


namespace NUMINAMATH_CALUDE_floor_ceiling_expression_l878_87819

theorem floor_ceiling_expression : 
  ⌊⌈(12 / 5 : ℚ)^2⌉ * 3 + 14 / 3⌋ = 22 := by
  sorry

end NUMINAMATH_CALUDE_floor_ceiling_expression_l878_87819


namespace NUMINAMATH_CALUDE_bipartite_ramsey_theorem_l878_87835

/-- A bipartite graph -/
structure BipartiteGraph where
  X : Type
  Y : Type
  E : X → Y → Prop

/-- An edge coloring of a bipartite graph -/
def EdgeColoring (G : BipartiteGraph) := G.X → G.Y → Bool

/-- A homomorphism between bipartite graphs -/
structure BipartiteHomomorphism (G H : BipartiteGraph) where
  φX : G.X → H.X
  φY : G.Y → H.Y
  preserves_edges : ∀ x y, G.E x y → H.E (φX x) (φY y)

/-- The main theorem -/
theorem bipartite_ramsey_theorem :
  ∀ P : BipartiteGraph, ∃ P' : BipartiteGraph,
    ∀ c : EdgeColoring P',
      ∃ φ : BipartiteHomomorphism P P',
        ∃ color : Bool,
          ∀ x y, P.E x y → c (φ.φX x) (φ.φY y) = color :=
sorry

end NUMINAMATH_CALUDE_bipartite_ramsey_theorem_l878_87835


namespace NUMINAMATH_CALUDE_fermat_like_prime_condition_l878_87882

theorem fermat_like_prime_condition (a n : ℕ) (ha : a ≥ 2) (hn : n ≥ 2) 
  (h_prime : Nat.Prime (a^n - 1)) : a = 2 ∧ Nat.Prime n := by
  sorry

end NUMINAMATH_CALUDE_fermat_like_prime_condition_l878_87882


namespace NUMINAMATH_CALUDE_quadratic_solution_sum_l878_87841

theorem quadratic_solution_sum (a b : ℝ) : 
  (∀ x : ℂ, 5 * x^2 - 4 * x + 15 = 0 ↔ x = a + b * I ∨ x = a - b * I) →
  a + b^2 = 81/25 := by
sorry

end NUMINAMATH_CALUDE_quadratic_solution_sum_l878_87841


namespace NUMINAMATH_CALUDE_sphere_volume_l878_87850

theorem sphere_volume (r : ℝ) (h1 : r > 0) (h2 : π = (r ^ 2 - 1 ^ 2)) : 
  (4 / 3 : ℝ) * π * r ^ 3 = (8 * Real.sqrt 2 / 3 : ℝ) * π := by
  sorry

end NUMINAMATH_CALUDE_sphere_volume_l878_87850


namespace NUMINAMATH_CALUDE_summit_conference_attendance_l878_87840

/-- The number of diplomats attending a summit conference -/
def D : ℕ := 150

/-- The number of diplomats who speak French -/
def french_speakers : ℕ := 17

/-- The number of diplomats who do not speak Russian -/
def non_russian_speakers : ℕ := 32

/-- The percentage of diplomats who speak neither French nor Russian -/
def neither_percentage : ℚ := 1/5

/-- The percentage of diplomats who speak both French and Russian -/
def both_percentage : ℚ := 1/10

theorem summit_conference_attendance :
  D = 150 ∧
  french_speakers = 17 ∧
  non_russian_speakers = 32 ∧
  neither_percentage = 1/5 ∧
  both_percentage = 1/10 ∧
  (D : ℚ) * neither_percentage + (D : ℚ) * both_percentage + french_speakers + (D - non_russian_speakers) = D := by
  sorry

#check summit_conference_attendance

end NUMINAMATH_CALUDE_summit_conference_attendance_l878_87840


namespace NUMINAMATH_CALUDE_sum_of_fourth_fifth_sixth_terms_l878_87854

def arithmetic_sequence (a : ℕ → ℤ) : Prop :=
  ∃ d : ℤ, ∀ n : ℕ, a (n + 1) = a n + d

theorem sum_of_fourth_fifth_sixth_terms
  (a : ℕ → ℤ)
  (h_arithmetic : arithmetic_sequence a)
  (h_first_term : a 1 = 2)
  (h_third_term : a 3 = -10) :
  a 4 + a 5 + a 6 = -66 := by
sorry

end NUMINAMATH_CALUDE_sum_of_fourth_fifth_sixth_terms_l878_87854


namespace NUMINAMATH_CALUDE_sqrt_3_minus_3_power_0_minus_2_power_neg_1_l878_87860

theorem sqrt_3_minus_3_power_0_minus_2_power_neg_1 :
  (Real.sqrt 3 - 3) ^ 0 - 2 ^ (-1 : ℤ) = 1/2 := by sorry

end NUMINAMATH_CALUDE_sqrt_3_minus_3_power_0_minus_2_power_neg_1_l878_87860


namespace NUMINAMATH_CALUDE_parallelogram_area_l878_87807

def v : Fin 2 → ℝ := ![5, -3]
def w : Fin 2 → ℝ := ![11, -2]

theorem parallelogram_area : 
  abs (v 0 * w 1 - v 1 * w 0) = 23 := by sorry

end NUMINAMATH_CALUDE_parallelogram_area_l878_87807


namespace NUMINAMATH_CALUDE_function_ratio_bounds_l878_87849

open Real

theorem function_ratio_bounds (f : ℝ → ℝ) (hf : ∀ x > 0, f x > 0)
  (hf' : ∀ x > 0, 2 * f x < x * (deriv f x) ∧ x * (deriv f x) < 3 * f x) :
  8/27 < f 2 / f 3 ∧ f 2 / f 3 < 4/9 := by
  sorry

end NUMINAMATH_CALUDE_function_ratio_bounds_l878_87849


namespace NUMINAMATH_CALUDE_percentage_change_xyz_l878_87899

theorem percentage_change_xyz (x y z : ℝ) (x_pos : x > 0) (y_pos : y > 0) (z_pos : z > 0) :
  let x' := 0.8 * x
  let y' := 0.8 * y
  let z' := 1.1 * z
  (x' * y' * z' - x * y * z) / (x * y * z) = -0.296 :=
by sorry

end NUMINAMATH_CALUDE_percentage_change_xyz_l878_87899


namespace NUMINAMATH_CALUDE_seven_point_four_five_repeating_equals_82_11_l878_87844

/-- Represents a repeating decimal with an integer part and a repeating fractional part -/
structure RepeatingDecimal where
  integerPart : ℤ
  repeatingPart : ℕ

/-- Converts a RepeatingDecimal to a rational number -/
def repeatingDecimalToRational (x : RepeatingDecimal) : ℚ :=
  x.integerPart + (x.repeatingPart : ℚ) / (99 : ℚ)

/-- The repeating decimal 7.45̄ -/
def seven_point_four_five_repeating : RepeatingDecimal :=
  { integerPart := 7, repeatingPart := 45 }

theorem seven_point_four_five_repeating_equals_82_11 :
  repeatingDecimalToRational seven_point_four_five_repeating = 82 / 11 := by
  sorry

end NUMINAMATH_CALUDE_seven_point_four_five_repeating_equals_82_11_l878_87844


namespace NUMINAMATH_CALUDE_sum_of_a_and_b_l878_87809

theorem sum_of_a_and_b (a b : ℝ) : 
  (∀ x : ℝ, (x-a)/(x-b) > 0 ↔ x ∈ Set.Ioi 4 ∪ Set.Iic 1) → 
  a + b = 5 := by
sorry

end NUMINAMATH_CALUDE_sum_of_a_and_b_l878_87809


namespace NUMINAMATH_CALUDE_orchard_apples_count_l878_87804

theorem orchard_apples_count (total_apples : ℕ) : 
  (40 : ℕ) * total_apples = (100 : ℕ) * (40 : ℕ) * (24 : ℕ) / ((100 : ℕ) - (70 : ℕ)) →
  total_apples = 200 := by
  sorry

end NUMINAMATH_CALUDE_orchard_apples_count_l878_87804


namespace NUMINAMATH_CALUDE_no_integer_solution_l878_87842

theorem no_integer_solution : ¬∃ (x : ℤ), ((7 * (x + 5)) / 5) - 5 = 33 := by
  sorry

end NUMINAMATH_CALUDE_no_integer_solution_l878_87842


namespace NUMINAMATH_CALUDE_remaining_terms_geometric_l878_87827

/-- A geometric sequence with first term a and common ratio q -/
def geometric_sequence (a q : ℝ) : ℕ → ℝ := fun n => a * q^(n - 1)

theorem remaining_terms_geometric (a q : ℝ) (k : ℕ) :
  let original_seq := geometric_sequence a q
  let remaining_seq := fun n => original_seq (n + k)
  ∃ a', remaining_seq = geometric_sequence a' q :=
sorry

end NUMINAMATH_CALUDE_remaining_terms_geometric_l878_87827


namespace NUMINAMATH_CALUDE_repeating_decimal_equals_fraction_l878_87893

/-- The value of the repeating decimal 0.565656... -/
def repeating_decimal : ℚ := 0.56565656

/-- The fraction 56/99 -/
def fraction : ℚ := 56 / 99

/-- Theorem stating that the repeating decimal 0.565656... is equal to 56/99 -/
theorem repeating_decimal_equals_fraction : repeating_decimal = fraction := by
  sorry

end NUMINAMATH_CALUDE_repeating_decimal_equals_fraction_l878_87893


namespace NUMINAMATH_CALUDE_language_group_selection_ways_l878_87833

/-- Represents a group of people who know languages -/
structure LanguageGroup where
  total : Nat
  english : Nat
  japanese : Nat
  both : Nat

/-- The number of ways to select one person who knows English and another who knows Japanese -/
def selectWays (group : LanguageGroup) : Nat :=
  (group.english - group.both) * group.japanese + group.both * (group.japanese - 1)

/-- Theorem stating the number of ways to select people in the given scenario -/
theorem language_group_selection_ways :
  ∃ (group : LanguageGroup),
    group.total = 9 ∧
    group.english = 7 ∧
    group.japanese = 3 ∧
    group.total = (group.english - group.both) + (group.japanese - group.both) + group.both ∧
    selectWays group = 20 := by
  sorry

end NUMINAMATH_CALUDE_language_group_selection_ways_l878_87833


namespace NUMINAMATH_CALUDE_max_diagonal_bd_l878_87884

/-- Represents the side lengths of a quadrilateral --/
structure QuadrilateralSides where
  AB : ℕ
  BC : ℕ
  CD : ℕ
  DA : ℕ

/-- Checks if the given side lengths form a valid cyclic quadrilateral --/
def is_valid_cyclic_quadrilateral (sides : QuadrilateralSides) : Prop :=
  sides.AB < 10 ∧ sides.BC < 10 ∧ sides.CD < 10 ∧ sides.DA < 10 ∧
  sides.AB ≠ sides.BC ∧ sides.AB ≠ sides.CD ∧ sides.AB ≠ sides.DA ∧
  sides.BC ≠ sides.CD ∧ sides.BC ≠ sides.DA ∧ sides.CD ≠ sides.DA ∧
  sides.BC + sides.CD = sides.AB + sides.DA

/-- Calculates the square of the diagonal BD --/
def diagonal_bd_squared (sides : QuadrilateralSides) : ℚ :=
  (sides.AB^2 + sides.BC^2 + sides.CD^2 + sides.DA^2) / 2

theorem max_diagonal_bd (sides : QuadrilateralSides) :
  is_valid_cyclic_quadrilateral sides →
  diagonal_bd_squared sides ≤ 191/2 :=
sorry

end NUMINAMATH_CALUDE_max_diagonal_bd_l878_87884


namespace NUMINAMATH_CALUDE_sum_of_ages_l878_87887

-- Define the present ages of father and son
def father_age : ℚ := sorry
def son_age : ℚ := sorry

-- Define the conditions
def present_ratio : father_age / son_age = 7 / 4 := sorry
def future_ratio : (father_age + 10) / (son_age + 10) = 5 / 3 := sorry

-- Theorem to prove
theorem sum_of_ages : father_age + son_age = 220 := by sorry

end NUMINAMATH_CALUDE_sum_of_ages_l878_87887


namespace NUMINAMATH_CALUDE_triangle_perimeter_l878_87877

theorem triangle_perimeter (a b c : ℕ) : 
  a = 7 → b = 2 → Odd c → a + b + c = 16 := by sorry

end NUMINAMATH_CALUDE_triangle_perimeter_l878_87877


namespace NUMINAMATH_CALUDE_coin_order_correct_l878_87806

-- Define the type for coins
inductive Coin : Type
  | A | B | C | D | E | F

-- Define a relation for one coin being above another
def IsAbove : Coin → Coin → Prop := sorry

-- Define the correct order of coins
def CorrectOrder : List Coin := [Coin.F, Coin.C, Coin.E, Coin.D, Coin.A, Coin.B]

-- State the theorem
theorem coin_order_correct (coins : List Coin) 
  (h1 : IsAbove Coin.F Coin.C)
  (h2 : IsAbove Coin.F Coin.E)
  (h3 : IsAbove Coin.F Coin.A)
  (h4 : IsAbove Coin.F Coin.D)
  (h5 : IsAbove Coin.F Coin.B)
  (h6 : IsAbove Coin.C Coin.A)
  (h7 : IsAbove Coin.C Coin.D)
  (h8 : IsAbove Coin.C Coin.B)
  (h9 : IsAbove Coin.E Coin.A)
  (h10 : IsAbove Coin.E Coin.B)
  (h11 : IsAbove Coin.D Coin.B)
  (h12 : coins.length = 6)
  (h13 : coins.Nodup)
  (h14 : ∀ c, c ∈ coins ↔ c ∈ [Coin.A, Coin.B, Coin.C, Coin.D, Coin.E, Coin.F]) :
  coins = CorrectOrder := by sorry


end NUMINAMATH_CALUDE_coin_order_correct_l878_87806


namespace NUMINAMATH_CALUDE_gumball_sale_revenue_l878_87828

theorem gumball_sale_revenue (num_gumballs : ℕ) (price_per_gumball : ℕ) : 
  num_gumballs = 4 → price_per_gumball = 8 → num_gumballs * price_per_gumball = 32 := by
  sorry

end NUMINAMATH_CALUDE_gumball_sale_revenue_l878_87828


namespace NUMINAMATH_CALUDE_sin_plus_two_cos_for_point_l878_87802

/-- Given a point P(-3,4) on the terminal side of angle α, prove that sin α + 2 cos α = -2/5 -/
theorem sin_plus_two_cos_for_point (α : Real) :
  let P : ℝ × ℝ := (-3, 4)
  let r : ℝ := Real.sqrt (P.1^2 + P.2^2)
  Real.sin α = P.2 / r ∧ Real.cos α = P.1 / r →
  Real.sin α + 2 * Real.cos α = -2/5 := by
sorry


end NUMINAMATH_CALUDE_sin_plus_two_cos_for_point_l878_87802


namespace NUMINAMATH_CALUDE_perpendicular_planes_from_lines_l878_87812

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the relations
variable (perpendicular : Line → Plane → Prop)
variable (parallel_lines : Line → Line → Prop)
variable (parallel_plane_line : Plane → Line → Prop)
variable (perpendicular_planes : Plane → Plane → Prop)

-- State the theorem
theorem perpendicular_planes_from_lines
  (m n : Line) (α β : Plane)
  (h1 : perpendicular m α)
  (h2 : parallel_lines m n)
  (h3 : parallel_plane_line β n) :
  perpendicular_planes α β :=
sorry

end NUMINAMATH_CALUDE_perpendicular_planes_from_lines_l878_87812


namespace NUMINAMATH_CALUDE_division_problem_l878_87852

theorem division_problem (a b q : ℕ) (h1 : a - b = 1365) (h2 : a = 1620) (h3 : a = b * q + 15) : q = 6 := by
  sorry

end NUMINAMATH_CALUDE_division_problem_l878_87852


namespace NUMINAMATH_CALUDE_complex_product_quadrant_l878_87874

theorem complex_product_quadrant : 
  let z : ℂ := (1 + 3*I) * (3 - I)
  (z.re > 0) ∧ (z.im > 0) :=
by
  sorry

end NUMINAMATH_CALUDE_complex_product_quadrant_l878_87874


namespace NUMINAMATH_CALUDE_f_range_on_interval_l878_87848

open Real

noncomputable def f (x : ℝ) : ℝ := 2 * cos x - 2 * sin x ^ 2

theorem f_range_on_interval :
  ∀ x ∈ Set.Icc (π / 2) (3 * π / 2),
  ∃ y ∈ Set.Icc (-5/2) (-2), f x = y ∧
  ∀ z, f x = z → z ∈ Set.Icc (-5/2) (-2) :=
by sorry

end NUMINAMATH_CALUDE_f_range_on_interval_l878_87848


namespace NUMINAMATH_CALUDE_third_guinea_pig_extra_food_l878_87811

/-- The amount of food eaten by Rollo's guinea pigs -/
def GuineaPigFood : Type := 
  { food : ℕ × ℕ × ℕ // 
    food.1 = 2 ∧ 
    food.2.1 = 2 * food.1 ∧ 
    food.1 + food.2.1 + food.2.2 = 13 }

theorem third_guinea_pig_extra_food (food : GuineaPigFood) : 
  food.val.2.2 - food.val.2.1 = 3 := by sorry

end NUMINAMATH_CALUDE_third_guinea_pig_extra_food_l878_87811


namespace NUMINAMATH_CALUDE_p_is_power_of_two_l878_87814

def is_power_of_two (x : ℕ) : Prop := ∃ k : ℕ, x = 2^k

theorem p_is_power_of_two (p : ℕ) (h1 : p > 2) (h2 : ∃! d : ℕ, Odd d ∧ (32 * p) % d = 0) :
  is_power_of_two p := by
sorry

end NUMINAMATH_CALUDE_p_is_power_of_two_l878_87814


namespace NUMINAMATH_CALUDE_fraction_simplification_l878_87813

theorem fraction_simplification (b y : ℝ) (h : b^2 + y^3 ≠ 0) :
  (Real.sqrt (b^2 + y^3) - (y^3 - b^2) / Real.sqrt (b^2 + y^3)) / (b^2 + y^3) = 
  2 * b^2 / (b^2 + y^3)^(3/2) := by
  sorry

end NUMINAMATH_CALUDE_fraction_simplification_l878_87813


namespace NUMINAMATH_CALUDE_total_weight_of_four_l878_87831

theorem total_weight_of_four (jim steve stan tim : ℕ) : 
  jim = 110 →
  steve = jim - 8 →
  stan = steve + 5 →
  tim = stan + 12 →
  jim + steve + stan + tim = 438 := by
sorry

end NUMINAMATH_CALUDE_total_weight_of_four_l878_87831


namespace NUMINAMATH_CALUDE_problem_solution_l878_87880

theorem problem_solution (x y : ℚ) 
  (sum_eq : x + y = 500)
  (ratio_eq : x / y = 7 / 8) :
  y - x = 100 / 3 := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l878_87880


namespace NUMINAMATH_CALUDE_triangle_division_ratio_l878_87829

/-- Given a triangle ABC, this theorem proves that if point F divides side AC in the ratio 2:3,
    G is the midpoint of BF, and E is the point of intersection of side BC and AG,
    then E divides side BC in the ratio 2:5. -/
theorem triangle_division_ratio (A B C F G E : ℝ × ℝ) : 
  (∃ k : ℝ, F = A + (2/5 : ℝ) • (C - A)) →  -- F divides AC in ratio 2:3
  G = B + (1/2 : ℝ) • (F - B) →              -- G is midpoint of BF
  (∃ t : ℝ, E = B + t • (C - B) ∧ 
            E = A + t • (G - A)) →           -- E is intersection of BC and AG
  (∃ s : ℝ, E = B + (2/7 : ℝ) • (C - B)) :=   -- E divides BC in ratio 2:5
by sorry


end NUMINAMATH_CALUDE_triangle_division_ratio_l878_87829


namespace NUMINAMATH_CALUDE_cubic_extremum_l878_87801

/-- Given a cubic function f(x) = x³ + 3ax² + bx + a² with an extremum of 0 at x = -1,
    prove that a - b = -7 -/
theorem cubic_extremum (a b : ℝ) : 
  let f : ℝ → ℝ := λ x => x^3 + 3*a*x^2 + b*x + a^2
  (∃ ε > 0, ∀ x ∈ Set.Ioo (-1-ε) (-1+ε), f x ≥ f (-1)) ∧ 
  (∃ ε > 0, ∀ x ∈ Set.Ioo (-1-ε) (-1+ε), f x ≤ f (-1)) ∧
  f (-1) = 0 →
  a - b = -7 :=
by sorry


end NUMINAMATH_CALUDE_cubic_extremum_l878_87801


namespace NUMINAMATH_CALUDE_ritas_money_theorem_l878_87816

/-- Calculates the remaining money after Rita's purchases --/
def ritas_remaining_money (initial_amount dresses_cost pants_cost jackets_cost transportation : ℕ) : ℕ :=
  initial_amount - (5 * dresses_cost + 3 * pants_cost + 4 * jackets_cost + transportation)

/-- Theorem stating that Rita's remaining money is 139 --/
theorem ritas_money_theorem :
  ritas_remaining_money 400 20 12 30 5 = 139 := by
  sorry

end NUMINAMATH_CALUDE_ritas_money_theorem_l878_87816


namespace NUMINAMATH_CALUDE_quadratic_equation_roots_l878_87869

theorem quadratic_equation_roots (b c : ℝ) : 
  (∀ x, x^2 - b*x + c = 0 ↔ x = 1 ∨ x = -2) → 
  b = -1 ∧ c = -2 := by
sorry

end NUMINAMATH_CALUDE_quadratic_equation_roots_l878_87869


namespace NUMINAMATH_CALUDE_ruth_shared_apples_l878_87830

/-- The number of apples Ruth shared with Peter -/
def apples_shared (initial : ℕ) (remaining : ℕ) : ℕ :=
  initial - remaining

/-- Theorem stating that Ruth shared 5 apples with Peter -/
theorem ruth_shared_apples : apples_shared 89 84 = 5 := by
  sorry

end NUMINAMATH_CALUDE_ruth_shared_apples_l878_87830


namespace NUMINAMATH_CALUDE_largest_prime_divisor_l878_87800

def base_8_number : ℕ := 201021022

theorem largest_prime_divisor :
  let decimal_number := 35661062
  ∃ (p : ℕ), Nat.Prime p ∧ p ∣ decimal_number ∧ ∀ (q : ℕ), Nat.Prime q → q ∣ decimal_number → q ≤ p ∧ p = 17830531 := by
  sorry

#eval base_8_number

end NUMINAMATH_CALUDE_largest_prime_divisor_l878_87800


namespace NUMINAMATH_CALUDE_infinite_geometric_series_first_term_l878_87845

theorem infinite_geometric_series_first_term 
  (r : ℝ) (S : ℝ) (a : ℝ) 
  (h1 : r = -1/3) 
  (h2 : S = 9) 
  (h3 : S = a / (1 - r)) : a = 12 := by
sorry

end NUMINAMATH_CALUDE_infinite_geometric_series_first_term_l878_87845


namespace NUMINAMATH_CALUDE_arithmetic_calculation_l878_87837

theorem arithmetic_calculation : 2^3 + 2 * 5 - 3 + 6 = 21 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_calculation_l878_87837


namespace NUMINAMATH_CALUDE_ball_probability_l878_87846

theorem ball_probability (p_red p_yellow p_blue : ℝ) : 
  p_red = 0.48 → p_yellow = 0.35 → p_red + p_yellow + p_blue = 1 → p_blue = 0.17 := by
  sorry

end NUMINAMATH_CALUDE_ball_probability_l878_87846


namespace NUMINAMATH_CALUDE_consecutive_product_not_power_l878_87889

theorem consecutive_product_not_power (m k n : ℕ) (hn : n > 1) :
  m * (m + 1) ≠ k^n := by
  sorry

end NUMINAMATH_CALUDE_consecutive_product_not_power_l878_87889


namespace NUMINAMATH_CALUDE_inequality_solution_set_l878_87826

theorem inequality_solution_set : 
  {x : ℝ | (3 : ℝ) / (5 - 3 * x) > 1} = {x : ℝ | 2/3 < x ∧ x < 5/3} := by
  sorry

end NUMINAMATH_CALUDE_inequality_solution_set_l878_87826


namespace NUMINAMATH_CALUDE_A_share_of_profit_l878_87822

/-- Calculates the share of profit for partner A in a business partnership --/
def calculate_A_share_of_profit (a_initial : ℕ) (b_initial : ℕ) (a_withdrawal : ℕ) (b_addition : ℕ) (months_before_change : ℕ) (total_months : ℕ) (total_profit : ℕ) : ℚ :=
  let a_investment_months := a_initial * months_before_change + (a_initial - a_withdrawal) * (total_months - months_before_change)
  let b_investment_months := b_initial * months_before_change + (b_initial + b_addition) * (total_months - months_before_change)
  let total_investment_months := a_investment_months + b_investment_months
  (a_investment_months : ℚ) / total_investment_months * total_profit

theorem A_share_of_profit :
  calculate_A_share_of_profit 3000 4000 1000 1000 8 12 630 = 240 := by
  sorry

end NUMINAMATH_CALUDE_A_share_of_profit_l878_87822


namespace NUMINAMATH_CALUDE_sin_alpha_minus_9pi_over_2_l878_87855

theorem sin_alpha_minus_9pi_over_2 (α : Real) 
  (h1 : π / 2 < α) 
  (h2 : α < π) 
  (h3 : 3 * Real.sin (2 * α) = 2 * Real.cos α) : 
  Real.sin (α - 9 * π / 2) = 2 * Real.sqrt 2 / 3 := by
  sorry

end NUMINAMATH_CALUDE_sin_alpha_minus_9pi_over_2_l878_87855


namespace NUMINAMATH_CALUDE_decimal_to_binary_38_l878_87847

theorem decimal_to_binary_38 : 
  (38 : ℕ).digits 2 = [0, 1, 1, 0, 0, 1] :=
sorry

end NUMINAMATH_CALUDE_decimal_to_binary_38_l878_87847


namespace NUMINAMATH_CALUDE_num_adoption_ways_l878_87876

/-- The number of parrots available for adoption -/
def num_parrots : ℕ := 20

/-- The number of snakes available for adoption -/
def num_snakes : ℕ := 10

/-- The number of rabbits available for adoption -/
def num_rabbits : ℕ := 12

/-- The set of possible animal types -/
inductive AnimalType
| Parrot
| Snake
| Rabbit

/-- A function representing Emily's constraint -/
def emily_constraint (a : AnimalType) : Prop :=
  a = AnimalType.Parrot ∨ a = AnimalType.Rabbit

/-- A function representing John's constraint (can adopt any animal) -/
def john_constraint (a : AnimalType) : Prop := True

/-- A function representing Susan's constraint -/
def susan_constraint (a : AnimalType) : Prop :=
  a = AnimalType.Snake

/-- The theorem stating the number of ways to adopt animals -/
theorem num_adoption_ways :
  (num_parrots * num_snakes * num_rabbits) +
  (num_rabbits * num_snakes * num_parrots) = 4800 := by
  sorry

end NUMINAMATH_CALUDE_num_adoption_ways_l878_87876


namespace NUMINAMATH_CALUDE_minor_premise_identification_l878_87803

-- Define the basic shapes
inductive Shape
| Rectangle
| Parallelogram
| Triangle

-- Define the properties of shapes
def isParallelogram : Shape → Prop
  | Shape.Rectangle => true
  | Shape.Parallelogram => true
  | Shape.Triangle => false

-- Define the syllogism structure
structure Syllogism where
  majorPremise : Prop
  minorPremise : Prop
  conclusion : Prop

-- Define our specific syllogism
def ourSyllogism : Syllogism := {
  majorPremise := isParallelogram Shape.Rectangle
  minorPremise := ¬ isParallelogram Shape.Triangle
  conclusion := Shape.Triangle ≠ Shape.Rectangle
}

-- Theorem to prove
theorem minor_premise_identification :
  ourSyllogism.minorPremise = ¬ isParallelogram Shape.Triangle :=
by sorry

end NUMINAMATH_CALUDE_minor_premise_identification_l878_87803


namespace NUMINAMATH_CALUDE_mira_jogging_hours_l878_87890

/-- Mira's jogging problem -/
theorem mira_jogging_hours :
  ∀ (h : ℝ),
  (h > 0) →  -- Ensure positive jogging time
  (5 * h * 5 = 50) →  -- Total distance covered in 5 days
  h = 2 := by
sorry

end NUMINAMATH_CALUDE_mira_jogging_hours_l878_87890


namespace NUMINAMATH_CALUDE_f_inequality_l878_87856

/-- The function f(x) = x^2 - 2x + c -/
def f (c : ℝ) (x : ℝ) : ℝ := x^2 - 2*x + c

/-- Theorem stating that f(0) < f(4) < f(-4) for any real c -/
theorem f_inequality (c : ℝ) : f c 0 < f c 4 ∧ f c 4 < f c (-4) := by
  sorry

end NUMINAMATH_CALUDE_f_inequality_l878_87856


namespace NUMINAMATH_CALUDE_f_increasing_on_negative_reals_l878_87897

-- Define the function f
def f (m : ℝ) (x : ℝ) : ℝ := (m - 1) * x^2 + 2 * m * x + 1

-- State the theorem
theorem f_increasing_on_negative_reals (m : ℝ) :
  (∀ x : ℝ, f m x = f m (-x)) →  -- f is an even function
  (∀ x y : ℝ, x < y ∧ y ≤ 0 → f m x < f m y) :=  -- f is increasing on (-∞, 0]
by
  sorry

end NUMINAMATH_CALUDE_f_increasing_on_negative_reals_l878_87897


namespace NUMINAMATH_CALUDE_tangent_circles_parallelism_l878_87883

-- Define the types for our points and circles
variable (Point : Type) (Circle : Type)

-- Define the basic geometric relations
variable (on_circle : Point → Circle → Prop)
variable (on_line : Point → Point → Point → Prop)
variable (between : Point → Point → Point → Prop)
variable (tangent : Point → Point → Circle → Prop)
variable (intersect : Circle → Circle → Point → Point → Prop)
variable (cuts : Point → Point → Circle → Point → Prop)
variable (parallel : Point → Point → Point → Point → Prop)

-- Define our specific points and circles
variable (A B C P Q R S X Y Z : Point)
variable (C1 C2 : Circle)

-- State the theorem
theorem tangent_circles_parallelism 
  (h1 : intersect C1 C2 A B)
  (h2 : on_line A B C ∧ between A B C)
  (h3 : on_circle P C1 ∧ on_circle Q C2)
  (h4 : tangent C P C1 ∧ tangent C Q C2)
  (h5 : ¬on_circle P C2 ∧ ¬on_circle Q C1)
  (h6 : cuts P Q C1 R ∧ cuts P Q C2 S)
  (h7 : R ≠ P ∧ R ≠ Q ∧ R ≠ B ∧ S ≠ P ∧ S ≠ Q ∧ S ≠ B)
  (h8 : cuts C R C1 X ∧ cuts C S C2 Y)
  (h9 : on_line X Y Z) :
  parallel S Z Q X ↔ parallel P Z R X :=
sorry

end NUMINAMATH_CALUDE_tangent_circles_parallelism_l878_87883


namespace NUMINAMATH_CALUDE_exists_valid_statement_l878_87815

-- Define the types of people on the island
inductive PersonType
  | Knight
  | Liar
  | Normal

-- Define a statement type
structure Statement where
  content : String
  canBeMadeBy : PersonType → Prop
  truthValueKnown : Prop

-- Define the property of a valid statement
def validStatement (s : Statement) : Prop :=
  (s.canBeMadeBy PersonType.Normal) ∧
  (¬s.canBeMadeBy PersonType.Knight) ∧
  (¬s.canBeMadeBy PersonType.Liar) ∧
  (¬s.truthValueKnown)

-- Theorem: There exists a valid statement
theorem exists_valid_statement : ∃ s : Statement, validStatement s := by
  sorry

end NUMINAMATH_CALUDE_exists_valid_statement_l878_87815


namespace NUMINAMATH_CALUDE_pool_radius_l878_87878

/-- Proves that a circular pool with a surrounding concrete wall has a radius of 20 feet
    given specific conditions on the wall width and area ratio. -/
theorem pool_radius (r : ℝ) : 
  r > 0 → -- The radius is positive
  (π * ((r + 4)^2 - r^2) = (11/25) * π * r^2) → -- Area ratio condition
  r = 20 := by
  sorry

end NUMINAMATH_CALUDE_pool_radius_l878_87878


namespace NUMINAMATH_CALUDE_complex_equation_implies_unit_magnitude_l878_87866

theorem complex_equation_implies_unit_magnitude (z : ℂ) : 
  11 * z^10 + 10 * Complex.I * z^9 + 10 * Complex.I * z - 11 = 0 → Complex.abs z = 1 := by
  sorry

end NUMINAMATH_CALUDE_complex_equation_implies_unit_magnitude_l878_87866


namespace NUMINAMATH_CALUDE_expand_product_l878_87886

theorem expand_product (x : ℝ) : (x + 5) * (x + 7) = x^2 + 12*x + 35 := by
  sorry

end NUMINAMATH_CALUDE_expand_product_l878_87886


namespace NUMINAMATH_CALUDE_lcm_of_16_27_35_l878_87825

theorem lcm_of_16_27_35 : Nat.lcm (Nat.lcm 16 27) 35 = 15120 := by
  sorry

end NUMINAMATH_CALUDE_lcm_of_16_27_35_l878_87825


namespace NUMINAMATH_CALUDE_alcohol_mixture_percentage_l878_87857

/-- Calculates the final alcohol percentage in a solution after partial replacement -/
theorem alcohol_mixture_percentage 
  (initial_volume : ℝ)
  (initial_percentage : ℝ)
  (drained_volume : ℝ)
  (replacement_percentage : ℝ)
  (h1 : initial_volume = 1)
  (h2 : initial_percentage = 0.75)
  (h3 : drained_volume = 0.4)
  (h4 : replacement_percentage = 0.5)
  (h5 : initial_volume - drained_volume + drained_volume = initial_volume) :
  let remaining_volume := initial_volume - drained_volume
  let remaining_alcohol := remaining_volume * initial_percentage
  let added_alcohol := drained_volume * replacement_percentage
  let total_alcohol := remaining_alcohol + added_alcohol
  let final_percentage := total_alcohol / initial_volume
  final_percentage = 0.65 := by sorry

end NUMINAMATH_CALUDE_alcohol_mixture_percentage_l878_87857


namespace NUMINAMATH_CALUDE_complex_expression_equals_nine_l878_87818

theorem complex_expression_equals_nine :
  (Real.rpow 1.5 (1/3) * Real.rpow 12 (1/6))^2 + 8 * Real.rpow 1 0.75 - Real.rpow (-1/4) (-2) - 5 * Real.rpow 0.125 0 = 9 := by
  sorry

end NUMINAMATH_CALUDE_complex_expression_equals_nine_l878_87818


namespace NUMINAMATH_CALUDE_arithmetic_sequence_sum_l878_87859

/-- An arithmetic sequence with first term a₁ and common difference d -/
structure ArithmeticSequence where
  a₁ : ℤ
  d : ℤ

/-- Sum of the first n terms of an arithmetic sequence -/
def sum_n (seq : ArithmeticSequence) (n : ℕ) : ℤ :=
  n * seq.a₁ + (n * (n - 1) / 2) * seq.d

theorem arithmetic_sequence_sum (seq : ArithmeticSequence) :
  seq.a₁ = -2014 →
  (sum_n seq 2012 / 2012 : ℚ) - (sum_n seq 10 / 10 : ℚ) = 2002 →
  sum_n seq 2016 = 2016 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_sum_l878_87859


namespace NUMINAMATH_CALUDE_min_m_for_24m_eq_n4_l878_87823

theorem min_m_for_24m_eq_n4 (m n : ℕ+) (h : 24 * m = n ^ 4) :
  ∀ k : ℕ+, 24 * k = (k : ℕ+) ^ 4 → m ≤ k :=
sorry

end NUMINAMATH_CALUDE_min_m_for_24m_eq_n4_l878_87823


namespace NUMINAMATH_CALUDE_fraction_equality_l878_87872

theorem fraction_equality (a b : ℚ) (h : b / a = 5 / 13) : 
  (a - b) / (a + b) = 4 / 9 := by sorry

end NUMINAMATH_CALUDE_fraction_equality_l878_87872


namespace NUMINAMATH_CALUDE_solution_set_of_polynomial_equation_l878_87870

theorem solution_set_of_polynomial_equation :
  let S := {x : ℝ | x = 0 ∨ 
                   x = Real.sqrt ((5 + Real.sqrt 5) / 2) ∨ 
                   x = -Real.sqrt ((5 + Real.sqrt 5) / 2) ∨ 
                   x = Real.sqrt ((5 - Real.sqrt 5) / 2) ∨ 
                   x = -Real.sqrt ((5 - Real.sqrt 5) / 2)}
  ∀ x : ℝ, (5*x - 5*x^3 + x^5 = 0) ↔ x ∈ S :=
by sorry

end NUMINAMATH_CALUDE_solution_set_of_polynomial_equation_l878_87870


namespace NUMINAMATH_CALUDE_group_size_l878_87896

theorem group_size (average_increase : ℝ) (weight_difference : ℝ) :
  average_increase = 3.5 →
  weight_difference = 28 →
  weight_difference = average_increase * 8 :=
by sorry

end NUMINAMATH_CALUDE_group_size_l878_87896


namespace NUMINAMATH_CALUDE_jason_initial_cards_l878_87898

theorem jason_initial_cards (cards_sold : ℕ) (cards_remaining : ℕ) : 
  cards_sold = 224 → cards_remaining = 452 → cards_sold + cards_remaining = 676 := by
  sorry

end NUMINAMATH_CALUDE_jason_initial_cards_l878_87898


namespace NUMINAMATH_CALUDE_unique_four_digit_numbers_l878_87868

theorem unique_four_digit_numbers : ∃! (x y : ℕ), 
  (1000 ≤ x ∧ x < 10000) ∧ 
  (1000 ≤ y ∧ y < 10000) ∧ 
  y > x ∧ 
  (∃ (a n : ℕ), 1 ≤ a ∧ a < 10 ∧ y = a * 10^n) ∧
  (x / 1000 + (x / 100) % 10 = y - x) ∧
  (y - x = 5 * (y / 1000)) ∧
  x = 1990 ∧ 
  y = 2000 := by
sorry

end NUMINAMATH_CALUDE_unique_four_digit_numbers_l878_87868


namespace NUMINAMATH_CALUDE_two_zeros_sum_less_than_neg_two_l878_87871

noncomputable section

-- Define the functions f and g
def f (x : ℝ) : ℝ := x * Real.exp x
def g (x : ℝ) : ℝ := (x + 1)^2

-- Define the function G
def G (a : ℝ) (x : ℝ) : ℝ := a * f x + g x

-- Theorem statement
theorem two_zeros_sum_less_than_neg_two (a : ℝ) (x₁ x₂ : ℝ) :
  a > 0 →
  G a x₁ = 0 →
  G a x₂ = 0 →
  x₁ ≠ x₂ →
  x₁ + x₂ + 2 < 0 :=
by sorry

end

end NUMINAMATH_CALUDE_two_zeros_sum_less_than_neg_two_l878_87871


namespace NUMINAMATH_CALUDE_carly_butterfly_practice_l878_87808

/-- The number of days Carly practices butterfly stroke per week -/
def butterfly_days : ℕ := sorry

/-- Hours of butterfly stroke practice per day -/
def butterfly_hours_per_day : ℕ := 3

/-- Days of backstroke practice per week -/
def backstroke_days_per_week : ℕ := 6

/-- Hours of backstroke practice per day -/
def backstroke_hours_per_day : ℕ := 2

/-- Total hours of swimming practice per month -/
def total_practice_hours : ℕ := 96

/-- Number of weeks in a month -/
def weeks_per_month : ℕ := 4

theorem carly_butterfly_practice :
  butterfly_days = 4 ∧
  butterfly_days * butterfly_hours_per_day * weeks_per_month +
  backstroke_days_per_week * backstroke_hours_per_day * weeks_per_month =
  total_practice_hours :=
sorry

end NUMINAMATH_CALUDE_carly_butterfly_practice_l878_87808


namespace NUMINAMATH_CALUDE_line_plane_relationship_l878_87839

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the relationships between lines and planes
variable (parallel : Plane → Plane → Prop)
variable (perpendicular : Line → Plane → Prop)
variable (perpendicular_lines : Line → Line → Prop)
variable (contained_in : Line → Plane → Prop)
variable (parallel_line_plane : Line → Plane → Prop)

-- State the theorem
theorem line_plane_relationship 
  (m n : Line) (α β : Plane) 
  (h1 : parallel α β) 
  (h2 : perpendicular m α) 
  (h3 : perpendicular_lines m n) :
  contained_in n β ∨ parallel_line_plane n β :=
sorry

end NUMINAMATH_CALUDE_line_plane_relationship_l878_87839


namespace NUMINAMATH_CALUDE_expression_value_l878_87851

theorem expression_value (a b : ℤ) (ha : a = 4) (hb : b = -2) : 
  -a - b^2 + a*b + a^2 = 0 := by sorry

end NUMINAMATH_CALUDE_expression_value_l878_87851


namespace NUMINAMATH_CALUDE_missile_interception_time_l878_87862

/-- The time for a missile to intercept a circling plane -/
theorem missile_interception_time
  (r : ℝ)             -- radius of the plane's circular path
  (v : ℝ)             -- speed of both the plane and the missile
  (h : r = 10)        -- given radius is 10 km
  (k : v = 1000)      -- given speed is 1000 km/h
  : ∃ t : ℝ,          -- there exists a time t such that
    t = 18 * Real.pi ∧ -- t equals 18π
    t * (5 / 18) = (2 * Real.pi * r) / 4 / v -- t converted to hours equals quarter circumference divided by speed
    :=
by sorry

end NUMINAMATH_CALUDE_missile_interception_time_l878_87862

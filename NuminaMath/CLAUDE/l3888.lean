import Mathlib

namespace NUMINAMATH_CALUDE_broadcast_orders_count_l3888_388893

/-- The number of ways to arrange 6 commercial ads and 2 public service ads 
    with specific constraints -/
def broadcast_orders : ℕ :=
  let n_commercials : ℕ := 6
  let n_public_service : ℕ := 2
  let n_spaces : ℕ := n_commercials - 1
  let ways_to_place_public_service : ℕ := n_spaces * (n_spaces - 2)
  Nat.factorial n_commercials * ways_to_place_public_service

/-- Theorem stating the number of different broadcast orders -/
theorem broadcast_orders_count :
  broadcast_orders = 10800 := by
  sorry

end NUMINAMATH_CALUDE_broadcast_orders_count_l3888_388893


namespace NUMINAMATH_CALUDE_unique_tangent_implies_radius_l3888_388820

/-- A circle in the x-y plane -/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

/-- A point in the x-y plane -/
def Point := ℝ × ℝ

/-- The number of tangent lines from a point to a circle -/
def numTangentLines (c : Circle) (p : Point) : ℕ := sorry

/-- The distance between two points -/
def distance (p1 p2 : Point) : ℝ := sorry

theorem unique_tangent_implies_radius (c : Circle) (p : Point) :
  c.center = (3, -1) →
  p = (-2, 1) →
  numTangentLines c p = 1 →
  c.radius = Real.sqrt 29 := by sorry

end NUMINAMATH_CALUDE_unique_tangent_implies_radius_l3888_388820


namespace NUMINAMATH_CALUDE_cube_surface_area_l3888_388813

def edge_length : ℝ := 7

def surface_area_of_cube (edge : ℝ) : ℝ := 6 * edge^2

theorem cube_surface_area : 
  surface_area_of_cube edge_length = 294 := by sorry

end NUMINAMATH_CALUDE_cube_surface_area_l3888_388813


namespace NUMINAMATH_CALUDE_election_result_l3888_388805

theorem election_result (total_votes : ℕ) (invalid_percent : ℚ) (second_candidate_votes : ℕ) :
  total_votes = 7500 →
  invalid_percent = 20 / 100 →
  second_candidate_votes = 2700 →
  (↑((total_votes * (1 - invalid_percent)).floor - second_candidate_votes) / ↑((total_votes * (1 - invalid_percent)).floor) : ℚ) = 55 / 100 := by
  sorry

end NUMINAMATH_CALUDE_election_result_l3888_388805


namespace NUMINAMATH_CALUDE_sum_of_cubes_l3888_388844

theorem sum_of_cubes (a b s p : ℝ) (h1 : s = a + b) (h2 : p = a * b) : 
  a^3 + b^3 = s^3 - 3*s*p := by sorry

end NUMINAMATH_CALUDE_sum_of_cubes_l3888_388844


namespace NUMINAMATH_CALUDE_rationalize_and_simplify_l3888_388890

theorem rationalize_and_simplify :
  (Real.sqrt 8 + Real.sqrt 3) / (Real.sqrt 5 + Real.sqrt 3) =
  Real.sqrt 10 - Real.sqrt 6 + (Real.sqrt 15) / 2 - 3 / 2 := by
  sorry

end NUMINAMATH_CALUDE_rationalize_and_simplify_l3888_388890


namespace NUMINAMATH_CALUDE_complex_multiplication_l3888_388867

theorem complex_multiplication (z : ℂ) (h : z = 1 + Complex.I) : (1 + z) * z = 1 + 3 * Complex.I := by
  sorry

end NUMINAMATH_CALUDE_complex_multiplication_l3888_388867


namespace NUMINAMATH_CALUDE_quadratic_equation_solution_l3888_388899

theorem quadratic_equation_solution : ∃ x1 x2 : ℝ, 
  x1 = 95 ∧ 
  x2 = -105 ∧ 
  x1^2 + 10*x1 - 9975 = 0 ∧ 
  x2^2 + 10*x2 - 9975 = 0 := by
sorry

end NUMINAMATH_CALUDE_quadratic_equation_solution_l3888_388899


namespace NUMINAMATH_CALUDE_bob_cannot_win_and_prevent_alice_l3888_388855

def game_number : Set ℕ := {19, 20}
def start_number : Set ℕ := {9, 10}

theorem bob_cannot_win_and_prevent_alice (s : ℕ) (a : ℕ) :
  s ∈ start_number →
  a ∈ game_number →
  (∀ n : ℤ, s + 39 * n ≠ 2019) ∧
  (s = 9 → ∀ n : ℤ, s + 39 * n + a ≠ 2019) :=
by sorry

end NUMINAMATH_CALUDE_bob_cannot_win_and_prevent_alice_l3888_388855


namespace NUMINAMATH_CALUDE_ellipse_triangle_area_l3888_388886

/-- An ellipse with given properties -/
structure Ellipse :=
  (A B E F : ℝ × ℝ)
  (AB_length : Real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2) = 4)
  (AF_length : Real.sqrt ((A.1 - F.1)^2 + (A.2 - F.2)^2) = 2 + Real.sqrt 3)

/-- A point on the ellipse satisfying the given condition -/
def PointOnEllipse (Γ : Ellipse) (P : ℝ × ℝ) : Prop :=
  Real.sqrt ((P.1 - Γ.E.1)^2 + (P.2 - Γ.E.2)^2) *
  Real.sqrt ((P.1 - Γ.F.1)^2 + (P.2 - Γ.F.2)^2) = 2

/-- The theorem to be proved -/
theorem ellipse_triangle_area (Γ : Ellipse) (P : ℝ × ℝ) (h : PointOnEllipse Γ P) :
  (1/2) * Real.sqrt ((P.1 - Γ.E.1)^2 + (P.2 - Γ.E.2)^2) *
         Real.sqrt ((P.1 - Γ.F.1)^2 + (P.2 - Γ.F.2)^2) *
         Real.sin (Real.arccos (
           ((P.1 - Γ.E.1) * (P.1 - Γ.F.1) + (P.2 - Γ.E.2) * (P.2 - Γ.F.2)) /
           (Real.sqrt ((P.1 - Γ.E.1)^2 + (P.2 - Γ.E.2)^2) *
            Real.sqrt ((P.1 - Γ.F.1)^2 + (P.2 - Γ.F.2)^2))
         )) = 1 := by
  sorry

end NUMINAMATH_CALUDE_ellipse_triangle_area_l3888_388886


namespace NUMINAMATH_CALUDE_max_basketballs_part1_max_basketballs_part2_l3888_388832

/-- Represents the prices and quantities of basketballs and soccer balls -/
structure BallPurchase where
  basketball_price : ℕ
  soccer_ball_price : ℕ
  basketball_quantity : ℕ
  soccer_ball_quantity : ℕ

/-- Calculates the total cost of the purchase -/
def total_cost (purchase : BallPurchase) : ℕ :=
  purchase.basketball_price * purchase.basketball_quantity +
  purchase.soccer_ball_price * purchase.soccer_ball_quantity

/-- Calculates the total quantity of balls purchased -/
def total_quantity (purchase : BallPurchase) : ℕ :=
  purchase.basketball_quantity + purchase.soccer_ball_quantity

/-- Theorem for part 1 of the problem -/
theorem max_basketballs_part1 (purchase : BallPurchase) 
  (h1 : purchase.basketball_price = 100)
  (h2 : purchase.soccer_ball_price = 80)
  (h3 : total_cost purchase = 5600)
  (h4 : total_quantity purchase = 60) :
  purchase.basketball_quantity = 40 ∧ purchase.soccer_ball_quantity = 20 := by
  sorry

/-- Theorem for part 2 of the problem -/
theorem max_basketballs_part2 (purchase : BallPurchase) 
  (h1 : purchase.basketball_price = 100)
  (h2 : purchase.soccer_ball_price = 80)
  (h3 : total_cost purchase ≤ 6890)
  (h4 : total_quantity purchase = 80) :
  purchase.basketball_quantity ≤ 24 := by
  sorry

end NUMINAMATH_CALUDE_max_basketballs_part1_max_basketballs_part2_l3888_388832


namespace NUMINAMATH_CALUDE_distance_minimized_at_eight_sevenths_l3888_388804

/-- Given two points A and B in 3D space, prove that their distance is minimized when x = 8/7 -/
theorem distance_minimized_at_eight_sevenths (x : ℝ) :
  let A := (x, 5 - x, 2*x - 1)
  let B := (1, x + 2, 2 - x)
  let distance := Real.sqrt ((x - 1)^2 + (x + 2 - (5 - x))^2 + (2 - x - (2*x - 1))^2)
  (∀ y : ℝ, distance ≤ Real.sqrt ((y - 1)^2 + (y + 2 - (5 - y))^2 + (2 - y - (2*y - 1))^2)) ↔
  x = 8/7 := by
sorry


end NUMINAMATH_CALUDE_distance_minimized_at_eight_sevenths_l3888_388804


namespace NUMINAMATH_CALUDE_detergent_per_pound_l3888_388876

/-- Given that 18 ounces of detergent are used for 9 pounds of clothes,
    prove that 2 ounces of detergent are used per pound of clothes. -/
theorem detergent_per_pound (total_detergent : ℝ) (total_clothes : ℝ) 
  (h1 : total_detergent = 18) (h2 : total_clothes = 9) :
  total_detergent / total_clothes = 2 := by
  sorry

end NUMINAMATH_CALUDE_detergent_per_pound_l3888_388876


namespace NUMINAMATH_CALUDE_circle_and_tangents_l3888_388801

-- Define the circles and points
def circle_C (a : ℝ) : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | (p.1 - a)^2 + p.2^2 = (a + 1)^2 + 1}

def circle_D : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | (p.1 - 4)^2 + p.2^2 = 4}

-- Define the conditions
def passes_through (C : Set (ℝ × ℝ)) (p : ℝ × ℝ) : Prop :=
  p ∈ C

def tangent_line (P A : ℝ × ℝ) (C : Set (ℝ × ℝ)) : Prop :=
  ∃ (m : ℝ), ∀ (x y : ℝ),
    (y - A.2 = m * (x - A.1)) →
    ((x, y) ∈ C → (x, y) = A ∨ (x, y) = P)

-- State the theorem
theorem circle_and_tangents :
  ∀ (a : ℝ),
    passes_through (circle_C a) (0, 0) →
    passes_through (circle_C a) (-1, 1) →
    (∀ (P : ℝ × ℝ), P ∈ circle_D →
      ∃ (A B : ℝ × ℝ),
        A.1 = 0 ∧ B.1 = 0 ∧
        tangent_line P A (circle_C a) ∧
        tangent_line P B (circle_C a)) →
  (∀ (x y : ℝ), (x, y) ∈ circle_C a ↔ (x + 1)^2 + y^2 = 1) ∧
  (∃ (min max : ℝ),
    min = 5 * Real.sqrt 2 / 4 ∧
    max = Real.sqrt 2 ∧
    (∀ (P : ℝ × ℝ), P ∈ circle_D →
      ∃ (A B : ℝ × ℝ),
        A.1 = 0 ∧ B.1 = 0 ∧
        tangent_line P A (circle_C a) ∧
        tangent_line P B (circle_C a) ∧
        min ≤ |A.2 - B.2| ∧ |A.2 - B.2| ≤ max)) :=
by sorry

end NUMINAMATH_CALUDE_circle_and_tangents_l3888_388801


namespace NUMINAMATH_CALUDE_bee_multiple_l3888_388858

theorem bee_multiple (bees_day1 bees_day2 : ℕ) : 
  bees_day1 = 144 → bees_day2 = 432 → bees_day2 / bees_day1 = 3 := by
  sorry

end NUMINAMATH_CALUDE_bee_multiple_l3888_388858


namespace NUMINAMATH_CALUDE_neg_p_sufficient_not_necessary_for_neg_q_l3888_388836

-- Define the conditions
def p (x : ℝ) := x^2 - 1 > 0
def q (x : ℝ) := x < -2

-- State the theorem
theorem neg_p_sufficient_not_necessary_for_neg_q :
  (∀ x, ¬(p x) → ¬(q x)) ∧ 
  (∃ x, ¬(q x) ∧ p x) := by
  sorry

end NUMINAMATH_CALUDE_neg_p_sufficient_not_necessary_for_neg_q_l3888_388836


namespace NUMINAMATH_CALUDE_inequality_proof_l3888_388829

theorem inequality_proof (x y z : ℝ) 
  (sum_zero : x + y + z = 0)
  (abs_sum_le_one : |x| + |y| + |z| ≤ 1) :
  x + y/3 + z/5 ≤ 2/5 := by
sorry

end NUMINAMATH_CALUDE_inequality_proof_l3888_388829


namespace NUMINAMATH_CALUDE_fraction_addition_simplest_form_l3888_388878

theorem fraction_addition : (13 : ℚ) / 15 + (7 : ℚ) / 9 = (74 : ℚ) / 45 := by
  sorry

theorem simplest_form : Int.gcd 74 45 = 1 := by
  sorry

end NUMINAMATH_CALUDE_fraction_addition_simplest_form_l3888_388878


namespace NUMINAMATH_CALUDE_pounds_per_pillow_is_two_l3888_388826

-- Define the constants from the problem
def feathers_per_pound : ℕ := 300
def total_feathers : ℕ := 3600
def number_of_pillows : ℕ := 6

-- Define the function to calculate pounds of feathers needed per pillow
def pounds_per_pillow : ℚ :=
  (total_feathers / feathers_per_pound) / number_of_pillows

-- Theorem to prove
theorem pounds_per_pillow_is_two : pounds_per_pillow = 2 := by
  sorry


end NUMINAMATH_CALUDE_pounds_per_pillow_is_two_l3888_388826


namespace NUMINAMATH_CALUDE_tan_beta_value_l3888_388833

theorem tan_beta_value (α β : Real) 
  (h1 : Real.tan α = 1/3) 
  (h2 : Real.tan (α + β) = 1/2) : 
  Real.tan β = 1/7 := by
  sorry

end NUMINAMATH_CALUDE_tan_beta_value_l3888_388833


namespace NUMINAMATH_CALUDE_seeds_in_small_gardens_l3888_388879

theorem seeds_in_small_gardens (total_seeds : ℕ) (big_garden_seeds : ℕ) (num_small_gardens : ℕ) :
  total_seeds = 42 →
  big_garden_seeds = 36 →
  num_small_gardens = 3 →
  num_small_gardens > 0 →
  total_seeds ≥ big_garden_seeds →
  (total_seeds - big_garden_seeds) % num_small_gardens = 0 →
  (total_seeds - big_garden_seeds) / num_small_gardens = 2 := by
sorry

end NUMINAMATH_CALUDE_seeds_in_small_gardens_l3888_388879


namespace NUMINAMATH_CALUDE_rational_inequality_solution_l3888_388870

theorem rational_inequality_solution (x : ℝ) :
  (x^2 - 9) / (x^2 - 4) > 0 ↔ x < -3 ∨ x > 3 :=
sorry

end NUMINAMATH_CALUDE_rational_inequality_solution_l3888_388870


namespace NUMINAMATH_CALUDE_max_sum_is_21_l3888_388892

/-- Represents a nonzero digit (1-9) -/
def NonzeroDigit := { d : ℕ // 1 ≤ d ∧ d ≤ 9 }

/-- Calculates An for a given nonzero digit a and positive integer n -/
def An (a : NonzeroDigit) (n : ℕ+) : ℕ :=
  a.val * (10^n.val - 1) / 9

/-- Calculates Bn for a given nonzero digit b and positive integer n -/
def Bn (b : NonzeroDigit) (n : ℕ+) : ℕ :=
  b.val * (10^n.val - 1) / 9

/-- Calculates Cn for a given nonzero digit c and positive integer n -/
def Cn (c : NonzeroDigit) (n : ℕ+) : ℕ :=
  c.val * (10^(n.val + 1) - 1) / 9

/-- Checks if the equation Cn - Bn = An^2 holds for given a, b, c, and n -/
def EquationHolds (a b c : NonzeroDigit) (n : ℕ+) : Prop :=
  Cn c n - Bn b n = (An a n)^2

/-- Checks if there exist at least two distinct positive integers n for which the equation holds -/
def ExistTwoDistinctN (a b c : NonzeroDigit) : Prop :=
  ∃ n₁ n₂ : ℕ+, n₁ ≠ n₂ ∧ EquationHolds a b c n₁ ∧ EquationHolds a b c n₂

/-- The main theorem stating that the maximum value of a + b + c is 21 -/
theorem max_sum_is_21 :
  ∀ a b c : NonzeroDigit,
  ExistTwoDistinctN a b c →
  a.val + b.val + c.val ≤ 21 :=
sorry

end NUMINAMATH_CALUDE_max_sum_is_21_l3888_388892


namespace NUMINAMATH_CALUDE_geometric_sequence_product_l3888_388854

/-- A geometric sequence -/
def geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = r * a n

/-- Theorem: In a geometric sequence where a_4 = 3, a_2 * a_6 = 9 -/
theorem geometric_sequence_product (a : ℕ → ℝ) 
  (h_geom : geometric_sequence a) (h_a4 : a 4 = 3) : 
  a 2 * a 6 = 9 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_product_l3888_388854


namespace NUMINAMATH_CALUDE_three_heads_probability_l3888_388812

/-- The probability of getting heads on a single flip of a fair coin -/
def prob_heads : ℚ := 1/2

/-- The probability of getting three heads in a row when flipping a fair coin -/
def prob_three_heads : ℚ := prob_heads * prob_heads * prob_heads

theorem three_heads_probability : prob_three_heads = 1/8 := by
  sorry

end NUMINAMATH_CALUDE_three_heads_probability_l3888_388812


namespace NUMINAMATH_CALUDE_subtraction_of_fractions_l3888_388818

theorem subtraction_of_fractions : (8 : ℚ) / 19 - (5 : ℚ) / 57 = (1 : ℚ) / 3 := by
  sorry

end NUMINAMATH_CALUDE_subtraction_of_fractions_l3888_388818


namespace NUMINAMATH_CALUDE_area_between_circles_l3888_388865

-- Define the circles
def outer_circle_radius : ℝ := 12
def chord_length : ℝ := 20

-- Define the theorem
theorem area_between_circles :
  ∃ (inner_circle_radius : ℝ),
    inner_circle_radius > 0 ∧
    inner_circle_radius < outer_circle_radius ∧
    chord_length^2 = 4 * (outer_circle_radius^2 - inner_circle_radius^2) ∧
    π * (outer_circle_radius^2 - inner_circle_radius^2) = 100 * π :=
by
  sorry


end NUMINAMATH_CALUDE_area_between_circles_l3888_388865


namespace NUMINAMATH_CALUDE_square_area_from_perimeter_l3888_388817

/-- For a square with perimeter 48 meters, its area is 144 square meters. -/
theorem square_area_from_perimeter : 
  ∀ (s : Real), 
    (4 * s = 48) →  -- perimeter = 4 * side length = 48
    (s * s = 144)   -- area = side length * side length = 144
:= by sorry

end NUMINAMATH_CALUDE_square_area_from_perimeter_l3888_388817


namespace NUMINAMATH_CALUDE_altitude_intersection_property_l3888_388823

/-- Represents a point in 2D space -/
structure Point :=
  (x : ℝ)
  (y : ℝ)

/-- Represents a triangle -/
structure Triangle :=
  (A : Point)
  (B : Point)
  (C : Point)

/-- Checks if a triangle is acute -/
def isAcute (t : Triangle) : Prop := sorry

/-- Calculates the distance between two points -/
def distance (p1 p2 : Point) : ℝ := sorry

/-- Checks if a line is perpendicular to another line -/
def isPerpendicular (p1 p2 p3 p4 : Point) : Prop := sorry

/-- Finds the intersection point of two lines -/
def intersectionPoint (p1 p2 p3 p4 : Point) : Point := sorry

/-- Main theorem -/
theorem altitude_intersection_property (t : Triangle) (P Q H : Point) :
  isAcute t →
  isPerpendicular t.B t.C t.A P →
  isPerpendicular t.A t.C t.B Q →
  H = intersectionPoint t.A P t.B Q →
  distance H P = 4 →
  distance H Q = 3 →
  let B' := intersectionPoint t.A t.C t.B P
  let C' := intersectionPoint t.A t.B t.C P
  let A' := intersectionPoint t.B t.C t.A Q
  let C'' := intersectionPoint t.A t.B t.C Q
  (distance t.B P * distance P C') - (distance t.A Q * distance Q C'') = 7 := by
  sorry

end NUMINAMATH_CALUDE_altitude_intersection_property_l3888_388823


namespace NUMINAMATH_CALUDE_perpendicular_lines_parallel_perpendicular_to_parallel_planes_l3888_388810

-- Define the basic types
variable (Point : Type) (Line : Type) (Plane : Type)

-- Define the basic relations
variable (perpendicular : Line → Plane → Prop)
variable (parallel : Line → Line → Prop)
variable (plane_parallel : Plane → Plane → Prop)

-- Theorem 1: If two lines are perpendicular to the same plane, then they are parallel
theorem perpendicular_lines_parallel 
  (m n : Line) (α : Plane) 
  (h1 : perpendicular m α) (h2 : perpendicular n α) : 
  parallel m n :=
sorry

-- Theorem 2: If three planes are parallel and a line is perpendicular to one of them, 
-- then it is perpendicular to all of them
theorem perpendicular_to_parallel_planes 
  (m : Line) (α β γ : Plane)
  (h1 : plane_parallel α β) (h2 : plane_parallel β γ) 
  (h3 : perpendicular m α) :
  perpendicular m γ :=
sorry

end NUMINAMATH_CALUDE_perpendicular_lines_parallel_perpendicular_to_parallel_planes_l3888_388810


namespace NUMINAMATH_CALUDE_complex_number_quadrant_l3888_388869

theorem complex_number_quadrant : ∃ (z : ℂ), z * (1 - Complex.I) = Complex.I ∧ z.re < 0 ∧ z.im > 0 := by
  sorry

end NUMINAMATH_CALUDE_complex_number_quadrant_l3888_388869


namespace NUMINAMATH_CALUDE_total_employees_after_increase_l3888_388850

-- Define the initial conditions
def initial_total : ℕ := 1200
def initial_production : ℕ := 800
def initial_admin : ℕ := 400
def production_increase : ℚ := 35 / 100
def admin_increase : ℚ := 3 / 5

-- Define the theorem
theorem total_employees_after_increase : 
  initial_production * (1 + production_increase) + initial_admin * (1 + admin_increase) = 1720 := by
  sorry

end NUMINAMATH_CALUDE_total_employees_after_increase_l3888_388850


namespace NUMINAMATH_CALUDE_jims_age_l3888_388822

theorem jims_age (j t : ℕ) (h1 : j = 3 * t + 10) (h2 : j + t = 70) : j = 55 := by
  sorry

end NUMINAMATH_CALUDE_jims_age_l3888_388822


namespace NUMINAMATH_CALUDE_correct_operation_l3888_388866

theorem correct_operation : 
  (-2^2 ≠ 4) ∧ 
  ((-2)^3 ≠ -6) ∧ 
  ((-1/2)^3 = -1/8) ∧ 
  ((-7/3)^3 ≠ -8/27) :=
by
  sorry

end NUMINAMATH_CALUDE_correct_operation_l3888_388866


namespace NUMINAMATH_CALUDE_sweets_neither_red_nor_green_l3888_388887

/-- Given a bowl of sweets with red, green, and other colors, calculate the number of sweets that are neither red nor green. -/
theorem sweets_neither_red_nor_green 
  (total : ℕ) 
  (red : ℕ) 
  (green : ℕ) 
  (h_total : total = 285) 
  (h_red : red = 49) 
  (h_green : green = 59) :
  total - (red + green) = 177 := by
  sorry

end NUMINAMATH_CALUDE_sweets_neither_red_nor_green_l3888_388887


namespace NUMINAMATH_CALUDE_triangle_inradius_circumradius_inequality_l3888_388873

/-- For any triangle with inradius r, circumradius R, and an angle α, 
    the inequality r / R ≤ 2 sin(α / 2)(1 - sin(α / 2)) holds. -/
theorem triangle_inradius_circumradius_inequality 
  (r R α : ℝ) 
  (hr : r > 0) 
  (hR : R > 0) 
  (hα : 0 < α ∧ α < π) : 
  r / R ≤ 2 * Real.sin (α / 2) * (1 - Real.sin (α / 2)) :=
sorry

end NUMINAMATH_CALUDE_triangle_inradius_circumradius_inequality_l3888_388873


namespace NUMINAMATH_CALUDE_womans_swimming_speed_l3888_388875

/-- Given a woman who swims downstream and upstream with specific distances and times,
    this theorem proves her speed in still water. -/
theorem womans_swimming_speed
  (downstream_distance : ℝ)
  (upstream_distance : ℝ)
  (downstream_time : ℝ)
  (upstream_time : ℝ)
  (h_downstream : downstream_distance = 54)
  (h_upstream : upstream_distance = 6)
  (h_time : downstream_time = 6 ∧ upstream_time = 6)
  : ∃ (speed_still_water : ℝ) (stream_speed : ℝ),
    speed_still_water = 5 ∧
    downstream_distance / downstream_time = speed_still_water + stream_speed ∧
    upstream_distance / upstream_time = speed_still_water - stream_speed :=
by sorry

end NUMINAMATH_CALUDE_womans_swimming_speed_l3888_388875


namespace NUMINAMATH_CALUDE_tan_360_minus_45_l3888_388847

theorem tan_360_minus_45 : Real.tan (360 * π / 180 - 45 * π / 180) = -1 := by sorry

end NUMINAMATH_CALUDE_tan_360_minus_45_l3888_388847


namespace NUMINAMATH_CALUDE_bookstore_optimal_price_l3888_388852

/-- Profit function for the bookstore -/
def P (p : ℝ) : ℝ := 150 * p - 4 * p^2 - 200

/-- The maximum allowed price -/
def max_price : ℝ := 30

/-- The optimal price that maximizes profit -/
def optimal_price : ℝ := 18.75

theorem bookstore_optimal_price :
  optimal_price ≤ max_price ∧
  ∀ p : ℝ, p ≤ max_price → P p ≤ P optimal_price :=
sorry

end NUMINAMATH_CALUDE_bookstore_optimal_price_l3888_388852


namespace NUMINAMATH_CALUDE_is_hyperbola_center_l3888_388800

/-- The equation of the hyperbola -/
def hyperbola_equation (x y : ℝ) : Prop :=
  9 * x^2 - 54 * x - 36 * y^2 + 288 * y - 576 = 0

/-- The center of the hyperbola -/
def hyperbola_center : ℝ × ℝ := (3, 4)

/-- Theorem stating that the given point is the center of the hyperbola -/
theorem is_hyperbola_center : 
  ∀ (x y : ℝ), hyperbola_equation x y ↔ 
    ((x - hyperbola_center.1)^2 / 9 - (y - hyperbola_center.2)^2 / 4 = 1) :=
by sorry

end NUMINAMATH_CALUDE_is_hyperbola_center_l3888_388800


namespace NUMINAMATH_CALUDE_baseball_cards_per_page_l3888_388896

theorem baseball_cards_per_page : 
  ∀ (cards_per_page : ℕ+) (full_pages : ℕ+),
  cards_per_page.val * full_pages.val + 1 = 7 →
  cards_per_page = 2 := by
sorry

end NUMINAMATH_CALUDE_baseball_cards_per_page_l3888_388896


namespace NUMINAMATH_CALUDE_least_subtraction_for_divisibility_l3888_388860

theorem least_subtraction_for_divisibility : 
  ∃ (n : ℕ), n = 72 ∧ 
  (∀ (m : ℕ), m < n → ¬(127 ∣ (100203 - m))) ∧ 
  (127 ∣ (100203 - n)) := by
sorry

end NUMINAMATH_CALUDE_least_subtraction_for_divisibility_l3888_388860


namespace NUMINAMATH_CALUDE_fair_products_l3888_388859

/-- The number of recycled materials made by the group -/
def group_materials : ℕ := 65

/-- The number of recycled materials made by the teachers -/
def teacher_materials : ℕ := 28

/-- The total number of recycled products to sell at the fair -/
def total_products : ℕ := group_materials + teacher_materials

theorem fair_products : total_products = 93 := by
  sorry

end NUMINAMATH_CALUDE_fair_products_l3888_388859


namespace NUMINAMATH_CALUDE_hockey_league_games_l3888_388837

/-- The total number of games played in a hockey league season -/
def total_games (n : ℕ) (g : ℕ) : ℕ :=
  n * (n - 1) * g / 2

/-- Theorem: In a league with 12 teams, where each team plays 4 games with each other team,
    the total number of games played is 264 -/
theorem hockey_league_games :
  total_games 12 4 = 264 := by
  sorry

end NUMINAMATH_CALUDE_hockey_league_games_l3888_388837


namespace NUMINAMATH_CALUDE_unattainable_y_l3888_388827

theorem unattainable_y (x : ℝ) :
  (2 * x^2 + 3 * x + 4 ≠ 0) →
  ∃ y : ℝ, y = (1 - x) / (2 * x^2 + 3 * x + 4) ∧ y ≠ 0 :=
by sorry

end NUMINAMATH_CALUDE_unattainable_y_l3888_388827


namespace NUMINAMATH_CALUDE_binary_rep_of_23_l3888_388830

/-- Converts a natural number to its binary representation as a list of bits -/
def toBinary (n : ℕ) : List Bool :=
  if n = 0 then [false] else
    let rec go (m : ℕ) : List Bool :=
      if m = 0 then [] else (m % 2 = 1) :: go (m / 2)
    go n

/-- Theorem: The binary representation of 23 is [true, true, true, false, true] -/
theorem binary_rep_of_23 : toBinary 23 = [true, true, true, false, true] := by
  sorry

end NUMINAMATH_CALUDE_binary_rep_of_23_l3888_388830


namespace NUMINAMATH_CALUDE_sequence_inequality_l3888_388814

theorem sequence_inequality (a b : ℝ) (h1 : a > 0) (h2 : b > 0) :
  let z := Complex.mk a b
  let seq := fun (n : ℕ+) => z ^ n.val
  let a_n := fun (n : ℕ+) => (seq n).re
  let b_n := fun (n : ℕ+) => (seq n).im
  ∀ n : ℕ+, (Complex.abs (a_n (n + 1)) + Complex.abs (b_n (n + 1))) / (Complex.abs (a_n n) + Complex.abs (b_n n)) ≥ (a^2 + b^2) / (a + b) :=
by sorry


end NUMINAMATH_CALUDE_sequence_inequality_l3888_388814


namespace NUMINAMATH_CALUDE_perpendicular_tangents_ratio_l3888_388851

/-- Given a line ax - by - 2 = 0 and a curve y = x^3 intersecting at point P(1, 1),
    if the tangent lines at P are perpendicular, then a/b = -1/3 -/
theorem perpendicular_tangents_ratio (a b : ℝ) : 
  (∀ x y, a * x - b * y - 2 = 0 → y = x^3) →  -- Line and curve equations
  (a * 1 - b * 1 - 2 = 0) →                   -- Point P(1, 1) satisfies line equation
  (1 = 1^3) →                                 -- Point P(1, 1) satisfies curve equation
  (∃ k₁ k₂ : ℝ, k₁ * k₂ = -1 ∧                -- Perpendicular tangent lines condition
              k₁ = a / b ∧                    -- Slope of line
              k₂ = 3 * 1^2) →                 -- Slope of curve at P(1, 1)
  a / b = -1 / 3 := by
sorry

end NUMINAMATH_CALUDE_perpendicular_tangents_ratio_l3888_388851


namespace NUMINAMATH_CALUDE_percentage_difference_l3888_388894

theorem percentage_difference (x y : ℝ) (h : x = y * (1 - 0.45)) :
  y = x * (1 + 0.45) := by
  sorry

end NUMINAMATH_CALUDE_percentage_difference_l3888_388894


namespace NUMINAMATH_CALUDE_smallest_k_multiple_of_200_l3888_388872

theorem smallest_k_multiple_of_200 : ∃ (k : ℕ), k > 0 ∧ 
  (∀ (n : ℕ), n > 0 → n < k → ¬(200 ∣ (n * (n + 1) * (2 * n + 1)) / 6)) ∧ 
  (200 ∣ (k * (k + 1) * (2 * k + 1)) / 6) ∧
  k = 31 := by
sorry

end NUMINAMATH_CALUDE_smallest_k_multiple_of_200_l3888_388872


namespace NUMINAMATH_CALUDE_twenty_four_bananas_cost_l3888_388846

/-- The cost of fruits at Lisa's Fruit Stand -/
structure FruitCost where
  banana_apple_ratio : ℚ  -- 4 bananas = 3 apples
  apple_orange_ratio : ℚ  -- 8 apples = 5 oranges

/-- Calculate the number of oranges equivalent in cost to a given number of bananas -/
def bananas_to_oranges (cost : FruitCost) (num_bananas : ℕ) : ℚ :=
  let apples := (num_bananas : ℚ) * cost.banana_apple_ratio
  apples * cost.apple_orange_ratio

/-- Theorem: 24 bananas cost approximately as much as 11 oranges -/
theorem twenty_four_bananas_cost (cost : FruitCost) 
  (h1 : cost.banana_apple_ratio = 3 / 4)
  (h2 : cost.apple_orange_ratio = 5 / 8) :
  ⌊bananas_to_oranges cost 24⌋ = 11 := by
  sorry

#eval ⌊(24 : ℚ) * (3 / 4) * (5 / 8)⌋  -- Expected output: 11

end NUMINAMATH_CALUDE_twenty_four_bananas_cost_l3888_388846


namespace NUMINAMATH_CALUDE_fraction_non_negative_iff_positive_denominator_l3888_388838

theorem fraction_non_negative_iff_positive_denominator :
  ∀ x : ℝ, (2 / x ≥ 0) ↔ (x > 0) := by sorry

end NUMINAMATH_CALUDE_fraction_non_negative_iff_positive_denominator_l3888_388838


namespace NUMINAMATH_CALUDE_trapezoid_determines_unique_plane_l3888_388839

/-- A point in 3D space -/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- A trapezoid in 3D space -/
structure Trapezoid where
  p1 : Point3D
  p2 : Point3D
  p3 : Point3D
  p4 : Point3D
  is_trapezoid : ∃ (a b : ℝ), a ≠ b ∧
    (p2.x - p1.x) * (p4.y - p3.y) = (p2.y - p1.y) * (p4.x - p3.x) ∧
    (p3.x - p2.x) * (p1.y - p4.y) = (p3.y - p2.y) * (p1.x - p4.x)

/-- A plane in 3D space -/
structure Plane where
  a : ℝ
  b : ℝ
  c : ℝ
  d : ℝ

/-- Definition of a point lying on a plane -/
def Point3D.on_plane (p : Point3D) (plane : Plane) : Prop :=
  plane.a * p.x + plane.b * p.y + plane.c * p.z + plane.d = 0

/-- Theorem: A trapezoid determines a unique plane -/
theorem trapezoid_determines_unique_plane (t : Trapezoid) :
  ∃! (plane : Plane), t.p1.on_plane plane ∧ t.p2.on_plane plane ∧
                      t.p3.on_plane plane ∧ t.p4.on_plane plane :=
sorry

end NUMINAMATH_CALUDE_trapezoid_determines_unique_plane_l3888_388839


namespace NUMINAMATH_CALUDE_correct_observation_value_l3888_388882

theorem correct_observation_value (n : ℕ) (initial_mean corrected_mean wrong_value : ℝ) 
  (h1 : n = 50)
  (h2 : initial_mean = 36)
  (h3 : corrected_mean = 36.02)
  (h4 : wrong_value = 47) :
  let total_sum := n * initial_mean
  let remaining_sum := total_sum - wrong_value
  let corrected_total := n * corrected_mean
  corrected_total - remaining_sum = 48 := by
  sorry

end NUMINAMATH_CALUDE_correct_observation_value_l3888_388882


namespace NUMINAMATH_CALUDE_sphere_radius_equal_volume_cone_l3888_388857

/-- The radius of a sphere with the same volume as a cone -/
theorem sphere_radius_equal_volume_cone (r h : ℝ) (hr : r = 2) (hh : h = 8) :
  ∃ (r_sphere : ℝ), (1/3 * π * r^2 * h) = (4/3 * π * r_sphere^3) ∧ r_sphere = 2 * (2 : ℝ)^(1/3) :=
sorry

end NUMINAMATH_CALUDE_sphere_radius_equal_volume_cone_l3888_388857


namespace NUMINAMATH_CALUDE_line_inclination_angle_l3888_388825

/-- The inclination angle of the line x*sin(π/7) + y*cos(π/7) = 0 is 6π/7 -/
theorem line_inclination_angle : 
  let line_eq := fun (x y : ℝ) => x * Real.sin (π / 7) + y * Real.cos (π / 7) = 0
  ∃ (α : ℝ), α = 6 * π / 7 ∧ 
    (∀ (x y : ℝ), line_eq x y → 
      Real.tan α = - (Real.sin (π / 7) / Real.cos (π / 7))) :=
by sorry

end NUMINAMATH_CALUDE_line_inclination_angle_l3888_388825


namespace NUMINAMATH_CALUDE_evaluate_nested_fraction_l3888_388840

theorem evaluate_nested_fraction : 1 - (1 / (1 - (1 / (1 + 2)))) = -1 / 2 := by sorry

end NUMINAMATH_CALUDE_evaluate_nested_fraction_l3888_388840


namespace NUMINAMATH_CALUDE_polynomial_value_at_zero_l3888_388845

/-- A polynomial P(x) = x^2 + bx + c satisfying specific conditions -/
def P (b c : ℝ) (x : ℝ) : ℝ := x^2 + b*x + c

theorem polynomial_value_at_zero 
  (b c : ℝ) 
  (h1 : P b c (P b c 1) = 0)
  (h2 : P b c (P b c 2) = 0)
  (h3 : P b c 1 ≠ P b c 2) :
  P b c 0 = -3/2 :=
sorry

end NUMINAMATH_CALUDE_polynomial_value_at_zero_l3888_388845


namespace NUMINAMATH_CALUDE_square_side_length_l3888_388864

theorem square_side_length (d : ℝ) (h : d = 2 * Real.sqrt 2) :
  ∃ s : ℝ, s * Real.sqrt 2 = d ∧ s = 2 := by
  sorry

end NUMINAMATH_CALUDE_square_side_length_l3888_388864


namespace NUMINAMATH_CALUDE_radical_simplification_l3888_388856

theorem radical_simplification (a : ℝ) (ha : a > 0) :
  Real.sqrt (50 * a^3) * Real.sqrt (18 * a^2) * Real.sqrt (98 * a^5) = 42 * a^5 * Real.sqrt 10 := by
  sorry

end NUMINAMATH_CALUDE_radical_simplification_l3888_388856


namespace NUMINAMATH_CALUDE_arithmetic_sequence_remainder_l3888_388861

def arithmetic_sequence (a₁ : ℤ) (d : ℤ) (n : ℕ) : ℤ := a₁ + (n - 1) * d

def sequence_sum (a₁ : ℤ) (aₙ : ℤ) (n : ℕ) : ℤ := n * (a₁ + aₙ) / 2

theorem arithmetic_sequence_remainder (a₁ : ℤ) (aₙ : ℤ) (d : ℤ) (n : ℕ) :
  a₁ = 3 ∧ aₙ = 347 ∧ d = 8 →
  (sequence_sum a₁ aₙ n) % 8 = 4 := by
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_remainder_l3888_388861


namespace NUMINAMATH_CALUDE_barbara_initial_candies_l3888_388881

/-- The number of candies Barbara used -/
def candies_used : ℝ := 9.0

/-- The number of candies Barbara has left -/
def candies_left : ℕ := 9

/-- The initial number of candies Barbara had -/
def initial_candies : ℝ := candies_used + candies_left

/-- Theorem stating that Barbara initially had 18 candies -/
theorem barbara_initial_candies : initial_candies = 18 := by
  sorry

end NUMINAMATH_CALUDE_barbara_initial_candies_l3888_388881


namespace NUMINAMATH_CALUDE_exists_rational_triangle_l3888_388849

/-- A triangle with integer sides, height, and median, all less than 100 -/
structure RationalTriangle where
  a : ℕ
  b : ℕ
  c : ℕ
  height : ℕ
  median : ℕ
  a_lt_100 : a < 100
  b_lt_100 : b < 100
  c_lt_100 : c < 100
  height_lt_100 : height < 100
  median_lt_100 : median < 100
  triangle_inequality : a + b > c ∧ b + c > a ∧ c + a > b
  not_right_triangle : a^2 + b^2 ≠ c^2 ∧ b^2 + c^2 ≠ a^2 ∧ c^2 + a^2 ≠ b^2

/-- There exists a triangle with integer sides, height, and median, all less than 100, that is not a right triangle -/
theorem exists_rational_triangle : ∃ t : RationalTriangle, True := by
  sorry

end NUMINAMATH_CALUDE_exists_rational_triangle_l3888_388849


namespace NUMINAMATH_CALUDE_sphere_volume_equals_cone_cylinder_volume_l3888_388843

/-- Given a cone with height 6 and radius 1.5, and a cylinder with the same height and volume as the cone,
    prove that a sphere with radius 1.5 has the same volume as both the cone and cylinder. -/
theorem sphere_volume_equals_cone_cylinder_volume :
  let cone_height : ℝ := 6
  let cone_radius : ℝ := 1.5
  let cylinder_height : ℝ := cone_height
  let cone_volume : ℝ := (1 / 3) * Real.pi * cone_radius^2 * cone_height
  let cylinder_volume : ℝ := cone_volume
  let sphere_radius : ℝ := 1.5
  let sphere_volume : ℝ := (4 / 3) * Real.pi * sphere_radius^3
  sphere_volume = cone_volume := by
  sorry


end NUMINAMATH_CALUDE_sphere_volume_equals_cone_cylinder_volume_l3888_388843


namespace NUMINAMATH_CALUDE_f_five_equals_142_l3888_388874

-- Define the function f
def f (x y : ℝ) : ℝ := 2 * x^2 + y

-- State the theorem
theorem f_five_equals_142 :
  ∃ y : ℝ, (f 2 y = 100) ∧ (f 5 y = 142) := by
  sorry

end NUMINAMATH_CALUDE_f_five_equals_142_l3888_388874


namespace NUMINAMATH_CALUDE_average_score_is_correct_l3888_388809

def total_students : ℕ := 120

-- Define the score distribution
def score_distribution : List (ℕ × ℕ) := [
  (95, 12),
  (85, 24),
  (75, 30),
  (65, 20),
  (55, 18),
  (45, 10),
  (35, 6)
]

-- Calculate the average score
def average_score : ℚ :=
  let total_score : ℕ := (score_distribution.map (λ (score, count) => score * count)).sum
  (total_score : ℚ) / total_students

-- Theorem to prove
theorem average_score_is_correct :
  average_score = 8380 / 120 := by sorry

end NUMINAMATH_CALUDE_average_score_is_correct_l3888_388809


namespace NUMINAMATH_CALUDE_distance_from_circle_center_to_point_l3888_388880

-- Define the circle equation
def circle_equation (x y : ℝ) : Prop :=
  x^2 + y^2 = 4*x + 6*y + 3

-- Define the center of the circle
def circle_center : ℝ × ℝ :=
  let center_x := 2
  let center_y := 3
  (center_x, center_y)

-- Define the given point
def given_point : ℝ × ℝ := (10, 5)

-- State the theorem
theorem distance_from_circle_center_to_point :
  let (cx, cy) := circle_center
  let (px, py) := given_point
  Real.sqrt ((px - cx)^2 + (py - cy)^2) = 2 * Real.sqrt 17 :=
by sorry

end NUMINAMATH_CALUDE_distance_from_circle_center_to_point_l3888_388880


namespace NUMINAMATH_CALUDE_min_value_condition_l3888_388877

theorem min_value_condition (a : ℝ) : 
  (∀ x > a, x + 4 / (x - a) > 9) → a > 5 := by
  sorry

#check min_value_condition

end NUMINAMATH_CALUDE_min_value_condition_l3888_388877


namespace NUMINAMATH_CALUDE_notebook_distribution_l3888_388815

theorem notebook_distribution (total notebooks k v y s se : ℕ) : 
  notebooks = 100 ∧
  k + v = 52 ∧
  v + y = 43 ∧
  y + s = 34 ∧
  s + se = 30 ∧
  k + v + y + s + se = notebooks →
  k = 27 ∧ v = 25 ∧ y = 18 ∧ s = 16 ∧ se = 14 := by
  sorry

end NUMINAMATH_CALUDE_notebook_distribution_l3888_388815


namespace NUMINAMATH_CALUDE_product_remainder_remainder_1287_1499_300_l3888_388816

theorem product_remainder (a b m : ℕ) (h : m > 0) : (a * b) % m = ((a % m) * (b % m)) % m := by sorry

theorem remainder_1287_1499_300 : (1287 * 1499) % 300 = 213 := by sorry

end NUMINAMATH_CALUDE_product_remainder_remainder_1287_1499_300_l3888_388816


namespace NUMINAMATH_CALUDE_product_of_divisors_36_l3888_388824

theorem product_of_divisors_36 (n : Nat) (h : n = 36) :
  (Finset.prod (Finset.filter (· ∣ n) (Finset.range (n + 1))) id) = 10077696 := by
  sorry

end NUMINAMATH_CALUDE_product_of_divisors_36_l3888_388824


namespace NUMINAMATH_CALUDE_binary_calculation_l3888_388888

-- Define binary numbers as natural numbers
def bin110110 : ℕ := 54  -- 110110 in base 2 is 54 in base 10
def bin101110 : ℕ := 46  -- 101110 in base 2 is 46 in base 10
def bin100 : ℕ := 4      -- 100 in base 2 is 4 in base 10
def bin11100011110 : ℕ := 1886  -- 11100011110 in base 2 is 1886 in base 10

-- State the theorem
theorem binary_calculation :
  (bin110110 / bin100) * bin101110 = bin11100011110 := by
  sorry

end NUMINAMATH_CALUDE_binary_calculation_l3888_388888


namespace NUMINAMATH_CALUDE_isabel_initial_candy_l3888_388803

/- Given conditions -/
def initial_candy : ℕ → Prop := λ x => True  -- Initial amount of candy (unknown)
def friend_gave : ℕ := 25                    -- Amount of candy given by friend
def total_candy : ℕ := 93                    -- Total amount of candy after receiving from friend

/- Theorem to prove -/
theorem isabel_initial_candy :
  ∃ x : ℕ, initial_candy x ∧ x + friend_gave = total_candy ∧ x = 68 :=
by sorry

end NUMINAMATH_CALUDE_isabel_initial_candy_l3888_388803


namespace NUMINAMATH_CALUDE_high_school_students_l3888_388895

/-- The number of students in a high school, given information about music and art classes -/
theorem high_school_students (music_students : ℕ) (art_students : ℕ) (both_students : ℕ) (neither_students : ℕ)
  (h1 : music_students = 20)
  (h2 : art_students = 20)
  (h3 : both_students = 10)
  (h4 : neither_students = 470) :
  music_students + art_students - both_students + neither_students = 500 :=
by sorry

end NUMINAMATH_CALUDE_high_school_students_l3888_388895


namespace NUMINAMATH_CALUDE_unique_pair_for_squared_difference_l3888_388841

theorem unique_pair_for_squared_difference : 
  ∃! (a b : ℕ), a^2 - b^2 = 25 ∧ a = 13 ∧ b = 12 :=
by sorry

end NUMINAMATH_CALUDE_unique_pair_for_squared_difference_l3888_388841


namespace NUMINAMATH_CALUDE_complex_equation_sum_l3888_388885

theorem complex_equation_sum (a b : ℝ) : 
  (a - Complex.I = 2 + b * Complex.I) → a + b = 1 := by
  sorry

end NUMINAMATH_CALUDE_complex_equation_sum_l3888_388885


namespace NUMINAMATH_CALUDE_license_plate_combinations_l3888_388891

/-- The number of possible letter combinations in the license plate -/
def letter_combinations : ℕ := Nat.choose 26 2 * 3

/-- The number of possible digit combinations in the license plate -/
def digit_combinations : ℕ := 10 * 9 * 3

/-- The total number of possible license plate combinations -/
def total_combinations : ℕ := letter_combinations * digit_combinations

theorem license_plate_combinations :
  total_combinations = 877500 := by
  sorry

end NUMINAMATH_CALUDE_license_plate_combinations_l3888_388891


namespace NUMINAMATH_CALUDE_sum_of_cyclic_relations_l3888_388821

theorem sum_of_cyclic_relations (p q r : ℝ) : 
  p ≠ q ∧ q ≠ r ∧ r ≠ p →
  q = p * (4 - p) →
  r = q * (4 - q) →
  p = r * (4 - r) →
  p + q + r = 6 ∨ p + q + r = 7 := by
sorry

end NUMINAMATH_CALUDE_sum_of_cyclic_relations_l3888_388821


namespace NUMINAMATH_CALUDE_exists_valid_coloring_l3888_388884

-- Define the color type
inductive Color
  | BLUE
  | GREEN
  | RED
  | YELLOW

-- Define the coloring function type
def ColoringFunction := ℤ → Color

-- Define the property that the coloring function must satisfy
def ValidColoring (f : ColoringFunction) : Prop :=
  ∀ a b c d : ℤ, f a = f b ∧ f b = f c ∧ f c = f d ∧ (a ≠ 0 ∨ b ≠ 0 ∨ c ≠ 0 ∨ d ≠ 0) →
    3 * a - 2 * b ≠ 2 * c - 3 * d

-- State the theorem
theorem exists_valid_coloring : ∃ f : ColoringFunction, ValidColoring f := by
  sorry

end NUMINAMATH_CALUDE_exists_valid_coloring_l3888_388884


namespace NUMINAMATH_CALUDE_angle_B_measure_max_perimeter_max_perimeter_achieved_l3888_388811

-- Define the triangle
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

-- Define the given condition
def condition (t : Triangle) : Prop :=
  2 * t.a * Real.cos t.B + t.b * Real.cos t.C + t.c * Real.cos t.B = 0

-- Part I: Prove that angle B is 2π/3
theorem angle_B_measure (t : Triangle) (h : condition t) : t.B = 2 * Real.pi / 3 := by
  sorry

-- Part II: Prove the maximum perimeter
theorem max_perimeter (t : Triangle) (h : condition t) (hb : t.b = Real.sqrt 3) :
  t.a + t.b + t.c ≤ Real.sqrt 3 + 2 := by
  sorry

-- Prove that the maximum perimeter is achieved
theorem max_perimeter_achieved : ∃ (t : Triangle), condition t ∧ t.b = Real.sqrt 3 ∧ t.a + t.b + t.c = Real.sqrt 3 + 2 := by
  sorry

end NUMINAMATH_CALUDE_angle_B_measure_max_perimeter_max_perimeter_achieved_l3888_388811


namespace NUMINAMATH_CALUDE_baking_scoops_l3888_388862

/-- Calculates the number of scoops needed given the amount of ingredient in cups and the size of the scoop -/
def scoops_needed (cups : ℚ) (scoop_size : ℚ) : ℕ :=
  (cups / scoop_size).ceil.toNat

/-- The total number of scoops needed for flour and sugar -/
def total_scoops : ℕ :=
  scoops_needed 3 (1/3) + scoops_needed 2 (1/3)

theorem baking_scoops : total_scoops = 15 := by
  sorry

end NUMINAMATH_CALUDE_baking_scoops_l3888_388862


namespace NUMINAMATH_CALUDE_graph_passes_through_second_and_fourth_quadrants_l3888_388868

-- Define the function
def f (x : ℝ) : ℝ := -3 * x

-- State the theorem
theorem graph_passes_through_second_and_fourth_quadrants :
  (∃ x y, x < 0 ∧ y > 0 ∧ f x = y) ∧ 
  (∃ x y, x > 0 ∧ y < 0 ∧ f x = y) :=
by sorry

end NUMINAMATH_CALUDE_graph_passes_through_second_and_fourth_quadrants_l3888_388868


namespace NUMINAMATH_CALUDE_metal_collection_contest_solution_l3888_388889

/-- Represents the metal collection contest between boys and girls -/
structure MetalContest where
  totalMetal : ℕ
  boyAverage : ℕ
  girlAverage : ℕ
  numBoys : ℕ
  numGirls : ℕ

/-- Checks if the given numbers satisfy the contest conditions -/
def isValidContest (contest : MetalContest) : Prop :=
  contest.boyAverage * contest.numBoys + contest.girlAverage * contest.numGirls = contest.totalMetal

/-- Checks if boys won the contest -/
def boysWon (contest : MetalContest) : Prop :=
  contest.boyAverage * contest.numBoys > contest.girlAverage * contest.numGirls

/-- Theorem stating the solution to the metal collection contest -/
theorem metal_collection_contest_solution :
  ∃ (contest : MetalContest),
    contest.totalMetal = 2831 ∧
    contest.boyAverage = 95 ∧
    contest.girlAverage = 74 ∧
    contest.numBoys = 15 ∧
    contest.numGirls = 19 ∧
    isValidContest contest ∧
    boysWon contest :=
  sorry

end NUMINAMATH_CALUDE_metal_collection_contest_solution_l3888_388889


namespace NUMINAMATH_CALUDE_crayons_in_box_l3888_388819

def blue_crayons : ℕ := 3

def red_crayons : ℕ := 4 * blue_crayons

def total_crayons : ℕ := red_crayons + blue_crayons

theorem crayons_in_box : total_crayons = 15 := by
  sorry

end NUMINAMATH_CALUDE_crayons_in_box_l3888_388819


namespace NUMINAMATH_CALUDE_cosine_difference_equals_negative_seven_thousandths_l3888_388897

theorem cosine_difference_equals_negative_seven_thousandths :
  let α := Real.arcsin (3/5)
  let β := Real.arcsin (4/5)
  (Real.cos (3*Real.pi/2 - α/2))^6 - (Real.cos (5*Real.pi/2 + β/2))^6 = -7/1000 := by
sorry

end NUMINAMATH_CALUDE_cosine_difference_equals_negative_seven_thousandths_l3888_388897


namespace NUMINAMATH_CALUDE_solution_characterization_l3888_388807

def valid_solution (a b c x y z : ℕ) : Prop :=
  a + b + c = x * y * z ∧
  x + y + z = a * b * c ∧
  a ≥ b ∧ b ≥ c ∧ c ≥ 1 ∧
  x ≥ y ∧ y ≥ z ∧ z ≥ 1

def solution_set : Set (ℕ × ℕ × ℕ × ℕ × ℕ × ℕ) :=
  {(2, 2, 2, 6, 1, 1), (5, 2, 1, 8, 1, 1), (3, 3, 1, 7, 1, 1), (3, 2, 1, 6, 2, 1)}

theorem solution_characterization :
  ∀ a b c x y z : ℕ, valid_solution a b c x y z ↔ (a, b, c, x, y, z) ∈ solution_set :=
sorry

end NUMINAMATH_CALUDE_solution_characterization_l3888_388807


namespace NUMINAMATH_CALUDE_initial_amount_proof_l3888_388831

theorem initial_amount_proof (P : ℝ) : 
  (P * (1 + 1/8) * (1 + 1/8) = 2025) → P = 1600 := by
  sorry

end NUMINAMATH_CALUDE_initial_amount_proof_l3888_388831


namespace NUMINAMATH_CALUDE_book_cost_proof_l3888_388828

theorem book_cost_proof (initial_amount : ℕ) (num_books : ℕ) (remaining_amount : ℕ) 
  (h1 : initial_amount = 79)
  (h2 : num_books = 9)
  (h3 : remaining_amount = 16) :
  (initial_amount - remaining_amount) / num_books = 7 :=
by
  sorry

end NUMINAMATH_CALUDE_book_cost_proof_l3888_388828


namespace NUMINAMATH_CALUDE_roots_of_equation_l3888_388871

theorem roots_of_equation :
  let f : ℝ → ℝ := λ x => x * (x + 2) + x + 2
  (f (-2) = 0) ∧ (f (-1) = 0) ∧
  (∀ x : ℝ, f x = 0 → x = -2 ∨ x = -1) :=
by sorry

end NUMINAMATH_CALUDE_roots_of_equation_l3888_388871


namespace NUMINAMATH_CALUDE_shop_owner_gain_percentage_l3888_388834

/-- Calculates the shop owner's total gain percentage --/
theorem shop_owner_gain_percentage
  (cost_A cost_B cost_C : ℝ)
  (markup_A markup_B markup_C : ℝ)
  (quantity_A quantity_B quantity_C : ℝ)
  (discount tax : ℝ)
  (h1 : cost_A = 4)
  (h2 : cost_B = 6)
  (h3 : cost_C = 8)
  (h4 : markup_A = 0.25)
  (h5 : markup_B = 0.30)
  (h6 : markup_C = 0.20)
  (h7 : quantity_A = 25)
  (h8 : quantity_B = 15)
  (h9 : quantity_C = 10)
  (h10 : discount = 0.05)
  (h11 : tax = 0.05) :
  ∃ (gain_percentage : ℝ), abs (gain_percentage - 0.2487) < 0.0001 := by
  sorry


end NUMINAMATH_CALUDE_shop_owner_gain_percentage_l3888_388834


namespace NUMINAMATH_CALUDE_product_digit_sum_l3888_388848

def number1 : ℕ := 707070707070707070707070707070707070707070707070707070707070707070707070707070707070707070707070707
def number2 : ℕ := 909090909090909090909090909090909090909090909090909090909090909090909090909090909090909090909090909

theorem product_digit_sum :
  let product := number1 * number2
  let thousands_digit := (product / 1000) % 10
  let units_digit := product % 10
  thousands_digit + units_digit = 5 := by
sorry

end NUMINAMATH_CALUDE_product_digit_sum_l3888_388848


namespace NUMINAMATH_CALUDE_point_equidistant_from_axes_l3888_388835

theorem point_equidistant_from_axes (a : ℝ) : 
  (∀ (x y : ℝ), x = a - 2 ∧ y = 6 - 2*a → |x| = |y|) → 
  (a = 8/3 ∨ a = 4) :=
sorry

end NUMINAMATH_CALUDE_point_equidistant_from_axes_l3888_388835


namespace NUMINAMATH_CALUDE_decreasing_f_implies_a_leq_neg_three_l3888_388898

/-- The function f(x) defined as x^2 + 2(a-1)x + 2 -/
def f (a : ℝ) (x : ℝ) : ℝ := x^2 + 2*(a-1)*x + 2

/-- The theorem stating that if f(x) is decreasing on (-∞, 4), then a ≤ -3 -/
theorem decreasing_f_implies_a_leq_neg_three (a : ℝ) :
  (∀ x y, x < y → y < 4 → f a x > f a y) → a ≤ -3 := by
  sorry

end NUMINAMATH_CALUDE_decreasing_f_implies_a_leq_neg_three_l3888_388898


namespace NUMINAMATH_CALUDE_square_equation_proof_l3888_388808

theorem square_equation_proof (h1 : 3 > 1) (h2 : 1 > 1) : (3 * (1^3 + 3))^2 = 8339 := by
  sorry

end NUMINAMATH_CALUDE_square_equation_proof_l3888_388808


namespace NUMINAMATH_CALUDE_only_D_correct_l3888_388802

/-- Represents the contestants in the singing competition -/
inductive Contestant : Type
  | one | two | three | four | five | six

/-- Represents the students who made guesses -/
inductive Student : Type
  | A | B | C | D

/-- The guess made by each student -/
def studentGuess (s : Student) : Contestant → Prop :=
  match s with
  | Student.A => λ c => c = Contestant.four ∨ c = Contestant.five
  | Student.B => λ c => c ≠ Contestant.three
  | Student.C => λ c => c = Contestant.one ∨ c = Contestant.two ∨ c = Contestant.six
  | Student.D => λ c => c ≠ Contestant.four ∧ c ≠ Contestant.five ∧ c ≠ Contestant.six

/-- The theorem to be proved -/
theorem only_D_correct :
  ∃ (winner : Contestant),
    (∀ (s : Student), s ≠ Student.D → ¬(studentGuess s winner)) ∧
    (studentGuess Student.D winner) :=
  sorry

end NUMINAMATH_CALUDE_only_D_correct_l3888_388802


namespace NUMINAMATH_CALUDE_triangles_in_4x4_grid_l3888_388883

/-- Represents a triangular grid with side length n --/
def TriangularGrid (n : ℕ) := Unit

/-- Counts the number of triangles in a triangular grid --/
def countTriangles (grid : TriangularGrid 4) : ℕ := sorry

/-- Theorem: The number of triangles in a 4x4 triangular grid is 20 --/
theorem triangles_in_4x4_grid :
  ∀ (grid : TriangularGrid 4), countTriangles grid = 20 := by sorry

end NUMINAMATH_CALUDE_triangles_in_4x4_grid_l3888_388883


namespace NUMINAMATH_CALUDE_todays_production_l3888_388853

theorem todays_production (n : ℕ) (past_average current_average : ℝ) 
  (h1 : n = 11)
  (h2 : past_average = 50)
  (h3 : current_average = 55) :
  (n + 1) * current_average - n * past_average = 110 :=
by sorry

end NUMINAMATH_CALUDE_todays_production_l3888_388853


namespace NUMINAMATH_CALUDE_sum_due_proof_l3888_388806

/-- Represents the relationship between banker's discount, true discount, and face value. -/
def bankers_discount_relation (bd td fv : ℚ) : Prop :=
  bd = td + (td * bd / fv)

/-- Proves that given a banker's discount of 80 and a true discount of 70,
    the face value (sum due) is 560. -/
theorem sum_due_proof :
  ∃ (fv : ℚ), bankers_discount_relation 80 70 fv ∧ fv = 560 :=
by sorry

end NUMINAMATH_CALUDE_sum_due_proof_l3888_388806


namespace NUMINAMATH_CALUDE_smallest_number_with_conditions_l3888_388842

def containsOnly3And4 (n : ℕ) : Prop :=
  ∀ d, d ∈ n.digits 10 → d = 3 ∨ d = 4

def contains3And4 (n : ℕ) : Prop :=
  3 ∈ n.digits 10 ∧ 4 ∈ n.digits 10

def isMultipleOf3And4 (n : ℕ) : Prop :=
  n % 3 = 0 ∧ n % 4 = 0

theorem smallest_number_with_conditions :
  ∀ n : ℕ, 
    containsOnly3And4 n ∧ 
    contains3And4 n ∧ 
    isMultipleOf3And4 n →
    n ≥ 3444 :=
by sorry

end NUMINAMATH_CALUDE_smallest_number_with_conditions_l3888_388842


namespace NUMINAMATH_CALUDE_basketball_handshakes_l3888_388863

theorem basketball_handshakes : 
  let team_size : ℕ := 6
  let referee_count : ℕ := 3
  let inter_team_handshakes := team_size * team_size
  let player_referee_handshakes := (2 * team_size) * referee_count
  inter_team_handshakes + player_referee_handshakes = 72 :=
by sorry

end NUMINAMATH_CALUDE_basketball_handshakes_l3888_388863

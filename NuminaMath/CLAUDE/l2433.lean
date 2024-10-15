import Mathlib

namespace NUMINAMATH_CALUDE_circle_extrema_l2433_243387

theorem circle_extrema (x y : ℝ) (h : (x - 3)^2 + (y - 3)^2 = 6) :
  (∃ (x₁ y₁ x₂ y₂ : ℝ), 
    (x₁ - 3)^2 + (y₁ - 3)^2 = 6 ∧ 
    (x₂ - 3)^2 + (y₂ - 3)^2 = 6 ∧
    (∀ (x' y' : ℝ), (x' - 3)^2 + (y' - 3)^2 = 6 → y' / x' ≤ y₁ / x₁ ∧ y' / x' ≥ y₂ / x₂) ∧
    y₁ / x₁ = 3 + 2 * Real.sqrt 2 ∧
    y₂ / x₂ = 3 - 2 * Real.sqrt 2) ∧
  (∃ (x₃ y₃ x₄ y₄ : ℝ),
    (x₃ - 3)^2 + (y₃ - 3)^2 = 6 ∧
    (x₄ - 3)^2 + (y₄ - 3)^2 = 6 ∧
    (∀ (x' y' : ℝ), (x' - 3)^2 + (y' - 3)^2 = 6 → 
      Real.sqrt ((x' - 2)^2 + y'^2) ≤ Real.sqrt ((x₃ - 2)^2 + y₃^2) ∧
      Real.sqrt ((x' - 2)^2 + y'^2) ≥ Real.sqrt ((x₄ - 2)^2 + y₄^2)) ∧
    Real.sqrt ((x₃ - 2)^2 + y₃^2) = Real.sqrt 10 + Real.sqrt 6 ∧
    Real.sqrt ((x₄ - 2)^2 + y₄^2) = Real.sqrt 10 - Real.sqrt 6) := by
  sorry

end NUMINAMATH_CALUDE_circle_extrema_l2433_243387


namespace NUMINAMATH_CALUDE_correct_calculation_l2433_243336

theorem correct_calculation (a b : ℝ) : -7 * a * b^2 + 4 * a * b^2 = -3 * a * b^2 := by
  sorry

end NUMINAMATH_CALUDE_correct_calculation_l2433_243336


namespace NUMINAMATH_CALUDE_right_triangle_xy_length_l2433_243343

/-- Given a right triangle XYZ where YZ = 20 and tan Z = 3 * cos Y, 
    the length of XY is (40 * √2) / 3 -/
theorem right_triangle_xy_length (X Y Z : ℝ) : 
  -- Triangle XYZ is right-angled at X
  X + Y + Z = Real.pi ∧ X = Real.pi / 2 →
  -- YZ = 20
  Real.sqrt ((Y - Z)^2 + X^2) = 20 →
  -- tan Z = 3 * cos Y
  Real.tan Z = 3 * Real.cos Y →
  -- XY = (40 * √2) / 3
  Real.sqrt (Y^2 + Z^2) = (40 * Real.sqrt 2) / 3 := by
  sorry

end NUMINAMATH_CALUDE_right_triangle_xy_length_l2433_243343


namespace NUMINAMATH_CALUDE_min_reciprocal_sum_l2433_243318

theorem min_reciprocal_sum (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a + b = 2) :
  (1 / a + 1 / b) ≥ 2 :=
sorry

end NUMINAMATH_CALUDE_min_reciprocal_sum_l2433_243318


namespace NUMINAMATH_CALUDE_intersection_point_existence_l2433_243307

theorem intersection_point_existence : ∃! x₀ : ℝ, x₀ ∈ Set.Ioo 1 2 ∧ x₀^3 = (1/2)^(x₀ - 2) := by
  sorry

end NUMINAMATH_CALUDE_intersection_point_existence_l2433_243307


namespace NUMINAMATH_CALUDE_sphere_volume_of_hexagonal_prism_l2433_243368

/-- A hexagonal prism with specific properties -/
structure HexagonalPrism where
  -- The base is a regular hexagon
  base_is_regular : Bool
  -- Side edges are perpendicular to the base
  edges_perpendicular : Bool
  -- All vertices lie on the same spherical surface
  vertices_on_sphere : Bool
  -- Volume of the prism
  volume : ℝ
  -- Perimeter of the base
  base_perimeter : ℝ

/-- Theorem stating the volume of the sphere containing the hexagonal prism -/
theorem sphere_volume_of_hexagonal_prism (prism : HexagonalPrism)
    (h1 : prism.base_is_regular = true)
    (h2 : prism.edges_perpendicular = true)
    (h3 : prism.vertices_on_sphere = true)
    (h4 : prism.volume = 9/8)
    (h5 : prism.base_perimeter = 3) :
    ∃ (sphere_volume : ℝ), sphere_volume = 4/3 * Real.pi := by
  sorry

end NUMINAMATH_CALUDE_sphere_volume_of_hexagonal_prism_l2433_243368


namespace NUMINAMATH_CALUDE_event_committee_count_l2433_243339

/-- The number of teams in the league -/
def num_teams : ℕ := 5

/-- The number of members in each team -/
def team_size : ℕ := 8

/-- The number of members selected from the host team -/
def host_selection : ℕ := 4

/-- The number of members selected from each non-host team -/
def non_host_selection : ℕ := 3

/-- The total number of possible event committees -/
def total_committees : ℕ := 3442073600

/-- Theorem stating the number of possible event committees -/
theorem event_committee_count :
  (num_teams : ℕ) *
  (Nat.choose team_size host_selection) *
  (Nat.choose team_size non_host_selection)^(num_teams - 1) =
  total_committees := by sorry

end NUMINAMATH_CALUDE_event_committee_count_l2433_243339


namespace NUMINAMATH_CALUDE_max_ab_empty_solution_set_l2433_243335

theorem max_ab_empty_solution_set (a b : ℝ) : 
  (∀ x > 0, x - a * Real.log x + a - b ≥ 0) → 
  ab ≤ (1/2 : ℝ) * Real.exp 3 := by
  sorry

end NUMINAMATH_CALUDE_max_ab_empty_solution_set_l2433_243335


namespace NUMINAMATH_CALUDE_calculation_proof_l2433_243394

theorem calculation_proof :
  ((-1 : ℝ) ^ 2021 + |-(Real.sqrt 3)| + (8 : ℝ) ^ (1/3) - Real.sqrt 16 = -3 + Real.sqrt 3) ∧
  (-(1 : ℝ) ^ 2 - (27 : ℝ) ^ (1/3) + |1 - Real.sqrt 2| = -5 + Real.sqrt 2) := by
  sorry

end NUMINAMATH_CALUDE_calculation_proof_l2433_243394


namespace NUMINAMATH_CALUDE_addition_preserves_inequality_l2433_243324

theorem addition_preserves_inequality (a b c d : ℝ) :
  a > b → c > d → a + c > b + d := by sorry

end NUMINAMATH_CALUDE_addition_preserves_inequality_l2433_243324


namespace NUMINAMATH_CALUDE_transform_result_l2433_243319

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Rotates a point 90 degrees counterclockwise around (1, 5) -/
def rotate90 (p : Point) : Point :=
  Point.mk (-(p.y - 5) + 1) ((p.x - 1) + 5)

/-- Reflects a point about the line y = -x -/
def reflectAboutNegativeX (p : Point) : Point :=
  Point.mk (-p.y) (-p.x)

/-- The final transformation applied to the initial point -/
def transform (p : Point) : Point :=
  reflectAboutNegativeX (rotate90 p)

theorem transform_result (a b : ℝ) : 
  transform (Point.mk a b) = Point.mk (-6) 3 → b - a = 7 := by
  sorry

end NUMINAMATH_CALUDE_transform_result_l2433_243319


namespace NUMINAMATH_CALUDE_polynomial_property_l2433_243359

/-- A polynomial of the form x^2 + bx + c -/
def P (b c : ℝ) (x : ℝ) : ℝ := x^2 + b * x + c

/-- Theorem stating that if P(P(1)) = 0, P(P(-2)) = 0, and P(1) ≠ P(-2), then P(0) = -5/2 -/
theorem polynomial_property (b c : ℝ) :
  (P b c (P b c 1) = 0) →
  (P b c (P b c (-2)) = 0) →
  (P b c 1 ≠ P b c (-2)) →
  P b c 0 = -5/2 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_property_l2433_243359


namespace NUMINAMATH_CALUDE_not_right_angled_triangle_l2433_243397

theorem not_right_angled_triangle (A B C : ℝ) (h1 : A = B) (h2 : A = 2 * C) 
  (h3 : A + B + C = 180) : A ≠ 90 ∧ B ≠ 90 ∧ C ≠ 90 := by
  sorry

#check not_right_angled_triangle

end NUMINAMATH_CALUDE_not_right_angled_triangle_l2433_243397


namespace NUMINAMATH_CALUDE_min_sum_abc_l2433_243391

def is_min_sum (a b c : ℕ) : Prop :=
  ∀ x y z : ℕ, 
    (Nat.lcm (Nat.lcm x y) z = 48) → 
    (Nat.gcd x y = 4) → 
    (Nat.gcd y z = 3) → 
    a + b + c ≤ x + y + z

theorem min_sum_abc : 
  ∃ a b c : ℕ,
    (Nat.lcm (Nat.lcm a b) c = 48) ∧ 
    (Nat.gcd a b = 4) ∧ 
    (Nat.gcd b c = 3) ∧ 
    (is_min_sum a b c) ∧ 
    (a + b + c = 31) :=
sorry

end NUMINAMATH_CALUDE_min_sum_abc_l2433_243391


namespace NUMINAMATH_CALUDE_total_fish_count_l2433_243349

theorem total_fish_count (jerk_tuna : ℕ) (tall_tuna : ℕ) (swell_tuna : ℕ) : 
  jerk_tuna = 144 →
  tall_tuna = 2 * jerk_tuna →
  swell_tuna = tall_tuna + (tall_tuna / 2) →
  jerk_tuna + tall_tuna + swell_tuna = 864 :=
by
  sorry

end NUMINAMATH_CALUDE_total_fish_count_l2433_243349


namespace NUMINAMATH_CALUDE_nathan_ate_twenty_packages_l2433_243326

/-- The number of gumballs in each package -/
def gumballs_per_package : ℕ := 5

/-- The total number of gumballs Nathan ate -/
def total_gumballs_eaten : ℕ := 100

/-- The number of packages Nathan ate -/
def packages_eaten : ℕ := total_gumballs_eaten / gumballs_per_package

theorem nathan_ate_twenty_packages : packages_eaten = 20 := by
  sorry

end NUMINAMATH_CALUDE_nathan_ate_twenty_packages_l2433_243326


namespace NUMINAMATH_CALUDE_max_perimeter_of_special_triangle_l2433_243344

/-- Represents the sides of a triangle --/
structure Triangle where
  a : ℕ
  b : ℕ
  c : ℕ

/-- Checks if the given sides form a valid triangle --/
def isValidTriangle (t : Triangle) : Prop :=
  t.a + t.b > t.c ∧ t.b + t.c > t.a ∧ t.c + t.a > t.b

/-- Calculates the perimeter of a triangle --/
def perimeter (t : Triangle) : ℕ :=
  t.a + t.b + t.c

/-- Theorem stating the maximum perimeter of the triangle --/
theorem max_perimeter_of_special_triangle :
  ∃ (t : Triangle),
    t.a = 5 ∧
    t.b = 6 ∧
    isValidTriangle t ∧
    (∀ (t' : Triangle),
      t'.a = 5 →
      t'.b = 6 →
      isValidTriangle t' →
      perimeter t' ≤ perimeter t) ∧
    perimeter t = 21 :=
  sorry

end NUMINAMATH_CALUDE_max_perimeter_of_special_triangle_l2433_243344


namespace NUMINAMATH_CALUDE_ceiling_floor_difference_one_implies_fractional_part_l2433_243310

theorem ceiling_floor_difference_one_implies_fractional_part (x : ℝ) :
  ⌈x⌉ - ⌊x⌋ = 1 → 0 < x - ⌊x⌋ ∧ x - ⌊x⌋ < 1 := by sorry

end NUMINAMATH_CALUDE_ceiling_floor_difference_one_implies_fractional_part_l2433_243310


namespace NUMINAMATH_CALUDE_point_on_exponential_graph_l2433_243378

theorem point_on_exponential_graph (a : ℝ) (ha : a > 0) (ha_neq : a ≠ 1) :
  let f := fun x => a^(x - 1)
  f 1 = 1 := by sorry

end NUMINAMATH_CALUDE_point_on_exponential_graph_l2433_243378


namespace NUMINAMATH_CALUDE_square_difference_l2433_243360

theorem square_difference (m : ℕ) : (m + 1)^2 - m^2 = 2*m + 1 := by
  sorry

end NUMINAMATH_CALUDE_square_difference_l2433_243360


namespace NUMINAMATH_CALUDE_root_sum_theorem_l2433_243303

-- Define the quadratic equation
def quadratic_eq (k x : ℝ) : ℝ := k * (x^2 - x) + x + 5

-- Define the condition for k1 and k2
def k_condition (k : ℝ) : Prop :=
  ∃ a b : ℝ, quadratic_eq k a = 0 ∧ quadratic_eq k b = 0 ∧ a / b + b / a = 4 / 5

-- Theorem statement
theorem root_sum_theorem (k1 k2 : ℝ) :
  k_condition k1 ∧ k_condition k2 → k1 / k2 + k2 / k1 = 254 := by
  sorry

end NUMINAMATH_CALUDE_root_sum_theorem_l2433_243303


namespace NUMINAMATH_CALUDE_divisibility_implies_sum_divisibility_l2433_243317

def sum_of_digits (n : ℕ) : ℕ :=
  if n < 10 then n
  else (n % 10) + sum_of_digits (n / 10)

theorem divisibility_implies_sum_divisibility (n : ℕ) 
  (h1 : n < 10000) (h2 : n % 99 = 0) : 
  (sum_of_digits n) % 18 = 0 := by
sorry

end NUMINAMATH_CALUDE_divisibility_implies_sum_divisibility_l2433_243317


namespace NUMINAMATH_CALUDE_star_commutative_star_not_distributive_no_star_identity_star_not_associative_l2433_243312

-- Define the binary operation ⋆
def star (x y : ℝ) : ℝ := x^2 * y^2 + x + y

-- Commutativity
theorem star_commutative : ∀ x y : ℝ, star x y = star y x := by sorry

-- Non-distributivity
theorem star_not_distributive : ¬(∀ x y z : ℝ, star x (y + z) = star x y + star x z) := by sorry

-- Non-existence of identity element
theorem no_star_identity : ¬(∃ e : ℝ, ∀ x : ℝ, star x e = x ∧ star e x = x) := by sorry

-- Non-associativity
theorem star_not_associative : ¬(∀ x y z : ℝ, star (star x y) z = star x (star y z)) := by sorry

end NUMINAMATH_CALUDE_star_commutative_star_not_distributive_no_star_identity_star_not_associative_l2433_243312


namespace NUMINAMATH_CALUDE_correct_arrangement_count_l2433_243379

/-- The number of ways to arrange 3 boys and 2 girls in a line, with the girls being adjacent -/
def arrangement_count : ℕ := 48

/-- The number of boys -/
def num_boys : ℕ := 3

/-- The number of girls -/
def num_girls : ℕ := 2

/-- Theorem stating that the number of arrangements is correct -/
theorem correct_arrangement_count : 
  arrangement_count = 
    (Nat.factorial num_boys) * 
    (Nat.choose (num_boys + 1) 1) * 
    (Nat.factorial num_girls) :=
by sorry

end NUMINAMATH_CALUDE_correct_arrangement_count_l2433_243379


namespace NUMINAMATH_CALUDE_operation_result_l2433_243366

-- Define the operations
def op1 (m n : ℤ) : ℤ := n^2 - m
def op2 (m k : ℚ) : ℚ := (k + 2*m) / 3

-- Theorem statement
theorem operation_result : (op2 (op1 3 3) (op1 2 5)) = 35/3 := by
  sorry

end NUMINAMATH_CALUDE_operation_result_l2433_243366


namespace NUMINAMATH_CALUDE_min_PM_dot_PF_l2433_243346

/-- Parabola C: y^2 = 2px (p > 0) -/
def parabola (p : ℝ) (x y : ℝ) : Prop := y^2 = 2*p*x ∧ p > 0

/-- Circle M with center on positive x-axis and tangent to y-axis -/
def circle_M (center_x : ℝ) (radius : ℝ) (x y : ℝ) : Prop :=
  (x - center_x)^2 + y^2 = radius^2 ∧ center_x > 0 ∧ center_x = radius

/-- Line m passing through origin with inclination angle π/3 -/
def line_m (x y : ℝ) : Prop := y = x * Real.sqrt 3

/-- Point A on directrix l and point B on circle M, both on line m -/
def points_A_B (A B : ℝ × ℝ) : Prop :=
  line_m A.1 A.2 ∧ line_m B.1 B.2 ∧ A.1^2 + A.2^2 = 4 ∧ B.1^2 + B.2^2 = 4

/-- Theorem: Minimum value of PM⋅PF is 2 -/
theorem min_PM_dot_PF (p : ℝ) (center_x radius : ℝ) (A B : ℝ × ℝ) :
  parabola p 1 2 →
  circle_M center_x radius center_x 0 →
  points_A_B A B →
  (∀ x y : ℝ, parabola p x y → 
    (x^2 - center_x*x + (center_x^2)/4 + y^2) ≥ 2) :=
sorry

end NUMINAMATH_CALUDE_min_PM_dot_PF_l2433_243346


namespace NUMINAMATH_CALUDE_length_in_cube4_is_4root3_l2433_243358

/-- The length of the portion of the line segment from (0,0,0) to (5,5,11) 
    contained in the cube with edge length 4, which extends from (0,0,5) to (4,4,9) -/
def lengthInCube4 : ℝ := sorry

/-- The coordinates of the entry point of the line segment into the cube with edge length 4 -/
def entryPoint : Fin 3 → ℝ
| 0 => 0
| 1 => 0
| 2 => 5

/-- The coordinates of the exit point of the line segment from the cube with edge length 4 -/
def exitPoint : Fin 3 → ℝ
| 0 => 4
| 1 => 4
| 2 => 9

theorem length_in_cube4_is_4root3 : lengthInCube4 = 4 * Real.sqrt 3 := by sorry

end NUMINAMATH_CALUDE_length_in_cube4_is_4root3_l2433_243358


namespace NUMINAMATH_CALUDE_plant_beds_calculation_l2433_243396

/-- Calculate the number of plant beds required for given vegetable plantings -/
theorem plant_beds_calculation (bean_seedlings pumpkin_seeds radishes : ℕ)
  (bean_per_row pumpkin_per_row radish_per_row : ℕ)
  (rows_per_bed : ℕ)
  (h1 : bean_seedlings = 64)
  (h2 : pumpkin_seeds = 84)
  (h3 : radishes = 48)
  (h4 : bean_per_row = 8)
  (h5 : pumpkin_per_row = 7)
  (h6 : radish_per_row = 6)
  (h7 : rows_per_bed = 2) :
  (bean_seedlings / bean_per_row + pumpkin_seeds / pumpkin_per_row + radishes / radish_per_row) / rows_per_bed = 14 := by
  sorry

end NUMINAMATH_CALUDE_plant_beds_calculation_l2433_243396


namespace NUMINAMATH_CALUDE_circle_equation_is_correct_l2433_243370

/-- The circle with center M(2, -1) that is tangent to the line x - 2y + 1 = 0 -/
def tangent_circle : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | (p.1 - 2)^2 + (p.2 + 1)^2 = 5}

/-- The line x - 2y + 1 = 0 -/
def tangent_line : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.1 - 2*p.2 + 1 = 0}

/-- The center of the circle -/
def circle_center : ℝ × ℝ := (2, -1)

theorem circle_equation_is_correct :
  ∃! (r : ℝ), r > 0 ∧
  (∀ p ∈ tangent_circle, dist p circle_center = r) ∧
  (∃ q ∈ tangent_line, dist q circle_center = r) ∧
  (∀ q ∈ tangent_line, dist q circle_center ≥ r) :=
sorry


end NUMINAMATH_CALUDE_circle_equation_is_correct_l2433_243370


namespace NUMINAMATH_CALUDE_cubic_roots_squared_l2433_243314

def f (x : ℝ) : ℝ := x^3 + 2*x^2 + 3*x + 4

def g (b c d : ℝ) (x : ℝ) : ℝ := x^3 + b*x^2 + c*x + d

theorem cubic_roots_squared (b c d : ℝ) :
  (∃ r₁ r₂ r₃ : ℝ, r₁ ≠ r₂ ∧ r₂ ≠ r₃ ∧ r₁ ≠ r₃ ∧
    f r₁ = 0 ∧ f r₂ = 0 ∧ f r₃ = 0) →
  (∀ x : ℝ, f x = 0 → g b c d (x^2) = 0) →
  b = 4 ∧ c = -15 ∧ d = -32 := by
sorry

end NUMINAMATH_CALUDE_cubic_roots_squared_l2433_243314


namespace NUMINAMATH_CALUDE_equation_solutions_l2433_243302

/-- The equation we're solving -/
def equation (x : ℝ) : Prop :=
  (15*x - x^2) / (x + 2) * (x + (15 - x) / (x + 2)) = 54

/-- The set of solutions to the equation -/
def solutions : Set ℝ := {0, -1, -3, -3.5}

/-- Theorem stating that the solutions are correct -/
theorem equation_solutions :
  ∀ x : ℝ, x ∈ solutions ↔ equation x :=
by sorry

end NUMINAMATH_CALUDE_equation_solutions_l2433_243302


namespace NUMINAMATH_CALUDE_ellipse_theorem_parabola_theorem_l2433_243363

-- Define the ellipses
def ellipse1 (x y : ℝ) := x^2/9 + y^2/4 = 1
def ellipse2 (x y : ℝ) := x^2/12 + y^2/7 = 1

-- Define the parabolas
def parabola1 (x y : ℝ) := x^2 = -2 * Real.sqrt 2 * y
def parabola2 (x y : ℝ) := y^2 = -8 * x

-- Theorem for the ellipse
theorem ellipse_theorem :
  (ellipse2 (-3) 2) ∧
  (∀ (x y : ℝ), ellipse1 x y ↔ ellipse2 x y) := by sorry

-- Theorem for the parabolas
theorem parabola_theorem :
  (parabola1 (-4) (-4 * Real.sqrt 2)) ∧
  (parabola2 (-4) (-4 * Real.sqrt 2)) ∧
  (∀ (x y : ℝ), parabola1 x y → x = 0 ∨ y = 0) ∧
  (∀ (x y : ℝ), parabola2 x y → x = 0 ∨ y = 0) := by sorry

end NUMINAMATH_CALUDE_ellipse_theorem_parabola_theorem_l2433_243363


namespace NUMINAMATH_CALUDE_boys_average_age_l2433_243334

theorem boys_average_age (a b c : ℕ) (h1 : a = 15) (h2 : b = 3 * a) (h3 : c = 4 * a) :
  (a + b + c) / 3 = 40 := by
  sorry

end NUMINAMATH_CALUDE_boys_average_age_l2433_243334


namespace NUMINAMATH_CALUDE_odd_function_root_property_l2433_243383

-- Define an odd function
def OddFunction (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = -f x

-- Define the property of being a root
def IsRoot (f : ℝ → ℝ) (x : ℝ) : Prop := f x = 0

-- State the theorem
theorem odd_function_root_property
  (f : ℝ → ℝ) (x₀ : ℝ)
  (h_odd : OddFunction f)
  (h_root : IsRoot (fun x => f x - Real.exp x) x₀) :
  IsRoot (fun x => f x * Real.exp x + 1) (-x₀) := by
sorry

end NUMINAMATH_CALUDE_odd_function_root_property_l2433_243383


namespace NUMINAMATH_CALUDE_no_solution_condition_l2433_243375

theorem no_solution_condition (k : ℝ) : 
  (∀ x : ℝ, x ≠ 0 → x ≠ -1 → (1 / (x + 1) ≠ 3 * k / x)) ↔ (k = 0 ∨ k = 1/3) :=
sorry

end NUMINAMATH_CALUDE_no_solution_condition_l2433_243375


namespace NUMINAMATH_CALUDE_job_pay_difference_l2433_243322

/-- Proves that the difference between two job pays is $375 given the total pay and the pay of the first job. -/
theorem job_pay_difference (total_pay first_job_pay : ℕ) 
  (h1 : total_pay = 3875)
  (h2 : first_job_pay = 2125) :
  first_job_pay - (total_pay - first_job_pay) = 375 := by
  sorry

end NUMINAMATH_CALUDE_job_pay_difference_l2433_243322


namespace NUMINAMATH_CALUDE_square_root_problem_l2433_243390

theorem square_root_problem (x y : ℝ) (h : Real.sqrt (2 * x - 16) + |x - 2 * y + 2| = 0) :
  Real.sqrt (x - 4 / 5 * y) = 2 ∨ Real.sqrt (x - 4 / 5 * y) = -2 := by
  sorry

end NUMINAMATH_CALUDE_square_root_problem_l2433_243390


namespace NUMINAMATH_CALUDE_marie_erasers_l2433_243376

/-- Given that Marie starts with 95 erasers and loses 42, prove that she ends with 53 erasers. -/
theorem marie_erasers : 
  let initial_erasers : ℕ := 95
  let lost_erasers : ℕ := 42
  initial_erasers - lost_erasers = 53 := by sorry

end NUMINAMATH_CALUDE_marie_erasers_l2433_243376


namespace NUMINAMATH_CALUDE_equation_solution_l2433_243392

theorem equation_solution : ∃ x : ℝ, 30 - (5 * 2) = 3 + x ∧ x = 17 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l2433_243392


namespace NUMINAMATH_CALUDE_cos_arcsin_eight_seventeenths_l2433_243381

theorem cos_arcsin_eight_seventeenths : 
  Real.cos (Real.arcsin (8/17)) = 15/17 := by sorry

end NUMINAMATH_CALUDE_cos_arcsin_eight_seventeenths_l2433_243381


namespace NUMINAMATH_CALUDE_euler_family_mean_age_l2433_243389

-- Define the list of ages
def euler_family_ages : List ℕ := [6, 6, 9, 11, 13, 16]

-- Theorem statement
theorem euler_family_mean_age :
  let ages := euler_family_ages
  let sum_ages := ages.sum
  let num_children := ages.length
  (sum_ages : ℚ) / num_children = 61 / 6 := by sorry

end NUMINAMATH_CALUDE_euler_family_mean_age_l2433_243389


namespace NUMINAMATH_CALUDE_celine_change_l2433_243348

def laptop_base_price : ℚ := 600
def smartphone_base_price : ℚ := 400
def tablet_base_price : ℚ := 250
def headphone_base_price : ℚ := 100

def laptop_discount : ℚ := 0.15
def smartphone_increase : ℚ := 0.10
def tablet_discount : ℚ := 0.20

def sales_tax : ℚ := 0.06

def laptop_quantity : ℕ := 2
def smartphone_quantity : ℕ := 3
def tablet_quantity : ℕ := 4
def headphone_quantity : ℕ := 6

def celine_budget : ℚ := 6000

theorem celine_change : 
  let laptop_price := laptop_base_price * (1 - laptop_discount)
  let smartphone_price := smartphone_base_price * (1 + smartphone_increase)
  let tablet_price := tablet_base_price * (1 - tablet_discount)
  let headphone_price := headphone_base_price

  let total_before_tax := 
    laptop_price * laptop_quantity +
    smartphone_price * smartphone_quantity +
    tablet_price * tablet_quantity +
    headphone_price * headphone_quantity

  let total_with_tax := total_before_tax * (1 + sales_tax)

  celine_budget - total_with_tax = 2035.60 := by sorry

end NUMINAMATH_CALUDE_celine_change_l2433_243348


namespace NUMINAMATH_CALUDE_smallest_value_in_range_l2433_243340

theorem smallest_value_in_range (x : ℝ) (h : 1 ≤ x ∧ x ≤ 2) :
  (1 / x ≤ x) ∧ (1 / x ≤ x^2) ∧ (1 / x ≤ 2*x) ∧ (1 / x ≤ Real.sqrt x) := by
  sorry

end NUMINAMATH_CALUDE_smallest_value_in_range_l2433_243340


namespace NUMINAMATH_CALUDE_absolute_value_sum_l2433_243357

theorem absolute_value_sum (a b : ℝ) : a^2 + b^2 > 1 → |a| + |b| > 1 := by
  sorry

end NUMINAMATH_CALUDE_absolute_value_sum_l2433_243357


namespace NUMINAMATH_CALUDE_solve_system_for_q_l2433_243311

theorem solve_system_for_q :
  ∀ p q : ℚ,
  5 * p + 3 * q = 7 →
  3 * p + 5 * q = 8 →
  q = 19 / 16 :=
by
  sorry

end NUMINAMATH_CALUDE_solve_system_for_q_l2433_243311


namespace NUMINAMATH_CALUDE_number_of_ways_to_choose_cards_l2433_243301

/-- Represents a standard deck of cards -/
def StandardDeck : ℕ := 52

/-- Number of suits in a standard deck -/
def NumberOfSuits : ℕ := 4

/-- Number of cards per suit in a standard deck -/
def CardsPerSuit : ℕ := 13

/-- Number of cards to be chosen -/
def CardsToChoose : ℕ := 4

/-- Number of cards to be chosen from one suit -/
def CardsFromOneSuit : ℕ := 2

/-- Calculate the number of ways to choose cards according to the problem conditions -/
def calculateWays : ℕ :=
  Nat.choose NumberOfSuits 3 *  -- Choose 3 suits from 4
  3 *  -- Choose which of the 3 suits will have 2 cards
  Nat.choose CardsPerSuit 2 *  -- Choose 2 cards from the chosen suit
  CardsPerSuit * CardsPerSuit  -- Choose 1 card each from the other two suits

/-- Theorem stating that the number of ways to choose cards is 158184 -/
theorem number_of_ways_to_choose_cards :
  calculateWays = 158184 := by sorry

end NUMINAMATH_CALUDE_number_of_ways_to_choose_cards_l2433_243301


namespace NUMINAMATH_CALUDE_solution_set_when_m_eq_2_range_of_m_when_f_leq_5_l2433_243350

-- Define the function f
def f (x m : ℝ) : ℝ := |x - 1| - |x - m|

-- Theorem for part I
theorem solution_set_when_m_eq_2 :
  {x : ℝ | f x 2 ≥ 1} = {x : ℝ | x ≥ 2} := by sorry

-- Theorem for part II
theorem range_of_m_when_f_leq_5 :
  {m : ℝ | ∀ x, f x m ≤ 5} = {m : ℝ | -4 ≤ m ∧ m ≤ 6} := by sorry

end NUMINAMATH_CALUDE_solution_set_when_m_eq_2_range_of_m_when_f_leq_5_l2433_243350


namespace NUMINAMATH_CALUDE_complex_magnitude_three_fourths_plus_three_i_l2433_243330

theorem complex_magnitude_three_fourths_plus_three_i :
  Complex.abs (3 / 4 + 3 * Complex.I) = Real.sqrt 153 / 4 := by
  sorry

end NUMINAMATH_CALUDE_complex_magnitude_three_fourths_plus_three_i_l2433_243330


namespace NUMINAMATH_CALUDE_range_of_fraction_l2433_243369

theorem range_of_fraction (a b : ℝ) (h1 : 0 < a) (h2 : a ≤ 2) (h3 : b ≥ 1) (h4 : b ≤ a^2) :
  ∃ (t : ℝ), t = b / a ∧ 1/2 ≤ t ∧ t ≤ 2 :=
by sorry

end NUMINAMATH_CALUDE_range_of_fraction_l2433_243369


namespace NUMINAMATH_CALUDE_arithmetic_sequence_third_term_l2433_243328

theorem arithmetic_sequence_third_term
  (a : ℤ) (d : ℤ) -- First term and common difference
  (h1 : a + 14 * d = 14) -- 15th term is 14
  (h2 : a + 15 * d = 17) -- 16th term is 17
  : a + 2 * d = -22 := -- 3rd term is -22
by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_third_term_l2433_243328


namespace NUMINAMATH_CALUDE_problem_solution_l2433_243338

theorem problem_solution (p q : ℤ) 
  (h1 : p > 1) 
  (h2 : q > 1) 
  (h3 : ∃ k : ℤ, (2 * p - 1) = k * q) 
  (h4 : ∃ m : ℤ, (2 * q - 1) = m * p) : 
  p + q = 8 := by
sorry

end NUMINAMATH_CALUDE_problem_solution_l2433_243338


namespace NUMINAMATH_CALUDE_grid_sequence_problem_l2433_243327

theorem grid_sequence_problem (row : List ℤ) (d_col : ℤ) (last_col : ℤ) (M : ℤ) :
  row = [15, 11, 7] →
  d_col = -5 →
  last_col = -4 →
  M = last_col - 4 * d_col →
  M = 6 := by
  sorry

end NUMINAMATH_CALUDE_grid_sequence_problem_l2433_243327


namespace NUMINAMATH_CALUDE_parabola_points_ordering_l2433_243367

/-- The parabola function -/
def f (x : ℝ) : ℝ := -x^2 - 2*x + 2

/-- Point A on the parabola -/
def A : ℝ × ℝ := (-2, f (-2))

/-- Point B on the parabola -/
def B : ℝ × ℝ := (1, f 1)

/-- Point C on the parabola -/
def C : ℝ × ℝ := (2, f 2)

/-- y₁ is the y-coordinate of point A -/
def y₁ : ℝ := A.2

/-- y₂ is the y-coordinate of point B -/
def y₂ : ℝ := B.2

/-- y₃ is the y-coordinate of point C -/
def y₃ : ℝ := C.2

theorem parabola_points_ordering : y₁ > y₂ ∧ y₂ > y₃ := by
  sorry

end NUMINAMATH_CALUDE_parabola_points_ordering_l2433_243367


namespace NUMINAMATH_CALUDE_right_triangle_has_multiple_altitudes_l2433_243365

/-- A right triangle is a triangle with one right angle. -/
structure RightTriangle where
  vertices : Fin 3 → ℝ × ℝ
  is_right_angle : sorry

/-- An altitude of a triangle is a line segment from a vertex perpendicular to the opposite side. -/
def altitude (t : RightTriangle) (v : Fin 3) : ℝ × ℝ := sorry

/-- The number of altitudes in a right triangle -/
def num_altitudes (t : RightTriangle) : ℕ := sorry

theorem right_triangle_has_multiple_altitudes (t : RightTriangle) : num_altitudes t > 1 := by
  sorry

end NUMINAMATH_CALUDE_right_triangle_has_multiple_altitudes_l2433_243365


namespace NUMINAMATH_CALUDE_western_village_conscription_l2433_243372

theorem western_village_conscription 
  (north_pop : ℕ) 
  (west_pop : ℕ) 
  (south_pop : ℕ) 
  (total_conscripts : ℕ) 
  (h1 : north_pop = 8758) 
  (h2 : west_pop = 7236) 
  (h3 : south_pop = 8356) 
  (h4 : total_conscripts = 378) : 
  (west_pop : ℚ) / (north_pop + west_pop + south_pop : ℚ) * total_conscripts = 112 := by
sorry

end NUMINAMATH_CALUDE_western_village_conscription_l2433_243372


namespace NUMINAMATH_CALUDE_numerals_with_prime_first_digit_l2433_243388

/-- The set of prime digits less than 10 -/
def primedigits : Finset ℕ := {2, 3, 5, 7}

/-- The number of numerals with prime first digit -/
def num_numerals : ℕ := 400

/-- The number of digits in the numerals -/
def num_digits : ℕ := 3

theorem numerals_with_prime_first_digit :
  (primedigits.card : ℝ) * (10 ^ (num_digits - 1)) = num_numerals := by sorry

end NUMINAMATH_CALUDE_numerals_with_prime_first_digit_l2433_243388


namespace NUMINAMATH_CALUDE_rectangle_max_area_l2433_243355

theorem rectangle_max_area (x y : ℝ) (h : x > 0 ∧ y > 0) :
  2 * x + 2 * y = 60 → x * y ≤ 225 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_max_area_l2433_243355


namespace NUMINAMATH_CALUDE_orange_price_calculation_l2433_243374

-- Define the price function for oranges
def orange_price (mass : ℝ) : ℝ := sorry

-- State the theorem
theorem orange_price_calculation 
  (proportional : ∀ m₁ m₂ : ℝ, orange_price m₁ / m₁ = orange_price m₂ / m₂)
  (given_price : orange_price 12 = 36) :
  orange_price 2 = 6 := by sorry

end NUMINAMATH_CALUDE_orange_price_calculation_l2433_243374


namespace NUMINAMATH_CALUDE_fish_distribution_theorem_l2433_243345

theorem fish_distribution_theorem (a b c d e f : ℕ) : 
  a + b + c + d + e + f = 100 ∧ 
  a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ a ≠ e ∧ a ≠ f ∧ 
  b ≠ c ∧ b ≠ d ∧ b ≠ e ∧ b ≠ f ∧ 
  c ≠ d ∧ c ≠ e ∧ c ≠ f ∧ 
  d ≠ e ∧ d ≠ f ∧ 
  e ≠ f ∧
  (b + c + d + e + f) % 5 = 0 ∧
  (a + c + d + e + f) % 5 = 0 ∧
  (a + b + d + e + f) % 5 = 0 ∧
  (a + b + c + e + f) % 5 = 0 ∧
  (a + b + c + d + f) % 5 = 0 ∧
  (a + b + c + d + e) % 5 = 0 →
  a = 20 ∨ b = 20 ∨ c = 20 ∨ d = 20 ∨ e = 20 ∨ f = 20 :=
by sorry

end NUMINAMATH_CALUDE_fish_distribution_theorem_l2433_243345


namespace NUMINAMATH_CALUDE_cubic_equation_solution_l2433_243386

theorem cubic_equation_solution : 
  ∃ x : ℝ, x^3 + 2*(x+1)^3 + (x+2)^3 = (x+4)^3 ∧ x = 3 :=
by
  sorry

end NUMINAMATH_CALUDE_cubic_equation_solution_l2433_243386


namespace NUMINAMATH_CALUDE_third_plus_fifth_sum_l2433_243371

/-- A geometric sequence with the given properties -/
structure GeometricSequence where
  a : ℕ → ℝ
  q : ℝ
  first_third_sum : a 1 + a 3 = 5
  common_ratio : q = 2
  is_geometric : ∀ n : ℕ, a (n + 1) = a n * q

/-- The theorem stating that a_3 + a_5 = 20 for the given geometric sequence -/
theorem third_plus_fifth_sum (seq : GeometricSequence) : seq.a 3 + seq.a 5 = 20 := by
  sorry

end NUMINAMATH_CALUDE_third_plus_fifth_sum_l2433_243371


namespace NUMINAMATH_CALUDE_total_campers_is_150_l2433_243354

/-- The total number of campers recorded for the past three weeks -/
def total_campers (three_weeks_ago two_weeks_ago last_week : ℕ) : ℕ :=
  three_weeks_ago + two_weeks_ago + last_week

/-- Proof that the total number of campers is 150 -/
theorem total_campers_is_150 :
  ∃ (three_weeks_ago two_weeks_ago last_week : ℕ),
    two_weeks_ago = 40 ∧
    two_weeks_ago = three_weeks_ago + 10 ∧
    last_week = 80 ∧
    total_campers three_weeks_ago two_weeks_ago last_week = 150 :=
by
  sorry

end NUMINAMATH_CALUDE_total_campers_is_150_l2433_243354


namespace NUMINAMATH_CALUDE_power_inequality_l2433_243315

theorem power_inequality (a b : ℝ) (h1 : 0 < a) (h2 : a < b) (h3 : b < 2) : a^b < b^a := by
  sorry

end NUMINAMATH_CALUDE_power_inequality_l2433_243315


namespace NUMINAMATH_CALUDE_decreasing_cubic_implies_m_leq_neg_three_exists_m_leq_neg_three_not_decreasing_l2433_243351

/-- A function f : ℝ → ℝ is decreasing if for all x y, x < y implies f x > f y -/
def DecreasingFunction (f : ℝ → ℝ) : Prop :=
  ∀ x y, x < y → f x > f y

/-- The cubic function f(x) = mx³ + 3x² - x + 1 -/
def f (m : ℝ) (x : ℝ) : ℝ := m * x^3 + 3 * x^2 - x + 1

theorem decreasing_cubic_implies_m_leq_neg_three :
  ∀ m : ℝ, DecreasingFunction (f m) → m ≤ -3 :=
sorry

theorem exists_m_leq_neg_three_not_decreasing :
  ∃ m : ℝ, m ≤ -3 ∧ ¬(DecreasingFunction (f m)) :=
sorry

end NUMINAMATH_CALUDE_decreasing_cubic_implies_m_leq_neg_three_exists_m_leq_neg_three_not_decreasing_l2433_243351


namespace NUMINAMATH_CALUDE_min_points_in_segment_seven_is_minimum_l2433_243373

-- Define the type for points on the number line
def Point := ℝ

-- Define the segments
def Segment := Set Point

-- Define the three segments
def leftSegment : Segment := {x : ℝ | x < -2}
def middleSegment : Segment := {x : ℝ | -2 ≤ x ∧ x ≤ 2}
def rightSegment : Segment := {x : ℝ | x > 2}

-- Define a property for a set of points
def hasThreePointsInOneSegment (points : Set Point) : Prop :=
  (points ∩ leftSegment).ncard ≥ 3 ∨
  (points ∩ middleSegment).ncard ≥ 3 ∨
  (points ∩ rightSegment).ncard ≥ 3

-- The main theorem
theorem min_points_in_segment :
  ∀ n : ℕ, n ≥ 7 →
    ∀ points : Set Point, points.ncard = n →
      hasThreePointsInOneSegment points :=
sorry

theorem seven_is_minimum :
  ∃ points : Set Point, points.ncard = 6 ∧
    ¬hasThreePointsInOneSegment points :=
sorry

end NUMINAMATH_CALUDE_min_points_in_segment_seven_is_minimum_l2433_243373


namespace NUMINAMATH_CALUDE_smallest_number_of_pens_l2433_243361

theorem smallest_number_of_pens (pen_package_size : Nat) (pencil_package_size : Nat)
  (h1 : pen_package_size = 12)
  (h2 : pencil_package_size = 15) :
  Nat.lcm pen_package_size pencil_package_size = 60 := by
  sorry

end NUMINAMATH_CALUDE_smallest_number_of_pens_l2433_243361


namespace NUMINAMATH_CALUDE_min_distance_point_l2433_243332

def A : ℝ × ℝ := (1, -1)
def B : ℝ × ℝ := (2, 2)

def distance_squared (p1 p2 : ℝ × ℝ) : ℝ :=
  (p1.1 - p2.1)^2 + (p1.2 - p2.2)^2

def sum_of_distances (p : ℝ × ℝ) : ℝ :=
  distance_squared p A + distance_squared p B

def is_on_line (p : ℝ × ℝ) : Prop :=
  p.1 = p.2

theorem min_distance_point :
  ∃ (p : ℝ × ℝ), is_on_line p ∧
    ∀ (q : ℝ × ℝ), is_on_line q → sum_of_distances p ≤ sum_of_distances q :=
by sorry

end NUMINAMATH_CALUDE_min_distance_point_l2433_243332


namespace NUMINAMATH_CALUDE_beverage_selection_probabilities_l2433_243377

def total_cups : ℕ := 5
def type_a_cups : ℕ := 3
def type_b_cups : ℕ := 2
def cups_to_select : ℕ := 3

def probability_all_correct : ℚ := 1 / 10
def probability_at_least_two_correct : ℚ := 7 / 10

theorem beverage_selection_probabilities :
  (total_cups = type_a_cups + type_b_cups) →
  (cups_to_select = type_a_cups) →
  (probability_all_correct = 1 / (Nat.choose total_cups cups_to_select)) ∧
  (probability_at_least_two_correct = 
    (Nat.choose type_a_cups cups_to_select + 
     Nat.choose type_a_cups (cups_to_select - 1) * Nat.choose type_b_cups 1) / 
    (Nat.choose total_cups cups_to_select)) := by
  sorry

end NUMINAMATH_CALUDE_beverage_selection_probabilities_l2433_243377


namespace NUMINAMATH_CALUDE_sin_n_eq_cos_630_l2433_243342

theorem sin_n_eq_cos_630 (n : ℤ) :
  -180 ≤ n ∧ n ≤ 180 →
  (Real.sin (n * π / 180) = Real.cos (630 * π / 180) ↔ n = 0 ∨ n = -180 ∨ n = 180) := by
sorry

end NUMINAMATH_CALUDE_sin_n_eq_cos_630_l2433_243342


namespace NUMINAMATH_CALUDE_min_blue_chips_correct_l2433_243325

/-- Represents the number of chips of each color in the box -/
structure ChipCounts where
  white : ℕ
  blue : ℕ
  red : ℕ

/-- Checks if the chip counts satisfy the given conditions -/
def satisfiesConditions (counts : ChipCounts) : Prop :=
  counts.blue ≥ counts.white / 3 ∧
  counts.blue ≤ counts.red / 4 ∧
  counts.white + counts.blue ≥ 75

/-- The minimum number of blue chips that satisfies the conditions -/
def minBlueChips : ℕ := 19

theorem min_blue_chips_correct :
  (∀ counts : ChipCounts, satisfiesConditions counts → counts.blue ≥ minBlueChips) ∧
  (∃ counts : ChipCounts, satisfiesConditions counts ∧ counts.blue = minBlueChips) := by
  sorry

end NUMINAMATH_CALUDE_min_blue_chips_correct_l2433_243325


namespace NUMINAMATH_CALUDE_max_gcd_of_sum_1071_l2433_243300

theorem max_gcd_of_sum_1071 :
  ∃ (m : ℕ), m > 0 ∧ 
  (∀ (x y : ℕ), x > 0 → y > 0 → x + y = 1071 → Nat.gcd x y ≤ m) ∧
  (∃ (x y : ℕ), x > 0 ∧ y > 0 ∧ x + y = 1071 ∧ Nat.gcd x y = m) ∧
  m = 357 :=
sorry

end NUMINAMATH_CALUDE_max_gcd_of_sum_1071_l2433_243300


namespace NUMINAMATH_CALUDE_circle_tangent_sum_radii_l2433_243323

/-- A circle with center C(r,r) is tangent to the positive x-axis and y-axis,
    and externally tangent to a circle centered at (4,0) with radius 2.
    The sum of all possible radii of the circle with center C is 12. -/
theorem circle_tangent_sum_radii :
  ∀ r : ℝ,
  (r > 0) →
  ((r - 4)^2 + r^2 = (r + 2)^2) →
  (∃ r₁ r₂ : ℝ, (r₁ > 0 ∧ r₂ > 0) ∧ 
    ((r₁ - 4)^2 + r₁^2 = (r₁ + 2)^2) ∧
    ((r₂ - 4)^2 + r₂^2 = (r₂ + 2)^2) ∧
    r₁ + r₂ = 12) :=
by
  sorry

#check circle_tangent_sum_radii

end NUMINAMATH_CALUDE_circle_tangent_sum_radii_l2433_243323


namespace NUMINAMATH_CALUDE_nikita_mistaken_l2433_243352

theorem nikita_mistaken (b s : ℕ) : 
  (9 * b + 4 * s) - (4 * b + 9 * s) ≠ 49 := by
  sorry

end NUMINAMATH_CALUDE_nikita_mistaken_l2433_243352


namespace NUMINAMATH_CALUDE_two_digit_penultimate_five_l2433_243313

/-- A function that returns the penultimate digit of a natural number -/
def penultimateDigit (n : ℕ) : ℕ :=
  (n / 10) % 10

/-- A predicate that checks if a number is a two-digit number -/
def isTwoDigit (n : ℕ) : Prop :=
  10 ≤ n ∧ n ≤ 99

theorem two_digit_penultimate_five :
  ∀ x : ℕ, isTwoDigit x →
    (∃ k : ℤ, penultimateDigit (x * k.natAbs) = 5) ↔
    (x = 25 ∨ x = 50 ∨ x = 75) :=
by sorry

end NUMINAMATH_CALUDE_two_digit_penultimate_five_l2433_243313


namespace NUMINAMATH_CALUDE_min_consecutive_sum_36_proof_l2433_243333

/-- The sum of N consecutive integers starting from a -/
def sum_consecutive (a : ℤ) (N : ℕ) : ℤ := N * (2 * a + N - 1) / 2

/-- Predicate to check if a sequence of N consecutive integers starting from a sums to 36 -/
def is_valid_sequence (a : ℤ) (N : ℕ) : Prop := sum_consecutive a N = 36

/-- The minimum number of consecutive integers that sum to 36 -/
def min_consecutive_sum_36 : ℕ := 3

theorem min_consecutive_sum_36_proof :
  (∃ a : ℤ, is_valid_sequence a min_consecutive_sum_36) ∧
  (∀ N : ℕ, N < min_consecutive_sum_36 → ∀ a : ℤ, ¬is_valid_sequence a N) :=
sorry

end NUMINAMATH_CALUDE_min_consecutive_sum_36_proof_l2433_243333


namespace NUMINAMATH_CALUDE_max_m_is_zero_l2433_243362

/-- The condition function as described in the problem -/
def condition (m : ℝ) : Prop :=
  ∀ x₁ x₂ : ℝ, x₁ < x₂ ∧ x₁ < m ∧ x₂ < m → (x₂ * Real.exp x₁ - x₁ * Real.exp x₂) / (Real.exp x₂ - Real.exp x₁) > 1

/-- The theorem stating that the maximum value of m for which the condition holds is 0 -/
theorem max_m_is_zero :
  ∀ m : ℝ, (∀ m' > m, ¬ condition m') → m ≤ 0 :=
sorry

end NUMINAMATH_CALUDE_max_m_is_zero_l2433_243362


namespace NUMINAMATH_CALUDE_intersection_of_A_and_B_l2433_243393

def A : Set ℝ := {-2, -1, 0, 1, 2}
def B : Set ℝ := {x | -Real.sqrt 3 ≤ x ∧ x ≤ 1}

theorem intersection_of_A_and_B : A ∩ B = {-1, 0, 1} := by sorry

end NUMINAMATH_CALUDE_intersection_of_A_and_B_l2433_243393


namespace NUMINAMATH_CALUDE_unique_number_property_l2433_243382

theorem unique_number_property : ∃! (a : ℕ), a > 1 ∧
  ∀ (p : ℕ), Prime p → (p ∣ (a^6 - 1) → (p ∣ (a^3 - 1) ∨ p ∣ (a^2 - 1))) →
  a = 2 :=
by sorry

end NUMINAMATH_CALUDE_unique_number_property_l2433_243382


namespace NUMINAMATH_CALUDE_equation_solutions_l2433_243306

theorem equation_solutions :
  (∃ x : ℚ, (17/2 : ℚ) * x = (17/2 : ℚ) + x ∧ x = (17/15 : ℚ)) ∧
  (∃ y : ℚ, y / (2/3 : ℚ) = y + (2/3 : ℚ) ∧ y = (4/3 : ℚ)) :=
by sorry

end NUMINAMATH_CALUDE_equation_solutions_l2433_243306


namespace NUMINAMATH_CALUDE_parabola_symmetry_line_l2433_243356

/-- The parabola function -/
def parabola (x : ℝ) : ℝ := 2 * x^2

/-- The line of symmetry -/
def symmetry_line (x m : ℝ) : ℝ := x + m

/-- Theorem: For a parabola y = 2x² with two points symmetric about y = x + m, 
    and their x-coordinates multiply to -1/2, m equals 3/2 -/
theorem parabola_symmetry_line (x₁ x₂ y₁ y₂ m : ℝ) : 
  y₁ = parabola x₁ →
  y₂ = parabola x₂ →
  (y₁ + y₂) / 2 = symmetry_line ((x₁ + x₂) / 2) m →
  x₁ * x₂ = -1/2 →
  m = 3/2 := by sorry

end NUMINAMATH_CALUDE_parabola_symmetry_line_l2433_243356


namespace NUMINAMATH_CALUDE_plane_equation_transformation_l2433_243384

theorem plane_equation_transformation (A B C D : ℝ) 
  (hA : A ≠ 0) (hB : B ≠ 0) (hC : C ≠ 0) (hD : D ≠ 0) :
  ∃ p q r : ℝ, 
    (∀ x y z : ℝ, A * x + B * y + C * z + D = 0 ↔ x / p + y / q + z / r = 1) ∧
    p = -D / A ∧ q = -D / B ∧ r = -D / C :=
by sorry

end NUMINAMATH_CALUDE_plane_equation_transformation_l2433_243384


namespace NUMINAMATH_CALUDE_line_circle_intersection_l2433_243320

theorem line_circle_intersection (a b : ℝ) (h : ∃ (x y : ℝ), x^2 + y^2 = 1 ∧ x/a + y/b = 1) :
  1/a^2 + 1/b^2 ≥ 1 := by
sorry

end NUMINAMATH_CALUDE_line_circle_intersection_l2433_243320


namespace NUMINAMATH_CALUDE_paintings_removed_l2433_243329

theorem paintings_removed (initial : ℕ) (final : ℕ) (h1 : initial = 98) (h2 : final = 95) :
  initial - final = 3 := by
  sorry

end NUMINAMATH_CALUDE_paintings_removed_l2433_243329


namespace NUMINAMATH_CALUDE_martha_has_115_cards_l2433_243395

/-- The number of cards Martha has at the end of the transactions -/
def martha_final_cards : ℕ :=
  let initial_cards : ℕ := 3
  let cards_from_emily : ℕ := 25
  let cards_from_alex : ℕ := 43
  let cards_from_jenny : ℕ := 58
  let cards_given_to_sam : ℕ := 14
  initial_cards + cards_from_emily + cards_from_alex + cards_from_jenny - cards_given_to_sam

/-- Theorem stating that Martha ends up with 115 cards -/
theorem martha_has_115_cards : martha_final_cards = 115 := by
  sorry

end NUMINAMATH_CALUDE_martha_has_115_cards_l2433_243395


namespace NUMINAMATH_CALUDE_root_product_equals_eight_l2433_243316

theorem root_product_equals_eight : 
  (64 : ℝ) ^ (1/6) * (8 : ℝ) ^ (1/3) * (16 : ℝ) ^ (1/2) = 8 := by
  sorry

end NUMINAMATH_CALUDE_root_product_equals_eight_l2433_243316


namespace NUMINAMATH_CALUDE_cube_root_of_21952_l2433_243380

theorem cube_root_of_21952 : ∃ n : ℕ, n^3 = 21952 ∧ n = 28 := by
  sorry

end NUMINAMATH_CALUDE_cube_root_of_21952_l2433_243380


namespace NUMINAMATH_CALUDE_trajectory_is_ray_l2433_243353

/-- The set of complex numbers z satisfying |z+1| - |z-1| = 2 -/
def S : Set ℂ :=
  {z : ℂ | Complex.abs (z + 1) - Complex.abs (z - 1) = 2}

/-- A ray starting from (1, 0) and extending to the right -/
def R : Set ℂ :=
  {z : ℂ | ∃ (t : ℝ), t ≥ 0 ∧ z = 1 + t}

/-- Theorem stating that S equals R -/
theorem trajectory_is_ray : S = R := by
  sorry

end NUMINAMATH_CALUDE_trajectory_is_ray_l2433_243353


namespace NUMINAMATH_CALUDE_lowest_possible_score_l2433_243364

def test_count : Nat := 6
def max_score : Nat := 100
def target_average : Nat := 85
def min_score : Nat := 75

def first_four_scores : List Nat := [79, 88, 94, 91]

theorem lowest_possible_score :
  ∀ (score1 score2 : Nat),
  (score1 ≥ min_score) →
  (score2 ≥ min_score) →
  (List.sum first_four_scores + score1 + score2) / test_count = target_average →
  (∀ (s : Nat), s ≥ min_score ∧ s < score1 →
    (List.sum first_four_scores + s + score2) / test_count < target_average) →
  score1 = min_score ∨ score2 = min_score :=
by sorry

end NUMINAMATH_CALUDE_lowest_possible_score_l2433_243364


namespace NUMINAMATH_CALUDE_correct_geometry_problems_l2433_243305

theorem correct_geometry_problems (total_problems : ℕ) (total_algebra : ℕ) 
  (algebra_correct_ratio : ℚ) (algebra_incorrect_ratio : ℚ)
  (geometry_correct_ratio : ℚ) (geometry_incorrect_ratio : ℚ) :
  total_problems = 60 →
  total_algebra = 25 →
  algebra_correct_ratio = 3 →
  algebra_incorrect_ratio = 2 →
  geometry_correct_ratio = 4 →
  geometry_incorrect_ratio = 1 →
  ∃ (correct_geometry : ℕ), correct_geometry = 28 ∧
    correct_geometry * (geometry_correct_ratio + geometry_incorrect_ratio) = 
    (total_problems - total_algebra) * geometry_correct_ratio :=
by sorry

end NUMINAMATH_CALUDE_correct_geometry_problems_l2433_243305


namespace NUMINAMATH_CALUDE_circle_parameter_range_l2433_243331

-- Define the equation of the circle
def circle_equation (x y a : ℝ) : Prop :=
  x^2 + y^2 - 2*x + 4*y + 1 + a = 0

-- Define what it means for an equation to represent a circle
def represents_circle (a : ℝ) : Prop :=
  ∃ (h k r : ℝ), r > 0 ∧ ∀ (x y : ℝ), circle_equation x y a ↔ (x - h)^2 + (y - k)^2 = r^2

-- Theorem statement
theorem circle_parameter_range (a : ℝ) :
  represents_circle a → a < 4 := by
  sorry

end NUMINAMATH_CALUDE_circle_parameter_range_l2433_243331


namespace NUMINAMATH_CALUDE_tom_profit_l2433_243385

/-- Represents the types of properties Tom mows --/
inductive PropertyType
| Small
| Medium
| Large

/-- Calculates the total earnings from lawn mowing --/
def lawnMowingEarnings (smallCount medium_count largeCount : ℕ) : ℕ :=
  12 * smallCount + 15 * medium_count + 20 * largeCount

/-- Calculates the total earnings from side tasks --/
def sideTaskEarnings (taskCount : ℕ) : ℕ :=
  10 * taskCount

/-- Calculates the total expenses --/
def totalExpenses : ℕ := 20 + 10

/-- Calculates the total profit --/
def totalProfit (lawnEarnings sideEarnings : ℕ) : ℕ :=
  lawnEarnings + sideEarnings - totalExpenses

/-- Theorem stating Tom's profit for the given month --/
theorem tom_profit :
  totalProfit (lawnMowingEarnings 2 2 1) (sideTaskEarnings 5) = 94 := by
  sorry

end NUMINAMATH_CALUDE_tom_profit_l2433_243385


namespace NUMINAMATH_CALUDE_fraction_sum_equation_l2433_243399

theorem fraction_sum_equation (x y : ℕ+) : 
  x ≠ y → 
  (1 : ℚ) / x + (1 : ℚ) / y = 2 / 7 → 
  ((x = 4 ∧ y = 28) ∨ (x = 28 ∧ y = 4)) :=
by sorry

end NUMINAMATH_CALUDE_fraction_sum_equation_l2433_243399


namespace NUMINAMATH_CALUDE_hyperbola_focal_length_l2433_243341

/-- The focal length of a hyperbola with equation x²/2 - y²/2 = 1 is 2√2 -/
theorem hyperbola_focal_length : 
  ∃ (f : ℝ), f = 2 * Real.sqrt 2 ∧ 
  ∀ (x y : ℝ), x^2/2 - y^2/2 = 1 → 
  f = 2 * Real.sqrt ((x^2/2) + (y^2/2)) := by
  sorry

end NUMINAMATH_CALUDE_hyperbola_focal_length_l2433_243341


namespace NUMINAMATH_CALUDE_probability_four_threes_eight_dice_l2433_243304

def num_dice : ℕ := 8
def num_sides : ℕ := 8
def target_value : ℕ := 3
def num_target : ℕ := 4

def probability_exact_dice : ℚ :=
  (num_dice.choose num_target) *
  (1 / num_sides) ^ num_target *
  ((num_sides - 1) / num_sides) ^ (num_dice - num_target)

theorem probability_four_threes_eight_dice :
  probability_exact_dice = 168070 / 16777216 := by
  sorry

end NUMINAMATH_CALUDE_probability_four_threes_eight_dice_l2433_243304


namespace NUMINAMATH_CALUDE_comic_books_left_l2433_243337

theorem comic_books_left (initial_total : ℕ) (sold : ℕ) (left : ℕ) : 
  initial_total = 90 → sold = 65 → left = initial_total - sold → left = 25 := by
  sorry

end NUMINAMATH_CALUDE_comic_books_left_l2433_243337


namespace NUMINAMATH_CALUDE_check_mistake_problem_l2433_243308

theorem check_mistake_problem :
  ∃ (x y : ℕ), 
    10 ≤ x ∧ x < 100 ∧
    10 ≤ y ∧ y < 100 ∧
    100 * y + x - (100 * x + y) = 2556 ∧
    (x + y) % 11 = 0 ∧
    x = 9 := by
  sorry

end NUMINAMATH_CALUDE_check_mistake_problem_l2433_243308


namespace NUMINAMATH_CALUDE_inverse_of_P_l2433_243309

-- Define the original proposition P
def P : Prop → Prop → Prop := λ odd prime => odd → prime

-- Define the inverse proposition
def inverse_prop (p : Prop → Prop → Prop) : Prop → Prop → Prop :=
  λ a b => p b a

-- Theorem stating that the inverse of P is as described
theorem inverse_of_P :
  inverse_prop P = (λ prime odd => prime → odd) :=
by sorry

end NUMINAMATH_CALUDE_inverse_of_P_l2433_243309


namespace NUMINAMATH_CALUDE_trigonometric_equation_solution_l2433_243398

theorem trigonometric_equation_solution (x : ℝ) :
  (8.4743 * Real.tan (2 * x) - 4 * Real.tan (3 * x) = Real.tan (3 * x)^2 * Real.tan (2 * x)) ↔
  (∃ k : ℤ, x = k * Real.pi ∨ x = Real.arctan (Real.sqrt (3 / 5)) + k * Real.pi ∨ 
   x = -Real.arctan (Real.sqrt (3 / 5)) + k * Real.pi) :=
by sorry

end NUMINAMATH_CALUDE_trigonometric_equation_solution_l2433_243398


namespace NUMINAMATH_CALUDE_missing_number_proof_l2433_243347

def known_numbers : List ℝ := [13, 8, 13, 21, 23]

theorem missing_number_proof (mean : ℝ) (h_mean : mean = 14.2) :
  ∃ x : ℝ, (known_numbers.sum + x) / 6 = mean ∧ x = 7.2 := by
  sorry

end NUMINAMATH_CALUDE_missing_number_proof_l2433_243347


namespace NUMINAMATH_CALUDE_situp_ratio_l2433_243321

theorem situp_ratio (ken_situps : ℕ) (nathan_ratio : ℚ) (bob_situps : ℕ) :
  ken_situps = 20 →
  bob_situps = (ken_situps + nathan_ratio * ken_situps) / 2 →
  bob_situps = ken_situps + 10 →
  nathan_ratio = 2 :=
by sorry

end NUMINAMATH_CALUDE_situp_ratio_l2433_243321

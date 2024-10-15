import Mathlib

namespace NUMINAMATH_CALUDE_g_composition_of_3_l2550_255092

def g (x : ℤ) : ℤ :=
  if x % 3 = 0 then x / 3 else 4 * x - 1

theorem g_composition_of_3 : g (g (g (g 3))) = 3 := by
  sorry

end NUMINAMATH_CALUDE_g_composition_of_3_l2550_255092


namespace NUMINAMATH_CALUDE_right_triangle_inscribed_circle_l2550_255043

theorem right_triangle_inscribed_circle 
  (a b T C : ℝ) 
  (h_positive : a > 0 ∧ b > 0 ∧ T > 0 ∧ C > 0) 
  (h_right_triangle : ∃ (c h : ℝ), c = a + b ∧ T = (1/2) * c * h ∧ h^2 = a * b) 
  (h_inscribed : ∃ (R : ℝ), C = π * R^2 ∧ 2 * R = a + b) : 
  a * b = π * T^2 / C := by
sorry

end NUMINAMATH_CALUDE_right_triangle_inscribed_circle_l2550_255043


namespace NUMINAMATH_CALUDE_same_color_probability_l2550_255074

/-- The number of red candies in the box -/
def num_red : ℕ := 12

/-- The number of green candies in the box -/
def num_green : ℕ := 8

/-- The number of candies Alice and Bob each pick -/
def num_pick : ℕ := 3

/-- The probability that Alice and Bob pick the same number of candies of each color -/
def same_color_prob : ℚ := 231 / 1060

theorem same_color_probability :
  let total := num_red + num_green
  same_color_prob = (Nat.choose num_red num_pick * Nat.choose (num_red - num_pick) num_pick) / 
    (Nat.choose total num_pick * Nat.choose (total - num_pick) num_pick) +
    (Nat.choose num_red 2 * Nat.choose num_green 1 * Nat.choose (num_red - 2) 2 * 
    Nat.choose (num_green - 1) 1) / (Nat.choose total num_pick * Nat.choose (total - num_pick) num_pick) := by
  sorry

end NUMINAMATH_CALUDE_same_color_probability_l2550_255074


namespace NUMINAMATH_CALUDE_tangent_line_equation_l2550_255093

/-- Given a function f(x) = x³ + ax² + bx + 1 with f'(1) = 2a and f'(2) = -b,
    where a and b are real constants, the equation of the tangent line
    to y = f(x) at (1, f(1)) is 6x + 2y - 1 = 0. -/
theorem tangent_line_equation (a b : ℝ) :
  let f (x : ℝ) := x^3 + a*x^2 + b*x + 1
  let f' (x : ℝ) := 3*x^2 + 2*a*x + b
  f' 1 = 2*a →
  f' 2 = -b →
  ∃ (m c : ℝ), m = -3 ∧ c = -1/2 ∧ ∀ x y, y - f 1 = m * (x - 1) ↔ 6*x + 2*y - 1 = 0 :=
by sorry

end NUMINAMATH_CALUDE_tangent_line_equation_l2550_255093


namespace NUMINAMATH_CALUDE_no_number_decreases_by_1981_l2550_255016

theorem no_number_decreases_by_1981 : 
  ¬ ∃ (N : ℕ), 
    ∃ (M : ℕ), 
      (N ≠ 0) ∧ 
      (M ≠ 0) ∧
      (N = 1981 * M) ∧
      (∃ (d : ℕ) (k : ℕ), N = d * 10^k + M ∧ 1 ≤ d ∧ d ≤ 9) :=
by sorry

end NUMINAMATH_CALUDE_no_number_decreases_by_1981_l2550_255016


namespace NUMINAMATH_CALUDE_fraction_equality_implies_numerator_equality_l2550_255058

theorem fraction_equality_implies_numerator_equality 
  (a b c : ℝ) (h : c ≠ 0) : a / c = b / c → a = b := by
  sorry

end NUMINAMATH_CALUDE_fraction_equality_implies_numerator_equality_l2550_255058


namespace NUMINAMATH_CALUDE_platform_length_l2550_255064

/-- Given a train and two structures it passes through, calculate the length of the second structure. -/
theorem platform_length
  (train_length : ℝ)
  (tunnel_length : ℝ)
  (tunnel_time : ℝ)
  (platform_time : ℝ)
  (h1 : train_length = 330)
  (h2 : tunnel_length = 1200)
  (h3 : tunnel_time = 45)
  (h4 : platform_time = 15) :
  (tunnel_length + train_length) / tunnel_time * platform_time - train_length = 180 :=
by sorry

end NUMINAMATH_CALUDE_platform_length_l2550_255064


namespace NUMINAMATH_CALUDE_quadratic_roots_condition_l2550_255010

theorem quadratic_roots_condition (c : ℝ) :
  (∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ x₁^2 + 2*x₁ + 4*c = 0 ∧ x₂^2 + 2*x₂ + 4*c = 0) →
  c < (1/4 : ℝ) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_roots_condition_l2550_255010


namespace NUMINAMATH_CALUDE_binomial_sum_equals_sixteen_l2550_255091

theorem binomial_sum_equals_sixteen (h : (Complex.exp (Complex.I * Real.pi / 4))^10 = Complex.I) :
  Nat.choose 10 1 - Nat.choose 10 3 + (Nat.choose 10 5 / 2) = 2^4 := by
  sorry

end NUMINAMATH_CALUDE_binomial_sum_equals_sixteen_l2550_255091


namespace NUMINAMATH_CALUDE_rotated_point_coordinates_l2550_255094

def triangle_OAB (A : ℝ × ℝ) : Prop :=
  A.1 ≥ 0 ∧ A.2 > 0

def angle_ABO_is_right (A : ℝ × ℝ) : Prop :=
  (A.2 / A.1) * 10 = A.2

def angle_AOB_is_30 (A : ℝ × ℝ) : Prop :=
  A.2 / A.1 = Real.sqrt 3 / 3

def rotate_90_ccw (p : ℝ × ℝ) : ℝ × ℝ :=
  (-p.2, p.1)

theorem rotated_point_coordinates (A : ℝ × ℝ) :
  triangle_OAB A →
  angle_ABO_is_right A →
  angle_AOB_is_30 A →
  rotate_90_ccw A = (-10 * Real.sqrt 3 / 3, 10) := by
  sorry

end NUMINAMATH_CALUDE_rotated_point_coordinates_l2550_255094


namespace NUMINAMATH_CALUDE_pascal_triangle_41st_number_l2550_255057

/-- The binomial coefficient (n choose k) -/
def binomial (n k : ℕ) : ℕ := 
  if k > n then 0
  else (Nat.factorial n) / (Nat.factorial k * Nat.factorial (n - k))

/-- The number of elements in a row of Pascal's triangle -/
def pascal_row_length (row : ℕ) : ℕ := row + 1

/-- The row number (0-indexed) of Pascal's triangle with 43 numbers -/
def target_row : ℕ := 42

/-- The index (0-indexed) of the target number in the row -/
def target_index : ℕ := 40

theorem pascal_triangle_41st_number : 
  binomial target_row target_index = 861 ∧ 
  pascal_row_length target_row = 43 := by
  sorry

end NUMINAMATH_CALUDE_pascal_triangle_41st_number_l2550_255057


namespace NUMINAMATH_CALUDE_hen_duck_speed_ratio_l2550_255015

/-- The number of leaps a hen makes for every 8 leaps of a duck -/
def hen_leaps : ℕ := 6

/-- The number of duck leaps that equal 3 hen leaps -/
def duck_leaps_equal : ℕ := 4

/-- The number of hen leaps that equal 4 duck leaps -/
def hen_leaps_equal : ℕ := 3

/-- The number of duck leaps for which we compare hen leaps -/
def duck_comparison_leaps : ℕ := 8

theorem hen_duck_speed_ratio :
  (hen_leaps : ℚ) / duck_comparison_leaps = 1 := by
  sorry

end NUMINAMATH_CALUDE_hen_duck_speed_ratio_l2550_255015


namespace NUMINAMATH_CALUDE_F_two_F_perfect_square_l2550_255028

def best_decomposition (n : ℕ+) : ℕ+ × ℕ+ :=
  sorry

def F (n : ℕ+) : ℚ :=
  let (p, q) := best_decomposition n
  (p : ℚ) / q

theorem F_two : F 2 = 1/2 := by sorry

theorem F_perfect_square (n : ℕ+) (h : ∃ m : ℕ+, n = m * m) : F n = 1 := by sorry

end NUMINAMATH_CALUDE_F_two_F_perfect_square_l2550_255028


namespace NUMINAMATH_CALUDE_monochromatic_triangle_k6_exists_no_monochromatic_triangle_k5_l2550_255089

/-- A type representing the two colors used in the graph coloring -/
inductive Color
| Red
| Blue

/-- A type representing a complete graph with n vertices -/
def CompleteGraph (n : ℕ) := Fin n → Fin n → Color

/-- A predicate that checks if a triangle is monochromatic in a given graph -/
def HasMonochromaticTriangle (g : CompleteGraph n) : Prop :=
  ∃ (i j k : Fin n), i ≠ j ∧ j ≠ k ∧ i ≠ k ∧
    g i j = g j k ∧ g j k = g i k

theorem monochromatic_triangle_k6 :
  ∀ (g : CompleteGraph 6), HasMonochromaticTriangle g :=
sorry

theorem exists_no_monochromatic_triangle_k5 :
  ∃ (g : CompleteGraph 5), ¬HasMonochromaticTriangle g :=
sorry

end NUMINAMATH_CALUDE_monochromatic_triangle_k6_exists_no_monochromatic_triangle_k5_l2550_255089


namespace NUMINAMATH_CALUDE_sector_max_area_angle_l2550_255099

/-- Given a sector with circumference 20, the radian measure of its central angle is 2 when the area of the sector is maximized. -/
theorem sector_max_area_angle (r : ℝ) (θ : ℝ) : 
  0 < r ∧ r < 10 →  -- Ensure r is in the valid range
  r * θ + 2 * r = 20 →  -- Circumference of sector is 20
  (∀ r' θ' : ℝ, 0 < r' ∧ r' < 10 → r' * θ' + 2 * r' = 20 → 
    r * θ / 2 ≥ r' * θ' / 2) →  -- Area is maximized
  θ = 2 := by
sorry

end NUMINAMATH_CALUDE_sector_max_area_angle_l2550_255099


namespace NUMINAMATH_CALUDE_simplify_expression_1_simplify_expression_2_l2550_255011

-- Theorem 1
theorem simplify_expression_1 (a : ℝ) : a^2 - 2*a - 3*a^2 + 4*a = -2*a^2 + 2*a := by
  sorry

-- Theorem 2
theorem simplify_expression_2 (x : ℝ) : 4*(x^2 - 2) - 2*(2*x^2 + 3*x + 3) + 7*x = x - 14 := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_1_simplify_expression_2_l2550_255011


namespace NUMINAMATH_CALUDE_intersection_point_parameter_l2550_255098

/-- Given three lines that intersect at a single point but do not form a triangle, 
    prove that the parameter 'a' in one of the lines must equal -1. -/
theorem intersection_point_parameter (a : ℝ) : 
  (∃ (x y : ℝ), ax + 2*y + 8 = 0 ∧ 4*x + 3*y = 10 ∧ 2*x - y = 10) →
  (∀ (x₁ y₁ x₂ y₂ : ℝ), 
    (ax₁ + 2*y₁ + 8 = 0 ∧ 4*x₁ + 3*y₁ = 10) →
    (ax₂ + 2*y₂ + 8 = 0 ∧ 2*x₂ - y₂ = 10) →
    x₁ ≠ x₂ ∨ y₁ ≠ y₂) →
  (∀ (x₁ y₁ x₂ y₂ x₃ y₃ : ℝ),
    (ax₁ + 2*y₁ + 8 = 0 ∧ 4*x₂ + 3*y₂ = 10 ∧ 2*x₃ - y₃ = 10) →
    ¬(Set.ncard {(x₁, y₁), (x₂, y₂), (x₃, y₃)} = 3)) →
  a = -1 :=
sorry

end NUMINAMATH_CALUDE_intersection_point_parameter_l2550_255098


namespace NUMINAMATH_CALUDE_equation_solution_l2550_255035

theorem equation_solution (x : ℝ) : (x + 6) / (x - 3) = 4 → x = 6 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l2550_255035


namespace NUMINAMATH_CALUDE_set_operations_l2550_255079

-- Define the universal set U
def U : Set ℕ := {x | x < 10}

-- Define set A
def A : Set ℕ := {x ∈ U | ∃ k, x = 2 * k}

-- Define set B
def B : Set ℕ := {x | x^2 - 3*x + 2 = 0}

-- Theorem statement
theorem set_operations :
  (A ∩ B = {2}) ∧
  (A ∪ B = {0, 1, 2, 4, 6, 8}) ∧
  (U \ A = {1, 3, 5, 7, 9}) := by
  sorry

end NUMINAMATH_CALUDE_set_operations_l2550_255079


namespace NUMINAMATH_CALUDE_max_queens_on_chessboard_l2550_255047

/-- Represents a position on the chessboard -/
structure Position :=
  (row : Fin 8)
  (col : Fin 8)

/-- Checks if two positions are on the same diagonal -/
def on_same_diagonal (p1 p2 : Position) : Prop :=
  (p1.row.val : Int) - (p1.col.val : Int) = (p2.row.val : Int) - (p2.col.val : Int) ∨
  (p1.row.val : Int) + (p1.col.val : Int) = (p2.row.val : Int) + (p2.col.val : Int)

/-- Checks if two queens attack each other -/
def queens_attack (p1 p2 : Position) : Prop :=
  p1.row = p2.row ∨ p1.col = p2.col ∨ on_same_diagonal p1 p2

/-- A valid placement of queens on the chessboard -/
structure QueenPlacement :=
  (black : List Position)
  (white : List Position)
  (black_valid : black.length ≤ 8)
  (white_valid : white.length ≤ 8)
  (no_attack : ∀ b ∈ black, ∀ w ∈ white, ¬queens_attack b w)

/-- The theorem to be proved -/
theorem max_queens_on_chessboard :
  ∃ (placement : QueenPlacement), placement.black.length = 8 ∧ placement.white.length = 8 ∧
  ∀ (other : QueenPlacement), other.black.length ≤ 8 ∧ other.white.length ≤ 8 :=
sorry

end NUMINAMATH_CALUDE_max_queens_on_chessboard_l2550_255047


namespace NUMINAMATH_CALUDE_angle_A_measure_triangle_area_l2550_255017

-- Define the triangle ABC
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

-- Define the given condition
def triangle_condition (t : Triangle) : Prop :=
  (t.a - t.b) / t.c = (Real.sin t.B + Real.sin t.C) / (Real.sin t.B + Real.sin t.A)

-- Theorem 1: Prove that angle A measures 2π/3
theorem angle_A_measure (t : Triangle) (h : triangle_condition t) : t.A = 2 * Real.pi / 3 := by
  sorry

-- Theorem 2: Prove the area of the triangle when a = √7 and b = 2c
theorem triangle_area (t : Triangle) (h1 : triangle_condition t) (h2 : t.a = Real.sqrt 7) (h3 : t.b = 2 * t.c) :
  (1 / 2) * t.b * t.c * Real.sin t.A = Real.sqrt 3 / 2 := by
  sorry

end NUMINAMATH_CALUDE_angle_A_measure_triangle_area_l2550_255017


namespace NUMINAMATH_CALUDE_max_cables_theorem_l2550_255096

/-- Represents the number of employees using each brand of computer -/
structure EmployeeCount where
  total : Nat
  brandX : Nat
  brandY : Nat

/-- Represents the constraints on connections -/
structure ConnectionConstraints where
  maxConnectionPercentage : Real

/-- Calculates the maximum number of cables that can be installed -/
def maxCables (employees : EmployeeCount) (constraints : ConnectionConstraints) : Nat :=
  sorry

/-- Theorem stating the maximum number of cables that can be installed -/
theorem max_cables_theorem (employees : EmployeeCount) (constraints : ConnectionConstraints) :
  employees.total = 50 →
  employees.brandX = 30 →
  employees.brandY = 20 →
  constraints.maxConnectionPercentage = 0.95 →
  maxCables employees constraints = 300 :=
  sorry

end NUMINAMATH_CALUDE_max_cables_theorem_l2550_255096


namespace NUMINAMATH_CALUDE_smallest_n_with_conditions_l2550_255063

def digit_product (n : ℕ) : ℕ :=
  if n < 10 then n else (n % 10) * digit_product (n / 10)

theorem smallest_n_with_conditions : 
  ∀ n : ℕ, n > 0 → n % 3 = 0 → digit_product n = 882 → n ≥ 13677 := by
  sorry

#check smallest_n_with_conditions

end NUMINAMATH_CALUDE_smallest_n_with_conditions_l2550_255063


namespace NUMINAMATH_CALUDE_unique_solution_fourth_power_equation_l2550_255061

theorem unique_solution_fourth_power_equation :
  ∃! (x : ℝ), x ≠ 0 ∧ (4 * x)^5 = (8 * x)^4 ∧ x = 4 := by
  sorry

end NUMINAMATH_CALUDE_unique_solution_fourth_power_equation_l2550_255061


namespace NUMINAMATH_CALUDE_puppies_theorem_l2550_255083

/-- The number of puppies Alyssa gave away -/
def alyssa_gave : ℕ := 20

/-- The number of puppies Alyssa kept -/
def alyssa_kept : ℕ := 8

/-- The number of puppies Bella gave away -/
def bella_gave : ℕ := 10

/-- The number of puppies Bella kept -/
def bella_kept : ℕ := 6

/-- The total number of puppies Alyssa and Bella had to start with -/
def total_puppies : ℕ := alyssa_gave + alyssa_kept + bella_gave + bella_kept

theorem puppies_theorem : total_puppies = 44 := by
  sorry

end NUMINAMATH_CALUDE_puppies_theorem_l2550_255083


namespace NUMINAMATH_CALUDE_cube_surface_area_l2550_255086

/-- A point in 3D space -/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- Calculate the squared distance between two points -/
def squaredDistance (p q : Point3D) : ℝ :=
  (p.x - q.x)^2 + (p.y - q.y)^2 + (p.z - q.z)^2

/-- The vertices of the cube -/
def A : Point3D := ⟨5, 7, 15⟩
def B : Point3D := ⟨6, 3, 6⟩
def C : Point3D := ⟨9, -2, 14⟩

/-- Theorem: The surface area of the cube with vertices A, B, and C is 294 -/
theorem cube_surface_area : ∃ (side : ℝ), 
  squaredDistance A B = 2 * side^2 ∧ 
  squaredDistance A C = 2 * side^2 ∧ 
  squaredDistance B C = 2 * side^2 ∧ 
  6 * side^2 = 294 := by
  sorry


end NUMINAMATH_CALUDE_cube_surface_area_l2550_255086


namespace NUMINAMATH_CALUDE_smallest_common_rose_purchase_l2550_255031

theorem smallest_common_rose_purchase : Nat.lcm 9 19 = 171 := by
  sorry

end NUMINAMATH_CALUDE_smallest_common_rose_purchase_l2550_255031


namespace NUMINAMATH_CALUDE_cube_tetrahedrons_l2550_255018

/-- A cube has 8 vertices -/
def cube_vertices : ℕ := 8

/-- Number of vertices needed to form a tetrahedron -/
def tetrahedron_vertices : ℕ := 4

/-- Number of coplanar sets in a cube (faces + diagonal planes) -/
def coplanar_sets : ℕ := 12

/-- The number of different tetrahedrons that can be formed from the vertices of a cube -/
def num_tetrahedrons : ℕ := Nat.choose cube_vertices tetrahedron_vertices - coplanar_sets

theorem cube_tetrahedrons :
  num_tetrahedrons = Nat.choose cube_vertices tetrahedron_vertices - coplanar_sets :=
sorry

end NUMINAMATH_CALUDE_cube_tetrahedrons_l2550_255018


namespace NUMINAMATH_CALUDE_sqrt_equation_solution_l2550_255082

theorem sqrt_equation_solution : 
  ∃ x : ℚ, (Real.sqrt (8 * x) / Real.sqrt (2 * (x - 2)) = 3) ∧ (x = 18/5) := by
  sorry

end NUMINAMATH_CALUDE_sqrt_equation_solution_l2550_255082


namespace NUMINAMATH_CALUDE_bicycle_profit_percentage_l2550_255037

theorem bicycle_profit_percentage 
  (final_price : ℝ) 
  (initial_cost : ℝ) 
  (intermediate_profit_percentage : ℝ) 
  (h1 : final_price = 225)
  (h2 : initial_cost = 112.5)
  (h3 : intermediate_profit_percentage = 25) :
  let intermediate_cost := final_price / (1 + intermediate_profit_percentage / 100)
  let initial_profit_percentage := (intermediate_cost - initial_cost) / initial_cost * 100
  initial_profit_percentage = 60 := by
sorry

end NUMINAMATH_CALUDE_bicycle_profit_percentage_l2550_255037


namespace NUMINAMATH_CALUDE_sufficient_condition_range_l2550_255066

theorem sufficient_condition_range (x a : ℝ) :
  (∀ x, (|x - a| < 1 → x^2 + x - 2 > 0) ∧
   ∃ x, x^2 + x - 2 > 0 ∧ |x - a| ≥ 1) →
  a ≤ -3 ∨ a ≥ 2 :=
by sorry

end NUMINAMATH_CALUDE_sufficient_condition_range_l2550_255066


namespace NUMINAMATH_CALUDE_soccer_teams_count_l2550_255039

theorem soccer_teams_count : 
  ∃ (n : ℕ), 
    n > 0 ∧ 
    n * (n - 1) = 20 ∧ 
    n = 5 := by
  sorry

end NUMINAMATH_CALUDE_soccer_teams_count_l2550_255039


namespace NUMINAMATH_CALUDE_friends_chicken_pieces_l2550_255007

/-- Given a total number of chicken pieces, the number eaten by Lyndee, and the number of friends,
    calculate the number of pieces each friend ate. -/
def chicken_per_friend (total : ℕ) (lyndee_ate : ℕ) (num_friends : ℕ) : ℕ :=
  (total - lyndee_ate) / num_friends

theorem friends_chicken_pieces :
  chicken_per_friend 11 1 5 = 2 := by
sorry

end NUMINAMATH_CALUDE_friends_chicken_pieces_l2550_255007


namespace NUMINAMATH_CALUDE_max_product_at_endpoints_l2550_255075

/-- A quadratic function with integer coefficients -/
structure QuadraticFunction where
  m : ℤ
  n : ℤ
  f : ℝ → ℝ := λ x ↦ 10 * x^2 + m * x + n

/-- The property that a function has two distinct real roots in (1, 3) -/
def has_two_distinct_roots_in_open_interval (f : ℝ → ℝ) : Prop :=
  ∃ x y : ℝ, 1 < x ∧ x < y ∧ y < 3 ∧ f x = 0 ∧ f y = 0

/-- The theorem statement -/
theorem max_product_at_endpoints (qf : QuadraticFunction) 
  (h : has_two_distinct_roots_in_open_interval qf.f) :
  (qf.f 1) * (qf.f 3) ≤ 99 := by
  sorry

end NUMINAMATH_CALUDE_max_product_at_endpoints_l2550_255075


namespace NUMINAMATH_CALUDE_math_club_payment_l2550_255008

theorem math_club_payment (B : ℕ) : 
  B < 10 →  -- Ensure B is a single digit
  (100 + 10 * B + 8) % 9 = 0 →  -- The number 1B8 is divisible by 9
  B = 0 := by
sorry

end NUMINAMATH_CALUDE_math_club_payment_l2550_255008


namespace NUMINAMATH_CALUDE_greatest_of_three_consecutive_integers_l2550_255051

theorem greatest_of_three_consecutive_integers (x : ℤ) :
  (x + (x + 1) + (x + 2) = 18) → (max x (max (x + 1) (x + 2)) = 7) :=
by sorry

end NUMINAMATH_CALUDE_greatest_of_three_consecutive_integers_l2550_255051


namespace NUMINAMATH_CALUDE_yard_length_l2550_255009

theorem yard_length (n : ℕ) (d : ℝ) (h1 : n = 26) (h2 : d = 24) : 
  (n - 1) * d = 600 := by
  sorry

end NUMINAMATH_CALUDE_yard_length_l2550_255009


namespace NUMINAMATH_CALUDE_vector_norm_difference_l2550_255005

theorem vector_norm_difference (a b : ℝ × ℝ) 
  (h1 : a + b = (2, 3)) 
  (h2 : a - b = (-2, 1)) : 
  ‖a‖^2 - ‖b‖^2 = -1 := by sorry

end NUMINAMATH_CALUDE_vector_norm_difference_l2550_255005


namespace NUMINAMATH_CALUDE_domain_of_w_l2550_255001

-- Define the function w(y)
def w (y : ℝ) : ℝ := (y - 3) ^ (1/3) + (15 - y) ^ (1/3)

-- State the theorem about the domain of w
theorem domain_of_w :
  ∀ y : ℝ, ∃ z : ℝ, w y = z :=
sorry

end NUMINAMATH_CALUDE_domain_of_w_l2550_255001


namespace NUMINAMATH_CALUDE_sum_of_three_numbers_l2550_255090

theorem sum_of_three_numbers (a b c : ℝ) 
  (h1 : a^2 + b^2 + c^2 = 241) 
  (h2 : a*b + b*c + c*a = 100) : 
  a + b + c = 21 := by
sorry

end NUMINAMATH_CALUDE_sum_of_three_numbers_l2550_255090


namespace NUMINAMATH_CALUDE_all_numbers_in_first_hundred_l2550_255080

/-- Represents the color of a number in the sequence -/
inductive Color
| Blue
| Red

/-- Represents the sequence of 200 numbers with their colors -/
def Sequence := Fin 200 → (ℕ × Color)

/-- The blue numbers form a sequence from 1 to 100 in ascending order -/
def blue_ascending (s : Sequence) : Prop :=
  ∀ i j : Fin 200, i < j →
    (s i).2 = Color.Blue → (s j).2 = Color.Blue →
      (s i).1 < (s j).1

/-- The red numbers form a sequence from 100 to 1 in descending order -/
def red_descending (s : Sequence) : Prop :=
  ∀ i j : Fin 200, i < j →
    (s i).2 = Color.Red → (s j).2 = Color.Red →
      (s i).1 > (s j).1

/-- The blue numbers are all natural numbers from 1 to 100 -/
def blue_range (s : Sequence) : Prop :=
  ∀ n : ℕ, 1 ≤ n ∧ n ≤ 100 →
    ∃ i : Fin 200, (s i).1 = n ∧ (s i).2 = Color.Blue

/-- The red numbers are all natural numbers from 1 to 100 -/
def red_range (s : Sequence) : Prop :=
  ∀ n : ℕ, 1 ≤ n ∧ n ≤ 100 →
    ∃ i : Fin 200, (s i).1 = n ∧ (s i).2 = Color.Red

theorem all_numbers_in_first_hundred (s : Sequence)
  (h1 : blue_ascending s)
  (h2 : red_descending s)
  (h3 : blue_range s)
  (h4 : red_range s) :
  ∀ n : ℕ, 1 ≤ n ∧ n ≤ 100 →
    ∃ i : Fin 100, (s i).1 = n :=
by sorry

end NUMINAMATH_CALUDE_all_numbers_in_first_hundred_l2550_255080


namespace NUMINAMATH_CALUDE_old_record_calculation_old_record_proof_l2550_255085

/-- Calculates the old record given James' performance and the points he beat the record by -/
theorem old_record_calculation (touchdowns_per_game : ℕ) (points_per_touchdown : ℕ) 
  (games_in_season : ℕ) (two_point_conversions : ℕ) (points_beat_record_by : ℕ) : ℕ :=
  let james_total_points := touchdowns_per_game * points_per_touchdown * games_in_season + 
                            two_point_conversions * 2
  james_total_points - points_beat_record_by

/-- Proves that the old record was 300 points given James' performance -/
theorem old_record_proof :
  old_record_calculation 4 6 15 6 72 = 300 := by
  sorry

end NUMINAMATH_CALUDE_old_record_calculation_old_record_proof_l2550_255085


namespace NUMINAMATH_CALUDE_quadratic_minimum_l2550_255056

theorem quadratic_minimum (b : ℝ) :
  let f : ℝ → ℝ := fun x ↦ 3 * x^2 - 18 * x + b
  ∃ (min_value : ℝ), (∀ x, f x ≥ min_value) ∧ (f 3 = min_value) ∧ (min_value = b - 27) := by
  sorry

end NUMINAMATH_CALUDE_quadratic_minimum_l2550_255056


namespace NUMINAMATH_CALUDE_principal_is_20000_l2550_255014

/-- Calculates the principal amount given the interest rate, time, and total interest -/
def calculate_principal (rate : ℚ) (time : ℕ) (interest : ℚ) : ℚ :=
  (interest * 100) / (rate * time)

/-- Theorem stating that given the specified conditions, the principal amount is 20000 -/
theorem principal_is_20000 : 
  let rate : ℚ := 12
  let time : ℕ := 3
  let interest : ℚ := 7200
  calculate_principal rate time interest = 20000 := by
  sorry

end NUMINAMATH_CALUDE_principal_is_20000_l2550_255014


namespace NUMINAMATH_CALUDE_expected_vote_percentage_a_l2550_255036

/-- Percentage of registered voters who are Democrats -/
def democrat_percentage : ℝ := 70

/-- Percentage of registered voters who are Republicans -/
def republican_percentage : ℝ := 100 - democrat_percentage

/-- Percentage of Democrats expected to vote for candidate A -/
def democrat_vote_a : ℝ := 80

/-- Percentage of Republicans expected to vote for candidate A -/
def republican_vote_a : ℝ := 30

/-- Theorem stating the percentage of registered voters expected to vote for candidate A -/
theorem expected_vote_percentage_a : 
  (democrat_percentage / 100 * democrat_vote_a + 
   republican_percentage / 100 * republican_vote_a) = 65 := by
  sorry

end NUMINAMATH_CALUDE_expected_vote_percentage_a_l2550_255036


namespace NUMINAMATH_CALUDE_quadratic_equation_from_means_l2550_255038

theorem quadratic_equation_from_means (a b : ℝ) : 
  (a + b) / 2 = 8 → 
  Real.sqrt (a * b) = 12 → 
  ∀ x, x^2 - 16*x + 144 = 0 ↔ (x = a ∨ x = b) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_equation_from_means_l2550_255038


namespace NUMINAMATH_CALUDE_inscribed_polygon_existence_l2550_255053

/-- Represents a line in a plane --/
structure Line where
  -- Add necessary fields for a line

/-- Represents a circle in a plane --/
structure Circle where
  -- Add necessary fields for a circle

/-- Represents a polygon --/
structure Polygon where
  -- Add necessary fields for a polygon

/-- Function to check if two lines are parallel --/
def are_parallel (l1 l2 : Line) : Prop :=
  sorry

/-- Function to check if a polygon is inscribed in a circle --/
def is_inscribed (p : Polygon) (c : Circle) : Prop :=
  sorry

/-- Function to check if a polygon side is parallel to a given line --/
def side_parallel_to_line (p : Polygon) (l : Line) : Prop :=
  sorry

/-- Main theorem statement --/
theorem inscribed_polygon_existence 
  (c : Circle) (lines : List Line) (n : Nat) 
  (h1 : lines.length = n)
  (h2 : ∀ (l1 l2 : Line), l1 ∈ lines → l2 ∈ lines → l1 ≠ l2 → ¬(are_parallel l1 l2)) :
  (n % 2 = 0 → 
    (∃ (p : Polygon), is_inscribed p c ∧ 
      (∀ (side : Line), side_parallel_to_line p side → side ∈ lines)) ∨
    (∀ (p : Polygon), ¬(is_inscribed p c ∧ 
      (∀ (side : Line), side_parallel_to_line p side → side ∈ lines)))) ∧
  (n % 2 = 1 → 
    ∃! (p1 p2 : Polygon), p1 ≠ p2 ∧ 
      is_inscribed p1 c ∧ is_inscribed p2 c ∧
      (∀ (side : Line), side_parallel_to_line p1 side → side ∈ lines) ∧
      (∀ (side : Line), side_parallel_to_line p2 side → side ∈ lines)) :=
sorry

end NUMINAMATH_CALUDE_inscribed_polygon_existence_l2550_255053


namespace NUMINAMATH_CALUDE_no_prime_satisfies_equation_l2550_255067

theorem no_prime_satisfies_equation : 
  ¬ ∃ (p : ℕ), Nat.Prime p ∧ 
  (2 * p^3 + 0 * p^2 + 3 * p + 4) + 
  (4 * p^2 + 0 * p + 5) + 
  (2 * p^2 + 1 * p + 7) + 
  (1 * p^2 + 5 * p + 0) + 
  4 = 
  (3 * p^2 + 0 * p + 2) + 
  (5 * p^2 + 2 * p + 0) + 
  (4 * p^2 + 3 * p + 1) :=
sorry

end NUMINAMATH_CALUDE_no_prime_satisfies_equation_l2550_255067


namespace NUMINAMATH_CALUDE_prism_faces_count_l2550_255042

/-- A prism with n sides, where n is at least 3 --/
structure Prism (n : ℕ) where
  sides : n ≥ 3

/-- The number of faces in a prism --/
def num_faces (n : ℕ) (p : Prism n) : ℕ := n + 2

theorem prism_faces_count (n : ℕ) (p : Prism n) : 
  num_faces n p ≠ n :=
sorry

end NUMINAMATH_CALUDE_prism_faces_count_l2550_255042


namespace NUMINAMATH_CALUDE_pyramid_missing_number_l2550_255059

/-- Represents a row in the pyramid -/
structure PyramidRow :=
  (left : ℚ) (middle : ℚ) (right : ℚ)

/-- Represents the pyramid structure -/
structure Pyramid :=
  (top_row : PyramidRow)
  (middle_row : PyramidRow)
  (bottom_row : PyramidRow)

/-- Checks if the pyramid satisfies the product rule -/
def is_valid_pyramid (p : Pyramid) : Prop :=
  p.middle_row.left = p.top_row.left * p.top_row.middle ∧
  p.middle_row.middle = p.top_row.middle * p.top_row.right ∧
  p.bottom_row.left = p.middle_row.left * p.middle_row.middle ∧
  p.bottom_row.middle = p.middle_row.middle * p.middle_row.right

/-- The main theorem -/
theorem pyramid_missing_number :
  ∀ (p : Pyramid),
    is_valid_pyramid p →
    p.middle_row = ⟨3, 2, 5⟩ →
    p.bottom_row.left = 10 →
    p.top_row.middle = 5/3 :=
by sorry

end NUMINAMATH_CALUDE_pyramid_missing_number_l2550_255059


namespace NUMINAMATH_CALUDE_sqrt_500_simplification_l2550_255023

theorem sqrt_500_simplification : Real.sqrt 500 = 10 * Real.sqrt 5 := by sorry

end NUMINAMATH_CALUDE_sqrt_500_simplification_l2550_255023


namespace NUMINAMATH_CALUDE_quadratic_roots_abs_less_than_one_l2550_255046

theorem quadratic_roots_abs_less_than_one (a b : ℝ) 
  (h1 : abs a + abs b < 1) 
  (h2 : a^2 - 4*b ≥ 0) : 
  ∀ x, x^2 + a*x + b = 0 → abs x < 1 := by
sorry

end NUMINAMATH_CALUDE_quadratic_roots_abs_less_than_one_l2550_255046


namespace NUMINAMATH_CALUDE_sci_fi_readers_l2550_255003

theorem sci_fi_readers (total : ℕ) (literary : ℕ) (both : ℕ) (sci_fi : ℕ) : 
  total = 250 → literary = 88 → both = 18 → sci_fi = total + both - literary :=
by
  sorry

end NUMINAMATH_CALUDE_sci_fi_readers_l2550_255003


namespace NUMINAMATH_CALUDE_parabola_intersection_value_l2550_255097

-- Define the parabola
def parabola (x : ℝ) : ℝ := x^2 - x - 1

-- Define the theorem
theorem parabola_intersection_value :
  ∀ m : ℝ, parabola m = 0 → -2 * m^2 + 2 * m + 2023 = 2021 := by
  sorry

end NUMINAMATH_CALUDE_parabola_intersection_value_l2550_255097


namespace NUMINAMATH_CALUDE_largest_n_for_product_1764_l2550_255084

/-- Two arithmetic sequences with integer terms -/
def ArithmeticSequence (a : ℕ → ℤ) : Prop :=
  ∃ d : ℤ, ∀ n : ℕ, a (n + 1) = a n + d

theorem largest_n_for_product_1764 
  (a b : ℕ → ℤ) 
  (ha : ArithmeticSequence a) 
  (hb : ArithmeticSequence b) 
  (h_a1 : a 1 = 1) 
  (h_b1 : b 1 = 1) 
  (h_a2_le_b2 : a 2 ≤ b 2) 
  (h_product : ∃ n : ℕ, a n * b n = 1764) :
  (∀ m : ℕ, (∃ k : ℕ, a k * b k = 1764) → m ≤ 44) ∧ 
  (∃ n : ℕ, a n * b n = 1764 ∧ n = 44) :=
sorry

end NUMINAMATH_CALUDE_largest_n_for_product_1764_l2550_255084


namespace NUMINAMATH_CALUDE_eggs_per_student_l2550_255000

theorem eggs_per_student (total_eggs : ℕ) (num_students : ℕ) (eggs_per_student : ℕ)
  (h1 : total_eggs = 56)
  (h2 : num_students = 7)
  (h3 : total_eggs = num_students * eggs_per_student) :
  eggs_per_student = 8 := by
  sorry

end NUMINAMATH_CALUDE_eggs_per_student_l2550_255000


namespace NUMINAMATH_CALUDE_water_level_rise_l2550_255049

/-- The rise in water level when a sphere is fully immersed in a rectangular vessel --/
theorem water_level_rise (sphere_radius : ℝ) (vessel_length : ℝ) (vessel_width : ℝ) :
  sphere_radius = 10 →
  vessel_length = 30 →
  vessel_width = 25 →
  ∃ (water_rise : ℝ), abs (water_rise - 5.59) < 0.01 :=
by
  sorry

end NUMINAMATH_CALUDE_water_level_rise_l2550_255049


namespace NUMINAMATH_CALUDE_white_triangle_pairs_coincide_l2550_255072

/-- Represents the number of triangles of each color in each half -/
structure TriangleCounts where
  red : ℕ
  blue : ℕ
  white : ℕ

/-- Represents the number of coinciding pairs of triangles -/
structure CoincidingPairs where
  red_red : ℕ
  blue_blue : ℕ
  red_white : ℕ
  blue_white : ℕ

/-- Theorem stating that given the conditions in the problem, 
    the number of coinciding white triangle pairs is 3 -/
theorem white_triangle_pairs_coincide 
  (counts : TriangleCounts)
  (pairs : CoincidingPairs)
  (h1 : counts.red = 4)
  (h2 : counts.blue = 6)
  (h3 : counts.white = 9)
  (h4 : pairs.red_red = 3)
  (h5 : pairs.blue_blue = 4)
  (h6 : pairs.red_white = 3)
  (h7 : pairs.blue_white = 3) :
  ∃ (white_white_pairs : ℕ), white_white_pairs = 3 :=
sorry

end NUMINAMATH_CALUDE_white_triangle_pairs_coincide_l2550_255072


namespace NUMINAMATH_CALUDE_triangle_area_l2550_255013

theorem triangle_area (A B C : EuclideanSpace ℝ (Fin 2)) :
  let b : ℝ := 1
  let c : ℝ := Real.sqrt 3
  let angle_A : ℝ := π / 4
  let area : ℝ := (1 / 2) * b * c * Real.sin angle_A
  area = Real.sqrt 6 / 4 := by
sorry

end NUMINAMATH_CALUDE_triangle_area_l2550_255013


namespace NUMINAMATH_CALUDE_card_purchase_cost_l2550_255068

/-- Calculates the total cost of cards purchased, including discounts and tax --/
def totalCost (typeA_price typeB_price typeC_price typeD_price : ℚ)
               (typeA_count typeB_count typeC_count typeD_count : ℕ)
               (discount_AB discount_CD : ℚ)
               (min_count_AB min_count_CD : ℕ)
               (tax_rate : ℚ) : ℚ :=
  let subtotal := typeA_price * typeA_count + typeB_price * typeB_count +
                  typeC_price * typeC_count + typeD_price * typeD_count
  let discount_amount := 
    (if typeA_count ≥ min_count_AB ∧ typeB_count ≥ min_count_AB then
      discount_AB * (typeA_price * typeA_count + typeB_price * typeB_count)
    else 0) +
    (if typeC_count ≥ min_count_CD ∧ typeD_count ≥ min_count_CD then
      discount_CD * (typeC_price * typeC_count + typeD_price * typeD_count)
    else 0)
  let discounted_total := subtotal - discount_amount
  let tax := tax_rate * discounted_total
  discounted_total + tax

/-- The total cost of cards is $60.82 given the specified conditions --/
theorem card_purchase_cost : 
  totalCost 1.25 1.50 2.25 2.50  -- Card prices
            6 4 10 12            -- Number of cards purchased
            0.1 0.15             -- Discount rates
            5 8                  -- Minimum count for discounts
            0.06                 -- Tax rate
  = 60.82 := by
  sorry

end NUMINAMATH_CALUDE_card_purchase_cost_l2550_255068


namespace NUMINAMATH_CALUDE_midpoint_coordinate_ratio_range_l2550_255024

/-- Given two parallel lines and a point between them, prove the ratio of its coordinates is within a specific range. -/
theorem midpoint_coordinate_ratio_range 
  (P : ℝ × ℝ) (Q : ℝ × ℝ) (x₀ : ℝ) (y₀ : ℝ)
  (hP : P.1 + 2 * P.2 - 1 = 0)
  (hQ : Q.1 + 2 * Q.2 + 3 = 0)
  (hM : (x₀, y₀) = ((P.1 + Q.1) / 2, (P.2 + Q.2) / 2))
  (h_ineq : y₀ > x₀ + 2)
  : -1/2 < y₀ / x₀ ∧ y₀ / x₀ < -1/5 :=
sorry

end NUMINAMATH_CALUDE_midpoint_coordinate_ratio_range_l2550_255024


namespace NUMINAMATH_CALUDE_square_plot_area_l2550_255070

/-- Proves that a square plot with given fencing costs has an area of 36 square feet -/
theorem square_plot_area (cost_per_foot : ℝ) (total_cost : ℝ) :
  cost_per_foot = 58 →
  total_cost = 1392 →
  ∃ (side_length : ℝ),
    side_length > 0 ∧
    4 * side_length * cost_per_foot = total_cost ∧
    side_length ^ 2 = 36 := by
  sorry

#check square_plot_area

end NUMINAMATH_CALUDE_square_plot_area_l2550_255070


namespace NUMINAMATH_CALUDE_coordinate_system_change_l2550_255034

/-- Represents a point in a 2D coordinate system -/
structure Point where
  x : ℝ
  y : ℝ

/-- The relative position of two points -/
def relativePosition (p q : Point) : Point :=
  ⟨p.x - q.x, p.y - q.y⟩

theorem coordinate_system_change (A B : Point) :
  relativePosition A B = ⟨2, 5⟩ → relativePosition B A = ⟨-2, -5⟩ := by
  sorry

end NUMINAMATH_CALUDE_coordinate_system_change_l2550_255034


namespace NUMINAMATH_CALUDE_product_inequality_l2550_255052

theorem product_inequality (a b c d : ℝ) 
  (ha : 1 ≤ a ∧ a ≤ 2) 
  (hb : 1 ≤ b ∧ b ≤ 2) 
  (hc : 1 ≤ c ∧ c ≤ 2) 
  (hd : 1 ≤ d ∧ d ≤ 2) : 
  |((a - b) * (b - c) * (c - d) * (d - a))| ≤ (a * b * c * d) / 4 := by
sorry

end NUMINAMATH_CALUDE_product_inequality_l2550_255052


namespace NUMINAMATH_CALUDE_largest_consecutive_even_l2550_255022

theorem largest_consecutive_even : 
  ∀ (x : ℕ), 
  (x + (x + 2) + (x + 4) + (x + 6) + (x + 8) + (x + 10) + (x + 12) + (x + 14) = 424) → 
  (x + 14 = 60) := by
sorry

end NUMINAMATH_CALUDE_largest_consecutive_even_l2550_255022


namespace NUMINAMATH_CALUDE_box_volume_l2550_255033

/-- Given a rectangular box with dimensions L, W, and H satisfying certain conditions,
    prove that its volume is 5184. -/
theorem box_volume (L W H : ℝ) (h1 : H * W = 288) (h2 : L * W = 1.5 * 288) 
    (h3 : L * H = 0.5 * (L * W)) : L * W * H = 5184 := by
  sorry

end NUMINAMATH_CALUDE_box_volume_l2550_255033


namespace NUMINAMATH_CALUDE_white_shirts_count_l2550_255060

/-- The number of white t-shirts in each pack -/
def white_shirts_per_pack : ℕ := sorry

/-- The number of packs of white t-shirts bought -/
def white_packs : ℕ := 5

/-- The number of packs of blue t-shirts bought -/
def blue_packs : ℕ := 3

/-- The number of blue t-shirts in each pack -/
def blue_shirts_per_pack : ℕ := 9

/-- The total number of t-shirts bought -/
def total_shirts : ℕ := 57

theorem white_shirts_count : white_shirts_per_pack = 6 :=
  by sorry

end NUMINAMATH_CALUDE_white_shirts_count_l2550_255060


namespace NUMINAMATH_CALUDE_kubosdivision_l2550_255045

theorem kubosdivision (k m : ℕ) (hk : k > 0) (hm : m > 0) (hkm : k > m) 
  (hdiv : (k^3 - m^3) ∣ (k * m * (k^2 - m^2))) : (k - m)^3 > 3 * k * m := by
  sorry

end NUMINAMATH_CALUDE_kubosdivision_l2550_255045


namespace NUMINAMATH_CALUDE_convention_handshakes_correct_l2550_255065

/-- Represents the number of handshakes at a twins and triplets convention -/
def convention_handshakes (twin_sets : ℕ) (triplet_sets : ℕ) : ℕ :=
  let twins := twin_sets * 2
  let triplets := triplet_sets * 3
  let twin_handshakes := twins * (twins - 2)
  let triplet_handshakes := triplets * (triplets - 3)
  let cross_handshakes := twins * (triplets / 2) * 2
  (twin_handshakes + triplet_handshakes + cross_handshakes) / 2

theorem convention_handshakes_correct :
  convention_handshakes 9 6 = 441 := by sorry

end NUMINAMATH_CALUDE_convention_handshakes_correct_l2550_255065


namespace NUMINAMATH_CALUDE_max_knight_moves_5x6_l2550_255077

/-- Represents a chess board --/
structure ChessBoard :=
  (rows : Nat)
  (cols : Nat)

/-- Represents a knight's move type --/
inductive MoveType
  | Normal
  | Short

/-- Represents a sequence of knight moves --/
def MoveSequence := List MoveType

/-- Checks if a move sequence is valid (alternating between Normal and Short, starting with Normal) --/
def isValidMoveSequence : MoveSequence → Bool
  | [] => true
  | [MoveType.Normal] => true
  | (MoveType.Normal :: MoveType.Short :: rest) => isValidMoveSequence rest
  | _ => false

/-- The maximum number of moves a knight can make on the given board --/
def maxKnightMoves (board : ChessBoard) (seq : MoveSequence) : Nat :=
  seq.length

/-- The main theorem to prove --/
theorem max_knight_moves_5x6 :
  ∀ (seq : MoveSequence),
    isValidMoveSequence seq →
    maxKnightMoves ⟨5, 6⟩ seq ≤ 24 :=
by sorry

end NUMINAMATH_CALUDE_max_knight_moves_5x6_l2550_255077


namespace NUMINAMATH_CALUDE_exam_day_percentage_l2550_255002

/-- Proves that 70% of students took the exam on the assigned day given the conditions of the problem -/
theorem exam_day_percentage :
  ∀ (x : ℝ),
  (x ≥ 0) →
  (x ≤ 100) →
  (0.6 * x + 0.9 * (100 - x) = 69) →
  x = 70 := by
  sorry

end NUMINAMATH_CALUDE_exam_day_percentage_l2550_255002


namespace NUMINAMATH_CALUDE_C_on_or_inside_circle_O_l2550_255004

-- Define the circle O and points A, B, C
variable (O : ℝ × ℝ) (A B C : ℝ × ℝ)

-- Define the radius of circle O
def radius_O : ℝ := 10

-- Define that A is on circle O
def A_on_circle_O : (A.1 - O.1)^2 + (A.2 - O.2)^2 = radius_O^2 := by sorry

-- Define B as the midpoint of OA
def B_midpoint_OA : B = ((O.1 + A.1)/2, (O.2 + A.2)/2) := by sorry

-- Define the distance between B and C
def BC_distance : (B.1 - C.1)^2 + (B.2 - C.2)^2 = 5^2 := by sorry

-- Theorem to prove
theorem C_on_or_inside_circle_O :
  (C.1 - O.1)^2 + (C.2 - O.2)^2 ≤ radius_O^2 := by sorry

end NUMINAMATH_CALUDE_C_on_or_inside_circle_O_l2550_255004


namespace NUMINAMATH_CALUDE_sarah_initial_followers_l2550_255081

/-- Represents the number of followers gained by Sarah in a week -/
structure WeeklyGain where
  first : ℕ
  second : ℕ
  third : ℕ

/-- Represents the data for a student's social media followers -/
structure StudentData where
  school_size : ℕ
  initial_followers : ℕ
  weekly_gain : WeeklyGain

theorem sarah_initial_followers (susy sarah : StudentData) 
  (h1 : susy.school_size = 800)
  (h2 : sarah.school_size = 300)
  (h3 : susy.initial_followers = 100)
  (h4 : sarah.weekly_gain.first = 90)
  (h5 : sarah.weekly_gain.second = sarah.weekly_gain.first / 3)
  (h6 : sarah.weekly_gain.third = sarah.weekly_gain.second / 3)
  (h7 : max (susy.initial_followers + susy.weekly_gain.first + susy.weekly_gain.second + susy.weekly_gain.third)
            (sarah.initial_followers + sarah.weekly_gain.first + sarah.weekly_gain.second + sarah.weekly_gain.third) = 180) :
  sarah.initial_followers = 50 := by
  sorry


end NUMINAMATH_CALUDE_sarah_initial_followers_l2550_255081


namespace NUMINAMATH_CALUDE_equations_equivalence_l2550_255073

-- Define the equations
def equation1 (x : ℝ) : Prop := (-x - 2) / (x - 3) = (x + 1) / (x - 3)
def equation2 (x : ℝ) : Prop := -x - 2 = x + 1
def equation3 (x : ℝ) : Prop := (-x - 2) * (x - 3) = (x + 1) * (x - 3)

-- Theorem statement
theorem equations_equivalence :
  (∀ x : ℝ, x ≠ 3 → (equation1 x ↔ equation2 x)) ∧
  (¬ ∀ x : ℝ, equation2 x ↔ equation3 x) ∧
  (¬ ∀ x : ℝ, equation1 x ↔ equation3 x) :=
sorry

end NUMINAMATH_CALUDE_equations_equivalence_l2550_255073


namespace NUMINAMATH_CALUDE_f_neg_two_value_l2550_255030

def F (f : ℝ → ℝ) (x : ℝ) : ℝ := f x + x^2

theorem f_neg_two_value (f : ℝ → ℝ) :
  (∀ x, F f x = -F f (-x)) →  -- F is an odd function
  f 2 = 1 →                   -- f(2) = 1
  f (-2) = -9 :=               -- Conclusion: f(-2) = -9
by
  sorry

end NUMINAMATH_CALUDE_f_neg_two_value_l2550_255030


namespace NUMINAMATH_CALUDE_remainder_scaling_l2550_255071

theorem remainder_scaling (a b : ℕ) (c r : ℕ) (h : a = b * c + r) (hr : r = 7) :
  ∃ (c' : ℕ), 10 * a = 10 * b * c' + 70 :=
sorry

end NUMINAMATH_CALUDE_remainder_scaling_l2550_255071


namespace NUMINAMATH_CALUDE_unique_valid_number_l2550_255012

def is_valid_number (n : ℕ) : Prop :=
  765400 ≤ n ∧ n ≤ 765499 ∧ n % 24 = 0

theorem unique_valid_number :
  ∃! n : ℕ, is_valid_number n ∧ n = 765455 :=
sorry

end NUMINAMATH_CALUDE_unique_valid_number_l2550_255012


namespace NUMINAMATH_CALUDE_average_marks_proof_l2550_255029

/-- Given the marks for three subjects, proves that the average is 65 -/
theorem average_marks_proof (physics chemistry maths : ℝ) : 
  physics = 125 → 
  (physics + maths) / 2 = 90 → 
  (physics + chemistry) / 2 = 70 → 
  (physics + chemistry + maths) / 3 = 65 := by
sorry


end NUMINAMATH_CALUDE_average_marks_proof_l2550_255029


namespace NUMINAMATH_CALUDE_inequality_system_solution_l2550_255095

theorem inequality_system_solution (x : ℝ) : 
  (2 * x - 1 > 0 ∧ 3 * x > 2 * x + 2) ↔ x > 2 := by sorry

end NUMINAMATH_CALUDE_inequality_system_solution_l2550_255095


namespace NUMINAMATH_CALUDE_paintball_cost_per_box_l2550_255050

/-- Calculates the cost per box of paintballs -/
def cost_per_box (plays_per_month : ℕ) (boxes_per_play : ℕ) (total_monthly_cost : ℕ) : ℚ :=
  total_monthly_cost / (plays_per_month * boxes_per_play)

/-- Theorem: Given the problem conditions, the cost per box of paintballs is $25 -/
theorem paintball_cost_per_box :
  let plays_per_month : ℕ := 3
  let boxes_per_play : ℕ := 3
  let total_monthly_cost : ℕ := 225
  cost_per_box plays_per_month boxes_per_play total_monthly_cost = 25 := by
  sorry


end NUMINAMATH_CALUDE_paintball_cost_per_box_l2550_255050


namespace NUMINAMATH_CALUDE_k_range_theorem_l2550_255069

-- Define the function f(x) = (e^x / x) + x^2 - 2x
noncomputable def f (x : ℝ) : ℝ := (Real.exp x / x) + x^2 - 2*x

theorem k_range_theorem (k : ℝ) : 
  (∀ x ∈ Set.Ioo 0 2, x / Real.exp x < 1 / (k + 2*x - x^2)) → 
  k ∈ Set.Icc 0 (Real.exp 1 - 1) :=
sorry

end NUMINAMATH_CALUDE_k_range_theorem_l2550_255069


namespace NUMINAMATH_CALUDE_international_shipping_charge_l2550_255087

/-- The additional charge per letter for international shipping -/
def additional_charge (standard_postage : ℚ) (total_letters : ℕ) (international_letters : ℕ) (total_cost : ℚ) : ℚ :=
  ((total_cost - (standard_postage * total_letters)) / international_letters) * 100

theorem international_shipping_charge :
  let standard_postage : ℚ := 108 / 100  -- $1.08 in decimal form
  let total_letters : ℕ := 4
  let international_letters : ℕ := 2
  let total_cost : ℚ := 460 / 100  -- $4.60 in decimal form
  additional_charge standard_postage total_letters international_letters total_cost = 14 := by
  sorry

#eval additional_charge (108/100) 4 2 (460/100)

end NUMINAMATH_CALUDE_international_shipping_charge_l2550_255087


namespace NUMINAMATH_CALUDE_unique_positive_solution_l2550_255040

theorem unique_positive_solution (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a > b) :
  ∃! x : ℝ, x > 0 ∧ 
    (2 * (a + b) * x + 2 * a * b) / (4 * x + a + b) = ((a^(1/3) + b^(1/3)) / 2)^3 := by
  sorry

end NUMINAMATH_CALUDE_unique_positive_solution_l2550_255040


namespace NUMINAMATH_CALUDE_vector_BC_l2550_255025

/-- Given points A(0,1), B(3,2), and vector AC(-4,-3), prove that vector BC is (-7,-4) -/
theorem vector_BC (A B C : ℝ × ℝ) : 
  A = (0, 1) → 
  B = (3, 2) → 
  C - A = (-4, -3) → 
  C - B = (-7, -4) := by
sorry

end NUMINAMATH_CALUDE_vector_BC_l2550_255025


namespace NUMINAMATH_CALUDE_vector_angle_difference_l2550_255078

theorem vector_angle_difference (α β : Real) (a b : Real × Real) 
  (h1 : 0 < α) (h2 : α < β) (h3 : β < π)
  (h4 : a = (Real.cos α, Real.sin α))
  (h5 : b = (Real.cos β, Real.sin β))
  (h6 : ‖3 • a + b‖ = ‖a - 2 • b‖) : 
  β - α = 2 * π / 3 := by
  sorry

end NUMINAMATH_CALUDE_vector_angle_difference_l2550_255078


namespace NUMINAMATH_CALUDE_pigeonhole_on_permutation_sums_l2550_255076

theorem pigeonhole_on_permutation_sums (n : ℕ) (h_even : Even n) (h_pos : 0 < n)
  (A B : Fin n → Fin n) (h_A : Function.Bijective A) (h_B : Function.Bijective B) :
  ∃ (i j : Fin n), i ≠ j ∧ (A i + B i) % n = (A j + B j) % n := by
sorry

end NUMINAMATH_CALUDE_pigeonhole_on_permutation_sums_l2550_255076


namespace NUMINAMATH_CALUDE_function_difference_bound_l2550_255044

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a^x - x * Real.log a

theorem function_difference_bound
  (a : ℝ) (ha : 1 < a ∧ a ≤ Real.exp 1) :
  ∀ x₁ x₂ : ℝ, x₁ ∈ Set.Icc (-1) 1 → x₂ ∈ Set.Icc (-1) 1 →
  |f a x₁ - f a x₂| ≤ Real.exp 1 - 2 :=
sorry

end NUMINAMATH_CALUDE_function_difference_bound_l2550_255044


namespace NUMINAMATH_CALUDE_positive_number_equation_solution_l2550_255020

theorem positive_number_equation_solution :
  ∃ (x : ℝ), x > 0 ∧ Real.sqrt ((10 * x) / 3) = x ∧ x = 10 / 3 := by
  sorry

end NUMINAMATH_CALUDE_positive_number_equation_solution_l2550_255020


namespace NUMINAMATH_CALUDE_common_chord_equation_l2550_255032

/-- Given two circles with equations x^2 + y^2 - 4x = 0 and x^2 + y^2 - 4y = 0,
    the equation of the line containing their common chord is x - y = 0 -/
theorem common_chord_equation (x y : ℝ) :
  (x^2 + y^2 - 4*x = 0) ∧ (x^2 + y^2 - 4*y = 0) → x - y = 0 := by
  sorry

end NUMINAMATH_CALUDE_common_chord_equation_l2550_255032


namespace NUMINAMATH_CALUDE_log_sum_equation_l2550_255088

theorem log_sum_equation (p q : ℝ) (hp : p > 0) (hq : q > 0) :
  Real.log p + Real.log q = Real.log (p + q + p * q) → p = -q :=
by sorry

end NUMINAMATH_CALUDE_log_sum_equation_l2550_255088


namespace NUMINAMATH_CALUDE_intersection_of_A_and_B_l2550_255055

def set_A : Set ℝ := {x | |x - 2| ≤ 1}
def set_B : Set ℝ := {x | (x - 5) / (2 - x) > 0}

theorem intersection_of_A_and_B :
  ∀ x : ℝ, x ∈ set_A ∩ set_B ↔ 2 < x ∧ x ≤ 3 := by sorry

end NUMINAMATH_CALUDE_intersection_of_A_and_B_l2550_255055


namespace NUMINAMATH_CALUDE_unique_paths_in_grid_l2550_255054

/-- The number of rows in the grid -/
def rows : ℕ := 6

/-- The number of columns in the grid -/
def cols : ℕ := 6

/-- The number of moves required to reach the bottom-right corner -/
def moves : ℕ := 5

/-- A function that calculates the number of unique paths in the grid -/
def uniquePaths (r : ℕ) (c : ℕ) (m : ℕ) : ℕ := 2^m

/-- Theorem stating that the number of unique paths in the given grid is 32 -/
theorem unique_paths_in_grid : uniquePaths rows cols moves = 32 := by
  sorry

end NUMINAMATH_CALUDE_unique_paths_in_grid_l2550_255054


namespace NUMINAMATH_CALUDE_median_equation_equal_intercepts_equation_l2550_255062

-- Define the vertices of triangle ABC
def A : ℝ × ℝ := (-2, 4)
def B : ℝ × ℝ := (4, -6)
def C : ℝ × ℝ := (5, 1)

-- Define the equation of a line
def is_line_equation (a b c : ℝ) (x y : ℝ) : Prop :=
  a * x + b * y + c = 0

-- Theorem for the median equation
theorem median_equation :
  is_line_equation 1 (-2) (-3) (A.1 + (B.1 - A.1)/2) (A.2 + (B.2 - A.2)/2) ∧
  is_line_equation 1 (-2) (-3) C.1 C.2 :=
sorry

-- Theorem for the line with equal intercepts
theorem equal_intercepts_equation :
  is_line_equation 1 1 (-2) A.1 A.2 ∧
  ∃ (t : ℝ), is_line_equation 1 1 (-2) t 0 ∧ is_line_equation 1 1 (-2) 0 t :=
sorry

end NUMINAMATH_CALUDE_median_equation_equal_intercepts_equation_l2550_255062


namespace NUMINAMATH_CALUDE_light_path_in_cube_l2550_255048

/-- Represents a point on the face of a cube -/
structure FacePoint where
  x : ℝ
  y : ℝ

/-- Represents the path of a light beam in a cube -/
def LightPath (cube_side : ℝ) (p : FacePoint) : ℝ := sorry

theorem light_path_in_cube (cube_side : ℝ) (p : FacePoint) 
  (h_side : cube_side = 10)
  (h_p : p = ⟨3, 4⟩) :
  ∃ (r s : ℕ), LightPath cube_side p = r * Real.sqrt s ∧ r + s = 55 ∧ 
  ∀ (prime : ℕ), Nat.Prime prime → ¬(s.gcd (prime ^ 2) > 1) := by
  sorry

end NUMINAMATH_CALUDE_light_path_in_cube_l2550_255048


namespace NUMINAMATH_CALUDE_max_integer_k_no_real_roots_l2550_255019

theorem max_integer_k_no_real_roots (k : ℤ) : 
  (∀ x : ℝ, x^2 - 2*x - k ≠ 0) → k ≤ -2 ∧ ∀ m : ℤ, (∀ x : ℝ, x^2 - 2*x - m ≠ 0) → m ≤ k :=
by sorry

end NUMINAMATH_CALUDE_max_integer_k_no_real_roots_l2550_255019


namespace NUMINAMATH_CALUDE_average_divisible_by_seven_l2550_255006

theorem average_divisible_by_seven : ∃ (S : Finset ℕ), 
  (∀ n ∈ S, 12 < n ∧ n < 150 ∧ n % 7 = 0) ∧ 
  (∀ n, 12 < n → n < 150 → n % 7 = 0 → n ∈ S) ∧
  (S.sum id / S.card : ℚ) = 161/2 := by
  sorry

end NUMINAMATH_CALUDE_average_divisible_by_seven_l2550_255006


namespace NUMINAMATH_CALUDE_reciprocal_fraction_l2550_255026

theorem reciprocal_fraction (x : ℝ) (y : ℝ) (h1 : x > 0) (h2 : x = 1) 
  (h3 : (2/3) * x = y * (1/x)) : y = 2/3 := by
  sorry

end NUMINAMATH_CALUDE_reciprocal_fraction_l2550_255026


namespace NUMINAMATH_CALUDE_suggestion_difference_l2550_255021

def student_count : ℕ := 1200

def food_suggestions : List ℕ := [408, 305, 137, 213, 137]

theorem suggestion_difference : 
  (List.maximum food_suggestions).get! - (List.minimum food_suggestions).get! = 271 := by
  sorry

end NUMINAMATH_CALUDE_suggestion_difference_l2550_255021


namespace NUMINAMATH_CALUDE_second_year_undeclared_fraction_l2550_255027

theorem second_year_undeclared_fraction :
  let total : ℚ := 1
  let first_year : ℚ := 1/4
  let second_year : ℚ := 1/2
  let third_year : ℚ := 1/6
  let fourth_year : ℚ := 1/12
  let first_year_undeclared : ℚ := 4/5
  let second_year_undeclared : ℚ := 3/4
  let third_year_undeclared : ℚ := 1/3
  let fourth_year_undeclared : ℚ := 1/6
  
  first_year + second_year + third_year + fourth_year = total →
  second_year * second_year_undeclared = 1/3
  := by sorry

end NUMINAMATH_CALUDE_second_year_undeclared_fraction_l2550_255027


namespace NUMINAMATH_CALUDE_negation_equivalence_l2550_255041

theorem negation_equivalence :
  (¬ ∀ x : ℝ, ∃ n : ℕ+, (n : ℝ) ≥ x^2) ↔ (∃ x : ℝ, ∀ n : ℕ+, (n : ℝ) < x^2) :=
by sorry

end NUMINAMATH_CALUDE_negation_equivalence_l2550_255041

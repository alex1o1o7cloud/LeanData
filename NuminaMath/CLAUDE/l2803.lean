import Mathlib

namespace NUMINAMATH_CALUDE_married_men_fraction_l2803_280361

structure Gathering where
  total_women : ℕ
  single_women : ℕ
  married_women : ℕ
  married_men : ℕ

def Gathering.total_people (g : Gathering) : ℕ :=
  g.total_women + g.married_men

def Gathering.prob_single_woman (g : Gathering) : ℚ :=
  g.single_women / g.total_women

def Gathering.fraction_married_men (g : Gathering) : ℚ :=
  g.married_men / g.total_people

theorem married_men_fraction (g : Gathering) 
  (h1 : g.married_women = g.married_men)
  (h2 : g.total_women = g.single_women + g.married_women)
  (h3 : g.prob_single_woman = 1/4) :
  g.fraction_married_men = 3/7 := by
sorry

end NUMINAMATH_CALUDE_married_men_fraction_l2803_280361


namespace NUMINAMATH_CALUDE_constant_sequence_is_ap_and_gp_l2803_280348

def constant_sequence : ℕ → ℝ := λ n => 7

def is_arithmetic_progression (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

def is_geometric_progression (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, r ≠ 0 ∧ ∀ n : ℕ, a (n + 1) = a n * r

theorem constant_sequence_is_ap_and_gp :
  is_arithmetic_progression constant_sequence ∧
  is_geometric_progression constant_sequence := by
  sorry

#check constant_sequence_is_ap_and_gp

end NUMINAMATH_CALUDE_constant_sequence_is_ap_and_gp_l2803_280348


namespace NUMINAMATH_CALUDE_woodchopper_theorem_l2803_280324

/-- A woodchopper who gets a certain number of wood blocks per tree and chops a certain number of trees per day -/
structure Woodchopper where
  blocks_per_tree : ℕ
  trees_per_day : ℕ

/-- Calculate the total number of wood blocks obtained after a given number of days -/
def total_blocks (w : Woodchopper) (days : ℕ) : ℕ :=
  w.blocks_per_tree * w.trees_per_day * days

/-- Theorem: A woodchopper who gets 3 blocks per tree and chops 2 trees per day obtains 30 blocks after 5 days -/
theorem woodchopper_theorem :
  let ragnar : Woodchopper := { blocks_per_tree := 3, trees_per_day := 2 }
  total_blocks ragnar 5 = 30 := by sorry

end NUMINAMATH_CALUDE_woodchopper_theorem_l2803_280324


namespace NUMINAMATH_CALUDE_distance_point_to_line_l2803_280360

/-- The distance from the point (√2, -√2) to the line x + y = 1 is √2/2 -/
theorem distance_point_to_line : 
  let point : ℝ × ℝ := (Real.sqrt 2, -Real.sqrt 2)
  let line (x y : ℝ) : Prop := x + y = 1
  abs (point.1 + point.2 - 1) / Real.sqrt 2 = Real.sqrt 2 / 2 := by sorry

end NUMINAMATH_CALUDE_distance_point_to_line_l2803_280360


namespace NUMINAMATH_CALUDE_perpendicular_line_x_intercept_l2803_280368

/-- Given a line L1: 4x - 3y = 12, prove that a line L2 perpendicular to L1 
    with y-intercept 3 has x-intercept 4 -/
theorem perpendicular_line_x_intercept :
  let L1 : ℝ → ℝ → Prop := λ x y => 4 * x - 3 * y = 12
  let m1 : ℝ := 4 / 3  -- slope of L1
  let m2 : ℝ := -3 / 4  -- slope of L2 (perpendicular to L1)
  let L2 : ℝ → ℝ → Prop := λ x y => y = m2 * x + 3  -- L2 with y-intercept 3
  ∃ x : ℝ, x = 4 ∧ L2 x 0 :=
by
  sorry

#check perpendicular_line_x_intercept

end NUMINAMATH_CALUDE_perpendicular_line_x_intercept_l2803_280368


namespace NUMINAMATH_CALUDE_projection_scalar_multiple_l2803_280391

def proj_w (v : ℝ × ℝ) (w : ℝ × ℝ) : ℝ × ℝ := sorry

theorem projection_scalar_multiple (v w : ℝ × ℝ) :
  proj_w v w = (4, 3) → proj_w (7 • v) w = (28, 21) := by
  sorry

end NUMINAMATH_CALUDE_projection_scalar_multiple_l2803_280391


namespace NUMINAMATH_CALUDE_win_sector_area_l2803_280310

/-- Proves that for a circular spinner with given radius and winning probabilities,
    the combined area of winning sectors is as calculated. -/
theorem win_sector_area (r : ℝ) (p : ℝ) (h1 : r = 15) (h2 : p = 1/6) :
  2 * p * π * r^2 = 75 * π := by
  sorry

end NUMINAMATH_CALUDE_win_sector_area_l2803_280310


namespace NUMINAMATH_CALUDE_slope_of_CD_is_one_l2803_280364

/-- Given a line y = kx (k > 0) passing through the origin and intersecting the curve y = e^(x-1)
    at two distinct points A(x₁, y₁) and B(x₂, y₂) where x₁ > 0 and x₂ > 0, and points C(x₁, ln x₁)
    and D(x₂, ln x₂) on the curve y = ln x, prove that the slope of line CD is 1. -/
theorem slope_of_CD_is_one (k x₁ x₂ : ℝ) (hk : k > 0) (hx₁ : x₁ > 0) (hx₂ : x₂ > 0)
  (hy₁ : k * x₁ = Real.exp (x₁ - 1)) (hy₂ : k * x₂ = Real.exp (x₂ - 1)) :
  (Real.log x₂ - Real.log x₁) / (x₂ - x₁) = 1 :=
by sorry

end NUMINAMATH_CALUDE_slope_of_CD_is_one_l2803_280364


namespace NUMINAMATH_CALUDE_line_ellipse_intersection_range_l2803_280381

/-- The range of b values for which the line y = kx + b always has two common points with the ellipse x²/9 + y²/4 = 1 -/
theorem line_ellipse_intersection_range :
  ∀ (k : ℝ), 
  (∀ (b : ℝ), (∃! (x₁ y₁ x₂ y₂ : ℝ), 
    x₁ ≠ x₂ ∧ 
    y₁ = k * x₁ + b ∧ 
    y₂ = k * x₂ + b ∧ 
    x₁^2 / 9 + y₁^2 / 4 = 1 ∧ 
    x₂^2 / 9 + y₂^2 / 4 = 1)) ↔ 
  (-2 < b ∧ b < 2) :=
sorry

end NUMINAMATH_CALUDE_line_ellipse_intersection_range_l2803_280381


namespace NUMINAMATH_CALUDE_g_of_x_plus_3_l2803_280369

/-- Given a function g(x) = (x^2 + 3x) / 2, prove that g(x+3) = (x^2 + 9x + 18) / 2 for all real x -/
theorem g_of_x_plus_3 (x : ℝ) : 
  let g : ℝ → ℝ := λ x ↦ (x^2 + 3*x) / 2
  g (x + 3) = (x^2 + 9*x + 18) / 2 := by
sorry

end NUMINAMATH_CALUDE_g_of_x_plus_3_l2803_280369


namespace NUMINAMATH_CALUDE_ellipse_chord_y_diff_l2803_280330

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents an ellipse -/
structure Ellipse where
  a : ℝ
  b : ℝ

/-- Theorem: For the given ellipse, if a chord AB passes through the left focus and the
    inscribed circle of triangle ABF₂ has circumference π, then |y₁ - y₂| = 5/4 -/
theorem ellipse_chord_y_diff (e : Ellipse) (A B : Point) (F₁ F₂ : Point) : 
  e.a = 5 → 
  e.b = 4 → 
  F₁.x = -3 → 
  F₁.y = 0 → 
  F₂.x = 3 → 
  F₂.y = 0 → 
  (A.x - F₁.x) * (B.y - F₁.y) = (B.x - F₁.x) * (A.y - F₁.y) →  -- AB passes through F₁
  2 * π * (A.x * (B.y - F₂.y) + B.x * (F₂.y - A.y) + F₂.x * (A.y - B.y)) / 
    (A.x * (B.y - F₂.y) + B.x * (F₂.y - A.y) + F₂.x * (A.y - B.y) + 
     (A.x - F₂.x) * (B.y - F₂.y) - (B.x - F₂.x) * (A.y - F₂.y)) = π →  -- Inscribed circle circumference
  |A.y - B.y| = 5/4 := by
sorry


end NUMINAMATH_CALUDE_ellipse_chord_y_diff_l2803_280330


namespace NUMINAMATH_CALUDE_ali_age_difference_l2803_280397

/-- Given the ages of Ali and Umar, and the relationship between Umar and Yusaf's ages,
    prove that Ali is 3 years older than Yusaf. -/
theorem ali_age_difference (ali_age umar_age : ℕ) (h1 : ali_age = 8) (h2 : umar_age = 10)
  (h3 : umar_age = 2 * (umar_age / 2)) : ali_age - (umar_age / 2) = 3 := by
  sorry

end NUMINAMATH_CALUDE_ali_age_difference_l2803_280397


namespace NUMINAMATH_CALUDE_max_markable_nodes_6x6_l2803_280308

/-- Represents a square grid -/
structure Grid :=
  (size : Nat)

/-- A node in the grid -/
structure Node :=
  (x : Nat)
  (y : Nat)

/-- Checks if a node is on the edge of the grid -/
def isEdgeNode (g : Grid) (n : Node) : Bool :=
  n.x = 0 || n.x = g.size || n.y = 0 || n.y = g.size

/-- Checks if a node is a corner node -/
def isCornerNode (g : Grid) (n : Node) : Bool :=
  (n.x = 0 || n.x = g.size) && (n.y = 0 || n.y = g.size)

/-- Counts the number of nodes in a grid -/
def nodeCount (g : Grid) : Nat :=
  (g.size + 1) * (g.size + 1)

/-- Theorem: The maximum number of markable nodes in a 6x6 grid is 45 -/
theorem max_markable_nodes_6x6 (g : Grid) (h : g.size = 6) :
  nodeCount g - (4 : Nat) = 45 := by
  sorry

#check max_markable_nodes_6x6

end NUMINAMATH_CALUDE_max_markable_nodes_6x6_l2803_280308


namespace NUMINAMATH_CALUDE_smallest_batch_size_l2803_280379

theorem smallest_batch_size (N : ℕ) (h1 : N > 70) (h2 : (21 * N) % 70 = 0) :
  N ≥ 80 ∧ ∀ m : ℕ, m > 70 ∧ (21 * m) % 70 = 0 → m ≥ N := by
  sorry

end NUMINAMATH_CALUDE_smallest_batch_size_l2803_280379


namespace NUMINAMATH_CALUDE_max_value_expression_l2803_280398

theorem max_value_expression (x y : ℝ) : 
  ∃ (M : ℝ), M = 24 - 2 * Real.sqrt 7 ∧ 
  ∀ (a b : ℝ), a ≤ M ∧ 
  (∃ (x y : ℝ), a = (Real.sqrt (9 - Real.sqrt 7) * Real.sin x - Real.sqrt (2 * (1 + Real.cos (2 * x))) - 1) * 
                   (3 + 2 * Real.sqrt (13 - Real.sqrt 7) * Real.cos y - Real.cos (2 * y))) :=
by sorry

end NUMINAMATH_CALUDE_max_value_expression_l2803_280398


namespace NUMINAMATH_CALUDE_parallel_lines_k_value_l2803_280338

/-- Two lines are parallel if their slopes are equal -/
def parallel (m₁ m₂ : ℝ) : Prop := m₁ = m₂

/-- The first line has equation y = 3x + 5 -/
def line1 : ℝ → ℝ := λ x => 3 * x + 5

/-- The second line has equation y = (6k)x + 1 -/
def line2 (k : ℝ) : ℝ → ℝ := λ x => 6 * k * x + 1

theorem parallel_lines_k_value :
  ∀ k : ℝ, parallel (line1 0 - line1 1) (line2 k 0 - line2 k 1) → k = 1/2 := by
sorry

end NUMINAMATH_CALUDE_parallel_lines_k_value_l2803_280338


namespace NUMINAMATH_CALUDE_number_ratio_l2803_280390

theorem number_ratio (f s t : ℝ) : 
  s = 4 * f →
  (f + s + t) / 3 = 77 →
  f = 33 →
  f ≤ s ∧ f ≤ t →
  t / f = 2 := by
sorry

end NUMINAMATH_CALUDE_number_ratio_l2803_280390


namespace NUMINAMATH_CALUDE_product_expansion_l2803_280352

theorem product_expansion (x : ℝ) : 
  (3*x - 4) * (2*x^2 + 3*x - 1) = 6*x^3 + x^2 - 15*x + 4 := by
  sorry

end NUMINAMATH_CALUDE_product_expansion_l2803_280352


namespace NUMINAMATH_CALUDE_vending_machine_probability_l2803_280366

/-- Represents the vending machine scenario --/
structure VendingMachine where
  numToys : Nat
  priceStep : Rat
  minPrice : Rat
  maxPrice : Rat
  numFavoriteToys : Nat
  favoriteToyPrice : Rat
  initialQuarters : Nat

/-- Calculates the probability of needing to exchange the $20 bill --/
def probabilityNeedExchange (vm : VendingMachine) : Rat :=
  sorry

/-- The main theorem to prove --/
theorem vending_machine_probability (vm : VendingMachine) :
  vm.numToys = 10 ∧
  vm.priceStep = 1/2 ∧
  vm.minPrice = 1/2 ∧
  vm.maxPrice = 5 ∧
  vm.numFavoriteToys = 2 ∧
  vm.favoriteToyPrice = 9/2 ∧
  vm.initialQuarters = 12
  →
  probabilityNeedExchange vm = 15/25 :=
by sorry

end NUMINAMATH_CALUDE_vending_machine_probability_l2803_280366


namespace NUMINAMATH_CALUDE_not_arithmetic_sequence_l2803_280371

theorem not_arithmetic_sequence : ¬∃ (m n k : ℤ) (a d : ℝ), 
  m < n ∧ n < k ∧ 
  1 = a + (m - 1) * d ∧ 
  Real.sqrt 3 = a + (n - 1) * d ∧ 
  2 = a + (k - 1) * d :=
sorry

end NUMINAMATH_CALUDE_not_arithmetic_sequence_l2803_280371


namespace NUMINAMATH_CALUDE_ellipse_eccentricity_l2803_280392

/-- Ellipse C with foci F₁ and F₂, and point P on C -/
structure Ellipse :=
  (a b : ℝ)
  (h_ab : a > b ∧ b > 0)
  (F₁ F₂ P : ℝ × ℝ)
  (h_on_ellipse : (P.1^2 / a^2) + (P.2^2 / b^2) = 1)
  (h_perp : (P.1 - F₂.1) * (F₂.1 - F₁.1) + (P.2 - F₂.2) * (F₂.2 - F₁.2) = 0)
  (h_angle : Real.cos (30 * π / 180) = 
    ((P.1 - F₁.1) * (F₂.1 - F₁.1) + (P.2 - F₁.2) * (F₂.2 - F₁.2)) / 
    (Real.sqrt ((P.1 - F₁.1)^2 + (P.2 - F₁.2)^2) * Real.sqrt ((F₂.1 - F₁.1)^2 + (F₂.2 - F₁.2)^2)))

/-- The eccentricity of an ellipse is √3/3 -/
theorem ellipse_eccentricity (C : Ellipse) : 
  Real.sqrt ((C.F₂.1 - C.F₁.1)^2 + (C.F₂.2 - C.F₁.2)^2) / (2 * C.a) = Real.sqrt 3 / 3 := by
  sorry

end NUMINAMATH_CALUDE_ellipse_eccentricity_l2803_280392


namespace NUMINAMATH_CALUDE_existence_of_squares_with_difference_2023_l2803_280301

theorem existence_of_squares_with_difference_2023 :
  ∃ (x y : ℕ), x > 0 ∧ y > 0 ∧ x^2 = y^2 + 2023 ∧
  ((x = 1012 ∧ y = 1011) ∨ (x = 148 ∧ y = 141) ∨ (x = 68 ∧ y = 51)) :=
by sorry

end NUMINAMATH_CALUDE_existence_of_squares_with_difference_2023_l2803_280301


namespace NUMINAMATH_CALUDE_interest_calculation_l2803_280305

/-- Represents the problem of finding the minimum number of years for a specific interest calculation. -/
theorem interest_calculation (principal1 principal2 rate1 rate2 target_interest : ℚ) :
  principal1 = 800 →
  principal2 = 1400 →
  rate1 = 3 / 100 →
  rate2 = 5 / 100 →
  target_interest = 350 →
  (∃ (n : ℕ), (principal1 * rate1 * n + principal2 * rate2 * n ≥ target_interest) ∧
    (∀ (m : ℕ), m < n → principal1 * rate1 * m + principal2 * rate2 * m < target_interest)) →
  (∃ (n : ℕ), (principal1 * rate1 * n + principal2 * rate2 * n ≥ target_interest) ∧
    (∀ (m : ℕ), m < n → principal1 * rate1 * m + principal2 * rate2 * m < target_interest) ∧
    n = 4) :=
by sorry

end NUMINAMATH_CALUDE_interest_calculation_l2803_280305


namespace NUMINAMATH_CALUDE_problem_grid_triangles_l2803_280322

/-- Represents a triangular grid with a given number of rows -/
structure TriangularGrid where
  rows : ℕ

/-- Calculates the total number of triangles in a triangular grid -/
def totalTriangles (grid : TriangularGrid) : ℕ :=
  sorry

/-- The specific triangular grid described in the problem -/
def problemGrid : TriangularGrid :=
  { rows := 4 }

theorem problem_grid_triangles :
  totalTriangles problemGrid = 18 := by
  sorry

end NUMINAMATH_CALUDE_problem_grid_triangles_l2803_280322


namespace NUMINAMATH_CALUDE_function_periodicity_l2803_280346

-- Define the function f
variable (f : ℝ → ℝ)

-- Define the functional equation
def functionalEquation (f : ℝ → ℝ) : Prop :=
  ∀ x y : ℝ, f (x + y) + f (x - y) = 2 * f x * f y

-- Define the existence of c
def existsC (f : ℝ → ℝ) : Prop :=
  ∃ c : ℝ, c > 0 ∧ f (c / 2) = 0

-- Define periodicity
def isPeriodic (f : ℝ → ℝ) (T : ℝ) : Prop :=
  ∀ x : ℝ, f (x + T) = f x

-- Theorem statement
theorem function_periodicity (f : ℝ → ℝ) 
  (h1 : functionalEquation f) 
  (h2 : existsC f) :
  ∃ T : ℝ, T > 0 ∧ isPeriodic f T :=
sorry

end NUMINAMATH_CALUDE_function_periodicity_l2803_280346


namespace NUMINAMATH_CALUDE_last_name_length_proof_l2803_280395

/-- Given information about the lengths of last names, prove the length of another person's last name --/
theorem last_name_length_proof (samantha_length bobbie_length other_length : ℕ) : 
  samantha_length = 7 →
  bobbie_length = samantha_length + 3 →
  bobbie_length - 2 = 2 * other_length →
  other_length = 4 := by
  sorry

end NUMINAMATH_CALUDE_last_name_length_proof_l2803_280395


namespace NUMINAMATH_CALUDE_profit_percentage_calculation_l2803_280389

theorem profit_percentage_calculation (cost_price selling_price : ℝ) 
  (h1 : cost_price = 800)
  (h2 : selling_price = 1080) :
  (selling_price - cost_price) / cost_price * 100 = 35 := by
  sorry

end NUMINAMATH_CALUDE_profit_percentage_calculation_l2803_280389


namespace NUMINAMATH_CALUDE_min_value_x_plus_y_l2803_280325

theorem min_value_x_plus_y (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : 2*x - x*y + 6*y = 0) :
  ∀ z w : ℝ, z > 0 ∧ w > 0 ∧ 2*z - z*w + 6*w = 0 → x + y ≤ z + w ∧ x + y = 8 + 4 * Real.sqrt 3 :=
by sorry

end NUMINAMATH_CALUDE_min_value_x_plus_y_l2803_280325


namespace NUMINAMATH_CALUDE_max_b_no_lattice_points_b_max_is_maximum_l2803_280344

/-- Represents a lattice point with integer coordinates -/
structure LatticePoint where
  x : Int
  y : Int

/-- Checks if a given point lies on the line y = mx + 3 -/
def lies_on_line (m : ℚ) (p : LatticePoint) : Prop :=
  p.y = m * p.x + 3

/-- The maximum value of b we want to prove -/
def b_max : ℚ := 76 / 151

theorem max_b_no_lattice_points :
  ∀ m : ℚ, 1/2 < m → m < b_max →
    ∀ x : ℤ, 0 < x → x ≤ 150 →
      ¬∃ p : LatticePoint, p.x = x ∧ lies_on_line m p :=
sorry

theorem b_max_is_maximum :
  ∀ b : ℚ, b > b_max →
    ∃ m : ℚ, 1/2 < m ∧ m < b ∧
      ∃ x : ℤ, 0 < x ∧ x ≤ 150 ∧
        ∃ p : LatticePoint, p.x = x ∧ lies_on_line m p :=
sorry

end NUMINAMATH_CALUDE_max_b_no_lattice_points_b_max_is_maximum_l2803_280344


namespace NUMINAMATH_CALUDE_concentration_a_is_45_percent_l2803_280307

/-- The concentration of spirit in vessel a -/
def concentration_a : ℝ := 45

/-- The concentration of spirit in vessel b -/
def concentration_b : ℝ := 30

/-- The concentration of spirit in vessel c -/
def concentration_c : ℝ := 10

/-- The volume taken from vessel a -/
def volume_a : ℝ := 4

/-- The volume taken from vessel b -/
def volume_b : ℝ := 5

/-- The volume taken from vessel c -/
def volume_c : ℝ := 6

/-- The concentration of spirit in the resultant solution -/
def concentration_result : ℝ := 26

/-- Theorem stating that the concentration of spirit in vessel a is 45% -/
theorem concentration_a_is_45_percent :
  (volume_a * concentration_a / 100 + 
   volume_b * concentration_b / 100 + 
   volume_c * concentration_c / 100) / 
  (volume_a + volume_b + volume_c) * 100 = concentration_result :=
by sorry

end NUMINAMATH_CALUDE_concentration_a_is_45_percent_l2803_280307


namespace NUMINAMATH_CALUDE_min_sum_reciprocals_l2803_280353

theorem min_sum_reciprocals (n : ℕ+) (a b : ℝ) 
  (ha : a > 0) (hb : b > 0) (hab : a + b = 2) : 
  (1 / (1 + a ^ n.val)) + (1 / (1 + b ^ n.val)) ≥ 1 ∧ 
  ((1 / (1 + a ^ n.val)) + (1 / (1 + b ^ n.val)) = 1 ↔ a = 1 ∧ b = 1) :=
by sorry

end NUMINAMATH_CALUDE_min_sum_reciprocals_l2803_280353


namespace NUMINAMATH_CALUDE_expression_value_l2803_280349

theorem expression_value : 
  (10^2005 + 10^2007) / (10^2006 + 10^2006) = 101 / 20 := by
  sorry

end NUMINAMATH_CALUDE_expression_value_l2803_280349


namespace NUMINAMATH_CALUDE_simplify_fraction_l2803_280377

theorem simplify_fraction : (130 : ℚ) / 16900 * 65 = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_simplify_fraction_l2803_280377


namespace NUMINAMATH_CALUDE_balls_in_box_l2803_280380

theorem balls_in_box (initial_balls : ℕ) (balls_taken : ℕ) (balls_left : ℕ) : 
  initial_balls = 10 → balls_taken = 3 → balls_left = initial_balls - balls_taken → balls_left = 7 := by
  sorry

end NUMINAMATH_CALUDE_balls_in_box_l2803_280380


namespace NUMINAMATH_CALUDE_remainder_two_power_1000_mod_17_l2803_280367

theorem remainder_two_power_1000_mod_17 : 2^1000 % 17 = 1 := by
  sorry

end NUMINAMATH_CALUDE_remainder_two_power_1000_mod_17_l2803_280367


namespace NUMINAMATH_CALUDE_saree_price_calculation_l2803_280317

theorem saree_price_calculation (final_price : ℝ) 
  (h1 : final_price = 224) 
  (discount1 : ℝ) (h2 : discount1 = 0.3)
  (discount2 : ℝ) (h3 : discount2 = 0.2) : 
  ∃ (original_price : ℝ), 
    original_price = 400 ∧ 
    final_price = original_price * (1 - discount1) * (1 - discount2) :=
by
  sorry

end NUMINAMATH_CALUDE_saree_price_calculation_l2803_280317


namespace NUMINAMATH_CALUDE_one_and_one_third_problem_l2803_280356

theorem one_and_one_third_problem : ∃ x : ℚ, (4/3) * x = 36 ∧ x = 27 := by
  sorry

end NUMINAMATH_CALUDE_one_and_one_third_problem_l2803_280356


namespace NUMINAMATH_CALUDE_u_diff_divisible_by_factorial_l2803_280340

/-- The sequence u_k defined recursively -/
def u (a : ℕ+) : ℕ → ℕ
  | 0 => 1
  | k + 1 => a ^ (u a k)

/-- Theorem stating that n! divides u_{n+1} - u_n for all n ≥ 1 -/
theorem u_diff_divisible_by_factorial (a : ℕ+) (n : ℕ) (h : n ≥ 1) :
  (n.factorial : ℤ) ∣ (u a (n + 1) : ℤ) - (u a n : ℤ) := by
  sorry

end NUMINAMATH_CALUDE_u_diff_divisible_by_factorial_l2803_280340


namespace NUMINAMATH_CALUDE_factorial_simplification_l2803_280342

theorem factorial_simplification : (12 : ℕ).factorial / ((10 : ℕ).factorial + 3 * (9 : ℕ).factorial) = 1320 / 13 := by
  sorry

end NUMINAMATH_CALUDE_factorial_simplification_l2803_280342


namespace NUMINAMATH_CALUDE_add_fractions_with_same_denominator_l2803_280350

theorem add_fractions_with_same_denominator (a : ℝ) (h : a ≠ 0) :
  3 / a + 2 / a = 5 / a := by sorry

end NUMINAMATH_CALUDE_add_fractions_with_same_denominator_l2803_280350


namespace NUMINAMATH_CALUDE_angle_range_in_special_triangle_l2803_280331

/-- Given a triangle ABC with sides a, b, c opposite to angles A, B, C respectively,
    if a + b = 2c, then 0 < C ≤ π/3 -/
theorem angle_range_in_special_triangle (a b c : ℝ) (A B C : ℝ) :
  a > 0 → b > 0 → c > 0 →
  A > 0 → B > 0 → C > 0 →
  A + B + C = π →
  a + b = 2 * c →
  0 < C ∧ C ≤ π / 3 := by
  sorry


end NUMINAMATH_CALUDE_angle_range_in_special_triangle_l2803_280331


namespace NUMINAMATH_CALUDE_symmetry_and_rotation_sum_l2803_280329

/-- The number of sides in our regular polygon -/
def n : ℕ := 17

/-- The number of lines of symmetry in a regular n-gon -/
def L (n : ℕ) : ℕ := n

/-- The smallest positive angle (in degrees) for which a regular n-gon has rotational symmetry -/
def R (n : ℕ) : ℚ := 360 / n

/-- Theorem: For a regular 17-gon, the sum of its number of lines of symmetry 
    and its smallest positive angle of rotational symmetry (in degrees) 
    is equal to 17 + 360/17 -/
theorem symmetry_and_rotation_sum : 
  (L n : ℚ) + R n = 17 + 360 / 17 := by
  sorry

end NUMINAMATH_CALUDE_symmetry_and_rotation_sum_l2803_280329


namespace NUMINAMATH_CALUDE_lcm_16_35_l2803_280358

theorem lcm_16_35 : Nat.lcm 16 35 = 560 := by
  sorry

end NUMINAMATH_CALUDE_lcm_16_35_l2803_280358


namespace NUMINAMATH_CALUDE_quadratic_specific_value_l2803_280378

def quadratic (a b c : ℝ) (x : ℝ) : ℝ := a * x^2 + b * x + c

theorem quadratic_specific_value (a b c : ℝ) :
  (∃ (f : ℝ → ℝ), f = quadratic a b c) →
  (∀ x, quadratic a b c x ≥ -4) →
  (quadratic a b c (-5) = -4) →
  (quadratic a b c 0 = 6) →
  (quadratic a b c (-3) = -2.4) :=
sorry

end NUMINAMATH_CALUDE_quadratic_specific_value_l2803_280378


namespace NUMINAMATH_CALUDE_comparison_of_b_and_c_l2803_280343

theorem comparison_of_b_and_c (a b c : ℝ) 
  (h1 : 2*a^3 - b^3 + 2*c^3 - 6*a^2*b + 3*a*b^2 - 3*a*c^2 - 3*b*c^2 + 6*a*b*c = 0)
  (h2 : a < b) : 
  b < c ∧ c < 2*b - a := by
  sorry

end NUMINAMATH_CALUDE_comparison_of_b_and_c_l2803_280343


namespace NUMINAMATH_CALUDE_ellipse_properties_l2803_280314

/-- Definition of the ellipse -/
def is_ellipse (x y : ℝ) : Prop := 16 * x^2 + 25 * y^2 = 400

/-- Theorem about the properties of the ellipse -/
theorem ellipse_properties :
  ∃ (a b c : ℝ),
    a = 5 ∧ b = 4 ∧ c = 3 ∧
    (∀ x y, is_ellipse x y →
      (2 * a = 10 ∧ b = 4) ∧
      (is_ellipse (-c) 0 ∧ is_ellipse c 0) ∧
      (is_ellipse (-a) 0 ∧ is_ellipse a 0 ∧ is_ellipse 0 b ∧ is_ellipse 0 (-b))) :=
by sorry

end NUMINAMATH_CALUDE_ellipse_properties_l2803_280314


namespace NUMINAMATH_CALUDE_senior_ticket_price_l2803_280385

/-- Proves that the price of senior citizen tickets is $10 -/
theorem senior_ticket_price
  (total_tickets : ℕ)
  (regular_price : ℕ)
  (total_sales : ℕ)
  (regular_tickets : ℕ)
  (h1 : total_tickets = 65)
  (h2 : regular_price = 15)
  (h3 : total_sales = 855)
  (h4 : regular_tickets = 41)
  : (total_sales - regular_tickets * regular_price) / (total_tickets - regular_tickets) = 10 := by
  sorry

end NUMINAMATH_CALUDE_senior_ticket_price_l2803_280385


namespace NUMINAMATH_CALUDE_problem_statement_l2803_280319

open Real

theorem problem_statement :
  (¬ (∀ x : ℝ, sin x ≠ 1) ↔ (∃ x : ℝ, sin x = 1)) ∧
  ((∀ α : ℝ, α = π/6 → sin α = 1/2) ∧ ¬(∀ α : ℝ, sin α = 1/2 → α = π/6)) ∧
  ¬(∀ a : ℕ → ℝ, (∀ n : ℕ, a (n+1) = 3 * a n) ↔ (∃ r : ℝ, ∀ n : ℕ, a (n+1) = r * a n)) :=
by
  sorry


end NUMINAMATH_CALUDE_problem_statement_l2803_280319


namespace NUMINAMATH_CALUDE_min_sum_inverse_squares_l2803_280384

/-- Given two circles with equations x^2 + y^2 + 2ax + a^2 - 4 = 0 and x^2 + y^2 - 4by - 1 + 4b^2 = 0,
    where a and b are real numbers, ab ≠ 0, and the circles have exactly three common tangent lines,
    prove that the minimum value of 1/a^2 + 1/b^2 is 1. -/
theorem min_sum_inverse_squares (a b : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) 
    (h_tangent : ∃ (t1 t2 t3 : ℝ × ℝ), 
      t1 ≠ t2 ∧ t1 ≠ t3 ∧ t2 ≠ t3 ∧
      (∀ (x y : ℝ), (x^2 + y^2 + 2*a*x + a^2 - 4 = 0 ∨ x^2 + y^2 - 4*b*y - 1 + 4*b^2 = 0) →
        ((x - t1.1)^2 + (y - t1.2)^2 = 0 ∨
         (x - t2.1)^2 + (y - t2.2)^2 = 0 ∨
         (x - t3.1)^2 + (y - t3.2)^2 = 0))) :
  (∀ c d : ℝ, c ≠ 0 → d ≠ 0 → 
    (∃ (t1' t2' t3' : ℝ × ℝ), 
      t1' ≠ t2' ∧ t1' ≠ t3' ∧ t2' ≠ t3' ∧
      (∀ (x y : ℝ), (x^2 + y^2 + 2*c*x + c^2 - 4 = 0 ∨ x^2 + y^2 - 4*d*y - 1 + 4*d^2 = 0) →
        ((x - t1'.1)^2 + (y - t1'.2)^2 = 0 ∨
         (x - t2'.1)^2 + (y - t2'.2)^2 = 0 ∨
         (x - t3'.1)^2 + (y - t3'.2)^2 = 0))) →
    1 / c^2 + 1 / d^2 ≥ 1) ∧
  (1 / a^2 + 1 / b^2 = 1) := by
sorry

end NUMINAMATH_CALUDE_min_sum_inverse_squares_l2803_280384


namespace NUMINAMATH_CALUDE_triangle_ratio_bound_l2803_280363

theorem triangle_ratio_bound (a b c : ℝ) (h : 0 < a ∧ 0 < b ∧ 0 < c ∧ a + b > c ∧ b + c > a ∧ c + a > b) :
  (a^2 + b^2 + c^2) / (a*b + b*c + c*a) ≤ 1 :=
sorry

end NUMINAMATH_CALUDE_triangle_ratio_bound_l2803_280363


namespace NUMINAMATH_CALUDE_smallest_factorial_divisible_by_2016_smallest_factorial_divisible_by_2016_power_10_l2803_280334

def factorial (n : ℕ) : ℕ := (List.range n).foldl (·*·) 1

theorem smallest_factorial_divisible_by_2016 :
  ∀ n : ℕ, n < 8 → ¬(2016 ∣ factorial n) ∧ (2016 ∣ factorial 8) :=
sorry

theorem smallest_factorial_divisible_by_2016_power_10 :
  ∀ n : ℕ, n < 63 → ¬(2016^10 ∣ factorial n) ∧ (2016^10 ∣ factorial 63) :=
sorry

end NUMINAMATH_CALUDE_smallest_factorial_divisible_by_2016_smallest_factorial_divisible_by_2016_power_10_l2803_280334


namespace NUMINAMATH_CALUDE_mary_flour_calculation_l2803_280359

/-- The amount of flour Mary has already put in the cake -/
def flour_put_in : ℕ := sorry

/-- The total amount of flour required by the recipe -/
def total_flour_required : ℕ := 12

/-- The amount of flour still needed -/
def flour_still_needed : ℕ := 2

theorem mary_flour_calculation :
  flour_put_in = total_flour_required - flour_still_needed :=
sorry

end NUMINAMATH_CALUDE_mary_flour_calculation_l2803_280359


namespace NUMINAMATH_CALUDE_fraction_sum_equality_l2803_280333

theorem fraction_sum_equality (a b c : ℝ) 
  (h : a / (20 - a) + b / (75 - b) + c / (55 - c) = 8) :
  4 / (20 - a) + 15 / (75 - b) + 11 / (55 - c) = 8.8 := by
  sorry

end NUMINAMATH_CALUDE_fraction_sum_equality_l2803_280333


namespace NUMINAMATH_CALUDE_complex_sum_exponential_form_l2803_280306

theorem complex_sum_exponential_form :
  10 * Complex.exp (2 * π * I / 11) + 10 * Complex.exp (15 * π * I / 22) =
  10 * Real.sqrt 2 * Complex.exp (19 * π * I / 44) := by
  sorry

end NUMINAMATH_CALUDE_complex_sum_exponential_form_l2803_280306


namespace NUMINAMATH_CALUDE_largest_five_digit_congruent_to_18_mod_25_l2803_280300

theorem largest_five_digit_congruent_to_18_mod_25 : ∃ n : ℕ,
  n = 99993 ∧
  n ≥ 10000 ∧ n < 100000 ∧
  n % 25 = 18 ∧
  ∀ m : ℕ, m ≥ 10000 ∧ m < 100000 ∧ m % 25 = 18 → m ≤ n :=
by sorry

end NUMINAMATH_CALUDE_largest_five_digit_congruent_to_18_mod_25_l2803_280300


namespace NUMINAMATH_CALUDE_x_percent_of_z_l2803_280323

theorem x_percent_of_z (x y z : ℝ) (h1 : x = 1.20 * y) (h2 : y = 0.50 * z) : x = 0.60 * z := by
  sorry

end NUMINAMATH_CALUDE_x_percent_of_z_l2803_280323


namespace NUMINAMATH_CALUDE_stratified_sampling_male_count_l2803_280386

theorem stratified_sampling_male_count 
  (total_male : ℕ) 
  (total_female : ℕ) 
  (sample_size : ℕ) 
  (h1 : total_male = 32) 
  (h2 : total_female = 24) 
  (h3 : sample_size = 14) :
  (total_male * sample_size) / (total_male + total_female) = 8 := by
  sorry

#check stratified_sampling_male_count

end NUMINAMATH_CALUDE_stratified_sampling_male_count_l2803_280386


namespace NUMINAMATH_CALUDE_fraction_simplification_l2803_280311

theorem fraction_simplification (x : ℝ) 
  (h1 : x + 1 ≠ 0) (h2 : 2 + x ≠ 0) (h3 : 2 - x ≠ 0) (h4 : x = 0) : 
  (x^2 - 4*x + 4) / (x + 1) / ((3 / (x + 1)) - x + 1) = 1 := by
  sorry

end NUMINAMATH_CALUDE_fraction_simplification_l2803_280311


namespace NUMINAMATH_CALUDE_money_division_l2803_280365

theorem money_division (total : ℝ) (p q r : ℝ) : 
  p + q + r = total →
  p / q = 3 / 7 →
  q / r = 7 / 12 →
  q - p = 3600 →
  r - q = 4500 := by
sorry

end NUMINAMATH_CALUDE_money_division_l2803_280365


namespace NUMINAMATH_CALUDE_gcd_problem_l2803_280374

theorem gcd_problem (b : ℤ) (h : 2373 ∣ b) : 
  Nat.gcd (Int.natAbs (b^2 + 13*b + 40)) (Int.natAbs (b + 5)) = 5 := by
  sorry

end NUMINAMATH_CALUDE_gcd_problem_l2803_280374


namespace NUMINAMATH_CALUDE_right_triangle_hypotenuse_l2803_280355

theorem right_triangle_hypotenuse (m1 m2 : ℝ) (h_m1 : m1 = 6) (h_m2 : m2 = Real.sqrt 50) :
  ∃ a b h : ℝ,
    a > 0 ∧ b > 0 ∧
    m1^2 = a^2 + (b/2)^2 ∧
    m2^2 = b^2 + (a/2)^2 ∧
    h^2 = (2*a)^2 + (2*b)^2 ∧
    h = Real.sqrt 275.2 :=
by sorry

end NUMINAMATH_CALUDE_right_triangle_hypotenuse_l2803_280355


namespace NUMINAMATH_CALUDE_sum_inequality_l2803_280321

theorem sum_inequality (a b c : ℝ) 
  (ha : 1/Real.sqrt 2 ≤ a ∧ a ≤ Real.sqrt 2)
  (hb : 1/Real.sqrt 2 ≤ b ∧ b ≤ Real.sqrt 2)
  (hc : 1/Real.sqrt 2 ≤ c ∧ c ≤ Real.sqrt 2) :
  (3/(a+2*b) + 3/(b+2*c) + 3/(c+2*a)) ≥ (2/(a+b) + 2/(b+c) + 2/(c+a)) := by
  sorry

end NUMINAMATH_CALUDE_sum_inequality_l2803_280321


namespace NUMINAMATH_CALUDE_brother_birth_year_l2803_280383

/-- Given Karina's birth year, current age, and the fact that she is twice as old as her brother,
    prove her brother's birth year. -/
theorem brother_birth_year
  (karina_birth_year : ℕ)
  (karina_current_age : ℕ)
  (h_karina_birth : karina_birth_year = 1970)
  (h_karina_age : karina_current_age = 40)
  (h_twice_age : karina_current_age = 2 * (karina_current_age / 2)) :
  karina_birth_year + karina_current_age - (karina_current_age / 2) = 1990 := by
  sorry

end NUMINAMATH_CALUDE_brother_birth_year_l2803_280383


namespace NUMINAMATH_CALUDE_pizza_combinations_l2803_280316

theorem pizza_combinations (n : ℕ) (h : n = 7) : 
  n + (n.choose 2) + (n.choose 3) = 63 := by sorry

end NUMINAMATH_CALUDE_pizza_combinations_l2803_280316


namespace NUMINAMATH_CALUDE_units_digit_of_n_l2803_280320

def units_digit (a : ℕ) : ℕ := a % 10

theorem units_digit_of_n (m n : ℕ) (h1 : m * n = 14^8) (h2 : units_digit m = 6) :
  units_digit n = 1 := by
  sorry

end NUMINAMATH_CALUDE_units_digit_of_n_l2803_280320


namespace NUMINAMATH_CALUDE_ship_always_illuminated_l2803_280362

/-- A lighthouse with a rotating beam -/
structure Lighthouse where
  /-- The position of the lighthouse -/
  position : ℝ × ℝ
  /-- The distance the beam reaches -/
  beam_distance : ℝ
  /-- The velocity of the beam's extremity -/
  beam_velocity : ℝ

/-- A ship moving towards the lighthouse -/
structure Ship where
  /-- The initial position of the ship -/
  initial_position : ℝ × ℝ
  /-- The maximum speed of the ship -/
  max_speed : ℝ

/-- The theorem stating that a ship moving at most v/8 cannot reach the lighthouse without being illuminated -/
theorem ship_always_illuminated (L : Lighthouse) (S : Ship) 
    (h1 : S.max_speed ≤ L.beam_velocity / 8)
    (h2 : dist S.initial_position L.position ≤ L.beam_distance) :
    ∃ (t : ℝ), t ∈ Set.Icc 0 (Real.pi * L.beam_distance / L.beam_velocity) ∧ 
    dist (S.initial_position + t • (L.position - S.initial_position)) L.position ≤ L.beam_distance :=
  sorry


end NUMINAMATH_CALUDE_ship_always_illuminated_l2803_280362


namespace NUMINAMATH_CALUDE_no_solution_iff_k_equals_seven_l2803_280303

theorem no_solution_iff_k_equals_seven :
  ∀ k : ℝ, (∀ x : ℝ, x ≠ 4 ∧ x ≠ 8 → (x - 3) / (x - 4) ≠ (x - k) / (x - 8)) ↔ k = 7 := by
  sorry

end NUMINAMATH_CALUDE_no_solution_iff_k_equals_seven_l2803_280303


namespace NUMINAMATH_CALUDE_temperature_conversion_fraction_l2803_280336

theorem temperature_conversion_fraction : 
  ∀ (t k : ℝ) (fraction : ℝ),
    t = fraction * (k - 32) →
    (t = 20 ∧ k = 68) →
    fraction = 5 / 9 := by
  sorry

end NUMINAMATH_CALUDE_temperature_conversion_fraction_l2803_280336


namespace NUMINAMATH_CALUDE_quadratic_strictly_increasing_iff_l2803_280345

/-- A function f: ℝ → ℝ is strictly increasing if for all x < y, f(x) < f(y) -/
def StrictlyIncreasing (f : ℝ → ℝ) : Prop :=
  ∀ x y, x < y → f x < f y

/-- The quadratic function f(x) = ax^2 + 2x - 3 -/
def f (a : ℝ) (x : ℝ) : ℝ := a * x^2 + 2 * x - 3

theorem quadratic_strictly_increasing_iff (a : ℝ) :
  StrictlyIncreasing (f a) ↔ -1/4 ≤ a ∧ a ≤ 0 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_strictly_increasing_iff_l2803_280345


namespace NUMINAMATH_CALUDE_furniture_payment_l2803_280318

theorem furniture_payment (a b c d e : ℝ) : 
  a + b + c + d + e = 120 ∧
  a = (1/3) * (b + c + d + e) ∧
  b = (1/4) * (a + c + d + e) ∧
  c = (1/5) * (a + b + d + e) ∧
  d = (1/6) * (a + b + c + e) →
  e = 41.33 := by sorry

end NUMINAMATH_CALUDE_furniture_payment_l2803_280318


namespace NUMINAMATH_CALUDE_min_value_h_l2803_280315

theorem min_value_h (x : ℝ) (hx : x > 0) :
  x^2 + 1/x^2 + 1/(x^2 + 1/x^2) ≥ 2.5 := by
  sorry

end NUMINAMATH_CALUDE_min_value_h_l2803_280315


namespace NUMINAMATH_CALUDE_reverse_geometric_difference_l2803_280337

/-- A 3-digit number is reverse geometric if it has 3 distinct digits which,
    when read from right to left, form a geometric sequence. -/
def is_reverse_geometric (n : ℕ) : Prop :=
  100 ≤ n ∧ n < 1000 ∧
  ∃ (a b c : ℕ) (r : ℚ),
    n = 100 * a + 10 * b + c ∧
    a ≠ b ∧ b ≠ c ∧ a ≠ c ∧
    0 < r ∧
    (b : ℚ) = c * r ∧
    (a : ℚ) = b * r

def largest_reverse_geometric : ℕ := sorry

def smallest_reverse_geometric : ℕ := sorry

theorem reverse_geometric_difference :
  largest_reverse_geometric - smallest_reverse_geometric = 789 :=
sorry

end NUMINAMATH_CALUDE_reverse_geometric_difference_l2803_280337


namespace NUMINAMATH_CALUDE_geometric_subsequence_k4_l2803_280375

/-- An arithmetic sequence with a non-zero common difference -/
structure ArithmeticSequence where
  a : ℕ → ℝ
  d : ℝ
  h_d : d ≠ 0
  h_arith : ∀ n : ℕ, a (n + 1) = a n + d

/-- A subsequence of an arithmetic sequence that forms a geometric sequence -/
structure GeometricSubsequence (as : ArithmeticSequence) where
  k : ℕ → ℕ
  q : ℝ
  h_geom : ∀ n : ℕ, as.a (k (n + 1)) = q * as.a (k n)
  h_k1 : k 1 ≠ 1
  h_k2 : k 2 ≠ 2
  h_k3 : k 3 ≠ 6

/-- The main theorem -/
theorem geometric_subsequence_k4 (as : ArithmeticSequence) (gs : GeometricSubsequence as) :
  gs.k 4 = 22 := by
  sorry

end NUMINAMATH_CALUDE_geometric_subsequence_k4_l2803_280375


namespace NUMINAMATH_CALUDE_solve_linear_system_l2803_280382

theorem solve_linear_system (b : ℚ) : 
  (∃ x y : ℚ, x + b * y = 0 ∧ x + y = -1 ∧ x = 1) →
  b = 1/2 := by
sorry

end NUMINAMATH_CALUDE_solve_linear_system_l2803_280382


namespace NUMINAMATH_CALUDE_minimizing_n_is_six_l2803_280341

/-- An arithmetic sequence with specific properties -/
structure ArithmeticSequence where
  a : ℕ → ℤ  -- The sequence
  sum : ℕ → ℤ  -- The sum function
  first_fifth_sum : a 1 + a 5 = -14
  ninth_sum : sum 9 = -27

/-- The value of n that minimizes the sum of the first n terms -/
def minimizing_n (seq : ArithmeticSequence) : ℕ :=
  6

/-- Theorem stating that 6 is the value of n that minimizes S_n -/
theorem minimizing_n_is_six (seq : ArithmeticSequence) :
  ∀ n : ℕ, seq.sum n ≥ seq.sum (minimizing_n seq) :=
sorry

end NUMINAMATH_CALUDE_minimizing_n_is_six_l2803_280341


namespace NUMINAMATH_CALUDE_parallel_transitivity_counterexample_l2803_280302

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the parallel relation for lines and planes
variable (parallel_line_plane : Line → Plane → Prop)
variable (parallel_plane_plane : Plane → Plane → Prop)

-- State the theorem
theorem parallel_transitivity_counterexample 
  (a : Line) (α β : Plane) :
  ¬(∀ a α β, parallel_line_plane a α → parallel_line_plane a β → 
    parallel_plane_plane α β) :=
sorry

end NUMINAMATH_CALUDE_parallel_transitivity_counterexample_l2803_280302


namespace NUMINAMATH_CALUDE_sonia_and_joss_moving_l2803_280326

/-- Calculates the time spent filling the car per trip given the total moving time,
    number of trips, and driving time per trip. -/
def time_filling_car_per_trip (total_moving_time : ℕ) (num_trips : ℕ) (driving_time_per_trip : ℕ) : ℕ :=
  let total_minutes := total_moving_time * 60
  let total_driving_time := driving_time_per_trip * num_trips
  let total_filling_time := total_minutes - total_driving_time
  total_filling_time / num_trips

/-- Theorem stating that given the specific conditions of the problem,
    the time spent filling the car per trip is 40 minutes. -/
theorem sonia_and_joss_moving (total_moving_time : ℕ) (num_trips : ℕ) (driving_time_per_trip : ℕ) :
  total_moving_time = 7 →
  num_trips = 6 →
  driving_time_per_trip = 30 →
  time_filling_car_per_trip total_moving_time num_trips driving_time_per_trip = 40 :=
by
  sorry

#eval time_filling_car_per_trip 7 6 30

end NUMINAMATH_CALUDE_sonia_and_joss_moving_l2803_280326


namespace NUMINAMATH_CALUDE_prime_arithmetic_progression_difference_l2803_280335

theorem prime_arithmetic_progression_difference (a : ℕ → ℕ) (d : ℕ) :
  (∀ k, k ∈ Finset.range 15 → Nat.Prime (a k)) →
  (∀ k, k ∈ Finset.range 14 → a (k + 1) = a k + d) →
  (∀ k l, k < l → k ∈ Finset.range 15 → l ∈ Finset.range 15 → a k < a l) →
  d > 30000 := by
  sorry

end NUMINAMATH_CALUDE_prime_arithmetic_progression_difference_l2803_280335


namespace NUMINAMATH_CALUDE_rectangle_shorter_side_l2803_280312

theorem rectangle_shorter_side 
  (perimeter : ℝ) 
  (area : ℝ) 
  (h_perimeter : perimeter = 60) 
  (h_area : area = 200) :
  ∃ (shorter_side longer_side : ℝ),
    shorter_side ≤ longer_side ∧
    2 * (shorter_side + longer_side) = perimeter ∧
    shorter_side * longer_side = area ∧
    shorter_side = 10 :=
by sorry

end NUMINAMATH_CALUDE_rectangle_shorter_side_l2803_280312


namespace NUMINAMATH_CALUDE_unique_solution_factorial_equation_l2803_280373

theorem unique_solution_factorial_equation :
  ∃! (n : ℕ), n * n.factorial + 2 * n.factorial = 5040 :=
by
  -- Proof goes here
  sorry

end NUMINAMATH_CALUDE_unique_solution_factorial_equation_l2803_280373


namespace NUMINAMATH_CALUDE_yvonne_success_probability_l2803_280357

theorem yvonne_success_probability 
  (p_xavier : ℝ) 
  (p_zelda : ℝ) 
  (p_xavier_yvonne_not_zelda : ℝ) 
  (h1 : p_xavier = 1/5)
  (h2 : p_zelda = 5/8)
  (h3 : p_xavier_yvonne_not_zelda = 0.0375) :
  ∃ p_yvonne : ℝ, 
    p_xavier * p_yvonne * (1 - p_zelda) = p_xavier_yvonne_not_zelda ∧ 
    p_yvonne = 1/2 :=
by sorry

end NUMINAMATH_CALUDE_yvonne_success_probability_l2803_280357


namespace NUMINAMATH_CALUDE_percentage_pies_with_forks_l2803_280354

def total_pies : ℕ := 2000
def pies_not_with_forks : ℕ := 640

theorem percentage_pies_with_forks :
  (total_pies - pies_not_with_forks : ℚ) / total_pies * 100 = 68 := by
  sorry

end NUMINAMATH_CALUDE_percentage_pies_with_forks_l2803_280354


namespace NUMINAMATH_CALUDE_circle_condition_l2803_280339

/-- The equation of a potential circle -/
def circle_equation (x y a : ℝ) : Prop :=
  x^2 + y^2 - 2*x + 2*y + a = 0

/-- Definition of a circle in 2D space -/
def is_circle (center : ℝ × ℝ) (radius : ℝ) (x y : ℝ) : Prop :=
  (x - center.1)^2 + (y - center.2)^2 = radius^2

theorem circle_condition (a : ℝ) :
  (∀ x y, ∃ center radius, circle_equation x y a → is_circle center radius x y) ↔ a < 2 :=
sorry

end NUMINAMATH_CALUDE_circle_condition_l2803_280339


namespace NUMINAMATH_CALUDE_xyz_sum_l2803_280309

theorem xyz_sum (x y z : ℕ+) (h : (x + y * Complex.I)^2 - 46 * Complex.I = z) :
  x + y + z = 552 := by
  sorry

end NUMINAMATH_CALUDE_xyz_sum_l2803_280309


namespace NUMINAMATH_CALUDE_chocolates_in_box_l2803_280372

/-- Represents the dimensions of a cuboid -/
structure Dimensions where
  width : ℝ
  length : ℝ
  height : ℝ

/-- Calculates the volume of a cuboid given its dimensions -/
def volume (d : Dimensions) : ℝ :=
  d.width * d.length * d.height

/-- The dimensions of the box -/
def box_dimensions : Dimensions :=
  { width := 30, length := 20, height := 5 }

/-- The dimensions of a single chocolate -/
def chocolate_dimensions : Dimensions :=
  { width := 6, length := 4, height := 1 }

/-- Theorem stating that the number of chocolates in the box is 125 -/
theorem chocolates_in_box :
  (volume box_dimensions) / (volume chocolate_dimensions) = 125 := by
  sorry

end NUMINAMATH_CALUDE_chocolates_in_box_l2803_280372


namespace NUMINAMATH_CALUDE_oldest_person_is_A_l2803_280347

-- Define the set of people
inductive Person : Type
  | A : Person
  | B : Person
  | C : Person
  | D : Person

-- Define the age relation
def olderThan : Person → Person → Prop := sorry

-- Define the statements made by each person
def statementA : Prop := olderThan Person.B Person.D
def statementB : Prop := olderThan Person.C Person.A
def statementC : Prop := olderThan Person.D Person.C
def statementD : Prop := olderThan Person.B Person.C

-- Define a function to check if a statement is true
def isTrueStatement (p : Person) : Prop :=
  match p with
  | Person.A => statementA
  | Person.B => statementB
  | Person.C => statementC
  | Person.D => statementD

-- Theorem to prove
theorem oldest_person_is_A :
  (∀ (p q : Person), p ≠ q → olderThan p q ∨ olderThan q p) →
  (∀ (p q r : Person), olderThan p q → olderThan q r → olderThan p r) →
  (∃! (p : Person), isTrueStatement p) →
  (∀ (p : Person), isTrueStatement p → ∀ (q : Person), q ≠ p → olderThan p q) →
  (∀ (p : Person), olderThan Person.A p ∨ p = Person.A) :=
sorry

end NUMINAMATH_CALUDE_oldest_person_is_A_l2803_280347


namespace NUMINAMATH_CALUDE_age_problem_l2803_280376

theorem age_problem (billy joe sarah : ℕ) 
  (h1 : billy = 3 * joe)
  (h2 : billy + joe = 48)
  (h3 : joe + sarah = 30) :
  billy = 36 ∧ joe = 12 ∧ sarah = 18 := by
sorry

end NUMINAMATH_CALUDE_age_problem_l2803_280376


namespace NUMINAMATH_CALUDE_M_inequalities_l2803_280327

/-- M_(n,k,h) is the maximum number of h-element subsets of an n-element set X with property P_k(X) -/
def M (n k h : ℕ) : ℕ := sorry

/-- The three inequalities for M_(n,k,h) -/
theorem M_inequalities (n k h : ℕ) (hn : n > 0) (hk : k > 0) (hh : h > 0) (hnkh : n ≥ k) (hkh : k ≥ h) :
  (M n k h ≤ (n / h) * M (n-1) (k-1) (h-1)) ∧
  (M n k h ≥ (n / (n-h)) * M (n-1) k h) ∧
  (M n k h ≤ M (n-1) (k-1) (h-1) + M (n-1) k h) :=
sorry

end NUMINAMATH_CALUDE_M_inequalities_l2803_280327


namespace NUMINAMATH_CALUDE_isosceles_smallest_hypotenuse_l2803_280387

-- Define a triangle with sides a, b, c and angles α, β, γ
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  α : ℝ
  β : ℝ
  γ : ℝ
  -- Ensure angles sum to π
  angle_sum : α + β + γ = π
  -- Ensure sides are positive
  positive_sides : 0 < a ∧ 0 < b ∧ 0 < c
  -- Law of sines
  law_of_sines : a / Real.sin α = b / Real.sin β
                ∧ b / Real.sin β = c / Real.sin γ

-- Define the perimeter of a triangle
def perimeter (t : Triangle) : ℝ := t.a + t.b + t.c

-- Theorem: Among all triangles with fixed perimeter and angle γ,
-- the isosceles triangle (α = β) has the smallest hypotenuse
theorem isosceles_smallest_hypotenuse 
  (t1 t2 : Triangle) 
  (same_perimeter : perimeter t1 = perimeter t2)
  (same_gamma : t1.γ = t2.γ)
  (t2_isosceles : t2.α = t2.β)
  : t2.c ≤ t1.c := by
  sorry

end NUMINAMATH_CALUDE_isosceles_smallest_hypotenuse_l2803_280387


namespace NUMINAMATH_CALUDE_solve_video_game_problem_l2803_280351

def video_game_problem (initial_players : ℕ) (lives_per_player : ℕ) (total_lives : ℕ) : Prop :=
  let remaining_players := total_lives / lives_per_player
  let players_quit := initial_players - remaining_players
  players_quit = 5

theorem solve_video_game_problem :
  video_game_problem 11 5 30 :=
by
  sorry

end NUMINAMATH_CALUDE_solve_video_game_problem_l2803_280351


namespace NUMINAMATH_CALUDE_inscribed_triangle_circumscribed_square_l2803_280394

theorem inscribed_triangle_circumscribed_square (r : ℝ) : 
  r > 0 → 
  let triangle_side := r * Real.sqrt 3
  let triangle_perimeter := 3 * triangle_side
  let square_side := r * Real.sqrt 2
  let square_area := square_side ^ 2
  triangle_perimeter = square_area →
  r = 3 * Real.sqrt 3 / 4 := by
sorry

end NUMINAMATH_CALUDE_inscribed_triangle_circumscribed_square_l2803_280394


namespace NUMINAMATH_CALUDE_solution_value_l2803_280332

theorem solution_value (r s : ℝ) : 
  (3 * r^2 - 5 * r = 7) → 
  (3 * s^2 - 5 * s = 7) → 
  r ≠ s →
  (9 * r^2 - 9 * s^2) / (r - s) = 15 := by
sorry

end NUMINAMATH_CALUDE_solution_value_l2803_280332


namespace NUMINAMATH_CALUDE_equation_value_l2803_280393

theorem equation_value (x y : ℝ) (h : 2*x - y = -1) : 3 + 4*x - 2*y = 1 := by
  sorry

end NUMINAMATH_CALUDE_equation_value_l2803_280393


namespace NUMINAMATH_CALUDE_percent_k_equal_to_125_percent_j_l2803_280370

theorem percent_k_equal_to_125_percent_j (j k l m : ℝ) 
  (h1 : 1.25 * j = (12.5 / 100) * k)
  (h2 : 1.5 * k = 0.5 * l)
  (h3 : 1.75 * l = 0.75 * m)
  (h4 : 0.2 * m = 3.5 * (2 * j)) :
  (12.5 / 100) * k = 1.25 * j := by
  sorry

end NUMINAMATH_CALUDE_percent_k_equal_to_125_percent_j_l2803_280370


namespace NUMINAMATH_CALUDE_product_third_fourth_term_l2803_280388

/-- An arithmetic sequence with common difference 2 and eighth term 20 -/
def ArithmeticSequence (a : ℕ) : ℕ → ℕ :=
  fun n => a + (n - 1) * 2

theorem product_third_fourth_term (a : ℕ) :
  ArithmeticSequence a 8 = 20 →
  ArithmeticSequence a 3 * ArithmeticSequence a 4 = 120 := by
  sorry

end NUMINAMATH_CALUDE_product_third_fourth_term_l2803_280388


namespace NUMINAMATH_CALUDE_constant_term_expansion_l2803_280313

theorem constant_term_expansion (x : ℝ) (x_ne_zero : x ≠ 0) : 
  ∃ (c : ℕ), c = 17920 ∧ 
  ∃ (f : ℝ → ℝ), (λ x => (2*x + 2/x)^8) = (λ x => c + f x) ∧ 
  (∀ x ≠ 0, f x ≠ 0 → ∃ (n : ℤ), n ≠ 0 ∧ f x = x^n * (f x / x^n)) :=
by sorry

end NUMINAMATH_CALUDE_constant_term_expansion_l2803_280313


namespace NUMINAMATH_CALUDE_parallelogram_perimeter_l2803_280399

theorem parallelogram_perimeter (n : ℕ) (h : n = 92) :
  ∃ (a b : ℕ), a * b = n ∧ (2 * a + 2 * b = 94 ∨ 2 * a + 2 * b = 50) :=
by
  sorry

end NUMINAMATH_CALUDE_parallelogram_perimeter_l2803_280399


namespace NUMINAMATH_CALUDE_parabola_line_intersection_l2803_280328

/-- The parabola equation -/
def parabola (a x y : ℝ) : Prop := y = a * x^2 + 5 * x + 2

/-- The line equation -/
def line (x y : ℝ) : Prop := y = -2 * x + 1

/-- The intersection condition -/
def intersect_once (a : ℝ) : Prop :=
  ∃! p : ℝ × ℝ, parabola a p.1 p.2 ∧ line p.1 p.2

/-- The theorem statement -/
theorem parabola_line_intersection (a : ℝ) :
  intersect_once a ↔ a = 49 / 4 := by sorry

end NUMINAMATH_CALUDE_parabola_line_intersection_l2803_280328


namespace NUMINAMATH_CALUDE_angle_cosine_relation_l2803_280304

/-- Given a point Q in 3D space with positive coordinates, and angles α, β, γ between OQ and the x, y, z axes respectively, prove that if cos α = 2/5 and cos β = 1/4, then cos γ = √(311)/20 -/
theorem angle_cosine_relation (Q : ℝ × ℝ × ℝ) (α β γ : ℝ) 
  (h_pos : Q.1 > 0 ∧ Q.2.1 > 0 ∧ Q.2.2 > 0)
  (h_α : α = Real.arccos (Q.1 / Real.sqrt (Q.1^2 + Q.2.1^2 + Q.2.2^2)))
  (h_β : β = Real.arccos (Q.2.1 / Real.sqrt (Q.1^2 + Q.2.1^2 + Q.2.2^2)))
  (h_γ : γ = Real.arccos (Q.2.2 / Real.sqrt (Q.1^2 + Q.2.1^2 + Q.2.2^2)))
  (h_cos_α : Real.cos α = 2/5)
  (h_cos_β : Real.cos β = 1/4) :
  Real.cos γ = Real.sqrt 311 / 20 := by
  sorry

end NUMINAMATH_CALUDE_angle_cosine_relation_l2803_280304


namespace NUMINAMATH_CALUDE_souvenir_purchasing_plans_l2803_280396

def number_of_purchasing_plans (total_items : ℕ) (types : ℕ) (items_per_type : ℕ) : ℕ :=
  let f (x : ℕ → ℕ) := x 1 + x 2 + x 3 + x 4 + x 5 + x 6 + x 7 + x 8 + x 9 + x 10
  let coefficient_of_x25 := (Nat.choose 24 3) - 4 * (Nat.choose 14 3) + 6 * (Nat.choose 4 3)
  coefficient_of_x25

theorem souvenir_purchasing_plans :
  number_of_purchasing_plans 25 4 10 = 592 :=
sorry

end NUMINAMATH_CALUDE_souvenir_purchasing_plans_l2803_280396

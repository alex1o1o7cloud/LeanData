import Mathlib

namespace NUMINAMATH_CALUDE_tenth_term_of_specific_geometric_sequence_l1885_188578

/-- Given a geometric sequence with first term a and second term b,
    this function returns the nth term of the sequence. -/
def geometric_sequence_term (a b : ℚ) (n : ℕ) : ℚ :=
  let r := b / a
  a * r ^ (n - 1)

/-- Theorem stating that the 10th term of the geometric sequence
    with first term 8 and second term -16/3 is -4096/19683. -/
theorem tenth_term_of_specific_geometric_sequence :
  geometric_sequence_term 8 (-16/3) 10 = -4096/19683 := by
  sorry

#eval geometric_sequence_term 8 (-16/3) 10

end NUMINAMATH_CALUDE_tenth_term_of_specific_geometric_sequence_l1885_188578


namespace NUMINAMATH_CALUDE_binary_1011_is_11_l1885_188575

def binary_to_decimal (b : List Bool) : ℕ :=
  List.foldl (λ acc d => 2 * acc + if d then 1 else 0) 0 b

theorem binary_1011_is_11 :
  binary_to_decimal [true, false, true, true] = 11 := by
  sorry

end NUMINAMATH_CALUDE_binary_1011_is_11_l1885_188575


namespace NUMINAMATH_CALUDE_integer_solutions_quadratic_equation_l1885_188596

theorem integer_solutions_quadratic_equation :
  ∀ m n : ℤ, n^2 - 3*m*n + m - n = 0 ↔ (m = 0 ∧ n = 0) ∨ (m = 0 ∧ n = 1) := by
sorry

end NUMINAMATH_CALUDE_integer_solutions_quadratic_equation_l1885_188596


namespace NUMINAMATH_CALUDE_percentage_to_pass_l1885_188559

def max_marks : ℕ := 780
def passing_marks : ℕ := 234

theorem percentage_to_pass : 
  (passing_marks : ℝ) / max_marks * 100 = 30 := by sorry

end NUMINAMATH_CALUDE_percentage_to_pass_l1885_188559


namespace NUMINAMATH_CALUDE_victoria_work_hours_l1885_188565

/-- Calculates the number of hours worked per day given the total hours and number of weeks worked. -/
def hours_per_day (total_hours : ℕ) (weeks : ℕ) : ℚ :=
  total_hours / (weeks * 7)

/-- Theorem: Given 315 total hours worked over 5 weeks, the number of hours worked per day is 9. -/
theorem victoria_work_hours :
  hours_per_day 315 5 = 9 := by
  sorry

end NUMINAMATH_CALUDE_victoria_work_hours_l1885_188565


namespace NUMINAMATH_CALUDE_condition_equivalence_l1885_188546

theorem condition_equivalence (a : ℝ) (h : a > 0) : (a > 1) ↔ (a > Real.sqrt a) := by
  sorry

end NUMINAMATH_CALUDE_condition_equivalence_l1885_188546


namespace NUMINAMATH_CALUDE_regular_polygon_sides_l1885_188577

theorem regular_polygon_sides (n : ℕ) (interior_angle : ℝ) : 
  interior_angle = 165 → (n : ℝ) * interior_angle = (n - 2 : ℝ) * 180 → n = 24 := by
  sorry

end NUMINAMATH_CALUDE_regular_polygon_sides_l1885_188577


namespace NUMINAMATH_CALUDE_equation_solution_l1885_188591

theorem equation_solution (x k : ℝ) : 
  (7 * x + 2 = 3 * x - 6) ∧ (x + 1 = k) → 3 * k^2 - 1 = 2 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l1885_188591


namespace NUMINAMATH_CALUDE_parallelogram_area_l1885_188518

/-- A parallelogram bounded by lines y = a, y = -b, x = -c + 2y, and x = d - 2y -/
structure Parallelogram where
  a : ℝ
  b : ℝ
  c : ℝ
  d : ℝ
  a_pos : 0 < a
  b_pos : 0 < b
  c_pos : 0 < c
  d_pos : 0 < d

/-- The area of the parallelogram -/
def area (p : Parallelogram) : ℝ :=
  p.a * p.d + p.a * p.c + p.b * p.d + p.b * p.c

theorem parallelogram_area (p : Parallelogram) :
  area p = p.a * p.d + p.a * p.c + p.b * p.d + p.b * p.c := by
  sorry

end NUMINAMATH_CALUDE_parallelogram_area_l1885_188518


namespace NUMINAMATH_CALUDE_abs_diff_bound_l1885_188547

theorem abs_diff_bound (a b c h : ℝ) (ha : |a - c| < h) (hb : |b - c| < h) : |a - b| < 2 * h := by
  sorry

end NUMINAMATH_CALUDE_abs_diff_bound_l1885_188547


namespace NUMINAMATH_CALUDE_plant_purchase_cost_l1885_188522

/-- Calculates the actual amount spent on plants given the original cost and discount. -/
def actualCost (originalCost discount : ℚ) : ℚ :=
  originalCost - discount

/-- Theorem stating that given the specific original cost and discount, the actual amount spent is $68.00. -/
theorem plant_purchase_cost :
  let originalCost : ℚ := 467
  let discount : ℚ := 399
  actualCost originalCost discount = 68 := by
sorry

end NUMINAMATH_CALUDE_plant_purchase_cost_l1885_188522


namespace NUMINAMATH_CALUDE_fraction_exists_l1885_188526

theorem fraction_exists : ∃ n : ℕ, (n : ℚ) / 22 = 9545 / 10000 := by
  sorry

end NUMINAMATH_CALUDE_fraction_exists_l1885_188526


namespace NUMINAMATH_CALUDE_stem_and_leaf_preserves_info_l1885_188528

/-- Represents different types of statistical charts -/
inductive StatChart
  | BarChart
  | PieChart
  | LineChart
  | StemAndLeafPlot

/-- Predicate to determine if a chart loses information -/
def loses_information (chart : StatChart) : Prop :=
  match chart with
  | StatChart.BarChart => True
  | StatChart.PieChart => True
  | StatChart.LineChart => True
  | StatChart.StemAndLeafPlot => False

/-- Theorem stating that only the stem-and-leaf plot does not lose information -/
theorem stem_and_leaf_preserves_info :
  ∀ (chart : StatChart), ¬(loses_information chart) ↔ chart = StatChart.StemAndLeafPlot :=
by sorry


end NUMINAMATH_CALUDE_stem_and_leaf_preserves_info_l1885_188528


namespace NUMINAMATH_CALUDE_no_rain_probability_l1885_188598

theorem no_rain_probability (p : ℝ) (h : p = 2/3) : (1 - p)^5 = 1/243 := by
  sorry

end NUMINAMATH_CALUDE_no_rain_probability_l1885_188598


namespace NUMINAMATH_CALUDE_three_zeros_condition_l1885_188567

-- Define the function f(x) = x^3 + ax + 2
def f (a : ℝ) (x : ℝ) : ℝ := x^3 + a*x + 2

-- Theorem statement
theorem three_zeros_condition (a : ℝ) :
  (∃ x y z : ℝ, x ≠ y ∧ y ≠ z ∧ x ≠ z ∧ f a x = 0 ∧ f a y = 0 ∧ f a z = 0) ↔ a < -3 :=
sorry

end NUMINAMATH_CALUDE_three_zeros_condition_l1885_188567


namespace NUMINAMATH_CALUDE_parabola_intersection_slope_l1885_188553

/-- Parabola structure -/
structure Parabola where
  p : ℝ
  h_p_pos : p > 0

/-- Point on a parabola -/
structure ParabolaPoint (C : Parabola) where
  x : ℝ
  y : ℝ
  h_on_parabola : y^2 = 2 * C.p * x

/-- Theorem: For a parabola y² = 2px with p > 0, if a line through M(-p/2, 0) with slope k
    intersects the parabola at A(x₀, y₀) such that |AM| = 5/4 * |AF|, then k = ±3/4 -/
theorem parabola_intersection_slope (C : Parabola) (A : ParabolaPoint C) (k : ℝ) :
  let M : ℝ × ℝ := (-C.p/2, 0)
  let F : ℝ × ℝ := (C.p/2, 0)
  let AM := Real.sqrt ((A.x + C.p/2)^2 + A.y^2)
  let AF := A.x + C.p/2
  (A.y - 0) / (A.x - (-C.p/2)) = k →
  AM = 5/4 * AF →
  k = 3/4 ∨ k = -3/4 := by
sorry

end NUMINAMATH_CALUDE_parabola_intersection_slope_l1885_188553


namespace NUMINAMATH_CALUDE_line_slope_is_one_l1885_188561

/-- Given a line in the xy-plane with y-intercept -2 and passing through the midpoint
    of the line segment with endpoints (2, 8) and (14, 4), its slope is 1. -/
theorem line_slope_is_one (m : Set (ℝ × ℝ)) : 
  (∀ (x y : ℝ), (x, y) ∈ m ↔ y = x - 2) →  -- y-intercept is -2
  ((8 : ℝ), 6) ∈ m →  -- passes through midpoint ((2+14)/2, (8+4)/2) = (8, 6)
  (∃ (k b : ℝ), ∀ (x y : ℝ), (x, y) ∈ m ↔ y = k * x + b) →  -- m is a line
  (∃ (k : ℝ), ∀ (x y : ℝ), (x, y) ∈ m ↔ y = k * x - 2) →  -- combine line equation with y-intercept
  ∀ (x y : ℝ), (x, y) ∈ m ↔ y = x - 2 :=
by sorry

end NUMINAMATH_CALUDE_line_slope_is_one_l1885_188561


namespace NUMINAMATH_CALUDE_common_root_existence_l1885_188514

/-- The common rational root of two polynomials -/
def k : ℚ := -3/5

/-- First polynomial -/
def P (x : ℚ) (a b c : ℚ) : ℚ := 90 * x^4 + a * x^3 + b * x^2 + c * x + 15

/-- Second polynomial -/
def Q (x : ℚ) (d e f g : ℚ) : ℚ := 15 * x^5 + d * x^4 + e * x^3 + f * x^2 + g * x + 90

theorem common_root_existence (a b c d e f g : ℚ) :
  (k ≠ 0) ∧ 
  (k < 0) ∧ 
  (∃ (m n : ℤ), k = m / n ∧ m ≠ 0 ∧ n ≠ 0 ∧ Int.gcd m n = 1) ∧
  (P k a b c = 0) ∧ 
  (Q k d e f g = 0) :=
sorry

end NUMINAMATH_CALUDE_common_root_existence_l1885_188514


namespace NUMINAMATH_CALUDE_regression_lines_intersect_at_mean_l1885_188586

/-- Represents a linear regression line -/
structure RegressionLine where
  slope : ℝ
  intercept : ℝ

/-- Checks if a point lies on a regression line -/
def point_on_line (l : RegressionLine) (x y : ℝ) : Prop :=
  y = l.slope * x + l.intercept

/-- Theorem: Two different regression lines for the same dataset intersect at the sample mean -/
theorem regression_lines_intersect_at_mean 
  (l₁ l₂ : RegressionLine) 
  (x_mean y_mean : ℝ) 
  (h_different : l₁ ≠ l₂) 
  (h_on_l₁ : point_on_line l₁ x_mean y_mean)
  (h_on_l₂ : point_on_line l₂ x_mean y_mean) : 
  ∃ (x y : ℝ), x = x_mean ∧ y = y_mean ∧ point_on_line l₁ x y ∧ point_on_line l₂ x y :=
sorry

end NUMINAMATH_CALUDE_regression_lines_intersect_at_mean_l1885_188586


namespace NUMINAMATH_CALUDE_value_equals_eleven_l1885_188520

theorem value_equals_eleven (number : ℝ) (value : ℝ) : 
  number = 10 → 
  (number / 2) + 6 = value → 
  value = 11 := by
sorry

end NUMINAMATH_CALUDE_value_equals_eleven_l1885_188520


namespace NUMINAMATH_CALUDE_union_of_positive_and_square_ge_self_is_reals_l1885_188525

open Set

theorem union_of_positive_and_square_ge_self_is_reals :
  let M : Set ℝ := {x | x > 0}
  let N : Set ℝ := {x | x^2 ≥ x}
  M ∪ N = univ := by sorry

end NUMINAMATH_CALUDE_union_of_positive_and_square_ge_self_is_reals_l1885_188525


namespace NUMINAMATH_CALUDE_mans_downstream_rate_l1885_188507

/-- A man rowing in a river with a current -/
theorem mans_downstream_rate 
  (rate_still_water : ℝ) 
  (rate_current : ℝ) 
  (rate_upstream : ℝ) 
  (h1 : rate_still_water = 6) 
  (h2 : rate_current = 6) 
  (h3 : rate_upstream = 6) :
  rate_still_water + rate_current = 12 := by sorry

end NUMINAMATH_CALUDE_mans_downstream_rate_l1885_188507


namespace NUMINAMATH_CALUDE_reciprocal_of_negative_three_l1885_188580

theorem reciprocal_of_negative_three :
  (1 : ℚ) / (-3 : ℚ) = -1/3 := by sorry

end NUMINAMATH_CALUDE_reciprocal_of_negative_three_l1885_188580


namespace NUMINAMATH_CALUDE_exists_k_fourth_power_between_1000_and_2000_l1885_188555

theorem exists_k_fourth_power_between_1000_and_2000 : 
  ∃ (k : ℤ), 1000 < k^4 ∧ k^4 < 2000 ∧ ∃ (n : ℤ), k^4 = n^2 := by
  sorry

end NUMINAMATH_CALUDE_exists_k_fourth_power_between_1000_and_2000_l1885_188555


namespace NUMINAMATH_CALUDE_silverware_probability_l1885_188536

theorem silverware_probability (forks spoons knives : ℕ) (h1 : forks = 6) (h2 : spoons = 6) (h3 : knives = 6) :
  let total := forks + spoons + knives
  let ways_to_choose_three := Nat.choose total 3
  let ways_to_choose_one_each := forks * spoons * knives
  (ways_to_choose_one_each : ℚ) / ways_to_choose_three = 9 / 34 :=
by
  sorry

end NUMINAMATH_CALUDE_silverware_probability_l1885_188536


namespace NUMINAMATH_CALUDE_xyz_sum_l1885_188588

theorem xyz_sum (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0)
  (eq1 : x^2 + x*y + y^2 = 147)
  (eq2 : y^2 + y*z + z^2 = 16)
  (eq3 : z^2 + x*z + x^2 = 163) :
  x*y + y*z + x*z = 56 := by
sorry

end NUMINAMATH_CALUDE_xyz_sum_l1885_188588


namespace NUMINAMATH_CALUDE_imaginary_part_of_2_minus_i_l1885_188566

theorem imaginary_part_of_2_minus_i :
  let z : ℂ := 2 - I
  Complex.im z = -1 := by sorry

end NUMINAMATH_CALUDE_imaginary_part_of_2_minus_i_l1885_188566


namespace NUMINAMATH_CALUDE_arithmetic_sequence_sum_l1885_188543

/-- An arithmetic sequence -/
def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_sum
  (a : ℕ → ℝ)
  (h_arith : arithmetic_sequence a)
  (h_sum : a 5 + a 6 + a 7 + a 8 + a 9 = 450) :
  a 3 + a 11 = 180 := by
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_sum_l1885_188543


namespace NUMINAMATH_CALUDE_polar_to_cartesian_l1885_188540

-- Define the polar equation
def polar_equation (ρ θ : ℝ) : Prop :=
  ρ^2 * Real.cos θ + ρ - 3 * ρ * Real.cos θ - 3 = 0

-- Define the Cartesian equations
def circle_equation (x y : ℝ) : Prop :=
  x^2 + y^2 = 9

def line_equation (x : ℝ) : Prop :=
  x = -1

-- Theorem statement
theorem polar_to_cartesian :
  ∀ ρ θ x y : ℝ, 
    polar_equation ρ θ ↔ 
    (circle_equation x y ∨ line_equation x) ∧
    x = ρ * Real.cos θ ∧
    y = ρ * Real.sin θ :=
sorry

end NUMINAMATH_CALUDE_polar_to_cartesian_l1885_188540


namespace NUMINAMATH_CALUDE_tennis_tournament_matches_l1885_188571

theorem tennis_tournament_matches (n : ℕ) (byes : ℕ) (wildcard : ℕ) : 
  n = 128 → byes = 36 → wildcard = 1 →
  ∃ (total_matches : ℕ), 
    total_matches = n - 1 + wildcard ∧ 
    total_matches = 128 ∧
    total_matches % 2 = 0 :=
by sorry

end NUMINAMATH_CALUDE_tennis_tournament_matches_l1885_188571


namespace NUMINAMATH_CALUDE_contrapositive_equality_l1885_188551

theorem contrapositive_equality (a b : ℝ) : 
  (¬(|a| = |b|) → ¬(a = -b)) ↔ (a = -b → |a| = |b|) := by sorry

end NUMINAMATH_CALUDE_contrapositive_equality_l1885_188551


namespace NUMINAMATH_CALUDE_megan_cupcakes_per_package_l1885_188549

/-- Calculates the number of cupcakes per package given the initial number of cupcakes,
    the number of cupcakes eaten, and the number of packages. -/
def cupcakes_per_package (initial : ℕ) (eaten : ℕ) (packages : ℕ) : ℕ :=
  (initial - eaten) / packages

/-- Proves that given 68 initial cupcakes, 32 cupcakes eaten, and 6 packages,
    the number of cupcakes in each package is 6. -/
theorem megan_cupcakes_per_package :
  cupcakes_per_package 68 32 6 = 6 := by
  sorry

end NUMINAMATH_CALUDE_megan_cupcakes_per_package_l1885_188549


namespace NUMINAMATH_CALUDE_geometric_progression_proof_l1885_188587

theorem geometric_progression_proof (x : ℝ) :
  (30 + x)^2 = (10 + x) * (90 + x) →
  x = 0 ∧ (30 + x) / (10 + x) = 3 := by
  sorry

end NUMINAMATH_CALUDE_geometric_progression_proof_l1885_188587


namespace NUMINAMATH_CALUDE_touching_sphere_surface_area_l1885_188511

/-- A sphere touching a rectangle and additional segments -/
structure TouchingSphere where
  -- The rectangle ABCD
  ab : ℝ
  bc : ℝ
  -- The segment EF
  ef : ℝ
  -- EF is parallel to the plane of ABCD
  ef_parallel : True
  -- All sides of ABCD and segments AE, BE, CF, DF, EF touch the sphere
  all_touch : True
  -- Given conditions
  ef_length : ef = 3
  bc_length : bc = 5

/-- The surface area of the sphere is 180π/7 -/
theorem touching_sphere_surface_area (s : TouchingSphere) : 
  ∃ (r : ℝ), 4 * Real.pi * r^2 = (180 * Real.pi) / 7 :=
sorry

end NUMINAMATH_CALUDE_touching_sphere_surface_area_l1885_188511


namespace NUMINAMATH_CALUDE_doughnuts_per_box_l1885_188589

theorem doughnuts_per_box (total_doughnuts : ℕ) (num_boxes : ℕ) (doughnuts_per_box : ℕ) : 
  total_doughnuts = 48 → 
  num_boxes = 4 → 
  total_doughnuts = num_boxes * doughnuts_per_box →
  doughnuts_per_box = 12 := by
  sorry

end NUMINAMATH_CALUDE_doughnuts_per_box_l1885_188589


namespace NUMINAMATH_CALUDE_correct_statements_l1885_188570

-- Define the proof methods
inductive ProofMethod
| Synthetic
| Analytic
| Contradiction

-- Define the characteristics of proof methods
def isCauseAndEffect (m : ProofMethod) : Prop := sorry
def isResultToCause (m : ProofMethod) : Prop := sorry
def isDirectProof (m : ProofMethod) : Prop := sorry

-- Define the statements
def statement1 : Prop := isCauseAndEffect ProofMethod.Synthetic
def statement2 : Prop := ¬(isDirectProof ProofMethod.Analytic)
def statement3 : Prop := isResultToCause ProofMethod.Analytic
def statement4 : Prop := isDirectProof ProofMethod.Contradiction

-- Theorem stating which statements are correct
theorem correct_statements :
  statement1 ∧ statement3 ∧ ¬statement2 ∧ ¬statement4 := by sorry

end NUMINAMATH_CALUDE_correct_statements_l1885_188570


namespace NUMINAMATH_CALUDE_equation_solution_l1885_188537

theorem equation_solution :
  ∃! x : ℚ, x ≠ 0 ∧ x ≠ 2 ∧ (2 * x) / (x - 2) - 2 = 1 / (x * (x - 2)) ∧ x = 1 / 4 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l1885_188537


namespace NUMINAMATH_CALUDE_intersection_M_notN_l1885_188510

-- Define the universal set U as ℝ
def U : Set ℝ := Set.univ

-- Define set M
def M : Set ℝ := {x | x^2 - 2*x - 3 ≤ 0}

-- Define set N
def N : Set ℝ := {y | ∃ x, y = x^2 + 1}

-- Define the complement of N in U
def notN : Set ℝ := U \ N

-- Theorem statement
theorem intersection_M_notN : M ∩ notN = Set.Icc (-1) 1 := by sorry

end NUMINAMATH_CALUDE_intersection_M_notN_l1885_188510


namespace NUMINAMATH_CALUDE_quadratic_equation_transformation_l1885_188523

theorem quadratic_equation_transformation (x : ℝ) : 
  ((x - 1) * (x + 1) = 1) ↔ (x^2 - 2 = 0) := by
sorry

end NUMINAMATH_CALUDE_quadratic_equation_transformation_l1885_188523


namespace NUMINAMATH_CALUDE_consistent_number_theorem_l1885_188505

def is_consistent_number (m : ℕ) : Prop :=
  ∃ (a b c d : ℕ), m = 1000 * a + 100 * b + 10 * c + d ∧
    a ≠ 0 ∧ b ≠ 0 ∧ c ≠ 0 ∧ d ≠ 0 ∧
    a + b = c + d

def F (m : ℕ) : ℚ :=
  let m' := (m / 10) % 10 * 1000 + m % 10 * 100 + m / 1000 * 10 + (m / 100) % 10
  (m + m') / 101

def G (N : ℕ) : ℕ := N / 10 + N % 10

theorem consistent_number_theorem :
  ∀ (m : ℕ), is_consistent_number m →
    let a := m / 1000
    let b := (m / 100) % 10
    let c := (m / 10) % 10
    let d := m % 10
    let N := 10 * a + 2 * b
    a ≤ 8 →
    d = 1 →
    Even (G N) →
    ∃ (k : ℤ), F m - G N - 4 * a = k^2 + 3 →
    (k = 6 ∨ k = -6) ∧ m = 2231 := by
  sorry

end NUMINAMATH_CALUDE_consistent_number_theorem_l1885_188505


namespace NUMINAMATH_CALUDE_milk_needed_for_cookies_l1885_188560

/-- The number of cookies in a dozen -/
def dozen : ℕ := 12

/-- The number of cookies that can be baked with 10 half-gallons of milk -/
def cookies_per_ten_halfgallons : ℕ := 40

/-- The number of half-gallons of milk needed for 40 cookies -/
def milk_for_forty_cookies : ℕ := 10

/-- The number of dozens of cookies to be baked -/
def dozens_to_bake : ℕ := 200

theorem milk_needed_for_cookies : 
  (dozens_to_bake * dozen * milk_for_forty_cookies) / cookies_per_ten_halfgallons = 600 := by
  sorry

end NUMINAMATH_CALUDE_milk_needed_for_cookies_l1885_188560


namespace NUMINAMATH_CALUDE_no_periodic_sum_l1885_188513

/-- A function is periodic if it takes at least two different values and there exists a positive real number p such that f(x + p) = f(x) for all x. -/
def Periodic (f : ℝ → ℝ) : Prop :=
  (∃ x y, f x ≠ f y) ∧ ∃ p > 0, ∀ x, f (x + p) = f x

/-- g is a periodic function with period 1 -/
def g : ℝ → ℝ := sorry

/-- h is a periodic function with period π -/
def h : ℝ → ℝ := sorry

/-- g has period 1 -/
axiom g_periodic : Periodic g ∧ ∀ x, g (x + 1) = g x

/-- h has period π -/
axiom h_periodic : Periodic h ∧ ∀ x, h (x + Real.pi) = h x

/-- Theorem: It's not possible to construct non-trivial periodic functions g and h
    with periods 1 and π respectively, such that g + h is also a periodic function -/
theorem no_periodic_sum : ¬(Periodic (g + h)) := by sorry

end NUMINAMATH_CALUDE_no_periodic_sum_l1885_188513


namespace NUMINAMATH_CALUDE_min_abs_plus_2023_min_value_abs_plus_2023_l1885_188593

theorem min_abs_plus_2023 (a : ℚ) : 
  (|a| + 2023 : ℚ) ≥ 2023 := by sorry

theorem min_value_abs_plus_2023 : 
  ∃ (m : ℚ), ∀ (a : ℚ), (|a| + 2023 : ℚ) ≥ m ∧ ∃ (b : ℚ), (|b| + 2023 : ℚ) = m := by
  use 2023
  sorry

end NUMINAMATH_CALUDE_min_abs_plus_2023_min_value_abs_plus_2023_l1885_188593


namespace NUMINAMATH_CALUDE_lexie_crayon_count_l1885_188590

/-- The number of crayons that can fit in each box -/
def crayons_per_box : ℕ := 8

/-- The number of crayon boxes Lexie needs -/
def number_of_boxes : ℕ := 10

/-- The total number of crayons Lexie has -/
def total_crayons : ℕ := crayons_per_box * number_of_boxes

theorem lexie_crayon_count : total_crayons = 80 := by
  sorry

end NUMINAMATH_CALUDE_lexie_crayon_count_l1885_188590


namespace NUMINAMATH_CALUDE_fraction_to_decimal_l1885_188524

theorem fraction_to_decimal : (47 : ℚ) / (2^3 * 5^4) = 0.0094 := by sorry

end NUMINAMATH_CALUDE_fraction_to_decimal_l1885_188524


namespace NUMINAMATH_CALUDE_eighth_week_hours_l1885_188527

def hours_worked : List ℕ := [9, 13, 8, 14, 12, 10, 11]
def total_weeks : ℕ := 8
def target_average : ℕ := 12

theorem eighth_week_hours : 
  ∃ x : ℕ, 
    (List.sum hours_worked + x) / total_weeks = target_average ∧ 
    x = 19 := by
  sorry

end NUMINAMATH_CALUDE_eighth_week_hours_l1885_188527


namespace NUMINAMATH_CALUDE_log_division_simplification_l1885_188569

theorem log_division_simplification : 
  Real.log 16 / Real.log (1/16) = -1 := by
  sorry

end NUMINAMATH_CALUDE_log_division_simplification_l1885_188569


namespace NUMINAMATH_CALUDE_perfect_squares_digit_parity_l1885_188503

def is_perfect_square (n : ℕ) : Prop := ∃ m : ℕ, n = m^2

def units_digit (n : ℕ) : ℕ := n % 10

def tens_digit (n : ℕ) : ℕ := (n / 10) % 10

theorem perfect_squares_digit_parity (a b x y : ℕ) :
  is_perfect_square a →
  is_perfect_square b →
  units_digit a = 1 →
  tens_digit a = x →
  units_digit b = 6 →
  tens_digit b = y →
  Even x ∧ Odd y :=
sorry

end NUMINAMATH_CALUDE_perfect_squares_digit_parity_l1885_188503


namespace NUMINAMATH_CALUDE_eighth_diagram_fully_shaded_l1885_188519

/-- The number of shaded squares in the n-th diagram -/
def shaded_squares (n : ℕ) : ℕ := n^2

/-- The total number of squares in the n-th diagram -/
def total_squares (n : ℕ) : ℕ := n^2

/-- The fraction of shaded squares in the n-th diagram -/
def shaded_fraction (n : ℕ) : ℚ :=
  (shaded_squares n : ℚ) / (total_squares n : ℚ)

theorem eighth_diagram_fully_shaded :
  shaded_fraction 8 = 1 := by sorry

end NUMINAMATH_CALUDE_eighth_diagram_fully_shaded_l1885_188519


namespace NUMINAMATH_CALUDE_collinear_points_k_value_l1885_188532

/-- Three points are collinear if they lie on the same straight line -/
def collinear (p1 p2 p3 : ℝ × ℝ) : Prop :=
  (p2.2 - p1.2) * (p3.1 - p2.1) = (p3.2 - p2.2) * (p2.1 - p1.1)

/-- The theorem stating that if (-2, -4), (5, k), and (15, 1) are collinear, then k = -33/17 -/
theorem collinear_points_k_value :
  ∀ k : ℝ, collinear (-2, -4) (5, k) (15, 1) → k = -33/17 := by
  sorry

end NUMINAMATH_CALUDE_collinear_points_k_value_l1885_188532


namespace NUMINAMATH_CALUDE_balance_problem_l1885_188597

/-- The problem of balancing weights on a scale --/
theorem balance_problem :
  let total_weight : ℝ := 4.5 -- in kg
  let num_weights : ℕ := 9
  let weight_per_item : ℝ := total_weight / num_weights -- in kg
  let pencil_case_weight : ℝ := 0.85 -- in kg
  let dictionary_weight : ℝ := 1.05 -- in kg
  let num_weights_on_scale : ℕ := 2
  let num_dictionaries : ℕ := 5
  ∃ (num_pencil_cases : ℕ),
    (num_weights_on_scale * weight_per_item + num_pencil_cases * pencil_case_weight) =
    (num_dictionaries * dictionary_weight) ∧
    num_pencil_cases = 5 :=
by sorry

end NUMINAMATH_CALUDE_balance_problem_l1885_188597


namespace NUMINAMATH_CALUDE_orange_juice_percentage_l1885_188581

theorem orange_juice_percentage (total : ℝ) (watermelon_percent : ℝ) (grape_ounces : ℝ) :
  total = 140 →
  watermelon_percent = 60 →
  grape_ounces = 35 →
  (15 : ℝ) / 100 * total = total - watermelon_percent / 100 * total - grape_ounces :=
by sorry

end NUMINAMATH_CALUDE_orange_juice_percentage_l1885_188581


namespace NUMINAMATH_CALUDE_min_value_of_z_l1885_188584

theorem min_value_of_z (x y : ℝ) (h1 : x - 1 ≥ 0) (h2 : 2 * x - y - 1 ≤ 0) (h3 : x + y - 3 ≤ 0) :
  ∃ (z : ℝ), z = x - y ∧ z ≥ -1 ∧ ∀ (w : ℝ), w = x - y → w ≥ z :=
sorry

end NUMINAMATH_CALUDE_min_value_of_z_l1885_188584


namespace NUMINAMATH_CALUDE_solution_k_l1885_188574

theorem solution_k (h : 2 * k - (-4) = 2) : k = -1 := by
  sorry

end NUMINAMATH_CALUDE_solution_k_l1885_188574


namespace NUMINAMATH_CALUDE_five_n_plus_three_composite_l1885_188535

theorem five_n_plus_three_composite (n : ℕ) 
  (h1 : ∃ k : ℕ, 2 * n + 1 = k^2) 
  (h2 : ∃ m : ℕ, 3 * n + 1 = m^2) : 
  ¬(Nat.Prime (5 * n + 3)) :=
sorry

end NUMINAMATH_CALUDE_five_n_plus_three_composite_l1885_188535


namespace NUMINAMATH_CALUDE_simplify_and_evaluate_l1885_188502

theorem simplify_and_evaluate : 
  let x : ℚ := 3/2
  (3 + x)^2 - (x + 5) * (x - 1) = 17 := by
sorry

end NUMINAMATH_CALUDE_simplify_and_evaluate_l1885_188502


namespace NUMINAMATH_CALUDE_digit_sum_theorem_l1885_188545

-- Define the conditions
def is_valid_digits (A B C : ℕ) : Prop :=
  A ≠ 0 ∧ B ≠ 0 ∧ C ≠ 0 ∧ A < 10 ∧ B < 10 ∧ C < 10

def BC (B C : ℕ) : ℕ := 10 * B + C

def ABC (A B C : ℕ) : ℕ := 100 * A + 10 * B + C

-- State the theorem
theorem digit_sum_theorem (A B C : ℕ) :
  is_valid_digits A B C →
  BC B C + ABC A B C + ABC A B C = 876 →
  A + B + C = 14 := by
sorry

end NUMINAMATH_CALUDE_digit_sum_theorem_l1885_188545


namespace NUMINAMATH_CALUDE_part_one_part_two_l1885_188573

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := a * x^2 - (a + 1) * x + 2

-- Part I
theorem part_one : 
  {x : ℝ | f 2 x > 1} = {x : ℝ | x < 1/2 ∨ x > 1} := by sorry

-- Part II
theorem part_two : 
  (∀ x ∈ Set.Icc (-1) 3, f a x ≥ 0) → 
  1/6 ≤ a ∧ a ≤ 3 + 2 * Real.sqrt 2 := by sorry

end NUMINAMATH_CALUDE_part_one_part_two_l1885_188573


namespace NUMINAMATH_CALUDE_simon_is_ten_l1885_188558

def alvin_age : ℕ := 30

def simon_age : ℕ := alvin_age / 2 - 5

theorem simon_is_ten : simon_age = 10 := by
  sorry

end NUMINAMATH_CALUDE_simon_is_ten_l1885_188558


namespace NUMINAMATH_CALUDE_product_difference_bound_l1885_188534

theorem product_difference_bound (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) 
  (h1 : x * y - z = x * z - y) (h2 : x * z - y = y * z - x) : 
  x * y - z ≥ -1/4 := by
sorry

end NUMINAMATH_CALUDE_product_difference_bound_l1885_188534


namespace NUMINAMATH_CALUDE_intersection_equality_implies_a_range_l1885_188576

-- Define sets A and B
def A : Set ℝ := {x | |x + 1| < 4}
def B (a : ℝ) : Set ℝ := {x | (x - 1) * (x - 2*a) < 0}

-- Theorem statement
theorem intersection_equality_implies_a_range (a : ℝ) :
  A ∩ B a = B a → a ∈ Set.Icc (-2.5) 1.5 := by
  sorry

end NUMINAMATH_CALUDE_intersection_equality_implies_a_range_l1885_188576


namespace NUMINAMATH_CALUDE_max_abs_z_purely_imaginary_l1885_188529

theorem max_abs_z_purely_imaginary (z : ℂ) :
  (∃ (t : ℝ), (z - Complex.I) / (z - 1) = Complex.I * t) → Complex.abs z ≤ Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_max_abs_z_purely_imaginary_l1885_188529


namespace NUMINAMATH_CALUDE_gamma_donuts_l1885_188533

/-- Proves that Gamma received 8 donuts given the conditions of the problem -/
theorem gamma_donuts : 
  ∀ (gamma_donuts : ℕ),
  (40 : ℕ) = 8 + 3 * gamma_donuts + gamma_donuts →
  gamma_donuts = 8 := by
sorry

end NUMINAMATH_CALUDE_gamma_donuts_l1885_188533


namespace NUMINAMATH_CALUDE_largest_prime_factor_of_1581_l1885_188530

def largest_prime_factor (n : ℕ) : ℕ := sorry

theorem largest_prime_factor_of_1581 : 
  largest_prime_factor 1581 = 113 := by sorry

end NUMINAMATH_CALUDE_largest_prime_factor_of_1581_l1885_188530


namespace NUMINAMATH_CALUDE_special_right_triangle_hypotenuse_l1885_188583

/-- A right triangle with specific leg relationship and area -/
structure SpecialRightTriangle where
  shorter_leg : ℝ
  longer_leg : ℝ
  hypotenuse : ℝ
  leg_relationship : longer_leg = 3 * shorter_leg - 3
  area_condition : (1 / 2) * shorter_leg * longer_leg = 108
  right_triangle : shorter_leg ^ 2 + longer_leg ^ 2 = hypotenuse ^ 2

/-- The hypotenuse of the special right triangle is √657 -/
theorem special_right_triangle_hypotenuse (t : SpecialRightTriangle) : t.hypotenuse = Real.sqrt 657 := by
  sorry

end NUMINAMATH_CALUDE_special_right_triangle_hypotenuse_l1885_188583


namespace NUMINAMATH_CALUDE_greatest_integer_difference_l1885_188506

theorem greatest_integer_difference (x y : ℝ) (hx : 4 < x ∧ x < 6) (hy : 6 < y ∧ y < 10) :
  (∀ (a b : ℝ), 4 < a ∧ a < 6 → 6 < b ∧ b < 10 → ⌊b - a⌋ ≤ 4) ∧ 
  (∃ (a b : ℝ), 4 < a ∧ a < 6 ∧ 6 < b ∧ b < 10 ∧ ⌊b - a⌋ = 4) :=
by sorry

end NUMINAMATH_CALUDE_greatest_integer_difference_l1885_188506


namespace NUMINAMATH_CALUDE_remainder_8354_mod_11_l1885_188585

theorem remainder_8354_mod_11 : 8354 % 11 = 6 := by
  sorry

end NUMINAMATH_CALUDE_remainder_8354_mod_11_l1885_188585


namespace NUMINAMATH_CALUDE_min_value_reciprocal_sum_l1885_188572

theorem min_value_reciprocal_sum (a b : ℝ) 
  (ha : a > 0) (hb : b > 0) 
  (h_geometric_mean : 4 = Real.sqrt (2^a * 2^b)) : 
  (∀ x y : ℝ, x > 0 → y > 0 → 1/x + 1/y ≥ 1/a + 1/b) → 1/a + 1/b = 1 := by
  sorry

end NUMINAMATH_CALUDE_min_value_reciprocal_sum_l1885_188572


namespace NUMINAMATH_CALUDE_admin_staff_selected_is_six_l1885_188515

/-- Represents the total number of staff members -/
def total_staff : ℕ := 200

/-- Represents the number of administrative staff members -/
def admin_staff : ℕ := 24

/-- Represents the total number of samples to be taken -/
def total_samples : ℕ := 50

/-- Calculates the number of administrative staff to be selected in a stratified sampling -/
def admin_staff_selected : ℕ := (admin_staff * total_samples) / total_staff

/-- Theorem stating that the number of administrative staff to be selected is 6 -/
theorem admin_staff_selected_is_six : admin_staff_selected = 6 := by
  sorry

end NUMINAMATH_CALUDE_admin_staff_selected_is_six_l1885_188515


namespace NUMINAMATH_CALUDE_perfect_square_condition_l1885_188595

theorem perfect_square_condition (m : ℝ) : 
  (∃ k : ℝ, ∀ x : ℝ, x^2 + m*x + 9 = (x + k)^2) → (m = 6 ∨ m = -6) :=
by sorry

end NUMINAMATH_CALUDE_perfect_square_condition_l1885_188595


namespace NUMINAMATH_CALUDE_square_area_from_diagonal_l1885_188582

theorem square_area_from_diagonal (x : ℝ) (h : x > 0) : 
  ∃ (s : ℝ), s > 0 ∧ s * s = x * x / 2 :=
by
  sorry

#check square_area_from_diagonal

end NUMINAMATH_CALUDE_square_area_from_diagonal_l1885_188582


namespace NUMINAMATH_CALUDE_f_properties_l1885_188504

-- Define the function f(x) = x^3 - 3x
def f (x : ℝ) : ℝ := x^3 - 3*x

-- Define the interval [-1, 3]
def interval : Set ℝ := Set.Icc (-1) 3

-- Theorem for monotonicity and extreme values
theorem f_properties :
  (∀ x y, x < y ∧ x < -1 → f x < f y) ∧  -- Increasing on (-∞, -1)
  (∀ x y, -1 < x ∧ x < y ∧ y < 1 → f x > f y) ∧  -- Decreasing on (-1, 1)
  (∀ x y, 1 < x ∧ x < y → f x < f y) ∧  -- Increasing on (1, +∞)
  (∀ x ∈ interval, f x ≤ 18) ∧  -- Maximum value
  (∀ x ∈ interval, f x ≥ -2) ∧  -- Minimum value
  (∃ x ∈ interval, f x = 18) ∧  -- Maximum is attained
  (∃ x ∈ interval, f x = -2) :=  -- Minimum is attained
by sorry

end NUMINAMATH_CALUDE_f_properties_l1885_188504


namespace NUMINAMATH_CALUDE_triangle_abc_properties_l1885_188579

open Real

theorem triangle_abc_properties (A B C : ℝ) (a b c : ℝ) :
  c = 2 →
  C = π / 3 →
  2 * sin (2 * A) + sin (2 * B + C) = sin C →
  (∃ S : ℝ, S = (2 * Real.sqrt 3) / 3 ∧ S = (1 / 2) * a * b * sin C) ∧
  (∃ P : ℝ, P ≤ 6 ∧ P = a + b + c) :=
by sorry

end NUMINAMATH_CALUDE_triangle_abc_properties_l1885_188579


namespace NUMINAMATH_CALUDE_triple_transmission_more_reliable_l1885_188562

/-- Represents a binary signal transmission channel with error probabilities α and β. -/
structure Channel where
  α : ℝ
  β : ℝ
  α_pos : 0 < α
  α_lt_half : α < 0.5
  α_lt_one : α < 1
  β_pos : 0 < β
  β_lt_one : β < 1

/-- Probability of correctly decoding a 0 signal using single transmission. -/
def singleTransmissionProb (c : Channel) : ℝ := 1 - c.α

/-- Probability of correctly decoding a 0 signal using triple transmission. -/
def tripleTransmissionProb (c : Channel) : ℝ :=
  3 * c.α * (1 - c.α)^2 + (1 - c.α)^3

/-- Theorem stating that triple transmission is more reliable than single transmission for decoding 0 signals when 0 < α < 0.5. -/
theorem triple_transmission_more_reliable (c : Channel) :
  tripleTransmissionProb c > singleTransmissionProb c := by
  sorry


end NUMINAMATH_CALUDE_triple_transmission_more_reliable_l1885_188562


namespace NUMINAMATH_CALUDE_intersection_A_B_l1885_188544

def A : Set ℝ := {x | x * (x - 2) < 0}
def B : Set ℝ := {-1, 0, 1, 2}

theorem intersection_A_B : A ∩ B = {1} := by sorry

end NUMINAMATH_CALUDE_intersection_A_B_l1885_188544


namespace NUMINAMATH_CALUDE_coat_price_proof_l1885_188594

theorem coat_price_proof (reduction : ℝ) (percentage : ℝ) (original_price : ℝ) : 
  reduction = 400 →
  percentage = 0.8 →
  percentage * original_price = reduction →
  original_price = 500 := by
sorry

end NUMINAMATH_CALUDE_coat_price_proof_l1885_188594


namespace NUMINAMATH_CALUDE_present_cost_difference_l1885_188508

theorem present_cost_difference (cost_first cost_second cost_third : ℕ) : 
  cost_first = 18 →
  cost_third = cost_first - 11 →
  cost_first + cost_second + cost_third = 50 →
  cost_second > cost_first →
  cost_second - cost_first = 7 := by
sorry

end NUMINAMATH_CALUDE_present_cost_difference_l1885_188508


namespace NUMINAMATH_CALUDE_sin_90_degrees_l1885_188557

theorem sin_90_degrees : Real.sin (π / 2) = 1 := by sorry

end NUMINAMATH_CALUDE_sin_90_degrees_l1885_188557


namespace NUMINAMATH_CALUDE_f_three_zeros_c_range_l1885_188568

-- Define the function f(x)
def f (c : ℝ) (x : ℝ) : ℝ := x^3 - 12*x + c

-- Define the derivative of f(x)
def f' (x : ℝ) : ℝ := 3*x^2 - 12

-- Theorem statement
theorem f_three_zeros_c_range (c : ℝ) :
  (∃ x₁ x₂ x₃ : ℝ, x₁ ≠ x₂ ∧ x₂ ≠ x₃ ∧ x₁ ≠ x₃ ∧ f c x₁ = 0 ∧ f c x₂ = 0 ∧ f c x₃ = 0) →
  -16 < c ∧ c < 16 := by sorry

end NUMINAMATH_CALUDE_f_three_zeros_c_range_l1885_188568


namespace NUMINAMATH_CALUDE_actual_miles_traveled_l1885_188592

/-- A function that counts the number of integers from 0 to n (inclusive) that contain the digit 3 --/
def countWithThree (n : ℕ) : ℕ := sorry

/-- The odometer reading --/
def odometerReading : ℕ := 3008

/-- Theorem stating that the actual miles traveled is 2465 when the odometer reads 3008 --/
theorem actual_miles_traveled :
  odometerReading - countWithThree odometerReading = 2465 := by sorry

end NUMINAMATH_CALUDE_actual_miles_traveled_l1885_188592


namespace NUMINAMATH_CALUDE_stating_total_seats_is_680_l1885_188599

/-- 
Calculates the total number of seats in a theater given the following conditions:
- The first row has 15 seats
- Each row has 2 more seats than the previous row
- The last row has 53 seats
-/
def theaterSeats : ℕ := by
  -- Define the number of seats in the first row
  let firstRow : ℕ := 15
  -- Define the increase in seats per row
  let seatIncrease : ℕ := 2
  -- Define the number of seats in the last row
  let lastRow : ℕ := 53
  
  -- Calculate the number of rows
  let numRows : ℕ := (lastRow - firstRow) / seatIncrease + 1
  
  -- Calculate the total number of seats
  let totalSeats : ℕ := numRows * (firstRow + lastRow) / 2
  
  exact totalSeats

/-- 
Theorem stating that the total number of seats in the theater is 680
-/
theorem total_seats_is_680 : theaterSeats = 680 := by
  sorry

end NUMINAMATH_CALUDE_stating_total_seats_is_680_l1885_188599


namespace NUMINAMATH_CALUDE_quadratic_factorization_l1885_188550

theorem quadratic_factorization (m : ℝ) : m^2 - 2*m + 1 = (m - 1)^2 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_factorization_l1885_188550


namespace NUMINAMATH_CALUDE_infinitely_many_good_numbers_good_not_divisible_by_seven_l1885_188531

/-- A natural number n is good if there exist natural numbers a and b
    such that a + b = n and ab | n^2 + n + 1 -/
def is_good (n : ℕ) : Prop :=
  ∃ a b : ℕ, a + b = n ∧ (n^2 + n + 1) % (a * b) = 0

theorem infinitely_many_good_numbers :
  ∀ k : ℕ, ∃ n : ℕ, n > k ∧ is_good n :=
sorry

theorem good_not_divisible_by_seven :
  ∀ n : ℕ, is_good n → ¬(7 ∣ n) :=
sorry

end NUMINAMATH_CALUDE_infinitely_many_good_numbers_good_not_divisible_by_seven_l1885_188531


namespace NUMINAMATH_CALUDE_competition_scores_l1885_188556

theorem competition_scores (n k : ℕ) : 
  n ≥ 2 → 
  k > 0 → 
  k * (n * (n + 1) / 2) = 26 * n → 
  ((n = 25 ∧ k = 2) ∨ (n = 12 ∧ k = 4) ∨ (n = 3 ∧ k = 13)) :=
by sorry

end NUMINAMATH_CALUDE_competition_scores_l1885_188556


namespace NUMINAMATH_CALUDE_remainder_of_binary_number_div_4_l1885_188563

def binary_number : ℕ := 0b111001101101

theorem remainder_of_binary_number_div_4 :
  binary_number % 4 = 1 := by sorry

end NUMINAMATH_CALUDE_remainder_of_binary_number_div_4_l1885_188563


namespace NUMINAMATH_CALUDE_cylinder_base_area_l1885_188512

/-- Represents a container with a base area and height increase when a stone is submerged -/
structure Container where
  base_area : ℝ
  height_increase : ℝ

/-- Proves that the base area of the cylinder is 42 square centimeters -/
theorem cylinder_base_area
  (cylinder : Container)
  (prism : Container)
  (h1 : cylinder.height_increase = 8)
  (h2 : prism.height_increase = 6)
  (h3 : cylinder.base_area + prism.base_area = 98)
  : cylinder.base_area = 42 := by
  sorry

end NUMINAMATH_CALUDE_cylinder_base_area_l1885_188512


namespace NUMINAMATH_CALUDE_product_divide_theorem_l1885_188521

theorem product_divide_theorem : (3.6 * 0.3) / 0.6 = 1.8 := by
  sorry

end NUMINAMATH_CALUDE_product_divide_theorem_l1885_188521


namespace NUMINAMATH_CALUDE_tan_eleven_pi_thirds_l1885_188509

theorem tan_eleven_pi_thirds : Real.tan (11 * Real.pi / 3) = -Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_tan_eleven_pi_thirds_l1885_188509


namespace NUMINAMATH_CALUDE_min_value_of_function_l1885_188500

theorem min_value_of_function (x : ℝ) (h : x > 0) : 
  8 + x/2 + 2/x ≥ 10 ∧ ∃ y > 0, 8 + y/2 + 2/y = 10 := by
  sorry

end NUMINAMATH_CALUDE_min_value_of_function_l1885_188500


namespace NUMINAMATH_CALUDE_seventh_term_l1885_188554

/-- A geometric sequence {a_n} with a_3 = 3 and a_6 = 24 -/
def geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ (r : ℝ), r ≠ 0 ∧ (∀ n, a (n + 1) = r * a n) ∧ a 3 = 3 ∧ a 6 = 24

/-- The 7th term of the geometric sequence is 48 -/
theorem seventh_term (a : ℕ → ℝ) (h : geometric_sequence a) : a 7 = 48 := by
  sorry

end NUMINAMATH_CALUDE_seventh_term_l1885_188554


namespace NUMINAMATH_CALUDE_triangle_existence_condition_l1885_188539

theorem triangle_existence_condition 
  (k : ℝ) (α : ℝ) (m_a : ℝ) 
  (h_k : k > 0) 
  (h_α : 0 < α ∧ α < π) 
  (h_m_a : m_a > 0) : 
  (∃ (a b c : ℝ), 
    a > 0 ∧ b > 0 ∧ c > 0 ∧ 
    a + b + c = k ∧
    ∃ (β γ : ℝ), 
      0 < β ∧ 0 < γ ∧
      α + β + γ = π ∧
      m_a = (b * c * Real.sin α) / (b + c)) ↔ 
  m_a ≤ (k / 2) * ((1 - Real.sin (α / 2)) / Real.cos (α / 2)) :=
sorry

end NUMINAMATH_CALUDE_triangle_existence_condition_l1885_188539


namespace NUMINAMATH_CALUDE_weight_of_replaced_person_l1885_188542

theorem weight_of_replaced_person (initial_count : ℕ) (avg_increase : ℚ) (new_weight : ℚ) :
  initial_count = 6 →
  avg_increase = 4.5 →
  new_weight = 102 →
  ∃ (old_weight : ℚ), old_weight = 75 ∧ new_weight = old_weight + initial_count * avg_increase :=
by sorry

end NUMINAMATH_CALUDE_weight_of_replaced_person_l1885_188542


namespace NUMINAMATH_CALUDE_a_equals_two_sufficient_not_necessary_l1885_188516

theorem a_equals_two_sufficient_not_necessary (a : ℝ) :
  (a = 2 → |a| = 2) ∧ (∃ b : ℝ, b ≠ 2 ∧ |b| = 2) := by
  sorry

end NUMINAMATH_CALUDE_a_equals_two_sufficient_not_necessary_l1885_188516


namespace NUMINAMATH_CALUDE_apple_students_count_l1885_188548

/-- Represents the total number of degrees in a circle -/
def total_degrees : ℕ := 360

/-- Represents the number of degrees in a right angle -/
def right_angle : ℕ := 90

/-- Represents the number of students who chose bananas -/
def banana_students : ℕ := 168

/-- Calculates the number of students who chose apples given the conditions -/
def apple_students : ℕ :=
  (right_angle * (banana_students * 4 / 3)) / total_degrees

theorem apple_students_count : apple_students = 56 := by
  sorry

end NUMINAMATH_CALUDE_apple_students_count_l1885_188548


namespace NUMINAMATH_CALUDE_weekend_weather_probability_l1885_188564

/-- The probability of rain on each day -/
def rain_prob : ℝ := 0.75

/-- The number of days in the weekend -/
def num_days : ℕ := 3

/-- The number of desired sunny days -/
def desired_sunny_days : ℕ := 2

/-- Theorem: The probability of having exactly two sunny days and one rainy day
    during a three-day period, where the probability of rain each day is 0.75,
    is equal to 27/64 -/
theorem weekend_weather_probability :
  (Nat.choose num_days desired_sunny_days : ℝ) *
  (1 - rain_prob) ^ desired_sunny_days *
  rain_prob ^ (num_days - desired_sunny_days) =
  27 / 64 := by sorry

end NUMINAMATH_CALUDE_weekend_weather_probability_l1885_188564


namespace NUMINAMATH_CALUDE_defective_units_percentage_l1885_188541

theorem defective_units_percentage
  (shipped_defective_ratio : Real)
  (total_shipped_defective_ratio : Real)
  (h1 : shipped_defective_ratio = 0.04)
  (h2 : total_shipped_defective_ratio = 0.0036) :
  ∃ (defective_ratio : Real),
    defective_ratio * shipped_defective_ratio = total_shipped_defective_ratio ∧
    defective_ratio = 0.09 := by
  sorry

end NUMINAMATH_CALUDE_defective_units_percentage_l1885_188541


namespace NUMINAMATH_CALUDE_jim_tire_repairs_l1885_188517

/-- Represents the financial details of Jim's bike shop for a month. -/
structure BikeShopFinances where
  tireFee : ℕ           -- Fee charged for fixing a bike tire
  tireCost : ℕ          -- Cost of parts for fixing a bike tire
  complexRepairs : ℕ    -- Number of complex repairs
  complexFee : ℕ        -- Fee charged for a complex repair
  complexCost : ℕ       -- Cost of parts for a complex repair
  retailProfit : ℕ      -- Profit from retail sales
  fixedExpenses : ℕ     -- Monthly fixed expenses
  totalProfit : ℕ       -- Total profit for the month

/-- Calculates the number of bike tire repairs given the shop's finances. -/
def calculateTireRepairs (finances : BikeShopFinances) : ℕ :=
  let tireProfit := finances.tireFee - finances.tireCost
  let complexProfit := finances.complexRepairs * (finances.complexFee - finances.complexCost)
  (finances.totalProfit + finances.fixedExpenses - finances.retailProfit - complexProfit) / tireProfit

/-- Theorem stating that Jim does 300 bike tire repairs in a month. -/
theorem jim_tire_repairs : 
  let finances : BikeShopFinances := {
    tireFee := 20
    tireCost := 5
    complexRepairs := 2
    complexFee := 300
    complexCost := 50
    retailProfit := 2000
    fixedExpenses := 4000
    totalProfit := 3000
  }
  calculateTireRepairs finances = 300 := by
  sorry


end NUMINAMATH_CALUDE_jim_tire_repairs_l1885_188517


namespace NUMINAMATH_CALUDE_correct_num_boys_l1885_188501

/-- The number of trees -/
def total_trees : ℕ := 29

/-- The number of trees left unwatered -/
def unwatered_trees : ℕ := 2

/-- The number of boys who went to water the trees -/
def num_boys : ℕ := 3

/-- Theorem stating that the number of boys is correct -/
theorem correct_num_boys :
  ∃ (trees_per_boy : ℕ), 
    num_boys * trees_per_boy = total_trees - unwatered_trees ∧ 
    trees_per_boy > 0 :=
sorry

end NUMINAMATH_CALUDE_correct_num_boys_l1885_188501


namespace NUMINAMATH_CALUDE_investment_period_proof_l1885_188538

/-- Calculates the simple interest given principal, rate, and time -/
def simpleInterest (principal : ℝ) (rate : ℝ) (time : ℝ) : ℝ :=
  principal * rate * time

theorem investment_period_proof (principal : ℝ) (rate1 rate2 : ℝ) (time : ℝ) 
    (h1 : principal = 900)
    (h2 : rate1 = 0.04)
    (h3 : rate2 = 0.045)
    (h4 : simpleInterest principal rate2 time - simpleInterest principal rate1 time = 31.5) :
  time = 7 := by
  sorry

end NUMINAMATH_CALUDE_investment_period_proof_l1885_188538


namespace NUMINAMATH_CALUDE_block_stacks_height_difference_main_theorem_l1885_188552

/-- Proves that the height difference between the final stack and the second stack is 7 blocks -/
theorem block_stacks_height_difference : ℕ → ℕ → ℕ → ℕ → ℕ → Prop :=
  fun (first_stack : ℕ) (second_stack : ℕ) (final_stack : ℕ) (fallen_blocks : ℕ) (height_diff : ℕ) =>
    first_stack = 7 ∧
    second_stack = first_stack + 5 ∧
    final_stack = second_stack + height_diff ∧
    fallen_blocks = first_stack + (second_stack - 2) + (final_stack - 3) ∧
    fallen_blocks = 33 →
    height_diff = 7

/-- The main theorem stating that the height difference is 7 blocks -/
theorem main_theorem : ∃ (first_stack second_stack final_stack fallen_blocks : ℕ),
  block_stacks_height_difference first_stack second_stack final_stack fallen_blocks 7 :=
sorry

end NUMINAMATH_CALUDE_block_stacks_height_difference_main_theorem_l1885_188552

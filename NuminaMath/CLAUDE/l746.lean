import Mathlib

namespace NUMINAMATH_CALUDE_racing_game_cost_l746_74630

/-- Given that Joan spent $9.43 on video games in total and
    purchased a basketball game for $5.2, prove that the
    cost of the racing game is $9.43 - $5.2. -/
theorem racing_game_cost (total_spent : ℝ) (basketball_cost : ℝ)
    (h1 : total_spent = 9.43)
    (h2 : basketball_cost = 5.2) :
    total_spent - basketball_cost = 9.43 - 5.2 := by
  sorry

end NUMINAMATH_CALUDE_racing_game_cost_l746_74630


namespace NUMINAMATH_CALUDE_x_range_l746_74608

theorem x_range (x : ℝ) (h1 : (1 : ℝ) / x < 3) (h2 : (1 : ℝ) / x > -2) :
  x > (1 : ℝ) / 3 ∨ x < -(1 : ℝ) / 2 := by
  sorry

end NUMINAMATH_CALUDE_x_range_l746_74608


namespace NUMINAMATH_CALUDE_exists_monochromatic_trapezoid_l746_74627

/-- A color is represented as a natural number -/
def Color := ℕ

/-- A point on a circle -/
structure CirclePoint where
  angle : ℝ
  color : Color

/-- A circle with colored points -/
structure ColoredCircle where
  points : Set CirclePoint
  num_colors : ℕ
  color_bound : num_colors ≥ 2

/-- A trapezoid inscribed in a circle -/
structure InscribedTrapezoid where
  p1 : CirclePoint
  p2 : CirclePoint
  p3 : CirclePoint
  p4 : CirclePoint
  trapezoid_condition : (p2.angle - p1.angle) = (p4.angle - p3.angle)

/-- The main theorem -/
theorem exists_monochromatic_trapezoid (c : ColoredCircle) :
  ∃ t : InscribedTrapezoid, 
    t.p1 ∈ c.points ∧ 
    t.p2 ∈ c.points ∧ 
    t.p3 ∈ c.points ∧ 
    t.p4 ∈ c.points ∧
    t.p1.color = t.p2.color ∧ 
    t.p2.color = t.p3.color ∧ 
    t.p3.color = t.p4.color :=
  sorry

end NUMINAMATH_CALUDE_exists_monochromatic_trapezoid_l746_74627


namespace NUMINAMATH_CALUDE_score_difference_l746_74607

def score_distribution : List (ℝ × ℝ) := [
  (0.15, 60),
  (0.25, 75),
  (0.35, 85),
  (0.20, 95),
  (0.05, 110)
]

def median_score : ℝ := 85

def mean_score : ℝ := (score_distribution.map (λ (p, s) => p * s)).sum

theorem score_difference : median_score - mean_score = 3 := by sorry

end NUMINAMATH_CALUDE_score_difference_l746_74607


namespace NUMINAMATH_CALUDE_gadget_production_l746_74673

/-- Represents the time (in hours) required for one worker to produce one gizmo -/
def gizmo_time : ℚ := sorry

/-- Represents the time (in hours) required for one worker to produce one gadget -/
def gadget_time : ℚ := sorry

/-- The number of gadgets produced by 30 workers in 4 hours -/
def n : ℕ := sorry

theorem gadget_production :
  -- In 1 hour, 80 workers produce 200 gizmos and 160 gadgets
  80 * (200 * gizmo_time + 160 * gadget_time) = 1 →
  -- In 2 hours, 40 workers produce 160 gizmos and 240 gadgets
  40 * (160 * gizmo_time + 240 * gadget_time) = 2 →
  -- In 4 hours, 30 workers produce 120 gizmos and n gadgets
  30 * (120 * gizmo_time + n * gadget_time) = 4 →
  -- The number of gadgets produced by 30 workers in 4 hours is 135680
  n = 135680 := by
  sorry

end NUMINAMATH_CALUDE_gadget_production_l746_74673


namespace NUMINAMATH_CALUDE_last_term_before_one_l746_74692

def arithmetic_sequence (a : ℝ) (d : ℝ) (n : ℕ) : ℝ := a + (n - 1 : ℝ) * d

theorem last_term_before_one (a : ℝ) (d : ℝ) (n : ℕ) :
  a = 100 ∧ d = -4 →
  arithmetic_sequence a d 25 > 1 ∧
  arithmetic_sequence a d 26 ≤ 1 := by
sorry

end NUMINAMATH_CALUDE_last_term_before_one_l746_74692


namespace NUMINAMATH_CALUDE_megan_deleted_files_l746_74645

/-- Calculates the number of deleted files given the initial number of files,
    number of folders after organizing, and number of files per folder. -/
def deleted_files (initial_files : ℕ) (num_folders : ℕ) (files_per_folder : ℕ) : ℕ :=
  initial_files - (num_folders * files_per_folder)

/-- Proves that Megan deleted 21 files given the problem conditions. -/
theorem megan_deleted_files :
  deleted_files 93 9 8 = 21 := by
  sorry

end NUMINAMATH_CALUDE_megan_deleted_files_l746_74645


namespace NUMINAMATH_CALUDE_min_real_roots_l746_74613

/-- A polynomial of degree 10 with real coefficients -/
structure Polynomial10 where
  coeffs : Fin 11 → ℝ
  lead_coeff_nonzero : coeffs 10 ≠ 0

/-- The roots of a polynomial -/
def roots (p : Polynomial10) : Multiset ℂ := sorry

/-- The number of distinct absolute values among the roots -/
def distinct_abs_values (p : Polynomial10) : ℕ := sorry

/-- The number of real roots of a polynomial -/
def num_real_roots (p : Polynomial10) : ℕ := sorry

/-- If a polynomial of degree 10 with real coefficients has exactly 6 distinct absolute values
    among its roots, then it has at least 3 real roots -/
theorem min_real_roots (p : Polynomial10) :
  distinct_abs_values p = 6 → num_real_roots p ≥ 3 := by sorry

end NUMINAMATH_CALUDE_min_real_roots_l746_74613


namespace NUMINAMATH_CALUDE_locus_of_points_with_constant_sum_of_distances_l746_74684

/-- Represents a line in a plane -/
structure Line where
  -- Add necessary fields for a line

/-- Represents a point in a plane -/
structure Point where
  -- Add necessary fields for a point

/-- Perpendicular distance between a point and a line -/
def perpendicularDistance (p : Point) (l : Line) : ℝ := sorry

/-- Check if two lines are parallel -/
def areParallel (l1 l2 : Line) : Prop := sorry

/-- The locus of points satisfying the given conditions -/
inductive Locus
  | Empty
  | Parallelogram
  | CentrallySymmetricOctagon

/-- The main theorem statement -/
theorem locus_of_points_with_constant_sum_of_distances
  (L₁ L₂ L₃ L₄ : Line)
  (h_distinct : L₁ ≠ L₂ ∧ L₁ ≠ L₃ ∧ L₁ ≠ L₄ ∧ L₂ ≠ L₃ ∧ L₂ ≠ L₄ ∧ L₃ ≠ L₄)
  (h_parallel₁ : areParallel L₁ L₃)
  (h_parallel₂ : areParallel L₂ L₄)
  (D : ℝ)
  (x : ℝ) -- perpendicular distance between L₁ and L₃
  (y : ℝ) -- perpendicular distance between L₂ and L₄
  (h_x : ∀ (p : Point), perpendicularDistance p L₁ + perpendicularDistance p L₃ = x)
  (h_y : ∀ (p : Point), perpendicularDistance p L₂ + perpendicularDistance p L₄ = y) :
  (∀ (p : Point), 
    perpendicularDistance p L₁ + perpendicularDistance p L₂ + 
    perpendicularDistance p L₃ + perpendicularDistance p L₄ = D) →
  Locus :=
by sorry

end NUMINAMATH_CALUDE_locus_of_points_with_constant_sum_of_distances_l746_74684


namespace NUMINAMATH_CALUDE_spider_web_problem_l746_74604

theorem spider_web_problem (S : ℕ) : 
  (∃ (W D : ℕ), 
    S = W ∧              -- Number of spiders equals number of webs made by each spider
    S = D ∧              -- Number of spiders equals number of days taken
    7 * S = W * D) →     -- Relationship between 1 spider making 1 web in 7 days
  S = 7 := by
sorry

end NUMINAMATH_CALUDE_spider_web_problem_l746_74604


namespace NUMINAMATH_CALUDE_correct_ages_l746_74654

/-- Teacher Zhang's current age -/
def zhang_age : ℕ := sorry

/-- Wang Bing's current age -/
def wang_age : ℕ := sorry

/-- The relationship between Teacher Zhang's and Wang Bing's ages -/
axiom age_relation : zhang_age = 3 * wang_age + 4

/-- The relationship between their ages 10 years ago and 10 years from now -/
axiom age_time_relation : zhang_age - 10 = wang_age + 10

/-- Theorem stating the correct ages -/
theorem correct_ages : zhang_age = 28 ∧ wang_age = 8 := by sorry

end NUMINAMATH_CALUDE_correct_ages_l746_74654


namespace NUMINAMATH_CALUDE_chris_savings_proof_l746_74600

def chris_birthday_savings (grandmother_gift aunt_uncle_gift parents_gift chores_money friend_gift_cost total_after : ℝ) : Prop :=
  let total_received := grandmother_gift + aunt_uncle_gift + parents_gift + chores_money
  let additional_amount := total_received - friend_gift_cost
  let savings_before := total_after - additional_amount
  let percentage_increase := (additional_amount / savings_before) * 100
  savings_before = 144 ∧ percentage_increase = 93.75

theorem chris_savings_proof :
  chris_birthday_savings 25 20 75 30 15 279 :=
sorry

end NUMINAMATH_CALUDE_chris_savings_proof_l746_74600


namespace NUMINAMATH_CALUDE_angle_side_inequality_l746_74652

-- Define a structure for triangles
structure Triangle where
  A : Real
  B : Real
  C : Real
  a : Real
  b : Real
  c : Real

-- Define the property that larger angles are opposite longer sides
axiom larger_angle_longer_side {t : Triangle} : 
  ∀ (x y : Real), (x = t.A ∧ y = t.a) ∨ (x = t.B ∧ y = t.b) ∨ (x = t.C ∧ y = t.c) →
  ∀ (p q : Real), (p = t.A ∧ q = t.a) ∨ (p = t.B ∧ q = t.b) ∨ (p = t.C ∧ q = t.c) →
  x > p → y > q

-- Theorem statement
theorem angle_side_inequality (t : Triangle) : t.A < t.B → t.a < t.b := by
  sorry

end NUMINAMATH_CALUDE_angle_side_inequality_l746_74652


namespace NUMINAMATH_CALUDE_function_inequality_l746_74670

open Real

/-- Given two differentiable functions f and g on ℝ, if f'(x) > g'(x) for all x,
    then for a < x < b, we have f(x) + g(b) < g(x) + f(b) and f(x) + g(a) > g(x) + f(a) -/
theorem function_inequality (f g : ℝ → ℝ) (hf : Differentiable ℝ f) (hg : Differentiable ℝ g)
    (h_deriv : ∀ x, deriv f x > deriv g x) (a b x : ℝ) (h_x : a < x ∧ x < b) :
    (f x + g b < g x + f b) ∧ (f x + g a > g x + f a) := by
  sorry

end NUMINAMATH_CALUDE_function_inequality_l746_74670


namespace NUMINAMATH_CALUDE_linear_regression_intercept_l746_74698

theorem linear_regression_intercept 
  (x_mean y_mean b a : ℝ) 
  (h1 : y_mean = b * x_mean + a) 
  (h2 : b = 0.51) 
  (h3 : x_mean = 61.75) 
  (h4 : y_mean = 38.14) : 
  a = 6.65 := by sorry

end NUMINAMATH_CALUDE_linear_regression_intercept_l746_74698


namespace NUMINAMATH_CALUDE_complement_union_theorem_l746_74637

-- Define the universal set U
def U : Set Nat := {1, 2, 3, 4, 5}

-- Define set A
def A : Set Nat := {1, 2, 3}

-- Define set B
def B : Set Nat := {2, 3, 4}

-- Theorem to prove
theorem complement_union_theorem : 
  (U \ A) ∪ (U \ B) = {1, 4, 5} := by sorry

end NUMINAMATH_CALUDE_complement_union_theorem_l746_74637


namespace NUMINAMATH_CALUDE_product_of_real_parts_of_roots_l746_74641

-- Define the complex number i
def i : ℂ := Complex.I

-- Define the polynomial
def f (z : ℂ) : ℂ := z^2 - z - (10 - 6*i)

-- State the theorem
theorem product_of_real_parts_of_roots :
  ∃ (z₁ z₂ : ℂ), f z₁ = 0 ∧ f z₂ = 0 ∧ (z₁.re * z₂.re = -47/4) :=
sorry

end NUMINAMATH_CALUDE_product_of_real_parts_of_roots_l746_74641


namespace NUMINAMATH_CALUDE_forester_tree_planting_l746_74628

/-- A forester's tree planting problem --/
theorem forester_tree_planting (initial_trees : ℕ) (total_goal : ℕ) : 
  initial_trees = 30 →
  total_goal = 300 →
  let monday_planted := 2 * initial_trees
  let tuesday_planted := monday_planted / 3
  let wednesday_planted := 2 * tuesday_planted
  let total_planted := monday_planted + tuesday_planted + wednesday_planted
  total_planted = 120 ∧ initial_trees + total_planted = total_goal := by
  sorry

end NUMINAMATH_CALUDE_forester_tree_planting_l746_74628


namespace NUMINAMATH_CALUDE_jerry_earnings_l746_74622

/-- Calculates the total earnings for an independent contractor over a week -/
def total_earnings (pay_per_task : ℕ) (hours_per_task : ℕ) (hours_per_day : ℕ) (days_per_week : ℕ) : ℕ :=
  (pay_per_task * hours_per_day * days_per_week) / hours_per_task

/-- Proves that Jerry's total earnings for the week equal $1400 -/
theorem jerry_earnings :
  total_earnings 40 2 10 7 = 1400 := by
  sorry

end NUMINAMATH_CALUDE_jerry_earnings_l746_74622


namespace NUMINAMATH_CALUDE_PQRS_equals_negative_one_l746_74666

theorem PQRS_equals_negative_one :
  let P : ℝ := Real.sqrt 2007 + Real.sqrt 2008
  let Q : ℝ := -Real.sqrt 2007 - Real.sqrt 2008
  let R : ℝ := Real.sqrt 2007 - Real.sqrt 2008
  let S : ℝ := -Real.sqrt 2008 + Real.sqrt 2007
  P * Q * R * S = -1 := by sorry

end NUMINAMATH_CALUDE_PQRS_equals_negative_one_l746_74666


namespace NUMINAMATH_CALUDE_two_color_theorem_l746_74648

/-- A circle in a plane --/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

/-- A region in the plane --/
inductive Region
  | Inside (n : ℕ) -- Inside n circles
  | Outside        -- Outside all circles

/-- The type of coloring function --/
def Coloring := Region → Fin 2

/-- Two regions are adjacent if they differ by crossing one circle boundary --/
def adjacent (r1 r2 : Region) : Prop :=
  match r1, r2 with
  | Region.Inside n, Region.Inside m => n + 1 = m ∨ m + 1 = n
  | Region.Inside 1, Region.Outside => True
  | Region.Outside, Region.Inside 1 => True
  | _, _ => False

/-- A coloring is valid if adjacent regions have different colors --/
def valid_coloring (c : Coloring) : Prop :=
  ∀ r1 r2, adjacent r1 r2 → c r1 ≠ c r2

theorem two_color_theorem (circles : List Circle) :
  ∃ c : Coloring, valid_coloring c :=
sorry

end NUMINAMATH_CALUDE_two_color_theorem_l746_74648


namespace NUMINAMATH_CALUDE_casino_table_ratio_l746_74631

/-- Proves that the ratio of money on table B to table C is 2 given the casino table conditions -/
theorem casino_table_ratio : 
  ∀ (A B C : ℝ),
  A = 40 →
  C = A + 20 →
  A + B + C = 220 →
  B / C = 2 := by
sorry

end NUMINAMATH_CALUDE_casino_table_ratio_l746_74631


namespace NUMINAMATH_CALUDE_complex_number_problem_l746_74619

theorem complex_number_problem (a : ℝ) (h : a > 0) : 
  let z : ℂ := (a - Complex.I) / (1 - Complex.I)
  let ω : ℂ := z * (z + Complex.I)
  Complex.im ω - Complex.re ω = (3 : ℝ) / 2 → a = 2 := by
sorry

end NUMINAMATH_CALUDE_complex_number_problem_l746_74619


namespace NUMINAMATH_CALUDE_optimal_removal_l746_74691

-- Define the grid
inductive Square
| a | b | c | d | e | f | g | j | k | l | m | n

-- Define the initial shape
def initial_shape : Set Square :=
  {Square.a, Square.b, Square.c, Square.d, Square.e, Square.f, Square.g, Square.j, Square.k, Square.l, Square.m, Square.n}

-- Define adjacency relation
def adjacent : Square → Square → Prop := sorry

-- Define connectivity
def is_connected (shape : Set Square) : Prop := sorry

-- Define perimeter calculation
def perimeter (shape : Set Square) : ℕ := sorry

-- Define the set of all possible pairs of squares to remove
def removable_pairs : Set (Square × Square) := sorry

theorem optimal_removal :
  ∀ (pair : Square × Square),
    pair ∈ removable_pairs →
    is_connected (initial_shape \ {pair.1, pair.2}) →
    perimeter (initial_shape \ {pair.1, pair.2}) ≤ 
    max (perimeter (initial_shape \ {Square.d, Square.k}))
        (perimeter (initial_shape \ {Square.e, Square.k})) :=
sorry

end NUMINAMATH_CALUDE_optimal_removal_l746_74691


namespace NUMINAMATH_CALUDE_dress_cost_theorem_l746_74635

/-- The total cost of dresses for Patty, Ida, Jean, and Pauline -/
def total_cost (patty ida jean pauline : ℕ) : ℕ := patty + ida + jean + pauline

/-- Theorem stating the total cost of dresses given the conditions -/
theorem dress_cost_theorem :
  ∀ (patty ida jean pauline : ℕ),
    patty = ida + 10 →
    ida = jean + 30 →
    jean = pauline - 10 →
    pauline = 30 →
    total_cost patty ida jean pauline = 160 := by
  sorry

end NUMINAMATH_CALUDE_dress_cost_theorem_l746_74635


namespace NUMINAMATH_CALUDE_equation_solution_l746_74657

theorem equation_solution : 
  ∃ (x₁ x₂ : ℝ), x₁ > 0 ∧ x₂ > 0 ∧ 
  x₁ = (40 + Real.sqrt 1636) / 2 ∧ 
  x₂ = (-20 + Real.sqrt 388) / 2 ∧
  ∀ (x : ℝ), x > 0 → 
  ((3 / 5) * (2 * x^2 - 2) = (x^2 - 40*x - 8) * (x^2 + 20*x + 4)) ↔ 
  (x = x₁ ∨ x = x₂) :=
by sorry

end NUMINAMATH_CALUDE_equation_solution_l746_74657


namespace NUMINAMATH_CALUDE_product_expansion_sum_l746_74611

theorem product_expansion_sum (a b c d : ℝ) :
  (∀ x, (4*x^2 - 6*x + 5) * (8 - 3*x) = a*x^3 + b*x^2 + c*x + d) →
  8*a + 4*b + 2*c + d = 18 := by
sorry

end NUMINAMATH_CALUDE_product_expansion_sum_l746_74611


namespace NUMINAMATH_CALUDE_complex_magnitude_proof_l746_74660

theorem complex_magnitude_proof : ∀ (z : ℂ), 
  z = (3 : ℝ) / 4 - (5 : ℝ) / 6 * I → Complex.abs z = Real.sqrt 181 / 12 := by
  sorry

end NUMINAMATH_CALUDE_complex_magnitude_proof_l746_74660


namespace NUMINAMATH_CALUDE_polynomial_nonnegative_iff_equal_roots_l746_74621

theorem polynomial_nonnegative_iff_equal_roots (a b c : ℝ) :
  (∀ x : ℝ, (x - a) * (x - b) + (x - b) * (x - c) + (x - c) * (x - a) ≥ 0) ↔ 
  (a = b ∧ b = c) :=
by sorry

end NUMINAMATH_CALUDE_polynomial_nonnegative_iff_equal_roots_l746_74621


namespace NUMINAMATH_CALUDE_cos_2B_gt_cos_2A_necessary_not_sufficient_l746_74676

-- Define a structure for a triangle
structure Triangle where
  A : Real
  B : Real
  C : Real
  sum_angles : A + B + C = π
  positive_angles : 0 < A ∧ 0 < B ∧ 0 < C

-- Define the main theorem
theorem cos_2B_gt_cos_2A_necessary_not_sufficient (t : Triangle) :
  (∀ t : Triangle, t.A > t.B → Real.cos (2 * t.B) > Real.cos (2 * t.A)) ∧
  ¬(∀ t : Triangle, Real.cos (2 * t.B) > Real.cos (2 * t.A) → t.A > t.B) := by
  sorry

end NUMINAMATH_CALUDE_cos_2B_gt_cos_2A_necessary_not_sufficient_l746_74676


namespace NUMINAMATH_CALUDE_value_of_a_l746_74623

-- Define sets A and B
def A : Set ℝ := {x : ℝ | |x| = 1}
def B (a : ℝ) : Set ℝ := {x : ℝ | a * x = 1}

-- Theorem statement
theorem value_of_a (a : ℝ) : A ⊇ B a → a = 1 ∨ a = 0 ∨ a = -1 := by
  sorry

end NUMINAMATH_CALUDE_value_of_a_l746_74623


namespace NUMINAMATH_CALUDE_cake_muffin_probability_l746_74634

theorem cake_muffin_probability (total : ℕ) (cake : ℕ) (muffin : ℕ) (both : ℕ)
  (h_total : total = 100)
  (h_cake : cake = 50)
  (h_muffin : muffin = 40)
  (h_both : both = 17) :
  (total - (cake + muffin - both)) / total = 27 / 100 := by
sorry

end NUMINAMATH_CALUDE_cake_muffin_probability_l746_74634


namespace NUMINAMATH_CALUDE_negation_of_existence_negation_of_proposition_l746_74602

theorem negation_of_existence (p : ℕ → Prop) :
  (¬ ∃ n, p n) ↔ (∀ n, ¬ p n) :=
by sorry

theorem negation_of_proposition :
  (¬ ∃ n : ℕ, 2^n > 1000) ↔ (∀ n : ℕ, 2^n ≤ 1000) :=
by sorry

end NUMINAMATH_CALUDE_negation_of_existence_negation_of_proposition_l746_74602


namespace NUMINAMATH_CALUDE_christine_stickers_l746_74646

/-- The number of stickers Christine currently has -/
def current_stickers : ℕ := 11

/-- The number of stickers required for a prize -/
def required_stickers : ℕ := 30

/-- The number of additional stickers Christine needs -/
def additional_stickers : ℕ := required_stickers - current_stickers

theorem christine_stickers : additional_stickers = 19 := by
  sorry

end NUMINAMATH_CALUDE_christine_stickers_l746_74646


namespace NUMINAMATH_CALUDE_convex_polygon_area_bounds_l746_74685

/-- A convex polygon -/
structure ConvexPolygon where
  area : ℝ
  area_pos : area > 0

/-- A rectangle -/
structure Rectangle where
  area : ℝ
  area_pos : area > 0

/-- A parallelogram -/
structure Parallelogram where
  area : ℝ
  area_pos : area > 0

/-- Theorem: For any convex polygon, there exists an enclosing rectangle with area no more than twice the polygon's area, 
    and an inscribed parallelogram with area at least half the polygon's area -/
theorem convex_polygon_area_bounds (P : ConvexPolygon) :
  (∃ R : Rectangle, R.area ≤ 2 * P.area) ∧ 
  (∃ Q : Parallelogram, Q.area ≥ P.area / 2) :=
by sorry

end NUMINAMATH_CALUDE_convex_polygon_area_bounds_l746_74685


namespace NUMINAMATH_CALUDE_undefined_rational_function_l746_74629

theorem undefined_rational_function (x : ℝ) :
  (x^2 - 24*x + 144 = 0) ↔ (x = 12) :=
by sorry

end NUMINAMATH_CALUDE_undefined_rational_function_l746_74629


namespace NUMINAMATH_CALUDE_imaginary_part_of_complex_fraction_l746_74643

theorem imaginary_part_of_complex_fraction :
  let z : ℂ := (2 + Complex.I) / (1 + 3 * Complex.I)
  Complex.im z = -1/2 := by sorry

end NUMINAMATH_CALUDE_imaginary_part_of_complex_fraction_l746_74643


namespace NUMINAMATH_CALUDE_difference_of_squares_and_perfect_squares_l746_74651

theorem difference_of_squares_and_perfect_squares : 
  (102^2 - 98^2 = 800) ∧ 
  (¬ ∃ n : ℕ, n^2 = 102) ∧ 
  (¬ ∃ m : ℕ, m^2 = 98) := by
  sorry

end NUMINAMATH_CALUDE_difference_of_squares_and_perfect_squares_l746_74651


namespace NUMINAMATH_CALUDE_problem_1_l746_74693

theorem problem_1 : 
  (-1.75) - 6.3333333333 - 2.25 + (10/3) = -7 := by sorry

end NUMINAMATH_CALUDE_problem_1_l746_74693


namespace NUMINAMATH_CALUDE_range_of_a_l746_74606

/-- The range of values for a given the conditions -/
theorem range_of_a (a : ℝ) :
  (∀ x y : ℝ, x < y → (a^2 - 2*a - 2)^x < (a^2 - 2*a - 2)^y) ∧ 
  ¬(0 < a ∧ a < 4) →
  a ≥ 4 ∨ a < -1 :=
by sorry

end NUMINAMATH_CALUDE_range_of_a_l746_74606


namespace NUMINAMATH_CALUDE_savings_calculation_l746_74672

/-- The amount saved per month in dollars -/
def monthly_savings : ℕ := 3000

/-- The number of months in a year -/
def months_in_year : ℕ := 12

/-- The total amount saved after one year -/
def total_savings : ℕ := monthly_savings * months_in_year

theorem savings_calculation : total_savings = 36000 := by
  sorry

end NUMINAMATH_CALUDE_savings_calculation_l746_74672


namespace NUMINAMATH_CALUDE_bridget_apples_bridget_apples_solution_l746_74682

theorem bridget_apples : ℕ → Prop :=
  fun total_apples =>
    let apples_after_ann := (2 * total_apples) / 3
    let apples_after_cassie := apples_after_ann - 5
    apples_after_cassie = 7 → total_apples = 18

theorem bridget_apples_solution : bridget_apples 18 := by
  sorry

end NUMINAMATH_CALUDE_bridget_apples_bridget_apples_solution_l746_74682


namespace NUMINAMATH_CALUDE_grid_shading_contradiction_l746_74681

theorem grid_shading_contradiction (k n x y : ℕ) (hk : k > 0) (hn : n > 0) :
  ¬(k * (x + 5) = n * y ∧ k * x = n * (y - 3)) :=
by sorry

end NUMINAMATH_CALUDE_grid_shading_contradiction_l746_74681


namespace NUMINAMATH_CALUDE_complex_expression_equals_eight_l746_74663

theorem complex_expression_equals_eight :
  (1 / Real.sqrt 0.04) + ((1 / Real.sqrt 27) ^ (1/3)) + 
  ((Real.sqrt 2 + 1)⁻¹) - (2 ^ (1/2)) + ((-2) ^ 0) = 8 := by
  sorry

end NUMINAMATH_CALUDE_complex_expression_equals_eight_l746_74663


namespace NUMINAMATH_CALUDE_jen_age_theorem_l746_74625

def jen_age_when_son_born (jen_present_age : ℕ) (son_present_age : ℕ) : ℕ :=
  jen_present_age - son_present_age

theorem jen_age_theorem (jen_present_age : ℕ) (son_present_age : ℕ) :
  son_present_age = 16 →
  jen_present_age = 3 * son_present_age - 7 →
  jen_age_when_son_born jen_present_age son_present_age = 25 := by
  sorry

end NUMINAMATH_CALUDE_jen_age_theorem_l746_74625


namespace NUMINAMATH_CALUDE_bob_win_probability_l746_74638

theorem bob_win_probability (p_lose p_tie : ℚ) 
  (h_lose : p_lose = 5/8)
  (h_tie : p_tie = 1/8) : 
  1 - p_lose - p_tie = 1/4 := by
  sorry

end NUMINAMATH_CALUDE_bob_win_probability_l746_74638


namespace NUMINAMATH_CALUDE_fraction_reducible_by_11_l746_74609

theorem fraction_reducible_by_11 (k : ℕ) 
  (h : (k^2 - 5*k + 8) % 11 = 0 ∨ (k^2 + 6*k + 19) % 11 = 0) : 
  (k^2 - 5*k + 8) % 11 = 0 ∧ (k^2 + 6*k + 19) % 11 = 0 := by
  sorry

end NUMINAMATH_CALUDE_fraction_reducible_by_11_l746_74609


namespace NUMINAMATH_CALUDE_toms_sleep_deficit_l746_74688

/-- Calculates the sleep deficit for a week given ideal and actual sleep hours -/
def sleep_deficit (ideal_hours : ℕ) (weeknight_hours : ℕ) (weekend_hours : ℕ) : ℕ :=
  let ideal_total := ideal_hours * 7
  let actual_total := weeknight_hours * 5 + weekend_hours * 2
  ideal_total - actual_total

/-- Proves that Tom's sleep deficit for a week is 19 hours -/
theorem toms_sleep_deficit : sleep_deficit 8 5 6 = 19 := by
  sorry

end NUMINAMATH_CALUDE_toms_sleep_deficit_l746_74688


namespace NUMINAMATH_CALUDE_projection_property_l746_74653

def projection (v : ℝ × ℝ) : ℝ × ℝ := sorry

theorem projection_property :
  let p := projection
  p (3, 3) = (27/5, 9/5) →
  p (1, -1) = (3/5, 1/5) := by sorry

end NUMINAMATH_CALUDE_projection_property_l746_74653


namespace NUMINAMATH_CALUDE_rational_identity_product_l746_74699

theorem rational_identity_product (M₁ M₂ : ℝ) :
  (∀ x : ℝ, x ≠ 2 ∧ x ≠ 3 → (42 * x - 51) / (x^2 - 5*x + 6) = M₁ / (x - 2) + M₂ / (x - 3)) →
  M₁ * M₂ = -2981.25 := by sorry

end NUMINAMATH_CALUDE_rational_identity_product_l746_74699


namespace NUMINAMATH_CALUDE_rectangle_perimeter_l746_74662

/-- Given a rectangle with one side of length 18 and the sum of its area and perimeter
    equal to 2016, prove that its perimeter is 234. -/
theorem rectangle_perimeter (a : ℝ) : 
  a > 0 → 
  18 * a + 2 * (18 + a) = 2016 → 
  2 * (18 + a) = 234 :=
by
  sorry

end NUMINAMATH_CALUDE_rectangle_perimeter_l746_74662


namespace NUMINAMATH_CALUDE_range_of_a_l746_74690

theorem range_of_a (a : ℝ) (h1 : a > 0) 
  (h2 : ∀ x > 0, (1 / a) - (1 / x) ≤ 2 * x) : 
  a ≥ Real.sqrt 2 / 4 := by
sorry

end NUMINAMATH_CALUDE_range_of_a_l746_74690


namespace NUMINAMATH_CALUDE_acidic_solution_concentration_l746_74636

/-- Represents the properties of an acidic solution -/
structure AcidicSolution where
  initialVolume : ℝ
  removedVolume : ℝ
  finalConcentration : ℝ
  initialConcentration : ℝ

/-- Theorem stating the relationship between initial and final concentrations -/
theorem acidic_solution_concentration 
  (solution : AcidicSolution)
  (h1 : solution.initialVolume = 27)
  (h2 : solution.removedVolume = 9)
  (h3 : solution.finalConcentration = 60)
  (h4 : solution.initialConcentration * solution.initialVolume = 
        solution.finalConcentration * (solution.initialVolume - solution.removedVolume)) :
  solution.initialConcentration = 40 := by
  sorry

#check acidic_solution_concentration

end NUMINAMATH_CALUDE_acidic_solution_concentration_l746_74636


namespace NUMINAMATH_CALUDE_parabola_kite_sum_l746_74697

/-- The sum of coefficients for two parabolas forming a kite -/
theorem parabola_kite_sum (a b : ℝ) : 
  (∃ x₁ x₂ y₁ y₂ : ℝ, 
    -- Parabola 1 intersects x-axis
    a * x₁^2 - 4 = 0 ∧ 
    a * x₂^2 - 4 = 0 ∧ 
    x₁ ≠ x₂ ∧
    -- Parabola 2 intersects x-axis
    6 - b * x₁^2 = 0 ∧ 
    6 - b * x₂^2 = 0 ∧
    -- Parabolas intersect y-axis
    y₁ = -4 ∧
    y₂ = 6 ∧
    -- Area of kite formed by intersection points
    (1/2) * (x₂ - x₁) * (y₂ - y₁) = 16) →
  a + b = 3.9 := by
sorry

end NUMINAMATH_CALUDE_parabola_kite_sum_l746_74697


namespace NUMINAMATH_CALUDE_small_cube_edge_length_l746_74632

theorem small_cube_edge_length (large_cube_edge : ℕ) (small_cube_edge : ℕ) : 
  large_cube_edge = 12 →
  (large_cube_edge / small_cube_edge) > 0 →
  6 * ((large_cube_edge / small_cube_edge - 2) ^ 2) = 12 * (large_cube_edge / small_cube_edge - 2) →
  small_cube_edge = 3 := by
sorry

end NUMINAMATH_CALUDE_small_cube_edge_length_l746_74632


namespace NUMINAMATH_CALUDE_frog_vertical_side_probability_l746_74661

/-- Represents a point in the 2D plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents the square area -/
def Square : Set Point := {p | 0 ≤ p.x ∧ p.x ≤ 5 ∧ 0 ≤ p.y ∧ p.y ≤ 5}

/-- Represents a vertical side of the square -/
def VerticalSide : Set Point := {p | p.x = 0 ∨ p.x = 5}

/-- Represents a single jump of the frog -/
def Jump (p : Point) : Set Point :=
  {q | (q.x = p.x ∧ (q.y = p.y + 1 ∨ q.y = p.y - 1)) ∨
       (q.y = p.y ∧ (q.x = p.x + 1 ∨ q.x = p.x - 1))}

/-- The probability of ending on a vertical side given the starting point -/
noncomputable def ProbVerticalSide (p : Point) : ℝ := sorry

/-- The theorem stating the probability of ending on a vertical side is 1/2 -/
theorem frog_vertical_side_probability :
  ProbVerticalSide ⟨2, 2⟩ = 1/2 := by sorry

end NUMINAMATH_CALUDE_frog_vertical_side_probability_l746_74661


namespace NUMINAMATH_CALUDE_max_permissible_length_l746_74601

/-- A word is permissible if all adjacent letters are different and 
    it's not possible to obtain a word of the form abab by deleting letters, 
    where a and b are different. -/
def Permissible (word : List Char) (alphabet : List Char) : Prop := sorry

/-- The maximum length of a permissible word for an alphabet with n letters -/
def MaxPermissibleLength (n : ℕ) : ℕ := sorry

/-- Theorem: The maximum length of a permissible word for an alphabet with n letters is 2n - 1 -/
theorem max_permissible_length (n : ℕ) : MaxPermissibleLength n = 2 * n - 1 := by
  sorry

end NUMINAMATH_CALUDE_max_permissible_length_l746_74601


namespace NUMINAMATH_CALUDE_symmetry_implies_k_and_b_values_l746_74680

/-- A linear function f(x) = mx + c is symmetric with respect to the y-axis if f(x) = f(-x) for all x -/
def SymmetricToYAxis (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f x = f (-x)

/-- The first linear function f(x) = kx - 5 -/
def f (k : ℝ) (x : ℝ) : ℝ := k * x - 5

/-- The second linear function g(x) = 2x + b -/
def g (b : ℝ) (x : ℝ) : ℝ := 2 * x + b

theorem symmetry_implies_k_and_b_values :
  ∀ k b : ℝ, 
    SymmetricToYAxis (f k) ∧ 
    SymmetricToYAxis (g b) →
    k = -2 ∧ b = -5 := by
  sorry

end NUMINAMATH_CALUDE_symmetry_implies_k_and_b_values_l746_74680


namespace NUMINAMATH_CALUDE_colored_cards_permutations_l746_74655

/-- The number of distinct permutations of a multiset -/
def multiset_permutations (n : ℕ) (frequencies : List ℕ) : ℕ :=
  Nat.factorial n / (frequencies.map Nat.factorial).prod

/-- The problem statement -/
theorem colored_cards_permutations :
  let total_cards : ℕ := 11
  let card_frequencies : List ℕ := [5, 3, 2, 1]
  multiset_permutations total_cards card_frequencies = 27720 := by
  sorry

end NUMINAMATH_CALUDE_colored_cards_permutations_l746_74655


namespace NUMINAMATH_CALUDE_vector_sum_equality_l746_74610

theorem vector_sum_equality : 
  4 • ![-3, 6] + 3 • ![-2, 5] = ![-18, 39] := by
  sorry

end NUMINAMATH_CALUDE_vector_sum_equality_l746_74610


namespace NUMINAMATH_CALUDE_valid_assignment_l746_74617

-- Define the squares
inductive Square
| A | B | C | D | E | F | G

-- Define the arrow directions
def nextSquare : Square → Square
| Square.B => Square.E
| Square.E => Square.C
| Square.C => Square.D
| Square.D => Square.A
| Square.A => Square.G
| Square.G => Square.F
| Square.F => Square.A  -- This should point to the square with 9, which is not in our Square type

-- Define the assignment of numbers to squares
def assignment : Square → Fin 8
| Square.A => 6
| Square.B => 2
| Square.C => 4
| Square.D => 5
| Square.E => 3
| Square.F => 8
| Square.G => 7

-- Theorem statement
theorem valid_assignment : 
  (∀ s : Square, assignment (nextSquare s) = assignment s + 1) ∧
  (∀ i : Fin 8, ∃ s : Square, assignment s = i) :=
by sorry

end NUMINAMATH_CALUDE_valid_assignment_l746_74617


namespace NUMINAMATH_CALUDE_lines_parallel_perpendicular_l746_74644

-- Define the lines l₁ and l₂
def l₁ (a : ℝ) (x y : ℝ) : Prop := x + (1 + a) * y + a - 1 = 0
def l₂ (a : ℝ) (x y : ℝ) : Prop := a * x + 2 * y + 6 = 0

-- Define parallel and perpendicular conditions
def parallel (a : ℝ) : Prop := a = 1
def perpendicular (a : ℝ) : Prop := a = -2/3

-- Theorem statement
theorem lines_parallel_perpendicular :
  (∀ a : ℝ, (∀ x y : ℝ, l₁ a x y ∧ l₂ a x y) → parallel a) ∧
  (∀ a : ℝ, (∀ x y : ℝ, l₁ a x y ∧ l₂ a x y) → perpendicular a) :=
sorry

end NUMINAMATH_CALUDE_lines_parallel_perpendicular_l746_74644


namespace NUMINAMATH_CALUDE_a_range_l746_74616

/-- The line passing through points (x, y) with parameter a -/
def line (x y a : ℝ) : ℝ := x + y - a

/-- Predicate for points being on opposite sides of the line -/
def opposite_sides (a : ℝ) : Prop :=
  (line 0 0 a) * (line 1 1 a) < 0

/-- Theorem stating the range of a given the conditions -/
theorem a_range : 
  (∀ a : ℝ, opposite_sides a ↔ 0 < a ∧ a < 2) :=
sorry

end NUMINAMATH_CALUDE_a_range_l746_74616


namespace NUMINAMATH_CALUDE_flower_combinations_l746_74675

/-- The number of valid combinations of roses and carnations -/
def valid_combinations : ℕ := sorry

/-- Predicate for valid combination of roses and carnations -/
def is_valid_combination (r c : ℕ) : Prop :=
  3 * r + 2 * c = 70 ∧ r + c ≥ 20

theorem flower_combinations :
  valid_combinations = 12 ∨
  valid_combinations = 13 ∨
  valid_combinations = 15 ∨
  valid_combinations = 17 ∨
  valid_combinations = 18 :=
sorry

end NUMINAMATH_CALUDE_flower_combinations_l746_74675


namespace NUMINAMATH_CALUDE_pen_count_is_39_l746_74612

/-- Calculate the final number of pens after a series of operations -/
def final_pen_count (initial : ℕ) (mike_gives : ℕ) (sharon_takes : ℕ) : ℕ :=
  ((initial + mike_gives) * 2) - sharon_takes

/-- Theorem stating that given the initial conditions, the final number of pens is 39 -/
theorem pen_count_is_39 :
  final_pen_count 7 22 19 = 39 := by
  sorry

#eval final_pen_count 7 22 19

end NUMINAMATH_CALUDE_pen_count_is_39_l746_74612


namespace NUMINAMATH_CALUDE_smallest_two_digit_prime_with_composite_reverse_l746_74618

def is_prime (n : Nat) : Prop := n > 1 ∧ ∀ d : Nat, d > 1 → d < n → ¬(d ∣ n)

def is_composite (n : Nat) : Prop := n > 1 ∧ ∃ d : Nat, d > 1 ∧ d < n ∧ d ∣ n

def reverse_digits (n : Nat) : Nat :=
  let tens := n / 10
  let ones := n % 10
  ones * 10 + tens

def is_two_digit (n : Nat) : Prop := n ≥ 10 ∧ n < 100

theorem smallest_two_digit_prime_with_composite_reverse :
  ∃ (p : Nat),
    is_two_digit p ∧
    is_prime p ∧
    (p / 10 = 2) ∧
    is_composite (reverse_digits p) ∧
    (∀ q : Nat, is_two_digit q → is_prime q → (q / 10 = 2) → is_composite (reverse_digits q) → p ≤ q) ∧
    p = 23 :=
by sorry

end NUMINAMATH_CALUDE_smallest_two_digit_prime_with_composite_reverse_l746_74618


namespace NUMINAMATH_CALUDE_initial_oranges_count_prove_initial_oranges_count_l746_74664

/-- Given that 35 oranges were taken away and 25 oranges remained,
    prove that the initial number of oranges was 60. -/
theorem initial_oranges_count : ℕ → Prop :=
  fun initial : ℕ =>
    ∀ (taken remaining : ℕ),
      taken = 35 →
      remaining = 25 →
      initial = taken + remaining →
      initial = 60

-- The proof is omitted
theorem prove_initial_oranges_count : initial_oranges_count 60 := by sorry

end NUMINAMATH_CALUDE_initial_oranges_count_prove_initial_oranges_count_l746_74664


namespace NUMINAMATH_CALUDE_rancher_problem_l746_74665

theorem rancher_problem (s c : ℕ) : s > 0 ∧ c > 0 ∧ 30 * s + 31 * c = 1200 → s = 9 ∧ c = 30 := by
  sorry

end NUMINAMATH_CALUDE_rancher_problem_l746_74665


namespace NUMINAMATH_CALUDE_quadratic_one_solution_l746_74640

theorem quadratic_one_solution (n : ℝ) : 
  (∃! x, 4 * x^2 + n * x + 4 = 0) ↔ (n = 8 ∨ n = -8) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_one_solution_l746_74640


namespace NUMINAMATH_CALUDE_similar_triangle_perimeter_l746_74671

theorem similar_triangle_perimeter : ∀ (a b c : ℝ),
  a > 0 → b > 0 → c > 0 →
  a^2 + b^2 = c^2 →
  ∃ (k : ℝ), k > 0 ∧ 
  (k * a = 18 ∨ k * b = 18) →
  k * (a + b + c) = 72 :=
by
  sorry

end NUMINAMATH_CALUDE_similar_triangle_perimeter_l746_74671


namespace NUMINAMATH_CALUDE_rosie_pies_l746_74668

/-- Given that Rosie can make 3 pies out of 12 apples, 
    this theorem proves she can make 9 pies out of 36 apples. -/
theorem rosie_pies (apples_per_three_pies : ℕ) (total_apples : ℕ) : 
  apples_per_three_pies = 12 →
  total_apples = 36 →
  (total_apples / apples_per_three_pies) * 3 = 9 := by
  sorry

end NUMINAMATH_CALUDE_rosie_pies_l746_74668


namespace NUMINAMATH_CALUDE_percentage_caught_sampling_candy_l746_74626

/-- The percentage of customers caught sampling candy -/
def percentage_caught (total_sample_percent : ℝ) (not_caught_ratio : ℝ) : ℝ :=
  total_sample_percent - (not_caught_ratio * total_sample_percent)

/-- Theorem stating the percentage of customers caught sampling candy -/
theorem percentage_caught_sampling_candy :
  let total_sample_percent : ℝ := 23.913043478260867
  let not_caught_ratio : ℝ := 0.08
  percentage_caught total_sample_percent not_caught_ratio = 22 := by
  sorry


end NUMINAMATH_CALUDE_percentage_caught_sampling_candy_l746_74626


namespace NUMINAMATH_CALUDE_sequence_sum_property_l746_74695

theorem sequence_sum_property (a : ℕ → ℝ) (S : ℕ → ℝ) :
  (∀ n : ℕ, S n = n^2 + 2*n + 5) →
  (∀ n : ℕ, S (n+1) - S n = a (n+1)) →
  a 2 + a 3 + a 4 + a 4 + a 5 = 41 := by
sorry

end NUMINAMATH_CALUDE_sequence_sum_property_l746_74695


namespace NUMINAMATH_CALUDE_pastry_difference_l746_74649

/-- Represents the number of pastries each person has -/
structure Pastries where
  frank : ℕ
  calvin : ℕ
  phoebe : ℕ
  grace : ℕ

/-- The conditions of the pastry problem -/
def PastryProblem (p : Pastries) : Prop :=
  p.calvin = p.frank + 8 ∧
  p.phoebe = p.frank + 8 ∧
  p.grace = 30 ∧
  p.frank + p.calvin + p.phoebe + p.grace = 97

theorem pastry_difference (p : Pastries) (h : PastryProblem p) :
  p.grace - p.calvin = 5 ∧ p.grace - p.phoebe = 5 :=
sorry

end NUMINAMATH_CALUDE_pastry_difference_l746_74649


namespace NUMINAMATH_CALUDE_consecutive_integers_average_l746_74669

theorem consecutive_integers_average (a c : ℤ) : 
  (∀ k ∈ Finset.range 7, k + a > 0) →  -- Positive integers condition
  c = (Finset.sum (Finset.range 7) (λ k => k + a)) / 7 →  -- Average condition
  (Finset.sum (Finset.range 7) (λ k => k + c)) / 7 = a + 6 := by
  sorry

end NUMINAMATH_CALUDE_consecutive_integers_average_l746_74669


namespace NUMINAMATH_CALUDE_sum_of_reciprocals_squared_l746_74614

/-- Given the definitions of p, q, r, and s, prove that (1/p + 1/q + 1/r + 1/s)² = 560/151321 -/
theorem sum_of_reciprocals_squared (p q r s : ℝ) 
  (hp : p = Real.sqrt 5 + Real.sqrt 7 + Real.sqrt 35)
  (hq : q = -Real.sqrt 5 + Real.sqrt 7 + Real.sqrt 35)
  (hr : r = Real.sqrt 5 - Real.sqrt 7 + Real.sqrt 35)
  (hs : s = -Real.sqrt 5 - Real.sqrt 7 + Real.sqrt 35) :
  (1/p + 1/q + 1/r + 1/s)^2 = 560/151321 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_reciprocals_squared_l746_74614


namespace NUMINAMATH_CALUDE_linear_equation_exponent_l746_74694

theorem linear_equation_exponent (a : ℝ) : 
  (∀ x, ∃ k m : ℝ, 3 * x^(2*a - 1) - 4 = k * x + m) → a = 1 := by
  sorry

end NUMINAMATH_CALUDE_linear_equation_exponent_l746_74694


namespace NUMINAMATH_CALUDE_union_of_reduced_rectangles_l746_74620

-- Define a reduced rectangle as a set in ℝ²
def ReducedRectangle : Set (ℝ × ℝ) → Prop :=
  sorry

-- Define a family of reduced rectangles
def FamilyOfReducedRectangles : Set (Set (ℝ × ℝ)) → Prop :=
  sorry

-- The main theorem
theorem union_of_reduced_rectangles 
  (F : Set (Set (ℝ × ℝ))) 
  (h : FamilyOfReducedRectangles F) :
  ∃ (C : Set (Set (ℝ × ℝ))), 
    (C ⊆ F) ∧ 
    (Countable C) ∧ 
    (⋃₀ F = ⋃₀ C) :=
  sorry

end NUMINAMATH_CALUDE_union_of_reduced_rectangles_l746_74620


namespace NUMINAMATH_CALUDE_hyperbola_eccentricity_l746_74624

theorem hyperbola_eccentricity (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  let c := Real.sqrt (a^2 + b^2)
  ∃ (m : ℝ), 
    (a^2 / c)^2 + m^2 = c^2 ∧ 
    2 * c * m = 4 * a * b → 
    (a^2 + b^2) / a^2 = 3 :=
by sorry

end NUMINAMATH_CALUDE_hyperbola_eccentricity_l746_74624


namespace NUMINAMATH_CALUDE_x_not_equal_y_l746_74658

-- Define the sequences x_n and y_n
def x : ℕ → ℤ
  | 0 => 1
  | 1 => 1
  | (n + 2) => x (n + 1) + 2 * x n

def y : ℕ → ℤ
  | 0 => 1
  | 1 => 7
  | (n + 2) => 2 * y (n + 1) + 3 * y n

-- State the theorem
theorem x_not_equal_y (m n : ℕ) (hm : m > 0) (hn : n > 0) : x m ≠ y n := by
  sorry

end NUMINAMATH_CALUDE_x_not_equal_y_l746_74658


namespace NUMINAMATH_CALUDE_smallest_reachable_integer_l746_74687

-- Define the sequence u_n
def u : ℕ → ℕ
  | 0 => 2010^2010
  | (n+1) => if u n % 2 = 1 then u n + 7 else u n / 2

-- Define the property of being reachable by the sequence
def Reachable (m : ℕ) : Prop := ∃ n, u n = m

-- State the theorem
theorem smallest_reachable_integer : 
  (∃ m, Reachable m) ∧ (∀ k, Reachable k → k ≥ 1) := by sorry

end NUMINAMATH_CALUDE_smallest_reachable_integer_l746_74687


namespace NUMINAMATH_CALUDE_parallel_line_k_value_l746_74679

/-- Given a line passing through points (4, -7) and (k, 25) that is parallel to the line 3x + 4y = 12, 
    the value of k is -116/3. -/
theorem parallel_line_k_value : 
  ∀ k : ℚ, 
  (∃ m b : ℚ, (∀ x y : ℚ, y = m * x + b → (x = 4 ∧ y = -7) ∨ (x = k ∧ y = 25)) ∧ 
               m = -(3 / 4)) → 
  k = -116 / 3 :=
by sorry

end NUMINAMATH_CALUDE_parallel_line_k_value_l746_74679


namespace NUMINAMATH_CALUDE_opposite_of_negative_three_fifths_l746_74647

theorem opposite_of_negative_three_fifths :
  -((-3 : ℚ) / 5) = 3 / 5 := by
  sorry

end NUMINAMATH_CALUDE_opposite_of_negative_three_fifths_l746_74647


namespace NUMINAMATH_CALUDE_greatest_integer_fraction_inequality_l746_74603

theorem greatest_integer_fraction_inequality : 
  ∀ x : ℤ, (8 : ℚ) / 11 > (x : ℚ) / 15 ↔ x ≤ 10 :=
sorry

end NUMINAMATH_CALUDE_greatest_integer_fraction_inequality_l746_74603


namespace NUMINAMATH_CALUDE_expression_evaluation_l746_74650

theorem expression_evaluation :
  let x : ℚ := -1
  let y : ℚ := 2
  (2*x + y) * (2*x - y) - (8*x^3*y - 2*x*y^3 - x^2*y^2) / (2*x*y) = -1 :=
by sorry

end NUMINAMATH_CALUDE_expression_evaluation_l746_74650


namespace NUMINAMATH_CALUDE_train_distance_problem_l746_74674

theorem train_distance_problem (speed1 speed2 extra_distance : ℝ) 
  (h1 : speed1 = 50)
  (h2 : speed2 = 60)
  (h3 : extra_distance = 100)
  (h4 : speed1 > 0)
  (h5 : speed2 > 0) :
  ∃ (distance1 distance2 : ℝ),
    distance1 > 0 ∧
    distance2 > 0 ∧
    distance2 = distance1 + extra_distance ∧
    distance1 / speed1 = distance2 / speed2 ∧
    distance1 + distance2 = 1100 :=
by sorry

end NUMINAMATH_CALUDE_train_distance_problem_l746_74674


namespace NUMINAMATH_CALUDE_smallest_five_digit_divisible_by_2_3_8_9_l746_74696

theorem smallest_five_digit_divisible_by_2_3_8_9 : ∃ n : ℕ,
  (n ≥ 10000 ∧ n < 100000) ∧  -- five-digit number
  (∀ m : ℕ, m ≥ 10000 ∧ m < 100000 → 
    (m % 2 = 0 ∧ m % 3 = 0 ∧ m % 8 = 0 ∧ m % 9 = 0) → 
    n ≤ m) ∧  -- smallest such number
  (n % 2 = 0 ∧ n % 3 = 0 ∧ n % 8 = 0 ∧ n % 9 = 0) ∧  -- divisible by 2, 3, 8, and 9
  n = 10008  -- the specific value
:= by sorry

end NUMINAMATH_CALUDE_smallest_five_digit_divisible_by_2_3_8_9_l746_74696


namespace NUMINAMATH_CALUDE_synthetic_method_deduces_result_from_cause_l746_74639

/-- The synthetic method in mathematics -/
def synthetic_method : Type := Unit

/-- Property of deducing result from cause -/
def deduces_result_from_cause (m : Type) : Prop := sorry

/-- The synthetic method is a way of thinking in mathematics -/
axiom synthetic_method_is_way_of_thinking : synthetic_method = Unit

/-- Theorem: The synthetic method deduces the result from the cause -/
theorem synthetic_method_deduces_result_from_cause : 
  deduces_result_from_cause synthetic_method :=
sorry

end NUMINAMATH_CALUDE_synthetic_method_deduces_result_from_cause_l746_74639


namespace NUMINAMATH_CALUDE_greatest_possible_median_l746_74683

theorem greatest_possible_median (k m p r s t u : ℕ) : 
  (k + m + p + r + s + t + u) / 7 = 24 →
  0 < k → k < m → m < p → p < r → r < s → s < t → t < u →
  t = 54 →
  k + m ≤ 20 →
  r ≤ 53 ∧ ∃ (k' m' p' r' s' t' u' : ℕ), 
    (k' + m' + p' + r' + s' + t' + u') / 7 = 24 ∧
    0 < k' ∧ k' < m' ∧ m' < p' ∧ p' < r' ∧ r' < s' ∧ s' < t' ∧ t' < u' ∧
    t' = 54 ∧
    k' + m' ≤ 20 ∧
    r' = 53 := by
  sorry

end NUMINAMATH_CALUDE_greatest_possible_median_l746_74683


namespace NUMINAMATH_CALUDE_ratio_of_x_intercepts_l746_74667

/-- Given two lines with different non-zero y-intercepts:
    - First line has slope 8 and x-intercept (r, 0)
    - Second line has slope 4 and x-intercept (q, 0)
    - First line's y-intercept is double that of the second line
    Prove that the ratio of r to q is 1 -/
theorem ratio_of_x_intercepts (r q c : ℝ) : 
  (8 * r + 2 * c = 0) →  -- First line equation at x-intercept
  (4 * q + c = 0) →      -- Second line equation at x-intercept
  r / q = 1 := by
sorry

end NUMINAMATH_CALUDE_ratio_of_x_intercepts_l746_74667


namespace NUMINAMATH_CALUDE_incorrect_inequality_l746_74686

theorem incorrect_inequality (x y : ℝ) (h : x > y) : ¬(3 - x > 3 - y) := by
  sorry

end NUMINAMATH_CALUDE_incorrect_inequality_l746_74686


namespace NUMINAMATH_CALUDE_vector_BC_l746_74689

def A : ℝ × ℝ := (0, 1)
def B : ℝ × ℝ := (3, 2)
def AC : ℝ × ℝ := (-4, -3)

theorem vector_BC : 
  let C : ℝ × ℝ := (A.1 + AC.1, A.2 + AC.2)
  (C.1 - B.1, C.2 - B.2) = (-7, -4) := by sorry

end NUMINAMATH_CALUDE_vector_BC_l746_74689


namespace NUMINAMATH_CALUDE_max_triangle_area_l746_74615

def parabola (x : ℝ) : ℝ := -x^2 + 6*x - 5

theorem max_triangle_area :
  let A : ℝ × ℝ := (1, 0)
  let B : ℝ × ℝ := (4, 3)
  let C (p : ℝ) : ℝ × ℝ := (p, parabola p)
  let triangle_area (p : ℝ) : ℝ := 
    (1/2) * abs ((A.1 * B.2 + B.1 * (C p).2 + (C p).1 * A.2) - 
                 (A.2 * B.1 + B.2 * (C p).1 + (C p).2 * A.1))
  ∀ p : ℝ, 1 ≤ p ∧ p ≤ 4 → triangle_area p ≤ 27/8 :=
by
  sorry

#check max_triangle_area

end NUMINAMATH_CALUDE_max_triangle_area_l746_74615


namespace NUMINAMATH_CALUDE_possible_values_of_a_l746_74605

theorem possible_values_of_a (A B : Set ℝ) (a : ℝ) : 
  A = {x : ℝ | a * x + 2 = 0} → 
  B = {-1, 2} → 
  A ⊆ B → 
  {a | ∃ (A : Set ℝ), A = {x : ℝ | a * x + 2 = 0} ∧ A ⊆ B} = {-1, 0, 2} := by
sorry

end NUMINAMATH_CALUDE_possible_values_of_a_l746_74605


namespace NUMINAMATH_CALUDE_selina_shorts_sold_l746_74633

/-- Represents the number of pairs of shorts Selina sold -/
def shorts_sold : ℕ := sorry

/-- The price of a pair of pants in dollars -/
def pants_price : ℕ := 5

/-- The price of a pair of shorts in dollars -/
def shorts_price : ℕ := 3

/-- The price of a shirt in dollars -/
def shirt_price : ℕ := 4

/-- The number of pairs of pants Selina sold -/
def pants_sold : ℕ := 3

/-- The number of shirts Selina sold -/
def shirts_sold : ℕ := 5

/-- The price of each new shirt Selina bought -/
def new_shirt_price : ℕ := 10

/-- The number of new shirts Selina bought -/
def new_shirts_bought : ℕ := 2

/-- The amount of money Selina left the store with -/
def money_left : ℕ := 30

theorem selina_shorts_sold :
  shorts_sold = 5 ∧
  pants_sold * pants_price + shirts_sold * shirt_price + shorts_sold * shorts_price =
    money_left + new_shirts_bought * new_shirt_price :=
by sorry

end NUMINAMATH_CALUDE_selina_shorts_sold_l746_74633


namespace NUMINAMATH_CALUDE_total_houses_l746_74656

theorem total_houses (dogs cats both : ℕ) 
  (h_dogs : dogs = 40)
  (h_cats : cats = 30)
  (h_both : both = 10) :
  dogs + cats - both = 60 := by
  sorry

end NUMINAMATH_CALUDE_total_houses_l746_74656


namespace NUMINAMATH_CALUDE_division_problem_l746_74659

theorem division_problem (x y z : ℚ) 
  (h1 : x / y = 3)
  (h2 : y / z = 5 / 2) :
  z / x = 2 / 15 := by
  sorry

end NUMINAMATH_CALUDE_division_problem_l746_74659


namespace NUMINAMATH_CALUDE_impossibility_of_arrangement_l746_74678

/-- Represents a 6x7 grid of natural numbers -/
def Grid := Fin 6 → Fin 7 → ℕ

/-- Checks if a given grid is a valid arrangement of numbers 1 to 42 -/
def is_valid_arrangement (g : Grid) : Prop :=
  (∀ i j, g i j ≥ 1 ∧ g i j ≤ 42) ∧
  (∀ i j k l, (i ≠ k ∨ j ≠ l) → g i j ≠ g k l)

/-- Checks if the sum of numbers in each 1x2 vertical rectangle is even -/
def has_even_vertical_sums (g : Grid) : Prop :=
  ∀ i j, Even (g i j + g (i.succ) j)

theorem impossibility_of_arrangement :
  ¬∃ (g : Grid), is_valid_arrangement g ∧ has_even_vertical_sums g :=
sorry

end NUMINAMATH_CALUDE_impossibility_of_arrangement_l746_74678


namespace NUMINAMATH_CALUDE_barbara_initial_candies_l746_74642

/-- The number of candies Barbara bought -/
def candies_bought : ℕ := 18

/-- The total number of candies Barbara has after buying more -/
def total_candies : ℕ := 27

/-- The number of candies Barbara had initially -/
def initial_candies : ℕ := total_candies - candies_bought

theorem barbara_initial_candies : initial_candies = 9 := by
  sorry

end NUMINAMATH_CALUDE_barbara_initial_candies_l746_74642


namespace NUMINAMATH_CALUDE_divisibility_by_eighteen_l746_74677

theorem divisibility_by_eighteen (n : Nat) : n ≤ 9 → (315 * 10 + n) % 18 = 0 ↔ n = 0 := by
  sorry

end NUMINAMATH_CALUDE_divisibility_by_eighteen_l746_74677

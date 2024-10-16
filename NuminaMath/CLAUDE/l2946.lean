import Mathlib

namespace NUMINAMATH_CALUDE_no_solution_for_certain_a_l2946_294688

-- Define the equation
def equation (x a : ℝ) : ℝ := 6 * abs (x - 4*a) + abs (x - a^2) + 5*x - 4*a

-- State the theorem
theorem no_solution_for_certain_a :
  ∀ a : ℝ, (a < -12 ∨ a > 0) → ¬∃ x : ℝ, equation x a = 0 := by
  sorry

end NUMINAMATH_CALUDE_no_solution_for_certain_a_l2946_294688


namespace NUMINAMATH_CALUDE_repeating_decimal_35_sum_l2946_294636

/-- Represents a repeating decimal with a two-digit repetend -/
def RepeatingDecimal (n : ℕ) : ℚ :=
  n / 99

theorem repeating_decimal_35_sum : 
  ∀ a b : ℕ, 
  a > 0 → b > 0 →
  RepeatingDecimal 35 = a / b →
  Nat.gcd a b = 1 →
  a + b = 134 := by
  sorry

end NUMINAMATH_CALUDE_repeating_decimal_35_sum_l2946_294636


namespace NUMINAMATH_CALUDE_equation_solution_l2946_294664

theorem equation_solution : ∃ (a b : ℤ), a^2 * b^2 + a^2 + b^2 + 1 = 2005 ∧ (a = 2 ∧ b = 20) := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l2946_294664


namespace NUMINAMATH_CALUDE_ratio_c_d_equals_one_over_320_l2946_294620

theorem ratio_c_d_equals_one_over_320 (a b c d : ℝ) : 
  8 = 0.02 * a → 
  2 = 0.08 * b → 
  d = 0.05 * a → 
  c = b / a → 
  c / d = 1 / 320 := by
sorry

end NUMINAMATH_CALUDE_ratio_c_d_equals_one_over_320_l2946_294620


namespace NUMINAMATH_CALUDE_log_difference_negative_l2946_294698

theorem log_difference_negative (a b : ℝ) (h1 : 0 < a) (h2 : a < b) (h3 : b < 1) :
  Real.log (b - a) < 0 := by
  sorry

end NUMINAMATH_CALUDE_log_difference_negative_l2946_294698


namespace NUMINAMATH_CALUDE_smallest_sum_of_squares_l2946_294631

theorem smallest_sum_of_squares (x₁ x₂ x₃ : ℝ) (h_pos₁ : 0 < x₁) (h_pos₂ : 0 < x₂) (h_pos₃ : 0 < x₃)
  (h_sum : 2 * x₁ + 3 * x₂ + 4 * x₃ = 120) :
  ∃ (min : ℝ), min = 14400 / 29 ∧ x₁^2 + x₂^2 + x₃^2 ≥ min ∧
  ∃ (y₁ y₂ y₃ : ℝ), 0 < y₁ ∧ 0 < y₂ ∧ 0 < y₃ ∧
    2 * y₁ + 3 * y₂ + 4 * y₃ = 120 ∧ y₁^2 + y₂^2 + y₃^2 = min := by
  sorry

end NUMINAMATH_CALUDE_smallest_sum_of_squares_l2946_294631


namespace NUMINAMATH_CALUDE_intersection_line_passes_through_fixed_point_l2946_294625

/-- Definition of the ellipse C -/
def ellipse_C (x y : ℝ) : Prop := x^2 + 3 * y^2 = 6

/-- Right focus of the ellipse -/
def right_focus : ℝ × ℝ := (2, 0)

/-- Line passing through the right focus -/
def line_through_focus (k : ℝ) (x : ℝ) : ℝ := k * (x - 2)

/-- Intersection points of the line and the ellipse -/
def intersection_points (k : ℝ) : Set (ℝ × ℝ) :=
  {p | ∃ x y, p = (x, y) ∧ ellipse_C x y ∧ y = line_through_focus k x}

/-- Symmetric point about x-axis -/
def symmetric_point (p : ℝ × ℝ) : ℝ × ℝ := (p.1, -p.2)

/-- The fixed point on x-axis -/
def fixed_point : ℝ × ℝ := (3, 0)

theorem intersection_line_passes_through_fixed_point (k : ℝ) (hk : k ≠ 0) :
  ∀ p q, p ∈ intersection_points k → q ∈ intersection_points k → p ≠ q →
  ∃ t, (1 - t) • (symmetric_point p) + t • q = fixed_point :=
sorry

end NUMINAMATH_CALUDE_intersection_line_passes_through_fixed_point_l2946_294625


namespace NUMINAMATH_CALUDE_solution_set_is_open_interval_l2946_294618

-- Define the quadratic function
def f (x : ℝ) : ℝ := -2 * x^2 + 3 * x - 1

-- Define the solution set
def solution_set : Set ℝ := {x | f x > 0}

-- Theorem statement
theorem solution_set_is_open_interval :
  solution_set = Set.Ioo (1/2 : ℝ) 1 := by sorry

end NUMINAMATH_CALUDE_solution_set_is_open_interval_l2946_294618


namespace NUMINAMATH_CALUDE_fraction_sum_equals_decimal_l2946_294610

theorem fraction_sum_equals_decimal : 
  (3 : ℚ) / 10 + (3 : ℚ) / 100 + (3 : ℚ) / 1000 = (333 : ℚ) / 1000 := by
  sorry

end NUMINAMATH_CALUDE_fraction_sum_equals_decimal_l2946_294610


namespace NUMINAMATH_CALUDE_bench_seating_theorem_l2946_294697

/-- The number of ways to arrange people on a bench with empty seats -/
def seating_arrangements (total_seats : ℕ) (people : ℕ) (adjacent_empty : ℕ) : ℕ :=
  (people.factorial) * (people + 1).factorial / 2

/-- Theorem: There are 480 ways to arrange 4 people on a bench with 7 seats,
    such that exactly 2 of the 3 empty seats are adjacent -/
theorem bench_seating_theorem :
  seating_arrangements 7 4 2 = 480 := by
  sorry

#eval seating_arrangements 7 4 2

end NUMINAMATH_CALUDE_bench_seating_theorem_l2946_294697


namespace NUMINAMATH_CALUDE_inequality_not_always_hold_l2946_294642

theorem inequality_not_always_hold (a b c : ℝ) (h1 : a > b) (h2 : b > 0) (h3 : c ≠ 0) :
  ¬ (∀ a b c, a > b ∧ b > 0 ∧ c ≠ 0 → (a - b) / c > 0) := by
  sorry

end NUMINAMATH_CALUDE_inequality_not_always_hold_l2946_294642


namespace NUMINAMATH_CALUDE_perspective_drawing_preserves_parallel_equal_l2946_294604

/-- A plane figure -/
structure PlaneFigure where
  -- Add necessary fields

/-- A perspective drawing of a plane figure -/
structure PerspectiveDrawing where
  -- Add necessary fields

/-- A line segment in a plane figure or perspective drawing -/
structure LineSegment where
  -- Add necessary fields

/-- Predicate to check if two line segments are parallel -/
def are_parallel (s1 s2 : LineSegment) : Prop := sorry

/-- Predicate to check if two line segments are equal in length -/
def are_equal (s1 s2 : LineSegment) : Prop := sorry

/-- Function to get the corresponding line segments in a perspective drawing -/
def perspective_line_segments (pf : PlaneFigure) (pd : PerspectiveDrawing) (s1 s2 : LineSegment) : 
  (LineSegment × LineSegment) := sorry

theorem perspective_drawing_preserves_parallel_equal 
  (pf : PlaneFigure) (pd : PerspectiveDrawing) (s1 s2 : LineSegment) :
  are_parallel s1 s2 → are_equal s1 s2 → 
  let (p1, p2) := perspective_line_segments pf pd s1 s2
  are_parallel p1 p2 ∧ are_equal p1 p2 :=
by sorry

end NUMINAMATH_CALUDE_perspective_drawing_preserves_parallel_equal_l2946_294604


namespace NUMINAMATH_CALUDE_darcy_remaining_clothes_l2946_294680

def remaining_clothes_to_fold (total_shirts : ℕ) (total_shorts : ℕ) (folded_shirts : ℕ) (folded_shorts : ℕ) : ℕ :=
  (total_shirts + total_shorts) - (folded_shirts + folded_shorts)

theorem darcy_remaining_clothes : 
  remaining_clothes_to_fold 20 8 12 5 = 11 := by
  sorry

end NUMINAMATH_CALUDE_darcy_remaining_clothes_l2946_294680


namespace NUMINAMATH_CALUDE_min_cubes_satisfy_conditions_num_cubes_is_minimum_l2946_294671

/-- Represents the number of cubes in each identical box -/
def num_cubes : ℕ := 1344

/-- Represents the side length of the outer square in the first girl's frame -/
def frame_outer : ℕ := 50

/-- Represents the side length of the inner square in the first girl's frame -/
def frame_inner : ℕ := 34

/-- Represents the side length of the second girl's square -/
def square_second : ℕ := 62

/-- Represents the side length of the third girl's square -/
def square_third : ℕ := 72

/-- Theorem stating that the given number of cubes satisfies all conditions -/
theorem min_cubes_satisfy_conditions :
  (frame_outer^2 - frame_inner^2 = num_cubes) ∧
  (square_second^2 = num_cubes) ∧
  (square_third^2 + 4 = num_cubes) := by
  sorry

/-- Theorem stating that the given number of cubes is the minimum possible -/
theorem num_cubes_is_minimum (n : ℕ) :
  (n < num_cubes) →
  ¬((frame_outer^2 - frame_inner^2 = n) ∧
    (∃ m : ℕ, m^2 = n) ∧
    (∃ k : ℕ, k^2 + 4 = n)) := by
  sorry

end NUMINAMATH_CALUDE_min_cubes_satisfy_conditions_num_cubes_is_minimum_l2946_294671


namespace NUMINAMATH_CALUDE_expression_simplification_and_evaluation_l2946_294614

theorem expression_simplification_and_evaluation :
  let x : ℝ := Real.sqrt 2 + 1
  (1 - 1 / x) / ((x^2 - 2*x + 1) / x^2) = 1 + Real.sqrt 2 / 2 := by
  sorry

end NUMINAMATH_CALUDE_expression_simplification_and_evaluation_l2946_294614


namespace NUMINAMATH_CALUDE_box_volume_increase_l2946_294686

/-- Given a rectangular box with length l, width w, and height h, 
    if the volume is 3456, surface area is 1368, and sum of edges is 192,
    prove that increasing each dimension by 2 results in a volume of 5024 -/
theorem box_volume_increase (l w h : ℝ) 
  (hv : l * w * h = 3456)
  (hs : 2 * (l * w + w * h + h * l) = 1368)
  (he : 4 * (l + w + h) = 192) :
  (l + 2) * (w + 2) * (h + 2) = 5024 := by
  sorry

end NUMINAMATH_CALUDE_box_volume_increase_l2946_294686


namespace NUMINAMATH_CALUDE_irrational_sqrt_three_rational_others_l2946_294675

-- Define a rational number
def is_rational (x : ℝ) : Prop := ∃ (a b : ℤ), b ≠ 0 ∧ x = a / b

-- Define an irrational number
def is_irrational (x : ℝ) : Prop := ¬ is_rational x

theorem irrational_sqrt_three_rational_others : 
  is_irrational (Real.sqrt 3) ∧ 
  is_rational (-2) ∧ 
  is_rational (1/2) ∧ 
  is_rational 2 :=
sorry

end NUMINAMATH_CALUDE_irrational_sqrt_three_rational_others_l2946_294675


namespace NUMINAMATH_CALUDE_geometric_sequence_exists_l2946_294635

theorem geometric_sequence_exists : ∃ (a r : ℝ), 
  a ≠ 0 ∧ r ≠ 0 ∧ 
  a * r^2 = 3 ∧
  a * r^4 = 27 ∧
  a = -1/3 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_exists_l2946_294635


namespace NUMINAMATH_CALUDE_min_rings_to_connect_five_links_l2946_294674

/-- Represents a chain link with a specific number of rings -/
structure ChainLink where
  rings : ℕ

/-- Represents a collection of chain links -/
structure ChainCollection where
  links : List ChainLink

/-- The minimum number of rings needed to connect a chain collection into a single chain -/
def minRingsToConnect (c : ChainCollection) : ℕ :=
  sorry

/-- The problem statement -/
theorem min_rings_to_connect_five_links :
  let links := List.replicate 5 (ChainLink.mk 3)
  let chain := ChainCollection.mk links
  minRingsToConnect chain = 3 := by
  sorry

end NUMINAMATH_CALUDE_min_rings_to_connect_five_links_l2946_294674


namespace NUMINAMATH_CALUDE_symmetric_points_difference_l2946_294694

/-- Two points are symmetric with respect to the origin if their coordinates sum to zero -/
def symmetric_wrt_origin (p q : ℝ × ℝ) : Prop :=
  p.1 + q.1 = 0 ∧ p.2 + q.2 = 0

/-- Given that point A(-2,b) is symmetric to point B(a,3) with respect to the origin,
    prove that a - b = 5 -/
theorem symmetric_points_difference (a b : ℝ) 
    (h : symmetric_wrt_origin (-2, b) (a, 3)) : a - b = 5 := by
  sorry

end NUMINAMATH_CALUDE_symmetric_points_difference_l2946_294694


namespace NUMINAMATH_CALUDE_equation_solution_l2946_294699

theorem equation_solution : 
  ∃ x : ℝ, (x + 6) / (x - 3) = 4 ∧ x = 6 := by sorry

end NUMINAMATH_CALUDE_equation_solution_l2946_294699


namespace NUMINAMATH_CALUDE_max_page_number_with_fifteen_fives_l2946_294601

/-- Represents the count of a specific digit in a number -/
def digitCount (n : ℕ) (d : ℕ) : ℕ := sorry

/-- Represents the total count of a specific digit used in numbering pages from 1 to n -/
def totalDigitCount (n : ℕ) (d : ℕ) : ℕ := sorry

/-- The maximum page number that can be reached with a given number of a specific digit -/
def maxPageNumber (availableDigits : ℕ) (digit : ℕ) : ℕ := sorry

theorem max_page_number_with_fifteen_fives :
  maxPageNumber 15 5 = 59 := by sorry

end NUMINAMATH_CALUDE_max_page_number_with_fifteen_fives_l2946_294601


namespace NUMINAMATH_CALUDE_subtract_negative_three_l2946_294663

theorem subtract_negative_three : 0 - (-3) = 3 := by
  sorry

end NUMINAMATH_CALUDE_subtract_negative_three_l2946_294663


namespace NUMINAMATH_CALUDE_trajectory_of_P_l2946_294606

/-- The hyperbola equation -/
def hyperbola (x y : ℝ) : Prop := x^2 - y^2/4 = 1

/-- The line equation -/
def line (k m x y : ℝ) : Prop := y = k*x + m

/-- Condition that k is not equal to ±2 -/
def k_condition (k : ℝ) : Prop := k ≠ 2 ∧ k ≠ -2

/-- The trajectory equation -/
def trajectory (x y : ℝ) : Prop := x^2/25 - 4*y^2/25 = 1

/-- Main theorem: The trajectory of point P -/
theorem trajectory_of_P (k m x y : ℝ) :
  k_condition k →
  (∃ (x₀ y₀ : ℝ), hyperbola x₀ y₀ ∧ line k m x₀ y₀) →
  y ≠ 0 →
  trajectory x y :=
sorry

end NUMINAMATH_CALUDE_trajectory_of_P_l2946_294606


namespace NUMINAMATH_CALUDE_arctan_sum_equals_pi_half_l2946_294670

theorem arctan_sum_equals_pi_half (n : ℕ+) :
  Real.arctan (1/3) + Real.arctan (1/4) + Real.arctan (1/7) + Real.arctan (1/n) = π/2 → n = 54 := by
  sorry

end NUMINAMATH_CALUDE_arctan_sum_equals_pi_half_l2946_294670


namespace NUMINAMATH_CALUDE_at_least_one_not_in_area_l2946_294602

theorem at_least_one_not_in_area (p q : Prop) : 
  (¬p ∨ ¬q) ↔ (∃ x, x = p ∨ x = q) ∧ (x → False) :=
by sorry

end NUMINAMATH_CALUDE_at_least_one_not_in_area_l2946_294602


namespace NUMINAMATH_CALUDE_peters_expression_exists_l2946_294605

/-- An expression type that can represent sums and products of ones -/
inductive Expr
  | one : Expr
  | add : Expr → Expr → Expr
  | mul : Expr → Expr → Expr

/-- Evaluate an expression -/
def eval : Expr → Nat
  | Expr.one => 1
  | Expr.add e1 e2 => eval e1 + eval e2
  | Expr.mul e1 e2 => eval e1 * eval e2

/-- Swap addition and multiplication in an expression -/
def swap : Expr → Expr
  | Expr.one => Expr.one
  | Expr.add e1 e2 => Expr.mul (swap e1) (swap e2)
  | Expr.mul e1 e2 => Expr.add (swap e1) (swap e2)

/-- There exists an expression that evaluates to 2014 and still evaluates to 2014 after swapping + and × -/
theorem peters_expression_exists : ∃ e : Expr, eval e = 2014 ∧ eval (swap e) = 2014 := by
  sorry

end NUMINAMATH_CALUDE_peters_expression_exists_l2946_294605


namespace NUMINAMATH_CALUDE_trevor_age_ratio_l2946_294609

/-- The ratio of Trevor's brother's age to Trevor's age 20 years ago -/
def age_ratio : ℚ := 16 / 3

theorem trevor_age_ratio :
  let trevor_age_decade_ago : ℕ := 16
  let brother_current_age : ℕ := 32
  let trevor_current_age : ℕ := trevor_age_decade_ago + 10
  let trevor_age_20_years_ago : ℕ := trevor_current_age - 20
  (brother_current_age : ℚ) / trevor_age_20_years_ago = age_ratio := by
  sorry

end NUMINAMATH_CALUDE_trevor_age_ratio_l2946_294609


namespace NUMINAMATH_CALUDE_rectangles_in_6x6_grid_l2946_294630

/-- The number of ways to choose 2 items from n items -/
def choose_two (n : ℕ) : ℕ := n * (n - 1) / 2

/-- The number of rectangles in a grid of size n x n -/
def rectangles_in_grid (n : ℕ) : ℕ := (choose_two n) ^ 2

/-- Theorem: In a 6x6 grid, the number of rectangles is 225 -/
theorem rectangles_in_6x6_grid : rectangles_in_grid 6 = 225 := by
  sorry

end NUMINAMATH_CALUDE_rectangles_in_6x6_grid_l2946_294630


namespace NUMINAMATH_CALUDE_linear_decreasing_implies_second_or_third_quadrant_l2946_294650

/-- A linear function f(x) = kx + b is monotonically decreasing on ℝ -/
def MonotonicallyDecreasing (k b : ℝ) : Prop :=
  ∀ x y : ℝ, x < y → k * x + b > k * y + b

/-- The point (x, y) is in the second or third quadrant -/
def InSecondOrThirdQuadrant (x y : ℝ) : Prop :=
  x < 0

theorem linear_decreasing_implies_second_or_third_quadrant
  (k b : ℝ) (h : MonotonicallyDecreasing k b) :
  InSecondOrThirdQuadrant k b :=
sorry

end NUMINAMATH_CALUDE_linear_decreasing_implies_second_or_third_quadrant_l2946_294650


namespace NUMINAMATH_CALUDE_sufficient_not_necessary_l2946_294667

theorem sufficient_not_necessary (x : ℝ) :
  (∀ x, 0 < x ∧ x < 5 → -5 < x - 2 ∧ x - 2 < 5) ∧
  (∃ x, -5 < x - 2 ∧ x - 2 < 5 ∧ ¬(0 < x ∧ x < 5)) := by
  sorry

end NUMINAMATH_CALUDE_sufficient_not_necessary_l2946_294667


namespace NUMINAMATH_CALUDE_infinite_solutions_l2946_294617

-- Define the system of linear equations
def equation1 (x y : ℝ) : Prop := 2 * x - 3 * y = 5
def equation2 (x y : ℝ) : Prop := 4 * x - 6 * y = 10

-- Theorem stating that the system has infinitely many solutions
theorem infinite_solutions :
  ∃ (f : ℝ → ℝ × ℝ), ∀ (t : ℝ),
    let (x, y) := f t
    equation1 x y ∧ equation2 x y :=
sorry

end NUMINAMATH_CALUDE_infinite_solutions_l2946_294617


namespace NUMINAMATH_CALUDE_slope_angle_of_sqrt_three_line_l2946_294632

theorem slope_angle_of_sqrt_three_line :
  let line : ℝ → ℝ := λ x ↦ Real.sqrt 3 * x
  let slope : ℝ := Real.sqrt 3
  let angle : ℝ := 60 * Real.pi / 180
  (∀ x, line x = slope * x) ∧
  slope = Real.tan angle :=
by sorry

end NUMINAMATH_CALUDE_slope_angle_of_sqrt_three_line_l2946_294632


namespace NUMINAMATH_CALUDE_quadratic_root_m_l2946_294691

theorem quadratic_root_m (m : ℝ) : ((-1 : ℝ)^2 + m * (-1) - 5 = 0) → m = -4 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_root_m_l2946_294691


namespace NUMINAMATH_CALUDE_loan_interest_rate_l2946_294681

/-- Given a loan of $220 repaid with $242 after one year, prove the annual interest rate is 10% -/
theorem loan_interest_rate : 
  let principal : ℝ := 220
  let total_repayment : ℝ := 242
  let interest_rate : ℝ := (total_repayment - principal) / principal * 100
  interest_rate = 10 := by
  sorry

end NUMINAMATH_CALUDE_loan_interest_rate_l2946_294681


namespace NUMINAMATH_CALUDE_circle_equation_m_range_l2946_294653

/-- Given that the equation x^2 + y^2 - x + y + m = 0 represents a circle,
    prove that m < 1/2 -/
theorem circle_equation_m_range (m : ℝ) :
  (∃ (r : ℝ), r > 0 ∧ ∀ (x y : ℝ), x^2 + y^2 - x + y + m = 0 ↔ (x - 1/2)^2 + (y + 1/2)^2 = r^2) →
  m < 1/2 :=
by sorry

end NUMINAMATH_CALUDE_circle_equation_m_range_l2946_294653


namespace NUMINAMATH_CALUDE_min_packages_required_l2946_294661

/-- Represents a floor in the apartment building -/
inductive Floor
| First
| Second
| Third

/-- Calculates the number of times a specific digit appears on a floor -/
def digit_count (floor : Floor) (digit : ℕ) : ℕ :=
  match floor with
  | Floor.First => if digit = 1 then 52 else 0
  | Floor.Second => if digit = 2 then 52 else 0
  | Floor.Third => if digit = 3 then 52 else 0

/-- Theorem stating the minimum number of packages required -/
theorem min_packages_required : 
  (∀ (floor : Floor) (digit : ℕ), digit_count floor digit ≤ 52) ∧ 
  (∃ (floor : Floor) (digit : ℕ), digit_count floor digit = 52) → 
  (∀ n : ℕ, n < 52 → ¬(∀ (floor : Floor) (digit : ℕ), digit_count floor digit ≤ n)) :=
by sorry

end NUMINAMATH_CALUDE_min_packages_required_l2946_294661


namespace NUMINAMATH_CALUDE_xyz_value_l2946_294676

theorem xyz_value (x y z : ℝ) 
  (h1 : (x + y + z) * (x * y + x * z + y * z) = 40)
  (h2 : x^2 * (y + z) + y^2 * (x + z) + z^2 * (x + y) = 10)
  (h3 : x + y = 2 * z) :
  x * y * z = 6 := by sorry

end NUMINAMATH_CALUDE_xyz_value_l2946_294676


namespace NUMINAMATH_CALUDE_geometric_sequence_sum_l2946_294619

/-- Given a geometric sequence {aₙ} where the sum of the first n terms
    is given by Sₙ = a·2^(n-1) + 1/6, prove that a = -1/3 -/
theorem geometric_sequence_sum (a : ℝ) : 
  (∀ n : ℕ, ∃ Sn : ℝ, Sn = a * 2^(n-1) + 1/6) → a = -1/3 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_sum_l2946_294619


namespace NUMINAMATH_CALUDE_area_between_chords_approx_l2946_294649

noncomputable def circle_area_between_chords (r : ℝ) (d : ℝ) : ℝ :=
  let h := d / 2
  let chord_half_length := Real.sqrt (r^2 - h^2)
  let triangle_area := h * chord_half_length
  let angle := Real.arccos (h / r)
  let sector_area := (angle / (2 * Real.pi)) * Real.pi * r^2
  2 * (sector_area - triangle_area)

theorem area_between_chords_approx : 
  ∃ (ε : ℝ), ε > 0 ∧ ε < 0.01 ∧ 
  |circle_area_between_chords 10 12 - 44.73| < ε :=
sorry

end NUMINAMATH_CALUDE_area_between_chords_approx_l2946_294649


namespace NUMINAMATH_CALUDE_fixed_point_on_line_l2946_294613

/-- For any real number m, the line mx-y+1-3m=0 passes through the point (3, 1) -/
theorem fixed_point_on_line (m : ℝ) : m * 3 - 1 + 1 - 3 * m = 0 := by
  sorry

end NUMINAMATH_CALUDE_fixed_point_on_line_l2946_294613


namespace NUMINAMATH_CALUDE_gcd_digits_bound_l2946_294644

theorem gcd_digits_bound (a b : ℕ) (ha : 10^6 ≤ a ∧ a < 10^7) (hb : 10^6 ≤ b ∧ b < 10^7) 
  (hlcm : 10^10 ≤ Nat.lcm a b ∧ Nat.lcm a b < 10^11) : 
  Nat.gcd a b < 10^4 := by
sorry

end NUMINAMATH_CALUDE_gcd_digits_bound_l2946_294644


namespace NUMINAMATH_CALUDE_middle_number_value_l2946_294668

theorem middle_number_value 
  (a b c d e f g h i j k : ℝ) 
  (h1 : (a + b + c + d + e + f) / 6 = 10.5)
  (h2 : (f + g + h + i + j + k) / 6 = 11.4)
  (h3 : (a + b + c + d + e + f + g + h + i + j + k) / 11 = 9.9)
  (h4 : a + b + c = i + j + k) :
  f = 22.5 := by
    sorry

end NUMINAMATH_CALUDE_middle_number_value_l2946_294668


namespace NUMINAMATH_CALUDE_star_op_equation_solution_l2946_294640

-- Define the "※" operation
def star_op (a b : ℝ) : ℝ := a * b^2 + 2 * a * b

-- State the theorem
theorem star_op_equation_solution :
  ∃! x : ℝ, star_op 1 x = -1 ∧ x = -1 := by
  sorry

end NUMINAMATH_CALUDE_star_op_equation_solution_l2946_294640


namespace NUMINAMATH_CALUDE_max_value_z_l2946_294637

/-- The maximum value of z given the constraints -/
theorem max_value_z (x y : ℝ) (h1 : x - y ≤ 0) (h2 : 4 * x - y ≥ 0) (h3 : x + y ≤ 3) :
  ∃ (z : ℝ), z = x + 2 * y - 1 / x ∧ z ≤ 4 ∧ ∀ (w : ℝ), w = x + 2 * y - 1 / x → w ≤ z :=
by sorry

end NUMINAMATH_CALUDE_max_value_z_l2946_294637


namespace NUMINAMATH_CALUDE_superbloom_probability_l2946_294695

def campus : Finset Char := {'C', 'A', 'M', 'P', 'U', 'S'}
def sherbert : Finset Char := {'S', 'H', 'E', 'R', 'B', 'E', 'R', 'T'}
def globe : Finset Char := {'G', 'L', 'O', 'B', 'E'}
def superbloom : Finset Char := {'S', 'U', 'P', 'E', 'R', 'B', 'L', 'O', 'O', 'M'}

def probability_campus : ℚ := 1 / (campus.card.choose 3)
def probability_sherbert : ℚ := 9 / (sherbert.card.choose 5)
def probability_globe : ℚ := 1

theorem superbloom_probability :
  probability_campus * probability_sherbert * probability_globe = 9 / 1120 := by
  sorry

end NUMINAMATH_CALUDE_superbloom_probability_l2946_294695


namespace NUMINAMATH_CALUDE_sin_two_phi_l2946_294666

theorem sin_two_phi (φ : ℝ) (h : (7 : ℝ) / 13 + Real.sin φ = Real.cos φ) :
  Real.sin (2 * φ) = 120 / 169 := by
  sorry

end NUMINAMATH_CALUDE_sin_two_phi_l2946_294666


namespace NUMINAMATH_CALUDE_trig_identity_l2946_294623

theorem trig_identity (α φ : ℝ) : 
  4 * Real.cos α * Real.cos φ * Real.cos (α - φ) - 
  2 * (Real.cos (α - φ))^2 - Real.cos (2 * φ) = 
  Real.cos (2 * α) := by
  sorry

end NUMINAMATH_CALUDE_trig_identity_l2946_294623


namespace NUMINAMATH_CALUDE_min_red_chips_l2946_294626

/-- Represents a box of colored chips -/
structure ChipBox where
  red : ℕ
  white : ℕ
  blue : ℕ

/-- Checks if a ChipBox satisfies the given conditions -/
def isValidChipBox (box : ChipBox) : Prop :=
  box.blue ≥ box.white / 2 ∧
  box.blue ≤ box.red / 3 ∧
  box.white + box.blue ≥ 55

/-- The theorem stating the minimum number of red chips -/
theorem min_red_chips (box : ChipBox) :
  isValidChipBox box → box.red ≥ 57 := by
  sorry

end NUMINAMATH_CALUDE_min_red_chips_l2946_294626


namespace NUMINAMATH_CALUDE_parallel_planes_counterexample_l2946_294607

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the necessary relations
variable (subset : Line → Plane → Prop)
variable (parallel : Plane → Plane → Prop)
variable (not_parallel : Line → Plane → Prop)

-- State the theorem
theorem parallel_planes_counterexample 
  (a b : Line) (α β γ : Plane) : 
  ¬ (∀ (a b : Line) (α β γ : Plane), 
    (subset a α ∧ subset b α ∧ not_parallel a β ∧ not_parallel b β) 
    → ¬(parallel α β)) :=
sorry

end NUMINAMATH_CALUDE_parallel_planes_counterexample_l2946_294607


namespace NUMINAMATH_CALUDE_library_books_difference_l2946_294616

theorem library_books_difference (initial_books borrowed_books : ℕ) 
  (h1 : initial_books = 75)
  (h2 : borrowed_books = 18) :
  initial_books - borrowed_books = 57 := by
sorry

end NUMINAMATH_CALUDE_library_books_difference_l2946_294616


namespace NUMINAMATH_CALUDE_tangent_equation_solution_l2946_294641

open Real

theorem tangent_equation_solution :
  ∃! y : ℝ, 0 ≤ y ∧ y < 2 * π ∧
  tan (150 * π / 180 - y) = (sin (150 * π / 180) - sin y) / (cos (150 * π / 180) - cos y) →
  y = 0 ∨ y = 2 * π := by
sorry

end NUMINAMATH_CALUDE_tangent_equation_solution_l2946_294641


namespace NUMINAMATH_CALUDE_z_tetromino_placement_count_l2946_294600

/-- Represents a chessboard -/
structure Chessboard :=
  (size : Nat)

/-- Represents a tetromino -/
structure Tetromino :=
  (shape : String)

/-- Calculates the number of ways to place a rectangle on a chessboard -/
def placeRectangle (board : Chessboard) (width : Nat) (height : Nat) : Nat :=
  (board.size - width + 1) * (board.size - height + 1)

/-- Calculates the total number of ways to place a Z-shaped tetromino on a chessboard -/
def placeZTetromino (board : Chessboard) (tetromino : Tetromino) : Nat :=
  2 * (placeRectangle board 2 3 + placeRectangle board 3 2)

/-- The main theorem stating the number of ways to place a Z-shaped tetromino on an 8x8 chessboard -/
theorem z_tetromino_placement_count :
  let board : Chessboard := ⟨8⟩
  let tetromino : Tetromino := ⟨"Z"⟩
  placeZTetromino board tetromino = 168 := by
  sorry


end NUMINAMATH_CALUDE_z_tetromino_placement_count_l2946_294600


namespace NUMINAMATH_CALUDE_total_salary_after_layoffs_l2946_294683

def total_employees : ℕ := 450
def employees_2000 : ℕ := 150
def employees_2500 : ℕ := 200
def employees_3000 : ℕ := 100

def layoff_round1_2000 : ℚ := 0.20
def layoff_round1_2500 : ℚ := 0.25
def layoff_round1_3000 : ℚ := 0.15

def layoff_round2_2000 : ℚ := 0.10
def layoff_round2_2500 : ℚ := 0.15
def layoff_round2_3000 : ℚ := 0.05

def salary_2000 : ℕ := 2000
def salary_2500 : ℕ := 2500
def salary_3000 : ℕ := 3000

theorem total_salary_after_layoffs :
  let remaining_2000 := employees_2000 - ⌊employees_2000 * layoff_round1_2000⌋ - ⌊(employees_2000 - ⌊employees_2000 * layoff_round1_2000⌋) * layoff_round2_2000⌋
  let remaining_2500 := employees_2500 - ⌊employees_2500 * layoff_round1_2500⌋ - ⌊(employees_2500 - ⌊employees_2500 * layoff_round1_2500⌋) * layoff_round2_2500⌋
  let remaining_3000 := employees_3000 - ⌊employees_3000 * layoff_round1_3000⌋ - ⌊(employees_3000 - ⌊employees_3000 * layoff_round1_3000⌋) * layoff_round2_3000⌋
  remaining_2000 * salary_2000 + remaining_2500 * salary_2500 + remaining_3000 * salary_3000 = 776500 := by
sorry

end NUMINAMATH_CALUDE_total_salary_after_layoffs_l2946_294683


namespace NUMINAMATH_CALUDE_path_bounds_l2946_294652

/-- Represents a tile with two segments -/
structure Tile :=
  (segments : Fin 2 → Unit)

/-- Represents a 2N × 2N board assembled with tiles -/
structure Board (N : ℕ) :=
  (tiles : Fin (4 * N^2) → Tile)

/-- The number of paths on a board -/
def num_paths (N : ℕ) (board : Board N) : ℕ := sorry

theorem path_bounds (N : ℕ) (board : Board N) :
  4 * N ≤ num_paths N board ∧ num_paths N board ≤ 2 * N^2 + 2 * N :=
sorry

end NUMINAMATH_CALUDE_path_bounds_l2946_294652


namespace NUMINAMATH_CALUDE_colin_speed_proof_l2946_294685

theorem colin_speed_proof (bruce_speed tony_speed brandon_speed colin_speed : ℝ) : 
  bruce_speed = 1 →
  tony_speed = 2 * bruce_speed →
  brandon_speed = (1 / 3) * tony_speed →
  colin_speed = 4 →
  ∃ (multiple : ℝ), colin_speed = multiple * brandon_speed ∧ colin_speed = 4 := by
sorry

end NUMINAMATH_CALUDE_colin_speed_proof_l2946_294685


namespace NUMINAMATH_CALUDE_probability_score_difference_not_exceeding_three_l2946_294665

def group_A : List ℕ := [88, 89, 90]
def group_B : List ℕ := [87, 88, 92]

def total_possibilities : ℕ := group_A.length * group_B.length

def favorable_outcomes : ℕ :=
  (group_A.length * group_B.length) - 
  (group_A.filter (λ x => x = 88)).length * 
  (group_B.filter (λ x => x = 92)).length

theorem probability_score_difference_not_exceeding_three :
  (favorable_outcomes : ℚ) / total_possibilities = 8 / 9 := by
  sorry

end NUMINAMATH_CALUDE_probability_score_difference_not_exceeding_three_l2946_294665


namespace NUMINAMATH_CALUDE_counterexample_exists_l2946_294660

theorem counterexample_exists : ∃ (a b : ℕ), 
  (∃ (k : ℕ), a^7 = b^3 * k) ∧ 
  ¬(∃ (m : ℕ), a^2 = b * m) := by
  sorry

end NUMINAMATH_CALUDE_counterexample_exists_l2946_294660


namespace NUMINAMATH_CALUDE_lines_intersect_at_point_l2946_294657

-- Define the two lines in parametric form
def line1 (t : ℝ) : ℝ × ℝ := (1 - 2*t, 2 + 6*t)
def line2 (u : ℝ) : ℝ × ℝ := (3 + u, 8 + 3*u)

-- Define the intersection point
def intersection_point : ℝ × ℝ := (1, 2)

-- Theorem statement
theorem lines_intersect_at_point :
  ∃! p : ℝ × ℝ, (∃ t : ℝ, line1 t = p) ∧ (∃ u : ℝ, line2 u = p) ∧ p = intersection_point :=
sorry

end NUMINAMATH_CALUDE_lines_intersect_at_point_l2946_294657


namespace NUMINAMATH_CALUDE_B_is_criminal_l2946_294608

-- Define the set of individuals
inductive Individual : Type
  | A : Individual
  | B : Individual
  | C : Individual

-- Define a predicate for committing the crime
def committed_crime (x : Individual) : Prop := sorry

-- Define A's statement
def A_statement : Prop := ¬ committed_crime Individual.A

-- State the theorem
theorem B_is_criminal :
  (∃! x : Individual, committed_crime x) →
  A_statement →
  (A_statement ↔ true) →
  committed_crime Individual.B := by sorry

end NUMINAMATH_CALUDE_B_is_criminal_l2946_294608


namespace NUMINAMATH_CALUDE_position_number_difference_l2946_294629

structure Student where
  initial_i : ℤ
  initial_j : ℤ
  new_m : ℤ
  new_n : ℤ

def movement (s : Student) : ℤ × ℤ :=
  (s.initial_i - s.new_m, s.initial_j - s.new_n)

def position_number (s : Student) : ℤ :=
  let (a, b) := movement s
  a + b

def sum_position_numbers (students : List Student) : ℤ :=
  students.map position_number |>.sum

theorem position_number_difference (students : List Student) :
  ∃ (S_max S_min : ℤ),
    (∀ s, sum_position_numbers s ≤ S_max ∧ sum_position_numbers s ≥ S_min) ∧
    S_max - S_min = 12 :=
sorry

end NUMINAMATH_CALUDE_position_number_difference_l2946_294629


namespace NUMINAMATH_CALUDE_range_of_a_l2946_294646

theorem range_of_a (a : ℝ) : 
  (¬ ∃ t : ℝ, t^2 - 2*t - a < 0) → 
  (∀ x : ℝ, x ≤ a → x ≤ -1) ∧ a ≤ -1 :=
sorry

end NUMINAMATH_CALUDE_range_of_a_l2946_294646


namespace NUMINAMATH_CALUDE_largest_integer_with_mean_seven_l2946_294638

theorem largest_integer_with_mean_seven (a b c : ℕ) : 
  a ≠ b ∧ b ≠ c ∧ a ≠ c →
  a > 0 ∧ b > 0 ∧ c > 0 →
  (a + b + c) / 3 = 7 →
  ∀ x : ℕ, (x = a ∨ x = b ∨ x = c) → x ≤ 18 :=
by sorry

end NUMINAMATH_CALUDE_largest_integer_with_mean_seven_l2946_294638


namespace NUMINAMATH_CALUDE_bd_squared_equals_sixteen_l2946_294654

theorem bd_squared_equals_sixteen
  (h1 : a - b - c + d = 13)
  (h2 : a + b - c - d = 5)
  (h3 : 3 * a - 2 * b + 4 * c - d = 17)
  : (b - d)^2 = 16 := by
  sorry

end NUMINAMATH_CALUDE_bd_squared_equals_sixteen_l2946_294654


namespace NUMINAMATH_CALUDE_folded_area_ratio_l2946_294693

/-- Represents a rectangular paper with specific folding properties -/
structure FoldedPaper where
  width : ℝ
  length : ℝ
  area : ℝ
  foldedArea : ℝ
  lengthWidthRatio : length = Real.sqrt 3 * width
  areaDefinition : area = length * width
  foldedAreaDefinition : foldedArea = area - (Real.sqrt 3 * width^2) / 6

/-- The ratio of the folded area to the original area is 5/6 -/
theorem folded_area_ratio (paper : FoldedPaper) : 
  paper.foldedArea / paper.area = 5 / 6 := by
  sorry


end NUMINAMATH_CALUDE_folded_area_ratio_l2946_294693


namespace NUMINAMATH_CALUDE_max_visible_cubes_12x12x12_l2946_294634

/-- Represents a cube composed of unit cubes -/
structure Cube where
  size : ℕ

/-- Calculates the maximum number of visible unit cubes from a single point -/
def maxVisibleUnitCubes (c : Cube) : ℕ :=
  3 * c.size^2 - 3 * (c.size - 1) + 1

/-- Theorem: For a 12×12×12 cube, the maximum number of visible unit cubes is 400 -/
theorem max_visible_cubes_12x12x12 :
  let c : Cube := ⟨12⟩
  maxVisibleUnitCubes c = 400 := by
  sorry

#eval maxVisibleUnitCubes ⟨12⟩

end NUMINAMATH_CALUDE_max_visible_cubes_12x12x12_l2946_294634


namespace NUMINAMATH_CALUDE_right_triangle_side_length_l2946_294659

theorem right_triangle_side_length : ∀ (a b c : ℝ),
  a > 0 → b > 0 → c > 0 →
  c = 17 → a = 15 →
  c^2 = a^2 + b^2 →
  b = 8 := by
sorry

end NUMINAMATH_CALUDE_right_triangle_side_length_l2946_294659


namespace NUMINAMATH_CALUDE_work_completion_time_l2946_294612

/-- The number of days it takes A to complete the work alone -/
def a_days : ℕ := 30

/-- The total payment for the work in Rupees -/
def total_payment : ℕ := 1000

/-- B's share of the payment in Rupees -/
def b_share : ℕ := 600

/-- The number of days it takes B to complete the work alone -/
def b_days : ℕ := 20

theorem work_completion_time :
  a_days = 30 ∧ 
  total_payment = 1000 ∧ 
  b_share = 600 →
  b_days = 20 :=
by sorry

end NUMINAMATH_CALUDE_work_completion_time_l2946_294612


namespace NUMINAMATH_CALUDE_journey_distance_l2946_294677

theorem journey_distance (total_time : ℝ) (speed1 : ℝ) (speed2 : ℝ) : 
  total_time = 40 ∧ speed1 = 20 ∧ speed2 = 30 →
  ∃ (distance : ℝ), 
    distance / speed1 / 2 + distance / speed2 / 2 = total_time ∧ 
    distance = 960 := by
  sorry

end NUMINAMATH_CALUDE_journey_distance_l2946_294677


namespace NUMINAMATH_CALUDE_student_count_l2946_294628

theorem student_count (n : ℕ) (right_rank left_rank : ℕ) 
  (h1 : right_rank = 6) 
  (h2 : left_rank = 5) 
  (h3 : n = right_rank + left_rank - 1) : n = 10 :=
by sorry

end NUMINAMATH_CALUDE_student_count_l2946_294628


namespace NUMINAMATH_CALUDE_find_missing_score_l2946_294645

def scores : List ℕ := [87, 88, 89, 0, 91, 92, 92, 93, 94]

theorem find_missing_score (x : ℕ) (h : x ∈ scores) :
  (List.sum (List.filter (λ y => y ≠ 87 ∧ y ≠ 94) (List.map (λ y => if y = 0 then x else y) scores))) / 7 = 91 →
  x = 2 := by sorry

end NUMINAMATH_CALUDE_find_missing_score_l2946_294645


namespace NUMINAMATH_CALUDE_school_community_count_l2946_294696

theorem school_community_count (total_boys : ℕ) (muslim_percent hindu_percent sikh_percent : ℚ) :
  total_boys = 850 →
  muslim_percent = 40 / 100 →
  hindu_percent = 28 / 100 →
  sikh_percent = 10 / 100 →
  (total_boys : ℚ) * (1 - (muslim_percent + hindu_percent + sikh_percent)) = 187 := by
  sorry

end NUMINAMATH_CALUDE_school_community_count_l2946_294696


namespace NUMINAMATH_CALUDE_solution_set_implies_a_b_values_solution_on_interval_implies_a_range_three_integer_solutions_implies_a_range_l2946_294627

-- Define the function f
def f (x a b : ℝ) : ℝ := x^2 + (3-a)*x + 2 + 2*a + b

-- Theorem 1
theorem solution_set_implies_a_b_values (a b : ℝ) :
  (∀ x, f x a b > 0 ↔ x < -4 ∨ x > 2) →
  a = 1 ∧ b = -12 := by sorry

-- Theorem 2
theorem solution_on_interval_implies_a_range (a b : ℝ) :
  (∃ x ∈ Set.Icc 1 3, f x a b ≤ b) →
  a ≤ -6 ∨ a ≥ 20 := by sorry

-- Theorem 3
theorem three_integer_solutions_implies_a_range (a b : ℝ) :
  (∃! (s : Finset ℤ), s.card = 3 ∧ ∀ x ∈ s, f x a b < 12 + b) →
  (3 ≤ a ∧ a < 4) ∨ (10 < a ∧ a ≤ 11) := by sorry

end NUMINAMATH_CALUDE_solution_set_implies_a_b_values_solution_on_interval_implies_a_range_three_integer_solutions_implies_a_range_l2946_294627


namespace NUMINAMATH_CALUDE_chris_babysitting_hours_l2946_294655

/-- The number of hours Chris worked babysitting -/
def hours_worked : ℕ := 9

/-- The cost of the video game in dollars -/
def video_game_cost : ℕ := 60

/-- The cost of the candy in dollars -/
def candy_cost : ℕ := 5

/-- Chris's hourly rate for babysitting in dollars -/
def hourly_rate : ℕ := 8

/-- The amount of money Chris had left over after purchases -/
def money_left : ℕ := 7

theorem chris_babysitting_hours :
  hours_worked * hourly_rate = video_game_cost + candy_cost + money_left :=
by sorry

end NUMINAMATH_CALUDE_chris_babysitting_hours_l2946_294655


namespace NUMINAMATH_CALUDE_joan_initial_oranges_l2946_294603

/-- Proves that Joan initially picked 37 oranges given the conditions -/
theorem joan_initial_oranges (initial : ℕ) (sold : ℕ) (remaining : ℕ)
  (h1 : sold = 10)
  (h2 : remaining = 27)
  (h3 : initial = remaining + sold) :
  initial = 37 := by
  sorry

end NUMINAMATH_CALUDE_joan_initial_oranges_l2946_294603


namespace NUMINAMATH_CALUDE_circle_equation_l2946_294679

/-- A circle C with center (h, k) and radius r -/
structure Circle where
  h : ℝ
  k : ℝ
  r : ℝ

/-- The line x - 2y - 1 = 0 -/
def line (x y : ℝ) : Prop := x - 2*y - 1 = 0

/-- A point (x, y) lies on the circle C -/
def on_circle (C : Circle) (x y : ℝ) : Prop :=
  (x - C.h)^2 + (y - C.k)^2 = C.r^2

theorem circle_equation : ∃ C : Circle,
  (line C.h C.k) ∧
  (on_circle C 0 0) ∧
  (on_circle C 1 2) ∧
  (C.h = 7/4 ∧ C.k = 3/8 ∧ C.r^2 = 205/64) :=
sorry

end NUMINAMATH_CALUDE_circle_equation_l2946_294679


namespace NUMINAMATH_CALUDE_probability_both_primary_l2946_294692

/-- Represents the types of schools in the area -/
inductive SchoolType
| Primary
| Middle
| University

/-- Represents the total number of schools of each type -/
def totalSchools : SchoolType → Nat
| SchoolType.Primary => 21
| SchoolType.Middle => 14
| SchoolType.University => 7

/-- Represents the number of schools selected in stratified sampling -/
def selectedSchools : SchoolType → Nat
| SchoolType.Primary => 3
| SchoolType.Middle => 2
| SchoolType.University => 1

/-- The total number of schools selected -/
def totalSelected : Nat := 6

/-- The number of ways to choose 2 schools from the selected primary schools -/
def waysToChoosePrimary : Nat := 3

/-- The total number of ways to choose 2 schools from all selected schools -/
def totalWaysToChoose : Nat := 15

theorem probability_both_primary :
  (waysToChoosePrimary : Rat) / totalWaysToChoose = 1 / 5 := by
  sorry


end NUMINAMATH_CALUDE_probability_both_primary_l2946_294692


namespace NUMINAMATH_CALUDE_inequality_solution_l2946_294682

theorem inequality_solution :
  ∀ x : ℝ, (2 < (3 * x) / (4 * x - 7) ∧ (3 * x) / (4 * x - 7) ≤ 9) ↔ 
    (21 / 11 < x ∧ x ≤ 14 / 5) := by
  sorry

end NUMINAMATH_CALUDE_inequality_solution_l2946_294682


namespace NUMINAMATH_CALUDE_constant_distance_vector_l2946_294624

variable {V : Type*} [NormedAddCommGroup V] [InnerProductSpace ℝ V]

theorem constant_distance_vector (a b p : V) :
  ‖p - b‖ = 3 * ‖p - a‖ →
  ∃ (c : ℝ), ∀ (q : V), ‖p - q‖ = c ↔ q = (9/8 : ℝ) • a - (1/8 : ℝ) • b :=
sorry

end NUMINAMATH_CALUDE_constant_distance_vector_l2946_294624


namespace NUMINAMATH_CALUDE_employee_earnings_l2946_294689

/-- Calculates the total earnings for an employee based on their work schedule and pay rates. -/
theorem employee_earnings (regular_rate : ℝ) (overtime_multiplier : ℝ) (regular_hours : ℝ) 
  (first_three_days_hours : ℝ) (last_two_days_multiplier : ℝ) : 
  regular_rate = 30 →
  overtime_multiplier = 1.5 →
  regular_hours = 40 →
  first_three_days_hours = 6 →
  last_two_days_multiplier = 2 →
  let overtime_rate := regular_rate * overtime_multiplier
  let last_two_days_hours := first_three_days_hours * last_two_days_multiplier
  let total_hours := first_three_days_hours * 3 + last_two_days_hours * 2
  let overtime_hours := max (total_hours - regular_hours) 0
  let regular_pay := min total_hours regular_hours * regular_rate
  let overtime_pay := overtime_hours * overtime_rate
  regular_pay + overtime_pay = 1290 := by
sorry

end NUMINAMATH_CALUDE_employee_earnings_l2946_294689


namespace NUMINAMATH_CALUDE_quadratic_discriminant_l2946_294643

/-- The discriminant of the quadratic equation 2x^2 + (2 + 1/2)x + 1/2 is 9/4 -/
theorem quadratic_discriminant : 
  let a : ℚ := 2
  let b : ℚ := 2 + 1/2
  let c : ℚ := 1/2
  (b^2 - 4*a*c) = 9/4 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_discriminant_l2946_294643


namespace NUMINAMATH_CALUDE_new_person_weight_l2946_294662

theorem new_person_weight (initial_count : ℕ) (weight_increase : ℝ) (replaced_weight : ℝ) : 
  initial_count = 8 → 
  weight_increase = 2.5 → 
  replaced_weight = 55 → 
  (initial_count : ℝ) * weight_increase + replaced_weight = 75 :=
by sorry

end NUMINAMATH_CALUDE_new_person_weight_l2946_294662


namespace NUMINAMATH_CALUDE_repeating_decimal_to_fraction_l2946_294673

/-- The repeating decimal 0.5656... is equal to the fraction 56/99 -/
theorem repeating_decimal_to_fraction : 
  (∑' n, (56 : ℚ) / (100 ^ (n + 1))) = 56 / 99 := by
  sorry

end NUMINAMATH_CALUDE_repeating_decimal_to_fraction_l2946_294673


namespace NUMINAMATH_CALUDE_knitting_time_for_two_pairs_l2946_294611

/-- Given A's and B's knitting rates, prove the time needed to knit two pairs of socks together -/
theorem knitting_time_for_two_pairs 
  (rate_A : ℚ) -- A's knitting rate in pairs per day
  (rate_B : ℚ) -- B's knitting rate in pairs per day
  (h_rate_A : rate_A = 1/3) -- A can knit a pair in 3 days
  (h_rate_B : rate_B = 1/6) -- B can knit a pair in 6 days
  : (2 : ℚ) / (rate_A + rate_B) = 4 := by
  sorry

end NUMINAMATH_CALUDE_knitting_time_for_two_pairs_l2946_294611


namespace NUMINAMATH_CALUDE_same_color_eyes_percentage_l2946_294647

/-- Represents the proportion of students with a specific eye color combination -/
structure EyeColorProportion where
  eggCream : ℝ    -- proportion of students with eggshell and cream eyes
  eggCorn : ℝ     -- proportion of students with eggshell and cornsilk eyes
  eggEgg : ℝ      -- proportion of students with both eggshell eyes
  creamCorn : ℝ   -- proportion of students with cream and cornsilk eyes
  creamCream : ℝ  -- proportion of students with both cream eyes
  cornCorn : ℝ    -- proportion of students with both cornsilk eyes

/-- The conditions given in the problem -/
def eyeColorConditions (p : EyeColorProportion) : Prop :=
  p.eggCream + p.eggCorn + p.eggEgg = 0.3 ∧
  p.eggCream + p.creamCorn + p.creamCream = 0.4 ∧
  p.eggCorn + p.creamCorn + p.cornCorn = 0.5 ∧
  p.eggCream + p.eggCorn + p.eggEgg + p.creamCorn + p.creamCream + p.cornCorn = 1

/-- The theorem to be proved -/
theorem same_color_eyes_percentage (p : EyeColorProportion) 
  (h : eyeColorConditions p) : 
  p.eggEgg + p.creamCream + p.cornCorn = 0.8 := by
  sorry


end NUMINAMATH_CALUDE_same_color_eyes_percentage_l2946_294647


namespace NUMINAMATH_CALUDE_unique_three_digit_number_l2946_294656

/-- Represents a three-digit number as a tuple of its digits -/
def ThreeDigitNumber := (Nat × Nat × Nat)

/-- Converts a ThreeDigitNumber to its numerical value -/
def to_nat (n : ThreeDigitNumber) : Nat :=
  100 * n.1 + 10 * n.2.1 + n.2.2

/-- Checks if the digits of a ThreeDigitNumber are distinct -/
def has_distinct_digits (n : ThreeDigitNumber) : Prop :=
  n.1 ≠ n.2.1 ∧ n.1 ≠ n.2.2 ∧ n.2.1 ≠ n.2.2

/-- The main theorem stating that 156 is the only number satisfying the conditions -/
theorem unique_three_digit_number : 
  ∀ n : ThreeDigitNumber, 
    has_distinct_digits n → 
    (100 ≤ to_nat n) ∧ (to_nat n ≤ 999) → 
    (to_nat n = (n.1 + n.2.1 + n.2.2) * (n.1 + n.2.1 + n.2.2 + 1)) → 
    n = (1, 5, 6) :=
by sorry

end NUMINAMATH_CALUDE_unique_three_digit_number_l2946_294656


namespace NUMINAMATH_CALUDE_dice_faces_theorem_l2946_294621

theorem dice_faces_theorem (n m : ℕ) : 
  (n ≥ 1) → 
  (m ≥ 1) → 
  (∀ i ∈ Finset.range n, ∀ j ∈ Finset.range m, 
    (Finset.filter (λ (p : ℕ × ℕ) => p.1 + p.2 = 8) (Finset.product (Finset.range n) (Finset.range m))).card = 
    (1/2 : ℚ) * (Finset.filter (λ (p : ℕ × ℕ) => p.1 + p.2 = 11) (Finset.product (Finset.range n) (Finset.range m))).card) →
  ((Finset.filter (λ (p : ℕ × ℕ) => p.1 + p.2 = 13) (Finset.product (Finset.range n) (Finset.range m))).card : ℚ) / (n * m) = 1/15 →
  (∃ k : ℕ, n + m = 5 * k) →
  (∀ n' m' : ℕ, n' + m' < n + m → 
    ¬((n' ≥ 1) ∧ 
      (m' ≥ 1) ∧ 
      (∀ i ∈ Finset.range n', ∀ j ∈ Finset.range m', 
        (Finset.filter (λ (p : ℕ × ℕ) => p.1 + p.2 = 8) (Finset.product (Finset.range n') (Finset.range m'))).card = 
        (1/2 : ℚ) * (Finset.filter (λ (p : ℕ × ℕ) => p.1 + p.2 = 11) (Finset.product (Finset.range n') (Finset.range m'))).card) ∧
      ((Finset.filter (λ (p : ℕ × ℕ) => p.1 + p.2 = 13) (Finset.product (Finset.range n') (Finset.range m'))).card : ℚ) / (n' * m') = 1/15 ∧
      (∃ k : ℕ, n' + m' = 5 * k))) →
  n + m = 25 := by
sorry

end NUMINAMATH_CALUDE_dice_faces_theorem_l2946_294621


namespace NUMINAMATH_CALUDE_at_least_three_equal_l2946_294672

theorem at_least_three_equal (a b c d : ℕ) 
  (h1 : (a + b)^2 % (c * d) = 0)
  (h2 : (a + c)^2 % (b * d) = 0)
  (h3 : (a + d)^2 % (b * c) = 0)
  (h4 : (b + c)^2 % (a * d) = 0)
  (h5 : (b + d)^2 % (a * c) = 0)
  (h6 : (c + d)^2 % (a * b) = 0) :
  (a = b ∧ b = c) ∨ (a = b ∧ b = d) ∨ (a = c ∧ c = d) ∨ (b = c ∧ c = d) := by
sorry

end NUMINAMATH_CALUDE_at_least_three_equal_l2946_294672


namespace NUMINAMATH_CALUDE_trig_identity_l2946_294690

theorem trig_identity (x y : ℝ) : 
  Real.sin (x + y) * Real.sin x + Real.cos (x + y) * Real.cos x = Real.cos y := by
  sorry

end NUMINAMATH_CALUDE_trig_identity_l2946_294690


namespace NUMINAMATH_CALUDE_joanne_earnings_theorem_l2946_294678

/-- Calculates Joanne's total weekly earnings based on her work schedule and pay rates -/
def joanne_weekly_earnings (main_job_hours_per_day : ℕ) (main_job_rate : ℚ) 
  (part_time_hours_per_day : ℕ) (part_time_rate : ℚ) (days_per_week : ℕ) : ℚ :=
  (main_job_hours_per_day * main_job_rate + part_time_hours_per_day * part_time_rate) * days_per_week

/-- Theorem stating that Joanne's weekly earnings are $775.00 -/
theorem joanne_earnings_theorem : 
  joanne_weekly_earnings 8 16 2 (27/2) 5 = 775 := by
  sorry

end NUMINAMATH_CALUDE_joanne_earnings_theorem_l2946_294678


namespace NUMINAMATH_CALUDE_tangyuan_purchase_solution_l2946_294651

/-- Represents the number and price of tangyuan bags for two brands -/
structure TangyuanPurchase where
  brandA_quantity : ℕ
  brandB_quantity : ℕ
  brandA_price : ℕ
  brandB_price : ℕ

/-- Checks if a TangyuanPurchase satisfies all conditions -/
def is_valid_purchase (p : TangyuanPurchase) : Prop :=
  p.brandA_quantity + p.brandB_quantity = 1000 ∧
  p.brandA_quantity = 2 * p.brandB_quantity + 20 ∧
  p.brandB_price = p.brandA_price + 6 ∧
  5 * p.brandA_price = 3 * p.brandB_price

/-- The theorem to be proved -/
theorem tangyuan_purchase_solution :
  ∃ (p : TangyuanPurchase),
    is_valid_purchase p ∧
    p.brandA_quantity = 670 ∧
    p.brandB_quantity = 330 ∧
    p.brandA_price = 9 ∧
    p.brandB_price = 15 :=
  sorry

end NUMINAMATH_CALUDE_tangyuan_purchase_solution_l2946_294651


namespace NUMINAMATH_CALUDE_arithmetic_subsequence_l2946_294684

theorem arithmetic_subsequence (a : ℕ → ℝ) (d : ℝ) (h : ∀ n : ℕ, a (n + 1) = a n + d) :
  ∃ c : ℝ, ∀ k : ℕ+, a (3 * k - 1) = c + (k - 1) * (3 * d) :=
sorry

end NUMINAMATH_CALUDE_arithmetic_subsequence_l2946_294684


namespace NUMINAMATH_CALUDE_fraction_multiplication_l2946_294622

theorem fraction_multiplication :
  (3 : ℚ) / 4 * 4 / 5 * 5 / 6 * 6 / 7 = 3 / 7 := by
  sorry

end NUMINAMATH_CALUDE_fraction_multiplication_l2946_294622


namespace NUMINAMATH_CALUDE_division_problem_l2946_294648

theorem division_problem (L S q : ℕ) : 
  L - S = 2415 → 
  L = 2520 → 
  L = S * q + 15 → 
  q = 23 := by sorry

end NUMINAMATH_CALUDE_division_problem_l2946_294648


namespace NUMINAMATH_CALUDE_arithmetic_mean_reciprocals_first_five_primes_l2946_294633

def first_five_primes : List Nat := [2, 3, 5, 7, 11]

theorem arithmetic_mean_reciprocals_first_five_primes :
  let reciprocals := first_five_primes.map (λ x => (1 : ℚ) / x)
  (reciprocals.sum / reciprocals.length) = 2927 / 11550 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_mean_reciprocals_first_five_primes_l2946_294633


namespace NUMINAMATH_CALUDE_candy_distribution_l2946_294687

theorem candy_distribution (total : ℝ) (total_pos : total > 0) : 
  let initial_shares := [4/10, 3/10, 2/10, 1/10]
  let first_round := initial_shares.map (· * total)
  let remaining_after_first := total - first_round.sum
  let second_round := initial_shares.map (· * remaining_after_first)
  remaining_after_first - second_round.sum = 0 := by
sorry

end NUMINAMATH_CALUDE_candy_distribution_l2946_294687


namespace NUMINAMATH_CALUDE_area_of_triangle_FNV_l2946_294669

-- Define the rectangle EFGH
structure Rectangle where
  EF : ℝ
  EH : ℝ

-- Define the trapezoid KWFG
structure Trapezoid where
  KF : ℝ
  WG : ℝ
  height : ℝ
  area : ℝ

-- Define the theorem
theorem area_of_triangle_FNV (rect : Rectangle) (trap : Trapezoid) :
  rect.EF = 15 ∧
  trap.KF = 5 ∧
  trap.WG = 5 ∧
  trap.area = 150 ∧
  trap.KF = trap.WG →
  (1 / 2 : ℝ) * (1 / 2 : ℝ) * (trap.KF + rect.EF) * rect.EH = 125 := by
  sorry


end NUMINAMATH_CALUDE_area_of_triangle_FNV_l2946_294669


namespace NUMINAMATH_CALUDE_correct_operation_l2946_294615

theorem correct_operation : 
  (5 * Real.sqrt 3 - 2 * Real.sqrt 3 ≠ 3) ∧ 
  (2 * Real.sqrt 2 * 3 * Real.sqrt 2 ≠ 6) ∧ 
  (3 * Real.sqrt 3 / Real.sqrt 3 = 3) ∧ 
  (2 * Real.sqrt 3 + 3 * Real.sqrt 2 ≠ 5 * Real.sqrt 5) := by
  sorry

end NUMINAMATH_CALUDE_correct_operation_l2946_294615


namespace NUMINAMATH_CALUDE_triangle_side_length_l2946_294658

-- Define a triangle XYZ
structure Triangle where
  X : Real
  Y : Real
  Z : Real
  x : Real
  y : Real
  z : Real

-- State the theorem
theorem triangle_side_length (t : Triangle) : 
  t.y = 7 → 
  t.z = 3 → 
  Real.cos (t.Y - t.Z) = 17/32 → 
  t.x = Real.sqrt 41 := by
  sorry

end NUMINAMATH_CALUDE_triangle_side_length_l2946_294658


namespace NUMINAMATH_CALUDE_nonzero_terms_count_l2946_294639

/-- The number of nonzero terms in the expansion of (2x+3)(x^2 + 2x + 4) - 2(x^3 + x^2 - 3x + 1) + (x-2)(x+5) is 2 -/
theorem nonzero_terms_count (x : ℝ) : 
  let expansion := (2*x+3)*(x^2 + 2*x + 4) - 2*(x^3 + x^2 - 3*x + 1) + (x-2)*(x+5)
  ∃ (a b : ℝ), expansion = a*x^2 + b*x ∧ a ≠ 0 ∧ b ≠ 0 :=
by sorry

end NUMINAMATH_CALUDE_nonzero_terms_count_l2946_294639

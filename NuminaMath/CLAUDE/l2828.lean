import Mathlib

namespace NUMINAMATH_CALUDE_max_cubes_for_given_prism_l2828_282850

/-- Represents the dimensions and properties of a wooden rectangular prism --/
structure WoodenPrism where
  totalSurfaceArea : ℝ
  cubeSurfaceArea : ℝ
  wastePerCut : ℝ

/-- Calculates the maximum number of cubes that can be sawed from the prism --/
def maxCubes (prism : WoodenPrism) : ℕ :=
  sorry

/-- Theorem stating the maximum number of cubes for the given problem --/
theorem max_cubes_for_given_prism :
  let prism : WoodenPrism := {
    totalSurfaceArea := 2448,
    cubeSurfaceArea := 216,
    wastePerCut := 0.2
  }
  maxCubes prism = 15 := by
  sorry

end NUMINAMATH_CALUDE_max_cubes_for_given_prism_l2828_282850


namespace NUMINAMATH_CALUDE_birds_and_storks_l2828_282844

theorem birds_and_storks (initial_birds : ℕ) (storks : ℕ) (additional_birds : ℕ) : 
  initial_birds = 3 → storks = 4 → additional_birds = 2 →
  (initial_birds + additional_birds) - storks = 1 := by
  sorry

end NUMINAMATH_CALUDE_birds_and_storks_l2828_282844


namespace NUMINAMATH_CALUDE_foal_count_l2828_282821

def animal_count : ℕ := 11
def leg_count : ℕ := 30
def turkey_legs : ℕ := 2
def foal_legs : ℕ := 4

theorem foal_count (t f : ℕ) : 
  t + f = animal_count → 
  turkey_legs * t + foal_legs * f = leg_count → 
  f = 4 :=
by
  sorry

end NUMINAMATH_CALUDE_foal_count_l2828_282821


namespace NUMINAMATH_CALUDE_c_neq_zero_necessary_not_sufficient_l2828_282811

/-- Represents a conic section of the form ax^2 + y^2 = c -/
structure ConicSection where
  a : ℝ
  c : ℝ

/-- Determines if a conic section is an ellipse or hyperbola -/
def is_ellipse_or_hyperbola (conic : ConicSection) : Prop :=
  -- We don't define this explicitly as it's not given in the problem conditions
  sorry

/-- Theorem stating that c ≠ 0 is necessary but not sufficient for
    ax^2 + y^2 = c to represent an ellipse or hyperbola -/
theorem c_neq_zero_necessary_not_sufficient :
  (∀ conic : ConicSection, is_ellipse_or_hyperbola conic → conic.c ≠ 0) ∧
  (∃ conic : ConicSection, conic.c ≠ 0 ∧ ¬is_ellipse_or_hyperbola conic) :=
by
  sorry

end NUMINAMATH_CALUDE_c_neq_zero_necessary_not_sufficient_l2828_282811


namespace NUMINAMATH_CALUDE_min_dominoes_to_win_viktors_winning_strategy_l2828_282809

/-- Represents a square board -/
structure Board :=
  (size : ℕ)

/-- Represents a domino placement on the board -/
structure DominoPlacement :=
  (board : Board)
  (num_dominoes : ℕ)

/-- Theorem: The minimum number of dominoes Viktor needs to fix to win -/
theorem min_dominoes_to_win (b : Board) (d : DominoPlacement) : ℕ :=
  sorry

/-- The main theorem to prove -/
theorem viktors_winning_strategy (b : Board) (d : DominoPlacement) : 
  b.size = 2022 → d.board = b → d.num_dominoes = 2022 * 2022 / 2 → 
  min_dominoes_to_win b d = 1011^2 :=
sorry

end NUMINAMATH_CALUDE_min_dominoes_to_win_viktors_winning_strategy_l2828_282809


namespace NUMINAMATH_CALUDE_point_on_bisector_implies_a_eq_neg_five_l2828_282897

/-- A point P with coordinates (x, y) is on the bisector of the second and fourth quadrants if x + y = 0 -/
def on_bisector (x y : ℝ) : Prop := x + y = 0

/-- Given that point P (a+3, 7+a) is on the bisector of the second and fourth quadrants, prove that a = -5 -/
theorem point_on_bisector_implies_a_eq_neg_five (a : ℝ) :
  on_bisector (a + 3) (7 + a) → a = -5 := by
  sorry

end NUMINAMATH_CALUDE_point_on_bisector_implies_a_eq_neg_five_l2828_282897


namespace NUMINAMATH_CALUDE_valid_squares_count_l2828_282815

/-- Represents a square on the checkerboard -/
structure Square where
  size : Nat
  topLeft : Nat × Nat

/-- Checks if a square contains at least 6 black squares -/
def hasAtLeastSixBlackSquares (s : Square) : Bool :=
  sorry

/-- Counts the number of valid squares on the board -/
def countValidSquares (boardSize : Nat) : Nat :=
  sorry

theorem valid_squares_count :
  countValidSquares 10 = 140 :=
sorry

end NUMINAMATH_CALUDE_valid_squares_count_l2828_282815


namespace NUMINAMATH_CALUDE_speeding_ticket_problem_l2828_282867

theorem speeding_ticket_problem (total_motorists : ℝ) 
  (h1 : total_motorists > 0) 
  (h2 : total_motorists * 0.4 = total_motorists * 0.5 - (total_motorists * 0.5 - total_motorists * 0.4)) :
  (total_motorists * 0.5 - total_motorists * 0.4) / (total_motorists * 0.5) = 0.2 := by
  sorry

#check speeding_ticket_problem

end NUMINAMATH_CALUDE_speeding_ticket_problem_l2828_282867


namespace NUMINAMATH_CALUDE_unknown_bill_denomination_l2828_282878

/-- Represents the number of bills of each denomination --/
structure BillCount where
  twenty : Nat
  ten : Nat
  unknown : Nat

/-- Represents the total value of bills --/
def totalValue (b : BillCount) (unknownDenom : Nat) : Nat :=
  20 * b.twenty + 10 * b.ten + unknownDenom * b.unknown

/-- The problem statement --/
theorem unknown_bill_denomination (b : BillCount) (h1 : b.twenty = 10) (h2 : b.ten = 8) (h3 : b.unknown = 4) :
  ∃ (x : Nat), x = 5 ∧ totalValue b x = 300 := by
  sorry

end NUMINAMATH_CALUDE_unknown_bill_denomination_l2828_282878


namespace NUMINAMATH_CALUDE_f_of_4_equals_15_l2828_282806

/-- A function f(x) = cx^2 + dx + 3 satisfying f(1) = 3 and f(2) = 5 -/
def f (c d : ℝ) (x : ℝ) : ℝ := c * x^2 + d * x + 3

/-- The theorem stating that f(4) = 15 given the conditions -/
theorem f_of_4_equals_15 (c d : ℝ) :
  f c d 1 = 3 → f c d 2 = 5 → f c d 4 = 15 := by
  sorry

#check f_of_4_equals_15

end NUMINAMATH_CALUDE_f_of_4_equals_15_l2828_282806


namespace NUMINAMATH_CALUDE_no_blue_in_red_triangle_l2828_282866

-- Define the color of a point
inductive Color
| Red
| Blue

-- Define a point in the plane with integer coordinates
structure Point where
  x : Int
  y : Int

-- Define the coloring function
def coloring : Point → Color := sorry

-- Define the distance between two points
def distance (p q : Point) : ℝ := sorry

-- Define a predicate for a point being inside a triangle
def inside_triangle (p a b c : Point) : Prop := sorry

-- State the conditions
axiom condition1 : ∀ (p : Point), coloring p = Color.Red ∨ coloring p = Color.Blue

axiom condition2 : ∀ (p q : Point),
  coloring p = Color.Red → coloring q = Color.Red →
  ∀ (r : Point), inside_triangle r p q q → coloring r ≠ Color.Blue

axiom condition3 : ∀ (p q : Point),
  coloring p = Color.Blue → coloring q = Color.Blue →
  distance p q = 2 →
  coloring {x := (p.x + q.x) / 2, y := (p.y + q.y) / 2} = Color.Blue

-- State the theorem
theorem no_blue_in_red_triangle (a b c : Point) :
  coloring a = Color.Red → coloring b = Color.Red → coloring c = Color.Red →
  ∀ (p : Point), inside_triangle p a b c → coloring p ≠ Color.Blue :=
sorry

end NUMINAMATH_CALUDE_no_blue_in_red_triangle_l2828_282866


namespace NUMINAMATH_CALUDE_mixture_ratio_l2828_282822

/-- Given a mixture of liquids p and q, prove that the initial ratio is 3:2 -/
theorem mixture_ratio (p q : ℝ) : 
  p + q = 25 →                      -- Initial total volume
  p / (q + 2) = 5 / 4 →             -- Ratio after adding 2 liters of q
  p / q = 3 / 2 :=                  -- Initial ratio
by sorry

end NUMINAMATH_CALUDE_mixture_ratio_l2828_282822


namespace NUMINAMATH_CALUDE_max_value_abc_l2828_282865

theorem max_value_abc (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  (a * b + 2 * b * c) / (a^2 + b^2 + c^2) ≤ Real.sqrt 5 / 2 := by
  sorry

end NUMINAMATH_CALUDE_max_value_abc_l2828_282865


namespace NUMINAMATH_CALUDE_mirror_image_properties_l2828_282801

-- Define a point in 2D space
structure Point2D where
  x : ℝ
  y : ℝ

-- Define mirror image operations
def mirrorY (p : Point2D) : Point2D :=
  { x := -p.x, y := p.y }

def mirrorX (p : Point2D) : Point2D :=
  { x := p.x, y := -p.y }

def mirrorOrigin (p : Point2D) : Point2D :=
  { x := -p.x, y := -p.y }

def mirrorYEqualsX (p : Point2D) : Point2D :=
  { x := p.y, y := p.x }

def mirrorYEqualsNegX (p : Point2D) : Point2D :=
  { x := -p.y, y := -p.x }

-- Theorem stating the mirror image properties
theorem mirror_image_properties (p : Point2D) :
  (mirrorY p = { x := -p.x, y := p.y }) ∧
  (mirrorX p = { x := p.x, y := -p.y }) ∧
  (mirrorOrigin p = { x := -p.x, y := -p.y }) ∧
  (mirrorYEqualsX p = { x := p.y, y := p.x }) ∧
  (mirrorYEqualsNegX p = { x := -p.y, y := -p.x }) :=
by sorry

end NUMINAMATH_CALUDE_mirror_image_properties_l2828_282801


namespace NUMINAMATH_CALUDE_waiter_customers_l2828_282813

/-- Calculates the total number of customers for a waiter given the number of tables and customers per table. -/
def total_customers (num_tables : ℕ) (women_per_table : ℕ) (men_per_table : ℕ) : ℕ :=
  num_tables * (women_per_table + men_per_table)

/-- Theorem stating that a waiter with 5 tables, each having 5 women and 3 men, has a total of 40 customers. -/
theorem waiter_customers :
  total_customers 5 5 3 = 40 := by
  sorry

end NUMINAMATH_CALUDE_waiter_customers_l2828_282813


namespace NUMINAMATH_CALUDE_units_digit_of_1389_pow_1247_l2828_282834

theorem units_digit_of_1389_pow_1247 (n : ℕ) :
  n = 1389^1247 → n % 10 = 9 := by
  sorry

end NUMINAMATH_CALUDE_units_digit_of_1389_pow_1247_l2828_282834


namespace NUMINAMATH_CALUDE_area_of_U_l2828_282804

/-- A regular octagon centered at the origin in the complex plane -/
def regularOctagon : Set ℂ :=
  sorry

/-- The distance between opposite sides of the octagon is 2 units -/
def oppositeDistanceIs2 : ℝ :=
  sorry

/-- One pair of sides of the octagon is parallel to the real axis -/
def sideParallelToRealAxis : Prop :=
  sorry

/-- The region outside the octagon -/
def T : Set ℂ :=
  {z : ℂ | z ∉ regularOctagon}

/-- The set of reciprocals of points in T -/
def U : Set ℂ :=
  {w : ℂ | ∃ z ∈ T, w = 1 / z}

/-- The area of a set in the complex plane -/
def area : Set ℂ → ℝ :=
  sorry

theorem area_of_U : area U = π / 2 :=
  sorry

end NUMINAMATH_CALUDE_area_of_U_l2828_282804


namespace NUMINAMATH_CALUDE_john_overall_loss_l2828_282849

def grinder_cost : ℝ := 15000
def mobile_cost : ℝ := 8000
def bicycle_cost : ℝ := 12000
def laptop_cost : ℝ := 25000

def grinder_loss_percent : ℝ := 0.02
def mobile_profit_percent : ℝ := 0.10
def bicycle_profit_percent : ℝ := 0.15
def laptop_loss_percent : ℝ := 0.08

def total_cost : ℝ := grinder_cost + mobile_cost + bicycle_cost + laptop_cost

def grinder_sale : ℝ := grinder_cost * (1 - grinder_loss_percent)
def mobile_sale : ℝ := mobile_cost * (1 + mobile_profit_percent)
def bicycle_sale : ℝ := bicycle_cost * (1 + bicycle_profit_percent)
def laptop_sale : ℝ := laptop_cost * (1 - laptop_loss_percent)

def total_sale : ℝ := grinder_sale + mobile_sale + bicycle_sale + laptop_sale

theorem john_overall_loss : total_sale - total_cost = -700 := by sorry

end NUMINAMATH_CALUDE_john_overall_loss_l2828_282849


namespace NUMINAMATH_CALUDE_intersection_of_three_lines_l2828_282898

theorem intersection_of_three_lines (k : ℝ) : 
  (∃ x y : ℝ, y = 7 * x - 2 ∧ y = -3 * x + 14 ∧ y = 4 * x + k) → k = 2.8 := by
  sorry

end NUMINAMATH_CALUDE_intersection_of_three_lines_l2828_282898


namespace NUMINAMATH_CALUDE_unused_sector_angle_l2828_282879

/-- Given a circular piece of paper with radius r, from which a cone is formed
    with base radius 10 cm and volume 500π cm³, prove that the central angle
    of the unused sector is approximately 130.817°. -/
theorem unused_sector_angle (r : ℝ) : 
  r > 0 →
  (1 / 3 * π * 10^2 * (r^2 - 10^2).sqrt = 500 * π) →
  abs (360 - (20 * π / (2 * π * r)) * 360 - 130.817) < 0.001 := by
sorry


end NUMINAMATH_CALUDE_unused_sector_angle_l2828_282879


namespace NUMINAMATH_CALUDE_conical_container_height_l2828_282882

theorem conical_container_height (d : ℝ) (n : ℕ) (h r : ℝ) : 
  d = 64 ∧ n = 4 ∧ (π * d^2 / 4) = n * (π * r * (d / 2)) ∧ h^2 + r^2 = (d / 2)^2 
  → h = 8 * Real.sqrt 15 := by sorry

end NUMINAMATH_CALUDE_conical_container_height_l2828_282882


namespace NUMINAMATH_CALUDE_quadratic_equation_root_l2828_282855

theorem quadratic_equation_root (k : ℝ) : 
  (∃ x : ℂ, x^2 + 4*x + k = 0 ∧ x = -2 + 3*Complex.I) → k = 13 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_equation_root_l2828_282855


namespace NUMINAMATH_CALUDE_initial_money_calculation_l2828_282858

theorem initial_money_calculation (x : ℤ) : 
  ((x + 9) - 19 = 35) → (x = 45) := by
  sorry

end NUMINAMATH_CALUDE_initial_money_calculation_l2828_282858


namespace NUMINAMATH_CALUDE_no_arithmetic_mean_l2828_282861

theorem no_arithmetic_mean (f1 f2 f3 : ℚ) : 
  f1 = 5/8 ∧ f2 = 3/4 ∧ f3 = 9/12 →
  (f1 ≠ (f2 + f3) / 2) ∧ (f2 ≠ (f1 + f3) / 2) ∧ (f3 ≠ (f1 + f2) / 2) :=
by sorry

#check no_arithmetic_mean

end NUMINAMATH_CALUDE_no_arithmetic_mean_l2828_282861


namespace NUMINAMATH_CALUDE_square_diff_product_l2828_282842

theorem square_diff_product (m n : ℝ) (h1 : m - n = 4) (h2 : m * n = -3) :
  (m^2 - 4) * (n^2 - 4) = -15 := by sorry

end NUMINAMATH_CALUDE_square_diff_product_l2828_282842


namespace NUMINAMATH_CALUDE_candy_bar_distribution_l2828_282894

theorem candy_bar_distribution (total_candy_bars : ℝ) (num_people : ℝ) 
  (h1 : total_candy_bars = 5.0) 
  (h2 : num_people = 3.0) :
  ∃ (ε : ℝ), ε > 0 ∧ ε < 0.01 ∧ 
  |total_candy_bars / num_people - 1.67| < ε :=
sorry

end NUMINAMATH_CALUDE_candy_bar_distribution_l2828_282894


namespace NUMINAMATH_CALUDE_economics_law_tournament_l2828_282887

theorem economics_law_tournament (n : ℕ) (m : ℕ) : 
  220 < n → n < 254 →
  m < n →
  (n - 2 * m)^2 = n →
  ∀ k : ℕ, (220 < k ∧ k < 254 ∧ k < n ∧ (k - 2 * (n - k))^2 = k) → n - m ≤ k - (n - k) →
  n - m = 105 := by
sorry

end NUMINAMATH_CALUDE_economics_law_tournament_l2828_282887


namespace NUMINAMATH_CALUDE_infinite_geometric_series_first_term_l2828_282828

theorem infinite_geometric_series_first_term
  (a r : ℝ)
  (h1 : 0 ≤ r ∧ r < 1)  -- Condition for convergence of infinite geometric series
  (h2 : a / (1 - r) = 15)  -- Sum of the series
  (h3 : a^2 / (1 - r^2) = 45)  -- Sum of the squares of the terms
  : a = 5 := by
  sorry

end NUMINAMATH_CALUDE_infinite_geometric_series_first_term_l2828_282828


namespace NUMINAMATH_CALUDE_equation1_representation_equation2_representation_l2828_282862

-- Define the equations
def equation1 (x y : ℝ) : Prop := 4 * x^2 + 8 * y^2 + 8 * y * |y| = 1
def equation2 (x y : ℝ) : Prop := 2 * x^2 - 4 * x + 2 + 2 * (x - 1) * |x - 1| + 8 * y^2 - 8 * y * |y| = 1

-- Define the regions for equation1
def upper_ellipse (x y : ℝ) : Prop := y ≥ 0 ∧ 4 * x^2 + 16 * y^2 = 1
def vertical_lines (x y : ℝ) : Prop := y < 0 ∧ (x = 1/2 ∨ x = -1/2)

-- Define the regions for equation2
def elliptic_part (x y : ℝ) : Prop := x ≥ 1 ∧ 4 * (x - 1)^2 + 16 * y^2 = 1
def vertical_section (x y : ℝ) : Prop := x < 1 ∧ y = -1/4

-- Theorem statements
theorem equation1_representation :
  ∀ x y : ℝ, equation1 x y ↔ (upper_ellipse x y ∨ vertical_lines x y) :=
sorry

theorem equation2_representation :
  ∀ x y : ℝ, equation2 x y ↔ (elliptic_part x y ∨ vertical_section x y) :=
sorry

end NUMINAMATH_CALUDE_equation1_representation_equation2_representation_l2828_282862


namespace NUMINAMATH_CALUDE_major_axis_length_l2828_282853

/-- Represents a right circular cylinder. -/
structure RightCircularCylinder where
  radius : ℝ

/-- Represents an ellipse formed by the intersection of a plane and a cylinder. -/
structure Ellipse where
  minorAxis : ℝ
  majorAxis : ℝ

/-- The ellipse formed by the intersection of a plane and a right circular cylinder. -/
def intersectionEllipse (c : RightCircularCylinder) : Ellipse where
  minorAxis := 2 * c.radius
  majorAxis := 2 * c.radius * 1.5

theorem major_axis_length 
  (c : RightCircularCylinder) 
  (h : c.radius = 1) :
  (intersectionEllipse c).majorAxis = 3 := by
  sorry

end NUMINAMATH_CALUDE_major_axis_length_l2828_282853


namespace NUMINAMATH_CALUDE_perpendicular_vectors_l2828_282816

def a : Fin 2 → ℝ := ![3, 4]
def b : Fin 2 → ℝ := ![2, -1]

theorem perpendicular_vectors (x : ℝ) : 
  (∀ i : Fin 2, (a + x • b) i * (-b i) = 0) → x = -2/5 := by
  sorry

end NUMINAMATH_CALUDE_perpendicular_vectors_l2828_282816


namespace NUMINAMATH_CALUDE_square_root_of_nine_l2828_282836

theorem square_root_of_nine : ∃ x : ℝ, x ^ 2 = 9 ∧ x = 3 := by sorry

end NUMINAMATH_CALUDE_square_root_of_nine_l2828_282836


namespace NUMINAMATH_CALUDE_fraction_less_than_one_necessary_not_sufficient_l2828_282877

theorem fraction_less_than_one_necessary_not_sufficient (a : ℝ) :
  (∀ a, a > 1 → 1 / a < 1) ∧ (∃ a, 1 / a < 1 ∧ a ≤ 1) := by
  sorry

end NUMINAMATH_CALUDE_fraction_less_than_one_necessary_not_sufficient_l2828_282877


namespace NUMINAMATH_CALUDE_smallest_n_for_factorization_factorization_exists_for_31_l2828_282820

theorem smallest_n_for_factorization : 
  ∀ n : ℤ, n < 31 → 
  ¬∃ A B : ℤ, ∀ x : ℝ, 5 * x^2 + n * x + 48 = (5 * x + A) * (x + B) :=
by sorry

theorem factorization_exists_for_31 : 
  ∃ A B : ℤ, ∀ x : ℝ, 5 * x^2 + 31 * x + 48 = (5 * x + A) * (x + B) :=
by sorry

end NUMINAMATH_CALUDE_smallest_n_for_factorization_factorization_exists_for_31_l2828_282820


namespace NUMINAMATH_CALUDE_josh_has_eight_riddles_l2828_282810

/-- The number of riddles each person has -/
structure Riddles where
  taso : ℕ
  ivory : ℕ
  josh : ℕ

/-- Given conditions about the riddles -/
def riddle_conditions (r : Riddles) : Prop :=
  r.taso = 24 ∧
  r.taso = 2 * r.ivory ∧
  r.ivory = r.josh + 4

/-- Theorem stating that Josh has 8 riddles -/
theorem josh_has_eight_riddles (r : Riddles) 
  (h : riddle_conditions r) : r.josh = 8 := by
  sorry

end NUMINAMATH_CALUDE_josh_has_eight_riddles_l2828_282810


namespace NUMINAMATH_CALUDE_function_properties_l2828_282857

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := -x^2 + 2 * |x - a|

theorem function_properties (a : ℝ) :
  (∀ x, f a x = f a (-x)) ↔ a = 0 ∧
  (a = 1/2 → ∀ x, (x ≤ -1 ∨ (1/2 ≤ x ∧ x ≤ 1)) → 
    ∀ y, y < x → f (1/2) y < f (1/2) x) ∧
  (a > 0 → (∀ x : ℝ, x ≥ 0 → f a (x - 1) ≥ 2 * f a x) ↔ 
    (Real.sqrt 6 - 2 ≤ a ∧ a ≤ 1/2)) :=
by sorry

end NUMINAMATH_CALUDE_function_properties_l2828_282857


namespace NUMINAMATH_CALUDE_triangle_inequality_l2828_282856

theorem triangle_inequality (a b c : ℝ) : |a - c| ≤ |a - b| + |b - c| := by
  sorry

end NUMINAMATH_CALUDE_triangle_inequality_l2828_282856


namespace NUMINAMATH_CALUDE_ratio_problem_l2828_282848

theorem ratio_problem (q r s t u : ℚ) 
  (h1 : q / r = 8)
  (h2 : s / r = 5)
  (h3 : s / t = 1 / 4)
  (h4 : u / t = 3)
  : u / q = 15 / 2 := by
  sorry

end NUMINAMATH_CALUDE_ratio_problem_l2828_282848


namespace NUMINAMATH_CALUDE_shortest_piece_length_l2828_282838

theorem shortest_piece_length (total_length : ℝ) (piece1 piece2 piece3 : ℝ) : 
  total_length = 138 →
  piece1 + piece2 + piece3 = total_length →
  piece1 = 2 * piece2 →
  piece2 = 3 * piece3 →
  piece3 = 13.8 := by
sorry

end NUMINAMATH_CALUDE_shortest_piece_length_l2828_282838


namespace NUMINAMATH_CALUDE_zeros_in_square_of_nines_l2828_282889

/-- The number of zeros in the decimal expansion of (10^8 - 1)² is 7 -/
theorem zeros_in_square_of_nines : ∃ n : ℕ, n = 7 ∧ 
  (∃ m : ℕ, (10^8 - 1)^2 = m * 10^n + k ∧ k < 10^n ∧ k % 10 ≠ 0) := by
  sorry

end NUMINAMATH_CALUDE_zeros_in_square_of_nines_l2828_282889


namespace NUMINAMATH_CALUDE_lily_typing_speed_l2828_282812

/-- Represents Lily's typing scenario -/
structure TypingScenario where
  totalTime : ℕ -- Total time including breaks
  totalWords : ℕ -- Total words typed
  breakInterval : ℕ -- Interval between breaks
  breakDuration : ℕ -- Duration of each break

/-- Calculates the words typed per minute -/
def wordsPerMinute (scenario : TypingScenario) : ℚ :=
  let effectiveTypingTime := scenario.totalTime - (scenario.totalTime / scenario.breakInterval) * scenario.breakDuration
  scenario.totalWords / effectiveTypingTime

/-- Theorem stating that Lily types 15 words per minute -/
theorem lily_typing_speed :
  let scenario : TypingScenario := {
    totalTime := 19
    totalWords := 255
    breakInterval := 10
    breakDuration := 2
  }
  wordsPerMinute scenario = 15 := by
  sorry


end NUMINAMATH_CALUDE_lily_typing_speed_l2828_282812


namespace NUMINAMATH_CALUDE_min_values_xy_and_x_plus_y_l2828_282819

theorem min_values_xy_and_x_plus_y (x y : ℝ) 
  (hx : x > 0) (hy : y > 0) (h : 2/x + 8/y = 1) : 
  (x * y ≥ 64 ∧ x + y ≥ 18) ∧
  (∃ x₁ y₁ x₂ y₂ : ℝ, x₁ * y₁ = 64 ∧ x₂ + y₂ = 18 ∧
   2/x₁ + 8/y₁ = 1 ∧ 2/x₂ + 8/y₂ = 1 ∧
   x₁ > 0 ∧ y₁ > 0 ∧ x₂ > 0 ∧ y₂ > 0) :=
by sorry

end NUMINAMATH_CALUDE_min_values_xy_and_x_plus_y_l2828_282819


namespace NUMINAMATH_CALUDE_ratio_q_p_l2828_282876

def total_slips : ℕ := 50
def distinct_numbers : ℕ := 10
def slips_per_number : ℕ := 5
def drawn_slips : ℕ := 5

def p : ℚ := (distinct_numbers : ℚ) / (Nat.choose total_slips drawn_slips : ℚ)

def q : ℚ := (Nat.choose distinct_numbers 2 * Nat.choose slips_per_number 2 * Nat.choose slips_per_number 2 * (distinct_numbers - 2) : ℚ) / (Nat.choose total_slips drawn_slips : ℚ)

theorem ratio_q_p : q / p = 360 := by sorry

end NUMINAMATH_CALUDE_ratio_q_p_l2828_282876


namespace NUMINAMATH_CALUDE_function_range_condition_l2828_282890

def f (a x : ℝ) : ℝ := (a^2 - 2*a - 3)*x^2 + (a - 3)*x + 1

theorem function_range_condition (a : ℝ) :
  (∀ x, ∃ y, f a x = y) ∧ (∀ y, ∃ x, f a x = y) ↔ a > 3 ∨ a < -1 :=
sorry

end NUMINAMATH_CALUDE_function_range_condition_l2828_282890


namespace NUMINAMATH_CALUDE_min_n_is_correct_l2828_282884

/-- The minimum positive integer n for which (x^5 + 1/x)^n contains a constant term -/
def min_n : ℕ := 6

/-- Predicate to check if (x^5 + 1/x)^n contains a constant term -/
def has_constant_term (n : ℕ) : Prop :=
  ∃ r : ℕ, 5 * n = 6 * r

theorem min_n_is_correct :
  (has_constant_term min_n) ∧
  (∀ m : ℕ, m < min_n → ¬(has_constant_term m)) :=
by sorry

end NUMINAMATH_CALUDE_min_n_is_correct_l2828_282884


namespace NUMINAMATH_CALUDE_xiaohua_mother_age_ratio_l2828_282864

/-- The number of years ago when a mother's age was 5 times her child's age, given their current ages -/
def years_ago (child_age mother_age : ℕ) : ℕ :=
  child_age - (mother_age - child_age) / (5 - 1)

/-- Theorem stating that for Xiaohua (12 years old) and his mother (36 years old), 
    the mother's age was 5 times Xiaohua's age 6 years ago -/
theorem xiaohua_mother_age_ratio : years_ago 12 36 = 6 := by
  sorry

end NUMINAMATH_CALUDE_xiaohua_mother_age_ratio_l2828_282864


namespace NUMINAMATH_CALUDE_permutation_expressions_l2828_282854

open Nat

-- Define the permutation function A
def A (n k : ℕ) : ℕ := factorial n / factorial (n - k)

-- Theorem statement
theorem permutation_expressions (n : ℕ) : 
  (A (n + 1) n ≠ factorial n) ∧ 
  ((1 / (n + 1 : ℚ)) * A (n + 1) (n + 1) = factorial n) ∧
  (A n n = factorial n) ∧
  (n * A (n - 1) (n - 1) = factorial n) :=
sorry

end NUMINAMATH_CALUDE_permutation_expressions_l2828_282854


namespace NUMINAMATH_CALUDE_hyperbola_asymptotes_l2828_282805

/-- Given a hyperbola with equation x²/144 - y²/81 = 1, its asymptotes are y = ±(3/4)x -/
theorem hyperbola_asymptotes :
  ∀ x y : ℝ,
  x^2 / 144 - y^2 / 81 = 1 →
  ∃ m : ℝ, m > 0 ∧ (y = m * x ∨ y = -m * x) ∧ m = 3/4 := by
sorry

end NUMINAMATH_CALUDE_hyperbola_asymptotes_l2828_282805


namespace NUMINAMATH_CALUDE_square_inequality_l2828_282893

theorem square_inequality (α : ℝ) (x : ℝ) (h1 : α ≥ 0) (h2 : (x + 1)^2 ≥ α * (α + 1)) :
  x^2 ≥ α * (α - 1) := by
sorry

end NUMINAMATH_CALUDE_square_inequality_l2828_282893


namespace NUMINAMATH_CALUDE_father_son_age_ratio_l2828_282846

/-- Proves that the ratio of father's age to son's age is 4:1 given the conditions -/
theorem father_son_age_ratio :
  ∀ (father_age son_age : ℕ),
    father_age = 64 →
    son_age = 16 →
    father_age - 10 + son_age - 10 = 60 →
    father_age / son_age = 4 := by
  sorry

end NUMINAMATH_CALUDE_father_son_age_ratio_l2828_282846


namespace NUMINAMATH_CALUDE_negation_equivalence_l2828_282829

theorem negation_equivalence :
  (¬ ∃ x : ℝ, 2 * x^2 - 1 ≤ 0) ↔ (∀ x : ℝ, 2 * x^2 - 1 > 0) :=
by sorry

end NUMINAMATH_CALUDE_negation_equivalence_l2828_282829


namespace NUMINAMATH_CALUDE_arithmetic_sum_l2828_282840

def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sum (a : ℕ → ℝ) :
  arithmetic_sequence a →
  a 1 = 2 →
  a 2 + a 3 = 13 →
  a 4 + a 5 + a 6 = 42 :=
by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sum_l2828_282840


namespace NUMINAMATH_CALUDE_no_equilateral_triangle_2D_exists_regular_tetrahedron_3D_l2828_282832

-- Define a 2D point with integer coordinates
structure Point2D where
  x : ℤ
  y : ℤ

-- Define a 3D point with integer coordinates
structure Point3D where
  x : ℤ
  y : ℤ
  z : ℤ

-- Function to calculate the square of the distance between two 2D points
def distanceSquared2D (p1 p2 : Point2D) : ℤ :=
  (p1.x - p2.x)^2 + (p1.y - p2.y)^2

-- Function to calculate the square of the distance between two 3D points
def distanceSquared3D (p1 p2 : Point3D) : ℤ :=
  (p1.x - p2.x)^2 + (p1.y - p2.y)^2 + (p1.z - p2.z)^2

-- Theorem: No equilateral triangle exists with vertices at integer coordinate points in 2D
theorem no_equilateral_triangle_2D :
  ¬∃ (a b c : Point2D), 
    distanceSquared2D a b = distanceSquared2D b c ∧
    distanceSquared2D b c = distanceSquared2D c a ∧
    distanceSquared2D c a = distanceSquared2D a b :=
sorry

-- Theorem: A regular tetrahedron exists with vertices at integer coordinate points in 3D
theorem exists_regular_tetrahedron_3D :
  ∃ (a b c d : Point3D),
    distanceSquared3D a b = distanceSquared3D b c ∧
    distanceSquared3D b c = distanceSquared3D c d ∧
    distanceSquared3D c d = distanceSquared3D d a ∧
    distanceSquared3D d a = distanceSquared3D a b ∧
    distanceSquared3D a c = distanceSquared3D b d :=
sorry

end NUMINAMATH_CALUDE_no_equilateral_triangle_2D_exists_regular_tetrahedron_3D_l2828_282832


namespace NUMINAMATH_CALUDE_journey_speeds_correct_l2828_282833

/-- Represents the speeds and meeting times of pedestrians and cyclists --/
structure JourneyData where
  distance : ℝ
  pedestrian_start : ℝ
  cyclist1_start : ℝ
  cyclist2_start : ℝ
  pedestrian_speed : ℝ
  cyclist_speed : ℝ

/-- Checks if the given speeds satisfy the journey conditions --/
def satisfies_conditions (data : JourneyData) : Prop :=
  let first_meeting_time := data.cyclist1_start + (data.distance / 2 - data.pedestrian_speed * (data.cyclist1_start - data.pedestrian_start)) / (data.cyclist_speed - data.pedestrian_speed)
  let second_meeting_time := first_meeting_time + 1
  let pedestrian_distance_at_second_meeting := data.pedestrian_speed * (second_meeting_time - data.pedestrian_start)
  let cyclist2_distance := data.cyclist_speed * (second_meeting_time - data.cyclist2_start)
  first_meeting_time - data.pedestrian_start > 0 ∧
  first_meeting_time - data.cyclist1_start > 0 ∧
  second_meeting_time - data.cyclist2_start > 0 ∧
  pedestrian_distance_at_second_meeting + cyclist2_distance = data.distance

/-- The main theorem stating that the given speeds satisfy the journey conditions --/
theorem journey_speeds_correct : ∃ (data : JourneyData),
  data.distance = 40 ∧
  data.pedestrian_start = 0 ∧
  data.cyclist1_start = 10/3 ∧
  data.cyclist2_start = 4.5 ∧
  data.pedestrian_speed = 5 ∧
  data.cyclist_speed = 30 ∧
  satisfies_conditions data := by
  sorry


end NUMINAMATH_CALUDE_journey_speeds_correct_l2828_282833


namespace NUMINAMATH_CALUDE_last_week_viewers_correct_l2828_282839

/-- The number of people who watched the baseball games last week -/
def last_week_viewers : ℕ := 200

/-- The number of people who watched the second game this week -/
def second_game_viewers : ℕ := 80

/-- The number of people who watched the first game this week -/
def first_game_viewers : ℕ := second_game_viewers - 20

/-- The number of people who watched the third game this week -/
def third_game_viewers : ℕ := second_game_viewers + 15

/-- The total number of people who watched the games this week -/
def this_week_total : ℕ := first_game_viewers + second_game_viewers + third_game_viewers

/-- The difference in viewers between this week and last week -/
def viewer_difference : ℕ := 35

theorem last_week_viewers_correct : 
  last_week_viewers = this_week_total - viewer_difference := by
  sorry

end NUMINAMATH_CALUDE_last_week_viewers_correct_l2828_282839


namespace NUMINAMATH_CALUDE_triangle_problem_l2828_282835

theorem triangle_problem (A B C : Real) (a b c : Real) :
  -- Conditions
  a + b + c = 10 →
  Real.sin B + Real.sin C = 4 * Real.sin A →
  -- Part 1
  a = 2 ∧
  -- Additional condition for Part 2
  b * c = 16 →
  -- Part 2
  Real.cos A = 7/8 := by
sorry

end NUMINAMATH_CALUDE_triangle_problem_l2828_282835


namespace NUMINAMATH_CALUDE_gcd_372_684_l2828_282868

theorem gcd_372_684 : Nat.gcd 372 684 = 12 := by
  sorry

end NUMINAMATH_CALUDE_gcd_372_684_l2828_282868


namespace NUMINAMATH_CALUDE_better_misspellings_l2828_282800

/-- The word to be considered -/
def word : String := "better"

/-- The number of distinct letters in the word -/
def distinct_letters : Nat := 4

/-- The total number of letters in the word -/
def total_letters : Nat := 6

/-- The number of repeated letters in the word -/
def repeated_letters : Nat := 2

/-- The number of repetitions for each repeated letter -/
def repetitions : Nat := 2

/-- The number of misspellings of the word "better" -/
def misspellings : Nat := 179

theorem better_misspellings :
  (Nat.factorial total_letters / (Nat.factorial repetitions ^ repeated_letters)) - 1 = misspellings :=
sorry

end NUMINAMATH_CALUDE_better_misspellings_l2828_282800


namespace NUMINAMATH_CALUDE_num_connecting_lines_correct_l2828_282872

/-- The number of straight lines connecting the intersection points of n intersecting lines -/
def num_connecting_lines (n : ℕ) : ℚ :=
  (n^2 * (n-1)^2 - 2*n * (n-1)) / 8

/-- Theorem stating that num_connecting_lines gives the correct number of lines -/
theorem num_connecting_lines_correct (n : ℕ) :
  num_connecting_lines n = (n^2 * (n-1)^2 - 2*n * (n-1)) / 8 :=
by sorry

end NUMINAMATH_CALUDE_num_connecting_lines_correct_l2828_282872


namespace NUMINAMATH_CALUDE_gate_buyers_pay_more_l2828_282814

/-- Calculates the difference in total amount paid between gate buyers and pre-buyers --/
def ticketPriceDifference (preBuyerCount : ℕ) (gateBuyerCount : ℕ) (preBuyerPrice : ℕ) (gateBuyerPrice : ℕ) : ℕ :=
  gateBuyerCount * gateBuyerPrice - preBuyerCount * preBuyerPrice

theorem gate_buyers_pay_more :
  ticketPriceDifference 20 30 155 200 = 2900 := by
  sorry

end NUMINAMATH_CALUDE_gate_buyers_pay_more_l2828_282814


namespace NUMINAMATH_CALUDE_geometric_series_equality_l2828_282837

def C (n : ℕ) : ℚ := 2048 * (1 - (1 / 2^n))

def D (n : ℕ) : ℚ := (6144 / 3) * (1 - (1 / (-2)^n))

theorem geometric_series_equality (n : ℕ) (h : n ≥ 1) : C n = D n ↔ n = 6 := by
  sorry

end NUMINAMATH_CALUDE_geometric_series_equality_l2828_282837


namespace NUMINAMATH_CALUDE_equation_solution_l2828_282871

theorem equation_solution : ∃ x₁ x₂ : ℝ, x₁ = -5 ∧ x₂ = 3 ∧ 
  (∀ x : ℝ, (x + 3) * (x - 1) = 12 ↔ x = x₁ ∨ x = x₂) :=
by sorry

end NUMINAMATH_CALUDE_equation_solution_l2828_282871


namespace NUMINAMATH_CALUDE_sample_size_eq_selected_cards_l2828_282824

/-- Represents a statistical study of student report cards -/
structure ReportCardStudy where
  totalStudents : ℕ
  selectedCards : ℕ
  h_total : totalStudents = 1000
  h_selected : selectedCards = 100
  h_selected_le_total : selectedCards ≤ totalStudents

/-- The sample size of a report card study is equal to the number of selected cards -/
theorem sample_size_eq_selected_cards (study : ReportCardStudy) : 
  study.selectedCards = 100 := by
  sorry

#check sample_size_eq_selected_cards

end NUMINAMATH_CALUDE_sample_size_eq_selected_cards_l2828_282824


namespace NUMINAMATH_CALUDE_duck_price_is_correct_l2828_282873

/-- The price of a duck given the conditions of the problem -/
def duck_price : ℝ :=
  let chicken_price : ℝ := 8
  let num_chickens : ℕ := 5
  let num_ducks : ℕ := 2
  let additional_earnings : ℝ := 60
  10

theorem duck_price_is_correct :
  let chicken_price : ℝ := 8
  let num_chickens : ℕ := 5
  let num_ducks : ℕ := 2
  let additional_earnings : ℝ := 60
  let total_earnings := chicken_price * num_chickens + duck_price * num_ducks
  let wheelbarrow_cost := total_earnings / 2
  wheelbarrow_cost * 2 = additional_earnings ∧ duck_price = 10 := by
  sorry

end NUMINAMATH_CALUDE_duck_price_is_correct_l2828_282873


namespace NUMINAMATH_CALUDE_fixed_point_of_function_l2828_282852

theorem fixed_point_of_function (a : ℝ) (h1 : a > 0) (h2 : a ≠ 1) :
  let f := fun x : ℝ => a^(1 - x) - 2
  f 1 = -1 := by sorry

end NUMINAMATH_CALUDE_fixed_point_of_function_l2828_282852


namespace NUMINAMATH_CALUDE_simple_interest_problem_l2828_282807

/-- Proves that given a sum P put at simple interest for 4 years, 
    if increasing the interest rate by 2% results in $56 more interest, 
    then P = $700. -/
theorem simple_interest_problem (P R : ℝ) : 
  (P * (R + 2) * 4) / 100 - (P * R * 4) / 100 = 56 → P = 700 := by
  sorry

end NUMINAMATH_CALUDE_simple_interest_problem_l2828_282807


namespace NUMINAMATH_CALUDE_point_330_ratio_l2828_282859

/-- A point on the terminal side of a 330° angle, excluding the origin -/
structure Point330 where
  x : ℝ
  y : ℝ
  nonzero : x ≠ 0 ∨ y ≠ 0
  on_terminal_side : y / x = Real.tan (330 * π / 180)

/-- The ratio y/x for a point on the terminal side of a 330° angle is -√3/3 -/
theorem point_330_ratio (P : Point330) : P.y / P.x = -Real.sqrt 3 / 3 := by
  sorry

end NUMINAMATH_CALUDE_point_330_ratio_l2828_282859


namespace NUMINAMATH_CALUDE_quadratic_two_distinct_roots_l2828_282827

theorem quadratic_two_distinct_roots (a : ℝ) :
  (∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ (a - 3) * x₁^2 - 4 * x₁ - 1 = 0 ∧ (a - 3) * x₂^2 - 4 * x₂ - 1 = 0) ↔
  (a > -1 ∧ a ≠ 3) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_two_distinct_roots_l2828_282827


namespace NUMINAMATH_CALUDE_cloth_sale_problem_l2828_282817

/-- Proves that the number of meters of cloth sold is 400 -/
theorem cloth_sale_problem (total_selling_price : ℕ) (loss_per_meter : ℕ) (cost_price_per_meter : ℕ) :
  total_selling_price = 18000 →
  loss_per_meter = 5 →
  cost_price_per_meter = 50 →
  (total_selling_price / (cost_price_per_meter - loss_per_meter) : ℕ) = 400 := by
  sorry

end NUMINAMATH_CALUDE_cloth_sale_problem_l2828_282817


namespace NUMINAMATH_CALUDE_set_operations_and_intersection_l2828_282869

def A : Set ℝ := {x | 4 ≤ x ∧ x < 8}
def B : Set ℝ := {x | 2 < x ∧ x < 10}
def C (a : ℝ) : Set ℝ := {x | x < a}

theorem set_operations_and_intersection :
  (A ∪ B = {x | 2 < x ∧ x < 10}) ∧
  ((Aᶜ ∩ B) = {x | (8 ≤ x ∧ x < 10) ∨ (2 < x ∧ x < 4)}) ∧
  (∀ a : ℝ, (A ∩ C a).Nonempty ↔ a > 4) :=
sorry

end NUMINAMATH_CALUDE_set_operations_and_intersection_l2828_282869


namespace NUMINAMATH_CALUDE_unique_solution_condition_smallest_divisor_double_factorial_divides_sum_double_factorial_l2828_282895

-- Definition of double factorial
def double_factorial : ℕ → ℕ
  | 0 => 1
  | 1 => 1
  | n + 2 => (n + 2) * double_factorial n

-- Theorem for the first part of the problem
theorem unique_solution_condition (a : ℝ) :
  (∃! x y : ℝ, x + y - 144 = 0 ∧ x * y - 5184 - 0.1 * a^2 = 0) ↔ a = 0 := by sorry

-- Theorem for the second part of the problem
theorem smallest_divisor_double_factorial :
  ∀ n : ℕ, n > 2022 → n ∣ (double_factorial 2021 + double_factorial 2022) → n ≥ 2023 := by sorry

-- Theorem that 2023 divides the sum of double factorials
theorem divides_sum_double_factorial :
  2023 ∣ (double_factorial 2021 + double_factorial 2022) := by sorry

end NUMINAMATH_CALUDE_unique_solution_condition_smallest_divisor_double_factorial_divides_sum_double_factorial_l2828_282895


namespace NUMINAMATH_CALUDE_arccos_cos_three_l2828_282892

theorem arccos_cos_three : Real.arccos (Real.cos 3) = 3 := by
  sorry

end NUMINAMATH_CALUDE_arccos_cos_three_l2828_282892


namespace NUMINAMATH_CALUDE_adlai_has_two_dogs_l2828_282891

/-- The number of legs a dog has -/
def dog_legs : ℕ := 4

/-- The number of legs a chicken has -/
def chicken_legs : ℕ := 2

/-- The total number of animal legs -/
def total_legs : ℕ := 10

/-- The number of chickens Adlai has -/
def num_chickens : ℕ := 1

/-- Theorem stating that Adlai has 2 dogs -/
theorem adlai_has_two_dogs : 
  ∃ (num_dogs : ℕ), num_dogs * dog_legs + num_chickens * chicken_legs = total_legs ∧ num_dogs = 2 :=
sorry

end NUMINAMATH_CALUDE_adlai_has_two_dogs_l2828_282891


namespace NUMINAMATH_CALUDE_correct_calculation_l2828_282830

theorem correct_calculation (x : ℤ) : x - 32 = 33 → x + 32 = 97 := by
  sorry

end NUMINAMATH_CALUDE_correct_calculation_l2828_282830


namespace NUMINAMATH_CALUDE_circle_chords_and_regions_l2828_282808

/-- The number of chords that can be drawn between n points on a circle's circumference -/
def num_chords (n : ℕ) : ℕ := n.choose 2

/-- The number of regions formed inside a circle by chords connecting n points on its circumference -/
def num_regions (n : ℕ) : ℕ := 1 + n.choose 2 + n.choose 4

theorem circle_chords_and_regions (n : ℕ) (h : n = 10) :
  num_chords n = 45 ∧ num_regions n = 256 := by
  sorry

#eval num_chords 10
#eval num_regions 10

end NUMINAMATH_CALUDE_circle_chords_and_regions_l2828_282808


namespace NUMINAMATH_CALUDE_only_vertical_angles_true_l2828_282896

-- Define the propositions
def proposition1 := "Non-intersecting lines are parallel lines"
def proposition2 := "Corresponding angles are equal"
def proposition3 := "If the squares of two real numbers are equal, then the two real numbers are also equal"
def proposition4 := "Vertical angles are equal"

-- Define a function to check if a proposition is true
def is_true (p : String) : Prop :=
  p = proposition4

-- Theorem statement
theorem only_vertical_angles_true :
  (is_true proposition1 = false) ∧
  (is_true proposition2 = false) ∧
  (is_true proposition3 = false) ∧
  (is_true proposition4 = true) :=
by
  sorry


end NUMINAMATH_CALUDE_only_vertical_angles_true_l2828_282896


namespace NUMINAMATH_CALUDE_eggs_to_buy_l2828_282860

def total_eggs_needed : ℕ := 222
def eggs_received : ℕ := 155

theorem eggs_to_buy : total_eggs_needed - eggs_received = 67 := by
  sorry

end NUMINAMATH_CALUDE_eggs_to_buy_l2828_282860


namespace NUMINAMATH_CALUDE_sum_of_squares_and_product_l2828_282888

theorem sum_of_squares_and_product (x y : ℝ) : 
  x = 2 / (Real.sqrt 3 + 1) →
  y = 2 / (Real.sqrt 3 - 1) →
  x^2 + x*y + y^2 = 10 := by sorry

end NUMINAMATH_CALUDE_sum_of_squares_and_product_l2828_282888


namespace NUMINAMATH_CALUDE_triangle_problem_l2828_282831

theorem triangle_problem (A B C : ℝ) (a b c : ℝ) :
  0 < A ∧ 0 < B ∧ 0 < C ∧  -- Acute triangle
  A + B + C = π ∧
  a ≠ b ∧
  c = Real.sqrt 3 ∧
  Real.sqrt 3 * (Real.cos A)^2 - Real.sqrt 3 * (Real.cos B)^2 = Real.sin A * Real.cos A - Real.sin B * Real.cos B ∧
  Real.sin A = 4/5 →
  C = π/6 ∧
  1/2 * a * c * Real.sin B = (24 * Real.sqrt 3 + 18) / 25 :=
by sorry

end NUMINAMATH_CALUDE_triangle_problem_l2828_282831


namespace NUMINAMATH_CALUDE_science_club_enrollment_l2828_282818

theorem science_club_enrollment (total : ℕ) (chem : ℕ) (bio : ℕ) (both : ℕ) 
  (h1 : total = 75)
  (h2 : chem = 45)
  (h3 : bio = 30)
  (h4 : both = 18) :
  total - (chem + bio - both) = 18 := by
  sorry

end NUMINAMATH_CALUDE_science_club_enrollment_l2828_282818


namespace NUMINAMATH_CALUDE_linear_function_constant_point_l2828_282886

theorem linear_function_constant_point :
  ∀ (k : ℝ), (2 * k - 1) * 2 - (k + 3) * 3 - (k - 11) = 0 := by
  sorry

end NUMINAMATH_CALUDE_linear_function_constant_point_l2828_282886


namespace NUMINAMATH_CALUDE_complex_equation_sum_l2828_282875

theorem complex_equation_sum (a b : ℝ) : 
  (a + Complex.I) * Complex.I = b + (5 / (2 - Complex.I)) → a + b = -2 := by
  sorry

end NUMINAMATH_CALUDE_complex_equation_sum_l2828_282875


namespace NUMINAMATH_CALUDE_max_value_of_x_plus_inverse_l2828_282845

theorem max_value_of_x_plus_inverse (x : ℝ) (h : 11 = x^2 + 1/x^2) :
  ∃ (max : ℝ), max = Real.sqrt 13 ∧ x + 1/x ≤ max :=
by sorry

end NUMINAMATH_CALUDE_max_value_of_x_plus_inverse_l2828_282845


namespace NUMINAMATH_CALUDE_tan_negative_3645_degrees_l2828_282885

theorem tan_negative_3645_degrees : Real.tan ((-3645 : ℝ) * π / 180) = -1 := by
  sorry

end NUMINAMATH_CALUDE_tan_negative_3645_degrees_l2828_282885


namespace NUMINAMATH_CALUDE_two_sided_icing_cubes_count_l2828_282823

/-- Represents a 3D coordinate --/
structure Coord where
  x : Nat
  y : Nat
  z : Nat

/-- Represents a cake with dimensions and icing information --/
structure Cake where
  dim : Nat
  hasIcingTop : Bool
  hasIcingBottom : Bool
  hasIcingSides : Bool

/-- Counts the number of unit cubes with icing on exactly two sides --/
def countTwoSidedIcingCubes (c : Cake) : Nat :=
  sorry

/-- The main theorem to prove --/
theorem two_sided_icing_cubes_count (c : Cake) : 
  c.dim = 4 ∧ c.hasIcingTop ∧ ¬c.hasIcingBottom ∧ c.hasIcingSides → 
  countTwoSidedIcingCubes c = 20 := by
  sorry

end NUMINAMATH_CALUDE_two_sided_icing_cubes_count_l2828_282823


namespace NUMINAMATH_CALUDE_factor_quadratic_l2828_282880

theorem factor_quadratic (t : ℝ) : 
  (∃ k : ℝ, ∀ x : ℝ, -6 * x^2 + 17 * x + 7 = k * (x - t)) ↔ 
  (t = (17 + Real.sqrt 457) / 12 ∨ t = (17 - Real.sqrt 457) / 12) :=
by sorry

end NUMINAMATH_CALUDE_factor_quadratic_l2828_282880


namespace NUMINAMATH_CALUDE_hospital_bill_ambulance_cost_l2828_282863

theorem hospital_bill_ambulance_cost 
  (total_bill : ℝ)
  (medication_percentage : ℝ)
  (overnight_percentage : ℝ)
  (food_cost : ℝ)
  (h1 : total_bill = 5000)
  (h2 : medication_percentage = 0.5)
  (h3 : overnight_percentage = 0.25)
  (h4 : food_cost = 175) :
  let medication_cost := medication_percentage * total_bill
  let remaining_after_medication := total_bill - medication_cost
  let overnight_cost := overnight_percentage * remaining_after_medication
  let remaining_after_overnight := remaining_after_medication - overnight_cost
  let ambulance_cost := remaining_after_overnight - food_cost
  ambulance_cost = 1700 := by sorry

end NUMINAMATH_CALUDE_hospital_bill_ambulance_cost_l2828_282863


namespace NUMINAMATH_CALUDE_flower_garden_mystery_l2828_282843

/-- Represents a digit (0-9) -/
def Digit := Fin 10

/-- Represents the arrangement of digits in the problem -/
structure Arrangement where
  garden : Fin 10000
  love : Fin 100
  unknown : Fin 100

/-- The conditions of the problem -/
def problem_conditions (a : Arrangement) : Prop :=
  ∃ (flower : Digit),
    a.garden + 6 = 85613 ∧
    a.love = 41 + a.unknown ∧
    a.garden.val = flower.val * 1000 + 9 * 100 + flower.val * 10 + 3

/-- The main theorem: proving that "花园探秘" equals 9713 -/
theorem flower_garden_mystery (a : Arrangement) 
  (h : problem_conditions a) : a.garden = 9713 := by
  sorry


end NUMINAMATH_CALUDE_flower_garden_mystery_l2828_282843


namespace NUMINAMATH_CALUDE_cream_fraction_is_three_tenths_l2828_282899

-- Define the initial contents of the cups
def initial_A : ℚ := 8
def initial_B : ℚ := 6
def initial_C : ℚ := 4

-- Define the transfer fractions
def transfer_A_to_B : ℚ := 1/3
def transfer_B_to_A : ℚ := 1/2
def transfer_A_to_C : ℚ := 1/4
def transfer_C_to_A : ℚ := 1/3

-- Define the function to calculate the final fraction of cream in Cup A
def final_cream_fraction (
  initial_A initial_B initial_C : ℚ
) (
  transfer_A_to_B transfer_B_to_A transfer_A_to_C transfer_C_to_A : ℚ
) : ℚ :=
  sorry -- The actual calculation would go here

-- Theorem statement
theorem cream_fraction_is_three_tenths :
  final_cream_fraction
    initial_A initial_B initial_C
    transfer_A_to_B transfer_B_to_A transfer_A_to_C transfer_C_to_A
  = 3/10 := by
  sorry

end NUMINAMATH_CALUDE_cream_fraction_is_three_tenths_l2828_282899


namespace NUMINAMATH_CALUDE_existence_of_m_l2828_282874

def z : ℕ → ℚ
  | 0 => 3
  | n + 1 => (2 * (z n)^2 + 3 * (z n) + 6) / (z n + 8)

theorem existence_of_m :
  ∃ m : ℕ, m ∈ Finset.Icc 27 80 ∧
    z m ≤ 2 + 1 / 2^10 ∧
    ∀ k : ℕ, k > 0 ∧ k < 27 → z k > 2 + 1 / 2^10 :=
by sorry

end NUMINAMATH_CALUDE_existence_of_m_l2828_282874


namespace NUMINAMATH_CALUDE_max_value_complex_fraction_l2828_282826

theorem max_value_complex_fraction (z : ℂ) (h : Complex.abs z = 1) :
  ∃ (max_val : ℝ), max_val = (2 * Real.sqrt 5) / 3 ∧
  ∀ (w : ℂ), Complex.abs w = 1 →
    Complex.abs ((w + Complex.I) / (w + 2)) ≤ max_val :=
by sorry

end NUMINAMATH_CALUDE_max_value_complex_fraction_l2828_282826


namespace NUMINAMATH_CALUDE_decreasing_f_iff_a_in_range_l2828_282802

noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  if x < 0 then a^x else (a - 3) * x + 4 * a

def is_decreasing (f : ℝ → ℝ) : Prop :=
  ∀ x y, x < y → f x > f y

theorem decreasing_f_iff_a_in_range (a : ℝ) :
  is_decreasing (f a) ↔ 0 < a ∧ a ≤ 1/4 :=
sorry

end NUMINAMATH_CALUDE_decreasing_f_iff_a_in_range_l2828_282802


namespace NUMINAMATH_CALUDE_banana_change_l2828_282851

/-- Calculates the change received when buying bananas -/
theorem banana_change (num_bananas : ℕ) (cost_per_banana : ℚ) (amount_paid : ℚ) :
  num_bananas = 5 →
  cost_per_banana = 30 / 100 →
  amount_paid = 10 →
  amount_paid - (num_bananas : ℚ) * cost_per_banana = 17 / 2 :=
by sorry

end NUMINAMATH_CALUDE_banana_change_l2828_282851


namespace NUMINAMATH_CALUDE_original_number_proof_l2828_282847

theorem original_number_proof :
  ∀ x : ℕ,
  x < 10 →
  (x + 10) * ((x + 10) / x) = 72 →
  x = 2 :=
by
  sorry

end NUMINAMATH_CALUDE_original_number_proof_l2828_282847


namespace NUMINAMATH_CALUDE_savings_ratio_l2828_282881

/-- Represents the number of cans collected from different sources -/
structure CanCollection where
  home : ℕ
  grandparents : ℕ
  neighbor : ℕ
  office : ℕ

/-- Calculates the total number of cans collected -/
def total_cans (c : CanCollection) : ℕ :=
  c.home + c.grandparents + c.neighbor + c.office

/-- Represents the problem setup -/
structure RecyclingProblem where
  collection : CanCollection
  price_per_can : ℚ
  savings_amount : ℚ

/-- Main theorem: The ratio of savings to total amount collected is 1:2 -/
theorem savings_ratio (p : RecyclingProblem)
  (h1 : p.collection.home = 12)
  (h2 : p.collection.grandparents = 3 * p.collection.home)
  (h3 : p.collection.neighbor = 46)
  (h4 : p.collection.office = 250)
  (h5 : p.price_per_can = 1/4)
  (h6 : p.savings_amount = 43) :
  p.savings_amount / (p.price_per_can * total_cans p.collection) = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_savings_ratio_l2828_282881


namespace NUMINAMATH_CALUDE_log_product_change_base_l2828_282841

-- Define the logarithm function
noncomputable def log (base : ℝ) (x : ℝ) : ℝ := Real.log x / Real.log base

-- State the theorem
theorem log_product_change_base 
  (a b c : ℝ) (m n : ℝ) 
  (h1 : log b a = m) 
  (h2 : log c b = n) 
  (h3 : a > 0) (h4 : b > 1) (h5 : c > 1) :
  log (b * c) (a * b) = n * (m + 1) / (n + 1) := by
sorry

end NUMINAMATH_CALUDE_log_product_change_base_l2828_282841


namespace NUMINAMATH_CALUDE_f_symmetry_iff_a_eq_one_l2828_282825

/-- The function f(x) defined as -|x-a| -/
def f (a : ℝ) (x : ℝ) : ℝ := -|x - a|

/-- Theorem stating that f(1+x) = f(1-x) for all x is equivalent to a = 1 -/
theorem f_symmetry_iff_a_eq_one (a : ℝ) :
  (∀ x, f a (1 + x) = f a (1 - x)) ↔ a = 1 := by
  sorry

end NUMINAMATH_CALUDE_f_symmetry_iff_a_eq_one_l2828_282825


namespace NUMINAMATH_CALUDE_squirrel_acorns_count_l2828_282803

/-- Represents the number of acorns hidden per hole by each animal -/
structure AcornsPerHole where
  chipmunk : ℕ
  squirrel : ℕ
  rabbit : ℕ

/-- Represents the number of holes dug by each animal -/
structure HolesCounts where
  chipmunk : ℕ
  squirrel : ℕ
  rabbit : ℕ

/-- The main theorem stating the number of acorns hidden by the squirrel -/
theorem squirrel_acorns_count 
  (aph : AcornsPerHole) 
  (hc : HolesCounts) 
  (h1 : aph.chipmunk = 4) 
  (h2 : aph.squirrel = 5) 
  (h3 : aph.rabbit = 2) 
  (h4 : aph.chipmunk * hc.chipmunk = aph.squirrel * hc.squirrel) 
  (h5 : hc.squirrel = hc.chipmunk - 5) 
  (h6 : aph.rabbit * hc.rabbit = aph.squirrel * hc.squirrel) 
  (h7 : hc.rabbit = hc.squirrel + 10) : 
  aph.squirrel * hc.squirrel = 100 := by
  sorry

end NUMINAMATH_CALUDE_squirrel_acorns_count_l2828_282803


namespace NUMINAMATH_CALUDE_cube_volume_from_surface_area_l2828_282883

/-- Given a cube with surface area 150 square inches, its volume is 125 cubic inches -/
theorem cube_volume_from_surface_area :
  ∀ (edge_length : ℝ),
  (6 * edge_length^2 = 150) →
  edge_length^3 = 125 :=
by
  sorry

end NUMINAMATH_CALUDE_cube_volume_from_surface_area_l2828_282883


namespace NUMINAMATH_CALUDE_square_side_length_l2828_282870

/-- Given a circle with area 100 and a square whose perimeter equals the circle's area,
    the length of one side of the square is 25. -/
theorem square_side_length (circle_area : ℝ) (square_perimeter : ℝ) :
  circle_area = 100 →
  square_perimeter = circle_area →
  square_perimeter = 4 * 25 :=
by
  sorry

#check square_side_length

end NUMINAMATH_CALUDE_square_side_length_l2828_282870

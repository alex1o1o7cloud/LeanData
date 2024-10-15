import Mathlib

namespace NUMINAMATH_CALUDE_no_valid_a_l4075_407554

theorem no_valid_a : ¬∃ a : ℝ, ∀ x : ℝ, x^2 + a*x + a - 2 > 0 := by
  sorry

end NUMINAMATH_CALUDE_no_valid_a_l4075_407554


namespace NUMINAMATH_CALUDE_product_of_sqrt_diff_eq_one_l4075_407527

theorem product_of_sqrt_diff_eq_one :
  let A := Real.sqrt 3008 + Real.sqrt 3009
  let B := -Real.sqrt 3008 - Real.sqrt 3009
  let C := Real.sqrt 3008 - Real.sqrt 3009
  let D := Real.sqrt 3009 - Real.sqrt 3008
  A * B * C * D = 1 := by
sorry

end NUMINAMATH_CALUDE_product_of_sqrt_diff_eq_one_l4075_407527


namespace NUMINAMATH_CALUDE_quadratic_rational_solutions_l4075_407514

/-- A function that checks if a quadratic equation with rational coefficients has rational solutions -/
def has_rational_solutions (a b c : ℚ) : Prop :=
  ∃ x : ℚ, a * x^2 + b * x + c = 0

/-- The set of positive integer values of d for which 3x^2 + 7x + d = 0 has rational solutions -/
def D : Set ℕ+ :=
  {d : ℕ+ | has_rational_solutions 3 7 d.val}

theorem quadratic_rational_solutions :
  (∃ d1 d2 : ℕ+, d1 ≠ d2 ∧ D = {d1, d2}) ∧
  (∀ d1 d2 : ℕ+, d1 ∈ D → d2 ∈ D → d1.val * d2.val = 8) :=
sorry

end NUMINAMATH_CALUDE_quadratic_rational_solutions_l4075_407514


namespace NUMINAMATH_CALUDE_cos_four_theta_l4075_407543

theorem cos_four_theta (θ : Real) 
  (h : Real.exp (Real.log 2 * (-2 + 3 * Real.cos θ)) + 1 = Real.exp (Real.log 2 * (1/2 + 2 * Real.cos θ))) : 
  Real.cos (4 * θ) = -1/2 := by
sorry

end NUMINAMATH_CALUDE_cos_four_theta_l4075_407543


namespace NUMINAMATH_CALUDE_greatest_n_value_l4075_407587

theorem greatest_n_value (n : ℤ) (h : 101 * n^2 ≤ 6400) : n ≤ 7 ∧ ∃ m : ℤ, m = 7 ∧ 101 * m^2 ≤ 6400 := by
  sorry

end NUMINAMATH_CALUDE_greatest_n_value_l4075_407587


namespace NUMINAMATH_CALUDE_max_valid_ltrominos_eighteen_is_achievable_l4075_407584

-- Define the colors
inductive Color
  | Red
  | Green
  | Blue

-- Define the grid
def Grid := Fin 4 → Fin 4 → Color

-- Define an L-tromino
structure LTromino where
  x : Fin 4
  y : Fin 4
  orientation : Fin 4

-- Function to check if an L-tromino has one square of each color
def hasOneOfEachColor (g : Grid) (l : LTromino) : Bool := sorry

-- Function to count valid L-trominos in a grid
def countValidLTrominos (g : Grid) : Nat := sorry

-- Theorem statement
theorem max_valid_ltrominos (g : Grid) : 
  countValidLTrominos g ≤ 18 := sorry

-- Theorem stating that 18 is achievable
theorem eighteen_is_achievable : 
  ∃ g : Grid, countValidLTrominos g = 18 := sorry

end NUMINAMATH_CALUDE_max_valid_ltrominos_eighteen_is_achievable_l4075_407584


namespace NUMINAMATH_CALUDE_selling_price_ratio_l4075_407548

theorem selling_price_ratio (c x y : ℝ) (hx : x = 0.80 * c) (hy : y = 1.25 * c) :
  y / x = 25 / 16 := by
  sorry

end NUMINAMATH_CALUDE_selling_price_ratio_l4075_407548


namespace NUMINAMATH_CALUDE_min_triangles_to_cover_l4075_407541

theorem min_triangles_to_cover (small_side : ℝ) (large_side : ℝ) : 
  small_side = 1 →
  large_side = 15 →
  (large_side / small_side) ^ 2 = 225 := by
  sorry

end NUMINAMATH_CALUDE_min_triangles_to_cover_l4075_407541


namespace NUMINAMATH_CALUDE_petrol_consumption_reduction_l4075_407558

theorem petrol_consumption_reduction (P C : ℝ) (h : P > 0) (h' : C > 0) :
  let new_price := 1.25 * P
  let new_consumption := C * (P / new_price)
  (P * C = new_price * new_consumption) → (new_consumption / C = 0.8) :=
by sorry

end NUMINAMATH_CALUDE_petrol_consumption_reduction_l4075_407558


namespace NUMINAMATH_CALUDE_pizza_promotion_savings_l4075_407582

/-- The regular price of a medium pizza in dollars -/
def regular_price : ℝ := 18

/-- The promotional price of a medium pizza in dollars -/
def promo_price : ℝ := 5

/-- The number of pizzas eligible for the promotion -/
def num_pizzas : ℕ := 3

/-- The total savings when buying the promotional pizzas -/
def total_savings : ℝ := num_pizzas * (regular_price - promo_price)

theorem pizza_promotion_savings : total_savings = 39 := by
  sorry

end NUMINAMATH_CALUDE_pizza_promotion_savings_l4075_407582


namespace NUMINAMATH_CALUDE_photograph_perimeter_l4075_407565

theorem photograph_perimeter (w h m : ℝ) 
  (border_1 : (w + 2) * (h + 2) = m)
  (border_3 : (w + 6) * (h + 6) = m + 52) :
  2 * w + 2 * h = 10 := by
  sorry

end NUMINAMATH_CALUDE_photograph_perimeter_l4075_407565


namespace NUMINAMATH_CALUDE_rolling_circle_traces_line_l4075_407568

/-- A circle with radius R -/
structure SmallCircle (R : ℝ) where
  center : ℝ × ℝ
  radius : ℝ
  radius_eq : radius = R

/-- A circle with radius 2R -/
structure LargeCircle (R : ℝ) where
  center : ℝ × ℝ
  radius : ℝ
  radius_eq : radius = 2 * R

/-- A point on the circumference of the small circle -/
def PointOnSmallCircle (R : ℝ) (sc : SmallCircle R) : Type :=
  { p : ℝ × ℝ // (p.1 - sc.center.1)^2 + (p.2 - sc.center.2)^2 = R^2 }

/-- The path traced by a point on the small circle as it rolls inside the large circle -/
def TracedPath (R : ℝ) (sc : SmallCircle R) (lc : LargeCircle R) (p : PointOnSmallCircle R sc) : Set (ℝ × ℝ) :=
  sorry

/-- The statement that the traced path is a straight line -/
def IsStraitLine (path : Set (ℝ × ℝ)) : Prop :=
  sorry

/-- The main theorem -/
theorem rolling_circle_traces_line (R : ℝ) (sc : SmallCircle R) (lc : LargeCircle R) 
  (p : PointOnSmallCircle R sc) : 
  IsStraitLine (TracedPath R sc lc p) :=
sorry

end NUMINAMATH_CALUDE_rolling_circle_traces_line_l4075_407568


namespace NUMINAMATH_CALUDE_f_derivative_at_zero_l4075_407573

noncomputable def f (x : ℝ) : ℝ := Real.exp x * Real.sin x

theorem f_derivative_at_zero : 
  deriv f 0 = 1 := by sorry

end NUMINAMATH_CALUDE_f_derivative_at_zero_l4075_407573


namespace NUMINAMATH_CALUDE_intersection_single_point_l4075_407598

-- Define the sets A and B
def A : Set (ℝ × ℝ) := {p : ℝ × ℝ | p.1^2 + p.2^2 = 4}
def B (r : ℝ) : Set (ℝ × ℝ) := {p : ℝ × ℝ | (p.1 - 3)^2 + (p.2 - 4)^2 = r^2}

-- State the theorem
theorem intersection_single_point (r : ℝ) (h_r : r > 0) 
  (h_intersection : ∃! p, p ∈ A ∩ B r) : 
  r = 3 ∨ r = 7 := by sorry

end NUMINAMATH_CALUDE_intersection_single_point_l4075_407598


namespace NUMINAMATH_CALUDE_light_green_yellow_percentage_l4075_407526

-- Define the variables
def light_green_volume : ℝ := 5
def dark_green_volume : ℝ := 1.66666666667
def dark_green_yellow_percentage : ℝ := 0.4
def mixture_yellow_percentage : ℝ := 0.25

-- Define the theorem
theorem light_green_yellow_percentage :
  ∃ x : ℝ, 
    x * light_green_volume + dark_green_yellow_percentage * dark_green_volume = 
    mixture_yellow_percentage * (light_green_volume + dark_green_volume) ∧ 
    x = 0.2 := by
  sorry

end NUMINAMATH_CALUDE_light_green_yellow_percentage_l4075_407526


namespace NUMINAMATH_CALUDE_wildcats_score_is_36_l4075_407590

/-- The score of the Panthers -/
def panthers_score : ℕ := 17

/-- The difference between the Wildcats' and Panthers' scores -/
def score_difference : ℕ := 19

/-- The score of the Wildcats -/
def wildcats_score : ℕ := panthers_score + score_difference

theorem wildcats_score_is_36 : wildcats_score = 36 := by
  sorry

end NUMINAMATH_CALUDE_wildcats_score_is_36_l4075_407590


namespace NUMINAMATH_CALUDE_remaining_quantities_l4075_407516

theorem remaining_quantities (total : ℕ) (avg_all : ℚ) (subset : ℕ) (avg_subset : ℚ) (avg_remaining : ℚ) :
  total = 5 →
  avg_all = 12 →
  subset = 3 →
  avg_subset = 4 →
  avg_remaining = 24 →
  total - subset = 2 :=
by sorry

end NUMINAMATH_CALUDE_remaining_quantities_l4075_407516


namespace NUMINAMATH_CALUDE_all_digits_are_perfect_cube_units_l4075_407529

-- Define the set of possible units digits of perfect cubes modulo 10
def PerfectCubeUnitsDigits : Set ℕ :=
  {d | ∃ n : ℤ, (n^3 : ℤ) % 10 = d}

-- Theorem statement
theorem all_digits_are_perfect_cube_units : PerfectCubeUnitsDigits = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9} := by
  sorry

end NUMINAMATH_CALUDE_all_digits_are_perfect_cube_units_l4075_407529


namespace NUMINAMATH_CALUDE_equation_equivalence_l4075_407571

theorem equation_equivalence (x y : ℝ) :
  x^2 * (y + y^2) = y^3 + x^4 ↔ y = x ∨ y = -x ∨ y = x^2 := by
  sorry

end NUMINAMATH_CALUDE_equation_equivalence_l4075_407571


namespace NUMINAMATH_CALUDE_pears_left_l4075_407512

/-- 
Given that Jason picked 46 pears, Keith picked 47 pears, and Mike ate 12 pears,
prove that the number of pears left is 81.
-/
theorem pears_left (jason_pears keith_pears : ℕ) (mike_ate : ℕ) 
  (h1 : jason_pears = 46)
  (h2 : keith_pears = 47)
  (h3 : mike_ate = 12) :
  jason_pears + keith_pears - mike_ate = 81 := by
  sorry

end NUMINAMATH_CALUDE_pears_left_l4075_407512


namespace NUMINAMATH_CALUDE_system_solution_l4075_407561

/-- The solution to the system of equations:
     4x - 3y = -2.4
     5x + 6y = 7.5
-/
theorem system_solution :
  ∃ (x y : ℝ), 
    (4 * x - 3 * y = -2.4) ∧
    (5 * x + 6 * y = 7.5) ∧
    (x = 2.7 / 13) ∧
    (y = 1.0769) := by
  sorry

end NUMINAMATH_CALUDE_system_solution_l4075_407561


namespace NUMINAMATH_CALUDE_all_roots_integer_iff_a_eq_50_l4075_407546

/-- The polynomial P(x) = x^3 - 2x^2 - 25x + a -/
def P (a : ℝ) (x : ℝ) : ℝ := x^3 - 2*x^2 - 25*x + a

/-- A function that checks if a real number is an integer -/
def isInteger (x : ℝ) : Prop := ∃ n : ℤ, x = n

/-- The theorem stating that all roots of P(x) are integers iff a = 50 -/
theorem all_roots_integer_iff_a_eq_50 (a : ℝ) :
  (∀ x : ℝ, P a x = 0 → isInteger x) ↔ a = 50 := by sorry

end NUMINAMATH_CALUDE_all_roots_integer_iff_a_eq_50_l4075_407546


namespace NUMINAMATH_CALUDE_nico_reading_proof_l4075_407572

/-- The number of pages Nico read on Monday -/
def pages_monday : ℕ := 39

/-- The number of pages Nico read on Tuesday -/
def pages_tuesday : ℕ := 12

/-- The total number of pages Nico read over three days -/
def total_pages : ℕ := 51

/-- The number of books Nico borrowed -/
def num_books : ℕ := 3

/-- The number of days Nico read -/
def num_days : ℕ := 3

theorem nico_reading_proof :
  pages_monday = total_pages - pages_tuesday ∧
  pages_monday + pages_tuesday ≤ total_pages ∧
  num_books = num_days := by sorry

end NUMINAMATH_CALUDE_nico_reading_proof_l4075_407572


namespace NUMINAMATH_CALUDE_points_collinear_collinear_vectors_l4075_407567

variable {V : Type*} [AddCommGroup V] [Module ℝ V]

/-- Non-zero vectors e₁ and e₂ are not collinear -/
def not_collinear (e₁ e₂ : V) : Prop :=
  e₁ ≠ 0 ∧ e₂ ≠ 0 ∧ ∀ (r : ℝ), e₁ ≠ r • e₂

variable (e₁ e₂ : V) (h : not_collinear e₁ e₂)

/-- Vector AB -/
def AB : V := e₁ + e₂

/-- Vector BC -/
def BC : V := 2 • e₁ + 8 • e₂

/-- Vector CD -/
def CD : V := 3 • (e₁ - e₂)

/-- Three points are collinear if the vector between any two is a scalar multiple of the vector between the other two -/
def collinear (A B D : V) : Prop :=
  ∃ (r : ℝ), B - A = r • (D - B) ∨ D - B = r • (B - A)

theorem points_collinear :
  collinear (0 : V) (AB e₁ e₂) ((AB e₁ e₂) + (BC e₁ e₂) + (CD e₁ e₂)) :=
sorry

theorem collinear_vectors (k : ℝ) :
  (∃ (r : ℝ), k • e₁ + e₂ = r • (e₁ + k • e₂)) ↔ k = 1 ∨ k = -1 :=
sorry

end NUMINAMATH_CALUDE_points_collinear_collinear_vectors_l4075_407567


namespace NUMINAMATH_CALUDE_point_A_moves_to_vertex_3_l4075_407557

/-- Represents a vertex of a cube --/
structure Vertex where
  label : Nat
  onGreenFace : Bool
  onDistantWhiteFace : Bool
  onBottomRightWhiteFace : Bool

/-- Represents the rotation of a cube --/
def rotatedCube : List Vertex → List Vertex := sorry

/-- The initial position of point A --/
def pointA : Vertex := {
  label := 0,
  onGreenFace := true,
  onDistantWhiteFace := true,
  onBottomRightWhiteFace := true
}

/-- Theorem stating that point A moves to vertex 3 after rotation --/
theorem point_A_moves_to_vertex_3 (cube : List Vertex) :
  ∃ v ∈ rotatedCube cube,
    v.label = 3 ∧
    v.onGreenFace = true ∧
    v.onDistantWhiteFace = true ∧
    v.onBottomRightWhiteFace = true :=
  sorry

end NUMINAMATH_CALUDE_point_A_moves_to_vertex_3_l4075_407557


namespace NUMINAMATH_CALUDE_or_not_implies_right_l4075_407522

theorem or_not_implies_right (p q : Prop) : (p ∨ q) → ¬p → q := by
  sorry

end NUMINAMATH_CALUDE_or_not_implies_right_l4075_407522


namespace NUMINAMATH_CALUDE_last_two_digits_of_7_power_last_two_digits_of_7_2017_l4075_407535

theorem last_two_digits_of_7_power (n : ℕ) :
  (7^n) % 100 = (7^(n % 4 + 4)) % 100 :=
sorry

theorem last_two_digits_of_7_2017 :
  (7^2017) % 100 = 49 :=
sorry

end NUMINAMATH_CALUDE_last_two_digits_of_7_power_last_two_digits_of_7_2017_l4075_407535


namespace NUMINAMATH_CALUDE_A_union_complement_B_equals_one_three_l4075_407525

-- Define the universal set U
def U : Set Nat := {1, 2, 3}

-- Define set A
def A : Set Nat := {1}

-- Define set B
def B : Set Nat := {1, 2}

-- Theorem statement
theorem A_union_complement_B_equals_one_three :
  A ∪ (U \ B) = {1, 3} := by sorry

end NUMINAMATH_CALUDE_A_union_complement_B_equals_one_three_l4075_407525


namespace NUMINAMATH_CALUDE_negation_equivalence_l4075_407595

theorem negation_equivalence : 
  (¬(∀ x : ℝ, |x| ≥ 2 → (x ≥ 2 ∨ x ≤ -2))) ↔ 
  (∀ x : ℝ, |x| < 2 → (-2 < x ∧ x < 2)) :=
sorry

end NUMINAMATH_CALUDE_negation_equivalence_l4075_407595


namespace NUMINAMATH_CALUDE_john_took_six_pink_l4075_407564

def initial_pink : ℕ := 26
def initial_green : ℕ := 15
def initial_yellow : ℕ := 24
def carl_took : ℕ := 4
def total_remaining : ℕ := 43

def john_took_pink (p : ℕ) : Prop :=
  initial_pink - carl_took - p +
  initial_green - 2 * p +
  initial_yellow = total_remaining

theorem john_took_six_pink : john_took_pink 6 := by sorry

end NUMINAMATH_CALUDE_john_took_six_pink_l4075_407564


namespace NUMINAMATH_CALUDE_unit_circle_point_coordinate_l4075_407510

/-- Theorem: For a point P(x₀, y₀) on the unit circle in the xy-plane, where ∠xOP = α, 
α ∈ (π/4, 3π/4), and cos(α + π/4) = -12/13, the value of x₀ is equal to -7√2/26. -/
theorem unit_circle_point_coordinate (x₀ y₀ α : Real) : 
  x₀^2 + y₀^2 = 1 → -- Point P lies on the unit circle
  x₀ = Real.cos α → -- Definition of cosine
  y₀ = Real.sin α → -- Definition of sine
  π/4 < α → α < 3*π/4 → -- α ∈ (π/4, 3π/4)
  Real.cos (α + π/4) = -12/13 → -- Given condition
  x₀ = -7 * Real.sqrt 2 / 26 := by
sorry

end NUMINAMATH_CALUDE_unit_circle_point_coordinate_l4075_407510


namespace NUMINAMATH_CALUDE_initial_nickels_correct_l4075_407507

/-- The number of nickels Mike had initially -/
def initial_nickels : ℕ := 87

/-- The number of nickels Mike's dad borrowed -/
def borrowed_nickels : ℕ := 75

/-- The number of nickels Mike was left with -/
def remaining_nickels : ℕ := 12

/-- Theorem stating that the initial number of nickels is correct -/
theorem initial_nickels_correct : initial_nickels = borrowed_nickels + remaining_nickels := by
  sorry

end NUMINAMATH_CALUDE_initial_nickels_correct_l4075_407507


namespace NUMINAMATH_CALUDE_shaded_area_is_six_l4075_407563

/-- Represents a quadrilateral divided into four smaller quadrilaterals -/
structure DividedQuadrilateral where
  /-- Areas of the four smaller quadrilaterals -/
  area1 : ℝ
  area2 : ℝ
  area3 : ℝ
  area4 : ℝ
  /-- The sum of areas of two opposite quadrilaterals is 28 -/
  sum_opposite : area1 + area3 = 28
  /-- One of the quadrilaterals has an area of 8 -/
  known_area : area2 = 8

/-- The theorem to be proved -/
theorem shaded_area_is_six (q : DividedQuadrilateral) : q.area4 = 6 := by
  sorry

end NUMINAMATH_CALUDE_shaded_area_is_six_l4075_407563


namespace NUMINAMATH_CALUDE_polygon_sides_l4075_407513

theorem polygon_sides (n : ℕ) : n > 2 →
  ∃ (x : ℝ), x > 0 ∧ x < 180 ∧ (n - 2) * 180 + x = 1350 →
  n = 9 := by
sorry

end NUMINAMATH_CALUDE_polygon_sides_l4075_407513


namespace NUMINAMATH_CALUDE_soccer_ball_max_height_l4075_407581

/-- The height function of a soccer ball kicked vertically -/
def h (t : ℝ) : ℝ := -20 * t^2 + 40 * t + 20

/-- The maximum height achieved by the soccer ball -/
def max_height : ℝ := 40

/-- Theorem stating that the maximum height of the soccer ball is 40 feet -/
theorem soccer_ball_max_height :
  ∀ t : ℝ, h t ≤ max_height :=
by sorry

end NUMINAMATH_CALUDE_soccer_ball_max_height_l4075_407581


namespace NUMINAMATH_CALUDE_car_repair_cost_johns_car_repair_cost_l4075_407586

/-- Calculates the total cost of car repairs given labor rate, hours worked, and part cost -/
theorem car_repair_cost (labor_rate : ℕ) (hours : ℕ) (part_cost : ℕ) : 
  labor_rate * hours + part_cost = 2400 :=
by
  sorry

/-- Proves the specific case of John's car repair cost -/
theorem johns_car_repair_cost : 
  75 * 16 + 1200 = 2400 :=
by
  sorry

end NUMINAMATH_CALUDE_car_repair_cost_johns_car_repair_cost_l4075_407586


namespace NUMINAMATH_CALUDE_two_digit_gcd_theorem_l4075_407596

/-- Represents a two-digit decimal number as a pair of natural numbers (a, b) -/
def TwoDigitNumber := { p : ℕ × ℕ // p.1 ≤ 9 ∧ p.2 ≤ 9 ∧ p.1 ≠ 0 }

/-- Converts a two-digit number (a, b) to its decimal representation ab -/
def toDecimal (n : TwoDigitNumber) : ℕ :=
  10 * n.val.1 + n.val.2

/-- Converts a two-digit number (a, b) to its reversed decimal representation ba -/
def toReversedDecimal (n : TwoDigitNumber) : ℕ :=
  10 * n.val.2 + n.val.1

/-- Checks if a two-digit number satisfies the GCD condition -/
def satisfiesGCDCondition (n : TwoDigitNumber) : Prop :=
  Nat.gcd (toDecimal n) (toReversedDecimal n) = n.val.1^2 - n.val.2^2

theorem two_digit_gcd_theorem :
  ∃ (n1 n2 : TwoDigitNumber),
    satisfiesGCDCondition n1 ∧
    satisfiesGCDCondition n2 ∧
    toDecimal n1 = 21 ∧
    toDecimal n2 = 54 ∧
    (∀ (n : TwoDigitNumber), satisfiesGCDCondition n → (toDecimal n = 21 ∨ toDecimal n = 54)) :=
  sorry

end NUMINAMATH_CALUDE_two_digit_gcd_theorem_l4075_407596


namespace NUMINAMATH_CALUDE_other_root_of_quadratic_l4075_407555

theorem other_root_of_quadratic (a : ℝ) : 
  (1^2 + a*1 + 2 = 0) → ∃ b : ℝ, b ≠ 1 ∧ b^2 + a*b + 2 = 0 ∧ b = 2 :=
sorry

end NUMINAMATH_CALUDE_other_root_of_quadratic_l4075_407555


namespace NUMINAMATH_CALUDE_spending_ratio_l4075_407585

def monthly_allowance : ℚ := 12

def spending_scenario (first_week_spending : ℚ) : Prop :=
  let remaining_after_first_week := monthly_allowance - first_week_spending
  let second_week_spending := (1 / 4) * remaining_after_first_week
  monthly_allowance - first_week_spending - second_week_spending = 6

theorem spending_ratio :
  ∃ (first_week_spending : ℚ),
    spending_scenario first_week_spending ∧
    first_week_spending / monthly_allowance = 1 / 3 := by
  sorry

end NUMINAMATH_CALUDE_spending_ratio_l4075_407585


namespace NUMINAMATH_CALUDE_coin_bag_total_l4075_407533

theorem coin_bag_total (p : ℕ) : ∃ (p : ℕ), 
  (0.01 * p + 0.05 * (3 * p) + 0.50 * (12 * p) : ℚ) = 616 := by
  sorry

end NUMINAMATH_CALUDE_coin_bag_total_l4075_407533


namespace NUMINAMATH_CALUDE_pentagon_obtuse_angles_dodecagon_diagonals_four_sided_angle_sum_equality_l4075_407509

/-- A polygon with n sides -/
structure Polygon (n : ℕ) where
  -- Add necessary fields here

/-- The number of obtuse angles in a polygon -/
def numObtuseAngles (p : Polygon n) : ℕ := sorry

/-- The number of diagonals in a polygon -/
def numDiagonals (p : Polygon n) : ℕ := sorry

/-- The sum of interior angles of a polygon -/
def sumInteriorAngles (p : Polygon n) : ℝ := sorry

/-- The sum of exterior angles of a polygon -/
def sumExteriorAngles (p : Polygon n) : ℝ := sorry

theorem pentagon_obtuse_angles :
  ∀ p : Polygon 5, numObtuseAngles p ≥ 2 := by sorry

theorem dodecagon_diagonals :
  ∀ p : Polygon 12, numDiagonals p = 54 := by sorry

theorem four_sided_angle_sum_equality :
  ∀ n : ℕ, ∀ p : Polygon n,
    (sumInteriorAngles p = sumExteriorAngles p) ↔ n = 4 := by sorry

end NUMINAMATH_CALUDE_pentagon_obtuse_angles_dodecagon_diagonals_four_sided_angle_sum_equality_l4075_407509


namespace NUMINAMATH_CALUDE_inequality_system_solution_l4075_407508

theorem inequality_system_solution :
  let S := {x : ℝ | 3 < x ∧ x ≤ 4}
  S = {x : ℝ | 3*x + 4 ≥ 4*x ∧ 2*(x - 1) + x > 7} := by sorry

end NUMINAMATH_CALUDE_inequality_system_solution_l4075_407508


namespace NUMINAMATH_CALUDE_purely_imaginary_implies_m_eq_neg_three_l4075_407569

/-- A complex number z is purely imaginary if its real part is zero and its imaginary part is non-zero. -/
def is_purely_imaginary (z : ℂ) : Prop :=
  z.re = 0 ∧ z.im ≠ 0

/-- The complex number z defined in terms of a real parameter m. -/
def z (m : ℝ) : ℂ :=
  Complex.mk (m^2 + m - 6) (m - 2)

/-- If z(m) is purely imaginary, then m = -3. -/
theorem purely_imaginary_implies_m_eq_neg_three :
  ∀ m : ℝ, is_purely_imaginary (z m) → m = -3 := by
  sorry

end NUMINAMATH_CALUDE_purely_imaginary_implies_m_eq_neg_three_l4075_407569


namespace NUMINAMATH_CALUDE_collinear_points_x_value_l4075_407520

/-- Given three points in a 2D plane, checks if they are collinear --/
def collinear (x₁ y₁ x₂ y₂ x₃ y₃ : ℝ) : Prop :=
  (y₂ - y₁) * (x₃ - x₁) = (y₃ - y₁) * (x₂ - x₁)

/-- Theorem: If points A(1, 1), B(-4, 5), and C(x, 13) are collinear, then x = -14 --/
theorem collinear_points_x_value :
  ∀ x : ℝ, collinear 1 1 (-4) 5 x 13 → x = -14 := by
  sorry

end NUMINAMATH_CALUDE_collinear_points_x_value_l4075_407520


namespace NUMINAMATH_CALUDE_increasing_function_property_l4075_407515

def increasing_function (f : ℝ → ℝ) : Prop :=
  ∀ x y, x < y → f x < f y

theorem increasing_function_property (f : ℝ → ℝ) (h : increasing_function f) (a b : ℝ) :
  (a + b ≥ 0) ↔ (f a + f b ≥ f (-a) + f (-b)) :=
by sorry

end NUMINAMATH_CALUDE_increasing_function_property_l4075_407515


namespace NUMINAMATH_CALUDE_thanksgiving_to_christmas_l4075_407524

/-- Represents days of the week -/
inductive DayOfWeek
  | Sunday
  | Monday
  | Tuesday
  | Wednesday
  | Thursday
  | Friday
  | Saturday

/-- Represents months of the year -/
inductive Month
  | November
  | December

/-- Represents a date in a month -/
structure Date where
  month : Month
  day : Nat

/-- Function to get the next day of the week -/
def nextDay (d : DayOfWeek) : DayOfWeek :=
  match d with
  | DayOfWeek.Sunday => DayOfWeek.Monday
  | DayOfWeek.Monday => DayOfWeek.Tuesday
  | DayOfWeek.Tuesday => DayOfWeek.Wednesday
  | DayOfWeek.Wednesday => DayOfWeek.Thursday
  | DayOfWeek.Thursday => DayOfWeek.Friday
  | DayOfWeek.Friday => DayOfWeek.Saturday
  | DayOfWeek.Saturday => DayOfWeek.Sunday

/-- Function to get the day of the week after n days -/
def dayAfter (d : DayOfWeek) (n : Nat) : DayOfWeek :=
  match n with
  | 0 => d
  | n + 1 => dayAfter (nextDay d) n

/-- Theorem: If Thanksgiving is on Thursday, November 28, then December 25 falls on a Wednesday -/
theorem thanksgiving_to_christmas 
  (thanksgiving : Date)
  (thanksgiving_day : DayOfWeek)
  (h1 : thanksgiving.month = Month.November)
  (h2 : thanksgiving.day = 28)
  (h3 : thanksgiving_day = DayOfWeek.Thursday) :
  dayAfter thanksgiving_day 27 = DayOfWeek.Wednesday :=
by
  sorry

end NUMINAMATH_CALUDE_thanksgiving_to_christmas_l4075_407524


namespace NUMINAMATH_CALUDE_integer_roots_of_polynomial_l4075_407576

def polynomial (b₂ b₁ : ℤ) (x : ℤ) : ℤ := x^3 + b₂*x^2 + b₁*x + 18

def divisors_of_18 : Set ℤ := {-18, -9, -6, -3, -2, -1, 1, 2, 3, 6, 9, 18}

theorem integer_roots_of_polynomial (b₂ b₁ : ℤ) :
  {x : ℤ | polynomial b₂ b₁ x = 0} = divisors_of_18 :=
sorry

end NUMINAMATH_CALUDE_integer_roots_of_polynomial_l4075_407576


namespace NUMINAMATH_CALUDE_fraction_equality_l4075_407599

theorem fraction_equality (x y z : ℝ) 
  (hx : x ≠ 1) (hy : y ≠ 1) (hxy : x ≠ y) 
  (h : (y * z - x^2) / (1 - x) = (x * z - y^2) / (1 - y)) : 
  (y * z - x^2) / (1 - x) = x + y + z ∧ (x * z - y^2) / (1 - y) = x + y + z := by
  sorry

end NUMINAMATH_CALUDE_fraction_equality_l4075_407599


namespace NUMINAMATH_CALUDE_min_value_of_reciprocal_sum_l4075_407589

def is_arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem min_value_of_reciprocal_sum (a : ℕ → ℝ) 
  (h_pos : ∀ n, a n > 0)
  (h_arith : is_arithmetic_sequence a)
  (h_sum : a 1 + a 2014 = 2) :
  (∀ x y : ℝ, x > 0 → y > 0 → 1/x + 1/y ≥ 2) ∧ 
  (∃ x y : ℝ, x > 0 ∧ y > 0 ∧ 1/x + 1/y = 2) :=
sorry

end NUMINAMATH_CALUDE_min_value_of_reciprocal_sum_l4075_407589


namespace NUMINAMATH_CALUDE_sufficient_not_necessary_condition_l4075_407521

theorem sufficient_not_necessary_condition (x : ℝ) :
  (∀ x, (x + 1) * (x - 3) < 0 → x < 3) ∧
  (∃ x, x < 3 ∧ (x + 1) * (x - 3) ≥ 0) :=
sorry

end NUMINAMATH_CALUDE_sufficient_not_necessary_condition_l4075_407521


namespace NUMINAMATH_CALUDE_yoongi_multiplication_l4075_407500

theorem yoongi_multiplication (x : ℝ) : 8 * x = 64 → x = 8 := by
  sorry

end NUMINAMATH_CALUDE_yoongi_multiplication_l4075_407500


namespace NUMINAMATH_CALUDE_marble_problem_l4075_407511

theorem marble_problem (A V : ℤ) (x : ℤ) : 
  (A + x = V - x) ∧ (V + 2*x = A - 2*x + 30) → x = 5 :=
by
  sorry

end NUMINAMATH_CALUDE_marble_problem_l4075_407511


namespace NUMINAMATH_CALUDE_isosceles_triangle_leg_length_l4075_407544

/-- Represents an isosceles triangle with given properties -/
structure IsoscelesTriangle where
  perimeter : ℝ
  side_ratio : ℝ
  leg_length : ℝ
  h_perimeter_positive : perimeter > 0
  h_ratio_positive : side_ratio > 0
  h_leg_length_positive : leg_length > 0
  h_perimeter_eq : perimeter = (1 + 2 * side_ratio) * leg_length / side_ratio

/-- Theorem stating the possible leg lengths of the isosceles triangle -/
theorem isosceles_triangle_leg_length 
  (triangle : IsoscelesTriangle) 
  (h_perimeter : triangle.perimeter = 70) 
  (h_ratio : triangle.side_ratio = 3) : 
  triangle.leg_length = 14 ∨ triangle.leg_length = 30 := by
  sorry

#check isosceles_triangle_leg_length

end NUMINAMATH_CALUDE_isosceles_triangle_leg_length_l4075_407544


namespace NUMINAMATH_CALUDE_max_popsicles_with_10_dollars_l4075_407538

/-- Represents the number of popsicles that can be bought with a given amount of money -/
def max_popsicles (single_cost : ℕ) (box4_cost : ℕ) (box6_cost : ℕ) (budget : ℕ) : ℕ :=
  sorry

/-- Theorem stating that the maximum number of popsicles that can be bought with $10 is 14 -/
theorem max_popsicles_with_10_dollars :
  max_popsicles 1 3 4 10 = 14 :=
sorry

end NUMINAMATH_CALUDE_max_popsicles_with_10_dollars_l4075_407538


namespace NUMINAMATH_CALUDE_box_weights_sum_l4075_407559

/-- Given four boxes with weights a, b, c, and d, where:
    - The weight of box d is 60 pounds
    - The combined weight of (a,b) is 132 pounds
    - The combined weight of (a,c) is 136 pounds
    - The combined weight of (b,c) is 138 pounds
    Prove that the sum of weights a, b, and c is 203 pounds. -/
theorem box_weights_sum (a b c d : ℝ) 
    (hd : d = 60)
    (hab : a + b = 132)
    (hac : a + c = 136)
    (hbc : b + c = 138) :
    a + b + c = 203 := by
  sorry

end NUMINAMATH_CALUDE_box_weights_sum_l4075_407559


namespace NUMINAMATH_CALUDE_max_segment_sum_l4075_407547

/-- A rhombus constructed from two equal equilateral triangles, divided into 2n^2 smaller triangles --/
structure Rhombus (n : ℕ) where
  triangles : Fin (2 * n^2) → ℕ
  triangle_values : ∀ i, 1 ≤ triangles i ∧ triangles i ≤ 2 * n^2
  distinct_values : ∀ i j, i ≠ j → triangles i ≠ triangles j

/-- The sum of positive differences on common segments of the rhombus --/
def segmentSum (n : ℕ) (r : Rhombus n) : ℕ :=
  sorry

/-- Theorem: The maximum sum of positive differences on common segments is 3n^4 - 4n^2 + 4n - 2 --/
theorem max_segment_sum (n : ℕ) : 
  (∀ r : Rhombus n, segmentSum n r ≤ 3 * n^4 - 4 * n^2 + 4 * n - 2) ∧
  (∃ r : Rhombus n, segmentSum n r = 3 * n^4 - 4 * n^2 + 4 * n - 2) :=
sorry

end NUMINAMATH_CALUDE_max_segment_sum_l4075_407547


namespace NUMINAMATH_CALUDE_common_difference_proof_l4075_407519

def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem common_difference_proof (a : ℕ → ℝ) (h : arithmetic_sequence a) 
  (h2 : a 2 = 14) (h5 : a 5 = 5) : 
  ∃ d : ℝ, (∀ n : ℕ, a (n + 1) = a n + d) ∧ d = -3 := by
sorry

end NUMINAMATH_CALUDE_common_difference_proof_l4075_407519


namespace NUMINAMATH_CALUDE_arithmetic_geometric_sequence_relation_l4075_407545

theorem arithmetic_geometric_sequence_relation (a : ℕ → ℤ) (b : ℕ → ℝ) (d k m : ℕ) (q : ℝ) :
  (∀ n, a (n + 1) - a n = d) →
  (a k = k^2 + 2) →
  (a (2*k) = (k + 2)^2) →
  (k > 0) →
  (a 1 > 1) →
  (∀ n, b n = q^(n-1)) →
  (q > 0) →
  (∃ m : ℕ, m > 0 ∧ (3 * 2^2) / (3 * m^2) = 1 + q + q^2) →
  q = (Real.sqrt 13 - 1) / 2 :=
by sorry

end NUMINAMATH_CALUDE_arithmetic_geometric_sequence_relation_l4075_407545


namespace NUMINAMATH_CALUDE_business_partnership_gains_l4075_407588

/-- Represents the investment and gain of a partner in the business. -/
structure Partner where
  investment : ℕ
  time : ℕ
  gain : ℕ

/-- Represents the business partnership with four partners. -/
def BusinessPartnership (nandan gopal vishal krishan : Partner) : Prop :=
  -- Investment ratios
  krishan.investment = 6 * nandan.investment ∧
  gopal.investment = 3 * nandan.investment ∧
  vishal.investment = 2 * nandan.investment ∧
  -- Time ratios
  krishan.time = 2 * nandan.time ∧
  gopal.time = 3 * nandan.time ∧
  vishal.time = nandan.time ∧
  -- Nandan's gain
  nandan.gain = 6000 ∧
  -- Gain proportionality
  krishan.gain * nandan.investment * nandan.time = nandan.gain * krishan.investment * krishan.time ∧
  gopal.gain * nandan.investment * nandan.time = nandan.gain * gopal.investment * gopal.time ∧
  vishal.gain * nandan.investment * nandan.time = nandan.gain * vishal.investment * vishal.time

/-- The theorem to be proved -/
theorem business_partnership_gains 
  (nandan gopal vishal krishan : Partner) 
  (h : BusinessPartnership nandan gopal vishal krishan) : 
  krishan.gain = 72000 ∧ 
  gopal.gain = 54000 ∧ 
  vishal.gain = 12000 ∧ 
  nandan.gain + gopal.gain + vishal.gain + krishan.gain = 144000 := by
  sorry

end NUMINAMATH_CALUDE_business_partnership_gains_l4075_407588


namespace NUMINAMATH_CALUDE_parallel_perpendicular_plane_l4075_407593

/-- Two lines are parallel -/
def parallel (m n : Line) : Prop := sorry

/-- A line is perpendicular to a plane -/
def perpendicular (l : Line) (α : Plane) : Prop := sorry

/-- Main theorem: If two lines are parallel and one is perpendicular to a plane, 
    then the other is also perpendicular to that plane -/
theorem parallel_perpendicular_plane (m n : Line) (α : Plane) :
  parallel m n → perpendicular m α → perpendicular n α := by sorry

end NUMINAMATH_CALUDE_parallel_perpendicular_plane_l4075_407593


namespace NUMINAMATH_CALUDE_sunflower_seeds_weight_l4075_407536

/-- The weight of a bag of sunflower seeds in grams -/
def bag_weight : ℝ := 250

/-- The number of bags -/
def num_bags : ℕ := 8

/-- Conversion factor from grams to kilograms -/
def grams_to_kg : ℝ := 1000

theorem sunflower_seeds_weight :
  (bag_weight * num_bags) / grams_to_kg = 2 := by
  sorry

end NUMINAMATH_CALUDE_sunflower_seeds_weight_l4075_407536


namespace NUMINAMATH_CALUDE_milk_sold_in_fl_oz_l4075_407578

def monday_morning_milk : ℕ := 150 * 250 + 40 * 300 + 50 * 350
def monday_evening_milk : ℕ := 50 * 400 + 25 * 500 + 25 * 450
def tuesday_morning_milk : ℕ := 24 * 300 + 18 * 350 + 18 * 400
def tuesday_evening_milk : ℕ := 50 * 450 + 70 * 500 + 80 * 550

def total_milk_bought : ℕ := monday_morning_milk + monday_evening_milk + tuesday_morning_milk + tuesday_evening_milk
def remaining_milk : ℕ := 84000
def ml_per_fl_oz : ℕ := 30

theorem milk_sold_in_fl_oz :
  (total_milk_bought - remaining_milk) / ml_per_fl_oz = 4215 := by sorry

end NUMINAMATH_CALUDE_milk_sold_in_fl_oz_l4075_407578


namespace NUMINAMATH_CALUDE_cube_surface_area_equal_volume_l4075_407560

/-- The surface area of a cube with the same volume as a rectangular prism -/
theorem cube_surface_area_equal_volume (a b c : ℝ) (ha : a = 5) (hb : b = 7) (hc : c = 10) :
  6 * ((a * b * c) ^ (1/3 : ℝ))^2 = 6 * (350 ^ (1/3 : ℝ))^2 := by
  sorry

end NUMINAMATH_CALUDE_cube_surface_area_equal_volume_l4075_407560


namespace NUMINAMATH_CALUDE_calculation_proof_l4075_407551

theorem calculation_proof : (1/2)⁻¹ - 3 * Real.tan (30 * π / 180) + (1 - Real.pi)^0 + Real.sqrt 12 = Real.sqrt 3 + 3 := by
  sorry

end NUMINAMATH_CALUDE_calculation_proof_l4075_407551


namespace NUMINAMATH_CALUDE_gold_alloy_composition_l4075_407579

/-- Proves that adding 12 ounces of pure gold to an alloy weighing 48 ounces
    that is 25% gold will result in an alloy that is 40% gold. -/
theorem gold_alloy_composition (initial_weight : ℝ) (initial_gold_percentage : ℝ) 
    (final_gold_percentage : ℝ) (added_gold : ℝ) : 
  initial_weight = 48 →
  initial_gold_percentage = 0.25 →
  final_gold_percentage = 0.40 →
  added_gold = 12 →
  (initial_weight * initial_gold_percentage + added_gold) / (initial_weight + added_gold) = final_gold_percentage :=
by
  sorry

end NUMINAMATH_CALUDE_gold_alloy_composition_l4075_407579


namespace NUMINAMATH_CALUDE_triangle_altitude_segment_length_l4075_407542

theorem triangle_altitude_segment_length 
  (a b c h d : ℝ) 
  (triangle_sides : a = 30 ∧ b = 70 ∧ c = 80) 
  (altitude_condition : h^2 = b^2 - d^2) 
  (segment_condition : a^2 = h^2 + (c - d)^2) : 
  d = 65 := by sorry

end NUMINAMATH_CALUDE_triangle_altitude_segment_length_l4075_407542


namespace NUMINAMATH_CALUDE_dolls_in_big_box_l4075_407566

/-- Given information about big and small boxes containing dolls, 
    prove that each big box contains 7 dolls. -/
theorem dolls_in_big_box 
  (num_big_boxes : ℕ) 
  (num_small_boxes : ℕ) 
  (dolls_per_small_box : ℕ) 
  (total_dolls : ℕ) 
  (h1 : num_big_boxes = 5)
  (h2 : num_small_boxes = 9)
  (h3 : dolls_per_small_box = 4)
  (h4 : total_dolls = 71) :
  ∃ (dolls_per_big_box : ℕ), 
    dolls_per_big_box * num_big_boxes + 
    dolls_per_small_box * num_small_boxes = total_dolls ∧ 
    dolls_per_big_box = 7 :=
by sorry

end NUMINAMATH_CALUDE_dolls_in_big_box_l4075_407566


namespace NUMINAMATH_CALUDE_exactly_one_zero_in_interval_l4075_407553

def f (a : ℝ) (x : ℝ) : ℝ := x^2 - a*x + 1

theorem exactly_one_zero_in_interval (a : ℝ) (h : a > 3) :
  ∃! x, x ∈ (Set.Ioo 0 2) ∧ f a x = 0 := by
sorry

end NUMINAMATH_CALUDE_exactly_one_zero_in_interval_l4075_407553


namespace NUMINAMATH_CALUDE_tan_neg_five_pi_sixths_l4075_407503

theorem tan_neg_five_pi_sixths : Real.tan (-5 * π / 6) = 1 / Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_tan_neg_five_pi_sixths_l4075_407503


namespace NUMINAMATH_CALUDE_star_three_four_l4075_407504

/-- Custom binary operation ※ -/
def star (a b : ℝ) : ℝ := 2 * a + b

/-- Theorem stating that 3※4 = 10 -/
theorem star_three_four : star 3 4 = 10 := by
  sorry

end NUMINAMATH_CALUDE_star_three_four_l4075_407504


namespace NUMINAMATH_CALUDE_sams_work_days_l4075_407528

theorem sams_work_days (total_days : ℕ) (daily_wage : ℤ) (daily_loss : ℤ) (total_earnings : ℤ) :
  total_days = 20 ∧ daily_wage = 60 ∧ daily_loss = 30 ∧ total_earnings = 660 →
  ∃ (days_not_worked : ℕ),
    days_not_worked = 6 ∧
    days_not_worked ≤ total_days ∧
    (total_days - days_not_worked) * daily_wage - days_not_worked * daily_loss = total_earnings :=
by sorry

end NUMINAMATH_CALUDE_sams_work_days_l4075_407528


namespace NUMINAMATH_CALUDE_cube_root_of_negative_27_l4075_407577

theorem cube_root_of_negative_27 : ∃ x : ℝ, x^3 = -27 ∧ x = -3 := by sorry

end NUMINAMATH_CALUDE_cube_root_of_negative_27_l4075_407577


namespace NUMINAMATH_CALUDE_solution_equation1_solution_equation2_l4075_407574

-- Define the equations
def equation1 (x : ℝ) : Prop := 6 * x - 7 = 4 * x - 5
def equation2 (x : ℝ) : Prop := 4 / 3 - 8 * x = 3 - 11 / 2 * x

-- Theorem for the first equation
theorem solution_equation1 : ∃ x : ℝ, equation1 x ∧ x = 1 := by sorry

-- Theorem for the second equation
theorem solution_equation2 : ∃ x : ℝ, equation2 x ∧ x = -2/3 := by sorry

end NUMINAMATH_CALUDE_solution_equation1_solution_equation2_l4075_407574


namespace NUMINAMATH_CALUDE_polynomial_expansion_l4075_407550

/-- Proves the expansion of (3z^3 + 4z^2 - 2z + 1)(2z^2 - 3z + 5) -/
theorem polynomial_expansion (z : ℝ) :
  (3 * z^3 + 4 * z^2 - 2 * z + 1) * (2 * z^2 - 3 * z + 5) =
  10 * z^5 - 8 * z^4 + 11 * z^3 + 5 * z^2 - 10 * z + 5 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_expansion_l4075_407550


namespace NUMINAMATH_CALUDE_third_term_is_nine_l4075_407532

/-- A sequence of 5 numbers with specific properties -/
def MagazineSequence (a : Fin 5 → ℕ) : Prop :=
  a 0 = 3 ∧ a 1 = 4 ∧ a 3 = 9 ∧ a 4 = 13 ∧
  ∀ i : Fin 3, (a (i + 1) - a i) - (a (i + 2) - a (i + 1)) = 
               (a (i + 2) - a (i + 1)) - (a (i + 3) - a (i + 2))

/-- The third term in the sequence is 9 -/
theorem third_term_is_nine (a : Fin 5 → ℕ) (h : MagazineSequence a) : a 2 = 9 := by
  sorry

end NUMINAMATH_CALUDE_third_term_is_nine_l4075_407532


namespace NUMINAMATH_CALUDE_geometric_sequence_formula_l4075_407537

/-- A geometric sequence {a_n} with a_1 = 3 and a_4 = 81 has the general term formula a_n = 3^n -/
theorem geometric_sequence_formula (a : ℕ → ℝ) (h1 : a 1 = 3) (h4 : a 4 = 81) :
  ∀ n : ℕ, a n = 3^n := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_formula_l4075_407537


namespace NUMINAMATH_CALUDE_smallest_sum_B_plus_b_l4075_407552

/-- Given that B is a digit in base 5 and b is a base greater than 6,
    such that BBB₅ = 44ᵦ, prove that the smallest possible sum of B + b is 8. -/
theorem smallest_sum_B_plus_b : ∃ (B b : ℕ),
  (0 < B) ∧ (B < 5) ∧  -- B is a digit in base 5
  (b > 6) ∧            -- b is a base greater than 6
  (31 * B = 4 * b + 4) ∧  -- BBB₅ = 44ᵦ
  (∀ (B' b' : ℕ), 
    (0 < B') ∧ (B' < 5) ∧ (b' > 6) ∧ (31 * B' = 4 * b' + 4) →
    B + b ≤ B' + b') :=
by sorry

end NUMINAMATH_CALUDE_smallest_sum_B_plus_b_l4075_407552


namespace NUMINAMATH_CALUDE_log_equation_solution_l4075_407505

-- Define the logarithm function
noncomputable def log (base : ℝ) (x : ℝ) : ℝ := Real.log x / Real.log base

-- State the theorem
theorem log_equation_solution (x k : ℝ) :
  log k x * log 3 k = 4 → k = 9 → x = 81 := by
  sorry

end NUMINAMATH_CALUDE_log_equation_solution_l4075_407505


namespace NUMINAMATH_CALUDE_prob_win_5_eq_prob_win_total_eq_l4075_407506

/-- Probability of Team A winning a single game -/
def p : ℝ := 0.6

/-- Probability of Team B winning a single game -/
def q : ℝ := 1 - p

/-- Number of games in the series -/
def n : ℕ := 7

/-- Number of games needed to win the series -/
def k : ℕ := 4

/-- Probability of Team A winning the championship after exactly 5 games -/
def prob_win_5 : ℝ := Nat.choose 4 3 * p^4 * q

/-- Probability of Team A winning the championship -/
def prob_win_total : ℝ := 
  p^4 + Nat.choose 4 3 * p^4 * q + Nat.choose 5 3 * p^4 * q^2 + Nat.choose 6 3 * p^4 * q^3

/-- Theorem stating the probability of Team A winning after exactly 5 games -/
theorem prob_win_5_eq : prob_win_5 = 0.20736 := by sorry

/-- Theorem stating the overall probability of Team A winning the championship -/
theorem prob_win_total_eq : prob_win_total = 0.710208 := by sorry

end NUMINAMATH_CALUDE_prob_win_5_eq_prob_win_total_eq_l4075_407506


namespace NUMINAMATH_CALUDE_quadratic_function_m_value_l4075_407530

/-- A quadratic function of the form y = mx^2 - 8x + m(m-1) that passes through the origin -/
def quadratic_function (m : ℝ) (x : ℝ) : ℝ := m * x^2 - 8 * x + m * (m - 1)

/-- The quadratic function passes through the origin -/
def passes_through_origin (m : ℝ) : Prop := quadratic_function m 0 = 0

/-- The theorem stating that m = 1 for the given quadratic function passing through the origin -/
theorem quadratic_function_m_value :
  ∃ m : ℝ, passes_through_origin m ∧ m = 1 :=
sorry

end NUMINAMATH_CALUDE_quadratic_function_m_value_l4075_407530


namespace NUMINAMATH_CALUDE_rectangle_breadth_ratio_l4075_407591

/-- Given a rectangle where the length is halved and the area is reduced by 50%,
    prove that the ratio of new breadth to original breadth is 0.5 -/
theorem rectangle_breadth_ratio
  (L B : ℝ)  -- Original length and breadth
  (L' B' : ℝ) -- New length and breadth
  (h1 : L' = L / 2)  -- New length is half of original
  (h2 : L' * B' = (L * B) / 2)  -- New area is half of original
  : B' / B = 0.5 := by
  sorry


end NUMINAMATH_CALUDE_rectangle_breadth_ratio_l4075_407591


namespace NUMINAMATH_CALUDE_equation_equivalence_l4075_407539

theorem equation_equivalence (p q : ℝ) 
  (hp1 : p ≠ 0) (hp2 : p ≠ 5) (hq1 : q ≠ 0) (hq2 : q ≠ 7) :
  (3 / p + 4 / q = 1 / 3) ↔ (9 * q / (q - 12) = p) :=
by sorry

end NUMINAMATH_CALUDE_equation_equivalence_l4075_407539


namespace NUMINAMATH_CALUDE_small_circle_area_l4075_407575

/-- Configuration of circles -/
structure CircleConfiguration where
  large_circle_area : ℝ
  small_circle_count : ℕ
  small_circles_inscribed : Prop

/-- Theorem: In a configuration where 6 small circles of equal radius are inscribed 
    in a large circle with an area of 120, the area of each small circle is 40 -/
theorem small_circle_area 
  (config : CircleConfiguration) 
  (h1 : config.large_circle_area = 120)
  (h2 : config.small_circle_count = 6)
  (h3 : config.small_circles_inscribed) :
  ∃ (small_circle_area : ℝ), small_circle_area = 40 ∧ 
    config.small_circle_count * small_circle_area = config.large_circle_area :=
by
  sorry


end NUMINAMATH_CALUDE_small_circle_area_l4075_407575


namespace NUMINAMATH_CALUDE_scientific_notation_of_8500_l4075_407502

theorem scientific_notation_of_8500 : 
  ∃ (a : ℝ) (n : ℤ), 8500 = a * (10 : ℝ) ^ n ∧ 1 ≤ a ∧ a < 10 :=
by
  -- The proof would go here
  sorry

end NUMINAMATH_CALUDE_scientific_notation_of_8500_l4075_407502


namespace NUMINAMATH_CALUDE_water_bucket_addition_l4075_407583

theorem water_bucket_addition (initial_water : Real) (added_water : Real) :
  initial_water = 3 ∧ added_water = 6.8 → initial_water + added_water = 9.8 := by
  sorry

end NUMINAMATH_CALUDE_water_bucket_addition_l4075_407583


namespace NUMINAMATH_CALUDE_bridge_length_is_two_km_l4075_407594

/-- The length of a bridge crossed by a man -/
def bridge_length (speed : ℝ) (time : ℝ) : ℝ := speed * time

/-- Theorem: The length of a bridge is 2 km when crossed by a man walking at 8 km/hr in 15 minutes -/
theorem bridge_length_is_two_km :
  let speed := 8 -- km/hr
  let time := 15 / 60 -- 15 minutes converted to hours
  bridge_length speed time = 2 := by
  sorry

end NUMINAMATH_CALUDE_bridge_length_is_two_km_l4075_407594


namespace NUMINAMATH_CALUDE_fraction_equals_point_eight_seven_five_l4075_407540

theorem fraction_equals_point_eight_seven_five (a : ℕ+) :
  (a : ℚ) / (a + 35 : ℚ) = 7/8 → a = 245 := by
  sorry

end NUMINAMATH_CALUDE_fraction_equals_point_eight_seven_five_l4075_407540


namespace NUMINAMATH_CALUDE_bench_cost_l4075_407534

theorem bench_cost (bench_cost table_cost : ℕ) : 
  bench_cost + table_cost = 750 → 
  table_cost = 2 * bench_cost →
  bench_cost = 250 := by
  sorry

end NUMINAMATH_CALUDE_bench_cost_l4075_407534


namespace NUMINAMATH_CALUDE_sufficient_not_necessary_condition_l4075_407501

theorem sufficient_not_necessary_condition (a : ℝ) :
  (a > 1 → (1 / a < 1)) ∧ ¬((1 / a < 1) → (a > 1)) := by
  sorry

end NUMINAMATH_CALUDE_sufficient_not_necessary_condition_l4075_407501


namespace NUMINAMATH_CALUDE_warehouse_solution_l4075_407549

/-- Represents the problem of determining the number of warehouses on a straight road. -/
def WarehouseProblem (n : ℕ) : Prop :=
  -- n is odd
  ∃ k : ℕ, n = 2*k + 1 ∧
  -- Distance between warehouses is 1 km
  -- Each warehouse contains 8 tons of goods
  -- Truck capacity is 8 tons
  -- These conditions are implicit in the problem setup
  -- Optimal route distance is 300 km
  2 * k * (k + 1) - k = 300

/-- The solution to the warehouse problem is 25 warehouses. -/
theorem warehouse_solution : WarehouseProblem 25 := by
  sorry

#check warehouse_solution

end NUMINAMATH_CALUDE_warehouse_solution_l4075_407549


namespace NUMINAMATH_CALUDE_perpendicular_vectors_l4075_407531

/-- Given two vectors a and b in ℝ³, prove that they are perpendicular if and only if x = 10/3 -/
theorem perpendicular_vectors (a b : ℝ × ℝ × ℝ) :
  a = (2, -1, 3) → b = (-4, 2, x) → (a • b = 0 ↔ x = 10/3) :=
by sorry

end NUMINAMATH_CALUDE_perpendicular_vectors_l4075_407531


namespace NUMINAMATH_CALUDE_janes_drinks_l4075_407597

theorem janes_drinks (b m d : ℕ) : 
  b + m + d = 5 →
  (90 * b + 40 * m + 30 * d) % 100 = 0 →
  d = 4 := by
sorry

end NUMINAMATH_CALUDE_janes_drinks_l4075_407597


namespace NUMINAMATH_CALUDE_john_paid_1273_l4075_407523

/-- Calculates the amount John paid out of pocket for his purchases --/
def john_out_of_pocket (exchange_rate : ℝ) (computer_cost gaming_chair_cost accessories_cost : ℝ)
  (computer_discount gaming_chair_discount : ℝ) (sales_tax : ℝ)
  (playstation_value playstation_discount bicycle_price : ℝ) : ℝ :=
  let discounted_computer := computer_cost * (1 - computer_discount)
  let discounted_chair := gaming_chair_cost * (1 - gaming_chair_discount)
  let total_before_tax := discounted_computer + discounted_chair + accessories_cost
  let total_after_tax := total_before_tax * (1 + sales_tax)
  let playstation_sale := playstation_value * (1 - playstation_discount)
  let total_sales := playstation_sale + bicycle_price
  total_after_tax - total_sales

/-- Theorem stating that John paid $1273 out of pocket --/
theorem john_paid_1273 :
  john_out_of_pocket 100 1500 400 300 0.2 0.1 0.05 600 0.2 200 = 1273 := by
  sorry

end NUMINAMATH_CALUDE_john_paid_1273_l4075_407523


namespace NUMINAMATH_CALUDE_dodecahedron_intersection_area_l4075_407518

/-- The area of a regular pentagon formed by intersecting a plane with a regular dodecahedron -/
theorem dodecahedron_intersection_area (s : ℝ) :
  let dodecahedron_side_length : ℝ := s
  let intersection_pentagon_side_length : ℝ := s / 2
  let intersection_pentagon_area : ℝ := (5 / 4) * (intersection_pentagon_side_length ^ 2) * ((Real.sqrt 5 + 1) / 2)
  intersection_pentagon_area = (5 * s^2 * (Real.sqrt 5 + 1)) / 16 := by
  sorry

end NUMINAMATH_CALUDE_dodecahedron_intersection_area_l4075_407518


namespace NUMINAMATH_CALUDE_original_jeans_price_l4075_407562

/-- Proves that the original price of jeans is $49.00 given the discount conditions --/
theorem original_jeans_price (x : ℝ) : 
  (0.5 * x - 10 = 14.5) → x = 49 := by
  sorry

end NUMINAMATH_CALUDE_original_jeans_price_l4075_407562


namespace NUMINAMATH_CALUDE_scalene_triangle_two_angles_less_than_60_l4075_407580

/-- A scalene triangle with side lengths in arithmetic progression has two angles less than 60 degrees. -/
theorem scalene_triangle_two_angles_less_than_60 (a d : ℝ) 
  (h_d_pos : d > 0) 
  (h_scalene : a - d ≠ a ∧ a ≠ a + d ∧ a - d ≠ a + d) :
  ∃ (α β : ℝ), α + β + (180 - α - β) = 180 ∧ 
               0 < α ∧ α < 60 ∧ 
               0 < β ∧ β < 60 := by
  sorry


end NUMINAMATH_CALUDE_scalene_triangle_two_angles_less_than_60_l4075_407580


namespace NUMINAMATH_CALUDE_fuel_tank_capacity_l4075_407556

/-- Represents the problem of determining a fuel tank's capacity --/
theorem fuel_tank_capacity :
  ∀ (capacity : ℝ) 
    (ethanol_percent_A : ℝ) 
    (ethanol_percent_B : ℝ) 
    (total_ethanol : ℝ) 
    (fuel_A_volume : ℝ),
  ethanol_percent_A = 0.12 →
  ethanol_percent_B = 0.16 →
  total_ethanol = 30 →
  fuel_A_volume = 106 →
  ethanol_percent_A * fuel_A_volume + 
  ethanol_percent_B * (capacity - fuel_A_volume) = total_ethanol →
  capacity = 214 := by
sorry

end NUMINAMATH_CALUDE_fuel_tank_capacity_l4075_407556


namespace NUMINAMATH_CALUDE_total_stamps_is_72_l4075_407517

/-- Calculates the total number of stamps needed for Valerie's mailing --/
def total_stamps : ℕ :=
  let thank_you_cards := 5
  let thank_you_stamps_per_card := 2
  let water_bill_stamps := 3
  let electric_bill_stamps := 2
  let internet_bill_stamps := 5
  let rebate_stamps_per_envelope := 2
  let job_app_stamps_per_envelope := 1
  let bill_types := 3
  let additional_rebates := 3

  let bill_stamps := water_bill_stamps + electric_bill_stamps + internet_bill_stamps
  let rebates := bill_types + additional_rebates
  let job_applications := 2 * rebates

  thank_you_cards * thank_you_stamps_per_card +
  bill_stamps +
  rebates * rebate_stamps_per_envelope +
  job_applications * job_app_stamps_per_envelope

theorem total_stamps_is_72 : total_stamps = 72 := by
  sorry

end NUMINAMATH_CALUDE_total_stamps_is_72_l4075_407517


namespace NUMINAMATH_CALUDE_A_work_time_l4075_407570

/-- The number of days it takes B to complete the work alone -/
def B_days : ℝ := 8

/-- The total payment for the work -/
def total_payment : ℝ := 3600

/-- The payment to C -/
def C_payment : ℝ := 450

/-- The number of days it takes A, B, and C to complete the work together -/
def combined_days : ℝ := 3

/-- The number of days it takes A to complete the work alone -/
def A_days : ℝ := 56

theorem A_work_time :
  ∃ (C_rate : ℝ),
    (1 / A_days + 1 / B_days + C_rate = 1 / combined_days) ∧
    (1 / A_days : ℝ) / (1 / B_days) = (total_payment - C_payment) / C_payment :=
by sorry

end NUMINAMATH_CALUDE_A_work_time_l4075_407570


namespace NUMINAMATH_CALUDE_dog_food_calculation_l4075_407592

theorem dog_food_calculation (num_dogs : ℕ) (food_per_dog : ℕ) (vacation_days : ℕ) :
  num_dogs = 4 →
  food_per_dog = 250 →
  vacation_days = 14 →
  (num_dogs * food_per_dog * vacation_days : ℕ) / 1000 = 14 := by
  sorry

end NUMINAMATH_CALUDE_dog_food_calculation_l4075_407592

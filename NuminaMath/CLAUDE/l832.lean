import Mathlib

namespace NUMINAMATH_CALUDE_age_ratio_proof_l832_83268

/-- Proves that given the conditions about A's and B's ages, the ratio between A's age 4 years hence and B's age 4 years ago is 3:1 -/
theorem age_ratio_proof (x : ℕ) (h1 : 5 * x > 4) (h2 : 3 * x > 4) : 
  (5 * x + 4) / (3 * x - 4) = 3 := by
  sorry

end NUMINAMATH_CALUDE_age_ratio_proof_l832_83268


namespace NUMINAMATH_CALUDE_min_points_last_two_games_l832_83260

theorem min_points_last_two_games 
  (scores : List ℕ)
  (h1 : scores.length = 20)
  (h2 : scores[14] = 26 ∧ scores[15] = 15 ∧ scores[16] = 12 ∧ scores[17] = 24)
  (h3 : (scores.take 18).sum / 18 > (scores.take 14).sum / 14)
  (h4 : scores.sum / 20 > 20) :
  scores[18] + scores[19] ≥ 58 := by
  sorry

end NUMINAMATH_CALUDE_min_points_last_two_games_l832_83260


namespace NUMINAMATH_CALUDE_charity_amount_l832_83278

/-- The amount of money raised from the rubber duck race -/
def money_raised (small_price medium_price large_price : ℚ) 
  (small_qty medium_qty large_qty : ℕ) : ℚ :=
  small_price * small_qty + medium_price * medium_qty + large_price * large_qty

/-- Theorem stating the total amount raised for charity -/
theorem charity_amount : 
  money_raised 2 3 5 150 221 185 = 1888 := by sorry

end NUMINAMATH_CALUDE_charity_amount_l832_83278


namespace NUMINAMATH_CALUDE_jawbreakers_eaten_l832_83220

def package_size : ℕ := 8
def jawbreakers_left : ℕ := 4

theorem jawbreakers_eaten : ℕ := by
  sorry

end NUMINAMATH_CALUDE_jawbreakers_eaten_l832_83220


namespace NUMINAMATH_CALUDE_absolute_value_sum_l832_83231

theorem absolute_value_sum (a : ℝ) (h : 3 < a ∧ a < 4) : |a - 3| + |a - 4| = 1 := by
  sorry

end NUMINAMATH_CALUDE_absolute_value_sum_l832_83231


namespace NUMINAMATH_CALUDE_circle_equation_l832_83265

/-- The equation of a circle with center (2, 1) that shares a common chord with another circle,
    where the chord lies on a line passing through a specific point. -/
theorem circle_equation (x y : ℝ) : 
  ∃ (r : ℝ), 
    -- The first circle has center (2, 1) and radius r
    ((x - 2)^2 + (y - 1)^2 = r^2) ∧
    -- The second circle is described by x^2 + y^2 - 3x = 0
    (∃ (x₀ y₀ : ℝ), x₀^2 + y₀^2 - 3*x₀ = 0) ∧
    -- The common chord lies on a line passing through (5, -2)
    (∃ (a b c : ℝ), a*5 + b*(-2) + c = 0 ∧ a*x + b*y + c = 0) →
    -- The equation of the first circle is (x-2)^2 + (y-1)^2 = 4
    (x - 2)^2 + (y - 1)^2 = 4 :=
by
  sorry

end NUMINAMATH_CALUDE_circle_equation_l832_83265


namespace NUMINAMATH_CALUDE_g_of_2_eq_11_l832_83230

/-- The function g(x) = x^3 + x^2 - 1 -/
def g (x : ℝ) : ℝ := x^3 + x^2 - 1

/-- Theorem: g(2) = 11 -/
theorem g_of_2_eq_11 : g 2 = 11 := by
  sorry

end NUMINAMATH_CALUDE_g_of_2_eq_11_l832_83230


namespace NUMINAMATH_CALUDE_min_value_2a_plus_b_l832_83252

theorem min_value_2a_plus_b (a b : ℝ) (h : Real.log a + Real.log b = Real.log (a + 2*b)) :
  (∀ x y : ℝ, Real.log x + Real.log y = Real.log (x + 2*y) → 2*x + y ≥ 2*a + b) ∧ (∃ x y : ℝ, Real.log x + Real.log y = Real.log (x + 2*y) ∧ 2*x + y = 9) :=
by sorry

end NUMINAMATH_CALUDE_min_value_2a_plus_b_l832_83252


namespace NUMINAMATH_CALUDE_arithmetic_geometric_sequence_third_term_l832_83251

/-- An arithmetic-geometric sequence -/
structure ArithmeticGeometricSequence where
  a : ℕ → ℚ
  q : ℚ
  first_term : ℚ
  common_diff : ℚ
  seq_def : ∀ n : ℕ, a n = first_term * q^n + common_diff * (1 - q^n) / (1 - q)

/-- Sum of first n terms of an arithmetic-geometric sequence -/
def sum_n (seq : ArithmeticGeometricSequence) (n : ℕ) : ℚ :=
  seq.first_term * (1 - seq.q^n) / (1 - seq.q)

theorem arithmetic_geometric_sequence_third_term
  (seq : ArithmeticGeometricSequence)
  (h1 : sum_n seq 6 / sum_n seq 3 = -19/8)
  (h2 : seq.a 4 - seq.a 2 = -15/8) :
  seq.a 3 = 9/4 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_geometric_sequence_third_term_l832_83251


namespace NUMINAMATH_CALUDE_green_team_score_l832_83203

/-- Given a winning team's score and their lead over the opponent,
    calculate the opponent's (losing team's) score. -/
def opponent_score (winning_score lead : ℕ) : ℕ :=
  winning_score - lead

/-- Theorem stating that given a winning score of 68 and a lead of 29,
    the opponent's score is 39. -/
theorem green_team_score :
  opponent_score 68 29 = 39 := by
  sorry

end NUMINAMATH_CALUDE_green_team_score_l832_83203


namespace NUMINAMATH_CALUDE_unique_solution_condition_l832_83235

theorem unique_solution_condition (k : ℝ) : 
  (∃! p : ℝ × ℝ, p.2 = p.1^2 + 1 ∧ p.2 = 4*p.1 + k) ↔ k = 0 :=
by sorry

end NUMINAMATH_CALUDE_unique_solution_condition_l832_83235


namespace NUMINAMATH_CALUDE_planes_with_parallel_lines_are_parallel_or_intersecting_l832_83298

/-- Two planes in 3D space -/
structure Plane3D where
  -- Add necessary fields here
  
/-- A straight line in 3D space -/
structure Line3D where
  -- Add necessary fields here

/-- Predicate to check if a line is contained in a plane -/
def line_in_plane (l : Line3D) (p : Plane3D) : Prop :=
  sorry

/-- Predicate to check if two lines are parallel -/
def lines_parallel (l1 l2 : Line3D) : Prop :=
  sorry

/-- Predicate to check if two planes are parallel -/
def planes_parallel (p1 p2 : Plane3D) : Prop :=
  sorry

/-- Predicate to check if two planes are intersecting -/
def planes_intersecting (p1 p2 : Plane3D) : Prop :=
  sorry

theorem planes_with_parallel_lines_are_parallel_or_intersecting 
  (p1 p2 : Plane3D) (l1 l2 : Line3D) 
  (h1 : line_in_plane l1 p1) 
  (h2 : line_in_plane l2 p2) 
  (h3 : lines_parallel l1 l2) : 
  planes_parallel p1 p2 ∨ planes_intersecting p1 p2 :=
sorry

end NUMINAMATH_CALUDE_planes_with_parallel_lines_are_parallel_or_intersecting_l832_83298


namespace NUMINAMATH_CALUDE_number_operations_l832_83245

theorem number_operations (x : ℤ) : 
  (((x + 7) * 3 - 12) / 6 : ℚ) = -8 → x = -19 := by
  sorry

end NUMINAMATH_CALUDE_number_operations_l832_83245


namespace NUMINAMATH_CALUDE_fraction_product_square_l832_83253

theorem fraction_product_square : (8 / 9 : ℚ)^2 * (1 / 3 : ℚ)^2 = 64 / 729 := by
  sorry

end NUMINAMATH_CALUDE_fraction_product_square_l832_83253


namespace NUMINAMATH_CALUDE_absolute_value_ratio_l832_83218

theorem absolute_value_ratio (a b : ℝ) (h1 : a ≠ 0) (h2 : b ≠ 0) (h3 : a^2 + b^2 = 8*a*b) :
  |a + b| / |a - b| = Real.sqrt 15 / 3 := by
  sorry

end NUMINAMATH_CALUDE_absolute_value_ratio_l832_83218


namespace NUMINAMATH_CALUDE_negation_of_tangent_positive_l832_83261

open Real

theorem negation_of_tangent_positive :
  (¬ ∀ x : ℝ, x ∈ Set.Ioo (-π/2) (π/2) → tan x > 0) ↔
  (∃ x : ℝ, x ∈ Set.Ioo (-π/2) (π/2) ∧ tan x ≤ 0) :=
by sorry

end NUMINAMATH_CALUDE_negation_of_tangent_positive_l832_83261


namespace NUMINAMATH_CALUDE_perpendicular_planes_theorem_l832_83255

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the perpendicular and parallel relations
variable (perpLine : Line → Line → Prop)
variable (perpLinePlane : Line → Plane → Prop)
variable (perpPlane : Plane → Plane → Prop)
variable (parallelLinePlane : Line → Plane → Prop)

-- Define the theorem
theorem perpendicular_planes_theorem 
  (a b : Line) (α β : Plane) : 
  (∀ (l1 l2 : Line) (p1 p2 : Plane), 
    (perpLine l1 l2 ∧ perpLinePlane l1 p1 → ¬(parallelLinePlane l2 p1)) ∧
    (perpPlane p1 p2 ∧ parallelLinePlane l1 p1 → ¬(perpLinePlane l1 p2)) ∧
    (perpLinePlane l1 p2 ∧ perpPlane p1 p2 → ¬(parallelLinePlane l1 p1))) →
  (perpLine a b ∧ perpLinePlane a α ∧ perpLinePlane b β → perpPlane α β) :=
sorry

end NUMINAMATH_CALUDE_perpendicular_planes_theorem_l832_83255


namespace NUMINAMATH_CALUDE_filter_kit_price_calculation_l832_83287

/-- The price of a camera lens filter kit -/
def filter_kit_price (price1 price2 price3 : ℝ) (discount : ℝ) : ℝ :=
  let total_individual := 2 * price1 + 2 * price2 + price3
  total_individual * (1 - discount)

/-- Theorem stating the price of the filter kit -/
theorem filter_kit_price_calculation :
  filter_kit_price 16.45 14.05 19.50 0.08 = 74.06 := by
  sorry

end NUMINAMATH_CALUDE_filter_kit_price_calculation_l832_83287


namespace NUMINAMATH_CALUDE_v_2008_eq_352_l832_83259

/-- Defines the sequence v_n as described in the problem -/
def v : ℕ → ℕ 
| 0 => 1  -- First term
| n + 1 => 
  let group := (Nat.sqrt (8 * (n + 1) + 1) - 1) / 2  -- Determine which group n+1 belongs to
  let groupStart := group * (group + 1) / 2  -- Starting position of the group
  let offset := n + 1 - groupStart  -- Position within the group
  (group + 1) + 3 * ((groupStart - 1) + offset)  -- Calculate the term

/-- The 2008th term of the sequence is 352 -/
theorem v_2008_eq_352 : v 2007 = 352 := by
  sorry

end NUMINAMATH_CALUDE_v_2008_eq_352_l832_83259


namespace NUMINAMATH_CALUDE_all_propositions_false_l832_83272

-- Define a type for lines in 3D space
variable (Line : Type)

-- Define relationships between lines
variable (parallel : Line → Line → Prop)
variable (coplanar : Line → Line → Prop)
variable (intersect : Line → Line → Prop)

-- State the theorem
theorem all_propositions_false :
  (∀ a b c : Line,
    (parallel a b ∧ ¬coplanar a c) → ¬coplanar b c) ∧
  (∀ a b c : Line,
    (coplanar a b ∧ ¬coplanar b c) → ¬coplanar a c) ∧
  (∀ a b c : Line,
    (¬coplanar a b ∧ coplanar a c) → ¬coplanar b c) ∧
  (∀ a b c : Line,
    (¬coplanar a b ∧ ¬intersect b c) → ¬intersect a c) →
  False :=
sorry

end NUMINAMATH_CALUDE_all_propositions_false_l832_83272


namespace NUMINAMATH_CALUDE_max_xy_value_l832_83256

theorem max_xy_value (x y : ℕ+) (h : 7 * x + 4 * y = 140) : x * y ≤ 168 := by
  sorry

end NUMINAMATH_CALUDE_max_xy_value_l832_83256


namespace NUMINAMATH_CALUDE_cookie_boxes_theorem_l832_83270

theorem cookie_boxes_theorem (n : ℕ) : 
  (∃ (m a : ℕ), 
    m = n - 11 ∧ 
    a = n - 2 ∧ 
    m ≥ 1 ∧ 
    a ≥ 1 ∧ 
    m + a < n) → 
  n = 12 := by
  sorry

end NUMINAMATH_CALUDE_cookie_boxes_theorem_l832_83270


namespace NUMINAMATH_CALUDE_ten_by_ten_grid_triangles_l832_83238

/-- Represents a square grid -/
structure SquareGrid :=
  (size : ℕ)

/-- Counts the number of triangles formed by drawing a diagonal in a square grid -/
def countTriangles (grid : SquareGrid) : ℕ :=
  (grid.size + 1) * (grid.size + 1) - (grid.size + 1)

/-- Theorem: In a 10 × 10 square grid with one diagonal drawn, 110 triangles are formed -/
theorem ten_by_ten_grid_triangles :
  countTriangles { size := 10 } = 110 := by
  sorry

end NUMINAMATH_CALUDE_ten_by_ten_grid_triangles_l832_83238


namespace NUMINAMATH_CALUDE_derivative_at_one_l832_83273

open Real

noncomputable def f (x : ℝ) (f'1 : ℝ) : ℝ := 2 * x * f'1 + log x

theorem derivative_at_one (f'1 : ℝ) :
  (∀ x > 0, f x f'1 = 2 * x * f'1 + log x) →
  deriv (f · f'1) 1 = -1 :=
by sorry

end NUMINAMATH_CALUDE_derivative_at_one_l832_83273


namespace NUMINAMATH_CALUDE_quadratic_translation_l832_83246

-- Define the original function
def f (x : ℝ) : ℝ := x^2 + 3

-- Define the transformed function
def g (x : ℝ) : ℝ := (x + 2)^2 + 1

-- Theorem statement
theorem quadratic_translation (x : ℝ) : 
  g x = f (x + 2) - 2 := by sorry

end NUMINAMATH_CALUDE_quadratic_translation_l832_83246


namespace NUMINAMATH_CALUDE_hyperbola_asymptotes_l832_83223

/-- Given a hyperbola with equation x²/a² - y²/b² = 1 where a > 0, b > 0,
    and eccentricity 2, its asymptotes have the equation y = ± √3 x -/
theorem hyperbola_asymptotes (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  let e := 2  -- eccentricity
  let c := e * a  -- focal distance
  let hyperbola := fun (x y : ℝ) => x^2 / a^2 - y^2 / b^2 = 1
  let asymptote := fun (x y : ℝ) => y = Real.sqrt 3 * x ∨ y = -Real.sqrt 3 * x
  (∀ x y, hyperbola x y → b^2 = c^2 - a^2) →
  (∀ x y, asymptote x y ↔ (x / a - y / b = 0 ∨ x / a + y / b = 0)) :=
by sorry

end NUMINAMATH_CALUDE_hyperbola_asymptotes_l832_83223


namespace NUMINAMATH_CALUDE_amp_2_neg1_1_l832_83274

-- Define the & operation
def amp (a b c : ℝ) : ℝ := 3 * b^2 - 4 * a * c

-- Theorem statement
theorem amp_2_neg1_1 : amp 2 (-1) 1 = -5 := by
  sorry

end NUMINAMATH_CALUDE_amp_2_neg1_1_l832_83274


namespace NUMINAMATH_CALUDE_sequence_expression_l832_83213

theorem sequence_expression (a : ℕ → ℕ) (h1 : a 1 = 33) 
    (h2 : ∀ n : ℕ, a (n + 1) - a n = 2 * n) : 
  ∀ n : ℕ, a n = n^2 - n + 33 := by
sorry

end NUMINAMATH_CALUDE_sequence_expression_l832_83213


namespace NUMINAMATH_CALUDE_work_time_for_c_l832_83279

/-- The time it takes for worker c to complete the work alone, given the combined work rates of pairs of workers. -/
theorem work_time_for_c (a b c : ℝ) 
  (h1 : a + b = 1/4)   -- a and b can do the work in 4 days
  (h2 : b + c = 1/6)   -- b and c can do the work in 6 days
  (h3 : c + a = 1/3) : -- c and a can do the work in 3 days
  1/c = 8 := by sorry

end NUMINAMATH_CALUDE_work_time_for_c_l832_83279


namespace NUMINAMATH_CALUDE_simplify_fraction_l832_83258

theorem simplify_fraction :
  (5 : ℝ) / (Real.sqrt 50 + 3 * Real.sqrt 8 + 2 * Real.sqrt 18) = 5 * Real.sqrt 2 / 34 := by
  sorry

end NUMINAMATH_CALUDE_simplify_fraction_l832_83258


namespace NUMINAMATH_CALUDE_second_number_value_l832_83266

theorem second_number_value (x : ℚ) 
  (sum_condition : 2*x + x + (2/3)*x + (1/2)*x = 330) : x = 46 := by
  sorry

end NUMINAMATH_CALUDE_second_number_value_l832_83266


namespace NUMINAMATH_CALUDE_second_bill_overdue_months_l832_83257

/-- Calculates the number of months a bill is overdue given the total amount owed and the conditions of three bills -/
def months_overdue (total_owed : ℚ) (bill1_amount : ℚ) (bill1_interest_rate : ℚ) (bill1_months : ℕ)
                   (bill2_amount : ℚ) (bill2_fee : ℚ)
                   (bill3_fee1 : ℚ) (bill3_fee2 : ℚ) : ℕ :=
  let bill1_total := bill1_amount + bill1_amount * bill1_interest_rate * bill1_months
  let bill3_total := bill3_fee1 + bill3_fee2
  let bill2_overdue := total_owed - bill1_total - bill3_total
  Nat.ceil (bill2_overdue / bill2_fee)

/-- The number of months the second bill is overdue is 18 -/
theorem second_bill_overdue_months :
  months_overdue 1234 200 (1/10) 2 130 50 40 80 = 18 := by
  sorry

end NUMINAMATH_CALUDE_second_bill_overdue_months_l832_83257


namespace NUMINAMATH_CALUDE_f_2010_equals_zero_l832_83222

def is_odd_function (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = -f x

theorem f_2010_equals_zero
  (f : ℝ → ℝ)
  (h_odd : is_odd_function f)
  (h_period : ∀ x, f (x + 3) = -f x) :
  f 2010 = 0 := by
  sorry

end NUMINAMATH_CALUDE_f_2010_equals_zero_l832_83222


namespace NUMINAMATH_CALUDE_coins_percentage_of_dollar_l832_83294

/-- The value of a penny in cents -/
def penny_value : ℕ := 1

/-- The value of a nickel in cents -/
def nickel_value : ℕ := 5

/-- The value of a dime in cents -/
def dime_value : ℕ := 10

/-- The value of a quarter in cents -/
def quarter_value : ℕ := 25

/-- The number of pennies in Samantha's purse -/
def num_pennies : ℕ := 2

/-- The number of nickels in Samantha's purse -/
def num_nickels : ℕ := 1

/-- The number of dimes in Samantha's purse -/
def num_dimes : ℕ := 3

/-- The number of quarters in Samantha's purse -/
def num_quarters : ℕ := 2

/-- The total value of coins in Samantha's purse as a percentage of a dollar -/
theorem coins_percentage_of_dollar :
  (num_pennies * penny_value + num_nickels * nickel_value +
   num_dimes * dime_value + num_quarters * quarter_value) * 100 / 100 = 87 := by
  sorry

end NUMINAMATH_CALUDE_coins_percentage_of_dollar_l832_83294


namespace NUMINAMATH_CALUDE_derivative_positive_implies_increasing_l832_83269

open Set

theorem derivative_positive_implies_increasing
  {f : ℝ → ℝ} {I : Set ℝ} (hI : IsOpen I) (hf : DifferentiableOn ℝ f I)
  (h : ∀ x ∈ I, deriv f x > 0) :
  StrictMonoOn f I :=
sorry

end NUMINAMATH_CALUDE_derivative_positive_implies_increasing_l832_83269


namespace NUMINAMATH_CALUDE_sphere_only_orientation_independent_l832_83286

-- Define the types of 3D objects we're considering
inductive Object3D
  | Cube
  | RegularTetrahedron
  | RegularTriangularPyramid
  | Sphere

-- Define a function that determines if an object's orthographic projections are orientation-independent
def hasOrientationIndependentProjections (obj : Object3D) : Prop :=
  match obj with
  | Object3D.Sphere => True
  | _ => False

-- Theorem statement
theorem sphere_only_orientation_independent :
  ∀ (obj : Object3D), hasOrientationIndependentProjections obj ↔ obj = Object3D.Sphere :=
by sorry

end NUMINAMATH_CALUDE_sphere_only_orientation_independent_l832_83286


namespace NUMINAMATH_CALUDE_sum_of_greater_than_l832_83232

theorem sum_of_greater_than (a b c d : ℝ) (h1 : a > b) (h2 : c > d) : a + c > b + d := by
  sorry

end NUMINAMATH_CALUDE_sum_of_greater_than_l832_83232


namespace NUMINAMATH_CALUDE_cats_asleep_l832_83233

theorem cats_asleep (total : ℕ) (awake : ℕ) (h1 : total = 98) (h2 : awake = 6) :
  total - awake = 92 := by
  sorry

end NUMINAMATH_CALUDE_cats_asleep_l832_83233


namespace NUMINAMATH_CALUDE_bike_cost_theorem_l832_83248

theorem bike_cost_theorem (marion_cost : ℕ) : 
  marion_cost = 356 → 
  (marion_cost + 2 * marion_cost + 3 * marion_cost : ℕ) = 2136 := by
  sorry

end NUMINAMATH_CALUDE_bike_cost_theorem_l832_83248


namespace NUMINAMATH_CALUDE_smallest_x_squared_l832_83226

/-- Represents a trapezoid ABCD with specific properties -/
structure Trapezoid where
  AB : ℝ
  CD : ℝ
  x : ℝ
  h : AB = 120 ∧ CD = 25

/-- A circle is tangent to AD if its center is on AB and touches AD -/
def is_tangent_circle (t : Trapezoid) (center : ℝ) : Prop :=
  0 ≤ center ∧ center ≤ t.AB ∧ ∃ (point : ℝ), 0 ≤ point ∧ point ≤ t.x

/-- The theorem stating the smallest possible value of x^2 -/
theorem smallest_x_squared (t : Trapezoid) : 
  (∃ center, is_tangent_circle t center) → 
  (∀ y, (∃ center, is_tangent_circle { AB := t.AB, CD := t.CD, x := y, h := t.h } center) → 
    t.x^2 ≤ y^2) → 
  t.x^2 = 3443.75 := by
  sorry

end NUMINAMATH_CALUDE_smallest_x_squared_l832_83226


namespace NUMINAMATH_CALUDE_poem_lines_proof_l832_83207

/-- The number of lines added to the poem each month -/
def lines_per_month : ℕ := 3

/-- The number of months after which the poem will have 90 lines -/
def months : ℕ := 22

/-- The total number of lines in the poem after 22 months -/
def total_lines : ℕ := 90

/-- The current number of lines in the poem -/
def current_lines : ℕ := total_lines - (lines_per_month * months)

theorem poem_lines_proof : current_lines = 24 := by
  sorry

end NUMINAMATH_CALUDE_poem_lines_proof_l832_83207


namespace NUMINAMATH_CALUDE_fan_work_time_theorem_l832_83234

/-- Represents the fan's properties and operation --/
structure Fan where
  airflow_rate : ℝ  -- liters per second
  work_time : ℝ     -- minutes per day
  total_airflow : ℝ -- liters per week

/-- Theorem stating the relationship between fan operation and total airflow --/
theorem fan_work_time_theorem (f : Fan) (h1 : f.airflow_rate = 10) 
  (h2 : f.total_airflow = 42000) : 
  f.work_time = 10 ↔ f.total_airflow = 7 * f.work_time * 60 * f.airflow_rate := by
  sorry

#check fan_work_time_theorem

end NUMINAMATH_CALUDE_fan_work_time_theorem_l832_83234


namespace NUMINAMATH_CALUDE_modular_inverse_5_mod_23_l832_83243

theorem modular_inverse_5_mod_23 : ∃ (a : ℤ), 5 * a ≡ 1 [ZMOD 23] ∧ a = 14 := by
  sorry

end NUMINAMATH_CALUDE_modular_inverse_5_mod_23_l832_83243


namespace NUMINAMATH_CALUDE_three_plus_three_cubed_l832_83244

theorem three_plus_three_cubed : 3 + 3^3 = 30 := by
  sorry

end NUMINAMATH_CALUDE_three_plus_three_cubed_l832_83244


namespace NUMINAMATH_CALUDE_fixed_monthly_costs_l832_83236

/-- The fixed monthly costs for a computer manufacturer producing electronic components --/
theorem fixed_monthly_costs 
  (production_cost : ℝ) 
  (shipping_cost : ℝ) 
  (units_sold : ℕ) 
  (selling_price : ℝ) 
  (h1 : production_cost = 80)
  (h2 : shipping_cost = 3)
  (h3 : units_sold = 150)
  (h4 : selling_price = 191.67)
  (h5 : selling_price * (units_sold : ℝ) = (production_cost + shipping_cost) * (units_sold : ℝ) + fixed_costs) :
  fixed_costs = 16300.50 := by
  sorry

#check fixed_monthly_costs

end NUMINAMATH_CALUDE_fixed_monthly_costs_l832_83236


namespace NUMINAMATH_CALUDE_tan_4050_undefined_l832_83271

theorem tan_4050_undefined :
  ∀ x : ℝ, Real.tan (4050 * π / 180) = x → False :=
by
  sorry

end NUMINAMATH_CALUDE_tan_4050_undefined_l832_83271


namespace NUMINAMATH_CALUDE_john_excess_money_l832_83247

def earnings_day1 : ℚ := 20
def earnings_day2 : ℚ := 18
def earnings_day3 : ℚ := earnings_day2 / 2
def earnings_day4 : ℚ := earnings_day3 + (earnings_day3 * (25 / 100))
def earnings_day5 : ℚ := earnings_day4 + (earnings_day3 * (25 / 100))
def earnings_day6 : ℚ := earnings_day5 + (earnings_day5 * (15 / 100))
def earnings_day7 : ℚ := earnings_day6 - 10

def daily_increase : ℚ := 1
def pogo_stick_cost : ℚ := 60

def total_earnings : ℚ := 
  earnings_day1 + earnings_day2 + earnings_day3 + earnings_day4 + earnings_day5 + 
  earnings_day6 + earnings_day7 + 
  (earnings_day6 + daily_increase) + 
  (earnings_day6 + 2 * daily_increase) + 
  (earnings_day6 + 3 * daily_increase) + 
  (earnings_day6 + 4 * daily_increase) + 
  (earnings_day6 + 5 * daily_increase) + 
  (earnings_day6 + 6 * daily_increase) + 
  (earnings_day6 + 7 * daily_increase)

theorem john_excess_money : total_earnings - pogo_stick_cost = 170 := by
  sorry

end NUMINAMATH_CALUDE_john_excess_money_l832_83247


namespace NUMINAMATH_CALUDE_dodecahedron_face_centers_form_icosahedron_l832_83200

/-- A regular dodecahedron -/
structure RegularDodecahedron where
  -- Add necessary properties

/-- A regular icosahedron -/
structure RegularIcosahedron where
  -- Add necessary properties

/-- Function that connects centers of faces of a regular dodecahedron -/
def connectFaceCenters (d : RegularDodecahedron) : RegularIcosahedron :=
  sorry

/-- Theorem stating that connecting face centers of a regular dodecahedron results in a regular icosahedron -/
theorem dodecahedron_face_centers_form_icosahedron (d : RegularDodecahedron) :
  ∃ (i : RegularIcosahedron), connectFaceCenters d = i :=
sorry

end NUMINAMATH_CALUDE_dodecahedron_face_centers_form_icosahedron_l832_83200


namespace NUMINAMATH_CALUDE_k_value_l832_83225

theorem k_value (k : ℝ) (h1 : k ≠ 0) :
  (∀ x : ℝ, (x^2 - k) * (x + k) = x^3 + k * (x^2 - x - 6)) → k = 6 := by
  sorry

end NUMINAMATH_CALUDE_k_value_l832_83225


namespace NUMINAMATH_CALUDE_chocolate_bars_left_l832_83297

theorem chocolate_bars_left (initial_bars : ℕ) (people : ℕ) (given_to_mother : ℕ) (eaten : ℕ) : 
  initial_bars = 20 →
  people = 5 →
  given_to_mother = 3 →
  eaten = 2 →
  (initial_bars / people / 2 * people) - given_to_mother - eaten = 5 :=
by sorry

end NUMINAMATH_CALUDE_chocolate_bars_left_l832_83297


namespace NUMINAMATH_CALUDE_simplify_fraction_l832_83290

theorem simplify_fraction (x y : ℚ) (hx : x = 3) (hy : y = 2) :
  12 * x * y^3 / (9 * x^2 * y^2) = 8 / 9 := by
  sorry

end NUMINAMATH_CALUDE_simplify_fraction_l832_83290


namespace NUMINAMATH_CALUDE_f_of_2_equals_2_l832_83262

-- Define the function f
def f (x : ℝ) : ℝ := x^2 - x

-- Theorem statement
theorem f_of_2_equals_2 : f 2 = 2 := by
  sorry

end NUMINAMATH_CALUDE_f_of_2_equals_2_l832_83262


namespace NUMINAMATH_CALUDE_fraction_subtraction_l832_83292

theorem fraction_subtraction : (18 : ℚ) / 42 - 2 / 9 = 13 / 63 := by
  sorry

end NUMINAMATH_CALUDE_fraction_subtraction_l832_83292


namespace NUMINAMATH_CALUDE_correct_articles_for_categories_l832_83205

-- Define a type for grammatical articles
inductive Article
  | Indefinite -- represents "a/an"
  | Definite   -- represents "the"
  | None       -- represents no article (used for plural nouns)

-- Define a function to determine the correct article for a category
def correctArticle (isFirstCategory : Bool) (isPlural : Bool) : Article :=
  if isFirstCategory then
    Article.Indefinite
  else if isPlural then
    Article.None
  else
    Article.Definite

-- Theorem statement
theorem correct_articles_for_categories :
  ∀ (isFirstCategory : Bool) (isPlural : Bool),
    (isFirstCategory ∧ ¬isPlural) →
    (¬isFirstCategory ∧ isPlural) →
    (correctArticle isFirstCategory isPlural = Article.Indefinite ∧
     correctArticle (¬isFirstCategory) isPlural = Article.None) :=
by
  sorry


end NUMINAMATH_CALUDE_correct_articles_for_categories_l832_83205


namespace NUMINAMATH_CALUDE_exponent_division_l832_83239

theorem exponent_division (a : ℝ) (h : a ≠ 0) : a^5 / a = a^4 := by
  sorry

end NUMINAMATH_CALUDE_exponent_division_l832_83239


namespace NUMINAMATH_CALUDE_power_function_through_point_l832_83228

theorem power_function_through_point (α : ℝ) : 
  (∀ x : ℝ, x > 0 → (fun x => x^α) x = x^α) → 
  (2 : ℝ)^α = 4 → 
  α = 2 := by sorry

end NUMINAMATH_CALUDE_power_function_through_point_l832_83228


namespace NUMINAMATH_CALUDE_prob_at_least_one_in_three_games_l832_83284

/-- The probability of revealing a golden flower when smashing a single egg -/
def p : ℚ := 1/2

/-- The number of eggs smashed in each game -/
def n : ℕ := 3

/-- The number of games played -/
def games : ℕ := 3

/-- The probability of revealing at least one golden flower in a single game -/
def prob_at_least_one_in_game : ℚ := 1 - (1 - p)^n

/-- Theorem: The probability of revealing at least one golden flower in three games -/
theorem prob_at_least_one_in_three_games :
  (1 : ℚ) - (1 - prob_at_least_one_in_game)^games = 511/512 := by
  sorry

end NUMINAMATH_CALUDE_prob_at_least_one_in_three_games_l832_83284


namespace NUMINAMATH_CALUDE_correlation_identification_l832_83289

-- Define the relationships
def age_wealth_relation : Type := Unit
def point_coordinates_relation : Type := Unit
def apple_climate_relation : Type := Unit
def tree_diameter_height_relation : Type := Unit

-- Define the concept of correlation
def has_correlation (relation : Type) : Prop := sorry

-- Define the concept of deterministic relationship
def is_deterministic (relation : Type) : Prop := sorry

-- Theorem statement
theorem correlation_identification :
  (has_correlation age_wealth_relation) ∧
  (has_correlation apple_climate_relation) ∧
  (has_correlation tree_diameter_height_relation) ∧
  (is_deterministic point_coordinates_relation) ∧
  (¬ has_correlation point_coordinates_relation) := by sorry

end NUMINAMATH_CALUDE_correlation_identification_l832_83289


namespace NUMINAMATH_CALUDE_overall_average_calculation_l832_83201

theorem overall_average_calculation (math_score history_score third_score : ℚ) 
  (h1 : math_score = 74/100)
  (h2 : history_score = 81/100)
  (h3 : third_score = 70/100) :
  (math_score + history_score + third_score) / 3 = 75/100 := by
  sorry

end NUMINAMATH_CALUDE_overall_average_calculation_l832_83201


namespace NUMINAMATH_CALUDE_negative_two_x_times_three_y_l832_83291

theorem negative_two_x_times_three_y (x y : ℝ) : -2 * x * 3 * y = -6 * x * y := by
  sorry

end NUMINAMATH_CALUDE_negative_two_x_times_three_y_l832_83291


namespace NUMINAMATH_CALUDE_geometric_sequence_sum_l832_83217

def is_geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = a n * r

theorem geometric_sequence_sum (a : ℕ → ℝ) :
  is_geometric_sequence a →
  (∀ n, a n > 0) →
  a 2 * a 4 + 2 * a 3 * a 5 + a 4 * a 6 = 25 →
  a 3 + a 5 = 5 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_sum_l832_83217


namespace NUMINAMATH_CALUDE_abc_sum_mod_seven_l832_83215

theorem abc_sum_mod_seven (a b c : ℤ) : 
  a ∈ ({1, 2, 3, 4, 5, 6} : Set ℤ) →
  b ∈ ({1, 2, 3, 4, 5, 6} : Set ℤ) →
  c ∈ ({1, 2, 3, 4, 5, 6} : Set ℤ) →
  (a * b * c) % 7 = 1 →
  (2 * c) % 7 = 5 →
  (3 * b) % 7 = (4 + b) % 7 →
  (a + b + c) % 7 = 6 := by
sorry

end NUMINAMATH_CALUDE_abc_sum_mod_seven_l832_83215


namespace NUMINAMATH_CALUDE_sandwich_combinations_theorem_l832_83227

def num_meat_types : ℕ := 8
def num_cheese_types : ℕ := 7

def num_meat_combinations : ℕ := (num_meat_types * (num_meat_types - 1)) / 2
def num_cheese_combinations : ℕ := num_cheese_types

def total_sandwich_combinations : ℕ := num_meat_combinations * num_cheese_combinations

theorem sandwich_combinations_theorem : total_sandwich_combinations = 196 := by
  sorry

end NUMINAMATH_CALUDE_sandwich_combinations_theorem_l832_83227


namespace NUMINAMATH_CALUDE_power_of_power_l832_83212

theorem power_of_power : (2^2)^3 = 64 := by
  sorry

end NUMINAMATH_CALUDE_power_of_power_l832_83212


namespace NUMINAMATH_CALUDE_sports_club_players_l832_83250

/-- The number of players in a sports club with three games: kabaddi, kho kho, and badminton -/
theorem sports_club_players (kabaddi kho_kho_only badminton both_kabaddi_kho_kho both_kabaddi_badminton both_kho_kho_badminton all_three : ℕ) 
  (h1 : kabaddi = 20)
  (h2 : kho_kho_only = 50)
  (h3 : badminton = 25)
  (h4 : both_kabaddi_kho_kho = 15)
  (h5 : both_kabaddi_badminton = 10)
  (h6 : both_kho_kho_badminton = 5)
  (h7 : all_three = 3) :
  kabaddi + kho_kho_only + badminton - both_kabaddi_kho_kho - both_kabaddi_badminton - both_kho_kho_badminton + all_three = 68 := by
  sorry


end NUMINAMATH_CALUDE_sports_club_players_l832_83250


namespace NUMINAMATH_CALUDE_union_of_A_and_B_l832_83202

-- Define sets A and B
def A (a : ℝ) : Set ℝ := {|a + 1|, 3, 5}
def B (a : ℝ) : Set ℝ := {2*a + 1, a^2 + 2*a, a^2 + 2*a - 1}

-- Theorem statement
theorem union_of_A_and_B :
  ∃ a : ℝ, (A a ∩ B a = {2, 3}) → (A a ∪ B a = {-5, 2, 3, 5}) :=
by
  sorry

end NUMINAMATH_CALUDE_union_of_A_and_B_l832_83202


namespace NUMINAMATH_CALUDE_children_left_l832_83281

theorem children_left (total_guests : ℕ) (men : ℕ) (stayed : ℕ) :
  total_guests = 50 ∧ 
  men = 15 ∧ 
  stayed = 43 →
  (total_guests / 2 : ℕ) + men + ((total_guests - (total_guests / 2 + men)) - 
    (total_guests - stayed - men / 5)) = 4 := by
  sorry

end NUMINAMATH_CALUDE_children_left_l832_83281


namespace NUMINAMATH_CALUDE_symmetric_polynomial_n_l832_83275

/-- A polynomial p(x) is symmetric about x = m if p(m + k) = p(m - k) for all real k -/
def is_symmetric_about (p : ℝ → ℝ) (m : ℝ) : Prop :=
  ∀ k, p (m + k) = p (m - k)

/-- The polynomial p(x) = x^2 + 2nx + 3 -/
def p (n : ℝ) (x : ℝ) : ℝ := x^2 + 2*n*x + 3

theorem symmetric_polynomial_n (n : ℝ) :
  is_symmetric_about (p n) 5 → n = -5 := by
  sorry

end NUMINAMATH_CALUDE_symmetric_polynomial_n_l832_83275


namespace NUMINAMATH_CALUDE_license_plate_theorem_l832_83211

/-- The number of possible letters in the alphabet -/
def num_letters : ℕ := 26

/-- The number of possible digits -/
def num_digits : ℕ := 10

/-- The length of the license plate -/
def plate_length : ℕ := 5

/-- The number of letters at the start of the plate -/
def num_start_letters : ℕ := 2

/-- The number of digits at the end of the plate -/
def num_end_digits : ℕ := 3

/-- The number of ways to design a license plate with the given conditions -/
def license_plate_designs : ℕ :=
  num_letters * num_digits * (num_digits - 1)

theorem license_plate_theorem :
  license_plate_designs = 2340 :=
by sorry

end NUMINAMATH_CALUDE_license_plate_theorem_l832_83211


namespace NUMINAMATH_CALUDE_triangle_tan_b_l832_83219

/-- Given a triangle ABC with sides a, b, c opposite angles A, B, C respectively -/
theorem triangle_tan_b (a b c : ℝ) (A B C : ℝ) :
  /- a², b², c² form an arithmetic sequence -/
  (a^2 + c^2 = 2*b^2) →
  /- Area of triangle ABC is b²/3 -/
  (1/2 * a * c * Real.sin B = b^2/3) →
  /- Law of cosines -/
  (b^2 = a^2 + c^2 - 2*a*c*Real.cos B) →
  /- Then tan B = 4/3 -/
  Real.tan B = 4/3 := by
sorry

end NUMINAMATH_CALUDE_triangle_tan_b_l832_83219


namespace NUMINAMATH_CALUDE_ln2_greatest_l832_83293

-- Define the natural logarithm function
noncomputable def ln (x : ℝ) : ℝ := Real.log x

-- State the theorem
theorem ln2_greatest (h1 : ∀ x y : ℝ, x < y → ln x < ln y) (h2 : (2 : ℝ) < Real.exp 1) :
  ln 2 > (ln 2)^2 ∧ ln 2 > ln (ln 2) ∧ ln 2 > ln (Real.sqrt 2) := by
  sorry


end NUMINAMATH_CALUDE_ln2_greatest_l832_83293


namespace NUMINAMATH_CALUDE_amy_biking_week_l832_83221

def total_miles_biked (monday_miles : ℕ) : ℕ :=
  let tuesday_miles := 2 * monday_miles - 3
  let wednesday_miles := tuesday_miles + 2
  let thursday_miles := wednesday_miles + 2
  let friday_miles := thursday_miles + 2
  let saturday_miles := friday_miles + 2
  let sunday_miles := saturday_miles + 2
  monday_miles + tuesday_miles + wednesday_miles + thursday_miles + friday_miles + saturday_miles + sunday_miles

theorem amy_biking_week (monday_miles : ℕ) (h : monday_miles = 12) : 
  total_miles_biked monday_miles = 168 := by
  sorry

end NUMINAMATH_CALUDE_amy_biking_week_l832_83221


namespace NUMINAMATH_CALUDE_revenue_change_l832_83264

theorem revenue_change
  (T : ℝ) -- original tax rate (as a percentage)
  (C : ℝ) -- original consumption
  (h1 : T > 0)
  (h2 : C > 0) :
  let new_tax_rate := T * (1 - 0.16)
  let new_consumption := C * (1 + 0.15)
  let original_revenue := (T / 100) * C
  let new_revenue := (new_tax_rate / 100) * new_consumption
  (new_revenue - original_revenue) / original_revenue = -0.034 := by
sorry

end NUMINAMATH_CALUDE_revenue_change_l832_83264


namespace NUMINAMATH_CALUDE_sarah_scored_135_l832_83296

def sarahs_score (greg_score sarah_score : ℕ) : Prop :=
  greg_score + 50 = sarah_score ∧ (greg_score + sarah_score) / 2 = 110

theorem sarah_scored_135 :
  ∃ (greg_score : ℕ), sarahs_score greg_score 135 :=
by sorry

end NUMINAMATH_CALUDE_sarah_scored_135_l832_83296


namespace NUMINAMATH_CALUDE_sum_divided_by_ten_l832_83276

theorem sum_divided_by_ten : (10 + 20 + 30 + 40) / 10 = 10 := by
  sorry

end NUMINAMATH_CALUDE_sum_divided_by_ten_l832_83276


namespace NUMINAMATH_CALUDE_no_integer_satisfies_conditions_l832_83208

theorem no_integer_satisfies_conditions : ¬ ∃ m : ℤ, m % 9 = 2 ∧ m % 6 = 1 := by
  sorry

end NUMINAMATH_CALUDE_no_integer_satisfies_conditions_l832_83208


namespace NUMINAMATH_CALUDE_no_real_b_for_single_solution_l832_83224

theorem no_real_b_for_single_solution :
  ¬ ∃ b : ℝ, ∃! x : ℝ, |x^2 + 3*b*x + 5*b| ≤ 5 :=
by sorry

end NUMINAMATH_CALUDE_no_real_b_for_single_solution_l832_83224


namespace NUMINAMATH_CALUDE_sampling_probability_l832_83229

theorem sampling_probability (m : ℕ) (h_m : m ≥ 2017) :
  let systematic_prob := (3 : ℚ) / 2017
  let stratified_prob := (3 : ℚ) / 2017
  systematic_prob = stratified_prob := by sorry

end NUMINAMATH_CALUDE_sampling_probability_l832_83229


namespace NUMINAMATH_CALUDE_total_cost_calculation_l832_83216

def total_cost (total_bricks : ℕ) (discount1_percent : ℚ) (discount2_percent : ℚ) 
                (full_price : ℚ) (discount1_fraction : ℚ) (discount2_fraction : ℚ)
                (full_price_fraction : ℚ) (additional_cost : ℚ) : ℚ :=
  let discounted_price1 := full_price * (1 - discount1_percent)
  let discounted_price2 := full_price * (1 - discount2_percent)
  let cost1 := (total_bricks : ℚ) * discount1_fraction * discounted_price1
  let cost2 := (total_bricks : ℚ) * discount2_fraction * discounted_price2
  let cost3 := (total_bricks : ℚ) * full_price_fraction * full_price
  cost1 + cost2 + cost3 + additional_cost

theorem total_cost_calculation :
  total_cost 1000 (1/2) (1/5) (1/2) (3/10) (2/5) (3/10) 200 = 585 := by
  sorry

end NUMINAMATH_CALUDE_total_cost_calculation_l832_83216


namespace NUMINAMATH_CALUDE_length_ratio_theorem_l832_83209

/-- Represents a three-stage rocket with cylindrical stages -/
structure ThreeStageRocket where
  l₁ : ℝ  -- Length of the first stage
  l₂ : ℝ  -- Length of the second stage
  l₃ : ℝ  -- Length of the third stage

/-- The conditions for the three-stage rocket -/
def RocketConditions (r : ThreeStageRocket) : Prop :=
  r.l₂ = (r.l₁ + r.l₃) / 2 ∧
  r.l₂^3 = (6 / 13) * (r.l₁^3 + r.l₃^3)

/-- The theorem stating the ratio of lengths of the first and third stages -/
theorem length_ratio_theorem (r : ThreeStageRocket) (h : RocketConditions r) :
  r.l₁ / r.l₃ = 7 / 5 := by
  sorry


end NUMINAMATH_CALUDE_length_ratio_theorem_l832_83209


namespace NUMINAMATH_CALUDE_inscribed_sphere_surface_area_l832_83282

/-- The surface area of a sphere inscribed in a cube with edge length 2 is 4π. -/
theorem inscribed_sphere_surface_area (cube_edge : ℝ) (h : cube_edge = 2) :
  let sphere_radius := cube_edge / 2
  4 * Real.pi * sphere_radius ^ 2 = 4 * Real.pi := by
  sorry

end NUMINAMATH_CALUDE_inscribed_sphere_surface_area_l832_83282


namespace NUMINAMATH_CALUDE_square_circle_ratio_l832_83210

theorem square_circle_ratio : 
  let square_area : ℝ := 784
  let small_circle_circumference : ℝ := 8
  let larger_radius_ratio : ℝ := 7/3

  let square_side : ℝ := Real.sqrt square_area
  let small_circle_radius : ℝ := small_circle_circumference / (2 * Real.pi)
  let large_circle_radius : ℝ := larger_radius_ratio * small_circle_radius

  square_side / large_circle_radius = 3 * Real.pi :=
by sorry

end NUMINAMATH_CALUDE_square_circle_ratio_l832_83210


namespace NUMINAMATH_CALUDE_average_speed_two_hours_l832_83299

/-- The average speed of a car over two hours, given its speeds in each hour -/
theorem average_speed_two_hours (speed1 speed2 : ℝ) : 
  speed1 = 90 → speed2 = 75 → (speed1 + speed2) / 2 = 82.5 := by
  sorry

end NUMINAMATH_CALUDE_average_speed_two_hours_l832_83299


namespace NUMINAMATH_CALUDE_alexis_skirt_time_l832_83237

/-- The time it takes Alexis to sew a skirt -/
def skirt_time : ℝ := 2

/-- The time it takes Alexis to sew a coat -/
def coat_time : ℝ := 7

/-- The total time it takes Alexis to sew 6 skirts and 4 coats -/
def total_time : ℝ := 40

theorem alexis_skirt_time :
  skirt_time = 2 ∧
  coat_time = 7 ∧
  total_time = 40 ∧
  6 * skirt_time + 4 * coat_time = total_time :=
by sorry

end NUMINAMATH_CALUDE_alexis_skirt_time_l832_83237


namespace NUMINAMATH_CALUDE_sqrt_equation_solution_l832_83214

theorem sqrt_equation_solution (x : ℝ) :
  x ≥ 1 →
  (Real.sqrt (x + 3 - 4 * Real.sqrt (x - 1)) + Real.sqrt (x + 8 - 6 * Real.sqrt (x - 1)) = 1) ↔
  (5 ≤ x ∧ x ≤ 10) :=
by sorry

end NUMINAMATH_CALUDE_sqrt_equation_solution_l832_83214


namespace NUMINAMATH_CALUDE_units_digit_difference_largest_smallest_l832_83242

theorem units_digit_difference_largest_smallest (a b c d e : ℕ) :
  1 ≤ a ∧ a < b ∧ b < c ∧ c < d ∧ d < e ∧ e ≤ 9 →
  (100 * e + 10 * d + c) - (100 * a + 10 * b + c) ≡ 0 [MOD 10] :=
by sorry

end NUMINAMATH_CALUDE_units_digit_difference_largest_smallest_l832_83242


namespace NUMINAMATH_CALUDE_power_function_through_point_l832_83277

theorem power_function_through_point (a : ℝ) : (fun x : ℝ => x^a) 2 = 16 → a = 4 := by
  sorry

end NUMINAMATH_CALUDE_power_function_through_point_l832_83277


namespace NUMINAMATH_CALUDE_bears_in_stock_before_shipment_l832_83263

/-- The number of bears in a new shipment -/
def new_shipment : ℕ := 10

/-- The number of bears on each shelf -/
def bears_per_shelf : ℕ := 9

/-- The number of shelves used -/
def shelves_used : ℕ := 3

/-- The number of bears in stock before the new shipment -/
def bears_before_shipment : ℕ := shelves_used * bears_per_shelf - new_shipment

theorem bears_in_stock_before_shipment :
  bears_before_shipment = 17 := by
  sorry

end NUMINAMATH_CALUDE_bears_in_stock_before_shipment_l832_83263


namespace NUMINAMATH_CALUDE_election_votes_l832_83249

theorem election_votes (total_votes : ℕ) (invalid_percent : ℚ) (excess_percent : ℚ) : 
  total_votes = 5720 →
  invalid_percent = 1/5 →
  excess_percent = 3/20 →
  ∃ (a_votes b_votes : ℕ),
    (a_votes : ℚ) + b_votes = total_votes * (1 - invalid_percent) ∧
    (a_votes : ℚ) = b_votes + total_votes * excess_percent ∧
    b_votes = 1859 := by
sorry

end NUMINAMATH_CALUDE_election_votes_l832_83249


namespace NUMINAMATH_CALUDE_systematicSamplingExample_l832_83240

/-- Calculates the number of groups for systematic sampling -/
def systematicSamplingGroups (totalStudents : ℕ) (sampleSize : ℕ) : ℕ :=
  totalStudents / sampleSize

/-- Theorem stating that for 600 students and a sample size of 20, 
    the number of groups for systematic sampling is 30 -/
theorem systematicSamplingExample : 
  systematicSamplingGroups 600 20 = 30 := by
  sorry

end NUMINAMATH_CALUDE_systematicSamplingExample_l832_83240


namespace NUMINAMATH_CALUDE_stones_for_hall_l832_83285

/-- Calculates the number of stones required to pave a rectangular hall --/
def stones_required (hall_length hall_width stone_length stone_width : ℚ) : ℕ :=
  let hall_area := hall_length * hall_width * 100
  let stone_area := stone_length * stone_width
  (hall_area / stone_area).ceil.toNat

/-- Theorem stating that 4,500 stones are required to pave the given hall --/
theorem stones_for_hall : stones_required 72 30 6 8 = 4500 := by
  sorry

end NUMINAMATH_CALUDE_stones_for_hall_l832_83285


namespace NUMINAMATH_CALUDE_inequality_and_equality_condition_l832_83204

theorem inequality_and_equality_condition (x y z : ℝ) 
  (pos_x : x > 0) (pos_y : y > 0) (pos_z : z > 0)
  (h : x^2 + y^2 + z^2 + 3 = 2*(x*y + y*z + z*x)) :
  Real.sqrt (x*y) + Real.sqrt (y*z) + Real.sqrt (z*x) ≥ 3 ∧
  (Real.sqrt (x*y) + Real.sqrt (y*z) + Real.sqrt (z*x) = 3 ↔ x = 1 ∧ y = 1 ∧ z = 1) :=
by sorry

end NUMINAMATH_CALUDE_inequality_and_equality_condition_l832_83204


namespace NUMINAMATH_CALUDE_isosceles_triangle_l832_83267

theorem isosceles_triangle (A B C : Real) (h1 : A + B + C = π) (h2 : 2 * Real.sin A * Real.cos B = Real.sin C) : A = B := by
  sorry

end NUMINAMATH_CALUDE_isosceles_triangle_l832_83267


namespace NUMINAMATH_CALUDE_intersection_sum_l832_83295

-- Define the two equations
def f (x : ℝ) : ℝ := x^3 - 3*x + 1
def g (x y : ℝ) : Prop := x + 3*y = 3

-- Define the intersection points
def intersection_points : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | f p.1 = p.2 ∧ g p.1 p.2}

-- Theorem statement
theorem intersection_sum :
  ∃ (p₁ p₂ p₃ : ℝ × ℝ),
    p₁ ∈ intersection_points ∧
    p₂ ∈ intersection_points ∧
    p₃ ∈ intersection_points ∧
    p₁ ≠ p₂ ∧ p₁ ≠ p₃ ∧ p₂ ≠ p₃ ∧
    p₁.1 + p₂.1 + p₃.1 = 0 ∧
    p₁.2 + p₂.2 + p₃.2 = 3 :=
sorry

end NUMINAMATH_CALUDE_intersection_sum_l832_83295


namespace NUMINAMATH_CALUDE_max_white_pieces_l832_83288

/-- Represents the color of a piece -/
inductive Color
| Black
| White

/-- Represents the circle of pieces -/
def Circle := List Color

/-- The initial configuration of the circle -/
def initial_circle : Circle :=
  [Color.Black, Color.Black, Color.Black, Color.Black, Color.White]

/-- Applies the rules to place new pieces and remove old ones -/
def apply_rules (c : Circle) : Circle :=
  sorry

/-- Counts the number of white pieces in the circle -/
def count_white (c : Circle) : Nat :=
  sorry

/-- Theorem stating that the maximum number of white pieces is 3 -/
theorem max_white_pieces (c : Circle) : 
  count_white (apply_rules c) ≤ 3 :=
sorry

end NUMINAMATH_CALUDE_max_white_pieces_l832_83288


namespace NUMINAMATH_CALUDE_largest_among_abcd_l832_83206

theorem largest_among_abcd (a b c d : ℝ) 
  (h : a - 1 = b + 2 ∧ a - 1 = c - 3 ∧ a - 1 = d + 4) : 
  c ≥ a ∧ c ≥ b ∧ c ≥ d := by
sorry

end NUMINAMATH_CALUDE_largest_among_abcd_l832_83206


namespace NUMINAMATH_CALUDE_quadratic_roots_sum_inverse_squares_l832_83254

theorem quadratic_roots_sum_inverse_squares (a b c k : ℝ) (kr ks : ℝ) 
  (h1 : a ≠ 0) (h2 : b ≠ 0) 
  (h3 : a * kr^2 + k * c * kr + b = 0) 
  (h4 : a * ks^2 + k * c * ks + b = 0) 
  (h5 : kr ≠ 0) (h6 : ks ≠ 0) : 
  1 / kr^2 + 1 / ks^2 = (k^2 * c^2 - 2 * a * b) / b^2 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_roots_sum_inverse_squares_l832_83254


namespace NUMINAMATH_CALUDE_sequence_a_formula_l832_83241

def sequence_a (n : ℕ) : ℝ := sorry

def S (n : ℕ) : ℝ := sorry

axiom S_2 : S 2 = 4

axiom a_recursive (n : ℕ) : n ≥ 1 → sequence_a (n + 1) = 2 * S n + 1

theorem sequence_a_formula (n : ℕ) : n ≥ 1 → sequence_a n = 3^(n - 1) := by sorry

end NUMINAMATH_CALUDE_sequence_a_formula_l832_83241


namespace NUMINAMATH_CALUDE_corn_acreage_l832_83280

theorem corn_acreage (total_land : ℕ) (beans_ratio wheat_ratio corn_ratio : ℕ) 
  (h1 : total_land = 1034)
  (h2 : beans_ratio = 5)
  (h3 : wheat_ratio = 2)
  (h4 : corn_ratio = 4) :
  (total_land * corn_ratio) / (beans_ratio + wheat_ratio + corn_ratio) = 376 := by
  sorry

end NUMINAMATH_CALUDE_corn_acreage_l832_83280


namespace NUMINAMATH_CALUDE_units_digit_of_33_power_l832_83283

theorem units_digit_of_33_power (n : ℕ) : (33 ^ (33 * (22 ^ 22))) % 10 = 1 := by
  sorry

end NUMINAMATH_CALUDE_units_digit_of_33_power_l832_83283

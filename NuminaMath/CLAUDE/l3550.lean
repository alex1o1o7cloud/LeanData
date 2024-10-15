import Mathlib

namespace NUMINAMATH_CALUDE_function_properties_l3550_355046

/-- The function f(x) = 2|x-1| - a -/
def f (a : ℝ) (x : ℝ) : ℝ := 2 * abs (x - 1) - a

/-- The function g(x) = -|x+m| -/
def g (m : ℝ) (x : ℝ) : ℝ := -(abs (x + m))

/-- The statement that g(x) > -1 has exactly one integer solution, which is -3 -/
def has_unique_integer_solution (m : ℝ) : Prop :=
  ∃! (n : ℤ), g m (n : ℝ) > -1 ∧ n = -3

theorem function_properties (a m : ℝ) 
  (h_unique : has_unique_integer_solution m) :
  m = 3 ∧ (∀ x, f a x > g m x) → a < 4 := by sorry

end NUMINAMATH_CALUDE_function_properties_l3550_355046


namespace NUMINAMATH_CALUDE_award_distribution_probability_l3550_355054

def num_classes : ℕ := 4
def num_awards : ℕ := 8

def distribute_awards (n : ℕ) (k : ℕ) : ℕ := Nat.choose (n + k - 1) (k - 1)

theorem award_distribution_probability :
  let total_distributions := distribute_awards (num_awards - num_classes) num_classes
  let favorable_distributions := distribute_awards ((num_awards - num_classes) - 1) (num_classes - 1)
  (favorable_distributions : ℚ) / total_distributions = 2 / 7 := by
  sorry

end NUMINAMATH_CALUDE_award_distribution_probability_l3550_355054


namespace NUMINAMATH_CALUDE_power_sum_inequality_l3550_355026

theorem power_sum_inequality (a b c : ℝ) (n : ℕ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) 
  (h_pyth : a^2 + b^2 = c^2) (hn : n > 2) : a^n + b^n < c^n := by
  sorry

end NUMINAMATH_CALUDE_power_sum_inequality_l3550_355026


namespace NUMINAMATH_CALUDE_problem_statement_l3550_355094

theorem problem_statement : 4 * Real.sqrt (1/2) + 3 * Real.sqrt (1/3) - Real.sqrt 8 = Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_problem_statement_l3550_355094


namespace NUMINAMATH_CALUDE_cube_product_theorem_l3550_355041

theorem cube_product_theorem : 
  let f (n : ℕ) := (n^3 - 1) / (n^3 + 1)
  (f 3) * (f 4) * (f 5) * (f 6) * (f 7) * (f 8) = 73 / 256 := by
  sorry

end NUMINAMATH_CALUDE_cube_product_theorem_l3550_355041


namespace NUMINAMATH_CALUDE_line_parallel_perpendicular_implies_planes_perpendicular_l3550_355050

-- Define the types for lines and planes
variable (L : Type) [LinearOrder L]
variable (P : Type)

-- Define the relationships
variable (parallel : L → P → Prop)
variable (perpendicular : L → P → Prop)
variable (plane_perpendicular : P → P → Prop)

-- State the theorem
theorem line_parallel_perpendicular_implies_planes_perpendicular
  (ι : L) (α β : P) (h1 : parallel ι α) (h2 : perpendicular ι β) :
  plane_perpendicular α β :=
sorry

end NUMINAMATH_CALUDE_line_parallel_perpendicular_implies_planes_perpendicular_l3550_355050


namespace NUMINAMATH_CALUDE_translate_upward_5_units_l3550_355069

/-- Represents a linear function of the form y = mx + b -/
structure LinearFunction where
  m : ℝ  -- slope
  b : ℝ  -- y-intercept

/-- Translates a linear function vertically by a given amount -/
def translateVertically (f : LinearFunction) (δ : ℝ) : LinearFunction :=
  { m := f.m, b := f.b + δ }

/-- The theorem to prove -/
theorem translate_upward_5_units :
  let f : LinearFunction := { m := 2, b := -3 }
  let g : LinearFunction := translateVertically f 5
  g = { m := 2, b := 2 } := by sorry

end NUMINAMATH_CALUDE_translate_upward_5_units_l3550_355069


namespace NUMINAMATH_CALUDE_total_soccer_balls_donated_l3550_355076

-- Define the given conditions
def soccer_balls_per_class : ℕ := 5
def number_of_schools : ℕ := 2
def elementary_classes_per_school : ℕ := 4
def middle_classes_per_school : ℕ := 5

-- Define the theorem
theorem total_soccer_balls_donated : 
  soccer_balls_per_class * number_of_schools * (elementary_classes_per_school + middle_classes_per_school) = 90 := by
  sorry


end NUMINAMATH_CALUDE_total_soccer_balls_donated_l3550_355076


namespace NUMINAMATH_CALUDE_tangent_line_circle_m_value_l3550_355059

/-- A circle in the xy-plane -/
structure Circle where
  equation : ℝ → ℝ → ℝ → Prop

/-- A line in the xy-plane -/
structure Line where
  equation : ℝ → ℝ → ℝ → Prop

/-- Predicate to check if a line is tangent to a circle -/
def IsTangent (l : Line) (c : Circle) : Prop := sorry

/-- The main theorem -/
theorem tangent_line_circle_m_value (m : ℝ) :
  let c : Circle := ⟨λ x y m => x^2 + y^2 = m⟩
  let l : Line := ⟨λ x y m => x + y + m = 0⟩
  IsTangent l c → m = 2 := by sorry

end NUMINAMATH_CALUDE_tangent_line_circle_m_value_l3550_355059


namespace NUMINAMATH_CALUDE_base8_to_base5_conversion_l3550_355020

/-- Converts a number from base 8 to base 10 -/
def base8ToBase10 (n : ℕ) : ℕ := sorry

/-- Converts a number from base 10 to base 5 -/
def base10ToBase5 (n : ℕ) : ℕ := sorry

/-- The number 427 in base 8 -/
def num_base8 : ℕ := 427

/-- The number 2104 in base 5 -/
def num_base5 : ℕ := 2104

theorem base8_to_base5_conversion :
  base10ToBase5 (base8ToBase10 num_base8) = num_base5 := by sorry

end NUMINAMATH_CALUDE_base8_to_base5_conversion_l3550_355020


namespace NUMINAMATH_CALUDE_f_inequality_range_l3550_355099

-- Define the function f
def f (x : ℝ) : ℝ := |x - 1| + |x + 1|

-- State the theorem
theorem f_inequality_range (a : ℝ) :
  (∀ x : ℝ, f x - a ≤ 0) ↔ a ≥ 4 := by
  sorry

end NUMINAMATH_CALUDE_f_inequality_range_l3550_355099


namespace NUMINAMATH_CALUDE_triangle_intersection_area_l3550_355056

/-- Given a triangle PQR with vertices P(0, 10), Q(3, 0), R(9, 0),
    and a horizontal line y=s intersecting PQ at V and PR at W,
    if the area of triangle PVW is 18, then s = 10 - 2√15. -/
theorem triangle_intersection_area (s : ℝ) : 
  let P : ℝ × ℝ := (0, 10)
  let Q : ℝ × ℝ := (3, 0)
  let R : ℝ × ℝ := (9, 0)
  let V : ℝ × ℝ := ((3/10) * (10 - s), s)
  let W : ℝ × ℝ := ((9/10) * (10 - s), s)
  let area_PVW : ℝ := (1/2) * ((W.1 - V.1) * (P.2 - V.2))
  area_PVW = 18 → s = 10 - 2 * Real.sqrt 15 :=
by sorry

end NUMINAMATH_CALUDE_triangle_intersection_area_l3550_355056


namespace NUMINAMATH_CALUDE_match_processes_count_l3550_355070

def number_of_match_processes : ℕ := 2 * Nat.choose 13 6

theorem match_processes_count :
  number_of_match_processes = 3432 :=
by sorry

end NUMINAMATH_CALUDE_match_processes_count_l3550_355070


namespace NUMINAMATH_CALUDE_jogging_duration_sum_l3550_355085

/-- The duration in minutes between 5 p.m. and 6 p.m. -/
def total_duration : ℕ := 60

/-- The probability of one friend arriving while the other is jogging -/
def meeting_probability : ℚ := 1/2

/-- Represents the duration each friend stays for jogging -/
structure JoggingDuration where
  x : ℕ
  y : ℕ
  z : ℕ
  x_pos : 0 < x
  y_pos : 0 < y
  z_pos : 0 < z
  z_not_perfect_square : ∀ (p : ℕ), Prime p → ¬(p^2 ∣ z)
  duration_eq : (x : ℚ) - y * Real.sqrt z = total_duration - total_duration * Real.sqrt 2

theorem jogging_duration_sum (d : JoggingDuration) : d.x + d.y + d.z = 92 := by
  sorry

end NUMINAMATH_CALUDE_jogging_duration_sum_l3550_355085


namespace NUMINAMATH_CALUDE_jelly_bean_match_probability_l3550_355011

/-- Represents the distribution of jelly beans for a person -/
structure JellyBeanDistribution where
  green : ℕ
  blue : ℕ
  red : ℕ

/-- Calculates the total number of jelly beans -/
def JellyBeanDistribution.total (d : JellyBeanDistribution) : ℕ :=
  d.green + d.blue + d.red

/-- Lila's jelly bean distribution -/
def lila_beans : JellyBeanDistribution :=
  { green := 1, blue := 1, red := 1 }

/-- Max's jelly bean distribution -/
def max_beans : JellyBeanDistribution :=
  { green := 2, blue := 1, red := 3 }

/-- Calculates the probability of picking a specific color -/
def pick_probability (d : JellyBeanDistribution) (color : ℕ) : ℚ :=
  color / d.total

/-- Calculates the probability of both people picking the same color -/
def match_probability (d1 d2 : JellyBeanDistribution) : ℚ :=
  pick_probability d1 d1.green * pick_probability d2 d2.green +
  pick_probability d1 d1.blue * pick_probability d2 d2.blue +
  pick_probability d1 d1.red * pick_probability d2 d2.red

theorem jelly_bean_match_probability :
  match_probability lila_beans max_beans = 1/3 := by
  sorry

end NUMINAMATH_CALUDE_jelly_bean_match_probability_l3550_355011


namespace NUMINAMATH_CALUDE_min_value_reciprocal_sum_min_value_product_l3550_355071

-- Part 1
theorem min_value_reciprocal_sum (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : 2 * x + y = 1) :
  1 / x + 1 / y ≥ 3 + 2 * Real.sqrt 2 := by sorry

-- Part 2
theorem min_value_product (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : x + 8 * y - x * y = 0) :
  x * y ≥ 32 := by sorry

end NUMINAMATH_CALUDE_min_value_reciprocal_sum_min_value_product_l3550_355071


namespace NUMINAMATH_CALUDE_hazel_fish_count_l3550_355048

theorem hazel_fish_count (total : ℕ) (father : ℕ) (hazel : ℕ) : 
  total = 94 → father = 46 → total = father + hazel → hazel = 48 := by
  sorry

end NUMINAMATH_CALUDE_hazel_fish_count_l3550_355048


namespace NUMINAMATH_CALUDE_magnitude_of_OP_l3550_355015

/-- Given vectors OA and OB, and the relation between AP and AB, prove the magnitude of OP --/
theorem magnitude_of_OP (OA OB OP : ℝ × ℝ) : 
  OA = (1, 2) → 
  OB = (-2, -1) → 
  2 * (OP - OA) = OB - OA → 
  Real.sqrt ((OP.1)^2 + (OP.2)^2) = Real.sqrt 2 / 2 := by
  sorry

end NUMINAMATH_CALUDE_magnitude_of_OP_l3550_355015


namespace NUMINAMATH_CALUDE_sum_of_digits_l3550_355095

theorem sum_of_digits (a b c d e : ℕ) : 
  (10 ≤ 10*a + b) ∧ (10*a + b ≤ 99) ∧
  (100 ≤ 100*c + 10*d + e) ∧ (100*c + 10*d + e ≤ 999) ∧
  (10*a + b + 100*c + 10*d + e = 1079) →
  a + b + c + d + e = 35 := by
sorry

end NUMINAMATH_CALUDE_sum_of_digits_l3550_355095


namespace NUMINAMATH_CALUDE_inequality_proof_l3550_355030

theorem inequality_proof (a b c : ℝ) 
  (h_pos_a : a > 0) (h_pos_b : b > 0) (h_pos_c : c > 0)
  (h_eq : a^2 + b^2 + 4*c^2 = 3) : 
  (a + b + 2*c ≤ 3) ∧ (b = 2*c → 1/a + 1/c ≥ 3) := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l3550_355030


namespace NUMINAMATH_CALUDE_square_root_625_divided_by_5_l3550_355017

theorem square_root_625_divided_by_5 : Real.sqrt 625 / 5 = 5 := by
  sorry

end NUMINAMATH_CALUDE_square_root_625_divided_by_5_l3550_355017


namespace NUMINAMATH_CALUDE_line_chart_most_appropriate_l3550_355096

/-- Represents a chart type -/
inductive ChartType
| LineChart
| BarChart
| PieChart
| ScatterPlot

/-- Represents the requirements for a temperature chart -/
structure TemperatureChartRequirements where
  showsChangeOverTime : Bool
  reflectsAmountAndChanges : Bool
  showsIncreasesAndDecreases : Bool

/-- Defines the properties of a line chart -/
def lineChartProperties : TemperatureChartRequirements :=
  { showsChangeOverTime := true
  , reflectsAmountAndChanges := true
  , showsIncreasesAndDecreases := true }

/-- Determines if a chart type is appropriate for the given requirements -/
def isAppropriateChart (c : ChartType) (r : TemperatureChartRequirements) : Bool :=
  match c with
  | ChartType.LineChart => r.showsChangeOverTime ∧ r.reflectsAmountAndChanges ∧ r.showsIncreasesAndDecreases
  | _ => false

/-- Theorem: A line chart is the most appropriate for recording temperature changes of a feverish patient -/
theorem line_chart_most_appropriate :
  isAppropriateChart ChartType.LineChart lineChartProperties = true :=
sorry

end NUMINAMATH_CALUDE_line_chart_most_appropriate_l3550_355096


namespace NUMINAMATH_CALUDE_triangulation_reconstruction_l3550_355075

/-- A convex polygon represented by its vertices -/
structure ConvexPolygon where
  vertices : List ℝ × ℝ
  is_convex : sorry

/-- A triangulation of a convex polygon -/
structure Triangulation (P : ConvexPolygon) where
  diagonals : List (ℕ × ℕ)
  is_valid : sorry

/-- The number of triangles adjacent to each vertex in a triangulation -/
def adjacentTriangles (P : ConvexPolygon) (T : Triangulation P) : List ℕ :=
  sorry

/-- Theorem stating that a triangulation can be uniquely reconstructed from adjacent triangle counts -/
theorem triangulation_reconstruction
  (P : ConvexPolygon)
  (T1 T2 : Triangulation P)
  (h : adjacentTriangles P T1 = adjacentTriangles P T2) :
  T1 = T2 :=
sorry

end NUMINAMATH_CALUDE_triangulation_reconstruction_l3550_355075


namespace NUMINAMATH_CALUDE_g_neg_two_eq_eleven_l3550_355049

/-- The function g(x) = x^2 - 2x + 3 -/
def g (x : ℝ) : ℝ := x^2 - 2*x + 3

/-- Theorem: g(-2) = 11 -/
theorem g_neg_two_eq_eleven : g (-2) = 11 := by
  sorry

end NUMINAMATH_CALUDE_g_neg_two_eq_eleven_l3550_355049


namespace NUMINAMATH_CALUDE_units_digit_of_n_l3550_355010

/-- Given two natural numbers m and n, returns true if m has units digit 9 -/
def has_units_digit_9 (m : ℕ) : Prop :=
  m % 10 = 9

/-- Given two natural numbers m and n, returns true if their product equals 31^6 -/
def product_equals_31_pow_6 (m n : ℕ) : Prop :=
  m * n = 31^6

/-- Theorem stating that if m has units digit 9 and m * n = 31^6, then n has units digit 9 -/
theorem units_digit_of_n (m n : ℕ) 
  (h1 : has_units_digit_9 m) 
  (h2 : product_equals_31_pow_6 m n) : 
  n % 10 = 9 := by
  sorry

end NUMINAMATH_CALUDE_units_digit_of_n_l3550_355010


namespace NUMINAMATH_CALUDE_fence_length_not_eighteen_l3550_355065

theorem fence_length_not_eighteen (length width : ℝ) : 
  length = 6 → width = 3 → 
  ¬(length + 2 * width = 18 ∨ 2 * length + width = 18) :=
by
  sorry

end NUMINAMATH_CALUDE_fence_length_not_eighteen_l3550_355065


namespace NUMINAMATH_CALUDE_asparagus_cost_l3550_355001

def initial_amount : ℕ := 55
def banana_pack_cost : ℕ := 4
def banana_packs : ℕ := 2
def pear_cost : ℕ := 2
def chicken_cost : ℕ := 11
def remaining_amount : ℕ := 28

theorem asparagus_cost :
  ∃ (asparagus_cost : ℕ),
    initial_amount - (banana_pack_cost * banana_packs + pear_cost + chicken_cost + asparagus_cost) = remaining_amount ∧
    asparagus_cost = 6 := by
  sorry

end NUMINAMATH_CALUDE_asparagus_cost_l3550_355001


namespace NUMINAMATH_CALUDE_f_abs_x_is_even_l3550_355022

theorem f_abs_x_is_even (f : ℝ → ℝ) : 
  let g := fun (x : ℝ) ↦ f (|x|)
  ∀ x, g (-x) = g x := by sorry

end NUMINAMATH_CALUDE_f_abs_x_is_even_l3550_355022


namespace NUMINAMATH_CALUDE_cloud_computing_analysis_l3550_355064

/-- Cloud computing market data --/
structure MarketData :=
  (year : ℕ)
  (market_scale : ℝ)

/-- Regression equation coefficients --/
structure RegressionCoefficients :=
  (b : ℝ)
  (a : ℝ)

/-- Cloud computing market analysis --/
theorem cloud_computing_analysis 
  (data : List MarketData)
  (sum_ln_y : ℝ)
  (sum_x_ln_y : ℝ)
  (initial_error_variance : ℝ → ℝ)
  (initial_probability : ℝ)
  (new_error_variance : ℝ → ℝ) :
  ∃ (coef : RegressionCoefficients) 
    (new_probability : ℝ) 
    (cost_decrease : ℝ),
  (coef.b = 0.386 ∧ coef.a = 6.108) ∧
  (new_probability = 0.9545) ∧
  (cost_decrease = 3) :=
by
  sorry

end NUMINAMATH_CALUDE_cloud_computing_analysis_l3550_355064


namespace NUMINAMATH_CALUDE_expand_product_l3550_355045

theorem expand_product (x : ℝ) : (2*x + 3) * (x + 5) = 2*x^2 + 13*x + 15 := by
  sorry

end NUMINAMATH_CALUDE_expand_product_l3550_355045


namespace NUMINAMATH_CALUDE_number_puzzle_l3550_355044

theorem number_puzzle : ∃ x : ℝ, 3 * (2 * x + 9) = 81 := by
  sorry

end NUMINAMATH_CALUDE_number_puzzle_l3550_355044


namespace NUMINAMATH_CALUDE_dinner_bill_ratio_l3550_355019

/-- Given a dinner bill split between three people, this theorem proves
    the ratio of two people's payments given certain conditions. -/
theorem dinner_bill_ratio (total bill : ℚ) (daniel clarence matthew : ℚ) :
  bill = 20.20 →
  daniel = 6.06 →
  daniel = (1 / 2) * clarence →
  bill = daniel + clarence + matthew →
  clarence / matthew = 6 / 1 := by
  sorry

end NUMINAMATH_CALUDE_dinner_bill_ratio_l3550_355019


namespace NUMINAMATH_CALUDE_square_sum_reciprocal_l3550_355047

theorem square_sum_reciprocal (x : ℝ) (h : x + 1/x = 3) : x^2 + 1/x^2 = 7 := by
  sorry

end NUMINAMATH_CALUDE_square_sum_reciprocal_l3550_355047


namespace NUMINAMATH_CALUDE_at_least_three_lines_intersect_l3550_355032

/-- A line that divides a square into two quadrilaterals -/
structure DividingLine where
  divides_square : Bool
  area_ratio : Rat
  intersects_point : Point

/-- A square with dividing lines -/
structure DividedSquare where
  side_length : ℝ
  dividing_lines : List DividingLine

/-- The theorem statement -/
theorem at_least_three_lines_intersect (square : DividedSquare) :
  square.side_length > 0 ∧
  square.dividing_lines.length = 9 ∧
  (∀ l ∈ square.dividing_lines, l.divides_square ∧ l.area_ratio = 2 / 3) →
  ∃ p : Point, (square.dividing_lines.filter (λ l => l.intersects_point = p)).length ≥ 3 :=
sorry

end NUMINAMATH_CALUDE_at_least_three_lines_intersect_l3550_355032


namespace NUMINAMATH_CALUDE_regular_18gon_relation_l3550_355068

/-- A regular 18-gon inscribed in a circle -/
structure Regular18Gon where
  /-- The radius of the circumscribed circle -/
  r : ℝ
  /-- The side length of the 18-gon -/
  a : ℝ
  /-- The radius is positive -/
  r_pos : 0 < r

/-- Theorem: For a regular 18-gon inscribed in a circle, a^3 + r^3 = 3ar^2 -/
theorem regular_18gon_relation (polygon : Regular18Gon) : 
  polygon.a^3 + polygon.r^3 = 3 * polygon.a * polygon.r^2 := by
  sorry

end NUMINAMATH_CALUDE_regular_18gon_relation_l3550_355068


namespace NUMINAMATH_CALUDE_cookie_bringers_l3550_355029

theorem cookie_bringers (num_brownie_students : ℕ) (brownies_per_student : ℕ)
                        (num_donut_students : ℕ) (donuts_per_student : ℕ)
                        (cookies_per_student : ℕ) (price_per_item : ℚ)
                        (total_raised : ℚ) :
  num_brownie_students = 30 →
  brownies_per_student = 12 →
  num_donut_students = 15 →
  donuts_per_student = 12 →
  cookies_per_student = 24 →
  price_per_item = 2 →
  total_raised = 2040 →
  ∃ (num_cookie_students : ℕ),
    num_cookie_students = 20 ∧
    total_raised = price_per_item * (num_brownie_students * brownies_per_student +
                                     num_cookie_students * cookies_per_student +
                                     num_donut_students * donuts_per_student) :=
by sorry

end NUMINAMATH_CALUDE_cookie_bringers_l3550_355029


namespace NUMINAMATH_CALUDE_opposite_of_negative_2023_l3550_355072

theorem opposite_of_negative_2023 : -((-2023) : ℤ) = (2023 : ℤ) := by
  sorry

end NUMINAMATH_CALUDE_opposite_of_negative_2023_l3550_355072


namespace NUMINAMATH_CALUDE_intersection_locus_is_hyperbola_l3550_355037

/-- The locus of points (x, y) satisfying the given system of equations forms a hyperbola -/
theorem intersection_locus_is_hyperbola :
  ∀ (x y u : ℝ), 
  (2 * u * x - 3 * y - 4 * u = 0) →
  (x - 3 * u * y + 4 = 0) →
  ∃ (a b : ℝ), a > 0 ∧ b > 0 ∧ x^2 / a^2 - y^2 / b^2 = 1 :=
by sorry

end NUMINAMATH_CALUDE_intersection_locus_is_hyperbola_l3550_355037


namespace NUMINAMATH_CALUDE_expected_sum_of_two_marbles_l3550_355097

def marbleSet : Finset ℕ := Finset.range 6

def marblePairs : Finset (ℕ × ℕ) :=
  (marbleSet.product marbleSet).filter (fun p => p.1 < p.2)

def pairSum (p : ℕ × ℕ) : ℕ := p.1 + p.2 + 2

theorem expected_sum_of_two_marbles :
  (marblePairs.sum pairSum) / marblePairs.card = 7 := by
  sorry

end NUMINAMATH_CALUDE_expected_sum_of_two_marbles_l3550_355097


namespace NUMINAMATH_CALUDE_tan70_cos10_sqrt3tan20_minus1_eq_neg1_l3550_355088

theorem tan70_cos10_sqrt3tan20_minus1_eq_neg1 :
  Real.tan (70 * π / 180) * Real.cos (10 * π / 180) * (Real.sqrt 3 * Real.tan (20 * π / 180) - 1) = -1 := by
  sorry

end NUMINAMATH_CALUDE_tan70_cos10_sqrt3tan20_minus1_eq_neg1_l3550_355088


namespace NUMINAMATH_CALUDE_factorization_of_75x_plus_45_l3550_355057

theorem factorization_of_75x_plus_45 (x : ℝ) : 75 * x + 45 = 15 * (5 * x + 3) := by
  sorry

end NUMINAMATH_CALUDE_factorization_of_75x_plus_45_l3550_355057


namespace NUMINAMATH_CALUDE_smallest_y_for_perfect_cube_l3550_355035

/-- Given x = 4 * 21 * 63, the smallest positive integer y such that xy is a perfect cube is 14 -/
theorem smallest_y_for_perfect_cube (x : ℕ) (hx : x = 4 * 21 * 63) :
  ∃ y : ℕ, y > 0 ∧ 
    (∃ z : ℕ, x * y = z^3) ∧
    (∀ w : ℕ, w > 0 ∧ w < y → ¬∃ z : ℕ, x * w = z^3) ∧
    y = 14 := by
  sorry

end NUMINAMATH_CALUDE_smallest_y_for_perfect_cube_l3550_355035


namespace NUMINAMATH_CALUDE_family_composition_l3550_355000

theorem family_composition :
  ∀ (boys girls : ℕ),
  (boys > 0 ∧ girls > 0) →
  (boys - 1 = girls) →
  (boys = 2 * (girls - 1)) →
  (boys = 4 ∧ girls = 3) :=
by sorry

end NUMINAMATH_CALUDE_family_composition_l3550_355000


namespace NUMINAMATH_CALUDE_license_plate_theorem_l3550_355009

/-- The number of letters in the English alphabet -/
def alphabet_size : ℕ := 26

/-- The number of vowels (including Y) -/
def vowel_count : ℕ := 6

/-- The number of consonants -/
def consonant_count : ℕ := alphabet_size - vowel_count

/-- The number of possible three-character license plates with two consonants followed by a vowel -/
def license_plate_count : ℕ := consonant_count * consonant_count * vowel_count

theorem license_plate_theorem : license_plate_count = 2400 := by
  sorry

end NUMINAMATH_CALUDE_license_plate_theorem_l3550_355009


namespace NUMINAMATH_CALUDE_segment_existence_l3550_355053

theorem segment_existence (pencil_length eraser_length : ℝ) 
  (h_pencil : pencil_length > 0) (h_eraser : eraser_length > 0) : 
  ∃ (x y : ℝ), x > 0 ∧ y > 0 ∧ x + y = pencil_length ∧ Real.sqrt (x * y) = eraser_length :=
sorry

end NUMINAMATH_CALUDE_segment_existence_l3550_355053


namespace NUMINAMATH_CALUDE_optimal_price_l3550_355074

/-- Represents the daily sales volume as a function of price -/
def sales (x : ℝ) : ℝ := 400 - 20 * x

/-- Represents the daily profit as a function of price -/
def profit (x : ℝ) : ℝ := (x - 8) * sales x

theorem optimal_price :
  ∃ (x : ℝ), 8 ≤ x ∧ x ≤ 15 ∧ profit x = 640 :=
by sorry

end NUMINAMATH_CALUDE_optimal_price_l3550_355074


namespace NUMINAMATH_CALUDE_two_numbers_sum_667_lcm_gcd_120_l3550_355082

theorem two_numbers_sum_667_lcm_gcd_120 :
  ∀ a b : ℕ,
  a + b = 667 →
  (Nat.lcm a b) / (Nat.gcd a b) = 120 →
  ((a = 552 ∧ b = 115) ∨ (a = 115 ∧ b = 552) ∨ (a = 435 ∧ b = 232) ∨ (a = 232 ∧ b = 435)) :=
by sorry

end NUMINAMATH_CALUDE_two_numbers_sum_667_lcm_gcd_120_l3550_355082


namespace NUMINAMATH_CALUDE_smallest_square_containing_circle_l3550_355055

theorem smallest_square_containing_circle (r : ℝ) (h : r = 4) :
  (2 * r) ^ 2 = 64 := by
  sorry

end NUMINAMATH_CALUDE_smallest_square_containing_circle_l3550_355055


namespace NUMINAMATH_CALUDE_sum_of_divisors_l3550_355004

def isPrime (n : ℕ) : Prop := sorry

def numDivisors (n : ℕ) : ℕ := sorry

theorem sum_of_divisors (p q r : ℕ) (hp : isPrime p) (hq : isPrime q) (hr : isPrime r)
  (hpq : p ≠ q) (hpr : p ≠ r) (hqr : q ≠ r) :
  let a := p^4
  let b := q * r
  let k := a^5
  let m := b^2
  numDivisors k + numDivisors m = 30 := by sorry

end NUMINAMATH_CALUDE_sum_of_divisors_l3550_355004


namespace NUMINAMATH_CALUDE_cubic_roots_sum_of_cubes_l3550_355073

theorem cubic_roots_sum_of_cubes (a b c : ℂ) : 
  (a^3 - 2*a^2 + 3*a - 5 = 0) → 
  (b^3 - 2*b^2 + 3*b - 5 = 0) → 
  (c^3 - 2*c^2 + 3*c - 5 = 0) → 
  a^3 + b^3 + c^3 = 5 := by sorry

end NUMINAMATH_CALUDE_cubic_roots_sum_of_cubes_l3550_355073


namespace NUMINAMATH_CALUDE_no_infinite_sequence_sqrt_difference_l3550_355063

theorem no_infinite_sequence_sqrt_difference :
  ¬ (∃ (x : ℕ → ℝ), (∀ n, 0 < x n) ∧ 
    (∀ n, x (n + 2) = Real.sqrt (x (n + 1)) - Real.sqrt (x n))) := by
  sorry

end NUMINAMATH_CALUDE_no_infinite_sequence_sqrt_difference_l3550_355063


namespace NUMINAMATH_CALUDE_projectile_max_height_l3550_355036

/-- The height function of the projectile -/
def h (t : ℝ) : ℝ := -20 * t^2 + 100 * t + 50

/-- The maximum height reached by the projectile -/
def max_height : ℝ := 175

/-- Theorem stating that the maximum height reached by the projectile is 175 meters -/
theorem projectile_max_height :
  ∀ t : ℝ, h t ≤ max_height :=
by sorry

end NUMINAMATH_CALUDE_projectile_max_height_l3550_355036


namespace NUMINAMATH_CALUDE_sine_tangent_relation_l3550_355027

theorem sine_tangent_relation (α : Real) (h : 0 < α ∧ α < Real.pi) :
  (∃ β, (Real.sqrt 2 / 2 < Real.sin β ∧ Real.sin β < 1) ∧ ¬(Real.tan β > 1)) ∧
  (∀ γ, Real.tan γ > 1 → Real.sqrt 2 / 2 < Real.sin γ ∧ Real.sin γ < 1) :=
by sorry

end NUMINAMATH_CALUDE_sine_tangent_relation_l3550_355027


namespace NUMINAMATH_CALUDE_differential_equation_solution_l3550_355012

open Real

theorem differential_equation_solution (x : ℝ) (C : ℝ) :
  let y : ℝ → ℝ := λ x => cos x * (sin x + C)
  (deriv y) x + y x * tan x = cos x ^ 2 := by
  sorry

end NUMINAMATH_CALUDE_differential_equation_solution_l3550_355012


namespace NUMINAMATH_CALUDE_composite_sum_l3550_355051

theorem composite_sum (a b c d m n : ℕ) 
  (ha : a > b) (hb : b > c) (hc : c > d) 
  (hdiv : (a + b - c + d) ∣ (a * c + b * d))
  (hm : m > 0) (hn : Odd n) : 
  ∃ k > 1, k ∣ (a^n * b^m + c^m * d^n) :=
sorry

end NUMINAMATH_CALUDE_composite_sum_l3550_355051


namespace NUMINAMATH_CALUDE_min_value_theorem_l3550_355080

theorem min_value_theorem (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : x + 3 * y = 5 * x * y) :
  (∀ a b : ℝ, a > 0 → b > 0 → a + 3 * b = 5 * a * b → 3 * a + 4 * b ≥ 3 * x + 4 * y) →
  3 * x + 4 * y = 5 ∧ x + 4 * y = 3 := by sorry

end NUMINAMATH_CALUDE_min_value_theorem_l3550_355080


namespace NUMINAMATH_CALUDE_rectangular_solid_diagonal_l3550_355025

theorem rectangular_solid_diagonal (a b c : ℝ) 
  (h1 : 2 * (a * b + b * c + a * c) = 24)
  (h2 : 4 * (a + b + c) = 28) :
  Real.sqrt (a^2 + b^2 + c^2) = 5 := by
  sorry

end NUMINAMATH_CALUDE_rectangular_solid_diagonal_l3550_355025


namespace NUMINAMATH_CALUDE_local_minimum_implies_c_eq_two_l3550_355083

/-- The function f(x) -/
def f (c : ℝ) (x : ℝ) : ℝ := 2 * x * (x - c)^2 + 3

/-- Theorem: If f(x) has a local minimum at x = 2, then c = 2 -/
theorem local_minimum_implies_c_eq_two (c : ℝ) :
  (∃ δ > 0, ∀ x, |x - 2| < δ → f c x ≥ f c 2) →
  c = 2 :=
by sorry

end NUMINAMATH_CALUDE_local_minimum_implies_c_eq_two_l3550_355083


namespace NUMINAMATH_CALUDE_function_lower_bound_l3550_355031

theorem function_lower_bound
  (f : ℝ → ℝ)
  (h_cont : ContinuousOn f (Set.Ioi 0))
  (h_ineq : ∀ x > 0, f (x^2) ≥ f x)
  (h_f1 : f 1 = 5) :
  ∀ x > 0, f x ≥ 5 :=
by sorry

end NUMINAMATH_CALUDE_function_lower_bound_l3550_355031


namespace NUMINAMATH_CALUDE_no_solution_equation_l3550_355018

theorem no_solution_equation : ∀ x : ℝ, 
  4 * x * (10 * x - (-10 - (3 * x - 8 * (x + 1)))) + 
  5 * (12 - (4 * (x + 1) - 3 * x)) ≠ 
  18 * x^2 - (6 * x^2 - (7 * x + 4 * (2 * x^2 - x + 11))) := by
  sorry

end NUMINAMATH_CALUDE_no_solution_equation_l3550_355018


namespace NUMINAMATH_CALUDE_batsman_average_l3550_355028

theorem batsman_average (x : ℕ) : 
  (40 * x + 30 * 10) / (x + 10) = 35 → x = 10 := by
  sorry

end NUMINAMATH_CALUDE_batsman_average_l3550_355028


namespace NUMINAMATH_CALUDE_perpendicular_lines_m_value_l3550_355066

/-- 
Given two lines in the xy-plane:
  Line1: x - y - 2 = 0
  Line2: mx + y = 0
If Line1 is perpendicular to Line2, then m = 1
-/
theorem perpendicular_lines_m_value (m : ℝ) : 
  (∀ x y : ℝ, x - y - 2 = 0 → mx + y = 0 → (1 : ℝ) * m = -1) → 
  m = 1 := by sorry

end NUMINAMATH_CALUDE_perpendicular_lines_m_value_l3550_355066


namespace NUMINAMATH_CALUDE_rahim_pillows_l3550_355081

/-- The number of pillows Rahim bought initially -/
def initial_pillows : ℕ := 4

/-- The initial average cost of pillows -/
def initial_avg_cost : ℚ := 5

/-- The price of the fifth pillow -/
def fifth_pillow_price : ℚ := 10

/-- The new average price of 5 pillows -/
def new_avg_price : ℚ := 6

/-- Proof that the number of pillows Rahim bought initially is 4 -/
theorem rahim_pillows :
  (initial_avg_cost * initial_pillows + fifth_pillow_price) / (initial_pillows + 1) = new_avg_price :=
by sorry

end NUMINAMATH_CALUDE_rahim_pillows_l3550_355081


namespace NUMINAMATH_CALUDE_eating_contest_l3550_355079

/-- Eating contest problem -/
theorem eating_contest (hot_dog_weight burger_weight pie_weight : ℕ)
  (mason_hotdog_multiplier : ℕ) (noah_burger_count : ℕ) (mason_hotdog_total_weight : ℕ)
  (h1 : hot_dog_weight = 2)
  (h2 : burger_weight = 5)
  (h3 : pie_weight = 10)
  (h4 : mason_hotdog_multiplier = 3)
  (h5 : noah_burger_count = 8)
  (h6 : mason_hotdog_total_weight = 30) :
  ∃ (jacob_pie_count : ℕ),
    jacob_pie_count = 5 ∧
    mason_hotdog_total_weight = jacob_pie_count * mason_hotdog_multiplier * hot_dog_weight :=
by
  sorry


end NUMINAMATH_CALUDE_eating_contest_l3550_355079


namespace NUMINAMATH_CALUDE_ferris_wheel_seat_count_l3550_355042

/-- The number of seats on a Ferris wheel -/
def ferris_wheel_seats (total_people : ℕ) (people_per_seat : ℕ) : ℕ :=
  (total_people + people_per_seat - 1) / people_per_seat

/-- Theorem: The Ferris wheel has 3 seats -/
theorem ferris_wheel_seat_count : ferris_wheel_seats 8 3 = 3 := by
  sorry

end NUMINAMATH_CALUDE_ferris_wheel_seat_count_l3550_355042


namespace NUMINAMATH_CALUDE_age_ratio_after_five_years_l3550_355007

/-- Theorem: Ratio of parent's age to son's age after 5 years -/
theorem age_ratio_after_five_years
  (parent_age : ℕ)
  (son_age : ℕ)
  (h1 : parent_age = 45)
  (h2 : son_age = 15) :
  (parent_age + 5) / (son_age + 5) = 5 / 2 := by
  sorry

end NUMINAMATH_CALUDE_age_ratio_after_five_years_l3550_355007


namespace NUMINAMATH_CALUDE_perpendicular_to_parallel_planes_l3550_355013

-- Define the types for planes and lines
variable (Plane Line : Type)

-- Define the perpendicular and parallel relations
variable (perp : Line → Plane → Prop)
variable (para : Plane → Plane → Prop)

-- State the theorem
theorem perpendicular_to_parallel_planes 
  (α β : Plane) (l : Line) 
  (h1 : perp l α) (h2 : para α β) : 
  perp l β := by
  sorry

end NUMINAMATH_CALUDE_perpendicular_to_parallel_planes_l3550_355013


namespace NUMINAMATH_CALUDE_kindergarten_total_l3550_355086

/-- Represents the number of children in a kindergarten with different pet ownership patterns -/
structure KindergartenPets where
  dogs_only : ℕ
  both : ℕ
  cats_total : ℕ

/-- Calculates the total number of children in the kindergarten -/
def total_children (k : KindergartenPets) : ℕ :=
  k.dogs_only + k.both + (k.cats_total - k.both)

/-- Theorem stating the total number of children in the kindergarten -/
theorem kindergarten_total (k : KindergartenPets) 
  (h1 : k.dogs_only = 18)
  (h2 : k.both = 6)
  (h3 : k.cats_total = 12) :
  total_children k = 30 := by
  sorry

#check kindergarten_total

end NUMINAMATH_CALUDE_kindergarten_total_l3550_355086


namespace NUMINAMATH_CALUDE_middle_card_is_four_l3550_355043

/-- Represents a valid triple of card numbers -/
def ValidTriple (a b c : ℕ) : Prop :=
  0 < a ∧ a < b ∧ b < c ∧ a + b + c = 15

/-- Predicate for uncertainty about other numbers given the left card -/
def LeftUncertain (a : ℕ) : Prop :=
  ∃ b₁ c₁ b₂ c₂, b₁ ≠ b₂ ∧ ValidTriple a b₁ c₁ ∧ ValidTriple a b₂ c₂

/-- Predicate for uncertainty about other numbers given the right card -/
def RightUncertain (c : ℕ) : Prop :=
  ∃ a₁ b₁ a₂ b₂, a₁ ≠ a₂ ∧ ValidTriple a₁ b₁ c ∧ ValidTriple a₂ b₂ c

/-- Predicate for uncertainty about other numbers given the middle card -/
def MiddleUncertain (b : ℕ) : Prop :=
  ∃ a₁ c₁ a₂ c₂, a₁ ≠ a₂ ∧ ValidTriple a₁ b c₁ ∧ ValidTriple a₂ b c₂

theorem middle_card_is_four :
  ∀ a b c : ℕ,
    ValidTriple a b c →
    (∀ x, ValidTriple x b c → LeftUncertain x) →
    (∀ z, ValidTriple a b z → RightUncertain z) →
    MiddleUncertain b →
    b = 4 := by
  sorry

end NUMINAMATH_CALUDE_middle_card_is_four_l3550_355043


namespace NUMINAMATH_CALUDE_novel_pages_count_l3550_355090

theorem novel_pages_count : 
  ∀ (vacation_days : ℕ) 
    (first_two_days_avg : ℕ) 
    (next_three_days_avg : ℕ) 
    (last_day_pages : ℕ),
  vacation_days = 6 →
  first_two_days_avg = 42 →
  next_three_days_avg = 35 →
  last_day_pages = 15 →
  (2 * first_two_days_avg + 3 * next_three_days_avg + last_day_pages) = 204 := by
sorry

end NUMINAMATH_CALUDE_novel_pages_count_l3550_355090


namespace NUMINAMATH_CALUDE_matrix_inverse_equality_l3550_355077

/-- Given a 3x3 matrix B with a variable d in the (2,3) position, prove that if B^(-1) = k * B, then d = 13/9 and k = -329/52 -/
theorem matrix_inverse_equality (d k : ℚ) : 
  let B : Matrix (Fin 3) (Fin 3) ℚ := !![1, 2, 3; 4, 5, d; 6, 7, 8]
  (B⁻¹ = k • B) → (d = 13/9 ∧ k = -329/52) := by
  sorry

end NUMINAMATH_CALUDE_matrix_inverse_equality_l3550_355077


namespace NUMINAMATH_CALUDE_seating_chart_interpretation_l3550_355084

/-- Represents a seating chart configuration -/
structure SeatingChart where
  columns : ℕ
  rows : ℕ

/-- Interprets a pair of natural numbers as a seating chart -/
def interpretSeatingChart (pair : ℕ × ℕ) : SeatingChart :=
  ⟨pair.1, pair.2⟩

theorem seating_chart_interpretation :
  let chart := interpretSeatingChart (5, 4)
  chart.columns = 5 ∧ chart.rows = 4 := by
  sorry

end NUMINAMATH_CALUDE_seating_chart_interpretation_l3550_355084


namespace NUMINAMATH_CALUDE_repeating_decimal_equals_fraction_l3550_355014

-- Define the repeating decimal 0.4̅36̅
def repeating_decimal : ℚ := 0.4 + (36 / 990)

-- Theorem statement
theorem repeating_decimal_equals_fraction : repeating_decimal = 24 / 55 := by
  sorry

end NUMINAMATH_CALUDE_repeating_decimal_equals_fraction_l3550_355014


namespace NUMINAMATH_CALUDE_opposite_of_negative_half_l3550_355006

theorem opposite_of_negative_half : -(-(1/2)) = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_opposite_of_negative_half_l3550_355006


namespace NUMINAMATH_CALUDE_four_propositions_l3550_355052

-- Define the propositions
def opposite_numbers (x y : ℝ) : Prop := x = -y

def has_real_roots (a b c : ℝ) : Prop :=
  ∃ x : ℝ, a * x^2 + b * x + c = 0

def congruent_triangles (t1 t2 : Set ℝ × Set ℝ) : Prop :=
  sorry  -- Definition of congruent triangles

def equal_areas (t1 t2 : Set ℝ × Set ℝ) : Prop :=
  sorry  -- Definition of equal areas for triangles

def right_triangle (t : Set ℝ × Set ℝ) : Prop :=
  sorry  -- Definition of right triangle

def has_two_acute_angles (t : Set ℝ × Set ℝ) : Prop :=
  sorry  -- Definition of triangle with two acute angles

-- Theorem to prove
theorem four_propositions :
  (∀ x y : ℝ, opposite_numbers x y → x + y = 0) ∧
  (∀ q : ℝ, ¬(has_real_roots 1 2 q) → q > 1) ∧
  ¬(∀ t1 t2 : Set ℝ × Set ℝ, ¬(congruent_triangles t1 t2) → ¬(equal_areas t1 t2)) ∧
  ¬(∀ t : Set ℝ × Set ℝ, has_two_acute_angles t → right_triangle t) :=
by
  sorry

end NUMINAMATH_CALUDE_four_propositions_l3550_355052


namespace NUMINAMATH_CALUDE_sequence_proof_l3550_355087

def arithmetic_sequence (a b c : ℝ) : Prop := b - a = c - b

def geometric_sequence (f : ℕ → ℝ) : Prop := ∀ n : ℕ, f (n + 1) / f n = f 2 / f 1

def sum_sequence (f : ℕ → ℝ) : ℕ → ℝ
  | 0 => 0
  | n + 1 => sum_sequence f n + f (n + 1)

theorem sequence_proof 
  (a b c : ℝ) 
  (h_positive : a > 0 ∧ b > 0 ∧ c > 0)
  (h_arithmetic : arithmetic_sequence a b c)
  (h_sum : a + b + c = 15)
  (b_n : ℕ → ℝ)
  (h_geometric : geometric_sequence (λ n => b_n (n + 2)))
  (h_relation : b_n 3 = a + 2 ∧ b_n 4 = b + 5 ∧ b_n 5 = c + 13) :
  (∀ n : ℕ, b_n n = (5/4) * 2^(n-1)) ∧
  (geometric_sequence (λ n => sum_sequence b_n n + 5/4) ∧
   (sum_sequence b_n 1 + 5/4 = 5/2) ∧
   (∀ n : ℕ, (sum_sequence b_n (n+1) + 5/4) / (sum_sequence b_n n + 5/4) = 2)) :=
sorry

end NUMINAMATH_CALUDE_sequence_proof_l3550_355087


namespace NUMINAMATH_CALUDE_smallest_divisible_by_15_16_18_l3550_355092

theorem smallest_divisible_by_15_16_18 : 
  ∃ n : ℕ+, (∀ m : ℕ+, (15 ∣ m) ∧ (16 ∣ m) ∧ (18 ∣ m) → n ≤ m) ∧ 
             (15 ∣ n) ∧ (16 ∣ n) ∧ (18 ∣ n) :=
by
  use 720
  sorry

end NUMINAMATH_CALUDE_smallest_divisible_by_15_16_18_l3550_355092


namespace NUMINAMATH_CALUDE_factorization_equality_l3550_355021

theorem factorization_equality (x y : ℝ) :
  (1 - x^2) * (1 - y^2) - 4*x*y = (x*y - 1 + x + y) * (x*y - 1 - x - y) := by
  sorry

end NUMINAMATH_CALUDE_factorization_equality_l3550_355021


namespace NUMINAMATH_CALUDE_bike_price_proof_l3550_355003

theorem bike_price_proof (upfront_payment : ℝ) (upfront_percentage : ℝ) (total_price : ℝ) : 
  upfront_payment = 150 ∧ 
  upfront_percentage = 0.1 ∧ 
  upfront_payment = upfront_percentage * total_price →
  total_price = 1500 :=
by sorry

end NUMINAMATH_CALUDE_bike_price_proof_l3550_355003


namespace NUMINAMATH_CALUDE_octal_arithmetic_sum_1_to_30_l3550_355089

/-- Represents a number in base 8 -/
def OctalNum := Nat

/-- Convert a decimal number to its octal representation -/
def toOctal (n : Nat) : OctalNum := sorry

/-- Convert an octal number to its decimal representation -/
def fromOctal (n : OctalNum) : Nat := sorry

/-- Sum of arithmetic series in base 8 -/
def octalArithmeticSum (first last : OctalNum) : OctalNum := sorry

theorem octal_arithmetic_sum_1_to_30 :
  octalArithmeticSum (toOctal 1) (toOctal 24) = toOctal 300 := by
  sorry

end NUMINAMATH_CALUDE_octal_arithmetic_sum_1_to_30_l3550_355089


namespace NUMINAMATH_CALUDE_fifth_power_sum_l3550_355078

theorem fifth_power_sum (a b x y : ℝ) 
  (eq1 : a * x + b * y = 3)
  (eq2 : a * x^2 + b * y^2 = 7)
  (eq3 : a * x^3 + b * y^3 = 16)
  (eq4 : a * x^4 + b * y^4 = 42) :
  a * x^5 + b * y^5 = 99 := by
sorry

end NUMINAMATH_CALUDE_fifth_power_sum_l3550_355078


namespace NUMINAMATH_CALUDE_complex_equation_solution_l3550_355024

theorem complex_equation_solution : ∃ (x : ℂ), 5 + 2 * Complex.I * x = -3 - 6 * Complex.I * x ∧ x = Complex.I := by
  sorry

end NUMINAMATH_CALUDE_complex_equation_solution_l3550_355024


namespace NUMINAMATH_CALUDE_custom_operation_results_l3550_355058

-- Define the custom operation
def customOp (a b : ℤ) : ℤ := a^2 - (a + b) + a*b

-- State the theorem
theorem custom_operation_results :
  (customOp 2 (-3) = -1) ∧ (customOp 4 (customOp 2 (-3)) = 7) := by
  sorry

end NUMINAMATH_CALUDE_custom_operation_results_l3550_355058


namespace NUMINAMATH_CALUDE_equation_solution_l3550_355038

theorem equation_solution : 
  ∃! x : ℚ, (53 - 3*x)^(1/4) + (39 + 3*x)^(1/4) = 5 :=
by
  -- The unique solution is x = -23/3
  use -23/3
  sorry

end NUMINAMATH_CALUDE_equation_solution_l3550_355038


namespace NUMINAMATH_CALUDE_distance_to_specific_line_l3550_355091

/-- Polar coordinates of a point -/
structure PolarPoint where
  r : ℝ
  θ : ℝ

/-- Polar equation of a line -/
structure PolarLine where
  equation : ℝ → ℝ → Prop

/-- Distance from a point to a line -/
def distanceToLine (p : PolarPoint) (l : PolarLine) : ℝ := sorry

theorem distance_to_specific_line :
  let A : PolarPoint := ⟨2, 7 * π / 4⟩
  let L : PolarLine := ⟨fun ρ θ ↦ ρ * Real.sin (θ + π / 4) = Real.sqrt 2 / 2⟩
  distanceToLine A L = Real.sqrt 2 / 2 := by
  sorry

end NUMINAMATH_CALUDE_distance_to_specific_line_l3550_355091


namespace NUMINAMATH_CALUDE_prime_factors_count_l3550_355023

theorem prime_factors_count (p q r : ℕ) (h1 : p = 4) (h2 : q = 7) (h3 : r = 11) 
  (h4 : p = 2^2) (h5 : Nat.Prime q) (h6 : Nat.Prime r) : 
  (Nat.factors (p^11 * q^7 * r^2)).length = 31 := by
sorry

end NUMINAMATH_CALUDE_prime_factors_count_l3550_355023


namespace NUMINAMATH_CALUDE_crayon_selection_theorem_l3550_355002

/-- The number of ways to select k items from n items -/
def choose (n k : ℕ) : ℕ := Nat.choose n k

/-- The total number of crayons in the box -/
def total_crayons : ℕ := 15

/-- The number of red crayons in the box -/
def red_crayons : ℕ := 3

/-- The number of crayons to be selected -/
def selected_crayons : ℕ := 5

/-- The number of red crayons that must be selected -/
def selected_red : ℕ := 2

/-- The number of ways to select crayons under the given conditions -/
def ways_to_select : ℕ := choose red_crayons selected_red * choose (total_crayons - red_crayons) (selected_crayons - selected_red)

theorem crayon_selection_theorem : ways_to_select = 660 := by
  sorry

end NUMINAMATH_CALUDE_crayon_selection_theorem_l3550_355002


namespace NUMINAMATH_CALUDE_fools_gold_ounces_l3550_355016

def earnings_per_ounce : ℝ := 9
def fine : ℝ := 50
def remaining_money : ℝ := 22

theorem fools_gold_ounces :
  ∃ (x : ℝ), x * earnings_per_ounce - fine = remaining_money ∧ x = 8 := by
  sorry

end NUMINAMATH_CALUDE_fools_gold_ounces_l3550_355016


namespace NUMINAMATH_CALUDE_intersection_locus_l3550_355005

-- Define the two lines as functions of t
def line1 (x y t : ℝ) : Prop := 2 * x + 3 * y = t
def line2 (x y t : ℝ) : Prop := 5 * x - 7 * y = t

-- Define the locus line
def locusLine (x y : ℝ) : Prop := y = 0.3 * x

-- Theorem statement
theorem intersection_locus :
  ∀ (t : ℝ), ∃ (x y : ℝ), line1 x y t ∧ line2 x y t → locusLine x y :=
by sorry

end NUMINAMATH_CALUDE_intersection_locus_l3550_355005


namespace NUMINAMATH_CALUDE_min_value_quadratic_sum_l3550_355093

theorem min_value_quadratic_sum (a b c d : ℝ) (h : a * d + b * c = 1) :
  ∃ (min_val : ℝ), min_val = 2 * Real.sqrt 3 ∧
  ∀ (u : ℝ), u = a^2 + b^2 + c^2 + d^2 + (a + c)^2 + (b - d)^2 → u ≥ min_val :=
by sorry

end NUMINAMATH_CALUDE_min_value_quadratic_sum_l3550_355093


namespace NUMINAMATH_CALUDE_fraction_between_one_quarter_between_one_seventh_and_one_fourth_l3550_355062

theorem fraction_between (a b : ℚ) (t : ℚ) (h : 0 ≤ t ∧ t ≤ 1) :
  a + t * (b - a) = (1 - t) * a + t * b :=
by sorry

theorem one_quarter_between_one_seventh_and_one_fourth :
  (1 : ℚ)/7 + (1/4) * ((1/4) - (1/7)) = 23/112 :=
by sorry

end NUMINAMATH_CALUDE_fraction_between_one_quarter_between_one_seventh_and_one_fourth_l3550_355062


namespace NUMINAMATH_CALUDE_jesse_room_area_l3550_355039

/-- Calculates the area of a rectangle given its length and width -/
def rectangleArea (length width : ℝ) : ℝ := length * width

/-- Represents an L-shaped room with two rectangular parts -/
structure LShapedRoom where
  length1 : ℝ
  width1 : ℝ
  length2 : ℝ
  width2 : ℝ

/-- Calculates the total area of an L-shaped room -/
def totalArea (room : LShapedRoom) : ℝ :=
  rectangleArea room.length1 room.width1 + rectangleArea room.length2 room.width2

/-- Theorem: The total area of Jesse's L-shaped room is 120 square feet -/
theorem jesse_room_area :
  let room : LShapedRoom := { length1 := 12, width1 := 8, length2 := 6, width2 := 4 }
  totalArea room = 120 := by
  sorry

end NUMINAMATH_CALUDE_jesse_room_area_l3550_355039


namespace NUMINAMATH_CALUDE_sum_of_digits_power_product_l3550_355033

def power_product (a b c : ℕ) : ℕ := a^2010 * b^2012 * c

def sum_of_digits (n : ℕ) : ℕ := sorry

theorem sum_of_digits_power_product : sum_of_digits (power_product 2 5 7) = 13 := by sorry

end NUMINAMATH_CALUDE_sum_of_digits_power_product_l3550_355033


namespace NUMINAMATH_CALUDE_typing_time_proof_l3550_355060

/-- Calculates the time in hours required to type a research paper given the typing speed and number of words. -/
def time_to_type (typing_speed : ℕ) (total_words : ℕ) : ℚ :=
  (total_words : ℚ) / (typing_speed : ℚ) / 60

/-- Proves that given a typing speed of 38 words per minute and a research paper with 4560 words, the time required to type the paper is 2 hours. -/
theorem typing_time_proof :
  time_to_type 38 4560 = 2 := by
  sorry

end NUMINAMATH_CALUDE_typing_time_proof_l3550_355060


namespace NUMINAMATH_CALUDE_quadratic_factorization_l3550_355008

theorem quadratic_factorization (E F : ℤ) :
  (∀ y : ℝ, 15 * y^2 - 82 * y + 48 = (E * y - 16) * (F * y - 3)) →
  E * F + E = 20 := by
sorry

end NUMINAMATH_CALUDE_quadratic_factorization_l3550_355008


namespace NUMINAMATH_CALUDE_triangle_circumcircle_l3550_355040

-- Define the triangle ABC
def A : ℝ × ℝ := (1, 3)

-- Define the line BC
def line_BC (x y : ℝ) : Prop := y - 1 = 0

-- Define the median from A to BC
def median_A (x y : ℝ) : Prop := x - 3*y + 4 = 0

-- Define the circumcircle equation
def circumcircle (x y : ℝ) : Prop := (x - 1)^2 + (y - 1)^2 = 4

-- Theorem statement
theorem triangle_circumcircle : 
  ∀ (B C : ℝ × ℝ),
  line_BC B.1 B.2 ∧ line_BC C.1 C.2 ∧
  median_A ((B.1 + C.1)/2) ((B.2 + C.2)/2) →
  circumcircle B.1 B.2 ∧ circumcircle C.1 C.2 ∧ circumcircle A.1 A.2 :=
by sorry

end NUMINAMATH_CALUDE_triangle_circumcircle_l3550_355040


namespace NUMINAMATH_CALUDE_expression_value_l3550_355061

theorem expression_value (x y : ℚ) (hx : x ≠ 0) (hy : y ≠ 0) :
  x / |x| + |y| / y = 2 ∨ x / |x| + |y| / y = 0 ∨ x / |x| + |y| / y = -2 := by
  sorry

end NUMINAMATH_CALUDE_expression_value_l3550_355061


namespace NUMINAMATH_CALUDE_fencemaker_problem_l3550_355034

/-- Given a rectangular yard with one side of 40 feet and an area of 480 square feet,
    the perimeter minus one side is equal to 64 feet. -/
theorem fencemaker_problem (length width : ℝ) : 
  length = 40 ∧ 
  length * width = 480 ∧ 
  width > 0 → 
  2 * width + length = 64 := by
  sorry

end NUMINAMATH_CALUDE_fencemaker_problem_l3550_355034


namespace NUMINAMATH_CALUDE_invalid_reasoning_l3550_355067

-- Define the types of reasoning
inductive ReasoningType
  | Analogy
  | Inductive
  | Deductive

-- Define the concept of valid reasoning
def isValidReasoning (r : ReasoningType) : Prop :=
  match r with
  | ReasoningType.Analogy => true
  | ReasoningType.Inductive => true
  | ReasoningType.Deductive => true

-- Define the reasoning options
def optionA : ReasoningType := ReasoningType.Analogy
def optionB : ReasoningType := ReasoningType.Inductive
def optionC : ReasoningType := ReasoningType.Inductive
def optionD : ReasoningType := ReasoningType.Inductive

-- Theorem to prove
theorem invalid_reasoning :
  isValidReasoning optionA ∧
  isValidReasoning optionB ∧
  ¬(isValidReasoning optionC) ∧
  isValidReasoning optionD :=
by sorry

end NUMINAMATH_CALUDE_invalid_reasoning_l3550_355067


namespace NUMINAMATH_CALUDE_tangent_line_to_parabola_l3550_355098

theorem tangent_line_to_parabola (b : ℝ) :
  (∀ x y : ℝ, y = -2*x + b → y^2 = 8*x → (∀ ε > 0, ∃ δ > 0, ∀ x' y', 
    ((x' - x)^2 + (y' - y)^2 < δ^2) → (y' + 2*x' - b)^2 > 0 ∨ 
    ((y')^2 - 8*x')^2 > 0)) →
  b = -1 :=
sorry

end NUMINAMATH_CALUDE_tangent_line_to_parabola_l3550_355098

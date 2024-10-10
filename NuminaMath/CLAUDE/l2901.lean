import Mathlib

namespace equation_solution_l2901_290135

theorem equation_solution : ∃! x : ℚ, 3 * x - 4 = -6 * x + 11 ∧ x = 5 / 3 := by
  sorry

end equation_solution_l2901_290135


namespace vacation_fund_adjustment_l2901_290182

/-- Calculates the required hours per week to earn a target amount given initial conditions and unexpected events --/
theorem vacation_fund_adjustment (initial_weeks : ℕ) (initial_hours_per_week : ℝ) (sick_weeks : ℕ) (target_amount : ℝ) :
  let remaining_weeks := initial_weeks - sick_weeks
  let total_hours := initial_weeks * initial_hours_per_week
  let hourly_rate := target_amount / total_hours
  let required_hours_per_week := (target_amount / hourly_rate) / remaining_weeks
  required_hours_per_week = 31.25 := by
  sorry

end vacation_fund_adjustment_l2901_290182


namespace truncated_cube_edges_l2901_290157

/-- Represents a cube with truncated corners -/
structure TruncatedCube where
  /-- The number of vertices in the original cube -/
  originalVertices : Nat
  /-- The number of edges in the original cube -/
  originalEdges : Nat
  /-- The fraction of each edge removed by truncation -/
  truncationFraction : Rat
  /-- The number of edges affected by truncation at each vertex -/
  edgesAffectedPerVertex : Nat
  /-- The number of new edges created by truncation at each vertex -/
  newEdgesPerVertex : Nat

/-- The number of edges in a truncated cube -/
def edgesInTruncatedCube (c : TruncatedCube) : Nat :=
  c.originalEdges + c.originalVertices * c.newEdgesPerVertex

/-- Theorem stating that a cube with truncated corners has 36 edges -/
theorem truncated_cube_edges :
  ∀ (c : TruncatedCube),
    c.originalVertices = 8 ∧
    c.originalEdges = 12 ∧
    c.truncationFraction = 1/4 ∧
    c.edgesAffectedPerVertex = 2 ∧
    c.newEdgesPerVertex = 3 →
    edgesInTruncatedCube c = 36 :=
by sorry

end truncated_cube_edges_l2901_290157


namespace stating_count_line_segments_correct_l2901_290131

/-- Represents a regular n-sided convex polygon with n exterior points. -/
structure PolygonWithExteriorPoints (n : ℕ) where
  -- n ≥ 3 to ensure it's a valid polygon
  valid : n ≥ 3

/-- 
Calculates the number of line segments that can be drawn between all pairs 
of interior and exterior points of a regular n-sided convex polygon, 
excluding those connecting adjacent vertices.
-/
def countLineSegments (p : PolygonWithExteriorPoints n) : ℕ :=
  (n * (n - 3)) / 2 + n + n * (n - 3)

/-- 
Theorem stating that the number of line segments is correctly calculated 
by the formula (n(n-3)/2) + n + n(n-3).
-/
theorem count_line_segments_correct (p : PolygonWithExteriorPoints n) :
  countLineSegments p = (n * (n - 3)) / 2 + n + n * (n - 3) := by
  sorry

end stating_count_line_segments_correct_l2901_290131


namespace water_added_to_tank_l2901_290172

/-- The amount of water added to a tank -/
def water_added (capacity : ℚ) (initial_fraction : ℚ) (final_fraction : ℚ) : ℚ :=
  capacity * (final_fraction - initial_fraction)

/-- Theorem: The amount of water added to a 40-gallon tank, 
    initially 3/4 full and ending up 7/8 full, is 5 gallons -/
theorem water_added_to_tank : 
  water_added 40 (3/4) (7/8) = 5 := by
  sorry

end water_added_to_tank_l2901_290172


namespace john_uber_profit_l2901_290185

/-- Calculates the net profit of an Uber driver given their income and expenses --/
def uberDriverNetProfit (grossIncome : ℕ) (carPurchasePrice : ℕ) (monthlyMaintenance : ℕ) 
  (maintenancePeriod : ℕ) (annualInsurance : ℕ) (tireReplacement : ℕ) (tradeInValue : ℕ) 
  (taxRate : ℚ) : ℤ :=
  let totalMaintenance := monthlyMaintenance * maintenancePeriod
  let taxAmount := (grossIncome : ℚ) * taxRate
  let totalExpenses := carPurchasePrice + totalMaintenance + annualInsurance + tireReplacement + taxAmount.ceil
  (grossIncome : ℤ) - (totalExpenses : ℤ) + (tradeInValue : ℤ)

/-- Theorem stating that John's net profit as an Uber driver is $6,300 --/
theorem john_uber_profit : 
  uberDriverNetProfit 30000 20000 300 12 1200 400 6000 (15/100) = 6300 := by
  sorry

end john_uber_profit_l2901_290185


namespace inequalities_theorem_l2901_290107

theorem inequalities_theorem (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  (2 * a * b) / (a + b) ≤ (a + b) / 2 ∧
  Real.sqrt (a * b) ≤ (a + b) / 2 ∧
  (a + b) / 2 ≤ Real.sqrt ((a^2 + b^2) / 2) ∧
  b^2 / a + a^2 / b ≥ a + b :=
by sorry

end inequalities_theorem_l2901_290107


namespace sqrt_equation_solution_l2901_290153

theorem sqrt_equation_solution (x : ℝ) : 
  x > 2 → (Real.sqrt (8 * x) / Real.sqrt (4 * (x - 2)) = 5 / 2) → x = 50 / 17 := by
  sorry

end sqrt_equation_solution_l2901_290153


namespace simplify_cube_root_l2901_290128

theorem simplify_cube_root (a b : ℝ) (h : a < 0) : 
  Real.sqrt (a^3 * b) = -a * Real.sqrt (a * b) := by
  sorry

end simplify_cube_root_l2901_290128


namespace stock_price_increase_l2901_290114

theorem stock_price_increase (opening_price : ℝ) (increase_percentage : ℝ) : 
  opening_price = 10 → increase_percentage = 0.5 → 
  opening_price * (1 + increase_percentage) = 15 := by
  sorry

#check stock_price_increase

end stock_price_increase_l2901_290114


namespace bakery_rolls_distribution_l2901_290159

theorem bakery_rolls_distribution (n k : ℕ) (h1 : n = 4) (h2 : k = 3) :
  Nat.choose (n + k - 1) (k - 1) = 15 := by
  sorry

end bakery_rolls_distribution_l2901_290159


namespace walking_speed_problem_l2901_290198

/-- Given two people walking in the same direction for 10 hours, where one walks at 7.5 kmph
    and they end up 20 km apart, prove that the speed of the other person is 9.5 kmph. -/
theorem walking_speed_problem (v : ℝ) 
  (h1 : (v - 7.5) * 10 = 20) : v = 9.5 := by
  sorry

end walking_speed_problem_l2901_290198


namespace intersection_theorem_subset_theorem_l2901_290197

def A : Set ℝ := {x | x^2 - 2*x - 3 ≤ 0}
def B (m : ℝ) : Set ℝ := {x | x^2 - 2*m*x - 4 ≤ 0}

theorem intersection_theorem (m : ℝ) :
  A ∩ B m = {x | 1 ≤ x ∧ x ≤ 3} → m = 3 := by sorry

theorem subset_theorem (m : ℝ) :
  A ⊆ (Set.univ \ B m) → m < -3 ∨ m > 5 := by sorry

end intersection_theorem_subset_theorem_l2901_290197


namespace expression_independence_l2901_290161

theorem expression_independence (x a b c : ℝ) 
  (hxa : x ≠ a) (hxb : x ≠ b) (hxc : x ≠ c) : 
  (x - a) * (x - b) * (x - c) * 
  ((a - b) / (x - c) + (b - c) / (x - a) + (c - a) / (x - b)) = 
  (b - a) * (a - c) * (c - b) := by
  sorry

end expression_independence_l2901_290161


namespace regular_polygon_diagonals_l2901_290104

theorem regular_polygon_diagonals (n : ℕ) (h : n > 2) :
  (n * (n - 3) / 2 : ℚ) = 2 * n → n = 7 := by
  sorry

end regular_polygon_diagonals_l2901_290104


namespace f_min_max_l2901_290136

def f (x : ℝ) : ℝ := -2 * x + 1

theorem f_min_max :
  let a : ℝ := -2
  let b : ℝ := 2
  (∀ x ∈ Set.Icc a b, f x ≥ -3) ∧
  (∃ x ∈ Set.Icc a b, f x = -3) ∧
  (∀ x ∈ Set.Icc a b, f x ≤ 5) ∧
  (∃ x ∈ Set.Icc a b, f x = 5) :=
by sorry

end f_min_max_l2901_290136


namespace B_power_99_l2901_290142

def B : Matrix (Fin 3) (Fin 3) ℝ := !![0, 0, 0; 0, 0, 1; 0, -1, 0]

theorem B_power_99 : B^99 = !![0, 0, 0; 0, 0, -1; 0, 1, 0] := by sorry

end B_power_99_l2901_290142


namespace geometric_arithmetic_relation_l2901_290170

/-- A geometric sequence -/
def geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = r * a n

/-- An arithmetic sequence -/
def arithmetic_sequence (b : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, b (n + 1) = b n + d

theorem geometric_arithmetic_relation (a b : ℕ → ℝ) :
  geometric_sequence a →
  arithmetic_sequence b →
  a 3 * a 11 = 4 * a 7 →
  a 7 = b 7 →
  b 5 + b 9 = 8 := by
sorry

end geometric_arithmetic_relation_l2901_290170


namespace canteen_banana_units_l2901_290146

/-- Represents the number of bananas in a unit -/
def bananas_per_unit (daily_units : ℕ) (total_bananas : ℕ) (weeks : ℕ) : ℕ :=
  (total_bananas / (weeks * 7)) / daily_units

/-- Theorem stating that given the conditions, each unit consists of 12 bananas -/
theorem canteen_banana_units :
  bananas_per_unit 13 9828 9 = 12 := by
  sorry

end canteen_banana_units_l2901_290146


namespace P_divisibility_l2901_290112

/-- The polynomial P(x) -/
def P (a x : ℝ) : ℝ := a^3 * x^5 + (1 - a) * x^4 + (1 + a^3) * x^2 + (1 - 3*a) * x - a^3

/-- The set of values of a for which P(x) is divisible by (x-1) -/
def A : Set ℝ := {a | ∃ q : ℝ → ℝ, ∀ x, P a x = (x - 1) * q x}

theorem P_divisibility :
  A = {1, (-1 + Real.sqrt 13) / 2, (-1 - Real.sqrt 13) / 2} :=
sorry

end P_divisibility_l2901_290112


namespace sum_of_triangles_34_l2901_290199

/-- The triangle operation defined as a * b - c -/
def triangle_op (a b c : ℕ) : ℕ := a * b - c

/-- Theorem stating that the sum of two specific triangle operations equals 34 -/
theorem sum_of_triangles_34 : triangle_op 3 5 2 + triangle_op 4 6 3 = 34 := by
  sorry

end sum_of_triangles_34_l2901_290199


namespace sum_equality_seven_eight_l2901_290100

theorem sum_equality_seven_eight (S : Finset ℤ) (h : S.card = 15) :
  {s | ∃ (T : Finset ℤ), T ⊆ S ∧ T.card = 7 ∧ s = T.sum id} =
  {s | ∃ (T : Finset ℤ), T ⊆ S ∧ T.card = 8 ∧ s = T.sum id} :=
by sorry

end sum_equality_seven_eight_l2901_290100


namespace sons_age_l2901_290144

theorem sons_age (son_age father_age : ℕ) : 
  father_age = 6 * son_age →
  father_age + 6 + son_age + 6 = 68 →
  son_age = 8 := by
sorry

end sons_age_l2901_290144


namespace geometric_sequence_middle_term_l2901_290154

-- Define a geometric sequence of three terms
def is_geometric_sequence (a b c : ℝ) : Prop := b * b = a * c

-- Theorem statement
theorem geometric_sequence_middle_term :
  ∀ m : ℝ, is_geometric_sequence 1 m 4 → m = 2 ∨ m = -2 := by
  sorry

end geometric_sequence_middle_term_l2901_290154


namespace sum_product_equal_470_l2901_290155

theorem sum_product_equal_470 : 
  (4.7 * 13.26 + 4.7 * 9.43 + 4.7 * 77.31) = 470 := by
  sorry

end sum_product_equal_470_l2901_290155


namespace event_children_count_l2901_290192

/-- Calculates the number of children at an event after adding more children --/
theorem event_children_count (total_guests men_count added_children : ℕ) : 
  total_guests = 80 →
  men_count = 40 →
  added_children = 10 →
  let women_count := men_count / 2
  let initial_children := total_guests - (men_count + women_count)
  initial_children + added_children = 30 := by
  sorry

#check event_children_count

end event_children_count_l2901_290192


namespace transformed_line_equation_l2901_290119

/-- Given a line and a scaling transformation, prove the equation of the transformed line -/
theorem transformed_line_equation (x y x' y' : ℝ) :
  (x - 2 * y = 2) →  -- Original line equation
  (x' = x) →         -- Scaling transformation for x
  (y' = 2 * y) →     -- Scaling transformation for y
  (x' - y' - 2 = 0)  -- Resulting line equation
:= by sorry

end transformed_line_equation_l2901_290119


namespace number_of_math_classes_school_play_volunteers_l2901_290163

/-- Given information about volunteers for a school Christmas play, prove the number of participating math classes. -/
theorem number_of_math_classes (total_needed : ℕ) (students_per_class : ℕ) (teachers : ℕ) (more_needed : ℕ) : ℕ :=
  let current_volunteers := total_needed - more_needed
  let x := (current_volunteers - teachers) / students_per_class
  x

/-- Prove that the number of math classes participating is 6. -/
theorem school_play_volunteers : number_of_math_classes 50 5 13 7 = 6 := by
  sorry

end number_of_math_classes_school_play_volunteers_l2901_290163


namespace arctan_tan_difference_l2901_290130

/-- Proves that arctan(tan 75° - 3 tan 30°) is approximately 124.1°. -/
theorem arctan_tan_difference (ε : ℝ) (h : ε > 0) :
  ∃ θ : ℝ, 0 ≤ θ ∧ θ ≤ 180 ∧ |θ - 124.1| < ε ∧ θ = Real.arctan (Real.tan (75 * π / 180) - 3 * Real.tan (30 * π / 180)) * 180 / π :=
sorry

end arctan_tan_difference_l2901_290130


namespace elaine_jerry_ratio_l2901_290122

/-- Represents the time spent in the pool by each person --/
structure PoolTime where
  jerry : ℚ
  elaine : ℚ
  george : ℚ
  kramer : ℚ

/-- Conditions of the problem --/
def pool_conditions (t : PoolTime) : Prop :=
  t.jerry = 3 ∧
  t.george = t.elaine / 3 ∧
  t.kramer = 0 ∧
  t.jerry + t.elaine + t.george + t.kramer = 11

/-- The theorem to be proved --/
theorem elaine_jerry_ratio (t : PoolTime) :
  pool_conditions t → t.elaine / t.jerry = 2 := by
  sorry


end elaine_jerry_ratio_l2901_290122


namespace ahmed_has_thirteen_goats_l2901_290164

/-- The number of goats Adam has -/
def adam_goats : ℕ := 7

/-- The number of goats Andrew has -/
def andrew_goats : ℕ := 5 + 2 * adam_goats

/-- The number of goats Ahmed has -/
def ahmed_goats : ℕ := andrew_goats - 6

/-- Theorem stating that Ahmed has 13 goats -/
theorem ahmed_has_thirteen_goats : ahmed_goats = 13 := by
  sorry

end ahmed_has_thirteen_goats_l2901_290164


namespace sum_of_digits_of_sum_of_digits_of_1962_digit_number_div_by_9_l2901_290118

def is_1962_digit (n : ℕ) : Prop := 10^1961 ≤ n ∧ n < 10^1962

def sum_of_digits (n : ℕ) : ℕ := sorry

theorem sum_of_digits_of_sum_of_digits_of_1962_digit_number_div_by_9 
  (n : ℕ) 
  (h1 : is_1962_digit n) 
  (h2 : n % 9 = 0) : 
  let a := sum_of_digits n
  let b := sum_of_digits a
  let c := sum_of_digits b
  c = 9 := by sorry

end sum_of_digits_of_sum_of_digits_of_1962_digit_number_div_by_9_l2901_290118


namespace intersection_of_M_and_N_l2901_290186

def M : Set Int := {-1, 1, -2, 2}
def N : Set Int := {1, 4}

theorem intersection_of_M_and_N :
  M ∩ N = {1} := by sorry

end intersection_of_M_and_N_l2901_290186


namespace frisbee_sales_receipts_l2901_290166

/-- Represents the total receipts from frisbee sales for a week -/
def total_receipts (x y : ℕ) : ℕ := 3 * x + 4 * y

/-- Theorem stating that the total receipts from frisbee sales for the week is $200 -/
theorem frisbee_sales_receipts :
  ∃ (x y : ℕ), x + y = 60 ∧ y ≥ 20 ∧ total_receipts x y = 200 := by
  sorry

end frisbee_sales_receipts_l2901_290166


namespace min_value_x_plus_y_l2901_290156

theorem min_value_x_plus_y (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : x + 16 * y = x * y) :
  x + y ≥ 25 ∧ ∃ x₀ y₀ : ℝ, x₀ > 0 ∧ y₀ > 0 ∧ x₀ + 16 * y₀ = x₀ * y₀ ∧ x₀ + y₀ = 25 := by
  sorry

end min_value_x_plus_y_l2901_290156


namespace derivative_properties_neg_l2901_290191

open Real

-- Define the properties of functions f and g
def even_function (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = f x

def odd_function (g : ℝ → ℝ) : Prop := ∀ x, g (-x) = -g x

def positive_derivative_pos (f : ℝ → ℝ) : Prop := ∀ x > 0, deriv f x > 0

def negative_derivative_pos (g : ℝ → ℝ) : Prop := ∀ x > 0, deriv g x < 0

-- State the theorem
theorem derivative_properties_neg
  (f g : ℝ → ℝ)
  (hf : Differentiable ℝ f)
  (hg : Differentiable ℝ g)
  (hf_even : even_function f)
  (hg_odd : odd_function g)
  (hf_pos : positive_derivative_pos f)
  (hg_pos : negative_derivative_pos g) :
  ∀ x < 0, deriv f x < 0 ∧ deriv g x < 0 :=
sorry

end derivative_properties_neg_l2901_290191


namespace monotone_increasing_condition_monotone_increasing_sufficiency_l2901_290121

/-- A function f is monotonically increasing on an interval (a, +∞) if for any x₁, x₂ in the interval
    where x₁ < x₂, we have f(x₁) < f(x₂) -/
def MonotonicallyIncreasing (f : ℝ → ℝ) (a : ℝ) :=
  ∀ x₁ x₂, a < x₁ ∧ x₁ < x₂ → f x₁ < f x₂

/-- The function f(x) = x^2 + mx - 2 -/
def f (m : ℝ) (x : ℝ) : ℝ := x^2 + m*x - 2

/-- Theorem: If f(x) = x^2 + mx - 2 is monotonically increasing on (2, +∞), then m ≥ -4 -/
theorem monotone_increasing_condition (m : ℝ) :
  MonotonicallyIncreasing (f m) 2 → m ≥ -4 := by
  sorry

/-- Theorem: If m ≥ -4, then f(x) = x^2 + mx - 2 is monotonically increasing on (2, +∞) -/
theorem monotone_increasing_sufficiency (m : ℝ) :
  m ≥ -4 → MonotonicallyIncreasing (f m) 2 := by
  sorry

end monotone_increasing_condition_monotone_increasing_sufficiency_l2901_290121


namespace expression_evaluation_l2901_290187

theorem expression_evaluation :
  let a : ℤ := -1
  let b : ℤ := 2
  3 * (a^2 * b + a * b^2) - 2 * (a^2 * b - 1) - 2 * a * b^2 - 2 = -2 := by
  sorry

end expression_evaluation_l2901_290187


namespace repeating_decimal_sum_l2901_290165

theorem repeating_decimal_sum (c d : ℕ) : 
  (5 : ℚ) / 13 = (c * 10 + d : ℚ) / 99 → c + d = 11 := by
  sorry

end repeating_decimal_sum_l2901_290165


namespace pascal_triangle_24th_row_20th_number_l2901_290117

theorem pascal_triangle_24th_row_20th_number : 
  (Nat.choose 24 19) = 42504 := by sorry

end pascal_triangle_24th_row_20th_number_l2901_290117


namespace ellipse_properties_l2901_290175

/-- Given an ellipse C with equation (x^2 / a^2) + (y^2 / b^2) = 1, where a > b > 0,
    eccentricity 1/2, and the area of the quadrilateral formed by its vertices is 4√3,
    we prove properties about its equation and intersecting lines. -/
theorem ellipse_properties (a b : ℝ) (h1 : a > b) (h2 : b > 0) :
  let e := 1 / 2  -- eccentricity
  let quad_area := 4 * Real.sqrt 3  -- area of quadrilateral formed by vertices
  ∀ x y : ℝ,
    (x^2 / a^2 + y^2 / b^2 = 1) →  -- equation of ellipse C
    (e = Real.sqrt (1 - b^2 / a^2)) →  -- definition of eccentricity
    (quad_area = 4 * a * b) →  -- area of quadrilateral
    (∀ x₁ y₁ x₂ y₂ : ℝ,
      (x₁^2 / a^2 + y₁^2 / b^2 = 1) →  -- P(x₁, y₁) on ellipse
      (x₂^2 / a^2 + y₂^2 / b^2 = 1) →  -- Q(x₂, y₂) on ellipse
      (1/2 * |x₁ * y₂ - x₂ * y₁| = Real.sqrt 3) →  -- area of triangle OPQ is √3
      (x₁^2 / 4 + y₁^2 / 3 = 1) ∧  -- equation of ellipse C
      (x₂^2 / 4 + y₂^2 / 3 = 1) ∧  -- equation of ellipse C
      (x₁^2 + x₂^2 = 4))  -- constant sum of squares
  := by sorry

end ellipse_properties_l2901_290175


namespace rug_inner_length_is_four_l2901_290105

/-- Represents the dimensions of a rectangular region -/
structure RectDimensions where
  length : ℝ
  width : ℝ

/-- Calculates the area of a rectangle given its dimensions -/
def area (d : RectDimensions) : ℝ := d.length * d.width

/-- Represents the three colored regions of the rug -/
structure RugRegions where
  inner : RectDimensions
  middle : RectDimensions
  outer : RectDimensions

/-- Checks if three real numbers form an arithmetic progression -/
def isArithmeticProgression (a b c : ℝ) : Prop := b - a = c - b

theorem rug_inner_length_is_four :
  ∀ (r : RugRegions),
    r.inner.width = 2 →
    r.middle.length = r.inner.length + 4 →
    r.middle.width = r.inner.width + 4 →
    r.outer.length = r.middle.length + 4 →
    r.outer.width = r.middle.width + 4 →
    isArithmeticProgression (area r.inner) (area r.middle - area r.inner) (area r.outer - area r.middle) →
    r.inner.length = 4 := by
  sorry

end rug_inner_length_is_four_l2901_290105


namespace power_of_product_l2901_290103

theorem power_of_product (a b : ℝ) : (a * b^2)^3 = a^3 * b^6 := by
  sorry

end power_of_product_l2901_290103


namespace logarithm_sum_l2901_290169

theorem logarithm_sum (a b : ℝ) (ha : a = Real.log 8) (hb : b = Real.log 25) :
  5^(a/b) + 2^(b/a) = 2 * Real.sqrt 2 + 5^(2/3) := by
  sorry

end logarithm_sum_l2901_290169


namespace max_cards_purchasable_l2901_290132

def initial_money : ℚ := 965 / 100
def earned_money : ℚ := 535 / 100
def card_cost : ℚ := 95 / 100

theorem max_cards_purchasable : 
  ⌊(initial_money + earned_money) / card_cost⌋ = 15 := by sorry

end max_cards_purchasable_l2901_290132


namespace min_value_expression_min_value_attainable_l2901_290140

theorem min_value_expression (a b c : ℝ) 
  (h1 : 1 ≤ a) (h2 : a ≤ b) (h3 : b ≤ c) (h4 : c ≤ 4) :
  (a - 1)^2 + (b/a - 1)^2 + (c/b - 1)^2 + (4/c - 1)^2 ≥ 12 - 8 * Real.sqrt 2 :=
by sorry

theorem min_value_attainable :
  ∃ a b c : ℝ, 1 ≤ a ∧ a ≤ b ∧ b ≤ c ∧ c ≤ 4 ∧
  (a - 1)^2 + (b/a - 1)^2 + (c/b - 1)^2 + (4/c - 1)^2 = 12 - 8 * Real.sqrt 2 :=
by sorry

end min_value_expression_min_value_attainable_l2901_290140


namespace perpendicular_to_same_plane_implies_parallel_l2901_290115

-- Define a plane in 3D space
def Plane : Type := ℝ → ℝ → ℝ → Prop

-- Define a line in 3D space
def Line : Type := ℝ → ℝ × ℝ × ℝ

-- Define perpendicularity between a line and a plane
def perpendicular (l : Line) (p : Plane) : Prop := sorry

-- Define parallelism between two lines
def parallel (l1 l2 : Line) : Prop := sorry

-- Theorem statement
theorem perpendicular_to_same_plane_implies_parallel 
  (l1 l2 : Line) (p : Plane) 
  (h1 : perpendicular l1 p) (h2 : perpendicular l2 p) : 
  parallel l1 l2 := by sorry

end perpendicular_to_same_plane_implies_parallel_l2901_290115


namespace evaluate_expression_l2901_290174

theorem evaluate_expression : 
  2100^3 - 2 * 2099 * 2100^2 - 2099^2 * 2100 + 2099^3 = 4404902 := by
  sorry

end evaluate_expression_l2901_290174


namespace sum_of_roots_equals_target_l2901_290180

/-- A function f satisfying the given condition for all non-zero real x -/
noncomputable def f : ℝ → ℝ := sorry

/-- The condition that f satisfies for all non-zero real x -/
axiom f_condition (x : ℝ) (hx : x ≠ 0) : 2 * f x + f (1 / x) = 5 * x + 4

/-- The value we're looking for -/
def target_value : ℝ := 2004

/-- The theorem to prove -/
theorem sum_of_roots_equals_target (x : ℝ) :
  let a : ℝ := 1
  let b : ℝ := -((3 * target_value - 4) / 10)
  let c : ℝ := 5 / 2
  x^2 + b*x + c = 0 → x + (-b/a) = (3 * target_value - 4) / 10 :=
by sorry

end sum_of_roots_equals_target_l2901_290180


namespace cake_pieces_l2901_290162

theorem cake_pieces (cake_length : ℕ) (cake_width : ℕ) (piece_length : ℕ) (piece_width : ℕ) :
  cake_length = 24 →
  cake_width = 20 →
  piece_length = 3 →
  piece_width = 2 →
  (cake_length * cake_width) / (piece_length * piece_width) = 80 :=
by
  sorry

end cake_pieces_l2901_290162


namespace no_solutions_prime_equation_l2901_290139

theorem no_solutions_prime_equation (p a n : ℕ) : 
  Prime p → a > 0 → n > 0 → p^a - 1 ≠ 2^n * (p - 1) := by
  sorry

end no_solutions_prime_equation_l2901_290139


namespace largest_circle_at_a_l2901_290151

/-- A pentagon with circles centered at each vertex -/
structure PentagonWithCircles where
  -- Lengths of the pentagon sides
  ab : ℝ
  bc : ℝ
  cd : ℝ
  de : ℝ
  ea : ℝ
  -- Radii of the circles
  r_a : ℝ
  r_b : ℝ
  r_c : ℝ
  r_d : ℝ
  r_e : ℝ
  -- Conditions for circles touching on sides
  h_ab : r_a + r_b = ab
  h_bc : r_b + r_c = bc
  h_cd : r_c + r_d = cd
  h_de : r_d + r_e = de
  h_ea : r_e + r_a = ea

/-- The circle centered at A has the largest radius -/
theorem largest_circle_at_a (p : PentagonWithCircles)
  (h_ab : p.ab = 16)
  (h_bc : p.bc = 14)
  (h_cd : p.cd = 17)
  (h_de : p.de = 13)
  (h_ea : p.ea = 14) :
  p.r_a = max p.r_a (max p.r_b (max p.r_c (max p.r_d p.r_e))) :=
by sorry

end largest_circle_at_a_l2901_290151


namespace max_value_a_inequality_l2901_290138

theorem max_value_a_inequality (a : ℝ) : 
  (∀ x₁ x₂ : ℝ, 1 ≤ x₁ ∧ x₁ < x₂ ∧ x₂ ≤ π/2 → 
    (x₂ * Real.sin x₁ - x₁ * Real.sin x₂) / (x₁ - x₂) > a) →
  a ≤ -1 :=
by sorry

end max_value_a_inequality_l2901_290138


namespace rebus_solution_l2901_290167

theorem rebus_solution : ∃! (a b c d : ℕ),
  (a ≠ 0 ∧ b ≠ 0 ∧ c ≠ 0 ∧ d ≠ 0) ∧
  (a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d) ∧
  (1000 * a + 100 * b + 10 * c + a = 182 * (10 * c + d)) ∧
  (a = 2 ∧ b = 9 ∧ c = 1 ∧ d = 6) :=
by sorry

end rebus_solution_l2901_290167


namespace engineers_teachers_ratio_l2901_290149

theorem engineers_teachers_ratio (e t : ℕ) (he : e > 0) (ht : t > 0) :
  (40 * e + 55 * t : ℚ) / (e + t) = 46 →
  e / t = 3 / 2 := by
sorry

end engineers_teachers_ratio_l2901_290149


namespace pegboard_empty_holes_l2901_290141

/-- Represents a square pegboard -/
structure Pegboard :=
  (size : ℕ)

/-- Calculates the total number of holes on the pegboard -/
def total_holes (p : Pegboard) : ℕ := (p.size + 1) ^ 2

/-- Calculates the number of holes with pegs (on diagonals) -/
def holes_with_pegs (p : Pegboard) : ℕ := 2 * (p.size + 1) - 1

/-- Calculates the number of empty holes on the pegboard -/
def empty_holes (p : Pegboard) : ℕ := total_holes p - holes_with_pegs p

theorem pegboard_empty_holes :
  ∃ (p : Pegboard), p.size = 10 ∧ empty_holes p = 100 :=
sorry

end pegboard_empty_holes_l2901_290141


namespace decimal_place_values_l2901_290148

/-- Represents the place value in a decimal number system. -/
inductive PlaceValue
| Ones
| Tens
| Hundreds
| Thousands
| TenThousands
| HundredThousands
| Millions
| TenMillions
| HundredMillions

/-- Returns the position of a place value from right to left. -/
def position (pv : PlaceValue) : Nat :=
  match pv with
  | .Ones => 1
  | .Tens => 2
  | .Hundreds => 3
  | .Thousands => 4
  | .TenThousands => 5
  | .HundredThousands => 6
  | .Millions => 7
  | .TenMillions => 8
  | .HundredMillions => 9

theorem decimal_place_values :
  (position PlaceValue.Hundreds = 3) ∧
  (position PlaceValue.TenThousands = 5) ∧
  (position PlaceValue.Thousands = 4) := by
  sorry

end decimal_place_values_l2901_290148


namespace cos_alpha_minus_beta_l2901_290108

theorem cos_alpha_minus_beta (α β : ℝ) 
  (h1 : 2 * Real.cos α - Real.cos β = (3 : ℝ) / 2)
  (h2 : 2 * Real.sin α - Real.sin β = 2) :
  Real.cos (α - β) = -(5 : ℝ) / 16 := by
  sorry

end cos_alpha_minus_beta_l2901_290108


namespace m_zero_sufficient_not_necessary_l2901_290123

/-- Represents a quadratic equation in two variables -/
structure QuadraticEquation where
  a : ℝ
  b : ℝ
  c : ℝ
  d : ℝ
  e : ℝ
  f : ℝ

/-- Checks if a quadratic equation represents a circle -/
def isCircle (eq : QuadraticEquation) : Prop :=
  eq.a = eq.b ∧ eq.a ≠ 0 ∧ eq.c^2 + eq.d^2 - 4 * eq.a * eq.f > 0

/-- The specific equation x^2 + y^2 - 4x + 2y + m = 0 -/
def specificEquation (m : ℝ) : QuadraticEquation :=
  { a := 1, b := 1, c := -4, d := 2, e := 0, f := m }

/-- Theorem stating that m = 0 is sufficient but not necessary for the equation to represent a circle -/
theorem m_zero_sufficient_not_necessary :
  (∀ m : ℝ, m = 0 → isCircle (specificEquation m)) ∧
  ¬(∀ m : ℝ, isCircle (specificEquation m) → m = 0) :=
sorry

end m_zero_sufficient_not_necessary_l2901_290123


namespace cards_playing_with_l2901_290127

/-- The number of cards in a standard deck --/
def standard_deck : Nat := 52

/-- The number of cards kept away --/
def cards_kept_away : Nat := 7

/-- Theorem: The number of cards they were playing with is 45 --/
theorem cards_playing_with : 
  standard_deck - cards_kept_away = 45 := by
  sorry

end cards_playing_with_l2901_290127


namespace probability_odd_divisor_25_factorial_l2901_290152

theorem probability_odd_divisor_25_factorial (n : ℕ) (h : n = 25) :
  let factorial := n.factorial
  let total_divisors := (factorial.divisors.filter (· > 0)).card
  let odd_divisors := (factorial.divisors.filter (λ d => d > 0 ∧ d % 2 = 1)).card
  (odd_divisors : ℚ) / total_divisors = 1 / 23 := by
  sorry

end probability_odd_divisor_25_factorial_l2901_290152


namespace arithmetic_fraction_subtraction_l2901_290102

theorem arithmetic_fraction_subtraction :
  (1 + 3 + 5 + 7) / (2 + 4 + 6 + 8) - (2 + 4 + 6 + 8) / (1 + 3 + 5 + 7) = -9 / 20 := by
  sorry

end arithmetic_fraction_subtraction_l2901_290102


namespace final_running_distance_l2901_290188

/-- Calculates the final daily running distance after a 5-week program -/
theorem final_running_distance
  (initial_distance : ℕ)  -- Initial daily running distance in miles
  (increase_rate : ℕ)     -- Weekly increase in miles
  (increase_weeks : ℕ)    -- Number of weeks with distance increase
  (h1 : initial_distance = 3)
  (h2 : increase_rate = 1)
  (h3 : increase_weeks = 4)
  : initial_distance + increase_rate * increase_weeks = 7 :=
by sorry

end final_running_distance_l2901_290188


namespace systematic_sampling_seat_number_l2901_290111

/-- Systematic sampling function that returns the seat numbers in the sample -/
def systematicSample (totalStudents : ℕ) (sampleSize : ℕ) : List ℕ :=
  sorry

theorem systematic_sampling_seat_number
  (totalStudents : ℕ) (sampleSize : ℕ) (knownSeats : List ℕ) :
  totalStudents = 52 →
  sampleSize = 4 →
  knownSeats = [3, 29, 42] →
  let sample := systematicSample totalStudents sampleSize
  (∀ s ∈ knownSeats, s ∈ sample) →
  ∃ s ∈ sample, s = 16 ∧ s ∉ knownSeats :=
by sorry

end systematic_sampling_seat_number_l2901_290111


namespace frog_eyes_count_l2901_290124

/-- The number of frogs in the pond -/
def num_frogs : ℕ := 6

/-- The number of eyes each frog has -/
def eyes_per_frog : ℕ := 2

/-- The total number of frog eyes in the pond -/
def total_frog_eyes : ℕ := num_frogs * eyes_per_frog

theorem frog_eyes_count : total_frog_eyes = 12 := by
  sorry

end frog_eyes_count_l2901_290124


namespace reflection_of_circle_center_l2901_290194

/-- Reflects a point (x, y) about the line y = -x -/
def reflect_about_y_eq_neg_x (p : ℝ × ℝ) : ℝ × ℝ :=
  (- p.2, - p.1)

/-- The original center of the circle -/
def original_center : ℝ × ℝ := (3, -4)

/-- The expected center after reflection -/
def expected_reflected_center : ℝ × ℝ := (4, -3)

theorem reflection_of_circle_center :
  reflect_about_y_eq_neg_x original_center = expected_reflected_center :=
by sorry

end reflection_of_circle_center_l2901_290194


namespace total_area_form_and_sum_l2901_290113

/-- Represents a rectangular prism with dimensions 1 × 1 × 2 -/
structure RectangularPrism :=
  (length : ℝ := 1)
  (width : ℝ := 1)
  (height : ℝ := 2)

/-- Represents a triangle with vertices from the rectangular prism -/
structure PrismTriangle :=
  (vertices : Fin 3 → Fin 8)

/-- Calculates the area of a PrismTriangle -/
def triangleArea (prism : RectangularPrism) (triangle : PrismTriangle) : ℝ :=
  sorry

/-- The sum of areas of all triangles whose vertices are vertices of the prism -/
def totalTriangleArea (prism : RectangularPrism) : ℝ :=
  sorry

/-- Theorem stating the form of the total area and the sum of m, n, and p -/
theorem total_area_form_and_sum (prism : RectangularPrism) :
  ∃ (m n p : ℕ), totalTriangleArea prism = m + Real.sqrt n + Real.sqrt p ∧ m + n + p = 100 :=
sorry

end total_area_form_and_sum_l2901_290113


namespace class_size_l2901_290195

theorem class_size (average_weight : ℝ) (teacher_weight : ℝ) (new_average : ℝ) :
  average_weight = 35 →
  teacher_weight = 45 →
  new_average = 35.4 →
  ∃ n : ℕ, (n : ℝ) * average_weight + teacher_weight = new_average * ((n : ℝ) + 1) ∧ n = 24 :=
by sorry

end class_size_l2901_290195


namespace problem_solution_l2901_290184

theorem problem_solution (a b c d e : ℝ) 
  (h1 : a * b = 1)  -- a and b are reciprocals
  (h2 : c + d = 0)  -- c and d are opposites
  (h3 : e < 0)      -- e is negative
  (h4 : |e| = 1)    -- absolute value of e is 1
  : (-a*b)^2009 - (c+d)^2010 - e^2011 = 0 := by
  sorry

end problem_solution_l2901_290184


namespace clerical_staff_reduction_l2901_290133

theorem clerical_staff_reduction (total_employees : ℕ) 
  (initial_clerical_fraction : ℚ) (final_clerical_fraction : ℚ) 
  (h1 : total_employees = 3600)
  (h2 : initial_clerical_fraction = 1/3)
  (h3 : final_clerical_fraction = 1/5) : 
  ∃ (f : ℚ), 
    (initial_clerical_fraction * total_employees) * (1 - f) = 
    final_clerical_fraction * (total_employees - initial_clerical_fraction * total_employees * f) ∧ 
    f = 1/2 := by
  sorry

end clerical_staff_reduction_l2901_290133


namespace sqrt_x4_eq_x2_l2901_290101

theorem sqrt_x4_eq_x2 : ∀ x : ℝ, Real.sqrt (x^4) = x^2 := by sorry

end sqrt_x4_eq_x2_l2901_290101


namespace farmer_boso_animals_l2901_290116

theorem farmer_boso_animals (a b : ℕ) (h1 : 5 * b = b^(a-5)) (h2 : b = 5) (h3 : a = 7) : ∃ (L : ℕ), L = 3 ∧ 
  (4 * (5 * b) + 2 * (5 * a + 7) + 6 * b^(a-5) = 100 * L + 10 * L + L + 1) :=
sorry

end farmer_boso_animals_l2901_290116


namespace michael_earnings_l2901_290168

/-- Calculates the total money earned from selling birdhouses --/
def total_money_earned (large_price medium_price small_price : ℕ) 
                       (large_sold medium_sold small_sold : ℕ) : ℕ :=
  large_price * large_sold + medium_price * medium_sold + small_price * small_sold

/-- Theorem: Michael's earnings from selling birdhouses --/
theorem michael_earnings : 
  total_money_earned 22 16 7 2 2 3 = 97 := by sorry

end michael_earnings_l2901_290168


namespace max_a_value_l2901_290173

theorem max_a_value (a : ℝ) : 
  (∀ x ∈ Set.Icc (1/2 : ℝ) 2, x * Real.log x - (1 + a) * x + 1 ≥ 0) →
  a ≤ 0 :=
by sorry

end max_a_value_l2901_290173


namespace opposite_of_negative_2023_l2901_290125

theorem opposite_of_negative_2023 : -((-2023 : ℤ)) = 2023 := by sorry

end opposite_of_negative_2023_l2901_290125


namespace parabola_point_and_line_intersection_l2901_290110

/-- Parabola C defined by y^2 = 4x -/
def C : Set (ℝ × ℝ) := {p : ℝ × ℝ | p.2^2 = 4 * p.1}

/-- Point P on the parabola C -/
def P : ℝ × ℝ := (1, 2)

/-- Point Q symmetrical to P across the x-axis -/
def Q : ℝ × ℝ := (P.1, -P.2)

/-- Origin O -/
def O : ℝ × ℝ := (0, 0)

/-- Area of triangle POQ -/
def area_POQ : ℝ := 2

/-- Slopes of lines PA and PB -/
def k₁ : ℝ := sorry
def k₂ : ℝ := sorry

/-- Fixed point that AB passes through -/
def fixed_point : ℝ × ℝ := (0, -2)

theorem parabola_point_and_line_intersection :
  (P ∈ C) ∧
  (P.2 > 0) ∧
  (area_POQ = 2) ∧
  (k₁ * k₂ = 4) →
  (P = (1, 2)) ∧
  (∀ (A B : ℝ × ℝ), A ∈ C → B ∈ C →
    (A.2 - P.2) / (A.1 - P.1) = k₁ →
    (B.2 - P.2) / (B.1 - P.1) = k₂ →
    ∃ (m b : ℝ), (A.2 = m * A.1 + b) ∧ (B.2 = m * B.1 + b) ∧
    (fixed_point.2 = m * fixed_point.1 + b)) :=
sorry

end parabola_point_and_line_intersection_l2901_290110


namespace half_angle_quadrant_l2901_290145

-- Define what it means for an angle to be in the third quadrant
def in_third_quadrant (α : ℝ) : Prop :=
  ∃ k : ℤ, k * 360 + 180 < α ∧ α < k * 360 + 270

-- Define what it means for an angle to be in the second quadrant
def in_second_quadrant (α : ℝ) : Prop :=
  ∃ n : ℤ, n * 360 + 90 < α ∧ α < n * 360 + 180

-- Define what it means for an angle to be in the fourth quadrant
def in_fourth_quadrant (α : ℝ) : Prop :=
  ∃ n : ℤ, n * 360 + 270 < α ∧ α < n * 360 + 360

-- Theorem statement
theorem half_angle_quadrant (α : ℝ) :
  in_third_quadrant α → in_second_quadrant (α/2) ∨ in_fourth_quadrant (α/2) :=
by sorry

end half_angle_quadrant_l2901_290145


namespace solution_value_l2901_290183

-- Define the solution sets A and B
def A : Set ℝ := {x | x^2 - 2*x - 3 < 0}
def B : Set ℝ := {x | x^2 + x - 6 < 0}

-- Define the intersection of A and B
def A_intersect_B : Set ℝ := A ∩ B

-- Define the function representing x^2 + ax + b
def f (a b : ℝ) (x : ℝ) : ℝ := x^2 + a*x + b

-- State the theorem
theorem solution_value (a b : ℝ) : 
  (∀ x, x ∈ A_intersect_B ↔ f a b x < 0) → a + b = -3 :=
by sorry

end solution_value_l2901_290183


namespace average_position_l2901_290126

def fractions : List ℚ := [1/2, 1/3, 1/4, 1/5, 1/6, 1/7]

theorem average_position (average : ℚ := (fractions.sum) / 6) :
  average = 223/840 ∧ 1/4 < average ∧ average < 1/3 := by sorry

end average_position_l2901_290126


namespace geometric_sequence_sum_l2901_290134

/-- A sequence is geometric if the ratio between any two consecutive terms is constant. -/
def IsGeometric (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = r * a n

/-- Given a geometric sequence {a_n} where a_4 + a_6 = 8, 
    prove that a_1a_7 + 2a_3a_7 + a_3a_9 = 64. -/
theorem geometric_sequence_sum (a : ℕ → ℝ) 
    (h_geometric : IsGeometric a) 
    (h_sum : a 4 + a 6 = 8) : 
    a 1 * a 7 + 2 * a 3 * a 7 + a 3 * a 9 = 64 := by
  sorry

end geometric_sequence_sum_l2901_290134


namespace range_of_fraction_l2901_290171

theorem range_of_fraction (x y : ℝ) (hx : 1 ≤ x ∧ x ≤ 4) (hy : 3 ≤ y ∧ y ≤ 6) :
  (1/6 : ℝ) ≤ x/y ∧ x/y ≤ (4/3 : ℝ) := by
sorry

end range_of_fraction_l2901_290171


namespace workshop_salary_problem_l2901_290193

theorem workshop_salary_problem (total_workers : ℕ) (all_avg_salary : ℚ) 
  (num_technicians : ℕ) (tech_avg_salary : ℚ) :
  total_workers = 21 →
  all_avg_salary = 8000 →
  num_technicians = 7 →
  tech_avg_salary = 12000 →
  let remaining_workers := total_workers - num_technicians
  let total_salary := all_avg_salary * total_workers
  let tech_total_salary := tech_avg_salary * num_technicians
  let remaining_total_salary := total_salary - tech_total_salary
  let remaining_avg_salary := remaining_total_salary / remaining_workers
  remaining_avg_salary = 6000 := by
sorry

end workshop_salary_problem_l2901_290193


namespace symmetry_conditions_l2901_290129

/-- A function is symmetric about a point (a, b) if f(x) + f(2a - x) = 2b for all x in its domain -/
def SymmetricAbout (f : ℝ → ℝ) (a b : ℝ) : Prop :=
  ∀ x, f x + f (2 * a - x) = 2 * b

theorem symmetry_conditions (m a : ℝ) :
  let f := fun x : ℝ => (x^2 + m*x + m) / x
  let g := fun x : ℝ => if x > 0 then x^2 + a*x + 1 else -x^2 + a*x + 1
  (SymmetricAbout f 0 1) ∧
  (∀ x ≠ 0, SymmetricAbout g 0 1) ∧
  (∀ x t, x < 0 → t > 0 → g x < f t) →
  (m = 1) ∧
  (∀ x < 0, g x = -x^2 + a*x + 1) ∧
  (-2 * Real.sqrt 2 < a) := by
  sorry

end symmetry_conditions_l2901_290129


namespace lindas_savings_l2901_290137

theorem lindas_savings (savings : ℝ) : (1 / 4 : ℝ) * savings = 230 → savings = 920 := by
  sorry

end lindas_savings_l2901_290137


namespace max_ships_on_battleship_board_l2901_290179

/-- Represents a rectangular board -/
structure Board :=
  (rows : ℕ)
  (cols : ℕ)

/-- Represents a ship -/
structure Ship :=
  (length : ℕ)
  (width : ℕ)

/-- Represents a placement of ships on a board -/
def Placement := List (ℕ × ℕ)

/-- Checks if two ships are adjacent or overlapping -/
def are_adjacent_or_overlapping (p1 p2 : ℕ × ℕ) (s : Ship) : Prop := sorry

/-- Checks if a placement is valid (no adjacent or overlapping ships) -/
def is_valid_placement (b : Board) (s : Ship) (p : Placement) : Prop := sorry

/-- The maximum number of ships that can be placed on the board -/
def max_ships (b : Board) (s : Ship) : ℕ := sorry

/-- The main theorem stating the maximum number of 1x4 ships on a 10x10 board -/
theorem max_ships_on_battleship_board :
  let b : Board := ⟨10, 10⟩
  let s : Ship := ⟨4, 1⟩
  max_ships b s = 24 := by sorry

end max_ships_on_battleship_board_l2901_290179


namespace eggs_in_box_l2901_290176

/-- The number of eggs Harry takes from the box -/
def eggs_taken : ℕ := 5

/-- The number of eggs left in the box after Harry takes some -/
def eggs_left : ℕ := 42

/-- The initial number of eggs in the box -/
def initial_eggs : ℕ := eggs_taken + eggs_left

theorem eggs_in_box : initial_eggs = 47 := by
  sorry

end eggs_in_box_l2901_290176


namespace four_digit_solution_l2901_290143

def is_four_digit (n : ℕ) : Prop :=
  1000 ≤ n ∧ n ≤ 9999

def digit_value (n : ℕ) (place : ℕ) : ℕ :=
  (n / (10^place)) % 10

def number_from_digits (a b c d : ℕ) : ℕ :=
  1000 * a + 100 * b + 10 * c + d

theorem four_digit_solution :
  let abcd := 2996
  let dcba := number_from_digits (digit_value abcd 0) (digit_value abcd 1) (digit_value abcd 2) (digit_value abcd 3)
  is_four_digit abcd ∧ is_four_digit dcba ∧ 2 * abcd + 1000 = dcba := by
  sorry

end four_digit_solution_l2901_290143


namespace diophantine_equation_implication_l2901_290160

theorem diophantine_equation_implication (a b : ℤ) 
  (ha : ¬ ∃ (n : ℤ), a = n^2) 
  (hb : ¬ ∃ (n : ℤ), b = n^2) :
  (∃ (x0 y0 z0 w0 : ℤ), x0^2 - a*y0^2 - b*z0^2 + a*b*w0^2 = 0 ∧ (x0, y0, z0, w0) ≠ (0, 0, 0, 0)) →
  (∃ (x1 y1 z1 : ℤ), x1^2 - a*y1^2 - b*z1^2 = 0 ∧ (x1, y1, z1) ≠ (0, 0, 0)) :=
by sorry

end diophantine_equation_implication_l2901_290160


namespace inequality_proof_l2901_290181

theorem inequality_proof (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  (a + b) * (b + c) * (c + a) ≥ 8 * a * b * c := by
  sorry

end inequality_proof_l2901_290181


namespace rikki_poetry_pricing_l2901_290178

-- Define the constants
def words_per_interval : ℕ := 25
def minutes_per_interval : ℕ := 5
def total_minutes : ℕ := 120
def expected_earnings : ℚ := 6

-- Define the function to calculate the price per word
def price_per_word : ℚ :=
  let intervals : ℕ := total_minutes / minutes_per_interval
  let total_words : ℕ := words_per_interval * intervals
  expected_earnings / total_words

-- Theorem statement
theorem rikki_poetry_pricing :
  price_per_word = 1/100 := by sorry

end rikki_poetry_pricing_l2901_290178


namespace multiples_of_three_l2901_290158

theorem multiples_of_three (n : ℕ) : (∃ k, k = 33 ∧ k * 3 = n) ↔ n = 99 := by sorry

end multiples_of_three_l2901_290158


namespace tan_sum_specific_angles_l2901_290109

theorem tan_sum_specific_angles (α β : Real) (h1 : Real.tan α = 2) (h2 : Real.tan β = 3) :
  Real.tan (α + β) = -1 := by
  sorry

end tan_sum_specific_angles_l2901_290109


namespace inequality_range_l2901_290106

theorem inequality_range (a : ℝ) : 
  (∀ x : ℝ, |2 - x| + |3 + x| ≥ a^2 - 4*a) ↔ -1 ≤ a ∧ a ≤ 5 := by
  sorry

end inequality_range_l2901_290106


namespace cube_sum_plus_triple_product_l2901_290177

theorem cube_sum_plus_triple_product (x y : ℝ) (h : x + y = 1) :
  x^3 + y^3 + 3*x*y = 1 := by sorry

end cube_sum_plus_triple_product_l2901_290177


namespace correct_number_probability_l2901_290196

def first_four_options : List ℕ := [2960, 2961, 2990, 2991]
def last_three_digits : List ℕ := [6, 7, 8]

def total_possible_numbers : ℕ := (List.length first_four_options) * (Nat.factorial (List.length last_three_digits))

theorem correct_number_probability :
  (1 : ℚ) / total_possible_numbers = 1 / 24 :=
sorry

end correct_number_probability_l2901_290196


namespace refrigerator_price_l2901_290190

/-- The price paid for a refrigerator given specific conditions --/
theorem refrigerator_price (discount_rate : ℝ) (transport_cost : ℝ) (installation_cost : ℝ)
  (profit_rate : ℝ) (selling_price : ℝ) :
  discount_rate = 0.20 →
  transport_cost = 125 →
  installation_cost = 250 →
  profit_rate = 0.16 →
  selling_price = 18560 →
  ∃ (labelled_price : ℝ),
    selling_price = labelled_price * (1 + profit_rate) ∧
    labelled_price * (1 - discount_rate) + transport_cost + installation_cost = 13175 :=
by sorry

end refrigerator_price_l2901_290190


namespace square_root_sum_equals_eight_l2901_290147

theorem square_root_sum_equals_eight (x : ℝ) : 
  (Real.sqrt (49 - x^2) - Real.sqrt (25 - x^2) = 3) → 
  (Real.sqrt (49 - x^2) + Real.sqrt (25 - x^2) = 8) := by
  sorry

end square_root_sum_equals_eight_l2901_290147


namespace horner_method_multiplications_for_degree_5_l2901_290120

def horner_multiplications (n : ℕ) : ℕ := n

theorem horner_method_multiplications_for_degree_5 :
  ∀ (a₀ a₁ a₂ a₃ a₄ a₅ : ℝ),
  let f : ℝ → ℝ := λ x => a₅ * x^5 + a₄ * x^4 + a₃ * x^3 + a₂ * x^2 + a₁ * x + a₀
  horner_multiplications 5 = 5 :=
by sorry

end horner_method_multiplications_for_degree_5_l2901_290120


namespace tank_capacity_l2901_290189

theorem tank_capacity (initial_fill : ℚ) (added_gallons : ℚ) (final_fill : ℚ) :
  initial_fill = 3 / 4 →
  added_gallons = 9 →
  final_fill = 9 / 10 →
  ∃ (capacity : ℚ), capacity = 60 ∧ 
    final_fill * capacity = initial_fill * capacity + added_gallons :=
by sorry

end tank_capacity_l2901_290189


namespace triangle_right_angled_l2901_290150

theorem triangle_right_angled (a b c : ℝ) (h : 0 < a ∧ 0 < b ∧ 0 < c) 
  (eq : 2 * (a^8 + b^8 + c^8) = (a^4 + b^4 + c^4)^2) : 
  a^2 + b^2 = c^2 ∨ a^2 + c^2 = b^2 ∨ b^2 + c^2 = a^2 := by
sorry

end triangle_right_angled_l2901_290150

import Mathlib

namespace NUMINAMATH_CALUDE_total_cars_produced_l3571_357178

theorem total_cars_produced (north_america : ℕ) (europe : ℕ) 
  (h1 : north_america = 3884) (h2 : europe = 2871) : 
  north_america + europe = 6755 := by
  sorry

end NUMINAMATH_CALUDE_total_cars_produced_l3571_357178


namespace NUMINAMATH_CALUDE_geometric_sequence_product_l3571_357163

/-- Given a geometric sequence {a_n} with a_1 = 1 and common ratio q ≠ 1,
    if a_m = a_1 * a_2 * a_3 * a_4 * a_5, then m = 11 -/
theorem geometric_sequence_product (q : ℝ) (h : q ≠ 1) :
  let a : ℕ → ℝ := fun n => q^(n-1)
  ∃ m : ℕ, a m = (a 1) * (a 2) * (a 3) * (a 4) * (a 5) ∧ m = 11 := by
  sorry


end NUMINAMATH_CALUDE_geometric_sequence_product_l3571_357163


namespace NUMINAMATH_CALUDE_fourth_term_is_ten_l3571_357176

/-- An arithmetic sequence is a sequence where the difference between any two consecutive terms is constant. -/
structure ArithmeticSequence where
  a : ℝ  -- First term
  d : ℝ  -- Common difference

/-- The nth term of an arithmetic sequence -/
def ArithmeticSequence.nthTerm (seq : ArithmeticSequence) (n : ℕ) : ℝ :=
  seq.a + (n - 1) * seq.d

theorem fourth_term_is_ten
  (seq : ArithmeticSequence)
  (h : seq.nthTerm 2 + seq.nthTerm 6 = 20) :
  seq.nthTerm 4 = 10 := by
  sorry

#check fourth_term_is_ten

end NUMINAMATH_CALUDE_fourth_term_is_ten_l3571_357176


namespace NUMINAMATH_CALUDE_fixed_point_implies_sqrt_two_l3571_357188

noncomputable section

-- Define the logarithmic function
def log_func (a : ℝ) (x : ℝ) : ℝ := Real.log x / Real.log a

-- Define the power function
def power_func (α : ℝ) (x : ℝ) : ℝ := x ^ α

theorem fixed_point_implies_sqrt_two 
  (a : ℝ) 
  (h_a_pos : a > 0) 
  (h_a_neq_one : a ≠ 1) 
  (A : ℝ × ℝ) 
  (α : ℝ) 
  (h_log_point : log_func a (A.1 - 3) + 2 = A.2)
  (h_power_point : power_func α A.1 = A.2) :
  power_func α 2 = Real.sqrt 2 :=
sorry

end

end NUMINAMATH_CALUDE_fixed_point_implies_sqrt_two_l3571_357188


namespace NUMINAMATH_CALUDE_intersection_and_lines_l3571_357152

-- Define the lines l₁ and l₂
def l₁ (x y : ℝ) : Prop := x - 2*y + 4 = 0
def l₂ (x y : ℝ) : Prop := 4*x + 3*y + 5 = 0

-- Define point A
def A : ℝ × ℝ := (-1, -2)

-- Define the intersection point P
def P : ℝ × ℝ := (-2, 1)

-- Define the condition for a
def a_condition (a : ℝ) : Prop := a ≠ -2 ∧ a ≠ -1 ∧ a ≠ 8/3

-- Define the equations of line l
def l_eq₁ (x y : ℝ) : Prop := 4*x + 3*y + 5 = 0
def l_eq₂ (x y : ℝ) : Prop := x + 2 = 0

theorem intersection_and_lines :
  -- 1. P is the intersection point of l₁ and l₂
  (l₁ P.1 P.2 ∧ l₂ P.1 P.2) ∧
  -- 2. Condition for a to form a triangle
  (∀ a : ℝ, (∃ x y : ℝ, l₁ x y ∧ l₂ x y ∧ (a*x + 2*y - 6 = 0)) → a_condition a) ∧
  -- 3. Equations of line l passing through P with distance 1 from A
  (∀ x y : ℝ, (l_eq₁ x y ∨ l_eq₂ x y) ↔
    ((x - P.1)^2 + (y - P.2)^2 = 0 ∧
     ((x - A.1)^2 + (y - A.2)^2 - 1)^2 = 
     ((x - P.1)*(A.2 - P.2) - (y - P.2)*(A.1 - P.1))^2 / ((x - P.1)^2 + (y - P.2)^2)))
  := by sorry

end NUMINAMATH_CALUDE_intersection_and_lines_l3571_357152


namespace NUMINAMATH_CALUDE_lunch_break_is_60_minutes_l3571_357182

/-- Represents the painting rates and work done on each day -/
structure PaintingData where
  paula_rate : ℝ
  helpers_rate : ℝ
  day1_hours : ℝ
  day1_work : ℝ
  day2_hours : ℝ
  day2_work : ℝ
  day3_hours : ℝ
  day3_work : ℝ

/-- The lunch break duration in hours -/
def lunch_break : ℝ := 1

/-- Theorem stating that the lunch break is 60 minutes given the painting data -/
theorem lunch_break_is_60_minutes (data : PaintingData) : 
  (data.day1_hours - lunch_break) * (data.paula_rate + data.helpers_rate) = data.day1_work ∧
  (data.day2_hours - lunch_break) * data.helpers_rate = data.day2_work ∧
  (data.day3_hours - lunch_break) * data.paula_rate = data.day3_work →
  lunch_break * 60 = 60 := by
  sorry

#eval lunch_break * 60  -- Should output 60

end NUMINAMATH_CALUDE_lunch_break_is_60_minutes_l3571_357182


namespace NUMINAMATH_CALUDE_rectangle_length_l3571_357180

/-- Given a rectangle with perimeter 680 meters and breadth 82 meters, its length is 258 meters. -/
theorem rectangle_length (perimeter breadth : ℝ) (h1 : perimeter = 680) (h2 : breadth = 82) :
  (perimeter / 2) - breadth = 258 :=
by sorry

end NUMINAMATH_CALUDE_rectangle_length_l3571_357180


namespace NUMINAMATH_CALUDE_characterization_of_good_numbers_l3571_357168

def is_good (n : ℕ) : Prop :=
  ∀ d : ℕ, d ∣ n → (d + 1) ∣ (n + 1)

theorem characterization_of_good_numbers (n : ℕ) :
  is_good n ↔ n = 1 ∨ (Nat.Prime n ∧ n % 2 = 1) :=
sorry

end NUMINAMATH_CALUDE_characterization_of_good_numbers_l3571_357168


namespace NUMINAMATH_CALUDE_square_tiles_count_l3571_357183

/-- Represents the number of edges for each type of tile -/
def edges_per_tile : Fin 3 → ℕ
| 0 => 3  -- triangular
| 1 => 4  -- square
| 2 => 5  -- pentagonal
| _ => 0  -- unreachable

/-- Proves that given 30 tiles with 108 edges in total, there are 6 square tiles -/
theorem square_tiles_count 
  (total_tiles : ℕ) 
  (total_edges : ℕ) 
  (h_total_tiles : total_tiles = 30) 
  (h_total_edges : total_edges = 108) :
  ∃ (t s p : ℕ), 
    t + s + p = total_tiles ∧ 
    3 * t + 4 * s + 5 * p = total_edges ∧ 
    s = 6 :=
by
  sorry

#check square_tiles_count

end NUMINAMATH_CALUDE_square_tiles_count_l3571_357183


namespace NUMINAMATH_CALUDE_cake_box_height_l3571_357117

/-- Represents the dimensions of a rectangular object -/
structure Dimensions where
  length : ℕ
  width : ℕ
  height : ℕ

/-- Calculates the maximum number of items that can fit along a dimension -/
def maxItemsAlongDimension (containerSize itemSize : ℕ) : ℕ :=
  containerSize / itemSize

/-- Represents the problem of determining the height of cake boxes in a carton -/
def cakeBoxProblem (cartonDims : Dimensions) (cakeBoxBase : Dimensions) (maxBoxes : ℕ) : Prop :=
  let boxesAlongLength := maxItemsAlongDimension cartonDims.length cakeBoxBase.length
  let boxesAlongWidth := maxItemsAlongDimension cartonDims.width cakeBoxBase.width
  let boxesPerLayer := boxesAlongLength * boxesAlongWidth
  let numLayers := maxBoxes / boxesPerLayer
  let cakeBoxHeight := cartonDims.height / numLayers
  cakeBoxHeight = 5

/-- The main theorem stating that the height of a cake box is 5 inches -/
theorem cake_box_height :
  cakeBoxProblem
    (Dimensions.mk 25 42 60)  -- Carton dimensions
    (Dimensions.mk 8 7 0)     -- Cake box base dimensions (height is unknown)
    210                       -- Maximum number of boxes
  := by sorry

end NUMINAMATH_CALUDE_cake_box_height_l3571_357117


namespace NUMINAMATH_CALUDE_school_days_is_five_l3571_357101

/-- Represents the number of sheets of paper used per class per day -/
def sheets_per_class_per_day : ℕ := 200

/-- Represents the total number of sheets of paper used by the school per week -/
def total_sheets_per_week : ℕ := 9000

/-- Represents the number of classes in the school -/
def number_of_classes : ℕ := 9

/-- Calculates the number of school days in a week -/
def school_days_per_week : ℕ := total_sheets_per_week / (sheets_per_class_per_day * number_of_classes)

/-- Proves that the number of school days in a week is 5 -/
theorem school_days_is_five : school_days_per_week = 5 := by
  sorry

end NUMINAMATH_CALUDE_school_days_is_five_l3571_357101


namespace NUMINAMATH_CALUDE_point_k_value_l3571_357179

theorem point_k_value (A B C K : ℝ) : 
  A = -3 → B = -5 → C = 6 → 
  (A + B + C + K = -A - B - C - K) → 
  K = 2 := by sorry

end NUMINAMATH_CALUDE_point_k_value_l3571_357179


namespace NUMINAMATH_CALUDE_solve_ttakji_problem_l3571_357198

def ttakji_problem (initial_large : ℕ) (initial_small : ℕ) (final_total : ℕ) : Prop :=
  ∃ (lost_large : ℕ),
    initial_large ≥ lost_large ∧
    initial_small ≥ 3 * lost_large ∧
    initial_large + initial_small - lost_large - 3 * lost_large = final_total ∧
    lost_large = 4

theorem solve_ttakji_problem :
  ttakji_problem 12 34 30 := by sorry

end NUMINAMATH_CALUDE_solve_ttakji_problem_l3571_357198


namespace NUMINAMATH_CALUDE_investment_problem_l3571_357177

theorem investment_problem (total : ℝ) (rate_a rate_b rate_c : ℝ) 
  (h_total : total = 425)
  (h_rate_a : rate_a = 0.05)
  (h_rate_b : rate_b = 0.08)
  (h_rate_c : rate_c = 0.10)
  (h_equal_increase : ∃ (k : ℝ), k > 0 ∧ 
    ∀ (a b c : ℝ), a + b + c = total → 
    rate_a * a = k ∧ rate_b * b = k ∧ rate_c * c = k) :
  ∃ (a b c : ℝ), a + b + c = total ∧ 
    rate_a * a = rate_b * b ∧ rate_b * b = rate_c * c ∧ 
    c = 100 := by
  sorry

end NUMINAMATH_CALUDE_investment_problem_l3571_357177


namespace NUMINAMATH_CALUDE_sum_ab_over_2b_plus_1_geq_1_l3571_357156

variables (a b c : ℝ)

theorem sum_ab_over_2b_plus_1_geq_1
  (h_positive : a > 0 ∧ b > 0 ∧ c > 0)
  (h_sum : a + b + c = 3) :
  (a * b) / (2 * b + 1) + (b * c) / (2 * c + 1) + (c * a) / (2 * a + 1) ≥ 1 :=
sorry

end NUMINAMATH_CALUDE_sum_ab_over_2b_plus_1_geq_1_l3571_357156


namespace NUMINAMATH_CALUDE_water_needed_for_mixture_l3571_357157

/-- Given the initial mixture composition and the desired total volume, 
    prove that the amount of water needed is 0.24 liters. -/
theorem water_needed_for_mixture (initial_chemical_b : ℝ) (initial_water : ℝ) 
  (initial_mixture : ℝ) (desired_volume : ℝ) 
  (h1 : initial_chemical_b = 0.05)
  (h2 : initial_water = 0.03)
  (h3 : initial_mixture = 0.08)
  (h4 : desired_volume = 0.64)
  (h5 : initial_chemical_b + initial_water = initial_mixture) : 
  desired_volume * (initial_water / initial_mixture) = 0.24 := by
  sorry

end NUMINAMATH_CALUDE_water_needed_for_mixture_l3571_357157


namespace NUMINAMATH_CALUDE_min_a_for_decreasing_h_range_a_for_p_greater_q_l3571_357139

noncomputable section

-- Define the functions
def f (a : ℝ) (x : ℝ) : ℝ := x + 4 * a / x - 1
def g (a : ℝ) (x : ℝ) : ℝ := a * Real.log x
def h (a : ℝ) (x : ℝ) : ℝ := f a x - g a x
def p (x : ℝ) : ℝ := (2 - x^3) * Real.exp x
def q (a : ℝ) (x : ℝ) : ℝ := g a x / x + 2

-- Part I: Minimum value of a for h to be decreasing on [1,3]
theorem min_a_for_decreasing_h : 
  (∀ x ∈ Set.Icc 1 3, ∀ y ∈ Set.Icc 1 3, x ≤ y → h (9/7) x ≥ h (9/7) y) ∧
  (∀ a < 9/7, ∃ x ∈ Set.Icc 1 3, ∃ y ∈ Set.Icc 1 3, x < y ∧ h a x < h a y) :=
sorry

-- Part II: Range of a for p(x₁) > q(x₂) to hold for any x₁, x₂ ∈ (0,1)
theorem range_a_for_p_greater_q :
  (∀ a ≥ 0, ∀ x₁ ∈ Set.Ioo 0 1, ∀ x₂ ∈ Set.Ioo 0 1, p x₁ > q a x₂) ∧
  (∀ a < 0, ∃ x₁ ∈ Set.Ioo 0 1, ∃ x₂ ∈ Set.Ioo 0 1, p x₁ ≤ q a x₂) :=
sorry

end NUMINAMATH_CALUDE_min_a_for_decreasing_h_range_a_for_p_greater_q_l3571_357139


namespace NUMINAMATH_CALUDE_baker_cakes_left_l3571_357184

theorem baker_cakes_left (total_cakes sold_cakes : ℕ) 
  (h1 : total_cakes = 54)
  (h2 : sold_cakes = 41) :
  total_cakes - sold_cakes = 13 := by
  sorry

end NUMINAMATH_CALUDE_baker_cakes_left_l3571_357184


namespace NUMINAMATH_CALUDE_cos_is_even_and_has_zero_point_l3571_357145

-- Define what it means for a function to be even
def IsEven (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = f x

-- Define what it means for a function to have a zero point
def HasZeroPoint (f : ℝ → ℝ) : Prop :=
  ∃ x, f x = 0

theorem cos_is_even_and_has_zero_point :
  IsEven Real.cos ∧ HasZeroPoint Real.cos := by sorry

end NUMINAMATH_CALUDE_cos_is_even_and_has_zero_point_l3571_357145


namespace NUMINAMATH_CALUDE_mark_fruit_theorem_l3571_357174

/-- The number of fruit pieces Mark kept for next week -/
def fruit_kept_for_next_week (initial_fruit pieces_eaten_four_days pieces_for_friday : ℕ) : ℕ :=
  initial_fruit - pieces_eaten_four_days - pieces_for_friday

theorem mark_fruit_theorem (initial_fruit pieces_eaten_four_days pieces_for_friday : ℕ) 
  (h1 : initial_fruit = 10)
  (h2 : pieces_eaten_four_days = 5)
  (h3 : pieces_for_friday = 3) :
  fruit_kept_for_next_week initial_fruit pieces_eaten_four_days pieces_for_friday = 2 := by
  sorry

end NUMINAMATH_CALUDE_mark_fruit_theorem_l3571_357174


namespace NUMINAMATH_CALUDE_equality_of_four_reals_l3571_357153

theorem equality_of_four_reals (a b c d : ℝ) :
  a^2 + b^2 + c^2 + d^2 = a*b + b*c + c*d + d*a → a = b ∧ b = c ∧ c = d := by
  sorry

end NUMINAMATH_CALUDE_equality_of_four_reals_l3571_357153


namespace NUMINAMATH_CALUDE_tangent_line_condition_l3571_357114

-- Define the curves
def curve1 (x : ℝ) : ℝ := x^3
def curve2 (a x : ℝ) : ℝ := a*x^2 + x - 9

-- Define the tangent line condition
def is_tangent_to_both (a : ℝ) : Prop :=
  ∃ (m : ℝ), ∃ (x₀ : ℝ),
    -- The line passes through (1,0)
    m * (1 - x₀) = -curve1 x₀ ∧
    -- The line is tangent to y = x^3
    m = 3 * x₀^2 ∧
    -- The line is tangent to y = ax^2 + x - 9
    m = 2 * a * x₀ + 1 ∧
    -- The point (x₀, curve1 x₀) is on both curves
    curve1 x₀ = curve2 a x₀

-- The main theorem
theorem tangent_line_condition (a : ℝ) :
  is_tangent_to_both a → a = -1 ∨ a = -7 :=
sorry

end NUMINAMATH_CALUDE_tangent_line_condition_l3571_357114


namespace NUMINAMATH_CALUDE_hyperbola_dot_product_range_l3571_357150

/-- The hyperbola with center at origin and left focus at (-2,0) -/
structure Hyperbola where
  a : ℝ
  h_pos : a > 0

/-- A point on the right branch of the hyperbola -/
structure HyperbolaPoint (h : Hyperbola) where
  x : ℝ
  y : ℝ
  h_on_hyperbola : x^2 / h.a^2 - y^2 = 1
  h_right_branch : x ≥ h.a

/-- The theorem stating the range of the dot product -/
theorem hyperbola_dot_product_range (h : Hyperbola) (p : HyperbolaPoint h) :
  p.x * (p.x + 2) + p.y * p.y ≥ 3 + 2 * Real.sqrt 3 := by sorry

end NUMINAMATH_CALUDE_hyperbola_dot_product_range_l3571_357150


namespace NUMINAMATH_CALUDE_tan_negative_five_pi_thirds_equals_sqrt_three_l3571_357194

theorem tan_negative_five_pi_thirds_equals_sqrt_three :
  Real.tan (-5 * π / 3) = Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_tan_negative_five_pi_thirds_equals_sqrt_three_l3571_357194


namespace NUMINAMATH_CALUDE_crayons_per_row_l3571_357109

theorem crayons_per_row (rows : ℕ) (pencils_per_row : ℕ) (total_items : ℕ) 
  (h1 : rows = 11)
  (h2 : pencils_per_row = 31)
  (h3 : total_items = 638) :
  (total_items - rows * pencils_per_row) / rows = 27 := by
  sorry

end NUMINAMATH_CALUDE_crayons_per_row_l3571_357109


namespace NUMINAMATH_CALUDE_complex_equation_solution_l3571_357126

theorem complex_equation_solution (z : ℂ) : (Complex.I * (z - 1) = 1 - Complex.I) → z = -Complex.I := by
  sorry

end NUMINAMATH_CALUDE_complex_equation_solution_l3571_357126


namespace NUMINAMATH_CALUDE_product_approximation_l3571_357140

def is_approximately_equal (x y : ℕ) (tolerance : ℕ) : Prop :=
  (x ≤ y + tolerance) ∧ (y ≤ x + tolerance)

theorem product_approximation (tolerance : ℕ) :
  (is_approximately_equal (4 * 896) 3600 tolerance) ∧
  (is_approximately_equal (405 * 9) 3600 tolerance) ∧
  ¬(is_approximately_equal (6 * 689) 3600 tolerance) ∧
  ¬(is_approximately_equal (398 * 8) 3600 tolerance) :=
by sorry

end NUMINAMATH_CALUDE_product_approximation_l3571_357140


namespace NUMINAMATH_CALUDE_monthly_compounding_greater_than_annual_l3571_357161

theorem monthly_compounding_greater_than_annual : 
  (1 + 0.04 / 12) ^ 12 > 1 + 0.04 := by
  sorry

end NUMINAMATH_CALUDE_monthly_compounding_greater_than_annual_l3571_357161


namespace NUMINAMATH_CALUDE_identity_mapping_implies_sum_l3571_357151

theorem identity_mapping_implies_sum (a b : ℝ) : 
  let M : Set ℝ := {-1, b/a, 1}
  let N : Set ℝ := {a, b, b-a}
  (∀ x ∈ M, x ∈ N) → (a + b = 1 ∨ a + b = -1) := by
  sorry

end NUMINAMATH_CALUDE_identity_mapping_implies_sum_l3571_357151


namespace NUMINAMATH_CALUDE_blueberry_pies_count_l3571_357175

/-- Given a total of 30 pies and a ratio of 2:3:4:1 for apple:blueberry:cherry:peach pies,
    the number of blueberry pies is 9. -/
theorem blueberry_pies_count (total_pies : ℕ) (apple_ratio blueberry_ratio cherry_ratio peach_ratio : ℕ) :
  total_pies = 30 →
  apple_ratio = 2 →
  blueberry_ratio = 3 →
  cherry_ratio = 4 →
  peach_ratio = 1 →
  blueberry_ratio * (total_pies / (apple_ratio + blueberry_ratio + cherry_ratio + peach_ratio)) = 9 := by
  sorry

end NUMINAMATH_CALUDE_blueberry_pies_count_l3571_357175


namespace NUMINAMATH_CALUDE_noah_holidays_per_month_l3571_357190

/-- The number of holidays Noah takes in a year -/
def total_holidays : ℕ := 36

/-- The number of months in a year -/
def months_in_year : ℕ := 12

/-- The number of holidays Noah takes each month -/
def holidays_per_month : ℚ := total_holidays / months_in_year

theorem noah_holidays_per_month :
  holidays_per_month = 3 := by sorry

end NUMINAMATH_CALUDE_noah_holidays_per_month_l3571_357190


namespace NUMINAMATH_CALUDE_cycle_selling_price_l3571_357144

/-- Calculates the selling price of an item given its cost price and gain percent. -/
def sellingPrice (costPrice : ℕ) (gainPercent : ℕ) : ℕ :=
  costPrice + (costPrice * gainPercent) / 100

/-- Theorem stating that the selling price of a cycle with cost price 900 and gain percent 30 is 1170. -/
theorem cycle_selling_price :
  sellingPrice 900 30 = 1170 := by
  sorry

end NUMINAMATH_CALUDE_cycle_selling_price_l3571_357144


namespace NUMINAMATH_CALUDE_power_product_equality_l3571_357124

theorem power_product_equality (a b : ℝ) : (2 * a^2 * b)^3 = 8 * a^6 * b^3 := by
  sorry

end NUMINAMATH_CALUDE_power_product_equality_l3571_357124


namespace NUMINAMATH_CALUDE_sqrt_sum_equality_l3571_357191

theorem sqrt_sum_equality : Real.sqrt 18 + Real.sqrt 24 / Real.sqrt 3 = 5 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_sum_equality_l3571_357191


namespace NUMINAMATH_CALUDE_x_plus_y_value_l3571_357116

theorem x_plus_y_value (x y : ℝ) 
  (eq1 : x + Real.sin y = 2008)
  (eq2 : x + 2008 * Real.cos y = 2007)
  (h : 0 ≤ y ∧ y ≤ Real.pi / 2) :
  x + y = 2007 + Real.pi / 2 := by
  sorry

end NUMINAMATH_CALUDE_x_plus_y_value_l3571_357116


namespace NUMINAMATH_CALUDE_vector_operation_proof_l3571_357143

def a : Fin 2 → ℝ := ![3, 2]
def b : Fin 2 → ℝ := ![0, -1]

theorem vector_operation_proof :
  (3 • b - a) = ![(-3 : ℝ), -5] := by sorry

end NUMINAMATH_CALUDE_vector_operation_proof_l3571_357143


namespace NUMINAMATH_CALUDE_function_properties_l3571_357149

noncomputable section

variable (I : Set ℝ)
variable (f : ℝ → ℝ)
variable (f' : ℝ → ℝ)

theorem function_properties
  (h1 : ∀ x ∈ I, 0 < f' x ∧ f' x < 2)
  (h2 : ∀ x ∈ I, f' x ≠ 1)
  (h3 : ∃ c₁ ∈ I, f c₁ = c₁)
  (h4 : ∃ c₂ ∈ I, f c₂ = 2 * c₂)
  (h5 : ∀ a b, a ∈ I → b ∈ I → a ≤ b → ∃ x ∈ Set.Ioo a b, f b - f a = (b - a) * f' x) :
  (∀ x ∈ I, f x = x → x = Classical.choose h3) ∧
  (∀ x > Classical.choose h4, f x < 2 * x) :=
by sorry

end NUMINAMATH_CALUDE_function_properties_l3571_357149


namespace NUMINAMATH_CALUDE_heartsuit_squared_neq_diamondsuit_l3571_357102

-- Define the heartsuit operation
def heartsuit (x y : ℝ) : ℝ := |x - y|

-- Define the diamondsuit operation
def diamondsuit (z w : ℝ) : ℝ := (z + w)^2

-- Theorem statement
theorem heartsuit_squared_neq_diamondsuit :
  ∃ x y : ℝ, (heartsuit x y)^2 ≠ diamondsuit x y :=
sorry

end NUMINAMATH_CALUDE_heartsuit_squared_neq_diamondsuit_l3571_357102


namespace NUMINAMATH_CALUDE_unique_solution_condition_l3571_357192

theorem unique_solution_condition (a b : ℝ) : 
  (∃! x : ℝ, 5 * x - 7 + a = 2 * b * x + 3) ↔ (a ≠ 10 ∧ b ≠ 5/2) :=
sorry

end NUMINAMATH_CALUDE_unique_solution_condition_l3571_357192


namespace NUMINAMATH_CALUDE_hyperbola_eccentricity_sqrt_6_l3571_357100

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a parabola with equation y² = 4x -/
def Parabola := {p : Point | p.y^2 = 4 * p.x}

/-- Represents a hyperbola with equation x²/a² - y² = 1 -/
def Hyperbola (a : ℝ) := {p : Point | p.x^2 / a^2 - p.y^2 = 1}

/-- The directrix of the parabola y² = 4x -/
def directrix : Set Point := {p : Point | p.x = -1}

/-- The focus of the parabola y² = 4x -/
def focus : Point := ⟨1, 0⟩

/-- Predicate to check if three points form a right-angled triangle -/
def isRightTriangle (p q r : Point) : Prop := sorry

/-- The eccentricity of a hyperbola -/
def hyperbolaEccentricity (a : ℝ) : ℝ := sorry

theorem hyperbola_eccentricity_sqrt_6 (a : ℝ) (A B : Point) :
  A ∈ Hyperbola a →
  B ∈ Hyperbola a →
  A ∈ directrix →
  B ∈ directrix →
  isRightTriangle A B focus →
  hyperbolaEccentricity a = Real.sqrt 6 := by
  sorry

end NUMINAMATH_CALUDE_hyperbola_eccentricity_sqrt_6_l3571_357100


namespace NUMINAMATH_CALUDE_company_match_percentage_l3571_357146

/-- Proves that the company's 401K match percentage is 6% given the problem conditions --/
theorem company_match_percentage (
  paychecks_per_year : ℕ)
  (contribution_per_paycheck : ℚ)
  (total_contribution : ℚ)
  (h1 : paychecks_per_year = 26)
  (h2 : contribution_per_paycheck = 100)
  (h3 : total_contribution = 2756) :
  (total_contribution - (paychecks_per_year : ℚ) * contribution_per_paycheck) /
  ((paychecks_per_year : ℚ) * contribution_per_paycheck) * 100 = 6 :=
by sorry

end NUMINAMATH_CALUDE_company_match_percentage_l3571_357146


namespace NUMINAMATH_CALUDE_max_value_theorem_l3571_357125

theorem max_value_theorem (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a + b = 1) :
  ∃ (max : ℝ), max = -9/2 ∧ ∀ x y, x > 0 → y > 0 → x + y = 1 → -1/(2*x) - 2/y ≤ max :=
by sorry

end NUMINAMATH_CALUDE_max_value_theorem_l3571_357125


namespace NUMINAMATH_CALUDE_stratified_sample_composition_l3571_357197

theorem stratified_sample_composition 
  (total_athletes : ℕ) 
  (male_athletes : ℕ) 
  (female_athletes : ℕ) 
  (sample_size : ℕ) 
  (h1 : total_athletes = male_athletes + female_athletes)
  (h2 : total_athletes = 98)
  (h3 : male_athletes = 56)
  (h4 : female_athletes = 42)
  (h5 : sample_size = 14) :
  (male_athletes * sample_size / total_athletes : ℚ) = 8 ∧ 
  (female_athletes * sample_size / total_athletes : ℚ) = 6 :=
by sorry

end NUMINAMATH_CALUDE_stratified_sample_composition_l3571_357197


namespace NUMINAMATH_CALUDE_sat_score_improvement_l3571_357185

theorem sat_score_improvement (first_score second_score : ℝ) : 
  (second_score = first_score * 1.1) → 
  (second_score = 1100) → 
  (first_score = 1000) := by
sorry

end NUMINAMATH_CALUDE_sat_score_improvement_l3571_357185


namespace NUMINAMATH_CALUDE_biathlon_run_distance_l3571_357154

/-- Biathlon problem -/
theorem biathlon_run_distance
  (total_distance : ℝ)
  (bicycle_distance : ℝ)
  (bicycle_velocity : ℝ)
  (total_time : ℝ)
  (h1 : total_distance = 155)
  (h2 : bicycle_distance = 145)
  (h3 : bicycle_velocity = 29)
  (h4 : total_time = 6)
  : total_distance - bicycle_distance = 10 := by
  sorry

end NUMINAMATH_CALUDE_biathlon_run_distance_l3571_357154


namespace NUMINAMATH_CALUDE_total_spent_is_158_40_l3571_357164

/-- Calculates the total amount spent on a meal given the food price, sales tax rate, and tip rate -/
def total_amount_spent (food_price : ℝ) (sales_tax_rate : ℝ) (tip_rate : ℝ) : ℝ :=
  let price_with_tax := food_price * (1 + sales_tax_rate)
  let tip := price_with_tax * tip_rate
  price_with_tax + tip

/-- Theorem stating that the total amount spent is $158.40 given the conditions -/
theorem total_spent_is_158_40 :
  total_amount_spent 120 0.1 0.2 = 158.40 := by
  sorry

end NUMINAMATH_CALUDE_total_spent_is_158_40_l3571_357164


namespace NUMINAMATH_CALUDE_arithmetic_mean_of_fractions_l3571_357127

theorem arithmetic_mean_of_fractions :
  let a := 5 / 8
  let b := 9 / 16
  let c := 11 / 16
  a = (b + c) / 2 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_mean_of_fractions_l3571_357127


namespace NUMINAMATH_CALUDE_det_of_matrix_l3571_357166

def matrix : Matrix (Fin 2) (Fin 2) ℤ := !![5, -4; 2, 3]

theorem det_of_matrix : Matrix.det matrix = 23 := by sorry

end NUMINAMATH_CALUDE_det_of_matrix_l3571_357166


namespace NUMINAMATH_CALUDE_corner_sum_9x9_l3571_357171

def checkerboard_size : ℕ := 9

def corner_sum (n : ℕ) : ℕ :=
  1 + n + (n^2 - n + 1) + n^2

theorem corner_sum_9x9 :
  corner_sum checkerboard_size = 164 :=
by sorry

end NUMINAMATH_CALUDE_corner_sum_9x9_l3571_357171


namespace NUMINAMATH_CALUDE_no_rain_probability_l3571_357137

theorem no_rain_probability (p_rain_5th p_rain_6th : ℝ) 
  (h1 : p_rain_5th = 0.2) 
  (h2 : p_rain_6th = 0.4) 
  (h3 : 0 ≤ p_rain_5th ∧ p_rain_5th ≤ 1) 
  (h4 : 0 ≤ p_rain_6th ∧ p_rain_6th ≤ 1) :
  (1 - p_rain_5th) * (1 - p_rain_6th) = 0.48 := by
  sorry

end NUMINAMATH_CALUDE_no_rain_probability_l3571_357137


namespace NUMINAMATH_CALUDE_expression_evaluation_l3571_357128

theorem expression_evaluation (y : ℝ) : 
  (1 : ℝ)^(4*y - 1) / (2 * ((7 : ℝ)⁻¹ + (4 : ℝ)⁻¹)) = 14/11 := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l3571_357128


namespace NUMINAMATH_CALUDE_intersection_equals_N_implies_t_range_l3571_357147

-- Define the sets M and N
def M : Set ℝ := {x | -4 < x ∧ x < 3}
def N (t : ℝ) : Set ℝ := {x | t + 2 < x ∧ x < 2*t - 1}

-- State the theorem
theorem intersection_equals_N_implies_t_range (t : ℝ) : 
  M ∩ N t = N t → t ≤ 3 :=
by sorry

end NUMINAMATH_CALUDE_intersection_equals_N_implies_t_range_l3571_357147


namespace NUMINAMATH_CALUDE_long_tennis_players_l3571_357107

theorem long_tennis_players (total : ℕ) (football : ℕ) (both : ℕ) (neither : ℕ) :
  total = 40 →
  football = 26 →
  both = 17 →
  neither = 11 →
  ∃ long_tennis : ℕ,
    long_tennis = 20 ∧
    total = football + long_tennis - both + neither :=
by
  sorry

end NUMINAMATH_CALUDE_long_tennis_players_l3571_357107


namespace NUMINAMATH_CALUDE_smallest_absolute_value_l3571_357118

theorem smallest_absolute_value : ∃ x : ℝ, ∀ y : ℝ, abs x ≤ abs y :=
  sorry

end NUMINAMATH_CALUDE_smallest_absolute_value_l3571_357118


namespace NUMINAMATH_CALUDE_max_value_of_y_l3571_357111

open Complex

theorem max_value_of_y (θ : Real) (h : 0 < θ ∧ θ < Real.pi / 2) :
  let z : ℂ := 3 * cos θ + 2 * I * sin θ
  let y : Real := θ - arg z
  ∃ (max_y : Real), ∀ (θ' : Real), 0 < θ' ∧ θ' < Real.pi / 2 →
    let z' : ℂ := 3 * cos θ' + 2 * I * sin θ'
    let y' : Real := θ' - arg z'
    y' ≤ max_y ∧ max_y = Real.arctan (Real.sqrt 6 / 12) := by
  sorry

end NUMINAMATH_CALUDE_max_value_of_y_l3571_357111


namespace NUMINAMATH_CALUDE_billys_cherries_billys_cherries_proof_l3571_357103

theorem billys_cherries : ℕ → ℕ → ℕ → Prop :=
  fun initial eaten left =>
    initial = 74 ∧ eaten = 72 ∧ left = initial - eaten → left = 2

-- The proof is omitted as per instructions
theorem billys_cherries_proof : billys_cherries 74 72 2 := by sorry

end NUMINAMATH_CALUDE_billys_cherries_billys_cherries_proof_l3571_357103


namespace NUMINAMATH_CALUDE_equation_represents_three_lines_lines_do_not_all_intersect_l3571_357132

-- Define the equation
def equation (x y : ℝ) : Prop :=
  x^2 * (x - y - 2) = y^2 * (x - y - 2)

-- Define the three lines
def line1 (x y : ℝ) : Prop := y = x
def line2 (x y : ℝ) : Prop := y = -x
def line3 (x y : ℝ) : Prop := y = x - 2

-- Theorem statement
theorem equation_represents_three_lines :
  ∀ x y : ℝ, equation x y ↔ (line1 x y ∨ line2 x y ∨ line3 x y) :=
by sorry

-- Theorem stating that the three lines do not all intersect at a common point
theorem lines_do_not_all_intersect :
  ¬∃ x y : ℝ, line1 x y ∧ line2 x y ∧ line3 x y :=
by sorry

end NUMINAMATH_CALUDE_equation_represents_three_lines_lines_do_not_all_intersect_l3571_357132


namespace NUMINAMATH_CALUDE_triangle_bd_length_l3571_357169

-- Define the triangle ABC
structure Triangle :=
  (A B C : ℝ × ℝ)

-- Define the point D on AB
def D (t : Triangle) : ℝ × ℝ := sorry

-- State the theorem
theorem triangle_bd_length (t : Triangle) :
  -- Conditions
  (dist t.A t.C = 7) →
  (dist t.B t.C = 7) →
  (dist t.A (D t) = 8) →
  (dist t.C (D t) = 3) →
  -- Conclusion
  (dist t.B (D t) = 5) := by
  sorry

where
  dist : (ℝ × ℝ) → (ℝ × ℝ) → ℝ := sorry

end NUMINAMATH_CALUDE_triangle_bd_length_l3571_357169


namespace NUMINAMATH_CALUDE_no_negative_roots_l3571_357167

theorem no_negative_roots :
  ∀ x : ℝ, x < 0 → x^4 - 5*x^3 - 4*x^2 - 7*x + 4 ≠ 0 := by
sorry

end NUMINAMATH_CALUDE_no_negative_roots_l3571_357167


namespace NUMINAMATH_CALUDE_max_steps_to_empty_l3571_357135

/-- A function that checks if a natural number has repeated digits -/
def has_repeated_digits (n : ℕ) : Bool :=
  sorry

/-- A function that represents one step of the process -/
def step (list : List ℕ) : List ℕ :=
  sorry

/-- The initial list of the first 1000 positive integers -/
def initial_list : List ℕ :=
  sorry

/-- The number of steps required to empty the list -/
def steps_to_empty (list : List ℕ) : ℕ :=
  sorry

theorem max_steps_to_empty : steps_to_empty initial_list = 11 :=
  sorry

end NUMINAMATH_CALUDE_max_steps_to_empty_l3571_357135


namespace NUMINAMATH_CALUDE_interest_rate_difference_l3571_357108

/-- Proves that the difference in interest rates is 1% given the problem conditions --/
theorem interest_rate_difference 
  (principal : ℝ) 
  (time : ℝ) 
  (additional_interest : ℝ) 
  (h1 : principal = 2400) 
  (h2 : time = 3) 
  (h3 : additional_interest = 72) : 
  ∃ (r dr : ℝ), 
    principal * ((r + dr) / 100) * time - principal * (r / 100) * time = additional_interest ∧ 
    dr = 1 := by
  sorry

end NUMINAMATH_CALUDE_interest_rate_difference_l3571_357108


namespace NUMINAMATH_CALUDE_no_polyhedron_with_area_2015_l3571_357104

theorem no_polyhedron_with_area_2015 : ¬ ∃ (n k : ℕ), 6 * n - 2 * k = 2015 := by
  sorry

end NUMINAMATH_CALUDE_no_polyhedron_with_area_2015_l3571_357104


namespace NUMINAMATH_CALUDE_y_divisibility_l3571_357122

def y : ℕ := 48 + 72 + 144 + 216 + 432 + 648 + 2592

theorem y_divisibility :
  (∃ k : ℕ, y = 3 * k) ∧
  (∃ k : ℕ, y = 6 * k) ∧
  ¬(∀ k : ℕ, y = 9 * k) ∧
  ¬(∃ k : ℕ, y = 18 * k) :=
by sorry

end NUMINAMATH_CALUDE_y_divisibility_l3571_357122


namespace NUMINAMATH_CALUDE_tangent_point_coordinates_l3571_357133

theorem tangent_point_coordinates :
  ∀ x y : ℝ,
  y = x^2 →
  (2 : ℝ) = 2*x →
  x = 1 ∧ y = 1 :=
by sorry

end NUMINAMATH_CALUDE_tangent_point_coordinates_l3571_357133


namespace NUMINAMATH_CALUDE_xy_value_l3571_357119

theorem xy_value (x y : ℝ) (h1 : x + y = 9) (h2 : x^3 + y^3 = 351) : x * y = 14 := by
  sorry

end NUMINAMATH_CALUDE_xy_value_l3571_357119


namespace NUMINAMATH_CALUDE_y_derivative_l3571_357195

noncomputable def y (x : ℝ) : ℝ := 
  (Real.cos (Real.tan (1/3)) * (Real.sin (15*x))^2) / (15 * Real.cos (30*x))

theorem y_derivative (x : ℝ) : 
  deriv y x = (Real.cos (Real.tan (1/3)) * Real.tan (30*x)) / Real.cos (30*x) :=
by sorry

end NUMINAMATH_CALUDE_y_derivative_l3571_357195


namespace NUMINAMATH_CALUDE_arithmetic_sequence_sum_l3571_357123

/-- Given an arithmetic sequence with first term 3 and common difference 12,
    prove that the sum of the first 30 terms is 5310. -/
theorem arithmetic_sequence_sum : 
  let a : ℕ → ℤ := fun n => 3 + (n - 1) * 12
  let S : ℕ → ℤ := fun n => n * (a 1 + a n) / 2
  S 30 = 5310 := by sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_sum_l3571_357123


namespace NUMINAMATH_CALUDE_A_intersect_B_equals_three_l3571_357131

def A : Set ℝ := {0, 1, 3}
def B : Set ℝ := {x | ∃ y, y = Real.log (x - 1)}

theorem A_intersect_B_equals_three : A ∩ B = {3} := by sorry

end NUMINAMATH_CALUDE_A_intersect_B_equals_three_l3571_357131


namespace NUMINAMATH_CALUDE_inequality_and_equality_condition_l3571_357181

theorem inequality_and_equality_condition (x₁ x₂ : ℝ) 
  (h₁ : |x₁| ≤ 1) (h₂ : |x₂| ≤ 1) : 
  Real.sqrt (1 - x₁^2) + Real.sqrt (1 - x₂^2) ≤ 2 * Real.sqrt (1 - ((x₁ + x₂)/2)^2) ∧ 
  (Real.sqrt (1 - x₁^2) + Real.sqrt (1 - x₂^2) = 2 * Real.sqrt (1 - ((x₁ + x₂)/2)^2) ↔ x₁ = x₂) :=
by sorry

end NUMINAMATH_CALUDE_inequality_and_equality_condition_l3571_357181


namespace NUMINAMATH_CALUDE_x_minus_y_equals_one_l3571_357199

-- Define x and y based on the given conditions
def x : Int := 2 - 4 + 6
def y : Int := 1 - 3 + 5

-- State the theorem to be proved
theorem x_minus_y_equals_one : x - y = 1 := by
  sorry

end NUMINAMATH_CALUDE_x_minus_y_equals_one_l3571_357199


namespace NUMINAMATH_CALUDE_inverse_sum_mod_31_l3571_357158

theorem inverse_sum_mod_31 : ∃ (a b : ℤ), (5 * a) % 31 = 1 ∧ (5 * 5 * 5 * b) % 31 = 1 ∧ (a + b) % 31 = 5 := by
  sorry

end NUMINAMATH_CALUDE_inverse_sum_mod_31_l3571_357158


namespace NUMINAMATH_CALUDE_rectangle_width_length_ratio_l3571_357115

theorem rectangle_width_length_ratio (w : ℝ) : 
  w > 0 ∧ 2 * w + 2 * 10 = 30 → w / 10 = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_width_length_ratio_l3571_357115


namespace NUMINAMATH_CALUDE_cube_root_sum_equation_l3571_357172

theorem cube_root_sum_equation (x : ℝ) :
  x = (11 + Real.sqrt 337) ^ (1/3 : ℝ) + (11 - Real.sqrt 337) ^ (1/3 : ℝ) →
  x^3 + 18*x = 22 := by
sorry

end NUMINAMATH_CALUDE_cube_root_sum_equation_l3571_357172


namespace NUMINAMATH_CALUDE_root_product_theorem_l3571_357148

theorem root_product_theorem (y₁ y₂ y₃ y₄ y₅ : ℂ) : 
  (y₁^5 - 3*y₁^3 + 2 = 0) →
  (y₂^5 - 3*y₂^3 + 2 = 0) →
  (y₃^5 - 3*y₃^3 + 2 = 0) →
  (y₄^5 - 3*y₄^3 + 2 = 0) →
  (y₅^5 - 3*y₅^3 + 2 = 0) →
  ((y₁^2 - 3) * (y₂^2 - 3) * (y₃^2 - 3) * (y₄^2 - 3) * (y₅^2 - 3) = -32) :=
by sorry

end NUMINAMATH_CALUDE_root_product_theorem_l3571_357148


namespace NUMINAMATH_CALUDE_down_jacket_price_reduction_l3571_357187

/-- Represents the price reduction problem for down jackets --/
theorem down_jacket_price_reduction
  (initial_sales : ℕ)
  (initial_profit_per_piece : ℕ)
  (sales_increase_per_yuan : ℕ)
  (target_daily_profit : ℕ)
  (h1 : initial_sales = 20)
  (h2 : initial_profit_per_piece = 40)
  (h3 : sales_increase_per_yuan = 2)
  (h4 : target_daily_profit = 1200) :
  ∃ (price_reduction : ℕ),
    (initial_profit_per_piece - price_reduction) *
    (initial_sales + sales_increase_per_yuan * price_reduction) = target_daily_profit ∧
    price_reduction = 20 :=
by sorry

end NUMINAMATH_CALUDE_down_jacket_price_reduction_l3571_357187


namespace NUMINAMATH_CALUDE_decimal_to_scientific_notation_l3571_357193

/-- Expresses a given decimal number in scientific notation -/
def scientific_notation (x : ℝ) : ℝ × ℤ :=
  sorry

theorem decimal_to_scientific_notation :
  scientific_notation 0.00000011 = (1.1, -7) :=
sorry

end NUMINAMATH_CALUDE_decimal_to_scientific_notation_l3571_357193


namespace NUMINAMATH_CALUDE_total_hot_dogs_today_l3571_357186

def hot_dogs_lunch : ℕ := 9
def hot_dogs_dinner : ℕ := 2

theorem total_hot_dogs_today : hot_dogs_lunch + hot_dogs_dinner = 11 := by
  sorry

end NUMINAMATH_CALUDE_total_hot_dogs_today_l3571_357186


namespace NUMINAMATH_CALUDE_bicycle_cost_proof_l3571_357112

def bicycle_cost (car_wash_income : ℕ) (lawn_mow_income : ℕ) (additional_needed : ℕ) : ℕ :=
  car_wash_income + lawn_mow_income + additional_needed

theorem bicycle_cost_proof :
  let car_wash_income := 3 * 10
  let lawn_mow_income := 2 * 13
  let additional_needed := 24
  bicycle_cost car_wash_income lawn_mow_income additional_needed = 80 := by
  sorry

end NUMINAMATH_CALUDE_bicycle_cost_proof_l3571_357112


namespace NUMINAMATH_CALUDE_second_book_cost_l3571_357155

/-- Proves that the cost of the second book is $4 given the conditions of Shelby's book fair purchases. -/
theorem second_book_cost (initial_amount : ℕ) (first_book_cost : ℕ) (poster_cost : ℕ) (posters_bought : ℕ) :
  initial_amount = 20 →
  first_book_cost = 8 →
  poster_cost = 4 →
  posters_bought = 2 →
  ∃ (second_book_cost : ℕ),
    second_book_cost + first_book_cost + (poster_cost * posters_bought) = initial_amount ∧
    second_book_cost = 4 :=
by
  sorry

end NUMINAMATH_CALUDE_second_book_cost_l3571_357155


namespace NUMINAMATH_CALUDE_inverse_f_zero_l3571_357160

noncomputable def f (a b : ℝ) (x : ℝ) : ℝ := 1 / (2 * a * x + 3 * b)

theorem inverse_f_zero (a b : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) :
  ∃ x, f a b x = 1 / (3 * b) ∧ (∀ y, f a b y = x → y = 0) :=
by sorry

end NUMINAMATH_CALUDE_inverse_f_zero_l3571_357160


namespace NUMINAMATH_CALUDE_unique_x_l3571_357138

theorem unique_x : ∃! x : ℕ, x > 0 ∧ ∃ k : ℕ, x = 9 * k ∧ x^2 < 200 ∧ x < 25 := by
  sorry

end NUMINAMATH_CALUDE_unique_x_l3571_357138


namespace NUMINAMATH_CALUDE_polygon_d_largest_area_l3571_357110

-- Define the structure of a polygon
structure Polygon where
  unitSquares : ℕ
  rightTriangles : ℕ

-- Define the area calculation function
def area (p : Polygon) : ℚ :=
  p.unitSquares + p.rightTriangles / 2

-- Define the five polygons
def polygonA : Polygon := ⟨6, 0⟩
def polygonB : Polygon := ⟨3, 4⟩
def polygonC : Polygon := ⟨4, 5⟩
def polygonD : Polygon := ⟨7, 0⟩
def polygonE : Polygon := ⟨2, 6⟩

-- Define the list of all polygons
def allPolygons : List Polygon := [polygonA, polygonB, polygonC, polygonD, polygonE]

-- Theorem: Polygon D has the largest area
theorem polygon_d_largest_area :
  ∀ p ∈ allPolygons, area polygonD ≥ area p :=
sorry

end NUMINAMATH_CALUDE_polygon_d_largest_area_l3571_357110


namespace NUMINAMATH_CALUDE_remainder_3056_div_78_l3571_357159

theorem remainder_3056_div_78 : 3056 % 78 = 14 := by
  sorry

end NUMINAMATH_CALUDE_remainder_3056_div_78_l3571_357159


namespace NUMINAMATH_CALUDE_complement_A_in_U_l3571_357189

def U : Set ℕ := {x | x ≥ 3}
def A : Set ℕ := {x | x^2 ≥ 10}

theorem complement_A_in_U : (U \ A) = {3} := by sorry

end NUMINAMATH_CALUDE_complement_A_in_U_l3571_357189


namespace NUMINAMATH_CALUDE_ratio_equality_l3571_357106

variables {a b c : ℝ}

theorem ratio_equality (h1 : 7 * a = 8 * b) (h2 : 4 * a + 3 * c = 11 * b) (h3 : 2 * c - b = 5 * a) 
  (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0) : a / 8 = b / 7 := by
  sorry

end NUMINAMATH_CALUDE_ratio_equality_l3571_357106


namespace NUMINAMATH_CALUDE_vovochka_max_candies_l3571_357113

/-- Represents the problem of distributing candies among classmates. -/
structure CandyDistribution where
  totalCandies : Nat
  totalClassmates : Nat
  minGroupSize : Nat
  minGroupCandies : Nat

/-- Calculates the maximum number of candies Vovochka can keep. -/
def maxCandiesForVovochka (dist : CandyDistribution) : Nat :=
  sorry

/-- Theorem stating the maximum number of candies Vovochka can keep. -/
theorem vovochka_max_candies :
  let dist : CandyDistribution := {
    totalCandies := 200,
    totalClassmates := 25,
    minGroupSize := 16,
    minGroupCandies := 100
  }
  maxCandiesForVovochka dist = 37 := by sorry

end NUMINAMATH_CALUDE_vovochka_max_candies_l3571_357113


namespace NUMINAMATH_CALUDE_sum_of_coefficients_l3571_357165

-- Define the sets A and B
def A : Set ℝ := {x : ℝ | x^3 + 3*x^2 + 2*x > 0}
def B (a b : ℝ) : Set ℝ := {x : ℝ | x^2 + a*x + b ≤ 0}

-- State the theorem
theorem sum_of_coefficients (a b : ℝ) :
  (A ∩ B a b = {x : ℝ | 0 < x ∧ x ≤ 2}) ∧
  (A ∪ B a b = {x : ℝ | x > -2}) →
  a + b = -3 := by
sorry

end NUMINAMATH_CALUDE_sum_of_coefficients_l3571_357165


namespace NUMINAMATH_CALUDE_doris_earnings_l3571_357170

/-- Calculates the number of weeks needed for Doris to earn enough to cover her monthly expenses --/
def weeks_to_earn_expenses (hourly_rate : ℚ) (weekday_hours : ℚ) (saturday_hours : ℚ) (monthly_expense : ℚ) : ℚ :=
  let weekly_hours := 5 * weekday_hours + saturday_hours
  let weekly_earnings := weekly_hours * hourly_rate
  monthly_expense / weekly_earnings

theorem doris_earnings : 
  let hourly_rate : ℚ := 20
  let weekday_hours : ℚ := 3
  let saturday_hours : ℚ := 5
  let monthly_expense : ℚ := 1200
  weeks_to_earn_expenses hourly_rate weekday_hours saturday_hours monthly_expense = 3 := by
  sorry

end NUMINAMATH_CALUDE_doris_earnings_l3571_357170


namespace NUMINAMATH_CALUDE_complex_magnitude_squared_l3571_357141

theorem complex_magnitude_squared (z : ℂ) (h : z^2 + Complex.abs z^2 = 4 + 6*I) : 
  Complex.abs z^2 = 13/2 := by
  sorry

end NUMINAMATH_CALUDE_complex_magnitude_squared_l3571_357141


namespace NUMINAMATH_CALUDE_eleven_by_eleven_grid_segment_length_l3571_357121

/-- Represents a grid of lattice points -/
structure LatticeGrid where
  rows : ℕ
  columns : ℕ

/-- Calculates the total length of segments in a lattice grid -/
def totalSegmentLength (grid : LatticeGrid) : ℕ :=
  (grid.rows - 1) * grid.columns + (grid.columns - 1) * grid.rows

/-- Theorem: The total length of segments in an 11x11 lattice grid is 220 -/
theorem eleven_by_eleven_grid_segment_length :
  totalSegmentLength ⟨11, 11⟩ = 220 := by
  sorry

#eval totalSegmentLength ⟨11, 11⟩

end NUMINAMATH_CALUDE_eleven_by_eleven_grid_segment_length_l3571_357121


namespace NUMINAMATH_CALUDE_unique_polygon_diagonals_l3571_357136

/-- The number of diagonals in a convex polygon with k sides -/
def numDiagonals (k : ℕ) : ℚ := (k * (k - 3)) / 2

/-- The condition for the number of diagonals in the two polygons -/
def diagonalCondition (n : ℕ) : Prop :=
  numDiagonals (3 * n + 2) = (1 - 0.615) * numDiagonals (5 * n - 2)

theorem unique_polygon_diagonals : ∃! (n : ℕ), n > 0 ∧ diagonalCondition n :=
  sorry

end NUMINAMATH_CALUDE_unique_polygon_diagonals_l3571_357136


namespace NUMINAMATH_CALUDE_triangle_properties_l3571_357129

/-- Given a triangle ABC with sides a, b, c and angles A, B, C -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

/-- The theorem statement -/
theorem triangle_properties (t : Triangle) 
  (h1 : (2 * t.a - t.c) * Real.cos t.B = t.b * Real.cos t.C) 
  (h2 : t.a + t.c = 6) 
  (h3 : (1/2) * t.a * t.c * Real.sin t.B = Real.sqrt 3) : 
  t.B = π/3 ∧ t.a + t.b + t.c = 6 + 2 * Real.sqrt 6 := by
  sorry

end NUMINAMATH_CALUDE_triangle_properties_l3571_357129


namespace NUMINAMATH_CALUDE_melanie_dimes_value_l3571_357130

def initial_dimes : ℕ := 7
def dimes_from_dad : ℕ := 8
def dimes_from_mom : ℕ := 4
def dime_value : ℚ := 0.1

def total_dimes : ℕ := initial_dimes + dimes_from_dad + dimes_from_mom

theorem melanie_dimes_value :
  (total_dimes : ℚ) * dime_value = 1.9 := by sorry

end NUMINAMATH_CALUDE_melanie_dimes_value_l3571_357130


namespace NUMINAMATH_CALUDE_F_36_72_equals_48_max_F_happy_pair_equals_58_l3571_357162

/-- Function F calculates the sum of products of digits in two-digit numbers -/
def F (m n : ℕ) : ℕ :=
  (m / 10) * (n % 10) + (m % 10) * (n / 10)

/-- Swaps the digits of a two-digit number -/
def swapDigits (m : ℕ) : ℕ :=
  (m % 10) * 10 + (m / 10)

/-- Checks if two numbers form a "happy pair" -/
def isHappyPair (m n : ℕ) : Prop :=
  ∃ (a b : ℕ), 1 ≤ a ∧ a ≤ 4 ∧ 1 ≤ b ∧ b ≤ 5 ∧
  m = 21 * a + b ∧ n = 53 + b ∧
  (swapDigits m + 5 * (n % 10)) % 11 = 0

theorem F_36_72_equals_48 : F 36 72 = 48 := by sorry

theorem max_F_happy_pair_equals_58 :
  (∃ (m n : ℕ), isHappyPair m n ∧ F m n = 58) ∧
  (∀ (m n : ℕ), isHappyPair m n → F m n ≤ 58) := by sorry

end NUMINAMATH_CALUDE_F_36_72_equals_48_max_F_happy_pair_equals_58_l3571_357162


namespace NUMINAMATH_CALUDE_min_distance_line_parabola_l3571_357120

noncomputable def line (x : ℝ) : ℝ := (15/8) * x - 8

noncomputable def parabola (x : ℝ) : ℝ := x^2

theorem min_distance_line_parabola :
  ∃ (x₁ x₂ : ℝ),
    (∀ y₁ y₂ : ℝ,
      y₁ = line x₁ ∧ y₂ = parabola x₂ →
      (x₂ - x₁)^2 + (y₂ - y₁)^2 ≥ (1823/544)^2) :=
by sorry

end NUMINAMATH_CALUDE_min_distance_line_parabola_l3571_357120


namespace NUMINAMATH_CALUDE_paint_area_calculation_l3571_357196

/-- The height of the wall in feet -/
def wall_height : ℝ := 10

/-- The length of the wall in feet -/
def wall_length : ℝ := 15

/-- The height of the door in feet -/
def door_height : ℝ := 3

/-- The width of the door in feet -/
def door_width : ℝ := 5

/-- The area to paint in square feet -/
def area_to_paint : ℝ := wall_height * wall_length - door_height * door_width

theorem paint_area_calculation :
  area_to_paint = 135 := by sorry

end NUMINAMATH_CALUDE_paint_area_calculation_l3571_357196


namespace NUMINAMATH_CALUDE_plane_equation_theorem_l3571_357134

/-- Represents a point in 3D space -/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- Represents the coefficients of a plane equation Ax + By + Cz + D = 0 -/
structure PlaneCoefficients where
  A : ℤ
  B : ℤ
  C : ℤ
  D : ℤ

/-- The foot of the perpendicular from the origin to the plane -/
def footOfPerpendicular : Point3D :=
  { x := 10, y := -5, z := 4 }

/-- Check if the given coefficients satisfy the required conditions -/
def validCoefficients (coeff : PlaneCoefficients) : Prop :=
  coeff.A > 0 ∧ Nat.gcd (Int.natAbs coeff.A) (Int.natAbs coeff.B) = 1 ∧
  Nat.gcd (Int.natAbs coeff.A) (Int.natAbs coeff.C) = 1 ∧
  Nat.gcd (Int.natAbs coeff.A) (Int.natAbs coeff.D) = 1

/-- Check if a point satisfies the plane equation -/
def satisfiesPlaneEquation (p : Point3D) (coeff : PlaneCoefficients) : Prop :=
  coeff.A * p.x + coeff.B * p.y + coeff.C * p.z + coeff.D = 0

/-- The main theorem to prove -/
theorem plane_equation_theorem :
  ∃ (coeff : PlaneCoefficients),
    validCoefficients coeff ∧
    satisfiesPlaneEquation footOfPerpendicular coeff ∧
    coeff.A = 10 ∧ coeff.B = -5 ∧ coeff.C = 4 ∧ coeff.D = -141 :=
sorry

end NUMINAMATH_CALUDE_plane_equation_theorem_l3571_357134


namespace NUMINAMATH_CALUDE_score_difference_theorem_l3571_357142

def score_distribution : List (Float × Float) := [
  (75, 0.15),
  (85, 0.30),
  (90, 0.25),
  (95, 0.10),
  (100, 0.20)
]

def mean (dist : List (Float × Float)) : Float :=
  (dist.map (fun (score, freq) => score * freq)).sum

def median (dist : List (Float × Float)) : Float :=
  90  -- The median is 90 based on the given distribution

theorem score_difference_theorem :
  mean score_distribution - median score_distribution = -1.25 := by
  sorry

end NUMINAMATH_CALUDE_score_difference_theorem_l3571_357142


namespace NUMINAMATH_CALUDE_ice_cream_arrangement_l3571_357105

theorem ice_cream_arrangement (n : ℕ) (k : ℕ) (h1 : n = 5) (h2 : k = 2) :
  (n! / k!) = 60 := by
  sorry

end NUMINAMATH_CALUDE_ice_cream_arrangement_l3571_357105


namespace NUMINAMATH_CALUDE_intersection_when_m_neg_three_subset_condition_equivalent_l3571_357173

-- Define sets A and B
def A : Set ℝ := {x | -3 ≤ x ∧ x ≤ 4}
def B (m : ℝ) : Set ℝ := {x | 2*m - 1 ≤ x ∧ x ≤ m + 1}

-- Theorem for the first part of the problem
theorem intersection_when_m_neg_three :
  A ∩ B (-3) = {x | -3 ≤ x ∧ x ≤ -2} := by sorry

-- Theorem for the second part of the problem
theorem subset_condition_equivalent :
  ∀ m : ℝ, B m ⊆ A ↔ m ≥ -1 := by sorry

end NUMINAMATH_CALUDE_intersection_when_m_neg_three_subset_condition_equivalent_l3571_357173

import Mathlib

namespace NUMINAMATH_CALUDE_second_term_of_geometric_sequence_l711_71133

def geometric_sequence (a : ℕ → ℚ) : Prop :=
  ∃ r : ℚ, ∀ n : ℕ, a (n + 1) = r * a n

theorem second_term_of_geometric_sequence
  (a : ℕ → ℚ)
  (h_geometric : geometric_sequence a)
  (h_third : a 3 = 12)
  (h_fourth : a 4 = 18) :
  a 2 = 8 :=
sorry

end NUMINAMATH_CALUDE_second_term_of_geometric_sequence_l711_71133


namespace NUMINAMATH_CALUDE_solve_equation_l711_71159

theorem solve_equation (y z : ℝ) (h1 : y = -2.6) (h2 : z = 4.3) :
  ∃ x : ℝ, 5 * x - 2 * y + 3.7 * z = 1.45 ∧ x = -3.932 := by
  sorry

end NUMINAMATH_CALUDE_solve_equation_l711_71159


namespace NUMINAMATH_CALUDE_max_sections_five_lines_l711_71115

/- Define a function that calculates the maximum number of sections -/
def max_sections (n : ℕ) : ℕ :=
  if n = 0 then 1
  else if n = 1 then 2
  else 2 + (n - 1) * n / 2

/- Theorem statement -/
theorem max_sections_five_lines :
  max_sections 5 = 16 :=
by sorry

end NUMINAMATH_CALUDE_max_sections_five_lines_l711_71115


namespace NUMINAMATH_CALUDE_bus_seating_capacity_l711_71170

theorem bus_seating_capacity :
  let left_seats : ℕ := 15
  let right_seats : ℕ := left_seats - 3
  let people_per_seat : ℕ := 3
  let back_seat_capacity : ℕ := 7
  let total_capacity : ℕ := left_seats * people_per_seat + right_seats * people_per_seat + back_seat_capacity
  total_capacity = 88 := by
  sorry

end NUMINAMATH_CALUDE_bus_seating_capacity_l711_71170


namespace NUMINAMATH_CALUDE_shems_earnings_proof_l711_71124

/-- Calculates Shem's earnings for a workday given Kem's hourly rate, Shem's rate multiplier, and hours worked. -/
def shems_daily_earnings (kems_hourly_rate : ℝ) (shems_rate_multiplier : ℝ) (hours_worked : ℝ) : ℝ :=
  kems_hourly_rate * shems_rate_multiplier * hours_worked

/-- Proves that Shem's earnings for an 8-hour workday is $80, given the conditions. -/
theorem shems_earnings_proof (kems_hourly_rate : ℝ) (shems_rate_multiplier : ℝ) (hours_worked : ℝ) 
    (h1 : kems_hourly_rate = 4)
    (h2 : shems_rate_multiplier = 2.5)
    (h3 : hours_worked = 8) :
    shems_daily_earnings kems_hourly_rate shems_rate_multiplier hours_worked = 80 := by
  sorry

end NUMINAMATH_CALUDE_shems_earnings_proof_l711_71124


namespace NUMINAMATH_CALUDE_inverse_graph_coordinate_sum_l711_71188

-- Define the function f
noncomputable def f : ℝ → ℝ := sorry

-- Define the theorem
theorem inverse_graph_coordinate_sum :
  (∃ (f : ℝ → ℝ), f 2 = 4 ∧ (∃ (x : ℝ), f⁻¹ x = 2 ∧ x / 4 = 1 / 2)) →
  (∃ (x y : ℝ), y = f⁻¹ x / 4 ∧ x + y = 9 / 2) :=
by sorry

end NUMINAMATH_CALUDE_inverse_graph_coordinate_sum_l711_71188


namespace NUMINAMATH_CALUDE_fraction_integrality_l711_71197

theorem fraction_integrality (x y : ℕ) 
  (h : ∃ k : ℤ, (x^2 - 1 : ℚ) / (y + 1) + (y^2 - 1 : ℚ) / (x + 1) = k) : 
  ∃ m n : ℤ, (x^2 - 1 : ℚ) / (y + 1) = m ∧ (y^2 - 1 : ℚ) / (x + 1) = n :=
sorry

end NUMINAMATH_CALUDE_fraction_integrality_l711_71197


namespace NUMINAMATH_CALUDE_sum_of_roots_l711_71126

theorem sum_of_roots (p q r s : ℝ) : 
  p ≠ q ∧ p ≠ r ∧ p ≠ s ∧ q ≠ r ∧ q ≠ s ∧ r ≠ s →
  (∀ x : ℝ, x^2 - 12*p*x - 13*q = 0 ↔ x = r ∨ x = s) →
  (∀ x : ℝ, x^2 - 12*r*x - 13*s = 0 ↔ x = p ∨ x = q) →
  p + q + r + s = 2028 :=
by sorry

end NUMINAMATH_CALUDE_sum_of_roots_l711_71126


namespace NUMINAMATH_CALUDE_triangle_height_l711_71147

/-- Given a triangle with area 3 square meters and base 2 meters, its height is 3 meters -/
theorem triangle_height (area : ℝ) (base : ℝ) (height : ℝ) : 
  area = 3 → base = 2 → area = (base * height) / 2 → height = 3 := by
  sorry

end NUMINAMATH_CALUDE_triangle_height_l711_71147


namespace NUMINAMATH_CALUDE_G_equals_2F_l711_71193

noncomputable section

variable (x : ℝ)

def F (x : ℝ) : ℝ := Real.log ((1 + x) / (1 - x))

def G (x : ℝ) : ℝ := F ((x * (1 + x^2)) / (1 + x^4))

theorem G_equals_2F : G x = 2 * F x := by sorry

end NUMINAMATH_CALUDE_G_equals_2F_l711_71193


namespace NUMINAMATH_CALUDE_tangent_fraction_equality_l711_71110

theorem tangent_fraction_equality (α : Real) (h : Real.tan α = 3) :
  (4 * Real.sin α - 2 * Real.cos α) / (5 * Real.cos α + 3 * Real.sin α) = 5 / 7 := by
  sorry

end NUMINAMATH_CALUDE_tangent_fraction_equality_l711_71110


namespace NUMINAMATH_CALUDE_merchant_transaction_loss_l711_71174

theorem merchant_transaction_loss : 
  ∀ (cost_profit cost_loss : ℝ),
  cost_profit * 1.15 = 1955 →
  cost_loss * 0.85 = 1955 →
  (1955 + 1955) - (cost_profit + cost_loss) = -90 :=
by
  sorry

end NUMINAMATH_CALUDE_merchant_transaction_loss_l711_71174


namespace NUMINAMATH_CALUDE_triangle_coordinate_l711_71175

/-- Given a triangle with vertices A(0, 3), B(4, 0), and C(x, 5), where 0 < x < 4,
    if the area of the triangle is 8 square units, then x = 8/3. -/
theorem triangle_coordinate (x : ℝ) : 
  0 < x → x < 4 → 
  (1/2 : ℝ) * |0 * (0 - 5) + 4 * (5 - 3) + x * (3 - 0)| = 8 → 
  x = 8/3 :=
by sorry

end NUMINAMATH_CALUDE_triangle_coordinate_l711_71175


namespace NUMINAMATH_CALUDE_quadratic_root_one_l711_71155

theorem quadratic_root_one (a b c : ℝ) (h1 : a - b + c = 0) (h2 : a ≠ 0) :
  ∃ x : ℝ, a * x^2 - b * x + c = 0 ∧ x = 1 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_root_one_l711_71155


namespace NUMINAMATH_CALUDE_compound_rectangle_area_is_109_l711_71152

/-- The area of a compound rectangular shape -/
def compound_rectangle_area (main_width main_height top_right_width top_right_height bottom_right_width bottom_right_height : ℕ) : ℕ :=
  main_width * main_height - (top_right_width * top_right_height + bottom_right_width * bottom_right_height)

/-- Theorem stating that the area of the given compound rectangle is 109 square units -/
theorem compound_rectangle_area_is_109 :
  compound_rectangle_area 15 8 5 1 3 2 = 109 := by
  sorry

end NUMINAMATH_CALUDE_compound_rectangle_area_is_109_l711_71152


namespace NUMINAMATH_CALUDE_fourth_root_of_25000000_l711_71144

theorem fourth_root_of_25000000 : (70.7 : ℝ)^4 = 25000000 := by
  sorry

end NUMINAMATH_CALUDE_fourth_root_of_25000000_l711_71144


namespace NUMINAMATH_CALUDE_jar_water_problem_l711_71134

theorem jar_water_problem (s l w : ℝ) (hs : s > 0) (hl : l > 0) (hw : w > 0)
  (h1 : w = l / 2)  -- Larger jar is 1/2 full
  (h2 : w + w = 2 * l / 3)  -- When combined, 2/3 of larger jar is filled
  (h3 : s < l)  -- Smaller jar has less capacity
  : w = 3 * s / 4  -- Smaller jar was 3/4 full
  := by sorry

end NUMINAMATH_CALUDE_jar_water_problem_l711_71134


namespace NUMINAMATH_CALUDE_correct_parentheses_removal_l711_71141

theorem correct_parentheses_removal (x : ℝ) : -0.5 * (1 - 2 * x) = -0.5 + x := by
  sorry

end NUMINAMATH_CALUDE_correct_parentheses_removal_l711_71141


namespace NUMINAMATH_CALUDE_sara_received_four_onions_l711_71165

/-- The number of onions given to Sara -/
def onions_given_to_sara (sally_onions fred_onions remaining_onions : ℕ) : ℕ :=
  sally_onions + fred_onions - remaining_onions

/-- Theorem stating that Sara received 4 onions -/
theorem sara_received_four_onions :
  onions_given_to_sara 5 9 10 = 4 := by
  sorry

end NUMINAMATH_CALUDE_sara_received_four_onions_l711_71165


namespace NUMINAMATH_CALUDE_john_grandpa_money_l711_71125

theorem john_grandpa_money (x : ℝ) : 
  x > 0 ∧ x + 3 * x = 120 → x = 30 := by sorry

end NUMINAMATH_CALUDE_john_grandpa_money_l711_71125


namespace NUMINAMATH_CALUDE_sufficient_condition_for_vector_equality_l711_71102

variable {E : Type*} [NormedAddCommGroup E] [InnerProductSpace ℝ E]

theorem sufficient_condition_for_vector_equality 
  (a b : E) (ha : a ≠ 0) (hb : b ≠ 0) :
  a + 2 • b = 0 → ‖a - b‖ = ‖a‖ + ‖b‖ := by
  sorry

end NUMINAMATH_CALUDE_sufficient_condition_for_vector_equality_l711_71102


namespace NUMINAMATH_CALUDE_intersection_slope_range_l711_71122

/-- Given two points A and B, and a line l that intersects the line segment AB,
    prove that the slope k of line l is within a specific range. -/
theorem intersection_slope_range (A B : ℝ × ℝ) (k : ℝ) : 
  A = (1, 3) →
  B = (-2, -1) →
  (∃ x y : ℝ, x ∈ Set.Icc (min A.1 B.1) (max A.1 B.1) ∧ 
              y ∈ Set.Icc (min A.2 B.2) (max A.2 B.2) ∧
              y = k * (x - 2) + 1) →
  -2 ≤ k ∧ k ≤ 1/2 :=
by sorry

end NUMINAMATH_CALUDE_intersection_slope_range_l711_71122


namespace NUMINAMATH_CALUDE_simplify_expression_l711_71127

theorem simplify_expression (x : ℝ) : 5 * x + 2 * (4 + x) = 7 * x + 8 := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l711_71127


namespace NUMINAMATH_CALUDE_cake_box_theorem_l711_71166

/-- Represents the dimensions of a rectangular box -/
structure BoxDimensions where
  length : ℕ
  width : ℕ
  height : ℕ

/-- Calculates the volume of a box given its dimensions -/
def boxVolume (d : BoxDimensions) : ℕ :=
  d.length * d.width * d.height

/-- Represents the problem of fitting cake boxes into a carton -/
structure CakeBoxProblem where
  carton : BoxDimensions
  cakeBox : BoxDimensions

/-- Calculates the maximum number of cake boxes that can fit in a carton -/
def maxCakeBoxes (p : CakeBoxProblem) : ℕ :=
  (boxVolume p.carton) / (boxVolume p.cakeBox)

/-- The main theorem stating the maximum number of cake boxes that can fit in the given carton -/
theorem cake_box_theorem (p : CakeBoxProblem) 
  (h_carton : p.carton = ⟨25, 42, 60⟩) 
  (h_cake_box : p.cakeBox = ⟨8, 7, 5⟩) : 
  maxCakeBoxes p = 225 := by
  sorry

#eval maxCakeBoxes ⟨⟨25, 42, 60⟩, ⟨8, 7, 5⟩⟩

end NUMINAMATH_CALUDE_cake_box_theorem_l711_71166


namespace NUMINAMATH_CALUDE_shaded_area_circles_l711_71116

/-- The area of the shaded region formed by a larger circle and two smaller circles --/
theorem shaded_area_circles (R : ℝ) (h : R = 8) : 
  let r := R / 2
  let large_circle_area := π * R^2
  let small_circle_area := π * r^2
  large_circle_area - 2 * small_circle_area = 32 * π :=
by sorry

end NUMINAMATH_CALUDE_shaded_area_circles_l711_71116


namespace NUMINAMATH_CALUDE_inequality1_solution_inequality2_solution_inequality3_solution_inequality4_no_solution_l711_71105

-- Define the inequalities
def inequality1 (x : ℝ) : Prop := x^2 + 5*x + 6 < 0
def inequality2 (x : ℝ) : Prop := -x^2 + 9*x - 20 < 0
def inequality3 (x : ℝ) : Prop := x^2 + x - 56 < 0
def inequality4 (x : ℝ) : Prop := 9*x^2 + 4 < 12*x

-- State the theorems
theorem inequality1_solution : 
  ∀ x : ℝ, inequality1 x ↔ -3 < x ∧ x < -2 := by sorry

theorem inequality2_solution : 
  ∀ x : ℝ, inequality2 x ↔ x < 4 ∨ x > 5 := by sorry

theorem inequality3_solution : 
  ∀ x : ℝ, inequality3 x ↔ -8 < x ∧ x < 7 := by sorry

theorem inequality4_no_solution : 
  ¬∃ x : ℝ, inequality4 x := by sorry

end NUMINAMATH_CALUDE_inequality1_solution_inequality2_solution_inequality3_solution_inequality4_no_solution_l711_71105


namespace NUMINAMATH_CALUDE_problem_solving_probability_l711_71139

theorem problem_solving_probability (p1 p2 p3 : ℝ) 
  (h1 : p1 = 1/5) (h2 : p2 = 1/3) (h3 : p3 = 1/4) :
  1 - (1 - p1) * (1 - p2) * (1 - p3) = 3/5 := by
sorry

end NUMINAMATH_CALUDE_problem_solving_probability_l711_71139


namespace NUMINAMATH_CALUDE_xiaoming_red_pens_l711_71114

/-- The number of red pens bought by Xiaoming -/
def red_pens : ℕ := 36

/-- The total number of pens bought -/
def total_pens : ℕ := 66

/-- The original price of a red pen in yuan -/
def red_pen_price : ℚ := 5

/-- The original price of a black pen in yuan -/
def black_pen_price : ℚ := 9

/-- The discount rate for red pens -/
def red_discount : ℚ := 85 / 100

/-- The discount rate for black pens -/
def black_discount : ℚ := 80 / 100

/-- The discount rate on the total price -/
def total_discount : ℚ := 18 / 100

theorem xiaoming_red_pens :
  red_pens = 36 ∧
  red_pens ≤ total_pens ∧
  (red_pen_price * red_pens + black_pen_price * (total_pens - red_pens)) * (1 - total_discount) =
  red_pen_price * red_discount * red_pens + black_pen_price * black_discount * (total_pens - red_pens) :=
by sorry

end NUMINAMATH_CALUDE_xiaoming_red_pens_l711_71114


namespace NUMINAMATH_CALUDE_min_value_expression_l711_71164

theorem min_value_expression (a b : ℝ) (ha : a > 0) (hb : b > 0) (h : 2*a + b = 1) :
  1/a + 1/(2*b) ≥ 9/2 ∧ (1/a + 1/(2*b) = 9/2 ↔ a = 1/3 ∧ b = 1/3) :=
sorry

end NUMINAMATH_CALUDE_min_value_expression_l711_71164


namespace NUMINAMATH_CALUDE_rectangle_least_area_l711_71178

theorem rectangle_least_area :
  ∀ l w : ℕ,
  l = 3 * w →
  2 * (l + w) = 120 →
  ∀ l' w' : ℕ,
  l' = 3 * w' →
  2 * (l' + w') = 120 →
  l * w ≤ l' * w' →
  l * w = 675 :=
by sorry

end NUMINAMATH_CALUDE_rectangle_least_area_l711_71178


namespace NUMINAMATH_CALUDE_betsy_games_won_l711_71121

theorem betsy_games_won (betsy helen susan : ℕ) 
  (helen_games : helen = 2 * betsy)
  (susan_games : susan = 3 * betsy)
  (total_games : betsy + helen + susan = 30) :
  betsy = 5 := by
sorry

end NUMINAMATH_CALUDE_betsy_games_won_l711_71121


namespace NUMINAMATH_CALUDE_mowing_time_calculation_l711_71177

/-- Represents the dimensions of a rectangular section of the lawn -/
structure LawnSection where
  length : ℝ
  width : ℝ

/-- Represents the mower specifications -/
structure Mower where
  swath_width : ℝ
  overlap : ℝ

/-- Calculates the time required to mow an L-shaped lawn -/
def mowing_time (section1 : LawnSection) (section2 : LawnSection) (mower : Mower) (walking_rate : ℝ) : ℝ :=
  sorry

/-- Theorem stating the time required to mow the lawn -/
theorem mowing_time_calculation :
  let section1 : LawnSection := { length := 120, width := 50 }
  let section2 : LawnSection := { length := 70, width := 50 }
  let mower : Mower := { swath_width := 35 / 12, overlap := 5 / 12 }
  let walking_rate : ℝ := 4000
  mowing_time section1 section2 mower walking_rate = 0.95 :=
by sorry

end NUMINAMATH_CALUDE_mowing_time_calculation_l711_71177


namespace NUMINAMATH_CALUDE_lemonade_solution_water_content_l711_71181

theorem lemonade_solution_water_content 
  (L : ℝ) -- Amount of lemonade syrup
  (W : ℝ) -- Amount of water
  (removed : ℝ) -- Amount of solution removed and replaced with water
  (h1 : L = 7) -- 7 parts of lemonade syrup
  (h2 : removed = 2.1428571428571423) -- Amount removed and replaced
  (h3 : L / (L + W - removed + removed) = 0.4) -- 40% concentration after replacement
  : W = 10.5 := by
  sorry

end NUMINAMATH_CALUDE_lemonade_solution_water_content_l711_71181


namespace NUMINAMATH_CALUDE_hyperbola_equation_theorem_l711_71138

/-- A hyperbola with given asymptotes and a point it passes through -/
structure Hyperbola where
  /-- The slope of the asymptotes -/
  asymptote_slope : ℝ
  /-- A point that the hyperbola passes through -/
  point : ℝ × ℝ

/-- The equation of a hyperbola given its asymptotes and a point it passes through -/
def hyperbola_equation (h : Hyperbola) : ℝ → ℝ → Prop :=
  fun x y => x^2 - (y^2 / (4 * h.asymptote_slope^2)) = 1

/-- Theorem stating that a hyperbola with asymptotes y = ±2x passing through (√2, 2) has the equation x² - y²/4 = 1 -/
theorem hyperbola_equation_theorem (h : Hyperbola) 
    (h_asymptote : h.asymptote_slope = 2)
    (h_point : h.point = (Real.sqrt 2, 2)) :
    hyperbola_equation h = fun x y => x^2 - y^2/4 = 1 := by
  sorry

#check hyperbola_equation_theorem

end NUMINAMATH_CALUDE_hyperbola_equation_theorem_l711_71138


namespace NUMINAMATH_CALUDE_factor_expression_l711_71112

theorem factor_expression (x : ℝ) : 25 * x^2 + 10 * x = 5 * x * (5 * x + 2) := by
  sorry

end NUMINAMATH_CALUDE_factor_expression_l711_71112


namespace NUMINAMATH_CALUDE_joyce_initial_apples_l711_71154

/-- The number of apples Joyce gave to Larry -/
def apples_given_to_Larry : ℕ := 52

/-- The number of apples Joyce had left -/
def apples_left_with_Joyce : ℕ := 23

/-- The initial number of apples Joyce had -/
def initial_apples : ℕ := apples_given_to_Larry + apples_left_with_Joyce

theorem joyce_initial_apples :
  initial_apples = 75 := by sorry

end NUMINAMATH_CALUDE_joyce_initial_apples_l711_71154


namespace NUMINAMATH_CALUDE_intersection_implies_a_value_l711_71199

def A (a : ℝ) : Set ℝ := {-4, 2*a-1, a^2}
def B (a : ℝ) : Set ℝ := {a-5, 1-a, 9}

theorem intersection_implies_a_value :
  ∀ a : ℝ, (A a) ∩ (B a) = {9} → a = -3 := by sorry

end NUMINAMATH_CALUDE_intersection_implies_a_value_l711_71199


namespace NUMINAMATH_CALUDE_third_number_proof_l711_71143

/-- The largest five-digit number with all even digits -/
def largest_even_five_digit : ℕ := 88888

/-- The smallest four-digit number with all odd digits -/
def smallest_odd_four_digit : ℕ := 1111

/-- The sum of the three numbers -/
def total_sum : ℕ := 121526

/-- The third number -/
def third_number : ℕ := total_sum - largest_even_five_digit - smallest_odd_four_digit

theorem third_number_proof :
  third_number = 31527 :=
by sorry

end NUMINAMATH_CALUDE_third_number_proof_l711_71143


namespace NUMINAMATH_CALUDE_range_of_negative_power_function_l711_71103

/-- Given a function f(x) = x^k where k < 0, its range on [1, ∞) is (0, 1] -/
theorem range_of_negative_power_function (k : ℝ) (hk : k < 0) :
  let f : ℝ → ℝ := fun x ↦ x^k
  Set.range (f ∘ (fun x ↦ x + 1)) = Set.Ioc 0 1 := by
  sorry

end NUMINAMATH_CALUDE_range_of_negative_power_function_l711_71103


namespace NUMINAMATH_CALUDE_functions_intersect_at_negative_six_l711_71149

-- Define the two functions
def f (x : ℝ) : ℝ := 2 * x + 1
def g (x : ℝ) : ℝ := x - 5

-- State the theorem
theorem functions_intersect_at_negative_six : f (-6) = g (-6) := by
  sorry

end NUMINAMATH_CALUDE_functions_intersect_at_negative_six_l711_71149


namespace NUMINAMATH_CALUDE_quadratic_inequality_and_range_l711_71117

theorem quadratic_inequality_and_range (a b : ℝ) (k : ℝ) : 
  (∀ x, (x < 1 ∨ x > b) ↔ a * x^2 - 3 * x + 2 > 0) →
  (∀ x y, x > 0 → y > 0 → a / x + b / y = 1 → 2 * x + y ≥ k^2 + k + 2) →
  a = 1 ∧ b = 2 ∧ -3 ≤ k ∧ k ≤ 2 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_inequality_and_range_l711_71117


namespace NUMINAMATH_CALUDE_hyperbola_asymptote_implies_a_eq_2_l711_71156

-- Define the hyperbola equation
def hyperbola_eq (x y a : ℝ) : Prop := x^2 / a^2 - y^2 / 9 = 1

-- Define the asymptote equation
def asymptote_eq (x y : ℝ) : Prop := 3 * x + 2 * y = 0

-- Theorem statement
theorem hyperbola_asymptote_implies_a_eq_2 :
  ∀ a : ℝ, a > 0 →
  (∀ x y : ℝ, hyperbola_eq x y a ↔ asymptote_eq x y) →
  a = 2 :=
sorry

end NUMINAMATH_CALUDE_hyperbola_asymptote_implies_a_eq_2_l711_71156


namespace NUMINAMATH_CALUDE_f_equals_neg_tan_f_at_eight_pi_thirds_l711_71180

/-- The function f as defined in the problem -/
noncomputable def f (x : ℝ) : ℝ :=
  (Real.sin (Real.pi + x) * Real.cos (Real.pi - x) * Real.sin (2 * Real.pi - x)) /
  (Real.sin (Real.pi / 2 + x) * Real.cos (x - Real.pi / 2) * Real.cos (-x))

/-- Theorem stating that f(x) = -tan(x) for all x -/
theorem f_equals_neg_tan (x : ℝ) : f x = -Real.tan x := by sorry

/-- Theorem stating that f(8π/3) = -√3 -/
theorem f_at_eight_pi_thirds : f (8 * Real.pi / 3) = -Real.sqrt 3 := by sorry

end NUMINAMATH_CALUDE_f_equals_neg_tan_f_at_eight_pi_thirds_l711_71180


namespace NUMINAMATH_CALUDE_celsius_to_fahrenheit_55_l711_71142

/-- Converts Celsius to Fahrenheit -/
def celsius_to_fahrenheit (c : ℚ) : ℚ := (c * 9 / 5) + 32

/-- Water boiling point in Fahrenheit -/
def water_boiling_f : ℚ := 212

/-- Water boiling point in Celsius -/
def water_boiling_c : ℚ := 100

/-- Ice melting point in Fahrenheit -/
def ice_melting_f : ℚ := 32

/-- Ice melting point in Celsius -/
def ice_melting_c : ℚ := 0

/-- The temperature of the pot of water in Celsius -/
def pot_temp_c : ℚ := 55

/-- The temperature of the pot of water in Fahrenheit -/
def pot_temp_f : ℚ := 131

theorem celsius_to_fahrenheit_55 :
  celsius_to_fahrenheit pot_temp_c = pot_temp_f := by sorry

end NUMINAMATH_CALUDE_celsius_to_fahrenheit_55_l711_71142


namespace NUMINAMATH_CALUDE_line_properties_l711_71160

/-- Two lines in the plane, parameterized by a -/
def Line1 (a : ℝ) : ℝ → ℝ → Prop :=
  λ x y => a * x - y + 1 = 0

def Line2 (a : ℝ) : ℝ → ℝ → Prop :=
  λ x y => x + a * y + 1 = 0

/-- The theorem stating the properties of the two lines -/
theorem line_properties :
  ∀ a : ℝ,
    (∀ x y : ℝ, Line1 a x y → Line2 a x y → (a * 1 - 1 * a = 0)) ∧ 
    (Line1 a 0 1) ∧
    (Line2 a (-1) 0) ∧
    (∀ x y : ℝ, x ≠ 0 → y ≠ 0 → Line1 a x y → Line2 a x y → x^2 + x + y^2 - y = 0) :=
by sorry

end NUMINAMATH_CALUDE_line_properties_l711_71160


namespace NUMINAMATH_CALUDE_triangle_bd_length_l711_71187

/-- Given a triangle ABC with AC = BC = 10 and AB = 5, and a point D on line AB such that B is between A and D, and CD = 13, prove that BD ≈ 6.17 -/
theorem triangle_bd_length (A B C D : ℝ × ℝ) : 
  let dist := λ p q : ℝ × ℝ => Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2)
  dist A C = 10 →
  dist B C = 10 →
  dist A B = 5 →
  D.1 = A.1 + t * (B.1 - A.1) ∧ D.2 = A.2 + t * (B.2 - A.2) →
  (1 < t) →
  dist C D = 13 →
  ∃ ε > 0, |dist B D - 6.17| < ε :=
by sorry

end NUMINAMATH_CALUDE_triangle_bd_length_l711_71187


namespace NUMINAMATH_CALUDE_negative_a_sign_l711_71123

theorem negative_a_sign (a : ℝ) : ∃ (x y : ℝ), x < 0 ∧ y > 0 ∧ (-a = x ∨ -a = y) :=
  sorry

end NUMINAMATH_CALUDE_negative_a_sign_l711_71123


namespace NUMINAMATH_CALUDE_susan_missed_pay_l711_71100

/-- Calculates the missed pay for Susan's vacation --/
def missed_pay (weeks : ℕ) (work_days_per_week : ℕ) (paid_vacation_days : ℕ) 
                (hourly_rate : ℚ) (hours_per_day : ℕ) : ℚ :=
  let total_work_days := weeks * work_days_per_week
  let unpaid_days := total_work_days - paid_vacation_days
  let daily_pay := hourly_rate * hours_per_day
  unpaid_days * daily_pay

/-- Proves that Susan will miss $480 on her vacation --/
theorem susan_missed_pay : 
  missed_pay 2 5 6 15 8 = 480 := by
  sorry

end NUMINAMATH_CALUDE_susan_missed_pay_l711_71100


namespace NUMINAMATH_CALUDE_product_modulo_25_l711_71146

theorem product_modulo_25 :
  ∃ m : ℕ, 0 ≤ m ∧ m < 25 ∧ (68 * 95 * 113) % 25 = m ∧ m = 5 := by
  sorry

end NUMINAMATH_CALUDE_product_modulo_25_l711_71146


namespace NUMINAMATH_CALUDE_expression_evaluation_l711_71111

theorem expression_evaluation :
  let f (x : ℚ) := (2*x - 3) / (x + 2)
  let g (x : ℚ) := (2*(f x) - 3) / (f x + 2)
  g 2 = -10/9 := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l711_71111


namespace NUMINAMATH_CALUDE_append_self_perfect_square_l711_71130

theorem append_self_perfect_square :
  ∃ (A : ℕ) (n : ℕ), 
    (10^n ≤ A) ∧ (A < 10^(n+1)) ∧ 
    ∃ (k : ℕ), ((10^n + 1) * A = k^2) := by
  sorry

end NUMINAMATH_CALUDE_append_self_perfect_square_l711_71130


namespace NUMINAMATH_CALUDE_greatest_of_three_consecutive_integers_l711_71104

theorem greatest_of_three_consecutive_integers (x : ℤ) 
  (h : x + (x + 1) + (x + 2) = 18) : 
  max x (max (x + 1) (x + 2)) = 7 := by
  sorry

end NUMINAMATH_CALUDE_greatest_of_three_consecutive_integers_l711_71104


namespace NUMINAMATH_CALUDE_circle_tangency_l711_71158

/-- Two circles are externally tangent if the distance between their centers
    is equal to the sum of their radii -/
def externally_tangent (c1 c2 : ℝ × ℝ) (r1 r2 : ℝ) : Prop :=
  (c1.1 - c2.1)^2 + (c1.2 - c2.2)^2 = (r1 + r2)^2

theorem circle_tangency (t : ℝ) : 
  externally_tangent (0, 0) (t, 0) 2 1 → t = 3 ∨ t = -3 := by
  sorry

#check circle_tangency

end NUMINAMATH_CALUDE_circle_tangency_l711_71158


namespace NUMINAMATH_CALUDE_monotone_increasing_interval_l711_71135

/-- The function f(x) = (3 - x^2) * e^x is monotonically increasing on the interval (-3, 1) -/
theorem monotone_increasing_interval (x : ℝ) :
  StrictMonoOn (fun x => (3 - x^2) * Real.exp x) (Set.Ioo (-3) 1) := by
  sorry

end NUMINAMATH_CALUDE_monotone_increasing_interval_l711_71135


namespace NUMINAMATH_CALUDE_leap_stride_difference_proof_l711_71101

/-- The number of strides Elmer takes between consecutive poles -/
def elmer_strides : ℕ := 44

/-- The number of leaps Oscar takes between consecutive poles -/
def oscar_leaps : ℕ := 12

/-- The number of poles -/
def num_poles : ℕ := 41

/-- The total distance in feet -/
def total_distance : ℕ := 5280

/-- The difference between Oscar's leap length and Elmer's stride length in feet -/
def leap_stride_difference : ℚ := 8

theorem leap_stride_difference_proof :
  let total_gaps := num_poles - 1
  let elmer_total_strides := elmer_strides * total_gaps
  let oscar_total_leaps := oscar_leaps * total_gaps
  let elmer_stride_length := total_distance / elmer_total_strides
  let oscar_leap_length := total_distance / oscar_total_leaps
  oscar_leap_length - elmer_stride_length = leap_stride_difference := by
  sorry

end NUMINAMATH_CALUDE_leap_stride_difference_proof_l711_71101


namespace NUMINAMATH_CALUDE_unique_perfect_square_polynomial_l711_71173

theorem unique_perfect_square_polynomial : 
  ∃! x : ℤ, ∃ y : ℤ, x^4 + 8*x^3 + 18*x^2 + 8*x + 36 = y^2 :=
by sorry

end NUMINAMATH_CALUDE_unique_perfect_square_polynomial_l711_71173


namespace NUMINAMATH_CALUDE_sum_of_odd_numbers_l711_71192

theorem sum_of_odd_numbers (N : ℕ) : 
  1001 + 1003 + 1005 + 1007 + 1009 = 5100 - N → N = 75 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_odd_numbers_l711_71192


namespace NUMINAMATH_CALUDE_division_equation_solution_l711_71172

theorem division_equation_solution :
  ∃ x : ℝ, (0.009 / x = 0.1) ∧ (x = 0.09) := by
  sorry

end NUMINAMATH_CALUDE_division_equation_solution_l711_71172


namespace NUMINAMATH_CALUDE_smallest_y_for_perfect_cube_l711_71140

def x : ℕ := 7 * 24 * 48

def is_perfect_cube (n : ℕ) : Prop := ∃ m : ℕ, n = m^3

theorem smallest_y_for_perfect_cube :
  ∃! y : ℕ, y > 0 ∧ is_perfect_cube (x * y) ∧ ∀ z : ℕ, z > 0 ∧ z < y → ¬is_perfect_cube (x * z) :=
by sorry

end NUMINAMATH_CALUDE_smallest_y_for_perfect_cube_l711_71140


namespace NUMINAMATH_CALUDE_exradii_sum_equals_p_squared_l711_71169

/-- Given a triangle with sides a, b, c, exradii ra, rb, rc, and semi-perimeter p,
    if the products of exradii satisfy certain conditions, then the sum of these
    products equals p^2. -/
theorem exradii_sum_equals_p_squared
  (a b c ra rb rc p : ℝ)
  (h_pos_a : 0 < a) (h_pos_b : 0 < b) (h_pos_c : 0 < c)
  (h_pos_ra : 0 < ra) (h_pos_rb : 0 < rb) (h_pos_rc : 0 < rc)
  (h_semi_perimeter : p = (a + b + c) / 2)
  (h_ra_rb : ra * rb = p * (p - c))
  (h_rb_rc : rb * rc = p * (p - a))
  (h_rc_ra : rc * ra = p * (p - b)) :
  ra * rb + rb * rc + rc * ra = p^2 := by
  sorry

end NUMINAMATH_CALUDE_exradii_sum_equals_p_squared_l711_71169


namespace NUMINAMATH_CALUDE_fraction_sum_equals_two_l711_71186

theorem fraction_sum_equals_two (p q : ℚ) (h : p / q = 4 / 5) :
  11 / 7 + (2 * q - p) / (2 * q + p) = 2 := by
  sorry

end NUMINAMATH_CALUDE_fraction_sum_equals_two_l711_71186


namespace NUMINAMATH_CALUDE_geometric_sequence_sum_constant_l711_71108

/-- Given a geometric sequence {aₙ} where the sum of the first n terms
    is Sₙ = 3ⁿ + r, prove that r = -1 -/
theorem geometric_sequence_sum_constant (a : ℕ → ℝ) (S : ℕ → ℝ) (r : ℝ) :
  (∀ n : ℕ, S n = 3^n + r) →
  (∀ n : ℕ, n ≥ 2 → a n = S n - S (n-1)) →
  (a 1 = S 1) →
  (∀ n : ℕ, n ≥ 2 → a (n+1) = 3 * a n) →
  r = -1 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_sum_constant_l711_71108


namespace NUMINAMATH_CALUDE_sphere_radius_ratio_l711_71161

theorem sphere_radius_ratio (V_large : ℝ) (V_small : ℝ) :
  V_large = 288 * Real.pi →
  V_small = 0.125 * V_large →
  ∃ (r_large r_small : ℝ),
    V_large = (4 / 3) * Real.pi * r_large^3 ∧
    V_small = (4 / 3) * Real.pi * r_small^3 ∧
    r_small / r_large = 1 / 2 :=
by sorry

end NUMINAMATH_CALUDE_sphere_radius_ratio_l711_71161


namespace NUMINAMATH_CALUDE_power_of_power_at_three_l711_71150

theorem power_of_power_at_three : (3^3)^(3^3) = 27^27 := by
  sorry

end NUMINAMATH_CALUDE_power_of_power_at_three_l711_71150


namespace NUMINAMATH_CALUDE_arccos_lt_arcsin_iff_l711_71189

theorem arccos_lt_arcsin_iff (x : ℝ) : 
  Real.arccos x < Real.arcsin x ↔ x ∈ Set.Ioo (1 / Real.sqrt 2) 1 :=
by
  sorry

end NUMINAMATH_CALUDE_arccos_lt_arcsin_iff_l711_71189


namespace NUMINAMATH_CALUDE_sum_two_longest_altitudes_eq_14_l711_71107

/-- A triangle with sides 6, 8, and 10 -/
structure Triangle :=
  (side1 : ℝ)
  (side2 : ℝ)
  (side3 : ℝ)
  (side1_eq : side1 = 6)
  (side2_eq : side2 = 8)
  (side3_eq : side3 = 10)

/-- The length of an altitude in a triangle -/
def altitude_length (t : Triangle) : ℝ → ℝ :=
  sorry

/-- The sum of the two longest altitudes in the triangle -/
def sum_two_longest_altitudes (t : Triangle) : ℝ :=
  sorry

/-- Theorem: The sum of the two longest altitudes in a triangle with sides 6, 8, and 10 is 14 -/
theorem sum_two_longest_altitudes_eq_14 (t : Triangle) :
  sum_two_longest_altitudes t = 14 := by
  sorry

end NUMINAMATH_CALUDE_sum_two_longest_altitudes_eq_14_l711_71107


namespace NUMINAMATH_CALUDE_angle_sum_result_l711_71195

theorem angle_sum_result (a b : Real) (h1 : 0 < a ∧ a < π/2) (h2 : 0 < b ∧ b < π/2)
  (h3 : 5 * (Real.cos a)^2 + 3 * (Real.cos b)^2 = 2)
  (h4 : 5 * Real.sin (2*a) + 3 * Real.sin (2*b) = 0) :
  2*a + b = π/2 := by
sorry

end NUMINAMATH_CALUDE_angle_sum_result_l711_71195


namespace NUMINAMATH_CALUDE_max_value_of_f_l711_71157

-- Define the function f
def f (x : ℝ) : ℝ := 3 * x^3 - 18 * x^2 + 27 * x

-- State the theorem
theorem max_value_of_f :
  ∃ (M : ℝ), M = 12 ∧ ∀ (x : ℝ), x ≥ 0 → f x ≤ M :=
sorry

end NUMINAMATH_CALUDE_max_value_of_f_l711_71157


namespace NUMINAMATH_CALUDE_pascal_triangle_12th_row_4th_number_l711_71163

theorem pascal_triangle_12th_row_4th_number : Nat.choose 12 3 = 220 := by
  sorry

end NUMINAMATH_CALUDE_pascal_triangle_12th_row_4th_number_l711_71163


namespace NUMINAMATH_CALUDE_product_increase_by_2016_l711_71191

theorem product_increase_by_2016 : ∃ (a b c : ℕ), 
  (a - 3) * (b - 3) * (c - 3) = a * b * c + 2016 := by
sorry

end NUMINAMATH_CALUDE_product_increase_by_2016_l711_71191


namespace NUMINAMATH_CALUDE_min_x_prime_factorization_l711_71182

theorem min_x_prime_factorization (x y : ℕ+) (h : 13 * x^7 = 17 * y^11) :
  ∃ (a b c d : ℕ),
    x = a^c * b^d ∧
    x ≥ 17^15 * 13^2 ∧
    (∀ x' : ℕ+, 13 * x'^7 = 17 * y^11 → x' ≥ x) ∧
    a + b + c + d = 47 := by
  sorry

end NUMINAMATH_CALUDE_min_x_prime_factorization_l711_71182


namespace NUMINAMATH_CALUDE_acute_triangle_sine_cosine_inequality_l711_71198

theorem acute_triangle_sine_cosine_inequality (α β γ : Real) 
  (h_acute : α > 0 ∧ β > 0 ∧ γ > 0 ∧ α + β + γ = π) : 
  Real.sin α * Real.sin β * Real.sin γ > 5 * Real.cos α * Real.cos β * Real.cos γ := by
  sorry

end NUMINAMATH_CALUDE_acute_triangle_sine_cosine_inequality_l711_71198


namespace NUMINAMATH_CALUDE_pascal_ratio_row_34_l711_71106

/-- Pascal's Triangle entry -/
def pascal (n : ℕ) (k : ℕ) : ℕ := Nat.choose n k

/-- Check if three consecutive entries in a row are in the ratio 2:3:4 -/
def hasRatio234 (n : ℕ) (r : ℕ) : Prop :=
  4 * pascal n r = 3 * pascal n (r+1) ∧
  4 * pascal n (r+1) = 3 * pascal n (r+2)

theorem pascal_ratio_row_34 : ∃ r, hasRatio234 34 r := by
  sorry

#check pascal_ratio_row_34

end NUMINAMATH_CALUDE_pascal_ratio_row_34_l711_71106


namespace NUMINAMATH_CALUDE_sufficient_not_necessary_condition_l711_71168

theorem sufficient_not_necessary_condition (a : ℝ) : 
  (a = 2 → (a - 1) * (a - 2) = 0) ∧ 
  ¬(∀ a : ℝ, (a - 1) * (a - 2) = 0 → a = 2) :=
by sorry

end NUMINAMATH_CALUDE_sufficient_not_necessary_condition_l711_71168


namespace NUMINAMATH_CALUDE_quadratic_minimum_value_l711_71179

theorem quadratic_minimum_value : 
  ∀ x : ℝ, 3 * x^2 - 18 * x + 12 ≥ -15 ∧ 
  ∃ x : ℝ, 3 * x^2 - 18 * x + 12 = -15 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_minimum_value_l711_71179


namespace NUMINAMATH_CALUDE_marble_distribution_l711_71171

/-- The minimum number of additional marbles needed and the sum of marbles for specific friends -/
theorem marble_distribution (n : Nat) (initial_marbles : Nat) 
  (h1 : n = 12) (h2 : initial_marbles = 34) : 
  let additional_marbles := (n * (n + 1)) / 2 - initial_marbles
  let third_friend := 3
  let seventh_friend := 7
  let eleventh_friend := 11
  (additional_marbles = 44) ∧ 
  (third_friend + seventh_friend + eleventh_friend = 21) := by
  sorry

end NUMINAMATH_CALUDE_marble_distribution_l711_71171


namespace NUMINAMATH_CALUDE_average_age_proof_l711_71137

/-- Given three people a, b, and c, this theorem proves that if their average age is 28 years
    and the age of b is 26 years, then the average age of a and c is 29 years. -/
theorem average_age_proof (a b c : ℕ) : 
  (a + b + c) / 3 = 28 → b = 26 → (a + c) / 2 = 29 := by
  sorry

end NUMINAMATH_CALUDE_average_age_proof_l711_71137


namespace NUMINAMATH_CALUDE_arithmetic_sequence_fifth_term_l711_71176

/-- Given an arithmetic sequence {a_n} with a_1 = 2 and common difference d = 3,
    prove that the fifth term a_5 equals 14. -/
theorem arithmetic_sequence_fifth_term (a : ℕ → ℝ) :
  (∀ n, a (n + 1) = a n + 3) →  -- Common difference is 3
  a 1 = 2 →                    -- First term is 2
  a 5 = 14 :=                  -- Fifth term is 14
by sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_fifth_term_l711_71176


namespace NUMINAMATH_CALUDE_expected_distinct_colors_value_l711_71136

/-- The number of balls in the bag -/
def n : ℕ := 10

/-- The number of times a ball is picked -/
def k : ℕ := 4

/-- The probability of not picking a specific color in one draw -/
def p : ℚ := 9/10

/-- The expected number of distinct colors -/
def expected_distinct_colors : ℚ := n * (1 - p^k)

theorem expected_distinct_colors_value :
  expected_distinct_colors = 3439/1000 := by sorry

end NUMINAMATH_CALUDE_expected_distinct_colors_value_l711_71136


namespace NUMINAMATH_CALUDE_tangent_line_slope_range_l711_71145

open Real Set

noncomputable def f (x : ℝ) : ℝ := Real.sqrt 3 * Real.sin (2 * x) - Real.cos (2 * x)

theorem tangent_line_slope_range :
  let symmetry_axis : ℝ := π / 3
  let tangent_line (m c : ℝ) (x y : ℝ) : Prop := x + m * y + c = 0
  ∃ (c : ℝ), ∃ (x : ℝ), tangent_line m c x (f x) ↔ 
    m ∈ Iic (-1/4) ∪ Ici (1/4) :=
sorry

end NUMINAMATH_CALUDE_tangent_line_slope_range_l711_71145


namespace NUMINAMATH_CALUDE_simplify_fraction_l711_71119

theorem simplify_fraction (x : ℝ) (h : x = 2) : 15 * x^5 / (45 * x^3) = 4/3 := by
  sorry

end NUMINAMATH_CALUDE_simplify_fraction_l711_71119


namespace NUMINAMATH_CALUDE_quadratic_function_properties_l711_71185

/-- A quadratic function with given vertex and y-intercept -/
def quadratic_function (x : ℝ) : ℝ := 2 * (x - 2)^2 - 4

theorem quadratic_function_properties :
  (∀ x, quadratic_function x = 2 * (x - 2)^2 - 4) ∧
  (quadratic_function 2 = -4) ∧
  (quadratic_function 0 = 4) ∧
  (quadratic_function 3 ≠ 5) :=
sorry

#check quadratic_function_properties

end NUMINAMATH_CALUDE_quadratic_function_properties_l711_71185


namespace NUMINAMATH_CALUDE_z_in_second_quadrant_l711_71194

def complex_i : ℂ := Complex.I

def z : ℂ := complex_i + complex_i^2

def second_quadrant (c : ℂ) : Prop :=
  c.re < 0 ∧ c.im > 0

theorem z_in_second_quadrant : second_quadrant z := by
  sorry

end NUMINAMATH_CALUDE_z_in_second_quadrant_l711_71194


namespace NUMINAMATH_CALUDE_older_brother_stamps_l711_71162

theorem older_brother_stamps : 
  ∀ (younger older : ℕ), 
  younger + older = 25 → 
  older = 2 * younger + 1 → 
  older = 17 := by sorry

end NUMINAMATH_CALUDE_older_brother_stamps_l711_71162


namespace NUMINAMATH_CALUDE_problem_solution_l711_71196

theorem problem_solution (x : ℝ) 
  (h : x + Real.sqrt (x^2 - 4) + 1 / (x - Real.sqrt (x^2 - 4)) = 10) : 
  x^2 + Real.sqrt (x^4 - 4) + 1 / (x^2 + Real.sqrt (x^4 - 4)) = 841/100 := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l711_71196


namespace NUMINAMATH_CALUDE_rhombus_area_l711_71151

/-- The area of a rhombus with side length 4 cm and an interior angle of 30 degrees is 8 cm² -/
theorem rhombus_area (s : ℝ) (θ : ℝ) (h1 : s = 4) (h2 : θ = π / 6) :
  s * s * Real.sin θ = 8 :=
sorry

end NUMINAMATH_CALUDE_rhombus_area_l711_71151


namespace NUMINAMATH_CALUDE_negative_three_squared_l711_71128

theorem negative_three_squared : (-3 : ℤ)^2 = 9 := by
  sorry

end NUMINAMATH_CALUDE_negative_three_squared_l711_71128


namespace NUMINAMATH_CALUDE_parallel_lines_a_value_l711_71131

/-- Two lines are parallel if their slopes are equal but they are not the same line -/
def parallel (a b c d e f : ℝ) : Prop :=
  a / b = d / e ∧ a / b ≠ c / f

theorem parallel_lines_a_value (a : ℝ) :
  parallel a 2 0 3 (a + 1) 1 → a = -3 ∨ a = 2 := by
  sorry

#check parallel_lines_a_value

end NUMINAMATH_CALUDE_parallel_lines_a_value_l711_71131


namespace NUMINAMATH_CALUDE_absolute_sum_vs_square_sum_l711_71109

theorem absolute_sum_vs_square_sum :
  (∀ x y : ℝ, (abs x + abs y ≤ 1) → (x^2 + y^2 ≤ 1)) ∧
  (∃ x y : ℝ, (x^2 + y^2 ≤ 1) ∧ (abs x + abs y > 1)) := by
  sorry

end NUMINAMATH_CALUDE_absolute_sum_vs_square_sum_l711_71109


namespace NUMINAMATH_CALUDE_parallel_vectors_magnitude_l711_71183

/-- Given vectors a and b in ℝ², if a is parallel to b, then the magnitude of b is √5. -/
theorem parallel_vectors_magnitude (a b : ℝ × ℝ) :
  a = (1, 2) →
  b.1 = -1 →
  ∃ (k : ℝ), k ≠ 0 ∧ a = k • b →
  ‖b‖ = Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_parallel_vectors_magnitude_l711_71183


namespace NUMINAMATH_CALUDE_quadratic_roots_l711_71129

theorem quadratic_roots : ∃ (x₁ x₂ : ℝ), 
  (x₁ = -1 ∧ x₂ = 2) ∧ 
  (∀ x : ℝ, x * (x - 2) = 2 - x ↔ x = x₁ ∨ x = x₂) := by
sorry

end NUMINAMATH_CALUDE_quadratic_roots_l711_71129


namespace NUMINAMATH_CALUDE_stratified_sampling_appropriate_for_subgroups_investigation1_uses_stratified_sampling_investigation2_uses_simple_random_sampling_l711_71118

/-- Represents a sampling method -/
inductive SamplingMethod
  | StratifiedSampling
  | SimpleRandomSampling
  | SystematicSampling

/-- Represents a region with its number of outlets -/
structure Region where
  name : String
  outlets : Nat

/-- Represents the company's sales outlet distribution -/
structure CompanyDistribution where
  regions : List Region
  totalOutlets : Nat

/-- Represents a sampling scenario -/
structure SamplingScenario where
  population : CompanyDistribution
  sampleSize : Nat
  hasDistinctSubgroups : Bool

/-- Determines the most appropriate sampling method for a given scenario -/
def appropriateSamplingMethod (scenario : SamplingScenario) : SamplingMethod :=
  if scenario.hasDistinctSubgroups then
    SamplingMethod.StratifiedSampling
  else
    SamplingMethod.SimpleRandomSampling

/-- Theorem: Stratified sampling is more appropriate for populations with distinct subgroups -/
theorem stratified_sampling_appropriate_for_subgroups 
  (scenario : SamplingScenario) 
  (h : scenario.hasDistinctSubgroups = true) : 
  appropriateSamplingMethod scenario = SamplingMethod.StratifiedSampling :=
sorry

/-- Company distribution for the given problem -/
def companyDistribution : CompanyDistribution :=
  { regions := [
      { name := "A", outlets := 150 },
      { name := "B", outlets := 120 },
      { name := "C", outlets := 180 },
      { name := "D", outlets := 150 }
    ],
    totalOutlets := 600
  }

/-- Sampling scenario for investigation (1) -/
def investigation1 : SamplingScenario :=
  { population := companyDistribution,
    sampleSize := 100,
    hasDistinctSubgroups := true
  }

/-- Sampling scenario for investigation (2) -/
def investigation2 : SamplingScenario :=
  { population := { regions := [{ name := "C_large", outlets := 20 }], totalOutlets := 20 },
    sampleSize := 7,
    hasDistinctSubgroups := false
  }

/-- Theorem: The appropriate sampling method for investigation (1) is Stratified Sampling -/
theorem investigation1_uses_stratified_sampling :
  appropriateSamplingMethod investigation1 = SamplingMethod.StratifiedSampling :=
sorry

/-- Theorem: The appropriate sampling method for investigation (2) is Simple Random Sampling -/
theorem investigation2_uses_simple_random_sampling :
  appropriateSamplingMethod investigation2 = SamplingMethod.SimpleRandomSampling :=
sorry

end NUMINAMATH_CALUDE_stratified_sampling_appropriate_for_subgroups_investigation1_uses_stratified_sampling_investigation2_uses_simple_random_sampling_l711_71118


namespace NUMINAMATH_CALUDE_range_of_a_for_quadratic_inequality_l711_71190

theorem range_of_a_for_quadratic_inequality :
  ∀ a : ℝ, (∀ x : ℝ, x^2 + 2*a*x + a > 0) ↔ (0 < a ∧ a < 1) := by sorry

end NUMINAMATH_CALUDE_range_of_a_for_quadratic_inequality_l711_71190


namespace NUMINAMATH_CALUDE_congruence_solution_l711_71167

theorem congruence_solution (n : ℕ) : n = 21 → 0 ≤ n ∧ n < 47 ∧ (13 * n) % 47 = 8 := by
  sorry

end NUMINAMATH_CALUDE_congruence_solution_l711_71167


namespace NUMINAMATH_CALUDE_building_heights_l711_71148

/-- Given three buildings with specified height relationships, calculate their total height. -/
theorem building_heights (height_1 : ℝ) : 
  height_1 = 600 →
  let height_2 := 2 * height_1
  let height_3 := 3 * (height_1 + height_2)
  height_1 + height_2 + height_3 = 7200 := by
  sorry

end NUMINAMATH_CALUDE_building_heights_l711_71148


namespace NUMINAMATH_CALUDE_natural_triple_solutions_l711_71153

-- Define the natural triple (a, b, c)
def natural_triple (a b c : ℕ) : Prop :=
  a ≤ b ∧ b ≤ c

-- Define the GCD condition
def gcd_condition (a b c : ℕ) : Prop :=
  Nat.gcd a (Nat.gcd b c) = 1

-- Define the divisibility condition
def divisibility_condition (a b c : ℕ) : Prop :=
  (a^2 * b) ∣ (a^3 + b^3 + c^3) ∧
  (b^2 * c) ∣ (a^3 + b^3 + c^3) ∧
  (c^2 * a) ∣ (a^3 + b^3 + c^3)

-- Theorem statement
theorem natural_triple_solutions :
  ∀ a b c : ℕ,
    natural_triple a b c →
    gcd_condition a b c →
    divisibility_condition a b c →
    ((a, b, c) = (1, 1, 1) ∨ (a, b, c) = (1, 2, 3)) :=
by sorry

end NUMINAMATH_CALUDE_natural_triple_solutions_l711_71153


namespace NUMINAMATH_CALUDE_date_sum_equality_l711_71184

/-- Represents a calendar date sequence -/
structure DateSequence where
  x : ℕ  -- Date behind C
  dateA : ℕ := x + 2  -- Date behind A
  dateB : ℕ := x + 11  -- Date behind B
  dateP : ℕ := x + 13  -- Date behind P

/-- Theorem: The sum of dates behind C and P equals the sum of dates behind A and B -/
theorem date_sum_equality (d : DateSequence) : 
  d.x + d.dateP = d.dateA + d.dateB := by
  sorry

end NUMINAMATH_CALUDE_date_sum_equality_l711_71184


namespace NUMINAMATH_CALUDE_girls_candies_contradiction_l711_71132

theorem girls_candies_contradiction (cM cK cL cO : ℕ) : 
  (cM + cK = cL + cO + 12) → (cK + cL = cM + cO - 7) → False :=
by
  sorry

end NUMINAMATH_CALUDE_girls_candies_contradiction_l711_71132


namespace NUMINAMATH_CALUDE_leonard_younger_than_nina_l711_71113

/-- Given the ages of Leonard, Nina, and Jerome, prove that Leonard is 4 years younger than Nina. -/
theorem leonard_younger_than_nina :
  ∀ (leonard nina jerome : ℕ),
    leonard = 6 →
    nina = jerome / 2 →
    leonard + nina + jerome = 36 →
    nina - leonard = 4 :=
by sorry

end NUMINAMATH_CALUDE_leonard_younger_than_nina_l711_71113


namespace NUMINAMATH_CALUDE_negative_plus_square_not_always_positive_l711_71120

theorem negative_plus_square_not_always_positive : 
  ∃ x : ℝ, x < 0 ∧ x + x^2 ≤ 0 := by
  sorry

end NUMINAMATH_CALUDE_negative_plus_square_not_always_positive_l711_71120

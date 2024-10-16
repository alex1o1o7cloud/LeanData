import Mathlib

namespace NUMINAMATH_CALUDE_f_monotonicity_and_b_range_l2914_291437

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := (a / (a^2 - 1)) * (a^x - a^(-x))

def g (b : ℝ) (x : ℝ) : ℝ := x^2 + (2*b + 1)*x - b - 1

def prop_p (a b : ℝ) : Prop := ∀ x ∈ Set.Icc (-1) 1, f a x ≤ 2*b

def prop_q (b : ℝ) : Prop :=
  ∃ (x₁ x₂ : ℝ), x₁ ∈ Set.Ioo (-3) (-2) ∧ x₂ ∈ Set.Ioo 0 1 ∧
  g b x₁ = 0 ∧ g b x₂ = 0

theorem f_monotonicity_and_b_range (a : ℝ) (h₁ : a > 0) (h₂ : a ≠ 1) :
  (∀ x₁ x₂ : ℝ, x₁ < x₂ → f a x₁ < f a x₂) ∧
  {b : ℝ | (prop_p a b ∨ prop_q b) ∧ ¬(prop_p a b ∧ prop_q b)} =
  Set.Ioo (1/5) (1/2) ∪ Set.Ici (5/7) :=
sorry

end NUMINAMATH_CALUDE_f_monotonicity_and_b_range_l2914_291437


namespace NUMINAMATH_CALUDE_solution_set_of_inequality_range_of_a_l2914_291476

-- Define the function f(x)
def f (x : ℝ) : ℝ := |x - 2| + |x + 3|

-- Theorem for the first part of the problem
theorem solution_set_of_inequality (x : ℝ) :
  f x ≤ x + 5 ↔ 0 ≤ x ∧ x ≤ 4 := by sorry

-- Theorem for the second part of the problem
theorem range_of_a (a : ℝ) :
  (∀ x : ℝ, f x ≥ a^2 + 4*a) → -5 ≤ a ∧ a ≤ 1 := by sorry

end NUMINAMATH_CALUDE_solution_set_of_inequality_range_of_a_l2914_291476


namespace NUMINAMATH_CALUDE_gcd_654327_543216_l2914_291479

theorem gcd_654327_543216 : Nat.gcd 654327 543216 = 1 := by
  sorry

end NUMINAMATH_CALUDE_gcd_654327_543216_l2914_291479


namespace NUMINAMATH_CALUDE_right_triangle_hypotenuse_l2914_291483

/-- Given a right triangle with legs a and b, if rotating about leg a produces a cone
    of volume 500π cm³ and rotating about leg b produces a cone of volume 1800π cm³,
    then the length of the hypotenuse is approximately 24.46 cm. -/
theorem right_triangle_hypotenuse (a b : ℝ) (ha : a > 0) (hb : b > 0) : 
  (1/3 * π * a * b^2 = 500 * π) →
  (1/3 * π * b * a^2 = 1800 * π) →
  abs ((a^2 + b^2).sqrt - 24.46) < 0.01 := by
sorry

end NUMINAMATH_CALUDE_right_triangle_hypotenuse_l2914_291483


namespace NUMINAMATH_CALUDE_f_properties_l2914_291434

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := 2 * Real.sqrt 3 * Real.sin x * Real.cos x + 2 * (Real.cos x) ^ 2 + a

theorem f_properties (a : ℝ) :
  (∃ (T : ℝ), ∀ (x : ℝ), f a x = f a (x + T)) ∧ 
  (∃ (min_val : ℝ), min_val = 0 → 
    (a = 1 ∧ 
     (∃ (max_val : ℝ), max_val = 4 ∧ ∀ (x : ℝ), f a x ≤ max_val) ∧
     (∃ (k : ℤ), ∀ (x : ℝ), f a x = f a (↑k * Real.pi / 2 + Real.pi / 6 - x)))) :=
by sorry

end NUMINAMATH_CALUDE_f_properties_l2914_291434


namespace NUMINAMATH_CALUDE_product_expansion_l2914_291406

theorem product_expansion (x : ℝ) : (x^2 - 3*x + 3) * (x^2 + 3*x + 3) = x^4 - 3*x^2 + 9 := by
  sorry

end NUMINAMATH_CALUDE_product_expansion_l2914_291406


namespace NUMINAMATH_CALUDE_derivative_at_pi_third_l2914_291489

theorem derivative_at_pi_third (f : ℝ → ℝ) (f' : ℝ → ℝ) :
  (∀ x, f x = x^2 * f' (π/3) + Real.sin x) →
  (∀ x, HasDerivAt f (f' x) x) →
  f' (π/3) = 3 / (6 - 4*π) := by
sorry

end NUMINAMATH_CALUDE_derivative_at_pi_third_l2914_291489


namespace NUMINAMATH_CALUDE_regression_lines_intersect_l2914_291496

/-- Represents a linear regression line -/
structure RegressionLine where
  slope : ℝ
  intercept : ℝ

/-- The sample center point of a dataset -/
structure SampleCenterPoint where
  x : ℝ
  y : ℝ

/-- Theorem: Two regression lines with the same sample center point intersect -/
theorem regression_lines_intersect
  (l₁ l₂ : RegressionLine)
  (center : SampleCenterPoint)
  (h₁ : center.y = l₁.slope * center.x + l₁.intercept)
  (h₂ : center.y = l₂.slope * center.x + l₂.intercept) :
  ∃ (x y : ℝ), y = l₁.slope * x + l₁.intercept ∧ y = l₂.slope * x + l₂.intercept :=
sorry

end NUMINAMATH_CALUDE_regression_lines_intersect_l2914_291496


namespace NUMINAMATH_CALUDE_min_value_of_expression_l2914_291456

theorem min_value_of_expression (x : ℝ) (h : x > 0) :
  3 * x + 1 / (x^3) ≥ 4 ∧
  (3 * x + 1 / (x^3) = 4 ↔ x = 1) :=
by sorry

end NUMINAMATH_CALUDE_min_value_of_expression_l2914_291456


namespace NUMINAMATH_CALUDE_equation_represents_hyperbola_l2914_291491

/-- Represents a conic section --/
inductive ConicSection
  | Parabola
  | Circle
  | Ellipse
  | Hyperbola
  | Point
  | Line
  | TwoLines
  | Empty

/-- Determines the type of conic section represented by the given equation --/
def determineConicSection (a b c d e f : ℝ) : ConicSection :=
  sorry

/-- The equation x^2 - 64y^2 - 12x + 16y + 36 = 0 represents a hyperbola --/
theorem equation_represents_hyperbola :
  determineConicSection 1 (-64) 0 (-12) 16 36 = ConicSection.Hyperbola :=
by sorry

end NUMINAMATH_CALUDE_equation_represents_hyperbola_l2914_291491


namespace NUMINAMATH_CALUDE_equilateral_triangle_intersection_l2914_291442

theorem equilateral_triangle_intersection (a : ℝ) : 
  (∃ (A B : ℝ × ℝ), 
    (A.1 + A.2 = Real.sqrt 3 * a ∧ A.1^2 + A.2^2 = a^2 + (a-1)^2) ∧
    (B.1 + B.2 = Real.sqrt 3 * a ∧ B.1^2 + B.2^2 = a^2 + (a-1)^2) ∧
    (A.1 - B.1)^2 + (A.2 - B.2)^2 = A.1^2 + A.2^2 ∧
    (A.1 - B.1)^2 + (A.2 - B.2)^2 = B.1^2 + B.2^2) →
  a = 1/2 := by
sorry

end NUMINAMATH_CALUDE_equilateral_triangle_intersection_l2914_291442


namespace NUMINAMATH_CALUDE_mary_baseball_cards_l2914_291439

theorem mary_baseball_cards 
  (promised_to_fred : ℝ) 
  (bought : ℝ) 
  (left_after_giving : ℝ) 
  (h1 : promised_to_fred = 26.0)
  (h2 : bought = 40.0)
  (h3 : left_after_giving = 32.0) :
  ∃ initial : ℝ, initial = 18.0 ∧ 
    (initial + bought - promised_to_fred = left_after_giving) :=
by sorry

end NUMINAMATH_CALUDE_mary_baseball_cards_l2914_291439


namespace NUMINAMATH_CALUDE_school_purchase_cost_l2914_291446

/-- Calculates the total cost of pencils and pens after all applicable discounts -/
def totalCostAfterDiscounts (pencilPrice penPrice : ℚ) (pencilCount penCount : ℕ) 
  (pencilDiscountThreshold penDiscountThreshold : ℕ) 
  (pencilDiscountRate penDiscountRate additionalDiscountRate : ℚ)
  (additionalDiscountThreshold : ℚ) : ℚ :=
  sorry

theorem school_purchase_cost : 
  let pencilPrice : ℚ := 2.5
  let penPrice : ℚ := 3.5
  let pencilCount : ℕ := 38
  let penCount : ℕ := 56
  let pencilDiscountThreshold : ℕ := 30
  let penDiscountThreshold : ℕ := 50
  let pencilDiscountRate : ℚ := 0.1
  let penDiscountRate : ℚ := 0.15
  let additionalDiscountRate : ℚ := 0.05
  let additionalDiscountThreshold : ℚ := 250

  totalCostAfterDiscounts pencilPrice penPrice pencilCount penCount 
    pencilDiscountThreshold penDiscountThreshold 
    pencilDiscountRate penDiscountRate additionalDiscountRate
    additionalDiscountThreshold = 239.5 := by
  sorry

end NUMINAMATH_CALUDE_school_purchase_cost_l2914_291446


namespace NUMINAMATH_CALUDE_square_root_range_l2914_291450

theorem square_root_range (x : ℝ) : 3 - 2*x ≥ 0 → x ≤ 3/2 := by
  sorry

end NUMINAMATH_CALUDE_square_root_range_l2914_291450


namespace NUMINAMATH_CALUDE_pool_filling_time_l2914_291480

/-- Proves that it takes 33 hours to fill a 30,000-gallon pool with 5 hoses supplying 3 gallons per minute each -/
theorem pool_filling_time : 
  let pool_capacity : ℕ := 30000
  let num_hoses : ℕ := 5
  let flow_rate_per_hose : ℕ := 3
  let minutes_per_hour : ℕ := 60
  let total_flow_rate_per_hour : ℕ := num_hoses * flow_rate_per_hose * minutes_per_hour
  let filling_time_hours : ℕ := pool_capacity / total_flow_rate_per_hour
  filling_time_hours = 33 := by
  sorry


end NUMINAMATH_CALUDE_pool_filling_time_l2914_291480


namespace NUMINAMATH_CALUDE_angle_measure_in_special_quadrilateral_l2914_291488

/-- In a quadrilateral EFGH where ∠E = 3∠F = 4∠G = 6∠H, the measure of ∠E is 360 * (4/7) degrees. -/
theorem angle_measure_in_special_quadrilateral :
  ∀ (E F G H : ℝ),
  E + F + G + H = 360 →
  E = 3 * F →
  E = 4 * G →
  E = 6 * H →
  E = 360 * (4/7) := by
sorry

end NUMINAMATH_CALUDE_angle_measure_in_special_quadrilateral_l2914_291488


namespace NUMINAMATH_CALUDE_absolute_value_sqrt_five_l2914_291460

theorem absolute_value_sqrt_five (x : ℝ) : 
  |x| = Real.sqrt 5 → x = Real.sqrt 5 ∨ x = -Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_absolute_value_sqrt_five_l2914_291460


namespace NUMINAMATH_CALUDE_sqrt_720_simplified_l2914_291427

theorem sqrt_720_simplified : Real.sqrt 720 = 12 * Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_720_simplified_l2914_291427


namespace NUMINAMATH_CALUDE_beetle_speed_l2914_291495

/-- Proves that a beetle's speed is 2.7 km/h given specific conditions --/
theorem beetle_speed : 
  let ant_distance : ℝ := 600 -- meters
  let ant_time : ℝ := 10 -- minutes
  let beetle_distance_ratio : ℝ := 0.75 -- 25% less than ant
  let beetle_distance : ℝ := ant_distance * beetle_distance_ratio
  let km_per_meter : ℝ := 1 / 1000
  let hours_per_minute : ℝ := 1 / 60
  beetle_distance * km_per_meter / (ant_time * hours_per_minute) = 2.7 := by
sorry

end NUMINAMATH_CALUDE_beetle_speed_l2914_291495


namespace NUMINAMATH_CALUDE_range_of_m_l2914_291414

open Set Real

-- Define the sets A and B
def A : Set ℝ := {x | x^2 + 2*x - 8 < 0}
def B (m : ℝ) : Set ℝ := {x | 5 - m < x ∧ x < 2*m - 1}

-- Define the universe U as the set of real numbers
def U : Set ℝ := univ

-- Theorem statement
theorem range_of_m :
  ∀ m : ℝ, (A ∩ (U \ B m) = A) ↔ m ≤ 3 :=
sorry

end NUMINAMATH_CALUDE_range_of_m_l2914_291414


namespace NUMINAMATH_CALUDE_sum_of_tangent_slopes_l2914_291421

/-- The circle with center (2, -1) and radius √2 -/
def C : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | (p.1 - 2)^2 + (p.2 + 1)^2 = 2}

/-- A line passing through the origin with slope k -/
def tangentLine (k : ℝ) : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.2 = k * p.1}

/-- The set of slopes of lines passing through the origin and tangent to C -/
def tangentSlopes : Set ℝ :=
  {k : ℝ | ∃ p ∈ C, p ∈ tangentLine k ∧ (0, 0) ∈ tangentLine k}

theorem sum_of_tangent_slopes :
  ∃ (k₁ k₂ : ℝ), k₁ ∈ tangentSlopes ∧ k₂ ∈ tangentSlopes ∧ k₁ + k₂ = -2 :=
sorry

end NUMINAMATH_CALUDE_sum_of_tangent_slopes_l2914_291421


namespace NUMINAMATH_CALUDE_cube_sum_and_reciprocal_l2914_291445

theorem cube_sum_and_reciprocal (x : ℝ) (h : x + 1/x = 7) : x^3 + 1/x^3 = 322 := by
  sorry

end NUMINAMATH_CALUDE_cube_sum_and_reciprocal_l2914_291445


namespace NUMINAMATH_CALUDE_pens_distribution_l2914_291465

def number_of_friends (kendra_packs tony_packs pens_per_pack pens_kept_per_person : ℕ) : ℕ :=
  (kendra_packs + tony_packs) * pens_per_pack - 2 * pens_kept_per_person

theorem pens_distribution (kendra_packs tony_packs pens_per_pack pens_kept_per_person : ℕ) 
  (h1 : kendra_packs = 4)
  (h2 : tony_packs = 2)
  (h3 : pens_per_pack = 3)
  (h4 : pens_kept_per_person = 2) :
  number_of_friends kendra_packs tony_packs pens_per_pack pens_kept_per_person = 14 := by
  sorry

end NUMINAMATH_CALUDE_pens_distribution_l2914_291465


namespace NUMINAMATH_CALUDE_g_sum_property_l2914_291473

def g (x : ℝ) : ℝ := 2 * x^8 + 3 * x^6 - 5 * x^4 + 7

theorem g_sum_property : g 10 = 15 → g 10 + g (-10) = 30 := by
  sorry

end NUMINAMATH_CALUDE_g_sum_property_l2914_291473


namespace NUMINAMATH_CALUDE_bean_garden_columns_l2914_291459

/-- A garden with bean plants arranged in rows and columns. -/
structure BeanGarden where
  rows : ℕ
  columns : ℕ
  total_plants : ℕ
  h_total : total_plants = rows * columns

/-- The number of columns in a bean garden with 52 rows and 780 total plants is 15. -/
theorem bean_garden_columns (garden : BeanGarden) 
    (h_rows : garden.rows = 52) 
    (h_total : garden.total_plants = 780) : 
    garden.columns = 15 := by
  sorry

end NUMINAMATH_CALUDE_bean_garden_columns_l2914_291459


namespace NUMINAMATH_CALUDE_sarah_homework_problem_l2914_291402

/-- The total number of problems Sarah has to complete given her homework assignments -/
def total_problems (math_pages reading_pages science_pages : ℕ) 
  (math_problems_per_page reading_problems_per_page science_problems_per_page : ℕ) : ℕ :=
  math_pages * math_problems_per_page + 
  reading_pages * reading_problems_per_page + 
  science_pages * science_problems_per_page

theorem sarah_homework_problem :
  total_problems 4 6 5 4 4 6 = 70 := by
  sorry

end NUMINAMATH_CALUDE_sarah_homework_problem_l2914_291402


namespace NUMINAMATH_CALUDE_exist_irrational_with_natural_power_l2914_291467

theorem exist_irrational_with_natural_power : 
  ∃ (a b : ℝ), a > 0 ∧ b > 0 ∧ Irrational a ∧ Irrational b ∧ ∃ (n : ℕ), a^b = n :=
sorry

end NUMINAMATH_CALUDE_exist_irrational_with_natural_power_l2914_291467


namespace NUMINAMATH_CALUDE_students_opted_for_math_and_science_l2914_291412

/-- Given a class with the following properties:
  * There are 40 students in total.
  * 10 students did not opt for math.
  * 15 students did not opt for science.
  * 20 students did not opt for history.
  * 5 students did not opt for geography.
  * 2 students did not opt for either math or science.
  * 3 students did not opt for either math or history.
  * 4 students did not opt for either math or geography.
  * 7 students did not opt for either science or history.
  * 8 students did not opt for either science or geography.
  * 10 students did not opt for either history or geography.

  Prove that the number of students who opted for both math and science is 17. -/
theorem students_opted_for_math_and_science
  (total : ℕ) (not_math : ℕ) (not_science : ℕ) (not_history : ℕ) (not_geography : ℕ)
  (not_math_or_science : ℕ) (not_math_or_history : ℕ) (not_math_or_geography : ℕ)
  (not_science_or_history : ℕ) (not_science_or_geography : ℕ) (not_history_or_geography : ℕ)
  (h_total : total = 40)
  (h_not_math : not_math = 10)
  (h_not_science : not_science = 15)
  (h_not_history : not_history = 20)
  (h_not_geography : not_geography = 5)
  (h_not_math_or_science : not_math_or_science = 2)
  (h_not_math_or_history : not_math_or_history = 3)
  (h_not_math_or_geography : not_math_or_geography = 4)
  (h_not_science_or_history : not_science_or_history = 7)
  (h_not_science_or_geography : not_science_or_geography = 8)
  (h_not_history_or_geography : not_history_or_geography = 10) :
  (total - not_math) + (total - not_science) - (total - not_math_or_science) = 17 := by
  sorry

end NUMINAMATH_CALUDE_students_opted_for_math_and_science_l2914_291412


namespace NUMINAMATH_CALUDE_grape_bowl_comparison_l2914_291478

theorem grape_bowl_comparison (rob_grapes allie_grapes allyn_grapes : ℕ) : 
  rob_grapes = 25 →
  allie_grapes = rob_grapes + 2 →
  rob_grapes + allie_grapes + allyn_grapes = 83 →
  allyn_grapes - allie_grapes = 4 :=
by sorry

end NUMINAMATH_CALUDE_grape_bowl_comparison_l2914_291478


namespace NUMINAMATH_CALUDE_leap_year_53_mondays_probability_l2914_291498

/-- A leap year has 366 days -/
def leapYearDays : ℕ := 366

/-- A leap year has 52 weeks plus 2 days -/
def leapYearWeeksAndDays : ℕ × ℕ := (52, 2)

/-- There are 7 possible days for a year to start on -/
def possibleStartDays : ℕ := 7

/-- The probability of a randomly selected leap year having 53 Mondays -/
def probLeapYear53Mondays : ℚ := 2 / 7

theorem leap_year_53_mondays_probability :
  probLeapYear53Mondays = 2 / 7 := by sorry

end NUMINAMATH_CALUDE_leap_year_53_mondays_probability_l2914_291498


namespace NUMINAMATH_CALUDE_f_composition_value_l2914_291440

noncomputable def f (x : ℝ) : ℝ :=
  if x < 0 then 5 * x + 4 else 2^x

def angle_terminal_side_point (α : ℝ) (p : ℝ × ℝ) : Prop :=
  p.2 / p.1 = Real.tan α

theorem f_composition_value (α : ℝ) :
  angle_terminal_side_point α (4, -3) →
  f (f (Real.sin α)) = 2 := by
  sorry

end NUMINAMATH_CALUDE_f_composition_value_l2914_291440


namespace NUMINAMATH_CALUDE_system_of_equations_solution_l2914_291417

theorem system_of_equations_solution (x y m : ℝ) : 
  (3 * x + 5 * y = m + 2) → 
  (2 * x + 3 * y = m) → 
  (x + y = -10) → 
  (m^2 - 2*m + 1 = 81) := by
sorry

end NUMINAMATH_CALUDE_system_of_equations_solution_l2914_291417


namespace NUMINAMATH_CALUDE_value_of_a_l2914_291477

-- Define the functions f and g
def f (x : ℝ) : ℝ := 3 * x^2 + 14
def g (x : ℝ) : ℝ := x^3 - 4

-- State the theorem
theorem value_of_a (a : ℝ) (h1 : a > 0) (h2 : f (g a) = 18) :
  a = (4 + 2 * Real.sqrt 3 / 3) ^ (1/3) := by
  sorry


end NUMINAMATH_CALUDE_value_of_a_l2914_291477


namespace NUMINAMATH_CALUDE_intersection_of_A_and_B_l2914_291485

def A : Set ℝ := {x | x < 3}
def B : Set ℝ := {x | 2 - x < 1}

theorem intersection_of_A_and_B : A ∩ B = {x | 1 < x ∧ x < 3} := by sorry

end NUMINAMATH_CALUDE_intersection_of_A_and_B_l2914_291485


namespace NUMINAMATH_CALUDE_chopped_cube_height_l2914_291497

/-- Represents a 3D cube with a chopped corner --/
structure ChoppedCube where
  side_length : ℝ
  cut_ratio : ℝ

/-- The height of the remaining solid when the chopped face is placed on a table --/
def remaining_height (c : ChoppedCube) : ℝ :=
  c.side_length - c.cut_ratio * c.side_length

/-- Theorem stating that for a 2x2x2 cube with a corner chopped at midpoints, 
    the remaining height is 1 unit --/
theorem chopped_cube_height :
  let c : ChoppedCube := { side_length := 2, cut_ratio := 1/2 }
  remaining_height c = 1 := by
  sorry

end NUMINAMATH_CALUDE_chopped_cube_height_l2914_291497


namespace NUMINAMATH_CALUDE_expression_equals_negative_one_l2914_291449

theorem expression_equals_negative_one (a b : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) (hc : b ≠ a) (hd : b ≠ -a) :
  (a / (a + b) + b / (a - b)) / (b / (a + b) - a / (a - b)) = -1 := by
  sorry

end NUMINAMATH_CALUDE_expression_equals_negative_one_l2914_291449


namespace NUMINAMATH_CALUDE_exists_valid_configuration_l2914_291451

/-- Represents a lamp in a room -/
structure Lamp where
  room : Nat
  state : Bool

/-- Represents a switch controlling a pair of lamps -/
structure Switch where
  lamp1 : Lamp
  lamp2 : Lamp

/-- Configuration of lamps and switches -/
structure Configuration (k : Nat) where
  lamps : Fin (6 * k) → Lamp
  switches : Fin (3 * k) → Switch
  rooms : Fin (2 * k)

/-- Predicate to check if a room has at least one lamp on and one off -/
def validRoom (config : Configuration k) (room : Fin (2 * k)) : Prop :=
  ∃ (l1 l2 : Fin (6 * k)), 
    (config.lamps l1).room = room ∧ 
    (config.lamps l2).room = room ∧ 
    (config.lamps l1).state = true ∧ 
    (config.lamps l2).state = false

/-- Main theorem statement -/
theorem exists_valid_configuration (k : Nat) (h : k > 0) : 
  ∃ (config : Configuration k), ∀ (room : Fin (2 * k)), validRoom config room :=
sorry

end NUMINAMATH_CALUDE_exists_valid_configuration_l2914_291451


namespace NUMINAMATH_CALUDE_hyperbola_asymptote_l2914_291458

/-- Given a hyperbola with equation x²/a² - y²/4 = 1 (a > 0),
    if one of its asymptotes passes through the point (2, 1), then a = 4 -/
theorem hyperbola_asymptote (a : ℝ) (h1 : a > 0) : 
  (∃ x y, x^2 / a^2 - y^2 / 4 = 1 ∧ 
   ((y = (2/a) * x ∨ y = -(2/a) * x) ∧ x = 2 ∧ y = 1)) → 
  a = 4 := by
sorry

end NUMINAMATH_CALUDE_hyperbola_asymptote_l2914_291458


namespace NUMINAMATH_CALUDE_coefficient_x4_in_q_squared_l2914_291474

/-- Given q(x) = x^5 - 4x^2 + 3, prove that the coefficient of x^4 in (q(x))^2 is 16 -/
theorem coefficient_x4_in_q_squared (x : ℝ) : 
  let q : ℝ → ℝ := λ x => x^5 - 4*x^2 + 3
  (q x)^2 = x^10 - 8*x^7 + 16*x^4 + 6*x^5 - 24*x^2 + 9 := by
  sorry


end NUMINAMATH_CALUDE_coefficient_x4_in_q_squared_l2914_291474


namespace NUMINAMATH_CALUDE_f_extended_domain_l2914_291435

def is_even (f : ℝ → ℝ) : Prop := ∀ x, f x = f (-x)

def has_period (f : ℝ → ℝ) (p : ℝ) : Prop := ∀ x, f (x + p) = f x

theorem f_extended_domain (f : ℝ → ℝ) :
  is_even f →
  has_period f π →
  (∀ x ∈ Set.Icc 0 (π / 2), f x = 1 - Real.sin x) →
  ∀ x ∈ Set.Icc (5 * π / 2) (3 * π), f x = 1 - Real.sin x :=
by sorry

end NUMINAMATH_CALUDE_f_extended_domain_l2914_291435


namespace NUMINAMATH_CALUDE_daniel_noodles_left_l2914_291413

/-- Given that Daniel initially had 66 noodles and gave 12 noodles to William,
    prove that he now has 54 noodles. -/
theorem daniel_noodles_left (initial : ℕ) (given : ℕ) (h1 : initial = 66) (h2 : given = 12) :
  initial - given = 54 := by sorry

end NUMINAMATH_CALUDE_daniel_noodles_left_l2914_291413


namespace NUMINAMATH_CALUDE_smallest_n_congruence_l2914_291407

theorem smallest_n_congruence (n : ℕ) : n = 7 ↔ 
  (n > 0 ∧ 5 * n % 26 = 789 % 26 ∧ ∀ m : ℕ, m > 0 → m < n → 5 * m % 26 ≠ 789 % 26) :=
by sorry

end NUMINAMATH_CALUDE_smallest_n_congruence_l2914_291407


namespace NUMINAMATH_CALUDE_charity_donation_division_l2914_291492

theorem charity_donation_division (total : ℕ) (people : ℕ) (share : ℕ) : 
  total = 1800 → people = 10 → share = total / people → share = 180 := by
  sorry

end NUMINAMATH_CALUDE_charity_donation_division_l2914_291492


namespace NUMINAMATH_CALUDE_function_property_l2914_291409

theorem function_property (f : ℝ → ℝ) 
  (h1 : ∀ (x y : ℝ), x > 0 → y > 0 → f (x * y) = f x / y)
  (h2 : f 1000 = 6) : 
  f 800 = 7.5 := by
sorry

end NUMINAMATH_CALUDE_function_property_l2914_291409


namespace NUMINAMATH_CALUDE_complex_expression_squared_l2914_291461

theorem complex_expression_squared (x y z p : ℝ) 
  (h1 : x^2 + y^2 + z^2 = 15)
  (h2 : x * y = 3)
  (h3 : x * z = 4)
  (h4 : Real.cos x + Real.sin y + Real.tan z = p) :
  (x - y - z)^2 = (Real.sqrt ((15 + 5 * Real.sqrt 5) / 2) - 
                   3 / Real.sqrt ((15 + 5 * Real.sqrt 5) / 2) - 
                   4 / Real.sqrt ((15 + 5 * Real.sqrt 5) / 2))^2 := by
  sorry

end NUMINAMATH_CALUDE_complex_expression_squared_l2914_291461


namespace NUMINAMATH_CALUDE_tower_of_hanoi_minimum_moves_five_disks_minimum_moves_l2914_291415

/-- Minimum number of moves required to solve the Tower of Hanoi puzzle with n disks -/
def tower_of_hanoi_moves (n : ℕ) : ℕ := 2^n - 1

/-- The number of disks in our specific problem -/
def num_disks : ℕ := 5

theorem tower_of_hanoi_minimum_moves :
  ∀ n : ℕ, tower_of_hanoi_moves n = 2^n - 1 :=
sorry

theorem five_disks_minimum_moves :
  tower_of_hanoi_moves num_disks = 31 :=
sorry

end NUMINAMATH_CALUDE_tower_of_hanoi_minimum_moves_five_disks_minimum_moves_l2914_291415


namespace NUMINAMATH_CALUDE_probability_of_all_even_sums_l2914_291411

/-- Represents a tile with a number from 1 to 9 -/
def Tile := Fin 9

/-- Represents a player's selection of three tiles -/
def Selection := Fin 3 → Tile

/-- The set of all possible selections -/
def AllSelections := Fin 3 → Selection

/-- Checks if a selection results in an even sum -/
def isEvenSum (s : Selection) : Prop :=
  ∃ (a b c : Tile), s 0 = a ∧ s 1 = b ∧ s 2 = c ∧ (a.val + b.val + c.val) % 2 = 0

/-- Checks if all three players have even sums -/
def allEvenSums (selections : AllSelections) : Prop :=
  ∀ i : Fin 3, isEvenSum (selections i)

/-- The total number of possible ways to distribute the tiles -/
def totalOutcomes : ℕ := 1680

/-- The number of favorable outcomes where all players have even sums -/
def favorableOutcomes : ℕ := 400

theorem probability_of_all_even_sums :
  (favorableOutcomes : ℚ) / totalOutcomes = 5 / 21 := by sorry

end NUMINAMATH_CALUDE_probability_of_all_even_sums_l2914_291411


namespace NUMINAMATH_CALUDE_water_level_drop_l2914_291400

/-- The drop in water level when removing a partially submerged spherical ball from a prism-shaped glass -/
theorem water_level_drop (a r h : ℝ) (ha : a > 0) (hr : r > 0) (hh : h > 0) (hhr : h < r) :
  let base_area := (3 * Real.sqrt 3 * a^2) / 2
  let submerged_height := r - h
  let submerged_volume := π * submerged_height^2 * (3*r - submerged_height) / 3
  submerged_volume / base_area = (6 * π * Real.sqrt 3) / 25 :=
by sorry

end NUMINAMATH_CALUDE_water_level_drop_l2914_291400


namespace NUMINAMATH_CALUDE_centroids_form_equilateral_triangle_l2914_291401

/-- Given a triangle ABC with vertices z₁, z₂, z₃ in the complex plane,
    the centroids of equilateral triangles constructed externally on its sides
    form an equilateral triangle. -/
theorem centroids_form_equilateral_triangle (z₁ z₂ z₃ : ℂ) : 
  let g₁ := (z₁ * (1 + Complex.exp (Real.pi * Complex.I / 3)) + z₂ * (2 - Complex.exp (Real.pi * Complex.I / 3))) / 3
  let g₂ := (z₂ * (1 + Complex.exp (Real.pi * Complex.I / 3)) + z₃ * (2 - Complex.exp (Real.pi * Complex.I / 3))) / 3
  let g₃ := (z₃ * (1 + Complex.exp (Real.pi * Complex.I / 3)) + z₁ * (2 - Complex.exp (Real.pi * Complex.I / 3))) / 3
  (g₂ - g₁) = (g₃ - g₁) * Complex.exp ((2 * Real.pi * Complex.I) / 3) :=
by sorry


end NUMINAMATH_CALUDE_centroids_form_equilateral_triangle_l2914_291401


namespace NUMINAMATH_CALUDE_prime_equation_solutions_l2914_291494

theorem prime_equation_solutions (x y : ℕ) : 
  Prime x ∧ Prime y → (x^2 - y^2 = x * y^2 - 19 ↔ (x = 2 ∧ y = 3) ∨ (x = 2 ∧ y = 7)) := by
  sorry

end NUMINAMATH_CALUDE_prime_equation_solutions_l2914_291494


namespace NUMINAMATH_CALUDE_candy_bar_cost_l2914_291447

/-- Given that Dan spent $13 in total on a candy bar and a chocolate, 
    and the chocolate costs $6, prove that the candy bar costs $7. -/
theorem candy_bar_cost (total_spent : ℕ) (chocolate_cost : ℕ) (candy_bar_cost : ℕ) : 
  total_spent = 13 → chocolate_cost = 6 → candy_bar_cost = 7 := by
  sorry

end NUMINAMATH_CALUDE_candy_bar_cost_l2914_291447


namespace NUMINAMATH_CALUDE_square_root_three_expansion_l2914_291423

theorem square_root_three_expansion 
  (a b c d : ℕ+) 
  (h : (a : ℝ) + (b : ℝ) * Real.sqrt 3 = ((c : ℝ) + (d : ℝ) * Real.sqrt 3) ^ 2) : 
  a = c ^ 2 + 3 * d ^ 2 ∧ b = 2 * c * d :=
sorry

end NUMINAMATH_CALUDE_square_root_three_expansion_l2914_291423


namespace NUMINAMATH_CALUDE_triangle_properties_l2914_291464

-- Define a triangle ABC
structure Triangle where
  A : Real
  B : Real
  C : Real
  a : Real
  b : Real
  c : Real

-- Define the theorem
theorem triangle_properties (t : Triangle) 
  (h1 : Real.cos t.A * Real.cos t.C + Real.sin t.A * Real.sin t.C + Real.cos t.B = 3/2)
  (h2 : t.b^2 = t.a * t.c)  -- Geometric progression condition
  (h3 : t.a / Real.tan t.A + t.c / Real.tan t.C = 2 * t.b / Real.tan t.B) :
  t.B = π/3 ∧ t.A = π/3 ∧ t.C = π/3 := by
  sorry

end NUMINAMATH_CALUDE_triangle_properties_l2914_291464


namespace NUMINAMATH_CALUDE_largest_prime_factor_of_sum_factorials_l2914_291452

def factorial (n : ℕ) : ℕ := (List.range n).foldl (· * ·) 1

def sum_of_factorials : ℕ := factorial 5 + factorial 6

theorem largest_prime_factor_of_sum_factorials :
  (Nat.factors sum_of_factorials).maximum? = some 7 := by
  sorry

end NUMINAMATH_CALUDE_largest_prime_factor_of_sum_factorials_l2914_291452


namespace NUMINAMATH_CALUDE_student_pet_difference_l2914_291430

/-- Represents a fourth-grade classroom at Maplewood School -/
structure Classroom where
  students : ℕ
  rabbits : ℕ
  guinea_pigs : ℕ

/-- The number of fourth-grade classrooms -/
def num_classrooms : ℕ := 5

/-- A standard fourth-grade classroom at Maplewood School -/
def standard_classroom : Classroom :=
  { students := 24
  , rabbits := 3
  , guinea_pigs := 2 }

/-- The total number of students in all classrooms -/
def total_students : ℕ := num_classrooms * standard_classroom.students

/-- The total number of pets (rabbits and guinea pigs) in all classrooms -/
def total_pets : ℕ := num_classrooms * (standard_classroom.rabbits + standard_classroom.guinea_pigs)

/-- Theorem: The difference between the total number of students and the total number of pets is 95 -/
theorem student_pet_difference : total_students - total_pets = 95 := by
  sorry

end NUMINAMATH_CALUDE_student_pet_difference_l2914_291430


namespace NUMINAMATH_CALUDE_unit_circle_image_l2914_291433

def unit_circle_mapping (z : ℂ) : Prop := Complex.abs z = 1

theorem unit_circle_image :
  ∀ z : ℂ, unit_circle_mapping z → Complex.abs (z^2) = 1 := by
sorry

end NUMINAMATH_CALUDE_unit_circle_image_l2914_291433


namespace NUMINAMATH_CALUDE_no_solution_for_all_a_b_l2914_291416

theorem no_solution_for_all_a_b : ∃ (a b : ℤ), a ≠ 0 ∧ b ≠ 0 ∧
  ¬∃ (x y : ℝ), (Real.tan (13 * x) * Real.tan (a * y) = 1) ∧
                (Real.tan (21 * x) * Real.tan (b * y) = 1) :=
by sorry

end NUMINAMATH_CALUDE_no_solution_for_all_a_b_l2914_291416


namespace NUMINAMATH_CALUDE_max_rectangle_area_l2914_291431

/-- The equation that the vertex coordinates must satisfy -/
def vertex_equation (x y : ℝ) : Prop :=
  |y + 1| * (y^2 + 2*y + 28) + |x - 2| = 9 * (y^2 + 2*y + 4)

/-- The area function of the rectangle -/
def rectangle_area (x : ℝ) : ℝ :=
  -4 * x * (x - 3)^3

/-- Theorem stating the maximum area of the rectangle -/
theorem max_rectangle_area :
  ∃ (x y : ℝ), vertex_equation x y ∧
    ∀ (x' y' : ℝ), vertex_equation x' y' →
      rectangle_area x ≥ rectangle_area x' ∧
      rectangle_area x = 34.171875 :=
sorry

end NUMINAMATH_CALUDE_max_rectangle_area_l2914_291431


namespace NUMINAMATH_CALUDE_inheritance_tax_calculation_l2914_291454

theorem inheritance_tax_calculation (x : ℝ) : 
  x > 0 →
  (0.25 * x + 0.15 * (x - 0.25 * x) = 15000) →
  x = 41379.31 := by
sorry

end NUMINAMATH_CALUDE_inheritance_tax_calculation_l2914_291454


namespace NUMINAMATH_CALUDE_triangle_area_l2914_291482

/-- Given an oblique triangle ABC with sides a, b, c opposite to angles A, B, C respectively,
    prove that its area is (5 * √3) / 4 under certain conditions. -/
theorem triangle_area (a b c A B C : ℝ) : 
  0 < A ∧ 0 < B ∧ 0 < C ∧  -- Angles are positive
  A + B + C = π ∧  -- Sum of angles in a triangle
  c * Real.sin A = Real.sqrt 3 * a * Real.cos C ∧  -- Given condition
  c = Real.sqrt 21 ∧  -- Given condition
  Real.sin C + Real.sin (B - A) = 5 * Real.sin (2 * A) →  -- Given condition
  (1 / 2) * a * b * Real.sin C = (5 * Real.sqrt 3) / 4 := by
  sorry

end NUMINAMATH_CALUDE_triangle_area_l2914_291482


namespace NUMINAMATH_CALUDE_courier_distance_l2914_291457

/-- The total distance from A to B -/
def total_distance : ℝ := 412.5

/-- The additional distance traveled -/
def additional_distance : ℝ := 60

/-- The ratio of distance covered to remaining distance at the first point -/
def initial_ratio : ℚ := 2/3

/-- The ratio of distance covered to remaining distance after traveling the additional distance -/
def final_ratio : ℚ := 6/5

theorem courier_distance :
  ∃ (x : ℝ),
    (2 * x) / (3 * x) = initial_ratio ∧
    (2 * x + additional_distance) / (3 * x - additional_distance) = final_ratio ∧
    5 * x = total_distance :=
by sorry

end NUMINAMATH_CALUDE_courier_distance_l2914_291457


namespace NUMINAMATH_CALUDE_cucumber_water_percentage_l2914_291420

/-- Calculates the new water percentage in cucumbers after evaporation -/
theorem cucumber_water_percentage
  (initial_weight : ℝ)
  (initial_water_percentage : ℝ)
  (final_weight : ℝ)
  (h1 : initial_weight = 100)
  (h2 : initial_water_percentage = 99)
  (h3 : final_weight = 20)
  : (final_weight - (initial_weight * (1 - initial_water_percentage / 100))) / final_weight * 100 = 95 := by
  sorry

end NUMINAMATH_CALUDE_cucumber_water_percentage_l2914_291420


namespace NUMINAMATH_CALUDE_principal_calculation_l2914_291424

/-- Simple interest calculation -/
def simple_interest (principal : ℚ) (rate : ℚ) (time : ℚ) : ℚ :=
  principal * rate * time

/-- Convert paise to rupees -/
def paise_to_rupees (paise : ℚ) : ℚ :=
  paise / 100

theorem principal_calculation :
  ∃ (principal : ℚ),
    simple_interest principal (paise_to_rupees 5) 6 = 6 ∧
    principal = 20 := by
  sorry

end NUMINAMATH_CALUDE_principal_calculation_l2914_291424


namespace NUMINAMATH_CALUDE_johns_remaining_budget_l2914_291468

/-- Calculates the remaining budget after a purchase -/
def remaining_budget (initial : ℚ) (spent : ℚ) : ℚ :=
  initial - spent

/-- Proves that given an initial budget of $999.00 and a purchase of $165.00, the remaining amount is $834.00 -/
theorem johns_remaining_budget :
  remaining_budget 999 165 = 834 := by
  sorry

end NUMINAMATH_CALUDE_johns_remaining_budget_l2914_291468


namespace NUMINAMATH_CALUDE_adrianna_gum_theorem_l2914_291481

/-- Calculates the remaining pieces of gum after sharing with friends -/
def remaining_gum (initial : ℕ) (additional : ℕ) (friends : ℕ) : ℕ :=
  initial + additional - friends

/-- Theorem stating that Adrianna's remaining gum pieces is 2 -/
theorem adrianna_gum_theorem (initial : ℕ) (additional : ℕ) (friends : ℕ)
  (h1 : initial = 10)
  (h2 : additional = 3)
  (h3 : friends = 11) :
  remaining_gum initial additional friends = 2 := by
  sorry

#eval remaining_gum 10 3 11

end NUMINAMATH_CALUDE_adrianna_gum_theorem_l2914_291481


namespace NUMINAMATH_CALUDE_triangle_with_perimeter_7_l2914_291432

def is_valid_triangle (a b c : ℕ) : Prop :=
  a + b > c ∧ b + c > a ∧ c + a > b

theorem triangle_with_perimeter_7 :
  ∀ a b c : ℕ,
  a + b + c = 7 →
  is_valid_triangle a b c →
  (a = 1 ∨ a = 2 ∨ a = 3) ∧
  (b = 1 ∨ b = 2 ∨ b = 3) ∧
  (c = 1 ∨ c = 2 ∨ c = 3) :=
by sorry

end NUMINAMATH_CALUDE_triangle_with_perimeter_7_l2914_291432


namespace NUMINAMATH_CALUDE_probability_at_least_three_successes_in_five_trials_l2914_291455

theorem probability_at_least_three_successes_in_five_trials : 
  let n : ℕ := 5
  let p : ℝ := 1/2
  let binomial_probability (k : ℕ) := (n.choose k) * p^k * (1-p)^(n-k)
  (binomial_probability 3 + binomial_probability 4 + binomial_probability 5) = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_probability_at_least_three_successes_in_five_trials_l2914_291455


namespace NUMINAMATH_CALUDE_quadratic_two_distinct_roots_l2914_291444

theorem quadratic_two_distinct_roots (k : ℝ) :
  (∃ x y : ℝ, x ≠ y ∧ x^2 + 2*x - k = 0 ∧ y^2 + 2*y - k = 0) ↔ k > -1 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_two_distinct_roots_l2914_291444


namespace NUMINAMATH_CALUDE_solution_abs_difference_l2914_291418

theorem solution_abs_difference (x y : ℝ) : 
  (Int.floor x : ℝ) + (y - Int.floor y) = 3.7 →
  (x - Int.floor x) + (Int.floor y : ℝ) = 4.2 →
  |x - 2*y| = 6.2 := by
sorry

end NUMINAMATH_CALUDE_solution_abs_difference_l2914_291418


namespace NUMINAMATH_CALUDE_geometric_sequence_sum_of_squares_l2914_291428

/-- A geometric sequence -/
def GeometricSequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = r * a n

theorem geometric_sequence_sum_of_squares 
  (a : ℕ → ℝ) 
  (h_geo : GeometricSequence a) 
  (h_sum : a 3 + a 5 = 5) 
  (h_prod : a 2 * a 6 = 4) : 
  a 3 ^ 2 + a 5 ^ 2 = 17 := by
sorry

end NUMINAMATH_CALUDE_geometric_sequence_sum_of_squares_l2914_291428


namespace NUMINAMATH_CALUDE_smallest_integer_for_prime_quadratic_l2914_291469

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ d : ℕ, d > 1 → d < n → ¬(n % d = 0)

def abs_value (n : ℤ) : ℕ := Int.natAbs n

def quadratic_expression (x : ℤ) : ℤ := 8 * x^2 - 53 * x + 21

theorem smallest_integer_for_prime_quadratic :
  ∀ x : ℤ, x < 8 → ¬(is_prime (abs_value (quadratic_expression x))) ∧
  is_prime (abs_value (quadratic_expression 8)) :=
by sorry

end NUMINAMATH_CALUDE_smallest_integer_for_prime_quadratic_l2914_291469


namespace NUMINAMATH_CALUDE_incorrect_translation_l2914_291470

/-- Represents a parabola of the form y = (x + a)² + b -/
structure Parabola where
  a : ℝ
  b : ℝ

/-- Checks if a parabola passes through the origin -/
def passes_through_origin (p : Parabola) : Prop :=
  0 = (0 + p.a)^2 + p.b

/-- Translates a parabola vertically -/
def translate_vertical (p : Parabola) (d : ℝ) : Parabola :=
  { a := p.a, b := p.b - d }

theorem incorrect_translation :
  let original := Parabola.mk 3 (-4)
  let translated := translate_vertical original 4
  ¬ passes_through_origin translated :=
by sorry

end NUMINAMATH_CALUDE_incorrect_translation_l2914_291470


namespace NUMINAMATH_CALUDE_intersection_slope_l2914_291404

/-- The slope of the line passing through the intersection points of two circles -/
theorem intersection_slope (x y : ℝ) : 
  (x^2 + y^2 - 6*x + 4*y - 20 = 0) ∧ 
  (x^2 + y^2 - 16*x + 8*y + 40 = 0) → 
  (∃ (m : ℝ), m = 5/2 ∧ ∀ (x₁ y₁ x₂ y₂ : ℝ), 
    (x₁^2 + y₁^2 - 6*x₁ + 4*y₁ - 20 = 0) ∧ 
    (x₁^2 + y₁^2 - 16*x₁ + 8*y₁ + 40 = 0) ∧ 
    (x₂^2 + y₂^2 - 6*x₂ + 4*y₂ - 20 = 0) ∧ 
    (x₂^2 + y₂^2 - 16*x₂ + 8*y₂ + 40 = 0) ∧ 
    x₁ ≠ x₂ → 
    m = (y₂ - y₁) / (x₂ - x₁)) :=
by sorry

end NUMINAMATH_CALUDE_intersection_slope_l2914_291404


namespace NUMINAMATH_CALUDE_triangle_point_distance_inequality_l2914_291493

/-- Given a triangle ABC and a point P in its plane, this theorem proves that
    the sum of the ratios of distances from P to each vertex divided by the opposite side
    is greater than or equal to the square root of 3. -/
theorem triangle_point_distance_inequality 
  (A B C P : ℝ × ℝ) -- Points in 2D plane
  (a b c : ℝ) -- Side lengths of triangle ABC
  (u v ω : ℝ) -- Distances from P to vertices
  (h_positive : a > 0 ∧ b > 0 ∧ c > 0) -- Positive side lengths
  (h_triangle : dist B C = a ∧ dist C A = b ∧ dist A B = c) -- Triangle side lengths
  (h_distances : dist P A = u ∧ dist P B = v ∧ dist P C = ω) -- Distances from P to vertices
  : u / a + v / b + ω / c ≥ Real.sqrt 3 := by
  sorry

#check triangle_point_distance_inequality

end NUMINAMATH_CALUDE_triangle_point_distance_inequality_l2914_291493


namespace NUMINAMATH_CALUDE_max_product_sum_300_l2914_291403

theorem max_product_sum_300 : 
  (∃ a b : ℤ, a + b = 300 ∧ ∀ x y : ℤ, x + y = 300 → x * y ≤ a * b) ∧ 
  (∃ a b : ℤ, a + b = 300 ∧ a * b = 22500) := by
sorry

end NUMINAMATH_CALUDE_max_product_sum_300_l2914_291403


namespace NUMINAMATH_CALUDE_immediate_boarding_probability_l2914_291408

/-- Represents the cycle time of a subway train in minutes -/
def cycletime : ℝ := 10

/-- Represents the time the train stops at the station in minutes -/
def stoptime : ℝ := 1

/-- Theorem: The probability of a passenger arriving at the platform 
    and immediately boarding the train is 1/10 -/
theorem immediate_boarding_probability : 
  stoptime / cycletime = 1 / 10 := by sorry

end NUMINAMATH_CALUDE_immediate_boarding_probability_l2914_291408


namespace NUMINAMATH_CALUDE_horner_method_operations_l2914_291419

/-- Polynomial coefficients in descending order of degree -/
def poly_coeffs : List ℤ := [5, 4, 1, 3, -81, 9, -1]

/-- Degree of the polynomial -/
def poly_degree : ℕ := poly_coeffs.length - 1

/-- Horner's method evaluation point -/
def x : ℤ := 2

/-- Number of additions in Horner's method -/
def num_additions : ℕ := poly_degree

/-- Number of multiplications in Horner's method -/
def num_multiplications : ℕ := poly_degree

theorem horner_method_operations :
  num_additions = 6 ∧ num_multiplications = 6 := by sorry

end NUMINAMATH_CALUDE_horner_method_operations_l2914_291419


namespace NUMINAMATH_CALUDE_smallest_multiple_l2914_291453

theorem smallest_multiple (n : ℕ) : n = 714 ↔ 
  (∃ k : ℕ, n = 17 * k) ∧ 
  (∃ m : ℕ, n = 101 * m + 7) ∧
  (∀ x : ℕ, x < n → ¬((∃ k : ℕ, x = 17 * k) ∧ (∃ m : ℕ, x = 101 * m + 7))) :=
by sorry

end NUMINAMATH_CALUDE_smallest_multiple_l2914_291453


namespace NUMINAMATH_CALUDE_incorrect_statement_l2914_291426

theorem incorrect_statement : 
  ¬(∀ (p q : Prop), (p ∧ q = False) → (p = False ∧ q = False)) := by
  sorry

end NUMINAMATH_CALUDE_incorrect_statement_l2914_291426


namespace NUMINAMATH_CALUDE_shaded_area_between_circles_l2914_291429

/-- The area of the shaded region between a circle circumscribing two overlapping circles
    and the two smaller circles. -/
theorem shaded_area_between_circles (r₁ r₂ d R : ℝ) (h₁ : r₁ = 4) (h₂ : r₂ = 5) (h₃ : d = 6)
    (h₄ : R = r₁ + r₂ + d) : π * R^2 - (π * r₁^2 + π * r₂^2) = 184 * π := by
  sorry

#check shaded_area_between_circles

end NUMINAMATH_CALUDE_shaded_area_between_circles_l2914_291429


namespace NUMINAMATH_CALUDE_half_vector_MN_l2914_291499

/-- Given two vectors OM and ON in ℝ², prove that half of vector MN equals (-4, 1/2) -/
theorem half_vector_MN (OM ON : ℝ × ℝ) (h1 : OM = (3, -2)) (h2 : ON = (-5, -1)) :
  (1 / 2 : ℝ) • (ON - OM) = (-4, 1/2) := by
  sorry

end NUMINAMATH_CALUDE_half_vector_MN_l2914_291499


namespace NUMINAMATH_CALUDE_teena_current_distance_l2914_291436

/-- Represents the current situation and future state of two drivers on a road -/
structure DrivingSituation where
  teena_speed : ℝ  -- Teena's speed in miles per hour
  poe_speed : ℝ    -- Poe's speed in miles per hour
  time : ℝ          -- Time in hours
  future_distance : ℝ  -- Distance Teena will be ahead of Poe after the given time

/-- Calculates the current distance between two drivers given their future state -/
def current_distance (s : DrivingSituation) : ℝ :=
  ((s.teena_speed - s.poe_speed) * s.time) - s.future_distance

/-- Theorem stating that Teena is currently 7.5 miles behind Poe -/
theorem teena_current_distance :
  let s : DrivingSituation := {
    teena_speed := 55,
    poe_speed := 40,
    time := 1.5,  -- 90 minutes = 1.5 hours
    future_distance := 15
  }
  current_distance s = 7.5 := by
  sorry

end NUMINAMATH_CALUDE_teena_current_distance_l2914_291436


namespace NUMINAMATH_CALUDE_exactly_two_pairs_exist_l2914_291472

-- Define the type for a pair of real numbers
def RealPair := ℝ × ℝ

-- Define a function to check if two lines are identical
def are_lines_identical (b c : ℝ) : Prop :=
  ∃ (k : ℝ), k ≠ 0 ∧ 
    (2 = k * c) ∧ 
    (3 * b = k * 4) ∧ 
    (c = k * 16)

-- Define the set of pairs (b, c) that make the lines identical
def identical_line_pairs : Set RealPair :=
  {p : RealPair | are_lines_identical p.1 p.2}

-- Theorem statement
theorem exactly_two_pairs_exist : 
  ∃ (p₁ p₂ : RealPair), p₁ ≠ p₂ ∧ 
    p₁ ∈ identical_line_pairs ∧ 
    p₂ ∈ identical_line_pairs ∧ 
    ∀ (p : RealPair), p ∈ identical_line_pairs → p = p₁ ∨ p = p₂ :=
  sorry

end NUMINAMATH_CALUDE_exactly_two_pairs_exist_l2914_291472


namespace NUMINAMATH_CALUDE_total_time_is_541_l2914_291487

-- Define the structure for a cupcake batch
structure CupcakeBatch where
  name : String
  bakeTime : ℕ
  iceTime : ℕ
  decorateTimePerCupcake : ℕ

-- Define the number of cupcakes per batch
def cupcakesPerBatch : ℕ := 6

-- Define the batches
def chocolateBatch : CupcakeBatch := ⟨"Chocolate", 18, 25, 10⟩
def vanillaBatch : CupcakeBatch := ⟨"Vanilla", 20, 30, 15⟩
def redVelvetBatch : CupcakeBatch := ⟨"Red Velvet", 22, 28, 12⟩
def lemonBatch : CupcakeBatch := ⟨"Lemon", 24, 32, 20⟩

-- Define the list of all batches
def allBatches : List CupcakeBatch := [chocolateBatch, vanillaBatch, redVelvetBatch, lemonBatch]

-- Calculate the total time for a single batch
def batchTotalTime (batch : CupcakeBatch) : ℕ :=
  batch.bakeTime + batch.iceTime + (batch.decorateTimePerCupcake * cupcakesPerBatch)

-- Theorem: The total time to make, ice, and decorate all cupcakes is 541 minutes
theorem total_time_is_541 : (allBatches.map batchTotalTime).sum = 541 := by
  sorry

end NUMINAMATH_CALUDE_total_time_is_541_l2914_291487


namespace NUMINAMATH_CALUDE_min_value_of_f_l2914_291443

-- Define the function f(x)
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := -x^3 + 3*x^2 + 9*x + a

-- Define the interval
def I : Set ℝ := Set.Icc (-1) 2

-- State the theorem
theorem min_value_of_f (a : ℝ) :
  (∃ x ∈ I, ∀ y ∈ I, f a y ≤ f a x) ∧ (f a 2 = 20) →
  (∃ x ∈ I, ∀ y ∈ I, f a x ≤ f a y) ∧ (f a (-1) = -7) :=
by sorry

end NUMINAMATH_CALUDE_min_value_of_f_l2914_291443


namespace NUMINAMATH_CALUDE_cuboid_probabilities_l2914_291475

/-- Represents a cuboid with given dimensions -/
structure Cuboid where
  length : ℕ
  width : ℕ
  height : ℕ

/-- Calculates the total number of unit cubes in a cuboid -/
def Cuboid.totalUnitCubes (c : Cuboid) : ℕ := c.length * c.width * c.height

/-- Calculates the number of unit cubes with no faces painted -/
def Cuboid.noPaintedFaces (c : Cuboid) : ℕ := (c.length - 2) * (c.width - 2) * (c.height - 2)

/-- Calculates the number of unit cubes with two faces painted -/
def Cuboid.twoFacesPainted (c : Cuboid) : ℕ :=
  (c.length - 2) * c.width + (c.width - 2) * c.height + (c.height - 2) * c.length

/-- Calculates the number of unit cubes with three faces painted -/
def Cuboid.threeFacesPainted (c : Cuboid) : ℕ := 8

theorem cuboid_probabilities (c : Cuboid) (h1 : c.length = 3) (h2 : c.width = 4) (h3 : c.height = 5) :
  (c.noPaintedFaces : ℚ) / c.totalUnitCubes = 1 / 10 ∧
  ((c.twoFacesPainted + c.threeFacesPainted : ℚ) / c.totalUnitCubes = 8 / 15) := by
  sorry

end NUMINAMATH_CALUDE_cuboid_probabilities_l2914_291475


namespace NUMINAMATH_CALUDE_cubic_roots_sum_cubes_l2914_291490

theorem cubic_roots_sum_cubes (a b c : ℝ) : 
  (5 * a^3 - 2019 * a + 4029 = 0) →
  (5 * b^3 - 2019 * b + 4029 = 0) →
  (5 * c^3 - 2019 * c + 4029 = 0) →
  (a + b)^3 + (b + c)^3 + (c + a)^3 = 12087/5 := by
sorry

end NUMINAMATH_CALUDE_cubic_roots_sum_cubes_l2914_291490


namespace NUMINAMATH_CALUDE_complement_of_union_l2914_291466

open Set

def U : Finset ℕ := {1, 2, 3, 4}
def A : Finset ℕ := {1, 2}
def B : Finset ℕ := {2, 3}

theorem complement_of_union : (U \ (A ∪ B)) = {4} := by
  sorry

end NUMINAMATH_CALUDE_complement_of_union_l2914_291466


namespace NUMINAMATH_CALUDE_train_speed_calculation_l2914_291471

/-- Given a train that crosses a pole in a certain time, calculate its speed in kmph. -/
theorem train_speed_calculation (train_length : Real) (crossing_time : Real) :
  train_length = 800.064 →
  crossing_time = 18 →
  (train_length / 1000) / (crossing_time / 3600) = 160.0128 := by
  sorry

#check train_speed_calculation

end NUMINAMATH_CALUDE_train_speed_calculation_l2914_291471


namespace NUMINAMATH_CALUDE_min_value_implies_a_l2914_291486

def f (a : ℝ) (x : ℝ) : ℝ := x^2 - 2*a*x + a - 1

theorem min_value_implies_a (a : ℝ) : 
  (∀ x ∈ Set.Icc 0 1, f a x ≥ -2) ∧ 
  (∃ x ∈ Set.Icc 0 1, f a x = -2) → 
  a = -1 ∨ a = 2 := by sorry

end NUMINAMATH_CALUDE_min_value_implies_a_l2914_291486


namespace NUMINAMATH_CALUDE_gcd_888_1147_l2914_291410

theorem gcd_888_1147 : Nat.gcd 888 1147 = 37 := by
  sorry

end NUMINAMATH_CALUDE_gcd_888_1147_l2914_291410


namespace NUMINAMATH_CALUDE_two_blue_marbles_probability_l2914_291463

def total_marbles : ℕ := 3 + 4 + 9

def blue_marbles : ℕ := 4

def probability_two_blue_marbles : ℚ :=
  (blue_marbles / total_marbles) * ((blue_marbles - 1) / (total_marbles - 1))

theorem two_blue_marbles_probability :
  probability_two_blue_marbles = 1 / 20 :=
sorry

end NUMINAMATH_CALUDE_two_blue_marbles_probability_l2914_291463


namespace NUMINAMATH_CALUDE_deepaks_share_of_profit_l2914_291425

def anands_investment : ℕ := 22500
def deepaks_investment : ℕ := 35000
def total_profit : ℕ := 13800

theorem deepaks_share_of_profit :
  (deepaks_investment * total_profit) / (anands_investment + deepaks_investment) = 8400 :=
by sorry

end NUMINAMATH_CALUDE_deepaks_share_of_profit_l2914_291425


namespace NUMINAMATH_CALUDE_batsman_matches_l2914_291441

theorem batsman_matches (total_matches : ℕ) (last_matches : ℕ) (last_avg : ℚ) (overall_avg : ℚ) :
  total_matches = 35 →
  last_matches = 13 →
  last_avg = 15 →
  overall_avg = 23.17142857142857 →
  total_matches - last_matches = 22 :=
by sorry

end NUMINAMATH_CALUDE_batsman_matches_l2914_291441


namespace NUMINAMATH_CALUDE_ramanujan_number_l2914_291462

theorem ramanujan_number (h r : ℂ) : 
  h * r = 40 + 24 * I ∧ h = 3 + 7 * I → r = 4 - (104 / 29) * I :=
by sorry

end NUMINAMATH_CALUDE_ramanujan_number_l2914_291462


namespace NUMINAMATH_CALUDE_sqrt_equation_solution_l2914_291438

theorem sqrt_equation_solution (w : ℝ) :
  (Real.sqrt 1.1 / Real.sqrt 0.81 + Real.sqrt 1.44 / Real.sqrt w = 2.879628878919216) →
  w = 0.49 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_equation_solution_l2914_291438


namespace NUMINAMATH_CALUDE_simplify_expression_factorize_expression_l2914_291405

-- Problem 1
theorem simplify_expression (x : ℝ) : (-2*x)^2 + 3*x*x = 7*x^2 := by
  sorry

-- Problem 2
theorem factorize_expression (m a b : ℝ) : m*a^2 - m*b^2 = m*(a - b)*(a + b) := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_factorize_expression_l2914_291405


namespace NUMINAMATH_CALUDE_algebraic_expression_value_l2914_291484

theorem algebraic_expression_value (x y : ℝ) (h : x - y - 7 = 0) :
  x^2 - y^2 - 14*y = 49 := by
  sorry

end NUMINAMATH_CALUDE_algebraic_expression_value_l2914_291484


namespace NUMINAMATH_CALUDE_even_function_implies_f_3_equals_5_l2914_291448

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := (x + 2) * (x - a)

-- State the theorem
theorem even_function_implies_f_3_equals_5 :
  (∀ x : ℝ, f a x = f a (-x)) → f a 3 = 5 := by
  sorry

end NUMINAMATH_CALUDE_even_function_implies_f_3_equals_5_l2914_291448


namespace NUMINAMATH_CALUDE_shopping_solution_l2914_291422

/-- Represents the shopping problem with given prices, discounts, and taxes -/
def shopping_problem (initial_amount : ℝ) 
  (milk_price bread_price detergent_price banana_price_per_pound egg_price chicken_price apple_price : ℝ)
  (detergent_discount chicken_discount loyalty_discount milk_discount bread_discount : ℝ)
  (sales_tax : ℝ) : Prop :=
  let discounted_milk := milk_price * (1 - milk_discount)
  let discounted_bread := bread_price * (1 + 0.5) -- Buy one get one 50% off
  let discounted_detergent := detergent_price - detergent_discount
  let banana_total := banana_price_per_pound * 3
  let discounted_chicken := chicken_price * (1 - chicken_discount)
  let subtotal := discounted_milk + discounted_bread + discounted_detergent + banana_total + 
                  egg_price + discounted_chicken + apple_price
  let loyalty_discounted := subtotal * (1 - loyalty_discount)
  let total_with_tax := loyalty_discounted * (1 + sales_tax)
  initial_amount - total_with_tax = 38.25

/-- Theorem stating the solution to the shopping problem -/
theorem shopping_solution : 
  shopping_problem 75 3.80 4.25 11.50 0.95 2.80 8.45 6.30 2 0.20 0.10 0.15 0.50 0.08 := by
  sorry

end NUMINAMATH_CALUDE_shopping_solution_l2914_291422

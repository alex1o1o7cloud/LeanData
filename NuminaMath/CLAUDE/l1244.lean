import Mathlib

namespace NUMINAMATH_CALUDE_f_properties_l1244_124430

noncomputable def f (x : ℝ) : ℝ := (1 - Real.sin (2 * x)) / (Real.sin x - Real.cos x)

theorem f_properties :
  ∃ (T : ℝ) (max_value : ℝ) (max_set : Set ℝ),
    (∀ x, f (x + T) = f x) ∧  -- f has period T
    T = 2 * Real.pi ∧  -- The period is 2π
    (∀ x, f x ≤ max_value) ∧  -- max_value is an upper bound
    max_value = Real.sqrt 2 ∧  -- The maximum value is √2
    (∀ x, x ∈ max_set ↔ f x = max_value) ∧  -- max_set contains all x where f(x) is maximum
    (∀ k : ℤ, (2 * k : ℝ) * Real.pi + 3 * Real.pi / 4 ∈ max_set)  -- Characterization of max_set
    := by sorry

end NUMINAMATH_CALUDE_f_properties_l1244_124430


namespace NUMINAMATH_CALUDE_expand_and_simplify_expression_l1244_124487

theorem expand_and_simplify_expression (x : ℝ) : 
  6 * (x - 7) * (2 * x + 15) + (3 * x - 4) * (x + 5) = 15 * x^2 + 17 * x - 650 := by
  sorry

end NUMINAMATH_CALUDE_expand_and_simplify_expression_l1244_124487


namespace NUMINAMATH_CALUDE_tomato_planting_theorem_l1244_124473

def tomato_planting (total_seedlings : ℕ) (remi_day1 : ℕ) : Prop :=
  let remi_day2 := 2 * remi_day1
  let father_day3 := 3 * remi_day2
  let father_day4 := 4 * remi_day2
  let sister_day5 := remi_day1
  let sister_day6 := 5 * remi_day1
  let remi_total := remi_day1 + remi_day2
  let sister_total := sister_day5 + sister_day6
  let father_total := total_seedlings - remi_total - sister_total
  (remi_total = 600) ∧
  (sister_total = 1200) ∧
  (father_total = 6400) ∧
  (remi_total + sister_total + father_total = total_seedlings)

theorem tomato_planting_theorem :
  tomato_planting 8200 200 :=
by
  sorry

end NUMINAMATH_CALUDE_tomato_planting_theorem_l1244_124473


namespace NUMINAMATH_CALUDE_unique_number_solution_l1244_124411

def is_valid_number (a b c : ℕ) : Prop :=
  a < 10 ∧ b < 10 ∧ c < 10 ∧
  a + b + c = 10 ∧
  b = a + c ∧
  100 * c + 10 * b + a = 100 * a + 10 * b + c + 99

theorem unique_number_solution :
  ∃! (a b c : ℕ), is_valid_number a b c ∧ 100 * a + 10 * b + c = 203 :=
sorry

end NUMINAMATH_CALUDE_unique_number_solution_l1244_124411


namespace NUMINAMATH_CALUDE_ellipse_hyperbola_same_foci_l1244_124468

theorem ellipse_hyperbola_same_foci (k : ℝ) : k > 0 →
  (∀ x y : ℝ, x^2/9 + y^2/k^2 = 1 ↔ x^2/k - y^2/3 = 1) →
  k = 2 :=
by sorry

end NUMINAMATH_CALUDE_ellipse_hyperbola_same_foci_l1244_124468


namespace NUMINAMATH_CALUDE_joes_total_lift_weight_l1244_124410

/-- The total weight of Joe's two lifts is 1500 pounds, given the conditions of the weight-lifting competition. -/
theorem joes_total_lift_weight :
  let first_lift : ℕ := 600
  let second_lift : ℕ := 2 * first_lift - 300
  first_lift + second_lift = 1500 := by
  sorry

end NUMINAMATH_CALUDE_joes_total_lift_weight_l1244_124410


namespace NUMINAMATH_CALUDE_student_count_l1244_124433

theorem student_count (rank_from_right rank_from_left : ℕ) 
  (h1 : rank_from_right = 13) 
  (h2 : rank_from_left = 8) : 
  rank_from_right + rank_from_left - 1 = 20 := by
  sorry

end NUMINAMATH_CALUDE_student_count_l1244_124433


namespace NUMINAMATH_CALUDE_soap_box_theorem_l1244_124405

/-- The number of bars of soap in each box of bars -/
def bars_per_box : ℕ := 5

/-- The smallest number of each type of soap sold -/
def min_sold : ℕ := 95

/-- The number of bottles of soap in each box of bottles -/
def bottles_per_box : ℕ := 19

theorem soap_box_theorem :
  ∃ (bar_boxes bottle_boxes : ℕ),
    bar_boxes * bars_per_box = bottle_boxes * bottles_per_box ∧
    bar_boxes * bars_per_box = min_sold ∧
    bottle_boxes * bottles_per_box = min_sold ∧
    bottles_per_box > 1 ∧
    bottles_per_box < min_sold :=
by sorry

end NUMINAMATH_CALUDE_soap_box_theorem_l1244_124405


namespace NUMINAMATH_CALUDE_fractional_equation_solution_l1244_124477

theorem fractional_equation_solution :
  ∃ (x : ℝ), x ≠ 1 ∧ (3 / (x - 1) = 5 + 3 * x / (1 - x)) ↔ x = 4 := by
  sorry

end NUMINAMATH_CALUDE_fractional_equation_solution_l1244_124477


namespace NUMINAMATH_CALUDE_survey_result_l1244_124419

/-- Calculates the percentage of the surveyed population that supports a new environmental policy. -/
def survey_support_percentage (men_support_rate : ℚ) (women_support_rate : ℚ) (men_count : ℕ) (women_count : ℕ) : ℚ :=
  let total_count := men_count + women_count
  let supporting_count := men_support_rate * men_count + women_support_rate * women_count
  supporting_count / total_count

/-- Theorem stating that given the survey conditions, 74% of the population supports the policy. -/
theorem survey_result :
  let men_support_rate : ℚ := 70 / 100
  let women_support_rate : ℚ := 75 / 100
  let men_count : ℕ := 200
  let women_count : ℕ := 800
  survey_support_percentage men_support_rate women_support_rate men_count women_count = 74 / 100 := by
  sorry

#eval survey_support_percentage (70 / 100) (75 / 100) 200 800

end NUMINAMATH_CALUDE_survey_result_l1244_124419


namespace NUMINAMATH_CALUDE_reciprocal_of_2022_l1244_124406

theorem reciprocal_of_2022 : (2022⁻¹ : ℚ) = 1 / 2022 := by
  sorry

end NUMINAMATH_CALUDE_reciprocal_of_2022_l1244_124406


namespace NUMINAMATH_CALUDE_quadratic_function_unique_l1244_124422

/-- A quadratic function with specific properties -/
structure QuadraticFunction where
  f : ℝ → ℝ
  is_quadratic : ∃ a b c : ℝ, ∀ x, f x = a * x^2 + b * x + c
  equal_roots : ∃ r : ℝ, (∀ x, f x = 0 ↔ x = r)
  derivative : ∀ x, HasDerivAt f (2 * x + 2) x

/-- The main theorem: if f satisfies the given conditions, then f(x) = x^2 + 2x + 1 -/
theorem quadratic_function_unique (qf : QuadraticFunction) :
  ∀ x, qf.f x = x^2 + 2*x + 1 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_function_unique_l1244_124422


namespace NUMINAMATH_CALUDE_vector_computation_l1244_124407

theorem vector_computation :
  let v1 : Fin 3 → ℝ := ![3, -2, 5]
  let v2 : Fin 3 → ℝ := ![2, -3, 4]
  4 • v1 - 3 • v2 = ![6, 1, 8] :=
by
  sorry

end NUMINAMATH_CALUDE_vector_computation_l1244_124407


namespace NUMINAMATH_CALUDE_stones_for_hall_l1244_124457

-- Define the hall dimensions in decimeters
def hall_length : ℕ := 360
def hall_width : ℕ := 150

-- Define the stone dimensions in decimeters
def stone_length : ℕ := 3
def stone_width : ℕ := 5

-- Define the function to calculate the number of stones required
def stones_required (hall_l hall_w stone_l stone_w : ℕ) : ℕ :=
  (hall_l * hall_w) / (stone_l * stone_w)

-- Theorem statement
theorem stones_for_hall :
  stones_required hall_length hall_width stone_length stone_width = 3600 := by
  sorry

end NUMINAMATH_CALUDE_stones_for_hall_l1244_124457


namespace NUMINAMATH_CALUDE_students_in_all_three_sections_l1244_124420

/-- Represents the number of students in each section and their intersections -/
structure ClubSections where
  totalStudents : ℕ
  music : ℕ
  drama : ℕ
  dance : ℕ
  atLeastTwo : ℕ
  allThree : ℕ

/-- The theorem stating the number of students in all three sections -/
theorem students_in_all_three_sections 
  (club : ClubSections) 
  (h1 : club.totalStudents = 30)
  (h2 : club.music = 15)
  (h3 : club.drama = 18)
  (h4 : club.dance = 12)
  (h5 : club.atLeastTwo = 14)
  (h6 : ∀ s : ℕ, s ≤ club.totalStudents → s ≥ club.music ∨ s ≥ club.drama ∨ s ≥ club.dance) :
  club.allThree = 6 := by
  sorry


end NUMINAMATH_CALUDE_students_in_all_three_sections_l1244_124420


namespace NUMINAMATH_CALUDE_dorothy_profit_l1244_124432

/-- Given the cost of ingredients, number of doughnuts made, and selling price per doughnut,
    calculate the profit. -/
def calculate_profit (ingredient_cost : ℕ) (num_doughnuts : ℕ) (price_per_doughnut : ℕ) : ℕ :=
  num_doughnuts * price_per_doughnut - ingredient_cost

/-- Theorem stating that Dorothy's profit is $22 given the problem conditions. -/
theorem dorothy_profit :
  calculate_profit 53 25 3 = 22 := by
  sorry

end NUMINAMATH_CALUDE_dorothy_profit_l1244_124432


namespace NUMINAMATH_CALUDE_widgets_per_carton_is_three_l1244_124486

/-- Represents the dimensions of a box -/
structure BoxDimensions where
  width : ℕ
  length : ℕ
  height : ℕ

/-- Calculate the number of widgets per carton -/
def widgets_per_carton (carton : BoxDimensions) (shipping_box : BoxDimensions) (total_widgets : ℕ) : ℕ :=
  let cartons_per_layer := (shipping_box.width / carton.width) * (shipping_box.length / carton.length)
  let layers := shipping_box.height / carton.height
  let total_cartons := cartons_per_layer * layers
  total_widgets / total_cartons

/-- Theorem: Given the specified dimensions and total widgets, there are 3 widgets per carton -/
theorem widgets_per_carton_is_three :
  let carton := BoxDimensions.mk 4 4 5
  let shipping_box := BoxDimensions.mk 20 20 20
  let total_widgets := 300
  widgets_per_carton carton shipping_box total_widgets = 3 := by
  sorry


end NUMINAMATH_CALUDE_widgets_per_carton_is_three_l1244_124486


namespace NUMINAMATH_CALUDE_inequality_proof_l1244_124426

theorem inequality_proof (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  (a + b) * (a * b + 1) * (b + 1) ≥ (27 / 4) * a * b^2 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l1244_124426


namespace NUMINAMATH_CALUDE_buttons_solution_l1244_124415

def buttons_problem (mari kendra sue will lea : ℕ) : Prop :=
  mari = 8 ∧
  kendra = 5 * mari + 4 ∧
  sue = kendra / 2 ∧
  will = (5 * (kendra + sue)) / 2 ∧
  lea = will - will / 5

theorem buttons_solution :
  ∃ (mari kendra sue will lea : ℕ),
    buttons_problem mari kendra sue will lea ∧ lea = 132 := by
  sorry

end NUMINAMATH_CALUDE_buttons_solution_l1244_124415


namespace NUMINAMATH_CALUDE_temperature_matches_data_temperature_decreases_with_altitude_constant_temperature_change_rate_l1244_124478

-- Define the relationship between altitude and temperature
def temperature (h : ℝ) : ℝ := 20 - 6 * h

-- Define the set of data points from the table
def data_points : List (ℝ × ℝ) := [
  (0, 20), (1, 14), (2, 8), (3, 2), (4, -4), (5, -10)
]

-- Theorem stating that the temperature function matches the data points
theorem temperature_matches_data : ∀ (point : ℝ × ℝ), 
  point ∈ data_points → temperature point.1 = point.2 := by
  sorry

-- Theorem stating that the temperature decreases as altitude increases
theorem temperature_decreases_with_altitude : 
  ∀ (h1 h2 : ℝ), h1 < h2 → temperature h1 > temperature h2 := by
  sorry

-- Theorem stating that the rate of temperature change is constant
theorem constant_temperature_change_rate : 
  ∀ (h1 h2 : ℝ), h1 ≠ h2 → (temperature h2 - temperature h1) / (h2 - h1) = -6 := by
  sorry

end NUMINAMATH_CALUDE_temperature_matches_data_temperature_decreases_with_altitude_constant_temperature_change_rate_l1244_124478


namespace NUMINAMATH_CALUDE_principal_exists_l1244_124439

/-- The principal amount that satisfies the given conditions -/
def find_principal : ℝ → Prop := fun P =>
  let first_year_rate : ℝ := 0.10
  let second_year_rate : ℝ := 0.12
  let semi_annual_rate1 : ℝ := first_year_rate / 2
  let semi_annual_rate2 : ℝ := second_year_rate / 2
  let compound_factor : ℝ := (1 + semi_annual_rate1)^2 * (1 + semi_annual_rate2)^2
  let simple_interest_factor : ℝ := first_year_rate + second_year_rate
  P * (compound_factor - 1 - simple_interest_factor) = 15

/-- Theorem stating the existence of a principal amount satisfying the given conditions -/
theorem principal_exists : ∃ P : ℝ, find_principal P := by
  sorry

end NUMINAMATH_CALUDE_principal_exists_l1244_124439


namespace NUMINAMATH_CALUDE_range_of_a_l1244_124408

theorem range_of_a (a : ℝ) : (2 * a - 1) ^ 0 = 1 → a ≠ 1/2 :=
by sorry

end NUMINAMATH_CALUDE_range_of_a_l1244_124408


namespace NUMINAMATH_CALUDE_mike_weekly_pullups_l1244_124497

/-- Calculates the number of pull-ups Mike does in a week -/
def weekly_pullups (pullups_per_entry : ℕ) (entries_per_day : ℕ) (days_per_week : ℕ) : ℕ :=
  pullups_per_entry * entries_per_day * days_per_week

/-- Proves that Mike does 70 pull-ups in a week -/
theorem mike_weekly_pullups :
  weekly_pullups 2 5 7 = 70 := by
  sorry

end NUMINAMATH_CALUDE_mike_weekly_pullups_l1244_124497


namespace NUMINAMATH_CALUDE_plan1_more_cost_effective_l1244_124471

/-- Represents the cost of a fitness club plan as a function of the number of sessions -/
def PlanCost := ℕ → ℝ

/-- Cost function for Plan 1: 80 yuan fixed fee + 10 yuan per session -/
def plan1 : PlanCost := λ x => 10 * x + 80

/-- Cost function for Plan 2: 20 yuan per session, no fixed fee -/
def plan2 : PlanCost := λ x => 20 * x

/-- Theorem stating that Plan 1 is more cost-effective when the number of sessions is greater than 8 -/
theorem plan1_more_cost_effective (x : ℕ) : 
  x > 8 → plan1 x < plan2 x := by
  sorry

#check plan1_more_cost_effective

end NUMINAMATH_CALUDE_plan1_more_cost_effective_l1244_124471


namespace NUMINAMATH_CALUDE_special_rectangle_AB_length_l1244_124459

/-- Represents a rectangle with specific properties -/
structure SpecialRectangle where
  AB : ℝ
  BC : ℝ
  PQ : ℝ
  XY : ℝ
  equalAreas : Bool
  PQparallelAB : Bool
  XYequation : Bool

/-- The theorem statement -/
theorem special_rectangle_AB_length
  (rect : SpecialRectangle)
  (h1 : rect.BC = 19)
  (h2 : rect.PQ = 87)
  (h3 : rect.equalAreas)
  (h4 : rect.PQparallelAB)
  (h5 : rect.XYequation) :
  rect.AB = 193 := by
  sorry

end NUMINAMATH_CALUDE_special_rectangle_AB_length_l1244_124459


namespace NUMINAMATH_CALUDE_tangent_slope_parabola_l1244_124485

/-- The slope of the tangent line to y = (1/5)x^2 at x = 2 is 4/5 -/
theorem tangent_slope_parabola :
  let f : ℝ → ℝ := fun x ↦ (1/5) * x^2
  let a : ℝ := 2
  let slope : ℝ := (deriv f) a
  slope = 4/5 := by sorry

end NUMINAMATH_CALUDE_tangent_slope_parabola_l1244_124485


namespace NUMINAMATH_CALUDE_chairs_moved_by_alex_l1244_124447

/-- Given that Carey moves x chairs, Pat moves y chairs, and Alex moves z chairs,
    with a total of 74 chairs to be moved, prove that the number of chairs
    Alex moves is equal to 74 minus the sum of chairs moved by Carey and Pat. -/
theorem chairs_moved_by_alex (x y z : ℕ) (h : x + y + z = 74) :
  z = 74 - x - y := by sorry

end NUMINAMATH_CALUDE_chairs_moved_by_alex_l1244_124447


namespace NUMINAMATH_CALUDE_circle_construction_theorem_circle_line_construction_theorem_l1244_124424

-- Define a point in 2D space
structure Point where
  x : ℝ
  y : ℝ

-- Define a circle
structure Circle where
  center : Point
  radius : ℝ

-- Define a line
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ  -- ax + by + c = 0

-- Define tangency between circles
def CircleTangent (c1 c2 : Circle) : Prop := sorry

-- Define tangency between a circle and a line
def CircleLineTangent (c : Circle) (l : Line) : Prop := sorry

-- Define a circle passing through a point
def CirclePassesThrough (c : Circle) (p : Point) : Prop := sorry

theorem circle_construction_theorem 
  (P : Point) 
  (S1 S2 : Circle) : 
  ∃ (C : Circle), 
    CirclePassesThrough C P ∧ 
    CircleTangent C S1 ∧ 
    CircleTangent C S2 := by sorry

theorem circle_line_construction_theorem 
  (P : Point) 
  (S : Circle) 
  (L : Line) : 
  ∃ (C : Circle), 
    CirclePassesThrough C P ∧ 
    CircleTangent C S ∧ 
    CircleLineTangent C L := by sorry

end NUMINAMATH_CALUDE_circle_construction_theorem_circle_line_construction_theorem_l1244_124424


namespace NUMINAMATH_CALUDE_base_n_problem_l1244_124467

theorem base_n_problem (n d : ℕ) : 
  n > 0 → 
  d < 10 → 
  3 * n^2 + 2 * n + d = 263 → 
  3 * n^2 + 2 * n + 4 = 396 + 7 * d → 
  n + d = 11 := by sorry

end NUMINAMATH_CALUDE_base_n_problem_l1244_124467


namespace NUMINAMATH_CALUDE_february_to_january_sales_ratio_l1244_124449

/-- The ratio of window screens sold in February to January is 2:3 -/
theorem february_to_january_sales_ratio :
  ∀ (january february march : ℕ),
  february = march / 4 →
  march = 8800 →
  january + february + march = 12100 →
  (february : ℚ) / january = 2 / 3 := by
sorry

end NUMINAMATH_CALUDE_february_to_january_sales_ratio_l1244_124449


namespace NUMINAMATH_CALUDE_egg_price_calculation_l1244_124456

/-- The price of a dozen eggs given the number of chickens, eggs laid per day, and total earnings --/
def price_per_dozen (num_chickens : ℕ) (eggs_per_chicken_per_day : ℕ) (total_earnings : ℕ) : ℚ :=
  let total_days : ℕ := 28  -- 4 weeks
  let total_eggs : ℕ := num_chickens * eggs_per_chicken_per_day * total_days
  let total_dozens : ℕ := total_eggs / 12
  (total_earnings : ℚ) / total_dozens

theorem egg_price_calculation :
  price_per_dozen 8 3 280 = 5 := by
  sorry

end NUMINAMATH_CALUDE_egg_price_calculation_l1244_124456


namespace NUMINAMATH_CALUDE_actual_distance_traveled_l1244_124461

/-- Given a person walking at 4 km/hr, if increasing their speed to 5 km/hr
    would result in walking 6 km more in the same time, then the actual
    distance traveled is 24 km. -/
theorem actual_distance_traveled (actual_speed actual_distance : ℝ) 
    (h1 : actual_speed = 4)
    (h2 : actual_distance / actual_speed = (actual_distance + 6) / 5) :
  actual_distance = 24 := by
  sorry

end NUMINAMATH_CALUDE_actual_distance_traveled_l1244_124461


namespace NUMINAMATH_CALUDE_complex_magnitude_constraint_l1244_124470

theorem complex_magnitude_constraint (a : ℝ) :
  let z : ℂ := 1 + a * I
  (Complex.abs z < 2) → (-Real.sqrt 3 < a ∧ a < Real.sqrt 3) := by
sorry

end NUMINAMATH_CALUDE_complex_magnitude_constraint_l1244_124470


namespace NUMINAMATH_CALUDE_phone_call_cost_per_minute_l1244_124463

/-- Proves that the cost per minute of each phone call is $0.05 given the specified conditions --/
theorem phone_call_cost_per_minute 
  (call_duration : ℝ) 
  (customers_per_week : ℕ) 
  (monthly_bill : ℝ) 
  (weeks_per_month : ℕ) : 
  call_duration = 1 →
  customers_per_week = 50 →
  monthly_bill = 600 →
  weeks_per_month = 4 →
  (monthly_bill / (customers_per_week * weeks_per_month * call_duration * 60)) = 0.05 := by
  sorry

end NUMINAMATH_CALUDE_phone_call_cost_per_minute_l1244_124463


namespace NUMINAMATH_CALUDE_three_topping_pizzas_l1244_124475

theorem three_topping_pizzas (n : ℕ) (k : ℕ) : n = 7 → k = 3 → Nat.choose n k = 35 := by
  sorry

end NUMINAMATH_CALUDE_three_topping_pizzas_l1244_124475


namespace NUMINAMATH_CALUDE_power_sum_equals_fourteen_l1244_124464

theorem power_sum_equals_fourteen : 2^3 + 2^2 + 2^1 = 14 := by
  sorry

end NUMINAMATH_CALUDE_power_sum_equals_fourteen_l1244_124464


namespace NUMINAMATH_CALUDE_sin_270_degrees_l1244_124441

theorem sin_270_degrees : Real.sin (270 * π / 180) = -1 := by
  sorry

end NUMINAMATH_CALUDE_sin_270_degrees_l1244_124441


namespace NUMINAMATH_CALUDE_derivative_cos_squared_at_pi_eighth_l1244_124414

/-- Given a function f(x) = cos²(2x), its derivative at π/8 is -2. -/
theorem derivative_cos_squared_at_pi_eighth (f : ℝ → ℝ) (h : ∀ x, f x = Real.cos (2 * x) ^ 2) :
  deriv f (π / 8) = -2 := by
  sorry

end NUMINAMATH_CALUDE_derivative_cos_squared_at_pi_eighth_l1244_124414


namespace NUMINAMATH_CALUDE_luncheon_cost_theorem_l1244_124446

/-- The cost of a luncheon item combination -/
structure LuncheonCost where
  sandwiches : ℕ
  coffees : ℕ
  pies : ℕ
  total : ℚ

/-- The given luncheon costs -/
def givenLuncheons : List LuncheonCost := [
  ⟨5, 9, 2, 595/100⟩,
  ⟨7, 12, 2, 790/100⟩,
  ⟨3, 5, 1, 350/100⟩
]

/-- The theorem to prove -/
theorem luncheon_cost_theorem (s c p : ℚ) 
  (h1 : 5*s + 9*c + 2*p = 595/100)
  (h2 : 7*s + 12*c + 2*p = 790/100)
  (h3 : 3*s + 5*c + p = 350/100) :
  s + c + p = 105/100 := by
  sorry

end NUMINAMATH_CALUDE_luncheon_cost_theorem_l1244_124446


namespace NUMINAMATH_CALUDE_equation_solution_l1244_124427

theorem equation_solution (x : ℝ) (hx : x ≠ 0) :
  (5 * x)^10 = (10 * x)^5 ↔ x = 2/5 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l1244_124427


namespace NUMINAMATH_CALUDE_quadratic_propositions_l1244_124425

-- Define the quadratic equation
def quadratic (a b c x : ℝ) : Prop := a * x^2 + b * x + c = 0

-- Define the discriminant
def discriminant (a b c : ℝ) : ℝ := b^2 - 4*a*c

theorem quadratic_propositions (a b c : ℝ) (ha : a ≠ 0) :
  -- Proposition 1
  (a + b + c = 0 → discriminant a b c ≥ 0) ∧
  -- Proposition 2
  (∃ x y : ℝ, x = -1 ∧ y = 2 ∧ quadratic a b c x ∧ quadratic a b c y → 2*a + c = 0) ∧
  -- Proposition 3
  ((∃ x y : ℝ, x ≠ y ∧ quadratic a 0 c x ∧ quadratic a 0 c y) →
   ∃ u v : ℝ, u ≠ v ∧ quadratic a b c u ∧ quadratic a b c v) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_propositions_l1244_124425


namespace NUMINAMATH_CALUDE_quadratic_intersection_l1244_124496

/-- A quadratic function of the form y = ax² - 4x + 2 -/
def quadratic_function (a : ℝ) (x : ℝ) : ℝ := a * x^2 - 4 * x + 2

/-- The discriminant of the quadratic function -/
def discriminant (a : ℝ) : ℝ := 16 - 8 * a

theorem quadratic_intersection (a : ℝ) :
  (∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ quadratic_function a x₁ = 0 ∧ quadratic_function a x₂ = 0) →
  (0 < a ∧ a < 2) :=
sorry

end NUMINAMATH_CALUDE_quadratic_intersection_l1244_124496


namespace NUMINAMATH_CALUDE_odd_operations_l1244_124434

theorem odd_operations (a b : ℤ) (ha : Odd a) (hb : Odd b) :
  Odd (a * b) ∧ Odd (a ^ 2) ∧ ¬(∀ x y : ℤ, Odd x → Odd y → Odd (x + y)) ∧ ¬(∀ x y : ℤ, Odd x → Odd y → Odd (x - y)) :=
by sorry

end NUMINAMATH_CALUDE_odd_operations_l1244_124434


namespace NUMINAMATH_CALUDE_terry_commute_time_l1244_124431

/-- Calculates the total daily driving time for Terry's commute -/
theorem terry_commute_time : 
  let segment1_distance : ℝ := 15
  let segment1_speed : ℝ := 30
  let segment2_distance : ℝ := 35
  let segment2_speed : ℝ := 50
  let segment3_distance : ℝ := 10
  let segment3_speed : ℝ := 40
  let total_time := 
    (segment1_distance / segment1_speed + 
     segment2_distance / segment2_speed + 
     segment3_distance / segment3_speed) * 2
  total_time = 2.9 := by sorry

end NUMINAMATH_CALUDE_terry_commute_time_l1244_124431


namespace NUMINAMATH_CALUDE_stack_height_three_pipes_l1244_124436

/-- The height of a stack of identical cylindrical pipes -/
def stack_height (num_pipes : ℕ) (pipe_diameter : ℝ) : ℝ :=
  (num_pipes : ℝ) * pipe_diameter

/-- Theorem: The height of a stack of three identical cylindrical pipes with a diameter of 12 cm is 36 cm -/
theorem stack_height_three_pipes :
  stack_height 3 12 = 36 := by
  sorry

end NUMINAMATH_CALUDE_stack_height_three_pipes_l1244_124436


namespace NUMINAMATH_CALUDE_ellipse_foci_distance_l1244_124412

-- Define the ellipse
structure Ellipse where
  center : ℝ × ℝ
  a : ℝ
  b : ℝ

-- Define the conditions of the problem
def problem_ellipse : Ellipse :=
  { center := (5, 2)
  , a := 5
  , b := 2 }

-- Define the point that the ellipse passes through
def point_on_ellipse : ℝ × ℝ := (3, 1)

-- Theorem statement
theorem ellipse_foci_distance :
  let e := problem_ellipse
  let (x, y) := point_on_ellipse
  let (cx, cy) := e.center
  (((x - cx) / e.a) ^ 2 + ((y - cy) / e.b) ^ 2 ≤ 1) →
  (2 * Real.sqrt (e.a ^ 2 - e.b ^ 2) = 2 * Real.sqrt 21) :=
by sorry

end NUMINAMATH_CALUDE_ellipse_foci_distance_l1244_124412


namespace NUMINAMATH_CALUDE_min_distance_to_circle_l1244_124466

theorem min_distance_to_circle (x y : ℝ) :
  x^2 + y^2 - 4*x + 6*y + 4 = 0 →
  ∃ (min : ℝ), min = Real.sqrt 13 - 3 ∧
    ∀ (a b : ℝ), a^2 + b^2 - 4*a + 6*b + 4 = 0 →
      Real.sqrt (a^2 + b^2) ≥ min :=
by sorry

end NUMINAMATH_CALUDE_min_distance_to_circle_l1244_124466


namespace NUMINAMATH_CALUDE_arithmetic_progression_sine_squared_l1244_124448

theorem arithmetic_progression_sine_squared (x y z α : Real) : 
  (y = (x + z) / 2) →  -- x, y, z form an arithmetic progression
  (α = Real.arcsin (Real.sqrt 7 / 4)) →  -- α is defined as arcsin(√7/4)
  (8 / Real.sin y = 1 / Real.sin x + 1 / Real.sin z) →  -- 1/sin(x), 4/sin(y), 1/sin(z) form an arithmetic progression
  Real.sin y ^ 2 = 7 / 13 := by
sorry

end NUMINAMATH_CALUDE_arithmetic_progression_sine_squared_l1244_124448


namespace NUMINAMATH_CALUDE_tangent_sum_simplification_l1244_124404

theorem tangent_sum_simplification :
  let x := Real.pi / 9  -- 20°
  let y := Real.pi / 6  -- 30°
  let z := Real.pi / 3  -- 60°
  let w := 4 * Real.pi / 9  -- 80°
  (Real.tan x + Real.tan y + Real.tan z + Real.tan w) / Real.cos (Real.pi / 18) =
    2 / (Real.sqrt 3 * Real.sin (7 * Real.pi / 18) * (Real.sin (Real.pi / 18))^2) :=
by sorry

end NUMINAMATH_CALUDE_tangent_sum_simplification_l1244_124404


namespace NUMINAMATH_CALUDE_expenditure_recording_l1244_124429

/-- Represents the way a transaction is recorded in accounting -/
inductive AccountingRecord
  | Positive (amount : ℤ)
  | Negative (amount : ℤ)

/-- Records an income transaction -/
def recordIncome (amount : ℤ) : AccountingRecord :=
  AccountingRecord.Positive amount

/-- Records an expenditure transaction -/
def recordExpenditure (amount : ℤ) : AccountingRecord :=
  AccountingRecord.Negative amount

/-- The accounting principle for recording transactions -/
axiom accounting_principle (amount : ℤ) :
  (recordIncome amount = AccountingRecord.Positive amount) ∧
  (recordExpenditure amount = AccountingRecord.Negative amount)

/-- Theorem: An expenditure of 100 should be recorded as -100 -/
theorem expenditure_recording :
  recordExpenditure 100 = AccountingRecord.Negative 100 := by
  sorry

end NUMINAMATH_CALUDE_expenditure_recording_l1244_124429


namespace NUMINAMATH_CALUDE_other_solution_of_quadratic_equation_l1244_124488

theorem other_solution_of_quadratic_equation :
  let f (x : ℚ) := 77 * x^2 + 35 - (125 * x - 14)
  ∃ (x : ℚ), x ≠ 8/11 ∧ f x = 0 ∧ x = 1 := by
  sorry

end NUMINAMATH_CALUDE_other_solution_of_quadratic_equation_l1244_124488


namespace NUMINAMATH_CALUDE_product_of_distinct_roots_l1244_124409

theorem product_of_distinct_roots (x y : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) (hxy : x ≠ y) 
  (h : x + 3 / x = y + 3 / y) : x * y = 3 := by
  sorry

end NUMINAMATH_CALUDE_product_of_distinct_roots_l1244_124409


namespace NUMINAMATH_CALUDE_carrot_weight_problem_l1244_124480

/-- Given 30 carrots weighing 5.94 kg, and 27 of these carrots having an average weight of 200 grams,
    the average weight of the remaining 3 carrots is 180 grams. -/
theorem carrot_weight_problem (total_weight : ℝ) (avg_weight_27 : ℝ) :
  total_weight = 5.94 →
  avg_weight_27 = 0.2 →
  (total_weight * 1000 - 27 * avg_weight_27 * 1000) / 3 = 180 :=
by sorry

end NUMINAMATH_CALUDE_carrot_weight_problem_l1244_124480


namespace NUMINAMATH_CALUDE_breakfast_time_is_39_minutes_l1244_124440

def sausage_count : ℕ := 3
def egg_count : ℕ := 6
def sausage_time : ℕ := 5
def egg_time : ℕ := 4

def total_breakfast_time : ℕ := sausage_count * sausage_time + egg_count * egg_time

theorem breakfast_time_is_39_minutes : total_breakfast_time = 39 := by
  sorry

end NUMINAMATH_CALUDE_breakfast_time_is_39_minutes_l1244_124440


namespace NUMINAMATH_CALUDE_quadratic_inequality_l1244_124484

/-- Given a quadratic equation ax^2 + bx + c = 0 with roots -1 and 4, and a < 0,
    prove that ax^2 + bx + c < 0 when x < -1 or x > 4 -/
theorem quadratic_inequality (a b c : ℝ) (h1 : a < 0) 
  (h2 : ∀ x, a * x^2 + b * x + c = 0 ↔ x = -1 ∨ x = 4) :
  ∀ x, a * x^2 + b * x + c < 0 ↔ x < -1 ∨ x > 4 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_inequality_l1244_124484


namespace NUMINAMATH_CALUDE_correct_guess_probability_l1244_124451

-- Define a finite set with 4 elements
def GameOptions : Type := Fin 4

-- Define the power set of GameOptions
def PowerSet (α : Type) : Type := Set α

-- Define the number of elements in the power set of GameOptions
def NumPossibleAnswers : Nat := 2^4 - 1  -- Exclude the empty set

-- Define the probability of guessing correctly
def ProbCorrectGuess : ℚ := 1 / NumPossibleAnswers

-- Theorem statement
theorem correct_guess_probability :
  ProbCorrectGuess = 1 / 15 := by
  sorry

end NUMINAMATH_CALUDE_correct_guess_probability_l1244_124451


namespace NUMINAMATH_CALUDE_greater_number_problem_l1244_124416

theorem greater_number_problem (x y : ℝ) : 
  y = 2 * x ∧ x + y = 96 → y = 64 := by
  sorry

end NUMINAMATH_CALUDE_greater_number_problem_l1244_124416


namespace NUMINAMATH_CALUDE_prob_green_face_is_half_l1244_124492

/-- A six-faced dice with colored faces -/
structure ColoredDice :=
  (total_faces : ℕ)
  (green_faces : ℕ)
  (yellow_faces : ℕ)
  (purple_faces : ℕ)
  (h_total : total_faces = 6)
  (h_green : green_faces = 3)
  (h_yellow : yellow_faces = 2)
  (h_purple : purple_faces = 1)
  (h_sum : green_faces + yellow_faces + purple_faces = total_faces)

/-- The probability of rolling a green face on the colored dice -/
def prob_green_face (d : ColoredDice) : ℚ :=
  d.green_faces / d.total_faces

/-- Theorem: The probability of rolling a green face is 1/2 -/
theorem prob_green_face_is_half (d : ColoredDice) :
  prob_green_face d = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_prob_green_face_is_half_l1244_124492


namespace NUMINAMATH_CALUDE_cubic_equation_unique_solution_l1244_124400

theorem cubic_equation_unique_solution :
  ∀ (a : ℝ), ∃! (x : ℝ), x^3 - 2*a*x^2 + 3*a*x + a^2 - 2 = 0 := by
  sorry

end NUMINAMATH_CALUDE_cubic_equation_unique_solution_l1244_124400


namespace NUMINAMATH_CALUDE_cos_two_alpha_value_l1244_124437

theorem cos_two_alpha_value (α : Real) (h1 : π/8 < α) (h2 : α < 3*π/8) : 
  let f := fun x => Real.cos x * (Real.sin x + Real.cos x) - 1/2
  f α = Real.sqrt 2 / 6 → Real.cos (2 * α) = (Real.sqrt 2 - 4) / 6 := by
  sorry

end NUMINAMATH_CALUDE_cos_two_alpha_value_l1244_124437


namespace NUMINAMATH_CALUDE_triangle_similarity_theorem_l1244_124421

/-- Given a triangle ADE with point C on AD and point B on AC, prove that FC = 10 -/
theorem triangle_similarity_theorem 
  (DC : ℝ) (CB : ℝ) (AD : ℝ) (AB : ℝ) (ED : ℝ) (FC : ℝ) :
  DC = 9 →
  CB = 7 →
  AB = (1/3) * AD →
  ED = (2/3) * AD →
  FC = 10 := by
sorry

end NUMINAMATH_CALUDE_triangle_similarity_theorem_l1244_124421


namespace NUMINAMATH_CALUDE_profit_growth_equation_l1244_124490

/-- 
Given an initial profit of 250,000 yuan in May and an expected profit of 360,000 yuan in July,
with an average monthly growth rate of x over 2 months, prove that the equation 25(1+x)^2 = 36 holds true.
-/
theorem profit_growth_equation (x : ℝ) : 
  (250000 : ℝ) * (1 + x)^2 = 360000 → 25 * (1 + x)^2 = 36 := by
  sorry


end NUMINAMATH_CALUDE_profit_growth_equation_l1244_124490


namespace NUMINAMATH_CALUDE_mikes_bills_l1244_124494

theorem mikes_bills (total_amount : ℕ) (bill_denomination : ℕ) (h1 : total_amount = 45) (h2 : bill_denomination = 5) :
  total_amount / bill_denomination = 9 := by
  sorry

end NUMINAMATH_CALUDE_mikes_bills_l1244_124494


namespace NUMINAMATH_CALUDE_first_time_below_397_l1244_124442

def countingOff (n : ℕ) : ℕ := n - (n / 3)

def remainingStudents (initialCount : ℕ) (rounds : ℕ) : ℕ :=
  match rounds with
  | 0 => initialCount
  | n + 1 => countingOff (remainingStudents initialCount n)

theorem first_time_below_397 (initialCount : ℕ) (h : initialCount = 2010) :
  remainingStudents initialCount 5 ≤ 397 ∧
  ∀ k < 5, remainingStudents initialCount k > 397 :=
sorry

end NUMINAMATH_CALUDE_first_time_below_397_l1244_124442


namespace NUMINAMATH_CALUDE_single_point_ellipse_l1244_124455

/-- 
If the graph of 3x^2 + 4y^2 + 6x - 8y + c = 0 consists of a single point, then c = 7.
-/
theorem single_point_ellipse (c : ℝ) : 
  (∃! p : ℝ × ℝ, 3 * p.1^2 + 4 * p.2^2 + 6 * p.1 - 8 * p.2 + c = 0) → c = 7 := by
  sorry

end NUMINAMATH_CALUDE_single_point_ellipse_l1244_124455


namespace NUMINAMATH_CALUDE_valid_factorization_l1244_124443

theorem valid_factorization (x : ℝ) : x^2 - 2*x + 1 = (x - 1)^2 := by
  sorry

end NUMINAMATH_CALUDE_valid_factorization_l1244_124443


namespace NUMINAMATH_CALUDE_ellipse_equation_from_hyperbola_l1244_124493

/-- Given a hyperbola with equation 3x^2 - y^2 = 3, prove that an ellipse with the same foci
    and reciprocal eccentricity has the equation x^2/16 + y^2/12 = 1 -/
theorem ellipse_equation_from_hyperbola (x y : ℝ) :
  (3 * x^2 - y^2 = 3) →
  (∃ (a b : ℝ), a > 0 ∧ b > 0 ∧ x^2 / a^2 + y^2 / b^2 = 1 ∧
    a^2 - b^2 = 4 ∧ 2 / a = 1 / 2) →
  x^2 / 16 + y^2 / 12 = 1 := by
  sorry

end NUMINAMATH_CALUDE_ellipse_equation_from_hyperbola_l1244_124493


namespace NUMINAMATH_CALUDE_num_plane_line_pairs_is_48_l1244_124483

/-- A rectangular box -/
structure RectangularBox where
  -- We don't need to define the specifics of the box for this problem

/-- A line determined by two vertices of the box -/
structure BoxLine where
  box : RectangularBox
  -- We don't need to specify how the line is determined

/-- A plane containing four vertices of the box -/
structure BoxPlane where
  box : RectangularBox
  -- We don't need to specify how the plane is determined

/-- A plane-line pair in the box -/
structure PlaneLine where
  box : RectangularBox
  line : BoxLine
  plane : BoxPlane
  is_parallel : Bool -- Indicates if the line and plane are parallel

/-- The number of plane-line pairs in a rectangular box -/
def num_plane_line_pairs (box : RectangularBox) : Nat :=
  -- The actual implementation is not needed for the statement
  sorry

/-- Theorem stating that the number of plane-line pairs in a rectangular box is 48 -/
theorem num_plane_line_pairs_is_48 (box : RectangularBox) :
  num_plane_line_pairs box = 48 := by
  sorry

end NUMINAMATH_CALUDE_num_plane_line_pairs_is_48_l1244_124483


namespace NUMINAMATH_CALUDE_sqrt_equation_solution_l1244_124499

theorem sqrt_equation_solution :
  ∃! z : ℝ, Real.sqrt (-3 + 3 * z) = 9 :=
by
  -- The proof goes here
  sorry

end NUMINAMATH_CALUDE_sqrt_equation_solution_l1244_124499


namespace NUMINAMATH_CALUDE_final_plant_count_l1244_124453

def total_seeds : ℕ := 23
def marigold_seeds : ℕ := 10
def sunflower_seeds : ℕ := 8
def lavender_seeds : ℕ := 5
def seeds_not_grown : ℕ := 5

def marigold_growth_rate : ℚ := 2/5
def sunflower_growth_rate : ℚ := 3/5
def lavender_growth_rate : ℚ := 7/10

def squirrel_eat_rate : ℚ := 1/2
def rabbit_eat_rate : ℚ := 1/4

def pest_control_success_rate : ℚ := 3/4
def pest_control_reduction_rate : ℚ := 1/10

def weed_strangle_rate : ℚ := 1/3

def weeds_pulled : ℕ := 2
def weeds_kept : ℕ := 1

theorem final_plant_count :
  ∃ (grown_marigolds grown_sunflowers grown_lavenders : ℕ),
    grown_marigolds ≤ (marigold_seeds : ℚ) * marigold_growth_rate ∧
    grown_sunflowers ≤ (sunflower_seeds : ℚ) * sunflower_growth_rate ∧
    grown_lavenders ≤ (lavender_seeds : ℚ) * lavender_growth_rate ∧
    ∃ (eaten_marigolds eaten_sunflowers : ℕ),
      eaten_marigolds = ⌊(grown_marigolds : ℚ) * squirrel_eat_rate⌋ ∧
      eaten_sunflowers = ⌊(grown_sunflowers : ℚ) * rabbit_eat_rate⌋ ∧
      ∃ (protected_lavenders : ℕ),
        protected_lavenders ≤ ⌊(grown_lavenders : ℚ) * pest_control_success_rate⌋ ∧
        ∃ (final_marigolds final_sunflowers : ℕ),
          final_marigolds ≤ ⌊(grown_marigolds - eaten_marigolds : ℚ) * (1 - pest_control_reduction_rate)⌋ ∧
          final_sunflowers ≤ ⌊(grown_sunflowers - eaten_sunflowers : ℚ) * (1 - pest_control_reduction_rate)⌋ ∧
          ∃ (total_plants : ℕ),
            total_plants = final_marigolds + final_sunflowers + protected_lavenders ∧
            ∃ (strangled_plants : ℕ),
              strangled_plants = ⌊(total_plants : ℚ) * weed_strangle_rate⌋ ∧
              total_plants - strangled_plants + weeds_kept = 6 :=
by sorry

end NUMINAMATH_CALUDE_final_plant_count_l1244_124453


namespace NUMINAMATH_CALUDE_value_of_expression_l1244_124454

theorem value_of_expression (m n : ℤ) (h : m - n = 2) : 
  (n - m)^3 - (m - n)^2 + 1 = -11 := by
sorry

end NUMINAMATH_CALUDE_value_of_expression_l1244_124454


namespace NUMINAMATH_CALUDE_max_sum_of_two_integers_l1244_124495

theorem max_sum_of_two_integers (x y : ℕ+) : 
  y = 2 * x → x + y < 100 → (∀ a b : ℕ+, b = 2 * a → a + b < 100 → a + b ≤ x + y) → x + y = 99 :=
by sorry

end NUMINAMATH_CALUDE_max_sum_of_two_integers_l1244_124495


namespace NUMINAMATH_CALUDE_negative_one_exponent_division_l1244_124435

theorem negative_one_exponent_division : ((-1 : ℤ) ^ 2003) / ((-1 : ℤ) ^ 2004) = -1 := by
  sorry

end NUMINAMATH_CALUDE_negative_one_exponent_division_l1244_124435


namespace NUMINAMATH_CALUDE_parallel_planes_line_parallel_l1244_124403

-- Define the types for planes and lines
variable (Plane Line : Type)

-- Define the parallelism relations
variable (plane_parallel : Plane → Plane → Prop)
variable (line_parallel_plane : Line → Plane → Prop)
variable (line_parallel : Line → Line → Prop)

-- Define the "not contained in" relation
variable (line_not_in_plane : Line → Plane → Prop)

-- State the theorem
theorem parallel_planes_line_parallel 
  (α β : Plane) (m : Line) 
  (h1 : plane_parallel α β) 
  (h2 : line_parallel_plane m α) 
  (h3 : line_not_in_plane m β) : 
  line_parallel_plane m β :=
sorry

end NUMINAMATH_CALUDE_parallel_planes_line_parallel_l1244_124403


namespace NUMINAMATH_CALUDE_marc_watching_friends_l1244_124417

theorem marc_watching_friends (total_episodes : ℕ) (watch_fraction : ℚ) (days : ℕ) : 
  total_episodes = 50 → 
  watch_fraction = 1 / 10 → 
  (total_episodes : ℚ) * watch_fraction = (days : ℚ) → 
  days = 10 := by
sorry

end NUMINAMATH_CALUDE_marc_watching_friends_l1244_124417


namespace NUMINAMATH_CALUDE_regular_polygon_interior_angle_l1244_124498

theorem regular_polygon_interior_angle (n : ℕ) (n_ge_3 : n ≥ 3) :
  let interior_angle := (n - 2) * 180 / n
  interior_angle = 135 → n = 8 := by
sorry

end NUMINAMATH_CALUDE_regular_polygon_interior_angle_l1244_124498


namespace NUMINAMATH_CALUDE_imaginary_part_of_1_minus_2i_l1244_124423

theorem imaginary_part_of_1_minus_2i :
  Complex.im (1 - 2 * Complex.I) = -2 := by sorry

end NUMINAMATH_CALUDE_imaginary_part_of_1_minus_2i_l1244_124423


namespace NUMINAMATH_CALUDE_mn_product_is_66_l1244_124462

/-- A parabola shifted from y = x^2 --/
structure ShiftedParabola where
  m : ℝ
  n : ℝ
  h_shift : ℝ := 3  -- left shift
  v_shift : ℝ := 2  -- upward shift

/-- The product of m and n for a parabola y = x^2 + mx + n
    obtained by shifting y = x^2 up by 2 units and left by 3 units --/
def mn_product (p : ShiftedParabola) : ℝ := p.m * p.n

/-- Theorem: The product mn equals 66 for the specified shifted parabola --/
theorem mn_product_is_66 (p : ShiftedParabola) : mn_product p = 66 := by
  sorry

end NUMINAMATH_CALUDE_mn_product_is_66_l1244_124462


namespace NUMINAMATH_CALUDE_quadratic_solution_for_b_l1244_124413

theorem quadratic_solution_for_b (a b c m : ℝ) (h1 : m = c * a * (b - 1) / (a - b^2)) 
  (h2 : c * a ≠ 0) : m * b^2 + c * a * b - m * a - c * a = 0 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_solution_for_b_l1244_124413


namespace NUMINAMATH_CALUDE_matrix_power_4_l1244_124428

def A : Matrix (Fin 2) (Fin 2) ℤ := !![2, -1; 1, 0]

theorem matrix_power_4 :
  A ^ 4 = !![5, -4; 4, -3] := by sorry

end NUMINAMATH_CALUDE_matrix_power_4_l1244_124428


namespace NUMINAMATH_CALUDE_book_pages_l1244_124418

/-- Given a book where the total number of digits used in numbering its pages is 930,
    prove that the book has 346 pages. -/
theorem book_pages (total_digits : ℕ) (h : total_digits = 930) : ∃ (pages : ℕ), pages = 346 := by
  sorry

end NUMINAMATH_CALUDE_book_pages_l1244_124418


namespace NUMINAMATH_CALUDE_triangle_problem_l1244_124452

open Real

/-- Given a triangle ABC with sides a, b, c opposite to angles A, B, C respectively -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

/-- The theorem statement -/
theorem triangle_problem (t : Triangle) 
  (h1 : t.b = 2)
  (h2 : cos t.B = 4/5) : 
  (t.a = 5/3 → t.A = π/6) ∧ 
  (t.a + t.c = 2 * Real.sqrt 10 → 
    1/2 * t.a * t.c * sin t.B = 3) := by
  sorry

end NUMINAMATH_CALUDE_triangle_problem_l1244_124452


namespace NUMINAMATH_CALUDE_house_sale_profit_percentage_l1244_124401

/-- Calculates the profit percentage for a house sale --/
theorem house_sale_profit_percentage
  (purchase_price : ℝ)
  (selling_price : ℝ)
  (commission_rate : ℝ)
  (h1 : purchase_price = 80000)
  (h2 : selling_price = 100000)
  (h3 : commission_rate = 0.05)
  : (selling_price - commission_rate * purchase_price - purchase_price) / purchase_price = 0.2 := by
  sorry

end NUMINAMATH_CALUDE_house_sale_profit_percentage_l1244_124401


namespace NUMINAMATH_CALUDE_parallel_vectors_solution_l1244_124402

def vector_a : ℝ × ℝ := (2, -3)
def vector_b (x : ℝ) : ℝ × ℝ := (4, x^2 - 5*x)

theorem parallel_vectors_solution :
  ∀ x : ℝ, (∃ k : ℝ, vector_a = k • vector_b x) → x = 2 ∨ x = 3 := by
  sorry

end NUMINAMATH_CALUDE_parallel_vectors_solution_l1244_124402


namespace NUMINAMATH_CALUDE_rectangle_area_plus_perimeter_l1244_124465

/-- Represents a rectangle with positive integer side lengths -/
structure Rectangle where
  length : ℕ+
  width : ℕ+

/-- The area of a rectangle -/
def area (r : Rectangle) : ℕ := r.length.val * r.width.val

/-- The perimeter of a rectangle -/
def perimeter (r : Rectangle) : ℕ := 2 * (r.length.val + r.width.val)

/-- Predicate to check if a number can be expressed as the sum of area and perimeter -/
def canBeExpressedAsAreaPlusPerimeter (n : ℕ) : Prop :=
  ∃ r : Rectangle, area r + perimeter r = n

theorem rectangle_area_plus_perimeter :
  (canBeExpressedAsAreaPlusPerimeter 100) ∧
  (canBeExpressedAsAreaPlusPerimeter 104) ∧
  (canBeExpressedAsAreaPlusPerimeter 106) ∧
  (canBeExpressedAsAreaPlusPerimeter 108) ∧
  ¬(canBeExpressedAsAreaPlusPerimeter 102) := by sorry

end NUMINAMATH_CALUDE_rectangle_area_plus_perimeter_l1244_124465


namespace NUMINAMATH_CALUDE_factorization_of_quadratic_l1244_124444

theorem factorization_of_quadratic (x : ℝ) : x^2 - 5*x = x*(x - 5) := by
  sorry

end NUMINAMATH_CALUDE_factorization_of_quadratic_l1244_124444


namespace NUMINAMATH_CALUDE_yoongi_age_yoongi_age_when_namjoon_is_six_l1244_124479

theorem yoongi_age (namjoon_age : ℕ) (age_difference : ℕ) : ℕ :=
  namjoon_age - age_difference

theorem yoongi_age_when_namjoon_is_six :
  yoongi_age 6 3 = 3 := by
  sorry

end NUMINAMATH_CALUDE_yoongi_age_yoongi_age_when_namjoon_is_six_l1244_124479


namespace NUMINAMATH_CALUDE_sea_lion_count_l1244_124476

/-- Given the ratio of sea lions to penguins and their difference, 
    calculate the number of sea lions -/
theorem sea_lion_count (s p : ℕ) : 
  s * 11 = p * 4 →  -- ratio of sea lions to penguins is 4:11
  p = s + 84 →      -- 84 more penguins than sea lions
  s = 48 :=         -- prove that there are 48 sea lions
by sorry

end NUMINAMATH_CALUDE_sea_lion_count_l1244_124476


namespace NUMINAMATH_CALUDE_pumpkin_pie_pieces_l1244_124469

/-- The number of pieces a pumpkin pie is cut into -/
def pumpkin_pieces : ℕ := sorry

/-- The number of pieces a custard pie is cut into -/
def custard_pieces : ℕ := 6

/-- The price of a pumpkin pie slice in dollars -/
def pumpkin_price : ℕ := 5

/-- The price of a custard pie slice in dollars -/
def custard_price : ℕ := 6

/-- The number of pumpkin pies sold -/
def pumpkin_pies_sold : ℕ := 4

/-- The number of custard pies sold -/
def custard_pies_sold : ℕ := 5

/-- The total revenue in dollars -/
def total_revenue : ℕ := 340

theorem pumpkin_pie_pieces : 
  pumpkin_pieces * pumpkin_price * pumpkin_pies_sold + 
  custard_pieces * custard_price * custard_pies_sold = total_revenue → 
  pumpkin_pieces = 8 := by sorry

end NUMINAMATH_CALUDE_pumpkin_pie_pieces_l1244_124469


namespace NUMINAMATH_CALUDE_cost_per_serving_l1244_124491

/-- The cost per serving of a meal given the costs of ingredients and number of servings -/
theorem cost_per_serving 
  (pasta_cost : ℚ) 
  (sauce_cost : ℚ) 
  (meatballs_cost : ℚ) 
  (num_servings : ℕ) 
  (h1 : pasta_cost = 1)
  (h2 : sauce_cost = 2)
  (h3 : meatballs_cost = 5)
  (h4 : num_servings = 8) :
  (pasta_cost + sauce_cost + meatballs_cost) / num_servings = 1 := by
  sorry

end NUMINAMATH_CALUDE_cost_per_serving_l1244_124491


namespace NUMINAMATH_CALUDE_factorization_of_2x_squared_minus_8_l1244_124481

theorem factorization_of_2x_squared_minus_8 (x : ℝ) : 2 * x^2 - 8 = 2 * (x + 2) * (x - 2) := by
  sorry

end NUMINAMATH_CALUDE_factorization_of_2x_squared_minus_8_l1244_124481


namespace NUMINAMATH_CALUDE_six_balls_three_boxes_l1244_124450

/-- The number of ways to partition n indistinguishable objects into k or fewer non-empty, indistinguishable groups -/
def partition_count (n : ℕ) (k : ℕ) : ℕ := sorry

/-- Theorem: There are 6 ways to partition 6 indistinguishable objects into 3 or fewer non-empty, indistinguishable groups -/
theorem six_balls_three_boxes : partition_count 6 3 = 6 := by sorry

end NUMINAMATH_CALUDE_six_balls_three_boxes_l1244_124450


namespace NUMINAMATH_CALUDE_disjoint_subsets_remainder_l1244_124438

def S : Finset Nat := Finset.range 12

def count_disjoint_subsets (S : Finset Nat) : Nat :=
  (3^S.card - 2 * 2^S.card + 1) / 2

theorem disjoint_subsets_remainder (S : Finset Nat) (h : S = Finset.range 12) :
  count_disjoint_subsets S % 1000 = 625 := by
  sorry

end NUMINAMATH_CALUDE_disjoint_subsets_remainder_l1244_124438


namespace NUMINAMATH_CALUDE_intersection_of_M_and_N_l1244_124474

-- Define the sets M and N
def M : Set ℝ := {x : ℝ | -3 < x ∧ x ≤ 5}
def N : Set ℝ := {x : ℝ | -5 < x ∧ x < 5}

-- State the theorem
theorem intersection_of_M_and_N :
  M ∩ N = {x : ℝ | -3 < x ∧ x < 5} := by
  sorry

end NUMINAMATH_CALUDE_intersection_of_M_and_N_l1244_124474


namespace NUMINAMATH_CALUDE_retail_price_approx_163_59_l1244_124489

/-- Calculates the retail price of a machine before discount -/
def retail_price_before_discount (
  num_machines : ℕ) 
  (wholesale_price : ℚ) 
  (bulk_discount_rate : ℚ) 
  (sales_tax_rate : ℚ) 
  (profit_rate : ℚ) 
  (customer_discount_rate : ℚ) : ℚ :=
  let total_wholesale := num_machines * wholesale_price
  let bulk_discount := bulk_discount_rate * total_wholesale
  let total_cost_after_discount := total_wholesale - bulk_discount
  let profit_per_machine := profit_rate * wholesale_price
  let total_profit := num_machines * profit_per_machine
  let sales_tax := sales_tax_rate * total_profit
  let total_amount_after_tax := total_cost_after_discount + total_profit - sales_tax
  let price_before_discount := total_amount_after_tax / (num_machines * (1 - customer_discount_rate))
  price_before_discount

/-- Theorem stating that the retail price before discount is approximately $163.59 -/
theorem retail_price_approx_163_59 :
  ∃ (ε : ℚ), ε > 0 ∧ ε < 0.01 ∧ 
  |retail_price_before_discount 15 126 0.06 0.08 0.22 0.12 - 163.59| < ε :=
sorry

end NUMINAMATH_CALUDE_retail_price_approx_163_59_l1244_124489


namespace NUMINAMATH_CALUDE_polynomial_equation_l1244_124458

theorem polynomial_equation (a : ℝ) (A : ℝ → ℝ) :
  (∀ x, A x * (x + 1) = x^2 - 1) → A = fun x ↦ x - 1 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_equation_l1244_124458


namespace NUMINAMATH_CALUDE_S_bounds_l1244_124482

-- Define the function S
def S (x y z : ℝ) : ℝ := 2*x^2*y^2 + 2*x^2*z^2 + 2*y^2*z^2 - x^4 - y^4 - z^4

-- State the theorem
theorem S_bounds :
  ∀ x y z : ℝ,
  (0 ≤ x ∧ 0 ≤ y ∧ 0 ≤ z) →
  (5 ≤ x ∧ x ≤ 8) →
  (5 ≤ y ∧ y ≤ 8) →
  (5 ≤ z ∧ z ≤ 8) →
  1875 ≤ S x y z ∧ S x y z ≤ 31488 :=
by sorry

end NUMINAMATH_CALUDE_S_bounds_l1244_124482


namespace NUMINAMATH_CALUDE_exist_two_players_with_eight_points_no_undefeated_pair_from_sixth_round_l1244_124445

/-- Represents a chess tournament with the given rules --/
structure ChessTournament where
  players : ℕ
  rounds : ℕ
  win_points : ℚ
  draw_points : ℚ
  loss_points : ℚ
  bye_points : ℚ
  max_byes : ℕ

/-- Defines the specific tournament in the problem --/
def problem_tournament : ChessTournament :=
  { players := 29
  , rounds := 9
  , win_points := 1
  , draw_points := 1/2
  , loss_points := 0
  , bye_points := 1
  , max_byes := 1 }

/-- Represents the state of a player after a certain number of rounds --/
structure PlayerState where
  wins : ℕ
  losses : ℕ
  byes : ℕ

/-- Calculates the total points for a player --/
def total_points (t : ChessTournament) (p : PlayerState) : ℚ :=
  p.wins * t.win_points + p.losses * t.loss_points + min p.byes t.max_byes * t.bye_points

/-- Theorem stating that two players can have 8 points each before the final round --/
theorem exist_two_players_with_eight_points (t : ChessTournament) :
  t = problem_tournament →
  ∃ (p1 p2 : PlayerState),
    total_points t p1 = 8 ∧
    total_points t p2 = 8 ∧
    p1.wins + p1.losses + p1.byes < t.rounds ∧
    p2.wins + p2.losses + p2.byes < t.rounds :=
  sorry

/-- Theorem stating that from the 6th round, no two undefeated players can meet --/
theorem no_undefeated_pair_from_sixth_round (t : ChessTournament) :
  t = problem_tournament →
  ∀ (r : ℕ), r ≥ 6 →
  ¬∃ (p1 p2 : PlayerState),
    p1.wins = r - 1 ∧
    p2.wins = r - 1 ∧
    p1.losses = 0 ∧
    p2.losses = 0 :=
  sorry

end NUMINAMATH_CALUDE_exist_two_players_with_eight_points_no_undefeated_pair_from_sixth_round_l1244_124445


namespace NUMINAMATH_CALUDE_cube_sum_factorization_l1244_124472

theorem cube_sum_factorization (p q r s t u : ℤ) :
  (∀ x, 729 * x^3 + 64 = (p * x^2 + q * x + r) * (s * x^2 + t * x + u)) →
  p^2 + q^2 + r^2 + s^2 + t^2 + u^2 = 8210 := by
  sorry

end NUMINAMATH_CALUDE_cube_sum_factorization_l1244_124472


namespace NUMINAMATH_CALUDE_division_problem_l1244_124460

theorem division_problem (x y z : ℚ) 
  (h1 : x / y = 3)
  (h2 : y / z = 5/2) :
  z / x = 2/15 := by
sorry

end NUMINAMATH_CALUDE_division_problem_l1244_124460

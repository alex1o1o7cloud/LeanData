import Mathlib

namespace weeks_to_buy_bike_l1068_106806

def mountain_bike_cost : ℕ := 600
def birthday_money : ℕ := 60 + 40 + 20 + 30
def weekly_earnings : ℕ := 18

theorem weeks_to_buy_bike : 
  ∃ (weeks : ℕ), birthday_money + weeks * weekly_earnings = mountain_bike_cost ∧ weeks = 25 :=
by sorry

end weeks_to_buy_bike_l1068_106806


namespace element_in_set_l1068_106867

def U : Set Nat := {1, 2, 3, 4, 5}

theorem element_in_set (M : Set Nat) (h : Set.compl M = {1, 3}) : 2 ∈ M := by
  sorry

end element_in_set_l1068_106867


namespace sum_70_is_negative_350_l1068_106896

/-- An arithmetic progression with specified properties -/
structure ArithmeticProgression where
  /-- First term of the progression -/
  a : ℚ
  /-- Common difference of the progression -/
  d : ℚ
  /-- Sum of first 20 terms is 200 -/
  sum_20 : (20 : ℚ) / 2 * (2 * a + (20 - 1) * d) = 200
  /-- Sum of first 50 terms is 50 -/
  sum_50 : (50 : ℚ) / 2 * (2 * a + (50 - 1) * d) = 50

/-- The sum of the first 70 terms of the arithmetic progression is -350 -/
theorem sum_70_is_negative_350 (ap : ArithmeticProgression) :
  (70 : ℚ) / 2 * (2 * ap.a + (70 - 1) * ap.d) = -350 := by
  sorry

end sum_70_is_negative_350_l1068_106896


namespace smallest_number_with_remainders_l1068_106886

theorem smallest_number_with_remainders : ∃! a : ℕ+, 
  (a : ℤ) % 4 = 1 ∧ 
  (a : ℤ) % 3 = 1 ∧ 
  (a : ℤ) % 5 = 2 ∧ 
  (∀ n : ℕ+, n < a → ((n : ℤ) % 4 ≠ 1 ∨ (n : ℤ) % 3 ≠ 1 ∨ (n : ℤ) % 5 ≠ 2)) ∧
  a = 37 :=
by sorry

end smallest_number_with_remainders_l1068_106886


namespace bicycle_wheels_count_l1068_106821

theorem bicycle_wheels_count (num_bicycles num_tricycles tricycle_wheels total_wheels : ℕ) 
  (h1 : num_bicycles = 24)
  (h2 : num_tricycles = 14)
  (h3 : tricycle_wheels = 3)
  (h4 : total_wheels = 90)
  (h5 : total_wheels = num_bicycles * bicycle_wheels + num_tricycles * tricycle_wheels) :
  bicycle_wheels = 2 := by
  sorry

end bicycle_wheels_count_l1068_106821


namespace car_journey_metrics_l1068_106840

/-- Represents the car's journey with given conditions -/
structure CarJourney where
  total_distance : ℝ
  acc_dec_distance : ℝ
  constant_speed_distance : ℝ
  acc_dec_time : ℝ
  reference_speed : ℝ
  reference_time : ℝ

/-- Calculates the acceleration rate, deceleration rate, and highest speed of the car -/
def calculate_car_metrics (journey : CarJourney) : 
  (ℝ × ℝ × ℝ) :=
  sorry

/-- Theorem stating the correct acceleration, deceleration, and highest speed -/
theorem car_journey_metrics :
  let journey : CarJourney := {
    total_distance := 100,
    acc_dec_distance := 1,
    constant_speed_distance := 98,
    acc_dec_time := 100,
    reference_speed := 40 / 3600,
    reference_time := 90
  }
  let (acc_rate, dec_rate, highest_speed) := calculate_car_metrics journey
  acc_rate = 0.0002 ∧ dec_rate = 0.0002 ∧ highest_speed = 72 / 3600 :=
by sorry

end car_journey_metrics_l1068_106840


namespace expression_value_l1068_106826

theorem expression_value (a b c d m : ℚ) 
  (h1 : a = -b) 
  (h2 : c * d = 1) 
  (h3 : |m| = 4) : 
  2*a - 5*c*d - m + 2*b = -9 ∨ 2*a - 5*c*d - m + 2*b = -1 := by
  sorry

end expression_value_l1068_106826


namespace trapezium_other_side_length_l1068_106814

theorem trapezium_other_side_length 
  (a : ℝ) -- Area of the trapezium
  (b : ℝ) -- Length of one parallel side
  (h : ℝ) -- Distance between parallel sides
  (x : ℝ) -- Length of the other parallel side
  (h1 : a = 380) -- Area is 380 square centimeters
  (h2 : b = 18)  -- One parallel side is 18 cm
  (h3 : h = 20)  -- Distance between parallel sides is 20 cm
  (h4 : a = (1/2) * (x + b) * h) -- Area formula for trapezium
  : x = 20 := by
  sorry

end trapezium_other_side_length_l1068_106814


namespace complex_function_minimum_on_unit_circle_l1068_106897

/-- For a ∈ (0,1) and f(z) = z^2 - z + a, for any complex number z with |z| ≥ 1,
    there exists a complex number z₀ with |z₀| = 1 such that |f(z₀)| ≤ |f(z)| -/
theorem complex_function_minimum_on_unit_circle (a : ℝ) (ha : 0 < a ∧ a < 1) :
  ∀ z : ℂ, Complex.abs z ≥ 1 →
    ∃ z₀ : ℂ, Complex.abs z₀ = 1 ∧
      Complex.abs (z₀^2 - z₀ + a) ≤ Complex.abs (z^2 - z + a) :=
by sorry

end complex_function_minimum_on_unit_circle_l1068_106897


namespace white_coinciding_pairs_l1068_106849

/-- Represents the number of triangles of each color in each half of the figure -/
structure TriangleCount where
  red : ℕ
  blue : ℕ
  green : ℕ
  white : ℕ

/-- Represents the number of coinciding pairs of triangles when the figure is folded -/
structure CoincidingPairs where
  red_red : ℕ
  blue_blue : ℕ
  green_green : ℕ
  red_white : ℕ
  green_blue : ℕ

/-- Theorem stating that the number of coinciding white triangle pairs is 4 -/
theorem white_coinciding_pairs
  (half_count : TriangleCount)
  (coinciding : CoincidingPairs)
  (h1 : half_count.red = 4)
  (h2 : half_count.blue = 4)
  (h3 : half_count.green = 2)
  (h4 : half_count.white = 6)
  (h5 : coinciding.red_red = 3)
  (h6 : coinciding.blue_blue = 2)
  (h7 : coinciding.green_green = 1)
  (h8 : coinciding.red_white = 2)
  (h9 : coinciding.green_blue = 1) :
  ∃ (white_pairs : ℕ), white_pairs = 4 ∧ 
  white_pairs = half_count.white - coinciding.red_white := by
  sorry

end white_coinciding_pairs_l1068_106849


namespace symmetric_curve_equation_l1068_106883

/-- Given a curve C defined by F(x, y) = 0 and a point of symmetry (a, b),
    the equation of the curve symmetric to C about (a, b) is F(2a-x, 2b-y) = 0 -/
theorem symmetric_curve_equation (F : ℝ → ℝ → ℝ) (a b : ℝ) :
  (∀ x y, F x y = 0 ↔ F (2*a - x) (2*b - y) = 0) :=
by sorry

end symmetric_curve_equation_l1068_106883


namespace chandler_can_buy_bike_l1068_106854

/-- The cost of the mountain bike in dollars -/
def bike_cost : ℕ := 800

/-- The total amount of gift money Chandler received in dollars -/
def gift_money : ℕ := 100 + 50 + 20 + 30

/-- The amount Chandler earns per week from his paper route in dollars -/
def weekly_earnings : ℕ := 20

/-- The number of weeks Chandler needs to save to buy the mountain bike -/
def weeks_to_save : ℕ := 30

/-- Theorem stating that Chandler can buy the bike after saving for the calculated number of weeks -/
theorem chandler_can_buy_bike :
  gift_money + weekly_earnings * weeks_to_save = bike_cost :=
by sorry

end chandler_can_buy_bike_l1068_106854


namespace problem_solution_l1068_106829

theorem problem_solution (y : ℝ) (d e f : ℕ+) :
  y = Real.sqrt ((Real.sqrt 75) / 2 + 5 / 2) →
  y^100 = 3*y^98 + 18*y^96 + 15*y^94 - y^50 + (d : ℝ)*y^46 + (e : ℝ)*y^44 + (f : ℝ)*y^40 →
  (d : ℝ) + (e : ℝ) + (f : ℝ) = 556.5 := by
  sorry

end problem_solution_l1068_106829


namespace water_evaporation_rate_l1068_106855

theorem water_evaporation_rate 
  (initial_water : ℝ) 
  (days : ℕ) 
  (evaporation_percentage : ℝ) 
  (h1 : initial_water = 10)
  (h2 : days = 50)
  (h3 : evaporation_percentage = 40) : 
  (initial_water * evaporation_percentage / 100) / days = 0.08 := by
  sorry

end water_evaporation_rate_l1068_106855


namespace negation_equivalence_l1068_106810

theorem negation_equivalence (m : ℤ) :
  (¬ ∃ x : ℤ, x^2 + 2*x + m < 0) ↔ (∀ x : ℤ, x^2 + 2*x + m ≥ 0) := by
  sorry

end negation_equivalence_l1068_106810


namespace swallow_pests_calculation_l1068_106825

/-- The number of pests a frog can catch per day -/
def frog_pests : ℕ := 145

/-- The multiplier for how many times more pests a swallow can eliminate compared to a frog -/
def swallow_multiplier : ℕ := 12

/-- The number of pests a swallow can eliminate per day -/
def swallow_pests : ℕ := frog_pests * swallow_multiplier

theorem swallow_pests_calculation : swallow_pests = 1740 := by
  sorry

end swallow_pests_calculation_l1068_106825


namespace sports_participation_l1068_106885

theorem sports_participation (total_students : ℕ) (basketball cricket soccer : ℕ)
  (basketball_cricket basketball_soccer cricket_soccer : ℕ) (all_three : ℕ)
  (h1 : total_students = 50)
  (h2 : basketball = 16)
  (h3 : cricket = 11)
  (h4 : soccer = 10)
  (h5 : basketball_cricket = 5)
  (h6 : basketball_soccer = 4)
  (h7 : cricket_soccer = 3)
  (h8 : all_three = 2) :
  basketball + cricket + soccer - (basketball_cricket + basketball_soccer + cricket_soccer) + all_three = 27 := by
  sorry

end sports_participation_l1068_106885


namespace shoveling_time_l1068_106842

theorem shoveling_time (kevin dave john allison : ℝ)
  (h_kevin : kevin = 12)
  (h_dave : dave = 8)
  (h_john : john = 6)
  (h_allison : allison = 4) :
  (1 / kevin + 1 / dave + 1 / john + 1 / allison)⁻¹ * 60 = 96 := by
  sorry

end shoveling_time_l1068_106842


namespace hexagon_area_l1068_106851

/-- A regular hexagon divided by three diagonals -/
structure RegularHexagon where
  /-- The area of one small triangle formed by the diagonals -/
  small_triangle_area : ℝ
  /-- The total number of small triangles in the hexagon -/
  total_triangles : ℕ
  /-- The number of shaded triangles -/
  shaded_triangles : ℕ
  /-- The total shaded area -/
  shaded_area : ℝ
  /-- The hexagon is divided into 12 congruent triangles -/
  triangle_count : total_triangles = 12
  /-- Two regions (equivalent to 5 small triangles) are shaded -/
  shaded_count : shaded_triangles = 5
  /-- The total shaded area is 20 cm² -/
  shaded_area_value : shaded_area = 20

/-- The theorem stating the area of the hexagon -/
theorem hexagon_area (h : RegularHexagon) : h.total_triangles * h.small_triangle_area = 48 := by
  sorry

end hexagon_area_l1068_106851


namespace upper_limit_y_l1068_106852

theorem upper_limit_y (x y : ℤ) (h1 : 5 < x) (h2 : x < 8) (h3 : 8 < y) 
  (h4 : ∀ (a b : ℤ), 5 < a → a < 8 → 8 < b → b - a ≤ 7) : y ≤ 14 := by
  sorry

end upper_limit_y_l1068_106852


namespace general_term_max_sum_l1068_106857

/-- An arithmetic sequence with given properties -/
structure ArithmeticSequence where
  a : ℕ → ℤ  -- General term of the sequence
  S : ℕ → ℤ  -- Sum function of the sequence
  sum_3 : S 3 = 42
  sum_6 : S 6 = 57

/-- The general term of the sequence is 20 - 3n -/
theorem general_term (seq : ArithmeticSequence) : 
  ∀ n : ℕ, seq.a n = 20 - 3 * n := by sorry

/-- The sum S_n is maximized when n = 6 -/
theorem max_sum (seq : ArithmeticSequence) : 
  ∃ n : ℕ, ∀ m : ℕ, seq.S n ≥ seq.S m ∧ n = 6 := by sorry

end general_term_max_sum_l1068_106857


namespace quadratic_even_function_sum_l1068_106898

/-- A quadratic function of the form f(x) = x^2 + (a-1)x + a + b -/
def f (a b : ℝ) (x : ℝ) : ℝ := x^2 + (a - 1) * x + a + b

/-- f is an even function -/
def is_even_function (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = f x

theorem quadratic_even_function_sum (a b : ℝ) :
  is_even_function (f a b) → f a b 2 = 0 → a + b = -4 := by
  sorry

end quadratic_even_function_sum_l1068_106898


namespace stratified_sampling_survey_size_l1068_106882

/-- Proves that the total number of surveyed students is 10 given the conditions of the problem -/
theorem stratified_sampling_survey_size 
  (total_students : ℕ) 
  (female_students : ℕ) 
  (sampled_females : ℕ) 
  (h1 : total_students = 50)
  (h2 : female_students = 20)
  (h3 : sampled_females = 4)
  (h4 : female_students < total_students) :
  ∃ (surveyed_students : ℕ), 
    surveyed_students * female_students = sampled_females * total_students ∧ 
    surveyed_students = 10 := by
  sorry

end stratified_sampling_survey_size_l1068_106882


namespace sum_reciprocals_l1068_106858

theorem sum_reciprocals (a b c d : ℝ) (ω : ℂ) : 
  a ≠ -1 → b ≠ -1 → c ≠ -1 → d ≠ -1 →
  ω^4 = 1 → ω ≠ 1 →
  (1 / (a + ω) + 1 / (b + ω) + 1 / (c + ω) + 1 / (d + ω) = 3 / ω) →
  (1 / (a + 2) + 1 / (b + 2) + 1 / (c + 2) + 1 / (d + 2) = 1 / 4) :=
by sorry

end sum_reciprocals_l1068_106858


namespace at_least_one_nonnegative_l1068_106893

theorem at_least_one_nonnegative (m n : ℝ) (hm : m > 0) (hn : n > 0) (hmn : m * n > 1) :
  let f := fun x => x^2 - x
  f m ≥ 0 ∨ f n ≥ 0 := by
sorry

end at_least_one_nonnegative_l1068_106893


namespace marketValueTheorem_l1068_106843

/-- Calculates the market value of a machine after two years, given initial value,
    depreciation rates, and inflation rate. -/
def marketValueAfterTwoYears (initialValue : ℝ) (depreciation1 : ℝ) (depreciation2 : ℝ) (inflation : ℝ) : ℝ :=
  let value1 := initialValue * (1 - depreciation1) * (1 + inflation)
  let value2 := value1 * (1 - depreciation2) * (1 + inflation)
  value2

/-- Theorem stating that the market value of a machine with given parameters
    after two years is approximately 4939.20. -/
theorem marketValueTheorem :
  ∃ ε > 0, ε < 0.01 ∧ 
  |marketValueAfterTwoYears 8000 0.3 0.2 0.05 - 4939.20| < ε :=
sorry

end marketValueTheorem_l1068_106843


namespace percentage_problem_l1068_106881

theorem percentage_problem :
  let percentage := 6.620000000000001
  let value := 66.2
  let x := value / (percentage / 100)
  x = 1000 := by
  sorry

end percentage_problem_l1068_106881


namespace inscribed_squares_inequality_l1068_106890

/-- Given a triangle ABC with sides a, b, and c, and inscribed squares with side lengths x, y, and z
    on sides BC, AC, and AB respectively, prove that (a/x) + (b/y) + (c/z) ≥ 3 + 2√3. -/
theorem inscribed_squares_inequality (a b c x y z : ℝ) (h_pos : a > 0 ∧ b > 0 ∧ c > 0 ∧ x > 0 ∧ y > 0 ∧ z > 0)
    (h_triangle : a + b > c ∧ b + c > a ∧ c + a > b)
    (h_square_x : x ≤ b ∧ x ≤ c)
    (h_square_y : y ≤ c ∧ y ≤ a)
    (h_square_z : z ≤ a ∧ z ≤ b) :
  a / x + b / y + c / z ≥ 3 + 2 * Real.sqrt 3 :=
by sorry

end inscribed_squares_inequality_l1068_106890


namespace hyperbola_eccentricity_l1068_106892

/-- Given a hyperbola with equation x²/a² - y²/b² = 1 where a > 0 and b > 0,
    if one of its asymptotes forms a 30° angle with the x-axis,
    then its eccentricity is 2√3/3. -/
theorem hyperbola_eccentricity (a b : ℝ) (ha : a > 0) (hb : b > 0) 
  (h_asymptote : b / a = Real.tan (π / 6)) :
  let e := Real.sqrt (1 + (b / a)^2)
  e = 2 * Real.sqrt 3 / 3 := by sorry

end hyperbola_eccentricity_l1068_106892


namespace valid_draw_count_l1068_106801

def total_cards : ℕ := 16
def cards_per_color : ℕ := 4
def cards_drawn : ℕ := 3

def valid_draw (total : ℕ) (per_color : ℕ) (drawn : ℕ) : ℕ :=
  Nat.choose total drawn - 
  4 * Nat.choose per_color drawn - 
  Nat.choose per_color 2 * Nat.choose (total - per_color) 1

theorem valid_draw_count :
  valid_draw total_cards cards_per_color cards_drawn = 472 := by
  sorry

end valid_draw_count_l1068_106801


namespace not_parallel_to_skew_line_l1068_106869

-- Define the concept of lines in 3D space
variable (Line : Type)

-- Define the relationships between lines
variable (skew : Line → Line → Prop)
variable (parallel : Line → Line → Prop)

-- Theorem statement
theorem not_parallel_to_skew_line 
  (a b c : Line) 
  (h1 : skew a b) 
  (h2 : parallel c a) : 
  ¬ parallel c b :=
sorry

end not_parallel_to_skew_line_l1068_106869


namespace marbles_collection_sum_l1068_106891

def total_marbles (adam mary greg john sarah : ℕ) : ℕ :=
  adam + mary + greg + john + sarah

theorem marbles_collection_sum :
  ∀ (adam mary greg john sarah : ℕ),
    adam = 29 →
    mary = adam - 11 →
    greg = adam + 14 →
    john = 2 * mary →
    sarah = greg - 7 →
    total_marbles adam mary greg john sarah = 162 :=
by
  sorry

end marbles_collection_sum_l1068_106891


namespace rajs_house_bathrooms_l1068_106819

/-- Represents the floor plan of Raj's house -/
structure HouseFloorPlan where
  total_area : ℕ
  bedroom_count : ℕ
  bedroom_side : ℕ
  bathroom_length : ℕ
  bathroom_width : ℕ
  kitchen_area : ℕ

/-- Calculates the number of bathrooms in Raj's house -/
def calculate_bathrooms (house : HouseFloorPlan) : ℕ :=
  let bedroom_area := house.bedroom_count * house.bedroom_side * house.bedroom_side
  let living_area := house.kitchen_area
  let remaining_area := house.total_area - (bedroom_area + house.kitchen_area + living_area)
  let bathroom_area := house.bathroom_length * house.bathroom_width
  remaining_area / bathroom_area

/-- Theorem stating that Raj's house has exactly 2 bathrooms -/
theorem rajs_house_bathrooms :
  let house : HouseFloorPlan := {
    total_area := 1110,
    bedroom_count := 4,
    bedroom_side := 11,
    bathroom_length := 6,
    bathroom_width := 8,
    kitchen_area := 265
  }
  calculate_bathrooms house = 2 := by
  sorry


end rajs_house_bathrooms_l1068_106819


namespace slope_of_line_l_l1068_106876

-- Define the ellipse
def ellipse (x y : ℝ) : Prop := x^2/16 + y^2/4 = 1

-- Define the point M
def M : ℝ × ℝ := (2, 1)

-- Define the line l passing through M with slope m
def line_l (m : ℝ) (x y : ℝ) : Prop :=
  y - M.2 = m * (x - M.1)

-- Define the intersection points A and B
def intersection_points (m : ℝ) :=
  ∃ (xa ya xb yb : ℝ),
    ellipse xa ya ∧ ellipse xb yb ∧
    line_l m xa ya ∧ line_l m xb yb ∧
    (xa, ya) ≠ (xb, yb)

-- Define M as the trisection point of AB
def M_is_trisection (m : ℝ) :=
  ∃ (xa ya xb yb : ℝ),
    ellipse xa ya ∧ ellipse xb yb ∧
    line_l m xa ya ∧ line_l m xb yb ∧
    2 * M.1 = xa + xb ∧ 2 * M.2 = ya + yb

-- The main theorem
theorem slope_of_line_l :
  ∃ (m : ℝ), intersection_points m ∧ M_is_trisection m ∧
  (m = (-4 + Real.sqrt 7) / 6 ∨ m = (-4 - Real.sqrt 7) / 6) :=
sorry

end slope_of_line_l_l1068_106876


namespace tangent_product_equals_two_l1068_106863

theorem tangent_product_equals_two :
  let tan17 := Real.tan (17 * π / 180)
  let tan28 := Real.tan (28 * π / 180)
  (∀ a b, Real.tan (a + b) = (Real.tan a + Real.tan b) / (1 - Real.tan a * Real.tan b)) →
  17 + 28 = 45 →
  Real.tan (45 * π / 180) = 1 →
  (1 + tan17) * (1 + tan28) = 2 := by
sorry

end tangent_product_equals_two_l1068_106863


namespace yardley_snowfall_l1068_106828

/-- The total snowfall in Yardley is the sum of morning and afternoon snowfall -/
theorem yardley_snowfall (morning_snowfall afternoon_snowfall : ℚ) 
  (h1 : morning_snowfall = 0.125)
  (h2 : afternoon_snowfall = 0.5) :
  morning_snowfall + afternoon_snowfall = 0.625 := by
  sorry

end yardley_snowfall_l1068_106828


namespace intersection_P_T_l1068_106802

def P : Set ℝ := {x | x^2 - x - 2 = 0}
def T : Set ℝ := {x | -1 < x ∧ x ≤ 2}

theorem intersection_P_T : P ∩ T = {2} := by sorry

end intersection_P_T_l1068_106802


namespace regular_polygon_sides_l1068_106811

/-- A regular polygon with an exterior angle of 18° has 20 sides -/
theorem regular_polygon_sides (n : ℕ) (ext_angle : ℝ) : 
  ext_angle = 18 → n * ext_angle = 360 → n = 20 := by sorry

end regular_polygon_sides_l1068_106811


namespace rebus_puzzle_solution_l1068_106871

theorem rebus_puzzle_solution :
  ∃! (A B C : Nat),
    A ≠ 0 ∧ B ≠ 0 ∧ C ≠ 0 ∧
    A ≠ B ∧ B ≠ C ∧ A ≠ C ∧
    A < 10 ∧ B < 10 ∧ C < 10 ∧
    (100 * A + 10 * B + A) + (100 * A + 10 * B + C) + (100 * A + 10 * C + C) = 1416 ∧
    A = 4 ∧ B = 7 ∧ C = 6 := by
  sorry

end rebus_puzzle_solution_l1068_106871


namespace prime_divisor_of_2p_minus_1_l1068_106899

theorem prime_divisor_of_2p_minus_1 (p : ℕ) (hp : Prime p) :
  ∀ q : ℕ, Prime q → q ∣ (2^p - 1) → q > p :=
by sorry

end prime_divisor_of_2p_minus_1_l1068_106899


namespace candy_problem_l1068_106878

theorem candy_problem (r g b : ℕ+) (n : ℕ) : 
  (10 * r = 16 * g) → 
  (16 * g = 18 * b) → 
  (18 * b = 25 * n) → 
  (∀ m : ℕ, m < n → ¬(∃ r' g' b' : ℕ+, 10 * r' = 16 * g' ∧ 16 * g' = 18 * b' ∧ 18 * b' = 25 * m)) →
  n = 29 := by
sorry

end candy_problem_l1068_106878


namespace lcm_1540_2310_l1068_106800

theorem lcm_1540_2310 : Nat.lcm 1540 2310 = 4620 := by
  sorry

end lcm_1540_2310_l1068_106800


namespace trigonometric_identities_l1068_106884

theorem trigonometric_identities (α β : Real) 
  (h1 : 0 < α ∧ α < π / 2) 
  (h2 : 0 < β ∧ β < π / 2)
  (h3 : 3 * Real.sin α = 4 * Real.cos α)
  (h4 : Real.cos (α + β) = -(2 * Real.sqrt 5) / 5) :
  Real.cos (2 * α) = -7 / 25 ∧ Real.sin β = (2 * Real.sqrt 5) / 5 := by
  sorry

end trigonometric_identities_l1068_106884


namespace nested_abs_ratio_values_l1068_106824

/-- Recursive function representing nested absolute value operations -/
def nestedAbs (n : ℕ) (x y : ℝ) : ℝ :=
  match n with
  | 0 => x
  | n + 1 => |nestedAbs n x y - y|

/-- The equation condition from the problem -/
def equationCondition (x y : ℝ) : Prop :=
  nestedAbs 2019 x y = nestedAbs 2019 y x

/-- The theorem statement -/
theorem nested_abs_ratio_values (x y : ℝ) (hx : x > 0) (hy : y > 0) 
  (h_eq : equationCondition x y) :
  x / y = 1/3 ∨ x / y = 1 ∨ x / y = 3 := by
  sorry

end nested_abs_ratio_values_l1068_106824


namespace negation_of_all_x_squared_positive_negation_is_true_l1068_106813

theorem negation_of_all_x_squared_positive :
  (¬ (∀ x : ℝ, x^2 > 0)) ↔ (∃ x : ℝ, x^2 ≤ 0) :=
by sorry

theorem negation_is_true : ∃ x : ℝ, x^2 ≤ 0 :=
by sorry

end negation_of_all_x_squared_positive_negation_is_true_l1068_106813


namespace range_of_m_range_of_a_l1068_106807

-- Part I
theorem range_of_m (m : ℝ) : 
  (∀ x : ℝ, 1/3 < x ∧ x < 1/2 → |x - m| < 1) → 
  -1/2 ≤ m ∧ m ≤ 4/3 := by
sorry

-- Part II
theorem range_of_a (a : ℝ) :
  (∃ x : ℝ, |x - 3| + |x - 5| < a) →
  a > 2 := by
sorry

end range_of_m_range_of_a_l1068_106807


namespace crayons_left_correct_l1068_106812

/-- Represents the number of crayons and erasers Paul has -/
structure PaulsCrayonsAndErasers where
  initial_crayons : ℕ
  initial_erasers : ℕ
  remaining_difference : ℕ

/-- Calculates the number of crayons Paul has left -/
def crayons_left (p : PaulsCrayonsAndErasers) : ℕ :=
  p.initial_erasers + p.remaining_difference

theorem crayons_left_correct (p : PaulsCrayonsAndErasers) 
  (h : p.initial_crayons = 531 ∧ p.initial_erasers = 38 ∧ p.remaining_difference = 353) : 
  crayons_left p = 391 := by
  sorry

end crayons_left_correct_l1068_106812


namespace square_root_of_4096_l1068_106868

theorem square_root_of_4096 (x : ℝ) (h1 : x > 0) (h2 : x^2 = 4096) : x = 64 := by
  sorry

end square_root_of_4096_l1068_106868


namespace rectangle_diagonal_l1068_106834

theorem rectangle_diagonal (side : ℝ) (area : ℝ) (diagonal : ℝ) : 
  side = 15 → area = 120 → diagonal = 17 → 
  ∃ other_side : ℝ, 
    area = side * other_side ∧ 
    diagonal^2 = side^2 + other_side^2 :=
by
  sorry

end rectangle_diagonal_l1068_106834


namespace disjoint_iff_valid_range_l1068_106845

/-- Set M represents a unit circle centered at the origin -/
def M : Set (ℝ × ℝ) := {p | p.1^2 + p.2^2 = 1}

/-- Set N represents a diamond centered at (1, 1) with side length a/√2 -/
def N (a : ℝ) : Set (ℝ × ℝ) := {p | |p.1 - 1| + |p.2 - 1| = a}

/-- The range of a for which M and N are disjoint -/
def valid_range : Set ℝ := {a | a < 2 - Real.sqrt 2 ∨ a > 2 + Real.sqrt 2}

theorem disjoint_iff_valid_range (a : ℝ) : 
  Disjoint (M : Set (ℝ × ℝ)) (N a) ↔ a ∈ valid_range := by sorry

end disjoint_iff_valid_range_l1068_106845


namespace unique_a_value_l1068_106803

def A (a : ℝ) : Set ℝ := {2, 3, a^2 - 3*a, a + 2/a + 7}
def B (a : ℝ) : Set ℝ := {|a - 2|, 3}

theorem unique_a_value : ∃! a : ℝ, (4 ∈ A a) ∧ (4 ∉ B a) := by
  sorry

end unique_a_value_l1068_106803


namespace ratio_of_squares_to_products_l1068_106823

theorem ratio_of_squares_to_products (x y z : ℝ) 
  (h_distinct : x ≠ y ∧ y ≠ z ∧ x ≠ z) 
  (h_sum : x + 2*y + 3*z = 0) : 
  (x^2 + y^2 + z^2) / (x*y + y*z + z*x) = -4 := by
  sorry

end ratio_of_squares_to_products_l1068_106823


namespace complement_A_intersect_B_l1068_106835

-- Define the sets A and B
def A : Set ℝ := {x | x ≤ -2}
def B : Set ℝ := {x | x < 1}

-- State the theorem
theorem complement_A_intersect_B :
  (Aᶜ ∩ B) = {x | -2 < x ∧ x < 1} := by sorry

end complement_A_intersect_B_l1068_106835


namespace equal_bill_time_l1068_106895

/-- United Telephone's base rate -/
def united_base : ℝ := 8

/-- United Telephone's per-minute rate -/
def united_per_minute : ℝ := 0.25

/-- Atlantic Call's base rate -/
def atlantic_base : ℝ := 12

/-- Atlantic Call's per-minute rate -/
def atlantic_per_minute : ℝ := 0.20

/-- The number of minutes at which the bills are equal -/
def equal_minutes : ℝ := 80

theorem equal_bill_time :
  united_base + united_per_minute * equal_minutes =
  atlantic_base + atlantic_per_minute * equal_minutes :=
by sorry

end equal_bill_time_l1068_106895


namespace contractor_absent_days_l1068_106822

/-- Represents the contractor's work scenario -/
structure ContractorScenario where
  totalDays : ℕ
  payPerWorkDay : ℚ
  finePerAbsentDay : ℚ
  totalPay : ℚ

/-- Calculates the number of absent days for a given contractor scenario -/
def absentDays (scenario : ContractorScenario) : ℚ :=
  (scenario.totalDays * scenario.payPerWorkDay - scenario.totalPay) / (scenario.payPerWorkDay + scenario.finePerAbsentDay)

/-- Theorem stating that for the given scenario, the number of absent days is 2 -/
theorem contractor_absent_days :
  let scenario : ContractorScenario := {
    totalDays := 30,
    payPerWorkDay := 25,
    finePerAbsentDay := 7.5,
    totalPay := 685
  }
  absentDays scenario = 2 := by sorry

end contractor_absent_days_l1068_106822


namespace max_quotient_value_l1068_106827

theorem max_quotient_value (x y : ℝ) (hx : 100 ≤ x ∧ x ≤ 300) (hy : 900 ≤ y ∧ y ≤ 1800) :
  (∀ x' y', 100 ≤ x' ∧ x' ≤ 300 → 900 ≤ y' ∧ y' ≤ 1800 → y' / x' ≤ 18) ∧
  (∃ x' y', 100 ≤ x' ∧ x' ≤ 300 ∧ 900 ≤ y' ∧ y' ≤ 1800 ∧ y' / x' = 18) :=
by sorry

end max_quotient_value_l1068_106827


namespace greatest_number_l1068_106839

theorem greatest_number (A B C : ℤ) : 
  A = 95 - 35 →
  B = A + 12 →
  C = B - 19 →
  A ≠ B ∧ B ≠ C ∧ A ≠ C →
  B > A ∧ B > C :=
by
  sorry

end greatest_number_l1068_106839


namespace cake_and_bread_weight_l1068_106860

/-- Given the weight of 4 cakes and the weight difference between a cake and a piece of bread,
    calculate the total weight of 3 cakes and 5 pieces of bread. -/
theorem cake_and_bread_weight (cake_weight : ℕ) (bread_weight : ℕ) : 
  (4 * cake_weight = 800) →
  (cake_weight = bread_weight + 100) →
  (3 * cake_weight + 5 * bread_weight = 1100) :=
by sorry

end cake_and_bread_weight_l1068_106860


namespace apples_in_basket_proof_l1068_106815

/-- Given a total number of apples and the capacity of each box,
    calculate the number of apples left for the basket. -/
def applesInBasket (totalApples : ℕ) (applesPerBox : ℕ) : ℕ :=
  totalApples - (totalApples / applesPerBox) * applesPerBox

/-- Prove that with 138 total apples and boxes of 18 apples each,
    there will be 12 apples left for the basket. -/
theorem apples_in_basket_proof :
  applesInBasket 138 18 = 12 := by
  sorry

end apples_in_basket_proof_l1068_106815


namespace cyclic_power_inequality_l1068_106836

theorem cyclic_power_inequality (a b c r s : ℝ) 
  (hr : r > s) (hs : s > 0) (hab : a > b) (hbc : b > c) :
  a^r * b^s + b^r * c^s + c^r * a^s ≥ a^s * b^r + b^s * c^r + c^s * a^r :=
by sorry

end cyclic_power_inequality_l1068_106836


namespace f_minimum_value_l1068_106818

def f (x y : ℝ) : ℝ := (1 - y)^2 + (x + y - 3)^2 + (2*x + y - 6)^2

theorem f_minimum_value :
  ∀ x y : ℝ, f x y ≥ 1/6 ∧
  (f x y = 1/6 ↔ x = 17/4 ∧ y = 1/4) := by sorry

end f_minimum_value_l1068_106818


namespace rationalize_denominator_l1068_106820

theorem rationalize_denominator :
  (Real.sqrt 12 + Real.sqrt 5) / (Real.sqrt 3 + Real.sqrt 5) = (Real.sqrt 15 - 1) / 2 := by
  sorry

end rationalize_denominator_l1068_106820


namespace days_at_grandparents_l1068_106880

def vacation_duration : ℕ := 21  -- 3 weeks * 7 days

def travel_to_grandparents : ℕ := 1
def travel_to_brother : ℕ := 1
def stay_at_brother : ℕ := 5
def travel_to_sister : ℕ := 2
def stay_at_sister : ℕ := 5
def travel_home : ℕ := 2

def known_days : ℕ := travel_to_grandparents + travel_to_brother + stay_at_brother + travel_to_sister + stay_at_sister + travel_home

theorem days_at_grandparents :
  vacation_duration - known_days = 5 :=
by sorry

end days_at_grandparents_l1068_106880


namespace fossil_age_count_is_40_l1068_106877

/-- The number of possible 6-digit numbers formed using the digits 1 (three times), 4, 8, and 9,
    where the number must end with an even digit. -/
def fossil_age_count : ℕ :=
  let digits : List ℕ := [1, 1, 1, 4, 8, 9]
  let even_endings : List ℕ := [4, 8]
  2 * (Nat.factorial 5 / Nat.factorial 3)

theorem fossil_age_count_is_40 : fossil_age_count = 40 := by
  sorry

end fossil_age_count_is_40_l1068_106877


namespace intersection_M_N_l1068_106875

def M : Set ℝ := {-1, 0, 1, 2}
def N : Set ℝ := {x | x^2 - x - 2 < 0}

theorem intersection_M_N : M ∩ N = {0, 1} := by
  sorry

end intersection_M_N_l1068_106875


namespace range_of_a_l1068_106850

theorem range_of_a (a : ℝ) : 
  (∀ x : ℝ, x^2 - 2*x - a < 0) ∧ 
  (∃ x : ℝ, x^2 + x + 2*a - 1 ≤ 0) → 
  -1 < a ∧ a ≤ 5/8 :=
by sorry

end range_of_a_l1068_106850


namespace tens_digit_of_3_power_205_l1068_106865

theorem tens_digit_of_3_power_205 : ∃ n : ℕ, 3^205 ≡ 40 + n [ZMOD 100] ∧ n < 10 := by
  sorry

end tens_digit_of_3_power_205_l1068_106865


namespace largest_box_size_l1068_106837

theorem largest_box_size (olivia noah liam : ℕ) 
  (h_olivia : olivia = 48)
  (h_noah : noah = 60)
  (h_liam : liam = 72) :
  Nat.gcd olivia (Nat.gcd noah liam) = 12 := by
  sorry

end largest_box_size_l1068_106837


namespace real_part_of_inverse_one_minus_z_squared_l1068_106848

theorem real_part_of_inverse_one_minus_z_squared (z : ℂ) 
  (h1 : z ≠ (z.re : ℂ)) -- z is nonreal
  (h2 : Complex.abs z = 1) :
  Complex.re (1 / (1 - z^2)) = (1 - z.re^2) / 2 := by sorry

end real_part_of_inverse_one_minus_z_squared_l1068_106848


namespace large_paintings_sold_is_five_l1068_106873

/-- Represents the sale of paintings at an art show -/
structure PaintingSale where
  large_price : ℕ
  small_price : ℕ
  small_count : ℕ
  total_earnings : ℕ

/-- Calculates the number of large paintings sold -/
def large_paintings_sold (sale : PaintingSale) : ℕ :=
  (sale.total_earnings - sale.small_price * sale.small_count) / sale.large_price

/-- Theorem stating that the number of large paintings sold is 5 -/
theorem large_paintings_sold_is_five (sale : PaintingSale)
  (h1 : sale.large_price = 100)
  (h2 : sale.small_price = 80)
  (h3 : sale.small_count = 8)
  (h4 : sale.total_earnings = 1140) :
  large_paintings_sold sale = 5 := by
  sorry

end large_paintings_sold_is_five_l1068_106873


namespace last_two_digits_2018_power_2018_base_7_l1068_106870

theorem last_two_digits_2018_power_2018_base_7 : 
  (2018^2018 : ℕ) % 49 = 32 :=
sorry

end last_two_digits_2018_power_2018_base_7_l1068_106870


namespace second_subject_grade_l1068_106808

theorem second_subject_grade (grade1 grade2 grade3 average : ℚ) : 
  grade1 = 50 →
  grade3 = 90 →
  average = 70 →
  (grade1 + grade2 + grade3) / 3 = average →
  grade2 = 70 := by
sorry

end second_subject_grade_l1068_106808


namespace three_lines_intersection_l1068_106830

/-- Three lines intersect at a single point if and only if m = 9 -/
theorem three_lines_intersection (m : ℝ) : 
  (∃ (x y : ℝ), y = 2*x ∧ x + y = 3 ∧ m*x - 2*y - 5 = 0) ↔ m = 9 :=
by sorry

end three_lines_intersection_l1068_106830


namespace waiter_customer_count_l1068_106866

/-- Calculates the final number of customers after a series of arrivals and departures. -/
def finalCustomerCount (initial : ℕ) (left1 left2 : ℕ) (arrived1 arrived2 : ℕ) : ℕ :=
  initial - left1 + arrived1 + arrived2 - left2

/-- Theorem stating that given the specific customer movements, the final count is 14. -/
theorem waiter_customer_count : 
  finalCustomerCount 13 5 6 4 8 = 14 := by
  sorry

end waiter_customer_count_l1068_106866


namespace custom_mult_theorem_l1068_106856

/-- Custom multiplication operation -/
def custom_mult (m n : ℝ) : ℝ := 2 * m - 3 * n

/-- Theorem stating that if x satisfies the given condition, then x = 7 -/
theorem custom_mult_theorem (x : ℝ) : 
  (∀ m n : ℝ, custom_mult m n = 2 * m - 3 * n) → 
  custom_mult x 7 = custom_mult 7 x → 
  x = 7 := by
  sorry

end custom_mult_theorem_l1068_106856


namespace sum_of_roots_l1068_106809

theorem sum_of_roots (a b : ℝ) : 
  a * (a - 4) = 21 → 
  b * (b - 4) = 21 → 
  a ≠ b → 
  a + b = 4 := by
sorry

end sum_of_roots_l1068_106809


namespace cos_48_degrees_l1068_106859

theorem cos_48_degrees : Real.cos (48 * π / 180) = (Real.sqrt 3 - 1) / 2 := by
  sorry

end cos_48_degrees_l1068_106859


namespace cone_lateral_surface_angle_l1068_106833

/-- Given a cone where the total surface area is three times its base area,
    the central angle of the sector in the lateral surface development diagram is 180 degrees. -/
theorem cone_lateral_surface_angle (r l : ℝ) (h : r > 0) (h' : l > 0) : 
  (π * r^2 + π * r * l = 3 * π * r^2) → 
  (2 * π * r / l) * (180 / π) = 180 := by
  sorry

end cone_lateral_surface_angle_l1068_106833


namespace opposite_of_one_over_23_l1068_106816

theorem opposite_of_one_over_23 : 
  -(1 / 23) = -1 / 23 := by sorry

end opposite_of_one_over_23_l1068_106816


namespace min_value_of_fraction_lower_bound_achievable_l1068_106874

theorem min_value_of_fraction (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (hsum : a + b + c = 3) :
  (a + b) / (a * b * c) ≥ 16 / 9 :=
sorry

theorem lower_bound_achievable :
  ∃ (a b c : ℝ), a > 0 ∧ b > 0 ∧ c > 0 ∧ a + b + c = 3 ∧ (a + b) / (a * b * c) = 16 / 9 :=
sorry

end min_value_of_fraction_lower_bound_achievable_l1068_106874


namespace range_of_g_l1068_106804

def f (x : ℝ) : ℝ := 2 * x + 3

def g (x : ℝ) : ℝ := f (f (f (f x)))

theorem range_of_g :
  ∀ x : ℝ, -1 ≤ x ∧ x ≤ 3 → 29 ≤ g x ∧ g x ≤ 93 :=
by sorry

end range_of_g_l1068_106804


namespace smallest_x_quadratic_equation_l1068_106832

theorem smallest_x_quadratic_equation :
  let f : ℝ → ℝ := λ x => 4*x^2 + 6*x + 1
  ∃ x : ℝ, (f x = 5) ∧ (∀ y : ℝ, f y = 5 → x ≤ y) ∧ x = -2 :=
by sorry

end smallest_x_quadratic_equation_l1068_106832


namespace triangle_max_sin_sum_l1068_106879

theorem triangle_max_sin_sum (A B C : ℝ) (a b c : ℝ) : 
  0 < A ∧ 0 < B ∧ 0 < C ∧ 
  A + B + C = π ∧
  c * Real.sin A = Real.sqrt 3 * a * Real.cos C →
  (∃ (max : ℝ), max = Real.sqrt 3 ∧ 
    ∀ A' B' : ℝ, 0 < A' ∧ 0 < B' ∧ A' + B' = 2*π/3 →
      Real.sin A' + Real.sin B' ≤ max) :=
sorry

end triangle_max_sin_sum_l1068_106879


namespace imaginary_part_of_i_cubed_plus_one_l1068_106894

-- Define the imaginary unit i
noncomputable def i : ℂ := Complex.I

-- State the theorem
theorem imaginary_part_of_i_cubed_plus_one (h : i^2 = -1) :
  (i * (i^3 + 1)).im = 1 := by sorry

end imaginary_part_of_i_cubed_plus_one_l1068_106894


namespace lidia_app_purchase_l1068_106889

/-- Proves that Lidia will be left with $15 after purchasing apps with a discount --/
theorem lidia_app_purchase (app_cost : ℝ) (num_apps : ℕ) (budget : ℝ) (discount_rate : ℝ) :
  app_cost = 4 →
  num_apps = 15 →
  budget = 66 →
  discount_rate = 0.15 →
  budget - (num_apps * app_cost * (1 - discount_rate)) = 15 :=
by
  sorry

#check lidia_app_purchase

end lidia_app_purchase_l1068_106889


namespace product_digit_sum_l1068_106846

/-- The number of digits in each factor -/
def n : ℕ := 2012

/-- The first factor in the multiplication -/
def first_factor : ℕ := (10^n - 1) * 4 / 9

/-- The second factor in the multiplication -/
def second_factor : ℕ := 10^n - 1

/-- Function to calculate the sum of digits of a natural number -/
def sum_of_digits (k : ℕ) : ℕ :=
  if k < 10 then k else k % 10 + sum_of_digits (k / 10)

/-- The main theorem to be proved -/
theorem product_digit_sum :
  sum_of_digits (first_factor * second_factor) = 18108 := by
  sorry

end product_digit_sum_l1068_106846


namespace cookie_brownie_difference_l1068_106861

/-- Represents the daily consumption of cookies and brownies --/
structure DailyConsumption where
  cookies : Nat
  brownies : Nat

/-- Calculates the remaining items after a week of consumption --/
def remainingItems (initial : Nat) (daily : List Nat) : Nat :=
  max (initial - daily.sum) 0

theorem cookie_brownie_difference :
  let initialCookies : Nat := 60
  let initialBrownies : Nat := 10
  let weeklyConsumption : List DailyConsumption := [
    ⟨2, 1⟩, ⟨4, 2⟩, ⟨3, 1⟩, ⟨5, 1⟩, ⟨4, 3⟩, ⟨3, 2⟩, ⟨2, 1⟩
  ]
  let cookiesLeft := remainingItems initialCookies (weeklyConsumption.map DailyConsumption.cookies)
  let browniesLeft := remainingItems initialBrownies (weeklyConsumption.map DailyConsumption.brownies)
  cookiesLeft - browniesLeft = 37 := by
  sorry

end cookie_brownie_difference_l1068_106861


namespace right_handed_players_count_l1068_106847

theorem right_handed_players_count (total_players throwers : ℕ) 
  (h1 : total_players = 70)
  (h2 : throwers = 46)
  (h3 : throwers ≤ total_players)
  (h4 : (total_players - throwers) % 3 = 0) -- Ensure non-throwers are divisible by 3
  : (throwers + (2 * (total_players - throwers) / 3)) = 62 := by
  sorry

end right_handed_players_count_l1068_106847


namespace six_digit_divisibility_l1068_106838

theorem six_digit_divisibility (a b : Nat) : 
  (a < 10 ∧ b < 10) →  -- Ensure a and b are single digits
  (201000 + 100 * a + 10 * b + 7) % 11 = 0 →  -- Divisible by 11
  (201000 + 100 * a + 10 * b + 7) % 13 = 0 →  -- Divisible by 13
  10 * a + b = 48 := by
sorry

end six_digit_divisibility_l1068_106838


namespace factorial_sum_power_of_two_solutions_l1068_106831

def is_solution (a b c n : ℕ) : Prop :=
  Nat.factorial a + Nat.factorial b + Nat.factorial c = 2^n

theorem factorial_sum_power_of_two_solutions :
  ∀ a b c n : ℕ,
    is_solution a b c n ↔
      ((a, b, c) = (1, 1, 2) ∧ n = 2) ∨
      ((a, b, c) = (1, 1, 3) ∧ n = 3) ∨
      ((a, b, c) = (2, 3, 4) ∧ n = 5) ∨
      ((a, b, c) = (2, 3, 5) ∧ n = 7) :=
by sorry

end factorial_sum_power_of_two_solutions_l1068_106831


namespace inequality_proof_l1068_106887

theorem inequality_proof (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (h_abc : a * b * c = 1) :
  1 / (a^3 * (b + c)) + 1 / (b^3 * (a + c)) + 1 / (c^3 * (a + b)) ≥ 3/2 := by
  sorry

end inequality_proof_l1068_106887


namespace snail_climb_problem_l1068_106864

/-- The number of days required for a snail to climb out of a well -/
def days_to_climb (well_height : ℕ) (day_climb : ℕ) (night_slip : ℕ) : ℕ :=
  sorry

theorem snail_climb_problem :
  let well_height : ℕ := 12
  let day_climb : ℕ := 3
  let night_slip : ℕ := 2
  days_to_climb well_height day_climb night_slip = 10 := by sorry

end snail_climb_problem_l1068_106864


namespace females_who_chose_malt_l1068_106817

/-- Represents the number of cheerleaders who chose each drink -/
structure CheerleaderChoices where
  coke : ℕ
  malt : ℕ

/-- Represents the gender distribution of cheerleaders -/
structure CheerleaderGenders where
  males : ℕ
  females : ℕ

theorem females_who_chose_malt 
  (choices : CheerleaderChoices)
  (genders : CheerleaderGenders)
  (h1 : genders.males = 10)
  (h2 : genders.females = 16)
  (h3 : choices.malt = 2 * choices.coke)
  (h4 : choices.coke + choices.malt = genders.males + genders.females)
  (h5 : choices.malt ≥ genders.males)
  (h6 : genders.males = 6) :
  choices.malt - genders.males = 10 := by
sorry

end females_who_chose_malt_l1068_106817


namespace geometric_series_common_ratio_l1068_106853

theorem geometric_series_common_ratio : 
  let a₁ := 7 / 8
  let a₂ := -5 / 12
  let a₃ := 25 / 144
  let r := a₂ / a₁
  r = -10 / 21 := by sorry

end geometric_series_common_ratio_l1068_106853


namespace min_value_expression_l1068_106805

/-- Given positive real numbers a and b satisfying a + 3b = 7, 
    the expression 1/(1+a) + 4/(2+b) has a minimum value of (13 + 4√3)/14 -/
theorem min_value_expression (a b : ℝ) (ha : a > 0) (hb : b > 0) (heq : a + 3*b = 7) :
  (1 / (1 + a) + 4 / (2 + b)) ≥ (13 + 4 * Real.sqrt 3) / 14 ∧
  ∃ (a₀ b₀ : ℝ), a₀ > 0 ∧ b₀ > 0 ∧ a₀ + 3*b₀ = 7 ∧
    1 / (1 + a₀) + 4 / (2 + b₀) = (13 + 4 * Real.sqrt 3) / 14 :=
by sorry

end min_value_expression_l1068_106805


namespace two_number_difference_l1068_106888

theorem two_number_difference (a b : ℕ) : 
  a + b = 22305 →
  a % 5 = 0 →
  b = (a / 10) + 3 →
  a - b = 14872 :=
by sorry

end two_number_difference_l1068_106888


namespace cone_lateral_area_l1068_106844

/-- The lateral area of a cone with base radius 1 and height √3 is 2π. -/
theorem cone_lateral_area : 
  let r : ℝ := 1
  let h : ℝ := Real.sqrt 3
  let l : ℝ := Real.sqrt (r^2 + h^2)
  let S : ℝ := π * r * l
  S = 2 * π :=
by sorry

end cone_lateral_area_l1068_106844


namespace division_of_fractions_l1068_106862

theorem division_of_fractions : (5 : ℚ) / 6 / ((9 : ℚ) / 10) = 25 / 27 := by
  sorry

end division_of_fractions_l1068_106862


namespace rational_sum_and_square_sum_integer_implies_integer_l1068_106872

theorem rational_sum_and_square_sum_integer_implies_integer (a b : ℚ) 
  (h1 : ∃ n : ℤ, (a + b : ℚ) = n)
  (h2 : ∃ m : ℤ, (a^2 + b^2 : ℚ) = m) :
  ∃ (x y : ℤ), (a = x ∧ b = y) :=
sorry

end rational_sum_and_square_sum_integer_implies_integer_l1068_106872


namespace largest_n_with_special_divisor_property_l1068_106841

theorem largest_n_with_special_divisor_property : ∃ N : ℕ, 
  (∀ m : ℕ, m > N → ¬(∃ d₁ d₂ d₃ : ℕ, 
    d₁ ∣ m ∧ d₂ ∣ m ∧ d₃ ∣ m ∧
    (∀ x : ℕ, x ∣ m → x = 1 ∨ x ≥ d₁) ∧
    (∀ x : ℕ, x ∣ m → x = 1 ∨ x = d₁ ∨ x ≥ d₂) ∧
    (∃ y z : ℕ, y ∣ m ∧ z ∣ m ∧ y > d₃ ∧ z > y) ∧
    d₃ = 21 * d₁)) ∧
  (∃ d₁ d₂ d₃ : ℕ, 
    d₁ ∣ N ∧ d₂ ∣ N ∧ d₃ ∣ N ∧
    (∀ x : ℕ, x ∣ N → x = 1 ∨ x ≥ d₁) ∧
    (∀ x : ℕ, x ∣ N → x = 1 ∨ x = d₁ ∨ x ≥ d₂) ∧
    (∃ y z : ℕ, y ∣ N ∧ z ∣ N ∧ y > d₃ ∧ z > y) ∧
    d₃ = 21 * d₁) ∧
  N = 441 := by
  sorry

end largest_n_with_special_divisor_property_l1068_106841

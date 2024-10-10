import Mathlib

namespace range_of_m_l3640_364078

theorem range_of_m (x y m : ℝ) (hx : x > 0) (hy : y > 0) 
  (h : ∀ x y, x > 0 → y > 0 → (2 * y / x + 8 * x / y ≥ m^2 + 2*m)) : 
  m ∈ Set.Icc (-4 : ℝ) 2 := by
sorry

end range_of_m_l3640_364078


namespace haircut_tip_percentage_l3640_364091

theorem haircut_tip_percentage (womens_haircut_cost : ℝ) (childrens_haircut_cost : ℝ) 
  (num_children : ℕ) (tip_amount : ℝ) :
  womens_haircut_cost = 48 →
  childrens_haircut_cost = 36 →
  num_children = 2 →
  tip_amount = 24 →
  (tip_amount / (womens_haircut_cost + num_children * childrens_haircut_cost)) * 100 = 20 := by
  sorry

end haircut_tip_percentage_l3640_364091


namespace proportion_fourth_number_l3640_364025

theorem proportion_fourth_number (x y : ℝ) : 
  (0.75 : ℝ) / x = 5 / y → x = 1.05 → y = 7 := by sorry

end proportion_fourth_number_l3640_364025


namespace equation_solution_l3640_364097

theorem equation_solution : ∃! x : ℝ, (x^2 - x - 2) / (x + 2) = x - 1 := by
  sorry

end equation_solution_l3640_364097


namespace clock_angle_at_2_30_l3640_364083

/-- The number of degrees in a circle -/
def circle_degrees : ℕ := 360

/-- The number of hours on a clock face -/
def clock_hours : ℕ := 12

/-- The number of minutes in an hour -/
def minutes_per_hour : ℕ := 60

/-- The angle moved by the hour hand in one hour -/
def hour_hand_degrees_per_hour : ℚ := circle_degrees / clock_hours

/-- The angle moved by the minute hand in one minute -/
def minute_hand_degrees_per_minute : ℚ := circle_degrees / minutes_per_hour

/-- The position of the hour hand at 2:30 -/
def hour_hand_position : ℚ := 2.5 * hour_hand_degrees_per_hour

/-- The position of the minute hand at 2:30 -/
def minute_hand_position : ℚ := 30 * minute_hand_degrees_per_minute

/-- The angle between the hour hand and minute hand at 2:30 -/
def angle_between_hands : ℚ := |minute_hand_position - hour_hand_position|

theorem clock_angle_at_2_30 :
  min angle_between_hands (circle_degrees - angle_between_hands) = 105 :=
sorry

end clock_angle_at_2_30_l3640_364083


namespace sum_of_fourth_powers_l3640_364050

theorem sum_of_fourth_powers (a b c : ℝ) 
  (sum_zero : a + b + c = 0) 
  (sum_squares : a^2 + b^2 + c^2 = 4) : 
  a^4 + b^4 + c^4 = 8 := by
  sorry

end sum_of_fourth_powers_l3640_364050


namespace vector_problem_l3640_364094

/-- Given a vector a and a unit vector b not parallel to the x-axis such that a · b = √3, prove that b = (1/2, √3/2) -/
theorem vector_problem (a b : ℝ × ℝ) : 
  a = (Real.sqrt 3, 1) →
  ‖b‖ = 1 →
  b.1 ≠ b.2 →
  a.1 * b.1 + a.2 * b.2 = Real.sqrt 3 →
  b = (1/2, Real.sqrt 3 / 2) := by
sorry

end vector_problem_l3640_364094


namespace units_digit_sum_series_l3640_364024

def factorial (n : ℕ) : ℕ := (List.range n).foldl (·*·) 1

def units_digit (n : ℕ) : ℕ := n % 10

def sum_series : ℕ := 
  (units_digit (factorial 1)) + 
  (units_digit ((factorial 2)^2)) + 
  (units_digit (factorial 3)) + 
  (units_digit ((factorial 4)^2)) + 
  (units_digit (factorial 5)) + 
  (units_digit ((factorial 6)^2)) + 
  (units_digit (factorial 7)) + 
  (units_digit ((factorial 8)^2)) + 
  (units_digit (factorial 9)) + 
  (units_digit ((factorial 10)^2))

theorem units_digit_sum_series : units_digit sum_series = 7 := by
  sorry

end units_digit_sum_series_l3640_364024


namespace sine_function_property_l3640_364068

/-- Given a function f(x) = sin(ωx) where ω > 0, if f(x - 1/2) = f(x + 1/2) for all real x,
    and f(-1/4) = a, then f(9/4) = -a -/
theorem sine_function_property (ω : ℝ) (a : ℝ) (h_ω : ω > 0) :
  (∀ x : ℝ, Real.sin (ω * (x - 1/2)) = Real.sin (ω * (x + 1/2))) →
  Real.sin (ω * (-1/4)) = a →
  Real.sin (ω * (9/4)) = -a :=
by sorry

end sine_function_property_l3640_364068


namespace square_ratio_sum_l3640_364087

theorem square_ratio_sum (area_ratio : ℚ) (a b c : ℕ) : 
  area_ratio = 75 / 128 →
  (∃ (side_ratio : ℝ), side_ratio = Real.sqrt (area_ratio) ∧ 
    side_ratio = a * Real.sqrt b / c) →
  a + b + c = 27 := by
sorry

end square_ratio_sum_l3640_364087


namespace flagpole_height_l3640_364030

/-- Given a lamppost height and shadow length, calculate the height of another object with a known shadow length -/
theorem flagpole_height
  (lamppost_height : ℝ) 
  (lamppost_shadow : ℝ) 
  (flagpole_shadow : ℝ) 
  (h1 : lamppost_height = 50)
  (h2 : lamppost_shadow = 12)
  (h3 : flagpole_shadow = 18 / 12)  -- Convert 18 inches to feet
  : ∃ (flagpole_height : ℝ), 
    flagpole_height * lamppost_shadow = lamppost_height * flagpole_shadow ∧ 
    flagpole_height * 12 = 75 :=
by sorry

end flagpole_height_l3640_364030


namespace jack_initial_money_l3640_364072

def initial_bottles : ℕ := 4
def bottle_cost : ℚ := 2
def cheese_weight : ℚ := 1/2
def cheese_cost_per_pound : ℚ := 10
def remaining_money : ℚ := 71

theorem jack_initial_money :
  let total_bottles := initial_bottles + 2 * initial_bottles
  let water_cost := total_bottles * bottle_cost
  let cheese_cost := cheese_weight * cheese_cost_per_pound
  let total_spent := water_cost + cheese_cost
  total_spent + remaining_money = 100 := by sorry

end jack_initial_money_l3640_364072


namespace goldfish_equality_exists_l3640_364061

theorem goldfish_equality_exists : ∃ n : ℕ+, 
  8 * (5 : ℝ)^n.val = 200 * (3 : ℝ)^n.val + 20 * ((3 : ℝ)^n.val - 1) / 2 := by
  sorry

end goldfish_equality_exists_l3640_364061


namespace circle_area_in_square_l3640_364064

theorem circle_area_in_square (square_area : ℝ) (h : square_area = 400) :
  let square_side := Real.sqrt square_area
  let circle_radius := square_side / 2
  let circle_area := Real.pi * circle_radius ^ 2
  circle_area = 100 * Real.pi := by sorry

end circle_area_in_square_l3640_364064


namespace exists_number_with_properties_l3640_364079

/-- Sum of digits of a natural number -/
def sum_of_digits (n : ℕ) : ℕ := sorry

theorem exists_number_with_properties : ∃ n : ℕ, 
  2019 ∣ n ∧ 2019 ∣ sum_of_digits n := by sorry

end exists_number_with_properties_l3640_364079


namespace opposite_sides_range_l3640_364013

/-- A point in 2D space -/
structure Point2D where
  x : ℝ
  y : ℝ

/-- Determines if two points are on opposite sides of a line -/
def oppositeSides (p1 p2 : Point2D) (a : ℝ) : Prop :=
  (3 * p1.x - 2 * p1.y + a) * (3 * p2.x - 2 * p2.y + a) < 0

/-- The theorem stating the range of 'a' for which the given points are on opposite sides of the line -/
theorem opposite_sides_range :
  ∀ a : ℝ, 
    oppositeSides (Point2D.mk 3 1) (Point2D.mk (-4) 6) a ↔ -7 < a ∧ a < 24 := by
  sorry

end opposite_sides_range_l3640_364013


namespace stratified_sampling_total_students_l3640_364037

theorem stratified_sampling_total_students 
  (total_sample : ℕ) 
  (grade_10_sample : ℕ) 
  (grade_11_sample : ℕ) 
  (grade_12_students : ℕ) 
  (h1 : total_sample = 100)
  (h2 : grade_10_sample = 24)
  (h3 : grade_11_sample = 26)
  (h4 : grade_12_students = 600)
  (h5 : grade_12_students * total_sample = 
        (total_sample - grade_10_sample - grade_11_sample) * total_students) : 
  total_students = 1200 := by
  sorry

end stratified_sampling_total_students_l3640_364037


namespace equilateral_triangle_roots_l3640_364076

/-- Given complex roots z₁ and z₂ of z² + az + b = 0 where a and b are complex,
    and z₂ = ω z₁ with ω = e^(2πi/3), prove that a²/b = 1 -/
theorem equilateral_triangle_roots (a b z₁ z₂ : ℂ) : 
  z₁^2 + a*z₁ + b = 0 →
  z₂^2 + a*z₂ + b = 0 →
  z₂ = (Complex.exp (2 * Complex.I * Real.pi / 3)) * z₁ →
  a^2 / b = 1 := by
  sorry

end equilateral_triangle_roots_l3640_364076


namespace city_wall_length_l3640_364071

/-- Represents a city layout with 5 congruent squares in an isosceles cross shape -/
structure CityLayout where
  square_side : ℝ
  num_squares : Nat
  num_squares_eq : num_squares = 5

/-- Calculates the perimeter of the city layout -/
def perimeter (city : CityLayout) : ℝ :=
  12 * city.square_side

/-- Calculates the area of the city layout -/
def area (city : CityLayout) : ℝ :=
  city.num_squares * city.square_side^2

/-- Theorem stating that if the perimeter equals the area, then the perimeter is 28.8 km -/
theorem city_wall_length (city : CityLayout) :
  perimeter city = area city → perimeter city = 28.8 := by
  sorry


end city_wall_length_l3640_364071


namespace johns_apartment_paint_area_l3640_364073

/-- Represents the dimensions of a bedroom -/
structure BedroomDimensions where
  length : ℝ
  width : ℝ
  height : ℝ

/-- Calculates the total area to be painted in a single bedroom -/
def area_to_paint (dim : BedroomDimensions) (unpainted_area : ℝ) : ℝ :=
  2 * (dim.length * dim.height + dim.width * dim.height) + 
  dim.length * dim.width - unpainted_area

/-- Theorem stating the total area to be painted in John's apartment -/
theorem johns_apartment_paint_area :
  let bedroom_dim : BedroomDimensions := ⟨15, 12, 10⟩
  let unpainted_area : ℝ := 70
  let num_bedrooms : ℕ := 2
  num_bedrooms * (area_to_paint bedroom_dim unpainted_area) = 1300 := by
  sorry


end johns_apartment_paint_area_l3640_364073


namespace min_value_of_trig_function_l3640_364098

open Real

theorem min_value_of_trig_function :
  ∃ (x : ℝ), ∀ (y : ℝ), 2 * sin (π / 3 - x) - cos (π / 6 + x) ≥ -1 := by
  sorry

end min_value_of_trig_function_l3640_364098


namespace single_intersection_l3640_364063

def f (a x : ℝ) : ℝ := (a - 1) * x^2 - 4 * x + 2 * a

theorem single_intersection (a : ℝ) : 
  (∃! x, f a x = 0) ↔ (a = -1 ∨ a = 2 ∨ a = 1) := by
  sorry

end single_intersection_l3640_364063


namespace intersection_implies_a_value_l3640_364040

def A : Set ℝ := {x | x^2 - 2*x - 3 ≤ 0}

def B (a : ℝ) : Set ℝ := {x | x ≥ a}

theorem intersection_implies_a_value (a : ℝ) :
  A ∩ B a = {3} → a = 3 := by
  sorry

end intersection_implies_a_value_l3640_364040


namespace inequality_system_solutions_l3640_364077

theorem inequality_system_solutions (a : ℝ) : 
  (∃ (x₁ x₂ x₃ : ℤ), x₁ > 4 ∧ x₁ ≤ a ∧ 
                      x₂ > 4 ∧ x₂ ≤ a ∧ 
                      x₃ > 4 ∧ x₃ ≤ a ∧ 
                      x₁ ≠ x₂ ∧ x₁ ≠ x₃ ∧ x₂ ≠ x₃ ∧
                      (∀ (y : ℤ), y > 4 ∧ y ≤ a → y = x₁ ∨ y = x₂ ∨ y = x₃)) →
  7 ≤ a ∧ a < 8 :=
by sorry

end inequality_system_solutions_l3640_364077


namespace max_min_m_values_l3640_364084

/-- Given conditions p and q, find the maximum and minimum values of m -/
theorem max_min_m_values (m : ℝ) (h_m_pos : m > 0) : 
  (∀ x : ℝ, |x| ≤ m → -1 ≤ x ∧ x ≤ 4) ∧ 
  (∀ x : ℝ, -1 ≤ x ∧ x ≤ 4 → |x| ≤ m) → 
  m = 4 := by sorry

end max_min_m_values_l3640_364084


namespace solve_for_y_l3640_364049

theorem solve_for_y : ∃ y : ℝ, (3 * y) / 4 = 15 ∧ y = 20 := by sorry

end solve_for_y_l3640_364049


namespace complex_sum_problem_l3640_364000

theorem complex_sum_problem (p q r s t u : ℝ) : 
  s = 5 →
  t = -p - r →
  (p + q * Complex.I) + (r + s * Complex.I) + (t + u * Complex.I) = -6 * Complex.I →
  u + q = -11 := by
  sorry

end complex_sum_problem_l3640_364000


namespace candidates_scientific_notation_l3640_364067

/-- The number of candidates for the high school entrance examination in Guangdong Province in 2023 -/
def candidates : ℝ := 1108200

/-- The scientific notation representation of the number of candidates -/
def scientific_notation : ℝ := 1.1082 * (10 ^ 6)

/-- Theorem stating that the number of candidates is equal to its scientific notation representation -/
theorem candidates_scientific_notation : candidates = scientific_notation := by
  sorry

end candidates_scientific_notation_l3640_364067


namespace shaded_quadrilateral_area_l3640_364099

theorem shaded_quadrilateral_area : 
  let small_square_side : ℝ := 3
  let medium_square_side : ℝ := 5
  let large_square_side : ℝ := 7
  let total_base : ℝ := small_square_side + medium_square_side + large_square_side
  let diagonal_slope : ℝ := large_square_side / total_base
  let small_triangle_height : ℝ := small_square_side * diagonal_slope
  let medium_triangle_height : ℝ := (small_square_side + medium_square_side) * diagonal_slope
  let trapezoid_area : ℝ := (medium_square_side * (small_triangle_height + medium_triangle_height)) / 2
  trapezoid_area = 12.825 := by
  sorry

end shaded_quadrilateral_area_l3640_364099


namespace icosahedron_painting_ways_l3640_364041

/-- Represents a regular icosahedron -/
structure Icosahedron where
  faces : Nat
  rotationalSymmetries : Nat

/-- Represents the number of ways to paint an icosahedron -/
def paintingWays (i : Icosahedron) (colors : Nat) : Nat :=
  Nat.factorial (colors - 1) / i.rotationalSymmetries

/-- Theorem stating the number of distinguishable ways to paint an icosahedron -/
theorem icosahedron_painting_ways (i : Icosahedron) (h1 : i.faces = 20) (h2 : i.rotationalSymmetries = 60) :
  paintingWays i 20 = Nat.factorial 19 / 60 := by
  sorry

#check icosahedron_painting_ways

end icosahedron_painting_ways_l3640_364041


namespace yellow_balls_count_l3640_364086

theorem yellow_balls_count (red blue yellow green : ℕ) : 
  red + blue + yellow + green = 531 →
  red + blue = yellow + green + 31 →
  yellow = green + 22 →
  yellow = 136 := by
sorry

end yellow_balls_count_l3640_364086


namespace kolya_always_wins_l3640_364043

/-- Represents a player's move in the game -/
inductive Move
| ChangeA (delta : Int) : Move
| ChangeB (delta : Int) : Move

/-- Represents the state of the game -/
structure GameState where
  a : Int
  b : Int

/-- Defines a valid move for Petya -/
def validPetyaMove (m : Move) : Prop :=
  match m with
  | Move.ChangeA delta => delta = 1 ∨ delta = -1
  | Move.ChangeB delta => delta = 1 ∨ delta = -1

/-- Defines a valid move for Kolya -/
def validKolyaMove (m : Move) : Prop :=
  match m with
  | Move.ChangeA delta => delta = 1 ∨ delta = -1 ∨ delta = 3 ∨ delta = -3
  | Move.ChangeB delta => delta = 1 ∨ delta = -1 ∨ delta = 3 ∨ delta = -3

/-- Applies a move to the game state -/
def applyMove (state : GameState) (m : Move) : GameState :=
  match m with
  | Move.ChangeA delta => { state with a := state.a + delta }
  | Move.ChangeB delta => { state with b := state.b + delta }

/-- Checks if the polynomial has integer roots -/
def hasIntegerRoots (state : GameState) : Prop :=
  ∃ x y : Int, x^2 + state.a * x + state.b = 0 ∧ y^2 + state.a * y + state.b = 0 ∧ x ≠ y

/-- Theorem stating Kolya can always win -/
theorem kolya_always_wins :
  ∀ (initial : GameState),
  ∃ (kolyaMoves : List Move),
    (∀ m ∈ kolyaMoves, validKolyaMove m) ∧
    (∀ (petyaMoves : List Move),
      (petyaMoves.length = kolyaMoves.length) →
      (∀ m ∈ petyaMoves, validPetyaMove m) →
      ∃ (finalState : GameState),
        finalState = (kolyaMoves.zip petyaMoves).foldl
          (λ state (km, pm) => applyMove (applyMove state pm) km)
          initial ∧
        hasIntegerRoots finalState) :=
sorry

end kolya_always_wins_l3640_364043


namespace workshop_workers_count_l3640_364047

/-- Proves that the total number of workers in a workshop is 14, given specific salary conditions -/
theorem workshop_workers_count : ∀ (W : ℕ) (N : ℕ),
  (W : ℚ) * 8000 = 70000 + (N : ℚ) * 6000 →
  W = 7 + N →
  W = 14 :=
by
  sorry

end workshop_workers_count_l3640_364047


namespace expression_evaluation_l3640_364060

theorem expression_evaluation : 5 * 401 + 4 * 401 + 3 * 401 + 400 = 5212 := by
  sorry

end expression_evaluation_l3640_364060


namespace systematic_sampling_probability_l3640_364026

/-- Represents a systematic sampling scenario -/
structure SystematicSampling where
  population_size : ℕ
  sample_size : ℕ
  h_pop_size : population_size > 0
  h_sample_size : sample_size > 0
  h_sample_le_pop : sample_size ≤ population_size

/-- The probability of an individual being selected in systematic sampling -/
def selection_probability (s : SystematicSampling) : ℚ :=
  s.sample_size / s.population_size

/-- Theorem stating the probability of selection in the given scenario -/
theorem systematic_sampling_probability 
  (s : SystematicSampling) 
  (h_pop : s.population_size = 42) 
  (h_sample : s.sample_size = 10) : 
  selection_probability s = 5 / 21 := by
  sorry

end systematic_sampling_probability_l3640_364026


namespace non_similar_1200_pointed_stars_l3640_364090

/-- Definition of a regular n-pointed star (placeholder) -/
def RegularStar (n : ℕ) : Type := sorry

/-- Counts the number of non-similar regular n-pointed stars -/
def countNonSimilarStars (n : ℕ) : ℕ := sorry

/-- Euler's totient function -/
def phi : ℕ → ℕ := sorry

theorem non_similar_1200_pointed_stars :
  countNonSimilarStars 1200 = 160 :=
by sorry

end non_similar_1200_pointed_stars_l3640_364090


namespace max_value_sqrt_sum_l3640_364081

theorem max_value_sqrt_sum (a b : Real) 
  (ha : 0 ≤ a ∧ a ≤ 1) (hb : 0 ≤ b ∧ b ≤ 1) : 
  Real.sqrt (a * b) + Real.sqrt ((1 - a) * (1 - b)) ≤ 1 := by
  sorry

end max_value_sqrt_sum_l3640_364081


namespace pen_price_calculation_l3640_364023

theorem pen_price_calculation (total_cost : ℝ) (num_pens : ℕ) (num_pencils : ℕ) (pencil_price : ℝ) :
  total_cost = 450 →
  num_pens = 30 →
  num_pencils = 75 →
  pencil_price = 2 →
  (total_cost - (num_pencils : ℝ) * pencil_price) / (num_pens : ℝ) = 10 := by
  sorry

end pen_price_calculation_l3640_364023


namespace flight_distance_calculation_l3640_364039

/-- Calculates the total flight distance with headwinds and tailwinds -/
def total_flight_distance (spain_russia : ℝ) (spain_germany : ℝ) (germany_france : ℝ) (france_russia : ℝ) 
  (headwind_increase : ℝ) (tailwind_decrease : ℝ) : ℝ :=
  let france_russia_with_headwind := france_russia * (1 + headwind_increase)
  let russia_spain_via_germany := (spain_russia + spain_germany) * (1 - tailwind_decrease)
  france_russia_with_headwind + russia_spain_via_germany

/-- The total flight distance is approximately 14863.98 km -/
theorem flight_distance_calculation :
  let spain_russia : ℝ := 7019
  let spain_germany : ℝ := 1615
  let germany_france : ℝ := 956
  let france_russia : ℝ := 6180
  let headwind_increase : ℝ := 0.05
  let tailwind_decrease : ℝ := 0.03
  abs (total_flight_distance spain_russia spain_germany germany_france france_russia 
    headwind_increase tailwind_decrease - 14863.98) < 0.01 := by
  sorry


end flight_distance_calculation_l3640_364039


namespace cube_sum_power_of_two_l3640_364069

theorem cube_sum_power_of_two (x y : ℤ) :
  x^3 + y^3 = 2^30 ↔ (x = 0 ∧ y = 2^10) ∨ (x = 2^10 ∧ y = 0) := by
  sorry

end cube_sum_power_of_two_l3640_364069


namespace susan_apples_l3640_364032

/-- The number of apples each person has -/
structure Apples where
  phillip : ℝ
  ben : ℝ
  tom : ℝ
  susan : ℝ

/-- The conditions of the problem -/
def apple_conditions (a : Apples) : Prop :=
  a.phillip = 38.25 ∧
  a.ben = a.phillip + 8.5 ∧
  a.tom = (3/8) * a.ben ∧
  a.susan = (1/2) * a.tom + 7

/-- The theorem stating that under the given conditions, Susan has 15.765625 apples -/
theorem susan_apples (a : Apples) (h : apple_conditions a) : a.susan = 15.765625 := by
  sorry

end susan_apples_l3640_364032


namespace perpendicular_vectors_l3640_364004

def a : ℝ × ℝ := (1, 2)
def b (m : ℝ) : ℝ × ℝ := (2*m - 1, -1)

theorem perpendicular_vectors (m : ℝ) : 
  (a.1 * (b m).1 + a.2 * (b m).2 = 0) → m = 3/2 :=
by sorry

end perpendicular_vectors_l3640_364004


namespace travel_options_l3640_364009

theorem travel_options (train_services : ℕ) (ferry_services : ℕ) : 
  train_services = 3 → ferry_services = 2 → train_services * ferry_services = 6 := by
  sorry

#check travel_options

end travel_options_l3640_364009


namespace sin_equality_integer_solutions_l3640_364093

theorem sin_equality_integer_solutions (m : ℤ) :
  -180 ≤ m ∧ m ≤ 180 ∧ Real.sin (m * π / 180) = Real.sin (750 * π / 180) →
  m = 30 ∨ m = 150 := by
sorry

end sin_equality_integer_solutions_l3640_364093


namespace quadratic_root_zero_l3640_364092

/-- 
Given a quadratic equation (k+2)x^2 + 6x + k^2 + k - 2 = 0 where 0 is one of its roots,
prove that k = 1.
-/
theorem quadratic_root_zero (k : ℝ) : 
  (∀ x, (k + 2) * x^2 + 6 * x + k^2 + k - 2 = 0 ↔ x = 0 ∨ x = -(6 / (k + 2))) →
  k + 2 ≠ 0 →
  k = 1 := by
  sorry

end quadratic_root_zero_l3640_364092


namespace modulo_equivalence_problem_l3640_364003

theorem modulo_equivalence_problem : ∃! n : ℕ, 0 ≤ n ∧ n ≤ 14 ∧ n ≡ 15478 [MOD 15] ∧ n = 13 := by
  sorry

end modulo_equivalence_problem_l3640_364003


namespace alcohol_mixture_concentration_l3640_364045

/-- Proves that the new concentration of the mixture is 29% given the initial conditions --/
theorem alcohol_mixture_concentration
  (vessel1_capacity : ℝ)
  (vessel1_alcohol_percentage : ℝ)
  (vessel2_capacity : ℝ)
  (vessel2_alcohol_percentage : ℝ)
  (total_liquid : ℝ)
  (final_vessel_capacity : ℝ)
  (h1 : vessel1_capacity = 2)
  (h2 : vessel1_alcohol_percentage = 25)
  (h3 : vessel2_capacity = 6)
  (h4 : vessel2_alcohol_percentage = 40)
  (h5 : total_liquid = 8)
  (h6 : final_vessel_capacity = 10) :
  (vessel1_capacity * vessel1_alcohol_percentage / 100 +
   vessel2_capacity * vessel2_alcohol_percentage / 100) /
  final_vessel_capacity * 100 = 29 := by
  sorry


end alcohol_mixture_concentration_l3640_364045


namespace original_number_is_six_l3640_364058

/-- Represents a person in the circle with their chosen number and announced average -/
structure Person where
  chosen : ℝ
  announced : ℝ

/-- The circle of 12 people -/
def Circle := Fin 12 → Person

theorem original_number_is_six
  (circle : Circle)
  (h_average : ∀ i : Fin 12, (circle i).announced = ((circle (i - 1)).chosen + (circle (i + 1)).chosen) / 2)
  (h_person : ∃ i : Fin 12, (circle i).announced = 8 ∧
    (circle (i - 1)).announced = 5 ∧ (circle (i + 1)).announced = 11) :
  ∃ i : Fin 12, (circle i).announced = 8 ∧ (circle i).chosen = 6 := by
  sorry

end original_number_is_six_l3640_364058


namespace classroom_difference_l3640_364022

/-- Proves that the difference between the total number of students and books in 6 classrooms is 90 -/
theorem classroom_difference : 
  let students_per_classroom : ℕ := 20
  let books_per_classroom : ℕ := 5
  let num_classrooms : ℕ := 6
  let total_students : ℕ := students_per_classroom * num_classrooms
  let total_books : ℕ := books_per_classroom * num_classrooms
  total_students - total_books = 90 := by
  sorry


end classroom_difference_l3640_364022


namespace bowling_team_score_l3640_364080

/-- Represents the scores of a bowling team with three members -/
structure BowlingTeam where
  first_bowler : ℕ
  second_bowler : ℕ
  third_bowler : ℕ

/-- Calculates the total score of a bowling team -/
def total_score (team : BowlingTeam) : ℕ :=
  team.first_bowler + team.second_bowler + team.third_bowler

/-- Theorem stating the total score of the bowling team under given conditions -/
theorem bowling_team_score :
  ∃ (team : BowlingTeam),
    team.third_bowler = 162 ∧
    team.second_bowler = 3 * team.third_bowler ∧
    team.first_bowler = team.second_bowler / 3 ∧
    total_score team = 810 := by
  sorry


end bowling_team_score_l3640_364080


namespace correct_average_calculation_l3640_364012

theorem correct_average_calculation (n : ℕ) (initial_avg : ℚ) (wrong_num correct_num : ℚ) :
  n = 10 ∧ initial_avg = 15 ∧ wrong_num = 26 ∧ correct_num = 36 →
  (n : ℚ) * initial_avg - wrong_num + correct_num = n * 16 :=
by sorry

end correct_average_calculation_l3640_364012


namespace prob_three_non_defective_pencils_l3640_364046

/-- The probability of selecting 3 non-defective pencils from a box of 7 pencils with 2 defective pencils -/
theorem prob_three_non_defective_pencils :
  let total_pencils : ℕ := 7
  let defective_pencils : ℕ := 2
  let selected_pencils : ℕ := 3
  let non_defective_pencils : ℕ := total_pencils - defective_pencils
  let ways_to_select_all := Nat.choose total_pencils selected_pencils
  let ways_to_select_non_defective := Nat.choose non_defective_pencils selected_pencils
  (ways_to_select_non_defective : ℚ) / ways_to_select_all = 2 / 7 :=
by sorry

end prob_three_non_defective_pencils_l3640_364046


namespace remaining_integers_l3640_364053

theorem remaining_integers (T : Finset ℕ) : 
  T = Finset.range 100 → 
  (T.filter (λ x => x % 4 ≠ 0 ∧ x % 5 ≠ 0)).card = 60 :=
by
  sorry

end remaining_integers_l3640_364053


namespace library_sunday_visitors_l3640_364075

/-- Calculates the average number of visitors on Sundays in a library -/
theorem library_sunday_visitors
  (total_days : Nat)
  (sunday_count : Nat)
  (non_sunday_visitors : Nat)
  (total_average : Nat)
  (h1 : total_days = 30)
  (h2 : sunday_count = 5)
  (h3 : non_sunday_visitors = 240)
  (h4 : total_average = 295) :
  (total_average * total_days - non_sunday_visitors * (total_days - sunday_count)) / sunday_count = 570 := by
  sorry

end library_sunday_visitors_l3640_364075


namespace find_genuine_coin_l3640_364028

/-- Represents a coin, which can be either genuine or counterfeit -/
inductive Coin
| genuine : Coin
| counterfeit : ℕ → Coin

/-- Represents the result of weighing two coins -/
inductive WeighingResult
| equal : WeighingResult
| unequal : WeighingResult

/-- Represents a collection of coins -/
def CoinSet := List Coin

/-- Represents a weighing action -/
def Weighing := Coin → Coin → WeighingResult

/-- Represents a strategy to find a genuine coin -/
def Strategy := CoinSet → Weighing → Option Coin

theorem find_genuine_coin 
  (coins : CoinSet) 
  (h_total : coins.length = 9)
  (h_counterfeit : (coins.filter (λ c => match c with 
    | Coin.counterfeit _ => true 
    | _ => false)).length = 4)
  (h_genuine_equal : ∀ c1 c2, c1 = Coin.genuine ∧ c2 = Coin.genuine → 
    (λ _ _ => WeighingResult.equal) c1 c2 = WeighingResult.equal)
  (h_counterfeit_differ : ∀ c1 c2, c1 ≠ c2 → 
    (c1 = Coin.genuine ∨ (∃ n, c1 = Coin.counterfeit n)) ∧ 
    (c2 = Coin.genuine ∨ (∃ m, c2 = Coin.counterfeit m)) → 
    (λ _ _ => WeighingResult.unequal) c1 c2 = WeighingResult.unequal)
  : ∃ (s : Strategy), ∀ w : Weighing, 
    (∃ c, s coins w = some c ∧ c = Coin.genuine) ∧ 
    (s coins w).isSome → (Nat.card {p : Coin × Coin | w p.1 p.2 ≠ WeighingResult.equal}) ≤ 4 :=
sorry

end find_genuine_coin_l3640_364028


namespace hula_hoop_ratio_l3640_364018

def nancy_time : ℕ := 10
def casey_time : ℕ := nancy_time - 3
def morgan_time : ℕ := 21

theorem hula_hoop_ratio : 
  ∃ (k : ℕ), k > 0 ∧ morgan_time = k * casey_time ∧ morgan_time / casey_time = 3 := by
  sorry

end hula_hoop_ratio_l3640_364018


namespace particular_number_exists_l3640_364020

theorem particular_number_exists : ∃ x : ℝ, 4 * 25 * x = 812 := by
  sorry

end particular_number_exists_l3640_364020


namespace parabola_b_value_l3640_364038

/-- A parabola passing through two given points has a specific 'b' value -/
theorem parabola_b_value (b c : ℝ) : 
  ((-1)^2 + b*(-1) + c = -8) → 
  (2^2 + b*2 + c = 10) → 
  b = 5 := by
sorry

end parabola_b_value_l3640_364038


namespace circle_radius_l3640_364095

theorem circle_radius (x y d : Real) (h : x + y + d = 164 * Real.pi) :
  ∃ (r : Real), r = 10 ∧ x = Real.pi * r^2 ∧ y = 2 * Real.pi * r ∧ d = 2 * r := by
  sorry

end circle_radius_l3640_364095


namespace meetings_percentage_is_24_l3640_364085

/-- Represents the duration of a work day in minutes -/
def work_day_minutes : ℕ := 10 * 60

/-- Represents the duration of a break in minutes -/
def break_minutes : ℕ := 30

/-- Represents the duration of the first meeting in minutes -/
def first_meeting_minutes : ℕ := 60

/-- Represents the duration of the second meeting in minutes -/
def second_meeting_minutes : ℕ := 75

/-- Calculates the effective work minutes (excluding break) -/
def effective_work_minutes : ℕ := work_day_minutes - break_minutes

/-- Calculates the total meeting minutes -/
def total_meeting_minutes : ℕ := first_meeting_minutes + second_meeting_minutes

/-- Theorem stating that the percentage of effective work day spent in meetings is 24% -/
theorem meetings_percentage_is_24 : 
  (total_meeting_minutes : ℚ) / (effective_work_minutes : ℚ) * 100 = 24 := by
  sorry

end meetings_percentage_is_24_l3640_364085


namespace max_abs_z_on_circle_l3640_364062

theorem max_abs_z_on_circle (z : ℂ) (h : Complex.abs (z - Complex.I * 2) = 1) :
  ∃ (z_max : ℂ), Complex.abs z_max = 3 ∧ 
  ∀ (w : ℂ), Complex.abs (w - Complex.I * 2) = 1 → Complex.abs w ≤ Complex.abs z_max :=
sorry

end max_abs_z_on_circle_l3640_364062


namespace special_function_at_eight_l3640_364065

/-- A monotonic function on (0, +∞) satisfying certain conditions -/
def SpecialFunction (f : ℝ → ℝ) : Prop :=
  (∀ x y, 0 < x ∧ x < y → f x < f y) ∧ 
  (∀ x, x > 0 → f x > -4/x) ∧
  (∀ x, x > 0 → f (f x + 4/x) = 3)

/-- The main theorem stating that f(8) = 7/2 for a SpecialFunction -/
theorem special_function_at_eight (f : ℝ → ℝ) (h : SpecialFunction f) : f 8 = 7/2 := by
  sorry

end special_function_at_eight_l3640_364065


namespace opposite_values_l3640_364029

theorem opposite_values (a b c m : ℚ) 
  (eq1 : a + 2*b + 3*c = m) 
  (eq2 : a + b + 2*c = m) : 
  b = -c := by
sorry

end opposite_values_l3640_364029


namespace scout_troop_profit_l3640_364006

/-- Calculates the profit of a scout troop selling candy bars -/
theorem scout_troop_profit (num_bars : ℕ) (buy_price : ℚ) (sell_price : ℚ) : 
  num_bars = 1500 → 
  buy_price = 1/3 → 
  sell_price = 2/3 → 
  (sell_price - buy_price) * num_bars = 500 := by
  sorry

#check scout_troop_profit

end scout_troop_profit_l3640_364006


namespace jellybean_mass_theorem_l3640_364033

/-- The price of jellybeans in cents per gram -/
def price_per_gram : ℚ := 750 / 250

/-- The mass of jellybeans in grams that can be bought for 180 cents -/
def mass_for_180_cents : ℚ := 180 / price_per_gram

theorem jellybean_mass_theorem :
  mass_for_180_cents = 60 := by sorry

end jellybean_mass_theorem_l3640_364033


namespace average_monthly_sales_l3640_364070

def may_sales : ℝ := 150
def june_sales : ℝ := 75
def july_sales : ℝ := 50
def august_sales : ℝ := 175

def total_months : ℕ := 4

def total_sales : ℝ := may_sales + june_sales + july_sales + august_sales

theorem average_monthly_sales : 
  total_sales / total_months = 112.5 := by sorry

end average_monthly_sales_l3640_364070


namespace additional_monthly_income_l3640_364008

/-- Given a shoe company's current monthly sales and desired annual income,
    calculate the additional monthly income required to reach the annual goal. -/
theorem additional_monthly_income
  (current_monthly_sales : ℕ)
  (desired_annual_income : ℕ)
  (h1 : current_monthly_sales = 4000)
  (h2 : desired_annual_income = 60000) :
  (desired_annual_income - current_monthly_sales * 12) / 12 = 1000 :=
by sorry

end additional_monthly_income_l3640_364008


namespace conor_weekly_vegetables_l3640_364044

def eggplants : ℕ := 12
def carrots : ℕ := 9
def potatoes : ℕ := 8
def onions : ℕ := 15
def zucchinis : ℕ := 7
def work_days : ℕ := 6

def vegetables_per_day : ℕ := eggplants + carrots + potatoes + onions + zucchinis

theorem conor_weekly_vegetables :
  vegetables_per_day * work_days = 306 := by sorry

end conor_weekly_vegetables_l3640_364044


namespace rainfall_third_week_l3640_364034

theorem rainfall_third_week (total : ℝ) (week1 : ℝ) (week2 : ℝ) (week3 : ℝ)
  (h_total : total = 45)
  (h_week2 : week2 = 1.5 * week1)
  (h_week3 : week3 = 2 * week2)
  (h_sum : week1 + week2 + week3 = total) :
  week3 = 22.5 := by
  sorry

end rainfall_third_week_l3640_364034


namespace phils_remaining_pages_l3640_364096

/-- Given an initial number of books, pages per book, and books lost,
    calculate the total number of pages remaining. -/
def remaining_pages (initial_books : ℕ) (pages_per_book : ℕ) (books_lost : ℕ) : ℕ :=
  (initial_books - books_lost) * pages_per_book

/-- Theorem stating that with 10 initial books, 100 pages per book,
    and 2 books lost, the remaining pages total 800. -/
theorem phils_remaining_pages :
  remaining_pages 10 100 2 = 800 := by
  sorry

end phils_remaining_pages_l3640_364096


namespace circle_center_transformation_l3640_364088

/-- Reflect a point across the y-axis -/
def reflect_y (p : ℝ × ℝ) : ℝ × ℝ :=
  (-p.1, p.2)

/-- Translate a point vertically -/
def translate_up (p : ℝ × ℝ) (d : ℝ) : ℝ × ℝ :=
  (p.1, p.2 + d)

/-- The transformation described in the problem -/
def transform (p : ℝ × ℝ) : ℝ × ℝ :=
  translate_up (reflect_y p) 12

theorem circle_center_transformation :
  transform (3, -4) = (-3, 8) := by
  sorry

end circle_center_transformation_l3640_364088


namespace pencils_across_diameter_l3640_364007

theorem pencils_across_diameter (radius : ℝ) (pencil_length : ℝ) : 
  radius = 14 → pencil_length = 0.5 → 
  (2 * radius * 12) / pencil_length = 56 := by
  sorry

end pencils_across_diameter_l3640_364007


namespace base6_addition_example_l3640_364017

/-- Addition in base 6 -/
def base6_add (a b : ℕ) : ℕ := sorry

/-- Conversion from base 6 to base 10 -/
def base6_to_base10 (n : ℕ) : ℕ := sorry

/-- Conversion from base 10 to base 6 -/
def base10_to_base6 (n : ℕ) : ℕ := sorry

theorem base6_addition_example : base6_add 152 35 = 213 := by sorry

end base6_addition_example_l3640_364017


namespace M_greater_than_N_l3640_364066

theorem M_greater_than_N (a : ℝ) : 2 * a * (a - 2) > (a + 1) * (a - 3) := by
  sorry

end M_greater_than_N_l3640_364066


namespace unique_solution_l3640_364055

def product_of_digits (n : ℕ) : ℕ := sorry

theorem unique_solution : ∃! x : ℕ+, 
  (x : ℕ) > 0 ∧ product_of_digits x = x^2 - 10*x - 22 ∧ x = 12 := by sorry

end unique_solution_l3640_364055


namespace expression_evaluation_l3640_364021

theorem expression_evaluation : (25 + 15)^2 - (25^2 + 15^2 + 150) = 600 := by
  sorry

end expression_evaluation_l3640_364021


namespace cheaper_store_difference_l3640_364010

/-- The list price of Book Y in dollars -/
def list_price : ℚ := 24.95

/-- The discount amount at Readers' Delight in dollars -/
def readers_delight_discount : ℚ := 5

/-- The discount percentage at Book Bargains -/
def book_bargains_discount_percent : ℚ := 20

/-- The sale price at Readers' Delight in dollars -/
def readers_delight_price : ℚ := list_price - readers_delight_discount

/-- The sale price at Book Bargains in dollars -/
def book_bargains_price : ℚ := list_price * (1 - book_bargains_discount_percent / 100)

/-- The price difference in cents -/
def price_difference_cents : ℤ := ⌊(book_bargains_price - readers_delight_price) * 100⌋

theorem cheaper_store_difference :
  price_difference_cents = 1 :=
sorry

end cheaper_store_difference_l3640_364010


namespace mollys_age_l3640_364036

/-- Molly's age calculation --/
theorem mollys_age (initial_candles additional_candles : ℕ) :
  initial_candles = 14 → additional_candles = 6 →
  initial_candles + additional_candles = 20 := by
  sorry

end mollys_age_l3640_364036


namespace tan_two_implies_expression_eq_neg_two_l3640_364082

theorem tan_two_implies_expression_eq_neg_two (θ : Real) (h : Real.tan θ = 2) :
  (2 * Real.cos θ) / (Real.sin (π / 2 + θ) + Real.sin (π + θ)) = -2 := by
  sorry

end tan_two_implies_expression_eq_neg_two_l3640_364082


namespace inequality_system_solution_l3640_364016

theorem inequality_system_solution (x : ℝ) :
  (1 - x > 0) ∧ ((x + 2) / 3 - 1 ≤ x) → -1/2 ≤ x ∧ x < 1 := by
  sorry

end inequality_system_solution_l3640_364016


namespace sum_abs_bound_l3640_364027

theorem sum_abs_bound (x y z : ℝ) 
  (eq1 : x^2 + y^2 + z = 15)
  (eq2 : x + y + z^2 = 27)
  (eq3 : x*y + y*z + z*x = 7) :
  7 ≤ |x + y + z| ∧ |x + y + z| ≤ 8 := by
  sorry

end sum_abs_bound_l3640_364027


namespace polynomial_division_remainder_l3640_364089

theorem polynomial_division_remainder :
  ∃ (q r : Polynomial ℚ),
    3 * X^4 + 14 * X^3 - 50 * X^2 - 72 * X + 55 = (X^2 + 8 * X - 4) * q + r ∧
    r = 224 * X - 113 ∧
    r.degree < (X^2 + 8 * X - 4).degree :=
by sorry

end polynomial_division_remainder_l3640_364089


namespace inscribed_cube_volume_l3640_364074

/-- The volume of a cube inscribed in a sphere, which is itself inscribed in a larger cube -/
theorem inscribed_cube_volume (outer_cube_edge : ℝ) (h : outer_cube_edge = 12) :
  let sphere_diameter := outer_cube_edge
  let inner_cube_diagonal := sphere_diameter
  let inner_cube_edge := inner_cube_diagonal / Real.sqrt 3
  let inner_cube_volume := inner_cube_edge ^ 3
  inner_cube_volume = 192 * Real.sqrt 3 := by
  sorry

end inscribed_cube_volume_l3640_364074


namespace division_multiplication_result_l3640_364059

theorem division_multiplication_result : 
  let x : ℝ := 5.5
  let y : ℝ := (x / 6) * 12
  y = 11 := by sorry

end division_multiplication_result_l3640_364059


namespace connie_marbles_l3640_364054

/-- Calculates the number of marbles Connie has after giving some away -/
def marblesRemaining (initial : ℕ) (givenAway : ℕ) : ℕ :=
  initial - givenAway

/-- Proves that Connie has 3 marbles remaining after giving away 70 from her initial 73 -/
theorem connie_marbles : marblesRemaining 73 70 = 3 := by
  sorry

end connie_marbles_l3640_364054


namespace units_digit_factorial_sum_2006_l3640_364015

def factorial (n : ℕ) : ℕ := (List.range n).foldl (·*·) 1

def units_digit (n : ℕ) : ℕ := n % 10

def factorial_sum (n : ℕ) : ℕ := (List.range n).map factorial |>.sum

theorem units_digit_factorial_sum_2006 :
  units_digit (factorial_sum 2006) = 3 := by
  sorry

end units_digit_factorial_sum_2006_l3640_364015


namespace complex_power_eq_l3640_364048

theorem complex_power_eq (z : ℂ) : 
  (2 * Complex.cos (20 * π / 180) + 2 * Complex.I * Complex.sin (20 * π / 180)) ^ 6 = 
  -32 + 32 * Complex.I * Real.sqrt 3 := by
  sorry

end complex_power_eq_l3640_364048


namespace f_of_2_equals_6_l3640_364051

-- Define the function f
def f (x : ℝ) : ℝ := (x + 1)^2 - 3

-- Theorem statement
theorem f_of_2_equals_6 : f 2 = 6 := by
  sorry

end f_of_2_equals_6_l3640_364051


namespace max_value_theorem_l3640_364042

theorem max_value_theorem (x y : ℝ) (hx : x > 0) (hy : y > 0) 
  (h : 2 * x^2 - x*y + y^2 = 15) : 
  (∀ a b : ℝ, a > 0 → b > 0 → 2 * a^2 - a*b + b^2 = 15 → 
    2 * x^2 + x*y + y^2 ≥ 2 * a^2 + a*b + b^2) → 
  2 * x^2 + x*y + y^2 = (75 + 60 * Real.sqrt 2) / 7 := by
sorry

end max_value_theorem_l3640_364042


namespace inverse_mod_53_l3640_364001

theorem inverse_mod_53 (h : (17⁻¹ : ZMod 53) = 13) : (36⁻¹ : ZMod 53) = 40 := by
  sorry

end inverse_mod_53_l3640_364001


namespace stamp_cost_theorem_l3640_364057

theorem stamp_cost_theorem (total_stamps : ℕ) (high_value_stamps : ℕ) (high_value : ℚ) (low_value : ℚ) :
  total_stamps = 20 →
  high_value_stamps = 18 →
  high_value = 37 / 100 →
  low_value = 20 / 100 →
  (high_value_stamps * high_value + (total_stamps - high_value_stamps) * low_value) = 706 / 100 := by
  sorry

end stamp_cost_theorem_l3640_364057


namespace optimal_strategy_and_expected_red_balls_l3640_364002

-- Define the contents of A's box
structure BoxA where
  red : ℕ
  white : ℕ
  sum_eq_four : red + white = 4

-- Define the contents of B's box
def BoxB : Finset (Fin 4) := {0, 1, 2, 3}

-- Define the probability of winning for A given their box contents
def win_probability (box : BoxA) : ℚ :=
  (box.red * box.white * 2) / (12 * 6)

-- Define the expected number of red balls drawn
def expected_red_balls (box : BoxA) : ℚ :=
  (box.red * 2 / 6) + (2 / 4)

-- Theorem statement
theorem optimal_strategy_and_expected_red_balls :
  ∃ (box : BoxA),
    (∀ (other : BoxA), win_probability box ≥ win_probability other) ∧
    (box.red = 2 ∧ box.white = 2) ∧
    (expected_red_balls box = 3/2) := by
  sorry

end optimal_strategy_and_expected_red_balls_l3640_364002


namespace cone_volume_from_cylinder_volume_l3640_364035

/-- Given a cylinder with volume 72π cm³ and height twice its radius,
    prove that a cone with the same radius and height has a volume of 144π cm³. -/
theorem cone_volume_from_cylinder_volume (r h : ℝ) : 
  (π * r^2 * h = 72 * π) → 
  (h = 2 * r) → 
  ((1/3) * π * r^2 * h = 144 * π) := by
  sorry

end cone_volume_from_cylinder_volume_l3640_364035


namespace factorization_of_2x_squared_minus_8_l3640_364031

theorem factorization_of_2x_squared_minus_8 (x : ℝ) : 2 * x^2 - 8 = 2 * (x + 2) * (x - 2) := by
  sorry

end factorization_of_2x_squared_minus_8_l3640_364031


namespace sum_of_coefficients_equals_88_875_l3640_364019

/-- Represents the grid and shapes configuration --/
structure GridConfig where
  gridSize : ℕ
  squareSide : ℝ
  smallCircleDiameter : ℝ
  largeCircleDiameter : ℝ
  hexagonSide : ℝ
  smallCircleCount : ℕ

/-- Calculates the coefficients A, B, and C for the shaded area expression --/
def calculateCoefficients (config : GridConfig) : ℝ × ℝ × ℝ :=
  sorry

/-- Theorem stating that the sum of coefficients equals 88.875 for the given configuration --/
theorem sum_of_coefficients_equals_88_875 : 
  let config : GridConfig := {
    gridSize := 6,
    squareSide := 1.5,
    smallCircleDiameter := 1.5,
    largeCircleDiameter := 3,
    hexagonSide := 1.5,
    smallCircleCount := 4
  }
  let (A, B, C) := calculateCoefficients config
  A + B + C = 88.875 := by sorry

end sum_of_coefficients_equals_88_875_l3640_364019


namespace min_distance_sum_l3640_364052

-- Define the parabola
def parabola (x y : ℝ) : Prop := y^2 = -4*x

-- Define the focus F (we don't know its exact coordinates, but we know it exists)
axiom F : ℝ × ℝ

-- Define point A
def A : ℝ × ℝ := (-2, 1)

-- Define a point P on the parabola
structure PointOnParabola where
  P : ℝ × ℝ
  on_parabola : parabola P.1 P.2

-- State the theorem
theorem min_distance_sum (p : PointOnParabola) :
  ∃ (min : ℝ), min = 3 ∧ ∀ (q : PointOnParabola), 
    Real.sqrt ((q.P.1 - F.1)^2 + (q.P.2 - F.2)^2) +
    Real.sqrt ((q.P.1 - A.1)^2 + (q.P.2 - A.2)^2) ≥ min :=
sorry

end min_distance_sum_l3640_364052


namespace min_value_theorem_l3640_364014

theorem min_value_theorem (x : ℝ) (h : x > 9) :
  (x^2 + 81) / (x - 9) ≥ 27 ∧ ∃ y > 9, (y^2 + 81) / (y - 9) = 27 := by
  sorry

end min_value_theorem_l3640_364014


namespace possible_values_of_a_l3640_364056

theorem possible_values_of_a (a b c : ℤ) :
  (∀ x : ℤ, (x - a) * (x - 5) + 3 = (x + b) * (x + c)) →
  (a = 1 ∨ a = 9) :=
by sorry

end possible_values_of_a_l3640_364056


namespace triangle_properties_l3640_364011

/-- Represents an acute triangle ABC with sides a, b, c opposite to angles A, B, C respectively. -/
structure AcuteTriangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ
  acute : 0 < A ∧ A < π/2 ∧ 0 < B ∧ B < π/2 ∧ 0 < C ∧ C < π/2

/-- The main theorem about the specific acute triangle. -/
theorem triangle_properties (t : AcuteTriangle)
    (h1 : Real.sqrt 3 * t.a - 2 * t.b * Real.sin t.A = 0)
    (h2 : t.a + t.c = 5)
    (h3 : t.a > t.c)
    (h4 : t.b = Real.sqrt 7) :
    t.B = π/3 ∧ (1/2 * t.a * t.c * Real.sin t.B = 3 * Real.sqrt 3 / 2) := by
  sorry

end triangle_properties_l3640_364011


namespace intersection_P_Q_l3640_364005

-- Define the sets P and Q
def P : Set ℝ := {x | x > 0}
def Q : Set ℝ := {x | -1 < x ∧ x < 2}

-- State the theorem
theorem intersection_P_Q : P ∩ Q = {x : ℝ | 0 < x ∧ x < 2} := by sorry

end intersection_P_Q_l3640_364005

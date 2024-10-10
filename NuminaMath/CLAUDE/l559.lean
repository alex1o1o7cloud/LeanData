import Mathlib

namespace group_average_before_new_member_l559_55979

theorem group_average_before_new_member (group : Finset ℕ) (group_sum : ℕ) (new_member : ℕ) :
  Finset.card group = 7 →
  group_sum / Finset.card group = 20 →
  new_member = 56 →
  group_sum / Finset.card group = 20 := by
sorry

end group_average_before_new_member_l559_55979


namespace school_sections_l559_55934

theorem school_sections (num_boys num_girls : ℕ) (h1 : num_boys = 408) (h2 : num_girls = 240) :
  let section_size := Nat.gcd num_boys num_girls
  let boys_sections := num_boys / section_size
  let girls_sections := num_girls / section_size
  boys_sections + girls_sections = 27 := by
sorry

end school_sections_l559_55934


namespace characterize_valid_functions_l559_55922

def is_valid_function (f : ℕ → ℕ) : Prop :=
  (∀ x, f x ≤ x^2) ∧
  (∀ x y, x > y → (x - y) ∣ (f x - f y))

theorem characterize_valid_functions :
  ∀ f : ℕ → ℕ, is_valid_function f ↔
    (∀ x, f x = 0) ∨
    (∀ x, f x = x) ∨
    (∀ x, f x = x^2 - x) ∨
    (∀ x, f x = x^2) :=
sorry


end characterize_valid_functions_l559_55922


namespace rectangular_plot_breadth_l559_55917

theorem rectangular_plot_breadth (length breadth area : ℝ) : 
  length = 3 * breadth →
  area = length * breadth →
  area = 675 →
  breadth = 15 :=
by sorry

end rectangular_plot_breadth_l559_55917


namespace f_properties_l559_55988

open Real

noncomputable def f (m : ℝ) (x : ℝ) : ℝ := (1/2) * x^2 - m * log x

theorem f_properties (m : ℝ) (h : m ≥ 1) :
  (∃! (x : ℝ), x > 0 ∧ f m x = x^2 - (m + 1) * x) ∧
  (∃ (x : ℝ), x > 0 ∧ ∀ (y : ℝ), y > 0 → f m x ≤ f m y) ∧
  (∃ (x : ℝ), x > 0 ∧ f m x = (m/2) * (1 - log m)) := by
sorry

end f_properties_l559_55988


namespace imaginary_unit_seventh_power_l559_55909

theorem imaginary_unit_seventh_power :
  ∀ i : ℂ, i^2 = -1 → i^7 = -i :=
by
  sorry

end imaginary_unit_seventh_power_l559_55909


namespace system_properties_l559_55921

-- Define the system of equations
def system (x y a : ℝ) : Prop :=
  x + 3 * y = 4 - a ∧ x - y = 3 * a

-- Define the statements to be proven
theorem system_properties :
  -- Statement 1
  (∃ x y : ℝ, system x y 2 ∧ x = 5 ∧ y = -1) ∧
  -- Statement 2
  (∃ x y : ℝ, system x y (-2) ∧ x = -y) ∧
  -- Statement 3
  (∀ x y a : ℝ, system x y a → x + 2 * y = 3) ∧
  -- Statement 4
  (∃ x y : ℝ, system x y (-1) ∧ x + y ≠ 4 - (-1)) :=
by sorry

end system_properties_l559_55921


namespace integer_subset_condition_l559_55968

theorem integer_subset_condition (a b : ℤ) : 
  (a * b * (a - b) ≠ 0) →
  (∃ (Z₀ : Set ℤ), ∀ (n : ℤ), (n ∈ Z₀ ∨ (n + a) ∈ Z₀ ∨ (n + b) ∈ Z₀) ∧ 
    ¬(n ∈ Z₀ ∧ (n + a) ∈ Z₀) ∧ ¬(n ∈ Z₀ ∧ (n + b) ∈ Z₀) ∧ ¬((n + a) ∈ Z₀ ∧ (n + b) ∈ Z₀)) ↔
  (∃ (k y z : ℤ), a = k * y ∧ b = k * z ∧ y % 3 ≠ 0 ∧ z % 3 ≠ 0 ∧ (y - z) % 3 ≠ 0) :=
by sorry

end integer_subset_condition_l559_55968


namespace quadratic_equation_root_l559_55972

theorem quadratic_equation_root (k : ℝ) : 
  (∃ x : ℂ, 3 * x^2 + k * x + 18 = 0 ∧ x = 2 - 3*I) → k = -12 := by
  sorry

end quadratic_equation_root_l559_55972


namespace product_increase_thirteen_times_l559_55946

theorem product_increase_thirteen_times :
  ∃ (a b c d e f g : ℕ),
    (a - 3) * (b - 3) * (c - 3) * (d - 3) * (e - 3) * (f - 3) * (g - 3) = 13 * (a * b * c * d * e * f * g) :=
by sorry

end product_increase_thirteen_times_l559_55946


namespace nancy_tortilla_chips_nancy_final_chips_l559_55965

/-- Calculates the number of tortilla chips Nancy has left after sharing with her family members -/
theorem nancy_tortilla_chips (initial_chips : ℝ) (brother_chips : ℝ) 
  (sister_fraction : ℝ) (cousin_percent : ℝ) : ℝ :=
  let remaining_after_brother := initial_chips - brother_chips
  let sister_chips := sister_fraction * remaining_after_brother
  let remaining_after_sister := remaining_after_brother - sister_chips
  let cousin_chips := (cousin_percent / 100) * remaining_after_sister
  let final_chips := remaining_after_sister - cousin_chips
  final_chips

/-- Proves that Nancy has 18.75 tortilla chips left for herself -/
theorem nancy_final_chips : 
  nancy_tortilla_chips 50 12.5 (1/3) 25 = 18.75 := by
  sorry

end nancy_tortilla_chips_nancy_final_chips_l559_55965


namespace largest_n_multiple_of_seven_largest_n_is_99996_l559_55924

theorem largest_n_multiple_of_seven (n : ℕ) : n < 100000 →
  (9 * (n - 3)^6 - n^3 + 16 * n - 27) % 7 = 0 →
  n ≤ 99996 :=
by sorry

theorem largest_n_is_99996 :
  (9 * (99996 - 3)^6 - 99996^3 + 16 * 99996 - 27) % 7 = 0 ∧
  99996 < 100000 ∧
  ∀ m : ℕ, m < 100000 →
    (9 * (m - 3)^6 - m^3 + 16 * m - 27) % 7 = 0 →
    m ≤ 99996 :=
by sorry

end largest_n_multiple_of_seven_largest_n_is_99996_l559_55924


namespace semicircle_arc_length_l559_55954

-- Define the right triangle with inscribed semicircle
structure RightTriangleWithSemicircle where
  -- Hypotenuse segments
  a : ℝ
  b : ℝ
  -- Assumption: a and b are positive
  ha : a > 0
  hb : b > 0
  -- Assumption: The semicircle is inscribed in the right triangle
  -- with its diameter on the hypotenuse

-- Define the theorem
theorem semicircle_arc_length 
  (triangle : RightTriangleWithSemicircle) 
  (h_a : triangle.a = 30) 
  (h_b : triangle.b = 40) : 
  ∃ (arc_length : ℝ), arc_length = 12 * Real.pi := by
  sorry


end semicircle_arc_length_l559_55954


namespace water_consumption_percentage_difference_l559_55943

theorem water_consumption_percentage_difference : 
  let yesterday_consumption : ℝ := 48
  let two_days_ago_consumption : ℝ := 50
  let difference := two_days_ago_consumption - yesterday_consumption
  let percentage_difference := (difference / two_days_ago_consumption) * 100
  percentage_difference = 4 := by
sorry

end water_consumption_percentage_difference_l559_55943


namespace equal_temperament_sequence_l559_55978

theorem equal_temperament_sequence (a : ℕ → ℝ) :
  (∀ n, 1 ≤ n → n ≤ 13 → a n > 0) →
  (∀ n, 1 ≤ n → n < 13 → a (n + 1) / a n = a 2 / a 1) →
  a 1 = 1 →
  a 13 = 2 →
  a 3 = 2^(1/6) :=
by sorry

end equal_temperament_sequence_l559_55978


namespace danny_wrappers_found_l559_55915

theorem danny_wrappers_found (initial_caps : ℕ) (found_caps : ℕ) (total_caps : ℕ) (total_wrappers : ℕ) 
  (h1 : initial_caps = 6)
  (h2 : found_caps = 22)
  (h3 : total_caps = 28)
  (h4 : total_wrappers = 63)
  (h5 : found_caps = total_caps - initial_caps)
  : ∃ (found_wrappers : ℕ), found_wrappers = 22 ∧ total_wrappers ≥ found_wrappers :=
by
  sorry

#check danny_wrappers_found

end danny_wrappers_found_l559_55915


namespace ice_cube_calculation_l559_55920

theorem ice_cube_calculation (cubes_per_tray : ℕ) (num_trays : ℕ) 
  (h1 : cubes_per_tray = 9) 
  (h2 : num_trays = 8) : 
  cubes_per_tray * num_trays = 72 := by
  sorry

end ice_cube_calculation_l559_55920


namespace violet_balloons_remaining_l559_55999

def initial_violet_balloons : ℕ := 7
def lost_violet_balloons : ℕ := 3

theorem violet_balloons_remaining :
  initial_violet_balloons - lost_violet_balloons = 4 := by
  sorry

end violet_balloons_remaining_l559_55999


namespace path_area_and_cost_l559_55949

/-- Calculates the area of a rectangular path surrounding a field -/
def path_area (field_length field_width path_width : ℝ) : ℝ :=
  (field_length + 2 * path_width) * (field_width + 2 * path_width) - field_length * field_width

/-- Calculates the cost of constructing a path given its area and cost per unit area -/
def construction_cost (path_area cost_per_unit : ℝ) : ℝ :=
  path_area * cost_per_unit

theorem path_area_and_cost (field_length field_width path_width cost_per_unit : ℝ) 
  (h1 : field_length = 75)
  (h2 : field_width = 40)
  (h3 : path_width = 2.5)
  (h4 : cost_per_unit = 2) :
  path_area field_length field_width path_width = 600 ∧ 
  construction_cost (path_area field_length field_width path_width) cost_per_unit = 1200 := by
  sorry

#eval path_area 75 40 2.5
#eval construction_cost (path_area 75 40 2.5) 2

end path_area_and_cost_l559_55949


namespace solid_is_frustum_l559_55932

/-- A solid with specified view characteristics -/
structure Solid where
  top_view : Bool
  bottom_view : Bool
  front_view : Bool
  side_view : Bool

/-- Definition of a frustum based on its views -/
def is_frustum (s : Solid) : Prop :=
  s.top_view = true ∧ 
  s.bottom_view = true ∧ 
  s.front_view = true ∧ 
  s.side_view = true

/-- Theorem: A solid with circular top and bottom views, and trapezoidal front and side views, is a frustum -/
theorem solid_is_frustum (s : Solid) 
  (h_top : s.top_view = true)
  (h_bottom : s.bottom_view = true)
  (h_front : s.front_view = true)
  (h_side : s.side_view = true) : 
  is_frustum s := by
  sorry

end solid_is_frustum_l559_55932


namespace sqrt_calculation_and_exponent_simplification_l559_55980

theorem sqrt_calculation_and_exponent_simplification :
  (∃ x : ℝ, x^2 = 18) ∧ (∃ y : ℝ, y^2 = 32) ∧ (∃ z : ℝ, z^2 = 2) →
  (∃ a : ℝ, a^2 = 3) →
  (∀ x y z : ℝ, x^2 = 18 ∧ y^2 = 32 ∧ z^2 = 2 → x - y + z = 0) ∧
  (∀ a : ℝ, a^2 = 3 → (a + 2)^2022 * (a - 2)^2021 * (a - 3) = 3 + a) :=
by sorry

end sqrt_calculation_and_exponent_simplification_l559_55980


namespace cos_identity_l559_55929

theorem cos_identity : 
  (2 * (Real.cos (15 * π / 180))^2 - Real.cos (30 * π / 180) = 1) :=
by
  have h : Real.cos (30 * π / 180) = 2 * (Real.cos (15 * π / 180))^2 - 1 := by sorry
  sorry

end cos_identity_l559_55929


namespace tyson_basketball_score_l559_55906

theorem tyson_basketball_score (three_point_shots two_point_shots one_point_shots : ℕ) 
  (h1 : three_point_shots = 15)
  (h2 : two_point_shots = 12)
  (h3 : 3 * three_point_shots + 2 * two_point_shots + one_point_shots = 75) :
  one_point_shots = 6 := by
  sorry

end tyson_basketball_score_l559_55906


namespace max_distance_between_functions_l559_55964

theorem max_distance_between_functions : ∃ (C : ℝ),
  C = Real.sqrt 5 ∧ 
  ∀ x : ℝ, |2 * Real.sin x - Real.sin (π / 2 - x)| ≤ C ∧
  ∃ x₀ : ℝ, |2 * Real.sin x₀ - Real.sin (π / 2 - x₀)| = C :=
by sorry

end max_distance_between_functions_l559_55964


namespace negation_of_all_cars_are_fast_l559_55982

variable (U : Type) -- Universe of discourse
variable (car : U → Prop) -- Predicate for being a car
variable (fast : U → Prop) -- Predicate for being fast

theorem negation_of_all_cars_are_fast :
  ¬(∀ x, car x → fast x) ↔ ∃ x, car x ∧ ¬(fast x) :=
by sorry

end negation_of_all_cars_are_fast_l559_55982


namespace boys_camp_total_l559_55900

theorem boys_camp_total (total : ℝ) 
  (h1 : 0.2 * total = total_school_A)
  (h2 : 0.3 * total_school_A = science_school_A)
  (h3 : total_school_A - science_school_A = 77) : 
  total = 550 := by
sorry

end boys_camp_total_l559_55900


namespace polynomial_expansion_l559_55930

theorem polynomial_expansion (x : ℝ) : 
  (2*x^2 + 5*x + 8)*(x+1) - (x+1)*(x^2 - 2*x + 50) + (3*x - 7)*(x+1)*(x - 2) = 
  4*x^3 - 2*x^2 - 34*x - 28 := by
sorry

end polynomial_expansion_l559_55930


namespace frank_remaining_money_l559_55957

def calculate_remaining_money (initial_amount : ℕ) 
                              (action_figure_cost : ℕ) (action_figure_count : ℕ)
                              (board_game_cost : ℕ) (board_game_count : ℕ)
                              (puzzle_set_cost : ℕ) (puzzle_set_count : ℕ) : ℕ :=
  initial_amount - 
  (action_figure_cost * action_figure_count + 
   board_game_cost * board_game_count + 
   puzzle_set_cost * puzzle_set_count)

theorem frank_remaining_money :
  calculate_remaining_money 100 12 3 11 2 6 4 = 18 := by
  sorry

end frank_remaining_money_l559_55957


namespace place_value_ratio_l559_55996

def number : ℚ := 86549.2047

theorem place_value_ratio : 
  let thousands_place_value : ℚ := 1000
  let tenths_place_value : ℚ := 0.1
  thousands_place_value / tenths_place_value = 10000 := by sorry

end place_value_ratio_l559_55996


namespace white_balls_count_l559_55984

theorem white_balls_count (total : ℕ) (yellow_probability : ℚ) : 
  total = 20 → yellow_probability = 3/5 → total - (total * yellow_probability).num = 8 := by
  sorry

end white_balls_count_l559_55984


namespace wilmas_garden_red_flowers_l559_55986

/-- Wilma's Garden Flower Count Theorem -/
theorem wilmas_garden_red_flowers :
  let total_flowers : ℕ := 6 * 13
  let yellow_flowers : ℕ := 12
  let green_flowers : ℕ := 2 * yellow_flowers
  let red_flowers : ℕ := total_flowers - (yellow_flowers + green_flowers)
  red_flowers = 42 := by sorry

end wilmas_garden_red_flowers_l559_55986


namespace range_of_a_l559_55948

theorem range_of_a (a : ℝ) : 
  (∀ x : ℝ, a * x^2 - a * x - 2 < 0) ↔ -8 < a ∧ a ≤ 0 :=
sorry

end range_of_a_l559_55948


namespace line_perpendicular_to_plane_l559_55989

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the relations
variable (parallel : Line → Line → Prop)
variable (perpendicular_line_plane : Line → Plane → Prop)

-- Theorem statement
theorem line_perpendicular_to_plane 
  (l n : Line) (α : Plane) :
  parallel l n → perpendicular_line_plane n α → perpendicular_line_plane l α :=
sorry

end line_perpendicular_to_plane_l559_55989


namespace functional_equation_solution_l559_55941

/-- A function satisfying the given functional equation -/
def FunctionalEquation (f : ℝ → ℝ) : Prop :=
  ∀ x, f x - (1/2) * f (x/2) = x^2

/-- The theorem stating that the function satisfying the equation is f(x) = (8/7) * x^2 -/
theorem functional_equation_solution (f : ℝ → ℝ) (h : FunctionalEquation f) :
  f = fun x ↦ (8/7) * x^2 := by
  sorry

end functional_equation_solution_l559_55941


namespace g_minus_one_eq_zero_l559_55963

/-- The function g(x) as defined in the problem -/
def g (x s : ℝ) : ℝ := 3 * x^5 - 2 * x^3 + x^2 - 4 * x + s

/-- Theorem stating that g(-1) = 0 when s = -4 -/
theorem g_minus_one_eq_zero :
  g (-1) (-4) = 0 := by
  sorry

end g_minus_one_eq_zero_l559_55963


namespace inscribed_rectangle_area_l559_55903

-- Define the triangle PQR
structure Triangle :=
  (P Q R : Point)
  (altitude : ℝ)
  (base : ℝ)

-- Define the rectangle ABCD
structure Rectangle :=
  (A B C D : Point)
  (width : ℝ)
  (height : ℝ)

-- Define the problem
def inscribed_rectangle_problem (triangle : Triangle) (rect : Rectangle) : Prop :=
  -- Rectangle ABCD is inscribed in triangle PQR
  -- Side AD of the rectangle is on side PR of the triangle
  -- Triangle's altitude from vertex Q to side PR is 8 inches
  triangle.altitude = 8 ∧
  -- PR = 12 inches
  triangle.base = 12 ∧
  -- Length of AB is equal to a third the length of AD
  rect.width = rect.height / 3 ∧
  -- The area of the rectangle is 64/3 square inches
  rect.width * rect.height = 64 / 3

-- Theorem statement
theorem inscribed_rectangle_area 
  (triangle : Triangle) (rect : Rectangle) :
  inscribed_rectangle_problem triangle rect → 
  rect.width * rect.height = 64 / 3 :=
by
  sorry

end inscribed_rectangle_area_l559_55903


namespace expected_heads_is_60_l559_55912

/-- The number of coins -/
def num_coins : ℕ := 64

/-- The probability of getting heads on a single flip -/
def p_heads : ℚ := 1/2

/-- The number of possible tosses for each coin -/
def max_tosses : ℕ := 4

/-- The probability of a coin showing heads after up to four tosses -/
def prob_heads_after_four : ℚ :=
  p_heads + (1 - p_heads) * p_heads + 
  (1 - p_heads)^2 * p_heads + (1 - p_heads)^3 * p_heads

/-- The expected number of coins showing heads after the series of tosses -/
def expected_heads : ℚ := num_coins * prob_heads_after_four

theorem expected_heads_is_60 : expected_heads = 60 := by
  sorry

end expected_heads_is_60_l559_55912


namespace total_distance_is_490_l559_55983

/-- Represents a segment of the journey -/
structure JourneySegment where
  speed : ℝ
  time : ℝ

/-- Calculates the distance traveled in a journey segment -/
def distanceTraveled (segment : JourneySegment) : ℝ :=
  segment.speed * segment.time

/-- Represents the entire journey -/
def Journey : List JourneySegment := [
  { speed := 90, time := 2 },
  { speed := 60, time := 1 },
  { speed := 100, time := 2.5 }
]

/-- Theorem: The total distance traveled in the journey is 490 km -/
theorem total_distance_is_490 : 
  (Journey.map distanceTraveled).sum = 490 := by sorry

end total_distance_is_490_l559_55983


namespace mistaken_division_l559_55992

theorem mistaken_division (n : ℕ) (h : n = 172) :
  ∃! x : ℕ, x > 0 ∧ n % x = 7 ∧ n / x = n / 4 - 28 := by
  sorry

end mistaken_division_l559_55992


namespace beach_shells_problem_l559_55959

theorem beach_shells_problem (jillian_shells savannah_shells clayton_shells : ℕ) 
  (friend_count friend_received : ℕ) :
  jillian_shells = 29 →
  clayton_shells = 8 →
  friend_count = 2 →
  friend_received = 27 →
  jillian_shells + savannah_shells + clayton_shells = friend_count * friend_received →
  savannah_shells = 17 := by
  sorry

end beach_shells_problem_l559_55959


namespace probability_less_than_4_l559_55939

/-- A square in the 2D plane -/
structure Square where
  bottomLeft : ℝ × ℝ
  sideLength : ℝ

/-- The probability that a randomly chosen point in the square satisfies a condition -/
def probability (s : Square) (condition : ℝ × ℝ → Prop) : ℝ :=
  sorry

/-- The specific square with vertices (0,0), (0,3), (3,3), and (3,0) -/
def specificSquare : Square :=
  { bottomLeft := (0, 0), sideLength := 3 }

/-- The condition x + y < 4 -/
def conditionLessThan4 (p : ℝ × ℝ) : Prop :=
  p.1 + p.2 < 4

theorem probability_less_than_4 :
  probability specificSquare conditionLessThan4 = 7/9 := by
  sorry

end probability_less_than_4_l559_55939


namespace prob_one_two_given_different_l559_55955

/-- The probability of at least one die showing 2, given that two fair dice show different numbers -/
theorem prob_one_two_given_different : ℝ := by
  -- Define the sample space of outcomes where the two dice show different numbers
  let different_outcomes : Finset (ℕ × ℕ) := sorry

  -- Define the event of at least one die showing 2, given different numbers
  let event_one_two : Finset (ℕ × ℕ) := sorry

  -- Define the probability measure
  let prob : Finset (ℕ × ℕ) → ℝ := sorry

  -- The probability is the measure of the event divided by the measure of the sample space
  have h : prob event_one_two / prob different_outcomes = 1 / 3 := by sorry

  exact 1 / 3

end prob_one_two_given_different_l559_55955


namespace jose_share_correct_l559_55973

/-- Calculates an investor's share of the profit based on their investment amount, duration, and the total profit, given the investments and durations of all participants. -/
def calculate_share (tom_investment : ℕ) (tom_duration : ℕ) (jose_investment : ℕ) (jose_duration : ℕ) (maria_investment : ℕ) (maria_duration : ℕ) (total_profit : ℕ) : ℚ :=
  let total_capital_months : ℕ := tom_investment * tom_duration + jose_investment * jose_duration + maria_investment * maria_duration
  (jose_investment * jose_duration : ℚ) / total_capital_months * total_profit

/-- Proves that Jose's share of the profit is correct given the specific investments and durations. -/
theorem jose_share_correct (total_profit : ℕ) : 
  calculate_share 30000 12 45000 10 60000 8 total_profit = 
  (45000 * 10 : ℚ) / (30000 * 12 + 45000 * 10 + 60000 * 8) * total_profit :=
by sorry

end jose_share_correct_l559_55973


namespace imaginary_part_of_z_l559_55956

theorem imaginary_part_of_z (i : ℂ) (h : i * i = -1) : 
  Complex.im ((2 - i) ^ 2) = -4 := by
sorry

end imaginary_part_of_z_l559_55956


namespace rectangular_field_width_l559_55944

theorem rectangular_field_width :
  ∀ (width length perimeter : ℝ),
    length = (7 / 5) * width →
    perimeter = 2 * length + 2 * width →
    perimeter = 360 →
    width = 75 := by
  sorry

end rectangular_field_width_l559_55944


namespace arithmetic_calculations_l559_55975

theorem arithmetic_calculations :
  ((-15) - (-5) + 6 = -4) ∧
  (81 / (-9/5) * (5/9) = -25) :=
by sorry

end arithmetic_calculations_l559_55975


namespace train_length_is_400_l559_55947

/-- Calculates the length of a train given its speed, the speed and length of a platform
    moving in the opposite direction, and the time taken to cross the platform. -/
def trainLength (trainSpeed : ℝ) (platformSpeed : ℝ) (platformLength : ℝ) (crossingTime : ℝ) : ℝ :=
  (trainSpeed + platformSpeed) * crossingTime - platformLength

/-- Theorem stating that under the given conditions, the train length is 400 meters. -/
theorem train_length_is_400 :
  let trainSpeed : ℝ := 20
  let platformSpeed : ℝ := 5
  let platformLength : ℝ := 250
  let crossingTime : ℝ := 26
  trainLength trainSpeed platformSpeed platformLength crossingTime = 400 := by
sorry

end train_length_is_400_l559_55947


namespace binary_110101_to_base7_l559_55977

def binary_to_decimal (b : List Bool) : ℕ :=
  b.enum.foldl (λ acc (i, bit) => acc + if bit then 2^i else 0) 0

def decimal_to_base7 (n : ℕ) : List ℕ :=
  if n = 0 then [0] else
    let rec aux (m : ℕ) (acc : List ℕ) :=
      if m = 0 then acc else aux (m / 7) ((m % 7) :: acc)
    aux n []

theorem binary_110101_to_base7 :
  decimal_to_base7 (binary_to_decimal [true, false, true, false, true, true]) = [1, 0, 4] :=
sorry

end binary_110101_to_base7_l559_55977


namespace triangle_formation_l559_55951

-- Define a function to check if three lengths can form a triangle
def can_form_triangle (a b c : ℝ) : Prop :=
  a + b > c ∧ b + c > a ∧ c + a > b

-- Theorem statement
theorem triangle_formation :
  can_form_triangle 4 4 7 ∧
  ¬ can_form_triangle 2 3 6 ∧
  ¬ can_form_triangle 3 4 8 ∧
  ¬ can_form_triangle 5 8 13 :=
by sorry

end triangle_formation_l559_55951


namespace divisibility_by_480_l559_55936

theorem divisibility_by_480 (n : ℤ) 
  (h2 : ¬ 2 ∣ n) 
  (h3 : ¬ 3 ∣ n) 
  (h5 : ¬ 5 ∣ n) : 
  480 ∣ (n^8 - 1) := by
sorry

end divisibility_by_480_l559_55936


namespace tournament_has_24_players_l559_55927

/-- Represents a tournament with the given conditions --/
structure Tournament where
  n : ℕ  -- Total number of players
  pointsAgainstLowest12 : ℕ → ℚ  -- Points each player earned against the lowest 12
  totalPoints : ℕ → ℚ  -- Total points of each player

/-- The conditions of the tournament --/
def tournamentConditions (t : Tournament) : Prop :=
  -- Each player plays against every other player
  ∀ i, t.totalPoints i ≤ (t.n - 1 : ℚ)
  -- Half of each player's points are from the lowest 12
  ∧ ∀ i, 2 * t.pointsAgainstLowest12 i = t.totalPoints i
  -- There are exactly 12 lowest-scoring players
  ∧ ∃ lowest12 : Finset ℕ, lowest12.card = 12 
    ∧ ∀ i ∈ lowest12, ∀ j ∉ lowest12, t.totalPoints i ≤ t.totalPoints j

/-- The theorem stating that the tournament has 24 players --/
theorem tournament_has_24_players (t : Tournament) 
  (h : tournamentConditions t) : t.n = 24 :=
sorry

end tournament_has_24_players_l559_55927


namespace chess_master_exhibition_l559_55942

theorem chess_master_exhibition (x : ℝ) 
  (h1 : 0.1 * x + 8 + 0.1 * (0.9 * x - 8) + 2 + 7 = x) : x = 20 := by
  sorry

end chess_master_exhibition_l559_55942


namespace gcd_polynomial_and_multiple_l559_55969

theorem gcd_polynomial_and_multiple (x : ℤ) : 
  (∃ k : ℤ, x = 32515 * k) →
  Int.gcd ((3*x+5)*(5*x+3)*(11*x+7)*(x+17)) x = 35 := by
  sorry

end gcd_polynomial_and_multiple_l559_55969


namespace cover_triangles_l559_55987

/-- The side length of the small equilateral triangle -/
def small_side : ℝ := 0.5

/-- The side length of the large equilateral triangle -/
def large_side : ℝ := 10

/-- The minimum number of small triangles needed to cover the large triangle -/
def min_triangles : ℕ := 400

theorem cover_triangles : 
  ∀ (n : ℕ), n * (small_side^2 * Real.sqrt 3 / 4) ≥ large_side^2 * Real.sqrt 3 / 4 → n ≥ min_triangles :=
by sorry

end cover_triangles_l559_55987


namespace circle_center_correct_l559_55904

/-- The equation of a circle in the xy-plane -/
def CircleEquation (x y : ℝ) : Prop :=
  x^2 + y^2 - 2*x + 2*y - 2 = 0

/-- The center of a circle -/
def CircleCenter : ℝ × ℝ := (1, -1)

/-- Theorem: The center of the circle defined by CircleEquation is CircleCenter -/
theorem circle_center_correct :
  ∀ (x y : ℝ), CircleEquation x y ↔ (x - CircleCenter.1)^2 + (y - CircleCenter.2)^2 = 4 :=
by sorry

end circle_center_correct_l559_55904


namespace ellipse_axis_endpoint_distance_l559_55998

/-- The distance between an endpoint of the major axis and an endpoint of the minor axis of the ellipse 4(x+2)^2 + 16y^2 = 64 is 2√5 -/
theorem ellipse_axis_endpoint_distance : 
  ∃ (C D : ℝ × ℝ),
    (C.1 + 2)^2 / 16 + C.2^2 / 4 = 1 ∧  -- C is on the ellipse
    (D.1 + 2)^2 / 16 + D.2^2 / 4 = 1 ∧  -- D is on the ellipse
    C.2 = 0 ∧                           -- C is on the x-axis (major axis)
    D.1 = -2 ∧                          -- D is on the y-axis (minor axis)
    Real.sqrt ((C.1 - D.1)^2 + (C.2 - D.2)^2) = 2 * Real.sqrt 5 :=
by sorry

end ellipse_axis_endpoint_distance_l559_55998


namespace dye_per_dot_l559_55950

/-- The amount of dye per dot given the number of dots per blouse, 
    total amount of dye, and number of blouses -/
theorem dye_per_dot 
  (dots_per_blouse : ℕ) 
  (total_dye : ℕ) 
  (num_blouses : ℕ) 
  (h1 : dots_per_blouse = 20)
  (h2 : total_dye = 50 * 400)
  (h3 : num_blouses = 100) :
  total_dye / (dots_per_blouse * num_blouses) = 10 := by
  sorry

#check dye_per_dot

end dye_per_dot_l559_55950


namespace twenty_two_percent_of_300_prove_twenty_two_percent_of_300_l559_55923

theorem twenty_two_percent_of_300 : ℝ → Prop :=
  fun result => (22 / 100 : ℝ) * 300 = result

theorem prove_twenty_two_percent_of_300 : twenty_two_percent_of_300 66 := by
  sorry

end twenty_two_percent_of_300_prove_twenty_two_percent_of_300_l559_55923


namespace inequality_and_function_property_l559_55901

def f (x : ℝ) := |x - 1|

theorem inequality_and_function_property :
  (∀ x : ℝ, f x + f (x + 4) ≥ 8 ↔ x ≤ -5 ∨ x ≥ 3) ∧
  (∀ a b : ℝ, |a| < 1 → |b| < 1 → a ≠ 0 → f (a * b) > |a| * f (b / a)) :=
by sorry

end inequality_and_function_property_l559_55901


namespace largest_integer_satisfying_inequality_l559_55997

theorem largest_integer_satisfying_inequality :
  ∀ x : ℤ, x ≤ -3 ↔ x - 5 > 3*x - 1 :=
by sorry

end largest_integer_satisfying_inequality_l559_55997


namespace dads_strawberries_weight_l559_55928

/-- The weight of Marco's dad's strawberries -/
def dads_strawberries (total_weight marco_weight : ℕ) : ℕ :=
  total_weight - marco_weight

/-- Theorem stating that Marco's dad's strawberries weigh 9 pounds -/
theorem dads_strawberries_weight :
  dads_strawberries 23 14 = 9 := by
  sorry

end dads_strawberries_weight_l559_55928


namespace intersection_of_sets_l559_55913

theorem intersection_of_sets (a : ℝ) : 
  let A : Set ℝ := {-a, a^2, a^2 + a}
  let B : Set ℝ := {-1, -1 - a, 1 + a^2}
  (A ∩ B).Nonempty → A ∩ B = {-1, 2} := by
sorry

end intersection_of_sets_l559_55913


namespace standard_deviation_reflects_fluctuation_amplitude_l559_55918

/-- Standard deviation of a sample -/
def standard_deviation (sample : List ℝ) : ℝ := sorry

/-- Fluctuation amplitude of a population -/
def fluctuation_amplitude (population : List ℝ) : ℝ := sorry

/-- The standard deviation of a sample approximately reflects 
    the fluctuation amplitude of a population -/
theorem standard_deviation_reflects_fluctuation_amplitude 
  (sample : List ℝ) (population : List ℝ) :
  ∃ (ε : ℝ), ε > 0 ∧ |standard_deviation sample - fluctuation_amplitude population| < ε :=
sorry

end standard_deviation_reflects_fluctuation_amplitude_l559_55918


namespace flea_difference_l559_55967

def flea_treatment (initial_fleas : ℕ) (treatments : ℕ) : ℕ :=
  initial_fleas / (2^treatments)

theorem flea_difference (initial_fleas : ℕ) :
  flea_treatment initial_fleas 4 = 14 →
  initial_fleas - flea_treatment initial_fleas 4 = 210 := by
sorry

end flea_difference_l559_55967


namespace total_weight_is_130_l559_55940

-- Define the weights as real numbers
variable (M D C : ℝ)

-- State the conditions
variable (h1 : D + C = 60)
variable (h2 : C = (1/5) * M)
variable (h3 : D = 46)

-- Theorem to prove
theorem total_weight_is_130 : M + D + C = 130 := by
  sorry

end total_weight_is_130_l559_55940


namespace sector_area_l559_55995

theorem sector_area (α : Real) (l : Real) (S : Real) :
  α = π / 9 →
  l = π / 3 →
  S = (1 / 2) * l * (l / α) →
  S = π / 2 := by
sorry

end sector_area_l559_55995


namespace circle_radius_l559_55993

theorem circle_radius (M N : ℝ) (h1 : M > 0) (h2 : N > 0) (h3 : M / N = 15) :
  ∃ r : ℝ, r > 0 ∧ M = π * r^2 ∧ N = 2 * π * r ∧ r = 30 := by
  sorry

end circle_radius_l559_55993


namespace olivias_race_time_l559_55990

def total_time : ℕ := 112  -- 1 hour 52 minutes in minutes

theorem olivias_race_time (olivia_time : ℕ) 
  (h1 : olivia_time + (olivia_time - 4) = total_time) : 
  olivia_time = 58 := by
  sorry

end olivias_race_time_l559_55990


namespace negation_of_proposition_l559_55916

theorem negation_of_proposition :
  (¬ ∀ x : ℝ, x > 0 → x + 1/x ≥ 2) ↔ (∃ x : ℝ, x > 0 ∧ x + 1/x < 2) :=
by sorry

end negation_of_proposition_l559_55916


namespace center_sum_coords_l559_55981

/-- Defines a circle with the equation x^2 + y^2 = 6x - 8y + 24 -/
def circle_equation (x y : ℝ) : Prop :=
  x^2 + y^2 = 6*x - 8*y + 24

/-- Defines the center of a circle -/
def is_center (h x y : ℝ) : Prop :=
  ∀ (a b : ℝ), circle_equation a b → (a - x)^2 + (b - y)^2 = h^2

theorem center_sum_coords :
  ∃ (x y : ℝ), is_center 7 x y ∧ x + y = -1 :=
sorry

end center_sum_coords_l559_55981


namespace remaining_eggs_eggs_after_three_days_l559_55985

/-- Calculates the remaining eggs after consumption --/
theorem remaining_eggs (initial : ℕ) (consumed : ℕ) (h : initial ≥ consumed) : 
  initial - consumed = 75 - 49 → initial - consumed = 26 := by
  sorry

/-- Proves that 26 eggs remain after 3 days --/
theorem eggs_after_three_days : 
  ∃ (initial consumed : ℕ), initial = 75 ∧ consumed = 49 ∧ initial - consumed = 26 := by
  sorry

end remaining_eggs_eggs_after_three_days_l559_55985


namespace garden_shape_is_square_l559_55905

theorem garden_shape_is_square (cabbages_this_year : ℕ) (cabbage_increase : ℕ) 
  (h1 : cabbages_this_year = 11236)
  (h2 : cabbage_increase = 211)
  (h3 : ∃ (n : ℕ), n ^ 2 = cabbages_this_year)
  (h4 : ∃ (m : ℕ), m ^ 2 = cabbages_this_year - cabbage_increase) :
  ∃ (side : ℕ), side ^ 2 = cabbages_this_year := by
  sorry

end garden_shape_is_square_l559_55905


namespace fibonacci_divisibility_l559_55937

-- Define Fibonacci sequence
def fib : ℕ → ℕ
  | 0 => 1
  | 1 => 1
  | (n + 2) => fib (n + 1) + fib n

-- Main theorem
theorem fibonacci_divisibility (A B h k : ℕ) : 
  A > 0 → B > 0 → 
  (∃ m : ℕ, B^93 = m * A^19) →
  (∃ n : ℕ, A^93 = n * B^19) →
  (∃ i : ℕ, h = fib i ∧ k = fib (i + 1)) →
  (∃ p : ℕ, (A^4 + B^8)^k = p * (A * B)^h) :=
by sorry

end fibonacci_divisibility_l559_55937


namespace combine_like_terms_1_combine_like_terms_2_l559_55907

-- Define variables
variable (x y : ℝ)

-- Theorem 1
theorem combine_like_terms_1 : 2*x - (x - y) + (x + y) = 2*x + 2*y := by
  sorry

-- Theorem 2
theorem combine_like_terms_2 : 3*x^2 - 9*x + 2 - x^2 + 4*x - 6 = 2*x^2 - 5*x - 4 := by
  sorry

end combine_like_terms_1_combine_like_terms_2_l559_55907


namespace polynomial_existence_l559_55961

theorem polynomial_existence (n : ℕ) : ∃ P : Polynomial ℤ,
  (∀ (i : ℕ), (P.coeff i) ∈ ({0, -1, 1} : Set ℤ)) ∧
  (P.degree ≤ 2^n) ∧
  ((X - 1)^n ∣ P) ∧
  (P ≠ 0) := by
  sorry

end polynomial_existence_l559_55961


namespace vectors_form_basis_l559_55931

theorem vectors_form_basis (a b : ℝ × ℝ) : 
  a = (1, -2) ∧ b = (3, 5) → 
  (∃ (x y : ℝ), ∀ v : ℝ × ℝ, v = x • a + y • b) ∧ 
  (¬ ∃ (k : ℝ), a = k • b) :=
sorry

end vectors_form_basis_l559_55931


namespace seven_balls_four_boxes_l559_55971

/-- The number of ways to distribute n indistinguishable balls into k indistinguishable boxes -/
def distribute_balls (n : ℕ) (k : ℕ) : ℕ := sorry

/-- Theorem: There are 11 ways to distribute 7 indistinguishable balls into 4 indistinguishable boxes -/
theorem seven_balls_four_boxes : distribute_balls 7 4 = 11 := by sorry

end seven_balls_four_boxes_l559_55971


namespace zoo_trip_students_l559_55976

theorem zoo_trip_students (buses : Nat) (students_per_bus : Nat) (car_students : Nat) :
  buses = 7 →
  students_per_bus = 53 →
  car_students = 4 →
  buses * students_per_bus + car_students = 375 := by
  sorry

end zoo_trip_students_l559_55976


namespace least_number_divisibility_l559_55908

theorem least_number_divisibility (n : ℕ) : 
  (∃ k : ℕ, n = 9 * k) ∧ 
  (∀ m : ℕ, m < n → ¬(∃ k : ℕ, m = 9 * k)) ∧
  (∃ r : ℕ, r < 5 ∧ r < 6 ∧ r < 7 ∧ r < 8 ∧
    n % 5 = r ∧ n % 6 = r ∧ n % 7 = r ∧ n % 8 = r) ∧
  n = 1680 →
  n % 5 = 0 ∧ n % 6 = 0 ∧ n % 7 = 0 ∧ n % 8 = 0 :=
by sorry

end least_number_divisibility_l559_55908


namespace problem_statement_l559_55925

theorem problem_statement (a b : ℝ) (ha : a > 0) (hb : b > 0) (h : 1/a + 1/b = 1) :
  ((a - 1) * (b - 1) = 1) ∧
  (∀ x y, x > 0 → y > 0 → 1/x + 1/y = 1 → a + 4*b ≤ x + 4*y) ∧
  (∀ x y, x > 0 → y > 0 → 1/x + 1/y = 1 → 1/a^2 + 2/b^2 ≤ 1/x^2 + 2/y^2) ∧
  (∃ x y, x > 0 ∧ y > 0 ∧ 1/x + 1/y = 1 ∧ a + 4*b = x + 4*y ∧ a + 4*b = 9) ∧
  (∃ x y, x > 0 ∧ y > 0 ∧ 1/x + 1/y = 1 ∧ 1/a^2 + 2/b^2 = 1/x^2 + 2/y^2 ∧ 1/a^2 + 2/b^2 = 2/3) :=
by sorry

end problem_statement_l559_55925


namespace intersection_y_intercept_sum_l559_55938

/-- Given two lines that intersect at a specific point, prove the sum of their y-intercepts. -/
theorem intersection_y_intercept_sum (a b : ℝ) : 
  (∀ x y : ℝ, x = 3 * y + a ∧ y = 3 * x + b → x = 4 ∧ y = 1) →
  a + b = -10 := by
  sorry

end intersection_y_intercept_sum_l559_55938


namespace right_triangle_hypotenuse_l559_55991

theorem right_triangle_hypotenuse : 
  ∀ (a b c : ℝ), 
    a > 0 → b > 0 → c > 0 →
    a^2 + b^2 = c^2 →  -- right-angled triangle condition
    a^2 + b^2 + c^2 = 2500 →  -- sum of squares condition
    c = 25 * Real.sqrt 2 := by
  sorry

end right_triangle_hypotenuse_l559_55991


namespace perez_class_cans_collected_l559_55933

/-- Calculates the total number of cans collected by a class during a food drive. -/
def totalCansCollected (totalStudents : ℕ) (halfStudentsCans : ℕ) (nonCollectingStudents : ℕ) (remainingStudentsCans : ℕ) : ℕ :=
  let halfStudents := totalStudents / 2
  let remainingStudents := totalStudents - halfStudents - nonCollectingStudents
  halfStudents * halfStudentsCans + remainingStudents * remainingStudentsCans

/-- Proves that Ms. Perez's class collected 232 cans in total. -/
theorem perez_class_cans_collected :
  totalCansCollected 30 12 2 4 = 232 := by
  sorry

#eval totalCansCollected 30 12 2 4

end perez_class_cans_collected_l559_55933


namespace second_term_of_geometric_series_l559_55911

/-- Given an infinite geometric series with common ratio 1/4 and sum 16,
    the second term of the sequence is 3. -/
theorem second_term_of_geometric_series (a : ℝ) :
  (∑' n, a * (1/4)^n = 16) → a * (1/4) = 3 := by
  sorry

end second_term_of_geometric_series_l559_55911


namespace line_intersection_l559_55958

/-- Two lines in 2D space -/
structure Line2D where
  point : ℝ × ℝ
  direction : ℝ × ℝ

/-- Check if two lines are parallel -/
def parallel (l1 l2 : Line2D) : Prop :=
  ∃ (c : ℝ), l1.direction = (c * l2.direction.1, c * l2.direction.2)

/-- The problem statement -/
theorem line_intersection (p : ℝ) : 
  let line1 : Line2D := ⟨(2, 3), (5, -8)⟩
  let line2 : Line2D := ⟨(-1, 4), (3, p)⟩
  parallel line1 line2 → p = -24/5 := by
  sorry

end line_intersection_l559_55958


namespace davonte_mercedes_difference_l559_55966

/-- Proves that Davonte ran 2 kilometers farther than Mercedes -/
theorem davonte_mercedes_difference (jonathan_distance : ℝ) 
  (h1 : jonathan_distance = 7.5)
  (mercedes_distance : ℝ) 
  (h2 : mercedes_distance = 2 * jonathan_distance)
  (davonte_distance : ℝ)
  (h3 : mercedes_distance + davonte_distance = 32) :
  davonte_distance - mercedes_distance = 2 := by
  sorry

end davonte_mercedes_difference_l559_55966


namespace trig_expression_equals_negative_one_l559_55919

theorem trig_expression_equals_negative_one :
  let cos30 : ℝ := Real.sqrt 3 / 2
  let sin30 : ℝ := 1 / 2
  let cos60 : ℝ := 1 / 2
  let sin60 : ℝ := Real.sqrt 3 / 2
  (1 - 1 / cos30) * (1 + 1 / sin60) * (1 - 1 / sin30) * (1 + 1 / cos60) = -1 := by
  sorry

end trig_expression_equals_negative_one_l559_55919


namespace circle_diameter_ratio_l559_55974

theorem circle_diameter_ratio (D C : Real) (shaded_ratio : Real) :
  D = 24 →  -- Diameter of circle D
  C < D →   -- Circle C is inside circle D
  shaded_ratio = 7 →  -- Ratio of shaded area to area of circle C
  C = 6 * Real.sqrt 2 := by
  sorry

end circle_diameter_ratio_l559_55974


namespace arman_age_to_40_l559_55902

/-- Given that Arman is six times older than his sister and his sister was 2 years old four years ago,
    prove that Arman will be 40 years old in 4 years. -/
theorem arman_age_to_40 (arman_age sister_age : ℕ) : 
  sister_age = 2 + 4 →  -- Sister's current age
  arman_age = 6 * sister_age →  -- Arman's current age
  40 - arman_age = 4 :=
by sorry

end arman_age_to_40_l559_55902


namespace employee_transportation_difference_l559_55953

/-- Proves the difference between employees who drive and those who take public transportation -/
theorem employee_transportation_difference
  (total_employees : ℕ)
  (drive_percentage : ℚ)
  (public_transport_fraction : ℚ)
  (h_total : total_employees = 200)
  (h_drive : drive_percentage = 3/5)
  (h_public : public_transport_fraction = 1/2) :
  (drive_percentage * total_employees : ℚ) -
  (public_transport_fraction * (total_employees - drive_percentage * total_employees) : ℚ) = 80 := by
  sorry

end employee_transportation_difference_l559_55953


namespace alster_frogs_l559_55970

theorem alster_frogs (alster quinn bret : ℕ) 
  (h1 : quinn = 2 * alster)
  (h2 : bret = 3 * quinn)
  (h3 : bret = 12) :
  alster = 2 := by
sorry

end alster_frogs_l559_55970


namespace min_value_f_l559_55914

def f (c d x : ℝ) : ℝ := x^3 + c*x + d

theorem min_value_f (c d : ℝ) (h : c = 0) : 
  ∀ x, f c d x ≥ d :=
sorry

end min_value_f_l559_55914


namespace olivia_wallet_remainder_l559_55962

/-- Calculates the remaining money in Olivia's wallet after visiting the supermarket. -/
def remaining_money (initial : ℕ) (collected : ℕ) (spent : ℕ) : ℕ :=
  initial + collected - spent

/-- Proves that given the initial amount, collected amount, and spent amount,
    the remaining money in Olivia's wallet is 159 dollars. -/
theorem olivia_wallet_remainder :
  remaining_money 100 148 89 = 159 := by
  sorry

end olivia_wallet_remainder_l559_55962


namespace arithmetic_sequence_ratio_l559_55910

/-- An arithmetic sequence -/
structure ArithmeticSequence where
  first : ℚ
  diff : ℚ

/-- Sum of first n terms of an arithmetic sequence -/
def sum_n_terms (a : ArithmeticSequence) (n : ℕ) : ℚ :=
  n / 2 * (2 * a.first + (n - 1) * a.diff)

/-- The nth term of an arithmetic sequence -/
def nth_term (a : ArithmeticSequence) (n : ℕ) : ℚ :=
  a.first + (n - 1) * a.diff

theorem arithmetic_sequence_ratio (a b : ArithmeticSequence) :
  (∀ n : ℕ, sum_n_terms a n / sum_n_terms b n = n / (n + 1)) →
  nth_term a 4 / nth_term b 4 = 7 / 8 := by
  sorry

end arithmetic_sequence_ratio_l559_55910


namespace both_activities_count_l559_55926

/-- Represents a group of people with preferences for reading books and listening to songs -/
structure GroupPreferences where
  total : ℕ
  book_lovers : ℕ
  song_lovers : ℕ
  both_lovers : ℕ

/-- The principle of inclusion-exclusion for two sets -/
def inclusion_exclusion (g : GroupPreferences) : Prop :=
  g.total = g.book_lovers + g.song_lovers - g.both_lovers

/-- Theorem stating the number of people who like both activities -/
theorem both_activities_count (g : GroupPreferences) 
  (h1 : g.total = 100)
  (h2 : g.book_lovers = 50)
  (h3 : g.song_lovers = 70)
  (h4 : inclusion_exclusion g) : 
  g.both_lovers = 20 := by
  sorry


end both_activities_count_l559_55926


namespace sand_gravel_transport_l559_55960

theorem sand_gravel_transport :
  ∃ (x y : ℕ), 3 * x + 5 * y = 20 ∧ ((x = 5 ∧ y = 1) ∨ (x = 0 ∧ y = 4)) := by
  sorry

end sand_gravel_transport_l559_55960


namespace range_of_k_in_linear_system_l559_55994

/-- Given a system of linear equations and an inequality constraint,
    prove the range of the parameter k. -/
theorem range_of_k_in_linear_system (x y k : ℝ) :
  (2 * x - y = k + 1) →
  (x - y = -3) →
  (x + y > 2) →
  k > -4.5 := by
  sorry

end range_of_k_in_linear_system_l559_55994


namespace carpet_dimensions_l559_55935

/-- Represents a rectangular carpet -/
structure Carpet where
  length : ℕ
  width : ℕ

/-- Represents a rectangular room -/
structure Room where
  length : ℕ
  width : ℕ

/-- Check if a carpet fits perfectly in a room -/
def fits_perfectly (c : Carpet) (r : Room) : Prop :=
  c.length * c.length + c.width * c.width = r.length * r.length + r.width * r.width

/-- The main theorem -/
theorem carpet_dimensions (c : Carpet) (r1 r2 : Room) (h1 : r1.length = r2.length)
  (h2 : r1.width = 38) (h3 : r2.width = 50) (h4 : fits_perfectly c r1) (h5 : fits_perfectly c r2) :
  c.length = 25 ∧ c.width = 50 := by
  sorry

#check carpet_dimensions

end carpet_dimensions_l559_55935


namespace scrap_cookie_radius_l559_55952

theorem scrap_cookie_radius 
  (R : ℝ) 
  (r : ℝ) 
  (n : ℕ) 
  (h1 : R = 3.5) 
  (h2 : r = 1) 
  (h3 : n = 9) : 
  ∃ (x : ℝ), x^2 = R^2 * π - n * r^2 * π ∧ x = Real.sqrt 3.25 := by
  sorry

end scrap_cookie_radius_l559_55952


namespace final_value_calculation_l559_55945

theorem final_value_calculation (initial_number : ℕ) : 
  initial_number = 10 → 3 * (2 * initial_number + 8) = 84 := by
  sorry

end final_value_calculation_l559_55945

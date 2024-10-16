import Mathlib

namespace NUMINAMATH_CALUDE_no_valid_polygon_pairs_l2429_242998

theorem no_valid_polygon_pairs : ¬∃ (y l : ℕ), 
  (∃ (k : ℕ), y = 30 * k) ∧ 
  (l > 1) ∧
  (∃ (n : ℕ), y = 180 - 360 / n) ∧
  (∃ (m : ℕ), l * y = 180 - 360 / m) := by
  sorry

end NUMINAMATH_CALUDE_no_valid_polygon_pairs_l2429_242998


namespace NUMINAMATH_CALUDE_son_work_time_l2429_242922

-- Define the work rates
def man_rate : ℚ := 1 / 5
def combined_rate : ℚ := 1 / 4

-- Define the son's work rate
def son_rate : ℚ := combined_rate - man_rate

-- Theorem statement
theorem son_work_time :
  son_rate = 1 / 20 ∧ (1 / son_rate : ℚ) = 20 := by
  sorry

end NUMINAMATH_CALUDE_son_work_time_l2429_242922


namespace NUMINAMATH_CALUDE_system_solution_l2429_242900

theorem system_solution (a b : ℝ) : 
  (∃ x y : ℝ, b * x - 3 * y = 2 ∧ a * x + y = 2) ∧
  (b * 4 - 3 * 2 = 2 ∧ a * 4 + 2 = 2) →
  a = 0 ∧ b = 2 := by sorry

end NUMINAMATH_CALUDE_system_solution_l2429_242900


namespace NUMINAMATH_CALUDE_range_of_m_l2429_242950

/-- Given that p: m - 1 < x < m + 1, q: (x - 2)(x - 6) < 0, and q is a necessary but not sufficient
condition for p, prove that the range of values for m is [3, 5]. -/
theorem range_of_m (m x : ℝ) 
  (hp : m - 1 < x ∧ x < m + 1)
  (hq : (x - 2) * (x - 6) < 0)
  (h_nec_not_suff : ∀ y, (m - 1 < y ∧ y < m + 1) → (y - 2) * (y - 6) < 0)
  (h_not_suff : ∃ z, (z - 2) * (z - 6) < 0 ∧ ¬(m - 1 < z ∧ z < m + 1)) :
  3 ≤ m ∧ m ≤ 5 := by
sorry

end NUMINAMATH_CALUDE_range_of_m_l2429_242950


namespace NUMINAMATH_CALUDE_class_average_problem_l2429_242916

theorem class_average_problem (avg_class1 avg_combined : ℝ) (n1 n2 : ℕ) 
  (h1 : avg_class1 = 40)
  (h2 : n1 = 24)
  (h3 : n2 = 50)
  (h4 : avg_combined = 53.513513513513516)
  (h5 : (n1 : ℝ) * avg_class1 + (n2 : ℝ) * (((n1 + n2 : ℕ) : ℝ) * avg_combined - (n1 : ℝ) * avg_class1) / (n2 : ℝ) = 
        (n1 + n2 : ℕ) * avg_combined) :
  (((n1 + n2 : ℕ) : ℝ) * avg_combined - (n1 : ℝ) * avg_class1) / (n2 : ℝ) = 60 := by
  sorry

#check class_average_problem

end NUMINAMATH_CALUDE_class_average_problem_l2429_242916


namespace NUMINAMATH_CALUDE_pollen_mass_scientific_notation_l2429_242902

/-- Represents a number in scientific notation -/
structure ScientificNotation where
  coefficient : ℝ
  exponent : ℤ
  h_coefficient : 1 ≤ coefficient ∧ coefficient < 10

/-- Converts a real number to scientific notation -/
def toScientificNotation (x : ℝ) : ScientificNotation :=
  sorry

theorem pollen_mass_scientific_notation :
  let mass : ℝ := 0.000037
  let scientific := toScientificNotation mass
  scientific.coefficient = 3.7 ∧ scientific.exponent = -5 :=
sorry

end NUMINAMATH_CALUDE_pollen_mass_scientific_notation_l2429_242902


namespace NUMINAMATH_CALUDE_tesla_ownership_l2429_242928

/-- The number of Teslas owned by different individuals and their relationships. -/
theorem tesla_ownership (chris sam elon : ℕ) : 
  chris = 6 → 
  sam = chris / 2 → 
  elon = 13 → 
  elon - sam = 10 := by
sorry

end NUMINAMATH_CALUDE_tesla_ownership_l2429_242928


namespace NUMINAMATH_CALUDE_geometric_progression_equality_l2429_242972

theorem geometric_progression_equality (a r : ℝ) (n : ℕ) (hr : r ≠ 1) :
  let S : ℕ → ℝ := λ m ↦ a * (r^m - 1) / (r - 1)
  (S n) / (S (2*n) - S n) = (S (2*n) - S n) / (S (3*n) - S (2*n)) := by
  sorry

end NUMINAMATH_CALUDE_geometric_progression_equality_l2429_242972


namespace NUMINAMATH_CALUDE_newberg_airport_passengers_l2429_242958

theorem newberg_airport_passengers : 
  let on_time_passengers : ℕ := 14507
  let late_passengers : ℕ := 213
  on_time_passengers + late_passengers = 14720 := by sorry

end NUMINAMATH_CALUDE_newberg_airport_passengers_l2429_242958


namespace NUMINAMATH_CALUDE_jameson_medals_l2429_242960

theorem jameson_medals (total_medals track_medals : ℕ) 
  (h1 : total_medals = 20)
  (h2 : track_medals = 5)
  (h3 : ∃ swimming_medals : ℕ, swimming_medals = 2 * track_medals) :
  ∃ badminton_medals : ℕ, badminton_medals = total_medals - (track_medals + 2 * track_medals) ∧ badminton_medals = 5 := by
  sorry

end NUMINAMATH_CALUDE_jameson_medals_l2429_242960


namespace NUMINAMATH_CALUDE_binomial_510_510_l2429_242985

theorem binomial_510_510 : (510 : ℕ).choose 510 = 1 := by sorry

end NUMINAMATH_CALUDE_binomial_510_510_l2429_242985


namespace NUMINAMATH_CALUDE_hank_lawn_mowing_earnings_l2429_242997

/-- Proves that Hank made $50 from mowing lawns given the specified conditions -/
theorem hank_lawn_mowing_earnings :
  let carwash_earnings : ℝ := 100
  let carwash_donation_rate : ℝ := 0.9
  let bake_sale_earnings : ℝ := 80
  let bake_sale_donation_rate : ℝ := 0.75
  let lawn_mowing_donation_rate : ℝ := 1
  let total_donation : ℝ := 200
  let lawn_mowing_earnings : ℝ := 
    total_donation - 
    (carwash_earnings * carwash_donation_rate + 
     bake_sale_earnings * bake_sale_donation_rate)
  lawn_mowing_earnings = 50 := by sorry

end NUMINAMATH_CALUDE_hank_lawn_mowing_earnings_l2429_242997


namespace NUMINAMATH_CALUDE_alberts_earnings_increase_l2429_242991

theorem alberts_earnings_increase (E : ℝ) (p : ℝ) 
  (h1 : 1.27 * E = 567)
  (h2 : E + p * E = 562.54) :
  p = 0.26 := by
sorry

end NUMINAMATH_CALUDE_alberts_earnings_increase_l2429_242991


namespace NUMINAMATH_CALUDE_movie_production_cost_l2429_242981

def opening_weekend_revenue : ℝ := 120000000
def total_revenue_multiplier : ℝ := 3.5
def production_company_share : ℝ := 0.60
def profit : ℝ := 192000000

theorem movie_production_cost :
  let total_revenue := opening_weekend_revenue * total_revenue_multiplier
  let production_company_revenue := total_revenue * production_company_share
  let production_cost := production_company_revenue - profit
  production_cost = 60000000 := by sorry

end NUMINAMATH_CALUDE_movie_production_cost_l2429_242981


namespace NUMINAMATH_CALUDE_health_drink_sales_correct_l2429_242910

/-- Represents the health drink inventory and sales data -/
structure HealthDrinkSales where
  first_batch_cost : ℝ
  second_batch_cost : ℝ
  unit_price_increase : ℝ
  selling_price : ℝ
  discounted_quantity : ℕ
  discount_rate : ℝ

/-- Calculates the quantity of the first batch and the total profit -/
def calculate_quantity_and_profit (sales : HealthDrinkSales) : ℝ × ℝ :=
  sorry

/-- Theorem stating the correctness of the calculation -/
theorem health_drink_sales_correct (sales : HealthDrinkSales) 
  (h1 : sales.first_batch_cost = 40000)
  (h2 : sales.second_batch_cost = 88000)
  (h3 : sales.unit_price_increase = 2)
  (h4 : sales.selling_price = 28)
  (h5 : sales.discounted_quantity = 100)
  (h6 : sales.discount_rate = 0.2) :
  let (quantity, profit) := calculate_quantity_and_profit sales
  quantity = 2000 ∧ profit = 39440 :=
sorry

end NUMINAMATH_CALUDE_health_drink_sales_correct_l2429_242910


namespace NUMINAMATH_CALUDE_larger_part_is_30_l2429_242938

theorem larger_part_is_30 (x y : ℕ) 
  (sum_eq_52 : x + y = 52) 
  (weighted_sum_eq_780 : 10 * x + 22 * y = 780) : 
  max x y = 30 := by
sorry

end NUMINAMATH_CALUDE_larger_part_is_30_l2429_242938


namespace NUMINAMATH_CALUDE_min_value_expression_l2429_242968

theorem min_value_expression (a : ℝ) (h : a > 0) :
  (a - 1) * (4 * a - 1) / a ≥ -1 ∧
  ∃ a₀ > 0, (a₀ - 1) * (4 * a₀ - 1) / a₀ = -1 :=
sorry

end NUMINAMATH_CALUDE_min_value_expression_l2429_242968


namespace NUMINAMATH_CALUDE_sin_cos_difference_equals_sqrt3_over_2_l2429_242901

theorem sin_cos_difference_equals_sqrt3_over_2 :
  Real.sin (135 * π / 180) * Real.cos (15 * π / 180) - 
  Real.cos (45 * π / 180) * Real.sin (-15 * π / 180) = 
  Real.sqrt 3 / 2 := by
  sorry

end NUMINAMATH_CALUDE_sin_cos_difference_equals_sqrt3_over_2_l2429_242901


namespace NUMINAMATH_CALUDE_quadratic_has_two_distinct_roots_l2429_242969

theorem quadratic_has_two_distinct_roots 
  (a b c : ℝ) 
  (h1 : 5*a + 3*b + 2*c = 0) 
  (h2 : a ≠ 0) : 
  ∃ x y : ℝ, x ≠ y ∧ a*x^2 + b*x + c = 0 ∧ a*y^2 + b*y + c = 0 :=
sorry

end NUMINAMATH_CALUDE_quadratic_has_two_distinct_roots_l2429_242969


namespace NUMINAMATH_CALUDE_bela_wins_iff_m_odd_l2429_242984

/-- The game interval --/
def GameInterval (m : ℕ) := Set.Icc (0 : ℝ) m

/-- Predicate for a valid move --/
def ValidMove (m : ℕ) (prev_moves : List ℝ) (x : ℝ) : Prop :=
  x ∈ GameInterval m ∧ ∀ y ∈ prev_moves, |x - y| > 2

/-- The game result --/
inductive GameResult
  | BelaWins
  | JennWins

/-- The game outcome based on the optimal strategy --/
def GameOutcome (m : ℕ) : GameResult :=
  if m % 2 = 1 then GameResult.BelaWins else GameResult.JennWins

/-- The main theorem --/
theorem bela_wins_iff_m_odd (m : ℕ) (h : m > 2) :
  GameOutcome m = GameResult.BelaWins ↔ m % 2 = 1 := by sorry

end NUMINAMATH_CALUDE_bela_wins_iff_m_odd_l2429_242984


namespace NUMINAMATH_CALUDE_rectangle_to_triangle_altitude_l2429_242941

/-- Given a 7x21 rectangle that can be rearranged into a triangle, 
    prove that if the base of the triangle is 21 units, 
    then the altitude of the triangle is 14 units. -/
theorem rectangle_to_triangle_altitude 
  (rectangle_width : ℝ) 
  (rectangle_height : ℝ) 
  (triangle_base : ℝ) 
  (h1 : rectangle_width = 7)
  (h2 : rectangle_height = 21)
  (h3 : triangle_base = rectangle_height) :
  let rectangle_area := rectangle_width * rectangle_height
  let triangle_altitude := (2 * rectangle_area) / triangle_base
  triangle_altitude = 14 := by
sorry

end NUMINAMATH_CALUDE_rectangle_to_triangle_altitude_l2429_242941


namespace NUMINAMATH_CALUDE_student_arrangement_count_l2429_242948

/-- The number of ways to select and arrange students with non-adjacent boys -/
def student_arrangements (num_boys num_girls select_boys select_girls : ℕ) : ℕ :=
  Nat.choose num_boys select_boys *
  Nat.choose num_girls select_girls *
  Nat.factorial select_girls *
  Nat.factorial (select_girls + 1)

/-- Theorem: The number of arrangements of 2 boys from 4 and 3 girls from 6,
    where the boys are not adjacent, is 8640 -/
theorem student_arrangement_count :
  student_arrangements 4 6 2 3 = 8640 := by
sorry

end NUMINAMATH_CALUDE_student_arrangement_count_l2429_242948


namespace NUMINAMATH_CALUDE_max_value_of_function_l2429_242915

theorem max_value_of_function (x : ℝ) (h : -1 < x ∧ x < 1) : 
  (∀ y : ℝ, -1 < y ∧ y < 1 → x / (x - 1) + x ≥ y / (y - 1) + y) → 
  x / (x - 1) + x = 0 :=
by sorry

end NUMINAMATH_CALUDE_max_value_of_function_l2429_242915


namespace NUMINAMATH_CALUDE_negative_a_sixth_div_a_cube_l2429_242911

theorem negative_a_sixth_div_a_cube (a : ℝ) : (-a)^6 / a^3 = a^3 := by sorry

end NUMINAMATH_CALUDE_negative_a_sixth_div_a_cube_l2429_242911


namespace NUMINAMATH_CALUDE_backyard_area_l2429_242926

/-- A rectangular backyard satisfying certain conditions -/
structure Backyard where
  length : ℝ
  width : ℝ
  length_condition : 25 * length = 1000
  perimeter_condition : 10 * (2 * (length + width)) = 1000

/-- The area of a backyard is 400 square meters -/
theorem backyard_area (b : Backyard) : b.length * b.width = 400 := by
  sorry


end NUMINAMATH_CALUDE_backyard_area_l2429_242926


namespace NUMINAMATH_CALUDE_agate_precious_stones_l2429_242923

theorem agate_precious_stones (agate olivine diamond : ℕ) : 
  olivine = agate + 5 →
  diamond = olivine + 11 →
  agate + olivine + diamond = 111 →
  agate = 30 := by
sorry

end NUMINAMATH_CALUDE_agate_precious_stones_l2429_242923


namespace NUMINAMATH_CALUDE_integer_solution_range_l2429_242931

theorem integer_solution_range (b : ℝ) : 
  (∀ x : ℤ, |3 * (x : ℝ) - b| < 4 ↔ x = 1 ∨ x = 2 ∨ x = 3) → 
  (5 < b ∧ b < 7) :=
by sorry

end NUMINAMATH_CALUDE_integer_solution_range_l2429_242931


namespace NUMINAMATH_CALUDE_laptop_price_l2429_242909

theorem laptop_price : ∃ (x : ℝ), x = 400 ∧ 
  (∃ (price_C price_D : ℝ), 
    price_C = 0.8 * x - 60 ∧ 
    price_D = 0.7 * x ∧ 
    price_D - price_C = 20) := by
  sorry

end NUMINAMATH_CALUDE_laptop_price_l2429_242909


namespace NUMINAMATH_CALUDE_inequality_proof_l2429_242936

theorem inequality_proof (a b : ℝ) : (a^4 + a^2*b^2 + b^4) / 3 ≥ (a^3*b + b^3*a) / 2 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l2429_242936


namespace NUMINAMATH_CALUDE_factorial_500_trailing_zeros_l2429_242963

/-- The number of trailing zeros in n! -/
def trailingZeros (n : ℕ) : ℕ :=
  (n / 5) + (n / 25) + (n / 125)

theorem factorial_500_trailing_zeros :
  trailingZeros 500 = 124 := by
  sorry

end NUMINAMATH_CALUDE_factorial_500_trailing_zeros_l2429_242963


namespace NUMINAMATH_CALUDE_evening_temp_calculation_l2429_242976

/-- Given a noon temperature and a temperature drop, calculate the evening temperature. -/
def evening_temperature (noon_temp : ℤ) (temp_drop : ℕ) : ℤ :=
  noon_temp - temp_drop

/-- Theorem: If the noon temperature is 2°C and it drops by 3°C, the evening temperature is -1°C. -/
theorem evening_temp_calculation :
  evening_temperature 2 3 = -1 := by
  sorry

end NUMINAMATH_CALUDE_evening_temp_calculation_l2429_242976


namespace NUMINAMATH_CALUDE_hyper_box_side_sum_l2429_242973

/-- The sum of side lengths of a four-dimensional rectangular hyper-box with given face volumes -/
theorem hyper_box_side_sum (W X Y Z : ℝ) 
  (h1 : W * X * Y = 60)
  (h2 : W * X * Z = 80)
  (h3 : W * Y * Z = 120)
  (h4 : X * Y * Z = 60) :
  W + X + Y + Z = 318.5 := by
  sorry

end NUMINAMATH_CALUDE_hyper_box_side_sum_l2429_242973


namespace NUMINAMATH_CALUDE_steves_return_speed_l2429_242937

/-- Proves that given a round trip of 35 km each way, where the return speed is twice the outbound speed, 
    and the total travel time is 6 hours, the return speed is 17.5 km/h. -/
theorem steves_return_speed (distance : ℝ) (total_time : ℝ) : 
  distance = 35 →
  total_time = 6 →
  ∃ (outbound_speed : ℝ),
    outbound_speed > 0 ∧
    distance / outbound_speed + distance / (2 * outbound_speed) = total_time ∧
    2 * outbound_speed = 17.5 := by
  sorry

end NUMINAMATH_CALUDE_steves_return_speed_l2429_242937


namespace NUMINAMATH_CALUDE_multiply_64_56_l2429_242906

theorem multiply_64_56 : 64 * 56 = 3584 := by
  sorry

end NUMINAMATH_CALUDE_multiply_64_56_l2429_242906


namespace NUMINAMATH_CALUDE_quadratic_intersection_and_vertex_l2429_242917

-- Define the quadratic function
def f (m : ℝ) (x : ℝ) : ℝ := x^2 + 2*(m+1)*x - m + 1

-- Define the discriminant of the quadratic function
def discriminant (m : ℝ) : ℝ := 4*(m^2 + 3*m)

-- Define the x-coordinate of the vertex
def vertex_x (m : ℝ) : ℝ := -(m + 1)

-- Define the y-coordinate of the vertex
def vertex_y (m : ℝ) : ℝ := -(m^2 + 3*m)

theorem quadratic_intersection_and_vertex (m : ℝ) :
  -- Part 1: The number of intersection points with the x-axis is 0, 1, or 2
  (∃ x : ℝ, f m x = 0 ∧ 
    (∀ y : ℝ, f m y = 0 → y = x ∨ 
    (∃ z : ℝ, z ≠ x ∧ z ≠ y ∧ f m z = 0))) ∨
  (∀ x : ℝ, f m x ≠ 0) ∧
  -- Part 2: If the line y = x + 1 passes through the vertex, then m = -2 or m = 0
  (vertex_y m = vertex_x m + 1 → m = -2 ∨ m = 0) :=
sorry

end NUMINAMATH_CALUDE_quadratic_intersection_and_vertex_l2429_242917


namespace NUMINAMATH_CALUDE_power_of_five_sum_equality_l2429_242966

theorem power_of_five_sum_equality (x : ℕ) : 5^6 + 5^6 + 5^6 = 5^x ↔ x = 6 := by
  sorry

end NUMINAMATH_CALUDE_power_of_five_sum_equality_l2429_242966


namespace NUMINAMATH_CALUDE_smallest_y_value_l2429_242990

theorem smallest_y_value (y : ℝ) : 
  (3 * y^2 + 33 * y - 90 = y * (y + 18)) → y ≥ -18 :=
by sorry

end NUMINAMATH_CALUDE_smallest_y_value_l2429_242990


namespace NUMINAMATH_CALUDE_perpendicular_line_through_point_l2429_242952

-- Define the given line
def given_line (x y : ℝ) : Prop := 2 * x + y - 5 = 0

-- Define the point A
def point_A : ℝ × ℝ := (2, 3)

-- Define the perpendicular line
def perpendicular_line (x y : ℝ) : Prop := x - 2 * y + 4 = 0

-- Theorem statement
theorem perpendicular_line_through_point :
  (perpendicular_line point_A.1 point_A.2) ∧
  (∀ x y : ℝ, perpendicular_line x y → given_line x y →
    (y - point_A.2) = 2 * (x - point_A.1)) :=
sorry

end NUMINAMATH_CALUDE_perpendicular_line_through_point_l2429_242952


namespace NUMINAMATH_CALUDE_circle_equation_solution_l2429_242956

theorem circle_equation_solution :
  ∃! (x y : ℝ), (x - 12)^2 + (y - 13)^2 + (x - y)^2 = 1/3 ∧ 
  x = 37/3 ∧ y = 38/3 := by
sorry

end NUMINAMATH_CALUDE_circle_equation_solution_l2429_242956


namespace NUMINAMATH_CALUDE_sum_of_digits_l2429_242903

theorem sum_of_digits (a b c d : ℕ) : 
  a < 10 ∧ b < 10 ∧ c < 10 ∧ d < 10 →
  a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d →
  100 * a + 10 * b + c + 100 * d + 10 * c + b = 1100 →
  a + b + c + d = 20 := by
sorry

end NUMINAMATH_CALUDE_sum_of_digits_l2429_242903


namespace NUMINAMATH_CALUDE_line_passes_through_fixed_point_l2429_242934

/-- The line x + (l-m)y + 3 = 0 always passes through the point (-3, 0) for any real number m -/
theorem line_passes_through_fixed_point :
  ∀ (m : ℝ), (-3 : ℝ) + (1 - m) * (0 : ℝ) + 3 = 0 := by
  sorry

end NUMINAMATH_CALUDE_line_passes_through_fixed_point_l2429_242934


namespace NUMINAMATH_CALUDE_tanner_money_left_l2429_242940

def savings : List ℝ := [17, 48, 25, 55]
def video_game_price : ℝ := 49
def shoes_price : ℝ := 65
def discount_rate : ℝ := 0.1
def tax_rate : ℝ := 0.05

def total_savings : ℝ := savings.sum

def discounted_video_game_price : ℝ := video_game_price * (1 - discount_rate)

def total_cost_before_tax : ℝ := discounted_video_game_price + shoes_price

def sales_tax : ℝ := total_cost_before_tax * tax_rate

def total_cost_with_tax : ℝ := total_cost_before_tax + sales_tax

def money_left : ℝ := total_savings - total_cost_with_tax

theorem tanner_money_left :
  ∃ (ε : ℝ), money_left = 30.44 + ε ∧ abs ε < 0.005 := by
  sorry

end NUMINAMATH_CALUDE_tanner_money_left_l2429_242940


namespace NUMINAMATH_CALUDE_problem_solution_l2429_242933

/-- The problem setup and proof statements -/
theorem problem_solution :
  let A : ℝ × ℝ := (1, 3)
  let B : ℝ × ℝ := (2, -2)
  let C : ℝ × ℝ := (4, 1)
  let D : ℝ × ℝ := (5, -4)
  let a : ℝ × ℝ := (1, -5)
  let b : ℝ × ℝ := (2, 3)
  let k : ℝ := -1/3
  -- Part 1
  (B.1 - A.1, B.2 - A.2) = (D.1 - C.1, D.2 - C.2) ∧
  -- Part 2
  ∃ (t : ℝ), t ≠ 0 ∧ (k * a.1 - b.1, k * a.2 - b.2) = (t * (a.1 + 3 * b.1), t * (a.2 + 3 * b.2)) :=
by sorry

end NUMINAMATH_CALUDE_problem_solution_l2429_242933


namespace NUMINAMATH_CALUDE_sqrt_164_between_12_and_13_l2429_242908

theorem sqrt_164_between_12_and_13 : 12 < Real.sqrt 164 ∧ Real.sqrt 164 < 13 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_164_between_12_and_13_l2429_242908


namespace NUMINAMATH_CALUDE_rhombus_existence_and_uniqueness_l2429_242978

/-- Represents a rhombus -/
structure Rhombus where
  side : ℝ
  diag1 : ℝ
  diag2 : ℝ
  angle : ℝ

/-- Given the sum of diagonals and an opposite angle, a unique rhombus can be determined -/
theorem rhombus_existence_and_uniqueness 
  (diag_sum : ℝ) 
  (opp_angle : ℝ) 
  (h_pos : diag_sum > 0) 
  (h_angle : 0 < opp_angle ∧ opp_angle < π) :
  ∃! r : Rhombus, r.diag1 + r.diag2 = diag_sum ∧ r.angle = opp_angle :=
sorry

end NUMINAMATH_CALUDE_rhombus_existence_and_uniqueness_l2429_242978


namespace NUMINAMATH_CALUDE_passes_through_fixed_point_not_in_fourth_quadrant_min_area_and_equation_l2429_242942

/-- Definition of the line l with parameter k -/
def line_l (k : ℝ) (x y : ℝ) : Prop := k * x - y + 1 + 2 * k = 0

/-- The fixed point that the line passes through -/
def fixed_point : ℝ × ℝ := (-2, 1)

/-- Theorem 1: The line passes through the fixed point for all real k -/
theorem passes_through_fixed_point (k : ℝ) :
  line_l k (fixed_point.1) (fixed_point.2) := by sorry

/-- Theorem 2: The line does not pass through the fourth quadrant iff k ≥ 0 -/
theorem not_in_fourth_quadrant (k : ℝ) :
  (∀ x y, x > 0 → y < 0 → ¬line_l k x y) ↔ k ≥ 0 := by sorry

/-- Function to calculate the area of the triangle formed by the line's intersections -/
noncomputable def triangle_area (k : ℝ) : ℝ :=
  if k ≠ 0 then
    (1 + 2 * k) * ((1 + 2 * k) / k) / 2
  else 0

/-- Theorem 3: The minimum area of the triangle is 4, occurring when k = 1/2 -/
theorem min_area_and_equation :
  (∀ k, k > 0 → triangle_area k ≥ 4) ∧
  triangle_area (1/2) = 4 ∧
  line_l (1/2) x y ↔ x - 2 * y + 4 = 0 := by sorry

end NUMINAMATH_CALUDE_passes_through_fixed_point_not_in_fourth_quadrant_min_area_and_equation_l2429_242942


namespace NUMINAMATH_CALUDE_arccos_equation_solution_l2429_242992

theorem arccos_equation_solution :
  ∀ x : ℝ, (Real.arccos (3 * x) - Real.arccos x = π / 6) ↔ (x = 1/12 ∨ x = -1/12) :=
by sorry

end NUMINAMATH_CALUDE_arccos_equation_solution_l2429_242992


namespace NUMINAMATH_CALUDE_greatest_divisor_with_remainders_l2429_242924

theorem greatest_divisor_with_remainders (a b r1 r2 : ℕ) (ha : a = 6215) (hb : b = 7373) (hr1 : r1 = 23) (hr2 : r2 = 29) :
  Nat.gcd (a - r1) (b - r2) = 96 := by
  sorry

end NUMINAMATH_CALUDE_greatest_divisor_with_remainders_l2429_242924


namespace NUMINAMATH_CALUDE_share_of_b_l2429_242974

theorem share_of_b (a b c : ℕ) : 
  a = 3 * b → 
  b = c + 25 → 
  a + b + c = 645 → 
  b = 134 := by
sorry

end NUMINAMATH_CALUDE_share_of_b_l2429_242974


namespace NUMINAMATH_CALUDE_double_markup_percentage_l2429_242994

theorem double_markup_percentage (original_price : ℝ) (markup_percentage : ℝ) : 
  markup_percentage = 40 →
  let first_markup := original_price * (1 + markup_percentage / 100)
  let second_markup := first_markup * (1 + markup_percentage / 100)
  (second_markup - original_price) / original_price * 100 = 96 := by
sorry

end NUMINAMATH_CALUDE_double_markup_percentage_l2429_242994


namespace NUMINAMATH_CALUDE_golf_balls_needed_l2429_242953

def weekend_goal : ℕ := 48
def saturday_balls : ℕ := 16
def sunday_balls : ℕ := 18

theorem golf_balls_needed : weekend_goal - (saturday_balls + sunday_balls) = 14 := by
  sorry

end NUMINAMATH_CALUDE_golf_balls_needed_l2429_242953


namespace NUMINAMATH_CALUDE_range_of_x_l2429_242964

def P (x : ℝ) : Prop := (x + 1) / (x - 3) ≥ 0

def Q (x : ℝ) : Prop := |1 - x/2| < 1

theorem range_of_x (x : ℝ) : 
  P x ∧ ¬Q x ↔ x ≤ -1 ∨ x ≥ 4 :=
by sorry

end NUMINAMATH_CALUDE_range_of_x_l2429_242964


namespace NUMINAMATH_CALUDE_max_tangent_segment_length_l2429_242951

/-- Given a triangle ABC with perimeter 2p, the maximum length of a segment
    parallel to BC and tangent to the inscribed circle is p/4, and this
    maximum is achieved when BC = p/2. -/
theorem max_tangent_segment_length (p : ℝ) (h : p > 0) :
  ∃ (a b c : ℝ),
    a > 0 ∧ b > 0 ∧ c > 0 ∧
    a + b + c = 2 * p ∧
    (∀ (x y z : ℝ),
      x > 0 → y > 0 → z > 0 → x + y + z = 2 * p →
      x * (p - x) / p ≤ p / 4) ∧
    a * (p - a) / p = p / 4 ∧
    a = p / 2 := by
  sorry


end NUMINAMATH_CALUDE_max_tangent_segment_length_l2429_242951


namespace NUMINAMATH_CALUDE_C_symmetric_origin_C_area_greater_than_pi_l2429_242979

-- Define the curve C
def C : Set (ℝ × ℝ) := {p : ℝ × ℝ | p.1^4 + p.2^2 = 1}

-- Symmetry with respect to the origin
theorem C_symmetric_origin : ∀ (x y : ℝ), (x, y) ∈ C ↔ (-x, -y) ∈ C := by sorry

-- Area enclosed by C is greater than π
theorem C_area_greater_than_pi : ∃ (A : ℝ), A > π ∧ (∀ (x y : ℝ), (x, y) ∈ C → x^2 + y^2 ≤ A) := by sorry

end NUMINAMATH_CALUDE_C_symmetric_origin_C_area_greater_than_pi_l2429_242979


namespace NUMINAMATH_CALUDE_union_when_a_eq_2_union_eq_B_iff_l2429_242954

-- Define set A
def A : Set ℝ := {x | (x - 1) / (x - 2) ≤ 1 / 2}

-- Define set B parameterized by a
def B (a : ℝ) : Set ℝ := {x | x^2 - (a + 2) * x + 2 * a ≤ 0}

-- Theorem for part (1)
theorem union_when_a_eq_2 : A ∪ B 2 = {x | 0 ≤ x ∧ x ≤ 2} := by sorry

-- Theorem for part (2)
theorem union_eq_B_iff (a : ℝ) : A ∪ B a = B a ↔ a ≤ 0 := by sorry

end NUMINAMATH_CALUDE_union_when_a_eq_2_union_eq_B_iff_l2429_242954


namespace NUMINAMATH_CALUDE_total_miles_driven_l2429_242947

/-- The total miles driven by Darius and Julia -/
def total_miles (darius_miles julia_miles : ℕ) : ℕ :=
  darius_miles + julia_miles

/-- Theorem stating that the total miles driven by Darius and Julia is 1677 -/
theorem total_miles_driven :
  total_miles 679 998 = 1677 := by
  sorry

end NUMINAMATH_CALUDE_total_miles_driven_l2429_242947


namespace NUMINAMATH_CALUDE_composition_ratio_l2429_242989

def f (x : ℝ) : ℝ := 3 * x + 4

def g (x : ℝ) : ℝ := 4 * x - 3

theorem composition_ratio : f (g (f 3)) / g (f (g 3)) = 151 / 121 := by
  sorry

end NUMINAMATH_CALUDE_composition_ratio_l2429_242989


namespace NUMINAMATH_CALUDE_sin_sixty_degrees_l2429_242904

theorem sin_sixty_degrees : Real.sin (π / 3) = Real.sqrt 3 / 2 := by sorry

end NUMINAMATH_CALUDE_sin_sixty_degrees_l2429_242904


namespace NUMINAMATH_CALUDE_max_triangles_three_families_ten_lines_l2429_242967

/-- Represents a family of parallel lines -/
structure ParallelLineFamily :=
  (num_lines : ℕ)

/-- Represents the configuration of three families of parallel lines -/
structure ThreeParallelLineFamilies :=
  (family1 : ParallelLineFamily)
  (family2 : ParallelLineFamily)
  (family3 : ParallelLineFamily)

/-- Calculates the maximum number of triangles formed by three families of parallel lines -/
def max_triangles (config : ThreeParallelLineFamilies) : ℕ :=
  sorry

/-- Theorem stating that the maximum number of triangles formed by three families of 10 parallel lines is 150 -/
theorem max_triangles_three_families_ten_lines :
  ∀ (config : ThreeParallelLineFamilies),
    config.family1.num_lines = 10 →
    config.family2.num_lines = 10 →
    config.family3.num_lines = 10 →
    max_triangles config = 150 :=
  sorry

end NUMINAMATH_CALUDE_max_triangles_three_families_ten_lines_l2429_242967


namespace NUMINAMATH_CALUDE_magnitude_of_b_l2429_242957

def a : ℝ × ℝ := (2, 3)

theorem magnitude_of_b (b : ℝ × ℝ) 
  (h : (a.1 + b.1, a.2 + b.2) • (a.1 - b.1, a.2 - b.2) = 0) : 
  Real.sqrt (b.1^2 + b.2^2) = Real.sqrt 13 := by
  sorry

end NUMINAMATH_CALUDE_magnitude_of_b_l2429_242957


namespace NUMINAMATH_CALUDE_average_rainfall_virginia_l2429_242927

theorem average_rainfall_virginia (march april may june july : ℝ) 
  (h_march : march = 3.79)
  (h_april : april = 4.5)
  (h_may : may = 3.95)
  (h_june : june = 3.09)
  (h_july : july = 4.67) :
  (march + april + may + june + july) / 5 = 4 := by
  sorry

end NUMINAMATH_CALUDE_average_rainfall_virginia_l2429_242927


namespace NUMINAMATH_CALUDE_elephant_received_503_pills_l2429_242943

/-- The number of pills given to four animals by Dr. Aibolit -/
def total_pills : ℕ := 2006

/-- The number of pills received by the crocodile -/
def crocodile_pills : ℕ := sorry

/-- The number of pills received by the rhinoceros -/
def rhinoceros_pills : ℕ := crocodile_pills + 1

/-- The number of pills received by the hippopotamus -/
def hippopotamus_pills : ℕ := rhinoceros_pills + 1

/-- The number of pills received by the elephant -/
def elephant_pills : ℕ := hippopotamus_pills + 1

/-- Theorem stating that the elephant received 503 pills -/
theorem elephant_received_503_pills : 
  crocodile_pills + rhinoceros_pills + hippopotamus_pills + elephant_pills = total_pills ∧ 
  elephant_pills = 503 := by
  sorry

end NUMINAMATH_CALUDE_elephant_received_503_pills_l2429_242943


namespace NUMINAMATH_CALUDE_quadratic_equations_solutions_l2429_242914

theorem quadratic_equations_solutions :
  (∃ x₁ x₂ : ℝ, x₁ = 2 + Real.sqrt 5 ∧ x₂ = 2 - Real.sqrt 5 ∧ x₁^2 - 4*x₁ - 1 = 0 ∧ x₂^2 - 4*x₂ - 1 = 0) ∧
  (∃ x₁ x₂ : ℝ, x₁ = -1/3 ∧ x₂ = 2 ∧ x₁*(3*x₁ + 1) = 2*(3*x₁ + 1) ∧ x₂*(3*x₂ + 1) = 2*(3*x₂ + 1)) ∧
  (∃ x₁ x₂ : ℝ, x₁ = (-1 + Real.sqrt 33) / 4 ∧ x₂ = (-1 - Real.sqrt 33) / 4 ∧ 2*x₁^2 + x₁ - 4 = 0 ∧ 2*x₂^2 + x₂ - 4 = 0) ∧
  (∀ x : ℝ, 4*x^2 - 3*x + 1 ≠ 0) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_equations_solutions_l2429_242914


namespace NUMINAMATH_CALUDE_lines_parallel_iff_a_eq_zero_l2429_242959

/-- Two lines in the form of x-2ay=1 and 2x-2ay=1 are parallel if and only if a=0 -/
theorem lines_parallel_iff_a_eq_zero (a : ℝ) :
  (∀ x y : ℝ, x - 2*a*y = 1 ↔ 2*x - 2*a*y = 1) ↔ a = 0 := by
  sorry

end NUMINAMATH_CALUDE_lines_parallel_iff_a_eq_zero_l2429_242959


namespace NUMINAMATH_CALUDE_keys_for_52_phones_l2429_242977

/-- Represents the warehouse setup and the task of retrieving phones -/
structure WarehouseSetup where
  total_cabinets : ℕ
  boxes_per_cabinet : ℕ
  phones_per_box : ℕ
  phones_to_retrieve : ℕ

/-- Calculates the minimum number of keys required to retrieve the specified number of phones -/
def min_keys_required (setup : WarehouseSetup) : ℕ :=
  let boxes_needed := (setup.phones_to_retrieve + setup.phones_per_box - 1) / setup.phones_per_box
  let cabinets_needed := (boxes_needed + setup.boxes_per_cabinet - 1) / setup.boxes_per_cabinet
  boxes_needed + cabinets_needed + 1

/-- The theorem stating that for the given setup, 9 keys are required -/
theorem keys_for_52_phones :
  let setup : WarehouseSetup := {
    total_cabinets := 8,
    boxes_per_cabinet := 4,
    phones_per_box := 10,
    phones_to_retrieve := 52
  }
  min_keys_required setup = 9 := by
  sorry

end NUMINAMATH_CALUDE_keys_for_52_phones_l2429_242977


namespace NUMINAMATH_CALUDE_symmetric_line_equation_l2429_242905

/-- Given a line y = (1/2)x and a line of symmetry x = 1, 
    the symmetric line has the equation x + 2y - 2 = 0 -/
theorem symmetric_line_equation : 
  ∀ (x y : ℝ), 
  (y = (1/2) * x) →  -- Original line
  (∃ (x' y' : ℝ), 
    (x' = 1) ∧  -- Line of symmetry
    (y' = y) ∧ 
    (x - 1 = 1 - x')) →  -- Symmetry condition
  (x + 2*y - 2 = 0)  -- Equation of symmetric line
:= by sorry

end NUMINAMATH_CALUDE_symmetric_line_equation_l2429_242905


namespace NUMINAMATH_CALUDE_square_sum_inequality_l2429_242996

theorem square_sum_inequality (a b : ℝ) : a^2 + b^2 - 1 - a^2*b^2 ≥ 0 := by
  sorry

end NUMINAMATH_CALUDE_square_sum_inequality_l2429_242996


namespace NUMINAMATH_CALUDE_product_evaluation_l2429_242971

theorem product_evaluation (a b c d : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (hd : d > 0) :
  (a + b + c + d)⁻¹ * (a⁻¹ + b⁻¹ + c⁻¹ + d⁻¹) * (a*b + b*c + c*d + d*a + a*c + b*d)⁻¹ *
  ((a*b)⁻¹ + (b*c)⁻¹ + (c*d)⁻¹ + (d*a)⁻¹ + (a*c)⁻¹ + (b*d)⁻¹) = (a*a*b*b*c*c*d*d)⁻¹ :=
by sorry

end NUMINAMATH_CALUDE_product_evaluation_l2429_242971


namespace NUMINAMATH_CALUDE_mod_equivalence_problem_l2429_242944

theorem mod_equivalence_problem : ∃ n : ℤ, 0 ≤ n ∧ n < 21 ∧ -200 ≡ n [ZMOD 21] ∧ n = 10 := by
  sorry

end NUMINAMATH_CALUDE_mod_equivalence_problem_l2429_242944


namespace NUMINAMATH_CALUDE_real_roots_iff_k_nonzero_l2429_242975

theorem real_roots_iff_k_nonzero (K : ℝ) :
  (∃ x : ℝ, x = K^2 * (x - 1) * (x - 2) * (x - 3)) ↔ K ≠ 0 := by
  sorry

end NUMINAMATH_CALUDE_real_roots_iff_k_nonzero_l2429_242975


namespace NUMINAMATH_CALUDE_problem_statement_l2429_242955

theorem problem_statement (y : ℝ) (hy : y > 0) : 
  ∃ y, ((3/5 * 2500) * (2/7 * ((5/8 * 4000) + (1/4 * 3600) - ((11/20 * 7200) / (3/10 * y))))) = 25000 :=
by sorry

end NUMINAMATH_CALUDE_problem_statement_l2429_242955


namespace NUMINAMATH_CALUDE_total_pictures_correct_l2429_242939

/-- The number of pictures Bianca uploaded to Facebook -/
def total_pictures : ℕ := 33

/-- The number of pictures in the first album -/
def first_album_pictures : ℕ := 27

/-- The number of additional albums -/
def additional_albums : ℕ := 3

/-- The number of pictures in each additional album -/
def pictures_per_additional_album : ℕ := 2

/-- Theorem stating that the total number of pictures is correct -/
theorem total_pictures_correct : 
  total_pictures = first_album_pictures + additional_albums * pictures_per_additional_album := by
  sorry

end NUMINAMATH_CALUDE_total_pictures_correct_l2429_242939


namespace NUMINAMATH_CALUDE_arithmetic_sequence_a10_l2429_242918

/-- An arithmetic sequence with specific properties -/
structure ArithmeticSequence where
  a : ℕ → ℚ
  is_arithmetic : ∀ n : ℕ, a (n + 1) - a n = a 1 - a 0
  a3_eq_2 : a 3 = 2
  a5_plus_a8_eq_15 : a 5 + a 8 = 15

/-- The 10th term of the arithmetic sequence is 13 -/
theorem arithmetic_sequence_a10 (seq : ArithmeticSequence) : seq.a 10 = 13 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_a10_l2429_242918


namespace NUMINAMATH_CALUDE_sum_of_100th_terms_l2429_242970

/-- Given two arithmetic sequences {a_n} and {b_n} satisfying certain conditions,
    prove that the sum of their 100th terms is 383. -/
theorem sum_of_100th_terms (a b : ℕ → ℝ) : 
  (∀ n m : ℕ, a (n + 1) - a n = a (m + 1) - a m) →  -- a_n is arithmetic
  (∀ n m : ℕ, b (n + 1) - b n = b (m + 1) - b m) →  -- b_n is arithmetic
  a 5 + b 5 = 3 →
  a 9 + b 9 = 19 →
  a 100 + b 100 = 383 := by
sorry

end NUMINAMATH_CALUDE_sum_of_100th_terms_l2429_242970


namespace NUMINAMATH_CALUDE_exists_counterexample_to_inequality_l2429_242949

theorem exists_counterexample_to_inequality (a b c : ℝ) 
  (h1 : c < b) (h2 : b < a) (h3 : a * c < 0) :
  ∃ (a b c : ℝ), c < b ∧ b < a ∧ a * c < 0 ∧ c * b^2 ≥ a * b^2 :=
sorry

end NUMINAMATH_CALUDE_exists_counterexample_to_inequality_l2429_242949


namespace NUMINAMATH_CALUDE_puppies_adopted_per_day_l2429_242913

theorem puppies_adopted_per_day 
  (initial_puppies : ℕ) 
  (additional_puppies : ℕ) 
  (adoption_days : ℕ) 
  (h1 : initial_puppies = 2)
  (h2 : additional_puppies = 34)
  (h3 : adoption_days = 9)
  (h4 : (initial_puppies + additional_puppies) % adoption_days = 0) :
  (initial_puppies + additional_puppies) / adoption_days = 4 := by
sorry

end NUMINAMATH_CALUDE_puppies_adopted_per_day_l2429_242913


namespace NUMINAMATH_CALUDE_triangle_inequality_with_median_l2429_242920

/-- 
For any triangle with side lengths a, b, and c, and median length m_a 
from vertex A to the midpoint of side BC, the inequality a^2 + 4m_a^2 ≤ (b+c)^2 holds.
-/
theorem triangle_inequality_with_median 
  (a b c m_a : ℝ) 
  (h_positive : 0 < a ∧ 0 < b ∧ 0 < c ∧ 0 < m_a) 
  (h_triangle : a < b + c ∧ b < a + c ∧ c < a + b) 
  (h_median : m_a > 0) : 
  a^2 + 4 * m_a^2 ≤ (b + c)^2 := by
  sorry

end NUMINAMATH_CALUDE_triangle_inequality_with_median_l2429_242920


namespace NUMINAMATH_CALUDE_emily_widget_difference_l2429_242935

-- Define the variables
variable (t : ℝ)
variable (w : ℝ)

-- Define the conditions
def monday_production := w * t
def tuesday_production := (w + 6) * (t - 3)

-- Define the relationship between w and t
axiom w_eq_2t : w = 2 * t

-- State the theorem
theorem emily_widget_difference :
  monday_production - tuesday_production = 18 := by
  sorry

end NUMINAMATH_CALUDE_emily_widget_difference_l2429_242935


namespace NUMINAMATH_CALUDE_coefficient_x6_is_180_l2429_242983

/-- The coefficient of x^6 in the binomial expansion of (x - 2/x)^10 -/
def coefficient_x6 : ℤ := 
  let n : ℕ := 10
  let k : ℕ := (n - 6) / 2
  (n.choose k) * (-2)^k

theorem coefficient_x6_is_180 : coefficient_x6 = 180 := by
  sorry

end NUMINAMATH_CALUDE_coefficient_x6_is_180_l2429_242983


namespace NUMINAMATH_CALUDE_solve_for_a_l2429_242965

theorem solve_for_a (x a : ℝ) (h1 : 2 * x - 5 * a = 3 * a + 22) (h2 : x = 3) : a = -2 := by
  sorry

end NUMINAMATH_CALUDE_solve_for_a_l2429_242965


namespace NUMINAMATH_CALUDE_even_sum_from_even_expression_l2429_242907

theorem even_sum_from_even_expression (n m : ℤ) : 
  Even (n^2 + m^2 + n*m) → Even (n + m) := by
  sorry

end NUMINAMATH_CALUDE_even_sum_from_even_expression_l2429_242907


namespace NUMINAMATH_CALUDE_hotel_has_21_rooms_l2429_242961

/-- Represents the inventory and room requirements for a hotel. -/
structure HotelInventory where
  total_lamps : ℕ
  total_chairs : ℕ
  total_bed_sheets : ℕ
  lamps_per_room : ℕ
  chairs_per_room : ℕ
  bed_sheets_per_room : ℕ

/-- Calculates the number of rooms in a hotel based on its inventory and room requirements. -/
def calculateRooms (inventory : HotelInventory) : ℕ :=
  min (inventory.total_lamps / inventory.lamps_per_room)
    (min (inventory.total_chairs / inventory.chairs_per_room)
      (inventory.total_bed_sheets / inventory.bed_sheets_per_room))

/-- Theorem stating that the hotel has 21 rooms based on the given inventory. -/
theorem hotel_has_21_rooms (inventory : HotelInventory)
    (h1 : inventory.total_lamps = 147)
    (h2 : inventory.total_chairs = 84)
    (h3 : inventory.total_bed_sheets = 210)
    (h4 : inventory.lamps_per_room = 7)
    (h5 : inventory.chairs_per_room = 4)
    (h6 : inventory.bed_sheets_per_room = 10) :
    calculateRooms inventory = 21 := by
  sorry

end NUMINAMATH_CALUDE_hotel_has_21_rooms_l2429_242961


namespace NUMINAMATH_CALUDE_polynomial_intersection_l2429_242982

-- Define the polynomials f and g
def f (a b x : ℝ) : ℝ := x^2 + a*x + b
def g (c d x : ℝ) : ℝ := x^2 + c*x + d

-- State the theorem
theorem polynomial_intersection (a b c d : ℝ) : 
  -- f and g are distinct
  f a b ≠ g c d →
  -- The x-coordinate of the vertex of f is a root of g
  g c d (-a/2) = 0 →
  -- The x-coordinate of the vertex of g is a root of f
  f a b (-c/2) = 0 →
  -- Both f and g have the same minimum value
  (∃ (k : ℝ), ∀ (x : ℝ), f a b x ≥ k ∧ g c d x ≥ k) →
  -- The graphs of f and g intersect at (200, -200)
  f a b 200 = -200 ∧ g c d 200 = -200 →
  -- Conclusion: a + c = -800
  a + c = -800 := by
sorry

end NUMINAMATH_CALUDE_polynomial_intersection_l2429_242982


namespace NUMINAMATH_CALUDE_binomial_probability_problem_l2429_242919

/-- A random variable following a binomial distribution -/
structure BinomialRV where
  n : ℕ
  p : ℝ
  h_p : 0 ≤ p ∧ p ≤ 1

/-- Probability of a binomial random variable being greater than or equal to k -/
def prob_ge (X : BinomialRV) (k : ℕ) : ℝ :=
  sorry

theorem binomial_probability_problem (X Y : BinomialRV) 
  (hX : X.n = 2) (hY : Y.n = 4) (hp : X.p = Y.p)
  (h_prob : prob_ge X 1 = 5/9) : 
  prob_ge Y 2 = 11/27 := by
  sorry

end NUMINAMATH_CALUDE_binomial_probability_problem_l2429_242919


namespace NUMINAMATH_CALUDE_equation_solutions_l2429_242946

theorem equation_solutions : 
  let f (x : ℝ) := (x - 1) * (x - 3) * (x - 5) * (x - 7) * (x - 5) * (x - 3) * (x - 1)
  let g (x : ℝ) := (x - 3) * (x - 5) * (x - 3)
  ∀ x : ℝ, (x ≠ 3 ∧ x ≠ 5) → 
    (f x / g x = 1 ↔ x = 4 ∨ x = 4 + 2 * Real.sqrt 10 ∨ x = 4 - 2 * Real.sqrt 10) :=
by sorry

end NUMINAMATH_CALUDE_equation_solutions_l2429_242946


namespace NUMINAMATH_CALUDE_triangle_inequality_l2429_242921

theorem triangle_inequality (a b c : ℝ) 
  (h_triangle : a > 0 ∧ b > 0 ∧ c > 0 ∧ a + b > c ∧ b + c > a ∧ c + a > b) 
  (h_sum : a + b + c = 2) : 
  a^2 + b^2 + c^2 + 2*a*b*c < 2 := by
sorry

end NUMINAMATH_CALUDE_triangle_inequality_l2429_242921


namespace NUMINAMATH_CALUDE_lee_science_class_l2429_242986

theorem lee_science_class (total : ℕ) (girls_ratio : ℕ) (boys_ratio : ℕ) : 
  total = 56 → girls_ratio = 4 → boys_ratio = 3 → 
  (girls_ratio + boys_ratio) * (total / (girls_ratio + boys_ratio)) * boys_ratio / girls_ratio = 24 := by
sorry

end NUMINAMATH_CALUDE_lee_science_class_l2429_242986


namespace NUMINAMATH_CALUDE_train_bridge_crossing_time_l2429_242993

theorem train_bridge_crossing_time (train_length bridge_length : ℝ) (train_speed_kmph : ℝ) :
  train_length = 130 →
  bridge_length = 150 →
  train_speed_kmph = 36 →
  (train_length + bridge_length) / (train_speed_kmph * 1000 / 3600) = 28 := by
  sorry

end NUMINAMATH_CALUDE_train_bridge_crossing_time_l2429_242993


namespace NUMINAMATH_CALUDE_octal_minus_base9_equals_152294_l2429_242925

def base_to_decimal (digits : List Nat) (base : Nat) : Nat :=
  digits.foldr (fun d acc => d + base * acc) 0

theorem octal_minus_base9_equals_152294 :
  let octal_num := [5, 4, 3, 2, 1, 0]
  let base9_num := [4, 3, 2, 1, 0]
  base_to_decimal octal_num 8 - base_to_decimal base9_num 9 = 152294 := by
  sorry

end NUMINAMATH_CALUDE_octal_minus_base9_equals_152294_l2429_242925


namespace NUMINAMATH_CALUDE_sum_inequality_l2429_242912

theorem sum_inequality (a b c : ℝ) (k : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) 
  (habc : a * b * c = 1) (hk : k ≥ 3) : 
  (1 / (a^k * (b + c)) + 1 / (b^k * (a + c)) + 1 / (c^k * (a + b))) ≥ 3/2 := by
  sorry

end NUMINAMATH_CALUDE_sum_inequality_l2429_242912


namespace NUMINAMATH_CALUDE_students_passed_at_least_one_subject_l2429_242988

theorem students_passed_at_least_one_subject 
  (failed_hindi : ℝ) 
  (failed_english : ℝ) 
  (failed_both : ℝ) 
  (h1 : failed_hindi = 32) 
  (h2 : failed_english = 56) 
  (h3 : failed_both = 12) : 
  100 - (failed_hindi + failed_english - failed_both) = 24 := by
  sorry

end NUMINAMATH_CALUDE_students_passed_at_least_one_subject_l2429_242988


namespace NUMINAMATH_CALUDE_production_volume_equation_l2429_242987

theorem production_volume_equation (x : ℝ) : 
  (200 : ℝ) + 200 * (1 + x) + 200 * (1 + x)^2 = 1400 ↔ 
  (∃ y : ℝ, y > 0 ∧ 
    200 * (1 + y + (1 + y)^2) = 1400 ∧
    (∀ n : ℕ, n ≥ 1 ∧ n ≤ 3 → 
      (200 * (1 + y)^(n - 1) = 200 * (1 + x)^(n - 1)))) :=
by sorry

end NUMINAMATH_CALUDE_production_volume_equation_l2429_242987


namespace NUMINAMATH_CALUDE_special_sequence_1000th_term_l2429_242980

/-- A sequence satisfying the given conditions -/
def SpecialSequence (a : ℕ → ℕ) : Prop :=
  a 1 = 2007 ∧ 
  a 2 = 2008 ∧ 
  ∀ n : ℕ, n ≥ 1 → a n + a (n + 1) + a (n + 2) = n

/-- The 1000th term of the special sequence is 2340 -/
theorem special_sequence_1000th_term (a : ℕ → ℕ) (h : SpecialSequence a) : 
  a 1000 = 2340 := by
  sorry

end NUMINAMATH_CALUDE_special_sequence_1000th_term_l2429_242980


namespace NUMINAMATH_CALUDE_negation_of_exponential_proposition_l2429_242995

theorem negation_of_exponential_proposition :
  (¬ ∀ x : ℝ, x > 0 → Real.exp x ≥ 1) ↔ (∃ x : ℝ, x > 0 ∧ Real.exp x < 1) := by
  sorry

end NUMINAMATH_CALUDE_negation_of_exponential_proposition_l2429_242995


namespace NUMINAMATH_CALUDE_abs_2x_minus_5_l2429_242929

theorem abs_2x_minus_5 (x : ℝ) (h : abs (2*x - 3) - 3 + 2*x = 0) : abs (2*x - 5) = 5 - 2*x := by
  sorry

end NUMINAMATH_CALUDE_abs_2x_minus_5_l2429_242929


namespace NUMINAMATH_CALUDE_pistachio_price_per_can_l2429_242932

/-- The price of a can of pistachios given James' consumption habits and weekly spending -/
theorem pistachio_price_per_can 
  (can_size : ℝ) 
  (consumption_per_5_days : ℝ) 
  (weekly_spending : ℝ) 
  (h1 : can_size = 5) 
  (h2 : consumption_per_5_days = 30) 
  (h3 : weekly_spending = 84) : 
  weekly_spending / ((7 / 5) * consumption_per_5_days / can_size) = 10 := by
sorry

end NUMINAMATH_CALUDE_pistachio_price_per_can_l2429_242932


namespace NUMINAMATH_CALUDE_book_sale_loss_percentage_l2429_242999

/-- Given two books with specified costs and selling conditions, prove the loss percentage on the first book. -/
theorem book_sale_loss_percentage
  (total_cost : ℝ)
  (cost_book1 : ℝ)
  (gain_percentage : ℝ)
  (h_total_cost : total_cost = 480)
  (h_cost_book1 : cost_book1 = 280)
  (h_gain_percentage : gain_percentage = 19)
  (h_same_selling_price : ∃ (selling_price : ℝ),
    selling_price = cost_book1 * (1 - (loss_percentage / 100)) ∧
    selling_price = (total_cost - cost_book1) * (1 + (gain_percentage / 100)))
  : ∃ (loss_percentage : ℝ), loss_percentage = 15 := by
  sorry


end NUMINAMATH_CALUDE_book_sale_loss_percentage_l2429_242999


namespace NUMINAMATH_CALUDE_cake_frosting_time_difference_l2429_242962

/-- The time difference in frosting cakes with normal and sprained conditions -/
theorem cake_frosting_time_difference 
  (normal_time : ℕ) -- Time to frost one cake under normal conditions
  (sprained_time : ℕ) -- Time to frost one cake with sprained wrist
  (num_cakes : ℕ) -- Number of cakes to frost
  (h1 : normal_time = 5) -- Normal frosting time is 5 minutes
  (h2 : sprained_time = 8) -- Sprained wrist frosting time is 8 minutes
  (h3 : num_cakes = 10) -- Number of cakes to frost is 10
  : sprained_time * num_cakes - normal_time * num_cakes = 30 := by
  sorry

end NUMINAMATH_CALUDE_cake_frosting_time_difference_l2429_242962


namespace NUMINAMATH_CALUDE_min_value_quadratic_function_l2429_242930

theorem min_value_quadratic_function (a c : ℝ) (h1 : a > 0) (h2 : c > 0) :
  (∀ x, ax^2 - 2*x + c ≥ 0) →
  (∃ x, ax^2 - 2*x + c = 0) →
  (∀ y : ℝ, ∃ x, ax^2 - 2*x + c = y → y ≥ 0) →
  (9/a + 1/c ≥ 6) ∧ (∃ a' c', 9/a' + 1/c' = 6 ∧ a' > 0 ∧ c' > 0 ∧
    (∀ x, a'*x^2 - 2*x + c' ≥ 0) ∧
    (∃ x, a'*x^2 - 2*x + c' = 0) ∧
    (∀ y : ℝ, ∃ x, a'*x^2 - 2*x + c' = y → y ≥ 0)) :=
by sorry

end NUMINAMATH_CALUDE_min_value_quadratic_function_l2429_242930


namespace NUMINAMATH_CALUDE_quadrilateral_second_offset_l2429_242945

/-- Given a quadrilateral with one diagonal of 50 cm, one offset of 10 cm, and an area of 450 cm^2,
    prove that the length of the second offset is 8 cm. -/
theorem quadrilateral_second_offset (diagonal : ℝ) (offset1 : ℝ) (area : ℝ) (offset2 : ℝ) :
  diagonal = 50 → offset1 = 10 → area = 450 →
  area = 1/2 * diagonal * (offset1 + offset2) →
  offset2 = 8 := by sorry

end NUMINAMATH_CALUDE_quadrilateral_second_offset_l2429_242945

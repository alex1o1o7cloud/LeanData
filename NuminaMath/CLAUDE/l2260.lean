import Mathlib

namespace NUMINAMATH_CALUDE_unwashed_shirts_l2260_226023

theorem unwashed_shirts (short_sleeve : ℕ) (long_sleeve : ℕ) (washed : ℕ) : 
  short_sleeve = 9 → long_sleeve = 21 → washed = 29 → 
  short_sleeve + long_sleeve - washed = 1 := by
sorry

end NUMINAMATH_CALUDE_unwashed_shirts_l2260_226023


namespace NUMINAMATH_CALUDE_probability_red_black_heart_value_l2260_226024

/-- The probability of drawing a red card first, then a black card, and then a red heart
    from a deck of 104 cards with 52 red cards (of which 26 are hearts) and 52 black cards. -/
def probability_red_black_heart (total_cards : ℕ) (red_cards : ℕ) (black_cards : ℕ) (heart_cards : ℕ) : ℚ :=
  (red_cards : ℚ) / total_cards *
  (black_cards : ℚ) / (total_cards - 1) *
  (heart_cards - 1 : ℚ) / (total_cards - 2)

/-- The probability of drawing a red card first, then a black card, and then a red heart
    from a deck of 104 cards with 52 red cards (of which 26 are hearts) and 52 black cards
    is equal to 25/3978. -/
theorem probability_red_black_heart_value :
  probability_red_black_heart 104 52 52 26 = 25 / 3978 :=
by
  sorry

#eval probability_red_black_heart 104 52 52 26

end NUMINAMATH_CALUDE_probability_red_black_heart_value_l2260_226024


namespace NUMINAMATH_CALUDE_subtracted_number_proof_l2260_226030

theorem subtracted_number_proof (initial_number : ℝ) (subtracted_number : ℝ) : 
  initial_number = 22.142857142857142 →
  ((initial_number + 5) * 7) / 5 - subtracted_number = 33 →
  subtracted_number = 5 := by
sorry

end NUMINAMATH_CALUDE_subtracted_number_proof_l2260_226030


namespace NUMINAMATH_CALUDE_salary_ratio_degree_to_diploma_l2260_226016

/-- Represents the monthly salary of a diploma holder in dollars. -/
def diploma_monthly_salary : ℕ := 4000

/-- Represents the annual salary of a degree holder in dollars. -/
def degree_annual_salary : ℕ := 144000

/-- Represents the number of months in a year. -/
def months_per_year : ℕ := 12

/-- Theorem stating that the ratio of annual salaries between degree and diploma holders is 3:1. -/
theorem salary_ratio_degree_to_diploma :
  (degree_annual_salary : ℚ) / (diploma_monthly_salary * months_per_year) = 3 := by
  sorry

#check salary_ratio_degree_to_diploma

end NUMINAMATH_CALUDE_salary_ratio_degree_to_diploma_l2260_226016


namespace NUMINAMATH_CALUDE_graph_connectivity_probability_l2260_226097

def num_vertices : Nat := 20
def num_edges_removed : Nat := 35

theorem graph_connectivity_probability :
  let total_edges := num_vertices * (num_vertices - 1) / 2
  let remaining_edges := total_edges - num_edges_removed
  let prob_connected := 1 - (num_vertices * Nat.choose remaining_edges (remaining_edges - num_vertices + 1)) / Nat.choose total_edges num_edges_removed
  prob_connected = 1 - (20 * Nat.choose 171 16) / Nat.choose 190 35 := by
  sorry

end NUMINAMATH_CALUDE_graph_connectivity_probability_l2260_226097


namespace NUMINAMATH_CALUDE_yard_sale_problem_l2260_226036

theorem yard_sale_problem (total_items video_games dvds books working_video_games working_dvds : ℕ) 
  (h1 : total_items = 56)
  (h2 : video_games = 30)
  (h3 : dvds = 15)
  (h4 : books = total_items - video_games - dvds)
  (h5 : working_video_games = 20)
  (h6 : working_dvds = 10) :
  (video_games - working_video_games) + (dvds - working_dvds) = 15 := by
  sorry

end NUMINAMATH_CALUDE_yard_sale_problem_l2260_226036


namespace NUMINAMATH_CALUDE_min_cars_with_racing_stripes_l2260_226051

theorem min_cars_with_racing_stripes 
  (total_cars : ℕ) 
  (cars_without_ac : ℕ) 
  (max_ac_no_stripes : ℕ) 
  (h1 : total_cars = 100)
  (h2 : cars_without_ac = 47)
  (h3 : max_ac_no_stripes = 47)
  (h4 : cars_without_ac ≤ total_cars)
  (h5 : max_ac_no_stripes ≤ total_cars - cars_without_ac) :
  ∃ (min_cars_with_stripes : ℕ), 
    min_cars_with_stripes = total_cars - cars_without_ac - max_ac_no_stripes ∧ 
    min_cars_with_stripes = 6 :=
by
  sorry

#check min_cars_with_racing_stripes

end NUMINAMATH_CALUDE_min_cars_with_racing_stripes_l2260_226051


namespace NUMINAMATH_CALUDE_sum_of_squares_l2260_226077

theorem sum_of_squares (x y : ℝ) (h1 : x * y = 120) (h2 : x + y = 23) : x^2 + y^2 = 289 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_squares_l2260_226077


namespace NUMINAMATH_CALUDE_tenth_term_is_three_point_five_l2260_226038

/-- The nth term of an arithmetic sequence -/
def arithmetic_sequence (a₁ : ℚ) (d : ℚ) (n : ℕ) : ℚ := a₁ + (n - 1 : ℚ) * d

/-- The first term of our sequence -/
def a₁ : ℚ := 1/2

/-- The second term of our sequence -/
def a₂ : ℚ := 5/6

/-- The common difference of our sequence -/
def d : ℚ := a₂ - a₁

theorem tenth_term_is_three_point_five :
  arithmetic_sequence a₁ d 10 = 7/2 := by sorry

end NUMINAMATH_CALUDE_tenth_term_is_three_point_five_l2260_226038


namespace NUMINAMATH_CALUDE_one_percent_of_x_l2260_226061

theorem one_percent_of_x (x : ℝ) (h : (89 / 100) * 19 = (19 / 100) * x) : 
  (1 / 100) * x = 89 / 100 := by
sorry

end NUMINAMATH_CALUDE_one_percent_of_x_l2260_226061


namespace NUMINAMATH_CALUDE_swimmers_second_meeting_time_l2260_226001

theorem swimmers_second_meeting_time
  (pool_length : ℝ)
  (henry_speed : ℝ)
  (george_speed : ℝ)
  (first_meeting_time : ℝ)
  (h1 : pool_length = 100)
  (h2 : george_speed = 2 * henry_speed)
  (h3 : first_meeting_time = 1)
  (h4 : henry_speed * first_meeting_time + george_speed * first_meeting_time = pool_length) :
  let second_meeting_time := 2 * first_meeting_time
  ∃ (distance_henry distance_george : ℝ),
    distance_henry + distance_george = pool_length ∧
    distance_henry = henry_speed * second_meeting_time ∧
    distance_george = george_speed * second_meeting_time :=
by sorry


end NUMINAMATH_CALUDE_swimmers_second_meeting_time_l2260_226001


namespace NUMINAMATH_CALUDE_complex_cube_sum_magnitude_l2260_226095

theorem complex_cube_sum_magnitude (w z : ℂ) 
  (h1 : Complex.abs (w + z) = 3)
  (h2 : Complex.abs (w^2 + z^2) = 18) :
  Complex.abs (w^3 + z^3) = 81/2 := by
  sorry

end NUMINAMATH_CALUDE_complex_cube_sum_magnitude_l2260_226095


namespace NUMINAMATH_CALUDE_charging_piles_growth_equation_l2260_226055

/-- Given the number of charging piles built in the first and third months, 
    and the monthly average growth rate, this theorem states the equation 
    that relates these quantities. -/
theorem charging_piles_growth_equation 
  (initial_piles : ℕ) 
  (final_piles : ℕ) 
  (x : ℝ) 
  (h1 : initial_piles = 301)
  (h2 : final_piles = 500)
  (h3 : x ≥ 0) -- Assuming non-negative growth rate
  (h4 : x ≤ 1) -- Assuming growth rate is at most 100%
  : initial_piles * (1 + x)^2 = final_piles := by
  sorry

end NUMINAMATH_CALUDE_charging_piles_growth_equation_l2260_226055


namespace NUMINAMATH_CALUDE_stamp_arrangement_exists_l2260_226083

/-- Represents the quantity of each stamp denomination -/
def stamp_quantities : List Nat := [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

/-- Represents the value of each stamp denomination -/
def stamp_values : List Nat := [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

/-- A function to calculate the number of unique stamp arrangements -/
def count_stamp_arrangements (quantities : List Nat) (values : List Nat) (target : Nat) : Nat :=
  sorry

/-- Theorem stating that there exists a positive number of unique arrangements -/
theorem stamp_arrangement_exists :
  ∃ n : Nat, n > 0 ∧ count_stamp_arrangements stamp_quantities stamp_values 15 = n :=
sorry

end NUMINAMATH_CALUDE_stamp_arrangement_exists_l2260_226083


namespace NUMINAMATH_CALUDE_lipstick_cost_l2260_226002

/-- Calculates the cost of each lipstick given the order details -/
theorem lipstick_cost (total_items : ℕ) (num_slippers : ℕ) (slipper_price : ℚ)
  (num_lipsticks : ℕ) (num_hair_colors : ℕ) (hair_color_price : ℚ) (total_paid : ℚ) :
  total_items = num_slippers + num_lipsticks + num_hair_colors →
  total_items = 18 →
  num_slippers = 6 →
  slipper_price = 5/2 →
  num_lipsticks = 4 →
  num_hair_colors = 8 →
  hair_color_price = 3 →
  total_paid = 44 →
  (total_paid - (num_slippers * slipper_price + num_hair_colors * hair_color_price)) / num_lipsticks = 5/4 :=
by sorry

end NUMINAMATH_CALUDE_lipstick_cost_l2260_226002


namespace NUMINAMATH_CALUDE_inequality_solution_l2260_226041

theorem inequality_solution : 
  {x : ℕ | x > 0 ∧ 3 * x - 4 < 2 * x} = {1, 2, 3} := by sorry

end NUMINAMATH_CALUDE_inequality_solution_l2260_226041


namespace NUMINAMATH_CALUDE_money_bounds_l2260_226098

theorem money_bounds (a b : ℝ) 
  (h1 : 4 * a + b < 60) 
  (h2 : 6 * a - b = 30) : 
  a < 9 ∧ b < 24 := by
  sorry

end NUMINAMATH_CALUDE_money_bounds_l2260_226098


namespace NUMINAMATH_CALUDE_remainder_problem_l2260_226090

theorem remainder_problem : 29 * 169^1990 ≡ 7 [MOD 11] := by
  sorry

end NUMINAMATH_CALUDE_remainder_problem_l2260_226090


namespace NUMINAMATH_CALUDE_standard_deviation_is_eight_l2260_226029

/-- Represents the age distribution of job applicants -/
structure AgeDistribution where
  average_age : ℕ
  num_different_ages : ℕ
  standard_deviation : ℕ

/-- Checks if the age distribution satisfies the given conditions -/
def is_valid_distribution (d : AgeDistribution) : Prop :=
  d.average_age = 30 ∧
  d.num_different_ages = 17 ∧
  d.num_different_ages = 2 * d.standard_deviation + 1

/-- Theorem stating that the standard deviation must be 8 given the conditions -/
theorem standard_deviation_is_eight (d : AgeDistribution) 
  (h : is_valid_distribution d) : d.standard_deviation = 8 := by
  sorry

#check standard_deviation_is_eight

end NUMINAMATH_CALUDE_standard_deviation_is_eight_l2260_226029


namespace NUMINAMATH_CALUDE_imaginary_part_of_complex_fraction_l2260_226068

theorem imaginary_part_of_complex_fraction (i : ℂ) (h : i * i = -1) :
  Complex.im ((1 - i) / ((1 + i)^2)) = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_imaginary_part_of_complex_fraction_l2260_226068


namespace NUMINAMATH_CALUDE_complex_equality_condition_l2260_226086

theorem complex_equality_condition :
  ∃ (x y : ℂ), x + y * Complex.I = 1 + Complex.I ∧ (x ≠ 1 ∨ y ≠ 1) :=
by sorry

end NUMINAMATH_CALUDE_complex_equality_condition_l2260_226086


namespace NUMINAMATH_CALUDE_cake_ratio_theorem_l2260_226091

/-- Proves that the ratio of cakes sold to total cakes baked is 1:2 --/
theorem cake_ratio_theorem (cakes_per_day : ℕ) (days_baked : ℕ) (cakes_left : ℕ) :
  cakes_per_day = 20 →
  days_baked = 9 →
  cakes_left = 90 →
  let total_cakes := cakes_per_day * days_baked
  let cakes_sold := total_cakes - cakes_left
  (cakes_sold : ℚ) / total_cakes = 1 / 2 :=
by
  sorry

#check cake_ratio_theorem

end NUMINAMATH_CALUDE_cake_ratio_theorem_l2260_226091


namespace NUMINAMATH_CALUDE_subtraction_example_l2260_226079

theorem subtraction_example : (3.75 : ℝ) - 1.46 = 2.29 := by sorry

end NUMINAMATH_CALUDE_subtraction_example_l2260_226079


namespace NUMINAMATH_CALUDE_sin_780_degrees_l2260_226044

theorem sin_780_degrees : Real.sin (780 * π / 180) = Real.sqrt 3 / 2 := by
  sorry

end NUMINAMATH_CALUDE_sin_780_degrees_l2260_226044


namespace NUMINAMATH_CALUDE_probability_not_hearing_favorite_in_6_minutes_l2260_226093

/-- Represents a playlist of songs with increasing durations -/
structure Playlist where
  num_songs : ℕ
  duration_increment : ℕ
  shortest_duration : ℕ
  favorite_duration : ℕ

/-- Calculates the probability of not hearing the entire favorite song 
    within a given time limit -/
def probability_not_hearing_favorite (p : Playlist) (time_limit : ℕ) : ℚ :=
  sorry

/-- The specific playlist described in the problem -/
def marcel_playlist : Playlist :=
  { num_songs := 12
  , duration_increment := 30
  , shortest_duration := 60
  , favorite_duration := 300 }

/-- The main theorem to prove -/
theorem probability_not_hearing_favorite_in_6_minutes :
  probability_not_hearing_favorite marcel_playlist 360 = 1813 / 1980 := by
  sorry

end NUMINAMATH_CALUDE_probability_not_hearing_favorite_in_6_minutes_l2260_226093


namespace NUMINAMATH_CALUDE_production_quantities_max_type_A_for_school_l2260_226063

-- Define the parameters
def total_production : ℕ := 400000
def cost_A : ℚ := 1.2
def cost_B : ℚ := 0.4
def price_A : ℚ := 1.6
def price_B : ℚ := 0.6
def profit : ℚ := 110000
def school_budget : ℚ := 7680
def discount_A : ℚ := 0.1
def school_purchase : ℕ := 10000

-- Part 1: Production quantities
theorem production_quantities :
  ∃ (x y : ℕ),
    x + y = total_production ∧
    (price_A - cost_A) * x + (price_B - cost_B) * y = profit ∧
    x = 15000 ∧
    y = 25000 :=
sorry

-- Part 2: Maximum type A books for school
theorem max_type_A_for_school :
  ∃ (m : ℕ),
    m ≤ school_purchase ∧
    price_A * (1 - discount_A) * m + price_B * (school_purchase - m) ≤ school_budget ∧
    m = 2000 ∧
    ∀ n, n > m → 
      price_A * (1 - discount_A) * n + price_B * (school_purchase - n) > school_budget :=
sorry

end NUMINAMATH_CALUDE_production_quantities_max_type_A_for_school_l2260_226063


namespace NUMINAMATH_CALUDE_simplify_expression_l2260_226048

theorem simplify_expression (x : ℝ) :
  (2*x - 1)^2 - (3*x + 1)*(3*x - 1) + 5*x*(x - 1) = -9*x + 2 := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l2260_226048


namespace NUMINAMATH_CALUDE_three_A_plus_six_B_m_value_when_independent_l2260_226009

-- Define A and B as functions of x and m
def A (x m : ℝ) : ℝ := 2*x^2 + 3*m*x - 2*x - 1
def B (x m : ℝ) : ℝ := -x^2 + m*x - 1

-- Theorem 1: 3A + 6B = (15m-6)x - 9
theorem three_A_plus_six_B (x m : ℝ) : 
  3 * A x m + 6 * B x m = (15*m - 6)*x - 9 := by sorry

-- Theorem 2: When 3A + 6B is independent of x, m = 2/5
theorem m_value_when_independent (m : ℝ) :
  (∀ x : ℝ, 3 * A x m + 6 * B x m = (15*m - 6)*x - 9) →
  (∀ x y : ℝ, 3 * A x m + 6 * B x m = 3 * A y m + 6 * B y m) →
  m = 2/5 := by sorry

end NUMINAMATH_CALUDE_three_A_plus_six_B_m_value_when_independent_l2260_226009


namespace NUMINAMATH_CALUDE_pens_given_to_sharon_proof_l2260_226013

def initial_pens : ℕ := 7
def mikes_pens : ℕ := 22
def final_pens : ℕ := 39

def pens_given_to_sharon : ℕ := 19

theorem pens_given_to_sharon_proof :
  ((initial_pens + mikes_pens) * 2) - final_pens = pens_given_to_sharon := by
  sorry

end NUMINAMATH_CALUDE_pens_given_to_sharon_proof_l2260_226013


namespace NUMINAMATH_CALUDE_fraction_simplification_l2260_226037

theorem fraction_simplification : 
  (1 / 2 + 1 / 5) / (3 / 7 - 1 / 14) = 49 / 25 := by
  sorry

end NUMINAMATH_CALUDE_fraction_simplification_l2260_226037


namespace NUMINAMATH_CALUDE_geometric_series_sum_l2260_226074

theorem geometric_series_sum (y : ℚ) : y = 23 / 13 ↔ 
  (∑' n, (1 / 3 : ℚ) ^ n) + (∑' n, (-1/4 : ℚ) ^ n) = ∑' n, (1 / y : ℚ) ^ n :=
by sorry

end NUMINAMATH_CALUDE_geometric_series_sum_l2260_226074


namespace NUMINAMATH_CALUDE_complex_number_imaginary_part_l2260_226069

theorem complex_number_imaginary_part (a : ℝ) : 
  let z : ℂ := (1 + a * Complex.I) / (1 - Complex.I)
  (z.im = 2) → a = 3 := by
  sorry

end NUMINAMATH_CALUDE_complex_number_imaginary_part_l2260_226069


namespace NUMINAMATH_CALUDE_unique_solution_to_system_l2260_226034

/-- The number of integer solutions to the system of equations:
    x^2 - 4xy + 3y^2 + z^2 = 45
    x^2 + 5yz - z^2 = -52
    -2x^2 + xy - 7z^2 = -101 -/
theorem unique_solution_to_system : 
  ∃! (x y z : ℤ), 
    x^2 - 4*x*y + 3*y^2 + z^2 = 45 ∧ 
    x^2 + 5*y*z - z^2 = -52 ∧ 
    -2*x^2 + x*y - 7*z^2 = -101 := by
  sorry

end NUMINAMATH_CALUDE_unique_solution_to_system_l2260_226034


namespace NUMINAMATH_CALUDE_battery_life_comparison_l2260_226007

-- Define the battery characteristics
def tablet_standby : ℚ := 18
def tablet_continuous : ℚ := 6
def smartphone_standby : ℚ := 30
def smartphone_continuous : ℚ := 4

-- Define the usage
def tablet_total_time : ℚ := 14
def tablet_usage_time : ℚ := 2
def smartphone_total_time : ℚ := 20
def smartphone_usage_time : ℚ := 3

-- Define the battery consumption rates
def tablet_standby_rate : ℚ := 1 / tablet_standby
def tablet_usage_rate : ℚ := 1 / tablet_continuous
def smartphone_standby_rate : ℚ := 1 / smartphone_standby
def smartphone_usage_rate : ℚ := 1 / smartphone_continuous

-- Define the theorem
theorem battery_life_comparison : 
  let tablet_battery_used := (tablet_total_time - tablet_usage_time) * tablet_standby_rate + tablet_usage_time * tablet_usage_rate
  let smartphone_battery_used := (smartphone_total_time - smartphone_usage_time) * smartphone_standby_rate + smartphone_usage_time * smartphone_usage_rate
  let smartphone_battery_remaining := 1 - smartphone_battery_used
  tablet_battery_used ≥ 1 ∧ 
  smartphone_battery_remaining / smartphone_standby_rate = 9 :=
by sorry

end NUMINAMATH_CALUDE_battery_life_comparison_l2260_226007


namespace NUMINAMATH_CALUDE_divisor_of_a_l2260_226060

theorem divisor_of_a (a b c d : ℕ+) 
  (h1 : Nat.gcd a b = 18)
  (h2 : Nat.gcd b c = 45)
  (h3 : Nat.gcd c d = 60)
  (h4 : 90 < Nat.gcd d a ∧ Nat.gcd d a < 120) :
  5 ∣ a := by
  sorry

end NUMINAMATH_CALUDE_divisor_of_a_l2260_226060


namespace NUMINAMATH_CALUDE_chord_intersects_diameter_l2260_226028

/-- In a circle with radius 6, a chord of length 10 intersects a diameter,
    dividing it into segments of lengths 6 - √11 and 6 + √11 -/
theorem chord_intersects_diameter (r : ℝ) (chord_length : ℝ) 
  (h1 : r = 6) (h2 : chord_length = 10) : 
  ∃ (s1 s2 : ℝ), s1 = 6 - Real.sqrt 11 ∧ s2 = 6 + Real.sqrt 11 ∧ s1 + s2 = 2 * r :=
sorry

end NUMINAMATH_CALUDE_chord_intersects_diameter_l2260_226028


namespace NUMINAMATH_CALUDE_birthday_cookies_l2260_226065

theorem birthday_cookies (friends : ℕ) (packages : ℕ) (cookies_per_package : ℕ) :
  friends = 4 →
  packages = 3 →
  cookies_per_package = 25 →
  (packages * cookies_per_package) / (friends + 1) = 15 :=
by sorry

end NUMINAMATH_CALUDE_birthday_cookies_l2260_226065


namespace NUMINAMATH_CALUDE_factorization_ab_minus_a_l2260_226084

theorem factorization_ab_minus_a (a b : ℝ) : a * b - a = a * (b - 1) := by
  sorry

end NUMINAMATH_CALUDE_factorization_ab_minus_a_l2260_226084


namespace NUMINAMATH_CALUDE_triangle_side_function_is_identity_l2260_226053

/-- A function satisfying the triangle side and perimeter conditions -/
def TriangleSideFunction (f : ℝ → ℝ) : Prop :=
  (∀ x, x > 0 → f x > 0) ∧ 
  (∀ x y z, x > 0 → y > 0 → z > 0 →
    (x + f y > f (f z) + f x ∧ 
     f (f y) + z > x + f y ∧
     f (f z) + f x > f (f y) + z)) ∧
  (∀ p, p > 0 → ∃ x y z, x > 0 ∧ y > 0 ∧ z > 0 ∧
    x + f y + f (f y) + z + f (f z) + f x = p)

/-- The main theorem stating that the identity function is the only function
    satisfying the triangle side and perimeter conditions -/
theorem triangle_side_function_is_identity 
  (f : ℝ → ℝ) (hf : TriangleSideFunction f) : 
  ∀ x, x > 0 → f x = x :=
sorry

end NUMINAMATH_CALUDE_triangle_side_function_is_identity_l2260_226053


namespace NUMINAMATH_CALUDE_parking_cost_theorem_l2260_226078

/-- The number of hours for the initial parking cost -/
def initial_hours : ℝ := 2

/-- The initial parking cost -/
def initial_cost : ℝ := 9

/-- The cost per hour for excess hours -/
def excess_cost_per_hour : ℝ := 1.75

/-- The total number of hours parked -/
def total_hours : ℝ := 9

/-- The average cost per hour for the total parking time -/
def average_cost_per_hour : ℝ := 2.361111111111111

theorem parking_cost_theorem :
  initial_hours = 2 ∧
  initial_cost + excess_cost_per_hour * (total_hours - initial_hours) =
    average_cost_per_hour * total_hours :=
by sorry

end NUMINAMATH_CALUDE_parking_cost_theorem_l2260_226078


namespace NUMINAMATH_CALUDE_six_digit_difference_l2260_226031

/-- Function f for 6-digit numbers -/
def f (n : ℕ) : ℕ :=
  let u := n / 100000 % 10
  let v := n / 10000 % 10
  let w := n / 1000 % 10
  let x := n / 100 % 10
  let y := n / 10 % 10
  let z := n % 10
  2^u * 3^v * 5^w * 7^x * 11^y * 13^z

/-- Theorem: If f(abcdef) = 13 * f(ghijkl), then abcdef - ghijkl = 1 -/
theorem six_digit_difference (abcdef ghijkl : ℕ) 
  (h1 : 100000 ≤ abcdef ∧ abcdef < 1000000)
  (h2 : 100000 ≤ ghijkl ∧ ghijkl < 1000000)
  (h3 : f abcdef = 13 * f ghijkl) : 
  abcdef - ghijkl = 1 := by
  sorry

end NUMINAMATH_CALUDE_six_digit_difference_l2260_226031


namespace NUMINAMATH_CALUDE_two_numbers_divisible_by_three_l2260_226076

def numbers : List Nat := [222, 2222, 22222, 222222]

theorem two_numbers_divisible_by_three : 
  (numbers.filter (fun n => n % 3 = 0)).length = 2 := by sorry

end NUMINAMATH_CALUDE_two_numbers_divisible_by_three_l2260_226076


namespace NUMINAMATH_CALUDE_min_value_quadratic_form_l2260_226020

theorem min_value_quadratic_form (x y : ℝ) :
  x^2 - x*y + 4*y^2 ≥ 0 ∧ (x^2 - x*y + 4*y^2 = 0 ↔ x = 0 ∧ y = 0) := by
  sorry

end NUMINAMATH_CALUDE_min_value_quadratic_form_l2260_226020


namespace NUMINAMATH_CALUDE_dress_original_price_l2260_226052

/-- The original price of a dress given shopping conditions --/
theorem dress_original_price (shoe_discount : ℚ) (dress_discount : ℚ) 
  (shoe_original_price : ℚ) (shoe_quantity : ℕ) (total_spent : ℚ) :
  shoe_discount = 40 / 100 →
  dress_discount = 20 / 100 →
  shoe_original_price = 50 →
  shoe_quantity = 2 →
  total_spent = 140 →
  ∃ (dress_original_price : ℚ),
    dress_original_price = 100 ∧
    total_spent = shoe_quantity * (shoe_original_price * (1 - shoe_discount)) +
                  dress_original_price * (1 - dress_discount) :=
by sorry

end NUMINAMATH_CALUDE_dress_original_price_l2260_226052


namespace NUMINAMATH_CALUDE_negation_equivalence_l2260_226070

-- Define the lines l₁ and l₂
def l₁ (a x y : ℝ) : Prop := x + a * y + 1 = 0
def l₂ (a x y : ℝ) : Prop := a * x + y + 2 = 0

-- Define when two lines are parallel
def parallel (a : ℝ) : Prop := ∀ x y, l₁ a x y ↔ l₂ a x y

-- State the theorem
theorem negation_equivalence :
  ¬(((a = 1) ∨ (a = -1)) → parallel a) ↔ 
  ((a ≠ 1) ∧ (a ≠ -1)) → ¬(parallel a) :=
sorry

end NUMINAMATH_CALUDE_negation_equivalence_l2260_226070


namespace NUMINAMATH_CALUDE_divide_fractions_l2260_226012

theorem divide_fractions (a b c : ℚ) 
  (h1 : a / b = 5 / 3) 
  (h2 : b / c = 4 / 7) : 
  c / a = 21 / 20 := by
sorry

end NUMINAMATH_CALUDE_divide_fractions_l2260_226012


namespace NUMINAMATH_CALUDE_variance_of_successes_l2260_226099

/-- The number of experiments -/
def n : ℕ := 30

/-- The probability of success in a single experiment -/
def p : ℚ := 5/9

/-- The variance of the number of successes in n independent experiments 
    with probability of success p -/
def variance (n : ℕ) (p : ℚ) : ℚ := n * p * (1 - p)

theorem variance_of_successes : variance n p = 200/27 := by
  sorry

end NUMINAMATH_CALUDE_variance_of_successes_l2260_226099


namespace NUMINAMATH_CALUDE_sum_of_seven_squares_not_perfect_square_l2260_226043

theorem sum_of_seven_squares_not_perfect_square (n : ℤ) : 
  ¬∃ (m : ℤ), 7 * (n ^ 2 + 4) = m ^ 2 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_seven_squares_not_perfect_square_l2260_226043


namespace NUMINAMATH_CALUDE_max_sum_of_digits_l2260_226056

theorem max_sum_of_digits (x z : ℕ) : 
  x ≤ 9 → z ≤ 9 → x > z → 99 * (x - z) = 693 → 
  ∃ d : ℕ, d = 11 ∧ ∀ x' z' : ℕ, x' ≤ 9 → z' ≤ 9 → x' > z' → 99 * (x' - z') = 693 → x' + z' ≤ d :=
by sorry

end NUMINAMATH_CALUDE_max_sum_of_digits_l2260_226056


namespace NUMINAMATH_CALUDE_not_in_E_iff_perfect_square_l2260_226094

/-- The set E of floor values of n + √n + 1/2 for natural numbers n -/
def E : Set ℕ := {m | ∃ n : ℕ, m = ⌊(n : ℝ) + Real.sqrt n + 1/2⌋}

/-- A positive integer m is not in set E if and only if it's a perfect square -/
theorem not_in_E_iff_perfect_square (m : ℕ) (hm : m > 0) : 
  m ∉ E ↔ ∃ k : ℕ, m = k^2 := by sorry

end NUMINAMATH_CALUDE_not_in_E_iff_perfect_square_l2260_226094


namespace NUMINAMATH_CALUDE_tileD_in_rectangleII_l2260_226081

-- Define the structure for a tile
structure Tile where
  top : Nat
  right : Nat
  bottom : Nat
  left : Nat

-- Define the tiles
def tileA : Tile := ⟨3, 5, 2, 0⟩
def tileB : Tile := ⟨2, 0, 5, 3⟩
def tileC : Tile := ⟨5, 3, 1, 2⟩
def tileD : Tile := ⟨0, 1, 3, 5⟩

-- Define a function to check if two tiles match on their adjacent sides
def matchTiles (t1 t2 : Tile) (side : Nat) : Prop :=
  match side with
  | 0 => t1.right = t2.left   -- Right of t1 matches Left of t2
  | 1 => t1.bottom = t2.top   -- Bottom of t1 matches Top of t2
  | 2 => t1.left = t2.right   -- Left of t1 matches Right of t2
  | 3 => t1.top = t2.bottom   -- Top of t1 matches Bottom of t2
  | _ => False

-- Theorem stating that Tile D must be in Rectangle II
theorem tileD_in_rectangleII : ∃ (t1 t2 t3 : Tile), 
  (t1 = tileA ∨ t1 = tileB ∨ t1 = tileC) ∧
  (t2 = tileA ∨ t2 = tileB ∨ t2 = tileC) ∧
  (t3 = tileA ∨ t3 = tileB ∨ t3 = tileC) ∧
  t1 ≠ t2 ∧ t1 ≠ t3 ∧ t2 ≠ t3 ∧
  matchTiles t1 tileD 0 ∧
  matchTiles tileD t2 0 ∧
  matchTiles t3 tileD 3 :=
by sorry

end NUMINAMATH_CALUDE_tileD_in_rectangleII_l2260_226081


namespace NUMINAMATH_CALUDE_negation_of_existence_negation_of_proposition_l2260_226022

theorem negation_of_existence (P : ℝ → Prop) :
  (¬ ∃ x, P x) ↔ (∀ x, ¬ P x) := by sorry

theorem negation_of_proposition :
  (¬ ∃ x : ℝ, x^2 + 1 < 2*x) ↔ (∀ x : ℝ, x^2 + 1 ≥ 2*x) := by sorry

end NUMINAMATH_CALUDE_negation_of_existence_negation_of_proposition_l2260_226022


namespace NUMINAMATH_CALUDE_inequality_proof_l2260_226045

theorem inequality_proof (a b : ℝ) (n : ℕ) (ha : 0 < a) (hb : 0 < b) (hab : a ≠ b) :
  (a * b^n)^(1/(n+1 : ℝ)) < (a + n * b) / (n + 1 : ℝ) := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l2260_226045


namespace NUMINAMATH_CALUDE_percentage_problem_l2260_226003

theorem percentage_problem (x : ℝ) : 
  (20 / 100 * 40) + (x / 100 * 60) = 23 ↔ x = 25 := by
  sorry

end NUMINAMATH_CALUDE_percentage_problem_l2260_226003


namespace NUMINAMATH_CALUDE_linear_function_not_in_quadrant_I_l2260_226018

/-- A linear function defined by its slope and y-intercept -/
structure LinearFunction where
  slope : ℝ
  yIntercept : ℝ

/-- Defines the four quadrants of the coordinate plane -/
inductive Quadrant
  | I
  | II
  | III
  | IV

/-- Checks if a point (x, y) is in a given quadrant -/
def inQuadrant (x y : ℝ) (q : Quadrant) : Prop :=
  match q with
  | Quadrant.I => x > 0 ∧ y > 0
  | Quadrant.II => x < 0 ∧ y > 0
  | Quadrant.III => x < 0 ∧ y < 0
  | Quadrant.IV => x > 0 ∧ y < 0

/-- Theorem: The graph of y = -2x - 1 does not pass through Quadrant I -/
theorem linear_function_not_in_quadrant_I :
  let f : LinearFunction := { slope := -2, yIntercept := -1 }
  ∀ x y : ℝ, y = f.slope * x + f.yIntercept → ¬(inQuadrant x y Quadrant.I) :=
by
  sorry


end NUMINAMATH_CALUDE_linear_function_not_in_quadrant_I_l2260_226018


namespace NUMINAMATH_CALUDE_triangle_angle_value_l2260_226011

/-- Theorem: In a triangle with angles 40°, 3x, and x, the value of x is 35°. -/
theorem triangle_angle_value (x : ℝ) : 
  40 + 3 * x + x = 180 → x = 35 := by
  sorry

end NUMINAMATH_CALUDE_triangle_angle_value_l2260_226011


namespace NUMINAMATH_CALUDE_problem_solution_l2260_226006

theorem problem_solution (x y : ℝ) (h1 : y > x) (h2 : x > 0) (h3 : x/y + y/x = 8) :
  (x + y)/(x - y) = Real.sqrt 15 / 3 := by
sorry

end NUMINAMATH_CALUDE_problem_solution_l2260_226006


namespace NUMINAMATH_CALUDE_farmer_land_ownership_l2260_226066

theorem farmer_land_ownership (total_land : ℝ) : 
  (0.9 * total_land * 0.1 = 90) →
  total_land = 1000 := by
  sorry

end NUMINAMATH_CALUDE_farmer_land_ownership_l2260_226066


namespace NUMINAMATH_CALUDE_negation_of_proposition_l2260_226057

theorem negation_of_proposition (p : Prop) :
  (¬ (∀ m : ℝ, m ≥ 0 → 4^m ≥ 4*m)) ↔ (∃ m : ℝ, m ≥ 0 ∧ 4^m < 4*m) :=
by sorry

end NUMINAMATH_CALUDE_negation_of_proposition_l2260_226057


namespace NUMINAMATH_CALUDE_square_plus_reciprocal_square_l2260_226033

theorem square_plus_reciprocal_square (x : ℝ) (h : x + 1/x = 2) : x^2 + 1/x^2 = 2 := by
  sorry

end NUMINAMATH_CALUDE_square_plus_reciprocal_square_l2260_226033


namespace NUMINAMATH_CALUDE_gargamel_tire_savings_l2260_226015

/-- The total amount saved when buying tires on sale -/
def total_savings (num_tires : ℕ) (original_price sale_price : ℚ) : ℚ :=
  (original_price - sale_price) * num_tires

/-- Proof that Gargamel saved $36 on his tire purchase -/
theorem gargamel_tire_savings :
  let num_tires : ℕ := 4
  let original_price : ℚ := 84
  let sale_price : ℚ := 75
  total_savings num_tires original_price sale_price = 36 := by
  sorry

end NUMINAMATH_CALUDE_gargamel_tire_savings_l2260_226015


namespace NUMINAMATH_CALUDE_fraction_simplification_l2260_226042

theorem fraction_simplification (x y : ℝ) (h : x ≠ y) :
  2 / (x + y) - (x - 3*y) / (x^2 - y^2) = 1 / (x - y) := by sorry

end NUMINAMATH_CALUDE_fraction_simplification_l2260_226042


namespace NUMINAMATH_CALUDE_largest_inscribed_circle_and_dot_product_range_l2260_226062

-- Define the plane region
def plane_region (x y : ℝ) : Prop :=
  x - Real.sqrt 3 * y + 4 ≥ 0 ∧ 
  x + Real.sqrt 3 * y + 4 ≥ 0 ∧ 
  x ≤ 2

-- Define the largest inscribed circle
def largest_circle (x y : ℝ) : Prop :=
  x^2 + y^2 = 4

-- Define the geometric sequence condition
def geometric_sequence (pa pm pb : ℝ) : Prop :=
  ∃ r : ℝ, (pa = pm * r ∧ pm = pb * r) ∨ (pm = pa * r ∧ pb = pm * r)

-- Define the dot product of PA and PB
def dot_product_pa_pb (px py : ℝ) : ℝ :=
  (px + 2) * (px - 2) + py * (-py)

-- The main theorem
theorem largest_inscribed_circle_and_dot_product_range :
  (∀ x y : ℝ, plane_region x y → largest_circle x y) ∧
  (∀ px py : ℝ, largest_circle px py →
    (∀ pa pm pb : ℝ, geometric_sequence pa pm pb →
      -2 ≤ dot_product_pa_pb px py ∧ dot_product_pa_pb px py < 0)) := by
  sorry

end NUMINAMATH_CALUDE_largest_inscribed_circle_and_dot_product_range_l2260_226062


namespace NUMINAMATH_CALUDE_wage_ratio_is_two_to_one_l2260_226075

/-- The ratio of a man's daily wage to a woman's daily wage -/
def wage_ratio (men_wage women_wage : ℚ) : ℚ := men_wage / women_wage

/-- The total earnings of a group of workers over a period of time -/
def total_earnings (num_workers : ℕ) (days : ℕ) (daily_wage : ℚ) : ℚ :=
  (num_workers : ℚ) * (days : ℚ) * daily_wage

theorem wage_ratio_is_two_to_one 
  (men_wage women_wage : ℚ)
  (h1 : total_earnings 16 25 men_wage = 14400)
  (h2 : total_earnings 40 30 women_wage = 21600) :
  wage_ratio men_wage women_wage = 2 := by
  sorry

#eval wage_ratio 36 18  -- Expected output: 2

end NUMINAMATH_CALUDE_wage_ratio_is_two_to_one_l2260_226075


namespace NUMINAMATH_CALUDE_power_of_two_plus_three_l2260_226005

/-- Definition of the sequences a_i and b_i -/
def sequence_step (a b : ℤ) : ℤ × ℤ :=
  if a < b then (2*a + 1, b - a - 1)
  else if a > b then (a - b - 1, 2*b + 1)
  else (a, b)

/-- Theorem statement -/
theorem power_of_two_plus_three (n : ℕ) :
  (∃ k : ℕ, ∃ a b : ℕ → ℤ,
    a 0 = 1 ∧ b 0 = n ∧
    (∀ i : ℕ, i > 0 → (a i, b i) = sequence_step (a (i-1)) (b (i-1))) ∧
    a k = b k) →
  ∃ m : ℕ, n + 3 = 2^m := by
  sorry

end NUMINAMATH_CALUDE_power_of_two_plus_three_l2260_226005


namespace NUMINAMATH_CALUDE_geometric_sequence_monotonicity_l2260_226035

/-- A geometric sequence -/
def GeometricSequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = r * a n

/-- A monotonically increasing sequence -/
def MonotonicallyIncreasing (a : ℕ → ℝ) : Prop :=
  ∀ n : ℕ, a n < a (n + 1)

/-- The first three terms of a sequence are strictly increasing -/
def FirstThreeIncreasing (a : ℕ → ℝ) : Prop :=
  a 1 < a 2 ∧ a 2 < a 3

theorem geometric_sequence_monotonicity
  (a : ℕ → ℝ) (h : GeometricSequence a) :
  FirstThreeIncreasing a ↔ MonotonicallyIncreasing a :=
by sorry

end NUMINAMATH_CALUDE_geometric_sequence_monotonicity_l2260_226035


namespace NUMINAMATH_CALUDE_solution_of_equation_l2260_226021

-- Define the piecewise function f
noncomputable def f (x : ℝ) : ℝ :=
  if x > 0 then -3^x else 1 - x^2

-- State the theorem
theorem solution_of_equation (x : ℝ) :
  f x = -3 ↔ x = 1 ∨ x = -2 :=
sorry

end NUMINAMATH_CALUDE_solution_of_equation_l2260_226021


namespace NUMINAMATH_CALUDE_plane_perpendicular_sufficient_not_necessary_l2260_226087

-- Define the basic types
variable (Point Line Plane : Type)

-- Define the relations
variable (in_plane : Line → Plane → Prop)
variable (intersect : Plane → Plane → Line → Prop)
variable (perpendicular : Line → Line → Prop)
variable (plane_perpendicular : Plane → Plane → Prop)

-- State the theorem
theorem plane_perpendicular_sufficient_not_necessary
  (α β : Plane) (m b c : Line) :
  intersect α β m →
  in_plane b α →
  in_plane c β →
  perpendicular c m →
  (plane_perpendicular α β → perpendicular c b) ∧
  ¬(perpendicular c b → plane_perpendicular α β) :=
sorry

end NUMINAMATH_CALUDE_plane_perpendicular_sufficient_not_necessary_l2260_226087


namespace NUMINAMATH_CALUDE_line_perp_parallel_implies_planes_perp_l2260_226072

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the relations
variable (perpendicular : Line → Plane → Prop)
variable (parallel : Line → Plane → Prop)
variable (perp_planes : Plane → Plane → Prop)

-- State the theorem
theorem line_perp_parallel_implies_planes_perp
  (a : Line) (M N : Plane)
  (h1 : perpendicular a M)
  (h2 : parallel a N) :
  perp_planes M N :=
sorry

end NUMINAMATH_CALUDE_line_perp_parallel_implies_planes_perp_l2260_226072


namespace NUMINAMATH_CALUDE_A_intersect_B_l2260_226082

def A : Set ℕ := {0, 1, 2, 3}
def B : Set ℕ := {x | x < 3}

theorem A_intersect_B : A ∩ B = {0, 1, 2} := by
  sorry

end NUMINAMATH_CALUDE_A_intersect_B_l2260_226082


namespace NUMINAMATH_CALUDE_rectangle_area_diagonal_l2260_226071

/-- Given a rectangle with length-to-width ratio of 5:2 and diagonal d, 
    prove that its area A can be expressed as A = kd^2, where k = 10/29 -/
theorem rectangle_area_diagonal (d : ℝ) (h : d > 0) : ∃ (l w : ℝ),
  l > 0 ∧ w > 0 ∧ l / w = 5 / 2 ∧ l ^ 2 + w ^ 2 = d ^ 2 ∧ l * w = (10 / 29) * d ^ 2 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_area_diagonal_l2260_226071


namespace NUMINAMATH_CALUDE_robert_ate_more_than_nickel_l2260_226000

/-- The number of chocolates Robert ate -/
def robert_chocolates : ℕ := 7

/-- The number of chocolates Nickel ate -/
def nickel_chocolates : ℕ := 5

/-- The difference in chocolates eaten between Robert and Nickel -/
def chocolate_difference : ℕ := robert_chocolates - nickel_chocolates

theorem robert_ate_more_than_nickel : chocolate_difference = 2 := by
  sorry

end NUMINAMATH_CALUDE_robert_ate_more_than_nickel_l2260_226000


namespace NUMINAMATH_CALUDE_inequality_solution_l2260_226032

theorem inequality_solution (a : ℝ) : 
  (∀ x : ℝ, x > 0 → (a * x - 20) * Real.log (2 * a / x) ≤ 0) ↔ a = Real.sqrt 10 := by
  sorry

end NUMINAMATH_CALUDE_inequality_solution_l2260_226032


namespace NUMINAMATH_CALUDE_third_smallest_four_digit_in_pascal_l2260_226088

/-- Pascal's triangle coefficient -/
def pascal (n k : ℕ) : ℕ := sorry

/-- The n-th row of Pascal's triangle -/
def pascal_row (n : ℕ) : List ℕ := sorry

/-- Predicate for a number being four digits -/
def is_four_digit (n : ℕ) : Prop := 1000 ≤ n ∧ n ≤ 9999

/-- The third smallest four-digit number in Pascal's triangle -/
theorem third_smallest_four_digit_in_pascal : 
  ∃ (n k : ℕ), pascal n k = 1002 ∧ 
  (∀ (m l : ℕ), pascal m l < 1002 → ¬(is_four_digit (pascal m l))) ∧
  (∃! (p q r s : ℕ), 
    p ≠ q ∧ q ≠ r ∧ p ≠ r ∧
    is_four_digit (pascal p s) ∧ 
    is_four_digit (pascal q s) ∧
    pascal p s < pascal q s ∧
    pascal q s < 1002) := by sorry

end NUMINAMATH_CALUDE_third_smallest_four_digit_in_pascal_l2260_226088


namespace NUMINAMATH_CALUDE_jar_to_pot_ratio_l2260_226026

/-- Proves that the ratio of jars to clay pots is 2:1 given the problem conditions --/
theorem jar_to_pot_ratio :
  ∀ (num_pots : ℕ),
  (∃ (k : ℕ), 16 = k * num_pots) →
  16 * 5 + num_pots * (5 * 3) = 200 →
  (16 : ℚ) / num_pots = 2 := by
  sorry

end NUMINAMATH_CALUDE_jar_to_pot_ratio_l2260_226026


namespace NUMINAMATH_CALUDE_square_difference_l2260_226046

theorem square_difference (x y : ℚ) 
  (h1 : x + y = 9/17) 
  (h2 : x - y = 1/119) : 
  x^2 - y^2 = 9/2003 := by
  sorry

end NUMINAMATH_CALUDE_square_difference_l2260_226046


namespace NUMINAMATH_CALUDE_sum_of_powers_l2260_226010

theorem sum_of_powers (x : ℝ) (h1 : x^10 - 3*x + 2 = 0) (h2 : x ≠ 1) :
  x^9 + x^8 + x^7 + x^6 + x^5 + x^4 + x^3 + x^2 + x + 1 = 3 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_powers_l2260_226010


namespace NUMINAMATH_CALUDE_min_value_fraction_l2260_226049

theorem min_value_fraction (x y : ℝ) (hx : 0 ≤ x ∧ x ≤ 3) (hy : 1 ≤ y ∧ y ≤ 4) :
  (x + y) / x ≥ 4/3 := by
  sorry

end NUMINAMATH_CALUDE_min_value_fraction_l2260_226049


namespace NUMINAMATH_CALUDE_problem_solution_l2260_226054

theorem problem_solution (θ : Real) (x : Real) :
  let A := (5 * Real.sin θ + 4 * Real.cos θ) / (3 * Real.sin θ + Real.cos θ)
  let B := x^3 + 1/x^3
  Real.tan θ = 2 →
  x + 1/x = 2 * A →
  A = 2 ∧ B = 52 := by
sorry

end NUMINAMATH_CALUDE_problem_solution_l2260_226054


namespace NUMINAMATH_CALUDE_eleven_power_2023_mod_5_l2260_226050

theorem eleven_power_2023_mod_5 : 11^2023 % 5 = 1 := by sorry

end NUMINAMATH_CALUDE_eleven_power_2023_mod_5_l2260_226050


namespace NUMINAMATH_CALUDE_point_coordinates_l2260_226085

theorem point_coordinates (M N P : ℝ × ℝ) : 
  M = (3, -2) → 
  N = (-5, -1) → 
  P - M = (1/2 : ℝ) • (N - M) → 
  P = (-1, -3/2) := by
  sorry

end NUMINAMATH_CALUDE_point_coordinates_l2260_226085


namespace NUMINAMATH_CALUDE_expand_expression_l2260_226047

theorem expand_expression (x y : ℝ) : 25 * (3 * x + 6 - 4 * y) = 75 * x + 150 - 100 * y := by
  sorry

end NUMINAMATH_CALUDE_expand_expression_l2260_226047


namespace NUMINAMATH_CALUDE_modular_inverse_30_mod_31_l2260_226089

theorem modular_inverse_30_mod_31 : ∃ x : ℕ, x ≤ 31 ∧ (30 * x) % 31 = 1 :=
by
  -- The proof goes here
  sorry

end NUMINAMATH_CALUDE_modular_inverse_30_mod_31_l2260_226089


namespace NUMINAMATH_CALUDE_line_l_equation_circle_M_equations_l2260_226064

-- Define the lines l₁ and l₂
def l₁ (x y : ℝ) : Prop := 2 * x - y = 0
def l₂ (x y : ℝ) : Prop := x + y + 2 = 0

-- Define the point P
def P : ℝ × ℝ := (1, 1)

-- Define perpendicularity of lines
def perpendicular (m₁ m₂ : ℝ) : Prop := m₁ * m₂ = -1

-- Define the equation of line l
def l (x y : ℝ) : Prop := x + 2 * y - 3 = 0

-- Define the equations of circle M
def M₁ (x y : ℝ) : Prop := (x + 5/7)^2 + (y + 10/7)^2 = 25/49
def M₂ (x y : ℝ) : Prop := (x + 1)^2 + (y + 2)^2 = 1

-- Theorem for the equation of line l
theorem line_l_equation : 
  ∀ x y : ℝ, l x y ↔ (∃ m : ℝ, perpendicular m 2 ∧ y - P.2 = m * (x - P.1)) :=
sorry

-- Theorem for the equations of circle M
theorem circle_M_equations :
  ∀ x y : ℝ, 
    (∃ a b r : ℝ, 
      l₁ a b ∧ 
      (∀ t : ℝ, (t - a)^2 + b^2 = r^2 → t = 0) ∧ 
      ((a + b + 2)^2 / 2 + 1/2 = r^2) ∧
      ((x - a)^2 + (y - b)^2 = r^2)) 
    ↔ (M₁ x y ∨ M₂ x y) :=
sorry

end NUMINAMATH_CALUDE_line_l_equation_circle_M_equations_l2260_226064


namespace NUMINAMATH_CALUDE_pepperoni_pizza_coverage_l2260_226073

theorem pepperoni_pizza_coverage (pizza_diameter : ℝ) (pepperoni_count : ℕ) 
  (pepperoni_across : ℕ) : 
  pizza_diameter = 12 →
  pepperoni_across = 8 →
  pepperoni_count = 32 →
  (pepperoni_count * (pizza_diameter / pepperoni_across / 2)^2) / 
  (pizza_diameter / 2)^2 = 1 / 2 := by
  sorry

#check pepperoni_pizza_coverage

end NUMINAMATH_CALUDE_pepperoni_pizza_coverage_l2260_226073


namespace NUMINAMATH_CALUDE_profit_percentage_l2260_226019

theorem profit_percentage (cost_price selling_price : ℝ) 
  (h : cost_price = 0.96 * selling_price) : 
  (selling_price - cost_price) / cost_price * 100 = (1 / 0.96 - 1) * 100 := by
  sorry

end NUMINAMATH_CALUDE_profit_percentage_l2260_226019


namespace NUMINAMATH_CALUDE_expression_simplification_l2260_226067

theorem expression_simplification (a b : ℚ) (ha : a = -1) (hb : b = 1/2) :
  2 * a^2 * b - (3 * a * b^2 - (4 * a * b^2 - 2 * a^2 * b)) = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_expression_simplification_l2260_226067


namespace NUMINAMATH_CALUDE_sparcs_characterization_l2260_226092

-- Define "grows to"
def grows_to (s r : ℝ) : Prop :=
  ∃ n : ℕ+, s ^ (n : ℝ) = r

-- Define "sparcs"
def sparcs (r : ℝ) : Prop :=
  {s : ℝ | grows_to s r}.Finite

-- Theorem statement
theorem sparcs_characterization (r : ℝ) :
  sparcs r ↔ r = -1 ∨ r = 0 ∨ r = 1 := by
  sorry

end NUMINAMATH_CALUDE_sparcs_characterization_l2260_226092


namespace NUMINAMATH_CALUDE_translate_f_to_g_l2260_226004

def f (x : ℝ) : ℝ := 2 * x^2

def g (x : ℝ) : ℝ := 2 * (x + 1)^2 + 3

theorem translate_f_to_g : 
  ∀ x : ℝ, g x = f (x + 1) + 3 := by sorry

end NUMINAMATH_CALUDE_translate_f_to_g_l2260_226004


namespace NUMINAMATH_CALUDE_work_rate_proof_l2260_226058

/-- The work rate of person A per day -/
def work_rate_A : ℚ := 1 / 4

/-- The work rate of person B per day -/
def work_rate_B : ℚ := 1 / 2

/-- The work rate of person C per day -/
def work_rate_C : ℚ := 1 / 8

/-- The combined work rate of A, B, and C per day -/
def combined_work_rate : ℚ := work_rate_A + work_rate_B + work_rate_C

theorem work_rate_proof : combined_work_rate = 7 / 8 := by
  sorry

end NUMINAMATH_CALUDE_work_rate_proof_l2260_226058


namespace NUMINAMATH_CALUDE_expression_value_l2260_226025

theorem expression_value (a b c d m : ℝ) 
  (h1 : a + b = 0)  -- a and b are opposite numbers
  (h2 : c * d = 1)  -- c and d are reciprocals
  (h3 : |m| = 3)    -- absolute value of m is 3
  : (a + b) / m + m^2 - c * d = 8 := by
  sorry

end NUMINAMATH_CALUDE_expression_value_l2260_226025


namespace NUMINAMATH_CALUDE_two_digit_number_sum_l2260_226027

theorem two_digit_number_sum (n : ℕ) : 
  (10 ≤ n ∧ n < 100) →  -- n is a two-digit number
  (n / 2 : ℚ) = (n / 4 : ℚ) + 3 →  -- one half of n exceeds its one fourth by 3
  (n / 10 + n % 10 = 3) :=  -- sum of digits is 3
by
  sorry

end NUMINAMATH_CALUDE_two_digit_number_sum_l2260_226027


namespace NUMINAMATH_CALUDE_solution_set_part1_range_of_a_part2_l2260_226014

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := x^2 + 2*x + a

-- Part 1: Solution set for f(x) > 1 when a = -2
theorem solution_set_part1 :
  {x : ℝ | f (-2) x > 1} = {x : ℝ | x < -3 ∨ x > 1} := by sorry

-- Part 2: Range of a when f(x) > 0 for all x ∈ [1, +∞)
theorem range_of_a_part2 :
  (∀ x : ℝ, x ≥ 1 → f a x > 0) ↔ a > -3 := by sorry

end NUMINAMATH_CALUDE_solution_set_part1_range_of_a_part2_l2260_226014


namespace NUMINAMATH_CALUDE_triangle_properties_l2260_226096

/-- Triangle ABC with side lengths a, b, c corresponding to angles A, B, C -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

/-- The theorem statement -/
theorem triangle_properties (t : Triangle) 
  (h1 : Real.sqrt 3 * Real.sin t.B - Real.cos t.B = 2 * Real.sin (t.B - π / 6))
  (h2 : t.b = 1)
  (h3 : t.A = 5 * π / 12) :
  (t.c = Real.sqrt 6 / 3) ∧ 
  (∀ h : ℝ, h ≤ Real.sqrt 3 / 2 → 
    ∃ (a c : ℝ), 
      a > 0 ∧ c > 0 ∧ 
      a * c ≤ 1 ∧ 
      h = Real.sqrt 3 / 2 * a * c) := by
  sorry

end NUMINAMATH_CALUDE_triangle_properties_l2260_226096


namespace NUMINAMATH_CALUDE_min_value_expression_l2260_226039

theorem min_value_expression (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  1/a + a/(b^2) + b ≥ 2 * Real.sqrt 2 ∧
  (1/a + a/(b^2) + b = 2 * Real.sqrt 2 ↔ a = Real.sqrt 2 ∧ b = Real.sqrt 2) :=
by sorry

end NUMINAMATH_CALUDE_min_value_expression_l2260_226039


namespace NUMINAMATH_CALUDE_least_positive_angle_theorem_l2260_226008

/-- The least positive angle θ (in degrees) satisfying sin 15° = cos 40° + cos θ is 115° -/
theorem least_positive_angle_theorem : 
  let θ : ℝ := 115
  ∀ φ : ℝ, 0 < φ ∧ φ < θ → 
    Real.sin (15 * π / 180) ≠ Real.cos (40 * π / 180) + Real.cos (φ * π / 180) ∧
    Real.sin (15 * π / 180) = Real.cos (40 * π / 180) + Real.cos (θ * π / 180) :=
by sorry

end NUMINAMATH_CALUDE_least_positive_angle_theorem_l2260_226008


namespace NUMINAMATH_CALUDE_negative_sqrt_two_less_than_negative_one_l2260_226040

theorem negative_sqrt_two_less_than_negative_one : -Real.sqrt 2 < -1 := by
  sorry

end NUMINAMATH_CALUDE_negative_sqrt_two_less_than_negative_one_l2260_226040


namespace NUMINAMATH_CALUDE_g_4_equals_7_5_l2260_226080

noncomputable def f (x : ℝ) : ℝ := 4 / (3 - x)

noncomputable def g (x : ℝ) : ℝ := 1 / (f⁻¹ x) + 7

theorem g_4_equals_7_5 : g 4 = 7.5 := by sorry

end NUMINAMATH_CALUDE_g_4_equals_7_5_l2260_226080


namespace NUMINAMATH_CALUDE_envelope_length_l2260_226017

/-- Given a rectangular envelope with width 4 inches and area 16 square inches,
    prove that its length is 4 inches. -/
theorem envelope_length (width : ℝ) (area : ℝ) (length : ℝ) : 
  width = 4 → area = 16 → area = width * length → length = 4 := by
  sorry

end NUMINAMATH_CALUDE_envelope_length_l2260_226017


namespace NUMINAMATH_CALUDE_swallowing_not_complete_disappearance_l2260_226059

/-- Represents a snake in the swallowing process --/
structure Snake where
  length : ℝ
  swallowed : ℝ

/-- Represents the state of two snakes swallowing each other --/
structure SwallowingState where
  snake1 : Snake
  snake2 : Snake
  ring_size : ℝ

/-- The swallowing process between two snakes --/
def swallowing_process (initial_state : SwallowingState) : Prop :=
  ∀ t : ℝ, t ≥ 0 →
    ∃ state : SwallowingState,
      state.ring_size < initial_state.ring_size ∧
      state.snake1.swallowed > initial_state.snake1.swallowed ∧
      state.snake2.swallowed > initial_state.snake2.swallowed ∧
      state.snake1.length + state.snake2.length > 0

/-- Theorem stating that the swallowing process does not result in complete disappearance --/
theorem swallowing_not_complete_disappearance (initial_state : SwallowingState) :
  swallowing_process initial_state →
  ∃ final_state : SwallowingState, final_state.snake1.length + final_state.snake2.length > 0 :=
by sorry

end NUMINAMATH_CALUDE_swallowing_not_complete_disappearance_l2260_226059

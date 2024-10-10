import Mathlib

namespace total_germs_count_l2779_277904

/-- The number of petri dishes in the biology lab. -/
def num_dishes : ℕ := 10800

/-- The number of germs in a single petri dish. -/
def germs_per_dish : ℕ := 500

/-- The total number of germs in the biology lab. -/
def total_germs : ℕ := num_dishes * germs_per_dish

/-- Theorem stating that the total number of germs is 5,400,000. -/
theorem total_germs_count : total_germs = 5400000 := by
  sorry

end total_germs_count_l2779_277904


namespace longest_perimeter_l2779_277968

theorem longest_perimeter (x : ℝ) 
  (hx : x > 1)
  (perimeterA : ℝ := 4 + 6*x)
  (perimeterB : ℝ := 2 + 10*x)
  (perimeterC : ℝ := 7 + 5*x)
  (perimeterD : ℝ := 6 + 6*x)
  (perimeterE : ℝ := 1 + 11*x) :
  perimeterE > perimeterA ∧ 
  perimeterE > perimeterB ∧ 
  perimeterE > perimeterC ∧ 
  perimeterE > perimeterD :=
by
  sorry

end longest_perimeter_l2779_277968


namespace exam_time_allocation_l2779_277909

theorem exam_time_allocation :
  ∀ (total_time : ℕ) (total_questions : ℕ) (type_a_questions : ℕ) (type_b_questions : ℕ),
    total_time = 3 * 60 →
    total_questions = 200 →
    type_a_questions = 50 →
    type_b_questions = total_questions - type_a_questions →
    2 * (total_time / total_questions) * type_b_questions = 
      (total_time / total_questions) * type_a_questions →
    (total_time / total_questions) * type_a_questions = 72 :=
by sorry

end exam_time_allocation_l2779_277909


namespace f_of_two_equals_one_l2779_277919

-- Define the function f
def f : ℝ → ℝ := fun x => (x - 1)^2

-- Theorem statement
theorem f_of_two_equals_one : f 2 = 1 := by
  sorry

end f_of_two_equals_one_l2779_277919


namespace solution_set_equation_l2779_277964

theorem solution_set_equation : 
  ∀ x : ℝ, ((x - 1) / x)^2 - (7/2) * ((x - 1) / x) + 3 = 0 ↔ x = -1 ∨ x = -2 := by
  sorry

end solution_set_equation_l2779_277964


namespace task_fraction_by_B_l2779_277933

theorem task_fraction_by_B (a b : ℚ) : 
  (a = (2/5) * b) → (b = (5/7) * (a + b)) := by
  sorry

end task_fraction_by_B_l2779_277933


namespace cos_135_degrees_l2779_277914

theorem cos_135_degrees : Real.cos (135 * π / 180) = -Real.sqrt 2 / 2 := by
  sorry

end cos_135_degrees_l2779_277914


namespace fraction_zero_implies_x_negative_two_l2779_277979

theorem fraction_zero_implies_x_negative_two (x : ℝ) :
  (abs x - 2) / (2 - x) = 0 → x = -2 :=
sorry

end fraction_zero_implies_x_negative_two_l2779_277979


namespace x_eq_one_iff_quadratic_eq_zero_l2779_277947

theorem x_eq_one_iff_quadratic_eq_zero : ∀ x : ℝ, x = 1 ↔ x^2 - 2*x + 1 = 0 := by sorry

end x_eq_one_iff_quadratic_eq_zero_l2779_277947


namespace computer_literate_female_employees_l2779_277962

/-- Given an office in Singapore with the following conditions:
  * There are 1100 total employees
  * 60% of employees are female
  * 50% of male employees are computer literate
  * 62% of all employees are computer literate
  Prove that the number of female employees who are computer literate is 462 -/
theorem computer_literate_female_employees 
  (total_employees : ℕ) 
  (female_percentage : ℚ)
  (male_literate_percentage : ℚ)
  (total_literate_percentage : ℚ)
  (h1 : total_employees = 1100)
  (h2 : female_percentage = 60 / 100)
  (h3 : male_literate_percentage = 50 / 100)
  (h4 : total_literate_percentage = 62 / 100) :
  ↑⌊(total_literate_percentage * total_employees - 
     male_literate_percentage * ((1 - female_percentage) * total_employees))⌋ = 462 := by
  sorry

end computer_literate_female_employees_l2779_277962


namespace expression_simplification_l2779_277982

theorem expression_simplification (a : ℝ) (h : a^2 + 4*a + 1 = 0) :
  ((a + 2) / (a^2 - 2*a) + 8 / (4 - a^2)) / ((a^2 - 4) / a) = 1/3 := by
sorry

end expression_simplification_l2779_277982


namespace sum_of_x_and_y_l2779_277974

theorem sum_of_x_and_y (x y : ℝ) (hx : x + 2 = 10) (hy : y - 1 = 6) : x + y = 15 := by
  sorry

end sum_of_x_and_y_l2779_277974


namespace consecutive_integers_median_l2779_277924

theorem consecutive_integers_median (n : ℕ) (sum : ℕ) (h1 : n = 25) (h2 : sum = 3125) :
  (sum : ℚ) / n = 125 := by
  sorry

end consecutive_integers_median_l2779_277924


namespace sphere_in_cylindrical_hole_l2779_277967

theorem sphere_in_cylindrical_hole (r : ℝ) (h : ℝ) :
  h = 2 ∧ 
  6^2 + (r - h)^2 = r^2 →
  r = 10 ∧ 4 * Real.pi * r^2 = 400 * Real.pi :=
by sorry

end sphere_in_cylindrical_hole_l2779_277967


namespace shift_direct_proportion_l2779_277934

/-- Represents a linear function in the form y = mx + b -/
structure LinearFunction where
  m : ℝ
  b : ℝ

/-- Represents a horizontal shift transformation on a function -/
def horizontalShift (f : ℝ → ℝ) (shift : ℝ) : ℝ → ℝ :=
  fun x => f (x - shift)

/-- The original direct proportion function y = -2x -/
def originalFunction : ℝ → ℝ :=
  fun x => -2 * x

theorem shift_direct_proportion :
  ∃ (f : LinearFunction),
    f.m = -2 ∧
    f.b = 6 ∧
    (∀ x, (horizontalShift originalFunction 3) x = f.m * x + f.b) := by
  sorry

end shift_direct_proportion_l2779_277934


namespace count_divisors_with_specific_remainder_l2779_277918

theorem count_divisors_with_specific_remainder :
  ∃ (S : Finset ℕ), 
    (∀ n ∈ S, n > 17 ∧ 2017 % n = 17) ∧
    (∀ n : ℕ, n > 17 ∧ 2017 % n = 17 → n ∈ S) ∧
    S.card = 13 :=
by sorry

end count_divisors_with_specific_remainder_l2779_277918


namespace percentage_difference_l2779_277917

theorem percentage_difference : (65 / 100 * 40) - (4 / 5 * 25) = 6 := by
  sorry

end percentage_difference_l2779_277917


namespace candy_cost_620_l2779_277973

/-- Calculates the cost of buying candies given the pricing structure -/
def candy_cost (total_candies : ℕ) : ℕ :=
  let regular_price := 8
  let discount_price := 7
  let candies_per_box := 40
  let discount_threshold := 500
  let full_price_boxes := min (total_candies / candies_per_box) (discount_threshold / candies_per_box)
  let discounted_boxes := (total_candies - full_price_boxes * candies_per_box + candies_per_box - 1) / candies_per_box
  full_price_boxes * regular_price + discounted_boxes * discount_price

theorem candy_cost_620 : candy_cost 620 = 125 := by
  sorry

end candy_cost_620_l2779_277973


namespace sqrt_inequality_l2779_277939

theorem sqrt_inequality (a b : ℝ) (ha : a > 0) (hb : 1/b - 1/a > 1) :
  Real.sqrt (1 + a) > 1 / Real.sqrt (1 - b) := by
  sorry

end sqrt_inequality_l2779_277939


namespace inverse_variation_problem_l2779_277922

theorem inverse_variation_problem (x y : ℝ) :
  (∀ (x y : ℝ), x > 0 ∧ y > 0) →
  (∃ (k : ℝ), ∀ (x y : ℝ), x^3 * y = k) →
  (2^3 * 5 = k) →
  (x^3 * 2000 = k) →
  x = 1 / Real.rpow 50 (1/3) :=
by sorry

end inverse_variation_problem_l2779_277922


namespace equation_equality_l2779_277925

theorem equation_equality (a b : ℝ) : -a*b + 3*b*a = 2*a*b := by
  sorry

end equation_equality_l2779_277925


namespace restaurant_weekday_earnings_l2779_277951

/-- Represents the daily earnings of a restaurant on weekdays -/
def weekday_earnings : ℝ := sorry

/-- Represents the daily earnings of a restaurant on weekend days -/
def weekend_earnings : ℝ := 2 * weekday_earnings

/-- The number of weeks in a month -/
def weeks_in_month : ℕ := 4

/-- The number of weekdays in a week -/
def weekdays_in_week : ℕ := 5

/-- The total monthly earnings of the restaurant -/
def total_monthly_earnings : ℝ := 21600

/-- Theorem stating that the daily weekday earnings of the restaurant are $600 -/
theorem restaurant_weekday_earnings :
  weekday_earnings = 600 :=
by sorry

end restaurant_weekday_earnings_l2779_277951


namespace quadratic_inequality_solution_l2779_277930

theorem quadratic_inequality_solution (m n : ℝ) : 
  (∀ x, x^2 - m*x + n ≤ 0 ↔ -5 ≤ x ∧ x ≤ 1) → m = -4 ∧ n = -5 := by
  sorry

end quadratic_inequality_solution_l2779_277930


namespace intersection_difference_l2779_277938

theorem intersection_difference (A B : Set ℕ) (m n : ℕ) :
  A = {1, 2, m} →
  B = {2, 3, 4, n} →
  A ∩ B = {1, 2, 3} →
  m - n = 2 := by
sorry

end intersection_difference_l2779_277938


namespace james_pays_40_l2779_277949

/-- The amount James pays for stickers -/
def james_payment (packs : ℕ) (stickers_per_pack : ℕ) (cost_per_sticker : ℚ) : ℚ :=
  (packs * stickers_per_pack * cost_per_sticker) / 2

/-- Theorem: James pays $40 for the stickers -/
theorem james_pays_40 :
  james_payment 8 40 (1/4) = 40 := by
  sorry

end james_pays_40_l2779_277949


namespace midpoint_dot_product_sum_of_squares_l2779_277935

/-- Given vectors a and b in ℝ², if m is their midpoint [3, 7] and their dot product is 6,
    then the sum of their squared norms is 220. -/
theorem midpoint_dot_product_sum_of_squares (a b : Fin 2 → ℝ) :
  let m : Fin 2 → ℝ := ![3, 7]
  (∀ i, m i = (a i + b i) / 2) →
  a • b = 6 →
  ‖a‖^2 + ‖b‖^2 = 220 := by
  sorry

end midpoint_dot_product_sum_of_squares_l2779_277935


namespace pet_store_puppies_l2779_277912

def initial_puppies (sold : ℕ) (puppies_per_cage : ℕ) (cages_used : ℕ) : ℕ :=
  sold + (puppies_per_cage * cages_used)

theorem pet_store_puppies :
  initial_puppies 7 2 3 = 13 := by
  sorry

end pet_store_puppies_l2779_277912


namespace mixture_composition_l2779_277996

theorem mixture_composition 
  (p_carbonated : ℝ) 
  (q_carbonated : ℝ) 
  (mixture_carbonated : ℝ) 
  (h1 : p_carbonated = 0.80) 
  (h2 : q_carbonated = 0.55) 
  (h3 : mixture_carbonated = 0.72) :
  let p := (mixture_carbonated - q_carbonated) / (p_carbonated - q_carbonated)
  p = 0.68 := by sorry

end mixture_composition_l2779_277996


namespace sequence_next_terms_l2779_277959

def sequence1 : ℕ → ℕ
  | 0 => 2
  | n + 1 => sequence1 n + 2

def sequence2 : ℕ → ℕ
  | 0 => 3
  | n + 1 => sequence2 n * 2

def sequence3 : ℕ → ℕ
  | 0 => 36
  | 1 => 11
  | n + 2 => sequence3 n + 2

theorem sequence_next_terms :
  (sequence1 5 = 12 ∧ sequence1 6 = 14) ∧
  (sequence2 5 = 96) ∧
  (sequence3 8 = 44 ∧ sequence3 9 = 19) := by
  sorry

end sequence_next_terms_l2779_277959


namespace hector_gumballs_l2779_277906

/-- The number of gumballs Hector gave to Todd -/
def todd_gumballs : ℕ := 4

/-- The number of gumballs Hector gave to Alisha -/
def alisha_gumballs : ℕ := 2 * todd_gumballs

/-- The number of gumballs Hector gave to Bobby -/
def bobby_gumballs : ℕ := 4 * alisha_gumballs - 5

/-- The number of gumballs Hector had remaining -/
def remaining_gumballs : ℕ := 6

/-- The total number of gumballs Hector purchased -/
def total_gumballs : ℕ := todd_gumballs + alisha_gumballs + bobby_gumballs + remaining_gumballs

theorem hector_gumballs : total_gumballs = 45 := by
  sorry

end hector_gumballs_l2779_277906


namespace train_length_l2779_277987

/-- Proves that a train with the given conditions has a length of 300 meters -/
theorem train_length (train_speed : ℝ) (crossing_time : ℝ) (train_length platform_length : ℝ) : 
  train_speed = 36 * 5/18 → -- Convert 36 km/hr to m/s
  crossing_time = 60 → -- One minute in seconds
  train_length = platform_length →
  train_length + platform_length = train_speed * crossing_time →
  train_length = 300 := by
  sorry

end train_length_l2779_277987


namespace sam_bought_cards_l2779_277955

/-- The number of baseball cards Sam bought from Mike -/
def cards_bought (initial_cards current_cards : ℕ) : ℕ :=
  initial_cards - current_cards

/-- Theorem stating that the number of cards Sam bought is the difference between Mike's initial and current number of cards -/
theorem sam_bought_cards (mike_initial mike_current : ℕ) 
  (h1 : mike_initial = 87) 
  (h2 : mike_current = 74) : 
  cards_bought mike_initial mike_current = 13 := by
  sorry

end sam_bought_cards_l2779_277955


namespace divisibility_implication_l2779_277936

theorem divisibility_implication (x y : ℤ) :
  ∃ k : ℤ, 14 * x + 13 * y = 11 * k → ∃ m : ℤ, 19 * x + 9 * y = 11 * m :=
by sorry

end divisibility_implication_l2779_277936


namespace largest_mersenne_prime_under_500_l2779_277948

def is_mersenne_prime (p : ℕ) : Prop :=
  ∃ n : ℕ, Prime n ∧ p = 2^n - 1 ∧ Prime p

theorem largest_mersenne_prime_under_500 :
  ∃ p : ℕ, p = 127 ∧ 
    is_mersenne_prime p ∧ 
    p < 500 ∧ 
    ∀ q : ℕ, is_mersenne_prime q → q < 500 → q ≤ p :=
by sorry

end largest_mersenne_prime_under_500_l2779_277948


namespace team_average_typing_speed_l2779_277952

def team_size : ℕ := 5

def typing_speeds : List ℕ := [64, 76, 91, 80, 89]

def average_typing_speed : ℚ := (typing_speeds.sum : ℚ) / team_size

theorem team_average_typing_speed :
  average_typing_speed = 80 := by sorry

end team_average_typing_speed_l2779_277952


namespace perpendicular_diagonals_imply_square_rectangle_is_not_square_l2779_277960

-- Define a quadrilateral
structure Quadrilateral :=
  (has_right_angles : Bool)
  (opposite_sides_parallel_equal : Bool)
  (diagonals_bisect : Bool)
  (diagonals_perpendicular : Bool)

-- Define a rectangle
def Rectangle : Quadrilateral :=
  { has_right_angles := true,
    opposite_sides_parallel_equal := true,
    diagonals_bisect := true,
    diagonals_perpendicular := false }

-- Define a square
def Square : Quadrilateral :=
  { has_right_angles := true,
    opposite_sides_parallel_equal := true,
    diagonals_bisect := true,
    diagonals_perpendicular := true }

-- Theorem: A quadrilateral with right angles, opposite sides parallel and equal,
-- and perpendicular diagonals that bisect each other is a square
theorem perpendicular_diagonals_imply_square (q : Quadrilateral) :
  q.has_right_angles = true →
  q.opposite_sides_parallel_equal = true →
  q.diagonals_bisect = true →
  q.diagonals_perpendicular = true →
  q = Square := by
  sorry

-- Theorem: A rectangle is not a square
theorem rectangle_is_not_square : Rectangle ≠ Square := by
  sorry

end perpendicular_diagonals_imply_square_rectangle_is_not_square_l2779_277960


namespace quadratic_function_inequality_l2779_277977

theorem quadratic_function_inequality (a : ℝ) (x₁ x₂ : ℝ) 
  (h₁ : 0 < a) (h₂ : a < 3) (h₃ : x₁ < x₂) (h₄ : x₁ + x₂ ≠ 1 - a) :
  let f := fun x => a * x^2 + 2 * a * x + 4
  f x₁ < f x₂ := by
  sorry

end quadratic_function_inequality_l2779_277977


namespace complex_power_equivalence_l2779_277966

theorem complex_power_equivalence :
  (Complex.exp (Complex.I * Real.pi * (35 / 180)))^100 = Complex.exp (Complex.I * Real.pi * (20 / 180)) := by
  sorry

end complex_power_equivalence_l2779_277966


namespace ferry_time_difference_l2779_277999

/-- Proves that the difference in travel time between Ferry Q and Ferry P is 1 hour -/
theorem ferry_time_difference
  (time_p : ℝ) (speed_p : ℝ) (speed_difference : ℝ) (route_factor : ℝ) :
  time_p = 3 →
  speed_p = 8 →
  speed_difference = 4 →
  route_factor = 2 →
  let distance_p := time_p * speed_p
  let distance_q := route_factor * distance_p
  let speed_q := speed_p + speed_difference
  let time_q := distance_q / speed_q
  time_q - time_p = 1 := by
sorry

end ferry_time_difference_l2779_277999


namespace cone_volume_over_pi_l2779_277994

-- Define the given parameters
def sector_angle : ℝ := 240
def circle_radius : ℝ := 15

-- Define the theorem
theorem cone_volume_over_pi (sector_angle : ℝ) (circle_radius : ℝ) :
  sector_angle = 240 ∧ circle_radius = 15 →
  ∃ (cone_volume : ℝ),
    cone_volume / π = 500 * Real.sqrt 5 / 3 := by
  sorry


end cone_volume_over_pi_l2779_277994


namespace geometric_progression_values_l2779_277927

theorem geometric_progression_values (p : ℝ) : 
  (∃ r : ℝ, r ≠ 0 ∧ (9 * p + 10) * r = 3 * p ∧ (3 * p) * r = |p - 8|) ↔ 
  (p = -1 ∨ p = 40 / 9) := by
  sorry

end geometric_progression_values_l2779_277927


namespace infinite_sets_with_special_divisibility_l2779_277932

theorem infinite_sets_with_special_divisibility :
  ∃ f : ℕ → Fin 1983 → ℕ,
    (∀ k : ℕ, ∀ i j : Fin 1983, i < j → f k i < f k j) ∧
    (∀ k : ℕ, ∀ i : Fin 1983, ∃ a : ℕ, a > 1 ∧ (a ^ 1983 ∣ f k i)) ∧
    (∀ k : ℕ, ∀ i : Fin 1983, i.val < 1982 → f k i.succ = f k i + 1) :=
by sorry

end infinite_sets_with_special_divisibility_l2779_277932


namespace students_not_enrolled_l2779_277905

theorem students_not_enrolled (total : ℕ) (french : ℕ) (german : ℕ) (both : ℕ) 
  (h1 : total = 79)
  (h2 : french = 41)
  (h3 : german = 22)
  (h4 : both = 9) :
  total - (french + german - both) = 25 := by
  sorry

end students_not_enrolled_l2779_277905


namespace new_average_age_l2779_277900

theorem new_average_age (initial_people : ℕ) (initial_avg : ℚ) (leaving_age : ℕ) (entering_age : ℕ) :
  initial_people = 7 →
  initial_avg = 28 →
  leaving_age = 22 →
  entering_age = 30 →
  round ((initial_people * initial_avg - leaving_age + entering_age) / initial_people) = 29 := by
  sorry

end new_average_age_l2779_277900


namespace imaginary_part_of_complex_fraction_l2779_277907

theorem imaginary_part_of_complex_fraction (z : ℂ) : z = (Complex.I : ℂ) / (1 + 2 * Complex.I) → Complex.im z = 1/5 := by
  sorry

end imaginary_part_of_complex_fraction_l2779_277907


namespace zoo_bus_distribution_l2779_277998

theorem zoo_bus_distribution (total_people : ℕ) (num_buses : ℕ) (h1 : total_people = 219) (h2 : num_buses = 3) :
  total_people % num_buses = 0 →
  total_people / num_buses = 73 := by
sorry

end zoo_bus_distribution_l2779_277998


namespace peach_baskets_l2779_277983

theorem peach_baskets (red_per_basket : ℕ) (total_red : ℕ) (h1 : red_per_basket = 16) (h2 : total_red = 96) :
  total_red / red_per_basket = 6 := by
  sorry

end peach_baskets_l2779_277983


namespace expected_weekly_rainfall_is_28_7_l2779_277923

/-- Weather forecast for a single day -/
structure DailyForecast where
  prob_sun : ℝ
  prob_light_rain : ℝ
  prob_heavy_rain : ℝ
  light_rain_amount : ℝ
  heavy_rain_amount : ℝ

/-- Calculate the expected rainfall for a single day -/
def expected_daily_rainfall (f : DailyForecast) : ℝ :=
  f.prob_light_rain * f.light_rain_amount + f.prob_heavy_rain * f.heavy_rain_amount

/-- Calculate the expected total rainfall for a week -/
def expected_weekly_rainfall (f : DailyForecast) : ℝ :=
  7 * expected_daily_rainfall f

/-- The weather forecast for each day of the week -/
def weekly_forecast : DailyForecast :=
  { prob_sun := 0.3
  , prob_light_rain := 0.3
  , prob_heavy_rain := 0.4
  , light_rain_amount := 3
  , heavy_rain_amount := 8 }

theorem expected_weekly_rainfall_is_28_7 :
  expected_weekly_rainfall weekly_forecast = 28.7 := by
  sorry

end expected_weekly_rainfall_is_28_7_l2779_277923


namespace annual_interest_calculation_l2779_277915

def total_amount : ℝ := 3000
def first_part : ℝ := 299.99999999999994
def second_part : ℝ := total_amount - first_part
def interest_rate1 : ℝ := 0.03
def interest_rate2 : ℝ := 0.05

theorem annual_interest_calculation :
  let interest1 := first_part * interest_rate1
  let interest2 := second_part * interest_rate2
  interest1 + interest2 = 144 := by sorry

end annual_interest_calculation_l2779_277915


namespace correct_initial_lives_l2779_277944

/-- The number of lives a player starts with in a game -/
def initial_lives : ℕ := 2

/-- The number of extra lives gained in the first level -/
def extra_lives_level1 : ℕ := 6

/-- The number of extra lives gained in the second level -/
def extra_lives_level2 : ℕ := 11

/-- The total number of lives after two levels -/
def total_lives : ℕ := 19

theorem correct_initial_lives :
  initial_lives + extra_lives_level1 + extra_lives_level2 = total_lives :=
by sorry

end correct_initial_lives_l2779_277944


namespace log_expression_arbitrarily_small_l2779_277920

theorem log_expression_arbitrarily_small :
  ∀ ε > 0, ∃ x > (2/3 : ℝ), Real.log (x^2 + 3) - 2 * Real.log x < ε :=
by sorry

end log_expression_arbitrarily_small_l2779_277920


namespace rd_sum_formula_count_rd_sum_3883_is_18_count_rd_sum_equal_is_143_l2779_277957

/-- Represents a four-digit positive integer ABCD where A and D are non-zero digits -/
structure FourDigitNumber where
  a : Nat
  b : Nat
  c : Nat
  d : Nat
  h1 : a > 0 ∧ a < 10
  h2 : b < 10
  h3 : c < 10
  h4 : d > 0 ∧ d < 10

/-- Calculates the reverse of a four-digit number -/
def reverse (n : FourDigitNumber) : Nat :=
  1000 * n.d + 100 * n.c + 10 * n.b + n.a

/-- Calculates the RD sum of a four-digit number -/
def rdSum (n : FourDigitNumber) : Nat :=
  (1000 * n.a + 100 * n.b + 10 * n.c + n.d) + reverse n

/-- Theorem: The RD sum of ABCD is equal to 1001(A + D) + 110(B + C) -/
theorem rd_sum_formula (n : FourDigitNumber) :
  rdSum n = 1001 * (n.a + n.d) + 110 * (n.b + n.c) := by
  sorry

/-- The number of four-digit integers whose RD sum is 3883 -/
def count_rd_sum_3883 : Nat := 18

/-- Theorem: The number of four-digit integers whose RD sum is 3883 is 18 -/
theorem count_rd_sum_3883_is_18 :
  count_rd_sum_3883 = 18 := by
  sorry

/-- The number of four-digit integers that are equal to the RD sum of a four-digit integer -/
def count_rd_sum_equal : Nat := 143

/-- Theorem: The number of four-digit integers that are equal to the RD sum of a four-digit integer is 143 -/
theorem count_rd_sum_equal_is_143 :
  count_rd_sum_equal = 143 := by
  sorry

end rd_sum_formula_count_rd_sum_3883_is_18_count_rd_sum_equal_is_143_l2779_277957


namespace exists_closer_vertex_l2779_277981

-- Define a convex polygon
def ConvexPolygon (vertices : Set (ℝ × ℝ)) : Prop := sorry

-- Define a point being inside a polygon
def InsidePolygon (p : ℝ × ℝ) (polygon : Set (ℝ × ℝ)) : Prop := sorry

-- Define the distance between two points
def distance (p q : ℝ × ℝ) : ℝ := sorry

theorem exists_closer_vertex 
  (vertices : Set (ℝ × ℝ)) 
  (P Q : ℝ × ℝ) 
  (h_convex : ConvexPolygon vertices)
  (h_P_inside : InsidePolygon P vertices)
  (h_Q_inside : InsidePolygon Q vertices) :
  ∃ V ∈ vertices, distance V Q < distance V P := by
  sorry

end exists_closer_vertex_l2779_277981


namespace two_talents_count_l2779_277978

def num_students : ℕ := 120

def num_cant_sing : ℕ := 50
def num_cant_dance : ℕ := 75
def num_cant_act : ℕ := 35

def num_can_sing : ℕ := num_students - num_cant_sing
def num_can_dance : ℕ := num_students - num_cant_dance
def num_can_act : ℕ := num_students - num_cant_act

theorem two_talents_count :
  ∀ (x : ℕ),
    x ≤ num_students →
    (num_can_sing + num_can_dance + num_can_act) - (num_students - x) = 80 + x →
    x = 0 →
    (num_can_sing + num_can_dance + num_can_act) - num_students = 80 :=
by sorry

end two_talents_count_l2779_277978


namespace all_propositions_false_l2779_277958

-- Define the concept of skew lines
def are_skew (l1 l2 : Line3D) : Prop := sorry

-- Define the concept of perpendicular lines
def is_perpendicular (l1 l2 : Line3D) : Prop := sorry

-- Define the concept of intersecting lines
def intersect (l1 l2 : Line3D) : Prop := sorry

-- Define the concept of lines in different planes
def in_different_planes (l1 l2 : Line3D) : Prop := sorry

theorem all_propositions_false :
  (∀ l1 l2 : Line3D, in_different_planes l1 l2 → are_skew l1 l2) = False ∧
  (∃! l : Line3D, ∀ l1 l2 : Line3D, are_skew l1 l2 → is_perpendicular l l1 ∧ is_perpendicular l l2) = False ∧
  (∀ l1 l2 l3 l4 : Line3D, are_skew l1 l2 → intersect l3 l1 → intersect l3 l2 → intersect l4 l1 → intersect l4 l2 → are_skew l3 l4) = False ∧
  (∀ a b c : Line3D, are_skew a b → are_skew b c → are_skew a c) = False :=
by sorry

end all_propositions_false_l2779_277958


namespace total_pencils_l2779_277963

/-- Given that each child has 2 pencils and there are 11 children, 
    prove that the total number of pencils is 22. -/
theorem total_pencils (pencils_per_child : ℕ) (num_children : ℕ) 
  (h1 : pencils_per_child = 2) 
  (h2 : num_children = 11) : 
  pencils_per_child * num_children = 22 := by
sorry

end total_pencils_l2779_277963


namespace buffer_saline_volume_l2779_277942

theorem buffer_saline_volume 
  (total_buffer : ℚ) 
  (solution_b_volume : ℚ) 
  (saline_volume : ℚ) 
  (initial_mixture_volume : ℚ) :
  total_buffer = 3/2 →
  solution_b_volume = 1/4 →
  saline_volume = 1/6 →
  initial_mixture_volume = 5/12 →
  solution_b_volume + saline_volume = initial_mixture_volume →
  (total_buffer * (saline_volume / initial_mixture_volume) : ℚ) = 3/5 :=
by sorry

end buffer_saline_volume_l2779_277942


namespace surfers_problem_l2779_277931

/-- The number of surfers on the beach with fewer surfers -/
def x : ℕ := sorry

/-- The number of surfers on Malibu beach -/
def y : ℕ := sorry

/-- The total number of surfers on both beaches -/
def total : ℕ := 60

theorem surfers_problem :
  (y = 2 * x) ∧ (x + y = total) → x = 20 := by sorry

end surfers_problem_l2779_277931


namespace total_removed_volume_l2779_277975

/-- The edge length of the cube -/
def cube_edge : ℝ := 2

/-- The number of sides in the resulting polygon on each face after slicing -/
def hexadecagon_sides : ℕ := 16

/-- The volume of a single removed tetrahedron -/
noncomputable def tetrahedron_volume : ℝ := 
  let y := 2 * (Real.sqrt 2 - 1)
  let height := 3 - 2 * Real.sqrt 2
  let base_area := (1 / 2) * ((2 - Real.sqrt 2) ^ 2)
  (1 / 3) * base_area * height

/-- The number of corners in a cube -/
def cube_corners : ℕ := 8

/-- Theorem stating the total volume of removed tetrahedra -/
theorem total_removed_volume : 
  cube_corners * tetrahedron_volume = -64 * Real.sqrt 2 := by sorry

end total_removed_volume_l2779_277975


namespace refrigerator_installation_cost_l2779_277989

/-- Calculates the installation cost for a refrigerator sale --/
theorem refrigerator_installation_cost 
  (purchased_price : ℚ) 
  (discount_rate : ℚ) 
  (transport_cost : ℚ) 
  (profit_rate : ℚ) 
  (final_selling_price : ℚ) 
  (h1 : purchased_price = 12500)
  (h2 : discount_rate = 1/5)
  (h3 : transport_cost = 125)
  (h4 : profit_rate = 4/25)
  (h5 : final_selling_price = 18560) : 
  ∃ (installation_cost : ℚ), installation_cost = 310 := by
  sorry


end refrigerator_installation_cost_l2779_277989


namespace pen_measurement_properties_l2779_277970

def measured_length : Float := 0.06250

-- Function to count significant figures
def count_significant_figures (x : Float) : Nat :=
  sorry

-- Function to determine the place of accuracy
def place_of_accuracy (x : Float) : String :=
  sorry

theorem pen_measurement_properties :
  (count_significant_figures measured_length = 4) ∧
  (place_of_accuracy measured_length = "hundred-thousandth") :=
by sorry

end pen_measurement_properties_l2779_277970


namespace monotonic_function_theorem_l2779_277976

/-- A monotonic function is either non-increasing or non-decreasing --/
def Monotonic (f : ℝ → ℝ) : Prop :=
  (∀ x y, x ≤ y → f x ≤ f y) ∨ (∀ x y, x ≤ y → f x ≥ f y)

/-- The main theorem --/
theorem monotonic_function_theorem (f : ℝ → ℝ) (hf : Monotonic f)
    (h : ∀ x y : ℝ, f (f x - y) + f (x + y) = 0) :
    (∀ x : ℝ, f x = 0) ∨ (∀ x : ℝ, f x = -x) := by
  sorry


end monotonic_function_theorem_l2779_277976


namespace total_apples_picked_l2779_277972

theorem total_apples_picked (benny_apples dan_apples : ℕ) : 
  benny_apples = 2 → dan_apples = 9 → benny_apples + dan_apples = 11 := by
  sorry

end total_apples_picked_l2779_277972


namespace repeating_decimal_to_fraction_l2779_277991

theorem repeating_decimal_to_fraction : ∃ (x : ℚ), x = 4/11 ∧ (∀ (n : ℕ), x = (36 * (100^n - 1)) / (99 * 100^n)) := by
  sorry

end repeating_decimal_to_fraction_l2779_277991


namespace order_of_abc_l2779_277993

theorem order_of_abc : 
  let a : ℝ := Real.rpow 0.9 (1/3)
  let b : ℝ := Real.rpow (1/3) 0.9
  let c : ℝ := (1/2) * (Real.log 9 / Real.log 27)
  c < b ∧ b < a := by sorry

end order_of_abc_l2779_277993


namespace value_of_x_l2779_277937

theorem value_of_x : (2023^2 - 2023) / 2023 = 2022 := by
  sorry

end value_of_x_l2779_277937


namespace greenfield_high_school_teachers_l2779_277946

/-- The number of students at Greenfield High School -/
def num_students : ℕ := 900

/-- The number of classes each student takes per day -/
def classes_per_student : ℕ := 6

/-- The number of classes each teacher teaches -/
def classes_per_teacher : ℕ := 5

/-- The number of students in each class -/
def students_per_class : ℕ := 25

/-- The number of teachers in each class -/
def teachers_per_class : ℕ := 1

/-- The number of teachers at Greenfield High School -/
def num_teachers : ℕ := 44

theorem greenfield_high_school_teachers :
  (num_students * classes_per_student) / students_per_class / classes_per_teacher = num_teachers := by
  sorry

end greenfield_high_school_teachers_l2779_277946


namespace max_abc_constrained_polynomial_l2779_277997

/-- A polynomial of degree 4 with specific constraints on its coefficients. -/
structure ConstrainedPolynomial where
  a : ℝ
  b : ℝ
  c : ℝ
  pos_a : 0 < a
  pos_b : 0 < b
  pos_c : 0 < c
  bound_a : a < 3
  bound_b : b < 3
  bound_c : c < 3
  p : ℝ → ℝ := λ x => x^4 + a*x^3 + b*x^2 + c*x + 1
  no_real_roots : ∀ x : ℝ, p x ≠ 0

/-- The maximum value of abc for polynomials satisfying the given constraints is 18.75. -/
theorem max_abc_constrained_polynomial (poly : ConstrainedPolynomial) :
  ∃ M : ℝ, M = 18.75 ∧ poly.a * poly.b * poly.c ≤ M ∧
  ∀ ε > 0, ∃ poly' : ConstrainedPolynomial, poly'.a * poly'.b * poly'.c > M - ε :=
sorry

end max_abc_constrained_polynomial_l2779_277997


namespace johns_donation_is_100_l2779_277908

/-- The size of John's donation to a charity fund --/
def johns_donation (initial_average : ℝ) (num_initial_contributions : ℕ) (new_average : ℝ) : ℝ :=
  (num_initial_contributions + 1) * new_average - num_initial_contributions * initial_average

/-- Theorem stating that John's donation is $100 given the problem conditions --/
theorem johns_donation_is_100 :
  johns_donation 50 1 75 = 100 := by
  sorry

end johns_donation_is_100_l2779_277908


namespace max_games_24_l2779_277929

/-- Represents a chess tournament with 8 players -/
structure ChessTournament where
  players : Finset (Fin 8)
  games : Finset (Fin 8 × Fin 8)
  hplayers : players.card = 8
  hgames : ∀ (i j : Fin 8), (i, j) ∈ games → i ≠ j
  hunique : ∀ (i j : Fin 8), (i, j) ∈ games → (j, i) ∉ games

/-- No five players all play each other -/
def noFiveAllPlay (t : ChessTournament) : Prop :=
  ∀ (s : Finset (Fin 8)), s.card = 5 →
    ∃ (i j : Fin 8), i ∈ s ∧ j ∈ s ∧ (i, j) ∉ t.games ∧ (j, i) ∉ t.games

/-- The main theorem: maximum number of games is 24 -/
theorem max_games_24 (t : ChessTournament) (h : noFiveAllPlay t) :
  t.games.card ≤ 24 :=
sorry

end max_games_24_l2779_277929


namespace quadratic_c_value_l2779_277971

/-- The quadratic function f(x) = -x^2 + cx + 8 is positive only on the open interval (2,6) -/
def quadratic_positive_on_interval (c : ℝ) : Prop :=
  ∀ x : ℝ, (-x^2 + c*x + 8 > 0) ↔ (2 < x ∧ x < 6)

/-- The value of c for which the quadratic function is positive only on (2,6) is 8 -/
theorem quadratic_c_value : ∃! c : ℝ, quadratic_positive_on_interval c ∧ c = 8 := by
  sorry

end quadratic_c_value_l2779_277971


namespace dimes_spent_l2779_277990

/-- Given Joan's initial and remaining dimes, calculate the number of dimes spent. -/
theorem dimes_spent (initial : ℕ) (remaining : ℕ) (h : remaining ≤ initial) :
  initial - remaining = initial - remaining :=
by sorry

end dimes_spent_l2779_277990


namespace ab_value_l2779_277921

theorem ab_value (a b : ℝ) : (a - b - 3) * (a - b + 3) = 40 → (a - b = 7 ∨ a - b = -7) := by
  sorry

end ab_value_l2779_277921


namespace compute_expression_l2779_277956

theorem compute_expression : (3 + 7)^2 + Real.sqrt (3^2 + 7^2) = 100 + Real.sqrt 58 := by
  sorry

end compute_expression_l2779_277956


namespace binomial_coefficient_19_11_l2779_277980

theorem binomial_coefficient_19_11 :
  (Nat.choose 19 11 = 82654) ∧ (Nat.choose 17 9 = 24310) ∧ (Nat.choose 17 7 = 19448) → 
  Nat.choose 19 11 = 82654 := by
  sorry

end binomial_coefficient_19_11_l2779_277980


namespace z_purely_imaginary_iff_m_eq_neg_one_l2779_277941

/-- The imaginary unit -/
def i : ℂ := Complex.I

/-- A complex number z defined as i(5-i) + m -/
def z (m : ℝ) : ℂ := i * (5 - i) + m

/-- Theorem stating that z is purely imaginary if and only if m = -1 -/
theorem z_purely_imaginary_iff_m_eq_neg_one (m : ℝ) :
  z m = Complex.I * (z m).im ↔ m = -1 := by sorry

end z_purely_imaginary_iff_m_eq_neg_one_l2779_277941


namespace total_amount_correct_l2779_277992

/-- The amount of money Mrs. Hilt needs to share -/
def total_amount : ℝ := 3.75

/-- The number of people sharing the money -/
def number_of_people : ℕ := 3

/-- The amount each person receives -/
def amount_per_person : ℝ := 1.25

/-- Theorem stating that the total amount is correct given the conditions -/
theorem total_amount_correct : 
  total_amount = (number_of_people : ℝ) * amount_per_person :=
by sorry

end total_amount_correct_l2779_277992


namespace two_numbers_problem_l2779_277954

theorem two_numbers_problem :
  ∃ (x y : ℝ), y = 33 ∧ x + y = 51 ∧ y = 2 * x - 3 :=
by sorry

end two_numbers_problem_l2779_277954


namespace frog_jump_probability_l2779_277945

-- Define the probability function
noncomputable def Q (x y : ℝ) : ℝ := sorry

-- Define the boundary conditions
axiom vertical_boundary : ∀ y, 0 ≤ y ∧ y ≤ 6 → Q 0 y = 1 ∧ Q 6 y = 1
axiom horizontal_boundary : ∀ x, 0 ≤ x ∧ x ≤ 6 → Q x 0 = 0 ∧ Q x 6 = 0

-- Define the recursive relation
axiom recursive_relation : 
  Q 2 3 = (1/4) * Q 1 3 + (1/4) * Q 3 3 + (1/4) * Q 2 2 + (1/4) * Q 2 4

-- Theorem to prove
theorem frog_jump_probability : Q 2 3 = 5/8 := by sorry

end frog_jump_probability_l2779_277945


namespace factoring_quadratic_l2779_277965

theorem factoring_quadratic (a : ℝ) : a^2 - 4*a + 3 = (a - 1) * (a - 3) := by
  sorry

#check factoring_quadratic

end factoring_quadratic_l2779_277965


namespace line_tangent_to_ellipse_l2779_277986

/-- 
Theorem: If a line y = kx + 2 is tangent to the ellipse x^2/2 + 2y^2 = 2, 
then k^2 = 3/4.
-/
theorem line_tangent_to_ellipse (k : ℝ) : 
  (∀ x y : ℝ, y = k * x + 2 → x^2 / 2 + 2 * y^2 = 2) →
  (∃! p : ℝ × ℝ, p.1^2 / 2 + 2 * p.2^2 = 2 ∧ p.2 = k * p.1 + 2) →
  k^2 = 3/4 := by
sorry

end line_tangent_to_ellipse_l2779_277986


namespace new_person_weight_l2779_277953

/-- Given 8 persons, if replacing one person weighing 50 kg with a new person 
    increases the average weight by 2.5 kg, then the weight of the new person is 70 kg. -/
theorem new_person_weight (initial_count : Nat) (weight_increase : ℝ) (replaced_weight : ℝ) :
  initial_count = 8 →
  weight_increase = 2.5 →
  replaced_weight = 50 →
  ∃ (new_weight : ℝ),
    new_weight = initial_count * weight_increase + replaced_weight ∧
    new_weight = 70 := by
  sorry

end new_person_weight_l2779_277953


namespace stock_percentage_example_l2779_277913

/-- The percentage of stock that yields a given income from a given investment --/
def stock_percentage (income : ℚ) (investment : ℚ) : ℚ :=
  (income * 100) / investment

/-- Theorem: The stock percentage for an income of 15000 and investment of 37500 is 40% --/
theorem stock_percentage_example : stock_percentage 15000 37500 = 40 := by
  sorry

end stock_percentage_example_l2779_277913


namespace johns_toy_store_spending_l2779_277940

/-- Proves that the fraction of John's remaining allowance spent at the toy store is 1/3 -/
theorem johns_toy_store_spending (
  total_allowance : ℚ)
  (arcade_fraction : ℚ)
  (candy_store_amount : ℚ)
  (h1 : total_allowance = 33/10)
  (h2 : arcade_fraction = 3/5)
  (h3 : candy_store_amount = 88/100) :
  let remaining_after_arcade := total_allowance - arcade_fraction * total_allowance
  let toy_store_amount := remaining_after_arcade - candy_store_amount
  toy_store_amount / remaining_after_arcade = 1/3 := by
sorry


end johns_toy_store_spending_l2779_277940


namespace total_capsules_sold_l2779_277988

def weekly_earnings_100mg : ℕ := 80
def weekly_earnings_500mg : ℕ := 60
def cost_per_capsule_100mg : ℕ := 5
def cost_per_capsule_500mg : ℕ := 2

def capsules_100mg_per_week : ℕ := weekly_earnings_100mg / cost_per_capsule_100mg
def capsules_500mg_per_week : ℕ := weekly_earnings_500mg / cost_per_capsule_500mg

def total_capsules_2_weeks : ℕ := 2 * (capsules_100mg_per_week + capsules_500mg_per_week)

theorem total_capsules_sold :
  total_capsules_2_weeks = 92 :=
by sorry

end total_capsules_sold_l2779_277988


namespace sun_radius_scientific_notation_l2779_277995

theorem sun_radius_scientific_notation : 369000 = 3.69 * (10 ^ 5) := by
  sorry

end sun_radius_scientific_notation_l2779_277995


namespace a5_value_l2779_277961

def arithmetic_sequence (a : ℕ → ℤ) : Prop :=
  ∃ d : ℤ, ∀ n : ℕ, a (n + 1) = a n + d

theorem a5_value (a : ℕ → ℤ) (h_arith : arithmetic_sequence a) 
  (h3 : a 3 = -5) (h7 : a 7 = -1) : a 5 = -3 := by
  sorry

end a5_value_l2779_277961


namespace marble_ratio_l2779_277969

theorem marble_ratio (selma_marbles merill_marbles elliot_marbles : ℕ) : 
  selma_marbles = 50 →
  merill_marbles = 30 →
  merill_marbles + elliot_marbles = selma_marbles - 5 →
  merill_marbles / elliot_marbles = 2 := by
  sorry

end marble_ratio_l2779_277969


namespace book_exchange_count_l2779_277950

/-- Represents a book exchange in a book club --/
structure BookExchange where
  friends : Finset (Fin 6)
  give : Fin 6 → Fin 6
  receive : Fin 6 → Fin 6

/-- Conditions for a valid book exchange --/
def ValidExchange (e : BookExchange) : Prop :=
  (∀ i, i ∈ e.friends) ∧
  (∀ i, e.give i ≠ i) ∧
  (∀ i, e.receive i ≠ i) ∧
  (∀ i, e.give i ≠ e.receive i) ∧
  (∀ i j, e.give i = e.give j → i = j) ∧
  (∀ i j, e.receive i = e.receive j → i = j)

/-- The number of valid book exchanges --/
def NumberOfExchanges : ℕ := sorry

/-- Theorem stating that the number of valid book exchanges is 160 --/
theorem book_exchange_count : NumberOfExchanges = 160 := by sorry

end book_exchange_count_l2779_277950


namespace plate_count_l2779_277901

theorem plate_count (n : ℕ) 
  (h1 : 500 < n ∧ n < 600)
  (h2 : n % 10 = 7)
  (h3 : n % 12 = 7) : 
  n = 547 := by
sorry

end plate_count_l2779_277901


namespace preston_order_calculation_l2779_277902

/-- The total amount Preston received from Abra Company's order -/
def total_received (sandwich_price : ℚ) (delivery_fee : ℚ) (num_sandwiches : ℕ) (tip_percentage : ℚ) : ℚ :=
  let subtotal := sandwich_price * num_sandwiches + delivery_fee
  subtotal + subtotal * tip_percentage

/-- Preston's sandwich shop order calculation -/
theorem preston_order_calculation :
  total_received 5 20 18 (1/10) = 121 := by
  sorry

end preston_order_calculation_l2779_277902


namespace maximize_x_cube_y_fourth_l2779_277928

/-- 
Given positive real numbers x and y such that x + y = 50,
x^3 * y^4 is maximized when x = 150/7 and y = 200/7.
-/
theorem maximize_x_cube_y_fourth (x y : ℝ) 
  (pos_x : 0 < x) (pos_y : 0 < y) (sum_xy : x + y = 50) :
  x^3 * y^4 ≤ (150/7)^3 * (200/7)^4 ∧ 
  x^3 * y^4 = (150/7)^3 * (200/7)^4 ↔ x = 150/7 ∧ y = 200/7 := by
  sorry

#check maximize_x_cube_y_fourth

end maximize_x_cube_y_fourth_l2779_277928


namespace difference_of_squares_75_45_l2779_277916

theorem difference_of_squares_75_45 : 75^2 - 45^2 = 3600 := by
  sorry

end difference_of_squares_75_45_l2779_277916


namespace min_value_theorem_l2779_277984

theorem min_value_theorem (x y : ℝ) (h : x^2 + y^2 = 1) :
  ∃ (m : ℝ), m = 1 - Real.sqrt 2 ∧ ∀ z, z = (2 * x * y) / (x + y - 1) → m ≤ z :=
sorry

end min_value_theorem_l2779_277984


namespace circle_tangency_l2779_277985

/-- Circle C with equation x^2 + y^2 - 2x - 4y + m = 0 -/
def circle_C (m : ℝ) : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.1^2 + p.2^2 - 2*p.1 - 4*p.2 + m = 0}

/-- Circle D with equation (x+2)^2 + (y+2)^2 = 1 -/
def circle_D : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | (p.1 + 2)^2 + (p.2 + 2)^2 = 1}

/-- The number of common tangents between two circles -/
def common_tangents (C D : Set (ℝ × ℝ)) : ℕ := sorry

theorem circle_tangency (m : ℝ) :
  common_tangents (circle_C m) circle_D = 3 → m = -11 := by sorry

end circle_tangency_l2779_277985


namespace inequality_on_unit_circle_l2779_277910

theorem inequality_on_unit_circle (a b c d : ℝ) (x₁ y₁ x₂ y₂ x₃ y₃ x₄ y₄ : ℝ) 
  (ha : a > 0) (hb : b > 0) (hc : c > 0) (hd : d > 0)
  (hab : a * b + c * d = 1)
  (h1 : x₁^2 + y₁^2 = 1) (h2 : x₂^2 + y₂^2 = 1) 
  (h3 : x₃^2 + y₃^2 = 1) (h4 : x₄^2 + y₄^2 = 1) :
  (a * y₁ + b * y₂ + c * y₃ + d * y₄)^2 + (a * x₄ + b * x₃ + c * x₂ + d * x₁)^2 ≤ 
  2 * ((a^2 + b^2) / (a * b) + (c^2 + d^2) / (c * d)) := by
sorry

end inequality_on_unit_circle_l2779_277910


namespace locus_of_point_P_l2779_277911

-- Define the 2D plane
variable (V : Type*) [NormedAddCommGroup V] [InnerProductSpace ℝ V] [FiniteDimensional ℝ V]
variable [Fact (finrank ℝ V = 2)]

-- Define points A, B, and P
variable (A B P : V)

-- Define the distance function
def dist (x y : V) : ℝ := ‖x - y‖

-- Theorem statement
theorem locus_of_point_P (h1 : dist A B = 3) (h2 : dist A P + dist B P = 3) :
  ∃ t : ℝ, 0 ≤ t ∧ t ≤ 1 ∧ P = (1 - t) • A + t • B :=
sorry

end locus_of_point_P_l2779_277911


namespace arithmetic_sequence_sum_l2779_277943

theorem arithmetic_sequence_sum (n : ℕ) (a₁ : ℤ) : n > 1 → (∃ k : ℕ, n * k = 2000) →
  (n * (2 * a₁ + (n - 1) * 2)) / 2 = 2000 ↔ n ∣ 2000 :=
by sorry

end arithmetic_sequence_sum_l2779_277943


namespace min_ferries_required_l2779_277903

def ferry_capacity : ℕ := 45
def people_to_transport : ℕ := 523

theorem min_ferries_required : 
  ∃ (n : ℕ), n * ferry_capacity ≥ people_to_transport ∧ 
  ∀ (m : ℕ), m * ferry_capacity ≥ people_to_transport → m ≥ n :=
by
  -- The proof goes here
  sorry

end min_ferries_required_l2779_277903


namespace number_and_square_difference_l2779_277926

theorem number_and_square_difference (N : ℝ) : N^2 - N = 12 ↔ N = 4 ∨ N = -3 := by
  sorry

end number_and_square_difference_l2779_277926

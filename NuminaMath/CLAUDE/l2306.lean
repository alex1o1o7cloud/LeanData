import Mathlib

namespace NUMINAMATH_CALUDE_plum_jelly_sales_l2306_230629

/-- Represents the number of jars sold for each type of jelly -/
structure JellySales where
  grape : ℕ
  strawberry : ℕ
  raspberry : ℕ
  plum : ℕ

/-- Defines the conditions for jelly sales -/
def validJellySales (sales : JellySales) : Prop :=
  sales.grape = 2 * sales.strawberry ∧
  sales.raspberry = 2 * sales.plum ∧
  sales.raspberry = sales.grape / 3 ∧
  sales.strawberry = 18

/-- Theorem stating that given the conditions, 6 jars of plum jelly were sold -/
theorem plum_jelly_sales (sales : JellySales) (h : validJellySales sales) : sales.plum = 6 := by
  sorry

end NUMINAMATH_CALUDE_plum_jelly_sales_l2306_230629


namespace NUMINAMATH_CALUDE_mel_age_is_21_l2306_230621

/-- Katherine's age in years -/
def katherine_age : ℕ := 24

/-- The age difference between Katherine and Mel in years -/
def age_difference : ℕ := 3

/-- Mel's age in years -/
def mel_age : ℕ := katherine_age - age_difference

theorem mel_age_is_21 : mel_age = 21 := by
  sorry

end NUMINAMATH_CALUDE_mel_age_is_21_l2306_230621


namespace NUMINAMATH_CALUDE_min_value_expression_l2306_230663

theorem min_value_expression (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) 
  (h : x * y + z^2 = 8) : 
  (x + y) / z + (y + z) / x^2 + (z + x) / y^2 ≥ 4 := by
  sorry

end NUMINAMATH_CALUDE_min_value_expression_l2306_230663


namespace NUMINAMATH_CALUDE_eliza_almonds_l2306_230631

theorem eliza_almonds (eliza_almonds daniel_almonds : ℕ) : 
  eliza_almonds = daniel_almonds + 8 →
  daniel_almonds = eliza_almonds / 3 →
  eliza_almonds = 12 := by
sorry

end NUMINAMATH_CALUDE_eliza_almonds_l2306_230631


namespace NUMINAMATH_CALUDE_sum_of_numbers_l2306_230659

/-- Represents the numbers of various individuals in a mathematical problem. -/
structure Numbers where
  k : ℕ
  joyce : ℕ
  xavier : ℕ
  coraline : ℕ
  jayden : ℕ
  mickey : ℕ
  yvonne : ℕ
  natalie : ℕ

/-- The conditions of the problem are satisfied. -/
def satisfies_conditions (n : Numbers) : Prop :=
  n.k > 1 ∧
  n.joyce = 5 * n.k ∧
  n.xavier = 4 * n.joyce ∧
  n.coraline = n.xavier + 50 ∧
  n.jayden = n.coraline - 40 ∧
  n.mickey = n.jayden + 20 ∧
  n.yvonne = (n.xavier + n.joyce) * n.k ∧
  n.natalie = (n.yvonne - n.coraline) / 2

/-- The theorem to be proved. -/
theorem sum_of_numbers (n : Numbers) 
  (h : satisfies_conditions n) : 
  n.joyce + n.xavier + n.coraline + n.jayden + n.mickey + n.yvonne + n.natalie = 365 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_numbers_l2306_230659


namespace NUMINAMATH_CALUDE_sqrt_one_minus_sqrt_two_squared_l2306_230678

theorem sqrt_one_minus_sqrt_two_squared (h : 1 < Real.sqrt 2) :
  Real.sqrt ((1 - Real.sqrt 2) ^ 2) = Real.sqrt 2 - 1 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_one_minus_sqrt_two_squared_l2306_230678


namespace NUMINAMATH_CALUDE_johns_growth_per_month_l2306_230641

/-- Proves that John's growth per month is 2 inches given his original height, new height, and growth period. -/
theorem johns_growth_per_month 
  (original_height : ℕ) 
  (new_height_feet : ℕ) 
  (growth_period : ℕ) 
  (h1 : original_height = 66)
  (h2 : new_height_feet = 6)
  (h3 : growth_period = 3) :
  (new_height_feet * 12 - original_height) / growth_period = 2 := by
  sorry

#check johns_growth_per_month

end NUMINAMATH_CALUDE_johns_growth_per_month_l2306_230641


namespace NUMINAMATH_CALUDE_iphone_price_decrease_l2306_230680

def initial_price : ℝ := 1000
def first_month_decrease : ℝ := 0.1
def final_price : ℝ := 720

theorem iphone_price_decrease : 
  let price_after_first_month := initial_price * (1 - first_month_decrease)
  let second_month_decrease := (price_after_first_month - final_price) / price_after_first_month
  second_month_decrease = 0.2 := by
  sorry

end NUMINAMATH_CALUDE_iphone_price_decrease_l2306_230680


namespace NUMINAMATH_CALUDE_fraction_simplification_l2306_230656

theorem fraction_simplification :
  (1 / 4 - 1 / 6) / (1 / 3 - 1 / 4) = 1 := by
  sorry

end NUMINAMATH_CALUDE_fraction_simplification_l2306_230656


namespace NUMINAMATH_CALUDE_derivative_of_sine_at_pi_sixth_l2306_230674

/-- Given f(x) = sin(2x + π/6), prove that f'(π/6) = 0 -/
theorem derivative_of_sine_at_pi_sixth (f : ℝ → ℝ) (h : ∀ x, f x = Real.sin (2 * x + π / 6)) :
  deriv f (π / 6) = 0 := by
  sorry

end NUMINAMATH_CALUDE_derivative_of_sine_at_pi_sixth_l2306_230674


namespace NUMINAMATH_CALUDE_minor_arc_probability_l2306_230630

/-- The probability that the length of the minor arc is less than 1 on a circle
    with circumference 3, given a fixed point A and a randomly selected point B. -/
theorem minor_arc_probability (circle_circumference : ℝ) (arc_length : ℝ) :
  circle_circumference = 3 →
  arc_length = 1 →
  (2 * arc_length) / circle_circumference = 2/3 := by
  sorry

end NUMINAMATH_CALUDE_minor_arc_probability_l2306_230630


namespace NUMINAMATH_CALUDE_walking_speed_calculation_l2306_230668

/-- Given a person who jogs and walks, this theorem proves their walking speed. -/
theorem walking_speed_calculation 
  (jog_speed : ℝ) 
  (jog_distance : ℝ) 
  (total_time : ℝ) 
  (h1 : jog_speed = 2) 
  (h2 : jog_distance = 3) 
  (h3 : total_time = 3) : 
  jog_distance / (total_time - jog_distance / jog_speed) = 2 := by
  sorry

end NUMINAMATH_CALUDE_walking_speed_calculation_l2306_230668


namespace NUMINAMATH_CALUDE_sequence_roots_theorem_l2306_230652

theorem sequence_roots_theorem (b c : ℕ → ℝ) : 
  (∀ n : ℕ, n ≥ 1 → b n ≤ c n) → 
  (∀ n : ℕ, n ≥ 1 → (b (n + 1))^2 + (b n) * (b (n + 1)) + (c n) = 0 ∧ 
                     (c (n + 1))^2 + (b n) * (c (n + 1)) + (c n) = 0) →
  (∀ n : ℕ, n ≥ 1 → b n = 0 ∧ c n = 0) :=
by sorry

end NUMINAMATH_CALUDE_sequence_roots_theorem_l2306_230652


namespace NUMINAMATH_CALUDE_f_sum_negative_l2306_230660

/-- The function f satisfying the given conditions -/
noncomputable def f (m : ℝ) (x : ℝ) : ℝ := (m^2 - m - 1) * x^(m^2 + m - 3)

/-- The theorem statement -/
theorem f_sum_negative (m : ℝ) (a b : ℝ) :
  (∀ (x₁ x₂ : ℝ), x₁ > 0 ∧ x₂ > 0 ∧ x₁ ≠ x₂ → (f m x₁ - f m x₂) / (x₁ - x₂) < 0) →
  a < 0 →
  0 < b →
  abs a < abs b →
  f m a + f m b < 0 :=
by sorry

end NUMINAMATH_CALUDE_f_sum_negative_l2306_230660


namespace NUMINAMATH_CALUDE_complement_intersection_theorem_l2306_230687

def I : Set ℕ := Set.univ
def A : Set ℕ := {1,2,3,4,5,6}
def B : Set ℕ := {2,3,5}

theorem complement_intersection_theorem :
  (I \ B) ∩ A = {1,4,6} := by sorry

end NUMINAMATH_CALUDE_complement_intersection_theorem_l2306_230687


namespace NUMINAMATH_CALUDE_marilyn_shared_bottle_caps_l2306_230639

/-- The number of bottle caps Marilyn shared with Nancy -/
def shared_bottle_caps (initial : ℕ) (remaining : ℕ) : ℕ :=
  initial - remaining

theorem marilyn_shared_bottle_caps :
  shared_bottle_caps 51 15 = 36 :=
by sorry

end NUMINAMATH_CALUDE_marilyn_shared_bottle_caps_l2306_230639


namespace NUMINAMATH_CALUDE_mango_rice_flour_cost_l2306_230640

/-- Given the cost relationships between mangos, rice, and flour, 
    prove that the total cost of 4 kg of mangos, 3 kg of rice, and 5 kg of flour is $1027.2 -/
theorem mango_rice_flour_cost 
  (mango_cost rice_cost flour_cost : ℝ) 
  (h1 : 10 * mango_cost = 24 * rice_cost) 
  (h2 : 6 * flour_cost = 2 * rice_cost) 
  (h3 : flour_cost = 24) : 
  4 * mango_cost + 3 * rice_cost + 5 * flour_cost = 1027.2 := by
sorry

end NUMINAMATH_CALUDE_mango_rice_flour_cost_l2306_230640


namespace NUMINAMATH_CALUDE_num_tetrahedrons_in_cube_l2306_230602

/-- The number of vertices in a cube. -/
def cube_vertices : ℕ := 8

/-- The number of vertices required to form a tetrahedron. -/
def tetrahedron_vertices : ℕ := 4

/-- The number of coplanar combinations in a cube (faces and diagonals). -/
def coplanar_combinations : ℕ := 12

/-- Calculates the number of ways to choose k items from n items. -/
def choose (n k : ℕ) : ℕ := (Nat.factorial n) / (Nat.factorial k * Nat.factorial (n - k))

/-- The number of tetrahedrons that can be formed using the vertices of a cube. -/
theorem num_tetrahedrons_in_cube : 
  choose cube_vertices tetrahedron_vertices - coplanar_combinations = 58 := by
  sorry

end NUMINAMATH_CALUDE_num_tetrahedrons_in_cube_l2306_230602


namespace NUMINAMATH_CALUDE_pencils_remaining_l2306_230647

theorem pencils_remaining (x : ℕ) : ℕ :=
  let initial_pencils_per_child : ℕ := 2
  let number_of_children : ℕ := 15
  let total_initial_pencils : ℕ := initial_pencils_per_child * number_of_children
  let pencils_given_away : ℕ := number_of_children * x
  total_initial_pencils - pencils_given_away

#check pencils_remaining

end NUMINAMATH_CALUDE_pencils_remaining_l2306_230647


namespace NUMINAMATH_CALUDE_square_sum_inequality_l2306_230635

theorem square_sum_inequality (a b : ℝ) 
  (h1 : a^3 - b^3 = 2) 
  (h2 : a^5 - b^5 ≥ 4) : 
  a^2 + b^2 ≥ 2 := by
  sorry

end NUMINAMATH_CALUDE_square_sum_inequality_l2306_230635


namespace NUMINAMATH_CALUDE_one_element_set_l2306_230653

def A (k : ℝ) : Set ℝ := {x | k * x^2 - 4 * x + 2 = 0}

theorem one_element_set (k : ℝ) :
  (∃! x, x ∈ A k) → (k = 0 ∧ A k = {1/2}) ∨ (k = 2 ∧ A k = {1}) := by
  sorry

end NUMINAMATH_CALUDE_one_element_set_l2306_230653


namespace NUMINAMATH_CALUDE_cube_surface_area_from_prism_l2306_230691

/-- The surface area of a cube with the same volume as a rectangular prism -/
theorem cube_surface_area_from_prism (l w h : ℝ) (h1 : l = 8) (h2 : w = 2) (h3 : h = 32) :
  let prism_volume := l * w * h
  let cube_edge := (prism_volume) ^ (1/3 : ℝ)
  let cube_surface_area := 6 * cube_edge ^ 2
  cube_surface_area = 384 := by sorry

end NUMINAMATH_CALUDE_cube_surface_area_from_prism_l2306_230691


namespace NUMINAMATH_CALUDE_parabola_point_distance_l2306_230643

theorem parabola_point_distance (x y : ℝ) : 
  x^2 = 4*y →                             -- P is on the parabola x^2 = 4y
  (x^2 + (y - 1)^2 = 4) →                 -- Distance from P to A(0,1) is 2
  y = 1 :=                                -- Distance from P to x-axis is 1
by sorry

end NUMINAMATH_CALUDE_parabola_point_distance_l2306_230643


namespace NUMINAMATH_CALUDE_solution_set_of_inequality_l2306_230638

theorem solution_set_of_inequality (x : ℝ) :
  (x - 3) / (x + 2) < 0 ↔ -2 < x ∧ x < 3 := by sorry

end NUMINAMATH_CALUDE_solution_set_of_inequality_l2306_230638


namespace NUMINAMATH_CALUDE_square_root_equation_l2306_230683

theorem square_root_equation (a : ℝ) : Real.sqrt (a^2) = 3 → a = 3 ∨ a = -3 := by
  sorry

end NUMINAMATH_CALUDE_square_root_equation_l2306_230683


namespace NUMINAMATH_CALUDE_bus_speed_problem_l2306_230699

/-- Given a bus that stops for 15 minutes per hour and has an average speed of 45 km/hr
    including stoppages, its average speed excluding stoppages is 60 km/hr. -/
theorem bus_speed_problem (stop_time : ℝ) (avg_speed_with_stops : ℝ) :
  stop_time = 15 →
  avg_speed_with_stops = 45 →
  ∃ (avg_speed_without_stops : ℝ),
    avg_speed_without_stops = 60 ∧
    avg_speed_with_stops * 1 = avg_speed_without_stops * ((60 - stop_time) / 60) := by
  sorry

end NUMINAMATH_CALUDE_bus_speed_problem_l2306_230699


namespace NUMINAMATH_CALUDE_banana_permutations_l2306_230651

def word_length : ℕ := 6
def a_count : ℕ := 3
def n_count : ℕ := 2

theorem banana_permutations :
  (word_length.factorial) / (a_count.factorial * n_count.factorial) = 60 := by
  sorry

end NUMINAMATH_CALUDE_banana_permutations_l2306_230651


namespace NUMINAMATH_CALUDE_seating_arrangements_l2306_230616

def total_people : ℕ := 12
def people_per_table : ℕ := 6
def num_tables : ℕ := 2

def arrange_people (n : ℕ) (k : ℕ) : ℕ := (n.factorial * 14400) / (k.factorial * k.factorial)

def arrange_couples (n : ℕ) (k : ℕ) : ℕ := (n.factorial * 14400 * 4096) / (k.factorial * k.factorial)

theorem seating_arrangements :
  (arrange_people total_people people_per_table = (total_people.factorial * 14400) / (people_per_table.factorial * people_per_table.factorial)) ∧
  (arrange_couples total_people people_per_table = (total_people.factorial * 14400 * 4096) / (people_per_table.factorial * people_per_table.factorial)) :=
by sorry

end NUMINAMATH_CALUDE_seating_arrangements_l2306_230616


namespace NUMINAMATH_CALUDE_dog_escape_ways_l2306_230600

def base_7_to_10 (d₂ d₁ d₀ : ℕ) : ℕ :=
  d₀ * 7^0 + d₁ * 7^1 + d₂ * 7^2

theorem dog_escape_ways : base_7_to_10 2 3 1 = 120 := by
  sorry

end NUMINAMATH_CALUDE_dog_escape_ways_l2306_230600


namespace NUMINAMATH_CALUDE_expression_evaluation_l2306_230673

theorem expression_evaluation : (3 / 2) * 12 - 3 = 15 := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l2306_230673


namespace NUMINAMATH_CALUDE_prob_four_correct_zero_l2306_230696

/-- Represents the number of people and letters -/
def n : ℕ := 5

/-- The probability of exactly (n-1) people receiving their correct letter
    in a random distribution of n letters to n people -/
def prob_n_minus_one_correct (n : ℕ) : ℝ := 
  if n ≥ 2 then 0 else 1

/-- Theorem stating that the probability of exactly 4 out of 5 people
    receiving their correct letter is 0 -/
theorem prob_four_correct_zero : 
  prob_n_minus_one_correct n = 0 := by sorry

end NUMINAMATH_CALUDE_prob_four_correct_zero_l2306_230696


namespace NUMINAMATH_CALUDE_union_of_A_and_B_l2306_230645

-- Define the sets A and B
def A : Set ℝ := {x | x^2 - 1 < 0}
def B : Set ℝ := {x | ∃ y, y = Real.log x}

-- State the theorem
theorem union_of_A_and_B : A ∪ B = {x | x > -1} := by sorry

end NUMINAMATH_CALUDE_union_of_A_and_B_l2306_230645


namespace NUMINAMATH_CALUDE_intersection_of_A_and_B_l2306_230684

def A : Set ℝ := {x | x^2 - 4*x - 5 < 0}

def B : Set ℝ := {x | |x - 1| > 1}

theorem intersection_of_A_and_B :
  A ∩ B = {x | (-1 < x ∧ x < 0) ∨ (2 < x ∧ x < 5)} :=
by sorry

end NUMINAMATH_CALUDE_intersection_of_A_and_B_l2306_230684


namespace NUMINAMATH_CALUDE_even_function_property_l2306_230624

-- Define the function f
def f : ℝ → ℝ := sorry

-- Define the property of being an even function
def isEven (f : ℝ → ℝ) : Prop :=
  ∀ x, f x = f (-x)

-- Theorem statement
theorem even_function_property 
  (h1 : isEven f) 
  (h2 : ∀ x ∈ Set.Icc (-5 : ℝ) 5, ∃ y, f x = y)
  (h3 : f 3 > f 1) : 
  f (-1) < f 3 := by
  sorry

end NUMINAMATH_CALUDE_even_function_property_l2306_230624


namespace NUMINAMATH_CALUDE_stratified_sampling_result_l2306_230689

/-- Calculates the number of students selected from a class in stratified sampling -/
def stratified_sample (class_size : ℕ) (total_size : ℕ) (sample_size : ℕ) : ℕ :=
  (class_size * sample_size) / total_size

/-- Represents the stratified sampling scenario -/
structure StratifiedSampling where
  class1_size : ℕ
  class2_size : ℕ
  total_sample_size : ℕ

/-- Theorem stating the result of the stratified sampling problem -/
theorem stratified_sampling_result (s : StratifiedSampling) 
  (h1 : s.class1_size = 36)
  (h2 : s.class2_size = 42)
  (h3 : s.total_sample_size = 13) :
  stratified_sample s.class2_size (s.class1_size + s.class2_size) s.total_sample_size = 7 := by
  sorry

#eval stratified_sample 42 (36 + 42) 13

end NUMINAMATH_CALUDE_stratified_sampling_result_l2306_230689


namespace NUMINAMATH_CALUDE_questions_per_exam_l2306_230609

theorem questions_per_exam
  (num_classes : ℕ)
  (students_per_class : ℕ)
  (total_questions : ℕ)
  (h1 : num_classes = 5)
  (h2 : students_per_class = 35)
  (h3 : total_questions = 1750) :
  total_questions / (num_classes * students_per_class) = 10 := by
sorry

end NUMINAMATH_CALUDE_questions_per_exam_l2306_230609


namespace NUMINAMATH_CALUDE_adams_fair_expense_l2306_230607

def fair_problem (initial_tickets : ℕ) (ferris_wheel_cost : ℕ) (roller_coaster_cost : ℕ) 
  (remaining_tickets : ℕ) (ticket_price : ℕ) (snack_price : ℕ) : Prop :=
  let used_tickets := initial_tickets - remaining_tickets
  let ride_cost := used_tickets * ticket_price
  let total_spent := ride_cost + snack_price
  total_spent = 99

theorem adams_fair_expense :
  fair_problem 13 2 3 4 9 18 := by
  sorry

end NUMINAMATH_CALUDE_adams_fair_expense_l2306_230607


namespace NUMINAMATH_CALUDE_trampoline_jumps_l2306_230632

theorem trampoline_jumps (ronald_jumps rupert_extra_jumps : ℕ) 
  (h1 : ronald_jumps = 157)
  (h2 : rupert_extra_jumps = 86) : 
  ronald_jumps + (ronald_jumps + rupert_extra_jumps) = 400 := by
  sorry

end NUMINAMATH_CALUDE_trampoline_jumps_l2306_230632


namespace NUMINAMATH_CALUDE_transistors_2010_l2306_230610

/-- Moore's law: Number of transistors doubles every 18 months -/
def moores_law_doubling_period : ℕ := 18

/-- Number of transistors in 1995 -/
def transistors_1995 : ℕ := 2500000

/-- Calculate the number of transistors after a given number of months -/
def transistors_after (initial_transistors : ℕ) (months : ℕ) : ℕ :=
  initial_transistors * 2^(months / moores_law_doubling_period)

/-- Theorem: Number of transistors in 2010 according to Moore's law -/
theorem transistors_2010 :
  transistors_after transistors_1995 ((2010 - 1995) * 12) = 2560000000 := by
  sorry

end NUMINAMATH_CALUDE_transistors_2010_l2306_230610


namespace NUMINAMATH_CALUDE_cookie_price_l2306_230670

/-- Proves that the price of each cookie is $0.50 given the conditions of the basketball team's sales and purchases. -/
theorem cookie_price (cupcake_count : ℕ) (cupcake_price : ℚ) (cookie_count : ℕ) 
  (basketball_count : ℕ) (basketball_price : ℚ) (drink_count : ℕ) (drink_price : ℚ) :
  cupcake_count = 50 →
  cupcake_price = 2 →
  cookie_count = 40 →
  basketball_count = 2 →
  basketball_price = 40 →
  drink_count = 20 →
  drink_price = 2 →
  ∃ (cookie_price : ℚ),
    cupcake_count * cupcake_price + cookie_count * cookie_price = 
    basketball_count * basketball_price + drink_count * drink_price ∧
    cookie_price = 1/2 := by
  sorry

#check cookie_price

end NUMINAMATH_CALUDE_cookie_price_l2306_230670


namespace NUMINAMATH_CALUDE_village_population_l2306_230679

theorem village_population (population : ℝ) : 
  (0.9 * population = 45000) → population = 50000 := by
  sorry

end NUMINAMATH_CALUDE_village_population_l2306_230679


namespace NUMINAMATH_CALUDE_john_walks_to_school_l2306_230646

/-- The distance Nina walks to school in miles -/
def nina_distance : ℝ := 0.4

/-- The additional distance John walks compared to Nina in miles -/
def additional_distance : ℝ := 0.3

/-- John's distance to school in miles -/
def john_distance : ℝ := nina_distance + additional_distance

/-- Theorem stating that John walks 0.7 miles to school -/
theorem john_walks_to_school : john_distance = 0.7 := by
  sorry

end NUMINAMATH_CALUDE_john_walks_to_school_l2306_230646


namespace NUMINAMATH_CALUDE_misread_number_correction_l2306_230692

theorem misread_number_correction (n : ℕ) (incorrect_avg correct_avg misread_value : ℚ) 
  (h1 : n = 10)
  (h2 : incorrect_avg = 18)
  (h3 : correct_avg = 22)
  (h4 : misread_value = 26) :
  ∃ (actual_value : ℚ), 
    n * correct_avg = n * incorrect_avg - misread_value + actual_value ∧ 
    actual_value = 66 := by
  sorry

end NUMINAMATH_CALUDE_misread_number_correction_l2306_230692


namespace NUMINAMATH_CALUDE_existence_of_odd_powers_sum_l2306_230628

theorem existence_of_odd_powers_sum (m : ℤ) :
  ∃ (a b k : ℤ), 
    Odd a ∧ 
    Odd b ∧ 
    k > 0 ∧ 
    2 * m = a^19 + b^99 + k * 2^1999 := by
  sorry

end NUMINAMATH_CALUDE_existence_of_odd_powers_sum_l2306_230628


namespace NUMINAMATH_CALUDE_ellipse_properties_l2306_230676

/-- An ellipse with focal length 2 passing through the point (3/2, √6) -/
structure Ellipse where
  a : ℝ
  b : ℝ
  h : a > b ∧ b > 0
  focal_length : a^2 - b^2 = 1
  passes_through : (3/2)^2 / a^2 + 6 / b^2 = 1

/-- The standard equation of the ellipse -/
def standard_equation (e : Ellipse) : Prop :=
  ∀ x y : ℝ, x^2 / 9 + y^2 / 8 = 1 ↔ x^2 / e.a^2 + y^2 / e.b^2 = 1

/-- The trajectory of point E -/
def trajectory_equation (e : Ellipse) : Prop :=
  ∀ x y : ℝ, x ≠ 3 ∧ x ≠ -3 → (x^2 / 9 - y^2 / 8 = 1 ↔
    ∃ x₁ y₁ : ℝ, x₁^2 / e.a^2 + y₁^2 / e.b^2 = 1 ∧ x₁ ≠ 0 ∧ |x₁| < e.a ∧
      y / y₁ = (x + e.a) / (x₁ + e.a) ∧
      y / (-y₁) = (x - e.a) / (x₁ - e.a))

/-- The main theorem to be proved -/
theorem ellipse_properties (e : Ellipse) :
  standard_equation e ∧ trajectory_equation e :=
sorry

end NUMINAMATH_CALUDE_ellipse_properties_l2306_230676


namespace NUMINAMATH_CALUDE_f_2012_equals_2_l2306_230698

def is_odd_function (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = -f x

theorem f_2012_equals_2 
  (f : ℝ → ℝ) 
  (h_odd : is_odd_function f)
  (h_period : ∀ x, f (x + 3) = -f x)
  (h_f_1 : f 1 = 2) : 
  f 2012 = 2 := by
  sorry

end NUMINAMATH_CALUDE_f_2012_equals_2_l2306_230698


namespace NUMINAMATH_CALUDE_fruit_eating_arrangements_l2306_230669

theorem fruit_eating_arrangements : 
  let total_fruits : ℕ := 4 + 2 + 1
  let apples : ℕ := 4
  let oranges : ℕ := 2
  let bananas : ℕ := 1
  (Nat.factorial total_fruits) / (Nat.factorial apples * Nat.factorial oranges * Nat.factorial bananas) = 105 := by
  sorry

end NUMINAMATH_CALUDE_fruit_eating_arrangements_l2306_230669


namespace NUMINAMATH_CALUDE_smallest_digit_change_l2306_230666

def original_sum : ℕ := 2457
def correct_sum : ℕ := 2547
def discrepancy : ℕ := correct_sum - original_sum

def num1 : ℕ := 731
def num2 : ℕ := 964
def num3 : ℕ := 852

def is_smallest_change (d : ℕ) : Prop :=
  d ≤ 9 ∧ 
  (num1 - d * 100 + num2 + num3 = correct_sum) ∧
  ∀ (d' : ℕ), d' < d → (num1 - d' * 100 + num2 + num3 ≠ correct_sum ∧
                        num1 + num2 - d' * 100 + num3 ≠ correct_sum ∧
                        num1 + num2 + num3 - d' * 100 ≠ correct_sum)

theorem smallest_digit_change :
  is_smallest_change 1 :=
sorry

end NUMINAMATH_CALUDE_smallest_digit_change_l2306_230666


namespace NUMINAMATH_CALUDE_shaded_area_in_square_with_triangles_l2306_230667

/-- The area of the shaded region in a square with two right-angle triangles -/
theorem shaded_area_in_square_with_triangles (square_side : ℝ) (triangle_leg : ℝ)
  (h_square : square_side = 40)
  (h_triangle : triangle_leg = 25) :
  square_side ^ 2 - 2 * (triangle_leg ^ 2 / 2) = 975 :=
by sorry

end NUMINAMATH_CALUDE_shaded_area_in_square_with_triangles_l2306_230667


namespace NUMINAMATH_CALUDE_tangent_slope_angle_l2306_230623

/-- The slope angle of the tangent line to y = x^3 forming an isosceles triangle -/
theorem tangent_slope_angle (x₀ : ℝ) : 
  let B : ℝ × ℝ := (x₀, x₀^3)
  let slope : ℝ := 3 * x₀^2
  let A : ℝ × ℝ := ((2/3) * x₀, 0)
  (x₀^4 = 1/3) →  -- This ensures OAB is isosceles
  (slope = Real.sqrt 3) →
  Real.arctan slope = π/3 :=
by sorry

end NUMINAMATH_CALUDE_tangent_slope_angle_l2306_230623


namespace NUMINAMATH_CALUDE_square_side_lengths_l2306_230688

theorem square_side_lengths (a b : ℕ) : 
  a > b → a ^ 2 - b ^ 2 = 2001 → a ∈ ({1001, 335, 55, 49} : Set ℕ) :=
by sorry

end NUMINAMATH_CALUDE_square_side_lengths_l2306_230688


namespace NUMINAMATH_CALUDE_inequality_proof_l2306_230657

theorem inequality_proof (a b c d : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (hd : d > 0) :
  (a^6 - a^2 + 4) * (b^6 - b^2 + 4) * (c^6 - c^2 + 4) * (d^6 - d^2 + 4) ≥ (a + b + c + d)^4 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l2306_230657


namespace NUMINAMATH_CALUDE_divisibility_relation_l2306_230672

theorem divisibility_relation :
  (∀ n : ℤ, n % 6 = 0 → n % 2 = 0) ∧
  (∃ n : ℤ, n % 2 = 0 ∧ n % 6 ≠ 0) := by
  sorry

end NUMINAMATH_CALUDE_divisibility_relation_l2306_230672


namespace NUMINAMATH_CALUDE_consecutive_powers_divisibility_l2306_230619

theorem consecutive_powers_divisibility (a : ℝ) (n : ℕ) :
  ∃ k : ℤ, a^n + a^(n+1) = k * a * (a + 1) := by sorry

end NUMINAMATH_CALUDE_consecutive_powers_divisibility_l2306_230619


namespace NUMINAMATH_CALUDE_fourth_person_win_prob_is_one_thirtieth_l2306_230658

/-- Represents the probability of winning for the fourth person in a
    coin-flipping game with four players where the first to get heads wins. -/
def fourth_person_win_probability : ℚ := 1 / 30

/-- The probability of getting tails on a fair coin flip. -/
def prob_tails : ℚ := 1 / 2

/-- The number of players in the game. -/
def num_players : ℕ := 4

/-- Theorem stating that the probability of the fourth person winning
    in a coin-flipping game with four players is 1/30. -/
theorem fourth_person_win_prob_is_one_thirtieth :
  fourth_person_win_probability = 
    (prob_tails ^ num_players) / (1 - prob_tails ^ num_players) :=
sorry

end NUMINAMATH_CALUDE_fourth_person_win_prob_is_one_thirtieth_l2306_230658


namespace NUMINAMATH_CALUDE_min_transportation_cost_l2306_230626

/-- Represents the transportation problem between cities A, B, C, and D. -/
structure TransportationProblem where
  inventory_A : ℕ := 12
  inventory_B : ℕ := 8
  demand_C : ℕ := 10
  demand_D : ℕ := 10
  cost_A_to_C : ℕ := 300
  cost_A_to_D : ℕ := 500
  cost_B_to_C : ℕ := 400
  cost_B_to_D : ℕ := 800

/-- The total cost function for the transportation problem. -/
def total_cost (tp : TransportationProblem) (x : ℕ) : ℕ :=
  200 * x + 8400

/-- The theorem stating that the minimum total transportation cost is 8800 yuan. -/
theorem min_transportation_cost (tp : TransportationProblem) :
  ∃ (x : ℕ), 2 ≤ x ∧ x ≤ 10 ∧ (∀ (y : ℕ), 2 ≤ y ∧ y ≤ 10 → total_cost tp x ≤ total_cost tp y) ∧
  total_cost tp x = 8800 :=
sorry

#check min_transportation_cost

end NUMINAMATH_CALUDE_min_transportation_cost_l2306_230626


namespace NUMINAMATH_CALUDE_largest_value_l2306_230675

def expr_a : ℕ := 3 + 1 + 2 + 8
def expr_b : ℕ := 3 * 1 + 2 + 8
def expr_c : ℕ := 3 + 1 * 2 + 8
def expr_d : ℕ := 3 + 1 + 2 * 8
def expr_e : ℕ := 3 * 1 * 2 * 8

theorem largest_value :
  expr_e ≥ expr_a ∧ 
  expr_e ≥ expr_b ∧ 
  expr_e ≥ expr_c ∧ 
  expr_e ≥ expr_d :=
by sorry

end NUMINAMATH_CALUDE_largest_value_l2306_230675


namespace NUMINAMATH_CALUDE_line_through_M_and_origin_parallel_line_perpendicular_line_main_theorem_l2306_230650

-- Define the lines
def l₁ (x y : ℝ) : Prop := 3*x + 4*y + 5 = 0
def l₂ (x y : ℝ) : Prop := 2*x - 3*y - 8 = 0
def l₃ (x y : ℝ) : Prop := 2*x + y + 5 = 0

-- Define the intersection point M
def M : ℝ × ℝ := (1, -2)

-- Theorem for the line passing through M and the origin
theorem line_through_M_and_origin :
  ∃ (k : ℝ), ∀ (x y : ℝ), l₁ x y ∧ l₂ x y → (y = k * x) ∧ k = -2 :=
sorry

-- Theorem for the parallel line
theorem parallel_line :
  ∃ (t : ℝ), ∀ (x y : ℝ), l₁ (M.1) (M.2) ∧ l₂ (M.1) (M.2) →
    (2*x + y + t = 0) ∧ t = 0 :=
sorry

-- Theorem for the perpendicular line
theorem perpendicular_line :
  ∃ (s : ℝ), ∀ (x y : ℝ), l₁ (M.1) (M.2) ∧ l₂ (M.1) (M.2) →
    (x - 2*y + s = 0) ∧ s = -5 :=
sorry

-- Main theorem combining all conditions
theorem main_theorem :
  (∀ (x y : ℝ), l₁ x y ∧ l₂ x y → 2*x + y = 0) ∧
  (∀ (x y : ℝ), l₁ x y ∧ l₂ x y → x - 2*y - 5 = 0) :=
sorry

end NUMINAMATH_CALUDE_line_through_M_and_origin_parallel_line_perpendicular_line_main_theorem_l2306_230650


namespace NUMINAMATH_CALUDE_coefficient_x3_in_2x_plus_1_power_5_l2306_230644

-- Define the binomial coefficient function
def binomial (n k : ℕ) : ℕ := sorry

-- Define the coefficient of x^3 in the expansion of (2x+1)^5
def coefficient_x3 : ℕ := binomial 5 2 * 2^3

-- Theorem statement
theorem coefficient_x3_in_2x_plus_1_power_5 : coefficient_x3 = 80 := by sorry

end NUMINAMATH_CALUDE_coefficient_x3_in_2x_plus_1_power_5_l2306_230644


namespace NUMINAMATH_CALUDE_sin_2alpha_plus_2pi_3_l2306_230601

/-- Given an angle α in a Cartesian coordinate system with its vertex at the origin,
    its initial side on the non-negative x-axis, and its terminal side passing through (-1, 2),
    prove that sin(2α + 2π/3) = (4 - 3√3) / 10 -/
theorem sin_2alpha_plus_2pi_3 (α : Real) :
  (∃ (x y : Real), x = -1 ∧ y = 2 ∧ x = Real.cos α * Real.sqrt (x^2 + y^2) ∧ y = Real.sin α * Real.sqrt (x^2 + y^2)) →
  Real.sin (2 * α + 2 * Real.pi / 3) = (4 - 3 * Real.sqrt 3) / 10 := by
  sorry

end NUMINAMATH_CALUDE_sin_2alpha_plus_2pi_3_l2306_230601


namespace NUMINAMATH_CALUDE_four_line_angles_l2306_230661

/-- Given four lines on a plane with angles α, β, and γ between some of them,
    prove that the angles between the remaining pairs of lines are as stated. -/
theorem four_line_angles (α β γ : ℝ) 
  (h_α : α = 110)
  (h_β : β = 60)
  (h_γ : γ = 80) :
  ∃ x y z : ℝ, 
    x = α - γ ∧ 
    z = β - x ∧
    y = α - β ∧
    x = 30 ∧ 
    y = 50 ∧ 
    z = 30 :=
by sorry

end NUMINAMATH_CALUDE_four_line_angles_l2306_230661


namespace NUMINAMATH_CALUDE_apples_handed_out_to_students_l2306_230608

/-- Proves that the number of apples handed out to students is 42 -/
theorem apples_handed_out_to_students (initial_apples : ℕ) (pies : ℕ) (apples_per_pie : ℕ) 
  (h1 : initial_apples = 96)
  (h2 : pies = 9)
  (h3 : apples_per_pie = 6) : 
  initial_apples - pies * apples_per_pie = 42 := by
  sorry

#check apples_handed_out_to_students

end NUMINAMATH_CALUDE_apples_handed_out_to_students_l2306_230608


namespace NUMINAMATH_CALUDE_count_non_divisible_is_30_l2306_230606

/-- g(n) is the product of the proper positive integer divisors of n -/
def g (n : ℕ) : ℕ := sorry

/-- Counts the numbers n between 2 and 100 (inclusive) for which n does not divide g(n) -/
def count_non_divisible : ℕ := sorry

theorem count_non_divisible_is_30 : count_non_divisible = 30 := by sorry

end NUMINAMATH_CALUDE_count_non_divisible_is_30_l2306_230606


namespace NUMINAMATH_CALUDE_arithmetic_sequence_sum_l2306_230681

def arithmetic_sequence (a : ℕ → ℚ) : Prop :=
  ∃ d : ℚ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_sum (a : ℕ → ℚ) :
  arithmetic_sequence a →
  (∀ n : ℕ, a n > 0) →
  a 1 = 1 →
  a 1 + a 3 + a 5 = 21 →
  a 2 + a 4 + a 6 = 42 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_sum_l2306_230681


namespace NUMINAMATH_CALUDE_f_6_l2306_230634

/-- A function satisfying f(x) = f(x - 2) + 3 for all real x, with f(2) = 4 -/
def f : ℝ → ℝ :=
  sorry

/-- The functional equation for f -/
axiom f_eq (x : ℝ) : f x = f (x - 2) + 3

/-- The initial condition for f -/
axiom f_2 : f 2 = 4

/-- Theorem: f(6) = 10 -/
theorem f_6 : f 6 = 10 := by
  sorry

end NUMINAMATH_CALUDE_f_6_l2306_230634


namespace NUMINAMATH_CALUDE_board_cut_theorem_l2306_230636

theorem board_cut_theorem (total_length : ℝ) (short_length : ℝ) : 
  total_length = 6 →
  short_length + 2 * short_length = total_length →
  short_length = 2 := by
sorry

end NUMINAMATH_CALUDE_board_cut_theorem_l2306_230636


namespace NUMINAMATH_CALUDE_function_minimum_implies_a_less_than_one_l2306_230605

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := x^2 - 2*a*x + a

-- State the theorem
theorem function_minimum_implies_a_less_than_one :
  ∀ a : ℝ, (∃ m : ℝ, ∀ x < 1, f a x ≥ f a m) → a < 1 := by
  sorry

end NUMINAMATH_CALUDE_function_minimum_implies_a_less_than_one_l2306_230605


namespace NUMINAMATH_CALUDE_mode_not_necessarily_same_l2306_230649

-- Define the number of shots
def num_shots : ℕ := 10

-- Define the average score for both persons
def average_score : ℝ := 8

-- Define the variances for person A and B
def variance_A : ℝ := 1.2
def variance_B : ℝ := 1.6

-- Define a type for a person's shooting results
structure ShootingResult where
  scores : List ℝ
  average : ℝ
  variance : ℝ

-- Define person A and B's shooting results
def person_A : ShootingResult := {
  scores := [], -- We don't know the actual scores
  average := average_score,
  variance := variance_A
}

def person_B : ShootingResult := {
  scores := [], -- We don't know the actual scores
  average := average_score,
  variance := variance_B
}

-- Theorem: It cannot be concluded that the mode of person A and B's scores must be the same
theorem mode_not_necessarily_same (A B : ShootingResult) 
  (h1 : A.scores.length = num_shots) 
  (h2 : B.scores.length = num_shots)
  (h3 : A.average = average_score) 
  (h4 : B.average = average_score)
  (h5 : A.variance = variance_A) 
  (h6 : B.variance = variance_B) : 
  ¬ (∀ (mode_A mode_B : ℝ), 
    (mode_A ∈ A.scores ∧ (∀ x ∈ A.scores, (A.scores.count mode_A) ≥ (A.scores.count x))) →
    (mode_B ∈ B.scores ∧ (∀ y ∈ B.scores, (B.scores.count mode_B) ≥ (B.scores.count y))) →
    mode_A = mode_B) :=
sorry

end NUMINAMATH_CALUDE_mode_not_necessarily_same_l2306_230649


namespace NUMINAMATH_CALUDE_sum_and_count_theorem_l2306_230662

def sum_range (a b : ℕ) : ℕ := (b - a + 1) * (a + b) / 2

def count_even_in_range (a b : ℕ) : ℕ := ((b - a) / 2) + 1

theorem sum_and_count_theorem :
  sum_range 60 80 + count_even_in_range 60 80 = 1481 := by
  sorry

end NUMINAMATH_CALUDE_sum_and_count_theorem_l2306_230662


namespace NUMINAMATH_CALUDE_shared_bikes_theorem_l2306_230615

def a (n : ℕ+) : ℕ :=
  if n ≤ 3 then 5 * n^4 + 15 else 470 - 10 * n

def b (n : ℕ+) : ℕ := n + 5

def S (n : ℕ+) : ℕ := 8800 - 4 * (n - 46)^2

def remaining_bikes (n : ℕ+) : ℕ := 
  (Finset.range n).sum (λ i => a ⟨i + 1, Nat.succ_pos i⟩) - 
  (Finset.range n).sum (λ i => b ⟨i + 1, Nat.succ_pos i⟩)

theorem shared_bikes_theorem :
  remaining_bikes 4 = 945 ∧
  remaining_bikes 42 = 8782 ∧
  S 42 = 8736 ∧
  remaining_bikes 42 > S 42 :=
sorry

end NUMINAMATH_CALUDE_shared_bikes_theorem_l2306_230615


namespace NUMINAMATH_CALUDE_arithmetic_sequence_general_term_l2306_230642

/-- An arithmetic sequence satisfying given conditions has one of two specific general terms -/
theorem arithmetic_sequence_general_term 
  (a : ℕ → ℚ) 
  (h_arithmetic : ∀ n : ℕ, a (n + 1) - a n = a (n + 2) - a (n + 1)) 
  (h_sum : a 2 + a 7 + a 12 = 12) 
  (h_product : a 2 * a 7 * a 12 = 28) :
  (∃ C : ℚ, ∀ n : ℕ, a n = 3/5 * n - 1/5 + C) ∨ 
  (∃ C : ℚ, ∀ n : ℕ, a n = -3/5 * n + 41/5 + C) :=
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_general_term_l2306_230642


namespace NUMINAMATH_CALUDE_female_employees_percentage_l2306_230614

/-- The percentage of female employees in an office -/
def percentage_female_employees (total_employees : ℕ) 
  (percent_computer_literate : ℚ) 
  (female_computer_literate : ℕ) 
  (percent_male_computer_literate : ℚ) : ℚ :=
  sorry

/-- Theorem stating the percentage of female employees is 60% -/
theorem female_employees_percentage 
  (h1 : total_employees = 1500)
  (h2 : percent_computer_literate = 62 / 100)
  (h3 : female_computer_literate = 630)
  (h4 : percent_male_computer_literate = 1 / 2) :
  percentage_female_employees total_employees percent_computer_literate 
    female_computer_literate percent_male_computer_literate = 60 / 100 :=
by sorry

end NUMINAMATH_CALUDE_female_employees_percentage_l2306_230614


namespace NUMINAMATH_CALUDE_negation_of_universal_negation_of_proposition_l2306_230627

theorem negation_of_universal (p : ℝ → Prop) :
  (¬ ∀ x : ℝ, p x) ↔ (∃ x : ℝ, ¬ p x) := by sorry

theorem negation_of_proposition :
  (¬ ∀ x : ℝ, 2*x - 1 > 0) ↔ (∃ x : ℝ, 2*x - 1 ≤ 0) := by sorry

end NUMINAMATH_CALUDE_negation_of_universal_negation_of_proposition_l2306_230627


namespace NUMINAMATH_CALUDE_valid_distributions_count_l2306_230685

/-- Represents a triangular array of 8 rows -/
def TriangularArray := Fin 8 → Fin 8 → ℕ

/-- The bottom row of the triangular array -/
def BottomRow := Fin 8 → Fin 2

/-- Checks if a number is a multiple of 5 -/
def IsMultipleOf5 (n : ℕ) : Prop := ∃ k, n = 5 * k

/-- Calculates the value of a square based on the two squares below it -/
def CalculateSquareValue (arr : TriangularArray) (row : Fin 8) (col : Fin 8) : ℕ :=
  if row = 0 then arr 0 col
  else arr (row - 1) col + arr (row - 1) (col + 1)

/-- Builds the triangular array from the bottom row -/
def BuildArray (bottom : BottomRow) : TriangularArray :=
  sorry

/-- Counts the number of valid bottom row distributions -/
def CountValidDistributions : ℕ :=
  sorry

/-- The main theorem stating that the count of valid distributions is 32 -/
theorem valid_distributions_count :
  CountValidDistributions = 32 :=
sorry

end NUMINAMATH_CALUDE_valid_distributions_count_l2306_230685


namespace NUMINAMATH_CALUDE_cos_sin_cos_bounds_l2306_230637

theorem cos_sin_cos_bounds (x y z : ℝ) 
  (h1 : x ≥ y) (h2 : y ≥ z) (h3 : z ≥ π/12) 
  (h4 : x + y + z = π/2) : 
  1/8 ≤ Real.cos x * Real.sin y * Real.cos z ∧ 
  Real.cos x * Real.sin y * Real.cos z ≤ (2 + Real.sqrt 3) / 8 := by
  sorry

end NUMINAMATH_CALUDE_cos_sin_cos_bounds_l2306_230637


namespace NUMINAMATH_CALUDE_kaleb_restaurant_bill_l2306_230613

/-- Calculates the total bill for a group at Kaleb's Restaurant -/
def total_bill (num_adults : ℕ) (num_children : ℕ) (adult_meal_cost : ℕ) (child_meal_cost : ℕ) (soda_cost : ℕ) : ℕ :=
  num_adults * adult_meal_cost + num_children * child_meal_cost + (num_adults + num_children) * soda_cost

/-- Theorem: The total bill for a group of 6 adults and 2 children at Kaleb's Restaurant is $60 -/
theorem kaleb_restaurant_bill :
  total_bill 6 2 6 4 2 = 60 := by
  sorry

end NUMINAMATH_CALUDE_kaleb_restaurant_bill_l2306_230613


namespace NUMINAMATH_CALUDE_friends_for_games_only_l2306_230664

-- Define the variables
def movie : ℕ := 10
def picnic : ℕ := 20
def movie_and_picnic : ℕ := 4
def movie_and_games : ℕ := 2
def picnic_and_games : ℕ := 0
def all_three : ℕ := 2
def total_students : ℕ := 31

-- Theorem to prove
theorem friends_for_games_only : 
  ∃ (movie_only picnic_only games_only : ℕ),
    movie_only + picnic_only + games_only + movie_and_picnic + movie_and_games + picnic_and_games + all_three = total_students ∧
    movie_only + movie_and_picnic + movie_and_games + all_three = movie ∧
    picnic_only + movie_and_picnic + picnic_and_games + all_three = picnic ∧
    games_only = 1 := by
  sorry

end NUMINAMATH_CALUDE_friends_for_games_only_l2306_230664


namespace NUMINAMATH_CALUDE_polynomial_sum_simplification_l2306_230648

theorem polynomial_sum_simplification :
  ∀ x : ℝ, (2 * x^3 - 3 * x^2 + 5 * x - 6) + (5 * x^4 - 2 * x^3 - 4 * x^2 - x + 8) =
            5 * x^4 - 7 * x^2 + 4 * x + 2 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_sum_simplification_l2306_230648


namespace NUMINAMATH_CALUDE_zero_in_M_l2306_230690

def M : Set ℝ := {x | x^2 - 3 ≤ 0}

theorem zero_in_M : (0 : ℝ) ∈ M := by
  sorry

end NUMINAMATH_CALUDE_zero_in_M_l2306_230690


namespace NUMINAMATH_CALUDE_trig_product_value_l2306_230655

theorem trig_product_value : 
  Real.sin (4/3 * Real.pi) * Real.cos (5/6 * Real.pi) * Real.tan (-4/3 * Real.pi) = -3 * Real.sqrt 3 / 4 := by
  sorry

end NUMINAMATH_CALUDE_trig_product_value_l2306_230655


namespace NUMINAMATH_CALUDE_pauls_crayons_l2306_230665

theorem pauls_crayons (erasers_birthday : ℕ) (crayons_left : ℕ) (eraser_crayon_diff : ℕ) 
  (h1 : erasers_birthday = 406)
  (h2 : crayons_left = 336)
  (h3 : eraser_crayon_diff = 70)
  (h4 : erasers_birthday = crayons_left + eraser_crayon_diff) :
  crayons_left + eraser_crayon_diff = 406 := by
  sorry

end NUMINAMATH_CALUDE_pauls_crayons_l2306_230665


namespace NUMINAMATH_CALUDE_condition_necessary_not_sufficient_l2306_230622

theorem condition_necessary_not_sufficient (a b : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) :
  (∀ a b, a > b ∧ b > 0 → 1 / a < 1 / b) ∧
  (∃ a b, 1 / a < 1 / b ∧ ¬(a > b ∧ b > 0)) :=
by sorry

end NUMINAMATH_CALUDE_condition_necessary_not_sufficient_l2306_230622


namespace NUMINAMATH_CALUDE_geometric_sum_first_8_terms_l2306_230697

def geometric_sum (a : ℚ) (r : ℚ) (n : ℕ) : ℚ :=
  a * (1 - r^n) / (1 - r)

theorem geometric_sum_first_8_terms :
  let a : ℚ := 1/3
  let r : ℚ := 1/2
  let n : ℕ := 8
  geometric_sum a r n = 85/128 := by
sorry

end NUMINAMATH_CALUDE_geometric_sum_first_8_terms_l2306_230697


namespace NUMINAMATH_CALUDE_smallest_integer_solution_l2306_230625

theorem smallest_integer_solution (x : ℤ) : 
  (∀ y : ℤ, 2*y + 5 < 3*y - 10 → y ≥ 16) ∧ (2*16 + 5 < 3*16 - 10) := by
  sorry

end NUMINAMATH_CALUDE_smallest_integer_solution_l2306_230625


namespace NUMINAMATH_CALUDE_max_min_sum_l2306_230682

noncomputable def y (x : ℝ) : ℝ :=
  (2 * Real.sin x ^ 2 + Real.sin (3 * x / 2) - 4) / (Real.sin x ^ 2 + 2 * Real.cos x ^ 2)

theorem max_min_sum (M m : ℝ) 
  (hM : ∀ x, y x ≤ M) 
  (hm : ∀ x, m ≤ y x) 
  (hM_exists : ∃ x, y x = M) 
  (hm_exists : ∃ x, y x = m) : 
  M + m = -4 := by
  sorry

end NUMINAMATH_CALUDE_max_min_sum_l2306_230682


namespace NUMINAMATH_CALUDE_point_in_fourth_quadrant_l2306_230671

/-- The point P(1+m^2, -1) lies in the fourth quadrant for any real number m. -/
theorem point_in_fourth_quadrant (m : ℝ) : 
  let x : ℝ := 1 + m^2
  let y : ℝ := -1
  x > 0 ∧ y < 0 := by
sorry


end NUMINAMATH_CALUDE_point_in_fourth_quadrant_l2306_230671


namespace NUMINAMATH_CALUDE_dinner_cakes_l2306_230620

def total_cakes : ℕ := 15
def lunch_cakes : ℕ := 6

theorem dinner_cakes : total_cakes - lunch_cakes = 9 := by
  sorry

end NUMINAMATH_CALUDE_dinner_cakes_l2306_230620


namespace NUMINAMATH_CALUDE_correct_average_after_errors_l2306_230611

theorem correct_average_after_errors (n : ℕ) (initial_avg : ℚ) 
  (error1 : ℚ) (error2 : ℚ) (error3 : ℚ) : 
  n = 15 → 
  initial_avg = 24 → 
  error1 = 65 - 45 → 
  error2 = 42 - 28 → 
  error3 = 75 - 55 → 
  (n : ℚ) * initial_avg + error1 + error2 + error3 = n * (27.6 : ℚ) := by
  sorry

end NUMINAMATH_CALUDE_correct_average_after_errors_l2306_230611


namespace NUMINAMATH_CALUDE_ralph_peanuts_l2306_230618

def initial_peanuts : ℕ := 74
def lost_peanuts : ℕ := 59

theorem ralph_peanuts : initial_peanuts - lost_peanuts = 15 := by
  sorry

end NUMINAMATH_CALUDE_ralph_peanuts_l2306_230618


namespace NUMINAMATH_CALUDE_tournament_three_cycle_l2306_230654

/-- Represents a tournament with n contestants. -/
structure Tournament (n : ℕ) where
  -- n ≥ 3
  contestants_count : n ≥ 3
  -- Represents the result of matches between contestants
  defeats : Fin n → Fin n → Prop
  -- Each pair of contestants plays exactly one match
  one_match (i j : Fin n) : i ≠ j → (defeats i j ∨ defeats j i) ∧ ¬(defeats i j ∧ defeats j i)
  -- No contestant wins all their matches
  no_perfect_winner (i : Fin n) : ∃ j : Fin n, j ≠ i ∧ defeats j i

/-- 
There exist three contestants A, B, and C such that A defeats B, B defeats C, and C defeats A.
-/
theorem tournament_three_cycle {n : ℕ} (t : Tournament n) :
  ∃ (a b c : Fin n), a ≠ b ∧ b ≠ c ∧ c ≠ a ∧ 
    t.defeats a b ∧ t.defeats b c ∧ t.defeats c a :=
sorry

end NUMINAMATH_CALUDE_tournament_three_cycle_l2306_230654


namespace NUMINAMATH_CALUDE_intersection_M_N_l2306_230633

def M : Set ℝ := {x : ℝ | (x + 2) * (x - 2) > 0}

def N : Set ℝ := {-3, -2, 2, 3, 4}

theorem intersection_M_N : M ∩ N = {-3, 3, 4} := by sorry

end NUMINAMATH_CALUDE_intersection_M_N_l2306_230633


namespace NUMINAMATH_CALUDE_family_heights_l2306_230686

/-- Represents the heights of family members and proves statements about their relationships -/
theorem family_heights (binbin_height mother_height : Real) 
  (father_taller_by : Real) (h1 : binbin_height = 1.46) 
  (h2 : father_taller_by = 0.32) (h3 : mother_height = 1.5) : 
  (binbin_height + father_taller_by = 1.78) ∧ 
  ((binbin_height + father_taller_by) - mother_height = 0.28) := by
  sorry

#check family_heights

end NUMINAMATH_CALUDE_family_heights_l2306_230686


namespace NUMINAMATH_CALUDE_average_of_abc_l2306_230617

theorem average_of_abc (A B C : ℚ) 
  (eq1 : 2002 * C - 3003 * A = 6006)
  (eq2 : 2002 * B + 4004 * A = 8008)
  (eq3 : B - C = A + 1) :
  (A + B + C) / 3 = 7 / 3 := by
  sorry

end NUMINAMATH_CALUDE_average_of_abc_l2306_230617


namespace NUMINAMATH_CALUDE_min_value_z_l2306_230612

theorem min_value_z (x y : ℝ) : 3 * x^2 + 5 * y^2 + 6 * x - 4 * y + 3 * x^3 + 15 ≥ 8.2 := by
  sorry

end NUMINAMATH_CALUDE_min_value_z_l2306_230612


namespace NUMINAMATH_CALUDE_tea_blend_gain_percent_l2306_230603

/-- Represents the cost and quantity of a tea variety -/
structure TeaVariety where
  cost : ℚ
  quantity : ℚ

/-- Calculates the gain percent for a tea blend -/
def gainPercent (tea1 : TeaVariety) (tea2 : TeaVariety) (sellingPrice : ℚ) : ℚ :=
  let totalCost := tea1.cost * tea1.quantity + tea2.cost * tea2.quantity
  let totalQuantity := tea1.quantity + tea2.quantity
  let costPrice := totalCost / totalQuantity
  ((sellingPrice - costPrice) / costPrice) * 100

/-- Theorem stating that the gain percent for the given tea blend is 12% -/
theorem tea_blend_gain_percent :
  let tea1 := TeaVariety.mk 18 5
  let tea2 := TeaVariety.mk 20 3
  let sellingPrice := 21
  gainPercent tea1 tea2 sellingPrice = 12 := by
  sorry

#eval gainPercent (TeaVariety.mk 18 5) (TeaVariety.mk 20 3) 21

end NUMINAMATH_CALUDE_tea_blend_gain_percent_l2306_230603


namespace NUMINAMATH_CALUDE_alpha_equals_five_l2306_230695

-- Define the grid as a 3x3 matrix of natural numbers
def Grid := Matrix (Fin 3) (Fin 3) Nat

-- Define a predicate to check if a number is a non-zero digit
def IsNonZeroDigit (n : Nat) : Prop := 0 < n ∧ n ≤ 9

-- Define a predicate to check if all elements in the grid are distinct
def AllDistinct (g : Grid) : Prop :=
  ∀ i j k l, (i, j) ≠ (k, l) → g i j ≠ g k l

-- Define a predicate to check if all elements in the grid are non-zero digits
def AllNonZeroDigits (g : Grid) : Prop :=
  ∀ i j, IsNonZeroDigit (g i j)

-- Define a predicate to check if all horizontal expressions are correct
def HorizontalExpressionsCorrect (g : Grid) : Prop :=
  (g 0 0 + g 0 1 = g 0 2) ∧
  (g 1 0 - g 1 1 = g 1 2) ∧
  (g 2 0 * g 2 1 = g 2 2)

-- Define a predicate to check if all vertical expressions are correct
def VerticalExpressionsCorrect (g : Grid) : Prop :=
  (g 0 0 + g 1 0 = g 2 0) ∧
  (g 0 1 - g 1 1 = g 2 1) ∧
  (g 0 2 * g 1 2 = g 2 2)

-- Main theorem
theorem alpha_equals_five (g : Grid) (α : Nat)
  (h1 : AllDistinct g)
  (h2 : AllNonZeroDigits g)
  (h3 : HorizontalExpressionsCorrect g)
  (h4 : VerticalExpressionsCorrect g)
  (h5 : ∃ i j, g i j = α) :
  α = 5 := by
  sorry

end NUMINAMATH_CALUDE_alpha_equals_five_l2306_230695


namespace NUMINAMATH_CALUDE_divide_fractions_and_mixed_number_l2306_230677

theorem divide_fractions_and_mixed_number :
  (5 : ℚ) / 6 / (1 + 3 / 9) = 5 / 8 := by
  sorry

end NUMINAMATH_CALUDE_divide_fractions_and_mixed_number_l2306_230677


namespace NUMINAMATH_CALUDE_complex_number_quadrant_l2306_230693

theorem complex_number_quadrant : 
  let z : ℂ := (2 - Complex.I) / (1 + Complex.I)
  (z.re > 0 ∧ z.im < 0) := by sorry

end NUMINAMATH_CALUDE_complex_number_quadrant_l2306_230693


namespace NUMINAMATH_CALUDE_intersection_when_a_is_3_intersection_equals_A_iff_l2306_230604

def A (a : ℝ) := { x : ℝ | a ≤ x ∧ x ≤ a + 3 }
def B := { x : ℝ | x < -1 ∨ x > 5 }

theorem intersection_when_a_is_3 :
  A 3 ∩ B = { x : ℝ | 5 < x ∧ x ≤ 6 } := by sorry

theorem intersection_equals_A_iff (a : ℝ) :
  A a ∩ B = A a ↔ a < -4 ∨ a > 5 := by sorry

end NUMINAMATH_CALUDE_intersection_when_a_is_3_intersection_equals_A_iff_l2306_230604


namespace NUMINAMATH_CALUDE_inverse_mod_78_l2306_230694

theorem inverse_mod_78 (h : (7⁻¹ : ZMod 78) = 55) : (49⁻¹ : ZMod 78) = 61 := by
  sorry

end NUMINAMATH_CALUDE_inverse_mod_78_l2306_230694

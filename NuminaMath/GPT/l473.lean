import Mathlib

namespace NUMINAMATH_GPT_circle_possible_m_values_l473_47385

theorem circle_possible_m_values (m : ℝ) :
  (∃ x y : ℝ, x^2 + y^2 + m * x - m * y + 2 = 0) ↔ m > 2 ∨ m < -2 :=
by
  sorry

end NUMINAMATH_GPT_circle_possible_m_values_l473_47385


namespace NUMINAMATH_GPT_problem_statement_l473_47303

variable {a b c d : ℚ}

-- Conditions
axiom h1 : a / b = 3
axiom h2 : b / c = 3 / 4
axiom h3 : c / d = 2 / 3

-- Goal
theorem problem_statement : d / a = 2 / 3 := by
  sorry

end NUMINAMATH_GPT_problem_statement_l473_47303


namespace NUMINAMATH_GPT_find_number_l473_47376

theorem find_number (N : ℚ) (h : (5 / 6) * N = (5 / 16) * N + 100) : N = 192 :=
sorry

end NUMINAMATH_GPT_find_number_l473_47376


namespace NUMINAMATH_GPT_jodi_walked_miles_per_day_l473_47302

theorem jodi_walked_miles_per_day (x : ℕ) 
  (h1 : 6 * x + 12 + 18 + 24 = 60) : 
  x = 1 :=
by
  sorry

end NUMINAMATH_GPT_jodi_walked_miles_per_day_l473_47302


namespace NUMINAMATH_GPT_solve_for_x_l473_47322

theorem solve_for_x (x : ℝ) (h : 3 * x + 15 = 1 / 3 * (6 * x + 45)) : x = 0 :=
sorry

end NUMINAMATH_GPT_solve_for_x_l473_47322


namespace NUMINAMATH_GPT_natural_numbers_divisible_by_6_l473_47397

theorem natural_numbers_divisible_by_6 :
  {n : ℕ | 2 ≤ n ∧ n ≤ 88 ∧ 6 ∣ n} = {n | n = 6 * k ∧ 1 ≤ k ∧ k ≤ 14} :=
by
  sorry

end NUMINAMATH_GPT_natural_numbers_divisible_by_6_l473_47397


namespace NUMINAMATH_GPT_tenth_term_is_correct_l473_47314

-- Definitions corresponding to the problem conditions
def sequence_term (n : ℕ) : ℚ := (-1)^n * (2 * n + 1) / (n^2 + 1)

-- Theorem statement for the equivalent proof problem
theorem tenth_term_is_correct : sequence_term 10 = 21 / 101 := by sorry

end NUMINAMATH_GPT_tenth_term_is_correct_l473_47314


namespace NUMINAMATH_GPT_sequence_type_l473_47395

-- Definitions based on the conditions
def Sn (a : ℝ) (n : ℕ) : ℝ := a^n - 1

def sequence_an (a : ℝ) (n : ℕ) : ℝ :=
  if n = 1 then a - 1 else (Sn a n - Sn a (n - 1))

-- Proving the mathematical statement
theorem sequence_type (a : ℝ) (h : a ≠ 0) : 
  (∀ n > 1, (sequence_an a n = sequence_an a 1 + (n - 1) * (sequence_an a 2 - sequence_an a 1)) ∨
  (∀ n > 2, sequence_an a n / sequence_an a (n-1) = a)) :=
sorry

end NUMINAMATH_GPT_sequence_type_l473_47395


namespace NUMINAMATH_GPT_same_solution_set_l473_47320

theorem same_solution_set :
  (∀ x : ℝ, (x - 1) / (x - 2) ≤ 0 ↔ (x^3 - x^2 + x - 1) / (x - 2) ≤ 0) :=
sorry

end NUMINAMATH_GPT_same_solution_set_l473_47320


namespace NUMINAMATH_GPT_total_enemies_l473_47368

theorem total_enemies (points_per_enemy : ℕ) (points_earned : ℕ) (enemies_left : ℕ) (enemies_defeated : ℕ) :  
  (3 = points_per_enemy) → 
  (12 = points_earned) → 
  (2 = enemies_left) → 
  (points_earned / points_per_enemy = enemies_defeated) → 
  (enemies_defeated + enemies_left = 6) := 
by
  intros
  sorry

end NUMINAMATH_GPT_total_enemies_l473_47368


namespace NUMINAMATH_GPT_find_possible_values_l473_47370

def real_number_y (y : ℝ) := (3 < y ∧ y < 4)

theorem find_possible_values (y : ℝ) (h : real_number_y y) : 
  42 < (y^2 + 7*y + 12) ∧ (y^2 + 7*y + 12) < 56 := 
sorry

end NUMINAMATH_GPT_find_possible_values_l473_47370


namespace NUMINAMATH_GPT_total_accepted_cartons_l473_47388

theorem total_accepted_cartons 
  (total_cartons : ℕ) 
  (customers : ℕ) 
  (damaged_cartons : ℕ)
  (h1 : total_cartons = 400)
  (h2 : customers = 4)
  (h3 : damaged_cartons = 60)
  : total_cartons / customers * (customers - (damaged_cartons / (total_cartons / customers))) = 160 := by
  sorry

end NUMINAMATH_GPT_total_accepted_cartons_l473_47388


namespace NUMINAMATH_GPT_sheep_count_l473_47373

theorem sheep_count (S H : ℕ) (h1 : S / H = 3 / 7) (h2 : H * 230 = 12880) : S = 24 :=
by
  sorry

end NUMINAMATH_GPT_sheep_count_l473_47373


namespace NUMINAMATH_GPT_sum_of_midpoint_coordinates_l473_47366

theorem sum_of_midpoint_coordinates :
  let x1 := 3
  let y1 := -1
  let x2 := 11
  let y2 := 21
  let midpoint_x := (x1 + x2) / 2
  let midpoint_y := (y1 + y2) / 2
  midpoint_x + midpoint_y = 17 := by
  sorry

end NUMINAMATH_GPT_sum_of_midpoint_coordinates_l473_47366


namespace NUMINAMATH_GPT_ratio_of_doctors_to_nurses_l473_47391

def total_staff : ℕ := 250
def nurses : ℕ := 150
def doctors : ℕ := total_staff - nurses

theorem ratio_of_doctors_to_nurses : 
  (doctors : ℚ) / (nurses : ℚ) = 2 / 3 := by
  sorry

end NUMINAMATH_GPT_ratio_of_doctors_to_nurses_l473_47391


namespace NUMINAMATH_GPT_John_writing_years_l473_47380

def books_written (total_earnings per_book_earning : ℕ) : ℕ :=
  total_earnings / per_book_earning

def books_per_year (months_in_year months_per_book : ℕ) : ℕ :=
  months_in_year / months_per_book

def years_writing (total_books books_per_year : ℕ) : ℕ :=
  total_books / books_per_year

theorem John_writing_years :
  let total_earnings := 3600000
  let per_book_earning := 30000
  let months_in_year := 12
  let months_per_book := 2
  let total_books := books_written total_earnings per_book_earning
  let books_per_year := books_per_year months_in_year months_per_book
  years_writing total_books books_per_year = 20 := by
sorry

end NUMINAMATH_GPT_John_writing_years_l473_47380


namespace NUMINAMATH_GPT_solve_system_of_equations_l473_47308

theorem solve_system_of_equations :
  (∃ x y : ℝ, (x / y + y / x = 173 / 26) ∧ (1 / x + 1 / y = 15 / 26) ∧ ((x = 13 ∧ y = 2) ∨ (x = 2 ∧ y = 13))) :=
by
  sorry

end NUMINAMATH_GPT_solve_system_of_equations_l473_47308


namespace NUMINAMATH_GPT_binding_cost_is_correct_l473_47357

-- Definitions for the conditions used in the problem
def total_cost : ℝ := 250      -- Total cost to copy and bind 10 manuscripts
def copy_cost_per_page : ℝ := 0.05   -- Cost per page to copy
def pages_per_manuscript : ℕ := 400  -- Number of pages in each manuscript
def num_manuscripts : ℕ := 10      -- Number of manuscripts

-- The target value we want to prove
def binding_cost_per_manuscript : ℝ := 5 

-- The theorem statement proving the binding cost per manuscript
theorem binding_cost_is_correct :
  let copy_cost_per_manuscript := pages_per_manuscript * copy_cost_per_page
  let total_copy_cost := num_manuscripts * copy_cost_per_manuscript
  let total_binding_cost := total_cost - total_copy_cost
  (total_binding_cost / num_manuscripts) = binding_cost_per_manuscript :=
by
  sorry

end NUMINAMATH_GPT_binding_cost_is_correct_l473_47357


namespace NUMINAMATH_GPT_constant_term_equality_l473_47367

theorem constant_term_equality (a : ℝ) 
  (h1 : ∃ T, T = (x : ℝ)^2 + 2 / x ∧ T^9 = 64 * ↑(Nat.choose 9 6)) 
  (h2 : ∃ T, T = (x : ℝ) + a / (x^2) ∧ T^9 = a^3 * ↑(Nat.choose 9 3)):
  a = 4 := 
sorry

end NUMINAMATH_GPT_constant_term_equality_l473_47367


namespace NUMINAMATH_GPT_find_positive_solutions_l473_47372

noncomputable def satisfies_eq1 (x y : ℝ) : Prop :=
  2 * x - Real.sqrt (x * y) - 4 * Real.sqrt (x / y) + 2 = 0

noncomputable def satisfies_eq2 (x y : ℝ) : Prop :=
  2 * x^2 + x^2 * y^4 = 18 * y^2

theorem find_positive_solutions (x y : ℝ) (hx : 0 < x) (hy : 0 < y) :
  satisfies_eq1 x y ∧ satisfies_eq2 x y ↔ 
  (x = 2 ∧ y = 2) ∨ 
  (x = Real.sqrt 286^(1/4) / 4 ∧ y = Real.sqrt 286^(1/4)) :=
sorry

end NUMINAMATH_GPT_find_positive_solutions_l473_47372


namespace NUMINAMATH_GPT_num_valid_arrangements_l473_47371

-- Define the people and the days of the week
inductive Person := | A | B | C | D | E
inductive DayOfWeek := | Monday | Tuesday | Wednesday | Thursday | Friday

-- Define the arrangement function type
def Arrangement := DayOfWeek → Person

/-- The total number of valid arrangements for 5 people
    (A, B, C, D, E) on duty from Monday to Friday such that:
    - A and B are not on duty on adjacent days,
    - B and C are on duty on adjacent days,
    is 36.
-/
theorem num_valid_arrangements : 
  ∃ (arrangements : Finset (Arrangement)), arrangements.card = 36 ∧
  (∀ (x : Arrangement), x ∈ arrangements →
    (∀ (d1 d2 : DayOfWeek), 
      (d1 = Monday ∧ d2 = Tuesday ∨ d1 = Tuesday ∧ d2 = Wednesday ∨
       d1 = Wednesday ∧ d2 = Thursday ∨ d1 = Thursday ∧ d2 = Friday) →
      ¬(x d1 = Person.A ∧ x d2 = Person.B)) ∧
    (∃ (d1 d2 : DayOfWeek),
      (d1 = Monday ∧ d2 = Tuesday ∨ d1 = Tuesday ∧ d2 = Wednesday ∨
       d1 = Wednesday ∧ d2 = Thursday ∨ d1 = Thursday ∧ d2 = Friday) ∧
      (x d1 = Person.B ∧ x d2 = Person.C)))
  := sorry

end NUMINAMATH_GPT_num_valid_arrangements_l473_47371


namespace NUMINAMATH_GPT_successful_experimental_operation_l473_47394

/-- Problem statement:
Given the following biological experimental operations:
1. spreading diluted E. coli culture on solid medium,
2. introducing sterile air into freshly inoculated grape juice with yeast,
3. inoculating soil leachate on beef extract peptone medium,
4. using slightly opened rose flowers as experimental material for anther culture.

Prove that spreading diluted E. coli culture on solid medium can successfully achieve the experimental objective of obtaining single colonies.
-/
theorem successful_experimental_operation :
  ∃ objective_result,
    (objective_result = "single_colonies" →
     let operation_A := "spreading diluted E. coli culture on solid medium"
     let operation_B := "introducing sterile air into freshly inoculated grape juice with yeast"
     let operation_C := "inoculating soil leachate on beef extract peptone medium"
     let operation_D := "slightly opened rose flowers as experimental material for anther culture"
     ∃ successful_operation,
       successful_operation = operation_A
       ∧ (successful_operation = operation_A → objective_result = "single_colonies")
       ∧ (successful_operation = operation_B → objective_result ≠ "single_colonies")
       ∧ (successful_operation = operation_C → objective_result ≠ "single_colonies")
       ∧ (successful_operation = operation_D → objective_result ≠ "single_colonies")) :=
sorry

end NUMINAMATH_GPT_successful_experimental_operation_l473_47394


namespace NUMINAMATH_GPT_bounded_expression_l473_47386

theorem bounded_expression (x y z : ℝ) (hx : 0 ≤ x) (hy : 0 ≤ y) (hz : 0 ≤ z) (h_sum : x + y + z = 1) :
  0 ≤ y * z + z * x + x * y - 2 * x * y * z ∧ y * z + z * x + x * y - 2 * x * y * z ≤ 7 / 27 := 
sorry

end NUMINAMATH_GPT_bounded_expression_l473_47386


namespace NUMINAMATH_GPT_find_g_neg3_l473_47392

def f (x : ℚ) : ℚ := 4 * x - 6
def g (u : ℚ) : ℚ := 3 * (f u)^2 + 4 * (f u) - 2

theorem find_g_neg3 : g (-3) = 43 / 16 := by
  sorry

end NUMINAMATH_GPT_find_g_neg3_l473_47392


namespace NUMINAMATH_GPT_tricycles_count_l473_47360

theorem tricycles_count {s t : Nat} (h1 : s + t = 10) (h2 : 2 * s + 3 * t = 26) : t = 6 :=
sorry

end NUMINAMATH_GPT_tricycles_count_l473_47360


namespace NUMINAMATH_GPT_josef_game_l473_47304

theorem josef_game : 
  ∃ S : Finset ℕ, 
    (∀ n ∈ S, 1 ≤ n ∧ n ≤ 1440 ∧ 1440 % n = 0 ∧ n % 5 = 0) ∧ 
    S.card = 18 := sorry

end NUMINAMATH_GPT_josef_game_l473_47304


namespace NUMINAMATH_GPT_recording_time_is_one_hour_l473_47375

-- Define the recording interval and number of instances
def recording_interval : ℕ := 5 -- The device records data every 5 seconds
def number_of_instances : ℕ := 720 -- The device recorded 720 instances of data

-- Prove that the total recording time is 1 hour
theorem recording_time_is_one_hour : (recording_interval * number_of_instances) / 3600 = 1 := by
  sorry

end NUMINAMATH_GPT_recording_time_is_one_hour_l473_47375


namespace NUMINAMATH_GPT_water_flow_into_sea_per_minute_l473_47377

noncomputable def river_flow_rate_kmph : ℝ := 4
noncomputable def river_depth_m : ℝ := 5
noncomputable def river_width_m : ℝ := 19
noncomputable def hours_to_minutes : ℝ := 60
noncomputable def km_to_m : ℝ := 1000

noncomputable def flow_rate_m_per_min : ℝ := (river_flow_rate_kmph * km_to_m) / hours_to_minutes
noncomputable def cross_sectional_area_m2 : ℝ := river_depth_m * river_width_m
noncomputable def volume_per_minute_m3 : ℝ := cross_sectional_area_m2 * flow_rate_m_per_min

theorem water_flow_into_sea_per_minute :
  volume_per_minute_m3 = 6333.65 := by 
  -- Proof would go here
  sorry

end NUMINAMATH_GPT_water_flow_into_sea_per_minute_l473_47377


namespace NUMINAMATH_GPT_trigonometric_identity_l473_47313

theorem trigonometric_identity 
  (α : ℝ) 
  (h : Real.sin (α + π / 3) = 12 / 13) 
  : Real.cos (π / 6 - α) = 12 / 13 := 
sorry

end NUMINAMATH_GPT_trigonometric_identity_l473_47313


namespace NUMINAMATH_GPT_payal_finished_fraction_l473_47326

-- Define the conditions
variables (x : ℕ)

-- Given conditions
-- 1. Total pages in the book
def total_pages : ℕ := 60
-- 2. Payal has finished 20 more pages than she has yet to read.
def pages_yet_to_read (x : ℕ) : ℕ := x - 20

-- Main statement to prove: the fraction of the pages finished is 2/3
theorem payal_finished_fraction (h : x + (x - 20) = 60) : (x : ℚ) / 60 = 2 / 3 :=
sorry

end NUMINAMATH_GPT_payal_finished_fraction_l473_47326


namespace NUMINAMATH_GPT_number_of_clients_l473_47307

theorem number_of_clients (num_cars num_selections_per_car num_cars_per_client total_selections num_clients : ℕ)
  (h1 : num_cars = 15)
  (h2 : num_selections_per_car = 3)
  (h3 : num_cars_per_client = 3)
  (h4 : total_selections = num_cars * num_selections_per_car)
  (h5 : num_clients = total_selections / num_cars_per_client) :
  num_clients = 15 := 
by
  sorry

end NUMINAMATH_GPT_number_of_clients_l473_47307


namespace NUMINAMATH_GPT_find_special_three_digit_numbers_l473_47361

theorem find_special_three_digit_numbers :
  {A : ℕ | 100 ≤ A ∧ A < 1000 ∧ (A^2 % 1000 = A)} = {376, 625} :=
by
  sorry

end NUMINAMATH_GPT_find_special_three_digit_numbers_l473_47361


namespace NUMINAMATH_GPT_smallest_n_for_gcd_lcm_l473_47346

theorem smallest_n_for_gcd_lcm (n a b : ℕ) (h_gcd : Nat.gcd a b = 999) (h_lcm : Nat.lcm a b = Nat.factorial n) :
  n = 37 := sorry

end NUMINAMATH_GPT_smallest_n_for_gcd_lcm_l473_47346


namespace NUMINAMATH_GPT_leaks_drain_time_l473_47332

-- Definitions from conditions
def pump_rate : ℚ := 1 / 2 -- tanks per hour
def leak1_rate : ℚ := 1 / 6 -- tanks per hour
def leak2_rate : ℚ := 1 / 9 -- tanks per hour

-- Proof statement
theorem leaks_drain_time : (leak1_rate + leak2_rate)⁻¹ = 3.6 :=
by
  sorry

end NUMINAMATH_GPT_leaks_drain_time_l473_47332


namespace NUMINAMATH_GPT_room_length_l473_47339

theorem room_length (L : ℝ) (width : ℝ := 4) (total_cost : ℝ := 20900) (rate : ℝ := 950) :
  L * width = total_cost / rate → L = 5.5 :=
by
  sorry

end NUMINAMATH_GPT_room_length_l473_47339


namespace NUMINAMATH_GPT_max_comic_books_l473_47321

namespace JasmineComicBooks

-- Conditions
def total_money : ℝ := 12.50
def comic_book_cost : ℝ := 1.15

-- Statement of the theorem
theorem max_comic_books (n : ℕ) (h : n * comic_book_cost ≤ total_money) : n ≤ 10 := by
  sorry

end JasmineComicBooks

end NUMINAMATH_GPT_max_comic_books_l473_47321


namespace NUMINAMATH_GPT_trevor_eggs_l473_47328

theorem trevor_eggs :
  let gertrude := 4
  let blanche := 3
  let nancy := 2
  let martha := 2
  let ophelia := 5
  let penelope := 1
  let quinny := 3
  let dropped := 2
  let gifted := 3
  let total_collected := gertrude + blanche + nancy + martha + ophelia + penelope + quinny
  let remaining_after_drop := total_collected - dropped
  let final_eggs := remaining_after_drop - gifted
  final_eggs = 15 := by
    sorry

end NUMINAMATH_GPT_trevor_eggs_l473_47328


namespace NUMINAMATH_GPT_find_z_l473_47310

theorem find_z (z : ℝ) (v : ℝ × ℝ × ℝ) (u : ℝ × ℝ × ℝ)
  (h_v : v = (4, 1, z)) (h_u : u = (2, -3, 4))
  (h_eq : (4 * 2 + 1 * -3 + z * 4) / (2 * 2 + -3 * -3 + 4 * 4) = 5 / 29) :
  z = 0 :=
by
  sorry

end NUMINAMATH_GPT_find_z_l473_47310


namespace NUMINAMATH_GPT_largest_prime_factor_5985_l473_47337

theorem largest_prime_factor_5985 : ∃ p, Nat.Prime p ∧ p ∣ 5985 ∧ ∀ q, Nat.Prime q ∧ q ∣ 5985 → q ≤ p :=
sorry

end NUMINAMATH_GPT_largest_prime_factor_5985_l473_47337


namespace NUMINAMATH_GPT_swapped_two_digit_number_l473_47323

variable (a : ℕ)

theorem swapped_two_digit_number (h : a < 10) (sum_digits : ∃ t : ℕ, t + a = 13) : 
    ∃ n : ℕ, n = 9 * a + 13 :=
by
  sorry

end NUMINAMATH_GPT_swapped_two_digit_number_l473_47323


namespace NUMINAMATH_GPT_area_of_triangle_from_squares_l473_47348

theorem area_of_triangle_from_squares :
  ∃ (a b c : ℕ), (a = 15 ∧ b = 15 ∧ c = 6 ∧ (1/2 : ℚ) * a * c = 45) :=
by
  let a := 15
  let b := 15
  let c := 6
  have h1 : (1/2 : ℚ) * a * c = 45 := sorry
  exact ⟨a, b, c, ⟨rfl, rfl, rfl, h1⟩⟩

end NUMINAMATH_GPT_area_of_triangle_from_squares_l473_47348


namespace NUMINAMATH_GPT_Sophie_donuts_l473_47356

theorem Sophie_donuts 
  (boxes : ℕ)
  (donuts_per_box : ℕ)
  (boxes_given_mom : ℕ)
  (donuts_given_sister : ℕ)
  (h1 : boxes = 4)
  (h2 : donuts_per_box = 12)
  (h3 : boxes_given_mom = 1)
  (h4 : donuts_given_sister = 6) :
  (boxes * donuts_per_box) - (boxes_given_mom * donuts_per_box) - donuts_given_sister = 30 :=
by
  sorry

end NUMINAMATH_GPT_Sophie_donuts_l473_47356


namespace NUMINAMATH_GPT_first_place_friend_distance_friend_running_distance_l473_47350

theorem first_place_friend_distance (distance_mina_finish : ℕ) (halfway_condition : ∀ x, x = distance_mina_finish / 2) :
  (∃ y, y = distance_mina_finish / 2) :=
by
  sorry

-- Given conditions
def distance_mina_finish : ℕ := 200
noncomputable def first_place_friend_position := distance_mina_finish / 2

-- The theorem we need to prove
theorem friend_running_distance : first_place_friend_position = 100 :=
by
  sorry

end NUMINAMATH_GPT_first_place_friend_distance_friend_running_distance_l473_47350


namespace NUMINAMATH_GPT_max_log_sum_value_l473_47343

noncomputable def max_log_sum (x y : ℝ) (h1 : x > 0) (h2 : y > 0) (h3 : x + 4 * y = 40) : ℝ :=
  Real.log x + Real.log y

theorem max_log_sum_value : ∀ (x y : ℝ), x > 0 → y > 0 → x + 4 * y = 40 → max_log_sum x y sorry sorry sorry = 2 :=
by
  intro x y h1 h2 h3
  sorry

end NUMINAMATH_GPT_max_log_sum_value_l473_47343


namespace NUMINAMATH_GPT_geometric_sequence_sufficient_condition_l473_47398

theorem geometric_sequence_sufficient_condition 
  (a_1 : ℝ) (q : ℝ) (h_a1 : a_1 < 0) (h_q : 0 < q ∧ q < 1) :
  ∀ n : ℕ, n > 0 -> a_1 * q^(n-1) < a_1 * q^n :=
sorry

end NUMINAMATH_GPT_geometric_sequence_sufficient_condition_l473_47398


namespace NUMINAMATH_GPT_calculate_sum_and_difference_l473_47333

theorem calculate_sum_and_difference : 0.5 - 0.03 + 0.007 = 0.477 := sorry

end NUMINAMATH_GPT_calculate_sum_and_difference_l473_47333


namespace NUMINAMATH_GPT_initial_milk_quantity_l473_47374

theorem initial_milk_quantity 
  (milk_left_in_tank : ℕ) -- the remaining milk in the tank
  (pumping_rate : ℕ) -- the rate at which milk was pumped out
  (pumping_hours : ℕ) -- hours during which milk was pumped out
  (adding_rate : ℕ) -- the rate at which milk was added
  (adding_hours : ℕ) -- hours during which milk was added 
  (initial_milk : ℕ) -- initial milk collected
  (h1 : milk_left_in_tank = 28980) -- condition 3
  (h2 : pumping_rate = 2880) -- condition 1 (rate)
  (h3 : pumping_hours = 4) -- condition 1 (hours)
  (h4 : adding_rate = 1500) -- condition 2 (rate)
  (h5 : adding_hours = 7) -- condition 2 (hours)
  : initial_milk = 30000 :=
by
  sorry

end NUMINAMATH_GPT_initial_milk_quantity_l473_47374


namespace NUMINAMATH_GPT_intersection_complement_l473_47309

variable (U : Set ℕ) (A : Set ℕ) (B : Set ℕ)
variable (hU : U = {1, 2, 3, 4, 5}) (hA : A = {1, 2, 3}) (hB : B = {1, 4})

theorem intersection_complement :
  A ∩ (U \ B) = {2, 3} := by
  sorry

end NUMINAMATH_GPT_intersection_complement_l473_47309


namespace NUMINAMATH_GPT_opposite_of_negative_2023_l473_47353

theorem opposite_of_negative_2023 : -(-2023) = 2023 :=
by
  sorry

end NUMINAMATH_GPT_opposite_of_negative_2023_l473_47353


namespace NUMINAMATH_GPT_neg_abs_nonneg_l473_47327

theorem neg_abs_nonneg :
  (¬ ∀ x : ℝ, |x| ≥ 0) ↔ (∃ x : ℝ, |x| < 0) := by
  sorry

end NUMINAMATH_GPT_neg_abs_nonneg_l473_47327


namespace NUMINAMATH_GPT_sin_cos_from_tan_l473_47324

variable {α : Real} (hα : α > 0) -- Assume α is an acute angle

theorem sin_cos_from_tan (h : Real.tan α = 2) : 
  Real.sin α = 2 / Real.sqrt 5 ∧ Real.cos α = 1 / Real.sqrt 5 := 
by sorry

end NUMINAMATH_GPT_sin_cos_from_tan_l473_47324


namespace NUMINAMATH_GPT_radius_of_circle_l473_47344

theorem radius_of_circle (r : ℝ) (h : 6 * Real.pi * r + 6 = 2 * Real.pi * r^2) : 
  r = (3 + Real.sqrt 21) / 2 :=
by
  sorry

end NUMINAMATH_GPT_radius_of_circle_l473_47344


namespace NUMINAMATH_GPT_six_by_six_board_partition_l473_47390

theorem six_by_six_board_partition (P : Prop) (Q : Prop) 
(board : ℕ × ℕ) (domino : ℕ × ℕ) 
(h1 : board = (6, 6)) 
(h2 : domino = (2, 1)) 
(h3 : P → Q ∧ Q → P) :
  ∃ R₁ R₂ : ℕ × ℕ, (R₁ = (p, q) ∧ R₂ = (r, s) ∧ ((R₁.1 * R₁.2 + R₂.1 * R₂.2) = 36)) :=
sorry

end NUMINAMATH_GPT_six_by_six_board_partition_l473_47390


namespace NUMINAMATH_GPT_rectangular_solid_surface_area_l473_47329

-- Definitions based on conditions
def is_prime (n : ℕ) := n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

def rectangular_solid (a b c : ℕ) :=
  is_prime a ∧ is_prime b ∧ is_prime c ∧ a * b * c = 231

noncomputable def surface_area (a b c : ℕ) := 2 * (a * b + b * c + c * a)

-- Main theorem based on question and answer
theorem rectangular_solid_surface_area :
  ∃ (a b c : ℕ), rectangular_solid a b c ∧ surface_area a b c = 262 := by
  sorry

end NUMINAMATH_GPT_rectangular_solid_surface_area_l473_47329


namespace NUMINAMATH_GPT_hillary_climbing_rate_l473_47383

theorem hillary_climbing_rate :
  ∀ (H : ℕ) (Eddy_rate : ℕ) (Hillary_climb : ℕ) (Hillary_descend_rate : ℕ) (pass_time : ℕ) (start_to_summit : ℕ),
    Eddy_rate = 500 →
    Hillary_climb = 4000 →
    Hillary_descend_rate = 1000 →
    pass_time = 6 →
    start_to_summit = 5000 →
    (Hillary_climb + Eddy_rate * pass_time = Hillary_climb + (pass_time - Hillary_climb / H) * Hillary_descend_rate) →
    H = 800 :=
by
  intros H Eddy_rate Hillary_climb Hillary_descend_rate pass_time start_to_summit
  intro h1 h2 h3 h4 h5 h6
  sorry

end NUMINAMATH_GPT_hillary_climbing_rate_l473_47383


namespace NUMINAMATH_GPT_total_cost_shoes_and_jerseys_l473_47342

theorem total_cost_shoes_and_jerseys 
  (shoes : ℕ) (jerseys : ℕ) (cost_shoes : ℕ) (cost_jersey : ℕ) 
  (cost_total_shoes : ℕ) (cost_per_shoe : ℕ) (cost_per_jersey : ℕ) 
  (h1 : shoes = 6)
  (h2 : jerseys = 4) 
  (h3 : cost_per_jersey = cost_per_shoe / 4)
  (h4 : cost_total_shoes = 480)
  (h5 : cost_per_shoe = cost_total_shoes / shoes)
  (h6 : cost_per_jersey = cost_per_shoe / 4)
  (total_cost : ℕ) 
  (h7 : total_cost = cost_total_shoes + cost_per_jersey * jerseys) :
  total_cost = 560 :=
sorry

end NUMINAMATH_GPT_total_cost_shoes_and_jerseys_l473_47342


namespace NUMINAMATH_GPT_flat_fee_l473_47359

theorem flat_fee (f n : ℝ) (h1 : f + 3 * n = 215) (h2 : f + 6 * n = 385) : f = 45 :=
  sorry

end NUMINAMATH_GPT_flat_fee_l473_47359


namespace NUMINAMATH_GPT_find_m_n_l473_47334

theorem find_m_n :
  ∀ (m n : ℤ), (∀ x : ℤ, (x - 4) * (x + 8) = x^2 + m * x + n) → 
  (m = 4 ∧ n = -32) :=
by
  intros m n h
  let x := 0
  sorry

end NUMINAMATH_GPT_find_m_n_l473_47334


namespace NUMINAMATH_GPT_daily_shoppers_correct_l473_47382

noncomputable def daily_shoppers (P : ℝ) : Prop :=
  let weekly_taxes : ℝ := 6580
  let daily_taxes := weekly_taxes / 7
  let percent_taxes := 0.94
  percent_taxes * P = daily_taxes

theorem daily_shoppers_correct : ∃ P : ℝ, daily_shoppers P ∧ P = 1000 :=
by
  sorry

end NUMINAMATH_GPT_daily_shoppers_correct_l473_47382


namespace NUMINAMATH_GPT_smallest_solution_of_equation_l473_47355

theorem smallest_solution_of_equation :
  ∃ x : ℝ, (9 * x^2 - 45 * x + 50 = 0) ∧ (∀ y : ℝ, 9 * y^2 - 45 * y + 50 = 0 → x ≤ y) :=
sorry

end NUMINAMATH_GPT_smallest_solution_of_equation_l473_47355


namespace NUMINAMATH_GPT_modulo_arithmetic_l473_47325

theorem modulo_arithmetic :
  (222 * 15 - 35 * 9 + 2^3) % 18 = 17 :=
by
  sorry

end NUMINAMATH_GPT_modulo_arithmetic_l473_47325


namespace NUMINAMATH_GPT_cape_may_multiple_l473_47396

theorem cape_may_multiple :
  ∃ x : ℕ, 26 = x * 7 + 5 ∧ x = 3 :=
by
  sorry

end NUMINAMATH_GPT_cape_may_multiple_l473_47396


namespace NUMINAMATH_GPT_original_useful_item_is_pencil_l473_47365

def code_language (x : String) : String :=
  if x = "item" then "pencil"
  else if x = "pencil" then "mirror"
  else if x = "mirror" then "board"
  else x

theorem original_useful_item_is_pencil : 
  (code_language "item" = "pencil") ∧
  (code_language "pencil" = "mirror") ∧
  (code_language "mirror" = "board") ∧
  (code_language "item" = "pencil") ∧
  (code_language "pencil" = "mirror") ∧
  (code_language "mirror" = "board") 
  → "mirror" = "pencil" :=
by sorry

end NUMINAMATH_GPT_original_useful_item_is_pencil_l473_47365


namespace NUMINAMATH_GPT_total_workers_count_l473_47389

theorem total_workers_count 
  (W N : ℕ)
  (h1 : (W : ℝ) * 9000 = 7 * 12000 + N * 6000)
  (h2 : W = 7 + N) 
  : W = 14 :=
sorry

end NUMINAMATH_GPT_total_workers_count_l473_47389


namespace NUMINAMATH_GPT_sum_of_integers_is_23_l473_47379

theorem sum_of_integers_is_23
  (x y : ℕ) (x_pos : 0 < x) (y_pos : 0 < y) (h : x * y + x + y = 155) 
  (rel_prime : Nat.gcd x y = 1) (x_lt_30 : x < 30) (y_lt_30 : y < 30) :
  x + y = 23 :=
by
  sorry

end NUMINAMATH_GPT_sum_of_integers_is_23_l473_47379


namespace NUMINAMATH_GPT_part1_part2_l473_47312

noncomputable def triangle_area (A B C : ℝ) (a b c : ℝ) : ℝ :=
  1/2 * a * c * Real.sin B

theorem part1 
  (A B C : ℝ) (a b c : ℝ)
  (h₁ : A = π / 6)
  (h₂ : a = 2)
  (h₃ : 2 * a * c * Real.sin A + a^2 + c^2 - b^2 = 0) :
  triangle_area A B C a b c = Real.sqrt 3 :=
sorry

theorem part2 
  (A B C : ℝ) (a b c : ℝ)
  (h₁ : A = π / 6)
  (h₂ : a = 2)
  (h₃ : 2 * a * c * Real.sin A + a^2 + c^2 - b^2 = 0) :
  ∃ B, 
  (B = 2 * π / 3) ∧ (4 * Real.sin C^2 + 3 * Real.sin A^2 + 2) / (Real.sin B^2) = 5 :=
sorry

end NUMINAMATH_GPT_part1_part2_l473_47312


namespace NUMINAMATH_GPT_total_wash_time_l473_47399

theorem total_wash_time (clothes_time : ℕ) (towels_time : ℕ) (sheets_time : ℕ) (total_time : ℕ) 
  (h1 : clothes_time = 30) 
  (h2 : towels_time = 2 * clothes_time) 
  (h3 : sheets_time = towels_time - 15) 
  (h4 : total_time = clothes_time + towels_time + sheets_time) : 
  total_time = 135 := 
by 
  sorry

end NUMINAMATH_GPT_total_wash_time_l473_47399


namespace NUMINAMATH_GPT_sum_of_distinct_digits_base6_l473_47393

theorem sum_of_distinct_digits_base6 (A B C : ℕ) (hA : A < 6) (hB : B < 6) (hC : C < 6) 
  (h_distinct : A ≠ B ∧ B ≠ C ∧ A ≠ C)
  (h_first_col : C + C % 6 = 4)
  (h_second_col : B + B % 6 = C)
  (h_third_col : A + B % 6 = A) :
  A + B + C = 6 := by
  sorry

end NUMINAMATH_GPT_sum_of_distinct_digits_base6_l473_47393


namespace NUMINAMATH_GPT_total_amount_spent_l473_47330

-- Define the prices of the CDs
def price_life_journey : ℕ := 100
def price_day_life : ℕ := 50
def price_when_rescind : ℕ := 85

-- Define the discounted price for The Life Journey CD
def discount_life_journey : ℕ := 20 -- 20% discount equivalent to $20
def discounted_price_life_journey : ℕ := price_life_journey - discount_life_journey

-- Define the number of CDs bought
def num_life_journey : ℕ := 3
def num_day_life : ℕ := 4
def num_when_rescind : ℕ := 2

-- Define the function to calculate money spent on each type with offers in consideration
def cost_life_journey : ℕ := num_life_journey * discounted_price_life_journey
def cost_day_life : ℕ := (num_day_life / 2) * price_day_life -- Buy one get one free offer
def cost_when_rescind : ℕ := num_when_rescind * price_when_rescind

-- Calculate the total cost
def total_cost := cost_life_journey + cost_day_life + cost_when_rescind

-- Define Lean theorem to prove the total cost
theorem total_amount_spent : total_cost = 510 :=
  by
    -- Skipping the actual proof as the prompt specifies
    sorry

end NUMINAMATH_GPT_total_amount_spent_l473_47330


namespace NUMINAMATH_GPT_cos_double_angle_l473_47341

theorem cos_double_angle (theta : ℝ) (h : Real.cos theta = 1/4) : Real.cos (2 * theta) = -7/8 :=
by
  sorry

end NUMINAMATH_GPT_cos_double_angle_l473_47341


namespace NUMINAMATH_GPT_remainder_of_E_div_88_l473_47362

-- Define the given expression E and the binomial coefficient 
noncomputable def E : ℤ :=
  1 - 90 * Nat.choose 10 1 + 90 ^ 2 * Nat.choose 10 2 - 90 ^ 3 * Nat.choose 10 3 + 
  90 ^ 4 * Nat.choose 10 4 - 90 ^ 5 * Nat.choose 10 5 + 90 ^ 6 * Nat.choose 10 6 - 
  90 ^ 7 * Nat.choose 10 7 + 90 ^ 8 * Nat.choose 10 8 - 90 ^ 9 * Nat.choose 10 9 + 
  90 ^ 10 * Nat.choose 10 10

-- The theorem that we need to prove
theorem remainder_of_E_div_88 : E % 88 = 1 := by
  sorry

end NUMINAMATH_GPT_remainder_of_E_div_88_l473_47362


namespace NUMINAMATH_GPT_surface_area_correct_l473_47301

def w := 3 -- width in cm
def l := 4 -- length in cm
def h := 5 -- height in cm

def surface_area : Nat := 
  2 * (h * w) + 2 * (l * w) + 2 * (l * h)

theorem surface_area_correct : surface_area = 94 := 
  by
    sorry

end NUMINAMATH_GPT_surface_area_correct_l473_47301


namespace NUMINAMATH_GPT_Cathy_wins_l473_47369

theorem Cathy_wins (n k : ℕ) (hn : n > 0) (hk : k > 0) : (∃ box_count : ℕ, box_count = 1) :=
  if h : n ≤ 2^(k-1) then
    sorry
  else
    sorry

end NUMINAMATH_GPT_Cathy_wins_l473_47369


namespace NUMINAMATH_GPT_inequality_pow_gt_linear_l473_47311

theorem inequality_pow_gt_linear {a : ℝ} (n : ℕ) (h₁ : a > -1) (h₂ : a ≠ 0) (h₃ : n ≥ 2) :
  (1 + a:ℝ)^n > 1 + n * a :=
sorry

end NUMINAMATH_GPT_inequality_pow_gt_linear_l473_47311


namespace NUMINAMATH_GPT_triangle_is_isosceles_l473_47363

open Real

variables (α β γ : ℝ) (a b : ℝ)

theorem triangle_is_isosceles
(h1 : a + b = tan (γ / 2) * (a * tan α + b * tan β)) :
α = β :=
by
  sorry

end NUMINAMATH_GPT_triangle_is_isosceles_l473_47363


namespace NUMINAMATH_GPT_cost_price_of_computer_table_l473_47331

variable (C : ℝ) (SP : ℝ)
variable (h1 : SP = 5400)
variable (h2 : SP = C * 1.32)

theorem cost_price_of_computer_table : C = 5400 / 1.32 :=
by
  -- We are required to prove C = 5400 / 1.32
  sorry

end NUMINAMATH_GPT_cost_price_of_computer_table_l473_47331


namespace NUMINAMATH_GPT_find_number_l473_47316

theorem find_number (x : ℝ) (h : x / 3 = x - 3) : x = 4.5 := 
sorry

end NUMINAMATH_GPT_find_number_l473_47316


namespace NUMINAMATH_GPT_multiplier_for_deans_height_l473_47364

theorem multiplier_for_deans_height (h_R : ℕ) (h_R_eq : h_R = 13) (d : ℕ) (d_eq : d = 255) (h_D : ℕ) (h_D_eq : h_D = h_R + 4) : 
  d / h_D = 15 := by
  sorry

end NUMINAMATH_GPT_multiplier_for_deans_height_l473_47364


namespace NUMINAMATH_GPT_max_plus_min_l473_47318

noncomputable def f : ℝ → ℝ := sorry

axiom condition1 (x₁ x₂ : ℝ) : f (x₁ + x₂) = f x₁ + f x₂ - 2016
axiom condition2 (x : ℝ) : x > 0 → f x > 2016

theorem max_plus_min (M N : ℝ) (hM : M = f 2016) (hN : N = f (-2016)) : M + N = 4032 :=
by
  sorry

end NUMINAMATH_GPT_max_plus_min_l473_47318


namespace NUMINAMATH_GPT_tangent_points_l473_47300

noncomputable def f (x : ℝ) : ℝ := x^3 + 1
def P : ℝ × ℝ := (-2, 1)
noncomputable def f' (x : ℝ) : ℝ := 3 * x^2

theorem tangent_points (x0 : ℝ) (y0 : ℝ) (hP : P = (-2, 1)) (hf : y0 = f x0) :
  (3 * x0^2 = (y0 - 1) / (x0 + 2)) → (x0 = 0 ∨ x0 = -3) :=
by
  sorry

end NUMINAMATH_GPT_tangent_points_l473_47300


namespace NUMINAMATH_GPT_equilateral_triangle_area_percentage_l473_47336

noncomputable def percentage_area_of_triangle_in_pentagon (s : ℝ) : ℝ :=
  ((4 * Real.sqrt 3 - 3) / 13) * 100

theorem equilateral_triangle_area_percentage
  (s : ℝ) :
  let pentagon_area := s^2 * (1 + Real.sqrt 3 / 4)
  let triangle_area := (Real.sqrt 3 / 4) * s^2
  (triangle_area / pentagon_area) * 100 = percentage_area_of_triangle_in_pentagon s :=
by
  sorry

end NUMINAMATH_GPT_equilateral_triangle_area_percentage_l473_47336


namespace NUMINAMATH_GPT_ice_cream_depth_l473_47349

noncomputable def volume_sphere (r : ℝ) := (4/3) * Real.pi * r^3
noncomputable def volume_cylinder (r h : ℝ) := Real.pi * r^2 * h

theorem ice_cream_depth
  (radius_sphere : ℝ)
  (radius_cylinder : ℝ)
  (density_constancy : volume_sphere radius_sphere = volume_cylinder radius_cylinder (h : ℝ)) :
  h = 9 / 25 := by
  sorry

end NUMINAMATH_GPT_ice_cream_depth_l473_47349


namespace NUMINAMATH_GPT_symmetric_angle_set_l473_47358

theorem symmetric_angle_set (α β : ℝ) (k : ℤ) 
  (h1 : β = 2 * (k : ℝ) * Real.pi + Real.pi / 12)
  (h2 : α = -Real.pi / 3)
  (symmetric : α + β = -Real.pi / 4) :
  ∃ k : ℤ, β = 2 * (k : ℝ) * Real.pi + Real.pi / 12 :=
sorry

end NUMINAMATH_GPT_symmetric_angle_set_l473_47358


namespace NUMINAMATH_GPT_canoe_stream_speed_l473_47354

theorem canoe_stream_speed (C S : ℝ) (h1 : C - S = 9) (h2 : C + S = 12) : S = 1.5 :=
by
  sorry

end NUMINAMATH_GPT_canoe_stream_speed_l473_47354


namespace NUMINAMATH_GPT_values_of_x0_l473_47340

noncomputable def x_seq (x_0 : ℝ) (n : ℕ) : ℝ :=
  match n with
  | 0 => x_0
  | n + 1 => if 3 * (x_seq x_0 n) < 1 then 3 * (x_seq x_0 n)
             else if 3 * (x_seq x_0 n) < 2 then 3 * (x_seq x_0 n) - 1
             else 3 * (x_seq x_0 n) - 2

theorem values_of_x0 (x_0 : ℝ) (h : 0 ≤ x_0 ∧ x_0 < 1) :
  (∃! x_0, x_0 = x_seq x_0 6) → (x_seq x_0 6 = x_0) :=
  sorry

end NUMINAMATH_GPT_values_of_x0_l473_47340


namespace NUMINAMATH_GPT_mandy_yoga_time_l473_47351

theorem mandy_yoga_time (G B Y : ℕ) (h1 : 2 * B = 3 * G) (h2 : 3 * Y = 2 * (G + B)) (h3 : Y = 30) : Y = 30 := by
  sorry

end NUMINAMATH_GPT_mandy_yoga_time_l473_47351


namespace NUMINAMATH_GPT_buttons_ratio_l473_47338

theorem buttons_ratio
  (initial_buttons : ℕ)
  (shane_multiplier : ℕ)
  (final_buttons : ℕ)
  (total_buttons_after_shane : ℕ) :
  initial_buttons = 14 →
  shane_multiplier = 3 →
  final_buttons = 28 →
  total_buttons_after_shane = initial_buttons + shane_multiplier * initial_buttons →
  (total_buttons_after_shane - final_buttons) / total_buttons_after_shane = 1 / 2 :=
by
  intros
  sorry

end NUMINAMATH_GPT_buttons_ratio_l473_47338


namespace NUMINAMATH_GPT_find_x_eq_e_l473_47347

noncomputable def f (x : ℝ) : ℝ := x + x * (Real.log x) ^ 2

noncomputable def f' (x : ℝ) : ℝ :=
  1 + (Real.log x) ^ 2 + 2 * Real.log x

theorem find_x_eq_e : ∃ (x : ℝ), (x * f' x = 2 * f x) ∧ (x = Real.exp 1) :=
by
  sorry

end NUMINAMATH_GPT_find_x_eq_e_l473_47347


namespace NUMINAMATH_GPT_solve_equation_one_solve_equation_two_l473_47319

theorem solve_equation_one (x : ℝ) : 3 * x + 7 = 32 - 2 * x → x = 5 :=
by
  intro h
  sorry

theorem solve_equation_two (x : ℝ) : (2 * x - 3) / 5 = (3 * x - 1) / 2 + 1 → x = -1 :=
by
  intro h
  sorry

end NUMINAMATH_GPT_solve_equation_one_solve_equation_two_l473_47319


namespace NUMINAMATH_GPT_maximum_value_of_f_on_interval_l473_47387

noncomputable def f (x : ℝ) : ℝ := x^3 + 3 * x^2 + 3

theorem maximum_value_of_f_on_interval :
  (∀ x ∈ Set.Icc (-3 : ℝ) 3, f x ≥ 3) →
  ∃ x ∈ Set.Icc (-3 : ℝ) 3, f x = 57 :=
by
  sorry

end NUMINAMATH_GPT_maximum_value_of_f_on_interval_l473_47387


namespace NUMINAMATH_GPT_chocolates_initial_count_l473_47378

theorem chocolates_initial_count (remaining_chocolates: ℕ) 
    (daily_percentage: ℝ) (days: ℕ) 
    (final_chocolates: ℝ) 
    (remaining_fraction_proof: remaining_fraction = 0.7) 
    (days_proof: days = 3) 
    (final_chocolates_proof: final_chocolates = 28): 
    (remaining_fraction^days * (initial_chocolates:ℝ) = final_chocolates) → 
    (initial_chocolates = 82) := 
by 
  sorry

end NUMINAMATH_GPT_chocolates_initial_count_l473_47378


namespace NUMINAMATH_GPT_pat_earns_per_photo_l473_47345

-- Defining conditions
def minutes_per_shark := 10
def fuel_cost_per_hour := 50
def hunting_hours := 5
def expected_profit := 200

-- Defining intermediate calculations based on the conditions
def sharks_per_hour := 60 / minutes_per_shark
def total_sharks := sharks_per_hour * hunting_hours
def total_fuel_cost := fuel_cost_per_hour * hunting_hours
def total_earnings := expected_profit + total_fuel_cost
def earnings_per_photo := total_earnings / total_sharks

-- Main theorem: Prove that Pat earns $15 for each photo
theorem pat_earns_per_photo : earnings_per_photo = 15 := by
  -- The proof would be here
  sorry

end NUMINAMATH_GPT_pat_earns_per_photo_l473_47345


namespace NUMINAMATH_GPT_torn_out_sheets_count_l473_47381

theorem torn_out_sheets_count :
  ∃ (sheets : ℕ), (first_page = 185 ∧
                   last_page = 518 ∧
                   pages_torn_out = last_page - first_page + 1 ∧ 
                   sheets = pages_torn_out / 2 ∧
                   sheets = 167) :=
by
  sorry

end NUMINAMATH_GPT_torn_out_sheets_count_l473_47381


namespace NUMINAMATH_GPT_max_beds_120_l473_47315

/-- The dimensions of the park. --/
def park_length : ℕ := 60
def park_width : ℕ := 30

/-- The dimensions of each flower bed. --/
def bed_length : ℕ := 3
def bed_width : ℕ := 5

/-- The available fencing length. --/
def total_fencing : ℕ := 2400

/-- Calculate the largest number of flower beds that can be created. --/
def max_flower_beds (park_length park_width bed_length bed_width total_fencing : ℕ) : ℕ := 
  let n := park_width / bed_width  -- number of beds per column
  let m := park_length / bed_length  -- number of beds per row
  let vertical_fencing := bed_width * (n - 1) * m
  let horizontal_fencing := bed_length * (m - 1) * n
  if vertical_fencing + horizontal_fencing <= total_fencing then n * m else 0

theorem max_beds_120 : max_flower_beds 60 30 3 5 2400 = 120 := by
  unfold max_flower_beds
  rfl

end NUMINAMATH_GPT_max_beds_120_l473_47315


namespace NUMINAMATH_GPT_fraction_of_earth_surface_humans_can_inhabit_l473_47352

theorem fraction_of_earth_surface_humans_can_inhabit :
  (1 / 3) * (2 / 3) = (2 / 9) :=
by
  sorry

end NUMINAMATH_GPT_fraction_of_earth_surface_humans_can_inhabit_l473_47352


namespace NUMINAMATH_GPT_linear_function_through_origin_l473_47306

theorem linear_function_through_origin (k : ℝ) (h : ∃ x y : ℝ, (x = 0 ∧ y = 0) ∧ y = (k - 2) * x + (k^2 - 4)) : k = -2 :=
by
  sorry

end NUMINAMATH_GPT_linear_function_through_origin_l473_47306


namespace NUMINAMATH_GPT_velocity_at_3_velocity_at_4_l473_47317

-- Define the distance as a function of time
def s (t : ℝ) : ℝ := 3 * t^2 + 2 * t

-- Define the velocity as the derivative of the distance
noncomputable def v (t : ℝ) : ℝ := deriv s t

theorem velocity_at_3 : v 3 = 20 :=
by
  sorry

theorem velocity_at_4 : v 4 = 26 :=
by
  sorry

end NUMINAMATH_GPT_velocity_at_3_velocity_at_4_l473_47317


namespace NUMINAMATH_GPT_find_A_l473_47305

theorem find_A (A B C : ℝ) :
  (∀ x : ℝ, x^3 - 2 * x ^ 2 - 13 * x + 10 ≠ 0 → 1 / (x ^ 3 - 2 * x ^ 2 - 13 * x + 10) = A / (x + 2) + B / (x - 1) + C / (x - 1) ^ 2)
  → A = 1 / 9 := 
sorry

end NUMINAMATH_GPT_find_A_l473_47305


namespace NUMINAMATH_GPT_sequences_correct_l473_47384

def arithmetic_sequence (a b c : ℕ) : Prop :=
  2 * b = a + c

def geometric_sequence (b a₁ b₁ : ℕ) : Prop :=
  a₁ * a₁ = b * b₁

noncomputable def sequence_a (n : ℕ) :=
  (n * (n + 1)) / 2

noncomputable def sequence_b (n : ℕ) :=
  ((n + 1) * (n + 1)) / 2

theorem sequences_correct :
  (∀ n : ℕ,
    n ≥ 1 →
    arithmetic_sequence (sequence_a n) (sequence_b n) (sequence_a (n + 1)) ∧
    geometric_sequence (sequence_b n) (sequence_a (n + 1)) (sequence_b (n + 1))) ∧
  (sequence_a 1 = 1) ∧
  (sequence_b 1 = 2) ∧
  (sequence_a 2 = 3) :=
by
  sorry

end NUMINAMATH_GPT_sequences_correct_l473_47384


namespace NUMINAMATH_GPT_general_formula_arithmetic_sequence_l473_47335

variable (a : ℕ → ℤ)

def isArithmeticSequence (a : ℕ → ℤ) :=
  ∃ d : ℤ, ∀ n : ℕ, a (n + 1) = a n + d

theorem general_formula_arithmetic_sequence :
  isArithmeticSequence a →
  a 5 = 9 →
  a 1 + a 7 = 14 →
  ∀ n : ℕ, a n = 2 * n - 1 :=
by
  intros h_seq h_a5 h_a17
  sorry

end NUMINAMATH_GPT_general_formula_arithmetic_sequence_l473_47335

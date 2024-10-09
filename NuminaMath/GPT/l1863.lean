import Mathlib

namespace sine_triangle_sides_l1863_186305

variable {α β γ : ℝ}

-- Given conditions: α, β, γ are angles of a triangle.
def is_triangle_angles (α β γ : ℝ) : Prop :=
  α + β + γ = Real.pi ∧ 0 < α ∧ α < Real.pi ∧
  0 < β ∧ β < Real.pi ∧ 0 < γ ∧ γ < Real.pi

-- The proof statement: Prove that there exists a triangle with sides sin α, sin β, sin γ
theorem sine_triangle_sides (h : is_triangle_angles α β γ) :
  ∃ (x y z : ℝ), x = Real.sin α ∧ y = Real.sin β ∧ z = Real.sin γ ∧
  (x + y > z) ∧ (x + z > y) ∧ (y + z > x) := sorry

end sine_triangle_sides_l1863_186305


namespace parabola_focus_l1863_186318

theorem parabola_focus :
  ∃ f, (∀ x : ℝ, (x^2 + (2*x^2 - f)^2 = (2*x^2 - (-f + 1/4))^2)) ∧ f = 1/8 :=
by
  sorry

end parabola_focus_l1863_186318


namespace coordinates_of_D_l1863_186304

-- Definitions of the points and translation conditions
def A : (ℝ × ℝ) := (-1, 4)
def B : (ℝ × ℝ) := (-4, -1)
def C : (ℝ × ℝ) := (4, 7)

theorem coordinates_of_D :
  ∃ (D : ℝ × ℝ), D = (1, 2) ∧
  ∀ (translate : ℝ × ℝ), translate = (C.1 - A.1, C.2 - A.2) → 
  D = (B.1 + translate.1, B.2 + translate.2) :=
by
  sorry

end coordinates_of_D_l1863_186304


namespace find_m_l1863_186340

-- Definitions for the lines and the condition of parallelism
def line1 (m : ℝ) (x y : ℝ): Prop := x + m * y + 6 = 0
def line2 (m : ℝ) (x y : ℝ): Prop := 3 * x + (m - 2) * y + 2 * m = 0

-- Condition for lines being parallel
def parallel_lines (m : ℝ) : Prop := 1 * (m - 2) - 3 * m = 0

-- Main formal statement
theorem find_m (m : ℝ) (h1 : ∀ x y, line1 m x y)
                (h2 : ∀ x y, line2 m x y)
                (h_parallel : parallel_lines m) : m = -1 :=
sorry

end find_m_l1863_186340


namespace water_required_to_prepare_saline_solution_l1863_186326

theorem water_required_to_prepare_saline_solution (water_ratio : ℝ) (required_volume : ℝ) : 
  water_ratio = 3 / 8 ∧ required_volume = 0.64 → required_volume * water_ratio = 0.24 :=
by
  sorry

end water_required_to_prepare_saline_solution_l1863_186326


namespace largest_four_digit_sum_20_l1863_186333

-- Defining the four-digit number and conditions.
def is_four_digit_number (n : ℕ) : Prop :=
  n >= 1000 ∧ n < 10000

def digits_sum_to (n : ℕ) (s : ℕ) : Prop :=
  ∃ a b c d : ℕ, a + b + c + d = s ∧ n = 1000 * a + 100 * b + 10 * c + d

-- Proof problem statement.
theorem largest_four_digit_sum_20 : ∃ n, is_four_digit_number n ∧ digits_sum_to n 20 ∧ ∀ m, is_four_digit_number m ∧ digits_sum_to m 20 → m ≤ n :=
  sorry

end largest_four_digit_sum_20_l1863_186333


namespace total_hangers_is_65_l1863_186361

noncomputable def calculate_hangers_total : ℕ :=
  let pink := 7
  let green := 4
  let blue := green - 1
  let yellow := blue - 1
  let orange := 2 * (pink + green)
  let purple := (blue - yellow) + 3
  let red := (pink + green + blue) / 3
  let brown := 3 * red + 1
  let gray := (3 * purple) / 5
  pink + green + blue + yellow + orange + purple + red + brown + gray

theorem total_hangers_is_65 : calculate_hangers_total = 65 := 
by 
  sorry

end total_hangers_is_65_l1863_186361


namespace convert_base10_to_base9_l1863_186347

theorem convert_base10_to_base9 : 
  (2 * 9^3 + 6 * 9^2 + 7 * 9^1 + 7 * 9^0) = 2014 :=
by
  sorry

end convert_base10_to_base9_l1863_186347


namespace final_temperature_is_correct_l1863_186351

def initial_temperature : ℝ := 40
def after_jerry_temperature (T : ℝ) : ℝ := 2 * T
def after_dad_temperature (T : ℝ) : ℝ := T - 30
def after_mother_temperature (T : ℝ) : ℝ := T - 0.30 * T
def after_sister_temperature (T : ℝ) : ℝ := T + 24

theorem final_temperature_is_correct :
  after_sister_temperature (after_mother_temperature (after_dad_temperature (after_jerry_temperature initial_temperature))) = 59 :=
sorry

end final_temperature_is_correct_l1863_186351


namespace hindi_books_count_l1863_186334

theorem hindi_books_count (H : ℕ) (h1 : 22 = 22) (h2 : Nat.choose 23 H = 1771) : H = 3 :=
sorry

end hindi_books_count_l1863_186334


namespace triangle_side_c_l1863_186354

noncomputable def area_of_triangle (a b C : ℝ) : ℝ :=
  0.5 * a * b * Real.sin C

noncomputable def law_of_cosines (a b C : ℝ) : ℝ :=
  Real.sqrt (a^2 + b^2 - 2 * a * b * Real.cos C)

theorem triangle_side_c (a b C : ℝ) (h1 : a = 3) (h2 : C = Real.pi * 2 / 3) (h3 : area_of_triangle a b C = 15 * Real.sqrt 3 / 4) : law_of_cosines a b C = 2 :=
by
  sorry

end triangle_side_c_l1863_186354


namespace petya_cannot_have_equal_coins_l1863_186311

theorem petya_cannot_have_equal_coins
  (transact : ℕ → ℕ)
  (initial_two_kopeck : ℕ)
  (total_operations : ℕ)
  (insertion_machine : ℕ)
  (by_insert_two : ℕ)
  (by_insert_ten : ℕ)
  (odd : ℕ)
  :
  (initial_two_kopeck = 1) ∧ 
  (by_insert_two = 5) ∧ 
  (by_insert_ten = 5) ∧
  (∀ n, transact n = 1 + 4 * n) →
  (odd % 2 = 1) →
  (total_operations = transact insertion_machine) →
  (total_operations % 2 = 1) →
  (∀ x y, (x + y = total_operations) → (x = y) → False) :=
sorry

end petya_cannot_have_equal_coins_l1863_186311


namespace fisherman_bass_count_l1863_186365

theorem fisherman_bass_count (B T G : ℕ) (h1 : T = B / 4) (h2 : G = 2 * B) (h3 : B + T + G = 104) : B = 32 :=
by
  sorry

end fisherman_bass_count_l1863_186365


namespace quadratic_inequality_solution_l1863_186382

theorem quadratic_inequality_solution (m : ℝ) : 
  (∃ x : ℝ, x^2 - 2 * x + m ≤ 0) ↔ m ≤ 1 :=
sorry

end quadratic_inequality_solution_l1863_186382


namespace order_of_abc_l1863_186360

noncomputable def a := Real.log 1.2
noncomputable def b := (11 / 10) - (10 / 11)
noncomputable def c := 1 / (5 * Real.exp 0.1)

theorem order_of_abc : b > a ∧ a > c :=
by
  sorry

end order_of_abc_l1863_186360


namespace new_probability_of_blue_ball_l1863_186325

theorem new_probability_of_blue_ball 
  (initial_total_balls : ℕ) (initial_blue_balls : ℕ) (removed_blue_balls : ℕ) :
  initial_total_balls = 18 →
  initial_blue_balls = 6 →
  removed_blue_balls = 3 →
  (initial_blue_balls - removed_blue_balls) / (initial_total_balls - removed_blue_balls) = 1 / 5 :=
by
  sorry

end new_probability_of_blue_ball_l1863_186325


namespace sum_sequence_formula_l1863_186339

theorem sum_sequence_formula (a : ℕ → ℚ) (S : ℕ → ℚ) :
  (∀ n : ℕ, n > 0 → S n = n^2 * a n) ∧ a 1 = 1 →
  ∀ n : ℕ, n > 0 → S n = 2 * n / (n + 1) :=
by sorry

end sum_sequence_formula_l1863_186339


namespace triangle_inequality_l1863_186397

theorem triangle_inequality (a b c : ℝ) (h1 : a + b > c) (h2 : b + c > a) (h3 : c + a > b) :
  a^2 * (b + c - a) + b^2 * (c + a - b) + c^2 * (a + b - c) ≤ 3 * a * b * c :=
sorry

end triangle_inequality_l1863_186397


namespace find_number_of_rabbits_l1863_186350

variable (R P : ℕ)

theorem find_number_of_rabbits (h1 : R + P = 60) (h2 : 4 * R + 2 * P = 192) : R = 36 := 
by
  sorry

end find_number_of_rabbits_l1863_186350


namespace increase_in_average_weight_l1863_186345

theorem increase_in_average_weight :
  let initial_group_size := 6
  let initial_weight := 65
  let new_weight := 74
  let initial_avg_weight := A
  (new_weight - initial_weight) / initial_group_size = 1.5 := by
    sorry

end increase_in_average_weight_l1863_186345


namespace solve_system_of_equations_l1863_186343

theorem solve_system_of_equations :
  ∃ (x y z : ℤ), (x + y + z = 6) ∧ (x + y * z = 7) ∧ 
  ((x = 7 ∧ y = 0 ∧ z = -1) ∨ 
   (x = 7 ∧ y = -1 ∧ z = 0) ∨ 
   (x = 1 ∧ y = 3 ∧ z = 2) ∨ 
   (x = 1 ∧ y = 2 ∧ z = 3)) :=
sorry

end solve_system_of_equations_l1863_186343


namespace average_speed_of_train_l1863_186380

-- Condition: Distance traveled is 42 meters
def distance : ℕ := 42

-- Condition: Time taken is 6 seconds
def time : ℕ := 6

-- Average speed computation
theorem average_speed_of_train : distance / time = 7 := by
  -- Left to the prover
  sorry

end average_speed_of_train_l1863_186380


namespace repeating_decimals_difference_l1863_186315

theorem repeating_decimals_difference :
  let x := 234 / 999
  let y := 567 / 999
  let z := 891 / 999
  x - y - z = -408 / 333 :=
by
  sorry

end repeating_decimals_difference_l1863_186315


namespace flowmaster_pump_output_l1863_186384

theorem flowmaster_pump_output (hourly_rate : ℕ) (time_minutes : ℕ) (output_gallons : ℕ) 
  (h1 : hourly_rate = 600) 
  (h2 : time_minutes = 30) 
  (h3 : output_gallons = (hourly_rate * time_minutes) / 60) : 
  output_gallons = 300 :=
by sorry

end flowmaster_pump_output_l1863_186384


namespace ratio_of_areas_of_concentric_circles_l1863_186399

theorem ratio_of_areas_of_concentric_circles
  (C1 C2 : ℝ) -- circumferences of the smaller and larger circle
  (h : (1 / 6) * C1 = (2 / 15) * C2) -- condition given: 60-degree arc on the smaller circle equals 48-degree arc on the larger circle
  : (C1 / C2)^2 = (16 / 25) := by
  sorry

end ratio_of_areas_of_concentric_circles_l1863_186399


namespace problem_statement_l1863_186314

noncomputable def g (x : ℝ) : ℝ :=
  sorry

theorem problem_statement : (∀ x : ℝ, g x + 3 * g (1 - x) = 4 * x^3 + x^2) → g 3 = -201 / 8 :=
by
  intro h
  sorry

end problem_statement_l1863_186314


namespace sum_of_number_and_reverse_l1863_186307

theorem sum_of_number_and_reverse (a b : Nat) (h1 : 1 ≤ a ∧ a ≤ 9) (h2 : 0 ≤ b ∧ b ≤ 9)
  (h3 : (10 * a + b) - (10 * b + a) = 7 * (a + b)) :
  (10 * a + b) + (10 * b + a) = 99 := by
  sorry

end sum_of_number_and_reverse_l1863_186307


namespace jan_skips_in_5_minutes_l1863_186309

theorem jan_skips_in_5_minutes 
  (original_speed : ℕ)
  (time_in_minutes : ℕ)
  (doubled : ℕ)
  (new_speed : ℕ)
  (skips_in_5_minutes : ℕ) : 
  original_speed = 70 →
  doubled = 2 →
  new_speed = original_speed * doubled →
  time_in_minutes = 5 →
  skips_in_5_minutes = new_speed * time_in_minutes →
  skips_in_5_minutes = 700 :=
by
  intros 
  sorry

end jan_skips_in_5_minutes_l1863_186309


namespace option_A_cannot_be_true_l1863_186355

variable (a : ℕ → ℝ) (S : ℕ → ℝ)
variable (r : ℝ) -- common ratio for the geometric sequence
variable (n : ℕ) -- number of terms

def is_geometric_sequence (a : ℕ → ℝ) (r : ℝ) :=
  ∀ n, a (n + 1) = r * a n

def sum_of_geometric_sequence (a : ℕ → ℝ) (S : ℕ → ℝ) :=
  S 0 = a 0 ∧ ∀ n, S (n + 1) = S n + a (n + 1)

theorem option_A_cannot_be_true
  (h_geom : is_geometric_sequence a r)
  (h_sum : sum_of_geometric_sequence a S) :
  a 2016 * (S 2016 - S 2015) ≠ 0 :=
sorry

end option_A_cannot_be_true_l1863_186355


namespace radius_of_circle_l1863_186346

noncomputable def circle_area (r : ℝ) : ℝ := Real.pi * r^2
noncomputable def circle_circumference (r : ℝ) : ℝ := 2 * Real.pi * r

theorem radius_of_circle : 
  ∃ r : ℝ, circle_area r = circle_circumference r → r = 2 := 
by 
  sorry

end radius_of_circle_l1863_186346


namespace distribution_ways_l1863_186357

theorem distribution_ways :
  let friends := 12
  let problems := 6
  (friends ^ problems = 2985984) :=
by
  sorry

end distribution_ways_l1863_186357


namespace divide_condition_l1863_186337

theorem divide_condition (a b : ℕ) (ha : 0 < a) (hb : 0 < b) : 
  ∃ n : ℕ, 0 < n ∧ a ∣ (b^n - n) :=
by
  sorry

end divide_condition_l1863_186337


namespace area_triangle_tangent_circles_l1863_186366

theorem area_triangle_tangent_circles :
  ∃ (A B C : Type) (radius1 radius2 : ℝ) 
    (tangent1 tangent2 : ℝ → ℝ → Prop)
    (congruent_sides : ℝ → Prop),
    radius1 = 1 ∧ radius2 = 2 ∧
    (∀ x y, tangent1 x y) ∧ (∀ x y, tangent2 x y) ∧
    congruent_sides 1 ∧ congruent_sides 2 ∧
    ∃ (area : ℝ), area = 16 * Real.sqrt 2 :=
by
  -- This is where the proof would be written
  sorry

end area_triangle_tangent_circles_l1863_186366


namespace interval_contains_n_l1863_186374

theorem interval_contains_n (n : ℕ) (h1 : n < 1000) (h2 : n ∣ 999) (h3 : n + 6 ∣ 99) : 1 ≤ n ∧ n ≤ 250 := 
sorry

end interval_contains_n_l1863_186374


namespace sqrt_inequality_l1863_186336

theorem sqrt_inequality (n : ℕ) : 
  (n ≥ 0) → (Real.sqrt (n + 2) - Real.sqrt (n + 1) ≤ Real.sqrt (n + 1) - Real.sqrt n) := 
by
  intro h
  sorry

end sqrt_inequality_l1863_186336


namespace average_salary_is_8000_l1863_186383

def salary_A : ℕ := 8000
def salary_B : ℕ := 5000
def salary_C : ℕ := 11000
def salary_D : ℕ := 7000
def salary_E : ℕ := 9000

def total_salary : ℕ := salary_A + salary_B + salary_C + salary_D + salary_E
def num_people : ℕ := 5

def average_salary : ℕ := total_salary / num_people

theorem average_salary_is_8000 : average_salary = 8000 := by
  sorry

end average_salary_is_8000_l1863_186383


namespace harold_grocery_expense_l1863_186323

theorem harold_grocery_expense:
  ∀ (income rent car_payment savings utilities remaining groceries : ℝ),
    income = 2500 →
    rent = 700 →
    car_payment = 300 →
    utilities = 0.5 * car_payment →
    remaining = income - rent - car_payment - utilities →
    savings = 0.5 * remaining →
    (remaining - savings) = 650 →
    groceries = (remaining - 650) →
    groceries = 50 :=
by
  intros income rent car_payment savings utilities remaining groceries
  intro h_income
  intro h_rent
  intro h_car_payment
  intro h_utilities
  intro h_remaining
  intro h_savings
  intro h_final_remaining
  intro h_groceries
  sorry

end harold_grocery_expense_l1863_186323


namespace total_cost_of_items_is_correct_l1863_186331

theorem total_cost_of_items_is_correct :
  ∀ (M R F : ℝ),
  (10 * M = 24 * R) →
  (F = 2 * R) →
  (F = 24) →
  (4 * M + 3 * R + 5 * F = 271.2) :=
by
  intros M R F h1 h2 h3
  sorry

end total_cost_of_items_is_correct_l1863_186331


namespace mean_goals_l1863_186373

theorem mean_goals :
  let goals := 2 * 3 + 4 * 2 + 5 * 1 + 6 * 1
  let players := 3 + 2 + 1 + 1
  goals / players = 25 / 7 :=
by
  sorry

end mean_goals_l1863_186373


namespace Shane_current_age_44_l1863_186324

-- Declaring the known conditions and definitions
variable (Garret_present_age : ℕ) (Shane_past_age : ℕ) (Shane_present_age : ℕ)
variable (h1 : Garret_present_age = 12)
variable (h2 : Shane_past_age = 2 * Garret_present_age)
variable (h3 : Shane_present_age = Shane_past_age + 20)

theorem Shane_current_age_44 : Shane_present_age = 44 :=
by
  -- Proof to be filled here
  sorry

end Shane_current_age_44_l1863_186324


namespace sum_of_f_values_l1863_186376

noncomputable def is_odd_function (f : ℝ → ℝ) := ∀ x, f (-x) = -f x

theorem sum_of_f_values 
  (f : ℝ → ℝ)
  (hf_odd : is_odd_function f)
  (hf_periodic : ∀ x, f (2 - x) = f x)
  (hf_neg_one : f (-1) = 1) :
  f 1 + f 2 + f 3 + f 4 + (502 * (f 1 + f 2 + f 3 + f 4)) = -1 := 
sorry

end sum_of_f_values_l1863_186376


namespace quadratic_roots_condition_l1863_186394

theorem quadratic_roots_condition (k : ℝ) : 
  ((∃ x : ℝ, (k - 1) * x^2 + 4 * x + 1 = 0) ∧ ∃ x1 x2 : ℝ, x1 ≠ x2) ↔ (k < 5 ∧ k ≠ 1) :=
by {
  sorry  
}

end quadratic_roots_condition_l1863_186394


namespace campaign_meaning_l1863_186322

-- Define a function that gives the meaning of "campaign" as a noun
def meaning_of_campaign_noun : String :=
  "campaign, activity"

-- The theorem asserts that the meaning of "campaign" as a noun is "campaign, activity"
theorem campaign_meaning : meaning_of_campaign_noun = "campaign, activity" :=
by
  -- We add sorry here because we are not required to provide the proof
  sorry

end campaign_meaning_l1863_186322


namespace least_positive_divisible_by_five_primes_l1863_186359

-- Define the smallest 5 primes
def smallest_five_primes : List ℕ := [2, 3, 5, 7, 11]

-- Define the least positive whole number divisible by these primes
def least_positive_divisible_by_primes (primes : List ℕ) : ℕ :=
  primes.foldl (· * ·) 1

-- State the theorem
theorem least_positive_divisible_by_five_primes :
  least_positive_divisible_by_primes smallest_five_primes = 2310 :=
by
  sorry

end least_positive_divisible_by_five_primes_l1863_186359


namespace log_eq_15_given_log_base3_x_eq_5_l1863_186348

variable (x : ℝ)
variable (log_base3_x : ℝ)
variable (h : log_base3_x = 5)

theorem log_eq_15_given_log_base3_x_eq_5 (h : log_base3_x = 5) : log_base3_x * 3 = 15 :=
by
  sorry

end log_eq_15_given_log_base3_x_eq_5_l1863_186348


namespace probability_of_positive_l1863_186393

-- Definitions based on the conditions
def balls : List ℚ := [-2, 0, 1/4, 3]
def total_balls : ℕ := 4
def positive_filter (x : ℚ) : Bool := x > 0
def positive_balls : List ℚ := balls.filter positive_filter
def positive_count : ℕ := positive_balls.length
def probability : ℚ := positive_count / total_balls

-- Statement to prove
theorem probability_of_positive : probability = 1 / 2 := by
  sorry

end probability_of_positive_l1863_186393


namespace no_valid_height_configuration_l1863_186387

-- Define the heights and properties
variables {a : Fin 7 → ℝ}
variables {p : ℝ}

-- Define the condition as a theorem
theorem no_valid_height_configuration (h : ∀ n : Fin 7, p * a n + (1 - p) * a (n + 2) % 7 > 
                                         p * a (n + 3) % 7 + (1 - p) * a (n + 1) % 7) :
  ¬ (∃ (a : Fin 7 → ℝ), 
    (∀ n : Fin 7, p * a n + (1 - p) * a (n + 2) % 7 > 
                  p * a (n + 3) % 7 + (1 - p) * a (n + 1) % 7) ∧
    true) :=
sorry

end no_valid_height_configuration_l1863_186387


namespace fiona_pairs_l1863_186358

theorem fiona_pairs : Nat.choose 12 2 = 66 := by
  sorry

end fiona_pairs_l1863_186358


namespace value_of_x_l1863_186320

theorem value_of_x (x : ℝ) (h : 2 ≤ |x - 3| ∧ |x - 3| ≤ 6) : x ∈ Set.Icc (-3 : ℝ) 1 ∪ Set.Icc 5 9 :=
by
  sorry

end value_of_x_l1863_186320


namespace max_value_of_expression_l1863_186301

theorem max_value_of_expression (x y z : ℝ) (h : x^2 + y^2 = z^2) :
  ∃ t, t = (3 * Real.sqrt 2) / 2 ∧ ∀ u, u = (x + 2 * y) / z → u ≤ t := by
  sorry

end max_value_of_expression_l1863_186301


namespace percentage_increase_first_year_l1863_186370

theorem percentage_increase_first_year (P : ℝ) (x : ℝ) :
  (1 + x / 100) * 0.7 = 1.0499999999999998 → x = 50 := 
by
  sorry

end percentage_increase_first_year_l1863_186370


namespace smallest_n_exists_unique_k_l1863_186377

/- The smallest positive integer n for which there exists
   a unique integer k such that 9/16 < n / (n + k) < 7/12 is n = 1. -/

theorem smallest_n_exists_unique_k :
  ∃! (n : ℕ), n > 0 ∧ (∃! (k : ℤ), (9 : ℚ)/16 < (n : ℤ)/(n + k) ∧ (n : ℤ)/(n + k) < (7 : ℚ)/12) :=
sorry

end smallest_n_exists_unique_k_l1863_186377


namespace rectangle_invalid_perimeter_l1863_186372

-- Define conditions
def positive_integer (n : ℕ) : Prop := n > 0

-- Define the rectangle with given area
def area_24 (length width : ℕ) : Prop := length * width = 24

-- Define the function to calculate perimeter for given length and width
def perimeter (length width : ℕ) : ℕ := 2 * (length + width)

-- The theorem to prove
theorem rectangle_invalid_perimeter (length width : ℕ) (h₁ : positive_integer length) (h₂ : positive_integer width) (h₃ : area_24 length width) : 
  (perimeter length width) ≠ 36 :=
sorry

end rectangle_invalid_perimeter_l1863_186372


namespace find_principal_amount_l1863_186329

-- Define the parameters
def R : ℝ := 11.67
def T : ℝ := 5
def A : ℝ := 950

-- State the theorem
theorem find_principal_amount : ∃ P : ℝ, A = P * (1 + (R/100) * T) :=
by { 
  use 600, 
  -- Skip the proof 
  sorry 
}

end find_principal_amount_l1863_186329


namespace tom_drives_distance_before_karen_wins_l1863_186352

def karen_late_minutes := 4
def karen_speed_mph := 60
def tom_speed_mph := 45

theorem tom_drives_distance_before_karen_wins : 
  ∃ d : ℝ, d = 21 := by
  sorry

end tom_drives_distance_before_karen_wins_l1863_186352


namespace max_f_l1863_186392

noncomputable def f (x : ℝ) : ℝ :=
  1 / (|x + 3| + |x + 1| + |x - 2| + |x - 5|)

theorem max_f : ∃ x : ℝ, f x = 1 / 11 :=
by
  sorry

end max_f_l1863_186392


namespace new_average_after_doubling_l1863_186300

theorem new_average_after_doubling (n : ℕ) (avg : ℝ) (h_n : n = 12) (h_avg : avg = 50) :
  2 * avg = 100 :=
by
  sorry

end new_average_after_doubling_l1863_186300


namespace oil_output_per_capita_l1863_186385

theorem oil_output_per_capita 
  (total_oil_output_russia : ℝ := 13737.1 * 100 / 9)
  (population_russia : ℝ := 147)
  (population_non_west : ℝ := 6.9)
  (oil_output_non_west : ℝ := 1480.689)
  : 
  (55.084 : ℝ) = 55.084 ∧ 
    (214.59 : ℝ) = (1480.689 / 6.9) ∧ 
    (1038.33 : ℝ) = (total_oil_output_russia / population_russia) :=
by
  sorry

end oil_output_per_capita_l1863_186385


namespace total_lives_correct_l1863_186313

-- Define the initial number of friends
def initial_friends : ℕ := 16

-- Define the number of lives each player has
def lives_per_player : ℕ := 10

-- Define the number of additional players that joined
def additional_players : ℕ := 4

-- Define the initial total number of lives
def initial_lives : ℕ := initial_friends * lives_per_player

-- Define the additional lives from the new players
def additional_lives : ℕ := additional_players * lives_per_player

-- Define the final total number of lives
def total_lives : ℕ := initial_lives + additional_lives

-- The proof goal
theorem total_lives_correct : total_lives = 200 :=
by
  -- This is where the proof would be written, but it is omitted.
  sorry

end total_lives_correct_l1863_186313


namespace max_area_curves_intersection_l1863_186306

open Real

def C₁ (x : ℝ) : ℝ := x^3 - x
def C₂ (x a : ℝ) : ℝ := (x - a)^3 - (x - a)

theorem max_area_curves_intersection (a : ℝ) (h : ∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ C₁ x₁ = C₂ x₁ a ∧ C₁ x₂ = C₂ x₂ a) :
  ∃ A_max : ℝ, A_max = 3 / 4 :=
by
  -- TODO: Provide the proof here
  sorry

end max_area_curves_intersection_l1863_186306


namespace distinct_digits_sum_base7_l1863_186369

theorem distinct_digits_sum_base7
    (A B C : ℕ)
    (h_distinct : A ≠ B ∧ B ≠ C ∧ C ≠ A)
    (h_nonzero : A ≠ 0 ∧ B ≠ 0 ∧ C ≠ 0)
    (h_base7 : A < 7 ∧ B < 7 ∧ C < 7)
    (h_sum_eq : ((7^2 * A + 7 * B + C) + (7^2 * B + 7 * C + A) + (7^2 * C + 7 * A + B)) = (7^3 * A + 7^2 * A + 7 * A)) :
    B + C = 6 :=
by {
    sorry
}

end distinct_digits_sum_base7_l1863_186369


namespace hannah_bought_two_sets_of_measuring_spoons_l1863_186310

-- Definitions of conditions
def number_of_cookies_sold : ℕ := 40
def price_per_cookie : ℝ := 0.8
def number_of_cupcakes_sold : ℕ := 30
def price_per_cupcake : ℝ := 2.0
def cost_per_measuring_spoon_set : ℝ := 6.5
def remaining_money : ℝ := 79

-- Definition of total money made from selling cookies and cupcakes
def total_money_made : ℝ := (number_of_cookies_sold * price_per_cookie) + (number_of_cupcakes_sold * price_per_cupcake)

-- Definition of money spent on measuring spoons
def money_spent_on_measuring_spoons : ℝ := total_money_made - remaining_money

-- Theorem statement
theorem hannah_bought_two_sets_of_measuring_spoons :
  (money_spent_on_measuring_spoons / cost_per_measuring_spoon_set) = 2 := by
  sorry

end hannah_bought_two_sets_of_measuring_spoons_l1863_186310


namespace complex_addition_l1863_186319

def c : ℂ := 3 - 2 * Complex.I
def d : ℂ := 1 + 3 * Complex.I

theorem complex_addition : 3 * c + 4 * d = 13 + 6 * Complex.I := by
  -- proof goes here
  sorry

end complex_addition_l1863_186319


namespace inequality_proof_l1863_186328

variables (x y : ℝ) (n : ℕ)

theorem inequality_proof (h1 : x > 0) (h2 : y > 0) (h3 : x + y = 1) (h4 : n ≥ 2) :
  (x^n / (x + y^3) + y^n / (x^3 + y)) ≥ (2^(4-n) / 5) := by
  sorry

end inequality_proof_l1863_186328


namespace uncle_bradley_bills_l1863_186363

theorem uncle_bradley_bills :
  ∃ (fifty_bills hundred_bills : ℕ),
    (fifty_bills = 300 / 50) ∧ (hundred_bills = 700 / 100) ∧ (300 + 700 = 1000) ∧ (50 * fifty_bills + 100 * hundred_bills = 1000) ∧ (fifty_bills + hundred_bills = 13) :=
by
  sorry

end uncle_bradley_bills_l1863_186363


namespace sum_of_consecutive_powers_divisible_l1863_186390

theorem sum_of_consecutive_powers_divisible (a : ℕ) (n : ℕ) (h : 0 ≤ n) : 
  a^n + a^(n + 1) ∣ a * (a + 1) :=
sorry

end sum_of_consecutive_powers_divisible_l1863_186390


namespace triangle_third_side_l1863_186388

theorem triangle_third_side {x : ℕ} (h1 : 3 < x) (h2 : x < 7) (h3 : x % 2 = 1) : x = 5 := by
  sorry

end triangle_third_side_l1863_186388


namespace expression_evaluation_l1863_186312

theorem expression_evaluation :
  (5^1001 + 6^1002)^2 - (5^1001 - 6^1002)^2 = 24 * 30^1001 := by
  sorry

end expression_evaluation_l1863_186312


namespace intervals_increasing_max_min_value_range_of_m_l1863_186364

noncomputable def a (x : ℝ) : ℝ × ℝ := (Real.sin x, -Real.cos x)
noncomputable def b (x : ℝ) : ℝ × ℝ := (Real.sqrt 3 * Real.cos x, Real.cos x)
noncomputable def f (x : ℝ) : ℝ := (a x).1 * (b x).1 + (a x).2 * (b x).2 - 1/2

theorem intervals_increasing : ∀ (x : ℝ), ∃ k : ℤ, -π/6 + k * π ≤ x ∧ x ≤ π/3 + k * π := sorry

theorem max_min_value (x : ℝ) (hx : π/4 ≤ x ∧ x ≤ π/2) :
  (f (π/3) = 0) ∧ (f (π/2) = -1/2) :=
  sorry

theorem range_of_m (x : ℝ) (hx : π/4 ≤ x ∧ x ≤ π/2) :
  ∀ m : ℝ, (∀ y : ℝ, (π/4 ≤ y ∧ y ≤ π/2) → |f y - m| < 1) ↔ (-1 < m ∧ m < 1/2) :=
  sorry

end intervals_increasing_max_min_value_range_of_m_l1863_186364


namespace jackie_first_tree_height_l1863_186379

theorem jackie_first_tree_height
  (h : ℝ)
  (avg_height : (h + 2 * (h / 2) + (h + 200)) / 4 = 800) :
  h = 1000 :=
by
  sorry

end jackie_first_tree_height_l1863_186379


namespace parallel_lines_sufficient_but_not_necessary_l1863_186368

theorem parallel_lines_sufficient_but_not_necessary (a : ℝ) :
  (a = 1 ↔ ((ax + y - 1 = 0) ∧ (x + ay + 1 = 0) → False)) := 
sorry

end parallel_lines_sufficient_but_not_necessary_l1863_186368


namespace find_smallest_d_l1863_186389

-- Given conditions: The known digits sum to 26
def sum_known_digits : ℕ := 5 + 2 + 4 + 7 + 8 

-- Define the smallest digit d such that 52,d47,8 is divisible by 9
def smallest_d (d : ℕ) (sum_digits_with_d : ℕ) : Prop :=
  sum_digits_with_d = sum_known_digits + d ∧ (sum_digits_with_d % 9 = 0)

theorem find_smallest_d : ∃ d : ℕ, smallest_d d 27 :=
sorry

end find_smallest_d_l1863_186389


namespace blocks_combination_count_l1863_186356

-- Definition statements reflecting all conditions in the problem
def select_4_blocks_combinations : ℕ :=
  let choose (n k : ℕ) := Nat.choose n k
  let factorial (n : ℕ) := Nat.factorial n
  choose 6 4 * choose 6 4 * factorial 4

-- Theorem stating the result we want to prove
theorem blocks_combination_count : select_4_blocks_combinations = 5400 :=
by
  -- We will provide the proof steps here
  sorry

end blocks_combination_count_l1863_186356


namespace roots_negative_reciprocal_l1863_186398

theorem roots_negative_reciprocal (a b c : ℝ) (α β : ℝ) (h_eq : a * α ^ 2 + b * α + c = 0)
  (h_roots : α * β = -1) : c = -a :=
sorry

end roots_negative_reciprocal_l1863_186398


namespace calculate_order_cost_l1863_186338

-- Defining the variables and given conditions
variables (C E S D W : ℝ)

-- Given conditions as assumptions
axiom h1 : (2 / 5) * C = E * S
axiom h2 : (1 / 4) * (3 / 5) * C = D * W

-- Theorem statement for the amount paid for the orders
theorem calculate_order_cost (C E S D W : ℝ) (h1 : (2 / 5) * C = E * S) (h2 : (1 / 4) * (3 / 5) * C = D * W) : 
  (9 / 20) * C = C - ((2 / 5) * C + (3 / 20) * C) :=
sorry

end calculate_order_cost_l1863_186338


namespace ball_bounces_height_l1863_186362

theorem ball_bounces_height : ∃ k : ℕ, ∀ n ≥ k, 800 * (2 / 3: ℝ) ^ n < 10 :=
by
  sorry

end ball_bounces_height_l1863_186362


namespace correct_comprehensive_survey_l1863_186344

-- Definitions for the types of surveys.
inductive Survey
| A : Survey
| B : Survey
| C : Survey
| D : Survey

-- Function that identifies the survey suitable for a comprehensive survey.
def is_comprehensive_survey (s : Survey) : Prop :=
  match s with
  | Survey.A => False            -- A is for sampling, not comprehensive
  | Survey.B => False            -- B is for sampling, not comprehensive
  | Survey.C => False            -- C is for sampling, not comprehensive
  | Survey.D => True             -- D is suitable for comprehensive survey

-- The theorem to prove that D is the correct answer.
theorem correct_comprehensive_survey : is_comprehensive_survey Survey.D = True := by
  sorry

end correct_comprehensive_survey_l1863_186344


namespace min_expression_value_l1863_186342

open Real

-- Define the conditions given in the problem: x, y, z are positive reals and their product is 32
variables {x y z : ℝ} (h₁ : 0 < x) (h₂ : 0 < y) (h₃ : 0 < z) (h₄ : x * y * z = 32)

-- Define the expression that we want to find the minimum for: x^2 + 4xy + 4y^2 + 2z^2
def expression (x y z : ℝ) : ℝ := x^2 + 4 * x * y + 4 * y^2 + 2 * z^2

-- State the theorem: proving that the minimum value of the expression given the conditions is 96
theorem min_expression_value : ∃ (x y z : ℝ), 0 < x ∧ 0 < y ∧ 0 < z ∧ x * y * z = 32 ∧ expression x y z = 96 :=
sorry

end min_expression_value_l1863_186342


namespace opposite_of_negative_2023_l1863_186330

theorem opposite_of_negative_2023 : (- -2023) = 2023 := 
by {
sorry
}

end opposite_of_negative_2023_l1863_186330


namespace total_marks_secured_l1863_186341

-- Define the conditions
def correct_points_per_question := 4
def wrong_points_per_question := 1
def total_questions := 60
def correct_questions := 40

-- Calculate the remaining incorrect questions
def wrong_questions := total_questions - correct_questions

-- Calculate total marks secured by the student
def total_marks := (correct_questions * correct_points_per_question) - (wrong_questions * wrong_points_per_question)

-- The statement to be proven
theorem total_marks_secured : total_marks = 140 := by
  -- This will be proven in Lean's proof assistant
  sorry

end total_marks_secured_l1863_186341


namespace proof_problem_l1863_186327

variable {ι : Type} [LinearOrderedField ι]

-- Let A be a family of sets indexed by natural numbers
variables {A : ℕ → Set ι}

-- Hypotheses
def condition1 (A : ℕ → Set ι) : Prop :=
  (⋃ i, A i) = Set.univ

def condition2 (A : ℕ → Set ι) (a : ι) : Prop :=
  ∀ i b c, b > c → b - c ≥ a ^ i → b ∈ A i → c ∈ A i

theorem proof_problem (A : ℕ → Set ι) (a : ι) :
  condition1 A → condition2 A a → 0 < a → a < 2 :=
sorry

end proof_problem_l1863_186327


namespace quadratic_root_in_interval_l1863_186317

variable (a b c : ℝ)

theorem quadratic_root_in_interval 
  (h : 2 * a + 3 * b + 6 * c = 0) :
  ∃ x : ℝ, 0 < x ∧ x < 1 ∧ a * x^2 + b * x + c = 0 :=
by
  sorry

end quadratic_root_in_interval_l1863_186317


namespace certain_number_is_36_75_l1863_186349

theorem certain_number_is_36_75 (A B C X : ℝ) (h_ratio_A : A = 5 * (C / 8)) (h_ratio_B : B = 6 * (C / 8)) (h_C : C = 42) (h_relation : A + C = B + X) :
  X = 36.75 :=
by
  sorry

end certain_number_is_36_75_l1863_186349


namespace maximum_correct_answers_l1863_186367

theorem maximum_correct_answers (a b c : ℕ) (h1 : a + b + c = 60)
  (h2 : 5 * a - 2 * c = 150) : a ≤ 38 :=
by
  sorry

end maximum_correct_answers_l1863_186367


namespace inequality_f_l1863_186332

-- Definitions of the given conditions
def f (x : ℝ) (a b : ℝ) : ℝ := x^2 + a*x + b

-- Theorem statement
theorem inequality_f (a b : ℝ) : 
  abs (f 1 a b) + 2 * abs (f 2 a b) + abs (f 3 a b) ≥ 2 :=
by sorry

end inequality_f_l1863_186332


namespace determine_GH_l1863_186316

-- Define a structure for a Tetrahedron with edge lengths as given conditions
structure Tetrahedron :=
  (EF FG EH FH EG GH : ℕ)

-- Instantiate the Tetrahedron with the given edge lengths
def tetrahedron_EFGH := Tetrahedron.mk 42 14 37 19 28 14

-- State the theorem
theorem determine_GH (t : Tetrahedron) (hEF : t.EF = 42) :
  t.GH = 14 :=
sorry

end determine_GH_l1863_186316


namespace inequality_proof_l1863_186371

theorem inequality_proof (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  a / (b * c) + b / (a * c) + c / (a * b) ≥ 2 / a + 2 / b - 2 / c := 
  sorry

end inequality_proof_l1863_186371


namespace determine_a_l1863_186321

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a * x ^ 2 + 2 * x

theorem determine_a (a : ℝ) : (∀ x : ℝ, f a (-x) = -f a x) → a = 0 :=
by
  intros h
  sorry

end determine_a_l1863_186321


namespace units_digit_of_pow_sum_is_correct_l1863_186308

theorem units_digit_of_pow_sum_is_correct : 
  (24^3 + 42^3) % 10 = 2 := by
  -- Start the proof block
  sorry

end units_digit_of_pow_sum_is_correct_l1863_186308


namespace parallel_lines_from_perpendicularity_l1863_186391

variables (a b : Type) (α β : Type)

-- Define the necessary conditions
def is_line (l : Type) : Prop := sorry
def is_plane (p : Type) : Prop := sorry
def perpendicular (l : Type) (p : Type) : Prop := sorry
def parallel (l1 l2 : Type) : Prop := sorry

axiom line_a : is_line a
axiom line_b : is_line b
axiom plane_alpha : is_plane α
axiom plane_beta : is_plane β
axiom a_perp_alpha : perpendicular a α
axiom b_perp_alpha : perpendicular b α

-- State the theorem
theorem parallel_lines_from_perpendicularity : parallel a b :=
  sorry

end parallel_lines_from_perpendicularity_l1863_186391


namespace solve_inequality_solution_set_l1863_186302

def solution_set (x : ℝ) : Prop := -x^2 + 5 * x > 6

theorem solve_inequality_solution_set :
  { x : ℝ | solution_set x } = { x : ℝ | 2 < x ∧ x < 3 } :=
sorry

end solve_inequality_solution_set_l1863_186302


namespace cookie_sheet_perimeter_l1863_186303

theorem cookie_sheet_perimeter :
  let width_in_inches := 15.2
  let length_in_inches := 3.7
  let conversion_factor := 2.54
  let width_in_cm := width_in_inches * conversion_factor
  let length_in_cm := length_in_inches * conversion_factor
  2 * (width_in_cm + length_in_cm) = 96.012 :=
by
  sorry

end cookie_sheet_perimeter_l1863_186303


namespace at_least_one_negative_l1863_186353

-- Defining the circle partition and the properties given in the problem.
def circle_partition (a : Fin 7 → ℤ) : Prop :=
  ∃ (l1 l2 l3 : Finset (Fin 7)),
    l1.card = 4 ∧ l2.card = 4 ∧ l3.card = 4 ∧
    (∀ i ∈ l1, ∀ j ∉ l1, a i + a j = 0) ∧
    (∀ i ∈ l2, ∀ j ∉ l2, a i + a j = 0) ∧
    (∀ i ∈ l3, ∀ j ∉ l3, a i + a j = 0) ∧
    ∃ i, a i = 0

-- The main theorem to prove.
theorem at_least_one_negative : 
  ∀ (a : Fin 7 → ℤ), 
  circle_partition a → 
  ∃ i, a i < 0 :=
by
  sorry

end at_least_one_negative_l1863_186353


namespace courtyard_length_is_60_l1863_186396

noncomputable def stone_length : ℝ := 2.5
noncomputable def stone_breadth : ℝ := 2.0
noncomputable def num_stones : ℕ := 198
noncomputable def courtyard_breadth : ℝ := 16.5

theorem courtyard_length_is_60 :
  ∃ (courtyard_length : ℝ), courtyard_length = 60 ∧
  num_stones * (stone_length * stone_breadth) = courtyard_length * courtyard_breadth :=
sorry

end courtyard_length_is_60_l1863_186396


namespace unique_solution_triplet_l1863_186395

theorem unique_solution_triplet :
  ∃! (x y z : ℕ), 0 < x ∧ 0 < y ∧ 0 < z ∧ (x^y + y^x = z^y ∧ x^y + 2012 = y^(z+1)) ∧ (x = 6 ∧ y = 2 ∧ z = 10) := 
by {
  sorry
}

end unique_solution_triplet_l1863_186395


namespace total_fruit_salads_correct_l1863_186335

-- Definitions for the conditions
def alayas_fruit_salads : ℕ := 200
def angels_fruit_salads : ℕ := 2 * alayas_fruit_salads
def total_fruit_salads : ℕ := alayas_fruit_salads + angels_fruit_salads

-- Theorem statement
theorem total_fruit_salads_correct : total_fruit_salads = 600 := by
  -- Proof goes here, but is not required for this task
  sorry

end total_fruit_salads_correct_l1863_186335


namespace find_third_number_l1863_186375

theorem find_third_number 
  (h1 : (14 + 32 + x) / 3 = (21 + 47 + 22) / 3 + 3) : x = 53 := by
  sorry

end find_third_number_l1863_186375


namespace f_sin_periodic_f_monotonically_increasing_f_minus_2_not_even_f_symmetric_about_point_l1863_186381

noncomputable def f (x : ℝ) : ℝ := (4 * Real.exp x) / (Real.exp x + 1)

theorem f_sin_periodic : ∀ x, f (Real.sin (x + 2 * Real.pi)) = f (Real.sin x) := sorry

theorem f_monotonically_increasing : ∀ x y, x < y → f x < f y := sorry

theorem f_minus_2_not_even : ¬(∀ x, f x - 2 = f (-x) - 2) := sorry

theorem f_symmetric_about_point : ∀ x, f x + f (-x) = 4 := sorry

end f_sin_periodic_f_monotonically_increasing_f_minus_2_not_even_f_symmetric_about_point_l1863_186381


namespace sufficient_but_not_necessary_condition_for_prop_l1863_186386

theorem sufficient_but_not_necessary_condition_for_prop :
  (∀ x : ℝ, 1 ≤ x ∧ x ≤ 2 → x^2 - a ≤ 0) → a ≥ 5 :=
sorry

end sufficient_but_not_necessary_condition_for_prop_l1863_186386


namespace fraction_difference_l1863_186378

def A : ℕ := 3 + 6 + 9
def B : ℕ := 2 + 5 + 8

theorem fraction_difference : (A / B) - (B / A) = 11 / 30 := by
  sorry

end fraction_difference_l1863_186378

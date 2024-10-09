import Mathlib

namespace tourist_tax_l1432_143266

theorem tourist_tax (total_value : ℝ) (non_taxable_amount : ℝ) (tax_rate : ℝ) 
  (h1 : total_value = 1720) (h2 : non_taxable_amount = 600) (h3 : tax_rate = 0.08) : 
  ((total_value - non_taxable_amount) * tax_rate = 89.60) :=
by 
  sorry

end tourist_tax_l1432_143266


namespace figure_area_l1432_143223

-- Given conditions
def right_angles (α β γ δ: ℕ): Prop :=
  α = 90 ∧ β = 90 ∧ γ = 90 ∧ δ = 90

def segment_lengths (a b c d e f g: ℕ): Prop :=
  a = 15 ∧ b = 8 ∧ c = 7 ∧ d = 3 ∧ e = 4 ∧ f = 2 ∧ g = 5

-- Define the problem
theorem figure_area :
  ∀ (α β γ δ a b c d e f g: ℕ),
    right_angles α β γ δ →
    segment_lengths a b c d e f g →
    a * b - (g * 1 + (d * f)) = 109 :=
by
  sorry

end figure_area_l1432_143223


namespace bananas_to_oranges_l1432_143264

theorem bananas_to_oranges :
  (3 / 4) * 16 * (1 / 1 : ℝ) = 10 * (1 / 1 : ℝ) → 
  (3 / 5) * 15 * (1 / 1 : ℝ) = 7.5 * (1 / 1 : ℝ) := 
by
  intros h
  sorry

end bananas_to_oranges_l1432_143264


namespace retirement_hire_year_l1432_143239

theorem retirement_hire_year (A : ℕ) (R : ℕ) (Y : ℕ) (W : ℕ) 
  (h1 : A + W = 70) 
  (h2 : A = 32) 
  (h3 : R = 2008) 
  (h4 : W = R - Y) : Y = 1970 :=
by
  sorry

end retirement_hire_year_l1432_143239


namespace sum_of_digits_square_1111111_l1432_143227

noncomputable def sum_of_digits (n : ℕ) : ℕ := sorry

theorem sum_of_digits_square_1111111 :
  sum_of_digits (1111111 * 1111111) = 49 :=
sorry

end sum_of_digits_square_1111111_l1432_143227


namespace arithmetic_geometric_l1432_143253

theorem arithmetic_geometric (a : ℕ → ℤ) (d : ℤ) (h1 : d = 2)
  (h2 : ∀ n, a (n + 1) - a n = d)
  (h3 : ∃ r, a 1 * r = a 3 ∧ a 3 * r = a 4) :
  a 2 = -6 :=
by sorry

end arithmetic_geometric_l1432_143253


namespace tens_digit_of_seven_times_cubed_is_one_l1432_143272

-- Variables and definitions
variables (p : ℕ) (h1 : p < 10)

-- Main theorem statement
theorem tens_digit_of_seven_times_cubed_is_one (hp : p < 10) :
  let N := 11 * p
  let m := 7
  let result := m * N^3
  (result / 10) % 10 = 1 := 
sorry

end tens_digit_of_seven_times_cubed_is_one_l1432_143272


namespace min_workers_to_profit_l1432_143255

/-- Definitions of constants used in the problem. --/
def daily_maintenance_cost : ℕ := 500
def wage_per_hour : ℕ := 20
def widgets_per_hour_per_worker : ℕ := 5
def sell_price_per_widget : ℕ := 350 / 100 -- since the input is 3.50
def workday_hours : ℕ := 8

/-- Profit condition: the revenue should be greater than the cost. 
    The problem specifies that the number of workers must be at least 26 to make a profit. --/

theorem min_workers_to_profit (n : ℕ) :
  (widgets_per_hour_per_worker * workday_hours * sell_price_per_widget * n > daily_maintenance_cost + (workday_hours * wage_per_hour * n)) → n ≥ 26 :=
sorry


end min_workers_to_profit_l1432_143255


namespace triangular_weight_is_60_l1432_143216

/-- Suppose there are weights: 5 identical round, 2 identical triangular, and 1 rectangular weight of 90 grams.
    The conditions are: 
    1. One round weight and one triangular weight balance three round weights.
    2. Four round weights and one triangular weight balance one triangular weight, one round weight, and one rectangular weight.
    Prove that the weight of the triangular weight is 60 grams. -/
theorem triangular_weight_is_60 
  (R T : ℕ)  -- We declare weights of round and triangular weights as natural numbers
  (h1 : R + T = 3 * R)  -- The first balance condition
  (h2 : 4 * R + T = T + R + 90)  -- The second balance condition
  : T = 60 := 
by
  sorry  -- Proof omitted

end triangular_weight_is_60_l1432_143216


namespace geometric_series_sum_l1432_143293

theorem geometric_series_sum :
  let a := 2
  let r := 3
  let n := 6
  S = a * (r ^ n - 1) / (r - 1) → S = 728 :=
by
  intros a r n h
  sorry

end geometric_series_sum_l1432_143293


namespace andrew_vacation_days_l1432_143230

-- Andrew's working days and vacation accrual rate
def days_worked : ℕ := 300
def vacation_rate : Nat := 10
def vacation_days_earned : ℕ := days_worked / vacation_rate

-- Days off in March and September
def days_off_march : ℕ := 5
def days_off_september : ℕ := 2 * days_off_march
def total_days_off : ℕ := days_off_march + days_off_september

-- Remaining vacation days calculation
def remaining_vacation_days : ℕ := vacation_days_earned - total_days_off

-- Problem statement to prove
theorem andrew_vacation_days : remaining_vacation_days = 15 :=
by
  -- Substitute the known values and perform the calculation
  unfold remaining_vacation_days vacation_days_earned total_days_off vacation_rate days_off_march days_off_september days_worked
  norm_num
  sorry

end andrew_vacation_days_l1432_143230


namespace smallest_integer_of_inequality_l1432_143252

theorem smallest_integer_of_inequality :
  ∃ x : ℤ, (8 - 7 * x ≥ 4 * x - 3) ∧ (∀ y : ℤ, (8 - 7 * y ≥ 4 * y - 3) → y ≥ x) ∧ x = 1 :=
sorry

end smallest_integer_of_inequality_l1432_143252


namespace cubic_inequality_solution_l1432_143263

theorem cubic_inequality_solution (x : ℝ) : x^3 - 12 * x^2 + 27 * x > 0 ↔ (0 < x ∧ x < 3) ∨ (9 < x) :=
by sorry

end cubic_inequality_solution_l1432_143263


namespace exists_same_color_rectangle_l1432_143281

open Finset

-- Define the grid size
def gridSize : ℕ := 12

-- Define the type of colors
inductive Color
| red
| white
| blue

-- Define a point in the grid
structure Point :=
(x : ℕ)
(y : ℕ)
(hx : x ≥ 1 ∧ x ≤ gridSize)
(hy : y ≥ 1 ∧ y ≤ gridSize)

-- Assume a coloring function
def color (p : Point) : Color := sorry

-- The theorem statement
theorem exists_same_color_rectangle :
  ∃ (p1 p2 p3 p4 : Point),
    p1.x = p2.x ∧ p3.x = p4.x ∧
    p1.y = p3.y ∧ p2.y = p4.y ∧
    color p1 = color p2 ∧
    color p1 = color p3 ∧
    color p1 = color p4 :=
sorry

end exists_same_color_rectangle_l1432_143281


namespace smallest_positive_integer_solution_l1432_143269

theorem smallest_positive_integer_solution (x : ℤ) 
  (hx : |5 * x - 8| = 47) : x = 11 :=
by
  sorry

end smallest_positive_integer_solution_l1432_143269


namespace white_balls_count_l1432_143271

theorem white_balls_count (a : ℕ) (h : 3 / (3 + a) = 3 / 7) : a = 4 :=
by sorry

end white_balls_count_l1432_143271


namespace maximize_profit_l1432_143224

noncomputable def annual_profit : ℝ → ℝ
| x => if x < 80 then - (1/3) * x^2 + 40 * x - 250 
       else 1200 - (x + 10000 / x)

theorem maximize_profit : ∃ x : ℝ, x = 100 ∧ annual_profit x = 1000 :=
by
  sorry

end maximize_profit_l1432_143224


namespace green_papayas_left_l1432_143232

/-- Define the initial number of green papayas on the tree -/
def initial_green_papayas : ℕ := 14

/-- Define the number of papayas that turned yellow on Friday -/
def friday_yellow_papayas : ℕ := 2

/-- Define the number of papayas that turned yellow on Sunday -/
def sunday_yellow_papayas : ℕ := 2 * friday_yellow_papayas

/-- The remaining number of green papayas after Friday and Sunday -/
def remaining_green_papayas : ℕ := initial_green_papayas - friday_yellow_papayas - sunday_yellow_papayas

theorem green_papayas_left : remaining_green_papayas = 8 := by
  sorry

end green_papayas_left_l1432_143232


namespace find_other_number_l1432_143203

theorem find_other_number (A B : ℕ) (h_lcm : Nat.lcm A B = 2310) (h_hcf : Nat.gcd A B = 61) (h_a : A = 210) : B = 671 :=
by
  sorry

end find_other_number_l1432_143203


namespace max_min_values_of_function_l1432_143290

theorem max_min_values_of_function :
  (∀ x, 0 ≤ 2 * Real.sin x + 2 ∧ 2 * Real.sin x + 2 ≤ 4) ↔ (∃ x, 2 * Real.sin x + 2 = 0) ∧ (∃ y, 2 * Real.sin y + 2 = 4) :=
by
  sorry

end max_min_values_of_function_l1432_143290


namespace derivative_of_f_l1432_143220

noncomputable def f (x : ℝ) : ℝ := 2 * x^2 - 2 * x + 1

theorem derivative_of_f :
  ∀ x : ℝ, deriv f x = 4 * x - 2 :=
by
  intro x
  -- proof skipped
  sorry

end derivative_of_f_l1432_143220


namespace imaginary_part_of_complex_l1432_143245

open Complex

theorem imaginary_part_of_complex (i : ℂ) (z : ℂ) (h1 : i^2 = -1) (h2 : z = (3 - 2 * i^3) / (1 + i)) : z.im = -1 / 2 :=
by {
  -- Proof would go here
  sorry
}

end imaginary_part_of_complex_l1432_143245


namespace vote_percentage_for_candidate_A_l1432_143236

noncomputable def percent_democrats : ℝ := 0.60
noncomputable def percent_republicans : ℝ := 0.40
noncomputable def percent_voting_a_democrats : ℝ := 0.70
noncomputable def percent_voting_a_republicans : ℝ := 0.20

theorem vote_percentage_for_candidate_A :
    (percent_democrats * percent_voting_a_democrats + percent_republicans * percent_voting_a_republicans) * 100 = 50 := by
  sorry

end vote_percentage_for_candidate_A_l1432_143236


namespace initial_men_garrison_l1432_143231

-- Conditions:
-- A garrison has provisions for 31 days.
-- At the end of 16 days, a reinforcement of 300 men arrives.
-- The provisions last only for 5 days more after the reinforcement arrives.

theorem initial_men_garrison (M : ℕ) (P : ℕ) (d1 d2 : ℕ) (r : ℕ) (remaining1 remaining2 : ℕ) :
  P = M * d1 →
  remaining1 = P - M * d2 →
  remaining2 = r * (d1 - d2) →
  remaining1 = remaining2 →
  r = M + 300 →
  d1 = 31 →
  d2 = 16 →
  M = 150 :=
by 
  sorry

end initial_men_garrison_l1432_143231


namespace area_ratio_of_triangles_l1432_143270

theorem area_ratio_of_triangles (AC AD : ℝ) (h : ℝ) (hAC : AC = 1) (hAD : AD = 4) :
  (AC * h / 2) / ((AD - AC) * h / 2) = 1 / 3 :=
by
  sorry

end area_ratio_of_triangles_l1432_143270


namespace pow_mul_eq_add_l1432_143274

theorem pow_mul_eq_add (a : ℝ) : a^3 * a^4 = a^7 :=
by
  -- This is where the proof would go.
  sorry

end pow_mul_eq_add_l1432_143274


namespace find_m_l1432_143228

theorem find_m (m : ℕ) : 5 ^ m = 5 * 25 ^ 2 * 125 ^ 3 ↔ m = 14 := by
  sorry

end find_m_l1432_143228


namespace weight_of_sugar_is_16_l1432_143215

def weight_of_sugar_bag (weight_of_sugar weight_of_salt remaining_weight weight_removed : ℕ) : Prop :=
  weight_of_sugar + weight_of_salt - weight_removed = remaining_weight

theorem weight_of_sugar_is_16 :
  ∃ (S : ℕ), weight_of_sugar_bag S 30 42 4 ∧ S = 16 :=
by
  sorry

end weight_of_sugar_is_16_l1432_143215


namespace find_a2016_l1432_143219

-- Define the sequence according to the conditions
def seq (a : ℕ → ℤ) : Prop :=
  a 1 = 1 ∧ a 2 = 5 ∧ ∀ n : ℕ, n ≥ 1 → a (n + 2) = a (n + 1) - a n

-- State the main theorem we want to prove
theorem find_a2016 :
  ∃ a : ℕ → ℤ, seq a ∧ a 2016 = -4 :=
by
  sorry

end find_a2016_l1432_143219


namespace find_a1_l1432_143211

variable {a_n : ℕ → ℤ}
variable (common_difference : ℤ) (a1 : ℤ)

-- Define that a_n is an arithmetic sequence with common difference of 2
def is_arithmetic_seq (a_n : ℕ → ℤ) (common_difference : ℤ) : Prop :=
  ∀ n, a_n (n + 1) - a_n n = common_difference

-- State the condition that a1, a2, a4 form a geometric sequence
def forms_geometric_seq (a_n : ℕ → ℤ) : Prop :=
  ∃ a1 a2 a4, a2 * a2 = a1 * a4 ∧ a_n 1 = a1 ∧ a_n 2 = a2 ∧ a_n 4 = a4

-- Define the problem statement
theorem find_a1 (h_arith : is_arithmetic_seq a_n 2) (h_geom : forms_geometric_seq a_n) :
  a_n 1 = 2 :=
by
  sorry

end find_a1_l1432_143211


namespace xyz_sum_eq_48_l1432_143250

theorem xyz_sum_eq_48 (x y z : ℕ) (pos_x : 0 < x) (pos_y : 0 < y) (pos_z : 0 < z)
  (h1 : x * y + z = 47) (h2 : y * z + x = 47) (h3 : x * z + y = 47) : 
  x + y + z = 48 := by
  sorry

end xyz_sum_eq_48_l1432_143250


namespace number_of_people_l1432_143256

open Nat

theorem number_of_people (n : ℕ) (h : n^2 = 100) : n = 10 := by
  sorry

end number_of_people_l1432_143256


namespace number_of_students_taking_math_l1432_143295

variable (totalPlayers physicsOnly physicsAndMath mathOnly : ℕ)
variable (h1 : totalPlayers = 15) (h2 : physicsOnly = 9) (h3 : physicsAndMath = 3)

theorem number_of_students_taking_math : mathOnly = 9 :=
by {
  sorry
}

end number_of_students_taking_math_l1432_143295


namespace amount_of_flour_per_large_tart_l1432_143237

-- Statement without proof
theorem amount_of_flour_per_large_tart 
  (num_small_tarts : ℕ) (flour_per_small_tart : ℚ) 
  (num_large_tarts : ℕ) (total_flour : ℚ) 
  (h1 : num_small_tarts = 50) 
  (h2 : flour_per_small_tart = 1/8) 
  (h3 : num_large_tarts = 25) 
  (h4 : total_flour = num_small_tarts * flour_per_small_tart) : 
  total_flour = num_large_tarts * (1/4) := 
sorry

end amount_of_flour_per_large_tart_l1432_143237


namespace exists_N_binary_representation_l1432_143286

theorem exists_N_binary_representation (n p : ℕ) (h_composite : ∃ d : ℕ, d > 1 ∧ d < n ∧ n % d = 0) (h_proper_divisor : p > 0 ∧ p < n ∧ n % p = 0) :
  ∃ N : ℕ, ((1 + 2^p + 2^(n-p)) * N) % 2^n = 1 % 2^n :=
by
  sorry

end exists_N_binary_representation_l1432_143286


namespace choose_with_at_least_one_girl_l1432_143206

theorem choose_with_at_least_one_girl :
  let boys := 4
  let girls := 2
  let total_students := boys + girls
  let ways_choose_4 := Nat.choose total_students 4
  let ways_all_boys := Nat.choose boys 4
  ways_choose_4 - ways_all_boys = 14 := by
  sorry

end choose_with_at_least_one_girl_l1432_143206


namespace roots_difference_squared_l1432_143204

-- Defining the solutions to the quadratic equation
def quadratic_equation_roots (a b : ℚ) : Prop :=
  (2 * a^2 - 7 * a + 6 = 0) ∧ (2 * b^2 - 7 * b + 6 = 0)

-- The main theorem we aim to prove
theorem roots_difference_squared (a b : ℚ) (h : quadratic_equation_roots a b) :
    (a - b)^2 = 1 / 4 := 
  sorry

end roots_difference_squared_l1432_143204


namespace average_salary_l1432_143276

theorem average_salary (R S T : ℝ) 
  (h1 : (R + S) / 2 = 4000) 
  (h2 : T = 7000) : 
  (R + S + T) / 3 = 5000 :=
by
  sorry

end average_salary_l1432_143276


namespace find_a_l1432_143202

def A : Set ℝ := {x | x^2 - 3 * x + 2 = 0}
def B (a : ℝ) : Set ℝ := {x | x^2 + 2 * (a + 1) * x + (a^2 - 5) = 0}

theorem find_a (a : ℝ) (h : {x | x^2 - 3 * x + 2 = 0} ∩ {x | x^2 + 2 * (a + 1) * x + (a^2 - 5) = 0} = {2}) :
  a = -3 ∨ a = -1 :=
by
  sorry

end find_a_l1432_143202


namespace group_B_population_calculation_l1432_143208

variable {total_population : ℕ}
variable {sample_size : ℕ}
variable {sample_A : ℕ}
variable {total_B : ℕ}

theorem group_B_population_calculation 
  (h_total : total_population = 200)
  (h_sample_size : sample_size = 40)
  (h_sample_A : sample_A = 16)
  (h_sample_B : sample_size - sample_A = 24) :
  total_B = 120 :=
sorry

end group_B_population_calculation_l1432_143208


namespace chef_bought_almonds_l1432_143277

theorem chef_bought_almonds (total_nuts pecans : ℝ)
  (h1 : total_nuts = 0.52) (h2 : pecans = 0.38) :
  total_nuts - pecans = 0.14 :=
by
  sorry

end chef_bought_almonds_l1432_143277


namespace emma_bank_account_balance_l1432_143292

theorem emma_bank_account_balance
  (initial_balance : ℕ)
  (daily_spend : ℕ)
  (days_in_week : ℕ)
  (unit_bill : ℕ) :
  initial_balance = 100 → daily_spend = 8 → days_in_week = 7 → unit_bill = 5 →
  (initial_balance - daily_spend * days_in_week) % unit_bill = 4 :=
by
  intros h1 h2 h3 h4
  sorry

end emma_bank_account_balance_l1432_143292


namespace combined_height_of_rockets_l1432_143273

noncomputable def height_of_rocket (a t : ℝ) : ℝ := (1/2) * a * t^2

theorem combined_height_of_rockets
  (h_A_ft : ℝ)
  (fuel_type_B_coeff : ℝ)
  (g : ℝ)
  (ft_to_m : ℝ)
  (h_combined : ℝ) :
  h_A_ft = 850 →
  fuel_type_B_coeff = 1.7 →
  g = 9.81 →
  ft_to_m = 0.3048 →
  h_combined = 348.96 :=
by sorry

end combined_height_of_rockets_l1432_143273


namespace buy_beams_l1432_143241

theorem buy_beams (C T x : ℕ) (hC : C = 6210) (hT : T = 3) (hx: x > 0):
  T * (x - 1) = C / x :=
by
  rw [hC, hT]
  sorry

end buy_beams_l1432_143241


namespace correlation_statements_l1432_143242

variables {x y : ℝ}
variables (r : ℝ) (h1 : r > 0) (h2 : r = 1) (h3 : r = -1)

theorem correlation_statements :
  (r > 0 → (∀ x y, x > 0 → y > 0)) ∧
  (r = 1 ∨ r = -1 → (∀ x y, ∃ m b : ℝ, y = m * x + b)) :=
sorry

end correlation_statements_l1432_143242


namespace problem_a_l1432_143278

def continuous (f : ℝ → ℝ) : Prop := sorry -- Assume this is properly defined somewhere in Mathlib
def monotonic (f : ℝ → ℝ) : Prop := sorry -- Assume this is properly defined somewhere in Mathlib

theorem problem_a :
  ¬ (∀ (f : ℝ → ℝ), continuous f ∧ (∀ y, ∃ x, f x = y) → monotonic f) := sorry

end problem_a_l1432_143278


namespace right_triangle_altitude_l1432_143209

theorem right_triangle_altitude {DE DF EF altitude : ℝ} (h_right_triangle : DE^2 = DF^2 + EF^2)
  (h_DE : DE = 15) (h_DF : DF = 9) (h_EF : EF = 12) (h_area : (DF * EF) / 2 = 54) :
  altitude = 7.2 := 
  sorry

end right_triangle_altitude_l1432_143209


namespace identical_digit_square_l1432_143212

theorem identical_digit_square {b x y : ℕ} (hb : b ≥ 2) (hx : x < b) (hy : y < b) (hx_pos : x ≠ 0) (hy_pos : y ≠ 0) :
  (x * b + x)^2 = y * b^3 + y * b^2 + y * b + y ↔ b = 7 :=
by
  sorry

end identical_digit_square_l1432_143212


namespace find_foreign_language_score_l1432_143289

variable (c m f : ℝ)

theorem find_foreign_language_score
  (h1 : (c + m + f) / 3 = 94)
  (h2 : (c + m) / 2 = 92) :
  f = 98 := by
  sorry

end find_foreign_language_score_l1432_143289


namespace lemon_pie_degrees_l1432_143258

noncomputable def num_students := 45
noncomputable def chocolate_pie_students := 15
noncomputable def apple_pie_students := 9
noncomputable def blueberry_pie_students := 9
noncomputable def other_pie_students := num_students - (chocolate_pie_students + apple_pie_students + blueberry_pie_students)
noncomputable def each_remaining_pie_students := other_pie_students / 3
noncomputable def fraction_lemon_pie := each_remaining_pie_students / num_students
noncomputable def degrees_lemon_pie := fraction_lemon_pie * 360

theorem lemon_pie_degrees : degrees_lemon_pie = 32 :=
sorry

end lemon_pie_degrees_l1432_143258


namespace find_a_range_of_a_l1432_143225

noncomputable def f (x a : ℝ) := x + a * Real.log x

-- Proof problem 1: Prove that a = 2 given f' (1) = 3 for f (x) = x + a log x
theorem find_a (a : ℝ) : 
  (1 + a = 3) → (a = 2) := sorry

-- Proof problem 2: Prove that the range of a such that f(x) ≥ a always holds is [-e^2, 0]
theorem range_of_a (a : ℝ) :
  (∀ x > 0, f x a ≥ a) → (-Real.exp 2 ≤ a ∧ a ≤ 0) := sorry

end find_a_range_of_a_l1432_143225


namespace ratio_josh_to_selena_l1432_143244

def total_distance : ℕ := 36
def selena_distance : ℕ := 24

def josh_distance (td sd : ℕ) : ℕ := td - sd

theorem ratio_josh_to_selena : (josh_distance total_distance selena_distance) / selena_distance = 1 / 2 :=
by
  sorry

end ratio_josh_to_selena_l1432_143244


namespace new_determinant_l1432_143229

-- Given the condition that the determinant of the original matrix is 12
def original_determinant (x y z w : ℝ) : Prop :=
  x * w - y * z = 12

-- Proof that the determinant of the new matrix equals the expected result
theorem new_determinant (x y z w : ℝ) (h : original_determinant x y z w) :
  (2 * x + z) * w - (2 * y - w) * z = 24 + z * w + w * z := by
  sorry

end new_determinant_l1432_143229


namespace average_abc_l1432_143291

theorem average_abc (A B C : ℚ) 
  (h1 : 2002 * C - 3003 * A = 6006) 
  (h2 : 2002 * B + 4004 * A = 8008) 
  (h3 : B - C = A + 1) :
  (A + B + C) / 3 = 7 / 3 := 
sorry

end average_abc_l1432_143291


namespace can_cut_rectangle_l1432_143262

def original_rectangle_width := 100
def original_rectangle_height := 70
def total_area := original_rectangle_width * original_rectangle_height

def area1 := 1000
def area2 := 2000
def area3 := 4000

theorem can_cut_rectangle : 
  (area1 + area2 + area3 = total_area) ∧ 
  (area1 * 2 = area2) ∧ 
  (area1 * 4 = area3) ∧ 
  (area1 > 0) ∧ (area2 > 0) ∧ (area3 > 0) ∧
  (∃ (w1 h1 w2 h2 w3 h3 : ℕ), 
    w1 * h1 = area1 ∧ w2 * h2 = area2 ∧ w3 * h3 = area3 ∧
    ((w1 + w2 ≤ original_rectangle_width ∧ max h1 h2 + h3 ≤ original_rectangle_height) ∨
     (h1 + h2 ≤ original_rectangle_height ∧ max w1 w2 + w3 ≤ original_rectangle_width)))
:=
  sorry

end can_cut_rectangle_l1432_143262


namespace third_consecutive_even_l1432_143247

theorem third_consecutive_even {a b c d : ℕ} (h1 : b = a + 2) (h2 : c = a + 4) (h3 : d = a + 6) (h_sum : a + b + c + d = 52) : c = 14 :=
by
  sorry

end third_consecutive_even_l1432_143247


namespace quadratic_inequality_solution_l1432_143201

theorem quadratic_inequality_solution (x m : ℝ) :
  (x^2 + (2*m + 1)*x + m^2 + m > 0) ↔ (x > -m ∨ x < -m - 1) :=
by
  sorry

end quadratic_inequality_solution_l1432_143201


namespace second_number_is_34_l1432_143234

theorem second_number_is_34 (x y z : ℝ) (h1 : x + y + z = 120) 
  (h2 : x / y = 3 / 4) (h3 : y / z = 4 / 7) : y = 34 :=
by 
  sorry

end second_number_is_34_l1432_143234


namespace solve_n_l1432_143282

open Nat

def condition (n : ℕ) : Prop := 2^(n + 1) * 2^3 = 2^10

theorem solve_n (n : ℕ) (hn_pos : 0 < n) (h_cond : condition n) : n = 6 :=
by
  sorry

end solve_n_l1432_143282


namespace pencil_cost_is_4_l1432_143214

variables (pencils pens : ℕ) (pen_cost total_cost : ℕ)

def total_pencils := 15 * 80
def total_pens := (2 * total_pencils) + 300
def total_pen_cost := total_pens * pen_cost
def total_pencil_cost := total_cost - total_pen_cost
def pencil_cost := total_pencil_cost / total_pencils

theorem pencil_cost_is_4
  (pen_cost_eq_5 : pen_cost = 5)
  (total_cost_eq_18300 : total_cost = 18300)
  : pencil_cost = 4 :=
by
  sorry

end pencil_cost_is_4_l1432_143214


namespace final_number_correct_l1432_143294

noncomputable def initial_number : ℝ := 1256
noncomputable def first_increase_rate : ℝ := 3.25
noncomputable def second_increase_rate : ℝ := 1.47

theorem final_number_correct :
  initial_number * first_increase_rate * second_increase_rate = 6000.54 := 
by
  sorry

end final_number_correct_l1432_143294


namespace initial_black_pieces_is_118_l1432_143279

open Nat

-- Define the initial conditions and variables
variables (b w n : ℕ)

-- Hypotheses based on the conditions
axiom h1 : b = 2 * w
axiom h2 : w - 2 * n = 1
axiom h3 : b - 3 * n = 31

-- Goal to prove the initial number of black pieces were 118
theorem initial_black_pieces_is_118 : b = 118 :=
by 
  -- We only state the theorem, proof will be added as sorry
  sorry

end initial_black_pieces_is_118_l1432_143279


namespace units_digit_34_pow_30_l1432_143299

theorem units_digit_34_pow_30 :
  (34 ^ 30) % 10 = 6 :=
by
  sorry

end units_digit_34_pow_30_l1432_143299


namespace certain_number_l1432_143205

theorem certain_number (n w : ℕ) (h1 : w = 132)
  (h2 : ∃ m1 m2 m3, 32 = 2^5 * 3^3 * 11^2 * m1 * m2 * m3)
  (h3 : n * w = 132 * 2^3 * 3^2 * 11)
  (h4 : m1 = 1) (h5 : m2 = 1) (h6 : m3 = 1): 
  n = 792 :=
by sorry

end certain_number_l1432_143205


namespace min_value_of_S_l1432_143218

variable (x : ℝ)
def S (x : ℝ) : ℝ := (x - 10)^2 + (x + 5)^2

theorem min_value_of_S : ∀ x : ℝ, S x ≥ 112.5 :=
by
  sorry

end min_value_of_S_l1432_143218


namespace harry_worked_34_hours_l1432_143267

noncomputable def Harry_hours_worked (x : ℝ) : ℝ := 34

theorem harry_worked_34_hours (x : ℝ)
  (H : ℝ) (James_hours : ℝ) (Harry_pay James_pay: ℝ) 
  (h1 : Harry_pay = 18 * x + 1.5 * x * (H - 18)) 
  (h2 : James_pay = 40 * x + 2 * x * (James_hours - 40)) 
  (h3 : James_hours = 41) 
  (h4 : Harry_pay = James_pay) : 
  H = Harry_hours_worked x :=
by
  sorry

end harry_worked_34_hours_l1432_143267


namespace smallest_x_abs_eq_9_l1432_143268

theorem smallest_x_abs_eq_9 : ∃ x : ℝ, |x - 4| = 9 ∧ ∀ y : ℝ, |y - 4| = 9 → x ≤ y :=
by
  -- Prove there exists an x such that |x - 4| = 9 and for all y satisfying |y - 4| = 9, x is the minimum.
  sorry

end smallest_x_abs_eq_9_l1432_143268


namespace simplify_and_evaluate_l1432_143200

theorem simplify_and_evaluate (x : ℝ) (h : x^2 + 4 * x - 4 = 0) :
  3 * (x - 2) ^ 2 - 6 * (x + 1) * (x - 1) = 6 :=
by
  sorry

end simplify_and_evaluate_l1432_143200


namespace expected_value_of_die_l1432_143249

noncomputable def expected_value : ℚ :=
  (1/14) * 1 + (1/14) * 2 + (1/14) * 3 + (1/14) * 4 + (1/14) * 5 + (1/14) * 6 + (1/14) * 7 + (3/8) * 8

theorem expected_value_of_die : expected_value = 5 :=
by
  sorry

end expected_value_of_die_l1432_143249


namespace find_number_of_dimes_l1432_143261

def total_value (pennies nickels dimes quarters half_dollars : Nat) : Nat :=
  pennies * 1 + nickels * 5 + dimes * 10 + quarters * 25 + half_dollars * 50

def number_of_coins (pennies nickels dimes quarters half_dollars : Nat) : Nat :=
  pennies + nickels + dimes + quarters + half_dollars

theorem find_number_of_dimes
  (pennies nickels dimes quarters half_dollars : Nat)
  (h_value : total_value pennies nickels dimes quarters half_dollars = 163)
  (h_coins : number_of_coins pennies nickels dimes quarters half_dollars = 13)
  (h_penny : 1 ≤ pennies)
  (h_nickel : 1 ≤ nickels)
  (h_dime : 1 ≤ dimes)
  (h_quarter : 1 ≤ quarters)
  (h_half_dollar : 1 ≤ half_dollars) :
  dimes = 3 :=
sorry

end find_number_of_dimes_l1432_143261


namespace problem1_l1432_143246

theorem problem1 (α : Real) (h : Real.tan (Real.pi / 4 + α) = 2) : 
  (Real.sin α + 3 * Real.cos α) / (Real.sin α - Real.cos α) = -5 := 
sorry

end problem1_l1432_143246


namespace library_book_configurations_l1432_143265

def number_of_valid_configurations (total_books : ℕ) (min_in_library : ℕ) (min_checked_out : ℕ) : ℕ :=
  (total_books - (min_in_library + min_checked_out + 1)) + 1

theorem library_book_configurations : number_of_valid_configurations 8 2 2 = 5 :=
by
  -- Here we would write the Lean proof, but since we are only interested in the statement:
  sorry

end library_book_configurations_l1432_143265


namespace problem_l1432_143207

theorem problem (x : ℝ) (h : 3 * x^2 - 2 * x - 3 = 0) : 
  (x - 1)^2 + x * (x + 2 / 3) = 3 :=
by
  sorry

end problem_l1432_143207


namespace tan_periodic_n_solution_l1432_143280

open Real

theorem tan_periodic_n_solution :
  ∃ n : ℤ, -180 < n ∧ n < 180 ∧ tan (n * (π / 180)) = tan (1540 * (π / 180)) ∧ n = 40 :=
by
  sorry

end tan_periodic_n_solution_l1432_143280


namespace pow_five_2010_mod_seven_l1432_143213

theorem pow_five_2010_mod_seven :
  (5 ^ 2010) % 7 = 1 :=
by
  have h : (5 ^ 6) % 7 = 1 := sorry
  sorry

end pow_five_2010_mod_seven_l1432_143213


namespace fractional_inequality_solution_l1432_143221

theorem fractional_inequality_solution :
  {x : ℝ | (2 * x - 1) / (x + 1) < 0} = {x : ℝ | -1 < x ∧ x < 1 / 2} := 
by
  sorry

end fractional_inequality_solution_l1432_143221


namespace percent_decaffeinated_second_batch_l1432_143283

theorem percent_decaffeinated_second_batch :
  ∀ (initial_stock : ℝ) (initial_percent : ℝ) (additional_stock : ℝ) (total_percent : ℝ) (second_batch_percent : ℝ),
  initial_stock = 400 →
  initial_percent = 0.20 →
  additional_stock = 100 →
  total_percent = 0.26 →
  (initial_percent * initial_stock + second_batch_percent * additional_stock = total_percent * (initial_stock + additional_stock)) →
  second_batch_percent = 0.50 :=
by
  intros initial_stock initial_percent additional_stock total_percent second_batch_percent
  intros h1 h2 h3 h4 h5
  sorry

end percent_decaffeinated_second_batch_l1432_143283


namespace parabola_standard_equation_l1432_143226

theorem parabola_standard_equation (directrix : ℝ) (h_directrix : directrix = 1) : 
  ∃ (a : ℝ), y^2 = a * x ∧ a = -4 :=
by
  sorry

end parabola_standard_equation_l1432_143226


namespace find_value_of_expression_l1432_143240

theorem find_value_of_expression (a b c d : ℤ) (h₁ : a = -1) (h₂ : b + c = 0) (h₃ : abs d = 2) :
  4 * a + (b + c) - abs (3 * d) = -10 := by
  sorry

end find_value_of_expression_l1432_143240


namespace find_pairs_l1432_143296

-- Define predicative statements for the conditions
def is_integer (x : ℝ) : Prop :=
  ∃ (n : ℤ), x = n

def condition1 (m n : ℕ) : Prop := 
  (n^2 + 1) % (2 * m) = 0

def condition2 (m n : ℕ) : Prop := 
  is_integer (Real.sqrt (2^(n-1) + m + 4))

-- The goal is to find the pairs of positive integers
theorem find_pairs (m n : ℕ) (h1: condition1 m n) (h2: condition2 m n) : 
  (m = 61 ∧ n = 11) :=
sorry

end find_pairs_l1432_143296


namespace range_of_m_l1432_143254

-- Define the constants used in the problem
def a : ℝ := 0.8
def b : ℝ := 1.2

-- Define the logarithmic inequality problem
theorem range_of_m (m : ℝ) : (a^(b^m) < b^(a^m)) → m < 0 := sorry

end range_of_m_l1432_143254


namespace functional_eq_one_l1432_143217

theorem functional_eq_one (f : ℝ → ℝ) (h1 : ∀ x, 0 < x → 0 < f x) 
    (h2 : ∀ x > 0, ∀ y > 0, f x * f (y * f x) = f (x + y)) :
    ∀ x, 0 < x → f x = 1 := 
by
  sorry

end functional_eq_one_l1432_143217


namespace find_f_log_log_3_value_l1432_143251

noncomputable def f (a b : ℝ) (x : ℝ) : ℝ := a * Real.sin x - b * Real.logb 3 (Real.sqrt (x*x + 1) - x) + 1

theorem find_f_log_log_3_value
  (a b : ℝ)
  (h1 : f a b (Real.log 10 / Real.log 3) = 5) :
  f a b (-Real.log 10 / Real.log 3) = -3 :=
  sorry

end find_f_log_log_3_value_l1432_143251


namespace standard_heat_of_formation_Fe2O3_l1432_143260

def Q_form_Al2O3 := 1675.5 -- kJ/mol

def Q1 := 854.2 -- kJ

-- Definition of the standard heat of formation of Fe2O3
def Q_form_Fe2O3 := Q_form_Al2O3 - Q1

-- The proof goal
theorem standard_heat_of_formation_Fe2O3 : Q_form_Fe2O3 = 821.3 := by
  sorry

end standard_heat_of_formation_Fe2O3_l1432_143260


namespace sweatshirt_sales_l1432_143287

variables (S H : ℝ)

theorem sweatshirt_sales (h1 : 13 * S + 9 * H = 370) (h2 : 9 * S + 2 * H = 180) :
  12 * S + 6 * H = 300 :=
sorry

end sweatshirt_sales_l1432_143287


namespace budget_spent_on_utilities_l1432_143288

noncomputable def budget_is_correct : Prop :=
  let total_budget := 100
  let salaries := 60
  let r_and_d := 9
  let equipment := 4
  let supplies := 2
  let degrees_in_circle := 360
  let transportation_degrees := 72
  let transportation_percentage := (transportation_degrees * total_budget) / degrees_in_circle
  let known_percentages := salaries + r_and_d + equipment + supplies + transportation_percentage
  let utilities_percentage := total_budget - known_percentages
  utilities_percentage = 5

theorem budget_spent_on_utilities : budget_is_correct :=
  sorry

end budget_spent_on_utilities_l1432_143288


namespace eval_expression_l1432_143285

theorem eval_expression : (5 + 2 + 6) * 2 / 3 - 4 / 3 = 22 / 3 := sorry

end eval_expression_l1432_143285


namespace sampling_method_D_is_the_correct_answer_l1432_143259

def sampling_method_A_is_simple_random_sampling : Prop :=
  false

def sampling_method_B_is_simple_random_sampling : Prop :=
  false

def sampling_method_C_is_simple_random_sampling : Prop :=
  false

def sampling_method_D_is_simple_random_sampling : Prop :=
  true

theorem sampling_method_D_is_the_correct_answer :
  sampling_method_A_is_simple_random_sampling = false ∧
  sampling_method_B_is_simple_random_sampling = false ∧
  sampling_method_C_is_simple_random_sampling = false ∧
  sampling_method_D_is_simple_random_sampling = true :=
by
  sorry

end sampling_method_D_is_the_correct_answer_l1432_143259


namespace sum_of_variables_is_38_l1432_143243

theorem sum_of_variables_is_38
  (x y z w : ℤ)
  (h₁ : x - y + z = 10)
  (h₂ : y - z + w = 15)
  (h₃ : z - w + x = 9)
  (h₄ : w - x + y = 4) :
  x + y + z + w = 38 := by
  sorry

end sum_of_variables_is_38_l1432_143243


namespace geometric_sequence_common_ratio_l1432_143275

theorem geometric_sequence_common_ratio (a : ℕ → ℝ) (q : ℝ)
    (h1 : a 2 = a 1 * q)
    (h2 : a 5 = a 1 * q ^ 4)
    (h3 : a 2 = 8)
    (h4 : a 5 = 64) :
    q = 2 := 
sorry

end geometric_sequence_common_ratio_l1432_143275


namespace Newville_Academy_fraction_l1432_143298

theorem Newville_Academy_fraction :
  let total_students := 100
  let enjoy_sports := 0.7 * total_students
  let not_enjoy_sports := 0.3 * total_students
  let say_enjoy_right := 0.75 * enjoy_sports
  let say_not_enjoy_wrong := 0.25 * enjoy_sports
  let say_not_enjoy_right := 0.85 * not_enjoy_sports
  let say_enjoy_wrong := 0.15 * not_enjoy_sports
  let say_not_enjoy_total := say_not_enjoy_wrong + say_not_enjoy_right
  let say_not_enjoy_but_enjoy := say_not_enjoy_wrong
  (say_not_enjoy_but_enjoy / say_not_enjoy_total) = (7 / 17) := by
  sorry

end Newville_Academy_fraction_l1432_143298


namespace quadratic_root_relationship_l1432_143233

theorem quadratic_root_relationship (a b c : ℂ) (alpha beta : ℂ) (h1 : a ≠ 0) (h2 : alpha + beta = -b / a) (h3 : alpha * beta = c / a) (h4 : beta = 3 * alpha) : 3 * b ^ 2 = 16 * a * c := by
  sorry

end quadratic_root_relationship_l1432_143233


namespace middle_elementary_students_l1432_143238

theorem middle_elementary_students (S S_PS S_MS S_MR : ℕ) 
  (h1 : S = 12000)
  (h2 : S_PS = (15 * S) / 16)
  (h3 : S_MS = S - S_PS)
  (h4 : S_MR + S_MS = (S_PS) / 2) : 
  S_MR = 4875 :=
by
  sorry

end middle_elementary_students_l1432_143238


namespace sin_double_angle_identity_l1432_143297

open Real

theorem sin_double_angle_identity (α : ℝ) (h : sin (α - π / 4) = 3 / 5) : sin (2 * α) = 7 / 25 :=
by
  sorry

end sin_double_angle_identity_l1432_143297


namespace student_allowance_l1432_143222

theorem student_allowance (A : ℝ) (h1 : A * (2/5) = A - (A * (3/5)))
  (h2 : (A - (A * (2/5))) * (1/3) = ((A - (A * (2/5))) - ((A - (A * (2/5))) * (1/3))) * (1/3))
  (h3 : ((A - (A * (2/5))) - ((A - (A * (2/5))) * (1/3))) = 1.20) :
  A = 3.00 :=
by
  sorry

end student_allowance_l1432_143222


namespace find_ABC_sum_l1432_143235

-- Conditions
def poly (A B C : ℤ) (x : ℤ) := x^3 + A * x^2 + B * x + C
def roots_condition (A B C : ℤ) := poly A B C (-1) = 0 ∧ poly A B C 3 = 0 ∧ poly A B C 4 = 0

-- Proof goal
theorem find_ABC_sum (A B C : ℤ) (h : roots_condition A B C) : A + B + C = 11 :=
sorry

end find_ABC_sum_l1432_143235


namespace proof_problem_l1432_143210

-- Given conditions for propositions p and q
def p (a : ℝ) := ∀ x : ℝ, 1 ≤ x ∧ x ≤ 2 → x^2 - a ≥ 0
def q (a : ℝ) := ∃ x : ℝ, x^2 + 2 * a * x + a + 2 = 0

-- Combined proposition p and q
def p_and_q (a : ℝ) := p a ∧ q a

-- Statement of the proof problem: Prove that p_and_q a → a ≤ -1
theorem proof_problem (a : ℝ) : p_and_q a → (a ≤ -1) :=
by
  sorry

end proof_problem_l1432_143210


namespace range_of_a_l1432_143248

noncomputable def f (a x : ℝ) : ℝ :=
  if x > 1 then x + a / x + 1 else -x^2 + 2 * x

theorem range_of_a (a : ℝ) (h : ∀ x y : ℝ, x ≤ y → f a x ≤ f a y) : -1 ≤ a ∧ a ≤ 1 := 
by
  sorry

end range_of_a_l1432_143248


namespace compute_radii_sum_l1432_143284

def points_on_circle (A B C D : ℝ × ℝ) (r : ℝ) : Prop :=
  let dist := λ p q : ℝ × ℝ => Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2)
  (dist A B) * (dist C D) = (dist A C) * (dist B D)

theorem compute_radii_sum :
  ∃ (r1 r2 : ℝ), points_on_circle (0,0) (-1,-1) (5,2) (6,2) r1
               ∧ points_on_circle (0,0) (-1,-1) (34,14) (35,14) r2
               ∧ r1 > 0
               ∧ r2 > 0
               ∧ r1 < r2
               ∧ r1^2 + r2^2 = 1381 :=
by {
  sorry -- proof not required
}

end compute_radii_sum_l1432_143284


namespace Pam_read_more_than_Harrison_l1432_143257

theorem Pam_read_more_than_Harrison :
  ∀ (assigned : ℕ) (Harrison : ℕ) (Pam : ℕ) (Sam : ℕ),
    assigned = 25 →
    Harrison = assigned + 10 →
    Sam = 2 * Pam →
    Sam = 100 →
    Pam - Harrison = 15 :=
by
  intros assigned Harrison Pam Sam h1 h2 h3 h4
  sorry

end Pam_read_more_than_Harrison_l1432_143257

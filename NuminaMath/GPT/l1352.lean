import Mathlib

namespace NUMINAMATH_GPT_mass_percentage_Ca_in_mixture_l1352_135234

theorem mass_percentage_Ca_in_mixture :
  let mass_CaCO3 := 20.0
  let mass_MgCl2 := 10.0
  let molar_mass_Ca := 40.08
  let molar_mass_C := 12.01
  let molar_mass_O := 16.00
  let molar_mass_CaCO3 := molar_mass_Ca + molar_mass_C + 3 * molar_mass_O
  let mass_Ca_in_CaCO3 := mass_CaCO3 * (molar_mass_Ca / molar_mass_CaCO3)
  let total_mass := mass_CaCO3 + mass_MgCl2
  let percentage_Ca := (mass_Ca_in_CaCO3 / total_mass) * 100
  percentage_Ca = 26.69 :=
by
  let mass_CaCO3 := 20.0
  let mass_MgCl2 := 10.0
  let molar_mass_Ca := 40.08
  let molar_mass_C := 12.01
  let molar_mass_O := 16.00
  let molar_mass_CaCO3 := molar_mass_Ca + molar_mass_C + 3 * molar_mass_O
  let mass_Ca_in_CaCO3 := mass_CaCO3 * (molar_mass_Ca / molar_mass_CaCO3)
  let total_mass := mass_CaCO3 + mass_MgCl2
  let percentage_Ca := (mass_Ca_in_CaCO3 / total_mass) * 100
  have : percentage_Ca = 26.69 := by sorry
  exact this

end NUMINAMATH_GPT_mass_percentage_Ca_in_mixture_l1352_135234


namespace NUMINAMATH_GPT_part1_solution_part2_solution_l1352_135262

-- Part (1)
theorem part1_solution (x : ℝ) : (|x - 2| + |x - 1| ≥ 2) ↔ (x ≥ 2.5 ∨ x ≤ 0.5) := sorry

-- Part (2)
theorem part2_solution (a : ℝ) (h : a > 0) : (∀ x, |a * x - 2| + |a * x - a| ≥ 2) → a ≥ 4 := sorry

end NUMINAMATH_GPT_part1_solution_part2_solution_l1352_135262


namespace NUMINAMATH_GPT_smallest_positive_k_l1352_135238

theorem smallest_positive_k (k a n : ℕ) (h_pos : k > 0) (h_cond : 3^3 + 4^3 + 5^3 = 216) (h_eq : k * 216 = a^n) (h_n : n > 1) : k = 1 :=
by {
    sorry
}

end NUMINAMATH_GPT_smallest_positive_k_l1352_135238


namespace NUMINAMATH_GPT_total_cost_is_716_mom_has_enough_money_l1352_135221

/-- Definition of the price of the table lamp -/
def table_lamp_price : ℕ := 86

/-- Definition of the price of the electric fan -/
def electric_fan_price : ℕ := 185

/-- Definition of the price of the bicycle -/
def bicycle_price : ℕ := 445

/-- The total cost of buying all three items -/
def total_cost : ℕ := table_lamp_price + electric_fan_price + bicycle_price

/-- Mom's money -/
def mom_money : ℕ := 300

/-- Problem 1: Prove that the total cost equals 716 -/
theorem total_cost_is_716 : total_cost = 716 := 
by 
  sorry

/-- Problem 2: Prove that Mom has enough money to buy a table lamp and an electric fan -/
theorem mom_has_enough_money : table_lamp_price + electric_fan_price ≤ mom_money :=
by 
  sorry

end NUMINAMATH_GPT_total_cost_is_716_mom_has_enough_money_l1352_135221


namespace NUMINAMATH_GPT_road_trip_total_miles_l1352_135263

theorem road_trip_total_miles (tracy_miles michelle_miles katie_miles : ℕ) (h_michelle : michelle_miles = 294)
    (h_tracy : tracy_miles = 2 * michelle_miles + 20) (h_katie : michelle_miles = 3 * katie_miles):
  tracy_miles + michelle_miles + katie_miles = 1000 :=
by
  sorry

end NUMINAMATH_GPT_road_trip_total_miles_l1352_135263


namespace NUMINAMATH_GPT_integer_classes_mod4_l1352_135245

theorem integer_classes_mod4:
  (2021 % 4) = 1 ∧ (∀ a b : ℤ, (a % 4 = 2) ∧ (b % 4 = 3) → (a + b) % 4 = 1) := by
  sorry

end NUMINAMATH_GPT_integer_classes_mod4_l1352_135245


namespace NUMINAMATH_GPT_number_of_red_items_l1352_135272

-- Define the mathematics problem
theorem number_of_red_items (R : ℕ) : 
  (23 + 1) + (11 + 1) + R = 66 → 
  R = 30 := 
by 
  intro h
  sorry

end NUMINAMATH_GPT_number_of_red_items_l1352_135272


namespace NUMINAMATH_GPT_average_salary_of_all_workers_l1352_135243

-- Definitions of conditions
def num_technicians : ℕ := 7
def num_total_workers : ℕ := 12
def num_other_workers : ℕ := num_total_workers - num_technicians

def avg_salary_technicians : ℝ := 12000
def avg_salary_others : ℝ := 6000

-- Total salary calculations
def total_salary_technicians : ℝ := num_technicians * avg_salary_technicians
def total_salary_others : ℝ := num_other_workers * avg_salary_others

def total_salary : ℝ := total_salary_technicians + total_salary_others

-- Proof statement: the average salary of all workers is 9500
theorem average_salary_of_all_workers : total_salary / num_total_workers = 9500 :=
by
  sorry

end NUMINAMATH_GPT_average_salary_of_all_workers_l1352_135243


namespace NUMINAMATH_GPT_max_area_of_rectangle_l1352_135203

theorem max_area_of_rectangle (L : ℝ) (hL : L = 16) :
  ∃ (A : ℝ), (∀ (x : ℝ), 0 < x ∧ x < 8 → A = x * (8 - x)) ∧ A = 16 :=
by
  sorry

end NUMINAMATH_GPT_max_area_of_rectangle_l1352_135203


namespace NUMINAMATH_GPT_R_depends_on_d_and_n_l1352_135268

def arith_seq_sum (a d n : ℕ) (S1 S2 S3 : ℕ) : Prop := 
  (S1 = n * (a + (n - 1) * d / 2)) ∧ 
  (S2 = n * (2 * a + (2 * n - 1) * d)) ∧ 
  (S3 = 3 * n * (a + (3 * n - 1) * d / 2))

theorem R_depends_on_d_and_n (a d n S1 S2 S3 : ℕ) 
  (hS1 : S1 = n * (a + (n - 1) * d / 2))
  (hS2 : S2 = n * (2 * a + (2 * n - 1) * d))
  (hS3 : S3 = 3 * n * (a + (3 * n - 1) * d / 2)) 
  : S3 - S2 - S1 = 2 * n^2 * d  :=
by
  sorry

end NUMINAMATH_GPT_R_depends_on_d_and_n_l1352_135268


namespace NUMINAMATH_GPT_find_second_dimension_l1352_135233

theorem find_second_dimension (x : ℕ) 
    (h1 : 12 * x * 16 / (3 * 7 * 2) = 64) : 
    x = 14 := by
    sorry

end NUMINAMATH_GPT_find_second_dimension_l1352_135233


namespace NUMINAMATH_GPT_singer_worked_10_hours_per_day_l1352_135209

noncomputable def hours_per_day_worked_on_one_song (total_songs : ℕ) (days_per_song : ℕ) (total_hours : ℕ) : ℕ :=
  total_hours / (total_songs * days_per_song)

theorem singer_worked_10_hours_per_day :
  hours_per_day_worked_on_one_song 3 10 300 = 10 := 
by
  sorry

end NUMINAMATH_GPT_singer_worked_10_hours_per_day_l1352_135209


namespace NUMINAMATH_GPT_compute_expression_l1352_135261

theorem compute_expression : (23 + 12)^2 - (23 - 12)^2 = 1104 := by
  sorry

end NUMINAMATH_GPT_compute_expression_l1352_135261


namespace NUMINAMATH_GPT_seats_not_occupied_l1352_135270

def seats_per_row : ℕ := 8
def total_rows : ℕ := 12
def seat_utilization_ratio : ℚ := 3 / 4

theorem seats_not_occupied : 
  (seats_per_row * total_rows) - (seats_per_row * seat_utilization_ratio * total_rows) = 24 := 
by
  sorry

end NUMINAMATH_GPT_seats_not_occupied_l1352_135270


namespace NUMINAMATH_GPT_alyssa_photos_vacation_l1352_135239

theorem alyssa_photos_vacation
  (pages_first_section : ℕ)
  (photos_per_page_first_section : ℕ)
  (pages_second_section : ℕ)
  (photos_per_page_second_section : ℕ)
  (pages_total : ℕ)
  (photos_per_page_remaining : ℕ)
  (pages_remaining : ℕ)
  (h_total_pages : pages_first_section + pages_second_section + pages_remaining = pages_total)
  (h_photos_first_section : photos_per_page_first_section = 3)
  (h_photos_second_section : photos_per_page_second_section = 4)
  (h_pages_first_section : pages_first_section = 10)
  (h_pages_second_section : pages_second_section = 10)
  (h_photos_remaining : photos_per_page_remaining = 3)
  (h_pages_total : pages_total = 30)
  (h_pages_remaining : pages_remaining = 10) :
  pages_first_section * photos_per_page_first_section +
  pages_second_section * photos_per_page_second_section +
  pages_remaining * photos_per_page_remaining = 100 := by
sorry

end NUMINAMATH_GPT_alyssa_photos_vacation_l1352_135239


namespace NUMINAMATH_GPT_solve_inequality_l1352_135275

theorem solve_inequality (x : ℝ) : (x^2 + 5 * x - 14 < 0) ↔ (-7 < x ∧ x < 2) :=
sorry

end NUMINAMATH_GPT_solve_inequality_l1352_135275


namespace NUMINAMATH_GPT_sequence_property_l1352_135269

variable (a : ℕ → ℕ)

theorem sequence_property
  (h_bij : Function.Bijective a) (n : ℕ) :
  ∃ k, k < n ∧ a (n - k) < a n ∧ a n < a (n + k) :=
sorry

end NUMINAMATH_GPT_sequence_property_l1352_135269


namespace NUMINAMATH_GPT_sin_1035_eq_neg_sqrt2_div_2_l1352_135216

theorem sin_1035_eq_neg_sqrt2_div_2 : Real.sin (1035 * Real.pi / 180) = - Real.sqrt 2 / 2 := by
    sorry

end NUMINAMATH_GPT_sin_1035_eq_neg_sqrt2_div_2_l1352_135216


namespace NUMINAMATH_GPT_I_consecutive_integers_I_consecutive_even_integers_II_consecutive_integers_II_consecutive_even_integers_II_consecutive_odd_integers_l1352_135208

-- Define the problems
theorem I_consecutive_integers:
  ∃ (x y z : ℕ), 2 * x + y + z = 47 ∧ y = x + 1 ∧ z = x + 2 :=
sorry

theorem I_consecutive_even_integers:
  ¬ ∃ (x y z : ℕ), 2 * x + y + z = 47 ∧ y = x + 2 ∧ z = x + 4 :=
sorry

theorem II_consecutive_integers:
  ∃ (x y z w : ℕ), 2 * x + y + z + w = 47 ∧ y = x + 1 ∧ z = x + 2 ∧ w = x + 3 :=
sorry

theorem II_consecutive_even_integers:
  ¬ ∃ (x y z w : ℕ), 2 * x + y + z + w = 47 ∧ y = x + 2 ∧ z = x + 4 ∧ w = x + 6 :=
sorry

theorem II_consecutive_odd_integers:
  ¬ ∃ (x y z w : ℕ), 2 * x + y + z + w = 47 ∧ y = x + 2 ∧ z = x + 4 ∧ w = x + 6 ∧ x % 2 = 1 ∧ y % 2 = 1 ∧ z % 2 = 1 ∧ w % 2 = 1 :=
sorry

end NUMINAMATH_GPT_I_consecutive_integers_I_consecutive_even_integers_II_consecutive_integers_II_consecutive_even_integers_II_consecutive_odd_integers_l1352_135208


namespace NUMINAMATH_GPT_altitude_point_intersect_and_length_equalities_l1352_135264

variables (A B C D E H : Type)
variables (triangle : A ≠ B ∧ B ≠ C ∧ C ≠ A)
variables (acute : ∀ (a b c : A), True) -- Placeholder for the acute triangle condition
variables (altitude_AD : True) -- Placeholder for the specific definition of altitude AD
variables (altitude_BE : True) -- Placeholder for the specific definition of altitude BE
variables (HD HE AD : ℝ)
variables (BD DC AE EC : ℝ)

theorem altitude_point_intersect_and_length_equalities
  (HD_eq : HD = 3)
  (HE_eq : HE = 4) 
  (sim1 : BD / 3 = (AD + 3) / DC)
  (sim2 : AE / 4 = (BE + 4) / EC)
  (sim3 : 4 * AD = 3 * BE) :
  (BD * DC) - (AE * EC) = 3 * AD - 7 := by
  sorry

end NUMINAMATH_GPT_altitude_point_intersect_and_length_equalities_l1352_135264


namespace NUMINAMATH_GPT_average_price_per_racket_l1352_135228

theorem average_price_per_racket (total_amount : ℕ) (pairs_sold : ℕ) (expected_average : ℚ) 
  (h1 : total_amount = 637) (h2 : pairs_sold = 65) : 
  expected_average = total_amount / pairs_sold := 
by
  sorry

end NUMINAMATH_GPT_average_price_per_racket_l1352_135228


namespace NUMINAMATH_GPT_problem1_l1352_135210

theorem problem1 : 13 + (-24) - (-40) = 29 := by
  sorry

end NUMINAMATH_GPT_problem1_l1352_135210


namespace NUMINAMATH_GPT_future_ratio_l1352_135250

variable (j e : ℕ)

-- Conditions
axiom condition1 : j - 3 = 4 * (e - 3)
axiom condition2 : j - 5 = 5 * (e - 5)

-- Theorem to be proved
theorem future_ratio : ∃ x : ℕ, x = 1 ∧ ((j + x) / (e + x) = 3) := by
  sorry

end NUMINAMATH_GPT_future_ratio_l1352_135250


namespace NUMINAMATH_GPT_ratio_girls_total_members_l1352_135230

theorem ratio_girls_total_members {p_boy p_girl : ℚ} (h_prob_ratio : p_girl = (3/5) * p_boy) (h_total_prob : p_boy + p_girl = 1) :
  p_girl / (p_boy + p_girl) = 3 / 8 :=
by
  sorry

end NUMINAMATH_GPT_ratio_girls_total_members_l1352_135230


namespace NUMINAMATH_GPT_find_initial_students_l1352_135287

def initial_students (S : ℕ) : Prop :=
  S - 4 + 42 = 48 

theorem find_initial_students (S : ℕ) (h : initial_students S) : S = 10 :=
by {
  -- The proof can be filled out here but we skip it using sorry
  sorry
}

end NUMINAMATH_GPT_find_initial_students_l1352_135287


namespace NUMINAMATH_GPT_problem_statement_l1352_135290

-- Initial sequence and Z expansion definition
def initial_sequence := [1, 2, 3]

def z_expand (seq : List ℕ) : List ℕ :=
  match seq with
  | [] => []
  | [a] => [a]
  | a :: b :: rest => a :: (a + b) :: z_expand (b :: rest)

-- Define a_n
def a_sequence (n : ℕ) : List ℕ :=
  Nat.iterate z_expand n initial_sequence

def a_n (n : ℕ) : ℕ :=
  (a_sequence n).sum

-- Define b_n
def b_n (n : ℕ) : ℕ :=
  a_n n - 2

-- Problem statement
theorem problem_statement :
    a_n 1 = 14 ∧
    a_n 2 = 38 ∧
    a_n 3 = 110 ∧
    ∀ n, b_n n = 4 * (3 ^ n) := sorry

end NUMINAMATH_GPT_problem_statement_l1352_135290


namespace NUMINAMATH_GPT_rain_puddle_depth_l1352_135297

theorem rain_puddle_depth
  (rain_rate : ℝ) (wait_time : ℝ) (puddle_area : ℝ) 
  (h_rate : rain_rate = 10) (h_time : wait_time = 3) (h_area : puddle_area = 300) :
  ∃ (depth : ℝ), depth = rain_rate * wait_time :=
by
  use 30
  simp [h_rate, h_time]
  sorry

end NUMINAMATH_GPT_rain_puddle_depth_l1352_135297


namespace NUMINAMATH_GPT_neither_sufficient_nor_necessary_condition_l1352_135236

theorem neither_sufficient_nor_necessary_condition (a b : ℝ) :
  ¬ ((a < 0 ∧ b < 0) → (a * b * (a - b) > 0)) ∧
  ¬ ((a * b * (a - b) > 0) → (a < 0 ∧ b < 0)) :=
by
  sorry

end NUMINAMATH_GPT_neither_sufficient_nor_necessary_condition_l1352_135236


namespace NUMINAMATH_GPT_max_value_of_n_l1352_135282

theorem max_value_of_n (a : ℕ → ℝ) (S : ℕ → ℝ)
  (h_arith_seq : ∀ n, a (n + 1) - a n = a 2 - a 1)
  (h_cond : a 11 / a 10 < -1)
  (h_maximum : ∃ N, ∀ n > N, S n ≤ S N) :
  ∃ N, S N > 0 ∧ ∀ m, S m > 0 → m ≤ N :=
by
  sorry

end NUMINAMATH_GPT_max_value_of_n_l1352_135282


namespace NUMINAMATH_GPT_part_I_part_II_l1352_135279

-- Define the function f(x)
def f (x : ℝ) (m : ℝ) := abs (x + m) + abs (2 * x - 1)

-- Part (I)
theorem part_I (x : ℝ) : (f x (-1) ≤ 2) ↔ (0 ≤ x ∧ x ≤ (4 / 3)) :=
by sorry

-- Part (II)
theorem part_II (m : ℝ) : (∀ x, (3 / 4) ≤ x ∧ x ≤ 2 → f x m ≤ abs (2 * x + 1)) ↔ (-11 / 4) ≤ m ∧ m ≤ 0 :=
by sorry

end NUMINAMATH_GPT_part_I_part_II_l1352_135279


namespace NUMINAMATH_GPT_min_value2k2_minus_4n_l1352_135274

-- We state the problem and set up the conditions
variable (k n : ℝ)
variable (nonneg_k : k ≥ 0)
variable (nonneg_n : n ≥ 0)
variable (eq1 : 2 * k + n = 2)

-- Main statement to prove
theorem min_value2k2_minus_4n : ∃ k n : ℝ, k ≥ 0 ∧ n ≥ 0 ∧ 2 * k + n = 2 ∧ (∀ k' n' : ℝ, k' ≥ 0 ∧ n' ≥ 0 ∧ 2 * k' + n' = 2 → 2 * k'^2 - 4 * n' ≥ -8) := 
sorry

end NUMINAMATH_GPT_min_value2k2_minus_4n_l1352_135274


namespace NUMINAMATH_GPT_sequence_problem_l1352_135273

noncomputable def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
∀ n m k, n ≠ m → a n = a m + (n - m) * k

theorem sequence_problem
  (a : ℕ → ℝ)
  (h_arith : arithmetic_sequence a)
  (h_sum : a 2003 + a 2005 + a 2007 + a 2009 + a 2011 + a 2013 = 120) :
  2 * a 2018 - a 2028 = 20 :=
sorry

end NUMINAMATH_GPT_sequence_problem_l1352_135273


namespace NUMINAMATH_GPT_doughnuts_in_each_box_l1352_135235

theorem doughnuts_in_each_box (total_doughnuts : ℕ) (boxes : ℕ) (h1 : total_doughnuts = 48) (h2 : boxes = 4) : total_doughnuts / boxes = 12 :=
by
  sorry

end NUMINAMATH_GPT_doughnuts_in_each_box_l1352_135235


namespace NUMINAMATH_GPT_evaluate_nested_function_l1352_135289

noncomputable def f (x : ℝ) : ℝ :=
  if x < 1 then 3 * x - 1 / 2 else 2 ^ x

theorem evaluate_nested_function : f (f (1 / 2)) = 2 := 
by
  sorry

end NUMINAMATH_GPT_evaluate_nested_function_l1352_135289


namespace NUMINAMATH_GPT_canonical_equations_of_line_l1352_135211

-- Definitions for the normal vectors of the planes
def n1 : ℝ × ℝ × ℝ := (2, 3, -2)
def n2 : ℝ × ℝ × ℝ := (1, -3, 1)

-- Define the equations of the planes
def plane1 (x y z : ℝ) : Prop := 2 * x + 3 * y - 2 * z + 6 = 0
def plane2 (x y z : ℝ) : Prop := x - 3 * y + z + 3 = 0

-- The canonical equations of the line of intersection
def canonical_eq (x y z : ℝ) : Prop := (z * (-4)) = (y * (-9)) ∧ (z * (-3)) = (x + 3) * (-9)

theorem canonical_equations_of_line :
  ∀ x y z : ℝ, (plane1 x y z) ∧ (plane2 x y z) → canonical_eq x y z :=
by
  sorry

end NUMINAMATH_GPT_canonical_equations_of_line_l1352_135211


namespace NUMINAMATH_GPT_median_of_100_numbers_l1352_135222

theorem median_of_100_numbers (x : Fin 100 → ℝ)
  (h1 : ∀ i j, i ≠ j → x i = 78 → x j = 66 → i = 51 ∧ j = 50 ∨ i = 50 ∧ j = 51)
  (h2 : ∀ i, i ≠ 51 → x 51 = 78)
  (h3 : ∀ i, i ≠ 50 → x 50 = 66) :
  (x 50 + x 51) / 2 = 72 :=
by sorry

end NUMINAMATH_GPT_median_of_100_numbers_l1352_135222


namespace NUMINAMATH_GPT_no_b_gt_4_such_that_143b_is_square_l1352_135242

theorem no_b_gt_4_such_that_143b_is_square :
  ∀ (b : ℕ), 4 < b → ¬ ∃ (n : ℕ), b^2 + 4 * b + 3 = n^2 :=
by sorry

end NUMINAMATH_GPT_no_b_gt_4_such_that_143b_is_square_l1352_135242


namespace NUMINAMATH_GPT_ratio_x_y_half_l1352_135271

variable (x y z : ℝ)

theorem ratio_x_y_half (h1 : (x + 4) / 2 = (y + 9) / (z - 3))
                      (h2 : (y + 9) / (z - 3) = (x + 5) / (z - 5)) :
  x / y = 1 / 2 :=
by
  sorry

end NUMINAMATH_GPT_ratio_x_y_half_l1352_135271


namespace NUMINAMATH_GPT_machines_job_time_l1352_135252

theorem machines_job_time (D : ℝ) (h1 : 15 * D = D * 20 * (3 / 4)) : ¬ ∃ t : ℝ, t = D :=
by
  sorry

end NUMINAMATH_GPT_machines_job_time_l1352_135252


namespace NUMINAMATH_GPT_tangent_line_at_e_intervals_of_monotonicity_l1352_135251
open Real

noncomputable def f (x : ℝ) : ℝ := x * log x

theorem tangent_line_at_e :
  ∃ (y : ℝ → ℝ), (∀ x : ℝ, y x = 2 * x - exp 1) ∧ (y (exp 1) = f (exp 1)) ∧ (deriv f (exp 1) = deriv y (exp 1)) :=
sorry

theorem intervals_of_monotonicity :
  (∀ x : ℝ, 0 < x ∧ x < exp (-1) → deriv f x < 0) ∧ (∀ x : ℝ, exp (-1) < x → deriv f x > 0) :=
sorry

end NUMINAMATH_GPT_tangent_line_at_e_intervals_of_monotonicity_l1352_135251


namespace NUMINAMATH_GPT_sequence_proof_l1352_135240

theorem sequence_proof (S : ℕ → ℝ) (a : ℕ → ℝ)
  (h : ∀ n : ℕ, n > 0 → a n = 2 - S n)
  (hS : ∀ n : ℕ, S (n + 1) = S n + a (n + 1) ) :
  (a 1 = 1 ∧ a 2 = 1/2 ∧ a 3 = 1/4 ∧ a 4 = 1/8) ∧ (∀ n : ℕ, n > 0 → a n = (1/2)^(n-1)) :=
by
  sorry

end NUMINAMATH_GPT_sequence_proof_l1352_135240


namespace NUMINAMATH_GPT_q1_q2_q3_l1352_135214

noncomputable def quadratic_function (a x: ℝ) : ℝ := x^2 - 2 * a * x + a + 2

theorem q1 (a : ℝ) : (∀ {x : ℝ}, quadratic_function a x = 0 → x < 2) ∧ (quadratic_function a 2 > 0) ∧ (2 * a ≠ 0) → a < -1 := 
by 
  sorry

theorem q2 (a : ℝ) : (∀ x : ℝ, quadratic_function a x ≥ -1 - a * x) → -2 ≤ a ∧ a ≤ 6 := 
by 
  sorry
  
theorem q3 (a : ℝ) : (∀ x : ℝ, 0 ≤ x ∧ x ≤ 2 → quadratic_function a x ≤ 4) → a = 2 ∨ a = 2 / 3 := 
by 
  sorry

end NUMINAMATH_GPT_q1_q2_q3_l1352_135214


namespace NUMINAMATH_GPT_max_min_product_xy_l1352_135255

-- Definition of conditions
variables (a x y : ℝ)
def condition_1 : Prop := x + y = a
def condition_2 : Prop := x^2 + y^2 = -a^2 + 2

-- The main theorem statement
theorem max_min_product_xy (a : ℝ) (ha_range : -2 ≤ a ∧ a ≤ 2): 
  (∀ x y : ℝ, condition_1 a x y ∧ condition_2 a x y → (x * y) ≤ (1 / 3)) ∧
  (∀ x y : ℝ, condition_1 a x y ∧ condition_2 a x y → (x * y) ≥ (-1)) :=
sorry

end NUMINAMATH_GPT_max_min_product_xy_l1352_135255


namespace NUMINAMATH_GPT_min_points_condition_met_l1352_135284

noncomputable def min_points_on_circle (L : ℕ) : ℕ := 1304

theorem min_points_condition_met (L : ℕ) (hL : L = 1956) :
  (∀ (points : ℕ → ℕ), (∀ n, points n ≠ points (n + 1) ∧ points n ≠ points (n + 2)) ∧ (∀ n, points n < L)) →
  min_points_on_circle L = 1304 :=
by
  -- Proof steps omitted
  sorry

end NUMINAMATH_GPT_min_points_condition_met_l1352_135284


namespace NUMINAMATH_GPT_total_distance_travelled_l1352_135278

def walking_distance_flat_surface (speed_flat : ℝ) (time_flat : ℝ) : ℝ := speed_flat * time_flat
def running_distance_downhill (speed_downhill : ℝ) (time_downhill : ℝ) : ℝ := speed_downhill * time_downhill
def walking_distance_hilly (speed_hilly_walk : ℝ) (time_hilly_walk : ℝ) : ℝ := speed_hilly_walk * time_hilly_walk
def running_distance_hilly (speed_hilly_run : ℝ) (time_hilly_run : ℝ) : ℝ := speed_hilly_run * time_hilly_run

def total_distance (ds1 ds2 ds3 ds4 : ℝ) : ℝ := ds1 + ds2 + ds3 + ds4

theorem total_distance_travelled :
  let speed_flat := 8
  let time_flat := 3
  let speed_downhill := 24
  let time_downhill := 1.5
  let speed_hilly_walk := 6
  let time_hilly_walk := 2
  let speed_hilly_run := 18
  let time_hilly_run := 1
  total_distance (walking_distance_flat_surface speed_flat time_flat) (running_distance_downhill speed_downhill time_downhill)
                            (walking_distance_hilly speed_hilly_walk time_hilly_walk) (running_distance_hilly speed_hilly_run time_hilly_run) = 90 := 
by
  sorry

end NUMINAMATH_GPT_total_distance_travelled_l1352_135278


namespace NUMINAMATH_GPT_cartesian_to_polar_circle_l1352_135215

open Real

theorem cartesian_to_polar_circle (x y : ℝ) (ρ θ : ℝ) 
  (h1 : x = ρ * cos θ) 
  (h2 : y = ρ * sin θ) 
  (h3 : x^2 + y^2 - 2 * x = 0) : 
  ρ = 2 * cos θ :=
sorry

end NUMINAMATH_GPT_cartesian_to_polar_circle_l1352_135215


namespace NUMINAMATH_GPT_least_common_multiple_of_marble_sharing_l1352_135276

theorem least_common_multiple_of_marble_sharing : 
  Nat.lcm (Nat.lcm (Nat.lcm (Nat.lcm (Nat.lcm 2 4) 5) 7) 8) 10 = 280 :=
sorry

end NUMINAMATH_GPT_least_common_multiple_of_marble_sharing_l1352_135276


namespace NUMINAMATH_GPT_pyramid_volume_is_one_sixth_l1352_135229

noncomputable def volume_of_pyramid_in_cube : ℝ :=
  let edge_length := 1
  let base_area := (1 / 2) * edge_length * edge_length
  let height := edge_length
  (1 / 3) * base_area * height

theorem pyramid_volume_is_one_sixth : volume_of_pyramid_in_cube = 1 / 6 :=
by
  -- Let edge_length = 1, base_area = 1 / 2 * edge_length * edge_length = 1 / 2, 
  -- height = edge_length = 1. Then volume = 1 / 3 * base_area * height = 1 / 6.
  sorry

end NUMINAMATH_GPT_pyramid_volume_is_one_sixth_l1352_135229


namespace NUMINAMATH_GPT_combined_mpg_proof_l1352_135256

noncomputable def combined_mpg (d : ℝ) : ℝ :=
  let ray_mpg := 50
  let tom_mpg := 20
  let alice_mpg := 25
  let total_fuel := (d / ray_mpg) + (d / tom_mpg) + (d / alice_mpg)
  let total_distance := 3 * d
  total_distance / total_fuel

theorem combined_mpg_proof :
  ∀ d : ℝ, d > 0 → combined_mpg d = 300 / 11 :=
by
  intros d hd
  rw [combined_mpg]
  simp only [div_eq_inv_mul, mul_inv, inv_inv]
  sorry

end NUMINAMATH_GPT_combined_mpg_proof_l1352_135256


namespace NUMINAMATH_GPT_xyz_inequality_l1352_135212

-- Definitions for the conditions and the statement of the problem
theorem xyz_inequality (x y z : ℝ) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) 
  (h_ineq : x * y * z ≥ x * y + y * z + z * x) : 
  x * y * z ≥ 3 * (x + y + z) :=
by
  sorry

end NUMINAMATH_GPT_xyz_inequality_l1352_135212


namespace NUMINAMATH_GPT_rectangular_prism_volume_l1352_135232

theorem rectangular_prism_volume (h : ℝ) : 
  ∃ (V : ℝ), V = 120 * h :=
by
  sorry

end NUMINAMATH_GPT_rectangular_prism_volume_l1352_135232


namespace NUMINAMATH_GPT_no_heptagon_cross_section_l1352_135283

-- Define what it means for a plane to intersect a cube and form a shape.
noncomputable def possible_cross_section_shapes (P : Plane) (C : Cube) : Set Polygon :=
  sorry -- Placeholder for the actual definition which involves geometric computations.

-- Prove that a heptagon cannot be one of the possible cross-sectional shapes of a cube.
theorem no_heptagon_cross_section (P : Plane) (C : Cube) : 
  Heptagon ∉ possible_cross_section_shapes P C :=
sorry -- Placeholder for the proof.

end NUMINAMATH_GPT_no_heptagon_cross_section_l1352_135283


namespace NUMINAMATH_GPT_work_efficiency_ratio_l1352_135247

theorem work_efficiency_ratio
  (A B : ℝ)
  (h1 : A + B = 1 / 18)
  (h2 : B = 1 / 27) :
  A / B = 1 / 2 := 
by
  sorry

end NUMINAMATH_GPT_work_efficiency_ratio_l1352_135247


namespace NUMINAMATH_GPT_find_davids_marks_in_physics_l1352_135285

theorem find_davids_marks_in_physics (marks_english : ℕ) (marks_math : ℕ) (marks_chemistry : ℕ) (marks_biology : ℕ)
  (average_marks : ℕ) (num_subjects : ℕ) (H1 : marks_english = 61) 
  (H2 : marks_math = 65) (H3 : marks_chemistry = 67) 
  (H4 : marks_biology = 85) (H5 : average_marks = 72) (H6 : num_subjects = 5) :
  ∃ (marks_physics : ℕ), marks_physics = 82 :=
by
  sorry

end NUMINAMATH_GPT_find_davids_marks_in_physics_l1352_135285


namespace NUMINAMATH_GPT_minimum_x_condition_l1352_135260

theorem minimum_x_condition (x y : ℝ) (hxy_pos : 0 < x ∧ 0 < y) (h : x - 2 * y = (x + 16 * y) / (2 * x * y)) : 
  x ≥ 4 :=
sorry

end NUMINAMATH_GPT_minimum_x_condition_l1352_135260


namespace NUMINAMATH_GPT_problem_statement_l1352_135299

-- Defining the terms x, y, and d as per the problem conditions
def x : ℕ := 2351
def y : ℕ := 2250
def d : ℕ := 121

-- Stating the proof problem in Lean
theorem problem_statement : (x - y)^2 / d = 84 := by
  sorry

end NUMINAMATH_GPT_problem_statement_l1352_135299


namespace NUMINAMATH_GPT_certain_event_among_options_l1352_135204

-- Definition of the proof problem
theorem certain_event_among_options (is_random_A : Prop) (is_random_C : Prop) (is_random_D : Prop) (is_certain_B : Prop) :
  (is_random_A → (¬is_certain_B)) ∧
  (is_random_C → (¬is_certain_B)) ∧
  (is_random_D → (¬is_certain_B)) ∧
  (is_certain_B ∧ ((¬is_random_A) ∧ (¬is_random_C) ∧ (¬is_random_D))) :=
by
  sorry

end NUMINAMATH_GPT_certain_event_among_options_l1352_135204


namespace NUMINAMATH_GPT_compute_difference_l1352_135257

def bin_op (x y : ℤ) : ℤ := x^2 * y - 3 * x

theorem compute_difference :
  (bin_op 5 3) - (bin_op 3 5) = 24 := by
  sorry

end NUMINAMATH_GPT_compute_difference_l1352_135257


namespace NUMINAMATH_GPT_num_black_cars_l1352_135246

theorem num_black_cars (total_cars : ℕ) (one_third_blue : ℚ) (one_half_red : ℚ) 
  (h1 : total_cars = 516) (h2 : one_third_blue = 1/3) (h3 : one_half_red = 1/2) :
  total_cars - (total_cars * one_third_blue + total_cars * one_half_red) = 86 :=
by
  sorry

end NUMINAMATH_GPT_num_black_cars_l1352_135246


namespace NUMINAMATH_GPT_correct_calculation_l1352_135244

theorem correct_calculation (a b : ℕ) : a^3 * b^3 = (a * b)^3 :=
sorry

end NUMINAMATH_GPT_correct_calculation_l1352_135244


namespace NUMINAMATH_GPT_intersection_M_N_l1352_135202

open Set

def M : Set ℝ := { x | (x - 1)^2 < 4 }
def N : Set ℝ := {-1, 0, 1, 2, 3}

theorem intersection_M_N : M ∩ N = {0, 1, 2} := 
by 
sorry

end NUMINAMATH_GPT_intersection_M_N_l1352_135202


namespace NUMINAMATH_GPT_estimate_red_balls_l1352_135227

-- Definitions based on conditions
def total_balls : ℕ := 20
def total_draws : ℕ := 100
def red_draws : ℕ := 30

-- The theorem statement
theorem estimate_red_balls (h1 : total_balls = 20) (h2 : total_draws = 100) (h3 : red_draws = 30) :
  (total_balls * (red_draws / total_draws) : ℤ) = 6 := 
by
  sorry

end NUMINAMATH_GPT_estimate_red_balls_l1352_135227


namespace NUMINAMATH_GPT_set_of_values_a_l1352_135217

theorem set_of_values_a (a : ℝ) : (2 ∉ {x : ℝ | x - a < 0}) ↔ (a ≤ 2) :=
by
  sorry

end NUMINAMATH_GPT_set_of_values_a_l1352_135217


namespace NUMINAMATH_GPT_largest_angle_of_consecutive_odd_int_angles_is_125_l1352_135224

-- Definitions for a convex hexagon with six consecutive odd integer interior angles
def is_consecutive_odd_integers (xs : List ℕ) : Prop :=
  ∀ n, 0 ≤ n ∧ n < 5 → xs.get! n + 2 = xs.get! (n + 1)

def hexagon_angles_sum_720 (xs : List ℕ) : Prop :=
  xs.length = 6 ∧ xs.sum = 720

-- Main theorem statement
theorem largest_angle_of_consecutive_odd_int_angles_is_125 (xs : List ℕ) 
(h1 : is_consecutive_odd_integers xs) 
(h2 : hexagon_angles_sum_720 xs) : 
  xs.maximum = 125 := 
sorry

end NUMINAMATH_GPT_largest_angle_of_consecutive_odd_int_angles_is_125_l1352_135224


namespace NUMINAMATH_GPT_area_of_triangle_AEB_is_correct_l1352_135291

noncomputable def area_triangle_AEB : ℚ :=
by
  -- Definitions of given conditions
  let AB := 5
  let BC := 3
  let DF := 1
  let GC := 2

  -- Conditions of the problem
  have h1 : AB = 5 := rfl
  have h2 : BC = 3 := rfl
  have h3 : DF = 1 := rfl
  have h4 : GC = 2 := rfl

  -- The goal to prove
  exact 25 / 2

-- Statement in Lean 4 with the conditions and the correct answer
theorem area_of_triangle_AEB_is_correct :
  area_triangle_AEB = 25 / 2 := sorry -- The proof is omitted for this example

end NUMINAMATH_GPT_area_of_triangle_AEB_is_correct_l1352_135291


namespace NUMINAMATH_GPT_sin_cos_tan_min_value_l1352_135258

open Real

theorem sin_cos_tan_min_value :
  ∀ x : ℝ, (sin x)^2 + (cos x)^2 = 1 → (sin x)^4 + (cos x)^4 + (tan x)^2 ≥ 3/2 :=
by
  sorry

end NUMINAMATH_GPT_sin_cos_tan_min_value_l1352_135258


namespace NUMINAMATH_GPT_min_groups_required_l1352_135223

-- Define the conditions
def total_children : ℕ := 30
def max_children_per_group : ℕ := 12
def largest_divisor (n : ℕ) (d : ℕ) : Prop := d ∣ n ∧ d ≤ max_children_per_group

-- Define the property that we are interested in: the minimum number of groups required
def min_num_groups (total : ℕ) (group_size : ℕ) : ℕ := total / group_size

-- Prove the minimum number of groups is 3 given the conditions
theorem min_groups_required : ∃ d, largest_divisor total_children d ∧ min_num_groups total_children d = 3 :=
sorry

end NUMINAMATH_GPT_min_groups_required_l1352_135223


namespace NUMINAMATH_GPT_compute_u2_plus_v2_l1352_135225

theorem compute_u2_plus_v2 (u v : ℝ) (hu : 1 < u) (hv : 1 < v)
  (h : (Real.log u / Real.log 3)^4 + (Real.log v / Real.log 7)^4 = 10 * (Real.log u / Real.log 3) * (Real.log v / Real.log 7)) :
  u^2 + v^2 = 3^(Real.sqrt 5) + 7^(Real.sqrt 5) :=
by
  sorry

end NUMINAMATH_GPT_compute_u2_plus_v2_l1352_135225


namespace NUMINAMATH_GPT_officers_on_duty_l1352_135292

theorem officers_on_duty
  (F : ℕ)                             -- Total female officers on the police force
  (on_duty_percentage : ℕ)            -- On duty percentage of female officers
  (H1 : on_duty_percentage = 18)      -- 18% of the female officers were on duty
  (H2 : F = 500)                      -- There were 500 female officers on the police force
  : ∃ T : ℕ, T = 2 * (on_duty_percentage * F) / 100 ∧ T = 180 :=
by
  sorry

end NUMINAMATH_GPT_officers_on_duty_l1352_135292


namespace NUMINAMATH_GPT_jerome_money_left_l1352_135254

-- Given conditions
def half_of_money (m : ℕ) : Prop := m / 2 = 43
def amount_given_to_meg (x : ℕ) : Prop := x = 8
def amount_given_to_bianca (x : ℕ) : Prop := x = 3 * 8

-- Problem statement
theorem jerome_money_left (m : ℕ) (x : ℕ) (y : ℕ) (h1 : half_of_money m) (h2 : amount_given_to_meg x) (h3 : amount_given_to_bianca y) : m - x - y = 54 :=
sorry

end NUMINAMATH_GPT_jerome_money_left_l1352_135254


namespace NUMINAMATH_GPT_gabrielle_saw_more_birds_l1352_135213

def birds_seen (robins cardinals blue_jays : Nat) : Nat :=
  robins + cardinals + blue_jays

def percentage_difference (g c : Nat) : Nat :=
  ((g - c) * 100) / c

theorem gabrielle_saw_more_birds :
  let gabrielle := birds_seen 5 4 3
  let chase := birds_seen 2 5 3
  percentage_difference gabrielle chase = 20 := 
by
  sorry

end NUMINAMATH_GPT_gabrielle_saw_more_birds_l1352_135213


namespace NUMINAMATH_GPT_original_number_in_magician_game_l1352_135265

theorem original_number_in_magician_game (a b c : ℕ) (habc : 100 * a + 10 * b + c = 332) (N : ℕ) (hN : N = 4332) :
    222 * (a + b + c) = 4332 → 100 * a + 10 * b + c = 332 :=
by 
  sorry

end NUMINAMATH_GPT_original_number_in_magician_game_l1352_135265


namespace NUMINAMATH_GPT_find_n_l1352_135267

theorem find_n (n : ℕ) (h1 : Nat.gcd n 180 = 12) (h2 : Nat.lcm n 180 = 720) : n = 48 := 
by
  sorry

end NUMINAMATH_GPT_find_n_l1352_135267


namespace NUMINAMATH_GPT_ratio_of_surface_areas_of_spheres_l1352_135205

theorem ratio_of_surface_areas_of_spheres (V1 V2 S1 S2 : ℝ) 
(h : V1 / V2 = 8 / 27) 
(h1 : S1 = 4 * π * (V1^(2/3)) / (2 * π)^(2/3))
(h2 : S2 = 4 * π * (V2^(2/3)) / (3 * π)^(2/3)) :
S1 / S2 = 4 / 9 :=
sorry

end NUMINAMATH_GPT_ratio_of_surface_areas_of_spheres_l1352_135205


namespace NUMINAMATH_GPT_largest_triangle_perimeter_maximizes_l1352_135259

theorem largest_triangle_perimeter_maximizes 
  (y : ℤ) 
  (h1 : 3 ≤ y) 
  (h2 : y < 16) : 
  (7 + 9 + y) = 31 ↔ y = 15 := 
by 
  sorry

end NUMINAMATH_GPT_largest_triangle_perimeter_maximizes_l1352_135259


namespace NUMINAMATH_GPT_lowest_possible_sale_price_percentage_l1352_135200

noncomputable def list_price : ℝ := 80
noncomputable def max_initial_discount_percent : ℝ := 0.5
noncomputable def summer_sale_discount_percent : ℝ := 0.2
noncomputable def membership_discount_percent : ℝ := 0.1
noncomputable def coupon_discount_percent : ℝ := 0.05

theorem lowest_possible_sale_price_percentage :
  let max_initial_discount := max_initial_discount_percent * list_price
  let summer_sale_discount := summer_sale_discount_percent * list_price
  let membership_discount := membership_discount_percent * list_price
  let coupon_discount := coupon_discount_percent * list_price
  let lowest_sale_price := list_price * (1 - max_initial_discount_percent) - summer_sale_discount - membership_discount - coupon_discount
  (lowest_sale_price / list_price) * 100 = 15 :=
by
  sorry

end NUMINAMATH_GPT_lowest_possible_sale_price_percentage_l1352_135200


namespace NUMINAMATH_GPT_train_lengths_l1352_135281

variable (P L_A L_B : ℝ)

noncomputable def speedA := 180 * 1000 / 3600
noncomputable def speedB := 240 * 1000 / 3600

-- Train A crosses platform P in one minute
axiom hA : speedA * 60 = L_A + P

-- Train B crosses platform P in 45 seconds
axiom hB : speedB * 45 = L_B + P

-- Sum of the lengths of Train A and platform P is twice the length of Train B
axiom hSum : L_A + P = 2 * L_B

theorem train_lengths : L_A = 1500 ∧ L_B = 1500 :=
by
  sorry

end NUMINAMATH_GPT_train_lengths_l1352_135281


namespace NUMINAMATH_GPT_one_greater_than_17_over_10_l1352_135296

theorem one_greater_than_17_over_10 (a b c : ℝ) (h_pos : a > 0 ∧ b > 0 ∧ c > 0) (h_eq : a + b + c = a * b * c) : 
  a > 17 / 10 ∨ b > 17 / 10 ∨ c > 17 / 10 :=
by
  sorry

end NUMINAMATH_GPT_one_greater_than_17_over_10_l1352_135296


namespace NUMINAMATH_GPT_sum_of_three_numbers_l1352_135286

theorem sum_of_three_numbers (a b c : ℕ) (mean_least difference greatest_diff : ℕ)
  (h1 : a ≤ b) (h2 : b ≤ c)
  (h3 : mean_least = 8) (h4 : greatest_diff = 25)
  (h5 : c - a = 26)
  (h6 : (a + b + c) / 3 = a + mean_least) 
  (h7 : (a + b + c) / 3 = c - greatest_diff) : 
a + b + c = 81 := 
sorry

end NUMINAMATH_GPT_sum_of_three_numbers_l1352_135286


namespace NUMINAMATH_GPT_constant_term_binomial_expansion_l1352_135249

theorem constant_term_binomial_expansion : 
  let a := (1 : ℚ) / (x : ℚ) -- Note: Here 'x' is not bound, in actual Lean code x should be a declared variable in ℚ.
  let b := 2 * (x : ℚ)
  let n := 6
  let T (r : ℕ) := (Nat.choose n r : ℚ) * a^(n - r) * b^r
  (T 3) = (160 : ℚ) := by
  sorry

end NUMINAMATH_GPT_constant_term_binomial_expansion_l1352_135249


namespace NUMINAMATH_GPT_max_remaining_grapes_l1352_135207

theorem max_remaining_grapes (x : ℕ) : x % 7 ≤ 6 :=
  sorry

end NUMINAMATH_GPT_max_remaining_grapes_l1352_135207


namespace NUMINAMATH_GPT_zain_coin_total_l1352_135218

def zain_coins (q d n : ℕ) := q + d + n
def emerie_quarters : ℕ := 6
def emerie_dimes : ℕ := 7
def emerie_nickels : ℕ := 5
def zain_quarters : ℕ := emerie_quarters + 10
def zain_dimes : ℕ := emerie_dimes + 10
def zain_nickels : ℕ := emerie_nickels + 10

theorem zain_coin_total : zain_coins zain_quarters zain_dimes zain_nickels = 48 := 
by
  unfold zain_coins zain_quarters zain_dimes zain_nickels emerie_quarters emerie_dimes emerie_nickels
  rfl

end NUMINAMATH_GPT_zain_coin_total_l1352_135218


namespace NUMINAMATH_GPT_total_cost_with_discounts_l1352_135201

theorem total_cost_with_discounts :
  let red_roses := 2 * 12
  let white_roses := 1 * 12
  let yellow_roses := 2 * 12
  let cost_red := red_roses * 6
  let cost_white := white_roses * 7
  let cost_yellow := yellow_roses * 5
  let total_cost_before_discount := cost_red + cost_white + cost_yellow
  let first_discount := 0.15 * total_cost_before_discount
  let cost_after_first_discount := total_cost_before_discount - first_discount
  let additional_discount := 0.10 * cost_after_first_discount
  let total_cost := cost_after_first_discount - additional_discount
  total_cost = 266.22 := by
  sorry

end NUMINAMATH_GPT_total_cost_with_discounts_l1352_135201


namespace NUMINAMATH_GPT_odd_if_and_only_if_m_even_l1352_135298

variables (o n m : ℕ)

theorem odd_if_and_only_if_m_even
  (h_o_odd : o % 2 = 1) :
  ((o^3 + n*o + m) % 2 = 1) ↔ (m % 2 = 0) :=
sorry

end NUMINAMATH_GPT_odd_if_and_only_if_m_even_l1352_135298


namespace NUMINAMATH_GPT_parabola_equation_maximum_area_of_triangle_l1352_135231

-- Definitions of the conditions
def parabola_eq (x y : ℝ) (p : ℝ) : Prop := x^2 = 2 * p * y ∧ p > 0
def distances_equal (AO AF : ℝ) : Prop := AO = 3 / 2 ∧ AF = 3 / 2
def line_eq (x k b y : ℝ) : Prop := y = k * x + b
def midpoint_y (y1 y2 : ℝ) : Prop := (y1 + y2) / 2 = 1

-- Part (I)
theorem parabola_equation (p : ℝ) (x y AO AF : ℝ) (h1 : parabola_eq x y p)
  (h2 : distances_equal AO AF) :
  x^2 = 4 * y :=
sorry

-- Part (II)
theorem maximum_area_of_triangle (p k b AO AF x1 y1 x2 y2 : ℝ)
  (h1 : parabola_eq x1 y1 p) (h2 : parabola_eq x2 y2 p)
  (h3 : distances_equal AO AF) (h4 : line_eq x1 k b y1) 
  (h5 : line_eq x2 k b y2) (h6 : midpoint_y y1 y2)
  : ∃ (area : ℝ), area = 2 :=
sorry

end NUMINAMATH_GPT_parabola_equation_maximum_area_of_triangle_l1352_135231


namespace NUMINAMATH_GPT_symmetric_curve_eq_l1352_135280

-- Define the original curve equation and line of symmetry
def original_curve (x y : ℝ) : Prop := y^2 = 4 * x
def line_of_symmetry (x : ℝ) : Prop := x = 2

-- The equivalent Lean 4 statement
theorem symmetric_curve_eq (x y : ℝ) (hx : line_of_symmetry 2) :
  (∀ (x' y' : ℝ), original_curve (4 - x') y' → y^2 = 16 - 4 * x) :=
sorry

end NUMINAMATH_GPT_symmetric_curve_eq_l1352_135280


namespace NUMINAMATH_GPT_company_profits_ratio_l1352_135293

def companyN_2008_profits (RN : ℝ) : ℝ := 0.08 * RN
def companyN_2009_profits (RN : ℝ) : ℝ := 0.15 * (0.8 * RN)
def companyN_2010_profits (RN : ℝ) : ℝ := 0.10 * (1.3 * 0.8 * RN)

def companyM_2008_profits (RM : ℝ) : ℝ := 0.12 * RM
def companyM_2009_profits (RM : ℝ) : ℝ := 0.18 * RM
def companyM_2010_profits (RM : ℝ) : ℝ := 0.14 * RM

def total_profits_N (RN : ℝ) : ℝ :=
  companyN_2008_profits RN + companyN_2009_profits RN + companyN_2010_profits RN

def total_profits_M (RM : ℝ) : ℝ :=
  companyM_2008_profits RM + companyM_2009_profits RM + companyM_2010_profits RM

theorem company_profits_ratio (RN RM : ℝ) :
  total_profits_N RN / total_profits_M RM = (0.304 * RN) / (0.44 * RM) :=
by
  unfold total_profits_N companyN_2008_profits companyN_2009_profits companyN_2010_profits
  unfold total_profits_M companyM_2008_profits companyM_2009_profits companyM_2010_profits
  simp
  sorry

end NUMINAMATH_GPT_company_profits_ratio_l1352_135293


namespace NUMINAMATH_GPT_smallest_value_of_a_l1352_135219

theorem smallest_value_of_a (a b c d : ℤ) (h1 : (a - 2 * b) > 0) (h2 : (b - 3 * c) > 0) (h3 : (c - 4 * d) > 0) (h4 : d > 100) : a ≥ 2433 := sorry

end NUMINAMATH_GPT_smallest_value_of_a_l1352_135219


namespace NUMINAMATH_GPT_antonio_age_in_months_l1352_135226

-- Definitions based on the conditions
def is_twice_as_old (isabella_age antonio_age : ℕ) : Prop :=
  isabella_age = 2 * antonio_age

def future_age (current_age months_future : ℕ) : ℕ :=
  current_age + months_future

-- Given the conditions
variables (isabella_age antonio_age : ℕ)
variables (future_age_18months target_age : ℕ)

-- Conditions
axiom condition1 : is_twice_as_old isabella_age antonio_age
axiom condition2 : future_age_18months = 18
axiom condition3 : target_age = 10 * 12

-- Assertion that we need to prove
theorem antonio_age_in_months :
  ∃ (antonio_age : ℕ), future_age isabella_age future_age_18months = target_age → antonio_age = 51 :=
by
  sorry

end NUMINAMATH_GPT_antonio_age_in_months_l1352_135226


namespace NUMINAMATH_GPT_sin_thirty_degrees_l1352_135206

theorem sin_thirty_degrees : Real.sin (Real.pi / 6) = 1 / 2 := 
by sorry

end NUMINAMATH_GPT_sin_thirty_degrees_l1352_135206


namespace NUMINAMATH_GPT_tens_digit_of_2013_pow_2018_minus_2019_l1352_135266

theorem tens_digit_of_2013_pow_2018_minus_2019 :
  (2013 ^ 2018 - 2019) % 100 / 10 % 10 = 5 := sorry

end NUMINAMATH_GPT_tens_digit_of_2013_pow_2018_minus_2019_l1352_135266


namespace NUMINAMATH_GPT_total_pies_sold_l1352_135294

theorem total_pies_sold :
  let shepherd_slices := 52
  let chicken_slices := 80
  let shepherd_pieces_per_pie := 4
  let chicken_pieces_per_pie := 5
  let shepherd_pies := shepherd_slices / shepherd_pieces_per_pie
  let chicken_pies := chicken_slices / chicken_pieces_per_pie
  shepherd_pies + chicken_pies = 29 :=
by
  sorry

end NUMINAMATH_GPT_total_pies_sold_l1352_135294


namespace NUMINAMATH_GPT_tv_purchase_time_l1352_135295

-- Define the constants
def monthly_income : ℕ := 30000
def food_expenses : ℕ := 15000
def utilities_expenses : ℕ := 5000
def other_expenses : ℕ := 2500
def current_savings : ℕ := 10000
def tv_cost : ℕ := 25000

-- Define the total expenses
def total_expenses : ℕ := food_expenses + utilities_expenses + other_expenses

-- Define the disposable income
def disposable_income : ℕ := monthly_income - total_expenses

-- Define the amount needed to buy the TV
def amount_needed : ℕ := tv_cost - current_savings

-- Define the number of months needed to save the amount needed
def number_of_months : ℕ := amount_needed / disposable_income

-- The theorem specifying that we need 2 months to save enough money for the TV
theorem tv_purchase_time : number_of_months = 2 := by
  sorry

end NUMINAMATH_GPT_tv_purchase_time_l1352_135295


namespace NUMINAMATH_GPT_sum_series_equals_l1352_135288

theorem sum_series_equals :
  (∑' n : ℕ, if n ≥ 2 then 1 / (n * (n + 3)) else 0) = 13 / 36 :=
by
  sorry

end NUMINAMATH_GPT_sum_series_equals_l1352_135288


namespace NUMINAMATH_GPT_smallest_integer_satisfying_conditions_l1352_135241

theorem smallest_integer_satisfying_conditions :
  ∃ M : ℕ, M % 7 = 6 ∧ M % 8 = 7 ∧ M % 9 = 8 ∧ M % 10 = 9 ∧ M % 11 = 10 ∧ M % 12 = 11 ∧ M = 27719 := by
  sorry

end NUMINAMATH_GPT_smallest_integer_satisfying_conditions_l1352_135241


namespace NUMINAMATH_GPT_minimum_x_plus_y_l1352_135237

theorem minimum_x_plus_y (x y : ℕ) (hx_pos : 0 < x) (hy_pos : 0 < y) 
    (h1 : x - y < 1) (h2 : 2 * x - y > 2) (h3 : x < 5) : 
    x + y ≥ 6 :=
sorry

end NUMINAMATH_GPT_minimum_x_plus_y_l1352_135237


namespace NUMINAMATH_GPT_investment_a_l1352_135248

/-- Given:
  * b's profit share is Rs. 1800,
  * the difference between a's and c's profit shares is Rs. 720,
  * b invested Rs. 10000,
  * c invested Rs. 12000,
  prove that a invested Rs. 16000. -/
theorem investment_a (P_b : ℝ) (P_a : ℝ) (P_c : ℝ) (B : ℝ) (C : ℝ) (A : ℝ)
  (h1 : P_b = 1800)
  (h2 : P_a - P_c = 720)
  (h3 : B = 10000)
  (h4 : C = 12000)
  (h5 : P_b / B = P_c / C)
  (h6 : P_a / A = P_b / B) : A = 16000 :=
sorry

end NUMINAMATH_GPT_investment_a_l1352_135248


namespace NUMINAMATH_GPT_max_consecutive_integers_sum_le_500_l1352_135277

def consecutive_sum (n : ℕ) : ℕ :=
  -- Formula for sum starting from 3
  (n * (n + 1)) / 2 - 3

theorem max_consecutive_integers_sum_le_500 : ∃ n : ℕ, consecutive_sum n ≤ 500 ∧ ∀ m : ℕ, m > n → consecutive_sum m > 500 :=
by
  sorry

end NUMINAMATH_GPT_max_consecutive_integers_sum_le_500_l1352_135277


namespace NUMINAMATH_GPT_tan_value_l1352_135220

theorem tan_value (x : ℝ) (hx : x ∈ Set.Ioo (-π / 2) 0) (hcos : Real.cos x = 4 / 5) : Real.tan x = -3 / 4 :=
sorry

end NUMINAMATH_GPT_tan_value_l1352_135220


namespace NUMINAMATH_GPT_total_money_is_220_l1352_135253

-- Define the amounts on Table A, B, and C
def tableA := 40
def tableC := tableA + 20
def tableB := 2 * tableC

-- Define the total amount of money on all tables
def total_money := tableA + tableB + tableC

-- The main theorem to prove
theorem total_money_is_220 : total_money = 220 :=
by
  sorry

end NUMINAMATH_GPT_total_money_is_220_l1352_135253

import Mathlib

namespace NUMINAMATH_GPT_competition_results_l1355_135533

variables (x : ℝ) (freq1 freq3 freq4 freq5 freq2 : ℝ)

/-- Axiom: Given frequencies of groups and total frequency, determine the total number of participants and the probability of an excellent score -/
theorem competition_results :
  freq1 = 0.30 ∧
  freq3 = 0.15 ∧
  freq4 = 0.10 ∧
  freq5 = 0.05 ∧
  freq2 = 40 / x ∧
  (freq1 + freq2 + freq3 + freq4 + freq5 = 1) ∧
  (x * freq2 = 40) →
  x = 100 ∧ (freq4 + freq5 = 0.15) := sorry

end NUMINAMATH_GPT_competition_results_l1355_135533


namespace NUMINAMATH_GPT_hexagon_largest_angle_l1355_135593

theorem hexagon_largest_angle (x : ℝ) (h : 3 * x + 3 * x + 3 * x + 4 * x + 5 * x + 6 * x = 720) : 
  6 * x = 180 :=
by
  sorry

end NUMINAMATH_GPT_hexagon_largest_angle_l1355_135593


namespace NUMINAMATH_GPT_mean_of_squares_eq_l1355_135552

noncomputable def sum_of_squares (n : ℕ) : ℚ := (n * (n + 1) * (2 * n + 1)) / 6

noncomputable def arithmetic_mean_of_squares (n : ℕ) : ℚ := sum_of_squares n / n

theorem mean_of_squares_eq (n : ℕ) (h : n ≠ 0) : arithmetic_mean_of_squares n = ((n + 1) * (2 * n + 1)) / 6 :=
by
  sorry

end NUMINAMATH_GPT_mean_of_squares_eq_l1355_135552


namespace NUMINAMATH_GPT_ratio_new_radius_l1355_135525

theorem ratio_new_radius (r R h : ℝ) (h₀ : π * r^2 * h = 6) (h₁ : π * R^2 * h = 186) : R / r = Real.sqrt 31 :=
by
  sorry

end NUMINAMATH_GPT_ratio_new_radius_l1355_135525


namespace NUMINAMATH_GPT_probability_of_selecting_double_l1355_135516

-- Define the conditions and the question
def total_integers : ℕ := 13

def number_of_doubles : ℕ := total_integers

def total_pairings : ℕ := 
  (total_integers * (total_integers + 1)) / 2

def probability_double : ℚ := 
  number_of_doubles / total_pairings

-- Statement to be proved 
theorem probability_of_selecting_double : 
  probability_double = 1/7 := 
sorry

end NUMINAMATH_GPT_probability_of_selecting_double_l1355_135516


namespace NUMINAMATH_GPT_find_a_2b_3c_value_l1355_135519

-- Problem statement and conditions
theorem find_a_2b_3c_value (a b c : ℝ)
  (h : ∀ x : ℝ, (x < -1 ∨ abs (x - 10) ≤ 2) ↔ (x - a) * (x - b) / (x - c) ≤ 0)
  (h_ab : a < b) : a + 2 * b + 3 * c = 29 := 
sorry

end NUMINAMATH_GPT_find_a_2b_3c_value_l1355_135519


namespace NUMINAMATH_GPT_batsman_boundaries_l1355_135567

theorem batsman_boundaries
  (total_runs : ℕ)
  (boundaries : ℕ)
  (sixes : ℕ)
  (runs_by_running : ℕ)
  (runs_by_sixes : ℕ)
  (runs_by_boundaries : ℕ)
  (half_runs : ℕ)
  (sixes_runs : ℕ)
  (boundaries_runs : ℕ)
  (total_runs_eq : total_runs = 120)
  (sixes_eq : sixes = 8)
  (half_total_eq : half_runs = total_runs / 2)
  (runs_by_running_eq : runs_by_running = half_runs)
  (sixes_runs_eq : runs_by_sixes = sixes * 6)
  (boundaries_runs_eq : runs_by_boundaries = total_runs - runs_by_running - runs_by_sixes)
  (boundaries_eq : boundaries_runs = boundaries * 4) :
  boundaries = 3 :=
by
  sorry

end NUMINAMATH_GPT_batsman_boundaries_l1355_135567


namespace NUMINAMATH_GPT_percentage_correct_l1355_135521

noncomputable def part : ℝ := 172.8
noncomputable def whole : ℝ := 450.0
noncomputable def percentage (part whole : ℝ) := (part / whole) * 100

theorem percentage_correct : percentage part whole = 38.4 := by
  sorry

end NUMINAMATH_GPT_percentage_correct_l1355_135521


namespace NUMINAMATH_GPT_train_truck_load_l1355_135515

variables (x y : ℕ)

def transport_equations (x y : ℕ) : Prop :=
  (2 * x + 5 * y = 120) ∧ (8 * x + 10 * y = 440)

def tonnage (x y : ℕ) : ℕ :=
  5 * x + 8 * y

theorem train_truck_load
  (x y : ℕ)
  (h : transport_equations x y) :
  tonnage x y = 282 :=
sorry

end NUMINAMATH_GPT_train_truck_load_l1355_135515


namespace NUMINAMATH_GPT_exterior_angle_sum_l1355_135598

theorem exterior_angle_sum (n : ℕ) (h_n : 3 ≤ n) :
  let polygon_exterior_angle_sum := 360
  let triangle_exterior_angle_sum := 0
  (polygon_exterior_angle_sum + triangle_exterior_angle_sum = 360) :=
by sorry

end NUMINAMATH_GPT_exterior_angle_sum_l1355_135598


namespace NUMINAMATH_GPT_find_3m_plus_n_l1355_135559

theorem find_3m_plus_n (m n : ℕ) (h1 : m > n) (h2 : 3 * (3 * m * n - 2)^2 - 2 * (3 * m - 3 * n)^2 = 2019) : 3 * m + n = 46 :=
sorry

end NUMINAMATH_GPT_find_3m_plus_n_l1355_135559


namespace NUMINAMATH_GPT_smallest_next_divisor_221_l1355_135569

structure Conditions (m : ℕ) :=
  (m_even : m % 2 = 0)
  (m_4digit : 1000 ≤ m ∧ m < 10000)
  (m_div_221 : 221 ∣ m)

theorem smallest_next_divisor_221 (m : ℕ) (h : Conditions m) : ∃ k, k > 221 ∧ k ∣ m ∧ k = 289 := by
  sorry

end NUMINAMATH_GPT_smallest_next_divisor_221_l1355_135569


namespace NUMINAMATH_GPT_solve_trig_eq_l1355_135571

theorem solve_trig_eq (k : ℤ) : 
  ∃ x : ℝ, 12 * Real.sin x - 5 * Real.cos x = 13 ∧ 
           x = Real.pi / 2 + Real.arctan (5 / 12) + 2 * ↑k * Real.pi :=
by
  sorry

end NUMINAMATH_GPT_solve_trig_eq_l1355_135571


namespace NUMINAMATH_GPT_percentage_decrease_l1355_135548

theorem percentage_decrease (original_salary new_salary decreased_salary : ℝ) (p : ℝ) (D : ℝ) : 
  original_salary = 4000.0000000000005 →
  p = 10 →
  new_salary = original_salary * (1 + p/100) →
  decreased_salary = 4180 →
  decreased_salary = new_salary * (1 - D / 100) →
  D = 5 :=
by
  intros h1 h2 h3 h4 h5
  sorry

end NUMINAMATH_GPT_percentage_decrease_l1355_135548


namespace NUMINAMATH_GPT_school_students_l1355_135574

theorem school_students (T S : ℕ) (h1 : T = 6 * S - 78) (h2 : T - S = 2222) : T = 2682 :=
by
  sorry

end NUMINAMATH_GPT_school_students_l1355_135574


namespace NUMINAMATH_GPT_average_value_of_items_in_loot_box_l1355_135528

-- Definitions as per the given conditions
def cost_per_loot_box : ℝ := 5
def total_spent : ℝ := 40
def total_loss : ℝ := 12

-- Proving the average value of items inside each loot box
theorem average_value_of_items_in_loot_box :
  (total_spent - total_loss) / (total_spent / cost_per_loot_box) = 3.50 := by
  sorry

end NUMINAMATH_GPT_average_value_of_items_in_loot_box_l1355_135528


namespace NUMINAMATH_GPT_other_root_of_quadratic_l1355_135512

theorem other_root_of_quadratic (k : ℝ) (h : -2 * 1 = -2) (h_eq : x^2 + k * x - 2 = 0) :
  1 * -2 = -2 :=
by
  sorry

end NUMINAMATH_GPT_other_root_of_quadratic_l1355_135512


namespace NUMINAMATH_GPT_gratuity_percentage_l1355_135509

open Real

theorem gratuity_percentage (num_bankers num_clients : ℕ) (total_bill per_person_cost : ℝ) 
    (h1 : num_bankers = 4) (h2 : num_clients = 5) (h3 : total_bill = 756) 
    (h4 : per_person_cost = 70) : 
    ((total_bill - (num_bankers + num_clients) * per_person_cost) / 
     ((num_bankers + num_clients) * per_person_cost)) = 0.2 :=
by 
  sorry

end NUMINAMATH_GPT_gratuity_percentage_l1355_135509


namespace NUMINAMATH_GPT_positive_solution_of_x_l1355_135504

theorem positive_solution_of_x :
  ∃ x y z : ℝ, (x * y = 6 - 2 * x - 3 * y) ∧ (y * z = 6 - 4 * y - 2 * z) ∧ (x * z = 30 - 4 * x - 3 * z) ∧ x > 0 ∧ x = 3 :=
by
  sorry

end NUMINAMATH_GPT_positive_solution_of_x_l1355_135504


namespace NUMINAMATH_GPT_avg_words_per_hour_l1355_135541

theorem avg_words_per_hour (words hours : ℝ) (h_words : words = 40000) (h_hours : hours = 80) :
  words / hours = 500 :=
by
  rw [h_words, h_hours]
  norm_num
  done

end NUMINAMATH_GPT_avg_words_per_hour_l1355_135541


namespace NUMINAMATH_GPT_triangle_angle_B_eq_60_l1355_135506

theorem triangle_angle_B_eq_60 {A B C : ℝ} (h1 : B = 2 * A) (h2 : C = 3 * A) (h3 : A + B + C = 180) : B = 60 :=
by sorry

end NUMINAMATH_GPT_triangle_angle_B_eq_60_l1355_135506


namespace NUMINAMATH_GPT_solve_triplet_l1355_135538

theorem solve_triplet (x y z : ℕ) (h : 2^x * 3^y + 1 = 7^z) :
  (x = 1 ∧ y = 1 ∧ z = 1) ∨ (x = 4 ∧ y = 1 ∧ z = 2) :=
 by sorry

end NUMINAMATH_GPT_solve_triplet_l1355_135538


namespace NUMINAMATH_GPT_find_oysters_first_day_l1355_135565

variable (O : ℕ)  -- Number of oysters on the rocks on the first day

def count_crabs_first_day := 72  -- Number of crabs on the beach on the first day

def oysters_second_day := O / 2  -- Number of oysters on the rocks on the second day

def crabs_second_day := (2 / 3) * count_crabs_first_day  -- Number of crabs on the beach on the second day

def total_count := 195  -- Total number of oysters and crabs counted over the two days

theorem find_oysters_first_day (h:  O + oysters_second_day O + count_crabs_first_day + crabs_second_day = total_count) : 
  O = 50 := by
  sorry

end NUMINAMATH_GPT_find_oysters_first_day_l1355_135565


namespace NUMINAMATH_GPT_evaluate_expression_l1355_135590

theorem evaluate_expression :
  let x := (1 : ℚ) / 2
  let y := (3 : ℚ) / 4
  let z := -6
  let w := 2
  (x^2 * y^4 * z * w = - (243 / 256)) := 
by {
  let x := (1 : ℚ) / 2
  let y := (3 : ℚ) / 4
  let z := -6
  let w := 2
  sorry
}

end NUMINAMATH_GPT_evaluate_expression_l1355_135590


namespace NUMINAMATH_GPT_harrison_annual_croissant_expenditure_l1355_135507

-- Define the different costs and frequency of croissants.
def cost_regular_croissant : ℝ := 3.50
def cost_almond_croissant : ℝ := 5.50
def cost_chocolate_croissant : ℝ := 4.50
def cost_ham_cheese_croissant : ℝ := 6.00

def frequency_regular_croissant : ℕ := 52
def frequency_almond_croissant : ℕ := 52
def frequency_chocolate_croissant : ℕ := 52
def frequency_ham_cheese_croissant : ℕ := 26

-- Calculate annual expenditure for each type of croissant.
def annual_expenditure (cost : ℝ) (frequency : ℕ) : ℝ :=
  cost * frequency

-- Total annual expenditure on croissants.
def total_annual_expenditure : ℝ :=
  annual_expenditure cost_regular_croissant frequency_regular_croissant +
  annual_expenditure cost_almond_croissant frequency_almond_croissant +
  annual_expenditure cost_chocolate_croissant frequency_chocolate_croissant +
  annual_expenditure cost_ham_cheese_croissant frequency_ham_cheese_croissant

-- The theorem to prove.
theorem harrison_annual_croissant_expenditure :
  total_annual_expenditure = 858 := by
  sorry

end NUMINAMATH_GPT_harrison_annual_croissant_expenditure_l1355_135507


namespace NUMINAMATH_GPT_relationship_between_b_and_g_l1355_135594

-- Definitions based on the conditions
def n_th_boy_dances (n : ℕ) : ℕ := n + 5
def last_boy_dances_with_all : Prop := ∃ b g : ℕ, (n_th_boy_dances b = g)

-- The main theorem to prove the relationship between b and g
theorem relationship_between_b_and_g (b g : ℕ) (h : last_boy_dances_with_all) : b = g - 5 :=
by
  sorry

end NUMINAMATH_GPT_relationship_between_b_and_g_l1355_135594


namespace NUMINAMATH_GPT_trail_length_l1355_135580

theorem trail_length (v_Q : ℝ) (v_P : ℝ) (d_P d_Q : ℝ) 
  (h_vP: v_P = 1.25 * v_Q) 
  (h_dP: d_P = 20) 
  (h_meet: d_P / v_P = d_Q / v_Q) :
  d_P + d_Q = 36 :=
sorry

end NUMINAMATH_GPT_trail_length_l1355_135580


namespace NUMINAMATH_GPT_milk_price_increase_l1355_135560

theorem milk_price_increase
  (P : ℝ) (C : ℝ) (P_new : ℝ)
  (h1 : P * C = P_new * (5 / 6) * C) :
  (P_new - P) / P * 100 = 20 :=
by
  sorry

end NUMINAMATH_GPT_milk_price_increase_l1355_135560


namespace NUMINAMATH_GPT_range_of_a_l1355_135556

theorem range_of_a (a : ℝ) : (∃ x : ℝ, |x - 4| + |x - 3| < a) ↔ a > 1 := 
sorry

end NUMINAMATH_GPT_range_of_a_l1355_135556


namespace NUMINAMATH_GPT_decimal_addition_l1355_135577

theorem decimal_addition : 0.4 + 0.02 + 0.006 = 0.426 := by
  sorry

end NUMINAMATH_GPT_decimal_addition_l1355_135577


namespace NUMINAMATH_GPT_initial_time_for_train_l1355_135540

theorem initial_time_for_train (S : ℝ)
  (length_initial : ℝ := 12 * 15)
  (length_detached : ℝ := 11 * 15)
  (time_detached : ℝ := 16.5)
  (speed_constant : S = length_detached / time_detached) :
  (length_initial / S = 18) :=
by
  sorry

end NUMINAMATH_GPT_initial_time_for_train_l1355_135540


namespace NUMINAMATH_GPT_car_clock_problem_l1355_135568

-- Define the conditions and statements required for the proof
variable (t₀ : ℕ) -- Initial time in minutes corresponding to 2:00 PM
variable (t₁ : ℕ) -- Time in minutes when the accurate watch shows 2:40 PM
variable (t₂ : ℕ) -- Time in minutes when the car clock shows 2:50 PM
variable (t₃ : ℕ) -- Time in minutes when the car clock shows 8:00 PM
variable (rate : ℚ) -- Rate of the car clock relative to real time

-- Define the initial condition
def initial_time := (t₀ = 0)

-- Define the time gain from 2:00 PM to 2:40 PM on the accurate watch
def accurate_watch_time := (t₁ = 40)

-- Define the time gain for car clock from 2:00 PM to 2:50 PM
def car_clock_time := (t₂ = 50)

-- Define the rate of the car clock relative to real time as 5/4
def car_clock_rate := (rate = 5/4)

-- Define the car clock reading at 8:00 PM
def car_clock_later := (t₃ = 8 * 60)

-- Define the actual time corresponding to the car clock reading 8:00 PM
def actual_time : ℚ := (t₀ + (t₃ - t₀) * (4/5))

-- Define the statement theorem using the defined conditions and variables
theorem car_clock_problem 
  (h₀ : initial_time t₀) 
  (h₁ : accurate_watch_time t₁) 
  (h₂ : car_clock_time t₂) 
  (h₃ : car_clock_rate rate) 
  (h₄ : car_clock_later t₃) 
  : actual_time t₀ t₃ = 8 * 60 + 24 :=
by sorry

end NUMINAMATH_GPT_car_clock_problem_l1355_135568


namespace NUMINAMATH_GPT_sum_of_integers_l1355_135570

theorem sum_of_integers :
  ∃ a b : ℕ, 0 < a ∧ 0 < b ∧ a < 30 ∧ b < 30 ∧ (a * b + a + b = 167) ∧ Nat.gcd a b = 1 ∧ (a + b = 24) :=
by {
  sorry
}

end NUMINAMATH_GPT_sum_of_integers_l1355_135570


namespace NUMINAMATH_GPT_largest_constant_l1355_135529

theorem largest_constant (x y z : ℝ) : (x^2 + y^2 + z^2 + 3 ≥ 2 * (x + y + z)) :=
by
  sorry

end NUMINAMATH_GPT_largest_constant_l1355_135529


namespace NUMINAMATH_GPT_intersection_of_M_and_N_l1355_135579

theorem intersection_of_M_and_N :
  let M := { x : ℝ | -6 ≤ x ∧ x < 4 }
  let N := { x : ℝ | -2 < x ∧ x ≤ 8 }
  M ∩ N = { x | -2 < x ∧ x < 4 } :=
by
  sorry -- Proof is omitted

end NUMINAMATH_GPT_intersection_of_M_and_N_l1355_135579


namespace NUMINAMATH_GPT_solve_for_y_l1355_135539

theorem solve_for_y (y : ℝ) (h : y + 81 / (y - 3) = -12) : y = -6 ∨ y = -3 :=
sorry

end NUMINAMATH_GPT_solve_for_y_l1355_135539


namespace NUMINAMATH_GPT_eccentricity_of_ellipse_l1355_135572

variables {E F1 F2 P Q : Type}
variables (a c : ℝ) 

-- Define the foci and intersection conditions
def is_right_foci (F1 F2 : Type) (E : Type) : Prop := sorry
def line_intersects_ellipse (E : Type) (P Q : Type) (slope : ℝ) : Prop := sorry
def is_right_triangle (P F2 : Type) : Prop := sorry

-- Prove the eccentricity condition
theorem eccentricity_of_ellipse
  (h_foci : is_right_foci F1 F2 E)
  (h_line : line_intersects_ellipse E P Q (4 / 3))
  (h_triangle : is_right_triangle P F2) :
  (c / a) = (5 / 7) :=
sorry

end NUMINAMATH_GPT_eccentricity_of_ellipse_l1355_135572


namespace NUMINAMATH_GPT_initial_birds_l1355_135576

theorem initial_birds (B : ℕ) (h1 : B + 21 = 35) : B = 14 :=
by
  sorry

end NUMINAMATH_GPT_initial_birds_l1355_135576


namespace NUMINAMATH_GPT_polynomial_integer_roots_k_zero_l1355_135542

theorem polynomial_integer_roots_k_zero :
  (∃ a b c : ℤ, a ≠ b ∧ b ≠ c ∧ a ≠ c ∧
  (∀ x : ℤ, (x - a) * (x - b) * (x - c) = x^3 - x + 0) ∨
  (∀ x : ℤ, (x - a) * (x - b) * (x - c) = x^3 - x + k)) →
  k = 0 :=
sorry

end NUMINAMATH_GPT_polynomial_integer_roots_k_zero_l1355_135542


namespace NUMINAMATH_GPT_hairstylist_weekly_earnings_l1355_135584

-- Definition of conditions as given in part a)
def cost_normal_haircut := 5
def cost_special_haircut := 6
def cost_trendy_haircut := 8

def number_normal_haircuts_per_day := 5
def number_special_haircuts_per_day := 3
def number_trendy_haircuts_per_day := 2

def working_days_per_week := 7

-- The goal is to prove that the hairstylist's weekly earnings equal to 413 dollars
theorem hairstylist_weekly_earnings : 
  (number_normal_haircuts_per_day * cost_normal_haircut +
  number_special_haircuts_per_day * cost_special_haircut +
  number_trendy_haircuts_per_day * cost_trendy_haircut) * 
  working_days_per_week = 413 := 
by sorry -- We use "by sorry" to skip the proof

end NUMINAMATH_GPT_hairstylist_weekly_earnings_l1355_135584


namespace NUMINAMATH_GPT_maximum_value_cosine_sine_combination_l1355_135586

noncomputable def max_cosine_sine_combination : Real :=
  let g (θ : Real) := (Real.cos (θ / 2)) * (1 + Real.sin θ)
  have h₁ : ∃ θ : Real, -Real.pi / 2 < θ ∧ θ < Real.pi / 2 :=
    sorry -- Existence of such θ is trivial
  Real.sqrt 2

theorem maximum_value_cosine_sine_combination :
  ∀ θ : Real, -Real.pi / 2 < θ ∧ θ < Real.pi / 2 →
  (Real.cos (θ / 2)) * (1 + Real.sin θ) ≤ Real.sqrt 2 :=
by
  intros θ h
  let y := (Real.cos (θ / 2)) * (1 + Real.sin θ)
  have hy : y ≤ Real.sqrt 2 := sorry
  exact hy

end NUMINAMATH_GPT_maximum_value_cosine_sine_combination_l1355_135586


namespace NUMINAMATH_GPT_trumpet_cost_l1355_135550

/-
  Conditions:
  1. Cost of the music tool: $9.98
  2. Cost of the song book: $4.14
  3. Total amount Joan spent at the music store: $163.28

  Prove that the cost of the trumpet is $149.16
-/

theorem trumpet_cost :
  let c_mt := 9.98
  let c_sb := 4.14
  let t_sp := 163.28
  let c_trumpet := t_sp - (c_mt + c_sb)
  c_trumpet = 149.16 :=
by
  sorry

end NUMINAMATH_GPT_trumpet_cost_l1355_135550


namespace NUMINAMATH_GPT_greatest_root_of_f_one_is_root_of_f_l1355_135557

def f (x : ℝ) : ℝ := 16 * x^6 - 15 * x^4 + 4 * x^2 - 1

theorem greatest_root_of_f :
  ∀ x : ℝ, f x = 0 → x ≤ 1 :=
sorry

theorem one_is_root_of_f :
  f 1 = 0 :=
sorry

end NUMINAMATH_GPT_greatest_root_of_f_one_is_root_of_f_l1355_135557


namespace NUMINAMATH_GPT_marked_price_percentage_l1355_135585

variables (L M: ℝ)

-- The store owner purchases items at a 25% discount of the list price.
def cost_price (L : ℝ) := 0.75 * L

-- The store owner plans to mark them up such that after a 10% discount on the marked price,
-- he achieves a 25% profit on the selling price.
def selling_price (M : ℝ) := 0.9 * M

-- Given condition: cost price is 75% of selling price
theorem marked_price_percentage (h : cost_price L = 0.75 * selling_price M) : 
  M = 1.111 * L :=
by 
  sorry

end NUMINAMATH_GPT_marked_price_percentage_l1355_135585


namespace NUMINAMATH_GPT_future_value_proof_l1355_135589

noncomputable def present_value : ℝ := 1093.75
noncomputable def interest_rate : ℝ := 0.04
noncomputable def years : ℕ := 2

def future_value (PV : ℝ) (r : ℝ) (n : ℕ) : ℝ :=
  PV * (1 + r) ^ n

theorem future_value_proof :
  future_value present_value interest_rate years = 1183.06 :=
by
  -- Calculation details skipped here, assuming the required proof steps are completed.
  sorry

end NUMINAMATH_GPT_future_value_proof_l1355_135589


namespace NUMINAMATH_GPT_solve_equation_l1355_135583

theorem solve_equation (y : ℝ) : 
  5 * (y + 2) + 9 = 3 * (1 - y) ↔ y = -2 := 
by 
  sorry

end NUMINAMATH_GPT_solve_equation_l1355_135583


namespace NUMINAMATH_GPT_abs_inequality_solution_l1355_135581

theorem abs_inequality_solution (x : ℝ) : |x - 3| ≥ |x| ↔ x ≤ 3 / 2 :=
by
  sorry

end NUMINAMATH_GPT_abs_inequality_solution_l1355_135581


namespace NUMINAMATH_GPT_shaded_region_area_l1355_135530

theorem shaded_region_area (r : ℝ) (n : ℕ) (shaded_area : ℝ) (h_r : r = 3) (h_n : n = 6) :
  shaded_area = 27 * Real.pi - 54 := by
  sorry

end NUMINAMATH_GPT_shaded_region_area_l1355_135530


namespace NUMINAMATH_GPT_second_hand_travel_distance_l1355_135527

theorem second_hand_travel_distance (radius : ℝ) (time_minutes : ℕ) (C : ℝ) (distance : ℝ) 
    (h1 : radius = 8) (h2 : time_minutes = 45) 
    (h3 : C = 2 * Real.pi * radius) 
    (h4 : distance = time_minutes * C)
    : distance = 720 * Real.pi := 
by 
  rw [h1, h2, h3] at *
  sorry

end NUMINAMATH_GPT_second_hand_travel_distance_l1355_135527


namespace NUMINAMATH_GPT_sin_double_angle_tan_double_angle_l1355_135500

-- Step 1: Define the first problem in Lean 4.
theorem sin_double_angle (α : ℝ) (h1 : Real.sin α = 12 / 13) (h2 : α ∈ Set.Ioo (Real.pi / 2) Real.pi) :
  Real.sin (2 * α) = -120 / 169 := 
sorry

-- Step 2: Define the second problem in Lean 4.
theorem tan_double_angle (α : ℝ) (h1 : Real.tan α = 1 / 2) :
  Real.tan (2 * α) = 4 / 3 := 
sorry

end NUMINAMATH_GPT_sin_double_angle_tan_double_angle_l1355_135500


namespace NUMINAMATH_GPT_triangle_shading_probability_l1355_135518

theorem triangle_shading_probability (n_triangles: ℕ) (n_shaded: ℕ) (h1: n_triangles > 4) (h2: n_shaded = 4) (h3: n_triangles = 10) :
  (n_shaded / n_triangles) = 2 / 5 := 
by
  sorry

end NUMINAMATH_GPT_triangle_shading_probability_l1355_135518


namespace NUMINAMATH_GPT_integer_solutions_l1355_135587

theorem integer_solutions (x y k : ℤ) :
  21 * x + 48 * y = 6 ↔ ∃ k : ℤ, x = -2 + 16 * k ∧ y = 1 - 7 * k :=
by
  sorry

end NUMINAMATH_GPT_integer_solutions_l1355_135587


namespace NUMINAMATH_GPT_calculate_value_l1355_135534

def f (x : ℝ) : ℝ := 9 - x
def g (x : ℝ) : ℝ := x - 9

theorem calculate_value : g (f 15) = -15 := by
  sorry

end NUMINAMATH_GPT_calculate_value_l1355_135534


namespace NUMINAMATH_GPT_min_value_a_plus_2b_l1355_135514

theorem min_value_a_plus_2b (a b : ℝ) (h₁ : a > 0) (h₂ : b > 0) (h₃ : a + 2 * b + 2 * a * b = 8) :
  a + 2 * b ≥ 4 :=
sorry

end NUMINAMATH_GPT_min_value_a_plus_2b_l1355_135514


namespace NUMINAMATH_GPT_total_letters_received_l1355_135502

theorem total_letters_received 
  (Brother_received Greta_received Mother_received : ℕ) 
  (h1 : Greta_received = Brother_received + 10)
  (h2 : Brother_received = 40)
  (h3 : Mother_received = 2 * (Greta_received + Brother_received)) :
  Brother_received + Greta_received + Mother_received = 270 := 
sorry

end NUMINAMATH_GPT_total_letters_received_l1355_135502


namespace NUMINAMATH_GPT_completing_the_square_l1355_135508

theorem completing_the_square (x : ℝ) : x^2 + 8 * x + 9 = 0 → (x + 4)^2 = 7 :=
by 
  intro h
  sorry

end NUMINAMATH_GPT_completing_the_square_l1355_135508


namespace NUMINAMATH_GPT_domain_of_f_i_l1355_135505

variable (f : ℝ → ℝ)

theorem domain_of_f_i (h : ∀ x, -1 ≤ x + 1 ∧ x + 1 ≤ 1) : ∀ x, -2 ≤ x ∧ x ≤ 0 :=
by
  intro x
  specialize h x
  sorry

end NUMINAMATH_GPT_domain_of_f_i_l1355_135505


namespace NUMINAMATH_GPT_total_spent_l1355_135596

def original_cost_vacuum_cleaner : ℝ := 250
def discount_vacuum_cleaner : ℝ := 0.20
def cost_dishwasher : ℝ := 450
def special_offer_discount : ℝ := 75

theorem total_spent :
  let discounted_vacuum_cleaner := original_cost_vacuum_cleaner * (1 - discount_vacuum_cleaner)
  let total_before_special := discounted_vacuum_cleaner + cost_dishwasher
  total_before_special - special_offer_discount = 575 := by
  sorry

end NUMINAMATH_GPT_total_spent_l1355_135596


namespace NUMINAMATH_GPT_factor_example_solve_equation_example_l1355_135544

-- Factorization proof problem
theorem factor_example (m a b : ℝ) : 
  (m * a ^ 2 - 4 * m * b ^ 2) = m * (a + 2 * b) * (a - 2 * b) :=
sorry

-- Solving the equation proof problem
theorem solve_equation_example (x : ℝ) (hx1: x ≠ 2) (hx2: x ≠ 0) : 
  (1 / (x - 2) = 3 / x) ↔ x = 3 :=
sorry

end NUMINAMATH_GPT_factor_example_solve_equation_example_l1355_135544


namespace NUMINAMATH_GPT_everyone_can_cross_l1355_135517

-- Define each agent
inductive Agent
| C   -- Princess Sonya
| K (i : Fin 8) -- Knights numbered 1 to 7

open Agent

-- Define friendships
def friends (a b : Agent) : Prop :=
  match a, b with
  | C, (K 4) => False
  | (K 4), C => False
  | _, _ => (∃ i : Fin 8, a = K i ∧ b = K (i+1)) ∨ (∃ i : Fin 7, a = K (i+1) ∧ b = K i) ∨ a = C ∨ b = C

-- Define the crossing conditions
def boatCanCarry : List Agent → Prop
| [a, b] => friends a b
| [a, b, c] => friends a b ∧ friends b c ∧ friends a c
| _ => False

-- The main statement to prove
theorem everyone_can_cross (agents : List Agent) (steps : List (List Agent)) :
  agents = [C, K 0, K 1, K 2, K 3, K 4, K 5, K 6, K 7] →
  (∀ step ∈ steps, boatCanCarry step) →
  (∃ final_state : List (List Agent), final_state = [[C, K 0, K 1, K 2, K 3, K 4, K 5, K 6, K 7]]) :=
by 
  -- The proof is omitted.
  sorry

end NUMINAMATH_GPT_everyone_can_cross_l1355_135517


namespace NUMINAMATH_GPT_tetrahedron_face_inequality_l1355_135573

theorem tetrahedron_face_inequality
    (A B C D : ℝ) :
    |A^2 + B^2 - C^2 - D^2| ≤ 2 * (A * B + C * D) := by
  sorry

end NUMINAMATH_GPT_tetrahedron_face_inequality_l1355_135573


namespace NUMINAMATH_GPT_solve_for_x_l1355_135599

theorem solve_for_x :
  ∀ (x : ℚ), x = 45 / (8 - 3 / 7) → x = 315 / 53 :=
by
  sorry

end NUMINAMATH_GPT_solve_for_x_l1355_135599


namespace NUMINAMATH_GPT_total_surface_area_hemisphere_l1355_135553

theorem total_surface_area_hemisphere (A : ℝ) (r : ℝ) : (A = 100 * π) → (r = 10) → (2 * π * r^2 + A = 300 * π) :=
by
  intro hA hr
  sorry

end NUMINAMATH_GPT_total_surface_area_hemisphere_l1355_135553


namespace NUMINAMATH_GPT_baker_number_of_eggs_l1355_135536

theorem baker_number_of_eggs (flour cups eggs : ℕ) (h1 : eggs = 3 * (flour / 2)) (h2 : flour = 6) : eggs = 9 :=
by
  sorry

end NUMINAMATH_GPT_baker_number_of_eggs_l1355_135536


namespace NUMINAMATH_GPT_positive_real_inequality_l1355_135597

noncomputable def positive_real_sum_condition (u v w : ℝ) [OrderedRing ℝ] :=
  u + v + w + Real.sqrt (u * v * w) = 4

theorem positive_real_inequality (u v w : ℝ) (hu : 0 < u) (hv : 0 < v) (hw : 0 < w) :
  positive_real_sum_condition u v w →
  Real.sqrt (v * w / u) + Real.sqrt (u * w / v) + Real.sqrt (u * v / w) ≥ u + v + w :=
by
  sorry

end NUMINAMATH_GPT_positive_real_inequality_l1355_135597


namespace NUMINAMATH_GPT_desired_interest_rate_l1355_135575

def nominalValue : ℝ := 20
def dividendRate : ℝ := 0.09
def marketValue : ℝ := 15

theorem desired_interest_rate : (dividendRate * nominalValue / marketValue) * 100 = 12 := by
  sorry

end NUMINAMATH_GPT_desired_interest_rate_l1355_135575


namespace NUMINAMATH_GPT_infinitely_many_87_b_seq_l1355_135564

def a_seq : ℕ → ℕ
| 0 => 3
| (n + 1) => 3 ^ (a_seq n)

def b_seq (n : ℕ) : ℕ := (a_seq n) % 100

theorem infinitely_many_87_b_seq (n : ℕ) (hn : n ≥ 2) : b_seq n = 87 := by
  sorry

end NUMINAMATH_GPT_infinitely_many_87_b_seq_l1355_135564


namespace NUMINAMATH_GPT_parabola_chord_midpoint_l1355_135511

/-- 
If the point (3, 1) is the midpoint of a chord of the parabola y^2 = 2px, 
and the slope of the line containing this chord is 2, then p = 2. 
-/
theorem parabola_chord_midpoint (p : ℝ) :
    (∃ (m : ℝ), (m = 2) ∧ ∀ (x y : ℝ), y = 2 * x - 5 → y^2 = 2 * p * x → 
        ((x1 = 0 ∧ y1 = 0 ∧ x2 = 6 ∧ y2 = 6) → 
            (x1 + x2 = 6) ∧ (y1 + y2 = 2) ∧ (p = 2))) :=
sorry

end NUMINAMATH_GPT_parabola_chord_midpoint_l1355_135511


namespace NUMINAMATH_GPT_final_fraction_of_water_is_243_over_1024_l1355_135591

theorem final_fraction_of_water_is_243_over_1024 :
  let initial_volume := 20
  let replaced_volume := 5
  let cycles := 5
  let initial_fraction_of_water := 1
  let final_fraction_of_water :=
        (initial_fraction_of_water * (initial_volume - replaced_volume) / initial_volume) ^ cycles
  final_fraction_of_water = 243 / 1024 :=
by
  sorry

end NUMINAMATH_GPT_final_fraction_of_water_is_243_over_1024_l1355_135591


namespace NUMINAMATH_GPT_ratio_humans_to_beavers_l1355_135561

-- Define the conditions
def humans : ℕ := 38 * 10^6
def moose : ℕ := 1 * 10^6
def beavers : ℕ := 2 * moose

-- Define the theorem to prove the ratio of humans to beavers
theorem ratio_humans_to_beavers : humans / beavers = 19 := by
  sorry

end NUMINAMATH_GPT_ratio_humans_to_beavers_l1355_135561


namespace NUMINAMATH_GPT_cylinder_lateral_surface_area_l1355_135595

-- Define structures for the problem
structure Cylinder where
  generatrix : ℝ
  base_radius : ℝ

-- Define the conditions
def cylinder_conditions : Cylinder :=
  { generatrix := 1, base_radius := 1 }

-- The theorem statement
theorem cylinder_lateral_surface_area (cyl : Cylinder) (h_gen : cyl.generatrix = 1) (h_rad : cyl.base_radius = 1) :
  ∀ (area : ℝ), area = 2 * Real.pi :=
sorry

end NUMINAMATH_GPT_cylinder_lateral_surface_area_l1355_135595


namespace NUMINAMATH_GPT_route_Y_is_quicker_l1355_135551

noncomputable def route_time (distance : ℝ) (speed : ℝ) : ℝ :=
  distance / speed

def route_X_distance : ℝ := 8
def route_X_speed : ℝ := 40

def route_Y_total_distance : ℝ := 7
def route_Y_construction_distance : ℝ := 1
def route_Y_construction_speed : ℝ := 20
def route_Y_regular_speed_distance : ℝ := 6
def route_Y_regular_speed : ℝ := 50

noncomputable def route_X_time : ℝ :=
  route_time route_X_distance route_X_speed * 60  -- converting to minutes

noncomputable def route_Y_time : ℝ :=
  (route_time route_Y_regular_speed_distance route_Y_regular_speed +
  route_time route_Y_construction_distance route_Y_construction_speed) * 60 -- converting to minutes

theorem route_Y_is_quicker : route_X_time - route_Y_time = 1.8 :=
  by
    sorry

end NUMINAMATH_GPT_route_Y_is_quicker_l1355_135551


namespace NUMINAMATH_GPT_max_x_minus_2y_l1355_135555

open Real

theorem max_x_minus_2y (x y : ℝ) (h : (x^2) / 16 + (y^2) / 9 = 1) : 
  ∃ t : ℝ, t = 2 * sqrt 13 ∧ x - 2 * y = t := 
sorry

end NUMINAMATH_GPT_max_x_minus_2y_l1355_135555


namespace NUMINAMATH_GPT_probability_of_selecting_GEARS_letter_l1355_135531

def bag : List Char := ['A', 'L', 'G', 'E', 'B', 'R', 'A', 'S']
def target_word : List Char := ['G', 'E', 'A', 'R', 'S']

theorem probability_of_selecting_GEARS_letter :
  (6 : ℚ) / 8 = 3 / 4 :=
by
  sorry

end NUMINAMATH_GPT_probability_of_selecting_GEARS_letter_l1355_135531


namespace NUMINAMATH_GPT_range_of_a_l1355_135537

theorem range_of_a (a : ℝ) : (∀ x : ℝ, 1 ≤ x → (x^2 + 2*x + a) / x > 0) ↔ a > -3 :=
by
  sorry

end NUMINAMATH_GPT_range_of_a_l1355_135537


namespace NUMINAMATH_GPT_independence_of_events_l1355_135562

noncomputable def is_independent (A B : Prop) (chi_squared : ℝ) := 
  chi_squared ≤ 3.841

theorem independence_of_events (A B : Prop) (chi_squared : ℝ) : 
  is_independent A B chi_squared → A ↔ B :=
by
  sorry

end NUMINAMATH_GPT_independence_of_events_l1355_135562


namespace NUMINAMATH_GPT_range_of_f_l1355_135543

noncomputable def f (x : ℝ) : ℝ := 3^(x - 2)

theorem range_of_f : Set.Icc 1 9 = {y : ℝ | ∃ x : ℝ, 2 ≤ x ∧ x ≤ 4 ∧ f x = y} :=
by
  sorry

end NUMINAMATH_GPT_range_of_f_l1355_135543


namespace NUMINAMATH_GPT_cube_root_floor_equality_l1355_135549

theorem cube_root_floor_equality (n : ℕ) : 
  (⌊(n : ℝ)^(1/3) + (n+1 : ℝ)^(1/3)⌋ : ℝ) = ⌊(8*n + 3 : ℝ)^(1/3)⌋ :=
sorry

end NUMINAMATH_GPT_cube_root_floor_equality_l1355_135549


namespace NUMINAMATH_GPT_value_of_k_l1355_135546

theorem value_of_k (m n k : ℝ) (h1 : 3 ^ m = k) (h2 : 5 ^ n = k) (h3 : 1 / m + 1 / n = 2) : k = Real.sqrt 15 :=
  sorry

end NUMINAMATH_GPT_value_of_k_l1355_135546


namespace NUMINAMATH_GPT_find_num_large_envelopes_l1355_135532

def numLettersInSmallEnvelopes : Nat := 20
def totalLetters : Nat := 150
def totalLettersInMediumLargeEnvelopes := totalLetters - numLettersInSmallEnvelopes -- 130
def lettersPerLargeEnvelope : Nat := 5
def lettersPerMediumEnvelope : Nat := 3
def numLargeEnvelopes (L : Nat) : Prop := 5 * L + 6 * L = totalLettersInMediumLargeEnvelopes

theorem find_num_large_envelopes : ∃ L : Nat, numLargeEnvelopes L ∧ L = 11 := by
  sorry

end NUMINAMATH_GPT_find_num_large_envelopes_l1355_135532


namespace NUMINAMATH_GPT_prime_solution_l1355_135510

theorem prime_solution (p : ℕ) (hp : Nat.Prime p) :
  (∃ x y : ℕ, 0 < x ∧ 0 < y ∧ x * (y^2 - p) + y * (x^2 - p) = 5 * p) ↔ (p = 2 ∨ p = 3 ∨ p = 7) :=
by
  sorry

end NUMINAMATH_GPT_prime_solution_l1355_135510


namespace NUMINAMATH_GPT_total_amount_spent_l1355_135554

theorem total_amount_spent (tax_paid : ℝ) (tax_rate : ℝ) (tax_free_cost : ℝ) (total_spent : ℝ) :
  tax_paid = 30 → tax_rate = 0.06 → tax_free_cost = 19.7 →
  total_spent = 30 / 0.06 + 19.7 :=
by
  -- Definitions for assumptions
  intro h1 h2 h3
  -- Skip the proof here
  sorry

end NUMINAMATH_GPT_total_amount_spent_l1355_135554


namespace NUMINAMATH_GPT_find_y_l1355_135582

theorem find_y :
  ∃ y : ℝ, ((0.47 * 1442) - (0.36 * y) + 65 = 5) ∧ y = 2049.28 :=
by
  sorry

end NUMINAMATH_GPT_find_y_l1355_135582


namespace NUMINAMATH_GPT_math_problem_modulo_l1355_135513

theorem math_problem_modulo :
    (245 * 15 - 20 * 8 + 5) % 17 = 1 := 
by
  sorry

end NUMINAMATH_GPT_math_problem_modulo_l1355_135513


namespace NUMINAMATH_GPT_complex_division_l1355_135501

theorem complex_division (i : ℂ) (h : i * i = -1) : 3 / (1 - i) ^ 2 = (3 / 2) * i :=
by
  sorry

end NUMINAMATH_GPT_complex_division_l1355_135501


namespace NUMINAMATH_GPT_general_term_arithmetic_sequence_l1355_135503

variable {α : Type*}
variables (a_n a : ℕ → ℕ) (d a_1 a_2 a_3 a_4 n : ℕ)

-- Define the arithmetic sequence condition
def arithmetic_sequence (a_n : ℕ → ℕ) (d : ℕ) :=
  ∀ n, a_n (n + 1) = a_n n + d

-- Define the inequality solution condition 
def inequality_solution_set (a_1 a_2 : ℕ) (x : ℕ) :=
  a_1 ≤ x ∧ x ≤ a_2

theorem general_term_arithmetic_sequence :
  arithmetic_sequence a_n d ∧ (d ≠ 0) ∧ 
  (∀ x, x^2 - a_3 * x + a_4 ≤ 0 ↔ inequality_solution_set a_1 a_2 x) →
  a_n = 2 * n :=
by
  sorry

end NUMINAMATH_GPT_general_term_arithmetic_sequence_l1355_135503


namespace NUMINAMATH_GPT_AndrewAge_l1355_135524

variable (a f g : ℚ)
axiom h1 : f = 8 * a
axiom h2 : g = 3 * f
axiom h3 : g - a = 72

theorem AndrewAge : a = 72 / 23 :=
by
  sorry

end NUMINAMATH_GPT_AndrewAge_l1355_135524


namespace NUMINAMATH_GPT_slope_range_PA2_l1355_135535

-- Define the given conditions
def ellipse (x y : ℝ) : Prop := (x^2 / 4) + (y^2 / 3) = 1

def A1 : ℝ × ℝ := (-2, 0)
def A2 : ℝ × ℝ := (2, 0)
def on_ellipse (P : ℝ × ℝ) : Prop := ellipse P.fst P.snd

-- Define the range of the slope of line PA1
def slope_range_PA1 (k_PA1 : ℝ) : Prop := -2 ≤ k_PA1 ∧ k_PA1 ≤ -1

-- Main theorem
theorem slope_range_PA2 (x0 y0 k_PA1 k_PA2 : ℝ) (h1 : on_ellipse (x0, y0)) (h2 : slope_range_PA1 k_PA1) :
  k_PA1 = (y0 / (x0 + 2)) →
  k_PA2 = (y0 / (x0 - 2)) →
  - (3 / 4) = k_PA1 * k_PA2 →
  (3 / 8) ≤ k_PA2 ∧ k_PA2 ≤ (3 / 4) :=
by
  sorry

end NUMINAMATH_GPT_slope_range_PA2_l1355_135535


namespace NUMINAMATH_GPT_shortest_is_Bob_l1355_135520

variable {Person : Type}
variable [LinearOrder Person]

variable (Amy Bob Carla Dan Eric : Person)

-- Conditions
variable (h1 : Amy > Carla)
variable (h2 : Dan < Eric)
variable (h3 : Dan > Bob)
variable (h4 : Eric < Carla)

theorem shortest_is_Bob : ∀ p : Person, p = Bob :=
by
  intro p
  sorry

end NUMINAMATH_GPT_shortest_is_Bob_l1355_135520


namespace NUMINAMATH_GPT_holds_under_condition_l1355_135578

theorem holds_under_condition (a b c : ℕ) (ha : a ≤ 10) (hb : b ≤ 10) (hc : c ≤ 10) (cond : b + 11 * c = 10 * a) :
  (10 * a + b) * (10 * a + c) = 100 * a * a + 100 * a + 11 * b * c :=
by
  sorry

end NUMINAMATH_GPT_holds_under_condition_l1355_135578


namespace NUMINAMATH_GPT_min_value_fraction_sum_l1355_135547

variable (a b : ℝ)
variable (ha : a > 0)
variable (hb : b > 0)
variable (h_collinear : 3 * a + 2 * b = 1)

theorem min_value_fraction_sum (a b : ℝ) (ha : a > 0) (hb : b > 0) (h_collinear : 3 * a + 2 * b = 1) : 
  (3 / a + 1 / b) = 11 + 6 * Real.sqrt 2 :=
by
  sorry

end NUMINAMATH_GPT_min_value_fraction_sum_l1355_135547


namespace NUMINAMATH_GPT_sum_of_repeating_decimals_l1355_135522

/-- Definitions of the repeating decimals as real numbers. --/
def x : ℝ := 0.3 -- This actually represents 0.\overline{3} in Lean
def y : ℝ := 0.04 -- This actually represents 0.\overline{04} in Lean
def z : ℝ := 0.005 -- This actually represents 0.\overline{005} in Lean

/-- The theorem stating that the sum of these repeating decimals is a specific fraction. --/
theorem sum_of_repeating_decimals : x + y + z = (14 : ℝ) / 37 := 
by 
  sorry -- Placeholder for the proof

end NUMINAMATH_GPT_sum_of_repeating_decimals_l1355_135522


namespace NUMINAMATH_GPT_remaining_last_year_budget_is_13_l1355_135588

-- Variables representing the conditions of the problem
variable (cost1 cost2 given_budget remaining this_year_spent remaining_last_year : ℤ)

-- Define the conditions as hypotheses
def conditions : Prop :=
  cost1 = 13 ∧ cost2 = 24 ∧ 
  given_budget = 50 ∧ 
  remaining = 19 ∧ 
  (cost1 + cost2 = 37) ∧
  (this_year_spent = given_budget - remaining) ∧
  (remaining_last_year + (cost1 + cost2 - this_year_spent) = remaining)

-- The statement that needs to be proven
theorem remaining_last_year_budget_is_13 : conditions cost1 cost2 given_budget remaining this_year_spent remaining_last_year → remaining_last_year = 13 :=
by 
  intro h
  sorry

end NUMINAMATH_GPT_remaining_last_year_budget_is_13_l1355_135588


namespace NUMINAMATH_GPT_students_play_basketball_l1355_135592

theorem students_play_basketball 
  (total_students : ℕ)
  (cricket_players : ℕ)
  (both_players : ℕ)
  (total_students_eq : total_students = 880)
  (cricket_players_eq : cricket_players = 500)
  (both_players_eq : both_players = 220) 
  : ∃ B : ℕ, B = 600 :=
by
  sorry

end NUMINAMATH_GPT_students_play_basketball_l1355_135592


namespace NUMINAMATH_GPT_log_property_l1355_135523

theorem log_property (x : ℝ) (h₁ : Real.log x > 0) (h₂ : x > 1) : x > Real.exp 1 := by 
  sorry

end NUMINAMATH_GPT_log_property_l1355_135523


namespace NUMINAMATH_GPT_value_of_M_l1355_135526

theorem value_of_M (M : ℕ) : (32^3) * (16^3) = 2^M → M = 27 :=
by
  sorry

end NUMINAMATH_GPT_value_of_M_l1355_135526


namespace NUMINAMATH_GPT_robin_earns_30_percent_more_than_erica_l1355_135558

variable (E R C : ℝ)

theorem robin_earns_30_percent_more_than_erica
  (h1 : C = 1.60 * E)
  (h2 : C = 1.23076923076923077 * R) :
  R = 1.30 * E :=
by
  sorry

end NUMINAMATH_GPT_robin_earns_30_percent_more_than_erica_l1355_135558


namespace NUMINAMATH_GPT_cherry_pie_degrees_l1355_135566

theorem cherry_pie_degrees :
  ∀ (total_students chocolate_students apple_students blueberry_students : ℕ),
  total_students = 36 →
  chocolate_students = 12 →
  apple_students = 8 →
  blueberry_students = 6 →
  (total_students - chocolate_students - apple_students - blueberry_students) / 2 = 5 →
  ((5 : ℕ) * 360 / total_students) = 50 := 
by
  sorry

end NUMINAMATH_GPT_cherry_pie_degrees_l1355_135566


namespace NUMINAMATH_GPT_flour_more_than_sugar_l1355_135545

-- Define the conditions.
def sugar_needed : ℕ := 9
def total_flour_needed : ℕ := 14
def salt_needed : ℕ := 40
def flour_already_added : ℕ := 4

-- Define the target proof statement.
theorem flour_more_than_sugar :
  (total_flour_needed - flour_already_added) - sugar_needed = 1 :=
by
  -- sorry is used here to skip the proof.
  sorry

end NUMINAMATH_GPT_flour_more_than_sugar_l1355_135545


namespace NUMINAMATH_GPT_euler_children_mean_age_l1355_135563

-- Define the ages of each child
def ages : List ℕ := [8, 8, 8, 13, 13, 16]

-- Define the total number of children
def total_children := 6

-- Define the correct sum of ages
def total_sum_ages := 66

-- Define the correct answer (mean age)
def mean_age := 11

-- Prove that the mean (average) age of these children is 11
theorem euler_children_mean_age : (List.sum ages) / total_children = mean_age :=
by
  sorry

end NUMINAMATH_GPT_euler_children_mean_age_l1355_135563

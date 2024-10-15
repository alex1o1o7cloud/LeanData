import Mathlib

namespace NUMINAMATH_GPT_equation_has_one_integral_root_l1293_129388

theorem equation_has_one_integral_root:
  ∃ x : ℤ, (x - 9 / (x + 4 : ℝ) = 2 - 9 / (x + 4 : ℝ)) ∧ ∀ y : ℤ, 
  (y - 9 / (y + 4 : ℝ) = 2 - 9 / (y + 4 : ℝ)) → y = x := 
by
  sorry

end NUMINAMATH_GPT_equation_has_one_integral_root_l1293_129388


namespace NUMINAMATH_GPT_find_f_10_l1293_129342

-- Defining the function f as an odd, periodic function with period 2
def odd_func_periodic (f : ℝ → ℝ) : Prop :=
  (∀ x : ℝ, f (-x) = - f x) ∧ (∀ x : ℝ, f (x + 2) = f x)

-- Stating the theorem that f(10) is 0 given the conditions
theorem find_f_10 (f : ℝ → ℝ) (h1 : odd_func_periodic f) : f 10 = 0 :=
sorry

end NUMINAMATH_GPT_find_f_10_l1293_129342


namespace NUMINAMATH_GPT_find_lambda_l1293_129315

variables {a b : ℝ} (lambda : ℝ)

-- Conditions
def orthogonal (x y : ℝ) : Prop := x * y = 0
def magnitude_a : ℝ := 2
def magnitude_b : ℝ := 3
def is_perpendicular (x y : ℝ) : Prop := x * y = 0

-- Proof statement
theorem find_lambda (h₁ : orthogonal a b)
  (h₂ : magnitude_a = 2)
  (h₃ : magnitude_b = 3)
  (h₄ : is_perpendicular (3 * a + 2 * b) (lambda * a - b)) :
  lambda = 3 / 2 :=
sorry

end NUMINAMATH_GPT_find_lambda_l1293_129315


namespace NUMINAMATH_GPT_matt_needs_38_plates_l1293_129376

def plates_needed (days_with_only_matt_and_son days_with_parents plates_per_day plates_per_person_with_parents : ℕ) : ℕ :=
  (days_with_only_matt_and_son * plates_per_day) + (days_with_parents * 4 * plates_per_person_with_parents)

theorem matt_needs_38_plates :
  plates_needed 3 4 2 2 = 38 :=
by
  sorry

end NUMINAMATH_GPT_matt_needs_38_plates_l1293_129376


namespace NUMINAMATH_GPT_cows_dogs_ratio_l1293_129393

theorem cows_dogs_ratio (C D : ℕ) (hC : C = 184) (hC_remain : 3 / 4 * C = 138)
  (hD_remain : 1 / 4 * D + 138 = 161) : C / D = 2 :=
sorry

end NUMINAMATH_GPT_cows_dogs_ratio_l1293_129393


namespace NUMINAMATH_GPT_stutterer_square_number_unique_l1293_129302

-- Definitions based on problem conditions
def is_stutterer (n : ℕ) : Prop :=
  (1000 ≤ n ∧ n < 10000) ∧ (n / 100 = (n % 1000) / 100) ∧ ((n % 1000) % 100 = n % 10 * 10 + n % 10)

def is_square (n : ℕ) : Prop :=
  ∃ m : ℕ, m * m = n

-- The theorem statement
theorem stutterer_square_number_unique : ∃ n, is_stutterer n ∧ is_square n ∧ n = 7744 :=
by
  sorry

end NUMINAMATH_GPT_stutterer_square_number_unique_l1293_129302


namespace NUMINAMATH_GPT_exists_six_digit_no_identical_six_endings_l1293_129383

theorem exists_six_digit_no_identical_six_endings :
  ∃ (A : ℕ), (100000 ≤ A ∧ A < 1000000) ∧ ∀ (k : ℕ), (1 ≤ k ∧ k ≤ 500000) → 
  (∀ d, d ≠ 0 → d < 10 → (k * A) % 1000000 ≠ d * 111111) :=
by
  sorry

end NUMINAMATH_GPT_exists_six_digit_no_identical_six_endings_l1293_129383


namespace NUMINAMATH_GPT_minimize_transport_cost_l1293_129320

noncomputable def total_cost (v : ℝ) (a : ℝ) : ℝ :=
  if v > 0 ∧ v ≤ 80 then
    1000 * (v / 4 + a / v)
  else
    0

theorem minimize_transport_cost :
  ∀ v a : ℝ, a = 400 → (0 < v ∧ v ≤ 80) → total_cost v a = 20000 → v = 40 :=
by
  intros v a ha h_dom h_cost
  sorry

end NUMINAMATH_GPT_minimize_transport_cost_l1293_129320


namespace NUMINAMATH_GPT_James_gold_bars_l1293_129353

theorem James_gold_bars (P : ℝ) (h_condition1 : 60 - P / 100 * 60 = 54) : P = 10 := 
  sorry

end NUMINAMATH_GPT_James_gold_bars_l1293_129353


namespace NUMINAMATH_GPT_find_c_minus_a_l1293_129362

theorem find_c_minus_a (a b c : ℝ) (h1 : (a + b) / 2 = 45) (h2 : (b + c) / 2 = 50) : c - a = 10 :=
sorry

end NUMINAMATH_GPT_find_c_minus_a_l1293_129362


namespace NUMINAMATH_GPT_num_roots_of_unity_satisfy_cubic_l1293_129355

def root_of_unity (z : ℂ) (n : ℕ) : Prop :=
  z ^ n = 1

def cubic_eqn_root (z : ℂ) (a b c : ℤ) : Prop :=
  z^3 + (a:ℂ) * z^2 + (b:ℂ) * z + (c:ℂ) = 0

theorem num_roots_of_unity_satisfy_cubic (a b c : ℤ) (n : ℕ) 
    (h_n : n ≥ 1) : ∃! z : ℂ, root_of_unity z n ∧ cubic_eqn_root z a b c := sorry

end NUMINAMATH_GPT_num_roots_of_unity_satisfy_cubic_l1293_129355


namespace NUMINAMATH_GPT_fraction_of_sum_l1293_129387

theorem fraction_of_sum (l : List ℝ) (n : ℝ) (h_len : l.length = 21) (h_mem : n ∈ l)
  (h_n_avg : n = 4 * (l.erase n).sum / 20) :
  n / l.sum = 1 / 6 := by
  sorry

end NUMINAMATH_GPT_fraction_of_sum_l1293_129387


namespace NUMINAMATH_GPT_num_congruent_2_mod_11_l1293_129306

theorem num_congruent_2_mod_11 : 
  ∃ (n : ℕ), n = 28 ∧ ∀ k : ℤ, 1 ≤ 11 * k + 2 ∧ 11 * k + 2 ≤ 300 ↔ 0 ≤ k ∧ k ≤ 27 :=
sorry

end NUMINAMATH_GPT_num_congruent_2_mod_11_l1293_129306


namespace NUMINAMATH_GPT_damage_in_usd_correct_l1293_129323

def exchange_rate := (125 : ℚ) / 100
def damage_CAD := 45000000
def damage_USD := damage_CAD / exchange_rate

theorem damage_in_usd_correct (CAD_to_USD : exchange_rate = (125 : ℚ) / 100) (damage_in_cad : damage_CAD = 45000000) : 
  damage_USD = 36000000 :=
by
  sorry

end NUMINAMATH_GPT_damage_in_usd_correct_l1293_129323


namespace NUMINAMATH_GPT_lunchroom_tables_l1293_129318

/-- Given the total number of students and the number of students per table, 
    prove the number of tables in the lunchroom. -/
theorem lunchroom_tables (total_students : ℕ) (students_per_table : ℕ) 
  (h_total : total_students = 204) (h_per_table : students_per_table = 6) : 
  total_students / students_per_table = 34 := 
by
  sorry

end NUMINAMATH_GPT_lunchroom_tables_l1293_129318


namespace NUMINAMATH_GPT_no_positive_integer_n_such_that_14n_plus_19_is_prime_l1293_129327

theorem no_positive_integer_n_such_that_14n_plus_19_is_prime :
  ∀ n : Nat, 0 < n → ¬ Nat.Prime (14^n + 19) :=
by
  intro n hn
  sorry

end NUMINAMATH_GPT_no_positive_integer_n_such_that_14n_plus_19_is_prime_l1293_129327


namespace NUMINAMATH_GPT_fraction_calculation_l1293_129397

theorem fraction_calculation :
  ((1 / 2 + 1 / 5) / (3 / 7 - 1 / 14) = 49 / 25) := 
by 
  sorry

end NUMINAMATH_GPT_fraction_calculation_l1293_129397


namespace NUMINAMATH_GPT_intersection_eq_l1293_129365

noncomputable def A := {x : ℝ | x^2 - 4*x + 3 < 0 }
noncomputable def B := {x : ℝ | 2*x - 3 > 0 }

theorem intersection_eq : (A ∩ B) = {x : ℝ | (3 / 2) < x ∧ x < 3} := by
  sorry

end NUMINAMATH_GPT_intersection_eq_l1293_129365


namespace NUMINAMATH_GPT_necessary_and_sufficient_condition_l1293_129391

theorem necessary_and_sufficient_condition (x : ℝ) :
  x > 0 ↔ x + 1/x ≥ 2 :=
by sorry

end NUMINAMATH_GPT_necessary_and_sufficient_condition_l1293_129391


namespace NUMINAMATH_GPT_students_scoring_80_percent_l1293_129347

theorem students_scoring_80_percent
  (x : ℕ)
  (h1 : 10 * 90 + x * 80 = 25 * 84)
  (h2 : x + 10 = 25) : x = 15 := 
by {
  -- Proof goes here
  sorry
}

end NUMINAMATH_GPT_students_scoring_80_percent_l1293_129347


namespace NUMINAMATH_GPT_max_percentage_l1293_129339

def total_students : ℕ := 100
def group_size : ℕ := 66
def min_percentage (scores : Fin 100 → ℝ) : Prop :=
  ∀ (S : Finset (Fin 100)), S.card = 66 → (S.sum scores) / (Finset.univ.sum scores) ≥ 0.5

theorem max_percentage (scores : Fin 100 → ℝ) (h : min_percentage scores) :
  ∃ (x : ℝ), ∀ i : Fin 100, scores i <= x ∧ x <= 0.25 * (Finset.univ.sum scores) := sorry

end NUMINAMATH_GPT_max_percentage_l1293_129339


namespace NUMINAMATH_GPT_find_e_l1293_129399

variable (p j t e : ℝ)

def condition1 : Prop := j = 0.75 * p
def condition2 : Prop := j = 0.80 * t
def condition3 : Prop := t = p * (1 - e / 100)

theorem find_e (h1 : condition1 p j)
               (h2 : condition2 j t)
               (h3 : condition3 t e p) : e = 6.25 :=
by sorry

end NUMINAMATH_GPT_find_e_l1293_129399


namespace NUMINAMATH_GPT_stratified_sampling_major_C_l1293_129380

theorem stratified_sampling_major_C
  (students_A : ℕ) (students_B : ℕ) (students_C : ℕ) (students_D : ℕ)
  (total_students : ℕ) (sample_size : ℕ)
  (hA : students_A = 150) (hB : students_B = 150) (hC : students_C = 400) (hD : students_D = 300)
  (hTotal : total_students = students_A + students_B + students_C + students_D)
  (hSample : sample_size = 40)
  : students_C * (sample_size / total_students) = 16 :=
by
  sorry

end NUMINAMATH_GPT_stratified_sampling_major_C_l1293_129380


namespace NUMINAMATH_GPT_initial_balance_l1293_129329

theorem initial_balance (X : ℝ) : 
  (X - 60 - 30 - 0.25 * (X - 60 - 30) - 10 = 100) ↔ (X = 236.67) := 
  by
    sorry

end NUMINAMATH_GPT_initial_balance_l1293_129329


namespace NUMINAMATH_GPT_correct_statements_l1293_129384

def f : ℝ → ℝ := sorry

axiom even_function (x : ℝ) : f (-x) = f x
axiom monotonic_increasing_on_neg1_0 : ∀ ⦃x y : ℝ⦄, -1 ≤ x → x ≤ y → y ≤ 0 → f x ≤ f y
axiom functional_eqn (x : ℝ) : f (1 - x) + f (1 + x) = 0

theorem correct_statements :
  (∀ x, f (1 - x) = -f (1 + x)) ∧ f 2 ≤ f x :=
by
  sorry

end NUMINAMATH_GPT_correct_statements_l1293_129384


namespace NUMINAMATH_GPT_bananas_count_l1293_129309

/-- Elias bought some bananas and ate 1 of them. 
    After eating, he has 11 bananas left.
    Prove that Elias originally bought 12 bananas. -/
theorem bananas_count (x : ℕ) (h1 : x - 1 = 11) : x = 12 := by
  sorry

end NUMINAMATH_GPT_bananas_count_l1293_129309


namespace NUMINAMATH_GPT_joan_spent_on_trucks_l1293_129366

-- Define constants for the costs
def cost_cars : ℝ := 14.88
def cost_skateboard : ℝ := 4.88
def total_toys : ℝ := 25.62
def cost_trucks : ℝ := 25.62 - (14.88 + 4.88)

-- Statement to prove
theorem joan_spent_on_trucks : cost_trucks = 5.86 := by
  sorry

end NUMINAMATH_GPT_joan_spent_on_trucks_l1293_129366


namespace NUMINAMATH_GPT_rectangle_ratio_l1293_129335

theorem rectangle_ratio (s : ℝ) (x y : ℝ) 
  (h_outer_area : x * y * 4 + s^2 = 9 * s^2)
  (h_inner_outer_relation : s + 2 * y = 3 * s) :
  x / y = 2 :=
by {
  sorry
}

end NUMINAMATH_GPT_rectangle_ratio_l1293_129335


namespace NUMINAMATH_GPT_adam_apples_count_l1293_129344

variable (Jackie_apples : ℕ)
variable (extra_apples : ℕ)
variable (Adam_apples : ℕ)

theorem adam_apples_count (h1 : Jackie_apples = 9) (h2 : extra_apples = 5) (h3 : Adam_apples = Jackie_apples + extra_apples) :
  Adam_apples = 14 := 
by 
  sorry

end NUMINAMATH_GPT_adam_apples_count_l1293_129344


namespace NUMINAMATH_GPT_total_monsters_l1293_129333

theorem total_monsters (a1 a2 a3 a4 a5 : ℕ) 
  (h1 : a1 = 2) 
  (h2 : a2 = 2 * a1) 
  (h3 : a3 = 2 * a2) 
  (h4 : a4 = 2 * a3) 
  (h5 : a5 = 2 * a4) : 
  a1 + a2 + a3 + a4 + a5 = 62 :=
by
  sorry

end NUMINAMATH_GPT_total_monsters_l1293_129333


namespace NUMINAMATH_GPT_det_A_is_half_l1293_129382

noncomputable def A : Matrix (Fin 2) (Fin 2) ℝ :=
![![Real.cos (20 * Real.pi / 180), Real.sin (40 * Real.pi / 180)], ![Real.sin (20 * Real.pi / 180), Real.cos (40 * Real.pi / 180)]]

theorem det_A_is_half : A.det = 1 / 2 := by
  sorry

end NUMINAMATH_GPT_det_A_is_half_l1293_129382


namespace NUMINAMATH_GPT_slower_speed_is_l1293_129317

def slower_speed_problem
  (faster_speed : ℝ)
  (additional_distance : ℝ)
  (actual_distance : ℝ)
  (v : ℝ) :
  Prop :=
  actual_distance / v = (actual_distance + additional_distance) / faster_speed

theorem slower_speed_is
  (h1 : faster_speed = 25)
  (h2 : additional_distance = 20)
  (h3 : actual_distance = 13.333333333333332)
  : ∃ v : ℝ,  slower_speed_problem faster_speed additional_distance actual_distance v ∧ v = 10 :=
by {
  sorry
}

end NUMINAMATH_GPT_slower_speed_is_l1293_129317


namespace NUMINAMATH_GPT_correct_average_marks_l1293_129356

def incorrect_average := 100
def number_of_students := 10
def incorrect_mark := 60
def correct_mark := 10
def difference := incorrect_mark - correct_mark
def incorrect_total := incorrect_average * number_of_students
def correct_total := incorrect_total - difference

theorem correct_average_marks : correct_total / number_of_students = 95 := by
  sorry

end NUMINAMATH_GPT_correct_average_marks_l1293_129356


namespace NUMINAMATH_GPT_inequality_proof_l1293_129358

variable {a b c : ℝ}

theorem inequality_proof (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  (a / (1 + a + a * b)) + (b / (1 + b + b * c)) + (c / (1 + c + c * a)) ≤ 1 :=
by
  sorry

end NUMINAMATH_GPT_inequality_proof_l1293_129358


namespace NUMINAMATH_GPT_oprod_eval_l1293_129396

def oprod (a b : ℕ) : ℕ :=
  (a * 2 + b) / 2

theorem oprod_eval : oprod (oprod 4 6) 8 = 11 :=
by
  -- Definitions given in conditions
  let r := (4 * 2 + 6) / 2
  have h1 : oprod 4 6 = r := by rfl
  let s := (r * 2 + 8) / 2
  have h2 : oprod r 8 = s := by rfl
  exact (show s = 11 from sorry)

end NUMINAMATH_GPT_oprod_eval_l1293_129396


namespace NUMINAMATH_GPT_pond_water_after_45_days_l1293_129364

theorem pond_water_after_45_days :
  let initial_amount := 300
  let daily_evaporation := 1
  let rain_every_third_day := 2
  let total_days := 45
  let non_third_days := total_days - (total_days / 3)
  let third_days := total_days / 3
  let total_net_change := (non_third_days * (-daily_evaporation)) + (third_days * (rain_every_third_day - daily_evaporation))
  let final_amount := initial_amount + total_net_change
  final_amount = 285 :=
by
  sorry

end NUMINAMATH_GPT_pond_water_after_45_days_l1293_129364


namespace NUMINAMATH_GPT_pipes_height_l1293_129361

theorem pipes_height (d : ℝ) (h : ℝ) (r : ℝ) (s : ℝ)
  (hd : d = 12)
  (hs : s = d)
  (hr : r = d / 2)
  (heq : h = 6 * Real.sqrt 3 + r) :
  h = 6 * Real.sqrt 3 + 6 :=
by
  sorry

end NUMINAMATH_GPT_pipes_height_l1293_129361


namespace NUMINAMATH_GPT_same_points_among_teams_l1293_129398

theorem same_points_among_teams :
  ∀ (n : Nat), n = 28 → 
  ∀ (G D N : Nat), G = 378 → D >= 284 → N <= 94 →
  (∃ (team_scores : Fin n → Int), ∀ (i j : Fin n), i ≠ j → team_scores i = team_scores j) := by
sorry

end NUMINAMATH_GPT_same_points_among_teams_l1293_129398


namespace NUMINAMATH_GPT_trim_length_l1293_129395

theorem trim_length {π : ℝ} (r : ℝ)
  (π_approx : π = 22 / 7)
  (area : π * r^2 = 616) :
  2 * π * r + 5 = 93 :=
by
  sorry

end NUMINAMATH_GPT_trim_length_l1293_129395


namespace NUMINAMATH_GPT_novel_corona_high_students_l1293_129304

theorem novel_corona_high_students (students_know_it_all students_karen_high total_students students_novel_corona : ℕ)
  (h1 : students_know_it_all = 50)
  (h2 : students_karen_high = 3 / 5 * students_know_it_all)
  (h3 : total_students = 240)
  (h4 : students_novel_corona = total_students - (students_know_it_all + students_karen_high))
  : students_novel_corona = 160 :=
sorry

end NUMINAMATH_GPT_novel_corona_high_students_l1293_129304


namespace NUMINAMATH_GPT_find_value_of_y_l1293_129300

variable (p y : ℝ)
variable (h1 : p > 45)
variable (h2 : p * p / 100 = (2 * p / 300) * (p + y))

theorem find_value_of_y (h1 : p > 45) (h2 : p * p / 100 = (2 * p / 300) * (p + y)) : y = p / 2 :=
sorry

end NUMINAMATH_GPT_find_value_of_y_l1293_129300


namespace NUMINAMATH_GPT_seq_a3_eq_1_l1293_129346

theorem seq_a3_eq_1 (a : ℕ → ℤ) (h₁ : ∀ n ≥ 1, a (n + 1) = a n - 3) (h₂ : a 1 = 7) : a 3 = 1 :=
by
  sorry

end NUMINAMATH_GPT_seq_a3_eq_1_l1293_129346


namespace NUMINAMATH_GPT_least_positive_integer_l1293_129326

theorem least_positive_integer (a : ℕ) :
  (a % 2 = 0) ∧ (a % 5 = 1) ∧ (a % 4 = 2) → a = 6 :=
by
  sorry

end NUMINAMATH_GPT_least_positive_integer_l1293_129326


namespace NUMINAMATH_GPT_product_of_fractions_l1293_129303

theorem product_of_fractions :
  (2 / 3) * (5 / 8) * (1 / 4) = 5 / 48 := by
  sorry

end NUMINAMATH_GPT_product_of_fractions_l1293_129303


namespace NUMINAMATH_GPT_cannot_be_the_lengths_l1293_129337

theorem cannot_be_the_lengths (x y z : ℝ) (h1 : x^2 + y^2 = 16) (h2 : x^2 + z^2 = 25) (h3 : y^2 + z^2 = 49) : false :=
by
  sorry

end NUMINAMATH_GPT_cannot_be_the_lengths_l1293_129337


namespace NUMINAMATH_GPT_reggie_games_lost_l1293_129338

-- Define the necessary conditions
def initial_marbles : ℕ := 100
def bet_per_game : ℕ := 10
def marbles_after_games : ℕ := 90
def total_games : ℕ := 9

-- Define the proof problem statement
theorem reggie_games_lost : (initial_marbles - marbles_after_games) / bet_per_game = 1 := by
  sorry

end NUMINAMATH_GPT_reggie_games_lost_l1293_129338


namespace NUMINAMATH_GPT_find_y_value_l1293_129350

def op (a b : ℤ) : ℤ := 4 * a + 2 * b

theorem find_y_value : ∃ y : ℤ, op 3 (op 4 y) = -14 ∧ y = -29 / 2 := sorry

end NUMINAMATH_GPT_find_y_value_l1293_129350


namespace NUMINAMATH_GPT_gcd_of_g_y_l1293_129331

def g (y : ℕ) : ℕ := (3 * y + 4) * (8 * y + 3) * (11 * y + 5) * (y + 11)

theorem gcd_of_g_y (y : ℕ) (hy : ∃ k, y = 30492 * k) : Nat.gcd (g y) y = 660 :=
by
  sorry

end NUMINAMATH_GPT_gcd_of_g_y_l1293_129331


namespace NUMINAMATH_GPT_calculate_expression_l1293_129340

theorem calculate_expression : 
  let x := 7.5
  let y := 2.5
  (x ^ y + Real.sqrt x + y ^ x) - (x ^ 2 + y ^ y + Real.sqrt y) = 679.2044 :=
by
  sorry

end NUMINAMATH_GPT_calculate_expression_l1293_129340


namespace NUMINAMATH_GPT_magnitude_z_is_sqrt_2_l1293_129325

open Complex

noncomputable def z (x y : ℝ) : ℂ := x + y * I

theorem magnitude_z_is_sqrt_2 (x y : ℝ) (h1 : (2 * x) / (1 - I) = 1 + y * I) : abs (z x y) = Real.sqrt 2 :=
by
  -- You would fill in the proof steps here based on the problem's solution.
  sorry

end NUMINAMATH_GPT_magnitude_z_is_sqrt_2_l1293_129325


namespace NUMINAMATH_GPT_exists_linear_function_intersecting_negative_axes_l1293_129332

theorem exists_linear_function_intersecting_negative_axes :
  ∃ (k b : ℝ), k < 0 ∧ b < 0 ∧ (∃ x, k * x + b = 0 ∧ x < 0) ∧ (k * 0 + b < 0) :=
by
  sorry

end NUMINAMATH_GPT_exists_linear_function_intersecting_negative_axes_l1293_129332


namespace NUMINAMATH_GPT_sum_of_first_five_terms_l1293_129375

noncomputable def S₅ (a : ℕ → ℝ) := (a 1 + a 5) / 2 * 5

theorem sum_of_first_five_terms (a : ℕ → ℝ) (a_2 a_4 : ℝ)
  (h1 : a 2 = 4)
  (h2 : a 4 = 2)
  (h3 : ∀ n : ℕ, a n = a 1 + (n - 1) * (a 2 - a 1)) :
  S₅ a = 15 :=
sorry

end NUMINAMATH_GPT_sum_of_first_five_terms_l1293_129375


namespace NUMINAMATH_GPT_decreased_amount_l1293_129367

theorem decreased_amount {N A : ℝ} (h₁ : 0.20 * N - A = 6) (h₂ : N = 50) : A = 4 := by
  sorry

end NUMINAMATH_GPT_decreased_amount_l1293_129367


namespace NUMINAMATH_GPT_equal_areas_of_parts_l1293_129357

theorem equal_areas_of_parts :
  ∀ (S1 S2 S3 S4 : ℝ), 
    S1 = S2 → S2 = S3 → 
    (S1 + S2 = S3 + S4) → 
    (S2 + S3 = S1 + S4) → 
    S1 = S2 ∧ S2 = S3 ∧ S3 = S4 :=
by
  intros S1 S2 S3 S4 h1 h2 h3 h4
  sorry

end NUMINAMATH_GPT_equal_areas_of_parts_l1293_129357


namespace NUMINAMATH_GPT_solve_for_y_l1293_129316

theorem solve_for_y (y : ℝ) (h : 3 / y + 4 / y / (6 / y) = 1.5) : y = 3.6 :=
sorry

end NUMINAMATH_GPT_solve_for_y_l1293_129316


namespace NUMINAMATH_GPT_solution_set_of_inequality_l1293_129377

theorem solution_set_of_inequality (x : ℝ) :
  (|x| - 2) * (x - 1) ≥ 0 ↔ (-2 ≤ x ∧ x ≤ 1) ∨ (x ≥ 2) :=
by
  sorry

end NUMINAMATH_GPT_solution_set_of_inequality_l1293_129377


namespace NUMINAMATH_GPT_find_m_of_slope_is_12_l1293_129370

theorem find_m_of_slope_is_12 (m : ℝ) :
  let A := (-m, 6)
  let B := (1, 3 * m)
  let slope := (3 * m - 6) / (1 + m)
  slope = 12 → m = -2 :=
by
  sorry

end NUMINAMATH_GPT_find_m_of_slope_is_12_l1293_129370


namespace NUMINAMATH_GPT_solve_for_x_l1293_129349

theorem solve_for_x : ∃ x : ℝ, (6 * x) / 1.5 = 3.8 ∧ x = 0.95 := by
  use 0.95
  exact ⟨by norm_num, by norm_num⟩

end NUMINAMATH_GPT_solve_for_x_l1293_129349


namespace NUMINAMATH_GPT_find_k_l1293_129389

noncomputable def is_perfect_square (k : ℝ) : Prop :=
  ∀ x : ℝ, ∃ a : ℝ, x^2 + 2*(k-1)*x + 64 = (x + a)^2

theorem find_k (k : ℝ) : is_perfect_square k ↔ (k = 9 ∨ k = -7) :=
sorry

end NUMINAMATH_GPT_find_k_l1293_129389


namespace NUMINAMATH_GPT_polynomial_horner_v4_value_l1293_129385

-- Define the polynomial f(x)
def f (x : ℤ) : ℤ := x^6 - 12*x^5 + 60*x^4 - 160*x^3 + 240*x^2 - 192*x + 64

-- Define Horner's Rule step by step for x = 2
def horner_eval (x : ℤ) : ℤ :=
  let v0 := 1
  let v1 := v0 * x - 12
  let v2 := v1 * x + 60
  let v3 := v2 * x - 160
  let v4 := v3 * x + 240
  v4

-- Prove that the value of v4 when x = 2 is 80
theorem polynomial_horner_v4_value : horner_eval 2 = 80 := by
  sorry

end NUMINAMATH_GPT_polynomial_horner_v4_value_l1293_129385


namespace NUMINAMATH_GPT_cost_per_dozen_l1293_129307

theorem cost_per_dozen (total_cost : ℝ) (total_rolls dozens : ℝ) (cost_per_dozen : ℝ) (h₁ : total_cost = 15) (h₂ : total_rolls = 36) (h₃ : dozens = total_rolls / 12) (h₄ : cost_per_dozen = total_cost / dozens) : cost_per_dozen = 5 :=
by
  sorry

end NUMINAMATH_GPT_cost_per_dozen_l1293_129307


namespace NUMINAMATH_GPT_sqrt_a_squared_b_l1293_129371

variable {a b : ℝ}

theorem sqrt_a_squared_b (h: a * b < 0) : Real.sqrt (a^2 * b) = -a * Real.sqrt b := by
  sorry

end NUMINAMATH_GPT_sqrt_a_squared_b_l1293_129371


namespace NUMINAMATH_GPT_necessary_but_not_sufficient_l1293_129381

open Set

namespace Mathlib

noncomputable def M : Set ℝ := {x | 0 < x ∧ x ≤ 3}
noncomputable def N : Set ℝ := {x | 0 < x ∧ x ≤ 2}

theorem necessary_but_not_sufficient (a : ℝ) : 
  (a ∈ M → a ∈ N) ∧ ¬(a ∈ N → a ∈ M) :=
by
  sorry

end Mathlib

end NUMINAMATH_GPT_necessary_but_not_sufficient_l1293_129381


namespace NUMINAMATH_GPT_common_speed_is_10_l1293_129390

noncomputable def speed_jack (x : ℝ) : ℝ := x^2 - 11 * x - 22
noncomputable def speed_jill (x : ℝ) : ℝ := 
  if x = -6 then 0 else (x^2 - 4 * x - 12) / (x + 6)

theorem common_speed_is_10 (x : ℝ) (h : speed_jack x = speed_jill x) (hx : x = 16) : 
  speed_jack x = 10 :=
by
  sorry

end NUMINAMATH_GPT_common_speed_is_10_l1293_129390


namespace NUMINAMATH_GPT_rectangle_diagonal_length_l1293_129348

theorem rectangle_diagonal_length :
  ∀ (length width diagonal : ℝ), length = 6 ∧ length * width = 48 ∧ diagonal = Real.sqrt (length^2 + width^2) → diagonal = 10 :=
by
  intro length width diagonal
  rintro ⟨hl, area_eq, diagonal_eq⟩
  sorry

end NUMINAMATH_GPT_rectangle_diagonal_length_l1293_129348


namespace NUMINAMATH_GPT_sequence_property_l1293_129330

theorem sequence_property (a : ℕ → ℕ) (h1 : a 1 = 1)
  (h_rec : ∀ m n : ℕ, a (m + n) = a m + a n + m * n) :
  a 10 = 55 :=
sorry

end NUMINAMATH_GPT_sequence_property_l1293_129330


namespace NUMINAMATH_GPT_distance_traveled_l1293_129312

-- Define the variables for speed of slower and faster bike
def slower_speed := 60
def faster_speed := 64

-- Define the condition that slower bike takes 1 hour more than faster bike
def condition (D : ℝ) : Prop := (D / slower_speed) = (D / faster_speed) + 1

-- The theorem we need to prove
theorem distance_traveled : ∃ (D : ℝ), condition D ∧ D = 960 := 
by
  sorry

end NUMINAMATH_GPT_distance_traveled_l1293_129312


namespace NUMINAMATH_GPT_point_A_coordinates_l1293_129301

noncomputable def f (a x : ℝ) : ℝ := a * x - 1

theorem point_A_coordinates (a : ℝ) (h1 : a > 0) (h2 : a ≠ 1) : f a 1 = 1 :=
sorry

end NUMINAMATH_GPT_point_A_coordinates_l1293_129301


namespace NUMINAMATH_GPT_inequality_must_hold_l1293_129321

variable (a b c : ℝ)

theorem inequality_must_hold (h1 : a > b) (h2 : c < 0) : a * (c - 1) < b * (c - 1) := 
sorry

end NUMINAMATH_GPT_inequality_must_hold_l1293_129321


namespace NUMINAMATH_GPT_number_of_ways_to_assign_volunteers_l1293_129314

/-- Theorem: The number of ways to assign 5 volunteers to 3 venues such that each venue has at least one volunteer is 150. -/
theorem number_of_ways_to_assign_volunteers :
  let total_ways := 3^5
  let subtract_one_empty := 3 * 2^5
  let add_back_two_empty := 3 * 1^5
  (total_ways - subtract_one_empty + add_back_two_empty) = 150 :=
by
  sorry

end NUMINAMATH_GPT_number_of_ways_to_assign_volunteers_l1293_129314


namespace NUMINAMATH_GPT_balls_picked_at_random_eq_two_l1293_129345

-- Define the initial conditions: number of balls of each color
def num_red_balls : ℕ := 5
def num_blue_balls : ℕ := 4
def num_green_balls : ℕ := 3

-- Define the total number of balls
def total_balls : ℕ := num_red_balls + num_blue_balls + num_green_balls

-- Define the given probability
def given_probability : ℚ := 0.15151515151515152

-- Define the probability calculation for picking two red balls
def probability_two_reds : ℚ :=
  (num_red_balls / total_balls) * ((num_red_balls - 1) / (total_balls - 1))

-- The theorem to prove
theorem balls_picked_at_random_eq_two :
  probability_two_reds = given_probability → n = 2 :=
by
  sorry

end NUMINAMATH_GPT_balls_picked_at_random_eq_two_l1293_129345


namespace NUMINAMATH_GPT_value_of_f_neg1_l1293_129341

def f (x : ℤ) : ℤ := x^2 - 2 * x

theorem value_of_f_neg1 : f (-1) = 3 := by
  sorry

end NUMINAMATH_GPT_value_of_f_neg1_l1293_129341


namespace NUMINAMATH_GPT_topping_cost_l1293_129352

noncomputable def cost_of_topping (ic_cost sundae_cost number_of_toppings: ℝ) : ℝ :=
(sundae_cost - ic_cost) / number_of_toppings

theorem topping_cost
  (ic_cost : ℝ)
  (sundae_cost : ℝ)
  (number_of_toppings : ℕ)
  (h_ic_cost : ic_cost = 2)
  (h_sundae_cost : sundae_cost = 7)
  (h_number_of_toppings : number_of_toppings = 10) :
  cost_of_topping ic_cost sundae_cost number_of_toppings = 0.5 :=
  by
  -- Proof will be here
  sorry

end NUMINAMATH_GPT_topping_cost_l1293_129352


namespace NUMINAMATH_GPT_betty_blue_beads_l1293_129378

theorem betty_blue_beads (r b : ℕ) (h1 : r = 30) (h2 : 3 * b = 2 * r) : b = 20 :=
by
  sorry

end NUMINAMATH_GPT_betty_blue_beads_l1293_129378


namespace NUMINAMATH_GPT_credit_card_more_beneficial_l1293_129374

def gift_cost : ℝ := 8000
def credit_card_cashback_rate : ℝ := 0.005
def debit_card_cashback_rate : ℝ := 0.0075
def debit_card_interest_rate : ℝ := 0.005

def credit_card_total_income : ℝ := gift_cost * (credit_card_cashback_rate + debit_card_interest_rate)
def debit_card_total_income : ℝ := gift_cost * debit_card_cashback_rate

theorem credit_card_more_beneficial :
  credit_card_total_income > debit_card_total_income :=
by
  sorry

end NUMINAMATH_GPT_credit_card_more_beneficial_l1293_129374


namespace NUMINAMATH_GPT_prove_f_of_increasing_l1293_129354

noncomputable def odd_function (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f (-x) = -f x

noncomputable def strictly_increasing_on_positives (f : ℝ → ℝ) : Prop :=
  ∀ x₁ x₂ : ℝ, 0 < x₁ ∧ 0 < x₂ ∧ x₁ ≠ x₂ → (f x₁ - f x₂) / (x₁ - x₂) > 0

theorem prove_f_of_increasing {f : ℝ → ℝ}
  (h_odd : odd_function f)
  (h_incr : strictly_increasing_on_positives f) :
  f (-3) > f (-5) :=
by
  sorry

end NUMINAMATH_GPT_prove_f_of_increasing_l1293_129354


namespace NUMINAMATH_GPT_all_equal_l1293_129368

theorem all_equal (xs xsp : Fin 2011 → ℝ) (h : ∀ i : Fin 2011, xs i + xs ((i + 1) % 2011) = 2 * xsp i) (perm : ∃ σ : Fin 2011 ≃ Fin 2011, ∀ i, xsp i = xs (σ i)) :
  ∀ i j : Fin 2011, xs i = xs j := 
sorry

end NUMINAMATH_GPT_all_equal_l1293_129368


namespace NUMINAMATH_GPT_loss_percentage_l1293_129334

theorem loss_percentage
  (CP : ℝ := 1166.67)
  (SP : ℝ)
  (H : SP + 140 = CP + 0.02 * CP) :
  ((CP - SP) / CP) * 100 = 10 := 
by 
  sorry

end NUMINAMATH_GPT_loss_percentage_l1293_129334


namespace NUMINAMATH_GPT_minimize_sum_find_c_l1293_129360

theorem minimize_sum_find_c (a b c d e f : ℕ) (h : a + 2 * b + 6 * c + 30 * d + 210 * e + 2310 * f = 2 ^ 15) 
  (h_min : ∀ a' b' c' d' e' f' : ℕ, a' + 2 * b' + 6 * c' + 30 * d' + 210 * e' + 2310 * f' = 2 ^ 15 → 
  a' + b' + c' + d' + e' + f' ≥ a + b + c + d + e + f) :
  c = 1 :=
sorry

end NUMINAMATH_GPT_minimize_sum_find_c_l1293_129360


namespace NUMINAMATH_GPT_sector_area_l1293_129313

theorem sector_area (θ r a : ℝ) (hθ : θ = 2) (haarclength : r * θ = 4) : 
  (1/2) * r * r * θ = 4 :=
by {
  -- Proof goes here
  sorry
}

end NUMINAMATH_GPT_sector_area_l1293_129313


namespace NUMINAMATH_GPT_price_of_33_kgs_l1293_129305

theorem price_of_33_kgs (l q : ℝ) 
  (h1 : l * 20 = 100) 
  (h2 : l * 30 + q * 6 = 186) : 
  l * 30 + q * 3 = 168 := 
by
  sorry

end NUMINAMATH_GPT_price_of_33_kgs_l1293_129305


namespace NUMINAMATH_GPT_brazil_medal_fraction_closest_l1293_129394

theorem brazil_medal_fraction_closest :
  let frac_win : ℚ := 23 / 150
  let frac_1_6 : ℚ := 1 / 6
  let frac_1_7 : ℚ := 1 / 7
  let frac_1_8 : ℚ := 1 / 8
  let frac_1_9 : ℚ := 1 / 9
  let frac_1_10 : ℚ := 1 / 10
  abs (frac_win - frac_1_7) < abs (frac_win - frac_1_6) ∧
  abs (frac_win - frac_1_7) < abs (frac_win - frac_1_8) ∧
  abs (frac_win - frac_1_7) < abs (frac_win - frac_1_9) ∧
  abs (frac_win - frac_1_7) < abs (frac_win - frac_1_10) :=
by
  sorry

end NUMINAMATH_GPT_brazil_medal_fraction_closest_l1293_129394


namespace NUMINAMATH_GPT_isosceles_triangle_area_l1293_129343

theorem isosceles_triangle_area (a b c : ℝ) (h: a = 5 ∧ b = 5 ∧ c = 6)
  (altitude_splits_base : ∀ (h : 3^2 + x^2 = 25), x = 4) : 
  ∃ (area : ℝ), area = 12 := 
by
  sorry

end NUMINAMATH_GPT_isosceles_triangle_area_l1293_129343


namespace NUMINAMATH_GPT_solve_for_x_l1293_129359

theorem solve_for_x (x : ℝ) (h : 1 = 1 / (4 * x^2 + 2 * x + 1)) : 
  x = 0 ∨ x = -1 / 2 := 
by sorry

end NUMINAMATH_GPT_solve_for_x_l1293_129359


namespace NUMINAMATH_GPT_triple_integral_value_l1293_129369

theorem triple_integral_value :
  (∫ x in (-1 : ℝ)..1, ∫ y in (x^2 : ℝ)..1, ∫ z in (0 : ℝ)..y, (4 + z) ) = (16 / 3 : ℝ) :=
by
  sorry

end NUMINAMATH_GPT_triple_integral_value_l1293_129369


namespace NUMINAMATH_GPT_linda_total_distance_l1293_129311

theorem linda_total_distance :
  ∃ x : ℕ, (60 % x = 0) ∧ ((75 % (x + 3)) = 0) ∧ ((90 % (x + 6)) = 0) ∧
  (60 / x + 75 / (x + 3) + 90 / (x + 6) = 15) :=
sorry

end NUMINAMATH_GPT_linda_total_distance_l1293_129311


namespace NUMINAMATH_GPT_egg_distribution_l1293_129308

-- Definitions of the conditions
def total_eggs := 10.0
def large_eggs := 6.0
def small_eggs := 4.0

def box_A_capacity := 5.0
def box_B_capacity := 4.0
def box_C_capacity := 6.0

def at_least_one_small_egg (box_A_small box_B_small box_C_small : Float) := 
  box_A_small >= 1.0 ∧ box_B_small >= 1.0 ∧ box_C_small >= 1.0

-- Problem statement
theorem egg_distribution : 
  ∃ (box_A_small box_A_large box_B_small box_B_large box_C_small box_C_large : Float),
  box_A_small + box_A_large <= box_A_capacity ∧
  box_B_small + box_B_large <= box_B_capacity ∧
  box_C_small + box_C_large <= box_C_capacity ∧
  box_A_small + box_B_small + box_C_small = small_eggs ∧
  box_A_large + box_B_large + box_C_large = large_eggs ∧
  at_least_one_small_egg box_A_small box_B_small box_C_small :=
sorry

end NUMINAMATH_GPT_egg_distribution_l1293_129308


namespace NUMINAMATH_GPT_arithmetic_problem_l1293_129319

theorem arithmetic_problem : 90 + 5 * 12 / (180 / 3) = 91 := by
  sorry

end NUMINAMATH_GPT_arithmetic_problem_l1293_129319


namespace NUMINAMATH_GPT_green_pill_cost_l1293_129324

-- Definitions for the problem conditions
def number_of_days : ℕ := 21
def total_cost : ℚ := 819
def daily_cost : ℚ := total_cost / number_of_days
def cost_green_pill (x : ℚ) : ℚ := x
def cost_pink_pill (x : ℚ) : ℚ := x - 1
def total_daily_pill_cost (x : ℚ) : ℚ := cost_green_pill x + 2 * cost_pink_pill x

-- Theorem to be proven
theorem green_pill_cost : ∃ x : ℚ, total_daily_pill_cost x = daily_cost ∧ x = 41 / 3 :=
sorry

end NUMINAMATH_GPT_green_pill_cost_l1293_129324


namespace NUMINAMATH_GPT_triangle_with_consecutive_sides_and_angle_property_l1293_129351

theorem triangle_with_consecutive_sides_and_angle_property :
  ∃ (a b c : ℕ), (b = a + 1) ∧ (c = b + 1) ∧
    (∃ (α β γ : ℝ), 2 * α = γ ∧
      (a * a + b * b = c * c + 2 * a * b * α.cos) ∧
      (b * b + c * c = a * a + 2 * b * c * β.cos) ∧
      (c * c + a * a = b * b + 2 * c * a * γ.cos) ∧
      (a = 4) ∧ (b = 5) ∧ (c = 6) ∧
      (γ.cos = 1 / 8)) :=
sorry

end NUMINAMATH_GPT_triangle_with_consecutive_sides_and_angle_property_l1293_129351


namespace NUMINAMATH_GPT_base_of_parallelogram_l1293_129363

variable (Area Height Base : ℝ)

def parallelogram_area (base height : ℝ) : ℝ := base * height

theorem base_of_parallelogram
  (h_area : Area = 200)
  (h_height : Height = 20)
  (h_area_def : parallelogram_area Base Height = Area) :
  Base = 10 :=
by sorry

end NUMINAMATH_GPT_base_of_parallelogram_l1293_129363


namespace NUMINAMATH_GPT_brooke_earns_144_dollars_l1293_129386

-- Definitions based on the identified conditions
def price_of_milk_per_gallon : ℝ := 3
def production_cost_per_gallon_of_butter : ℝ := 0.5
def sticks_of_butter_per_gallon : ℝ := 2
def price_of_butter_per_stick : ℝ := 1.5
def number_of_cows : ℕ := 12
def milk_per_cow : ℝ := 4
def number_of_customers : ℕ := 6
def min_milk_per_customer : ℝ := 4
def max_milk_per_customer : ℝ := 8

-- Auxiliary calculations
def total_milk_produced : ℝ := number_of_cows * milk_per_cow
def min_total_customer_demand : ℝ := number_of_customers * min_milk_per_customer
def max_total_customer_demand : ℝ := number_of_customers * max_milk_per_customer

-- Problem statement
theorem brooke_earns_144_dollars :
  (0 <= total_milk_produced) ∧
  (min_total_customer_demand <= max_total_customer_demand) ∧
  (total_milk_produced = max_total_customer_demand) →
  (total_milk_produced * price_of_milk_per_gallon = 144) :=
by
  -- Sorry is added here since the proof is not required
  sorry

end NUMINAMATH_GPT_brooke_earns_144_dollars_l1293_129386


namespace NUMINAMATH_GPT_right_triangle_AB_CA_BC_l1293_129328

namespace TriangleProof

def point := ℝ × ℝ

def dist (p1 p2 : point) : ℝ :=
  (p1.1 - p2.1) ^ 2 + (p1.2 - p2.2) ^ 2

def A : point := (5, -2)
def B : point := (1, 5)
def C : point := (-1, 2)

def AB2 := dist A B
def BC2 := dist B C
def CA2 := dist C A

theorem right_triangle_AB_CA_BC : CA2 + BC2 = AB2 :=
by 
  -- proof will be filled here
  sorry

end TriangleProof

end NUMINAMATH_GPT_right_triangle_AB_CA_BC_l1293_129328


namespace NUMINAMATH_GPT_solve_linear_system_l1293_129392

variable {x y : ℚ}

theorem solve_linear_system (h1 : 4 * x - 3 * y = -17) (h2 : 5 * x + 6 * y = -4) :
  (x, y) = (-(74 / 13 : ℚ), -(25 / 13 : ℚ)) :=
by
  sorry

end NUMINAMATH_GPT_solve_linear_system_l1293_129392


namespace NUMINAMATH_GPT_simplify_fraction_product_l1293_129322

theorem simplify_fraction_product : 
  (256 / 20 : ℚ) * (10 / 160) * ((16 / 6) ^ 2) = 256 / 45 :=
by norm_num

end NUMINAMATH_GPT_simplify_fraction_product_l1293_129322


namespace NUMINAMATH_GPT_hall_width_to_length_ratio_l1293_129310

def width (w l : ℝ) : Prop := w * l = 578
def length_width_difference (w l : ℝ) : Prop := l - w = 17

theorem hall_width_to_length_ratio (w l : ℝ) (hw : width w l) (hl : length_width_difference w l) : (w / l = 1 / 2) :=
by
  sorry

end NUMINAMATH_GPT_hall_width_to_length_ratio_l1293_129310


namespace NUMINAMATH_GPT_find_larger_number_l1293_129379

variable {x y : ℕ} 

theorem find_larger_number (h_ratio : 4 * x = 3 * y) (h_sum : x + y + 100 = 500) : y = 1600 / 7 := by 
  sorry

end NUMINAMATH_GPT_find_larger_number_l1293_129379


namespace NUMINAMATH_GPT_probability_four_of_eight_show_three_l1293_129336

def probability_exactly_four_show_three : ℚ :=
  let num_ways := Nat.choose 8 4
  let prob_four_threes := (1 / 6) ^ 4
  let prob_four_not_threes := (5 / 6) ^ 4
  (num_ways * prob_four_threes * prob_four_not_threes)

theorem probability_four_of_eight_show_three :
  probability_exactly_four_show_three = 43750 / 1679616 :=
by 
  sorry

end NUMINAMATH_GPT_probability_four_of_eight_show_three_l1293_129336


namespace NUMINAMATH_GPT_marble_problem_l1293_129372

-- Define the initial number of marbles
def initial_marbles : Prop :=
  ∃ (x y : ℕ), (y - 4 = 2 * (x + 4)) ∧ (y + 2 = 11 * (x - 2)) ∧ (y = 20) ∧ (x = 4)

-- The main theorem to prove the initial number of marbles
theorem marble_problem (x y : ℕ) (cond1 : y - 4 = 2 * (x + 4)) (cond2 : y + 2 = 11 * (x - 2)) :
  y = 20 ∧ x = 4 :=
sorry

end NUMINAMATH_GPT_marble_problem_l1293_129372


namespace NUMINAMATH_GPT_urea_moles_produced_l1293_129373

-- Define the reaction
def chemical_reaction (CO2 NH3 Urea Water : ℕ) :=
  CO2 = 1 ∧ NH3 = 2 ∧ Urea = 1 ∧ Water = 1

-- Given initial moles of reactants
def initial_moles (CO2 NH3 : ℕ) :=
  CO2 = 1 ∧ NH3 = 2

-- The main theorem to prove
theorem urea_moles_produced (CO2 NH3 Urea Water : ℕ) :
  initial_moles CO2 NH3 → chemical_reaction CO2 NH3 Urea Water → Urea = 1 :=
by
  intro H1 H2
  rcases H1 with ⟨HCO2, HNH3⟩
  rcases H2 with ⟨HCO2', HNH3', HUrea, _⟩
  sorry

end NUMINAMATH_GPT_urea_moles_produced_l1293_129373

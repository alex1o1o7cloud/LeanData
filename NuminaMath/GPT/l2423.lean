import Mathlib

namespace NUMINAMATH_GPT_solve_fractional_equation_l2423_242365

theorem solve_fractional_equation : ∀ (x : ℝ), x ≠ 0 ∧ x ≠ -3 → (1 / x = 2 / (x + 3) ↔ x = 3) :=
by
  intros x hx
  sorry

end NUMINAMATH_GPT_solve_fractional_equation_l2423_242365


namespace NUMINAMATH_GPT_similar_triangle_shortest_side_l2423_242378

theorem similar_triangle_shortest_side
  (a₁ : ℕ) (c₁ : ℕ) (c₂ : ℕ)
  (h₁ : a₁ = 15) (h₂ : c₁ = 17) (h₃ : c₂ = 68)
  (right_triangle_1 : a₁^2 + b₁^2 = c₁^2)
  (similar_triangles : ∃ k : ℕ, c₂ = k * c₁) :
  shortest_side = 32 := 
sorry

end NUMINAMATH_GPT_similar_triangle_shortest_side_l2423_242378


namespace NUMINAMATH_GPT_swiss_probability_is_30_percent_l2423_242316

def total_cheese_sticks : Nat := 22 + 34 + 29 + 45 + 20

def swiss_cheese_sticks : Nat := 45

def probability_swiss : Nat :=
  (swiss_cheese_sticks * 100) / total_cheese_sticks

theorem swiss_probability_is_30_percent :
  probability_swiss = 30 := by
  sorry

end NUMINAMATH_GPT_swiss_probability_is_30_percent_l2423_242316


namespace NUMINAMATH_GPT_Sue_made_22_buttons_l2423_242356

def Mari_buttons : Nat := 8
def Kendra_buttons : Nat := 5 * Mari_buttons + 4
def Sue_buttons : Nat := Kendra_buttons / 2

theorem Sue_made_22_buttons : Sue_buttons = 22 :=
by
  -- proof to be added
  sorry

end NUMINAMATH_GPT_Sue_made_22_buttons_l2423_242356


namespace NUMINAMATH_GPT_subset_M_N_l2423_242395

def M : Set ℝ := {-1, 1}
def N : Set ℝ := { x | (1 / x < 2) }

theorem subset_M_N : M ⊆ N :=
by
  sorry -- Proof omitted as per the guidelines

end NUMINAMATH_GPT_subset_M_N_l2423_242395


namespace NUMINAMATH_GPT_least_integer_square_eq_12_more_than_three_times_l2423_242373

theorem least_integer_square_eq_12_more_than_three_times (x : ℤ) (h : x^2 = 3 * x + 12) : x = -3 :=
sorry

end NUMINAMATH_GPT_least_integer_square_eq_12_more_than_three_times_l2423_242373


namespace NUMINAMATH_GPT_candy_count_l2423_242314

theorem candy_count (S : ℕ) (H1 : 32 + S - 35 = 39) : S = 42 :=
by
  sorry

end NUMINAMATH_GPT_candy_count_l2423_242314


namespace NUMINAMATH_GPT_sqrt_square_l2423_242396

theorem sqrt_square (n : ℝ) : (Real.sqrt 2023) ^ 2 = 2023 :=
by
  sorry

end NUMINAMATH_GPT_sqrt_square_l2423_242396


namespace NUMINAMATH_GPT_max_fm_n_l2423_242325

noncomputable def ln (x : ℝ) : ℝ := Real.log x

def g (m n x : ℝ) : ℝ := (2 * m + 3) * x + n

def condition_f_g (m n : ℝ) : Prop := ∀ x > 0, ln x ≤ g m n x

def f (m : ℝ) : ℝ := 2 * m + 3

theorem max_fm_n (m n : ℝ) (h : condition_f_g m n) : (f m) * n ≤ 1 / Real.exp 2 := sorry

end NUMINAMATH_GPT_max_fm_n_l2423_242325


namespace NUMINAMATH_GPT_num_girls_at_park_l2423_242329

theorem num_girls_at_park (G : ℕ) (h1 : 11 + 50 + G = 3 * 25) : G = 14 := by
  sorry

end NUMINAMATH_GPT_num_girls_at_park_l2423_242329


namespace NUMINAMATH_GPT_train_speed_correct_l2423_242388

def train_length : ℝ := 1500
def crossing_time : ℝ := 15
def correct_speed : ℝ := 100

theorem train_speed_correct : (train_length / crossing_time) = correct_speed := by 
  sorry

end NUMINAMATH_GPT_train_speed_correct_l2423_242388


namespace NUMINAMATH_GPT_length_of_train_is_correct_l2423_242384

noncomputable def train_length (speed_km_hr : ℝ) (time_sec : ℝ) : ℝ :=
  let speed_m_s := (speed_km_hr * 1000) / 3600
  speed_m_s * time_sec

theorem length_of_train_is_correct (speed_km_hr : ℝ) (time_sec : ℝ) (expected_length : ℝ) :
  speed_km_hr = 60 → time_sec = 21 → expected_length = 350.07 →
  train_length speed_km_hr time_sec = expected_length :=
by
  intros h1 h2 h3
  simp [h1, h2, train_length]
  sorry

end NUMINAMATH_GPT_length_of_train_is_correct_l2423_242384


namespace NUMINAMATH_GPT_correct_diagram_l2423_242393

-- Definitions based on the conditions
def word : String := "KANGAROO"
def diagrams : List (String × Bool) :=
  [("Diagram A", False), ("Diagram B", False), ("Diagram C", False),
   ("Diagram D", False), ("Diagram E", True)]

-- Statement to prove that Diagram E correctly shows "KANGAROO"
theorem correct_diagram :
  ∃ d, (d.1 = "Diagram E") ∧ d.2 = True ∧ d ∈ diagrams :=
by
-- skipping the proof for now
sorry

end NUMINAMATH_GPT_correct_diagram_l2423_242393


namespace NUMINAMATH_GPT_area_of_ABCD_l2423_242303

theorem area_of_ABCD 
  (AB CD DA: ℝ) (angle_CDA: ℝ) (a b c: ℕ) 
  (H1: AB = 10) 
  (H2: BC = 6) 
  (H3: CD = 13) 
  (H4: DA = 13) 
  (H5: angle_CDA = 45) 
  (H_area: a = 8 ∧ b = 30 ∧ c = 2) :

  ∃ (a b c : ℝ), a + b + c = 40 := 
by
  sorry

end NUMINAMATH_GPT_area_of_ABCD_l2423_242303


namespace NUMINAMATH_GPT_expressions_equal_iff_sum_zero_l2423_242309

theorem expressions_equal_iff_sum_zero (p q r : ℝ) : (p + qr = (p + q) * (p + r)) ↔ (p + q + r = 0) :=
sorry

end NUMINAMATH_GPT_expressions_equal_iff_sum_zero_l2423_242309


namespace NUMINAMATH_GPT_cassie_nails_claws_total_l2423_242386

theorem cassie_nails_claws_total :
  let dogs := 4
  let parrots := 8
  let cats := 2
  let rabbits := 6
  let lizards := 5
  let tortoises := 3

  let dog_nails := dogs * 4 * 4

  let normal_parrots := 6
  let parrot_with_extra_toe := 1
  let parrot_missing_toe := 1
  let parrot_claws := (normal_parrots * 2 * 3) + (parrot_with_extra_toe * 2 * 4) + (parrot_missing_toe * 2 * 2)

  let normal_cats := 1
  let deformed_cat := 1
  let cat_toes := (1 * 4 * 5) + (1 * 4 * 4) + 1 

  let normal_rabbits := 5
  let deformed_rabbit := 1
  let rabbit_nails := (normal_rabbits * 4 * 9) + (3 * 9 + 2)

  let normal_lizards := 4
  let deformed_lizard := 1
  let lizard_toes := (normal_lizards * 4 * 5) + (deformed_lizard * 4 * 4)
  
  let normal_tortoises := 1
  let tortoise_with_extra_claw := 1
  let tortoise_missing_claw := 1
  let tortoise_claws := (normal_tortoises * 4 * 4) + (3 * 4 + 5) + (3 * 4 + 3)

  let total_nails_claws := dog_nails + parrot_claws + cat_toes + rabbit_nails + lizard_toes + tortoise_claws

  total_nails_claws = 524 :=
by
  sorry

end NUMINAMATH_GPT_cassie_nails_claws_total_l2423_242386


namespace NUMINAMATH_GPT_find_d_minus_r_l2423_242339

theorem find_d_minus_r :
  ∃ d r : ℕ, d > 1 ∧ (1059 % d = r) ∧ (1417 % d = r) ∧ (2312 % d = r) ∧ (d - r = 15) :=
sorry

end NUMINAMATH_GPT_find_d_minus_r_l2423_242339


namespace NUMINAMATH_GPT_ages_sum_l2423_242338

theorem ages_sum (a b c : ℕ) (h1 : a = b) (h2 : a * b * c = 72) : a + b + c = 14 :=
by sorry

end NUMINAMATH_GPT_ages_sum_l2423_242338


namespace NUMINAMATH_GPT_nancy_packs_of_crayons_l2423_242333

def total_crayons : ℕ := 615
def crayons_per_pack : ℕ := 15

theorem nancy_packs_of_crayons : total_crayons / crayons_per_pack = 41 :=
by
  sorry

end NUMINAMATH_GPT_nancy_packs_of_crayons_l2423_242333


namespace NUMINAMATH_GPT_union_of_sets_l2423_242374

def A : Set ℝ := { x : ℝ | x * (x + 1) ≤ 0 }
def B : Set ℝ := { x : ℝ | -1 < x ∧ x < 1 }
def C : Set ℝ := { x : ℝ | -1 ≤ x ∧ x < 1 }

theorem union_of_sets :
  A ∪ B = C := 
sorry

end NUMINAMATH_GPT_union_of_sets_l2423_242374


namespace NUMINAMATH_GPT_carlos_payment_l2423_242359

theorem carlos_payment (A B C : ℝ) (hB_lt_A : B < A) (hB_lt_C : B < C) :
    B + (0.35 * (A + B + C) - B) = 0.35 * A - 0.65 * B + 0.35 * C :=
by sorry

end NUMINAMATH_GPT_carlos_payment_l2423_242359


namespace NUMINAMATH_GPT_bonus_distributed_correctly_l2423_242317

def amount_received (A B C D E F : ℝ) :=
  -- Conditions
  (A = 2 * B) ∧ 
  (B = C) ∧ 
  (D = 2 * B - 1500) ∧ 
  (E = C + 2000) ∧ 
  (F = 1/2 * (A + D)) ∧ 
  -- Total bonus amount
  (A + B + C + D + E + F = 25000)

theorem bonus_distributed_correctly :
  ∃ (A B C D E F : ℝ), 
    amount_received A B C D E F ∧ 
    A = 4950 ∧ 
    B = 2475 ∧ 
    C = 2475 ∧ 
    D = 3450 ∧ 
    E = 4475 ∧ 
    F = 4200 :=
sorry

end NUMINAMATH_GPT_bonus_distributed_correctly_l2423_242317


namespace NUMINAMATH_GPT_books_remaining_correct_l2423_242381

-- Define the total number of books and the number of books read
def total_books : ℕ := 32
def books_read : ℕ := 17

-- Define the number of books remaining to be read
def books_remaining : ℕ := total_books - books_read

-- Prove that the number of books remaining to be read is 15
theorem books_remaining_correct : books_remaining = 15 := by
  sorry

end NUMINAMATH_GPT_books_remaining_correct_l2423_242381


namespace NUMINAMATH_GPT_consecutive_integer_sum_l2423_242337

noncomputable def sqrt17 : ℝ := Real.sqrt 17

theorem consecutive_integer_sum : ∃ (a b : ℤ), (b = a + 1) ∧ (a < sqrt17 ∧ sqrt17 < b) ∧ (a + b = 9) :=
by
  sorry

end NUMINAMATH_GPT_consecutive_integer_sum_l2423_242337


namespace NUMINAMATH_GPT_bernie_postcards_l2423_242344

theorem bernie_postcards :
  let initial_postcards := 18
  let price_sell := 15
  let price_buy := 5
  let sold_postcards := initial_postcards / 2
  let earned_money := sold_postcards * price_sell
  let bought_postcards := earned_money / price_buy
  let remaining_postcards := initial_postcards - sold_postcards
  let final_postcards := remaining_postcards + bought_postcards
  final_postcards = 36 := by sorry

end NUMINAMATH_GPT_bernie_postcards_l2423_242344


namespace NUMINAMATH_GPT_match_end_time_is_17_55_l2423_242349

-- Definitions corresponding to conditions
def start_time : ℕ := 15 * 60 + 30  -- Convert 15:30 to minutes past midnight
def duration : ℕ := 145  -- Duration in minutes

-- Definition corresponding to the question
def end_time : ℕ := start_time + duration 

-- Assertion corresponding to the correct answer
theorem match_end_time_is_17_55 : end_time = 17 * 60 + 55 :=
by
  -- Proof steps and actual proof will go here
  sorry

end NUMINAMATH_GPT_match_end_time_is_17_55_l2423_242349


namespace NUMINAMATH_GPT_dentist_age_l2423_242340

theorem dentist_age (x : ℝ) (h : (x - 8) / 6 = (x + 8) / 10) : x = 32 :=
  by
  sorry

end NUMINAMATH_GPT_dentist_age_l2423_242340


namespace NUMINAMATH_GPT_solve_inequality_system_l2423_242351

theorem solve_inequality_system (x : ℝ) :
  (x + 2 > -1) ∧ (x - 5 < 3 * (x - 1)) ↔ (x > -1) :=
by
  sorry

end NUMINAMATH_GPT_solve_inequality_system_l2423_242351


namespace NUMINAMATH_GPT_find_r_l2423_242342

theorem find_r (r : ℝ) (h : 5 * (r - 9) = 6 * (3 - 3 * r) + 6) : r = 3 :=
by
  sorry

end NUMINAMATH_GPT_find_r_l2423_242342


namespace NUMINAMATH_GPT_sum_of_consecutive_pairs_eq_pow_two_l2423_242361

theorem sum_of_consecutive_pairs_eq_pow_two (n m : ℕ) :
  ∃ n m : ℕ, (n * (n + 1) + m * (m + 1) = 2 ^ 2021) :=
sorry

end NUMINAMATH_GPT_sum_of_consecutive_pairs_eq_pow_two_l2423_242361


namespace NUMINAMATH_GPT_solve_and_sum_solutions_l2423_242330

theorem solve_and_sum_solutions :
  (∀ x : ℝ, x^2 > 9 → 
  (∃ S : ℝ, (∀ x : ℝ, (x / 3 + x / Real.sqrt (x^2 - 9) = 35 / 12) → S = 8.75))) :=
sorry

end NUMINAMATH_GPT_solve_and_sum_solutions_l2423_242330


namespace NUMINAMATH_GPT_exponent_neg_power_l2423_242399

theorem exponent_neg_power (a : ℝ) : -(a^3)^4 = -a^(3 * 4) := 
by
  sorry

end NUMINAMATH_GPT_exponent_neg_power_l2423_242399


namespace NUMINAMATH_GPT_solve_for_x_l2423_242343

theorem solve_for_x (x : ℤ) (h : 158 - x = 59) : x = 99 :=
by
  sorry

end NUMINAMATH_GPT_solve_for_x_l2423_242343


namespace NUMINAMATH_GPT_power_mod_l2423_242346

theorem power_mod (a : ℕ) : 5 ^ 2023 % 17 = 2 := by
  sorry

end NUMINAMATH_GPT_power_mod_l2423_242346


namespace NUMINAMATH_GPT_exponential_comparison_l2423_242319

theorem exponential_comparison (a b c : ℝ) (h₁ : a = 0.5^((1:ℝ)/2))
                                          (h₂ : b = 0.5^((1:ℝ)/3))
                                          (h₃ : c = 0.5^((1:ℝ)/4)) : 
  a < b ∧ b < c := by
  sorry

end NUMINAMATH_GPT_exponential_comparison_l2423_242319


namespace NUMINAMATH_GPT_range_of_m_l2423_242328

theorem range_of_m (m x : ℝ) : (m-1 < x ∧ x < m+1) → (1/3 < x ∧ x < 1/2) → (-1/2 ≤ m ∧ m ≤ 4/3) :=
by
  intros h1 h2
  have h3 : 1/3 < m + 1 := by sorry
  have h4 : m - 1 < 1/2 := by sorry
  have h5 : -1/2 ≤ m := by sorry
  have h6 : m ≤ 4/3 := by sorry
  exact ⟨h5, h6⟩

end NUMINAMATH_GPT_range_of_m_l2423_242328


namespace NUMINAMATH_GPT_greg_sarah_apples_l2423_242318

-- Definitions and Conditions
variable {G : ℕ}
variable (H0 : 2 * G + 2 * G + (2 * G - 5) = 49)

-- Statement of the problem
theorem greg_sarah_apples : 
  2 * G = 18 :=
by
  sorry

end NUMINAMATH_GPT_greg_sarah_apples_l2423_242318


namespace NUMINAMATH_GPT_isabella_total_haircut_length_l2423_242389

theorem isabella_total_haircut_length :
  (18 - 14) + (14 - 9) = 9 := 
sorry

end NUMINAMATH_GPT_isabella_total_haircut_length_l2423_242389


namespace NUMINAMATH_GPT_notebook_cost_l2423_242370

open Nat

theorem notebook_cost
  (s : ℕ) (c : ℕ) (n : ℕ)
  (h_majority : s > 21)
  (h_notebooks : n > 2)
  (h_cost : c > n)
  (h_total : s * c * n = 2773) : c = 103 := 
sorry

end NUMINAMATH_GPT_notebook_cost_l2423_242370


namespace NUMINAMATH_GPT_total_cost_for_trip_l2423_242341

def cost_of_trip (students : ℕ) (teachers : ℕ) (seats_per_bus : ℕ) (cost_per_bus : ℕ) (toll_per_bus : ℕ) : ℕ :=
  let total_people := students + teachers
  let buses_required := (total_people + seats_per_bus - 1) / seats_per_bus -- ceiling division
  let total_rent_cost := buses_required * cost_per_bus
  let total_toll_cost := buses_required * toll_per_bus
  total_rent_cost + total_toll_cost

theorem total_cost_for_trip
  (students : ℕ := 252)
  (teachers : ℕ := 8)
  (seats_per_bus : ℕ := 41)
  (cost_per_bus : ℕ := 300000)
  (toll_per_bus : ℕ := 7500) :
  cost_of_trip students teachers seats_per_bus cost_per_bus toll_per_bus = 2152500 := by
  sorry -- Proof to be filled

end NUMINAMATH_GPT_total_cost_for_trip_l2423_242341


namespace NUMINAMATH_GPT_geom_series_sum_n_eq_728_div_729_l2423_242300

noncomputable def a : ℚ := 1 / 3
noncomputable def r : ℚ := 1 / 3
noncomputable def S_n (n : ℕ) : ℚ := a * ((1 - r^n) / (1 - r))

theorem geom_series_sum_n_eq_728_div_729 (n : ℕ) (h : S_n n = 728 / 729) : n = 6 :=
by
  sorry

end NUMINAMATH_GPT_geom_series_sum_n_eq_728_div_729_l2423_242300


namespace NUMINAMATH_GPT_day_100_days_from_friday_l2423_242310

-- Define the days of the week
inductive Day : Type
| Sunday | Monday | Tuesday | Wednesday | Thursday | Friday | Saturday

open Day

-- Define a function to get the day of the week after a given number of days
def dayOfWeekAfter (start : Day) (n : ℕ) : Day :=
  match start with
  | Sunday    => match n % 7 with
                  | 0 => Sunday
                  | 1 => Monday
                  | 2 => Tuesday
                  | 3 => Wednesday
                  | 4 => Thursday
                  | 5 => Friday
                  | 6 => Saturday
                  | _ => start
  | Monday    => match n % 7 with
                  | 0 => Monday
                  | 1 => Tuesday
                  | 2 => Wednesday
                  | 3 => Thursday
                  | 4 => Friday
                  | 5 => Saturday
                  | 6 => Sunday
                  | _ => start
  | Tuesday   => match n % 7 with
                  | 0 => Tuesday
                  | 1 => Wednesday
                  | 2 => Thursday
                  | 3 => Friday
                  | 4 => Saturday
                  | 5 => Sunday
                  | 6 => Monday
                  | _ => start
  | Wednesday => match n % 7 with
                  | 0 => Wednesday
                  | 1 => Thursday
                  | 2 => Friday
                  | 3 => Saturday
                  | 4 => Sunday
                  | 5 => Monday
                  | 6 => Tuesday
                  | _ => start
  | Thursday  => match n % 7 with
                  | 0 => Thursday
                  | 1 => Friday
                  | 2 => Saturday
                  | 3 => Sunday
                  | 4 => Monday
                  | 5 => Tuesday
                  | 6 => Wednesday
                  | _ => start
  | Friday    => match n % 7 with
                  | 0 => Friday
                  | 1 => Saturday
                  | 2 => Sunday
                  | 3 => Monday
                  | 4 => Tuesday
                  | 5 => Wednesday
                  | 6 => Thursday
                  | _ => start
  | Saturday  => match n % 7 with
                  | 0 => Saturday
                  | 1 => Sunday
                  | 2 => Monday
                  | 3 => Tuesday
                  | 4 => Wednesday
                  | 5 => Thursday
                  | 6 => Friday
                  | _ => start

-- The proof problem as a Lean theorem
theorem day_100_days_from_friday : dayOfWeekAfter Friday 100 = Sunday := by
  -- Proof will go here
  sorry

end NUMINAMATH_GPT_day_100_days_from_friday_l2423_242310


namespace NUMINAMATH_GPT_calc_fraction_power_l2423_242353

theorem calc_fraction_power (n m : ℤ) (h_n : n = 2023) (h_m : m = 2022) :
  (- (2 / 3 : ℚ))^n * ((3 / 2 : ℚ))^m = - (2 / 3) := by
  sorry

end NUMINAMATH_GPT_calc_fraction_power_l2423_242353


namespace NUMINAMATH_GPT_compare_a_b_c_l2423_242391

def a : ℝ := 2^(1/2)
def b : ℝ := 3^(1/3)
def c : ℝ := 5^(1/5)

theorem compare_a_b_c : b > a ∧ a > c :=
  by
  sorry

end NUMINAMATH_GPT_compare_a_b_c_l2423_242391


namespace NUMINAMATH_GPT_gcd_lcm_sum_l2423_242312

theorem gcd_lcm_sum : Nat.gcd 24 54 + Nat.lcm 40 10 = 46 := by
  sorry

end NUMINAMATH_GPT_gcd_lcm_sum_l2423_242312


namespace NUMINAMATH_GPT_alpha_in_second_quadrant_l2423_242390

variable (α : ℝ)

-- Conditions that P(tan α, cos α) is in the third quadrant
def P_in_third_quadrant (α : ℝ) : Prop := (Real.tan α < 0) ∧ (Real.cos α < 0)

-- Theorem statement
theorem alpha_in_second_quadrant (h : P_in_third_quadrant α) : 
  π/2 < α ∧ α < π :=
sorry

end NUMINAMATH_GPT_alpha_in_second_quadrant_l2423_242390


namespace NUMINAMATH_GPT_average_grade_of_females_is_92_l2423_242302

theorem average_grade_of_females_is_92 (F : ℝ) : 
  (∀ (overall_avg male_avg : ℝ) (num_male num_female : ℕ), 
    overall_avg = 90 ∧ male_avg = 82 ∧ num_male = 8 ∧ num_female = 32 → 
    overall_avg = (num_male * male_avg + num_female * F) / (num_male + num_female) → F = 92) :=
sorry

end NUMINAMATH_GPT_average_grade_of_females_is_92_l2423_242302


namespace NUMINAMATH_GPT_total_animals_l2423_242358

theorem total_animals (B : ℕ) (h1 : 4 * B + 8 = 44) : B + 4 = 13 := by
  sorry

end NUMINAMATH_GPT_total_animals_l2423_242358


namespace NUMINAMATH_GPT_collinear_points_l2423_242352

theorem collinear_points (k : ℝ) :
  let p1 := (3, 1)
  let p2 := (6, 4)
  let p3 := (10, k + 9)
  let slope (a b : ℝ × ℝ) : ℝ := (b.snd - a.snd) / (b.fst - a.fst)
  slope p1 p2 = slope p1 p3 → k = -1 :=
by 
  let p1 := (3, 1)
  let p2 := (6, 4)
  let p3 := (10, k + 9)
  let slope (a b : ℝ × ℝ) : ℝ := (b.snd - a.snd) / (b.fst - a.fst)
  sorry

end NUMINAMATH_GPT_collinear_points_l2423_242352


namespace NUMINAMATH_GPT_student_in_eighth_group_l2423_242332

-- Defining the problem: total students and their assignment into groups
def total_students : ℕ := 50
def students_assigned_numbers (n : ℕ) : Prop := n > 0 ∧ n ≤ total_students

-- Grouping students: Each group has 5 students
def grouped_students (group_num student_num : ℕ) : Prop := 
  student_num > (group_num - 1) * 5 ∧ student_num ≤ group_num * 5

-- Condition: Student 12 is selected from the third group
def condition : Prop := grouped_students 3 12

-- Goal: the number of the student selected from the eighth group is 37
theorem student_in_eighth_group : condition → grouped_students 8 37 :=
by
  sorry

end NUMINAMATH_GPT_student_in_eighth_group_l2423_242332


namespace NUMINAMATH_GPT_possible_values_of_quadratic_l2423_242334

theorem possible_values_of_quadratic (x : ℝ) (h : x^2 - 5 * x + 4 < 0) : 10 < x^2 + 4 * x + 5 ∧ x^2 + 4 * x + 5 < 37 :=
by
  sorry

end NUMINAMATH_GPT_possible_values_of_quadratic_l2423_242334


namespace NUMINAMATH_GPT_residue_neg_1234_mod_31_l2423_242311

theorem residue_neg_1234_mod_31 : -1234 % 31 = 6 := 
by sorry

end NUMINAMATH_GPT_residue_neg_1234_mod_31_l2423_242311


namespace NUMINAMATH_GPT_house_number_digits_cost_l2423_242322

/-
The constants represent:
- cost_1: the cost of 1 unit (1000 rubles)
- cost_12: the cost of 12 units (2000 rubles)
- cost_512: the cost of 512 units (3000 rubles)
- P: the cost per digit of a house number (1000 rubles)
- n: the number of digits in a house number
- The goal is to prove that the cost for 1, 12, and 512 units follows the pattern described
-/

theorem house_number_digits_cost :
  ∃ (P : ℕ),
    (P = 1000) ∧
    (∃ (cost_1 cost_12 cost_512 : ℕ),
      cost_1 = 1000 ∧
      cost_12 = 2000 ∧
      cost_512 = 3000 ∧
      (∃ n1 n2 n3 : ℕ,
        n1 = 1 ∧
        n2 = 2 ∧
        n3 = 3 ∧
        cost_1 = P * n1 ∧
        cost_12 = P * n2 ∧
        cost_512 = P * n3)) :=
by
  sorry

end NUMINAMATH_GPT_house_number_digits_cost_l2423_242322


namespace NUMINAMATH_GPT_num_non_congruent_triangles_with_perimeter_12_l2423_242376

noncomputable def count_non_congruent_triangles_with_perimeter_12 : ℕ :=
  sorry -- This is where the actual proof or computation would go.

theorem num_non_congruent_triangles_with_perimeter_12 :
  count_non_congruent_triangles_with_perimeter_12 = 3 :=
  sorry -- This is the theorem stating the result we want to prove.

end NUMINAMATH_GPT_num_non_congruent_triangles_with_perimeter_12_l2423_242376


namespace NUMINAMATH_GPT_complement_union_result_l2423_242326

open Set

variable (U : Set ℕ)
variable (A : Set ℕ)
variable (B : Set ℕ)

theorem complement_union_result :
    U = { x | x < 6 } →
    A = {1, 2, 3} → 
    B = {2, 4, 5} → 
    (U \ A) ∪ (U \ B) = {0, 1, 3, 4, 5} :=
by
    intros hU hA hB
    sorry

end NUMINAMATH_GPT_complement_union_result_l2423_242326


namespace NUMINAMATH_GPT_Petya_can_verify_coins_l2423_242380

theorem Petya_can_verify_coins :
  ∃ (c₁ c₂ c₃ c₅ : ℕ), 
  (c₁ = 1 ∧ c₂ = 2 ∧ c₃ = 3 ∧ c₅ = 5) ∧
  (∃ (w : ℕ), w = 9) ∧
  (∃ (cond : ℕ → Prop), 
    cond 1 ∧ cond 2 ∧ cond 3 ∧ cond 5) := sorry

end NUMINAMATH_GPT_Petya_can_verify_coins_l2423_242380


namespace NUMINAMATH_GPT_find_angle_A_correct_l2423_242367

noncomputable def find_angle_A (BC AB angleC : ℝ) : ℝ :=
if BC = 3 ∧ AB = Real.sqrt 6 ∧ angleC = Real.pi / 4 then
  Real.pi / 3
else
  sorry

theorem find_angle_A_correct : find_angle_A 3 (Real.sqrt 6) (Real.pi / 4) = Real.pi / 3 :=
by
  -- proof goes here
  sorry

end NUMINAMATH_GPT_find_angle_A_correct_l2423_242367


namespace NUMINAMATH_GPT_necessary_condition_x_pow_2_minus_x_lt_0_l2423_242357

theorem necessary_condition_x_pow_2_minus_x_lt_0 (x : ℝ) : (x^2 - x < 0) → (-1 < x ∧ x < 1) := by
  intro hx
  sorry

end NUMINAMATH_GPT_necessary_condition_x_pow_2_minus_x_lt_0_l2423_242357


namespace NUMINAMATH_GPT_new_trailers_added_l2423_242301

theorem new_trailers_added (n : ℕ) :
  let original_trailers := 15
  let original_age := 12
  let years_passed := 3
  let current_total_trailers := original_trailers + n
  let current_average_age := 10
  let total_age_three_years_ago := original_trailers * original_age
  let new_trailers_age := 3
  let total_current_age := (original_trailers * (original_age + years_passed)) + (n * new_trailers_age)
  (total_current_age / current_total_trailers = current_average_age) ↔ (n = 10) :=
by
  sorry

end NUMINAMATH_GPT_new_trailers_added_l2423_242301


namespace NUMINAMATH_GPT_longest_side_eq_24_l2423_242313

noncomputable def x : Real := 19 / 3

def side1 (x : Real) : Real := x + 3
def side2 (x : Real) : Real := 2 * x - 1
def side3 (x : Real) : Real := 3 * x + 5

def perimeter (x : Real) : Prop :=
  side1 x + side2 x + side3 x = 45

theorem longest_side_eq_24 : perimeter x → max (max (side1 x) (side2 x)) (side3 x) = 24 :=
by
  sorry

end NUMINAMATH_GPT_longest_side_eq_24_l2423_242313


namespace NUMINAMATH_GPT_ratio_w_y_l2423_242336

theorem ratio_w_y 
  (w x y z : ℚ) 
  (h1 : w / x = 4 / 3)
  (h2 : y / z = 3 / 2)
  (h3 : z / x = 1 / 6) : 
  w / y = 16 / 3 :=
sorry

end NUMINAMATH_GPT_ratio_w_y_l2423_242336


namespace NUMINAMATH_GPT_find_b_l2423_242375

theorem find_b (A B C : ℝ) (a b c : ℝ)
  (h1 : Real.tan A = 1 / 3)
  (h2 : Real.tan B = 1 / 2)
  (h3 : a = 1)
  (h4 : A + B + C = π) -- This condition is added because angles in a triangle sum up to π.
  : b = Real.sqrt 2 :=
by
  sorry

end NUMINAMATH_GPT_find_b_l2423_242375


namespace NUMINAMATH_GPT_min_additional_cells_l2423_242348

-- Definitions based on conditions
def num_cells_shape : Nat := 32
def side_length_square : Nat := 9
def area_square : Nat := side_length_square * side_length_square

-- The statement to prove
theorem min_additional_cells (num_cells_given : Nat := num_cells_shape) 
(side_length : Nat := side_length_square)
(area : Nat := area_square) :
  area - num_cells_given = 49 :=
by
  sorry

end NUMINAMATH_GPT_min_additional_cells_l2423_242348


namespace NUMINAMATH_GPT_population_growth_l2423_242355

theorem population_growth 
  (P₀ : ℝ) (P₂ : ℝ) (r : ℝ)
  (hP₀ : P₀ = 15540) 
  (hP₂ : P₂ = 25460.736)
  (h_growth : P₂ = P₀ * (1 + r)^2) :
  r = 0.28 :=
by 
  sorry

end NUMINAMATH_GPT_population_growth_l2423_242355


namespace NUMINAMATH_GPT_maximize_savings_l2423_242397

-- Definitions for the conditions
def initial_amount : ℝ := 15000

def discount_option1 (amount : ℝ) : ℝ := 
  let after_first : ℝ := amount * 0.75
  let after_second : ℝ := after_first * 0.90
  after_second * 0.95

def discount_option2 (amount : ℝ) : ℝ := 
  let after_first : ℝ := amount * 0.70
  let after_second : ℝ := after_first * 0.90
  after_second * 0.90

-- Theorem to compare the final amounts
theorem maximize_savings : discount_option2 initial_amount < discount_option1 initial_amount := 
  sorry

end NUMINAMATH_GPT_maximize_savings_l2423_242397


namespace NUMINAMATH_GPT_max_value_of_expr_l2423_242383

noncomputable def max_expr_value (x : ℝ) : ℝ :=
  x^6 / (x^12 + 4*x^9 - 6*x^6 + 16*x^3 + 64)

theorem max_value_of_expr : ∀ x : ℝ, max_expr_value x ≤ 1/26 :=
by
  sorry

end NUMINAMATH_GPT_max_value_of_expr_l2423_242383


namespace NUMINAMATH_GPT_simplify_expression_l2423_242345

theorem simplify_expression (x y : ℝ) :
  3 * x + 6 * x + 9 * x + 12 * y + 15 * y + 18 + 21 = 18 * x + 27 * y + 39 :=
by
  sorry

end NUMINAMATH_GPT_simplify_expression_l2423_242345


namespace NUMINAMATH_GPT_deepak_share_l2423_242394

theorem deepak_share (investment_Anand investment_Deepak total_profit : ℕ)
  (h₁ : investment_Anand = 2250) (h₂ : investment_Deepak = 3200) (h₃ : total_profit = 1380) :
  ∃ share_Deepak, share_Deepak = 810 := sorry

end NUMINAMATH_GPT_deepak_share_l2423_242394


namespace NUMINAMATH_GPT_abc_value_l2423_242377

noncomputable def find_abc (a b c : ℝ) (h1 : a * (b + c) = 152) (h2 : b * (c + a) = 162) (h3 : c * (a + b) = 170) : ℝ :=
  a * b * c

theorem abc_value (a b c : ℝ) (h1 : a * (b + c) = 152) (h2 : b * (c + a) = 162) (h3 : c * (a + b) = 170) :
  a * b * c = 720 :=
by
  -- We skip the proof by providing sorry.
  sorry

end NUMINAMATH_GPT_abc_value_l2423_242377


namespace NUMINAMATH_GPT_distinguishable_squares_count_l2423_242347

theorem distinguishable_squares_count :
  let colors := 5  -- Number of different colors
  let total_corner_sets :=
    5 + -- All four corners the same color
    5 * 4 + -- Three corners the same color
    Nat.choose 5 2 * 2 + -- Two pairs of corners with the same color
    5 * 4 * 3 * 2 -- All four corners different
  let total_corner_together := total_corner_sets
  let total := 
    (4 * 5 + -- One corner color used
    3 * (5 * 4 + Nat.choose 5 2 * 2) + -- Two corner colors used
    2 * (5 * 4 * 3 * 2) + -- Three corner colors used
    1 * (5 * 4 * 3 * 2)) -- Four corner colors used
  total_corner_together * colors / 10
= 540 :=
by
  sorry

end NUMINAMATH_GPT_distinguishable_squares_count_l2423_242347


namespace NUMINAMATH_GPT_unique_solution_l2423_242354

theorem unique_solution :
  ∀ a b c : ℕ,
    a > 0 → b > 0 → c > 0 →
    (3 * a * b * c + 11 * (a + b + c) = 6 * (a * b + b * c + c * a) + 18) →
    (a = 1 ∧ b = 2 ∧ c = 3) :=
by
  intros a b c ha hb hc h
  have h1 : a = 1 := sorry
  have h2 : b = 2 := sorry
  have h3 : c = 3 := sorry
  exact ⟨h1, h2, h3⟩

end NUMINAMATH_GPT_unique_solution_l2423_242354


namespace NUMINAMATH_GPT_total_wait_time_l2423_242372

def customs_wait : ℕ := 20
def quarantine_days : ℕ := 14
def hours_per_day : ℕ := 24

theorem total_wait_time :
  customs_wait + quarantine_days * hours_per_day = 356 := 
by
  sorry

end NUMINAMATH_GPT_total_wait_time_l2423_242372


namespace NUMINAMATH_GPT_competition_order_l2423_242392

variable (A B C D : ℕ)

-- Conditions as given in the problem
axiom cond1 : B + D = 2 * A
axiom cond2 : A + C < B + D
axiom cond3 : A < B + C

-- The desired proof statement
theorem competition_order : D > B ∧ B > A ∧ A > C :=
by
  sorry

end NUMINAMATH_GPT_competition_order_l2423_242392


namespace NUMINAMATH_GPT_a_100_value_l2423_242320

variables (S : ℕ → ℚ) (a : ℕ → ℚ)

def S_n (n : ℕ) : ℚ := S n
def a_n (n : ℕ) : ℚ := a n

axiom a1_eq_3 : a 1 = 3
axiom a_n_formula (n : ℕ) (hn : n ≥ 2) : a n = (3 * S n ^ 2) / (3 * S n - 2)

theorem a_100_value : a 100 = -3 / 88401 :=
sorry

end NUMINAMATH_GPT_a_100_value_l2423_242320


namespace NUMINAMATH_GPT_greene_family_amusement_park_spending_l2423_242362

def spent_on_admission : ℝ := 45
def original_ticket_cost : ℝ := 50
def spent_less_than_original_cost_on_food_and_beverages : ℝ := 13
def spent_on_souvenir_Mr_Greene : ℝ := 15
def spent_on_souvenir_Mrs_Greene : ℝ := 2 * spent_on_souvenir_Mr_Greene
def cost_per_game : ℝ := 9
def number_of_children : ℝ := 3
def spent_on_transportation : ℝ := 25
def tax_rate : ℝ := 0.08

def food_and_beverages_cost : ℝ := original_ticket_cost - spent_less_than_original_cost_on_food_and_beverages
def games_cost : ℝ := number_of_children * cost_per_game
def taxable_amount : ℝ := food_and_beverages_cost + spent_on_souvenir_Mr_Greene + spent_on_souvenir_Mrs_Greene + games_cost
def tax : ℝ := tax_rate * taxable_amount
def total_expenditure : ℝ := spent_on_admission + food_and_beverages_cost + spent_on_souvenir_Mr_Greene + spent_on_souvenir_Mrs_Greene + games_cost + spent_on_transportation + tax

theorem greene_family_amusement_park_spending : total_expenditure = 187.72 :=
by {
  sorry
}

end NUMINAMATH_GPT_greene_family_amusement_park_spending_l2423_242362


namespace NUMINAMATH_GPT_sachin_younger_than_rahul_l2423_242371

theorem sachin_younger_than_rahul :
  ∀ (sachin_age rahul_age : ℕ),
  (sachin_age / rahul_age = 6 / 9) →
  (sachin_age = 14) →
  (rahul_age - sachin_age = 7) :=
by
  sorry

end NUMINAMATH_GPT_sachin_younger_than_rahul_l2423_242371


namespace NUMINAMATH_GPT_chad_savings_correct_l2423_242379

variable (earnings_mowing : ℝ := 600)
variable (earnings_birthday : ℝ := 250)
variable (earnings_video_games : ℝ := 150)
variable (earnings_odd_jobs : ℝ := 150)
variable (tax_rate : ℝ := 0.10)

noncomputable def total_earnings : ℝ := 
  earnings_mowing + earnings_birthday + earnings_video_games + earnings_odd_jobs

noncomputable def taxes : ℝ := 
  tax_rate * total_earnings

noncomputable def money_after_taxes : ℝ := 
  total_earnings - taxes

noncomputable def savings_mowing : ℝ := 
  0.50 * earnings_mowing

noncomputable def savings_birthday : ℝ := 
  0.30 * earnings_birthday

noncomputable def savings_video_games : ℝ := 
  0.40 * earnings_video_games

noncomputable def savings_odd_jobs : ℝ := 
  0.20 * earnings_odd_jobs

noncomputable def total_savings : ℝ := 
  savings_mowing + savings_birthday + savings_video_games + savings_odd_jobs

theorem chad_savings_correct : total_savings = 465 := by
  sorry

end NUMINAMATH_GPT_chad_savings_correct_l2423_242379


namespace NUMINAMATH_GPT_simplify_and_evaluate_expr_l2423_242305

noncomputable def original_expr (x : ℝ) : ℝ := 
  ((x / (x - 1)) - (x / (x^2 - 1))) / ((x^2 - x) / (x^2 - 2*x + 1))

noncomputable def x_val : ℝ := Real.sqrt 2 - 1

theorem simplify_and_evaluate_expr : original_expr x_val = 1 - (Real.sqrt 2) / 2 :=
  by
    sorry

end NUMINAMATH_GPT_simplify_and_evaluate_expr_l2423_242305


namespace NUMINAMATH_GPT_vincent_spent_224_l2423_242363

-- Defining the given conditions as constants
def num_books_animal : ℕ := 10
def num_books_outer_space : ℕ := 1
def num_books_trains : ℕ := 3
def cost_per_book : ℕ := 16

-- Summarizing the total number of books
def total_books : ℕ := num_books_animal + num_books_outer_space + num_books_trains
-- Calculating the total cost
def total_cost : ℕ := total_books * cost_per_book

-- Lean statement to prove that Vincent spent $224
theorem vincent_spent_224 : total_cost = 224 := by
  sorry

end NUMINAMATH_GPT_vincent_spent_224_l2423_242363


namespace NUMINAMATH_GPT_clark_paid_correct_amount_l2423_242307

-- Definitions based on the conditions
def cost_per_part : ℕ := 80
def number_of_parts : ℕ := 7
def total_discount : ℕ := 121

-- Given conditions
def total_cost_without_discount : ℕ := cost_per_part * number_of_parts
def expected_total_cost_after_discount : ℕ := 439

-- Theorem to prove the amount Clark paid after the discount is correct
theorem clark_paid_correct_amount : total_cost_without_discount - total_discount = expected_total_cost_after_discount := by
  sorry

end NUMINAMATH_GPT_clark_paid_correct_amount_l2423_242307


namespace NUMINAMATH_GPT_find_k_l2423_242321

theorem find_k (m n k : ℤ) (h1 : m = 2 * n + 5) (h2 : m + 2 = 2 * (n + k) + 5) : k = 0 := by
  sorry

end NUMINAMATH_GPT_find_k_l2423_242321


namespace NUMINAMATH_GPT_red_marbles_more_than_yellow_l2423_242387

-- Define the conditions
def total_marbles : ℕ := 19
def yellow_marbles : ℕ := 5
def ratio_parts_blue : ℕ := 3
def ratio_parts_red : ℕ := 4

-- Prove the number of more red marbles than yellow marbles is equal to 3
theorem red_marbles_more_than_yellow :
  let remainder_marbles := total_marbles - yellow_marbles
  let total_parts := ratio_parts_blue + ratio_parts_red
  let marbles_per_part := remainder_marbles / total_parts
  let blue_marbles := ratio_parts_blue * marbles_per_part
  let red_marbles := ratio_parts_red * marbles_per_part
  red_marbles - yellow_marbles = 3 := by
  sorry

end NUMINAMATH_GPT_red_marbles_more_than_yellow_l2423_242387


namespace NUMINAMATH_GPT_possible_values_of_a_l2423_242324

def line1 (x y : ℝ) := x + y + 1 = 0
def line2 (x y : ℝ) := 2 * x - y + 8 = 0
def line3 (a : ℝ) (x y : ℝ) := a * x + 3 * y - 5 = 0

theorem possible_values_of_a :
  {a : ℝ | ∃ (x y : ℝ), line1 x y ∧ line2 x y ∧ line3 a x y} ⊆ {1/3, 3, -6} ∧
  {1/3, 3, -6} ⊆ {a : ℝ | ∃ (x y : ℝ), line1 x y ∧ line2 x y ∧ line3 a x y} :=
sorry

end NUMINAMATH_GPT_possible_values_of_a_l2423_242324


namespace NUMINAMATH_GPT_remainder_division_l2423_242364

theorem remainder_division
  (j : ℕ) (h_pos : 0 < j)
  (h_rem : ∃ b : ℕ, 72 = b * j^2 + 8) :
  150 % j = 6 :=
sorry

end NUMINAMATH_GPT_remainder_division_l2423_242364


namespace NUMINAMATH_GPT_quadratic_real_solutions_l2423_242385

theorem quadratic_real_solutions (p : ℝ) : (∃ x : ℝ, x^2 + p = 0) ↔ p ≤ 0 :=
sorry

end NUMINAMATH_GPT_quadratic_real_solutions_l2423_242385


namespace NUMINAMATH_GPT_perp_vectors_dot_product_eq_zero_l2423_242315

noncomputable def vector_a : ℝ × ℝ := (1, 2)
noncomputable def vector_b (x : ℝ) : ℝ × ℝ := (x, 4)

def dot_product (v1 v2 : ℝ × ℝ) : ℝ :=
  v1.1 * v2.1 + v1.2 * v2.2

theorem perp_vectors_dot_product_eq_zero (x : ℝ) (h : dot_product vector_a (vector_b x) = 0) : x = -8 :=
  by sorry

end NUMINAMATH_GPT_perp_vectors_dot_product_eq_zero_l2423_242315


namespace NUMINAMATH_GPT_bug_paths_from_A_to_B_l2423_242308

-- Define the positions A and B and intermediate red and blue points in the lattice
inductive Position
| A
| B
| red1
| red2
| blue1
| blue2

open Position

-- Define the possible directed paths in the lattice
def paths : List (Position × Position) :=
[(A, red1), (A, red2), 
 (red1, blue1), (red1, blue2), 
 (red2, blue1), (red2, blue2), 
 (blue1, B), (blue1, B), (blue1, B), 
 (blue2, B), (blue2, B), (blue2, B)]

-- Define a function that calculates the number of unique paths from A to B without repeating any path
def count_paths : ℕ := sorry

-- The mathematical problem statement
theorem bug_paths_from_A_to_B : count_paths = 24 := sorry

end NUMINAMATH_GPT_bug_paths_from_A_to_B_l2423_242308


namespace NUMINAMATH_GPT_find_angle_A_determine_triangle_shape_l2423_242366

noncomputable def angle_A (A : ℝ) (m n : ℝ × ℝ) : Prop :=
  m.1 * n.1 + m.2 * n.2 = 7 / 2 ∧ m = (Real.cos (A / 2)^2, Real.cos (2 * A)) ∧ 
  n = (4, -1)

theorem find_angle_A : 
  ∃ A : ℝ,  (0 < A ∧ A < Real.pi) ∧ angle_A A (Real.cos (A / 2)^2, Real.cos (2 * A)) (4, -1) 
  := sorry

noncomputable def triangle_shape (a b c : ℝ) (A : ℝ) : Prop :=
  a = Real.sqrt 3 ∧ a^2 = b^2 + c^2 - b * c * Real.cos (A)

theorem determine_triangle_shape :
  ∀ (b c : ℝ), (b * c ≤ 3) → triangle_shape (Real.sqrt 3) b c (Real.pi / 3) →
  (b = Real.sqrt 3 ∧ c = Real.sqrt 3)
  := sorry


end NUMINAMATH_GPT_find_angle_A_determine_triangle_shape_l2423_242366


namespace NUMINAMATH_GPT_buyers_muffin_mix_l2423_242306

variable (P C M CM: ℕ)

theorem buyers_muffin_mix
    (h_total: P = 100)
    (h_cake: C = 50)
    (h_both: CM = 17)
    (h_neither: P - (C + M - CM) = 27)
    : M = 73 :=
by sorry

end NUMINAMATH_GPT_buyers_muffin_mix_l2423_242306


namespace NUMINAMATH_GPT_adam_played_rounds_l2423_242350

theorem adam_played_rounds (total_points points_per_round : ℕ) (h_total : total_points = 283) (h_per_round : points_per_round = 71) : total_points / points_per_round = 4 := by
  -- sorry is a placeholder for the actual proof
  sorry

end NUMINAMATH_GPT_adam_played_rounds_l2423_242350


namespace NUMINAMATH_GPT_largest_three_digit_number_with_7_in_hundreds_l2423_242304

def is_three_digit_number_with_7_in_hundreds (n : ℕ) : Prop := 
  100 ≤ n ∧ n < 1000 ∧ (n / 100) = 7

theorem largest_three_digit_number_with_7_in_hundreds : 
  ∀ (n : ℕ), is_three_digit_number_with_7_in_hundreds n → n ≤ 799 :=
by sorry

end NUMINAMATH_GPT_largest_three_digit_number_with_7_in_hundreds_l2423_242304


namespace NUMINAMATH_GPT_solve_equation_l2423_242335

theorem solve_equation (x : ℝ) (h : (x^2 + x + 1) / (x + 1) = x + 2) : x = -1/2 :=
by sorry

end NUMINAMATH_GPT_solve_equation_l2423_242335


namespace NUMINAMATH_GPT_wire_division_l2423_242327

theorem wire_division (L_wire_ft : Nat) (L_wire_inch : Nat) (L_part : Nat) (H1 : L_wire_ft = 5) (H2 : L_wire_inch = 4) (H3 : L_part = 16) :
  (L_wire_ft * 12 + L_wire_inch) / L_part = 4 :=
by 
  sorry

end NUMINAMATH_GPT_wire_division_l2423_242327


namespace NUMINAMATH_GPT_emissions_from_tap_water_l2423_242398

def carbon_dioxide_emission (x : ℕ) : ℕ := 9 / 10 * x  -- Note: using 9/10 instead of 0.9 to maintain integer type

theorem emissions_from_tap_water : carbon_dioxide_emission 10 = 9 :=
by
  sorry

end NUMINAMATH_GPT_emissions_from_tap_water_l2423_242398


namespace NUMINAMATH_GPT_compute_star_l2423_242331

def star (x y : ℕ) := 4 * x + 6 * y

theorem compute_star : star 3 4 = 36 := 
by
  sorry

end NUMINAMATH_GPT_compute_star_l2423_242331


namespace NUMINAMATH_GPT_tangent_sum_l2423_242382

theorem tangent_sum (tan : ℝ → ℝ)
  (h1 : ∀ A B, tan (A + B) = (tan A + tan B) / (1 - tan A * tan B))
  (h2 : tan 60 = Real.sqrt 3) :
  tan 20 + tan 40 + Real.sqrt 3 * tan 20 * tan 40 = Real.sqrt 3 := 
by
  sorry

end NUMINAMATH_GPT_tangent_sum_l2423_242382


namespace NUMINAMATH_GPT_sum_of_odd_integers_21_to_51_l2423_242368

noncomputable def arithmetic_seq (a d : ℕ) (n : ℕ) : ℕ := a + (n - 1) * d

noncomputable def sum_arithmetic_seq (a d l : ℕ) : ℕ :=
  let n := (l - a) / d + 1
  n * (a + l) / 2

theorem sum_of_odd_integers_21_to_51 : sum_arithmetic_seq 21 2 51 = 576 := by
  sorry

end NUMINAMATH_GPT_sum_of_odd_integers_21_to_51_l2423_242368


namespace NUMINAMATH_GPT_price_per_eraser_l2423_242360

-- Definitions of the given conditions
def boxes_donated : ℕ := 48
def erasers_per_box : ℕ := 24
def total_money_made : ℝ := 864

-- The Lean statement to prove the price per eraser is $0.75
theorem price_per_eraser : (total_money_made / (boxes_donated * erasers_per_box) = 0.75) := by
  sorry

end NUMINAMATH_GPT_price_per_eraser_l2423_242360


namespace NUMINAMATH_GPT_max_value_of_g_l2423_242369

def g : ℕ → ℕ
| n => if n < 7 then n + 7 else g (n - 3)

theorem max_value_of_g : ∀ (n : ℕ), g n ≤ 13 ∧ (∃ n0, g n0 = 13) := by
  sorry

end NUMINAMATH_GPT_max_value_of_g_l2423_242369


namespace NUMINAMATH_GPT_probability_ace_king_queen_l2423_242323

-- Definitions based on the conditions
def total_cards := 52
def aces := 4
def kings := 4
def queens := 4

def probability_first_ace := aces / total_cards
def probability_second_king := kings / (total_cards - 1)
def probability_third_queen := queens / (total_cards - 2)

theorem probability_ace_king_queen :
  (probability_first_ace * probability_second_king * probability_third_queen) = (8 / 16575) :=
by sorry

end NUMINAMATH_GPT_probability_ace_king_queen_l2423_242323

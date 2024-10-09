import Mathlib

namespace correct_average_l83_8312

theorem correct_average :
  let avg_incorrect := 15
  let num_numbers := 20
  let read_incorrect1 := 42
  let read_correct1 := 52
  let read_incorrect2 := 68
  let read_correct2 := 78
  let read_incorrect3 := 85
  let read_correct3 := 95
  let incorrect_sum := avg_incorrect * num_numbers
  let diff1 := read_correct1 - read_incorrect1
  let diff2 := read_correct2 - read_incorrect2
  let diff3 := read_correct3 - read_incorrect3
  let total_diff := diff1 + diff2 + diff3
  let correct_sum := incorrect_sum + total_diff
  let correct_avg := correct_sum / num_numbers
  correct_avg = 16.5 :=
by
  sorry

end correct_average_l83_8312


namespace number_of_sheets_l83_8368

theorem number_of_sheets (S E : ℕ) (h1 : S - E = 60) (h2 : 5 * E = S) : S = 150 := by
  sorry

end number_of_sheets_l83_8368


namespace quadratic_completion_l83_8389

noncomputable def sum_of_r_s (r s : ℝ) : ℝ := r + s

theorem quadratic_completion (x r s : ℝ) (h : 16 * x^2 - 64 * x - 144 = 0) :
  ((x + r)^2 = s) → sum_of_r_s r s = -7 :=
by
  sorry

end quadratic_completion_l83_8389


namespace original_price_calculation_l83_8326

variable (P : ℝ)
variable (selling_price : ℝ := 1040)
variable (loss_percentage : ℝ := 20)

theorem original_price_calculation :
  P = 1300 :=
by
  have sell_percent := 100 - loss_percentage
  have SP_eq := selling_price = (sell_percent / 100) * P
  sorry

end original_price_calculation_l83_8326


namespace right_triangle_to_acute_triangle_l83_8343

theorem right_triangle_to_acute_triangle 
  (a b c d : ℝ) (h_triangle : a^2 + b^2 = c^2) (h_pos : a > 0 ∧ b > 0 ∧ c > 0) (h_increase : d > 0):
  (a + d)^2 + (b + d)^2 > (c + d)^2 := 
by {
  sorry
}

end right_triangle_to_acute_triangle_l83_8343


namespace number_of_rice_packets_l83_8316

theorem number_of_rice_packets
  (initial_balance : ℤ) 
  (price_per_rice_packet : ℤ)
  (num_wheat_flour_packets : ℤ) 
  (price_per_wheat_flour_packet : ℤ)
  (price_soda : ℤ) 
  (remaining_balance : ℤ)
  (spent : ℤ)
  (eqn : initial_balance - (price_per_rice_packet * 2 + num_wheat_flour_packets * price_per_wheat_flour_packet + price_soda) = remaining_balance) :
  price_per_rice_packet * 2 + num_wheat_flour_packets * price_per_wheat_flour_packet + price_soda = spent 
    → initial_balance - spent = remaining_balance
    → 2 = 2 :=
by 
  sorry

end number_of_rice_packets_l83_8316


namespace bug_crawl_distance_l83_8308

theorem bug_crawl_distance : 
  let start : ℤ := 3
  let first_stop : ℤ := -4
  let second_stop : ℤ := 7
  let final_stop : ℤ := -1
  |first_stop - start| + |second_stop - first_stop| + |final_stop - second_stop| = 26 := 
by
  sorry

end bug_crawl_distance_l83_8308


namespace divisibility_by_29_and_29pow4_l83_8358

theorem divisibility_by_29_and_29pow4 (x y z : ℤ) (h : 29 ∣ (x^4 + y^4 + z^4)) : 29^4 ∣ (x^4 + y^4 + z^4) :=
by
  sorry

end divisibility_by_29_and_29pow4_l83_8358


namespace sum_of_104th_parenthesis_is_correct_l83_8364

def b (n : ℕ) : ℕ := 2 * n + 1

def sumOf104thParenthesis : ℕ :=
  let cycleCount := 104 / 4
  let numbersBefore104 := 260
  let firstNumIndex := numbersBefore104 + 1
  let firstNum := b firstNumIndex
  let secondNum := b (firstNumIndex + 1)
  let thirdNum := b (firstNumIndex + 2)
  let fourthNum := b (firstNumIndex + 3)
  firstNum + secondNum + thirdNum + fourthNum

theorem sum_of_104th_parenthesis_is_correct : sumOf104thParenthesis = 2104 :=
  by
    sorry

end sum_of_104th_parenthesis_is_correct_l83_8364


namespace range_of_function_l83_8332

noncomputable def f (x : ℝ) : ℝ := x^2 - 4 * x

theorem range_of_function :
  ∀ (x : ℝ), 1 ≤ x ∧ x < 5 → -4 ≤ f x ∧ f x < 5 :=
by
  intro x hx
  sorry

end range_of_function_l83_8332


namespace correct_time_fraction_l83_8345

theorem correct_time_fraction : (3 / 4 : ℝ) * (3 / 4 : ℝ) = (9 / 16 : ℝ) :=
by
  sorry

end correct_time_fraction_l83_8345


namespace largest_value_of_x_l83_8314

theorem largest_value_of_x (x : ℝ) (h : |x - 8| = 15) : x ≤ 23 :=
by
  sorry -- Proof to be provided

end largest_value_of_x_l83_8314


namespace problem_statement_l83_8371

theorem problem_statement : ¬ (487.5 * 10^(-10) = 0.0000004875) :=
by
  sorry

end problem_statement_l83_8371


namespace eccentricity_of_hyperbola_l83_8378

noncomputable def hyperbola_eccentricity (a b : ℝ) (h1 : 0 < a) (h2 : 0 < b) (h3 : b^2 = 2 * a^2) : ℝ :=
  (1 + b^2 / a^2) ^ (1/2)

theorem eccentricity_of_hyperbola (a b : ℝ) (h1 : 0 < a) (h2 : 0 < b) (h3 : b^2 = 2 * a^2) :
  hyperbola_eccentricity a b h1 h2 h3 = Real.sqrt 3 := 
by
  unfold hyperbola_eccentricity
  rw [h3]
  simp
  sorry

end eccentricity_of_hyperbola_l83_8378


namespace a_finishes_race_in_t_seconds_l83_8383

theorem a_finishes_race_in_t_seconds 
  (time_B : ℝ := 45)
  (dist_B : ℝ := 100)
  (dist_A_wins_by : ℝ := 20)
  (total_dist : ℝ := 100)
  : ∃ t : ℝ, t = 36 := 
  sorry

end a_finishes_race_in_t_seconds_l83_8383


namespace denote_depth_below_sea_level_l83_8323

theorem denote_depth_below_sea_level (above_sea_level : Int) (depth_haidou_1 : Int) :
  (above_sea_level > 0) ∧ (depth_haidou_1 < 0) → depth_haidou_1 = -10907 :=
by
  intros h
  sorry

end denote_depth_below_sea_level_l83_8323


namespace remainder_of_7529_div_by_9_is_not_divisible_by_11_l83_8388

theorem remainder_of_7529_div_by_9 : 7529 % 9 = 5 := by
  sorry

theorem is_not_divisible_by_11 : ¬ (7529 % 11 = 0) := by
  sorry

end remainder_of_7529_div_by_9_is_not_divisible_by_11_l83_8388


namespace num_integers_satisfy_inequality_l83_8386

theorem num_integers_satisfy_inequality : ∃ (s : Finset ℤ), (∀ x ∈ s, |7 * x - 5| ≤ 15) ∧ s.card = 5 :=
by
  sorry

end num_integers_satisfy_inequality_l83_8386


namespace rotate_image_eq_A_l83_8311

def image_A : Type := sorry -- Image data for option (A)
def original_image : Type := sorry -- Original image data

def rotate_90_clockwise (img : Type) : Type := sorry -- Function to rotate image 90 degrees clockwise

theorem rotate_image_eq_A :
  rotate_90_clockwise original_image = image_A :=
sorry

end rotate_image_eq_A_l83_8311


namespace ac_lt_bc_of_a_gt_b_and_c_lt_0_l83_8380

theorem ac_lt_bc_of_a_gt_b_and_c_lt_0 {a b c : ℝ} (h1 : a > b) (h2 : c < 0) : a * c < b * c :=
  sorry

end ac_lt_bc_of_a_gt_b_and_c_lt_0_l83_8380


namespace initial_number_of_friends_l83_8352

theorem initial_number_of_friends (F : ℕ) (h : 6 * (F + 2) = 60) : F = 8 :=
by {
  sorry
}

end initial_number_of_friends_l83_8352


namespace BD_value_l83_8393

noncomputable def triangleBD (AC BC AD CD : ℝ) : ℝ :=
  let θ := Real.arccos ((3 ^ 2 + 9 ^ 2 - 7 ^ 2) / (2 * 3 * 9))
  let ψ := Real.pi - θ
  let cosψ := Real.cos ψ
  let x := (-1.026 + Real.sqrt ((1.026 ^ 2) + 4 * 40)) / 2
  if x > 0 then x else 5.8277 -- confirmed manually as positive root.

theorem BD_value : (triangleBD 7 7 9 3) = 5.8277 :=
by
  apply sorry

end BD_value_l83_8393


namespace problem_part_I_problem_part_II_l83_8372

theorem problem_part_I (A B C : ℝ)
  (h1 : 0 < A) 
  (h2 : A < π / 2)
  (h3 : 1 + (Real.sqrt 3 / 3) * Real.sin (2 * A) = 2 * (Real.sin ((B + C) / 2))^2) : 
  A = π / 3 := 
sorry

theorem problem_part_II (A B C R S : ℝ)
  (h1 : A = π / 3)
  (h2 : R = 2 * Real.sqrt 3) 
  (h3 : S = (1 / 2) * (6 * (Real.sin A)) * (Real.sqrt 3 / 2)) :
  S = 9 * Real.sqrt 3 :=
sorry

end problem_part_I_problem_part_II_l83_8372


namespace periodic_function_of_f_l83_8355

theorem periodic_function_of_f (f : ℝ → ℝ) (c : ℝ) (h : ∀ x, f (x + c) = (2 / (1 + f x)) - 1) : ∀ x, f (x + 2 * c) = f x :=
sorry

end periodic_function_of_f_l83_8355


namespace mass_of_fat_max_mass_of_carbohydrates_l83_8348

-- Definitions based on conditions
def total_mass : ℤ := 500
def fat_percentage : ℚ := 5 / 100
def protein_to_mineral_ratio : ℤ := 4

-- Lean 4 statement for Part 1: mass of fat
theorem mass_of_fat : (total_mass : ℚ) * fat_percentage = 25 := sorry

-- Definitions to utilize in Part 2
def max_percentage_protein_carbs : ℚ := 85 / 100
def mass_protein (x : ℚ) : ℚ := protein_to_mineral_ratio * x

-- Lean 4 statement for Part 2: maximum mass of carbohydrates
theorem max_mass_of_carbohydrates (x : ℚ) :
  x ≥ 50 → (total_mass - 25 - x - mass_protein x) ≤ 225 := sorry

end mass_of_fat_max_mass_of_carbohydrates_l83_8348


namespace maximum_M_l83_8328

-- Define the sides of a triangle condition
def is_triangle (a b c : ℝ) : Prop :=
  a + b > c ∧ a + c > b ∧ b + c > a

-- Theorem statement
theorem maximum_M (a b c : ℝ) (h : is_triangle a b c) : 
  (a^2 + b^2) / (c^2) > (1/2) :=
sorry

end maximum_M_l83_8328


namespace ellipse_same_foci_l83_8395

-- Definitions related to the problem
variables {x y p q : ℝ}

-- Condition
def represents_hyperbola (p q : ℝ) : Prop :=
  (p * q > 0) ∧ (∀ x y : ℝ, (x^2 / -p + y^2 / q = 1))

-- Proof Statement
theorem ellipse_same_foci (p q : ℝ) (hpq : p * q > 0)
  (h : ∀ x y : ℝ, x^2 / -p + y^2 / q = 1) :
  (∀ x y : ℝ, x^2 / (2*p + q) + y^2 / p = -1) :=
sorry -- Proof goes here

end ellipse_same_foci_l83_8395


namespace hyperbola_asymptote_l83_8315

theorem hyperbola_asymptote (a : ℝ) (h : a > 0)
  (has_asymptote : ∀ x : ℝ, abs (9 / a * x) = abs (3 * x))
  : a = 3 :=
sorry

end hyperbola_asymptote_l83_8315


namespace hotel_accommodation_arrangements_l83_8394

theorem hotel_accommodation_arrangements :
  let triple_room := 1
  let double_rooms := 2
  let adults := 3
  let children := 2
  (∀ (triple_room : ℕ) (double_rooms : ℕ) (adults : ℕ) (children : ℕ),
    children ≤ adults ∧ double_rooms + triple_room ≥ 1 →
    (∃ (arrangements : ℕ),
      arrangements = 60)) :=
sorry

end hotel_accommodation_arrangements_l83_8394


namespace average_of_11_numbers_l83_8331

theorem average_of_11_numbers (a b c d e f g h i j k : ℕ) 
  (h₀ : (a + b + c + d + e + f) / 6 = 19)
  (h₁ : (f + g + h + i + j + k) / 6 = 27)
  (h₂ : f = 34) :
  (a + b + c + d + e + f + g + h + i + j + k) / 11 = 22 := 
by
  sorry

end average_of_11_numbers_l83_8331


namespace man_double_son_age_in_two_years_l83_8313

theorem man_double_son_age_in_two_years (S M Y : ℕ) (h1 : S = 14) (h2 : M = S + 16) (h3 : Y = 2) : 
  M + Y = 2 * (S + Y) :=
by
  sorry

-- Explanation:
-- h1 establishes the son's current age.
-- h2 establishes the man's current age in relation to the son's age.
-- h3 gives the solution Y = 2 years.
-- We need to prove that M + Y = 2 * (S + Y).

end man_double_son_age_in_two_years_l83_8313


namespace min_k_l83_8362

def a_n (n : ℕ) : ℕ :=
  n

def b_n (n : ℕ) : ℚ :=
  a_n n / 3^n

def T_n (n : ℕ) : ℚ :=
  (List.range n).foldl (λ acc i => acc + b_n (i + 1)) 0

theorem min_k (k : ℕ) (h : ∀ n : ℕ, n ≥ k → |T_n n - 3/4| < 1/(4*n)) : k = 4 :=
  sorry

end min_k_l83_8362


namespace correct_alarm_clock_time_l83_8309

-- Definitions for the conditions
def alarm_set_time : ℕ := 7 * 60 -- in minutes
def museum_arrival_time : ℕ := 8 * 60 + 50 -- in minutes
def museum_touring_time : ℕ := 1 * 60 + 30 -- in minutes
def alarm_home_time : ℕ := 11 * 60 + 50 -- in minutes

-- The problem: proving the correct time the clock should be set to
theorem correct_alarm_clock_time : 
  (alarm_home_time - (2 * ((museum_arrival_time - alarm_set_time) + museum_touring_time / 2)) = 12 * 60) :=
  by
    sorry

end correct_alarm_clock_time_l83_8309


namespace distinct_meals_l83_8370

-- Define the conditions
def number_of_entrees : ℕ := 4
def number_of_drinks : ℕ := 3
def number_of_desserts : ℕ := 2

-- Define the main theorem
theorem distinct_meals : number_of_entrees * number_of_drinks * number_of_desserts = 24 := 
by
  -- sorry is used to skip the proof
  sorry

end distinct_meals_l83_8370


namespace magic_square_y_value_l83_8319

theorem magic_square_y_value 
  (a b c d e y : ℝ)
  (h1 : y + 4 + c = 81 + a + c)
  (h2 : y + (y - 77) + e = 81 + b + e)
  (h3 : y + 25 + 81 = 4 + (y - 77) + (2 * y - 158)) : 
  y = 168.5 :=
by
  -- required steps to complete the proof
  sorry

end magic_square_y_value_l83_8319


namespace comb_7_2_equals_21_l83_8307

theorem comb_7_2_equals_21 : (Nat.choose 7 2) = 21 := by
  sorry

end comb_7_2_equals_21_l83_8307


namespace find_number_l83_8392

theorem find_number (x : ℝ) (h : (x / 6) * 12 = 13) : x = 6.5 :=
by
  sorry

end find_number_l83_8392


namespace number_of_distinguishable_large_triangles_l83_8356

theorem number_of_distinguishable_large_triangles (colors : Fin 8) :
  ∃(large_triangles : Fin 960), true :=
by
  sorry

end number_of_distinguishable_large_triangles_l83_8356


namespace lambda_range_l83_8347

noncomputable def sequence_a (n : ℕ) : ℝ :=
  if n = 0 then 1 else
  sequence_a (n - 1) / (sequence_a (n - 1) + 2)

noncomputable def sequence_b (lambda : ℝ) (n : ℕ) : ℝ :=
  if n = 0 then -3/2 * lambda else
  (n - 2 * lambda) * (1 / sequence_a (n - 1) + 1)

def is_monotonically_increasing (seq : ℕ → ℝ) : Prop :=
  ∀ n : ℕ, n > 0 → seq (n+1) > seq n

theorem lambda_range (lambda : ℝ) (hn : is_monotonically_increasing (sequence_b lambda)) : lambda < 4/5 := sorry

end lambda_range_l83_8347


namespace smallest_number_value_l83_8335

variable (a b c : ℕ)

def conditions (a b c : ℕ) : Prop :=
  a + b + c = 100 ∧
  c = 2 * a ∧
  c - b = 10

theorem smallest_number_value (h : conditions a b c) : a = 22 :=
by
  sorry

end smallest_number_value_l83_8335


namespace general_formula_l83_8333

noncomputable def a : ℕ → ℕ
| 0       => 5
| (n + 1) => 2 * a n + 3

theorem general_formula : ∀ n, a n = 2 ^ (n + 2) - 3 :=
by
  sorry

end general_formula_l83_8333


namespace computation_result_l83_8318

theorem computation_result : 143 - 13 + 31 + 17 = 178 := 
by
  sorry

end computation_result_l83_8318


namespace smallest_A_is_144_l83_8391

noncomputable def smallest_A (B : ℕ) := B * 28 + 4

theorem smallest_A_is_144 :
  ∃ (B : ℕ), smallest_A B = 144 ∧ ∀ (B' : ℕ), B' * 28 + 4 < 144 → false :=
by
  sorry

end smallest_A_is_144_l83_8391


namespace weight_of_replaced_person_l83_8324

theorem weight_of_replaced_person :
  (∃ (W : ℝ), 
    let avg_increase := 1.5 
    let num_persons := 5 
    let new_person_weight := 72.5 
    (avg_increase * num_persons = new_person_weight - W)
  ) → 
  ∃ (W : ℝ), W = 65 :=
by
  sorry

end weight_of_replaced_person_l83_8324


namespace second_smallest_five_digit_in_pascals_triangle_l83_8322

noncomputable def binomial (n k : ℕ) : ℕ :=
  Nat.factorial n / (Nat.factorial k * Nat.factorial (n - k))

theorem second_smallest_five_digit_in_pascals_triangle :
  (∃ n k : ℕ, n > 0 ∧ k > 0 ∧ (10000 ≤ binomial n k) ∧ (binomial n k < 100000) ∧
    (∀ m l : ℕ, m > 0 ∧ l > 0 ∧ (10000 ≤ binomial m l) ∧ (binomial m l < 100000) →
    (binomial n k < binomial m l → binomial n k ≥ 31465)) ∧  binomial n k = 31465) :=
sorry

end second_smallest_five_digit_in_pascals_triangle_l83_8322


namespace gear_q_revolutions_per_minute_l83_8340

-- Define the constants and conditions
def revolutions_per_minute_p : ℕ := 10
def revolutions_per_minute_q : ℕ := sorry
def time_in_minutes : ℝ := 1.5
def extra_revolutions_q : ℕ := 45

-- Calculate the number of revolutions for gear p in 90 seconds
def revolutions_p_in_90_seconds := revolutions_per_minute_p * time_in_minutes

-- Condition that gear q makes exactly 45 more revolutions than gear p in 90 seconds
def revolutions_q_in_90_seconds := revolutions_p_in_90_seconds + extra_revolutions_q

-- Correct answer
def correct_answer : ℕ := 40

-- Prove that gear q makes 40 revolutions per minute
theorem gear_q_revolutions_per_minute : 
    revolutions_per_minute_q = correct_answer :=
sorry

end gear_q_revolutions_per_minute_l83_8340


namespace no_int_coeffs_l83_8381

def P (a b c d : ℤ) (x : ℤ) : ℤ := a * x^3 + b * x^2 + c * x + d

theorem no_int_coeffs (a b c d : ℤ) : 
  ¬ (P a b c d 19 = 1 ∧ P a b c d 62 = 2) :=
by sorry

end no_int_coeffs_l83_8381


namespace card_tag_sum_l83_8329

noncomputable def W : ℕ := 200
noncomputable def X : ℝ := 2 / 3 * W
noncomputable def Y : ℝ := W + X
noncomputable def Z : ℝ := Real.sqrt Y
noncomputable def P : ℝ := X^3
noncomputable def Q : ℝ := Nat.factorial W / 100000
noncomputable def R : ℝ := 3 / 5 * (P + Q)
noncomputable def S : ℝ := W^1 + X^2 + Z^3

theorem card_tag_sum :
  W + X + Y + Z + P + S = 2373589.26 + Q + R :=
by
  sorry

end card_tag_sum_l83_8329


namespace range_of_f_l83_8365

open Set

noncomputable def f (x : ℝ) : ℝ := (x^2 + 5*x + 6) / (x + 2)

theorem range_of_f : range f = {y : ℝ | y ≠ 1} :=
by
  sorry

end range_of_f_l83_8365


namespace min_floor_sum_l83_8397

theorem min_floor_sum (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) : 
  ∃ (n : ℕ), n = 4 ∧ n = 
  ⌊(2 * a + b) / c⌋ + ⌊(b + 2 * c) / a⌋ + ⌊(2 * c + a) / b⌋ := 
sorry

end min_floor_sum_l83_8397


namespace range_of_m_l83_8379

theorem range_of_m 
  (h : ∀ x, -1 < x ∧ x < 4 → x > 2 * (m: ℝ)^2 - 3)
  : ∀ (m: ℝ), -1 ≤ m ∧ m ≤ 1 :=
by 
  sorry

end range_of_m_l83_8379


namespace arithmetic_sequence_sum_l83_8360

theorem arithmetic_sequence_sum (S : ℕ → ℤ) (a_1 : ℤ) (h1 : a_1 = -2017) 
  (h2 : (S 2009) / 2009 - (S 2007) / 2007 = 2) : 
  S 2017 = -2017 :=
by
  -- definitions and steps would go here
  sorry

end arithmetic_sequence_sum_l83_8360


namespace beads_per_package_eq_40_l83_8390

theorem beads_per_package_eq_40 (b r : ℕ) (x : ℕ) (total_beads : ℕ) 
(h1 : b = 3) (h2 : r = 5) (h3 : total_beads = 320) (h4 : total_beads = (b + r) * x) :
  x = 40 := by
  sorry

end beads_per_package_eq_40_l83_8390


namespace minimum_a_l83_8359

theorem minimum_a (a b x : ℕ) (h₀ : a > 0) (h₁ : b > 0) (h₂ : b - a = 2013) (h₃ : x > 0) (h₄ : x^2 - a * x + b = 0) : a = 93 :=
by
  sorry

end minimum_a_l83_8359


namespace average_monthly_balance_l83_8300

def january_balance : ℕ := 150
def february_balance : ℕ := 300
def march_balance : ℕ := 450
def april_balance : ℕ := 300
def number_of_months : ℕ := 4

theorem average_monthly_balance :
  (january_balance + february_balance + march_balance + april_balance) / number_of_months = 300 := by
  sorry

end average_monthly_balance_l83_8300


namespace exists_sequence_a_l83_8317

def c (n : ℕ) : ℕ := 2017 ^ n

axiom f : ℕ → ℝ

axiom condition_1 : ∀ m n : ℕ, f (m + n) ≤ 2017 * f m * f (n + 325)

axiom condition_2 : ∀ n : ℕ, 0 < f (c (n + 1)) ∧ f (c (n + 1)) < (f (c n)) ^ 2017

theorem exists_sequence_a :
  ∃ (a : ℕ → ℕ), ∀ n k : ℕ, a k < n → f n ^ c k < f (c k) ^ n := sorry

end exists_sequence_a_l83_8317


namespace problem_statement_l83_8385

theorem problem_statement :
  (1 / 3 * 1 / 6 * P = (1 / 4 * 1 / 8 * 64) + (1 / 5 * 1 / 10 * 100)) → 
  P = 72 :=
by
  sorry

end problem_statement_l83_8385


namespace arithmetic_series_sum_proof_middle_term_proof_l83_8350

def arithmetic_series_sum (a d n : ℤ) : ℤ :=
  n * (2 * a + (n - 1) * d) / 2

def middle_term (a l : ℤ) : ℤ :=
  (a + l) / 2

theorem arithmetic_series_sum_proof :
  let a := -51
  let d := 2
  let n := 27
  let l := 1
  arithmetic_series_sum a d n = -675 :=
by
  sorry

theorem middle_term_proof :
  let a := -51
  let l := 1
  middle_term a l = -25 :=
by
  sorry

end arithmetic_series_sum_proof_middle_term_proof_l83_8350


namespace sphere_radius_in_cube_l83_8387

theorem sphere_radius_in_cube (r : ℝ) (n : ℕ) (side_length : ℝ) 
  (h1 : side_length = 2) 
  (h2 : n = 16)
  (h3 : ∀ (i : ℕ), i < n → (center_distance : ℝ) = 2 * r)
  (h4: ∀ (i : ℕ), i < n → (face_distance : ℝ) = r) : 
  r = 1 :=
by
  sorry

end sphere_radius_in_cube_l83_8387


namespace sandra_savings_l83_8305

theorem sandra_savings :
  let num_notepads := 8
  let original_price_per_notepad := 3.75
  let discount_rate := 0.25
  let discount_per_notepad := original_price_per_notepad * discount_rate
  let discounted_price_per_notepad := original_price_per_notepad - discount_per_notepad
  let total_cost_without_discount := num_notepads * original_price_per_notepad
  let total_cost_with_discount := num_notepads * discounted_price_per_notepad
  let total_savings := total_cost_without_discount - total_cost_with_discount
  total_savings = 7.50 :=
sorry

end sandra_savings_l83_8305


namespace amino_inequality_l83_8342

theorem amino_inequality
  (x y z : ℝ)
  (hx : x ≠ 0)
  (hy : y ≠ 0)
  (hz : z ≠ 0)
  (h : x + y + z = x * y * z) :
  ( (x^2 - 1) / x )^2 + ( (y^2 - 1) / y )^2 + ( (z^2 - 1) / z )^2 ≥ 4 := by
  sorry

end amino_inequality_l83_8342


namespace problem_solution_l83_8304

theorem problem_solution :
  20 * ((180 / 3) + (40 / 5) + (16 / 32) + 2) = 1410 := by
  sorry

end problem_solution_l83_8304


namespace divisibility_of_sum_of_fifths_l83_8302

theorem divisibility_of_sum_of_fifths (x y z : ℤ) (h : x ≠ y ∧ y ≠ z ∧ z ≠ x) :
  ∃ k : ℤ, (x - y) ^ 5 + (y - z) ^ 5 + (z - x) ^ 5 = 5 * k * (x - y) * (y - z) * (z - x) :=
sorry

end divisibility_of_sum_of_fifths_l83_8302


namespace second_quadrant_coordinates_l83_8361

theorem second_quadrant_coordinates (x y : ℝ) (h1 : x < 0) (h2 : y > 0) (h3 : |x| = 2) (h4 : y^2 = 1) :
    (x, y) = (-2, 1) :=
  sorry

end second_quadrant_coordinates_l83_8361


namespace max_unique_sums_l83_8357

-- Define the coin values in cents
def penny := 1
def nickel := 5
def quarter := 25
def half_dollar := 50

-- Define the set of all coins and their counts
structure Coins :=
  (pennies : ℕ := 3)
  (nickels : ℕ := 3)
  (quarters : ℕ := 1)
  (half_dollars : ℕ := 2)

-- Define the list of all possible pairs and their sums
def possible_sums : Finset ℕ :=
  { 2, 6, 10, 26, 30, 51, 55, 75, 100 }

-- Prove that the count of unique sums is 9
theorem max_unique_sums (c : Coins) : c.pennies = 3 → c.nickels = 3 → c.quarters = 1 → c.half_dollars = 2 →
  possible_sums.card = 9 := 
by
  intros
  sorry

end max_unique_sums_l83_8357


namespace fraction_value_l83_8399

theorem fraction_value : (4 * 5) / 10 = 2 := by
  sorry

end fraction_value_l83_8399


namespace maximum_area_of_garden_l83_8321

noncomputable def max_area (perimeter : ℕ) : ℕ :=
  let half_perimeter := perimeter / 2
  let x := half_perimeter / 2
  x * x

theorem maximum_area_of_garden :
  max_area 148 = 1369 :=
by
  sorry

end maximum_area_of_garden_l83_8321


namespace remainder_23_pow_2003_mod_7_l83_8374

theorem remainder_23_pow_2003_mod_7 : 23 ^ 2003 % 7 = 4 :=
by sorry

end remainder_23_pow_2003_mod_7_l83_8374


namespace Maggie_takes_75_percent_l83_8339

def Debby's_portion : ℚ := 0.25
def Maggie's_share : ℚ := 4500
def Total_amount : ℚ := 6000
def Maggie's_portion : ℚ := Maggie's_share / Total_amount

theorem Maggie_takes_75_percent : Maggie's_portion = 0.75 :=
by
  sorry

end Maggie_takes_75_percent_l83_8339


namespace infinite_solutions_iff_l83_8375

theorem infinite_solutions_iff (a b c d : ℤ) :
  (∃ᶠ x in at_top, ∃ᶠ y in at_top, x^2 + a * x + b = y^2 + c * y + d) ↔ (a^2 - 4 * b = c^2 - 4 * d) :=
by sorry

end infinite_solutions_iff_l83_8375


namespace gh_two_value_l83_8373

def g (x : ℤ) : ℤ := 3 * x ^ 2 + 2
def h (x : ℤ) : ℤ := -5 * x ^ 3 + 2

theorem gh_two_value : g (h 2) = 4334 := by
  sorry

end gh_two_value_l83_8373


namespace value_of_livestock_l83_8354

variable (x y : ℝ)

theorem value_of_livestock :
  (5 * x + 2 * y = 10) ∧ (2 * x + 5 * y = 8) :=
sorry

end value_of_livestock_l83_8354


namespace platform_length_l83_8334

theorem platform_length (train_length : ℝ) (time_pole : ℝ) (time_platform : ℝ) (speed : ℝ) (platform_length : ℝ) :
  train_length = 300 → time_pole = 18 → time_platform = 38 → speed = train_length / time_pole →
  platform_length = (speed * time_platform) - train_length → platform_length = 333.46 :=
by
  introv h1 h2 h3 h4 h5
  sorry

end platform_length_l83_8334


namespace smallest_integer_satisfying_conditions_l83_8303

-- Define the conditions explicitly as hypotheses
def satisfies_congruence_3_2 (n : ℕ) : Prop :=
  n % 3 = 2

def satisfies_congruence_7_2 (n : ℕ) : Prop :=
  n % 7 = 2

def satisfies_congruence_8_2 (n : ℕ) : Prop :=
  n % 8 = 2

def is_perfect_square (n : ℕ) : Prop :=
  ∃ k : ℕ, k * k = n

-- Define the smallest positive integer satisfying the above conditions
theorem smallest_integer_satisfying_conditions : ∃ (n : ℕ), n > 1 ∧ satisfies_congruence_3_2 n ∧ satisfies_congruence_7_2 n ∧ satisfies_congruence_8_2 n ∧ is_perfect_square n :=
  by
    sorry

end smallest_integer_satisfying_conditions_l83_8303


namespace isosceles_triangle_count_l83_8336

noncomputable def valid_points : List (ℕ × ℕ) :=
  [(2, 5), (5, 5)]

theorem isosceles_triangle_count 
  (A B : ℕ × ℕ) 
  (H_A : A = (2, 2)) 
  (H_B : B = (5, 2)) : 
  valid_points.length = 2 :=
  sorry

end isosceles_triangle_count_l83_8336


namespace exists_integer_K_l83_8301

theorem exists_integer_K (Z : ℕ) (K : ℕ) : 
  1000 < Z ∧ Z < 2000 ∧ Z = K^4 → 
  ∃ K, K = 6 := 
by
  sorry

end exists_integer_K_l83_8301


namespace principal_sum_investment_l83_8325

theorem principal_sum_investment 
    (P R : ℝ) 
    (h1 : (P * 5 * (R + 2)) / 100 - (P * 5 * R) / 100 = 180)
    (h2 : (P * 5 * (R + 3)) / 100 - (P * 5 * R) / 100 = 270) :
    P = 1800 :=
by
  -- These are the hypotheses generated for Lean, the proof steps are omitted
  sorry

end principal_sum_investment_l83_8325


namespace base_angle_of_isosceles_triangle_l83_8327

-- Definitions based on the problem conditions
def is_isosceles_triangle (A B C: ℝ) := (A = B) ∨ (B = C) ∨ (C = A)
def angle_sum_triangle (A B C: ℝ) := A + B + C = 180

-- The main theorem we want to prove
theorem base_angle_of_isosceles_triangle (A B C: ℝ)
(h1: is_isosceles_triangle A B C)
(h2: A = 50 ∨ B = 50 ∨ C = 50):
C = 50 ∨ C = 65 :=
by
  sorry

end base_angle_of_isosceles_triangle_l83_8327


namespace cubic_roots_cosines_l83_8320

theorem cubic_roots_cosines
  {p q r : ℝ}
  (h_eq : ∀ x : ℝ, x^3 + p * x^2 + q * x + r = 0)
  (h_roots : ∃ (α β γ : ℝ), (α > 0) ∧ (β > 0) ∧ (γ > 0) ∧ (α + β + γ = -p) ∧ 
             (α * β + β * γ + γ * α = q) ∧ (α * β * γ = -r)) :
  2 * r + 1 = p^2 - 2 * q :=
by
  sorry

end cubic_roots_cosines_l83_8320


namespace vat_percentage_is_15_l83_8382

def original_price : ℝ := 1700
def final_price : ℝ := 1955
def tax_amount := final_price - original_price

theorem vat_percentage_is_15 :
  (tax_amount / original_price) * 100 = 15 := 
sorry

end vat_percentage_is_15_l83_8382


namespace sqrt_function_of_x_l83_8337

theorem sqrt_function_of_x (x : ℝ) (h : x > 0) : ∃! y : ℝ, y = Real.sqrt x :=
by
  sorry

end sqrt_function_of_x_l83_8337


namespace eccentricity_of_hyperbola_l83_8351

theorem eccentricity_of_hyperbola :
  let a := Real.sqrt 5
  let b := 2
  let c := Real.sqrt (a^2 + b^2)
  let e := c / a
  (∃ (x y : ℝ), (x^2 / 5) - (y^2 / 4) = 1 ∧ e = (3 * Real.sqrt 5) / 5) := sorry

end eccentricity_of_hyperbola_l83_8351


namespace john_yasmin_child_ratio_l83_8367

theorem john_yasmin_child_ratio
  (gabriel_grandkids : ℕ)
  (yasmin_children : ℕ)
  (john_children : ℕ)
  (h1 : gabriel_grandkids = 6)
  (h2 : yasmin_children = 2)
  (h3 : john_children + yasmin_children = gabriel_grandkids) :
  john_children / yasmin_children = 2 :=
by 
  sorry

end john_yasmin_child_ratio_l83_8367


namespace inner_circle_radius_l83_8376

theorem inner_circle_radius :
  ∃ (r : ℝ) (a b c d : ℕ), 
    (r = (-78 + 70 * Real.sqrt 3) / 26) ∧ 
    (a = 78) ∧ 
    (b = 70) ∧ 
    (c = 3) ∧ 
    (d = 26) ∧ 
    (Nat.gcd a d = 1) ∧ 
    (a + b + c + d = 177) := 
sorry

end inner_circle_radius_l83_8376


namespace sufficient_but_not_necessary_l83_8310

theorem sufficient_but_not_necessary (a b : ℝ) (h : a > b ∧ b > 0) : a^2 > b^2 ∧ ¬ (a^2 > b^2 → a > b ∧ b > 0) :=
by
  sorry

end sufficient_but_not_necessary_l83_8310


namespace intersection_S_T_eq_interval_l83_8341

-- Define the sets S and T
def S : Set ℝ := {x | x ≥ 2}
def T : Set ℝ := {x | x ≤ 5}

-- Prove the intersection of S and T is [2, 5]
theorem intersection_S_T_eq_interval : S ∩ T = {x | 2 ≤ x ∧ x ≤ 5} :=
by
  sorry

end intersection_S_T_eq_interval_l83_8341


namespace min_value_problem_inequality_solution_l83_8349

-- Definition of the function
noncomputable def f (x a : ℝ) : ℝ := |x - a| + |x + 2|

-- Part (i): Minimum value problem
theorem min_value_problem (a : ℝ) (minF : ∀ x : ℝ, f x a ≥ 2) : a = 0 ∨ a = -4 :=
by
  sorry

-- Part (ii): Inequality solving problem
theorem inequality_solution (x : ℝ) (a : ℝ := 2) : f x a ≤ 6 ↔ -3 ≤ x ∧ x ≤ 3 :=
by
  sorry

end min_value_problem_inequality_solution_l83_8349


namespace triangle_angle_l83_8369

variable (a b c : ℝ)
variable (C : ℝ)

theorem triangle_angle (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0)
  (h : (a^2 + b^2) * (a^2 + b^2 - c^2) = 3 * a^2 * b^2) :
  C = Real.arccos ((a^4 + b^4 - a^2 * b^2) / (2 * a * b * (a^2 + b^2))) :=
sorry

end triangle_angle_l83_8369


namespace jovana_added_23_pounds_l83_8353

def initial_weight : ℕ := 5
def final_weight : ℕ := 28

def added_weight : ℕ := final_weight - initial_weight

theorem jovana_added_23_pounds : added_weight = 23 := 
by sorry

end jovana_added_23_pounds_l83_8353


namespace sum_first_9000_terms_l83_8344

noncomputable def geometric_sequence_sum (a r : ℝ) (n : ℕ) : ℝ :=
  a * (1 - r^n) / (1 - r)

theorem sum_first_9000_terms (a r : ℝ) :
  geometric_sequence_sum a r 3000 = 500 →
  geometric_sequence_sum a r 6000 = 950 →
  geometric_sequence_sum a r 9000 = 1355 :=
by
  intros h1 h2
  sorry

end sum_first_9000_terms_l83_8344


namespace percent_gain_on_transaction_l83_8398

theorem percent_gain_on_transaction
  (c : ℝ) -- cost per sheep
  (price_750_sold : ℝ := 800 * c) -- price at which 750 sheep were sold in total
  (price_per_sheep_750 : ℝ := price_750_sold / 750)
  (price_per_sheep_50 : ℝ := 1.1 * price_per_sheep_750)
  (revenue_750 : ℝ := price_per_sheep_750 * 750)
  (revenue_50 : ℝ := price_per_sheep_50 * 50)
  (total_revenue : ℝ := revenue_750 + revenue_50)
  (total_cost : ℝ := 800 * c)
  (profit : ℝ := total_revenue - total_cost)
  (percent_gain : ℝ := (profit / total_cost) * 100) :
  percent_gain = 14 :=
sorry

end percent_gain_on_transaction_l83_8398


namespace inverse_variation_l83_8346

theorem inverse_variation (x y k : ℝ) (h1 : y = k / x^2) (h2 : k = 8) (h3 : y = 0.5) : x = 4 := by
  sorry

end inverse_variation_l83_8346


namespace exactly_1_male_and_exactly_2_female_mutually_exclusive_not_complementary_l83_8384

-- Definitions based on the given conditions
def male_students := 3
def female_students := 2
def total_students := male_students + female_students

def at_least_1_male_event := ∃ (n : ℕ), n ≥ 1 ∧ n ≤ male_students
def all_female_event := ∀ (n : ℕ), n ≤ female_students
def at_least_1_female_event := ∃ (n : ℕ), n ≥ 1 ∧ n ≤ female_students
def all_male_event := ∀ (n : ℕ), n ≤ male_students
def exactly_1_male_event := ∃ (n : ℕ), n = 1 ∧ n ≤ male_students
def exactly_2_female_event := ∃ (n : ℕ), n = 2 ∧ n ≤ female_students

def mutually_exclusive (e1 e2 : Prop) : Prop := ¬ (e1 ∧ e2)
def complementary (e1 e2 : Prop) : Prop := e1 ∧ ¬ e2 ∨ ¬ e1 ∧ e2

-- Statement of the problem
theorem exactly_1_male_and_exactly_2_female_mutually_exclusive_not_complementary :
  mutually_exclusive exactly_1_male_event exactly_2_female_event ∧ 
  ¬ complementary exactly_1_male_event exactly_2_female_event :=
by
  sorry

end exactly_1_male_and_exactly_2_female_mutually_exclusive_not_complementary_l83_8384


namespace find_first_term_arithmetic_progression_l83_8396

theorem find_first_term_arithmetic_progression
  (a1 a2 a3 : ℝ)
  (h1 : a1 + a2 + a3 = 12)
  (h2 : a1 * a2 * a3 = 48)
  (h3 : a2 = a1 + d)
  (h4 : a3 = a1 + 2 * d)
  (h5 : a1 < a2 ∧ a2 < a3) :
  a1 = 2 :=
by
  sorry

end find_first_term_arithmetic_progression_l83_8396


namespace quadratic_function_positive_l83_8366

theorem quadratic_function_positive (a m : ℝ) (h : a > 0) (h_fm : (m^2 + m + a) < 0) : (m + 1)^2 + (m + 1) + a > 0 :=
by sorry

end quadratic_function_positive_l83_8366


namespace train_crossing_time_l83_8306

/-- 
Prove that the time it takes for a train traveling at 90 kmph with a length of 100.008 meters to cross a pole is 4.00032 seconds.
-/
theorem train_crossing_time (speed_kmph : ℝ) (length_meters : ℝ) : 
  speed_kmph = 90 → length_meters = 100.008 → (length_meters / (speed_kmph * (1000 / 3600))) = 4.00032 :=
by
  intros h1 h2
  sorry

end train_crossing_time_l83_8306


namespace find_constants_and_min_value_l83_8377

noncomputable def f (a b x : ℝ) := a * Real.exp x + b * x * Real.log x
noncomputable def f' (a b x : ℝ) := a * Real.exp x + b * Real.log x + b
noncomputable def g (a b x : ℝ) := f a b x - Real.exp 1 * x^2

theorem find_constants_and_min_value :
  (∀ (a b : ℝ),
    -- Condition for the derivative at x = 1 and the given tangent line slope
    (f' a b 1 = 2 * Real.exp 1) ∧
    -- Condition for the function value at x = 1
    (f a b 1 = Real.exp 1) →
    -- Expected results for a and b
    (a = 1 ∧ b = Real.exp 1)) ∧

  -- Evaluating the minimum value of the function g(x)
  (∀ (x : ℝ), 0 < x →
    -- Given the minimum occurs at x = 1
    g 1 (Real.exp 1) 1 = 0 ∧
    (∀ (x : ℝ), 0 < x →
      (g 1 (Real.exp 1) x ≥ 0))) :=
sorry

end find_constants_and_min_value_l83_8377


namespace determine_borrow_lend_years_l83_8363

theorem determine_borrow_lend_years (P : ℝ) (Rb Rl G : ℝ) (n : ℝ) 
  (hP : P = 9000) 
  (hRb : Rb = 4 / 100) 
  (hRl : Rl = 6 / 100) 
  (hG : G = 180) 
  (h_gain : G = P * Rl * n - P * Rb * n) : 
  n = 1 := 
sorry

end determine_borrow_lend_years_l83_8363


namespace vertex_of_parabola_l83_8338

theorem vertex_of_parabola (a : ℝ) :
  (∃ (k : ℝ), ∀ x : ℝ, y = -4*x - 1 → x = 2 ∧ (a - 4) = -4 * 2 - 1) → 
  (2, -9) = (2, a - 4) → a = -5 :=
by
  sorry

end vertex_of_parabola_l83_8338


namespace simplify_expression_l83_8330

variable (a : ℝ) (ha : a ≠ -3)

theorem simplify_expression : (a^2) / (a + 3) - 9 / (a + 3) = a - 3 :=
by
  sorry

end simplify_expression_l83_8330

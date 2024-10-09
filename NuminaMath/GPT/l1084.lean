import Mathlib

namespace solve_equation_l1084_108466

theorem solve_equation (x y : ℤ) (eq : (x^2 - y^2)^2 = 16 * y + 1) : 
  (x = 1 ∧ y = 0) ∨ (x = -1 ∧ y = 0) ∨ 
  (x = 4 ∧ y = 3) ∨ (x = -4 ∧ y = 3) ∨ 
  (x = 4 ∧ y = 5) ∨ (x = -4 ∧ y = 5) :=
sorry

end solve_equation_l1084_108466


namespace trigonometric_proof_l1084_108415

theorem trigonometric_proof (α : ℝ) (h : 3 * Real.sin α + Real.cos α = 0) : 
  1 / (Real.cos α ^ 2 + 2 * Real.sin α * Real.cos α) = 10 / 3 :=
by sorry

end trigonometric_proof_l1084_108415


namespace tens_digit_13_power_1987_l1084_108411

theorem tens_digit_13_power_1987 : (13^1987)%100 / 10 = 1 :=
by
  sorry

end tens_digit_13_power_1987_l1084_108411


namespace tan_ratio_l1084_108496

open Real

variables (p q : ℝ)

-- Conditions
def cond1 := (sin p / cos q + sin q / cos p = 2)
def cond2 := (cos p / sin q + cos q / sin p = 3)

-- Proof statement
theorem tan_ratio (hpq : cond1 p q) (hq : cond2 p q) :
  (tan p / tan q + tan q / tan p = 8 / 5) :=
sorry

end tan_ratio_l1084_108496


namespace max_5x_plus_3y_l1084_108407

theorem max_5x_plus_3y (x y : ℝ) (h : x^2 + y^2 = 10 * x + 8 * y + 10) : 5 * x + 3 * y ≤ 105 :=
sorry

end max_5x_plus_3y_l1084_108407


namespace total_days_correct_l1084_108442

-- Defining the years and the conditions given.
def year_1999 := 1999
def year_2000 := 2000
def year_2001 := 2001
def year_2002 := 2002

-- Defining the leap year and regular year days
def days_in_regular_year := 365
def days_in_leap_year := 366

-- Noncomputable version to skip the proof
noncomputable def total_days_from_1999_to_2002 : ℕ :=
  3 * days_in_regular_year + days_in_leap_year

-- The theorem stating the problem, which we need to prove
theorem total_days_correct : total_days_from_1999_to_2002 = 1461 := by
  sorry

end total_days_correct_l1084_108442


namespace orchard_tree_growth_problem_l1084_108424

theorem orchard_tree_growth_problem
  (T0 : ℕ) (Tn : ℕ) (n : ℕ)
  (h1 : T0 = 1280)
  (h2 : Tn = 3125)
  (h3 : Tn = (5/4 : ℚ) ^ n * T0) :
  n = 4 :=
by
  sorry

end orchard_tree_growth_problem_l1084_108424


namespace original_percentage_alcohol_l1084_108490

-- Definitions of the conditions
def original_mixture_volume : ℝ := 15
def additional_water_volume : ℝ := 3
def final_percentage_alcohol : ℝ := 20.833333333333336
def final_mixture_volume : ℝ := original_mixture_volume + additional_water_volume

-- Lean statement to prove
theorem original_percentage_alcohol (A : ℝ) :
  (A / 100 * original_mixture_volume) = (final_percentage_alcohol / 100 * final_mixture_volume) →
  A = 25 :=
by
  sorry

end original_percentage_alcohol_l1084_108490


namespace blend_pieces_eq_two_l1084_108401

variable (n_silk n_cashmere total_pieces : ℕ)

def luther_line := n_silk = 10 ∧ n_cashmere = n_silk / 2 ∧ total_pieces = 13

theorem blend_pieces_eq_two : luther_line n_silk n_cashmere total_pieces → (n_cashmere - (total_pieces - n_silk) = 2) :=
by
  intros
  sorry

end blend_pieces_eq_two_l1084_108401


namespace distinct_pairs_count_l1084_108470

theorem distinct_pairs_count : 
  (∃ (s : Finset (ℕ × ℕ)), (∀ p ∈ s, ∃ (a b : ℕ), 1 ≤ a ∧ 1 ≤ b ∧ a + b = 40 ∧ p = (a, b)) ∧ s.card = 39) := sorry

end distinct_pairs_count_l1084_108470


namespace vector_subtraction_l1084_108444

-- Definitions of given conditions
def OA : ℝ × ℝ := (2, 1)
def OB : ℝ × ℝ := (-3, 4)

-- Definition of vector subtraction
def vector_sub (v1 v2 : ℝ × ℝ) : ℝ × ℝ := (v1.1 - v2.1, v1.2 - v2.2)

theorem vector_subtraction : vector_sub OB OA = (-5, 3) :=
by 
  -- The proof would go here.
  sorry

end vector_subtraction_l1084_108444


namespace largest_sum_ABC_l1084_108416

noncomputable def max_sum_ABC (A B C : ℕ) : ℕ :=
if A ≠ B ∧ B ≠ C ∧ A ≠ C ∧ A * B * C = 2310 then
  A + B + C
else
  0

theorem largest_sum_ABC : ∃ A B C : ℕ, A ≠ B ∧ B ≠ C ∧ A ≠ C ∧ A * B * C = 2310 ∧ max_sum_ABC A B C = 52 :=
sorry

end largest_sum_ABC_l1084_108416


namespace only_function_B_has_inverse_l1084_108457

-- Definitions based on the problem conditions
def function_A (x : ℝ) : ℝ := 3 - x^2 -- Parabola opening downwards with vertex at (0,3)
def function_B (x : ℝ) : ℝ := x -- Straight line with slope 1 passing through (0,0) and (1,1)
def function_C (x y : ℝ) : Prop := x^2 + y^2 = 4 -- Circle centered at (0,0) with radius 2

-- Theorem stating that only function B has an inverse
theorem only_function_B_has_inverse :
  (∀ y : ℝ, ∃! x : ℝ, function_B x = y) ∧
  (¬∀ y : ℝ, ∃! x : ℝ, function_A x = y) ∧
  (¬∀ y : ℝ, ∃! x : ℝ, ∃ y1 y2 : ℝ, function_C x y1 ∧ function_C x y2 ∧ y1 ≠ y2) :=
  by 
  sorry -- Proof not required

end only_function_B_has_inverse_l1084_108457


namespace binary_to_decimal_and_octal_conversion_l1084_108459

-- Definition of the binary number in question
def bin_num : ℕ := 0b1011

-- The expected decimal equivalent
def dec_num : ℕ := 11

-- The expected octal equivalent
def oct_num : ℤ := 0o13

-- Proof problem statement
theorem binary_to_decimal_and_octal_conversion :
  bin_num = dec_num ∧ dec_num = oct_num := 
by 
  sorry

end binary_to_decimal_and_octal_conversion_l1084_108459


namespace largest_possible_M_l1084_108493

theorem largest_possible_M (x y z : ℝ) (h : x > 0 ∧ y > 0 ∧ z > 0) (h_cond : x * y + y * z + z * x = 1) :
    ∃ M, ∀ x y z, x > 0 ∧ y > 0 ∧ z > 0 ∧ x * y + y * z + z * x = 1 → 
    (x / (1 + yz/x) + y / (1 + zx/y) + z / (1 + xy/z) ≥ M) → 
        M = 3 / (Real.sqrt 3 + 1) :=
by
  sorry        

end largest_possible_M_l1084_108493


namespace camera_sticker_price_l1084_108446

theorem camera_sticker_price (p : ℝ)
  (h1 : p > 0)
  (hx : ∀ x, x = 0.80 * p - 50)
  (hy : ∀ y, y = 0.65 * p)
  (hs : 0.80 * p - 50 = 0.65 * p - 40) :
  p = 666.67 :=
by sorry

end camera_sticker_price_l1084_108446


namespace vanessa_phone_pictures_l1084_108413

theorem vanessa_phone_pictures
  (C : ℕ) (P : ℕ) (hC : C = 7)
  (hAlbums : 5 * 6 = 30)
  (hTotal : 30 = P + C) :
  P = 23 := by
  sorry

end vanessa_phone_pictures_l1084_108413


namespace y_intercept_of_line_l1084_108497

theorem y_intercept_of_line (x y : ℝ) (eq : 2 * x - 3 * y = 6) (hx : x = 0) : y = -2 := 
by
  sorry

end y_intercept_of_line_l1084_108497


namespace union_sets_l1084_108426

def A : Set ℕ := {1, 2}
def B : Set ℕ := {2, 4, 6}

theorem union_sets : A ∪ B = {1, 2, 4, 6} := by
  sorry

end union_sets_l1084_108426


namespace investment_return_l1084_108427

theorem investment_return 
  (investment1 : ℝ) (investment2 : ℝ) 
  (return1 : ℝ) (combined_return_percent : ℝ) : 
  investment1 = 500 → 
  investment2 = 1500 → 
  return1 = 0.07 → 
  combined_return_percent = 0.085 → 
  (500 * 0.07 + 1500 * r = 2000 * 0.085) → 
  r = 0.09 :=
by
  intros h1 h2 h3 h4 h5
  sorry

end investment_return_l1084_108427


namespace lines_forming_angle_bamboo_pole_longest_shadow_angle_l1084_108495

-- Define the angle between sunlight and ground
def angle_sunlight_ground : ℝ := 60

-- Proof problem 1 statement
theorem lines_forming_angle (A : ℝ) : 
  (A > angle_sunlight_ground → ∃ l : ℕ, l = 0) ∧ (A < angle_sunlight_ground → ∃ l : ℕ, ∀ n : ℕ, n > l) :=
  sorry

-- Proof problem 2 statement
theorem bamboo_pole_longest_shadow_angle : 
  ∀ bamboo_pole_angle ground_angle : ℝ, 
  (ground_angle = 60 → bamboo_pole_angle = 30) :=
  sorry

end lines_forming_angle_bamboo_pole_longest_shadow_angle_l1084_108495


namespace abs_value_equation_l1084_108483

-- Define the main proof problem
theorem abs_value_equation (a b c d : ℝ)
  (h : ∀ x : ℝ, |2 * x + 4| + |a * x + b| = |c * x + d|) :
  d = 2 * c :=
sorry -- Proof skipped for this exercise

end abs_value_equation_l1084_108483


namespace percentage_reduction_l1084_108432

variable (C S newS newC : ℝ)
variable (P : ℝ)
variable (hC : C = 50)
variable (hS : S = 1.25 * C)
variable (hNewS : newS = S - 10.50)
variable (hGain30 : newS = 1.30 * newC)
variable (hNewC : newC = C - P * C)

theorem percentage_reduction (C S newS newC : ℝ) (hC : C = 50) 
  (hS : S = 1.25 * C) (hNewS : newS = S - 10.50) 
  (hGain30 : newS = 1.30 * newC) 
  (hNewC : newC = C - P * C) : 
  P = 0.20 :=
by
  sorry

end percentage_reduction_l1084_108432


namespace complement_union_correct_l1084_108468

def U : Set ℕ := {0, 1, 3, 5, 6, 8}
def A : Set ℕ := {1, 5, 8}
def B : Set ℕ := {2}

theorem complement_union_correct :
  ((U \ A) ∪ B) = {0, 2, 3, 6} :=
by
  sorry

end complement_union_correct_l1084_108468


namespace divisible_by_12_l1084_108480

theorem divisible_by_12 (n : ℤ) : 12 ∣ (n^4 - n^2) := sorry

end divisible_by_12_l1084_108480


namespace find_k_check_divisibility_l1084_108438

-- Define the polynomial f(x) as 2x^3 - 8x^2 + kx - 10
def f (x k : ℝ) : ℝ := 2 * x^3 - 8 * x^2 + k * x - 10

-- Define the polynomial g(x) as 2x^3 - 8x^2 + 13x - 10 after finding k = 13
def g (x : ℝ) : ℝ := 2 * x^3 - 8 * x^2 + 13 * x - 10

-- The first proof problem: Finding k
theorem find_k : (f 2 k = 0) → k = 13 := 
sorry

-- The second proof problem: Checking divisibility by 2x^2 - 1
theorem check_divisibility : ¬ (∃ h : ℝ → ℝ, g x = (2 * x^2 - 1) * h x) := 
sorry

end find_k_check_divisibility_l1084_108438


namespace initial_dimes_l1084_108487

theorem initial_dimes (dimes_received_from_dad : ℕ) (dimes_received_from_mom : ℕ) (total_dimes_now : ℕ) : 
  dimes_received_from_dad = 8 → dimes_received_from_mom = 4 → total_dimes_now = 19 → 
  total_dimes_now - (dimes_received_from_dad + dimes_received_from_mom) = 7 :=
by
  intros
  sorry

end initial_dimes_l1084_108487


namespace total_cards_across_decks_l1084_108436

-- Conditions
def DeckA_cards : ℕ := 52
def DeckB_cards : ℕ := 40
def DeckC_cards : ℕ := 50
def DeckD_cards : ℕ := 48

-- Question as a statement
theorem total_cards_across_decks : (DeckA_cards + DeckB_cards + DeckC_cards + DeckD_cards = 190) := by
  sorry

end total_cards_across_decks_l1084_108436


namespace tens_digit_of_9_pow_2023_l1084_108474

theorem tens_digit_of_9_pow_2023 : (9 ^ 2023) % 100 / 10 = 2 :=
by sorry

end tens_digit_of_9_pow_2023_l1084_108474


namespace domain_of_function_l1084_108405

theorem domain_of_function :
  {x : ℝ | (x + 1 ≥ 0) ∧ (2 - x ≠ 0)} = {x : ℝ | -1 ≤ x ∧ x ≠ 2} :=
by {
  sorry
}

end domain_of_function_l1084_108405


namespace factor_expression_l1084_108448

theorem factor_expression (x : ℝ) : 92 * x^3 - 184 * x^6 = 92 * x^3 * (1 - 2 * x^3) :=
by
  sorry

end factor_expression_l1084_108448


namespace jessicas_score_l1084_108463

theorem jessicas_score (average_20 : ℕ) (average_21 : ℕ) (n : ℕ) (jessica_score : ℕ) 
  (h1 : average_20 = 75)
  (h2 : average_21 = 76)
  (h3 : n = 20)
  (h4 : jessica_score = (average_21 * (n + 1)) - (average_20 * n)) :
  jessica_score = 96 :=
by 
  sorry

end jessicas_score_l1084_108463


namespace final_sign_is_minus_l1084_108482

theorem final_sign_is_minus 
  (plus_count : ℕ) 
  (minus_count : ℕ) 
  (h_plus : plus_count = 2004) 
  (h_minus : minus_count = 2005) 
  (transform : (ℕ → ℕ → ℕ × ℕ) → Prop) :
  transform (fun plus minus =>
    if plus >= 2 then (plus - 1, minus)
    else if minus >= 2 then (plus, minus - 1)
    else if plus > 0 && minus > 0 then (plus - 1, minus - 1)
    else (0, 0)) →
  (plus_count = 0 ∧ minus_count = 1) := sorry

end final_sign_is_minus_l1084_108482


namespace jean_pairs_of_pants_l1084_108447

theorem jean_pairs_of_pants
  (retail_price : ℝ)
  (discount_rate : ℝ)
  (tax_rate : ℝ)
  (total_paid : ℝ)
  (number_of_pairs : ℝ)
  (h1 : retail_price = 45)
  (h2 : discount_rate = 0.20)
  (h3 : tax_rate = 0.10)
  (h4 : total_paid = 396)
  (h5 : number_of_pairs = total_paid / ((retail_price * (1 - discount_rate)) * (1 + tax_rate))) :
  number_of_pairs = 10 :=
by
  sorry

end jean_pairs_of_pants_l1084_108447


namespace power_quotient_l1084_108404

theorem power_quotient (a m n : ℕ) (h_a : a = 19) (h_m : m = 11) (h_n : n = 8) : a^m / a^n = 6859 := by
  sorry

end power_quotient_l1084_108404


namespace more_elements_in_set_N_l1084_108451

theorem more_elements_in_set_N 
  (M N : Finset ℕ) 
  (h_partition : ∀ x, x ∈ M ∨ x ∈ N) 
  (h_disjoint : ∀ x, x ∈ M → x ∉ N) 
  (h_total_2000 : M.card + N.card = 10^2000 - 10^1999) 
  (h_total_1000 : (10^1000 - 10^999) * (10^1000 - 10^999) < 10^2000 - 10^1999) : 
  N.card > M.card :=
by { sorry }

end more_elements_in_set_N_l1084_108451


namespace negation_proposition_l1084_108412

theorem negation_proposition :
  (¬ (∀ x : ℝ, 1 ≤ x ∧ x ≤ 2 → x^2 - 3 * x + 2 ≤ 0)) =
  (∃ x : ℝ, 1 ≤ x ∧ x ≤ 2 ∧ x^2 - 3 * x + 2 > 0) := 
sorry

end negation_proposition_l1084_108412


namespace Julio_limes_expense_l1084_108400

/-- Julio's expense on limes after 30 days --/
theorem Julio_limes_expense :
  ((30 * (1 / 2)) / 3) * 1 = 5 := 
by
  sorry

end Julio_limes_expense_l1084_108400


namespace quadratic_roots_range_l1084_108465

variable (a : ℝ)

theorem quadratic_roots_range (h : ∀ b c (eq : b = -a ∧ c = a^2 - 4), ∃ x y, x ≠ y ∧ x^2 + b * x + c = 0 ∧ x > 0 ∧ y^2 + b * y + c = 0) :
  -2 ≤ a ∧ a ≤ 2 :=
by sorry

end quadratic_roots_range_l1084_108465


namespace intersection_three_points_l1084_108494

def circle_eq (a : ℝ) (x y : ℝ) : Prop := x^2 + y^2 = a^2
def parabola_eq (a : ℝ) (x y : ℝ) : Prop := y = x^2 - 3 * a

theorem intersection_three_points (a : ℝ) :
  (∃ x1 y1 x2 y2 x3 y3 : ℝ, 
    circle_eq a x1 y1 ∧ parabola_eq a x1 y1 ∧
    circle_eq a x2 y2 ∧ parabola_eq a x2 y2 ∧
    circle_eq a x3 y3 ∧ parabola_eq a x3 y3 ∧
    (x1 ≠ x2 ∨ y1 ≠ y2) ∧
    (x1 ≠ x3 ∨ y1 ≠ y3) ∧
    (x2 ≠ x3 ∨ y2 ≠ y3)) ↔ 
  a = 1/3 := by
  sorry

end intersection_three_points_l1084_108494


namespace max_expression_value_l1084_108414

theorem max_expression_value (x y z : ℕ) (hx : 10 ≤ x ∧ x < 100) (hy : 10 ≤ y ∧ y < 100) (hz : 10 ≤ z ∧ z < 100) (hmean : (x + y + z) / 3 = 60) :
  (x + y) / z ≤ 17 :=
sorry

end max_expression_value_l1084_108414


namespace solution_part_1_solution_part_2_l1084_108458

def cost_price_of_badges (x y : ℕ) : Prop :=
  (x - y = 4) ∧ (6 * x = 10 * y)

theorem solution_part_1 (x y : ℕ) :
  cost_price_of_badges x y → x = 10 ∧ y = 6 :=
by
  sorry

def maximizing_profit (m : ℕ) (w : ℕ) : Prop :=
  (10 * m + 6 * (400 - m) ≤ 2800) ∧ (w = m + 800)

theorem solution_part_2 (m : ℕ) :
  maximizing_profit m 900 → m = 100 :=
by
  sorry


end solution_part_1_solution_part_2_l1084_108458


namespace part1_a_eq_neg1_inter_part1_a_eq_neg1_union_part2_a_range_l1084_108491

def A (x : ℝ) : Prop := x^2 - 4 * x - 5 ≥ 0
def B (x a : ℝ) : Prop := 2 * a ≤ x ∧ x ≤ a + 2

theorem part1_a_eq_neg1_inter (a : ℝ) (h : a = -1) : 
  {x : ℝ | A x} ∩ {x : ℝ | B x a} = {x : ℝ | -2 ≤ x ∧ x ≤ -1} :=
by sorry

theorem part1_a_eq_neg1_union (a : ℝ) (h : a = -1) : 
  {x : ℝ | A x} ∪ {x : ℝ | B x a} = {x : ℝ | x ≤ 1 ∨ x ≥ 5} :=
by sorry

theorem part2_a_range (a : ℝ) : 
  ({x : ℝ | A x} ∩ {x : ℝ | B x a} = {x : ℝ | B x a}) → 
  a ∈ {a : ℝ | a > 2 ∨ a ≤ -3} :=
by sorry

end part1_a_eq_neg1_inter_part1_a_eq_neg1_union_part2_a_range_l1084_108491


namespace car_z_mpg_decrease_l1084_108403

theorem car_z_mpg_decrease :
  let mpg_45 := 51
  let mpg_60 := 408 / 10
  let decrease := mpg_45 - mpg_60
  let percentage_decrease := (decrease / mpg_45) * 100
  percentage_decrease = 20 := by
  sorry

end car_z_mpg_decrease_l1084_108403


namespace payment_to_C_l1084_108425

theorem payment_to_C (A_days B_days total_payment days_taken : ℕ) 
  (A_work_rate B_work_rate : ℚ)
  (work_fraction_by_A_and_B : ℚ)
  (remaining_work_fraction_by_C : ℚ)
  (C_payment : ℚ) :
  A_days = 6 →
  B_days = 8 →
  total_payment = 3360 →
  days_taken = 3 →
  A_work_rate = 1/6 →
  B_work_rate = 1/8 →
  work_fraction_by_A_and_B = (A_work_rate + B_work_rate) * days_taken →
  remaining_work_fraction_by_C = 1 - work_fraction_by_A_and_B →
  C_payment = total_payment * remaining_work_fraction_by_C →
  C_payment = 420 := 
by
  intros hA hB hTP hD hAR hBR hWF hRWF hCP
  sorry

end payment_to_C_l1084_108425


namespace jen_hours_per_week_l1084_108475

theorem jen_hours_per_week (B : ℕ) (h1 : ∀ t : ℕ, t * (B + 7) = 6 * B) : B + 7 = 21 := by
  sorry

end jen_hours_per_week_l1084_108475


namespace question1_question2_question3_question4_l1084_108489

theorem question1 : (2 * 3) ^ 2 = 2 ^ 2 * 3 ^ 2 := by admit

theorem question2 : (-1 / 2 * 2) ^ 3 = (-1 / 2) ^ 3 * 2 ^ 3 := by admit

theorem question3 : (3 / 2) ^ 2019 * (-2 / 3) ^ 2019 = -1 := by admit

theorem question4 (a b : ℝ) (n : ℕ) (h : 0 < n): (a * b) ^ n = a ^ n * b ^ n := by admit

end question1_question2_question3_question4_l1084_108489


namespace return_trip_avg_speed_l1084_108479

noncomputable def avg_speed_return_trip : ℝ := 
  let distance_ab_to_sy := 120
  let rate_ab_to_sy := 50
  let total_time := 5.5
  let time_ab_to_sy := distance_ab_to_sy / rate_ab_to_sy
  let time_return_trip := total_time - time_ab_to_sy
  distance_ab_to_sy / time_return_trip

theorem return_trip_avg_speed 
  (distance_ab_to_sy : ℝ := 120)
  (rate_ab_to_sy : ℝ := 50)
  (total_time : ℝ := 5.5) 
  : avg_speed_return_trip = 38.71 :=
by
  sorry

end return_trip_avg_speed_l1084_108479


namespace min_balls_to_draw_l1084_108461

theorem min_balls_to_draw {red green yellow blue white black : ℕ} 
    (h_red : red = 28) 
    (h_green : green = 20) 
    (h_yellow : yellow = 19) 
    (h_blue : blue = 13) 
    (h_white : white = 11) 
    (h_black : black = 9) :
    ∃ n, n = 76 ∧ 
    (∀ drawn, (drawn < n → (drawn ≤ 14 + 14 + 14 + 13 + 11 + 9)) ∧ (drawn >= n → (∃ c, c ≥ 15))) :=
sorry

end min_balls_to_draw_l1084_108461


namespace tangent_line_at_one_minimum_a_range_of_a_l1084_108486

-- Definitions for the given functions
def g (a x : ℝ) := a * x^2 - (a + 2) * x
noncomputable def h (x : ℝ) := Real.log x
noncomputable def f (a x : ℝ) := g a x + h x

-- Part (1): Prove the tangent line equation at x = 1 for a = 1
theorem tangent_line_at_one (x y : ℝ) (h_x : x = 1) (h_a : 1 = (1 : ℝ)) :
  x + y + 1 = 0 := by
  sorry

-- Part (2): Prove the minimum value of a given certain conditions
theorem minimum_a (a : ℝ) (h_a_pos : 0 < a) (h_x : 1 ≤ x ∧ x ≤ Real.exp 1)
  (h_fmin : ∀ x, f a x ≥ -2) : 
  a = 1 := by
  sorry

-- Part (3): Prove the range of values for a given a condition
theorem range_of_a (a x₁ x₂ : ℝ) (h_x : 0 < x₁ ∧ x₁ < x₂) 
  (h_f : ∀ x₁ x₂, (f a x₁ - f a x₂) / (x₁ - x₂) > -2) :
  0 ≤ a ∧ a ≤ 8 := by
  sorry

end tangent_line_at_one_minimum_a_range_of_a_l1084_108486


namespace smaug_silver_coins_l1084_108418

theorem smaug_silver_coins :
  ∀ (num_gold num_copper num_silver : ℕ)
  (value_per_silver value_per_gold conversion_factor value_total : ℕ),
  num_gold = 100 →
  num_copper = 33 →
  value_per_silver = 8 →
  value_per_gold = 3 →
  conversion_factor = value_per_gold * value_per_silver →
  value_total = 2913 →
  (num_gold * conversion_factor + num_silver * value_per_silver + num_copper = value_total) →
  num_silver = 60 :=
by
  intros num_gold num_copper num_silver value_per_silver value_per_gold conversion_factor value_total
  intros h1 h2 h3 h4 h5 h6 h_eq
  sorry

end smaug_silver_coins_l1084_108418


namespace quadrilateral_area_l1084_108428

variable (d : ℝ) (o₁ : ℝ) (o₂ : ℝ)

theorem quadrilateral_area (h₁ : d = 28) (h₂ : o₁ = 8) (h₃ : o₂ = 2) : 
  (1 / 2 * d * o₁) + (1 / 2 * d * o₂) = 140 := 
  by
    rw [h₁, h₂, h₃]
    sorry

end quadrilateral_area_l1084_108428


namespace grain_output_scientific_notation_l1084_108429

theorem grain_output_scientific_notation :
    682.85 * 10^6 = 6.8285 * 10^8 := 
by sorry

end grain_output_scientific_notation_l1084_108429


namespace statement_2_statement_4_l1084_108434

-- Definitions and conditions
variables {Point Line Plane : Type}
variable (a b : Line)
variable (α : Plane)

def parallel (l1 l2 : Line) : Prop := sorry  -- Define parallel relation
def perp (l1 l2 : Line) : Prop := sorry  -- Define perpendicular relation
def perp_plane (l : Line) (p : Plane) : Prop := sorry  -- Define line-plane perpendicular relation
def lies_in (l : Line) (p : Plane) : Prop := sorry  -- Define line lies in plane relation

-- Problem statement 2: If a ∥ b and a ⟂ α, then b ⟂ α
theorem statement_2 (h1 : parallel a b) (h2 : perp_plane a α) : perp_plane b α := sorry

-- Problem statement 4: If a ⟂ α and b ⟂ a, then a ∥ b
theorem statement_4 (h1 : perp_plane a α) (h2 : perp b a) : parallel a b := sorry

end statement_2_statement_4_l1084_108434


namespace factor_expression_l1084_108439

theorem factor_expression :
  (8 * x ^ 4 + 34 * x ^ 3 - 120 * x + 150) - (-2 * x ^ 4 + 12 * x ^ 3 - 5 * x + 10) 
  = 5 * x * (2 * x ^ 3 + (22 / 5) * x ^ 2 - 23 * x + 28) :=
sorry

end factor_expression_l1084_108439


namespace train_pass_tree_l1084_108423

theorem train_pass_tree
  (L : ℝ) (S : ℝ) (conv_factor : ℝ) 
  (hL : L = 275)
  (hS : S = 90)
  (hconv : conv_factor = 5 / 18) :
  L / (S * conv_factor) = 11 :=
by
  sorry

end train_pass_tree_l1084_108423


namespace fraction_identity_l1084_108420

theorem fraction_identity (a b : ℝ) (h1 : 1/a + 2/b = 1) (h2 : a ≠ -b) : 
  (ab - a)/(a + b) = 1 := 
by 
  sorry

end fraction_identity_l1084_108420


namespace min_packages_l1084_108476

theorem min_packages (p : ℕ) (N : ℕ) :
  (N = 19 * p) →
  (N % 7 = 4) →
  (N % 11 = 1) →
  p = 40 :=
by
  sorry

end min_packages_l1084_108476


namespace find_q_l1084_108454

theorem find_q (p q : ℝ) (h1 : 1 < p) (h2 : p < q) (h3 : 1 / p + 1 / q = 1) (h4 : p * q = 8) :
  q = 4 + 2 * Real.sqrt 2 :=
by
  sorry

end find_q_l1084_108454


namespace find_a_l1084_108455

noncomputable def a_value_given_conditions : ℝ :=
  let A := 30 * Real.pi / 180
  let C := 105 * Real.pi / 180
  let B := 180 * Real.pi / 180 - A - C
  let b := 8
  let a := (b * Real.sin A) / Real.sin B
  a

theorem find_a :
  a_value_given_conditions = 4 * Real.sqrt 2 :=
by
  -- We assume that the value computation as specified is correct
  -- hence this is just stating the problem.
  sorry

end find_a_l1084_108455


namespace germs_per_dish_l1084_108408

theorem germs_per_dish (total_germs : ℝ) (num_dishes : ℝ) 
(h1 : total_germs = 5.4 * 10^6) 
(h2 : num_dishes = 10800) : total_germs / num_dishes = 502 :=
sorry

end germs_per_dish_l1084_108408


namespace min_height_bounces_l1084_108477

noncomputable def geometric_sequence (a r: ℝ) (n: ℕ) : ℝ := 
  a * r^n

theorem min_height_bounces (k : ℕ) : 
  ∀ k, 20 * (2 / 3 : ℝ) ^ k < 3 → k ≥ 7 := 
by
  sorry

end min_height_bounces_l1084_108477


namespace dominic_domino_problem_l1084_108488

theorem dominic_domino_problem 
  (num_dominoes : ℕ)
  (pips_pairs : ℕ → ℕ)
  (hexagonal_ring : ℕ → ℕ → Prop) : 
  ∀ (adj : ℕ → ℕ → Prop), 
  num_dominoes = 6 → 
  (∀ i j, hexagonal_ring i j → pips_pairs i = pips_pairs j) →
  ∃ k, k = 2 :=
by {
  sorry
}

end dominic_domino_problem_l1084_108488


namespace blue_string_length_l1084_108452

def length_red := 8
def length_white := 5 * length_red
def length_blue := length_white / 8

theorem blue_string_length : length_blue = 5 := by
  sorry

end blue_string_length_l1084_108452


namespace number_of_ways_to_choose_one_book_l1084_108421

-- Defining the conditions
def num_chinese_books : ℕ := 5
def num_math_books : ℕ := 4

-- Statement of the theorem
theorem number_of_ways_to_choose_one_book : num_chinese_books + num_math_books = 9 :=
by
  -- Skipping the proof as instructed
  sorry

end number_of_ways_to_choose_one_book_l1084_108421


namespace interior_diagonals_sum_l1084_108485

theorem interior_diagonals_sum (a b c : ℝ) 
  (h1 : 4 * (a + b + c) = 52)
  (h2 : 2 * (a * b + b * c + c * a) = 118) :
  4 * Real.sqrt (a^2 + b^2 + c^2) = 4 * Real.sqrt 51 := 
by
  sorry

end interior_diagonals_sum_l1084_108485


namespace apple_production_l1084_108484

variable {S1 S2 S3 : ℝ}

theorem apple_production (h1 : S2 = 0.8 * S1) 
                         (h2 : S3 = 2 * S2) 
                         (h3 : S1 + S2 + S3 = 680) : 
                         S1 = 200 := 
by
  sorry

end apple_production_l1084_108484


namespace project_selection_probability_l1084_108473

/-- Each employee can randomly select one project from four optional assessment projects. -/
def employees : ℕ := 4

def projects : ℕ := 4

def total_events (e : ℕ) (p : ℕ) : ℕ := p^e

def choose_exactly_one_project_not_selected_probability (e : ℕ) (p : ℕ) : ℚ :=
  (Nat.choose p 2 * Nat.factorial 3) / (p^e : ℚ)

theorem project_selection_probability :
  choose_exactly_one_project_not_selected_probability employees projects = 9 / 16 :=
by
  sorry

end project_selection_probability_l1084_108473


namespace gcd_lcm_product_correct_l1084_108440

noncomputable def gcd_lcm_product : ℕ :=
  let a := 90
  let b := 135
  gcd a b * lcm a b

theorem gcd_lcm_product_correct : gcd_lcm_product = 12150 :=
  by
  sorry

end gcd_lcm_product_correct_l1084_108440


namespace wood_stove_afternoon_burn_rate_l1084_108422

-- Conditions extracted as definitions
def morning_burn_rate : ℝ := 2
def morning_duration : ℝ := 4
def initial_wood : ℝ := 30
def final_wood : ℝ := 3
def afternoon_duration : ℝ := 4

-- Theorem statement matching the conditions and correct answer
theorem wood_stove_afternoon_burn_rate :
  let morning_burned := morning_burn_rate * morning_duration
  let total_burned := initial_wood - final_wood
  let afternoon_burned := total_burned - morning_burned
  ∃ R : ℝ, (afternoon_burned = R * afternoon_duration) ∧ (R = 4.75) :=
by
  sorry

end wood_stove_afternoon_burn_rate_l1084_108422


namespace prob_sum_equals_15_is_0_l1084_108417

theorem prob_sum_equals_15_is_0 (coin1 coin2 : ℕ) (die_min die_max : ℕ) (age : ℕ)
  (h1 : coin1 = 5) (h2 : coin2 = 15) (h3 : die_min = 1) (h4 : die_max = 6) (h5 : age = 15) :
  ((coin1 = 5 ∨ coin2 = 15) → die_min ≤ ((if coin1 = 5 then 5 else 15) + (die_max - die_min + 1)) ∧ 
   (die_min ≤ 6) ∧ 6 ≤ die_max) → 
  0 = 0 :=
by
  sorry

end prob_sum_equals_15_is_0_l1084_108417


namespace expressions_same_type_l1084_108464

def same_type_as (e1 e2 : ℕ × ℕ) : Prop :=
  e1 = e2

def exp_of_expr (a_exp b_exp : ℕ) : ℕ × ℕ :=
  (a_exp, b_exp)

def exp_3a2b := exp_of_expr 2 1
def exp_neg_ba2 := exp_of_expr 2 1

theorem expressions_same_type :
  same_type_as exp_neg_ba2 exp_3a2b :=
by
  sorry

end expressions_same_type_l1084_108464


namespace polynomial_inequality_l1084_108431

noncomputable def F (x a_3 a_2 a_1 k : ℝ) : ℝ :=
  x^4 + a_3 * x^3 + a_2 * x^2 + a_1 * x + k^4

theorem polynomial_inequality 
  (p k : ℝ) 
  (a_3 a_2 a_1 : ℝ) 
  (h_p : 0 < p) 
  (h_k : 0 < k) 
  (h_roots : ∃ x1 x2 x3 x4 : ℝ, 0 < x1 ∧ 0 < x2 ∧ 0 < x3 ∧ 0 < x4 ∧
    F (-x1) a_3 a_2 a_1 k = 0 ∧
    F (-x2) a_3 a_2 a_1 k = 0 ∧
    F (-x3) a_3 a_2 a_1 k = 0 ∧
    F (-x4) a_3 a_2 a_1 k = 0) :
  F p a_3 a_2 a_1 k ≥ (p + k)^4 := 
sorry

end polynomial_inequality_l1084_108431


namespace arrests_per_day_in_each_city_l1084_108478

-- Define the known conditions
def daysOfProtest := 30
def numberOfCities := 21
def daysInJailBeforeTrial := 4
def daysInJailAfterTrial := 7 / 2 * 7 -- half of a 2-week sentence in days, converted from weeks to days
def combinedJailTimeInWeeks := 9900
def combinedJailTimeInDays := combinedJailTimeInWeeks * 7

-- Define the proof statement
theorem arrests_per_day_in_each_city :
  (combinedJailTimeInDays / (daysInJailBeforeTrial + daysInJailAfterTrial)) / daysOfProtest / numberOfCities = 10 := 
by
  sorry

end arrests_per_day_in_each_city_l1084_108478


namespace at_least_one_number_greater_than_16000_l1084_108433

theorem at_least_one_number_greater_than_16000 
    (numbers : Fin 20 → ℕ) 
    (h_distinct : Function.Injective numbers)
    (h_square_product : ∀ i : Fin 19, ∃ k : ℕ, numbers i * numbers (i + 1) = k^2)
    (h_first : numbers 0 = 42) :
    ∃ i : Fin 20, numbers i > 16000 :=
by
  sorry

end at_least_one_number_greater_than_16000_l1084_108433


namespace xy_equals_twelve_l1084_108498

theorem xy_equals_twelve (x y : ℝ) (h : x * (x + y) = x^2 + 12) : x * y = 12 :=
by
  sorry

end xy_equals_twelve_l1084_108498


namespace percentage_off_sale_l1084_108430

theorem percentage_off_sale (original_price sale_price : ℝ) (h₁ : original_price = 350) (h₂ : sale_price = 140) :
  ((original_price - sale_price) / original_price) * 100 = 60 :=
by
  sorry

end percentage_off_sale_l1084_108430


namespace calculate_color_cartridges_l1084_108409

theorem calculate_color_cartridges (c b : ℕ) (h1 : 32 * c + 27 * b = 123) (h2 : b ≥ 1) : c = 3 :=
by
  sorry

end calculate_color_cartridges_l1084_108409


namespace inequality_solution_l1084_108471

theorem inequality_solution (x : ℝ) : 
  (∃ (y : ℝ), y = 1 / (3 ^ x) ∧ y * (y - 2) < 15) ↔ x > - (Real.log 5 / Real.log 3) :=
by 
    sorry

end inequality_solution_l1084_108471


namespace angle_E_measure_l1084_108435

theorem angle_E_measure (H F G E : ℝ) 
  (h1 : E = 2 * F) (h2 : F = 2 * G) (h3 : G = 1.25 * H) 
  (h4 : E + F + G + H = 360) : E = 150 := by
  sorry

end angle_E_measure_l1084_108435


namespace speed_in_still_water_l1084_108469

theorem speed_in_still_water (upstream_speed downstream_speed : ℕ) (h_up : upstream_speed = 26) (h_down : downstream_speed = 30) :
  (upstream_speed + downstream_speed) / 2 = 28 := by
  sorry

end speed_in_still_water_l1084_108469


namespace problem_l1084_108453

theorem problem (a b c d : ℝ) (h1 : a - b - c + d = 18) (h2 : a + b - c - d = 6) : (b - d) ^ 2 = 36 :=
by
  sorry

end problem_l1084_108453


namespace outfit_count_l1084_108443

theorem outfit_count (shirts pants ties belts : ℕ) (h_shirts : shirts = 8) (h_pants : pants = 5) (h_ties : ties = 4) (h_belts : belts = 2) :
  shirts * pants * (ties + 1) * (belts + 1) = 600 := by
  sorry

end outfit_count_l1084_108443


namespace proportional_function_y_decreases_l1084_108472

theorem proportional_function_y_decreases (k : ℝ) (h₀ : k ≠ 0) (h₁ : (4 : ℝ) * k = -1) :
  ∀ x₁ x₂ : ℝ, x₁ < x₂ → k * x₁ > k * x₂ :=
by 
  sorry

end proportional_function_y_decreases_l1084_108472


namespace polynomial_condition_satisfied_l1084_108456

-- Definitions as per conditions:
def p (x : ℝ) : ℝ := x^2 + 1

-- Conditions:
axiom cond1 : p 3 = 10
axiom cond2 : ∀ (x y : ℝ), p x * p y = p x + p y + p (x * y) - 2

-- Theorem to prove:
theorem polynomial_condition_satisfied : (p 3 = 10) ∧ (∀ (x y : ℝ), p x * p y = p x + p y + p (x * y) - 2) :=
by
  apply And.intro cond1
  apply cond2

end polynomial_condition_satisfied_l1084_108456


namespace find_a_l1084_108402

theorem find_a (a : ℝ) (h : (a + 3) = 0) : a = -3 :=
by sorry

end find_a_l1084_108402


namespace find_focus_of_parabola_l1084_108406

-- Define the given parabola equation
def parabola_eqn (x : ℝ) : ℝ := -4 * x^2

-- Define a predicate to check if the point is the focus
def is_focus (x y : ℝ) := x = 0 ∧ y = -1 / 16

theorem find_focus_of_parabola :
  is_focus 0 (parabola_eqn 0) :=
sorry

end find_focus_of_parabola_l1084_108406


namespace seventh_place_is_unspecified_l1084_108467

noncomputable def charlie_position : ℕ := 5
noncomputable def emily_position : ℕ := charlie_position + 5
noncomputable def dana_position : ℕ := 10
noncomputable def bob_position : ℕ := dana_position - 2
noncomputable def alice_position : ℕ := emily_position + 3

theorem seventh_place_is_unspecified :
  ∀ (x : ℕ), 1 ≤ x ∧ x ≤ 15 ∧ x ≠ charlie_position ∧ x ≠ emily_position ∧
  x ≠ dana_position ∧ x ≠ bob_position ∧ x ≠ alice_position →
  x = 7 → false := 
by
  sorry

end seventh_place_is_unspecified_l1084_108467


namespace extreme_value_h_at_a_zero_range_of_a_l1084_108441

noncomputable def f (x : ℝ) : ℝ := 1 - Real.exp (-x)
noncomputable def g (x : ℝ) (a : ℝ) : ℝ := x / (a * x + 1)

noncomputable def h (x : ℝ) (a : ℝ) : ℝ := (Real.exp (-x)) * (g x a)

-- Statement for the first proof problem
theorem extreme_value_h_at_a_zero :
  ∀ x : ℝ, h x 0 ≤ 1 / Real.exp 1 :=
sorry

-- Statement for the second proof problem
theorem range_of_a:
  ∀ x : ℝ, (0 ≤ x → x ≤ 1 / 2) → (f x ≤ g x x) :=
sorry

end extreme_value_h_at_a_zero_range_of_a_l1084_108441


namespace letter_150_in_pattern_l1084_108450

-- Define the repeating pattern
def pattern : List Char := ['A', 'B', 'C', 'D']

-- Define the function to get the n-th letter in the infinite repetition of the pattern
def nth_letter_in_pattern (n : Nat) : Char :=
  pattern.get! ((n - 1) % pattern.length)

-- Theorem statement
theorem letter_150_in_pattern : nth_letter_in_pattern 150 = 'B' :=
  sorry

end letter_150_in_pattern_l1084_108450


namespace carl_gave_beth_35_coins_l1084_108499

theorem carl_gave_beth_35_coins (x : ℕ) (h1 : ∃ n, n = 125) (h2 : ∃ m, m = (125 + x) / 2) (h3 : m = 80) : x = 35 :=
by
  sorry

end carl_gave_beth_35_coins_l1084_108499


namespace parabola_focus_distance_l1084_108419

theorem parabola_focus_distance (p : ℝ) (hp : p > 0) (A : ℝ × ℝ)
  (hA_on_parabola : A.2 ^ 2 = 2 * p * A.1)
  (hA_focus_dist : dist A (p / 2, 0) = 12)
  (hA_yaxis_dist : abs A.1 = 9) : p = 6 :=
sorry

end parabola_focus_distance_l1084_108419


namespace seq_form_l1084_108445

-- Define the sequence a as a function from natural numbers to natural numbers
def seq (a : ℕ → ℕ) : Prop :=
  ∀ m n : ℕ, 0 < m → 0 < n → ⌊(a m : ℚ) / a n⌋ = ⌊(m : ℚ) / n⌋

-- Define the statement that all sequences satisfying the condition must be of the form k * i
theorem seq_form (a : ℕ → ℕ) : seq a → ∃ k : ℕ, (0 < k) ∧ (∀ n, 0 < n → a n = k * n) := 
by
  intros h
  sorry

end seq_form_l1084_108445


namespace taxi_fare_distance_l1084_108449

theorem taxi_fare_distance (x : ℝ) : 
  (8 + if x ≤ 3 then 0 else if x ≤ 8 then 2.15 * (x - 3) else 2.15 * 5 + 2.85 * (x - 8)) + 1 = 31.15 → x = 11.98 :=
by 
  sorry

end taxi_fare_distance_l1084_108449


namespace value_of_expression_l1084_108437

theorem value_of_expression (x y z : ℝ) (h : x * y * z = 1) :
  1 / (1 + x + x * y) + 1 / (1 + y + y * z) + 1 / (1 + z + z * x) = 1 :=
sorry

end value_of_expression_l1084_108437


namespace price_per_glass_first_day_l1084_108462

theorem price_per_glass_first_day (O P2 P1: ℝ) (H1 : O > 0) (H2 : P2 = 0.2) (H3 : 2 * O * P1 = 3 * O * P2) : P1 = 0.3 :=
by
  sorry

end price_per_glass_first_day_l1084_108462


namespace initial_salmons_l1084_108460

theorem initial_salmons (x : ℕ) (hx : 10 * x = 5500) : x = 550 := 
by
  sorry

end initial_salmons_l1084_108460


namespace sequence_sum_periodic_l1084_108410

theorem sequence_sum_periodic (a : ℕ → ℕ) (a1 a8 : ℕ) :
  a 1 = 11 →
  a 8 = 12 →
  (∀ i, 1 ≤ i → i ≤ 6 → a i + a (i + 1) + a (i + 2) = 50) →
  (a 1 = 11 ∧ a 2 = 12 ∧ a 3 = 27 ∧ a 4 = 11 ∧ a 5 = 12 ∧ a 6 = 27 ∧ a 7 = 11 ∧ a 8 = 12) :=
by
  intros h1 h8 hsum
  sorry

end sequence_sum_periodic_l1084_108410


namespace positive_solution_l1084_108492

variable {x y z : ℝ}

theorem positive_solution (h1 : x * y = 8 - 2 * x - 3 * y)
    (h2 : y * z = 8 - 4 * y - 2 * z)
    (h3 : x * z = 40 - 5 * x - 3 * z) :
    x = 10 := by
  sorry

end positive_solution_l1084_108492


namespace ice_cream_vendor_l1084_108481

theorem ice_cream_vendor (choco : ℕ) (mango : ℕ) (sold_choco : ℚ) (sold_mango : ℚ) 
  (h_choco : choco = 50) (h_mango : mango = 54) (h_sold_choco : sold_choco = 3/5) 
  (h_sold_mango : sold_mango = 2/3) : 
  choco - (choco * sold_choco) + mango - (mango * sold_mango) = 38 := 
by 
  sorry

end ice_cream_vendor_l1084_108481

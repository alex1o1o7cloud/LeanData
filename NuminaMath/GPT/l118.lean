import Mathlib

namespace gaeun_taller_than_nana_l118_118601

def nana_height_m : ℝ := 1.618
def gaeun_height_cm : ℝ := 162.3
def nana_height_cm : ℝ := nana_height_m * 100

theorem gaeun_taller_than_nana : gaeun_height_cm - nana_height_cm = 0.5 := by
  sorry

end gaeun_taller_than_nana_l118_118601


namespace mean_of_combined_sets_l118_118026

theorem mean_of_combined_sets 
  (mean1 mean2 mean3 : ℚ)
  (count1 count2 count3 : ℕ)
  (h1 : mean1 = 15)
  (h2 : mean2 = 20)
  (h3 : mean3 = 12)
  (hc1 : count1 = 7)
  (hc2 : count2 = 8)
  (hc3 : count3 = 5) :
  ((count1 * mean1 + count2 * mean2 + count3 * mean3) / (count1 + count2 + count3)) = 16.25 :=
by
  sorry

end mean_of_combined_sets_l118_118026


namespace collinear_vectors_sum_l118_118695

theorem collinear_vectors_sum (x y : ℝ) 
  (h1 : ∃ λ : ℝ, (-1, y, 2) = (λ * x, λ * (3 / 2), λ * 3)) : 
  x + y = -1 / 2 :=
sorry

end collinear_vectors_sum_l118_118695


namespace polynomial_solutions_l118_118993

theorem polynomial_solutions (P : Polynomial ℝ) :
  (∀ x : ℝ, P.eval x * P.eval (x + 1) = P.eval (x^2 - x + 3)) →
  (P = 0 ∨ ∃ n : ℕ, P = (Polynomial.C 1) * (Polynomial.X^2 - 2 * Polynomial.X + 3)^n) :=
by
  sorry

end polynomial_solutions_l118_118993


namespace inequality_proof_equality_condition_l118_118549

variable {x1 x2 y1 y2 z1 z2 : ℝ}

-- Conditions
axiom x1_pos : x1 > 0
axiom x2_pos : x2 > 0
axiom x1y1_gz1sq : x1 * y1 > z1 ^ 2
axiom x2y2_gz2sq : x2 * y2 > z2 ^ 2

theorem inequality_proof : 
  8 / ((x1 + x2) * (y1 + y2) - (z1 + z2) ^ 2) <= 
  1 / (x1 * y1 - z1 ^ 2) + 1 / (x2 * y2 - z2 ^ 2) :=
sorry

theorem equality_condition : 
  8 / ((x1 + x2) * (y1 + y2) - (z1 + z2) ^ 2) = 
  1 / (x1 * y1 - z1 ^ 2) + 1 / (x2 * y2 - z2 ^ 2) ↔ 
  (x1 = x2 ∧ y1 = y2 ∧ z1 = z2) :=
sorry

end inequality_proof_equality_condition_l118_118549


namespace relationship_between_a_and_b_l118_118845

variable (a b : ℝ)

def in_interval (x : ℝ) := 0 < x ∧ x < 1

theorem relationship_between_a_and_b 
  (ha : in_interval a)
  (hb : in_interval b)
  (h : (1 - a) * b > 1 / 4) : a < b :=
sorry

end relationship_between_a_and_b_l118_118845


namespace original_cost_is_49_l118_118239

-- Define the conditions as assumptions
def original_cost_of_jeans (x : ℝ) : Prop :=
  let discounted_price := x / 2
  let wednesday_price := discounted_price - 10
  wednesday_price = 14.5

-- The theorem to prove
theorem original_cost_is_49 :
  ∃ x : ℝ, original_cost_of_jeans x ∧ x = 49 :=
by
  sorry

end original_cost_is_49_l118_118239


namespace kelly_chickens_l118_118741

theorem kelly_chickens
  (chicken_egg_rate : ℕ)
  (chickens : ℕ)
  (egg_price_per_dozen : ℕ)
  (total_money : ℕ)
  (weeks : ℕ)
  (days_per_week : ℕ)
  (dozen : ℕ)
  (total_eggs_sold : ℕ)
  (total_days : ℕ)
  (total_eggs_laid : ℕ) : 
  chicken_egg_rate = 3 →
  egg_price_per_dozen = 5 →
  total_money = 280 →
  weeks = 4 →
  days_per_week = 7 →
  dozen = 12 →
  total_eggs_sold = total_money / egg_price_per_dozen * dozen →
  total_days = weeks * days_per_week →
  total_eggs_laid = chickens * chicken_egg_rate * total_days →
  total_eggs_sold = total_eggs_laid →
  chickens = 8 :=
by
  intros
  sorry

end kelly_chickens_l118_118741


namespace div_problem_l118_118611

variables (A B C : ℝ)

theorem div_problem (h1 : A = (2/3) * B) (h2 : B = (1/4) * C) (h3 : A + B + C = 527) : B = 93 :=
by {
  sorry
}

end div_problem_l118_118611


namespace days_to_finish_together_l118_118975

-- Define the work rate of B
def work_rate_B : ℚ := 1 / 12

-- Define the work rate of A
def work_rate_A : ℚ := 2 * work_rate_B

-- Combined work rate of A and B
def combined_work_rate : ℚ := work_rate_A + work_rate_B

-- Prove that the number of days required for A and B to finish the work together is 4
theorem days_to_finish_together : (1 / combined_work_rate) = 4 := 
by
  sorry

end days_to_finish_together_l118_118975


namespace rectangle_ratio_l118_118609

theorem rectangle_ratio (a b c d : ℝ)
  (h1 : (a * b) / (c * d) = 0.16)
  (h2 : a / c = b / d) :
  a / c = 0.4 ∧ b / d = 0.4 :=
by 
  sorry

end rectangle_ratio_l118_118609


namespace must_divide_a_l118_118284

-- Definitions of positive integers and their gcd conditions
variables {a b c d : ℕ}

-- The conditions given in the problem
axiom h1 : gcd a b = 24
axiom h2 : gcd b c = 36
axiom h3 : gcd c d = 54
axiom h4 : 70 < gcd d a ∧ gcd d a < 100

-- We need to prove that 13 divides a
theorem must_divide_a : 13 ∣ a :=
by sorry

end must_divide_a_l118_118284


namespace find_m_l118_118837

variables (a b : ℝ) (x y m : ℝ) (A B F1 F2 : ℝ × ℝ)

-- Given conditions
def ellipse := (x^2 / 3) + y^2 = 1
def line := y = x + m
def foci_F1 := F1 = (-sqrt 2, 0)
def foci_F2 := F2 = (sqrt 2, 0)
def area_F1_AB := λ A B : ℝ × ℝ, 2 * |det(F1, A, B)| = |det(F2, A, B)|

-- The proof problem statement
theorem find_m : 
  ellipse ∧ 
  line ∧ 
  foci_F1 ∧ 
  foci_F2 ∧ 
  (area_F1_AB A B) →
  m = -sqrt(2) / 3 := sorry

end find_m_l118_118837


namespace time_left_for_exercises_l118_118870

theorem time_left_for_exercises (total_minutes : ℕ) (piano_minutes : ℕ) (writing_minutes : ℕ) (reading_minutes : ℕ) : 
  total_minutes = 120 ∧ piano_minutes = 30 ∧ writing_minutes = 25 ∧ reading_minutes = 38 → 
  total_minutes - (piano_minutes + writing_minutes + reading_minutes) = 27 :=
by
  intro h
  cases h with h_total h
  cases h with h_piano h
  cases h with h_writing h_reading
  rw [h_total, h_piano, h_writing, h_reading]
  exactly rfl

end time_left_for_exercises_l118_118870


namespace jenna_peeled_potatoes_l118_118401

-- Definitions of constants
def initial_potatoes : ℕ := 60
def homer_rate : ℕ := 4
def jenna_rate : ℕ := 6
def combined_rate : ℕ := homer_rate + jenna_rate
def homer_time : ℕ := 6
def remaining_potatoes : ℕ := initial_potatoes - (homer_rate * homer_time)
def combined_time : ℕ := 4 -- Rounded from 3.6

-- Statement to prove
theorem jenna_peeled_potatoes : remaining_potatoes / combined_rate * jenna_rate = 24 :=
by
  sorry

end jenna_peeled_potatoes_l118_118401


namespace fraction_decimal_comparison_l118_118316

theorem fraction_decimal_comparison :
  (1 / 3 : ℚ) = (3333 / 10000 : ℚ) + (1 / 10000 : ℚ) :=
by
  sorry

end fraction_decimal_comparison_l118_118316


namespace geometric_sequence_common_ratio_l118_118394

theorem geometric_sequence_common_ratio (a : ℕ → ℤ) (q : ℤ)  
  (h1 : a 1 = 3) 
  (h2 : ∀ n : ℕ, a (n + 1) = a n * q) 
  (h3 : 4 * a 1 + a 3 = 4 * a 2) : 
  q = 2 := 
by {
  -- Proof is omitted here
  sorry
}

end geometric_sequence_common_ratio_l118_118394


namespace quadrant_of_angle_l118_118391

-- Definitions for conditions
def sin_pos_cos_pos (α : ℝ) : Prop := (Real.sin α) * (Real.cos α) > 0

-- The theorem to prove
theorem quadrant_of_angle (α : ℝ) (h : sin_pos_cos_pos α) : 
  (0 < α ∧ α < π / 2) ∨ (π < α ∧ α < 3 * π / 2) :=
sorry

end quadrant_of_angle_l118_118391


namespace roots_sum_powers_l118_118428

theorem roots_sum_powers (t : ℕ → ℝ) (b d f : ℝ)
  (ht0 : t 0 = 3)
  (ht1 : t 1 = 6)
  (ht2 : t 2 = 11)
  (hrec : ∀ k ≥ 2, t (k + 1) = b * t k + d * t (k - 1) + f * t (k - 2))
  (hpoly : ∀ x : ℝ, x^3 - 6 * x^2 + 11 * x - 6 = 0) :
  b + d + f = 13 :=
sorry

end roots_sum_powers_l118_118428


namespace B_takes_6_days_to_complete_work_alone_l118_118203

theorem B_takes_6_days_to_complete_work_alone 
    (work_duration_A : ℕ) 
    (work_payment : ℚ)
    (work_days_with_C : ℕ) 
    (payment_C : ℚ) 
    (combined_work_rate_A_B_C : ℚ)
    (amount_to_be_shared_A_B : ℚ) 
    (combined_daily_earning_A_B : ℚ) :
  work_duration_A = 6 ∧
  work_payment = 3360 ∧ 
  work_days_with_C = 3 ∧ 
  payment_C = 420.00000000000017 ∧ 
  combined_work_rate_A_B_C = 1 / 3 ∧ 
  amount_to_be_shared_A_B = 2940 ∧ 
  combined_daily_earning_A_B = 980 → 
  work_duration_A = 6 ∧
  (∃ (work_duration_B : ℕ), work_duration_B = 6) :=
by 
  sorry

end B_takes_6_days_to_complete_work_alone_l118_118203


namespace four_digit_numbers_count_l118_118263

theorem four_digit_numbers_count :
  ∃ (n : ℕ),  n = 24 ∧
  let s := λ (a b c d : ℕ), (a + b + c + d = 10 ∧ a ≠ 0 ∧ ((a + c) - (b + d)) ∈ [(-11), 0, 11]) in
  (card {abcd : Fin 10 × Fin 10 × Fin 10 × Fin 10 | (s abcd.1.1 abcd.1.2 abcd.2.1 abcd.2.2)}) = n :=
by
  sorry

end four_digit_numbers_count_l118_118263


namespace fraction_of_a_mile_additional_charge_l118_118280

-- Define the conditions
def initial_fee : ℚ := 2.25
def charge_per_fraction : ℚ := 0.25
def total_charge : ℚ := 4.50
def total_distance : ℚ := 3.6

-- Define the problem statement to prove
theorem fraction_of_a_mile_additional_charge :
  initial_fee = 2.25 →
  charge_per_fraction = 0.25 →
  total_charge = 4.50 →
  total_distance = 3.6 →
  total_distance - (total_charge - initial_fee) = 1.35 :=
by
  intros
  sorry

end fraction_of_a_mile_additional_charge_l118_118280


namespace tan_alpha_value_l118_118110

open Real

theorem tan_alpha_value 
  (α : ℝ) 
  (hα_range : 0 < α ∧ α < π) 
  (h_cos_alpha : cos α = -3/5) :
  tan α = -4/3 := 
by
  sorry

end tan_alpha_value_l118_118110


namespace intersection_A_B_l118_118557

def A := {x : ℝ | |x| < 1}
def B := {x : ℝ | -2 < x ∧ x < 0}

theorem intersection_A_B : A ∩ B = {x : ℝ | -1 < x ∧ x < 0} :=
by
  sorry

end intersection_A_B_l118_118557


namespace moles_of_HCl_formed_l118_118995

-- Define the reaction as given in conditions
def reaction (C2H6 Cl2 C2H4Cl2 HCl : ℝ) := C2H6 + Cl2 = C2H4Cl2 + 2 * HCl

-- Define the initial moles of reactants
def moles_C2H6 : ℝ := 2
def moles_Cl2 : ℝ := 2

-- State the expected moles of HCl produced
def expected_moles_HCl : ℝ := 4

-- The theorem stating the problem to prove
theorem moles_of_HCl_formed : ∃ HCl : ℝ, reaction moles_C2H6 moles_Cl2 0 HCl ∧ HCl = expected_moles_HCl :=
by
  -- Skipping detailed proof with sorry
  sorry

end moles_of_HCl_formed_l118_118995


namespace value_of_expression_l118_118716

theorem value_of_expression (x y : ℝ) (h1 : x = -2) (h2 : y = 1/2) :
  y * (x + y) + (x + y) * (x - y) - x^2 = -1 :=
by
  sorry

end value_of_expression_l118_118716


namespace hamburgers_total_l118_118798

theorem hamburgers_total (initial_hamburgers : ℝ) (additional_hamburgers : ℝ) (h₁ : initial_hamburgers = 9.0) (h₂ : additional_hamburgers = 3.0) : initial_hamburgers + additional_hamburgers = 12.0 :=
by
  rw [h₁, h₂]
  norm_num

end hamburgers_total_l118_118798


namespace curve_is_circle_l118_118899

theorem curve_is_circle (ρ θ : ℝ) (h : ρ = 5 * Real.sin θ) : 
  ∃ (C : ℝ × ℝ) (r : ℝ), ∀ (x y : ℝ),
  (x, y) = (ρ * Real.cos θ, ρ * Real.sin θ) → 
  (x - C.1) ^ 2 + (y - C.2) ^ 2 = r ^ 2 :=
by
  existsi (0, 5 / 2), 5 / 2
  sorry

end curve_is_circle_l118_118899


namespace arithmetic_seq_necessary_not_sufficient_l118_118881

noncomputable def arithmetic_sequence (a b c : ℝ) : Prop :=
  a + c = 2 * b

noncomputable def proposition_B (a b c : ℝ) : Prop :=
  b ≠ 0 ∧ (a / b) + (c / b) = 2

theorem arithmetic_seq_necessary_not_sufficient (a b c : ℝ) :
  (arithmetic_sequence a b c → proposition_B a b c) ∧ 
  (∃ a' b' c', arithmetic_sequence a' b' c' ∧ ¬ proposition_B a' b' c') := by
  sorry

end arithmetic_seq_necessary_not_sufficient_l118_118881


namespace volunteer_allocation_scheme_l118_118497

def num_allocation_schemes : ℕ :=
  let num_ways_choose_2_from_5 := Nat.choose 5 2
  let num_ways_arrange_4_groups := Nat.factorial 4
  num_ways_choose_2_from_5 * num_ways_arrange_4_groups

theorem volunteer_allocation_scheme :
  num_allocation_schemes = 240 :=
by
  sorry

end volunteer_allocation_scheme_l118_118497


namespace maximum_value_of_a_exists_a_eq_29_greatest_possible_value_of_a_l118_118315

theorem maximum_value_of_a (x : ℤ) (a : ℤ) (h1 : x^2 + a * x = -28) (h2 : a > 0) : a ≤ 29 := 
by 
-- add proof here 
sorry

theorem exists_a_eq_29 (x : ℤ) (h1 : x^2 + 29 * x = -28) : ∃ (x : ℤ), x^2 + 29 * x = -28 :=
by 
-- add proof here 
sorry

theorem greatest_possible_value_of_a : ∃ (a : ℤ), (∀ x : ℤ, x^2 + a * x = -28 → a ≤ 29) ∧ (∃ x : ℤ, x^2 + 29 * x = -28) := 
by
  use 29
  split
  { intros x h1 
    apply maximum_value_of_a x 29 h1
    show 29 > 0, from nat.succ_pos' 28 }
  { apply exists_a_eq_29 } 

end maximum_value_of_a_exists_a_eq_29_greatest_possible_value_of_a_l118_118315


namespace fraction_sum_l118_118097

theorem fraction_sum : (1/4 : ℚ) + (3/9 : ℚ) = (7/12 : ℚ) := 
  by 
  sorry

end fraction_sum_l118_118097


namespace simon_age_is_10_l118_118503

-- Declare the variables
variable (alvin_age : ℕ) (simon_age : ℕ)

-- Define the conditions
def condition1 : Prop := alvin_age = 30
def condition2 : Prop := simon_age = (alvin_age / 2) - 5

-- Formalize the proof problem
theorem simon_age_is_10 (h1 : condition1) (h2 : condition2) : simon_age = 10 := by
  sorry

end simon_age_is_10_l118_118503


namespace five_cds_cost_with_discount_l118_118780

theorem five_cds_cost_with_discount
  (price_2_cds : ℝ)
  (discount_rate : ℝ)
  (num_cds : ℕ)
  (total_cost : ℝ) 
  (h1 : price_2_cds = 40)
  (h2 : discount_rate = 0.10)
  (h3 : num_cds = 5)
  : total_cost = 90 :=
by
  sorry

end five_cds_cost_with_discount_l118_118780


namespace dividend_is_144_l118_118174

theorem dividend_is_144 
  (Q : ℕ) (D : ℕ) (M : ℕ)
  (h1 : M = 6 * D)
  (h2 : D = 4 * Q) 
  (Q_eq_6 : Q = 6) : 
  M = 144 := 
sorry

end dividend_is_144_l118_118174


namespace charity_event_fund_raising_l118_118477

theorem charity_event_fund_raising :
  let n := 9
  let I := 2000
  let p := 0.10
  let increased_total := I * (1 + p)
  let amount_per_person := increased_total / n
  amount_per_person = 244.44 := by
  sorry

end charity_event_fund_raising_l118_118477


namespace positive_integers_solving_inequality_l118_118826

theorem positive_integers_solving_inequality (n : ℕ) (h1: 0 < n) : 25 - 5 * n < 15 → 2 < n := by
  sorry

end positive_integers_solving_inequality_l118_118826


namespace product_of_two_equal_numbers_l118_118171

theorem product_of_two_equal_numbers :
  ∃ (x : ℕ), (5 * 20 = 12 + 22 + 16 + 2 * x) ∧ (x * x = 625) :=
by
  sorry

end product_of_two_equal_numbers_l118_118171


namespace total_payment_l118_118596

-- Define the basic conditions
def hours_first_day : ℕ := 10
def hours_second_day : ℕ := 8
def hours_third_day : ℕ := 15
def hourly_wage : ℕ := 10
def number_of_workers : ℕ := 2

-- Define the proof problem
theorem total_payment : 
  (hours_first_day + hours_second_day + hours_third_day) * hourly_wage * number_of_workers = 660 := 
by
  sorry

end total_payment_l118_118596


namespace evaluate_72_squared_minus_48_squared_l118_118821

theorem evaluate_72_squared_minus_48_squared :
  (72:ℤ)^2 - (48:ℤ)^2 = 2880 :=
by
  sorry

end evaluate_72_squared_minus_48_squared_l118_118821


namespace total_buckets_poured_l118_118303

-- Define given conditions
def initial_buckets : ℝ := 1
def additional_buckets : ℝ := 8.8

-- Theorem to prove the total number of buckets poured
theorem total_buckets_poured : 
  initial_buckets + additional_buckets = 9.8 :=
by
  sorry

end total_buckets_poured_l118_118303


namespace cannot_form_3x3_square_l118_118063

def square_pieces (squares : ℕ) (rectangles : ℕ) (triangles : ℕ) := 
  squares = 4 ∧ rectangles = 1 ∧ triangles = 1

def area (squares : ℕ) (rectangles : ℕ) (triangles : ℕ) : ℕ := 
  squares * 1 * 1 + rectangles * 2 * 1 + triangles * (1 * 1 / 2)

theorem cannot_form_3x3_square : 
  ∀ squares rectangles triangles, 
  square_pieces squares rectangles triangles → 
  area squares rectangles triangles < 9 := by
  intros squares rectangles triangles h
  unfold square_pieces at h
  unfold area
  sorry

end cannot_form_3x3_square_l118_118063


namespace solution_to_functional_equation_l118_118100

noncomputable def find_functions (f : ℝ → ℝ) : Prop :=
∀ x : ℝ, f x = f (x / 2) + (x / 2) * (deriv f x)

theorem solution_to_functional_equation :
  ∀ (f : ℝ → ℝ), (∀ x : ℝ, f x = f (x / 2) + (x / 2) * (deriv f x)) ↔ (∃ c b : ℝ, ∀ x : ℝ, f x = c * x + b) :=
by {
  sorry
}

end solution_to_functional_equation_l118_118100


namespace num_valid_programs_l118_118215

-- Define the set of courses
def courses := {"English", "Algebra", "Geometry", "History", "Art", "Science", "Latin"}

-- Define the set of mathematics courses
def math_courses := {"Algebra", "Geometry"}

-- Definition of the problem conditions
def is_valid_program (program : set string) : Prop :=
  "English" ∈ program ∧
  (∃ M ⊆ program, M ⊆ math_courses ∧ 2 ≤ M.size) ∧
  program.size = 5

-- The statement of the proof problem
theorem num_valid_programs : set.count {program | program ⊆ courses ∧ is_valid_program program} = 6 :=
sorry

end num_valid_programs_l118_118215


namespace train_speed_before_accident_l118_118967

theorem train_speed_before_accident (d v : ℝ) (hv_pos : v > 0) (hd_pos : d > 0) :
  (d / ((3/4) * v) - d / v = 35 / 60) ∧
  (d - 24) / ((3/4) * v) - (d - 24) / v = 25 / 60 → 
  v = 64 :=
by
  sorry

end train_speed_before_accident_l118_118967


namespace mascots_arrangement_count_l118_118452

-- Define the entities
def bing_dung_dung_mascots := 4
def xue_rong_rong_mascots := 3

-- Define the conditions
def xue_rong_rong_a_and_b_adjacent := true
def xue_rong_rong_c_not_adjacent_to_ab := true

-- Theorem stating the problem and asserting the answer
theorem mascots_arrangement_count : 
  (xue_rong_rong_a_and_b_adjacent ∧ xue_rong_rong_c_not_adjacent_to_ab) →
  (number_of_arrangements = 960) := by
  sorry

end mascots_arrangement_count_l118_118452


namespace triangle_perimeter_l118_118178

theorem triangle_perimeter (a b : ℝ) (f : ℝ → Prop) 
  (h₁ : a = 7) (h₂ : b = 11)
  (eqn : ∀ x, f x ↔ x^2 - 25 = 2 * (x - 5)^2)
  (h₃ : ∃ x, f x ∧ 4 < x ∧ x < 18) :
  ∃ p : ℝ, (p = a + b + 5 ∨ p = a + b + 15) :=
by
  sorry

end triangle_perimeter_l118_118178


namespace matrix_solution_l118_118383

-- Define the 2x2 matrix N
def N : Matrix (Fin 2) (Fin 2) ℚ := 
  ![ ![30 / 7, -13 / 7], 
     ![-6 / 7, -10 / 7] ]

-- Define the vectors
def vec1 : Fin 2 → ℚ := ![2, 3]
def vec2 : Fin 2 → ℚ := ![4, -1]

-- Expected results
def result1 : Fin 2 → ℚ := ![3, -6]
def result2 : Fin 2 → ℚ := ![19, -2]

-- The proof statement
theorem matrix_solution : (N.mulVec vec1 = result1) ∧ (N.mulVec vec2 = result2) :=
  by sorry

end matrix_solution_l118_118383


namespace hexagon_indistinguishable_under_rotation_l118_118859

noncomputable def hexagon_colorings : Nat :=
  let G := CyclicGroup 6
  let fixed_points : Finset Nat := {64, 2, 4, 8, 4, 2}
  (1 / 6 * (64 + 2 + 4 + 8 + 4 + 2))

theorem hexagon_indistinguishable_under_rotation :
  hexagon_colorings = 14 :=
by
  have h : 1 / 6 * (64 + 2 + 4 + 8 + 4 + 2) = 14 :=
    sorry
  exact h

end hexagon_indistinguishable_under_rotation_l118_118859


namespace angle_AMD_deg_l118_118165

open Real

-- Conditions definition
def is_rectangle (A B C D : ℝ × ℝ) (AB BC CD DA : ℝ) := 
  AB = 8 ∧ BC = 4 ∧ (∃ M, M.1 = (A.1 + B.1) / 3 ∧ M.2 = A.2 ∧ ∠ AMD = ∠ CMD ∧ dist M D = 8 / 3)

-- Theorem statement
theorem angle_AMD_deg {A B C D M : ℝ × ℝ} 
  (hR : is_rectangle A B C D 8 4) :
  ∠ AMD = arccos (9 / 16) :=
sorry

end angle_AMD_deg_l118_118165


namespace prime_pairs_square_l118_118382

noncomputable def is_square (n : ℤ) : Prop := ∃ m : ℤ, m * m = n

def is_prime (n : ℤ) : Prop := n > 1 ∧ ∀ m : ℤ, m ∣ n → m = 1 ∨ m = n

theorem prime_pairs_square (a b : ℤ) (ha : is_prime a) (hb : is_prime b) :
  is_square (3 * a^2 * b + 16 * a * b^2) ↔ (a = 19 ∧ b = 19) ∨ (a = 2 ∧ b = 3) :=
by
  sorry

end prime_pairs_square_l118_118382


namespace evaluate_difference_of_squares_l118_118819
-- Import necessary libraries

-- Define the specific values for a and b
def a : ℕ := 72
def b : ℕ := 48

-- State the theorem to be proved
theorem evaluate_difference_of_squares : a^2 - b^2 = (a + b) * (a - b) ∧ (a + b) * (a - b) = 2880 := 
by
  -- The proof would go here but should be omitted as per directions
  sorry

end evaluate_difference_of_squares_l118_118819


namespace sum_of_consecutive_integers_sqrt_28_l118_118113

theorem sum_of_consecutive_integers_sqrt_28 (a b : ℤ) (h1 : a < b) (h2 : b = a + 1) (h3 : a < sqrt 28) (h4 : sqrt 28 < b) : a + b = 11 := by
  sorry

end sum_of_consecutive_integers_sqrt_28_l118_118113


namespace probability_of_one_head_in_three_flips_l118_118271

open Classical

theorem probability_of_one_head_in_three_flips : 
  ∀ (p : ℝ) (n k : ℕ), p = 0.5 → n = 3 → k = 1 → 
  (Nat.choose n k * p^k * (1 - p)^(n - k)) = 0.375 := 
by 
  intros p n k hp hn hk
  rw [hp, hn, hk]
  norm_num
  sorry

end probability_of_one_head_in_three_flips_l118_118271


namespace mathematicians_correctness_l118_118902

theorem mathematicians_correctness :
  ∃ (scenario1_w1 s_w1 : ℕ) (scenario1_w2 s_w2 : ℕ) (scenario2_w1 s2_w1 : ℕ) (scenario2_w2 s2_w2 : ℕ),
    scenario1_w1 = 4 ∧ s_w1 = 7 ∧ scenario1_w2 = 3 ∧ s_w2 = 5 ∧
    scenario2_w1 = 8 ∧ s2_w1 = 14 ∧ scenario2_w2 = 3 ∧ s2_w2 = 5 ∧
    let total_white1 := scenario1_w1 + scenario1_w2,
        total_choco1 := s_w1 + s_w2,
        prob1 := (total_white1 : ℚ) / total_choco1,
        total_white2 := scenario2_w1 + scenario2_w2,
        total_choco2 := s2_w1 + s2_w2,
        prob2 := (total_white2 : ℚ) / total_choco2,
        prob_box1 := 4 / 7,
        prob_box2 := 3 / 5 in
    (prob1 = 7 / 12 ∧ prob2 = 11 / 19 ∧
    (19 / 35 < prob_box1 ∧ prob_box1 < prob_box2) ∧
    (prob_box1 ≠ 19 / 35 ∧ prob_box1 ≠ 3 / 5)) :=
sorry

end mathematicians_correctness_l118_118902


namespace second_metal_gold_percentage_l118_118346

theorem second_metal_gold_percentage (w_final : ℝ) (p_final : ℝ) (w_part : ℝ) (p_part1 : ℝ) (w_part1 : ℝ) (w_part2 : ℝ)
  (h_w_final : w_final = 12.4) (h_p_final : p_final = 0.5) (h_w_part : w_part = 6.2) (h_p_part1 : p_part1 = 0.6)
  (h_w_part1 : w_part1 = 6.2) (h_w_part2 : w_part2 = 6.2) :
  ∃ p_part2 : ℝ, p_part2 = 0.4 :=
by sorry

end second_metal_gold_percentage_l118_118346


namespace rhombus_area_l118_118898

theorem rhombus_area (d1 d2 : ℝ) (h_d1 : d1 = 5) (h_d2 : d2 = 8) : 
  (1 / 2) * d1 * d2 = 20 :=
by
  sorry

end rhombus_area_l118_118898


namespace find_number_l118_118216

theorem find_number {x : ℝ} 
  (h : 973 * x - 739 * x = 110305) : 
  x = 471.4 := 
by 
  sorry

end find_number_l118_118216


namespace range_of_a_l118_118851

noncomputable def f : ℝ → ℝ
| x => if x ≤ 0 then -x^2 - 1 else Real.log (x + 1)

theorem range_of_a (a : ℝ) : (∀ x : ℝ, f x ≤ a * x) ↔ 1 ≤ a ∧ a ≤ 2 := by
  sorry

end range_of_a_l118_118851


namespace fraction_identity_l118_118267

theorem fraction_identity (m n r t : ℚ) (h1 : m / n = 5 / 3) (h2 : r / t = 8 / 15) : 
  (4 * m * r - 2 * n * t) / (5 * n * t - 9 * m * r) = -14 / 27 :=
by 
  sorry

end fraction_identity_l118_118267


namespace value_of_expression_l118_118670

theorem value_of_expression (a b c d : ℝ) (h : a + b + c + d = 4) : 12 * a - 6 * b + 3 * c - 2 * d = 40 :=
by sorry

end value_of_expression_l118_118670


namespace expression_non_negative_l118_118696

theorem expression_non_negative (a b c : ℝ) (h1 : a > b) (h2 : b > c) : 
  (1 / (a - b)) + (1 / (b - c)) + (4 / (c - a)) ≥ 0 :=
by
  sorry

end expression_non_negative_l118_118696


namespace pyramid_partition_volumes_l118_118395

noncomputable def pyramid_partition_ratios (S A B C D P Q V1 V2 : ℝ) : Prop :=
  let P := ((S + B) / 2 : ℝ)
  let Q := ((S + D) / 2 : ℝ)
  (V1 < V2) → 
  (V2 / V1 = 5)

theorem pyramid_partition_volumes
  (S A B C D P Q : ℝ)
  (V1 V2 : ℝ)
  (hP : P = (S + B) / 2)
  (hQ : Q = (S + D) / 2)
  (hV1 : V1 < V2)
  : V2 / V1 = 5 := 
sorry

end pyramid_partition_volumes_l118_118395


namespace geometric_sequence_product_l118_118135

theorem geometric_sequence_product 
  {a : ℕ → ℝ} (h : ∀ n, 0 < a n)
  (hyp : geometric_sequence a)
  (h_root : (a 1, a 19) are roots of x^2 - 10 * x + 16) : 
  a 8 * a 10 * a 12 = 64 := 
by
  sorry

end geometric_sequence_product_l118_118135


namespace endomorphisms_of_Z2_are_linear_functions_l118_118830

namespace GroupEndomorphism

-- Definition of an endomorphism: a homomorphism from Z² to itself
def is_endomorphism (f : ℤ × ℤ → ℤ × ℤ) : Prop :=
  ∀ a b : ℤ × ℤ, f (a + b) = f a + f b

-- Definition of the specific form of endomorphisms for Z²
def specific_endomorphism_form (u v : ℤ × ℤ) (φ : ℤ × ℤ) : ℤ × ℤ :=
  (φ.1 * u.1 + φ.2 * v.1, φ.1 * u.2 + φ.2 * v.2)

-- Main theorem:
theorem endomorphisms_of_Z2_are_linear_functions :
  ∀ φ : ℤ × ℤ → ℤ × ℤ, is_endomorphism φ →
  ∃ u v : ℤ × ℤ, φ = specific_endomorphism_form u v := by
  sorry

end GroupEndomorphism

end endomorphisms_of_Z2_are_linear_functions_l118_118830


namespace smallest_positive_integer_l118_118193

theorem smallest_positive_integer (n : ℕ) (h : 721 * n % 30 = 1137 * n % 30) :
  ∃ k : ℕ, k > 0 ∧ n = 2 * k :=
by
  sorry

end smallest_positive_integer_l118_118193


namespace initial_money_l118_118282

/-- Given the following conditions:
  (1) June buys 4 maths books at $20 each.
  (2) June buys 6 more science books than maths books at $10 each.
  (3) June buys twice as many art books as maths books at $20 each.
  (4) June spends $160 on music books.
  Prove that June had initially $500 for buying school supplies. -/
theorem initial_money (maths_books : ℕ) (science_books : ℕ) (art_books : ℕ) (music_books_cost : ℕ)
  (h_math_books : maths_books = 4) (price_per_math_book : ℕ) (price_per_science_book : ℕ) 
  (price_per_art_book : ℕ) (price_per_music_books_cost : ℕ) (h_maths_price : price_per_math_book = 20)
  (h_science_books : science_books = maths_books + 6) (h_science_price : price_per_science_book = 10)
  (h_art_books : art_books = 2 * maths_books) (h_art_price : price_per_art_book = 20)
  (h_music_books_cost : music_books_cost = 160) :
  4 * 20 + (4 + 6) * 10 + (2 * 4) * 20 + 160 = 500 :=
by sorry

end initial_money_l118_118282


namespace jake_has_more_balloons_l118_118218

-- Defining the given conditions as parameters
def initial_balloons_allan : ℕ := 2
def initial_balloons_jake : ℕ := 6
def additional_balloons_allan : ℕ := 3

-- Calculate total balloons each person has
def total_balloons_allan : ℕ := initial_balloons_allan + additional_balloons_allan
def total_balloons_jake : ℕ := initial_balloons_jake

-- Formalize the statement to be proved
theorem jake_has_more_balloons :
  total_balloons_jake - total_balloons_allan = 1 :=
by
  -- Proof will be added here
  sorry

end jake_has_more_balloons_l118_118218


namespace john_bought_notebooks_l118_118143

def pages_per_notebook : ℕ := 40
def pages_per_day : ℕ := 4
def total_days : ℕ := 50

theorem john_bought_notebooks : (pages_per_day * total_days) / pages_per_notebook = 5 :=
by
  sorry

end john_bought_notebooks_l118_118143


namespace line_through_point_parallel_to_y_axis_eq_x_eq_neg1_l118_118625

-- Define the point (M) and the properties of the line
def point_M : ℝ × ℝ := (-1, 3)

def parallel_to_y_axis (line : ℝ × ℝ → Prop) : Prop :=
  ∃ b : ℝ, ∀ y : ℝ, line (b, y)

-- Statement we need to prove
theorem line_through_point_parallel_to_y_axis_eq_x_eq_neg1 :
  (∃ line : ℝ × ℝ → Prop, line point_M ∧ parallel_to_y_axis line) → ∀ p : ℝ × ℝ, (p.1 = -1 ↔ (∃ line : ℝ × ℝ → Prop, line p ∧ line point_M ∧ parallel_to_y_axis line)) :=
by
  sorry

end line_through_point_parallel_to_y_axis_eq_x_eq_neg1_l118_118625


namespace radius_of_circle_l118_118095

theorem radius_of_circle (r : ℝ) : 
  (∀ x : ℝ, (4 * x^2 + r = x) → (1 - 16 * r = 0)) → r = 1 / 16 :=
by
  intro H
  have h := H 0
  simp at h
  sorry

end radius_of_circle_l118_118095


namespace line_equation_passes_through_l118_118272

theorem line_equation_passes_through (a b : ℝ) (x y : ℝ) 
  (h_intercept : b = a + 1)
  (h_point : (6 * b) + (-2 * a) = a * b) :
  (x + 2 * y - 2 = 0 ∨ 2 * x + 3 * y - 6 = 0) := 
sorry

end line_equation_passes_through_l118_118272


namespace payment_correct_l118_118598

def total_payment (hours1 hours2 hours3 : ℕ) (rate_per_hour : ℕ) (num_men : ℕ) : ℕ :=
  (hours1 + hours2 + hours3) * rate_per_hour * num_men

theorem payment_correct :
  total_payment 10 8 15 10 2 = 660 :=
by
  -- We skip the proof here
  sorry

end payment_correct_l118_118598


namespace exists_unique_representation_l118_118437

theorem exists_unique_representation (n : ℕ) : 
  ∃! (x y : ℕ), n = ((x + y)^2 + 3 * x + y) / 2 :=
sorry

end exists_unique_representation_l118_118437


namespace solve_for_a_l118_118720

theorem solve_for_a (a : ℝ) 
  (h : (2 * a + 16 + (3 * a - 8)) / 2 = 89) : 
  a = 34 := 
sorry

end solve_for_a_l118_118720


namespace ajax_store_price_l118_118612

theorem ajax_store_price (original_price : ℝ) (first_discount_rate : ℝ) (second_discount_rate : ℝ)
    (h_original: original_price = 180)
    (h_first_discount : first_discount_rate = 0.5)
    (h_second_discount : second_discount_rate = 0.2) :
    let first_discount_price := original_price * (1 - first_discount_rate)
    let saturday_price := first_discount_price * (1 - second_discount_rate)
    saturday_price = 72 :=
by
    sorry

end ajax_store_price_l118_118612


namespace probability_all_red_or_all_white_l118_118201

theorem probability_all_red_or_all_white :
  let red_marbles := 5
  let white_marbles := 4
  let blue_marbles := 6
  let total_marbles := red_marbles + white_marbles + blue_marbles
  let probability_red := (red_marbles / total_marbles) * ((red_marbles - 1) / (total_marbles - 1)) * ((red_marbles - 2) / (total_marbles - 2))
  let probability_white := (white_marbles / total_marbles) * ((white_marbles - 1) / (total_marbles - 1)) * ((white_marbles - 2) / (total_marbles - 2))
  (probability_red + probability_white) = (14 / 455) :=
by
  sorry

end probability_all_red_or_all_white_l118_118201


namespace probability_both_hit_target_l118_118781

def prob_A : ℝ := 0.8
def prob_B : ℝ := 0.7

theorem probability_both_hit_target :
  prob_A * prob_B = 0.56 :=
by sorry

end probability_both_hit_target_l118_118781


namespace rectangular_solid_surface_area_l118_118377

-- Definitions based on conditions
def is_prime (n : ℕ) := n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

def rectangular_solid (a b c : ℕ) :=
  is_prime a ∧ is_prime b ∧ is_prime c ∧ a * b * c = 231

noncomputable def surface_area (a b c : ℕ) := 2 * (a * b + b * c + c * a)

-- Main theorem based on question and answer
theorem rectangular_solid_surface_area :
  ∃ (a b c : ℕ), rectangular_solid a b c ∧ surface_area a b c = 262 := by
  sorry

end rectangular_solid_surface_area_l118_118377


namespace fencing_required_l118_118935

theorem fencing_required
  (L : ℝ) (A : ℝ) (h_L : L = 20) (h_A : A = 400) : 
  (2 * (A / L) + L) = 60 :=
by
  sorry

end fencing_required_l118_118935


namespace marble_count_l118_118508

theorem marble_count (a : ℕ) (h1 : a + 3 * a + 6 * a + 30 * a = 120) : a = 3 :=
  sorry

end marble_count_l118_118508


namespace complement_intersection_l118_118556

open Set

variable (U : Set ℕ := {1, 2, 3, 4, 5, 6, 7, 8})
variable (A : Set ℕ := {2, 5, 8})
variable (B : Set ℕ := {1, 3, 5, 7})

theorem complement_intersection (CUA : Set ℕ := {1, 3, 4, 6, 7}) :
  (CUA ∩ B) = {1, 3, 7} := by
  sorry

end complement_intersection_l118_118556


namespace number_of_pairings_l118_118089

-- Definitions for conditions.
def bowls : Finset String := {"red", "blue", "yellow", "green"}
def glasses : Finset String := {"red", "blue", "yellow", "green"}

-- The theorem statement
theorem number_of_pairings : bowls.card * glasses.card = 16 := by
  sorry

end number_of_pairings_l118_118089


namespace principal_sum_l118_118784

theorem principal_sum (A1 A2 : ℝ) (I P : ℝ) 
  (hA1 : A1 = 1717) 
  (hA2 : A2 = 1734) 
  (hI : I = A2 - A1)
  (h_simple_interest : A1 = P + I) : P = 1700 :=
by
  sorry

end principal_sum_l118_118784


namespace angle_SQR_measure_l118_118017

theorem angle_SQR_measure
    (angle_PQR : ℝ)
    (angle_PQS : ℝ)
    (h1 : angle_PQR = 40)
    (h2 : angle_PQS = 15) : 
    angle_PQR - angle_PQS = 25 := 
by
    sorry

end angle_SQR_measure_l118_118017


namespace average_salary_correct_l118_118456

def salary_A : ℕ := 8000
def salary_B : ℕ := 5000
def salary_C : ℕ := 16000
def salary_D : ℕ := 7000
def salary_E : ℕ := 9000

def total_salary : ℕ := salary_A + salary_B + salary_C + salary_D + salary_E
def number_of_people : ℕ := 5

def average_salary : ℕ := total_salary / number_of_people

theorem average_salary_correct : average_salary = 8800 := by
  sorry

end average_salary_correct_l118_118456


namespace smallest_portion_proof_l118_118035

theorem smallest_portion_proof :
  ∃ (a d : ℚ), 5 * a = 100 ∧ 3 * (a + d) = 2 * d + 7 * (a - 2 * d) ∧ a - 2 * d = 5 / 3 :=
by
  sorry

end smallest_portion_proof_l118_118035


namespace polly_breakfast_minutes_l118_118885
open Nat

theorem polly_breakfast_minutes (B : ℕ) 
  (lunch_minutes : ℕ)
  (dinner_4_days_minutes : ℕ)
  (dinner_3_days_minutes : ℕ)
  (total_minutes : ℕ)
  (h1 : lunch_minutes = 5 * 7)
  (h2 : dinner_4_days_minutes = 10 * 4)
  (h3 : dinner_3_days_minutes = 30 * 3)
  (h4 : total_minutes = 305) 
  (h5 : 7 * B + lunch_minutes + dinner_4_days_minutes + dinner_3_days_minutes = total_minutes) :
  B = 20 :=
by
  -- proof omitted
  sorry

end polly_breakfast_minutes_l118_118885


namespace xy_in_B_l118_118550

def A : Set ℤ := 
  {z | ∃ a b k m : ℤ, z = m * a^2 + k * a * b + m * b^2}

def B : Set ℤ := 
  {z | ∃ a b k m : ℤ, z = a^2 + k * a * b + m^2 * b^2}

theorem xy_in_B (x y : ℤ) (h1 : x ∈ A) (h2 : y ∈ A) : x * y ∈ B := by
  sorry

end xy_in_B_l118_118550


namespace parallel_lines_slope_l118_118813

theorem parallel_lines_slope (k : ℝ) : 
  (∀ x : ℝ, 5 * x + 3 = (3 * k) * x + 7) → k = 5 / 3 := 
sorry

end parallel_lines_slope_l118_118813


namespace min_value_reciprocal_sum_l118_118291

theorem min_value_reciprocal_sum (a b : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : a + b = 1) : 
  ∀ (c : ℝ), c = (1 / a) + (4 / b) → c ≥ 9 :=
by
  intros c hc
  sorry

end min_value_reciprocal_sum_l118_118291


namespace problem1_problem2_l118_118698

-- Definitions for conditions
def p (x : ℝ) : Prop := x^2 - 7 * x + 10 < 0
def q (x : ℝ) (m : ℝ) : Prop := x^2 - 4 * m * x + 3 * m^2 < 0

-- Problem 1: For m = 4, p ∧ q implies 4 < x < 5
theorem problem1 (x : ℝ) (h : 4 < x ∧ x < 5) : 
  p x ∧ q x 4 :=
sorry

-- Problem 2: ∃ m, m > 0, m ≤ 2, and 3m ≥ 5 implies (5/3 ≤ m ≤ 2)
theorem problem2 (m : ℝ) (h1 : m > 0) (h2 : m ≤ 2) (h3 : 3 * m ≥ 5) : 
  5 / 3 ≤ m ∧ m ≤ 2 :=
sorry

end problem1_problem2_l118_118698


namespace B_is_345_complement_U_A_inter_B_is_3_l118_118560

-- Define the universal set U
def U : Set ℕ := {1, 2, 3, 4, 5, 6, 7}

-- Define set A
def A : Set ℕ := {2, 4, 5}

-- Define set B as given in the conditions
def B : Set ℕ := {x ∈ U | 2 < x ∧ x < 6}

-- Prove that B is {3, 4, 5}
theorem B_is_345 : B = {3, 4, 5} := by
  sorry

-- Define the complement of A with respect to U
def complement_U_A : Set ℕ := U \ A

-- Prove the intersection of the complement of A and B is {3}
theorem complement_U_A_inter_B_is_3 : (complement_U_A ∩ B) = {3} := by
  sorry

end B_is_345_complement_U_A_inter_B_is_3_l118_118560


namespace measure_of_angle_B_and_area_of_triangle_l118_118581

theorem measure_of_angle_B_and_area_of_triangle 
    (a b c : ℝ) 
    (A B C : ℝ) 
    (condition : 2 * c = a + (Real.cos A * (b / (Real.cos B))))
    (sum_sides : a + c = 3 * Real.sqrt 2)
    (side_b : b = 4)
    (angle_B : B = Real.pi / 3) :
    B = Real.pi / 3 ∧ 
    (1/2 * a * c * (Real.sin B) = Real.sqrt 3 / 6) :=
by
    sorry

end measure_of_angle_B_and_area_of_triangle_l118_118581


namespace eval_expression_l118_118234

theorem eval_expression : (Real.sqrt (16 - 8 * Real.sqrt 6) + Real.sqrt (16 + 8 * Real.sqrt 6)) = 4 * Real.sqrt 6 :=
by
  sorry

end eval_expression_l118_118234


namespace ellipse_intersection_area_condition_l118_118838

theorem ellipse_intersection_area_condition :
  let ellipse := ∀ (x y : ℝ), (x^2 / 3 + y^2 = 1)
  let F1 := (-Real.sqrt 2, 0)
  let F2 := (Real.sqrt 2, 0)
  let line m := ∀ (x y : ℝ), (y = x + m)
  ∀ (A B : ℝ × ℝ),
    ((A.1^2 / 3 + A.2^2 = 1) ∧ (y = A.1 + m)) ∧
    ((B.1^2 / 3 + B.2^2 = 1) ∧ (y = B.1 + m)) ∧
    (area (F1, A, B) = 2 * area (F2, A, B)) →
    m = -Real.sqrt 2 / 3 := by sorry

end ellipse_intersection_area_condition_l118_118838


namespace rate_per_square_meter_l118_118451

theorem rate_per_square_meter 
  (L : ℝ) (W : ℝ) (C : ℝ)
  (hL : L = 5.5) 
  (hW : W = 3.75)
  (hC : C = 20625)
  : C / (L * W) = 1000 :=
by
  sorry

end rate_per_square_meter_l118_118451


namespace shaded_area_l118_118448

noncomputable def squareArea (a : ℝ) : ℝ := a * a

theorem shaded_area {s : ℝ} (h1 : squareArea s = 1) (h2 : s / s = 2) : 
  ∃ (shaded : ℝ), shaded = 1 / 3 :=
by
  sorry

end shaded_area_l118_118448


namespace sum_inequality_l118_118894

variable {n : ℕ}
variables {a b : Fin n → ℝ}

-- Conditions
def cond1 (h : ∀ i j : Fin n, (i ≤ j) → a i ≥ a j) := ∀ i j : Fin n, (i ≤ j) → a i ≥ a j
def cond2 (h : ∀ i : Fin n, 0 < a i) := ∀ i : Fin n, 0 < a i
def cond3 (h : b 0 ≥ a 0) := b 0 ≥ a 0
def cond4 (h : ∀ k : Fin n, (∏ i in (Finset.range k.succ).to_finset, b i) ≥ (∏ i in (Finset.range k.succ).to_finset, a i)) := ∀ k : Fin n, (∏ i in (Finset.range k.succ).to_finset, b i) ≥ (∏ i in (Finset.range k.succ).to_finset, a i)

-- The inequality to be proved
theorem sum_inequality (h₁ : cond1 a) (h₂ : cond2 a)
  (h₃ : cond3 b) (h₄ : cond4 a b) : 
  (∑ i, b i) ≥ (∑ i, a i) :=
sorry

end sum_inequality_l118_118894


namespace cannot_form_right_triangle_l118_118062

theorem cannot_form_right_triangle (a b c : ℝ) (h1 : a = 2) (h2 : b = 3) (h3 : c = 4) :
  a^2 + b^2 ≠ c^2 :=
by
  rw [h1, h2, h3]
  sorry

end cannot_form_right_triangle_l118_118062


namespace price_decrease_proof_l118_118673

-- Definitions based on the conditions
def original_price (C : ℝ) : ℝ := C
def new_price (C : ℝ) : ℝ := 0.76 * C

theorem price_decrease_proof (C : ℝ) : new_price C = 421.05263157894734 :=
by
  sorry

end price_decrease_proof_l118_118673


namespace reciprocal_neg_3_div_4_l118_118455

theorem reciprocal_neg_3_div_4 : (- (3 / 4 : ℚ))⁻¹ = -(4 / 3 : ℚ) :=
by
  sorry

end reciprocal_neg_3_div_4_l118_118455


namespace n_values_satisfy_condition_l118_118553

-- Define the exponential functions
def exp1 (n : ℤ) : ℚ := (-1/2) ^ n
def exp2 (n : ℤ) : ℚ := (-1/5) ^ n

-- Define the set of possible values for n
def valid_n : List ℤ := [-2, -1, 0, 1, 2, 3]

-- Define the condition for n to satisfy the inequality
def satisfies_condition (n : ℤ) : Prop := exp1 n > exp2 n

-- Prove that the only values of n that satisfy the condition are -1 and 2
theorem n_values_satisfy_condition :
  ∀ n ∈ valid_n, satisfies_condition n ↔ (n = -1 ∨ n = 2) :=
by
  intro n
  sorry

end n_values_satisfy_condition_l118_118553


namespace random_event_proof_l118_118081

-- Definitions based on conditions
def event1 := "Tossing a coin twice in a row, and both times it lands heads up."
def event2 := "Opposite charges attract each other."
def event3 := "Water freezes at 1℃ under standard atmospheric pressure."

def is_random_event (event: String) : Prop :=
  event = event1 ∨ event = event2 ∨ event = event3 → event = event1

theorem random_event_proof : is_random_event event1 ∧ ¬is_random_event event2 ∧ ¬is_random_event event3 :=
by
  -- Proof goes here
  sorry

end random_event_proof_l118_118081


namespace sum_of_coordinates_l118_118161

theorem sum_of_coordinates (C D : ℝ × ℝ) (hC : C = (0, 0)) (hD : D.snd = 6) (h_slope : (D.snd - C.snd) / (D.fst - C.fst) = 3/4) : D.fst + D.snd = 14 :=
sorry

end sum_of_coordinates_l118_118161


namespace angle_ABC_30_degrees_l118_118266

theorem angle_ABC_30_degrees 
    (angle_CBD : ℝ)
    (angle_ABD : ℝ)
    (angle_ABC : ℝ)
    (h1 : angle_CBD = 90)
    (h2 : angle_ABC + angle_ABD + angle_CBD = 180)
    (h3 : angle_ABD = 60) :
    angle_ABC = 30 :=
by
  sorry

end angle_ABC_30_degrees_l118_118266


namespace parabola_opens_downwards_l118_118158

theorem parabola_opens_downwards (a : ℝ) (h : ℝ) (k : ℝ) :
  a < 0 → h = 3 → ∃ k, (∀ x, y = a * (x - h) ^ 2 + k → y = -(x - 3)^2 + k) :=
by
  intros ha hh
  use k
  sorry

end parabola_opens_downwards_l118_118158


namespace eiffel_tower_scale_l118_118311

theorem eiffel_tower_scale (height_tower_m : ℝ) (height_model_cm : ℝ) :
    height_tower_m = 324 →
    height_model_cm = 50 →
    (height_tower_m * 100) / height_model_cm = 648 →
    (648 / 100) = 6.48 :=
by
  intro h_tower h_model h_ratio
  rw [h_tower, h_model] at h_ratio
  sorry

end eiffel_tower_scale_l118_118311


namespace find_a_l118_118547

theorem find_a :
  ∃ (a : ℤ), (∀ (x y : ℤ),
    (∃ (m n : ℤ), (x - 8 + m * y) * (x + 3 + n * y) = x^2 + 7 * x * y + a * y^2 - 5 * x - 45 * y - 24) ↔ a = 6) := 
sorry

end find_a_l118_118547


namespace inverse_function_f_l118_118235

noncomputable def f (x : ℝ) : ℝ := 2^(x + 1)

noncomputable def f_inv (x : ℝ) : ℝ := Real.log x / Real.log 2 - 1

theorem inverse_function_f : ∀ x > 0, f_inv (f x) = x :=
by
  intro x hx
  dsimp [f, f_inv]
  sorry

end inverse_function_f_l118_118235


namespace pyramid_apex_angle_l118_118897

theorem pyramid_apex_angle (A B C D E O : Type) 
  (square_base : Π (P Q : Type), Prop) 
  (isosceles_triangle : Π (R S T : Type), Prop)
  (AEB_angle : Π (X Y Z : Type), Prop) 
  (angle_AOB : ℝ)
  (angle_AEB : ℝ)
  (square_base_conditions : square_base A B ∧ square_base B C ∧ square_base C D ∧ square_base D A)
  (isosceles_triangle_conditions : isosceles_triangle A E B ∧ isosceles_triangle B E C ∧ isosceles_triangle C E D ∧ isosceles_triangle D E A)
  (center : O)
  (diagonals_intersect_at_right_angle : angle_AOB = 90)
  (measured_angle_at_apex : angle_AEB = 100) :
False :=
sorry

end pyramid_apex_angle_l118_118897


namespace different_algorithms_for_same_problem_l118_118649

-- Define the basic concept of a problem
def Problem := Type

-- Define what it means for something to be an algorithm solving a problem
def Algorithm (P : Problem) := P -> Prop

-- Define the statement to be true: Different algorithms can solve the same problem
theorem different_algorithms_for_same_problem (P : Problem) (A1 A2 : Algorithm P) :
  P = P -> A1 ≠ A2 -> true :=
by
  sorry

end different_algorithms_for_same_problem_l118_118649


namespace dakotas_medical_bill_l118_118371

variable (days_in_hospital : ℕ) (bed_cost_per_day : ℕ) (specialist_cost_per_hour : ℕ) (specialist_time_in_hours : ℚ) (num_specialists : ℕ) (ambulance_cost : ℕ)

theorem dakotas_medical_bill 
  (h1 : days_in_hospital = 3) 
  (h2 : bed_cost_per_day = 900)
  (h3 : specialist_cost_per_hour = 250)
  (h4 : specialist_time_in_hours = 0.25)
  (h5 : num_specialists = 2)
  (h6 : ambulance_cost = 1800) : 

  let bed_total := bed_cost_per_day * days_in_hospital,
      specialists_total := (specialist_cost_per_hour * specialist_time_in_hours * num_specialists).toNat,
      total_cost := bed_total + specialists_total + ambulance_cost
  in 
  total_cost = 4750 := 
by 
  sorry

end dakotas_medical_bill_l118_118371


namespace even_product_probability_l118_118925

theorem even_product_probability :
  let S := {1, 2, 3, 4, 5}
  let total_ways := Nat.choose 5 2
  let odd_ways := Nat.choose 3 2
  let even_product_ways := total_ways - odd_ways
  (even_product_ways : ℚ) / total_ways = 7 / 10 :=
by
  let S := {1, 2, 3, 4, 5}
  let total_ways := Nat.choose 5 2
  let odd_ways := Nat.choose 3 2
  let even_product_ways := total_ways - odd_ways
  have h1 : (even_product_ways : ℚ) = 7 := sorry
  have h2 : (total_ways : ℚ) = 10 := sorry
  rw [h1, h2]
  norm_num
  done

end even_product_probability_l118_118925


namespace license_plate_count_is_correct_l118_118858

/-- Define the number of consonants in the English alphabet --/
def num_consonants : Nat := 20

/-- Define the number of possibilities for 'A' --/
def num_A : Nat := 1

/-- Define the number of even digits --/
def num_even_digits : Nat := 5

/-- Define the total number of valid four-character license plates --/
def total_license_plate_count : Nat :=
  num_consonants * num_A * num_consonants * num_even_digits

/-- Theorem stating that the total number of license plates is 2000 --/
theorem license_plate_count_is_correct : 
  total_license_plate_count = 2000 :=
  by
    -- The proof is omitted
    sorry

end license_plate_count_is_correct_l118_118858


namespace average_speed_l118_118186

theorem average_speed (s₁ s₂ s₃ s₄ s₅ : ℝ) (h₁ : s₁ = 85) (h₂ : s₂ = 45) (h₃ : s₃ = 60) (h₄ : s₄ = 75) (h₅ : s₅ = 50) : 
  (s₁ + s₂ + s₃ + s₄ + s₅) / 5 = 63 := 
by 
  sorry

end average_speed_l118_118186


namespace range_of_a_l118_118567

-- Definitions based on conditions
def f (a x : ℝ) : ℝ := x^2 + 2 * (a - 1) * x + 4

-- Statement of the theorem to be proven
theorem range_of_a (a : ℝ) : (∀ x : ℝ, x ≤ 4 → f a x ≤ f a 4) → a ≤ -3 :=
by
  sorry

end range_of_a_l118_118567


namespace expand_and_simplify_l118_118098

theorem expand_and_simplify : ∀ x : ℝ, (7 * x + 5) * 3 * x^2 = 21 * x^3 + 15 * x^2 :=
by
  intro x
  sorry

end expand_and_simplify_l118_118098


namespace operation_on_b_l118_118173

variables (t b b' : ℝ)
variable (C : ℝ := t * b ^ 4)
variable (e : ℝ := 16 * C)

theorem operation_on_b :
  tb'^4 = 16 * tb^4 → b' = 2 * b := by
  sorry

end operation_on_b_l118_118173


namespace induction_step_l118_118462

theorem induction_step (k : ℕ) : ((k + 1 + k) * (k + 1 + k + 1) / (k + 1)) = 2 * (2 * k + 1) := by
  sorry

end induction_step_l118_118462


namespace point_in_fourth_quadrant_l118_118195

def Point : Type := ℤ × ℤ

def in_fourth_quadrant (p : Point) : Prop :=
  p.1 > 0 ∧ p.2 < 0

def A : Point := (-3, 7)
def B : Point := (3, -7)
def C : Point := (3, 7)
def D : Point := (-3, -7)

theorem point_in_fourth_quadrant : in_fourth_quadrant B :=
by {
  -- skipping the proof steps for the purpose of this example
  sorry
}

end point_in_fourth_quadrant_l118_118195


namespace max_tiles_on_floor_l118_118438

-- Definitions based on the given conditions
def tile_length1 := 35 -- in cm
def tile_length2 := 30 -- in cm
def floor_length := 1000 -- in cm
def floor_width := 210 -- in cm

-- Lean 4 statement for the proof problem
theorem max_tiles_on_floor : 
  (max ((floor_length / tile_length1) * (floor_width / tile_length2))
       ((floor_length / tile_length2) * (floor_width / tile_length1))) = 198 := by
  sorry

end max_tiles_on_floor_l118_118438


namespace values_of_x_that_satisfy_gg_x_eq_g_x_l118_118590

noncomputable def g (x : ℝ) : ℝ := x^2 - 5 * x

theorem values_of_x_that_satisfy_gg_x_eq_g_x :
  {x : ℝ | g (g x) = g x} = {0, 5, -2, 3} :=
by
  sorry

end values_of_x_that_satisfy_gg_x_eq_g_x_l118_118590


namespace calculate_expression_l118_118361

theorem calculate_expression : (2^1234 + 5^1235)^2 - (2^1234 - 5^1235)^2 = 20 * 10^1234 := 
by 
  sorry

end calculate_expression_l118_118361


namespace find_m_for_area_of_triangles_l118_118840

noncomputable def ellipse_foci := (-real.sqrt 2, 0, real.sqrt 2, 0)

theorem find_m_for_area_of_triangles : 
  (∃ (m : ℝ), 
    let F1 := (-real.sqrt 2, 0) in
    let F2 := (real.sqrt 2, 0) in
    let A B : ℝ × ℝ in
    let ellipse : ∀ x y : ℝ, (x^2 / 3) + y^2 = 1 := sorry in
    let line := λ x y : ℝ, y = x + m in
    line_intersects_ellipse : ∀ A B : ℝ × ℝ, disjoint ((A, B) ∪ ellipse) := sorry in
    (area_of_triangle(F1, A, B) = 2 * area_of_triangle(F2, A, B)) → m = -real.sqrt 2 / 3) :=
sorry

end find_m_for_area_of_triangles_l118_118840


namespace emily_total_spent_l118_118674

-- Define the given conditions.
def cost_per_flower : ℕ := 3
def num_roses : ℕ := 2
def num_daisies : ℕ := 2

-- Calculate the total number of flowers and the total cost.
def total_flowers : ℕ := num_roses + num_daisies
def total_cost : ℕ := total_flowers * cost_per_flower

-- Statement: Prove that Emily spent 12 dollars.
theorem emily_total_spent : total_cost = 12 := by
  sorry

end emily_total_spent_l118_118674


namespace weekly_milk_production_l118_118822

theorem weekly_milk_production 
  (bess_milk_per_day : ℕ) 
  (brownie_milk_per_day : ℕ) 
  (daisy_milk_per_day : ℕ) 
  (total_milk_per_day : ℕ) 
  (total_milk_per_week : ℕ) 
  (h1 : bess_milk_per_day = 2) 
  (h2 : brownie_milk_per_day = 3 * bess_milk_per_day) 
  (h3 : daisy_milk_per_day = bess_milk_per_day + 1) 
  (h4 : total_milk_per_day = bess_milk_per_day + brownie_milk_per_day + daisy_milk_per_day)
  (h5 : total_milk_per_week = total_milk_per_day * 7) : 
  total_milk_per_week = 77 := 
by sorry

end weekly_milk_production_l118_118822


namespace lying_dwarf_number_is_possible_l118_118941

def dwarfs_sum (a1 a2 a3 a4 a5 a6 a7 : ℕ) : Prop :=
  a2 = a1 ∧
  a3 = a1 + a2 ∧
  a4 = a1 + a2 + a3 ∧
  a5 = a1 + a2 + a3 + a4 ∧
  a6 = a1 + a2 + a3 + a4 + a5 ∧
  a7 = a1 + a2 + a3 + a4 + a5 + a6 ∧
  a1 + a2 + a3 + a4 + a5 + a6 + a7 = 58

theorem lying_dwarf_number_is_possible (a1 a2 a3 a4 a5 a6 a7 : ℕ) :
  dwarfs_sum a1 a2 a3 a4 a5 a6 a7 →
  (a1 = 13 ∨ a1 = 26) :=
sorry

end lying_dwarf_number_is_possible_l118_118941


namespace weight_ratio_l118_118594

-- Conditions
def initial_weight : ℕ := 99
def initial_loss : ℕ := 12
def weight_added_back (x : ℕ) : Prop := x = 81 + 30 - initial_weight
def times_lost : ℕ := 3 * initial_loss
def final_gain : ℕ := 6
def final_weight : ℕ := 81

-- Question
theorem weight_ratio (x : ℕ)
  (H1 : weight_added_back x)
  (H2 : initial_weight - initial_loss + x - times_lost + final_gain = final_weight) :
  x / initial_loss = 2 := by
  sorry

end weight_ratio_l118_118594


namespace Im_abcd_eq_zero_l118_118289

noncomputable def normalized (z : ℂ) : ℂ := z / Complex.abs z

theorem Im_abcd_eq_zero (a b c d : ℂ)
  (h1 : ∃ α : ℝ, ∃ w : ℂ, w = Complex.cos α + Complex.sin α * Complex.I ∧ (normalized b = w * normalized a) ∧ (normalized d = w * normalized c)) :
  Complex.im (a * b * c * d) = 0 :=
by
  sorry

end Im_abcd_eq_zero_l118_118289


namespace mathematicians_correct_l118_118907

noncomputable def scenario1 (whites1 : ℕ) (total1 : ℕ) (whites2 : ℕ) (total2 : ℕ) : Prop :=
  whites1 = 4 ∧ total1 = 7 ∧ whites2 = 3 ∧ total2 = 5

noncomputable def scenario2 (whites1 : ℕ) (total1 : ℕ) (whites2 : ℕ) (total2 : ℕ) : Prop :=
  whites1 = 8 ∧ total1 = 14 ∧ whites2 = 3 ∧ total2 = 5

noncomputable def scenario3 (whites1 : ℕ) (total1 : ℕ) (whites2 : ℕ) (total2 : ℕ) : Prop :=
  (19 / 35 < 4 / 7) ∧ (4 / 7 < 3 / 5)

noncomputable def probability (whites1 : ℕ) (total1 : ℕ) (whites2 : ℕ) (total2 : ℕ) : ℝ :=
  (whites1 + whites2) / (total1 + total2)

theorem mathematicians_correct :
  let whites1_s1 := 4 in
  let total1_s1 := 7 in
  let whites2_s1 := 3 in
  let total2_s1 := 5 in
  let whites1_s2 := 8 in
  let total1_s2 := 14 in
  let whites2_s2 := 3 in
  let total2_s2 := 5 in
  scenario1 whites1_s1 total1_s1 whites2_s1 total2_s1 →
  scenario2 whites1_s2 total1_s2 whites2_s2 total2_s2 →
  scenario3 whites1_s1 total1_s1 whites2_s2 total2_s2 →
  probability whites1_s1 total1_s1 whites2_s1 total2_s1 ≤ 3 / 5 ∨
  probability whites1_s1 total1_s1 whites2_s1 total2_s1 = 4 / 7 ∨
  probability whites1_s2 total1_s2 whites2_s2 total2_s2 ≤ 3 / 5 ∨
  probability whites1_s2 total1_s2 whites2_s2 total2_s2 = 4 / 7 :=
begin
  intros,
  sorry

end mathematicians_correct_l118_118907


namespace absolute_difference_l118_118339

theorem absolute_difference : |8 - 3^2| - |4^2 - 6*3| = -1 := by
  sorry

end absolute_difference_l118_118339


namespace calc_expr_eq_simplify_expr_eq_l118_118948

-- Problem 1: Calculation
theorem calc_expr_eq : 
  ((1 / 2) ^ (-2) - (Real.pi - Real.sqrt 5) ^ 0 - Real.sqrt 20) = 3 - 2 * Real.sqrt 5 := 
  by
  sorry

-- Problem 2: Simplification
theorem simplify_expr_eq (x : ℝ) (hx : x ≠ 0): 
  ((x^2 - 2 * x + 1) / (x^2 - 1) / (x - 1) / (x^2 + x)) = x := 
  by
  sorry

end calc_expr_eq_simplify_expr_eq_l118_118948


namespace expression_evaluation_l118_118982

theorem expression_evaluation : (3 * 4 * 5) * (1/3 + 1/4 + 1/5) = 47 := by
  sorry

end expression_evaluation_l118_118982


namespace badgers_win_at_least_five_games_prob_l118_118034

noncomputable def probability_Badgers_win_at_least_five_games : ℚ :=
  let p : ℚ := 1 / 2
  let n : ℕ := 9
  (1 / 2)^n * ∑ k in finset.range (n + 1), if k >= 5 then (nat.choose n k : ℚ) else 0

theorem badgers_win_at_least_five_games_prob :
  probability_Badgers_win_at_least_five_games = 1 / 2 :=
by sorry

end badgers_win_at_least_five_games_prob_l118_118034


namespace projectile_time_l118_118797

theorem projectile_time : ∃ t : ℝ, (60 - 8 * t - 5 * t^2 = 30) ∧ t = 1.773 := by
  sorry

end projectile_time_l118_118797


namespace mixture_proportion_exists_l118_118856

-- Define the ratios and densities of the liquids
variables (k : ℝ) (ρ1 ρ2 ρ3 : ℝ) (m1 m2 m3 : ℝ)
variables (x y : ℝ)

-- Given conditions
def density_ratio : Prop := 
  ρ1 = 6 * k ∧ ρ2 = 3 * k ∧ ρ3 = 2 * k

def mass_condition : Prop := 
  m2 / m1 ≤ 2 / 7

-- Must prove that a solution exists where the resultant density is the arithmetic mean
def mixture_density : Prop := 
  (m1 + m2 + m3) / ((m1 / ρ1) + (m2 / ρ2) + (m3 / ρ3)) = (ρ1 + ρ2 + ρ3) / 3

-- Statement (No proof provided)
theorem mixture_proportion_exists (k : ℝ) (m1 m2 m3 : ℝ) (ρ1 ρ2 ρ3 : ℝ) (x y : ℝ) :
  density_ratio k ρ1 ρ2 ρ3 →
  mass_condition m1 m2 →
  mixture_density m1 m2 m3 k ρ1 ρ2 ρ3 :=
sorry

end mixture_proportion_exists_l118_118856


namespace part1_solution_set_part2_range_of_a_l118_118546

-- Definitions of f and g as provided in the problem.
def f (x : ℝ) : ℝ := |x + 1| + |x - 2|
def g (x a : ℝ) : ℝ := |x + 1| - |x - a| + a

-- Problem 1: Prove the solution set for f(x) ≤ 5 is [-2, 3]
theorem part1_solution_set : { x : ℝ | f x ≤ 5 } = { x : ℝ | -2 ≤ x ∧ x ≤ 3 } :=
  sorry

-- Problem 2: Prove the range of a when f(x) ≥ g(x) always holds is (-∞, 1]
theorem part2_range_of_a (a : ℝ) (h : ∀ x : ℝ, f x ≥ g x a) : a ≤ 1 :=
  sorry

end part1_solution_set_part2_range_of_a_l118_118546


namespace remainder_3n_plus_2_l118_118718

-- Define the condition
def n_condition (n : ℤ) : Prop := n % 7 = 5

-- Define the theorem to be proved
theorem remainder_3n_plus_2 (n : ℤ) (h : n_condition n) : (3 * n + 2) % 7 = 3 := 
by sorry

end remainder_3n_plus_2_l118_118718


namespace dakotas_medical_bill_l118_118372

theorem dakotas_medical_bill :
  let days_in_hospital := 3
  let hospital_bed_cost_per_day := 900
  let specialists_rate_per_hour := 250
  let specialist_minutes_per_day := 15
  let num_specialists := 2
  let ambulance_cost := 1800

  let hospital_bed_cost := hospital_bed_cost_per_day * days_in_hospital
  let specialists_total_minutes := specialist_minutes_per_day * num_specialists
  let specialists_hours := specialists_total_minutes / 60.0
  let specialists_cost := specialists_hours * specialists_rate_per_hour

  let total_medical_bill := hospital_bed_cost + specialists_cost + ambulance_cost

  total_medical_bill = 4625 := 
by
  sorry

end dakotas_medical_bill_l118_118372


namespace event_B_C_mutually_exclusive_l118_118240

-- Define the events based on the given conditions
def EventA (products : Type) (is_defective : products → Prop) (x y : products) : Prop :=
  ¬is_defective x ∧ ¬is_defective y

def EventB (products : Type) (is_defective : products → Prop) (x y : products) : Prop :=
  is_defective x ∧ is_defective y

def EventC (products : Type) (is_defective : products → Prop) (x y : products) : Prop :=
  ¬(is_defective x ∧ is_defective y)

-- Prove that Event B and Event C are mutually exclusive
theorem event_B_C_mutually_exclusive (products : Type) (is_defective : products → Prop) (x y : products) :
  (EventB products is_defective x y) → ¬(EventC products is_defective x y) :=
sorry

end event_B_C_mutually_exclusive_l118_118240


namespace range_of_k_intersecting_AB_l118_118006

theorem range_of_k_intersecting_AB 
  (A B : ℝ × ℝ) 
  (hA : A = (2, 7)) 
  (hB : B = (9, 6)) 
  (k : ℝ) 
  (hk : k ≠ 0) 
  (H : ∃ x : ℝ, A.2 = k * A.1 ∧ B.2 = k * B.1):
  (2 / 3) ≤ k ∧ k ≤ 7 / 2 :=
by sorry

end range_of_k_intersecting_AB_l118_118006


namespace people_in_group_l118_118312

theorem people_in_group
  (N : ℕ)
  (h1 : ∃ w1 w2 : ℝ, w1 = 65 ∧ w2 = 71 ∧ w2 - w1 = 6)
  (h2 : ∃ avg_increase : ℝ, avg_increase = 1.5 ∧ 6 = avg_increase * N) :
  N = 4 :=
sorry

end people_in_group_l118_118312


namespace master_wang_resting_on_sunday_again_l118_118883

theorem master_wang_resting_on_sunday_again (n : ℕ) 
  (works_days := 8) 
  (rest_days := 2) 
  (week_days := 7) 
  (cycle_days := works_days + rest_days) 
  (initial_rest_saturday_sunday : Prop) : 
  (initial_rest_saturday_sunday → ∃ n : ℕ, (week_days * n) % cycle_days = rest_days) → 
  (∃ n : ℕ, n = 7) :=
by
  sorry

end master_wang_resting_on_sunday_again_l118_118883


namespace factorize1_factorize2_factorize3_l118_118992

-- Proof problem 1: Prove m^2 + 4m + 4 = (m + 2)^2
theorem factorize1 (m : ℝ) : m^2 + 4 * m + 4 = (m + 2)^2 :=
sorry

-- Proof problem 2: Prove a^2 b - 4ab^2 + 3b^3 = b(a-b)(a-3b)
theorem factorize2 (a b : ℝ) : a^2 * b - 4 * a * b^2 + 3 * b^3 = b * (a - b) * (a - 3 * b) :=
sorry

-- Proof problem 3: Prove (x^2 + y^2)^2 - 4x^2 y^2 = (x + y)^2 (x - y)^2
theorem factorize3 (x y : ℝ) : (x^2 + y^2)^2 - 4 * x^2 * y^2 = (x + y)^2 * (x - y)^2 :=
sorry

end factorize1_factorize2_factorize3_l118_118992


namespace mathematicians_correctness_l118_118903

theorem mathematicians_correctness :
  ∃ (scenario1_w1 s_w1 : ℕ) (scenario1_w2 s_w2 : ℕ) (scenario2_w1 s2_w1 : ℕ) (scenario2_w2 s2_w2 : ℕ),
    scenario1_w1 = 4 ∧ s_w1 = 7 ∧ scenario1_w2 = 3 ∧ s_w2 = 5 ∧
    scenario2_w1 = 8 ∧ s2_w1 = 14 ∧ scenario2_w2 = 3 ∧ s2_w2 = 5 ∧
    let total_white1 := scenario1_w1 + scenario1_w2,
        total_choco1 := s_w1 + s_w2,
        prob1 := (total_white1 : ℚ) / total_choco1,
        total_white2 := scenario2_w1 + scenario2_w2,
        total_choco2 := s2_w1 + s2_w2,
        prob2 := (total_white2 : ℚ) / total_choco2,
        prob_box1 := 4 / 7,
        prob_box2 := 3 / 5 in
    (prob1 = 7 / 12 ∧ prob2 = 11 / 19 ∧
    (19 / 35 < prob_box1 ∧ prob_box1 < prob_box2) ∧
    (prob_box1 ≠ 19 / 35 ∧ prob_box1 ≠ 3 / 5)) :=
sorry

end mathematicians_correctness_l118_118903


namespace find_a_and_tangent_point_l118_118210

noncomputable def tangent_line_and_curve (a : ℚ) (P : ℚ × ℚ) : Prop :=
  ∃ (x₀ : ℚ), (P = (x₀, x₀ + a)) ∧ (P = (x₀, x₀^3 - x₀^2 + 1)) ∧ (3*x₀^2 - 2*x₀ = 1)

theorem find_a_and_tangent_point :
  ∃ (a : ℚ) (P : ℚ × ℚ), tangent_line_and_curve a P ∧ a = 32/27 ∧ P = (-1/3, 23/27) :=
sorry

end find_a_and_tangent_point_l118_118210


namespace solution_set_of_inequality_l118_118323

theorem solution_set_of_inequality :
  {x : ℝ | -x^2 + 2*x + 15 ≥ 0} = {x : ℝ | -3 ≤ x ∧ x ≤ 5} := 
sorry

end solution_set_of_inequality_l118_118323


namespace mixture_ratio_l118_118279

variables (p q : ℝ)

theorem mixture_ratio 
  (h1 : (5/8) * p + (1/4) * q = 0.5)
  (h2 : (3/8) * p + (3/4) * q = 0.5) : 
  p / q = 1 := 
by 
  sorry

end mixture_ratio_l118_118279


namespace percent_equivalence_l118_118570

variable (x : ℝ)
axiom condition : 0.30 * 0.15 * x = 18

theorem percent_equivalence :
  0.15 * 0.30 * x = 18 := sorry

end percent_equivalence_l118_118570


namespace xy_is_perfect_cube_l118_118668

theorem xy_is_perfect_cube (x y : ℕ) (h₁ : x = 5 * 2^4 * 3^3) (h₂ : y = 2^2 * 5^2) : ∃ z : ℕ, (x * y) = z^3 :=
by
  sorry

end xy_is_perfect_cube_l118_118668


namespace correlation_highly_related_l118_118179

-- Conditions:
-- Let corr be the linear correlation coefficient of product output and unit cost.
-- Let rel be the relationship between product output and unit cost.

def corr : ℝ := -0.87

-- Proof Goal:
-- If corr = -0.87, then the relationship is "highly related".

theorem correlation_highly_related (h : corr = -0.87) : rel = "highly related" := by
  sorry

end correlation_highly_related_l118_118179


namespace cost_price_of_watch_l118_118493

-- Let C be the cost price of the watch
variable (C : ℝ)

-- Conditions: The selling price at a loss of 8% and the selling price with a gain of 4% if sold for Rs. 140 more
axiom loss_condition : 0.92 * C + 140 = 1.04 * C

-- Objective: Prove that C = 1166.67
theorem cost_price_of_watch : C = 1166.67 :=
by
  have h := loss_condition
  sorry

end cost_price_of_watch_l118_118493


namespace find_w_when_x_is_six_l118_118031

variable {x w : ℝ}
variable (h1 : x = 3)
variable (h2 : w = 16)
variable (h3 : ∀ (x w : ℝ), x^4 * w^(1 / 4) = 162)

theorem find_w_when_x_is_six : x = 6 → w = 1 / 4096 :=
by
  intro hx
  sorry

end find_w_when_x_is_six_l118_118031


namespace smallest_k_sum_sequence_l118_118658

theorem smallest_k_sum_sequence (n : ℕ) (h : 100 = (n + 1) * (2 * 9 + n) / 2) : k = 9 := 
sorry

end smallest_k_sum_sequence_l118_118658


namespace total_money_taken_l118_118519

def individual_bookings : ℝ := 12000
def group_bookings : ℝ := 16000
def returned_due_to_cancellations : ℝ := 1600

def total_taken (individual_bookings : ℝ) (group_bookings : ℝ) (returned_due_to_cancellations : ℝ) : ℝ :=
  (individual_bookings + group_bookings) - returned_due_to_cancellations

theorem total_money_taken :
  total_taken individual_bookings group_bookings returned_due_to_cancellations = 26400 := by
  sorry

end total_money_taken_l118_118519


namespace log_2_64_sqrt_2_l118_118380

theorem log_2_64_sqrt_2 : log 2 (64 * real.sqrt 2) = 13 / 2 :=
by
  have h1 : 64 = 2^6 := by norm_num
  have h2 : real.sqrt 2 = 2^(1/2 : ℝ) := by rw real.sqrt_eq_rpow; norm_num
  sorry

end log_2_64_sqrt_2_l118_118380


namespace real_solutions_system_l118_118534

theorem real_solutions_system (x y z : ℝ) : 
  (x = 4 * z^2 / (1 + 4 * z^2) ∧ y = 4 * x^2 / (1 + 4 * x^2) ∧ z = 4 * y^2 / (1 + 4 * y^2)) ↔ 
  (x = 0 ∧ y = 0 ∧ z = 0) ∨ (x = 1/2 ∧ y = 1/2 ∧ z = 1/2) :=
by
  sorry

end real_solutions_system_l118_118534


namespace triangle_equilateral_from_midpoint_circles_l118_118927

theorem triangle_equilateral_from_midpoint_circles (a b c : ℝ)
  (h1 : ∃ E F G : ℝ → ℝ, ∀ x, (|E x| = a/4 ∨ |F x| = b/4 ∨ |G x| = c/4))
  (h2 : (|a/2| ≤ a/4 + b/4) ∧ (|b/2| ≤ b/4 + c/4) ∧ (|c/2| ≤ c/4 + a/4)) :
  a = b ∧ b = c :=
sorry

end triangle_equilateral_from_midpoint_circles_l118_118927


namespace cost_difference_l118_118986

-- Define the costs
def cost_chocolate : ℕ := 3
def cost_candy_bar : ℕ := 7

-- Define the difference to be proved
theorem cost_difference :
  cost_candy_bar - cost_chocolate = 4 :=
by
  -- trivial proof steps
  sorry

end cost_difference_l118_118986


namespace quick_calc_formula_l118_118694

variables (a b A B C : ℤ)

theorem quick_calc_formula (h1 : (100 - a) * (100 - b) = (A + B - 100) * 100 + C)
                           (h2 : (100 + a) * (100 + b) = (A + B - 100) * 100 + C) :
  A = 100 ∨ A = 100 ∧ B = 100 ∨ B = 100 ∧ C = a * b :=
sorry

end quick_calc_formula_l118_118694


namespace intersect_P_M_l118_118743

def P : Set ℝ := {x | 0 ≤ x ∧ x < 3}
def M : Set ℝ := {x | |x| ≤ 3}

theorem intersect_P_M : (P ∩ M) = {x | 0 ≤ x ∧ x < 3} := by
  sorry

end intersect_P_M_l118_118743


namespace prob_A_not_losing_prob_A_not_winning_l118_118002

-- Definitions based on the conditions
def prob_winning : ℝ := 0.41
def prob_tie : ℝ := 0.27

-- The probability of A not losing
def prob_not_losing : ℝ := prob_winning + prob_tie

-- The probability of A not winning
def prob_not_winning : ℝ := 1 - prob_winning

-- Proof problems
theorem prob_A_not_losing : prob_not_losing = 0.68 := by
  sorry

theorem prob_A_not_winning : prob_not_winning = 0.59 := by
  sorry

end prob_A_not_losing_prob_A_not_winning_l118_118002


namespace probability_red_or_yellow_l118_118069

-- Definitions and conditions
def p_green : ℝ := 0.25
def p_blue : ℝ := 0.35
def total_probability := 1
def p_red_and_yellow := total_probability - (p_green + p_blue)

-- Theorem statement
theorem probability_red_or_yellow :
  p_red_and_yellow = 0.40 :=
by
  -- Here we would prove that the combined probability of selecting either a red or yellow jelly bean is 0.40, given the conditions.
  sorry

end probability_red_or_yellow_l118_118069


namespace compute_d_for_ellipse_l118_118977

theorem compute_d_for_ellipse
  (in_first_quadrant : true)
  (is_tangent_x_axis : true)
  (is_tangent_y_axis : true)
  (focus1 : (ℝ × ℝ) := (5, 4))
  (focus2 : (ℝ × ℝ) := (d, 4)) :
  d = 3.2 := by
  sorry

end compute_d_for_ellipse_l118_118977


namespace abs_gt_two_nec_but_not_suff_l118_118787

theorem abs_gt_two_nec_but_not_suff (x : ℝ) : (|x| > 2 → x < -2) ∧ (¬ (|x| > 2 ↔ x < -2)) := 
sorry

end abs_gt_two_nec_but_not_suff_l118_118787


namespace supplementary_angle_60_eq_120_l118_118566

def supplementary_angle (α : ℝ) : ℝ :=
  180 - α

theorem supplementary_angle_60_eq_120 :
  supplementary_angle 60 = 120 :=
by
  -- the proof should be filled here
  sorry

end supplementary_angle_60_eq_120_l118_118566


namespace area_of_45_45_90_triangle_l118_118039

noncomputable def leg_length (hypotenuse : ℝ) : ℝ :=
  hypotenuse / Real.sqrt 2

theorem area_of_45_45_90_triangle (hypotenuse : ℝ) (h : hypotenuse = 13) : 
  (1 / 2) * (leg_length hypotenuse) * (leg_length hypotenuse) = 84.5 :=
by
  sorry

end area_of_45_45_90_triangle_l118_118039


namespace tins_left_after_damage_l118_118490

theorem tins_left_after_damage (cases : ℕ) (tins_per_case : ℕ) (damage_rate : ℚ) 
    (total_cases : cases = 15) (tins_per_case_value : tins_per_case = 24)
    (damage_rate_value : damage_rate = 0.05) :
    let total_tins := cases * tins_per_case
        damaged_tins := damage_rate * total_tins
        remaining_tins := total_tins - damaged_tins in
    remaining_tins = 342 := 
by
  sorry

end tins_left_after_damage_l118_118490


namespace container_emptying_l118_118777

theorem container_emptying (a b c : ℕ) : ∃ m n k : ℕ,
  (m = 0 ∨ n = 0 ∨ k = 0) ∧
  (∀ a' b' c', 
    (a' = a ∧ b' = b ∧ c' = c) ∨ 
    (a' + 2 * b' = a' ∧ b' = b ∧ c' + 2 * b' = c') ∨ 
    (a' + 2 * c' = a' ∧ b' + 2 * c' = b' ∧ c' = c') ∨ 
    (a + 2 * b' + c' = a' + 2 * m * (a + b') ∧ b' = n * (a + b') ∧ c' = k * (a + b')) 
  -> (a' = 0 ∨ b' = 0 ∨ c' = 0)) :=
sorry

end container_emptying_l118_118777


namespace exponentiation_evaluation_l118_118687

theorem exponentiation_evaluation :
  (8^3 / 8^2) * 2^10 = 8192 := by
  sorry

end exponentiation_evaluation_l118_118687


namespace fraction_value_l118_118831

theorem fraction_value :
  (1^4 + 2009^4 + 2010^4) / (1^2 + 2009^2 + 2010^2) = 4038091 := by
  sorry

end fraction_value_l118_118831


namespace frog_escape_probability_l118_118274

def P : ℕ → ℚ
| 0 := 0
| 12 := 1
| n := if 0 < n ∧ n < 12 then (↑(n + 1) / 13) * P (n - 1) + (1 - (↑(n + 1) / 13)) * P (n + 1) else 0

theorem frog_escape_probability : P 3 = 101 / 223 := by sorry

end frog_escape_probability_l118_118274


namespace julie_upstream_distance_l118_118145

noncomputable def speed_of_stream : ℝ := 0.5
noncomputable def distance_downstream : ℝ := 72
noncomputable def time_spent : ℝ := 4
noncomputable def speed_of_julie_in_still_water : ℝ := 17.5
noncomputable def distance_upstream : ℝ := 68

theorem julie_upstream_distance :
  (distance_upstream / (speed_of_julie_in_still_water - speed_of_stream) = time_spent) ∧
  (distance_downstream / (speed_of_julie_in_still_water + speed_of_stream) = time_spent) →
  distance_upstream = 68 :=
by 
  sorry

end julie_upstream_distance_l118_118145


namespace max_value_f_on_interval_l118_118514

noncomputable def f (x : ℝ) : ℝ := Real.exp x - x

theorem max_value_f_on_interval : 
  ∃ x ∈ Set.Icc (0 : ℝ) 1, ∀ y ∈ Set.Icc (0 : ℝ) 1, f y ≤ f x ∧ f x = Real.exp 1 - 1 := sorry

end max_value_f_on_interval_l118_118514


namespace find_m_for_local_minimum_l118_118317

noncomputable def f (x m : ℝ) := x * (x - m) ^ 2

theorem find_m_for_local_minimum :
  ∃ m : ℝ, (∀ x : ℝ, (x = 1 → deriv (λ x => f x m) x = 0) ∧ 
                  (x = 1 → deriv (deriv (λ x => f x m)) x > 0)) ∧ 
            m = 1 :=
by
  sorry

end find_m_for_local_minimum_l118_118317


namespace fruit_seller_gain_l118_118511

-- Define necessary variables
variables {C S : ℝ} (G : ℝ)

-- Given conditions
def selling_price_def (C : ℝ) : ℝ := 1.25 * C
def total_cost_price (C : ℝ) : ℝ := 150 * C
def total_selling_price (C : ℝ) : ℝ := 150 * (selling_price_def C)
def gain (C : ℝ) : ℝ := total_selling_price C - total_cost_price C

-- Statement to prove: number of apples' selling price gained by the fruit-seller is 30
theorem fruit_seller_gain : G = 30 ↔ gain C = G * (selling_price_def C) :=
by
  sorry

end fruit_seller_gain_l118_118511


namespace john_speed_above_limit_l118_118737

theorem john_speed_above_limit (distance : ℝ) (time : ℝ) (speed_limit : ℝ) 
  (h1 : distance = 150) (h2 : time = 2) (h3 : speed_limit = 60) : 
  (distance / time) - speed_limit = 15 :=
by
  -- steps to show the proof
  sorry

end john_speed_above_limit_l118_118737


namespace problem_k_value_l118_118619

theorem problem_k_value (a b c : ℕ) (h1 : a + b / c = 101) (h2 : a / c + b = 68) :
  (a + b) / c = 13 :=
sorry

end problem_k_value_l118_118619


namespace square_of_area_of_equilateral_triangle_l118_118082

noncomputable def equilateral_triangle_area_squared (x1 x2 x3 y1 y2 y3 : ℝ) (s : ℝ) :=
  let centroid_eq := (x1 + x2 + x3) / 3 = 1 ∧ (y1 + y2 + y3) / 3 = 1
  let vertices_on_hyperbola := x1 * y1 = 3 ∧ x2 * y2 = 3 ∧ x3 * y3 = 3
  let side_length := s = Math.sqrt ((x1 - x2) ^ 2 + (y1 - y2) ^ 2)
  ∧ s = Math.sqrt ((x2 - x3) ^ 2 + (y2 - y3) ^ 2)
  ∧ s = Math.sqrt ((x3 - x1) ^ 2 + (y3 - y1) ^ 2)
  let area_sq := (s * s * Math.sqrt 3 / 4) ^ 2
  centroid_eq ∧ vertices_on_hyperbola ∧ side_length → area_sq

theorem square_of_area_of_equilateral_triangle (x1 x2 x3 y1 y2 y3 s : ℝ) :
  equilateral_triangle_area_squared x1 x2 x3 y1 y2 y3 s :=
sorry

end square_of_area_of_equilateral_triangle_l118_118082


namespace values_of_x_for_g_l118_118588

def g (x : ℝ) : ℝ := x^2 - 5 * x

theorem values_of_x_for_g (x : ℝ) :
  g (g x) = g x ↔ x = 0 ∨ x = 5 ∨ x = 6 ∨ x = -1 := by
    sorry

end values_of_x_for_g_l118_118588


namespace lucas_income_36000_l118_118914

variable (q I : ℝ)

-- Conditions as Lean 4 definitions
def tax_below_30000 : ℝ := 0.01 * q * 30000
def tax_above_30000 (I : ℝ) : ℝ := 0.01 * (q + 3) * (I - 30000)
def total_tax (I : ℝ) : ℝ := tax_below_30000 q + tax_above_30000 q I
def total_tax_condition (I : ℝ) : Prop := total_tax q I = 0.01 * (q + 0.5) * I

theorem lucas_income_36000 (h : total_tax_condition q I) : I = 36000 := by
  sorry

end lucas_income_36000_l118_118914


namespace calculate_A_l118_118415

theorem calculate_A (D B E C A : ℝ) :
  D = 2 * 4 →
  B = 2 * D →
  E = 7 * 2 →
  C = 7 * E →
  A^2 = B * C →
  A = 28 * Real.sqrt 2 :=
by
  sorry

end calculate_A_l118_118415


namespace percentage_of_apples_after_removal_l118_118485

-- Declare the initial conditions as Lean definitions
def initial_apples : Nat := 12
def initial_oranges : Nat := 23
def removed_oranges : Nat := 15

-- Calculate the new totals
def new_oranges : Nat := initial_oranges - removed_oranges
def new_total_fruit : Nat := initial_apples + new_oranges

-- Define the expected percentage of apples as a real number
def expected_percentage_apples : Nat := 60

-- Prove that the percentage of apples after removing the specified number of oranges is 60%
theorem percentage_of_apples_after_removal :
  (initial_apples * 100 / new_total_fruit) = expected_percentage_apples := by
  sorry

end percentage_of_apples_after_removal_l118_118485


namespace michael_brought_5000_rubber_bands_l118_118295

noncomputable def totalRubberBands
  (small_band_count : ℕ) (large_band_count : ℕ)
  (small_ball_count : ℕ := 22) (large_ball_count : ℕ := 13)
  (rubber_bands_per_small : ℕ := 50) (rubber_bands_per_large : ℕ := 300) 
: ℕ :=
small_ball_count * rubber_bands_per_small + large_ball_count * rubber_bands_per_large

theorem michael_brought_5000_rubber_bands :
  totalRubberBands 22 13 = 5000 := by
  sorry

end michael_brought_5000_rubber_bands_l118_118295


namespace arithmetic_expression_evaluation_l118_118324

theorem arithmetic_expression_evaluation : 
  2000 - 80 + 200 - 120 = 2000 := by
  sorry

end arithmetic_expression_evaluation_l118_118324


namespace mass_proportion_l118_118857

namespace DensityMixture

variables (k m_1 m_2 m_3 : ℝ)
def rho_1 := 6 * k
def rho_2 := 3 * k
def rho_3 := 2 * k
def arithmetic_mean := (rho_1 k + rho_2 k + rho_3 k) / 3
def density_mixture := (m_1 + m_2 + m_3) / 
    (m_1 / rho_1 k + m_2 / rho_2 k + m_3 / rho_3 k)
def mass_ratio_condition := m_1 / m_2 ≥ 3.5

theorem mass_proportion 
  (k_pos : 0 < k)
  (mass_cond : mass_ratio_condition k m_1 m_2) :
  ∃ (x y : ℝ), (4 * x + 15 * y = 7) ∧ (density_mixture k m_1 m_2 m_3 = arithmetic_mean k) ∧ mass_cond := 
sorry

end DensityMixture

end mass_proportion_l118_118857


namespace stimulus_check_total_l118_118298

def find_stimulus_check (T : ℝ) : Prop :=
  let amount_after_wife := T * (3/5)
  let amount_after_first_son := amount_after_wife * (3/5)
  let amount_after_second_son := amount_after_first_son * (3/5)
  amount_after_second_son = 432

theorem stimulus_check_total (T : ℝ) : find_stimulus_check T → T = 2000 := by
  sorry

end stimulus_check_total_l118_118298


namespace smallest_add_to_multiple_of_4_l118_118056

theorem smallest_add_to_multiple_of_4 : ∃ n : ℕ, n > 0 ∧ (587 + n) % 4 = 0 ∧ ∀ m : ℕ, m > 0 ∧ (587 + m) % 4 = 0 → n ≤ m :=
  sorry

end smallest_add_to_multiple_of_4_l118_118056


namespace hot_dog_cost_l118_118221

variables (h d : ℝ)

theorem hot_dog_cost :
  (3 * h + 4 * d = 10) →
  (2 * h + 3 * d = 7) →
  d = 1 :=
by
  intros h_eq d_eq
  -- Proof skipped
  sorry

end hot_dog_cost_l118_118221


namespace standard_eq_of_largest_circle_l118_118005

theorem standard_eq_of_largest_circle 
  (m : ℝ)
  (hm : 0 < m) :
  ∃ r : ℝ, 
  (∀ x y : ℝ, (x^2 + (y - 1)^2 = 8) ↔ 
      (x^2 + (y - 1)^2 = r)) :=
sorry

end standard_eq_of_largest_circle_l118_118005


namespace find_m_l118_118842

noncomputable def ellipse : set (ℝ × ℝ) := { p | (p.1)^2 / 3 + (p.2)^2 = 1 }
def f1 : ℝ × ℝ := (- real.sqrt 2, 0)
def f2 : ℝ × ℝ := (real.sqrt 2, 0)

def line (m : ℝ) : set (ℝ × ℝ) := { p | p.2 = p.1 + m }

theorem find_m (m : ℝ) :
  (∃ A B : ℝ × ℝ, A ∈ ellipse ∧ B ∈ ellipse ∧ A ∈ line m ∧ B ∈ line m ∧
   2 * (abs ((f1.1 * (A.2 - B.2) + A.1 * (B.2 - f1.2) + B.1 * (f1.2 - A.2)) / 2))
 = abs ((f2.1 * (A.2 - B.2) + A.1 * (B.2 - f2.2) + B.1 * (f2.2 - A.2)) / 2)) →
  m = - real.sqrt 2 / 3 :=
sorry

end find_m_l118_118842


namespace greatest_possible_x_lcm_l118_118535

theorem greatest_possible_x_lcm (x : ℕ) (h : Nat.lcm (Nat.lcm x 15) 21 = 105): x = 105 := 
sorry

end greatest_possible_x_lcm_l118_118535


namespace remainder_of_first_105_sum_div_5280_l118_118334

theorem remainder_of_first_105_sum_div_5280:
  let n := 105
  let d := 5280
  let sum := n * (n + 1) / 2
  sum % d = 285 := by
  sorry

end remainder_of_first_105_sum_div_5280_l118_118334


namespace sin_neg_1920_eq_neg_sqrt3_div_2_l118_118058

open Real

theorem sin_neg_1920_eq_neg_sqrt3_div_2 : sin (-1920 * pi / 180) = - (sqrt 3 / 2) :=
by
  -- Proof omitted, focuses on the statement
  sorry

end sin_neg_1920_eq_neg_sqrt3_div_2_l118_118058


namespace continuity_at_x_2_l118_118389

noncomputable def f (x : ℝ) (b : ℝ) : ℝ :=
if x > 2 then x + 4 else 3 * x + b

theorem continuity_at_x_2 (b : ℝ) : 
  (∀ ε > 0, ∃ δ > 0, ∀ x, abs (x - 2) < δ → abs (f x b - f 2 b) < ε) ↔ b = 0 :=
by
  sorry

end continuity_at_x_2_l118_118389


namespace total_money_taken_l118_118518

def individual_bookings : ℝ := 12000
def group_bookings : ℝ := 16000
def returned_due_to_cancellations : ℝ := 1600

def total_taken (individual_bookings : ℝ) (group_bookings : ℝ) (returned_due_to_cancellations : ℝ) : ℝ :=
  (individual_bookings + group_bookings) - returned_due_to_cancellations

theorem total_money_taken :
  total_taken individual_bookings group_bookings returned_due_to_cancellations = 26400 := by
  sorry

end total_money_taken_l118_118518


namespace calculation_correct_l118_118540

theorem calculation_correct : 67897 * 67898 - 67896 * 67899 = 2 := by
  sorry

end calculation_correct_l118_118540


namespace simon_age_l118_118500

theorem simon_age : 
  ∃ s : ℕ, 
  ∀ a : ℕ,
    a = 30 → 
    s = (a / 2) - 5 → 
    s = 10 := 
by
  sorry

end simon_age_l118_118500


namespace find_a_perpendicular_lines_l118_118864

theorem find_a_perpendicular_lines 
  (a : ℤ)
  (l1 : ∀ x y : ℤ, a * x + 4 * y + 7 = 0)
  (l2 : ∀ x y : ℤ, 2 * x - 3 * y - 1 = 0) : 
  (∃ a : ℤ, a = 6) :=
by sorry

end find_a_perpendicular_lines_l118_118864


namespace chord_length_perpendicular_bisector_l118_118663

theorem chord_length_perpendicular_bisector (r : ℝ) (h : r = 10) :
  ∃ (CD : ℝ), CD = 10 * Real.sqrt 3 :=
by
  -- The proof is omitted.
  sorry

end chord_length_perpendicular_bisector_l118_118663


namespace sin_double_angle_plus_pi_div_two_l118_118108

open Real

theorem sin_double_angle_plus_pi_div_two (θ : ℝ) (h₀ : 0 < θ) (h₁ : θ < π / 2) (h₂ : sin θ = 1 / 3) :
  sin (2 * θ + π / 2) = 7 / 9 :=
by
  sorry

end sin_double_angle_plus_pi_div_two_l118_118108


namespace tracy_sold_paintings_l118_118053

-- Definitions of conditions
def total_customers := 20
def first_group_customers := 4
def paintings_per_first_group_customer := 2
def second_group_customers := 12
def paintings_per_second_group_customer := 1
def third_group_customers := 4
def paintings_per_third_group_customer := 4

-- Statement of the problem
theorem tracy_sold_paintings :
  (first_group_customers * paintings_per_first_group_customer) +
  (second_group_customers * paintings_per_second_group_customer) +
  (third_group_customers * paintings_per_third_group_customer) = 36 :=
by
  sorry

end tracy_sold_paintings_l118_118053


namespace team_OT_matches_l118_118923

variable (T x M: Nat)

-- Condition: Team C played T matches in the first week.
def team_C_matches_T : Nat := T

-- Condition: Team C played x matches in the first week.
def team_C_matches_x : Nat := x

-- Condition: Team O played M matches in the first week.
def team_O_matches_M : Nat := M

-- Condition: Team C has not played against Team A.
axiom C_not_played_A : ¬ (team_C_matches_T = team_C_matches_x)

-- Condition: Team B has not played against a specified team (interpreted).
axiom B_not_played_specified : ∀ x, ¬ (team_C_matches_x = x)

-- The proof for the number of matches played by team \(\overrightarrow{OT}\).
theorem team_OT_matches : T = 4 := 
    sorry

end team_OT_matches_l118_118923


namespace largest_number_not_sum_of_two_composites_l118_118104

-- Definitions related to composite numbers and natural numbers
def is_composite (n : ℕ) : Prop :=
  n > 1 ∧ ∃ d : ℕ, d ∣ n ∧ d ≠ 1 ∧ d ≠ n

def cannot_be_sum_of_two_composites (n : ℕ) : Prop :=
  ¬(∃ a b : ℕ, is_composite a ∧ is_composite b ∧ n = a + b)

-- Lean statement of the problem
theorem largest_number_not_sum_of_two_composites : ∀ n : ℕ,
  (∀ m : ℕ, m > n → cannot_be_sum_of_two_composites m → false) → n = 11 :=
by
  sorry

end largest_number_not_sum_of_two_composites_l118_118104


namespace remainder_when_divided_by_30_l118_118309

theorem remainder_when_divided_by_30 (x : ℤ) : 
  (4 + x) % 8 = 9 % 8 ∧
  (6 + x) % 27 = 4 % 27 ∧
  (8 + x) % 125 = 49 % 125 
  → x % 30 = 1 % 30 := by
  sorry

end remainder_when_divided_by_30_l118_118309


namespace part1_part2_l118_118949

-- Part 1: Expression simplification
theorem part1 (a : ℝ) : (a - 3)^2 + a * (4 - a) = -2 * a + 9 := 
by
  sorry

-- Part 2: Solution set of inequalities
theorem part2 (x : ℝ) : 
  (3 * x - 5 < x + 1) ∧ (2 * (2 * x - 1) ≥ 3 * x - 4) ↔ (-2 ≤ x ∧ x < 3) := 
by
  sorry

end part1_part2_l118_118949


namespace eval_expr_at_neg3_l118_118817

theorem eval_expr_at_neg3 : 
  (5 + 2 * (-3) * ((-3) + 5) - 5^2) / (2 * (-3) - 5 + 2 * (-3)^3) = 32 / 65 := 
by 
  sorry

end eval_expr_at_neg3_l118_118817


namespace cloud9_total_money_l118_118520

-- Definitions
def I : ℕ := 12000
def G : ℕ := 16000
def R : ℕ := 1600

-- Total money taken after cancellations
def T : ℕ := I + G - R

-- Theorem stating that T is 26400
theorem cloud9_total_money : T = 26400 := by
  unfold T I G R
  sorry

end cloud9_total_money_l118_118520


namespace greatest_b_value_l118_118644

theorem greatest_b_value (b : ℤ) : 
  (∀ x : ℝ, x^2 + (b:ℝ) * x + 15 ≠ -9) ↔ b = 9 :=
sorry

end greatest_b_value_l118_118644


namespace ascending_order_perimeters_l118_118541

noncomputable def hypotenuse (r : ℝ) : ℝ := r * Real.sqrt 2

noncomputable def perimeter_P (r : ℝ) : ℝ := (2 + 3 * Real.sqrt 2) * r
noncomputable def perimeter_Q (r : ℝ) : ℝ := (6 + Real.sqrt 2) * r
noncomputable def perimeter_R (r : ℝ) : ℝ := (4 + 3 * Real.sqrt 2) * r

theorem ascending_order_perimeters (r : ℝ) (h_r_pos : 0 < r) : 
  perimeter_P r < perimeter_Q r ∧ perimeter_Q r < perimeter_R r := by
  sorry

end ascending_order_perimeters_l118_118541


namespace percent_of_number_l118_118954

theorem percent_of_number (N : ℝ) (h : (4 / 5) * (3 / 8) * N = 24) : 2.5 * N = 200 :=
by
  sorry

end percent_of_number_l118_118954


namespace find_third_root_l118_118926

variables (a b : ℝ)

def poly (x : ℝ) : ℝ := a * x^3 + (a + 3 * b) * x^2 + (b - 4 * a) * x + (10 - a)

def root1 := -3
def root2 := 4

axiom root1_cond : poly a b root1 = 0
axiom root2_cond : poly a b root2 = 0

theorem find_third_root (a b : ℝ) (h1 : poly a b root1 = 0) (h2 : poly a b root2 = 0) : 
  ∃ r3 : ℝ, r3 = -1/2 :=
sorry

end find_third_root_l118_118926


namespace ratio_of_a_and_b_l118_118478

theorem ratio_of_a_and_b (x y a b : ℝ) (h1 : x / y = 3) (h2 : (2 * a - x) / (3 * b - y) = 3) : a / b = 9 / 2 :=
by
  sorry

end ratio_of_a_and_b_l118_118478


namespace sum_of_m_n_l118_118348

-- Define the setup for the problem
def side_length_of_larger_square := 3
def side_length_of_smaller_square := 1
def side_length_of_given_rectangle_l1 := 1
def side_length_of_given_rectangle_l2 := 3
def total_area_of_larger_square := side_length_of_larger_square * side_length_of_larger_square
def area_of_smaller_square := side_length_of_smaller_square * side_length_of_smaller_square
def area_of_given_rectangle := side_length_of_given_rectangle_l1 * side_length_of_given_rectangle_l2

-- Define the variable for the area of rectangle R
def area_of_R := total_area_of_larger_square - (area_of_smaller_square + area_of_given_rectangle)

-- Given the problem statement, we need to find m and n such that the area of R is m/n.
def m := 5
def n := 1

-- We need to prove that m + n = 6 given these conditions
theorem sum_of_m_n : m + n = 6 := by
  sorry

end sum_of_m_n_l118_118348


namespace sum_of_digits_of_square_99999_l118_118057

noncomputable def sum_of_digits (n : ℕ) : ℕ :=
  n.digits 10 |>.sum

theorem sum_of_digits_of_square_99999 : sum_of_digits ((99999 : ℕ)^2) = 45 := by
  sorry

end sum_of_digits_of_square_99999_l118_118057


namespace length_segment_ZZ_l118_118639

variable (Z : ℝ × ℝ) (Z' : ℝ × ℝ)

theorem length_segment_ZZ' 
  (h_Z : Z = (-5, 3)) (h_Z' : Z' = (5, 3)) : 
  dist Z Z' = 10 := by
  sorry

end length_segment_ZZ_l118_118639


namespace infinite_series_sum_l118_118088

theorem infinite_series_sum (a r : ℝ) (h₀ : -1 < r) (h₁ : r < 1) :
    (∑' n, if (n % 2 = 0) then a * r^(n/2) else a^2 * r^((n+1)/2)) = (a * (1 + a * r))/(1 - r^2) :=
by
  sorry

end infinite_series_sum_l118_118088


namespace longest_side_of_triangle_l118_118079

variable (x y : ℝ)

def side1 := 10
def side2 := 2*y + 3
def side3 := 3*x + 2

theorem longest_side_of_triangle
  (h_perimeter : side1 + side2 + side3 = 45)
  (h_side2_pos : side2 > 0)
  (h_side3_pos : side3 > 0) :
  side3 = 32 :=
sorry

end longest_side_of_triangle_l118_118079


namespace intersection_A_B_eq_union_A_B_eq_intersection_A_C_U_B_eq_l118_118107

def U := ℝ
def A : Set ℝ := {x | 0 ≤ x ∧ x < 5}
def B : Set ℝ := {x | -2 ≤ x ∧ x < 4}
def C_U_B : Set ℝ := {x | x < -2 ∨ x ≥ 4}

theorem intersection_A_B_eq : A ∩ B = {x | 0 ≤ x ∧ x < 4} := by
  sorry

theorem union_A_B_eq : A ∪ B = {x | -2 ≤ x ∧ x < 5} := by
  sorry

theorem intersection_A_C_U_B_eq : A ∩ C_U_B = {x | 4 ≤ x ∧ x < 5} := by
  sorry

end intersection_A_B_eq_union_A_B_eq_intersection_A_C_U_B_eq_l118_118107


namespace total_population_l118_118662

theorem total_population (P : ℝ) : 0.96 * P = 23040 → P = 24000 :=
by
  sorry

end total_population_l118_118662


namespace karen_savings_over_30_years_l118_118740

theorem karen_savings_over_30_years 
  (P_exp : ℕ) (L_exp : ℕ) 
  (P_cheap : ℕ) (L_cheap : ℕ) 
  (T : ℕ)
  (hP_exp : P_exp = 300)
  (hL_exp : L_exp = 15)
  (hP_cheap : P_cheap = 120)
  (hL_cheap : L_cheap = 5)
  (hT : T = 30) : 
  (P_cheap * (T / L_cheap) - P_exp * (T / L_exp)) = 120 := 
by 
  sorry

end karen_savings_over_30_years_l118_118740


namespace max_ab_condition_max_ab_value_l118_118722

theorem max_ab_condition (a b : ℝ) (h1 : a + b = 1) (h2 : a > 0) (h3 : b > 0) : ab ≤ 1 / 4 :=
sorry

theorem max_ab_value (a b : ℝ) (h1 : a + b = 1) (h2 : a = b) : ab = 1 / 4 :=
sorry

end max_ab_condition_max_ab_value_l118_118722


namespace simplify_sqrt_expression_l118_118615

theorem simplify_sqrt_expression :
  (Real.sqrt (8 + 6 * Real.sqrt 2) + Real.sqrt (8 - 6 * Real.sqrt 2)) = 6 :=
by
  sorry

end simplify_sqrt_expression_l118_118615


namespace Marilyn_end_caps_l118_118294

def starting_caps := 51
def shared_caps := 36
def ending_caps := starting_caps - shared_caps

theorem Marilyn_end_caps : ending_caps = 15 := by
  -- proof omitted
  sorry

end Marilyn_end_caps_l118_118294


namespace simplify_sqrt_product_l118_118616

theorem simplify_sqrt_product : (Real.sqrt (3 * 5) * Real.sqrt (3 ^ 5 * 5 ^ 5) = 3375) :=
  sorry

end simplify_sqrt_product_l118_118616


namespace dash_cam_mounts_max_profit_l118_118604

noncomputable def monthly_profit (x t : ℝ) : ℝ :=
  (48 + t / (2 * x)) * x - 32 * x - 3 - t

theorem dash_cam_mounts_max_profit :
  ∃ (x t : ℝ), 1 < x ∧ x < 3 ∧ x = 3 - 2 / (t + 1) ∧
  monthly_profit x t = 37.5 := by
sorry

end dash_cam_mounts_max_profit_l118_118604


namespace percent_diamond_jewels_l118_118083

def percent_beads : ℝ := 0.3
def percent_ruby_jewels : ℝ := 0.5

theorem percent_diamond_jewels (percent_beads percent_ruby_jewels : ℝ) : 
  (1 - percent_beads) * (1 - percent_ruby_jewels) = 0.35 :=
by
  -- We insert the proof steps here
  sorry

end percent_diamond_jewels_l118_118083


namespace initial_kittens_l118_118047

-- Define the number of kittens given to Jessica and Sara, and the number of kittens currently Tim has.
def kittens_given_to_Jessica : ℕ := 3
def kittens_given_to_Sara : ℕ := 6
def kittens_left_with_Tim : ℕ := 9

-- Define the theorem to prove the initial number of kittens Tim had.
theorem initial_kittens (kittens_given_to_Jessica kittens_given_to_Sara kittens_left_with_Tim : ℕ) 
    (h1 : kittens_given_to_Jessica = 3)
    (h2 : kittens_given_to_Sara = 6)
    (h3 : kittens_left_with_Tim = 9) :
    (kittens_given_to_Jessica + kittens_given_to_Sara + kittens_left_with_Tim) = 18 := 
    sorry

end initial_kittens_l118_118047


namespace least_number_of_froods_l118_118277

def dropping_score (n : ℕ) : ℕ := (n * (n + 1)) / 2
def eating_score (n : ℕ) : ℕ := 15 * n

theorem least_number_of_froods : ∃ n : ℕ, (dropping_score n > eating_score n) ∧ (∀ m < n, dropping_score m ≤ eating_score m) :=
  exists.intro 30 
    (and.intro 
      (by simp [dropping_score, eating_score]; linarith)
      (by intros m hmn; simp [dropping_score, eating_score]; linarith [hmn]))

end least_number_of_froods_l118_118277


namespace num_divisors_not_divisible_by_three_l118_118408

theorem num_divisors_not_divisible_by_three (n : ℕ) (h : n = 180) : 
  ∃ d, d = 6 ∧ ∀ k, k > 0 ∧ k ∣ n → (¬ 3 ∣ k → k ∈ {d | d ∣ n ∧ ¬ 3 ∣ d}) :=
by 
  have h180 : 180 = 2^2 * 3^2 * 5, by norm_num
  rw h at h180
  use 6
  sorry

end num_divisors_not_divisible_by_three_l118_118408


namespace average_shift_l118_118835

variable (a b c : ℝ)

-- Given condition: The average of the data \(a\), \(b\), \(c\) is 5.
def average_is_five := (a + b + c) / 3 = 5

-- Define the statement to prove: The average of the data \(a-2\), \(b-2\), \(c-2\) is 3.
theorem average_shift (h : average_is_five a b c) : ((a - 2) + (b - 2) + (c - 2)) / 3 = 3 :=
by
  sorry

end average_shift_l118_118835


namespace doughnut_machine_completion_time_l118_118342

noncomputable def start_time : ℕ := 8 * 60 + 30  -- 8:30 AM in minutes
noncomputable def one_third_time : ℕ := 11 * 60 + 10  -- 11:10 AM in minutes
noncomputable def total_time_minutes : ℕ := 8 * 60  -- 8 hours in minutes
noncomputable def expected_completion_time : ℕ := 16 * 60 + 30  -- 4:30 PM in minutes

theorem doughnut_machine_completion_time :
  one_third_time - start_time = total_time_minutes / 3 →
  start_time + total_time_minutes = expected_completion_time :=
by
  intros h1
  sorry

end doughnut_machine_completion_time_l118_118342


namespace solve_for_x_l118_118953

theorem solve_for_x (x : ℝ) (h : 3 * x + 36 = 48) : x = 4 :=
by
  sorry

end solve_for_x_l118_118953


namespace inequality_proof_l118_118397

theorem inequality_proof
  (x y z : ℝ)
  (hx : x > 0)
  (hy : y > 0)
  (hz : z > 0)
  (hxyz : x * y * z ≥ 1) :
  (x^3 - x^2) / (x^5 + y^2 + z^2) +
  (y^5 - y^2) / (y^5 + z^2 + x^2) +
  (z^5 - z^2) / (z^5 + x^2 + y^2) ≥ 0 := 
sorry

end inequality_proof_l118_118397


namespace function_passes_through_point_l118_118708

noncomputable def special_function (a : ℝ) (x : ℝ) := a^(x - 1) + 1

theorem function_passes_through_point (a : ℝ) (h₁ : a > 0) (h₂ : a ≠ 1) :
  special_function a 1 = 2 :=
by
  -- skip the proof
  sorry

end function_passes_through_point_l118_118708


namespace sum_arithmetic_sequence_l118_118008

variable (a : ℕ → ℝ)

def is_arithmetic_sequence (a : ℕ → ℝ) : Prop :=
    ∃ d, ∀ n, a (n+1) = a n + d

-- The conditions
def condition_1 (a : ℕ → ℝ) : Prop :=
    (a 1 + a 2 + a 3 = 6)

def condition_2 (a : ℕ → ℝ) : Prop :=
    (a 10 + a 11 + a 12 = 9)

-- The Theorem statement
theorem sum_arithmetic_sequence :
    is_arithmetic_sequence a →
    condition_1 a →
    condition_2 a →
    (a 1 + a 2 + a 3 + a 4 + a 5 + a 6 + a 7 + a 8 + a 9 + a 10 + a 11 + a 12 = 30) :=
by
  intro h1 h2 h3
  sorry

end sum_arithmetic_sequence_l118_118008


namespace parallel_lines_slope_l118_118811

theorem parallel_lines_slope (k : ℝ) : 
  (∀ x : ℝ, 5 * x + 3 = (3 * k) * x + 7) → k = 5 / 3 := 
sorry

end parallel_lines_slope_l118_118811


namespace time_to_park_l118_118124

-- distance from house to market in miles
def d_market : ℝ := 5

-- distance from house to park in miles
def d_park : ℝ := 3

-- time to market in minutes
def t_market : ℝ := 30

-- assuming constant speed, calculate time to park
theorem time_to_park : (3 / 5) * 30 = 18 := by
  sorry

end time_to_park_l118_118124


namespace cistern_water_breadth_l118_118207

theorem cistern_water_breadth 
  (length width : ℝ) (wet_surface_area : ℝ) 
  (hl : length = 9) (hw : width = 6) (hwsa : wet_surface_area = 121.5) : 
  ∃ h : ℝ, 54 + 18 * h + 12 * h = 121.5 ∧ h = 2.25 := 
by 
  sorry

end cistern_water_breadth_l118_118207


namespace susan_avg_speed_l118_118480

theorem susan_avg_speed 
  (speed1 : ℕ)
  (distance1 : ℕ)
  (speed2 : ℕ)
  (distance2 : ℕ)
  (no_stops : Prop) 
  (H1 : speed1 = 15)
  (H2 : distance1 = 40)
  (H3 : speed2 = 60)
  (H4 : distance2 = 20)
  (H5 : no_stops) :
  let total_distance := distance1 + distance2
  let time1 := distance1 / speed1
  let time2 := distance2 / speed2
  let total_time := time1 + time2
  let avg_speed := total_distance / total_time
  avg_speed = 20 := by
  sorry

end susan_avg_speed_l118_118480


namespace percentage_range_l118_118071

noncomputable def minimum_maximum_percentage (x y z n m : ℝ) (hx1 : 0 < x) (hx2 : 0 < y) (hx3 : 0 < z) (hx4 : 0 < n) (hx5 : 0 < m)
    (h1 : 4 * x * n = y * m) 
    (h2 : x * n + y * m = z * (m + n)) 
    (h3 : 16 ≤ y - x ∧ y - x ≤ 20) 
    (h4 : 42 ≤ z ∧ z ≤ 60) : ℝ × ℝ := sorry

theorem percentage_range (x y z n m : ℝ) (hx1 : 0 < x) (hx2 : 0 < y) (hx3 : 0 < z) (hx4 : 0 < n) (hx5 : 0 < m)
    (h1 : 4 * x * n = y * m) 
    (h2 : x * n + y * m = z * (m + n)) 
    (h3 : 16 ≤ y - x ∧ y - x ≤ 20) 
    (h4 : 42 ≤ z ∧ z ≤ 60) : 
    minimum_maximum_percentage x y z n m hx1 hx2 hx3 hx4 hx5 h1 h2 h3 h4 = (12.5, 15) :=
sorry

end percentage_range_l118_118071


namespace num_ordered_pairs_l118_118236

theorem num_ordered_pairs : ∃! n : ℕ, n = 4 ∧ 
  ∃ (x y : ℤ), y = (x - 90)^2 - 4907 ∧ 
  (∃ m : ℕ, y = m^2) := 
sorry

end num_ordered_pairs_l118_118236


namespace intersection_of_sets_l118_118253

open Set

variable {x : ℝ}

theorem intersection_of_sets : 
  let A := {x : ℝ | x^2 - 4*x + 3 < 0}
  let B := {x : ℝ | x > 2}
  A ∩ B = {x : ℝ | 2 < x ∧ x < 3} :=
by
  sorry

end intersection_of_sets_l118_118253


namespace joan_mortgage_payback_months_l118_118281

-- Define the conditions and statement

def first_payment : ℕ := 100
def total_amount : ℕ := 2952400

theorem joan_mortgage_payback_months :
  ∃ n : ℕ, 100 * (3^n - 1) / (3 - 1) = 2952400 ∧ n = 10 :=
by
  sorry

end joan_mortgage_payback_months_l118_118281


namespace solution_for_a_l118_118413

theorem solution_for_a :
  ∀ a x : ℝ, (2 - a - x = 0) ∧ (2x + 1 = 3) → a = 1 := 
by
  intros a x h,
  cases h with h1 h2,
  have x_eq := by linarith,
  have a_eq := by linarith,
  exact a_eq

end solution_for_a_l118_118413


namespace fraction_exponent_product_l118_118643

theorem fraction_exponent_product :
  ( (5/6: ℚ)^2 * (2/3: ℚ)^3 = 50/243 ) :=
by
  sorry

end fraction_exponent_product_l118_118643


namespace range_of_a_l118_118574

theorem range_of_a (a : ℝ) : 
  (∀ x : ℝ, (a^2 - 1) * x^2 - (a - 1) * x - 1 < 0) ↔ (-3/5 < a ∧ a ≤ 1) := 
sorry

end range_of_a_l118_118574


namespace value_of_x_l118_118715

theorem value_of_x (x : ℝ) (h₁ : x > 0) (h₂ : x^3 = 19683) : x = 27 :=
sorry

end value_of_x_l118_118715


namespace range_of_m_l118_118676

theorem range_of_m (a b m : ℝ) (h1 : a > 0) (h2 : b > 1) (h3 : a + b = 2) (h4 : 4 / a + 1 / (b - 1) > m^2 + 8 * m) : -9 < m ∧ m < 1 :=
by
  sorry

end range_of_m_l118_118676


namespace tracy_sold_paintings_l118_118052

-- Definitions of conditions
def total_customers := 20
def first_group_customers := 4
def paintings_per_first_group_customer := 2
def second_group_customers := 12
def paintings_per_second_group_customer := 1
def third_group_customers := 4
def paintings_per_third_group_customer := 4

-- Statement of the problem
theorem tracy_sold_paintings :
  (first_group_customers * paintings_per_first_group_customer) +
  (second_group_customers * paintings_per_second_group_customer) +
  (third_group_customers * paintings_per_third_group_customer) = 36 :=
by
  sorry

end tracy_sold_paintings_l118_118052


namespace variance_of_temperatures_l118_118627

def temperatures : List ℕ := [28, 21, 22, 26, 28, 25]

noncomputable def mean (l : List ℕ) : ℚ :=
  (l.sum : ℚ) / l.length

noncomputable def variance (l : List ℕ) : ℚ :=
  let m := mean l
  (l.map (λ x => (x - m)^2)).sum / l.length

theorem variance_of_temperatures : variance temperatures = 22 / 3 := 
by
  sorry

end variance_of_temperatures_l118_118627


namespace simon_age_l118_118501

theorem simon_age : 
  ∃ s : ℕ, 
  ∀ a : ℕ,
    a = 30 → 
    s = (a / 2) - 5 → 
    s = 10 := 
by
  sorry

end simon_age_l118_118501


namespace scientific_notation_of_10900_l118_118635

theorem scientific_notation_of_10900 : ∃ (x : ℝ) (n : ℤ), 10900 = x * 10^n ∧ x = 1.09 ∧ n = 4 := by
  use 1.09
  use 4
  sorry

end scientific_notation_of_10900_l118_118635


namespace midpoint_product_l118_118875

theorem midpoint_product (x y : ℝ) (h1 : (4 : ℝ) = (x + 10) / 2) (h2 : (-2 : ℝ) = (-6 + y) / 2) : x * y = -4 := by
  sorry

end midpoint_product_l118_118875


namespace paul_work_days_l118_118605

theorem paul_work_days (P : ℕ) (h : 1 / P + 1 / 120 = 1 / 48) : P = 80 := 
by 
  sorry

end paul_work_days_l118_118605


namespace total_legs_l118_118531

-- Define the conditions
def chickens : Nat := 7
def sheep : Nat := 5
def legs_chicken : Nat := 2
def legs_sheep : Nat := 4

-- State the problem as a theorem
theorem total_legs :
  chickens * legs_chicken + sheep * legs_sheep = 34 :=
by
  sorry -- Proof not provided

end total_legs_l118_118531


namespace log_neq_x_minus_one_l118_118912

theorem log_neq_x_minus_one (x : ℝ) (h₁ : 0 < x) : Real.log x ≠ x - 1 :=
sorry

end log_neq_x_minus_one_l118_118912


namespace sum_a_b_l118_118678

theorem sum_a_b (a b : ℕ) (h_pos_a : a > 0) (h_pos_b : b > 2) (h_bound : a^b < 500)
  (h_max : ∀ a' b', a' > 0 → b' > 2 → a'^b' < 500 → a'^b' ≤ a^b) :
  a + b = 8 :=
by sorry

end sum_a_b_l118_118678


namespace total_money_taken_l118_118517

def individual_bookings : ℝ := 12000
def group_bookings : ℝ := 16000
def returned_due_to_cancellations : ℝ := 1600

def total_taken (individual_bookings : ℝ) (group_bookings : ℝ) (returned_due_to_cancellations : ℝ) : ℝ :=
  (individual_bookings + group_bookings) - returned_due_to_cancellations

theorem total_money_taken :
  total_taken individual_bookings group_bookings returned_due_to_cancellations = 26400 := by
  sorry

end total_money_taken_l118_118517


namespace line_intersects_parabola_at_one_point_l118_118772
   
   theorem line_intersects_parabola_at_one_point (k : ℝ) :
     (∃ y : ℝ, (x = 3 * y^2 - 7 * y + 2 ∧ x = k) → x = k) ↔ k = (-25 / 12) :=
   by
     -- your proof goes here
     sorry
   
end line_intersects_parabola_at_one_point_l118_118772


namespace heather_blocks_l118_118255

theorem heather_blocks (x : ℝ) (h1 : x + 41 = 127) : x = 86 := by
  sorry

end heather_blocks_l118_118255


namespace clock_angle_7_15_l118_118332

noncomputable def hour_angle_at (hour : ℕ) (minutes : ℕ) : ℝ :=
  hour * 30 + (minutes * 0.5)

noncomputable def minute_angle_at (minutes : ℕ) : ℝ :=
  minutes * 6

noncomputable def small_angle (angle1 angle2 : ℝ) : ℝ :=
  let diff := abs (angle1 - angle2)
  if diff <= 180 then diff else 360 - diff

theorem clock_angle_7_15 : small_angle (hour_angle_at 7 15) (minute_angle_at 15) = 127.5 :=
by
  sorry

end clock_angle_7_15_l118_118332


namespace gcd_765432_654321_l118_118827

-- Define the two integers 765432 and 654321
def a : ℕ := 765432
def b : ℕ := 654321

-- State the main theorem to prove the gcd
theorem gcd_765432_654321 : Nat.gcd a b = 3 := 
by 
  sorry

end gcd_765432_654321_l118_118827


namespace solve_for_x_l118_118125

theorem solve_for_x (x : ℝ) (h : x / 6 = 15 / 10) : x = 9 :=
by
  sorry

end solve_for_x_l118_118125


namespace greatest_prime_factor_294_l118_118055

theorem greatest_prime_factor_294 : ∃ p, Nat.Prime p ∧ p ∣ 294 ∧ ∀ q, Nat.Prime q ∧ q ∣ 294 → q ≤ p := 
by
  let prime_factors := [2, 3, 7]
  have h1 : 294 = 2 * 3 * 7 * 7 := by
    -- Proof of factorization should be inserted here
    sorry

  have h2 : ∀ p, p ∣ 294 → p = 2 ∨ p = 3 ∨ p = 7 := by
    -- Proof of prime factor correctness should be inserted here
    sorry

  use 7
  -- Prove 7 is the greatest prime factor here
  sorry

end greatest_prime_factor_294_l118_118055


namespace greatest_possible_a_l118_118314

theorem greatest_possible_a (a : ℕ) :
  (∃ x : ℤ, x^2 + a * x = -28) → a = 29 :=
sorry

end greatest_possible_a_l118_118314


namespace smallest_a1_value_l118_118880

noncomputable def a_seq (n : ℕ) : ℝ :=
if n = 0 then 29 / 98 else if n > 0 then 15 * a_seq (n - 1) - 2 * n else 0

theorem smallest_a1_value :
  (∃ f : ℕ → ℝ, (∀ n > 0, f n = 15 * f (n - 1) - 2 * n) ∧ (∀ n, f n > 0) ∧ (f 1 = 29 / 98)) :=
sorry

end smallest_a1_value_l118_118880


namespace distance_to_workplace_l118_118310

def driving_speed : ℕ := 40
def driving_time : ℕ := 3
def total_distance := driving_speed * driving_time
def one_way_distance := total_distance / 2

theorem distance_to_workplace : one_way_distance = 60 := by
  sorry

end distance_to_workplace_l118_118310


namespace Peter_work_rate_l118_118750

theorem Peter_work_rate:
  ∀ (m p j : ℝ),
    (m + p + j) * 20 = 1 →
    (m + p + j) * 10 = 0.5 →
    (p + j) * 10 = 0.5 →
    j * 15 = 0.5 →
    p * 60 = 1 :=
by
  intros m p j h1 h2 h3 h4
  sorry

end Peter_work_rate_l118_118750


namespace smallest_k_l118_118468

theorem smallest_k :
  ∃ k : ℤ, k > 1 ∧ k % 13 = 1 ∧ k % 8 = 1 ∧ k % 4 = 1 ∧ k = 105 :=
by
  sorry

end smallest_k_l118_118468


namespace multiply_by_3_l118_118976

variable (x : ℕ)  -- Declare x as a natural number

-- Define the conditions
def condition : Prop := x + 14 = 56

-- The goal to prove
theorem multiply_by_3 (h : condition x) : 3 * x = 126 := sorry

end multiply_by_3_l118_118976


namespace mass_of_compound_l118_118333

-- Constants as per the conditions
def molecular_weight : ℕ := 444           -- The molecular weight in g/mol.
def number_of_moles : ℕ := 6             -- The number of moles.

-- Defining the main theorem we want to prove.
theorem mass_of_compound : (number_of_moles * molecular_weight) = 2664 := by 
  sorry

end mass_of_compound_l118_118333


namespace sum_coefficients_l118_118200

theorem sum_coefficients (a : ℤ) (f : ℤ → ℤ) :
  f x = (1 - 2 * x)^7 ∧ a_0 = f 0 ∧ a_1_plus_a_7 = f 1 - f 0 
→ a_1_plus_a_7 = -2 :=
by sorry

end sum_coefficients_l118_118200


namespace sinB_law_of_sines_l118_118869

variable (A B C : Type)
variable [MetricSpace A] [MetricSpace B] [MetricSpace C]

-- Assuming a triangle with sides and angles as described
variable (a b : ℝ) (sinA sinB : ℝ)
variable (h₁ : a = 3) (h₂ : b = 5) (h₃ : sinA = 1 / 3)

theorem sinB_law_of_sines : sinB = 5 / 9 :=
by
  -- Placeholder for the proof
  sorry

end sinB_law_of_sines_l118_118869


namespace total_profit_calculation_l118_118217

theorem total_profit_calculation (A B C : ℕ) (C_share total_profit : ℕ) 
  (hA : A = 27000) 
  (hB : B = 72000) 
  (hC : C = 81000) 
  (hC_share : C_share = 36000) 
  (h_ratio : C_share * 20 = total_profit * 9) :
  total_profit = 80000 := by
  sorry

end total_profit_calculation_l118_118217


namespace tire_circumference_l118_118937

variable (rpm : ℕ) (car_speed_kmh : ℕ) (circumference : ℝ)

-- Define the conditions
def conditions : Prop :=
  rpm = 400 ∧ car_speed_kmh = 24

-- Define the statement to prove
theorem tire_circumference (h : conditions rpm car_speed_kmh) : circumference = 1 :=
sorry

end tire_circumference_l118_118937


namespace select_eight_genuine_dinars_l118_118889

theorem select_eight_genuine_dinars (coins : Fin 11 → ℝ) :
  (∃ (fake_coin : Option (Fin 11)), 
    ((∀ i j : Fin 11, i ≠ j → coins i = coins j) ∨
    (∀ (genuine_coins impostor_coins : Finset (Fin 11)), 
      genuine_coins ∪ impostor_coins = Finset.univ →
      impostor_coins.card = 1 →
      (∃ difference : ℝ, ∀ i ∈ genuine_coins, coins i = difference) ∧
      (∃ i ∈ impostor_coins, coins i ≠ difference)))) →
  (∃ (selected_coins : Finset (Fin 11)), selected_coins.card = 8 ∧
   (∀ i j : Fin 11, i ∈ selected_coins → j ∈ selected_coins → coins i = coins j)) :=
sorry

end select_eight_genuine_dinars_l118_118889


namespace find_k_if_lines_parallel_l118_118816

theorem find_k_if_lines_parallel (k : ℝ) : (∀ x y, y = 5 * x + 3 → y = (3 * k) * x + 7) → k = 5 / 3 :=
by {
  sorry
}

end find_k_if_lines_parallel_l118_118816


namespace correct_system_of_equations_l118_118776

theorem correct_system_of_equations (x y : ℝ) 
  (h1 : 3 * x = 5 * y - 6)
  (h2 : y = 2 * x - 10) : 
  (3 * x = 5 * y - 6) ∧ (y = 2 * x - 10) :=
by
  sorry

end correct_system_of_equations_l118_118776


namespace find_g_of_2_l118_118766

-- Define the assumptions
variables (g : ℝ → ℝ)
axiom condition : ∀ x : ℝ, x ≠ 0 → 5 * g (1 / x) + (3 * g x) / x = Real.sqrt x

-- State the theorem to prove
theorem find_g_of_2 : g 2 = -(Real.sqrt 2) / 16 :=
by
  sorry

end find_g_of_2_l118_118766


namespace rectangular_solid_surface_area_l118_118376

open Nat

theorem rectangular_solid_surface_area (a b c : ℕ) 
  (h_prime_a : Prime a)
  (h_prime_b : Prime b) 
  (h_prime_c : Prime c)
  (h_volume : a * b * c = 231) :
  2 * (a * b + b * c + c * a) = 262 :=
by
  sorry

end rectangular_solid_surface_area_l118_118376


namespace domain_of_f_l118_118770

open Real

noncomputable def f (x : ℝ) : ℝ := log ((2 - x) / (2 + x))

theorem domain_of_f : ∀ x : ℝ, (2 - x) / (2 + x) > 0 ∧ 2 + x ≠ 0 ↔ -2 < x ∧ x < 2 :=
by
  intro x
  sorry

end domain_of_f_l118_118770


namespace find_n_l118_118241

theorem find_n (n : ℝ) (h : (2 : ℂ) / (1 - I) = (1 : ℂ) + n * I) : n = 1 := by
  sorry

end find_n_l118_118241


namespace blue_markers_count_l118_118368

-- Definitions based on given conditions
def total_markers : ℕ := 3343
def red_markers : ℕ := 2315

-- Main statement to prove
theorem blue_markers_count : total_markers - red_markers = 1028 := by
  sorry

end blue_markers_count_l118_118368


namespace naturals_less_than_10_l118_118991

theorem naturals_less_than_10 :
  {n : ℕ | n < 10} = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9} :=
by 
  sorry

end naturals_less_than_10_l118_118991


namespace correct_number_of_paths_l118_118542

-- Define the number of paths for each segment.
def paths_A_to_B : ℕ := 2
def paths_B_to_D : ℕ := 2
def paths_D_to_C : ℕ := 2
def direct_path_A_to_C : ℕ := 1

-- Define the function to calculate the total paths from A to C.
def total_paths_A_to_C : ℕ :=
  (paths_A_to_B * paths_B_to_D * paths_D_to_C) + direct_path_A_to_C

-- Prove that the total number of paths from A to C is 9.
theorem correct_number_of_paths : total_paths_A_to_C = 9 := by
  -- This is where the proof would go, but it is not required for this task.
  sorry

end correct_number_of_paths_l118_118542


namespace sum_of_distances_l118_118423

theorem sum_of_distances (A B C : ℝ × ℝ) (hA : A.2^2 = 8 * A.1) (hB : B.2^2 = 8 * B.1) 
(hC : C.2^2 = 8 * C.1) (h_centroid : (A.1 + B.1 + C.1) / 3 = 2) : 
  dist (2, 0) A + dist (2, 0) B + dist (2, 0) C = 12 := 
sorry

end sum_of_distances_l118_118423


namespace fraction_proof_l118_118338

theorem fraction_proof (w x y z : ℚ) 
  (h1 : w / x = 1 / 3)
  (h2 : w / y = 3 / 4)
  (h3 : x / z = 2 / 5) : 
  (x + y) / (y + z) = 26 / 53 := 
by
  sorry

end fraction_proof_l118_118338


namespace fractional_part_inequality_l118_118126

noncomputable def frac (z : ℝ) : ℝ := z - ⌊z⌋

theorem fractional_part_inequality (x y : ℝ) : frac (x + y) ≤ frac x + frac y := 
sorry

end fractional_part_inequality_l118_118126


namespace inverse_proposition_l118_118628

theorem inverse_proposition (q_1 q_2 : ℚ) :
  (q_1 ^ 2 = q_2 ^ 2 → q_1 = q_2) ↔ (q_1 = q_2 → q_1 ^ 2 = q_2 ^ 2) :=
sorry

end inverse_proposition_l118_118628


namespace local_minimum_bounded_area_l118_118979

noncomputable def f (x : ℝ) : ℝ := x * (1 - x^2) * Real.exp (x^2)

theorem local_minimum : f (-1 / Real.sqrt 2) = -Real.sqrt (Real.exp 1) / (2 * Real.sqrt 2) :=
sorry

theorem bounded_area : (∫ x in -1..1, f x) = Real.exp 1 - 2 :=
sorry

end local_minimum_bounded_area_l118_118979


namespace least_positive_integer_solution_l118_118331

theorem least_positive_integer_solution :
  ∃ b : ℕ, b ≡ 2 [MOD 3] ∧ b ≡ 3 [MOD 4] ∧ b ≡ 4 [MOD 5] ∧ b ≡ 8 [MOD 9] ∧ b = 179 :=
by
  sorry

end least_positive_integer_solution_l118_118331


namespace count_valid_ways_l118_118209

theorem count_valid_ways (n : ℕ) (h1 : n = 6) : 
  ∀ (library : ℕ), (1 ≤ library) → (library ≤ 5) → ∃ (checked_out : ℕ), 
  (checked_out = n - library) := 
sorry

end count_valid_ways_l118_118209


namespace intersection_points_of_lines_l118_118642

theorem intersection_points_of_lines :
  (∃ (x y : ℚ), 2 * y - 3 * x = 4 ∧ x + 3 * y = 3 ∧ x = 10 / 11 ∧ y = 13 / 11) ∧
  (∃ (x y : ℚ), 2 * y - 3 * x = 4 ∧ 5 * x - 3 * y = 6 ∧ x = 24 ∧ y = 38) :=
by
  sorry

end intersection_points_of_lines_l118_118642


namespace cloud9_total_money_l118_118521

-- Definitions
def I : ℕ := 12000
def G : ℕ := 16000
def R : ℕ := 1600

-- Total money taken after cancellations
def T : ℕ := I + G - R

-- Theorem stating that T is 26400
theorem cloud9_total_money : T = 26400 := by
  unfold T I G R
  sorry

end cloud9_total_money_l118_118521


namespace determine_digits_l118_118631

def product_consecutive_eq_120_times_ABABAB (n A B : ℕ) : Prop :=
  (n - 2) * (n - 1) * n * (n + 1) * (n + 2) = 120 * (A * 101010101 + B * 10101010 + A * 1010101 + B * 101010 + A * 10101 + B * 1010 + A * 101 + B * 10 + A)

theorem determine_digits (A B : ℕ) (h : ∃ n, product_consecutive_eq_120_times_ABABAB n A B):
  A = 5 ∧ B = 7 :=
sorry

end determine_digits_l118_118631


namespace find_k_if_lines_parallel_l118_118814

theorem find_k_if_lines_parallel (k : ℝ) : (∀ x y, y = 5 * x + 3 → y = (3 * k) * x + 7) → k = 5 / 3 :=
by {
  sorry
}

end find_k_if_lines_parallel_l118_118814


namespace number_of_apples_ratio_of_mixed_fruits_total_weight_of_oranges_l118_118637

theorem number_of_apples (total_fruit : ℕ) (oranges_fraction : ℚ) (peaches_fraction : ℚ) (apples_mult : ℕ) (total_fruit_value : total_fruit = 56) :
    oranges_fraction = 1/4 → 
    peaches_fraction = 1/2 → 
    apples_mult = 5 → 
    (apples_mult * peaches_fraction * oranges_fraction * total_fruit) = 35 :=
by
  intros h1 h2 h3
  sorry

theorem ratio_of_mixed_fruits (total_fruit : ℕ) (oranges_fraction : ℚ) (peaches_fraction : ℚ) (mixed_fruits_mult : ℕ) (total_fruit_value : total_fruit = 56) :
    oranges_fraction = 1/4 → 
    peaches_fraction = 1/2 → 
    mixed_fruits_mult = 2 → 
    (mixed_fruits_mult * peaches_fraction * oranges_fraction * total_fruit) / total_fruit = 1/4 :=
by
  intros h1 h2 h3
  sorry

theorem total_weight_of_oranges (total_fruit : ℕ) (oranges_fraction : ℚ) (orange_weight : ℕ) (total_fruit_value : total_fruit = 56) :
    oranges_fraction = 1/4 → 
    orange_weight = 200 → 
    (orange_weight * oranges_fraction * total_fruit) = 2800 :=
by
  intros h1 h2
  sorry

end number_of_apples_ratio_of_mixed_fruits_total_weight_of_oranges_l118_118637


namespace original_price_l118_118978

theorem original_price (p q: ℝ) (h₁ : p ≠ 0) (h₂ : q ≠ 0) : 
  let x := 20000 / (10000^2 - (p^2 + q^2) * 10000 + p^2 * q^2)
  (x : ℝ) * (1 - p^2 / 10000) * (1 - q^2 / 10000) = 2 :=
by
  sorry

end original_price_l118_118978


namespace downward_parabola_with_symmetry_l118_118160

-- Define the general form of the problem conditions in Lean
theorem downward_parabola_with_symmetry (k : ℝ) :
  ∃ a : ℝ, a < 0 ∧ ∃ h : ℝ, h = 3 ∧ ∃ k : ℝ, k = k ∧ ∃ (y x : ℝ), y = a * (x - h)^2 + k :=
sorry

end downward_parabola_with_symmetry_l118_118160


namespace final_water_level_l118_118329

-- Define the conditions
def h_initial (h: ℝ := 0.4): ℝ := 0.4  -- Initial height in meters, 0.4m = 40 cm
def rho_water : ℝ := 1000 -- Density of water in kg/m³
def rho_oil : ℝ := 700 -- Density of oil in kg/m³
def g : ℝ := 9.81 -- Acceleration due to gravity in m/s² (value is standard, provided here for completeness)

-- Statement of the problem in Lean 4
theorem final_water_level (h_initial : ℝ) (rho_water : ℝ) (rho_oil : ℝ) (g : ℝ):
  ∃ h_final : ℝ, 
  ρ_water * g * h_final = ρ_oil * g * (h_initial - h_final) ∧
  h_final = 0.34 :=
begin
  sorry
end

end final_water_level_l118_118329


namespace george_room_painting_l118_118577

-- Define the number of ways to choose 2 colors out of 9 without considering the restriction
def num_ways_total : ℕ := Nat.choose 9 2

-- Define the restriction that red and pink should not be combined
def num_restricted_ways : ℕ := 1

-- Define the final number of permissible combinations
def num_permissible_combinations : ℕ := num_ways_total - num_restricted_ways

theorem george_room_painting :
  num_permissible_combinations = 35 :=
by
  sorry

end george_room_painting_l118_118577


namespace oranges_for_profit_l118_118792

theorem oranges_for_profit (cost_buy: ℚ) (number_buy: ℚ) (cost_sell: ℚ) (number_sell: ℚ)
  (desired_profit: ℚ) (h₁: cost_buy / number_buy = 3.75) (h₂: cost_sell / number_sell = 4.5)
  (h₃: desired_profit = 120) :
  ∃ (oranges_to_sell: ℚ), oranges_to_sell = 160 ∧ (desired_profit / ((cost_sell / number_sell) - (cost_buy / number_buy))) = oranges_to_sell :=
by
  sorry

end oranges_for_profit_l118_118792


namespace tan_beta_l118_118996

noncomputable def tan_eq_2 (α β : ℝ) (h1 : Real.tan (α + β) = 3) (h2 : Real.tan (α + π / 4) = 2) : Real :=
2

theorem tan_beta (α β : ℝ) (h1 : Real.tan (α + β) = 3) (h2 : Real.tan (α + π / 4) = 2) :
  Real.tan β = tan_eq_2 α β h1 h2 := by
  sorry

end tan_beta_l118_118996


namespace distance_3D_l118_118465

theorem distance_3D : 
  let A := (2, -2, 0)
  let B := (8, 8, 3)
  let d := Real.sqrt ((8 - 2) ^ 2 + (8 - (-2)) ^ 2 + (3 - 0) ^ 2)
  d = Real.sqrt 145 :=
by
  let A := (2, -2, 0)
  let B := (8, 8, 3)
  let d := Real.sqrt ((8 - 2) ^ 2 + (8 - (-2)) ^ 2 + (3 - 0) ^ 2)
  dsimp [A, B, d]
  sorry

end distance_3D_l118_118465


namespace factorial_div_42_40_l118_118984

theorem factorial_div_42_40 : (42! / 40! = 1722) := 
by 
  sorry

end factorial_div_42_40_l118_118984


namespace original_cost_of_car_l118_118887

-- Conditions
variables (C : ℝ)
variables (spent_on_repairs : ℝ := 8000)
variables (selling_price : ℝ := 68400)
variables (profit_percent : ℝ := 54.054054054054056)

-- Statement to be proved
theorem original_cost_of_car :
  C + spent_on_repairs = selling_price - (profit_percent / 100) * C :=
sorry

end original_cost_of_car_l118_118887


namespace largest_inscribed_rectangle_l118_118038

theorem largest_inscribed_rectangle {a b m : ℝ} (h : m ≥ b) :
  ∃ (base height area : ℝ),
    base = a * (b + m) / m ∧ 
    height = (b + m) / 2 ∧ 
    area = a * (b + m)^2 / (2 * m) :=
sorry

end largest_inscribed_rectangle_l118_118038


namespace area_S_inequality_l118_118806

noncomputable def F (t : ℝ) : ℝ := 2 * (t - ⌊t⌋)

def S (t : ℝ) : Set (ℝ × ℝ) := { p : ℝ × ℝ | (p.1 - F t) * (p.1 - F t) + p.2 * p.2 ≤ (F t) * (F t) }

theorem area_S_inequality (t : ℝ) : 0 ≤ π * (F t) ^ 2 ∧ π * (F t) ^ 2 ≤ 4 * π := 
by sorry

end area_S_inequality_l118_118806


namespace values_of_x_that_satisfy_gg_x_eq_g_x_l118_118591

noncomputable def g (x : ℝ) : ℝ := x^2 - 5 * x

theorem values_of_x_that_satisfy_gg_x_eq_g_x :
  {x : ℝ | g (g x) = g x} = {0, 5, -2, 3} :=
by
  sorry

end values_of_x_that_satisfy_gg_x_eq_g_x_l118_118591


namespace display_glasses_count_l118_118683

noncomputable def tall_cupboards := 2
noncomputable def wide_cupboards := 2
noncomputable def narrow_cupboards := 2
noncomputable def shelves_per_narrow_cupboard := 3
noncomputable def glasses_tall_cupboard := 30
noncomputable def glasses_wide_cupboard := 2 * glasses_tall_cupboard
noncomputable def glasses_narrow_cupboard := 45
noncomputable def broken_shelf_glasses := glasses_narrow_cupboard / shelves_per_narrow_cupboard

theorem display_glasses_count :
  (tall_cupboards * glasses_tall_cupboard) +
  (wide_cupboards * glasses_wide_cupboard) +
  (1 * (broken_shelf_glasses * (shelves_per_narrow_cupboard - 1)) + glasses_narrow_cupboard) =
  255 :=
by sorry

end display_glasses_count_l118_118683


namespace find_x2_div_c2_add_y2_div_a2_add_z2_div_b2_l118_118564

variable (a b c x y z : ℝ)

theorem find_x2_div_c2_add_y2_div_a2_add_z2_div_b2 
  (h1 : a * (x / c) + b * (y / a) + c * (z / b) = 5) 
  (h2 : c / x + a / y + b / z = 0) : 
  x^2 / c^2 + y^2 / a^2 + z^2 / b^2 = 25 := 
sorry

end find_x2_div_c2_add_y2_div_a2_add_z2_div_b2_l118_118564


namespace min_value_of_fraction_l118_118287

theorem min_value_of_fraction (a b : ℝ) (h_pos_a : 0 < a) (h_pos_b : 0 < b) (h_sum : a + b = 2) :
  (1 / (a + 1) + 4 / (b + 1)) = 9 / 4 := 
sorry

end min_value_of_fraction_l118_118287


namespace seq_a2020_l118_118094

def seq (a : ℕ → ℕ) : Prop :=
(∀ n : ℕ, (a n + a (n+1) ≠ a (n+2) + a (n+3))) ∧
(∀ n : ℕ, (a n + a (n+1) + a (n+2) ≠ a (n+3) + a (n+4) + a (n+5))) ∧
(a 1 = 0)

theorem seq_a2020 (a : ℕ → ℕ) (h : seq a) : a 2020 = 1 :=
sorry

end seq_a2020_l118_118094


namespace total_cost_of_toys_l118_118735

def cost_of_toy_cars : ℝ := 14.88
def cost_of_toy_trucks : ℝ := 5.86

theorem total_cost_of_toys :
  cost_of_toy_cars + cost_of_toy_trucks = 20.74 :=
by
  sorry

end total_cost_of_toys_l118_118735


namespace leadership_selection_ways_l118_118350

theorem leadership_selection_ways (M : ℕ) (chiefs : ℕ) (supporting_chiefs : ℕ) (officers_per_supporting_chief : ℕ) 
  (M_eq : M = 15) (chiefs_eq : chiefs = 1) (supporting_chiefs_eq : supporting_chiefs = 2) 
  (officers_eq : officers_per_supporting_chief = 3) : 
  (M * (M - 1) * (M - 2) * (Nat.choose (M - 3) officers_per_supporting_chief) * (Nat.choose (M - 6) officers_per_supporting_chief)) = 3243240 := by
  simp [M_eq, chiefs_eq, supporting_chiefs_eq, officers_eq]
  norm_num
  sorry

end leadership_selection_ways_l118_118350


namespace unique_solution_for_divisibility_l118_118824

theorem unique_solution_for_divisibility (a b : ℕ) (ha : a > 0) (hb : b > 0) :
  (a^2 + b^2) ∣ (a^3 + 1) ∧ (a^2 + b^2) ∣ (b^3 + 1) → (a = 1 ∧ b = 1) :=
by
  intro h
  sorry

end unique_solution_for_divisibility_l118_118824


namespace value_of_a_l118_118111

variable (a : ℝ)

noncomputable def f (x : ℝ) := x^2 + 8
noncomputable def g (x : ℝ) := x^2 - 4

theorem value_of_a
  (h0 : a > 0)
  (h1 : f (g a) = 8) : a = 2 :=
by
  -- conditions are used as assumptions
  let f := f
  let g := g
  sorry

end value_of_a_l118_118111


namespace smallest_bisecting_segment_l118_118475

-- Define a structure for a triangle in a plane
structure Triangle (α β γ : Type u) :=
(vertex1 : α) 
(vertex2 : β) 
(vertex3 : γ) 
(area : ℝ)

-- Define a predicate for an excellent line
def is_excellent_line {α β γ : Type u} (T : Triangle α β γ) (A : α) (line : ℝ → ℝ → ℝ) : Prop :=
-- define properties that make a line excellent here, e.g., dividing area in half
sorry

-- Define a function to get the length of a line segment within the triangle
def length_within_triangle {α β γ : Type u} (T : Triangle α β γ) (line : ℝ → ℝ → ℝ) : ℝ :=
-- compute the length of the segment within the triangle
sorry

-- Define predicates for triangles with specific properties like medians
def is_median {α β γ : Type u} (T : Triangle α β γ) (line : ℝ → ℝ → ℝ) : Prop :=
-- define properties that make a line a median
sorry

theorem smallest_bisecting_segment {α β γ : Type u} (T : Triangle α β γ) (A : α) (median : ℝ → ℝ → ℝ) : 
  (∀ line, is_excellent_line T A line → length_within_triangle T line ≥ length_within_triangle T median) →
  median = line  := 
-- show that the median from the vertex opposite the smallest angle has the smallest segment
sorry

end smallest_bisecting_segment_l118_118475


namespace find_digits_for_divisibility_l118_118853

theorem find_digits_for_divisibility (d1 d2 : ℕ) (h1 : d1 < 10) (h2 : d2 < 10) :
  (32 * 10^7 + d1 * 10^6 + 35717 * 10 + d2) % 72 = 0 →
  d1 = 2 ∧ d2 = 6 :=
by
  sorry

end find_digits_for_divisibility_l118_118853


namespace quadratic_has_distinct_real_roots_l118_118419

theorem quadratic_has_distinct_real_roots {k : ℝ} (hk : k < 0) : 
  ∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ (x₁^2 - x₁ + k = 0) ∧ (x₂^2 - x₂ + k = 0) :=
by
  -- Proof goes here.
  sorry

end quadratic_has_distinct_real_roots_l118_118419


namespace number_of_situations_l118_118085

def total_athletes : ℕ := 6
def taken_own_coats : ℕ := 2
def taken_wrong_coats : ℕ := 4

-- Combination of choosing 2 athletes out of 6
def combinations : ℕ := Nat.choose total_athletes taken_own_coats

-- Number of derangements for 4 athletes (permutations with no fixed points)
def derangements (n : ℕ) : ℕ := 
  if n = 0 then 1
  else if n = 1 then 0
  else (n - 1) * (derangements (n - 1) + derangements (n - 2))

-- Total number of situations where 4 athletes took someone else's coats
theorem number_of_situations : combinations * derangements taken_wrong_coats = 135 := by
  sorry

end number_of_situations_l118_118085


namespace natural_number_between_squares_l118_118073

open Nat

theorem natural_number_between_squares (n m k l : ℕ)
  (h1 : n > m^2)
  (h2 : n < (m+1)^2)
  (h3 : n - k = m^2)
  (h4 : n + l = (m+1)^2) : ∃ x : ℕ, n - k * l = x^2 := by
  sorry

end natural_number_between_squares_l118_118073


namespace ellipse_foci_on_x_axis_major_axis_twice_minor_axis_l118_118175

theorem ellipse_foci_on_x_axis_major_axis_twice_minor_axis (m : ℝ) :
  (∀ x y : ℝ, x^2 + m * y^2 = 1) → (∃ a b : ℝ, a = 1 ∧ b = Real.sqrt (1 / m) ∧ a = 2 * b) → m = 4 :=
by
  sorry

end ellipse_foci_on_x_axis_major_axis_twice_minor_axis_l118_118175


namespace clock_malfunction_fraction_correct_l118_118669

theorem clock_malfunction_fraction_correct : 
  let hours_total := 24
  let hours_incorrect := 6
  let minutes_total := 60
  let minutes_incorrect := 6
  let fraction_correct_hours := (hours_total - hours_incorrect) / hours_total
  let fraction_correct_minutes := (minutes_total - minutes_incorrect) / minutes_total
  (fraction_correct_hours * fraction_correct_minutes) = 27 / 40
:= 
by
  sorry

end clock_malfunction_fraction_correct_l118_118669


namespace proof_of_min_value_l118_118150

def constraints_on_powers (a b c d : ℝ) : Prop :=
  a^4 + b^4 + c^4 + d^4 = 16

noncomputable def minimum_third_power_sum (a b c d : ℝ) : ℝ :=
  a^3 + b^3 + c^3 + d^3

theorem proof_of_min_value : 
  ∃ a b c d : ℝ, constraints_on_powers a b c d → ∃ min_val : ℝ, min_val = minimum_third_power_sum a b c d :=
sorry -- Further method to rigorously find the minimum value.

end proof_of_min_value_l118_118150


namespace four_digit_number_divisible_by_11_l118_118258

theorem four_digit_number_divisible_by_11 :
  ∃ (a b c d : ℕ), 
    a > 0 ∧ a < 10 ∧ b < 10 ∧ c < 10 ∧ d < 10 ∧ 
    a + b + c + d = 10 ∧ 
    (a + c) % 11 = (b + d) % 11 ∧
    (10 - a != 0 ∨ 10 - c != 0 ∨ 10 - b != 0 ∨ 10 - d != 0) := sorry

end four_digit_number_divisible_by_11_l118_118258


namespace sum_of_ages_l118_118751

variable (M E : ℝ)
variable (h1 : M = E + 9)
variable (h2 : M + 5 = 3 * (E - 3))

theorem sum_of_ages : M + E = 32 :=
by
  sorry

end sum_of_ages_l118_118751


namespace num_non_divisible_by_3_divisors_l118_118409

theorem num_non_divisible_by_3_divisors (a b c : ℕ) (h1: 0 ≤ a ∧ a ≤ 2) (h2: 0 ≤ b ∧ b ≤ 2) (h3: 0 ≤ c ∧ c ≤ 1) :
  (3 * 2 = 6) :=
by sorry

end num_non_divisible_by_3_divisors_l118_118409


namespace four_digit_sum_ten_divisible_by_eleven_l118_118261

theorem four_digit_sum_ten_divisible_by_eleven : 
  {n | (10^3 ≤ n ∧ n < 10^4) ∧ (∑ i in (Int.toString n).toList, i.toNat) = 10 ∧ (let digits := (Int.toString n).toList.map (λ c, c.toNat) in ((digits.nth 0).getD 0 + (digits.nth 2).getD 0) - ((digits.nth 1).getD 0 + (digits.nth 3).getD 0)) % 11 == 0}.card = 30 := sorry

end four_digit_sum_ten_divisible_by_eleven_l118_118261


namespace dance_lesson_cost_l118_118327

-- Define the conditions
variable (total_lessons : Nat) (free_lessons : Nat) (paid_lessons_cost : Nat)

-- State the problem with the given conditions
theorem dance_lesson_cost
  (h1 : total_lessons = 10)
  (h2 : free_lessons = 2)
  (h3 : paid_lessons_cost = 80) :
  let number_of_paid_lessons := total_lessons - free_lessons
  number_of_paid_lessons ≠ 0 -> 
  (paid_lessons_cost / number_of_paid_lessons) = 10 := by
  sorry

end dance_lesson_cost_l118_118327


namespace find_constants_l118_118699

-- Definitions based on the given problem
def inequality_in_x (a : ℝ) (x : ℝ) : Prop :=
  a * x^2 - 3 * x + 2 > 0

def roots_eq (a : ℝ) (r1 r2 : ℝ) : Prop :=
  a * r1^2 - 3 * r1 + 2 = 0 ∧ a * r2^2 - 3 * r2 + 2 = 0

def solution_set (a b : ℝ) (x : ℝ) : Prop :=
  x < 1 ∨ x > b

-- Problem statement: given conditions find a and b
theorem find_constants (a b : ℝ) (h1 : 1 < b) (h2 : 0 < a) :
  roots_eq a 1 b ∧ solution_set a b 1 ∧ solution_set a b b :=
sorry

end find_constants_l118_118699


namespace permutation_problem_l118_118199

noncomputable def permutation (n r : ℕ) : ℕ := (n.factorial) / ( (n - r).factorial)

theorem permutation_problem : 5 * permutation 5 3 + 4 * permutation 4 2 = 348 := by
  sorry

end permutation_problem_l118_118199


namespace frood_points_l118_118276

theorem frood_points (n : ℕ) (h : n > 29) : (n * (n + 1) / 2) > 15 * n := by
  sorry

end frood_points_l118_118276


namespace comic_books_left_l118_118308

theorem comic_books_left (total : ℕ) (sold : ℕ) (left : ℕ) (h1 : total = 90) (h2 : sold = 65) :
  left = total - sold → left = 25 := by
  sorry

end comic_books_left_l118_118308


namespace quarters_count_l118_118753

theorem quarters_count (total_money : ℝ) (value_of_quarter : ℝ) (h1 : total_money = 3) (h2 : value_of_quarter = 0.25) : total_money / value_of_quarter = 12 :=
by sorry

end quarters_count_l118_118753


namespace unique_n_value_l118_118285

theorem unique_n_value (n : ℕ) (d : ℕ → ℕ) (h1 : 1 = d 1) (h2 : ∀ i, d i ≤ n) (h3 : ∀ i j, i < j → d i < d j) 
                       (h4 : d (n - 1) = n) (h5 : ∃ k, k ≥ 4 ∧ ∀ i ≤ k, d i ∣ n)
                       (h6 : ∃ d1 d2 d3 d4, d 1 = d1 ∧ d 2 = d2 ∧ d 3 = d3 ∧ d 4 = d4 ∧ n = d1^2 + d2^2 + d3^2 + d4^2) : 
                       n = 130 := sorry

end unique_n_value_l118_118285


namespace large_pizza_cost_l118_118563

theorem large_pizza_cost
  (small_side : ℕ) (small_cost : ℝ) (large_side : ℕ) (friend_money : ℝ) (extra_square_inches : ℝ)
  (A_small : small_side * small_side = 196)
  (A_large : large_side * large_side = 441)
  (small_cost_per_sq_in : 196 / small_cost = 19.6)
  (individual_area : (30 / small_cost) * 196 = 588)
  (total_individual_area : 2 * 588 = 1176)
  (pool_area_eq : (60 / (441 / x)) = 1225)
  : (x = 21.6) := 
by
  sorry

end large_pizza_cost_l118_118563


namespace planes_parallel_from_plane_l118_118061

-- Define the relationship functions
def parallel (P Q : Plane) : Prop := sorry -- Define parallelism predicate
def perpendicular (l : Line) (P : Plane) : Prop := sorry -- Define perpendicularity predicate

-- Declare the planes α, β, and γ
variable (α β γ : Plane)

-- Main theorem statement
theorem planes_parallel_from_plane (h1 : parallel γ α) (h2 : parallel γ β) : parallel α β := 
sorry

end planes_parallel_from_plane_l118_118061


namespace nickel_ate_2_chocolates_l118_118440

def nickels_chocolates (r n : Nat) : Prop :=
r = n + 7

theorem nickel_ate_2_chocolates (r : Nat) (h : r = 9) (h1 : nickels_chocolates r 2) : 2 = 2 :=
by
  sorry

end nickel_ate_2_chocolates_l118_118440


namespace eggs_total_l118_118606

-- Definitions of the conditions in Lean
def num_people : ℕ := 3
def omelets_per_person : ℕ := 3
def eggs_per_omelet : ℕ := 4

-- The claim we need to prove
theorem eggs_total : (num_people * omelets_per_person) * eggs_per_omelet = 36 :=
by
  sorry

end eggs_total_l118_118606


namespace arithmetic_geometric_sequence_l118_118115

theorem arithmetic_geometric_sequence (a b : ℝ) (h1 : 2 * a = 1 + b) (h2 : b^2 = a) (h3 : a ≠ b) :
  7 * a * Real.log (-b) / Real.log a = 7 / 8 :=
by
  sorry

end arithmetic_geometric_sequence_l118_118115


namespace ellipse_foci_y_axis_iff_l118_118624

theorem ellipse_foci_y_axis_iff (m n : ℝ) (h : m > n ∧ n > 0) :
  (m > n ∧ n > 0) ↔ (∀ (x y : ℝ), m * x^2 + n * y^2 = 1 → ∃ a b : ℝ, a^2 - b^2 = 1 ∧ x^2/b^2 + y^2/a^2 = 1 ∧ a > b) :=
sorry

end ellipse_foci_y_axis_iff_l118_118624


namespace solution_set_inequality_k_l118_118412

theorem solution_set_inequality_k (k : ℚ) :
  (∀ x : ℚ, 3 * x - (2 * k - 3) < 4 * x + 3 * k + 6 ↔ x > 1) → k = -4/5 :=
by
  sorry

end solution_set_inequality_k_l118_118412


namespace nuts_division_pattern_l118_118654

noncomputable def smallest_number_of_nuts : ℕ := 15621

theorem nuts_division_pattern :
  ∃ N : ℕ, N = smallest_number_of_nuts ∧ 
  ∀ k : ℕ, 1 ≤ k ∧ k ≤ 5 → 
  (∃ M : ℕ, (N - k) % 4 = 0 ∧ (N - k) / 4 * 5 + 1 = N) := sorry

end nuts_division_pattern_l118_118654


namespace mike_drive_average_rate_l118_118479

open Real

variables (total_distance first_half_distance second_half_distance first_half_speed second_half_speed first_half_time second_half_time total_time avg_rate j : ℝ)

theorem mike_drive_average_rate :
  total_distance = 640 ∧
  first_half_distance = total_distance / 2 ∧
  second_half_distance = total_distance / 2 ∧
  first_half_speed = 80 ∧
  first_half_distance / first_half_speed = first_half_time ∧
  second_half_time = 3 * first_half_time ∧
  second_half_distance / second_half_time = second_half_speed ∧
  total_time = first_half_time + second_half_time ∧
  avg_rate = total_distance / total_time →
  j = 40 :=
by
  intro h
  sorry

end mike_drive_average_rate_l118_118479


namespace solve_inequalities_solve_fruit_purchase_l118_118788

-- Part 1: Inequalities
theorem solve_inequalities {x : ℝ} : 
  (2 * x < 16) ∧ (3 * x > 2 * x + 3) → (3 < x ∧ x < 8) := by
  sorry

-- Part 2: Fruit Purchase
theorem solve_fruit_purchase {x y : ℝ} : 
  (x + y = 7) ∧ (5 * x + 8 * y = 41) → (x = 5 ∧ y = 2) := by
  sorry

end solve_inequalities_solve_fruit_purchase_l118_118788


namespace equilateral_triangle_of_ap_angles_gp_sides_l118_118140

theorem equilateral_triangle_of_ap_angles_gp_sides
  (A B C : ℝ)
  (α β γ : ℝ)
  (hαβγ_sum : α + β + γ = 180)
  (h_ap_angles : 2 * β = α + γ)
  (a b c : ℝ)
  (h_gp_sides : b^2 = a * c) :
  α = β ∧ β = γ ∧ a = b ∧ b = c :=
sorry

end equilateral_triangle_of_ap_angles_gp_sides_l118_118140


namespace total_animals_in_savanna_l118_118305

/-- Define the number of lions, snakes, and giraffes in Safari National Park. --/
def safari_lions : ℕ := 100
def safari_snakes : ℕ := safari_lions / 2
def safari_giraffes : ℕ := safari_snakes - 10

/-- Define the number of lions, snakes, and giraffes in Savanna National Park based on conditions. --/
def savanna_lions : ℕ := 2 * safari_lions
def savanna_snakes : ℕ := 3 * safari_snakes
def savanna_giraffes : ℕ := safari_giraffes + 20

/-- Calculate the total number of animals in Savanna National Park. --/
def total_savanna_animals : ℕ := savanna_lions + savanna_snakes + savanna_giraffes

/-- Proof statement that the total number of animals in Savanna National Park is 410.
My goal is to prove that total_savanna_animals is equal to 410. --/
theorem total_animals_in_savanna : total_savanna_animals = 410 :=
by
  sorry

end total_animals_in_savanna_l118_118305


namespace tangent_line_is_tangent_l118_118709

noncomputable def func1 (x : ℝ) : ℝ := x + 1 + Real.log x
noncomputable def func2 (a x : ℝ) : ℝ := a * x^2 + (a + 2) * x + 1

theorem tangent_line_is_tangent
  (a : ℝ) (h_tangent : ∃ x₀ : ℝ, func2 a x₀ = 2 * x₀ ∧ (deriv (func2 a) x₀ = 2))
  (deriv_eq : deriv func1 1 = 2)
  : a = 4 :=
by
  sorry

end tangent_line_is_tangent_l118_118709


namespace number_of_men_at_picnic_l118_118211

theorem number_of_men_at_picnic (total persons W M A C : ℕ) (h1 : total = 200) 
  (h2 : M = W + 20) (h3 : A = C + 20) (h4 : A = M + W) : M = 65 :=
by
  -- Proof can be filled in here
  sorry

end number_of_men_at_picnic_l118_118211


namespace grade_point_average_one_third_classroom_l118_118176

theorem grade_point_average_one_third_classroom
  (gpa1 : ℝ) -- grade point average of one third of the classroom
  (gpa_rest : ℝ) -- grade point average of the rest of the classroom
  (gpa_whole : ℝ) -- grade point average of the whole classroom
  (h_rest : gpa_rest = 45)
  (h_whole : gpa_whole = 48) :
  gpa1 = 54 :=
by
  sorry

end grade_point_average_one_third_classroom_l118_118176


namespace total_amount_after_refunds_l118_118525

def individual_bookings : ℕ := 12000
def group_bookings : ℕ := 16000
def refunds : ℕ := 1600

theorem total_amount_after_refunds : 
  individual_bookings + group_bookings - refunds = 26400 :=
by 
  -- The proof goes here.
  sorry

end total_amount_after_refunds_l118_118525


namespace earnings_per_weed_is_six_l118_118023

def flower_bed_weeds : ℕ := 11
def vegetable_patch_weeds : ℕ := 14
def grass_weeds : ℕ := 32
def grass_weeds_half : ℕ := grass_weeds / 2
def soda_cost : ℕ := 99
def money_left : ℕ := 147
def total_weeds : ℕ := flower_bed_weeds + vegetable_patch_weeds + grass_weeds_half
def total_money : ℕ := money_left + soda_cost

theorem earnings_per_weed_is_six :
  total_money / total_weeds = 6 :=
by
  sorry

end earnings_per_weed_is_six_l118_118023


namespace same_terminal_side_angle_l118_118139

theorem same_terminal_side_angle (k : ℤ) : 
  0 ≤ (k * 360 - 35) ∧ (k * 360 - 35) < 360 → (k * 360 - 35) = 325 :=
by
  sorry

end same_terminal_side_angle_l118_118139


namespace percentage_not_drop_l118_118014

def P_trip : ℝ := 0.40
def P_drop_given_trip : ℝ := 0.25
def P_not_drop : ℝ := 0.90

theorem percentage_not_drop :
  (1 - P_trip * P_drop_given_trip) = P_not_drop :=
sorry

end percentage_not_drop_l118_118014


namespace digit_B_for_divisibility_l118_118771

theorem digit_B_for_divisibility (B : ℕ) (h : (40000 + 1000 * B + 100 * B + 20 + 6) % 7 = 0) : B = 1 :=
sorry

end digit_B_for_divisibility_l118_118771


namespace calculate_total_area_of_figure_l118_118417

-- Defining the lengths of the segments according to the problem conditions.
def length_1 : ℕ := 8
def length_2 : ℕ := 6
def length_3 : ℕ := 3
def length_4 : ℕ := 5
def length_5 : ℕ := 2
def length_6 : ℕ := 4

-- Using the given lengths to compute the areas of the smaller rectangles
def area_A : ℕ := length_1 * length_2
def area_B : ℕ := length_4 * (10 - 6)
def area_C : ℕ := (6 - 3) * (15 - 10)

-- The total area of the figure is the sum of the areas of the smaller rectangles
def total_area : ℕ := area_A + area_B + area_C

-- The statement to prove
theorem calculate_total_area_of_figure : total_area = 83 := by
  -- Proof goes here
  sorry

end calculate_total_area_of_figure_l118_118417


namespace eggs_processed_per_day_l118_118000

/-- In a certain egg-processing plant, every egg must be inspected, and is either accepted for processing or rejected. For every 388 eggs accepted for processing, 12 eggs are rejected.

If, on a particular day, 37 additional eggs were accepted, but the overall number of eggs inspected remained the same, the ratio of those accepted to those rejected would be 405 to 3.

Prove that the number of eggs processed per day, given these conditions, is 125763.
-/
theorem eggs_processed_per_day : ∃ (E : ℕ), (∃ (R : ℕ), 38 * R = 3 * (E - 37) ∧  E = 32 * R + E / 33 ) ∧ (E = 125763) :=
sorry

end eggs_processed_per_day_l118_118000


namespace negation_of_existential_proposition_l118_118319

theorem negation_of_existential_proposition : 
  (¬ ∃ x : ℝ, x^2 - x + 3 = 0) = (∀ x : ℝ, x^2 - x + 3 ≠ 0) :=
by
  sorry

end negation_of_existential_proposition_l118_118319


namespace probability_sum_of_five_l118_118030

def total_outcomes : ℕ := 36
def favorable_outcomes : ℕ := 4

theorem probability_sum_of_five :
  favorable_outcomes / total_outcomes = 1 / 9 := 
by
  sorry

end probability_sum_of_five_l118_118030


namespace evaluate_expression_l118_118381

def a : ℚ := 7/3

theorem evaluate_expression :
  (4 * a^2 - 11 * a + 7) * (2 * a - 3) = 140 / 27 :=
by
  sorry

end evaluate_expression_l118_118381


namespace matrix_inverse_proof_l118_118537

open Matrix

def matrix_inverse_problem : Prop :=
  let A := ![
    ![7, -2],
    ![-3, 1]
  ]
  let A_inv := ![
    ![1, 2],
    ![3, 7]
  ]
  A.mul A_inv = (1 : ℤ) • (1 : Matrix (Fin 2) (Fin 2))
  
theorem matrix_inverse_proof : matrix_inverse_problem :=
  by
  sorry

end matrix_inverse_proof_l118_118537


namespace sum_of_two_primes_is_multiple_of_six_l118_118430

theorem sum_of_two_primes_is_multiple_of_six
  (p q r : ℕ)
  (hp : Nat.Prime p) (hq : Nat.Prime q) (hr : Nat.Prime r)
  (hp_gt_3 : p > 3) (hq_gt_3 : q > 3) (hr_gt_3 : r > 3)
  (h_sum_prime : Nat.Prime (p + q + r)) : 
  (p + q) % 6 = 0 ∨ (p + r) % 6 = 0 ∨ (q + r) % 6 = 0 :=
sorry

end sum_of_two_primes_is_multiple_of_six_l118_118430


namespace speed_of_stream_l118_118791

-- Definitions based on conditions
def boat_speed_still_water : ℕ := 24
def travel_time : ℕ := 4
def downstream_distance : ℕ := 112

-- Theorem statement
theorem speed_of_stream : 
  ∀ (v : ℕ), downstream_distance = travel_time * (boat_speed_still_water + v) → v = 4 :=
by
  intros v h
  -- Proof omitted
  sorry

end speed_of_stream_l118_118791


namespace remaining_stock_weight_l118_118593

def green_beans_weight : ℕ := 80
def rice_weight : ℕ := green_beans_weight - 30
def sugar_weight : ℕ := green_beans_weight - 20
def flour_weight : ℕ := 2 * sugar_weight
def lentils_weight : ℕ := flour_weight - 10

def rice_remaining_weight : ℕ := rice_weight - rice_weight / 3
def sugar_remaining_weight : ℕ := sugar_weight - sugar_weight / 5
def flour_remaining_weight : ℕ := flour_weight - flour_weight / 4
def lentils_remaining_weight : ℕ := lentils_weight - lentils_weight / 6

def total_remaining_weight : ℕ :=
  rice_remaining_weight + sugar_remaining_weight + flour_remaining_weight + lentils_remaining_weight + green_beans_weight

theorem remaining_stock_weight :
  total_remaining_weight = 343 := by
  sorry

end remaining_stock_weight_l118_118593


namespace proof_problem_exists_R1_R2_l118_118576

def problem (R1 R2 : ℕ) : Prop :=
  let F1_R1 := (4 * R1 + 5) / (R1^2 - 1)
  let F2_R1 := (5 * R1 + 4) / (R1^2 - 1)
  let F1_R2 := (3 * R2 + 2) / (R2^2 - 1)
  let F2_R2 := (2 * R2 + 3) / (R2^2 - 1)
  F1_R1 = F1_R2 ∧ F2_R1 = F2_R2 ∧ R1 + R2 = 14

theorem proof_problem_exists_R1_R2 : ∃ (R1 R2 : ℕ), problem R1 R2 :=
sorry

end proof_problem_exists_R1_R2_l118_118576


namespace graphs_intersect_at_three_points_l118_118170

noncomputable def is_invertible (f : ℝ → ℝ) := ∃ (g : ℝ → ℝ), ∀ x : ℝ, f (g x) = x ∧ g (f x) = x

theorem graphs_intersect_at_three_points (f : ℝ → ℝ) (h_inv : is_invertible f) :
  ∃ (xs : Finset ℝ), (∀ x ∈ xs, f (x^2) = f (x^6)) ∧ xs.card = 3 :=
by 
  sorry

end graphs_intersect_at_three_points_l118_118170


namespace blue_notebook_cost_l118_118752

theorem blue_notebook_cost 
  (total_spent : ℕ)
  (total_notebooks : ℕ)
  (red_notebooks : ℕ)
  (cost_per_red : ℕ)
  (green_notebooks : ℕ)
  (cost_per_green : ℕ)
  (blue_notebooks : ℕ)
  (total_cost_blue : ℕ)
  (cost_per_blue : ℕ)
  (h1 : total_spent = 37)
  (h2 : total_notebooks = 12)
  (h3 : red_notebooks = 3)
  (h4 : cost_per_red = 4)
  (h5 : green_notebooks = 2)
  (h6 : cost_per_green = 2)
  (h7 : total_cost_blue = total_spent - (red_notebooks * cost_per_red + green_notebooks * cost_per_green))
  (h8 : blue_notebooks = total_notebooks - (red_notebooks + green_notebooks))
  (h9 : cost_per_blue = total_cost_blue / blue_notebooks)
  : cost_per_blue = 3 :=
sorry

end blue_notebook_cost_l118_118752


namespace circle_covers_three_points_l118_118482

open Real

theorem circle_covers_three_points 
  (points : Finset (ℝ × ℝ))
  (h_points : points.card = 111)
  (triangle_side : ℝ)
  (h_side : triangle_side = 15) :
  ∃ (circle_center : ℝ × ℝ), ∃ (circle_radius : ℝ), circle_radius = sqrt 3 / 2 ∧ 
  ∃ p1 p2 p3, p1 ∈ points ∧ p2 ∈ points ∧ p3 ∈ points ∧ 
              dist circle_center p1 ≤ circle_radius ∧ 
              dist circle_center p2 ≤ circle_radius ∧ 
              dist circle_center p3 ≤ circle_radius :=
by
  sorry

end circle_covers_three_points_l118_118482


namespace oranges_bought_l118_118739

theorem oranges_bought (total_cost : ℝ) 
  (selling_price_per_orange : ℝ) 
  (profit_per_orange : ℝ) 
  (cost_price_per_orange : ℝ) 
  (h1 : total_cost = 12.50)
  (h2 : selling_price_per_orange = 0.60)
  (h3 : profit_per_orange = 0.10)
  (h4 : cost_price_per_orange = selling_price_per_orange - profit_per_orange) :
  (total_cost / cost_price_per_orange) = 25 := 
by
  sorry

end oranges_bought_l118_118739


namespace wickets_before_last_match_l118_118783

theorem wickets_before_last_match (R W : ℕ) 
  (initial_average : ℝ) (runs_last_match wickets_last_match : ℕ) (average_decrease : ℝ)
  (h_initial_avg : initial_average = 12.4)
  (h_last_match_runs : runs_last_match = 26)
  (h_last_match_wickets : wickets_last_match = 5)
  (h_avg_decrease : average_decrease = 0.4)
  (h_initial_runs_eq : R = initial_average * W)
  (h_new_average : (R + runs_last_match) / (W + wickets_last_match) = initial_average - average_decrease) :
  W = 85 :=
by
  sorry

end wickets_before_last_match_l118_118783


namespace find_geometric_sequence_element_l118_118732

theorem find_geometric_sequence_element (a b c d e : ℕ) (r : ℚ)
  (h1 : 2 ≤ a ∧ a < b ∧ b < c ∧ c < d ∧ d < e ∧ e < 100)
  (h2 : Nat.gcd a e = 1)
  (h3 : r > 1 ∧ b = a * r ∧ c = a * r^2 ∧ d = a * r^3 ∧ e = a * r^4)
  : c = 36 :=
  sorry

end find_geometric_sequence_element_l118_118732


namespace john_amount_share_l118_118197

theorem john_amount_share {total_amount : ℕ} {total_parts john_share : ℕ} (h1 : total_amount = 4200) (h2 : total_parts = 2 + 4 + 6) (h3 : john_share = 2) :
  john_share * (total_amount / total_parts) = 700 :=
by
  sorry

end john_amount_share_l118_118197


namespace volunteer_allocation_scheme_l118_118496

def num_allocation_schemes : ℕ :=
  let num_ways_choose_2_from_5 := Nat.choose 5 2
  let num_ways_arrange_4_groups := Nat.factorial 4
  num_ways_choose_2_from_5 * num_ways_arrange_4_groups

theorem volunteer_allocation_scheme :
  num_allocation_schemes = 240 :=
by
  sorry

end volunteer_allocation_scheme_l118_118496


namespace total_pencils_sold_l118_118225

theorem total_pencils_sold (price_reduced: Bool)
  (day1_students : ℕ) (first4_d1 : ℕ) (next3_d1 : ℕ) (last3_d1 : ℕ)
  (day2_students : ℕ) (first5_d2 : ℕ) (next6_d2 : ℕ) (last4_d2 : ℕ)
  (day3_students : ℕ) (first10_d3 : ℕ) (next10_d3 : ℕ) (last10_d3 : ℕ)
  (day1_total : day1_students = 10 ∧ first4_d1 = 4 ∧ next3_d1 = 3 ∧ last3_d1 = 3 ∧
    (first4_d1 * 5) + (next3_d1 * 7) + (last3_d1 * 3) = 50)
  (day2_total : day2_students = 15 ∧ first5_d2 = 5 ∧ next6_d2 = 6 ∧ last4_d2 = 4 ∧
    (first5_d2 * 4) + (next6_d2 * 9) + (last4_d2 * 6) = 98)
  (day3_total : day3_students = 2 * day2_students ∧ first10_d3 = 10 ∧ next10_d3 = 10 ∧ last10_d3 = 10 ∧
    (first10_d3 * 2) + (next10_d3 * 8) + (last10_d3 * 4) = 140) :
  (50 + 98 + 140 = 288) :=
sorry

end total_pencils_sold_l118_118225


namespace quadratic_max_value_l118_118181

theorem quadratic_max_value :
  ∀ (x : ℝ), -x^2 - 2*x - 3 ≤ -2 :=
begin
  sorry,
end

end quadratic_max_value_l118_118181


namespace cuberoot_inequality_l118_118028

theorem cuberoot_inequality (a b : ℝ) : a < b → (∃ x y : ℝ, x^3 = a ∧ y^3 = b ∧ (x = y ∨ x > y)) := 
sorry

end cuberoot_inequality_l118_118028


namespace Valley_Forge_High_School_winter_carnival_l118_118084

noncomputable def number_of_girls (total_students : ℕ) (total_participants : ℕ) (fraction_girls_participating : ℚ) (fraction_boys_participating : ℚ) : ℕ := sorry

theorem Valley_Forge_High_School_winter_carnival
  (total_students : ℕ)
  (total_participants : ℕ)
  (fraction_girls_participating : ℚ)
  (fraction_boys_participating : ℚ)
  (h_total_students : total_students = 1500)
  (h_total_participants : total_participants = 900)
  (h_fraction_girls : fraction_girls_participating = 3 / 4)
  (h_fraction_boys : fraction_boys_participating = 2 / 3) :
  number_of_girls total_students total_participants fraction_girls_participating fraction_boys_participating = 900 := sorry

end Valley_Forge_High_School_winter_carnival_l118_118084


namespace find_a_l118_118250

theorem find_a (a : ℝ) : (1 : ℝ)^2 + 1 + 2 * a = 0 → a = -1 := 
by 
  sorry

end find_a_l118_118250


namespace largest_number_not_sum_of_two_composites_l118_118102

-- Define composite numbers
def is_composite (n : ℕ) : Prop := n > 1 ∧ ∃ d, d ∣ n ∧ d ≠ 1 ∧ d ≠ n

-- Define the problem predicate
def cannot_be_expressed_as_sum_of_two_composites (n : ℕ) : Prop :=
  ¬ ∃ a b : ℕ, is_composite a ∧ is_composite b ∧ n = a + b

theorem largest_number_not_sum_of_two_composites : 
  ∃ n, cannot_be_expressed_as_sum_of_two_composites n ∧ 
  (∀ m, cannot_be_expressed_as_sum_of_two_composites m → m ≤ n) ∧ 
  n = 11 :=
by {
  sorry
}

end largest_number_not_sum_of_two_composites_l118_118102


namespace sqrt_range_l118_118721

theorem sqrt_range (x : ℝ) : (∃ y : ℝ, y = Real.sqrt (x - 3)) ↔ x ≥ 3 :=
by
  sorry

end sqrt_range_l118_118721


namespace find_a_l118_118249

theorem find_a (a : ℝ) : (1 : ℝ)^2 + 1 + 2 * a = 0 → a = -1 := 
by 
  sorry

end find_a_l118_118249


namespace bert_total_stamp_cost_l118_118086

theorem bert_total_stamp_cost :
    let numA := 150
    let numB := 90
    let numC := 60
    let priceA := 2
    let priceB := 3
    let priceC := 5
    let costA := numA * priceA
    let costB := numB * priceB
    let costC := numC * priceC
    let total_cost := costA + costB + costC
    total_cost = 870 := 
by
    sorry

end bert_total_stamp_cost_l118_118086


namespace leos_time_is_1230_l118_118915

theorem leos_time_is_1230
  (theo_watch_slow: Int)
  (theo_watch_fast_belief: Int)
  (leo_watch_fast: Int)
  (leo_watch_slow_belief: Int)
  (theo_thinks_time: Int):
  theo_watch_slow = 10 ∧
  theo_watch_fast_belief = 5 ∧
  leo_watch_fast = 5 ∧
  leo_watch_slow_belief = 10 ∧
  theo_thinks_time = 720
  → leo_thinks_time = 750 :=
by
  sorry

end leos_time_is_1230_l118_118915


namespace Megan_bought_24_eggs_l118_118595

def eggs_problem : Prop :=
  ∃ (p c b : ℕ),
    b = 3 ∧
    c = 2 * b ∧
    p - c = 9 ∧
    p + c + b = 24

theorem Megan_bought_24_eggs : eggs_problem :=
  sorry

end Megan_bought_24_eggs_l118_118595


namespace find_n_value_l118_118003

theorem find_n_value : 
  ∃ (n : ℕ), ∀ (a b c : ℕ), 
    a + b + c = 200 ∧ 
    (∃ bc ca ab : ℕ, bc = b * c ∧ ca = c * a ∧ ab = a * b ∧ n = bc ∧ n = ca ∧ n = ab) → 
    n = 199 := sorry

end find_n_value_l118_118003


namespace dentist_filling_cost_l118_118929

variable (F : ℝ)
variable (total_bill : ℝ := 5 * F)
variable (cleaning_cost : ℝ := 70)
variable (extraction_cost : ℝ := 290)
variable (two_fillings_cost : ℝ := 2 * F)

theorem dentist_filling_cost :
  total_bill = cleaning_cost + two_fillings_cost + extraction_cost → 
  F = 120 :=
by
  intros h
  sorry

end dentist_filling_cost_l118_118929


namespace rotational_homothety_commute_iff_centers_coincide_l118_118878

-- Define rotational homothety and its properties
structure RotationalHomothety (P : Type*) :=
(center : P)
(apply : P → P)
(is_homothety : ∀ p, apply (apply p) = apply p)

variables {P : Type*} [TopologicalSpace P] (H1 H2 : RotationalHomothety P)

-- Prove the equivalence statement
theorem rotational_homothety_commute_iff_centers_coincide :
  (H1.center = H2.center) ↔ (H1.apply ∘ H2.apply = H2.apply ∘ H1.apply) :=
sorry

end rotational_homothety_commute_iff_centers_coincide_l118_118878


namespace probability_r25_to_r35_l118_118895

theorem probability_r25_to_r35 (n : ℕ) (r : Fin n → ℕ) (h : n = 50) 
  (distinct : ∀ i j : Fin n, i ≠ j → r i ≠ r j) : 1 + 1260 = 1261 :=
by
  sorry

end probability_r25_to_r35_l118_118895


namespace regular_icosahedron_edges_l118_118402

-- Define the concept of a regular icosahedron.
structure RegularIcosahedron :=
  (vertices : ℕ)
  (faces : ℕ)
  (edges : ℕ)

-- Define the properties of a regular icosahedron.
def regular_icosahedron_properties (ico : RegularIcosahedron) : Prop :=
  ico.vertices = 12 ∧ ico.faces = 20 ∧ ico.edges = 30

-- Statement of the proof problem: The number of edges in a regular icosahedron is 30.
theorem regular_icosahedron_edges : ∀ (ico : RegularIcosahedron), regular_icosahedron_properties ico → ico.edges = 30 :=
by
  sorry

end regular_icosahedron_edges_l118_118402


namespace simon_age_is_10_l118_118502

-- Declare the variables
variable (alvin_age : ℕ) (simon_age : ℕ)

-- Define the conditions
def condition1 : Prop := alvin_age = 30
def condition2 : Prop := simon_age = (alvin_age / 2) - 5

-- Formalize the proof problem
theorem simon_age_is_10 (h1 : condition1) (h2 : condition2) : simon_age = 10 := by
  sorry

end simon_age_is_10_l118_118502


namespace greg_books_difference_l118_118432

theorem greg_books_difference (M K G X : ℕ)
  (hM : M = 32)
  (hK : K = M / 4)
  (hG : G = 2 * K + X)
  (htotal : M + K + G = 65) :
  X = 9 :=
by
  sorry

end greg_books_difference_l118_118432


namespace train_speed_l118_118785

def train_length : ℝ := 400  -- Length of the train in meters
def crossing_time : ℝ := 40  -- Time to cross the electric pole in seconds

theorem train_speed : train_length / crossing_time = 10 := by
  sorry  -- Proof to be completed

end train_speed_l118_118785


namespace rob_has_24_cards_l118_118439

theorem rob_has_24_cards 
  (r : ℕ) -- total number of baseball cards Rob has
  (dr : ℕ) -- number of doubles Rob has
  (hj: dr = 1 / 3 * r) -- one third of Rob's cards are doubles
  (jess_doubles : ℕ) -- number of doubles Jess has
  (hj_mult : jess_doubles = 5 * dr) -- Jess has 5 times as many doubles as Rob
  (jess_doubles_40 : jess_doubles = 40) -- Jess has 40 doubles baseball cards
: r = 24 :=
by
  sorry

end rob_has_24_cards_l118_118439


namespace quadratic_conversion_l118_118985

theorem quadratic_conversion (x : ℝ) :
  (2*x - 1)^2 = (x + 1)*(3*x + 4) →
  ∃ a b c : ℝ, a = 1 ∧ b = -11 ∧ c = -3 ∧ a*x^2 + b*x + c = 0 :=
by simp [pow_two, mul_add, add_mul, mul_comm]; sorry

end quadratic_conversion_l118_118985


namespace constant_chromosome_number_l118_118886

theorem constant_chromosome_number (rabbits : Type) 
  (sex_reproduction : rabbits → Prop)
  (maintain_chromosome_number : Prop)
  (meiosis : Prop)
  (fertilization : Prop) : 
  (meiosis ∧ fertilization) ↔ maintain_chromosome_number :=
sorry

end constant_chromosome_number_l118_118886


namespace initially_tagged_fish_l118_118001

theorem initially_tagged_fish (second_catch_total : ℕ) (second_catch_tagged : ℕ)
  (total_fish_pond : ℕ) (approx_ratio : ℚ) 
  (h1 : second_catch_total = 50)
  (h2 : second_catch_tagged = 2)
  (h3 : total_fish_pond = 1750)
  (h4 : approx_ratio = (second_catch_tagged : ℚ) / second_catch_total) :
  ∃ T : ℕ, T = 70 :=
by
  sorry

end initially_tagged_fish_l118_118001


namespace badgers_win_at_least_five_games_l118_118033

-- Define the problem conditions and the required probability calculation
theorem badgers_win_at_least_five_games :
  let p := 0.5 in
  let n := 9 in
  let probability_at_least_five_wins :=
    ∑ k in Finset.range (n + 1), if k >= 5 then (nat.choose n k) * (p^k) * ((1 - p)^(n - k)) else 0
  in
  probability_at_least_five_wins = 1 / 2 :=
by
  sorry

end badgers_win_at_least_five_games_l118_118033


namespace triangle_sine_ratio_l118_118867

-- Define points A and C
def A : ℝ × ℝ := (-1, 0)
def C : ℝ × ℝ := (1, 0)

-- Define the condition of point B being on the ellipse
def isOnEllipse (B : ℝ × ℝ) : Prop :=
  (B.1 ^ 2) / 4 + (B.2 ^ 2) / 3 = 1

-- Define the sin law ratio we need to prove
noncomputable def sin_ratio (sin_A sin_C sin_B : ℝ) : ℝ := 
  (sin_A + sin_C) / sin_B

-- Prove the required sine ratio condition
theorem triangle_sine_ratio (B : ℝ × ℝ) (sin_A sin_C sin_B : ℝ)
  (hB : isOnEllipse B) (hA : sin_A = 0) (hC : sin_C = 0) (hB_nonzero : sin_B ≠ 0) :
  sin_ratio sin_A sin_C sin_B = 2 :=
by
  -- Skipping proof
  sorry

end triangle_sine_ratio_l118_118867


namespace rate_up_the_mountain_l118_118068

noncomputable def mountain_trip_rate (R : ℝ) : ℝ := 1.5 * R

theorem rate_up_the_mountain : 
  ∃ R : ℝ, (2 * 1.5 * R = 18) ∧ (1.5 * R = 9) → R = 6 :=
by
  sorry

end rate_up_the_mountain_l118_118068


namespace percentage_of_total_money_raised_from_donations_l118_118458

-- Define the conditions
def max_donation := 1200
def num_donors_max := 500
def half_donation := max_donation / 2
def num_donors_half := 3 * num_donors_max
def total_money_raised := 3750000

-- Define the amounts collected from each group
def amount_from_max_donors := num_donors_max * max_donation
def amount_from_half_donors := num_donors_half * half_donation
def total_amount_from_donations := amount_from_max_donors + amount_from_half_donors

-- Define the percentage calculation
def percentage_of_total := (total_amount_from_donations / total_money_raised) * 100

-- State the theorem (but not the proof)
theorem percentage_of_total_money_raised_from_donations : 
  percentage_of_total = 40 := by
  sorry

end percentage_of_total_money_raised_from_donations_l118_118458


namespace events_complementary_l118_118638

def event_A (n : ℕ) : Prop := n ≤ 3
def event_B (n : ℕ) : Prop := n ≥ 4
def valid_outcomes : Finset ℕ := {1, 2, 3, 4, 5, 6}

theorem events_complementary :
  (∀ n, n ∈ valid_outcomes → event_A n ∨ event_B n) ∧ (∀ n, n ∈ valid_outcomes → ¬(event_A n ∧ event_B n)) :=
by
  sorry

end events_complementary_l118_118638


namespace rectangle_area_correct_l118_118040

theorem rectangle_area_correct (l r s : ℝ) (b : ℝ := 10) (h1 : l = (1 / 4) * r) (h2 : r = s) (h3 : s^2 = 1225) :
  l * b = 87.5 :=
by
  sorry

end rectangle_area_correct_l118_118040


namespace bridge_length_is_219_l118_118065

noncomputable def length_of_bridge (train_length : ℕ) (train_speed_kmh : ℤ) (time_seconds : ℕ) : ℝ :=
  let train_speed_ms : ℝ := train_speed_kmh * (1000 / 3600)
  let total_distance : ℝ := train_speed_ms * time_seconds
  total_distance - train_length

theorem bridge_length_is_219 :
  length_of_bridge 156 45 30 = 219 :=
by
  sorry

end bridge_length_is_219_l118_118065


namespace tracy_sold_paintings_l118_118051

-- Definitions based on conditions
def total_customers := 20
def customers_2_paintings := 4
def customers_1_painting := 12
def customers_4_paintings := 4

def paintings_per_customer_2 := 2
def paintings_per_customer_1 := 1
def paintings_per_customer_4 := 4

-- Theorem statement
theorem tracy_sold_paintings : 
  (customers_2_paintings * paintings_per_customer_2) + 
  (customers_1_painting * paintings_per_customer_1) + 
  (customers_4_paintings * paintings_per_customer_4) = 36 := 
by
  sorry

end tracy_sold_paintings_l118_118051


namespace tim_stacked_bales_today_l118_118778

theorem tim_stacked_bales_today (initial_bales : ℕ) (current_bales : ℕ) (initial_eq : initial_bales = 54) (current_eq : current_bales = 82) : 
  current_bales - initial_bales = 28 :=
by
  -- conditions
  have h1 : initial_bales = 54 := initial_eq
  have h2 : current_bales = 82 := current_eq
  sorry

end tim_stacked_bales_today_l118_118778


namespace knight_will_be_freed_l118_118794

/-- Define a structure to hold the state of the piles -/
structure PileState where
  pile1_magical : ℕ
  pile1_non_magical : ℕ
  pile2_magical : ℕ
  pile2_non_magical : ℕ
deriving Repr

-- Function to move one coin from pile1 to pile2
def move_coin (state : PileState) : PileState :=
  if state.pile1_magical > 0 then
    { state with
      pile1_magical := state.pile1_magical - 1,
      pile2_magical := state.pile2_magical + 1 }
  else if state.pile1_non_magical > 0 then
    { state with
      pile1_non_magical := state.pile1_non_magical - 1,
      pile2_non_magical := state.pile2_non_magical + 1 }
  else
    state -- If no coins to move, the state remains unchanged

-- The initial state of the piles
def initial_state : PileState :=
  { pile1_magical := 0, pile1_non_magical := 49, pile2_magical := 50, pile2_non_magical := 1 }

-- Check if the knight can be freed (both piles have the same number of magical or non-magical coins)
def knight_free (state : PileState) : Prop :=
  state.pile1_magical = state.pile2_magical ∨ state.pile1_non_magical = state.pile2_non_magical

noncomputable def knight_can_be_freed_by_25th_day : Prop :=
  exists n : ℕ, n ≤ 25 ∧ knight_free (Nat.iterate move_coin n initial_state)

theorem knight_will_be_freed : knight_can_be_freed_by_25th_day :=
  sorry

end knight_will_be_freed_l118_118794


namespace will_pages_needed_l118_118476

theorem will_pages_needed :
  let new_cards_2020 := 8
  let old_cards := 10
  let duplicates := 2
  let cards_per_page := 3
  let unique_old_cards := old_cards - duplicates
  let pages_needed_for_2020 := (new_cards_2020 + cards_per_page - 1) / cards_per_page -- ceil(new_cards_2020 / cards_per_page)
  let pages_needed_for_old := (unique_old_cards + cards_per_page - 1) / cards_per_page -- ceil(unique_old_cards / cards_per_page)
  let pages_needed := pages_needed_for_2020 + pages_needed_for_old
  pages_needed = 6 :=
by
  sorry

end will_pages_needed_l118_118476


namespace target_expression_l118_118132

variable (a b : ℤ)

-- Definitions based on problem conditions
def op1 (x y : ℤ) : ℤ := x + y  -- "!" could be addition
def op2 (x y : ℤ) : ℤ := x - y  -- "?" could be subtraction in one order

-- Using these operations to create expressions
def exp1 (a b : ℤ) := op1 (op2 a b) (op2 b a)

def exp2 (x y : ℤ) := op2 (op2 x 0) (op2 0 y)

-- The final expression we need to check
def final_exp (a b : ℤ) := exp1 (20 * a) (18 * b)

-- Theorem proving the final expression equals target
theorem target_expression : final_exp a b = 20 * a - 18 * b :=
sorry

end target_expression_l118_118132


namespace average_viewing_times_correct_l118_118434

-- Define the viewing times for each family member per week
def Evelyn_week1 : ℕ := 10
def Evelyn_week2 : ℕ := 8
def Evelyn_week3 : ℕ := 6

def Eric_week1 : ℕ := 8
def Eric_week2 : ℕ := 6
def Eric_week3 : ℕ := 5

def Kate_week2_episodes : ℕ := 12
def minutes_per_episode : ℕ := 40
def Kate_week2 : ℕ := (Kate_week2_episodes * minutes_per_episode) / 60
def Kate_week3 : ℕ := 4

def John_week2 : ℕ := (Kate_week2_episodes * minutes_per_episode) / 60
def John_week3 : ℕ := 8

-- Calculate the averages
def average (total : ℚ) (weeks : ℚ) : ℚ := total / weeks

-- Define the total viewing time for each family member
def Evelyn_total : ℕ := Evelyn_week1 + Evelyn_week2 + Evelyn_week3
def Eric_total : ℕ := Eric_week1 + Eric_week2 + Eric_week3
def Kate_total : ℕ := 0 + Kate_week2 + Kate_week3
def John_total : ℕ := 0 + John_week2 + John_week3

-- Define the expected averages
def Evelyn_expected_avg : ℚ := 8
def Eric_expected_avg : ℚ := 19 / 3
def Kate_expected_avg : ℚ := 4
def John_expected_avg : ℚ := 16 / 3

-- The theorem to prove that the calculated averages are correct
theorem average_viewing_times_correct :
  average Evelyn_total 3 = Evelyn_expected_avg ∧
  average Eric_total 3 = Eric_expected_avg ∧
  average Kate_total 3 = Kate_expected_avg ∧
  average John_total 3 = John_expected_avg :=
by sorry

end average_viewing_times_correct_l118_118434


namespace intersection_A_B_range_of_m_l118_118701

-- Step 1: Define sets A, B, and C
def A : Set ℝ := {x | ∃ y, y = Real.log (x - 1)}

def B : Set ℝ := {x | -1 < x ∧ x < 3}

def C (m : ℝ) : Set ℝ := {x | m < x ∧ x < 2 * m - 1}

-- Step 2: Lean statements for the proof

-- (1) Prove A ∩ B = {x | 1 < x < 3}
theorem intersection_A_B : (A ∩ B) = {x | 1 < x ∧ x < 3} :=
by
  sorry

-- (2) Prove the range of m such that C ∪ B = B is (-∞, 2]
theorem range_of_m (m : ℝ) : (C m ∪ B = B) ↔ m ≤ 2 :=
by
  sorry

end intersection_A_B_range_of_m_l118_118701


namespace evaluate_72_squared_minus_48_squared_l118_118820

theorem evaluate_72_squared_minus_48_squared :
  (72:ℤ)^2 - (48:ℤ)^2 = 2880 :=
by
  sorry

end evaluate_72_squared_minus_48_squared_l118_118820


namespace jake_not_drop_coffee_percentage_l118_118013

-- Definitions for the conditions
def trip_probability : ℝ := 0.40
def drop_when_trip_probability : ℝ := 0.25

-- The question and proof statement
theorem jake_not_drop_coffee_percentage :
  100 * (1 - trip_probability * drop_when_trip_probability) = 90 :=
by
  sorry

end jake_not_drop_coffee_percentage_l118_118013


namespace calculation_proof_l118_118680

theorem calculation_proof
    (a : ℝ) (b : ℝ) (c : ℝ)
    (h1 : a = 3.6)
    (h2 : b = 0.25)
    (h3 : c = 0.5) :
    (a * b) / c = 1.8 := 
by
  sorry

end calculation_proof_l118_118680


namespace range_of_m_l118_118863

-- Definitions based on conditions
def inequality_holds (m : ℝ) : Prop :=
  ∀ x : ℝ, 0 < x → (m * Real.exp x / x ≥ 6 - 4 * x)

-- The statement to be proved
theorem range_of_m (m : ℝ) : inequality_holds m → m ≥ 2 * Real.exp (-(1 / 2)) :=
by
  sorry

end range_of_m_l118_118863


namespace number_replacement_l118_118059

theorem number_replacement :
  ∃ x : ℝ, ( (x / (1 / 2) * x) / (x * (1 / 2) / x) = 25 ) ↔ x = 2.5 :=
by 
  sorry

end number_replacement_l118_118059


namespace minimum_number_of_tiles_l118_118966

def tile_width_in_inches : ℕ := 6
def tile_height_in_inches : ℕ := 4
def region_width_in_feet : ℕ := 3
def region_height_in_feet : ℕ := 8

def inches_to_feet (i : ℕ) : ℚ :=
  i / 12

def tile_width_in_feet : ℚ :=
  inches_to_feet tile_width_in_inches

def tile_height_in_feet : ℚ :=
  inches_to_feet tile_height_in_inches

def tile_area_in_square_feet : ℚ :=
  tile_width_in_feet * tile_height_in_feet

def region_area_in_square_feet : ℚ :=
  region_width_in_feet * region_height_in_feet

def number_of_tiles : ℚ :=
  region_area_in_square_feet / tile_area_in_square_feet

theorem minimum_number_of_tiles :
  number_of_tiles = 144 := by
    sorry

end minimum_number_of_tiles_l118_118966


namespace quadratic_solution_l118_118247

theorem quadratic_solution (a : ℝ) (h : (1 : ℝ)^2 + 1 + 2 * a = 0) : a = -1 :=
by {
  sorry
}

end quadratic_solution_l118_118247


namespace prob_at_least_one_first_class_expected_daily_profit_production_increase_decision_l118_118956

section 
open ProbabilityTheory 

-- Given conditions
def prob_first_class : ℚ := 0.5
def prob_second_class : ℚ := 0.4
def prob_third_class : ℚ := 0.1

def profit_first_class : ℚ := 0.8
def profit_second_class : ℚ := 0.6
def profit_third_class : ℚ := -0.3

def daily_output : ℕ := 2

-- Proof statements
theorem prob_at_least_one_first_class : 
  let prob_event := (prob_first_class * prob_first_class) + 
                    2 * (prob_first_class * (1 - prob_first_class))
  in prob_event = 0.75 := 
  by sorry

theorem expected_daily_profit : 
  let exp_profit := -0.6 * (prob_third_class ^ 2) + 0.3 * (2 * prob_second_class * prob_third_class) +
                    0.5 * (2 * prob_first_class * prob_third_class) + 1.2 * (prob_second_class ^ 2) +
                    1.4 * (2 * prob_first_class * prob_second_class) + 1.6 * (prob_first_class ^ 2)
  in exp_profit = 1.22 :=
  by sorry

theorem production_increase_decision: 
  let avg_profit_per_unit := 1.22 / daily_output
  let net_profit (n : ℕ) := avg_profit_per_unit * n - (n - log (n))
  in ∀ n : ℕ, net_profit n ≤ 0 := 
  by sorry

end

end prob_at_least_one_first_class_expected_daily_profit_production_increase_decision_l118_118956


namespace geometric_sequence_S5_eq_11_l118_118018

theorem geometric_sequence_S5_eq_11 
  (a : ℕ → ℤ) 
  (S : ℕ → ℤ) 
  (q : ℤ)
  (h1 : a 1 = 1)
  (h4 : a 4 = -8)
  (h_geom : ∀ n, a (n + 1) = a n * q) 
  (h_S : ∀ n, S n = a 1 * (1 - q ^ n) / (1 - q)) :
  S 5 = 11 := 
by
  -- Proof omitted
  sorry

end geometric_sequence_S5_eq_11_l118_118018


namespace extra_people_got_on_the_train_l118_118046

-- Definitions corresponding to the conditions
def initial_people_on_train : ℕ := 78
def people_got_off : ℕ := 27
def current_people_on_train : ℕ := 63

-- The mathematical equivalent proof problem
theorem extra_people_got_on_the_train :
  (initial_people_on_train - people_got_off + extra_people = current_people_on_train) → (extra_people = 12) :=
by
  sorry

end extra_people_got_on_the_train_l118_118046


namespace total_legs_l118_118532

theorem total_legs (chickens sheep : ℕ) (chicken_legs sheep_legs total_legs : ℕ) (h1 : chickens = 7) (h2 : sheep = 5) 
(h3 : chicken_legs = 2) (h4 : sheep_legs = 4) (h5 : total_legs = chickens * chicken_legs + sheep * sheep_legs) : 
total_legs = 34 := 
by {
  rw [h1, h2, h3, h4],
  exact h5,
  sorry
}

end total_legs_l118_118532


namespace ellipse_equation_correct_coordinates_c_correct_l118_118246

-- Definition of the ellipse Γ with given properties
def ellipse_properties (a b : ℝ) (ecc : ℝ) (c_len : ℝ) :=
  a > b ∧ b > 0 ∧ ecc = (Real.sqrt 2) / 2 ∧ c_len = Real.sqrt 2

-- Correct answer for the equation of the ellipse
def correct_ellipse_equation := ∀ x y : ℝ, (x^2) / 2 + y^2 = 1

-- Proving that given the properties of the ellipse, the equation is as stated
theorem ellipse_equation_correct (a b : ℝ) (h : ellipse_properties a b (Real.sqrt 2 / 2) (Real.sqrt 2)) :
  (x^2) / 2 + y^2 = 1 := 
  sorry

-- Definition of the conditions for points A, B, and C
def triangle_conditions (a b : ℝ) (area : ℝ) :=
  ∀ A B : ℝ × ℝ,
    A.1^2 / a^2 + A.2^2 / b^2 = 1 ∧
    B.1^2 / a^2 + B.2^2 / b^2 = 1 ∧
    area = 3 * Real.sqrt 6 / 4

-- Correct coordinates of point C given the conditions
def correct_coordinates_c (C : ℝ × ℝ) :=
  (C = (1, Real.sqrt 2 / 2) ∨ C = (2, 1))

-- Proving that given the conditions, the coordinates of point C are correct
theorem coordinates_c_correct (a b : ℝ) (h : triangle_conditions a b (3 * Real.sqrt 6 / 4)) (C : ℝ × ℝ) :
  correct_coordinates_c C :=
  sorry

end ellipse_equation_correct_coordinates_c_correct_l118_118246


namespace kaleb_clothing_problem_l118_118198

theorem kaleb_clothing_problem 
  (initial_clothing : ℕ) 
  (one_load : ℕ) 
  (remaining_loads : ℕ) : 
  initial_clothing = 39 → one_load = 19 → remaining_loads = 5 → (initial_clothing - one_load) / remaining_loads = 4 :=
sorry

end kaleb_clothing_problem_l118_118198


namespace percentage_not_drop_l118_118015

def P_trip : ℝ := 0.40
def P_drop_given_trip : ℝ := 0.25
def P_not_drop : ℝ := 0.90

theorem percentage_not_drop :
  (1 - P_trip * P_drop_given_trip) = P_not_drop :=
sorry

end percentage_not_drop_l118_118015


namespace combined_molecular_weight_l118_118512

theorem combined_molecular_weight 
  (atomic_weight_N : ℝ)
  (atomic_weight_O : ℝ)
  (atomic_weight_H : ℝ)
  (atomic_weight_C : ℝ)
  (moles_N2O3 : ℝ)
  (moles_H2O : ℝ)
  (moles_CO2 : ℝ)
  (molecular_weight_N2O3 : ℝ)
  (molecular_weight_H2O : ℝ)
  (molecular_weight_CO2 : ℝ)
  (weight_N2O3 : ℝ)
  (weight_H2O : ℝ)
  (weight_CO2 : ℝ)
  : 
  moles_N2O3 = 4 →
  moles_H2O = 3.5 →
  moles_CO2 = 2 →
  atomic_weight_N = 14.01 →
  atomic_weight_O = 16.00 →
  atomic_weight_H = 1.01 →
  atomic_weight_C = 12.01 →
  molecular_weight_N2O3 = (2 * atomic_weight_N) + (3 * atomic_weight_O) →
  molecular_weight_H2O = (2 * atomic_weight_H) + atomic_weight_O →
  molecular_weight_CO2 = atomic_weight_C + (2 * atomic_weight_O) →
  weight_N2O3 = moles_N2O3 * molecular_weight_N2O3 →
  weight_H2O = moles_H2O * molecular_weight_H2O →
  weight_CO2 = moles_CO2 * molecular_weight_CO2 →
  weight_N2O3 + weight_H2O + weight_CO2 = 455.17 :=
by 
  intros;
  sorry

end combined_molecular_weight_l118_118512


namespace rita_bought_4_jackets_l118_118610

/-
Given:
  - Rita bought 5 short dresses costing $20 each.
  - Rita bought 3 pairs of pants costing $12 each.
  - The jackets cost $30 each.
  - She spent an additional $5 on transportation.
  - Rita had $400 initially.
  - Rita now has $139.

Prove that the number of jackets Rita bought is 4.
-/

theorem rita_bought_4_jackets :
  let dresses_cost := 5 * 20
  let pants_cost := 3 * 12
  let transportation_cost := 5
  let initial_amount := 400
  let remaining_amount := 139
  let jackets_cost_per_unit := 30
  let total_spent := initial_amount - remaining_amount
  let total_clothes_transportation_cost := dresses_cost + pants_cost + transportation_cost
  let jackets_cost := total_spent - total_clothes_transportation_cost
  let number_of_jackets := jackets_cost / jackets_cost_per_unit
  number_of_jackets = 4 :=
by
  sorry

end rita_bought_4_jackets_l118_118610


namespace exponent_zero_nonneg_l118_118029

theorem exponent_zero_nonneg (a : ℝ) (h : a ≠ -1) : (a + 1) ^ 0 = 1 :=
sorry

end exponent_zero_nonneg_l118_118029


namespace veronica_max_area_l118_118393

noncomputable def max_area_garden : ℝ :=
  let l := 105
  let w := 420 - 2 * l
  l * w

theorem veronica_max_area : ∃ (A : ℝ), max_area_garden = 22050 :=
by
  use 22050
  show max_area_garden = 22050
  sorry

end veronica_max_area_l118_118393


namespace officeEmployees_l118_118416

noncomputable def totalEmployees 
  (averageSalaryAll : ℝ) 
  (averageSalaryOfficers : ℝ) 
  (averageSalaryManagers : ℝ) 
  (averageSalaryWorkers : ℝ) 
  (numOfficers : ℕ) 
  (numManagers : ℕ) 
  (numWorkers : ℕ) : ℕ := 
  if (numOfficers * averageSalaryOfficers + numManagers * averageSalaryManagers + numWorkers * averageSalaryWorkers) 
      = (numOfficers + numManagers + numWorkers) * averageSalaryAll 
  then numOfficers + numManagers + numWorkers 
  else 0

theorem officeEmployees
  (averageSalaryAll : ℝ)
  (averageSalaryOfficers : ℝ)
  (averageSalaryManagers : ℝ)
  (averageSalaryWorkers : ℝ)
  (numOfficers : ℕ)
  (numManagers : ℕ)
  (numWorkers : ℕ) :
  averageSalaryAll = 720 →
  averageSalaryOfficers = 1320 →
  averageSalaryManagers = 840 →
  averageSalaryWorkers = 600 →
  numOfficers = 10 →
  numManagers = 20 →
  (numOfficers * averageSalaryOfficers + numManagers * averageSalaryManagers + numWorkers * averageSalaryWorkers) 
    = (numOfficers + numManagers + numWorkers) * averageSalaryAll →
  totalEmployees averageSalaryAll averageSalaryOfficers averageSalaryManagers averageSalaryWorkers numOfficers numManagers numWorkers = 100 :=
by
  intros h1 h2 h3 h4 h5 h6 h7
  rw [h2, h3, h4, h5, h6] at h7
  rw [h1]
  simp [totalEmployees, h7]
  sorry

end officeEmployees_l118_118416


namespace power_equation_l118_118834

theorem power_equation (x a : ℝ) (h : x^(-a) = 3) : x^(2 * a) = 1 / 9 :=
sorry

end power_equation_l118_118834


namespace nth_term_l118_118988

theorem nth_term (b : ℕ → ℝ) (h₀ : b 1 = 1)
  (h_rec : ∀ n ≥ 1, (b (n + 1))^2 = 36 * (b n)^2) : 
  b 50 = 6^49 :=
by
  sorry

end nth_term_l118_118988


namespace eggs_in_each_basket_l118_118582

theorem eggs_in_each_basket
  (total_red_eggs : ℕ)
  (total_orange_eggs : ℕ)
  (h_red : total_red_eggs = 30)
  (h_orange : total_orange_eggs = 45)
  (eggs_in_each_basket : ℕ)
  (h_at_least : eggs_in_each_basket ≥ 5) :
  (total_red_eggs % eggs_in_each_basket = 0) ∧ 
  (total_orange_eggs % eggs_in_each_basket = 0) ∧
  eggs_in_each_basket = 15 := sorry

end eggs_in_each_basket_l118_118582


namespace union_complement_A_eq_l118_118122

open Set

universe u

noncomputable def U : Set ℝ := univ
def A : Set ℝ := { x | x < 2 }
def B : Set ℝ := { y | ∃ (x : ℝ), y = x^2 + 1 }

theorem union_complement_A_eq :
  A ∪ ((U \ B : Set ℝ) : Set ℝ) = { x | x < 2 } := by
  sorry

end union_complement_A_eq_l118_118122


namespace value_of_v_l118_118774

theorem value_of_v (n : ℝ) (v : ℝ) (h1 : 10 * n = v - 2 * n) (h2 : n = -4.5) : v = -9 := by
  sorry

end value_of_v_l118_118774


namespace scientific_notation_of_1_656_million_l118_118671

theorem scientific_notation_of_1_656_million :
  (1.656 * 10^6 = 1656000) := by
sorry

end scientific_notation_of_1_656_million_l118_118671


namespace length_of_CD_l118_118633

theorem length_of_CD
  (radius : ℝ)
  (length : ℝ)
  (total_volume : ℝ)
  (cylinder_volume : ℝ := π * radius^2 * length)
  (hemisphere_volume : ℝ := (2 * (2/3) * π * radius^3))
  (h1 : radius = 4)
  (h2 : total_volume = 432 * π)
  (h3 : total_volume = cylinder_volume + hemisphere_volume) :
  length = 22 := by
sorry

end length_of_CD_l118_118633


namespace kola_age_l118_118283

variables (x y : ℕ)

-- Condition 1: Kolya is twice as old as Olya was when Kolya was as old as Olya is now
def condition1 : Prop := x = 2 * (2 * y - x)

-- Condition 2: When Olya is as old as Kolya is now, their combined age will be 36 years.
def condition2 : Prop := (3 * x - y = 36)

theorem kola_age : condition1 x y → condition2 x y → x = 16 :=
by { sorry }

end kola_age_l118_118283


namespace logical_inconsistency_in_dihedral_angle_def_l118_118829

-- Define the given incorrect definition
def incorrect_dihedral_angle_def : String :=
  "A dihedral angle is an angle formed by two half-planes originating from one straight line."

-- Define the correct definition
def correct_dihedral_angle_def : String :=
  "A dihedral angle is a spatial figure consisting of two half-planes that share a common edge."

-- Define the logical inconsistency
theorem logical_inconsistency_in_dihedral_angle_def :
  incorrect_dihedral_angle_def ≠ correct_dihedral_angle_def := by
  sorry

end logical_inconsistency_in_dihedral_angle_def_l118_118829


namespace average_age_increase_l118_118623

theorem average_age_increase (average_age_students : ℕ) (num_students : ℕ) (teacher_age : ℕ) (new_avg_age : ℕ)
                             (h1 : average_age_students = 26) (h2 : num_students = 25) (h3 : teacher_age = 52)
                             (h4 : new_avg_age = (650 + teacher_age) / (num_students + 1))
                             (h5 : 650 = average_age_students * num_students) :
  new_avg_age - average_age_students = 1 := 
by
  sorry

end average_age_increase_l118_118623


namespace difference_of_squares_divisible_by_9_l118_118614

theorem difference_of_squares_divisible_by_9 (a b : ℤ) : 
  9 ∣ ((3 * a + 2)^2 - (3 * b + 2)^2) :=
by
  sorry

end difference_of_squares_divisible_by_9_l118_118614


namespace quadratic_trinomials_unique_root_value_l118_118460

theorem quadratic_trinomials_unique_root_value (p q : ℝ) :
  ∀ x, (x^2 + p * x + q) + (x^2 + q * x + p) = (2 * x^2 + (p + q) * x + (p + q)) →
  (((p + q = 0 ∨ p + q = 8) → (2 * (2:ℝ)^2 + (p + q) * 2 + (p + q) = 8 ∨ 2 * (2:ℝ)^2 + (p + q) * 2 + (p + q) = 32))) :=
by
  sorry

end quadratic_trinomials_unique_root_value_l118_118460


namespace cabbage_production_l118_118666

theorem cabbage_production (x y : ℕ) 
  (h1 : y^2 - x^2 = 127) 
  (h2 : y - x = 1) 
  (h3 : 2 * y = 128) : y^2 = 4096 := by
  sorry

end cabbage_production_l118_118666


namespace john_spent_on_sweets_l118_118748

theorem john_spent_on_sweets (initial_amount : ℝ) (amount_given_per_friend : ℝ) (friends : ℕ) (amount_left : ℝ) (total_spent_on_sweets : ℝ) :
  initial_amount = 20.10 →
  amount_given_per_friend = 1.00 →
  friends = 2 →
  amount_left = 17.05 →
  total_spent_on_sweets = initial_amount - (amount_given_per_friend * friends) - amount_left →
  total_spent_on_sweets = 1.05 :=
by
  intros h_initial h_given h_friends h_left h_spent
  sorry

end john_spent_on_sweets_l118_118748


namespace arithmetic_sequence_sum_ratio_l118_118149

variable {a : ℕ → ℚ}
variable {S : ℕ → ℚ}

-- Definition of arithmetic sequence sum
def arithmeticSum (n : ℕ) : ℚ :=
  (n / 2) * (a 1 + a n)

-- Given condition
axiom condition : (a 6) / (a 5) = 9 / 11

theorem arithmetic_sequence_sum_ratio :
  (S 11) / (S 9) = 1 :=
by
  sorry

end arithmetic_sequence_sum_ratio_l118_118149


namespace total_number_of_animals_is_650_l118_118358

def snake_count : Nat := 100
def arctic_fox_count : Nat := 80
def leopard_count : Nat := 20
def bee_eater_count : Nat := 10 * leopard_count
def cheetah_count : Nat := snake_count / 2
def alligator_count : Nat := 2 * (arctic_fox_count + leopard_count)

def total_animal_count : Nat :=
  snake_count + arctic_fox_count + leopard_count + bee_eater_count + cheetah_count + alligator_count

theorem total_number_of_animals_is_650 :
  total_animal_count = 650 :=
by
  sorry

end total_number_of_animals_is_650_l118_118358


namespace binomial_9_3_l118_118366

theorem binomial_9_3 : (Nat.choose 9 3) = 84 := by
  sorry

end binomial_9_3_l118_118366


namespace expression_bounds_l118_118422

theorem expression_bounds (x y z w : ℝ) (hx : 0 ≤ x ∧ x ≤ 1) (hy : 0 ≤ y ∧ y ≤ 1) (hz : 0 ≤ z ∧ z ≤ 1) (hw : 0 ≤ w ∧ w ≤ 1) :
  2 * Real.sqrt 2 ≤ (Real.sqrt (x^2 + (1 - y)^2) + Real.sqrt (y^2 + (1 - z)^2) +
                     Real.sqrt (z^2 + (1 - w)^2) + Real.sqrt (w^2 + (1 - x)^2)) ∧
  (Real.sqrt (x^2 + (1 - y)^2) + Real.sqrt (y^2 + (1 - z)^2) +
   Real.sqrt (z^2 + (1 - w)^2) + Real.sqrt (w^2 + (1 - x)^2)) ≤ 4 := sorry

end expression_bounds_l118_118422


namespace picnic_problem_l118_118964

variables (M W C A : ℕ)

theorem picnic_problem
  (H1 : M + W + C = 200)
  (H2 : A = C + 20)
  (H3 : M = 65)
  (H4 : A = M + W) :
  M - W = 20 :=
by sorry

end picnic_problem_l118_118964


namespace rain_on_both_days_l118_118650

-- Define the events probabilities
variables (P_M P_T P_N P_MT : ℝ)

-- Define the initial conditions
axiom h1 : P_M = 0.6
axiom h2 : P_T = 0.55
axiom h3 : P_N = 0.25

-- Define the statement to prove
theorem rain_on_both_days : P_MT = 0.4 :=
by
  -- The proof is omitted for now
  sorry

end rain_on_both_days_l118_118650


namespace bill_bathroom_visits_per_day_l118_118802

theorem bill_bathroom_visits_per_day
  (squares_per_use : ℕ)
  (rolls : ℕ)
  (squares_per_roll : ℕ)
  (days_supply : ℕ)
  (total_uses : squares_per_use = 5)
  (total_rolls : rolls = 1000)
  (squares_from_each_roll : squares_per_roll = 300)
  (total_days : days_supply = 20000) :
  ( (rolls * squares_per_roll) / days_supply / squares_per_use ) = 3 :=
by
  sorry

end bill_bathroom_visits_per_day_l118_118802


namespace negation_of_existential_proposition_l118_118318

theorem negation_of_existential_proposition : 
  (¬ ∃ x : ℝ, x^2 - x + 3 = 0) = (∀ x : ℝ, x^2 - x + 3 ≠ 0) :=
by
  sorry

end negation_of_existential_proposition_l118_118318


namespace not_coincidence_l118_118037

theorem not_coincidence (G : Type) [Fintype G] [DecidableEq G]
    (friend_relation : G → G → Prop)
    (h_friend : ∀ (a b : G), friend_relation a b → friend_relation b a)
    (initial_condition : ∀ (subset : Finset G), subset.card = 4 → 
         ∃ x ∈ subset, ∀ y ∈ subset, x ≠ y → friend_relation x y) :
    ∀ (subset : Finset G), subset.card = 4 → 
        ∃ x ∈ subset, ∀ y ∈ Finset.univ, x ≠ y → friend_relation x y :=
by
  intros subset h_card
  -- The proof would be constructed here
  sorry

end not_coincidence_l118_118037


namespace solve_equation_l118_118892

theorem solve_equation :
  (∀ x : ℝ, x ≠ 2 / 3 → (7 * x + 3) / (3 * x ^ 2 + 7 * x - 6) = (3 * x) / (3 * x - 2) → (x = 1 / 3) ∨ (x = -3)) :=
by
  sorry

end solve_equation_l118_118892


namespace largest_number_not_sum_of_two_composites_l118_118105

-- Definitions related to composite numbers and natural numbers
def is_composite (n : ℕ) : Prop :=
  n > 1 ∧ ∃ d : ℕ, d ∣ n ∧ d ≠ 1 ∧ d ≠ n

def cannot_be_sum_of_two_composites (n : ℕ) : Prop :=
  ¬(∃ a b : ℕ, is_composite a ∧ is_composite b ∧ n = a + b)

-- Lean statement of the problem
theorem largest_number_not_sum_of_two_composites : ∀ n : ℕ,
  (∀ m : ℕ, m > n → cannot_be_sum_of_two_composites m → false) → n = 11 :=
by
  sorry

end largest_number_not_sum_of_two_composites_l118_118105


namespace book_length_ratio_is_4_l118_118749

-- Define the initial conditions
def pages_when_6 : ℕ := 8
def age_when_start := 6
def multiple_at_twice_age := 5
def multiple_eight_years_after := 3
def current_pages : ℕ := 480

def pages_when_12 := pages_when_6 * multiple_at_twice_age
def pages_when_20 := pages_when_12 * multiple_eight_years_after

theorem book_length_ratio_is_4 :
  (current_pages : ℚ) / pages_when_20 = 4 := by
  -- We need to show the proof for the equality
  sorry

end book_length_ratio_is_4_l118_118749


namespace power_sum_divisible_by_5_l118_118373

theorem power_sum_divisible_by_5 (n : ℕ) : (2^(4*n + 1) + 3^(4*n + 1)) % 5 = 0 :=
by
  sorry

end power_sum_divisible_by_5_l118_118373


namespace total_selling_price_16800_l118_118072

noncomputable def total_selling_price (CP_per_toy : ℕ) : ℕ :=
  let CP_18 := 18 * CP_per_toy
  let Gain := 3 * CP_per_toy
  CP_18 + Gain

theorem total_selling_price_16800 :
  total_selling_price 800 = 16800 :=
by
  sorry

end total_selling_price_16800_l118_118072


namespace total_students_suggestion_l118_118226

theorem total_students_suggestion :
  let m := 324
  let b := 374
  let t := 128
  m + b + t = 826 := by
  sorry

end total_students_suggestion_l118_118226


namespace desk_length_l118_118130

theorem desk_length (width perimeter length : ℤ) (h1 : width = 9) (h2 : perimeter = 46) (h3 : perimeter = 2 * (length + width)) : length = 14 :=
by
  rw [h1, h2] at h3
  sorry

end desk_length_l118_118130


namespace replace_asterisk_l118_118060

theorem replace_asterisk :
  ∃ x : ℤ, (x / 21) * (63 / 189) = 1 ∧ x = 63 := sorry

end replace_asterisk_l118_118060


namespace find_numbers_l118_118461

theorem find_numbers (x y : ℕ) (hx : 10 ≤ x ∧ x < 100) (hy : 10 ≤ y ∧ y < 100)
                     (hxy_mul : 2000 ≤ x * y ∧ x * y < 3000) (hxy_add : 100 ≤ x + y ∧ x + y < 1000)
                     (h_digit_relation : x * y = 2000 + x + y) : 
                     (x = 24 ∧ y = 88) ∨ (x = 88 ∧ y = 24) ∨ (x = 30 ∧ y = 70) ∨ (x = 70 ∧ y = 30) :=
by
  -- The proof will go here
  sorry

end find_numbers_l118_118461


namespace cloud9_total_money_l118_118522

-- Definitions
def I : ℕ := 12000
def G : ℕ := 16000
def R : ℕ := 1600

-- Total money taken after cancellations
def T : ℕ := I + G - R

-- Theorem stating that T is 26400
theorem cloud9_total_money : T = 26400 := by
  unfold T I G R
  sorry

end cloud9_total_money_l118_118522


namespace smallest_k_l118_118655

def arith_seq_sum (k n : ℕ) : ℕ :=
  (n + 1) * (2 * k + n) / 2

theorem smallest_k (k n : ℕ) (h_sum : arith_seq_sum k n = 100) :
  k = 9 :=
by
  sorry

end smallest_k_l118_118655


namespace volume_of_prism_l118_118621

variables (a b c : ℝ)
variables (ab_prod : a * b = 36) (ac_prod : a * c = 48) (bc_prod : b * c = 72)

theorem volume_of_prism : a * b * c = 352.8 :=
by
  sorry

end volume_of_prism_l118_118621


namespace mathematicians_correctness_l118_118905

theorem mathematicians_correctness:
  ∀ (w1₁ w1₂ t1₁ t1₂ w2₁ w2₂ t2₁ t2₂: ℕ),
  (w1₁ = 4) ∧ (t1₁ = 7) ∧ (w1₂ = 3) ∧ (t1₂ = 5) →
  (w2₁ = 8) ∧ (t2₁ = 14) ∧ (w2₂ = 3) ∧ (t2₂ = 5) →
  (4/7 : ℚ) < 3/5 →
  (w1₁ + w1₂) / (t1₁ + t1₂) = 7 / 12 ∨ (w2₁ + w2₂) / (t2₁ + t2₂) = 11 / 19 →
  ¬ (w1₁ + w1₂) / (t1₁ + t1₂) < 19 / 35 ∧ (w2₁ + w2₂) / (t2₁ + t2₂) < 19 / 35 :=
by
  sorry

end mathematicians_correctness_l118_118905


namespace ellipse_area_condition_l118_118839

open Real

theorem ellipse_area_condition (m : ℝ) :
  (∀ x y : ℝ, ((x^2)/3 + y^2 = 1) → (y = x + m) → 
  let F₁ := (-sqrt 2, 0)
      F₂ := (sqrt 2, 0)
  in distance F₁ (x, y) = 2 * distance F₂ (x, y)) → 
  m = -sqrt 2 / 3 :=
by
  sorry

end ellipse_area_condition_l118_118839


namespace blocks_selection_count_l118_118955

theorem blocks_selection_count :
  let n := 6 in
  let k := 4 in
  (Nat.choose n k * Nat.choose n k * Nat.factorial k = 5400) :=
by
  let n := 6
  let k := 4
  have h1 : Nat.choose n k = 15 := by sorry
  have h2 : Nat.factorial k = 24 := by sorry
  calc
    Nat.choose n k * Nat.choose n k * Nat.factorial k
      = 15 * 15 * 24 : by rw [h1, h1, h2]
  ... = 5400 : by norm_num

end blocks_selection_count_l118_118955


namespace intersection_point_l118_118232

theorem intersection_point :
  (∃ (x y : ℝ), 5 * x - 3 * y = 15 ∧ 4 * x + 2 * y = 14)
  → (∃ (x y : ℝ), x = 3 ∧ y = 1) :=
by
  intro h
  sorry

end intersection_point_l118_118232


namespace eggs_per_basket_l118_118016

theorem eggs_per_basket (n : ℕ) (total_eggs_red total_eggs_orange min_eggs_per_basket : ℕ) (h_red : total_eggs_red = 20) (h_orange : total_eggs_orange = 30) (h_min : min_eggs_per_basket = 5) (h_div_red : total_eggs_red % n = 0) (h_div_orange : total_eggs_orange % n = 0) (h_at_least : n ≥ min_eggs_per_basket) : n = 5 :=
sorry

end eggs_per_basket_l118_118016


namespace negation_of_implication_l118_118697

variable (a b c : ℝ)

theorem negation_of_implication :
  (¬(a + b + c = 3) → a^2 + b^2 + c^2 < 3) ↔
  ¬((a + b + c = 3) → a^2 + b^2 + c^2 ≥ 3) := by
sorry

end negation_of_implication_l118_118697


namespace determine_m_l118_118707

noncomputable def f (m x : ℝ) := (m^2 - m - 1) * x^(-5 * m - 3)

theorem determine_m : ∃ m : ℝ, (∀ x > 0, f m x = (m^2 - m - 1) * x^(-5 * m - 3)) ∧ (∀ x > 0, (m^2 - m - 1) * x^(-(5 * m + 3)) = (m^2 - m - 1) * x^(-5 * m - 3) → -5 * m - 3 > 0) ∧ m = -1 :=
by
  sorry

end determine_m_l118_118707


namespace union_A_B_l118_118551

-- Definitions for the sets A and B
def A : Set ℝ := {1, 2, 3, 4}
def B : Set ℝ := {x | -1 < x ∧ x < 3}

-- The statement to be proven
theorem union_A_B :
  A ∪ B = {x | (-1 < x ∧ x ≤ 3) ∨ x = 4} :=
sorry

end union_A_B_l118_118551


namespace valid_probabilities_and_invalid_probability_l118_118901

theorem valid_probabilities_and_invalid_probability :
  (let first_box_1 := (4, 7)
       second_box_1 := (3, 5)
       combined_prob_1 := (first_box_1.1 + second_box_1.1) / (first_box_1.2 + second_box_1.2),
       first_box_2 := (8, 14)
       second_box_2 := (3, 5)
       combined_prob_2 := (first_box_2.1 + second_box_2.1) / (first_box_2.2 + second_box_2.2),
       prob_1 := first_box_1.1 / first_box_1.2,
       prob_2 := second_box_2.1 / second_box_2.2
     in (combined_prob_1 = 7 / 12 ∧ combined_prob_2 = 11 / 19) ∧ (19 / 35 < prob_1 ∧ prob_1 < prob_2) → False) :=
by
  sorry

end valid_probabilities_and_invalid_probability_l118_118901


namespace probability_irrational_number_l118_118918

open Real

def cards : List ℝ := [22/7, sqrt 6, -0.5, Real.pi, 0]

theorem probability_irrational_number :
  (∃ P : ℙ, P.event_set = (λ x, x ∈ {sqrt 6, Real.pi}).to_finset) →
  P.probability (λ x, x ∈ {sqrt 6, Real.pi}.to_finset) = 2 / 5 := by
  sorry

end probability_irrational_number_l118_118918


namespace find_b_l118_118254

noncomputable def circle1 (x y a : ℝ) : Prop := x^2 + y^2 - 4*x + 2*y + 5 - a^2 = 0
noncomputable def circle2 (x y b : ℝ) : Prop := x^2 + y^2 - (2*b - 10)*x - 2*b*y + 2*b^2 - 10*b + 16 = 0
def is_intersection (x1 y1 x2 y2 : ℝ) : Prop := x1^2 + y1^2 = x2^2 + y2^2

theorem find_b (a x1 y1 x2 y2 : ℝ) (b : ℝ) :
  (circle1 x1 y1 a) ∧ (circle1 x2 y2 a) ∧ 
  (circle2 x1 y1 b) ∧ (circle2 x2 y2 b) ∧ 
  is_intersection x1 y1 x2 y2 →
  b = 5 / 3 :=
sorry

end find_b_l118_118254


namespace not_P_4_given_not_P_5_l118_118965

-- Define the proposition P for natural numbers
def P (n : ℕ) : Prop := sorry

-- Define the statement we need to prove
theorem not_P_4_given_not_P_5 (h1 : ∀ k : ℕ, P k → P (k + 1)) (h2 : ¬ P 5) : ¬ P 4 := by
  sorry

end not_P_4_given_not_P_5_l118_118965


namespace arrangement_count_5_l118_118457

open Finset

theorem arrangement_count_5 (A B : Fin 5) :
  (card ((finPerm 5).filter (λ σ, σ 0 ≠ A ∧ σ 4 ≠ B))) = 72 := 
sorry

end arrangement_count_5_l118_118457


namespace solve_fraction_eq_l118_118764

theorem solve_fraction_eq : 
  ∀ x : ℝ, (x - 3) ≠ 0 → (x + 6) / (x - 3) = 4 → x = 6 := by
  intros x h_ne_zero h_eq
  sorry

end solve_fraction_eq_l118_118764


namespace find_x_l118_118292

def binary_operation (a b c d : Int) : Int × Int := (a - c, b + d)

theorem find_x (x y : Int)
  (H1 : binary_operation 6 5 2 3 = (4, 8))
  (H2 : binary_operation x y 5 4 = (4, 8)) :
  x = 9 :=
by
  -- Necessary conditions and hypotheses are provided
  sorry -- Proof not required

end find_x_l118_118292


namespace ratio_problem_l118_118169

theorem ratio_problem 
  (a b c d : ℚ)
  (h1 : a / b = 3)
  (h2 : b / c = 2 / 5)
  (h3 : c / d = 9) : 
  d / a = 5 / 54 :=
by
  sorry

end ratio_problem_l118_118169


namespace probability_at_least_one_of_any_two_probability_at_least_one_of_three_l118_118664

open ProbabilityTheory

noncomputable def stock_profitability (pA pB pC : ℝ) : Prop :=
  let A := Event pA
  let B := Event pB
  let C := Event pC
  independent A B ∧ independent A C ∧ independent B C

theorem probability_at_least_one_of_any_two (pA pB pC : ℝ)
  (h_ind: stock_profitability pA pB pC)
  (hA : pA = 0.8)
  (hB : pB = 0.6)
  (hC : pC = 0.5) :
  P(at_least_two_of_three(A, B, C)) = 0.7 := sorry

theorem probability_at_least_one_of_three (pA pB pC : ℝ)
  (h_ind: stock_profitability pA pB pC)
  (hA : pA = 0.8)
  (hB : pB = 0.6)
  (hC : pC = 0.5) :
  P(at_least_one_of_three(A, B, C)) = 0.96 := sorry

def at_least_two_of_three (A B C : Event ℝ) : Event ℝ :=
  A ∧ B ∨ A ∧ C ∨ B ∧ C ∨ A ∧ B ∧ C

def at_least_one_of_three (A B C : Event ℝ) : Event ℝ :=
  A ∨ B ∨ C

end probability_at_least_one_of_any_two_probability_at_least_one_of_three_l118_118664


namespace general_formula_arithmetic_sequence_l118_118396

theorem general_formula_arithmetic_sequence :
  (∃ (a_n : ℕ → ℕ) (d : ℕ), d ≠ 0 ∧ 
    (a_2 = a_1 + d) ∧ 
    (a_4 = a_1 + 3 * d) ∧ 
    (a_2^2 = a_1 * a_4) ∧
    (a_5 = a_1 + 4 * d) ∧ 
    (a_6 = a_1 + 5 * d) ∧ 
    (a_5 + a_6 = 11) ∧ 
    ∀ n, a_n = a_1 + (n - 1) * d) → 
  ∀ n, a_n = n := 
sorry

end general_formula_arithmetic_sequence_l118_118396


namespace least_n_divisibility_l118_118782

theorem least_n_divisibility : ∃ n : ℕ, 
  (n > 0) ∧ 
  (∀ k : ℕ, 1 ≤ k ∧ k ≤ n → n * (n - 1) % k = 0) ∧ 
  (∃ k : ℕ, 1 ≤ k ∧ k ≤ n ∧ n * (n - 1) % k ≠ 0) ∧ 
  n = 5 :=
by
  sorry

end least_n_divisibility_l118_118782


namespace price_per_postcard_is_correct_l118_118227

noncomputable def initial_postcards : ℕ := 18
noncomputable def sold_postcards : ℕ := initial_postcards / 2
noncomputable def price_per_postcard_sold : ℕ := 15
noncomputable def total_earned : ℕ := sold_postcards * price_per_postcard_sold
noncomputable def total_postcards_after : ℕ := 36
noncomputable def remaining_original_postcards : ℕ := initial_postcards - sold_postcards
noncomputable def new_postcards_bought : ℕ := total_postcards_after - remaining_original_postcards
noncomputable def price_per_new_postcard : ℕ := total_earned / new_postcards_bought

theorem price_per_postcard_is_correct:
  price_per_new_postcard = 5 :=
by
  sorry

end price_per_postcard_is_correct_l118_118227


namespace find_circle_equation_l118_118703

-- Define the intersection point of the lines x + y + 1 = 0 and x - y - 1 = 0
def center : ℝ × ℝ := (0, -1)

-- Define the chord length AB
def chord_length : ℝ := 6

-- Line equation that intersects the circle
def line_eq (x y : ℝ) : Prop := 3 * x + 4 * y - 11 = 0

-- Circle equation to be proven
def circle_eq (x y : ℝ) : Prop := x^2 + (y + 1)^2 = 18

-- Main theorem: Prove that the given circle equation is correct under the conditions
theorem find_circle_equation (x y : ℝ) (hc : x + y + 1 = 0) (hc' : x - y - 1 = 0) 
  (hl : line_eq x y) : circle_eq x y :=
sorry

end find_circle_equation_l118_118703


namespace doug_age_l118_118164

theorem doug_age (Qaddama Jack Doug : ℕ) 
  (h1 : Qaddama = Jack + 6)
  (h2 : Jack = Doug - 3)
  (h3 : Qaddama = 19) : 
  Doug = 16 := 
by 
  sorry

end doug_age_l118_118164


namespace solve_exp_equation_l118_118167

theorem solve_exp_equation (e : ℝ) (x : ℝ) (h_e : e = Real.exp 1) :
  e^x + 2 * e^(-x) = 3 ↔ x = 0 ∨ x = Real.log 2 :=
sorry

end solve_exp_equation_l118_118167


namespace four_digit_numbers_sum_30_l118_118544

-- Definitions of the variables and constraints
def valid_digit (n : ℕ) : Prop := 0 ≤ n ∧ n ≤ 9

-- The main statement we aim to prove
theorem four_digit_numbers_sum_30 : 
  ∃ (count : ℕ), 
  count = 20 ∧ 
  ∃ (a b c d : ℕ), 
  (1 ≤ a ∧ valid_digit a) ∧ 
  (valid_digit b) ∧ 
  (valid_digit c) ∧ 
  (valid_digit d) ∧ 
  a + b + c + d = 30 := sorry

end four_digit_numbers_sum_30_l118_118544


namespace tracy_sold_paintings_l118_118049

-- Definitions based on conditions
def total_customers := 20
def customers_2_paintings := 4
def customers_1_painting := 12
def customers_4_paintings := 4

def paintings_per_customer_2 := 2
def paintings_per_customer_1 := 1
def paintings_per_customer_4 := 4

-- Theorem statement
theorem tracy_sold_paintings : 
  (customers_2_paintings * paintings_per_customer_2) + 
  (customers_1_painting * paintings_per_customer_1) + 
  (customers_4_paintings * paintings_per_customer_4) = 36 := 
by
  sorry

end tracy_sold_paintings_l118_118049


namespace part1_part2_l118_118129

-- Condition for exponents of x to be equal
def condition1 (a : ℤ) : Prop := (3 : ℤ) = 2 * a - 3

-- Condition for exponents of y to be equal
def condition2 (b : ℤ) : Prop := b = 1

noncomputable def a_value : ℤ := 3
noncomputable def b_value : ℤ := 1

-- Theorem for part (1): values of a and b
theorem part1 : condition1 3 ∧ condition2 1 :=
by
  have ha : condition1 3 := by sorry
  have hb : condition2 1 := by sorry
  exact And.intro ha hb

-- Theorem for part (2): value of (7a - 22)^2024 given a = 3
theorem part2 : (7 * a_value - 22) ^ 2024 = 1 :=
by
  have hx : 7 * a_value - 22 = -1 := by sorry
  have hres : (-1) ^ 2024 = 1 := by sorry
  exact Eq.trans (congrArg (fun x => x ^ 2024) hx) hres

end part1_part2_l118_118129


namespace inequality_proof_l118_118692

variable {x y z : ℝ}

theorem inequality_proof (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) :
  (1 + x + 2 * x^2) * (2 + 3 * y + y^2) * (4 + z + z^2) ≥ 60 * x * y * z :=
by
  sorry

end inequality_proof_l118_118692


namespace children_l118_118888

theorem children's_book_pages (P : ℝ)
  (h1 : P > 0)
  (c1 : ∃ P_rem, P_rem = P - (0.2 * P))
  (c2 : ∃ P_today, P_today = (0.35 * (P - (0.2 * P))))
  (c3 : ∃ Pages_left, Pages_left = (P - (0.2 * P) - (0.35 * (P - (0.2 * P)))) ∧ Pages_left = 130) :
  P = 250 := by
  sorry

end children_l118_118888


namespace lying_dwarf_possible_numbers_l118_118942

theorem lying_dwarf_possible_numbers (a_1 a_2 a_3 a_4 a_5 a_6 a_7 : ℕ)
  (h2 : a_2 = a_1) 
  (h3 : a_3 = a_1 + a_2) 
  (h4 : a_4 = a_1 + a_2 + a_3) 
  (h5 : a_5 = a_1 + a_2 + a_3 + a_4) 
  (h6 : a_6 = a_1 + a_2 + a_3 + a_4 + a_5) 
  (h7 : a_7 = a_1 + a_2 + a_3 + a_4 + a_5 + a_6)
  (h_sum : a_1 + a_2 + a_3 + a_4 + a_5 + a_6 + a_7 = 58)
  (h_lied : ∃ (d : ℕ), d ∈ [a_1, a_2, a_3, a_4, a_5, a_6, a_7] ∧ (d ≠ a_1 ∧ d ≠ a_2 ∧ d ≠ a_3 ∧ d ≠ a_4 ∧ d ≠ a_5 ∧ d ≠ a_6 ∧ d ≠ a_7)) : 
  ∃ x : ℕ, x = 13 ∨ x = 26 :=
by
  sorry

end lying_dwarf_possible_numbers_l118_118942


namespace payment_correct_l118_118599

def total_payment (hours1 hours2 hours3 : ℕ) (rate_per_hour : ℕ) (num_men : ℕ) : ℕ :=
  (hours1 + hours2 + hours3) * rate_per_hour * num_men

theorem payment_correct :
  total_payment 10 8 15 10 2 = 660 :=
by
  -- We skip the proof here
  sorry

end payment_correct_l118_118599


namespace donut_combinations_l118_118981

theorem donut_combinations (donuts types : ℕ) (at_least_one : ℕ) :
  donuts = 7 ∧ types = 5 ∧ at_least_one = 4 → ∃ combinations : ℕ, combinations = 100 :=
by
  intros h
  sorry

end donut_combinations_l118_118981


namespace range_of_k_l118_118849

noncomputable def operation (a b : ℝ) : ℝ := Real.sqrt (a * b) + a + b

theorem range_of_k (k : ℝ) (h : operation 1 (k^2) < 3) : -1 < k ∧ k < 1 :=
by
  sorry

end range_of_k_l118_118849


namespace each_half_month_has_15_days_l118_118453

noncomputable def days_in_each_half (total_days : ℕ) (mean_profit_total: ℚ) 
  (mean_profit_first_half: ℚ) (mean_profit_last_half: ℚ) : ℕ :=
  let first_half_days := total_days / 2
  let second_half_days := total_days - first_half_days
  first_half_days

theorem each_half_month_has_15_days (total_days : ℕ) (mean_profit_total: ℚ) 
  (mean_profit_first_half: ℚ) (mean_profit_last_half: ℚ) :
  total_days = 30 → mean_profit_total = 350 → mean_profit_first_half = 275 → mean_profit_last_half = 425 → 
  days_in_each_half total_days mean_profit_total mean_profit_first_half mean_profit_last_half = 15 :=
by
  intros h_days h_total h_first h_last
  sorry

end each_half_month_has_15_days_l118_118453


namespace count_divisible_subsets_l118_118592

open Finset BigOperators

theorem count_divisible_subsets (p : ℕ) (hp : Nat.Prime p) (hp_odd: p % 2 = 1) :
  let F := (Finset.range p).erase 0
  let s (M : Finset ℕ) := M.sum id
  let count_subsets := (F.powerset.filter (λ T, (¬ T = ∅ ∧ p ∣ s T))).card
  count_subsets = (2^(p-1) - 1) / p := by
  sorry

end count_divisible_subsets_l118_118592


namespace trains_meet_distance_from_delhi_l118_118510

-- Define the speeds of the trains as constants
def speed_bombay_express : ℕ := 60  -- kmph
def speed_rajdhani_express : ℕ := 80  -- kmph

-- Define the time difference in hours between the departures of the two trains
def time_difference : ℕ := 2  -- hours

-- Define the distance the Bombay Express travels before the Rajdhani Express starts
def distance_head_start : ℕ := speed_bombay_express * time_difference

-- Define the relative speed between the two trains
def relative_speed : ℕ := speed_rajdhani_express - speed_bombay_express

-- Define the time taken for the Rajdhani Express to catch up with the Bombay Express
def time_to_meet : ℕ := distance_head_start / relative_speed

-- The final meeting distance from Delhi for the Rajdhani Express
def meeting_distance : ℕ := speed_rajdhani_express * time_to_meet

-- Theorem stating the solution to the problem
theorem trains_meet_distance_from_delhi : meeting_distance = 480 :=
by sorry  -- proof is omitted

end trains_meet_distance_from_delhi_l118_118510


namespace quadratic_eq_coeffs_l118_118494

theorem quadratic_eq_coeffs (x : ℝ) : 
  ∃ a b c : ℝ, 3 * x^2 + 1 - 6 * x = a * x^2 + b * x + c ∧ a = 3 ∧ b = -6 ∧ c = 1 :=
by sorry

end quadratic_eq_coeffs_l118_118494


namespace common_difference_of_arithmetic_seq_l118_118007

def arithmetic_sequence (a : ℕ → ℝ) (d : ℝ) : Prop :=
  ∀ n : ℕ, a n = a 0 + n * d

theorem common_difference_of_arithmetic_seq :
  ∀ (a : ℕ → ℝ) (d : ℝ),
  arithmetic_sequence a d →
  (a 4 + a 8 = 10) →
  (a 10 = 6) →
  d = 1 / 4 :=
by
  intros a d h_seq h1 h2
  sorry

end common_difference_of_arithmetic_seq_l118_118007


namespace loom_weaving_rate_l118_118344

theorem loom_weaving_rate :
  (119.04761904761905 : ℝ) > 0 ∧ (15 : ℝ) > 0 ∧ ∃ rate : ℝ, rate = 15 / 119.04761904761905 → rate = 0.126 :=
by sorry

end loom_weaving_rate_l118_118344


namespace above_limit_l118_118736

/-- John travels 150 miles in 2 hours. -/
def john_travel_distance : ℝ := 150

/-- John travels for 2 hours. -/
def john_travel_time : ℝ := 2

/-- The speed limit is 60 mph. -/
def speed_limit : ℝ := 60

/-- The speed of John during his travel. -/
def john_speed : ℝ := john_travel_distance / john_travel_time

/-- How many mph above the speed limit was John driving? -/
def speed_above_limit : ℝ := john_speed - speed_limit

theorem above_limit : speed_above_limit = 15 := 
by
  unfold speed_above_limit john_speed john_travel_distance john_travel_time speed_limit
  have h1: 150 / 2 = 75 := by norm_num
  have h2: 75 - 60 = 15 := by norm_num
  rw [h1, h2]
  refl

end above_limit_l118_118736


namespace rightmost_three_digits_of_7_pow_1987_l118_118330

theorem rightmost_three_digits_of_7_pow_1987 :
  (7^1987 : ℕ) % 1000 = 643 := 
by 
  sorry

end rightmost_three_digits_of_7_pow_1987_l118_118330


namespace probability_same_flips_l118_118134

theorem probability_same_flips (prob_heads_faircoin : ℝ)
  (prob_heads_biasedcoin : ℝ)
  (h_faircoin : prob_heads_faircoin = 1/2)
  (h_biasedcoin : prob_heads_biasedcoin = 1/3) :
  ∑ n in finset.range (n + 1), (prob_heads_faircoin ^ (2 * n) * ((2/3) ^ (n - 1)) * prob_heads_biasedcoin) =
    1/17 :=
by
  sorry

end probability_same_flips_l118_118134


namespace fraction_identity_l118_118833

theorem fraction_identity (x y z : ℤ) (h : x / 2 = y / 3 ∧ y / 3 = z / 4) :
  (x + y) / (3 * y - 2 * z) = 5 :=
by
  sorry

end fraction_identity_l118_118833


namespace initial_red_balloons_l118_118307

variable (initial_red : ℕ)
variable (given_away : ℕ := 24)
variable (left_with : ℕ := 7)

theorem initial_red_balloons : initial_red = given_away + left_with :=
by sorry

end initial_red_balloons_l118_118307


namespace batsman_average_after_17th_inning_l118_118066

theorem batsman_average_after_17th_inning :
  ∀ (A : ℕ), (16 * A + 50) / 17 = A + 2 → A = 16 → A + 2 = 18 := by
  intros A h1 h2
  rw [h2] at h1
  linarith

end batsman_average_after_17th_inning_l118_118066


namespace convex_quadrilateral_division_l118_118759

-- Definitions for convex quadrilateral and some basic geometric objects.
structure Point :=
  (x : ℝ)
  (y : ℝ)

structure Quadrilateral :=
  (A B C D : Point)
  (convex : ∀ (X Y Z : Point), (X ≠ Y) ∧ (Y ≠ Z) ∧ (Z ≠ X))

-- Definitions for lines and midpoints.
def is_midpoint (M X Y : Point) : Prop :=
  M.x = (X.x + Y.x) / 2 ∧ M.y = (X.y + Y.y) / 2

-- Preliminary to determining equal area division.
def equal_area_division (Q : Quadrilateral) (L : Point → Point → Prop) : Prop :=
  ∃ F,
    is_midpoint F Q.A Q.B ∧
    -- Assuming some way to relate area with F and L
    L Q.D F ∧
    -- Placeholder for equality of areas (details depend on how we calculate area)
    sorry

-- Problem statement in Lean 4
theorem convex_quadrilateral_division (Q : Quadrilateral) :
  ∃ L, equal_area_division Q L :=
by
  -- Proof will be constructed here based on steps in the solution
  sorry

end convex_quadrilateral_division_l118_118759


namespace sum_red_equals_sum_blue_l118_118958

variable (r1 r2 r3 r4 b1 b2 b3 b4 w1 w2 w3 w4 : ℝ)

theorem sum_red_equals_sum_blue (h : (r1 + w1 / 2) + (r2 + w2 / 2) + (r3 + w3 / 2) + (r4 + w4 / 2) 
                                 = (b1 + w1 / 2) + (b2 + w2 / 2) + (b3 + w3 / 2) + (b4 + w4 / 2)) : 
  r1 + r2 + r3 + r4 = b1 + b2 + b3 + b4 :=
by sorry

end sum_red_equals_sum_blue_l118_118958


namespace max_value_inequality_l118_118288

theorem max_value_inequality (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : 5 * x + 3 * y < 90) : 
  x * y * (90 - 5 * x - 3 * y) ≤ 1800 := 
sorry

end max_value_inequality_l118_118288


namespace samson_mother_age_l118_118306

variable (S M : ℕ)
variable (x : ℕ)

def problem_statement : Prop :=
  S = 6 ∧
  S - x = 2 ∧
  M - x = 4 * 2 →
  M = 16

theorem samson_mother_age (S M x : ℕ) (h : problem_statement S M x) : M = 16 :=
by
  sorry

end samson_mother_age_l118_118306


namespace derivative_and_value_l118_118823

-- Given conditions
def eqn (x y : ℝ) : Prop := 10 * x^3 + 4 * x^2 * y + y^2 = 0

-- The derivative y'
def y_prime (x y y' : ℝ) : Prop := y' = (-15 * x^2 - 4 * x * y) / (2 * x^2 + y)

-- Specific values derivatives
def y_prime_at_x_neg2_y_4 (y' : ℝ) : Prop := y' = -7 / 3

-- The main theorem
theorem derivative_and_value (x y y' : ℝ) 
  (h1 : eqn x y) (x_neg2 : x = -2) (y_4 : y = 4) : 
  y_prime x y y' ∧ y_prime_at_x_neg2_y_4 y' :=
sorry

end derivative_and_value_l118_118823


namespace find_three_digit_number_l118_118042

theorem find_three_digit_number : 
  ∀ (c d e : ℕ), 0 ≤ c ∧ c < 10 ∧ 0 ≤ d ∧ d < 10 ∧ 0 ≤ e ∧ e < 10 ∧ 
  (10 * c + d) / 99 + (100 * c + 10 * d + e) / 999 = 44 / 99 → 
  100 * c + 10 * d + e = 400 :=
by {
  sorry
}

end find_three_digit_number_l118_118042


namespace number_of_people_is_ten_l118_118022

-- Define the total number of Skittles and the number of Skittles per person.
def total_skittles : ℕ := 20
def skittles_per_person : ℕ := 2

-- Define the number of people as the total Skittles divided by the Skittles per person.
def number_of_people : ℕ := total_skittles / skittles_per_person

-- Theorem stating that the number of people is 10.
theorem number_of_people_is_ten : number_of_people = 10 := sorry

end number_of_people_is_ten_l118_118022


namespace mean_temperature_correct_l118_118032

def temperatures : List ℤ := [-6, -3, -3, -4, 2, 4, 1]

def mean_temperature (temps : List ℤ) : ℚ :=
  (temps.sum : ℚ) / temps.length

theorem mean_temperature_correct :
  mean_temperature temperatures = -9 / 7 := 
by
  sorry

end mean_temperature_correct_l118_118032


namespace y_decreases_as_x_less_than_4_l118_118710

theorem y_decreases_as_x_less_than_4 (x : ℝ) : (x < 4) → ((x - 4)^2 + 3 < (4 - 4)^2 + 3) :=
by
  sorry

end y_decreases_as_x_less_than_4_l118_118710


namespace cost_of_adult_ticket_eq_19_l118_118153

variables (X : ℝ)
-- Condition 1: The cost of an adult ticket is $6 more than the cost of a child ticket.
def cost_of_child_ticket : ℝ := X - 6

-- Condition 2: The total cost of the 5 tickets is $77.
axiom total_cost_eq : 2 * X + 3 * (X - 6) = 77

-- Prove that the cost of an adult ticket is 19 dollars
theorem cost_of_adult_ticket_eq_19 (h : total_cost_eq) : X = 19 := 
by
  -- Here we would provide the actual proof steps
  sorry

end cost_of_adult_ticket_eq_19_l118_118153


namespace common_difference_arithmetic_sequence_l118_118421

theorem common_difference_arithmetic_sequence (a b : ℝ) :
  ∃ d : ℝ, b = a + 6 * d ∧ d = (b - a) / 6 :=
by
  sorry

end common_difference_arithmetic_sequence_l118_118421


namespace percentage_difference_l118_118807

variable {P Q : ℝ}

theorem percentage_difference (P Q : ℝ) : (100 * (Q - P)) / Q = ((Q - P) / Q) * 100 :=
by
  sorry

end percentage_difference_l118_118807


namespace ratio_of_students_to_dishes_l118_118290

theorem ratio_of_students_to_dishes (m n : ℕ) 
  (h_students : n > 0)
  (h_dishes : ∃ dishes : Finset ℕ, dishes.card = 100)
  (h_each_student_tastes_10 : ∀ student : Finset ℕ, student.card = 10) 
  (h_pairs_taste_by_m_students : ∀ {d1 d2 : ℕ} (hd1 : d1 ∈ Finset.range 100) (hd2 : d2 ∈ Finset.range 100), m = 10) 
  : n / m = 110 := by
  sorry

end ratio_of_students_to_dishes_l118_118290


namespace cos_monotonic_increasing_interval_l118_118630

open Real

noncomputable def monotonic_increasing_interval (k : ℤ) : Set ℝ :=
  {x : ℝ | k * π - π / 3 ≤ x ∧ x ≤ k * π + π / 6}

theorem cos_monotonic_increasing_interval (k : ℤ) :
  ∀ x : ℝ,
    (∃ y, y = cos (π / 3 - 2 * x)) →
    (monotonic_increasing_interval k x) :=
by
  sorry

end cos_monotonic_increasing_interval_l118_118630


namespace cost_of_adult_ticket_l118_118155

theorem cost_of_adult_ticket (c : ℝ) 
  (h₁ : 2 * (c + 6) + 3 * c = 77)
  : c + 6 = 19 :=
sorry

end cost_of_adult_ticket_l118_118155


namespace age_problem_l118_118724

theorem age_problem (M D : ℕ) (h1 : M = 40) (h2 : 2 * D + M = 70) : 2 * M + D = 95 := by
  sorry

end age_problem_l118_118724


namespace contrapositive_of_proposition_l118_118313

theorem contrapositive_of_proposition (a b : ℝ) : (a > b → a + 1 > b) ↔ (a + 1 ≤ b → a ≤ b) :=
sorry

end contrapositive_of_proposition_l118_118313


namespace rahul_spends_10_percent_on_clothes_l118_118390

theorem rahul_spends_10_percent_on_clothes 
    (salary : ℝ) (house_rent_percent : ℝ) (education_percent : ℝ) (remaining_after_expense : ℝ) (expenses : ℝ) (clothes_percent : ℝ) 
    (h_salary : salary = 2125) 
    (h_house_rent_percent : house_rent_percent = 0.20)
    (h_education_percent : education_percent = 0.10)
    (h_remaining_after_expense : remaining_after_expense = 1377)
    (h_expenses : expenses = salary * house_rent_percent + (salary - salary * house_rent_percent) * education_percent + (salary - salary * house_rent_percent - (salary - salary * house_rent_percent) * education_percent) * clothes_percent)
    (h_clothes_expense : remaining_after_expense = salary - (salary * house_rent_percent + (salary - salary * house_rent_percent) * education_percent + (salary - salary * house_rent_percent - (salary - salary * house_rent_percent) * education_percent) * clothes_percent)) :
    clothes_percent = 0.10 := 
by 
  sorry

end rahul_spends_10_percent_on_clothes_l118_118390


namespace range_of_m_max_value_of_t_l118_118238

-- Define the conditions for the quadratic equation problem
def quadratic_eq_has_real_roots (m n : ℝ) := 
  m^2 - 4 * n ≥ 0

def roots_are_negative (m : ℝ) := 
  2 ≤ m ∧ m < 3

-- Question 1: Prove range of m
theorem range_of_m (m : ℝ) (h1 : quadratic_eq_has_real_roots m (3 - m)) : 
  roots_are_negative m :=
sorry

-- Define the conditions for the inequality problem
def quadratic_inequality (m n : ℝ) (t : ℝ) := 
  t ≤ (m-1)^2 + (n-1)^2 + (m-n)^2

-- Question 2: Prove maximum value of t
theorem max_value_of_t (m n t : ℝ) (h1 : quadratic_eq_has_real_roots m n) : 
  quadratic_inequality m n t -> t ≤ 9/8 :=
sorry

end range_of_m_max_value_of_t_l118_118238


namespace units_digit_7_pow_1995_l118_118719

theorem units_digit_7_pow_1995 : 
  ∃ a : ℕ, a = 3 ∧ ∀ n : ℕ, (7^n % 10 = a) → ((n % 4) + 1 = 3) := 
by
  sorry

end units_digit_7_pow_1995_l118_118719


namespace golf_ratio_l118_118219

-- Definitions based on conditions
def first_turn_distance : ℕ := 180
def excess_distance : ℕ := 20
def total_distance_to_hole : ℕ := 250

-- Derived definitions based on conditions
def second_turn_distance : ℕ := (total_distance_to_hole - first_turn_distance) + excess_distance

-- Lean proof problem statement
theorem golf_ratio : (second_turn_distance : ℚ) / first_turn_distance = 1 / 2 :=
by
  -- use sorry to skip the proof
  sorry

end golf_ratio_l118_118219


namespace convert_spherical_coordinates_l118_118998

theorem convert_spherical_coordinates (
  ρ θ φ : ℝ
) (h1 : ρ = 5) (h2 : θ = 3 * Real.pi / 4) (h3 : φ = 9 * Real.pi / 4) : 
ρ = 5 ∧ 0 ≤ 7 * Real.pi / 4 ∧ 7 * Real.pi / 4 < 2 * Real.pi ∧ 0 ≤ Real.pi / 4 ∧ Real.pi / 4 ≤ Real.pi :=
by
  sorry

end convert_spherical_coordinates_l118_118998


namespace triangle_YZ_length_l118_118278

/-- In triangle XYZ, sides XY and XZ have lengths 6 and 8 inches respectively, 
    and the median XM from vertex X to the midpoint of side YZ is 5 inches. 
    Prove that the length of YZ is 10 inches. -/
theorem triangle_YZ_length
  (XY XZ XM : ℝ)
  (hXY : XY = 6)
  (hXZ : XZ = 8)
  (hXM : XM = 5) :
  ∃ (YZ : ℝ), YZ = 10 := 
by
  sorry

end triangle_YZ_length_l118_118278


namespace geom_seq_fraction_l118_118418

theorem geom_seq_fraction (a : ℕ → ℝ) (q : ℝ)
  (h_seq : ∀ n, a (n + 1) = q * a n)
  (h_sum1 : a 1 + a 2 = 1)
  (h_sum4 : a 4 + a 5 = -8) :
  (a 7 + a 8) / (a 5 + a 6) = -4 :=
sorry

end geom_seq_fraction_l118_118418


namespace officer_selection_at_least_two_past_l118_118137

theorem officer_selection_at_least_two_past (n m k : ℕ) (h₀ : n = 18) (h₁ : m = 6) (h₂ : k = 8) :
  let total := (nat.choose n m),
      zero_past := (nat.choose (n - k) m),
      one_past := k * (nat.choose (n - k) (m - 1)),
      at_least_two_past := total - (zero_past + one_past)
  in at_least_two_past = 16338 :=
by {
  rw [h₀, h₁, h₂],
  let total := nat.choose 18 6,
  let zero_past := nat.choose 10 6,
  let one_past := 8 * nat.choose 10 5,
  let at_least_two_past := total - (zero_past + one_past),
  have : total = 18564 := by simp [nat.choose],
  have : zero_past = 210 := by simp [nat.choose],
  have : one_past = 2016 := by simp [nat.choose],
  exact calc
    at_least_two_past
      = 18564 - (210 + 2016) : by { rw [this, this, this] }
      ... = 16338 : by norm_num
}

end officer_selection_at_least_two_past_l118_118137


namespace locus_of_centers_of_circles_l118_118690

structure Point (α : Type _) :=
(x : α)
(y : α)

noncomputable def perpendicular_bisector {α : Type _} [LinearOrderedField α] (A B : Point α) : Set (Point α) :=
  {C | ∃ m b : α, C.y = m * C.x + b ∧ A.y = m * A.x + b ∧ B.y = m * B.x + b ∧
                 (A.x - B.x) * C.x + (A.y - B.y) * C.y = (A.x^2 + A.y^2 - B.x^2 - B.y^2) / 2}

theorem locus_of_centers_of_circles {α : Type _} [LinearOrderedField α] (A B : Point α) :
  (∀ (C : Point α), (∃ r : α, r > 0 ∧ ∃ k: α, (C.x - A.x)^2 + (C.y - A.y)^2 = r^2 ∧ (C.x - B.x)^2 + (C.y - B.y)^2 = r^2) 
  → C ∈ perpendicular_bisector A B) :=
by
  sorry

end locus_of_centers_of_circles_l118_118690


namespace domain_of_f_l118_118450

-- Define the conditions
def sqrt_domain (x : ℝ) : Prop := x + 1 ≥ 0
def log_domain (x : ℝ) : Prop := 3 - x > 0

-- Define the function
noncomputable def f (x : ℝ) : ℝ := Real.sqrt (x + 1) + Real.log (3 - x)

-- Statement of the theorem
theorem domain_of_f : ∀ x, sqrt_domain x ∧ log_domain x ↔ -1 ≤ x ∧ x < 3 := by
  sorry

end domain_of_f_l118_118450


namespace find_unique_number_l118_118712

theorem find_unique_number : 
  ∃ X : ℕ, 
    (X % 1000 = 376 ∨ X % 1000 = 625) ∧ 
    (X * (X - 1) % 10000 = 0) ∧ 
    (Nat.gcd X (X - 1) = 1) ∧ 
    ((X % 625 = 0) ∨ ((X - 1) % 625 = 0)) ∧ 
    ((X % 16 = 0) ∨ ((X - 1) % 16 = 0)) ∧ 
    X = 9376 :=
by sorry

end find_unique_number_l118_118712


namespace log_sum_eq_two_l118_118325

theorem log_sum_eq_two : 
  ∀ (lg : ℝ → ℝ),
  (∀ x y : ℝ, lg (x * y) = lg x + lg y) →
  (∀ x y : ℝ, lg (x ^ y) = y * lg x) →
  lg 4 + 2 * lg 5 = 2 :=
by
  intros lg h1 h2
  sorry

end log_sum_eq_two_l118_118325


namespace students_sign_up_ways_l118_118661

theorem students_sign_up_ways :
  let students := 4
  let choices_per_student := 3
  (choices_per_student ^ students) = 3^4 :=
by
  sorry

end students_sign_up_ways_l118_118661


namespace max_y_value_l118_118717

theorem max_y_value (x : ℝ) : ∃ y : ℝ, y = -x^2 + 4 * x + 3 ∧ y ≤ 7 :=
by
  sorry

end max_y_value_l118_118717


namespace glove_selection_l118_118832

theorem glove_selection :
  let n := 6                -- Number of pairs
  let k := 4                -- Number of selected gloves
  let m := 1                -- Number of matching pairs
  let total_ways := n * 10 * 8 / 2  -- Calculation based on solution steps
  total_ways = 240 := by
  sorry

end glove_selection_l118_118832


namespace unique_solution_l118_118789

def unique_ordered_pair : Prop :=
  ∃ (m n : ℕ), m > 0 ∧ n > 0 ∧ 
               (∃ x : ℝ, x = (m : ℝ)^(1/3) - (n : ℝ)^(1/3) ∧ x^6 + 4 * x^3 - 36 * x^2 + 4 = 0) ∧
               m = 2 ∧ n = 4

theorem unique_solution : unique_ordered_pair := sorry

end unique_solution_l118_118789


namespace total_animals_count_l118_118359

def snakes := 100
def arctic_foxes := 80
def leopards := 20
def bee_eaters := 10 * leopards
def cheetahs := snakes / 2
def alligators := 2 * (arctic_foxes + leopards)

theorem total_animals_count : snakes + arctic_foxes + leopards + bee_eaters + cheetahs + alligators = 650 := by
  calc
    snakes + arctic_foxes + leopards + bee_eaters + cheetahs + alligators
        = 100 + 80 + 20 + 200 + 50 + 200 : by sorry -- Use the defined values
    ... = 650 : by sorry -- Perform the final addition

end total_animals_count_l118_118359


namespace ratio_of_areas_l118_118645
-- Define the conditions and the ratio to be proven
theorem ratio_of_areas (t r : ℝ) (h : 3 * t = 2 * π * r) : 
  (π^2 / 18) = (π^2 * r^2 / 9) / (2 * r^2) :=
by 
  sorry

end ratio_of_areas_l118_118645


namespace total_people_in_bus_l118_118196

-- Definitions based on the conditions
def left_seats : Nat := 15
def right_seats := left_seats - 3
def people_per_seat := 3
def back_seat_people := 9

-- Theorem statement
theorem total_people_in_bus : 
  (left_seats * people_per_seat) +
  (right_seats * people_per_seat) + 
  back_seat_people = 90 := 
by sorry

end total_people_in_bus_l118_118196


namespace solve_N1N2_identity_l118_118744

theorem solve_N1N2_identity :
  (∃ N1 N2 : ℚ,
    (∀ x : ℚ, x ≠ 1 ∧ x ≠ 3 →
      (42 * x - 37) / (x^2 - 4 * x + 3) =
      N1 / (x - 1) + N2 / (x - 3)) ∧ 
      N1 * N2 = -445 / 4) :=
by
  sorry

end solve_N1N2_identity_l118_118744


namespace count_four_digit_numbers_l118_118265

open Int

def is_digit (n : ℕ) : Prop := n >= 0 ∧ n <= 9

def valid_combination (a b c d : ℕ) : Prop :=
  is_digit a ∧ is_digit b ∧ is_digit c ∧ is_digit d ∧
  a ≠ 0 ∧
  a + b + c + d = 10 ∧
  (a + c) - (b + d) % 11 = 0

theorem count_four_digit_numbers :
  ∃ n, n > 0 ∧ ∀ a b c d : ℕ, valid_combination a b c d → n = sorry :=
sorry

end count_four_digit_numbers_l118_118265


namespace four_digit_number_divisible_by_11_l118_118259

theorem four_digit_number_divisible_by_11 :
  ∃ (a b c d : ℕ), 
    a > 0 ∧ a < 10 ∧ b < 10 ∧ c < 10 ∧ d < 10 ∧ 
    a + b + c + d = 10 ∧ 
    (a + c) % 11 = (b + d) % 11 ∧
    (10 - a != 0 ∨ 10 - c != 0 ∨ 10 - b != 0 ∨ 10 - d != 0) := sorry

end four_digit_number_divisible_by_11_l118_118259


namespace pufferfish_count_l118_118919

theorem pufferfish_count (s p : ℕ) (h1 : s = 5 * p) (h2 : s + p = 90) : p = 15 :=
by
  sorry

end pufferfish_count_l118_118919


namespace count_divisible_by_11_with_digits_sum_10_l118_118256

-- Definitions and conditions from step a)
def digits_add_up_to_10 (n : ℕ) : Prop :=
  let ds := n.digits 10 in ds.length = 4 ∧ ds.sum = 10

def divisible_by_11 (n : ℕ) : Prop :=
  let ds := n.digits 10 in
  (ds.nth 0).get_or_else 0 +
  (ds.nth 2).get_or_else 0 -
  (ds.nth 1).get_or_else 0 -
  (ds.nth 3).get_or_else 0 ∣ 11

-- The tuple (question, conditions, answer) formalized in Lean
theorem count_divisible_by_11_with_digits_sum_10 :
  ∃ N : ℕ, N = (
  (Finset.filter (λ x, digits_add_up_to_10 x ∧ divisible_by_11 x)
  (Finset.range 10000)).card
) := 
sorry

end count_divisible_by_11_with_digits_sum_10_l118_118256


namespace no_base_b_square_of_integer_l118_118847

theorem no_base_b_square_of_integer (b : ℕ) : ¬(∃ n : ℕ, n^2 = b^2 + 3 * b + 1) → b < 4 ∨ b > 8 := by
  sorry

end no_base_b_square_of_integer_l118_118847


namespace infinite_series_evaluation_l118_118379

theorem infinite_series_evaluation :
  (∑' m : ℕ, ∑' n : ℕ, 1 / (m * n * (m + n + 2))) = 3 :=
  sorry

end infinite_series_evaluation_l118_118379


namespace revenue_after_fall_is_correct_l118_118351

variable (originalRevenue : ℝ) (percentageDecrease : ℝ)

theorem revenue_after_fall_is_correct :
    originalRevenue = 69 ∧ percentageDecrease = 39.130434782608695 →
    originalRevenue - (originalRevenue * (percentageDecrease / 100)) = 42 := by
  intro h
  rcases h with ⟨h1, h2⟩
  sorry

end revenue_after_fall_is_correct_l118_118351


namespace A_B_work_together_finish_l118_118972
noncomputable def work_rate_B := 1 / 12
noncomputable def work_rate_A := 2 * work_rate_B
noncomputable def combined_work_rate := work_rate_A + work_rate_B

theorem A_B_work_together_finish (hB: work_rate_B = 1/12) (hA: work_rate_A = 2 * work_rate_B) :
  (1 / combined_work_rate) = 4 :=
by
  -- Placeholder for the proof, we don't need to provide the proof steps
  sorry

end A_B_work_together_finish_l118_118972


namespace solve_star_eq_five_l118_118693

def star (a b : ℝ) : ℝ := a + b^2

theorem solve_star_eq_five :
  ∃ x₁ x₂ : ℝ, star x₁ (x₁ + 1) = 5 ∧ star x₂ (x₂ + 1) = 5 ∧ x₁ = 1 ∧ x₂ = -4 :=
by
  sorry

end solve_star_eq_five_l118_118693


namespace solve_system_l118_118893

theorem solve_system (a b c x y z : ℝ) (h1 : a ≠ 0) (h2 : b ≠ 0) (h3 : c ≠ 0) 
  (eq1 : a * y + b * x = c)
  (eq2 : c * x + a * z = b)
  (eq3 : b * z + c * y = a) :
  x = (b^2 + c^2 - a^2) / (2 * b * c) ∧ 
  y = (a^2 + c^2 - b^2) / (2 * a * c) ∧ 
  z = (a^2 + b^2 - c^2) / (2 * a * b) :=
sorry

end solve_system_l118_118893


namespace JeremyTotalExpenses_l118_118142

noncomputable def JeremyExpenses : ℝ :=
  let motherGift := 400
  let fatherGift := 280
  let sisterGift := 100
  let brotherGift := 60
  let friendGift := 50
  let giftWrappingRate := 0.07
  let taxRate := 0.09
  let miscExpenses := 40
  let wrappingCost := motherGift * giftWrappingRate
                  + fatherGift * giftWrappingRate
                  + sisterGift * giftWrappingRate
                  + brotherGift * giftWrappingRate
                  + friendGift * giftWrappingRate
  let totalGiftCost := motherGift + fatherGift + sisterGift + brotherGift + friendGift
  let totalTax := totalGiftCost * taxRate
  wrappingCost + totalTax + miscExpenses

theorem JeremyTotalExpenses : JeremyExpenses = 182.40 := by
  sorry

end JeremyTotalExpenses_l118_118142


namespace mathematician_correctness_l118_118909

theorem mathematician_correctness (box1_white1 box1_total1 box2_white1 box2_total1 : ℕ)
                                  (prob1 prob2 : ℚ) :
  (box1_white1 = 4 ∧ box1_total1 = 7 ∧ box2_white1 = 3 ∧ box2_total1 = 5 ∧ prob1 = (4:ℚ) / 7 ∧ prob2 = (3:ℚ) / 5) →
  (box1_white2 = 8 ∧ box1_total2 = 14 ∧ box2_white2 = 3 ∧ box2_total2 = 5 ∧ prob1 = (8:ℚ) / 14 ∧ prob2 = (3:ℚ) / 5) →
  (prob1 < (7:ℚ) / 12 ∧ prob2 > (11:ℚ)/19) →
  ¬((19:ℚ)/35 > (4:ℚ)/7 ∧ (19:ℚ)/35 < (3:ℚ)/5) :=
by
  sorry

end mathematician_correctness_l118_118909


namespace inequality_for_positive_reals_l118_118844

variable {a b c : ℝ}
variable {k : ℕ}

theorem inequality_for_positive_reals 
  (hab : a > 0) 
  (hbc : b > 0) 
  (hac : c > 0) 
  (hprod : a * b * c = 1) 
  (hk : k ≥ 2) 
  : (a ^ k) / (a + b) + (b ^ k) / (b + c) + (c ^ k) / (c + a) ≥ 3 / 2 := 
sorry

end inequality_for_positive_reals_l118_118844


namespace value_of_star_15_25_l118_118805

noncomputable def star (x y : ℝ) : ℝ := Real.log x / Real.log y

axiom condition1 (x y : ℝ) (hxy : x > 0 ∧ y > 0) : star (star (x^2) y) y = star x y
axiom condition2 (x y : ℝ) (hxy : x > 0 ∧ y > 0) : star x (star y y) = star (star x y) (star x 1)
axiom condition3 (h : 1 > 0) : star 1 1 = 0

theorem value_of_star_15_25 : star 15 25 = (Real.log 3 / (2 * Real.log 5)) + 1 / 2 := 
by 
  sorry

end value_of_star_15_25_l118_118805


namespace nested_sqrt_expr_l118_118530

theorem nested_sqrt_expr (M : ℝ) (h : M > 1) : (↑(M) ^ (1 / 4) ^ (1 / 4) ^ (1 / 4)) = M ^ (21 / 64) :=
by
  sorry

end nested_sqrt_expr_l118_118530


namespace range_of_a_l118_118128

open Real

theorem range_of_a (a : ℝ) :
  (∀ x : ℝ, 0 < x → 2 * x * log x ≥ -x^2 + a * x - 3) → a ≤ 4 :=
by 
  sorry

end range_of_a_l118_118128


namespace range_of_p_l118_118548

noncomputable def proof_problem (p : ℝ) : Prop :=
  (∀ x : ℝ, (4 * x + p < 0) → (x < -1 ∨ x > 2)) → (p ≥ 4)

theorem range_of_p (p : ℝ) : proof_problem p :=
by
  intros h
  sorry

end range_of_p_l118_118548


namespace boxes_containing_pans_l118_118763

def num_boxes : Nat := 26
def num_teacups_per_box : Nat := 20
def num_cups_broken_per_box : Nat := 2
def teacups_left : Nat := 180

def num_teacup_boxes (num_boxes : Nat) (num_teacups_per_box : Nat) (num_cups_broken_per_box : Nat) (teacups_left : Nat) : Nat :=
  teacups_left / (num_teacups_per_box - num_cups_broken_per_box)

def num_remaining_boxes (num_boxes : Nat) (num_teacup_boxes : Nat) : Nat :=
  num_boxes - num_teacup_boxes

def num_pans_boxes (num_remaining_boxes : Nat) : Nat :=
  num_remaining_boxes / 2

theorem boxes_containing_pans : ∀ (num_boxes num_teacups_per_box num_cups_broken_per_box teacups_left : Nat),
  num_boxes = 26 →
  num_teacups_per_box = 20 →
  num_cups_broken_per_box = 2 →
  teacups_left = 180 →
  num_pans_boxes (num_remaining_boxes num_boxes (num_teacup_boxes num_boxes num_teacups_per_box num_cups_broken_per_box teacups_left)) = 8 :=
by
  intros
  sorry

end boxes_containing_pans_l118_118763


namespace minor_premise_incorrect_verification_l118_118946

-- Define the conditions
def exponential_function (a : ℝ) (hx : x : ℝ) : ℝ := a ^ x
def power_function (alpha : ℝ) (x : ℝ) : ℝ := x ^ alpha 

-- The major premise: Exponential functions are increasing for a > 1
axiom exp_increasing (a : ℝ) (h : a > 1) : StrictMono (exponential_function a)

-- The minor premise: The statement should consider y = x ^ α is an exponential function
axiom minor_premise_incorrect (α : ℝ) (h : α > 1) : ¬ (power_function α = exponential_function a)

-- The conclusion appropriately derived
theorem minor_premise_incorrect_verification
  (α : ℝ) (a : ℝ) (h1 : α > 1) (h2 : a > 1) :
  ¬ (power_function α = exponential_function a) :=
by
  apply minor_premise_incorrect; assumption

end minor_premise_incorrect_verification_l118_118946


namespace prob_task1_and_not_task2_l118_118939

def prob_task1_completed : ℚ := 5 / 8
def prob_task2_completed : ℚ := 3 / 5

theorem prob_task1_and_not_task2 : 
  ((prob_task1_completed) * (1 - prob_task2_completed)) = 1 / 4 := 
by 
  sorry

end prob_task1_and_not_task2_l118_118939


namespace matrix_inverse_proof_l118_118536

open Matrix

def matrix_inverse_problem : Prop :=
  let A := ![
    ![7, -2],
    ![-3, 1]
  ]
  let A_inv := ![
    ![1, 2],
    ![3, 7]
  ]
  A.mul A_inv = (1 : ℤ) • (1 : Matrix (Fin 2) (Fin 2))
  
theorem matrix_inverse_proof : matrix_inverse_problem :=
  by
  sorry

end matrix_inverse_proof_l118_118536


namespace lioness_hyena_age_ratio_l118_118767

variables {k H : ℕ}

-- Conditions
def lioness_age (lioness_age hyena_age : ℕ) : Prop := ∃ k, lioness_age = k * hyena_age
def lioness_is_12 (lioness_age : ℕ) : Prop := lioness_age = 12
def baby_age (mother_age baby_age : ℕ) : Prop := baby_age = mother_age / 2
def baby_ages_sum_in_5_years (baby_l_age baby_h_age sum : ℕ) : Prop := 
  (baby_l_age + 5) + (baby_h_age + 5) = sum

-- The statement to be proved
theorem lioness_hyena_age_ratio (H : ℕ)
  (h1 : lioness_age 12 H) 
  (h2 : baby_age 12 6) 
  (h3 : baby_age H (H / 2)) 
  (h4 : baby_ages_sum_in_5_years 6 (H / 2) 19) : 12 / H = 2 := 
sorry

end lioness_hyena_age_ratio_l118_118767


namespace sin_square_eq_c_div_a2_plus_b2_l118_118304

theorem sin_square_eq_c_div_a2_plus_b2 
  (a b c : ℝ) (α β : ℝ)
  (h1 : a * Real.cos α + b * Real.sin α = c)
  (h2 : a * Real.cos β + b * Real.sin β = 0)
  (not_all_zero : a ≠ 0 ∨ b ≠ 0 ∨ c ≠ 0) : 
  Real.sin (α - β) ^ 2 = c ^ 2 / (a ^ 2 + b ^ 2) :=
by
  sorry

end sin_square_eq_c_div_a2_plus_b2_l118_118304


namespace tracy_sold_paintings_l118_118050

-- Definitions based on conditions
def total_customers := 20
def customers_2_paintings := 4
def customers_1_painting := 12
def customers_4_paintings := 4

def paintings_per_customer_2 := 2
def paintings_per_customer_1 := 1
def paintings_per_customer_4 := 4

-- Theorem statement
theorem tracy_sold_paintings : 
  (customers_2_paintings * paintings_per_customer_2) + 
  (customers_1_painting * paintings_per_customer_1) + 
  (customers_4_paintings * paintings_per_customer_4) = 36 := 
by
  sorry

end tracy_sold_paintings_l118_118050


namespace simon_age_is_10_l118_118504

def alvin_age : ℕ := 30

def simon_age (alvin_age : ℕ) : ℕ :=
  (alvin_age / 2) - 5

theorem simon_age_is_10 : simon_age alvin_age = 10 := by
  sorry

end simon_age_is_10_l118_118504


namespace distance_between_lines_correct_l118_118558

noncomputable def distance_between_parallel_lines 
  (a b c₁ c₂ : ℝ) : ℝ :=
  |c₁ - c₂| / Real.sqrt (a^2 + b^2)

theorem distance_between_lines_correct :
  distance_between_parallel_lines 4 2 (-2) 1 = 3 * Real.sqrt 5 / 10 :=
by
  -- Proof steps would go here
  sorry

end distance_between_lines_correct_l118_118558


namespace pyramid_cone_radius_l118_118420

open Real

-- Definition of a regular pyramid and the parameters.
variable (a : ℝ) -- Side of the base of the pyramid
variable (BE OE OB r x : ℝ) 

-- Conditions from the problem statement
axiom ratio_condition : OE = 2 / 3 * BE
axiom height_condition : BE = a * sqrt 3 / 2
axiom radius_condition : r = a / 4

-- Statement encompassing the problem's solution
theorem pyramid_cone_radius (a : ℝ) (BE OE OB r x : ℝ) 
  (h1 : OE = 2 / 3 * BE)
  (h2 : BE = a * sqrt 3 / 2)
  (h3 : r = a / 4) :
  r = a / 4 ∧ 
  x = a * sqrt 3 / (2 * (2 * cos (1 / 3 * π / 2))) :=
sorry

end pyramid_cone_radius_l118_118420


namespace max_area_of_rectangular_playground_l118_118141

theorem max_area_of_rectangular_playground (P : ℕ) (hP : P = 160) :
  (∃ (x y : ℕ), 2 * (x + y) = P ∧ x * y = 1600) :=
by
  sorry

end max_area_of_rectangular_playground_l118_118141


namespace square_of_binomial_l118_118384

theorem square_of_binomial {a r s : ℚ} 
  (h1 : r^2 = a)
  (h2 : 2 * r * s = 18)
  (h3 : s^2 = 16) : 
  a = 81 / 16 :=
by sorry

end square_of_binomial_l118_118384


namespace distribute_tourists_l118_118866

-- Define the number of ways k tourists can distribute among n cinemas
def num_ways (n k : ℕ) : ℕ := n^k

-- Theorem stating the number of distribution ways
theorem distribute_tourists (n k : ℕ) : num_ways n k = n^k :=
by sorry

end distribute_tourists_l118_118866


namespace find_m_value_l118_118843

-- Define the ellipse equation and its foci
def ellipse (x y : ℝ) : Prop := (x^2 / 3) + y^2 = 1

-- Define the foci of the ellipse C
def F1 : ℝ × ℝ := (-Real.sqrt 2, 0)
def F2 : ℝ × ℝ := (Real.sqrt 2, 0)

-- Define the line equation
def line (m : ℝ) (x y : ℝ) : Prop := y = x + m

-- Define the condition for the areas of triangles F1AB and F2AB
def triangle_area_condition (F1 F2 : ℝ × ℝ) (m : ℝ) : Prop :=
  let A := sorry in -- Coordinates of intersection points need to be defined
  let B := sorry in -- Coordinates of intersection points need to be defined
  sorry /- The actual calculation of the areas would require more complex integration and geometry.
           Here we just outline the condition -/

-- The proof statement
theorem find_m_value :
  ∃ m : ℝ, 
    ∀ x y : ℝ,
      ellipse x y →
      line m x y →
      triangle_area_condition F1 F2 m →
      m = -Real.sqrt 2 / 3 :=
  sorry

end find_m_value_l118_118843


namespace hours_of_use_per_charge_l118_118873

theorem hours_of_use_per_charge
  (c h u : ℕ)
  (h_c : c = 10)
  (h_fraction : h = 6)
  (h_use : 6 * u = 12) :
  u = 2 :=
sorry

end hours_of_use_per_charge_l118_118873


namespace correct_average_l118_118464

-- let's define the numbers as a list
def numbers : List ℕ := [1200, 1300, 1510, 1520, 1530, 1200]

-- the condition given in the problem: the stated average is 1380
def stated_average : ℕ := 1380

-- given the correct calculation of average, let's write the theorem statement
theorem correct_average : (numbers.foldr (· + ·) 0) / numbers.length = 1460 :=
by
  -- we would prove it here
  sorry

end correct_average_l118_118464


namespace simplify_expression_l118_118513

theorem simplify_expression : 
  (((5 + 7 + 3) * 2 - 4) / 2 - (5 / 2) = 21 / 2) :=
by
  sorry

end simplify_expression_l118_118513


namespace pencils_problem_l118_118043

theorem pencils_problem (x : ℕ) :
  2 * x + 6 * 3 + 2 * 1 = 24 → x = 2 :=
by
  sorry

end pencils_problem_l118_118043


namespace range_of_a_l118_118399

open Real

theorem range_of_a (a b c : ℝ) (h₁ : a ≠ 0) (h₂ : a - b + c = 3) (h₃ : a + b + c = 1) (h₄ : 0 < c ∧ c < 1) : 1 < a ∧ a < 2 :=
by
  sorry

end range_of_a_l118_118399


namespace imaginary_part_of_z_l118_118117

theorem imaginary_part_of_z (z : ℂ) (h : z = 2 / (-1 + I)) : z.im = -1 :=
sorry

end imaginary_part_of_z_l118_118117


namespace cubic_roots_sum_cubes_l118_118019

theorem cubic_roots_sum_cubes
  (p q r : ℂ)
  (h_eq_root : ∀ x, x = p ∨ x = q ∨ x = r → x^3 - 2 * x^2 + 3 * x - 1 = 0)
  (h_sum : p + q + r = 2)
  (h_prod_sum : p * q + q * r + r * p = 3)
  (h_prod : p * q * r = 1) :
  p^3 + q^3 + r^3 = -7 := by
  sorry

end cubic_roots_sum_cubes_l118_118019


namespace probability_between_C_and_E_l118_118607

theorem probability_between_C_and_E
  (AB AD BC BE : ℝ)
  (h₁ : AB = 4 * AD)
  (h₂ : AB = 8 * BC)
  (h₃ : AB = 2 * BE) : 
  (AB / 2 - AB / 8) / AB = 3 / 8 :=
by 
  sorry

end probability_between_C_and_E_l118_118607


namespace solve_for_x_l118_118950

theorem solve_for_x :
  ∀ x : ℤ, 3 * x + 36 = 48 → x = 4 :=
by
  sorry

end solve_for_x_l118_118950


namespace calculate_expression_l118_118803

theorem calculate_expression :
  let s1 := 3 + 6 + 9
  let s2 := 4 + 8 + 12
  s1 = 18 → s2 = 24 → (s1 / s2 + s2 / s1) = 25 / 12 :=
by
  intros
  sorry

end calculate_expression_l118_118803


namespace roots_sum_of_squares_l118_118877

theorem roots_sum_of_squares {p q r : ℝ} 
  (h₁ : ∀ x : ℝ, (x - p) * (x - q) * (x - r) = x^3 - 24 * x^2 + 50 * x - 35) :
  (p + q)^2 + (q + r)^2 + (r + p)^2 = 1052 :=
by
  have h_sum : p + q + r = 24 := by sorry
  have h_product : p * q + q * r + r * p = 50 := by sorry
  sorry

end roots_sum_of_squares_l118_118877


namespace quadratic_solution_l118_118248

theorem quadratic_solution (a : ℝ) (h : (1 : ℝ)^2 + 1 + 2 * a = 0) : a = -1 :=
by {
  sorry
}

end quadratic_solution_l118_118248


namespace eccentricity_of_ellipse_l118_118850

theorem eccentricity_of_ellipse (a c : ℝ) (h1 : 2 * c = a) : (c / a) = (1 / 2) :=
by
  -- This is where we would write the proof, but we're using sorry to skip the proof steps.
  sorry

end eccentricity_of_ellipse_l118_118850


namespace circle_tangent_to_x_axis_at_origin_l118_118862

theorem circle_tangent_to_x_axis_at_origin
  (D E F : ℝ)
  (h_circle : ∀ x y : ℝ, x^2 + y^2 + Dx + Ey + F = 0)
  (h_tangent : ∃ x, x^2 + (0 : ℝ)^2 + Dx + E * 0 + F = 0 ∧ ∃ r : ℝ, ∀ x y, x^2 + (y - r)^2 = r^2) :
  D = 0 ∧ E ≠ 0 ∧ F ≠ 0 :=
by
  sorry

end circle_tangent_to_x_axis_at_origin_l118_118862


namespace total_amount_paid_l118_118487

theorem total_amount_paid (sales_tax : ℝ) (tax_rate : ℝ) (cost_tax_free_items : ℝ) : 
  sales_tax = 1.28 → tax_rate = 0.08 → cost_tax_free_items = 12.72 → 
  (sales_tax / tax_rate + sales_tax + cost_tax_free_items) = 30.00 :=
by
  intros h1 h2 h3
  -- Proceed with the proof using h1, h2, and h3
  sorry

end total_amount_paid_l118_118487


namespace obtuse_triangle_iff_distinct_real_roots_l118_118146

theorem obtuse_triangle_iff_distinct_real_roots
  (A B C : ℝ)
  (h_triangle : 2 * A + B = Real.pi)
  (h_isosceles : A = C) :
  (B > Real.pi / 2) ↔ (B^2 - 4 * A * C > 0) :=
sorry

end obtuse_triangle_iff_distinct_real_roots_l118_118146


namespace values_of_x_for_g_l118_118589

def g (x : ℝ) : ℝ := x^2 - 5 * x

theorem values_of_x_for_g (x : ℝ) :
  g (g x) = g x ↔ x = 0 ∨ x = 5 ∨ x = 6 ∨ x = -1 := by
    sorry

end values_of_x_for_g_l118_118589


namespace Mary_more_than_Tim_l118_118025

-- Define the incomes
variables (J T M : ℝ)

-- Conditions
def Tim_income : Prop := T = 0.80 * J
def Mary_income : Prop := M = 1.28 * J

-- Theorem statement to prove
theorem Mary_more_than_Tim (J T M : ℝ) (h1 : Tim_income J T)
  (h2 : Mary_income J M) : ((M - T) / T) * 100 = 60 :=
by
  -- Including sorry to skip the proof
  sorry

end Mary_more_than_Tim_l118_118025


namespace calculate_outlet_requirements_l118_118961

def outlets_needed := 10
def suites_outlets_needed := 15
def num_standard_rooms := 50
def num_suites := 10
def type_a_percentage := 0.40
def type_b_percentage := 0.60
def type_c_percentage := 1.0

noncomputable def total_outlets_needed := 500 + 150
noncomputable def type_a_outlets_needed := 0.40 * 500
noncomputable def type_b_outlets_needed := 0.60 * 500
noncomputable def type_c_outlets_needed := 150

theorem calculate_outlet_requirements :
  total_outlets_needed = 650 ∧
  type_a_outlets_needed = 200 ∧
  type_b_outlets_needed = 300 ∧
  type_c_outlets_needed = 150 :=
by
  sorry

end calculate_outlet_requirements_l118_118961


namespace amy_carl_distance_after_2_hours_l118_118495

-- Conditions
def amy_rate : ℤ := 1
def carl_rate : ℤ := 2
def amy_interval : ℤ := 20
def carl_interval : ℤ := 30
def time_hours : ℤ := 2
def minutes_per_hour : ℤ := 60

-- Derived values
def time_minutes : ℤ := time_hours * minutes_per_hour
def amy_distance : ℤ := time_minutes / amy_interval * amy_rate
def carl_distance : ℤ := time_minutes / carl_interval * carl_rate

-- Question and answer pair
def distance_amy_carl : ℤ := amy_distance + carl_distance
def expected_distance : ℤ := 14

-- The theorem to prove
theorem amy_carl_distance_after_2_hours : distance_amy_carl = expected_distance := by
  sorry

end amy_carl_distance_after_2_hours_l118_118495


namespace company_salary_decrease_l118_118301

variables {E S : ℝ} -- Let the initial number of employees be E and the initial average salary be S

theorem company_salary_decrease :
  (0.8 * E * (1.15 * S)) / (E * S) = 0.92 := 
by
  -- The proof will go here, but we use sorry to skip it for now
  sorry

end company_salary_decrease_l118_118301


namespace find_value_of_xy_plus_yz_plus_xz_l118_118119

variable (x y z : ℝ)

-- Conditions
def cond1 : Prop := x^2 + x * y + y^2 = 108
def cond2 : Prop := y^2 + y * z + z^2 = 64
def cond3 : Prop := z^2 + x * z + x^2 = 172

-- Theorem statement
theorem find_value_of_xy_plus_yz_plus_xz (hx : cond1 x y) (hy : cond2 y z) (hz : cond3 z x) : 
  x * y + y * z + x * z = 96 :=
sorry

end find_value_of_xy_plus_yz_plus_xz_l118_118119


namespace least_common_multiple_of_wang_numbers_l118_118868

noncomputable def wang_numbers (n : ℕ) : List ℕ :=
  -- A function that returns the wang numbers in the set from 1 to n
  sorry

noncomputable def LCM (list : List ℕ) : ℕ :=
  -- A function that computes the least common multiple of a list of natural numbers
  sorry

theorem least_common_multiple_of_wang_numbers :
  LCM (wang_numbers 100) = 10080 :=
sorry

end least_common_multiple_of_wang_numbers_l118_118868


namespace perimeter_rectangle_l118_118214

-- Defining the width and length of the rectangle based on the conditions
def width (a : ℝ) := a
def length (a : ℝ) := 2 * a + 1

-- Statement of the problem: proving the perimeter
theorem perimeter_rectangle (a : ℝ) :
  let W := width a
  let L := length a
  2 * W + 2 * L = 6 * a + 2 :=
by
  sorry

end perimeter_rectangle_l118_118214


namespace simon_age_is_10_l118_118505

def alvin_age : ℕ := 30

def simon_age (alvin_age : ℕ) : ℕ :=
  (alvin_age / 2) - 5

theorem simon_age_is_10 : simon_age alvin_age = 10 := by
  sorry

end simon_age_is_10_l118_118505


namespace students_no_A_in_any_subject_l118_118414

def total_students : ℕ := 50
def a_in_history : ℕ := 9
def a_in_math : ℕ := 15
def a_in_science : ℕ := 12
def a_in_math_and_history : ℕ := 5
def a_in_history_and_science : ℕ := 3
def a_in_science_and_math : ℕ := 4
def a_in_all_three : ℕ := 1

theorem students_no_A_in_any_subject : 
  (total_students - (a_in_history + a_in_math + a_in_science 
                      - a_in_math_and_history - a_in_history_and_science - a_in_science_and_math 
                      + a_in_all_three)) = 28 := by
  sorry

end students_no_A_in_any_subject_l118_118414


namespace point_A_2019_pos_l118_118212

noncomputable def A : ℕ → ℤ
| 0       => 2
| (n + 1) =>
    if (n + 1) % 2 = 1 then A n - (n + 1)
    else A n + (n + 1)

theorem point_A_2019_pos : A 2019 = -1008 := by
  sorry

end point_A_2019_pos_l118_118212


namespace find_weight_of_first_new_player_l118_118045

variable (weight_of_first_new_player : ℕ)
variable (weight_of_second_new_player : ℕ := 60) -- Second new player's weight is a given constant
variable (num_of_original_players : ℕ := 7)
variable (avg_weight_of_original_players : ℕ := 121)
variable (new_avg_weight : ℕ := 113)
variable (num_of_new_players : ℕ := 2)

def total_weight_of_original_players : ℕ := 
  num_of_original_players * avg_weight_of_original_players

def total_weight_of_new_players : ℕ :=
  num_of_new_players * new_avg_weight

def combined_weight_without_first_new_player : ℕ := 
  total_weight_of_original_players + weight_of_second_new_player

def weight_of_first_new_player_proven : Prop :=
  total_weight_of_new_players - combined_weight_without_first_new_player = weight_of_first_new_player

theorem find_weight_of_first_new_player : weight_of_first_new_player = 110 :=
by 
  sorry

end find_weight_of_first_new_player_l118_118045


namespace mean_of_four_numbers_l118_118449

theorem mean_of_four_numbers (a b c d : ℝ) (h : (a + b + c + d + 130) / 5 = 90) : (a + b + c + d) / 4 = 80 := by
  sorry

end mean_of_four_numbers_l118_118449


namespace find_X_l118_118790

theorem find_X (X : ℕ) : 
  (∃ k : ℕ, X = 26 * k + k) ∧ (∃ m : ℕ, X = 29 * m + m) → (X = 270 ∨ X = 540) :=
by
  sorry

end find_X_l118_118790


namespace cannot_form_optionE_l118_118934

-- Define the 4x4 tile
structure Tile4x4 :=
(matrix : Fin 4 → Fin 4 → Bool) -- Boolean to represent black or white

-- Define the condition of alternating rows and columns
def alternating_pattern (tile : Tile4x4) : Prop :=
  (∀ i, tile.matrix i 0 ≠ tile.matrix i 1 ∧
         tile.matrix i 2 ≠ tile.matrix i 3) ∧
  (∀ j, tile.matrix 0 j ≠ tile.matrix 1 j ∧
         tile.matrix 2 j ≠ tile.matrix 3 j)

-- Example tiles for options A, B, C, D, E
def optionA : Tile4x4 := sorry
def optionB : Tile4x4 := sorry
def optionC : Tile4x4 := sorry
def optionD : Tile4x4 := sorry
def optionE : Tile4x4 := sorry

-- Given pieces that can form a 4x4 alternating tile
axiom given_piece1 : Tile4x4
axiom given_piece2 : Tile4x4

-- Combining given pieces to form a 4x4 tile
def combine_pieces (p1 p2 : Tile4x4) : Tile4x4 := sorry -- Combination logic here

-- Proposition stating the problem
theorem cannot_form_optionE :
  (∀ tile, tile = optionA ∨ tile = optionB ∨ tile = optionC ∨ tile = optionD ∨ tile = optionE →
    (tile = optionA ∨ tile = optionB ∨ tile = optionC ∨ tile = optionD → alternating_pattern tile) ∧
    tile = optionE → ¬alternating_pattern tile) :=
sorry

end cannot_form_optionE_l118_118934


namespace inequality_sum_l118_118398

variable {a b c d : ℝ}

theorem inequality_sum (h1 : a > b) (h2 : c > d) : a + c > b + d := 
by {
  sorry
}

end inequality_sum_l118_118398


namespace ratio_Nikki_to_Michael_l118_118872

theorem ratio_Nikki_to_Michael
  (M Joyce Nikki Ryn : ℕ)
  (h1 : Joyce = M + 2)
  (h2 : Nikki = 30)
  (h3 : Ryn = (4 / 5) * Nikki)
  (h4 : M + Joyce + Nikki + Ryn = 76) :
  Nikki / M = 3 :=
by {
  sorry
}

end ratio_Nikki_to_Michael_l118_118872


namespace mathematician_correctness_l118_118908

theorem mathematician_correctness (box1_white1 box1_total1 box2_white1 box2_total1 : ℕ)
                                  (prob1 prob2 : ℚ) :
  (box1_white1 = 4 ∧ box1_total1 = 7 ∧ box2_white1 = 3 ∧ box2_total1 = 5 ∧ prob1 = (4:ℚ) / 7 ∧ prob2 = (3:ℚ) / 5) →
  (box1_white2 = 8 ∧ box1_total2 = 14 ∧ box2_white2 = 3 ∧ box2_total2 = 5 ∧ prob1 = (8:ℚ) / 14 ∧ prob2 = (3:ℚ) / 5) →
  (prob1 < (7:ℚ) / 12 ∧ prob2 > (11:ℚ)/19) →
  ¬((19:ℚ)/35 > (4:ℚ)/7 ∧ (19:ℚ)/35 < (3:ℚ)/5) :=
by
  sorry

end mathematician_correctness_l118_118908


namespace continuity_at_x0_l118_118944

def f (x : ℝ) : ℝ := -5 * x ^ 2 - 8
def x0 : ℝ := 2

theorem continuity_at_x0 :
  ∀ (ε : ℝ), ε > 0 → ∃ (δ : ℝ), δ > 0 ∧ (∀ (x : ℝ), abs (x - x0) < δ → abs (f x - f x0) < ε) :=
by
  sorry

end continuity_at_x0_l118_118944


namespace inequality_solution_l118_118825

theorem inequality_solution (x : ℝ) (h : x ≠ 0) : 
  (1 / (x^2 + 1) > 2 * x^2 / x + 13 / 10) ↔ (x ∈ Set.Ioo (-1.6) 0 ∨ x ∈ Set.Ioi 0.8) :=
by sorry

end inequality_solution_l118_118825


namespace smallest_class_size_l118_118729

theorem smallest_class_size (n : ℕ) (h : 5 * n + 1 > 40) : ∃ k : ℕ, k >= 41 :=
by sorry

end smallest_class_size_l118_118729


namespace simplify_expression_l118_118983

theorem simplify_expression (x : ℝ) : 3 * x^2 - 4 * x^2 = -x^2 :=
by
  sorry

end simplify_expression_l118_118983


namespace condition1_num_registration_methods_condition2_num_registration_methods_condition3_num_registration_methods_l118_118890

-- Definitions corresponding to each condition
def numMethods_participates_in_one_event (students events : ℕ) : ℕ :=
  events ^ students

def numMethods_event_limit_one_person (students events : ℕ) : ℕ :=
  students * (students - 1) * (students - 2)

def numMethods_person_limit_in_events (students events : ℕ) : ℕ :=
  students ^ events

-- Theorems to be proved
theorem condition1_num_registration_methods : 
  numMethods_participates_in_one_event 6 3 = 729 :=
by
  sorry

theorem condition2_num_registration_methods : 
  numMethods_event_limit_one_person 6 3 = 120 :=
by
  sorry

theorem condition3_num_registration_methods : 
  numMethods_person_limit_in_events 6 3 = 216 :=
by
  sorry

end condition1_num_registration_methods_condition2_num_registration_methods_condition3_num_registration_methods_l118_118890


namespace four_digit_numbers_count_l118_118262

theorem four_digit_numbers_count :
  ∃ (n : ℕ),  n = 24 ∧
  let s := λ (a b c d : ℕ), (a + b + c + d = 10 ∧ a ≠ 0 ∧ ((a + c) - (b + d)) ∈ [(-11), 0, 11]) in
  (card {abcd : Fin 10 × Fin 10 × Fin 10 × Fin 10 | (s abcd.1.1 abcd.1.2 abcd.2.1 abcd.2.2)}) = n :=
by
  sorry

end four_digit_numbers_count_l118_118262


namespace correct_option_for_ruler_length_l118_118600

theorem correct_option_for_ruler_length (A B C D : String) (correct_answer : String) : 
  A = "two times as longer as" ∧ 
  B = "twice the length of" ∧ 
  C = "three times longer of" ∧ 
  D = "twice long than" ∧ 
  correct_answer = B := 
by
  sorry

end correct_option_for_ruler_length_l118_118600


namespace custom_op_value_l118_118183

variable {a b : ℤ}
def custom_op (a b : ℤ) := 1/a + 1/b

axiom h1 : a + b = 15
axiom h2 : a * b = 56

theorem custom_op_value : custom_op a b = 15/56 :=
by
  sorry

end custom_op_value_l118_118183


namespace probability_of_sum_23_l118_118506

def is_valid_time (h m : ℕ) : Prop :=
  0 ≤ h ∧ h < 24 ∧ 0 ≤ m ∧ m < 60

def sum_of_digits (n : ℕ) : ℕ :=
  n / 10 + n % 10

def sum_of_time_digits (h m : ℕ) : ℕ :=
  sum_of_digits h + sum_of_digits m

theorem probability_of_sum_23 :
  (∃ h m, is_valid_time h m ∧ sum_of_time_digits h m = 23) →
  (4 / 1440 : ℚ) = (1 / 360 : ℚ) :=
by
  sorry

end probability_of_sum_23_l118_118506


namespace determinant_positive_l118_118874

variable {n : Type*} [Fintype n] [DecidableEq n]
variable (A : Matrix n n ℝ)

theorem determinant_positive (h : A + A.transpose = 1) : det A > 0 := sorry

end determinant_positive_l118_118874


namespace find_sum_3xyz_l118_118568

variables (x y z : ℚ)

def equation1 : Prop := y + z = 18 - 4 * x
def equation2 : Prop := x + z = 16 - 4 * y
def equation3 : Prop := x + y = 9 - 4 * z

theorem find_sum_3xyz (h1 : equation1 x y z) (h2 : equation2 x y z) (h3 : equation3 x y z) : 
  3 * x + 3 * y + 3 * z = 43 / 2 := 
sorry

end find_sum_3xyz_l118_118568


namespace problem1_problem2_l118_118947

-- Problem 1
theorem problem1 : (1 / 2) ^ (-2 : ℤ) - (Real.pi - Real.sqrt 5) ^ 0 - Real.sqrt 20 = 3 - 2 * Real.sqrt 5 :=
by sorry

-- Problem 2
theorem problem2 (x : ℝ) (hx1 : x ≠ 0) (hx2 : x ≠ -1) : 
  ((x ^ 2 - 2 * x + 1) / (x ^ 2 - 1)) / ((x - 1) / (x ^ 2 + x)) = x :=
by sorry

end problem1_problem2_l118_118947


namespace eval_expression_l118_118096

theorem eval_expression (x y : ℕ) (h_x : x = 2001) (h_y : y = 2002) :
  (x^3 - 3*x^2*y + 5*x*y^2 - y^3 - 2) / (x * y) = 1999 :=
  sorry

end eval_expression_l118_118096


namespace number_of_satisfying_ns_l118_118545

noncomputable def a_n (n : ℕ) : ℕ := (n-1)*(2*n-1)

def b_n (n : ℕ) : ℕ := 2^n * n

def condition (n : ℕ) : Prop := b_n n ≤ 2019 * a_n n

theorem number_of_satisfying_ns : 
  ∃ n : ℕ, n = 14 ∧ ∀ k : ℕ, (1 ≤ k ∧ k ≤ 14) → condition k := 
by
  sorry

end number_of_satisfying_ns_l118_118545


namespace perpendicular_tangents_add_l118_118747

open Real

noncomputable def f1 (x : ℝ): ℝ := x^2 - 2 * x + 2
noncomputable def f2 (x : ℝ) (a : ℝ) (b : ℝ): ℝ := -x^2 + a * x + b

-- Definitions of derivatives for the given functions
noncomputable def f1' (x : ℝ): ℝ := 2 * x - 2
noncomputable def f2' (x : ℝ) (a : ℝ): ℝ := -2 * x + a

theorem perpendicular_tangents_add (x0 y0 a b : ℝ)
  (h1 : y0 = f1 x0)
  (h2 : y0 = f2 x0 a b)
  (h3 : f1' x0 * f2' x0 a = -1) :
  a + b = 5 / 2 := sorry

end perpendicular_tangents_add_l118_118747


namespace four_digit_sum_ten_divisible_by_eleven_l118_118260

theorem four_digit_sum_ten_divisible_by_eleven : 
  {n | (10^3 ≤ n ∧ n < 10^4) ∧ (∑ i in (Int.toString n).toList, i.toNat) = 10 ∧ (let digits := (Int.toString n).toList.map (λ c, c.toNat) in ((digits.nth 0).getD 0 + (digits.nth 2).getD 0) - ((digits.nth 1).getD 0 + (digits.nth 3).getD 0)) % 11 == 0}.card = 30 := sorry

end four_digit_sum_ten_divisible_by_eleven_l118_118260


namespace area_excluding_hole_l118_118795

def area_large_rectangle (x : ℝ) : ℝ :=
  (2 * x + 9) * (x + 6)

def area_square_hole (x : ℝ) : ℝ :=
  (x - 1) * (x - 1)

theorem area_excluding_hole (x : ℝ) : 
  area_large_rectangle x - area_square_hole x = x^2 + 23 * x + 53 :=
by
  sorry

end area_excluding_hole_l118_118795


namespace find_radius_of_sphere_l118_118603

noncomputable def radius_of_sphere : ℝ :=
  let r1 := 1
  let r2 := 2
  let r3 := 3
  let h := λ (x: ℝ), x -- Equivalent heights of the cones
  let d12 := 3
  let d13 := 4
  let d23 := 5
  let R := 1
  R

theorem find_radius_of_sphere :
  let r1 := 1
  let r2 := 2
  let r3 := 3
  let h := λ (x: ℝ), x -- Equivalent heights of the cones
  let d12 := 3
  let d13 := 4
  let d23 := 5
  ∃ R: ℝ, R = 1 :=
by {
  -- skipping the proof
  sorry,
}

end find_radius_of_sphere_l118_118603


namespace factor_x10_minus_1296_l118_118365

theorem factor_x10_minus_1296 (x : ℝ) : (x^10 - 1296) = (x^5 + 36) * (x^5 - 36) :=
  by
  sorry

end factor_x10_minus_1296_l118_118365


namespace intersection_of_sets_l118_118846

def SetA : Set ℝ := {x | 0 < x ∧ x < 3}
def SetB : Set ℝ := {x | x > 2}
def SetC : Set ℝ := {x | 2 < x ∧ x < 3}

theorem intersection_of_sets :
  SetA ∩ SetB = SetC :=
by
  sorry

end intersection_of_sets_l118_118846


namespace complex_fraction_simplification_l118_118360

open Complex

theorem complex_fraction_simplification : (1 + 2 * I) / (1 - I)^2 = 1 - 1 / 2 * I :=
by
  -- Proof omitted
  sorry

end complex_fraction_simplification_l118_118360


namespace find_5_minus_c_l118_118860

theorem find_5_minus_c (c d : ℤ) (h₁ : 5 + c = 6 - d) (h₂ : 3 + d = 8 + c) : 5 - c = 7 := by
  sorry

end find_5_minus_c_l118_118860


namespace parallel_lines_l118_118810

def line1 (x : ℝ) : ℝ := 5 * x + 3
def line2 (x k : ℝ) : ℝ := 3 * k * x + 7

theorem parallel_lines (k : ℝ) : (∀ x : ℝ, line1 x = line2 x k) → k = 5 / 3 := 
by
  intros h_parallel
  sorry

end parallel_lines_l118_118810


namespace sophia_pages_difference_l118_118617

theorem sophia_pages_difference (total_pages : ℕ) (f_fraction : ℚ) (l_fraction : ℚ) 
  (finished_pages : ℕ) (left_pages : ℕ) :
  f_fraction = 2/3 ∧ 
  l_fraction = 1/3 ∧
  total_pages = 270 ∧
  finished_pages = f_fraction * total_pages ∧
  left_pages = l_fraction * total_pages
  →
  finished_pages - left_pages = 90 :=
by
  intro h
  sorry

end sophia_pages_difference_l118_118617


namespace problem_statement_l118_118269

theorem problem_statement (p q : ℚ) (hp : 3 / p = 4) (hq : 3 / q = 18) : p - q = 7 / 12 := 
by
  sorry

end problem_statement_l118_118269


namespace probability_check_l118_118092

def total_students : ℕ := 12

def total_clubs : ℕ := 3

def equiprobable_clubs := ∀ s : Fin total_students, ∃ c : Fin total_clubs, true

noncomputable def probability_diff_students : ℝ := 1 - (34650 / (total_clubs ^ total_students))

theorem probability_check :
  equiprobable_clubs →
  probability_diff_students = 0.935 := 
by
  intros
  sorry

end probability_check_l118_118092


namespace cloud_ratio_l118_118363

theorem cloud_ratio (D Carson Total : ℕ) (h1 : Carson = 6) (h2 : Total = 24) (h3 : Carson + D = Total) :
  (D / Carson) = 3 := by
  sorry

end cloud_ratio_l118_118363


namespace cost_of_paving_l118_118177

-- Definitions based on the given conditions
def length : ℝ := 6.5
def width : ℝ := 2.75
def rate : ℝ := 600

-- Theorem statement to prove the cost of paving
theorem cost_of_paving : length * width * rate = 10725 := by
  -- Calculation steps would go here, but we omit them with sorry
  sorry

end cost_of_paving_l118_118177


namespace sum_of_integers_l118_118044

theorem sum_of_integers (m n p q : ℤ) 
(h1 : m ≠ n) (h2 : m ≠ p) 
(h3 : m ≠ q) (h4 : n ≠ p) 
(h5 : n ≠ q) (h6 : p ≠ q) 
(h7 : (5 - m) * (5 - n) * (5 - p) * (5 - q) = 9) : 
m + n + p + q = 20 :=
by
  sorry

end sum_of_integers_l118_118044


namespace lying_dwarf_number_is_possible_l118_118940

def dwarfs_sum (a1 a2 a3 a4 a5 a6 a7 : ℕ) : Prop :=
  a2 = a1 ∧
  a3 = a1 + a2 ∧
  a4 = a1 + a2 + a3 ∧
  a5 = a1 + a2 + a3 + a4 ∧
  a6 = a1 + a2 + a3 + a4 + a5 ∧
  a7 = a1 + a2 + a3 + a4 + a5 + a6 ∧
  a1 + a2 + a3 + a4 + a5 + a6 + a7 = 58

theorem lying_dwarf_number_is_possible (a1 a2 a3 a4 a5 a6 a7 : ℕ) :
  dwarfs_sum a1 a2 a3 a4 a5 a6 a7 →
  (a1 = 13 ∨ a1 = 26) :=
sorry

end lying_dwarf_number_is_possible_l118_118940


namespace determine_x_l118_118647

variable (a b c d x : ℝ)
variable (h1 : (a^2 + x)/(b^2 + x) = c/d)
variable (h2 : a ≠ b)
variable (h3 : b ≠ 0)
variable (h4 : d ≠ c) -- added condition from solution step

theorem determine_x : x = (a^2 * d - b^2 * c) / (c - d) := by
  sorry

end determine_x_l118_118647


namespace solve_for_x_l118_118691

theorem solve_for_x (x : ℝ) (h₀ : x > 0) (h₁ : 1 / 2 * x * (3 * x) = 96) : x = 8 :=
sorry

end solve_for_x_l118_118691


namespace price_of_peas_l118_118076

theorem price_of_peas
  (P : ℝ) -- price of peas per kg in rupees
  (price_soybeans : ℝ) (price_mixture : ℝ)
  (ratio_peas_soybeans : ℝ) :
  price_soybeans = 25 →
  price_mixture = 19 →
  ratio_peas_soybeans = 2 →
  P = 16 :=
by
  intros h_price_soybeans h_price_mixture h_ratio
  sorry

end price_of_peas_l118_118076


namespace pufferfish_count_l118_118920

theorem pufferfish_count (s p : ℕ) (h1 : s = 5 * p) (h2 : s + p = 90) : p = 15 :=
by
  sorry

end pufferfish_count_l118_118920


namespace arithmetic_geometric_sequence_general_term_l118_118036

theorem arithmetic_geometric_sequence_general_term :
  ∃ q a1 : ℕ, (∀ n : ℕ, a2 = 6 ∧ 6 * a1 + a3 = 30) →
  (∀ n : ℕ, (q = 2 ∧ a1 = 3 → a_n = 3 * 3^(n-1)) ∨ (q = 3 ∧ a1 = 2 → a_n = 2 * 2^(n-1))) :=
by
  sorry

end arithmetic_geometric_sequence_general_term_l118_118036


namespace normal_probability_l118_118882

noncomputable def ξ : MeasureTheory.ProbabilityMeasure ℝ :=
  MeasureTheory.ProbabilityMeasure.ofReal (probabilityMeasureOfNormal 1 σ²)

theorem normal_probability ξ : 
  P(ξ < 1) = 1/2 → P(ξ > 2) = p → P(0 < ξ < 1) = 1/2 - p :=
by
  intros h1 h2
  sorry

end normal_probability_l118_118882


namespace parallel_lines_l118_118808

def line1 (x : ℝ) : ℝ := 5 * x + 3
def line2 (x k : ℝ) : ℝ := 3 * k * x + 7

theorem parallel_lines (k : ℝ) : (∀ x : ℝ, line1 x = line2 x k) → k = 5 / 3 := 
by
  intros h_parallel
  sorry

end parallel_lines_l118_118808


namespace divisors_not_divisible_by_3_eq_6_l118_118406

theorem divisors_not_divisible_by_3_eq_6 :
  let n := 180 in
  let prime_factorization := (2^2 * 3^2 * 5^1) in
  (∀ d ∣ n, ¬ (3 ∣ d) → (∃! (a b c : ℕ), d = 2^a * 3^b * 5^c ∧ a ≤ 2 ∧ b = 0 ∧ c ≤ 1)) →
  (finset.card {d ∣ n | ¬ (3 ∣ d)} = 6) :=
by
  let n := 180
  let prime_factorization := (2^2 * 3^2 * 5^1)
  have h : prime_factorization = n := by norm_num
  let s := finset.filter (λ d, ¬3 ∣ d) (finset.divisors n)
  suffices : finset.card s = 6
  { exact this }
  sorry

end divisors_not_divisible_by_3_eq_6_l118_118406


namespace average_age_of_team_l118_118651

/--
The captain of a cricket team of 11 members is 26 years old and the wicket keeper is 
3 years older. If the ages of these two are excluded, the average age of the remaining 
players is one year less than the average age of the whole team. Prove that the average 
age of the whole team is 32 years.
-/
theorem average_age_of_team 
  (captain_age : Nat) (wicket_keeper_age : Nat) (remaining_9_average_age : Nat)
  (team_size : Nat) (total_team_age : Nat) (remaining_9_total_age : Nat)
  (A : Nat) :
  captain_age = 26 →
  wicket_keeper_age = captain_age + 3 →
  team_size = 11 →
  total_team_age = team_size * A →
  total_team_age = remaining_9_total_age + captain_age + wicket_keeper_age →
  remaining_9_total_age = 9 * (A - 1) →
  A = 32 :=
by
  sorry

end average_age_of_team_l118_118651


namespace abs_eq_abs_of_unique_solution_l118_118300

variables {a b : ℝ} (ha : a ≠ 0) (hb : b ≠ 0)
theorem abs_eq_abs_of_unique_solution
  (h : ∃ x : ℝ, ∀ y : ℝ, a * (y - a)^2 + b * (y - b)^2 = 0 ↔ y = x) :
  |a| = |b| :=
sorry

end abs_eq_abs_of_unique_solution_l118_118300


namespace set_C_is_basis_l118_118355

variables (e1 e2 : ℝ × ℝ)

def is_basis_set_C :=
  e1 = (1, -2) ∧ e2 = (2, 3) ∧ 
  (∀ (k : ℝ), e2 ≠ k • e1) ∧ 
  e1 ≠ (0, 0) ∧ e2 ≠ (0, 0)

theorem set_C_is_basis (e1 e2 : ℝ × ℝ) : 
  is_basis_set_C e1 e2 :=
by 
  sorry

end set_C_is_basis_l118_118355


namespace find_line_equation_l118_118385

-- Definitions: Point and Line in 2D
structure Point2D where
  x : ℝ
  y : ℝ

structure Line2D where
  a : ℝ
  b : ℝ
  c : ℝ

-- Line passes through the point
def line_through_point (L : Line2D) (P : Point2D) : Prop :=
  L.a * P.x + L.b * P.y + L.c = 0

-- Perpendicular lines condition: if Line L1 and Line L2 are perpendicular.
def perpendicular (L1 L2 : Line2D) : Prop :=
  L1.a * L2.a + L1.b * L2.b = 0

-- Define line1 and line2 as given
def line1 : Line2D := {a := 1, b := -2, c := 0} -- corresponds to x - 2y + m = 0

-- Define point P (-1, 3)
def P : Point2D := {x := -1, y := 3}

-- Required line passing through point P and perpendicular to line1
def required_line : Line2D := {a := 2, b := 1, c := -1}

-- The proof goal
theorem find_line_equation : (line_through_point required_line P) ∧ (perpendicular line1 required_line) :=
by
  sorry

end find_line_equation_l118_118385


namespace count_integers_in_interval_l118_118123

theorem count_integers_in_interval : 
  ∃ (k : ℤ), k = 46 ∧ 
  (∀ n : ℤ, -5 * (2.718 : ℝ) ≤ (n : ℝ) ∧ (n : ℝ) ≤ 12 * (2.718 : ℝ) → (-13 ≤ n ∧ n ≤ 32)) ∧ 
  (∀ n : ℤ, -13 ≤ n ∧ n ≤ 32 → -5 * (2.718 : ℝ) ≤ (n : ℝ) ∧ (n : ℝ) ≤ 12 * (2.718 : ℝ)) :=
sorry

end count_integers_in_interval_l118_118123


namespace no_solution_for_lcm_gcd_eq_l118_118194

theorem no_solution_for_lcm_gcd_eq (n : ℕ) (h₁ : n ∣ 60) (h₂ : Nat.Prime n) :
  ¬(Nat.lcm n 60 = Nat.gcd n 60 + 200) :=
  sorry

end no_solution_for_lcm_gcd_eq_l118_118194


namespace inequality_division_l118_118392

variable {a b c : ℝ}

theorem inequality_division (h1 : a > b) (h2 : b > 0) (h3 : c < 0) : 
  (a / (a - c)) > (b / (b - c)) := 
sorry

end inequality_division_l118_118392


namespace solve_for_x_l118_118951

theorem solve_for_x :
  ∀ x : ℤ, 3 * x + 36 = 48 → x = 4 :=
by
  sorry

end solve_for_x_l118_118951


namespace integer_add_results_in_perfect_square_l118_118467

theorem integer_add_results_in_perfect_square (x a b : ℤ) :
  (x + 100 = a^2 ∧ x + 164 = b^2) → (x = 125 ∨ x = -64 ∨ x = -100) :=
by
  intros h
  sorry

end integer_add_results_in_perfect_square_l118_118467


namespace min_product_ab_l118_118686

theorem min_product_ab (a b : ℝ) (h : 20 * a * b = 13 * a + 14 * b) (h_pos_a : 0 < a) (h_pos_b : 0 < b) : 
  a * b = 1.82 :=
sorry

end min_product_ab_l118_118686


namespace inequality_three_integer_solutions_l118_118685

theorem inequality_three_integer_solutions (c : ℤ) :
  (∃ s1 s2 s3 : ℤ, s1 < s2 ∧ s2 < s3 ∧ 
    (∀ x : ℤ, x^2 + c * x + 1 ≤ 0 ↔ x = s1 ∨ x = s2 ∨ x = s3)) ↔ (c = -4 ∨ c = 4) := 
by 
  sorry

end inequality_three_integer_solutions_l118_118685


namespace binomial_expansion_fraction_l118_118109

theorem binomial_expansion_fraction :
  let a0 := 32
  let a1 := -80
  let a2 := 80
  let a3 := -40
  let a4 := 10
  let a5 := -1
  (2 - x)^5 = a0 + a1 * x + a2 * x^2 + a3 * x^3 + a4 * x^4 + a5 * x^5 →
  (a0 + a2 + a4) / (a1 + a3) = -61 / 60 :=
by
  sorry

end binomial_expansion_fraction_l118_118109


namespace area_perimeter_ratio_eq_l118_118192

theorem area_perimeter_ratio_eq (s : ℝ) (s_eq : s = 10) : 
  let area := (sqrt 3) / 4 * s ^ 2
      perimeter := 3 * s
      ratio := area / (perimeter ^ 2)
  in ratio = (sqrt 3) / 36 :=
by sorry

end area_perimeter_ratio_eq_l118_118192


namespace find_locus_of_T_l118_118245

section Locus

variables {x y m : ℝ}
variable (M : ℝ × ℝ)

-- Condition: The equation of the ellipse
def ellipse (x y : ℝ) := (x^2 / 4) + y^2 = 1

-- Condition: Point P
def P := (1, 0)

-- Condition: M is any point on the ellipse, except A and B
def on_ellipse (M : ℝ × ℝ) := ellipse M.1 M.2 ∧ M ≠ (-2, 0) ∧ M ≠ (2, 0)

-- Condition: The intersection point N of line MP with the ellipse
def line_eq (m y : ℝ) := m * y + 1

-- Proposition: Locus of intersection point T of lines AM and BN
theorem find_locus_of_T 
  (hM : on_ellipse M)
  (hN : line_eq m M.2 = M.1)
  (hT : M.2 ≠ 0) :
  M.1 = 4 :=
sorry

end Locus

end find_locus_of_T_l118_118245


namespace students_taking_both_chorus_and_band_l118_118728

theorem students_taking_both_chorus_and_band (total_students : ℕ) 
                                             (chorus_students : ℕ)
                                             (band_students : ℕ)
                                             (not_enrolled_students : ℕ) : 
                                             total_students = 50 ∧
                                             chorus_students = 18 ∧
                                             band_students = 26 ∧
                                             not_enrolled_students = 8 →
                                             ∃ (both_chorus_and_band : ℕ), both_chorus_and_band = 2 :=
by
  intros h
  sorry

end students_taking_both_chorus_and_band_l118_118728


namespace electric_guitar_count_l118_118725

theorem electric_guitar_count (E A : ℤ) (h1 : E + A = 9) (h2 : 479 * E + 339 * A = 3611) (hE_nonneg : E ≥ 0) (hA_nonneg : A ≥ 0) : E = 4 :=
by
  sorry

end electric_guitar_count_l118_118725


namespace find_xyz_l118_118426

open Complex

theorem find_xyz (a b c x y z : ℂ)
(h1 : a ≠ 0) (h2 : b ≠ 0) (h3 : c ≠ 0) (h4 : x ≠ 0) (h5 : y ≠ 0) (h6 : z ≠ 0)
(h7 : a = (b + c) / (x - 3)) (h8 : b = (a + c) / (y - 3)) (h9 : c = (a + b) / (z - 3))
(h10 : x * y + x * z + y * z = 10) (h11 : x + y + z = 6) : 
(x * y * z = 15) :=
by
  sorry

end find_xyz_l118_118426


namespace smallest_k_sum_sequence_l118_118657

theorem smallest_k_sum_sequence (n : ℕ) (h : 100 = (n + 1) * (2 * 9 + n) / 2) : k = 9 := 
sorry

end smallest_k_sum_sequence_l118_118657


namespace digit_a_solution_l118_118930

theorem digit_a_solution :
  ∃ a : ℕ, a000 + a998 + a999 = 22997 → a = 7 :=
sorry

end digit_a_solution_l118_118930


namespace product_mod_five_remainder_l118_118388

theorem product_mod_five_remainder :
  (114 * 232 * 454 * 454 * 678) % 5 = 4 := by
  sorry

end product_mod_five_remainder_l118_118388


namespace problem_eq_solution_l118_118742

variables (a b x y : ℝ)

theorem problem_eq_solution
  (h1 : 0 < a) (h2 : 0 < b) (h3 : 0 < x) (h4 : 0 < y)
  (h5 : a + b + x + y < 2)
  (h6 : a + b^2 = x + y^2)
  (h7 : a^2 + b = x^2 + y) :
  a = x ∧ b = y :=
by
  sorry

end problem_eq_solution_l118_118742


namespace sum_of_digits_of_special_number_l118_118492

theorem sum_of_digits_of_special_number :
  ∀ (x y z : ℕ), 1 ≤ x ∧ x ≤ 9 ∧ 0 ≤ y ∧ y ≤ 9 ∧ 0 ≤ z ∧ z ≤ 9 ∧ (100 * x + 10 * y + z = x.factorial + y.factorial + z.factorial) →
  (x + y + z = 10) :=
by
  sorry

end sum_of_digits_of_special_number_l118_118492


namespace ratio_kittens_to_breeding_rabbits_l118_118431

def breeding_rabbits : ℕ := 10
def kittens_first_spring (k : ℕ) : ℕ := k * breeding_rabbits
def adopted_kittens_first_spring (k : ℕ) : ℕ := 5 * k
def returned_kittens : ℕ := 5
def remaining_kittens_first_spring (k : ℕ) : ℕ := (k * breeding_rabbits) / 2 + returned_kittens

def kittens_second_spring : ℕ := 60
def adopted_kittens_second_spring : ℕ := 4
def remaining_kittens_second_spring : ℕ := kittens_second_spring - adopted_kittens_second_spring

def total_rabbits (k : ℕ) : ℕ := 
  breeding_rabbits + remaining_kittens_first_spring k + remaining_kittens_second_spring

theorem ratio_kittens_to_breeding_rabbits (k : ℕ) (h : total_rabbits k = 121) :
  k = 10 :=
sorry

end ratio_kittens_to_breeding_rabbits_l118_118431


namespace isosceles_triangle_time_between_9_30_and_10_l118_118223

theorem isosceles_triangle_time_between_9_30_and_10 (time : ℕ) (h_time_range : 30 ≤ time ∧ time < 60)
  (h_isosceles : ∃ x : ℝ, 0 ≤ x ∧ x + 2 * x + 2 * x = 180) :
  time = 36 :=
  sorry

end isosceles_triangle_time_between_9_30_and_10_l118_118223


namespace student_passing_percentage_l118_118078

def student_marks : ℕ := 80
def shortfall_marks : ℕ := 100
def total_marks : ℕ := 600

def passing_percentage (student_marks shortfall_marks total_marks : ℕ) : ℕ :=
  (student_marks + shortfall_marks) * 100 / total_marks

theorem student_passing_percentage :
  passing_percentage student_marks shortfall_marks total_marks = 30 :=
by
  sorry

end student_passing_percentage_l118_118078


namespace correlation_non_deterministic_relationship_l118_118933

theorem correlation_non_deterministic_relationship
  (independent_var_fixed : Prop)
  (dependent_var_random : Prop)
  (correlation_def : Prop)
  (correlation_randomness : Prop) :
  (correlation_def → non_deterministic) :=
by
  sorry

end correlation_non_deterministic_relationship_l118_118933


namespace draw_probability_l118_118932

-- Define probabilities
def p_jian_wins : ℝ := 0.4
def p_gu_not_wins : ℝ := 0.6

-- Define the probability of the game ending in a draw
def p_draw : ℝ := p_gu_not_wins - p_jian_wins

-- State the theorem to be proved
theorem draw_probability : p_draw = 0.2 :=
by
  sorry

end draw_probability_l118_118932


namespace inverse_matrix_l118_118539

-- Define the matrix A
def A : Matrix (Fin 2) (Fin 2) ℤ :=
  ![![7, -2], ![-3, 1]]

-- Define the supposed inverse B
def B : Matrix (Fin 2) (Fin 2) ℤ :=
  ![![1, 2], ![3, 7]]

-- Define the condition that A * B should yield the identity matrix
theorem inverse_matrix :
  A * B = 1 :=
sorry

end inverse_matrix_l118_118539


namespace value_of_M_l118_118268

theorem value_of_M (M : ℝ) (h : 0.2 * M = 500) : M = 2500 :=
by
  sorry

end value_of_M_l118_118268


namespace passenger_catches_bus_l118_118222

-- Definitions based on conditions from part a)
def P_route3 := 0.20
def P_route6 := 0.60

-- Statement to prove based on part c)
theorem passenger_catches_bus : 
  P_route3 + P_route6 = 0.80 := 
by
  sorry

end passenger_catches_bus_l118_118222


namespace correct_equation_l118_118516

def initial_count_A : ℕ := 54
def initial_count_B : ℕ := 48
def new_count_A (x : ℕ) : ℕ := initial_count_A + x
def new_count_B (x : ℕ) : ℕ := initial_count_B - x

theorem correct_equation (x : ℕ) : new_count_A x = 2 * new_count_B x := 
sorry

end correct_equation_l118_118516


namespace equal_binomial_terms_l118_118322

theorem equal_binomial_terms (p q : ℝ) (h1 : 0 < p) (h2 : 0 < q) (h3 : p + q = 1)
    (h4 : 55 * p^9 * q^2 = 165 * p^8 * q^3) : p = 3 / 4 :=
by
  sorry

end equal_binomial_terms_l118_118322


namespace min_students_with_both_l118_118299

-- Given conditions
def total_students : ℕ := 35
def students_with_brown_eyes : ℕ := 18
def students_with_lunch_box : ℕ := 25

-- Mathematical statement to prove the least number of students with both attributes
theorem min_students_with_both :
  ∃ x : ℕ, students_with_brown_eyes + students_with_lunch_box - total_students ≤ x ∧ x = 8 :=
sorry

end min_students_with_both_l118_118299


namespace divisors_of_180_not_divisible_by_3_l118_118404

def is_divisor (n d : ℕ) := d ∣ n

def prime_factors (n : ℕ) :=
  if n = 180 then [2, 3, 5] else []

def divisor_not_divisible_by_3 (n d : ℕ) :=
  is_divisor n d ∧ ¬ (3 ∣ d)

def number_of_divisors_not_divisible_by_3 (n : ℕ) :=
  if n = 180 then 6 else 0

theorem divisors_of_180_not_divisible_by_3 : 
  ∀ n, n = 180 → (∑ d in (List.filter (divisor_not_divisible_by_3 n) (List.range (n+1))), 1) = 6 := by
  sorry

end divisors_of_180_not_divisible_by_3_l118_118404


namespace travelers_cross_river_l118_118924

variables (traveler1 traveler2 traveler3 : ℕ)  -- weights of travelers
variable (raft_capacity : ℕ)  -- maximum carrying capacity of the raft

-- Given conditions
def conditions :=
  traveler1 = 3 ∧ traveler2 = 3 ∧ traveler3 = 5 ∧ raft_capacity = 7

-- Prove that the travelers can all cross the river successfully
theorem travelers_cross_river :
  conditions traveler1 traveler2 traveler3 raft_capacity →
  (traveler1 + traveler2 ≤ raft_capacity) ∧
  (traveler1 ≤ raft_capacity) ∧
  (traveler3 ≤ raft_capacity) ∧
  (traveler1 + traveler2 ≤ raft_capacity) →
  true :=
by
  intros h_conditions h_validity
  sorry

end travelers_cross_river_l118_118924


namespace maximize_profit_l118_118202

-- Conditions
def price_bound (p : ℝ) := p ≤ 22
def books_sold (p : ℝ) := 110 - 4 * p
def profit (p : ℝ) := (p - 2) * books_sold p

-- The main theorem statement
theorem maximize_profit : ∃ p : ℝ, price_bound p ∧ profit p = profit 15 :=
sorry

end maximize_profit_l118_118202


namespace vacation_fund_percentage_l118_118233

variable (s : ℝ) (vs : ℝ)
variable (d : ℝ)
variable (v : ℝ)

-- conditions:
-- 1. Jill's net monthly salary
#check (s = 3700)
-- 2. Jill's discretionary income is one fifth of her salary
#check (d = s / 5)
-- 3. Savings percentage
#check (0.20 * d)
-- 4. Eating out and socializing percentage
#check (0.35 * d)
-- 5. Gifts and charitable causes
#check (111)

-- Prove: 
theorem vacation_fund_percentage : 
  s = 3700 -> d = s / 5 -> 
  (v * d + 0.20 * d + 0.35 * d + 111 = d) -> 
  v = 222 / 740 :=
by
  sorry -- proof skipped

end vacation_fund_percentage_l118_118233


namespace john_newspapers_l118_118871

theorem john_newspapers (N : ℕ) (selling_price buying_price total_cost total_revenue : ℝ) 
  (h1 : selling_price = 2)
  (h2 : buying_price = 0.25 * selling_price)
  (h3 : total_cost = N * buying_price)
  (h4 : total_revenue = 0.8 * N * selling_price)
  (h5 : total_revenue - total_cost = 550) :
  N = 500 := 
by 
  -- actual proof here
  sorry

end john_newspapers_l118_118871


namespace percent_of_dollar_in_pocket_l118_118641

def penny_value : ℕ := 1
def nickel_value : ℕ := 5
def dime_value : ℕ := 10
def quarter_value : ℕ := 25
def half_dollar_value : ℕ := 50

theorem percent_of_dollar_in_pocket :
  let total_cents := penny_value + nickel_value + dime_value + quarter_value + half_dollar_value
  total_cents = 91 := by
  sorry

end percent_of_dollar_in_pocket_l118_118641


namespace ramsey_K9_blue_K4_or_red_K3_l118_118093

theorem ramsey_K9_blue_K4_or_red_K3 (G : SimpleGraph (fin 9)) (hG : G.IsCompleteGraph) (color : G.Edge → Prop) :
  (∃ (V : finset (fin 9)), V.card = 4 ∧ ∀ (u v : fin 9) (hu : u ∈ V) (hv : v ∈ V), u ≠ v → color (u, v)) ∨
  (∃ (U : finset (fin 9)), U.card = 3 ∧ ∀ (u v : fin 9) (hu : u ∈ U) (hv : v ∈ U), u ≠ v → ¬color (u, v)) :=
sorry

end ramsey_K9_blue_K4_or_red_K3_l118_118093


namespace inequality_f_n_l118_118151

theorem inequality_f_n {f : ℕ → ℕ} {k : ℕ} (strict_mono_f : ∀ {a b : ℕ}, a < b → f a < f b)
  (h_f : ∀ n : ℕ, f (f n) = k * n) : ∀ n : ℕ, 
  (2 * k * n) / (k + 1) ≤ f n ∧ f n ≤ ((k + 1) * n) / 2 :=
by
  sorry

end inequality_f_n_l118_118151


namespace proof_two_digit_number_l118_118969

noncomputable def two_digit_number := {n : ℤ // 10 ≤ n ∧ n ≤ 99}

theorem proof_two_digit_number (n : two_digit_number) :
  (n.val % 2 = 0) ∧ 
  ((n.val + 1) % 3 = 0) ∧
  ((n.val + 2) % 4 = 0) ∧
  ((n.val + 3) % 5 = 0) →
  n.val = 62 :=
by sorry

end proof_two_digit_number_l118_118969


namespace trip_first_part_distance_l118_118962

theorem trip_first_part_distance (x : ℝ) :
  let total_distance : ℝ := 60
  let speed_first : ℝ := 48
  let speed_remaining : ℝ := 24
  let avg_speed : ℝ := 32
  (x / speed_first + (total_distance - x) / speed_remaining = total_distance / avg_speed) ↔ (x = 30) :=
by sorry

end trip_first_part_distance_l118_118962


namespace intersection_of_multiples_of_2_l118_118148

theorem intersection_of_multiples_of_2 : 
  let M := {1, 2, 4, 8}
  let N := {x : ℤ | ∃ k : ℤ, x = 2 * k}
  M ∩ N = {2, 4, 8} :=
by
  sorry

end intersection_of_multiples_of_2_l118_118148


namespace simplify_expression_l118_118166

variable (a b : ℤ) -- Define variables a and b

theorem simplify_expression : 
  (35 * a + 70 * b + 15) + (15 * a + 54 * b + 5) - (20 * a + 85 * b + 10) =
  30 * a + 39 * b + 10 := 
by sorry

end simplify_expression_l118_118166


namespace solve_for_x_l118_118689

theorem solve_for_x (x : ℝ) :
  5 * (x - 9) = 7 * (3 - 3 * x) + 10 → x = 38 / 13 :=
by
  intro h
  sorry

end solve_for_x_l118_118689


namespace sum_of_numbers_l118_118469

theorem sum_of_numbers : 1234 + 2341 + 3412 + 4123 = 11110 := by
  sorry

end sum_of_numbers_l118_118469


namespace ratio_and_tangent_l118_118116

-- Definitions for the problem
def acute_triangle (A B C : Point) : Prop := 
  -- acute angles condition
  sorry

def is_diameter (A B C D : Point) : Prop := 
  -- D is midpoint of BC condition
  sorry

def divide_in_half (A B C : Point) (D : Point) : Prop := 
  -- D divides BC in half condition
  sorry

def divide_in_ratio (A B C : Point) (D : Point) (ratio : ℚ) : Prop := 
  -- D divides AC in the given ratio condition
  sorry

def tan (angle : ℝ) : ℝ := 
  -- Tangent function
  sorry

def angle (A B C : Point) : ℝ := 
  -- Angle at B of triangle ABC
  sorry

-- The statement of the problem in Lean
theorem ratio_and_tangent (A B C D : Point) :
  acute_triangle A B C →
  is_diameter A B C D →
  divide_in_half A B C D →
  (divide_in_ratio A B C D (1 / 3) ↔ tan (angle A B C) = 2 * tan (angle A C B)) :=
by sorry

end ratio_and_tangent_l118_118116


namespace quadratic_minimum_eq_one_l118_118854

variable (p q : ℝ)

theorem quadratic_minimum_eq_one (hq : q = 1 + p^2 / 18) : 
  ∃ x : ℝ, 3 * x^2 + p * x + q = 1 :=
by
  sorry

end quadratic_minimum_eq_one_l118_118854


namespace trapezoid_circumscribed_radius_l118_118172

theorem trapezoid_circumscribed_radius 
  (a b : ℝ) 
  (height : ℝ)
  (ratio_ab : a / b = 5 / 12)
  (height_eq_midsegment : height = 17) :
  ∃ r : ℝ, r = 13 :=
by
  -- Assuming conditions directly as given
  have h1 : a / b = 5 / 12 := ratio_ab
  have h2 : height = 17 := height_eq_midsegment
  -- The rest of the proof goes here
  sorry

end trapezoid_circumscribed_radius_l118_118172


namespace cody_initial_tickets_l118_118224

def initial_tickets (lost : ℝ) (spent : ℝ) (left : ℝ) : ℝ :=
  lost + spent + left

theorem cody_initial_tickets : initial_tickets 6.0 25.0 18.0 = 49.0 := by
  sorry

end cody_initial_tickets_l118_118224


namespace does_not_determine_shape_l118_118433

-- Definition of a function that checks whether given data determine the shape of a triangle
def determines_shape (data : Type) : Prop := sorry

-- Various conditions about data
def ratio_two_angles_included_side : Type := sorry
def ratios_three_angle_bisectors : Type := sorry
def ratios_three_side_lengths : Type := sorry
def ratio_angle_bisector_opposite_side : Type := sorry
def three_angles : Type := sorry

-- The main theorem stating that the ratio of an angle bisector to its corresponding opposite side does not uniquely determine the shape of a triangle.
theorem does_not_determine_shape :
  ¬determines_shape ratio_angle_bisector_opposite_side := sorry

end does_not_determine_shape_l118_118433


namespace positive_sequence_unique_l118_118527

theorem positive_sequence_unique (x : Fin 2021 → ℝ) (h : ∀ i : Fin 2020, x i.succ = (x i ^ 3 + 2) / (3 * x i ^ 2)) (h' : x 2020 = x 0) : ∀ i, x i = 1 := by
  sorry

end positive_sequence_unique_l118_118527


namespace nickys_running_pace_l118_118754

theorem nickys_running_pace (head_start : ℕ) (pace_cristina : ℕ) (time_nicky : ℕ) (distance_meet : ℕ) :
  head_start = 12 →
  pace_cristina = 5 →
  time_nicky = 30 →
  distance_meet = (pace_cristina * (time_nicky - head_start)) →
  (distance_meet / time_nicky = 3) :=
by
  intros h_start h_pace_c h_time_n d_meet
  sorry

end nickys_running_pace_l118_118754


namespace find_x_l118_118622

theorem find_x 
  (x y z : ℝ)
  (h1 : (20 + 40 + 60 + x) / 4 = (10 + 70 + y + z) / 4 + 9)
  (h2 : y + z = 110) 
  : x = 106 := 
by 
  sorry

end find_x_l118_118622


namespace smaller_rectangle_dimensions_l118_118077

theorem smaller_rectangle_dimensions (side_length : ℝ) (L W : ℝ) 
  (h1 : side_length = 10) 
  (h2 : L + 2 * L = side_length) 
  (h3 : W = L) : 
  L = 10 / 3 ∧ W = 10 / 3 :=
by 
  sorry

end smaller_rectangle_dimensions_l118_118077


namespace problem_statement_l118_118684

def op (x y : ℕ) : ℕ := x^2 + 2*y

theorem problem_statement (a : ℕ) : op a (op a a) = 3*a^2 + 4*a := 
by sorry

end problem_statement_l118_118684


namespace quadratic_inequality_for_all_x_l118_118121

theorem quadratic_inequality_for_all_x {m : ℝ} :
  (∀ x : ℝ, (m^2 - 2 * m - 3) * x^2 - (m - 3) * x - 1 < 0) ↔ (-1 / 5 < m ∧ m ≤ 3) :=
begin
  sorry
end

end quadratic_inequality_for_all_x_l118_118121


namespace quadratic_func_max_value_l118_118244

theorem quadratic_func_max_value (b c x y : ℝ) (h1 : y = -x^2 + b * x + c)
(h1_x1 : (y = 0) → x = -1 ∨ x = 3) :
    -x^2 + 2 * x + 3 ≤ 4 :=
sorry

end quadratic_func_max_value_l118_118244


namespace populations_equal_in_years_l118_118463

-- Definitions
def populationX (n : ℕ) : ℤ := 68000 - 1200 * n
def populationY (n : ℕ) : ℤ := 42000 + 800 * n

-- Statement to prove
theorem populations_equal_in_years : ∃ n : ℕ, populationX n = populationY n ∧ n = 13 :=
sorry

end populations_equal_in_years_l118_118463


namespace two_digit_number_condition_l118_118970

theorem two_digit_number_condition :
  ∃ n : ℕ, 10 ≤ n ∧ n ≤ 99 ∧ n % 2 = 0 ∧ (n + 1) % 3 = 0 ∧ (n + 2) % 4 = 0 ∧ (n + 3) % 5 = 0 ∧ n = 62 :=
by
  sorry

end two_digit_number_condition_l118_118970


namespace interest_cannot_be_determined_without_investment_amount_l118_118488

theorem interest_cannot_be_determined_without_investment_amount :
  ∀ (interest_rate : ℚ) (price : ℚ) (invested_amount : Option ℚ),
  interest_rate = 0.16 → price = 128 → invested_amount = none → False :=
by
  sorry

end interest_cannot_be_determined_without_investment_amount_l118_118488


namespace simplify_and_evaluate_expression_l118_118445

theorem simplify_and_evaluate_expression (a : ℝ) (h : a^2 + 2 * a - 8 = 0) :
  ((a^2 - 4) / (a^2 - 4 * a + 4) - a / (a - 2)) / (a^2 + 2 * a) / (a - 2) = 1 / 4 := 
by 
  sorry

end simplify_and_evaluate_expression_l118_118445


namespace sum_of_consecutive_integers_sqrt_28_l118_118112

theorem sum_of_consecutive_integers_sqrt_28 (a b : ℤ) (h1 : b = a + 1) (h2 : a < Real.sqrt 28) (h3 : Real.sqrt 28 < b) : a + b = 11 :=
by 
    sorry

end sum_of_consecutive_integers_sqrt_28_l118_118112


namespace marching_band_members_l118_118180

theorem marching_band_members (B W P : ℕ) (h1 : P = 4 * W) (h2 : W = 2 * B) (h3 : B = 10) : B + W + P = 110 :=
by
  sorry

end marching_band_members_l118_118180


namespace probability_factor_120_lt_8_l118_118931

theorem probability_factor_120_lt_8 :
  let n := 120
  let total_factors := 16
  let favorable_factors := 6
  (6 / 16 : ℚ) = 3 / 8 :=
by 
  sorry

end probability_factor_120_lt_8_l118_118931


namespace triangle_inequality_proof_l118_118554

theorem triangle_inequality_proof (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) 
    (habc : a + b > c) (hbca : b + c > a) (hcab : c + a > b) :
    a^2 * (b + c - a) + b^2 * (c + a - b) + c^2 * (a + b - c) ≤ 3 * a * b * c := 
    sorry

end triangle_inequality_proof_l118_118554


namespace bisector_theorem_problem_l118_118578

theorem bisector_theorem_problem 
  (DE DF : ℝ) (h_DE : DE = 13) (h_DF : DF = 12)
  (D1F : ℝ) (h_D1F : D1F = 12 / 5) (D1E : ℝ) (h_D1E : D1E = 13 / 5)
  (XZ : ℝ) (h_XZ : XZ = 12 / 5) (XY : ℝ) (h_XY : XY = 13 / 5) :
  ∃ (XX1 : ℝ), XX1 = 12 / 25 :=
by
  use 12 / 25
  sorry

end bisector_theorem_problem_l118_118578


namespace probability_of_square_or_circle_is_seven_tenths_l118_118354

-- Define the total number of figures
def total_figures : ℕ := 10

-- Define the number of squares
def num_squares : ℕ := 4

-- Define the number of circles
def num_circles : ℕ := 3

-- The number of squares or circles
def num_squares_or_circles : ℕ := num_squares + num_circles

-- The probability of selecting a square or a circle
def probability_square_or_circle : ℚ := num_squares_or_circles / total_figures

-- The theorem stating the required proof
theorem probability_of_square_or_circle_is_seven_tenths :
  probability_square_or_circle = 7/10 :=
sorry -- proof goes here

end probability_of_square_or_circle_is_seven_tenths_l118_118354


namespace choir_member_count_l118_118911

theorem choir_member_count (n : ℕ) : 
  (n ≡ 4 [MOD 7]) ∧ 
  (n ≡ 8 [MOD 6]) ∧ 
  (50 ≤ n ∧ n ≤ 200) 
  ↔ 
  (n = 60 ∨ n = 102 ∨ n = 144 ∨ n = 186) := 
by 
  sorry

end choir_member_count_l118_118911


namespace circumscribed_sphere_surface_area_l118_118999

-- Define the setup and conditions for the right circular cone and its circumscribed sphere
theorem circumscribed_sphere_surface_area (PA PB PC AB R : ℝ)
  (h1 : AB = Real.sqrt 2)
  (h2 : PA = 1)
  (h3 : PB = 1)
  (h4 : PC = 1)
  (h5 : R = Real.sqrt 3 / 2 * PA) :
  4 * Real.pi * R ^ 2 = 3 * Real.pi :=
by
  sorry

end circumscribed_sphere_surface_area_l118_118999


namespace consecutive_integers_greatest_l118_118466

theorem consecutive_integers_greatest (n : ℤ) (h : n + 2 = 8) : 
  (n + 2 = 8) → (max n (max (n + 1) (n + 2)) = 8) :=
by {
  sorry
}

end consecutive_integers_greatest_l118_118466


namespace exponent_sum_l118_118804

theorem exponent_sum : (-2:ℝ) ^ 4 + (-2:ℝ) ^ (3 / 2) + (-2:ℝ) ^ 1 + 2 ^ 1 + 2 ^ (3 / 2) + 2 ^ 4 = 32 := by
  sorry

end exponent_sum_l118_118804


namespace geometric_sequence_product_l118_118580

variable {α : Type*} [LinearOrderedField α]
variable (a : ℕ → α)

theorem geometric_sequence_product (h : a 7 * a 12 = 5) : 
  a 8 * a 9 * a 10 * a 11 = 25 :=
by
  -- Proof goes here
  sorry

end geometric_sequence_product_l118_118580


namespace circle_center_radius_sum_l118_118876

theorem circle_center_radius_sum :
  ∃ (c d s : ℝ), c = -6 ∧ d = -7 ∧ s = Real.sqrt 13 ∧
  (x^2 + 14 * y + 72 = -y^2 - 12 * x → c + d + s = -13 + Real.sqrt 13) :=
sorry

end circle_center_radius_sum_l118_118876


namespace max_value_fraction_l118_118387

theorem max_value_fraction (x y : ℝ) : 
  (2 * x + 3 * y + 2) / Real.sqrt (x^2 + y^2 + 1) ≤ Real.sqrt 17 :=
by
  sorry

end max_value_fraction_l118_118387


namespace negation_of_there_exists_l118_118320

theorem negation_of_there_exists (x : ℝ) : ¬ (∃ x : ℝ, x^2 - x + 3 = 0) ↔ ∀ x : ℝ, x^2 - x + 3 ≠ 0 := by
  sorry

end negation_of_there_exists_l118_118320


namespace fifth_grade_soccer_students_l118_118009

variable (T B Gnp GP S : ℕ)
variable (p : ℝ)

theorem fifth_grade_soccer_students
  (hT : T = 420)
  (hB : B = 296)
  (hp_percent : p = 86 / 100)
  (hGnp : Gnp = 89)
  (hpercent_boys_playing_soccer : (1 - p) * S = GP)
  (hpercent_girls_playing_soccer : GP = 35) :
  S = 250 := by
  sorry

end fifth_grade_soccer_students_l118_118009


namespace select_from_companyA_l118_118682

noncomputable def companyA_representatives : ℕ := 40
noncomputable def companyB_representatives : ℕ := 60
noncomputable def total_representatives : ℕ := companyA_representatives + companyB_representatives
noncomputable def sample_size : ℕ := 10
noncomputable def sampling_ratio : ℚ := sample_size / total_representatives
noncomputable def selected_from_companyA : ℚ := companyA_representatives * sampling_ratio

theorem select_from_companyA : selected_from_companyA = 4 := by
  sorry


end select_from_companyA_l118_118682


namespace largest_perimeter_l118_118800

-- Define the problem's conditions
def side1 := 7
def side2 := 9
def integer_side (x : ℕ) : Prop := (x > 2) ∧ (x < 16)

-- Define the perimeter calculation
def perimeter (a b c : ℕ) := a + b + c

-- The theorem statement which we want to prove
theorem largest_perimeter : ∃ x : ℕ, integer_side x ∧ perimeter side1 side2 x = 31 :=
by
  sorry

end largest_perimeter_l118_118800


namespace wave_number_probability_l118_118345

-- Define the wave number concept
def is_wave_number (l : List ℕ) : Prop :=
  l.nth 0 < l.nth 1 ∧ l.nth 1 > l.nth 2 ∧ l.nth 2 < l.nth 3 ∧ l.nth 3 > l.nth 4

-- Define the set of digits
def digits : List ℕ := [1, 2, 3, 4, 5]

-- Define the property we want to prove
theorem wave_number_probability :
  (digits.permutations.count is_wave_number : ℚ) / digits.permutations.length = 2 / 15 :=
by
  sorry

end wave_number_probability_l118_118345


namespace rick_has_eaten_servings_l118_118917

theorem rick_has_eaten_servings (calories_per_serving block_servings remaining_calories total_calories servings_eaten : ℝ) 
  (h1 : calories_per_serving = 110) 
  (h2 : block_servings = 16) 
  (h3 : remaining_calories = 1210) 
  (h4 : total_calories = block_servings * calories_per_serving)
  (h5 : servings_eaten = (total_calories - remaining_calories) / calories_per_serving) :
  servings_eaten = 5 :=
by 
  sorry

end rick_has_eaten_servings_l118_118917


namespace sum_reciprocals_of_partial_fractions_l118_118427

noncomputable def f (s : ℝ) : ℝ := s^3 - 20 * s^2 + 125 * s - 500

theorem sum_reciprocals_of_partial_fractions :
  ∀ (p q r A B C : ℝ),
    p ≠ q ∧ q ≠ r ∧ r ≠ p ∧
    f p = 0 ∧ f q = 0 ∧ f r = 0 ∧
    (∀ s, s ≠ p ∧ s ≠ q ∧ s ≠ r → 
      (1 / f s = A / (s - p) + B / (s - q) + C / (s - r))) →
    1 / A + 1 / B + 1 / C = 720 :=
sorry

end sum_reciprocals_of_partial_fractions_l118_118427


namespace convex_over_real_l118_118828

def f (x : ℝ) : ℝ := x^4 - 2 * x^3 + 36 * x^2 - x + 7

theorem convex_over_real : ∀ x : ℝ, 0 ≤ (12 * x^2 - 12 * x + 72) :=
by sorry

end convex_over_real_l118_118828


namespace basic_astrophysics_degrees_l118_118957

theorem basic_astrophysics_degrees :
  let microphotonics_pct := 12
  let home_electronics_pct := 24
  let food_additives_pct := 15
  let gmo_pct := 29
  let industrial_lubricants_pct := 8
  let total_budget_percentage := 100
  let full_circle_degrees := 360
  let given_pct_sum := microphotonics_pct + home_electronics_pct + food_additives_pct + gmo_pct + industrial_lubricants_pct
  let astrophysics_pct := total_budget_percentage - given_pct_sum
  let astrophysics_degrees := (astrophysics_pct * full_circle_degrees) / total_budget_percentage
  astrophysics_degrees = 43.2 := by
  sorry

end basic_astrophysics_degrees_l118_118957


namespace geometric_sequence_common_ratio_l118_118400

theorem geometric_sequence_common_ratio (a : ℕ → ℝ) (q : ℝ)
  (h1 : ∀ n, a (n + 1) = a n * q)
  (h2 : ∀ n, 0 < a n)
  (h3 : 2 * (1 / 2 * a 3) = 3 * a 1 + 2 * a 2) : q = 3 := by
  sorry

end geometric_sequence_common_ratio_l118_118400


namespace range_of_f_minus_2_l118_118587

def f (x : ℝ) (a b : ℝ) : ℝ := a * x^2 + b * x

theorem range_of_f_minus_2 (a b : ℝ) (h1 : 1 ≤ f (-1) a b) (h2 : f (-1) a b ≤ 2) (h3 : 2 ≤ f 1 a b) (h4 : f 1 a b ≤ 4) :
  6 ≤ f (-2) a b ∧ f (-2) a b ≤ 10 :=
sorry

end range_of_f_minus_2_l118_118587


namespace proof_option_b_and_c_l118_118243

variable (a b c : ℝ)

theorem proof_option_b_and_c (h₀ : a > b) (h₁ : b > 0) (h₂ : c ≠ 0) :
  (b / a < (b + c^2) / (a + c^2)) ∧ (a^2 - 1 / a > b^2 - 1 / b) :=
by
  sorry

end proof_option_b_and_c_l118_118243


namespace minimum_additional_coins_needed_l118_118352

def sum_natural_numbers (n : ℕ) : ℕ :=
  n * (n + 1) / 2

theorem minimum_additional_coins_needed (friends : ℕ) (current_coins : ℕ) :
  friends = 15 → current_coins = 63 → 
  let required_coins := sum_natural_numbers friends in
  let additional_coins := required_coins - current_coins in
  additional_coins = 57 :=
by
  intros h_friends h_coins
  rw [h_friends, h_coins]
  let required_coins := sum_natural_numbers 15
  have h_required_coins : required_coins = 120 := by
    rw [sum_natural_numbers, Nat.mul, Nat.add, /, 2]
    norm_num
  
  let additional_coins := required_coins - 63
  have h_additional_coins : additional_coins = 57 := by
    rw [h_required_coins]
    norm_num
    
  exact h_additional_coins

end minimum_additional_coins_needed_l118_118352


namespace tank_saltwater_solution_l118_118349

theorem tank_saltwater_solution (x : ℝ) :
  let water1 := 0.75 * x
  let water1_evaporated := (1/3) * water1
  let water2 := water1 - water1_evaporated
  let salt2 := 0.25 * x
  let water3 := water2 + 12
  let salt3 := salt2 + 24
  let step2_eq := (salt3 / (water3 + 24)) = 0.4
  let water4 := water3 - (1/4) * water3
  let salt4 := salt3
  let water5 := water4 + 15
  let salt5 := salt4 + 30
  let step4_eq := (salt5 / (water5 + 30)) = 0.5
  step2_eq ∧ step4_eq → x = 192 :=
by
  sorry

end tank_saltwater_solution_l118_118349


namespace calculate_sequence_sum_l118_118515

noncomputable def sum_arithmetic_sequence (a l d: Int) : Int :=
  let n := ((l - a) / d) + 1
  (n * (a + l)) / 2

theorem calculate_sequence_sum :
  3 * (sum_arithmetic_sequence 45 93 2) + 2 * (sum_arithmetic_sequence (-4) 38 2) = 5923 := by
  sorry

end calculate_sequence_sum_l118_118515


namespace number_of_teachers_l118_118916

theorem number_of_teachers
  (students : ℕ) (lessons_per_student_per_day : ℕ) (lessons_per_teacher_per_day : ℕ) (students_per_class : ℕ)
  (h1 : students = 1200)
  (h2 : lessons_per_student_per_day = 5)
  (h3 : lessons_per_teacher_per_day = 4)
  (h4 : students_per_class = 30) :
  ∃ teachers : ℕ, teachers = 50 :=
by
  have total_lessons : ℕ := lessons_per_student_per_day * students
  have classes : ℕ := total_lessons / students_per_class
  have teachers : ℕ := classes / lessons_per_teacher_per_day
  use teachers
  sorry

end number_of_teachers_l118_118916


namespace Claudia_solution_l118_118364

noncomputable def Claudia_coins : Prop :=
  ∃ (x y : ℕ), x + y = 12 ∧ 23 - x = 17 ∧ y = 6

theorem Claudia_solution : Claudia_coins :=
by
  existsi 6
  existsi 6
  sorry

end Claudia_solution_l118_118364


namespace time_to_reach_rest_area_l118_118336

variable (rate_per_minute : ℕ) (remaining_distance_yards : ℕ)

theorem time_to_reach_rest_area (h_rate : rate_per_minute = 2) (h_distance : remaining_distance_yards = 50) :
  (remaining_distance_yards * 3) / rate_per_minute = 75 := by
  sorry

end time_to_reach_rest_area_l118_118336


namespace combo_discount_is_50_percent_l118_118187

noncomputable def combo_discount_percentage
  (ticket_cost : ℕ) (combo_cost : ℕ) (ticket_discount : ℕ) (total_savings : ℕ) : ℕ :=
  let ticket_savings := ticket_cost * ticket_discount / 100
  let combo_savings := total_savings - ticket_savings
  (combo_savings * 100) / combo_cost

theorem combo_discount_is_50_percent:
  combo_discount_percentage 10 10 20 7 = 50 :=
by
  sorry

end combo_discount_is_50_percent_l118_118187


namespace find_b_plus_c_l118_118997

theorem find_b_plus_c (a b c d : ℝ) 
    (h₁ : a + d = 6) 
    (h₂ : a * b + a * c + b * d + c * d = 40) : 
    b + c = 20 / 3 := 
sorry

end find_b_plus_c_l118_118997


namespace serpent_ridge_trail_length_l118_118027

/-- Phoenix hiked the Serpent Ridge Trail last week. It took her five days to complete the trip.
The first two days she hiked a total of 28 miles. The second and fourth days she averaged 15 miles per day.
The last three days she hiked a total of 42 miles. The total hike for the first and third days was 30 miles.
How many miles long was the trail? -/
theorem serpent_ridge_trail_length
  (a b c d e : ℕ)
  (h1 : a + b = 28)
  (h2 : b + d = 30)
  (h3 : c + d + e = 42)
  (h4 : a + c = 30) :
  a + b + c + d + e = 70 :=
sorry

end serpent_ridge_trail_length_l118_118027


namespace quadratic_one_root_iff_discriminant_zero_l118_118705

theorem quadratic_one_root_iff_discriminant_zero (m : ℝ) : 
  (∃ x : ℝ, ∀ y : ℝ, y^2 - m*y + 1 ≤ 0 ↔ y = x) ↔ (m = 2 ∨ m = -2) :=
by 
  -- We assume the discriminant condition which implies the result
  sorry

end quadratic_one_root_iff_discriminant_zero_l118_118705


namespace choir_members_count_l118_118727

theorem choir_members_count : ∃ n : ℕ, n = 226 ∧ 
  (n % 10 = 6) ∧ 
  (n % 11 = 6) ∧ 
  (200 < n ∧ n < 300) :=
by
  sorry

end choir_members_count_l118_118727


namespace fraction_product_l118_118634

theorem fraction_product (a b : ℕ) 
  (h1 : 1/5 < a / b)
  (h2 : a / b < 1/4)
  (h3 : b ≤ 19) :
  ∃ a1 a2 b1 b2, 4 * a2 < b1 ∧ b1 < 5 * a2 ∧ b2 ≤ 19 ∧ 4 * a2 < b2 ∧ b2 < 20 ∧ a = 4 ∧ b = 19 ∧ a1 = 2 ∧ b1 = 9 ∧ 
  (a + b = 23 ∨ a + b = 11) ∧ (23 * 11 = 253) := by
  sorry

end fraction_product_l118_118634


namespace train_length_l118_118799

theorem train_length (v : ℝ) (t : ℝ) (l_b : ℝ) (v_r : v = 52) (t_r : t = 34.61538461538461) (l_b_r : l_b = 140) : 
  ∃ l_t : ℝ, l_t = 360 :=
by
  have speed_ms := v * (1000 / 3600)
  have total_distance := speed_ms * t
  have length_train := total_distance - l_b
  use length_train
  sorry

end train_length_l118_118799


namespace find_number_l118_118660

theorem find_number (x : ℝ) (h : 140 = 3.5 * x) : x = 40 :=
by
  sorry

end find_number_l118_118660


namespace smallest_k_l118_118656

def arith_seq_sum (k n : ℕ) : ℕ :=
  (n + 1) * (2 * k + n) / 2

theorem smallest_k (k n : ℕ) (h_sum : arith_seq_sum k n = 100) :
  k = 9 :=
by
  sorry

end smallest_k_l118_118656


namespace sum_of_coordinates_of_point_D_l118_118162

theorem sum_of_coordinates_of_point_D : 
  ∀ {x : ℝ}, (y = 6) ∧ (x ≠ 0) ∧ ((6 - 0) / (x - 0) = 3 / 4) → x + y = 14 := by
  intros x hx hy hslope
  sorry

end sum_of_coordinates_of_point_D_l118_118162


namespace sum_arithmetic_sequence_terms_l118_118251

noncomputable def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
∀ n m : ℕ, a (n + m) = a n + m * (a 1 - a 0)

theorem sum_arithmetic_sequence_terms (a : ℕ → ℝ)
  (h_arith : arithmetic_sequence a) 
  (h₅ : a 5 = 8) :
  a 2 + a 4 + a 5 + a 9 = 32 :=
by
  sorry

end sum_arithmetic_sequence_terms_l118_118251


namespace benzoic_acid_molecular_weight_l118_118228

-- Definitions for atomic weights
def atomic_weight_C : ℝ := 12.01
def atomic_weight_H : ℝ := 1.008
def atomic_weight_O : ℝ := 16.00

-- Molecular formula for Benzoic acid: C7H6O2
def benzoic_acid_formula : ℕ × ℕ × ℕ := (7, 6, 2)

-- Definition for the molecular weight calculation
def molecular_weight := λ (c h o : ℝ) (nC nH nO : ℕ) => 
  (nC * c) + (nH * h) + (nO * o)

-- Proof statement
theorem benzoic_acid_molecular_weight :
  molecular_weight atomic_weight_C atomic_weight_H atomic_weight_O 7 6 2 = 122.118 := by
  sorry

end benzoic_acid_molecular_weight_l118_118228


namespace parabola_c_value_l118_118074

theorem parabola_c_value (b c : ℝ) 
  (h1 : 5 = 2 * 1^2 + b * 1 + c)
  (h2 : 17 = 2 * 3^2 + b * 3 + c) : 
  c = 5 := 
by
  sorry

end parabola_c_value_l118_118074


namespace jonas_tshirts_count_l118_118585

def pairs_to_individuals (pairs : Nat) : Nat := pairs * 2

variable (num_pairs_socks : Nat := 20)
variable (num_pairs_shoes : Nat := 5)
variable (num_pairs_pants : Nat := 10)
variable (num_additional_pairs_socks : Nat := 35)

def total_individual_items_without_tshirts : Nat :=
  pairs_to_individuals num_pairs_socks +
  pairs_to_individuals num_pairs_shoes +
  pairs_to_individuals num_pairs_pants

def total_individual_items_desired : Nat :=
  total_individual_items_without_tshirts +
  pairs_to_individuals num_additional_pairs_socks

def tshirts_jonas_needs : Nat :=
  total_individual_items_desired - total_individual_items_without_tshirts

theorem jonas_tshirts_count : tshirts_jonas_needs = 70 := by
  sorry

end jonas_tshirts_count_l118_118585


namespace calculate_fraction_l118_118760

theorem calculate_fraction :
  (-1 / 42) / (1 / 6 - 3 / 14 + 2 / 3 - 2 / 7) = -1 / 14 :=
by
  sorry

end calculate_fraction_l118_118760


namespace linear_func_3_5_l118_118626

def linear_function (f : ℝ → ℝ) : Prop :=
  ∀ x y : ℝ, f (x + y) = f x + f y

theorem linear_func_3_5 (f : ℝ → ℝ) (h_linear: linear_function f) 
  (h_diff: ∀ d : ℝ, f (d + 1) - f d = 3) : f 3 - f 5 = -6 :=
by
  sorry

end linear_func_3_5_l118_118626


namespace eliminate_y_by_subtraction_l118_118335

theorem eliminate_y_by_subtraction (m n : ℝ) :
  (6 * x + m * y = 3) ∧ (2 * x - n * y = -6) →
  (∀ x y : ℝ, 4 * x + (m + n) * y = 9) → (m + n = 0) :=
by
  intros h eq_subtracted
  sorry

end eliminate_y_by_subtraction_l118_118335


namespace jorge_land_fraction_clay_rich_soil_l118_118586

theorem jorge_land_fraction_clay_rich_soil 
  (total_acres : ℕ) 
  (yield_good_soil_per_acre : ℕ) 
  (yield_clay_soil_factor : ℕ)
  (total_yield : ℕ) 
  (fraction_clay_rich_soil : ℚ) :
  total_acres = 60 →
  yield_good_soil_per_acre = 400 →
  yield_clay_soil_factor = 2 →
  total_yield = 20000 →
  fraction_clay_rich_soil = 1/3 :=
by
  intro h_total_acres h_yield_good_soil_per_acre h_yield_clay_soil_factor h_total_yield
  -- math proof will be here
  sorry

end jorge_land_fraction_clay_rich_soil_l118_118586


namespace simplify_expression_l118_118879

variable (a b c d : ℝ)
variable (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0) (hd : d ≠ 0) (h : a + b + c = d)

theorem simplify_expression :
  (1 / (b^2 + c^2 - a^2)) + (1 / (a^2 + c^2 - b^2)) + (1 / (a^2 + b^2 - c^2)) = 0 := by
  sorry

end simplify_expression_l118_118879


namespace original_money_in_wallet_l118_118543

-- Definitions based on the problem's conditions
def grandmother_gift : ℕ := 20
def aunt_gift : ℕ := 25
def uncle_gift : ℕ := 30
def cost_per_game : ℕ := 35
def number_of_games : ℕ := 3
def money_left : ℕ := 20

-- Calculations as specified in the solution
def birthday_money := grandmother_gift + aunt_gift + uncle_gift
def total_game_cost := cost_per_game * number_of_games
def total_money_before_purchase := total_game_cost + money_left

-- Proof that the original amount of money in Geoffrey's wallet
-- was €50 before he got the birthday money and made the purchase.
theorem original_money_in_wallet : total_money_before_purchase - birthday_money = 50 := by
  sorry

end original_money_in_wallet_l118_118543


namespace negation_of_there_exists_l118_118321

theorem negation_of_there_exists (x : ℝ) : ¬ (∃ x : ℝ, x^2 - x + 3 = 0) ↔ ∀ x : ℝ, x^2 - x + 3 ≠ 0 := by
  sorry

end negation_of_there_exists_l118_118321


namespace Jake_not_drop_coffee_l118_118010

theorem Jake_not_drop_coffee :
  (40% / 100) * (25% / 100) = 10% / 100 → 
  100% / 100 - 10% / 100 = 90% / 100 :=
begin
  sorry
end

end Jake_not_drop_coffee_l118_118010


namespace unique_pegboard_arrangement_l118_118326

/-- Conceptually, we will set up a function to count valid arrangements of pegs
based on the given conditions and prove that there is exactly one such arrangement. -/
def triangular_pegboard_arrangements (yellow red green blue orange black : ℕ) : ℕ :=
  if yellow = 6 ∧ red = 5 ∧ green = 4 ∧ blue = 3 ∧ orange = 2 ∧ black = 1 then 1 else 0

theorem unique_pegboard_arrangement :
  triangular_pegboard_arrangements 6 5 4 3 2 1 = 1 :=
by
  -- Placeholder for proof
  sorry

end unique_pegboard_arrangement_l118_118326


namespace cost_of_socks_l118_118990

theorem cost_of_socks (cost_shirt_no_discount cost_pants_no_discount cost_shirt_discounted cost_pants_discounted cost_socks_discounted total_savings team_size socks_cost_no_discount : ℝ) 
    (h1 : cost_shirt_no_discount = 7.5)
    (h2 : cost_pants_no_discount = 15)
    (h3 : cost_shirt_discounted = 6.75)
    (h4 : cost_pants_discounted = 13.5)
    (h5 : cost_socks_discounted = 3.75)
    (h6 : total_savings = 36)
    (h7 : team_size = 12)
    (h8 : 12 * (7.5 + 15 + socks_cost_no_discount) - 12 * (6.75 + 13.5 + 3.75) = 36)
    : socks_cost_no_discount = 4.5 :=
by
  sorry

end cost_of_socks_l118_118990


namespace isosceles_triangle_bisector_properties_l118_118454

theorem isosceles_triangle_bisector_properties:
  ∀ (T : Type) (triangle : T)
  (is_isosceles : Prop) (vertex_angle_bisector_bisects_base : Prop) (vertex_angle_bisector_perpendicular_to_base : Prop),
  is_isosceles 
  → (vertex_angle_bisector_bisects_base ∧ vertex_angle_bisector_perpendicular_to_base) :=
sorry

end isosceles_triangle_bisector_properties_l118_118454


namespace residue_mod_neg_935_mod_24_l118_118989

theorem residue_mod_neg_935_mod_24 : (-935) % 24 = 1 :=
by
  sorry

end residue_mod_neg_935_mod_24_l118_118989


namespace johns_watermelon_weight_l118_118296

theorem johns_watermelon_weight (michael_weight clay_weight john_weight : ℕ)
  (h1 : michael_weight = 8)
  (h2 : clay_weight = 3 * michael_weight)
  (h3 : john_weight = clay_weight / 2) :
  john_weight = 12 :=
by
  sorry

end johns_watermelon_weight_l118_118296


namespace plane_equation_l118_118386

theorem plane_equation 
  (P Q : ℝ×ℝ×ℝ) (A B : ℝ×ℝ×ℝ)
  (hp : P = (-1, 2, 5))
  (hq : Q = (3, -4, 1))
  (ha : A = (0, -2, -1))
  (hb : B = (3, 2, -1)) :
  ∃ (a b c d : ℝ), (a = 3 ∧ b = 4 ∧ c = 0 ∧ d = 1) ∧ (∀ x y z : ℝ, a * (x - 1) + b * (y + 1) + c * (z - 3) = d) :=
by
  sorry

end plane_equation_l118_118386


namespace find_z_value_l118_118726

variables {BD FC GC FE : Prop}
variables {a b c d e f g z : ℝ}

-- Assume all given conditions
axiom BD_is_straight : BD
axiom FC_is_straight : FC
axiom GC_is_straight : GC
axiom FE_is_straight : FE
axiom sum_is_z : z = a + b + c + d + e + f + g

-- Goal to prove
theorem find_z_value : z = 540 :=
by
  sorry

end find_z_value_l118_118726


namespace xy_yz_zx_value_l118_118618

theorem xy_yz_zx_value (x y z : ℝ) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z)
  (h1 : x^2 + x * y + y^2 = 9) (h2 : y^2 + y * z + z^2 = 16) (h3 : z^2 + z * x + x^2 = 25) :
  x * y + y * z + z * x = 8 * Real.sqrt 3 :=
by sorry

end xy_yz_zx_value_l118_118618


namespace continuous_at_2_l118_118945

-- Define the function f
def f (x : ℝ) : ℝ := -5 * x^2 - 8

-- Define continuity at a point
def is_continuous_at (f : ℝ → ℝ) (x0 : ℝ) : Prop :=
  ∀ ε > 0, ∃ δ > 0, ∀ x, |x - x0| < δ → |f x - f x0| < ε

-- The theorem to prove the continuity of the function at x0 = 2
theorem continuous_at_2 : is_continuous_at f 2 :=
by
  intro ε ε_pos
  let δ := ε / 25
  use δ
  split
  · linarith
  · intro x hx
    have : |f x - f 2| < ε, sorry
    assumption

end continuous_at_2_l118_118945


namespace lisa_max_non_a_quizzes_l118_118677

def lisa_goal : ℕ := 34
def quizzes_total : ℕ := 40
def quizzes_taken_first : ℕ := 25
def quizzes_with_a_first : ℕ := 20
def remaining_quizzes : ℕ := quizzes_total - quizzes_taken_first
def additional_a_needed : ℕ := lisa_goal - quizzes_with_a_first

theorem lisa_max_non_a_quizzes : 
  additional_a_needed ≤ remaining_quizzes → 
  remaining_quizzes - additional_a_needed ≤ 1 :=
by
  sorry

end lisa_max_non_a_quizzes_l118_118677


namespace Jake_not_drop_coffee_l118_118011

theorem Jake_not_drop_coffee :
  (40% / 100) * (25% / 100) = 10% / 100 → 
  100% / 100 - 10% / 100 = 90% / 100 :=
begin
  sorry
end

end Jake_not_drop_coffee_l118_118011


namespace model_tower_height_l118_118378

-- Definitions based on conditions
def height_actual_tower : ℝ := 60
def volume_actual_tower : ℝ := 80000
def volume_model_tower : ℝ := 0.5

-- Theorem statement
theorem model_tower_height (h: ℝ) : h = 0.15 :=
by
  sorry

end model_tower_height_l118_118378


namespace p_divisible_by_5_l118_118928

noncomputable theory

open Nat

def q_infinity_cubed (q : ℚ) : ℕ → ℚ := 
λ k => (-1)^k * (2*k+1) * q^(k*(k+1)/2)

def q_infinity_quart (q : ℚ) (n : ℕ) : Prop :=
  ∑ k in range (n + 1), ((-1)^k * (2*k+1) * q^(k*(k+1)/2)) = 0

def mod_equiv1 (q : ℚ) : Prop :=
  (1 - q^5)/(1 - q)^5 ≡ 1 [MOD 5]

theorem p_divisible_by_5 (n : ℕ) : 
  ((∀ n : ℕ, q_infinity_cubed q ∑ k in range (n+1), k) ∨ q_infinity_quart q n) ∧
  mod_equiv1 q ∧ 
  p (5 * n + 4) ∣ 5 := sorry

end p_divisible_by_5_l118_118928


namespace real_number_set_condition_l118_118632

theorem real_number_set_condition (x : ℝ) :
  (x ≠ 1) ∧ (x^2 - x ≠ 1) ∧ (x^2 - x ≠ x) →
  x ≠ 0 ∧ x ≠ 1 ∧ x ≠ 2 ∧ x ≠ (1 + Real.sqrt 5) / 2 ∧ x ≠ (1 - Real.sqrt 5) / 2 := 
by
  sorry

end real_number_set_condition_l118_118632


namespace sin_pi_minus_alpha_l118_118702

theorem sin_pi_minus_alpha (α : ℝ) (h1 : α ∈ Set.Ioo 0 Real.pi) (h2 : Real.cos α = 4 / 5) :
  Real.sin (Real.pi - α) = 3 / 5 := 
sorry

end sin_pi_minus_alpha_l118_118702


namespace tins_of_beans_left_l118_118491

theorem tins_of_beans_left (cases : ℕ) (tins_per_case : ℕ) (damage_percentage : ℝ) (h_cases : cases = 15)
  (h_tins_per_case : tins_per_case = 24) (h_damage_percentage : damage_percentage = 0.05) :
  let total_tins := cases * tins_per_case
  let damaged_tins := total_tins * damage_percentage
  let tins_left := total_tins - damaged_tins
  tins_left = 342 :=
by
  sorry

end tins_of_beans_left_l118_118491


namespace find_sets_l118_118237

variable (A X Y : Set ℕ) -- Mimicking sets of natural numbers for generality.

theorem find_sets (h1 : X ∪ Y = A) (h2 : X ∩ A = Y) : X = A ∧ Y = A := by
  -- This would need a proof, which shows that: X = A and Y = A
  sorry

end find_sets_l118_118237


namespace february_saving_l118_118757

-- Definitions for the conditions
variable {F D : ℝ}

-- Condition 1: Saving in January
def january_saving : ℝ := 2

-- Condition 2: Saving in March
def march_saving : ℝ := 8

-- Condition 3: Total savings after 6 months
def total_savings : ℝ := 126

-- Condition 4: Savings increase by a fixed amount D each month
def fixed_increase : ℝ := D

-- Condition 5: Difference between savings in March and January
def difference_jan_mar : ℝ := 8 - 2

-- The main theorem to prove: Robi saved 50 in February
theorem february_saving : F = 50 :=
by
  -- The required proof is omitted
  sorry

end february_saving_l118_118757


namespace sum_of_ages_l118_118231

variables (Matthew Rebecca Freddy: ℕ)
variables (H1: Matthew = Rebecca + 2)
variables (H2: Matthew = Freddy - 4)
variables (H3: Freddy = 15)

theorem sum_of_ages
  (H1: Matthew = Rebecca + 2)
  (H2: Matthew = Freddy - 4)
  (H3: Freddy = 15):
  Matthew + Rebecca + Freddy = 35 :=
  sorry

end sum_of_ages_l118_118231


namespace A_B_work_together_finish_l118_118973
noncomputable def work_rate_B := 1 / 12
noncomputable def work_rate_A := 2 * work_rate_B
noncomputable def combined_work_rate := work_rate_A + work_rate_B

theorem A_B_work_together_finish (hB: work_rate_B = 1/12) (hA: work_rate_A = 2 * work_rate_B) :
  (1 / combined_work_rate) = 4 :=
by
  -- Placeholder for the proof, we don't need to provide the proof steps
  sorry

end A_B_work_together_finish_l118_118973


namespace range_of_m_l118_118555

theorem range_of_m (m : ℝ) :
  (∀ x : ℝ, (x >= (4 + m)) ∧ (x <= 3 * (x - 2) + 4) → (x ≥ 2)) →
  (-3 < m ∧ m <= -2) :=
sorry

end range_of_m_l118_118555


namespace triangle_eq_medians_incircle_l118_118188

-- Define a triangle and the properties of medians and incircle
structure Triangle (α : Type) [Nonempty α] :=
(A B C : α)

def is_equilateral {α : Type} [Nonempty α] (T : Triangle α) : Prop :=
  ∃ (d : α → α → ℝ), d T.A T.B = d T.B T.C ∧ d T.B T.C = d T.C T.A

def medians_segments_equal {α : Type} [Nonempty α] (T : Triangle α) (incr_len : (α → α → ℝ)) : Prop :=
  ∀ (MA MB MC : α), incr_len MA MB = incr_len MB MC ∧ incr_len MB MC = incr_len MC MA

-- The main theorem statement
theorem triangle_eq_medians_incircle {α : Type} [Nonempty α] 
  (T : Triangle α) (incr_len : α → α → ℝ) 
  (h : medians_segments_equal T incr_len) : is_equilateral T :=
sorry

end triangle_eq_medians_incircle_l118_118188


namespace complement_intersection_l118_118723

def M : Set ℝ := {x | x ≥ 1}
def N : Set ℝ := {x | 0 ≤ x ∧ x ≤ 3}

theorem complement_intersection : (M ∩ N)ᶜ = { x : ℝ | x < 1 ∨ x > 3 } :=
  sorry

end complement_intersection_l118_118723


namespace samara_tire_spending_l118_118080

theorem samara_tire_spending :
  ∀ (T : ℕ), 
    (2457 = 25 + 79 + T + 1886) → 
    T = 467 :=
by intros T h
   sorry

end samara_tire_spending_l118_118080


namespace find_a_and_b_l118_118020

theorem find_a_and_b (a b : ℤ) (h1 : 3 * (b + a^2) = 99) (h2 : 3 * a * b^2 = 162) : a = 6 ∧ b = -3 :=
sorry

end find_a_and_b_l118_118020


namespace two_digit_number_condition_l118_118971

theorem two_digit_number_condition :
  ∃ n : ℕ, 10 ≤ n ∧ n ≤ 99 ∧ n % 2 = 0 ∧ (n + 1) % 3 = 0 ∧ (n + 2) % 4 = 0 ∧ (n + 3) % 5 = 0 ∧ n = 62 :=
by
  sorry

end two_digit_number_condition_l118_118971


namespace downward_parabola_with_symmetry_l118_118159

-- Define the general form of the problem conditions in Lean
theorem downward_parabola_with_symmetry (k : ℝ) :
  ∃ a : ℝ, a < 0 ∧ ∃ h : ℝ, h = 3 ∧ ∃ k : ℝ, k = k ∧ ∃ (y x : ℝ), y = a * (x - h)^2 + k :=
sorry

end downward_parabola_with_symmetry_l118_118159


namespace simplify_and_evaluate_equals_l118_118442

noncomputable def simplify_and_evaluate (a : ℝ) : ℝ :=
  (a^2 - 4) / (a^2 - 4 * a + 4) - a / (a - 2) / (a^2 + 2 * a) / (a - 2)

theorem simplify_and_evaluate_equals (a : ℝ) (h : a^2 + 2 * a - 8 = 0) : 
  simplify_and_evaluate a = 1 / 4 :=
sorry

end simplify_and_evaluate_equals_l118_118442


namespace final_configuration_l118_118583

def initial_configuration : (String × String) :=
  ("bottom-right", "bottom-left")

def first_transformation (conf : (String × String)) : (String × String) :=
  match conf with
  | ("bottom-right", "bottom-left") => ("top-right", "top-left")
  | _ => conf

def second_transformation (conf : (String × String)) : (String × String) :=
  match conf with
  | ("top-right", "top-left") => ("top-left", "top-right")
  | _ => conf

theorem final_configuration :
  second_transformation (first_transformation initial_configuration) =
  ("top-left", "top-right") :=
by
  sorry

end final_configuration_l118_118583


namespace prime_solution_l118_118891

theorem prime_solution (a b c : ℕ) (ha : Nat.Prime a) (hb : Nat.Prime b) (hc : Nat.Prime c) 
  (h1 : 2 * a - b + 7 * c = 1826) (h2 : 3 * a + 5 * b + 7 * c = 2007) :
  a = 7 ∧ b = 29 ∧ c = 263 :=
by
  sorry

end prime_solution_l118_118891


namespace shelves_per_case_l118_118734

noncomputable section

-- Define the total number of ridges
def total_ridges : ℕ := 8640

-- Define the number of ridges per record
def ridges_per_record : ℕ := 60

-- Define the number of records per shelf when the shelf is 60% full
def records_per_shelf : ℕ := (60 * 20) / 100

-- Define the number of ridges per shelf
def ridges_per_shelf : ℕ := records_per_shelf * ridges_per_record

-- Given 4 cases, we need to determine the number of shelves per case
theorem shelves_per_case (cases shelves : ℕ) (h₁ : cases = 4) (h₂ : shelves * ridges_per_shelf = total_ridges) :
  shelves / cases = 3 := by
  sorry

end shelves_per_case_l118_118734


namespace ethanol_relationship_l118_118220

variables (a b c x : ℝ)
def total_capacity := a + b + c = 300
def ethanol_content := x = 0.10 * a + 0.15 * b + 0.20 * c
def ethanol_bounds := 30 ≤ x ∧ x ≤ 60

theorem ethanol_relationship : total_capacity a b c → ethanol_bounds x → ethanol_content a b c x :=
by
  intros h_total h_bounds
  unfold total_capacity at h_total
  unfold ethanol_bounds at h_bounds
  unfold ethanol_content
  sorry

end ethanol_relationship_l118_118220


namespace tracy_candies_l118_118779

variable (x : ℕ) -- number of candies Tracy started with

theorem tracy_candies (h1: x % 4 = 0)
                      (h2 : 46 ≤ x / 2 - 40 ∧ x / 2 - 40 ≤ 50) 
                      (h3 : ∃ k, 2 ≤ k ∧ k ≤ 6 ∧ x / 2 - 40 - k = 4) 
                      (h4 : ∃ n, x = 4 * n) : x = 96 :=
by
  sorry

end tracy_candies_l118_118779


namespace quadrilateral_EFGH_l118_118004

variable {EF FG GH HE EH : ℤ}

theorem quadrilateral_EFGH (h1 : EF = 6) (h2 : FG = 18) (h3 : GH = 6) (h4 : HE = 10) (h5 : 12 < EH) (h6 : EH < 24) : EH = 12 := 
sorry

end quadrilateral_EFGH_l118_118004


namespace radius_of_tangent_intersection_l118_118486

variable (x y : ℝ)

def circle_eq : Prop := x^2 + y^2 = 25

def tangent_condition : Prop := y = 5 ∧ x = 0

theorem radius_of_tangent_intersection (h1 : circle_eq x y) (h2 : tangent_condition x y) : ∃r : ℝ, r = 5 :=
by sorry

end radius_of_tangent_intersection_l118_118486


namespace number_of_parallelograms_l118_118375

-- Problem statement in Lean 4
theorem number_of_parallelograms (n : ℕ) : 
  let k := n + 1 in
  -- Number of parallelograms formed
  3 * (n * (n - 1) / 2) = 3 * nat.choose n 2 :=
by sorry

end number_of_parallelograms_l118_118375


namespace minimum_value_of_function_l118_118528

theorem minimum_value_of_function (x : ℝ) (hx : x > 1) : (x + 4 / (x - 1)) ≥ 5 := by
  sorry

end minimum_value_of_function_l118_118528


namespace number_of_whole_numbers_l118_118562

theorem number_of_whole_numbers (x y : ℝ) (hx : 2 < x ∧ x < 3) (hy : 8 < y ∧ y < 9) : 
  ∃ (n : ℕ), n = 6 := by
  sorry

end number_of_whole_numbers_l118_118562


namespace percent_problem_l118_118572

theorem percent_problem (x : ℝ) (h : 0.3 * 0.15 * x = 18) : 0.15 * 0.3 * x = 18 :=
by
  sorry

end percent_problem_l118_118572


namespace lcm_of_two_numbers_hcf_and_product_l118_118640

theorem lcm_of_two_numbers_hcf_and_product (a b : ℕ) (h_hcf : Nat.gcd a b = 20) (h_prod : a * b = 2560) :
  Nat.lcm a b = 128 :=
by
  sorry

end lcm_of_two_numbers_hcf_and_product_l118_118640


namespace tracy_sold_paintings_l118_118054

-- Definitions of conditions
def total_customers := 20
def first_group_customers := 4
def paintings_per_first_group_customer := 2
def second_group_customers := 12
def paintings_per_second_group_customer := 1
def third_group_customers := 4
def paintings_per_third_group_customer := 4

-- Statement of the problem
theorem tracy_sold_paintings :
  (first_group_customers * paintings_per_first_group_customer) +
  (second_group_customers * paintings_per_second_group_customer) +
  (third_group_customers * paintings_per_third_group_customer) = 36 :=
by
  sorry

end tracy_sold_paintings_l118_118054


namespace proof_two_digit_number_l118_118968

noncomputable def two_digit_number := {n : ℤ // 10 ≤ n ∧ n ≤ 99}

theorem proof_two_digit_number (n : two_digit_number) :
  (n.val % 2 = 0) ∧ 
  ((n.val + 1) % 3 = 0) ∧
  ((n.val + 2) % 4 = 0) ∧
  ((n.val + 3) % 5 = 0) →
  n.val = 62 :=
by sorry

end proof_two_digit_number_l118_118968


namespace least_four_digit_divisible_by_15_25_40_75_is_1200_l118_118189

def is_four_digit (n : ℕ) : Prop :=
  1000 ≤ n ∧ n < 10000

def divisible_by_15 (n : ℕ) : Prop :=
  n % 15 = 0

def divisible_by_25 (n : ℕ) : Prop :=
  n % 25 = 0

def divisible_by_40 (n : ℕ) : Prop :=
  n % 40 = 0

def divisible_by_75 (n : ℕ) : Prop :=
  n % 75 = 0

theorem least_four_digit_divisible_by_15_25_40_75_is_1200 :
  ∃ n : ℕ, is_four_digit n ∧ divisible_by_15 n ∧ divisible_by_25 n ∧ divisible_by_40 n ∧ divisible_by_75 n ∧
  (∀ m : ℕ, is_four_digit m ∧ divisible_by_15 m ∧ divisible_by_25 m ∧ divisible_by_40 m ∧ divisible_by_75 m → n ≤ m) ∧
  n = 1200 := 
sorry

end least_four_digit_divisible_by_15_25_40_75_is_1200_l118_118189


namespace max_n_satisfying_conditions_l118_118745

theorem max_n_satisfying_conditions : 
  ∃ n (a : Fin n → ℕ), 
    1 = a 0 ∧ 
    a (Fin.last n) = 2009 ∧ 
    (∀ i j, i < j → a i < a j) ∧ 
    (∀ i, (∑ k in Finset.univ.erase i, a k) % (n-1) = 0) :=
sorry

end max_n_satisfying_conditions_l118_118745


namespace staircase_steps_l118_118584

theorem staircase_steps (x : ℕ) (h1 : x + 2 * x + (2 * x - 10) = 2 * 45) : x = 20 :=
by 
  -- The proof is skipped
  sorry

end staircase_steps_l118_118584


namespace actual_time_when_watch_shows_8_PM_l118_118681

-- Definitions based on the problem's conditions
def initial_time := 8  -- 8:00 AM
def incorrect_watch_time := 14 * 60 + 42  -- 2:42 PM converted to minutes
def actual_time := 15 * 60  -- 3:00 PM converted to minutes
def target_watch_time := 20 * 60  -- 8:00 PM converted to minutes

-- Define to calculate the rate of time loss
def time_loss_rate := (actual_time - incorrect_watch_time) / (actual_time - initial_time * 60)

-- Hypothesis that the watch loses time at a constant rate
axiom constant_rate : ∀ t, t >= initial_time * 60 ∧ t <= actual_time → (t * time_loss_rate) = (actual_time - incorrect_watch_time)

-- Define the target time based on watch reading 8:00 PM
noncomputable def target_actual_time := target_watch_time / time_loss_rate

-- Main theorem: Prove that given the conditions, the target actual time is 8:32 PM
theorem actual_time_when_watch_shows_8_PM : target_actual_time = (20 * 60 + 32) :=
sorry

end actual_time_when_watch_shows_8_PM_l118_118681


namespace subtract_from_40_squared_l118_118048

theorem subtract_from_40_squared : 39 * 39 = 40 * 40 - 79 := by
  sorry

end subtract_from_40_squared_l118_118048


namespace rectangle_area_l118_118213

theorem rectangle_area (d : ℝ) (w : ℝ) (h : (3 * w)^2 + w^2 = d^2) : 
  3 * w^2 = d^2 / 10 :=
by
  sorry

end rectangle_area_l118_118213


namespace binomial_9_3_l118_118367

theorem binomial_9_3 : (Nat.choose 9 3) = 84 := by
  sorry

end binomial_9_3_l118_118367


namespace doughnut_completion_l118_118343

theorem doughnut_completion :
  let start_time := 8 * 60 + 30 in -- 8:30 AM in minutes
  let one_third_time := 11 * 60 + 10 - start_time in -- Duration from 8:30 AM to 11:10 AM in minutes
  let total_time := 3 * one_third_time in -- Total time to finish the job
  let completion_time := start_time + total_time in -- Completion time in minutes
  completion_time = 16 * 60 + 30 := -- 4:30 PM in minutes
by
  sorry

end doughnut_completion_l118_118343


namespace cheaperCandy_cost_is_5_l118_118959

def cheaperCandy (C : ℝ) : Prop :=
  let expensiveCandyCost := 20 * 8
  let cheaperCandyCost := 40 * C
  let totalWeight := 20 + 40
  let totalCost := 60 * 6
  expensiveCandyCost + cheaperCandyCost = totalCost

theorem cheaperCandy_cost_is_5 : cheaperCandy 5 :=
by
  unfold cheaperCandy
  -- SORRY is a placeholder for the proof steps, which are not required
  sorry 

end cheaperCandy_cost_is_5_l118_118959


namespace sandra_coffee_l118_118613

theorem sandra_coffee (S : ℕ) (H1 : 2 + S = 8) : S = 6 :=
by
  sorry

end sandra_coffee_l118_118613


namespace number_of_positive_divisors_not_divisible_by_3_of_180_l118_118410

theorem number_of_positive_divisors_not_divisible_by_3_of_180 : 
  (finset.card ((finset.range 3).bind (λ a, (finset.range 2).image (λ c, 2^a * 5^c)))) = 6 :=
sorry

end number_of_positive_divisors_not_divisible_by_3_of_180_l118_118410


namespace cost_to_marked_price_ratio_l118_118484

variables (p : ℝ) (discount : ℝ := 0.20) (cost_ratio : ℝ := 0.60)

theorem cost_to_marked_price_ratio :
  (cost_ratio * (1 - discount) * p) / p = 0.48 :=
by sorry

end cost_to_marked_price_ratio_l118_118484


namespace B_subset_A_implies_m_values_l118_118152

noncomputable def A : Set ℝ := { x | x^2 + x - 6 = 0 }
noncomputable def B (m : ℝ) : Set ℝ := { x | m * x + 1 = 0 }
def possible_m_values : Set ℝ := {1/3, -1/2}

theorem B_subset_A_implies_m_values (m : ℝ) : B m ⊆ A → m ∈ possible_m_values := by
  sorry

end B_subset_A_implies_m_values_l118_118152


namespace find_number_subtracted_l118_118168

theorem find_number_subtracted (x : ℕ) (h : 88 - x = 54) : x = 34 := by
  sorry

end find_number_subtracted_l118_118168


namespace days_to_finish_together_l118_118974

-- Define the work rate of B
def work_rate_B : ℚ := 1 / 12

-- Define the work rate of A
def work_rate_A : ℚ := 2 * work_rate_B

-- Combined work rate of A and B
def combined_work_rate : ℚ := work_rate_A + work_rate_B

-- Prove that the number of days required for A and B to finish the work together is 4
theorem days_to_finish_together : (1 / combined_work_rate) = 4 := 
by
  sorry

end days_to_finish_together_l118_118974


namespace regular_icosahedron_edges_l118_118403

-- Define what a regular icosahedron is
def is_regular_icosahedron (P : Type) := -- Definition placeholder for a regular icosahedron
  sorry

-- Define the function that counts edges of a polyhedron
def count_edges (P : Type) [is_regular_icosahedron P] : ℕ :=
  sorry

-- The proof statement
theorem regular_icosahedron_edges (P : Type) [h : is_regular_icosahedron P] : count_edges P = 30 :=
  sorry

end regular_icosahedron_edges_l118_118403


namespace find_k_if_lines_parallel_l118_118815

theorem find_k_if_lines_parallel (k : ℝ) : (∀ x y, y = 5 * x + 3 → y = (3 * k) * x + 7) → k = 5 / 3 :=
by {
  sorry
}

end find_k_if_lines_parallel_l118_118815


namespace sum_of_solutions_eq_35_over_3_l118_118865

theorem sum_of_solutions_eq_35_over_3 (a b : ℝ) 
  (h1 : 2 * a + b = 14) (h2 : a + 2 * b = 21) : 
  a + b = 35 / 3 := 
by
  sorry

end sum_of_solutions_eq_35_over_3_l118_118865


namespace calc_hash_2_5_3_l118_118021

def operation_hash (a b c : ℝ) : ℝ := b^2 - 4*a*c

theorem calc_hash_2_5_3 : operation_hash 2 5 3 = 1 := by
  sorry

end calc_hash_2_5_3_l118_118021


namespace greatest_ABCBA_l118_118489

/-
We need to prove that the greatest possible integer of the form AB,CBA 
that is both divisible by 11 and by 3, with A, B, and C being distinct digits, is 96569.
-/

theorem greatest_ABCBA (A B C : ℕ) 
  (h1 : A ≠ B ∧ B ≠ C ∧ A ≠ C) 
  (h2 : 1 ≤ A ∧ A ≤ 9 ∧ 0 ≤ B ∧ B ≤ 9 ∧ 0 ≤ C ∧ C ≤ 9) 
  (h3 : 10001 * A + 1010 * B + 100 * C < 100000) 
  (h4 : 2 * A - 2 * B + C ≡ 0 [MOD 11])
  (h5 : (2 * A + 2 * B + C) % 3 = 0) : 
  10001 * A + 1010 * B + 100 * C ≤ 96569 :=
sorry

end greatest_ABCBA_l118_118489


namespace hyperbola_other_asymptote_l118_118758

-- Define the problem conditions
def one_asymptote (x y : ℝ) : Prop := y = 2 * x
def foci_x_coordinate : ℝ := -4

-- Define the equation of the other asymptote
def other_asymptote (x y : ℝ) : Prop := y = -2 * x - 16

-- The statement to be proved
theorem hyperbola_other_asymptote : 
  (∀ x y, one_asymptote x y) → (∀ x, x = -4 → ∃ y, ∃ C, other_asymptote x y ∧ y = C + -2 * x - 8) :=
by
  sorry

end hyperbola_other_asymptote_l118_118758


namespace elimination_of_3_cliques_l118_118987

open Finset

variable {V : Type} [DecidableEq V]

-- Define what it means to be a k-clique
def is_k_clique (G : SimpleGraph V) (k : ℕ) (C : Finset V) : Prop :=
  C.card = k ∧ ∀ (a b : V), a ∈ C → b ∈ C → a ≠ b → G.adj a b

-- Assume conditions: Every two 3-cliques share at least one vertex and no 5-cliques exist
variable (G : SimpleGraph V)
variable (H1 : ∀ C1 C2 : Finset V, is_k_clique G 3 C1 → is_k_clique G 3 C2 → (C1 ∩ C2).nonempty)
variable (H2 : ¬ ∃ C : Finset V, is_k_clique G 5 C)

-- The proof goal: There exist at most two vertices whose removal eliminates all 3-cliques
theorem elimination_of_3_cliques (G : SimpleGraph V) (H1 : ∀ C1 C2 : Finset V, is_k_clique G 3 C1 → is_k_clique G 3 C2 → (C1 ∩ C2).nonempty)
  (H2 : ¬ ∃ C : Finset V, is_k_clique G 5 C) : ∃ (S : Finset V), S.card ≤ 2 ∧ ∀ C : Finset V, is_k_clique G 3 C → (C ∩ S).nonempty :=
sorry

end elimination_of_3_cliques_l118_118987


namespace total_animals_correct_l118_118357

section 
variable 
  (snakes : ℕ)
  (arctic_foxes : ℕ)
  (leopards : ℕ)
  (bee_eaters : ℕ)
  (cheetahs : ℕ)
  (alligators : ℕ)
  (total : ℕ)

-- Define the initial counts
def snakes := 100
def arctic_foxes := 80
def leopards := 20

-- Define the derived counts based on conditions
def bee_eaters := 10 * leopards
def cheetahs := snakes / 2
def alligators := 2 * (arctic_foxes + leopards)

-- Define the total number of animals based on the derived counts
def total := alligators + bee_eaters + snakes + arctic_foxes + leopards + cheetahs

-- Proof objective
theorem total_animals_correct : total = 670 := by 
  sorry
end

end total_animals_correct_l118_118357


namespace triangle_area_l118_118275

theorem triangle_area
  (area_WXYZ : ℝ)
  (side_small_squares : ℝ)
  (AB_eq_AC : ℝ)
  (A_coincides_with_O : ℝ)
  (area : ℝ) :
  area_WXYZ = 49 →  -- The area of square WXYZ is 49 cm^2
  side_small_squares = 2 → -- Sides of the smaller squares are 2 cm long
  AB_eq_AC = AB_eq_AC → -- Triangle ABC is isosceles with AB = AC
  A_coincides_with_O = A_coincides_with_O → -- A coincides with O
  area = 45 / 4 := -- The area of triangle ABC is 45/4 cm^2
by
  sorry

end triangle_area_l118_118275


namespace cafeteria_extra_apples_l118_118659

-- Define the conditions from the problem
def red_apples : ℕ := 33
def green_apples : ℕ := 23
def students : ℕ := 21

-- Define the total apples and apples given out based on the conditions
def total_apples : ℕ := red_apples + green_apples
def apples_given : ℕ := students

-- Define the extra apples as the difference between total apples and apples given out
def extra_apples : ℕ := total_apples - apples_given

-- The theorem to prove that the number of extra apples is 35
theorem cafeteria_extra_apples : extra_apples = 35 :=
by
  -- The structure of the proof would go here, but is omitted
  sorry

end cafeteria_extra_apples_l118_118659


namespace sum_of_reciprocals_l118_118636

-- Define the conditions formally
def total_students : ℕ := 1000
def num_classes : ℕ := 35

-- Class sizes as a list of natural numbers
variable (class_sizes : Fin num_classes → ℕ)
-- Constraint ensuring the total number of students across all classes
axiom class_sizes_sum : ∑ i, class_sizes i = total_students

-- Main theorem stating that the sum of the reciprocals of class sizes is 35
theorem sum_of_reciprocals (class_sizes_proper : ∀ i, class_sizes i > 0):
  (∑ i, (1 : ℝ) / (class_sizes i)) = 35 := by
  sorry

end sum_of_reciprocals_l118_118636


namespace simplify_and_evaluate_expression_l118_118444

theorem simplify_and_evaluate_expression (a : ℝ) (h : a^2 + 2 * a - 8 = 0) :
  ((a^2 - 4) / (a^2 - 4 * a + 4) - a / (a - 2)) / (a^2 + 2 * a) / (a - 2) = 1 / 4 := 
by 
  sorry

end simplify_and_evaluate_expression_l118_118444


namespace right_triangle_acute_angles_l118_118136

theorem right_triangle_acute_angles (α β : ℝ) 
  (h1 : α + β = 90)
  (h2 : ∀ (δ1 δ2 ε1 ε2 : ℝ), δ1 + ε1 = 135 ∧ δ1 / ε1 = 13 / 17 
                       ∧ ε2 = 180 - ε1 ∧ δ2 = 180 - δ1) :
  α = 63 ∧ β = 27 := 
  sorry

end right_triangle_acute_angles_l118_118136


namespace log_expression_evaluation_l118_118529

theorem log_expression_evaluation (x y : ℝ) (hx : 0 < x) (hy : 0 < y) :
  (Real.log (x^2) / Real.log (y^8)) * (Real.log (y^3) / Real.log (x^7)) * (Real.log (x^4) / Real.log (y^5)) * (Real.log (y^5) / Real.log (x^4)) * (Real.log (x^7) / Real.log (y^3)) =
  (1 / 4) * (Real.log x / Real.log y) := 
by
  sorry

end log_expression_evaluation_l118_118529


namespace lines_perpendicular_l118_118711

theorem lines_perpendicular (A1 B1 C1 A2 B2 C2 : ℝ) (h : A1 * A2 + B1 * B2 = 0) :
  ∃(x y : ℝ), A1 * x + B1 * y + C1 = 0 ∧ A2 * x + B2 * y + C2 = 0 → A1 * A2 + B1 * B2 = 0 :=
by
  sorry

end lines_perpendicular_l118_118711


namespace sum_of_consecutive_integers_bound_sqrt_40_l118_118714

theorem sum_of_consecutive_integers_bound_sqrt_40 (a b : ℤ) (h₁ : a < Real.sqrt 40) (h₂ : Real.sqrt 40 < b) (h₃ : b = a + 1) : a + b = 13 :=
by
  sorry

end sum_of_consecutive_integers_bound_sqrt_40_l118_118714


namespace div_expression_l118_118229

theorem div_expression : 180 / (12 + 13 * 2) = 90 / 19 := 
  sorry

end div_expression_l118_118229


namespace count_four_digit_numbers_l118_118264

open Int

def is_digit (n : ℕ) : Prop := n >= 0 ∧ n <= 9

def valid_combination (a b c d : ℕ) : Prop :=
  is_digit a ∧ is_digit b ∧ is_digit c ∧ is_digit d ∧
  a ≠ 0 ∧
  a + b + c + d = 10 ∧
  (a + c) - (b + d) % 11 = 0

theorem count_four_digit_numbers :
  ∃ n, n > 0 ∧ ∀ a b c d : ℕ, valid_combination a b c d → n = sorry :=
sorry

end count_four_digit_numbers_l118_118264


namespace expressions_positive_l118_118910

-- Definitions based on given conditions
def A := 2.5
def B := -0.8
def C := -2.2
def D := 1.1
def E := -3.1

-- The Lean statement to prove the necessary expressions are positive numbers.

theorem expressions_positive :
  (B + C) / E = 0.97 ∧
  B * D - A * C = 4.62 ∧
  C / (A * B) = 1.1 :=
by
  -- Assuming given conditions and steps to prove the theorem.
  sorry

end expressions_positive_l118_118910


namespace nancy_first_counted_l118_118602

theorem nancy_first_counted (x : ℤ) (h : (x + 12 + 1 + 12 + 7 + 3 + 8) / 6 = 7) : x = -1 := 
by 
  sorry

end nancy_first_counted_l118_118602


namespace no_such_function_l118_118994

theorem no_such_function :
  ¬ ∃ f : ℝ → ℝ, ∀ x y : ℝ, f (x + f y) = x - y :=
by
  sorry

end no_such_function_l118_118994


namespace cos2alpha_minus_sin2alpha_l118_118424

theorem cos2alpha_minus_sin2alpha (α : ℝ) (h1 : α ∈ Set.Icc (-π/2) 0) 
  (h2 : (Real.sin (3 * α)) / (Real.sin α) = 13 / 5) :
  Real.cos (2 * α) - Real.sin (2 * α) = (3 + Real.sqrt 91) / 10 :=
sorry

end cos2alpha_minus_sin2alpha_l118_118424


namespace sum_of_numbers_l118_118472

theorem sum_of_numbers : 1234 + 2341 + 3412 + 4123 = 11110 :=
by
  sorry

end sum_of_numbers_l118_118472


namespace stella_weeks_l118_118435

-- Define the constants used in the conditions
def rolls_per_bathroom_per_day : ℕ := 1
def bathrooms : ℕ := 6
def days_per_week : ℕ := 7
def rolls_per_pack : ℕ := 12
def packs_bought : ℕ := 14

-- Define the total number of rolls Stella uses per day and per week
def rolls_per_day := rolls_per_bathroom_per_day * bathrooms
def rolls_per_week := rolls_per_day * days_per_week

-- Calculate the total number of rolls bought
def total_rolls_bought := packs_bought * rolls_per_pack

-- Calculate the number of weeks Stella bought toilet paper for
def weeks := total_rolls_bought / rolls_per_week

theorem stella_weeks : weeks = 4 := by
  sorry

end stella_weeks_l118_118435


namespace max_min_values_of_function_l118_118041

theorem max_min_values_of_function :
  (∀ x, 0 ≤ 2 * Real.sin x + 2 ∧ 2 * Real.sin x + 2 ≤ 4) ↔ (∃ x, 2 * Real.sin x + 2 = 0) ∧ (∃ y, 2 * Real.sin y + 2 = 4) :=
by
  sorry

end max_min_values_of_function_l118_118041


namespace find_x_minus_y_l118_118569

theorem find_x_minus_y (x y : ℝ) (h1 : |x| + x - y = 14) (h2 : x + |y| + y = 6) : x - y = 8 :=
sorry

end find_x_minus_y_l118_118569


namespace parallel_lines_slope_l118_118812

theorem parallel_lines_slope (k : ℝ) : 
  (∀ x : ℝ, 5 * x + 3 = (3 * k) * x + 7) → k = 5 / 3 := 
sorry

end parallel_lines_slope_l118_118812


namespace arithmetic_sequence_problem_l118_118286

theorem arithmetic_sequence_problem (S : ℕ → ℝ) (a : ℕ → ℝ)
  (h1 : S 6 = 36)
  (h2 : S n = 324)
  (h3 : S (n - 6) = 144) :
  n = 18 := by
  sorry

end arithmetic_sequence_problem_l118_118286


namespace dice_probability_five_or_six_l118_118474

theorem dice_probability_five_or_six :
  let outcomes := 36
  let favorable := 18
  let probability := favorable / outcomes
  probability = 1 / 2 :=
by
  sorry

end dice_probability_five_or_six_l118_118474


namespace quadratic_min_value_l118_118182

theorem quadratic_min_value :
  ∃ x : ℝ, (∀ y : ℝ, y = x^2 - 4 * x + 7 → y ≥ 3) ∧ (x = 2 → (x^2 - 4 * x + 7 = 3)) :=
by
  sorry

end quadratic_min_value_l118_118182


namespace smallest_percent_increase_l118_118769

-- Define the values of each question
def question_values : List ℕ :=
  [150, 250, 400, 600, 1100, 2300, 4700, 9500, 19000, 38000, 76000, 150000, 300000, 600000, 1200000]

-- Define a function to calculate the percent increase between two questions
def percent_increase (v1 v2 : ℕ) : Float :=
  ((v2 - v1).toFloat / v1.toFloat) * 100

-- Define the specific question transitions and their percent increases
def percent_increase_1_to_4 : Float := percent_increase question_values[0] question_values[3]  -- Question 1 to 4
def percent_increase_2_to_6 : Float := percent_increase question_values[1] question_values[5]  -- Question 2 to 6
def percent_increase_5_to_10 : Float := percent_increase question_values[4] question_values[9]  -- Question 5 to 10
def percent_increase_9_to_15 : Float := percent_increase question_values[8] question_values[14] -- Question 9 to 15

-- Prove that the smallest percent increase is from Question 1 to 4
theorem smallest_percent_increase :
  percent_increase_1_to_4 < percent_increase_2_to_6 ∧
  percent_increase_1_to_4 < percent_increase_5_to_10 ∧
  percent_increase_1_to_4 < percent_increase_9_to_15 :=
by
  sorry

end smallest_percent_increase_l118_118769


namespace correct_operation_is_multiplication_by_3_l118_118963

theorem correct_operation_is_multiplication_by_3
  (x : ℝ)
  (percentage_error : ℝ)
  (correct_result : ℝ := 3 * x)
  (incorrect_result : ℝ := x / 5)
  (error_percentage : ℝ := (correct_result - incorrect_result) / correct_result * 100) :
  percentage_error = 93.33333333333333 → correct_result / x = 3 :=
by
  intro h
  sorry

end correct_operation_is_multiplication_by_3_l118_118963


namespace geometric_progression_condition_l118_118101

variables (a b c : ℝ) (k n p : ℕ)

theorem geometric_progression_condition :
  (a / b) ^ (k - p) = (a / c) ^ (k - n) :=
sorry

end geometric_progression_condition_l118_118101


namespace remainder_b22_div_35_l118_118425

def b_n (n : ℕ) : Nat :=
  ((List.range (n + 1)).drop 1).foldl (λ acc k => acc * 10^(Nat.digits 10 k).length + k) 0

theorem remainder_b22_div_35 : (b_n 22) % 35 = 17 :=
  sorry

end remainder_b22_div_35_l118_118425


namespace log_conversion_l118_118270

theorem log_conversion (a b : ℝ) (h1 : a = Real.log 225 / Real.log 8) (h2 : b = Real.log 15 / Real.log 2) : a = (2 * b) / 3 := 
sorry

end log_conversion_l118_118270


namespace circle_equation_through_points_l118_118091

-- Line and circle definitions
def line1 (x y : ℝ) : Prop := 2 * x - y + 1 = 0
def circle1 (x y : ℝ) : Prop := x^2 + y^2 - 2 * x - 15 = 0

-- Intersection point definition
def intersection_point (x y : ℝ) : Prop := line1 x y ∧ circle1 x y

-- Revised circle equation
def circle_equation (x y : ℝ) : Prop := x^2 + y^2 + 28 * x - 15 * y = 0

-- Proof statement
theorem circle_equation_through_points :
  (∀ x y, intersection_point x y → circle_equation x y) ∧ circle_equation 0 0 :=
sorry

end circle_equation_through_points_l118_118091


namespace table_fill_impossible_l118_118242

/-- Proposition: Given a 7x3 table filled with 0s and 1s, it is impossible to prevent any 2x2 submatrix from having all identical numbers. -/
theorem table_fill_impossible : 
  ¬ ∃ (M : (Fin 7) → (Fin 3) → Fin 2), 
      ∀ i j, (i < 6) → (j < 2) → 
              (M i j = M i.succ j) ∨ 
              (M i j = M i j.succ) ∨ 
              (M i j = M i.succ j.succ) ∨ 
              (M i.succ j = M i j.succ → M i j = M i.succ j.succ) :=
sorry

end table_fill_impossible_l118_118242


namespace female_officers_count_l118_118938

theorem female_officers_count (total_officers_on_duty : ℕ) 
  (percent_female_on_duty : ℝ) 
  (female_officers_on_duty : ℕ) 
  (half_of_total_on_duty_is_female : total_officers_on_duty / 2 = female_officers_on_duty) 
  (percent_condition : percent_female_on_duty * (total_officers_on_duty / 2) = female_officers_on_duty) :
  total_officers_on_duty = 250 :=
by
  sorry

end female_officers_count_l118_118938


namespace min_time_to_one_ball_l118_118755

-- Define the problem in Lean
theorem min_time_to_one_ball (n : ℕ) (h : n = 99) : 
  ∃ T : ℕ, T = 98 ∧ ∀ t < T, ∃ ball_count : ℕ, ball_count > 1 :=
by
  -- Since we are not providing the proof, we use "sorry"
  sorry

end min_time_to_one_ball_l118_118755


namespace purely_imaginary_x_value_l118_118565

theorem purely_imaginary_x_value (x : ℝ) (h1 : x^2 - 1 = 0) (h2 : x + 1 ≠ 0) : x = 1 :=
by
  sorry

end purely_imaginary_x_value_l118_118565


namespace multiple_of_Jills_age_l118_118762

theorem multiple_of_Jills_age (m : ℤ) : 
  ∀ (J R F : ℤ),
  J = 20 →
  F = 40 →
  R = m * J + 5 →
  (R + 15) - (J + 15) = (F + 15) - 30 →
  m = 2 :=
by
  intros J R F hJ hF hR hDiff
  sorry

end multiple_of_Jills_age_l118_118762


namespace count_divisors_not_divisible_by_3_l118_118405

theorem count_divisors_not_divisible_by_3 :
  let n := 180 
  let a_max := 2 
  let b_max := 2 
  let c_max := 1 
  let condition := λ (a b c : ℕ), a ≤ a_max ∧ b = 0 ∧ c ≤ c_max 
  (∑ a in finset.range (a_max + 1), ∑ c in finset.range (c_max + 1), if condition a 0 c then 1 else 0) = 6 :=
begin
  let n := 180,
  let a_max := 2,
  let b_max := 2,
  let c_max := 1,
  let condition := λ (a b c : ℕ), a ≤ a_max ∧ b = 0 ∧ c ≤ c_max,
  let valid_combinations := ∑ a in finset.range (a_max + 1), ∑ c in finset.range (c_max + 1), if condition a 0 c then 1 else 0,
  split,
  sorry
end

end count_divisors_not_divisible_by_3_l118_118405


namespace barycentric_vector_identity_l118_118761

variables {A B C X : Type} [AddCommGroup X] [Module ℝ X]
variables (α β γ : ℝ) (A B C X : X)

-- Defining the barycentric coordinates condition
axiom barycentric_coords : α • A + β • B + γ • C = X

-- Additional condition that sum of coordinates is 1
axiom sum_coords : α + β + γ = 1

-- The theorem to prove
theorem barycentric_vector_identity :
  (X - A) = β • (B - A) + γ • (C - A) :=
sorry

end barycentric_vector_identity_l118_118761


namespace water_level_height_l118_118507

/-- Problem: An inverted frustum with a bottom diameter of 12 and height of 18, filled with water, 
    is emptied into another cylindrical container with a bottom diameter of 24. Assuming the 
    cylindrical container is sufficiently tall, the height of the water level in the cylindrical container -/
theorem water_level_height
  (V_cone : ℝ := (1 / 3) * π * (12 / 2) ^ 2 * 18)
  (R_cyl : ℝ := 24 / 2)
  (H_cyl : ℝ) :
  V_cone = π * R_cyl ^ 2 * H_cyl →
  H_cyl = 1.5 :=
by 
  sorry

end water_level_height_l118_118507


namespace difference_fewer_children_than_adults_l118_118793

theorem difference_fewer_children_than_adults : 
  ∀ (C S : ℕ), 2 * C = S → 58 + C + S = 127 → (58 - C = 35) :=
by
  intros C S h1 h2
  sorry

end difference_fewer_children_than_adults_l118_118793


namespace tan_sum_l118_118861

theorem tan_sum (x y : ℝ) (h1 : Real.sin x + Real.sin y = 85 / 65) (h2 : Real.cos x + Real.cos y = 60 / 65) :
  Real.tan x + Real.tan y = 17 / 12 :=
sorry

end tan_sum_l118_118861


namespace subtracted_value_l118_118620

theorem subtracted_value (s : ℕ) (h : s = 4) (x : ℕ) (h2 : (s + s^2 - x = 4)) : x = 16 :=
by
  sorry

end subtracted_value_l118_118620


namespace candy_cost_l118_118960

theorem candy_cost (C : ℝ) 
  (h1 : 20 * C + 80 * 5 = 100 * 6) : 
  C = 10 := 
by
  sorry

end candy_cost_l118_118960


namespace liquid_mixture_ratio_l118_118855

theorem liquid_mixture_ratio (m1 m2 m3 : ℝ) (ρ1 ρ2 ρ3 : ℝ) (k : ℝ)
  (hρ1 : ρ1 = 6 * k) (hρ2 : ρ2 = 3 * k) (hρ3 : ρ3 = 2 * k)
  (h_condition : m1 ≥ 3.5 * m2)
  (h_arith_mean : (m1 + m2 + m3) / (m1 / ρ1 + m2 / ρ2 + m3 / ρ3) = (ρ1 + ρ2 + ρ3) / 3) :
    ∃ x y : ℝ, x ≤ 2/7 ∧ (4 * x + 15 * y = 7) := sorry

end liquid_mixture_ratio_l118_118855


namespace emily_spent_12_dollars_l118_118675

variables (cost_per_flower : ℕ)
variables (roses : ℕ)
variables (daisies : ℕ)

def total_flowers : ℕ := roses + daisies

def total_cost : ℕ := total_flowers * cost_per_flower

theorem emily_spent_12_dollars (h1 : cost_per_flower = 3)
                              (h2 : roses = 2)
                              (h3 : daisies = 2) :
  total_cost cost_per_flower roses daisies = 12 :=
by
  simp [total_cost, total_flowers, h1, h2, h3]
  sorry

end emily_spent_12_dollars_l118_118675


namespace tangent_eq_tangent_intersect_other_l118_118706

noncomputable def curve (x : ℝ) : ℝ := 3 * x^4 - 2 * x^3 - 9 * x^2 + 4

/-- Equation of the tangent line to curve C at x = 1 is y = -12x + 8 --/
theorem tangent_eq (tangent_line : ℝ → ℝ) (x : ℝ):
  tangent_line x = -12 * x + 8 :=
by
  sorry

/-- Apart from the tangent point (1, -4), the tangent line intersects the curve C at the points
    (-2, 32) and (2 / 3, 0) --/
theorem tangent_intersect_other (tangent_line : ℝ → ℝ) x:
  curve x = tangent_line x →
  (x = -2 ∧ curve (-2) = 32) ∨ (x = 2 / 3 ∧ curve (2 / 3) = 0) :=
by
  sorry

end tangent_eq_tangent_intersect_other_l118_118706


namespace cost_of_adult_ticket_l118_118156

theorem cost_of_adult_ticket (c : ℝ) 
  (h₁ : 2 * (c + 6) + 3 * c = 77)
  : c + 6 = 19 :=
sorry

end cost_of_adult_ticket_l118_118156


namespace find_all_solutions_l118_118099

def is_solution (f : ℕ → ℝ) : Prop :=
  (∀ n ≥ 1, f (n + 1) ≥ f n) ∧
  (∀ m n, Nat.gcd m n = 1 → f (m * n) = f m * f n)

theorem find_all_solutions :
  ∀ f : ℕ → ℝ, is_solution f →
    (∀ n, f n = 0) ∨ (∃ a ≥ 0, ∀ n, f n = n ^ a) :=
sorry

end find_all_solutions_l118_118099


namespace percent_problem_l118_118573

theorem percent_problem (x : ℝ) (h : 0.3 * 0.15 * x = 18) : 0.15 * 0.3 * x = 18 :=
by
  sorry

end percent_problem_l118_118573


namespace optimalBananaBuys_l118_118340

noncomputable def bananaPrices : List ℕ := [1, 5, 1, 6, 7, 8, 1, 8, 7, 2, 7, 8, 1, 9, 2, 8, 7, 1]

def days := List.range 18

def computeOptimalBuys : List ℕ :=
  sorry -- Implement the logic to compute the optimal number of bananas to buy each day.

theorem optimalBananaBuys :
  computeOptimalBuys = [4, 0, 0, 3, 0, 0, 7, 0, 0, 1, 0, 0, 4, 0, 0, 3, 0, 1] :=
sorry

end optimalBananaBuys_l118_118340


namespace volume_calculation_l118_118429

noncomputable def enclosedVolume : Real :=
  let f (x y z : Real) : Real := x^2016 + y^2016 + z^2
  let V : Real := 360
  V

theorem volume_calculation : enclosedVolume = 360 :=
by
  sorry

end volume_calculation_l118_118429


namespace solve_problem_l118_118483

def problem_statement : Prop :=
  ∀ (n1 n2 c1 : ℕ) (C : ℕ),
  n1 = 18 → 
  c1 = 60 → 
  n2 = 216 →
  n1 * c1 = n2 * C →
  C = 5

theorem solve_problem : problem_statement := by
  intros n1 n2 c1 C h1 h2 h3 h4
  -- Proof steps go here
  sorry

end solve_problem_l118_118483


namespace simplify_and_evaluate_equals_l118_118443

noncomputable def simplify_and_evaluate (a : ℝ) : ℝ :=
  (a^2 - 4) / (a^2 - 4 * a + 4) - a / (a - 2) / (a^2 + 2 * a) / (a - 2)

theorem simplify_and_evaluate_equals (a : ℝ) (h : a^2 + 2 * a - 8 = 0) : 
  simplify_and_evaluate a = 1 / 4 :=
sorry

end simplify_and_evaluate_equals_l118_118443


namespace solve_quadratic_equation_l118_118765

theorem solve_quadratic_equation (x : ℝ) : 2 * (x + 1) ^ 2 - 49 = 1 ↔ (x = 4 ∨ x = -6) := 
sorry

end solve_quadratic_equation_l118_118765


namespace initial_integers_is_three_l118_118796

def num_initial_integers (n m : Int) : Prop :=
  3 * n + m = 17 ∧ 2 * m + n = 23

theorem initial_integers_is_three {n m : Int} (h : num_initial_integers n m) : n = 3 :=
by
  sorry

end initial_integers_is_three_l118_118796


namespace total_amount_after_refunds_l118_118524

def individual_bookings : ℕ := 12000
def group_bookings : ℕ := 16000
def refunds : ℕ := 1600

theorem total_amount_after_refunds : 
  individual_bookings + group_bookings - refunds = 26400 :=
by 
  -- The proof goes here.
  sorry

end total_amount_after_refunds_l118_118524


namespace angle_bisectors_concurrent_l118_118481

open EuclideanGeometry

-- Definitions based on conditions
def cyclic_quadrilateral (A B C D : Point) : Prop :=
  ∃ (circle : Circle), circle.on_circle A ∧ circle.on_circle B ∧ circle.on_circle C ∧ circle.on_circle D

def midpoint (P Q M : Point) : Prop :=
  ∃ (R : Point), dist R P = dist R Q ∧ dist P M = dist Q M

axiom intersect (A B C D E F M N : Point)
  (hcyclic: cyclic_quadrilateral A B C D)
  (hE: line_intersects A B C D at E)
  (hF: line_intersects A D B C at F)
  (hM: midpoint B D M)
  (hN: midpoint E F N) :
  concurrency_of_angle_bisectors A E D A F B M N

-- The main theorem which we need to prove
theorem angle_bisectors_concurrent (A B C D E F M N : Point)
  (hcyclic: cyclic_quadrilateral A B C D)
  (hE: line_intersects A B C D at E)
  (hF: line_intersects A D B C at F)
  (hM: midpoint B D M)
  (hN: midpoint E F N):
  concurrency_of_angle_bisectors A E D A F B M N :=
sorry

end angle_bisectors_concurrent_l118_118481


namespace sum_of_numbers_l118_118470

theorem sum_of_numbers : 1234 + 2341 + 3412 + 4123 = 11110 := by
  sorry

end sum_of_numbers_l118_118470


namespace problem1_l118_118341

variables (m n : ℝ)

axiom cond1 : 4 * m + n = 90
axiom cond2 : 2 * m - 3 * n = 10

theorem problem1 : (m + 2 * n) ^ 2 - (3 * m - n) ^ 2 = -900 := sorry

end problem1_l118_118341


namespace part1_part2_l118_118848

-- Define the initial conditions and the given inequality.
def condition1 (x a : ℝ) : Prop := x^2 - 4 * a * x + 3 * a^2 < 0
def condition2 (m : ℝ) (x : ℝ) : Prop := x = (1/2)^(m - 1) ∧ 1 < m ∧ m < 2

-- Definitions of the correct ranges
def range_x (x : ℝ) : Prop := 1/2 < x ∧ x < 3/4
def range_a (a : ℝ) : Prop := 1/3 ≤ a ∧ a ≤ 1/2

-- Mathematical equivalent proof problem
theorem part1 {x : ℝ} (h1 : condition1 x (1/4)) (h2 : ∃ (m : ℝ), condition2 m x) : range_x x :=
sorry

theorem part2 {a : ℝ} (h : ∀ x : ℝ, (1/2 < x ∧ x < 1) → condition1 x a) : range_a a :=
sorry

end part1_part2_l118_118848


namespace least_positive_t_geometric_progression_l118_118230

open Real

theorem least_positive_t_geometric_progression (α : ℝ) (hα1 : 0 < α) (hα2 : α < π / 2) : 
  ∃ t : ℕ, ∀ t' : ℕ, (t' > 0) → 
  (|arcsin (sin (t' * α)) - 8 * α| = 0) → t = 8 :=
by
  sorry

end least_positive_t_geometric_progression_l118_118230


namespace product_of_numbers_l118_118652

theorem product_of_numbers (x y : ℝ) (h1 : x + y = 21) (h2 : x^2 + y^2 = 527) : x * y = -43.05 := by
  sorry

end product_of_numbers_l118_118652


namespace find_a_l118_118138

-- Define the polynomial expansion term conditions
def binomial_coefficient (n k : ℕ) := Nat.choose n k

def fourth_term_coefficient (x a : ℝ) : ℝ :=
  binomial_coefficient 9 3 * x^6 * a^3

theorem find_a (a : ℝ) (x : ℝ) (h : fourth_term_coefficient x a = 84) : a = 1 :=
by
  unfold fourth_term_coefficient at h
  sorry

end find_a_l118_118138


namespace percentage_first_less_third_l118_118347

variable (A B C : ℝ)

theorem percentage_first_less_third :
  B = 0.58 * C → B = 0.8923076923076923 * A → (100 - (A / C * 100)) = 35 :=
by
  intros h₁ h₂
  sorry

end percentage_first_less_third_l118_118347


namespace aquarium_pufferfish_problem_l118_118921

/-- Define the problem constants and equations -/
theorem aquarium_pufferfish_problem :
  ∃ (P S : ℕ), S = 5 * P ∧ S + P = 90 ∧ P = 15 :=
by
  sorry

end aquarium_pufferfish_problem_l118_118921


namespace solve_system_of_equations_l118_118446

def system_of_equations(x y z: ℝ): Prop :=
  (x * y + 2 * x * z + 3 * y * z = -6) ∧
  (x^2 * y^2 + 4 * x^2 * z^2 - 9 * y^2 * z^2 = 36) ∧
  (x^3 * y^3 + 8 * x^3 * z^3 + 27 * y^3 * z^3 = -216)

theorem solve_system_of_equations :
  ∀ (x y z: ℝ), system_of_equations x y z ↔
  (y = 0 ∧ x * z = -3) ∨
  (z = 0 ∧ x * y = -6) ∨
  (x = 3 ∧ y = -2 ∨ z = -1) ∨
  (x = -3 ∧ y = 2 ∨ z = 1) :=
by
  sorry

end solve_system_of_equations_l118_118446


namespace remainder_when_sum_divided_by_30_l118_118738

theorem remainder_when_sum_divided_by_30 {c d : ℕ} (p q : ℕ)
  (hc : c = 60 * p + 58)
  (hd : d = 90 * q + 85) :
  (c + d) % 30 = 23 :=
by
  sorry

end remainder_when_sum_divided_by_30_l118_118738


namespace probability_crisp_stops_on_dime_l118_118369

noncomputable def crisp_stops_on_dime_probability : ℚ :=
  let a := (2/3 : ℚ)
  let b := (1/3 : ℚ)
  let a1 := (15/31 : ℚ)
  let b1 := (30/31 : ℚ)
  (2 / 3) * a1 + (1 / 3) * b1

theorem probability_crisp_stops_on_dime :
  crisp_stops_on_dime_probability = 20 / 31 :=
by
  sorry

end probability_crisp_stops_on_dime_l118_118369


namespace brass_total_l118_118775

theorem brass_total (p_cu : ℕ) (p_zn : ℕ) (m_zn : ℕ) (B : ℕ) 
  (h_ratio : p_cu = 13) 
  (h_zn_ratio : p_zn = 7) 
  (h_zn_mass : m_zn = 35) : 
  (h_brass_total :  p_zn / (p_cu + p_zn) * B = m_zn) → B = 100 :=
sorry

end brass_total_l118_118775


namespace jose_fewer_rocks_l118_118144

theorem jose_fewer_rocks (J : ℕ) (H1 : 80 = J + 14) (H2 : J + 20 = 86) (H3 : J < 80) : J = 66 :=
by
  -- Installation of other conditions derived from the proof
  have H_albert_collected : 86 = 80 + 6 := by rfl
  have J_def : J = 86 - 20 := by sorry
  sorry

end jose_fewer_rocks_l118_118144


namespace complement_intersection_l118_118559

open Set

variable (U : Set ℤ) (A B : Set ℤ)

theorem complement_intersection (hU : U = univ)
                               (hA : A = {3, 4})
                               (h_union : A ∪ B = {1, 2, 3, 4}) :
  (U \ A) ∩ B = {1, 2} :=
by
  sorry

end complement_intersection_l118_118559


namespace distance_squared_from_B_to_origin_l118_118575

-- Conditions:
-- 1. the radius of the circle is 10 cm
-- 2. the length of AB is 8 cm
-- 3. the length of BC is 3 cm
-- 4. the angle ABC is a right angle
-- 5. the center of the circle is at the origin
-- a^2 + b^2 is the square of the distance from B to the center of the circle (origin)

theorem distance_squared_from_B_to_origin
  (a b : ℝ)
  (h1 : a^2 + (b + 8)^2 = 100)
  (h2 : (a + 3)^2 + b^2 = 100)
  (h3 : 6 * a - 16 * b = 55) : a^2 + b^2 = 50 :=
sorry

end distance_squared_from_B_to_origin_l118_118575


namespace percent_equivalence_l118_118646

theorem percent_equivalence (x : ℝ) : (0.6 * 0.3 * x - 0.1 * x) / x * 100 = 8 := by
  sorry

end percent_equivalence_l118_118646


namespace motorcycle_travel_distance_l118_118561

noncomputable def motorcycle_distance : ℝ :=
  let t : ℝ := 1 / 2  -- time in hours (30 minutes)
  let v_bus : ℝ := 90  -- speed of the bus in km/h
  let v_motorcycle : ℝ := (2 / 3) * v_bus  -- speed of the motorcycle in km/h
  v_motorcycle * t  -- calculates the distance traveled by the motorcycle in km

theorem motorcycle_travel_distance :
  motorcycle_distance = 30 := by
  sorry

end motorcycle_travel_distance_l118_118561


namespace measure_of_angle_A_l118_118509

theorem measure_of_angle_A (A B : ℝ) (h1 : A + B = 180) (h2 : A = 5 * B) : A = 150 := 
by 
  sorry

end measure_of_angle_A_l118_118509


namespace remaining_bananas_l118_118374

def original_bananas : ℕ := 46
def removed_bananas : ℕ := 5

theorem remaining_bananas : original_bananas - removed_bananas = 41 := by
  sorry

end remaining_bananas_l118_118374


namespace parabola_opens_downwards_l118_118157

theorem parabola_opens_downwards (a : ℝ) (h : ℝ) (k : ℝ) :
  a < 0 → h = 3 → ∃ k, (∀ x, y = a * (x - h) ^ 2 + k → y = -(x - 3)^2 + k) :=
by
  intros ha hh
  use k
  sorry

end parabola_opens_downwards_l118_118157


namespace greatest_ABCBA_divisible_by_13_l118_118075

theorem greatest_ABCBA_divisible_by_13 :
  ∃ A B C : ℕ, A ≠ B ∧ B ≠ C ∧ A ≠ C ∧
  1 ≤ A ∧ A < 10 ∧ 0 ≤ B ∧ B < 10 ∧ 0 ≤ C ∧ C < 10 ∧
  (10000 * A + 1000 * B + 100 * C + 10 * B + A) % 13 = 0 ∧
  (10000 * A + 1000 * B + 100 * C + 10 * B + A) = 95159 :=
by
  sorry

end greatest_ABCBA_divisible_by_13_l118_118075


namespace point_of_tangency_l118_118106

noncomputable def parabola1 (x : ℝ) : ℝ := 2 * x^2 + 10 * x + 14
noncomputable def parabola2 (y : ℝ) : ℝ := 4 * y^2 + 16 * y + 68

theorem point_of_tangency : 
  ∃ (x y : ℝ), parabola1 x = y ∧ parabola2 y = x ∧ x = -9/4 ∧ y = -15/8 :=
by
  -- The proof will show that the point of tangency is (-9/4, -15/8)
  sorry

end point_of_tangency_l118_118106


namespace percent_shaded_of_square_l118_118473

theorem percent_shaded_of_square (side_len : ℤ) (first_layer_side : ℤ) 
(second_layer_outer_side : ℤ) (second_layer_inner_side : ℤ)
(third_layer_outer_side : ℤ) (third_layer_inner_side : ℤ)
(h_side : side_len = 7) (h_first : first_layer_side = 2) 
(h_second_outer : second_layer_outer_side = 5) (h_second_inner : second_layer_inner_side = 3) 
(h_third_outer : third_layer_outer_side = 7) (h_third_inner : third_layer_inner_side = 6) : 
  (4 + (25 - 9) + (49 - 36)) / (side_len * side_len : ℝ) = 33 / 49 :=
by
  -- Sorry is used as we are only required to construct the statement, not the proof.
  sorry

end percent_shaded_of_square_l118_118473


namespace find_m_l118_118836

def ellipse (x y : ℝ) : Prop :=
  (x^2 / 3) + y^2 = 1

def focus1 : ℝ × ℝ := (-Real.sqrt 2, 0)
def focus2 : ℝ × ℝ := (Real.sqrt 2, 0)

def line (x y m : ℝ) : Prop :=
  y = x + m

def intersects (m : ℝ) (x y : ℝ) : Prop :=
  ellipse x y ∧ line x y m

def area_relation (m : ℝ) : Prop :=
  let x := -m -- This might be simplified as it's used to calculate distances
  let dist1 := Real.abs (focus1.fst + m)
  let dist2 := Real.abs (focus2.fst + m)
  dist1 = 2 * dist2

theorem find_m :
  area_relation (-Real.sqrt 2 / 3) :=
by
  sorry

end find_m_l118_118836


namespace arthur_walked_total_miles_l118_118980

def blocks_east := 8
def blocks_north := 15
def blocks_west := 3
def block_length := 1/2

def total_blocks := blocks_east + blocks_north + blocks_west
def total_miles := total_blocks * block_length

theorem arthur_walked_total_miles : total_miles = 13 := by
  sorry

end arthur_walked_total_miles_l118_118980


namespace smallest_possible_positive_value_l118_118362

theorem smallest_possible_positive_value (l w : ℕ) (hl : l > 0) (hw : w > 0) : ∃ x : ℕ, x = w - l + 1 ∧ x = 1 := 
by {
  sorry
}

end smallest_possible_positive_value_l118_118362


namespace percent_increase_l118_118205

theorem percent_increase (original new : ℕ) (h1 : original = 30) (h2 : new = 60) :
  ((new - original) / original) * 100 = 100 := 
by
  sorry

end percent_increase_l118_118205


namespace count_divisible_by_11_with_digits_sum_10_l118_118257

-- Definitions and conditions from step a)
def digits_add_up_to_10 (n : ℕ) : Prop :=
  let ds := n.digits 10 in ds.length = 4 ∧ ds.sum = 10

def divisible_by_11 (n : ℕ) : Prop :=
  let ds := n.digits 10 in
  (ds.nth 0).get_or_else 0 +
  (ds.nth 2).get_or_else 0 -
  (ds.nth 1).get_or_else 0 -
  (ds.nth 3).get_or_else 0 ∣ 11

-- The tuple (question, conditions, answer) formalized in Lean
theorem count_divisible_by_11_with_digits_sum_10 :
  ∃ N : ℕ, N = (
  (Finset.filter (λ x, digits_add_up_to_10 x ∧ divisible_by_11 x)
  (Finset.range 10000)).card
) := 
sorry

end count_divisible_by_11_with_digits_sum_10_l118_118257


namespace students_play_both_l118_118133

def students_total : ℕ := 25
def hockey_players : ℕ := 15
def basketball_players : ℕ := 16
def neither_players : ℕ := 4

theorem students_play_both : 
  (students_total - neither_players) + (hockey_players + basketball_players - students_total + neither_players - students_total) = 10 :=
by 
  sorry

end students_play_both_l118_118133


namespace divisors_not_divisible_by_3_l118_118407

/-!
# Number of divisors of 180 not divisible by 3

In this statement, we prove that the number of positive divisors of 180 that are not divisible by 3 is equal to 6.
-/

def prime_factorization_180 : ℕ → Prop
| 180 := (∀ {p : ℕ}, prime p → p ∣ 180 → p = 2 ∨ p = 3 ∨ p = 5)

theorem divisors_not_divisible_by_3 : 
  { d : ℕ // d ∣ 180 ∧ ( ∀ {p : ℕ}, prime p → p ∣ d → p ≠ 3) } = (6 : ℕ) :=
by {
  sorry
}

end divisors_not_divisible_by_3_l118_118407


namespace cat_daytime_catches_l118_118067

theorem cat_daytime_catches
  (D : ℕ)
  (night_catches : ℕ := 2 * D)
  (total_catches : ℕ := D + night_catches)
  (h : total_catches = 24) :
  D = 8 := by
  sorry

end cat_daytime_catches_l118_118067


namespace percent_profit_l118_118127

theorem percent_profit (C S : ℝ) (h : 55 * C = 50 * S) : 
  100 * ((S - C) / C) = 10 :=
by
  sorry

end percent_profit_l118_118127


namespace pradeep_pass_percentage_l118_118436

variable (marks_obtained : ℕ) (marks_short : ℕ) (max_marks : ℝ)

theorem pradeep_pass_percentage (h1 : marks_obtained = 150) (h2 : marks_short = 25) (h3 : max_marks = 500.00000000000006) :
  ((marks_obtained + marks_short) / max_marks) * 100 = 35 := 
by
  sorry

end pradeep_pass_percentage_l118_118436


namespace value_of_a_l118_118756

theorem value_of_a (a : ℝ) (H1 : A = a) (H2 : B = 1) (H3 : C = a - 3) (H4 : C + B = 0) : a = 2 := by
  sorry

end value_of_a_l118_118756


namespace tadd_2019th_number_l118_118459

def next_start_point (n : ℕ) : ℕ := 
    1 + (n * (2 * 3 + (n - 1) * 9)) / 2

def block_size (n : ℕ) : ℕ := 
    1 + 3 * (n - 1)

def nth_number_said_by_tadd (n : ℕ) (k : ℕ) : ℕ :=
    let block_n := next_start_point n
    block_n + k - 1

theorem tadd_2019th_number :
    nth_number_said_by_tadd 37 2019 = 5979 := 
sorry

end tadd_2019th_number_l118_118459


namespace cost_of_one_jacket_l118_118629

theorem cost_of_one_jacket
  (S J : ℝ)
  (h1 : 10 * S + 20 * J = 800)
  (h2 : 5 * S + 15 * J = 550) : J = 30 :=
sorry

end cost_of_one_jacket_l118_118629


namespace evaluate_difference_of_squares_l118_118818
-- Import necessary libraries

-- Define the specific values for a and b
def a : ℕ := 72
def b : ℕ := 48

-- State the theorem to be proved
theorem evaluate_difference_of_squares : a^2 - b^2 = (a + b) * (a - b) ∧ (a + b) * (a - b) = 2880 := 
by
  -- The proof would go here but should be omitted as per directions
  sorry

end evaluate_difference_of_squares_l118_118818


namespace max_points_on_segment_l118_118190

theorem max_points_on_segment (l d : ℝ) (n : ℕ)
  (hl : l = 1)
  (hcond : ∀ (d : ℝ), 0 < d ∧ d ≤ l → ∀ (segment : Set ℝ), segment ⊆ Icc 0 l → length segment = d → finset.card (finset.image (λ x, x) (finset.filter (λ x, x ∈ segment) (finset.range n))) ≤ 1 + 1000 * d^2) :
  n ≤ 32 :=
begin
  sorry
end

end max_points_on_segment_l118_118190


namespace lying_dwarf_possible_numbers_l118_118943

theorem lying_dwarf_possible_numbers (a_1 a_2 a_3 a_4 a_5 a_6 a_7 : ℕ)
  (h2 : a_2 = a_1) 
  (h3 : a_3 = a_1 + a_2) 
  (h4 : a_4 = a_1 + a_2 + a_3) 
  (h5 : a_5 = a_1 + a_2 + a_3 + a_4) 
  (h6 : a_6 = a_1 + a_2 + a_3 + a_4 + a_5) 
  (h7 : a_7 = a_1 + a_2 + a_3 + a_4 + a_5 + a_6)
  (h_sum : a_1 + a_2 + a_3 + a_4 + a_5 + a_6 + a_7 = 58)
  (h_lied : ∃ (d : ℕ), d ∈ [a_1, a_2, a_3, a_4, a_5, a_6, a_7] ∧ (d ≠ a_1 ∧ d ≠ a_2 ∧ d ≠ a_3 ∧ d ≠ a_4 ∧ d ≠ a_5 ∧ d ≠ a_6 ∧ d ≠ a_7)) : 
  ∃ x : ℕ, x = 13 ∨ x = 26 :=
by
  sorry

end lying_dwarf_possible_numbers_l118_118943


namespace constant_term_in_binomial_expansion_l118_118579

theorem constant_term_in_binomial_expansion : 
  ∀ (x : ℝ) (n : ℕ), (∑ i in Finset.range (n + 1), nat.choose n i) = 64 → 
  (let r := 2 in (nat.choose 6 r) * (3^r) = 135) :=
by
  intros x n sum_eq
  sorry

end constant_term_in_binomial_expansion_l118_118579


namespace mathematicians_correctness_l118_118904

theorem mathematicians_correctness:
  ∀ (w1₁ w1₂ t1₁ t1₂ w2₁ w2₂ t2₁ t2₂: ℕ),
  (w1₁ = 4) ∧ (t1₁ = 7) ∧ (w1₂ = 3) ∧ (t1₂ = 5) →
  (w2₁ = 8) ∧ (t2₁ = 14) ∧ (w2₂ = 3) ∧ (t2₂ = 5) →
  (4/7 : ℚ) < 3/5 →
  (w1₁ + w1₂) / (t1₁ + t1₂) = 7 / 12 ∨ (w2₁ + w2₂) / (t2₁ + t2₂) = 11 / 19 →
  ¬ (w1₁ + w1₂) / (t1₁ + t1₂) < 19 / 35 ∧ (w2₁ + w2₂) / (t2₁ + t2₂) < 19 / 35 :=
by
  sorry

end mathematicians_correctness_l118_118904


namespace squared_expression_l118_118114

variable (x : ℝ)

theorem squared_expression (h : x + 1/x = 3) : x^2 + 1/x^2 = 7 := by
  sorry

end squared_expression_l118_118114


namespace angle_sum_is_180_l118_118273

theorem angle_sum_is_180 (A B C : ℝ) (h_triangle : (A + B + C) = 180) (h_sum : A + B = 90) : C = 90 :=
by
  -- Proof placeholder
  sorry

end angle_sum_is_180_l118_118273


namespace min_value_of_expression_l118_118746

noncomputable def min_val_expr (x y : ℝ) : ℝ :=
  (8 / (x + 1)) + (1 / y)

theorem min_value_of_expression
  (x y : ℝ) 
  (hx : 0 < x) 
  (hy : 0 < y) 
  (hcond : 2 * x + y = 1) :
  min_val_expr x y = (25 / 3) :=
sorry

end min_value_of_expression_l118_118746


namespace angle_bisector_proportion_l118_118131

theorem angle_bisector_proportion
  (p q r : ℝ)
  (u v : ℝ)
  (h1 : 0 < p ∧ 0 < q ∧ 0 < r ∧ 0 < u ∧ 0 < v)
  (h2 : u + v = p)
  (h3 : u * q = v * r) :
  u / p = r / (r + q) :=
sorry

end angle_bisector_proportion_l118_118131


namespace sum_of_numbers_l118_118471

theorem sum_of_numbers : 1234 + 2341 + 3412 + 4123 = 11110 :=
by
  sorry

end sum_of_numbers_l118_118471


namespace smallest_prime_factor_in_C_l118_118441

def smallest_prime_factor_def (n : Nat) : Nat :=
  if n % 2 = 0 then 2 else
  if n % 3 = 0 then 3 else
  sorry /- Define a function to find the smallest prime factor of a number n -/

def is_prime (p : Nat) : Prop :=
  2 ≤ p ∧ ∀ d : Nat, 2 ≤ d → d ∣ p → d = p

def in_set (x : Nat) : Prop :=
  x = 64 ∨ x = 66 ∨ x = 67 ∨ x = 68 ∨ x = 71

theorem smallest_prime_factor_in_C : ∀ x, in_set x → 
  (smallest_prime_factor_def x = 2 ∨ smallest_prime_factor_def x = 67 ∨ smallest_prime_factor_def x = 71) :=
by
  intro x hx
  cases hx with
  | inl hx  => sorry
  | inr hx  => sorry

end smallest_prime_factor_in_C_l118_118441


namespace valid_probabilities_and_invalid_probability_l118_118900

theorem valid_probabilities_and_invalid_probability :
  (let first_box_1 := (4, 7)
       second_box_1 := (3, 5)
       combined_prob_1 := (first_box_1.1 + second_box_1.1) / (first_box_1.2 + second_box_1.2),
       first_box_2 := (8, 14)
       second_box_2 := (3, 5)
       combined_prob_2 := (first_box_2.1 + second_box_2.1) / (first_box_2.2 + second_box_2.2),
       prob_1 := first_box_1.1 / first_box_1.2,
       prob_2 := second_box_2.1 / second_box_2.2
     in (combined_prob_1 = 7 / 12 ∧ combined_prob_2 = 11 / 19) ∧ (19 / 35 < prob_1 ∧ prob_1 < prob_2) → False) :=
by
  sorry

end valid_probabilities_and_invalid_probability_l118_118900


namespace cost_of_adult_ticket_eq_19_l118_118154

variables (X : ℝ)
-- Condition 1: The cost of an adult ticket is $6 more than the cost of a child ticket.
def cost_of_child_ticket : ℝ := X - 6

-- Condition 2: The total cost of the 5 tickets is $77.
axiom total_cost_eq : 2 * X + 3 * (X - 6) = 77

-- Prove that the cost of an adult ticket is 19 dollars
theorem cost_of_adult_ticket_eq_19 (h : total_cost_eq) : X = 19 := 
by
  -- Here we would provide the actual proof steps
  sorry

end cost_of_adult_ticket_eq_19_l118_118154


namespace debby_bottles_per_day_l118_118090

theorem debby_bottles_per_day :
  let total_bottles := 153
  let days := 17
  total_bottles / days = 9 :=
by
  sorry

end debby_bottles_per_day_l118_118090


namespace number_of_allocation_schemes_l118_118498

/-- 
  Given 5 volunteers and 4 projects, each volunteer is assigned to only one project,
  and each project must have at least one volunteer.
  Prove that there are 240 different allocation schemes.
-/
theorem number_of_allocation_schemes (V P : ℕ) (hV : V = 5) (hP : P = 4) 
  (each_volunteer_one_project : ∀ v, ∃ p, v ≠ p) 
  (each_project_at_least_one : ∀ p, ∃ v, v ≠ p) : 
  ∃ n_ways : ℕ, n_ways = 240 :=
by
  sorry

end number_of_allocation_schemes_l118_118498


namespace bus_seating_options_l118_118184

theorem bus_seating_options :
  ∃! (x y : ℕ), 21*x + 10*y = 241 :=
sorry

end bus_seating_options_l118_118184


namespace total_length_remaining_l118_118672

def initial_figure_height : ℕ := 10
def initial_figure_width : ℕ := 7
def top_right_removed : ℕ := 2
def middle_left_removed : ℕ := 2
def bottom_removed : ℕ := 3
def near_top_left_removed : ℕ := 1

def remaining_top_length : ℕ := initial_figure_width - top_right_removed
def remaining_left_length : ℕ := initial_figure_height - middle_left_removed
def remaining_bottom_length : ℕ := initial_figure_width - bottom_removed
def remaining_right_length : ℕ := initial_figure_height - near_top_left_removed

theorem total_length_remaining :
  remaining_top_length + remaining_left_length + remaining_bottom_length + remaining_right_length = 26 := by
  sorry

end total_length_remaining_l118_118672


namespace bestCompletion_is_advantage_l118_118679

-- Defining the phrase and the list of options
def phrase : String := "British students have a language ____ for jobs in the USA and Australia"

def options : List (String × String) := 
  [("A", "chance"), ("B", "ability"), ("C", "possibility"), ("D", "advantage")]

-- Defining the best completion function (using a placeholder 'sorry' for the logic which is not the focus here)
noncomputable def bestCompletion (phrase : String) (options : List (String × String)) : String :=
  "advantage"  -- We assume given the problem that this function correctly identifies 'advantage'

-- Lean theorem stating the desired property
theorem bestCompletion_is_advantage : bestCompletion phrase options = "advantage" :=
by sorry

end bestCompletion_is_advantage_l118_118679


namespace inverse_matrix_l118_118538

-- Define the matrix A
def A : Matrix (Fin 2) (Fin 2) ℤ :=
  ![![7, -2], ![-3, 1]]

-- Define the supposed inverse B
def B : Matrix (Fin 2) (Fin 2) ℤ :=
  ![![1, 2], ![3, 7]]

-- Define the condition that A * B should yield the identity matrix
theorem inverse_matrix :
  A * B = 1 :=
sorry

end inverse_matrix_l118_118538


namespace largest_number_not_sum_of_two_composites_l118_118103

-- Define composite numbers
def is_composite (n : ℕ) : Prop := n > 1 ∧ ∃ d, d ∣ n ∧ d ≠ 1 ∧ d ≠ n

-- Define the problem predicate
def cannot_be_expressed_as_sum_of_two_composites (n : ℕ) : Prop :=
  ¬ ∃ a b : ℕ, is_composite a ∧ is_composite b ∧ n = a + b

theorem largest_number_not_sum_of_two_composites : 
  ∃ n, cannot_be_expressed_as_sum_of_two_composites n ∧ 
  (∀ m, cannot_be_expressed_as_sum_of_two_composites m → m ≤ n) ∧ 
  n = 11 :=
by {
  sorry
}

end largest_number_not_sum_of_two_composites_l118_118103


namespace percent_equivalence_l118_118571

variable (x : ℝ)
axiom condition : 0.30 * 0.15 * x = 18

theorem percent_equivalence :
  0.15 * 0.30 * x = 18 := sorry

end percent_equivalence_l118_118571


namespace water_level_after_opening_l118_118328

-- Let's define the densities and initial height as given
def ρ_water : ℝ := 1000
def ρ_oil : ℝ := 700
def initial_height : ℝ := 40  -- height in cm

-- Final heights after opening the valve (h' denotes final height)
def final_height_water : ℝ := 34

-- Using the principles described
theorem water_level_after_opening :
  ∃ h_oil : ℝ, ρ_water * final_height_water = ρ_oil * h_oil ∧ final_height_water + h_oil = initial_height :=
begin
  use initial_height - final_height_water,
  split,
  {
    field_simp,
    norm_num,
  },
  {
    field_simp,
    norm_num,
  }
end

end water_level_after_opening_l118_118328


namespace proof_problem_l118_118147

-- Triangle and Point Definitions
variables {A B C P : Type}
variables (BC : ℝ) (a b c : ℝ) (PA PB PC : ℝ)

-- Conditions: Triangle ABC with angle A = 90 degrees and P on BC
def is_right_triangle (A B C : Type) (a b c : ℝ) (BC : ℝ) (angleA : ℝ := 90) : Prop :=
a^2 + b^2 = c^2 ∧ c = BC

def on_hypotenuse (P : Type) (BC : ℝ) (PB PC : ℝ) : Prop :=
PB + PC = BC

-- The proof problem
theorem proof_problem (A B C P : Type) 
  (BC : ℝ) (a b c : ℝ) (PA PB PC : ℝ)
  (h1 : is_right_triangle A B C a b c BC)
  (h2 : on_hypotenuse P BC PB PC) :
  (a^2 / PC + b^2 / PB) ≥ (BC^3 / (PA^2 + PB * PC)) := 
sorry

end proof_problem_l118_118147


namespace cabbages_difference_l118_118667

noncomputable def numCabbagesThisYear : ℕ := 4096
noncomputable def numCabbagesLastYear : ℕ := 3969
noncomputable def diffCabbages : ℕ := numCabbagesThisYear - numCabbagesLastYear

theorem cabbages_difference :
  diffCabbages = 127 := by
  sorry

end cabbages_difference_l118_118667


namespace passengers_remaining_after_fourth_stop_l118_118411

theorem passengers_remaining_after_fourth_stop :
  let initial_passengers := 64
  let remaining_fraction := 2 / 3
  (initial_passengers * remaining_fraction * remaining_fraction * remaining_fraction * remaining_fraction = 1024 / 81) :=
by
  let initial_passengers := 64
  let remaining_fraction := 2 / 3
  have H1 : initial_passengers * remaining_fraction = 128 / 3 := sorry
  have H2 : (128 / 3) * remaining_fraction = 256 / 9 := sorry
  have H3 : (256 / 9) * remaining_fraction = 512 / 27 := sorry
  have H4 : (512 / 27) * remaining_fraction = 1024 / 81 := sorry
  exact H4

end passengers_remaining_after_fourth_stop_l118_118411


namespace panthers_score_points_l118_118208

theorem panthers_score_points (C P : ℕ) (h1 : C + P = 34) (h2 : C = P + 14) : P = 10 :=
by
  sorry

end panthers_score_points_l118_118208


namespace parallel_lines_l118_118809

def line1 (x : ℝ) : ℝ := 5 * x + 3
def line2 (x k : ℝ) : ℝ := 3 * k * x + 7

theorem parallel_lines (k : ℝ) : (∀ x : ℝ, line1 x = line2 x k) → k = 5 / 3 := 
by
  intros h_parallel
  sorry

end parallel_lines_l118_118809


namespace correctPropositions_l118_118552

-- Define the conditions and statement as Lean structures.
structure Geometry :=
  (Line : Type)
  (Plane : Type)
  (parallel : Plane → Plane → Prop)
  (parallelLine : Line → Plane → Prop)
  (perpendicular : Plane → Plane → Prop)
  (perpendicularLine : Line → Plane → Prop)
  (subsetLine : Line → Plane → Prop)

-- Main theorem to be proved in Lean 4
theorem correctPropositions (G : Geometry) :
  (∀ (α β : G.Plane) (a : G.Line), (G.parallel α β) → (G.subsetLine a α) → (G.parallelLine a β)) ∧ 
  (∀ (α β : G.Plane) (a : G.Line), (G.perpendicularLine a α) → (G.perpendicularLine a β) → (G.parallel α β)) :=
sorry -- The proof is omitted, as per instructions

end correctPropositions_l118_118552


namespace people_in_group_l118_118896

theorem people_in_group (n : ℕ) 
  (h1 : ∀ (new_weight old_weight : ℕ), old_weight = 70 → new_weight = 110 → (70 * n + (new_weight - old_weight) = 70 * n + 4 * n)) :
  n = 10 :=
sorry

end people_in_group_l118_118896


namespace projection_of_A_onto_Oxz_is_B_l118_118731

structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

def projection_onto_Oxz (A : Point3D) : Point3D :=
  { x := A.x, y := 0, z := A.z }

theorem projection_of_A_onto_Oxz_is_B :
  let A := Point3D.mk 2 3 6
  let B := Point3D.mk 2 0 6
  projection_onto_Oxz A = B :=
by
  let A := Point3D.mk 2 3 6
  let B := Point3D.mk 2 0 6
  have h : projection_onto_Oxz A = B := rfl
  exact h

end projection_of_A_onto_Oxz_is_B_l118_118731


namespace power_function_half_l118_118704

theorem power_function_half (f : ℝ → ℝ) (h : ∀ x : ℝ, f x = x ^ (1/2)) (hx : f 4 = 2) : 
  f (1/2) = (Real.sqrt 2) / 2 :=
by sorry

end power_function_half_l118_118704


namespace area_ratio_ellipse_l118_118841

noncomputable def ellipse := { x // (x.1^2 / 3 + x.2^2 = 1) }
def foci (a b : ℝ) : (ℝ × ℝ) × (ℝ × ℝ) := ((-real.sqrt (a^2 - b^2), 0), (real.sqrt (a^2 - b^2), 0))
def intersects (line : ℝ × ℝ) (C : ℝ × ℝ → Prop) := ∃ (x y : ℝ), (C (x, y)) ∧ (y = x + line.2)

theorem area_ratio_ellipse 
  (a b : ℝ) (h_ellipse : a^2 = 3 ∧ b^2 = 1) (m : ℝ) 
  (h_foci : foci a b = ((-real.sqrt (a^2 - b^2), 0), (real.sqrt (a^2 - b^2), 0))) 
  (h_condition : ∀ (x : ℝ), intersects (x, m) (λ pt, pt.1^2 / 3 + pt.2^2 = 1) →
     |((x - real.sqrt (a^2 - b^2)) + m)| = 2|((x + real.sqrt (a^2 - b^2)) + m)|) :
  m = -real.sqrt(2) / 3 := sorry

end area_ratio_ellipse_l118_118841


namespace part1_part2_i_part2_ii_l118_118852

noncomputable def f (k : ℝ) (x : ℝ) : ℝ := k * Real.log x + 1 / Real.exp x

theorem part1 (k : ℝ) (h : ¬ MonotoneOn (f k) (Set.Icc 2 3)) :
  3 / Real.exp 3 < k ∧ k < 2 / Real.exp 2 :=
sorry

variables {x1 x2 : ℝ}
variable (k : ℝ)
variable (h0 : 0 < x1)
variable (h1 : x1 < x2)
variable (h2 : k = x1 / Real.exp x1 ∧ k = x2 / Real.exp x2)

theorem part2_i :
  e / Real.exp x2 - e / Real.exp x1 > -Real.log (x2 / x1) ∧ -Real.log (x2 / x1) > 1 - x2 / x1 :=
sorry

theorem part2_ii : |f k x1 - f k x2| < 1 :=
sorry

end part1_part2_i_part2_ii_l118_118852


namespace equilateral_triangle_side_length_l118_118884

noncomputable def side_length (a : ℝ) := if a = 0 then 0 else (a : ℝ) * (3 : ℝ) / 2

theorem equilateral_triangle_side_length
  (a : ℝ)
  (h1 : a ≠ 0)
  (A := (a, - (1 / 3) * a^2))
  (B := (-a, - (1 / 3) * a^2))
  (Habo : (A.1 - 0)^2 + (A.2 - 0)^2 = (B.1 - 0)^2 + (B.2 - 0)^2) :
  ∃ s : ℝ, s = 9 / 2 :=
by
  sorry

end equilateral_triangle_side_length_l118_118884


namespace max_a_for_necessary_not_sufficient_condition_l118_118713

theorem max_a_for_necessary_not_sufficient_condition {x a : ℝ} (h : ∀ x, x^2 > 1 → x < a) : a = -1 :=
by sorry

end max_a_for_necessary_not_sufficient_condition_l118_118713


namespace mr_blues_yard_expectation_l118_118297

noncomputable def calculate_expected_harvest (length_steps : ℕ) (width_steps : ℕ) (step_length : ℝ) (yield_per_sqft : ℝ) : ℝ :=
  let length_feet := length_steps * step_length
  let width_feet := width_steps * step_length
  let area := length_feet * width_feet
  let total_yield := area * yield_per_sqft
  total_yield

theorem mr_blues_yard_expectation : calculate_expected_harvest 18 25 2.5 (3 / 4) = 2109.375 :=
by
  sorry

end mr_blues_yard_expectation_l118_118297


namespace solve_for_x_l118_118952

theorem solve_for_x (x : ℝ) (h : 3 * x + 36 = 48) : x = 4 :=
by
  sorry

end solve_for_x_l118_118952


namespace fraction_of_students_older_than_4_years_l118_118204

-- Definitions based on conditions
def total_students := 50
def students_younger_than_3 := 20
def students_not_between_3_and_4 := 25
def students_older_than_4 := students_not_between_3_and_4 - students_younger_than_3
def fraction_older_than_4 := students_older_than_4 / total_students

-- Theorem to prove the desired fraction
theorem fraction_of_students_older_than_4_years : fraction_older_than_4 = 1/10 :=
by
  sorry

end fraction_of_students_older_than_4_years_l118_118204


namespace tangent_line_eq_at_x1_enclosed_area_eq_l118_118120

noncomputable def f (x : ℝ) := x^3 - x + 2
noncomputable def f' (x : ℝ) := 3 * x^2 - 1

-- Prove the equation of the tangent line l
theorem tangent_line_eq_at_x1 :
  let x := 1
  let point := (x, f(x))
  let slope := f' x in
  ∃ (c : ℝ), ∀ y, y = slope * (x - 1) + c → y = 2 * x :=
by sorry

-- Prove the area enclosed by the line l and the graph of f'(x)
theorem enclosed_area_eq :
  ∫ x in -1/3..1, (2 * x - f' x) = 32 / 27 :=
by sorry

end tangent_line_eq_at_x1_enclosed_area_eq_l118_118120


namespace jake_not_drop_coffee_percentage_l118_118012

-- Definitions for the conditions
def trip_probability : ℝ := 0.40
def drop_when_trip_probability : ℝ := 0.25

-- The question and proof statement
theorem jake_not_drop_coffee_percentage :
  100 * (1 - trip_probability * drop_when_trip_probability) = 90 :=
by
  sorry

end jake_not_drop_coffee_percentage_l118_118012


namespace alex_needs_additional_coins_l118_118353

theorem alex_needs_additional_coins :
  let n := 15
  let current_coins := 63
  let target_sum := (n * (n + 1)) / 2
  let additional_coins := target_sum - current_coins
  additional_coins = 57 :=
by
  sorry

end alex_needs_additional_coins_l118_118353


namespace find_ck_l118_118913

-- Definitions based on the conditions
def arithmetic_sequence (d : ℕ) (n : ℕ) : ℕ :=
  1 + (n - 1) * d

def geometric_sequence (r : ℕ) (n : ℕ) : ℕ :=
  r^(n - 1)

def combined_sequence (d r : ℕ) (n : ℕ) : ℕ :=
  arithmetic_sequence d n + geometric_sequence r n

-- Given conditions
variable {d r k : ℕ}
variable (hd : combined_sequence d r (k-1) = 250)
variable (hk : combined_sequence d r (k+1) = 1250)

-- The theorem statement
theorem find_ck : combined_sequence d r k = 502 :=
  sorry

end find_ck_l118_118913


namespace thermostat_range_l118_118665

theorem thermostat_range (T : ℝ) : 
  |T - 22| ≤ 6 ↔ 16 ≤ T ∧ T ≤ 28 := 
by sorry

end thermostat_range_l118_118665


namespace ratio_movies_allowance_l118_118801

variable (M A : ℕ)
variable (weeklyAllowance moneyEarned endMoney : ℕ)
variable (H1 : weeklyAllowance = 8)
variable (H2 : moneyEarned = 8)
variable (H3 : endMoney = 12)
variable (H4 : weeklyAllowance + moneyEarned - M = endMoney)
variable (H5 : A = 8)
variable (H6 : M = weeklyAllowance + moneyEarned - endMoney / 1)

theorem ratio_movies_allowance (M A : ℕ) 
  (weeklyAllowance moneyEarned endMoney : ℕ)
  (H1 : weeklyAllowance = 8)
  (H2 : moneyEarned = 8)
  (H3 : endMoney = 12)
  (H4 : weeklyAllowance + moneyEarned - M = endMoney)
  (H5 : A = 8)
  (H6 : M = weeklyAllowance + moneyEarned - endMoney / 1) :
  M / A = 1 / 2 :=
sorry

end ratio_movies_allowance_l118_118801


namespace walkway_time_l118_118064

theorem walkway_time {v_p v_w : ℝ} 
  (cond1 : 60 = (v_p + v_w) * 30) 
  (cond2 : 60 = (v_p - v_w) * 120) 
  : 60 / v_p = 48 := 
by
  sorry

end walkway_time_l118_118064


namespace line_does_not_pass_second_quadrant_l118_118118

theorem line_does_not_pass_second_quadrant (a : ℝ) (ha : a ≠ 0) :
  ∀ (x y : ℝ), (x - y - a^2 = 0) → ¬(x < 0 ∧ y > 0) :=
sorry

end line_does_not_pass_second_quadrant_l118_118118


namespace solve_otimes_n_1_solve_otimes_2005_2_l118_118526

-- Define the operation ⊗
noncomputable def otimes (x y : ℕ) : ℕ :=
sorry -- the definition is abstracted away as per conditions

-- Conditions from the problem
axiom otimes_cond_1 : ∀ x : ℕ, otimes x 0 = x + 1
axiom otimes_cond_2 : ∀ x : ℕ, otimes 0 (x + 1) = otimes 1 x
axiom otimes_cond_3 : ∀ x y : ℕ, otimes (x + 1) (y + 1) = otimes (otimes x (y + 1)) y

-- Prove the required equalities
theorem solve_otimes_n_1 (n : ℕ) : otimes n 1 = n + 2 :=
sorry

theorem solve_otimes_2005_2 : otimes 2005 2 = 4013 :=
sorry

end solve_otimes_n_1_solve_otimes_2005_2_l118_118526


namespace infinite_natural_solutions_l118_118608

theorem infinite_natural_solutions : ∀ n : ℕ, ∃ x y z : ℕ, (x + y + z)^2 + 2 * (x + y + z) = 5 * (x * y + y * z + z * x) :=
by
  sorry

end infinite_natural_solutions_l118_118608


namespace scientific_notation_of_1_5_million_l118_118302

theorem scientific_notation_of_1_5_million : 
    (1.5 * 10^6 = 1500000) :=
by
    sorry

end scientific_notation_of_1_5_million_l118_118302


namespace calculate_expression_l118_118087

theorem calculate_expression :
  (5 * 7 + 10 * 4 - 35 / 5 + 18 / 3 : ℝ) = 74 := by
  sorry

end calculate_expression_l118_118087


namespace complex_modulus_square_l118_118293

open Complex

theorem complex_modulus_square (a b : ℝ) (h : 5 * (a + b * I) + 3 * Complex.abs (a + b * I) = 15 - 16 * I) :
  (Complex.abs (a + b * I))^2 = 256 / 25 :=
by sorry

end complex_modulus_square_l118_118293


namespace ratio_two_to_three_nights_ago_l118_118688

def question (x : ℕ) (k : ℕ) : (ℕ × ℕ) := (x, k)

def pages_three_nights_ago := 15
def additional_pages_last_night (x : ℕ) := x + 5
def total_pages := 100
def pages_tonight := 20

theorem ratio_two_to_three_nights_ago :
  ∃ (x : ℕ), 
    (x + additional_pages_last_night x = total_pages - (pages_three_nights_ago + pages_tonight)) 
    ∧ (x / pages_three_nights_ago = 2 / 1) :=
by
  sorry

end ratio_two_to_three_nights_ago_l118_118688


namespace distance_between_x_intercepts_l118_118070

theorem distance_between_x_intercepts (x1 y1 : ℝ) 
  (m1 m2 : ℝ)
  (hx1 : x1 = 10) (hy1 : y1 = 15)
  (hm1 : m1 = 3) (hm2 : m2 = 5) :
  let x_intercept1 := (y1 - m1 * x1) / -m1
  let x_intercept2 := (y1 - m2 * x1) / -m2
  dist (x_intercept1, 0) (x_intercept2, 0) = 2 :=
by
  sorry

end distance_between_x_intercepts_l118_118070


namespace number_of_allocation_schemes_l118_118499

/-- 
  Given 5 volunteers and 4 projects, each volunteer is assigned to only one project,
  and each project must have at least one volunteer.
  Prove that there are 240 different allocation schemes.
-/
theorem number_of_allocation_schemes (V P : ℕ) (hV : V = 5) (hP : P = 4) 
  (each_volunteer_one_project : ∀ v, ∃ p, v ≠ p) 
  (each_project_at_least_one : ∀ p, ∃ v, v ≠ p) : 
  ∃ n_ways : ℕ, n_ways = 240 :=
by
  sorry

end number_of_allocation_schemes_l118_118499


namespace percent_profit_l118_118337

theorem percent_profit (C S : ℝ) (h : 60 * C = 40 * S) : (S - C) / C * 100 = 50 := by
  sorry

end percent_profit_l118_118337


namespace part1_part2_l118_118252

noncomputable def f (a : ℝ) (x : ℝ) := (a - 1/2) * x^2 + Real.log x
noncomputable def g (a : ℝ) (x : ℝ) := f a x - 2 * a * x

theorem part1 (x : ℝ) (hxe : Real.exp (-1) ≤ x ∧ x ≤ Real.exp (1)) : 
    f (-1/2) x ≤ -1/2 - 1/2 * Real.log 2 ∧ f (-1/2) x ≥ 1 - Real.exp 2 := sorry

theorem part2 (h : ∀ x > 2, g a x < 0) : a ≤ 1/2 := sorry

end part1_part2_l118_118252


namespace initial_students_count_l118_118768

theorem initial_students_count (n W : ℝ)
    (h1 : W = n * 28)
    (h2 : W + 10 = (n + 1) * 27.4) :
    n = 29 :=
by
  sorry

end initial_students_count_l118_118768


namespace negation_exists_real_negation_of_quadratic_l118_118773

theorem negation_exists_real (P : ℝ → Prop) :
  (¬ ∀ x : ℝ, P x) ↔ ∃ x : ℝ, ¬ P x :=
by sorry

def quadratic (x : ℝ) : Prop := x^2 - 2*x + 3 ≤ 0

theorem negation_of_quadratic :
  (¬ ∀ x : ℝ, quadratic x) ↔ ∃ x : ℝ, ¬ quadratic x :=
by exact negation_exists_real quadratic

end negation_exists_real_negation_of_quadratic_l118_118773


namespace square_area_l118_118185

noncomputable def line_lies_on_square_side (a b : ℝ) : Prop :=
  ∃ (A B : ℝ × ℝ), A = (a, a + 4) ∧ B = (b, b + 4)

noncomputable def points_on_parabola (x y : ℝ) : Prop :=
  ∃ (C D : ℝ × ℝ), C = (y^2, y) ∧ D = (x^2, x)

theorem square_area (a b : ℝ) (x y : ℝ)
  (h1 : line_lies_on_square_side a b)
  (h2 : points_on_parabola x y) :
  ∃ (s : ℝ), s^2 = (boxed_solution) :=
sorry

end square_area_l118_118185


namespace minimum_value_l118_118700

theorem minimum_value (x y z : ℝ) (h : x + 2 * y + z = 1) : x^2 + 4 * y^2 + z^2 ≥ 1 / 3 :=
sorry

end minimum_value_l118_118700


namespace total_payment_l118_118597

-- Define the basic conditions
def hours_first_day : ℕ := 10
def hours_second_day : ℕ := 8
def hours_third_day : ℕ := 15
def hourly_wage : ℕ := 10
def number_of_workers : ℕ := 2

-- Define the proof problem
theorem total_payment : 
  (hours_first_day + hours_second_day + hours_third_day) * hourly_wage * number_of_workers = 660 := 
by
  sorry

end total_payment_l118_118597


namespace mathematicians_correct_l118_118906

noncomputable def scenario1 (whites1 : ℕ) (total1 : ℕ) (whites2 : ℕ) (total2 : ℕ) : Prop :=
  whites1 = 4 ∧ total1 = 7 ∧ whites2 = 3 ∧ total2 = 5

noncomputable def scenario2 (whites1 : ℕ) (total1 : ℕ) (whites2 : ℕ) (total2 : ℕ) : Prop :=
  whites1 = 8 ∧ total1 = 14 ∧ whites2 = 3 ∧ total2 = 5

noncomputable def scenario3 (whites1 : ℕ) (total1 : ℕ) (whites2 : ℕ) (total2 : ℕ) : Prop :=
  (19 / 35 < 4 / 7) ∧ (4 / 7 < 3 / 5)

noncomputable def probability (whites1 : ℕ) (total1 : ℕ) (whites2 : ℕ) (total2 : ℕ) : ℝ :=
  (whites1 + whites2) / (total1 + total2)

theorem mathematicians_correct :
  let whites1_s1 := 4 in
  let total1_s1 := 7 in
  let whites2_s1 := 3 in
  let total2_s1 := 5 in
  let whites1_s2 := 8 in
  let total1_s2 := 14 in
  let whites2_s2 := 3 in
  let total2_s2 := 5 in
  scenario1 whites1_s1 total1_s1 whites2_s1 total2_s1 →
  scenario2 whites1_s2 total1_s2 whites2_s2 total2_s2 →
  scenario3 whites1_s1 total1_s1 whites2_s2 total2_s2 →
  probability whites1_s1 total1_s1 whites2_s1 total2_s1 ≤ 3 / 5 ∨
  probability whites1_s1 total1_s1 whites2_s1 total2_s1 = 4 / 7 ∨
  probability whites1_s2 total1_s2 whites2_s2 total2_s2 ≤ 3 / 5 ∨
  probability whites1_s2 total1_s2 whites2_s2 total2_s2 = 4 / 7 :=
begin
  intros,
  sorry

end mathematicians_correct_l118_118906


namespace total_customers_in_line_l118_118370

-- Definition of the number of people standing in front of the last person
def num_people_in_front : Nat := 8

-- Definition of the last person in the line
def last_person : Nat := 1

-- Statement to prove
theorem total_customers_in_line : num_people_in_front + last_person = 9 := by
  sorry

end total_customers_in_line_l118_118370


namespace cheese_cost_l118_118024

theorem cheese_cost (bread_cost cheese_cost total_paid total_change coin_change nickels_value : ℝ) 
                    (quarter dime nickels_count : ℕ)
                    (h1 : bread_cost = 4.20)
                    (h2 : total_paid = 7.00)
                    (h3 : quarter = 1)
                    (h4 : dime = 1)
                    (h5 : nickels_count = 8)
                    (h6 : coin_change = (quarter * 0.25) + (dime * 0.10) + (nickels_count * 0.05))
                    (h7 : total_change = total_paid - bread_cost)
                    (h8 : cheese_cost = total_change - coin_change) :
                    cheese_cost = 2.05 :=
by {
    sorry
}

end cheese_cost_l118_118024


namespace smallest_D_l118_118730

theorem smallest_D {A B C D : ℕ} (h1 : A ≠ B ∧ A ≠ C ∧ A ≠ D ∧ B ≠ C ∧ B ≠ D ∧ C ≠ D) 
  (h2 : (A * 100 + B * 10 + C) * B = D * 1000 + C * 100 + B * 10 + D) : 
  D = 1 :=
sorry

end smallest_D_l118_118730


namespace area_of_trapezoid_l118_118206

theorem area_of_trapezoid
  (r : ℝ)
  (AD BC : ℝ)
  (center_on_base : Bool)
  (height : ℝ)
  (area : ℝ)
  (inscribed_circle : r = 6)
  (base_AD : AD = 8)
  (base_BC : BC = 4)
  (K_height : height = 4 * Real.sqrt 2)
  (calc_area : area = (1 / 2) * (AD + BC) * height)
  : area = 32 * Real.sqrt 2 := by
  sorry

end area_of_trapezoid_l118_118206


namespace triangle_internal_angle_60_l118_118163

theorem triangle_internal_angle_60 (A B C : ℝ) (h_sum : A + B + C = 180) : A >= 60 ∨ B >= 60 ∨ C >= 60 :=
sorry

end triangle_internal_angle_60_l118_118163


namespace aquarium_pufferfish_problem_l118_118922

/-- Define the problem constants and equations -/
theorem aquarium_pufferfish_problem :
  ∃ (P S : ℕ), S = 5 * P ∧ S + P = 90 ∧ P = 15 :=
by
  sorry

end aquarium_pufferfish_problem_l118_118922


namespace max_traffic_flow_at_v_40_traffic_flow_at_least_10_thousand_l118_118733

-- Define the traffic flow function
noncomputable def traffic_flow (v : ℝ) : ℝ := 920 * v / (v^2 + 3 * v + 1600)

-- Condition: v > 0
axiom v_pos (v : ℝ) : v > 0 → traffic_flow v ≥ 0

-- Prove that the average speed v = 40 results in the maximum traffic flow y = 920/83 ≈ 11.08
theorem max_traffic_flow_at_v_40 : traffic_flow 40 = 920 / 83 :=
sorry

-- Prove that to ensure the traffic flow is at least 10 thousand vehicles per hour,
-- the average speed v should be in the range [25, 64]
theorem traffic_flow_at_least_10_thousand (v : ℝ) (h : traffic_flow v ≥ 10) : 25 ≤ v ∧ v ≤ 64 :=
sorry

end max_traffic_flow_at_v_40_traffic_flow_at_least_10_thousand_l118_118733


namespace remaining_rectangle_area_l118_118356

theorem remaining_rectangle_area (s a b : ℕ) (hs : s = a + b) (total_area_cut : a^2 + b^2 = 40) : s^2 - 40 = 24 :=
by
  sorry

end remaining_rectangle_area_l118_118356


namespace trapezium_area_l118_118936

theorem trapezium_area (a b d : ℕ) (h₁ : a = 28) (h₂ : b = 18) (h₃ : d = 15) :
  (a + b) * d / 2 = 345 := by
{
  sorry
}

end trapezium_area_l118_118936


namespace number_of_cherry_pie_days_l118_118447

theorem number_of_cherry_pie_days (A C : ℕ) (h1 : A + C = 7) (h2 : 12 * A = 12 * C + 12) : C = 3 :=
sorry

end number_of_cherry_pie_days_l118_118447


namespace average_marks_l118_118653

-- Given conditions
variables (M P C : ℝ)
variables (h1 : M + P = 32) (h2 : C = P + 20)

-- Statement to be proved
theorem average_marks : (M + C) / 2 = 26 :=
by
  -- The proof will be inserted here
  sorry

end average_marks_l118_118653


namespace correct_calculated_value_l118_118648

theorem correct_calculated_value (n : ℕ) (h1 : n = 32 * 3) : n / 4 = 24 := 
by
  -- proof steps will be filled here
  sorry

end correct_calculated_value_l118_118648


namespace total_amount_after_refunds_l118_118523

def individual_bookings : ℕ := 12000
def group_bookings : ℕ := 16000
def refunds : ℕ := 1600

theorem total_amount_after_refunds : 
  individual_bookings + group_bookings - refunds = 26400 :=
by 
  -- The proof goes here.
  sorry

end total_amount_after_refunds_l118_118523


namespace find_d_l118_118533

theorem find_d (d : ℚ) (int_part frac_part : ℚ) 
  (h1 : 3 * int_part^2 + 19 * int_part - 28 = 0)
  (h2 : 4 * frac_part^2 - 11 * frac_part + 3 = 0)
  (h3 : frac_part ≥ 0 ∧ frac_part < 1)
  (h4 : d = int_part + frac_part) :
  d = -29 / 4 :=
by
  sorry

end find_d_l118_118533


namespace ratio_of_area_to_square_of_perimeter_of_equilateral_triangle_l118_118191

theorem ratio_of_area_to_square_of_perimeter_of_equilateral_triangle :
  let a := 10
  let area := (10 * 10 * (Real.sqrt 3) / 4)
  let perimeter := 3 * 10
  let square_of_perimeter := perimeter * perimeter
  (area / square_of_perimeter) = (Real.sqrt 3 / 36) := by
  -- Proof to be completed
  sorry

end ratio_of_area_to_square_of_perimeter_of_equilateral_triangle_l118_118191


namespace jen_age_proof_l118_118786

-- Definitions
def son_age := 16
def son_present_age := son_age
def jen_present_age := 41

-- Conditions
axiom jen_older_25 (x : ℕ) : ∀ y : ℕ, x = y + 25 → y = son_present_age
axiom jen_age_formula (j s : ℕ) : j = 3 * s - 7 → j = son_present_age + 25

-- Proof problem statement
theorem jen_age_proof : jen_present_age = 41 :=
by
  -- Declare variables
  let j := jen_present_age
  let s := son_present_age
  -- Apply conditions (in Lean, sorry will skip the proof)
  sorry

end jen_age_proof_l118_118786

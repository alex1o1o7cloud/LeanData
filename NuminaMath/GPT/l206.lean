import Mathlib

namespace molecular_weight_of_6_moles_l206_206690

-- Define the molecular weight of the compound
def molecular_weight : ℕ := 1404

-- Define the number of moles
def number_of_moles : ℕ := 6

-- The hypothesis would be the molecular weight condition
theorem molecular_weight_of_6_moles : number_of_moles * molecular_weight = 8424 :=
by sorry

end molecular_weight_of_6_moles_l206_206690


namespace nat_solution_unique_l206_206580

theorem nat_solution_unique (x y : ℕ) (h : x + y = x * y) : (x, y) = (2, 2) :=
sorry

end nat_solution_unique_l206_206580


namespace correct_quotient_of_original_division_operation_l206_206994

theorem correct_quotient_of_original_division_operation 
  (incorrect_divisor correct_divisor incorrect_quotient : ℕ)
  (h1 : incorrect_divisor = 102)
  (h2 : correct_divisor = 201)
  (h3 : incorrect_quotient = 753)
  (h4 : ∃ k, k = incorrect_quotient * 3) :
  ∃ q, q = 1146 ∧ (correct_divisor * q = incorrect_divisor * (incorrect_quotient * 3)) :=
by
  sorry

end correct_quotient_of_original_division_operation_l206_206994


namespace match_end_time_is_17_55_l206_206169

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

end match_end_time_is_17_55_l206_206169


namespace intersection_A_B_l206_206293

def A : Set ℝ := {x | x > 0}
def B : Set ℝ := {x | -1 < x ∧ x ≤ 2}
def intersection (S T : Set ℝ) : Set ℝ := {x | x ∈ S ∧ x ∈ T}

theorem intersection_A_B :
  (A ∩ B) = {x | 0 < x ∧ x ≤ 2} := by
  sorry

end intersection_A_B_l206_206293


namespace correct_discount_rate_l206_206789

def purchase_price : ℝ := 200
def marked_price : ℝ := 300
def desired_profit_percentage : ℝ := 0.20

theorem correct_discount_rate :
  ∃ (x : ℝ), 300 * x = 240 ∧ x = 0.80 := 
by
  sorry

end correct_discount_rate_l206_206789


namespace general_term_a_sum_Tn_l206_206614

section sequence_problem

variables {n : ℕ} (a : ℕ → ℝ) (S : ℕ → ℝ) (b : ℕ → ℝ) (T : ℕ → ℝ)

-- Problem 1: General term formula for {a_n}
axiom Sn_def : ∀ n, S n = 1/4 * (a n + 1)^2
axiom a1_def : a 1 = 1
axiom an_diff : ∀ n, a (n+1) - a n = 2

theorem general_term_a : a n = 2 * n - 1 := sorry

-- Problem 2: Sum of the first n terms of sequence {b_n}
axiom an_formula : ∀ n, a n = 2 * n - 1
axiom bn_def : ∀ n, b n = 1 / (a n * a (n+1))

theorem sum_Tn : T n = n / (2 * n + 1) := sorry

end sequence_problem

end general_term_a_sum_Tn_l206_206614


namespace intersection_complement_eq_l206_206521

open Set

namespace MathProof

variable (U A B : Set ℕ)

theorem intersection_complement_eq :
  U = {1, 2, 3, 4, 5, 6, 7} →
  A = {3, 4, 5} →
  B = {1, 3, 6} →
  A ∩ (U \ B) = {4, 5} :=
by
  intros hU hA hB
  sorry

end MathProof

end intersection_complement_eq_l206_206521


namespace student_in_eighth_group_l206_206141

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

end student_in_eighth_group_l206_206141


namespace number_of_officers_l206_206354

theorem number_of_officers
  (avg_all : ℝ := 120)
  (avg_officer : ℝ := 420)
  (avg_non_officer : ℝ := 110)
  (num_non_officer : ℕ := 450) :
  ∃ O : ℕ, avg_all * (O + num_non_officer) = avg_officer * O + avg_non_officer * num_non_officer ∧ O = 15 :=
by
  sorry

end number_of_officers_l206_206354


namespace quadratic_inequality_has_real_solutions_l206_206736

theorem quadratic_inequality_has_real_solutions (c : ℝ) (h : 0 < c) : 
  (∃ x : ℝ, x^2 - 6 * x + c < 0) ↔ (0 < c ∧ c < 9) :=
sorry

end quadratic_inequality_has_real_solutions_l206_206736


namespace find_integer_solutions_l206_206969

theorem find_integer_solutions :
  {p : ℤ × ℤ | 2 * p.1^3 + p.1 * p.2 = 7} = {(-7, -99), (-1, -9), (1, 5), (7, -97)} :=
by
  -- Proof not required
  sorry

end find_integer_solutions_l206_206969


namespace min_additional_cells_l206_206168

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

end min_additional_cells_l206_206168


namespace combined_total_l206_206424

variable (Jane Jean : ℕ)

theorem combined_total (h1 : Jean = 3 * Jane) (h2 : Jean = 57) : Jane + Jean = 76 := by
  sorry

end combined_total_l206_206424


namespace factorial_division_l206_206490

def factorial (n : ℕ) : ℕ :=
  if n = 0 then 1 else n * factorial (n - 1)

theorem factorial_division (n : ℕ) (h : n > 0) : factorial (n) / factorial (n - 1) = n :=
by
  sorry

example : factorial 12 / factorial 11 = 12 :=
by
  exact factorial_division 12 (by norm_num)

end factorial_division_l206_206490


namespace unique_solution_l206_206062

theorem unique_solution (a b : ℤ) : 
  (a^6 + 1 ∣ b^11 - 2023 * b^3 + 40 * b) ∧ (a^4 - 1 ∣ b^10 - 2023 * b^2 - 41) 
  ↔ (a = 0 ∧ ∃ c : ℤ, b = c) := 
by 
  sorry

end unique_solution_l206_206062


namespace max_value_of_g_l206_206138

def g : ℕ → ℕ
| n => if n < 7 then n + 7 else g (n - 3)

theorem max_value_of_g : ∀ (n : ℕ), g n ≤ 13 ∧ (∃ n0, g n0 = 13) := by
  sorry

end max_value_of_g_l206_206138


namespace intersection_of_asymptotes_l206_206750

noncomputable def f (x : ℝ) : ℝ := (x^2 - 6*x + 8) / (x^2 - 6*x + 9)

theorem intersection_of_asymptotes : f 3 = 1 :=
by sorry

end intersection_of_asymptotes_l206_206750


namespace largest_two_digit_integer_l206_206270

theorem largest_two_digit_integer
  (a b : ℕ) (h1 : 1 ≤ a ∧ a < 10) (h2 : 0 ≤ b ∧ b < 10)
  (h3 : 3 * (10 * a + b) = 10 * b + a + 5) :
  10 * a + b = 13 :=
by {
  -- Sorry is placed here to indicate that the proof is not provided
  sorry
}

end largest_two_digit_integer_l206_206270


namespace parabola_vertex_l206_206922

theorem parabola_vertex {a b c : ℝ} (h₁ : ∃ b c, ∀ x, a * x^2 + b * x + c = a * (x + 3)^2) (h₂ : a * (2 + 3)^2 = -50) : a = -2 :=
by
  sorry

end parabola_vertex_l206_206922


namespace E_union_F_eq_univ_l206_206948

-- Define the given conditions
def E : Set ℝ := { x | x^2 - 5 * x - 6 > 0 }
def F (a : ℝ) : Set ℝ := { x | x - 5 < a }
def I : Set ℝ := Set.univ
axiom a_gt_6 : ∃ a : ℝ, a > 6 ∧ 11 ∈ F a

-- State the theorem
theorem E_union_F_eq_univ (a : ℝ) (h₁ : a > 6) (h₂ : 11 ∈ F a) : E ∪ F a = I := by
  sorry

end E_union_F_eq_univ_l206_206948


namespace MaryIncomeIs64PercentOfJuanIncome_l206_206714

variable {J T M : ℝ}

-- Conditions
def TimIncome (J : ℝ) : ℝ := 0.40 * J
def MaryIncome (T : ℝ) : ℝ := 1.60 * T

-- Theorem to prove
theorem MaryIncomeIs64PercentOfJuanIncome (J : ℝ) :
  MaryIncome (TimIncome J) = 0.64 * J :=
by
  sorry

end MaryIncomeIs64PercentOfJuanIncome_l206_206714


namespace total_number_of_fish_l206_206883

noncomputable def number_of_stingrays : ℕ := 28

noncomputable def number_of_sharks : ℕ := 2 * number_of_stingrays

theorem total_number_of_fish : number_of_sharks + number_of_stingrays = 84 :=
by
  sorry

end total_number_of_fish_l206_206883


namespace option_A_correct_l206_206256

theorem option_A_correct (y x : ℝ) : y * x - 2 * (x * y) = - (x * y) :=
by
  sorry

end option_A_correct_l206_206256


namespace remainder_problem_l206_206957

theorem remainder_problem (f y z : ℤ) (k m n : ℤ) 
  (h1 : f % 5 = 3) 
  (h2 : y % 5 = 4)
  (h3 : z % 7 = 6)
  (h4 : (f + y) % 15 = 7)
  : (f + y + z) % 35 = 3 ∧ (f + y + z) % 105 = 3 :=
by
  sorry

end remainder_problem_l206_206957


namespace find_number_l206_206682

theorem find_number (N Q : ℕ) (h1 : N = 5 * Q) (h2 : Q + N + 5 = 65) : N = 50 :=
by
  sorry

end find_number_l206_206682


namespace cistern_fill_time_l206_206261

-- Let F be the rate at which the first tap fills the cistern (cisterns per hour)
def F : ℚ := 1 / 4

-- Let E be the rate at which the second tap empties the cistern (cisterns per hour)
def E : ℚ := 1 / 5

-- Prove that the time it takes to fill the cistern is 20 hours given the rates F and E
theorem cistern_fill_time : (1 / (F - E)) = 20 := 
by
  -- Insert necessary proofs here
  sorry

end cistern_fill_time_l206_206261


namespace number_of_customers_who_did_not_want_tires_change_l206_206077

noncomputable def total_cars_in_shop : Nat := 4 + 6
noncomputable def tires_per_car : Nat := 4
noncomputable def total_tires_bought : Nat := total_cars_in_shop * tires_per_car
noncomputable def half_tires_left : Nat := 2 * (tires_per_car / 2)
noncomputable def total_half_tires_left : Nat := 2 * half_tires_left
noncomputable def tires_left_after_half : Nat := 20
noncomputable def tires_left_after_half_customers : Nat := tires_left_after_half - total_half_tires_left
noncomputable def customers_who_did_not_change_tires : Nat := tires_left_after_half_customers / tires_per_car

theorem number_of_customers_who_did_not_want_tires_change : 
  customers_who_did_not_change_tires = 4 :=
by
  sorry 

end number_of_customers_who_did_not_want_tires_change_l206_206077


namespace ages_sum_l206_206180

theorem ages_sum (a b c : ℕ) (h1 : a = b) (h2 : a * b * c = 72) : a + b + c = 14 :=
by sorry

end ages_sum_l206_206180


namespace abs_sum_zero_l206_206679

theorem abs_sum_zero (a b : ℝ) (h : |a - 5| + |b + 8| = 0) : a + b = -3 := 
sorry

end abs_sum_zero_l206_206679


namespace distinct_permutations_mathematics_l206_206581

theorem distinct_permutations_mathematics : 
  let n := 11
  let freqM := 2
  let freqA := 2
  let freqT := 2
  (n.factorial / (freqM.factorial * freqA.factorial * freqT.factorial)) = 4989600 :=
by
  let n := 11
  let freqM := 2
  let freqA := 2
  let freqT := 2
  sorry

end distinct_permutations_mathematics_l206_206581


namespace range_of_quadratic_function_l206_206836

theorem range_of_quadratic_function : 
  ∀ x : ℝ, -1 ≤ x ∧ x ≤ 1 → 0 ≤ x^2 - 4 * x + 3 ∧ x^2 - 4 * x + 3 ≤ 8 :=
by
  intro x hx
  sorry

end range_of_quadratic_function_l206_206836


namespace find_f_neg_eight_l206_206927

-- Conditions based on the given problem
variable (f : ℤ → ℤ)
axiom func_property : ∀ x y : ℤ, f (x + y) = f x + f y + x * y + 1
axiom f1_is_one : f 1 = 1

-- Main theorem
theorem find_f_neg_eight : f (-8) = 19 := by
  sorry

end find_f_neg_eight_l206_206927


namespace incorrect_statement_C_l206_206676

-- Lean 4 statement to verify correctness of problem translation
theorem incorrect_statement_C (n : ℕ) (w : ℕ → ℕ) :
  (w 1 = 55) ∧
  (w 2 = 110) ∧
  (w 3 = 160) ∧
  (w 4 = 200) ∧
  (w 5 = 254) ∧
  (w 6 = 300) ∧
  (w 7 = 350) →
  ¬(∀ n, w n = 55 * n) :=
by
  intros h
  sorry

end incorrect_statement_C_l206_206676


namespace circle_center_and_radius_l206_206713

-- Define the equation of the circle
def circle_equation (x y : ℝ) : Prop :=
  x^2 + y^2 - 6*x + 5 = 0

-- Statement of the center and radius of the circle
theorem circle_center_and_radius :
  (∀ x y : ℝ, circle_equation x y) →
  (∃ (h k r : ℝ), (∀ x y : ℝ, circle_equation x y ↔ (x - h)^2 + (y - k)^2 = r^2) ∧ h = 3 ∧ k = 0 ∧ r = 2) :=
by
  sorry

end circle_center_and_radius_l206_206713


namespace find_10th_integer_l206_206983

-- Defining the conditions
def avg_20_consecutive_integers (avg : ℝ) : Prop :=
  avg = 23.65

def consecutive_integer_sequence (n : ℤ) (a : ℤ) : Prop :=
  a = n + 9

-- The main theorem statement
theorem find_10th_integer (n : ℤ) (avg : ℝ) (h_avg : avg_20_consecutive_integers avg) (h_seq : consecutive_integer_sequence n 23) :
  n = 14 :=
sorry

end find_10th_integer_l206_206983


namespace base_conversion_sum_correct_l206_206391

theorem base_conversion_sum_correct :
  (253 / 8 / 13 / 3 + 245 / 7 / 35 / 6 : ℚ) = 339 / 23 := sorry

end base_conversion_sum_correct_l206_206391


namespace sum_of_squares_of_coefficients_l206_206121

theorem sum_of_squares_of_coefficients :
  ∃ (a b c d e f : ℤ), (∀ x : ℤ, 8 * x ^ 3 + 64 = (a * x ^ 2 + b * x + c) * (d * x ^ 2 + e * x + f)) ∧ 
  (a ^ 2 + b ^ 2 + c ^ 2 + d ^ 2 + e ^ 2 + f ^ 2 = 356) := 
by
  sorry

end sum_of_squares_of_coefficients_l206_206121


namespace author_earnings_calculation_l206_206476

open Real

namespace AuthorEarnings

def paperCoverCopies  : ℕ := 32000
def paperCoverPrice   : ℝ := 0.20
def paperCoverPercent : ℝ := 0.06

def hardCoverCopies   : ℕ := 15000
def hardCoverPrice    : ℝ := 0.40
def hardCoverPercent  : ℝ := 0.12

def total_earnings_paper_cover : ℝ := paperCoverCopies * paperCoverPrice
def earnings_paper_cover : ℝ := total_earnings_paper_cover * paperCoverPercent

def total_earnings_hard_cover : ℝ := hardCoverCopies * hardCoverPrice
def earnings_hard_cover : ℝ := total_earnings_hard_cover * hardCoverPercent

def author_total_earnings : ℝ := earnings_paper_cover + earnings_hard_cover

theorem author_earnings_calculation : author_total_earnings = 1104 := by
  sorry

end AuthorEarnings

end author_earnings_calculation_l206_206476


namespace math_problem_l206_206634

def f (x : ℝ) : ℝ := x^2 + 3
def g (x : ℝ) : ℝ := 2 * x + 5

theorem math_problem : f (g 4) - g (f 4) = 129 := by
  sorry

end math_problem_l206_206634


namespace sqrt_calculation_l206_206434

theorem sqrt_calculation : Real.sqrt (36 * Real.sqrt 16) = 12 := 
by
  sorry

end sqrt_calculation_l206_206434


namespace apples_purchased_by_danny_l206_206587

theorem apples_purchased_by_danny (pinky_apples : ℕ) (total_apples : ℕ) (danny_apples : ℕ) :
  pinky_apples = 36 → total_apples = 109 → danny_apples = total_apples - pinky_apples → danny_apples = 73 :=
by 
  intros h1 h2 h3
  rw [h1, h2] at h3
  exact h3

end apples_purchased_by_danny_l206_206587


namespace male_students_count_l206_206993

variable (M F : ℕ)
variable (average_all average_male average_female : ℕ)
variable (total_male total_female total_all : ℕ)

noncomputable def male_students (M F : ℕ) : ℕ := 8

theorem male_students_count:
  F = 32 -> average_all = 90 -> average_male = 82 -> average_female = 92 ->
  total_male = average_male * M -> total_female = average_female * F -> 
  total_all = average_all * (M + F) -> total_male + total_female = total_all ->
  M = male_students M F := 
by
  intros hF hAvgAll hAvgMale hAvgFemale hTotalMale hTotalFemale hTotalAll hEqTotal
  sorry

end male_students_count_l206_206993


namespace terrell_lifting_l206_206575

theorem terrell_lifting :
  (3 * 25 * 10 = 3 * 20 * 12.5) :=
by
  sorry

end terrell_lifting_l206_206575


namespace number_of_20_paise_coins_l206_206364

theorem number_of_20_paise_coins (x y : ℕ) (h1 : x + y = 336) (h2 : (20 / 100 : ℚ) * x + (25 / 100 : ℚ) * y = 71) :
    x = 260 :=
by
  sorry

end number_of_20_paise_coins_l206_206364


namespace train_length_l206_206193

theorem train_length 
  (bridge_length train_length time_seconds v : ℝ)
  (h1 : bridge_length = 300)
  (h2 : time_seconds = 36)
  (h3 : v = 40) :
  (train_length = v * time_seconds - bridge_length) →
  (train_length = 1140) := by
  -- solve in a few lines
  -- This proof is omitted for the purpose of this task
  sorry

end train_length_l206_206193


namespace sum_MN_MK_eq_14_sqrt4_3_l206_206344

theorem sum_MN_MK_eq_14_sqrt4_3
  (MN MK : ℝ)
  (area: ℝ)
  (angle_LMN : ℝ)
  (h_area : area = 49)
  (h_angle_LMN : angle_LMN = 30) :
  MN + MK = 14 * (Real.sqrt (Real.sqrt 3)) :=
by
  sorry

end sum_MN_MK_eq_14_sqrt4_3_l206_206344


namespace alpha_less_than_60_degrees_l206_206577

theorem alpha_less_than_60_degrees
  (R r : ℝ)
  (b c : ℝ)
  (α : ℝ)
  (h1 : b * c = 8 * R * r) :
  α < 60 := sorry

end alpha_less_than_60_degrees_l206_206577


namespace total_amount_earned_l206_206933

-- Conditions
def avg_price_pair_rackets : ℝ := 9.8
def num_pairs_sold : ℕ := 60

-- Proof statement
theorem total_amount_earned :
  avg_price_pair_rackets * num_pairs_sold = 588 := by
    sorry

end total_amount_earned_l206_206933


namespace num_pupils_is_40_l206_206882

-- given conditions
def incorrect_mark : ℕ := 83
def correct_mark : ℕ := 63
def mark_difference : ℕ := incorrect_mark - correct_mark
def avg_increase : ℚ := 1 / 2

-- the main problem statement to prove
theorem num_pupils_is_40 (n : ℕ) (h : (mark_difference : ℚ) / n = avg_increase) : n = 40 := 
sorry

end num_pupils_is_40_l206_206882


namespace total_area_of_WIN_sectors_l206_206004

theorem total_area_of_WIN_sectors (r : ℝ) (A_total : ℝ) (Prob_WIN : ℝ) (A_WIN : ℝ) : 
  r = 15 → 
  A_total = π * r^2 → 
  Prob_WIN = 3/7 → 
  A_WIN = Prob_WIN * A_total → 
  A_WIN = 3/7 * 225 * π :=
by {
  intros;
  sorry
}

end total_area_of_WIN_sectors_l206_206004


namespace four_digit_number_exists_l206_206874

theorem four_digit_number_exists :
  ∃ n : ℕ, n >= 1000 ∧ n < 10000 ∧ 4 * n = (n % 10) * 1000 + ((n / 10) % 10) * 100 + ((n / 100) % 10) * 10 + (n / 1000) :=
sorry

end four_digit_number_exists_l206_206874


namespace remaining_money_after_payments_l206_206026

-- Conditions
def initial_money : ℕ := 100
def paid_colin : ℕ := 20
def paid_helen : ℕ := 2 * paid_colin
def paid_benedict : ℕ := paid_helen / 2
def total_paid : ℕ := paid_colin + paid_helen + paid_benedict

-- Proof
theorem remaining_money_after_payments : 
  initial_money - total_paid = 20 := by
  sorry

end remaining_money_after_payments_l206_206026


namespace average_side_length_of_squares_l206_206486

theorem average_side_length_of_squares :
  let a1 := 25
  let a2 := 64
  let a3 := 144
  let s1 := Real.sqrt a1
  let s2 := Real.sqrt a2
  let s3 := Real.sqrt a3
  (s1 + s2 + s3) / 3 = 25 / 3 := 
by
  sorry

end average_side_length_of_squares_l206_206486


namespace buns_left_l206_206257

theorem buns_left (buns_initial : ℕ) (h1 : buns_initial = 15)
                  (x : ℕ) (h2 : 13 * x ≤ buns_initial)
                  (buns_taken_by_bimbo : ℕ) (h3 : buns_taken_by_bimbo = x)
                  (buns_taken_by_little_boy : ℕ) (h4 : buns_taken_by_little_boy = 3 * x)
                  (buns_taken_by_karlsson : ℕ) (h5 : buns_taken_by_karlsson = 9 * x)
                  :
                  buns_initial - (buns_taken_by_bimbo + buns_taken_by_little_boy + buns_taken_by_karlsson) = 2 :=
by
  sorry

end buns_left_l206_206257


namespace distinct_prime_factors_90_l206_206116

def prime_factors (n : Nat) : List Nat :=
  sorry -- Implementation of prime factorization is skipped

noncomputable def num_distinct_prime_factors (n : Nat) : Nat :=
  (List.toFinset (prime_factors n)).card

theorem distinct_prime_factors_90 : num_distinct_prime_factors 90 = 3 :=
  by sorry

end distinct_prime_factors_90_l206_206116


namespace max_y_value_l206_206279

theorem max_y_value (x y : Int) (h : x * y + 3 * x + 2 * y = -4) : y ≤ -1 :=
by sorry

end max_y_value_l206_206279


namespace number_of_students_l206_206114

variable (F S J R T : ℕ)

axiom freshman_more_than_junior : F = (5 * J) / 4
axiom sophomore_fewer_than_freshman : S = 9 * F / 10
axiom total_students : T = F + S + J + R
axiom seniors_total : R = T / 5
axiom given_sophomores : S = 144

theorem number_of_students (T : ℕ) : T = 540 :=
by 
  sorry

end number_of_students_l206_206114


namespace additional_investment_interest_rate_l206_206031

theorem additional_investment_interest_rate :
  let initial_investment := 2400
  let initial_rate := 0.05
  let additional_investment := 600
  let total_investment := initial_investment + additional_investment
  let desired_total_income := 0.06 * total_investment
  let income_from_initial := initial_rate * initial_investment
  let additional_income_needed := desired_total_income - income_from_initial
  let additional_rate := additional_income_needed / additional_investment
  additional_rate * 100 = 10 :=
by
  sorry

end additional_investment_interest_rate_l206_206031


namespace vector_parallel_y_value_l206_206128

theorem vector_parallel_y_value (y : ℝ) 
  (a : ℝ × ℝ := (3, 2)) 
  (b : ℝ × ℝ := (6, y)) 
  (h_parallel : ∃ k : ℝ, b = (k * a.1, k * a.2)) : 
  y = 4 :=
by sorry

end vector_parallel_y_value_l206_206128


namespace find_p_q_sum_l206_206641

-- Define the conditions
def p (q : ℤ) : ℤ := q + 20

theorem find_p_q_sum (p q : ℤ) (hp : p * q = 1764) (hq : p - q = 20) :
  p + q = 86 :=
  sorry

end find_p_q_sum_l206_206641


namespace number_of_outfits_l206_206319

theorem number_of_outfits (num_shirts : ℕ) (num_pants : ℕ) (num_shoe_types : ℕ) (shoe_styles_per_type : ℕ) (h_shirts : num_shirts = 4) (h_pants : num_pants = 4) (h_shoes : num_shoe_types = 2) (h_styles : shoe_styles_per_type = 2) :
  num_shirts * num_pants * (num_shoe_types * shoe_styles_per_type) = 64 :=
by {
  sorry
}

end number_of_outfits_l206_206319


namespace find_removed_number_l206_206067

def list : List ℕ := [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]

def target_average : ℝ := 8.2

theorem find_removed_number (n : ℕ) (h : n ∈ list) :
  (list.sum - n) / (list.length - 1) = target_average -> n = 5 := by
  sorry

end find_removed_number_l206_206067


namespace watch_cost_price_l206_206196

theorem watch_cost_price (SP_loss SP_gain CP : ℝ) 
  (h1 : SP_loss = 0.9 * CP) 
  (h2 : SP_gain = 1.04 * CP) 
  (h3 : SP_gain - SP_loss = 196) 
  : CP = 1400 := 
sorry

end watch_cost_price_l206_206196


namespace binomial_16_4_l206_206950

theorem binomial_16_4 : Nat.choose 16 4 = 1820 :=
  sorry

end binomial_16_4_l206_206950


namespace point_b_in_third_quadrant_l206_206752

-- Definitions of the points with their coordinates
def PointA : ℝ × ℝ := (2, 3)
def PointB : ℝ × ℝ := (-1, -4)
def PointC : ℝ × ℝ := (-4, 1)
def PointD : ℝ × ℝ := (5, -3)

-- Definition of a point being in the third quadrant
def inThirdQuadrant (p : ℝ × ℝ) : Prop := p.1 < 0 ∧ p.2 < 0

-- The main Theorem to prove that PointB is in the third quadrant
theorem point_b_in_third_quadrant : inThirdQuadrant PointB :=
by sorry

end point_b_in_third_quadrant_l206_206752


namespace needed_people_l206_206938

theorem needed_people (n t t' k m : ℕ) (h1 : n = 6) (h2 : t = 8) (h3 : t' = 3) 
    (h4 : k = n * t) (h5 : k = m * t') : m - n = 10 :=
by
  sorry

end needed_people_l206_206938


namespace number_of_ordered_pairs_l206_206556

theorem number_of_ordered_pairs : 
  ∃ n, n = 325 ∧ ∀ (a b : ℤ), 
    1 ≤ a ∧ a ≤ 50 ∧ a % 2 = 1 ∧ 
    0 ≤ b ∧ b % 2 = 0 ∧ 
    ∃ r s : ℤ, r + s = -a ∧ r * s = b :=
sorry

end number_of_ordered_pairs_l206_206556


namespace vacation_books_l206_206488

-- Define the number of mystery, fantasy, and biography novels.
def num_mystery : ℕ := 3
def num_fantasy : ℕ := 4
def num_biography : ℕ := 3

-- Define the condition that we want to choose three books with no more than one from each genre.
def num_books_to_choose : ℕ := 3
def max_books_per_genre : ℕ := 1

-- The number of ways to choose one book from each genre
def num_combinations (m f b : ℕ) : ℕ :=
  m * f * b

-- Prove that the number of possible sets of books is 36
theorem vacation_books : num_combinations num_mystery num_fantasy num_biography = 36 := by
  sorry

end vacation_books_l206_206488


namespace angle_complement_supplement_l206_206448

theorem angle_complement_supplement (x : ℝ) (h : 90 - x = 3 / 4 * (180 - x)) : x = 180 :=
by
  sorry

end angle_complement_supplement_l206_206448


namespace parabola_coeff_sum_l206_206681

theorem parabola_coeff_sum (a b c : ℤ) (h₁ : a * (1:ℤ)^2 + b * 1 + c = 3)
                                      (h₂ : a * (-1)^2 + b * (-1) + c = 5)
                                      (vertex : ∀ x, a * (x + 1)^2 + 1 = a * x^2 + bx + c) :
a + b + c = 3 := 
sorry

end parabola_coeff_sum_l206_206681


namespace max_quartets_in_5x5_max_quartets_in_mxn_l206_206198

def quartet (c : Nat) : Bool := 
  c > 0

theorem max_quartets_in_5x5 : ∃ q, q = 5 ∧ 
  quartet q := by
  sorry

theorem max_quartets_in_mxn 
  (m n : Nat) (Hmn : m > 0 ∧ n > 0) :
  (∃ q, q = (m * (n - 1)) / 4 ∧ quartet q) ∨ 
  (∃ q, q = (m * (n - 1) - 2) / 4 ∧ quartet q) := by
  sorry

end max_quartets_in_5x5_max_quartets_in_mxn_l206_206198


namespace savings_wednesday_l206_206442

variable (m t s w : ℕ)

theorem savings_wednesday :
  m = 15 → t = 28 → s = 28 → 2 * s = 56 → 
  m + t + w = 56 → w = 13 :=
by
  intros h₁ h₂ h₃ h₄ h₅
  sorry

end savings_wednesday_l206_206442


namespace remainder_of_x_pow_77_eq_6_l206_206093

theorem remainder_of_x_pow_77_eq_6 (x : ℤ) (h : x^77 % 7 = 6) : x^77 % 7 = 6 :=
by
  sorry

end remainder_of_x_pow_77_eq_6_l206_206093


namespace possible_values_for_D_l206_206857

def distinct_digits (A B C D E : ℕ) : Prop :=
  A ≠ B ∧ A ≠ C ∧ A ≠ D ∧ A ≠ E ∧ B ≠ C ∧ B ≠ D ∧ B ≠ E ∧ C ≠ D ∧ C ≠ E ∧ D ≠ E

def digits_range (A B C D E : ℕ) : Prop :=
  0 ≤ A ∧ A ≤ 9 ∧ 0 ≤ B ∧ B ≤ 9 ∧ 0 ≤ C ∧ C ≤ 9 ∧ 0 ≤ D ∧ D ≤ 9 ∧ 0 ≤ E ∧ E ≤ 9

def addition_equation (A B C D E : ℕ) : Prop :=
  A * 10000 + B * 1000 + C * 100 + D * 10 + B +
  B * 10000 + C * 1000 + A * 100 + D * 10 + E = 
  E * 10000 + D * 1000 + D * 100 + E * 10 + E

theorem possible_values_for_D : 
  ∀ (A B C D E : ℕ),
  distinct_digits A B C D E →
  digits_range A B C D E →
  addition_equation A B C D E →
  ∃ (S : Finset ℕ), (∀ d ∈ S, 0 ≤ d ∧ d ≤ 9) ∧ (S.card = 2) :=
by
  -- Proof omitted
  sorry

end possible_values_for_D_l206_206857


namespace loss_equals_cost_price_of_some_balls_l206_206409

-- Conditions
def cost_price_per_ball := 60
def selling_price_for_17_balls := 720
def number_of_balls := 17

-- Calculations
def total_cost_price := number_of_balls * cost_price_per_ball
def loss := total_cost_price - selling_price_for_17_balls

-- Proof statement
theorem loss_equals_cost_price_of_some_balls : (loss / cost_price_per_ball) = 5 :=
by
  -- Proof would go here
  sorry

end loss_equals_cost_price_of_some_balls_l206_206409


namespace maximum_area_of_rectangular_playground_l206_206444

theorem maximum_area_of_rectangular_playground (P : ℕ) (A : ℕ) (h : P = 150) :
  ∃ (x y : ℕ), x + y = 75 ∧ A ≤ x * y ∧ A = 1406 :=
sorry

end maximum_area_of_rectangular_playground_l206_206444


namespace floor_equation_solution_l206_206835

theorem floor_equation_solution (a b : ℝ) :
  (∀ x y : ℝ, ⌊a * x + b * y⌋ + ⌊b * x + a * y⌋ = (a + b) * ⌊x + y⌋) → (a = 0 ∧ b = 0) ∨ (a = 1 ∧ b = 1) := by
  sorry

end floor_equation_solution_l206_206835


namespace sum_of_squares_and_product_l206_206470

theorem sum_of_squares_and_product (x y : ℕ) 
  (h1 : x^2 + y^2 = 130) 
  (h2 : x * y = 36) : 
  x + y = Real.sqrt 202 := 
by
  sorry

end sum_of_squares_and_product_l206_206470


namespace smallest_number_of_cubes_l206_206070

noncomputable def container_cubes (length_ft : ℕ) (height_ft : ℕ) (width_ft : ℕ) (prime_inch : ℕ) : ℕ :=
  let length_inch := length_ft * 12
  let height_inch := height_ft * 12
  let width_inch := width_ft * 12
  (length_inch / prime_inch) * (height_inch / prime_inch) * (width_inch / prime_inch)

theorem smallest_number_of_cubes :
  container_cubes 60 24 30 3 = 2764800 :=
by
  sorry

end smallest_number_of_cubes_l206_206070


namespace largest_integer_remainder_l206_206485

theorem largest_integer_remainder :
  ∃ (a : ℤ), a < 61 ∧ a % 6 = 5 ∧ ∀ b : ℤ, b < 61 ∧ b % 6 = 5 → b ≤ a :=
by
  sorry

end largest_integer_remainder_l206_206485


namespace gcd_of_ropes_l206_206508

theorem gcd_of_ropes : Nat.gcd (Nat.gcd 45 75) 90 = 15 := 
by
  sorry

end gcd_of_ropes_l206_206508


namespace cost_of_six_dozen_l206_206458

variable (cost_of_four_dozen : ℕ)
variable (dozens_to_purchase : ℕ)

theorem cost_of_six_dozen :
  cost_of_four_dozen = 24 →
  dozens_to_purchase = 6 →
  (dozens_to_purchase * (cost_of_four_dozen / 4)) = 36 :=
by
  intros h1 h2
  sorry

end cost_of_six_dozen_l206_206458


namespace sum_in_base_4_l206_206997

theorem sum_in_base_4 : 
  let n1 := 2
  let n2 := 23
  let n3 := 132
  let n4 := 1320
  let sum := 20200
  n1 + n2 + n3 + n4 = sum := 
by
  sorry

end sum_in_base_4_l206_206997


namespace necessary_not_sufficient_l206_206282

variable (a b : ℝ)

theorem necessary_not_sufficient : 
  (a > b) -> ¬ (a > b+1) ∨ (a > b+1 ∧ a > b) :=
by
  intro h
  have h1 : ¬ (a > b+1) := sorry
  have h2 : (a > b+1 -> a > b) := sorry
  exact Or.inl h1

end necessary_not_sufficient_l206_206282


namespace find_f_2015_l206_206901

noncomputable def f : ℝ → ℝ :=
sorry

lemma f_period : ∀ x : ℝ, f (x + 8) = f x :=
sorry

axiom f_func_eq : ∀ x : ℝ, f (x + 2) = (1 + f x) / (1 - f x)

axiom f_initial : f 1 = 1 / 4

theorem find_f_2015 : f 2015 = -3 / 5 :=
sorry

end find_f_2015_l206_206901


namespace no_n_makes_g_multiple_of_5_and_7_l206_206250

def g (n : ℕ) : ℕ := 4 + 2 * n + 3 * n^2 + n^3 + 4 * n^4 + 3 * n^5

theorem no_n_makes_g_multiple_of_5_and_7 :
  ¬ ∃ n, (2 ≤ n ∧ n ≤ 100) ∧ (g n % 5 = 0 ∧ g n % 7 = 0) :=
by
  -- Proof goes here
  sorry

end no_n_makes_g_multiple_of_5_and_7_l206_206250


namespace largest_r_l206_206998

theorem largest_r (p q r : ℝ) (h1 : p + q + r = 6) (h2 : p*q + p*r + q*r = 8) : 
  r ≤ 2 + Real.sqrt (20/3) := 
sorry

end largest_r_l206_206998


namespace solution_quadrant_I_l206_206098

theorem solution_quadrant_I (c x y : ℝ) :
  (x - y = 2 ∧ c * x + y = 3 ∧ x > 0 ∧ y > 0) ↔ (-1 < c ∧ c < 3/2) := by
  sorry

end solution_quadrant_I_l206_206098


namespace sum_of_squares_of_roots_l206_206094

theorem sum_of_squares_of_roots :
  (∃ x1 x2 : ℝ, 5 * x1^2 - 3 * x1 - 11 = 0 ∧ 5 * x2^2 - 3 * x2 - 11 = 0 ∧ x1 ≠ x2) →
  (x1 + x2 = 3 / 5 ∧ x1 * x2 = -11 / 5) →
  (x1^2 + x2^2 = 119 / 25) :=
by intro h1 h2; sorry

end sum_of_squares_of_roots_l206_206094


namespace like_terms_exponents_l206_206503

theorem like_terms_exponents (n m : ℕ) (h1 : n + 2 = 3) (h2 : 2 * m - 1 = 3) : n = 1 ∧ m = 2 :=
by sorry

end like_terms_exponents_l206_206503


namespace logistics_center_correct_l206_206298

noncomputable def rectilinear_distance (P Q : ℝ × ℝ) : ℝ :=
  |P.1 - Q.1| + |P.2 - Q.2|

def A : ℝ × ℝ := (2, 3)
def B : ℝ × ℝ := (-6, 9)
def C : ℝ × ℝ := (-3, -8)

def logistics_center : ℝ × ℝ := (-5, 0)

theorem logistics_center_correct : 
  ∀ L : ℝ × ℝ, 
  (rectilinear_distance L A = rectilinear_distance L B) ∧ 
  (rectilinear_distance L B = rectilinear_distance L C) ∧
  (rectilinear_distance L A = rectilinear_distance L C) → 
  L = logistics_center := sorry

end logistics_center_correct_l206_206298


namespace socks_count_l206_206862

theorem socks_count
  (black_socks : ℕ) 
  (white_socks : ℕ)
  (H1 : white_socks = 4 * black_socks)
  (H2 : black_socks = 6)
  (H3 : white_socks / 2 = white_socks - (white_socks / 2)) :
  (white_socks / 2) - black_socks = 6 := by
  sorry

end socks_count_l206_206862


namespace bookstore_earnings_difference_l206_206324

def base_price_TOP := 8.0
def base_price_ABC := 23.0
def discount_TOP := 0.10
def discount_ABC := 0.05
def sales_tax := 0.07
def num_TOP_sold := 13
def num_ABC_sold := 4

def discounted_price (base_price discount : Float) : Float :=
  base_price * (1.0 - discount)

def final_price (discounted_price tax : Float) : Float :=
  discounted_price * (1.0 + tax)

def total_earnings (final_price : Float) (quantity : Nat) : Float :=
  final_price * (quantity.toFloat)

theorem bookstore_earnings_difference :
  let discounted_price_TOP := discounted_price base_price_TOP discount_TOP
  let discounted_price_ABC := discounted_price base_price_ABC discount_ABC
  let final_price_TOP := final_price discounted_price_TOP sales_tax
  let final_price_ABC := final_price discounted_price_ABC sales_tax
  let total_earnings_TOP := total_earnings final_price_TOP num_TOP_sold
  let total_earnings_ABC := total_earnings final_price_ABC num_ABC_sold
  total_earnings_TOP - total_earnings_ABC = 6.634 :=
by
  sorry

end bookstore_earnings_difference_l206_206324


namespace simplify_fraction_l206_206296

variable {R : Type*} [Field R]
variables (x y z : R)

theorem simplify_fraction : (6 * x * y / (5 * z ^ 2)) * (10 * z ^ 3 / (9 * x * y)) = (4 * z) / 3 := by
  sorry

end simplify_fraction_l206_206296


namespace constant_k_for_linear_function_l206_206498

theorem constant_k_for_linear_function (k : ℝ) (h : ∀ (x : ℝ), y = x^(k-1) + 2 → y = a * x + b) : k = 2 :=
sorry

end constant_k_for_linear_function_l206_206498


namespace maria_cookies_left_l206_206192

def maria_cookies (initial: ℕ) (to_friend: ℕ) (to_family_divisor: ℕ) (eats: ℕ) : ℕ :=
  (initial - to_friend) / to_family_divisor - eats

theorem maria_cookies_left (h : maria_cookies 19 5 2 2 = 5): true :=
by trivial

end maria_cookies_left_l206_206192


namespace common_chord_of_circles_l206_206352

theorem common_chord_of_circles :
  ∀ (x y : ℝ), (x^2 + y^2 + 2 * x = 0) ∧ (x^2 + y^2 - 4 * y = 0) → (x + 2 * y = 0) :=
by
  sorry

end common_chord_of_circles_l206_206352


namespace length_AF_is_25_l206_206870

open Classical

noncomputable def length_AF : ℕ :=
  let AB := 5
  let AC := 11
  let DE := 8
  let EF := 4
  let BC := AC - AB
  let CD := BC / 3
  let AF := AB + BC + CD + DE + EF
  AF

theorem length_AF_is_25 :
  length_AF = 25 := by
  sorry

end length_AF_is_25_l206_206870


namespace age_of_hospital_l206_206203

theorem age_of_hospital (grant_current_age : ℕ) (future_ratio : ℚ)
                        (grant_future_age : grant_current_age + 5 = 30)
                        (hospital_age_ratio : future_ratio = 2 / 3) :
                        (grant_current_age = 25) → 
                        (grant_current_age + 5 = future_ratio * (grant_current_age + 5 + 5)) →
                        (grant_current_age + 5 + 5 - 5 = 40) :=
by
  sorry

end age_of_hospital_l206_206203


namespace circle_center_radius_l206_206946

theorem circle_center_radius :
  ∃ (h : ℝ × ℝ) (r : ℝ),
    (h = (1, -3)) ∧ (r = 2) ∧ ∀ x y : ℝ, 
    (x - h.1)^2 + (y - h.2)^2 = 4 → x^2 + y^2 - 2*x + 6*y + 6 = 0 :=
sorry

end circle_center_radius_l206_206946


namespace t_lt_s_l206_206900

noncomputable def t : ℝ := Real.sqrt 11 - 3
noncomputable def s : ℝ := Real.sqrt 7 - Real.sqrt 5

theorem t_lt_s : t < s :=
by
  sorry

end t_lt_s_l206_206900


namespace fraction_to_decimal_l206_206469

theorem fraction_to_decimal : (7 / 16 : ℝ) = 0.4375 := by
  sorry

end fraction_to_decimal_l206_206469


namespace solve_fraction_equation_l206_206259

theorem solve_fraction_equation :
  ∀ (x : ℚ), (5 * x + 3) / (7 * x - 4) = 4128 / 4386 → x = 115 / 27 := by
  sorry

end solve_fraction_equation_l206_206259


namespace circumscribed_circle_radius_l206_206780

theorem circumscribed_circle_radius (b c : ℝ) (A : ℝ) (R : ℝ)
  (h1 : b = 6) (h2 : c = 2) (h3 : A = π / 3) :
  R = (2 * Real.sqrt 21) / 3 :=
by
  sorry

end circumscribed_circle_radius_l206_206780


namespace geometric_sequence_a7_l206_206030

noncomputable def geometric_sequence (a : ℕ → ℝ) := ∃ q : ℝ, ∀ n : ℕ, a (n + 1) = a n * q

theorem geometric_sequence_a7 (a : ℕ → ℝ) (h_geom : geometric_sequence a) (h_a1 : a 1 = 2) (h_a3 : a 3 = 4) : a 7 = 16 := 
sorry

end geometric_sequence_a7_l206_206030


namespace minimum_additional_squares_to_symmetry_l206_206694

-- Define the type for coordinates in the grid
structure Coord where
  x : Nat
  y : Nat

-- Define the conditions
def initial_shaded_squares : List Coord := [
  ⟨2, 4⟩, ⟨3, 2⟩, ⟨5, 1⟩, ⟨1, 4⟩
]

def grid_size : Coord := ⟨6, 5⟩

def vertical_line_of_symmetry : Nat := 3 -- between columns 3 and 4
def horizontal_line_of_symmetry : Nat := 2 -- between rows 2 and 3

-- Define reflection across lines of symmetry
def reflect_vertical (c : Coord) : Coord :=
  ⟨2 * vertical_line_of_symmetry - c.x, c.y⟩

def reflect_horizontal (c : Coord) : Coord :=
  ⟨c.x, 2 * horizontal_line_of_symmetry - c.y⟩

def reflect_both (c : Coord) : Coord :=
  reflect_vertical (reflect_horizontal c)

-- Define the theorem
theorem minimum_additional_squares_to_symmetry :
  ∃ (additional_squares : Nat), additional_squares = 5 := 
sorry

end minimum_additional_squares_to_symmetry_l206_206694


namespace compare_y_coordinates_l206_206014

theorem compare_y_coordinates (x₁ x₂ y₁ y₂ : ℝ) 
  (h₁: (x₁ = -3) ∧ (y₁ = 2 * x₁ - 1)) 
  (h₂: (x₂ = -5) ∧ (y₂ = 2 * x₂ - 1)) : 
  y₁ > y₂ := 
by 
  sorry

end compare_y_coordinates_l206_206014


namespace calculate_f2_f_l206_206728

variable {f : ℝ → ℝ}

-- Definition of the conditions
def tangent_line_at_x2 (f : ℝ → ℝ) : Prop :=
  ∃ (L : ℝ → ℝ), (∀ x, L x = -x + 1) ∧ (∀ x, f x = L x + (f x - L 2))

theorem calculate_f2_f'2 (h : tangent_line_at_x2 f) :
  f 2 + deriv f 2 = -2 :=
sorry

end calculate_f2_f_l206_206728


namespace integer_solution_range_l206_206779

theorem integer_solution_range {m : ℝ} : 
  (∀ x : ℤ, -1 ≤ x → x < m → (x = -1 ∨ x = 0)) ↔ (0 < m ∧ m ≤ 1) :=
by 
  sorry

end integer_solution_range_l206_206779


namespace percent_decrease_l206_206980

theorem percent_decrease (original_price sale_price : ℝ) (h₀ : original_price = 100) (h₁ : sale_price = 30) :
  (original_price - sale_price) / original_price * 100 = 70 :=
by
  rw [h₀, h₁]
  norm_num

end percent_decrease_l206_206980


namespace orange_profit_loss_l206_206487

variable (C : ℝ) -- Cost price of one orange in rupees

-- Conditions as hypotheses
theorem orange_profit_loss :
  (1 / 16 - C) / C * 100 = 4 :=
by
  have h1 : 1.28 * C = 1 / 12 := sorry
  have h2 : C = 1 / (12 * 1.28) := sorry
  have h3 : C = 1 / 15.36 := sorry
  have h4 : (1/16 - C) = 1 / 384 := sorry
  -- Proof of main statement here
  sorry

end orange_profit_loss_l206_206487


namespace four_digit_number_count_l206_206432

theorem four_digit_number_count :
  (∃ (n : ℕ), n ≥ 1000 ∧ n < 10000 ∧ 
    ((n / 1000 < 5 ∧ (n / 100) % 10 < 5) ∨ (n / 1000 > 5 ∧ (n / 100) % 10 > 5)) ∧ 
    (((n % 100) / 10 < 5 ∧ n % 10 < 5) ∨ ((n % 100) / 10 > 5 ∧ n % 10 > 5))) →
    ∃ (count : ℕ), count = 1681 :=
by
  sorry

end four_digit_number_count_l206_206432


namespace find_three_digit_number_l206_206313

-- Define the function that calculates the total number of digits required
def total_digits (x : ℕ) : ℕ :=
  (if x >= 1 then 9 else 0) +
  (if x >= 10 then 90 * 2 else 0) +
  (if x >= 100 then 3 * (x - 99) else 0)

theorem find_three_digit_number : ∃ x : ℕ, 100 ≤ x ∧ x < 1000 ∧ 2 * x = total_digits x := by
  sorry

end find_three_digit_number_l206_206313


namespace solve_simultaneous_equations_l206_206489

theorem solve_simultaneous_equations :
  (∃ x y : ℝ, x^2 + 3 * y = 10 ∧ 3 + y = 10 / x) ↔ 
  (x = 3 ∧ y = 1 / 3) ∨ 
  (x = 2 ∧ y = 2) ∨ 
  (x = -5 ∧ y = -5) := by sorry

end solve_simultaneous_equations_l206_206489


namespace rate_per_sq_meter_l206_206687

def length : ℝ := 5.5
def width : ℝ := 3.75
def totalCost : ℝ := 14437.5

theorem rate_per_sq_meter : (totalCost / (length * width)) = 700 := 
by sorry

end rate_per_sq_meter_l206_206687


namespace forgot_to_mow_l206_206530

-- Definitions
def earning_per_lawn : ℕ := 9
def lawns_to_mow : ℕ := 12
def actual_earning : ℕ := 36

-- Statement to prove
theorem forgot_to_mow : (lawns_to_mow - (actual_earning / earning_per_lawn)) = 8 := by
  sorry

end forgot_to_mow_l206_206530


namespace solve_quadratic_l206_206546

theorem solve_quadratic (x : ℝ) : x^2 - 2*x = 0 ↔ (x = 0 ∨ x = 2) :=
by
  sorry

end solve_quadratic_l206_206546


namespace students_taking_neither_l206_206427

-- Defining given conditions as Lean definitions
def total_students : ℕ := 70
def students_math : ℕ := 42
def students_physics : ℕ := 35
def students_chemistry : ℕ := 25
def students_math_physics : ℕ := 18
def students_math_chemistry : ℕ := 10
def students_physics_chemistry : ℕ := 8
def students_all_three : ℕ := 5

-- Define the problem to prove
theorem students_taking_neither : total_students
  - (students_math - students_math_physics - students_math_chemistry + students_all_three
    + students_physics - students_math_physics - students_physics_chemistry + students_all_three
    + students_chemistry - students_math_chemistry - students_physics_chemistry + students_all_three
    + students_math_physics - students_all_three
    + students_math_chemistry - students_all_three
    + students_physics_chemistry - students_all_three
    + students_all_three) = 0 := by
  sorry

end students_taking_neither_l206_206427


namespace count_triples_l206_206110

open Set

theorem count_triples 
  (A B C : Set ℕ) 
  (h_union : A ∪ B ∪ C = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10})
  (h_inter : A ∩ B ∩ C = ∅) :
  (∃ n : ℕ, n = 60466176) :=
by
  -- Proof can be filled in here
  sorry

end count_triples_l206_206110


namespace rogers_spending_l206_206087

theorem rogers_spending (B m p : ℝ) (H1 : m = 0.25 * (B - p)) (H2 : p = 0.10 * (B - m)) : 
  m + p = (4 / 13) * B :=
sorry

end rogers_spending_l206_206087


namespace chess_tournament_games_l206_206054

theorem chess_tournament_games (n : ℕ) (h : n * (n - 1) / 2 = 190) : n = 20 :=
sorry

end chess_tournament_games_l206_206054


namespace log_relation_l206_206046

theorem log_relation (a b : ℝ) (log_7 : ℝ → ℝ) (log_6 : ℝ → ℝ) (log_6_343 : log_6 343 = a) (log_7_18 : log_7 18 = b) :
  a = 6 / (b + 2 * log_7 2) :=
by
  sorry

end log_relation_l206_206046


namespace distance_between_neg5_and_neg1_l206_206464

theorem distance_between_neg5_and_neg1 : 
  dist (-5 : ℝ) (-1) = 4 := by
sorry

end distance_between_neg5_and_neg1_l206_206464


namespace tim_tasks_per_day_l206_206671

theorem tim_tasks_per_day (earnings_per_task : ℝ) (days_per_week : ℕ) (weekly_earnings : ℝ) :
  earnings_per_task = 1.2 ∧ days_per_week = 6 ∧ weekly_earnings = 720 → (weekly_earnings / days_per_week / earnings_per_task = 100) :=
by
  sorry

end tim_tasks_per_day_l206_206671


namespace prime_addition_fraction_equivalence_l206_206754

theorem prime_addition_fraction_equivalence : 
  ∃ n : ℕ, Prime n ∧ (4 + n) * 8 = (7 + n) * 7 ∧ n = 17 := 
sorry

end prime_addition_fraction_equivalence_l206_206754


namespace num_possible_pairs_l206_206536

theorem num_possible_pairs (a b : ℕ) (h1 : b > a) (h2 : (a - 8) * (b - 8) = 32) : 
    (∃ n, n = 3) :=
by { sorry }

end num_possible_pairs_l206_206536


namespace simplify_expression_l206_206873

theorem simplify_expression :
  2^2 + 2^2 + 2^2 + 2^2 = 2^4 :=
sorry

end simplify_expression_l206_206873


namespace find_angle_A_determine_triangle_shape_l206_206145

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


end find_angle_A_determine_triangle_shape_l206_206145


namespace minimum_workers_in_team_A_l206_206663

variable (a b c : ℤ)

theorem minimum_workers_in_team_A (h1 : b + 90 = 2 * (a - 90))
                               (h2 : a + c = 6 * (b - c)) :
  ∃ a ≥ 148, a = 153 :=
by
  sorry

end minimum_workers_in_team_A_l206_206663


namespace calvin_weeks_buying_chips_l206_206505

variable (daily_spending : ℝ := 0.50)
variable (days_per_week : ℝ := 5)
variable (total_spending : ℝ := 10)
variable (spending_per_week := daily_spending * days_per_week)

theorem calvin_weeks_buying_chips :
  total_spending / spending_per_week = 4 := by
  sorry

end calvin_weeks_buying_chips_l206_206505


namespace shortest_path_from_vertex_to_center_of_non_adjacent_face_l206_206727

noncomputable def shortest_path_on_cube (edge_length : ℝ) : ℝ :=
  edge_length + (edge_length * Real.sqrt 2 / 2)

theorem shortest_path_from_vertex_to_center_of_non_adjacent_face :
  shortest_path_on_cube 1 = 1 + Real.sqrt 2 / 2 :=
by
  sorry

end shortest_path_from_vertex_to_center_of_non_adjacent_face_l206_206727


namespace profit_percentage_with_discount_correct_l206_206693

variable (CP SP_without_discount Discounted_SP : ℝ)
variable (profit_without_discount profit_with_discount : ℝ)
variable (discount_percentage profit_percentage_without_discount profit_percentage_with_discount : ℝ)
variable (h1 : CP = 100)
variable (h2 : SP_without_discount = CP + profit_without_discount)
variable (h3 : profit_without_discount = 1.20 * CP)
variable (h4 : Discounted_SP = SP_without_discount - discount_percentage * SP_without_discount)
variable (h5 : discount_percentage = 0.05)
variable (h6 : profit_with_discount = Discounted_SP - CP)
variable (h7 : profit_percentage_with_discount = (profit_with_discount / CP) * 100)

theorem profit_percentage_with_discount_correct : profit_percentage_with_discount = 109 := by
  sorry

end profit_percentage_with_discount_correct_l206_206693


namespace simplify_cubicroot_1600_l206_206629

theorem simplify_cubicroot_1600 : ∃ (c d : ℕ), c > 0 ∧ d > 0 ∧ (c^3 * d = 1600) ∧ (c + d = 102) := 
by 
  sorry

end simplify_cubicroot_1600_l206_206629


namespace count_red_balls_l206_206373

/-- Given conditions:
  - The total number of balls in the bag is 100.
  - There are 50 white, 20 green, 10 yellow, and 3 purple balls.
  - The probability that a ball will be neither red nor purple is 0.8.
  Prove that the number of red balls is 17. -/
theorem count_red_balls (total_balls white_balls green_balls yellow_balls purple_balls red_balls : ℕ)
  (h1 : total_balls = 100)
  (h2 : white_balls = 50)
  (h3 : green_balls = 20)
  (h4 : yellow_balls = 10)
  (h5 : purple_balls = 3)
  (h6 : (white_balls + green_balls + yellow_balls) = 80)
  (h7 : (white_balls + green_balls + yellow_balls) / (total_balls : ℝ) = 0.8) :
  red_balls = 17 :=
by
  sorry

end count_red_balls_l206_206373


namespace football_goals_even_more_probable_l206_206218

-- Define the problem statement and conditions
variable (p_1 : ℝ) (h₀ : 0 ≤ p_1 ∧ p_1 ≤ 1) (h₁ : q_1 = 1 - p_1)

-- Define even and odd goal probabilities for the total match
def p : ℝ := p_1^2 + (1 - p_1)^2
def q : ℝ := 2 * p_1 * (1 - p_1)

-- The main statement to prove
theorem football_goals_even_more_probable (h₂ : q_1 = 1 - p_1) : p_1^2 + (1 - p_1)^2 ≥ 2 * p_1 * (1 - p_1) :=
  sorry

end football_goals_even_more_probable_l206_206218


namespace proof_n_eq_neg2_l206_206286

theorem proof_n_eq_neg2 (n : ℤ) (h : |n + 6| = 2 - n) : n = -2 := 
by
  sorry

end proof_n_eq_neg2_l206_206286


namespace solve_eq1_solve_eq2_l206_206706

theorem solve_eq1 (x : ℝ) : (x - 2)^2 - 16 = 0 ↔ x = 6 ∨ x = -2 :=
by sorry

theorem solve_eq2 (x : ℝ) : (x + 3)^3 = -27 ↔ x = -6 :=
by sorry

end solve_eq1_solve_eq2_l206_206706


namespace least_integer_square_l206_206537

theorem least_integer_square (x : ℤ) : x^2 = 2 * x + 72 → x = -6 := 
by
  intro h
  sorry

end least_integer_square_l206_206537


namespace equation_I_consecutive_integers_equation_II_consecutive_even_integers_l206_206888

theorem equation_I_consecutive_integers :
  ∃ (x y z : ℕ), x + y + z = 48 ∧ (x = y - 1) ∧ (z = y + 1) := sorry

theorem equation_II_consecutive_even_integers :
  ∃ (x y z w : ℕ), x + y + z + w = 52 ∧ (y = x + 2) ∧ (z = x + 4) ∧ (w = x + 6) := sorry

end equation_I_consecutive_integers_equation_II_consecutive_even_integers_l206_206888


namespace minimum_value_2x_3y_l206_206864

theorem minimum_value_2x_3y (x y : ℝ) (hx : 0 < x) (hy : 0 < y)
  (hxy : x^2 * y * (4 * x + 3 * y) = 3) :
  2 * x + 3 * y ≥ 2 * Real.sqrt 3 := by
  sorry

end minimum_value_2x_3y_l206_206864


namespace minimize_expression_l206_206385

theorem minimize_expression : ∃ c : ℝ, (∀ x : ℝ, (1/3 * x^2 + 7*x - 4) ≥ (1/3 * c^2 + 7*c - 4)) ∧ (c = -21/2) :=
sorry

end minimize_expression_l206_206385


namespace four_racers_meet_l206_206318

/-- In a circular auto race, four racers participate. Their cars start simultaneously from 
the same point and move at constant speeds, and for any three cars, there is a moment 
when they meet. Prove that after the start of the race, there will be a moment when all 
four cars meet. (Assume the race continues indefinitely in time.) -/
theorem four_racers_meet (V1 V2 V3 V4 : ℝ) (L : ℝ) (t : ℝ) 
  (h1 : 0 ≤ t) 
  (h2 : V1 ≤ V2 ∧ V2 ≤ V3 ∧ V3 ≤ V4)
  (h3 : ∀ t1 t2 t3, ∃ t, t1 * V1 = t ∧ t2 * V2 = t ∧ t3 * V3 = t) :
  ∃ t, t > 0 ∧ ∃ t', V1 * t' % L = 0 ∧ V2 * t' % L = 0 ∧ V3 * t' % L = 0 ∧ V4 * t' % L = 0 :=
sorry

end four_racers_meet_l206_206318


namespace eccentricity_range_l206_206191

variable {a b c : ℝ} (P : ℝ × ℝ) (F₁ F₂ : ℝ × ℝ) (e : ℝ)

-- Assume a > 0, b > 0, and the eccentricity of the hyperbola is given by c = e * a.
variable (a_pos : 0 < a) (b_pos : 0 < b) (hyperbola : (P.1 / a)^2 - (P.2 / b)^2 = 1)
variable (on_right_branch : P.1 > 0)
variable (foci_condition : dist P F₁ = 4 * dist P F₂)
variable (eccentricity_def : c = e * a)

theorem eccentricity_range : 1 < e ∧ e ≤ 5 / 3 := by
  sorry

end eccentricity_range_l206_206191


namespace KatieMarbles_l206_206292

variable {O P : ℕ}

theorem KatieMarbles :
  13 + O + P = 33 → P = 4 * O → 13 - O = 9 :=
by
  sorry

end KatieMarbles_l206_206292


namespace total_masks_correct_l206_206534

-- Define the conditions
def boxes := 18
def capacity_per_box := 15
def deficiency_per_box := 3
def masks_per_box := capacity_per_box - deficiency_per_box
def total_masks := boxes * masks_per_box

-- The theorem statement we need to prove
theorem total_masks_correct : total_masks = 216 := by
  unfold total_masks boxes masks_per_box capacity_per_box deficiency_per_box
  sorry

end total_masks_correct_l206_206534


namespace probability_of_girls_under_18_l206_206084

theorem probability_of_girls_under_18
  (total_members : ℕ)
  (girls : ℕ)
  (boys : ℕ)
  (underaged_girls : ℕ)
  (two_members_chosen : ℕ)
  (total_ways_to_choose_two : ℕ)
  (ways_to_choose_two_girls : ℕ)
  (ways_to_choose_at_least_one_underaged : ℕ)
  (prob : ℚ)
  : 
  total_members = 15 →
  girls = 8 →
  boys = 7 →
  underaged_girls = 3 →
  two_members_chosen = 2 →
  total_ways_to_choose_two = (Nat.choose total_members two_members_chosen) →
  ways_to_choose_two_girls = (Nat.choose girls two_members_chosen) →
  ways_to_choose_at_least_one_underaged = 
    (Nat.choose underaged_girls 1 * Nat.choose (girls - underaged_girls) 1 + Nat.choose underaged_girls 2) →
  prob = (ways_to_choose_at_least_one_underaged : ℚ) / (total_ways_to_choose_two : ℚ) →
  prob = 6 / 35 :=
by
  intros
  sorry

end probability_of_girls_under_18_l206_206084


namespace probability_xavier_yvonne_not_zelda_wendell_l206_206669

theorem probability_xavier_yvonne_not_zelda_wendell
  (P_Xavier_solves : ℚ)
  (P_Yvonne_solves : ℚ)
  (P_Zelda_solves : ℚ)
  (P_Wendell_solves : ℚ) :
  P_Xavier_solves = 1/4 →
  P_Yvonne_solves = 1/3 →
  P_Zelda_solves = 5/8 →
  P_Wendell_solves = 1/2 →
  (P_Xavier_solves * P_Yvonne_solves * (1 - P_Zelda_solves) * (1 - P_Wendell_solves)) = 1/64 :=
by
  intros h1 h2 h3 h4
  rw [h1, h2, h3, h4]
  simp
  sorry

end probability_xavier_yvonne_not_zelda_wendell_l206_206669


namespace sum_of_three_positive_integers_l206_206884

theorem sum_of_three_positive_integers (n : ℕ) (h : n ≥ 3) :
  ∃ k : ℕ, k = (n - 1) * (n - 2) / 2 := 
sorry

end sum_of_three_positive_integers_l206_206884


namespace range_of_m_l206_206824

theorem range_of_m (m : ℝ) 
  (p : m < 0) 
  (q : ∀ x : ℝ, x^2 + m * x + 1 > 0) : 
  -2 < m ∧ m < 0 :=
by
  sorry

end range_of_m_l206_206824


namespace find_angle_C_l206_206357

variable {A B C a b c : ℝ}

theorem find_angle_C (h1 : 0 < A) (h2 : A < π) (h3 : 0 < B) (h4 : B < π) (h5 : 0 < C) (h6 : C < π)
  (h7 : A + B + C = π) (h8 : a > 0) (h9 : b > 0) (h10 : c > 0) 
  (h11 : (a + b - c) * (a + b + c) = a * b) : C = 2 * π / 3 :=
by
  sorry

end find_angle_C_l206_206357


namespace emily_eggs_collected_l206_206842

theorem emily_eggs_collected :
  let number_of_baskets := 1525
  let eggs_per_basket := 37.5
  let total_eggs := number_of_baskets * eggs_per_basket
  total_eggs = 57187.5 :=
by
  sorry

end emily_eggs_collected_l206_206842


namespace initial_percentage_alcohol_l206_206474

-- Define the initial conditions
variables (P : ℚ) -- percentage of alcohol in the initial solution
variables (V1 V2 : ℚ) -- volumes of the initial solution and added alcohol
variables (C2 : ℚ) -- concentration of the resulting solution

-- Given the initial conditions and additional parameters
def initial_solution_volume : ℚ := 6
def added_alcohol_volume : ℚ := 1.8
def final_solution_volume : ℚ := initial_solution_volume + added_alcohol_volume
def final_solution_concentration : ℚ := 0.5 -- 50%

-- The amount of alcohol initially = (P / 100) * V1
-- New amount of alcohol after adding pure alcohol
-- This should equal to the final concentration of the new volume

theorem initial_percentage_alcohol : 
  (P / 100 * initial_solution_volume) + added_alcohol_volume = final_solution_concentration * final_solution_volume → 
  P = 35 :=
sorry

end initial_percentage_alcohol_l206_206474


namespace initial_oranges_count_l206_206632

theorem initial_oranges_count
  (initial_apples : ℕ := 50)
  (apple_cost : ℝ := 0.80)
  (orange_cost : ℝ := 0.50)
  (total_earnings : ℝ := 49)
  (remaining_apples : ℕ := 10)
  (remaining_oranges : ℕ := 6)
  : initial_oranges = 40 := 
by
  sorry

end initial_oranges_count_l206_206632


namespace shorter_piece_is_20_l206_206796

def shorter_piece_length (total_length : ℕ) (ratio : ℚ) (shorter_piece : ℕ) : Prop :=
    shorter_piece * 7 = 2 * (total_length - shorter_piece)

theorem shorter_piece_is_20 : ∀ (total_length : ℕ) (shorter_piece : ℕ), 
    total_length = 90 ∧
    shorter_piece_length total_length (2/7 : ℚ) shorter_piece ->
    shorter_piece = 20 :=
by
  intro total_length shorter_piece
  intro h
  have h_total_length : total_length = 90 := h.1
  have h_equation : shorter_piece_length total_length (2/7 : ℚ) shorter_piece := h.2
  sorry

end shorter_piece_is_20_l206_206796


namespace arithmetic_mean_solve_x_l206_206712

theorem arithmetic_mean_solve_x (x : ℚ) :
  (x + 10 + 20 + 3 * x + 15 + 3 * x + 6) / 5 = 30 → x = 99 / 7 :=
by 
sorry

end arithmetic_mean_solve_x_l206_206712


namespace who_is_werewolf_choose_companion_l206_206695

-- Define inhabitants with their respective statements
inductive Inhabitant
| A | B | C

-- Assume each inhabitant can be either a knight (truth-teller) or a liar
def is_knight (i : Inhabitant) : Prop := sorry

-- Define statements made by each inhabitant
def A_statement : Prop := ∃ werewolf : Inhabitant, werewolf = Inhabitant.C
def B_statement : Prop := ¬(∃ werewolf : Inhabitant, werewolf = Inhabitant.B)
def C_statement : Prop := ∃ liar1 liar2 : Inhabitant, liar1 ≠ liar2 ∧ liar1 ≠ Inhabitant.C ∧ liar2 ≠ Inhabitant.C

-- Define who is the werewolf (liar)
def is_werewolf (i : Inhabitant) : Prop := ¬is_knight i

-- The given conditions from statements
axiom A_is_knight : is_knight Inhabitant.A ↔ A_statement
axiom B_is_knight : is_knight Inhabitant.B ↔ B_statement
axiom C_is_knight : is_knight Inhabitant.C ↔ C_statement

-- The conclusion: C is the werewolf and thus a liar.
theorem who_is_werewolf : is_werewolf Inhabitant.C :=
by sorry

-- Choosing a companion: 
-- If C is a werewolf, we prefer to pick A as a companion over B or C.
theorem choose_companion (worry_about_werewolf : Bool) : Inhabitant :=
if worry_about_werewolf then Inhabitant.A else sorry

end who_is_werewolf_choose_companion_l206_206695


namespace boat_speed_in_still_water_l206_206041

theorem boat_speed_in_still_water
  (V_s : ℝ) (t : ℝ) (d : ℝ) (V_b : ℝ)
  (h_stream_speed : V_s = 4)
  (h_travel_time : t = 7)
  (h_distance : d = 196)
  (h_downstream_speed : d / t = V_b + V_s) :
  V_b = 24 :=
by
  sorry

end boat_speed_in_still_water_l206_206041


namespace valve_rate_difference_l206_206336

section ValveRates

-- Conditions
variables (V1 V2 : ℝ) (t1 t2 : ℝ) (C : ℝ)
-- Given Conditions
-- The first valve alone would fill the pool in 2 hours (120 minutes)
def valve1_rate := V1 = 12000 / 120
-- With both valves open, the pool will be filled with water in 48 minutes
def combined_rate := V1 + V2 = 12000 / 48
-- Capacity of the pool is 12000 cubic meters
def pool_capacity := C = 12000

-- The Proof of the question
theorem valve_rate_difference : V1 = 100 → V2 = 150 → (V2 - V1) = 50 :=
by
  intros hV1 hV2
  rw [hV1, hV2]
  norm_num

end ValveRates

end valve_rate_difference_l206_206336


namespace infinitely_many_m_l206_206921

theorem infinitely_many_m (r : ℕ) (n : ℕ) (h_r : r > 1) (h_n : n > 0) : 
  ∃ m, m = 4 * r ^ 4 ∧ ¬Prime (n^4 + m) :=
by
  sorry

end infinitely_many_m_l206_206921


namespace arith_seq_sum_correct_l206_206986

-- Define the arithmetic sequence given the first term and common difference
def arith_seq (a₁ d : ℤ) (n : ℕ) : ℤ :=
  a₁ + (n - 1) * d

-- Define the sum of the first n terms of an arithmetic sequence
def arith_seq_sum (a₁ d : ℤ) (n : ℕ) : ℤ :=
  (n * (2 * a₁ + (n - 1) * d)) / 2

-- Given Problem Conditions
def a₁ := -5
def d := 3
def n := 20

-- Theorem: Sum of the first 20 terms of the arithmetic sequence is 470
theorem arith_seq_sum_correct : arith_seq_sum a₁ d n = 470 :=
  sorry

end arith_seq_sum_correct_l206_206986


namespace range_of_a_l206_206331

theorem range_of_a (a : ℝ) : (∀ x : ℝ, x^2 - a * x + 64 > 0) → -16 < a ∧ a < 16 :=
by
  -- The proof steps will go here
  sorry

end range_of_a_l206_206331


namespace given_inequality_l206_206851

theorem given_inequality (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (h: 1 + a + b + c = 2 * a * b * c) :
  ab / (1 + a + b) + bc / (1 + b + c) + ca / (1 + c + a) ≥ 3 / 2 :=
sorry

end given_inequality_l206_206851


namespace set_intersection_subset_condition_l206_206263

-- Define the sets A and B
def A (x : ℝ) : Prop := 1 < x - 1 ∧ x - 1 ≤ 4
def B (a : ℝ) (x : ℝ) : Prop := x < a

-- First proof problem: A ∩ B = {x | 2 < x < 3}
theorem set_intersection (a : ℝ) (x : ℝ) (h_a : a = 3) :
  A x ∧ B a x ↔ 2 < x ∧ x < 3 :=
by
  sorry

-- Second proof problem: a > 5 given A ⊆ B
theorem subset_condition (a : ℝ) :
  (∀ x, A x → B a x) ↔ a > 5 :=
by
  sorry

end set_intersection_subset_condition_l206_206263


namespace min_value_expr_l206_206450

theorem min_value_expr (n : ℕ) (hn : 0 < n) : 
  ∃ m : ℕ, m > 0 ∧ (forall (n : ℕ), 0 < n → (n/2 + 50/n : ℝ) ≥ 10) ∧ 
           (n = 10) → (n/2 + 50/n : ℝ) = 10 :=
by
  sorry

end min_value_expr_l206_206450


namespace set_M_roster_method_l206_206793

open Set

theorem set_M_roster_method :
  {a : ℤ | ∃ (n : ℕ), 6 = n * (5 - a)} = {-1, 2, 3, 4} := by
  sorry

end set_M_roster_method_l206_206793


namespace marbles_lost_l206_206369

theorem marbles_lost (initial_marbles lost_marbles gifted_marbles remaining_marbles : ℕ) 
  (h_initial : initial_marbles = 85)
  (h_gifted : gifted_marbles = 25)
  (h_remaining : remaining_marbles = 43)
  (h_before_gifting : remaining_marbles + gifted_marbles = initial_marbles - lost_marbles) :
  lost_marbles = 17 :=
by
  sorry

end marbles_lost_l206_206369


namespace gcd_lcm_sum_l206_206006

theorem gcd_lcm_sum : Nat.gcd 24 54 + Nat.lcm 40 10 = 46 := by
  sorry

end gcd_lcm_sum_l206_206006


namespace calculate_taxes_l206_206484

def gross_pay : ℝ := 4500
def tax_rate_1 : ℝ := 0.10
def tax_rate_2 : ℝ := 0.15
def tax_rate_3 : ℝ := 0.20
def income_bracket_1 : ℝ := 1500
def income_bracket_2 : ℝ := 2000
def income_bracket_remaining : ℝ := gross_pay - income_bracket_1 - income_bracket_2
def standard_deduction : ℝ := 100

theorem calculate_taxes :
  let tax_1 := tax_rate_1 * income_bracket_1
  let tax_2 := tax_rate_2 * income_bracket_2
  let tax_3 := tax_rate_3 * income_bracket_remaining
  let total_tax := tax_1 + tax_2 + tax_3
  let tax_after_deduction := total_tax - standard_deduction
  tax_after_deduction = 550 :=
by
  sorry

end calculate_taxes_l206_206484


namespace probability_of_red_black_or_white_l206_206504

def numberOfBalls := 12
def redBalls := 5
def blackBalls := 4
def whiteBalls := 2
def greenBalls := 1

def favorableOutcomes : Nat := redBalls + blackBalls + whiteBalls
def totalOutcomes : Nat := numberOfBalls

theorem probability_of_red_black_or_white :
  (favorableOutcomes : ℚ) / (totalOutcomes : ℚ) = 11 / 12 :=
by
  sorry

end probability_of_red_black_or_white_l206_206504


namespace students_attended_school_l206_206295

-- Definitions based on conditions
def total_students (S : ℕ) : Prop :=
  ∃ (L R : ℕ), 
    (L = S / 2) ∧ 
    (R = L / 4) ∧ 
    (5 = R / 5)

-- Theorem stating the problem
theorem students_attended_school (S : ℕ) : total_students S → S = 200 :=
by
  intro h
  sorry

end students_attended_school_l206_206295


namespace max_fm_n_l206_206182

noncomputable def ln (x : ℝ) : ℝ := Real.log x

def g (m n x : ℝ) : ℝ := (2 * m + 3) * x + n

def condition_f_g (m n : ℝ) : Prop := ∀ x > 0, ln x ≤ g m n x

def f (m : ℝ) : ℝ := 2 * m + 3

theorem max_fm_n (m n : ℝ) (h : condition_f_g m n) : (f m) * n ≤ 1 / Real.exp 2 := sorry

end max_fm_n_l206_206182


namespace log_one_eq_zero_l206_206807

theorem log_one_eq_zero : Real.log 1 = 0 := 
by
  sorry

end log_one_eq_zero_l206_206807


namespace solve_and_sum_solutions_l206_206157

theorem solve_and_sum_solutions :
  (∀ x : ℝ, x^2 > 9 → 
  (∃ S : ℝ, (∀ x : ℝ, (x / 3 + x / Real.sqrt (x^2 - 9) = 35 / 12) → S = 8.75))) :=
sorry

end solve_and_sum_solutions_l206_206157


namespace solve_fractional_equation_l206_206150

theorem solve_fractional_equation : ∀ (x : ℝ), x ≠ 0 ∧ x ≠ -3 → (1 / x = 2 / (x + 3) ↔ x = 3) :=
by
  intros x hx
  sorry

end solve_fractional_equation_l206_206150


namespace range_of_a_l206_206877

theorem range_of_a (a : ℝ) : 
  (∀ x : ℝ, (a - 2) * x^2 + 2 * (a - 2) * x + 4 ≥ 0) ↔ (2 ≤ a ∧ a ≤ 6) := 
sorry

end range_of_a_l206_206877


namespace number_of_sheep_l206_206211

theorem number_of_sheep (S H : ℕ) 
  (h1 : S / H = 5 / 7)
  (h2 : H * 230 = 12880) : 
  S = 40 :=
by
  sorry

end number_of_sheep_l206_206211


namespace min_quality_inspection_machines_l206_206566

theorem min_quality_inspection_machines (z x : ℕ) :
  (z + 30 * x) / 30 = 1 →
  (z + 10 * x) / 10 = 2 →
  (z + 5 * x) / 5 ≥ 4 :=
by
  intros h1 h2
  sorry

end min_quality_inspection_machines_l206_206566


namespace maxine_purchases_l206_206795

theorem maxine_purchases (x y z : ℕ) (h1 : x + y + z = 40) (h2 : 50 * x + 400 * y + 500 * z = 10000) : x = 40 :=
by
  sorry

end maxine_purchases_l206_206795


namespace find_maximum_k_l206_206777

theorem find_maximum_k {k : ℝ} 
  (h_eq : ∀ x, x^2 + k * x + 8 = 0)
  (h_roots_diff : ∀ x₁ x₂, x₁ - x₂ = 10) :
  k = 2 * Real.sqrt 33 := 
sorry

end find_maximum_k_l206_206777


namespace cos_pi_minus_2alpha_l206_206438

theorem cos_pi_minus_2alpha {α : ℝ} (h : Real.sin α = 2 / 3) : Real.cos (π - 2 * α) = -1 / 9 :=
by
  sorry

end cos_pi_minus_2alpha_l206_206438


namespace sum_of_consecutive_pairs_eq_pow_two_l206_206139

theorem sum_of_consecutive_pairs_eq_pow_two (n m : ℕ) :
  ∃ n m : ℕ, (n * (n + 1) + m * (m + 1) = 2 ^ 2021) :=
sorry

end sum_of_consecutive_pairs_eq_pow_two_l206_206139


namespace length_of_AB_l206_206939

noncomputable def hyperbola_conditions (a b : ℝ) (hac : a > 0) (hbc : b = 2 * a) :=
  ∃ x y : ℝ, (x^2 / a^2) - (y^2 / b^2) = 1

def circle_intersection_condition (A B : ℝ × ℝ) :=
  ∃ (x1 y1 x2 y2 : ℝ), 
  (A = (x1, y1)) ∧ (B = (x2, y2)) ∧ ((x1 - 2)^2 + (y1 - 3)^2 = 1 ∧ y1 = 2 * x1) ∧
  ((x2 - 2)^2 + (y2 - 3)^2 = 1 ∧ y2 = 2 * x2)

theorem length_of_AB {a b : ℝ} (hac : a > 0) (hb : b = 2 * a) :
  (hyperbola_conditions a b hac hb) →
  ∃ (A B : ℝ × ℝ), circle_intersection_condition A B → 
  dist A B = (4 * Real.sqrt 5) / 5 :=
by
  sorry

end length_of_AB_l206_206939


namespace max_value_ab_l206_206451

theorem max_value_ab (a b : ℝ) (ha : 0 < a) (hb : 0 < b) (h : 5 * a + 8 * b = 80) : ab ≤ 40 := 
  sorry

end max_value_ab_l206_206451


namespace ratio_of_x_to_y_l206_206367

theorem ratio_of_x_to_y (x y : ℝ) (h : (8 * x - 5 * y) / (11 * x - 3 * y) = 4 / 7) : x / y = 23 / 12 := 
by
  sorry

end ratio_of_x_to_y_l206_206367


namespace locus_of_centers_l206_206800

set_option pp.notation false -- To ensure nicer looking lean code.

-- Define conditions for circles C_3 and C_4
def C3 (x y : ℝ) : Prop := x^2 + y^2 = 1
def C4 (x y : ℝ) : Prop := (x - 3)^2 + y^2 = 81

-- Statement to prove the locus of centers satisfies the equation
theorem locus_of_centers (a b : ℝ) :
  (∃ r : ℝ, (a^2 + b^2 = (r + 1)^2) ∧ ((a - 3)^2 + b^2 = (9 - r)^2)) →
  (a^2 + 18 * b^2 - 6 * a - 440 = 0) :=
by
  sorry -- Proof not required as per the instructions

end locus_of_centers_l206_206800


namespace arithmetic_sequence_sum_ratio_l206_206518

theorem arithmetic_sequence_sum_ratio (a_n : ℕ → ℕ) (S : ℕ → ℕ) 
  (hS : ∀ n, S n = n * a_n 1 + n * (n - 1) / 2 * (a_n 2 - a_n 1)) 
  (h1 : S 6 / S 3 = 4) : S 9 / S 6 = 9 / 4 := 
by 
  sorry

end arithmetic_sequence_sum_ratio_l206_206518


namespace solve_for_x_y_l206_206552

noncomputable def x_y_2018_sum (x y : ℝ) : ℝ := x^2018 + y^2018

theorem solve_for_x_y (A B : Set ℝ) (x y : ℝ)
  (hA : A = {x, x * y, x + y})
  (hB : B = {0, |x|, y}) 
  (h : A = B) :
  x_y_2018_sum x y = 2 := 
by
  sorry

end solve_for_x_y_l206_206552


namespace problem1_problem2_l206_206858

noncomputable def f (a x : ℝ) := a - (2 / x)

theorem problem1 (a : ℝ) :
  (∀ x1 x2 : ℝ, 0 < x1 → x1 < x2 → (f a x1 < f a x2)) :=
sorry

theorem problem2 (a : ℝ) :
  (∀ x : ℝ, 1 < x → (f a x < 2 * x)) → a ≤ 3 :=
sorry

end problem1_problem2_l206_206858


namespace older_friend_is_38_l206_206967

-- Define the conditions
def younger_friend_age (x : ℕ) : Prop := 
  ∃ (y : ℕ), (y = x + 2 ∧ x + y = 74)

-- Define the age of the older friend
def older_friend_age (x : ℕ) : ℕ := x + 2

-- State the theorem
theorem older_friend_is_38 : ∃ x, younger_friend_age x ∧ older_friend_age x = 38 :=
by
  sorry

end older_friend_is_38_l206_206967


namespace find_M_M_superset_N_M_intersection_N_l206_206433

-- Define the set M as per the given condition
def M : Set ℝ := { x : ℝ | x^2 - x - 2 < 0 }

-- Define the set N based on parameters a and b
def N (a b : ℝ) : Set ℝ := { x : ℝ | a < x ∧ x < b }

-- Prove that M = (-1, 2)
theorem find_M : M = { x : ℝ | -1 < x ∧ x < 2 } :=
sorry

-- Prove that if M ⊇ N, then a ≥ -1
theorem M_superset_N (a b : ℝ) (h : M ⊇ N a b) : -1 ≤ a :=
sorry

-- Prove that if M ∩ N = M, then b ≥ 2
theorem M_intersection_N (a b : ℝ) (h : M ∩ (N a b) = M) : 2 ≤ b :=
sorry

end find_M_M_superset_N_M_intersection_N_l206_206433


namespace calculation_proof_l206_206288

theorem calculation_proof : 441 + 2 * 21 * 7 + 49 = 784 := by
  sorry

end calculation_proof_l206_206288


namespace arccos_neg_one_eq_pi_l206_206989

theorem arccos_neg_one_eq_pi : Real.arccos (-1) = Real.pi :=
by
  sorry

end arccos_neg_one_eq_pi_l206_206989


namespace doughnuts_remaining_l206_206040

theorem doughnuts_remaining 
  (total_doughnuts : ℕ)
  (total_staff : ℕ)
  (staff_3_doughnuts : ℕ)
  (doughnuts_eaten_by_3 : ℕ)
  (staff_2_doughnuts : ℕ)
  (doughnuts_eaten_by_2 : ℕ)
  (staff_4_doughnuts : ℕ)
  (doughnuts_eaten_by_4 : ℕ) :
  total_doughnuts = 120 →
  total_staff = 35 →
  staff_3_doughnuts = 15 →
  staff_2_doughnuts = 10 →
  doughnuts_eaten_by_3 = staff_3_doughnuts * 3 →
  doughnuts_eaten_by_2 = staff_2_doughnuts * 2 →
  staff_4_doughnuts = total_staff - (staff_3_doughnuts + staff_2_doughnuts) →
  doughnuts_eaten_by_4 = staff_4_doughnuts * 4 →
  total_doughnuts - (doughnuts_eaten_by_3 + doughnuts_eaten_by_2 + doughnuts_eaten_by_4) = 15 :=
by
  intros
  -- Proof goes here
  sorry

end doughnuts_remaining_l206_206040


namespace scientific_notation_l206_206416

theorem scientific_notation (x : ℝ) (h : x = 70819) : x = 7.0819 * 10^4 :=
by 
  -- Proof goes here
  sorry

end scientific_notation_l206_206416


namespace deductive_reasoning_is_option_A_l206_206205

-- Define the types of reasoning.
inductive ReasoningType
| Deductive
| Analogical
| Inductive

-- Define the options provided in the problem.
def OptionA : ReasoningType := ReasoningType.Deductive
def OptionB : ReasoningType := ReasoningType.Analogical
def OptionC : ReasoningType := ReasoningType.Inductive
def OptionD : ReasoningType := ReasoningType.Inductive

-- Statement to prove that Option A is Deductive reasoning.
theorem deductive_reasoning_is_option_A : OptionA = ReasoningType.Deductive := by
  -- proof
  sorry

end deductive_reasoning_is_option_A_l206_206205


namespace problem_f3_is_neg2_l206_206903

theorem problem_f3_is_neg2 (f : ℝ → ℝ) (h1 : ∀ x, f (-x) = -f x) (h2 : ∀ x, f (1 + x) = -f (1 - x)) (h3 : f 1 = 2) : f 3 = -2 :=
sorry

end problem_f3_is_neg2_l206_206903


namespace conjecture_a_n_l206_206223

noncomputable def a_n (n : ℕ) : ℚ := (2^n - 1) / 2^(n-1)

noncomputable def S_n (n : ℕ) : ℚ := 2 * n - a_n n

theorem conjecture_a_n (n : ℕ) (h : n > 0) : a_n n = (2^n - 1) / 2^(n-1) :=
by 
  sorry

end conjecture_a_n_l206_206223


namespace abs_AB_l206_206058

noncomputable def ellipse_foci (A B : ℝ) : Prop :=
  B^2 - A^2 = 25

noncomputable def hyperbola_foci (A B : ℝ) : Prop :=
  A^2 + B^2 = 64

theorem abs_AB (A B : ℝ) (h1 : ellipse_foci A B) (h2 : hyperbola_foci A B) :
  |A * B| = Real.sqrt 867.75 := 
sorry

end abs_AB_l206_206058


namespace number_of_lattice_points_in_triangle_l206_206830

theorem number_of_lattice_points_in_triangle (L : ℕ) (hL : L > 1) :
  ∃ I, I = (L^2 - 1) / 2 :=
by
  sorry

end number_of_lattice_points_in_triangle_l206_206830


namespace original_price_correct_l206_206961

noncomputable def original_price (selling_price : ℝ) (gain_percent : ℝ) : ℝ :=
  selling_price / (1 + gain_percent / 100)

theorem original_price_correct :
  original_price 35 75 = 20 :=
by
  sorry

end original_price_correct_l206_206961


namespace choose_3_of_9_colors_l206_206547

-- Define the combination function
noncomputable def combination (n k : ℕ) := n.choose k

-- Noncomputable because factorial and combination require division.
noncomputable def combination_9_3 := combination 9 3

-- State the theorem we are proving
theorem choose_3_of_9_colors : combination_9_3 = 84 :=
by
  -- Proof skipped
  sorry

end choose_3_of_9_colors_l206_206547


namespace inequality_ac2_bc2_implies_a_b_l206_206523

theorem inequality_ac2_bc2_implies_a_b (a b c : ℝ) (h : a * c^2 > b * c^2) : a > b :=
sorry

end inequality_ac2_bc2_implies_a_b_l206_206523


namespace pureGalaTrees_l206_206463

theorem pureGalaTrees {T F C : ℕ} (h1 : F + C = 204) (h2 : F = (3 / 4 : ℝ) * T) (h3 : C = (1 / 10 : ℝ) * T) : (0.15 * T : ℝ) = 36 :=
by
  sorry

end pureGalaTrees_l206_206463


namespace mary_initial_flour_l206_206509

theorem mary_initial_flour (F_total F_add F_initial : ℕ) 
  (h_total : F_total = 9)
  (h_add : F_add = 6)
  (h_initial : F_initial = F_total - F_add) :
  F_initial = 3 :=
sorry

end mary_initial_flour_l206_206509


namespace calculate_train_speed_l206_206310

def speed_train_excluding_stoppages (distance_per_hour_including_stoppages : ℕ) (stoppage_minutes_per_hour : ℕ) : ℕ :=
  let effective_running_time_per_hour := 60 - stoppage_minutes_per_hour
  let effective_running_time_in_hours := effective_running_time_per_hour / 60
  distance_per_hour_including_stoppages / effective_running_time_in_hours

theorem calculate_train_speed :
  speed_train_excluding_stoppages 42 4 = 45 :=
by
  sorry

end calculate_train_speed_l206_206310


namespace a_gt_b_iff_a_ln_a_gt_b_ln_b_l206_206351

theorem a_gt_b_iff_a_ln_a_gt_b_ln_b {a b : ℝ} (ha : a > 0) (hb : b > 0) : 
  (a > b) ↔ (a + Real.log a > b + Real.log b) :=
by sorry

end a_gt_b_iff_a_ln_a_gt_b_ln_b_l206_206351


namespace hare_wins_l206_206960

def hare_wins_race : Prop :=
  let hare_speed := 10
  let hare_run_time := 30
  let hare_nap_time := 30
  let tortoise_speed := 4
  let tortoise_delay := 10
  let total_race_time := 60
  let hare_distance := hare_speed * hare_run_time
  let tortoise_total_time := total_race_time - tortoise_delay
  let tortoise_distance := tortoise_speed * tortoise_total_time
  hare_distance > tortoise_distance

theorem hare_wins : hare_wins_race := by
  -- Proof here
  sorry

end hare_wins_l206_206960


namespace distance_AB_l206_206455

def A : ℝ := -1
def B : ℝ := 2023

theorem distance_AB : |B - A| = 2024 := by
  sorry

end distance_AB_l206_206455


namespace polynomial_remainder_l206_206020

theorem polynomial_remainder :
  (4 * (2.5 : ℝ)^5 - 9 * (2.5 : ℝ)^4 + 7 * (2.5 : ℝ)^2 - 2.5 - 35 = 45.3125) :=
by sorry

end polynomial_remainder_l206_206020


namespace VincentLearnedAtCamp_l206_206064

def VincentSongsBeforeSummerCamp : ℕ := 56
def VincentSongsAfterSummerCamp : ℕ := 74

theorem VincentLearnedAtCamp :
  VincentSongsAfterSummerCamp - VincentSongsBeforeSummerCamp = 18 := by
  sorry

end VincentLearnedAtCamp_l206_206064


namespace max_parabola_ratio_l206_206535

noncomputable def parabola_max_ratio (x y : ℝ) : ℝ :=
  let O : ℝ × ℝ := (0, 0)
  let F : ℝ × ℝ := (1, 0)
  let M : ℝ × ℝ := (x, y)
  
  let MO : ℝ := Real.sqrt (x^2 + y^2)
  let MF : ℝ := Real.sqrt ((x - 1)^2 + y^2)
  
  MO / MF

theorem max_parabola_ratio :
  ∃ x y : ℝ, y^2 = 4 * x ∧ parabola_max_ratio x y = 2 * Real.sqrt 3 / 3 :=
sorry

end max_parabola_ratio_l206_206535


namespace find_n_l206_206788

theorem find_n : ∃ n : ℕ, 
  (n % 4 = 1) ∧ (n % 5 = 2) ∧ (n % 6 = 3) ∧ (n = 57) := by
  sorry

end find_n_l206_206788


namespace apples_preference_count_l206_206066

theorem apples_preference_count (total_people : ℕ) (total_angle : ℝ) (apple_angle : ℝ) 
  (h_total_people : total_people = 530) 
  (h_total_angle : total_angle = 360) 
  (h_apple_angle : apple_angle = 285) : 
  round ((total_people : ℝ) * (apple_angle / total_angle)) = 419 := 
by 
  sorry

end apples_preference_count_l206_206066


namespace matches_start_with_l206_206955

-- Let M be the number of matches Nate started with
variables (M : ℕ)

-- Given conditions
def dropped_creek (dropped : ℕ) := dropped = 10
def eaten_by_dog (eaten : ℕ) := eaten = 2 * 10
def matches_left (final_matches : ℕ) := final_matches = 40

-- Prove that the number of matches Nate started with is 70
theorem matches_start_with 
  (h1 : dropped_creek 10)
  (h2 : eaten_by_dog 20)
  (h3 : matches_left 40) 
  : M = 70 :=
sorry

end matches_start_with_l206_206955


namespace sum_of_squares_eight_l206_206431

theorem sum_of_squares_eight (x y : ℝ) (h1 : x + y = 8) (h2 : x * y = 28) : x^2 + y^2 = 8 := 
  sorry

end sum_of_squares_eight_l206_206431


namespace simplify_exponent_fraction_l206_206642

theorem simplify_exponent_fraction : (3 ^ 2015 + 3 ^ 2013) / (3 ^ 2015 - 3 ^ 2013) = 5 / 4 := by
  sorry

end simplify_exponent_fraction_l206_206642


namespace find_tan_half_angle_l206_206741

variable {α : Real} (h₁ : Real.sin α = -24 / 25) (h₂ : α ∈ Set.Ioo (π:ℝ) (3 * π / 2))

theorem find_tan_half_angle : Real.tan (α / 2) = -4 / 3 :=
sorry

end find_tan_half_angle_l206_206741


namespace polygon_has_five_sides_l206_206767

theorem polygon_has_five_sides (angle : ℝ) (h : angle = 108) :
  (∃ n : ℕ, n > 2 ∧ (180 - angle) * n = 360) ↔ n = 5 := 
by
  sorry

end polygon_has_five_sides_l206_206767


namespace ratio_of_radii_l206_206291

namespace CylinderAndSphere

variable (r R : ℝ)
variable (h_cylinder : 2 * Real.pi * r * (4 * r) = 4 * Real.pi * R ^ 2)

theorem ratio_of_radii (r R : ℝ) (h_cylinder : 2 * Real.pi * r * (4 * r) = 4 * Real.pi * R ^ 2) :
    R / r = Real.sqrt 2 :=
by
  sorry

end CylinderAndSphere

end ratio_of_radii_l206_206291


namespace pencil_distribution_l206_206692

theorem pencil_distribution (x : ℕ) 
  (Alice Bob Charles : ℕ)
  (h1 : Alice = 2 * Bob)
  (h2 : Charles = Bob + 3)
  (h3 : Bob = x)
  (total_pencils : 53 = Alice + Bob + Charles) : 
  Bob = 13 ∧ Alice = 26 ∧ Charles = 16 :=
by
  sorry

end pencil_distribution_l206_206692


namespace carpool_commute_distance_l206_206975

theorem carpool_commute_distance :
  (∀ (D : ℕ),
    4 * 5 * ((2 * D : ℝ) / 30) * 2.50 = 5 * 14 →
    D = 21) :=
by
  intro D
  intro h
  sorry

end carpool_commute_distance_l206_206975


namespace strawberries_left_l206_206867

theorem strawberries_left (picked: ℕ) (eaten: ℕ) (initial_count: picked = 35) (eaten_count: eaten = 2) :
  picked - eaten = 33 :=
by
  sorry

end strawberries_left_l206_206867


namespace jellybean_total_l206_206033

theorem jellybean_total 
    (blackBeans : ℕ)
    (greenBeans : ℕ)
    (orangeBeans : ℕ)
    (h1 : blackBeans = 8)
    (h2 : greenBeans = blackBeans + 2)
    (h3 : orangeBeans = greenBeans - 1) :
    blackBeans + greenBeans + orangeBeans = 27 :=
by
    -- The proof will be placed here
    sorry

end jellybean_total_l206_206033


namespace original_decimal_l206_206572

theorem original_decimal (x : ℝ) (h : 1000 * x / 100 = 12.5) : x = 1.25 :=
by
  sorry

end original_decimal_l206_206572


namespace simplify_expression_l206_206596

variable (a b : ℤ)

theorem simplify_expression : (a - b) - (3 * (a + b)) - b = a - 8 * b := 
by sorry

end simplify_expression_l206_206596


namespace find_numbers_l206_206564

theorem find_numbers (a b c d : ℕ)
  (h1 : a + b + c = 21)
  (h2 : a + b + d = 28)
  (h3 : a + c + d = 29)
  (h4 : b + c + d = 30) : 
  a = 6 ∧ b = 7 ∧ c = 8 ∧ d = 15 :=
sorry

end find_numbers_l206_206564


namespace sufficient_but_not_necessary_condition_l206_206593

theorem sufficient_but_not_necessary_condition (x y : ℝ) :
  (x^2 + y^2 ≤ 1) → ((x - 1)^2 + y^2 ≤ 4) ∧ ¬ ((x - 1)^2 + y^2 ≤ 4 → x^2 + y^2 ≤ 1) :=
by sorry

end sufficient_but_not_necessary_condition_l206_206593


namespace solution_set_of_inequality_l206_206652

theorem solution_set_of_inequality 
  (f : ℝ → ℝ)
  (f_odd : ∀ x : ℝ, f (-x) = -f x)
  (f_at_2 : f 2 = 0)
  (condition : ∀ x : ℝ, 0 < x → x * (deriv f x) + f x > 0) :
  {x : ℝ | x * f x > 0} = {x : ℝ | x < -2} ∪ {x : ℝ | 2 < x} :=
sorry

end solution_set_of_inequality_l206_206652


namespace train_crosses_platform_in_20s_l206_206791

noncomputable def timeToCrossPlatform (train_length : ℝ) (platform_length : ℝ) (train_speed_kmph : ℝ) : ℝ :=
  let train_speed_mps := train_speed_kmph * (1000 / 3600)
  let total_distance := train_length + platform_length
  total_distance / train_speed_mps

theorem train_crosses_platform_in_20s :
  timeToCrossPlatform 120 213.36 60 = 20 :=
by
  sorry

end train_crosses_platform_in_20s_l206_206791


namespace min_sum_ab_l206_206551

theorem min_sum_ab {a b : ℤ} (h : a * b = 36) : a + b ≥ -37 := sorry

end min_sum_ab_l206_206551


namespace ratio_of_costs_l206_206554

-- Definitions based on conditions
def old_car_cost : ℕ := 1800
def new_car_cost : ℕ := 1800 + 2000

-- Theorem stating the desired proof
theorem ratio_of_costs :
  (new_car_cost / old_car_cost : ℚ) = 19 / 9 :=
by
  sorry

end ratio_of_costs_l206_206554


namespace new_trailers_added_l206_206013

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

end new_trailers_added_l206_206013


namespace Monroe_spiders_l206_206734

theorem Monroe_spiders (S : ℕ) (h1 : 12 * 6 + S * 8 = 136) : S = 8 :=
by
  sorry

end Monroe_spiders_l206_206734


namespace vector_addition_correct_l206_206833

def vec1 : ℤ × ℤ := (5, -9)
def vec2 : ℤ × ℤ := (-8, 14)
def vec_sum (v1 v2 : ℤ × ℤ) : ℤ × ℤ := (v1.1 + v2.1, v1.2 + v2.2)

theorem vector_addition_correct :
  vec_sum vec1 vec2 = (-3, 5) :=
by
  -- Proof omitted
  sorry

end vector_addition_correct_l206_206833


namespace not_possible_cut_l206_206481

theorem not_possible_cut (n : ℕ) : 
  let chessboard_area := 8 * 8
  let rectangle_area := 3
  let rectangles_needed := chessboard_area / rectangle_area
  rectangles_needed ≠ n :=
by
  sorry

end not_possible_cut_l206_206481


namespace remaining_bread_after_three_days_l206_206818

namespace BreadProblem

def InitialBreadCount : ℕ := 200

def FirstDayConsumption (bread : ℕ) : ℕ := bread / 4
def SecondDayConsumption (remainingBreadAfterFirstDay : ℕ) : ℕ := 2 * remainingBreadAfterFirstDay / 5
def ThirdDayConsumption (remainingBreadAfterSecondDay : ℕ) : ℕ := remainingBreadAfterSecondDay / 2

theorem remaining_bread_after_three_days : 
  let initialBread := InitialBreadCount 
  let breadAfterFirstDay := initialBread - FirstDayConsumption initialBread 
  let breadAfterSecondDay := breadAfterFirstDay - SecondDayConsumption breadAfterFirstDay 
  let breadAfterThirdDay := breadAfterSecondDay - ThirdDayConsumption breadAfterSecondDay 
  breadAfterThirdDay = 45 := 
by
  let initialBread := InitialBreadCount 
  let breadAfterFirstDay := initialBread - FirstDayConsumption initialBread 
  let breadAfterSecondDay := breadAfterFirstDay - SecondDayConsumption breadAfterFirstDay 
  let breadAfterThirdDay := breadAfterSecondDay - ThirdDayConsumption breadAfterSecondDay 
  have : breadAfterThirdDay = 45 := sorry
  exact this

end BreadProblem

end remaining_bread_after_three_days_l206_206818


namespace triangle_rational_segments_l206_206988

theorem triangle_rational_segments (a b c : ℚ) (h : a + b > c ∧ a + c > b ∧ b + c > a):
  ∃ (ab1 cb1 : ℚ), (ab1 + cb1 = b) := sorry

end triangle_rational_segments_l206_206988


namespace extra_kilometers_per_hour_l206_206834

theorem extra_kilometers_per_hour (S a : ℝ) (h : a > 2) : 
  (S / (a - 2)) - (S / a) = (S / (a - 2)) - (S / a) :=
by sorry

end extra_kilometers_per_hour_l206_206834


namespace students_left_during_year_l206_206468

theorem students_left_during_year (initial_students : ℕ) (new_students : ℕ) (final_students : ℕ) (students_left : ℕ) :
  initial_students = 4 →
  new_students = 42 →
  final_students = 43 →
  students_left = initial_students + new_students - final_students →
  students_left = 3 :=
by
  intro h_initial h_new h_final h_students_left
  rw [h_initial, h_new, h_final] at h_students_left
  exact h_students_left

end students_left_during_year_l206_206468


namespace find_student_hourly_rate_l206_206389

-- Definitions based on conditions
def janitor_work_time : ℝ := 8  -- Janitor can clean the school in 8 hours
def student_work_time : ℝ := 20  -- Student can clean the school in 20 hours
def janitor_hourly_rate : ℝ := 21  -- Janitor is paid $21 per hour
def cost_difference : ℝ := 8  -- The cost difference between janitor alone and both together is $8

-- The value we need to prove
def student_hourly_rate := 7

theorem find_student_hourly_rate
  (janitor_work_time : ℝ)
  (student_work_time : ℝ)
  (janitor_hourly_rate : ℝ)
  (cost_difference : ℝ) :
  S = 7 :=
by
  -- Calculations and logic can be filled here to prove the theorem
  sorry

end find_student_hourly_rate_l206_206389


namespace trigonometric_relationship_l206_206519

noncomputable def α : ℝ := Real.cos 4
noncomputable def b : ℝ := Real.cos (4 * Real.pi / 5)
noncomputable def c : ℝ := Real.sin (7 * Real.pi / 6)

theorem trigonometric_relationship : b < α ∧ α < c := 
by
  sorry

end trigonometric_relationship_l206_206519


namespace women_with_fair_hair_percentage_l206_206565

-- Define the conditions
variables {E : ℝ} (hE : E > 0)

def percent_factor : ℝ := 100

def employees_have_fair_hair (E : ℝ) : ℝ := 0.80 * E
def fair_hair_women (E : ℝ) : ℝ := 0.40 * (employees_have_fair_hair E)

-- Define the target proof statement
theorem women_with_fair_hair_percentage
  (h1 : E > 0)
  (h2 : employees_have_fair_hair E = 0.80 * E)
  (h3 : fair_hair_women E = 0.40 * (employees_have_fair_hair E)):
  (fair_hair_women E / E) * percent_factor = 32 := 
sorry

end women_with_fair_hair_percentage_l206_206565


namespace last_three_digits_of_8_pow_1000_l206_206601

theorem last_three_digits_of_8_pow_1000 (h : 8 ^ 125 ≡ 2 [MOD 1250]) : (8 ^ 1000) % 1000 = 256 :=
by
  sorry

end last_three_digits_of_8_pow_1000_l206_206601


namespace ellipse_properties_l206_206553

theorem ellipse_properties :
  (∀ x y: ℝ, (x^2)/100 + (y^2)/36 = 1) →
  ∃ a b c e : ℝ, 
  a = 10 ∧ 
  b = 6 ∧ 
  c = 8 ∧ 
  2 * a = 20 ∧ 
  e = 4 / 5 :=
by
  intros
  sorry

end ellipse_properties_l206_206553


namespace trigonometric_identity_l206_206266

open Real

variable (α : ℝ)
variable (h1 : π < α)
variable (h2 : α < 2 * π)
variable (h3 : cos (α - 7 * π) = -3 / 5)

theorem trigonometric_identity :
  sin (3 * π + α) * tan (α - 7 * π / 2) = 3 / 5 :=
by
  sorry

end trigonometric_identity_l206_206266


namespace arithmetic_progression_conditions_l206_206555

theorem arithmetic_progression_conditions (a d : ℝ) :
  let x := a
  let y := a + d
  let z := a + 2 * d
  (y^2 = (x^2 * z^2)^(1/2)) ↔ (d = 0 ∨ d = a * (-2 + Real.sqrt 2) ∨ d = a * (-2 - Real.sqrt 2)) :=
by
  intros
  sorry

end arithmetic_progression_conditions_l206_206555


namespace lemonade_third_intermission_l206_206928

theorem lemonade_third_intermission (a b c T : ℝ) (h1 : a = 0.25) (h2 : b = 0.42) (h3 : T = 0.92) (h4 : T = a + b + c) : c = 0.25 :=
by
  sorry

end lemonade_third_intermission_l206_206928


namespace hulk_strength_l206_206602

theorem hulk_strength:
    ∃ n: ℕ, (2^(n-1) > 1000) ∧ (∀ m: ℕ, (2^(m-1) > 1000 → n ≤ m)) := sorry

end hulk_strength_l206_206602


namespace friendly_point_pairs_l206_206820

def friendly_points (k : ℝ) (a : ℝ) (A B : ℝ × ℝ) : Prop :=
  A = (a, -1 / a) ∧ B = (-a, 1 / a) ∧
  B.2 = k * B.1 + 1 + k

theorem friendly_point_pairs : ∀ (k : ℝ), k ≥ 0 → 
  ∃ n, (n = 1 ∨ n = 2) ∧
  (∀ a : ℝ, a > 0 →
    friendly_points k a (a, -1 / a) (-a, 1 / a))
:= by
  sorry

end friendly_point_pairs_l206_206820


namespace sequence_square_terms_l206_206784

theorem sequence_square_terms (k : ℤ) (y : ℕ → ℤ) 
  (h1 : y 1 = 1)
  (h2 : y 2 = 1)
  (h3 : ∀ n ≥ 1, y (n + 2) = (4 * k - 5) * y (n + 1) - y n + 4 - 2 * k) :
  (∀ n, ∃ m : ℤ, y n = m ^ 2) ↔ k = 3 :=
by sorry

end sequence_square_terms_l206_206784


namespace simple_interest_rate_l206_206190

theorem simple_interest_rate (P : ℝ) (T : ℝ) (hT : T = 15)
  (doubles_in_15_years : ∃ R : ℝ, (P * 2 = P + (P * R * T) / 100)) :
  ∃ R : ℝ, R = 6.67 := 
by
  sorry

end simple_interest_rate_l206_206190


namespace shop_profit_correct_l206_206861

def profit_per_tire_repair : ℕ := 20 - 5
def total_tire_repairs : ℕ := 300
def profit_per_complex_repair : ℕ := 300 - 50
def total_complex_repairs : ℕ := 2
def retail_profit : ℕ := 2000
def fixed_expenses : ℕ := 4000

theorem shop_profit_correct :
  profit_per_tire_repair * total_tire_repairs +
  profit_per_complex_repair * total_complex_repairs +
  retail_profit - fixed_expenses = 3000 :=
by
  sorry

end shop_profit_correct_l206_206861


namespace sqrt_product_l206_206390

theorem sqrt_product (h54 : Real.sqrt 54 = 3 * Real.sqrt 6)
                     (h32 : Real.sqrt 32 = 4 * Real.sqrt 2)
                     (h6 : Real.sqrt 6 = Real.sqrt 6) :
    Real.sqrt 54 * Real.sqrt 32 * Real.sqrt 6 = 72 * Real.sqrt 2 := by
  sorry

end sqrt_product_l206_206390


namespace area_union_of_reflected_triangles_l206_206244

def A : ℝ × ℝ := (2, 3)
def B : ℝ × ℝ := (5, 7)
def C : ℝ × ℝ := (6, 2)
def A' : ℝ × ℝ := (3, 2)
def B' : ℝ × ℝ := (7, 5)
def C' : ℝ × ℝ := (2, 6)

noncomputable def triangle_area (A B C : ℝ × ℝ) : ℝ :=
  (1 / 2) * |A.1 * (B.2 - C.2) + B.1 * (C.2 - A.2) + C.1 * (A.2 - B.2)|

theorem area_union_of_reflected_triangles :
  let area_ABC := triangle_area A B C
  let area_A'B'C' := triangle_area A' B' C'
  area_ABC + area_A'B'C' = 19 := by
  sorry

end area_union_of_reflected_triangles_l206_206244


namespace count_four_digit_numbers_l206_206104

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

end count_four_digit_numbers_l206_206104


namespace other_solution_quadratic_l206_206589

theorem other_solution_quadratic (h : (49 : ℚ) * (5 / 7)^2 - 88 * (5 / 7) + 40 = 0) : 
  ∃ x : ℚ, x ≠ 5 / 7 ∧ (49 * x^2 - 88 * x + 40 = 0) ∧ x = 8 / 7 :=
by
  sorry

end other_solution_quadratic_l206_206589


namespace gas_consumption_100_l206_206612

noncomputable def gas_consumption (x : ℝ) : Prop :=
  60 * 1 + (x - 60) * 1.5 = 1.2 * x

theorem gas_consumption_100 (x : ℝ) (h : gas_consumption x) : x = 100 := 
by {
  sorry
}

end gas_consumption_100_l206_206612


namespace total_sticks_used_l206_206375

-- Define the number of sides an octagon has
def octagon_sides : ℕ := 8

-- Define the number of sticks each subsequent octagon needs, sharing one side with the previous one
def additional_sticks_per_octagon : ℕ := 7

-- Define the total number of octagons in the row
def total_octagons : ℕ := 700

-- Define the total number of sticks used
def total_sticks : ℕ := 
  let first_sticks := octagon_sides
  let additional_sticks := additional_sticks_per_octagon * (total_octagons - 1)
  first_sticks + additional_sticks

-- Statement to prove
theorem total_sticks_used : total_sticks = 4901 := by
  sorry

end total_sticks_used_l206_206375


namespace abs_five_minus_two_e_l206_206665

noncomputable def e : ℝ := 2.718

theorem abs_five_minus_two_e : |5 - 2 * e| = 0.436 := by
  sorry

end abs_five_minus_two_e_l206_206665


namespace largest_y_l206_206366

theorem largest_y (y : ℝ) (h : (⌊y⌋ / y) = 8 / 9) : y ≤ 63 / 8 :=
sorry

end largest_y_l206_206366


namespace monotonic_intervals_max_min_values_l206_206798

noncomputable def f : ℝ → ℝ := λ x => (1 / 3) * x^3 + x^2 - 3 * x + 1

theorem monotonic_intervals :
  (∀ x, x < -3 → deriv f x > 0) ∧
  (∀ x, x > 1 → deriv f x > 0) ∧
  (∀ x, -3 < x ∧ x < 1 → deriv f x < 0) :=
by
  sorry

theorem max_min_values :
  f 2 = 5 / 3 ∧ f 1 = -2 / 3 :=
by
  sorry

end monotonic_intervals_max_min_values_l206_206798


namespace average_grade_of_females_is_92_l206_206002

theorem average_grade_of_females_is_92 (F : ℝ) : 
  (∀ (overall_avg male_avg : ℝ) (num_male num_female : ℕ), 
    overall_avg = 90 ∧ male_avg = 82 ∧ num_male = 8 ∧ num_female = 32 → 
    overall_avg = (num_male * male_avg + num_female * F) / (num_male + num_female) → F = 92) :=
sorry

end average_grade_of_females_is_92_l206_206002


namespace length_of_segment_cutoff_l206_206056

-- Define the parabola equation
def parabola (x y : ℝ) := y^2 = 4 * (x + 1)

-- Define the line passing through the focus and perpendicular to the x-axis
def line_through_focus_perp_x_axis (x y : ℝ) := x = 0

-- The actual length calculation lemma
lemma segment_length : 
  ∀ (x y : ℝ), parabola x y → line_through_focus_perp_x_axis x y → y = 2 ∨ y = -2 :=
by sorry

-- The final theorem which gives the length of the segment
theorem length_of_segment_cutoff (y1 y2 : ℝ) :
  ∀ (x : ℝ), parabola x y1 → parabola x y2 → line_through_focus_perp_x_axis x y1 → line_through_focus_perp_x_axis x y2 → (y1 = 2 ∨ y1 = -2) ∧ (y2 = 2 ∨ y2 = -2) → abs (y2 - y1) = 4 :=
by sorry

end length_of_segment_cutoff_l206_206056


namespace distance_from_center_to_plane_correct_l206_206984

noncomputable def distance_from_center_to_plane (O A B C : ℝ × ℝ × ℝ) (radius : ℝ) (a b c : ℝ) : ℝ :=
  let s := (a + b + c) / 2
  let K := Real.sqrt (s * (s - a) * (s - b) * (s - c))
  let R := (a * b * c) / (4 * K)
  let OD := Real.sqrt (radius^2 - R^2)
  OD

theorem distance_from_center_to_plane_correct (O A B C : ℝ × ℝ × ℝ) :
  (dist O A = 20) →
  (dist O B = 20) →
  (dist O C = 20) →
  (dist A B = 13) →
  (dist B C = 14) →
  (dist C A = 15) →
  let m := 15
  let n := 95
  let k := 8
  m + n + k = 118 := by
  sorry

end distance_from_center_to_plane_correct_l206_206984


namespace remaining_money_l206_206890

-- Define the conditions
def num_pies : ℕ := 200
def price_per_pie : ℕ := 20
def fraction_for_ingredients : ℚ := 3 / 5

-- Define the total sales
def total_sales : ℕ := num_pies * price_per_pie

-- Define the cost for ingredients
def cost_for_ingredients : ℚ := fraction_for_ingredients * total_sales 

-- Prove the remaining money
theorem remaining_money : (total_sales : ℚ) - cost_for_ingredients = 1600 := 
by {
  -- This is where the proof would go
  sorry
}

end remaining_money_l206_206890


namespace range_of_a_l206_206109

-- Define the negation of the original proposition as a function
def negated_prop (a : ℝ) : Prop :=
  ∀ x : ℝ, a * x^2 + 4 * x + a > 0

-- State the theorem to be proven
theorem range_of_a (a : ℝ) (h : ¬∃ x : ℝ, a * x^2 + 4 * x + a ≤ 0) : a > 2 :=
  by
  -- Using the assumption to conclude the negated proposition holds
  let h_neg : negated_prop a := sorry
  
  -- Prove the range of a based on h_neg
  sorry

end range_of_a_l206_206109


namespace shakes_indeterminable_l206_206340

variable {B S C x : ℝ}

theorem shakes_indeterminable (h1 : 3 * B + x * S + C = 130) (h2 : 4 * B + 10 * S + C = 164.5) : 
  ¬ (∃ x, 3 * B + x * S + C = 130 ∧ 4 * B + 10 * S + C = 164.5) :=
by
  sorry

end shakes_indeterminable_l206_206340


namespace no_real_roots_of_quadratic_eq_l206_206246

theorem no_real_roots_of_quadratic_eq (k : ℝ) (h : k < -1) :
  ¬ ∃ x : ℝ, x^2 - 2 * x - k = 0 :=
by
  sorry

end no_real_roots_of_quadratic_eq_l206_206246


namespace janet_stuffies_l206_206419

theorem janet_stuffies (total_stuffies kept_stuffies given_away_stuffies janet_stuffies : ℕ) 
 (h1 : total_stuffies = 60)
 (h2 : kept_stuffies = total_stuffies / 3)
 (h3 : given_away_stuffies = total_stuffies - kept_stuffies)
 (h4 : janet_stuffies = given_away_stuffies / 4) : 
 janet_stuffies = 10 := 
sorry

end janet_stuffies_l206_206419


namespace radius_of_cone_is_8_l206_206207

noncomputable def r_cylinder := 8 -- cm
noncomputable def h_cylinder := 2 -- cm
noncomputable def h_cone := 6 -- cm

theorem radius_of_cone_is_8 :
  exists (r_cone : ℝ), r_cone = 8 ∧ π * r_cylinder^2 * h_cylinder = (1 / 3) * π * r_cone^2 * h_cone :=
by
  let r_cone := 8
  have eq_volumes : π * r_cylinder^2 * h_cylinder = (1 / 3) * π * r_cone^2 * h_cone := 
    sorry
  exact ⟨r_cone, by simp, eq_volumes⟩

end radius_of_cone_is_8_l206_206207


namespace side_length_c_4_l206_206225

theorem side_length_c_4 (A : ℝ) (b S c : ℝ) 
  (hA : A = 120) (hb : b = 2) (hS : S = 2 * Real.sqrt 3) : 
  c = 4 :=
sorry

end side_length_c_4_l206_206225


namespace trader_profit_percentage_l206_206199

-- Define the conditions.
variables (indicated_weight actual_weight_given claimed_weight : ℝ)
variable (profit_percentage : ℝ)

-- Given conditions
def conditions :=
  indicated_weight = 1000 ∧
  actual_weight_given = claimed_weight / 1.5 ∧
  claimed_weight = indicated_weight ∧
  profit_percentage = (claimed_weight - actual_weight_given) / actual_weight_given * 100

-- Prove that the profit percentage is 50%
theorem trader_profit_percentage : conditions indicated_weight actual_weight_given claimed_weight profit_percentage → profit_percentage = 50 :=
by
  sorry

end trader_profit_percentage_l206_206199


namespace Wayne_blocks_count_l206_206412

-- Statement of the proof problem
theorem Wayne_blocks_count (initial_blocks additional_blocks total_blocks : ℕ) 
  (h1 : initial_blocks = 9) 
  (h2 : additional_blocks = 6) 
  (h3 : total_blocks = initial_blocks + additional_blocks) : 
  total_blocks = 15 := 
by 
  -- proof would go here, but we will use sorry for now
  sorry

end Wayne_blocks_count_l206_206412


namespace olivia_did_not_sell_4_bars_l206_206635

-- Define the constants and conditions
def price_per_bar : ℕ := 3
def total_bars : ℕ := 7
def money_made : ℕ := 9

-- Calculate the number of bars sold
def bars_sold : ℕ := money_made / price_per_bar

-- Calculate the number of bars not sold
def bars_not_sold : ℕ := total_bars - bars_sold

-- Theorem to prove the answer
theorem olivia_did_not_sell_4_bars : bars_not_sold = 4 := 
by 
  sorry

end olivia_did_not_sell_4_bars_l206_206635


namespace fraction_sum_59_l206_206212

theorem fraction_sum_59 :
  ∃ (a b : ℕ), (0.84375 = (a : ℚ) / b) ∧ (Nat.gcd a b = 1) ∧ (a + b = 59) :=
sorry

end fraction_sum_59_l206_206212


namespace simplify_expression_l206_206189

theorem simplify_expression (x y : ℝ) :
  3 * x + 6 * x + 9 * x + 12 * y + 15 * y + 18 + 21 = 18 * x + 27 * y + 39 :=
by
  sorry

end simplify_expression_l206_206189


namespace vector_addition_proof_l206_206287

def vector_add (a b : ℤ × ℤ) : ℤ × ℤ :=
  (a.1 + b.1, a.2 + b.2)

theorem vector_addition_proof :
  let a := (2, 0)
  let b := (-1, -2)
  vector_add a b = (1, -2) :=
by
  sorry

end vector_addition_proof_l206_206287


namespace max_cake_boxes_in_carton_l206_206794

-- Define the dimensions of the carton as constants
def carton_length := 25
def carton_width := 42
def carton_height := 60

-- Define the dimensions of the cake box as constants
def box_length := 8
def box_width := 7
def box_height := 5

-- Define the volume of the carton and the volume of the cake box
def volume_carton := carton_length * carton_width * carton_height
def volume_box := box_length * box_width * box_height

-- Define the theorem statement
theorem max_cake_boxes_in_carton : 
  (volume_carton / volume_box) = 225 :=
by
  -- The proof is omitted.
  sorry

end max_cake_boxes_in_carton_l206_206794


namespace vincent_books_l206_206781

theorem vincent_books (x : ℕ) (h1 : 10 + 3 + x = 13 + x)
                      (h2 : 16 * (13 + x) = 224) : x = 1 :=
by sorry

end vincent_books_l206_206781


namespace cost_of_each_box_of_cereal_l206_206254

theorem cost_of_each_box_of_cereal
  (total_groceries_cost : ℝ)
  (gallon_of_milk_cost : ℝ)
  (number_of_cereal_boxes : ℕ)
  (banana_cost_each : ℝ)
  (number_of_bananas : ℕ)
  (apple_cost_each : ℝ)
  (number_of_apples : ℕ)
  (cookie_cost_multiplier : ℝ)
  (number_of_cookie_boxes : ℕ) :
  total_groceries_cost = 25 →
  gallon_of_milk_cost = 3 →
  number_of_cereal_boxes = 2 →
  banana_cost_each = 0.25 →
  number_of_bananas = 4 →
  apple_cost_each = 0.5 →
  number_of_apples = 4 →
  cookie_cost_multiplier = 2 →
  number_of_cookie_boxes = 2 →
  (total_groceries_cost - (gallon_of_milk_cost + (banana_cost_each * number_of_bananas) + 
                           (apple_cost_each * number_of_apples) + 
                           (number_of_cookie_boxes * (cookie_cost_multiplier * gallon_of_milk_cost)))) / 
  number_of_cereal_boxes = 3.5 := 
sorry

end cost_of_each_box_of_cereal_l206_206254


namespace find_B_l206_206742

theorem find_B (A B : ℕ) (h1 : Prime A) (h2 : Prime B) (h3 : A > 0) (h4 : B > 0) 
  (h5 : 1 / A - 1 / B = 192 / (2005^2 - 2004^2)) : B = 211 :=
sorry

end find_B_l206_206742


namespace even_function_analytic_expression_l206_206845

noncomputable def f (x : ℝ) : ℝ := 
if x ≥ 0 then Real.log (x^2 - 2 * x + 2) 
else Real.log (x^2 + 2 * x + 2)

theorem even_function_analytic_expression (f : ℝ → ℝ)
  (h_even : ∀ x : ℝ, f (-x) = f x)
  (h_nonneg : ∀ x : ℝ, 0 ≤ x → f x = Real.log (x^2 - 2 * x + 2)) :
  ∀ x : ℝ, x < 0 → f x = Real.log (x^2 + 2 * x + 2) :=
by
  sorry

end even_function_analytic_expression_l206_206845


namespace num_pure_Gala_trees_l206_206528

-- Define the problem statement conditions
variables (T F G H : ℝ)
variables (c1 : 0.125 * F + 0.075 * F + F = 315)
variables (c2 : F = (2 / 3) * T)
variables (c3 : H = (1 / 6) * T)
variables (c4 : T = F + G + H)

-- Prove the number of pure Gala trees G is 66
theorem num_pure_Gala_trees : G = 66 :=
by
  -- Proof will be filled out here
  sorry

end num_pure_Gala_trees_l206_206528


namespace equilateral_triangles_formed_l206_206732

theorem equilateral_triangles_formed :
  ∀ k : ℤ, -8 ≤ k ∧ k ≤ 8 →
  (∃ triangles : ℕ, triangles = 426) :=
by sorry

end equilateral_triangles_formed_l206_206732


namespace can_choose_P_l206_206386

-- Define the objects in the problem,
-- types, constants, and assumptions as per the problem statement.

theorem can_choose_P (cube : ℝ) (P Q R S T A B C D : ℝ)
  (edge_length : cube = 10)
  (AR_RB_eq_CS_SB : ∀ AR RB CS SB, (AR / RB = 7 / 3) ∧ (CS / SB = 7 / 3))
  : ∃ P, 2 * (Q - R) = (P - Q) + (R - S) := by
  sorry

end can_choose_P_l206_206386


namespace problem_statement_l206_206764

-- Define the roots of the quadratic as r and s
variables (r s : ℝ)

-- Given conditions
def root_condition (r s : ℝ) := (r + s = 2 * Real.sqrt 6) ∧ (r * s = 3)

theorem problem_statement (h : root_condition r s) : r^8 + s^8 = 93474 :=
sorry

end problem_statement_l206_206764


namespace license_plates_count_l206_206716

noncomputable def num_license_plates : Nat :=
  let num_w := 26 * 26      -- number of combinations for w
  let num_w_orders := 2     -- two possible orders for w
  let num_digits := 10 ^ 5  -- number of combinations for 5 digits
  let num_positions := 6    -- number of valid positions for w
  2 * num_positions * num_digits * num_w

theorem license_plates_count : num_license_plates = 809280000 := by
  sorry

end license_plates_count_l206_206716


namespace ratio_add_b_l206_206751

theorem ratio_add_b (a b : ℚ) (h : a / b = 3 / 5) : (a + b) / b = 8 / 5 :=
by
  sorry

end ratio_add_b_l206_206751


namespace z_share_in_profit_l206_206418

noncomputable def investment_share (investment : ℕ) (months : ℕ) : ℕ := investment * months

noncomputable def profit_share (profit : ℕ) (share : ℚ) : ℚ := (profit : ℚ) * share

theorem z_share_in_profit 
  (investment_X : ℕ := 36000)
  (investment_Y : ℕ := 42000)
  (investment_Z : ℕ := 48000)
  (months_X : ℕ := 12)
  (months_Y : ℕ := 12)
  (months_Z : ℕ := 8)
  (total_profit : ℕ := 14300) :
  profit_share total_profit (investment_share investment_Z months_Z / 
            (investment_share investment_X months_X + 
             investment_share investment_Y months_Y + 
             investment_share investment_Z months_Z)) = 2600 := 
by
  sorry

end z_share_in_profit_l206_206418


namespace decimals_between_6_1_and_6_4_are_not_two_l206_206730

-- Definitions from the conditions in a)
def is_between (x : ℝ) (a b : ℝ) : Prop := a < x ∧ x < b

-- The main theorem statement
theorem decimals_between_6_1_and_6_4_are_not_two :
  ∀ x, is_between x 6.1 6.4 → false :=
by
  sorry

end decimals_between_6_1_and_6_4_are_not_two_l206_206730


namespace find_x_l206_206251

theorem find_x :
  ∃ x : ℝ, 3 * x = (26 - x) + 14 ∧ x = 10 :=
by sorry

end find_x_l206_206251


namespace probability_consecutive_computer_scientists_l206_206337

theorem probability_consecutive_computer_scientists :
  let n := 12
  let k := 5
  let total_permutations := Nat.factorial (n - 1)
  let consecutive_permutations := Nat.factorial (7) * Nat.factorial (5)
  let probability := consecutive_permutations / total_permutations
  probability = (1 / 66) :=
by
  sorry

end probability_consecutive_computer_scientists_l206_206337


namespace calc_fraction_power_l206_206161

theorem calc_fraction_power (n m : ℤ) (h_n : n = 2023) (h_m : m = 2022) :
  (- (2 / 3 : ℚ))^n * ((3 / 2 : ℚ))^m = - (2 / 3) := by
  sorry

end calc_fraction_power_l206_206161


namespace find_r_l206_206133

theorem find_r (r : ℝ) (h : 5 * (r - 9) = 6 * (3 - 3 * r) + 6) : r = 3 :=
by
  sorry

end find_r_l206_206133


namespace horses_tiles_equation_l206_206522

-- Conditions from the problem
def total_horses (x y : ℕ) : Prop := x + y = 100
def total_tiles (x y : ℕ) : Prop := 3 * x + (1 / 3 : ℚ) * y = 100

-- The statement to prove
theorem horses_tiles_equation (x y : ℕ) :
  total_horses x y ∧ total_tiles x y ↔ 
  (x + y = 100 ∧ (3 * x + (1 / 3 : ℚ) * y = 100)) :=
by
  sorry

end horses_tiles_equation_l206_206522


namespace train_passes_jogger_l206_206698

noncomputable def speed_of_jogger_kmph := 9
noncomputable def speed_of_train_kmph := 45
noncomputable def jogger_lead_m := 270
noncomputable def train_length_m := 120

noncomputable def speed_of_jogger_mps := speed_of_jogger_kmph * (1000 / 3600) -- converting km/hr to m/s
noncomputable def speed_of_train_mps := speed_of_train_kmph * (1000 / 3600) -- converting km/hr to m/s
noncomputable def relative_speed_mps := speed_of_train_mps - speed_of_jogger_mps
noncomputable def total_distance_m := jogger_lead_m + train_length_m
noncomputable def time_to_pass_jogger := total_distance_m / relative_speed_mps

theorem train_passes_jogger : time_to_pass_jogger = 39 :=
  by
    -- Proof steps would be provided here
    sorry

end train_passes_jogger_l206_206698


namespace negation_of_forall_statement_l206_206529

theorem negation_of_forall_statement :
  ¬ (∀ x : ℝ, x^2 + 2 * x ≥ 0) ↔ ∃ x : ℝ, x^2 + 2 * x < 0 := 
by
  sorry

end negation_of_forall_statement_l206_206529


namespace center_coordinates_l206_206099

noncomputable def center_of_circle (x y : ℝ) : Prop := 
  x^2 + y^2 + 2*x - 4*y = 0

theorem center_coordinates : center_of_circle (-1) 2 :=
by sorry

end center_coordinates_l206_206099


namespace quartic_polynomial_root_l206_206872

noncomputable def Q (x : ℝ) : ℝ := x^4 - 4*x^3 + 6*x^2 - 4*x - 2

theorem quartic_polynomial_root :
  Q (Real.sqrt (Real.sqrt 3) + 1) = 0 :=
by
  sorry

end quartic_polynomial_root_l206_206872


namespace ball_hits_ground_at_5_over_2_l206_206339

noncomputable def ball_height (t : ℝ) : ℝ := -16 * t^2 + 40 * t + 60

theorem ball_hits_ground_at_5_over_2 :
  ∃ t : ℝ, t = 5 / 2 ∧ ball_height t = 0 :=
sorry

end ball_hits_ground_at_5_over_2_l206_206339


namespace range_of_a_l206_206361

noncomputable def line_eq (a : ℝ) (x y : ℝ) : ℝ := 3 * x - 2 * y + a 

def pointA : ℝ × ℝ := (3, 1)
def pointB : ℝ × ℝ := (-4, 6)

theorem range_of_a :
  (line_eq a pointA.1 pointA.2) * (line_eq a pointB.1 pointB.2) < 0 ↔ -7 < a ∧ a < 24 := sorry

end range_of_a_l206_206361


namespace car_tank_capacity_l206_206380

theorem car_tank_capacity
  (speed : ℝ) (usage_rate : ℝ) (time : ℝ) (used_fraction : ℝ) (distance : ℝ := speed * time) (gallons_used : ℝ := distance / usage_rate) 
  (fuel_used : ℝ := 10) (tank_capacity : ℝ := fuel_used / used_fraction)
  (h1 : speed = 60) (h2 : usage_rate = 30) (h3 : time = 5) (h4 : used_fraction = 0.8333333333333334) : 
  tank_capacity = 12 :=
by
  sorry

end car_tank_capacity_l206_206380


namespace Josh_pencils_left_l206_206231

theorem Josh_pencils_left (initial_pencils : ℕ) (given_pencils : ℕ) (remaining_pencils : ℕ) 
  (h_initial : initial_pencils = 142) 
  (h_given : given_pencils = 31) 
  (h_remaining : remaining_pencils = 111) : 
  initial_pencils - given_pencils = remaining_pencils :=
by
  sorry

end Josh_pencils_left_l206_206231


namespace additional_toothpicks_needed_l206_206327

theorem additional_toothpicks_needed 
  (t : ℕ → ℕ)
  (h1 : t 1 = 4)
  (h2 : t 2 = 10)
  (h3 : t 3 = 18)
  (h4 : t 4 = 28)
  (h5 : t 5 = 40)
  (h6 : t 6 = 54) :
  t 6 - t 4 = 26 :=
by
  sorry

end additional_toothpicks_needed_l206_206327


namespace alexis_sew_skirt_time_l206_206803

theorem alexis_sew_skirt_time : 
  ∀ (S : ℝ), 
  (∀ (C : ℝ), C = 7) → 
  (6 * S + 4 * 7 = 40) → 
  S = 2 := 
by
  intros S _ h
  sorry

end alexis_sew_skirt_time_l206_206803


namespace mean_combined_scores_l206_206810

theorem mean_combined_scores (M A : ℝ) (m a : ℕ) 
  (hM : M = 88) 
  (hA : A = 72) 
  (hm : (m:ℝ) / (a:ℝ) = 2 / 3) :
  (88 * m + 72 * a) / (m + a) = 78 :=
by
  sorry

end mean_combined_scores_l206_206810


namespace identify_7_real_coins_l206_206755

theorem identify_7_real_coins (coins : Fin 63 → ℝ) (fakes : Finset (Fin 63)) (h_fakes_count : fakes.card = 7) (real_weight fake_weight : ℝ)
  (h_weights : ∀ i, i ∉ fakes → coins i = real_weight) (h_fake_weights : ∀ i, i ∈ fakes → coins i = fake_weight) (h_lighter : fake_weight < real_weight) :
  ∃ real_coins : Finset (Fin 63), real_coins.card = 7 ∧ (∀ i, i ∈ real_coins → coins i = real_weight) :=
sorry

end identify_7_real_coins_l206_206755


namespace cost_per_sqft_is_3_l206_206802

def deck_length : ℝ := 30
def deck_width : ℝ := 40
def extra_cost_per_sqft : ℝ := 1
def total_cost : ℝ := 4800

theorem cost_per_sqft_is_3
    (area : ℝ := deck_length * deck_width)
    (sealant_cost : ℝ := area * extra_cost_per_sqft)
    (deck_construction_cost : ℝ := total_cost - sealant_cost) :
    deck_construction_cost / area = 3 :=
by
  sorry

end cost_per_sqft_is_3_l206_206802


namespace simple_interest_rate_l206_206735

theorem simple_interest_rate
  (A5 A8 : ℝ) (years_between : ℝ := 3) (I3 : ℝ) (annual_interest : ℝ)
  (P : ℝ) (R : ℝ)
  (h1 : A5 = 9800) -- Amount after 5 years is Rs. 9800
  (h2 : A8 = 12005) -- Amount after 8 years is Rs. 12005
  (h3 : I3 = A8 - A5) -- Interest for 3 years
  (h4 : annual_interest = I3 / years_between) -- Annual interest
  (h5 : P = 9800) -- Principal amount after 5 years
  (h6 : R = (annual_interest * 100) / P) -- Rate of interest formula revised
  : R = 7.5 := 
sorry

end simple_interest_rate_l206_206735


namespace gcd_45045_30030_l206_206661

/-- The greatest common divisor of 45045 and 30030 is 15015. -/
theorem gcd_45045_30030 : Nat.gcd 45045 30030 = 15015 :=
by 
  sorry

end gcd_45045_30030_l206_206661


namespace expected_coin_worth_is_two_l206_206408

-- Define the conditions
def p_heads : ℚ := 4 / 5
def p_tails : ℚ := 1 / 5
def gain_heads : ℚ := 5
def loss_tails : ℚ := -10

-- Expected worth calculation
def expected_worth : ℚ := (p_heads * gain_heads) + (p_tails * loss_tails)

-- Lean 4 statement to prove
theorem expected_coin_worth_is_two : expected_worth = 2 := by
  sorry

end expected_coin_worth_is_two_l206_206408


namespace find_x_of_equation_l206_206074

theorem find_x_of_equation :
  ∃ x : ℕ, 16^5 + 16^5 + 16^5 = 4^x ∧ x = 20 :=
by 
  sorry

end find_x_of_equation_l206_206074


namespace team_a_took_fewer_hours_l206_206658

/-- Two dogsled teams raced across a 300-mile course. 
Team A finished the course in fewer hours than Team E. 
Team A's average speed was 5 mph greater than Team E's, which was 20 mph. 
How many fewer hours did Team A take to finish the course compared to Team E? --/

theorem team_a_took_fewer_hours :
  let distance := 300
  let speed_e := 20
  let speed_a := speed_e + 5
  let time_e := distance / speed_e
  let time_a := distance / speed_a
  time_e - time_a = 3 := by
  sorry

end team_a_took_fewer_hours_l206_206658


namespace ratio_w_y_l206_206155

theorem ratio_w_y 
  (w x y z : ℚ) 
  (h1 : w / x = 4 / 3)
  (h2 : y / z = 3 / 2)
  (h3 : z / x = 1 / 6) : 
  w / y = 16 / 3 :=
sorry

end ratio_w_y_l206_206155


namespace total_volume_of_four_boxes_l206_206379

theorem total_volume_of_four_boxes :
  (∃ (V : ℕ), (∀ (edge_length : ℕ) (num_boxes : ℕ), edge_length = 6 → num_boxes = 4 → V = (edge_length ^ 3) * num_boxes)) :=
by
  let edge_length := 6
  let num_boxes := 4
  let volume := (edge_length ^ 3) * num_boxes
  use volume
  sorry

end total_volume_of_four_boxes_l206_206379


namespace solve_quadratic1_solve_quadratic2_l206_206943

-- Equation 1
theorem solve_quadratic1 (x : ℝ) :
  (x = 4 + 3 * Real.sqrt 2 ∨ x = 4 - 3 * Real.sqrt 2) ↔ x ^ 2 - 8 * x - 2 = 0 := by
  sorry

-- Equation 2
theorem solve_quadratic2 (x : ℝ) :
  (x = 3 / 2 ∨ x = -1) ↔ 2 * x ^ 2 - x - 3 = 0 := by
  sorry

end solve_quadratic1_solve_quadratic2_l206_206943


namespace quadratic_equation_correct_l206_206657

theorem quadratic_equation_correct :
    (∃ a b c : ℝ, a ≠ 0 ∧ (∀ x : ℝ, a * x^2 + b * x + c = 0) ↔ (∃ x : ℝ, x^2 = 5)) ∧
    ¬(∃ a b c : ℝ, a ≠ 0 ∧ (∀ x : ℝ, a * x^2 + b * x + c = 0) ↔ (∃ x y : ℝ, x + 2 * y = 0)) ∧
    ¬(∃ a b c : ℝ, a ≠ 0 ∧ (∀ x : ℝ, a * x^2 + b * x + c = 0) ↔ (∃ x : ℝ, x^2 + 1/x = 0)) ∧
    ¬(∃ a b c : ℝ, a ≠ 0 ∧ (∀ x : ℝ, a * x^2 + b * x + c = 0) ↔ (∃ x : ℝ, x^3 + x^2 = 0)) :=
by
  sorry

end quadratic_equation_correct_l206_206657


namespace hypotenuse_length_l206_206245

noncomputable def hypotenuse_of_30_60_90_triangle (r : ℝ) : ℝ :=
  let a := (r * 3) / Real.sqrt 3
  2 * a

theorem hypotenuse_length (r : ℝ) (h : r = 3) : hypotenuse_of_30_60_90_triangle r = 6 * Real.sqrt 3 :=
  by sorry

end hypotenuse_length_l206_206245


namespace total_messages_l206_206875

theorem total_messages (l1 l2 l3 a1 a2 a3 : ℕ)
  (h1 : l1 = 120)
  (h2 : a1 = l1 - 20)
  (h3 : l2 = l1 / 3)
  (h4 : a2 = 2 * a1)
  (h5 : l3 = l1)
  (h6 : a3 = a1) :
  l1 + l2 + l3 + a1 + a2 + a3 = 680 :=
by
  -- Proof steps would go here. Adding 'sorry' to skip proof.
  sorry

end total_messages_l206_206875


namespace Shannon_ratio_2_to_1_l206_206604

structure IceCreamCarton :=
  (scoops : ℕ)

structure PersonWants :=
  (vanilla : ℕ)
  (chocolate : ℕ)
  (strawberry : ℕ)

noncomputable def total_scoops_served (ethan_wants lucas_wants danny_wants connor_wants olivia_wants : PersonWants) : ℕ :=
  ethan_wants.vanilla + ethan_wants.chocolate +
  lucas_wants.chocolate +
  danny_wants.chocolate +
  connor_wants.chocolate +
  olivia_wants.vanilla + olivia_wants.strawberry

theorem Shannon_ratio_2_to_1 
    (cartons : List IceCreamCarton)
    (ethan_wants lucas_wants danny_wants connor_wants olivia_wants : PersonWants)
    (scoops_left : ℕ) : 
    -- Conditions
    (∀ carton ∈ cartons, carton.scoops = 10) →
    (cartons.length = 3) →
    (ethan_wants.vanilla = 1 ∧ ethan_wants.chocolate = 1) →
    (lucas_wants.chocolate = 2) →
    (danny_wants.chocolate = 2) →
    (connor_wants.chocolate = 2) →
    (olivia_wants.vanilla = 1 ∧ olivia_wants.strawberry = 1) →
    (scoops_left = 16) →
    -- To Prove
    4 / olivia_wants.vanilla + olivia_wants.strawberry = 2 := 
sorry

end Shannon_ratio_2_to_1_l206_206604


namespace circumference_of_smaller_circle_l206_206305

variable (R : ℝ)
variable (A_shaded : ℝ)

theorem circumference_of_smaller_circle :
  (A_shaded = (32 / π) ∧ 3 * (π * R ^ 2) - π * R ^ 2 = A_shaded) → 
  2 * π * R = 4 :=
by
  sorry

end circumference_of_smaller_circle_l206_206305


namespace last_score_is_71_l206_206821

theorem last_score_is_71 (scores : List ℕ) (h : scores = [71, 74, 79, 85, 88, 92]) (sum_eq: scores.sum = 489) :
  ∃ s : ℕ, s ∈ scores ∧ 
           (∃ avg : ℕ, avg = (scores.sum - s) / 5 ∧ 
           ∀ lst : List ℕ, lst = scores.erase s → (∀ n, n ∈ lst → lst.sum % (lst.length - 1) = 0)) :=
  sorry

end last_score_is_71_l206_206821


namespace snail_distance_l206_206625

def speed_A : ℝ := 10
def speed_B : ℝ := 15
def time_difference : ℝ := 0.5

theorem snail_distance : 
  ∃ (D : ℝ) (t_A t_B : ℝ), 
    D = speed_A * t_A ∧ 
    D = speed_B * t_B ∧
    t_A = t_B + time_difference ∧ 
    D = 15 := 
by
  sorry

end snail_distance_l206_206625


namespace joan_took_marbles_l206_206405

-- Each condition is used as a definition.
def original_marbles : ℕ := 86
def remaining_marbles : ℕ := 61

-- The theorem states that the number of marbles Joan took equals 25.
theorem joan_took_marbles : (original_marbles - remaining_marbles) = 25 := by
  sorry    -- Add sorry to skip the proof.

end joan_took_marbles_l206_206405


namespace problem1_problem2_l206_206425

-- Problem 1
theorem problem1 (α : ℝ) (h : Real.tan α = -3/4) :
  (Real.cos ((π / 2) + α) * Real.sin (-π - α)) /
  (Real.cos ((11 * π) / 2 - α) * Real.sin ((9 * π) / 2 + α)) = -3 / 4 :=
by sorry

-- Problem 2
theorem problem2 (α : ℝ) (h1 : 0 ≤ α ∧ α ≤ π) (h2 : Real.sin α + Real.cos α = 1 / 5) :
  Real.cos (2 * α - π / 4) = -31 * Real.sqrt 2 / 50 :=
by sorry

end problem1_problem2_l206_206425


namespace residue_neg_1234_mod_31_l206_206028

theorem residue_neg_1234_mod_31 : -1234 % 31 = 6 := 
by sorry

end residue_neg_1234_mod_31_l206_206028


namespace find_A_l206_206550

theorem find_A (A : ℝ) (h : (12 + 3) * (12 - A) = 120) : A = 4 :=
by sorry

end find_A_l206_206550


namespace sufficient_but_not_necessary_condition_l206_206785

theorem sufficient_but_not_necessary_condition (x : ℝ) :
  (x >= 3) → (x^2 - 2*x - 3 >= 0) ∧ ¬((x^2 - 2*x - 3 >= 0) → (x >= 3)) := by
  sorry

end sufficient_but_not_necessary_condition_l206_206785


namespace find_number_l206_206122

theorem find_number (x : ℝ) (h : x - (3/5 : ℝ) * x = 60) : x = 150 :=
sorry

end find_number_l206_206122


namespace rolls_combinations_l206_206615

theorem rolls_combinations (x1 x2 x3 : ℕ) (h1 : x1 + x2 + x3 = 2) : 
  (Nat.choose (2 + 3 - 1) (3 - 1) = 6) :=
by
  sorry

end rolls_combinations_l206_206615


namespace initial_customers_l206_206778

theorem initial_customers (tables : ℕ) (people_per_table : ℕ) (customers_left : ℕ) (h1 : tables = 5) (h2 : people_per_table = 9) (h3 : customers_left = 17) :
  tables * people_per_table + customers_left = 62 :=
by
  sorry

end initial_customers_l206_206778


namespace subtraction_equals_eleven_l206_206271

theorem subtraction_equals_eleven (K A N G R O : ℕ) (h1: K ≠ A) (h2: K ≠ N) (h3: K ≠ G) (h4: K ≠ R) (h5: K ≠ O) (h6: A ≠ N) (h7: A ≠ G) (h8: A ≠ R) (h9: A ≠ O) (h10: N ≠ G) (h11: N ≠ R) (h12: N ≠ O) (h13: G ≠ R) (h14: G ≠ O) (h15: R ≠ O) (sum_eq : 100 * K + 10 * A + N + 10 * G + A = 100 * R + 10 * O + O) : 
  (10 * R + N) - (10 * K + G) = 11 := 
by 
  sorry

end subtraction_equals_eleven_l206_206271


namespace five_times_remaining_is_400_l206_206482

-- Define the conditions
def original_marbles := 800
def marbles_per_friend := 120
def num_friends := 6

-- Calculate total marbles given away
def marbles_given_away := num_friends * marbles_per_friend

-- Calculate marbles remaining after giving away
def marbles_remaining := original_marbles - marbles_given_away

-- Question: what is five times the marbles remaining?
def five_times_remaining_marbles := 5 * marbles_remaining

-- The proof problem: prove that this equals 400
theorem five_times_remaining_is_400 : five_times_remaining_marbles = 400 :=
by
  -- The proof would go here
  sorry

end five_times_remaining_is_400_l206_206482


namespace distinct_real_roots_range_of_m_l206_206479

theorem distinct_real_roots_range_of_m (m : ℝ) : 
  (∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ (x₁^2 + x₁ - m = 0) ∧ (x₂^2 + x₂ - m = 0)) → m > -1/4 := 
sorry

end distinct_real_roots_range_of_m_l206_206479


namespace number_div_0_04_eq_100_9_l206_206608

theorem number_div_0_04_eq_100_9 :
  ∃ number : ℝ, (number / 0.04 = 100.9) ∧ (number = 4.036) :=
sorry

end number_div_0_04_eq_100_9_l206_206608


namespace aunt_may_morning_milk_l206_206219

-- Defining the known quantities as variables
def evening_milk : ℕ := 380
def sold_milk : ℕ := 612
def leftover_milk : ℕ := 15
def milk_left : ℕ := 148

-- Main statement to be proven
theorem aunt_may_morning_milk (M : ℕ) :
  M + evening_milk + leftover_milk - sold_milk = milk_left → M = 365 := 
by {
  -- Skipping the proof
  sorry
}

end aunt_may_morning_milk_l206_206219


namespace angle_733_in_first_quadrant_l206_206053

def in_first_quadrant (θ : ℝ) : Prop := 
  0 < θ ∧ θ < 90

theorem angle_733_in_first_quadrant :
  in_first_quadrant (733 % 360 : ℝ) :=
sorry

end angle_733_in_first_quadrant_l206_206053


namespace distance_from_Q_to_AD_l206_206323

-- Define the square $ABCD$ with side length 6
def square_ABCD (A B C D : ℝ × ℝ) : Prop :=
  A = (0, 6) ∧ B = (6, 6) ∧ C = (6, 0) ∧ D = (0, 0)

-- Define point $N$ as the midpoint of $\overline{CD}$
def midpoint_CD (C D N : ℝ × ℝ) : Prop :=
  N = ((C.1 + D.1) / 2, (C.2 + D.2) / 2)

-- Define the intersection condition of the circles centered at $N$ and $A$
def intersect_circles (N A Q D : ℝ × ℝ) : Prop :=
  (Q = D ∨ (∃ r₁ r₂, (Q.1 - N.1)^2 + Q.2^2 = r₁ ∧ Q.1^2 + (Q.2 - A.2)^2 = r₂))

-- Prove the distance from $Q$ to $\overline{AD}$ equals 12/5
theorem distance_from_Q_to_AD (A B C D N Q : ℝ × ℝ)
  (h_square : square_ABCD A B C D)
  (h_midpoint : midpoint_CD C D N)
  (h_intersect : intersect_circles N A Q D) :
  Q.2 = 12 / 5 :=
sorry

end distance_from_Q_to_AD_l206_206323


namespace other_factor_of_936_mul_w_l206_206907

theorem other_factor_of_936_mul_w (w : ℕ) (h_w_pos : 0 < w)
  (h_factors_936w : ∃ k, 936 * w = k * (3^3)) 
  (h_factors_936w_2 : ∃ m, 936 * w = m * (10^2))
  (h_w : w = 120):
  ∃ n, n = 45 :=
by
  sorry

end other_factor_of_936_mul_w_l206_206907


namespace find_d_minus_r_l206_206159

theorem find_d_minus_r :
  ∃ d r : ℕ, d > 1 ∧ (1059 % d = r) ∧ (1417 % d = r) ∧ (2312 % d = r) ∧ (d - r = 15) :=
sorry

end find_d_minus_r_l206_206159


namespace range_of_b_l206_206920

noncomputable def f (a x : ℝ) : ℝ := 
  Real.log x - (1 / 2) * a * x^2 - 2 * x

theorem range_of_b (a : ℝ) (b : ℝ) :
  (∀ x : ℝ, 0 < x ∧ x ≤ 1 → a ∈ Set.Ico (-1 : ℝ) (0 : ℝ) → f a x < b) ↔ b > -3 / 2 :=
by
  sorry

end range_of_b_l206_206920


namespace solution_set_of_inequality_l206_206376

variable (f : ℝ → ℝ)
variable (h_inc : ∀ x y : ℝ, 0 < x → 0 < y → x < y → f x < f y)

theorem solution_set_of_inequality :
  {x | 0 < x ∧ f x > f (2 * x - 4)} = {x | 2 < x ∧ x < 4} :=
by
  sorry

end solution_set_of_inequality_l206_206376


namespace typing_time_together_l206_206372

def meso_typing_rate : ℕ := 3 -- pages per minute
def tyler_typing_rate : ℕ := 5 -- pages per minute
def pages_to_type : ℕ := 40 -- pages

theorem typing_time_together :
  (meso_typing_rate + tyler_typing_rate) * 5 = pages_to_type :=
by
  sorry

end typing_time_together_l206_206372


namespace first_candidate_percentage_l206_206449

theorem first_candidate_percentage (P : ℝ) 
    (total_votes : ℝ) (votes_second : ℝ)
    (h_total_votes : total_votes = 1200)
    (h_votes_second : votes_second = 480) :
    (P / 100) * total_votes + votes_second = total_votes → P = 60 := 
by
  intro h
  rw [h_total_votes, h_votes_second] at h
  sorry

end first_candidate_percentage_l206_206449


namespace eliminating_y_l206_206316

theorem eliminating_y (x y : ℝ) (h1 : y = x + 3) (h2 : 2 * x - y = 5) : 2 * x - x - 3 = 5 :=
by {
  sorry
}

end eliminating_y_l206_206316


namespace expression_value_l206_206832

theorem expression_value (a b : ℤ) (h₁ : a = -5) (h₂ : b = 3) :
  -a - b^4 + a * b = -91 := by
  sorry

end expression_value_l206_206832


namespace john_money_left_l206_206822

variable (q : ℝ) 

def cost_soda := q
def cost_medium_pizza := 3 * q
def cost_small_pizza := 2 * q

def total_cost := 4 * cost_soda q + 2 * cost_medium_pizza q + 3 * cost_small_pizza q

theorem john_money_left (h : total_cost q = 16 * q) : 50 - total_cost q = 50 - 16 * q := by
  simp [total_cost, cost_soda, cost_medium_pizza, cost_small_pizza]
  sorry

end john_money_left_l206_206822


namespace coefficient_of_a3b2_in_expansions_l206_206894

theorem coefficient_of_a3b2_in_expansions 
  (a b c : ℝ) :
  (1 : ℝ) * (a + b)^5 * (c + c⁻¹)^8 = 700 :=
by 
  sorry

end coefficient_of_a3b2_in_expansions_l206_206894


namespace foma_should_give_ierema_55_coins_l206_206075

theorem foma_should_give_ierema_55_coins (F E Y : ℤ)
  (h1 : F - 70 = E + 70)
  (h2 : F - 40 = Y)
  (h3 : E + 70 = Y) :
  F - E = 110 → F - E = 55 :=
by
  sorry

end foma_should_give_ierema_55_coins_l206_206075


namespace capacity_of_each_bucket_in_second_case_final_proof_l206_206078

def tank_volume (buckets: ℕ) (bucket_capacity: ℝ) : ℝ := buckets * bucket_capacity

theorem capacity_of_each_bucket_in_second_case
  (total_volume: ℝ)
  (first_case_buckets : ℕ)
  (first_case_capacity : ℝ)
  (second_case_buckets : ℕ) :
  first_case_buckets * first_case_capacity = total_volume → 
  (total_volume / second_case_buckets) = 9 :=
by
  intros h
  sorry

-- Given the conditions:
noncomputable def total_volume := tank_volume 28 13.5

theorem final_proof :
  (tank_volume 28 13.5 = total_volume) → 
  (total_volume / 42 = 9) :=
by
  intro h
  exact capacity_of_each_bucket_in_second_case total_volume 28 13.5 42 h

end capacity_of_each_bucket_in_second_case_final_proof_l206_206078


namespace factorable_polynomial_l206_206329

theorem factorable_polynomial (m : ℤ) :
  (∃ A B C D E F : ℤ, 
    (A * D = 1 ∧ E + B = 4 ∧ C + F = 2 ∧ F + 3 * E + C = m + m^2 - 16)
    ∧ ((A * x + B * y + C) * (D * x + E * y + F) = x^2 + 4 * x * y + 2 * x + m * y + m^2 - 16)) ↔
  (m = 5 ∨ m = -6) :=
by
  sorry

end factorable_polynomial_l206_206329


namespace bernie_postcards_l206_206188

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

end bernie_postcards_l206_206188


namespace range_of_a_l206_206869

open Real

noncomputable def proposition_p (a : ℝ) : Prop := ∀ x : ℝ, x^2 + a * x + a > 0

noncomputable def proposition_q (a : ℝ) : Prop :=
  let Δ := 1 - 4 * a
  Δ ≥ 0

theorem range_of_a (a : ℝ) :
  ((proposition_p a ∨ proposition_q a) ∧ ¬(proposition_p a ∧ proposition_q a))
  ↔ (a ≤ 0 ∨ (1/4 : ℝ) < a ∧ a < 4) :=
by
  sorry

end range_of_a_l206_206869


namespace area_PCD_eq_l206_206392

/-- Define the points P, D, and C as given in the conditions. -/
structure Point where
  x : ℝ
  y : ℝ

def P : Point := ⟨0, 18⟩
def D : Point := ⟨3, 18⟩
def C (q : ℝ) : Point := ⟨0, q⟩

/-- Define the function to compute the area of triangle PCD given q. -/
noncomputable def area_triangle_PCD (q : ℝ) : ℝ :=
  1 / 2 * (D.x - P.x) * (P.y - q)

theorem area_PCD_eq (q : ℝ) : 
  area_triangle_PCD q = 27 - 3 / 2 * q := 
by 
  sorry

end area_PCD_eq_l206_206392


namespace georgie_initial_avocados_l206_206599

-- Define the conditions
def avocados_needed_per_serving := 3
def servings_made := 3
def avocados_bought_by_sister := 4
def total_avocados_needed := avocados_needed_per_serving * servings_made

-- The statement to prove
theorem georgie_initial_avocados : (total_avocados_needed - avocados_bought_by_sister) = 5 :=
sorry

end georgie_initial_avocados_l206_206599


namespace blue_face_area_factor_l206_206397

def cube_side_length : ℕ := 13

def original_red_face_area : ℕ := 6 * cube_side_length ^ 2
def total_mini_cube_face_area : ℕ := 6 * cube_side_length ^ 3
def blue_face_area : ℕ := total_mini_cube_face_area - original_red_face_area

theorem blue_face_area_factor :
  (blue_face_area / original_red_face_area) = 12 :=
by
  sorry

end blue_face_area_factor_l206_206397


namespace polynomial_horner_method_l206_206942

-- Define the polynomial f
def f (x : ℕ) :=
  7 * x ^ 7 + 6 * x ^ 6 + 5 * x ^ 5 + 4 * x ^ 4 + 3 * x ^ 3 + 2 * x ^ 2 + x

-- Define x as given in the condition
def x : ℕ := 3

-- State that f(x) = 262 when x = 3
theorem polynomial_horner_method : f x = 262 :=
  by
  sorry

end polynomial_horner_method_l206_206942


namespace prove_if_alpha_parallel_beta_and_a_perpendicular_beta_then_a_perpendicular_alpha_l206_206592

-- Definitions of the entities involved
variables {L : Type} -- All lines
variables {P : Type} -- All planes

-- Relations
variables (perpendicular : L → P → Prop)
variables (parallel : P → P → Prop)

-- Conditions
variables (a b : L)
variables (α β : P)

-- Statements we want to prove
theorem prove_if_alpha_parallel_beta_and_a_perpendicular_beta_then_a_perpendicular_alpha
  (H1 : parallel α β) 
  (H2 : perpendicular a β) : 
  perpendicular a α :=
  sorry

end prove_if_alpha_parallel_beta_and_a_perpendicular_beta_then_a_perpendicular_alpha_l206_206592


namespace brenda_skittles_l206_206281

theorem brenda_skittles (initial additional : ℕ) (h1 : initial = 7) (h2 : additional = 8) :
  initial + additional = 15 :=
by {
  -- Proof would go here
  sorry
}

end brenda_skittles_l206_206281


namespace people_in_room_proof_l206_206721

-- Definitions corresponding to the problem conditions
def people_in_room (total_people : ℕ) : ℕ := total_people
def seated_people (total_people : ℕ) : ℕ := (3 * total_people / 5)
def total_chairs (total_people : ℕ) : ℕ := (3 * (5 * people_in_room total_people) / 2 / 5 + 8)
def empty_chairs : ℕ := 8
def occupied_chairs (total_people : ℕ) : ℕ := (2 * total_chairs total_people / 3)

-- Proving that there are 27 people in the room
theorem people_in_room_proof (total_chairs : ℕ) :
  (seated_people 27 = 2 * total_chairs / 3) ∧ 
  (8 = total_chairs - 2 * total_chairs / 3) → 
  people_in_room 27 = 27 :=
by
  sorry

end people_in_room_proof_l206_206721


namespace time_difference_l206_206457

-- Define the conditions
def time_to_nile_delta : Nat := 4
def number_of_alligators : Nat := 7
def combined_walking_time : Nat := 46

-- Define the mathematical statement we want to prove
theorem time_difference (x : Nat) :
  4 + 7 * (time_to_nile_delta + x) = combined_walking_time → x = 2 :=
by
  sorry

end time_difference_l206_206457


namespace geometric_series_sum_eq_l206_206749

-- Given conditions
def a : ℚ := 1 / 2
def r : ℚ := 1 / 2
def n : ℕ := 5

-- Define the geometric series sum formula
def geometric_series_sum (a r : ℚ) (n : ℕ) : ℚ :=
  a * (1 - r^n) / (1 - r)

-- The main theorem to prove
theorem geometric_series_sum_eq : geometric_series_sum a r n = 31 / 32 := by
  sorry

end geometric_series_sum_eq_l206_206749


namespace parabola_distance_l206_206809

open Real

theorem parabola_distance (x₀ : ℝ) (h₁ : ∃ p > 0, (x₀^2 = 2 * p * 2) ∧ (2 + p / 2 = 5 / 2)) : abs (sqrt (x₀^2 + 4)) = 2 * sqrt 2 :=
by
  rcases h₁ with ⟨p, hp, h₀, h₂⟩
  sorry

end parabola_distance_l206_206809


namespace largest_angle_in_triangle_l206_206328

theorem largest_angle_in_triangle : 
  ∀ (A B C : ℝ), A + B + C = 180 ∧ A + B = 105 ∧ (A = B + 40)
  → (C = 75) :=
by
  sorry

end largest_angle_in_triangle_l206_206328


namespace robert_more_photos_than_claire_l206_206402

theorem robert_more_photos_than_claire
  (claire_photos : ℕ)
  (Lisa_photos : ℕ)
  (Robert_photos : ℕ)
  (Claire_takes_photos : claire_photos = 12)
  (Lisa_takes_photos : Lisa_photos = 3 * claire_photos)
  (Lisa_and_Robert_same_photos : Lisa_photos = Robert_photos) :
  Robert_photos - claire_photos = 24 := by
    sorry

end robert_more_photos_than_claire_l206_206402


namespace least_possible_value_of_smallest_integer_l206_206610

theorem least_possible_value_of_smallest_integer {A B C D : ℤ} 
  (h_diff: A ≠ B ∧ A ≠ C ∧ A ≠ D ∧ B ≠ C ∧ B ≠ D ∧ C ≠ D)
  (h_mean: (A + B + C + D) / 4 = 68)
  (h_largest: D = 90) :
  A ≥ 5 := 
sorry

end least_possible_value_of_smallest_integer_l206_206610


namespace pattern_D_cannot_form_tetrahedron_l206_206999

theorem pattern_D_cannot_form_tetrahedron :
  (¬ ∃ (f : ℝ × ℝ → ℝ × ℝ),
      f (0, 0) = (1, 1) ∧ f (1, 0) = (1, -1) ∧ f (2, 0) = (-1, 1) ∧ f (3, 0) = (-1, -1)) :=
by
  -- proof will go here
  sorry

end pattern_D_cannot_form_tetrahedron_l206_206999


namespace sandy_carrots_l206_206544

-- Definitions and conditions
def total_carrots : ℕ := 14
def mary_carrots : ℕ := 6

-- Proof statement
theorem sandy_carrots : (total_carrots - mary_carrots) = 8 :=
by
  -- sorry is used to bypass the actual proof steps
  sorry

end sandy_carrots_l206_206544


namespace polynomial_factor_l206_206209

theorem polynomial_factor (x : ℝ) : (x^2 - 4*x + 4) ∣ (x^4 + 16) :=
sorry

end polynomial_factor_l206_206209


namespace maximum_sum_of_factors_exists_maximum_sum_of_factors_l206_206315

theorem maximum_sum_of_factors {A B C : ℕ} (h1 : A ≠ B) (h2 : B ≠ C) (h3 : A ≠ C)
  (h4 : A * B * C = 2023) : A + B + C ≤ 297 :=
sorry

theorem exists_maximum_sum_of_factors : ∃ (A B C : ℕ), A ≠ B ∧ B ≠ C ∧ A ≠ C ∧ A * B * C = 2023 ∧ A + B + C = 297 :=
sorry

end maximum_sum_of_factors_exists_maximum_sum_of_factors_l206_206315


namespace total_resistance_l206_206611

theorem total_resistance (R₀ : ℝ) (h : R₀ = 10) : 
  let R₃ := R₀; let R₄ := R₀; let R₃₄ := R₃ + R₄;
  let R₂ := R₀; let R₅ := R₀; let R₂₃₄ := 1 / (1 / R₂ + 1 / R₃₄ + 1 / R₅);
  let R₁ := R₀; let R₆ := R₀; let R₁₂₃₄ := R₁ + R₂₃₄ + R₆;
  R₁₂₃₄ = 13.33 :=
by 
  sorry

end total_resistance_l206_206611


namespace smallest_y_for_perfect_cube_l206_206897

theorem smallest_y_for_perfect_cube (x y : ℕ) (x_def : x = 11 * 36 * 54) : 
  (∃ y : ℕ, y > 0 ∧ ∀ (n : ℕ), (x * y = n^3 ↔ y = 363)) := 
by 
  sorry

end smallest_y_for_perfect_cube_l206_206897


namespace represent_1917_as_sum_diff_of_squares_l206_206951

theorem represent_1917_as_sum_diff_of_squares : ∃ a b c : ℤ, 1917 = a^2 - b^2 + c^2 :=
by
  use 480, 478, 1
  sorry

end represent_1917_as_sum_diff_of_squares_l206_206951


namespace count_two_digit_integers_remainder_3_div_9_l206_206348

theorem count_two_digit_integers_remainder_3_div_9 :
  ∃ (N : ℕ), N = 10 ∧ ∀ k, (10 ≤ 9 * k + 3 ∧ 9 * k + 3 < 100) ↔ (1 ≤ k ∧ k ≤ 10) :=
by
  sorry

end count_two_digit_integers_remainder_3_div_9_l206_206348


namespace intersection_M_N_l206_206356

def M : Set ℝ := { x | -5 < x ∧ x < 3 }
def N : Set ℝ := { x | -2 < x ∧ x < 4 }

theorem intersection_M_N : M ∩ N = { x | -2 < x ∧ x < 3 } := 
by sorry

end intersection_M_N_l206_206356


namespace integer_root_sum_abs_l206_206768

theorem integer_root_sum_abs :
  ∃ a b c m : ℤ, 
    (a + b + c = 0 ∧ ab + bc + ca = -2023 ∧ |a| + |b| + |c| = 94) := sorry

end integer_root_sum_abs_l206_206768


namespace first_year_after_2023_with_digit_sum_8_l206_206312

def sum_of_digits (n : ℕ) : ℕ :=
  n.digits 10 |>.sum

theorem first_year_after_2023_with_digit_sum_8 : ∃ (y : ℕ), y > 2023 ∧ sum_of_digits y = 8 ∧ ∀ z, (z > 2023 ∧ sum_of_digits z = 8) → y ≤ z :=
by sorry

end first_year_after_2023_with_digit_sum_8_l206_206312


namespace possible_values_of_quadratic_l206_206135

theorem possible_values_of_quadratic (x : ℝ) (h : x^2 - 5 * x + 4 < 0) : 10 < x^2 + 4 * x + 5 ∧ x^2 + 4 * x + 5 < 37 :=
by
  sorry

end possible_values_of_quadratic_l206_206135


namespace line_circle_intersect_l206_206119

theorem line_circle_intersect (a : ℝ) (h : a > 1) :
  ∃ x y : ℝ, (x - a)^2 + (y - 1)^2 = 2 ∧ x - a * y - 2 = 0 :=
sorry

end line_circle_intersect_l206_206119


namespace opposite_of_neg_2023_l206_206965

theorem opposite_of_neg_2023 : ∃ x : ℤ, -2023 + x = 0 ∧ x = 2023 := 
by 
  use 2023
  constructor
  · exact rfl
  · exact rfl

end opposite_of_neg_2023_l206_206965


namespace discount_percentage_l206_206678

theorem discount_percentage (sale_price original_price : ℝ) (h1 : sale_price = 480) (h2 : original_price = 600) : 
  100 * (original_price - sale_price) / original_price = 20 := by 
  sorry

end discount_percentage_l206_206678


namespace find_mn_l206_206125

variable (OA OB OC : EuclideanSpace ℝ (Fin 3))
variable (AOC BOC : ℝ)

axiom length_OA : ‖OA‖ = 2
axiom length_OB : ‖OB‖ = 2
axiom length_OC : ‖OC‖ = 2 * Real.sqrt 3
axiom tan_angle_AOC : Real.tan AOC = 3 * Real.sqrt 3
axiom angle_BOC : BOC = Real.pi / 3

theorem find_mn : ∃ m n : ℝ, OC = m • OA + n • OB ∧ m = 5 / 3 ∧ n = 2 * Real.sqrt 3 := by
  sorry

end find_mn_l206_206125


namespace probability_two_identical_l206_206213

-- Define the number of ways to choose 3 out of 4 attractions
def choose_3_out_of_4 := Nat.choose 4 3

-- Define the total number of ways for both tourists to choose 3 attractions out of 4
def total_basic_events := choose_3_out_of_4 * choose_3_out_of_4

-- Define the number of ways to choose exactly 2 identical attractions
def ways_to_choose_2_identical := Nat.choose 4 2 * Nat.choose 2 1 * Nat.choose 1 1

-- The probability that they choose exactly 2 identical attractions
def probability : ℚ := ways_to_choose_2_identical / total_basic_events

-- Prove that this probability is 3/4
theorem probability_two_identical : probability = 3 / 4 := by
  have h1 : choose_3_out_of_4 = 4 := by sorry
  have h2 : total_basic_events = 16 := by sorry
  have h3 : ways_to_choose_2_identical = 12 := by sorry
  rw [probability, h2, h3]
  norm_num

end probability_two_identical_l206_206213


namespace inequality_always_true_l206_206567

theorem inequality_always_true (a b c d : ℝ) (h1 : a > b) (h2 : b > c) (h3 : c > d) : a + c > b + d :=
by sorry

end inequality_always_true_l206_206567


namespace find_g_l206_206640

open Real

def even (f : ℝ → ℝ) := ∀ x, f (-x) = f x
def odd (g : ℝ → ℝ) := ∀ x, g (-x) = -g x

theorem find_g 
  (f g : ℝ → ℝ) 
  (hf : even f) 
  (hg : odd g)
  (h : ∀ x, f x + g x = exp x) :
  ∀ x, g x = exp x - exp (-x) :=
by
  sorry

end find_g_l206_206640


namespace find_number_l206_206275

theorem find_number (N : ℝ) (h : 0.6 * (3 / 5) * N = 36) : N = 100 :=
by sorry

end find_number_l206_206275


namespace quadratic_inequality_l206_206971

theorem quadratic_inequality (c : ℝ) (h₁ : 0 < c) (h₂ : c < 16): ∃ x : ℝ, x^2 - 8 * x + c < 0 :=
sorry

end quadratic_inequality_l206_206971


namespace num_5_letter_words_with_at_least_one_A_l206_206414

theorem num_5_letter_words_with_at_least_one_A :
  let total := 6 ^ 5
  let without_A := 5 ^ 5
  total - without_A = 4651 := by
sorry

end num_5_letter_words_with_at_least_one_A_l206_206414


namespace range_of_m_l206_206311

theorem range_of_m (m : ℝ) : (∀ x : ℝ, 4^x - m * 2^x + 1 > 0) ↔ -2 < m ∧ m < 2 :=
by
  sorry

end range_of_m_l206_206311


namespace at_least_one_solves_l206_206531

/--
Given probabilities p1, p2, p3 that individuals A, B, and C solve a problem respectively,
prove that the probability that at least one of them solves the problem is 
1 - (1 - p1) * (1 - p2) * (1 - p3).
-/
theorem at_least_one_solves (p1 p2 p3 : ℝ) (h1 : 0 ≤ p1 ∧ p1 ≤ 1) (h2 : 0 ≤ p2 ∧ p2 ≤ 1) (h3 : 0 ≤ p3 ∧ p3 ≤ 1) :
  1 - (1 - p1) * (1 - p2) * (1 - p3) = 1 - (1 - p1) * (1 - p2) * (1 - p3) :=
by
  sorry

end at_least_one_solves_l206_206531


namespace two_quadratic_residue_mod_p_p_congruent_one_mod_2_pow_n_plus_two_l206_206338

variable (n : ℕ) (F : ℕ → ℕ) (p : ℕ)

-- Condition: F_n = 2^{2^n} + 1
def F_n (n : ℕ) : ℕ := 2^(2^n) + 1

-- Assuming n >= 2
def n_ge_two (n : ℕ) : Prop := n ≥ 2

-- Assuming p is a prime factor of F_n
def prime_factor_of_F_n (p : ℕ) (n : ℕ) : Prop := p ∣ (F_n n) ∧ Prime p

-- Part a: 2 is a quadratic residue modulo p
theorem two_quadratic_residue_mod_p (n : ℕ) (p : ℕ) (hn : n_ge_two n) (hp : prime_factor_of_F_n p n) :
  ∃ x : ℕ, x^2 ≡ 2 [MOD p] := sorry

-- Part b: p ≡ 1 (mod 2^(n+2))
theorem p_congruent_one_mod_2_pow_n_plus_two (n : ℕ) (p : ℕ) (hn : n_ge_two n) (hp : prime_factor_of_F_n p n) :
  p ≡ 1 [MOD 2^(n+2)] := sorry

end two_quadratic_residue_mod_p_p_congruent_one_mod_2_pow_n_plus_two_l206_206338


namespace proof_l206_206447

-- Define the equation and its conditions
def equation (x m : ℤ) : Prop := (3 * x - 1) / 2 + m = 3

-- Part 1: Prove that for m = 5, the corresponding x must be 1
def part1 : Prop :=
  ∃ x : ℤ, equation x 5 ∧ x = 1

-- Part 2: Prove that if the equation has a positive integer solution, the positive integer m must be 2
def part2 : Prop :=
  ∃ m x : ℤ, m > 0 ∧ x > 0 ∧ equation x m ∧ m = 2

theorem proof : part1 ∧ part2 :=
  by
    sorry

end proof_l206_206447


namespace invitations_per_package_l206_206240

-- Definitions based on conditions in the problem.
def numPackages : Nat := 5
def totalInvitations : Nat := 45

-- Definition of the problem and proof statement.
theorem invitations_per_package :
  totalInvitations / numPackages = 9 :=
by
  sorry

end invitations_per_package_l206_206240


namespace domain_f_l206_206637

noncomputable def f (x : ℝ) : ℝ := (x + 1) / (x^2 + 9*x + 18)

theorem domain_f :
  (∀ x : ℝ, (x ≠ -6) ∧ (x ≠ -3) → ∃ y : ℝ, y = f x) ∧
  (∀ x : ℝ, x = -6 ∨ x = -3 → ¬(∃ y : ℝ, y = f x)) :=
sorry

end domain_f_l206_206637


namespace quadrants_cos_sin_identity_l206_206582

theorem quadrants_cos_sin_identity (α : ℝ) 
  (h1 : π < α ∧ α < 2 * π)  -- α in the fourth quadrant
  (h2 : Real.cos α = 3 / 5) :
  (1 + Real.sqrt 2 * Real.cos (2 * α - π / 4)) / 
  (Real.sin (α + π / 2)) = -2 / 5 :=
by
  sorry

end quadrants_cos_sin_identity_l206_206582


namespace alice_bush_count_l206_206019

theorem alice_bush_count :
  let side_length := 24
  let num_sides := 3
  let bush_space := 3
  (num_sides * side_length) / bush_space = 24 :=
by
  sorry

end alice_bush_count_l206_206019


namespace part1_part2_l206_206280

variable (x : ℝ)

def A : Set ℝ := { x | 2 * x + 1 < 5 }
def B : Set ℝ := { x | x^2 - x - 2 < 0 }

theorem part1 : A ∩ B = { x | -1 < x ∧ x < 2 } :=
sorry

theorem part2 : A ∪ { x | x ≤ -1 ∨ x ≥ 2 } = Set.univ :=
sorry

end part1_part2_l206_206280


namespace votes_cast_proof_l206_206805

variable (V : ℝ)
variable (candidate_votes : ℝ)
variable (rival_votes : ℝ)

noncomputable def total_votes_cast : Prop :=
  candidate_votes = 0.40 * V ∧ 
  rival_votes = candidate_votes + 2000 ∧ 
  rival_votes = 0.60 * V ∧ 
  V = 10000

theorem votes_cast_proof : total_votes_cast V candidate_votes rival_votes :=
by {
  sorry
  }

end votes_cast_proof_l206_206805


namespace parabola_shift_l206_206069

-- Define the initial equation of the parabola
def initial_parabola (x : ℝ) : ℝ := x^2

-- Define the shift function for shifting the parabola right by 3 units
def shift_right (f : ℝ → ℝ) (a : ℝ) (x : ℝ) : ℝ := f (x - a)

-- Define the shift function for shifting the parabola up by 4 units
def shift_up (f : ℝ → ℝ) (b : ℝ) (y : ℝ) : ℝ := y + b

-- Define the transformed parabola
def transformed_parabola (x : ℝ) : ℝ := shift_up (shift_right initial_parabola 3) 4 (initial_parabola x)

-- Goal: Prove that the transformed parabola is y = (x - 3)^2 + 4
theorem parabola_shift (x : ℝ) : transformed_parabola x = (x - 3)^2 + 4 := sorry

end parabola_shift_l206_206069


namespace john_annual_profit_is_1800_l206_206227

def tenant_A_monthly_payment : ℕ := 350
def tenant_B_monthly_payment : ℕ := 400
def tenant_C_monthly_payment : ℕ := 450
def john_monthly_rent : ℕ := 900
def utility_cost : ℕ := 100
def maintenance_fee : ℕ := 50

noncomputable def annual_profit : ℕ :=
  let total_monthly_income := tenant_A_monthly_payment + tenant_B_monthly_payment + tenant_C_monthly_payment
  let total_monthly_expenses := john_monthly_rent + utility_cost + maintenance_fee
  let monthly_profit := total_monthly_income - total_monthly_expenses
  monthly_profit * 12

theorem john_annual_profit_is_1800 : annual_profit = 1800 := by
  sorry

end john_annual_profit_is_1800_l206_206227


namespace linear_dependent_iff_38_div_3_l206_206500

theorem linear_dependent_iff_38_div_3 (k : ℚ) :
  k = 38 / 3 ↔ ∃ (α β γ : ℚ), α ≠ 0 ∨ β ≠ 0 ∨ γ ≠ 0 ∧
    α * 1 + β * 4 + γ * 7 = 0 ∧
    α * 2 + β * 5 + γ * 8 = 0 ∧
    α * 3 + β * k + γ * 9 = 0 :=
by
  sorry

end linear_dependent_iff_38_div_3_l206_206500


namespace charles_earnings_l206_206080

def housesit_rate : ℝ := 15
def dog_walk_rate : ℝ := 22
def hours_housesit : ℝ := 10
def num_dogs : ℝ := 3

theorem charles_earnings :
  housesit_rate * hours_housesit + dog_walk_rate * num_dogs = 216 :=
by
  sorry

end charles_earnings_l206_206080


namespace incorrect_square_root_0_2_l206_206334

theorem incorrect_square_root_0_2 :
  (0.45)^2 = 0.2 ∧ (0.02)^2 ≠ 0.2 :=
by
  sorry

end incorrect_square_root_0_2_l206_206334


namespace part2_x_values_part3_no_real_x_for_2000_l206_206255

noncomputable def average_daily_sales (x : ℝ) : ℝ :=
  24 + 4 * x

noncomputable def profit_per_unit (x : ℝ) : ℝ :=
  60 - 5 * x

noncomputable def daily_sales_profit (x : ℝ) : ℝ :=
  (60 - 5 * x) * (24 + 4 * x)

theorem part2_x_values : 
  {x : ℝ | daily_sales_profit x = 1540} = {1, 5} := sorry

theorem part3_no_real_x_for_2000 : 
  ∀ x : ℝ, daily_sales_profit x ≠ 2000 := sorry

end part2_x_values_part3_no_real_x_for_2000_l206_206255


namespace find_xyz_sum_l206_206239

theorem find_xyz_sum (x y z : ℝ) (h1 : x^2 + x * y + y^2 = 108)
                               (h2 : y^2 + y * z + z^2 = 49)
                               (h3 : z^2 + z * x + x^2 = 157) :
  x * y + y * z + z * x = 84 :=
sorry

end find_xyz_sum_l206_206239


namespace distance_lines_eq_2_l206_206638

-- Define the first line in standard form
def line1 (x y : ℝ) : Prop := 3 * x + 4 * y + 3 = 0

-- Define the second line in standard form, established based on the parallel condition
def line2 (x y : ℝ) : Prop := 6 * x + 8 * y - 14 = 0

-- Define the condition for parallel lines which gives m
axiom parallel_lines_condition : ∀ (x y : ℝ), (line1 x y) → (line2 x y)

-- Define the distance between two parallel lines formula
noncomputable def distance_between_parallel_lines (a b c1 c2 : ℝ) : ℝ :=
  abs (c2 - c1) / (Real.sqrt (a ^ 2 + b ^ 2))

-- Prove the distance between the given lines is 2
theorem distance_lines_eq_2 : distance_between_parallel_lines 3 4 (-3) 7 = 2 :=
by
  -- Details of proof are omitted, but would show how to manipulate and calculate distances
  sorry

end distance_lines_eq_2_l206_206638


namespace solve_system_eqns_l206_206765

theorem solve_system_eqns (x y z : ℝ) (h1 : x^3 + y^3 + z^3 = 8)
  (h2 : x^2 + y^2 + z^2 = 22)
  (h3 : 1/x + 1/y + 1/z + z/(x * y) = 0) :
  (x = 3 ∧ y = 2 ∧ z = -3) ∨ (x = -3 ∧ y = 2 ∧ z = 3) ∨
  (x = 2 ∧ y = 3 ∧ z = -3) ∨ (x = 2 ∧ y = -3 ∧ z = 3) :=
by
  sorry

end solve_system_eqns_l206_206765


namespace square_area_is_correct_l206_206278

-- Define the condition: the side length of the square field
def side_length : ℝ := 7

-- Define the theorem to prove the area of the square field with given side length
theorem square_area_is_correct : side_length * side_length = 49 := by
  -- Proof goes here
  sorry

end square_area_is_correct_l206_206278


namespace count_CONES_paths_l206_206906

def diagram : List (List Char) :=
  [[' ', ' ', 'C', ' ', ' ', ' '],
   [' ', 'C', 'O', 'C', ' ', ' '],
   ['C', 'O', 'N', 'O', 'C', ' '],
   [' ', 'N', 'E', 'N', ' ', ' '],
   [' ', ' ', 'S', ' ', ' ', ' ']]

def is_adjacent (pos1 pos2 : (Nat × Nat)) : Bool :=
  (pos1.1 = pos2.1 ∨ pos1.1 + 1 = pos2.1 ∨ pos1.1 = pos2.1 + 1) ∧
  (pos1.2 = pos2.2 ∨ pos1.2 + 1 = pos2.2 ∨ pos1.2 = pos2.2 + 1)

def valid_paths (diagram : List (List Char)) : Nat :=
  -- Implementation of counting paths that spell "CONES" skipped
  sorry

theorem count_CONES_paths (d : List (List Char)) 
  (h : d = [[' ', ' ', 'C', ' ', ' ', ' '],
            [' ', 'C', 'O', 'C', ' ', ' '],
            ['C', 'O', 'N', 'O', 'C', ' '],
            [' ', 'N', 'E', 'N', ' ', ' '],
            [' ', ' ', 'S', ' ', ' ', ' ']]): valid_paths d = 6 := 
by
  sorry

end count_CONES_paths_l206_206906


namespace magnitude_product_complex_l206_206363

theorem magnitude_product_complex :
  let z1 := Complex.mk 7 (-4)
  let z2 := Complex.mk 3 11
  Complex.abs (z1 * z2) = Real.sqrt 8450 :=
by
  sorry

end magnitude_product_complex_l206_206363


namespace sin_three_pi_over_two_l206_206230

theorem sin_three_pi_over_two : Real.sin (3 * Real.pi / 2) = -1 :=
by
  sorry

end sin_three_pi_over_two_l206_206230


namespace car_distance_travelled_l206_206520

theorem car_distance_travelled (time_hours : ℝ) (time_minutes : ℝ) (time_seconds : ℝ)
    (actual_speed : ℝ) (reduced_speed : ℝ) (distance : ℝ) :
    time_hours = 1 → 
    time_minutes = 40 →
    time_seconds = 48 →
    actual_speed = 34.99999999999999 → 
    reduced_speed = (5 / 7) * actual_speed → 
    distance = reduced_speed * ((time_hours + time_minutes / 60 + time_seconds / 3600) : ℝ) →
    distance = 42 := sorry

end car_distance_travelled_l206_206520


namespace totalPeaches_l206_206241

-- Definition of conditions in the problem
def redPeaches : Nat := 4
def greenPeaches : Nat := 6
def numberOfBaskets : Nat := 1

-- Mathematical proof problem
theorem totalPeaches : numberOfBaskets * (redPeaches + greenPeaches) = 10 := by
  sorry

end totalPeaches_l206_206241


namespace sufficient_not_necessary_l206_206247

theorem sufficient_not_necessary (x : ℝ) :
  (|x - 1| < 2 → x^2 - 4 * x - 5 < 0) ∧ ¬(x^2 - 4 * x - 5 < 0 → |x - 1| < 2) :=
by
  sorry

end sufficient_not_necessary_l206_206247


namespace solve_for_z_l206_206898

theorem solve_for_z (x y : ℝ) (z : ℝ) (h : 2 / x - 1 / y = 3 / z) : 
  z = (2 * y - x) / 3 :=
by
  sorry

end solve_for_z_l206_206898


namespace eq_4_double_prime_l206_206597

-- Define the function f such that f(q) = 3q - 3
def f (q : ℕ) : ℕ := 3 * q - 3

-- Theorem statement to show that f(f(4)) = 24
theorem eq_4_double_prime : f (f 4) = 24 := by
  sorry

end eq_4_double_prime_l206_206597


namespace eq_rectangular_eq_of_polar_eq_max_m_value_l206_206226

def polar_to_rectangular (ρ θ : ℝ) : Prop := (ρ = 4 * Real.cos θ) → ∀ x y : ℝ, ρ^2 = x^2 + y^2

theorem eq_rectangular_eq_of_polar_eq (ρ θ : ℝ) :
  polar_to_rectangular ρ θ → ∀ x y : ℝ, (x - 2)^2 + y^2 = 4 :=
sorry

def max_m_condition (m : ℝ) : Prop :=
  ∀ x y : ℝ, (x - 2)^2 + y^2 = 4 → |4 + 2 * m| / Real.sqrt 5 ≤ 2

theorem max_m_value :
  (max_m_condition (Real.sqrt 5 - 2)) :=
sorry

end eq_rectangular_eq_of_polar_eq_max_m_value_l206_206226


namespace det_matrix_A_l206_206532

def matrix_A : Matrix (Fin 2) (Fin 2) ℤ :=
  ![![8, 4], ![-2, 3]]

def determinant_2x2 (A : Matrix (Fin 2) (Fin 2) ℤ) : ℤ :=
  A 0 0 * A 1 1 - A 0 1 * A 1 0

theorem det_matrix_A : determinant_2x2 matrix_A = 32 := by
  sorry

end det_matrix_A_l206_206532


namespace systematic_sampling_example_l206_206001

theorem systematic_sampling_example :
  ∃ (selected : Finset ℕ), 
    selected = {10, 30, 50, 70, 90} ∧
    ∀ n ∈ selected, 1 ≤ n ∧ n ≤ 100 ∧ 
    (∃ k, k > 0 ∧ k * 20 - 10∈ selected ∧ k * 20 - 10 ∈ Finset.range 101) := 
by
  sorry

end systematic_sampling_example_l206_206001


namespace molecular_weight_CuCO3_8_moles_l206_206808

-- Definitions for atomic weights
def atomic_weight_Cu : ℝ := 63.55
def atomic_weight_C : ℝ := 12.01
def atomic_weight_O : ℝ := 16.00

-- Definition for the molecular formula of CuCO3
def molecular_weight_CuCO3 :=
  atomic_weight_Cu + atomic_weight_C + 3 * atomic_weight_O

-- Number of moles
def moles : ℝ := 8

-- Total weight of 8 moles of CuCO3
def total_weight := moles * molecular_weight_CuCO3

-- Proof statement
theorem molecular_weight_CuCO3_8_moles :
  total_weight = 988.48 :=
  by
  sorry

end molecular_weight_CuCO3_8_moles_l206_206808


namespace factorial_mod_11_l206_206699

def factorial (n : ℕ) : ℕ :=
if n = 0 then 1 else n * factorial (n - 1)

theorem factorial_mod_11 : (factorial 13) % 11 = 0 := by
  sorry

end factorial_mod_11_l206_206699


namespace no_consecutive_positive_integers_have_sum_75_l206_206204

theorem no_consecutive_positive_integers_have_sum_75 :
  ∀ n a : ℕ, (n ≥ 2) → (a ≥ 1) → (n * (2 * a + n - 1) = 150) → False :=
by
  intros n a hn ha hsum
  sorry

end no_consecutive_positive_integers_have_sum_75_l206_206204


namespace polygon_sides_l206_206038

theorem polygon_sides (interior_angle: ℝ) (sum_exterior_angles: ℝ) (n: ℕ) (h: interior_angle = 108) (h1: sum_exterior_angles = 360): n = 5 :=
by 
  sorry

end polygon_sides_l206_206038


namespace Sue_made_22_buttons_l206_206151

def Mari_buttons : Nat := 8
def Kendra_buttons : Nat := 5 * Mari_buttons + 4
def Sue_buttons : Nat := Kendra_buttons / 2

theorem Sue_made_22_buttons : Sue_buttons = 22 :=
by
  -- proof to be added
  sorry

end Sue_made_22_buttons_l206_206151


namespace r_power_four_identity_l206_206210

-- Statement of the problem in Lean 4
theorem r_power_four_identity (r : ℝ) (h : (r + 1/r)^2 = 5) : r^4 + 1/r^4 = 7 :=
by sorry

end r_power_four_identity_l206_206210


namespace range_of_a_l206_206825

def increasing {α : Type*} [Preorder α] (f : α → α) := ∀ x y, x ≤ y → f x ≤ f y

theorem range_of_a
  (f : ℝ → ℝ)
  (increasing_f : increasing f)
  (h_domain : ∀ x, 1 ≤ x ∧ x ≤ 5 → (f x = f x))
  (h_ineq : ∀ a, 1 ≤ a + 1 ∧ a + 1 ≤ 5 ∧ 1 ≤ 2 * a - 1 ∧ 2 * a - 1 ≤ 5 ∧ f (a + 1) < f (2 * a - 1)) :
  (2 : ℝ) < a ∧ a ≤ (3 : ℝ) := 
by
  sorry

end range_of_a_l206_206825


namespace length_of_other_parallel_side_l206_206691

theorem length_of_other_parallel_side 
  (a : ℝ) (h : ℝ) (A : ℝ) (x : ℝ) 
  (h_a : a = 16) (h_h : h = 15) (h_A : A = 270) 
  (h_area_formula : A = 1 / 2 * (a + x) * h) : 
  x = 20 :=
sorry

end length_of_other_parallel_side_l206_206691


namespace correct_pairings_l206_206603

-- Define the employees
inductive Employee
| Jia
| Yi
| Bing
deriving DecidableEq

-- Define the wives
inductive Wife
| A
| B
| C
deriving DecidableEq

-- Define the friendship and age relationships
def isGoodFriend (x y : Employee) : Prop :=
  -- A's husband is Yi's good friend.
  (x = Employee.Jia ∧ y = Employee.Yi) ∨
  (x = Employee.Yi ∧ y = Employee.Jia)

def isYoungest (x : Employee) : Prop :=
  -- Specify that Jia is the youngest
  x = Employee.Jia

def isOlder (x y : Employee) : Prop :=
  -- Bing is older than C's husband.
  x = Employee.Bing ∧ y ≠ Employee.Bing

-- The pairings of husbands and wives: Jia—A, Yi—C, Bing—B.
def pairings (x : Employee) : Wife :=
  match x with
  | Employee.Jia => Wife.A
  | Employee.Yi => Wife.C
  | Employee.Bing => Wife.B

-- Proving the given pairings fit the conditions.
theorem correct_pairings : 
  ∀ (x : Employee), 
  isGoodFriend (Employee.Jia) (Employee.Yi) ∧ 
  isYoungest Employee.Jia ∧ 
  (isOlder Employee.Bing Employee.Jia ∨ isOlder Employee.Bing Employee.Yi) → 
  pairings x = match x with
               | Employee.Jia => Wife.A
               | Employee.Yi => Wife.C
               | Employee.Bing => Wife.B :=
by
  sorry

end correct_pairings_l206_206603


namespace pearls_problem_l206_206841

theorem pearls_problem :
  ∃ n : ℕ, (n % 8 = 6) ∧ (n % 7 = 5) ∧ (n = 54) ∧ (n % 9 = 0) :=
by sorry

end pearls_problem_l206_206841


namespace trey_nail_usage_l206_206719

theorem trey_nail_usage (total_decorations nails thumbtacks sticky_strips : ℕ) 
  (h1 : nails = 2 * total_decorations / 3)
  (h2 : sticky_strips = 15)
  (h3 : sticky_strips = 3 * (total_decorations - 2 * total_decorations / 3) / 5) :
  nails = 50 :=
by
  sorry

end trey_nail_usage_l206_206719


namespace find_f_value_l206_206804

noncomputable def f (x : ℝ) : ℝ := Real.sin (2 * x - 5 * Real.pi / 6)

theorem find_f_value :
  f ( -5 * Real.pi / 12 ) = Real.sqrt 3 / 2 :=
by
  sorry

end find_f_value_l206_206804


namespace arithmetic_sequence_common_difference_l206_206823

theorem arithmetic_sequence_common_difference :
  ∃ d : ℤ, 
    (∀ n, n ≤ 6 → 23 + (n - 1) * d > 0) ∧ 
    (∀ n, n ≥ 7 → 23 + (n - 1) * d < 0) ∧
    d = -4 :=
by
  sorry

end arithmetic_sequence_common_difference_l206_206823


namespace correct_statement_l206_206813

-- We assume the existence of lines and planes with certain properties.
variables {Line : Type} {Plane : Type}
variables {m n : Line} {alpha beta gamma : Plane}

-- Definitions for perpendicular and parallel relations
def perpendicular (p1 p2 : Plane) : Prop := sorry
def parallel (p1 p2 : Plane) : Prop := sorry
def line_perpendicular_to_plane (l : Line) (p : Plane) : Prop := sorry
def line_parallel_to_plane (l : Line) (p : Plane) : Prop := sorry

-- The theorem we aim to prove given the conditions
theorem correct_statement :
  line_perpendicular_to_plane m beta ∧ line_parallel_to_plane m alpha → perpendicular alpha beta :=
by sorry

end correct_statement_l206_206813


namespace soccer_and_volleyball_unit_prices_max_soccer_balls_l206_206892

-- Define the conditions and the problem
def unit_price_soccer_ball (x : ℕ) (y : ℕ) : Prop :=
  x = y + 15 ∧ 480 / x = 390 / y

def school_purchase (m : ℕ) : Prop :=
  m ≤ 70 ∧ 80 * m + 65 * (100 - m) ≤ 7550

-- Proof statement for the unit prices of soccer balls and volleyballs
theorem soccer_and_volleyball_unit_prices (x y : ℕ) (h : unit_price_soccer_ball x y) :
  x = 80 ∧ y = 65 :=
by
  sorry

-- Proof statement for the maximum number of soccer balls the school can purchase
theorem max_soccer_balls (m : ℕ) :
  school_purchase m :=
by
  sorry

end soccer_and_volleyball_unit_prices_max_soccer_balls_l206_206892


namespace find_point_A_coordinates_l206_206839

theorem find_point_A_coordinates (A B C : ℝ × ℝ)
  (hB : B = (1, 2)) (hC : C = (3, 4))
  (trans_left : ∃ l : ℝ, A = (B.1 + l, B.2))
  (trans_up : ∃ u : ℝ, A = (C.1, C.2 - u)) :
  A = (3, 2) := 
sorry

end find_point_A_coordinates_l206_206839


namespace wire_division_l206_206165

theorem wire_division (L_wire_ft : Nat) (L_wire_inch : Nat) (L_part : Nat) (H1 : L_wire_ft = 5) (H2 : L_wire_inch = 4) (H3 : L_part = 16) :
  (L_wire_ft * 12 + L_wire_inch) / L_part = 4 :=
by 
  sorry

end wire_division_l206_206165


namespace katrina_cookies_sale_l206_206949

/-- 
Katrina has 120 cookies in the beginning.
She sells 36 cookies in the morning.
She sells 16 cookies in the afternoon.
She has 11 cookies left to take home at the end of the day.
Prove that she sold 57 cookies during the lunch rush.
-/
theorem katrina_cookies_sale :
  let total_cookies := 120
  let morning_sales := 36
  let afternoon_sales := 16
  let cookies_left := 11
  let cookies_sold_lunch_rush := total_cookies - morning_sales - afternoon_sales - cookies_left
  cookies_sold_lunch_rush = 57 :=
by
  sorry

end katrina_cookies_sale_l206_206949


namespace yvette_final_bill_l206_206812

def cost_alicia : ℝ := 7.50
def cost_brant : ℝ := 10.00
def cost_josh : ℝ := 8.50
def cost_yvette : ℝ := 9.00
def tip_rate : ℝ := 0.20

def total_cost := cost_alicia + cost_brant + cost_josh + cost_yvette
def tip := tip_rate * total_cost
def final_bill := total_cost + tip

theorem yvette_final_bill :
  final_bill = 42.00 :=
  sorry

end yvette_final_bill_l206_206812


namespace correct_propositions_l206_206917

-- Definitions of propositions
def prop1 (f : ℝ → ℝ) : Prop :=
  f (-2) ≠ f (2) → ∀ x : ℝ, f (-x) ≠ f (x)

def prop2 : Prop :=
  ∀ n : ℕ, n = 0 ∨ n = 1 → (∀ x : ℝ, x ≠ 0 → x ^ n ≠ 0)

def prop3 : Prop :=
  ∀ a b : ℝ, (a ≠ 0 ∧ b ≠ 0) → (a * b ≠ 0) ∧ (a * b = 0 → a = 0 ∨ b = 0)

def prop4 (a b c d : ℝ) : Prop :=
  ∀ x : ℝ, ∃ k : ℝ, k = d → (3 * a * x ^ 2 + 2 * b * x + c ≠ 0 ∧ b ^ 2 - 3 * a * c ≥ 0)

-- Final proof statement
theorem correct_propositions (f : ℝ → ℝ) (a b c d : ℝ) :
  prop1 f ∧ ¬prop2 ∧ prop3 ∧ ¬prop4 a b c d :=
sorry

end correct_propositions_l206_206917


namespace A_eq_B_l206_206668

namespace SetsEquality

open Set

def A : Set ℝ := {x | ∃ a : ℝ, x = 5 - 4 * a + a^2}
def B : Set ℝ := {y | ∃ b : ℝ, y = 4 * b^2 + 4 * b + 2}

theorem A_eq_B : A = B := by
  sorry

end SetsEquality

end A_eq_B_l206_206668


namespace count_color_patterns_l206_206384

def regions := 6
def colors := 3

theorem count_color_patterns (h1 : regions = 6) (h2 : colors = 3) :
  3^6 - 3 * 2^6 + 3 * 1^6 = 540 := by
  sorry

end count_color_patterns_l206_206384


namespace possible_values_of_a_l206_206817

theorem possible_values_of_a :
  ∃ (a : ℤ), (∀ (b c : ℤ), (x : ℤ) → (x - a) * (x - 8) + 4 = (x + b) * (x + c)) → (a = 6 ∨ a = 10) :=
sorry

end possible_values_of_a_l206_206817


namespace total_persimmons_l206_206118

-- Definitions based on conditions in a)
def totalWeight (kg : ℕ) := kg = 3
def weightPerFivePersimmons (kg : ℕ) := kg = 1

-- The proof problem
theorem total_persimmons (k : ℕ) (w : ℕ) (x : ℕ) (h1 : totalWeight k) (h2 : weightPerFivePersimmons w) : x = 15 :=
by
  -- With the definitions totalWeight and weightPerFivePersimmons given in the conditions
  -- we aim to prove that the number of persimmons, x, is 15.
  sorry

end total_persimmons_l206_206118


namespace steve_correct_operations_l206_206991

theorem steve_correct_operations (x : ℕ) (h1 : x / 8 - 20 = 12) : ((x * 8) + 20) = 2068 :=
by
  sorry

end steve_correct_operations_l206_206991


namespace baker_batches_chocolate_chip_l206_206605

noncomputable def number_of_batches (total_cookies : ℕ) (oatmeal_cookies : ℕ) (cookies_per_batch : ℕ) : ℕ :=
  (total_cookies - oatmeal_cookies) / cookies_per_batch

theorem baker_batches_chocolate_chip (total_cookies oatmeal_cookies cookies_per_batch : ℕ) 
  (h_total : total_cookies = 10) 
  (h_oatmeal : oatmeal_cookies = 4) 
  (h_batch : cookies_per_batch = 3) : 
  number_of_batches total_cookies oatmeal_cookies cookies_per_batch = 2 :=
by
  sorry

end baker_batches_chocolate_chip_l206_206605


namespace num_of_friends_donated_same_l206_206569

def total_clothing_donated_by_adam (pants jumpers pajama_sets t_shirts : ℕ) : ℕ :=
  pants + jumpers + 2 * pajama_sets + t_shirts

def clothing_kept_by_adam (initial_donation : ℕ) : ℕ :=
  initial_donation / 2

def clothing_donated_by_friends (total_donated keeping friends_donation : ℕ) : ℕ :=
  total_donated - keeping

def num_friends (friends_donation adam_initial_donation : ℕ) : ℕ :=
  friends_donation / adam_initial_donation

theorem num_of_friends_donated_same (pants jumpers pajama_sets t_shirts total_donated : ℕ)
  (initial_donation := total_clothing_donated_by_adam pants jumpers pajama_sets t_shirts)
  (keeping := clothing_kept_by_adam initial_donation)
  (friends_donation := clothing_donated_by_friends total_donated keeping initial_donation)
  (friends := num_friends friends_donation initial_donation)
  (hp : pants = 4)
  (hj : jumpers = 4)
  (hps : pajama_sets = 4)
  (ht : t_shirts = 20)
  (htotal : total_donated = 126) :
  friends = 3 :=
by
  sorry

end num_of_friends_donated_same_l206_206569


namespace cone_volume_l206_206828

-- Define the condition
def cylinder_volume : ℝ := 30

-- Define the statement that needs to be proven
theorem cone_volume (h_cylinder_volume : cylinder_volume = 30) : cylinder_volume / 3 = 10 := 
by 
  -- Proof omitted
  sorry

end cone_volume_l206_206828


namespace triangle_perimeter_l206_206306

-- Let the lengths of the sides of the triangle be a, b, c.
variables (a b c : ℕ)
-- To represent the sides with specific lengths as stated in the problem.
def side1 := 2
def side2 := 5

-- The condition that the third side must be an odd integer.
def is_odd (n : ℕ) : Prop := ∃ k : ℕ, n = 2 * k + 1

-- Setting up the third side based on the given conditions.
def third_side_odd (c : ℕ) : Prop := 3 < c ∧ c < 7 ∧ is_odd c

-- The perimeter of the triangle.
def perimeter (a b c : ℕ) : ℕ := a + b + c

-- The main theorem to prove.
theorem triangle_perimeter (c : ℕ) (h_odd : third_side_odd c) : perimeter side1 side2 c = 12 :=
by
  sorry

end triangle_perimeter_l206_206306


namespace arithmetic_mean_of_p_and_q_l206_206583

variable (p q r : ℝ)

theorem arithmetic_mean_of_p_and_q
  (h1 : (p + q) / 2 = 10)
  (h2 : (q + r) / 2 = 22)
  (h3 : r - p = 24) :
  (p + q) / 2 = 10 :=
by
  sorry

end arithmetic_mean_of_p_and_q_l206_206583


namespace min_fence_length_l206_206656

theorem min_fence_length (x : ℝ) (h : x > 0) (A : x * (64 / x) = 64) : 2 * (x + 64 / x) ≥ 32 :=
by
  have t := (2 * (x + 64 / x)) 
  sorry -- Proof omitted, only statement provided as per instructions

end min_fence_length_l206_206656


namespace combined_salary_l206_206497

theorem combined_salary (S_B : ℝ) (S_A : ℝ) (h1 : S_B = 8000) (h2 : 0.20 * S_A = 0.15 * S_B) : 
S_A + S_B = 14000 :=
by {
  sorry
}

end combined_salary_l206_206497


namespace hens_count_l206_206394

theorem hens_count
  (H C : ℕ)
  (heads_eq : H + C = 48)
  (feet_eq : 2 * H + 4 * C = 136) :
  H = 28 :=
by
  sorry

end hens_count_l206_206394


namespace liangliang_distance_to_school_l206_206112

theorem liangliang_distance_to_school :
  (∀ (t : ℕ), (40 * t = 50 * (t - 5)) → (40 * 25 = 1000)) :=
sorry

end liangliang_distance_to_school_l206_206112


namespace Tim_Linda_Mow_Lawn_l206_206559

theorem Tim_Linda_Mow_Lawn :
  let tim_time := 1.5
  let linda_time := 2
  let tim_rate := 1 / tim_time
  let linda_rate := 1 / linda_time
  let combined_rate := tim_rate + linda_rate
  let combined_time_hours := 1 / combined_rate
  let combined_time_minutes := combined_time_hours * 60
  combined_time_minutes = 51.43 := 
by
    sorry

end Tim_Linda_Mow_Lawn_l206_206559


namespace find_N_l206_206462

theorem find_N (a b c N : ℚ) (h_sum : a + b + c = 84)
    (h_a : a - 7 = N) (h_b : b + 7 = N) (h_c : c / 7 = N) : 
    N = 28 / 3 :=
sorry

end find_N_l206_206462


namespace no_integer_solutions_l206_206623

theorem no_integer_solutions (x y : ℤ) : ¬ (x^2 + 4 * x - 11 = 8 * y) := 
by
  sorry

end no_integer_solutions_l206_206623


namespace sum_abc_of_quadrilateral_l206_206332

noncomputable def distance (p1 p2 : (ℝ × ℝ)) : ℝ :=
  Real.sqrt ((p2.1 - p1.1) ^ 2 + (p2.2 - p1.2) ^ 2)

theorem sum_abc_of_quadrilateral :
  let p1 := (0, 0)
  let p2 := (4, 3)
  let p3 := (5, 2)
  let p4 := (4, -1)
  let perimeter := 
    distance p1 p2 + distance p2 p3 + distance p3 p4 + distance p4 p1
  let a : ℤ := 1    -- corresponding to the equivalent simplified distances to √5 parts
  let b : ℤ := 2    -- corresponding to the equivalent simplified distances to √2 parts
  let c : ℤ := 9    -- rest constant integer simplified part
  a + b + c = 12 :=
by
  sorry

end sum_abc_of_quadrilateral_l206_206332


namespace determine_x_l206_206307

theorem determine_x (x : ℝ) : (∀ y : ℝ, 10 * x * y - 15 * y + 2 * x - 3 = 0) → x = 3 / 2 :=
by
  intro h
  have : ∀ y : ℝ, (5 * y + 1) * (2 * x - 3) = 0 := 
    sorry
  have : (2 * x - 3) = 0 := 
    sorry
  show x = 3 / 2
  sorry

end determine_x_l206_206307


namespace singers_in_choir_l206_206107

variable (X : ℕ)

/-- In the first verse, only half of the total singers sang -/ 
def first_verse_not_singing (X : ℕ) : ℕ := X / 2

/-- In the second verse, a third of the remaining singers joined in -/
def second_verse_joining (X : ℕ) : ℕ := (X / 2) / 3

/-- In the final third verse, 10 people joined so that the whole choir sang together -/
def remaining_singers_after_second_verse (X : ℕ) : ℕ := first_verse_not_singing X - second_verse_joining X

def final_verse_joining_condition (X : ℕ) : Prop := remaining_singers_after_second_verse X = 10

theorem singers_in_choir : ∃ (X : ℕ), final_verse_joining_condition X ∧ X = 30 :=
by
  sorry

end singers_in_choir_l206_206107


namespace cos_sum_is_one_or_cos_2a_l206_206995

open Real

theorem cos_sum_is_one_or_cos_2a (a b : ℝ) (h : ∫ x in a..b, sin x = 0) : cos (a + b) = 1 ∨ cos (a + b) = cos (2 * a) :=
  sorry

end cos_sum_is_one_or_cos_2a_l206_206995


namespace sufficient_not_necessary_condition_l206_206260

theorem sufficient_not_necessary_condition (a : ℝ) :
  (∀ x : ℝ, x ∈ Set.Iic (-2) → (x^2 + 2 * a * x - 2) ≤ ((x - 1)^2 + 2 * a * (x - 1) - 2)) ↔ a ≤ 2 := by
  sorry

end sufficient_not_necessary_condition_l206_206260


namespace adjacent_books_probability_l206_206932

def chinese_books : ℕ := 2
def math_books : ℕ := 2
def physics_books : ℕ := 1
def total_books : ℕ := chinese_books + math_books + physics_books

theorem adjacent_books_probability :
  (total_books = 5) →
  (chinese_books = 2) →
  (math_books = 2) →
  (physics_books = 1) →
  (∃ p : ℚ, p = 1 / 5) :=
by
  intros h1 h2 h3 h4
  -- Proof omitted.
  exact ⟨1 / 5, rfl⟩

end adjacent_books_probability_l206_206932


namespace smallest_boxes_l206_206936

theorem smallest_boxes (n : Nat) (h₁ : n % 5 = 0) (h₂ : n % 24 = 0) : n = 120 := 
  sorry

end smallest_boxes_l206_206936


namespace distance_AB_polar_l206_206039

open Real

/-- The distance between points A and B in polar coordinates, given that θ₁ - θ₂ = π. -/
theorem distance_AB_polar (A B : ℝ × ℝ) (r1 r2 : ℝ) (θ1 θ2 : ℝ) (hA : A = (r1, θ1)) (hB : B = (r2, θ2)) (hθ : θ1 - θ2 = π) :
  dist (r1 * cos θ1, r1 * sin θ1) (r2 * cos θ2, r2 * sin θ2) = r1 + r2 :=
sorry

end distance_AB_polar_l206_206039


namespace Mr_Caiden_payment_l206_206267

-- Defining the conditions as variables and constants
def total_roofing_needed : ℕ := 300
def cost_per_foot : ℕ := 8
def free_roofing : ℕ := 250

-- Define the remaining roofing needed and the total cost
def remaining_roofing : ℕ := total_roofing_needed - free_roofing
def total_cost : ℕ := remaining_roofing * cost_per_foot

-- The proof statement: 
theorem Mr_Caiden_payment : total_cost = 400 := 
by
  -- Proof omitted
  sorry

end Mr_Caiden_payment_l206_206267


namespace xiao_zhao_physical_education_grade_l206_206404

def classPerformanceScore : ℝ := 40
def midtermExamScore : ℝ := 50
def finalExamScore : ℝ := 45

def classPerformanceWeight : ℝ := 0.3
def midtermExamWeight : ℝ := 0.2
def finalExamWeight : ℝ := 0.5

def overallGrade : ℝ :=
  (classPerformanceScore * classPerformanceWeight) +
  (midtermExamScore * midtermExamWeight) +
  (finalExamScore * finalExamWeight)

theorem xiao_zhao_physical_education_grade : overallGrade = 44.5 := by
  sorry

end xiao_zhao_physical_education_grade_l206_206404


namespace evaluate_expression_l206_206677

theorem evaluate_expression : 2 + (2 / (2 + (2 / (2 + 3)))) = 17 / 6 := 
by
  sorry

end evaluate_expression_l206_206677


namespace log_inequality_l206_206579

variable (a b : ℝ)

theorem log_inequality (h1 : a > b) (h2 : b > 1) : a * Real.log a > b * Real.log b :=
sorry

end log_inequality_l206_206579


namespace seventy_times_reciprocal_l206_206852

theorem seventy_times_reciprocal (x : ℚ) (hx : 7 * x = 3) : 70 * (1 / x) = 490 / 3 :=
by 
  sorry

end seventy_times_reciprocal_l206_206852


namespace geometric_sequence_term_l206_206979

/-
Prove that the 303rd term in a geometric sequence with the first term a1 = 5 and the second term a2 = -10 is 5 * 2^302.
-/

theorem geometric_sequence_term :
  let a1 := 5
  let a2 := -10
  let r := a2 / a1
  let n := 303
  let a_n := a1 * r^(n-1)
  a_n = 5 * 2^302 :=
by
  let a1 := 5
  let a2 := -10
  let r := a2 / a1
  let n := 303
  have h1 : a1 * r^(n-1) = 5 * 2^302 := sorry
  exact h1

end geometric_sequence_term_l206_206979


namespace range_of_f_l206_206837

noncomputable def f (x : ℝ) : ℝ :=
  if h : x < 1 then 3^(-x) else x^2

theorem range_of_f (x : ℝ) : (f x > 9) ↔ (x < -2 ∨ x > 3) :=
by
  sorry

end range_of_f_l206_206837


namespace find_value_of_A_l206_206606

theorem find_value_of_A (A B : ℕ) (h_ratio : A * 5 = 3 * B) (h_diff : B - A = 12) : A = 18 :=
by
  sorry

end find_value_of_A_l206_206606


namespace probability_ace_king_queen_l206_206172

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

end probability_ace_king_queen_l206_206172


namespace evaluate_expression_l206_206401

theorem evaluate_expression (a b : ℚ) (h1 : a + b = 4) (h2 : a - b = 2) :
  ( (a^2 - 6 * a * b + 9 * b^2) / (a^2 - 2 * a * b) / ((5 * b^2 / (a - 2 * b)) - (a + 2 * b)) - 1 / a ) = -1 / 3 :=
by
  sorry

end evaluate_expression_l206_206401


namespace geometric_series_sum_l206_206000

theorem geometric_series_sum (a r : ℚ) (n : ℕ) (h_a : a = 1) (h_r : r = 1 / 2) (h_n : n = 5) :
  ((a * (1 - r^n)) / (1 - r)) = 31 / 16 := 
by
  sorry

end geometric_series_sum_l206_206000


namespace solve_for_x_l206_206131

theorem solve_for_x (x : ℤ) (h : 158 - x = 59) : x = 99 :=
by
  sorry

end solve_for_x_l206_206131


namespace expand_product_l206_206923

theorem expand_product (x : ℝ) : (x + 4) * (x - 9) = x^2 - 5*x - 36 :=
by
  sorry

end expand_product_l206_206923


namespace average_of_remaining_two_numbers_l206_206096

theorem average_of_remaining_two_numbers :
  ∀ (a b c d e f : ℝ),
    (a + b + c + d + e + f) / 6 = 3.95 →
    (a + b) / 2 = 3.6 →
    (c + d) / 2 = 3.85 →
    ((e + f) / 2 = 4.4) :=
by
  intros a b c d e f h1 h2 h3
  have h4 : a + b + c + d + e + f = 23.7 := sorry
  have h5 : a + b = 7.2 := sorry
  have h6 : c + d = 7.7 := sorry
  have h7 : e + f = 8.8 := sorry
  exact sorry

end average_of_remaining_two_numbers_l206_206096


namespace Ellen_won_17_legos_l206_206359

theorem Ellen_won_17_legos (initial_legos : ℕ) (current_legos : ℕ) (h₁ : initial_legos = 2080) (h₂ : current_legos = 2097) : 
  current_legos - initial_legos = 17 := 
  by 
    sorry

end Ellen_won_17_legos_l206_206359


namespace square_nonneg_of_nonneg_l206_206452

theorem square_nonneg_of_nonneg (x : ℝ) (hx : 0 ≤ x) : 0 ≤ x^2 :=
sorry

end square_nonneg_of_nonneg_l206_206452


namespace periodicity_f_l206_206526

noncomputable def vectorA (x : ℝ) : ℝ × ℝ := (2 * Real.sin x, Real.cos x)
noncomputable def vectorB (x : ℝ) : ℝ × ℝ := (Real.cos x, 2 * Real.sqrt 3 * Real.cos x)
noncomputable def f (x : ℝ) : ℝ :=
  let a := vectorA x
  let b := vectorB x
  a.1 * b.1 + a.2 * b.2

theorem periodicity_f :
  (∀ x, f (x + Real.pi) = f x) ∧
  (∀ x ∈ Set.Icc 0 (Real.pi / 2), (f x = 2 + Real.sqrt 3 ∨ f x = 0)) :=
by
  sorry

end periodicity_f_l206_206526


namespace inequality_proof_l206_206853

theorem inequality_proof (a b c d : ℝ) (h1 : 0 < a) (h2 : 0 < b) (h3 : 0 < c) (h4 : 0 < d) (h5 : a * b * c * d = 1) :
  1 < (b / (a * b + b + 1) + c / (b * c + c + 1) + d / (c * d + d + 1) + a / (d * a + a + 1)) ∧
  (b / (a * b + b + 1) + c / (b * c + c + 1) + d / (c * d + d + 1) + a / (d * a + a + 1)) < 2 :=
sorry

end inequality_proof_l206_206853


namespace line_intersects_xaxis_at_l206_206718

theorem line_intersects_xaxis_at (x y : ℝ) 
  (h : 4 * y - 5 * x = 15) 
  (hy : y = 0) : (x, y) = (-3, 0) :=
by
  sorry

end line_intersects_xaxis_at_l206_206718


namespace fewer_cubes_needed_l206_206461

variable (cubeVolume : ℕ) (length : ℕ) (width : ℕ) (depth : ℕ) (TVolume : ℕ)

theorem fewer_cubes_needed : 
  cubeVolume = 5 → 
  length = 7 → 
  width = 7 → 
  depth = 6 → 
  TVolume = 3 → 
  (length * width * depth - TVolume = 291) :=
by
  intros hc hl hw hd ht
  sorry

end fewer_cubes_needed_l206_206461


namespace smallest_c_for_polynomial_l206_206790

theorem smallest_c_for_polynomial :
  ∃ r1 r2 r3 : ℕ, (r1 * r2 * r3 = 2310) ∧ (r1 + r2 + r3 = 52) := sorry

end smallest_c_for_polynomial_l206_206790


namespace generalized_inequality_combinatorial_inequality_l206_206262

-- Part 1: Generalized Inequality
theorem generalized_inequality (n : ℕ) (a b : Fin n → ℝ) 
  (ha : ∀ i, 0 < a i) (hb : ∀ i, 0 < b i) :
  (Finset.univ.sum (fun i => (b i)^2 / (a i))) ≥
  ((Finset.univ.sum (fun i => b i))^2 / (Finset.univ.sum (fun i => a i))) :=
sorry

-- Part 2: Combinatorial Inequality
theorem combinatorial_inequality (n : ℕ) (hn : 0 < n) :
  (Finset.range (n + 1)).sum (fun k => (2 * k + 1) / (Nat.choose n k)) ≥
  ((n + 1)^3 / (2^n : ℝ)) :=
sorry

end generalized_inequality_combinatorial_inequality_l206_206262


namespace problem_a_b_squared_l206_206235

theorem problem_a_b_squared {a b : ℝ} (h1 : a + 3 = (b-1)^2) (h2 : b + 3 = (a-1)^2) (h3 : a ≠ b) : a^2 + b^2 = 10 :=
by
  sorry

end problem_a_b_squared_l206_206235


namespace cost_of_carpeting_l206_206985

noncomputable def cost_per_meter_in_paise (cost : ℝ) (length_in_meters : ℝ) : ℝ :=
  cost * 100 / length_in_meters

theorem cost_of_carpeting (room_length room_breadth carpet_width_m cost_total : ℝ) (h1 : room_length = 15) 
  (h2 : room_breadth = 6) (h3 : carpet_width_m = 0.75) (h4 : cost_total = 36) :
  cost_per_meter_in_paise cost_total (room_length * room_breadth / carpet_width_m) = 30 :=
by
  sorry

end cost_of_carpeting_l206_206985


namespace unattainable_y_l206_206274

theorem unattainable_y (x : ℝ) (h : x ≠ -(5 / 4)) :
    (∀ y : ℝ, y = (2 - 3 * x) / (4 * x + 5) → y ≠ -3 / 4) :=
by
  -- Placeholder for the proof
  sorry

end unattainable_y_l206_206274


namespace adam_played_rounds_l206_206144

theorem adam_played_rounds (total_points points_per_round : ℕ) (h_total : total_points = 283) (h_per_round : points_per_round = 71) : total_points / points_per_round = 4 := by
  -- sorry is a placeholder for the actual proof
  sorry

end adam_played_rounds_l206_206144


namespace train_pass_time_l206_206616

def speed_jogger := 9   -- in km/hr
def distance_ahead := 240   -- in meters
def length_train := 150   -- in meters
def speed_train := 45   -- in km/hr

noncomputable def time_to_pass_jogger : ℝ :=
  let speed_jogger_mps := speed_jogger * (1000 / 3600)
  let speed_train_mps := speed_train * (1000 / 3600)
  let relative_speed := speed_train_mps - speed_jogger_mps
  let total_distance := distance_ahead + length_train
  total_distance / relative_speed

theorem train_pass_time : time_to_pass_jogger = 39 :=
  by
    sorry

end train_pass_time_l206_206616


namespace geometric_sequence_sum_l206_206353

theorem geometric_sequence_sum (a : ℕ → ℝ) (q : ℝ) 
  (h₀ : ∀ n : ℕ, a (n + 1) = a n * q)
  (h₁ : a 3 = 4) (h₂ : a 2 + a 4 = -10) (h₃ : |q| > 1) : 
  (a 0 + a 1 + a 2 + a 3 = -5) := 
by 
  sorry

end geometric_sequence_sum_l206_206353


namespace div_result_l206_206249

theorem div_result : 2.4 / 0.06 = 40 := 
sorry

end div_result_l206_206249


namespace solve_eq1_solve_eq2_l206_206395

theorem solve_eq1 (x : ℝ) : 3 * x * (x + 3) = 2 * (x + 3) ↔ (x = -3 ∨ x = 2 / 3) :=
by sorry

theorem solve_eq2 (x : ℝ) : x^2 - 4 * x - 5 = 0 ↔ (x = 5 ∨ x = -1) :=
by sorry

end solve_eq1_solve_eq2_l206_206395


namespace inequality_div_two_l206_206919

theorem inequality_div_two (x y : ℝ) (h : x > y) : x / 2 > y / 2 := sorry

end inequality_div_two_l206_206919


namespace find_n_l206_206896

theorem find_n (n : ℕ) : 
  Nat.lcm n 12 = 48 ∧ Nat.gcd n 12 = 8 → n = 32 := 
by 
  sorry

end find_n_l206_206896


namespace fifteen_percent_of_x_is_ninety_l206_206398

theorem fifteen_percent_of_x_is_ninety :
  ∃ (x : ℝ), (15 / 100) * x = 90 ↔ x = 600 :=
by
  sorry

end fifteen_percent_of_x_is_ninety_l206_206398


namespace frame_percentage_l206_206688

theorem frame_percentage : 
  let side_length := 80
  let frame_width := 4
  let total_area := side_length * side_length
  let picture_side_length := side_length - 2 * frame_width
  let picture_area := picture_side_length * picture_side_length
  let frame_area := total_area - picture_area
  let frame_percentage := (frame_area * 100) / total_area
  frame_percentage = 19 := 
by
  sorry

end frame_percentage_l206_206688


namespace probability_of_odd_score_l206_206086

noncomputable def dartboard : Type := sorry

variables (r_inner r_outer : ℝ)
variables (inner_values outer_values : Fin 3 → ℕ)
variables (P_odd : ℚ)

-- Conditions
def dartboard_conditions (r_inner r_outer : ℝ) (inner_values outer_values : Fin 3 → ℕ) : Prop :=
  r_inner = 4 ∧ r_outer = 8 ∧
  inner_values 0 = 3 ∧ inner_values 1 = 1 ∧ inner_values 2 = 1 ∧
  outer_values 0 = 3 ∧ outer_values 1 = 2 ∧ outer_values 2 = 2

-- Correct Answer
def correct_odds_probability (P_odd : ℚ) : Prop :=
  P_odd = 4 / 9

-- Main Statement
theorem probability_of_odd_score (r_inner r_outer : ℝ) (inner_values outer_values : Fin 3 → ℕ) (P_odd : ℚ) :
  dartboard_conditions r_inner r_outer inner_values outer_values →
  correct_odds_probability P_odd :=
sorry

end probability_of_odd_score_l206_206086


namespace am_gm_inequality_l206_206221

theorem am_gm_inequality (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  (a^3 / (b * c)) + (b^3 / (c * a)) + (c^3 / (a * b)) ≥ a + b + c :=
  sorry

end am_gm_inequality_l206_206221


namespace Tim_took_out_11_rulers_l206_206009

-- Define the initial number of rulers
def initial_rulers := 14

-- Define the number of rulers left in the drawer
def rulers_left := 3

-- Define the number of rulers taken by Tim
def rulers_taken := initial_rulers - rulers_left

-- Statement to prove that the number of rulers taken by Tim is indeed 11
theorem Tim_took_out_11_rulers : rulers_taken = 11 := by
  sorry

end Tim_took_out_11_rulers_l206_206009


namespace certain_number_exists_l206_206925

theorem certain_number_exists :
  ∃ x : ℤ, 55 * x % 7 = 6 ∧ x % 7 = 1 := by
  sorry

end certain_number_exists_l206_206925


namespace quadratic_inequality_solution_l206_206514

theorem quadratic_inequality_solution : ∀ x : ℝ, -8 * x^2 + 4 * x - 7 ≤ 0 :=
by
  sorry

end quadratic_inequality_solution_l206_206514


namespace original_price_of_computer_l206_206502

theorem original_price_of_computer
  (P : ℝ)
  (h1 : 1.30 * P = 351)
  (h2 : 2 * P = 540) :
  P = 270 :=
by
  sorry

end original_price_of_computer_l206_206502


namespace angle_A_is_120_max_sin_B_plus_sin_C_l206_206321

-- Define the measures in degrees using real numbers
variable (a b c R : Real)
variable (A B C : ℝ) (sin cos : ℝ → ℝ)

-- Question 1: Prove A = 120 degrees given the initial condition
theorem angle_A_is_120
  (H1 : 2 * a * (sin A) = (2 * b + c) * (sin B) + (2 * c + b) * (sin C)) :
  A = 120 :=
by
  sorry

-- Question 2: Given the angles sum to 180 degrees and A = 120 degrees, prove the max value of sin B + sin C is 1
theorem max_sin_B_plus_sin_C
  (H2 : A + B + C = 180)
  (H3 : A = 120) :
  (sin B) + (sin C) ≤ 1 :=
by
  sorry

end angle_A_is_120_max_sin_B_plus_sin_C_l206_206321


namespace parking_spots_l206_206904

def numberOfLevels := 5
def openSpotsOnLevel1 := 4
def openSpotsOnLevel2 := openSpotsOnLevel1 + 7
def openSpotsOnLevel3 := openSpotsOnLevel2 + 6
def openSpotsOnLevel4 := 14
def openSpotsOnLevel5 := openSpotsOnLevel4 + 5
def totalOpenSpots := openSpotsOnLevel1 + openSpotsOnLevel2 + openSpotsOnLevel3 + openSpotsOnLevel4 + openSpotsOnLevel5

theorem parking_spots :
  openSpotsOnLevel5 = 19 ∧ totalOpenSpots = 65 := by
  sorry

end parking_spots_l206_206904


namespace carlos_payment_l206_206143

theorem carlos_payment (A B C : ℝ) (hB_lt_A : B < A) (hB_lt_C : B < C) :
    B + (0.35 * (A + B + C) - B) = 0.35 * A - 0.65 * B + 0.35 * C :=
by sorry

end carlos_payment_l206_206143


namespace problem_l206_206626

noncomputable def f (x : ℝ) (m : ℝ) (n : ℝ) (α1 : ℝ) (α2 : ℝ) :=
  m * Real.sin (Real.pi * x + α1) + n * Real.cos (Real.pi * x + α2)

variables (m n α1 α2 : ℝ) (h_m : m ≠ 0) (h_n : n ≠ 0) (h_α1 : α1 ≠ 0) (h_α2 : α2 ≠ 0)

theorem problem (h : f 2008 m n α1 α2 = 1) : f 2009 m n α1 α2 = -1 :=
  sorry

end problem_l206_206626


namespace integer_product_is_192_l206_206237

theorem integer_product_is_192 (A B C : ℤ)
  (h1 : A + B + C = 33)
  (h2 : C = 3 * B)
  (h3 : A = C - 23) :
  A * B * C = 192 :=
sorry

end integer_product_is_192_l206_206237


namespace line_through_intersection_parallel_to_given_line_l206_206667

theorem line_through_intersection_parallel_to_given_line :
  ∃ k : ℝ, (∀ x y : ℝ, (2 * x + 3 * y + k = 0 ↔ (x, y) = (2, 1)) ∧
  (∀ m n : ℝ, (2 * m + 3 * n + 5 = 0 → 2 * m + 3 * n + k = 0))) →
  2 * x + 3 * y - 7 = 0 :=
sorry

end line_through_intersection_parallel_to_given_line_l206_206667


namespace union_inter_complement_l206_206106

open Set

variable (U : Set ℝ := univ)
variable (A : Set ℝ := {x | abs (x - 2) > 3})
variable (B : Set ℝ := {x | x * (-2 - x) > 0})

theorem union_inter_complement 
  (C_U_A : Set ℝ := compl A)
  (A_def : A = {x | abs (x - 2) > 3})
  (B_def : B = {x | x * (-2 - x) > 0})
  (C_U_A_def : C_U_A = compl A) :
  (A ∪ B = {x : ℝ | x < 0} ∪ {x : ℝ | x > 5}) ∧ 
  ((C_U_A ∩ B) = {x : ℝ | -1 ≤ x ∧ x < 0}) :=
by
  sorry

end union_inter_complement_l206_206106


namespace number_of_possible_values_of_r_eq_894_l206_206557

noncomputable def r_possible_values : ℕ :=
  let lower_bound := 0.3125
  let upper_bound := 0.4018
  let min_r := 3125  -- equivalent to the lowest four-digit decimal ≥ 0.3125
  let max_r := 4018  -- equivalent to the highest four-digit decimal ≤ 0.4018
  1 + max_r - min_r  -- total number of possible values

theorem number_of_possible_values_of_r_eq_894 :
  r_possible_values = 894 :=
by
  sorry

end number_of_possible_values_of_r_eq_894_l206_206557


namespace new_job_larger_than_original_l206_206333

theorem new_job_larger_than_original (original_workers original_days new_workers new_days : ℕ) 
  (h_original_workers : original_workers = 250)
  (h_original_days : original_days = 16)
  (h_new_workers : new_workers = 600)
  (h_new_days : new_days = 20) :
  (new_workers * new_days) / (original_workers * original_days) = 3 := by
  sorry

end new_job_larger_than_original_l206_206333


namespace line_intersects_midpoint_l206_206968

theorem line_intersects_midpoint (c : ℤ) : 
  (∃x y : ℤ, 2 * x - y = c ∧ x = (1 + 5) / 2 ∧ y = (3 + 11) / 2) → c = -1 := by
  sorry

end line_intersects_midpoint_l206_206968


namespace combined_6th_grade_percent_is_15_l206_206542

-- Definitions
def annville_students := 100
def cleona_students := 200

def percent_6th_annville := 11
def percent_6th_cleona := 17

def total_students := annville_students + cleona_students
def total_6th_students := (percent_6th_annville * annville_students / 100) + (percent_6th_cleona * cleona_students / 100)

def percent_6th_combined := (total_6th_students * 100) / total_students

-- Theorem statement
theorem combined_6th_grade_percent_is_15 : percent_6th_combined = 15 :=
by
  sorry

end combined_6th_grade_percent_is_15_l206_206542


namespace simplify_eval_expression_l206_206238

theorem simplify_eval_expression (a b : ℤ) (h₁ : a = 2) (h₂ : b = -1) : 
  ((2 * a + 3 * b) * (2 * a - 3 * b) - (2 * a - b) ^ 2 - 2 * a * b) / (-2 * b) = -7 := 
by 
  sorry

end simplify_eval_expression_l206_206238


namespace solve_D_l206_206844

-- Define the digits represented by each letter
variable (P M T D E : ℕ)

-- Each letter represents a different digit (0-9) and should be distinct
axiom distinct_digits : (P ≠ M) ∧ (P ≠ T) ∧ (P ≠ D) ∧ (P ≠ E) ∧ 
                        (M ≠ T) ∧ (M ≠ D) ∧ (M ≠ E) ∧ 
                        (T ≠ D) ∧ (T ≠ E) ∧ 
                        (D ≠ E)

-- Each letter is a digit from 0 to 9
axiom digit_range : 0 ≤ P ∧ P ≤ 9 ∧ 0 ≤ M ∧ M ≤ 9 ∧ 
                    0 ≤ T ∧ T ≤ 9 ∧ 0 ≤ D ∧ D ≤ 9 ∧ 
                    0 ≤ E ∧ E ≤ 9

-- Each column sums to the digit below it, considering carry overs from right to left
axiom column1 : T + T + E = E ∨ T + T + E = 10 + E
axiom column2 : E + D + T + (if T + T + E = 10 + E then 1 else 0) = P
axiom column3 : P + M + (if E + D + T + (if T + T + E = 10 + E then 1 else 0) = 10 + P then 1 else 0) = M

-- Prove that D = 4 given the above conditions
theorem solve_D : D = 4 :=
by sorry

end solve_D_l206_206844


namespace erica_blank_question_count_l206_206420

variable {C W B : ℕ}

theorem erica_blank_question_count
  (h1 : C + W + B = 20)
  (h2 : 7 * C - 4 * W = 100) :
  B = 1 :=
by
  sorry

end erica_blank_question_count_l206_206420


namespace k_is_square_l206_206445

theorem k_is_square (a b : ℕ) (h_a : a > 0) (h_b : b > 0) (k : ℕ) (h_k : k > 0)
    (h : (a^2 + b^2) = k * (a * b + 1)) : ∃ (n : ℕ), n^2 = k :=
sorry

end k_is_square_l206_206445


namespace students_in_school_at_least_225_l206_206598

-- Conditions as definitions
def students_in_band := 85
def students_in_sports := 200
def students_in_both := 60
def students_in_either := 225

-- The proof statement
theorem students_in_school_at_least_225 :
  students_in_band + students_in_sports - students_in_both = students_in_either :=
by
  -- This statement will just assert the correctness as per given information in the problem
  sorry

end students_in_school_at_least_225_l206_206598


namespace smallest_n_digit_sum_l206_206816

theorem smallest_n_digit_sum :
  ∃ n : ℕ, (∃ (arrangements : ℕ), arrangements > 1000000 ∧ arrangements = (1/2 * ((n + 1) * (n + 2)))) ∧ (1 + n / 1000 + (n % 1000) / 100 + (n % 100) / 10 + n % 10 = 9) :=
sorry

end smallest_n_digit_sum_l206_206816


namespace blue_beads_l206_206595

-- Variables to denote the number of blue, red, white, and silver beads
variables (B R W S : ℕ)

-- Conditions derived from the problem statement
def conditions : Prop :=
  (R = 2 * B) ∧
  (W = B + R) ∧
  (S = 10) ∧
  (B + R + W + S = 40)

-- The theorem to prove
theorem blue_beads (B R W S : ℕ) (h : conditions B R W S) : B = 5 :=
by
  sorry

end blue_beads_l206_206595


namespace max_value_of_z_l206_206590

variable (x y z : ℝ)

def condition1 : Prop := 2 * x + y ≤ 4
def condition2 : Prop := x ≤ y
def condition3 : Prop := x ≥ 1 / 2
def objective_function : ℝ := 2 * x - y

theorem max_value_of_z :
  (∀ x y, condition1 x y ∧ condition2 x y ∧ condition3 x → z = objective_function x y) →
  z ≤ 4 / 3 :=
sorry

end max_value_of_z_l206_206590


namespace candies_in_box_more_than_pockets_l206_206276

theorem candies_in_box_more_than_pockets (x : ℕ) : 
  let initial_pockets := 2 * x
  let pockets_after_return := 2 * (x - 6)
  let candies_returned_to_box := 12
  let total_candies_after_return := initial_pockets + candies_returned_to_box
  (total_candies_after_return - pockets_after_return) = 24 :=
by
  sorry

end candies_in_box_more_than_pockets_l206_206276


namespace number_of_cirrus_clouds_l206_206879

def C_cb := 3
def C_cu := 12 * C_cb
def C_ci := 4 * C_cu

theorem number_of_cirrus_clouds : C_ci = 144 :=
by
  sorry

end number_of_cirrus_clouds_l206_206879


namespace largest_of_four_consecutive_integers_with_product_840_l206_206029

theorem largest_of_four_consecutive_integers_with_product_840 
  (a b c d : ℕ) (h1 : a + 1 = b) (h2 : b + 1 = c) (h3 : c + 1 = d) (h_pos : 0 < a) (h_prod : a * b * c * d = 840) : d = 7 :=
sorry

end largest_of_four_consecutive_integers_with_product_840_l206_206029


namespace total_feet_in_garden_l206_206515

theorem total_feet_in_garden (num_dogs num_ducks feet_per_dog feet_per_duck : ℕ)
  (h1 : num_dogs = 6) (h2 : num_ducks = 2)
  (h3 : feet_per_dog = 4) (h4 : feet_per_duck = 2) :
  num_dogs * feet_per_dog + num_ducks * feet_per_duck = 28 :=
by
  sorry

end total_feet_in_garden_l206_206515


namespace total_paintings_is_correct_l206_206757

-- Definitions for Philip's schedule and starting number of paintings
def philip_paintings_monday_and_tuesday := 3
def philip_paintings_wednesday := 2
def philip_paintings_thursday_and_friday := 5
def philip_initial_paintings := 20

-- Definitions for Amelia's schedule and starting number of paintings
def amelia_paintings_every_day := 2
def amelia_initial_paintings := 45

-- Calculation of total paintings after 5 weeks
def philip_weekly_paintings := 
  (2 * philip_paintings_monday_and_tuesday) + 
  philip_paintings_wednesday + 
  (2 * philip_paintings_thursday_and_friday)

def amelia_weekly_paintings := 
  7 * amelia_paintings_every_day

def total_paintings_after_5_weeks := 5 * philip_weekly_paintings + philip_initial_paintings + 5 * amelia_weekly_paintings + amelia_initial_paintings

-- Proof statement
theorem total_paintings_is_correct :
  total_paintings_after_5_weeks = 225 :=
  by sorry

end total_paintings_is_correct_l206_206757


namespace solve_system_1_solve_system_2_l206_206090

theorem solve_system_1 (x y : ℤ) (h1 : y = 2 * x - 3) (h2 : 3 * x + 2 * y = 8) : x = 2 ∧ y = 1 :=
by {
  sorry
}

theorem solve_system_2 (x y : ℤ) (h1 : 2 * x + 3 * y = 7) (h2 : 3 * x - 2 * y = 4) : x = 2 ∧ y = 1 :=
by {
  sorry
}

end solve_system_1_solve_system_2_l206_206090


namespace distance_between_A_and_B_l206_206981

-- Definitions and conditions
variables {A B C : Type}    -- Locations
variables {v1 v2 : ℕ}       -- Speeds of person A and person B
variables {distanceAB : ℕ}  -- Distance we want to find

noncomputable def first_meet_condition (v1 v2 : ℕ) : Prop :=
  ∃ t : ℕ, (v1 * t - 108 = v2 * t - 100)

noncomputable def second_meet_condition (v1 v2 distanceAB : ℕ) : Prop :=
  distanceAB = 3750

-- Theorem statement
theorem distance_between_A_and_B (v1 v2 distanceAB : ℕ) :
  first_meet_condition v1 v2 → second_meet_condition v1 v2 distanceAB →
  distanceAB = 3750 :=
by
  intros _ _ 
  sorry

end distance_between_A_and_B_l206_206981


namespace tan_of_angle_subtraction_l206_206814

theorem tan_of_angle_subtraction (a : ℝ) (h : Real.tan (a + Real.pi / 4) = 1 / 7) : Real.tan a = -3 / 4 :=
by
  sorry

end tan_of_angle_subtraction_l206_206814


namespace calculate_expression_l206_206545

theorem calculate_expression (x : ℝ) (h : x = 3) : (x^2 - 5 * x + 4) / (x - 4) = 2 :=
by
  rw [h]
  sorry

end calculate_expression_l206_206545


namespace bells_toll_together_l206_206224

theorem bells_toll_together {a b c d : ℕ} (h1 : a = 9) (h2 : b = 10) (h3 : c = 14) (h4 : d = 18) :
  Nat.lcm (Nat.lcm a b) (Nat.lcm c d) = 630 :=
by
  sorry

end bells_toll_together_l206_206224


namespace second_tap_empties_cistern_l206_206083

theorem second_tap_empties_cistern (t_fill: ℝ) (x: ℝ) (t_net: ℝ) : 
  (1 / 6) - (1 / x) = (1 / 12) → x = 12 := 
by
  sorry

end second_tap_empties_cistern_l206_206083


namespace find_x_l206_206770

noncomputable def log (b x : ℝ) : ℝ := Real.log x / Real.log b

theorem find_x (x : ℝ) (h₁ : log x 16 = log 4 256) : x = 2 := by
  sorry

end find_x_l206_206770


namespace find_number_l206_206035

theorem find_number (x : ℝ) :
  (10 + 30 + 50) / 3 = 30 →
  ((x + 40 + 6) / 3 = (10 + 30 + 50) / 3 - 8) →
  x = 20 :=
by
  intros h_avg1 h_avg2
  sorry

end find_number_l206_206035


namespace possible_values_of_a_l206_206173

def line1 (x y : ℝ) := x + y + 1 = 0
def line2 (x y : ℝ) := 2 * x - y + 8 = 0
def line3 (a : ℝ) (x y : ℝ) := a * x + 3 * y - 5 = 0

theorem possible_values_of_a :
  {a : ℝ | ∃ (x y : ℝ), line1 x y ∧ line2 x y ∧ line3 a x y} ⊆ {1/3, 3, -6} ∧
  {1/3, 3, -6} ⊆ {a : ℝ | ∃ (x y : ℝ), line1 x y ∧ line2 x y ∧ line3 a x y} :=
sorry

end possible_values_of_a_l206_206173


namespace remainder_of_16_pow_2048_mod_11_l206_206685

theorem remainder_of_16_pow_2048_mod_11 : (16^2048) % 11 = 4 := by
  sorry

end remainder_of_16_pow_2048_mod_11_l206_206685


namespace range_a_l206_206015

noncomputable def range_of_a (a : ℝ) : Prop :=
  ∃ x : ℝ, |x - a| + |x - 1| ≤ 3

theorem range_a (a : ℝ) : range_of_a a → -2 ≤ a ∧ a ≤ 4 :=
sorry

end range_a_l206_206015


namespace remainder_when_subtracted_l206_206675

theorem remainder_when_subtracted (s t : ℕ) (hs : s % 6 = 2) (ht : t % 6 = 3) (h : s > t) : (s - t) % 6 = 5 :=
by
  sorry -- Proof not required

end remainder_when_subtracted_l206_206675


namespace sams_weight_l206_206760

  theorem sams_weight (j s : ℝ) (h1 : j + s = 240) (h2 : s - j = j / 3) : s = 2880 / 21 :=
  by
    sorry
  
end sams_weight_l206_206760


namespace total_teaching_hours_l206_206346

-- Define the durations of the classes
def eduardo_math_classes : ℕ := 3
def eduardo_science_classes : ℕ := 4
def eduardo_history_classes : ℕ := 2

def math_class_duration : ℕ := 1
def science_class_duration : ℚ := 1.5
def history_class_duration : ℕ := 2

-- Define Eduardo's teaching time
def eduardo_total_time : ℚ :=
  eduardo_math_classes * math_class_duration +
  eduardo_science_classes * science_class_duration +
  eduardo_history_classes * history_class_duration

-- Define Frankie's teaching time (double the classes of Eduardo)
def frankie_total_time : ℚ :=
  2 * (eduardo_math_classes * math_class_duration) +
  2 * (eduardo_science_classes * science_class_duration) +
  2 * (eduardo_history_classes * history_class_duration)

-- Define the total teaching time for both Eduardo and Frankie
def total_teaching_time : ℚ :=
  eduardo_total_time + frankie_total_time

-- Theorem statement that both their total teaching time is 39 hours
theorem total_teaching_hours : total_teaching_time = 39 :=
by
  -- skipping the proof using sorry
  sorry

end total_teaching_hours_l206_206346


namespace part1_part2_l206_206666

open Real

noncomputable def f (x : ℝ) (a : ℝ) := |2 * x - 1| - |x - a|

theorem part1 (a : ℝ) (h : a = 0) :
  {x : ℝ | f x a < 1} = {x : ℝ | 0 < x ∧ x < 2} :=
by
  sorry

theorem part2 (a : ℝ) (h : ∀ x : ℝ, f x a < 1 → |(1 - 2 * a)^2 / 6| > 3 / 2) 
  : a < -1 :=
by
  sorry

end part1_part2_l206_206666


namespace probability_calculation_l206_206217

noncomputable def probability_in_ellipsoid : ℝ :=
  let prism_volume := (2 - (-2)) * (1 - (-1)) * (1 - (-1))
  let ellipsoid_volume := (4 * Real.pi / 3) * 1 * 2 * 2
  ellipsoid_volume / prism_volume

theorem probability_calculation :
  probability_in_ellipsoid = Real.pi / 3 :=
sorry

end probability_calculation_l206_206217


namespace price_reduction_for_2100_yuan_price_reduction_for_max_profit_l206_206558

-- Condition definitions based on the problem statement
def units_sold (x : ℝ) : ℝ := 30 + 2 * x
def profit_per_unit (x : ℝ) : ℝ := 50 - x
def daily_profit (x : ℝ) : ℝ := profit_per_unit x * units_sold x

-- Statement to prove the price reduction for achieving a daily profit of 2100 yuan
theorem price_reduction_for_2100_yuan : ∃ x : ℝ, daily_profit x = 2100 ∧ x = 20 :=
  sorry

-- Statement to prove the price reduction to maximize the daily profit
theorem price_reduction_for_max_profit : ∀ x : ℝ, ∃ y : ℝ, (∀ z : ℝ, daily_profit z ≤ y) ∧ x = 17.5 :=
  sorry

end price_reduction_for_2100_yuan_price_reduction_for_max_profit_l206_206558


namespace intersection_M_N_eq_l206_206365

def M : Set ℝ := {x | (x + 3) * (x - 2) < 0}
def N : Set ℝ := {x | 1 ≤ x ∧ x ≤ 3}

theorem intersection_M_N_eq : M ∩ N = {x | 1 ≤ x ∧ x < 2} := by
  sorry

end intersection_M_N_eq_l206_206365


namespace find_k_l206_206170

theorem find_k (m n k : ℤ) (h1 : m = 2 * n + 5) (h2 : m + 2 = 2 * (n + k) + 5) : k = 0 := by
  sorry

end find_k_l206_206170


namespace cos_330_eq_sqrt3_div_2_l206_206216

theorem cos_330_eq_sqrt3_div_2
    (h1 : ∀ θ : ℝ, Real.cos (2 * Real.pi - θ) = Real.cos θ)
    (h2 : Real.cos (Real.pi / 6) = Real.sqrt 3 / 2) :
    Real.cos (11 * Real.pi / 6) = Real.sqrt 3 / 2 :=
by
  -- Proof goes here
  sorry

end cos_330_eq_sqrt3_div_2_l206_206216


namespace candy_count_l206_206152

theorem candy_count (S : ℕ) (H1 : 32 + S - 35 = 39) : S = 42 :=
by
  sorry

end candy_count_l206_206152


namespace larger_number_l206_206302

theorem larger_number (a b : ℤ) (h1 : a - b = 5) (h2 : a + b = 37) : a = 21 :=
sorry

end larger_number_l206_206302


namespace megatek_manufacturing_percentage_l206_206417

theorem megatek_manufacturing_percentage (angle_manufacturing : ℝ) (full_circle : ℝ) 
  (h1 : angle_manufacturing = 162) (h2 : full_circle = 360) :
  (angle_manufacturing / full_circle) * 100 = 45 :=
by
  sorry

end megatek_manufacturing_percentage_l206_206417


namespace sum_of_first_10_terms_l206_206435

def general_term (n : ℕ) : ℕ := 2 * n + 1

def sequence_sum (n : ℕ) : ℕ := n / 2 * (general_term 1 + general_term n)

theorem sum_of_first_10_terms : sequence_sum 10 = 120 := by
  sorry

end sum_of_first_10_terms_l206_206435


namespace sequence_eventually_periodic_l206_206081

open Nat

noncomputable def sum_prime_factors_plus_one (K : ℕ) : ℕ := 
  (K.factors.sum) + 1

theorem sequence_eventually_periodic (K : ℕ) (hK : K ≥ 9) :
  ∃ m n : ℕ, m ≠ n ∧ sum_prime_factors_plus_one^[m] K = sum_prime_factors_plus_one^[n] K := 
sorry

end sequence_eventually_periodic_l206_206081


namespace calculation_correct_l206_206021

theorem calculation_correct :
  15 * ( (1/3 : ℚ) + (1/4) + (1/6) )⁻¹ = 20 := sorry

end calculation_correct_l206_206021


namespace width_of_shop_l206_206383

theorem width_of_shop 
  (monthly_rent : ℕ) 
  (length : ℕ) 
  (annual_rent_per_sqft : ℕ) 
  (h1 : monthly_rent = 3600) 
  (h2 : length = 18) 
  (h3 : annual_rent_per_sqft = 120) :
  ∃ width : ℕ, width = 20 :=
by
  sorry

end width_of_shop_l206_206383


namespace quadratic_roots_condition_l206_206102

theorem quadratic_roots_condition (a : ℝ) :
  (∃ α : ℝ, 5 * α = -(a - 4) ∧ 4 * α^2 = a - 5) ↔ (a = 7 ∨ a = 5) :=
by
  sorry

end quadratic_roots_condition_l206_206102


namespace line_intersects_y_axis_at_l206_206073

-- Define the two points the line passes through
structure Point (α : Type) :=
(x : α)
(y : α)

def p1 : Point ℤ := Point.mk 2 9
def p2 : Point ℤ := Point.mk 4 13

-- Define the function that describes the point where the line intersects the y-axis
def y_intercept : Point ℤ :=
  -- We are proving that the line intersects the y-axis at the point (0, 5)
  Point.mk 0 5

-- State the theorem to be proven
theorem line_intersects_y_axis_at (p1 p2 : Point ℤ) (yi : Point ℤ) :
  p1.x = 2 ∧ p1.y = 9 ∧ p2.x = 4 ∧ p2.y = 13 → yi = Point.mk 0 5 :=
by
  intros
  sorry

end line_intersects_y_axis_at_l206_206073


namespace number_mul_five_l206_206220

theorem number_mul_five (x : ℤ) (h : x - 7 = 9) : x * 5 = 80 :=
by
  sorry

end number_mul_five_l206_206220


namespace total_amount_after_5_months_l206_206426

-- Definitions from the conditions
def initial_deposit : ℝ := 100
def monthly_interest_rate : ℝ := 0.0036  -- 0.36% expressed as a decimal

-- Definition of the function relationship y with respect to x
def total_amount (x : ℕ) : ℝ := initial_deposit + initial_deposit * monthly_interest_rate * x

-- Prove the total amount after 5 months is 101.8
theorem total_amount_after_5_months : total_amount 5 = 101.8 :=
by
  sorry

end total_amount_after_5_months_l206_206426


namespace number_of_possible_scenarios_l206_206284

theorem number_of_possible_scenarios 
  (subjects : ℕ) 
  (students : ℕ) 
  (h_subjects : subjects = 4) 
  (h_students : students = 3) : 
  (subjects ^ students) = 64 := 
by
  -- Provide proof here
  sorry

end number_of_possible_scenarios_l206_206284


namespace measure_of_angleA_l206_206382

theorem measure_of_angleA (A B : ℝ) 
  (h1 : ∀ (x : ℝ), x ≠ A → x ≠ B → x ≠ (3 * B - 20) → (3 * x - 20 ≠ A)) 
  (h2 : A = 3 * B - 20) :
  A = 10 ∨ A = 130 :=
by
  sorry

end measure_of_angleA_l206_206382


namespace sum_a_16_to_20_l206_206673

variable (a : ℕ → ℝ) (S : ℕ → ℝ)

-- Conditions
axiom S_def : ∀ n, S n = a 0 * (1 - (a 1 / a 0) ^ n) / (1 - (a 1 / a 0))
axiom S_5_eq_2 : S 5 = 2
axiom S_10_eq_6 : S 10 = 6

-- Theorem to prove
theorem sum_a_16_to_20 : a 16 + a 17 + a 18 + a 19 + a 20 = 16 :=
by
  sorry

end sum_a_16_to_20_l206_206673


namespace ratio_of_goals_l206_206050

-- The conditions
def first_period_goals_kickers : ℕ := 2
def second_period_goals_kickers := 4
def first_period_goals_spiders := first_period_goals_kickers / 2
def second_period_goals_spiders := 2 * second_period_goals_kickers
def total_goals := first_period_goals_kickers + second_period_goals_kickers + first_period_goals_spiders + second_period_goals_spiders

-- The ratio to prove
def ratio_goals : ℕ := second_period_goals_kickers / first_period_goals_kickers

theorem ratio_of_goals : total_goals = 15 → ratio_goals = 2 := by
  intro h
  sorry

end ratio_of_goals_l206_206050


namespace vincent_spent_224_l206_206174

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

end vincent_spent_224_l206_206174


namespace f_1991_eq_1988_l206_206269

def f (n : ℕ) : ℕ := sorry

theorem f_1991_eq_1988 : f 1991 = 1988 :=
by sorry

end f_1991_eq_1988_l206_206269


namespace sum_denominators_l206_206048

theorem sum_denominators (a b: ℕ) (h_coprime : Nat.gcd a b = 1) :
  (3:ℚ) / (5 * b) + (2:ℚ) / (9 * b) + (4:ℚ) / (15 * b) = 28 / 45 →
  5 * b + 9 * b + 15 * b = 203 :=
by
  sorry

end sum_denominators_l206_206048


namespace x1_x2_lt_one_l206_206314

open Real

noncomputable def f (x : ℝ) (a : ℝ) : ℝ := a * x - log x
noncomputable def g (x : ℝ) : ℝ := x / exp x

theorem x1_x2_lt_one (k : ℝ) (x1 x2 : ℝ) (h : f x1 1 + g x1 - k = 0) (h2 : f x2 1 + g x2 - k = 0) (hx1 : 0 < x1) (hx2 : x1 < x2) : x1 * x2 < 1 :=
by
  sorry

end x1_x2_lt_one_l206_206314


namespace find_certain_number_l206_206460

theorem find_certain_number (N : ℝ) 
  (h : 3.6 * N * 2.50 / (0.12 * 0.09 * 0.5) = 800.0000000000001)
  : N = 0.48 :=
sorry

end find_certain_number_l206_206460


namespace percentage_failed_in_Hindi_l206_206236

-- Let Hindi_failed denote the percentage of students who failed in Hindi.
-- Let English_failed denote the percentage of students who failed in English.
-- Let Both_failed denote the percentage of students who failed in both Hindi and English.
-- Let Both_passed denote the percentage of students who passed in both subjects.

variables (Hindi_failed English_failed Both_failed Both_passed : ℝ)
  (H_condition1 : English_failed = 44)
  (H_condition2 : Both_failed = 22)
  (H_condition3 : Both_passed = 44)

theorem percentage_failed_in_Hindi:
  Hindi_failed = 34 :=
by 
  -- Proof goes here
  sorry

end percentage_failed_in_Hindi_l206_206236


namespace stephanie_falls_l206_206453

theorem stephanie_falls 
  (steven_falls : ℕ := 3)
  (sonya_falls : ℕ := 6)
  (h1 : sonya_falls = 6)
  (h2 : ∃ S : ℕ, sonya_falls = (S / 2) - 2 ∧ S > steven_falls) :
  ∃ S : ℕ, S - steven_falls = 13 :=
by
  sorry

end stephanie_falls_l206_206453


namespace integer_solution_unique_l206_206924

variable (x y : ℤ)

def nested_sqrt_1964_times (x : ℤ) : ℤ := 
  sorry -- (This should define the function for nested sqrt 1964 times, but we'll use sorry to skip the proof)

theorem integer_solution_unique : 
  nested_sqrt_1964_times x = y → x = 0 ∧ y = 0 :=
by
  intros h
  sorry -- Proof of the theorem goes here

end integer_solution_unique_l206_206924


namespace unique_solution_l206_206162

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

end unique_solution_l206_206162


namespace nat_prime_p_and_5p_plus_1_is_prime_l206_206436

theorem nat_prime_p_and_5p_plus_1_is_prime (p : ℕ) (hp : Nat.Prime p) (h5p1 : Nat.Prime (5 * p + 1)) : p = 2 := 
by 
  -- Sorry is added to skip the proof
  sorry 

end nat_prime_p_and_5p_plus_1_is_prime_l206_206436


namespace average_after_adding_ten_l206_206570

theorem average_after_adding_ten (avg initial_sum new_mean : ℕ) (n : ℕ) (h1 : n = 15) (h2 : avg = 40) (h3 : initial_sum = n * avg) (h4 : new_mean = (initial_sum + n * 10) / n) : new_mean = 50 := 
by
  sorry

end average_after_adding_ten_l206_206570


namespace ellipse_semi_minor_axis_is_2_sqrt_3_l206_206645

/-- 
  Given an ellipse with the center at (2, -1), 
  one focus at (2, -3), and one endpoint of a semi-major axis at (2, 3), 
  we prove that the semi-minor axis is 2√3.
-/
theorem ellipse_semi_minor_axis_is_2_sqrt_3 :
  let center := (2, -1)
  let focus := (2, -3)
  let endpoint := (2, 3)
  let c := Real.sqrt ((2 - 2)^2 + (-3 + 1)^2)
  let a := Real.sqrt ((2 - 2)^2 + (3 + 1)^2)
  let b2 := a^2 - c^2
  let b := Real.sqrt b2
  c = 2 ∧ a = 4 ∧ b = 2 * Real.sqrt 3 := 
by
  sorry

end ellipse_semi_minor_axis_is_2_sqrt_3_l206_206645


namespace graph_union_l206_206686

-- Definitions of the conditions from part a)
def graph1 (z y : ℝ) : Prop := z^4 - 6 * y^4 = 3 * z^2 - 2

def graph_hyperbola (z y : ℝ) : Prop := z^2 - 3 * y^2 = 2

def graph_ellipse (z y : ℝ) : Prop := z^2 - 2 * y^2 = 1

-- Lean statement to prove the question is equivalent to the answer
theorem graph_union (z y : ℝ) : graph1 z y ↔ (graph_hyperbola z y ∨ graph_ellipse z y) := 
sorry

end graph_union_l206_206686


namespace periodic_odd_function_l206_206662

theorem periodic_odd_function (f : ℝ → ℝ) (period : ℝ) (h_periodic : ∀ x, f (x + period) = f x) (h_odd : ∀ x, f (-x) = -f x) (h_value : f (-3) = 1) (α : ℝ) (h_tan : Real.tan α = 2) :
  f (20 * Real.sin α * Real.cos α) = -1 := 
sorry

end periodic_odd_function_l206_206662


namespace trig_identity_l206_206628

theorem trig_identity (α : ℝ) : 
  (2 * (Real.sin (4 * α))^2 - 1) / 
  (2 * (1 / Real.tan (Real.pi / 4 + 4 * α)) * (Real.cos (5 * Real.pi / 4 - 4 * α))^2) = -1 :=
by
  sorry

end trig_identity_l206_206628


namespace alex_age_div_M_l206_206866

variable {A M : ℕ}

-- Definitions provided by the conditions
def alex_age_current : ℕ := A
def sum_children_age : ℕ := A
def alex_age_M_years_ago (A M : ℕ) : ℕ := A - M
def children_age_M_years_ago (A M : ℕ) : ℕ := A - 4 * M

-- Given condition as a hypothesis
def condition (A M : ℕ) := alex_age_M_years_ago A M = 3 * children_age_M_years_ago A M

-- The theorem to prove
theorem alex_age_div_M (A M : ℕ) (h : condition A M) : A / M = 11 / 2 := 
by
  -- This is a placeholder for the actual proof.
  sorry

end alex_age_div_M_l206_206866


namespace tom_sold_4_books_l206_206992

-- Definitions based on conditions from the problem
def initial_books : ℕ := 5
def new_books : ℕ := 38
def final_books : ℕ := 39

-- The number of books Tom sold
def books_sold (S : ℕ) : Prop := initial_books - S + new_books = final_books

-- Our goal is to prove that Tom sold 4 books
theorem tom_sold_4_books : books_sold 4 :=
  by
    -- Implicitly here would be the proof, but we use sorry to skip it
    sorry

end tom_sold_4_books_l206_206992


namespace chicken_feathers_after_crossing_l206_206495

def cars_dodged : ℕ := 23
def initial_feathers : ℕ := 5263
def feathers_lost : ℕ := 2 * cars_dodged
def final_feathers : ℕ := initial_feathers - feathers_lost

theorem chicken_feathers_after_crossing :
  final_feathers = 5217 := by
sorry

end chicken_feathers_after_crossing_l206_206495


namespace candy_in_one_bowl_l206_206127

theorem candy_in_one_bowl (total_candies : ℕ) (eaten_candies : ℕ) (bowls : ℕ) (taken_per_bowl : ℕ) 
  (h1 : total_candies = 100) (h2 : eaten_candies = 8) (h3 : bowls = 4) (h4 : taken_per_bowl = 3) :
  (total_candies - eaten_candies) / bowls - taken_per_bowl = 20 :=
by
  sorry

end candy_in_one_bowl_l206_206127


namespace fred_balloon_count_l206_206709

variable (Fred_balloons Sam_balloons Mary_balloons total_balloons : ℕ)

/-- 
  Given:
  - Fred has some yellow balloons
  - Sam has 6 yellow balloons
  - Mary has 7 yellow balloons
  - Total number of yellow balloons (Fred's, Sam's, and Mary's balloons) is 18

  Prove: Fred has 5 yellow balloons.
-/
theorem fred_balloon_count :
  Sam_balloons = 6 →
  Mary_balloons = 7 →
  total_balloons = 18 →
  Fred_balloons = total_balloons - (Sam_balloons + Mary_balloons) →
  Fred_balloons = 5 :=
by
  sorry

end fred_balloon_count_l206_206709


namespace sum_of_eighth_powers_of_roots_l206_206072

noncomputable def quadratic_roots (a b c : ℝ) : ℝ × ℝ :=
  let discriminant := b^2 - 4 * a * c
  let root_disc := Real.sqrt discriminant
  ((-b + root_disc) / (2 * a), (-b - root_disc) / (2 * a))

theorem sum_of_eighth_powers_of_roots :
  let (p, q) := quadratic_roots 1 (-Real.sqrt 7) 1
  p^2 + q^2 = 5 ∧ p^4 + q^4 = 23 ∧ p^8 + q^8 = 527 :=
by
  sorry

end sum_of_eighth_powers_of_roots_l206_206072


namespace g_at_52_l206_206027

noncomputable def g : ℝ → ℝ := sorry

axiom g_multiplicative : ∀ (x y: ℝ), g (x * y) = y * g x
axiom g_at_1 : g 1 = 10

theorem g_at_52 : g 52 = 520 := sorry

end g_at_52_l206_206027


namespace alpha_lt_beta_of_acute_l206_206964

open Real

theorem alpha_lt_beta_of_acute (α β : ℝ) (hα : 0 < α ∧ α < π / 2) (hβ : 0 < β ∧ β < π / 2)
  (h : 2 * sin α = sin α * cos β + cos α * sin β) : α < β :=
by
  sorry

end alpha_lt_beta_of_acute_l206_206964


namespace reduced_price_per_kg_l206_206272

-- Definitions
variables {P R Q : ℝ}

-- Conditions
axiom reduction_price : R = P * 0.82
axiom original_quantity : Q * P = 1080
axiom reduced_quantity : (Q + 8) * R = 1080

-- Proof statement
theorem reduced_price_per_kg : R = 24.30 :=
by {
  sorry
}

end reduced_price_per_kg_l206_206272


namespace flyers_left_l206_206103

theorem flyers_left (total_flyers : ℕ) (jack_flyers : ℕ) (rose_flyers : ℕ) (h1 : total_flyers = 1236) (h2 : jack_flyers = 120) (h3 : rose_flyers = 320) : (total_flyers - (jack_flyers + rose_flyers) = 796) := 
by
  sorry

end flyers_left_l206_206103


namespace lifespan_of_bat_l206_206937

variable (B H F T : ℝ)

theorem lifespan_of_bat (h₁ : H = B - 6)
                        (h₂ : F = 4 * H)
                        (h₃ : T = 2 * B)
                        (h₄ : B + H + F + T = 62) :
  B = 11.5 :=
by
  sorry

end lifespan_of_bat_l206_206937


namespace a_3_eq_5_l206_206496

variable (a : ℕ → ℕ) -- Defines the arithmetic sequence
variable (S : ℕ → ℕ) -- The sum of the first n terms of the sequence

-- Condition: S_5 = 25
axiom S_5_eq_25 : S 5 = 25

-- Define what it means for S to be the sum of the first n terms of the arithmetic sequence
axiom sum_arith_seq : ∀ n, S n = n * (a 1 + a n) / 2

theorem a_3_eq_5 : a 3 = 5 :=
by
  -- Proof is skipped using sorry
  sorry

end a_3_eq_5_l206_206496


namespace painting_area_l206_206005

def wall_height : ℝ := 10
def wall_length : ℝ := 15
def door_height : ℝ := 3
def door_length : ℝ := 5

noncomputable def area_of_wall : ℝ :=
  wall_height * wall_length

noncomputable def area_of_door : ℝ :=
  door_height * door_length

noncomputable def area_to_paint : ℝ :=
  area_of_wall - area_of_door

theorem painting_area :
  area_to_paint = 135 := by
  sorry

end painting_area_l206_206005


namespace finite_set_cardinality_l206_206887

-- Define the main theorem statement
theorem finite_set_cardinality (m : ℕ) (A : Finset ℤ) (B : ℕ → Finset ℤ)
  (hm : m ≥ 2)
  (hB : ∀ k : ℕ, k ∈ Finset.range m.succ → (B k).sum id = m^k) :
  A.card ≥ m / 2 := 
sorry

end finite_set_cardinality_l206_206887


namespace find_missing_dimension_l206_206105

-- Definitions based on conditions
def is_dimension_greatest_area (x : ℝ) : Prop :=
  max (2 * x) (max (3 * x) 6) = 15

-- The final statement to prove
theorem find_missing_dimension (x : ℝ) (h1 : is_dimension_greatest_area x) : x = 5 :=
sorry

end find_missing_dimension_l206_206105


namespace sum_of_digits_l206_206588

-- Conditions setup
variables (a b c d : ℕ)
variables (h1 : a + c = 10) 
variables (h2 : b + c = 9) 
variables (h3 : a + d = 10)
variables (distinct : a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d)

theorem sum_of_digits : a + b + c + d = 19 :=
sorry

end sum_of_digits_l206_206588


namespace solution_set_of_inequality_system_l206_206891

theorem solution_set_of_inequality_system (x : ℝ) : (x - 1 < 0) ∧ (x + 1 > 0) ↔ (-1 < x ∧ x < 1) :=
by
  -- proof to be filled in
  sorry

end solution_set_of_inequality_system_l206_206891


namespace time_juan_ran_l206_206233

variable (Distance Speed : ℝ)
variable (h1 : Distance = 80)
variable (h2 : Speed = 10)

theorem time_juan_ran : (Distance / Speed) = 8 := by
  sorry

end time_juan_ran_l206_206233


namespace domain_transformation_l206_206024

-- Definitions of conditions
def domain_f (x : ℝ) : Prop := 0 ≤ x ∧ x ≤ 4

def domain_g (x : ℝ) : Prop := 1 < x ∧ x ≤ 3

-- Theorem stating the proof problem
theorem domain_transformation : 
  (∀ x, domain_f x → 0 ≤ x+1 ∧ x+1 ≤ 4) →
  (∀ x, (0 ≤ x+1 ∧ x+1 ≤ 4) → (x-1 > 0) → domain_g x) :=
by
  intros h1 x hx
  sorry

end domain_transformation_l206_206024


namespace cube_colorings_distinguishable_l206_206059

-- Define the problem
def cube_construction_distinguishable_ways : Nat :=
  30

-- The theorem we need to prove
theorem cube_colorings_distinguishable :
  ∃ (ways : Nat), ways = cube_construction_distinguishable_ways :=
by
  sorry

end cube_colorings_distinguishable_l206_206059


namespace tax_liability_difference_l206_206759

theorem tax_liability_difference : 
  let annual_income := 150000
  let old_tax_rate := 0.45
  let new_tax_rate_1 := 0.30
  let new_tax_rate_2 := 0.35
  let new_tax_rate_3 := 0.40
  let mortgage_interest := 10000
  let old_tax_liability := annual_income * old_tax_rate
  let taxable_income_new := annual_income - mortgage_interest
  let new_tax_liability := 
    if taxable_income_new <= 50000 then 
      taxable_income_new * new_tax_rate_1
    else if taxable_income_new <= 100000 then 
      50000 * new_tax_rate_1 + (taxable_income_new - 50000) * new_tax_rate_2
    else 
      50000 * new_tax_rate_1 + 50000 * new_tax_rate_2 + (taxable_income_new - 100000) * new_tax_rate_3
  let tax_liability_difference := old_tax_liability - new_tax_liability
  tax_liability_difference = 19000 := 
by
  sorry

end tax_liability_difference_l206_206759


namespace option_b_correct_l206_206627

theorem option_b_correct (a b c : ℝ) (hc : c ≠ 0) (h : a * c^2 > b * c^2) : a > b :=
sorry

end option_b_correct_l206_206627


namespace compute_star_l206_206158

def star (x y : ℕ) := 4 * x + 6 * y

theorem compute_star : star 3 4 = 36 := 
by
  sorry

end compute_star_l206_206158


namespace simplify_fraction_expression_l206_206548

variable (d : ℝ)

theorem simplify_fraction_expression :
  (6 + 5 * d) / 11 + 3 = (39 + 5 * d) / 11 := 
by
  sorry

end simplify_fraction_expression_l206_206548


namespace base_subtraction_problem_l206_206126

theorem base_subtraction_problem (b : ℕ) (C_b : ℕ) (hC : C_b = 12) : 
  b = 15 :=
by
  sorry

end base_subtraction_problem_l206_206126


namespace num_customers_after_family_l206_206865

-- Definitions
def soft_taco_price : ℕ := 2
def hard_taco_price : ℕ := 5
def family_hard_tacos : ℕ := 4
def family_soft_tacos : ℕ := 3
def total_income : ℕ := 66

-- Intermediate values which can be derived
def family_cost : ℕ := (family_hard_tacos * hard_taco_price) + (family_soft_tacos * soft_taco_price)
def remaining_income : ℕ := total_income - family_cost

-- Proposition: Number of customers after the family
def customers_after_family : ℕ := remaining_income / (2 * soft_taco_price)

-- Theorem to prove the number of customers is 10
theorem num_customers_after_family : customers_after_family = 10 := by
  sorry

end num_customers_after_family_l206_206865


namespace area_of_rectangle_l206_206941

theorem area_of_rectangle (x y : ℝ)
    (h1 : x * y = (x + 3) * (y - 1))
    (h2 : x * y = (x - 4) * (y + 3 / 2)) :
    x * y = 108 := by
  sorry

end area_of_rectangle_l206_206941


namespace quadratic_inequality_solution_l206_206303

theorem quadratic_inequality_solution (a m : ℝ) (h : a < 0) :
  (∀ x : ℝ, ax^2 + 6*x - a^2 < 0 ↔ (x < 1 ∨ x > m)) → m = 2 :=
by
  sorry

end quadratic_inequality_solution_l206_206303


namespace intersection_of_sets_l206_206956

def A : Set ℝ := { x | x < 2 }
def B : Set ℝ := { y | ∃ x : ℝ, y = 2^x - 1 }
def C : Set ℝ := { m | -1 < m ∧ m < 2 }

theorem intersection_of_sets : A ∩ B = C := 
by sorry

end intersection_of_sets_l206_206956


namespace seven_solutions_l206_206801

theorem seven_solutions: ∃ (pairs : List (ℕ × ℕ)), 
  (∀ (x y : ℕ), (x < y) → ((1 / (x : ℝ)) + (1 / (y : ℝ)) = 1 / 2007) ↔ (x, y) ∈ pairs) 
  ∧ pairs.length = 7 :=
sorry

end seven_solutions_l206_206801


namespace initial_incorrect_average_l206_206124

theorem initial_incorrect_average (S_correct S_wrong : ℝ) :
  (S_correct = S_wrong - 26 + 36) →
  (S_correct / 10 = 19) →
  (S_wrong / 10 = 18) :=
by
  sorry

end initial_incorrect_average_l206_206124


namespace moles_CH3COOH_equiv_l206_206345

theorem moles_CH3COOH_equiv (moles_NaOH moles_NaCH3COO : ℕ)
    (h1 : moles_NaOH = 1)
    (h2 : moles_NaCH3COO = 1) :
    moles_NaOH = moles_NaCH3COO :=
by
  sorry

end moles_CH3COOH_equiv_l206_206345


namespace average_temperature_l206_206195

def temperatures :=
  ∃ T_tue T_wed T_thu : ℝ,
    (44 + T_tue + T_wed + T_thu) / 4 = 48 ∧
    (T_tue + T_wed + T_thu + 36) / 4 = 46

theorem average_temperature :
  temperatures :=
by
  sorry

end average_temperature_l206_206195


namespace solve_for_x_l206_206129

theorem solve_for_x (x y : ℝ) (h1 : y = 1 / (5 * x + 2)) (h2 : y = 2) : x = -3 / 10 :=
by
  sorry

end solve_for_x_l206_206129


namespace flowchart_output_value_l206_206265

theorem flowchart_output_value :
  ∃ n : ℕ, S = n * (n + 1) / 2 ∧ n = 10 → S = 55 :=
by
  sorry

end flowchart_output_value_l206_206265


namespace father_current_age_l206_206117

variable (M F : ℕ)

/-- The man's current age is (2 / 5) of the age of his father. -/
axiom man_age : M = (2 / 5) * F

/-- After 12 years, the man's age will be (1 / 2) of his father's age. -/
axiom age_relation_in_12_years : (M + 12) = (1 / 2) * (F + 12)

/-- Prove that the father's current age, F, is 60. -/
theorem father_current_age : F = 60 :=
by
  sorry

end father_current_age_l206_206117


namespace eight_bees_have_48_legs_l206_206670

  def legs_per_bee : ℕ := 6
  def number_of_bees : ℕ := 8
  def total_legs : ℕ := 48

  theorem eight_bees_have_48_legs :
    number_of_bees * legs_per_bee = total_legs :=
  by
    sorry
  
end eight_bees_have_48_legs_l206_206670


namespace perp_vectors_dot_product_eq_zero_l206_206148

noncomputable def vector_a : ℝ × ℝ := (1, 2)
noncomputable def vector_b (x : ℝ) : ℝ × ℝ := (x, 4)

def dot_product (v1 v2 : ℝ × ℝ) : ℝ :=
  v1.1 * v2.1 + v1.2 * v2.2

theorem perp_vectors_dot_product_eq_zero (x : ℝ) (h : dot_product vector_a (vector_b x) = 0) : x = -8 :=
  by sorry

end perp_vectors_dot_product_eq_zero_l206_206148


namespace problem_statement_l206_206407

theorem problem_statement (a b c x y z : ℂ)
  (h1 : a = (b + c) / (x - 2))
  (h2 : b = (c + a) / (y - 2))
  (h3 : c = (a + b) / (z - 2))
  (h4 : x * y + y * z + z * x = 67)
  (h5 : x + y + z = 2010) :
  x * y * z = -5892 :=
by {
  sorry
}

end problem_statement_l206_206407


namespace relationship_a_b_c_l206_206746

noncomputable def distinct_positive_numbers (a b c : ℝ) : Prop := 
  a > 0 ∧ b > 0 ∧ c > 0 ∧ a ≠ b ∧ b ≠ c ∧ a ≠ c

theorem relationship_a_b_c (a b c : ℝ) (h1 : distinct_positive_numbers a b c) (h2 : a^2 + c^2 = 2 * b * c) : b > a ∧ a > c :=
by
  sorry

end relationship_a_b_c_l206_206746


namespace third_smallest_triangular_square_l206_206586

theorem third_smallest_triangular_square :
  ∃ n : ℕ, n = 1225 ∧ 
           (∃ x y : ℕ, y^2 - 8 * x^2 = 1 ∧ 
                        y = 99 ∧ x = 35) :=
by
  sorry

end third_smallest_triangular_square_l206_206586


namespace acrobat_count_l206_206253

theorem acrobat_count (a e c : ℕ) (h1 : 2 * a + 4 * e + 2 * c = 88) (h2 : a + e + c = 30) : a = 2 :=
by
  sorry

end acrobat_count_l206_206253


namespace cos_sequence_next_coeff_sum_eq_28_l206_206902

theorem cos_sequence_next_coeff_sum_eq_28 (α : ℝ) :
  let u := 2 * Real.cos α
  2 * Real.cos (8 * α) = u ^ 8 - 8 * u ^ 6 + 20 * u ^ 4 - 16 * u ^ 2 + 2 → 
  8 + (-8) + 6 + 20 + 2 = 28 :=
by intros u; sorry

end cos_sequence_next_coeff_sum_eq_28_l206_206902


namespace problem_f_prime_at_zero_l206_206120

noncomputable def f (x : ℝ) : ℝ := x * (x + 1) * (x + 2) * (x + 3) * (x + 4) * (x + 5) + 6

theorem problem_f_prime_at_zero : deriv f 0 = 120 :=
by
  -- Proof omitted
  sorry

end problem_f_prime_at_zero_l206_206120


namespace Matt_buys_10_key_chains_l206_206753

theorem Matt_buys_10_key_chains
  (cost_per_keychain_in_pack_of_10 : ℝ)
  (cost_per_keychain_in_pack_of_4 : ℝ)
  (number_of_keychains : ℝ)
  (savings : ℝ)
  (h1 : cost_per_keychain_in_pack_of_10 = 2)
  (h2 : cost_per_keychain_in_pack_of_4 = 3)
  (h3 : savings = 20)
  (h4 : 3 * number_of_keychains - 2 * number_of_keychains = savings) :
  number_of_keychains = 10 := 
by
  sorry

end Matt_buys_10_key_chains_l206_206753


namespace geometric_sequence_sum_l206_206683

variable {a b : ℝ} -- Parameters for real numbers a and b
variable (a_ne_zero : a ≠ 0) -- condition a ≠ 0

/-- Proof that in the geometric sequence {a_n}, given a_5 + a_6 = a and a_15 + a_16 = b, 
    a_25 + a_26 = b^2 / a --/
theorem geometric_sequence_sum (a5_plus_a6 : ℕ → ℝ) (a15_plus_a16 : ℕ → ℝ) (a25_plus_a26 : ℕ → ℝ)
  (h1 : a5_plus_a6 5 + a5_plus_a6 6 = a)
  (h2 : a15_plus_a16 15 + a15_plus_a16 16 = b) :
  a25_plus_a26 25 + a25_plus_a26 26 = b^2 / a :=
  sorry

end geometric_sequence_sum_l206_206683


namespace tangent_line_eq_l206_206350

noncomputable def f (x : ℝ) : ℝ := x^2 - 2*x - 3

def M : ℝ×ℝ := (2, -3)

theorem tangent_line_eq (x y : ℝ) (h : y = f x) (h' : (x, y) = M) :
  2 * x - y - 7 = 0 :=
sorry

end tangent_line_eq_l206_206350


namespace greene_family_amusement_park_spending_l206_206156

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

end greene_family_amusement_park_spending_l206_206156


namespace basketball_surface_area_l206_206396

theorem basketball_surface_area (C : ℝ) (r : ℝ) (A : ℝ) (π : ℝ) 
  (h1 : C = 30) 
  (h2 : C = 2 * π * r) 
  (h3 : A = 4 * π * r^2) 
  : A = 900 / π := by
  sorry

end basketball_surface_area_l206_206396


namespace negation_proposition_p_l206_206849

theorem negation_proposition_p (x y : ℝ) : (¬ ((x - 1) ^ 2 + (y - 2) ^ 2 = 0) → (x ≠ 1 ∨ y ≠ 2)) :=
by
  sorry

end negation_proposition_p_l206_206849


namespace sides_of_polygon_l206_206934

-- Define the conditions
def polygon_sum_interior_angles (n : ℕ) : ℝ := (n - 2) * 180

-- State the mathematically equivalent proof problem
theorem sides_of_polygon (n : ℕ) (h : polygon_sum_interior_angles n = 1260) : n = 9 := by
  sorry

end sides_of_polygon_l206_206934


namespace calculate_expression_l206_206646

variables (a b : ℝ)

theorem calculate_expression : -a^2 * 2 * a^4 * b = -2 * (a^6) * b :=
by
  sorry

end calculate_expression_l206_206646


namespace remainder_division_l206_206149

theorem remainder_division
  (j : ℕ) (h_pos : 0 < j)
  (h_rem : ∃ b : ℕ, 72 = b * j^2 + 8) :
  150 % j = 6 :=
sorry

end remainder_division_l206_206149


namespace find_f_neg2_l206_206289

noncomputable def f (x : ℝ) : ℝ :=
if x ≥ 0 then 3^x - 1 else -3^(-x) + 1

theorem find_f_neg2 : f (-2) = -8 := by
  sorry

end find_f_neg2_l206_206289


namespace remainder_of_8x_minus_5_l206_206527

theorem remainder_of_8x_minus_5 (x : ℕ) (h : x % 15 = 7) : (8 * x - 5) % 15 = 6 :=
by
  sorry

end remainder_of_8x_minus_5_l206_206527


namespace smallest_number_of_eggs_l206_206799

theorem smallest_number_of_eggs (c : ℕ) (h1 : 15 * c - 3 > 100) : 102 ≤ 15 * c - 3 :=
by
  sorry

end smallest_number_of_eggs_l206_206799


namespace simplify_expression_l206_206309

theorem simplify_expression (x : ℝ) : (3 * x + 8) + (50 * x + 25) = 53 * x + 33 := 
by sorry

end simplify_expression_l206_206309


namespace veranda_area_l206_206051

theorem veranda_area (room_length room_width veranda_length_width veranda_width_width : ℝ)
  (h1 : room_length = 19)
  (h2 : room_width = 12)
  (h3 : veranda_length_width = 2.5)
  (h4 : veranda_width_width = 3)
  : (room_length + 2 * veranda_length_width) * (room_width + 2 * veranda_width_width) - room_length * room_width = 204 :=
by
  simp [h1, h2, h3, h4]
  norm_num
  done

end veranda_area_l206_206051


namespace solve_for_x_l206_206947

theorem solve_for_x (x : ℚ) (h : (1 / 3 - 1 / 4 = 4 / x)) : x = 48 := by
  sorry

end solve_for_x_l206_206947


namespace intersection_locus_l206_206371

theorem intersection_locus
  (a b : ℝ) (a_gt_b : a > b) (b_gt_zero : b > 0)
  (ellipse_eq : ∀ x y : ℝ, (x^2)/(a^2) + (y^2)/(b^2) = 1) :
  ∃ (x y : ℝ), (x^2)/(a^2) - (y^2)/(b^2) = 1 :=
sorry

end intersection_locus_l206_206371


namespace quadratic_m_leq_9_l206_206720

-- Define the quadratic equation
def quadratic_eq_has_real_roots (a b c : ℝ) : Prop := 
  b^2 - 4*a*c ≥ 0

-- Define the specific property we need to prove
theorem quadratic_m_leq_9 (m : ℝ) : (quadratic_eq_has_real_roots 1 (-6) m) → (m ≤ 9) := 
by
  sorry

end quadratic_m_leq_9_l206_206720


namespace nancy_packs_of_crayons_l206_206142

def total_crayons : ℕ := 615
def crayons_per_pack : ℕ := 15

theorem nancy_packs_of_crayons : total_crayons / crayons_per_pack = 41 :=
by
  sorry

end nancy_packs_of_crayons_l206_206142


namespace cone_sections_equal_surface_area_l206_206748

theorem cone_sections_equal_surface_area {m r : ℝ} (h_r_pos : r > 0) (h_m_pos : m > 0) :
  ∃ (m1 m2 : ℝ), 
  (m1 = m / Real.sqrt 3) ∧ 
  (m2 = m / 3 * Real.sqrt 6) :=
sorry

end cone_sections_equal_surface_area_l206_206748


namespace shaded_region_is_correct_l206_206561

noncomputable def area_shaded_region : ℝ :=
  let r_small := (3 : ℝ) / 2
  let r_large := (15 : ℝ) / 2
  let area_small := (1 / 2) * Real.pi * r_small^2
  let area_large := (1 / 2) * Real.pi * r_large^2
  (area_large - 2 * area_small + 3 * area_small)

theorem shaded_region_is_correct :
  area_shaded_region = (117 / 4) * Real.pi :=
by
  -- The proof will go here.
  sorry

end shaded_region_is_correct_l206_206561


namespace kenneth_past_finish_line_l206_206290

theorem kenneth_past_finish_line (race_distance : ℕ) (biff_speed : ℕ) (kenneth_speed : ℕ) (time_biff : ℕ) (distance_kenneth : ℕ) :
  race_distance = 500 → biff_speed = 50 → kenneth_speed = 51 → time_biff = race_distance / biff_speed → distance_kenneth = kenneth_speed * time_biff → 
  distance_kenneth - race_distance = 10 := 
by
  intros h1 h2 h3 h4 h5
  sorry

end kenneth_past_finish_line_l206_206290


namespace weight_of_7th_person_l206_206197

/--
There are 6 people in the elevator with an average weight of 152 lbs.
Another person enters the elevator, increasing the average weight to 151 lbs.
Prove that the weight of the 7th person is 145 lbs.
-/
theorem weight_of_7th_person
  (W : ℕ) (X : ℕ) (h1 : W / 6 = 152) (h2 : (W + X) / 7 = 151) :
  X = 145 :=
sorry

end weight_of_7th_person_l206_206197


namespace no_integer_solutions_l206_206838

theorem no_integer_solutions (n : ℕ) (h : 2 ≤ n) :
  ¬ ∃ x y z : ℤ, x^2 + y^2 = z^n :=
sorry

end no_integer_solutions_l206_206838


namespace point_in_fourth_quadrant_l206_206492

-- Define the point (2, -3)
structure Point where
  x : ℤ
  y : ℤ

def A : Point := { x := 2, y := -3 }

-- Define what it means for a point to be in a specific quadrant
def inFirstQuadrant (P : Point) : Prop :=
  P.x > 0 ∧ P.y > 0

def inSecondQuadrant (P : Point) : Prop :=
  P.x < 0 ∧ P.y > 0

def inThirdQuadrant (P : Point) : Prop :=
  P.x < 0 ∧ P.y < 0

def inFourthQuadrant (P : Point) : Prop :=
  P.x > 0 ∧ P.y < 0

-- Define the theorem to prove that the point A lies in the fourth quadrant
theorem point_in_fourth_quadrant : inFourthQuadrant A :=
  sorry

end point_in_fourth_quadrant_l206_206492


namespace total_number_of_toys_is_105_l206_206987

-- Definitions
variables {a k : ℕ}

-- Conditions
def condition_1 (a k : ℕ) : Prop := k ≥ 2
def katya_toys (a : ℕ) : ℕ := a
def lena_toys (a k : ℕ) : ℕ := k * a
def masha_toys (a k : ℕ) : ℕ := k^2 * a

def after_katya_gave_toys (a : ℕ) : ℕ := a - 2
def after_lena_received_toys (a k : ℕ) : ℕ := k * a + 5
def after_masha_gave_toys (a k : ℕ) : ℕ := k^2 * a - 3

def arithmetic_progression (x1 x2 x3 : ℕ) : Prop :=
  2 * x2 = x1 + x3

-- Problem statement to prove
theorem total_number_of_toys_is_105 (a k : ℕ) (h1 : condition_1 a k)
  (h2 : arithmetic_progression (after_katya_gave_toys a) (after_lena_received_toys a k) (after_masha_gave_toys a k)) :
  katya_toys a + lena_toys a k + masha_toys a k = 105 :=
sorry

end total_number_of_toys_is_105_l206_206987


namespace max_value_f_compare_magnitude_l206_206766

open Real

def f (x : ℝ) : ℝ := abs (x - 1) - 2 * abs (x + 1)

-- 1. Prove that the maximum value of f(x) is 2.
theorem max_value_f : ∃ x : ℝ, f x = 2 :=
sorry

-- 2. Given the condition, prove 2m + n > 2.
theorem compare_magnitude (m n : ℝ) (hm : m > 0) (hn : n > 0) (h : (1 / m) + (1 / (2 * n)) = 2) : 
  2 * m + n > 2 :=
sorry

end max_value_f_compare_magnitude_l206_206766


namespace combined_stripes_eq_22_l206_206568

def stripes_olga_per_shoe : ℕ := 3
def shoes_per_person : ℕ := 2
def stripes_olga_total : ℕ := stripes_olga_per_shoe * shoes_per_person

def stripes_rick_per_shoe : ℕ := stripes_olga_per_shoe - 1
def stripes_rick_total : ℕ := stripes_rick_per_shoe * shoes_per_person

def stripes_hortense_per_shoe : ℕ := stripes_olga_per_shoe * 2
def stripes_hortense_total : ℕ := stripes_hortense_per_shoe * shoes_per_person

def total_stripes : ℕ := stripes_olga_total + stripes_rick_total + stripes_hortense_total

theorem combined_stripes_eq_22 : total_stripes = 22 := by
  sorry

end combined_stripes_eq_22_l206_206568


namespace percent_decrease_to_original_price_l206_206617

variable (x : ℝ) (p : ℝ)

def new_price (x : ℝ) : ℝ := 1.35 * x

theorem percent_decrease_to_original_price :
  ∀ (x : ℝ), x ≠ 0 → (1 - (7 / 27)) * (new_price x) = x := 
sorry

end percent_decrease_to_original_price_l206_206617


namespace sqrt_expr_evaluation_l206_206911

theorem sqrt_expr_evaluation :
  (Real.sqrt (5 + 4 * Real.sqrt 3) - Real.sqrt (5 - 4 * Real.sqrt 3)) = 2 * Real.sqrt 2 :=
  sorry

end sqrt_expr_evaluation_l206_206911


namespace penny_money_left_is_5_l206_206228

def penny_initial_money : ℤ := 20
def socks_pairs : ℤ := 4
def price_per_pair_of_socks : ℤ := 2
def price_of_hat : ℤ := 7

def total_cost_of_socks : ℤ := socks_pairs * price_per_pair_of_socks
def total_cost_of_hat_and_socks : ℤ := total_cost_of_socks + price_of_hat
def penny_money_left : ℤ := penny_initial_money - total_cost_of_hat_and_socks

theorem penny_money_left_is_5 : penny_money_left = 5 := by
  sorry

end penny_money_left_is_5_l206_206228


namespace binary_operation_correct_l206_206847

theorem binary_operation_correct :
  let b1 := 0b11011
  let b2 := 0b1011
  let b3 := 0b11100
  let b4 := 0b10101
  let b5 := 0b1001
  b1 + b2 - b3 + b4 - b5 = 0b11110 := by
  sorry

end binary_operation_correct_l206_206847


namespace gcd_square_product_l206_206990

theorem gcd_square_product (x y z : ℕ) (h : 1 / (x : ℝ) - 1 / (y : ℝ) = 1 / (z : ℝ)) : 
    ∃ n : ℕ, gcd x (gcd y z) * x * y * z = n * n := 
sorry

end gcd_square_product_l206_206990


namespace range_of_m_l206_206475

theorem range_of_m (m x : ℝ) (h : (x + m) / 3 - (2 * x - 1) / 2 = m) (hx : x ≤ 0) : m ≥ 3 / 4 := 
sorry

end range_of_m_l206_206475


namespace bob_total_miles_l206_206540

def total_miles_day1 (T : ℝ) := 0.20 * T
def remaining_miles_day1 (T : ℝ) := T - total_miles_day1 T
def total_miles_day2 (T : ℝ) := 0.50 * remaining_miles_day1 T
def remaining_miles_day2 (T : ℝ) := remaining_miles_day1 T - total_miles_day2 T
def total_miles_day3 (T : ℝ) := 28

theorem bob_total_miles (T : ℝ) (h : total_miles_day3 T = remaining_miles_day2 T) : T = 70 :=
by
  sorry

end bob_total_miles_l206_206540


namespace solve_xy_eq_yx_l206_206456

theorem solve_xy_eq_yx (x y : ℕ) (hxy : x ≠ y) : x^y = y^x ↔ ((x = 2 ∧ y = 4) ∨ (x = 4 ∧ y = 2)) :=
by
  sorry

end solve_xy_eq_yx_l206_206456


namespace inequality_problem_l206_206243

variable {R : Type*} [LinearOrderedField R]

theorem inequality_problem
  (a b : R) 
  (ha_pos : 0 < a)
  (hb_pos : 0 < b)
  (hab : a + b = 1) :
  (a + 1 / a)^2 + (b + 1 / b)^2 ≥ 25 / 2 := 
sorry

end inequality_problem_l206_206243


namespace ice_cream_ratio_l206_206045

theorem ice_cream_ratio
    (T : ℕ)
    (W : ℕ)
    (hT : T = 12000)
    (hMultiple : ∃ k : ℕ, W = k * T)
    (hTotal : T + W = 36000) :
    W / T = 2 :=
by
  -- Proof is omitted, so sorry is used
  sorry

end ice_cream_ratio_l206_206045


namespace right_angle_case_acute_angle_case_obtuse_angle_case_l206_206710

-- Definitions
def circumcenter (O : Type) (A B C : Type) : Prop := sorry -- Definition of circumcenter.

def orthocenter (H : Type) (A B C : Type) : Prop := sorry -- Definition of orthocenter.

noncomputable def R : ℝ := sorry -- Circumradius of the triangle.

-- Conditions
variables {A B C O H : Type}
  (h_circumcenter : circumcenter O A B C)
  (h_orthocenter : orthocenter H A B C)

-- The angles α β γ represent the angles of triangle ABC.
variables {α β γ : ℝ}

-- Statements
-- Case 1: ∠C = 90°
theorem right_angle_case (h_angle_C : γ = 90) (h_H_eq_C : H = C) (h_AB_eq_2R : AB = 2 * R) : AH + BH >= AB := by
  sorry

-- Case 2: ∠C < 90°
theorem acute_angle_case (h_angle_C_lt_90 : γ < 90) : O_in_triangle_AHB := by
  sorry

-- Case 3: ∠C > 90°
theorem obtuse_angle_case (h_angle_C_gt_90 : γ > 90) : AH + BH > 2 * R := by
  sorry

end right_angle_case_acute_angle_case_obtuse_angle_case_l206_206710


namespace geometric_sum_s9_l206_206787

variable (S : ℕ → ℝ)

theorem geometric_sum_s9
  (h1 : S 3 = 7)
  (h2 : S 6 = 63) :
  S 9 = 511 :=
by
  sorry

end geometric_sum_s9_l206_206787


namespace susan_age_indeterminate_l206_206300

-- Definitions and conditions
def james_age_in_15_years : ℕ := 37
def current_james_age : ℕ := james_age_in_15_years - 15
def james_age_8_years_ago : ℕ := current_james_age - 8
def janet_age_8_years_ago : ℕ := james_age_8_years_ago / 2
def current_janet_age : ℕ := janet_age_8_years_ago + 8

-- Problem: Prove that without Janet's age when Susan was born, we cannot determine Susan's age in 5 years.
theorem susan_age_indeterminate (susan_current_age : ℕ) : 
  (∃ janet_age_when_susan_born : ℕ, susan_current_age = current_janet_age - janet_age_when_susan_born) → 
  ¬ (∃ susan_age_in_5_years : ℕ, susan_age_in_5_years = susan_current_age + 5) := 
by
  sorry

end susan_age_indeterminate_l206_206300


namespace tan_theta_half_l206_206022

theorem tan_theta_half (θ : ℝ) (a b : ℝ × ℝ) 
  (h₀ : a = (Real.sin θ, 1)) 
  (h₁ : b = (-2, Real.cos θ)) 
  (h₂ : a.1 * b.1 + a.2 * b.2 = 0) : Real.tan θ = 1 / 2 :=
sorry

end tan_theta_half_l206_206022


namespace probability_problems_l206_206208

theorem probability_problems (x : ℕ) :
  (0 = (if 8 + 12 > 8 then 0 else 1)) ∧
  (1 = (1 - 0)) ∧
  (3 / 5 = 12 / 20) ∧
  (4 / 5 = (8 + x) / 20 → x = 8) := by sorry

end probability_problems_l206_206208


namespace perfect_squares_of_k_l206_206740

theorem perfect_squares_of_k (k : ℕ) (h : ∃ (a : ℕ), k * (k + 1) = 3 * a^2) : 
  ∃ (m n : ℕ), k = 3 * m^2 ∧ k + 1 = n^2 := 
sorry

end perfect_squares_of_k_l206_206740


namespace find_m_value_l206_206037

theorem find_m_value (m : ℤ) : (∃ a : ℤ, x^2 + 2 * (m + 1) * x + 25 = (x + a)^2) ↔ (m = 4 ∨ m = -6) := 
sorry

end find_m_value_l206_206037


namespace non_neg_int_solutions_m_value_integer_values_of_m_l206_206320

-- 1. Non-negative integer solutions of x + 2y = 3
theorem non_neg_int_solutions (x y : ℕ) :
  x + 2 * y = 3 ↔ (x = 3 ∧ y = 0) ∨ (x = 1 ∧ y = 1) :=
sorry

-- 2. If (x, y) = (1, 1) satisfies both x + 2y = 3 and x + y = 2, then m = -4
theorem m_value (m : ℝ) :
  (1 + 2 * 1 = 3) ∧ (1 + 1 = 2) ∧ (1 - 2 * 1 + m * 1 = -5) → m = -4 :=
sorry

-- 3. Given n = 3, integer values of m are -2 or 0
theorem integer_values_of_m (m : ℤ) :
  ∃ x y : ℤ, 3 * x + 4 * y = 5 ∧ x - 2 * y + m * x = -5 → m = -2 ∨ m = 0 :=
sorry

end non_neg_int_solutions_m_value_integer_values_of_m_l206_206320


namespace sachin_younger_than_rahul_l206_206187

theorem sachin_younger_than_rahul :
  ∀ (sachin_age rahul_age : ℕ),
  (sachin_age / rahul_age = 6 / 9) →
  (sachin_age = 14) →
  (rahul_age - sachin_age = 7) :=
by
  sorry

end sachin_younger_than_rahul_l206_206187


namespace prank_combinations_l206_206881

theorem prank_combinations :
  let monday_choices := 1
  let tuesday_choices := 2
  let wednesday_choices := 5
  let thursday_choices := 4
  let friday_choices := 1
  (monday_choices * tuesday_choices * wednesday_choices * thursday_choices * friday_choices) = 40 :=
by
  sorry

end prank_combinations_l206_206881


namespace suitable_sampling_method_l206_206826

-- Conditions given
def num_products : ℕ := 40
def num_top_quality : ℕ := 10
def num_second_quality : ℕ := 25
def num_defective : ℕ := 5
def draw_count : ℕ := 8

-- Possible sampling methods
inductive SamplingMethod
| DrawingLots : SamplingMethod
| RandomNumberTable : SamplingMethod
| Systematic : SamplingMethod
| Stratified : SamplingMethod

-- Problem statement (to be proved)
theorem suitable_sampling_method : 
  (num_products = 40) ∧ 
  (num_top_quality = 10) ∧ 
  (num_second_quality = 25) ∧ 
  (num_defective = 5) ∧ 
  (draw_count = 8) → 
  SamplingMethod.Stratified = SamplingMethod.Stratified :=
by sorry

end suitable_sampling_method_l206_206826


namespace investment_after_8_years_l206_206827

noncomputable def compound_interest (P : ℝ) (r : ℝ) (n : ℕ) : ℝ :=
  P * (1 + r)^n

theorem investment_after_8_years :
  let P := 500
  let r := 0.03
  let n := 8
  let A := compound_interest P r n
  round A = 633 :=
by
  sorry

end investment_after_8_years_l206_206827


namespace chengdu_chongqing_scientific_notation_l206_206974

theorem chengdu_chongqing_scientific_notation:
  (185000 : ℝ) = 1.85 * 10^5 :=
sorry

end chengdu_chongqing_scientific_notation_l206_206974


namespace exponential_comparison_l206_206176

theorem exponential_comparison (a b c : ℝ) (h₁ : a = 0.5^((1:ℝ)/2))
                                          (h₂ : b = 0.5^((1:ℝ)/3))
                                          (h₃ : c = 0.5^((1:ℝ)/4)) : 
  a < b ∧ b < c := by
  sorry

end exponential_comparison_l206_206176


namespace cement_used_tess_street_l206_206057

-- Define the given conditions
def cement_used_lexi_street : ℝ := 10
def total_cement_used : ℝ := 15.1

-- Define the statement to prove the amount of cement used for Tess's street
theorem cement_used_tess_street : total_cement_used - cement_used_lexi_street = 5.1 :=
by
  sorry

end cement_used_tess_street_l206_206057


namespace percentage_discount_proof_l206_206584

noncomputable def ticket_price : ℝ := 25
noncomputable def price_to_pay : ℝ := 18.75
noncomputable def discount_amount : ℝ := ticket_price - price_to_pay
noncomputable def percentage_discount : ℝ := (discount_amount / ticket_price) * 100

theorem percentage_discount_proof : percentage_discount = 25 := by
  sorry

end percentage_discount_proof_l206_206584


namespace calculate_expression_l206_206368

def thirteen_power_thirteen_div_thirteen_power_twelve := 13 ^ 13 / 13 ^ 12
def expression := (thirteen_power_thirteen_div_thirteen_power_twelve ^ 3) * (3 ^ 3)
/- We define the main statement to be proven -/
theorem calculate_expression : (expression / 2 ^ 6) = 926 := sorry

end calculate_expression_l206_206368


namespace denominator_of_expression_l206_206429

theorem denominator_of_expression (x : ℝ) (h : (1 / x) ^ 1 = 0.25) : x = 4 := by
  sorry

end denominator_of_expression_l206_206429


namespace min_book_corner_cost_l206_206697

theorem min_book_corner_cost :
  ∃ x : ℕ, 0 ≤ x ∧ x ≤ 30 ∧
  80 * x + 30 * (30 - x) ≤ 1900 ∧
  50 * x + 60 * (30 - x) ≤ 1620 ∧
  860 * x + 570 * (30 - x) = 22320 := sorry

end min_book_corner_cost_l206_206697


namespace geometric_sequence_sum_l206_206806

noncomputable def aₙ (n : ℕ) : ℝ := (2 / 3) ^ (n - 1)

noncomputable def Sₙ (n : ℕ) : ℝ := 3 * (1 - (2 / 3) ^ n)

theorem geometric_sequence_sum (n : ℕ) : Sₙ n = 3 - 2 * aₙ n := by
  sorry

end geometric_sequence_sum_l206_206806


namespace sequence_comparison_l206_206047

-- Define arithmetic sequence
def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d, ∀ n, a (n + 1) = a n + d

-- Define geometric sequence
def geometric_sequence (b : ℕ → ℝ) (q : ℝ) : Prop :=
  q ≠ 1 ∧ (∀ n, b (n + 1) = b n * q) ∧ (∀ i, i ≥ 1 → b i > 0)

-- Main theorem to prove
theorem sequence_comparison {a b : ℕ → ℝ} (q : ℝ) (h_a_arith : arithmetic_sequence a) 
  (h_b_geom : geometric_sequence b q) (h_eq_1 : a 1 = b 1) (h_eq_11 : a 11 = b 11) :
  a 6 > b 6 :=
sorry

end sequence_comparison_l206_206047


namespace smallest_x_solution_l206_206620

theorem smallest_x_solution :
  ∃ x : ℝ, x * |x| + 3 * x = 5 * x + 2 ∧ (∀ y : ℝ, y * |y| + 3 * y = 5 * y + 2 → x ≤ y)
:=
sorry

end smallest_x_solution_l206_206620


namespace water_tank_capacity_l206_206101

theorem water_tank_capacity
  (tank_capacity : ℝ)
  (h : 0.30 * tank_capacity = 0.90 * tank_capacity - 54) :
  tank_capacity = 90 :=
by
  -- proof goes here
  sorry

end water_tank_capacity_l206_206101


namespace area_of_ABCD_l206_206003

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

end area_of_ABCD_l206_206003


namespace Kishore_misc_expense_l206_206044

theorem Kishore_misc_expense:
  let savings := 2400
  let percent_saved := 0.10
  let rent := 5000
  let milk := 1500
  let groceries := 4500
  let education := 2500
  let petrol := 2000
  let total_salary := savings / percent_saved 
  let total_spent := rent + milk + groceries + education + petrol
  total_salary - (total_spent + savings) = 6100 := 
by
  sorry

end Kishore_misc_expense_l206_206044


namespace find_m_n_and_sqrt_l206_206594

-- definitions based on conditions
def condition_1 (m : ℤ) : Prop := m + 3 = 1
def condition_2 (n : ℤ) : Prop := 2 * n - 12 = 64

-- the proof problem statement
theorem find_m_n_and_sqrt (m n : ℤ) (h1 : condition_1 m) (h2 : condition_2 n) : 
  m = -2 ∧ n = 38 ∧ Int.sqrt (m + n) = 6 := 
sorry

end find_m_n_and_sqrt_l206_206594


namespace binom_12_6_l206_206908

theorem binom_12_6 : Nat.choose 12 6 = 924 :=
by
  sorry

end binom_12_6_l206_206908


namespace bucket_full_weight_l206_206465

theorem bucket_full_weight (x y c d : ℝ) 
  (h1 : x + (3/4) * y = c)
  (h2 : x + (3/5) * y = d) :
  x + y = (5/3) * c - (5/3) * d :=
by
  sorry

end bucket_full_weight_l206_206465


namespace cos_diff_expression_eq_half_l206_206916

theorem cos_diff_expression_eq_half :
  (Real.cos (Real.pi * 24 / 180) * Real.cos (Real.pi * 36 / 180) -
   Real.cos (Real.pi * 66 / 180) * Real.cos (Real.pi * 54 / 180)) = 1 / 2 := by
sorry

end cos_diff_expression_eq_half_l206_206916


namespace company_budget_salaries_degrees_l206_206619

theorem company_budget_salaries_degrees :
  let transportation := 0.20
  let research_and_development := 0.09
  let utilities := 0.05
  let equipment := 0.04
  let supplies := 0.02
  let total_budget := 1.0
  let total_percentage := transportation + research_and_development + utilities + equipment + supplies
  let salaries_percentage := total_budget - total_percentage
  let total_degrees := 360.0
  let degrees_salaries := salaries_percentage * total_degrees
  degrees_salaries = 216 :=
by
  sorry

end company_budget_salaries_degrees_l206_206619


namespace impossible_to_transport_stones_l206_206440

-- Define the conditions of the problem
def stones : List ℕ := List.range' 370 (468 - 370 + 2 + 1) 2
def truck_capacity : ℕ := 3000
def number_of_trucks : ℕ := 7
def number_of_stones : ℕ := 50

-- Prove that it is impossible to transport the stones using the given trucks
theorem impossible_to_transport_stones :
  stones.length = number_of_stones →
  (∀ weights ∈ stones.sublists, (weights.sum ≤ truck_capacity → List.length weights ≤ number_of_trucks)) → 
  false :=
by
  sorry

end impossible_to_transport_stones_l206_206440


namespace triangle_area_l206_206374

theorem triangle_area (x y : ℝ) (h : 3 * x + y = 9) : 
  (1 / 2) * 3 * 9 = 13.5 :=
sorry

end triangle_area_l206_206374


namespace janice_purchases_l206_206848

theorem janice_purchases (a b c : ℕ) : 
  a + b + c = 50 ∧ 30 * a + 200 * b + 300 * c = 5000 → a = 10 :=
sorry

end janice_purchases_l206_206848


namespace solve_inequality_system_l206_206179

theorem solve_inequality_system (x : ℝ) :
  (x + 2 > -1) ∧ (x - 5 < 3 * (x - 1)) ↔ (x > -1) :=
by
  sorry

end solve_inequality_system_l206_206179


namespace problem_proof_l206_206330

-- Define the conditions
def a (n : ℕ) : Real := sorry  -- a is some real number, so it's non-deterministic here

def a_squared (n : ℕ) : Real := a n ^ (2 * n)  -- a^(2n)

-- Main theorem to prove
theorem problem_proof (n : ℕ) (h : a_squared n = 3) : 2 * (a n ^ (6 * n)) - 1 = 53 :=
by
  sorry  -- Proof to be completed

end problem_proof_l206_206330


namespace circle_radius_l206_206696

theorem circle_radius (d : ℝ) (h : d = 10) : d / 2 = 5 :=
by
  sorry

end circle_radius_l206_206696


namespace ratio_third_to_second_year_l206_206731

-- Define the yearly production of the apple tree
def first_year_production : Nat := 40
def second_year_production : Nat := 2 * first_year_production + 8
def total_production_three_years : Nat := 194
def third_year_production : Nat := total_production_three_years - (first_year_production + second_year_production)

-- Define the ratio calculation
def ratio (a b : Nat) : (Nat × Nat) := 
  let gcd_ab := Nat.gcd a b 
  (a / gcd_ab, b / gcd_ab)

-- Prove the ratio of the third year's production to the second year's production
theorem ratio_third_to_second_year : 
  ratio third_year_production second_year_production = (3, 4) :=
  sorry

end ratio_third_to_second_year_l206_206731


namespace missing_digit_first_digit_l206_206111

-- Definitions derived from conditions
def is_three_digit_number (n : ℕ) : Prop := 100 ≤ n ∧ n ≤ 999
def is_divisible_by_six (n : ℕ) : Prop := n % 6 = 0
def multiply_by_two (d : ℕ) : ℕ := 2 * d

-- Main statement to prove
theorem missing_digit_first_digit (d : ℕ) (n : ℕ) 
  (h1 : multiply_by_two d = n) 
  (h2 : is_three_digit_number n) 
  (h3 : is_divisible_by_six n)
  (h4 : d = 2)
  : n / 100 = 2 :=
sorry

end missing_digit_first_digit_l206_206111


namespace count_integers_M_3_k_l206_206560

theorem count_integers_M_3_k (M : ℕ) (hM : M < 500) :
  (∃ k : ℕ, k ≥ 1 ∧ ∃ m : ℕ, m ≥ 1 ∧ M = 2 * k * (m + k - 1)) ∧
  (∃ k1 k2 k3 k4 : ℕ, k1 ≠ k2 ∧ k1 ≠ k3 ∧ k1 ≠ k4 ∧
    k2 ≠ k3 ∧ k2 ≠ k4 ∧ k3 ≠ k4 ∧
    (M / 2 = (k1 + k2 + k3 + k4) ∨ M / 2 = (k1 * k2 * k3 * k4))) →
  (∃ n : ℕ, n = 6) :=
by
  sorry

end count_integers_M_3_k_l206_206560


namespace point_on_curve_iff_F_eq_zero_l206_206633

variable (F : ℝ → ℝ → ℝ)
variable (a b : ℝ)

theorem point_on_curve_iff_F_eq_zero :
  (F a b = 0) ↔ (∃ P : ℝ × ℝ, P = (a, b) ∧ F P.1 P.2 = 0) :=
by
  sorry

end point_on_curve_iff_F_eq_zero_l206_206633


namespace swap_correct_l206_206415

variable (a b c : ℕ)

noncomputable def swap_and_verify (a : ℕ) (b : ℕ) : Prop :=
  let c := b
  let b := a
  let a := c
  a = 2012 ∧ b = 2011

theorem swap_correct :
  ∀ a b : ℕ, a = 2011 → b = 2012 → swap_and_verify a b :=
by
  intros a b ha hb
  sorry

end swap_correct_l206_206415


namespace terminating_decimal_expansion_of_13_over_320_l206_206194

theorem terminating_decimal_expansion_of_13_over_320 : ∃ (b : ℕ) (a : ℚ), (13 : ℚ) / 320 = a / 10 ^ b ∧ a / 10 ^ b = 0.650 :=
by
  sorry

end terminating_decimal_expansion_of_13_over_320_l206_206194


namespace colby_mangoes_harvested_60_l206_206643

variable (kg_left kg_each : ℕ)

def totalKgMangoes (x : ℕ) : Prop :=
  ∃ x : ℕ, 
  kg_left = (x - 20) / 2 ∧ 
  kg_each * kg_left = 160 ∧
  kg_each = 8

-- Problem Statement: Prove the total kilograms of mangoes harvested is 60 given the conditions.
theorem colby_mangoes_harvested_60 (x : ℕ) (h1 : x - 20 = 2 * kg_left)
(h2 : kg_each * kg_left = 160) (h3 : kg_each = 8) : x = 60 := by
  sorry

end colby_mangoes_harvested_60_l206_206643


namespace angle_measure_l206_206926

theorem angle_measure (x : ℝ) 
  (h1 : 90 - x = (2 / 5) * (180 - x)) :
  x = 30 :=
by
  sorry

end angle_measure_l206_206926


namespace gcd_16_12_eq_4_l206_206940

theorem gcd_16_12_eq_4 : Nat.gcd 16 12 = 4 := by
  -- Skipping proof using sorry
  sorry

end gcd_16_12_eq_4_l206_206940


namespace number_of_people_l206_206958

theorem number_of_people (x : ℕ) : 
  (x % 10 = 1) ∧
  (x % 9 = 1) ∧
  (x % 8 = 1) ∧
  (x % 7 = 1) ∧
  (x % 6 = 1) ∧
  (x % 5 = 1) ∧
  (x % 4 = 1) ∧
  (x % 3 = 1) ∧
  (x % 2 = 1) ∧
  (x < 5000) →
  x = 2521 :=
sorry

end number_of_people_l206_206958


namespace time_for_trains_to_cross_l206_206472

def length_train1 := 500 -- 500 meters
def length_train2 := 750 -- 750 meters
def speed_train1 := 60 * 1000 / 3600 -- 60 km/hr to m/s
def speed_train2 := 40 * 1000 / 3600 -- 40 km/hr to m/s
def relative_speed := speed_train1 + speed_train2 -- relative speed in m/s
def combined_length := length_train1 + length_train2 -- sum of lengths of both trains

theorem time_for_trains_to_cross :
  (combined_length / relative_speed) = 45 := 
by
  sorry

end time_for_trains_to_cross_l206_206472


namespace Q_has_negative_root_l206_206362

def Q (x : ℝ) : ℝ := x^7 + 2 * x^5 + 5 * x^3 - x + 12

theorem Q_has_negative_root : ∃ x : ℝ, x < 0 ∧ Q x = 0 :=
by
  sorry

end Q_has_negative_root_l206_206362


namespace white_balls_count_l206_206466

theorem white_balls_count (total_balls : ℕ) (green_balls : ℕ) (yellow_balls : ℕ) (red_balls : ℕ) (purple_balls : ℕ) (prob_neither_red_nor_purple : ℝ) (W : ℕ)
    (h_total : total_balls = 100)
    (h_green : green_balls = 30)
    (h_yellow : yellow_balls = 10)
    (h_red : red_balls = 37)
    (h_purple : purple_balls = 3)
    (h_prob : prob_neither_red_nor_purple = 0.6)
    (h_computation : W = total_balls * prob_neither_red_nor_purple - (green_balls + yellow_balls)) :
    W = 20 := 
sorry

end white_balls_count_l206_206466


namespace typing_speed_in_6_minutes_l206_206562

theorem typing_speed_in_6_minutes (total_chars : ℕ) (chars_first_minute : ℕ) (chars_last_minute : ℕ) (chars_other_minutes : ℕ) :
  total_chars = 2098 →
  chars_first_minute = 112 →
  chars_last_minute = 97 →
  chars_other_minutes = 1889 →
  (1889 / 6 : ℝ) < 315 → 
  ¬(∀ n, 1 ≤ n ∧ n ≤ 14 - 6 + 1 → chars_other_minutes / 6 ≥ 946) :=
by
  -- Given that analyzing the content, 
  -- proof is skipped here, replace this line with the actual proof.
  sorry

end typing_speed_in_6_minutes_l206_206562


namespace smallest_lcm_l206_206631

theorem smallest_lcm (k l : ℕ) (hk : 999 < k ∧ k < 10000) (hl : 999 < l ∧ l < 10000)
  (h_gcd : Nat.gcd k l = 5) : Nat.lcm k l = 201000 :=
sorry

end smallest_lcm_l206_206631


namespace brass_players_10_l206_206743

theorem brass_players_10:
  ∀ (brass woodwind percussion : ℕ),
    brass + woodwind + percussion = 110 →
    percussion = 4 * woodwind →
    woodwind = 2 * brass →
    brass = 10 :=
by
  intros brass woodwind percussion h1 h2 h3
  sorry

end brass_players_10_l206_206743


namespace swiss_probability_is_30_percent_l206_206167

def total_cheese_sticks : Nat := 22 + 34 + 29 + 45 + 20

def swiss_cheese_sticks : Nat := 45

def probability_swiss : Nat :=
  (swiss_cheese_sticks * 100) / total_cheese_sticks

theorem swiss_probability_is_30_percent :
  probability_swiss = 30 := by
  sorry

end swiss_probability_is_30_percent_l206_206167


namespace triangle_statements_l206_206091

-- Define the fundamental properties of the triangle
noncomputable def triangle (A B C : ℝ) (a b c : ℝ) : Prop :=
  a = 45 ∧ a = 2 ∧ b = 2 * Real.sqrt 2 ∧ 
  (a - b = c * Real.cos B - c * Real.cos A)

-- Statement A
def statement_A (A B C a b c : ℝ) (h : triangle A B C a b c) : Prop :=
  ∃ B, Real.sin B = 1

-- Statement B
def statement_B (A B C : ℝ) (v_AC v_AB : ℝ) : Prop :=
  v_AC * v_AB > 0 → Real.cos A > 0

-- Statement C
def statement_C (A B : ℝ) (a b : ℝ) : Prop :=
  Real.sin A > Real.sin B → a > b

-- Statement D
def statement_D (A B C a b c : ℝ) (h : triangle A B C a b c) : Prop :=
  (a - b = c * Real.cos B - c * Real.cos A) →
  (a = b ∨ c^2 = a^2 + b^2)

-- Final proof statement
theorem triangle_statements (A B C a b c : ℝ) (v_AC v_AB : ℝ) 
  (h_triangle : triangle A B C a b c) :
  (statement_A A B C a b c h_triangle) ∧
  ¬(statement_B A B C v_AC v_AB) ∧
  (statement_C A B a b) ∧
  (statement_D A B C a b c h_triangle) :=
by sorry

end triangle_statements_l206_206091


namespace profit_percentage_l206_206811

theorem profit_percentage (C S : ℝ) (hC : C = 60) (hS : S = 75) : ((S - C) / C) * 100 = 25 :=
by
  sorry

end profit_percentage_l206_206811


namespace range_of_a_l206_206659

theorem range_of_a (a : ℝ) :
  (∃ x1 x2 x3 : ℝ, x1 ≠ x2 ∧ x2 ≠ x3 ∧ x3 ≠ x1 ∧ x1^3 - 3*x1 + a = 0 ∧ x2^3 - 3*x2 + a = 0 ∧ x3^3 - 3*x3 + a = 0) 
  ↔ -2 < a ∧ a < 2 :=
sorry

end range_of_a_l206_206659


namespace min_cost_at_100_l206_206042

noncomputable def cost_function (v : ℝ) : ℝ :=
if (0 < v ∧ v ≤ 50) then (123000 / v + 690)
else if (v > 50) then (3 * v^2 / 50 + 120000 / v + 600)
else 0

theorem min_cost_at_100 : ∃ v : ℝ, v = 100 ∧ cost_function v = 2400 :=
by
  -- We are not proving but stating the theorem here
  sorry

end min_cost_at_100_l206_206042


namespace power_mod_l206_206164

theorem power_mod (a : ℕ) : 5 ^ 2023 % 17 = 2 := by
  sorry

end power_mod_l206_206164


namespace number_of_balls_greater_l206_206299

theorem number_of_balls_greater (n x : ℤ) (h1 : n = 25) (h2 : n - x = 30 - n) : x = 20 := by
  sorry

end number_of_balls_greater_l206_206299


namespace notched_circle_coordinates_l206_206618

variable (a b : ℝ)

theorem notched_circle_coordinates : 
  let sq_dist_from_origin := a^2 + b^2
  let A := (a, b + 5)
  let C := (a + 3, b)
  (a^2 + (b + 5)^2 = 36 ∧ (a + 3)^2 + b^2 = 36) →
  (sq_dist_from_origin = 4.16 ∧ A = (2, 5.4) ∧ C = (5, 0.4)) :=
by
  sorry

end notched_circle_coordinates_l206_206618


namespace sequences_equal_l206_206996

noncomputable def a : ℕ → ℚ
| 0 => 0
| 1 => 1
| (n+2) => (2018 / (n + 1)) * a (n + 1) + a n

noncomputable def b : ℕ → ℚ
| 0 => 0
| 1 => 1
| (n+2) => (2020 / (n + 1)) * b (n + 1) + b n

theorem sequences_equal :
  (a 1010) / 1010 = (b 1009) / 1009 :=
sorry

end sequences_equal_l206_206996


namespace new_people_moved_in_l206_206856

theorem new_people_moved_in (N : ℕ) : (∃ N, 1/16 * (780 - 400 + N : ℝ) = 60) → N = 580 := by
  intros hN
  sorry

end new_people_moved_in_l206_206856


namespace intersection_M_N_eq_2_4_l206_206294

def M : Set ℕ := {2, 4, 6, 8, 10}
def N : Set ℕ := {x | ∃ y, y = Real.log (6 - x) ∧ x < 6}

theorem intersection_M_N_eq_2_4 : M ∩ N = {2, 4} :=
by sorry

end intersection_M_N_eq_2_4_l206_206294


namespace intersect_inverse_l206_206341

theorem intersect_inverse (c d : ℤ) (h1 : 2 * (-4) + c = d) (h2 : 2 * d + c = -4) : d = -4 := 
by
  sorry

end intersect_inverse_l206_206341


namespace population_net_increase_l206_206049

-- Definitions for birth and death rate, and the number of seconds in a day
def birth_rate : ℕ := 10
def death_rate : ℕ := 2
def seconds_in_day : ℕ := 86400

-- Calculate the population net increase in one day
theorem population_net_increase (birth_rate death_rate seconds_in_day : ℕ) :
  (seconds_in_day / 2) * birth_rate - (seconds_in_day / 2) * death_rate = 345600 :=
by
  sorry

end population_net_increase_l206_206049


namespace sum_of_odd_integers_21_to_51_l206_206154

noncomputable def arithmetic_seq (a d : ℕ) (n : ℕ) : ℕ := a + (n - 1) * d

noncomputable def sum_arithmetic_seq (a d l : ℕ) : ℕ :=
  let n := (l - a) / d + 1
  n * (a + l) / 2

theorem sum_of_odd_integers_21_to_51 : sum_arithmetic_seq 21 2 51 = 576 := by
  sorry

end sum_of_odd_integers_21_to_51_l206_206154


namespace tangent_line_circle_l206_206737

theorem tangent_line_circle {m : ℝ} (tangent : ∀ x y : ℝ, x + y + m = 0 → x^2 + y^2 = m → false) : m = 2 :=
sorry

end tangent_line_circle_l206_206737


namespace house_cats_initial_l206_206868

def initial_house_cats (S A T H : ℝ) : Prop :=
  S + H + A = T

theorem house_cats_initial (S A T H : ℝ) (h1 : S = 13.0) (h2 : A = 10.0) (h3 : T = 28) :
  initial_house_cats S A T H ↔ H = 5 := by
sorry

end house_cats_initial_l206_206868


namespace laptop_repair_cost_l206_206776

theorem laptop_repair_cost
  (price_phone_repair : ℝ)
  (price_computer_repair : ℝ)
  (price_laptop_repair : ℝ)
  (condition1 : price_phone_repair = 11)
  (condition2 : price_computer_repair = 18)
  (condition3 : 5 * price_phone_repair + 2 * price_laptop_repair + 2 * price_computer_repair = 121) :
  price_laptop_repair = 15 :=
by
  sorry

end laptop_repair_cost_l206_206776


namespace find_y_interval_l206_206815

open Real

theorem find_y_interval {y : ℝ}
  (hy_nonzero : y ≠ 0)
  (h_denominator_nonzero : 1 + 3 * y - 4 * y^2 ≠ 0) :
  (y^2 + 9 * y - 1 = 0) →
  (∀ y, y ∈ Set.Icc (-(9 + sqrt 85)/2) (-(9 - sqrt 85)/2) \ {y | y = 0 ∨ 1 + 3 * y - 4 * y^2 = 0} ↔
  (y * (3 - 3 * y))/(1 + 3 * y - 4 * y^2) ≤ 1) :=
by
  sorry

end find_y_interval_l206_206815


namespace sum_powers_mod_5_l206_206660

theorem sum_powers_mod_5 (n : ℕ) (h : ¬ (n % 4 = 0)) : 
  (1^n + 2^n + 3^n + 4^n) % 5 = 0 :=
by
  sorry

end sum_powers_mod_5_l206_206660


namespace seq_not_square_l206_206648

open Nat

theorem seq_not_square (n : ℕ) (r : ℕ) :
  (r = 11 ∨ r = 111 ∨ r = 1111 ∨ ∃ k : ℕ, r = k * 10^(n + 1) + 1) →
  (r % 4 = 3) →
  (¬ ∃ m : ℕ, r = m^2) :=
by
  intro h_seq h_mod
  intro h_square
  sorry

end seq_not_square_l206_206648


namespace center_of_gravity_shift_center_of_gravity_shift_result_l206_206878

variable (l s : ℝ) (s_val : s = 60)
#check (s_val : s = 60)

theorem center_of_gravity_shift : abs ((l / 2) - ((l - s) / 2)) = s / 2 := 
by sorry

theorem center_of_gravity_shift_result : (s / 2 = 30) :=
by sorry

end center_of_gravity_shift_center_of_gravity_shift_result_l206_206878


namespace expression_divisible_by_11_l206_206959

theorem expression_divisible_by_11 (n : ℕ) : (3 ^ (2 * n + 2) + 2 ^ (6 * n + 1)) % 11 = 0 :=
sorry

end expression_divisible_by_11_l206_206959


namespace purely_imaginary_iff_real_iff_second_quadrant_iff_l206_206477

def Z (m : ℝ) : ℂ := ⟨m^2 - 2 * m - 3, m^2 + 3 * m + 2⟩

theorem purely_imaginary_iff (m : ℝ) : (Z m).re = 0 ∧ (Z m).im ≠ 0 ↔ m = 3 :=
by sorry

theorem real_iff (m : ℝ) : (Z m).im = 0 ↔ m = -1 ∨ m = -2 :=
by sorry

theorem second_quadrant_iff (m : ℝ) : (Z m).re < 0 ∧ (Z m).im > 0 ↔ -1 < m ∧ m < 3 :=
by sorry

end purely_imaginary_iff_real_iff_second_quadrant_iff_l206_206477


namespace dentist_age_l206_206160

theorem dentist_age (x : ℝ) (h : (x - 8) / 6 = (x + 8) / 10) : x = 32 :=
  by
  sorry

end dentist_age_l206_206160


namespace throwing_skips_l206_206591

theorem throwing_skips :
  ∃ x y : ℕ, 
  y > x ∧ 
  (∃ z : ℕ, z = 2 * y ∧ 
  (∃ w : ℕ, w = z - 3 ∧ 
  (∃ u : ℕ, u = w + 1 ∧ u = 8))) ∧ 
  x + y + 2 * y + (2 * y - 3) + (2 * y - 2) = 33 ∧ 
  y - x = 2 :=
sorry

end throwing_skips_l206_206591


namespace rachel_milk_amount_l206_206763

theorem rachel_milk_amount : 
  let don_milk := (3 : ℚ) / 7
  let rachel_fraction := 4 / 5
  let rachel_milk := rachel_fraction * don_milk
  rachel_milk = 12 / 35 :=
by sorry

end rachel_milk_amount_l206_206763


namespace sum_of_factors_l206_206761

theorem sum_of_factors (W F c : ℕ) (hW_gt_20: W > 20) (hF_gt_20: F > 20) (product_eq : W * F = 770) (sum_eq : W + F = c) :
  c = 57 :=
by sorry

end sum_of_factors_l206_206761


namespace line_eq_slope_form_l206_206473

theorem line_eq_slope_form (a b c : ℝ) (h : b ≠ 0) :
    ∃ k l : ℝ, ∀ x y : ℝ, (a * x + b * y + c = 0) ↔ (y = k * x + l) := 
sorry

end line_eq_slope_form_l206_206473


namespace contractor_male_workers_l206_206893

noncomputable def number_of_male_workers (M : ℕ) : Prop :=
  let female_wages : ℕ := 15 * 20
  let child_wages : ℕ := 5 * 8
  let total_wages : ℕ := 35 * M + female_wages + child_wages
  let total_workers : ℕ := M + 15 + 5
  (total_wages / total_workers) = 26

theorem contractor_male_workers : ∃ M : ℕ, number_of_male_workers M ∧ M = 20 :=
by
  use 20
  sorry

end contractor_male_workers_l206_206893


namespace zero_of_function_is_not_intersection_l206_206775

noncomputable def is_function_zero (f : ℝ → ℝ) (x : ℝ) : Prop :=
  f x = 0

theorem zero_of_function_is_not_intersection (f : ℝ → ℝ) :
  ¬ (∀ x : ℝ, is_function_zero f x ↔ (f x = 0 ∧ x ∈ {x | f x = 0})) :=
by
  sorry

end zero_of_function_is_not_intersection_l206_206775


namespace not_perfect_square_4_2021_l206_206650

-- Define what it means for a number to be a perfect square
def is_perfect_square (n : ℕ) : Prop :=
  ∃ x : ℕ, n = x * x

-- State the non-perfect square problem for the given choices
theorem not_perfect_square_4_2021 :
  ¬ is_perfect_square (4 ^ 2021) ∧
  is_perfect_square (1 ^ 2018) ∧
  is_perfect_square (6 ^ 2020) ∧
  is_perfect_square (5 ^ 2022) :=
by
  sorry

end not_perfect_square_4_2021_l206_206650


namespace katya_sum_greater_than_masha_l206_206918

theorem katya_sum_greater_than_masha (a b c d : ℝ) (h : a + b + c + d = 2021) :
  ((a+1)*(b+1) + (b+1)*(c+1) + (c+1)*(d+1) + (d+1)*(a+1)) - (a*b + b*c + c*d + d*a) = 4046 := by
  sorry

end katya_sum_greater_than_masha_l206_206918


namespace factorize_expr_l206_206258

theorem factorize_expr (x : ℝ) : 75 * x^19 + 165 * x^38 = 15 * x^19 * (5 + 11 * x^19) := 
by
  sorry

end factorize_expr_l206_206258


namespace distance_between_parallel_lines_l206_206089

-- Definitions
def line_eq1 (x y : ℝ) : Prop := 3 * x - 4 * y - 12 = 0
def line_eq2 (x y : ℝ) : Prop := 6 * x - 8 * y + 11 = 0

-- Statement of the problem
theorem distance_between_parallel_lines :
  (∀ x y : ℝ, line_eq1 x y ↔ line_eq2 x y) →
  (∃ d : ℝ, d = 7 / 2) :=
by
  sorry

end distance_between_parallel_lines_l206_206089


namespace a_100_value_l206_206140

variables (S : ℕ → ℚ) (a : ℕ → ℚ)

def S_n (n : ℕ) : ℚ := S n
def a_n (n : ℕ) : ℚ := a n

axiom a1_eq_3 : a 1 = 3
axiom a_n_formula (n : ℕ) (hn : n ≥ 2) : a n = (3 * S n ^ 2) / (3 * S n - 2)

theorem a_100_value : a 100 = -3 / 88401 :=
sorry

end a_100_value_l206_206140


namespace arithmetic_sequence_fifth_term_l206_206214

variable (a d : ℕ)

-- Conditions
def condition1 := (a + d) + (a + 3 * d) = 10
def condition2 := a + (a + 2 * d) = 8

-- Fifth term calculation
def fifth_term := a + 4 * d

theorem arithmetic_sequence_fifth_term (h1 : condition1 a d) (h2 : condition2 a d) : fifth_term a d = 7 :=
by
  sorry

end arithmetic_sequence_fifth_term_l206_206214


namespace is_divisible_by_7_l206_206571

theorem is_divisible_by_7 : ∃ k : ℕ, 42 = 7 * k := by
  sorry

end is_divisible_by_7_l206_206571


namespace find_number_l206_206088

-- Define the conditions.
def condition (x : ℚ) : Prop := x - (1 / 3) * x = 16 / 3

-- Define the theorem from the translated (question, conditions, correct answer) tuple
theorem find_number : ∃ x : ℚ, condition x ∧ x = 8 :=
by
  sorry

end find_number_l206_206088


namespace measure_angle_BAC_l206_206910

-- Define the elements in the problem
def triangle (A B C : Type) := (A ≠ B) ∧ (B ≠ C) ∧ (C ≠ A)

-- Define the lengths and angles
variables {A B C X Y : Type}

-- Define the conditions given in the problem
def conditions (AX XY YB BC : ℝ) (angleABC : ℝ) : Prop :=
  AX = XY ∧ XY = YB ∧ YB = BC ∧ angleABC = 100

-- The Lean 4 statement (proof outline is not required)
theorem measure_angle_BAC {A B C X Y : Type} (hT : triangle A B C)
  (AX XY YB BC : ℝ) (angleABC : ℝ) (hC : conditions AX XY YB BC angleABC) :
  ∃ (t : ℝ), t = 25 :=
sorry
 
end measure_angle_BAC_l206_206910


namespace road_repair_completion_time_l206_206786

theorem road_repair_completion_time (L R r : ℕ) (hL : L = 100) (hR : R = 64) (hr : r = 9) :
  (L - R) / r = 5 :=
by
  sorry

end road_repair_completion_time_l206_206786


namespace monthlyShoeSales_l206_206343

-- Defining the conditions
def pairsSoldLastWeek := 27
def pairsSoldThisWeek := 12
def pairsNeededToMeetGoal := 41

-- Defining the question as a statement to prove
theorem monthlyShoeSales : pairsSoldLastWeek + pairsSoldThisWeek + pairsNeededToMeetGoal = 80 := by
  sorry

end monthlyShoeSales_l206_206343


namespace sufficient_but_not_necessary_condition_l206_206393

theorem sufficient_but_not_necessary_condition : ∀ (y : ℝ), (y = 2 → y^2 = 4) ∧ (y^2 = 4 → (y = 2 ∨ y = -2)) :=
by
  sorry

end sufficient_but_not_necessary_condition_l206_206393


namespace complement_union_result_l206_206183

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

end complement_union_result_l206_206183


namespace find_measure_A_and_b_c_sum_l206_206651

open Real

noncomputable def triangle_abc (a b c A B C : ℝ) : Prop :=
  ∀ (A B C : ℝ),
  A + B + C = π ∧
  a = sin A ∧
  b = sin B ∧
  c = sin C ∧
  cos (A - C) - cos (A + C) = sqrt 3 * sin C

theorem find_measure_A_and_b_c_sum (a b c A B C : ℝ)
  (h_triangle : triangle_abc a b c A B C) 
  (h_area : (1/2) * b * c * (sin A) = (3 * sqrt 3) / 16) 
  (h_b_def : b = sin B) :
  A = π / 3 ∧ b + c = sqrt 3 := by
  sorry

end find_measure_A_and_b_c_sum_l206_206651


namespace percent_of_a_is_20_l206_206689

variable {a b c : ℝ}

theorem percent_of_a_is_20 (h1 : c = (x / 100) * a)
                          (h2 : c = 0.1 * b)
                          (h3 : b = 2 * a) :
  c = 0.2 * a := sorry

end percent_of_a_is_20_l206_206689


namespace shaded_area_of_squares_is_20_l206_206325

theorem shaded_area_of_squares_is_20 :
  ∀ (a b : ℝ), a = 2 → b = 6 → 
    (1/2) * a * a + (1/2) * b * b = 20 :=
by
  intros a b ha hb
  rw [ha, hb]
  sorry

end shaded_area_of_squares_is_20_l206_206325


namespace batsman_average_after_11th_inning_l206_206201

theorem batsman_average_after_11th_inning (A : ℝ) 
  (h1 : A + 5 = (10 * A + 85) / 11) : A + 5 = 35 :=
by
  sorry

end batsman_average_after_11th_inning_l206_206201


namespace proof_AC_time_l206_206242

noncomputable def A : ℝ := 1/10
noncomputable def B : ℝ := sorry
noncomputable def C : ℝ := 1/30

def rate_A_B (A B : ℝ) := A + B = 1/6
def rate_B_C (B C : ℝ) := B + C = 1/10
def rate_A_B_C (A B C : ℝ) := A + B + C = 1/5

theorem proof_AC_time {A B C : ℝ} (h1 : rate_A_B A B) (h2 : rate_B_C B C) (h3 : rate_A_B_C A B C) : 
  (1 : ℝ) / (A + C) = 7.5 :=
sorry

end proof_AC_time_l206_206242


namespace common_ratio_of_geometric_sequence_l206_206797

theorem common_ratio_of_geometric_sequence 
  (a : ℝ) (log2_3 log4_3 log8_3: ℝ)
  (h1: log4_3 = log2_3 / 2)
  (h2: log8_3 = log2_3 / 3) 
  (h_geometric: ∀ i j, 
    i = a + log2_3 → 
    j = a + log4_3 →
    j / i = a + log8_3 / j / i / j
  ) :
  (a + log4_3) / (a + log2_3) = 1/3 :=
by
  sorry

end common_ratio_of_geometric_sequence_l206_206797


namespace linear_equation_must_be_neg2_l206_206277

theorem linear_equation_must_be_neg2 {m : ℝ} (h1 : |m| - 1 = 1) (h2 : m ≠ 2) : m = -2 :=
sorry

end linear_equation_must_be_neg2_l206_206277


namespace tourists_count_l206_206711

theorem tourists_count (n k : ℕ) (h1 : 2 * n % k = 1) (h2 : 3 * n % k = 13) : k = 23 := 
sorry

end tourists_count_l206_206711


namespace line_intercepts_l206_206953

-- Definitions
def point_on_axis (a b : ℝ) : Prop := a = b
def passes_through_point (a b : ℝ) (x y : ℝ) : Prop := a * x + b * y = 1

theorem line_intercepts (a b x y : ℝ) (hx : x = -1) (hy : y = 2) (intercept_property : point_on_axis a b) (point_property : passes_through_point a b x y) :
  (2 * x + y = 0) ∨ (x + y - 1 = 0) :=
sorry

end line_intercepts_l206_206953


namespace new_person_weight_l206_206108

-- Define the initial conditions
def initial_average_weight (w : ℕ) := 6 * w -- The total weight of 6 persons

-- Define the scenario where the average weight increases by 2 kg
def total_weight_increase := 6 * 2 -- The total increase in weight due to an increase of 2 kg in average weight

def person_replaced := 75 -- The weight of the person being replaced

-- Define the expected condition on the weight of the new person
theorem new_person_weight (w_new : ℕ) :
  initial_average_weight person_replaced + total_weight_increase = initial_average_weight (w_new / 6) →
  w_new = 87 :=
sorry

end new_person_weight_l206_206108


namespace time_to_cross_pole_l206_206722

-- Setting up the definitions
def speed_kmh : ℤ := 72
def length_m : ℤ := 180

-- Conversion function from km/hr to m/s
def convert_speed (v : ℤ) : ℚ :=
  v * (1000 : ℚ) / 3600

-- Given conditions in mathematics
def speed_ms : ℚ := convert_speed speed_kmh

-- Desired proposition
theorem time_to_cross_pole : 
  length_m / speed_ms = 9 := 
by
  -- Temporarily skipping the proof
  sorry

end time_to_cross_pole_l206_206722


namespace arkansas_tshirts_sold_l206_206954

theorem arkansas_tshirts_sold (A T : ℕ) (h1 : A + T = 163) (h2 : 98 * A = 8722) : A = 89 := by
  -- We state the problem and add 'sorry' to skip the actual proof
  sorry

end arkansas_tshirts_sold_l206_206954


namespace probability_all_girls_is_correct_l206_206517

noncomputable def probability_all_girls : ℚ :=
  let total_members := 15
  let boys := 7
  let girls := 8
  let choose_3_from_15 := Nat.choose total_members 3
  let choose_3_from_8 := Nat.choose girls 3
  choose_3_from_8 / choose_3_from_15

theorem probability_all_girls_is_correct : 
  probability_all_girls = 8 / 65 := by
sorry

end probability_all_girls_is_correct_l206_206517


namespace find_n_l206_206792

theorem find_n (n : ℕ) : (1 / (n + 1 : ℝ) + 2 / (n + 1 : ℝ) + (n + 1) / (n + 1 : ℝ) = 2) → (n = 2) :=
by
  sorry

end find_n_l206_206792


namespace number_of_factors_of_x_l206_206507

theorem number_of_factors_of_x (a b c : ℕ) (h1 : Nat.Prime a) (h2 : Nat.Prime b) (h3 : Nat.Prime c) (h4 : a < b) (h5 : b < c) (h6 : ¬ a = b) (h7 : ¬ b = c) (h8 : ¬ a = c) :
  let x := 2^2 * a^3 * b^2 * c^4
  let num_factors := (2 + 1) * (3 + 1) * (2 + 1) * (4 + 1)
  num_factors = 180 := by
sorry

end number_of_factors_of_x_l206_206507


namespace reciprocal_of_neg_1_point_5_l206_206055

theorem reciprocal_of_neg_1_point_5 : (1 / (-1.5) = -2 / 3) :=
by
  sorry

end reciprocal_of_neg_1_point_5_l206_206055


namespace quadratic_roots_l206_206446

theorem quadratic_roots (a b c : ℝ) :
  ∃ x y : ℝ, (x ≠ y ∧ (x^2 - (a + b) * x + (ab - c^2) = 0) ∧ (y^2 - (a + b) * y + (ab - c^2) = 0)) ∧
  (x = y ↔ a = b ∧ c = 0) := sorry

end quadratic_roots_l206_206446


namespace number_of_jerseys_sold_l206_206935

-- Definitions based on conditions
def revenue_per_jersey : ℕ := 115
def revenue_per_tshirt : ℕ := 25
def tshirts_sold : ℕ := 113
def jersey_cost_difference : ℕ := 90

-- Main condition: Prove the number of jerseys sold is 113
theorem number_of_jerseys_sold : ∀ (J : ℕ), 
  (revenue_per_jersey = revenue_per_tshirt + jersey_cost_difference) →
  (J * revenue_per_jersey = tshirts_sold * revenue_per_tshirt) →
  J = 113 :=
by
  intros J h1 h2
  sorry

end number_of_jerseys_sold_l206_206935


namespace inequality_solution_set_l206_206322

theorem inequality_solution_set :
  {x : ℝ | (x^2 + 2*x + 2) / (x + 2) > 1} = {x : ℝ | (-2 < x ∧ x < -1) ∨ (0 < x)} :=
sorry

end inequality_solution_set_l206_206322


namespace simplify_fraction_l206_206248

theorem simplify_fraction (num denom : ℚ) (h_num: num = (3/7 + 5/8)) (h_denom: denom = (5/12 + 2/3)) :
  (num / denom) = (177/182) := 
  sorry

end simplify_fraction_l206_206248


namespace sum_of_solutions_eq_zero_l206_206674

theorem sum_of_solutions_eq_zero (x : ℝ) :
  (∃ x_1 x_2 : ℝ, (|x_1 - 20| + |x_2 + 20| = 2020) ∧ (x_1 + x_2 = 0)) :=
sorry

end sum_of_solutions_eq_zero_l206_206674


namespace determine_M_l206_206639

theorem determine_M (x y z M : ℚ) 
  (h1 : x + y + z = 120)
  (h2 : x - 10 = M)
  (h3 : y + 10 = M)
  (h4 : 10 * z = M) : 
  M = 400 / 7 := 
sorry

end determine_M_l206_206639


namespace min_am_hm_l206_206756

theorem min_am_hm (a b : ℝ) (ha : a > 0) (hb : b > 0) : (a + b) * (1/a + 1/b) ≥ 4 :=
by sorry

end min_am_hm_l206_206756


namespace tip_percentage_correct_l206_206301

def lunch_cost := 50.20
def total_spent := 60.24
def tip_percentage := ((total_spent - lunch_cost) / lunch_cost) * 100

theorem tip_percentage_correct : tip_percentage = 19.96 := 
by
  sorry

end tip_percentage_correct_l206_206301


namespace Lewis_more_items_than_Samantha_l206_206863

def Tanya_items : ℕ := 4
def Samantha_items : ℕ := 4 * Tanya_items
def Lewis_items : ℕ := 20

theorem Lewis_more_items_than_Samantha : (Lewis_items - Samantha_items) = 4 := by
  sorry

end Lewis_more_items_than_Samantha_l206_206863


namespace lower_base_length_l206_206843

variable (A B C D E : Type)
variable (AD BD BE DE : ℝ)

-- Conditions of the problem
axiom hAD : AD = 12  -- upper base
axiom hBD : BD = 18  -- height
axiom hBE_DE : BE = 2 * DE  -- ratio BE = 2 * DE

-- Define the trapezoid with given lengths and conditions
def trapezoid_exists (A B C D : Type) (AD BD BE DE : ℝ) :=
  AD = 12 ∧ BD = 18 ∧ BE = 2 * DE

-- The length of BC to be proven
def BC : ℝ := 24

-- The theorem to be proven
theorem lower_base_length (h : trapezoid_exists A B C D AD BD BE DE) : BC = 2 * AD :=
by
  sorry

end lower_base_length_l206_206843


namespace number_of_lines_passing_through_four_points_l206_206771

-- Defining the three-dimensional points and conditions
structure Point3D where
  x : ℕ
  y : ℕ
  z : ℕ
  h1 : 1 ≤ x ∧ x ≤ 5
  h2 : 1 ≤ y ∧ y ≤ 5
  h3 : 1 ≤ z ∧ z ≤ 5

-- Define a valid line passing through four distinct points (Readonly accessors for the conditions)
def valid_line (p1 p2 p3 p4 : Point3D) : Prop := 
  sorry -- Define conditions for points to be collinear and distinct

-- Main theorem statement
theorem number_of_lines_passing_through_four_points : 
  ∃ (lines : ℕ), lines = 150 :=
sorry

end number_of_lines_passing_through_four_points_l206_206771


namespace cos_sum_seventh_root_of_unity_l206_206782

theorem cos_sum_seventh_root_of_unity (z : ℂ) (α : ℝ) 
  (h1 : z ^ 7 = 1) (h2 : z ≠ 1) (h3 : ∃ k : ℤ, α = (2 * k * π) / 7 ) :
  (Real.cos α + Real.cos (2 * α) + Real.cos (4 * α)) = -1 / 2 :=
by 
  sorry

end cos_sum_seventh_root_of_unity_l206_206782


namespace cone_volume_l206_206945

theorem cone_volume (slant_height : ℝ) (central_angle_deg : ℝ) (volume : ℝ) :
  slant_height = 1 ∧ central_angle_deg = 120 ∧ volume = (2 * Real.sqrt 2 / 81) * Real.pi →
  ∃ r h, h = Real.sqrt (slant_height^2 - r^2) ∧
    r = (1/3) ∧
    h = (2 * Real.sqrt 2 / 3) ∧
    volume = (1/3) * Real.pi * r^2 * h := 
by
  sorry

end cone_volume_l206_206945


namespace probability_of_at_least_one_accurate_forecast_l206_206123

theorem probability_of_at_least_one_accurate_forecast (PA PB : ℝ) (hA : PA = 0.8) (hB : PB = 0.75) :
  1 - ((1 - PA) * (1 - PB)) = 0.95 :=
by
  rw [hA, hB]
  sorry

end probability_of_at_least_one_accurate_forecast_l206_206123


namespace find_function_l206_206573

-- Let f be a differentiable function over all real numbers
variable (f : ℝ → ℝ)
variable (k : ℝ)

-- Condition: f is differentiable over (-∞, ∞)
variable (h_diff : differentiable ℝ f)

-- Condition: f(0) = 1
variable (h_init : f 0 = 1)

-- Condition: for any x1, x2 in ℝ, f(x1 + x2) ≥ f(x1) f(x2)
variable (h_ineq : ∀ x1 x2 : ℝ, f (x1 + x2) ≥ f x1 * f x2)

-- We aim to prove: f(x) = e^(kx)
theorem find_function : ∃ k : ℝ, ∀ x : ℝ, f x = Real.exp (k * x) :=
sorry

end find_function_l206_206573


namespace electricity_average_l206_206705

-- Define the daily electricity consumptions
def electricity_consumptions : List ℕ := [110, 101, 121, 119, 114]

-- Define the function to calculate the average
def average (l : List ℕ) : ℕ := l.sum / l.length

-- Formalize the proof problem
theorem electricity_average :
  average electricity_consumptions = 113 :=
  sorry

end electricity_average_l206_206705


namespace change_calculation_l206_206885

-- Define the initial amounts of Lee and his friend
def lee_amount : ℕ := 10
def friend_amount : ℕ := 8

-- Define the cost of items they ordered
def chicken_wings : ℕ := 6
def chicken_salad : ℕ := 4
def soda : ℕ := 1
def soda_count : ℕ := 2
def tax : ℕ := 3

-- Define the total money they initially had
def total_money : ℕ := lee_amount + friend_amount

-- Define the total cost of the food without tax
def food_cost : ℕ := chicken_wings + chicken_salad + (soda * soda_count)

-- Define the total cost including tax
def total_cost : ℕ := food_cost + tax

-- Define the change they should receive
def change : ℕ := total_money - total_cost

theorem change_calculation : change = 3 := by
  -- Note: Proof here is omitted
  sorry

end change_calculation_l206_206885


namespace tangent_line_parabola_k_l206_206017

theorem tangent_line_parabola_k :
  ∃ (k : ℝ), (∀ (x y : ℝ), 4 * x + 7 * y + k = 0 → y^2 = 16 * x → (28 ^ 2 = 4 * 1 * 4 * k)) → k = 49 :=
by
  sorry

end tangent_line_parabola_k_l206_206017


namespace problem1_part1_problem1_part2_problem2_l206_206511

-- Definitions
def quadratic (a b c x : ℝ) := a * x ^ 2 + b * x + c
def has_two_real_roots (a b c : ℝ) := b ^ 2 - 4 * a * c ≥ 0 
def neighboring_root_equation (a b c : ℝ) :=
  ∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ quadratic a b c x₁ = 0 ∧ quadratic a b c x₂ = 0 ∧ |x₁ - x₂| = 1

-- Proof problem 1: Prove whether x^2 + x - 6 = 0 is a neighboring root equation
theorem problem1_part1 : ¬ neighboring_root_equation 1 1 (-6) := 
sorry

-- Proof problem 2: Prove whether 2x^2 - 2√5x + 2 = 0 is a neighboring root equation
theorem problem1_part2 : neighboring_root_equation 2 (-2 * Real.sqrt 5) 2 := 
sorry

-- Proof problem 3: Prove that m = -1 or m = -3 for x^2 - (m-2)x - 2m = 0 to be a neighboring root equation
theorem problem2 (m : ℝ) (h : neighboring_root_equation 1 (-(m-2)) (-2*m)) : 
  m = -1 ∨ m = -3 := 
sorry

end problem1_part1_problem1_part2_problem2_l206_206511


namespace price_per_eraser_l206_206177

-- Definitions of the given conditions
def boxes_donated : ℕ := 48
def erasers_per_box : ℕ := 24
def total_money_made : ℝ := 864

-- The Lean statement to prove the price per eraser is $0.75
theorem price_per_eraser : (total_money_made / (boxes_donated * erasers_per_box) = 0.75) := by
  sorry

end price_per_eraser_l206_206177


namespace number_of_female_students_l206_206976

theorem number_of_female_students
    (F : ℕ)  -- Number of female students
    (avg_all : ℝ)  -- Average score for all students
    (avg_male : ℝ)  -- Average score for male students
    (avg_female : ℝ)  -- Average score for female students
    (num_male : ℕ)  -- Number of male students
    (h_avg_all : avg_all = 90)
    (h_avg_male : avg_male = 82)
    (h_avg_female : avg_female = 92)
    (h_num_male : num_male = 8)
    (h_avg : avg_all * (num_male + F) = avg_male * num_male + avg_female * F) :
  F = 32 :=
by
  sorry

end number_of_female_students_l206_206976


namespace total_cost_for_trip_l206_206132

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

end total_cost_for_trip_l206_206132


namespace sufficient_condition_for_reciprocal_inequality_l206_206441

variable (a b : ℝ)

theorem sufficient_condition_for_reciprocal_inequality 
  (h1 : a * b ≠ 0) (h2 : a < b) (h3 : b < 0) :
  1 / a^2 > 1 / b^2 :=
sorry

end sufficient_condition_for_reciprocal_inequality_l206_206441


namespace greg_sarah_apples_l206_206175

-- Definitions and Conditions
variable {G : ℕ}
variable (H0 : 2 * G + 2 * G + (2 * G - 5) = 49)

-- Statement of the problem
theorem greg_sarah_apples : 
  2 * G = 18 :=
by
  sorry

end greg_sarah_apples_l206_206175


namespace complex_fraction_simplification_l206_206387

theorem complex_fraction_simplification (a b c d : ℂ) (h₁ : a = 3 + i) (h₂ : b = 1 + i) (h₃ : c = 1 - i) (h₄ : d = 2 - i) : (a / b) = d := by
  sorry

end complex_fraction_simplification_l206_206387


namespace fuel_consumption_new_model_l206_206970

variable (d_old : ℝ) (d_new : ℝ) (c_old : ℝ) (c_new : ℝ)

theorem fuel_consumption_new_model :
  (d_new = d_old + 4.4) →
  (c_new = c_old - 2) →
  (c_old = 100 / d_old) →
  d_old = 12.79 →
  c_new = 5.82 :=
by
  intro h1 h2 h3 h4
  sorry

end fuel_consumption_new_model_l206_206970


namespace find_angle_A_correct_l206_206178

noncomputable def find_angle_A (BC AB angleC : ℝ) : ℝ :=
if BC = 3 ∧ AB = Real.sqrt 6 ∧ angleC = Real.pi / 4 then
  Real.pi / 3
else
  sorry

theorem find_angle_A_correct : find_angle_A 3 (Real.sqrt 6) (Real.pi / 4) = Real.pi / 3 :=
by
  -- proof goes here
  sorry

end find_angle_A_correct_l206_206178


namespace find_m_l206_206966

noncomputable def f (m : ℝ) (x : ℝ) : ℝ := (m^2 - m - 5) * x^(m + 1)

theorem find_m (m : ℝ) :
  (∀ x > 0, f m x < 0) → m = -2 := by
  sorry

end find_m_l206_206966


namespace division_of_powers_of_ten_l206_206829

theorem division_of_powers_of_ten :
  (10 ^ 0.7 * 10 ^ 0.4) / (10 ^ 0.2 * 10 ^ 0.6 * 10 ^ 0.3) = 1 := by
  sorry

end division_of_powers_of_ten_l206_206829


namespace abs_sum_lt_abs_diff_l206_206423

theorem abs_sum_lt_abs_diff (a b : ℝ) (h : a * b < 0) : |a + b| < |a - b| := 
sorry

end abs_sum_lt_abs_diff_l206_206423


namespace birds_remaining_on_fence_l206_206025

noncomputable def initial_birds : ℝ := 15.3
noncomputable def birds_flew_away : ℝ := 6.5
noncomputable def remaining_birds : ℝ := initial_birds - birds_flew_away

theorem birds_remaining_on_fence : remaining_birds = 8.8 :=
by
  -- sorry is a placeholder for the proof, which is not required
  sorry

end birds_remaining_on_fence_l206_206025


namespace find_age_of_15th_person_l206_206513

-- Define the conditions given in the problem
def total_age_of_18_persons (avg_18 : ℕ) (num_18 : ℕ) : ℕ := avg_18 * num_18
def total_age_of_5_persons (avg_5 : ℕ) (num_5 : ℕ) : ℕ := avg_5 * num_5
def total_age_of_9_persons (avg_9 : ℕ) (num_9 : ℕ) : ℕ := avg_9 * num_9

-- Define the overall question which is the age of the 15th person
def age_of_15th_person (total_18 : ℕ) (total_5 : ℕ) (total_9 : ℕ) : ℕ :=
  total_18 - total_5 - total_9

-- Statement of the theorem to prove
theorem find_age_of_15th_person :
  let avg_18 := 15
  let num_18 := 18
  let avg_5 := 14
  let num_5 := 5
  let avg_9 := 16
  let num_9 := 9
  let total_18 := total_age_of_18_persons avg_18 num_18 
  let total_5 := total_age_of_5_persons avg_5 num_5
  let total_9 := total_age_of_9_persons avg_9 num_9
  age_of_15th_person total_18 total_5 total_9 = 56 :=
by
  -- Definitions for the total ages
  let avg_18 := 15
  let num_18 := 18
  let avg_5 := 14
  let num_5 := 5
  let avg_9 := 16
  let num_9 := 9
  let total_18 := total_age_of_18_persons avg_18 num_18 
  let total_5 := total_age_of_5_persons avg_5 num_5
  let total_9 := total_age_of_9_persons avg_9 num_9
  
  -- Goal: compute the age of the 15th person
  let answer := age_of_15th_person total_18 total_5 total_9

  -- Prove that the computed age is equal to 56
  show answer = 56
  sorry

end find_age_of_15th_person_l206_206513


namespace money_left_after_purchase_l206_206068

def initial_money : ℕ := 7
def cost_candy_bar : ℕ := 2
def cost_chocolate : ℕ := 3

def total_spent : ℕ := cost_candy_bar + cost_chocolate
def money_left : ℕ := initial_money - total_spent

theorem money_left_after_purchase : 
  money_left = 2 := by
  sorry

end money_left_after_purchase_l206_206068


namespace sum_of_numbers_l206_206252

theorem sum_of_numbers (x y : ℕ) (h1 : x = 18) (h2 : y = 2 * x - 3) : x + y = 51 :=
by
  sorry

end sum_of_numbers_l206_206252


namespace greatest_possible_a_l206_206915

theorem greatest_possible_a (a : ℕ) :
  (∃ x : ℤ, x^2 + a * x = -28) → a = 29 :=
sorry

end greatest_possible_a_l206_206915


namespace nine_div_one_plus_four_div_x_eq_one_l206_206032

theorem nine_div_one_plus_four_div_x_eq_one (x : ℝ) (h : x = 0.5) : 9 / (1 + 4 / x) = 1 := by
  sorry

end nine_div_one_plus_four_div_x_eq_one_l206_206032


namespace transformed_equation_solutions_l206_206510

theorem transformed_equation_solutions :
  (∀ x : ℝ, x^2 + 2 * x - 3 = 0 → (x = 1 ∨ x = -3)) →
  (∀ x : ℝ, (x + 3)^2 + 2 * (x + 3) - 3 = 0 → (x = -2 ∨ x = -6)) :=
by
  intro h
  sorry

end transformed_equation_solutions_l206_206510


namespace notebook_ratio_l206_206549

theorem notebook_ratio (C N : ℕ) (h1 : ∀ k, N = k / C)
  (h2 : ∃ k, N = k / (C / 2) ∧ 16 = k / (C / 2))
  (h3 : C * N = 512) : (N : ℚ) / C = 1 / 8 := 
by
  sorry

end notebook_ratio_l206_206549


namespace coloring_scheme_formula_l206_206982

noncomputable def number_of_coloring_schemes (m n : ℕ) : ℕ :=
  if h : (m ≥ 2) ∧ (n ≥ 2) then
    m * ((-1 : ℤ)^n * (m - 2 : ℤ)).natAbs + (m - 2)^n
  else 0

-- Formal statement verifying the formula for coloring schemes
theorem coloring_scheme_formula (m n : ℕ) (h1 : m ≥ 2) (h2 : n ≥ 2) :
  number_of_coloring_schemes m n = m * ((-1 : ℤ)^n * (m - 2 : ℤ)).natAbs + (m - 2)^n :=
by sorry

end coloring_scheme_formula_l206_206982


namespace average_leaves_per_hour_l206_206326

theorem average_leaves_per_hour :
  let leaves_first_hour := 7
  let leaves_second_hour := 4
  let leaves_third_hour := 4
  let total_hours := 3
  let total_leaves := leaves_first_hour + leaves_second_hour + leaves_third_hour
  let average_leaves_per_hour := total_leaves / total_hours
  average_leaves_per_hour = 5 := by
  sorry

end average_leaves_per_hour_l206_206326


namespace Minjeong_family_juice_consumption_l206_206944

theorem Minjeong_family_juice_consumption :
  (∀ (amount_per_time : ℝ) (times_per_day : ℕ) (days_per_week : ℕ),
  amount_per_time = 0.2 → times_per_day = 3 → days_per_week = 7 → 
  amount_per_time * times_per_day * days_per_week = 4.2) :=
by
  intros amount_per_time times_per_day days_per_week h1 h2 h3
  rw [h1, h2, h3]
  norm_num

end Minjeong_family_juice_consumption_l206_206944


namespace constant_sequence_l206_206963

theorem constant_sequence (a : ℕ → ℕ) (h : ∀ i j, 1 ≤ i ∧ i ≤ 2016 ∧ 1 ≤ j ∧ j ≤ 2016 → (i + j) ∣ (i * a i + j * a j)) :
  ∀ i j, 1 ≤ i ∧ i ≤ 2016 ∧ 1 ≤ j ∧ j ≤ 2016 → a i = a j :=
by
  sorry

end constant_sequence_l206_206963


namespace gcd_polynomial_l206_206978

theorem gcd_polynomial (b : ℤ) (hb : ∃ k : ℤ, b = 570 * k) :
  Int.gcd (5 * b^3 + b^2 + 6 * b + 95) b = 95 :=
by
  sorry

end gcd_polynomial_l206_206978


namespace nails_for_smaller_planks_l206_206355

def total_large_planks := 13
def nails_per_plank := 17
def total_nails := 229

def nails_for_large_planks : ℕ :=
  total_large_planks * nails_per_plank

theorem nails_for_smaller_planks :
  total_nails - nails_for_large_planks = 8 :=
by
  -- Proof goes here
  sorry

end nails_for_smaller_planks_l206_206355


namespace simplify_and_evaluate_division_l206_206664

theorem simplify_and_evaluate_division (m : ℕ) (h : m = 10) : 
  (1 - (m / (m + 2))) / ((m^2 - 4 * m + 4) / (m^2 - 4)) = 1 / 4 :=
by sorry

end simplify_and_evaluate_division_l206_206664


namespace original_price_of_suit_l206_206215

theorem original_price_of_suit (P : ℝ) (h : 0.96 * P = 144) : P = 150 :=
sorry

end original_price_of_suit_l206_206215


namespace mary_change_l206_206931

def cost_of_berries : ℝ := 7.19
def cost_of_peaches : ℝ := 6.83
def amount_paid : ℝ := 20.00

theorem mary_change : amount_paid - (cost_of_berries + cost_of_peaches) = 5.98 := by
  sorry

end mary_change_l206_206931


namespace solve_equation_l206_206136

theorem solve_equation (x : ℝ) (h : (x^2 + x + 1) / (x + 1) = x + 2) : x = -1/2 :=
by sorry

end solve_equation_l206_206136


namespace surface_area_of_large_cube_is_486_cm_squared_l206_206459

noncomputable def surfaceAreaLargeCube : ℕ :=
  let small_box_count := 27
  let edge_small_box := 3
  let edge_large_cube := (small_box_count^(1/3)) * edge_small_box
  6 * edge_large_cube^2

theorem surface_area_of_large_cube_is_486_cm_squared :
  surfaceAreaLargeCube = 486 := 
sorry

end surface_area_of_large_cube_is_486_cm_squared_l206_206459


namespace union_M_N_l206_206454

def M : Set ℝ := { x | -1 < x ∧ x < 3 }
def N : Set ℝ := { x | x ≥ 1 }

theorem union_M_N : M ∪ N = { x | x > -1 } := 
by sorry

end union_M_N_l206_206454


namespace Jan_is_6_inches_taller_than_Bill_l206_206539

theorem Jan_is_6_inches_taller_than_Bill :
  ∀ (Cary Bill Jan : ℕ),
    Cary = 72 →
    Bill = Cary / 2 →
    Jan = 42 →
    Jan - Bill = 6 :=
by
  intros
  sorry

end Jan_is_6_inches_taller_than_Bill_l206_206539


namespace total_animals_l206_206134

theorem total_animals (B : ℕ) (h1 : 4 * B + 8 = 44) : B + 4 = 13 := by
  sorry

end total_animals_l206_206134


namespace number_of_green_balls_l206_206430

theorem number_of_green_balls
  (total_balls white_balls yellow_balls red_balls purple_balls : ℕ)
  (prob : ℚ)
  (H_total : total_balls = 100)
  (H_white : white_balls = 50)
  (H_yellow : yellow_balls = 10)
  (H_red : red_balls = 7)
  (H_purple : purple_balls = 3)
  (H_prob : prob = 0.9) :
  ∃ (green_balls : ℕ), 
    (white_balls + green_balls + yellow_balls) / total_balls = prob ∧ green_balls = 30 := by
  sorry

end number_of_green_balls_l206_206430


namespace initial_marbles_count_l206_206700

-- Definitions as per conditions in the problem
variables (x y z : ℕ)

-- Condition 1: Removing one black marble results in one-eighth of the remaining marbles being black
def condition1 : Prop := (x - 1) * 8 = (x + y - 1)

-- Condition 2: Removing three white marbles results in one-sixth of the remaining marbles being black
def condition2 : Prop := x * 6 = (x + y - 3)

-- Proof that initial total number of marbles is 9 given conditions
theorem initial_marbles_count (h1 : condition1 x y) (h2 : condition2 x y) : x + y = 9 :=
by 
  sorry

end initial_marbles_count_l206_206700


namespace minimum_pipe_length_l206_206071

theorem minimum_pipe_length 
  (M S : ℝ × ℝ) 
  (horiz_dist : abs (M.1 - S.1) = 160)
  (vert_dist : abs (M.2 - S.2) = 120) :
  dist M S = 200 :=
by {
  sorry
}

end minimum_pipe_length_l206_206071


namespace part1_part2_l206_206707

open Set

def A (x : ℝ) : Prop := -1 < x ∧ x < 6
def B (x : ℝ) (a : ℝ) : Prop := x ≥ 1 + a ∨ x ≤ 1 - a

theorem part1 (a : ℝ) (hpos : 0 < a) :
  (∀ x, A x → ¬ B x a) ↔ a ≥ 5 :=
sorry

theorem part2 (a : ℝ) (hpos : 0 < a) :
  (∀ x, (¬ A x → B x a) ∧ ∃ x, ¬ A x ∧ ¬ B x a) ↔ (0 < a ∧ a ≤ 2) :=
sorry

end part1_part2_l206_206707


namespace angle_B_sum_a_c_l206_206036

theorem angle_B (a b c : ℝ) (A B C : ℝ) (T : ℝ)
  (h1 : 0 < A) (h2 : A < π)
  (h3 : 0 < B) (h4 : B < π)
  (h5 : 0 < C) (h6 : C < π)
  (h7 : (2 * a - c) * Real.cos B = b * Real.cos C)
  (h8 : b = Real.sqrt 3)
  (h9 : T = 1/2 * a * c * Real.sin B) :
  B = π / 3 :=
  sorry

theorem sum_a_c (a b c : ℝ) (A B C : ℝ) (T : ℝ)
  (h1 : 0 < A) (h2 : A < π)
  (h3 : 0 < B) (h4 : B < π)
  (h5 : 0 < C) (h6 : C < π)
  (h7 : (2 * a - c) * Real.cos B = b * Real.cos C)
  (h8 : b = Real.sqrt 3)
  (h9 : T = 1/2 * a * c * Real.sin B)
  (hB : B = π / 3) :
  a + c = Real.sqrt 15 :=
  sorry

end angle_B_sum_a_c_l206_206036


namespace parallel_resistance_example_l206_206600

theorem parallel_resistance_example :
  ∀ (R1 R2 : ℕ), R1 = 3 → R2 = 6 → 1 / (R : ℚ) = 1 / (R1 : ℚ) + 1 / (R2 : ℚ) → R = 2 := by
  intros R1 R2 hR1 hR2 h_formula
  -- Formulation of the resistance equations and assumptions
  sorry

end parallel_resistance_example_l206_206600


namespace playground_perimeter_l206_206533

theorem playground_perimeter (x y : ℝ) 
  (h1 : x^2 + y^2 = 289) 
  (h2 : x * y = 120) : 
  2 * (x + y) = 46 :=
by 
  sorry

end playground_perimeter_l206_206533


namespace imaginary_unit_cube_l206_206636

theorem imaginary_unit_cube (i : ℂ) (h : i^2 = -1) : 1 + i^3 = 1 - i :=
by
  sorry

end imaginary_unit_cube_l206_206636


namespace smallest_n_l206_206499

/--
Each of \( 2020 \) boxes in a line contains 2 red marbles, 
and for \( 1 \le k \le 2020 \), the box in the \( k \)-th 
position also contains \( k \) white marbles. 

Let \( Q(n) \) be the probability that James stops after 
drawing exactly \( n \) marbles. Prove that the smallest 
value of \( n \) for which \( Q(n) < \frac{1}{2020} \) 
is 31.
-/
theorem smallest_n (Q : ℕ → ℚ) (hQ : ∀ n, Q n = (2 : ℚ) / ((n + 1) * (n + 2)))
  : ∃ n, Q n < 1/2020 ∧ ∀ m < n, Q m ≥ 1/2020 := by
  sorry

end smallest_n_l206_206499


namespace books_taken_out_on_Tuesday_l206_206578

theorem books_taken_out_on_Tuesday (T : ℕ) (initial_books : ℕ) (returned_books : ℕ) (withdrawn_books : ℕ) (final_books : ℕ) :
  initial_books = 250 ∧
  returned_books = 35 ∧
  withdrawn_books = 15 ∧
  final_books = 150 →
  T = 120 :=
by
  sorry

end books_taken_out_on_Tuesday_l206_206578


namespace range_of_m_l206_206166

theorem range_of_m (m x : ℝ) : (m-1 < x ∧ x < m+1) → (1/3 < x ∧ x < 1/2) → (-1/2 ≤ m ∧ m ≤ 4/3) :=
by
  intros h1 h2
  have h3 : 1/3 < m + 1 := by sorry
  have h4 : m - 1 < 1/2 := by sorry
  have h5 : -1/2 ≤ m := by sorry
  have h6 : m ≤ 4/3 := by sorry
  exact ⟨h5, h6⟩

end range_of_m_l206_206166


namespace num_of_dogs_l206_206541

theorem num_of_dogs (num_puppies : ℕ) (dog_food_per_meal : ℕ) (dog_meals_per_day : ℕ) (total_food : ℕ)
  (h1 : num_puppies = 4)
  (h2 : dog_food_per_meal = 4)
  (h3 : dog_meals_per_day = 3)
  (h4 : total_food = 108)
  : ∃ (D : ℕ), num_puppies * (dog_food_per_meal / 2) * (dog_meals_per_day * 3) + D * (dog_food_per_meal * dog_meals_per_day) = total_food ∧ D = 3 :=
by
  sorry

end num_of_dogs_l206_206541


namespace max_light_window_l206_206585

noncomputable def max_window_light : Prop :=
  ∃ (x : ℝ), (4 - 2 * x) / 3 * x = -2 / 3 * (x - 1) ^ 2 + 2 / 3 ∧ x = 1 ∧ (4 - 2 * x) / 3 = 2 / 3

theorem max_light_window : max_window_light :=
by
  sorry

end max_light_window_l206_206585


namespace find_principal_amount_l206_206715

-- Define the conditions as constants and assumptions
def monthly_interest_payment : ℝ := 216
def annual_interest_rate : ℝ := 0.09

-- Define the Lean statement to show that the amount of the investment is 28800
theorem find_principal_amount (monthly_payment : ℝ) (annual_rate : ℝ) (P : ℝ) :
  monthly_payment = 216 →
  annual_rate = 0.09 →
  P = 28800 :=
by
  intros 
  sorry

end find_principal_amount_l206_206715


namespace A_work_days_l206_206317

theorem A_work_days {total_wages B_share : ℝ} (B_work_days : ℝ) (total_wages_eq : total_wages = 5000) 
    (B_share_eq : B_share = 3333) (B_rate : ℝ) (correct_rate : B_rate = 1 / B_work_days) :
    ∃x : ℝ, B_share / (total_wages - B_share) = B_rate / (1 / x) ∧ total_wages - B_share = 5000 - B_share ∧ B_work_days = 10 -> x = 20 :=
by
  sorry

end A_work_days_l206_206317


namespace evaluate_f_at_2_l206_206702

def f (x : ℝ) : ℝ := x^3 - 3 * x + 2

theorem evaluate_f_at_2 : f 2 = 4 :=
by
  -- Proof goes here
  sorry

end evaluate_f_at_2_l206_206702


namespace smallest_number_of_students_l206_206360

theorem smallest_number_of_students (a b c : ℕ) (h1 : 4 * c = 3 * a) (h2 : 7 * b = 5 * a) (h3 : 10 * c = 9 * b) : a + b + c = 66 := sorry

end smallest_number_of_students_l206_206360


namespace cos_diff_trigonometric_identity_l206_206774

-- Problem 1
theorem cos_diff :
  (Real.cos (25 * Real.pi / 180) * Real.cos (35 * Real.pi / 180) - 
   Real.cos (65 * Real.pi / 180) * Real.cos (55 * Real.pi / 180)) = 
  1/2 :=
sorry

-- Problem 2
theorem trigonometric_identity (θ : Real) (h : Real.sin θ + 2 * Real.cos θ = 0) :
  (Real.cos (2 * θ) - Real.sin (2 * θ)) / (1 + (Real.cos θ)^2) = 5/6 :=
sorry

end cos_diff_trigonometric_identity_l206_206774


namespace distance_between_trees_l206_206680

theorem distance_between_trees (yard_length : ℕ) (num_trees : ℕ)
  (h_yard : yard_length = 400) (h_trees : num_trees = 26) : 
  (yard_length / (num_trees - 1)) = 16 :=
by
  sorry

end distance_between_trees_l206_206680


namespace wrapping_paper_per_present_l206_206745

theorem wrapping_paper_per_present :
  let sum_paper := 1 / 2
  let num_presents := 5
  (sum_paper / num_presents) = 1 / 10 := by
  sorry

end wrapping_paper_per_present_l206_206745


namespace train_speed_l206_206421

theorem train_speed (length : ℝ) (time_seconds : ℝ) (speed : ℝ) :
  length = 320 → time_seconds = 16 → speed = 72 :=
by 
  sorry

end train_speed_l206_206421


namespace sum_of_roots_eq_36_l206_206819

theorem sum_of_roots_eq_36 :
  (∃ x1 x2 x3 : ℝ, (11 - x1) ^ 3 + (13 - x2) ^ 3 = (24 - 2 * x3) ^ 3 ∧ 
  (11 - x2) ^ 3 + (13 - x3) ^ 3 = (24 - 2 * x1) ^ 3 ∧ 
  (11 - x3) ^ 3 + (13 - x1) ^ 3 = (24 - 2 * x2) ^ 3 ∧
  x1 + x2 + x3 = 36) :=
sorry

end sum_of_roots_eq_36_l206_206819


namespace quotient_of_sum_l206_206480

theorem quotient_of_sum (a b c x y z : ℝ)
  (h1 : a^2 + b^2 + c^2 = 25)
  (h2 : x^2 + y^2 + z^2 = 36)
  (h3 : a * x + b * y + c * z = 30) :
  (a + b + c) / (x + y + z) = 5 / 6 :=
by
  sorry

end quotient_of_sum_l206_206480


namespace unique_solution_arith_prog_system_l206_206630

theorem unique_solution_arith_prog_system (x y : ℝ) : 
  (6 * x + 9 * y = 12) ∧ (15 * x + 18 * y = 21) ↔ (x = -1) ∧ (y = 2) :=
by sorry

end unique_solution_arith_prog_system_l206_206630


namespace ratio_is_one_half_l206_206065

noncomputable def ratio_dresses_with_pockets (D : ℕ) (total_pockets : ℕ) (pockets_two : ℕ) (pockets_three : ℕ) :=
  ∃ (P : ℕ), D = 24 ∧ total_pockets = 32 ∧
  (P / 3) * 2 + (2 * P / 3) * 3 = total_pockets ∧ 
  P / D = 1 / 2

theorem ratio_is_one_half :
  ratio_dresses_with_pockets 24 32 2 3 :=
by 
  sorry

end ratio_is_one_half_l206_206065


namespace reduction_in_consumption_l206_206758

def rate_last_month : ℝ := 16
def rate_current : ℝ := 20
def initial_consumption (X : ℝ) : ℝ := X

theorem reduction_in_consumption (X : ℝ) : initial_consumption X - (initial_consumption X * rate_last_month / rate_current) = initial_consumption X * 0.2 :=
by
  sorry

end reduction_in_consumption_l206_206758


namespace ratio_Lisa_Charlotte_l206_206063

def P_tot : ℕ := 100
def Pat_money : ℕ := 6
def Lisa_money : ℕ := 5 * Pat_money
def additional_required : ℕ := 49
def current_total_money : ℕ := P_tot - additional_required
def Pat_Lisa_total : ℕ := Pat_money + Lisa_money
def Charlotte_money : ℕ := current_total_money - Pat_Lisa_total

theorem ratio_Lisa_Charlotte : (Lisa_money : ℕ) / Charlotte_money = 2 :=
by
  -- Proof to be filled in later
  sorry

end ratio_Lisa_Charlotte_l206_206063


namespace john_books_nights_l206_206381

theorem john_books_nights (n : ℕ) (cost_per_night discount amount_paid : ℕ) 
  (h1 : cost_per_night = 250)
  (h2 : discount = 100)
  (h3 : amount_paid = 650)
  (h4 : amount_paid = cost_per_night * n - discount) : 
  n = 3 :=
by
  sorry

end john_books_nights_l206_206381


namespace union_M_N_l206_206739

namespace MyMath

def M : Set ℝ := {x | x^2 = 1}
def N : Set ℝ := {1, 2}

theorem union_M_N : M ∪ N = {-1, 1, 2} := sorry

end MyMath

end union_M_N_l206_206739


namespace longest_side_eq_24_l206_206153

noncomputable def x : Real := 19 / 3

def side1 (x : Real) : Real := x + 3
def side2 (x : Real) : Real := 2 * x - 1
def side3 (x : Real) : Real := 3 * x + 5

def perimeter (x : Real) : Prop :=
  side1 x + side2 x + side3 x = 45

theorem longest_side_eq_24 : perimeter x → max (max (side1 x) (side2 x)) (side3 x) = 24 :=
by
  sorry

end longest_side_eq_24_l206_206153


namespace trajectory_of_point_inside_square_is_conic_or_degenerates_l206_206052

noncomputable def is_conic_section (a : ℝ) (P : ℝ × ℝ) : Prop :=
  ∃ (m n l : ℝ) (x y : ℝ), 
    x = P.1 ∧ y = P.2 ∧ 
    (m^2 + n^2) * x^2 - 2 * n * (l + m) * x * y + (l^2 + n^2) * y^2 = (l * m - n^2)^2 ∧
    4 * n^2 * (l + m)^2 - 4 * (m^2 + n^2) * (l^2 + n^2) ≤ 0

theorem trajectory_of_point_inside_square_is_conic_or_degenerates
  (a : ℝ) (P : ℝ × ℝ)
  (h1 : 0 < P.1) (h2 : P.1 < 2 * a)
  (h3 : 0 < P.2) (h4 : P.2 < 2 * a)
  : is_conic_section a P :=
sorry

end trajectory_of_point_inside_square_is_conic_or_degenerates_l206_206052


namespace percentage_increase_l206_206342

variable (x y p : ℝ)

theorem percentage_increase (h : x = y + (p / 100) * y) : p = 100 * ((x - y) / y) := 
by 
  sorry

end percentage_increase_l206_206342


namespace complement_union_example_l206_206703

open Set

universe u

variable (U : Set ℕ) (A B : Set ℕ)

def U_def : Set ℕ := {0, 1, 2, 3, 4}
def A_def : Set ℕ := {0, 1, 2}
def B_def : Set ℕ := {2, 3}

theorem complement_union_example :
  (U \ A) ∪ B = {2, 3, 4} := 
by
  -- Proving the theorem considering
  -- complement and union operations on sets
  sorry

end complement_union_example_l206_206703


namespace weight_of_daughter_l206_206977

def mother_daughter_grandchild_weight (M D C : ℝ) :=
  M + D + C = 130 ∧
  D + C = 60 ∧
  C = 1/5 * M

theorem weight_of_daughter (M D C : ℝ) 
  (h : mother_daughter_grandchild_weight M D C) : D = 46 :=
by
  obtain ⟨h1, h2, h3⟩ := h
  sorry

end weight_of_daughter_l206_206977


namespace fraction_identity_proof_l206_206229

theorem fraction_identity_proof (a b : ℝ) (h : 2 / a - 1 / b = 1 / (a + 2 * b)) :
  4 / (a ^ 2) - 1 / (b ^ 2) = 1 / (a * b) :=
by
  sorry

end fraction_identity_proof_l206_206229


namespace robins_fraction_l206_206840

theorem robins_fraction (B R J : ℕ) (h1 : R + J = B)
  (h2 : 2/3 * (R : ℚ) + 1/3 * (J : ℚ) = 7/15 * (B : ℚ)) :
  (R : ℚ) / B = 2/5 :=
by
  sorry

end robins_fraction_l206_206840


namespace isosceles_triangle_base_length_l206_206092

theorem isosceles_triangle_base_length
  (perimeter : ℝ)
  (side1 side2 base : ℝ)
  (h_perimeter : perimeter = 18)
  (h_side1 : side1 = 4)
  (h_isosceles : side1 = side2 ∨ side1 = base ∨ side2 = base)
  (h_triangle : side1 + side2 + base = 18) :
  base = 7 := 
sorry

end isosceles_triangle_base_length_l206_206092


namespace probability_both_correct_given_any_correct_l206_206905

-- Defining the probabilities
def P_A : ℚ := 3 / 5
def P_B : ℚ := 1 / 3

-- Defining the events and their products
def P_AnotB : ℚ := P_A * (1 - P_B)
def P_notAB : ℚ := (1 - P_A) * P_B
def P_AB : ℚ := P_A * P_B

-- Calculated Probability of C
def P_C : ℚ := P_AnotB + P_notAB + P_AB

-- The proof statement
theorem probability_both_correct_given_any_correct : (P_AB / P_C) = 3 / 11 :=
by
  sorry

end probability_both_correct_given_any_correct_l206_206905


namespace original_population_has_factor_three_l206_206972

theorem original_population_has_factor_three (x y z : ℕ) 
  (hx : ∃ n : ℕ, x = n ^ 2) -- original population is a perfect square
  (h1 : x + 150 = y^2 - 1)  -- after increase of 150, population is one less than a perfect square
  (h2 : y^2 - 1 + 150 = z^2) -- after another increase of 150, population is a perfect square again
  : 3 ∣ x :=
sorry

end original_population_has_factor_three_l206_206972


namespace Mitzi_leftover_money_l206_206655

def A := 75
def T := 30
def F := 13
def S := 23
def R := A - (T + F + S)

theorem Mitzi_leftover_money : R = 9 := by
  sorry

end Mitzi_leftover_money_l206_206655


namespace mary_has_10_blue_marbles_l206_206400

-- Define the number of blue marbles Dan has
def dan_marbles : ℕ := 5

-- Define the factor by which Mary has more blue marbles than Dan
def factor : ℕ := 2

-- Define the number of blue marbles Mary has
def mary_marbles : ℕ := factor * dan_marbles

-- The theorem statement: Mary has 10 blue marbles
theorem mary_has_10_blue_marbles : mary_marbles = 10 :=
by
  -- Proof goes here
  sorry

end mary_has_10_blue_marbles_l206_206400


namespace perc_freshmen_in_SLA_l206_206297

variables (T : ℕ) (P : ℝ)

-- 60% of students are freshmen
def freshmen (T : ℕ) : ℝ := 0.60 * T

-- 4.8% of students are freshmen psychology majors in the school of liberal arts
def freshmen_psych_majors (T : ℕ) : ℝ := 0.048 * T

-- 20% of freshmen in the school of liberal arts are psychology majors
def perc_fresh_psych (F_LA : ℝ) : ℝ := 0.20 * F_LA

-- Number of freshmen in the school of liberal arts as a percentage P of the total number of freshmen
def fresh_in_SLA_as_perc (T : ℕ) (P : ℝ) : ℝ := P * (0.60 * T)

theorem perc_freshmen_in_SLA (T : ℕ) (P : ℝ) :
  (0.20 * (P * (0.60 * T)) = 0.048 * T) → P = 0.4 :=
sorry

end perc_freshmen_in_SLA_l206_206297


namespace find_number_l206_206574

theorem find_number (x : ℝ) (h : 54 / 2 + 3 * x = 75) : x = 16 :=
by
  sorry

end find_number_l206_206574


namespace probability_of_three_cards_l206_206443

-- Conditions
def deck_size : ℕ := 52
def spades : ℕ := 13
def spades_face_cards : ℕ := 3
def face_cards : ℕ := 12
def diamonds : ℕ := 13

-- Probability of drawing specific cards
def prob_first_spade_non_face : ℚ := 10 / 52
def prob_second_face_given_first_spade_non_face : ℚ := 12 / 51
def prob_third_diamond_given_first_two : ℚ := 13 / 50

def prob_first_spade_face : ℚ := 3 / 52
def prob_second_face_given_first_spade_face : ℚ := 9 / 51

-- Final probability
def final_probability := 
  (prob_first_spade_non_face * prob_second_face_given_first_spade_non_face * prob_third_diamond_given_first_two) +
  (prob_first_spade_face * prob_second_face_given_first_spade_face * prob_third_diamond_given_first_two)

theorem probability_of_three_cards :
  final_probability = 1911 / 132600 := 
by
  sorry

end probability_of_three_cards_l206_206443


namespace simplify_expression_l206_206644

theorem simplify_expression : 
  (1 / (1 / (1 / 2)^0 + 1 / (1 / 2)^1 + 1 / (1 / 2)^2 + 1 / (1 / 2)^3)) = 1 / 15 :=
by 
  sorry

end simplify_expression_l206_206644


namespace digits_base_d_l206_206411

theorem digits_base_d (d A B : ℕ) (h₀ : d > 7) (h₁ : A < d) (h₂ : B < d) 
  (h₃ : A * d + B + B * d + A = 2 * d^2 + 2) : A - B = 2 :=
by
  sorry

end digits_base_d_l206_206411


namespace polynomial_identity_l206_206232

theorem polynomial_identity
  (x a a_1 a_2 a_3 a_4 a_5 a_6 a_7 : ℝ)
  (h : (x - 1)^7 = a + a_1 * x + a_2 * x^2 + a_3 * x^3 + a_4 * x^4 + a_5 * x^5 + a_6 * x^6 + a_7 * x^7) :
  (a + a_2 + a_4 + a_6)^2 - (a_1 + a_3 + a_5 + a_7)^2 = 0 :=
by sorry

end polynomial_identity_l206_206232


namespace not_prime_41_squared_plus_41_plus_41_l206_206516

def is_prime (n : ℕ) : Prop := ∀ m k : ℕ, m * k = n → m = 1 ∨ k = 1

theorem not_prime_41_squared_plus_41_plus_41 :
  ¬ is_prime (41^2 + 41 + 41) :=
by {
  sorry
}

end not_prime_41_squared_plus_41_plus_41_l206_206516


namespace distinguishable_squares_count_l206_206181

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

end distinguishable_squares_count_l206_206181


namespace bonus_distributed_correctly_l206_206146

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

end bonus_distributed_correctly_l206_206146


namespace case_m_eq_neg_1_case_m_gt_neg_1_case_m_lt_neg_1_l206_206234

noncomputable def solution_set (m x : ℝ) : Prop :=
  x^2 + (m-1) * x - m > 0

theorem case_m_eq_neg_1 (x : ℝ) :
  solution_set (-1) x ↔ x ≠ 1 :=
sorry

theorem case_m_gt_neg_1 (m x : ℝ) (hm : m > -1) :
  solution_set m x ↔ (x < -m ∨ x > 1) :=
sorry

theorem case_m_lt_neg_1 (m x : ℝ) (hm : m < -1) :
  solution_set m x ↔ (x < 1 ∨ x > -m) :=
sorry

end case_m_eq_neg_1_case_m_gt_neg_1_case_m_lt_neg_1_l206_206234


namespace largest_three_digit_number_with_7_in_hundreds_l206_206010

def is_three_digit_number_with_7_in_hundreds (n : ℕ) : Prop := 
  100 ≤ n ∧ n < 1000 ∧ (n / 100) = 7

theorem largest_three_digit_number_with_7_in_hundreds : 
  ∀ (n : ℕ), is_three_digit_number_with_7_in_hundreds n → n ≤ 799 :=
by sorry

end largest_three_digit_number_with_7_in_hundreds_l206_206010


namespace remainder_when_dividing_l206_206349

theorem remainder_when_dividing (c d : ℕ) (p q : ℕ) :
  c = 60 * p + 47 ∧ d = 45 * q + 14 → (c + d) % 15 = 1 :=
by
  sorry

end remainder_when_dividing_l206_206349


namespace min_value_x_fraction_l206_206202

theorem min_value_x_fraction (x : ℝ) (h : x > 1) : 
  ∃ m, m = 3 ∧ ∀ y > 1, y + 1 / (y - 1) ≥ m :=
by
  sorry

end min_value_x_fraction_l206_206202


namespace beavers_help_l206_206264

theorem beavers_help (initial final : ℝ) (h_initial : initial = 2.0) (h_final : final = 3) : final - initial = 1 :=
  by
    sorry

end beavers_help_l206_206264


namespace gcd_a_b_l206_206512

noncomputable def a : ℕ := 3333333
noncomputable def b : ℕ := 666666666

theorem gcd_a_b : Nat.gcd a b = 3 := by
  sorry

end gcd_a_b_l206_206512


namespace monotonically_decreasing_interval_l206_206855

-- Given conditions
def f (x : ℝ) : ℝ := x^2 * (x - 3)

-- The proof problem statement
theorem monotonically_decreasing_interval :
  ∃ a b : ℝ, (0 ≤ a) ∧ (b ≤ 2) ∧ (∀ x : ℝ, a ≤ x ∧ x ≤ b → (deriv f x ≤ 0)) :=
sorry

end monotonically_decreasing_interval_l206_206855


namespace expression_value_l206_206483

theorem expression_value : (25 + 15)^2 - (25^2 + 15^2) = 750 := by
  sorry

end expression_value_l206_206483


namespace expression_min_value_l206_206200

theorem expression_min_value (a b c k : ℝ) (h1 : a < c) (h2 : c < b) (h3 : b = k * c) (h4 : k > 1) :
  (1 : ℝ) / c^2 * ((k * c - a)^2 + (a + c)^2 + (c - a)^2) ≥ k^2 / 3 + 2 :=
sorry

end expression_min_value_l206_206200


namespace simplify_expression_l206_206647

theorem simplify_expression (q : ℤ) : 
  (((7 * q - 2) + 2 * q * 3) * 4 + (5 + 2 / 2) * (4 * q - 6)) = 76 * q - 44 := by
  sorry

end simplify_expression_l206_206647


namespace mean_of_all_students_l206_206773

theorem mean_of_all_students (M A : ℕ) (m a : ℕ) (hM : M = 88) (hA : A = 68) (hRatio : m * 5 = 2 * a) : 
  (176 * a + 340 * a) / (7 * a) = 74 :=
by sorry

end mean_of_all_students_l206_206773


namespace notebook_cost_l206_206186

open Nat

theorem notebook_cost
  (s : ℕ) (c : ℕ) (n : ℕ)
  (h_majority : s > 21)
  (h_notebooks : n > 2)
  (h_cost : c > n)
  (h_total : s * c * n = 2773) : c = 103 := 
sorry

end notebook_cost_l206_206186


namespace sequence_infinite_integers_l206_206913

theorem sequence_infinite_integers (x : ℕ → ℝ) (x1 x2 : ℝ) 
  (h1 : x 1 = x1) 
  (h2 : x 2 = x2) 
  (h3 : ∀ n ≥ 3, x n = x (n - 2) * x (n - 1) / (2 * x (n - 2) - x (n - 1))) : 
  (∃ k : ℤ, x1 = k ∧ x2 = k) ↔ (∀ n, ∃ m : ℤ, x n = m) :=
sorry

end sequence_infinite_integers_l206_206913


namespace order_of_f_l206_206399

variable (f : ℝ → ℝ)

/-- Conditions:
1. f is an even function for all x ∈ ℝ
2. f is increasing on [0, +∞)
Question:
Prove that the order of f(-2), f(-π), f(3) is f(-2) < f(3) < f(-π) -/
theorem order_of_f (h_even : ∀ x : ℝ, f (-x) = f x)
                   (h_incr : ∀ {x y : ℝ}, 0 ≤ x → x ≤ y → f x ≤ f y) : 
                   f (-2) < f 3 ∧ f 3 < f (-π) :=
by
  sorry

end order_of_f_l206_206399


namespace sum_of_g1_l206_206563

-- Define the main conditions
variable {g : ℝ → ℝ}
variable (h_nonconst : ∀ a b : ℝ, a ≠ b → g a ≠ g b)
axiom main_condition : ∀ x : ℝ, x ≠ 0 → g (x - 1) + g x + g (x + 1) = (g x) ^ 2 / (2025 * x)

-- Define the goal
theorem sum_of_g1 :
  g 1 = 6075 :=
sorry

end sum_of_g1_l206_206563


namespace sum_of_edges_l206_206304

theorem sum_of_edges (a r : ℝ) 
  (h_vol : (a / r) * a * (a * r) = 432) 
  (h_surf_area : 2 * ((a * a) / r + (a * a) * r + a * a) = 384) 
  (h_geom_prog : r ≠ 1) :
  4 * ((6 * Real.sqrt 2) / r + 6 * Real.sqrt 2 + (6 * Real.sqrt 2) * r) = 72 * (Real.sqrt 2) := 
sorry

end sum_of_edges_l206_206304


namespace moles_of_NaCl_formed_l206_206471

-- Define the conditions
def moles_NaOH : ℕ := 3
def moles_HCl : ℕ := 3

-- Define the balanced chemical equation as a relation
def reaction (NaOH HCl NaCl H2O : ℕ) : Prop :=
  NaOH = HCl ∧ HCl = NaCl ∧ H2O = NaCl

-- Define the proof problem
theorem moles_of_NaCl_formed :
  ∀ (NaOH HCl NaCl H2O : ℕ), NaOH = 3 → HCl = 3 → reaction NaOH HCl NaCl H2O → NaCl = 3 :=
by
  intros NaOH HCl NaCl H2O hNa hHCl hReaction
  sorry

end moles_of_NaCl_formed_l206_206471


namespace polynomial_remainder_l206_206388

theorem polynomial_remainder 
  (y: ℤ) 
  (root_cond: y^3 + y^2 + y + 1 = 0) 
  (beta_is_root: ∃ β: ℚ, β^3 + β^2 + β + 1 = 0) 
  (beta_four: ∀ β: ℚ, β^3 + β^2 + β + 1 = 0 → β^4 = 1) : 
  ∃ q r, (y^20 + y^15 + y^10 + y^5 + 1) = q * (y^3 + y^2 + y + 1) + r ∧ (r = 1) :=
by
  sorry

end polynomial_remainder_l206_206388


namespace part_one_part_two_l206_206285

def discriminant (a b c : ℝ) := b^2 - 4*a*c

theorem part_one (a : ℝ) (h : 0 < a) : 
  (∃ x : ℝ, ax^2 - 3*x + 2 < 0) ↔ 0 < a ∧ a < 9/8 := 
by 
  sorry

theorem part_two (a x : ℝ) : 
  (ax^2 - 3*x + 2 > ax - 1) ↔ 
  (a = 0 ∧ x < 1) ∨ 
  (a < 0 ∧ 3/a < x ∧ x < 1) ∨ 
  (0 < a ∧ (a > 3 ∧ (x < 3/a ∨ x > 1)) ∨ (a = 3 ∧ x ≠ 1) ∨ (0 < a ∧ a < 3 ∧ (x < 1 ∨ x > 3/a))) :=
by 
  sorry

end part_one_part_two_l206_206285


namespace pencils_problem_l206_206962

theorem pencils_problem (x : ℕ) :
  2 * x + 6 * 3 + 2 * 1 = 24 → x = 2 :=
by
  sorry

end pencils_problem_l206_206962


namespace tank_a_height_l206_206403

theorem tank_a_height (h_B : ℝ) (C_A C_B : ℝ) (V_A : ℝ → ℝ) (V_B : ℝ) :
  C_A = 4 ∧ C_B = 10 ∧ h_B = 8 ∧ (∀ h_A : ℝ, V_A h_A = 0.10000000000000002 * V_B) →
  ∃ h_A : ℝ, h_A = 5 :=
by sorry

end tank_a_height_l206_206403


namespace paint_remaining_after_two_days_l206_206358

-- Define the conditions
def original_paint_amount := 1
def paint_used_day1 := original_paint_amount * (1/4)
def remaining_paint_after_day1 := original_paint_amount - paint_used_day1
def paint_used_day2 := remaining_paint_after_day1 * (1/2)
def remaining_paint_after_day2 := remaining_paint_after_day1 - paint_used_day2

-- Theorem to be proved
theorem paint_remaining_after_two_days :
  remaining_paint_after_day2 = (3/8) * original_paint_amount := sorry

end paint_remaining_after_two_days_l206_206358


namespace consecutive_integer_sum_l206_206185

noncomputable def sqrt17 : ℝ := Real.sqrt 17

theorem consecutive_integer_sum : ∃ (a b : ℤ), (b = a + 1) ∧ (a < sqrt17 ∧ sqrt17 < b) ∧ (a + b = 9) :=
by
  sorry

end consecutive_integer_sum_l206_206185


namespace Chloe_initial_picked_carrots_l206_206079

variable (x : ℕ)

theorem Chloe_initial_picked_carrots :
  (x - 45 + 42 = 45) → (x = 48) :=
by
  intro h
  sorry

end Chloe_initial_picked_carrots_l206_206079


namespace find_number_of_students_l206_206846

-- Conditions
def john_marks_wrongly_recorded : ℕ := 82
def john_actual_marks : ℕ := 62
def sarah_marks_wrongly_recorded : ℕ := 76
def sarah_actual_marks : ℕ := 66
def emily_marks_wrongly_recorded : ℕ := 92
def emily_actual_marks : ℕ := 78
def increase_in_average : ℚ := 1 / 2

-- Proof problem
theorem find_number_of_students (n : ℕ) 
    (h1 : john_marks_wrongly_recorded = 82)
    (h2 : john_actual_marks = 62)
    (h3 : sarah_marks_wrongly_recorded = 76)
    (h4 : sarah_actual_marks = 66)
    (h5 : emily_marks_wrongly_recorded = 92)
    (h6 : emily_actual_marks = 78) 
    (h7: increase_in_average = 1 / 2):
    n = 88 :=
by 
  sorry

end find_number_of_students_l206_206846


namespace spent_on_puzzle_l206_206772

-- Defining all given conditions
def initial_money : ℕ := 8
def saved_money : ℕ := 13
def spent_on_comic : ℕ := 2
def final_amount : ℕ := 1

-- Define the total money before spending on the puzzle
def total_before_puzzle := initial_money + saved_money - spent_on_comic

-- Prove that the amount spent on the puzzle is $18
theorem spent_on_puzzle : (total_before_puzzle - final_amount) = 18 := 
by {
  sorry
}

end spent_on_puzzle_l206_206772


namespace find_m_intersection_points_l206_206113

theorem find_m (m : ℝ) (hp : 2^2 + 2 * m + m^2 - 3 = 4) (h_pos : m > 0) : m = 1 := 
by
  sorry

theorem intersection_points (m : ℝ) (hp : 2^2 + 2 * m + m^2 - 3 = 4) (h_pos : m > 0) 
  (hm : m = 1) : ∃ x1 x2 : ℝ, (x^2 + x - 2 = 0) ∧ x1 ≠ x2 :=
by
  sorry

end find_m_intersection_points_l206_206113


namespace quadratic_has_two_real_roots_roots_form_rectangle_with_diagonal_l206_206467

-- Condition for the quadratic equation having two real roots
theorem quadratic_has_two_real_roots (k : ℝ) :
  (∃ x1 x2 : ℝ, (x1 + x2 = k + 1) ∧ (x1 * x2 = 1/4 * k^2 + 1)) ↔ (k ≥ 3 / 2) :=
sorry

-- Condition linking the roots of the equation and the properties of the rectangle
theorem roots_form_rectangle_with_diagonal (k : ℝ) 
  (h : k ≥ 3 / 2) :
  (∃ x1 x2 : ℝ, (x1 + x2 = k + 1) ∧ (x1 * x2 = 1/4 * k^2 + 1)
  ∧ (x1^2 + x2^2 = 5)) ↔ (k = 2) :=
sorry

end quadratic_has_two_real_roots_roots_form_rectangle_with_diagonal_l206_206467


namespace proportion_calculation_l206_206493

theorem proportion_calculation (x y : ℝ) (h1 : 0.75 / x = 5 / y) (h2 : x = 1.2) : y = 8 :=
by
  sorry

end proportion_calculation_l206_206493


namespace garden_length_l206_206622

theorem garden_length 
  (W : ℕ) (small_gate_width : ℕ) (large_gate_width : ℕ) (P : ℕ)
  (hW : W = 125)
  (h_small_gate : small_gate_width = 3)
  (h_large_gate : large_gate_width = 10)
  (hP : P = 687) :
  ∃ (L : ℕ), P = 2 * L + 2 * W - (small_gate_width + large_gate_width) ∧ L = 225 := by
  sorry

end garden_length_l206_206622


namespace no_natural_n_for_perfect_square_l206_206130

theorem no_natural_n_for_perfect_square :
  ¬ ∃ n : ℕ, ∃ k : ℕ, 2007 + 4^n = k^2 :=
by {
  sorry  -- Proof omitted
}

end no_natural_n_for_perfect_square_l206_206130


namespace pool_perimeter_l206_206973

theorem pool_perimeter (garden_length : ℝ) (plot_area : ℝ) (plot_count : ℕ) : 
  garden_length = 9 ∧ plot_area = 20 ∧ plot_count = 4 →
  ∃ (pool_perimeter : ℝ), pool_perimeter = 18 :=
by
  intros h
  obtain ⟨h1, h2, h3⟩ := h
  sorry

end pool_perimeter_l206_206973


namespace marked_box_in_second_row_l206_206854

theorem marked_box_in_second_row:
  ∀ a b c d e f g h : ℕ, 
  (e = a + b) → 
  (f = b + c) →
  (g = c + d) →
  (h = a + 2 * b + c) →
  ((a = 5) ∧ (d = 6)) →
  ((a = 3) ∨ (b = 3) ∨ (c = 3) ∨ (d = 3)) →
  (f = 3) :=
by
  sorry

end marked_box_in_second_row_l206_206854


namespace min_value_sum_inverse_squares_l206_206023

theorem min_value_sum_inverse_squares (a b c : ℝ) (h_pos_a : a > 0) (h_pos_b : b > 0) (h_pos_c : c > 0) (h_sum : a + b + c = 3) :
    (1 / (a + b)^2) + (1 / (a + c)^2) + (1 / (b + c)^2) >= 3 / 2 :=
sorry

end min_value_sum_inverse_squares_l206_206023


namespace connected_geometric_seq_a10_l206_206653

noncomputable def is_kth_order_geometric (a : ℕ → ℝ) (k : ℕ) : Prop :=
  ∃ q : ℝ, ∀ n : ℕ, a (n + k) = q * a n

theorem connected_geometric_seq_a10 (a : ℕ → ℝ) 
  (h : is_kth_order_geometric a 3) 
  (a1 : a 1 = 1) 
  (a4 : a 4 = 2) : 
  a 10 = 8 :=
sorry

end connected_geometric_seq_a10_l206_206653


namespace ella_distance_from_start_l206_206525

noncomputable def compute_distance (m1 : ℝ) (f1 f2 m_to_f : ℝ) : ℝ :=
  let f1' := m1 * m_to_f
  let total_west := f1' + f2
  let distance_in_feet := Real.sqrt (f1^2 + total_west^2)
  distance_in_feet / m_to_f

theorem ella_distance_from_start :
  let starting_west := 10
  let first_north := 30
  let second_west := 40
  let meter_to_feet := 3.28084 
  compute_distance starting_west first_north second_west meter_to_feet = 24.01 := sorry

end ella_distance_from_start_l206_206525


namespace other_root_eq_six_l206_206097

theorem other_root_eq_six (a : ℝ) (x1 : ℝ) (x2 : ℝ) 
  (h : x1 = -2) 
  (eqn : ∀ x, x^2 - a * x - 3 * a = 0 → (x = x1 ∨ x = x2)) :
  x2 = 6 :=
by
  sorry

end other_root_eq_six_l206_206097


namespace min_value_expression_l206_206095

theorem min_value_expression (x y : ℝ) : 
  (∃ (x_min y_min : ℝ), 
  (x_min = 1/2 ∧ y_min = 0) ∧ 
  3 * x^2 + 3 * x * y + y^2 - 3 * x + 3 * y + 9 = 39/4) :=
by
  sorry

end min_value_expression_l206_206095


namespace binary_to_decimal_l206_206012

theorem binary_to_decimal : (1 * 2^3 + 1 * 2^2 + 0 * 2^1 + 1 * 2^0 = 13) :=
by
  sorry

end binary_to_decimal_l206_206012


namespace find_hyperbola_focus_l206_206576

theorem find_hyperbola_focus : ∃ (x y : ℝ), 
  2 * x ^ 2 - 3 * y ^ 2 + 8 * x - 12 * y - 8 = 0 
  → (x, y) = (-2 + (Real.sqrt 30)/3, -2) :=
by
  sorry

end find_hyperbola_focus_l206_206576


namespace Razorback_tshirt_problem_l206_206725

theorem Razorback_tshirt_problem
  (A T : ℕ)
  (h1 : A + T = 186)
  (h2 : 78 * T = 1092) :
  A = 172 := by
  sorry

end Razorback_tshirt_problem_l206_206725


namespace log3_x_minus_1_increasing_l206_206613

noncomputable def log_base_3 (x : ℝ) : ℝ := Real.log x / Real.log 3

def is_increasing_on (f : ℝ → ℝ) (I : Set ℝ) : Prop :=
∀ ⦃x y⦄, x ∈ I → y ∈ I → x < y → f x < f y

theorem log3_x_minus_1_increasing : is_increasing_on (fun x => log_base_3 (x - 1)) (Set.Ioi 1) :=
sorry

end log3_x_minus_1_increasing_l206_206613


namespace number_of_preferred_groups_l206_206860

def preferred_group_sum_multiple_5 (n : Nat) : Nat := 
  (2^n) * ((2^(4*n) - 1) / 5 + 1) - 1

theorem number_of_preferred_groups :
  preferred_group_sum_multiple_5 400 = 2^400 * (2^1600 - 1) / 5 + 1 - 1 :=
sorry

end number_of_preferred_groups_l206_206860


namespace determine_M_l206_206609

theorem determine_M (M : ℕ) (h : 12 ^ 2 * 45 ^ 2 = 15 ^ 2 * M ^ 2) : M = 36 :=
by
  sorry

end determine_M_l206_206609


namespace train_speed_second_part_l206_206929

-- Define conditions
def distance_first_part (x : ℕ) := x
def speed_first_part := 40
def distance_second_part (x : ℕ) := 2 * x
def total_distance (x : ℕ) := 5 * x
def average_speed := 40

-- Define the problem
theorem train_speed_second_part (x : ℕ) (v : ℕ) (h1 : total_distance x = 5 * x)
  (h2 : total_distance x / average_speed = distance_first_part x / speed_first_part + distance_second_part x / v) :
  v = 20 :=
  sorry

end train_speed_second_part_l206_206929


namespace no_positive_a_b_for_all_primes_l206_206724

theorem no_positive_a_b_for_all_primes :
  ∀ (a b : ℕ), 0 < a → 0 < b → ∃ (p q : ℕ), p > 1000 ∧ q > 1000 ∧ p ≠ q ∧ Prime p ∧ Prime q ∧ ¬Prime (a * p + b * q) :=
by
  sorry

end no_positive_a_b_for_all_primes_l206_206724


namespace determine_a_l206_206723

theorem determine_a (a : ℝ) 
  (h1 : (a - 1) * (0:ℝ)^2 + 0 + a^2 - 1 = 0)
  (h2 : a - 1 ≠ 0) : 
  a = -1 := 
sorry

end determine_a_l206_206723


namespace find_m_l206_206428

open Set Real

noncomputable def setA : Set ℝ := {x | x < 2}
noncomputable def setB : Set ℝ := {x | x > 4}
noncomputable def setC (m : ℝ) : Set ℝ := {x | 1 - m ≤ x ∧ x ≤ m - 1}

theorem find_m (m : ℝ) : setC m ⊆ (setA ∪ setB) → m < 3 :=
by
  sorry

end find_m_l206_206428


namespace starting_weight_of_labrador_puppy_l206_206880

theorem starting_weight_of_labrador_puppy :
  ∃ L : ℝ,
    (L + 0.25 * L) - (12 + 0.25 * 12) = 35 ∧ 
    L = 40 :=
by
  use 40
  sorry

end starting_weight_of_labrador_puppy_l206_206880


namespace range_of_a_l206_206085

theorem range_of_a (a : ℝ) : 
  (∀ x : ℝ, (a - 1) * x > 2 ↔ x < 2 / (a - 1)) → a < 1 :=
by
  sorry

end range_of_a_l206_206085


namespace compound_interest_years_l206_206494

-- Definitions for the given conditions
def principal : ℝ := 1200
def rate : ℝ := 0.20
def compound_interest : ℝ := 873.60
def compounded_yearly : ℝ := 1

-- Calculate the future value from principal and compound interest
def future_value : ℝ := principal + compound_interest

-- Statement of the problem: Prove that the number of years t was 3 given the conditions
theorem compound_interest_years :
  ∃ (t : ℝ), future_value = principal * (1 + rate / compounded_yearly)^(compounded_yearly * t) := sorry

end compound_interest_years_l206_206494


namespace quadratic_equation_unique_solution_l206_206018

theorem quadratic_equation_unique_solution
  (a c : ℝ)
  (h_discriminant : 100 - 4 * a * c = 0)
  (h_sum : a + c = 12)
  (h_lt : a < c) :
  (a, c) = (6 - Real.sqrt 11, 6 + Real.sqrt 11) :=
sorry

end quadratic_equation_unique_solution_l206_206018


namespace number_of_boys_l206_206370

variables (total_girls total_teachers total_people : ℕ)
variables (total_girls_eq : total_girls = 315) (total_teachers_eq : total_teachers = 772) (total_people_eq : total_people = 1396)

theorem number_of_boys (total_boys : ℕ) : total_boys = total_people - total_girls - total_teachers :=
by sorry

end number_of_boys_l206_206370


namespace boat_crossing_l206_206347

theorem boat_crossing (students teacher trips people_in_boat : ℕ) (h_students : students = 13) (h_teacher : teacher = 1) (h_boat_capacity : people_in_boat = 5) :
  trips = (students + teacher + people_in_boat - 1) / (people_in_boat - 1) :=
by
  sorry

end boat_crossing_l206_206347


namespace mabel_tomatoes_l206_206410

theorem mabel_tomatoes :
  ∃ (t1 t2 t3 t4 : ℕ), 
    t1 = 8 ∧ 
    t2 = t1 + 4 ∧ 
    t3 = 3 * (t1 + t2) ∧ 
    t4 = 3 * (t1 + t2) ∧ 
    (t1 + t2 + t3 + t4) = 140 :=
by
  sorry

end mabel_tomatoes_l206_206410


namespace percentage_of_students_70_79_l206_206308

def tally_90_100 := 6
def tally_80_89 := 9
def tally_70_79 := 8
def tally_60_69 := 6
def tally_50_59 := 3
def tally_below_50 := 1

def total_students := tally_90_100 + tally_80_89 + tally_70_79 + tally_60_69 + tally_50_59 + tally_below_50

theorem percentage_of_students_70_79 : (tally_70_79 : ℚ) / total_students = 8 / 33 :=
by
  sorry

end percentage_of_students_70_79_l206_206308


namespace problem_statement_l206_206008

theorem problem_statement (f : ℕ → ℤ) (a b : ℤ) 
  (h1 : f 1 = 7) 
  (h2 : f 2 = 11)
  (h3 : ∀ x, f x = a * x^2 + b * x + 3) :
  f 3 = 15 := 
sorry

end problem_statement_l206_206008


namespace minimum_soldiers_to_add_l206_206762

theorem minimum_soldiers_to_add 
  (N : ℕ)
  (h1 : N % 7 = 2)
  (h2 : N % 12 = 2) : 
  (84 - N % 84) = 82 := 
by 
  sorry

end minimum_soldiers_to_add_l206_206762


namespace tomatoes_price_per_pound_l206_206377

noncomputable def price_per_pound (cost_per_pound : ℝ) (loss_percent : ℝ) (profit_percent : ℝ) : ℝ :=
  let remaining_percent := 1 - loss_percent / 100
  let desired_total := (1 + profit_percent / 100) * cost_per_pound
  desired_total / remaining_percent

theorem tomatoes_price_per_pound :
  price_per_pound 0.80 15 8 = 1.02 :=
by
  sorry

end tomatoes_price_per_pound_l206_206377


namespace even_and_monotonically_decreasing_l206_206889

noncomputable def f_B (x : ℝ) : ℝ := 1 / (x^2)

theorem even_and_monotonically_decreasing (x : ℝ) (h : x > 0) :
  (f_B x = f_B (-x)) ∧ (∀ {a b : ℝ}, a < b → a > 0 → b > 0 → f_B a > f_B b) :=
by
  sorry

end even_and_monotonically_decreasing_l206_206889


namespace probability_tile_from_ANGLE_l206_206909

def letters_in_ALGEBRA : List Char := ['A', 'L', 'G', 'E', 'B', 'R', 'A']
def letters_in_ANGLE : List Char := ['A', 'N', 'G', 'L', 'E']
def count_matching_letters (letters: List Char) (target: List Char) : Nat :=
  letters.foldr (fun l acc => if l ∈ target then acc + 1 else acc) 0

theorem probability_tile_from_ANGLE :
  (count_matching_letters letters_in_ALGEBRA letters_in_ANGLE : ℚ) / (letters_in_ALGEBRA.length : ℚ) = 5 / 7 :=
by
  sorry

end probability_tile_from_ANGLE_l206_206909


namespace sum_of_squares_l206_206850

theorem sum_of_squares (x y z : ℕ) (hx : x = 1) (hy : y = 2) (hz : z = 3) (h_sum : x * 1 + y * 2 + z * 3 = 12) : x^2 + y^2 + z^2 = 56 :=
by
  sorry

end sum_of_squares_l206_206850


namespace square_side_length_l206_206335

-- Definition of the problem (statements)
theorem square_side_length (A : ℝ) (s : ℝ) (h : A = s * s) (hA : A = 49) : s = 7 := 
by 
  sorry

end square_side_length_l206_206335


namespace frank_fence_length_l206_206607

theorem frank_fence_length (L W total_fence : ℝ) 
  (hW : W = 40) 
  (hArea : L * W = 200) 
  (htotal_fence : total_fence = 2 * L + W) : 
  total_fence = 50 := 
by 
  sorry

end frank_fence_length_l206_206607


namespace problem_statement_l206_206478

theorem problem_statement (a b c : ℤ) (h : c = b + 2) : 
  (a - (b + c)) - ((a + c) - b) = 0 :=
by
  sorry

end problem_statement_l206_206478


namespace domain_of_v_l206_206871

noncomputable def v (x : ℝ) : ℝ := 1 / (x - 1)^(1 / 3)

theorem domain_of_v :
  {x : ℝ | ∃ y : ℝ, y ≠ 0 ∧ y = (v x)} = {x | x ≠ 1} := by
  sorry

end domain_of_v_l206_206871


namespace factorize_expression_l206_206524

theorem factorize_expression (a x : ℝ) : -a*x^2 + 2*a*x - a = -a*(x - 1)^2 := 
sorry

end factorize_expression_l206_206524


namespace line_tangent_to_parabola_l206_206538

theorem line_tangent_to_parabola (k : ℝ) :
  (∀ (x y : ℝ), y^2 = 16 * x → 4 * x + 3 * y + k = 0) → k = 9 :=
by
  sorry

end line_tangent_to_parabola_l206_206538


namespace vertical_asymptote_c_values_l206_206061

theorem vertical_asymptote_c_values (c : ℝ) :
  (∃ x : ℝ, (x^2 - x - 6) = 0 ∧ (x^2 - 2*x + c) ≠ 0 ∧ ∀ y : ℝ, ((y ≠ x) → (x ≠ 3) ∧ (x ≠ -2)))
  → (c = -3 ∨ c = -8) :=
by sorry

end vertical_asymptote_c_values_l206_206061


namespace cost_of_materials_l206_206506

theorem cost_of_materials (initial_bracelets given_away : ℕ) (sell_price profit : ℝ)
  (h1 : initial_bracelets = 52) 
  (h2 : given_away = 8) 
  (h3 : sell_price = 0.25) 
  (h4 : profit = 8) :
  let remaining_bracelets := initial_bracelets - given_away
  let total_revenue := remaining_bracelets * sell_price
  let cost_of_materials := total_revenue - profit
  cost_of_materials = 3 := 
by
  sorry

end cost_of_materials_l206_206506


namespace apples_total_l206_206895

def benny_apples : ℕ := 2
def dan_apples : ℕ := 9
def total_apples : ℕ := benny_apples + dan_apples

theorem apples_total : total_apples = 11 :=
by
    sorry

end apples_total_l206_206895


namespace avg_speed_train_l206_206273

theorem avg_speed_train {D V : ℝ} (h1 : D = 20 * (90 / 60)) (h2 : 360 = 6 * 60) : 
  V = D / (360 / 60) :=
  by sorry

end avg_speed_train_l206_206273


namespace dealer_gross_profit_l206_206034

theorem dealer_gross_profit
  (purchase_price : ℝ)
  (markup_rate : ℝ)
  (discount_rate : ℝ)
  (initial_selling_price : ℝ)
  (final_selling_price : ℝ)
  (gross_profit : ℝ)
  (h0 : purchase_price = 150)
  (h1 : markup_rate = 0.5)
  (h2 : discount_rate = 0.2)
  (h3 : initial_selling_price = purchase_price + markup_rate * initial_selling_price)
  (h4 : final_selling_price = initial_selling_price - discount_rate * initial_selling_price)
  (h5 : gross_profit = final_selling_price - purchase_price) :
  gross_profit = 90 :=
sorry

end dealer_gross_profit_l206_206034


namespace problem_statement_l206_206060

theorem problem_statement :
  (81000 ^ 3) / (27000 ^ 3) = 27 :=
by sorry

end problem_statement_l206_206060


namespace minimum_value_inequality_l206_206439

theorem minimum_value_inequality (x y z : ℝ) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) :
  (x^2 + 5 * x + 1) * (y^2 + 5 * y + 1) * (z^2 + 5 * y + 1) / (x * y * z) ≥ 343 :=
by sorry

end minimum_value_inequality_l206_206439


namespace initial_distance_planes_l206_206701

theorem initial_distance_planes (speed_A speed_B : ℝ) (time_seconds : ℝ) (time_hours : ℝ) (distance_A distance_B : ℝ) (total_distance : ℝ) :
  speed_A = 240 ∧ speed_B = 360 ∧ time_seconds = 72000 ∧ time_hours = 20 ∧ 
  time_hours = time_seconds / 3600 ∧
  distance_A = speed_A * time_hours ∧ 
  distance_B = speed_B * time_hours ∧ 
  total_distance = distance_A + distance_B →
  total_distance = 12000 :=
by
  intros
  sorry

end initial_distance_planes_l206_206701


namespace compare_abc_l206_206543

noncomputable def a : ℝ := (1 / 6) ^ (1 / 2)
noncomputable def b : ℝ := Real.log 1 / 3 / Real.log 6
noncomputable def c : ℝ := Real.log 1 / 7 / Real.log (1 / 6)

theorem compare_abc : c > a ∧ a > b := by
  sorry

end compare_abc_l206_206543


namespace slices_per_pizza_l206_206744

def number_of_people : ℕ := 18
def slices_per_person : ℕ := 3
def number_of_pizzas : ℕ := 6
def total_slices : ℕ := number_of_people * slices_per_person

theorem slices_per_pizza : total_slices / number_of_pizzas = 9 :=
by
  -- proof steps would go here
  sorry

end slices_per_pizza_l206_206744


namespace cube_root_21952_is_28_l206_206708

theorem cube_root_21952_is_28 :
  ∃ n : ℕ, n^3 = 21952 ∧ n = 28 :=
sorry

end cube_root_21952_is_28_l206_206708


namespace collinear_points_l206_206147

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

end collinear_points_l206_206147


namespace impossible_to_convince_logical_jury_of_innocence_if_guilty_l206_206501

theorem impossible_to_convince_logical_jury_of_innocence_if_guilty :
  (guilty : Prop) →
  (jury_is_logical : Prop) →
  guilty →
  (∀ statement : Prop, (logical_deduction : Prop) → (logical_deduction → ¬guilty)) →
  False :=
by
  intro guilty jury_is_logical guilty_premise logical_argument
  sorry

end impossible_to_convince_logical_jury_of_innocence_if_guilty_l206_206501


namespace jack_received_emails_in_the_morning_l206_206704

theorem jack_received_emails_in_the_morning
  (total_emails : ℕ)
  (afternoon_emails : ℕ)
  (morning_emails : ℕ) 
  (h1 : total_emails = 8)
  (h2 : afternoon_emails = 5)
  (h3 : total_emails = morning_emails + afternoon_emails) :
  morning_emails = 3 :=
  by
    -- proof omitted
    sorry

end jack_received_emails_in_the_morning_l206_206704


namespace total_yards_in_marathons_eq_495_l206_206717

-- Definitions based on problem conditions
def marathon_miles : ℕ := 26
def marathon_yards : ℕ := 385
def yards_in_mile : ℕ := 1760
def marathons_run : ℕ := 15

-- Main proof statement
theorem total_yards_in_marathons_eq_495
  (miles_per_marathon : ℕ := marathon_miles)
  (yards_per_marathon : ℕ := marathon_yards)
  (yards_per_mile : ℕ := yards_in_mile)
  (marathons : ℕ := marathons_run) :
  let total_yards := marathons * yards_per_marathon
  let remaining_yards := total_yards % yards_per_mile
  remaining_yards = 495 :=
by
  sorry

end total_yards_in_marathons_eq_495_l206_206717


namespace trapezoid_area_difference_l206_206733

def trapezoid_area (base1 base2 height : ℝ) : ℝ :=
  0.5 * (base1 + base2) * height

def combined_area (base1 base2 height : ℝ) : ℝ :=
  2 * trapezoid_area base1 base2 height

theorem trapezoid_area_difference :
  let combined_area1 := combined_area 11 19 10
  let combined_area2 := combined_area 9.5 11 8
  combined_area1 - combined_area2 = 136 :=
by
  let combined_area1 := combined_area 11 19 10 
  let combined_area2 := combined_area 9.5 11 8 
  show combined_area1 - combined_area2 = 136
  sorry

end trapezoid_area_difference_l206_206733


namespace line_equation_passing_through_points_l206_206491

theorem line_equation_passing_through_points 
  (a₁ b₁ a₂ b₂ : ℝ)
  (h1 : 2 * a₁ + 3 * b₁ + 1 = 0)
  (h2 : 2 * a₂ + 3 * b₂ + 1 = 0)
  (h3 : ∀ (x y : ℝ), (x, y) = (2, 3) → a₁ * x + b₁ * y + 1 = 0 ∧ a₂ * x + b₂ * y + 1 = 0) :
  (∀ (x y : ℝ), (2 * x + 3 * y + 1 = 0) ↔ 
                (a₁ = x ∧ b₁ = y) ∨ (a₂ = x ∧ b₂ = y)) :=
by
  sorry

end line_equation_passing_through_points_l206_206491


namespace rectangle_area_proof_l206_206952

def rectangle_area (L W : ℝ) : ℝ := L * W

theorem rectangle_area_proof (L W : ℝ) (h1 : L + W = 23) (h2 : L^2 + W^2 = 289) : rectangle_area L W = 120 := by
  sorry

end rectangle_area_proof_l206_206952


namespace population_growth_l206_206163

theorem population_growth 
  (P₀ : ℝ) (P₂ : ℝ) (r : ℝ)
  (hP₀ : P₀ = 15540) 
  (hP₂ : P₂ = 25460.736)
  (h_growth : P₂ = P₀ * (1 + r)^2) :
  r = 0.28 :=
by 
  sorry

end population_growth_l206_206163


namespace positive_real_x_condition_l206_206831

-- We define the conditions:
variables (x : ℝ)
#check (1 - x^4)
#check (1 + x^4)

-- The main proof statement:
theorem positive_real_x_condition (h1 : x > 0) 
    (h2 : (Real.sqrt (Real.sqrt (1 - x^4)) + Real.sqrt (Real.sqrt (1 + x^4)) = 1)) :
    (x^8 = 35 / 36) :=
sorry

end positive_real_x_condition_l206_206831


namespace solve_system_of_equations_l206_206437

theorem solve_system_of_equations :
  ∃ (x y : ℝ), x - y = 2 ∧ 3 * x + y = 4 ∧ x = 1.5 ∧ y = -0.5 :=
by
  sorry

end solve_system_of_equations_l206_206437


namespace subset_property_l206_206406

theorem subset_property : {2} ⊆ {x | x ≤ 10} := 
by 
  sorry

end subset_property_l206_206406


namespace allocation_schemes_count_l206_206422

open BigOperators -- For working with big operator notations
open Finset -- For working with finite sets
open Nat -- For natural number operations

-- Define the number of students and dormitories
def num_students : ℕ := 7
def num_dormitories : ℕ := 2

-- Define the constraint for minimum students in each dormitory
def min_students_in_dormitory : ℕ := 2

-- Compute the number of ways to allocate students given the conditions
noncomputable def number_of_allocation_schemes : ℕ :=
  (Nat.choose num_students 3) * (Nat.choose 4 2) + (Nat.choose num_students 2) * (Nat.choose 5 2)

-- The theorem stating the total number of allocation schemes
theorem allocation_schemes_count :
  number_of_allocation_schemes = 112 :=
  by sorry

end allocation_schemes_count_l206_206422


namespace existence_of_k_good_function_l206_206930

def is_k_good_function (f : ℕ+ → ℕ+) (k : ℕ) : Prop :=
  ∀ m n : ℕ+, m ≠ n → Nat.gcd (f m + n) (f n + m) ≤ k

theorem existence_of_k_good_function (k : ℕ) :
  (∃ f : ℕ+ → ℕ+, is_k_good_function f k) ↔ k ≥ 2 := sorry

end existence_of_k_good_function_l206_206930


namespace intercepts_correct_l206_206011

-- Define the equation of the line
def line_eq (x y : ℝ) := 5 * x - 2 * y - 10 = 0

-- Define the intercepts
def x_intercept : ℝ := 2
def y_intercept : ℝ := -5

-- Prove that the intercepts are as stated
theorem intercepts_correct :
  (∃ x, line_eq x 0 ∧ x = x_intercept) ∧
  (∃ y, line_eq 0 y ∧ y = y_intercept) :=
by
  sorry

end intercepts_correct_l206_206011


namespace campaign_funds_total_l206_206413

variable (X : ℝ)

def campaign_funds (friends family remaining : ℝ) : Prop :=
  friends = 0.40 * X ∧
  family = 0.30 * (X - friends) ∧
  remaining = X - (friends + family) ∧
  remaining = 4200

theorem campaign_funds_total (X_val : ℝ) (friends family remaining : ℝ)
    (h : campaign_funds X friends family remaining) : X = 10000 :=
by
  have h_friends : friends = 0.40 * X := h.1
  have h_family : family = 0.30 * (X - friends) := h.2.1
  have h_remaining : remaining = X - (friends + family) := h.2.2.1
  have h_remaining_amount : remaining = 4200 := h.2.2.2
  sorry

end campaign_funds_total_l206_206413


namespace professional_doctors_percentage_l206_206876

-- Defining the context and conditions:

variable (total_percent : ℝ) (leaders_percent : ℝ) (nurses_percent : ℝ) (doctors_percent : ℝ)

-- Specifying the conditions:
def total_percentage_sum : Prop :=
  total_percent = 100

def leaders_percentage : Prop :=
  leaders_percent = 4

def nurses_percentage : Prop :=
  nurses_percent = 56

-- Stating the actual theorem to be proved:
theorem professional_doctors_percentage
  (h1 : total_percentage_sum total_percent)
  (h2 : leaders_percentage leaders_percent)
  (h3 : nurses_percentage nurses_percent) :
  doctors_percent = 100 - (leaders_percent + nurses_percent) := by
  sorry -- Proof placeholder

end professional_doctors_percentage_l206_206876


namespace initial_students_l206_206886

theorem initial_students {f : ℕ → ℕ} {g : ℕ → ℕ} (h_f : ∀ t, t ≥ 15 * 60 + 3 → (f t = 4 * ((t - (15 * 60 + 3)) / 3 + 1))) 
    (h_g : ∀ t, t ≥ 15 * 60 + 10 → (g t = 8 * ((t - (15 * 60 + 10)) / 10 + 1))) 
    (students_at_1544 : f 15 * 60 + 44 - g 15 * 60 + 44 + initial = 27) : 
    initial = 3 := 
sorry

end initial_students_l206_206886


namespace ellipse_foci_distance_l206_206043

noncomputable def distance_between_foci_of_ellipse (a b : ℝ) : ℝ :=
  Real.sqrt (a^2 - b^2)

theorem ellipse_foci_distance :
  ∀ (a b : ℝ), (a = 5) → (b = 2) →
  distance_between_foci_of_ellipse a b = Real.sqrt 21 :=
by
  intros a b ha hb
  rw [ha, hb]
  -- The rest of the proof is omitted
  sorry

end ellipse_foci_distance_l206_206043


namespace b_41_mod_49_l206_206729

noncomputable def b (n : ℕ) : ℕ :=
  6 ^ n + 8 ^ n

theorem b_41_mod_49 : b 41 % 49 = 35 := by
  sorry

end b_41_mod_49_l206_206729


namespace imaginary_part_of_z_squared_l206_206649

-- Let i be the imaginary unit
def i : ℂ := Complex.I

-- Define the complex number (1 - 2i)
def z : ℂ := 1 - 2 * i

-- Define the expanded form of (1 - 2i)^2
def z_squared : ℂ := z^2

-- State the problem of finding the imaginary part of (1 - 2i)^2
theorem imaginary_part_of_z_squared : (z_squared).im = -4 := by
  sorry

end imaginary_part_of_z_squared_l206_206649


namespace circle_equation_l206_206206

theorem circle_equation 
    (a : ℝ)
    (x y : ℝ)
    (tangent_lines : x + y = 0 ∧ x + y = 4)
    (center_line : x - y = a)
    (center_point : ∃ (a : ℝ), x = a ∧ y = a) :
    ∃ (x y : ℝ), (x - 1)^2 + (y - 1)^2 = 2 :=
by
  sorry

end circle_equation_l206_206206


namespace house_number_digits_cost_l206_206171

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

end house_number_digits_cost_l206_206171


namespace num_girls_at_park_l206_206184

theorem num_girls_at_park (G : ℕ) (h1 : 11 + 50 + G = 3 * 25) : G = 14 := by
  sorry

end num_girls_at_park_l206_206184


namespace complex_pow_diff_zero_l206_206222

theorem complex_pow_diff_zero {i : ℂ} (h : i^2 = -1) : (2 + i)^(12) - (2 - i)^(12) = 0 := by
  sorry

end complex_pow_diff_zero_l206_206222


namespace rhombus_diagonal_solution_l206_206268

variable (d1 : ℝ) (A : ℝ)

def rhombus_other_diagonal (d1 d2 A : ℝ) : Prop :=
  A = (d1 * d2) / 2

theorem rhombus_diagonal_solution (h1 : d1 = 16) (h2 : A = 80) : rhombus_other_diagonal d1 10 A :=
by
  rw [h1, h2]
  sorry

end rhombus_diagonal_solution_l206_206268


namespace max_min_value_of_f_l206_206283

theorem max_min_value_of_f (x y z : ℝ) :
  (-1 ≤ 2 * x + y - z) ∧ (2 * x + y - z ≤ 8) ∧
  (2 ≤ x - y + z) ∧ (x - y + z ≤ 9) ∧
  (-3 ≤ x + 2 * y - z) ∧ (x + 2 * y - z ≤ 7) →
  (-6 ≤ 7 * x + 5 * y - 2 * z) ∧ (7 * x + 5 * y - 2 * z ≤ 47) :=
by
  sorry

end max_min_value_of_f_l206_206283


namespace vanessas_mother_picked_14_carrots_l206_206076

-- Define the problem parameters
variable (V : Nat := 17)  -- Vanessa picked 17 carrots
variable (G : Nat := 24)  -- Total good carrots
variable (B : Nat := 7)   -- Total bad carrots

-- Define the proof goal: Vanessa's mother picked 14 carrots
theorem vanessas_mother_picked_14_carrots : (G + B) - V = 14 := by
  sorry

end vanessas_mother_picked_14_carrots_l206_206076


namespace xiaoning_comprehensive_score_l206_206624

theorem xiaoning_comprehensive_score
  (max_score : ℕ := 100)
  (midterm_weight : ℝ := 0.3)
  (final_weight : ℝ := 0.7)
  (midterm_score : ℕ := 80)
  (final_score : ℕ := 90) :
  (midterm_score * midterm_weight + final_score * final_weight) = 87 :=
by
  sorry

end xiaoning_comprehensive_score_l206_206624


namespace circumference_of_circle_l206_206859

/-- Given a circle with area 4 * π square units, prove that its circumference is 4 * π units. -/
theorem circumference_of_circle (r : ℝ) (h : π * r^2 = 4 * π) : 2 * π * r = 4 * π :=
sorry

end circumference_of_circle_l206_206859


namespace travel_time_total_l206_206747

theorem travel_time_total (dist1 dist2 dist3 speed1 speed2 speed3 : ℝ)
  (h_dist1 : dist1 = 50) (h_dist2 : dist2 = 100) (h_dist3 : dist3 = 150)
  (h_speed1 : speed1 = 50) (h_speed2 : speed2 = 80) (h_speed3 : speed3 = 120) :
  dist1 / speed1 + dist2 / speed2 + dist3 / speed3 = 3.5 :=
by
  sorry

end travel_time_total_l206_206747


namespace sixty_three_times_fifty_seven_l206_206378

theorem sixty_three_times_fifty_seven : 63 * 57 = 3591 := by
  sorry

end sixty_three_times_fifty_seven_l206_206378


namespace find_y_when_x_is_twelve_l206_206912

variables (x y k : ℝ)

theorem find_y_when_x_is_twelve
  (h1 : x * y = k)
  (h2 : x + y = 60)
  (h3 : x = 3 * y)
  (hx : x = 12) :
  y = 56.25 :=
sorry

end find_y_when_x_is_twelve_l206_206912


namespace length_of_second_train_l206_206007

/-- 
The length of the second train can be determined given the length and speed of the first train,
the speed of the second train, and the time they take to cross each other.
-/
theorem length_of_second_train (speed1_kmph : ℝ) (length1_m : ℝ) (speed2_kmph : ℝ) (time_s : ℝ) :
  (speed1_kmph = 120) →
  (length1_m = 230) →
  (speed2_kmph = 80) →
  (time_s = 9) →
  let relative_speed_m_per_s := (speed1_kmph * 1000 / 3600) + (speed2_kmph * 1000 / 3600)
  let total_distance := relative_speed_m_per_s * time_s
  let length2_m := total_distance - length1_m
  length2_m = 269.95 :=
by
  intros h₁ h₂ h₃ h₄
  rw [h₁, h₂, h₃, h₄]
  let relative_speed_m_per_s := (120 * 1000 / 3600) + (80 * 1000 / 3600)
  let total_distance := relative_speed_m_per_s * 9
  let length2_m := total_distance - 230
  exact sorry

end length_of_second_train_l206_206007


namespace smallest_n_for_terminating_decimal_l206_206899

theorem smallest_n_for_terminating_decimal :
  ∃ (n : ℕ), (∀ m : ℕ, (n = m → m > 0 → ∃ (a b : ℕ), n + 103 = 2^a * 5^b)) 
    ∧ n = 22 :=
sorry

end smallest_n_for_terminating_decimal_l206_206899


namespace speed_against_current_l206_206769

theorem speed_against_current (V_curr : ℝ) (V_man : ℝ) (V_curr_val : V_curr = 3.2) (V_man_with_curr : V_man = 15) :
    V_man - V_curr = 8.6 := 
by 
  rw [V_curr_val, V_man_with_curr]
  norm_num
  sorry

end speed_against_current_l206_206769


namespace find_b_l206_206621

theorem find_b (a u v w : ℝ) (b : ℝ)
  (h1 : ∀ x : ℝ, 12 * x^3 + 7 * a * x^2 + 6 * b * x + b = 0 → (x = u ∨ x = v ∨ x = w))
  (h2 : 0 < u ∧ 0 < v ∧ 0 < w)
  (h3 : u ≠ v ∧ v ≠ w ∧ u ≠ w)
  (h4 : Real.log u / Real.log 3 + Real.log v / Real.log 3 + Real.log w / Real.log 3 = 3):
  b = -324 := 
sorry

end find_b_l206_206621


namespace necessary_condition_x_pow_2_minus_x_lt_0_l206_206137

theorem necessary_condition_x_pow_2_minus_x_lt_0 (x : ℝ) : (x^2 - x < 0) → (-1 < x ∧ x < 1) := by
  intro hx
  sorry

end necessary_condition_x_pow_2_minus_x_lt_0_l206_206137


namespace Expected_and_Variance_l206_206783

variables (p : ℝ) (h : 0 ≤ p ∧ p ≤ 1)

def P (xi : ℕ) : ℝ := 
  if xi = 0 then p else if xi = 1 then 1 - p else 0

def E_xi : ℝ := 0 * P p 0 + 1 * P p 1

def D_xi : ℝ := (0 - E_xi p)^2 * P p 0 + (1 - E_xi p)^2 * P p 1

theorem Expected_and_Variance :
  (E_xi p = 1 - p) ∧ (D_xi p = p * (1 - p)) :=
sorry

end Expected_and_Variance_l206_206783


namespace smallest_AAB_value_l206_206672

theorem smallest_AAB_value {A B : ℕ} (hA : 1 ≤ A ∧ A ≤ 9) (hB : 1 ≤ B ∧ B ≤ 9) (h_distinct : A ≠ B) (h_eq : 10 * A + B = (1 / 9) * (100 * A + 10 * A + B)) :
  100 * A + 10 * A + B = 225 :=
by
  -- Insert proof here
  sorry

end smallest_AAB_value_l206_206672


namespace percentage_increase_is_20_l206_206115

-- Defining the original cost and new cost
def original_cost := 200
def new_total_cost := 480

-- Doubling the capacity means doubling the original cost
def doubled_old_cost := 2 * original_cost

-- The increase in cost
def increase_cost := new_total_cost - doubled_old_cost

-- The percentage increase in cost
def percentage_increase := (increase_cost / doubled_old_cost) * 100

-- The theorem we need to prove
theorem percentage_increase_is_20 : percentage_increase = 20 :=
  by
  sorry

end percentage_increase_is_20_l206_206115


namespace evaluate_expression_l206_206100

theorem evaluate_expression : 202 - 101 + 9 = 110 :=
by
  sorry

end evaluate_expression_l206_206100


namespace remainder_of_product_mod_12_l206_206738

-- Define the given constants
def a := 1125
def b := 1127
def c := 1129
def d := 12

-- State the conditions as Lean hypotheses
lemma mod_eq_1125 : a % d = 9 := by sorry
lemma mod_eq_1127 : b % d = 11 := by sorry
lemma mod_eq_1129 : c % d = 1 := by sorry

-- Define the theorem to prove
theorem remainder_of_product_mod_12 : (a * b * c) % d = 3 := by
  -- Use the conditions stated above to prove the theorem
  sorry

end remainder_of_product_mod_12_l206_206738


namespace book_area_correct_l206_206654

def book_length : ℝ := 5
def book_width : ℝ := 10
def book_area (length : ℝ) (width : ℝ) : ℝ := length * width

theorem book_area_correct :
  book_area book_length book_width = 50 :=
by
  sorry

end book_area_correct_l206_206654


namespace cost_of_two_pans_is_20_l206_206684

variable (cost_of_pan : ℕ)

-- Conditions
def pots_cost := 3 * 20
def total_cost := 100
def pans_eq_cost := total_cost - pots_cost
def cost_of_pan_per_pans := pans_eq_cost / 4

-- Proof statement
theorem cost_of_two_pans_is_20 
  (h1 : pots_cost = 60)
  (h2 : total_cost = 100)
  (h3 : pans_eq_cost = total_cost - pots_cost)
  (h4 : cost_of_pan_per_pans = pans_eq_cost / 4)
  : 2 * cost_of_pan_per_pans = 20 :=
by sorry

end cost_of_two_pans_is_20_l206_206684


namespace embankment_height_bounds_l206_206082

theorem embankment_height_bounds
  (a : ℝ) (b : ℝ) (h : ℝ)
  (a_eq : a = 5)
  (b_lower_bound : 2 ≤ b)
  (vol_lower_bound : 400 ≤ (25 * (a^2 - b^2)))
  (vol_upper_bound : (25 * (a^2 - b^2)) ≤ 500) :
  1 ≤ h ∧ h ≤ (5 - Real.sqrt 5) / 2 :=
by
  sorry

end embankment_height_bounds_l206_206082


namespace cos_double_angle_l206_206726

theorem cos_double_angle (α : ℝ) (h : Real.tan α = 2) : Real.cos (2 * α) = -3 / 5 := by
  sorry

end cos_double_angle_l206_206726


namespace general_formula_sequence_l206_206914

-- Define the sequence as an arithmetic sequence with the given first term and common difference
def arithmetic_sequence (a_1 d : ℕ) (n : ℕ) : ℕ := a_1 + (n - 1) * d

-- Define given values
def a_1 : ℕ := 1
def d : ℕ := 2

-- State the theorem to be proved
theorem general_formula_sequence :
  ∀ n : ℕ, n > 0 → arithmetic_sequence a_1 d n = 2 * n - 1 :=
by
  intro n hn
  sorry

end general_formula_sequence_l206_206914


namespace area_inequality_l206_206016

variable {a b c : ℝ} (h_triangle : 0 < a ∧ 0 < b ∧ 0 < c ∧ a + b > c ∧ a + c > b ∧ b + c > a)

noncomputable def semiperimeter (a b c : ℝ) : ℝ := (a + b + c) / 2

noncomputable def area (a b c : ℝ) (h : 0 < a ∧ 0 < b ∧ 0 < c ∧ a + b > c ∧ a + c > b ∧ b + c > a) : ℝ :=
  let p := semiperimeter a b c
  Real.sqrt (p * (p - a) * (p - b) * (p - c))

theorem area_inequality (a b c : ℝ) (h : 0 < a ∧ 0 < b ∧ 0 < c ∧ a + b > c ∧ a + c > b ∧ b + c > a) :
  (2 * (area a b c h))^3 < (a * b * c)^2 := sorry

end area_inequality_l206_206016

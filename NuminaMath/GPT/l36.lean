import Mathlib

namespace equal_parallelogram_faces_are_rhombuses_l36_3679

theorem equal_parallelogram_faces_are_rhombuses 
  (a b c : ℝ) 
  (h: a * b = b * c ∧ b * c = a * c ∧ a ≠ 0 ∧ b ≠ 0 ∧ c ≠ 0) :
  a = b ∧ b = c :=
sorry

end equal_parallelogram_faces_are_rhombuses_l36_3679


namespace mother_age_l36_3639

theorem mother_age (x : ℕ) (h1 : 3 * x + x = 40) : 3 * x = 30 :=
by
  -- Here we should provide the proof but for now we use sorry to skip it
  sorry

end mother_age_l36_3639


namespace solve_equation_l36_3663

theorem solve_equation :
  ∃ x : Real, (x = 2 ∨ x = (-(1:Real) - Real.sqrt 17) / 2) ∧ (x^2 - |x - 1| - 3 = 0) :=
by
  sorry

end solve_equation_l36_3663


namespace special_set_exists_l36_3689

def exists_special_set : Prop :=
  ∃ S : Finset ℕ, S.card = 4004 ∧ 
  (∀ A : Finset ℕ, A ⊆ S ∧ A.card = 2003 → (A.sum id % 2003 ≠ 0))

-- statement with sorry to skip the proof
theorem special_set_exists : exists_special_set :=
sorry

end special_set_exists_l36_3689


namespace trig_identity_l36_3661

noncomputable def tan_eq_neg_4_over_3 (theta : ℝ) : Prop := 
  Real.tan theta = -4 / 3

theorem trig_identity (theta : ℝ) (h : tan_eq_neg_4_over_3 theta) : 
  (Real.cos (π / 2 + θ) - Real.sin (-π - θ)) / (Real.cos (11 * π / 2 - θ) + Real.sin (9 * π / 2 + θ)) = 8 / 7 :=
by
  sorry

end trig_identity_l36_3661


namespace average_after_17th_inning_l36_3671

variable (A : ℕ)

-- Definition of total runs before the 17th inning
def total_runs_before := 16 * A

-- Definition of new total runs after the 17th inning
def total_runs_after := total_runs_before A + 87

-- Definition of new average after the 17th inning
def new_average := A + 4

-- Definition of new total runs in terms of new average
def new_total_runs := 17 * new_average A

-- The statement we want to prove
theorem average_after_17th_inning : total_runs_after A = new_total_runs A → new_average A = 23 := by
  sorry

end average_after_17th_inning_l36_3671


namespace correct_choices_l36_3627

theorem correct_choices :
  (∃ u : ℝ × ℝ, (2 * u.1 + u.2 + 3 = 0) → u = (1, -2)) ∧
  ¬ (∀ a : ℝ, (a = -1 ↔ a^2 * x - y + 1 = 0 ∧ x - a * y - 2 = 0) → a = -1) ∧
  ((∃ (l : ℝ) (P : ℝ × ℝ), l = x + y - 6 → P = (2, 4) → 2 + 4 = l) → x + y - 6 = 0) ∧
  ((∃ (m b : ℝ), y = m * x + b → b = -2) → y = 3 * x - 2) :=
sorry

end correct_choices_l36_3627


namespace shop_owner_cheat_selling_percentage_l36_3681

noncomputable def percentage_cheat_buying : ℝ := 12
noncomputable def profit_percentage : ℝ := 40
noncomputable def percentage_cheat_selling : ℝ := 20

theorem shop_owner_cheat_selling_percentage 
  (percentage_cheat_buying : ℝ := 12)
  (profit_percentage : ℝ := 40) :
  percentage_cheat_selling = 20 := 
sorry

end shop_owner_cheat_selling_percentage_l36_3681


namespace non_congruent_rectangles_count_l36_3680

theorem non_congruent_rectangles_count (h w : ℕ) (P : ℕ) (multiple_of_4: ℕ → Prop) :
  P = 80 →
  w ≥ 1 ∧ h ≥ 1 →
  P = 2 * (w + h) →
  (multiple_of_4 w ∨ multiple_of_4 h) →
  (∀ k, multiple_of_4 k ↔ ∃ m, k = 4 * m) →
  ∃ n, n = 5 :=
by
  sorry

end non_congruent_rectangles_count_l36_3680


namespace percentage_passed_in_all_three_subjects_l36_3643

-- Define the given failed percentages as real numbers
def A : ℝ := 0.25  -- 25%
def B : ℝ := 0.48  -- 48%
def C : ℝ := 0.35  -- 35%
def AB : ℝ := 0.27 -- 27%
def AC : ℝ := 0.20 -- 20%
def BC : ℝ := 0.15 -- 15%
def ABC : ℝ := 0.10 -- 10%

-- State the theorem to prove the percentage of students who passed in all three subjects
theorem percentage_passed_in_all_three_subjects : 
  1 - (A + B + C - AB - AC - BC + ABC) = 0.44 :=
by
  sorry

end percentage_passed_in_all_three_subjects_l36_3643


namespace no_unhappy_days_l36_3622

theorem no_unhappy_days (D R : ℕ) : 
  (D^2 + 4) * (R^2 + 4) - 2 * D * (R^2 + 4) - 2 * R * (D^2 + 4) ≥ 0 := 
  sorry

end no_unhappy_days_l36_3622


namespace cost_price_proof_l36_3626

def trader_sells_66m_for_660 : Prop := ∃ cp profit sp : ℝ, sp = 660 ∧ cp * 66 + profit * 66 = sp
def profit_5_per_meter : Prop := ∃ profit : ℝ, profit = 5
def cost_price_per_meter_is_5 : Prop := ∃ cp : ℝ, cp = 5

theorem cost_price_proof : trader_sells_66m_for_660 → profit_5_per_meter → cost_price_per_meter_is_5 :=
by
  intros h1 h2
  sorry

end cost_price_proof_l36_3626


namespace amy_height_l36_3636

variable (A H N : ℕ)

theorem amy_height (h1 : A = 157) (h2 : A = H + 4) (h3 : H = N + 3) :
  N = 150 := sorry

end amy_height_l36_3636


namespace find_principal_l36_3618

theorem find_principal (r t1 t2 ΔI : ℝ) (h_r : r = 0.15) (h_t1 : t1 = 3.5) (h_t2 : t2 = 5) (h_ΔI : ΔI = 144) :
  ∃ P : ℝ, P = 640 :=
by
  sorry

end find_principal_l36_3618


namespace freq_distribution_correct_l36_3617

variable (freqTable_isForm : Prop)
variable (freqHistogram_isForm : Prop)
variable (freqTable_isAccurate : Prop)
variable (freqHistogram_isIntuitive : Prop)

theorem freq_distribution_correct :
  ((freqTable_isForm ∧ freqHistogram_isForm) ∧
   (freqTable_isAccurate ∧ freqHistogram_isIntuitive)) →
  True :=
by
  intros _
  exact trivial

end freq_distribution_correct_l36_3617


namespace hyperbola_asymptotes_l36_3619

theorem hyperbola_asymptotes : 
  (∀ x y : ℝ, (x^2)/4 - y^2 = 1) →
  (∀ x : ℝ, y = x / 2 ∨ y = -x / 2) :=
by
  intro h1
  sorry

end hyperbola_asymptotes_l36_3619


namespace part_one_part_two_l36_3600

noncomputable def f (a x : ℝ) : ℝ := x * (a + Real.log x)
noncomputable def g (x : ℝ) : ℝ := x / Real.exp x

theorem part_one (a : ℝ) (h : ∀ x > 0, f a x ≥ -1/Real.exp 1) : a = 0 := sorry

theorem part_two {a x : ℝ} (ha : a > 0) (hx : x > 0) :
  g x - f a x < 2 / Real.exp 1 := sorry

end part_one_part_two_l36_3600


namespace remainder_of_M_mod_1000_l36_3653

def M : ℕ := Nat.choose 9 8

theorem remainder_of_M_mod_1000 : M % 1000 = 9 := by
  sorry

end remainder_of_M_mod_1000_l36_3653


namespace cube_expansion_l36_3674

variable {a b : ℝ}

theorem cube_expansion (a b : ℝ) : (-a * b^2)^3 = -a^3 * b^6 :=
  sorry

end cube_expansion_l36_3674


namespace measure_of_angle_C_l36_3602

theorem measure_of_angle_C (C D : ℝ) (h1 : C + D = 360) (h2 : C = 5 * D) : C = 300 := 
by sorry

end measure_of_angle_C_l36_3602


namespace deborah_total_cost_l36_3677

-- Standard postage per letter
def stdPostage : ℝ := 1.08

-- Additional charge for international shipping per letter
def intlAdditional : ℝ := 0.14

-- Number of domestic and international letters
def numDomestic : ℕ := 2
def numInternational : ℕ := 2

-- Expected total cost for four letters
def expectedTotalCost : ℝ := 4.60

theorem deborah_total_cost :
  (numDomestic * stdPostage) + (numInternational * (stdPostage + intlAdditional)) = expectedTotalCost :=
by
  -- proof skipped
  sorry

end deborah_total_cost_l36_3677


namespace cindy_correct_result_l36_3660

theorem cindy_correct_result (x : ℝ) (h: (x - 7) / 5 = 27) : (x - 5) / 7 = 20 :=
by
  sorry

end cindy_correct_result_l36_3660


namespace S8_is_80_l36_3684

variable {a : ℕ → ℝ} -- sequence definition
variable {S : ℕ → ℝ} -- sum of sequence

-- Conditions
variable (h_seq : ∀ n, a (n + 1) = a n + d) -- arithmetic sequence
variable (h_sum : ∀ n, S n = n * (a 1 + a n) / 2) -- sum of the first n terms
variable (h_cond : a 3 = 20 - a 6) -- given condition

theorem S8_is_80 (h_seq : ∀ n, a (n + 1) = a n + d) (h_sum : ∀ n, S n = n * (a 1 + a n) / 2) (h_cond : a 3 = 20 - a 6) :
  S 8 = 80 :=
sorry

end S8_is_80_l36_3684


namespace find_unknown_number_l36_3654

theorem find_unknown_number (x n : ℚ) (h1 : n + 7/x = 6 - 5/x) (h2 : x = 12) : n = 5 :=
by
  sorry

end find_unknown_number_l36_3654


namespace gcd_three_numbers_l36_3665

theorem gcd_three_numbers (a b c : ℕ) (h₁ : a = 13847) (h₂ : b = 21353) (h₃ : c = 34691) : Nat.gcd (Nat.gcd a b) c = 5 := by sorry

end gcd_three_numbers_l36_3665


namespace simplify_fraction_sum_l36_3604

theorem simplify_fraction_sum :
  (3 / 462) + (17 / 42) + (1 / 11) = 116 / 231 := 
by
  sorry

end simplify_fraction_sum_l36_3604


namespace probability_of_one_girl_conditional_probability_of_one_girl_given_at_least_one_l36_3659

/- Define number of boys and girls -/
def num_boys : ℕ := 5
def num_girls : ℕ := 3

/- Define number of students selected -/
def num_selected : ℕ := 2

/- Define the total number of ways to select -/
def total_ways : ℕ := Nat.choose (num_boys + num_girls) num_selected

/- Define the number of ways to select exactly one girl -/
def ways_one_girl : ℕ := Nat.choose num_girls 1 * Nat.choose num_boys 1

/- Define the number of ways to select at least one girl -/
def ways_at_least_one_girl : ℕ := total_ways - Nat.choose num_boys num_selected

/- Define the first probability: exactly one girl participates -/
def prob_one_girl : ℚ := ways_one_girl / total_ways

/- Define the second probability: exactly one girl given at least one girl -/
def prob_one_girl_given_at_least_one : ℚ := ways_one_girl / ways_at_least_one_girl

theorem probability_of_one_girl : prob_one_girl = 15 / 28 := by
  sorry

theorem conditional_probability_of_one_girl_given_at_least_one : prob_one_girl_given_at_least_one = 5 / 6 := by
  sorry

end probability_of_one_girl_conditional_probability_of_one_girl_given_at_least_one_l36_3659


namespace overtime_hours_proof_l36_3647

-- Define the conditions
variable (regular_pay_rate : ℕ := 3)
variable (regular_hours : ℕ := 40)
variable (overtime_multiplier : ℕ := 2)
variable (total_pay : ℕ := 180)

-- Calculate the regular pay for 40 hours
def regular_pay : ℕ := regular_pay_rate * regular_hours

-- Calculate the extra pay received beyond regular pay
def extra_pay : ℕ := total_pay - regular_pay

-- Calculate overtime pay rate
def overtime_pay_rate : ℕ := overtime_multiplier * regular_pay_rate

-- Calculate the number of overtime hours
def overtime_hours (extra_pay : ℕ) (overtime_pay_rate : ℕ) : ℕ :=
  extra_pay / overtime_pay_rate

-- The theorem to prove
theorem overtime_hours_proof :
  overtime_hours extra_pay overtime_pay_rate = 10 := by
  sorry

end overtime_hours_proof_l36_3647


namespace number_of_real_solutions_l36_3678

noncomputable def f (x : ℝ) : ℝ :=
  (Finset.range 50).sum (λ n => (n + 1 : ℝ) / (x - (n + 1 : ℝ)))

theorem number_of_real_solutions : ∃ n : ℕ, n = 51 ∧ ∀ x : ℝ, f x = x + 1 ↔ n = 51 :=
by
  sorry

end number_of_real_solutions_l36_3678


namespace rational_k_quadratic_solution_count_l36_3656

theorem rational_k_quadratic_solution_count (N : ℕ) :
  (N = 98) ↔ 
  (∃ (k : ℚ) (x : ℤ), |k| < 500 ∧ (3 * x^2 + k * x + 7 = 0)) :=
sorry

end rational_k_quadratic_solution_count_l36_3656


namespace number_is_minus_three_l36_3603

variable (x a : ℝ)

theorem number_is_minus_three (h1 : a = 0.5) (h2 : x / (a - 3) = 3 / (a + 2)) : x = -3 :=
by
  sorry

end number_is_minus_three_l36_3603


namespace total_ducats_is_160_l36_3608

variable (T : ℤ) (a b c d e : ℤ) -- Variables to represent the amounts taken by the robbers

-- Conditions
axiom h1 : a = 81                                            -- The strongest robber took 81 ducats
axiom h2 : b ≠ c ∧ b ≠ d ∧ b ≠ e ∧ c ≠ d ∧ c ≠ e ∧ d ≠ e    -- Each remaining robber took a different amount
axiom h3 : a + b + c + d + e = T                             -- Total amount of ducats
axiom redistribution : 
  -- Redistribution process leads to each robber having the same amount
  2*b + 2*c + 2*d + 2*e = T ∧
  2*(2*c + 2*d + 2*e) = T ∧
  2*(2*(2*d + 2*e)) = T ∧
  2*(2*(2*(2*e))) = T

-- Proof that verifies the total ducats is 160
theorem total_ducats_is_160 : T = 160 :=
by
  sorry

end total_ducats_is_160_l36_3608


namespace mrs_taylor_total_payment_l36_3666

-- Declaring the price of items and discounts
def price_tv : ℝ := 750
def price_soundbar : ℝ := 300

def discount_tv : ℝ := 0.15
def discount_soundbar : ℝ := 0.10

-- Total number of each items
def num_tv : ℕ := 2
def num_soundbar : ℕ := 3

-- Total cost calculation after discounts
def total_cost_tv := num_tv * price_tv * (1 - discount_tv)
def total_cost_soundbar := num_soundbar * price_soundbar * (1 - discount_soundbar)
def total_cost := total_cost_tv + total_cost_soundbar

-- The theorem we want to prove
theorem mrs_taylor_total_payment : total_cost = 2085 := by
  -- Skipping the proof
  sorry

end mrs_taylor_total_payment_l36_3666


namespace proof_problem_l36_3613

def intelligentFailRate (r1 r2 r3 : ℚ) : ℚ :=
  1 - r1 * r2 * r3

def phi (p : ℚ) : ℚ :=
  30 * p * (1 - p)^29

def derivativePhi (p : ℚ) : ℚ :=
  30 * (1 - p)^28 * (1 - 30 * p)

def qualifiedPassRate (intelligentPassRate comprehensivePassRate : ℚ) : ℚ :=
  intelligentPassRate * comprehensivePassRate

theorem proof_problem :
  let r1 := (99 : ℚ) / 100
  let r2 := (98 : ℚ) / 99
  let r3 := (97 : ℚ) / 98
  let p0 := (1 : ℚ) / 30
  let comprehensivePassRate := 1 - p0
  let qualifiedRate := qualifiedPassRate (r1 * r2 * r3) comprehensivePassRate
  (intelligentFailRate r1 r2 r3 = 3 / 100) ∧
  (derivativePhi p0 = 0) ∧
  (qualifiedRate < 96 / 100) :=
by
  sorry

end proof_problem_l36_3613


namespace band_song_average_l36_3668

/-- 
The school band has 30 songs in their repertoire. 
They played 5 songs in the first set and 7 songs in the second set. 
They will play 2 songs for their encore. 
Assuming the band plays through their entire repertoire, 
how many songs will they play on average in the third and fourth sets?
 -/
theorem band_song_average
    (total_songs : ℕ)
    (first_set_songs : ℕ)
    (second_set_songs : ℕ)
    (encore_songs : ℕ)
    (remaining_sets : ℕ)
    (h_total : total_songs = 30)
    (h_first : first_set_songs = 5)
    (h_second : second_set_songs = 7)
    (h_encore : encore_songs = 2)
    (h_remaining : remaining_sets = 2) :
    (total_songs - (first_set_songs + second_set_songs + encore_songs)) / remaining_sets = 8 := 
by
  -- The proof will go here.
  sorry

end band_song_average_l36_3668


namespace sams_trip_length_l36_3697

theorem sams_trip_length (total_trip : ℚ) 
  (h1 : total_trip / 4 + 24 + total_trip / 6 = total_trip) : 
  total_trip = 288 / 7 :=
by
  -- proof placeholder
  sorry

end sams_trip_length_l36_3697


namespace functional_identity_l36_3635

-- Define the set of non-negative integers
def S : Set ℕ := {n | n ≥ 0}

-- Define the function f with the required domain and codomain
def f (n : ℕ) : ℕ := n

-- The hypothesis: the functional equation satisfied by f
axiom functional_equation :
  ∀ m n : ℕ, f (m + f n) = f (f m) + f n

-- The theorem we want to prove
theorem functional_identity (n : ℕ) : f n = n :=
  sorry

end functional_identity_l36_3635


namespace solve_quadratic_equation_l36_3687

theorem solve_quadratic_equation:
  (∀ x : ℝ, (8 * x^2 + 52 * x + 4) / (3 * x + 13) = 2 * x + 3 →
    x = ( -17 + Real.sqrt 569) / 4 ∨ x = ( -17 - Real.sqrt 569) / 4) :=
by
  sorry

end solve_quadratic_equation_l36_3687


namespace fraction_sum_le_41_over_42_l36_3640

theorem fraction_sum_le_41_over_42 (a b c : ℕ) (h : 1/a + 1/b + 1/c < 1) : 1/a + 1/b + 1/c ≤ 41/42 :=
sorry

end fraction_sum_le_41_over_42_l36_3640


namespace braden_money_box_total_l36_3683

def initial_money : ℕ := 400

def correct_predictions : ℕ := 3

def betting_rules (correct_predictions : ℕ) : ℕ :=
  match correct_predictions with
  | 1 => 25
  | 2 => 50
  | 3 => 75
  | 4 => 200
  | _ => 0

theorem braden_money_box_total:
  let winnings := (betting_rules correct_predictions * initial_money) / 100
  initial_money + winnings = 700 := 
by
  let winnings := (betting_rules correct_predictions * initial_money) / 100
  show initial_money + winnings = 700
  sorry

end braden_money_box_total_l36_3683


namespace minimum_value_of_nS_n_l36_3649

noncomputable def a₁ (d : ℝ) : ℝ := -9/2 * d

noncomputable def S (n : ℕ) (d : ℝ) : ℝ :=
  n / 2 * (2 * a₁ d + (n - 1) * d)

theorem minimum_value_of_nS_n :
  S 10 (2/3) = 0 → S 15 (2/3) = 25 → ∃ (n : ℕ), (n * S n (2/3)) = -48 :=
by 
  intros h10 h15
  sorry

end minimum_value_of_nS_n_l36_3649


namespace parallel_a_b_projection_a_onto_b_l36_3624

noncomputable section

open Real

def a : ℝ × ℝ := (sqrt 3, 1)
def b (θ : ℝ) : ℝ × ℝ := (cos θ, sin θ)

theorem parallel_a_b (θ : ℝ) (hθ : 0 ≤ θ ∧ θ ≤ π) (h_parallel : (a.1 / a.2) = (b θ).1 / (b θ).2) : θ = π / 6 := sorry

theorem projection_a_onto_b (θ : ℝ) (hθ : 0 ≤ θ ∧ θ ≤ π) (h_proj : (sqrt 3 * cos θ + sin θ) = -sqrt 3) : b θ = (-1, 0) := sorry

end parallel_a_b_projection_a_onto_b_l36_3624


namespace prove_R_value_l36_3652

noncomputable def geometric_series (Q : ℕ) : ℕ :=
  (2^(Q + 1) - 1)

noncomputable def R (F : ℕ) : ℝ :=
  Real.sqrt (Real.log (1 + F) / Real.log 2)

theorem prove_R_value :
  let F := geometric_series 120
  R F = 11 :=
by
  sorry

end prove_R_value_l36_3652


namespace percentage_of_60_eq_15_l36_3662

-- Conditions provided in the problem
def percentage (p : ℚ) : ℚ := p / 100
def num : ℚ := 60
def fraction_of_num (p : ℚ) (n : ℚ) : ℚ := (percentage p) * n

-- Assertion to be proved
theorem percentage_of_60_eq_15 : fraction_of_num 25 num = 15 := 
by 
  show fraction_of_num 25 60 = 15
  sorry

end percentage_of_60_eq_15_l36_3662


namespace original_number_exists_l36_3614

theorem original_number_exists :
  ∃ x : ℝ, 10 * x = x + 2.7 ∧ x = 0.3 :=
by {
  sorry
}

end original_number_exists_l36_3614


namespace cone_volume_is_3_6_l36_3650

-- Define the given conditions
def is_maximum_volume_cone_with_cutoff (cone_volume cutoff_volume : ℝ) : Prop :=
  cutoff_volume = 2 * cone_volume

def volume_difference (cutoff_volume cone_volume difference : ℝ) : Prop :=
  cutoff_volume - cone_volume = difference

-- The theorem to prove the volume of the cone
theorem cone_volume_is_3_6 
  (cone_volume cutoff_volume difference: ℝ)  
  (h1: is_maximum_volume_cone_with_cutoff cone_volume cutoff_volume)
  (h2: volume_difference cutoff_volume cone_volume 3.6) 
  : cone_volume = 3.6 :=
sorry

end cone_volume_is_3_6_l36_3650


namespace area_of_rectangle_is_588_l36_3667

-- Define the conditions
def radius_of_circle := 7
def width_of_rectangle := 2 * radius_of_circle
def length_to_width_ratio := 3

-- Define the width and length of the rectangle based on the conditions
def width := width_of_rectangle
def length := length_to_width_ratio * width_of_rectangle

-- Define the area of the rectangle
def area_of_rectangle := length * width

-- The theorem to prove
theorem area_of_rectangle_is_588 : area_of_rectangle = 588 :=
by sorry -- Proof is not required

end area_of_rectangle_is_588_l36_3667


namespace rectangle_area_l36_3621

theorem rectangle_area (a : ℕ) (h : 2 * (3 * a + 2 * a) = 160) : 3 * a * 2 * a = 1536 :=
by
  sorry

end rectangle_area_l36_3621


namespace novel_pages_l36_3629

theorem novel_pages (x : ℕ) (pages_per_day_in_reality : ℕ) (planned_days actual_days : ℕ)
  (h1 : planned_days = 20)
  (h2 : actual_days = 15)
  (h3 : pages_per_day_in_reality = x + 20)
  (h4 : pages_per_day_in_reality * actual_days = x * planned_days) :
  x * planned_days = 1200 :=
by
  sorry

end novel_pages_l36_3629


namespace sampling_scheme_exists_l36_3630

theorem sampling_scheme_exists : 
  ∃ (scheme : List ℕ → List (List ℕ)), 
    ∀ (p : List ℕ), p.length = 100 → (scheme p).length = 20 :=
by
  sorry

end sampling_scheme_exists_l36_3630


namespace max_sum_of_positive_integers_with_product_144_l36_3607

theorem max_sum_of_positive_integers_with_product_144 :
  ∃ (a b c : ℕ), a > 0 ∧ b > 0 ∧ c > 0 ∧ a * b * c = 144 ∧ a ≠ b ∧ b ≠ c ∧ a ≠ c ∧ a + b + c = 75 := 
by
  sorry

end max_sum_of_positive_integers_with_product_144_l36_3607


namespace simplify_expression_l36_3694

theorem simplify_expression :
  ((4 * 7) / (12 * 14)) * ((9 * 12 * 14) / (4 * 7 * 9)) ^ 2 = 1 := 
by
  sorry

end simplify_expression_l36_3694


namespace range_of_k_l36_3632

theorem range_of_k (k : ℝ) :
  (∀ x : ℝ, x^2 - k*x + 1 > 0) → (-2 < k ∧ k < 2) :=
by
  sorry

end range_of_k_l36_3632


namespace fraction_of_total_students_l36_3642

variables (G B T : ℕ) (F : ℚ)

-- Given conditions
axiom ratio_boys_to_girls : (7 : ℚ) / 3 = B / G
axiom total_students : T = B + G
axiom fraction_equals_two_thirds_girls : (2 : ℚ) / 3 * G = F * T

-- Proof goal
theorem fraction_of_total_students : F = 1 / 5 :=
by
  sorry

end fraction_of_total_students_l36_3642


namespace find_a_minus_c_l36_3670

theorem find_a_minus_c (a b c : ℝ) (h1 : (a + b) / 2 = 110) (h2 : (b + c) / 2 = 170) : a - c = -120 :=
by
  sorry

end find_a_minus_c_l36_3670


namespace least_positive_divisible_l36_3610

/-- The first five different prime numbers are given as conditions: -/
def prime1 := 2
def prime2 := 3
def prime3 := 5
def prime4 := 7
def prime5 := 11

/-- The least positive whole number divisible by the first five primes is 2310. -/
theorem least_positive_divisible :
  ∃ n : ℕ, n > 0 ∧ (n % prime1 = 0) ∧ (n % prime2 = 0) ∧ (n % prime3 = 0) ∧ (n % prime4 = 0) ∧ (n % prime5 = 0) ∧ n = 2310 :=
sorry

end least_positive_divisible_l36_3610


namespace license_plate_combinations_l36_3628

-- Definitions of the conditions
def num_consonants : ℕ := 20
def num_vowels : ℕ := 6
def num_digits : ℕ := 10

-- The theorem statement
theorem license_plate_combinations : num_consonants * num_vowels * num_vowels * num_digits = 7200 := by
  sorry

end license_plate_combinations_l36_3628


namespace harmonic_mean_pairs_l36_3633

open Nat

theorem harmonic_mean_pairs :
  ∃ n, n = 199 ∧ 
  (∀ (x y : ℕ), 0 < x → 0 < y → 
  x < y → (2 * x * y) / (x + y) = 6^10 → 
  x * y - (3^10 * 2^9) * (x - 1) - (3^10 * 2^9) * (y - 1) = 3^20 * 2^18) :=
sorry

end harmonic_mean_pairs_l36_3633


namespace geometric_sequence_a9_l36_3669

theorem geometric_sequence_a9
  (a : ℕ → ℤ)
  (q : ℤ)
  (h1 : a 3 * a 6 = -32)
  (h2 : a 4 + a 5 = 4)
  (hq : ∃ n : ℤ, q = n)
  : a 10 = -256 := 
sorry

end geometric_sequence_a9_l36_3669


namespace remainder_is_90_l36_3606

theorem remainder_is_90:
  let larger_number := 2982
  let smaller_number := 482
  let quotient := 6
  (larger_number - smaller_number = 2500) ∧ 
  (larger_number = quotient * smaller_number + r) →
  (r = 90) :=
by
  sorry

end remainder_is_90_l36_3606


namespace number_of_zeros_among_50_numbers_l36_3623

theorem number_of_zeros_among_50_numbers :
  ∀ (m n p : ℕ), (m + n + p = 50) → (m * p = 500) → n = 5 :=
by
  intros m n p h1 h2
  sorry

end number_of_zeros_among_50_numbers_l36_3623


namespace midpoint_trajectory_extension_trajectory_l36_3638

-- Define the conditions explicitly

def is_midpoint (M A O : ℝ × ℝ) : Prop :=
  M = ((O.1 + A.1) / 2, (O.2 + A.2) / 2)

def on_circle (P : ℝ × ℝ) : Prop :=
  P.1 ^ 2 + P.2 ^ 2 - 8 * P.1 = 0

-- First problem: Trajectory equation of the midpoint M
theorem midpoint_trajectory (M O A : ℝ × ℝ) (hO : O = (0,0)) (hA : on_circle A) (hM : is_midpoint M A O) :
  M.1 ^ 2 + M.2 ^ 2 - 4 * M.1 = 0 :=
sorry

-- Define the condition for N
def extension_point (O A N : ℝ × ℝ) : Prop :=
  (A.1 - O.1) * 2 = N.1 - O.1 ∧ (A.2 - O.2) * 2 = N.2 - O.2

-- Second problem: Trajectory equation of the point N
theorem extension_trajectory (N O A : ℝ × ℝ) (hO : O = (0,0)) (hA : on_circle A) (hN : extension_point O A N) :
  N.1 ^ 2 + N.2 ^ 2 - 16 * N.1 = 0 :=
sorry

end midpoint_trajectory_extension_trajectory_l36_3638


namespace points_on_opposite_sides_of_line_l36_3682

theorem points_on_opposite_sides_of_line 
  (a : ℝ) 
  (h : (3 * -3 - 2 * -1 - a) * (3 * 4 - 2 * -6 - a) < 0) : 
  -7 < a ∧ a < 24 :=
sorry

end points_on_opposite_sides_of_line_l36_3682


namespace number_of_squares_in_grid_l36_3658

-- Grid of size 6 × 6 composed entirely of squares.
def grid_size : Nat := 6

-- Definition of the function that counts the number of squares of a given size in an n × n grid.
def count_squares (n : Nat) (size : Nat) : Nat :=
  (n - size + 1) * (n - size + 1)

noncomputable def total_squares : Nat :=
  List.sum (List.map (count_squares grid_size) (List.range grid_size).tail)  -- Using tail to skip zero size

theorem number_of_squares_in_grid : total_squares = 86 := by
  sorry

end number_of_squares_in_grid_l36_3658


namespace intersection_points_l36_3611

def curve (x y : ℝ) : Prop := x^2 + y^2 = 1
def line (x y : ℝ) : Prop := y = x + 1

theorem intersection_points :
  {p : ℝ × ℝ | curve p.1 p.2 ∧ line p.1 p.2} = {(-1, 0), (0, 1)} :=
by 
  sorry

end intersection_points_l36_3611


namespace Jerry_walked_9_miles_l36_3699

theorem Jerry_walked_9_miles (x : ℕ) (h : 2 * x = 18) : x = 9 := 
by
  sorry

end Jerry_walked_9_miles_l36_3699


namespace find_m_pure_imaginary_l36_3616

theorem find_m_pure_imaginary (m : ℝ) (h : m^2 + m - 2 + (m^2 - 1) * I = (0 : ℝ) + (m^2 - 1) * I) :
  m = -2 :=
by {
  sorry
}

end find_m_pure_imaginary_l36_3616


namespace gcd_4320_2550_l36_3637

-- Definitions for 4320 and 2550
def a : ℕ := 4320
def b : ℕ := 2550

-- Statement to prove the greatest common factor of a and b is 30
theorem gcd_4320_2550 : Nat.gcd a b = 30 := 
by 
  sorry

end gcd_4320_2550_l36_3637


namespace marbles_count_l36_3645

-- Define the condition variables
variable (M : ℕ) -- total number of marbles placed on Monday
variable (day2_marbles : ℕ) -- marbles remaining after second day
variable (day3_cleo_marbles : ℕ) -- marbles taken by Cleo on third day

-- Condition definitions
def condition1 : Prop := day2_marbles = 2 * M / 5
def condition2 : Prop := day3_cleo_marbles = (day2_marbles / 2)
def condition3 : Prop := day3_cleo_marbles = 15

-- The theorem to prove
theorem marbles_count : 
  condition1 M day2_marbles → 
  condition2 day2_marbles day3_cleo_marbles → 
  condition3 day3_cleo_marbles → 
  M = 75 :=
by
  intros h1 h2 h3
  sorry

end marbles_count_l36_3645


namespace number_of_girls_l36_3657

theorem number_of_girls (num_vans : ℕ) (students_per_van : ℕ) (num_boys : ℕ) (total_students : ℕ) (num_girls : ℕ) 
(h1 : num_vans = 5) 
(h2 : students_per_van = 28) 
(h3 : num_boys = 60) 
(h4 : total_students = num_vans * students_per_van) 
(h5 : num_girls = total_students - num_boys) : 
num_girls = 80 :=
by
  sorry

end number_of_girls_l36_3657


namespace total_cost_correct_l36_3695

def bun_price : ℝ := 0.1
def buns_count : ℝ := 10
def milk_price : ℝ := 2
def milk_count : ℝ := 2
def egg_price : ℝ := 3 * milk_price

def total_cost : ℝ := (buns_count * bun_price) + (milk_count * milk_price) + egg_price

theorem total_cost_correct : total_cost = 11 := by
  sorry

end total_cost_correct_l36_3695


namespace even_function_odd_function_neither_even_nor_odd_function_l36_3692

def f (x : ℝ) : ℝ := 1 + x^2 + x^4
def g (x : ℝ) : ℝ := x + x^3 + x^5
def h (x : ℝ) : ℝ := 1 + x + x^2 + x^3 + x^4

theorem even_function : ∀ x : ℝ, f (-x) = f x :=
by
  sorry

theorem odd_function : ∀ x : ℝ, g (-x) = -g x :=
by
  sorry

theorem neither_even_nor_odd_function : ∀ x : ℝ, (h (-x) ≠ h x) ∧ (h (-x) ≠ -h x) :=
by
  sorry

end even_function_odd_function_neither_even_nor_odd_function_l36_3692


namespace ellipse_standard_equation_and_point_l36_3672
  
noncomputable def ellipse_equation (x y : ℝ) : Prop :=
  (x^2) / 25 + (y^2) / 9 = 1

def exists_dot_product_zero_point (P : ℝ × ℝ) : Prop :=
  let F1 := (-4, 0)
  let F2 := (4, 0)
  (P.1 + 4) * (P.1 - 4) + P.2 * P.2 = 0

theorem ellipse_standard_equation_and_point :
  ∃ (P : ℝ × ℝ), ellipse_equation P.1 P.2 ∧ exists_dot_product_zero_point P ∧ 
    ((P = ((5 * Real.sqrt 7) / 4, 9 / 4)) ∨ (P = (-(5 * Real.sqrt 7) / 4, 9 / 4)) ∨ 
    (P = ((5 * Real.sqrt 7) / 4, -(9 / 4))) ∨ (P = (-(5 * Real.sqrt 7) / 4, -(9 / 4)))) :=
by 
  sorry

end ellipse_standard_equation_and_point_l36_3672


namespace solve_a_value_l36_3641

theorem solve_a_value (a b k : ℝ) 
  (h1 : a^3 * b^2 = k)
  (h2 : a = 5)
  (h3 : b = 2) :
  ∃ a', b = 8 → a' = 2.5 :=
by
  sorry

end solve_a_value_l36_3641


namespace trigonometric_identity_l36_3625

noncomputable def π := Real.pi
noncomputable def tan (x : ℝ) := Real.sin x / Real.cos x

theorem trigonometric_identity (α β : ℝ) 
  (hα : 0 < α ∧ α < π / 2) 
  (hβ : 0 < β ∧ β < π / 2) 
  (h : tan α = (1 + Real.sin β) / Real.cos β) :
  2 * α - β = π / 2 := 
sorry

end trigonometric_identity_l36_3625


namespace solve_cubic_eq_l36_3673

theorem solve_cubic_eq (z : ℂ) : z^3 = 27 ↔ (z = 3 ∨ z = - (3 / 2) + (3 / 2) * Complex.I * Real.sqrt 3 ∨ z = - (3 / 2) - (3 / 2) * Complex.I * Real.sqrt 3) :=
by
  sorry

end solve_cubic_eq_l36_3673


namespace find_d_l36_3605

theorem find_d (m a b d : ℕ) 
(hm : 0 < m) 
(ha : m^2 < a ∧ a < m^2 + m) 
(hb : m^2 < b ∧ b < m^2 + m) 
(hab : a ≠ b)
(hd : m^2 < d ∧ d < m^2 + m ∧ d ∣ (a * b)) : 
d = a ∨ d = b :=
sorry

end find_d_l36_3605


namespace farmer_cows_after_selling_l36_3664

theorem farmer_cows_after_selling
  (initial_cows : ℕ) (new_cows : ℕ) (quarter_factor : ℕ)
  (h_initial : initial_cows = 51)
  (h_new : new_cows = 5)
  (h_quarter : quarter_factor = 4) :
  initial_cows + new_cows - (initial_cows + new_cows) / quarter_factor = 42 :=
by
  sorry

end farmer_cows_after_selling_l36_3664


namespace determine_original_number_l36_3612

theorem determine_original_number (a b c : ℕ) (m : ℕ) (N : ℕ) 
  (h1 : N = 4410) 
  (h2 : (a + b + c) % 2 = 0)
  (h3 : m = 100 * a + 10 * b + c)
  (h4 : N + m = 222 * (a + b + c)) : 
  a = 4 ∧ b = 4 ∧ c = 4 :=
by 
  sorry

end determine_original_number_l36_3612


namespace complex_fourth_power_l36_3693

theorem complex_fourth_power (i : ℂ) (hi : i^2 = -1) : (1 - i)^4 = -4 := 
sorry

end complex_fourth_power_l36_3693


namespace probability_A_does_not_lose_l36_3685

theorem probability_A_does_not_lose (p_tie p_A_win : ℚ) (h_tie : p_tie = 1 / 2) (h_A_win : p_A_win = 1 / 3) :
  p_tie + p_A_win = 5 / 6 :=
by sorry

end probability_A_does_not_lose_l36_3685


namespace sqrt_sum_eq_five_l36_3698

theorem sqrt_sum_eq_five
  (x : ℝ)
  (h1 : -Real.sqrt 15 ≤ x ∧ x ≤ Real.sqrt 15)
  (h2 : Real.sqrt (25 - x^2) - Real.sqrt (15 - x^2) = 2) :
  Real.sqrt (25 - x^2) + Real.sqrt (15 - x^2) = 5 := by
  sorry

end sqrt_sum_eq_five_l36_3698


namespace xy_difference_squared_l36_3646

theorem xy_difference_squared (x y : ℝ) (h1 : x + y = 8) (h2 : x - y = 4) : x^2 - y^2 = 32 :=
by
  -- the proof goes here
  sorry

end xy_difference_squared_l36_3646


namespace y_intercept_of_line_l36_3675

/-- Let m be the slope of a line and (x_intercept, 0) be the x-intercept of the same line.
    If the line passes through the point (3, 0) and has a slope of -3, then its y-intercept is (0, 9). -/
theorem y_intercept_of_line 
    (m : ℝ) (x_intercept : ℝ) (x1 y1 : ℝ)
    (h1 : m = -3)
    (h2 : (x_intercept, 0) = (3, 0)) :
    (0, -m * x_intercept) = (0, 9) :=
by sorry

end y_intercept_of_line_l36_3675


namespace contradiction_divisible_by_2_l36_3601

open Nat

theorem contradiction_divisible_by_2 (a b : ℕ) (h : (a * b) % 2 = 0) : a % 2 = 0 ∨ b % 2 = 0 :=
by
  sorry

end contradiction_divisible_by_2_l36_3601


namespace complement_of_union_l36_3691

def A : Set ℤ := {x | ∃ k : ℤ, x = 3 * k + 1}
def B : Set ℤ := {x | ∃ k : ℤ, x = 3 * k + 2}
def U : Set ℤ := Set.univ

theorem complement_of_union (x : ℤ) : (x ∉ A ∪ B) ↔ ∃ k : ℤ, x = 3 * k := 
by {
  sorry
}

end complement_of_union_l36_3691


namespace valid_B_sets_l36_3688

def A : Set ℝ := {x | 0 < x ∧ x < 2}

theorem valid_B_sets (B : Set ℝ) : A ∩ B = B ↔ B = ∅ ∨ B = {1} ∨ B = A :=
by
  sorry

end valid_B_sets_l36_3688


namespace Tim_driving_hours_l36_3696

theorem Tim_driving_hours (D T : ℕ) (h1 : T = 2 * D) (h2 : D + T = 15) : D = 5 :=
by
  sorry

end Tim_driving_hours_l36_3696


namespace sector_angle_l36_3644

theorem sector_angle (r : ℝ) (S_sector : ℝ) (h_r : r = 2) (h_S : S_sector = (2 / 5) * π) : 
  (∃ α : ℝ, S_sector = (1 / 2) * α * r^2 ∧ α = (π / 5)) :=
by
  use π / 5
  sorry

end sector_angle_l36_3644


namespace necessary_but_not_sufficient_l36_3631

theorem necessary_but_not_sufficient (x : ℝ) : (x < 0) -> (x^2 + x < 0 ↔ -1 < x ∧ x < 0) :=
by
  sorry

end necessary_but_not_sufficient_l36_3631


namespace chocolate_bars_per_small_box_l36_3620

theorem chocolate_bars_per_small_box (total_chocolate_bars small_boxes : ℕ) 
  (h1 : total_chocolate_bars = 442) 
  (h2 : small_boxes = 17) : 
  total_chocolate_bars / small_boxes = 26 :=
by
  sorry

end chocolate_bars_per_small_box_l36_3620


namespace duration_of_each_turn_l36_3615

-- Definitions based on conditions
def Wa := 1 / 4
def Wb := 1 / 12

-- Define the duration of each turn as T
def T : ℝ := 1 -- This is the correct answer we proved

-- Given conditions
def total_work_done := 6 * Wa + 6 * Wb

-- Lean statement to prove 
theorem duration_of_each_turn : T = 1 := by
  -- According to conditions, the total work done by a and b should equal the whole work
  have h1 : 3 * Wa + 3 * Wb = 1 := by sorry
  -- Let's conclude that T = 1
  sorry

end duration_of_each_turn_l36_3615


namespace area_OMVK_l36_3648

def AreaOfQuadrilateral (S_OKSL S_ONAM S_OMVK : ℝ) : ℝ :=
  let S_ABCD := 4 * (S_OKSL + S_ONAM)
  S_ABCD - S_OKSL - 24 - S_ONAM

theorem area_OMVK {S_OKSL S_ONAM : ℝ} (h_OKSL : S_OKSL = 6) (h_ONAM : S_ONAM = 12) : 
  AreaOfQuadrilateral S_OKSL S_ONAM 30 = 30 :=
by
  sorry

end area_OMVK_l36_3648


namespace distance_ran_by_Juan_l36_3634

-- Definitions based on the condition
def speed : ℝ := 10 -- in miles per hour
def time : ℝ := 8 -- in hours

-- Theorem statement
theorem distance_ran_by_Juan : speed * time = 80 := by
  sorry

end distance_ran_by_Juan_l36_3634


namespace video_game_price_l36_3651

theorem video_game_price (total_games not_working_games : ℕ) (total_earnings : ℕ)
  (h1 : total_games = 10) (h2 : not_working_games = 2) (h3 : total_earnings = 32) :
  ((total_games - not_working_games) > 0) →
  (total_earnings / (total_games - not_working_games)) = 4 :=
by
  sorry

end video_game_price_l36_3651


namespace similar_terms_solution_l36_3690

theorem similar_terms_solution
  (a b : ℝ)
  (m n x y : ℤ)
  (h1 : m - 1 = n - 2 * m)
  (h2 : m + n = 3 * m + n - 4)
  (h3 : m * x + (n - 2) * y = 24)
  (h4 : 2 * m * x + n * y = 46) :
  x = 9 ∧ y = 2 := by
  sorry

end similar_terms_solution_l36_3690


namespace race_order_count_l36_3676

-- Define the problem conditions
def participants : List String := ["Harry", "Ron", "Neville", "Hermione"]
def no_ties : Prop := True -- Since no ties are given directly, we denote this as always true for simplicity

-- Define the proof problem statement
theorem race_order_count (h_no_ties : no_ties) : participants.permutations.length = 24 := 
by
  -- Placeholder for proof
  sorry

end race_order_count_l36_3676


namespace ants_species_A_count_l36_3609

theorem ants_species_A_count (a b : ℕ) (h1 : a + b = 30) (h2 : 2^5 * a + 3^5 * b = 3281) : 32 * a = 608 :=
by
  sorry

end ants_species_A_count_l36_3609


namespace total_cows_in_ranch_l36_3655

def WeThePeopleCows : ℕ := 17
def HappyGoodHealthyFamilyCows : ℕ := 3 * WeThePeopleCows + 2

theorem total_cows_in_ranch : WeThePeopleCows + HappyGoodHealthyFamilyCows = 70 := by
  sorry

end total_cows_in_ranch_l36_3655


namespace problem_statement_l36_3686

theorem problem_statement {x₁ x₂ : ℝ} (h1 : 3 * x₁^2 - 9 * x₁ - 21 = 0) (h2 : 3 * x₂^2 - 9 * x₂ - 21 = 0) :
  (3 * x₁ - 4) * (6 * x₂ - 8) = -202 := sorry

end problem_statement_l36_3686

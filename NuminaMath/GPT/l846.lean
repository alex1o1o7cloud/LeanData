import Mathlib

namespace inequality_solution_l846_84692

theorem inequality_solution (x y : ℝ) (h : 5 * x > -5 * y) : x + y > 0 :=
sorry

end inequality_solution_l846_84692


namespace arithmetic_sequence_a3_l846_84682

variable {a : ℕ → ℝ}  -- Define the sequence as a function from natural numbers to real numbers.

-- Definition that the sequence is arithmetic.
def is_arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∀ n m, a (n + 1) - a n = a (m + 1) - a m

-- The given condition in the problem
axiom h1 : a 1 + a 5 = 6

-- The statement to prove
theorem arithmetic_sequence_a3 (h : is_arithmetic_sequence a) : a 3 = 3 :=
by {
  -- The proof is omitted.
  sorry
}

end arithmetic_sequence_a3_l846_84682


namespace Randy_initial_money_l846_84610

theorem Randy_initial_money (M : ℝ) (r1 : M + 200 - 1200 = 2000) : M = 3000 :=
by
  sorry

end Randy_initial_money_l846_84610


namespace find_difference_l846_84643

theorem find_difference (m n : ℕ) (hm : ∃ x, m = 111 * x) (hn : ∃ y, n = 31 * y) (h_sum : m + n = 2017) :
  n - m = 463 :=
sorry

end find_difference_l846_84643


namespace MrFletcherPaymentPerHour_l846_84671

theorem MrFletcherPaymentPerHour :
  (2 * (10 + 8 + 15)) * x = 660 → x = 10 :=
by
  -- This is where you'd provide the proof, but we skip it as per instructions.
  sorry

end MrFletcherPaymentPerHour_l846_84671


namespace football_team_lineup_count_l846_84633

theorem football_team_lineup_count :
  let team_members := 12
  let offensive_lineman_choices := 5
  let remaining_choices := team_members - 1
  let quarterback_choices := remaining_choices
  let running_back_choices := remaining_choices - 1
  let wide_receiver_choices := remaining_choices - 2
  let tight_end_choices := remaining_choices - 3

  offensive_lineman_choices * quarterback_choices * running_back_choices * wide_receiver_choices * tight_end_choices = 39600 :=
by
  let team_members := 12
  let offensive_lineman_choices := 5
  let remaining_choices := team_members - 1
  let quarterback_choices := remaining_choices
  let running_back_choices := remaining_choices - 1
  let wide_receiver_choices := remaining_choices - 2
  let tight_end_choices := remaining_choices - 3
  
  exact sorry

end football_team_lineup_count_l846_84633


namespace arithmetic_sequence_8th_term_l846_84657

theorem arithmetic_sequence_8th_term (a d : ℤ)
  (h4 : a + 3 * d = 23) 
  (h6 : a + 5 * d = 47) : 
  a + 7 * d = 71 := 
by
  sorry

end arithmetic_sequence_8th_term_l846_84657


namespace pond_field_area_ratio_l846_84615

theorem pond_field_area_ratio
  (l : ℝ) (w : ℝ) (A_field : ℝ) (A_pond : ℝ)
  (h1 : l = 2 * w)
  (h2 : l = 16)
  (h3 : A_field = l * w)
  (h4 : A_pond = 8 * 8) :
  A_pond / A_field = 1 / 2 :=
by
  sorry

end pond_field_area_ratio_l846_84615


namespace weight_difference_l846_84684

theorem weight_difference :
  let Box_A := 2.4
  let Box_B := 5.3
  let Box_C := 13.7
  let Box_D := 7.1
  let Box_E := 10.2
  let Box_F := 3.6
  let Box_G := 9.5
  max Box_A (max Box_B (max Box_C (max Box_D (max Box_E (max Box_F Box_G))))) -
  min Box_A (min Box_B (min Box_C (min Box_D (min Box_E (min Box_F Box_G))))) = 11.3 :=
by
  sorry

end weight_difference_l846_84684


namespace scientific_notation_correct_l846_84648

def num : ℝ := 1040000000

theorem scientific_notation_correct : num = 1.04 * 10^9 := by
  sorry

end scientific_notation_correct_l846_84648


namespace distance_to_directrix_l846_84685

noncomputable def ellipse (x y : ℝ) : Prop := x^2 / 64 + y^2 / 28 = 1

noncomputable def left_focus : ℝ × ℝ := (-6, 0)

noncomputable def right_focus : ℝ × ℝ := (6, 0)

noncomputable def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p1.1 - p2.1) ^ 2 + (p1.2 - p2.2) ^ 2)

theorem distance_to_directrix (P : ℝ × ℝ) (hP : ellipse P.1 P.2) (hPF1 : distance P left_focus = 4) :
  distance P right_focus * 4 / 3 = 16 :=
sorry

end distance_to_directrix_l846_84685


namespace arccos_one_eq_zero_l846_84668

theorem arccos_one_eq_zero : Real.arccos 1 = 0 := by
  sorry

end arccos_one_eq_zero_l846_84668


namespace remainder_six_pow_4032_mod_13_l846_84678

theorem remainder_six_pow_4032_mod_13 : (6 ^ 4032) % 13 = 1 := 
by
  sorry

end remainder_six_pow_4032_mod_13_l846_84678


namespace batsman_sixes_l846_84651

theorem batsman_sixes (total_runs : ℕ) (boundaries : ℕ) (running_percentage : ℝ) (score_per_boundary : ℕ) (score_per_six : ℕ)
  (h1 : total_runs = 150)
  (h2 : boundaries = 5)
  (h3 : running_percentage = 66.67)
  (h4 : score_per_boundary = 4)
  (h5 : score_per_six = 6) :
  ∃ (sixes : ℕ), sixes = 5 :=
by
  -- Calculations omitted
  existsi 5
  sorry

end batsman_sixes_l846_84651


namespace cs_share_l846_84656

-- Definitions for the conditions
def daily_work (days: ℕ) : ℚ := 1 / days

def total_work_contribution (a_days: ℕ) (b_days: ℕ) (c_days: ℕ): ℚ := 
  daily_work a_days + daily_work b_days + daily_work c_days

def total_payment (payment: ℕ) (work_contribution: ℚ) : ℚ := 
  payment * work_contribution

-- The mathematically equivalent proof problem
theorem cs_share (a_days: ℕ) (b_days: ℕ) (total_days : ℕ) (payment: ℕ) : 
  a_days = 6 → b_days = 8 → total_days = 3 → payment = 1200 →
  total_payment payment (daily_work total_days - (daily_work a_days + daily_work b_days)) = 50 :=
sorry

end cs_share_l846_84656


namespace quadratic_root_l846_84680

theorem quadratic_root (a : ℝ) : (∃ x : ℝ, x = 1 ∧ a * x^2 + x - 2 = 0) → a = 1 := by
  sorry

end quadratic_root_l846_84680


namespace proof_of_problem_l846_84696

def problem_statement : Prop :=
  2 * Real.cos (Real.pi / 4) + abs (Real.sqrt 2 - 3)
  - (1 / 3) ^ (-2 : ℤ) + (2021 - Real.pi) ^ 0 = -5

theorem proof_of_problem : problem_statement :=
by
  sorry

end proof_of_problem_l846_84696


namespace inequality1_inequality2_l846_84655

-- Problem 1
def f (x : ℝ) : ℝ := |2 * x + 1| + |x - 1|

theorem inequality1 (x : ℝ) : f x > 2 ↔ x < -2/3 ∨ x > 0 := sorry

-- Problem 2
def g (x : ℝ) : ℝ := f x + f (-x)

theorem inequality2 (k : ℝ) (h : ∀ x : ℝ, |k - 1| < g x) : -3 < k ∧ k < 5 := sorry

end inequality1_inequality2_l846_84655


namespace solution_set_of_inequality_l846_84646

theorem solution_set_of_inequality (x : ℝ) :
  x * |x - 1| > 0 ↔ (0 < x ∧ x < 1) ∨ (x > 1) :=
sorry

end solution_set_of_inequality_l846_84646


namespace find_a5_from_geometric_sequence_l846_84649

def geo_seq (a : ℕ → ℝ) (q : ℝ) := ∀ n, a (n + 1) = a n * q

noncomputable def geometric_sequence_property (a : ℕ → ℝ) (q : ℝ) :=
  geo_seq a q ∧ 0 < a 1 ∧ 0 < q ∧ 
  (a 4 = (a 2) ^ 2) ∧ 
  (a 2 + a 4 = 5 / 16)

theorem find_a5_from_geometric_sequence :
  ∀ (a : ℕ → ℝ) (q : ℝ), geometric_sequence_property a q → 
  a 5 = 1 / 32 :=
by 
  sorry

end find_a5_from_geometric_sequence_l846_84649


namespace polar_to_cartesian_circle_l846_84689

theorem polar_to_cartesian_circle :
  ∀ (r : ℝ) (x y : ℝ), r = 3 → r = Real.sqrt (x^2 + y^2) → x^2 + y^2 = 9 :=
by
  intros r x y hr h
  sorry

end polar_to_cartesian_circle_l846_84689


namespace fraction_q_p_l846_84641

theorem fraction_q_p (k : ℝ) (c p q : ℝ) (h : 8 * k^2 - 12 * k + 20 = c * (k + p)^2 + q) :
  c = 8 ∧ p = -3/4 ∧ q = 31/2 → q / p = -62 / 3 :=
by
  intros hc_hp_hq
  sorry

end fraction_q_p_l846_84641


namespace pattern_continues_for_max_8_years_l846_84660

def is_adult_age (age : ℕ) := 18 ≤ age ∧ age < 40

def fits_pattern (p1 p2 n : ℕ) : Prop := 
  is_adult_age p1 ∧
  is_adult_age p2 ∧ 
  (∀ (k : ℕ), 1 ≤ k ∧ k ≤ n → 
    (k % (p1 + k) = 0 ∨ k % (p2 + k) = 0) ∧ ¬ (k % (p1 + k) = 0 ∧ k % (p2 + k) = 0))

theorem pattern_continues_for_max_8_years (p1 p2 : ℕ) : 
  fits_pattern p1 p2 8 := 
sorry

end pattern_continues_for_max_8_years_l846_84660


namespace reciprocal_of_neg_four_l846_84603

def is_reciprocal (x y : ℚ) : Prop := x * y = 1

theorem reciprocal_of_neg_four : is_reciprocal (-4) (-1/4) :=
by
  sorry

end reciprocal_of_neg_four_l846_84603


namespace number_of_rabbits_l846_84623

theorem number_of_rabbits
  (dogs : ℕ) (cats : ℕ) (total_animals : ℕ)
  (joins_each_cat : ℕ → ℕ)
  (hares_per_rabbit : ℕ)
  (h_dogs : dogs = 1)
  (h_cats : cats = 4)
  (h_total : total_animals = 37)
  (h_hares_per_rabbit : hares_per_rabbit = 3)
  (H : total_animals = dogs + cats + 4 * joins_each_cat cats + 3 * 4 * joins_each_cat cats) :
  joins_each_cat cats = 2 :=
by
  sorry

end number_of_rabbits_l846_84623


namespace smallest_value_of_c_l846_84628

theorem smallest_value_of_c :
  ∃ c : ℚ, (3 * c + 4) * (c - 2) = 9 * c ∧ (∀ d : ℚ, (3 * d + 4) * (d - 2) = 9 * d → c ≤ d) ∧ c = -8 / 3 := 
sorry

end smallest_value_of_c_l846_84628


namespace ratio_lcm_gcf_l846_84600

theorem ratio_lcm_gcf (a b : ℕ) (h₁ : a = 2^2 * 3^2 * 7) (h₂ : b = 2 * 3^2 * 5 * 7) :
  (Nat.lcm a b) / (Nat.gcd a b) = 10 := by
  sorry

end ratio_lcm_gcf_l846_84600


namespace remainder_division_l846_84605

theorem remainder_division (k : ℤ) (N : ℤ) (h : N = 133 * k + 16) : N % 50 = 49 := by
  sorry

end remainder_division_l846_84605


namespace david_marks_in_mathematics_l846_84675

-- Define marks in individual subjects and the average
def marks_in_english : ℝ := 70
def marks_in_physics : ℝ := 78
def marks_in_chemistry : ℝ := 60
def marks_in_biology : ℝ := 65
def average_marks : ℝ := 66.6
def number_of_subjects : ℕ := 5

-- Define a statement to be proven
theorem david_marks_in_mathematics : 
    average_marks * number_of_subjects 
    - (marks_in_english + marks_in_physics + marks_in_chemistry + marks_in_biology) = 60 := 
by simp [average_marks, number_of_subjects, marks_in_english, marks_in_physics, marks_in_chemistry, marks_in_biology]; sorry

end david_marks_in_mathematics_l846_84675


namespace geometric_sequence_properties_l846_84611

theorem geometric_sequence_properties 
  (a : ℕ → ℝ)
  (S : ℕ → ℝ)
  (h_geom : ∀ n, a (n + 1) / a n = a 2 / a 1)
  (h_pos : ∀ n, a n > 0)
  (h_a1 : a 1 = 1)
  (h_a3 : a 3 = 4) :
  (∀ n, a n = 2^(n - 1)) ∧ (S 6 = 63) := 
by 
  sorry

end geometric_sequence_properties_l846_84611


namespace remainder_of_n_plus_4500_l846_84622

theorem remainder_of_n_plus_4500 (n : ℕ) (h : n % 6 = 1) : (n + 4500) % 6 = 1 := 
by
  sorry

end remainder_of_n_plus_4500_l846_84622


namespace complex_purely_imaginary_l846_84639

theorem complex_purely_imaginary (x : ℝ) :
  (x^2 - 1 = 0) → (x - 1 ≠ 0) → x = -1 :=
by
  intro h1 h2
  sorry

end complex_purely_imaginary_l846_84639


namespace intersection_A_B_l846_84686

def set_A : Set ℝ := { x | x ≥ 0 }
def set_B : Set ℝ := { x | -1 < x ∧ x < 1 }

theorem intersection_A_B : set_A ∩ set_B = { x | 0 ≤ x ∧ x < 1 } :=
by
  sorry

end intersection_A_B_l846_84686


namespace fraction_simplify_l846_84658

theorem fraction_simplify :
  (3 + 6 - 12 + 24 + 48 - 96 + 192) / (6 + 12 - 24 + 48 + 96 - 192 + 384) = 1 / 2 :=
by sorry

end fraction_simplify_l846_84658


namespace burger_cost_l846_84647

theorem burger_cost {B : ℝ} (sandwich_cost : ℝ) (smoothies_cost : ℝ) (total_cost : ℝ)
  (H1 : sandwich_cost = 4)
  (H2 : smoothies_cost = 8)
  (H3 : total_cost = 17)
  (H4 : B + sandwich_cost + smoothies_cost = total_cost) :
  B = 5 :=
sorry

end burger_cost_l846_84647


namespace f_12_eq_12_l846_84602

noncomputable def f : ℕ → ℤ := sorry

axiom f_int (n : ℕ) (hn : 0 < n) : ∃ k : ℤ, f n = k
axiom f_2 : f 2 = 2
axiom f_mul (m n : ℕ) (hm : 0 < m) (hn : 0 < n) : f (m * n) = f m * f n
axiom f_monotonic (m n : ℕ) (hm : 0 < m) (hn : 0 < n) : m > n → f m > f n

theorem f_12_eq_12 : f 12 = 12 := sorry

end f_12_eq_12_l846_84602


namespace probability_of_females_right_of_males_l846_84691

-- Defining the total and favorable outcomes
def total_outcomes : ℕ := Nat.factorial 5
def favorable_outcomes : ℕ := Nat.factorial 3 * Nat.factorial 2

-- Defining the probability as a rational number
def probability_all_females_right : ℚ := favorable_outcomes / total_outcomes

-- Stating the theorem
theorem probability_of_females_right_of_males :
  probability_all_females_right = 1 / 10 :=
by
  -- Proof to be filled in
  sorry

end probability_of_females_right_of_males_l846_84691


namespace biscuits_afternoon_eq_40_l846_84677

-- Define the initial conditions given in the problem.
def butter_cookies_afternoon : Nat := 10
def additional_biscuits : Nat := 30

-- Define the number of biscuits based on the initial conditions.
def biscuits_afternoon : Nat := butter_cookies_afternoon + additional_biscuits

-- The statement to prove according to the problem.
theorem biscuits_afternoon_eq_40 : biscuits_afternoon = 40 := by
  -- The proof is to be done, hence we use 'sorry'.
  sorry

end biscuits_afternoon_eq_40_l846_84677


namespace possible_values_a_l846_84609

def A : Set ℝ := {-1, 2}
def B (a : ℝ) : Set ℝ := {x | a * x^2 = 2 ∧ a ≥ 0}

def whale_swallowing (S T : Set ℝ) : Prop :=
S ⊆ T ∨ T ⊆ S

def moth_eating (S T : Set ℝ) : Prop :=
(∃ x, x ∈ S ∧ x ∈ T) ∧ ¬(S ⊆ T) ∧ ¬(T ⊆ S)

def valid_a (a : ℝ) : Prop :=
whale_swallowing A (B a) ∨ moth_eating A (B a)

theorem possible_values_a :
  {a : ℝ | valid_a a} = {0, 1/2, 2} :=
sorry

end possible_values_a_l846_84609


namespace kangaroo_mob_has_6_l846_84653

-- Define the problem conditions
def mob_of_kangaroos (W : ℝ) (k : ℕ) : Prop :=
  ∃ (two_lightest three_heaviest remaining : ℝ) (n_two n_three n_rem : ℕ),
    two_lightest = 0.25 * W ∧
    three_heaviest = 0.60 * W ∧
    remaining = 0.15 * W ∧
    n_two = 2 ∧
    n_three = 3 ∧
    n_rem = 1 ∧
    k = n_two + n_three + n_rem

-- The theorem to be proven
theorem kangaroo_mob_has_6 (W : ℝ) : ∃ k, mob_of_kangaroos W k ∧ k = 6 :=
by
  sorry

end kangaroo_mob_has_6_l846_84653


namespace perfect_square_adjacent_smaller_l846_84687

noncomputable def is_perfect_square (n : ℕ) : Prop := 
    ∃ k : ℕ, k * k = n

theorem perfect_square_adjacent_smaller (m : ℕ) (hm : is_perfect_square m) : 
    ∃ k : ℕ, (k * k = m ∧ (k - 1) * (k - 1) = m - 2 * k + 1) := 
by 
  sorry

end perfect_square_adjacent_smaller_l846_84687


namespace additional_carpet_needed_l846_84640

-- Definitions according to the given conditions
def length_feet := 18
def width_feet := 12
def covered_area := 4 -- in square yards
def feet_per_yard := 3

-- Prove that the additional square yards needed to cover the remaining part of the floor is 20
theorem additional_carpet_needed : 
  ((length_feet / feet_per_yard) * (width_feet / feet_per_yard) - covered_area) = 20 := 
by
  sorry

end additional_carpet_needed_l846_84640


namespace new_barbell_cost_l846_84635

theorem new_barbell_cost (old_barbell_cost new_barbell_cost : ℝ) 
  (h1 : old_barbell_cost = 250)
  (h2 : new_barbell_cost = old_barbell_cost * 1.3) :
  new_barbell_cost = 325 := by
  sorry

end new_barbell_cost_l846_84635


namespace percentage_selected_in_state_B_l846_84625

theorem percentage_selected_in_state_B (appeared: ℕ) (selectedA: ℕ) (selected_diff: ℕ)
  (percentage_selectedA: ℝ)
  (h1: appeared = 8100)
  (h2: percentage_selectedA = 6.0)
  (h3: selectedA = appeared * (percentage_selectedA / 100))
  (h4: selected_diff = 81)
  (h5: selectedB = selectedA + selected_diff) :
  ((selectedB : ℝ) / appeared) * 100 = 7 := 
  sorry

end percentage_selected_in_state_B_l846_84625


namespace person_B_spheres_needed_l846_84637

-- Translate conditions to Lean definitions
def sum_squares (n : ℕ) : ℕ := (n * (n + 1) * (2 * n + 1)) / 6
def sum_triangulars (m : ℕ) : ℕ := (m * (m + 1) * (m + 2)) / 6

-- Define the main theorem
theorem person_B_spheres_needed (n m : ℕ) (hA : sum_squares n = 2109)
    (hB : m ≥ 25) : sum_triangulars m = 2925 :=
    sorry

end person_B_spheres_needed_l846_84637


namespace alice_ate_more_l846_84666

theorem alice_ate_more (cookies : Fin 8 → ℕ) (h_alice : cookies 0 = 8) (h_tom : cookies 7 = 1) :
  cookies 0 - cookies 7 = 7 :=
by
  -- Placeholder for the actual proof, which is not required here
  sorry

end alice_ate_more_l846_84666


namespace correct_operation_only_l846_84694

theorem correct_operation_only (a b x y : ℝ) : 
  (2 + Real.sqrt 2 ≠ 2 * Real.sqrt 2) ∧ 
  (4 * x^2 * y - x^2 * y ≠ 3) ∧ 
  ((a + b)^2 ≠ a^2 + b^2) ∧
  ((ab)^3 = a^3 * b^3) := 
by 
  sorry

end correct_operation_only_l846_84694


namespace probability_two_seeds_missing_seedlings_probability_two_seeds_no_strong_seedlings_probability_three_seeds_having_seedlings_probability_three_seeds_having_strong_seedlings_l846_84612

noncomputable def germination_rate : ℝ := 0.9
noncomputable def non_germination_rate : ℝ := 1 - germination_rate
noncomputable def strong_seedling_rate : ℝ := 0.6
noncomputable def non_strong_seedling_rate : ℝ := 1 - strong_seedling_rate

theorem probability_two_seeds_missing_seedlings :
  (non_germination_rate ^ 2) = 0.01 := sorry

theorem probability_two_seeds_no_strong_seedlings :
  (non_strong_seedling_rate ^ 2) = 0.16 := sorry

theorem probability_three_seeds_having_seedlings :
  (1 - non_germination_rate ^ 3) = 0.999 := sorry

theorem probability_three_seeds_having_strong_seedlings :
  (1 - non_strong_seedling_rate ^ 3) = 0.936 := sorry

end probability_two_seeds_missing_seedlings_probability_two_seeds_no_strong_seedlings_probability_three_seeds_having_seedlings_probability_three_seeds_having_strong_seedlings_l846_84612


namespace hulk_jump_distance_l846_84663

theorem hulk_jump_distance :
  ∃ n : ℕ, 3^n > 1500 ∧ ∀ m < n, 3^m ≤ 1500 := 
sorry

end hulk_jump_distance_l846_84663


namespace range_of_a_l846_84618

theorem range_of_a (a : ℝ) (h : ∀ (x1 x2 : ℝ), (0 < x1 ∧ x1 < x2 ∧ x2 < 1) → (a * x2 - x2^3) - (a * x1 - x1^3) > x2 - x1) : a ≥ 4 :=
sorry


end range_of_a_l846_84618


namespace value_of_abc_l846_84699

theorem value_of_abc : ∃ (a b c : ℕ), a > 0 ∧ b > 0 ∧ c > 0 ∧
  (ab + c + 10 = 51) ∧ (bc + a + 10 = 51) ∧ (ac + b + 10 = 51) ∧ (a + b + c = 41) :=
by
  sorry

end value_of_abc_l846_84699


namespace f_negative_l846_84674

-- Let f be a function defined on the real numbers
variable (f : ℝ → ℝ)

-- Conditions: f is odd and given form for non-negative x
axiom odd_f : ∀ x : ℝ, f (-x) = -f x
axiom f_positive : ∀ x : ℝ, 0 ≤ x → f x = x^2 - 2 * x

theorem f_negative (x : ℝ) (hx : x < 0) : f x = -x^2 + 2 * x := by
  sorry

end f_negative_l846_84674


namespace ariel_fish_l846_84667

theorem ariel_fish (total_fish : ℕ) (male_fraction female_fraction : ℚ) (h1 : total_fish = 45) (h2 : male_fraction = 2 / 3) (h3 : female_fraction = 1 - male_fraction) : total_fish * female_fraction = 15 :=
by
  sorry

end ariel_fish_l846_84667


namespace quadratic_term_elimination_l846_84659

theorem quadratic_term_elimination (m : ℝ) :
  (3 * (x : ℝ) ^ 2 - 10 - 2 * x - 4 * x ^ 2 + m * x ^ 2) = -(x : ℝ) * (2 * x + 10) ↔ m = 1 := 
by sorry

end quadratic_term_elimination_l846_84659


namespace extreme_points_inequality_l846_84608

noncomputable def f (a x : ℝ) : ℝ := a * Real.log (x + 1) + 1 / 2 * x ^ 2 - x

theorem extreme_points_inequality 
  (a : ℝ)
  (ha : 0 < a ∧ a < 1)
  (alpha beta : ℝ)
  (h_eq_alpha : alpha = -Real.sqrt (1 - a))
  (h_eq_beta : beta = Real.sqrt (1 - a))
  (h_order : alpha < beta) :
  (f a beta / alpha) < (1 / 2) :=
sorry

end extreme_points_inequality_l846_84608


namespace quadratic_equation_m_value_l846_84614

-- Definition of the quadratic equation having exactly one solution with the given parameters
def quadratic_equation_has_one_solution (a b c : ℚ) : Prop :=
  b^2 - 4 * a * c = 0

-- Given constants in the problem
def a : ℚ := 3
def b : ℚ := -7

-- The value of m we aim to prove
def m_correct : ℚ := 49 / 12

-- The theorem stating the problem
theorem quadratic_equation_m_value (m : ℚ) (h : quadratic_equation_has_one_solution a b m) : m = m_correct :=
  sorry

end quadratic_equation_m_value_l846_84614


namespace temperature_increase_per_century_l846_84698

def total_temperature_change_over_1600_years : ℕ := 64
def years_in_a_century : ℕ := 100
def years_overall : ℕ := 1600

theorem temperature_increase_per_century :
  total_temperature_change_over_1600_years / (years_overall / years_in_a_century) = 4 := by
  sorry

end temperature_increase_per_century_l846_84698


namespace fraction_of_70cm_ropes_l846_84629

theorem fraction_of_70cm_ropes (R : ℕ) (avg_all : ℚ) (avg_70 : ℚ) (avg_85 : ℚ) (total_len : R * avg_all = 480) 
  (total_ropes : R = 6) : 
  ∃ f : ℚ, f = 1 / 3 ∧ f * R * avg_70 + (R - f * R) * avg_85 = R * avg_all :=
by
  sorry

end fraction_of_70cm_ropes_l846_84629


namespace license_plates_count_l846_84662

theorem license_plates_count :
  let num_vowels := 5
  let num_letters := 26
  let num_odd_digits := 5
  let num_even_digits := 5
  num_vowels * num_letters * num_letters * num_odd_digits * num_even_digits = 84500 :=
by
  sorry

end license_plates_count_l846_84662


namespace range_of_3x_minus_y_l846_84636

-- Defining the conditions in Lean
variable (x y : ℝ)

-- Condition 1: -1 ≤ x + y ≤ 1
def cond1 : Prop := -1 ≤ x + y ∧ x + y ≤ 1

-- Condition 2: 1 ≤ x - y ≤ 3
def cond2 : Prop := 1 ≤ x - y ∧ x - y ≤ 3

-- The theorem statement to prove that the range of 3x - y is [1, 7]
theorem range_of_3x_minus_y (h1 : cond1 x y) (h2 : cond2 x y) : 1 ≤ 3 * x - y ∧ 3 * x - y ≤ 7 := by
  sorry

end range_of_3x_minus_y_l846_84636


namespace divisibility_properties_l846_84619

theorem divisibility_properties (a b : ℤ) (k : ℕ) :
  (¬(a + b ∣ a^(2*k) + b^(2*k)) ∧ ¬(a - b ∣ a^(2*k) + b^(2*k))) ∧ 
  ((a + b ∣ a^(2*k) - b^(2*k)) ∧ (a - b ∣ a^(2*k) - b^(2*k))) ∧ 
  (a + b ∣ a^(2*k + 1) + b^(2*k + 1)) ∧ 
  (a - b ∣ a^(2*k + 1) - b^(2*k + 1)) := 
by sorry

end divisibility_properties_l846_84619


namespace inequality_solution_l846_84631

theorem inequality_solution (x : ℝ) :
  (2 / (x + 2) + 9 / (x + 6) ≥ 2) ↔ (x ∈ Set.Ico (-6 : ℝ) (-3) ∪ Set.Ioc (-2) 3) := 
sorry

end inequality_solution_l846_84631


namespace Kuwabara_class_girls_percentage_l846_84673

variable (num_girls num_boys : ℕ)

def total_students (num_girls num_boys : ℕ) : ℕ :=
  num_girls + num_boys

def girls_percentage (num_girls num_boys : ℕ) : ℚ :=
  (num_girls : ℚ) / (total_students num_girls num_boys : ℚ) * 100

theorem Kuwabara_class_girls_percentage (num_girls num_boys : ℕ) (h1: num_girls = 10) (h2: num_boys = 15) :
  girls_percentage num_girls num_boys = 40 := 
by
  sorry

end Kuwabara_class_girls_percentage_l846_84673


namespace max_value_A_l846_84644

noncomputable def A (x y : ℝ) : ℝ :=
  ((x^2 - y) * Real.sqrt (y + x^3 - x * y) + (y^2 - x) * Real.sqrt (x + y^3 - x * y) + 1) /
  ((x - y)^2 + 1)

theorem max_value_A (x y : ℝ) (hx : 0 < x ∧ x ≤ 1) (hy : 0 < y ∧ y ≤ 1) :
  A x y ≤ 1 :=
sorry

end max_value_A_l846_84644


namespace min_value_is_neg_one_l846_84665

noncomputable def find_min_value (a b c : ℝ) (h : 4 * a^2 - 2 * a * b + b^2 - c = 0) (h1 : 0 < c) (h2 : |2 * a + b| = sorry) : ℝ :=
  1 / a + 2 / b + 4 / c

theorem min_value_is_neg_one (a b c : ℝ) (h : 4 * a^2 - 2 * a * b + b^2 - c = 0) (h1 : 0 < c) (h2 : |2 * a + b| = sorry) : 
  find_min_value a b c h h1 h2 = -1 :=
sorry

end min_value_is_neg_one_l846_84665


namespace height_comparison_of_cylinder_and_rectangular_solid_l846_84616

theorem height_comparison_of_cylinder_and_rectangular_solid
  (V : ℝ) (A : ℝ) (h_cylinder : ℝ) (h_rectangular_solid : ℝ)
  (equal_volume : V = V)
  (equal_base_areas : A = A)
  (height_cylinder_eq : h_cylinder = V / A)
  (height_rectangular_solid_eq : h_rectangular_solid = V / A)
  : ¬ (h_cylinder > h_rectangular_solid) :=
by {
  sorry
}

end height_comparison_of_cylinder_and_rectangular_solid_l846_84616


namespace value_of_x_when_z_is_32_l846_84601

variables {x y z k : ℝ}
variable (m n : ℝ)

def directly_proportional (x y : ℝ) (m : ℝ) := x = m * y^2
def inversely_proportional (y z : ℝ) (n : ℝ) := y = n / z^2

-- Our main proof goal
theorem value_of_x_when_z_is_32 (h1 : directly_proportional x y m) 
  (h2 : inversely_proportional y z n) (h3 : z = 8) (hx : x = 5) : 
  x = 5 / 256 :=
by
  let k := x * z^4
  have k_value : k = 20480 := by sorry
  have x_new : x = k / z^4 := by sorry
  have z_new : z = 32 := by sorry
  have x_final : x = 5 / 256 := by sorry
  exact x_final

end value_of_x_when_z_is_32_l846_84601


namespace chicken_price_reaches_81_in_2_years_l846_84613

theorem chicken_price_reaches_81_in_2_years :
  ∃ t : ℝ, (t / 12 = 2) ∧ (∃ n : ℕ, (3:ℝ)^(n / 6) = 81 ∧ n = t) :=
by
  sorry

end chicken_price_reaches_81_in_2_years_l846_84613


namespace max_children_arrangement_l846_84634

theorem max_children_arrangement (n : ℕ) (h1 : n = 49) 
  (h2 : ∀ i j, i ≠ j → 1 ≤ i ∧ i ≤ 49 → 1 ≤ j ∧ j ≤ 49 → (i * j < 100)) : 
  ∃ k, k = 18 :=
by
  sorry

end max_children_arrangement_l846_84634


namespace some_value_correct_l846_84672

theorem some_value_correct (w x y : ℝ) (some_value : ℝ)
  (h1 : 3 / w + some_value = 3 / y)
  (h2 : w * x = y)
  (h3 : (w + x) / 2 = 0.5) :
  some_value = 6 := by
  sorry

end some_value_correct_l846_84672


namespace remainder_when_divided_l846_84690
-- First, import the necessary library.

-- Define the problem conditions and the goal.
theorem remainder_when_divided (P Q Q' R R' S T D D' D'' : ℕ) 
  (h1 : P = Q * D + R)
  (h2 : Q = D' * Q' + D'' * R' + R')
  (h3 : S = D'' * T)
  (h4 : R' = S + T) :
  P % (D * D' * D'') = D * R' + R := by
  sorry

end remainder_when_divided_l846_84690


namespace river_depth_mid_may_l846_84645

-- Definitions corresponding to the conditions
def depth_mid_june (D : ℕ) : ℕ := D + 10
def depth_mid_july (D : ℕ) : ℕ := 3 * (depth_mid_june D)

-- The theorem statement
theorem river_depth_mid_may (D : ℕ) (h : depth_mid_july D = 45) : D = 5 :=
by
  sorry

end river_depth_mid_may_l846_84645


namespace a7_equals_21_l846_84670

-- Define the sequence {a_n} recursively
def seq (n : ℕ) : ℕ :=
  match n with
  | 0 => 1
  | 1 => 2
  | (n + 2) => seq n + seq (n + 1)

-- Statement to prove that a_7 = 21
theorem a7_equals_21 : seq 6 = 21 := 
  sorry

end a7_equals_21_l846_84670


namespace sum_of_a6_and_a7_l846_84638

theorem sum_of_a6_and_a7 (a : ℕ → ℝ) (h_arith : ∃ d, ∀ n, a (n + 1) = a n + d)
  (h_sum : a 2 + a 3 + a 10 + a 11 = 48) : a 6 + a 7 = 24 :=
by
  sorry

end sum_of_a6_and_a7_l846_84638


namespace represents_not_much_different_l846_84669

def not_much_different_from (x : ℝ) (c : ℝ) : Prop := x - c ≤ 0

theorem represents_not_much_different {x : ℝ} :
  (not_much_different_from x 2023) = (x - 2023 ≤ 0) :=
by
  sorry

end represents_not_much_different_l846_84669


namespace prove_avg_mark_of_batch3_l846_84661

noncomputable def avg_mark_of_batch3 (A1 A2 A3 : ℕ) (Marks1 Marks2 Marks3 : ℚ) : Prop :=
  A1 = 40 ∧ A2 = 50 ∧ A3 = 60 ∧ Marks1 = 45 ∧ Marks2 = 55 ∧ 
  (A1 * Marks1 + A2 * Marks2 + A3 * Marks3) / (A1 + A2 + A3) = 56.333333333333336 → 
  Marks3 = 65

theorem prove_avg_mark_of_batch3 : avg_mark_of_batch3 40 50 60 45 55 65 :=
by
  unfold avg_mark_of_batch3
  sorry

end prove_avg_mark_of_batch3_l846_84661


namespace zoey_finishes_20th_book_on_wednesday_l846_84632

theorem zoey_finishes_20th_book_on_wednesday :
  let days_spent := (20 * 21) / 2
  (days_spent % 7) = 0 → 
  (start_day : ℕ) → start_day = 3 → ((start_day + days_spent) % 7) = 3 :=
by
  sorry

end zoey_finishes_20th_book_on_wednesday_l846_84632


namespace sugar_cups_used_l846_84650

def ratio_sugar_water : ℕ × ℕ := (1, 2)
def total_cups : ℕ := 84

theorem sugar_cups_used (r : ℕ × ℕ) (tc : ℕ) (hsugar : r.1 = 1) (hwater : r.2 = 2) (htotal : tc = 84) :
  (tc * r.1) / (r.1 + r.2) = 28 :=
by
  sorry

end sugar_cups_used_l846_84650


namespace triangle_area_is_correct_l846_84676

noncomputable def area_of_triangle (A B C : (ℝ × ℝ)) : ℝ :=
  0.5 * abs (A.1 * (B.2 - C.2) + B.1 * (C.2 - A.2) + C.1 * (A.2 - B.2))

theorem triangle_area_is_correct :
  area_of_triangle (0, 3) (4, -2) (9, 6) = 16.5 :=
by
  sorry

end triangle_area_is_correct_l846_84676


namespace find_angle_A_l846_84654

theorem find_angle_A (A B C : ℝ) (a b c : ℝ) :
  (a * Real.sin A = b * Real.sin B + (c - b) * Real.sin C)
  → (A = π / 3) :=
sorry

end find_angle_A_l846_84654


namespace constant_term_expansion_l846_84607

theorem constant_term_expansion (x : ℝ) (hx : x ≠ 0) : 
  let term (r : ℕ) : ℝ := (1 / 2) ^ (9 - r) * (-1) ^ r * Nat.choose 9 r * x ^ (3 / 2 * r - 9)
  term 6 = 21 / 2 :=
by
  sorry

end constant_term_expansion_l846_84607


namespace distance_between_home_and_retreat_l846_84624

theorem distance_between_home_and_retreat (D : ℝ) 
  (h1 : D / 50 + D / 75 = 10) : D = 300 :=
sorry

end distance_between_home_and_retreat_l846_84624


namespace right_triangle_area_l846_84620

theorem right_triangle_area (a b c : ℕ) (habc : a = 3 ∧ b = 4 ∧ c = 5) : 
  (a * a + b * b = c * c) → 
  1 / 2 * (a * b) = 6 :=
by
  sorry

end right_triangle_area_l846_84620


namespace average_pages_per_book_l846_84630

-- Conditions
def book_thickness_in_inches : ℕ := 12
def pages_per_inch : ℕ := 80
def number_of_books : ℕ := 6

-- Given these conditions, we need to prove the average number of pages per book is 160.
theorem average_pages_per_book (book_thickness_in_inches : ℕ) (pages_per_inch : ℕ) (number_of_books : ℕ)
  (h1 : book_thickness_in_inches = 12)
  (h2 : pages_per_inch = 80)
  (h3 : number_of_books = 6) :
  (book_thickness_in_inches * pages_per_inch) / number_of_books = 160 := by
  sorry

end average_pages_per_book_l846_84630


namespace triangle_side_length_l846_84681

theorem triangle_side_length (a : ℝ) (h1 : 4 < a) (h2 : a < 8) : a = 6 :=
sorry

end triangle_side_length_l846_84681


namespace sum_x_y_l846_84604

theorem sum_x_y (x y : ℤ) (h1 : x - y = 40) (h2 : x = 32) : x + y = 24 := by
  sorry

end sum_x_y_l846_84604


namespace find_a_over_b_l846_84621

variable (x y z a b : ℝ)
variable (h₁ : 4 * x - 2 * y + z = a)
variable (h₂ : 6 * y - 12 * x - 3 * z = b)
variable (h₃ : b ≠ 0)

theorem find_a_over_b : a / b = -1 / 3 :=
by
  sorry

end find_a_over_b_l846_84621


namespace octagon_area_inscribed_in_square_l846_84627

noncomputable def side_length_of_square (perimeter : ℝ) : ℝ :=
  perimeter / 4

noncomputable def trisected_segment_length (side_length : ℝ) : ℝ :=
  side_length / 3

noncomputable def area_of_removed_triangle (segment_length : ℝ) : ℝ :=
  (segment_length * segment_length) / 2

noncomputable def total_area_removed_by_triangles (area_of_triangle : ℝ) : ℝ :=
  4 * area_of_triangle

noncomputable def area_of_square (side_length : ℝ) : ℝ :=
  side_length * side_length

noncomputable def area_of_octagon (area_of_square : ℝ) (total_area_removed : ℝ) : ℝ :=
  area_of_square - total_area_removed

theorem octagon_area_inscribed_in_square (perimeter : ℝ) (H : perimeter = 144) :
  area_of_octagon (area_of_square (side_length_of_square perimeter))
    (total_area_removed_by_triangles (area_of_removed_triangle (trisected_segment_length (side_length_of_square perimeter))))
  = 1008 :=
by
  rw [H]
  -- Intermediate steps would contain calculations for side_length_of_square, trisected_segment_length, area_of_removed_triangle, total_area_removed_by_triangles, and area_of_square based on the given perimeter.
  sorry

end octagon_area_inscribed_in_square_l846_84627


namespace even_n_condition_l846_84626

theorem even_n_condition (x : ℝ) (n : ℕ) (h : ∀ x, 3 * x^n + n * (x + 2) - 3 ≥ n * x^2) : n % 2 = 0 :=
sorry

end even_n_condition_l846_84626


namespace minimize_sum_of_reciprocals_l846_84652

theorem minimize_sum_of_reciprocals (a b : ℕ) (h : 4 * a + b = 6) : 
  a = 1 ∧ b = 2 ∨ a = 2 ∧ b = 1 :=
by
  sorry

end minimize_sum_of_reciprocals_l846_84652


namespace required_sticks_l846_84683

variables (x y : ℕ)
variables (h1 : 2 * x + 3 * y = 96)
variables (h2 : x + y = 40)

theorem required_sticks (x y : ℕ) (h1 : 2 * x + 3 * y = 96) (h2 : x + y = 40) : 
  x = 24 ∧ y = 16 ∧ (96 - (x * 2 + y * 3) / 2) = 116 :=
by
  sorry

end required_sticks_l846_84683


namespace lcm_18_24_l846_84695

theorem lcm_18_24 : Nat.lcm 18 24 = 72 := by
  -- Sorry is place-holder for the actual proof.
  sorry

end lcm_18_24_l846_84695


namespace sum_of_number_is_8_l846_84693

theorem sum_of_number_is_8 (x v : ℝ) (h1 : 0.75 * x + 2 = v) (h2 : x = 8.0) : v = 8.0 :=
by
  sorry

end sum_of_number_is_8_l846_84693


namespace remaining_paint_fraction_l846_84679

theorem remaining_paint_fraction :
  ∀ (initial_paint : ℝ) (half_usage : ℕ → ℝ → ℝ),
    initial_paint = 2 →
    half_usage 0 (2 : ℝ) = 1 →
    half_usage 1 (1 : ℝ) = 0.5 →
    half_usage 2 (0.5 : ℝ) = 0.25 →
    half_usage 3 (0.25 : ℝ) = (0.25 / initial_paint) := by
  sorry

end remaining_paint_fraction_l846_84679


namespace actual_tax_equals_600_l846_84688

-- Definition for the first condition: initial tax amount
variable (a : ℝ)

-- Define the first reduction: 25% reduction
def first_reduction (a : ℝ) : ℝ := 0.75 * a

-- Define the second reduction: further 20% reduction
def second_reduction (tax_after_first_reduction : ℝ) : ℝ := 0.80 * tax_after_first_reduction

-- Define the final reduction: combination of both reductions
def final_tax (a : ℝ) : ℝ := second_reduction (first_reduction a)

-- Proof that with a = 1000, the actual tax is 600 million euros
theorem actual_tax_equals_600 (a : ℝ) (h₁ : a = 1000) : final_tax a = 600 := by
    rw [h₁]
    simp [final_tax, first_reduction, second_reduction]
    sorry

end actual_tax_equals_600_l846_84688


namespace triangle_side_relationship_l846_84617

noncomputable def sin (x : ℝ) : ℝ := sorry
noncomputable def cos (x : ℝ) : ℝ := sorry

theorem triangle_side_relationship 
  (a b c : ℝ)
  (α β γ : ℝ)
  (hα : α = 40 * Real.pi / 180)
  (hβ : β = 60 * Real.pi / 180)
  (hγ : γ = 80 * Real.pi / 180)
  (h_angle_sum : α + β + γ = Real.pi) : 
  a * (a + b + c) = b * (b + c) :=
sorry

end triangle_side_relationship_l846_84617


namespace inequality_1_inequality_2_inequality_4_l846_84642

variable {a b : ℝ}

def condition (a b : ℝ) : Prop := (1/a < 1/b) ∧ (1/b < 0)

theorem inequality_1 (ha : a < 0) (hb : b < 0) (hc : condition a b) : a + b < a * b :=
sorry

theorem inequality_2 (hc : condition a b) : |a| < |b| :=
sorry

theorem inequality_4 (hc : condition a b) : (b / a) + (a / b) > 2 :=
sorry

end inequality_1_inequality_2_inequality_4_l846_84642


namespace recurrence_relation_l846_84606

noncomputable def p (n k : ℕ) : ℚ := sorry

theorem recurrence_relation (n k : ℕ) (h : k < n) :
  p n k = p (n - 1) k - (1 / 2^k) * p (n - k) k + (1 / 2^k) :=
by sorry

end recurrence_relation_l846_84606


namespace selling_price_of_book_l846_84664

theorem selling_price_of_book (cost_price : ℝ) (profit_percentage : ℝ) (profit : ℝ) (selling_price : ℝ) 
  (h₁ : cost_price = 60) 
  (h₂ : profit_percentage = 25) 
  (h₃ : profit = (profit_percentage / 100) * cost_price) 
  (h₄ : selling_price = cost_price + profit) : 
  selling_price = 75 := 
by
  sorry

end selling_price_of_book_l846_84664


namespace relationship_between_M_and_N_l846_84697

theorem relationship_between_M_and_N (a b : ℝ) (M N : ℝ) 
  (hM : M = a^2 - a * b) 
  (hN : N = a * b - b^2) : M ≥ N :=
by sorry

end relationship_between_M_and_N_l846_84697

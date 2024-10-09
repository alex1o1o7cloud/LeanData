import Mathlib

namespace geometric_sequence_sum_l1655_165552

theorem geometric_sequence_sum (a : ℕ → ℝ) (S : ℕ → ℝ) (a_pos : ∀ n, 0 < a n)
  (h_a2 : a 2 = 1) (h_a3a7_a5 : a 3 * a 7 - a 5 = 56)
  (S_eq : ∀ n, S n = (a 1 * (1 - (2 : ℝ) ^ n)) / (1 - 2)) :
  S 5 = 31 / 2 := by
  sorry

end geometric_sequence_sum_l1655_165552


namespace sum_possible_values_of_p_l1655_165507

theorem sum_possible_values_of_p (p q : ℤ) (h1 : p + q = 2010)
  (h2 : ∃ (α β : ℕ), (10 * α * β = q) ∧ (10 * (α + β) = -p)) :
  p = -3100 :=
by
  sorry

end sum_possible_values_of_p_l1655_165507


namespace percent_alcohol_in_new_solution_l1655_165563

theorem percent_alcohol_in_new_solution (orig_vol : ℝ) (orig_percent : ℝ) (add_alc : ℝ) (add_water : ℝ) :
  orig_percent = 5 → orig_vol = 40 → add_alc = 5.5 → add_water = 4.5 →
  (((orig_vol * (orig_percent / 100) + add_alc) / (orig_vol + add_alc + add_water)) * 100) = 15 :=
by
  intros h1 h2 h3 h4
  rw [h1, h2, h3, h4]
  sorry

end percent_alcohol_in_new_solution_l1655_165563


namespace rowing_velocity_l1655_165592

theorem rowing_velocity (v : ℝ) : 
  (∀ (d : ℝ) (s : ℝ) (total_time : ℝ), 
    s = 10 ∧ 
    total_time = 30 ∧ 
    d = 144 ∧ 
    (d / (s - v) + d / (s + v)) = total_time) → 
  v = 2 := 
by
  sorry

end rowing_velocity_l1655_165592


namespace side_length_of_base_l1655_165516

-- Given conditions
def lateral_face_area := 90 -- Area of one lateral face in square meters
def slant_height := 20 -- Slant height in meters

-- The theorem statement
theorem side_length_of_base 
  (s : ℝ)
  (h : ℝ := slant_height)
  (a : ℝ := lateral_face_area)
  (h_area : 2 * a = s * h) :
  s = 9 := 
sorry

end side_length_of_base_l1655_165516


namespace time_to_see_slow_train_l1655_165525

noncomputable def time_to_pass (length_fast_train length_slow_train relative_time_fast seconds_observed_by_slow : ℕ) : ℕ := 
  length_slow_train * seconds_observed_by_slow / length_fast_train

theorem time_to_see_slow_train :
  let length_fast_train := 150
  let length_slow_train := 200
  let seconds_observed_by_slow := 6
  let expected_time := 8
  time_to_pass length_fast_train length_slow_train length_fast_train seconds_observed_by_slow = expected_time :=
by sorry

end time_to_see_slow_train_l1655_165525


namespace rainwater_cows_l1655_165532

theorem rainwater_cows (chickens goats cows : ℕ) 
  (h1 : chickens = 18) 
  (h2 : goats = 2 * chickens) 
  (h3 : goats = 4 * cows) : 
  cows = 9 := 
sorry

end rainwater_cows_l1655_165532


namespace solution_of_abs_square_eq_zero_l1655_165539

-- Define the given conditions as hypotheses
variables {x y : ℝ}
theorem solution_of_abs_square_eq_zero (h : |x + 2| + (y - 1)^2 = 0) : x = -2 ∧ y = 1 :=
sorry

end solution_of_abs_square_eq_zero_l1655_165539


namespace solve_for_x_l1655_165538

theorem solve_for_x (x : ℝ) (h : x^4 = (-3)^4) : x = 3 ∨ x = -3 :=
sorry

end solve_for_x_l1655_165538


namespace neg_three_lt_neg_sqrt_eight_l1655_165587

theorem neg_three_lt_neg_sqrt_eight : -3 < -Real.sqrt 8 := 
sorry

end neg_three_lt_neg_sqrt_eight_l1655_165587


namespace negation_existence_l1655_165554

-- The problem requires showing the equivalence between the negation of an existential
-- proposition and a universal proposition in the context of real numbers.

theorem negation_existence (m : ℝ) :
  (¬ ∃ x : ℝ, x^2 - m * x - m < 0) → (∀ x : ℝ, x^2 - m * x - m ≥ 0) :=
by
  sorry

end negation_existence_l1655_165554


namespace yellow_block_heavier_than_green_l1655_165550

theorem yellow_block_heavier_than_green :
  let yellow_block_weight := 0.6
  let green_block_weight := 0.4
  yellow_block_weight - green_block_weight = 0.2 := by
  let yellow_block_weight := 0.6
  let green_block_weight := 0.4
  show yellow_block_weight - green_block_weight = 0.2
  sorry

end yellow_block_heavier_than_green_l1655_165550


namespace waiter_net_earning_l1655_165573

theorem waiter_net_earning (c1 c2 c3 m : ℤ) (h1 : c1 = 3) (h2 : c2 = 2) (h3 : c3 = 1) (t1 t2 t3 : ℤ) (h4 : t1 = 8) (h5 : t2 = 10) (h6 : t3 = 12) (hmeal : m = 5):
  c1 * t1 + c2 * t2 + c3 * t3 - m = 51 := 
by 
  sorry

end waiter_net_earning_l1655_165573


namespace factor_expression_l1655_165568

-- Define the variables
variables (x : ℝ)

-- State the theorem to prove
theorem factor_expression : 3 * x * (x + 1) + 7 * (x + 1) = (3 * x + 7) * (x + 1) :=
by
  sorry

end factor_expression_l1655_165568


namespace greatest_common_divisor_of_B_l1655_165566

def B : Set ℤ := {n | ∃ x : ℤ, n = 4*x + 2}

theorem greatest_common_divisor_of_B : Nat.gcd (4*x + 2) = 2 :=
by
  sorry

end greatest_common_divisor_of_B_l1655_165566


namespace initial_volume_of_mixture_l1655_165531

variable (V : ℝ)
variable (H1 : 0.2 * V + 12 = 0.25 * (V + 12))

theorem initial_volume_of_mixture (H : 0.2 * V + 12 = 0.25 * (V + 12)) : V = 180 := by
  sorry

end initial_volume_of_mixture_l1655_165531


namespace collinear_points_cube_l1655_165524

-- Define a function that counts the sets of three collinear points in the described structure.
def count_collinear_points : Nat :=
  -- Placeholders for the points (vertices, edge midpoints, face centers, center of the cube) and the count logic
  -- The calculation logic will be implemented as the proof
  49

theorem collinear_points_cube : count_collinear_points = 49 :=
  sorry

end collinear_points_cube_l1655_165524


namespace prime_divides_2_pow_n_minus_n_infinte_times_l1655_165583

theorem prime_divides_2_pow_n_minus_n_infinte_times (p : ℕ) (hp : Nat.Prime p) : ∃ᶠ n in at_top, p ∣ 2^n - n :=
sorry

end prime_divides_2_pow_n_minus_n_infinte_times_l1655_165583


namespace average_page_count_per_essay_l1655_165542

-- Conditions
def numberOfStudents := 15
def pagesFirstFive := 5 * 2
def pagesNextFive := 5 * 3
def pagesLastFive := 5 * 1

-- Total pages
def totalPages := pagesFirstFive + pagesNextFive + pagesLastFive

-- Proof problem statement
theorem average_page_count_per_essay : totalPages / numberOfStudents = 2 := by
  sorry

end average_page_count_per_essay_l1655_165542


namespace multiple_of_k_l1655_165597

theorem multiple_of_k (k : ℕ) (m : ℕ) (h₁ : 7 ^ k = 2) (h₂ : 7 ^ (m * k + 2) = 784) : m = 2 :=
sorry

end multiple_of_k_l1655_165597


namespace distribution_y_value_l1655_165518

theorem distribution_y_value :
  ∀ (x y : ℝ),
  (x + 0.1 + 0.3 + y = 1) →
  (7 * x + 8 * 0.1 + 9 * 0.3 + 10 * y = 8.9) →
  y = 0.4 :=
by
  intros x y h1 h2
  sorry

end distribution_y_value_l1655_165518


namespace proof_a_minus_b_l1655_165572

def S (a : ℕ) : Set ℕ := {1, 2, a}
def T (b : ℕ) : Set ℕ := {2, 3, 4, b}

theorem proof_a_minus_b (a b : ℕ)
  (hS : S a = {1, 2, a})
  (hT : T b = {2, 3, 4, b})
  (h_intersection : S a ∩ T b = {1, 2, 3}) :
  a - b = 2 := by
  sorry

end proof_a_minus_b_l1655_165572


namespace initial_percentage_of_gold_l1655_165570

theorem initial_percentage_of_gold (x : ℝ) (h₁ : 48 * x / 100 + 12 = 40 * 60 / 100) : x = 25 :=
by
  sorry

end initial_percentage_of_gold_l1655_165570


namespace hancho_tape_length_l1655_165503

noncomputable def tape_length (x : ℝ) : Prop :=
  (1 / 4) * (4 / 5) * x = 1.5

theorem hancho_tape_length : ∃ x : ℝ, tape_length x ∧ x = 7.5 :=
by sorry

end hancho_tape_length_l1655_165503


namespace projection_matrix_ordered_pair_l1655_165536

theorem projection_matrix_ordered_pair (a c : ℚ)
  (P : Matrix (Fin 2) (Fin 2) ℚ) 
  (P := ![![a, 15 / 34], ![c, 25 / 34]]) :
  P * P = P ->
  (a, c) = (9 / 34, 15 / 34) :=
by
  sorry

end projection_matrix_ordered_pair_l1655_165536


namespace greatest_int_less_neg_22_3_l1655_165528

theorem greatest_int_less_neg_22_3 : ∃ n : ℤ, n = -8 ∧ n < -22 / 3 ∧ ∀ m : ℤ, m < -22 / 3 → m ≤ n :=
by
  sorry

end greatest_int_less_neg_22_3_l1655_165528


namespace area_of_figure_eq_two_l1655_165514

theorem area_of_figure_eq_two :
  ∫ x in (1 / Real.exp 1)..(Real.exp 1), 1 / x = 2 :=
by sorry

end area_of_figure_eq_two_l1655_165514


namespace endomorphisms_of_Z2_are_linear_functions_l1655_165553

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

end endomorphisms_of_Z2_are_linear_functions_l1655_165553


namespace negation_p_l1655_165548

theorem negation_p (p : Prop) : 
  (∃ x : ℝ, x^2 ≥ x) ↔ ¬ (∀ x : ℝ, x^2 < x) :=
by 
  -- The proof is omitted
  sorry

end negation_p_l1655_165548


namespace proportion_of_boys_geq_35_percent_l1655_165509

variables (a b c d n : ℕ)

axiom room_constraint : 2 * (b + d) ≥ n
axiom girl_constraint : 3 * a ≥ 8 * b

theorem proportion_of_boys_geq_35_percent : (3 * c + 4 * d : ℚ) / (3 * a + 4 * b + 3 * c + 4 * d : ℚ) ≥ 0.35 :=
by 
  sorry

end proportion_of_boys_geq_35_percent_l1655_165509


namespace three_exp_eq_l1655_165586

theorem three_exp_eq (y : ℕ) (h : 3^y + 3^y + 3^y = 2187) : y = 6 :=
by
  sorry

end three_exp_eq_l1655_165586


namespace min_value_x2_y2_z2_l1655_165506

theorem min_value_x2_y2_z2 (x y z : ℝ) (h : 2 * x + 3 * y + 4 * z = 11) : 
  x^2 + y^2 + z^2 ≥ 121 / 29 :=
sorry

end min_value_x2_y2_z2_l1655_165506


namespace solve_equation_l1655_165543

theorem solve_equation (x : ℝ) (h : x ≠ 1) : 
  1 / (x - 1) + 1 = 3 / (2 * x - 2) ↔ x = 3 / 2 := by
  sorry

end solve_equation_l1655_165543


namespace temperature_difference_l1655_165513

theorem temperature_difference (initial_temp rise fall : ℤ) (h1 : initial_temp = 25)
    (h2 : rise = 3) (h3 : fall = 15) : initial_temp + rise - fall = 13 := by
  rw [h1, h2, h3]
  norm_num

end temperature_difference_l1655_165513


namespace votes_cast_l1655_165579

theorem votes_cast (A F T : ℕ) (h1 : A = 40 * T / 100) (h2 : F = A + 58) (h3 : T = F + A) : 
  T = 290 := 
by
  sorry

end votes_cast_l1655_165579


namespace ac_bd_leq_8_l1655_165511

theorem ac_bd_leq_8 (a b c d : ℝ) (h1 : a^2 + b^2 = 4) (h2 : c^2 + d^2 = 16) : ac + bd ≤ 8 :=
sorry

end ac_bd_leq_8_l1655_165511


namespace incorrect_judgment_l1655_165595

-- Define propositions p and q
def p : Prop := 2 + 2 = 5
def q : Prop := 3 > 2

-- The incorrect judgment in Lean statement
theorem incorrect_judgment : ¬((p ∧ q) ∧ ¬p) :=
by
  sorry

end incorrect_judgment_l1655_165595


namespace square_area_ratio_l1655_165557

theorem square_area_ratio (n : ℕ) (s₁ s₂: ℕ) (h1 : s₁ = 1) (h2 : s₂ = n^2) (h3 : 2 * s₂ - 1 = 17) :
  s₂ = 81 := 
sorry

end square_area_ratio_l1655_165557


namespace first_term_is_5_over_2_l1655_165575

-- Define the arithmetic sequence and the sum of the first n terms.
def arith_seq (a d : ℕ) (n : ℕ) := a + (n - 1) * d
def S (a d : ℕ) (n : ℕ) := (n * (2 * a + (n - 1) * d)) / 2

-- Define the constant ratio condition.
def const_ratio (a d : ℕ) (n : ℕ) (c : ℕ) :=
  (S a d (3 * n) * 2) = c * (S a d n * 2)

-- Prove the first term is 5/2 given the conditions.
theorem first_term_is_5_over_2 (c : ℕ) (n : ℕ) (h : const_ratio a 5 n 9) : 
  a = 5 / 2 :=
sorry

end first_term_is_5_over_2_l1655_165575


namespace calculate_cakes_left_l1655_165512

-- Define the conditions
def b_lunch : ℕ := 5
def s_dinner : ℕ := 6
def b_yesterday : ℕ := 3

-- Define the calculation of the total cakes baked and cakes left
def total_baked : ℕ := b_lunch + b_yesterday
def cakes_left : ℕ := total_baked - s_dinner

-- The theorem we want to prove
theorem calculate_cakes_left : cakes_left = 2 := 
by
  sorry

end calculate_cakes_left_l1655_165512


namespace remaining_milk_and_coffee_l1655_165558

/-- 
Given:
1. A cup initially contains 1 glass of coffee.
2. A quarter glass of milk is added to the cup.
3. The mixture is thoroughly stirred.
4. One glass of the mixture is poured back.

Prove:
The remaining content in the cup is 1/5 glass of milk and 4/5 glass of coffee. 
--/
theorem remaining_milk_and_coffee :
  let coffee_initial := 1  -- initial volume of coffee
  let milk_added := 1 / 4  -- volume of milk added
  let total_volume := coffee_initial + milk_added  -- total volume after mixing = 5/4 glasses
  let milk_fraction := milk_added / total_volume  -- fraction of milk in the mixture = 1/5
  let coffee_fraction := coffee_initial / total_volume  -- fraction of coffee in the mixture = 4/5
  let volume_poured := 1 / 4  -- volume of mixture poured out
  let milk_poured := (milk_fraction * volume_poured : ℝ)  -- volume of milk poured out = 1/20 glass
  let coffee_poured := (coffee_fraction * volume_poured : ℝ)  -- volume of coffee poured out = 1/5 glass
  let remaining_milk := milk_added - milk_poured  -- remaining volume of milk = 1/5 glass
  let remaining_coffee := coffee_initial - coffee_poured  -- remaining volume of coffee = 4/5 glass
  remaining_milk = 1 / 5 ∧ remaining_coffee = 4 / 5 :=
by
  sorry

end remaining_milk_and_coffee_l1655_165558


namespace parallel_lines_find_m_l1655_165571

theorem parallel_lines_find_m (m : ℝ) :
  (((3 + m) / 2 = 4 / (5 + m)) ∧ ((3 + m) / 2 ≠ (5 - 3 * m) / 8)) → m = -7 :=
sorry

end parallel_lines_find_m_l1655_165571


namespace solve_quadratic_equation_l1655_165508

theorem solve_quadratic_equation (x : ℝ) : 
  2 * x^2 - 4 * x = 6 - 3 * x ↔ (x = -3/2 ∨ x = 2) :=
by
  sorry

end solve_quadratic_equation_l1655_165508


namespace bob_cleaning_time_l1655_165598

theorem bob_cleaning_time (alice_time : ℕ) (h1 : alice_time = 25) (bob_ratio : ℚ) (h2 : bob_ratio = 2 / 5) : 
  bob_time = 10 :=
by
  -- Definitions for conditions
  let bob_time := bob_ratio * alice_time
  -- Sorry to represent the skipped proof
  sorry

end bob_cleaning_time_l1655_165598


namespace binom_difference_30_3_2_l1655_165530

-- Define the binomial coefficient function.
def binom (n k : ℕ) : ℕ := Nat.factorial n / (Nat.factorial k * Nat.factorial (n - k))

-- Theorem statement: binom(30, 3) - binom(30, 2) = 3625
theorem binom_difference_30_3_2 : binom 30 3 - binom 30 2 = 3625 := by
  sorry

end binom_difference_30_3_2_l1655_165530


namespace student_score_l1655_165526

theorem student_score (c w : ℕ) (h1 : c + w = 60) (h2 : 4 * c - w = 150) : c = 42 :=
by
-- Proof steps here, we skip by using sorry for now
sorry

end student_score_l1655_165526


namespace largest_base5_three_digits_is_124_l1655_165544

noncomputable def largest_base5_three_digits_to_base10 : ℕ :=
  4 * 5^2 + 4 * 5^1 + 4 * 5^0

theorem largest_base5_three_digits_is_124 :
  largest_base5_three_digits_to_base10 = 124 :=
by
  -- calculating 4 * 5^2 + 4 * 5^1 + 4 * 5^0 = 124
  sorry

end largest_base5_three_digits_is_124_l1655_165544


namespace hyperbola_eccentricity_l1655_165567

noncomputable def calculate_eccentricity (a b c x0 y0 : ℝ) : ℝ :=
  c / a

theorem hyperbola_eccentricity :
  ∀ (a b c x0 y0 : ℝ),
    (c = 2) →
    (a^2 + b^2 = 4) →
    (x0 = 3) →
    (y0^2 = 24) →
    (5 = x0 + 2) →
    calculate_eccentricity a b c x0 y0 = 2 := 
by 
  intros a b c x0 y0 h1 h2 h3 h4 h5
  sorry

end hyperbola_eccentricity_l1655_165567


namespace inequality_proof_l1655_165582

theorem inequality_proof (x y z : ℝ) (h : x * y * z + x + y + z = 4) : 
    (y * z + 6)^2 + (z * x + 6)^2 + (x * y + 6)^2 ≥ 8 * (x * y * z + 5) :=
by
  sorry

end inequality_proof_l1655_165582


namespace large_bucket_capacity_l1655_165589

variables (S L : ℝ)

theorem large_bucket_capacity (h1 : L = 2 * S + 3) (h2 : 2 * S + 5 * L = 63) : L = 11 :=
by sorry

end large_bucket_capacity_l1655_165589


namespace decompose_number_4705_l1655_165588

theorem decompose_number_4705 :
  4.705 = 4 * 1 + 7 * 0.1 + 0 * 0.01 + 5 * 0.001 := by
  sorry

end decompose_number_4705_l1655_165588


namespace problem_statement_l1655_165581

theorem problem_statement (x y z : ℝ) (h : x^2 + y^2 + z^2 - 2*x + 4*y - 6*z + 14 = 0) : (x - y - z) ^ 2002 = 0 :=
sorry

end problem_statement_l1655_165581


namespace find_number_l1655_165562

-- Defining the constants provided and the related condition
def eight_percent_of (x: ℝ) : ℝ := 0.08 * x
def ten_percent_of_40 : ℝ := 0.10 * 40
def is_solution (x: ℝ) : Prop := (eight_percent_of x) + ten_percent_of_40 = 5.92

-- Theorem statement
theorem find_number : ∃ x : ℝ, is_solution x ∧ x = 24 :=
by sorry

end find_number_l1655_165562


namespace june_biking_time_l1655_165529

theorem june_biking_time :
  ∀ (d_jj d_jb : ℕ) (t_jj : ℕ), (d_jj = 2) → (t_jj = 8) → (d_jb = 6) →
  (t_jb : ℕ) → t_jb = (d_jb * t_jj) / d_jj → t_jb = 24 :=
by
  intros d_jj d_jb t_jj h_djj h_tjj h_djb t_jb h_eq
  rw [h_djj, h_tjj, h_djb] at h_eq
  simp at h_eq
  exact h_eq

end june_biking_time_l1655_165529


namespace largest_mersenne_prime_less_than_500_l1655_165578

def is_mersenne_prime (p : ℕ) : Prop :=
  ∃ n : ℕ, Nat.Prime n ∧ p = 2^n - 1 ∧ Nat.Prime p

theorem largest_mersenne_prime_less_than_500 :
  ∃ p : ℕ, is_mersenne_prime p ∧ p < 500 ∧ ∀ q : ℕ, is_mersenne_prime q ∧ q < 500 → q ≤ p → p = 127 :=
by
  sorry

end largest_mersenne_prime_less_than_500_l1655_165578


namespace minyoung_division_l1655_165591

theorem minyoung_division : 
  ∃ x : ℝ, 107.8 / x = 9.8 ∧ x = 11 :=
by
  use 11
  simp
  sorry

end minyoung_division_l1655_165591


namespace candidates_count_l1655_165515

theorem candidates_count (n : ℕ) (h : n * (n - 1) = 42) : n = 7 :=
by sorry

end candidates_count_l1655_165515


namespace simplify_expression_l1655_165517

theorem simplify_expression : 
  let x := 2
  let y := -1 / 2
  (2 * x^2 + (-x^2 - 2 * x * y + 2 * y^2) - 3 * (x^2 - x * y + 2 * y^2)) = -10 := by
  sorry

end simplify_expression_l1655_165517


namespace find_width_of_room_l1655_165534

theorem find_width_of_room (length room_cost cost_per_sqm total_cost width W : ℕ) 
  (h1 : length = 13)
  (h2 : cost_per_sqm = 12)
  (h3 : total_cost = 1872)
  (h4 : room_cost = length * W * cost_per_sqm)
  (h5 : total_cost = room_cost) : 
  W = 12 := 
by sorry

end find_width_of_room_l1655_165534


namespace total_rainfall_2004_l1655_165559

def average_rainfall_2003 := 50 -- in mm
def extra_rainfall_2004 := 3 -- in mm
def average_rainfall_2004 := average_rainfall_2003 + extra_rainfall_2004 -- in mm
def days_february_2004 := 29
def days_other_months := 30
def months := 12
def months_without_february := months - 1

theorem total_rainfall_2004 : 
  (average_rainfall_2004 * days_february_2004) + (months_without_february * average_rainfall_2004 * days_other_months) = 19027 := 
by sorry

end total_rainfall_2004_l1655_165559


namespace article_cost_price_l1655_165599

theorem article_cost_price (SP : ℝ) (CP : ℝ) (h1 : SP = 455) (h2 : SP = CP + 0.3 * CP) : CP = 350 :=
by sorry

end article_cost_price_l1655_165599


namespace range_of_a_l1655_165501

noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
if x ≤ 2 then -x + 5 else a^x + 2 * a + 2

theorem range_of_a (a : ℝ) (h₁ : a > 0) (h₂ : a ≠ 1) :
  (∀ y ∈ Set.range (f a), y ≥ 3) ↔ (a ∈ Set.Ici (1/2) ∪ Set.Ioi 1) :=
sorry

end range_of_a_l1655_165501


namespace min_value_of_sequence_l1655_165549

theorem min_value_of_sequence 
  (a : ℤ) 
  (a_sequence : ℕ → ℤ) 
  (h₀ : a_sequence 0 = a)
  (h_rec : ∀ n, a_sequence (n + 1) = 2 * a_sequence n - n ^ 2)
  (h_pos : ∀ n, a_sequence n > 0) :
  ∃ k, a_sequence k = 3 := 
sorry

end min_value_of_sequence_l1655_165549


namespace proof_problem_l1655_165540

noncomputable def real_numbers (a x y : ℝ) (h₁ : 0 < a ∧ a < 1) (h₂ : a^x < a^y) : Prop :=
  x^3 > y^3

-- The theorem statement
theorem proof_problem (a x y : ℝ) (h₁ : 0 < a) (h₂ : a < 1) (h₃ : a^x < a^y) : x^3 > y^3 :=
by
  sorry

end proof_problem_l1655_165540


namespace shonda_kids_calculation_l1655_165580

def number_of_kids (B E P F A : Nat) : Nat :=
  let T := B * E
  let total_people := T / P
  total_people - (F + A + 1)

theorem shonda_kids_calculation :
  (number_of_kids 15 12 9 10 7) = 2 :=
by
  unfold number_of_kids
  exact rfl

end shonda_kids_calculation_l1655_165580


namespace truck_distance_in_3_hours_l1655_165519

theorem truck_distance_in_3_hours : 
  ∀ (speed_2miles_2_5minutes : ℝ) 
    (time_minutes : ℝ),
    (speed_2miles_2_5minutes = 2 / 2.5) →
    (time_minutes = 180) →
    (speed_2miles_2_5minutes * time_minutes = 144) :=
by
  intros
  sorry

end truck_distance_in_3_hours_l1655_165519


namespace anya_initial_seat_l1655_165574

theorem anya_initial_seat (V G D E A : ℕ) (A' : ℕ) 
  (h1 : V + G + D + E + A = 15)
  (h2 : V + 1 ≠ A')
  (h3 : G - 3 ≠ A')
  (h4 : (D = A' → E ≠ A') ∧ (E = A' → D ≠ A'))
  (h5 : A = 3 + 2)
  : A = 3 := by
  sorry

end anya_initial_seat_l1655_165574


namespace domain_of_sqrt_2cosx_plus_1_l1655_165505

noncomputable def domain_sqrt_2cosx_plus_1 (x : ℝ) : Prop :=
  ∃ (k : ℤ), (2 * k * Real.pi - 2 * Real.pi / 3) ≤ x ∧ x ≤ (2 * k * Real.pi + 2 * Real.pi / 3)

theorem domain_of_sqrt_2cosx_plus_1 :
  (∀ (x: ℝ), 0 ≤ 2 * Real.cos x + 1 ↔ domain_sqrt_2cosx_plus_1 x) :=
by
  sorry

end domain_of_sqrt_2cosx_plus_1_l1655_165505


namespace relationship_among_terms_l1655_165577

theorem relationship_among_terms (a : ℝ) (h : a ^ 2 + a < 0) : 
  -a > a ^ 2 ∧ a ^ 2 > -a ^ 2 ∧ -a ^ 2 > a :=
sorry

end relationship_among_terms_l1655_165577


namespace circumradius_inradius_perimeter_inequality_l1655_165561

open Real

variables {R r P : ℝ} -- circumradius, inradius, perimeter
variable (triangle_type : String) -- acute, obtuse, right

def satisfies_inequality (R r P : ℝ) (triangle_type : String) : Prop :=
  if triangle_type = "right" then
    R ≥ (sqrt 2) / 2 * sqrt (P * r)
  else
    R ≥ (sqrt 3) / 3 * sqrt (P * r)

theorem circumradius_inradius_perimeter_inequality :
  ∀ (R r P : ℝ) (triangle_type : String), satisfies_inequality R r P triangle_type :=
by 
  intros R r P triangle_type
  sorry -- proof steps go here

end circumradius_inradius_perimeter_inequality_l1655_165561


namespace inequality_solution_l1655_165560

theorem inequality_solution (x : ℝ) :
  -2 < (x^2 - 16 * x + 15) / (x^2 - 4 * x + 5) ∧
  (x^2 - 16 * x + 15) / (x^2 - 4 * x + 5) < 2 ↔
  4 - Real.sqrt 21 < x ∧ x < 4 + Real.sqrt 21 :=
sorry

end inequality_solution_l1655_165560


namespace fruit_bowl_apples_l1655_165555

theorem fruit_bowl_apples (A : ℕ) (total_oranges initial_oranges remaining_oranges : ℕ) (percentage_apples : ℝ) :
  total_oranges = 20 →
  initial_oranges = total_oranges →
  remaining_oranges = initial_oranges - 14 →
  percentage_apples = 0.70 →
  percentage_apples * (A + remaining_oranges) = A →
  A = 14 :=
by 
  intro h1 h2 h3 h4 h5
  -- Proof goes here
  sorry

end fruit_bowl_apples_l1655_165555


namespace acai_juice_cost_l1655_165500

noncomputable def cost_per_litre_juice (x : ℝ) : Prop :=
  let total_cost_cocktail := 1399.45 * 53.333333333333332
  let cost_mixed_fruit_juice := 32 * 262.85
  let cost_acai_juice := 21.333333333333332 * x
  total_cost_cocktail = cost_mixed_fruit_juice + cost_acai_juice

/-- The cost per litre of the açaí berry juice is $3105.00 given the specified conditions. -/
theorem acai_juice_cost : cost_per_litre_juice 3105.00 :=
  sorry

end acai_juice_cost_l1655_165500


namespace problem_statement_l1655_165504

theorem problem_statement :
  ¬ (3^2 = 6) ∧ 
  ¬ ((-1 / 4) / (-4) = 1) ∧
  ¬ ((-8)^2 = -16) ∧
  (-5 - (-2) = -3) := 
by 
  sorry

end problem_statement_l1655_165504


namespace four_distinct_real_roots_l1655_165547

theorem four_distinct_real_roots (m : ℝ) : 
  (∀ x : ℝ, |(x-1)*(x-3)| = m*x → ∃ a b c d : ℝ, a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d) ↔ 
  0 < m ∧ m < 4 - 2 * Real.sqrt 3 :=
by
  sorry

end four_distinct_real_roots_l1655_165547


namespace triangle_area_correct_l1655_165596

-- Define the vectors a, b, and c as given in the problem
def a : ℝ × ℝ := (3, 4)
def b : ℝ × ℝ := (6, 2)
def c : ℝ × ℝ := (1, -1)

-- Define the function to calculate the area of the triangle with the given vertices
def triangle_area (u v w : ℝ × ℝ) : ℝ :=
  0.5 * abs ((v.1 - u.1) * (w.2 - u.2) - (w.1 - u.1) * (v.2 - u.2))

-- State the proof problem
theorem triangle_area_correct : triangle_area c (a.1 + c.1, a.2 + c.2) (b.1 + c.1, b.2 + c.2) = 8.5 :=
by
  -- Proof can go here
  sorry

end triangle_area_correct_l1655_165596


namespace find_n_l1655_165545

def f (x n : ℝ) : ℝ := 2 * x^2 - 3 * x + n
def g (x n : ℝ) : ℝ := 2 * x^2 - 3 * x + 5 * n

theorem find_n (n : ℝ) (h : 3 * f 3 n = 2 * g 3 n) : n = 9 / 7 := by
  sorry

end find_n_l1655_165545


namespace max_a4_l1655_165533

variable {a_n : ℕ → ℝ}

-- Assume a_n is a positive geometric sequence
def is_geometric_seq (a_n : ℕ → ℝ) : Prop :=
  ∃ r > 0, ∀ n, a_n (n + 1) = a_n n * r

-- Given conditions
def condition1 (a_n : ℕ → ℝ) : Prop := is_geometric_seq a_n
def condition2 (a_n : ℕ → ℝ) : Prop := a_n 3 + a_n 5 = 4

theorem max_a4 (a_n : ℕ → ℝ) (h1 : condition1 a_n) (h2 : condition2 a_n) :
    ∃ max_a4 : ℝ, max_a4 = 2 :=
  sorry

end max_a4_l1655_165533


namespace p_is_sufficient_but_not_necessary_for_q_l1655_165520

variable (x : ℝ)

def p := x > 1
def q := x > 0

theorem p_is_sufficient_but_not_necessary_for_q : (p x → q x) ∧ ¬(q x → p x) := by
  sorry

end p_is_sufficient_but_not_necessary_for_q_l1655_165520


namespace savannah_wrapped_gifts_with_second_roll_l1655_165521

theorem savannah_wrapped_gifts_with_second_roll (total_gifts rolls_used roll_1_gifts roll_3_gifts roll_2_gifts : ℕ) 
  (h1 : total_gifts = 12) 
  (h2 : rolls_used = 3) 
  (h3 : roll_1_gifts = 3) 
  (h4 : roll_3_gifts = 4)
  (h5 : total_gifts - roll_1_gifts - roll_3_gifts = roll_2_gifts) :
  roll_2_gifts = 5 := 
by
  sorry

end savannah_wrapped_gifts_with_second_roll_l1655_165521


namespace factorize_quadratic_l1655_165569

theorem factorize_quadratic (x : ℝ) : 
  16 * x^2 - 40 * x + 25 = (4 * x - 5)^2 :=
sorry

end factorize_quadratic_l1655_165569


namespace yuna_correct_multiplication_l1655_165593

theorem yuna_correct_multiplication (x : ℕ) (h : 4 * x = 60) : 8 * x = 120 :=
by
  sorry

end yuna_correct_multiplication_l1655_165593


namespace min_value_is_neg_500000_l1655_165535

noncomputable def min_expression_value (a b : ℝ) : ℝ :=
  let term1 := a + 1/b
  let term2 := b + 1/a
  (term1 * (term1 - 1000) + term2 * (term2 - 1000))

theorem min_value_is_neg_500000 (a b : ℝ) (ha : 0 < a) (hb : 0 < b) :
  min_expression_value a b ≥ -500000 :=
sorry

end min_value_is_neg_500000_l1655_165535


namespace Eva_is_16_l1655_165590

def Clara_age : ℕ := 12
def Nora_age : ℕ := Clara_age + 3
def Liam_age : ℕ := Nora_age - 4
def Eva_age : ℕ := Liam_age + 5

theorem Eva_is_16 : Eva_age = 16 := by
  sorry

end Eva_is_16_l1655_165590


namespace b_value_l1655_165523

theorem b_value (x y b : ℝ) (h1 : x / (2 * y) = 3 / 2) (h2 : (7 * x + b * y) / (x - 2 * y) = 25) : b = 4 := 
by
  sorry

end b_value_l1655_165523


namespace shelly_total_money_l1655_165564

-- Define the conditions
def num_of_ten_dollar_bills : ℕ := 10
def num_of_five_dollar_bills : ℕ := num_of_ten_dollar_bills - 4

-- Problem statement: How much money does Shelly have in all?
theorem shelly_total_money :
  (num_of_ten_dollar_bills * 10) + (num_of_five_dollar_bills * 5) = 130 :=
by
  sorry

end shelly_total_money_l1655_165564


namespace initial_volume_mixture_l1655_165551

theorem initial_volume_mixture (V : ℝ) (h1 : 0.84 * V = 0.6 * (V + 24)) : V = 60 :=
by
  sorry

end initial_volume_mixture_l1655_165551


namespace baking_powder_difference_l1655_165594

-- Define the known quantities
def baking_powder_yesterday : ℝ := 0.4
def baking_powder_now : ℝ := 0.3

-- Define the statement to prove, i.e., the difference in baking powder
theorem baking_powder_difference : baking_powder_yesterday - baking_powder_now = 0.1 :=
by
  -- Proof omitted
  sorry

end baking_powder_difference_l1655_165594


namespace NaCl_yield_l1655_165585

structure Reaction :=
  (reactant1 : ℕ)
  (reactant2 : ℕ)
  (product : ℕ)

def NaOH := 3
def HCl := 3

theorem NaCl_yield : ∀ (R : Reaction), R.reactant1 = NaOH → R.reactant2 = HCl → R.product = 3 :=
by
  sorry

end NaCl_yield_l1655_165585


namespace parabola_focus_standard_equation_l1655_165556

theorem parabola_focus_standard_equation :
  ∃ (a b : ℝ), (a = 16 ∧ b = 0) ∨ (a = 0 ∧ b = -8) →
  (∃ (F : ℝ × ℝ), F = (4, 0) ∨ F = (0, -2) ∧ F ∈ {p : ℝ × ℝ | (p.1 - 2 * p.2 - 4 = 0)} →
  (∃ (x y : ℝ), (y^2 = a * x) ∨ (x^2 = b * y))) := sorry

end parabola_focus_standard_equation_l1655_165556


namespace parabola_focus_distance_l1655_165546

theorem parabola_focus_distance (p : ℝ) (y₀ : ℝ) (h₀ : p > 0) 
  (h₁ : y₀^2 = 2 * p * 4) 
  (h₂ : dist (4, y₀) (p/2, 0) = 3/2 * p) : 
  p = 4 := 
sorry

end parabola_focus_distance_l1655_165546


namespace total_birds_caught_l1655_165584

theorem total_birds_caught 
  (day_birds : ℕ) 
  (night_birds : ℕ)
  (h1 : day_birds = 8) 
  (h2 : night_birds = 2 * day_birds) 
  : day_birds + night_birds = 24 := 
by 
  sorry

end total_birds_caught_l1655_165584


namespace quadratic_equation_problems_l1655_165537

noncomputable def quadratic_has_real_roots (m : ℝ) : Prop :=
  let a := m
  let b := -(3 * m - 1)
  let c := 2 * m - 2
  let Δ := b ^ 2 - 4 * a * c
  Δ ≥ 0

noncomputable def valid_m_values (m : ℝ) : Prop :=
  let a := m
  let b := -(3 * m - 1)
  let c := 2 * m - 2
  let Δ := b ^ 2 - 4 * a * c
  1 = m ∨ -1 / 3 = m

theorem quadratic_equation_problems (m : ℝ) :
  quadratic_has_real_roots m ∧
  (∀ x1 x2 : ℝ, 
      (x1 ≠ x2) →
      x1 + x2 = -(3 * m - 1) / m →
      x1 * x2 = (2 * m - 2) / m →
      abs (x1 - x2) = 2 →
      valid_m_values m) :=
by 
  sorry

end quadratic_equation_problems_l1655_165537


namespace parallelogram_area_l1655_165522

theorem parallelogram_area
  (a b : ℕ)
  (h1 : a + b = 15)
  (h2 : 2 * a = 3 * b) :
  2 * a = 18 :=
by
  -- Proof is omitted; the statement shows what needs to be proven
  sorry

end parallelogram_area_l1655_165522


namespace solve_for_z_l1655_165541

theorem solve_for_z {x y z : ℝ} (h : (1 / x^2) - (1 / y^2) = 1 / z) :
  z = (x^2 * y^2) / (y^2 - x^2) :=
sorry

end solve_for_z_l1655_165541


namespace common_difference_of_arithmetic_sequence_l1655_165510

theorem common_difference_of_arithmetic_sequence
  (a : ℕ → ℝ) (d : ℝ)
  (h1 : a 2 = 9)
  (h2 : a 5 = 33)
  (h_arith_seq : ∀ n : ℕ, a (n + 1) = a n + d) :
  d = 8 :=
sorry

end common_difference_of_arithmetic_sequence_l1655_165510


namespace cosine_of_negative_135_l1655_165527

theorem cosine_of_negative_135 : Real.cos (-(135 * Real.pi / 180)) = -Real.sqrt 2 / 2 :=
by
  sorry

end cosine_of_negative_135_l1655_165527


namespace no_integer_b_satisfies_conditions_l1655_165565

theorem no_integer_b_satisfies_conditions :
  ¬ ∃ b : ℕ, b^6 ≤ 196 ∧ 196 < b^7 :=
by
  sorry

end no_integer_b_satisfies_conditions_l1655_165565


namespace number_equation_l1655_165502

-- Lean statement equivalent to the mathematical problem
theorem number_equation (x : ℝ) (h : 5 * x - 2 * x = 10) : 5 * x - 2 * x = 10 :=
by exact h

end number_equation_l1655_165502


namespace basketball_free_throws_l1655_165576

-- Define the given conditions as assumptions
variables {a b x : ℝ}
variables (h1 : 3 * b = 2 * a)
variables (h2 : x = 2 * a - 2)
variables (h3 : 2 * a + 3 * b + x = 78)

-- State the theorem to be proven
theorem basketball_free_throws : x = 74 / 3 :=
by {
  -- We will provide the proof later
  sorry
}

end basketball_free_throws_l1655_165576

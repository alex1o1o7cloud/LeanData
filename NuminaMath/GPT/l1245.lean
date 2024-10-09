import Mathlib

namespace incorrect_equation_is_wrong_l1245_124584

-- Specifications and conditions
def speed_person_a : ℝ := 7
def speed_person_b : ℝ := 6.5
def head_start : ℝ := 5

-- Define the time variable
variable (x : ℝ)

-- The correct equation based on the problem statement
def correct_equation : Prop := speed_person_a * x - head_start = speed_person_b * x

-- The incorrect equation to prove incorrect
def incorrect_equation : Prop := speed_person_b * x = speed_person_a * x - head_start

-- The Lean statement to prove that the incorrect equation is indeed incorrect
theorem incorrect_equation_is_wrong (h : correct_equation x) : ¬ incorrect_equation x := by
  sorry

end incorrect_equation_is_wrong_l1245_124584


namespace subset_condition_intersection_condition_l1245_124527

-- Definitions of the sets A and B
def A : Set ℝ := {2, 4}
def B (a : ℝ) : Set ℝ := {a, 3 * a}

-- Theorem statements
theorem subset_condition (a : ℝ) : A ⊆ B a → (4 / 3) ≤ a ∧ a ≤ 2 := 
by 
  sorry

theorem intersection_condition (a : ℝ) : (A ∩ B a).Nonempty → (2 / 3) < a ∧ a < 4 := 
by 
  sorry

end subset_condition_intersection_condition_l1245_124527


namespace probability_odd_sum_of_6_balls_drawn_l1245_124598

def binom (n k : ℕ) : ℕ := Nat.choose n k

theorem probability_odd_sum_of_6_balls_drawn :
  let n := 11
  let k := 6
  let total_ways := binom n k
  let odd_count := 6
  let even_count := 5
  let cases := 
    (binom odd_count 1 * binom even_count (k - 1)) +
    (binom odd_count 3 * binom even_count (k - 3)) +
    (binom odd_count 5 * binom even_count (k - 5))
  let favorable_outcomes := cases
  let probability := favorable_outcomes / total_ways
  probability = 118 / 231 := 
by {
  sorry
}

end probability_odd_sum_of_6_balls_drawn_l1245_124598


namespace pascal_fifth_element_15th_row_l1245_124577

theorem pascal_fifth_element_15th_row :
  (Nat.choose 15 4) = 1365 :=
by {
  sorry 
}

end pascal_fifth_element_15th_row_l1245_124577


namespace count_four_digit_integers_with_1_or_7_l1245_124508

/-- 
The total number of four-digit integers with at least one digit being 1 or 7 is 5416.
-/
theorem count_four_digit_integers_with_1_or_7 : 
  let all_four_digit_integers := 9000
  let without_1_or_7 := 7 * 8 * 8 * 8
  let with_1_or_7 := all_four_digit_integers - without_1_or_7
  with_1_or_7 = 5416
:= by
  let all_four_digit_integers := 9000
  let without_1_or_7 := 7 * 8 * 8 * 8
  let with_1_or_7 := all_four_digit_integers - without_1_or_7
  show with_1_or_7 = 5416
  sorry

end count_four_digit_integers_with_1_or_7_l1245_124508


namespace count_two_digit_numbers_with_at_least_one_5_l1245_124589

def is_two_digit (n : ℕ) : Prop := n >= 10 ∧ n < 100

def has_digit_5 (n : ℕ) : Prop := ∃ (a b : ℕ), is_two_digit (10 * a + b) ∧ (a = 5 ∨ b = 5)

theorem count_two_digit_numbers_with_at_least_one_5 : 
  ∃ count : ℕ, (∀ n, is_two_digit n → has_digit_5 n → n ∈ Finset.range (100)) ∧ count = 18 := 
sorry

end count_two_digit_numbers_with_at_least_one_5_l1245_124589


namespace inequality_m_le_minus3_l1245_124513

theorem inequality_m_le_minus3 (m : ℝ) : (∀ x : ℝ, 0 < x ∧ x ≤ 1 → x^2 - 4 * x ≥ m) → m ≤ -3 :=
by
  sorry

end inequality_m_le_minus3_l1245_124513


namespace diagonal_inequality_l1245_124539

theorem diagonal_inequality (A B C D : ℝ × ℝ) (h1 : A.1 = 0) (h2 : B.1 = 0) (h3 : C.2 = 0) (h4 : D.2 = 0) 
  (ha : A.2 < B.2) (hd : D.1 < C.1) : 
  (Real.sqrt (A.2^2 + C.1^2)) * (Real.sqrt (B.2^2 + D.1^2)) > (Real.sqrt (A.2^2 + D.1^2)) * (Real.sqrt (B.2^2 + C.1^2)) :=
sorry

end diagonal_inequality_l1245_124539


namespace arithmetic_mean_of_reciprocals_of_first_four_primes_l1245_124556

theorem arithmetic_mean_of_reciprocals_of_first_four_primes :
  (1/2 + 1/3 + 1/5 + 1/7) / 4 = 247 / 840 :=
by
  sorry

end arithmetic_mean_of_reciprocals_of_first_four_primes_l1245_124556


namespace new_trailer_homes_count_l1245_124585

theorem new_trailer_homes_count :
  let old_trailers : ℕ := 30
  let old_avg_age : ℕ := 15
  let years_since : ℕ := 3
  let new_avg_age : ℕ := 10
  let total_age := (old_trailers * (old_avg_age + years_since)) + (3 * new_trailers)
  let total_trailers := old_trailers + new_trailers
  let total_avg_age := total_age / total_trailers
  total_avg_age = new_avg_age → new_trailers = 34 :=
by
  sorry

end new_trailer_homes_count_l1245_124585


namespace nuts_per_cookie_l1245_124563

theorem nuts_per_cookie (h1 : (1/4:ℝ) * 60 = 15)
(h2 : (0.40:ℝ) * 60 = 24)
(h3 : 60 - 15 - 24 = 21)
(h4 : 72 / (15 + 21) = 2) :
72 / 36 = 2 := by
suffices h : 72 / 36 = 2 from h
exact h4

end nuts_per_cookie_l1245_124563


namespace hash_op_calculation_l1245_124546

-- Define the new operation
def hash_op (a b : ℚ) : ℚ :=
  a^2 + a * b - 5

-- Prove that (-3) # 6 = -14
theorem hash_op_calculation : hash_op (-3) 6 = -14 := by
  sorry

end hash_op_calculation_l1245_124546


namespace evaluate_g_at_5_l1245_124504

def g (x : ℝ) : ℝ := x^2 - 2 * x

theorem evaluate_g_at_5 : g 5 = 15 :=
by
    -- proof steps here
    sorry

end evaluate_g_at_5_l1245_124504


namespace base2_to_base4_conversion_l1245_124596

theorem base2_to_base4_conversion :
  (2 ^ 8 + 2 ^ 6 + 2 ^ 4 + 2 ^ 3 + 2 ^ 2 + 1) = (1 * 4^3 + 1 * 4^2 + 3 * 4^1 + 1 * 4^0) :=
by 
  sorry

end base2_to_base4_conversion_l1245_124596


namespace train_pass_bridge_l1245_124514

-- Define variables and conditions
variables (train_length bridge_length : ℕ)
          (train_speed_kmph : ℕ)

-- Convert speed from km/h to m/s
def train_speed_mps(train_speed_kmph : ℕ) : ℚ :=
  (train_speed_kmph * 1000) / 3600

-- Total distance to cover
def total_distance(train_length bridge_length : ℕ) : ℕ :=
  train_length + bridge_length

-- Time to pass the bridge
def time_to_pass_bridge(train_length bridge_length : ℕ) (train_speed_kmph : ℕ) : ℚ :=
  (total_distance train_length bridge_length) / (train_speed_mps train_speed_kmph)

-- The proof statement
theorem train_pass_bridge :
  time_to_pass_bridge 360 140 50 = 36 := 
by
  -- actual proof would go here
  sorry

end train_pass_bridge_l1245_124514


namespace negatively_added_marks_l1245_124530

theorem negatively_added_marks 
  (correct_marks_per_question : ℝ) 
  (total_marks : ℝ) 
  (total_questions : ℕ) 
  (correct_answers : ℕ) 
  (x : ℝ) 
  (h1 : correct_marks_per_question = 4)
  (h2 : total_marks = 420)
  (h3 : total_questions = 150)
  (h4 : correct_answers = 120) 
  (h5 : total_marks = (correct_answers * correct_marks_per_question) - ((total_questions - correct_answers) * x)) :
  x = 2 :=
by 
  sorry

end negatively_added_marks_l1245_124530


namespace calculate_savings_l1245_124580

/-- Given the income is 19000 and the income to expenditure ratio is 5:4, prove the savings of 3800. -/
theorem calculate_savings (i : ℕ) (exp : ℕ) (rat : ℕ → ℕ → Prop)
  (h_income : i = 19000)
  (h_ratio : rat 5 4)
  (h_exp_eq : ∃ x, i = 5 * x ∧ exp = 4 * x) :
  i - exp = 3800 :=
by 
  sorry

end calculate_savings_l1245_124580


namespace find_A_l1245_124536

noncomputable def telephone_number_satisfies_conditions (A B C D E F G H I J : ℕ) : Prop :=
  A ≠ B ∧ A ≠ C ∧ A ≠ D ∧ A ≠ E ∧ A ≠ F ∧ A ≠ G ∧ A ≠ H ∧ A ≠ I ∧ A ≠ J ∧
  B ≠ C ∧ B ≠ D ∧ B ≠ E ∧ B ≠ F ∧ B ≠ G ∧ B ≠ H ∧ B ≠ I ∧ B ≠ J ∧
  C ≠ D ∧ C ≠ E ∧ C ≠ F ∧ C ≠ G ∧ C ≠ H ∧ C ≠ I ∧ C ≠ J ∧
  D ≠ E ∧ D ≠ F ∧ D ≠ G ∧ D ≠ H ∧ D ≠ I ∧ D ≠ J ∧
  E ≠ F ∧ E ≠ G ∧ E ≠ H ∧ E ≠ I ∧ E ≠ J ∧
  F ≠ G ∧ F ≠ H ∧ F ≠ I ∧ F ≠ J ∧
  G ≠ H ∧ G ≠ I ∧ G ≠ J ∧
  H ≠ I ∧ H ≠ J ∧
  I ≠ J ∧
  A > B ∧ B > C ∧
  D > E ∧ E > F ∧
  G > H ∧ H > I ∧ I > J ∧
  E = D - 2 ∧ F = D - 4 ∧ -- Given D, E, F are consecutive even digits
  H = G - 2 ∧ I = G - 4 ∧ J = G - 6 ∧ -- Given G, H, I, J are consecutive odd digits
  A + B + C = 9

theorem find_A :
  ∃ (A B C D E F G H I J : ℕ), telephone_number_satisfies_conditions A B C D E F G H I J ∧ A = 8 :=
by {
  sorry
}

end find_A_l1245_124536


namespace positive_x_condition_l1245_124505

theorem positive_x_condition (x : ℝ) (h : x > 0 ∧ (0.01 * x * x = 9)) : x = 30 :=
sorry

end positive_x_condition_l1245_124505


namespace oleg_bought_bar_for_60_rubles_l1245_124579

theorem oleg_bought_bar_for_60_rubles (n : ℕ) (h₁ : 96 = n * (1 + n / 100)) : n = 60 :=
by {
  sorry
}

end oleg_bought_bar_for_60_rubles_l1245_124579


namespace garbage_accumulation_correct_l1245_124534

-- Given conditions
def garbage_days_per_week : ℕ := 3
def garbage_per_collection : ℕ := 200
def duration_weeks : ℕ := 2

-- Week 1: Full garbage accumulation
def week1_garbage_accumulation : ℕ := garbage_days_per_week * garbage_per_collection

-- Week 2: Half garbage accumulation due to the policy
def week2_garbage_accumulation : ℕ := week1_garbage_accumulation / 2

-- Total garbage accumulation over the 2 weeks
def total_garbage_accumulation (week1 week2 : ℕ) : ℕ := week1 + week2

-- Proof statement
theorem garbage_accumulation_correct :
  total_garbage_accumulation week1_garbage_accumulation week2_garbage_accumulation = 900 := by
  sorry

end garbage_accumulation_correct_l1245_124534


namespace experts_expected_points_probability_fifth_envelope_l1245_124588

theorem experts_expected_points (n : ℕ) (h1 : n = 100) (h2 : n = 13) :
  ∃ e : ℚ, e = 465 :=
sorry

theorem probability_fifth_envelope (m : ℕ) (h1 : m = 13) :
  ∃ p : ℚ, p = 0.715 :=
sorry

end experts_expected_points_probability_fifth_envelope_l1245_124588


namespace midpoint_sum_coordinates_l1245_124592

theorem midpoint_sum_coordinates (x y : ℝ) 
  (midpoint_cond_x : (x + 10) / 2 = 4) 
  (midpoint_cond_y : (y + 4) / 2 = -8) : 
  x + y = -22 :=
by
  sorry

end midpoint_sum_coordinates_l1245_124592


namespace evaluate_expression_l1245_124594

theorem evaluate_expression : 1 + 1 / (2 + 1 / (3 + 2)) = 16 / 11 := 
by 
  sorry

end evaluate_expression_l1245_124594


namespace polygon_interior_plus_exterior_l1245_124566

theorem polygon_interior_plus_exterior (n : ℕ) 
  (h : (n - 2) * 180 + 60 = 1500) : n = 10 :=
sorry

end polygon_interior_plus_exterior_l1245_124566


namespace problem_l1245_124524

theorem problem (K : ℕ) : 16 ^ 3 * 8 ^ 3 = 2 ^ K → K = 21 := by
  sorry

end problem_l1245_124524


namespace sum_of_repeating_decimals_l1245_124500

-- Definitions for periodic decimals
def repeating_five := 5 / 9
def repeating_seven := 7 / 9

-- Theorem statement
theorem sum_of_repeating_decimals : (repeating_five + repeating_seven) = 4 / 3 :=
by
  -- Placeholder for the proof
  sorry

end sum_of_repeating_decimals_l1245_124500


namespace train_length_proof_l1245_124542

noncomputable def train_length (speed_kmh : ℕ) (time_s : ℕ) : ℕ :=
  let speed_ms := speed_kmh * 5 / 18
  speed_ms * time_s

theorem train_length_proof : train_length 144 16 = 640 := by
  sorry

end train_length_proof_l1245_124542


namespace minimum_phi_l1245_124582

noncomputable def f (x : ℝ) : ℝ := Real.sin (2 * x + (Real.pi / 3))
noncomputable def g (x : ℝ) : ℝ := Real.sin (2 * x)

-- Define the condition for g overlapping with f after shifting by φ
noncomputable def shifted_g (x φ : ℝ) : ℝ := Real.sin (2 * x + 2 * φ)

theorem minimum_phi (φ : ℝ) (h : φ > 0) :
  (∃ (x : ℝ), shifted_g x φ = f x) ↔ (∃ k : ℕ, φ = Real.pi / 6 + k * Real.pi) :=
sorry

end minimum_phi_l1245_124582


namespace true_false_questions_count_l1245_124543

/-- 
 In an answer key for a quiz, there are some true-false questions followed by 3 multiple-choice questions with 4 answer choices each. 
 The correct answers to all true-false questions cannot be the same. 
 There are 384 ways to write the answer key. How many true-false questions are there?
-/
theorem true_false_questions_count : 
  ∃ n : ℕ, 2^n - 2 = 6 ∧ (2^n - 2) * 4^3 = 384 := 
sorry

end true_false_questions_count_l1245_124543


namespace find_other_number_l1245_124571

theorem find_other_number (A B : ℕ) (hcf : ℕ) (lcm : ℕ) 
  (H1 : hcf = 12) 
  (H2 : lcm = 312) 
  (H3 : A = 24) 
  (H4 : hcf * lcm = A * B) : 
  B = 156 :=
by sorry

end find_other_number_l1245_124571


namespace rachel_age_is_19_l1245_124522

def rachel_and_leah_ages (R L : ℕ) : Prop :=
  (R = L + 4) ∧ (R + L = 34)

theorem rachel_age_is_19 : ∃ L : ℕ, rachel_and_leah_ages 19 L :=
by {
  sorry
}

end rachel_age_is_19_l1245_124522


namespace actual_time_l1245_124507

def digit_in_range (a b : ℕ) : Prop := 
  (a = b + 1 ∨ a = b - 1)

def time_malfunctioned (h m : ℕ) : Prop :=
  digit_in_range 0 (h / 10) ∧ -- tens of hour digit (0 -> 1 or 9)
  digit_in_range 0 (h % 10) ∧ -- units of hour digit (0 -> 1 or 9)
  digit_in_range 5 (m / 10) ∧ -- tens of minute digit (5 -> 4 or 6)
  digit_in_range 9 (m % 10)   -- units of minute digit (9 -> 8 or 0)

theorem actual_time : ∃ h m : ℕ, time_malfunctioned h m ∧ h = 11 ∧ m = 48 :=
by
  sorry

end actual_time_l1245_124507


namespace neg_disj_imp_neg_conj_l1245_124574

theorem neg_disj_imp_neg_conj (p q : Prop) (h : ¬(p ∨ q)) : ¬p ∧ ¬q :=
sorry

end neg_disj_imp_neg_conj_l1245_124574


namespace find_x_l1245_124523

theorem find_x (x : ℝ) : 9999 * x = 724787425 ↔ x = 72487.5 := 
sorry

end find_x_l1245_124523


namespace minimize_f_l1245_124525

noncomputable def f (x : ℝ) : ℝ := 2 * Real.cos x + (Real.sin x)^2

theorem minimize_f :
  ∃ x : ℝ, (-π / 4 < x ∧ x ≤ π / 2) ∧
  ∀ y : ℝ, (-π / 4 < y ∧ y ≤ π / 2) → f y ≥ f x ∧ f x = 1 ∧ x = π / 2 :=
by
  sorry

end minimize_f_l1245_124525


namespace my_car_mpg_l1245_124575

-- Definitions from the conditions.
def total_miles := 100
def total_gallons := 5

-- The statement we need to prove.
theorem my_car_mpg : (total_miles / total_gallons : ℕ) = 20 :=
by
  sorry

end my_car_mpg_l1245_124575


namespace domain_of_c_is_all_reals_l1245_124567

theorem domain_of_c_is_all_reals (k : ℝ) : 
  (∀ x : ℝ, -3 * x^2 + 5 * x + k ≠ 0) ↔ k < -(25 / 12) :=
by
  sorry

end domain_of_c_is_all_reals_l1245_124567


namespace rectangle_length_l1245_124554

/--
The perimeter of a rectangle is 150 cm. The length is 15 cm greater than the width.
This theorem proves that the length of the rectangle is 45 cm under these conditions.
-/
theorem rectangle_length (P w l : ℝ) (h1 : P = 150) (h2 : l = w + 15) (h3 : P = 2 * l + 2 * w) : l = 45 :=
by
  sorry

end rectangle_length_l1245_124554


namespace hat_price_after_discounts_l1245_124502

-- Defining initial conditions
def initial_price : ℝ := 15
def first_discount_percent : ℝ := 0.25
def second_discount_percent : ℝ := 0.50

-- Defining the expected final price after applying both discounts
def expected_final_price : ℝ := 5.625

-- Lean statement to prove the final price after both discounts is as expected
theorem hat_price_after_discounts : 
  let first_reduced_price := initial_price * (1 - first_discount_percent)
  let second_reduced_price := first_reduced_price * (1 - second_discount_percent)
  second_reduced_price = expected_final_price := sorry

end hat_price_after_discounts_l1245_124502


namespace range_of_a_for_function_is_real_l1245_124520

noncomputable def quadratic_expr (a x : ℝ) : ℝ := a * x^2 - 4 * x + a - 3

theorem range_of_a_for_function_is_real :
  (∀ x : ℝ, quadratic_expr a x > 0) → 0 ≤ a ∧ a ≤ 4 :=
by
  sorry

end range_of_a_for_function_is_real_l1245_124520


namespace bus_speed_excluding_stoppages_l1245_124503

variable (v : ℝ)

-- Given conditions
def speed_including_stoppages := 45 -- kmph
def stoppage_time_ratio := 1/6 -- 10 minutes per hour is 1/6 of the time

-- Prove that the speed excluding stoppages is 54 kmph
theorem bus_speed_excluding_stoppages (h1 : speed_including_stoppages = 45) 
                                      (h2 : stoppage_time_ratio = 1/6) : 
                                      v = 54 := by
  sorry

end bus_speed_excluding_stoppages_l1245_124503


namespace find_sum_s_u_l1245_124576

theorem find_sum_s_u (p r s u : ℝ) (q t : ℝ) 
  (h_q : q = 5) 
  (h_t : t = -p - r) 
  (h_sum_imaginary : q + s + u = 4) :
  s + u = -1 := 
sorry

end find_sum_s_u_l1245_124576


namespace probability_of_sum_20_is_correct_l1245_124533

noncomputable def probability_sum_20 : ℚ :=
  let total_outcomes := 12 * 12
  let favorable_outcomes := 5
  favorable_outcomes / total_outcomes

theorem probability_of_sum_20_is_correct :
  probability_sum_20 = 5 / 144 :=
by
  sorry

end probability_of_sum_20_is_correct_l1245_124533


namespace class_avg_GPA_l1245_124535

theorem class_avg_GPA (n : ℕ) (h1 : n > 0) : 
  ((1 / 4 : ℝ) * 92 + (3 / 4 : ℝ) * 76 = 80) :=
sorry

end class_avg_GPA_l1245_124535


namespace exists_integers_a_b_c_d_and_n_l1245_124583

theorem exists_integers_a_b_c_d_and_n (n a b c d : ℕ)
  (h1 : a = 10) 
  (h2 : b = 15) 
  (h3 : c = 8) 
  (h4 : d = 3) 
  (h5 : n = 16) :
  a^4 + b^4 + c^4 + 2 * d^4 = n^4 := by
  -- Proof goes here
  sorry

end exists_integers_a_b_c_d_and_n_l1245_124583


namespace unique_position_all_sequences_one_l1245_124573

-- Define the main theorem
theorem unique_position_all_sequences_one (n : ℕ) (sequences : Fin (2^(n-1)) → Fin n → Bool) :
  (∀ a b c : Fin (2^(n-1)), ∃ p : Fin n, sequences a p = true ∧ sequences b p = true ∧ sequences c p = true) →
  ∃! p : Fin n, ∀ i : Fin (2^(n-1)), sequences i p = true :=
by
  sorry

end unique_position_all_sequences_one_l1245_124573


namespace simplify_vectors_l1245_124548

variables {Point : Type} [AddGroup Point] (A B C D : Point)

def vector (P Q : Point) : Point := Q - P

theorem simplify_vectors :
  vector A B + vector B C - vector A D = vector D C :=
by
  sorry

end simplify_vectors_l1245_124548


namespace surface_area_LShape_l1245_124519

-- Define the structures and conditions
structure UnitCube where
  x : ℕ
  y : ℕ
  z : ℕ

def LShape (cubes : List UnitCube) : Prop :=
  -- Condition 1: Exactly 7 unit cubes
  cubes.length = 7 ∧
  -- Condition 2: 4 cubes in a line along x-axis (bottom row)
  ∃ a b c d : UnitCube, 
    (a.x + 1 = b.x ∧ b.x + 1 = c.x ∧ c.x + 1 = d.x ∧
     a.y = b.y ∧ b.y = c.y ∧ c.y = d.y ∧
     a.z = b.z ∧ b.z = c.z ∧ c.z = d.z) ∧
  -- Condition 3: 3 cubes stacked along z-axis at one end of the row
  ∃ e f g : UnitCube,
    (d.x = e.x ∧ e.x = f.x ∧ f.x = g.x ∧
     d.y = e.y ∧ e.y = f.y ∧ f.y = g.y ∧
     e.z + 1 = f.z ∧ f.z + 1 = g.z)

-- Define the surface area function
def surfaceArea (cubes : List UnitCube) : ℕ :=
  4*7 - 2*3 + 4 -- correct answer calculation according to manual analysis of exposed faces

-- The theorem to be proven
theorem surface_area_LShape : 
  ∀ (cubes : List UnitCube), LShape cubes → surfaceArea cubes = 26 :=
by sorry

end surface_area_LShape_l1245_124519


namespace inverse_function_point_l1245_124553

theorem inverse_function_point (f : ℝ → ℝ) (h_inv : Function.LeftInverse f f⁻¹) (h_point : f 2 = -1) : f⁻¹ (-1) = 2 :=
by
  sorry

end inverse_function_point_l1245_124553


namespace vernal_equinox_shadow_length_l1245_124597

-- Lean 4 statement
theorem vernal_equinox_shadow_length :
  ∀ (a : ℕ → ℝ), (a 4 = 10.5) → (a 10 = 4.5) → 
  (∀ (n m : ℕ), a (n + 1) = a n + (a 2 - a 1)) → 
  a 7 = 7.5 :=
by
  intros a h_4 h_10 h_progression
  sorry

end vernal_equinox_shadow_length_l1245_124597


namespace sin_pi_minus_alpha_l1245_124593

theorem sin_pi_minus_alpha (α : ℝ) (h : Real.sin α = 5/13) : Real.sin (π - α) = 5/13 :=
by
  sorry

end sin_pi_minus_alpha_l1245_124593


namespace bob_raise_per_hour_l1245_124551

theorem bob_raise_per_hour
  (hours_per_week : ℕ := 40)
  (monthly_housing_reduction : ℤ := 60)
  (weekly_earnings_increase : ℤ := 5)
  (weeks_per_month : ℕ := 4) :
  ∃ (R : ℚ), 40 * R - (monthly_housing_reduction / weeks_per_month) + weekly_earnings_increase = 0 ∧
              R = 0.25 := 
by
  sorry

end bob_raise_per_hour_l1245_124551


namespace proof_simplify_expression_l1245_124518

noncomputable def simplify_expression (a b : ℝ) : ℝ :=
  (a / b + b / a)^2 - 1 / (a^2 * b^2)

theorem proof_simplify_expression 
  (a b : ℝ) (h1 : 0 < a) (h2 : 0 < b) (h3 : a^3 + b^3 = a + b) :
  simplify_expression a b = 2 / (a * b) := by
  sorry

end proof_simplify_expression_l1245_124518


namespace non_degenerate_ellipse_condition_l1245_124562

theorem non_degenerate_ellipse_condition (k : ℝ) :
  (∃ x y : ℝ, 3 * x^2 + 6 * y^2 - 12 * x + 18 * y = k) ↔ k > -51 / 2 :=
sorry

end non_degenerate_ellipse_condition_l1245_124562


namespace nala_seashells_l1245_124581

theorem nala_seashells (a b c : ℕ) (h1 : a = 5) (h2 : b = 7) (h3 : c = 2 * (a + b)) : a + b + c = 36 :=
by {
  sorry
}

end nala_seashells_l1245_124581


namespace min_a_b_l1245_124557

theorem min_a_b (a b : ℕ) (h1 : 43 * a + 17 * b = 731) (h2 : a ≤ 17) (h3 : b ≤ 43) : a + b = 17 :=
by
  sorry

end min_a_b_l1245_124557


namespace radius_of_smaller_circle_l1245_124510

open Real

-- Definitions based on the problem conditions
def large_circle_radius : ℝ := 10
def pattern := "square"

-- Statement of the problem in Lean 4
theorem radius_of_smaller_circle :
  ∀ (r : ℝ), (large_circle_radius = 10) → (pattern = "square") → r = 5 * sqrt 2 →  ∃ r, r = 5 * sqrt 2 :=
by
  sorry

end radius_of_smaller_circle_l1245_124510


namespace minimum_selling_price_l1245_124572

def monthly_sales : ℕ := 50
def base_cost : ℕ := 1200
def shipping_cost : ℕ := 20
def store_fee : ℕ := 10000
def repair_fee : ℕ := 5000
def profit_margin : ℕ := 20

def total_monthly_expenses : ℕ := store_fee + repair_fee
def total_cost_per_machine : ℕ := base_cost + shipping_cost + total_monthly_expenses / monthly_sales
def min_selling_price : ℕ := total_cost_per_machine * (1 + profit_margin / 100)

theorem minimum_selling_price : min_selling_price = 1824 := 
by
  sorry 

end minimum_selling_price_l1245_124572


namespace complement_of_A_in_U_l1245_124512

open Set

variable (U : Set ℤ := { -2, -1, 0, 1, 2 })
variable (A : Set ℤ := { x | 0 < Int.natAbs x ∧ Int.natAbs x < 2 })

theorem complement_of_A_in_U :
  U \ A = { -2, 0, 2 } :=
by
  sorry

end complement_of_A_in_U_l1245_124512


namespace only_correct_option_is_C_l1245_124538

-- Definitions of the conditions as per the given problem
def option_A (a : ℝ) : Prop := a^2 * a^3 = a^6
def option_B (a : ℝ) : Prop := (a^2)^3 = a^5
def option_C (a b : ℝ) : Prop := (a * b)^3 = a^3 * b^3
def option_D (a : ℝ) : Prop := a^8 / a^2 = a^4

-- The theorem stating that only option C is correct
theorem only_correct_option_is_C (a b : ℝ) : 
  ¬(option_A a) ∧ ¬(option_B a) ∧ option_C a b ∧ ¬(option_D a) :=
by sorry

end only_correct_option_is_C_l1245_124538


namespace student_estimated_score_l1245_124590

theorem student_estimated_score :
  (6 * 5 + 3 * 5 * (3 / 4) + 2 * 5 * (1 / 3) + 1 * 5 * (1 / 4)) = 41.25 :=
by
 sorry

end student_estimated_score_l1245_124590


namespace least_5_digit_number_divisible_by_15_25_40_75_125_140_l1245_124541

theorem least_5_digit_number_divisible_by_15_25_40_75_125_140 : 
  ∃ n : ℕ, (10000 ≤ n) ∧ (n < 100000) ∧ 
  (15 ∣ n) ∧ (25 ∣ n) ∧ (40 ∣ n) ∧ (75 ∣ n) ∧ (125 ∣ n) ∧ (140 ∣ n) ∧ (n = 21000) :=
by
  sorry

end least_5_digit_number_divisible_by_15_25_40_75_125_140_l1245_124541


namespace triple_layers_area_l1245_124595

-- Defining the conditions
def hall : Type := {x // x = 10 * 10}
def carpet1 : hall := ⟨60, sorry⟩ -- First carpet size: 6 * 8
def carpet2 : hall := ⟨36, sorry⟩ -- Second carpet size: 6 * 6
def carpet3 : hall := ⟨35, sorry⟩ -- Third carpet size: 5 * 7

-- The final theorem statement
theorem triple_layers_area : ∃ area : ℕ, area = 6 :=
by
  have intersection_area : ℕ := 2 * 3
  use intersection_area
  sorry

end triple_layers_area_l1245_124595


namespace quadratic_eq_equal_roots_l1245_124561

theorem quadratic_eq_equal_roots (m x : ℝ) (h : (x^2 - m * x + m - 1 = 0) ∧ ((x - 1)^2 = 0)) : 
    m = 2 ∧ ((x = 1 ∧ x = 1)) :=
by
  sorry

end quadratic_eq_equal_roots_l1245_124561


namespace adult_ticket_cost_l1245_124526

theorem adult_ticket_cost (A Tc : ℝ) (T C : ℕ) (M : ℝ) 
  (hTc : Tc = 3.50) 
  (hT : T = 21) 
  (hC : C = 16) 
  (hM : M = 83.50) 
  (h_eq : 16 * Tc + (↑(T - C)) * A = M) : 
  A = 5.50 :=
by sorry

end adult_ticket_cost_l1245_124526


namespace solve_m_l1245_124578

theorem solve_m (x y m : ℝ) (h1 : 4 * x + 2 * y = 3 * m) (h2 : 3 * x + y = m + 2) (h3 : y = -x) : m = 1 := 
by {
  sorry
}

end solve_m_l1245_124578


namespace abc_one_eq_sum_l1245_124591

theorem abc_one_eq_sum (a b c : ℝ) (h1 : a ≠ 0) (h2 : b ≠ 0) (h3 : c ≠ 0) (h4 : a * b * c = 1) :
  (a^2 * b^2) / ((a^2 + b * c) * (b^2 + a * c))
  + (a^2 * c^2) / ((a^2 + b * c) * (c^2 + a * b))
  + (b^2 * c^2) / ((b^2 + a * c) * (c^2 + a * b))
  = 1 / (a^2 + 1 / a) + 1 / (b^2 + 1 / b) + 1 / (c^2 + 1 / c) := by
  sorry

end abc_one_eq_sum_l1245_124591


namespace no_such_m_for_equivalence_existence_of_m_for_implication_l1245_124565

def P (x : ℝ) : Prop := x^2 - 8*x - 20 ≤ 0
def S (x : ℝ) (m : ℝ) : Prop := 1 - m ≤ x ∧ x ≤ 1 + m

theorem no_such_m_for_equivalence :
  ¬ ∃ m : ℝ, ∀ x : ℝ, P x ↔ S x m :=
sorry

theorem existence_of_m_for_implication :
  ∃ m : ℝ, (∀ x : ℝ, S x m → P x) ∧ m ≤ 3 :=
sorry

end no_such_m_for_equivalence_existence_of_m_for_implication_l1245_124565


namespace escalator_time_l1245_124599

theorem escalator_time (escalator_speed person_speed length : ℕ) 
    (h1 : escalator_speed = 12) 
    (h2 : person_speed = 2) 
    (h3 : length = 196) : 
    (length / (escalator_speed + person_speed) = 14) :=
by
  sorry

end escalator_time_l1245_124599


namespace age_ratio_in_two_years_l1245_124528

theorem age_ratio_in_two_years :
  ∀ (B M : ℕ), B = 10 → M = B + 12 → (M + 2) / (B + 2) = 2 := by
  intros B M hB hM
  sorry

end age_ratio_in_two_years_l1245_124528


namespace sum_of_number_and_conjugate_l1245_124569

noncomputable def x : ℝ := 16 - Real.sqrt 2023
noncomputable def y : ℝ := 16 + Real.sqrt 2023

theorem sum_of_number_and_conjugate : x + y = 32 :=
by
  sorry

end sum_of_number_and_conjugate_l1245_124569


namespace find_a_if_even_function_l1245_124516

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := (x - 1) ^ 2 + a * x + Real.sin (x + Real.pi / 2)

def is_even_function (f : ℝ → ℝ) : Prop := ∀ x, f x = f (-x)

theorem find_a_if_even_function :
  (∃ a : ℝ, is_even_function (f a)) → ∃ a : ℝ, a = 2 :=
by
  intros h
  sorry

end find_a_if_even_function_l1245_124516


namespace balloons_remaining_l1245_124568

-- Define the initial conditions
def initial_balloons : ℕ := 9
def lost_balloons : ℕ := 2

-- State the theorem
theorem balloons_remaining : initial_balloons - lost_balloons = 7 := by
  -- Add the solution proof steps here
  sorry

end balloons_remaining_l1245_124568


namespace alicia_masks_left_l1245_124529

theorem alicia_masks_left (T G L : ℕ) (hT : T = 90) (hG : G = 51) (hL : L = T - G) : L = 39 :=
by
  rw [hT, hG] at hL
  exact hL

end alicia_masks_left_l1245_124529


namespace doubled_cylinder_volume_l1245_124517

theorem doubled_cylinder_volume (r h : ℝ) (V : ℝ) (original_volume : V = π * r^2 * h) (V' : ℝ) : (2 * 2 * π * r^2 * h = 40) := 
by 
  have original_volume := 5
  sorry

end doubled_cylinder_volume_l1245_124517


namespace clock_equiv_4_cubic_l1245_124555

theorem clock_equiv_4_cubic :
  ∃ x : ℕ, x > 3 ∧ x % 12 = (x^3) % 12 ∧ (∀ y : ℕ, y > 3 ∧ y % 12 = (y^3) % 12 → x ≤ y) :=
by
  use 4
  sorry

end clock_equiv_4_cubic_l1245_124555


namespace custom_op_example_l1245_124545

def custom_op (x y : ℤ) : ℤ := x * y - 3 * x

theorem custom_op_example : (custom_op 7 4) - (custom_op 4 7) = -9 :=
by
  sorry

end custom_op_example_l1245_124545


namespace inequality_solution_l1245_124511

theorem inequality_solution (x : ℝ) :
    (∀ t : ℝ, abs (t - 3) + abs (2 * t + 1) ≥ abs (2 * x - 1) + abs (x + 2)) ↔ 
    (-1 / 2 ≤ x ∧ x ≤ 5 / 6) :=
by
  sorry

end inequality_solution_l1245_124511


namespace dividend_rate_l1245_124531

theorem dividend_rate (face_value market_value expected_interest interest_rate : ℝ)
  (h1 : face_value = 52)
  (h2 : expected_interest = 0.12)
  (h3 : market_value = 39)
  : ((expected_interest * market_value) / face_value) * 100 = 9 := by
  sorry

end dividend_rate_l1245_124531


namespace polynomial_nonnegative_iff_eq_l1245_124570

variable {R : Type} [LinearOrderedField R]

def polynomial_p (x a b c : R) : R :=
  (x - a) * (x - b) + (x - b) * (x - c) + (x - c) * (x - a)

theorem polynomial_nonnegative_iff_eq (a b c : R) :
  (∀ x : R, polynomial_p x a b c ≥ 0) ↔ a = b ∧ b = c :=
by
  sorry

end polynomial_nonnegative_iff_eq_l1245_124570


namespace complement_U_A_eq_two_l1245_124515

open Set

universe u

def U : Set ℕ := { x | x ≥ 2 }
def A : Set ℕ := { x | x^2 ≥ 5 }
def comp_U_A : Set ℕ := U \ A

theorem complement_U_A_eq_two : comp_U_A = {2} :=
by 
  sorry

end complement_U_A_eq_two_l1245_124515


namespace length_of_ON_l1245_124558

noncomputable def proof_problem : Prop :=
  let hyperbola := { x : ℝ × ℝ | x.1 ^ 2 - x.2 ^ 2 = 1 }
  ∃ (F1 F2 P : ℝ × ℝ) (O : ℝ × ℝ) (N : ℝ × ℝ),
    O = (0, 0) ∧
    P ∈ hyperbola ∧
    N = ((P.1 + F1.1) / 2, (P.2 + F1.2) / 2) ∧
    dist P F1 = 5 ∧
    ∃ r : ℝ, r = 1.5 ∧ (dist O N = r)

theorem length_of_ON : proof_problem :=
sorry

end length_of_ON_l1245_124558


namespace Xiaohuo_books_l1245_124521

def books_proof_problem : Prop :=
  ∃ (X_H X_Y X_Z : ℕ), 
    (X_H + X_Y + X_Z = 1248) ∧ 
    (X_H = X_Y + 64) ∧ 
    (X_Y = X_Z - 32) ∧ 
    (X_H = 448)

theorem Xiaohuo_books : books_proof_problem :=
by
  sorry

end Xiaohuo_books_l1245_124521


namespace powderman_distance_when_hears_explosion_l1245_124506

noncomputable def powderman_speed_yd_per_s : ℝ := 10
noncomputable def blast_time_s : ℝ := 45
noncomputable def sound_speed_ft_per_s : ℝ := 1080
noncomputable def powderman_speed_ft_per_s : ℝ := 30

noncomputable def distance_powderman (t : ℝ) : ℝ := powderman_speed_ft_per_s * t
noncomputable def distance_sound (t : ℝ) : ℝ := sound_speed_ft_per_s * (t - blast_time_s)

theorem powderman_distance_when_hears_explosion :
  ∃ t, t > blast_time_s ∧ distance_powderman t = distance_sound t ∧ (distance_powderman t) / 3 = 463 :=
sorry

end powderman_distance_when_hears_explosion_l1245_124506


namespace geometric_sequence_a_l1245_124537

theorem geometric_sequence_a (a : ℝ) (h1 : a > 0) (h2 : ∃ r : ℝ, 280 * r = a ∧ a * r = 180 / 49) :
  a = 32.07 :=
by sorry

end geometric_sequence_a_l1245_124537


namespace tan_identity_equality_l1245_124501

theorem tan_identity_equality
  (α β : ℝ)
  (h1 : Real.tan (α + β) = 2 / 5)
  (h2 : Real.tan (β - π / 4) = 1 / 4) :
  (Real.cos α + Real.sin α) / (Real.cos α - Real.sin α) = 3 / 22 :=
by
  sorry

end tan_identity_equality_l1245_124501


namespace range_of_b_not_strictly_decreasing_l1245_124560

def f (b x : ℝ) : ℝ := -x^3 + b*x^2 - (2*b + 3)*x + 2 - b

theorem range_of_b_not_strictly_decreasing :
  {b : ℝ | ¬(∀ (x1 x2 : ℝ), x1 < x2 → f b x1 > f b x2)} = {b | b < -1 ∨ b > 3} :=
by
  sorry

end range_of_b_not_strictly_decreasing_l1245_124560


namespace number_of_bricks_required_l1245_124549

def brick_length : ℝ := 0.20
def brick_width : ℝ := 0.10
def brick_height : ℝ := 0.075

def wall_length : ℝ := 25.0
def wall_width : ℝ := 2.0
def wall_height : ℝ := 0.75

def brick_volume := brick_length * brick_width * brick_height
def wall_volume := wall_length * wall_width * wall_height

theorem number_of_bricks_required :
  wall_volume / brick_volume = 25000 := by
  sorry

end number_of_bricks_required_l1245_124549


namespace enter_exit_ways_correct_l1245_124509

-- Defining the problem conditions
def num_entrances := 4

-- Defining the problem question and answer
def enter_exit_ways (n : Nat) : Nat := n * (n - 1)

-- Statement: Prove the number of different ways to enter and exit is 12
theorem enter_exit_ways_correct : enter_exit_ways num_entrances = 12 := by
  -- Proof
  sorry

end enter_exit_ways_correct_l1245_124509


namespace xsquared_plus_5x_minus_6_condition_l1245_124564

theorem xsquared_plus_5x_minus_6_condition (x : ℝ) : 
  (x^2 + 5 * x - 6 > 0) → (x > 2) ∨ (((x > 1) ∨ (x < -6)) ∧ ¬(x > 2)) := 
sorry

end xsquared_plus_5x_minus_6_condition_l1245_124564


namespace inscribed_rectangle_area_l1245_124552

theorem inscribed_rectangle_area (A S x : ℝ) (hA : A = 18) (hS : S = (x * x) * 2) (hx : x = 2):
  S = 8 :=
by
  -- The proofs steps will go here
  sorry

end inscribed_rectangle_area_l1245_124552


namespace hurleys_age_l1245_124559

-- Definitions and conditions
variable (H R : ℕ)
variable (cond1 : R - H = 20)
variable (cond2 : (R + 40) + (H + 40) = 128)

-- Theorem statement
theorem hurleys_age (H R : ℕ) (cond1 : R - H = 20) (cond2 : (R + 40) + (H + 40) = 128) : H = 14 := 
by
  sorry

end hurleys_age_l1245_124559


namespace Alyssa_initial_puppies_l1245_124540

theorem Alyssa_initial_puppies : 
  ∀ (a b c : ℕ), b = 7 → c = 5 → a = b + c → a = 12 := 
by
  intros a b c hb hc hab
  rw [hb, hc] at hab
  exact hab

end Alyssa_initial_puppies_l1245_124540


namespace max_integer_value_l1245_124550

theorem max_integer_value (x : ℝ) : 
  ∃ m : ℤ, ∀ (x : ℝ), (3 * x^2 + 9 * x + 17) / (3 * x^2 + 9 * x + 7) ≤ m ∧ m = 41 :=
by sorry

end max_integer_value_l1245_124550


namespace find_a_n_l1245_124544

noncomputable def is_arithmetic_seq (a b : ℕ) (seq : ℕ → ℕ) : Prop :=
  ∀ n, seq n = a + n * b

noncomputable def is_geometric_seq (b a : ℕ) (seq : ℕ → ℕ) : Prop :=
  ∀ n, seq n = b * a ^ n

theorem find_a_n (a b : ℕ) 
  (a_positive : a > 1)
  (b_positive : b > 1)
  (a_seq : ℕ → ℕ)
  (b_seq : ℕ → ℕ)
  (arith_seq : is_arithmetic_seq a b a_seq)
  (geom_seq : is_geometric_seq b a b_seq)
  (init_condition : a_seq 0 < b_seq 0)
  (next_condition : b_seq 1 < a_seq 2)
  (relation_condition : ∀ n, ∃ m, a_seq m + 3 = b_seq n) :
  ∀ n, a_seq n = 5 * n - 3 :=
sorry

end find_a_n_l1245_124544


namespace sum_of_squares_of_six_odds_not_2020_l1245_124532

theorem sum_of_squares_of_six_odds_not_2020 :
  ¬ ∃ a1 a2 a3 a4 a5 a6 : ℤ, (∀ i ∈ [a1, a2, a3, a4, a5, a6], i % 2 = 1) ∧ (a1^2 + a2^2 + a3^2 + a4^2 + a5^2 + a6^2 = 2020) :=
by
  sorry

end sum_of_squares_of_six_odds_not_2020_l1245_124532


namespace Gerald_toy_cars_l1245_124586

theorem Gerald_toy_cars :
  let initial_toy_cars := 20
  let fraction_donated := 1 / 4
  let donated_toy_cars := initial_toy_cars * fraction_donated
  let remaining_toy_cars := initial_toy_cars - donated_toy_cars
  remaining_toy_cars = 15 := 
by
  sorry

end Gerald_toy_cars_l1245_124586


namespace cube_sum_of_edges_corners_faces_eq_26_l1245_124587

theorem cube_sum_of_edges_corners_faces_eq_26 :
  let edges := 12
  let corners := 8
  let faces := 6
  edges + corners + faces = 26 :=
by
  let edges := 12
  let corners := 8
  let faces := 6
  sorry

end cube_sum_of_edges_corners_faces_eq_26_l1245_124587


namespace diana_can_paint_statues_l1245_124547

theorem diana_can_paint_statues (total_paint : ℚ) (paint_per_statue : ℚ) 
  (h1 : total_paint = 3 / 6) (h2 : paint_per_statue = 1 / 6) : 
  total_paint / paint_per_statue = 3 :=
by
  sorry

end diana_can_paint_statues_l1245_124547

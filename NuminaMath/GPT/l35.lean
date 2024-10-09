import Mathlib

namespace express_y_in_terms_of_x_l35_3578

theorem express_y_in_terms_of_x (x y : ℝ) (h : 5 * x + y = 4) : y = 4 - 5 * x :=
by
  /- Proof to be filled in here. -/
  sorry

end express_y_in_terms_of_x_l35_3578


namespace pythagorean_triple_9_12_15_l35_3582

theorem pythagorean_triple_9_12_15 : 9^2 + 12^2 = 15^2 :=
by 
  sorry

end pythagorean_triple_9_12_15_l35_3582


namespace board_tiling_condition_l35_3597

-- Define the problem in Lean

theorem board_tiling_condition (n : ℕ) : 
  (∃ m : ℕ, n * n = m + 4 * m) ↔ (∃ k : ℕ, n = 5 * k ∧ n > 5) := by 
sorry

end board_tiling_condition_l35_3597


namespace alberto_bikes_more_l35_3583

-- Definitions of given speeds
def alberto_speed : ℝ := 15
def bjorn_speed : ℝ := 11.25

-- The time duration considered
def time_hours : ℝ := 5

-- Calculate the distances each traveled
def alberto_distance : ℝ := alberto_speed * time_hours
def bjorn_distance : ℝ := bjorn_speed * time_hours

-- Calculate the difference in distances
def distance_difference : ℝ := alberto_distance - bjorn_distance

-- The theorem to be proved
theorem alberto_bikes_more : distance_difference = 18.75 := by
    sorry

end alberto_bikes_more_l35_3583


namespace right_triangle_perpendicular_ratio_l35_3501

theorem right_triangle_perpendicular_ratio {a b c r s : ℝ}
 (h : a^2 + b^2 = c^2)
 (perpendicular : r + s = c)
 (ratio_ab : a / b = 2 / 3) :
 r / s = 4 / 9 :=
sorry

end right_triangle_perpendicular_ratio_l35_3501


namespace proof_of_problem1_proof_of_problem2_proof_of_problem3_proof_of_problem4_l35_3530

noncomputable def problem1 (x y : ℝ) (h : x^2 - 6*x + 2*y = 0) : Prop :=
  y ≤ 4.5

noncomputable def problem2 (x y : ℝ) (h : 3*x^2 + 12*x - 2*y - 4 = 0) : Prop :=
  y ≥ -8

noncomputable def problem3 (x y : ℝ) (h : y = 2*x / (1 + x^2)) : Prop :=
  -1 ≤ y ∧ y ≤ 1

noncomputable def problem4 (x y : ℝ) (h : y = (2*x - 1) / (x^2 + 2*x + 1)) : Prop :=
  y ≤ 1/3

-- Proving that the properties hold:
theorem proof_of_problem1 (x y : ℝ) (h : x^2 - 6*x + 2*y = 0) : problem1 x y h :=
  sorry

theorem proof_of_problem2 (x y : ℝ) (h : 3*x^2 + 12*x - 2*y - 4 = 0) : problem2 x y h :=
  sorry

theorem proof_of_problem3 (x y : ℝ) (h : y = 2*x / (1 + x^2)) : problem3 x y h :=
  sorry

theorem proof_of_problem4 (x y : ℝ) (h : y = (2*x - 1) / (x^2 + 2*x + 1)) : problem4 x y h :=
  sorry

end proof_of_problem1_proof_of_problem2_proof_of_problem3_proof_of_problem4_l35_3530


namespace train_length_l35_3507

noncomputable def length_of_first_train (l2 : ℝ) (v1 : ℝ) (v2 : ℝ) (t : ℝ) : ℝ :=
  let v1_m_per_s := v1 * 1000 / 3600
  let v2_m_per_s := v2 * 1000 / 3600
  let relative_speed := v1_m_per_s + v2_m_per_s
  let combined_length := relative_speed * t
  combined_length - l2

theorem train_length (l2 : ℝ) (v1 : ℝ) (v2 : ℝ) (t : ℝ) (h_l2 : l2 = 200) 
  (h_v1 : v1 = 100) (h_v2 : v2 = 200) (h_t : t = 3.6) : length_of_first_train l2 v1 v2 t = 100 := by
  sorry

end train_length_l35_3507


namespace correct_survey_method_l35_3511

def service_life_of_light_tubes (survey_method : String) : Prop :=
  survey_method = "comprehensive"

def viewership_rate_of_spring_festival_gala (survey_method : String) : Prop :=
  survey_method = "comprehensive"

def crash_resistance_of_cars (survey_method : String) : Prop :=
  survey_method = "sample"

def fastest_student_for_sports_meeting (survey_method : String) : Prop :=
  survey_method = "sample"

theorem correct_survey_method :
  ¬(service_life_of_light_tubes "comprehensive") ∧
  ¬(viewership_rate_of_spring_festival_gala "comprehensive") ∧
  ¬(crash_resistance_of_cars "sample") ∧
  (fastest_student_for_sports_meeting "sample") :=
sorry

end correct_survey_method_l35_3511


namespace exactly_one_greater_than_one_l35_3524

theorem exactly_one_greater_than_one (x1 x2 x3 : ℝ) 
  (h1 : 0 < x1) (h2 : 0 < x2) (h3 : 0 < x3)
  (h4 : x1 * x2 * x3 = 1)
  (h5 : x1 + x2 + x3 > (1 / x1) + (1 / x2) + (1 / x3)) :
  (x1 > 1 ∧ x2 ≤ 1 ∧ x3 ≤ 1) ∨ 
  (x1 ≤ 1 ∧ x2 > 1 ∧ x3 ≤ 1) ∨ 
  (x1 ≤ 1 ∧ x2 ≤ 1 ∧ x3 > 1) :=
sorry

end exactly_one_greater_than_one_l35_3524


namespace balls_into_boxes_l35_3565

theorem balls_into_boxes : ∃ n : ℕ, n = 240 ∧ ∃ f : Fin 5 → Fin 4, ∀ i : Fin 4, ∃ j : Fin 5, f j = i := by
  sorry

end balls_into_boxes_l35_3565


namespace distance_against_stream_l35_3552

variable (vs : ℝ) -- speed of the stream

-- condition: in one hour, the boat goes 9 km along the stream
def cond1 (vs : ℝ) := 7 + vs = 9

-- condition: the speed of the boat in still water (7 km/hr)
def speed_still_water := 7

-- theorem to prove: the distance the boat goes against the stream in one hour
theorem distance_against_stream (vs : ℝ) (h : cond1 vs) : 
  (speed_still_water - vs) * 1 = 5 :=
by
  rw [speed_still_water, mul_one]
  sorry

end distance_against_stream_l35_3552


namespace jim_travel_distance_l35_3518

theorem jim_travel_distance
  (john_distance : ℕ := 15)
  (jill_distance : ℕ := john_distance - 5)
  (jim_distance : ℕ := jill_distance * 20 / 100) :
  jim_distance = 2 := 
by
  sorry

end jim_travel_distance_l35_3518


namespace sufficient_but_not_necessary_l35_3516

theorem sufficient_but_not_necessary (a : ℝ) (h1 : a > 0) (h2 : |a| > 0 → a > 0 ∨ a < 0) : 
  (a > 0 → |a| > 0) ∧ (¬(|a| > 0 → a > 0)) := 
by
  sorry

end sufficient_but_not_necessary_l35_3516


namespace smallest_positive_b_factors_l35_3542

theorem smallest_positive_b_factors (b : ℤ) : 
  (∃ p q : ℤ, x^2 + b * x + 2016 = (x + p) * (x + q) ∧ p + q = b ∧ p * q = 2016 ∧ p > 0 ∧ q > 0) → b = 95 := 
by {
  sorry
}

end smallest_positive_b_factors_l35_3542


namespace parametric_hyperbola_l35_3544

theorem parametric_hyperbola (t : ℝ) (ht : t ≠ 0) : 
  let x := t + 1 / t
  let y := t - 1 / t
  x^2 - y^2 = 4 :=
by
  let x := t + 1 / t
  let y := t - 1 / t
  sorry

end parametric_hyperbola_l35_3544


namespace smallest_n_inequality_l35_3593

theorem smallest_n_inequality :
  ∃ n : ℤ, (∀ x y z w : ℝ, (x^2 + y^2 + z^2 + w^2)^2 ≤ n * (x^4 + y^4 + z^4 + w^4)) ∧
           (∀ m : ℤ, (∀ x y z w : ℝ, (x^2 + y^2 + z^2 + w^2)^2 ≤ m * (x^4 + y^4 + z^4 + w^4)) → n ≤ m) ∧
           n = 4 :=
by
  let n := 4
  sorry

end smallest_n_inequality_l35_3593


namespace greatest_integer_radius_of_circle_l35_3587

theorem greatest_integer_radius_of_circle (r : ℕ) (A : ℝ) (hA : A < 80 * Real.pi) :
  r <= 8 ∧ r * r < 80 :=
sorry

end greatest_integer_radius_of_circle_l35_3587


namespace solve_for_x_l35_3557

theorem solve_for_x (x : ℝ) : (5 * x - 2) / (6 * x - 6) = 3 / 4 ↔ x = -5 := by
  sorry

end solve_for_x_l35_3557


namespace smallest_integer_coprime_with_462_l35_3554

theorem smallest_integer_coprime_with_462 :
  ∃ n, n > 1 ∧ Nat.gcd n 462 = 1 ∧ ∀ m, m > 1 ∧ Nat.gcd m 462 = 1 → n ≤ m → n = 13 := by
  sorry

end smallest_integer_coprime_with_462_l35_3554


namespace average_abcd_l35_3588

-- Define the average condition of the numbers 4, 6, 9, a, b, c, d given as 20
def average_condition (a b c d : ℝ) : Prop :=
  (4 + 6 + 9 + a + b + c + d) / 7 = 20

-- Prove that the average of a, b, c, and d is 30.25 given the above condition
theorem average_abcd (a b c d : ℝ) (h : average_condition a b c d) : 
  (a + b + c + d) / 4 = 30.25 :=
by
  sorry

end average_abcd_l35_3588


namespace squirrels_in_tree_l35_3537

theorem squirrels_in_tree (N S : ℕ) (h₁ : N = 2) (h₂ : S - N = 2) : S = 4 :=
by
  sorry

end squirrels_in_tree_l35_3537


namespace faster_speed_l35_3519

theorem faster_speed (v : ℝ) (h1 : ∀ (t : ℝ), t = 50 / 10) (h2 : ∀ (d : ℝ), d = 50 + 20) (h3 : ∀ (t : ℝ), t = 70 / v) : v = 14 :=
by
  sorry

end faster_speed_l35_3519


namespace find_fathers_age_l35_3525

noncomputable def sebastian_age : ℕ := 40
noncomputable def age_difference : ℕ := 10
noncomputable def sum_ages_five_years_ago_ratio : ℚ := (3 : ℚ) / 4

theorem find_fathers_age 
  (sebastian_age : ℕ) 
  (age_difference : ℕ) 
  (sum_ages_five_years_ago_ratio : ℚ) 
  (h1 : sebastian_age = 40) 
  (h2 : age_difference = 10) 
  (h3 : sum_ages_five_years_ago_ratio = 3 / 4) 
: ∃ father_age : ℕ, father_age = 85 :=
sorry

end find_fathers_age_l35_3525


namespace person_age_l35_3551

variable (x : ℕ) -- Define the variable for age

-- State the condition as a hypothesis
def condition (x : ℕ) : Prop :=
  3 * (x + 3) - 3 * (x - 3) = x

-- State the theorem to be proved
theorem person_age (x : ℕ) (h : condition x) : x = 18 := 
sorry

end person_age_l35_3551


namespace expand_and_simplify_expression_l35_3540

variable {x y : ℝ} {i : ℂ}

-- Declare i as the imaginary unit satisfying i^2 = -1
axiom imaginary_unit : i^2 = -1

theorem expand_and_simplify_expression :
  (x + 3 + i * y) * (x + 3 - i * y) + (x - 2 + 2 * i * y) * (x - 2 - 2 * i * y)
  = 2 * x^2 + 2 * x + 13 - 5 * y^2 :=
by
  sorry

end expand_and_simplify_expression_l35_3540


namespace find_k_l35_3556

theorem find_k (k : ℝ) (h : ∀ x y : ℝ, (x, y) = (-2, -1) → y = k * x + 2) : k = 3 / 2 :=
sorry

end find_k_l35_3556


namespace part1_average_decrease_rate_part2_unit_price_reduction_l35_3563

-- Part 1: Prove the average decrease rate is 10%
theorem part1_average_decrease_rate (p0 p2 : ℝ) (x : ℝ) 
    (h1 : p0 = 200) 
    (h2 : p2 = 162) 
    (hx : (1 - x)^2 = p2 / p0) : x = 0.1 :=
by {
    sorry
}

-- Part 2: Prove the unit price reduction should be 15 yuan
theorem part2_unit_price_reduction (p_sell p_factory profit : ℝ) (n_initial dn m : ℝ)
    (h3 : p_sell = 200)
    (h4 : p_factory = 162)
    (h5 : n_initial = 20)
    (h6 : dn = 10)
    (h7 : profit = 1150)
    (hx : (38 - m) * (n_initial + 2 * m) = profit) : m = 15 :=
by {
    sorry
}

end part1_average_decrease_rate_part2_unit_price_reduction_l35_3563


namespace find_varphi_l35_3508

theorem find_varphi (φ : ℝ) (h1 : 0 < φ ∧ φ < 2 * Real.pi) 
    (h2 : ∀ x, x = 2 → Real.sin (Real.pi * x + φ) = 1) : 
    φ = Real.pi / 2 :=
-- The following is left as a proof placeholder
sorry

end find_varphi_l35_3508


namespace sum_first_seven_terms_geometric_seq_l35_3581

theorem sum_first_seven_terms_geometric_seq :
  let a := (1 : ℝ) / 3
  let r := (1 : ℝ) / 2
  let S_7 := a * (1 - r^7) / (1 - r)
  S_7 = 127 / 192 := 
by
  let a := (1 : ℝ) / 3
  let r := (1 : ℝ) / 2
  let S_7 := a * (1 - r^7) / (1 - r)
  have h : S_7 = 127 / 192 := sorry
  exact h

end sum_first_seven_terms_geometric_seq_l35_3581


namespace excluded_students_count_l35_3506

theorem excluded_students_count 
  (N : ℕ) 
  (x : ℕ) 
  (average_marks : ℕ) 
  (excluded_average_marks : ℕ) 
  (remaining_average_marks : ℕ) 
  (total_students : ℕ)
  (h1 : average_marks = 80)
  (h2 : excluded_average_marks = 70)
  (h3 : remaining_average_marks = 90)
  (h4 : total_students = 10)
  (h5 : N = total_students)
  (h6 : 80 * N = 70 * x + 90 * (N - x))
  : x = 5 :=
by
  sorry

end excluded_students_count_l35_3506


namespace eval_expression_l35_3564

theorem eval_expression : (3 + 1) * (3^2 + 1^2) * (3^4 + 1^4) = 3280 := by
  sorry

end eval_expression_l35_3564


namespace ratio_of_square_areas_l35_3523

noncomputable def ratio_of_areas (s : ℝ) : ℝ := s^2 / (4 * s^2)

theorem ratio_of_square_areas (s : ℝ) (h : s ≠ 0) : ratio_of_areas s = 1 / 4 := 
by
  sorry

end ratio_of_square_areas_l35_3523


namespace a5_a6_values_b_n_general_formula_minimum_value_T_n_l35_3515

section sequence_problems

def sequence_n (n : ℕ) : ℤ :=
if n = 0 then 1
else if n = 1 then 1
else sequence_n (n - 2) + 2 * (-1)^(n - 2)

def b_sequence (n : ℕ) : ℤ :=
sequence_n (2 * n)

def S_n (n : ℕ) : ℤ :=
(n + 1) * (sequence_n n)

def T_n (n : ℕ) : ℤ :=
(S_n (2 * n) - 18)

theorem a5_a6_values :
  sequence_n 4 = -3 ∧ sequence_n 5 = 5 := by
  sorry

theorem b_n_general_formula (n : ℕ) :
  b_sequence n = 2 * n - 1 := by
  sorry

theorem minimum_value_T_n :
  ∃ n, T_n n = -72 := by
  sorry

end sequence_problems

end a5_a6_values_b_n_general_formula_minimum_value_T_n_l35_3515


namespace no_such_abc_exists_l35_3531

-- Define the conditions for the leading coefficients and constant terms
def leading_coeff_conditions (a b c : ℝ) : Prop :=
  ((a > 0 ∧ b > 0 ∧ c < 0) ∨ (a > 0 ∧ c > 0 ∧ b < 0) ∨ (b > 0 ∧ c > 0 ∧ a < 0))

def constant_term_conditions (a b c : ℝ) : Prop :=
  ((c > 0 ∧ a < 0 ∧ b < 0) ∨ (a > 0 ∧ b < 0 ∧ c < 0) ∨ (b > 0 ∧ c < 0 ∧ a < 0))

-- The final statement that encapsulates the contradiction
theorem no_such_abc_exists : ¬ ∃ a b c : ℝ, leading_coeff_conditions a b c ∧ constant_term_conditions a b c :=
by
  sorry

end no_such_abc_exists_l35_3531


namespace smallest_integer_of_lcm_gcd_l35_3567

theorem smallest_integer_of_lcm_gcd (m : ℕ) (h1 : m > 0) (h2 : Nat.lcm 60 m / Nat.gcd 60 m = 44) : m = 165 :=
sorry

end smallest_integer_of_lcm_gcd_l35_3567


namespace correct_inequality_incorrect_inequality1_incorrect_inequality2_correct_option_d_l35_3517

theorem correct_inequality:
    (-21 : ℤ) > (-21 : ℤ) := by sorry

theorem incorrect_inequality1 :
    -abs (10 + 1 / 2) < (8 + 2 / 3) := by sorry

theorem incorrect_inequality2 :
    (-abs (7 + 2 / 3)) ≠ (- (- (7 + 2 / 3))) := by sorry

theorem correct_option_d :
    (-5 / 6 : ℚ) < (-4 / 5 : ℚ) := by sorry

end correct_inequality_incorrect_inequality1_incorrect_inequality2_correct_option_d_l35_3517


namespace isosceles_triangle_base_length_l35_3514

theorem isosceles_triangle_base_length (P Q : ℕ) (x y : ℕ) (hP : P = 15) (hQ : Q = 12) (hPerimeter : 2 * x + y = 27) 
      (hCondition : (y = P ∧ (1 / 2) * x + x = P) ∨ (y = Q ∧ (1 / 2) * x + x = Q)) : 
  y = 7 ∨ y = 11 :=
sorry

end isosceles_triangle_base_length_l35_3514


namespace hyperbola_foci_l35_3561

/-- The coordinates of the foci of the hyperbola y^2 / 3 - x^2 = 1 are (0, ±2). -/
theorem hyperbola_foci (x y : ℝ) :
  x^2 - (y^2 / 3) = -1 → (0 = x ∧ (y = 2 ∨ y = -2)) :=
sorry

end hyperbola_foci_l35_3561


namespace waiting_probability_no_more_than_10_seconds_l35_3559

def total_cycle_time : ℕ := 30 + 10 + 40
def proceed_during_time : ℕ := 40 -- green time
def yellow_time : ℕ := 10

theorem waiting_probability_no_more_than_10_seconds :
  (proceed_during_time + yellow_time + yellow_time) / total_cycle_time = 3 / 4 := by
  sorry

end waiting_probability_no_more_than_10_seconds_l35_3559


namespace hermia_elected_probability_l35_3533

-- Define the problem statement and conditions in Lean 4
noncomputable def probability_hermia_elected (n : ℕ) (h_odd : (n % 2 = 1)) (h_pos : n > 0) : ℚ :=
  if n = 1 then 1 else (2^n - 1) / (n * 2^(n-1))

-- Lean theorem statement
theorem hermia_elected_probability (n : ℕ) (h_odd : (n % 2 = 1)) (h_pos : n > 0) : 
  probability_hermia_elected n h_odd h_pos = (2^n - 1) / (n * 2^(n-1)) :=
by
  sorry

end hermia_elected_probability_l35_3533


namespace binary_addition_is_correct_l35_3532

-- Definitions for the binary numbers
def bin1 := "10101"
def bin2 := "11"
def bin3 := "1010"
def bin4 := "11100"
def bin5 := "1101"

-- Function to convert binary string to nat (using built-in functionality)
def binStringToNat (s : String) : Nat :=
  String.foldl (fun n c => 2 * n + if c = '1' then 1 else 0) 0 s

-- Binary numbers converted to nat
def n1 := binStringToNat bin1
def n2 := binStringToNat bin2
def n3 := binStringToNat bin3
def n4 := binStringToNat bin4
def n5 := binStringToNat bin5

-- The expected result in nat
def expectedSum := binStringToNat "11101101"

-- Proof statement
theorem binary_addition_is_correct : n1 + n2 + n3 + n4 + n5 = expectedSum :=
  sorry

end binary_addition_is_correct_l35_3532


namespace towers_per_castle_jeff_is_5_l35_3541

-- Define the number of sandcastles on Mark's beach
def num_castles_mark : ℕ := 20

-- Define the number of towers per sandcastle on Mark's beach
def towers_per_castle_mark : ℕ := 10

-- Calculate the total number of towers on Mark's beach
def total_towers_mark : ℕ := num_castles_mark * towers_per_castle_mark

-- Define the number of sandcastles on Jeff's beach (3 times that of Mark's)
def num_castles_jeff : ℕ := 3 * num_castles_mark

-- Define the total number of sandcastles on both beaches
def total_sandcastles : ℕ := num_castles_mark + num_castles_jeff
  
-- Define the combined total number of sandcastles and towers on both beaches
def combined_total : ℕ := 580

-- Define the number of towers per sandcastle on Jeff's beach
def towers_per_castle_jeff : ℕ := sorry

-- Define the total number of towers on Jeff's beach
def total_towers_jeff (T : ℕ) : ℕ := num_castles_jeff * T

-- Prove that the number of towers per sandcastle on Jeff's beach is 5
theorem towers_per_castle_jeff_is_5 : 
    200 + total_sandcastles + total_towers_jeff towers_per_castle_jeff = combined_total → 
    towers_per_castle_jeff = 5
:= by
    sorry

end towers_per_castle_jeff_is_5_l35_3541


namespace max_t_subsets_of_base_set_l35_3502

theorem max_t_subsets_of_base_set (n : ℕ)
  (A : Fin (2 * n + 1) → Set (Fin n))
  (h : ∀ i j k : Fin (2 * n + 1), i < j → j < k → (A i ∩ A k) ⊆ A j) : 
  ∃ t : ℕ, t = 2 * n + 1 :=
by
  sorry

end max_t_subsets_of_base_set_l35_3502


namespace find_english_marks_l35_3500

variable (mathematics science social_studies english biology : ℕ)
variable (average_marks : ℕ)
variable (number_of_subjects : ℕ := 5)

-- Conditions
axiom score_math : mathematics = 76
axiom score_sci : science = 65
axiom score_ss : social_studies = 82
axiom score_bio : biology = 95
axiom average : average_marks = 77

-- The proof problem
theorem find_english_marks :
  english = 67 :=
  sorry

end find_english_marks_l35_3500


namespace first_sales_amount_l35_3547

-- Conditions from the problem
def first_sales_royalty : ℝ := 8 -- million dollars
def second_sales_royalty : ℝ := 9 -- million dollars
def second_sales_amount : ℝ := 108 -- million dollars
def decrease_percentage : ℝ := 0.7916666666666667

-- The goal is to determine the first sales amount, S, meeting the conditions.
theorem first_sales_amount :
  ∃ S : ℝ,
    (first_sales_royalty / S - second_sales_royalty / second_sales_amount = decrease_percentage * (first_sales_royalty / S)) ∧
    S = 20 :=
sorry

end first_sales_amount_l35_3547


namespace triangle_area_x_value_l35_3529

theorem triangle_area_x_value :
  ∃ x : ℝ, x > 0 ∧ 100 = (1 / 2) * x * (3 * x) ∧ x = 10 * Real.sqrt 6 / 3 :=
sorry

end triangle_area_x_value_l35_3529


namespace tangent_line_at_point_l35_3592

theorem tangent_line_at_point
  (x y : ℝ)
  (h_curve : y = x^3 - 3 * x^2 + 1)
  (h_point : (x, y) = (1, -1)) :
  ∃ m b : ℝ, (m = -3) ∧ (b = 2) ∧ (y = m * x + b) :=
sorry

end tangent_line_at_point_l35_3592


namespace dinosaur_dolls_distribution_l35_3580

-- Defining the conditions
def num_dolls : ℕ := 5
def num_friends : ℕ := 2

-- Lean theorem statement
theorem dinosaur_dolls_distribution :
  (num_dolls * (num_dolls - 1) = 20) :=
by
  -- Sorry placeholder for the proof
  sorry

end dinosaur_dolls_distribution_l35_3580


namespace select_k_numbers_l35_3558

theorem select_k_numbers (a : ℕ → ℝ) (k : ℕ) (h1 : ∀ n, 0 < a n) 
  (h2 : ∀ n m, n < m → a n ≥ a m) (h3 : a 1 = 1 / (2 * k)) 
  (h4 : ∑' n, a n = 1) :
  ∃ (f : ℕ → ℕ) (hf : ∀ i j, i ≠ j → f i ≠ f j), 
    (∀ i, i < k → a (f i) > 1/2 * a (f 0)) :=
by
  sorry

end select_k_numbers_l35_3558


namespace probability_of_blue_buttons_l35_3590

theorem probability_of_blue_buttons
  (orig_red_A : ℕ) (orig_blue_A : ℕ)
  (removed_red : ℕ) (removed_blue : ℕ)
  (target_ratio : ℚ)
  (final_red_A : ℕ) (final_blue_A : ℕ)
  (final_red_B : ℕ) (final_blue_B : ℕ)
  (orig_buttons_A : orig_red_A + orig_blue_A = 16)
  (removed_buttons : removed_red = 3 ∧ removed_blue = 5)
  (final_buttons_A : final_red_A + final_blue_A = 8)
  (buttons_ratio : target_ratio = 2 / 3)
  (final_ratio_A : final_red_A + final_blue_A = target_ratio * 16)
  (red_in_A : final_red_A = orig_red_A - removed_red)
  (blue_in_A : final_blue_A = orig_blue_A - removed_blue)
  (red_in_B : final_red_B = removed_red)
  (blue_in_B : final_blue_B = removed_blue):
  (final_blue_A / (final_red_A + final_blue_A)) * (final_blue_B / (final_red_B + final_blue_B)) = 25 / 64 := 
by
  sorry

end probability_of_blue_buttons_l35_3590


namespace geom_seq_sum_5_terms_l35_3573

theorem geom_seq_sum_5_terms (a : ℕ → ℝ) (q : ℝ) (h1 : a 4 = 8 * a 1) (h2 : 2 * (a 2 + 1) = a 1 + a 3) (h_q : q = 2) :
    a 1 * (1 - q^5) / (1 - q) = 62 :=
by
    sorry

end geom_seq_sum_5_terms_l35_3573


namespace segment_length_is_ten_l35_3521

-- Definition of the cube root function and the absolute value
def cube_root (x : ℝ) : ℝ := x^(1/3)

def absolute (x : ℝ) : ℝ := abs x

-- The prerequisites as conditions for the endpoints
def endpoints_satisfy (x : ℝ) : Prop := absolute (x - cube_root 27) = 5

-- Length of the segment determined by the endpoints
def segment_length (x1 x2 : ℝ) : ℝ := absolute (x2 - x1)

-- Theorem statement
theorem segment_length_is_ten : (∀ x, endpoints_satisfy x) → segment_length (-2) 8 = 10 :=
by
  intro h
  sorry

end segment_length_is_ten_l35_3521


namespace graph_not_in_third_quadrant_l35_3577

-- Define the conditions
variable (m : ℝ)
variable (h1 : 0 < m)
variable (h2 : m < 2)

-- Define the graph equation
noncomputable def line_eq (x : ℝ) : ℝ := (m - 2) * x + m

-- The proof problem: the graph does not pass through the third quadrant
theorem graph_not_in_third_quadrant : ¬ ∃ x y : ℝ, (x < 0 ∧ y < 0 ∧ y = (m - 2) * x + m) :=
sorry

end graph_not_in_third_quadrant_l35_3577


namespace range_of_f_l35_3522

noncomputable def f (x : ℝ) : ℝ := (3 * x + 8) / (x - 4)

theorem range_of_f : Set.range f = {y : ℝ | y ≠ 3} :=
sorry

end range_of_f_l35_3522


namespace equilibrium_and_stability_l35_3512

def system_in_equilibrium (G Q m r : ℝ) : Prop :=
    -- Stability conditions for points A and B, instability at C
    (G < (m-r)/(m-2*r)) ∧ (G > (m-r)/m)

-- Create a theorem to prove the system's equilibrium and stability
theorem equilibrium_and_stability (G Q m r : ℝ) 
  (h_gt_zero : G > 0) 
  (Q_gt_zero : Q > 0) 
  (m_gt_r : m > r) 
  (r_gt_zero : r > 0) : system_in_equilibrium G Q m r :=
by
  sorry   -- Proof omitted

end equilibrium_and_stability_l35_3512


namespace find_y_when_x_is_4_l35_3575

variables (x y : ℕ)
def inversely_proportional (C : ℕ) (x y : ℕ) : Prop := x * y = C

theorem find_y_when_x_is_4 :
  inversely_proportional 240 x y → x = 4 → y = 60 :=
by
  sorry

end find_y_when_x_is_4_l35_3575


namespace find_number_l35_3553

theorem find_number (x : ℕ) (h : 3 * x = 2 * 51 - 3) : x = 33 :=
by
  sorry

end find_number_l35_3553


namespace find_xyz_l35_3574

variable (x y z : ℝ)
variable (h1 : x = 80 + 0.11 * 80)
variable (h2 : y = 120 - 0.15 * 120)
variable (h3 : z = 0.20 * (0.40 * (x + y)) + 0.40 * (x + y))

theorem find_xyz (hx : x = 88.8) (hy : y = 102) (hz : z = 91.584) : 
  x = 88.8 ∧ y = 102 ∧ z = 91.584 := by
  sorry

end find_xyz_l35_3574


namespace range_of_a_l35_3504

theorem range_of_a (a : ℝ) : 
  (∀ (x1 : ℝ), ∃ (x2 : ℝ), |x1| = Real.log (a * x2^2 - 4 * x2 + 1)) → (0 ≤ a) :=
by
  sorry

end range_of_a_l35_3504


namespace max_value_fourth_power_l35_3526

theorem max_value_fourth_power (a b c d : ℝ) (h : a^3 + b^3 + c^3 + d^3 = 4) : 
  a^4 + b^4 + c^4 + d^4 ≤ 4^(4/3) :=
sorry

end max_value_fourth_power_l35_3526


namespace largest_gcd_sum780_l35_3527

theorem largest_gcd_sum780 (a b : ℕ) (h_pos_a : 0 < a) (h_pos_b : 0 < b) (h_sum : a + b = 780) : 
  ∃ d, d = Nat.gcd a b ∧ d ≤ 390 ∧ (∀ (d' : ℕ), d' = Nat.gcd a b → d' ≤ 390) :=
sorry

end largest_gcd_sum780_l35_3527


namespace steve_can_answer_38_questions_l35_3548

theorem steve_can_answer_38_questions (total_questions S : ℕ) 
  (h1 : total_questions = 45)
  (h2 : total_questions - S = 7) :
  S = 38 :=
by {
  -- The proof goes here
  sorry
}

end steve_can_answer_38_questions_l35_3548


namespace vector_subtraction_result_l35_3545

def a : ℝ × ℝ := (2, 1)
def b : ℝ × ℝ := (-3, 4)

theorem vector_subtraction_result :
  2 • a - b = (7, -2) :=
by
  simp [a, b]
  sorry

end vector_subtraction_result_l35_3545


namespace solve_system_of_equations_solve_linear_inequality_l35_3562

-- Part 1: System of equations
theorem solve_system_of_equations (x y : ℝ) (h1 : 5 * x + 2 * y = 25) (h2 : 3 * x + 4 * y = 15) : 
  x = 5 ∧ y = 0 := sorry

-- Part 2: Linear inequality
theorem solve_linear_inequality (x : ℝ) (h : 2 * x - 6 < 3 * x) : 
  x > -6 := sorry

end solve_system_of_equations_solve_linear_inequality_l35_3562


namespace track_team_children_l35_3571

/-- There were initially 18 girls and 15 boys on the track team.
    7 more girls joined the team, and 4 boys quit the team.
    The proof shows that the total number of children on the track team after the changes is 36. -/
theorem track_team_children (initial_girls initial_boys girls_joined boys_quit : ℕ)
  (h_initial_girls : initial_girls = 18)
  (h_initial_boys : initial_boys = 15)
  (h_girls_joined : girls_joined = 7)
  (h_boys_quit : boys_quit = 4) :
  initial_girls + girls_joined - boys_quit + initial_boys = 36 :=
by
  -- Placeholder to indicate the proof is omitted
  sorry

end track_team_children_l35_3571


namespace matt_paper_piles_l35_3509

theorem matt_paper_piles (n : ℕ) (h_n1 : 1000 < n) (h_n2 : n < 2000)
  (h2 : n % 2 = 1) (h3 : n % 3 = 1) (h4 : n % 4 = 1)
  (h5 : n % 5 = 1) (h6 : n % 6 = 1) (h7 : n % 7 = 1)
  (h8 : n % 8 = 1) : 
  ∃ k : ℕ, k ≠ 1 ∧ k ≠ n ∧ n = 1681 ∧ k = 41 :=
by
  use 41
  sorry

end matt_paper_piles_l35_3509


namespace grasshopper_flea_adjacency_l35_3528

-- We assume that grid cells are indexed by pairs of integers (i.e., positions in ℤ × ℤ)
-- Red cells and white cells are represented as sets of these positions
variable (red_cells : Set (ℤ × ℤ))
variable (white_cells : Set (ℤ × ℤ))

-- We define that the grasshopper can only jump between red cells
def grasshopper_jump (pos : ℤ × ℤ) (new_pos : ℤ × ℤ) : Prop :=
  pos ∈ red_cells ∧ new_pos ∈ red_cells ∧ (pos.1 = new_pos.1 ∨ pos.2 = new_pos.2)

-- We define that the flea can only jump between white cells
def flea_jump (pos : ℤ × ℤ) (new_pos : ℤ × ℤ) : Prop :=
  pos ∈ white_cells ∧ new_pos ∈ white_cells ∧ (pos.1 = new_pos.1 ∨ pos.2 = new_pos.2)

-- Main theorem to be proved
theorem grasshopper_flea_adjacency (g_start : ℤ × ℤ) (f_start : ℤ × ℤ) :
    g_start ∈ red_cells → f_start ∈ white_cells →
    ∃ g1 g2 g3 f1 f2 f3 : ℤ × ℤ,
    (
      grasshopper_jump red_cells g_start g1 ∧
      grasshopper_jump red_cells g1 g2 ∧
      grasshopper_jump red_cells g2 g3
    ) ∧ (
      flea_jump white_cells f_start f1 ∧
      flea_jump white_cells f1 f2 ∧
      flea_jump white_cells f2 f3
    ) ∧
    (abs (g3.1 - f3.1) + abs (g3.2 - f3.2) = 1) :=
  sorry

end grasshopper_flea_adjacency_l35_3528


namespace action_movies_rented_l35_3513

-- Defining the conditions as hypotheses
theorem action_movies_rented (a M A D : ℝ) (h1 : 0.64 * M = 10 * a)
                             (h2 : D = 5 * A)
                             (h3 : D + A = 0.36 * M) :
    A = 0.9375 * a :=
sorry

end action_movies_rented_l35_3513


namespace simon_can_make_blueberry_pies_l35_3555

theorem simon_can_make_blueberry_pies (bush1 bush2 blueberries_per_pie : ℕ) (h1 : bush1 = 100) (h2 : bush2 = 200) (h3 : blueberries_per_pie = 100) : 
  (bush1 + bush2) / blueberries_per_pie = 3 :=
by
  -- Proof goes here
  sorry

end simon_can_make_blueberry_pies_l35_3555


namespace ratio_between_second_and_third_l35_3569

noncomputable def ratio_second_third : ℚ := sorry

theorem ratio_between_second_and_third (A B C : ℕ) (h₁ : A + B + C = 98) (h₂ : A * 3 = B * 2) (h₃ : B = 30) :
  ratio_second_third = 5 / 8 := sorry

end ratio_between_second_and_third_l35_3569


namespace minjun_current_height_l35_3596

variable (initial_height : ℝ) (growth_last_year : ℝ) (growth_this_year : ℝ)

theorem minjun_current_height
  (h_initial : initial_height = 1.1)
  (h_growth_last_year : growth_last_year = 0.2)
  (h_growth_this_year : growth_this_year = 0.1) :
  initial_height + growth_last_year + growth_this_year = 1.4 :=
by
  sorry

end minjun_current_height_l35_3596


namespace log_properties_l35_3546

theorem log_properties :
  (Real.log 5) ^ 2 + (Real.log 2) * (Real.log 50) = 1 :=
by sorry

end log_properties_l35_3546


namespace true_statement_l35_3591

variables {Plane Line : Type}
variables (α β γ : Plane) (a b m n : Line)

-- Definitions for parallel and perpendicular relationships
def parallel (x y : Line) : Prop := sorry
def perpendicular (x y : Line) : Prop := sorry
def subset (l : Line) (p : Plane) : Prop := sorry
def intersect_line (p q : Plane) : Line := sorry

-- Given conditions for the problem
variables (h1 : (α ≠ β)) (h2 : (parallel α β))
variables (h3 : (intersect_line α γ = a)) (h4 : (intersect_line β γ = b))

-- Statement verifying the true condition based on the above givens
theorem true_statement : parallel a b :=
by sorry

end true_statement_l35_3591


namespace concyclic_iff_l35_3535

variables {A B C H O' N D : Type*}
variables [MetricSpace A] [MetricSpace B] [MetricSpace C] [MetricSpace H]
variables [MetricSpace O'] [MetricSpace N] [MetricSpace D]
variables (a b c R : ℝ)

-- Conditions from the problem
def is_orthocenter (H : Type*) (A B C : Type*) : Prop :=
  -- definition of orthocenter using suitable predicates (omitted for brevity) 
  sorry

def is_circumcenter (O' : Type*) (B H C : Type*) : Prop :=
  -- definition of circumcenter using suitable predicates (omitted for brevity) 
  sorry

def is_midpoint (N : Type*) (A O' : Type*) : Prop :=
  -- definition of midpoint using suitable predicates (omitted for brevity) 
  sorry

def is_reflection (N D : Type*) (B C : Type*) : Prop :=
  -- definition of reflection about the side BC (omitted for brevity) 
  sorry

-- Definition that points A, B, C, D are concyclic
def are_concyclic (A B C D : Type*) : Prop :=
  -- definition using suitable predicates (omitted for brevity)
  sorry

-- Main theorem statement
theorem concyclic_iff (h1 : is_orthocenter H A B C) (h2 : is_circumcenter O' B H C) 
                      (h3 : is_midpoint N A O') (h4 : is_reflection N D B C)
                      (ha : a = 1) (hb : b = 1) (hc : c = 1) (hR : R = 1) :
  are_concyclic A B C D ↔ b^2 + c^2 - a^2 = 3 * R^2 := 
sorry

end concyclic_iff_l35_3535


namespace solve_for_x_l35_3520

theorem solve_for_x : ∀ x : ℤ, 5 - x = 8 → x = -3 :=
by
  intros x h
  sorry

end solve_for_x_l35_3520


namespace equation_of_l_l35_3579

-- Defining the equations of the circles
def circle_O (x y : ℝ) := x^2 + y^2 = 4
def circle_C (x y : ℝ) := x^2 + y^2 + 4 * x - 4 * y + 4 = 0

-- Assuming the line l makes circles O and C symmetric
def symmetric (l : ℝ → ℝ → Prop) := ∀ (x y : ℝ), l x y → 
  (∃ (x' y' : ℝ), circle_O x y ∧ circle_C x' y' ∧ (x + x') / 2 = x' ∧ (y + y') / 2 = y')

-- Stating the theorem to be proven
theorem equation_of_l :
  ∀ l : ℝ → ℝ → Prop, symmetric l → (∀ x y : ℝ, l x y ↔ x - y + 2 = 0) :=
by
  sorry

end equation_of_l_l35_3579


namespace pyramid_volume_l35_3536

noncomputable def volume_of_pyramid (AB AD BD AE : ℝ) (p : AB = 9 ∧ AD = 10 ∧ BD = 11 ∧ AE = 10.5) : ℝ :=
  1 / 3 * (60 * (2 ^ (1 / 2))) * (5 * (2 ^ (1 / 2)))

theorem pyramid_volume (AB AD BD AE : ℝ) (h1 : AB = 9) (h2 : AD = 10) (h3 : BD = 11) (h4 : AE = 10.5)
  (V : ℝ) (hV : V = 200) : 
  volume_of_pyramid AB AD BD AE (⟨h1, ⟨h2, ⟨h3, h4⟩⟩⟩) = V :=
sorry

end pyramid_volume_l35_3536


namespace quadratic_no_real_roots_l35_3566

-- Define the quadratic equation
def quadratic_eq (a b c x : ℝ) : ℝ := a * x^2 + b * x + c

-- Define the discriminant of a quadratic equation
def discriminant (a b c : ℝ) : ℝ := b^2 - 4 * a * c

-- Conditions of the problem
def a : ℝ := 3
def b : ℝ := -6
def c : ℝ := 4

-- The proof statement
theorem quadratic_no_real_roots : discriminant a b c < 0 :=
by
  -- Calculate the discriminant to show it's negative
  let Δ := discriminant a b c
  show Δ < 0
  sorry

end quadratic_no_real_roots_l35_3566


namespace vector_ab_l35_3598

theorem vector_ab
  (A B : ℝ × ℝ)
  (hA : A = (1, -1))
  (hB : B = (1, 2)) :
  (B.1 - A.1, B.2 - A.2) = (0, 3) :=
by
  sorry

end vector_ab_l35_3598


namespace michael_ratio_zero_l35_3576

theorem michael_ratio_zero (M : ℕ) (h1: M ≤ 60) (h2: 15 = (60 - M) / 2 - 15) : M = 0 := by
  sorry 

end michael_ratio_zero_l35_3576


namespace time_difference_l35_3568

noncomputable def hour_angle (n : ℝ) : ℝ :=
  150 + (n / 2)

noncomputable def minute_angle (n : ℝ) : ℝ :=
  6 * n

theorem time_difference (n1 n2 : ℝ)
  (h1 : |(hour_angle n1) - (minute_angle n1)| = 120)
  (h2 : |(hour_angle n2) - (minute_angle n2)| = 120) :
  n2 - n1 = 43.64 := 
sorry

end time_difference_l35_3568


namespace op_example_l35_3538

variables {α β : ℚ}

def op (α β : ℚ) := α * β + 1

theorem op_example : op 2 (-3) = -5 :=
by
  -- The proof is omitted as requested
  sorry

end op_example_l35_3538


namespace slope_of_line_l35_3534

theorem slope_of_line : ∀ (x y : ℝ), 2 * x - 4 * y + 7 = 0 → (y = (1/2) * x - 7 / 4) :=
by
  intro x y h
  -- This would typically involve rearranging the given equation to the slope-intercept form
  -- but as we are focusing on creating the statement, we insert sorry to skip the proof
  sorry

end slope_of_line_l35_3534


namespace abs_eq_non_pos_2x_plus_4_l35_3586

-- Condition: |2x + 4| = 0
-- Conclusion: x = -2
theorem abs_eq_non_pos_2x_plus_4 (x : ℝ) : (|2 * x + 4| = 0) → x = -2 :=
by
  intro h
  -- Here lies the proof, but we use sorry to indicate the unchecked part.
  sorry

end abs_eq_non_pos_2x_plus_4_l35_3586


namespace shoot_down_probability_l35_3585

-- Define the probabilities
def P_hit_nose := 0.2
def P_hit_middle := 0.4
def P_hit_tail := 0.1
def P_miss := 0.3

-- Define the condition: probability of shooting down the plane with at most 2 shots
def condition := (P_hit_tail + (P_hit_nose * P_hit_nose) + (P_miss * P_hit_tail))

-- Proving the probability matches the required value
theorem shoot_down_probability : condition = 0.23 :=
by
  sorry

end shoot_down_probability_l35_3585


namespace tan_sum_pi_div_4_sin_fraction_simplifies_to_1_l35_3560

variable (α : ℝ)
variable (π : ℝ) [Fact (π > 0)]

-- Assume condition
axiom tan_alpha_eq_2 : Real.tan α = 2

-- Goal (1): Prove that tan(α + π/4) = -3
theorem tan_sum_pi_div_4 : Real.tan (α + π / 4) = -3 :=
by
  sorry

-- Goal (2): Prove that (sin(2α) / (sin^2(α) + sin(α) * cos(α) - cos(2α) - 1)) = 1
theorem sin_fraction_simplifies_to_1 :
  (Real.sin (2 * α)) / (Real.sin (α)^2 + Real.sin (α) * Real.cos (α) - Real.cos (2 * α) - 1) = 1 :=
by
  sorry

end tan_sum_pi_div_4_sin_fraction_simplifies_to_1_l35_3560


namespace units_digit_G_1000_l35_3550

def modified_fermat_number (n : ℕ) : ℕ := 5^(5^n) + 6

theorem units_digit_G_1000 : (modified_fermat_number 1000) % 10 = 1 :=
by
  -- The proof goes here
  sorry

end units_digit_G_1000_l35_3550


namespace find_m_l35_3539

noncomputable def slope_at_one (m : ℝ) := 2 + m

noncomputable def tangent_line_eq (m : ℝ) (x : ℝ) := (slope_at_one m) * x - 2 * m

noncomputable def y_intercept (m : ℝ) := tangent_line_eq m 0

noncomputable def x_intercept (m : ℝ) := - (y_intercept m) / (slope_at_one m)

noncomputable def intercept_sum_eq (m : ℝ) := (x_intercept m) + (y_intercept m)

theorem find_m (m : ℝ) (h : m ≠ -2) (h2 : intercept_sum_eq m = 12) : m = -3 ∨ m = -4 := 
sorry

end find_m_l35_3539


namespace derivative_correct_l35_3584

noncomputable def derivative_of_composite_function (x : ℝ) : Prop :=
  let y := (5 * x - 3) ^ 3
  let dy_dx := 3 * (5 * x - 3) ^ 2 * 5
  dy_dx = 15 * (5 * x - 3) ^ 2

theorem derivative_correct (x : ℝ) : derivative_of_composite_function x :=
by
  sorry

end derivative_correct_l35_3584


namespace solution_set_of_abs_x_plus_one_gt_one_l35_3572

theorem solution_set_of_abs_x_plus_one_gt_one :
  {x : ℝ | |x + 1| > 1} = {x : ℝ | x < -2 ∨ x > 0} :=
sorry

end solution_set_of_abs_x_plus_one_gt_one_l35_3572


namespace values_of_a_and_b_intervals_of_monotonicity_range_of_a_for_three_roots_l35_3570

noncomputable def f (x : ℝ) (a : ℝ) (b : ℝ) : ℝ := x^3 - 3 * a * x^2 + 2 * b * x

theorem values_of_a_and_b (h : ∀ x, f x (1 / 3) (-1 / 2) ≤ f 1 (1 / 3) (-1 / 2)) :
  (∃ a b, a = 1 / 3 ∧ b = -1 / 2) :=
sorry

theorem intervals_of_monotonicity (a b : ℝ) (h : ∀ x, f x a b ≤ f 1 a b) :
  (∀ x, (f x a b ≥ 0 ↔ x ≤ -1 / 3 ∨ x ≥ 1) ∧ (f x a b ≤ 0 ↔ -1 / 3 ≤ x ∧ x ≤ 1)) :=
sorry

theorem range_of_a_for_three_roots :
  (∃ a, -1 < a ∧ a < 5 / 27) :=
sorry

end values_of_a_and_b_intervals_of_monotonicity_range_of_a_for_three_roots_l35_3570


namespace temperature_on_friday_is_35_l35_3543

variables (M T W Th F : ℤ)

def is_odd (n : ℤ) : Prop := ∃ k : ℤ, n = 2 * k + 1

theorem temperature_on_friday_is_35
  (h1 : (M + T + W + Th) / 4 = 48)
  (h2 : (T + W + Th + F) / 4 = 46)
  (h3 : M = 43)
  (h4 : is_odd M)
  (h5 : is_odd T)
  (h6 : is_odd W)
  (h7 : is_odd Th)
  (h8 : is_odd F) : 
  F = 35 :=
sorry

end temperature_on_friday_is_35_l35_3543


namespace total_matches_played_l35_3549

theorem total_matches_played
  (avg_runs_first_20: ℕ) (num_first_20: ℕ) (avg_runs_next_10: ℕ) (num_next_10: ℕ) (overall_avg: ℕ) (total_matches: ℕ) :
  avg_runs_first_20 = 40 →
  num_first_20 = 20 →
  avg_runs_next_10 = 13 →
  num_next_10 = 10 →
  overall_avg = 31 →
  (num_first_20 + num_next_10 = total_matches) →
  total_matches = 30 :=
by
  intros h1 h2 h3 h4 h5 h6
  sorry

end total_matches_played_l35_3549


namespace multiplication_is_correct_l35_3510

theorem multiplication_is_correct : 209 * 209 = 43681 := sorry

end multiplication_is_correct_l35_3510


namespace sum_of_second_and_third_smallest_is_804_l35_3594

noncomputable def sum_of_second_and_third_smallest : Nat :=
  let digits := [1, 6, 8]
  let second_smallest := 186
  let third_smallest := 618
  second_smallest + third_smallest

theorem sum_of_second_and_third_smallest_is_804 :
  sum_of_second_and_third_smallest = 804 :=
by
  sorry

end sum_of_second_and_third_smallest_is_804_l35_3594


namespace inequality_solution_empty_solution_set_l35_3595

-- Problem 1: Prove the inequality and the solution range
theorem inequality_solution (x : ℝ) : (-7 < x ∧ x < 3) ↔ ( (x - 3)/(x + 7) < 0 ) :=
sorry

-- Problem 2: Prove the conditions for empty solution set
theorem empty_solution_set (a : ℝ) : (a > 0) ↔ ∀ x : ℝ, ¬ (x^2 - 4*a*x + 4*a^2 + a ≤ 0) :=
sorry

end inequality_solution_empty_solution_set_l35_3595


namespace distance_train_A_when_meeting_l35_3503

noncomputable def distance_traveled_by_train_A : ℝ :=
  let distance := 375
  let time_A := 36
  let time_B := 24
  let speed_A := distance / time_A
  let speed_B := distance / time_B
  let relative_speed := speed_A + speed_B
  let time_meeting := distance / relative_speed
  speed_A * time_meeting

theorem distance_train_A_when_meeting :
  distance_traveled_by_train_A = 150 := by
  sorry

end distance_train_A_when_meeting_l35_3503


namespace probability_only_one_l35_3589

-- Define the probabilities
def P_A : ℚ := 1 / 2
def P_B : ℚ := 1 / 3
def P_C : ℚ := 1 / 4

-- Define the complement probabilities
def not_P (P : ℚ) : ℚ := 1 - P
def P_not_A := not_P P_A
def P_not_B := not_P P_B
def P_not_C := not_P P_C

-- Expressions for probabilities where only one student solves the problem
def only_A_solves : ℚ := P_A * P_not_B * P_not_C
def only_B_solves : ℚ := P_B * P_not_A * P_not_C
def only_C_solves : ℚ := P_C * P_not_A * P_not_B

-- Total probability that only one student solves the problem
def P_only_one : ℚ := only_A_solves + only_B_solves + only_C_solves

-- The theorem to prove that the total probability matches
theorem probability_only_one : P_only_one = 11 / 24 := by
  sorry

end probability_only_one_l35_3589


namespace greatest_possible_perimeter_l35_3505

theorem greatest_possible_perimeter :
  ∃ (x : ℤ), x ≥ 4 ∧ x ≤ 5 ∧ (x + 4 * x + 18 = 43 ∧
    ∀ (y : ℤ), y ≥ 4 ∧ y ≤ 5 → y + 4 * y + 18 ≤ 43) :=
by
  sorry

end greatest_possible_perimeter_l35_3505


namespace javier_fraction_to_anna_zero_l35_3599

-- Variables
variable (l : ℕ) -- Lee's initial sticker count
variable (j : ℕ) -- Javier's initial sticker count
variable (a : ℕ) -- Anna's initial sticker count

-- Initial conditions
def conditions (l j a : ℕ) : Prop :=
  j = 4 * a ∧ a = 3 * l

-- Javier's final stickers count
def final_javier_stickers (ja : ℕ) (j : ℕ) : ℕ :=
  ja

-- Anna's final stickers count (af = final Anna's stickers)
def final_anna_stickers (af : ℕ) : ℕ :=
  af

-- Lee's final stickers count (lf = final Lee's stickers)
def final_lee_stickers (lf : ℕ) : ℕ :=
  lf

-- Final distribution requirements
def final_distribution (ja af lf : ℕ) : Prop :=
  ja = 2 * af ∧ ja = 3 * lf

-- Correct answer, fraction of stickers given to Anna
def fraction_given_to_anna (j ja : ℕ) : ℚ :=
  ((j - ja) : ℚ) / (j : ℚ)

-- Lean theorem statement to prove
theorem javier_fraction_to_anna_zero
  (l j a ja af lf : ℕ)
  (h_cond : conditions l j a)
  (h_final : final_distribution ja af lf) :
  fraction_given_to_anna j ja = 0 :=
by sorry

end javier_fraction_to_anna_zero_l35_3599

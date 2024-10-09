import Mathlib

namespace car_kilometers_per_gallon_l794_79436

-- Define the given conditions as assumptions
variable (total_distance : ℝ) (total_gallons : ℝ)
-- Assume the given conditions
axiom h1 : total_distance = 180
axiom h2 : total_gallons = 4.5

-- The statement to be proven
theorem car_kilometers_per_gallon : (total_distance / total_gallons) = 40 :=
by
  -- Sorry is used to skip the proof
  sorry

end car_kilometers_per_gallon_l794_79436


namespace Diana_additional_video_game_time_l794_79487

theorem Diana_additional_video_game_time 
    (original_reward_per_hour : ℕ := 30)
    (raise_percentage : ℕ := 20)
    (hours_read : ℕ := 12)
    (minutes_per_hour : ℕ := 60) :
    let raise := (raise_percentage * original_reward_per_hour) / 100
    let new_reward_per_hour := original_reward_per_hour + raise
    let total_time_after_raise := new_reward_per_hour * hours_read
    let total_time_before_raise := original_reward_per_hour * hours_read
    let additional_minutes := total_time_after_raise - total_time_before_raise
    additional_minutes = 72 :=
by sorry

end Diana_additional_video_game_time_l794_79487


namespace actual_length_correct_l794_79418

-- Definitions based on the conditions
def blueprint_scale : ℝ := 20
def measured_length_cm : ℝ := 16

-- Statement of the proof problem
theorem actual_length_correct :
  measured_length_cm * blueprint_scale = 320 := 
sorry

end actual_length_correct_l794_79418


namespace flower_pots_count_l794_79408

noncomputable def total_flower_pots (x : ℕ) : ℕ :=
  if h : ((x / 2) + (x / 4) + (x / 7) ≤ x - 1) then x else 0

theorem flower_pots_count : total_flower_pots 28 = 28 :=
by
  sorry

end flower_pots_count_l794_79408


namespace good_permutation_exists_iff_power_of_two_l794_79498

def is_good_permutation (n : ℕ) (a : ℕ → ℕ) : Prop :=
  ∀ i j k : ℕ, i < j → j < k → k < n → ¬ (↑n ∣ (a i + a k - 2 * a j))

theorem good_permutation_exists_iff_power_of_two (n : ℕ) (h : n ≥ 3) :
  (∃ a : ℕ → ℕ, (∀ i, i < n → a i < n) ∧ is_good_permutation n a) ↔ ∃ b : ℕ, 2 ^ b = n :=
sorry

end good_permutation_exists_iff_power_of_two_l794_79498


namespace T_30_is_13515_l794_79479

def sequence_first_element (n : ℕ) : ℕ := 1 + n * (n - 1) / 2

def sequence_last_element (n : ℕ) : ℕ := sequence_first_element n + n - 1

def sum_sequence_set (n : ℕ) : ℕ :=
  n * (sequence_first_element n + sequence_last_element n) / 2

theorem T_30_is_13515 : sum_sequence_set 30 = 13515 := by
  sorry

end T_30_is_13515_l794_79479


namespace leftmost_rectangle_is_B_l794_79495

def isLeftmostRectangle (wA wB wC wD wE : ℕ) : Prop := 
  wB < wD ∧ wB < wE

theorem leftmost_rectangle_is_B :
  let wA := 5
  let wB := 2
  let wC := 4
  let wD := 9
  let wE := 10
  let xA := 2
  let xB := 1
  let xC := 7
  let xD := 6
  let xE := 4
  let yA := 8
  let yB := 6
  let yC := 3
  let yD := 5
  let yE := 7
  let zA := 10
  let zB := 9
  let zC := 0
  let zD := 11
  let zE := 2
  isLeftmostRectangle wA wB wC wD wE :=
by
  simp only
  sorry

end leftmost_rectangle_is_B_l794_79495


namespace distance_p_runs_l794_79431

-- Given conditions
def runs_faster (speed_q : ℝ) : ℝ := 1.20 * speed_q
def head_start : ℝ := 50

-- Proof statement
theorem distance_p_runs (speed_q distance_q : ℝ) (h1 : runs_faster speed_q = 1.20 * speed_q)
                         (h2 : head_start = 50)
                         (h3 : (distance_q / speed_q) = ((distance_q + head_start) / (runs_faster speed_q))) :
                         (distance_q + head_start = 300) :=
by
  sorry

end distance_p_runs_l794_79431


namespace Wendy_did_not_recycle_2_bags_l794_79425

theorem Wendy_did_not_recycle_2_bags (points_per_bag : ℕ) (total_bags : ℕ) (points_earned : ℕ) (did_not_recycle : ℕ) : 
  points_per_bag = 5 → 
  total_bags = 11 → 
  points_earned = 45 → 
  5 * (11 - did_not_recycle) = 45 → 
  did_not_recycle = 2 :=
by
  intros h_points_per_bag h_total_bags h_points_earned h_equation
  sorry

end Wendy_did_not_recycle_2_bags_l794_79425


namespace horner_value_at_neg4_l794_79411

noncomputable def f (x : ℝ) : ℝ := 10 + 25 * x - 8 * x^2 + x^4 + 6 * x^5 + 2 * x^6

def horner_rewrite (x : ℝ) : ℝ := (((((2 * x + 6) * x + 1) * x + 0) * x - 8) * x + 25) * x + 10

theorem horner_value_at_neg4 : horner_rewrite (-4) = -36 :=
by sorry

end horner_value_at_neg4_l794_79411


namespace factor_difference_of_squares_l794_79442

theorem factor_difference_of_squares (x : ℝ) : x^2 - 1 = (x + 1) * (x - 1) := by
  sorry

end factor_difference_of_squares_l794_79442


namespace initial_mixture_two_l794_79422

theorem initial_mixture_two (x : ℝ) (h : 0.25 * (x + 0.4) = 0.10 * x + 0.4) : x = 2 :=
by
  sorry

end initial_mixture_two_l794_79422


namespace thirtieth_term_value_l794_79409

-- Define the initial term and common difference
def initial_term : ℤ := 3
def common_difference : ℤ := 4

-- Define the nth term of the arithmetic sequence
def nth_term (n : ℕ) : ℤ :=
  initial_term + (n - 1) * common_difference

-- Theorem stating the value of the 30th term
theorem thirtieth_term_value : nth_term 30 = 119 :=
by sorry

end thirtieth_term_value_l794_79409


namespace problems_solved_by_trainees_l794_79434

theorem problems_solved_by_trainees (n m : ℕ) (h : ∀ t, t < m → (∃ p, p < n → p ≥ n / 2)) :
  ∃ p < n, (∃ t, t < m → t ≥ m / 2) :=
by
  sorry

end problems_solved_by_trainees_l794_79434


namespace treasure_value_l794_79404

theorem treasure_value
    (fonzie_paid : ℕ) (auntbee_paid : ℕ) (lapis_paid : ℕ)
    (lapis_share : ℚ) (lapis_received : ℕ) (total_value : ℚ)
    (h1 : fonzie_paid = 7000) 
    (h2 : auntbee_paid = 8000) 
    (h3 : lapis_paid = 9000) 
    (h4 : fonzie_paid + auntbee_paid + lapis_paid = 24000) 
    (h5 : lapis_share = lapis_paid / (fonzie_paid + auntbee_paid + lapis_paid)) 
    (h6 : lapis_received = 337500) 
    (h7 : lapis_share * total_value = lapis_received) :
  total_value = 1125000 := by
  sorry

end treasure_value_l794_79404


namespace ten_elements_sequence_no_infinite_sequence_l794_79405

def is_valid_seq (a : ℕ → ℕ) : Prop :=
  ∀ n, (a (n + 1))^2 - 4 * (a n) * (a (n + 2)) ≥ 0

theorem ten_elements_sequence : 
  ∃ a : ℕ → ℕ, (a 9 + 1 = 10) ∧ is_valid_seq a :=
sorry

theorem no_infinite_sequence :
  ¬∃ a : ℕ → ℕ, is_valid_seq a ∧ ∀ n, a n ≥ 1 :=
sorry

end ten_elements_sequence_no_infinite_sequence_l794_79405


namespace same_terminal_side_angle_l794_79499

theorem same_terminal_side_angle (k : ℤ) : 
  ∃ (θ : ℤ), θ = k * 360 + 257 ∧ (θ % 360 = (-463) % 360) :=
by
  sorry

end same_terminal_side_angle_l794_79499


namespace sum_proof_l794_79462

theorem sum_proof (X Y : ℝ) (hX : 0.45 * X = 270) (hY : 0.35 * Y = 210) : 
  (0.75 * X) + (0.55 * Y) = 780 := by
  sorry

end sum_proof_l794_79462


namespace fourth_arithmetic_sequence_equation_l794_79497

-- Definition of an arithmetic sequence
def is_arithmetic_sequence (a : ℕ → ℝ) : Prop := ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

-- Given conditions
variables (a : ℕ → ℝ) (h : is_arithmetic_sequence a)
variable (h1 : a 1 - 2 * a 2 + a 3 = 0)
variable (h2 : a 1 - 3 * a 2 + 3 * a 3 - a 4 = 0)
variable (h3 : a 1 - 4 * a 2 + 6 * a 3 - 4 * a 4 + a 5 = 0)

-- Theorem statement to be proven
theorem fourth_arithmetic_sequence_equation : a 1 - 5 * a 2 + 10 * a 3 - 10 * a 4 + 5 * a 5 - a 6 = 0 :=
by
  sorry

end fourth_arithmetic_sequence_equation_l794_79497


namespace prove_fn_value_l794_79440

noncomputable def f (x : ℝ) : ℝ := 2^x / (2^x + 3 * x)

theorem prove_fn_value
  (m n : ℝ)
  (h1 : 2^(m + n) = 3 * m * n)
  (h2 : f m = -1 / 3) :
  f n = 4 :=
by
  sorry

end prove_fn_value_l794_79440


namespace ac_bd_sum_l794_79490

theorem ac_bd_sum (a b c d : ℝ) (h1 : a + b + c = 6) (h2 : a + b + d = -3) (h3 : a + c + d = 0) (h4 : b + c + d = -9) : 
  a * c + b * d = 23 := 
sorry

end ac_bd_sum_l794_79490


namespace abs_gt_one_iff_square_inequality_l794_79448

theorem abs_gt_one_iff_square_inequality (x : ℝ) : |x| > 1 ↔ x^2 - 1 > 0 := 
sorry

end abs_gt_one_iff_square_inequality_l794_79448


namespace emily_mean_seventh_score_l794_79429

theorem emily_mean_seventh_score :
  let a1 := 85
  let a2 := 88
  let a3 := 90
  let a4 := 94
  let a5 := 96
  let a6 := 92
  (a1 + a2 + a3 + a4 + a5 + a6 + a7) / 7 = 91 → a7 = 92 :=
by
  intros
  sorry

end emily_mean_seventh_score_l794_79429


namespace johns_original_earnings_l794_79420

def JohnsEarningsBeforeRaise (currentEarnings: ℝ) (percentageIncrease: ℝ) := 
  ∀ x, currentEarnings = x + x * percentageIncrease → x = 50

theorem johns_original_earnings : 
  JohnsEarningsBeforeRaise 80 0.60 :=
by
  intro x
  intro h
  sorry

end johns_original_earnings_l794_79420


namespace find_c_l794_79412

-- Definitions
def is_root (x c : ℝ) : Prop := x^2 - 3*x + c = 0

-- Main statement
theorem find_c (c : ℝ) (h : is_root 1 c) : c = 2 :=
sorry

end find_c_l794_79412


namespace cookie_problem_l794_79427

theorem cookie_problem : 
  ∃ (B : ℕ), B = 130 ∧ B - 80 = 50 ∧ B/2 + 20 = 85 :=
by
  sorry

end cookie_problem_l794_79427


namespace volumes_comparison_l794_79454

variable (a : ℝ) (h_a : a ≠ 3)

def volume_A := 3 * 3 * 3
def volume_B := 3 * 3 * a
def volume_C := a * a * 3
def volume_D := a * a * a

theorem volumes_comparison (h_a : a ≠ 3) :
  (volume_A + volume_D) > (volume_B + volume_C) :=
by
  have volume_A : ℝ := 27
  have volume_B := 9 * a
  have volume_C := 3 * a * a
  have volume_D := a * a * a
  sorry

end volumes_comparison_l794_79454


namespace cricket_initial_average_l794_79443

theorem cricket_initial_average (A : ℕ) (h1 : ∀ A, A * 20 + 137 = 21 * (A + 5)) : A = 32 := by
  -- assumption and proof placeholder
  sorry

end cricket_initial_average_l794_79443


namespace businessmen_drink_one_type_l794_79410

def total_businessmen : ℕ := 35
def coffee_drinkers : ℕ := 18
def tea_drinkers : ℕ := 15
def juice_drinkers : ℕ := 8
def coffee_and_tea_drinkers : ℕ := 6
def tea_and_juice_drinkers : ℕ := 4
def coffee_and_juice_drinkers : ℕ := 3
def all_three_drinkers : ℕ := 2

theorem businessmen_drink_one_type : 
  coffee_drinkers - coffee_and_tea_drinkers - coffee_and_juice_drinkers + all_three_drinkers +
  tea_drinkers - coffee_and_tea_drinkers - tea_and_juice_drinkers + all_three_drinkers +
  juice_drinkers - tea_and_juice_drinkers - coffee_and_juice_drinkers + all_three_drinkers = 21 := 
sorry

end businessmen_drink_one_type_l794_79410


namespace solve_inequality_f_ge_x_no_positive_a_b_satisfy_conditions_l794_79424

noncomputable def f (x : ℝ) : ℝ :=
  |2 * x - 1| - |2 * x - 2|

theorem solve_inequality_f_ge_x :
  {x : ℝ | f x >= x} = {x : ℝ | x <= -1 ∨ x = 1} :=
by sorry

theorem no_positive_a_b_satisfy_conditions :
  ∀ (a b : ℝ), a > 0 → b > 0 → (a + 2 * b = 1) → (2 / a + 1 / b = 4 - 1 / (a * b)) → false :=
by sorry

end solve_inequality_f_ge_x_no_positive_a_b_satisfy_conditions_l794_79424


namespace find_numbers_l794_79414

variables {x y : ℤ}

theorem find_numbers (x y : ℤ) (h1 : x - y = 11) (h2 : x^2 + y^2 = 185) (h3 : (x - y)^2 = 121) :
  (x = 13 ∧ y = 2) ∨ (x = -5 ∧ y = -16) :=
sorry

end find_numbers_l794_79414


namespace total_sum_of_money_l794_79413

theorem total_sum_of_money (x : ℝ) (A B C D E : ℝ) (hA : A = x) (hB : B = 0.75 * x) 
  (hC : C = 0.60 * x) (hD : D = 0.50 * x) (hE1 : E = 0.40 * x) (hE2 : E = 84) : 
  A + B + C + D + E = 682.50 := 
by sorry

end total_sum_of_money_l794_79413


namespace determine_max_weight_l794_79485

theorem determine_max_weight {a b : ℕ} (n : ℕ) (x : ℕ) (ha : a > 0) (hb : b > 0) (hx : 1 ≤ x ∧ x ≤ n) :
  n = 9 :=
sorry

end determine_max_weight_l794_79485


namespace multiply_millions_l794_79403

theorem multiply_millions :
  (5 * 10^6) * (8 * 10^6) = 40 * 10^12 :=
by 
  sorry

end multiply_millions_l794_79403


namespace baseball_glove_price_l794_79402

noncomputable def original_price_glove : ℝ := 42.50

theorem baseball_glove_price (cards bat glove_discounted cleats total : ℝ) 
  (h1 : cards = 25) 
  (h2 : bat = 10) 
  (h3 : cleats = 2 * 10)
  (h4 : total = 79) 
  (h5 : glove_discounted = total - (cards + bat + cleats)) 
  (h6 : glove_discounted = 0.80 * original_price_glove) : 
  original_price_glove = 42.50 := by 
  sorry

end baseball_glove_price_l794_79402


namespace base_length_of_isosceles_triangle_triangle_l794_79449

section Geometry

variable {b m x : ℝ}

-- Define the conditions
def isosceles_triangle (b : ℝ) : Prop :=
∀ {A B C : ℝ}, A = b ∧ B = b -- representing an isosceles triangle with two equal sides

def segment_length (m : ℝ) : Prop :=
∀ {D E : ℝ}, D - E = m -- the segment length between points where bisectors intersect sides is m

-- The theorem we want to prove
theorem base_length_of_isosceles_triangle_triangle (h1 : isosceles_triangle b) (h2 : segment_length m) : x = b * m / (b - m) :=
sorry

end Geometry

end base_length_of_isosceles_triangle_triangle_l794_79449


namespace f_of_f_inv_e_eq_inv_e_l794_79447

noncomputable def f : ℝ → ℝ := λ x =>
  if x ≤ 0 then Real.exp x else Real.log x

theorem f_of_f_inv_e_eq_inv_e : f (f (1 / Real.exp 1)) = 1 / Real.exp 1 := by
  sorry

end f_of_f_inv_e_eq_inv_e_l794_79447


namespace new_person_weight_l794_79417

theorem new_person_weight
  (initial_weight : ℝ)
  (average_increase : ℝ)
  (num_people : ℕ)
  (weight_replace : ℝ)
  (total_increase : ℝ)
  (W : ℝ)
  (h1 : num_people = 10)
  (h2 : average_increase = 3.5)
  (h3 : weight_replace = 65)
  (h4 : total_increase = num_people * average_increase)
  (h5 : total_increase = 35)
  (h6 : W = weight_replace + total_increase) :
  W = 100 := sorry

end new_person_weight_l794_79417


namespace problem_statement_l794_79457

open Set

def U : Set ℝ := univ
def M : Set ℝ := {x | x < 1}
def N : Set ℝ := {x | -6 < x ∧ x < 1}

theorem problem_statement : M ∩ N = N := by
  ext x
  constructor
  · intro h
    exact h.2
  · intro h
    exact ⟨h.2, h⟩

end problem_statement_l794_79457


namespace number_of_5_dollar_bills_l794_79465

theorem number_of_5_dollar_bills (x y : ℝ) (h1 : x + y = 54) (h2 : 5 * x + 20 * y = 780) : x = 20 :=
sorry

end number_of_5_dollar_bills_l794_79465


namespace sum_of_fractions_l794_79430

theorem sum_of_fractions :
  (1 / 10) + (2 / 10) + (3 / 10) + (4 / 10) + (5 / 10) + (6 / 10) + (7 / 10) + (8 / 10) + (10 / 10) + (60 / 10) = 10.6 := by
  sorry

end sum_of_fractions_l794_79430


namespace total_legs_among_tables_l794_79437

noncomputable def total_legs (total_tables four_legged_tables: ℕ) : ℕ :=
  let three_legged_tables := total_tables - four_legged_tables
  4 * four_legged_tables + 3 * three_legged_tables

theorem total_legs_among_tables : total_legs 36 16 = 124 := by
  sorry

end total_legs_among_tables_l794_79437


namespace fred_weekend_earnings_l794_79400

noncomputable def fred_initial_dollars : ℕ := 19
noncomputable def fred_final_dollars : ℕ := 40

theorem fred_weekend_earnings :
  fred_final_dollars - fred_initial_dollars = 21 :=
by
  sorry

end fred_weekend_earnings_l794_79400


namespace percentage_increase_formula_l794_79472

theorem percentage_increase_formula (A B C : ℝ) (h1 : A = 3 * B) (h2 : C = B - 30) :
  100 * ((A - C) / C) = 200 + 9000 / C := 
by 
  sorry

end percentage_increase_formula_l794_79472


namespace find_divisor_l794_79432

theorem find_divisor (d : ℕ) (N : ℕ) (a b : ℕ)
  (h1 : a = 9) (h2 : b = 79) (h3 : N = 7) :
  (∃ d, (∀ k : ℕ, a ≤ k*d ∧ k*d ≤ b → (k*d) % d = 0) ∧
   ∀ count : ℕ, count = (b / d) - ((a - 1) / d) → count = N) →
  d = 11 :=
by
  sorry

end find_divisor_l794_79432


namespace max_possible_value_of_gcd_l794_79435

theorem max_possible_value_of_gcd (n : ℕ) : gcd ((8^n - 1) / 7) ((8^(n+1) - 1) / 7) = 1 := by
  sorry

end max_possible_value_of_gcd_l794_79435


namespace base_6_addition_l794_79450

-- Definitions of base conversion and addition
def base_6_to_nat (n : ℕ) : ℕ :=
  n.div 100 * 36 + n.div 10 % 10 * 6 + n % 10

def nat_to_base_6 (n : ℕ) : ℕ :=
  let a := n.div 216
  let b := (n % 216).div 36
  let c := ((n % 216) % 36).div 6
  let d := n % 6
  a * 1000 + b * 100 + c * 10 + d

-- Conversion from base 6 to base 10 for the given numbers
def nat_256 := base_6_to_nat 256
def nat_130 := base_6_to_nat 130

-- The final theorem to prove
theorem base_6_addition : nat_to_base_6 (nat_256 + nat_130) = 1042 :=
by
  -- Proof omitted since it is not required
  sorry

end base_6_addition_l794_79450


namespace card_average_2023_l794_79426

theorem card_average_2023 (n : ℕ) (h_pos : 0 < n) (h_avg : (2 * n + 1) / 3 = 2023) : n = 3034 := by
  sorry

end card_average_2023_l794_79426


namespace tan_alpha_tan_beta_l794_79475

theorem tan_alpha_tan_beta (α β : ℝ) (h1 : Real.cos (α + β) = 3 / 5) (h2 : Real.cos (α - β) = 4 / 5) :
  Real.tan α * Real.tan β = 1 / 7 := by
  sorry

end tan_alpha_tan_beta_l794_79475


namespace fixed_point_l794_79473

noncomputable def function (a : ℝ) (h_pos : a > 0) (h_ne_one : a ≠ 1) (x : ℝ) : ℝ :=
  a ^ (x - 1) + 1

theorem fixed_point (a : ℝ) (h_pos : a > 0) (h_ne_one : a ≠ 1) :
  function a h_pos h_ne_one 1 = 2 :=
by
  sorry

end fixed_point_l794_79473


namespace div_identity_l794_79451

theorem div_identity :
  let a := 6 / 2
  let b := a * 3
  120 / b = 120 / 9 :=
by
  sorry

end div_identity_l794_79451


namespace range_of_a_l794_79478

theorem range_of_a (a : ℝ) : (∀ x : ℝ, 0 < x → a < x + (1 / x)) → a < 2 :=
by
  sorry

end range_of_a_l794_79478


namespace value_of_y_l794_79463

-- Problem: Prove that given the conditions \( x - y = 8 \) and \( x + y = 16 \),
-- the value of \( y \) is 4.
theorem value_of_y (x y : ℝ) (h1 : x - y = 8) (h2 : x + y = 16) : y = 4 := 
sorry

end value_of_y_l794_79463


namespace percentage_of_apples_after_removal_l794_79438

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

end percentage_of_apples_after_removal_l794_79438


namespace range_of_a_l794_79406

variable (a : ℝ)
def is_second_quadrant (z : ℂ) : Prop := z.re < 0 ∧ z.im > 0
def z : ℂ := 4 - 2 * Complex.I

theorem range_of_a (ha : is_second_quadrant ((z + a * Complex.I) ^ 2)) : a > 6 := by
  sorry

end range_of_a_l794_79406


namespace contribution_per_person_l794_79428

-- Define constants for the given conditions
def total_price : ℕ := 67
def coupon : ℕ := 4
def number_of_people : ℕ := 3

-- The theorem to prove
theorem contribution_per_person : (total_price - coupon) / number_of_people = 21 :=
by 
  -- The proof is omitted for this exercise
  sorry

end contribution_per_person_l794_79428


namespace grandparents_to_parents_ratio_l794_79458

-- Definitions corresponding to the conditions
def wallet_cost : ℕ := 100
def betty_half_money : ℕ := wallet_cost / 2
def parents_contribution : ℕ := 15
def betty_needs_more : ℕ := 5
def grandparents_contribution : ℕ := 95 - (betty_half_money + parents_contribution)

-- The mathematical statement for the proof
theorem grandparents_to_parents_ratio :
  grandparents_contribution / parents_contribution = 2 := by
  sorry

end grandparents_to_parents_ratio_l794_79458


namespace sum_of_first_10_bn_l794_79483

def a (n : ℕ) : ℚ :=
  (2 / 5) * n + (3 / 5)

def b (n : ℕ) : ℤ :=
  ⌊a n⌋

def sum_first_10_b : ℤ :=
  (b 1) + (b 2) + (b 3) + (b 4) + (b 5) + (b 6) + (b 7) + (b 8) + (b 9) + (b 10)

theorem sum_of_first_10_bn : sum_first_10_b = 24 :=
  by sorry

end sum_of_first_10_bn_l794_79483


namespace initial_pigs_l794_79401

theorem initial_pigs (x : ℕ) (h : x + 86 = 150) : x = 64 :=
by
  sorry

end initial_pigs_l794_79401


namespace sum_of_three_numbers_l794_79488

theorem sum_of_three_numbers (a b c : ℝ) 
  (h1 : a^2 + b^2 + c^2 = 156)
  (h2 : a * b + b * c + c * a = 50) :
  a + b + c = 16 :=
by
  sorry

end sum_of_three_numbers_l794_79488


namespace part_I_part_II_l794_79433

-- Conditions
def p (x m : ℝ) : Prop := x > m → 2 * x - 5 > 0
def q (m : ℝ) : Prop := ∃ x y : ℝ, (x^2 / (m - 1)) + (y^2 / (2 - m)) = 1

-- Statements for proof
theorem part_I (m x : ℝ) (hq: q m) (hp: p x m) : 
  m < 1 ∨ (2 < m ∧ m ≤ 5 / 2) :=
sorry

theorem part_II (m x : ℝ) (hq: ¬ q m ∧ ¬(p x m ∧ q m) ∧ (p x m ∨ q m)) : 
  (1 ≤ m ∧ m ≤ 2) ∨ (m > 5 / 2) :=
sorry

end part_I_part_II_l794_79433


namespace positive_integers_no_common_factor_l794_79471

theorem positive_integers_no_common_factor (X Y Z : ℕ) 
    (X_pos : 0 < X) (Y_pos : 0 < Y) (Z_pos : 0 < Z)
    (coprime_XYZ : Nat.gcd (Nat.gcd X Y) Z = 1)
    (eqn : X * (Real.log 3 / Real.log 100) + Y * (Real.log 4 / Real.log 100) = Z^2) :
    X + Y + Z = 4 :=
sorry

end positive_integers_no_common_factor_l794_79471


namespace rahul_savings_l794_79423

variable (NSC PPF total_savings : ℕ)

theorem rahul_savings (h1 : NSC / 3 = PPF / 2) (h2 : PPF = 72000) : total_savings = 180000 :=
by
  sorry

end rahul_savings_l794_79423


namespace pipe_fill_time_l794_79456

variable (t : ℝ)

theorem pipe_fill_time (h1 : 0 < t) (h2 : 0 < t / 5) (h3 : (1 / t) + (5 / t) = 1 / 5) : t = 30 :=
by
  sorry

end pipe_fill_time_l794_79456


namespace modified_monotonous_count_l794_79486

def is_modified_monotonous (n : ℕ) : Prop :=
  -- Definition that determines if a number is modified-monotonous
  -- Must include digit '5', and digits must form a strictly increasing or decreasing sequence
  sorry 

def count_modified_monotonous (n : ℕ) : ℕ :=
  2 * (8 * (2^8) + 2^8) + 1 -- Formula for counting modified-monotonous numbers including '5'

theorem modified_monotonous_count : count_modified_monotonous 5 = 4609 := 
  by 
    sorry

end modified_monotonous_count_l794_79486


namespace find_m_l794_79415

-- Given definitions and conditions
def is_ellipse (x y m : ℝ) := (x^2 / m) + (y^2 / 4) = 1
def eccentricity (m : ℝ) := Real.sqrt ((m - 4) / m) = 1 / 2

-- Prove that m = 16 / 3 given the conditions
theorem find_m (m : ℝ) (cond1 : is_ellipse 1 1 m) (cond2 : eccentricity m) (cond3 : m > 4) : m = 16 / 3 :=
by
  sorry

end find_m_l794_79415


namespace vector_subtraction_parallel_l794_79419

theorem vector_subtraction_parallel (t : ℝ) 
  (h_parallel : -1 / 2 = -3 / t) : 
  ( (-1 : ℝ), -3 ) - ( 2, t ) = (-3, -9) :=
by
  -- proof goes here
  sorry

end vector_subtraction_parallel_l794_79419


namespace union_of_A_and_B_l794_79445

def A : Set ℕ := {1, 2, 4}
def B : Set ℕ := {2, 4}

theorem union_of_A_and_B : A ∪ B = {1, 2, 4} := by
  sorry

end union_of_A_and_B_l794_79445


namespace units_digit_k_cube_plus_2_to_k_plus_1_mod_10_l794_79460

def k : ℕ := 2012 ^ 2 + 2 ^ 2012

theorem units_digit_k_cube_plus_2_to_k_plus_1_mod_10 : (k ^ 3 + 2 ^ (k + 1)) % 10 = 2 := 
by sorry

end units_digit_k_cube_plus_2_to_k_plus_1_mod_10_l794_79460


namespace minimum_value_is_six_l794_79461

noncomputable def minimum_value (m n : ℝ) (h : m > 2 * n) : ℝ :=
  m + (4 * n ^ 2 - 2 * m * n + 9) / (m - 2 * n)

theorem minimum_value_is_six (m n : ℝ) (h : m > 2 * n) : minimum_value m n h = 6 := 
sorry

end minimum_value_is_six_l794_79461


namespace cone_volume_proof_l794_79470

noncomputable def slant_height := 21
noncomputable def horizontal_semi_axis := 10
noncomputable def vertical_semi_axis := 12
noncomputable def equivalent_radius :=
  Real.sqrt (horizontal_semi_axis * vertical_semi_axis)
noncomputable def cone_height :=
  Real.sqrt (slant_height ^ 2 - equivalent_radius ^ 2)

noncomputable def cone_volume :=
  (1 / 3) * Real.pi * horizontal_semi_axis * vertical_semi_axis * cone_height

theorem cone_volume_proof :
  cone_volume = 2250.24 * Real.pi := sorry

end cone_volume_proof_l794_79470


namespace fraction_milk_in_mug1_is_one_fourth_l794_79494

-- Condition definitions
def initial_tea_mug1 := 6 -- ounces
def initial_milk_mug2 := 6 -- ounces
def tea_transferred_mug1_to_mug2 := initial_tea_mug1 / 3
def tea_remaining_mug1 := initial_tea_mug1 - tea_transferred_mug1_to_mug2
def total_liquid_mug2 := initial_milk_mug2 + tea_transferred_mug1_to_mug2
def portion_transferred_back := total_liquid_mug2 / 4
def tea_ratio_mug2 := tea_transferred_mug1_to_mug2 / total_liquid_mug2
def milk_ratio_mug2 := initial_milk_mug2 / total_liquid_mug2
def tea_transferred_back := portion_transferred_back * tea_ratio_mug2
def milk_transferred_back := portion_transferred_back * milk_ratio_mug2
def final_tea_mug1 := tea_remaining_mug1 + tea_transferred_back
def final_milk_mug1 := milk_transferred_back
def final_total_liquid_mug1 := final_tea_mug1 + final_milk_mug1

-- Lean statement of the problem
theorem fraction_milk_in_mug1_is_one_fourth :
  final_milk_mug1 / final_total_liquid_mug1 = 1 / 4 :=
by
  sorry

end fraction_milk_in_mug1_is_one_fourth_l794_79494


namespace stones_on_perimeter_of_square_l794_79480

theorem stones_on_perimeter_of_square (n : ℕ) (h : n = 5) : 
  4 * n - 4 = 16 :=
by
  sorry

end stones_on_perimeter_of_square_l794_79480


namespace solve_system_l794_79491

theorem solve_system : ∃ x y : ℚ, 
  (2 * x + 3 * y = 7 - 2 * x + 7 - 3 * y) ∧ 
  (3 * x - 2 * y = x - 2 + y - 2) ∧ 
  x = 3 / 4 ∧ 
  y = 11 / 6 := 
by 
  sorry

end solve_system_l794_79491


namespace no_integer_solution_system_l794_79441

theorem no_integer_solution_system (
  x y z : ℤ
) : x^6 + x^3 + x^3 * y + y ≠ 147 ^ 137 ∨ x^3 + x^3 * y + y^2 + y + z^9 ≠ 157 ^ 117 :=
by
  sorry

end no_integer_solution_system_l794_79441


namespace university_math_students_l794_79466

theorem university_math_students
  (total_students : ℕ)
  (math_only : ℕ)
  (stats_only : ℕ)
  (both_courses : ℕ)
  (H1 : total_students = 75)
  (H2 : math_only + stats_only + both_courses = total_students)
  (H3 : math_only = 2 * (stats_only + both_courses))
  (H4 : both_courses = 9) :
  math_only + both_courses = 53 :=
by
  sorry

end university_math_students_l794_79466


namespace cricketer_runs_l794_79467

theorem cricketer_runs (R x : ℝ) : 
  (R / 85 = 12.4) →
  ((R + x) / 90 = 12.0) →
  x = 26 := 
by
  sorry

end cricketer_runs_l794_79467


namespace revenue_ratio_l794_79439

variable (R_d : ℝ) (R_n : ℝ) (R_j : ℝ)

theorem revenue_ratio
  (nov_cond : R_n = 2 / 5 * R_d)
  (jan_cond : R_j = 1 / 2 * R_n) :
  R_d = 10 / 3 * ((R_n + R_j) / 2) := by
  -- Proof steps go here
  sorry

end revenue_ratio_l794_79439


namespace maria_baggies_count_l794_79416

def total_cookies (chocolate_chip : ℕ) (oatmeal : ℕ) : ℕ :=
  chocolate_chip + oatmeal

def baggies_count (total_cookies : ℕ) (cookies_per_bag : ℕ) : ℕ :=
  total_cookies / cookies_per_bag

theorem maria_baggies_count :
  let choco_chip := 2
  let oatmeal := 16
  let cookies_per_bag := 3
  baggies_count (total_cookies choco_chip oatmeal) cookies_per_bag = 6 :=
by
  sorry

end maria_baggies_count_l794_79416


namespace find_k_if_lines_parallel_l794_79484

theorem find_k_if_lines_parallel (k : ℝ) : (∀ x y, y = 5 * x + 3 → y = (3 * k) * x + 7) → k = 5 / 3 :=
by {
  sorry
}

end find_k_if_lines_parallel_l794_79484


namespace last_popsicle_melts_32_times_faster_l794_79492

theorem last_popsicle_melts_32_times_faster (t : ℕ) : 
  let time_first := t
  let time_sixth := t / 2^5
  (time_first / time_sixth) = 32 :=
by
  sorry

end last_popsicle_melts_32_times_faster_l794_79492


namespace cubic_polynomials_l794_79496

theorem cubic_polynomials (a b c a₁ b₁ c₁ a₂ b₂ c₂ : ℝ) 
    (h1 : a - 1/b = r₁ ∧ b - 1/c = r₂ ∧ c - 1/a = r₃)
    (h2 : r₁ + r₂ + r₃ = 5)
    (h3 : r₁ * r₂ + r₂ * r₃ + r₃ * r₁ = -15)
    (h4 : r₁ * r₂ * r₃ = -3)
    (h5 : a₁ * b₁ * c₁ = 1 + Real.sqrt 2 ∨ a₁ * b₁ * c₁ = 1 - Real.sqrt 2)
    (h6 : a₂ * b₂ * c₂ = 1 + Real.sqrt 2 ∨ a₂ * b₂ * c₂ = 1 - Real.sqrt 2) :
    (-(a₁ * b₁ * c₁))^3 + (-(a₂ * b₂ * c₂))^3 = -14 := sorry

end cubic_polynomials_l794_79496


namespace abs_conditions_iff_l794_79476

theorem abs_conditions_iff (x y : ℝ) :
  (|x| < 1 ∧ |y| < 1) ↔ (|x + y| + |x - y| < 2) :=
by
  sorry

end abs_conditions_iff_l794_79476


namespace sum_of_arithmetic_sequence_l794_79444

variable {α : Type*} [LinearOrderedField α]

def sum_arithmetic_sequence (a₁ d : α) (n : ℕ) : α :=
  n * (2 * a₁ + (n - 1) * d) / 2

theorem sum_of_arithmetic_sequence {a₁ d : α}
  (h₁ : sum_arithmetic_sequence a₁ d 10 = 12) :
  (a₁ + 4 * d) + (a₁ + 5 * d) = 12 / 5 :=
by
  sorry

end sum_of_arithmetic_sequence_l794_79444


namespace minimum_value_y_l794_79452

theorem minimum_value_y (x : ℝ) (hx : x > 2) : 
  ∃ y, y = x + 4 / (x - 2) ∧ ∀ z, (z = x + 4 / (x - 2) → z ≥ 6) :=
by
  sorry

end minimum_value_y_l794_79452


namespace rectangle_side_lengths_l794_79489

variables (x y m n S : ℝ) (hx_y_ratio : x / y = m / n) (hxy_area : x * y = S)

theorem rectangle_side_lengths :
  x = Real.sqrt (m * S / n) ∧ y = Real.sqrt (n * S / m) :=
sorry

end rectangle_side_lengths_l794_79489


namespace steve_more_than_wayne_first_time_at_2004_l794_79493

def initial_steve_money (year: ℕ) := if year = 2000 then 100 else 0
def initial_wayne_money (year: ℕ) := if year = 2000 then 10000 else 0

def steve_money (year: ℕ) : ℕ :=
  if year < 2000 then 0
  else if year = 2000 then initial_steve_money year
  else 2 * steve_money (year - 1)

def wayne_money (year: ℕ) : ℕ :=
  if year < 2000 then 0
  else if year = 2000 then initial_wayne_money year
  else wayne_money (year - 1) / 2

theorem steve_more_than_wayne_first_time_at_2004 :
  ∃ (year: ℕ), year = 2004 ∧ steve_money year > wayne_money year := by
  sorry

end steve_more_than_wayne_first_time_at_2004_l794_79493


namespace mark_boxes_sold_l794_79446

theorem mark_boxes_sold (n : ℕ) (M A : ℕ) (h1 : A = n - 2) (h2 : M + A < n) (h3 :  1 ≤ M) (h4 : 1 ≤ A) (hn : n = 12) : M = 1 :=
by
  sorry

end mark_boxes_sold_l794_79446


namespace cars_sold_l794_79455

theorem cars_sold (sales_Mon sales_Tue sales_Wed cars_Thu_Fri_Sat : ℕ) 
  (mean : ℝ) (h1 : sales_Mon = 8) 
  (h2 : sales_Tue = 3) 
  (h3 : sales_Wed = 10) 
  (h4 : mean = 5.5) 
  (h5 : mean * 6 = sales_Mon + sales_Tue + sales_Wed + cars_Thu_Fri_Sat):
  cars_Thu_Fri_Sat = 12 :=
sorry

end cars_sold_l794_79455


namespace mrs_lee_earnings_percentage_l794_79453

theorem mrs_lee_earnings_percentage 
  (M F : ℝ)
  (H1 : 1.20 * M = 0.5454545454545454 * (1.20 * M + F)) :
  M = 0.5 * (M + F) :=
by sorry

end mrs_lee_earnings_percentage_l794_79453


namespace no_integer_points_between_A_and_B_on_line_l794_79459

theorem no_integer_points_between_A_and_B_on_line
  (A : ℕ × ℕ) (B : ℕ × ℕ)
  (hA : A = (2, 3))
  (hB : B = (50, 500)) :
  ∀ (P : ℕ × ℕ), P.1 > 2 ∧ P.1 < 50 ∧ 
    (P.2 * 48 - P.1 * 497 = 2 * 497 - 3 * 48) →
    false := 
by
  sorry

end no_integer_points_between_A_and_B_on_line_l794_79459


namespace james_distance_ridden_l794_79477

theorem james_distance_ridden : 
  let speed := 16 
  let time := 5 
  speed * time = 80 := 
by
  sorry

end james_distance_ridden_l794_79477


namespace true_propositions_l794_79474

noncomputable def discriminant_leq_zero : Prop :=
  let a := 1
  let b := -1
  let c := 2
  b^2 - 4 * a * c ≤ 0

def proposition_1 : Prop := discriminant_leq_zero

def proposition_2 (x : ℝ) : Prop :=
  abs x ≥ 0 → x ≥ 0

def proposition_3 : Prop :=
  5 > 2 ∧ 3 < 7

theorem true_propositions : proposition_1 ∧ proposition_3 ∧ ¬∀ x : ℝ, proposition_2 x :=
by
  sorry

end true_propositions_l794_79474


namespace probability_white_ball_initial_probability_yellow_or_red_ball_initial_probability_white_ball_after_removal_l794_79464

def total_balls := 20
def red_balls := 10
def yellow_balls := 6
def white_balls := 4
def initial_white_balls_probability := (white_balls : ℚ) / total_balls
def initial_yellow_or_red_balls_probability := (yellow_balls + red_balls : ℚ) / total_balls

def removed_red_balls := 2
def removed_white_balls := 2
def remaining_balls := total_balls - (removed_red_balls + removed_white_balls)
def remaining_white_balls := white_balls - removed_white_balls
def remaining_white_balls_probability := (remaining_white_balls : ℚ) / remaining_balls

theorem probability_white_ball_initial : initial_white_balls_probability = 1 / 5 := by sorry
theorem probability_yellow_or_red_ball_initial : initial_yellow_or_red_balls_probability = 4 / 5 := by sorry
theorem probability_white_ball_after_removal : remaining_white_balls_probability = 1 / 8 := by sorry

end probability_white_ball_initial_probability_yellow_or_red_ball_initial_probability_white_ball_after_removal_l794_79464


namespace range_of_a_min_value_of_a_l794_79482

variable (f : ℝ → ℝ) (a x : ℝ)

-- Part 1
theorem range_of_a (f_def : ∀ x, f x = abs (x - a)) 
  (h₁ : ∀ x, 1 ≤ x ∧ x ≤ 3 → f x ≤ 3) : 0 ≤ a ∧ a ≤ 4 :=
sorry

-- Part 2
theorem min_value_of_a (f_def : ∀ x, f x = abs (x - a)) 
  (h₂ : ∀ x, f (x - a) + f (x + a) ≥ 1 - a) : a ≥ 1/3 :=
sorry

end range_of_a_min_value_of_a_l794_79482


namespace total_volume_needed_l794_79481

def box_length : ℕ := 20
def box_width : ℕ := 20
def box_height : ℕ := 12
def box_cost : ℕ := 50 -- in cents to avoid using floats
def total_spent : ℕ := 20000 -- $200 in cents

def volume_of_box : ℕ := box_length * box_width * box_height
def number_of_boxes : ℕ := total_spent / box_cost

theorem total_volume_needed : number_of_boxes * volume_of_box = 1920000 := by
  sorry

end total_volume_needed_l794_79481


namespace total_numbers_is_eight_l794_79407

theorem total_numbers_is_eight
  (avg_all : ∀ n : ℕ, (total_sum : ℝ) / n = 25)
  (avg_first_two : ∀ a₁ a₂ : ℝ, (a₁ + a₂) / 2 = 20)
  (avg_next_three : ∀ a₃ a₄ a₅ : ℝ, (a₃ + a₄ + a₅) / 3 = 26)
  (h_sixth : ∀ a₆ a₇ a₈ : ℝ, a₆ + 4 = a₇ ∧ a₆ + 6 = a₈)
  (last_num : ∀ a₈ : ℝ, a₈ = 30) :
  ∃ n : ℕ, n = 8 :=
by
  sorry

end total_numbers_is_eight_l794_79407


namespace all_possible_values_of_k_l794_79421

def is_partition_possible (k : ℕ) : Prop :=
  ∃ (A B : Finset ℕ), (A ∪ B = Finset.range (k + 1)) ∧ (A ∩ B = ∅) ∧ (A.sum id = 2 * B.sum id)

theorem all_possible_values_of_k (k : ℕ) : 
  is_partition_possible k → ∃ m : ℕ, k = 3 * m ∨ k = 3 * m - 1 :=
by
  intro h
  sorry

end all_possible_values_of_k_l794_79421


namespace continuous_implies_defined_defined_does_not_imply_continuous_l794_79469

-- Define function continuity at a point x = a
def continuous_at (f : ℝ → ℝ) (a : ℝ) : Prop :=
  ∀ ε > 0, ∃ δ > 0, ∀ x, abs (x - a) < δ → abs (f x - f a) < ε

-- Prove that if f is continuous at x = a, then f is defined at x = a
theorem continuous_implies_defined (f : ℝ → ℝ) (a : ℝ) : 
  continuous_at f a → ∃ y, f a = y :=
by
  sorry  -- Proof omitted

-- Prove that the definition of f at x = a does not guarantee continuity at x = a
theorem defined_does_not_imply_continuous (f : ℝ → ℝ) (a : ℝ) :
  (∃ y, f a = y) → ¬ continuous_at f a :=
by
  sorry  -- Proof omitted

end continuous_implies_defined_defined_does_not_imply_continuous_l794_79469


namespace expression_equals_base10_l794_79468

-- Define numbers in various bases
def base7ToDec (n : ℕ) : ℕ := 1 * (7^2) + 6 * (7^1) + 5 * (7^0)
def base2ToDec (n : ℕ) : ℕ := 1 * (2^1) + 1 * (2^0)
def base6ToDec (n : ℕ) : ℕ := 1 * (6^2) + 2 * (6^1) + 1 * (6^0)
def base3ToDec (n : ℕ) : ℕ := 2 * (3^1) + 1 * (3^0)

-- Prove the given expression equals 39 in base 10
theorem expression_equals_base10 :
  (base7ToDec 165 / base2ToDec 11) + (base6ToDec 121 / base3ToDec 21) = 39 :=
by
  -- Convert the base n numbers to base 10
  let num1 := base7ToDec 165
  let den1 := base2ToDec 11
  let num2 := base6ToDec 121
  let den2 := base3ToDec 21
  
  -- Simplify the expression (skipping actual steps for brevity, replaced by sorry)
  sorry

end expression_equals_base10_l794_79468

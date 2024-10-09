import Mathlib

namespace minimum_expression_l178_17842

variable (a b : ℝ)

theorem minimum_expression (h1 : 0 < a) (h2 : 0 < b) (h3 : a + b = 3) :
  (∃ a b : ℝ, 0 < a ∧ 0 < b ∧ a + b = 3 ∧ (∀ x y : ℝ, 0 < x → 0 < y → x + y = 3 → 
  x = a ∧ y = b  → ∃ m : ℝ, m ≥ 1 ∧ (m = (1/(a+1)) + 1/b))) := sorry

end minimum_expression_l178_17842


namespace determine_n_between_sqrt3_l178_17850

theorem determine_n_between_sqrt3 (n : ℕ) (hpos : 0 < n)
  (hineq : (n + 3) / n < Real.sqrt 3 ∧ Real.sqrt 3 < (n + 4) / (n + 1)) :
  n = 4 :=
sorry

end determine_n_between_sqrt3_l178_17850


namespace determine_d_l178_17812

def Q (x d : ℝ) : ℝ := x^3 - 3*x^2 + d*x - 8

theorem determine_d (d : ℝ) : (∃ d, Q (-2) d = 0) → d = -14 := by
  sorry

end determine_d_l178_17812


namespace reams_paper_l178_17892

theorem reams_paper (total_reams reams_haley reams_sister : Nat) 
    (h1 : total_reams = 5)
    (h2 : reams_haley = 2)
    (h3 : total_reams = reams_haley + reams_sister) : 
    reams_sister = 3 := by
  sorry

end reams_paper_l178_17892


namespace parabola_directrix_l178_17834

theorem parabola_directrix (x y : ℝ) (h : x^2 = 8 * y) : y = -2 :=
sorry

end parabola_directrix_l178_17834


namespace fence_poles_placement_l178_17855

def total_bridges_length (bridges : List ℕ) : ℕ :=
  bridges.sum

def effective_path_length (path_length : ℕ) (bridges_length : ℕ) : ℕ :=
  path_length - bridges_length

def poles_on_one_side (effective_length : ℕ) (interval : ℕ) : ℕ :=
  effective_length / interval

def total_poles (path_length : ℕ) (interval : ℕ) (bridges : List ℕ) : ℕ :=
  let bridges_length := total_bridges_length bridges
  let effective_length := effective_path_length path_length bridges_length
  let poles_one_side := poles_on_one_side effective_length interval
  2 * poles_one_side + 2

theorem fence_poles_placement :
  total_poles 2300 8 [48, 58, 62] = 534 := by
  sorry

end fence_poles_placement_l178_17855


namespace modular_inverse_of_17_mod_800_l178_17827

    theorem modular_inverse_of_17_mod_800 :
      ∃ x : ℤ, 0 ≤ x ∧ x < 800 ∧ (17 * x) % 800 = 1 :=
    by
      use 47
      sorry
    
end modular_inverse_of_17_mod_800_l178_17827


namespace spelling_bee_participants_l178_17899

theorem spelling_bee_participants (n : ℕ)
  (h1 : ∀ k, k > 0 → k ≤ n → k ≠ 75 → (k - 1 < 74 ∨ k - 1 > 74))
  (h2 : ∀ k, k > 0 → k ≤ n → k ≠ 75 → (75 - k > 0 ∨ k - 1 > 74)) :
  n = 149 := by
  sorry

end spelling_bee_participants_l178_17899


namespace problem1_problem2_l178_17873

noncomputable def h (x a : ℝ) : ℝ := (x - a) * Real.exp x + a
noncomputable def f (x b : ℝ) : ℝ := x^2 - 2 * b * x - 3 * Real.exp 1 + Real.exp 1 + 15 / 2

theorem problem1 (a : ℝ) :
  ∃ c, ∀ x ∈ Set.Icc (-1:ℝ) (1:ℝ), h x a ≥ c :=
by
  sorry

theorem problem2 (b : ℝ) :
  (∀ x1 ∈ Set.Icc (-1:ℝ) (1:ℝ), ∃ x2 ∈ Set.Icc (1:ℝ) (2:ℝ), h x1 3 ≥ f x2 b) →
  b ≥ 17 / 8 :=
by
  sorry

end problem1_problem2_l178_17873


namespace factors_of_180_l178_17822

theorem factors_of_180 : ∃ n : ℕ, n = 18 ∧ ∀ p q r : ℕ, 180 = p^2 * q^2 * r^1 → 
  n = (p + 1) * (q + 1) * (r) :=
by
  sorry

end factors_of_180_l178_17822


namespace problem_value_l178_17854

theorem problem_value :
  4 * (8 - 3) / 2 - 7 = 3 := 
by
  sorry

end problem_value_l178_17854


namespace john_dimes_l178_17808

theorem john_dimes :
  ∀ (d : ℕ), 
  (4 * 25 + d * 10 + 5) = 135 → (5: ℕ) + (d: ℕ) * 10 + 4 = 4 + 131 + 3*d → d = 3 :=
by
  sorry

end john_dimes_l178_17808


namespace paytons_score_l178_17878

theorem paytons_score (total_score_14_students : ℕ)
    (average_14_students : total_score_14_students / 14 = 80)
    (total_score_15_students : ℕ)
    (average_15_students : total_score_15_students / 15 = 81) :
  total_score_15_students - total_score_14_students = 95 :=
by
  sorry

end paytons_score_l178_17878


namespace max_value_at_log2_one_l178_17857

noncomputable def f (x : ℝ) : ℝ := 2 * x + 2 - 3 * (4 : ℝ) ^ x
def domain (x : ℝ) : Prop := x < 1 ∨ x > 3

theorem max_value_at_log2_one :
  (∃ x, domain x ∧ f x = 0) ∧ (∀ y, domain y → f y ≤ 0) :=
by
  sorry

end max_value_at_log2_one_l178_17857


namespace optionA_is_multiple_of_5_optionB_is_multiple_of_5_optionC_is_multiple_of_5_optionD_is_multiple_of_5_optionE_is_not_multiple_of_5_l178_17837

-- Definitions of the options
def optionA : ℕ := 2019^2 - 2014^2
def optionB : ℕ := 2019^2 * 10^2
def optionC : ℕ := 2020^2 / 101^2
def optionD : ℕ := 2010^2 - 2005^2
def optionE : ℕ := 2015^2 / 5^2

-- Statements to be proven
theorem optionA_is_multiple_of_5 : optionA % 5 = 0 := by sorry
theorem optionB_is_multiple_of_5 : optionB % 5 = 0 := by sorry
theorem optionC_is_multiple_of_5 : optionC % 5 = 0 := by sorry
theorem optionD_is_multiple_of_5 : optionD % 5 = 0 := by sorry
theorem optionE_is_not_multiple_of_5 : optionE % 5 ≠ 0 := by sorry

end optionA_is_multiple_of_5_optionB_is_multiple_of_5_optionC_is_multiple_of_5_optionD_is_multiple_of_5_optionE_is_not_multiple_of_5_l178_17837


namespace median_CD_eq_altitude_from_C_eq_centroid_G_eq_l178_17866

namespace Geometry

/-- Vertices of the triangle -/
def A : ℝ × ℝ := (4, 4)
def B : ℝ × ℝ := (-4, 2)
def C : ℝ × ℝ := (2, 0)

/-- Proof of the equation of the median CD on the side AB -/
theorem median_CD_eq : ∀ (x y : ℝ), 3 * x + 2 * y - 6 = 0 :=
sorry

/-- Proof of the equation of the altitude from C to AB -/
theorem altitude_from_C_eq : ∀ (x y : ℝ), 4 * x + y - 8 = 0 :=
sorry

/-- Proof of the coordinates of the centroid G of triangle ABC -/
theorem centroid_G_eq : ∃ (x y : ℝ), x = 2 / 3 ∧ y = 2 :=
sorry

end Geometry

end median_CD_eq_altitude_from_C_eq_centroid_G_eq_l178_17866


namespace turnip_mixture_l178_17848

theorem turnip_mixture (cups_potatoes total_turnips : ℕ) (h_ratio : 20 = 5 * 4) (h_turnips : total_turnips = 8) :
    cups_potatoes = 2 :=
by
    have ratio := h_ratio
    have turnips := h_turnips
    sorry

end turnip_mixture_l178_17848


namespace central_angle_star_in_polygon_l178_17839

theorem central_angle_star_in_polygon (n : ℕ) (h : 2 < n) : 
  ∃ C, C = 720 / n :=
by sorry

end central_angle_star_in_polygon_l178_17839


namespace triangle_inequality_example_l178_17817

theorem triangle_inequality_example {x : ℝ} (h1: 3 + 4 > x) (h2: abs (3 - 4) < x) : 1 < x ∧ x < 7 :=
  sorry

end triangle_inequality_example_l178_17817


namespace find_value_of_k_l178_17891

noncomputable def line_parallel_and_point_condition (k : ℝ) :=
  ∃ (m : ℝ), m = -5/4 ∧ (22 - (-8)) / (k - 3) = m

theorem find_value_of_k : ∃ k : ℝ, line_parallel_and_point_condition k ∧ k = -21 :=
by
  sorry

end find_value_of_k_l178_17891


namespace verify_p_q_l178_17832

open Matrix

def N : Matrix (Fin 2) (Fin 2) ℤ :=
  ![![3, 4], ![-5, 2]]

def p : ℤ := 5
def q : ℤ := -26

theorem verify_p_q :
  N * N = p • N + q • (1 : Matrix (Fin 2) (Fin 2) ℤ) :=
by
  -- Skipping the proof
  sorry

end verify_p_q_l178_17832


namespace odd_squarefree_integers_1_to_199_l178_17805

noncomputable def count_squarefree_odd_integers (n : ℕ) :=
  n - List.sum [
    n / 18,   -- for 3^2 = 9
    n / 50,   -- for 5^2 = 25
    n / 98,   -- for 7^2 = 49
    n / 162,  -- for 9^2 = 81
    n / 242,  -- for 11^2 = 121
    n / 338   -- for 13^2 = 169
  ]

theorem odd_squarefree_integers_1_to_199 : count_squarefree_odd_integers 198 = 79 := 
by
  sorry

end odd_squarefree_integers_1_to_199_l178_17805


namespace fernanda_savings_calc_l178_17820

noncomputable def aryan_debt : ℝ := 1200
noncomputable def kyro_debt : ℝ := aryan_debt / 2
noncomputable def aryan_payment : ℝ := 0.60 * aryan_debt
noncomputable def kyro_payment : ℝ := 0.80 * kyro_debt
noncomputable def initial_savings : ℝ := 300
noncomputable def total_payment_received : ℝ := aryan_payment + kyro_payment
noncomputable def total_savings : ℝ := initial_savings + total_payment_received

theorem fernanda_savings_calc : total_savings = 1500 := by
  sorry

end fernanda_savings_calc_l178_17820


namespace can_restore_axes_l178_17847

noncomputable def restore_axes (A : ℝ×ℝ) (hA : A.snd = 3 ^ A.fst) : Prop :=
  ∃ (B C D : ℝ×ℝ),
    (B.fst = A.fst ∧ B.snd = 0) ∧
    (C.fst = A.fst ∧ C.snd = A.snd) ∧
    (D.fst = A.fst ∧ D.snd = 3 ^ C.fst) ∧
    (∃ (extend_perpendicular : ∀ (x: ℝ), ℝ→ℝ), extend_perpendicular A.snd B.fst = D.snd)

theorem can_restore_axes (A : ℝ×ℝ) (hA : A.snd = 3 ^ A.fst) : restore_axes A hA :=
  sorry

end can_restore_axes_l178_17847


namespace complement_union_correct_l178_17814

-- Define the universal set U, set A, and set B
def U : Set ℕ := {1, 2, 3, 4, 5}
def A : Set ℕ := {1, 2}
def B : Set ℕ := {2, 3}

-- The complement of A with respect to U
def complement_U_A : Set ℕ := {x | x ∈ U ∧ x ∉ A}

-- The union of the complement of A and set B
def union_complement_U_A_B : Set ℕ := complement_U_A ∪ B

-- State the theorem to prove
theorem complement_union_correct : union_complement_U_A_B = {2, 3, 4, 5} := 
by 
  sorry

end complement_union_correct_l178_17814


namespace equations_have_one_contact_point_l178_17836

theorem equations_have_one_contact_point (c : ℝ):
  (∃ x : ℝ, x^2 + 1 = 4 * x + c) ∧ (∀ x1 x2 : ℝ, (x1 ≠ x2) → ¬(x1^2 + 1 = 4 * x1 + c ∧ x2^2 + 1 = 4 * x2 + c)) ↔ c = -3 :=
by
  sorry

end equations_have_one_contact_point_l178_17836


namespace central_angle_of_regular_hexagon_l178_17816

-- Define the total degrees in a circle
def total_degrees_in_circle : ℝ := 360

-- Define the number of sides in a regular hexagon
def sides_in_hexagon : ℕ := 6

-- Theorems to prove that the central angle of a regular hexagon is 60°
theorem central_angle_of_regular_hexagon :
  total_degrees_in_circle / sides_in_hexagon = 60 :=
by
  sorry

end central_angle_of_regular_hexagon_l178_17816


namespace quadratic_two_distinct_real_roots_l178_17826

theorem quadratic_two_distinct_real_roots (m : ℝ) (h : -4 * m > 0) : m = -1 :=
sorry

end quadratic_two_distinct_real_roots_l178_17826


namespace coffee_shop_cups_l178_17876

variables (A B X Y : ℕ) (Z : ℕ)

theorem coffee_shop_cups (h1 : Z = (A * B * X) + (A * (7 - B) * Y)) : 
  Z = (A * B * X) + (A * (7 - B) * Y) := 
by
  sorry

end coffee_shop_cups_l178_17876


namespace gena_hits_target_l178_17824

-- Definitions from the problem conditions
def initial_shots : ℕ := 5
def total_shots : ℕ := 17
def shots_per_hit : ℕ := 2

-- Mathematical equivalent proof statement
theorem gena_hits_target (G : ℕ) (H : G * shots_per_hit + initial_shots = total_shots) : G = 6 :=
by
  sorry

end gena_hits_target_l178_17824


namespace average_last_4_matches_l178_17829

theorem average_last_4_matches 
  (avg_10 : ℝ) (avg_6 : ℝ) (result : ℝ)
  (h1 : avg_10 = 38.9)
  (h2 : avg_6 = 42)
  (h3 : result = 34.25) :
  let total_runs_10 := avg_10 * 10
  let total_runs_6 := avg_6 * 6
  let total_runs_4 := total_runs_10 - total_runs_6
  let avg_4 := total_runs_4 / 4
  avg_4 = result :=
  sorry

end average_last_4_matches_l178_17829


namespace find_number_l178_17880

theorem find_number (x : ℝ) (h : (5 / 6) * x = (5 / 16) * x + 300) : x = 576 :=
sorry

end find_number_l178_17880


namespace number_identification_l178_17877

theorem number_identification (x : ℝ) (h : x ^ 655 / x ^ 650 = 100000) : x = 10 :=
by
  sorry

end number_identification_l178_17877


namespace necessary_but_not_sufficient_l178_17825

noncomputable def necessary_but_not_sufficient_condition (x : ℝ) : Prop :=
  -2 < x ∧ x < 3 → x^2 - 2 * x - 3 < 0

theorem necessary_but_not_sufficient (x : ℝ) : 
-2 < x ∧ x < 3 → x^2 - 2 * x - 3 < 0 := 
by
  sorry

end necessary_but_not_sufficient_l178_17825


namespace tan_of_acute_angle_l178_17807

theorem tan_of_acute_angle (α : ℝ) (h1 : α > 0 ∧ α < π / 2) (h2 : Real.cos (π / 2 + α) = -3/5) : Real.tan α = 3 / 4 :=
by
  sorry

end tan_of_acute_angle_l178_17807


namespace probability_sum_multiple_of_3_eq_one_third_probability_sum_prime_eq_five_twelfths_probability_second_greater_than_first_eq_five_twelfths_l178_17872

noncomputable def probability_sum_is_multiple_of_3 : ℝ :=
  let total_events := 36
  let favorable_events := 12
  favorable_events / total_events

noncomputable def probability_sum_is_prime : ℝ :=
  let total_events := 36
  let favorable_events := 15
  favorable_events / total_events

noncomputable def probability_second_greater_than_first : ℝ :=
  let total_events := 36
  let favorable_events := 15
  favorable_events / total_events

theorem probability_sum_multiple_of_3_eq_one_third :
  probability_sum_is_multiple_of_3 = 1 / 3 :=
by sorry

theorem probability_sum_prime_eq_five_twelfths :
  probability_sum_is_prime = 5 / 12 :=
by sorry

theorem probability_second_greater_than_first_eq_five_twelfths :
  probability_second_greater_than_first = 5 / 12 :=
by sorry

end probability_sum_multiple_of_3_eq_one_third_probability_sum_prime_eq_five_twelfths_probability_second_greater_than_first_eq_five_twelfths_l178_17872


namespace nonempty_solution_iff_a_gt_one_l178_17897

theorem nonempty_solution_iff_a_gt_one (a : ℝ) :
  (∃ x : ℝ, |x - 3| + |x - 4| < a) ↔ a > 1 :=
sorry

end nonempty_solution_iff_a_gt_one_l178_17897


namespace four_digit_perfect_square_exists_l178_17871

theorem four_digit_perfect_square_exists (x y : ℕ) (h1 : 10 ≤ x ∧ x < 100) (h2 : 10 ≤ y ∧ y < 100) (h3 : 101 * x + 100 = y^2) : 
  ∃ n, n = 8281 ∧ n = y^2 ∧ (((n / 100) : ℕ) = ((n % 100) : ℕ) + 1) :=
by 
  sorry

end four_digit_perfect_square_exists_l178_17871


namespace oranges_left_l178_17819

-- Main theorem statement: number of oranges left after specified increases and losses
theorem oranges_left (Mary Jason Tom Sarah : ℕ)
  (hMary : Mary = 122)
  (hJason : Jason = 105)
  (hTom : Tom = 85)
  (hSarah : Sarah = 134) 
  (round : ℝ → ℕ) 
  : round (round ( (Mary : ℝ) * 1.1) 
         + round ((Jason : ℝ) * 1.1) 
         + round ((Tom : ℝ) * 1.1) 
         + round ((Sarah : ℝ) * 1.1) 
         - round (0.15 * (round ((Mary : ℝ) * 1.1) 
                         + round ((Jason : ℝ) * 1.1)
                         + round ((Tom : ℝ) * 1.1) 
                         + round ((Sarah : ℝ) * 1.1)) )) = 417  := 
sorry

end oranges_left_l178_17819


namespace number_of_shirts_made_today_l178_17863

-- Define the rate of shirts made per minute.
def shirts_per_minute : ℕ := 6

-- Define the number of minutes the machine worked today.
def minutes_today : ℕ := 12

-- Define the total number of shirts made today.
def shirts_made_today : ℕ := shirts_per_minute * minutes_today

-- State the theorem for the number of shirts made today.
theorem number_of_shirts_made_today : shirts_made_today = 72 := 
by
  -- Proof is omitted
  sorry

end number_of_shirts_made_today_l178_17863


namespace minimum_rectangles_needed_l178_17881

/-- The theorem that defines the minimum number of rectangles needed to cover the specified figure -/
theorem minimum_rectangles_needed 
    (rectangles : ℕ) 
    (figure : Type)
    (covers : figure → Prop) :
  rectangles = 12 :=
sorry

end minimum_rectangles_needed_l178_17881


namespace geom_seq_general_formula_sum_first_n_terms_formula_l178_17801

namespace GeometricArithmeticSequences

def geom_seq_general (a_n : ℕ → ℝ) (n : ℕ) : Prop :=
  a_n 1 = 1 ∧ (2 * a_n 3 = a_n 2) → a_n n = 1 / (2 ^ (n - 1))

def sum_first_n_terms (a_n b_n : ℕ → ℝ) (S_n T_n : ℕ → ℝ) (n : ℕ) : Prop :=
  b_n 1 = 2 ∧ S_n 3 = b_n 2 + 6 → 
  T_n n = 6 - (n + 3) / (2 ^ (n - 1))

theorem geom_seq_general_formula :
  ∀ a_n : ℕ → ℝ, ∀ n : ℕ, geom_seq_general a_n n :=
by sorry

theorem sum_first_n_terms_formula :
  ∀ a_n b_n : ℕ → ℝ, ∀ S_n T_n : ℕ → ℝ, ∀ n : ℕ, sum_first_n_terms a_n b_n S_n T_n n :=
by sorry

end GeometricArithmeticSequences

end geom_seq_general_formula_sum_first_n_terms_formula_l178_17801


namespace jerrys_breakfast_calories_l178_17874

theorem jerrys_breakfast_calories 
    (num_pancakes : ℕ) (calories_per_pancake : ℕ) 
    (num_bacon : ℕ) (calories_per_bacon : ℕ) 
    (num_cereal : ℕ) (calories_per_cereal : ℕ) 
    (calories_total : ℕ) :
    num_pancakes = 6 →
    calories_per_pancake = 120 →
    num_bacon = 2 →
    calories_per_bacon = 100 →
    num_cereal = 1 →
    calories_per_cereal = 200 →
    calories_total = num_pancakes * calories_per_pancake
                   + num_bacon * calories_per_bacon
                   + num_cereal * calories_per_cereal →
    calories_total = 1120 :=
by
  intros h1 h2 h3 h4 h5 h6 h7
  rw [h1, h2, h3, h4, h5, h6] at h7
  assumption

end jerrys_breakfast_calories_l178_17874


namespace kia_vehicle_count_l178_17868

theorem kia_vehicle_count (total_vehicles : Nat) (dodge_vehicles : Nat) (hyundai_vehicles : Nat) 
    (h1 : total_vehicles = 400)
    (h2 : dodge_vehicles = total_vehicles / 2)
    (h3 : hyundai_vehicles = dodge_vehicles / 2) : 
    (total_vehicles - dodge_vehicles - hyundai_vehicles) = 100 := 
by sorry

end kia_vehicle_count_l178_17868


namespace geometric_sequence_a8_eq_pm1_l178_17844

variable {R : Type*} [LinearOrderedField R]

theorem geometric_sequence_a8_eq_pm1 :
  ∀ (a : ℕ → R), (∀ n : ℕ, ∃ r : R, r ≠ 0 ∧ a n = a 0 * r ^ n) → 
  (a 4 + a 12 = -3) ∧ (a 4 * a 12 = 1) → 
  (a 8 = 1 ∨ a 8 = -1) := by
  sorry

end geometric_sequence_a8_eq_pm1_l178_17844


namespace number_of_spinsters_l178_17811

-- Given conditions
variables (S C : ℕ)
axiom ratio_condition : S / C = 2 / 9
axiom difference_condition : C = S + 63

-- Theorem to prove
theorem number_of_spinsters : S = 18 :=
sorry

end number_of_spinsters_l178_17811


namespace find_ab_l178_17830

variables (a b c : ℝ)

-- Defining the conditions
def cond1 : Prop := a - b = 5
def cond2 : Prop := a^2 + b^2 = 34
def cond3 : Prop := a^3 - b^3 = 30
def cond4 : Prop := a^2 + b^2 - c^2 = 50

theorem find_ab (h1 : cond1 a b) (h2 : cond2 a b) (h3 : cond3 a b) (h4 : cond4 a b c) :
  a * b = 4.5 :=
sorry

end find_ab_l178_17830


namespace obtuse_scalene_triangle_l178_17882

theorem obtuse_scalene_triangle {k : ℕ} (h1 : 13 < k + 17) (h2 : 17 < 13 + k)
  (h3 : 13 < k + 17) (h4 : k ≠ 13) (h5 : k ≠ 17) 
  (h6 : 17^2 > 13^2 + k^2 ∨ k^2 > 13^2 + 17^2) 
  (h7 : (k = 5 ∨ k = 6 ∨ k = 7 ∨ k = 8 ∨ k = 9 ∨ k = 10 ∨ k = 22 ∨ 
        k = 23 ∨ k = 24 ∨ k = 25 ∨ k = 26 ∨ k = 27 ∨ k = 28 ∨ k = 29)) :
  ∃ n, n = 14 := 
by
  sorry

end obtuse_scalene_triangle_l178_17882


namespace three_digit_number_constraint_l178_17888

theorem three_digit_number_constraint (B : ℕ) (h1 : 30 ≤ B ∧ B < 40) (h2 : (330 + B) % 3 = 0) (h3 : (330 + B) % 7 = 0) : B = 6 :=
sorry

end three_digit_number_constraint_l178_17888


namespace count_divisible_by_35_l178_17840

theorem count_divisible_by_35 : 
  ∃! (n : ℕ), n = 13 ∧ (∀ (ab : ℕ), 10 ≤ ab ∧ ab ≤ 99 ∧ (∃ (a b : ℕ), a ≥ 1 ∧ a ≤ 9 ∧ b ≤ 9 ∧ ab = 10 * a + b) →
    (ab * 100 + 35) % 35 = 0 ↔ ab % 7 = 0) :=
by {
  sorry
}

end count_divisible_by_35_l178_17840


namespace units_digit_of_fraction_l178_17810

-- Define the problem
def units_digit (n : ℕ) : ℕ := n % 10

theorem units_digit_of_fraction :
  units_digit ((30 * 31 * 32 * 33 * 34 * 35) / 2500) = 2 := by
  sorry

end units_digit_of_fraction_l178_17810


namespace tank_full_capacity_l178_17886

theorem tank_full_capacity (C : ℝ) (H1 : 0.4 * C + 36 = 0.7 * C) : C = 120 :=
by
  sorry

end tank_full_capacity_l178_17886


namespace trapezoid_base_length_l178_17803

-- Definitions from the conditions
def trapezoid_area (a b h : ℕ) : ℕ := (1 / 2) * (a + b) * h

theorem trapezoid_base_length (b : ℕ) (h : ℕ) (a : ℕ) (A : ℕ) (H_area : A = 222) (H_upper_side : a = 23) (H_height : h = 12) :
  A = trapezoid_area a b h ↔ b = 14 :=
by sorry

end trapezoid_base_length_l178_17803


namespace product_of_nine_integers_16_to_30_equals_15_factorial_l178_17884

noncomputable def factorial (n : Nat) : Nat :=
match n with
| 0     => 1
| (n+1) => (n+1) * factorial n

theorem product_of_nine_integers_16_to_30_equals_15_factorial :
  (16 * 18 * 20 * 21 * 22 * 25 * 26 * 27 * 28) = factorial 15 := 
by sorry

end product_of_nine_integers_16_to_30_equals_15_factorial_l178_17884


namespace min_length_M_intersect_N_l178_17898

-- Define the sets M and N with the given conditions
def M (m : ℝ) : Set ℝ := {x | m ≤ x ∧ x ≤ m + 2/3}
def N (n : ℝ) : Set ℝ := {x | n - 3/4 ≤ x ∧ x ≤ n}
def M_intersect_N (m n : ℝ) : Set ℝ := M m ∩ N n

-- Define the condition that M and N are subsets of [0, 1]
def in_interval (m n : ℝ) := (M m ⊆ {x | 0 ≤ x ∧ x ≤ 1}) ∧ (N n ⊆ {x | 0 ≤ x ∧ x ≤ 1})

-- Define the length of a set given by an interval [a, b]
def length_interval (a b : ℝ) := b - a

-- Define the length of the intersection of M and N
noncomputable def length_M_intersect_N (m n : ℝ) : ℝ :=
  let a := max m (n - 3/4)
  let b := min (m + 2/3) n
  length_interval a b

-- Prove that the minimum length of M ∩ N is 5/12
theorem min_length_M_intersect_N (m n : ℝ) (h : in_interval m n) : length_M_intersect_N m n = 5 / 12 :=
by
  sorry

end min_length_M_intersect_N_l178_17898


namespace range_of_2_cos_sq_l178_17809

theorem range_of_2_cos_sq :
  ∀ x : ℝ, 0 ≤ 2 * (Real.cos x) ^ 2 ∧ 2 * (Real.cos x) ^ 2 ≤ 2 :=
by sorry

end range_of_2_cos_sq_l178_17809


namespace g_at_8_l178_17849

def g (x : ℝ) : ℝ := sorry

axiom g_property : ∀ x y : ℝ, x * g y = y * g x

axiom g_at_24 : g 24 = 12

theorem g_at_8 : g 8 = 4 := by
  sorry

end g_at_8_l178_17849


namespace tan_A_tan_B_l178_17828

theorem tan_A_tan_B (A B C : ℝ) (R : ℝ) (H F : ℝ)
  (HF : H + F = 26) (h1 : 2 * R * Real.cos A * Real.cos B = 8)
  (h2 : 2 * R * Real.sin A * Real.sin B = 26) :
  Real.tan A * Real.tan B = 13 / 4 :=
by
  sorry

end tan_A_tan_B_l178_17828


namespace cube_volume_l178_17862

theorem cube_volume (A : ℝ) (hA : A = 294) : ∃ (V : ℝ), V = 343 :=
by
  sorry

end cube_volume_l178_17862


namespace ordered_pairs_count_l178_17838

theorem ordered_pairs_count :
  (∃ (a b : ℝ), (∃ (x y : ℤ),
    a * (x : ℝ) + b * (y : ℝ) = 1 ∧
    (x : ℝ)^2 + (y : ℝ)^2 = 65)) →
  ∃ (n : ℕ), n = 128 :=
by
  sorry

end ordered_pairs_count_l178_17838


namespace correct_statements_are_C_and_D_l178_17883

theorem correct_statements_are_C_and_D
  (a b c m : ℝ)
  (ha1 : -1 < a) (ha2 : a < 5)
  (hb1 : -2 < b) (hb2 : b < 3)
  (hab : a > b)
  (h_ac2bc2 : a * c^2 > b * c^2) (hc2_pos : c^2 > 0)
  (h_ab_pos : a > b) (h_b_pos : b > 0) (hm_pos : m > 0) :
  (¬(1 < a - b ∧ a - b < 2)) ∧ (¬(a^2 > b^2)) ∧ (a > b) ∧ ((b + m) / (a + m) > b / a) :=
by sorry

end correct_statements_are_C_and_D_l178_17883


namespace initial_sentences_today_l178_17845

-- Definitions of the given conditions
def typing_rate : ℕ := 6
def initial_typing_time : ℕ := 20
def additional_typing_time : ℕ := 15
def erased_sentences : ℕ := 40
def post_meeting_typing_time : ℕ := 18
def total_sentences_end_of_day : ℕ := 536

def sentences_typed_before_break := initial_typing_time * typing_rate
def sentences_typed_after_break := additional_typing_time * typing_rate
def sentences_typed_post_meeting := post_meeting_typing_time * typing_rate
def sentences_today := sentences_typed_before_break + sentences_typed_after_break - erased_sentences + sentences_typed_post_meeting

theorem initial_sentences_today : total_sentences_end_of_day - sentences_today = 258 := by
  -- proof here
  sorry

end initial_sentences_today_l178_17845


namespace degrees_for_lemon_pie_l178_17831

theorem degrees_for_lemon_pie 
    (total_students : ℕ)
    (chocolate_lovers : ℕ)
    (apple_lovers : ℕ)
    (blueberry_lovers : ℕ)
    (remaining_students : ℕ)
    (lemon_pie_degrees : ℝ) :
    total_students = 42 →
    chocolate_lovers = 15 →
    apple_lovers = 9 →
    blueberry_lovers = 7 →
    remaining_students = total_students - (chocolate_lovers + apple_lovers + blueberry_lovers) →
    lemon_pie_degrees = (remaining_students / 2 / total_students * 360) →
    lemon_pie_degrees = 47.14 :=
by
  intros _ _ _ _ _ _
  sorry

end degrees_for_lemon_pie_l178_17831


namespace beth_total_crayons_l178_17802

theorem beth_total_crayons :
  let packs := 4
  let crayons_per_pack := 10
  let extra_crayons := 6
  packs * crayons_per_pack + extra_crayons = 46 :=
by
  let packs := 4
  let crayons_per_pack := 10
  let extra_crayons := 6
  show packs * crayons_per_pack + extra_crayons = 46
  sorry

end beth_total_crayons_l178_17802


namespace geometric_sequence_common_ratio_l178_17853

theorem geometric_sequence_common_ratio (a : ℕ → ℝ) (q : ℝ) 
  (h1 : a 1 = -1) 
  (h2 : a 2 + a 3 = -2) 
  (h_geom : ∀ n, a (n + 1) = a n * q) : 
  q = -2 ∨ q = 1 := 
by sorry

end geometric_sequence_common_ratio_l178_17853


namespace possible_measures_of_angle_X_l178_17835

theorem possible_measures_of_angle_X : 
  ∃ n : ℕ, n = 17 ∧ (∀ (X Y : ℕ), 
    X > 0 ∧ Y > 0 ∧ X + Y = 180 ∧ 
    ∃ m : ℕ, m ≥ 1 ∧ X = m * Y) :=
sorry

end possible_measures_of_angle_X_l178_17835


namespace sphere_pyramid_problem_l178_17879

theorem sphere_pyramid_problem (n m : ℕ) :
  (n * (n + 1) * (2 * n + 1)) / 6 + (m * (m + 1) * (m + 2)) / 6 = 605 → n = 10 ∧ m = 10 :=
by
  sorry

end sphere_pyramid_problem_l178_17879


namespace correct_quotient_l178_17896

variable (D : ℕ) (q1 q2 : ℕ)
variable (h1 : q1 = 4900) (h2 : D - 1000 = 1200 * q1)

theorem correct_quotient : q2 = D / 2100 → q2 = 2800 :=
by
  sorry

end correct_quotient_l178_17896


namespace Alyssa_puppies_l178_17856

theorem Alyssa_puppies (initial_puppies give_away_puppies : ℕ) (h_initial : initial_puppies = 12) (h_give_away : give_away_puppies = 7) :
  initial_puppies - give_away_puppies = 5 :=
by
  sorry

end Alyssa_puppies_l178_17856


namespace find_two_digit_numbers_l178_17890

def first_two_digit_number (x y : ℕ) : ℕ := 10 * x + y
def second_two_digit_number (x y : ℕ) : ℕ := 10 * (x + 5) + y

theorem find_two_digit_numbers :
  ∃ (x_2 y : ℕ), 
  (first_two_digit_number x_2 y = x_2^2 + x_2 * y + y^2) ∧ 
  (second_two_digit_number x_2 y = (x_2 + 5)^2 + (x_2 + 5) * y + y^2) ∧ 
  (second_two_digit_number x_2 y - first_two_digit_number x_2 y = 50) ∧ 
  (y = 1 ∨ y = 3) := 
sorry

end find_two_digit_numbers_l178_17890


namespace problem_part1_problem_part2_problem_part3_l178_17895

noncomputable def f (x : ℝ) : ℝ := x * Real.log x
noncomputable def g (a x : ℝ) : ℝ := a * x^2 - x

theorem problem_part1 :
  (∀ x, 0 < x -> x < 1 / Real.exp 1 -> f (Real.log x + 1) < 0) ∧ 
  (∀ x, x > 1 / Real.exp 1 -> f (Real.log x + 1) > 0) ∧ 
  (f (1 / Real.exp 1) = 1 / Real.exp 1 * Real.log (1 / Real.exp 1)) :=
sorry

theorem problem_part2 (a : ℝ) :
  (∀ x, x > 0 -> f x ≤ g a x) ↔ a ≥ 1 :=
sorry

theorem problem_part3 (a : ℝ) (m : ℝ) (ha : a = 1/8) :
  (∃ m, (∃ x₁ x₂ x₃ : ℝ, x₁ ≠ x₂ ∧ x₂ ≠ x₃ ∧ x₁ ≠ x₃ ∧ (3 * f x / (4 * x) + m + g a x = 0))) ↔ 
  (7/8 < m ∧ m < (15/8 - 3/4 * Real.log 3)) :=
sorry

end problem_part1_problem_part2_problem_part3_l178_17895


namespace cinnamon_balls_required_l178_17870

theorem cinnamon_balls_required 
  (num_family_members : ℕ) 
  (cinnamon_balls_per_day : ℕ) 
  (num_days : ℕ) 
  (h_family : num_family_members = 5) 
  (h_balls_per_day : cinnamon_balls_per_day = 5) 
  (h_days : num_days = 10) : 
  num_family_members * cinnamon_balls_per_day * num_days = 50 := by
  sorry

end cinnamon_balls_required_l178_17870


namespace neg_p_l178_17821

-- Define the initial proposition p
def p : Prop := ∀ (m : ℝ), m ≥ 0 → 4^m ≥ 4 * m

-- State the theorem to prove the negation of p
theorem neg_p : ¬p ↔ ∃ (m_0 : ℝ), m_0 ≥ 0 ∧ 4^m_0 < 4 * m_0 :=
by
  sorry

end neg_p_l178_17821


namespace platform_length_259_9584_l178_17843

noncomputable def length_of_platform (speed_kmph time_sec train_length_m : ℝ) : ℝ :=
  let speed_mps := speed_kmph * 1000 / 3600  -- conversion from kmph to m/s
  let distance_covered := speed_mps * time_sec
  distance_covered - train_length_m

theorem platform_length_259_9584 :
  length_of_platform 72 26 260.0416 = 259.9584 :=
by sorry

end platform_length_259_9584_l178_17843


namespace travel_time_home_to_community_center_l178_17818

-- Definitions and assumptions based on the conditions
def time_to_library := 30 -- in minutes
def distance_to_library := 5 -- in miles
def time_spent_at_library := 15 -- in minutes
def distance_to_community_center := 3 -- in miles
noncomputable def cycling_speed := time_to_library / distance_to_library -- in minutes per mile

-- Time calculation to reach the community center from the library
noncomputable def time_from_library_to_community_center := distance_to_community_center * cycling_speed -- in minutes

-- Total time spent to travel from home to the community center
noncomputable def total_time_home_to_community_center :=
  time_to_library + time_spent_at_library + time_from_library_to_community_center

-- The proof statement verifying the total time
theorem travel_time_home_to_community_center : total_time_home_to_community_center = 63 := by
  sorry

end travel_time_home_to_community_center_l178_17818


namespace total_cost_is_67_15_l178_17841

noncomputable def calculate_total_cost : ℝ :=
  let caramel_cost := 3
  let candy_bar_cost := 2 * caramel_cost
  let cotton_candy_cost := (candy_bar_cost * 4) / 2
  let chocolate_bar_cost := candy_bar_cost + caramel_cost
  let lollipop_cost := candy_bar_cost / 3

  let candy_bar_total := 6 * candy_bar_cost
  let caramel_total := 3 * caramel_cost
  let cotton_candy_total := 1 * cotton_candy_cost
  let chocolate_bar_total := 2 * chocolate_bar_cost
  let lollipop_total := 2 * lollipop_cost

  let discounted_candy_bar_total := candy_bar_total * 0.9
  let discounted_caramel_total := caramel_total * 0.85
  let discounted_cotton_candy_total := cotton_candy_total * 0.8
  let discounted_chocolate_bar_total := chocolate_bar_total * 0.75
  let discounted_lollipop_total := lollipop_total -- No additional discount

  discounted_candy_bar_total +
  discounted_caramel_total +
  discounted_cotton_candy_total +
  discounted_chocolate_bar_total +
  discounted_lollipop_total

theorem total_cost_is_67_15 : calculate_total_cost = 67.15 := by
  sorry

end total_cost_is_67_15_l178_17841


namespace union_M_N_l178_17813

def M : Set ℝ := { x | -3 < x ∧ x ≤ 5 }
def N : Set ℝ := { x | x > 3 }

theorem union_M_N : M ∪ N = { x | x > -3 } :=
by
  sorry

end union_M_N_l178_17813


namespace saleswoman_commission_l178_17800

theorem saleswoman_commission (x : ℝ) (h1 : ∀ sale : ℝ, sale = 800) (h2 : (x / 100) * 500 + 0.25 * (800 - 500) = 0.21875 * 800) : x = 20 := by
  sorry

end saleswoman_commission_l178_17800


namespace particle_speed_interval_l178_17851

noncomputable def particle_position (t : ℝ) : ℝ × ℝ :=
  (3 * t + 5, 5 * t - 7)

theorem particle_speed_interval (k : ℝ) :
  let start_pos := particle_position k
  let end_pos := particle_position (k + 2)
  let delta_x := end_pos.1 - start_pos.1
  let delta_y := end_pos.2 - start_pos.2
  let speed := Real.sqrt (delta_x^2 + delta_y^2)
  speed = 2 * Real.sqrt 34 := by
  sorry

end particle_speed_interval_l178_17851


namespace other_train_length_l178_17823

-- Define a theorem to prove that the length of the other train (L) is 413.95 meters
theorem other_train_length (length_first_train : ℝ) (speed_first_train_kmph : ℝ) 
                           (speed_second_train_kmph: ℝ) (time_crossing_seconds : ℝ) : 
                           length_first_train = 350 → 
                           speed_first_train_kmph = 150 →
                           speed_second_train_kmph = 100 →
                           time_crossing_seconds = 11 →
                           ∃ (L : ℝ), L = 413.95 :=
by
  intros h1 h2 h3 h4
  sorry

end other_train_length_l178_17823


namespace fraction_value_l178_17889

theorem fraction_value (a b : ℝ) (h₁ : a ≠ 0) (h₂ : b ≠ 0) (h₃ : (4 * a + b) / (a - 4 * b) = 3) :
  (a + 4 * b) / (4 * a - b) = 9 / 53 :=
by
  sorry

end fraction_value_l178_17889


namespace converse_of_P_inverse_of_P_contrapositive_of_P_negation_of_P_l178_17887

theorem converse_of_P (a b : ℤ) : (a - 2 > b - 2) → (a > b) :=
by
  intro h
  exact sorry

theorem inverse_of_P (a b : ℤ) : (a ≤ b) → (a - 2 ≤ b - 2) :=
by
  intro h
  exact sorry

theorem contrapositive_of_P (a b : ℤ) : (a - 2 ≤ b - 2) → (a ≤ b) :=
by
  intro h
  exact sorry

theorem negation_of_P (a b : ℤ) : (a > b) → ¬ (a - 2 ≤ b - 2) :=
by
  intro h
  exact sorry

end converse_of_P_inverse_of_P_contrapositive_of_P_negation_of_P_l178_17887


namespace max_probability_first_black_ace_l178_17804

def probability_first_black_ace(k : ℕ) : ℚ :=
  if 1 ≤ k ∧ k ≤ 51 then (52 - k) / 1326 else 0

theorem max_probability_first_black_ace : 
  ∀ k : ℕ, 1 ≤ k ∧ k ≤ 51 → probability_first_black_ace k ≤ probability_first_black_ace 1 :=
by
  sorry

end max_probability_first_black_ace_l178_17804


namespace jogging_walking_ratio_l178_17865

theorem jogging_walking_ratio (total_time walk_time jog_time: ℕ) (h1 : total_time = 21) (h2 : walk_time = 9) (h3 : jog_time = total_time - walk_time) :
  (jog_time : ℚ) / walk_time = 4 / 3 :=
by
  sorry

end jogging_walking_ratio_l178_17865


namespace milly_folds_count_l178_17806

theorem milly_folds_count (mixing_time baking_time total_minutes fold_time rest_time : ℕ) 
  (h : total_minutes = 360)
  (h_mixing_time : mixing_time = 10)
  (h_baking_time : baking_time = 30)
  (h_fold_time : fold_time = 5)
  (h_rest_time : rest_time = 75) : 
  (total_minutes - (mixing_time + baking_time)) / (fold_time + rest_time) = 4 := 
by
  sorry

end milly_folds_count_l178_17806


namespace calories_per_cookie_l178_17846

theorem calories_per_cookie (C : ℝ) (h1 : ∀ cracker, cracker = 15)
    (h2 : ∀ cookie, cookie = C)
    (h3 : 7 * C + 10 * 15 = 500) :
    C = 50 :=
  by
    sorry

end calories_per_cookie_l178_17846


namespace set_subtraction_M_N_l178_17867

-- Definitions
def A : Set ℝ := { x | ∃ y, y = Real.sqrt (1 - x) }
def B : Set ℝ := { y | ∃ x, y = x^2 ∧ -1 ≤ x ∧ x ≤ 1 }
def M : Set ℝ := { x | ∃ y, y = Real.sqrt (1 - x) }
def N : Set ℝ := { y | ∃ x, y = x^2 ∧ -1 ≤ x ∧ x ≤ 1 }

-- Statement
theorem set_subtraction_M_N : (M \ N) = { x | x < 0 } := by
  sorry

end set_subtraction_M_N_l178_17867


namespace pencils_in_drawer_l178_17893

theorem pencils_in_drawer (P : ℕ) (h1 : P + 19 + 16 = 78) : P = 43 :=
by
  sorry

end pencils_in_drawer_l178_17893


namespace tan_alpha_through_point_l178_17864

theorem tan_alpha_through_point (α : ℝ) (x y : ℝ) (h : (x, y) = (3, 4)) : Real.tan α = 4 / 3 :=
sorry

end tan_alpha_through_point_l178_17864


namespace tens_digit_of_9_pow_1024_l178_17885

theorem tens_digit_of_9_pow_1024 : 
  (9^1024 % 100) / 10 % 10 = 6 := 
sorry

end tens_digit_of_9_pow_1024_l178_17885


namespace max_n_for_factored_polynomial_l178_17852

theorem max_n_for_factored_polynomial :
  ∃ (n : ℤ), (∀ (A B : ℤ), 6 * 72 + n = 6 * B + A ∧ A * B = 72 → n ≤ 433) ∧
             (∃ (A B : ℤ), 6 * B + A = 433 ∧ A * B = 72) :=
by sorry

end max_n_for_factored_polynomial_l178_17852


namespace hexagon_colorings_correct_l178_17860

def valid_hexagon_colorings : Prop :=
  ∃ (colors : Fin 6 → Fin 7),
    (colors 0 ≠ colors 1) ∧
    (colors 1 ≠ colors 2) ∧
    (colors 2 ≠ colors 3) ∧
    (colors 3 ≠ colors 4) ∧
    (colors 4 ≠ colors 5) ∧
    (colors 5 ≠ colors 0) ∧
    (colors 0 ≠ colors 2) ∧
    (colors 1 ≠ colors 3) ∧
    (colors 2 ≠ colors 4) ∧
    (colors 3 ≠ colors 5) ∧
    ∃! (n : Nat), n = 12600

theorem hexagon_colorings_correct : valid_hexagon_colorings :=
sorry

end hexagon_colorings_correct_l178_17860


namespace negation_P1_is_false_negation_P2_is_false_l178_17859

-- Define the propositions
def isMultiDigitNumber (n : ℕ) : Prop := n >= 10
def lastDigitIsZero (n : ℕ) : Prop := n % 10 = 0
def isMultipleOfFive (n : ℕ) : Prop := n % 5 = 0
def isEven (n : ℕ) : Prop := n % 2 = 0

-- The propositions
def P1 (n : ℕ) : Prop := isMultiDigitNumber n → (lastDigitIsZero n → isMultipleOfFive n)
def P2 : Prop := ∀ n, isEven n → n % 2 = 0

-- The negations
def notP1 (n : ℕ) : Prop := isMultiDigitNumber n ∧ lastDigitIsZero n → ¬isMultipleOfFive n
def notP2 : Prop := ∃ n, isEven n ∧ ¬(n % 2 = 0)

-- The proof problems
theorem negation_P1_is_false (n : ℕ) : notP1 n → False := by
  sorry

theorem negation_P2_is_false : notP2 → False := by
  sorry

end negation_P1_is_false_negation_P2_is_false_l178_17859


namespace unique_solution_mod_37_system_l178_17894

theorem unique_solution_mod_37_system :
  ∃! (a b c d : ℤ), 
  (a^2 + b * c ≡ a [ZMOD 37]) ∧
  (b * (a + d) ≡ b [ZMOD 37]) ∧
  (c * (a + d) ≡ c [ZMOD 37]) ∧
  (b * c + d^2 ≡ d [ZMOD 37]) ∧
  (a * d - b * c ≡ 1 [ZMOD 37]) :=
sorry

end unique_solution_mod_37_system_l178_17894


namespace area_of_triangle_F1PF2_l178_17875

noncomputable def point_on_ellipse (P : ℝ × ℝ) : Prop :=
  let x := P.1
  let y := P.2
  (x^2 / 25) + (y^2 / 16) = 1

def is_focus (f : ℝ × ℝ) : Prop := 
  f = (3, 0) ∨ f = (-3, 0)

def right_angle_at_P (F1 P F2 : ℝ × ℝ) : Prop := 
  let a1 := (F1.1 - P.1, F1.2 - P.2)
  let a2 := (F2.1 - P.1, F2.2 - P.2)
  a1.1 * a2.1 + a1.2 * a2.2 = 0

theorem area_of_triangle_F1PF2
  (P F1 F2 : ℝ × ℝ)
  (hP : point_on_ellipse P)
  (hF1 : is_focus F1)
  (hF2 : is_focus F2)
  (h_angle : right_angle_at_P F1 P F2) :
  1/2 * (P.1 - F1.1) * (P.2 - F2.2) = 16 :=
sorry

end area_of_triangle_F1PF2_l178_17875


namespace commutating_matrices_l178_17861

noncomputable def A : Matrix (Fin 2) (Fin 2) ℝ :=  ![![2, 3], ![4, 5]]
noncomputable def B (x y z w : ℝ) : Matrix (Fin 2) (Fin 2) ℝ := ![![x, y], ![z, w]]

theorem commutating_matrices (x y z w : ℝ) (h1 : A * (B x y z w) = (B x y z w) * A) (h2 : 4 * y ≠ z) : 
  (x - w) / (z - 4 * y) = 1 / 2 := 
by
  sorry

end commutating_matrices_l178_17861


namespace original_number_doubled_added_trebled_l178_17833

theorem original_number_doubled_added_trebled (x : ℤ) : 3 * (2 * x + 9) = 75 → x = 8 :=
by
  intro h
  -- The proof is omitted as instructed.
  sorry

end original_number_doubled_added_trebled_l178_17833


namespace curves_intersection_four_points_l178_17858

theorem curves_intersection_four_points (b : ℝ) :
  (∃ x1 x2 x3 x4 y1 y2 y3 y4 : ℝ,
    x1^2 + y1^2 = b^2 ∧ y1 = x1^2 - b + 1 ∧
    x2^2 + y2^2 = b^2 ∧ y2 = x2^2 - b + 1 ∧
    x3^2 + y3^2 = b^2 ∧ y3 = x3^2 - b + 1 ∧
    x4^2 + y4^2 = b^2 ∧ y4 = x4^2 - b + 1 ∧
    (x1, y1) ≠ (x2, y2) ∧ (x1, y1) ≠ (x3, y3) ∧ (x1, y1) ≠ (x4, y4) ∧
    (x2, y2) ≠ (x3, y3) ∧ (x2, y2) ≠ (x4, y4) ∧
    (x3, y3) ≠ (x4, y4)) →
  b > 2 :=
sorry

end curves_intersection_four_points_l178_17858


namespace kim_distance_traveled_l178_17869

-- Definitions based on the problem conditions:
def infantry_column_length : ℝ := 1  -- The length of the infantry column in km.
def distance_inf_covered : ℝ := 2.4  -- Distance the infantrymen covered in km.

-- Theorem statement:
theorem kim_distance_traveled (column_length : ℝ) (inf_covered : ℝ) :
  column_length = 1 →
  inf_covered = 2.4 →
  ∃ d : ℝ, d = 3.6 :=
by
  sorry

end kim_distance_traveled_l178_17869


namespace fifteen_percent_of_x_is_ninety_l178_17815

theorem fifteen_percent_of_x_is_ninety (x : ℝ) (h : (15 / 100) * x = 90) : x = 600 :=
sorry

end fifteen_percent_of_x_is_ninety_l178_17815
